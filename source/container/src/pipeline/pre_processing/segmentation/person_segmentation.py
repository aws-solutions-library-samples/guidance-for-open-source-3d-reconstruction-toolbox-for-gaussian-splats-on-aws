# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY

"""Person segmentation for object (human) removal.

Grounding DINO (IDEA-Research/grounding-dino-base) finds a box around every person
("person." prompt) and SAM 2.1 (facebook/sam2.1-hiera-large) segments each box.
Both are Apache-2.0 (code and weights), loaded with Hugging Face transformers, and
are the same models sky_mask.py uses, so they are downloaded once per job.

This replaces the rembg u2net_human_seg + birefnet-portrait pair. Those are
salient-object / portrait-matting models: they segment whatever stands out (often
not a person), miss small or distant people, and BiRefNet runs at 1024x1024 on
ONNX Runtime (usually CPU in this image), so it took seconds per image and was
downloaded (~1 GB) on every job. Here images without a detected person cost one
Grounding DINO pass, and people are segmented as whole instances.

Output: RGBA PNG per image, <stem>.png, RGB = the image, alpha 255 = person
(same format as before, read by remove_object_using_mask.py,
erase_object_using_mask.py, remove_object_rig.py and panorama_sfm.py).
"""

import os
import time

import cv2
import numpy as np

DINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM2_MODEL = "facebook/sam2.1-hiera-large"
SAM1_MODEL = "facebook/sam-vit-huge"
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')


class PersonSegmenter:
    def __init__(self, device=None, box_threshold=0.3, text_threshold=0.25, prompt="person."):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold, self.text_threshold, self.prompt = box_threshold, text_threshold, prompt
        self.dino_proc = AutoProcessor.from_pretrained(DINO_MODEL)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL).to(self.device).eval()
        try:
            from transformers import Sam2Model, Sam2Processor
            self.sam_proc = Sam2Processor.from_pretrained(SAM2_MODEL)
            self.sam = Sam2Model.from_pretrained(SAM2_MODEL).to(self.device).eval()
            self.sam_name = SAM2_MODEL
        except ImportError:
            from transformers import SamModel, SamProcessor
            self.sam_proc = SamProcessor.from_pretrained(SAM1_MODEL)
            self.sam = SamModel.from_pretrained(SAM1_MODEL).to(self.device).eval()
            self.sam_name = SAM1_MODEL
        # bfloat16 autocast on Ampere+ GPUs (A10G, L4, L40S): ~2-3x faster SAM 2.1 and
        # ~1.5x faster Grounding DINO, same masks within a few edge pixels.
        self.amp = (str(self.device).startswith("cuda") and torch.cuda.is_bf16_supported()
                    and os.environ.get("PERSON_SEG_FP32", "") != "1")
        print(f"[Person] Models: {DINO_MODEL} + {self.sam_name} on {self.device}"
              f"{' (bfloat16 autocast)' if self.amp else ''}")

    def _ctx(self):
        torch = self.torch
        if self.amp:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def boxes(self, pil_image):
        torch = self.torch
        W, H = pil_image.size
        inp = self.dino_proc(images=pil_image, text=self.prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode(), self._ctx():
            out = self.dino(**inp)
        out.logits = out.logits.float()
        out.pred_boxes = out.pred_boxes.float()
        det = self.dino_proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[(H, W)])[0]
        return [[float(v) for v in b] for b in det["boxes"].tolist()]

    def __call__(self, pil_image):
        """Returns (uint8 HxW mask, 255 = person; number of people found)."""
        torch = self.torch
        W, H = pil_image.size
        boxes = self.boxes(pil_image)
        if not boxes:
            return np.zeros((H, W), np.uint8), 0
        sin = self.sam_proc(images=pil_image, input_boxes=[boxes], return_tensors="pt").to(self.device)
        with torch.inference_mode(), self._ctx():
            sout = self.sam(**sin, multimask_output=False)
        # Union of the people at SAM's low mask resolution, then one upsample to the
        # image size (upsampling every person first needs N x H x W floats).
        logits = sout.pred_masks.float().amax(dim=1, keepdim=True).cpu()
        mask = self.sam_proc.post_process_masks(logits, sin["original_sizes"].cpu(), binarize=False)[0]
        mask = (mask.reshape(-1, H, W)[0] > 0).numpy().astype(np.uint8) * 255
        return mask, len(boxes)


def grow_mask(mask, fraction=0.004, min_px=3):
    """Dilate a person mask by ~0.4% of the short image side so edges, hair and
    the contact shadow line are covered."""
    r = max(min_px, int(round(min(mask.shape[:2]) * fraction)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask, kernel)


def process_dir(input_dir, output_dir, device=None, box_threshold=0.3, text_threshold=0.25):
    """Writes <stem>.png (RGBA, alpha 255 = person) for every image in input_dir.
    Returns the number of images written."""
    from PIL import Image
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(IMAGE_EXTS))
    seg = PersonSegmenter(device, box_threshold, text_threshold)
    t0, n_with, n_people, written = time.time(), 0, 0, 0
    for k, name in enumerate(files, 1):
        path = os.path.join(input_dir, name)
        try:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"[Person] Could not read {name}; skipping")
                continue
            mask, n = seg(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            if n:
                mask = grow_mask(mask)
                n_with += 1
                n_people += n
            rgba = np.dstack([image, mask])
            cv2.imwrite(os.path.join(output_dir, os.path.splitext(name)[0] + ".png"), rgba,
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])
            written += 1
        except Exception as e:  # keep going: one bad image must not stop the job
            print(f"[Person] Error processing {name}: {e}")
        if k % 25 == 0 or k == len(files):
            dt = time.time() - t0
            print(f"[Person] {k}/{len(files)} images, {n_with} with people ({n_people} detections), "
                  f"{dt / k:.2f} s/image")
    return written
