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

"""Add the removed/erased people to the training masks (masks/, white = keep).

Object removal blacks out ("remove") or inpaints ("erase") people in the images.
Both leave pixels that are not real scene content: black silhouettes, or inpainted
fill that differs from view to view. Masking them means COLMAP extracts no
features there and the splat trainers give those pixels no loss, so the other
views fill the area in.

Person masks are read from --person-dir as <relative image stem>.png (an RGBA
remove_background output with alpha = person, or a grayscale/binary mask with
white = person), for example person-dir/cam0/frame_001.png for
images/cam0/frame_001.jpg. Every image gets a keep-mask written with mask_io
(masks/<image name>.png and masks/<image stem>.png, AND-ed with any mask already
there), including images without people (all white), because nerfstudio and
Faster-GS need a mask for every image once masks are used.
"""

import argparse
import os
import sys

import cv2
import numpy as np

# mask_io lives one folder up (pre_processing/), in the repo and in the container.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from mask_io import list_images, save_preview, write_keep_mask  # noqa: E402

SKIP_DIRS = ('refined_masks',)


def read_person(person_dir, rel_image, shape):
    """Bool person mask (image size) or None when there is no mask file."""
    stem = os.path.splitext(rel_image)[0]
    for name in (stem + ".png", rel_image + ".png"):
        p = os.path.join(person_dir, name)
        if not os.path.isfile(p):
            continue
        m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3 and m.shape[2] == 4:
            m = m[:, :, 3]
        elif m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        if m.shape != shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return m > 10
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--person-dir", required=True, help="Person masks, <relative image stem>.png")
    ap.add_argument("--masks-dir", required=True, help="Keep-masks to write (white = keep)")
    ap.add_argument("--grow-fraction", type=float, default=0.01,
                    help="Grow each person mask by this fraction of the short image side (covers the "
                         "eraser's blended border and the person's edge)")
    ap.add_argument("--debug-dir", default="", help="Overlay previews of the masked people")
    args = ap.parse_args()

    images = [r for r in list_images(args.images_dir)
              if not any(part in SKIP_DIRS for part in r.split("/")[:-1])]
    if not os.path.isdir(args.person_dir):
        print(f"[PersonMasks] {args.person_dir} does not exist; no person masks written")
        return
    n_people, masked_px, total_px = 0, 0, 0
    for rel in images:
        image = cv2.imread(os.path.join(args.images_dir, rel), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[PersonMasks] Could not read {rel}; skipping")
            continue
        shape = image.shape[:2]
        person = read_person(args.person_dir, rel, shape)
        if person is not None and person.any():
            r = max(1, int(round(min(shape) * args.grow_fraction)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            person = cv2.dilate(person.astype(np.uint8), kernel) > 0
            n_people += 1
            save_preview(args.debug_dir, rel, image, person, (255, 0, 255))
        else:
            person = np.zeros(shape, bool)
        write_keep_mask(args.masks_dir, rel, ~person)
        masked_px += int(person.sum())
        total_px += person.size
    print(f"[PersonMasks] Wrote keep-masks for {len(images)} images in {args.masks_dir}: "
          f"{n_people} with people ({masked_px / max(total_px, 1):.1%} of all pixels masked)")


if __name__ == "__main__":
    main()
