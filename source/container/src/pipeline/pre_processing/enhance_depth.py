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

"""Enhances sparse LiDAR depth images using PromptDA to produce dense, metric-aligned depth maps."""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Model loading (cached singleton)
# ---------------------------------------------------------------------------

_MODEL = None
_DEVICE = ''


def _get_model():
    global _MODEL, _DEVICE
    import torch
    from promptda.promptda import PromptDA
    if _MODEL is None:
        _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading PromptDA (depth-anything/promptda_vitl) on {_DEVICE}...")
        _MODEL = PromptDA.from_pretrained("depth-anything/promptda_vitl").to(_DEVICE).eval()
        print("PromptDA model ready.")
    return _MODEL, _DEVICE


# ---------------------------------------------------------------------------
# Per-image enhancement
# ---------------------------------------------------------------------------

def enhance_single(rgb_path: str, depth_path: str, mask_path, model, device: str) -> None:
    """Run PromptDA on one RGB+sparse-depth pair, optionally applying a mask.

    Masked regions (black pixels in mask) are zeroed in the sparse depth prompt
    so PromptDA does not use them as metric anchors, and zeroed again in the
    output so the GS trainer does not receive hallucinated depth there.
    Overwrites depth_path in-place.
    """
    import cv2
    import numpy as np
    import torch

    rgb_arr = cv2.imread(rgb_path)
    if rgb_arr is None:
        raise RuntimeError(f"Could not read RGB image: {rgb_path}")
    rgb_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb_arr.shape[:2]

    # Decode mask: white=valid, black=masked (scanner body / nadir)
    valid_mask = None
    if mask_path is not None and os.path.exists(mask_path):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape != (orig_h, orig_w):
                m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            valid_mask = m > 128

    # PromptDA requires dimensions divisible by 14 (DINOv2 patch size)
    proc_h = int(orig_h // 14 * 14)
    proc_w = int(orig_w // 14 * 14)
    if proc_h != orig_h or proc_w != orig_w:
        rgb_arr = cv2.resize(rgb_arr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

    img_t = (torch.from_numpy(rgb_arr.astype("float32") / 255.0)
             .permute(2, 0, 1).unsqueeze(0).to(device))

    depth_arr = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_arr is None:
        raise RuntimeError(f"Could not read depth image: {depth_path}")
    sparse_m = depth_arr.astype("float32") / 1000.0  # mm -> metres

    # Zero masked regions so PromptDA does not use them as metric anchors
    if valid_mask is not None:
        sparse_m[~valid_mask] = 0.0

    if proc_h != orig_h or proc_w != orig_w:
        sparse_m = cv2.resize(sparse_m, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)

    prompt = (torch.from_numpy(np.ascontiguousarray(sparse_m))
              .unsqueeze(0).unsqueeze(0).float().to(device))

    with torch.no_grad():
        depth_pred = model.predict(image=img_t, prompt_depth=prompt)

    depth_m = depth_pred.squeeze().cpu().numpy()

    if proc_h != orig_h or proc_w != orig_w:
        depth_m = cv2.resize(depth_m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype("uint16")

    # Zero masked regions in output so GS does not receive hallucinated depth
    if valid_mask is not None:
        depth_mm[~valid_mask] = 0

    cv2.imwrite(depth_path, depth_mm)


# ---------------------------------------------------------------------------
# Directory-level enhancement
# ---------------------------------------------------------------------------

def _find_mask(masks_dir: str, depth_fname: str) -> str | None:
    """Return the mask path for a depth file, or None if masks_dir is absent.

    Supports two mask naming conventions:
      - COLMAP convention:  <stem>.png.png  (extra .png suffix)
      - Plain convention:   <stem>.png
    """
    if masks_dir is None or not os.path.isdir(masks_dir):
        return None
    stem = os.path.splitext(depth_fname)[0]
    for candidate in (stem + ".png.png", stem + ".png"):
        path = os.path.join(masks_dir, candidate)
        if os.path.exists(path):
            return path
    return None


def enhance_depth_dir(depth_dir: str, images_dir: str, masks_dir: str | None, device: str) -> None:
    """Enhance all uint16 PNG depth images in depth_dir using matching RGB images.

    Args:
        depth_dir:  Directory containing uint16 mm PNG depth images (enhanced in-place).
        images_dir: Directory containing matching RGB images (same stem, any image ext).
        masks_dir:  Optional directory containing mask PNGs (white=valid, black=masked).
                    Supports both plain (<stem>.png) and COLMAP (<stem>.png.png) naming.
        device:     Torch device string ('cuda' or 'cpu').
    """
    import cv2

    # Collect depth files recursively to support face_XX subdirectory layouts
    depth_files = []
    for root, _, files in os.walk(depth_dir):
        for f in sorted(files):
            if f.lower().endswith(".png"):
                depth_files.append(os.path.relpath(os.path.join(root, f), depth_dir))

    if not depth_files:
        print(f"No PNG depth files found in {depth_dir}, skipping enhancement.")
        return

    # Validate that at least the first file is uint16 metric depth
    sample = cv2.imread(os.path.join(depth_dir, depth_files[0]), cv2.IMREAD_UNCHANGED)
    if sample is None or str(sample.dtype) != "uint16":
        print(f"Depth files are not uint16 PNG (dtype={getattr(sample, 'dtype', 'None')}), "
              f"skipping enhancement.")
        return

    model, device = _get_model()

    if masks_dir and os.path.isdir(masks_dir):
        print(f"Mask directory: {masks_dir}")
    else:
        print("No mask directory found — enhancing without masks.")
        masks_dir = None

    enhanced = 0
    skipped = 0
    for rel_path in depth_files:
        depth_path = os.path.join(depth_dir, rel_path)
        fname = os.path.basename(rel_path)
        subdir = os.path.dirname(rel_path)
        stem = os.path.splitext(fname)[0]

        # Find matching RGB: same relative subdir + stem, any image extension
        rgb_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(images_dir, subdir, stem + ext)
            if os.path.exists(candidate):
                rgb_path = candidate
                break

        if rgb_path is None:
            print(f"  WARNING: No matching RGB for {rel_path}, skipping.")
            skipped += 1
            continue

        # Find matching mask (supports both flat and subdirectory layouts)
        mask_path = None
        if masks_dir:
            for candidate in (
                os.path.join(masks_dir, subdir, fname + ".png"),  # COLMAP: <name>.png.png
                os.path.join(masks_dir, subdir, fname),           # plain:  <name>.png
            ):
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

        try:
            enhance_single(rgb_path, depth_path, mask_path, model, device)
            enhanced += 1
            if enhanced % 10 == 0:
                print(f"  Enhanced {enhanced}/{len(depth_files)}")
        except Exception as e:
            print(f"  WARNING: Failed to enhance {rel_path}: {e}, keeping original.")
            skipped += 1

    print(f"Enhancement complete: {enhanced} enhanced, {skipped} skipped out of {len(depth_files)} total.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Enhance sparse LiDAR depth images using PromptDA."
    )
    parser.add_argument("-d", "--depth-dir", required=True,
                        help="Directory containing uint16 mm PNG depth images to enhance in-place")
    parser.add_argument("-i", "--images-dir", required=True,
                        help="Directory containing matching RGB images")
    parser.add_argument("-m", "--masks-dir", default=None,
                        help="Optional directory containing mask PNGs (white=valid, black=masked)")
    parser.add_argument("--device", default=None,
                        help="Device to run on: 'cuda' or 'cpu' (default: auto-detect)")
    args = parser.parse_args()

    if not os.path.isdir(args.depth_dir):
        print(f"ERROR: depth-dir does not exist: {args.depth_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.images_dir):
        print(f"ERROR: images-dir does not exist: {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    if args.device:
        device = args.device
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-detected device: {device}")

    enhance_depth_dir(args.depth_dir, args.images_dir, args.masks_dir, device)


if __name__ == "__main__":
    main()
