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

# PromptDA was trained with dense, low-resolution iPhone LiDAR prompts (192x256)
# and bilinearly resizes the prompt inside the network. Its normalize() uses the
# prompt's min/max, and the sigmoid output is bounded by that range.
PROMPT_LONG_SIDE = 256
# Longest side PromptDA runs at (multiple of 14). 1008 px = 72 patches: ~5k tokens,
# a few GB of attention memory; PromptDA's own examples run at this scale.
PROC_MAX_SIDE = 1008
# A low-res prompt cell needs at least this fraction of valid samples.
PROMPT_MIN_CELL_SUPPORT = 0.25
# Skip images whose depth covers less than this fraction of the frame.
MIN_COVERAGE = 0.01
# Keep the raw depth when PromptDA disagrees with the measured depth this much
# (median |pred - raw| / raw over measured pixels).
MAX_MEDIAN_REL_ERROR = 0.10


def _fill_nearest(values, valid):
    """Fill invalid cells with the nearest valid value."""
    import numpy as np
    try:
        from scipy.ndimage import distance_transform_edt
        _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
        return values[iy, ix]
    except ImportError:
        import cv2
        return cv2.inpaint(values.astype(np.float32), (~valid).astype(np.uint8), 3, cv2.INPAINT_NS)


def build_prompt(depth_m, valid, long_side: int = PROMPT_LONG_SIDE):
    """Turn a full-res depth map with holes into a dense low-res metric prompt.

    Each low-res cell is the mean of the VALID samples under it (holes are not
    averaged in as 0 m). Cells with too little support, and masked regions, are
    filled from the nearest supported cell so the prompt has no zeros.
    """
    import cv2
    import numpy as np

    h, w = depth_m.shape
    scale = min(1.0, long_side / max(h, w))
    ph, pw = max(1, round(h * scale)), max(1, round(w * scale))
    v = valid.astype(np.float32)
    num = cv2.resize(depth_m * v, (pw, ph), interpolation=cv2.INTER_AREA)
    den = cv2.resize(v, (pw, ph), interpolation=cv2.INTER_AREA)
    cell_valid = den >= PROMPT_MIN_CELL_SUPPORT
    if not cell_valid.any():
        cell_valid = den > 0
    low = np.where(cell_valid, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    return _fill_nearest(low, cell_valid).astype(np.float32), float(cell_valid.mean())


def enhance_single(rgb_path: str, raw_depth_path: str, out_depth_path: str,
                   mask_path, model, device: str) -> dict:
    """Run PromptDA on one RGB + sparse-depth pair and write dense uint16 mm depth.

    Reads the raw depth from raw_depth_path (never modified) and writes to
    out_depth_path. Masked regions (black pixels in the mask) are excluded from
    the prompt and zeroed in the output. Pixels where PromptDA saturates at the
    prompt's depth range (it cannot predict beyond it) are left empty unless
    measured. If the prediction disagrees with the measured depth, the raw
    depth is written unchanged.
    """
    import cv2
    import numpy as np
    import torch

    rgb_arr = cv2.imread(rgb_path)
    if rgb_arr is None:
        raise RuntimeError(f"Could not read RGB image: {rgb_path}")
    rgb_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb_arr.shape[:2]

    depth_arr = cv2.imread(raw_depth_path, cv2.IMREAD_UNCHANGED)
    if depth_arr is None:
        raise RuntimeError(f"Could not read depth image: {raw_depth_path}")
    if depth_arr.ndim == 3:
        depth_arr = depth_arr[:, :, 0]
    raw_m = depth_arr.astype(np.float32) / 1000.0  # mm -> metres
    if raw_m.shape != (orig_h, orig_w):
        raw_m = cv2.resize(raw_m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    measured = raw_m > 0

    # Decode mask: white=valid, black=masked (scanner body / nadir)
    keep = np.ones((orig_h, orig_w), dtype=bool)
    if mask_path is not None and os.path.exists(mask_path):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape != (orig_h, orig_w):
                m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            keep = m > 128
    measured &= keep

    coverage = float(measured.sum()) / measured.size
    stats = {"coverage_in": coverage, "status": "raw"}
    if coverage < MIN_COVERAGE:
        stats["reason"] = f"coverage {coverage:.1%} < {MIN_COVERAGE:.0%}"
        if out_depth_path != raw_depth_path:
            cv2.imwrite(out_depth_path, depth_arr)
        return stats

    prompt_m, prompt_support = build_prompt(raw_m, measured)
    p_min, p_max = float(prompt_m.min()), float(prompt_m.max())
    stats.update(prompt_shape=prompt_m.shape, prompt_support=prompt_support,
                 prompt_range=(p_min, p_max))

    # PromptDA requires dimensions divisible by 14 (DINOv2 patch size). Its ViT-L
    # attention grows with the square of the patch count, so cap the working
    # resolution (a 2880x3840 input needs ~188 GiB); the prediction is resized back
    # to the original size below.
    scale = min(1.0, PROC_MAX_SIDE / max(orig_h, orig_w))
    proc_h = max(14, int(orig_h * scale) // 14 * 14)
    proc_w = max(14, int(orig_w * scale) // 14 * 14)
    if proc_h != orig_h or proc_w != orig_w:
        rgb_arr = cv2.resize(rgb_arr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    img_t = (torch.from_numpy(rgb_arr.astype(np.float32) / 255.0)
             .permute(2, 0, 1).unsqueeze(0).to(device))
    prompt = (torch.from_numpy(np.ascontiguousarray(prompt_m))
              .unsqueeze(0).unsqueeze(0).float().to(device))

    with torch.no_grad():
        depth_pred = model.predict(image=img_t, prompt_depth=prompt)
    pred_m = depth_pred.squeeze().float().cpu().numpy()
    if pred_m.shape != (orig_h, orig_w):
        pred_m = cv2.resize(pred_m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Sanity check against the measured depth before trusting the prediction.
    rel = np.abs(pred_m[measured] - raw_m[measured]) / raw_m[measured]
    med_rel = float(np.median(rel))
    stats["median_rel_error"] = med_rel
    if not np.isfinite(med_rel) or med_rel > MAX_MEDIAN_REL_ERROR:
        stats["reason"] = f"median rel. error {med_rel:.1%} > {MAX_MEDIAN_REL_ERROR:.0%}"
        if out_depth_path != raw_depth_path:
            cv2.imwrite(out_depth_path, depth_arr)
        return stats

    # The output can't leave [p_min, p_max]; pixels pinned to either end are
    # really "nearer/farther than the LiDAR saw", so leave them unknown.
    span = max(p_max - p_min, 1e-6)
    saturated = (pred_m >= p_max - 1e-3 * span) | (pred_m <= p_min + 1e-3 * span)
    out_valid = keep & (~saturated | measured)
    depth_mm = np.clip(pred_m * 1000.0, 0, 65535).astype(np.uint16)
    depth_mm[~out_valid] = 0
    cv2.imwrite(out_depth_path, depth_mm)

    stats.update(status="enhanced", coverage_out=float(out_valid.mean()))
    return stats


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
        depth_dir:  Directory of uint16 mm PNG depth images. Originals are copied
                    once to <depth_dir>_raw and always read from there; enhanced
                    maps are written to depth_dir.
        images_dir: Directory containing matching RGB images (same stem, any image ext).
        masks_dir:  Optional directory containing mask PNGs (white=valid, black=masked).
                    Supports both plain (<stem>.png) and COLMAP (<stem>.png.png) naming.
        device:     Torch device string ('cuda' or 'cpu').
    """
    import shutil

    import cv2
    import numpy as np

    depth_dir = os.path.normpath(depth_dir)

    # Keep the measured depth: copy it once to <depth_dir>_raw and always read
    # from there, so re-runs never feed an already-enhanced map back in.
    raw_dir = depth_dir + "_raw"
    if not os.path.isdir(raw_dir):
        shutil.copytree(depth_dir, raw_dir)
        print(f"Backed up raw depth to {raw_dir}")
    else:
        print(f"Reading raw depth from existing backup {raw_dir}")

    # Collect depth files recursively to support face_XX subdirectory layouts
    depth_files = []
    for root, _, files in os.walk(raw_dir):
        for f in sorted(files):
            if f.lower().endswith(".png"):
                depth_files.append(os.path.relpath(os.path.join(root, f), raw_dir))

    if not depth_files:
        print(f"No PNG depth files found in {raw_dir}, skipping enhancement.")
        return

    # Validate that at least the first file is uint16 metric depth
    sample = cv2.imread(os.path.join(raw_dir, depth_files[0]), cv2.IMREAD_UNCHANGED)
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
    rel_errors, cov_in, cov_out = [], [], []
    for rel_path in depth_files:
        raw_path = os.path.join(raw_dir, rel_path)
        depth_path = os.path.join(depth_dir, rel_path)
        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
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
            stats = enhance_single(rgb_path, raw_path, depth_path, mask_path, model, device)
        except Exception as e:
            print(f"  WARNING: Failed to enhance {rel_path}: {e}, keeping original.")
            shutil.copy2(raw_path, depth_path)
            skipped += 1
            continue

        if stats["status"] == "enhanced":
            enhanced += 1
            rel_errors.append(stats["median_rel_error"])
            cov_in.append(stats["coverage_in"])
            cov_out.append(stats["coverage_out"])
            if enhanced == 1:
                p0, p1 = stats["prompt_range"]
                print(f"  Prompt {stats['prompt_shape'][1]}x{stats['prompt_shape'][0]}, "
                      f"{stats['prompt_support']:.0%} cells measured, range {p0:.2f}-{p1:.2f} m")
            if enhanced % 10 == 0:
                print(f"  Enhanced {enhanced}/{len(depth_files)}")
        else:
            print(f"  Kept raw depth for {rel_path}: {stats.get('reason', 'not enhanced')}")
            skipped += 1

    print(f"Enhancement complete: {enhanced} enhanced, {skipped} kept raw out of {len(depth_files)} total.")
    if enhanced:
        print(f"  Coverage {np.median(cov_in):.0%} -> {np.median(cov_out):.0%} (median); "
              f"agreement with measured depth: median rel. error {np.median(rel_errors):.1%} "
              f"(worst image {max(rel_errors):.1%})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Enhance sparse LiDAR depth images using PromptDA."
    )
    parser.add_argument("-d", "--depth-dir", required=True,
                        help="Directory of uint16 mm PNG depth images. Enhanced maps are written here; "
                             "the originals are kept in <depth-dir>_raw")
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
