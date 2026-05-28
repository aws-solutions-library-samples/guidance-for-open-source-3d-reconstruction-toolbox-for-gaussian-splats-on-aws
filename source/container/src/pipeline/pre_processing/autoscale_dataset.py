#!/usr/bin/env python3
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

"""
Autoscale dataset to fit within GPU VRAM constraints.

Estimates whether the training dataset will fit in VRAM based on image count
and resolution, then either resizes images (RESIZE mode) or drops images
(DROPOUT mode) so the dataset fits within the available VRAM budget.

VRAM is detected at runtime from the GPU via PyTorch/CUDA.

VRAM estimation model (empirically calibrated on 24 GB GPUs):
    vram = BASE + GB_PER_MPX * megapixels + GB_PER_1K * min(n/1000, IMG_CAP_K)

    Calibration points:
        2000 imgs @ 2074x1372 (2.84 Mpx) -> fits   (est 18.5 GB)
        1774 imgs @ 3840x2160 (8.29 Mpx) -> fits   (est 20.1 GB)
        2242 imgs @ 2046x2046 (4.19 Mpx) -> ~OOM   (est 20.4 GB)

    The image count cost is capped because NerfStudio/gsplat use disk/CPU
    caching for large datasets (>500 images), so VRAM usage flattens beyond
    ~2300 images.

    The minimum output resolution is HD (1920x1080) to preserve Gaussian
    splat rendering quality.

Usage:
    python autoscale_dataset.py -i <image_dir> -m <RESIZE|DROPOUT>
"""

import argparse
import math
import os
import sys

import cv2

# Minimum resolution floor for Gaussian splat quality
MIN_EDGE = 1080

# Maximum resolution ceiling — cap at 4K to prevent excessive memory usage
MAX_LONG_EDGE = 3840

# Empirical VRAM model constants (calibrated on 24 GB GPUs)
BASE_OVERHEAD_GB = 7.3
GB_PER_MEGAPIXEL = 0.5
GB_PER_1K_IMAGES = 4.9
IMG_CAP_K = 2.3  # image count cost caps at this many thousands
VRAM_BUDGET_RATIO = 0.85


def get_vram_gb() -> float:
    """Detect total GPU VRAM in GB from the current device."""
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"Detected GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram:.2f} GB")
            return vram
    except Exception as e:
        print(f"Warning: Could not query GPU VRAM via PyTorch: {e}")
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        mb = float(result.stdout.strip().split("\n")[0])
        vram = mb / 1024.0
        print(f"Detected GPU VRAM via nvidia-smi: {vram:.2f} GB")
        return vram
    except Exception as e:
        print(f"Warning: Could not query GPU VRAM via nvidia-smi: {e}")
    print("Warning: No GPU detected, using conservative default of 24 GB")
    return 24.0


def estimate_vram_needed(num_images: int, width: int, height: int) -> float:
    """Estimate VRAM needed for training with the given dataset."""
    megapixels = (width * height) / 1e6
    capped_k = min(num_images / 1000.0, IMG_CAP_K)
    return BASE_OVERHEAD_GB + GB_PER_MEGAPIXEL * megapixels + GB_PER_1K_IMAGES * capped_k


def compute_target_resolution(num_images: int, vram_budget_gb: float,
                              orig_width: int, orig_height: int) -> tuple:
    """Compute the largest resolution (maintaining aspect ratio) that fits in VRAM.
    Will not go below HD (1920x1080)."""
    capped_k = min(num_images / 1000.0, IMG_CAP_K)
    img_cost = GB_PER_1K_IMAGES * capped_k
    available = vram_budget_gb - BASE_OVERHEAD_GB - img_cost
    if available <= 0:
        available = 0.5

    max_megapixels = available / GB_PER_MEGAPIXEL
    max_pixels = max_megapixels * 1e6
    current_pixels = orig_width * orig_height

    if current_pixels <= max_pixels:
        return orig_width, orig_height

    scale = math.sqrt(max_pixels / current_pixels)
    new_w = max(int(orig_width * scale) & ~1, 2)
    new_h = max(int(orig_height * scale) & ~1, 2)

    # Enforce minimum resolution floor
    if min(new_w, new_h) < MIN_EDGE:
        scale_up = MIN_EDGE / min(orig_width, orig_height)
        new_w = max(int(orig_width * scale_up) & ~1, MIN_EDGE)
        new_h = max(int(orig_height * scale_up) & ~1, MIN_EDGE)
        print(f"Resolution floor applied: minimum edge {MIN_EDGE}px")

    return new_w, new_h


def compute_max_images(vram_budget_gb: float, width: int, height: int) -> int:
    """Compute the maximum number of images that fit in VRAM at the given resolution."""
    megapixels = (width * height) / 1e6
    res_cost = GB_PER_MEGAPIXEL * megapixels
    available = vram_budget_gb - BASE_OVERHEAD_GB - res_cost
    if available <= 0:
        return 1
    max_k = available / GB_PER_1K_IMAGES
    return max(int(min(max_k, IMG_CAP_K) * 1000), 1)


def cap_to_4k(width: int, height: int) -> tuple:
    """Downscale to fit within 4K (3840 on the long edge), preserving aspect ratio."""
    long_edge = max(width, height)
    if long_edge <= MAX_LONG_EDGE:
        return width, height
    scale = MAX_LONG_EDGE / long_edge
    new_w = max(int(width * scale) & ~1, 2)
    new_h = max(int(height * scale) & ~1, 2)
    return new_w, new_h


def get_image_files(image_dir: str) -> list:
    """Return sorted list of image file paths, including subdirectories."""
    exts = ('.png', '.jpg', '.jpeg')
    files = []
    for root, _, filenames in os.walk(image_dir):
        for f in filenames:
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    return sorted(files)


def get_dataset_dimensions(image_files: list) -> tuple:
    """Read the first image to get representative dimensions."""
    for f in image_files:
        img = cv2.imread(f)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    return 0, 0


def resize_images(image_files: list, target_w: int, target_h: int):
    """Resize all images in-place to target dimensions."""
    resized = 0
    for fpath in image_files:
        img = cv2.imread(fpath)
        if img is None:
            continue
        h, w = img.shape[:2]
        if w != target_w or h != target_h:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(fpath, img)
            resized += 1
    print(f"Resized {resized}/{len(image_files)} images to {target_w}x{target_h}")


def dropout_images(image_files: list, max_images: int):
    """Uniformly sample max_images from the list, removing the rest."""
    if len(image_files) <= max_images:
        print(f"No dropout needed: {len(image_files)} images <= {max_images} max")
        return
    indices = set(round(i * (len(image_files) - 1) / (max_images - 1))
                  for i in range(max_images))
    removed = 0
    for idx, fpath in enumerate(image_files):
        if idx not in indices:
            os.remove(fpath)
            removed += 1
    print(f"Dropped {removed} images, kept {max_images}/{len(image_files)}")


def main():
    parser = argparse.ArgumentParser(description="Autoscale dataset for GPU VRAM")
    parser.add_argument("-i", "--image-dir", required=True, help="Path to images directory")
    parser.add_argument("-m", "--mode", default="RESIZE", choices=["RESIZE", "DROPOUT"],
                        help="Autoscale mode: RESIZE or DROPOUT")
    args = parser.parse_args()

    image_files = get_image_files(args.image_dir)
    if not image_files:
        print("No images found, nothing to autoscale")
        return

    num_images = len(image_files)
    orig_w, orig_h = get_dataset_dimensions(image_files)
    if orig_w == 0:
        print("Could not read any images")
        sys.exit(1)

    total_vram = get_vram_gb()
    vram_budget = total_vram * VRAM_BUDGET_RATIO
    estimated_vram = estimate_vram_needed(num_images, orig_w, orig_h)

    print(f"VRAM: {total_vram:.1f} GB, Budget: {vram_budget:.1f} GB")
    print(f"Dataset: {num_images} images @ {orig_w}x{orig_h} "
          f"({orig_w * orig_h / 1e6:.2f} Mpx), Est. VRAM: {estimated_vram:.2f} GB")

    if estimated_vram <= vram_budget:
        # Still cap to 4K even if VRAM is fine
        cap_w, cap_h = cap_to_4k(orig_w, orig_h)
        if cap_w != orig_w or cap_h != orig_h:
            print(f"Resolution exceeds 4K cap, downscaling to {cap_w}x{cap_h}")
            resize_images(image_files, cap_w, cap_h)
        else:
            print("Dataset fits in VRAM, no autoscaling needed")
        return

    print(f"Dataset exceeds VRAM budget by {estimated_vram - vram_budget:.2f} GB, mode={args.mode}")

    if args.mode == "RESIZE":
        target_w, target_h = compute_target_resolution(num_images, vram_budget, orig_w, orig_h)
        # Also enforce 4K ceiling on the computed target
        target_w, target_h = cap_to_4k(target_w, target_h)
        print(f"Target resolution: {target_w}x{target_h} "
              f"({target_w * target_h / 1e6:.2f} Mpx)")
        resize_images(image_files, target_w, target_h)
    else:  # DROPOUT
        max_imgs = compute_max_images(vram_budget, orig_w, orig_h)
        print(f"Max images at current resolution: {max_imgs}")
        dropout_images(image_files, max_imgs)


if __name__ == "__main__":
    main()