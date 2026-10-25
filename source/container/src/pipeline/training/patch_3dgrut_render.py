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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY

"""Patches 3DGRUT's threedgrut/render.py so LPIPS is computed on a downscaled copy of large images."""

import argparse
import sys

# Helper inserted into render.py. LPIPS (VGG) keeps several full-resolution feature
# maps per image; at 4052x4052 that alone exceeds ~35 GB. Downscaling with area
# averaging before LPIPS caps that memory; PSNR/SSIM still run at full resolution.
HELPER = '''

# Maximum pixel count passed to LPIPS (patched in by patch_3dgrut_render.py).
LPIPS_MAX_PIXELS = 2048 * 2048


# Computes LPIPS on an area-downscaled copy when the image exceeds LPIPS_MAX_PIXELS.
def _lpips_capped(metric, pred, target):
    height, width = pred.shape[-2:]
    if height * width > LPIPS_MAX_PIXELS:
        scale = (LPIPS_MAX_PIXELS / float(height * width)) ** 0.5
        size = (max(1, int(height * scale)), max(1, int(width * scale)))
        pred = torch.nn.functional.interpolate(pred, size=size, mode="area")
        target = torch.nn.functional.interpolate(target, size=size, mode="area")
    return metric(pred, target)

'''

ANCHOR = "\nclass Renderer:"
OLD_CALL = 'criterions["lpips"]('
NEW_CALL = '_lpips_capped(criterions["lpips"], '


# Applies the patch in place; safe to run more than once.
def patch(path):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if "_lpips_capped" in src:
        print(f"Already patched: {path}")
        return
    n_calls = src.count(OLD_CALL)
    if ANCHOR not in src or n_calls == 0:
        print(f"ERROR: expected code not found in {path} (anchor={ANCHOR in src}, lpips calls={n_calls})")
        sys.exit(1)
    src = src.replace(ANCHOR, HELPER + ANCHOR, 1).replace(OLD_CALL, NEW_CALL)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched {n_calls} LPIPS calls in {path} (max {2048 * 2048} pixels)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_py", help="path to 3dgrut/threedgrut/render.py")
    patch(parser.parse_args().render_py)
