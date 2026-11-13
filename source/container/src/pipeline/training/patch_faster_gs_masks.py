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

"""Build-time patch: make the Faster-GS trainer ignore masked pixels.

NeRFICG's COLMAP loader reads sfm_masks/<image name>.png into view.segmentation
(inverted, so 1 = masked) but the Faster-GS trainer never uses it. In masked
pixels the ground truth is replaced with the (detached) render, so the
photometric loss and its gradients are zero there, like the masked losses of the
other trainers. Without an sfm_masks/ folder, view.segmentation is None and
training is unchanged.

Usage: python patch_faster_gs_masks.py <path to src/Methods/FasterGS/Trainer.py>
"""

import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()

anchor = """        if (alpha_gt := view.alpha) is not None:
            rgb_gt = apply_background_color(rgb_gt, alpha_gt, bg_color)
        loss = self.loss(image, rgb_gt)"""
patch = """        if (alpha_gt := view.alpha) is not None:
            rgb_gt = apply_background_color(rgb_gt, alpha_gt, bg_color)
        # Toolbox patch: ignore masked pixels (sfm_masks/, 1 = masked after inversion)
        if (seg_gt := view.segmentation) is not None:
            keep = (seg_gt < 0.5).to(rgb_gt.dtype)
            rgb_gt = rgb_gt * keep + image.detach() * (1.0 - keep)
        loss = self.loss(image, rgb_gt)"""

if "Toolbox patch: ignore masked pixels" in src:
    print("Faster-GS mask patch already applied")
    sys.exit(0)
assert src.count(anchor) == 1, f"Faster-GS mask patch anchor not found exactly once in {path}"
open(path, "w", encoding="utf-8").write(src.replace(anchor, patch))
print("Patched Faster-GS Trainer.py: masked pixels are excluded from the training loss")
