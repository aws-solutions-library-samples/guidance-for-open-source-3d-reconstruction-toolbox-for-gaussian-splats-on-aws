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
#
# Patches gsplat examples/datasets/colmap.py and examples/simple_trainer.py:
#   1. colmap.py Parser: load per-image segmentation masks from masks/ directory
#   2. colmap.py Dataset.__getitem__: merge seg mask into existing mask
#   3. simple_trainer.py: guard depth loss against empty depth tensors (NaN)

import sys

code_path = sys.argv[1] if len(sys.argv) > 1 else "/opt/ml/code"

# --- Patch 1: colmap.py — add seg_mask_dict to Parser ---
path = f"{code_path}/gsplat/examples/datasets/colmap.py"
with open(path) as f:
    src = f.read()

anchor = 'print(\n            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."\n        )'
injection = '''# Load per-image segmentation masks from masks/ directory
        import imageio as _iio
        seg_masks_dir = os.path.join(data_dir, "masks")
        seg_mask_dict = {}
        if os.path.isdir(seg_masks_dir):
            for _img_name in [imdata[k].name for k in imdata]:
                _mpath = os.path.join(seg_masks_dir, _img_name)
                if os.path.isfile(_mpath):
                    _m = _iio.imread(_mpath)
                    if _m.ndim == 3:
                        _m = _m[..., 0]  # take first channel if RGB/RGBA
                    seg_mask_dict[_img_name] = (_m > 127).astype(bool)
        print(f"[Seg masks] Loaded {len(seg_mask_dict)} masks from {seg_masks_dir}")
        self.seg_mask_dict = seg_mask_dict

        '''
assert anchor in src, f"Patch 1 anchor not found in {path}"
src = src.replace(anchor, injection + anchor)
with open(path, "w") as f:
    f.write(src)
print("Patched colmap.py: seg_mask_dict added to Parser")

# --- Patch 2: colmap.py — merge seg_mask into mask in Dataset.__getitem__ ---
with open(path) as f:
    src = f.read()
old = "        mask = self.parser.mask_dict[camera_id]"
new = """        mask = self.parser.mask_dict[camera_id]
        seg_mask = self.parser.seg_mask_dict.get(self.parser.image_names[index])
        if seg_mask is None:
            # Fallback: try basename in case image_names has a path prefix
            seg_mask = self.parser.seg_mask_dict.get(os.path.basename(self.parser.image_names[index]))
        if seg_mask is not None:
            seg_mask = seg_mask.astype(bool)
            mask = (mask & seg_mask) if mask is not None else seg_mask"""
assert old in src, f"Patch 2 anchor not found in {path}"
src = src.replace(old, new)
with open(path, "w") as f:
    f.write(src)
print("Patched colmap.py: seg_mask merged into mask in Dataset.__getitem__")

# --- Patch 3: simple_trainer.py — guard depth loss against empty depths and zero values ---
# 1. When depths_gt is empty (no 3D point observations), skip depth loss.
# 2. When predicted or gt depth is 0, filter before depth_l1_loss (1/0 = inf -> NaN).
path = f"{code_path}/gsplat/examples/simple_trainer.py"
with open(path) as f:
    src = f.read()
old3 = "            if cfg.depth_loss:\n                # query depths from depth map"
new3 = "            depthloss = torch.tensor(0.0, device=device)\n            if cfg.depth_loss and depths_gt.numel() > 0:\n                # query depths from depth map"
assert old3 in src, f"Patch 3 anchor not found in {path}"
src = src.replace(old3, new3)
old3b = "                depthloss = depth_l1_loss(\n                    depths, depths_gt, scene_scale=self.scene_scale\n                )"
new3b = "                valid = (depths > 0) & (depths_gt > 0)\n                if valid.any():\n                    depthloss = depth_l1_loss(\n                        depths[valid], depths_gt[valid], scene_scale=self.scene_scale\n                    )\n                else:\n                    depthloss = torch.tensor(0.0, device=device)"
assert old3b in src, f"Patch 3b anchor not found in {path}"
src = src.replace(old3b, new3b)
with open(path, "w") as f:
    f.write(src)
print("Patched simple_trainer.py: depth loss guarded against empty/zero depths")

# --- Patch 4: simple_trainer.py — guard L1/SSIM loss against all-masked images ---
# When a segmentation mask covers all pixels, colors[masks] and pixels[masks]
# are empty tensors and .mean() returns NaN, corrupting training.
# Fully-masked images are removed from the dataset before training, but guard
# here as a safety net for any edge cases.
with open(path) as f:
    src = f.read()
old = "            if masks is not None:\n                # Exclude masked pixels (e.g. ego vehicle) from L1.\n                # For SSIM (patch-based), zero out both sides at masked locations\n                # so masked patches don't pull colors toward an arbitrary value.\n                l1loss = l1_loss(colors[masks], pixels[masks]).mean()\n                colors_ssim = colors * masks[..., None]\n                pixels_ssim = pixels * masks[..., None]\n            else:\n                l1loss = l1_loss(colors, pixels).mean()\n                colors_ssim = colors\n                pixels_ssim = pixels"
new = "            if masks is not None and masks.any():\n                # Exclude masked pixels (e.g. ego vehicle) from L1.\n                # For SSIM (patch-based), zero out both sides at masked locations\n                # so masked patches don't pull colors toward an arbitrary value.\n                l1loss = l1_loss(colors[masks], pixels[masks]).mean()\n                colors_ssim = colors * masks[..., None]\n                pixels_ssim = pixels * masks[..., None]\n            elif masks is not None and not masks.any():\n                # All pixels masked — skip loss for this image to avoid NaN from mean([]).\n                l1loss = torch.tensor(0.0, device=device)\n                colors_ssim = colors\n                pixels_ssim = pixels\n            else:\n                l1loss = l1_loss(colors, pixels).mean()\n                colors_ssim = colors\n                pixels_ssim = pixels"
assert old in src, f"Patch 4 anchor not found in {path}"
src = src.replace(old, new)
with open(path, "w") as f:
    f.write(src)
print("Patched simple_trainer.py: L1/SSIM loss guarded against all-masked images")
