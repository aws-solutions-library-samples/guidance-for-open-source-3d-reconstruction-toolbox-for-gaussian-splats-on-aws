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
                    seg_mask_dict[_img_name] = (_m > 127).astype(bool)
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
        if seg_mask is not None:
            seg_mask = seg_mask.astype(bool)
            mask = (mask & seg_mask) if mask is not None else seg_mask"""
assert old in src, f"Patch 2 anchor not found in {path}"
src = src.replace(old, new)
with open(path, "w") as f:
    f.write(src)
print("Patched colmap.py: seg_mask merged into mask in Dataset.__getitem__")

# --- Patch 3: simple_trainer.py — guard depth loss against empty depths ---
# When an image has no 3D point observations, depths_gt is empty and
# 1.0 / depths_gt produces NaN. Initialize depthloss=0 before the block
# so the desc line at the end of the loop always has a valid value,
# then skip the computation when depths_gt is empty.
path = f"{code_path}/gsplat/examples/simple_trainer.py"
with open(path) as f:
    src = f.read()
old = "            if cfg.depth_loss:\n                # query depths from depth map"
new = "            depthloss = torch.tensor(0.0, device=device)\n            if cfg.depth_loss and depths_gt.numel() > 0:\n                # query depths from depth map"
assert old in src, f"Patch 3 anchor not found in {path}"
src = src.replace(old, new)
with open(path, "w") as f:
    f.write(src)
print("Patched simple_trainer.py: depth loss guarded against empty depths")
