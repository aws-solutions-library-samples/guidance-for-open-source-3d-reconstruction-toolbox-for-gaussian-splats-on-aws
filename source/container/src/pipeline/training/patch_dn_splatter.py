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

"""Patches dn-splatter source files for compatibility with gsplat 1.5.3 and nerfstudio.

NOTE: dn_model.py is replaced entirely at build time via COPY dn_model_patched.py
so no patches for it are needed here.
"""

import sys
import os


def patch_dn_pipeline(filepath):
    """Remove self.datamanager.to(device) call that causes errors."""
    with open(filepath, 'r') as f:
        content = f.read()
    content = content.replace('self.datamanager.to(device)\n', '')
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Patched: {filepath}")


def patch_normal_nerfstudio(filepath):
    """Fix normals loading and mono depth path handling."""
    with open(filepath, 'r') as f:
        content = f.read()

    content = content.replace(
        'return natsorted(glob.glob(f"{self.normal_save_dir}/*.png"))',
        'return sorted(glob.glob(f"{self.normal_save_dir}/*.png"))'
    )

    content = content.replace(
        'normal_filenames = self.get_normal_filepaths()',
        'normal_filenames = self.get_normal_filepaths() if self.config.load_normals else []'
    )

    content = content.replace(
        '''        if self.config.load_depths and len(depth_filenames) == 0:\n            depth_filenames = self.get_depth_filepaths()\n            metadata["mono_depth_filenames"] = [\n                Path(depth_filenames[i]) for i in indices\n            ]''',
        '''        if self.config.load_depths and len(depth_filenames) == 0:\n            mono_depth_paths = self.get_depth_filepaths()\n            metadata["mono_depth_filenames"] = [\n                Path(mono_depth_paths[i]) for i in indices\n            ]'''
    )

    content = content.replace(
        '''            metadata={\n                "depth_filenames": depth_filenames\n                if len(depth_filenames) > 0\n                else None,\n                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,\n                "mask_color": self.config.mask_color,\n                **metadata,\n            },''',
        '''            metadata={\n                "depth_filenames": depth_filenames\n                if len(depth_filenames) > 0\n                else None,\n                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,\n                "depth_mode": "mono" if "mono_depth_filenames" in metadata else ("sensor" if len(depth_filenames) > 0 else "none"),\n                "mask_color": self.config.mask_color,\n                **metadata,\n            },'''
    )

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Patched: {filepath}")


def patch_normals_from_pretrain(filepath):
    """Make omnidata_tools import lazy so dsine path works without omnidata installed."""
    with open(filepath, 'r') as f:
        content = f.read()
    content = content.replace(
        'from omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel\n',
        ''
    )
    content = content.replace(
        'omnidata_pretrained_weights_path = (\n        omnidata_pretrained_weights_path / "omnidata_dpt_normal_v2.ckpt"\n    )\n    model = DPTDepthModel(',
        'omnidata_pretrained_weights_path = (\n        omnidata_pretrained_weights_path / "omnidata_dpt_normal_v2.ckpt"\n    )\n    from omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel\n    model = DPTDepthModel('
    )
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Patched: {filepath}")


def main():
    if len(sys.argv) < 2:
        print("Usage: patch_dn_splatter.py <dn_splatter_root>")
        sys.exit(1)

    root = sys.argv[1]

    patch_dn_pipeline(os.path.join(root, 'dn_splatter', 'dn_pipeline.py'))
    patch_normal_nerfstudio(os.path.join(root, 'dn_splatter', 'data', 'normal_nerfstudio.py'))
    patch_normals_from_pretrain(os.path.join(root, 'dn_splatter', 'scripts', 'normals_from_pretrain.py'))

    print("All dn-splatter patches applied successfully.")


if __name__ == "__main__":
    main()
