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

"""Wrapper to run dn-splatter or ags-mesh model training via nerfstudio ns-train."""

import sys
import os

# Patch torch.load to use weights_only=False for compatibility
import torch
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

def _patch_base_dataset_mask_resize():
    """Replace the hard mask-shape assertion in base_dataset.py with a
    nearest-neighbor resize so mismatched masks are corrected at load time."""
    import importlib.util
    spec = importlib.util.find_spec("nerfstudio.data.datasets.base_dataset")
    if spec is None or spec.origin is None:
        return
    path = spec.origin
    with open(path, "r") as f:
        src = f.read()
    old = (
        'assert data["mask"].shape[:2] == data["image"].shape[:2], ('
    )
    if old not in src:
        return
    new = (
        'if data["mask"].shape[:2] != data["image"].shape[:2]:\n'
        '                import torch.nn.functional as _F\n'
        '                _m = data["mask"]\n'
        '                _ndim = _m.ndim\n'
        '                if _ndim == 2:\n'
        '                    _m = _m.unsqueeze(0).unsqueeze(0).float()\n'
        '                else:\n'
        '                    _m = _m.permute(2, 0, 1).unsqueeze(0).float()\n'
        '                _m = _F.interpolate(_m, size=data["image"].shape[:2], mode="nearest")\n'
        '                if _ndim == 2:\n'
        '                    data["mask"] = _m.squeeze(0).squeeze(0).to(data["mask"].dtype)\n'
        '                else:\n'
        '                    data["mask"] = _m.squeeze(0).permute(1, 2, 0).to(data["mask"].dtype)\n'
        '            assert data["mask"].shape[:2] == data["image"].shape[:2], ('
    )
    patched = src.replace(old, new, 1)
    if patched != src:
        with open(path, "w") as f:
            f.write(patched)
        print(f"Patched base_dataset.py: mask auto-resize before shape assertion")

_patch_base_dataset_mask_resize()

from nerfstudio.scripts.train import entrypoint

if __name__ == "__main__":
    entrypoint()
