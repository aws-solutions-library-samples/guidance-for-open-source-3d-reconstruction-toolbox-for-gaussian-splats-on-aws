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

""" Small script that uses the nerfstudio library to convert a .pt Gaussian
splat model file into a .ply file.
Author: M. Wiebe (markw@) """

import argparse
import os
import numpy as np
from nerfstudio.scripts.exporter import ExportGaussianSplat
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pt")
    parser.add_argument("output_ply")
    args = parser.parse_args()

    ckpt_files = os.listdir(args.input_pt)
    ckpt_files = sorted(ckpt_files)
    print(', '.join(ckpt_files))

    last_ckpt_file = os.path.join(args.input_pt, ckpt_files[len(ckpt_files)-1])
    print(last_ckpt_file)
    
    ckpt = torch.load(
        last_ckpt_file,
        map_location=torch.device("cpu"),
        weights_only=True,
        pickle_module=None
    )
    splats = ckpt["splats"]

    position = splats["means"].cpu().numpy()
    ply_channels = [
        ("x", position[:, 0]),
        ("y", position[:, 1]),
        ("z", position[:, 2]),
        ("nx", np.zeros(len(position), dtype=np.float32)),
        ("ny", np.zeros(len(position), dtype=np.float32)),
        ("nz", np.zeros(len(position), dtype=np.float32)),
    ]
    f_dc = splats["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    for i in range(f_dc.shape[1]):
        ply_channels.append((f"f_dc_{i}", f_dc[:, i, np.newaxis]))
    f_rest = splats["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    for i in range(f_rest.shape[-1]):
        ply_channels.append((f"f_rest_{i}", f_rest[:, i, np.newaxis]))
    ply_channels.append(("opacity", splats["opacities"].detach().unsqueeze(-1).cpu().numpy()))
    scale = splats["scales"].detach().cpu().numpy()
    for i in range(3):
        ply_channels.append((f"scale_{i}", scale[:, i, np.newaxis]))
    rot = splats["quats"].detach().cpu().numpy()
    for i in range(4):
        ply_channels.append((f"rot_{i}", rot[:, i, np.newaxis]))

    ExportGaussianSplat.write_ply(args.output_ply, len(position), dict(ply_channels))
