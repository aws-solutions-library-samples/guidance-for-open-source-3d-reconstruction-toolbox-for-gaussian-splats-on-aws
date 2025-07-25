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

""" This script takes a sparse and dense colmap output and creates a
transforms.json that contains pertainent data for NeRF Studio input """

import os
import sys
import argparse
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from nerfstudio.nerfstudio.process_data.colmap_utils import colmap_to_json

# Create Argument Parser with Rich Formatter
parser = argparse.ArgumentParser(
    prog='create-transform',
    description='Create the NeRF Studio transform for COLMAP input data'
)

# Define the Arguments
parser.add_argument(
    '-d',
    '--data_dir',
    required=True,
    default=None,
    action='store',
    help='Target data directory for the COLMAP project root directory'
)

args = parser.parse_args()

path = str(args.data_dir)
sparse_path = f"{path}/sparse/0"
ply_path = f"{sparse_path}/sparse.ply"

if os.path.isdir(path):
    if os.path.isdir(sparse_path):
        print("Input path exists...creating transforms.json file")
        try:
			# Create json from colmap data
            print(f"Sparse Path: {sparse_path}")
            print(f"PLY Filename: {ply_path}")
            colmap_to_json(recon_dir=Path(sparse_path), output_dir=Path(path), ply_filename=ply_path)
        except Exception as e:
            error_str = str(e).lower()
            if "einsum" in error_str or "subscripts" in error_str or "dimensions" in error_str:
                raise RuntimeError(f"SfM failed to converge - insufficient 3D points generated. This usually indicates poor image quality, insufficient overlap, or challenging scene conditions. Try using more images with better overlap or improving image quality.") from e
            else:
                raise RuntimeError(f"Script failed to complete successfully: {e}") from e
    else:
        print(f"Sparse path does not currently exist: {sparse_path}")
else:
    print(f"Input path: {path} doesn't exist...exiting")
