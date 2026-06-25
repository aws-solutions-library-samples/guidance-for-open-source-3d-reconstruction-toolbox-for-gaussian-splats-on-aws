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

"""Generates a streamed LOD SOG bundle (lod-meta.json) from a Gaussian splat PLY using
splat-transform, producing 4 decimation levels at 100%, 50%, 25%, and 12.5% of Gaussians."""

import argparse
import os
import subprocess
import sys
import tempfile


# Builds a 4-level LOD chain by halving Gaussians at each step, then combines
# all levels into a single lod-meta.json streamed SOG bundle that viewers can
# stream progressively based on camera distance and available bandwidth.
def generate_lod(input_ply: str, output_dir: str) -> int:
    lod_files = []
    with tempfile.TemporaryDirectory() as tmp:
        prev = input_ply
        for level in range(4):
            if level == 0:
                lod_files.append((input_ply, 0))
            else:
                out = os.path.join(tmp, f"lod{level}.ply")
                cmd = ["splat-transform", prev, "--decimate", "50%", "-w", out]
                print(f"Generating LOD {level}: {' '.join(cmd)}")
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    return result.returncode
                lod_files.append((out, level))
                prev = out

        # Combine all LOD levels into streamed SOG bundle
        os.makedirs(output_dir, exist_ok=True)
        lod_meta_path = os.path.join(output_dir, "lod-meta.json")
        cmd = ["splat-transform"]
        for path, level in lod_files:
            cmd += [path, f"--lod={level}"]
        cmd += ["-w", lod_meta_path]
        print(f"Combining LOD levels: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Generate streamed LOD SOG bundle from a Gaussian splat PLY")
    parser.add_argument("-i", "--input", required=True, help="Input PLY file (full-resolution, LOD 0)")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for lod-meta.json and SOG chunks")
    args = parser.parse_args()

    sys.exit(generate_lod(args.input, args.output_dir))


if __name__ == "__main__":
    main()
