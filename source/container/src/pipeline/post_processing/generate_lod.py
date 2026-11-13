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
import json
import os
import subprocess
import sys
import tempfile

PLY_TYPE_SIZES = {
    "char": 1, "uchar": 1, "int8": 1, "uint8": 1,
    "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
    "int": 4, "uint": 4, "int32": 4, "uint32": 4, "float": 4, "float32": 4,
    "double": 8, "float64": 8,
}


# True if a binary PLY's size matches its header (every element fully written).
def ply_is_complete(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            header = b""
            while not header.endswith(b"end_header\n"):
                line = f.readline()
                if not line or len(header) > 65536:
                    return False
                header += line
        lines = header.decode("ascii", errors="replace").splitlines()
        if not any(l.startswith("format binary_little_endian") for l in lines):
            return os.path.getsize(path) > len(header)
        expected = len(header)
        count, stride = 0, 0
        for l in lines:
            parts = l.split()
            if parts[:1] == ["element"]:
                expected += count * stride
                count, stride = int(parts[2]), 0
            elif parts[:1] == ["property"]:
                if parts[1] == "list" or parts[1] not in PLY_TYPE_SIZES:
                    return os.path.getsize(path) > len(header)
                stride += PLY_TYPE_SIZES[parts[1]]
        expected += count * stride
        return expected > len(header) and os.path.getsize(path) == expected
    except (OSError, ValueError, IndexError):
        return False


def lod_meta_is_complete(path: str) -> bool:
    try:
        with open(path) as f:
            json.load(f)
        return True
    except (OSError, ValueError):
        return False


# splat-transform (WebGPU/Dawn) sometimes segfaults while exiting AFTER writing its
# output (exit 139 / signal 11, intermittent, seen with --decimate). Accept the run when the output
# file is complete; fail on any other error or on an incomplete file.
def run_splat_transform(cmd, output_path, is_complete) -> int:
    if os.path.exists(output_path):
        os.remove(output_path)
    result = subprocess.run(cmd)
    if result.returncode == 0:
        return 0
    if os.path.exists(output_path) and is_complete(output_path):
        print(f"WARNING: splat-transform exited with code {result.returncode} after writing "
              f"{output_path}; output is complete, continuing")
        return 0
    print(f"ERROR: splat-transform failed with code {result.returncode} and "
          f"{output_path} is missing or incomplete")
    return result.returncode or 1


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
                rc = run_splat_transform(cmd, out, ply_is_complete)
                if rc != 0:
                    return rc
                lod_files.append((out, level))
                prev = out

        # Combine all LOD levels into streamed SOG bundle
        os.makedirs(output_dir, exist_ok=True)
        lod_meta_path = os.path.join(output_dir, "lod-meta.json")
        cmd = ["splat-transform"]
        for path, level in lod_files:
            cmd += [path, "--tag-lod", str(level)]
        cmd += ["-w", lod_meta_path]
        print(f"Combining LOD levels: {' '.join(cmd)}")
        return run_splat_transform(cmd, lod_meta_path, lod_meta_is_complete)


def main():
    parser = argparse.ArgumentParser(description="Generate streamed LOD SOG bundle from a Gaussian splat PLY")
    parser.add_argument("-i", "--input", required=True, help="Input PLY file (full-resolution, LOD 0)")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for lod-meta.json and SOG chunks")
    args = parser.parse_args()

    sys.exit(generate_lod(args.input, args.output_dir))


if __name__ == "__main__":
    main()
