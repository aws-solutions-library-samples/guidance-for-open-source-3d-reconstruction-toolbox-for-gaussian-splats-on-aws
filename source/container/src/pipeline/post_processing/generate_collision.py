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

"""Generates a sparse voxel octree (.voxel.json/.voxel.bin) and collision mesh (.collision.glb)
from a Gaussian splat PLY using splat-transform."""

import argparse
import subprocess
import sys


# Builds the splat-transform command for collision generation based on scene type.
# indoor: seals the enclosed interior void then carves navigable space (rooms, buildings).
# outdoor: fills the ground beneath surfaces then carves navigable space (terrain, objects on ground).
# object: bare voxelization with no fill or carve (isolated objects with no walkable floor).
def build_command(input_ply: str, output_voxel: str, scene_type: str, seed_pos: str) -> list:
    cmd = [
        "splat-transform",
        input_ply,
        "--filter-cluster", f"--seed-pos={seed_pos}",
    ]
    if scene_type == "indoor":
        cmd += ["--voxel-external-fill", "--voxel-carve"]
    elif scene_type == "outdoor":
        cmd += ["--voxel-floor-fill", "--voxel-carve"]
    # object: no fill or carve — bare voxelization only
    cmd += ["-K", "smooth", "-w", output_voxel]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Generate collision voxel data from a Gaussian splat PLY")
    parser.add_argument("-i", "--input", required=True, help="Input PLY file")
    parser.add_argument("-o", "--output", required=True, help="Output .voxel.json path")
    parser.add_argument(
        "--scene-type",
        default="outdoor",
        choices=["indoor", "outdoor", "object"],
        help="Scene type controls fill/carve strategy: indoor (external-fill+carve), outdoor (floor-fill+carve), object (bare voxelization)"
    )
    parser.add_argument(
        "--seed-pos",
        default="0,0,0",
        help="Seed position x,y,z for filter-cluster and voxel fill/carve. Should be a known point inside the scene."
    )
    args = parser.parse_args()

    cmd = build_command(args.input, args.output, args.scene_type, args.seed_pos)
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

    # splat-transform may exit non-zero in headless GPU environments (XDG_RUNTIME_DIR warnings)
    # even when it completes successfully. Check output files exist instead.
    import os
    voxel_bin = args.output.replace(".voxel.json", ".voxel.bin")
    if not os.path.exists(args.output) or not os.path.exists(voxel_bin):
        print(f"ERROR: expected output files not found: {args.output}, {voxel_bin}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
