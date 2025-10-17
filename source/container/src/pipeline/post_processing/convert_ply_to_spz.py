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

import os
import spz
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert PLY file to compressed SPZ format")
    parser.add_argument("-i", "--input", required=True, help="Input PLY file path")
    args = parser.parse_args()

    # Generate output path in same directory as input
    input_dir = os.path.dirname(args.input)
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    output_path = os.path.join(input_dir, f"{input_name}.spz")

    # Load PLY file and convert to Babylon.js LUF coordinate system
    unpack_options = spz.UnpackOptions()
    unpack_options.to_coord = spz.CoordinateSystem.LUF
    
    cloud = spz.load_splat_from_ply(args.input, unpack_options)
    print(f"Loaded {cloud.num_points} gaussians from {args.input}")
    
    # Save as compressed SPZ format with Babylon.js LUF coordinate system
    pack_options = spz.PackOptions()
    pack_options.from_coord = spz.CoordinateSystem.LUF
    spz.save_spz(cloud, pack_options, output_path)
    
    # Check compression ratio
    ply_size = os.path.getsize(args.input)
    spz_size = os.path.getsize(output_path)
    compression_ratio = ply_size / spz_size
    print(f"Compression ratio: {compression_ratio:.1f}x smaller ({ply_size} → {spz_size} bytes)")
    print(f"SPZ file saved to: {output_path}")

if __name__ == "__main__":
    main()