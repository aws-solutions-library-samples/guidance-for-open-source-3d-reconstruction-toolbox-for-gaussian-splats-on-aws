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

"""
Script to extract a cube around the central object from a 3D Gaussian Splatting .ply file.
Supports multiple methods for determining the cube boundaries.
"""

import numpy as np
import argparse
import sys
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_ply, save_ply, filter_points_in_bounds, calculate_bounds_percentile, calculate_bounds_std, calculate_bounds_fixed, print_filter_stats

def calculate_cube_bounds(positions, method='percentile', cube_size=None, percentile=90, std_multiplier=2.0):
    """Calculate cube boundaries using the specified method."""
    if method == 'percentile':
        return calculate_bounds_percentile(positions, percentile)
    elif method == 'std':
        return calculate_bounds_std(positions, std_multiplier)
    elif method == 'fixed':
        if cube_size is None:
            raise ValueError("cube_size must be specified for 'fixed' method")
        return calculate_bounds_fixed(positions, cube_size)
    else:
        raise ValueError(f"Unknown method: {method}")

def main():
    parser = argparse.ArgumentParser(description='Extract cube around central object from Gaussian Splatting PLY')
    parser.add_argument('input_ply', help='Input PLY file path')
    parser.add_argument('output_ply', help='Output PLY file path')
    parser.add_argument('--method', choices=['percentile', 'std', 'fixed'], default='percentile',
                        help='Method for determining cube bounds')
    parser.add_argument('--percentile', type=float, default=90,
                        help='Percentile for outlier removal (percentile method)')
    parser.add_argument('--std-multiplier', type=float, default=2.0,
                        help='Standard deviation multiplier (std method)')
    parser.add_argument('--cube-size', type=float, default=None,
                        help='Fixed cube size (fixed method)')
    parser.add_argument('--quiet', action='store_true', help='Suppress output statistics')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_ply).exists():
        print(f"Error: Input file '{args.input_ply}' not found")
        sys.exit(1)
    
    # Load PLY data
    gaussian_data = load_ply(args.input_ply)
    original_count = len(gaussian_data['positions'])
    
    # Calculate cube bounds
    bounds, center = calculate_cube_bounds(
        gaussian_data['positions'],
        method=args.method,
        cube_size=args.cube_size,
        percentile=args.percentile,
        std_multiplier=args.std_multiplier
    )
    
    # Filter points
    mask = filter_points_in_bounds(gaussian_data, bounds)
    filtered_count = np.sum(mask)
    
    if filtered_count == 0:
        print("Error: No points found within the specified cube bounds")
        sys.exit(1)
    
    # Save filtered PLY
    save_ply(gaussian_data, mask, args.output_ply)
    
    # Print statistics
    if not args.quiet:
        print_filter_stats(original_count, filtered_count, bounds, center)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
