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
Iteratively refine splat extraction to focus on the rigid body in center.
"""

import numpy as np
import argparse
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_ply, save_ply, filter_points_in_bounds, calculate_bounds_percentile

def calculate_tight_bounds(positions, method='density_percentile', percentile=75, remove_floor_pct=None):
    """Calculate tighter bounds around the densest region."""
    center = np.mean(positions, axis=0)
    
    if method == 'density_percentile':
        bounds, center = calculate_bounds_percentile(positions, percentile)
        return bounds, center
        
    elif method == 'tight_std':
        # Use smaller std multiplier for tighter bounds
        std_devs = np.std(positions, axis=0)
        half_size = std_devs * 0.5  # Very tight
        
        bounds = {
            'x_min': center[0] - half_size[0],
            'x_max': center[0] + half_size[0],
            'y_min': center[1] - half_size[1],
            'y_max': center[1] + half_size[1],
            'z_min': center[2] - half_size[2],
            'z_max': center[2] + half_size[2]
        }
        
    elif method == 'core_mass':
        # Find the core 50% of points by density
        distances = np.linalg.norm(positions - center, axis=1)
        core_indices = np.argsort(distances)[:len(distances)//2]
        core_positions = positions[core_indices]
        
        # Tight bounds around core with small margin
        min_bounds = np.min(core_positions, axis=0)
        max_bounds = np.max(core_positions, axis=0)
        margin = (max_bounds - min_bounds) * 0.1  # 10% margin
        
        bounds = {
            'x_min': min_bounds[0] - margin[0],
            'x_max': max_bounds[0] + margin[0],
            'y_min': min_bounds[1] - margin[1],
            'y_max': max_bounds[1] + margin[1],
            'z_min': min_bounds[2] - margin[2],
            'z_max': max_bounds[2] + margin[2]
        }
    
    return bounds, center

def main():
    parser = argparse.ArgumentParser(description='Refine splat to extract rigid body in center')
    parser.add_argument('input_ply', help='Input PLY file')
    parser.add_argument('output_ply', help='Output PLY file')
    parser.add_argument('--method', choices=['density_percentile', 'tight_std', 'core_mass'], 
                        default='density_percentile', help='Refinement method')
    parser.add_argument('--percentile', type=float, default=75, 
                        help='Percentile for density_percentile method (lower = tighter)')
    parser.add_argument('--iterations', type=int, default=1, 
                        help='Number of refinement iterations')
    parser.add_argument('--remove-floor', type=float, default=None,
                        help='Remove bottom X percent of points (e.g., 15 for bottom 15%)')
    
    args = parser.parse_args()
    
    gaussian_data = load_ply(args.input_ply)
    original_count = len(gaussian_data['positions'])
    cumulative_mask = np.ones(original_count, dtype=bool)
    
    print(f"Starting with {original_count:,} points")
    
    current_positions = gaussian_data['positions'].copy()
    
    # Apply refinement iterations
    for i in range(args.iterations):
        bounds, center = calculate_tight_bounds(
            current_positions, 
            method=args.method, 
            percentile=args.percentile
        )
        
        # Create temporary gaussian_data for filtering
        temp_data = {'positions': current_positions}
        current_mask = filter_points_in_bounds(temp_data, bounds)
        
        # Update cumulative mask
        temp_mask = cumulative_mask.copy()
        temp_mask[cumulative_mask] = current_mask
        cumulative_mask = temp_mask
        
        filtered_count = np.sum(cumulative_mask)
        
        if filtered_count == 0:
            print(f"No points remaining after iteration {i+1}, stopping")
            break
            
        print(f"Iteration {i+1}: {filtered_count:,} points ({filtered_count/original_count*100:.1f}%)")
        
        # Update positions for next iteration
        current_positions = current_positions[current_mask]
    
    # Remove floor if requested
    if args.remove_floor is not None:
        positions = gaussian_data['positions'][cumulative_mask]
        z_values = positions[:, 2]
        z_threshold = np.percentile(z_values, args.remove_floor)
        
        floor_mask = positions[:, 2] > z_threshold
        
        # Update cumulative mask to exclude floor
        temp_mask = cumulative_mask.copy()
        temp_mask[cumulative_mask] = floor_mask
        cumulative_mask = temp_mask
        
        final_count = np.sum(cumulative_mask)
        print(f"After floor removal: {final_count:,} points ({final_count/original_count*100:.1f}%)")
    
    # Save final result using cumulative mask
    save_ply(gaussian_data, cumulative_mask, args.output_ply)
    print(f"Saved refined splat: {args.output_ply}")

if __name__ == "__main__":
    main()
