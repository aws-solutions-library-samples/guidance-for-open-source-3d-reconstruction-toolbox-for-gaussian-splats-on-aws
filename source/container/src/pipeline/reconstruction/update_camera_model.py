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
Update camera model in cameras.txt after undistortion.
This script converts SIMPLE_RADIAL camera models to PINHOLE after undistortion.
"""

import os
import argparse
import shutil

def update_cameras_txt(sparse_path):
    """
    Update cameras.txt to use PINHOLE model after undistortion.
    
    Args:
        sparse_path: Path to the sparse reconstruction directory
    """
    cameras_txt_path = os.path.join(sparse_path, "cameras.txt")
    
    if not os.path.exists(cameras_txt_path):
        print(f"Warning: cameras.txt not found at {cameras_txt_path}")
        return
    
    print(f"\n=== BEFORE UPDATE ===")
    with open(cameras_txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(content)
    
    # Create backup
    backup_path = cameras_txt_path + ".backup"
    shutil.copy2(cameras_txt_path, backup_path)
    
    # Read and update cameras.txt
    with open(cameras_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    updated_lines = []
    for line in lines:
        if line.startswith('#') or not line.strip():
            updated_lines.append(line)
            continue
        
        parts = line.strip().split()
        if len(parts) >= 5:
            camera_id = parts[0]
            model = parts[1]
            width = parts[2]
            height = parts[3]
            
            print(f"Processing camera {camera_id}: model={model}, params={parts[4:]}")
            
            # Convert various models to PINHOLE for 3DGRUT compatibility
            if model in ["SIMPLE_RADIAL", "SIMPLE_PINHOLE"]:
                if model == "SIMPLE_RADIAL" and len(parts) >= 8:
                    # SIMPLE_RADIAL params: f, cx, cy, k
                    f = parts[4]  # focal length
                    cx = parts[5]  # principal point x
                    cy = parts[6]  # principal point y
                    # k = parts[7]  # radial distortion (ignored for SIMPLE_PINHOLE)
                    
                    # Create SIMPLE_PINHOLE line
                    new_line = f"{camera_id} SIMPLE_PINHOLE {width} {height} {f} {cx} {cy}\n"
                    updated_lines.append(new_line)
                    print(f"Updated camera {camera_id}: SIMPLE_RADIAL -> SIMPLE_PINHOLE")
                elif model == "SIMPLE_PINHOLE" and len(parts) >= 7:
                    # Already SIMPLE_PINHOLE, keep as-is
                    updated_lines.append(line)
                    print(f"Camera {camera_id} already SIMPLE_PINHOLE, keeping as-is")
                else:
                    print(f"Warning: Insufficient parameters for camera {camera_id}")
                    updated_lines.append(line)
            else:
                print(f"Camera {camera_id} already has model {model}, keeping as-is")
                updated_lines.append(line)
        else:
            updated_lines.append(line)
    
    # Write updated file
    with open(cameras_txt_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print(f"\n=== AFTER UPDATE ===")
    with open(cameras_txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(content)
    
    print(f"Updated cameras.txt at {cameras_txt_path}")
    print(f"Backup saved to {backup_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update camera model after undistortion')
    parser.add_argument('-s', '--sparse_path', required=True, help='Path to sparse reconstruction directory')
    
    args = parser.parse_args()
    update_cameras_txt(args.sparse_path)