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
Generate masks from filtered images for object erasing.
This script takes filtered images (with alpha channels showing detected objects)
and creates binary masks where white pixels indicate areas to be erased.
"""

import os
import cv2
import argparse
import numpy as np

def generate_mask_from_filtered_image(filtered_image_path, output_mask_path):
    """
    Generate a binary mask from a filtered image with alpha channel.
    White pixels indicate areas where objects were detected (to be erased).
    Black pixels indicate areas to keep.
    """
    # Read the filtered image with alpha channel
    img = cv2.imread(filtered_image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"Could not read image: {filtered_image_path}")
    
    # Check if image has alpha channel
    if img.shape[2] == 4:  # BGRA
        # Extract alpha channel
        alpha = img[:, :, 3]
        # Create mask: white where alpha > 0 (object detected), black elsewhere
        mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
    else:  # BGR - fallback for images without alpha
        # Convert to grayscale and create mask from non-black pixels
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.where(gray > 10, 255, 0).astype(np.uint8)
    
    # Save the mask
    cv2.imwrite(output_mask_path, mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='generate-masks-from-filtered',
        description='Generate binary masks from filtered images for object erasing'
    )
    
    parser.add_argument(
        '-i', '--input_dir',
        required=True,
        help='Directory containing filtered images'
    )
    
    parser.add_argument(
        '-o', '--output_dir',
        required=True,
        help='Directory to save generated masks'
    )
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of filtered images
    filtered_files = [f for f in os.listdir(input_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(filtered_files)} filtered images...")
    
    for filtered_file in filtered_files:
        filtered_path = os.path.join(input_dir, filtered_file)
        
        # Generate output mask filename with output_ prefix to match expected pattern
        base_name = os.path.splitext(filtered_file)[0]
        mask_filename = f"output_{base_name}.png"
        mask_path = os.path.join(output_dir, mask_filename)
        
        try:
            generate_mask_from_filtered_image(filtered_path, mask_path)
            print(f"Generated mask: {mask_filename}")
        except Exception as e:
            print(f"Error processing {filtered_file}: {e}")
    
    print(f"Completed generating {len(filtered_files)} masks in {output_dir}")