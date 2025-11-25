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
Convert a sequence of rendered images to MP4 video.
"""

import argparse
import cv2
import os
import sys
import glob
import re

def find_latest_3dgrut_renders(base_path):
    """Find the latest 3DGRUT render directory with highest train time and ours steps."""
    # Find all train-stage directories and use the highest numbered one
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return None
        
    train_stage_dirs = [d for d in os.listdir(base_path) if d.startswith("train-stage-") and os.path.isdir(os.path.join(base_path, d))]
    if not train_stage_dirs:
        print(f"No train-stage directories found in {base_path}")
        return None
    
    latest_stage = max(train_stage_dirs, key=lambda x: int(x.split("-")[-1]))
    train_stage_path = os.path.join(base_path, latest_stage)
    print(f"Using train stage: {latest_stage}")
    
    # Find train directories with pattern train-{time}
    train_dirs = []
    for item in os.listdir(train_stage_path):
        if item.startswith("train-") and os.path.isdir(os.path.join(train_stage_path, item)):
            match = re.match(r'train-(\d+_\d+)', item)
            if match:
                train_dirs.append((item, match.group(1)))
    
    if not train_dirs:
        print(f"No train directories found in {train_stage_path}")
        return None
    
    # Get the latest train directory
    latest_train_dir = max(train_dirs, key=lambda x: x[1])[0]
    train_path = os.path.join(train_stage_path, latest_train_dir)
    print(f"Using train directory: {latest_train_dir}")
    
    # Find ours directories with pattern ours_{steps}
    ours_dirs = []
    for item in os.listdir(train_path):
        if item.startswith("ours_") and os.path.isdir(os.path.join(train_path, item)):
            match = re.match(r'ours_(\d+)', item)
            if match:
                ours_dirs.append((item, int(match.group(1))))
    
    if not ours_dirs:
        print(f"No ours directories found in {train_path}")
        return None
    
    # Get the ours directory with highest steps
    latest_ours_dir = max(ours_dirs, key=lambda x: x[1])[0]
    print(f"Using ours directory: {latest_ours_dir}")
    
    final_path = os.path.join(train_path, latest_ours_dir)
    print(f"Final render path: {final_path}")
    return final_path

def main():
    parser = argparse.ArgumentParser(description='Convert rendered images to MP4 video')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing rendered images')
    parser.add_argument('-o', '--output', required=True, help='Output MP4 file path')
    parser.add_argument('-r', '--framerate', type=int, default=10, help='Frame rate (default: 10)')
    
    args = parser.parse_args()
    
    try:
        # Find the latest 3DGRUT render directory
        render_dir = find_latest_3dgrut_renders(args.input)
        is_3dgrut = False
        if render_dir:
            print(f"Found 3DGRUT render directory: {render_dir}")
            is_3dgrut = True
            image_files = sorted(glob.glob(os.path.join(render_dir, "*.png")))
            if not image_files:
                print(f"No PNG files in {render_dir}, trying fallback search")
                image_files = sorted(glob.glob(os.path.join(args.input, "**", "*.png"), recursive=True))
        else:
            print(f"No 3DGRUT directory found, using fallback search")
            image_files = sorted(glob.glob(os.path.join(args.input, "**", "*.png"), recursive=True))
        
        if not image_files:
            print(f"Error: No PNG images found in {args.input}")
            sys.exit(1)
        
        print(f"Found {len(image_files)} images")
        
        # Read first image to get dimensions
        first_img = cv2.imread(image_files[0])
        if first_img is None:
            print(f"Error: Could not read first image {image_files[0]}")
            sys.exit(1)
        
        height, width, _ = first_img.shape
        
        # Try multiple codecs in order of preference
        codecs_to_try = ['mp4v', 'XVID', 'MJPG']
        out = None
        
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(args.output, fourcc, args.framerate, (width, height))
                if out.isOpened():
                    print(f"Using codec: {codec}")
                    break
                else:
                    out.release()
            except Exception as e:
                print(f"Failed to use codec {codec}: {e}")
                continue
        
        if out is None or not out.isOpened():
            print("Error: Could not initialize video writer with any codec")
            sys.exit(1)
        
        # Write images to video
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is not None:
                out.write(img)
        
        out.release()
        print(f"Successfully created video: {args.output}")
        
    except Exception as e:
        print(f"Error converting images to video: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()