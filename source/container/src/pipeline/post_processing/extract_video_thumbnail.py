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
Extract first frame from video, downsize by half, and save as PNG thumbnail.
"""

import argparse
import cv2
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Extract first frame from video as thumbnail')
    parser.add_argument('-i', '--input', required=True, help='Input video file path')
    parser.add_argument('-o', '--output', help='Output PNG file path (default: same directory as input)')
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if args.output:
        output_path = args.output
    else:
        input_dir = os.path.dirname(args.input)
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(input_dir, f"{input_name}_thumbnail.png")
    
    try:
        # Skip gracefully if input video doesn't exist (e.g. video export was skipped)
        if not os.path.exists(args.input):
            print(f"Info: Video file not found, skipping thumbnail extraction: {args.input}")
            sys.exit(0)

        # Open video file
        cap = cv2.VideoCapture(args.input)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {args.input}")
            sys.exit(1)
        
        # Read first frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Error: Could not read first frame from {args.input}")
            sys.exit(1)
        
        # Get original dimensions
        height, width = frame.shape[:2]
        
        # Resize to half size
        new_width = width // 2
        new_height = height // 2
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Save as PNG
        success = cv2.imwrite(output_path, resized_frame)
        
        if success:
            print(f"Successfully created thumbnail: {output_path} ({new_width}x{new_height})")
        else:
            print(f"Error: Failed to save thumbnail to {output_path}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error extracting video thumbnail: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()