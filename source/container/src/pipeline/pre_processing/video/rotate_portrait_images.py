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

"""Rotate portrait video frames 90 degrees clockwise"""

import os
import cv2
import argparse
import subprocess
import json

def check_and_rotate_portrait(image_dir: str, dataset_path: str) -> None:
    """Check if video was portrait using rotation metadata and rotate extracted frames if needed"""
    print(f"[DEBUG] Starting rotation check for directory: {image_dir}")
    print(f"[DEBUG] Dataset path: {dataset_path}")
    
    # Find original video file
    video_file = None
    for f in os.listdir(dataset_path):
        if f.lower().endswith(('.mov', '.mp4')):
            video_file = os.path.join(dataset_path, f)
            break
    
    if not video_file:
        print(f"[DEBUG] No video file found in {dataset_path}, skipping rotation check")
        return
    
    print(f"[DEBUG] Checking video file: {video_file}")
    
    # Use ffprobe to get rotation metadata
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', video_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        
        rotation = 0
        for stream in metadata.get('streams', []):
            if stream.get('codec_type') == 'video':
                # Check for rotation tag
                tags = stream.get('tags', {})
                if 'rotate' in tags:
                    rotation = int(tags['rotate'])
                # Also check side_data_list for display matrix rotation
                for side_data in stream.get('side_data_list', []):
                    if side_data.get('rotation'):
                        rotation = int(side_data['rotation'])
                break
        
        print(f"[DEBUG] Video rotation metadata: {rotation} degrees")
        
        # Portrait videos typically have 90 or 270 degree rotation
        is_portrait = abs(rotation) in [90, 270]
        
    except Exception as e:
        print(f"[DEBUG] Failed to read rotation metadata: {e}")
        # Fallback to dimension check
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"[DEBUG] Failed to open video file")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"[DEBUG] Video dimensions (fallback): {width}x{height}")
        is_portrait = height > width
    
    if is_portrait:
        print(f"[DEBUG] Portrait video detected, rotating extracted images")
        
        images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"[DEBUG] Found {len(images)} images to rotate")
        
        if not images:
            print(f"[DEBUG] No images found to rotate")
            return
        
        for img_file in images:
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(img_path, rotated)
        
        # Write marker file
        marker = os.path.join(dataset_path, '.video_orientation')
        with open(marker, 'w') as f:
            f.write('portrait')
        print(f"[DEBUG] Rotated {len(images)} images and created orientation marker")
    else:
        print(f"[DEBUG] Landscape video detected, no rotation needed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', required=True)
    parser.add_argument('-d', '--dataset_path', required=True)
    args = parser.parse_args()
    
    check_and_rotate_portrait(args.image_dir, args.dataset_path)
