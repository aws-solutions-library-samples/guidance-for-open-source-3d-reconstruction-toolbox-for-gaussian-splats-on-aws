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

"""Extract sharp frames from video using sharp-frame-extractor package"""

import os
import subprocess
import argparse
import logging
import cv2
from pathlib import Path

def extract_sharp_frames(
    video_path: str,
    output_dir: str,
    num_frames: int,
    log_level: str = "INFO"
) -> tuple[bool, bool]:
    """
    Extract sharp frames from video using sharp-frame-extractor.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        num_frames: Target number of frames to extract
        log_level: Logging level
    
    Returns:
        tuple: (success: bool, is_portrait: bool)
    """
    try:
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        
        # Validate inputs
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Check video orientation BEFORE extraction
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        is_portrait = height > width
        logging.info(f"Video orientation: {'Portrait' if is_portrait else 'Landscape'} ({width}x{height})")
        
        # Create temp directory for sharp-frame-extractor output
        temp_dir = os.path.join(os.path.dirname(output_dir), "temp_sharp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Build command - sharp-frame-extractor outputs to video_stem subdirectory
        cmd = [
            "sharp-frame-extractor",
            video_path,
            "--count", str(num_frames),
            "--output", temp_dir
        ]
        
        logging.info(f"Running: {' '.join(cmd)}")
        
        # Execute
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if result.stdout:
            logging.debug(result.stdout)
        
        # Move files from temp subdirectory to final output directory
        video_stem = Path(video_path).stem
        source_dir = os.path.join(temp_dir, video_stem)
        
        if os.path.exists(source_dir):
            os.makedirs(output_dir, exist_ok=True)
            for file in os.listdir(source_dir):
                src = os.path.join(source_dir, file)
                dst = os.path.join(output_dir, file)
                os.rename(src, dst)
            
            # Cleanup temp directory
            import shutil
            shutil.rmtree(temp_dir)
            
            logging.info(f"Successfully extracted {num_frames} sharp frames to {output_dir}")
            return True, is_portrait
        else:
            raise RuntimeError(f"Expected output directory not found: {source_dir}")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"sharp-frame-extractor failed: {e}")
        if e.stderr:
            logging.error(e.stderr)
        raise
    except Exception as e:
        logging.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract sharp frames from video')
    parser.add_argument('-i', '--video_path', required=True, help='Input video path')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory')
    parser.add_argument('-n', '--num_frames', type=int, required=True, help='Number of frames')
    parser.add_argument('-ll', '--log-level', default='INFO', help='Log level')
    
    args = parser.parse_args()
    
    success, is_portrait = extract_sharp_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        log_level=args.log_level
    )
