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

""" Simple script to extract frames from a video using OpenCV """

import os
import cv2
from pathlib import Path
import numpy as np
from typing import Union, Optional
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

def extract_frame_worker(args):
    """Worker function for extracting a single frame"""
    video_path, frame_idx, output_path, resize_width, resize_height = args
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    
    # Enable auto-rotation to preserve original orientation
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return False
    
    if resize_width and resize_height:
        frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    
    return cv2.imwrite(str(output_path), frame)

def extract_frames(
    video_path: Union[str, Path], 
    output_dir: Union[str, Path], 
    num_frames: int,
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    num_workers: int = 1,
) -> bool:
    """
    Extract frames uniformly distributed across a video and save as PNG images.
    
    Args:
        video_path: Path to input video file (.mov or .mp4)
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract
        resize_width: Optional width to resize frames to
        resize_height: Optional height to resize frames to
        start_time: Start time in seconds
        end_time: End time in seconds (optional)
    """
    try:
        # Convert paths to Path objects
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        # Verify video file exists and has valid extension
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if video_path.suffix.lower() not in ['.mp4', '.mov']:
            raise ValueError(f"Unsupported video format: {video_path.suffix}")

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Check for rotation metadata
        rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        logging.info(f"Video rotation metadata: {rotation}")

        # Calculate frame range based on time constraints
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps) if end_time else total_frames

        if start_frame >= end_frame:
            raise ValueError("Start time must be less than end time")

        logging.info(f"Video info:")
        logging.info(f"Resolution: {width}x{height}")
        logging.info(f"Total frames: {total_frames}")
        logging.info(f"FPS: {fps:.2f}")
        logging.info(f"Duration: {duration:.2f} seconds")
        logging.info(f"Processing range: {start_time:.2f}s to {end_time if end_time else duration:.2f}s")
        logging.info(f"Video orientation: {'Portrait' if height > width else 'Landscape'}")

        # Calculate frame indices to extract
        frame_indices = np.linspace(start_frame, end_frame-1, num_frames, dtype=int)
        
        # Release the initial video capture
        cap.release()
        
        # Prepare arguments for multiprocessing
        worker_args = []
        for i, frame_idx in enumerate(frame_indices):
            output_path = output_dir / f"frame_{i:04d}.png"
            worker_args.append((video_path, frame_idx, output_path, resize_width, resize_height))
        
        # Extract frames using multiprocessing
        if num_workers > 1:
            logging.info(f"Using {num_workers} workers for frame extraction")
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(extract_frame_worker, args) for args in worker_args]
                
                completed = 0
                for future in as_completed(futures):
                    success = future.result()
                    if not success:
                        logging.warning(f"Failed to extract frame")
                    completed += 1
                    if completed % 10 == 0:
                        logging.info(f"Processed {completed}/{num_frames} frames")
        else:
            # Single-threaded extraction
            for i, args in enumerate(worker_args):
                success = extract_frame_worker(args)
                if not success:
                    logging.warning(f"Failed to extract frame {i}")
                if (i + 1) % 10 == 0:
                    logging.info(f"Processed {i + 1}/{num_frames} frames")

        logging.info(f"Successfully extracted {num_frames} frames to {output_dir}")
        return True

    except Exception as e:
        if 'cap' in locals():
            cap.release()
        raise RuntimeError(f"Error extracting frames: {str(e)}") from e

def rotate_images_in_directory(input_dir, degrees=90):
    """
    Rotates all images in a directory by 90 degrees clockwise using OpenCV
    and overwrites the original images.
    
    Args:
        input_dir (str): Path to directory containing images to rotate
        degrees (int, optional): Degrees to rotate. Default is 90 degrees clockwise.
    
    Returns:
        int: Number of images successfully rotated
    """
    # Define supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    # Counter for successfully rotated images
    success_count = 0
    
    # Get all files in the directory
    files = os.listdir(input_dir)
    
    for filename in files:
        # Check if file is an image
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(input_dir, filename)
            
            try:
                # Read the image
                img = cv2.imread(input_path)
                
                if img is None:
                    print(f"Warning: Could not read image {filename}")
                    continue
                
                # Rotate the image 90 degrees clockwise
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                
                # Save the rotated image, overwriting the original
                cv2.imwrite(input_path, rotated_img)
                success_count += 1
                print(f"Rotated and overwritten: {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print(f"Successfully rotated {success_count} images")
    return success_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from a video file.')

    # Required arguments
    parser.add_argument(
        '-i', '--video_path',
        type=str,
        help='Path to the input video file (.mp4 or .mov)'
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        help='Directory to save extracted frames'
    )
    parser.add_argument(
        '-n', '--num_frames',
        type=int,
        help='Number of frames to extract'
    )

    # Optional arguments
    parser.add_argument(
        '-rw', '--resize-width',
        type=int,
        help='Width to resize frames to'
    )
    parser.add_argument(
        '-rh', '--resize-height',
        type=int,
        help='Height to resize frames to'
    )
    parser.add_argument(
        '-st', '--start-time',
        type=float,
        default=0.0,
        help='Start time in seconds (default: 0.0)'
    )
    parser.add_argument(
        '-et', '--end-time',
        type=float,
        help='End time in seconds (default: end of video)'
    )
    parser.add_argument(
        '-nw', '--num-workers',
        type=int,
        default=1,
        help='Number of worker processes for frame extraction (default: 1)'
    )
    parser.add_argument(
        '-ll', '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Validate resize arguments
    if (args.resize_width is None) != (args.resize_height is None):
        parser.error("Both --resize-width and --resize-height must be provided together")

    # Validate number of frames
    if args.num_frames <= 0:
        parser.error("Number of frames must be positive")

    # Validate time range
    if args.start_time < 0:
        args.start_time = 0.0
    if args.end_time is not None and args.end_time <= 0:
        args.end_time = None

    # Extract frames
    success = extract_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        start_time=args.start_time,
        end_time=args.end_time,
        num_workers=args.num_workers
    )

    if not success:
        parser.exit(1, "Frame extraction failed\n")
