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
EQUIRECTANGULAR TO PERSPECTIVE IMAGE SEQUENCE OPTIMIZER
AUTHOR: ERIC CORNWELL

This script transforms equirectangular (360°) images into optimized perspective image sequences
for Structure-from-Motion (SfM) processing. The algorithm addresses the challenge of creating
spatially and temporally consistent image sequences from spherical imagery.

ALGORITHM OVERVIEW:
==================

1. EQUIRECTANGULAR TO CUBEMAP CONVERSION:
   - Converts each ERP image to 6 cubemap faces (front, back, left, right, up, down)
   - Applies optional face filtering to remove unwanted views
   - Reconstructs filtered ERP images from modified cubemaps

2. CONNECTIVE IMAGE GENERATION:
   - Generates perspective images at multiple horizontal angles (15°, 30°, 45°, 60°)
   - Generates perspective images at multiple vertical angles (0° to 135°)
   - Creates images for key frames: start (0%), middle (50%), end (100%)
   - Creates additional frames at multiple insertion distances: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%

3. VIEW-BASED SEQUENCE ORGANIZATION:
   - Reorganizes images by cubemap face rather than temporal sequence
   - Creates separate directories for each view: left, front, right, back, up, down
   - Optimizes view order for sequential SfM matching

4. VIEW NODE INSERTION (SPATIAL CONSISTENCY):
   Each view gets multiple "node images" inserted at specific positions to improve spatial continuity:
   
   - LEFT VIEW (multiple insertions at 20%, 40%, 60%, 80%):
     * Uses frames at respective positions as sources
     * Inserts 16 perspective images per position (4 angles × 4 perspectives)
     * Perspectives: 04→03→02→01 (reverse order)
   
   - BACK VIEW (multiple insertions at 10%, 30%, 50%, 70%):
     * Uses frames at respective positions as sources
     * Reverses file order first, then adjusts insertions to 90%, 70%, 50%, 30% of reversed sequence
     * Inserts 16 perspective images per position (4 angles × 4 perspectives)
     * Perspectives: 04→01→02→03
   
   - RIGHT VIEW (multiple insertions at 20%, 40%, 60%, 80%):
     * Uses frames at respective positions as sources
     * Inserts 16 perspective images per position (4 angles × 4 perspectives) 
     * Perspectives: 02→01→04→03 (reverse angle order)
   
   - FRONT VIEW (multiple insertions at 30%, 50%, 70%, 90%):
     * Uses frames at respective positions as sources
     * Reverses file order first, then adjusts insertions to 70%, 50%, 30%, 10% of reversed sequence
     * Inserts 16 perspective images per position (4 angles × 4 perspectives)
     * Perspectives: 02→03→04→01
   
   - UP/DOWN VIEWS:
     * UP: Rotates images 90°, adds vertical connective images
       - With angled_up_views option: Adds additional 360° rotation with camera angled up (75° and 90°)
     * DOWN: Rotates images -90°, reverses order after processing
       - With angled_down_views option: Adds additional 360° rotation with camera angled down (15° and 30°)

5. CONNECTIVE IMAGE INSERTION:
   - Adds transition images between views to improve feature matching
   - Uses images from temporally adjacent frames (first/last)
   - Appends to end of each view sequence

6. FINAL SEQUENCE ASSEMBLY:
   - Reorders views: Left→Front→Right→Back→Up→Down
   - Renumbers all images sequentially (00001, 00002, etc.)
   - Creates final optimized sequence for SfM processing

SPATIAL CONSISTENCY PRINCIPLE:
=============================
The algorithm ensures that view nodes use source images from frames that correspond
to their insertion position in the temporal sequence. This maintains spatial coherence
by using multiple insertion points across the sequence:

- LEFT VIEW: 20%, 40%, 60%, 80% insertions use corresponding frame sources
- BACK VIEW: 10%, 30%, 50%, 70% insertions use corresponding frame sources
- RIGHT VIEW: 20%, 40%, 60%, 80% insertions use corresponding frame sources
- FRONT VIEW: 30%, 50%, 70%, 90% insertions use corresponding frame sources

This multi-node approach significantly improves SfM convergence by providing dense,
spatially consistent feature correspondences across the optimized image sequence.

This approach significantly improves SfM convergence by providing spatially consistent
feature correspondences across the optimized image sequence.
"""

import os
import re
import cv2
import math
import argparse
import torch
import numpy as np
from imageio.v2 import imread, imwrite
import Equirec2Cube
from PIL import Image
import py360convert
import subprocess
import multiprocessing
import shutil
import glob
import logging
import concurrent.futures
from rich.logging import RichHandler
from functools import partial

def reverse_file_order(directory_path):
    """
    Reverses the order of sequentially named files in a directory.
    Example: 00000.png -> 00099.png, 00001.png -> 00098.png, etc.
    Uses a more efficient approach with in-memory mapping.
    
    Args:
        directory_path (str): Path to the directory containing the files
    """
    try:
        # Get list of files and sort them
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        files.sort()
        
        if not files:
            return  # No files to process
            
        # Create temporary directory
        temp_dir = os.path.join(directory_path, 'temp_reverse')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Get the total number of files and adjust for zero-based naming
        total_files = len(files) - 1  # Subtract 1 to account for starting at 0
        width = len(files[0].split('.')[0])  # Get width of number portion
        
        # Create a mapping of old to new filenames to minimize disk operations
        file_mapping = []
        for i, filename in enumerate(files):
            name, ext = os.path.splitext(filename)
            new_name = str(total_files - i).zfill(width) + ext
            old_path = os.path.join(directory_path, filename)
            temp_path = os.path.join(temp_dir, new_name)
            file_mapping.append((old_path, temp_path))
        
        # Process files in batches to improve performance
        batch_size = 50  # Adjust based on memory constraints
        for i in range(0, len(file_mapping), batch_size):
            batch = file_mapping[i:i+batch_size]
            # Copy files to temp directory
            for old_path, temp_path in batch:
                shutil.copy2(old_path, temp_path)
        
        # Move files back to original directory
        for filename in sorted(os.listdir(temp_dir)):
            temp_path = os.path.join(temp_dir, filename)
            new_path = os.path.join(directory_path, filename)
            shutil.move(temp_path, new_path)

        # Remove temporary directory
        os.rmdir(temp_dir)
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise RuntimeError(f"An error occurred reversing file order: {str(e)}") from e

# Define a top-level function for image rotation
def _rotate_single_image(image_path_angle):
    image_path, angle = image_path_angle
    img = cv2.imread(image_path)
    if img is None:
        return f"Failed to read: {image_path}"
        
    # Get image dimensions
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate image
    rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h))
    
    # Save with same filename
    success = cv2.imwrite(image_path, rotated_img)
    if success:
        return f"Rotated: {image_path}"
    else:
        return f"Failed to save: {os.path.basename(image_path)}"

def rotate_images(path, angle):
    """
    Rotate image(s) by specified angle and save with same name.
    Uses sequential processing to avoid multiprocessing issues.
    
    Args:
        path (str): Path to image file or folder containing images
        angle (float): Rotation angle in degrees (positive = counterclockwise)
    """
    if os.path.isfile(path):
        # Single file
        print(f"Rotating image: {path} by {angle} degrees")
        result = _rotate_single_image((path, angle))
        print(result)
    elif os.path.isdir(path):
        # Directory
        print(f"Rotating images in: {path} by {angle} degrees")
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(path, ext)))
            image_files.extend(glob.glob(os.path.join(path, ext.upper())))
        image_files = list(set(image_files))  # Remove duplicates
        print(f"Found {len(image_files)} images to rotate")
        
        # Process sequentially to avoid multiprocessing issues
        for image_path in image_files:
            _rotate_single_image((image_path, angle))
    else:
        print(f"Error: {path} is not a valid file or directory")
        return

# Define a top-level function for multiprocessing
def _copy_file(src_dst):
    src, dst = src_dst
    shutil.copy(src, dst)
    return dst

def insert_view_node(view_subfolder, node_image_paths, insertion_index, view_num_len, tail, log):
    """Insert view node images at specified position in sequence.
    Optimized with batch operations and parallel processing."""
    current_files = [f for f in os.listdir(view_subfolder) if f.endswith(tail)]

    # Log insertion details for debugging
    view_name = os.path.basename(os.path.normpath(view_subfolder))
    log.info(f"Inserting {len(node_image_paths)} node images at index {insertion_index} in view '{view_name}'")
    log.info(f"Current file count: {len(current_files)}")

    # Debug: List existing files around insertion point
    if log.level == logging.DEBUG:
        sorted_files = sorted([int(os.path.splitext(f)[0]) for f in current_files])
        insertion_range = [i for i in sorted_files if abs(i - insertion_index) <= 5]
        log.debug(f"Files around insertion point {insertion_index}: {insertion_range}")

    # Create a list of file operations to perform
    move_operations = []
    copy_operations = []
    
    # Identify files that need to be shifted
    current_files.sort(reverse=True)  # Process in reverse order to avoid overwriting
    for existing_file in current_files:
        file_num = int(os.path.splitext(existing_file)[0])
        if file_num >= insertion_index:
            old_path = os.path.join(view_subfolder, existing_file)
            new_path = os.path.join(view_subfolder, f"{file_num + len(node_image_paths):0{view_num_len}d}{tail}")
            move_operations.append((old_path, new_path))
    
    # Identify valid node images to copy
    for i, node_image_path in enumerate(node_image_paths):
        if node_image_path and os.path.isfile(node_image_path):
            destination_path = os.path.join(view_subfolder, f"{insertion_index + i:0{view_num_len}d}{tail}")
            copy_operations.append((node_image_path, destination_path))
        elif node_image_path:
            log.warning(f"Error: {node_image_path} is not a valid file path.")
    
    # Execute move operations in batches
    log.info(f"Shifting {len(move_operations)} existing files to make room for node images")
    batch_size = 50  # Adjust based on system capabilities
    for i in range(0, len(move_operations), batch_size):
        batch = move_operations[i:i+batch_size]
        for old_path, new_path in batch:
            shutil.move(old_path, new_path)
    
    # Execute copy operations in parallel if there are many files
    log.info(f"Copying {len(copy_operations)} node images")
    
    # Use sequential processing for all copy operations to avoid multiprocessing issues
    for src, dst in copy_operations:
        shutil.copy(src, dst)
    
    log.info(f"Successfully inserted {len(copy_operations)} of {len(node_image_paths)} node images")

    # Verify final file count
    final_files = [f for f in os.listdir(view_subfolder) if f.endswith(tail)]
    log.info(f"Final file count: {len(final_files)}")

    if len(copy_operations) < len(node_image_paths):
        log.warning(f"Some node images could not be inserted. Expected {len(node_image_paths)}, inserted {len(copy_operations)}.")



def get_node_image_paths(data_dir, source_frame, angles_horiz, perspectives, include_angled_up=False, angles_up=None, include_angled_down=False, angles_down=None):
    """Generate list of node image paths for given frame and perspectives.
    
    Args:
        data_dir: Base data directory
        source_frame: Frame to use for node images
        angles_horiz: Horizontal angles list (may be reversed for some views)
        perspectives: Perspective numbers list
        include_angled_up: Whether to include angled up views
        angles_up: List of upward angles to include
        include_angled_down: Whether to include angled down views
        angles_down: List of downward angles to include
        
    Returns:
        List of image paths in consistent order: all angles for perspective 1, then all angles for perspective 2, etc.
    """
    paths = []

    # Generate paths directly in the correct order
    for perspective in perspectives:
        for angle in angles_horiz:
            # Add standard horizontal perspective
            path = os.path.join(data_dir, source_frame, "filtered_imgs",
                              f"pers_imgs_{angle}_horiz",
                              f"{source_frame}_perspective_{perspective:02d}.png")
            paths.append(path)
            
            # Add angled down perspectives if requested
            if include_angled_down and angles_down:
                for down_angle in angles_down:
                    down_path = os.path.join(data_dir, source_frame, "filtered_imgs",
                                  f"pers_imgs_{angle}_horiz_{down_angle}_down",
                                  f"{source_frame}_perspective_{perspective:02d}.png")
                    paths.append(down_path)
            
            # Add angled up perspectives if requested - using vertical perspective images
            if include_angled_up and angles_up:
                for up_angle in angles_up:
                    up_path = os.path.join(data_dir, source_frame, "filtered_imgs",
                                  f"pers_imgs_{up_angle}_vert_{angle}_horiz",
                                  f"{source_frame}_perspective_{perspective:02d}.png")
                    paths.append(up_path)

    return paths

def get_oval_node_paths(data_dir, center_frame, neighbor_frames, angles_horiz, perspectives, include_angled_up=False, angles_up=None, include_angled_down=False, angles_down=None):
    """Generate oval view node paths using center frame and neighbors for temporal translation.
    
    Creates an elliptical camera path by using different temporal frames for different angles
    while maintaining the original perspective ordering.
    
    Args:
        data_dir: Base data directory
        center_frame: Primary frame at insertion position
        neighbor_frames: List of neighboring frame names [prev, next]
        angles_horiz: Horizontal angles list (may be reversed for some views)
        perspectives: Perspective numbers list
        include_angled_up: Whether to include angled up views
        angles_up: List of upward angles to include
        include_angled_down: Whether to include angled down views
        angles_down: List of downward angles to include
    
    Returns:
        List of image paths creating oval camera motion in consistent order
    """
    paths = []

    # Validate neighbor frames have required connective images, fallback to center frame if not
    def validate_frame(frame_name, perspective, angle, tilt_type=None, tilt_angle=None):
        if tilt_type is None:
            test_path = os.path.join(data_dir, frame_name, "filtered_imgs",
                                    f"pers_imgs_{angle}_horiz",
                                    f"{frame_name}_perspective_{perspective:02d}.png")
        elif tilt_type == "down":
            test_path = os.path.join(data_dir, frame_name, "filtered_imgs",
                                    f"pers_imgs_{angle}_horiz_{tilt_angle}_{tilt_type}",
                                    f"{frame_name}_perspective_{perspective:02d}.png")
        elif tilt_type == "up":
            # For up views, use vertical perspective images
            test_path = os.path.join(data_dir, frame_name, "filtered_imgs",
                                    f"pers_imgs_{tilt_angle}_vert_{angle}_horiz",
                                    f"{frame_name}_perspective_{perspective:02d}.png")
        return os.path.exists(test_path)

    # Check if neighbor frames have all required perspective images
    def validate_all_perspectives(frame_name, perspectives, angles, angles_up=None, angles_down=None):
        for perspective in perspectives:
            for angle in angles:
                if not validate_frame(frame_name, perspective, angle):
                    return False
                if angles_up:
                    for up_angle in angles_up:
                        if not validate_frame(frame_name, perspective, angle, "up", up_angle):
                            return False
                if angles_down:
                    for down_angle in angles_down:
                        if not validate_frame(frame_name, perspective, angle, "down", down_angle):
                            return False
        return True

    prev_frame = neighbor_frames[0] if len(neighbor_frames) > 0 and validate_all_perspectives(neighbor_frames[0], perspectives, angles_horiz, 
                                                                                             angles_up if include_angled_up else None,
                                                                                             angles_down if include_angled_down else None) else center_frame
    next_frame = neighbor_frames[1] if len(neighbor_frames) > 1 and validate_all_perspectives(neighbor_frames[1], perspectives, angles_horiz, 
                                                                                             angles_up if include_angled_up else None,
                                                                                             angles_down if include_angled_down else None) else center_frame

    # Create fixed frame assignments for each angle position
    frame_sources = [center_frame, prev_frame, center_frame, next_frame]

    # Generate paths directly in the correct order
    for perspective in perspectives:
        for i, angle in enumerate(angles_horiz):
            # Use modulo to cycle through frame sources
            frame_source = frame_sources[i % len(frame_sources)]
            
            # Add standard horizontal perspective
            path = os.path.join(data_dir, frame_source, "filtered_imgs",
                              f"pers_imgs_{angle}_horiz",
                              f"{frame_source}_perspective_{perspective:02d}.png")
            paths.append(path)
            
            # Add angled down perspectives if requested
            if include_angled_down and angles_down:
                for down_angle in angles_down:
                    down_path = os.path.join(data_dir, frame_source, "filtered_imgs",
                                  f"pers_imgs_{angle}_horiz_{down_angle}_down",
                                  f"{frame_source}_perspective_{perspective:02d}.png")
                    paths.append(down_path)
            
            # Add angled up perspectives if requested - using vertical perspective images
            if include_angled_up and angles_up:
                for up_angle in angles_up:
                    up_path = os.path.join(data_dir, frame_source, "filtered_imgs",
                                  f"pers_imgs_{up_angle}_vert_{angle}_horiz",
                                  f"{frame_source}_perspective_{perspective:02d}.png")
                    paths.append(up_path)
                    
    return paths

def process_view(view, view_subfolder, data_dir, angles_horiz, angles_vert, 
                original_file_count, view_num_len, tail, frame_names, log):
    """Process a specific view with its node insertion and connective images."""
    persp_image_paths = []

    if view == "left":
        # 20% 40% 60% 80%
        # For left view, we need to:
        # 1. Get the current files
        # 2. Insert nodes at the correct positions
        # 3. This ensures proper SfM sequence

        # First, get all current files and their count
        current_files = [f for f in os.listdir(view_subfolder) if f.endswith(tail)]
        file_count = len(current_files)
        log.info(f"Left view: processing {file_count} files")

        # Now insert nodes at specific positions
        # We'll use absolute positions rather than relative ones
        # This ensures consistent spacing regardless of previous insertions

        # Calculate insertion positions
        positions = [
            (0.2, frame_names['frame_20']),  # 20% position
            (0.4, frame_names['frame_40']),  # 40% position
            (0.6, frame_names['frame_60']),  # 60% position
            (0.8, frame_names['frame_80'])   # 80% position
        ]

        # Sort positions from highest to lowest to avoid shifting issues
        # When we insert at higher positions first, lower positions remain stable
        positions.sort(reverse=True)

        # Insert nodes at each position
        for pos, frame in positions:
            # Calculate insertion index
            insertion_index = int(file_count * pos)
            log.info(f"Left view: inserting at position {pos} (index {insertion_index})")

            # Get node paths
            if frame_names.get('use_oval', False):
                node_paths = get_oval_node_paths(data_dir, frame, frame_names[f'neighbors_{int(pos*100)}'], 
                                               angles_horiz[::-1], [4, 3, 2, 1],
                                               include_angled_up=frame_names.get('angled_up_views', False),
                                               angles_up=frame_names.get('angles_up'),
                                               include_angled_down=frame_names.get('angled_down_views', False),
                                               angles_down=frame_names.get('angles_down'))
            else:
                node_paths = get_node_image_paths(data_dir, frame, angles_horiz[::-1], [4, 3, 2, 1],
                                               include_angled_up=frame_names.get('angled_up_views', False),
                                               angles_up=frame_names.get('angles_up'),
                                               include_angled_down=frame_names.get('angled_down_views', False),
                                               angles_down=frame_names.get('angles_down'))

            # Insert nodes
            insert_view_node(view_subfolder, node_paths, insertion_index, view_num_len, tail, log)
        # LEFT-TO-FRONT connective images
        for angle in angles_horiz:
            persp_image_paths.append(os.path.join(data_dir, frame_names['last'], "filtered_imgs",
                                                f"pers_imgs_{angle}_horiz", f"{frame_names['last']}_perspective_01.png"))
    elif view == "front":
        # 10, 30, 50, 70
        # 30, 50, 70, 90
        # For front view, we need a different approach:
        # 1. Get the current files
        # 2. Reverse their order
        # 3. Insert nodes at the correct positions in a single operation
        # 4. This ensures proper SfM sequence

        # First, get all current files and their count
        current_files = [f for f in os.listdir(view_subfolder) if f.endswith(tail)]
        file_count = len(current_files)
        log.info(f"Front view: processing {file_count} files")

        # Reverse the file order
        reverse_file_order(view_subfolder)

        # For front view, we need to insert all nodes at once to maintain proper spacing
        # This is critical for SfM sequence consistency

        # Define the insertion positions and corresponding frames
        # For reversed views, we need to use the opposite positions (1-pos)
        insertion_data = [
            # (position, frame_name, insertion_index)
            (0.3, frame_names['frame_30'], int(file_count * 0.7)),  # 30% → 70% in reversed order
            (0.5, frame_names['frame_50'], int(file_count * 0.5)),  # 50% → 50% in reversed order
            (0.7, frame_names['frame_70'], int(file_count * 0.3)),  # 70% → 30% in reversed order
            (0.9, frame_names['frame_90'], int(file_count * 0.1))   # 90% → 10% in reversed order
        ]

        # Insert nodes at each position
        for pos, frame, idx in insertion_data:
            log.info(f"Front view: inserting frame {frame} at position {pos} (index {idx})")

            # Get node paths
            if frame_names.get('use_oval', False):
                node_paths = get_oval_node_paths(data_dir, frame, frame_names[f'neighbors_{int(pos*100)}'], 
                                               angles_horiz, [2, 3, 4, 1],
                                               include_angled_up=frame_names.get('angled_up_views', False),
                                               angles_up=frame_names.get('angles_up'),
                                               include_angled_down=frame_names.get('angled_down_views', False),
                                               angles_down=frame_names.get('angles_down'))
            else:
                node_paths = get_node_image_paths(data_dir, frame, angles_horiz, [2, 3, 4, 1],
                                               include_angled_up=frame_names.get('angled_up_views', False),
                                               angles_up=frame_names.get('angles_up'),
                                               include_angled_down=frame_names.get('angled_down_views', False),
                                               angles_down=frame_names.get('angles_down'))

            # Insert nodes
            insert_view_node(view_subfolder, node_paths, idx, view_num_len, tail, log)

        # FRONT-TO-RIGHT connective images
        rev_angles_horiz = angles_horiz[::-1]
        for angle in rev_angles_horiz:
            persp_image_paths.append(os.path.join(data_dir, frame_names['first'], "filtered_imgs",
                                                f"pers_imgs_{angle}_horiz", f"{frame_names['first']}_perspective_01.png"))
        for angle in rev_angles_horiz:
            persp_image_paths.append(os.path.join(data_dir, frame_names['first'], "filtered_imgs",
                                                f"pers_imgs_{angle}_horiz", f"{frame_names['first']}_perspective_04.png"))
        for angle in rev_angles_horiz:
            persp_image_paths.append(os.path.join(data_dir, frame_names['first'], "filtered_imgs",
                                                f"pers_imgs_{angle}_horiz", f"{frame_names['first']}_perspective_03.png"))
    elif view == "right":
        # 20%, 40%, 60%, 80%
        # For right view, we need to:
        # 1. Get the current files
        # 2. Insert nodes at the correct positions
        # 3. This ensures proper SfM sequence

        # First, get all current files and their count
        current_files = [f for f in os.listdir(view_subfolder) if f.endswith(tail)]
        file_count = len(current_files)
        log.info(f"Right view: processing {file_count} files")

        # Now insert nodes at specific positions
        # We'll use absolute positions rather than relative ones
        # This ensures consistent spacing regardless of previous insertions

        # Calculate insertion positions
        positions = [
            (0.2, frame_names['frame_20']),  # 20% position
            (0.4, frame_names['frame_40']),  # 40% position
            (0.6, frame_names['frame_60']),  # 60% position
            (0.8, frame_names['frame_80'])   # 80% position
        ]

        # Sort positions from highest to lowest to avoid shifting issues
        # When we insert at higher positions first, lower positions remain stable
        positions.sort(reverse=True)

        # Insert nodes at each position
        for pos, frame in positions:
            # Calculate insertion index
            insertion_index = int(file_count * pos)
            log.info(f"Right view: inserting at position {pos} (index {insertion_index})")

            # Get node paths
            if frame_names.get('use_oval', False):
                node_paths = get_oval_node_paths(data_dir, frame, frame_names[f'neighbors_{int(pos*100)}'], 
                                               angles_horiz[::-1], [2, 1, 4, 3],
                                               include_angled_up=frame_names.get('angled_up_views', False),
                                               angles_up=frame_names.get('angles_up'),
                                               include_angled_down=frame_names.get('angled_down_views', False),
                                               angles_down=frame_names.get('angles_down'))
            else:
                node_paths = get_node_image_paths(data_dir, frame, angles_horiz[::-1], [2, 1, 4, 3],
                                               include_angled_up=frame_names.get('angled_up_views', False),
                                               angles_up=frame_names.get('angles_up'),
                                               include_angled_down=frame_names.get('angled_down_views', False),
                                               angles_down=frame_names.get('angles_down'))

            # Insert nodes
            insert_view_node(view_subfolder, node_paths, insertion_index, view_num_len, tail, log)
        # RIGHT-TO-BACK connective images
        for angle in angles_horiz:
            persp_image_paths.append(os.path.join(data_dir, frame_names['last'], "filtered_imgs",
                                                f"pers_imgs_{angle}_horiz", f"{frame_names['last']}_perspective_03.png"))
    elif view == "back":
        # 90, 70, 50, 30
        # 10, 30, 50, 70
        # For back view, we need a different approach:
        # 1. Get the current files
        # 2. Reverse their order
        # 3. Insert nodes at the correct positions in a single operation
        # 4. This ensures proper SfM sequence

        # First, get all current files and their count
        current_files = [f for f in os.listdir(view_subfolder) if f.endswith(tail)]
        file_count = len(current_files)
        log.info(f"Back view: processing {file_count} files")

        # Reverse the file order
        reverse_file_order(view_subfolder)

        # For back view, we need to insert all nodes at once to maintain proper spacing
        # This is critical for SfM sequence consistency

        # Define the insertion positions and corresponding frames
        # For reversed views, we need to use the opposite positions (1-pos)
        insertion_data = [
            # (position, frame_name, insertion_index)
            (0.1, frame_names['frame_10'], int(file_count * 0.9)),  # 10% → 90% in reversed order
            (0.3, frame_names['frame_30'], int(file_count * 0.7)),  # 30% → 70% in reversed order
            (0.5, frame_names['frame_50'], int(file_count * 0.5)),  # 50% → 50% in reversed order
            (0.7, frame_names['frame_70'], int(file_count * 0.3))   # 70% → 30% in reversed order
        ]

        # Insert nodes at each position
        for pos, frame, idx in insertion_data:
            log.info(f"Back view: inserting frame {frame} at position {pos} (index {idx})")

            # Get node paths
            if frame_names.get('use_oval', False):
                node_paths = get_oval_node_paths(data_dir, frame, frame_names[f'neighbors_{int(pos*100)}'], 
                                               angles_horiz, [4, 1, 2, 3],
                                               include_angled_up=frame_names.get('angled_up_views', False),
                                               angles_up=frame_names.get('angles_up'),
                                               include_angled_down=frame_names.get('angled_down_views', False),
                                               angles_down=frame_names.get('angles_down'))
            else:
                node_paths = get_node_image_paths(data_dir, frame, angles_horiz, [4, 1, 2, 3],
                                               include_angled_up=frame_names.get('angled_up_views', False),
                                               angles_up=frame_names.get('angles_up'),
                                               include_angled_down=frame_names.get('angled_down_views', False),
                                               angles_down=frame_names.get('angles_down'))

            # Insert nodes
            insert_view_node(view_subfolder, node_paths, idx, view_num_len, tail, log)

        # BACK-to-UP connective images
        rev_angles_vert = angles_vert[::-1][4:-1]
        for angle in rev_angles_vert:
            persp_image_paths.append(os.path.join(data_dir, frame_names['first'], "filtered_imgs",
                                                f"pers_imgs_{angle}_vert", f"{frame_names['first']}_perspective_04.png"))
    elif view == "up":
        rotate_images(view_subfolder, 90)
        
        # If angled up views are enabled, add additional angled up views to the up view
        # (angled_up_views will only be true if 'up' is not in remove_face_list)
        if frame_names.get('angled_up_views', False) and frame_names.get('angles_up'):
            # Get all files in the view subfolder
            current_files = [f for f in os.listdir(view_subfolder) if f.endswith(tail)]
            file_count = len(current_files)
            log.info(f"Up view: adding angled up views, current file count: {file_count}")
            
            # Add angled up views for each horizontal angle
            additional_paths = []
            for angle in angles_horiz:
                for up_angle in frame_names.get('angles_up'):
                    # Use the middle frame as source for angled up views
                    src_path = os.path.join(data_dir, frame_names['middle'], "filtered_imgs",
                                          f"pers_imgs_{up_angle}_vert_{angle}_horiz",
                                          f"{frame_names['middle']}_perspective_01.png")
                    if os.path.isfile(src_path):
                        dest_path = os.path.join(view_subfolder, f"{file_count:0{view_num_len}d}{tail}")
                        log.info(f"Adding angled up view ({up_angle}°) for angle {angle}° to up view")
                        shutil.copy(src_path, dest_path)
                        file_count += 1
                        additional_paths.append(dest_path)
            
            # Rotate all the additional images
            for path in additional_paths:
                rotate_images(path, 90)
        
        # UP-TO-DOWN connective images
        for angle in angles_vert[1:]:
            filename = os.path.join(data_dir, frame_names['last'], "filtered_imgs", f"pers_imgs_{angle}_vert", f"{frame_names['last']}_perspective_02.png")
            rotate_images(filename, 180)
            persp_image_paths.append(filename)

    elif view == "down":
        rotate_images(view_subfolder, -90)
        
        # If angled down views are enabled, add additional angled down views to the down view
        # (angled_down_views will only be true if 'down' is not in remove_face_list)
        if frame_names.get('angled_down_views', False) and frame_names.get('angles_down'):
            # Get all files in the view subfolder
            current_files = [f for f in os.listdir(view_subfolder) if f.endswith(tail)]
            file_count = len(current_files)
            log.info(f"Down view: adding angled down views, current file count: {file_count}")
            
            # Add angled down views for each horizontal angle
            additional_paths = []
            for angle in angles_horiz:
                for down_angle in frame_names.get('angles_down'):
                    # Use the middle frame as source for angled down views
                    # For downward views, we use the horizontal perspective images with adjusted vertical angle
                    src_path = os.path.join(data_dir, frame_names['middle'], "filtered_imgs",
                                          f"pers_imgs_{angle}_horiz_{down_angle}_down",
                                          f"{frame_names['middle']}_perspective_01.png")
                    if os.path.isfile(src_path):
                        dest_path = os.path.join(view_subfolder, f"{file_count:0{view_num_len}d}{tail}")
                        log.info(f"Adding angled down view ({down_angle}°) for angle {angle}° to down view")
                        shutil.copy(src_path, dest_path)
                        file_count += 1
                        additional_paths.append(dest_path)
            
            # Rotate all the additional images
            for path in additional_paths:
                rotate_images(path, -90)

    return persp_image_paths

if __name__ == '__main__':
    # Create Argument Parser with Rich Formatter
    parser = argparse.ArgumentParser(
    prog='equirectangular-to-perspective-images',
    description='Transform ERP images into a sequence of perspective views \
        using cube maps. An optimization regime can be applied to ensure views \
        are sequentially and spatially consistent with the other views, \
        thus improving sequential SfM matching. \
        A filter can be applied to remove unwanted cube faces'
    )

    # Define the Arguments
    parser.add_argument(
        '-d', '--data_dir',
        required=True,
        default=None,
        action='store',
        help='Target data directory for the ERP images')

    parser.add_argument(
        '-rf', '--remove_faces',
        type=str,
        default='',
        help="""Comma-separated list of faces to remove.
        Can be 'back,down,front,left,right,up'"""
    )

    parser.add_argument(
        '-ossfo', '--optimize_sequential_spherical_frame_order',
        required=False,
        default='true',
        action='store',
        help='Whether to enable optimization of spherical video frames \
            to help solve SfM (default is "true")'
    )

    parser.add_argument(
        '-gpu', '--use_gpu',
        required=False,
        default='true',
        action='store',
        help='Whether to enable GPU acceleration (default is "true")'
    )

    parser.add_argument(
        '-log', '--log_level',
        required=False,
        default='info',
        action='store',
        help='Level of logs to write to stdout \
            (default is "info", can be "error" or "debug")'
    )

    parser.add_argument(
        '-oval', '--use_oval_nodes',
        required=False,
        default='true',
        action='store',
        help='Whether to use oval view node paths for better SfM convergence (default is "false")'
    )
    
    parser.add_argument(
        '-down', '--angled_down_views',
        required=False,
        default='true',
        action='store',
        help='Whether to include additional perspective images angled slightly down (15° and 30°) (default is "false")'
    )
    
    parser.add_argument(
        '-up', '--angled_up_views',
        required=False,
        default='true',
        action='store',
        help='Whether to include additional perspective images angled up (75° and 90°) (default is "false")'
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.INFO
    if str(args.log_level).lower() == "debug":
        level = logging.DEBUG
    elif str(args.log_level).lower() == "error":
        level = logging.ERROR
    logging.basicConfig(
        level = level,
        format = "%(message)s",
        handlers = [RichHandler()]
    )
    log = logging.getLogger()

    # Setup paths
    data_dir = str(args.data_dir)
    # Parse comma-separated faces
    if args.remove_faces:
        remove_face_list = [face.strip() for face in args.remove_faces.split(',') if face.strip()]
    else:
        remove_face_list = []
    thread_count = multiprocessing.cpu_count()
    optimize_seq_spherical_frames = True
    if str(args.optimize_sequential_spherical_frame_order).lower() == "true":
        optimize_seq_spherical_frames = True
    else:
        optimize_seq_spherical_frames = False

    # If you need to use GPU to accelerate (especially for the need of converting many images)
    USE_GPU = False
    if str(args.use_gpu).lower() == "true":
        USE_GPU = True
    Image.MAX_IMAGE_PIXELS = 1000000000

    # Enable oval view nodes for better SfM convergence
    use_oval_nodes = False
    if str(args.use_oval_nodes).lower() == "true":
        use_oval_nodes = True
        
    # Enable angled down views (only if down face is not removed)
    angled_down_views = False
    if str(args.angled_down_views).lower() == "true" and 'down' not in remove_face_list:
        angled_down_views = True
    elif str(args.angled_down_views).lower() == "true" and 'down' in remove_face_list:
        log.warning("Angled down views requested but 'down' face is in remove_face_list. Disabling angled down views.")
        
    # Enable angled up views (only if up face is not removed)
    angled_up_views = False
    if str(args.angled_up_views).lower() == "true" and 'up' not in remove_face_list:
        angled_up_views = True
    elif str(args.angled_up_views).lower() == "true" and 'up' in remove_face_list:
        log.warning("Angled up views requested but 'up' face is in remove_face_list. Disabling angled up views.")

    angles_horiz = [15, 30, 45, 60]
    angles_vert = [135, 120, 105, 90, 75, 60, 45, 30, 15, 0]
    angles_down = [15, 30]  # Angles for downward-tilted views (15° and 30° down)
    angles_up = [75, 90]    # Angles for upward-tilted views (75° and 90° up - using vertical perspective images)

    try:
        # Check that input directory exists
        if os.path.isdir(data_dir):
            # Get list of all files in data directory
            filenames = os.listdir(data_dir)
            filenames = sorted(filenames)
            start_frame_filename = filenames[0]
            stop_frame_filename = filenames[-1]
            middle_frame_filename = filenames[int(math.ceil(len(filenames)//2))]

            # Additional frames for view nodes at insertion distances
            frame_10_filename = filenames[int(len(filenames) * 0.1)]
            frame_20_filename = filenames[int(len(filenames) * 0.2)]
            frame_30_filename = filenames[int(len(filenames) * 0.3)]
            frame_40_filename = filenames[int(len(filenames) * 0.4)]
            frame_50_filename = filenames[int(len(filenames) * 0.5)]
            frame_60_filename = filenames[int(len(filenames) * 0.6)]
            frame_70_filename = filenames[int(len(filenames) * 0.7)]
            frame_80_filename = filenames[int(len(filenames) * 0.8)]
            frame_90_filename = filenames[int(len(filenames) * 0.9)]

            img = cv2.imread(os.path.join(data_dir, start_frame_filename))
            height, width = img.shape[:2]
            pers_dim = int(float(max(height, width))/4)
            if filenames is not None:
                for filename in filenames:
                    base_name, extension = os.path.splitext(filename)
                    # if using opencv frame extraction, will be in format "frame_298_01520.png"
                    if base_name[:5].lower() == "frame":
                        img_num = base_name.split('_')[1] # only grab the new frame numbers
                    else:
                        # this will be case for no image processing like deblur
                        img_num = os.path.basename(base_name)
                    if extension.lower() == ".jpeg" or \
                        extension.lower() == ".jpg" or \
                        extension.lower() == ".png":
                        log.info(f"+++ Processing ERP image {str(int(img_num)+1)} of {str(len(filenames))} +++")
                        orig_path = os.path.join(data_dir, f"{base_name}{extension}")

                        # Prepare images into separate sequential directories for
                        # reordering based on neighboring faces
                        new_dir = os.path.join(data_dir, img_num)

                        if not os.path.isdir(new_dir):
                            os.mkdir(new_dir)

                        # Move input file to its own directory
                        new_path = os.path.join(new_dir, f"{img_num}{extension}")

                        if not os.path.isfile(new_path):
                            log.info(f"Moving {orig_path} to {new_path}")
                            shutil.move(orig_path, new_path)

                        # Read the Equirectangular to Cubemap projection
                        img = imread(new_path, pilmode='RGBA')
                        #img = cv2.resize(, (2048, 1024), interpolation=cv2.INTER_AREA)
                        dims = img.shape

                        # Equirectangular to Cubemap
                        try:
                            # The parameters are equirectangular height/width, cubemap dim, and use GPU or not
                            e2c = Equirec2Cube.Equirec2Cube(dims[0], dims[1], int(float(dims[0])/2), CUDA=USE_GPU)

                            batch = torch.FloatTensor(img.astype(float)/255).permute(2, 0, 1)[None, ...]
                            if USE_GPU: batch = batch.cuda()

                            # First convert the image to cubemap
                            cubemap_tensor = e2c(batch)
                            cubemap = cubemap_tensor.permute(0, 2, 3, 1).cpu().numpy()
                        except Exception as e:
                            raise RuntimeError(f"An error occurred during Equirectangular to Cubemap: {str(e)}") from e

                        # Now we save the cubemap to disk
                        order = ['right', 'down', 'left', 'back', 'front', 'up']
                        for i, term in enumerate(order):
                            face = (cubemap[i] * 255).astype(np.uint8)
                            if not os.path.isdir(f"{new_dir}/faces"):
                                os.mkdir(f"{new_dir}/faces")
                            log.info(f"Saving face {term} to {new_dir}/faces/{term}.png")
                            imwrite(f"{new_dir}/faces/{term}.png", face)

                        # Remove the unwanted faces
                        if len(remove_face_list) > 0:
                            if remove_face_list[0] != '' and remove_face_list[0] != "":
                                for remove_face in remove_face_list:
                                    # Create a transparent image and overwrite the face image
                                    img_height, img_width = int(float(dims[0])/2), int(float(dims[0])/2)
                                    n_channels = 4
                                    transparent_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)
                                    cubemap_face_filename = os.path.join(new_dir, "faces", f"{str(remove_face).lower()}.png")
                                    # Save the image for visualization
                                    cv2.imwrite(cubemap_face_filename, transparent_img)

                        # Cubemap image faces to Equirectangular
                        cube_back = np.array(Image.open(os.path.join(new_dir, "faces", "back.png")))
                        cube_down = np.array(Image.open(os.path.join(new_dir, "faces", "down.png")))
                        cube_front = Image.open(os.path.join(new_dir, "faces", "front.png"))
                        cube_left = np.array(Image.open(os.path.join(new_dir, "faces", "left.png")))
                        cube_right = Image.open(os.path.join(new_dir, "faces", "right.png"))
                        cube_up = Image.open(os.path.join(new_dir, "faces", "up.png"))

                        # Flip faces to correspond to mapping
                        flip_cube_front = np.array(cube_front.transpose(Image.FLIP_LEFT_RIGHT))
                        flip_cube_up = np.array(cube_up.transpose(Image.FLIP_TOP_BOTTOM))
                        flip_cube_right = np.array(cube_right.transpose(Image.FLIP_LEFT_RIGHT))

                        # Convert Cubemap to ERP
                        cube_dice = [cube_left, flip_cube_front, flip_cube_right, cube_back, flip_cube_up, cube_down]
                        try:
                            erp_img = py360convert.c2e(cubemap=cube_dice, h=dims[0], w= dims[1], cube_format='list')
                        except Exception as e:
                            raise RuntimeError(f"An error occurred converting cubemap to ERP: {str(e)}") from e

                        filtered_img_dir = os.path.join(new_dir, "filtered_imgs")
                        if not os.path.isdir(filtered_img_dir):
                            os.mkdir(filtered_img_dir)
                        Image.fromarray(erp_img.astype(np.uint8)).save(os.path.join(filtered_img_dir, f"{img_num}.png"))

                        # Generate "connective images" between change in views to increase sfm convergence
                        if optimize_seq_spherical_frames is True:
                            # Include neighbor frames for oval view nodes if enabled
                            key_frames = [start_frame_filename, stop_frame_filename, middle_frame_filename,
                                        frame_10_filename, frame_20_filename, frame_30_filename, frame_40_filename,
                                        frame_50_filename, frame_60_filename, frame_70_filename, frame_80_filename, frame_90_filename]

                            # Add neighbor frames for oval nodes and additional frames for multiple nodes
                            insertion_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

                            if use_oval_nodes:
                                # Add neighbors of key insertion frames
                                neighbor_indices = set()
                                for pct in insertion_points:
                                    idx = int(len(filenames) * pct)
                                    # Add previous and next frames if they exist
                                    if idx > 0:
                                        neighbor_indices.add(idx - 1)
                                    if idx < len(filenames) - 1:
                                        neighbor_indices.add(idx + 1)
                                    # Always add the frame itself
                                    neighbor_indices.add(idx)
                                
                                # Convert indices to filenames and add to key_frames
                                neighbor_frames = [filenames[i] for i in neighbor_indices]
                                unique_neighbors = list(set(neighbor_frames))
                                key_frames.extend(unique_neighbors)
                                log.info(f"Generating connective images for {len(unique_neighbors)} frames")
                                if level == logging.DEBUG:
                                    log.debug(f"Frame list: {unique_neighbors}")
                            
                            if filename in key_frames:
                                # Horizontal Connective Images
                                for angle in angles_horiz:
                                    pers_img_dir_horiz = os.path.join(filtered_img_dir, f"pers_imgs_{str(angle)}_horiz")
                                    if not os.path.isdir(pers_img_dir_horiz):
                                        os.mkdir(pers_img_dir_horiz)
                                    try:
                                        log.info(f"Extracting connective view images for horizontal angle {str(angle)} into directory {pers_img_dir_horiz}")
                                        # Run the converter script for ERP to perspective images
                                        subprocess.run([
                                            "python", "spherical/360ImageConverterforColmap.py",
                                            "-i", filtered_img_dir,
                                            "-o", pers_img_dir_horiz,
                                            "--overlap", "0",
                                            "--fov", "90", "90",
                                            "--base_angle", str(angle), "45",
                                            "--resolution", str(pers_dim), str(pers_dim),
                                            "--threads", str(thread_count),
                                            "--exclude_v_angles", "90"
                                        ], check=True)
                                        
                                        # If angled down views are enabled, generate additional angled down views
                                        if angled_down_views:
                                            for down_angle in angles_down:
                                                # Create directory for angled down views
                                                pers_img_dir_down = os.path.join(filtered_img_dir, f"pers_imgs_{angle}_horiz_{down_angle}_down")
                                                if not os.path.isdir(pers_img_dir_down):
                                                    os.mkdir(pers_img_dir_down)
                                                    
                                                log.info(f"Extracting angled down ({down_angle}°) view images for horizontal angle {str(angle)}")
                                                # Run the converter script with vertical angle adjusted downward
                                                # For downward angles of 15° and 30°, we add to 45° (45° is horizontal, higher values tilt down)
                                                vertical_angle = 45 + down_angle
                                                subprocess.run([
                                                    "python", "spherical/360ImageConverterforColmap.py",
                                                    "-i", filtered_img_dir,
                                                    "-o", pers_img_dir_down,
                                                    "--overlap", "0",
                                                    "--fov", "90", "90",
                                                    "--base_angle", str(angle), str(vertical_angle),
                                                    "--resolution", str(pers_dim), str(pers_dim),
                                                    "--threads", str(thread_count),
                                                    "--exclude_v_angles", "90"
                                                ], check=True)
                                                
                                        # If angled up views are enabled, generate additional angled up views
                                        # For upward views, we'll use the vertical perspective images (75° and 90°)
                                        if angled_up_views:
                                            for up_angle in angles_up:
                                                # Create directory for angled up views
                                                pers_img_dir_up = os.path.join(filtered_img_dir, f"pers_imgs_{up_angle}_vert_{angle}_horiz")
                                                if not os.path.isdir(pers_img_dir_up):
                                                    os.mkdir(pers_img_dir_up)
                                                    
                                                log.info(f"Extracting angled up ({up_angle}°) view images for horizontal angle {str(angle)}")
                                                # Run the converter script with vertical angle set to up_angle
                                                # For 75° and 90°, these are already defined in angles_vert
                                                subprocess.run([
                                                    "python", "spherical/360ImageConverterforColmap.py",
                                                    "-i", filtered_img_dir,
                                                    "-o", pers_img_dir_up,
                                                    "--overlap", "0",
                                                    "--fov", "90", "90",
                                                    "--base_angle", str(angle), str(up_angle),
                                                    "--resolution", str(pers_dim), str(pers_dim),
                                                    "--threads", str(thread_count)
                                                ], check=True)
                                    except Exception as e:
                                        raise RuntimeError(f"An error occurred converting ERP to perspective images: {str(e)}") from e
                                # Vertical Connective Images
                                for angle in angles_vert:
                                    pers_img_dir_vert = os.path.join(filtered_img_dir, f"pers_imgs_{str(angle)}_vert")
                                    if not os.path.isdir(pers_img_dir_vert):
                                        os.mkdir(pers_img_dir_vert)
                                    try:
                                        log.info(f"Extracting connective view images for vertical angle {str(angle)} into directory {pers_img_dir_vert}")
                                        # Run the converter script for ERP to perspective images
                                        subprocess.run([
                                            "python", "spherical/360ImageConverterforColmap.py",
                                            "-i", filtered_img_dir,
                                            "-o", pers_img_dir_vert,
                                            "--overlap", "0",
                                            "--fov", "90", "90",
                                            "--base_angle", "0", str(angle),
                                            "--resolution", str(pers_dim), str(pers_dim),
                                            "--threads", str(thread_count),
                                        ], check=True)
                                    except Exception as e:
                                        raise RuntimeError(f"An error occurred converting ERP to perspective images: {str(e)}") from e
                # Reorder the image sequence to be primarily ordered by view across frames instead of inverse
                # Theoretically, this will allow better matching during SfM due to parallax effect in sequential frames
                # Only get subfolders in the input directory                # .../images/01
                subfolders = [ f.path for f in os.scandir(data_dir) if f.is_dir() ]
                subfolders = sorted(subfolders)
                view_dir_path_list = []
                
                # Create a dictionary to organize files by view
                view_files = {}
                
                # First pass: collect all files and organize by view
                for subfolder in subfolders:
                    faces_subfolder = os.path.join(subfolder, "faces")
                    if not os.path.exists(faces_subfolder):
                        continue
                        
                    files = [os.path.join(path, name) for path, subdirs, files in os.walk(faces_subfolder) for name in files]
                    
                    for img_file in files:
                        head, tail = os.path.splitext(img_file)
                        view = os.path.basename(head)
                        path_parts = re.split(r"[/\\]", head)
                        path_parts.pop()
                        path_parts.pop()
                        img_num = f"{path_parts[len(path_parts)-1]}"
                        path_parts.pop()
                        path_parts.pop()
                        
                        if path_parts[0] == '':
                            root_view_path = os.path.join("/", *path_parts, "views")
                        else:
                            root_view_path = os.path.join(*path_parts, "views")
                            
                        view_dir = os.path.join(root_view_path, view)
                        filename_str = f"{img_num}{tail}"
                        dest_path = os.path.join(view_dir, filename_str)
                        
                        if view not in view_files:
                            view_files[view] = []
                        view_files[view].append((img_file, dest_path))
                
                # Create view directories
                if not os.path.isdir(root_view_path):
                    os.makedirs(root_view_path, exist_ok=True)
                    
                for view, files in view_files.items():
                    view_dir = os.path.join(root_view_path, view)
                    if view_dir not in view_dir_path_list:
                        view_dir_path_list.append(view_dir)
                    if not os.path.isdir(view_dir):
                        os.makedirs(view_dir, exist_ok=True)
                
                # Process files in batches by view
                for view, files in view_files.items():
                    log.info(f"Moving {len(files)} files to {view} view")
                    
                    # Process in batches
                    batch_size = 50
                    for i in range(0, len(files), batch_size):
                        batch = files[i:i+batch_size]
                        for src, dst in batch:
                            try:
                                shutil.move(src, dst)
                            except Exception as e:
                                log.warning(f"Error moving {src} to {dst}: {str(e)}")
                                # Continue processing other files instead of failing completely

                image_folders = os.listdir(data_dir)
                view_num_len = len(image_folders[0])

                # Only get subfolders in the view directory                # .../views/back
                view_path = os.path.join(data_dir, "..", "views")
                view_subfolders = [ f.path for f in os.scandir(view_path) if f.is_dir() ]
                view_subfolders = sorted(view_subfolders)

                if optimize_seq_spherical_frames is True:
                    # Optimize the views
                    # Reverse order for particular views, add supplementary images between views
                    # Left (1/5)
                    # Front (rev) (4/5)
                    # Right (3/5)
                    # Back (rev) (2/5)
                    # Up
                    # Down (rev)
                    # Index view folders, adding connective images
                    view_images = [ f.path for f in os.scandir(view_subfolders[0]) if f.is_file() ]
                    view_images = sorted(view_images)
                    file_count = len(view_images)
                    first_filename = view_images[0]
                    last_filename = view_images[file_count-1]
                    middle_filename = view_images[math.ceil(len(view_images)//2)]
                    log.info(f"Middle filename: {middle_filename}")

                    head_first_fn, tail = os.path.splitext(first_filename)
                    head_last_fn, tail = os.path.splitext(last_filename)
                    head_middle_fn, tail = os.path.splitext(middle_filename)

                    first_view = os.path.basename(head_first_fn)
                    first_view =  f"{int(first_view):0{view_num_len}d}"
                    log.info(f"First view: {first_view}")

                    last_view = os.path.basename(head_last_fn)
                    last_view =  f"{int(last_view):0{view_num_len}d}"
                    log.info(f"Last view: {last_view}")

                    middle_view = os.path.basename(head_middle_fn)
                    middle_view =  f"{int(middle_view):0{view_num_len}d}"
                    log.info(f"Middle view: {middle_view}")

                    # Get frame names for insertion distances
                    def extract_frame_num(filename):
                        base_name = os.path.splitext(filename)[0]
                        if base_name.startswith("frame_"):
                            return base_name.split('_')[1]
                        return base_name

                    frame_10_name = f"{int(extract_frame_num(frame_10_filename)):0{view_num_len}d}"
                    frame_20_name = f"{int(extract_frame_num(frame_20_filename)):0{view_num_len}d}"
                    frame_30_name = f"{int(extract_frame_num(frame_30_filename)):0{view_num_len}d}"
                    frame_40_name = f"{int(extract_frame_num(frame_40_filename)):0{view_num_len}d}"
                    frame_50_name = f"{int(extract_frame_num(frame_50_filename)):0{view_num_len}d}"
                    frame_60_name = f"{int(extract_frame_num(frame_60_filename)):0{view_num_len}d}"
                    frame_70_name = f"{int(extract_frame_num(frame_70_filename)):0{view_num_len}d}"
                    frame_80_name = f"{int(extract_frame_num(frame_80_filename)):0{view_num_len}d}"
                    frame_90_name = f"{int(extract_frame_num(frame_90_filename)):0{view_num_len}d}"

                    # Calculate neighbor frames for oval view nodes
                    def get_neighbor_frames(target_idx, total_frames, view_num_len):
                        prev_idx = max(0, target_idx - 1)
                        next_idx = min(total_frames - 1, target_idx + 1)
                        prev_name = f"{int(extract_frame_num(filenames[prev_idx])):0{view_num_len}d}"
                        next_name = f"{int(extract_frame_num(filenames[next_idx])):0{view_num_len}d}"
                        return [prev_name, next_name]

                    # Store original file count for consistent insertion calculations
                    original_file_count = file_count

                    # Prepare frame names dictionary for process_view function
                    frame_names = {
                        'first': first_view,
                        'last': last_view,
                        'middle': middle_view,
                        'frame_10': frame_10_name,
                        'frame_20': frame_20_name,
                        'frame_30': frame_30_name,
                        'frame_40': frame_40_name,
                        'frame_50': frame_50_name,
                        'frame_60': frame_60_name,
                        'frame_70': frame_70_name,
                        'frame_80': frame_80_name,
                        'frame_90': frame_90_name,
                        'neighbors_10': get_neighbor_frames(int(len(filenames) * 0.1), len(filenames), view_num_len),
                        'neighbors_20': get_neighbor_frames(int(len(filenames) * 0.2), len(filenames), view_num_len),
                        'neighbors_30': get_neighbor_frames(int(len(filenames) * 0.3), len(filenames), view_num_len),
                        'neighbors_40': get_neighbor_frames(int(len(filenames) * 0.4), len(filenames), view_num_len),
                        'neighbors_50': get_neighbor_frames(int(len(filenames) * 0.5), len(filenames), view_num_len),
                        'neighbors_60': get_neighbor_frames(int(len(filenames) * 0.6), len(filenames), view_num_len),
                        'neighbors_70': get_neighbor_frames(int(len(filenames) * 0.7), len(filenames), view_num_len),
                        'neighbors_80': get_neighbor_frames(int(len(filenames) * 0.8), len(filenames), view_num_len),
                        'neighbors_90': get_neighbor_frames(int(len(filenames) * 0.9), len(filenames), view_num_len),
                        'use_oval': use_oval_nodes,
                        'angled_up_views': angled_up_views,
                        'angles_up': angles_up,
                        'angled_down_views': angled_down_views,
                        'angles_down': angles_down
                    }

                    # Insert the new perspective images into the already existing view images
                    for i, view_subfolder in enumerate(view_subfolders):
                        # Remove views that have been configured to be removed
                        if os.path.basename(view_subfolder) in remove_face_list:
                            shutil.rmtree(view_subfolder)
                        else:
                            view = os.path.basename(os.path.normpath(view_subfolder))

                            # Process view using refactored function
                            persp_image_paths = process_view(view, view_subfolder, data_dir, angles_horiz, angles_vert,
                                                           original_file_count, view_num_len, tail, frame_names, log)

                            # Update file count after processing
                            file_count = len([f for f in os.listdir(view_subfolder) if f.endswith(tail)])

                            # Copy connective images over to view folder
                            for persp_image_path in persp_image_paths:
                                if persp_image_path != "":
                                    if os.path.isfile(persp_image_path):
                                        destination_path = os.path.join(view_subfolder, f"{file_count:0{view_num_len}d}{tail}")
                                        log.info(f"Copying {persp_image_path} to {destination_path}")
                                        shutil.copy(persp_image_path, destination_path)
                                        file_count = file_count + 1
                                    else:
                                        log.warning(f"Skipping missing connective image: {persp_image_path}")
                                        # Skip missing files instead of raising error

                            # Apply reverse_file_order to down view after all processing
                            if view == "down":
                                reverse_file_order(view_subfolder)

                # Only get subfolders in the view directory                # .../views/back
                view_path = os.path.join(data_dir, "..", "views")
                view_subfolders = [ f.path for f in os.scandir(view_path) if f.is_dir() ]
                log.info(f"Found {len(view_subfolders)} views in {view_path}")

                # Remove original image folder after moving them to view path
                shutil.rmtree(data_dir)
                os.mkdir(data_dir)

                # VIEW ORDER = Left -> Front (rev) -> Right -> Back (rev) -> Up -> Down (rev)
                # Reorder the view path to coordinate with optimal view pattern
                for view_dir_path in view_subfolders:
                    rest, view = os.path.split(view_dir_path)
                    view_order = ""
                    log.info(f"Processing {view} view")
                    # Map views to image order using if-elif chain
                    view_lower = str(view).lower()
                    if view_lower == "up":
                        if optimize_seq_spherical_frames is True:
                            view_order = "05"
                        else:
                            view_order = "06"
                    elif view_lower == "back":
                        if optimize_seq_spherical_frames is True:
                            view_order = "04"
                        else:
                            view_order = "02"
                    elif view_lower == "down":
                        if optimize_seq_spherical_frames is True:
                            view_order = "06"
                        else:
                            view_order = "05"
                    elif view_lower == "front":
                        if optimize_seq_spherical_frames is True:
                            view_order = "02"
                        else:
                            view_order = "01"
                    elif view_lower == "left":
                        if optimize_seq_spherical_frames is True:
                            view_order = "01"
                        else:
                            view_order = "03"
                    elif view_lower == "right":
                        if optimize_seq_spherical_frames is True:
                            view_order = "03"
                        else:
                            view_order = "04"
                    else:
                        raise RuntimeError(f"Error: {view} is not a valid view.")
                    os.rename(view_dir_path, os.path.join(rest, view_order))

                root_path_parts = re.split(r"[/\\]", data_dir)
                root_path_parts.pop()
                if root_path_parts[0] == '':
                    view_path = os.path.join("/", *root_path_parts, "views")
                    img_path = os.path.join("/", *root_path_parts, "images")
                else:
                    view_path = os.path.join(*root_path_parts, "views")
                    img_path = os.path.join(*root_path_parts, "images")
                view_dirs = os.listdir(view_path)
                view_dirs = sorted(view_dirs)
                log.info(f"Found {len(view_dirs)} views in {view_path}")

                # Move the images from the "view" directory to the "images" directory
                # Use a global counter to ensure each image gets a unique sequential number
                global_img_counter = 1
                
                # First, collect all file operations
                move_operations = []
                for view_dir in view_dirs:
                    view_dir_path = os.path.join(view_path, view_dir)
                    img_filenames = os.listdir(view_dir_path)
                    img_filenames = sorted(img_filenames)
                    log.info(f"Found {len(img_filenames)} images in {view_dir_path}")
                    
                    for img_filename in img_filenames:
                        input_img_filename_path = os.path.join(view_dir_path, img_filename)
                        head, extension = os.path.splitext(input_img_filename_path)
                        output_img_filename_path = os.path.join(img_path, f"{global_img_counter:05d}{extension}")
                        move_operations.append((input_img_filename_path, output_img_filename_path))
                        global_img_counter += 1
                
                # Create images directory if it doesn't exist
                os.makedirs(img_path, exist_ok=True)
                
                # Process moves in batches
                log.info(f"Moving {len(move_operations)} images to final sequence")
                batch_size = 100
                for i in range(0, len(move_operations), batch_size):
                    batch = move_operations[i:i+batch_size]
                    
                    # Process files sequentially to avoid multiprocessing issues
                    success_count = 0
                    for src, dst in batch:
                        try:
                            shutil.move(src, dst)
                            success_count += 1
                        except Exception as e:
                            log.warning(f"Error moving {src} to {dst}: {str(e)}")
                    
                    log.info(f"Moved batch of {success_count} files")
                
                # Remove view directory after all files are moved
                shutil.rmtree(view_path)
            else:
                log.warning(f"No supported images present in {data_dir}.")
        else:
            log.error("Input directory is not valid")
    except Exception as e:
        raise RuntimeError("Error running spherical to perspective transformation. {e}") from e
