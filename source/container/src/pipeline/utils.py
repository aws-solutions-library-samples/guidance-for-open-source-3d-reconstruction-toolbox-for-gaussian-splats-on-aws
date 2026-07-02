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
This file serves as the utilities that are used in the container image for
creating splats using NerfStudio, GSplat, Colmap, etc
"""

import os
import shutil
import tarfile
import trimesh
import cv2
import numpy as np
from tqdm import tqdm as tqdm_func
from PIL import Image
import glob
import torch
import boto3
from botocore.exceptions import ClientError
from decimal import Decimal

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
        files = [
            f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))
        ]
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
def rotate_single_image(image_path_angle):
    image_path, angle = image_path_angle
    try:
        img = cv2.imread(image_path)
        if img is None:
            return f"Failed to read: {image_path}"

        # Get image dimensions
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new dimensions after rotation to avoid cropping
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int((h * sin_val) + (w * cos_val))
        new_height = int((h * cos_val) + (w * sin_val))
        
        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Rotate image with new dimensions
        rotated_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height))

        # Save with same filename
        success = cv2.imwrite(image_path, rotated_img)
        if success:
            return f"Rotated: {os.path.basename(image_path)}"
        else:
            return f"Failed to save: {os.path.basename(image_path)}"
    except Exception as e:
        return f"Error rotating {os.path.basename(image_path)}: {str(e)}"

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
        result = rotate_single_image((path, angle))
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

        # Process sequentially and track results
        success_count = 0
        for image_path in image_files:
            result = rotate_single_image((image_path, angle))
            if "Rotated:" in result:
                success_count += 1
            else:
                print(f"Failed to rotate: {image_path}")
        
        print(f"Successfully rotated {success_count}/{len(image_files)} images")
    else:
        print(f"Error: {path} is not a valid file or directory")

def resize_to_4k(image_path, spherical_camera=False):
    """
    Resize image to 4K if the largest dimension exceeds 4K threshold.
    Uses the largest side (width or height) to determine if resizing is needed.
    
    Args:
        image_path: string path to the image
        spherical_camera: boolean indicating if this is a spherical camera (skip resizing if True)
        
    Returns:
        numpy array: resized image if largest dimension exceeded 4K threshold, original image otherwise
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Get current dimensions
    height, width = image.shape[:2]
    
    # Find the largest dimension
    max_dimension = max(width, height)
    
    # 4K threshold - use 3840 as the maximum allowed dimension
    max_4k_dimension = 3840
    
    print(f"Image {os.path.basename(image_path)}: {width}x{height}, max={max_dimension}, threshold={max_4k_dimension}")
    
    # Skip resizing for spherical cameras or if dimension is at or below threshold
    if spherical_camera:
        print(f"Skipping resize for spherical camera: {image_path}")
        return image
    
    # Only resize if the largest dimension exceeds the 4K threshold
    if max_dimension > max_4k_dimension:
        print(f"Resizing image to 4K: {image_path}")
        # Calculate scale factor to bring largest dimension to 4K
        scale_factor = max_4k_dimension / max_dimension
        
        # Calculate new dimensions maintaining aspect ratio
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize using INTER_AREA for downscaling (better quality)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Save the resized image back to the same path
        cv2.imwrite(image_path, resized)
        return resized
    
    # Return original image if no resize needed
    return image

def resize_images_to_common_dimensions(image_dir):
    """
    Resize all images in a directory to the smallest common dimensions
    found among the images. Images with a different aspect ratio are
    center-cropped to the target aspect ratio first, then scaled to
    the exact target dimensions. This ensures every image ends up at
    identical width x height without distortion.

    Args:
        image_dir (str): Path to directory containing images

    Returns:
        tuple: (target_width, target_height) used, or None if no images found
    """
    extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(extensions) and os.path.isfile(os.path.join(image_dir, f))]

    if not image_files:
        print(f"No images found in {image_dir}")
        return None

    # Collect all unique dimensions
    dimensions = set()
    for f in image_files:
        img = cv2.imread(os.path.join(image_dir, f))
        if img is not None:
            h, w = img.shape[:2]
            dimensions.add((w, h))

    if not dimensions:
        print("Could not read any images")
        return None

    if len(dimensions) == 1:
        w, h = dimensions.pop()
        print(f"All {len(image_files)} images already have uniform dimensions: {w}x{h}")
        return (w, h)

    target_w = min(d[0] for d in dimensions)
    target_h = min(d[1] for d in dimensions)
    print(f"Found {len(dimensions)} unique sizes across {len(image_files)} images. "
          f"Target dimensions: {target_w}x{target_h}")

    resized_count = 0
    for f in image_files:
        filepath = os.path.join(image_dir, f)
        img = cv2.imread(filepath)
        if img is None:
            continue
        h, w = img.shape[:2]
        if w == target_w and h == target_h:
            continue
        # Center-crop to target aspect ratio, then resize
        target_ratio = target_w / target_h
        img_ratio = w / h
        if img_ratio > target_ratio:
            # Image is wider — crop width
            crop_w = int(h * target_ratio)
            x_start = (w - crop_w) // 2
            img = img[:, x_start:x_start + crop_w]
        elif img_ratio < target_ratio:
            # Image is taller — crop height
            crop_h = int(w / target_ratio)
            y_start = (h - crop_h) // 2
            img = img[y_start:y_start + crop_h, :]
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(filepath, img)
        resized_count += 1

    print(f"Resized {resized_count}/{len(image_files)} images to {target_w}x{target_h}")
    return (target_w, target_h)

def read_camera_params_from_file(cameras_txt_path):
    """Read camera parameters from cameras.txt file"""
    try:
        with open(cameras_txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
                
            # Parse camera line
            # Format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            parts = line.split()
            if len(parts) >= 5:
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                
                # Get the parameters - they might be comma-separated or space-separated
                params_str = ' '.join(parts[4:])

                # First, normalize the input by replacing commas with spaces
                normalized_params = params_str.replace(',', ' ')

                # Split by whitespace to get individual parameters
                param_list = normalized_params.split()

                # Join the parameters with commas to create the final comma-separated list
                comma_separated = ','.join(param_list)

                # Return the first camera entry
                return {
                    'id': camera_id,
                    'model': model,
                    'width': width,
                    'height': height,
                    'params_str': comma_separated
                }
        
        return None
    except Exception as e:
        print(f"Error Code 700: error reading camera parameters from file: {str(e)}")
        return None
        
def validate_input_media(filename: str)->bool:
    """
    # Validation Check if single images or video is input
    """
    ext = str(os.path.splitext(filename)[1]).lower()
    if ext == ".mp4" or ext == ".mov":
        return True
    elif ext == ".zip" or ext == ".gz":
        return False
    else:
        return False

# PLY utility functions for post-processing
def load_ply(ply_path):
    """
    Load PLY file and extract Gaussian data.
    
    Args:
        ply_path: Path to the PLY file
        
    Returns:
        Dictionary containing positions, vertices, plydata, and property names
    """
    import plyfile
    plydata = plyfile.PlyData.read(ply_path)
    vertices = plydata['vertex']
    
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    return {
        'positions': positions,
        'vertices': vertices,
        'plydata': plydata,
        'property_names': [prop.name for prop in vertices.properties]
    }

def save_ply(gaussian_data, mask, output_path):
    """
    Save filtered PLY file with points selected by mask.
    
    Args:
        gaussian_data: Dictionary from load_ply()
        mask: Boolean array indicating which points to keep
        output_path: Path to save the PLY file
    """
    import plyfile
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    original_vertices = gaussian_data['vertices']
    filtered_count = np.sum(mask)
    
    # Create new vertex array with filtered data
    properties = [(prop.name, 'f4') for prop in original_vertices.properties]
    new_vertices = np.zeros(filtered_count, dtype=properties)
    
    # Copy filtered data for each property
    for prop_name in gaussian_data['property_names']:
        new_vertices[prop_name] = original_vertices[prop_name][mask]
    
    # Create PLY element and save
    vertex_element = plyfile.PlyElement.describe(new_vertices, 'vertex')
    ply_data = plyfile.PlyData([vertex_element], comments=gaussian_data['plydata'].comments)
    ply_data.write(output_path)

def filter_points_in_bounds(gaussian_data, bounds):
    """
    Filter points that fall within the specified bounds.
    
    Args:
        gaussian_data: Dictionary from load_ply()
        bounds: Dictionary with x_min, x_max, y_min, y_max, z_min, z_max
        
    Returns:
        Boolean mask array
    """
    positions = gaussian_data['positions']
    
    mask = (
        (positions[:, 0] >= bounds['x_min']) & (positions[:, 0] <= bounds['x_max']) &
        (positions[:, 1] >= bounds['y_min']) & (positions[:, 1] <= bounds['y_max']) &
        (positions[:, 2] >= bounds['z_min']) & (positions[:, 2] <= bounds['z_max'])
    )
    
    return mask

def calculate_bounds_percentile(positions, percentile=90):
    """Calculate bounds using percentile method."""
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    max_distance = np.percentile(distances, percentile)
    
    return {
        'x_min': center[0] - max_distance,
        'x_max': center[0] + max_distance,
        'y_min': center[1] - max_distance,
        'y_max': center[1] + max_distance,
        'z_min': center[2] - max_distance,
        'z_max': center[2] + max_distance
    }, center

def calculate_bounds_std(positions, std_multiplier=2.0):
    """Calculate bounds using standard deviation method."""
    # Use median for Y-axis (vertical) to better handle ground plane noise
    center = np.array([
        np.mean(positions[:, 0]),  # X: use mean
        np.median(positions[:, 1]),  # Y: use median for better vertical centering
        np.mean(positions[:, 2])   # Z: use mean
    ])
    std_devs = np.std(positions, axis=0)
    half_size = std_devs * std_multiplier
    
    return {
        'x_min': center[0] - half_size[0],
        'x_max': center[0] + half_size[0],
        'y_min': center[1] - half_size[1],
        'y_max': center[1] + half_size[1],
        'z_min': center[2] - half_size[2],
        'z_max': center[2] + half_size[2]
    }, center

def calculate_bounds_fixed(positions, cube_size):
    """Calculate bounds using fixed cube size method."""
    center = np.mean(positions, axis=0)
    half_size = cube_size / 2
    
    return {
        'x_min': center[0] - half_size,
        'x_max': center[0] + half_size,
        'y_min': center[1] - half_size,
        'y_max': center[1] + half_size,
        'z_min': center[2] - half_size,
        'z_max': center[2] + half_size
    }, center

def print_filter_stats(original_count, filtered_count, bounds=None, center=None):
    """Print statistics about filtering operation."""
    print(f"Original points: {original_count:,}")
    print(f"Filtered points: {filtered_count:,}")
    print(f"Reduction: {(1 - filtered_count/original_count)*100:.1f}%")
    
    if bounds and center is not None:
        print(f"\nCube center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        print(f"Cube bounds:")
        print(f"  X: {bounds['x_min']:.3f} to {bounds['x_max']:.3f}")
        print(f"  Y: {bounds['y_min']:.3f} to {bounds['y_max']:.3f}")
        print(f"  Z: {bounds['z_min']:.3f} to {bounds['z_max']:.3f}")

def validate_input_media(filename: str)->bool:
    """
    # Validation Check if single images or video is input
    """
    ext = str(os.path.splitext(filename)[1]).lower()
    if ext == ".mp4" or ext == ".mov":
        return True
    elif ext == ".zip" or ext == ".gz":
        return False
    else:
        return False

def load_config(config_names: list, config_values: list)->dict:
    """
    # Load configuration from environment variables into a dict
    # Handles both SageMaker and AWS Batch environments
    """
    # Keys that should not be converted to lowercase
    preserve_case_keys = {'DATASET_PATH', 'CODE_PATH', 'UUID', 'S3_INPUT', 'S3_OUTPUT', 'FILENAME', 'OBJECT_REMOVAL_OBJECTS', 'INSTANCE_TYPE', 'TASK_TOKEN'}
    
    # Detect environment and set appropriate paths
    is_batch = 'AWS_BATCH_JOB_ID' in os.environ
    is_sagemaker = os.path.exists('/opt/ml')
    
    # Set environment-specific paths
    if is_batch:
        # AWS Batch environment
        os.environ.setdefault('SM_MODEL_DIR', '/tmp/model')
        os.environ.setdefault('SM_CHANNEL_TRAIN', '/tmp/input/train')
        os.environ.setdefault('SM_CHANNEL_MODEL', '/tmp/input/model')
        os.environ.setdefault('SM_OUTPUT_DATA_DIR', '/tmp/output')
    elif is_sagemaker:
        # SageMaker environment (default paths already set)
        pass
    else:
        # Local development environment
        os.environ.setdefault('SM_MODEL_DIR', '/opt/ml/model')
        os.environ.setdefault('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')
        os.environ.setdefault('SM_CHANNEL_MODEL', '/opt/ml/input/data/model')
        os.environ.setdefault('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
    
    for i, config_name in enumerate(config_names):
        if config_name in os.environ:
            if config_name in preserve_case_keys:
                config_values[i] = str(os.environ[config_name])
            else:
                config_values[i] = str(os.environ[config_name]).lower().strip()

    conf = dict(zip(config_names, config_values))
    
    # Ensure all non-preserve-case values are lowercase
    for key, value in conf.items():
        if key not in preserve_case_keys:
            conf[key] = str(value).lower().strip()
    
    # Debug: Print all config key-value pairs, masking sensitive tokens
    print("=== CONFIG DEBUG OUTPUT ===")
    for key, value in conf.items():
        if key == 'TASK_TOKEN' and value:
            print(f"{key}: '[REDACTED]' (type: {type(value).__name__})")
        else:
            print(f"{key}: '{value}' (type: {type(value).__name__})")
    print("=== END CONFIG DEBUG ===")
    
    return conf

def obj_to_glb(obj_path: str, glb_path: str)->None:
    """
        Export the obj and material as a .glb file
    """
    mesh = trimesh.load(
        obj_path,
        file_type='obj',
        process=False,
        force='mesh',
        skip_texture=False,
        split_object=False,
        group_material=False
    )
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh = mesh.apply_transform(rot)
    # Export directly without trying to load again
    mesh.export(glb_path, file_type='glb')

def count_up_to(n):
    """
        Convert an integer to a list of numbers as string
    """
    return ','.join(str(i) for i in range(n))

def untar_gz(file_path, extract_path='.'):
    """
    Extracts a .tar.gz file.

    Args:
        file_path (str): The path to the .tar.gz file.
        extract_path (str, optional): The directory to extract to. Defaults to current directory.
    """
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            # Validate each member before extraction
            for member in tar.getmembers():
                # Check for directory traversal attempts
                if os.path.isabs(member.name) or ".." in member.name:
                    print(f"Warning: Skipping potentially dangerous path: {member.name}")
                    continue
                # Check for symlinks pointing outside extraction directory
                if member.issym() or member.islnk():
                    if os.path.isabs(member.linkname) or ".." in member.linkname:
                        print(f"Warning: Skipping potentially dangerous link: {member.name} -> {member.linkname}")
                        continue
            # Extract only safe members - validated to prevent directory traversal
            safe_members = [m for m in tar.getmembers() 
                          if not (os.path.isabs(m.name) or ".." in m.name or 
                                 ((m.issym() or m.islnk()) and (os.path.isabs(m.linkname) or ".." in m.linkname)))]
            tar.extractall(extract_path, members=safe_members)  # nosemgrep: dangerous-tarfile-extractall
            # Members are validated above to prevent directory traversal attacks
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found: '{file_path}'") from e
    except tarfile.ReadError as e:
        raise tarfile.ReadError(f"Could not open '{file_path}' with read mode 'r:gz'. File may be corrupted or not a valid tar.gz archive.") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while extracting '{file_path}': {e}") from e

def has_alpha_channel(image_path):
    """
    Check if an image has an alpha channel.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        bool: True if the image has an alpha channel, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            return img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)
    except Exception as e:
        print(f"Error checking alpha channel in {image_path}: {str(e)}")
        return False

def process_images(input_dir, output_dir=None, preserve_alpha=False):
    """
    Process RGBA images by:
    1. Converting them to RGB (unless preserve_alpha=True)
    2. Creating mask files from the alpha channel
    
    Args:
        input_dir: Directory containing RGBA images
        output_dir: Directory to save processed images (if None, will use input_dir)
        preserve_alpha: If True, keep RGBA format instead of converting to RGB
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm_func(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, image_file)
        
        # Open the image
        img = Image.open(input_path)
        
        # Check if the image has an alpha channel
        if img.mode == 'RGBA':
            # Extract alpha channel for mask
            alpha = img.split()[3]
            
            if preserve_alpha:
                # Keep RGBA format - save as PNG to support transparency
                base_name = os.path.splitext(image_file)[0]
                output_rgba_path = os.path.join(input_dir, f"{base_name}.png")
                img.save(output_rgba_path, 'PNG')
                # Remove original if it was JPEG
                if image_file.lower().endswith(('.jpg', '.jpeg')) and output_rgba_path != input_path:
                    os.remove(input_path)
            else:
                # Convert to RGB
                rgb = img.convert('RGB')
                output_rgb_path = os.path.join(input_dir, image_file)
                rgb.save(output_rgb_path)
            
            # Save alpha as mask
            # COLMAP expects mask filename to be original filename + .png extension
            mask_file = f"{image_file}.png"
            output_mask_path = os.path.join(output_dir, mask_file)
            alpha.save(output_mask_path)

def select_largest_colmap_model(sparse_path):
    """
    Select the largest COLMAP sparse model and move it to sparse/0.
    If only one model exists (sparse/0), skip the function.
    
    Args:
        sparse_path: Path to the sparse directory containing numbered model folders
    """
    if not os.path.exists(sparse_path):
        return
    
    # Get all numbered directories
    model_dirs = [d for d in os.listdir(sparse_path) 
                  if os.path.isdir(os.path.join(sparse_path, d)) and d.isdigit()]
    
    # If only one model or no models, skip
    if len(model_dirs) <= 1:
        return
    
    # Find the largest model by counting points in points3D.txt
    largest_model = None
    max_points = 0
    
    for model_dir in model_dirs:
        points_file = os.path.join(sparse_path, model_dir, "points3D.txt")
        if os.path.exists(points_file):
            point_count = 0
            with open(points_file, 'r') as f:
                for line in f:
                    if not line.startswith('#') and line.strip():
                        point_count += 1
            
            if point_count > max_points:
                max_points = point_count
                largest_model = model_dir
    
    # If largest model is already at 0, we're done
    if largest_model == "0" or largest_model is None:
        return
    
    # Create temporary directory for shuffling
    temp_dir = os.path.join(sparse_path, "temp_shuffle")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Move current 0 to temp if it exists
        zero_path = os.path.join(sparse_path, "0")
        if os.path.exists(zero_path):
            shutil.move(zero_path, os.path.join(temp_dir, "old_0"))
        
        # Move largest model to 0
        largest_path = os.path.join(sparse_path, largest_model)
        shutil.move(largest_path, zero_path)
        
        # Move old 0 to largest model's position if it existed
        old_zero_temp = os.path.join(temp_dir, "old_0")
        if os.path.exists(old_zero_temp):
            shutil.move(old_zero_temp, largest_path)
        
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def create_tarball(source_path, output_path, arcname=None):
    """
    Create a compressed tarball archive from a file or directory.
    
    Args:
        source_path (str): Path to the file or directory to archive
        output_path (str): Path where the archive should be created
        arcname (str): Name to use for the archive entry (defaults to basename of source_path)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if arcname is None:
        arcname = os.path.basename(source_path)
    with tarfile.open(output_path, "w:gz", dereference=True) as tar:
        tar.add(source_path, arcname=arcname, recursive=True)

def remove_cubemap_faces(erp_images_dir, remove_face_list):
    """
    Remove unwanted cubemap faces from equirectangular images and overwrite originals.
    
    Args:
        erp_images_dir (str): Directory containing equirectangular images
        remove_face_list (list): List of face names to remove (e.g., ['back', 'down'])
    """
    import torch
    from imageio.v2 import imread, imwrite
    import sys
    import os
    # Add spherical directory to path for Equirec2Cube import
    spherical_path = os.path.join(os.path.dirname(__file__), 'spherical')
    if spherical_path not in sys.path:
        sys.path.insert(0, spherical_path)
    import Equirec2Cube
    from PIL import Image
    import py360convert
    
    if len(remove_face_list) == 0 or remove_face_list[0] == '':
        return
        
    for filename in os.listdir(erp_images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(erp_images_dir, filename)
            img = imread(img_path, pilmode='RGBA')
            dims = img.shape
            
            # Convert ERP to cubemap
            e2c = Equirec2Cube.Equirec2Cube(dims[0], dims[1], int(float(dims[0])/2), CUDA=False)
            batch = torch.FloatTensor(img.astype(float)/255).permute(2, 0, 1)[None, ...]
            cubemap_tensor = e2c(batch)
            cubemap = cubemap_tensor.permute(0, 2, 3, 1).cpu().numpy()
            
            # Remove unwanted faces
            for i, face_name in enumerate(['right', 'down', 'left', 'back', 'front', 'up']):
                if face_name in remove_face_list:
                    print(f"Removing face: {face_name} for file {img_path}")
                    img_height, img_width = int(float(dims[0])/2), int(float(dims[0])/2)
                    # Fill with neutral gray instead of zeros to prevent masking issues
                    neutral_color = np.full((img_height, img_width, 4), [0.5, 0.5, 0.5, 1.0], dtype=np.float32)
                    cubemap[i] = neutral_color
            
            # Convert back to ERP
            cube_dice = [cubemap[i] for i in [2, 4, 0, 3, 5, 1]]  # reorder faces
            erp_img = py360convert.c2e(cubemap=cube_dice, h=dims[0], w=dims[1], cube_format='list')
            
            # Overwrite original
            Image.fromarray(erp_img.astype(np.uint8)).save(img_path)

# Clean up dataset before moving to output location
def cleanup_dataset(dataset_path):
    """Remove empty directories and unwanted files from dataset"""
    # Remove stable-diffusion-xl-base-1.0 directory if it exists
    sd_dir = os.path.join(dataset_path, "stable-diffusion-xl-base-1.0")
    if os.path.exists(sd_dir):
        shutil.rmtree(sd_dir)
        print(f"Removed stable-diffusion-xl-base-1.0 directory")
    
    # Remove empty directories
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):  # Directory is empty
                    os.rmdir(dir_path)
                    print(f"Removed empty directory: {dir_path}")
            except OSError:
                pass  # Directory not empty or permission issue

def extract_images_from_zip_temp(temp_path, image_path, dataset_path, log):
    """
    Walk temp_path, routing image files to image_path and mask images
    (those inside a folder named 'mask' or 'masks') to dataset_path/masks/.
    Sub-folders are flattened; non-image files are silently ignored.
    Only intended for run_recon == true jobs.
    """
    IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
    os.makedirs(image_path, exist_ok=True)
    masks_dest = os.path.join(dataset_path, 'masks')
    seen_names = {}
    for root, dirs, files in os.walk(temp_path):
        rel_root = os.path.relpath(root, temp_path)
        path_parts = rel_root.replace('\\', '/').split('/')
        is_mask_folder = any(p.lower() in ('mask', 'masks') for p in path_parts)
        for fname in files:
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            src = os.path.join(root, fname)
            if is_mask_folder:
                os.makedirs(masks_dest, exist_ok=True)
                dst = os.path.join(masks_dest, fname)
            else:
                dst = os.path.join(image_path, fname)
            if dst in seen_names:
                base, ext = os.path.splitext(fname)
                dst = os.path.join(os.path.dirname(dst), f"{base}_{seen_names[dst]}{ext}")
            seen_names[dst] = seen_names.get(dst, 0) + 1
            shutil.move(src, dst)
    log.info(
        f"Extracted {len(os.listdir(image_path))} images"
        + (f" and {len(os.listdir(masks_dest))} masks" if os.path.isdir(masks_dest) else "")
        + " from zip (sub-folders flattened, non-image files ignored)"
    )


def validate_and_resize_images(image_path, config, log, pipeline):
    """
    Validate image files and resize them if needed.
    
    Args:
        image_path: Path to directory containing images
        config: Configuration dictionary
        log: Logger instance
        pipeline: Pipeline instance for error reporting
    """
    IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
    all_files = os.listdir(image_path)
    filenames = [f for f in all_files if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    if filenames:
        log.info(f"Found {len(filenames)} images in archive (skipped {len(all_files) - len(filenames)} non-image files).")
        # Check first image to determine if resizing is needed
        first_filepath = os.path.join(image_path, filenames[0])
        first_image = cv2.imread(first_filepath)
        if first_image is not None:
            height, width = first_image.shape[:2]
            max_dimension = max(width, height)
            needs_resize = max_dimension > 3840 and config['SPHERICAL_CAMERA'] != "true"
            if needs_resize:
                log.info(f"Images need resizing (first image: {width}x{height}). Processing all {len(filenames)} images...")
                for filename in filenames:
                    filepath = os.path.join(image_path, filename)
                    resize_to_4k(filepath, config['SPHERICAL_CAMERA'] == "true")
            else:
                log.info(f"Images do not need resizing (first image: {width}x{height}). Skipping resize for all {len(filenames)} images.")
        else:
            log.warning("Could not read first image for dimension check, processing all images individually")
            for filename in filenames:
                filepath = os.path.join(image_path, filename)
                resize_to_4k(filepath, config['SPHERICAL_CAMERA'] == "true")
    else:
        log.info("No image files found in image directory - this is expected for resume training")

# Clean up GPU memory
def cleanup_cuda_memory():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            # If synchronize fails, just do cache cleanup
            torch.cuda.empty_cache()
        import gc
        gc.collect()

def setup_local_debug(config, log):
    """Setup local debug mode paths and directories"""
    LOCAL_DEBUG = os.environ.get('LOCAL_DEBUG', config.get('LOCAL_DEBUG', 'false')).lower() == 'true'
    
    if LOCAL_DEBUG:
        if log:
            log.info("LOCAL_DEBUG mode enabled - using local filesystem instead of S3")
        else:
            print("LOCAL_DEBUG mode enabled - using local filesystem instead of S3")
        
        LOCAL_MOUNT = os.environ.get('LOCAL_MOUNT', '/mnt/data')
        
        config['DATASET_PATH'] = os.path.join(LOCAL_MOUNT, 'workflow-input')
        config['S3_INPUT'] = os.path.join(LOCAL_MOUNT, 'media-input')
        config['S3_OUTPUT'] = os.path.join(LOCAL_MOUNT, 'workflow-output')
        
        if 'CODE_PATH' not in config or not config['CODE_PATH']:
            config['CODE_PATH'] = '/opt/ml/code'
        
        os.environ['MODEL_PATH'] = os.path.join(LOCAL_MOUNT, 'models')
        
        os.makedirs(config['DATASET_PATH'], exist_ok=True)
        os.makedirs(config['S3_INPUT'], exist_ok=True)
        os.makedirs(config['S3_OUTPUT'], exist_ok=True)
        os.makedirs(os.environ['MODEL_PATH'], exist_ok=True)
        
        if log:
            log.info(f"Local paths: DATASET={config['DATASET_PATH']}, OUTPUT={config['S3_OUTPUT']}, CODE={config['CODE_PATH']}")
        else:
            print(f"Local paths: DATASET={config['DATASET_PATH']}, OUTPUT={config['S3_OUTPUT']}, CODE={config['CODE_PATH']}")
    
    return LOCAL_DEBUG

def copy_to_local_output(source_path, config, filename, log):
    """Copy file to local output directory"""
    local_output = os.path.join(config['S3_OUTPUT'], config['UUID'])
    os.makedirs(local_output, exist_ok=True)
    dest_path = os.path.join(local_output, filename)
    if os.path.exists(source_path):
        shutil.copy2(source_path, dest_path)
        log.info(f"Copied {filename} to local output: {local_output}")
        return True
    return False

def print_container_version_info():
    """Print container version information"""
    import sys
    import subprocess
    import torch
    import torchvision
    import pycolmap
    
    print("=== CONTAINER VERSION INFORMATION ===")
    print(f"  Python: {sys.version.split()[0]}")
    
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0 and 'release' in result.stdout:
            cuda_version = result.stdout.split('release ')[1].split(',')[0]
            print(f"  CUDA: {cuda_version}")
        else:
            print("  CUDA: Not found")
    except Exception as e:
        print(f"  CUDA: Not found ({e})")
    
    try:
        print(f"  PyTorch: {torch.__version__}")
    except Exception as e:
        print(f"  PyTorch: Not found ({e})")
    
    try:
        print(f"  TorchVision: {torchvision.__version__}")
    except Exception as e:
        print(f"  TorchVision: Not found ({e})")
    
    try:
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU: {torch.cuda.get_device_name()} ({vram:.2f}GB VRAM)")
    except Exception as e:
        print(f"  GPU: Not available ({e})")
    
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            mem_total = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1]) / 1024**2
            print(f"  System RAM: {mem_total:.2f}GB")
    except Exception as e:
        print(f"  System RAM: Not available ({e})")
    
    try:
        shm_result = subprocess.check_output(['df', '-h', '/dev/shm']).decode().split('\n')[1].split()
        print(f"  Shared memory (/dev/shm): Size={shm_result[1]}, Used={shm_result[2]}, Available={shm_result[3]}")
    except Exception as e:
        print(f"  Shared memory: Not available ({e})")
    
    print(f"  Execution mode: {'AWS Batch' if 'AWS_BATCH_JOB_ID' in os.environ else 'SageMaker'}")
    
    if 'AWS_BATCH_JOB_ID' in os.environ:
        print(f"  Batch Job ID: {os.environ.get('AWS_BATCH_JOB_ID')}")
        print(f"  Batch Job Queue: {os.environ.get('AWS_BATCH_JQ_NAME', 'N/A')}")
        try:
            ulimit_result = subprocess.check_output(['/bin/bash', '-c', 'ulimit -a']).decode()
            print(f"  Resource limits:\n{ulimit_result}")
        except Exception as e:
            print(f"  Resource limits: Not available ({e})")
    
    try:
        result = subprocess.run(['colmap', '-h'], capture_output=True, text=True)
        if result.returncode == 0 and 'COLMAP' in result.stdout:
            colmap_version = result.stdout.split('\n')[0].split()[1]
            print(f"  COLMAP: {colmap_version}")
        else:
            print("  COLMAP: Not found")
    except Exception as e:
        print(f"  COLMAP: Not found ({e})")
    
    try:
        print(f"  pycolmap: {pycolmap.__version__}")
    except Exception as e:
        print(f"  pycolmap: Not available ({e})")
    
    print("=== END VERSION INFORMATION ===")
    print()

def send_task_heartbeat(task_token: str, log=None) -> None:
    # Send a heartbeat to Step Functions to keep the waitForTaskToken state alive during
    # long-running components. Must be called more frequently than HeartbeatSeconds (172800s).
    # Swallows InvalidToken and TaskTimedOut silently — if the execution is gone, the
    # container should finish naturally and the final SendTaskSuccess/Failure will also fail.
    try:
        region = os.environ.get('AWS_DEFAULT_REGION', os.environ.get('AWS_REGION', 'us-east-1'))
        sfn_client = boto3.client('stepfunctions', region_name=region)
        sfn_client.send_task_heartbeat(taskToken=task_token)
        if log:
            log.debug("Step Functions heartbeat sent")
    except Exception:
        pass


def send_task_success(task_token: str, output: dict, log=None) -> None:
    # Notify Step Functions that the Batch job completed successfully using the waitForTaskToken callback.
    # Called at the end of the pipeline when IS_BATCH is True and a task token was provided.
    # InvalidToken and AccessDeniedException are swallowed — the pipeline completed successfully
    # regardless of whether the callback reaches Step Functions.
    try:
        import json
        region = os.environ.get('AWS_DEFAULT_REGION', os.environ.get('AWS_REGION', 'us-east-1'))
        sfn_client = boto3.client('stepfunctions', region_name=region)
        sfn_client.send_task_success(
            taskToken=task_token,
            output=json.dumps(output)
        )
        if log:
            log.info("Step Functions task success callback sent")
    except Exception as e:
        # Swallow all callback errors — the pipeline completed successfully.
        # Common causes: InvalidToken (execution timed out/aborted),
        # AccessDeniedException (IAM not yet deployed), network errors.
        # These must not cause the container to exit with an error.
        if log:
            log.warning(f"Step Functions task success callback failed (pipeline still succeeded): {e}")

def send_task_failure(task_token: str, error: str, cause: str, log=None) -> None:
    # Notify Step Functions that the Batch job failed using the waitForTaskToken callback.
    # Called in the pipeline exception handler when IS_BATCH is True and a task token was provided.
    # InvalidToken is swallowed — it means Step Functions already timed out or moved on.
    try:
        region = os.environ.get('AWS_DEFAULT_REGION', os.environ.get('AWS_REGION', 'us-east-1'))
        sfn_client = boto3.client('stepfunctions', region_name=region)
        sfn_client.send_task_failure(
            taskToken=task_token,
            error=error[:256],
            cause=cause[:32768]
        )
        if log:
            log.info("Step Functions task failure callback sent")
    except Exception as e:
        if log:
            log.warning(f"Failed to send Step Functions task failure: {e}")

def update_dynamodb_metrics(uuid, table_name, comp_group_elapsed_time=None, metrics=None, log=None):
    """
    Update DynamoDB with component timing and training metrics.
    
    Args:
        uuid: Job UUID
        table_name: DynamoDB table name
        comp_group_elapsed_time: List of elapsed times for each component group [pre_processing, reconstruction, training, post_processing]
        metrics: Dictionary containing training metrics (psnr, ssim, lpips)
        log: Logger instance
    """
    if os.environ.get('LOCAL_DEBUG', '').lower() == 'true':
        if log:
            log.info("LOCAL_DEBUG mode - skipping DynamoDB update")
        return
    
    try:
        region = os.environ.get('AWS_DEFAULT_REGION', os.environ.get('AWS_REGION', 'us-east-1'))
        dynamodb = boto3.resource('dynamodb', region_name=region)
        table = dynamodb.Table(table_name)
        
        update_parts = []
        expression_values = {}
        
        if comp_group_elapsed_time:
            update_parts.append('componentGroupElapsedTime = :times')
            expression_values[':times'] = comp_group_elapsed_time
        
        if metrics:
            # Convert float values to Decimal for DynamoDB
            decimal_metrics = {k: Decimal(str(v)) for k, v in metrics.items()}
            update_parts.append('evaluationMetrics = :metrics')
            expression_values[':metrics'] = decimal_metrics
        
        if update_parts:
            table.update_item(
                Key={os.environ.get('DDB_KEY_NAME', 'uuid'): uuid},
                UpdateExpression='SET ' + ', '.join(update_parts),
                ExpressionAttributeValues=expression_values
            )
            if log:
                log.info(f"Updated DynamoDB with timing/metrics for job {uuid}")
    except ClientError as e:
        if log:
            log.warning(f"Failed to update DynamoDB: {e}")
    except Exception as e:
        if log:
            log.warning(f"Error updating DynamoDB: {e}")

def update_component_phase_completion(uuid, table_name, phase_name, elapsed_time, log=None):
    """
    Update DynamoDB when a component phase completes.
    
    Args:
        uuid: Job UUID
        table_name: DynamoDB table name
        phase_name: Name of the phase (PRE_PROCESSING, RECONSTRUCTION, TRAINING, POST_PROCESSING)
        elapsed_time: Time in seconds for this phase
        log: Logger instance
    """
    if os.environ.get('LOCAL_DEBUG', '').lower() == 'true':
        if log:
            log.info(f"LOCAL_DEBUG mode - skipping phase completion update for {phase_name}")
        return
    
    try:
        region = os.environ.get('AWS_DEFAULT_REGION', os.environ.get('AWS_REGION', 'us-east-1'))
        dynamodb = boto3.resource('dynamodb', region_name=region)
        table = dynamodb.Table(table_name)
        
        # Update the specific phase completion
        # Convert phase name to attribute: PRE_PROCESSING -> pre_processingElapsedTime, RECONSTRUCTION -> reconstructionElapsedTime
        phase_lower = phase_name.lower()
        if '_' in phase_lower:
            parts = phase_lower.split('_')
            attr_name = parts[0] + '_' + ''.join(parts[1:]) + 'ElapsedTime'
        else:
            attr_name = phase_lower + 'ElapsedTime'
        
        table.update_item(
            Key={os.environ.get('DDB_KEY_NAME', 'uuid'): uuid},
            UpdateExpression='SET #phase = :time, lastUpdatedPhase = :phase_name',
            ExpressionAttributeNames={'#phase': attr_name},
            ExpressionAttributeValues={
                ':time': elapsed_time,
                ':phase_name': phase_name
            }
        )
        if log:
            log.info(f"Updated {phase_name} completion: {elapsed_time}s")
    except Exception as e:
        if log:
            log.warning(f"Error updating phase completion: {e}")

def parse_3dgrut_metrics_from_log(log_output, output_json_path):
    """
    Parse 3DGRUT evaluation metrics from console output and save to JSON.
    
    Args:
        log_output: String containing console output from 3DGRUT render.py
        output_json_path: Path to save the metrics JSON file
    
    Returns:
        Dictionary with parsed metrics or None if parsing failed
    """
    import re
    import json
    
    metrics = {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0}
    
    # Parse from table format: │ 22.387 │ 0.643 │ 0.379 │ ...
    # 3DGRUT render.py outputs columns in order: mean_psnr | mean_ssim | mean_lpips
    table_match = re.search(r'│\s*([0-9.]+)\s*│\s*([0-9.]+)\s*│\s*([0-9.]+)\s*│', log_output)
    
    if table_match:
        metrics['psnr'] = float(table_match.group(1))
        metrics['ssim'] = float(table_match.group(2))
        metrics['lpips'] = float(table_match.group(3))
    
    # Only save if at least one metric was found
    if any(v > 0 for v in metrics.values()):
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump({'results': metrics}, f, indent=2)
        return metrics
    
    return None

def parse_gsplat_metrics_from_log(log_output, output_json_path):
    """
    Parse gsplat-mcmc evaluation metrics from console output and save to JSON.
    Format: PSNR: 24.901, SSIM: 0.8576, LPIPS: 0.167 Time: 0.031s/image Number of GS: 2929743
    
    Args:
        log_output: String containing console output from gsplat simple_trainer.py
        output_json_path: Path to save the metrics JSON file
    
    Returns:
        Dictionary with parsed metrics or None if parsing failed
    """
    import re
    import json
    
    psnr_match = re.search(r'PSNR:\s*([0-9.]+)', log_output)
    ssim_match = re.search(r'SSIM:\s*([0-9.]+)', log_output)
    lpips_match = re.search(r'LPIPS:\s*([0-9.]+)', log_output)
    
    if psnr_match and ssim_match and lpips_match:
        metrics = {
            'psnr': float(psnr_match.group(1)),
            'ssim': float(ssim_match.group(1)),
            'lpips': float(lpips_match.group(1))
        }
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump({'results': metrics}, f, indent=2)
        return metrics

    # Fallback: old trainer format "Eval PSNR: X.XX dB" (PSNR only)
    eval_psnr_match = re.search(r'Eval PSNR:\s*([0-9.]+)', log_output)
    if eval_psnr_match:
        metrics = {
            'psnr': float(eval_psnr_match.group(1)),
            'ssim': 0.0,
            'lpips': 0.0
        }
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump({'results': metrics}, f, indent=2)
        return metrics
    
    return None




def images_have_subdirectories(image_path):
    """
    Returns True if any immediate children of image_path are directories,
    indicating a subdirectory image layout (e.g. face_00/pano_006.png).
    """
    if not os.path.isdir(image_path):
        return False
    return any(os.path.isdir(os.path.join(image_path, e)) for e in os.listdir(image_path))


def flatten_images_for_gsplat(image_path, sparse_0_path, log):
    """
    Ensure images/ files and images.bin both use flat names (subdir_file.ext).

    Two scenarios handled:
    1. Files already flat (clean_images_dir ran): only update images.bin.
    2. Files still in subdirs (gsplat-depth, no dn-splatter preprocess):
       rename files AND update images.bin.
    """
    import struct

    images_bin = os.path.join(sparse_0_path, "images.bin")
    if not os.path.exists(images_bin):
        log.warning(f"flatten_images_for_gsplat: images.bin not found at {images_bin}, skipping")
        return

    try:
        with open(images_bin, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            records = []
            for _ in range(num_images):
                image_id = struct.unpack("<I", f.read(4))[0]
                qvec = struct.unpack("<4d", f.read(32))
                tvec = struct.unpack("<3d", f.read(24))
                camera_id = struct.unpack("<I", f.read(4))[0]
                name_bytes = b""
                while True:
                    c = f.read(1)
                    if c == b"\x00":
                        break
                    name_bytes += c
                name = name_bytes.decode("utf-8")
                num_pts = struct.unpack("<Q", f.read(8))[0]
                points2d_raw = f.read(num_pts * 24) if num_pts else b""
                records.append((image_id, qvec, tvec, camera_id, name, num_pts, points2d_raw))
    except Exception as e:
        log.error(f"flatten_images_for_gsplat: failed to read images.bin: {e}")
        return

    needs_flatten = any("/" in r[4] or "\\" in r[4] for r in records)
    if not needs_flatten:
        log.info("flatten_images_for_gsplat: images.bin already has flat names, nothing to do")
        return

    # Build rename map: original subdir name -> flat name
    rename_map = {}
    for _, _, _, _, name, _, _ in records:
        norm = name.replace("\\", "/")
        if "/" in norm:
            flat = norm.replace("/", "_")
            rename_map[norm] = flat

    # Rename image files on disk if they still have subdirectory structure
    renamed_files = 0
    for old_rel, flat_name in rename_map.items():
        src = os.path.join(image_path, old_rel)
        dst = os.path.join(image_path, flat_name)
        if os.path.exists(src) and not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst) if os.path.dirname(flat_name) else image_path, exist_ok=True)
            os.rename(src, dst)
            renamed_files += 1

    # Remove now-empty subdirectories
    if renamed_files > 0:
        for entry in os.listdir(image_path):
            entry_path = os.path.join(image_path, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path, ignore_errors=True)
        log.info(f"flatten_images_for_gsplat: renamed {renamed_files} image files on disk")

    # Update images.bin with flat names
    shutil.copy2(images_bin, images_bin + ".bak")
    try:
        with open(images_bin, "wb") as f:
            f.write(struct.pack("<Q", num_images))
            for image_id, qvec, tvec, camera_id, name, num_pts, points2d_raw in records:
                norm = name.replace("\\", "/")
                flat_name = norm.replace("/", "_") if "/" in norm else norm
                f.write(struct.pack("<I", image_id))
                f.write(struct.pack("<4d", *qvec))
                f.write(struct.pack("<3d", *tvec))
                f.write(struct.pack("<I", camera_id))
                f.write(flat_name.encode("utf-8") + b"\x00")
                f.write(struct.pack("<Q", num_pts))
                f.write(points2d_raw)
        count = len(rename_map)
        log.info(f"flatten_images_for_gsplat: updated images.bin — {count} names flattened")
    except Exception as e:
        log.error(f"flatten_images_for_gsplat: failed to write images.bin: {e}, restoring backup")
        shutil.copy2(images_bin + ".bak", images_bin)


def remove_unobserved_images_for_gsplat(sparse_0_path, log):
    """
    Remove images from images.bin that have zero valid 3D point references.

    gsplat's depth loss path calls point_indices[image_name] for every image
    without guarding for missing keys. Images with no triangulated points are
    absent from point_indices, causing a KeyError.

    Determines observed images by reading point3D_id references in images.bin
    directly (point3D_id >= 0 means the keypoint is matched to a 3D point).
    This handles COLMAP exports where points3D.bin tracks are stripped but
    images.bin still contains valid per-keypoint point3D_id references.

    Args:
        sparse_0_path: Path to colmap/sparse/0 containing images.bin
        log:           Logger instance
    """
    import struct

    images_bin = os.path.join(sparse_0_path, "images.bin")
    if not os.path.exists(images_bin):
        log.warning("remove_unobserved_images_for_gsplat: images.bin missing, skipping")
        return

    # Read images.bin, tracking which images have at least one valid point3D reference
    with open(images_bin, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        records = []
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")
            num_pts = struct.unpack("<Q", f.read(8))[0]
            points2d_raw = f.read(num_pts * 24) if num_pts else b""
            # Count valid point3D_id references (signed int64, -1 = unmatched)
            has_observations = False
            if num_pts:
                for j in range(num_pts):
                    p3d_id = struct.unpack("<q", points2d_raw[j*24+16 : j*24+24])[0]
                    if p3d_id >= 0:
                        has_observations = True
                        break
            records.append((image_id, qvec, tvec, camera_id, name, num_pts, points2d_raw, has_observations))

    kept = [r for r in records if r[7]]
    removed = [r[4] for r in records if not r[7]]

    if not removed:
        log.info("remove_unobserved_images_for_gsplat: all images have 3D observations, nothing to remove")
        return

    log.info(f"remove_unobserved_images_for_gsplat: removing {len(removed)} unobserved images: {removed}")

    shutil.copy2(images_bin, images_bin + ".bak_unobserved")
    with open(images_bin, "wb") as f:
        f.write(struct.pack("<Q", len(kept)))
        for image_id, qvec, tvec, camera_id, name, num_pts, points2d_raw, _ in kept:
            f.write(struct.pack("<I", image_id))
            f.write(struct.pack("<4d", *qvec))
            f.write(struct.pack("<3d", *tvec))
            f.write(struct.pack("<I", camera_id))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", num_pts))
            f.write(points2d_raw)

    log.info(f"remove_unobserved_images_for_gsplat: kept {len(kept)}/{num_images} images")


def remove_fully_masked_images_for_gsplat(image_path, sparse_0_path, masks_dir, log):
    """
    Remove images from images.bin (and from disk) where the segmentation mask
    covers all pixels — i.e. there are no valid (unmasked) pixels to train on.

    Args:
        image_path:    Path to images/ directory
        sparse_0_path: Path to colmap/sparse/0 containing images.bin
        masks_dir:     Path to masks/ directory (flat, after flattening)
        log:           Logger instance
    """
    import struct
    import cv2

    if not os.path.isdir(masks_dir):
        return

    images_bin = os.path.join(sparse_0_path, "images.bin")
    if not os.path.exists(images_bin):
        return

    with open(images_bin, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        records = []
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")
            num_pts = struct.unpack("<Q", f.read(8))[0]
            points2d_raw = f.read(num_pts * 24) if num_pts else b""
            records.append((image_id, qvec, tvec, camera_id, name, num_pts, points2d_raw))

    fully_masked = []
    for _, _, _, _, name, _, _ in records:
        mask_path = os.path.join(masks_dir, name)
        if os.path.isfile(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None and not (m > 127).any():
                fully_masked.append(name)

    if not fully_masked:
        log.info("remove_fully_masked_images_for_gsplat: no fully-masked images found")
        return

    log.info(f"remove_fully_masked_images_for_gsplat: removing {len(fully_masked)} fully-masked images: {fully_masked}")

    kept = [r for r in records if r[4] not in fully_masked]
    shutil.copy2(images_bin, images_bin + ".bak_fullymask")
    with open(images_bin, "wb") as f:
        f.write(struct.pack("<Q", len(kept)))
        for image_id, qvec, tvec, camera_id, name, num_pts, points2d_raw in kept:
            f.write(struct.pack("<I", image_id))
            f.write(struct.pack("<4d", *qvec))
            f.write(struct.pack("<3d", *tvec))
            f.write(struct.pack("<I", camera_id))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", num_pts))
            f.write(points2d_raw)

    # Remove image files from disk so the dataloader doesn't try to open them
    for name in fully_masked:
        img_file = os.path.join(image_path, name)
        if os.path.isfile(img_file):
            os.remove(img_file)

    log.info(f"remove_fully_masked_images_for_gsplat: kept {len(kept)}/{num_images} images")


def rebuild_points3d_tracks_for_gsplat(sparse_0_path, log):
    """
    Rebuild points3D.bin track data from images.bin point2D->point3D_id references.

    COLMAP 4.x rig reconstructions store points3D with track_length=0 because the
    reverse index is not populated in the rig format. gsplat's colmap parser builds
    point_indices from points3D tracks, resulting in an empty dict and KeyError.

    This function reads the forward references (point3D_id per keypoint in images.bin)
    and writes them back into points3D.bin as proper track entries, making the file
    compatible with gsplat's older pycolmap SceneManager.

    Uses pycolmap to read the reconstruction correctly (handles rig format).

    Args:
        sparse_0_path: Path to colmap/sparse/0
        log:           Logger instance
    """
    try:
        import pycolmap
    except ImportError:
        log.warning("rebuild_points3d_tracks_for_gsplat: pycolmap not available, skipping")
        return

    import struct

    points3d_bin = os.path.join(sparse_0_path, "points3D.bin")
    images_bin = os.path.join(sparse_0_path, "images.bin")

    if not os.path.exists(points3d_bin) or not os.path.exists(images_bin):
        log.warning("rebuild_points3d_tracks_for_gsplat: missing files, skipping")
        return

    # Check if tracks already populated
    with open(points3d_bin, "rb") as f:
        num_pts = struct.unpack("<Q", f.read(8))[0]
        if num_pts == 0:
            log.info("rebuild_points3d_tracks_for_gsplat: no points, skipping")
            return
        # Sample first point track length
        f.read(8)   # point3d_id
        f.read(24)  # xyz
        f.read(3)   # rgb
        f.read(8)   # error
        sample_track_len = struct.unpack("<Q", f.read(8))[0]

    if sample_track_len > 0:
        log.info("rebuild_points3d_tracks_for_gsplat: tracks already populated, skipping")
        return

    log.info("rebuild_points3d_tracks_for_gsplat: rebuilding tracks from images.bin point2D references")

    # Read reconstruction via pycolmap (handles rig format correctly)
    r = pycolmap.Reconstruction()
    r.read(sparse_0_path)

    # Build track index: point3D_id -> [(image_id, point2D_idx), ...]
    tracks = {}
    for image_id, image in r.images.items():
        for pt2d_idx, pt2d in enumerate(image.points2D):
            if pt2d.has_point3D():
                p3d_id = pt2d.point3D_id
                if p3d_id not in tracks:
                    tracks[p3d_id] = []
                tracks[p3d_id].append((image_id, pt2d_idx))

    log.info(f"rebuild_points3d_tracks_for_gsplat: built tracks for {len(tracks)}/{num_pts} points")

    if not tracks:
        log.warning("rebuild_points3d_tracks_for_gsplat: no tracks found in images, skipping")
        return

    # Rewrite points3D.bin with track data
    shutil.copy2(points3d_bin, points3d_bin + ".bak_tracks")

    # Read existing points3D data
    point_data = {}
    with open(points3d_bin, "rb") as f:
        num_pts = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_pts):
            p3d_id = struct.unpack("<Q", f.read(8))[0]
            xyz = f.read(24)
            rgb = f.read(3)
            error = f.read(8)
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)  # skip empty tracks
            point_data[p3d_id] = (xyz, rgb, error)

    with open(points3d_bin, "wb") as f:
        f.write(struct.pack("<Q", num_pts))
        for p3d_id, (xyz, rgb, error) in point_data.items():
            track = tracks.get(p3d_id, [])
            f.write(struct.pack("<Q", p3d_id))
            f.write(xyz)
            f.write(rgb)
            f.write(error)
            f.write(struct.pack("<Q", len(track)))
            for image_id, pt2d_idx in track:
                f.write(struct.pack("<I", image_id))
                f.write(struct.pack("<I", pt2d_idx))

    log.info(f"rebuild_points3d_tracks_for_gsplat: wrote {num_pts} points with track data")


def apply_camera_masks_to_images(image_path, masks_path, log):
    """
    Pre-apply per-camera static masks to images for gsplat depth loss training.

    gsplat's colmap dataset only supports camera-level masks for fisheye ROI and
    has no per-image mask loading. Since our masks are static per camera (same mask
    for all images from a given camera face), we apply them directly to the image
    pixels: masked pixels (value=0 in mask) are zeroed out in the image.

    The mask subdirectory name (e.g. 'face_01') is matched to the image subdirectory
    or flat image prefix after flattening (e.g. 'face_01_pano_001.png').

    Args:
        image_path:  Path to images directory (flat after flatten_images_for_gsplat)
        masks_path:  Path to masks directory containing per-camera subdirectories
        log:         Logger instance
    """
    import cv2

    if not os.path.isdir(masks_path):
        log.info("apply_camera_masks_to_images: no masks directory, skipping")
        return

    # Build camera_name -> mask array mapping (one mask per camera subdir)
    camera_masks = {}
    for cam_name in os.listdir(masks_path):
        cam_mask_dir = os.path.join(masks_path, cam_name)
        if not os.path.isdir(cam_mask_dir):
            continue
        mask_files = [f for f in os.listdir(cam_mask_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not mask_files:
            continue
        mask_img = cv2.imread(os.path.join(cam_mask_dir, mask_files[0]),
                              cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue
        # Normalize to binary: >0 = valid pixel
        mask_bin = (mask_img > 0).astype('uint8') * 255
        camera_masks[cam_name] = mask_bin
        log.info(f"apply_camera_masks_to_images: loaded mask for {cam_name} "
                 f"shape={mask_bin.shape} valid_frac={mask_bin.mean()/255:.2f}")

    if not camera_masks:
        log.info("apply_camera_masks_to_images: no camera masks found, skipping")
        return

    # Apply masks to images - match by camera name prefix
    image_files = [f for f in os.listdir(image_path)
                   if os.path.isfile(os.path.join(image_path, f))
                   and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    applied = 0
    for img_file in image_files:
        # Find matching camera mask by prefix (e.g. face_01_pano_001.png -> face_01)
        matched_cam = None
        for cam_name in camera_masks:
            if img_file.startswith(cam_name + "_") or img_file.startswith(cam_name + "/"):
                matched_cam = cam_name
                break

        if matched_cam is None:
            continue

        img_path = os.path.join(image_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        mask = camera_masks[matched_cam]

        # Resize mask to image dimensions if needed
        h, w = img.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Apply mask: zero out pixels where mask=0
        mask_3ch = mask[:, :, np.newaxis] if img.ndim == 3 else mask
        img_masked = (img * (mask_3ch / 255)).astype(img.dtype)
        cv2.imwrite(img_path, img_masked)
        applied += 1

    log.info(f"apply_camera_masks_to_images: applied masks to {applied}/{len(image_files)} images")
