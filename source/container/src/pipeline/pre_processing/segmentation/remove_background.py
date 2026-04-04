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

""" Remove the background of images given a directory of images using BackgroundRemover """

import os
import cv2
import sys
import io
import argparse
import subprocess
import shutil
import numpy as np
from PIL import Image

def copy_images_to_temp(original_dir, temp_dir):
    """
    Copy images from original directory to temp directory
    
    Args:
        original_dir (str): Path to original directory
        temp_dir (str): Path to temp directory
    
    Returns:
        int: Number of images copied
    """
    # Define supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [
        f for f in os.listdir(original_dir) 
        if os.path.isfile(os.path.join(original_dir, f)) 
        and any(f.lower().endswith(ext) for ext in image_extensions)
    ]
    
    # Copy images to temp directory
    copied_count = 0
    for filename in image_files:
        src = os.path.join(original_dir, filename)
        dst = os.path.join(temp_dir, filename)
        try:
            shutil.copy2(src, dst)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {filename}: {str(e)}")
    
    return copied_count

def has_alpha_channel(image_path):
    """
    Check if an image has an alpha channel
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        bool: True if image has alpha channel, False otherwise
    """
    try:
        # Read image with unchanged flag to preserve alpha channel if present
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError("Failed to load image")

        # Check number of channels
        # If image has 4 channels (BGRA), it has an alpha channel
        return img.shape[-1] == 4

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False

def segment_human_rembg(image_path, session):
    """Segment human using rembg library with enhanced processing"""
    # Load image
    with open(image_path, 'rb') as f:
        input_data = f.read()
    
    # Remove background
    from rembg import remove
    output_data = remove(input_data, session=session)
    
    # Convert to PIL images
    original_image = Image.open(image_path).convert('RGB')
    segmented_image = Image.open(io.BytesIO(output_data)).convert('RGBA')
    
    # Extract mask from alpha channel
    mask = np.array(segmented_image)[:, :, 3] / 255.0
    
    return mask, original_image, segmented_image

def apply_enhanced_mask(image, mask, threshold=0.05):
    """Apply enhanced mask processing for better human segmentation"""
    img_array = np.array(image)
    
    # Normalize mask to 0-255 range
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Apply closing to fill small gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    
    # Apply dilation to expand the mask slightly
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(mask_closed, kernel_dilate, iterations=1)
    
    # Apply Gaussian blur for smoother edges
    mask_blurred = cv2.GaussianBlur(mask_dilated, (3, 3), 0)
    
    # Convert back to 0-1 range
    enhanced_mask = mask_blurred / 255.0
    
    # Create RGBA image with enhanced mask
    result = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
    result[:, :, :3] = img_array
    result[:, :, 3] = (enhanced_mask * 255).astype(np.uint8)
    
    return Image.fromarray(result, 'RGBA')

def process_human_segmentation(input_dir, output_dir):
    """Process images using enhanced human segmentation with dual-model detection.
    Runs u2net_human_seg + birefnet-portrait and merges masks to catch both
    prominent and distant/small humans.
    """
    try:
        from rembg import remove, new_session
        session_u2net = new_session('u2net_human_seg')
        session_birefnet = new_session('birefnet-portrait')
    except ImportError:
        raise RuntimeError("rembg library not available. Please install with: pip install rembg")
    
    # Make remove function available in the scope
    global remove_func
    remove_func = remove
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and any(f.lower().endswith(ext) for ext in image_extensions)
    ]
    
    processed_count = 0
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        base_name = os.path.splitext(image_file)[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")
        
        try:
            original_image = Image.open(input_path).convert('RGB')
            
            # Pass 1: u2net_human_seg - good for prominent humans
            mask1, _, _ = segment_human_rembg(input_path, session_u2net)
            
            # Pass 2: birefnet-portrait - better for distant/small humans
            mask2, _, _ = segment_human_rembg(input_path, session_birefnet)
            
            # Merge: take the maximum (union) of both masks
            merged_mask = np.maximum(mask1, mask2)
            
            # Apply enhanced processing on merged mask
            enhanced_image = apply_enhanced_mask(original_image, merged_mask)
            
            # Save result
            enhanced_image.save(output_path)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
    
    return processed_count

if __name__ == '__main__':
    # Create Argument Parser
    parser = argparse.ArgumentParser(
        prog='background-remover',
        description='Remove background from a directory of images'
    )

    # Define the Arguments
    parser.add_argument(
        '-i', '--input_dir',
        required=True,
        default=None,
        action='store',
        help='Target data directory for the images'
    )

    parser.add_argument(
        '-o', '--output_dir',
        required=True,
        default=None,
        action='store',
        help='Target data directory for the images'
    )

    parser.add_argument(
        '-m', '--model',
        required=False,
        default="u2net",
        action='store',
        help='The name of the background model to use (u2net, u2net_human_seg for enhanced human segmentation)'
    )

    parser.add_argument(
        '-nt', '--num_threads',
        required=False,
        default=None,
        action='store',
        help='The total number of threads to use'
    )

    parser.add_argument(
        '-ng', '--num_gpus',
        required=False,
        default=None,
        action='store',
        help='The total number of GPUs to use'
    )

    args = parser.parse_args()
    input_dir_path = args.input_dir
    output_dir_path = args.output_dir
    num_threads = args.num_threads
    num_gpus = args.num_gpus
    model = args.model

    if os.path.isdir(input_dir_path) is False:
        print(f"Input data directory {input_dir_path} does not exist.")
        sys.exit(1)
    temp_path = None
    if input_dir_path == output_dir_path:
        temp_path = f"{input_dir_path}_temp"
        
        # Check if temp directory already exists (from a previous run)
        if os.path.exists(temp_path):
            # Use existing temp directory
            print(f"Using existing temp directory: {temp_path}")
            input_dir_path = temp_path
        else:
            # Create temp directory and move files
            try:
                os.rename(input_dir_path, temp_path)
                os.makedirs(input_dir_path)
                input_dir_path = temp_path
            except Exception as e:
                print(f"Error renaming directory: {e}")
                
                # Try to copy files instead
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path)
                
                # Check if original images directory exists
                original_images = input_dir_path.replace('_temp', '')
                if os.path.exists(original_images) and os.path.isdir(original_images):
                    print(f"Copying files from {original_images} to {temp_path}")
                    copied_count = copy_images_to_temp(original_images, temp_path)
                    print(f"Copied {copied_count} images from {original_images} to {temp_path}")
                    input_dir_path = temp_path

    # Get a list of all image file names
    files = [f for f in os.listdir(input_dir_path) \
            if (os.path.isfile(os.path.join(input_dir_path, f))) and \
                (str((os.path.splitext(f)[1]).lower() == ".jpg" or \
                    str(os.path.splitext(f)[1]).lower() == ".png") or \
                        str((os.path.splitext(f)[1]).lower() == ".jpeg"))]
    if len(files) == 0:
        print(f"Input data directory {input_dir_path} does not contain any images.")
        
        # Try to find images in the original directory
        original_dir = input_dir_path.replace('_temp', '')
        if os.path.exists(original_dir) and os.path.isdir(original_dir):
            print(f"Checking original directory {original_dir} for images...")
            copied_count = copy_images_to_temp(original_dir, input_dir_path)
            
            if copied_count > 0:
                print(f"Copied {copied_count} images from {original_dir} to {input_dir_path}")
                # Refresh the files list
                files = [f for f in os.listdir(input_dir_path) \
                        if (os.path.isfile(os.path.join(input_dir_path, f))) and \
                            (str((os.path.splitext(f)[1]).lower() == ".jpg" or \
                                str(os.path.splitext(f)[1]).lower() == ".png") or \
                                    str((os.path.splitext(f)[1]).lower() == ".jpeg"))]
                files = sorted(files)
            else:
                print("No images found in original directory either.")
                # Exit gracefully if no images found
                if temp_path is not None:
                    # Restore original directory if we renamed it
                    shutil.rmtree(output_dir_path, ignore_errors=True)
                    if os.path.exists(temp_path):
                        try:
                            os.rename(temp_path, output_dir_path)
                            print(f"Restored original directory {output_dir_path}")
                        except Exception as e:
                            print(f"Error restoring original directory: {e}")
                sys.exit(0)
        else:
            # Exit gracefully if no images found
            if temp_path is not None:
                # Restore original directory if we renamed it
                shutil.rmtree(output_dir_path, ignore_errors=True)
                if os.path.exists(temp_path):
                    try:
                        os.rename(temp_path, output_dir_path)
                        print(f"Restored original directory {output_dir_path}")
                    except Exception as e:
                        print(f"Error restoring original directory: {e}")
            sys.exit(0)

    files = sorted(files)

    # Check first file to see if alpha channel exists.
    # Assume all other images will be the same.
    if len(files) == 0:
        print("No images found after all attempts. Exiting.")
        sys.exit(0)
        
    has_alpha = has_alpha_channel(os.path.join(input_dir_path, files[0]))
    print(f"Has_alpha:{has_alpha}")
    try:
        # Validate model parameter
        allowed_models = ["u2net", "u2net_human_seg", "birefnet-portrait"]
        if model not in allowed_models:
            raise ValueError(f"Invalid model: {model}")
        
        # Use enhanced human segmentation for 'human' model
        if model == "u2net_human_seg":
            print("Using enhanced human segmentation...")
            processed_count = process_human_segmentation(input_dir_path, output_dir_path)
            print(f"Processed {processed_count} images with enhanced human segmentation")
        else:
            # Use original backgroundremover for other models
            # Validate numeric parameters
            if num_threads and not str(num_threads).isdigit():
                raise ValueError(f"Invalid num_threads: {num_threads}")
            if num_gpus and not str(num_gpus).isdigit():
                raise ValueError(f"Invalid num_gpus: {num_gpus}")
            
            # Build command arguments - input validation above prevents injection
            args = [
                sys.executable, "-m", 
                "backgroundremover.backgroundremover.cmd.cli",
                "-wn", str(num_threads) if num_threads else "1",
                "-gb", str(num_gpus) if num_gpus else "0",
                "-m", model,
                "-if", input_dir_path,
                "-of", output_dir_path
            ]

            # Improve the mask if alpha channel exists, but skip alpha matting for human segmentation
            if has_alpha is True and model != "u2net_human_seg":
                args.extend(["-a", "-ae", "15"])

            subprocess.run(args, check=True)  # nosemgrep: dangerous-subprocess-use-audit,dangerous-subprocess-use-tainted-env-args

    except Exception as e:
        raise RuntimeError(f"Error running background removal component: {e}") from e
