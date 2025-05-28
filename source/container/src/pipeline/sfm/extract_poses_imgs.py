"""
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Pose prior extraction

This script will extract the images and transforms.json for the pipeline.
In particular, it will accept the output of the NerfCapture app (https://github.com/jc211/NeRFCapture) to input
camera, image, and pose data into a Gaussian Splatting pipeline using SfM pose-priors to expedite the pipeline.


Usage:
    python extract_poses_imgs.py

Required Archive Structure:
    working_directory/archive.zip
    ├── transforms.json
    └── images/
        └── *.{png,jpg,jpeg}  # Image files must be in sequential order

Input:
    - a zip archive file that includes:
    - images/ directory with images (depth images are optional)
    - transforms.json: camera parameters and poses from https://github.com/jc211/NeRFCapture
        Format:
        {
            "frames": [
                {
                    "file_path": str,
                    "transform_matrix": 4x4 matrix,
                    "w": int,
                    "h": int,
                    "fl_x": float,
                    "fl_y": float,
                    "cx": float,
                    "cy": float,
                    "depth_path": str
                },
                ...
            ]
        }

Output:
    workspace_dir/
    ├── transforms-in.json
    ├── normalized_poses.csv  # Pose normalization parameters
    ├── cameras.csv           # Initial camera parameters
    └── images/
        ├── *.{png,jpg,jpeg}  # Image files
    └── depth_images/
        ├── *.{png,jpg,jpeg}  # Image files

Author: @eecorn
Date: 2025-01-28
Version: 1.0
"""

import os
import time
import shutil
import zipfile
import argparse
from pathlib import Path
from typing import Union

def get_image_extension(depth_path):
    """
    Get the image extension from transforms file using the depth image
    """
    return os.path.splitext(depth_path.replace('.depth', ''))[1]

def separate_depth_images(source_dir, depth_dir_name="depth_images"):
    """
    Moves images with 'depth' in their filename to a separate directory.
    
    Args:
        source_dir (str): Source directory containing the mixed images
        depth_dir_name (str): Name for the depth images directory
    """
    # Create Path objects
    source_path = Path(source_dir)
    # Create depth directory next to the images directory instead of using relative path
    depth_path = source_path.parent / depth_dir_name

    # Create depth directory and any necessary parent directories
    depth_path.mkdir(parents=True, exist_ok=True)

    # Count for reporting
    moved_count = 0

    # Move depth images to the new directory
    if source_path.exists():
        for file_path in source_path.iterdir():
            if file_path.is_file() and 'depth' in file_path.name.lower():
                try:
                    shutil.move(str(file_path), str(depth_path / file_path.name))
                    moved_count += 1
                except Exception as e:
                    raise RuntimeError(f"Error moving {file_path.name}: {e}") from e

    print(f"Moved {moved_count} depth images to {depth_dir_name}")

def extract_zip(zip_path: Union[str, Path], use_transforms: str) -> bool:
    try:
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        if zip_path.suffix.lower() != '.zip':
            raise ValueError(f"Not a valid zip file: {zip_path}")

        extract_dir = zip_path.parent
        images_dir = extract_dir / 'images'
        depth_dir = extract_dir / 'depth_images'
        sparse_dir = extract_dir / 'sparse'  # Add sparse directory

        # Create all necessary directories
        for directory in [images_dir, depth_dir, sparse_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        print(f"Extracting: {zip_path}")
        print(f"To: {extract_dir}")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            print(f"Found {total_files} files in archive")

            transforms_found = False
            processed_count = 0

            for i, file in enumerate(file_list, 1):
                path_parts = Path(file).parts
                filename = path_parts[-1]

                # Skip directory entries
                if not filename or filename.endswith('/'):
                    continue

                # Determine target location based on file path and name
                if filename == 'transforms.json':
                    target_path = extract_dir / filename
                    transforms_found = True
                elif file.startswith('sparse/'):
                    # Handle files in sparse directory
                    relative_path = Path(file)
                    target_path = extract_dir / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                elif 'depth' in filename.lower():
                    target_path = depth_dir / filename
                elif any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                    target_path = images_dir / filename
                else:
                    print(f"Skipping unknown file type: {filename}")
                    continue

                # Extract the file
                try:
                    # Skip if it's a directory
                    if not zip_ref.getinfo(file).is_dir():
                        with zip_ref.open(file) as source, open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        processed_count += 1
                        print(f"{processed_count}/{total_files}: {filename} -> {target_path}")
                except Exception as e:
                    print(f"Warning: Could not extract {file}: {str(e)}")
                    continue

            print(f"use_transforms={use_transforms}")
            if not transforms_found and use_transforms.lower() == "true":
                raise ValueError("transforms.json not found in the archive")

        print("\nExtraction completed successfully")
        return True

    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Error: File is not a valid zip archive: {zip_path}") from e
    except Exception as e:
        raise RuntimeError(f"Error extracting zip file: {str(e)}") from e


if __name__ == '__main__':
    try:
        # Create Argument Parser with Rich Formatter
        parser = argparse.ArgumentParser(
        prog='extract-poses-imgs',
        description='Extract an archive with prior poses and images for SfM'
        )

        # Define the Arguments
        parser.add_argument(
            '-i', '--input_zip_path',
            required=True,
            default=None,
            action='store',
            help='Target data archive for the reconstruction'
        )

        # Define the Arguments
        parser.add_argument(
            '-t', '--use_transforms',
            required=True,
            default=None,
            action='store',
            help='Whether to process the transforms.json file'
        )

        args = parser.parse_args()

        start_time = time.time()
        zip_path = args.input_zip_path
        data_path = os.path.dirname(zip_path)
        use_transforms = str(args.use_transforms).lower()
        print(f"use_transforms={use_transforms}")
        extract_zip(zip_path, use_transforms)

        transforms_path = os.path.join(data_path, 'transforms.json')
        image_dir = os.path.join(data_path, 'images')
        print(f"Workspace directory: {data_path}")
        print(f"Image directory: {image_dir}")

        if use_transforms == "true":
            print(f"Using transforms.json file for poses...")
            if os.path.isfile(transforms_path):
                # Rename the input transforms.json so to not conflict with NS transforms.json file
                head, tail = os.path.splitext(transforms_path)
                transforms_in_path = f"{head}-in{tail}"
                os.rename(transforms_path, transforms_in_path)

                # Move the depth images outside the "images" folder
                separate_depth_images(image_dir)
            else:
                raise RuntimeError("Pose prior transform file not found.")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
    except Exception as e:
        raise RuntimeError(f"Error running poses to SfM: {e}") from e
