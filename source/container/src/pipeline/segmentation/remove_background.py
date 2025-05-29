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

""" Remove the background of images given a directory of images """

import os
import cv2
import sys
import argparse
import subprocess
import shutil

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

if __name__ == '__main__':
    # Create Argument Parser
    parser = argparse.ArgumentParser(
        prog='',
        description=''
    )

    # Define the Arguments
    parser.add_argument(
        '-i', '--input_dir',
        required=True,
        default=None,
        action='store',
        help='Target data directory for the images'
    )

    # Define the Arguments
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
        help='The name of the background model to use (u2net of u2net_human_seg)'
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
        os.rename(input_dir_path, temp_path)
        os.makedirs(input_dir_path)
        input_dir_path = temp_path

    # Get a list of all image file names
    files = [f for f in os.listdir(input_dir_path) \
            if (os.path.isfile(os.path.join(input_dir_path, f))) and \
                (str((os.path.splitext(f)[1]).lower() == ".jpg" or \
                    str(os.path.splitext(f)[1]).lower() == ".png") or \
                        str((os.path.splitext(f)[1]).lower() == ".jpeg"))]
    if len(files) == 0:
        print(f"Input data directory {input_dir_path} does not contain any images.")

    files = sorted(files)

    # Check first file to see if alpha channel exists.
    # Assume all other images will be the same.
    has_alpha = has_alpha_channel(os.path.join(input_dir_path, files[0]))
    print(f"Has_alpha:{has_alpha}")
    try:
        for i, file in enumerate(files):
            args = [
                "python", "-m", 
                "backgroundremover.backgroundremover.cmd.cli",
                "-wn", num_threads,
                "-gb", num_gpus,
                "-m", model,
                "-i", os.path.join(input_dir_path, file),
                "-o", os.path.join(output_dir_path, file)
            ]

            # Improve the mask if alpha channel exists
            if has_alpha is True:
                args.extend(["-a", "-ae", "15"])

            subprocess.run(args, check=True)
        if temp_path is not None:
            shutil.rmtree(temp_path, ignore_errors=True)

    except Exception as e:
        raise RuntimeError(f"Error running background removal component: {e}") from e
