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

import argparse
import subprocess
import sys
import tempfile
import os
import logging

def main():
    parser = argparse.ArgumentParser(description='Crop bounding box for 3D gaussian splats')
    parser.add_argument('input_path', help='Input PLY file path')
    parser.add_argument('output_path', help='Output PLY file path')
    parser.add_argument('--mode', choices=['rigid_body', 'environment'], required=True,
                       help='Processing mode: rigid_body or environment')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                       help='Set logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(levelname)s: %(message)s')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    # Use temp file if input and output paths are the same
    if os.path.abspath(args.input_path) == os.path.abspath(args.output_path):
        temp_fd, temp_path = tempfile.mkstemp(suffix='.ply')
        os.close(temp_fd)
        final_output = temp_path
    else:
        final_output = args.output_path
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create intermediate output path in same directory as input
    input_dir = os.path.dirname(os.path.abspath(args.input_path))
    intermediate_output = os.path.join(input_dir, 'temp_output.ply')
    
    # Run extract_center_cube.py
    cmd1 = [
        'python', os.path.join(script_dir, 'extract_center_cube.py'),
        args.input_path, intermediate_output,
        '--method', 'std',
        '--std-multiplier', '3.0' # Higher value = less crop, expand bounding box
        # 1-Std=68%, 2-Std=95%, 3-Std=99.7%
    ]
    
    try:
        logging.info(f"Running extract_center_cube.py with mode: {args.mode}")
        subprocess.run(cmd1, check=True)
        
        if args.mode == 'rigid_body':
            # Run refine_center_object.py for rigid body mode
            logging.info("Running refine_center_object.py for rigid body refinement")
            cmd2 = [
                'python', os.path.join(script_dir, 'refine_center_object.py'),
                intermediate_output, final_output,
                '--method', 'density_percentile',
                '--percentile', '89',
                '--iterations', '11'
            ]
            subprocess.run(cmd2, check=True)
        else:
            # For environment mode, just copy the intermediate result
            logging.info("Environment mode: copying intermediate result")
            subprocess.run(['cp', intermediate_output, final_output], check=True)
        
        # If using temp file, move it to original location
        if final_output != args.output_path:
            logging.debug("Moving temp file to final location")
            subprocess.run(['mv', final_output, args.output_path], check=True)
        
        # Clean up intermediate file
        if os.path.exists(intermediate_output):
            os.unlink(intermediate_output)
            
        logging.info(f"Processing complete. Output saved to: {args.output_path}")
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        # Clean up intermediate and temp files
        if os.path.exists(intermediate_output):
            os.unlink(intermediate_output)
        if final_output != args.output_path and os.path.exists(final_output):
            os.unlink(final_output)
        sys.exit(1)

if __name__ == '__main__':
    main()
