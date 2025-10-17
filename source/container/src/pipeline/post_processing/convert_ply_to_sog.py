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
Convert PLY file to SOG format using PlayCanvas splat-transform.
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Convert PLY file to SOG format')
    parser.add_argument('-i', '--input', required=True, help='Input PLY file path')
    parser.add_argument('-o', '--output', help='Output SOG file path (default: same name with .sog extension)')
    parser.add_argument('-w', '--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('-c', '--cpu', action='store_true', help='Use CPU processing for large datasets')
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if args.output:
        output_path = args.output
    else:
        input_dir = os.path.dirname(args.input)
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(input_dir, f"{input_name}.sog")

    # Build command for splat-transform
    cmd = ['splat-transform', args.input, output_path]
    if args.overwrite:
        cmd.append('-w')
    if args.cpu:
        cmd.append('-c')
    
    print(f"Converting {args.input} to SOG format...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create if there's a directory part
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"Running command: {' '.join(cmd)}")
        print(f"Input file exists: {os.path.exists(args.input)}")
        print(f"Output directory: {os.path.dirname(output_path)}")
        print(f"Output directory exists: {os.path.exists(os.path.dirname(output_path))}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(f"Command completed with return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        
        # List all files in output directory
        output_dir = os.path.dirname(output_path)
        if os.path.exists(output_dir):
            print(f"Files in output directory {output_dir}:")
            for f in os.listdir(output_dir):
                full_path = os.path.join(output_dir, f)
                size = os.path.getsize(full_path) if os.path.isfile(full_path) else 'DIR'
                print(f"  {f} ({size} bytes)")
        else:
            print(f"Output directory {output_dir} does not exist")
        
        # Check if the main SOG file was created (that's all we need)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Successfully converted to {output_path} ({os.path.getsize(output_path)} bytes)")
        else:
            print(f"Error converting to SOG: output file missing or empty")
            print(f"Return code: {result.returncode}")
            print(f"SOG conversion failed gracefully. Pipeline will continue without SOG export.")
            sys.exit(0)  # Exit gracefully instead of with error
            
    except Exception as e:
        print(f"Error running splat-transform: {e}")
        print(f"SOG conversion failed gracefully. Pipeline will continue without SOG export.")
        sys.exit(0)  # Exit gracefully instead of with error

if __name__ == "__main__":
    main()