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
Create derivative PLY files for different export formats.
Renames original PLY to orig.ply and creates copies for enabled export formats.
"""

import os
import shutil
import argparse

def create_derivative_plys(input_ply, orig_ply=None, spz_ply=None, sog_ply=None, usdz_ply=None):
    """
    Create derivative PLY files for different export formats.
    
    Args:
        input_ply: Path to the input PLY file
        orig_ply: Path for orig.ply (optional, defaults to same dir as input)
        spz_ply: Path for spz.ply (optional, creates if provided)
        sog_ply: Path for sog.ply (optional, creates if provided)
        usdz_ply: Path for usdz.ply (optional, creates if provided)
    """
    if not os.path.exists(input_ply):
        raise FileNotFoundError(f"Input PLY file not found: {input_ply}")
    
    # Default orig_ply path if not provided
    if orig_ply is None:
        output_dir = os.path.dirname(input_ply)
        orig_ply = os.path.join(output_dir, "orig.ply")
    
    # Rename original to orig.ply
    print(f"Renaming {input_ply} to {orig_ply}")
    shutil.move(input_ply, orig_ply)
    
    # Create copies for specified paths
    if spz_ply:
        print(f"Creating {spz_ply}")
        shutil.copy2(orig_ply, spz_ply)
    
    if sog_ply:
        print(f"Creating {sog_ply}")
        shutil.copy2(orig_ply, sog_ply)
    
    if usdz_ply:
        print(f"Creating {usdz_ply}")
        shutil.copy2(orig_ply, usdz_ply)
    
    # Copy orig.ply back to original name for final PLY export
    print(f"Creating final {input_ply}")
    shutil.copy2(orig_ply, input_ply)
    
    print("Derivative PLY files created successfully")

def main():
    parser = argparse.ArgumentParser(description="Create derivative PLY files for export formats")
    parser.add_argument("-i", "--input", required=True, help="Input PLY file path")
    parser.add_argument("--orig-ply", help="Path for orig.ply")
    parser.add_argument("--spz-ply", help="Path for spz.ply")
    parser.add_argument("--sog-ply", help="Path for sog.ply")
    parser.add_argument("--usdz-ply", help="Path for usdz.ply")
    
    args = parser.parse_args()
    
    create_derivative_plys(
        args.input,
        orig_ply=args.orig_ply,
        spz_ply=args.spz_ply,
        sog_ply=args.sog_ply,
        usdz_ply=args.usdz_ply
    )

if __name__ == "__main__":
    main()
