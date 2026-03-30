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
This file serves as the utility to convert coordinate systems from one system to another
right-hand, y-up (rhyu) - gradio.Model3D,playcanvas,three.js,webgl,godot
left-hand, y-up (lhyu) - Babylon.js, Unity
right-hand, z-up (rhzu) - Blender
left-hand, z-up (lhzu) - Unreal
"""

import sys
import argparse
import subprocess

TRANSFORMS = {
    'rhyu': '-180,0,0',
    'lhyu': '-180,180,0',
    'rhzu': '0,0,0',
    'lhzu': '0,180,0',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input PLY file')
    parser.add_argument('-o', '--output', required=True, help='Output PLY file')
    parser.add_argument('--target', default='rhyu', help='Target coordinate system')
    args = parser.parse_args()
    
    rotation = TRANSFORMS.get(args.target, TRANSFORMS['rhyu'])
    
    cmd = [
        'splat-transform',
        args.input,
        args.output,
        f'--rotate={rotation}',
        '-w'
    ]
    
    result = subprocess.run(cmd, check=True)
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
