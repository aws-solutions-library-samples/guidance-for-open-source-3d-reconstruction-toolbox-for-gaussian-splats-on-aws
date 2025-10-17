#!/usr/bin/env python3
"""
Coordinate transformation wrapper for Gaussian Splat files.
Transforms from NerfStudio coordinate system to various target systems.
"""

import argparse
import subprocess
import sys
import os

# Coordinate system transformation mappings
# From NerfStudio (Y-up, Z-forward, X-right) to target systems
COORDINATE_TRANSFORMS = {
    'babylon': 'x:270,y:180,z:0',      # Babylon.js (Y-up, Z-forward, X-right) - current default
    'unity': 'x:0,y:180,z:0',          # Unity (Y-up, Z-forward, X-right) - flip around Y
    'unreal': 'x:270,y:0,z:180',       # Unreal (Z-up, X-forward, Y-right) - rotate to Z-up
    'opengl': 'x:0,y:0,z:0',           # OpenGL (Y-up, -Z-forward, X-right) - identity
    'opencv': 'x:180,y:0,z:0',         # OpenCV (Y-down, Z-forward, X-right) - flip Y
    'supersplat': 'x:0,y:0,z:180',     # SuperSplat (Y-down by default) - Z flip for orientation
}

def main():
    parser = argparse.ArgumentParser(description='Transform Gaussian Splat coordinates to target system')
    parser.add_argument('-i', '--input', required=True, help='Input PLY file path')
    parser.add_argument('-o', '--output', help='Output PLY file path (default: overwrite input)')
    parser.add_argument('--target', choices=list(COORDINATE_TRANSFORMS.keys()), 
                       default='babylon', help='Target coordinate system')
    parser.add_argument('--custom', help='Custom rotation string (e.g., "x:90,y:180,z:45")')
    
    args = parser.parse_args()
    
    # Determine output path
    output_path = args.output if args.output else args.input
    
    # Get rotation string
    if args.custom:
        rotation_string = args.custom
    else:
        rotation_string = COORDINATE_TRANSFORMS[args.target]
    
    # Build command for rotate_splat.py
    rotate_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'rotate_splat.py'),
        '-i', args.input,
        '-o', output_path,
        '--rotations', rotation_string
    ]
    
    print(f"Transforming to {args.target} coordinate system...")
    print(f"Applying rotations: {rotation_string}")
    
    # Execute the rotation
    try:
        result = subprocess.run(rotate_cmd, check=True, capture_output=True, text=True)
        print(f"Successfully transformed coordinates to {args.target}")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error transforming coordinates: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()