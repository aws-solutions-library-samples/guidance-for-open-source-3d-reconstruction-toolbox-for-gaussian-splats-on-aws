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
Script to rotate a 3D Gaussian Splatting asset with spherical harmonics.
This script handles both the rotation of Gaussian positions and the transformation
of spherical harmonic coefficients.
"""

import numpy as np
import argparse
from scipy.spatial.transform import Rotation
import plyfile
import sys

def load_ply(ply_path):
    """
    Load a PLY file containing Gaussian Splatting data.
    
    Args:
        ply_path: Path to the PLY file
        
    Returns:
        Dictionary containing the Gaussian parameters and the original vertex data
    """
    print(f"Loading PLY file from {ply_path}")
    plydata = plyfile.PlyData.read(ply_path)
    
    # Extract vertex data
    vertices = plydata['vertex']
    
    # Get all property names from the vertex element
    property_names = [prop.name for prop in vertices.properties]
    print(f"Available fields in PLY: {property_names}")
    
    # Create a dictionary to store all parameters
    gaussian_data = {
        'positions': np.vstack([vertices['x'], vertices['y'], vertices['z']]).T,
        'rotations': np.vstack([vertices['rot_0'], vertices['rot_1'], vertices['rot_2'], vertices['rot_3']]).T,
        'scales': np.vstack([vertices['scale_0'], vertices['scale_1'], vertices['scale_2']]).T,
        'opacities': vertices['opacity'],
        'original_vertices': vertices,  # Store the original vertex data
        'property_names': property_names,  # Store property names for later
        'plydata': plydata  # Store the original plydata
    }
    
    # Extract spherical harmonic coefficients if they exist
    sh_fields = {'dc': [], 'rest': []}
    
    for field in property_names:
        if field.startswith('f_dc_'):
            sh_fields['dc'].append(field)
        elif field.startswith('f_rest_'):
            sh_fields['rest'].append(field)
    
    # Sort the fields to ensure correct order
    sh_fields['dc'].sort(key=lambda x: int(x.split('_')[-1]))
    sh_fields['rest'].sort(key=lambda x: int(x.split('_')[-1]))
    
    print(f"Found DC fields: {sh_fields['dc']}")
    print(f"Found rest fields: {sh_fields['rest']}")
    
    # Extract DC components
    if sh_fields['dc']:
        dc_components = [vertices[field] for field in sh_fields['dc']]
        gaussian_data['sh_dc'] = np.vstack(dc_components).T
        gaussian_data['sh_dc_fields'] = sh_fields['dc']
    
    # Extract rest components
    if sh_fields['rest']:
        rest_components = [vertices[field] for field in sh_fields['rest']]
        gaussian_data['sh_rest'] = np.vstack(rest_components).T
        gaussian_data['sh_rest_fields'] = sh_fields['rest']
    
    return gaussian_data

def rotate_gaussians(gaussian_data, rotation_matrix):
    """
    Rotate Gaussian positions and transform spherical harmonic coefficients.
    
    Args:
        gaussian_data: Dictionary containing Gaussian parameters
        rotation_matrix: 3x3 rotation matrix
        
    Returns:
        Dictionary with rotated Gaussian parameters
    """
    rotated_data = gaussian_data.copy()
    
    # Rotate positions
    rotated_data['positions'] = np.dot(gaussian_data['positions'], rotation_matrix.T)
    
    # Transform quaternion rotations
    # Convert rotation matrix to quaternion
    rot = Rotation.from_matrix(rotation_matrix)
    rot_quat = rot.as_quat()  # x, y, z, w format
    
    # Convert to scalar-first format if needed
    rot_quat_scalar_first = np.array([rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2]])
    
    # Apply quaternion multiplication to existing rotations
    existing_quats = gaussian_data['rotations']
    
    # Convert to scipy Rotation objects and compose
    rotated_quats = []
    for q in existing_quats:
        # Convert from scalar-first to scalar-last if needed
        q_scipy_format = np.array([q[1], q[2], q[3], q[0]])
        existing_rot = Rotation.from_quat(q_scipy_format)
        new_rot = rot * existing_rot
        new_quat = new_rot.as_quat()  # x, y, z, w format
        # Convert back to scalar-first format
        rotated_quats.append([new_quat[3], new_quat[0], new_quat[1], new_quat[2]])
    
    rotated_data['rotations'] = np.array(rotated_quats)
    
    # Transform spherical harmonic coefficients
    if 'sh_dc' in gaussian_data and 'sh_rest' in gaussian_data:
        rotated_data['sh_dc'], rotated_data['sh_rest'] = rotate_sh_coefficients(
            gaussian_data['sh_dc'], 
            gaussian_data['sh_rest'], 
            rotation_matrix
        )
    
    return rotated_data

def rotate_sh_coefficients(sh_dc, sh_rest, rotation_matrix):
    """
    Rotate spherical harmonic coefficients.
    
    Args:
        sh_dc: DC component of spherical harmonics (Nx3)
        sh_rest: Rest of SH coefficients
        rotation_matrix: 3x3 rotation matrix
        
    Returns:
        Tuple of rotated (sh_dc, sh_rest)
    """
    # For a proper implementation, we would need to compute the Wigner D-matrices
    # for the given rotation and apply them to the SH coefficients.
    # This is a simplified implementation that works for degree 1 SH.
    
    # Rotate DC component (these are just RGB values, no rotation needed)
    rotated_dc = sh_dc.copy()
    
    # For the rest of the coefficients, we need to apply the rotation
    # This is a simplified approach - for higher degree SH, a more complex transformation is needed
    rotated_rest = sh_rest.copy()
    
    # For degree 1 SH (linear terms), we can directly apply the rotation matrix
    # For higher degrees, we would need to use Wigner D-matrices
    
    # Assuming the first 9 coefficients correspond to degree 1 SH (3 components per RGB)
    if sh_rest.shape[1] >= 9:
        for i in range(0, 9, 3):
            # Extract the 3 coefficients for one color channel
            coeff = sh_rest[:, i:i+3]
            # Apply rotation
            rotated_coeff = np.dot(coeff, rotation_matrix.T)
            # Put back
            rotated_rest[:, i:i+3] = rotated_coeff
    
    # Note: For higher degree SH, a more complex transformation using Wigner D-matrices is needed
    
    return rotated_dc, rotated_rest

def save_ply(gaussian_data, output_path):
    """
    Save Gaussian data to a PLY file, preserving all original fields.
    
    Args:
        gaussian_data: Dictionary containing Gaussian parameters
        output_path: Path to save the PLY file
    """
    print(f"Saving rotated Gaussian data to {output_path}")
    
    # Get the original vertex data and property names
    original_vertices = gaussian_data['original_vertices']
    property_names = gaussian_data['property_names']
    
    # Create a list to store the new vertex data
    vertex_data = []
    num_points = len(original_vertices)
    
    # Create a list of properties for the new PLY file
    properties = []
    for prop in original_vertices.properties:
        properties.append((prop.name, 'f4'))
    
    # Create a structured array with the properties
    new_vertices = np.zeros(num_points, dtype=properties)
    
    # Copy all fields from the original data
    for prop_name in property_names:
        if prop_name in ['x', 'y', 'z']:
            continue  # We'll update these separately
        if prop_name.startswith('rot_'):
            continue  # We'll update these separately
        if prop_name.startswith('f_dc_') and 'sh_dc' in gaussian_data:
            continue  # We'll update these separately
        if prop_name.startswith('f_rest_') and 'sh_rest' in gaussian_data:
            continue  # We'll update these separately
        
        # Copy the original field
        new_vertices[prop_name] = original_vertices[prop_name]
    
    # Update the rotated positions
    new_vertices['x'] = gaussian_data['positions'][:, 0]
    new_vertices['y'] = gaussian_data['positions'][:, 1]
    new_vertices['z'] = gaussian_data['positions'][:, 2]
    
    # Update the rotated quaternions
    new_vertices['rot_0'] = gaussian_data['rotations'][:, 0]
    new_vertices['rot_1'] = gaussian_data['rotations'][:, 1]
    new_vertices['rot_2'] = gaussian_data['rotations'][:, 2]
    new_vertices['rot_3'] = gaussian_data['rotations'][:, 3]
    
    # Update spherical harmonic DC coefficients if they exist
    if 'sh_dc' in gaussian_data and 'sh_dc_fields' in gaussian_data:
        for i, field in enumerate(gaussian_data['sh_dc_fields']):
            new_vertices[field] = gaussian_data['sh_dc'][:, i]
    
    # Update spherical harmonic rest coefficients if they exist
    if 'sh_rest' in gaussian_data and 'sh_rest_fields' in gaussian_data:
        for i, field in enumerate(gaussian_data['sh_rest_fields']):
            new_vertices[field] = gaussian_data['sh_rest'][:, i]
    
    # Create PLY element
    vertex_element = plyfile.PlyElement.describe(new_vertices, 'vertex')
    
    # Create and write PLY file
    ply_data = plyfile.PlyData([vertex_element], comments=gaussian_data['plydata'].comments)
    ply_data.write(output_path)

def create_rotation_matrix(axis, angle_degrees):
    """
    Create a rotation matrix for the specified axis and angle.
    
    Args:
        axis: Rotation axis ('x', 'y', or 'z')
        angle_degrees: Rotation angle in degrees
        
    Returns:
        3x3 rotation matrix
    """
    angle_rad = np.radians(angle_degrees)
    
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif axis == 'y':
        return np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    else:  # z-axis
        return np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])

def parse_rotation_spec(rotation_spec):
    """
    Parse a rotation specification string in the format "axis:angle,axis:angle,..."
    
    Args:
        rotation_spec: String in the format "x:90,y:180,z:45" or similar
        
    Returns:
        List of (axis, angle) tuples
    """
    if not rotation_spec:
        return []
        
    rotations = []
    parts = rotation_spec.split(',')
    
    for part in parts:
        if ':' in part:
            axis, angle = part.split(':')
            axis = axis.strip().lower()
            if axis in ['x', 'y', 'z']:
                try:
                    angle = float(angle.strip())
                    rotations.append((axis, angle))
                except ValueError:
                    print(f"Warning: Invalid angle value in '{part}', skipping")
        else:
            print(f"Warning: Invalid rotation specification '{part}', skipping")
    
    return rotations

def main():
    parser = argparse.ArgumentParser(description='Rotate a Gaussian Splatting asset with spherical harmonics')
    parser.add_argument('input_ply', nargs='?', default=None, help='Path to input PLY file')
    parser.add_argument('output_ply', nargs='?', default=None, help='Path to output PLY file')
    parser.add_argument('--input', '-i', type=str, default=None, help='Path to input PLY file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Path to output PLY file')
    parser.add_argument('--angle', type=float, default=None, help='Rotation angle in degrees (legacy)')
    parser.add_argument('--axis', choices=['x', 'y', 'z'], default=None, help='Rotation axis (legacy)')
    parser.add_argument('--rotations', type=str, default=None, 
                        help='Rotation specification in format "x:90,y:180,z:45" for multiple rotations')
    
    args = parser.parse_args()
    
    # Prioritize explicit flags over positional arguments
    input_path = args.input if args.input is not None else args.input_ply
    output_path = args.output if args.output is not None else args.output_ply
    
    if input_path is None:
        parser.error("Input path is required. Use positional argument or --input/-i flag.")
        
    # If output is not specified, use input path (in-place rotation)
    if output_path is None:
        output_path = input_path
    
    # Load Gaussian data
    gaussian_data = load_ply(input_path)
    
    # Determine rotation method
    if args.rotations:
        # Parse the rotation specification
        rotations = parse_rotation_spec(args.rotations)
        if not rotations:
            print("No valid rotations specified, using legacy parameters if available")
            if args.axis and args.angle is not None:
                rotations = [(args.axis, args.angle)]
            else:
                print("No rotations to apply, exiting")
                return
    else:
        # Use legacy parameters
        if args.axis and args.angle is not None:
            rotations = [(args.axis, args.angle)]
        else:
            print("No rotations specified, exiting")
            return
    
    # Apply all rotations in sequence
    for axis, angle in rotations:
        # Create rotation matrix
        rotation_matrix = create_rotation_matrix(axis, angle)
        print(f"Rotating around {axis}-axis by {angle} degrees")
        
        # Rotate Gaussian data
        gaussian_data = rotate_gaussians(gaussian_data, rotation_matrix)
    
    # Debug info about the data we're saving
    print(f"Saving data with fields: {list(gaussian_data.keys())}")
    if 'sh_dc' in gaussian_data:
        print(f"sh_dc shape: {gaussian_data['sh_dc'].shape}")
    if 'sh_rest' in gaussian_data:
        print(f"sh_rest shape: {gaussian_data['sh_rest'].shape}")
    
    # Save rotated data
    save_ply(gaussian_data, output_path)
    
    print(f"Rotation complete! Applied {len(rotations)} rotation(s)")

if __name__ == "__main__":
    # Example usage:
    # python rotate_splat_simple.py input.ply output.ply --rotations "x:90,y:180,z:45"
    # python rotate_splat_simple.py input.ply output.ply --axis x --angle 90
    # python rotate_splat_simple.py --input input.ply --output output.ply --rotations "x:90,y:180,z:45"
    # python rotate_splat_simple.py --input input.ply --rotations "x:90,y:180,z:45"  # In-place rotation
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)