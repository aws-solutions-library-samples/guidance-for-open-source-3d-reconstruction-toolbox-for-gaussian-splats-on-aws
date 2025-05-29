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

""" Transform target poses into colmap coordinates and create the colmap model files """

import os
import json
import argparse
import sqlite3
import numpy as np
from pathlib import Path
import shutil
from scipy.spatial.transform import Rotation

def inspect_colmap_database(db_path):
    """
    Inspect a COLMAP database and print its schema to the screen.
    
    Args:
        db_path: Path to the COLMAP SQLite database file
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of all tables
        cursor.execute(
            """
            SELECT name 
            FROM sqlite_master 
            ORDER BY image_id
            WHERE type='table'
            """
        )
        #cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"Database: {db_path}")
        print(f"Found {len(tables)} tables:")
        
        # For each table, print its schema
        for table_name in tables:
            table_name = table_name[0]
            print(f"\n{'=' * 50}")
            print(f"TABLE: {table_name}")
            print(f"{'=' * 50}")
            
            # Get table schema
            cursor.execute(
                f"""
                PRAGMA table_info({table_name})
                """
            )
            #cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            # Print column information
            print(f"{'Column Name':<20} {'Type':<15} {'NotNull':<8} {'Default Value':<15} {'Primary Key'}")
            print(f"{'-' * 70}")
            for col in columns:
                col_id, name, type_, not_null, default_val, primary_key = col
                print(f"{name:<20} {type_:<15} {not_null:<8} {str(default_val):<15} {primary_key}")
            
            # Print a few sample rows if the table has data
            cursor.execute(
                f"""
                SELECT COUNT(*) FROM {table_name}
                """
            )
            #cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            
            if count > 0:
                print(f"\nSample data ({min(3, count)} rows):")
                cursor.execute(
                    f"""
                    SELECT * FROM {table_name} LIMIT 3
                    """
                )
                #cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                rows = cursor.fetchall()
                for row in rows:
                    print(row)
            else:
                print("\nTable is empty.")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error: {e}")

def sync_images_txt_with_db(database_path, images_dir, sparse_path):
    """
    Synchronizes the images.txt file order with the database.db file order.
    
    Args:
        database_path (str): Path to the COLMAP database.db file
        images_dir (str): Path to the directory containing the source images
        sparse_path (str): Path to the COLMAP sparse reconstruction directory containing images.txt
    """
    # Convert paths to Path objects
    db_path = Path(database_path)
    images_dir = Path(images_dir)
    sparse_path = Path(sparse_path)
    images_txt_path = sparse_path / 'images.txt'
    
    # Verify paths exist
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not images_txt_path.exists():
        raise FileNotFoundError(f"images.txt not found: {images_txt_path}")
    
    # Create backup of original images.txt
    backup_path = images_txt_path.with_suffix('.txt.backup')
    shutil.copy2(images_txt_path, backup_path)
    
    # Read image information from database
    db_images = {}
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT image_id, name 
            FROM images 
            ORDER BY image_id
            """
        )
        #cursor.execute("SELECT image_id, name FROM images ORDER BY image_id;")
        for image_id, name in cursor.fetchall():
            db_images[image_id] = name
    
    # Read images.txt content
    images_txt_content = {}
    current_image_block = []
    
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
        
    # Parse images.txt
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#'):
            i += 1
            continue
            
        if line:  # First line of image block
            image_data = line.split()
            image_id = int(image_data[0])  # First element is IMAGE_ID
            current_image_block = [line]
            
            # Next line contains the second row of image data
            if i + 1 < len(lines):
                current_image_block.append(lines[i + 1].rstrip())
                
            images_txt_content[image_id] = current_image_block
            i += 2
        else:
            i += 1
    
    # Write new images.txt with correct order
    with open(images_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for image_id in sorted(db_images.keys()):
            if image_id in images_txt_content:
                # Get the original line and update the image name
                line_parts = images_txt_content[image_id][0].split()
                line_parts[9] = db_images[image_id]  # Update the image name
                new_line = ' '.join(line_parts)
                
                f.write(new_line + '\n')
                f.write(images_txt_content[image_id][1] + '\n')
            else:
                print(f"Warning: Image ID {image_id} ({db_images[image_id]}) found in database but not in images.txt")
    
    print(f"Successfully synchronized images.txt with database.db")
    print(f"Original images.txt backed up to: {backup_path}")
    print(f"Processed {len(db_images)} images from database")
    print(f"Found {len(images_txt_content)} images in images.txt")

def get_image_extension(depth_path):
    """
    Get the extension of the images
    """
    return os.path.splitext(depth_path.replace('.depth', ''))[1]

def process_transforms(transforms_json):
    """
    Process pose transforms
    """
    print("Processing transforms...")
    frames = transforms_json['frames']
    frames.sort(key=lambda x: x['file_path'])
    
    poses = []
    image_names = []
    cameras = []

    img_ext = get_image_extension(frames[0]['depth_path'])
    print(f"Detected image extension: {img_ext}")
    
    for frame in frames:
        transform = np.array(frame['transform_matrix'])
        poses.append(transform)
        
        # Extract just the filename part and add extension
        base_name = os.path.basename(frame['file_path'])
        image_name = base_name + img_ext
        image_names.append(image_name)
        
        cameras.append({
            'w': frame['w'],
            'h': frame['h'],
            'fl_x': frame['fl_x'],
            'cx': frame['cx'],
            'cy': frame['cy']
        })
    
    print(f"Processed {len(frames)} frames")
    return poses, image_names, cameras

def load_transforms(transforms_path):
    """
    Load the json file with the intrinsic and extrinsic parameters
    """
    print(f"Loading transforms from {transforms_path}")
    with open(transforms_path, 'r') as f:
        return json.load(f)
    
def normalize_poses(poses):
    """
    Normalize poses to be centered around origin and scaled to reasonable bounds
    """
    print("Normalizing poses...")
    
    # Extract camera centers (translation vectors)
    centers = np.array([pose[:3, 3] for pose in poses])
    
    # Calculate centroid and scale
    centroid = centers.mean(axis=0)
    scale = np.max(np.abs(centers - centroid)) * 1.1  # Add 10% margin
    
    normalized_poses = []
    for pose in poses:
        normalized = pose.copy()
        # Center and scale the translation
        normalized[:3, 3] = (pose[:3, 3] - centroid) / scale
        normalized_poses.append(normalized)
    
    print(f"Poses normalized. Centroid: {centroid}, Scale: {scale}")
    return normalized_poses, centroid, scale

def pose_to_colmap_matrix(source_matrix, source_coord_name, is_world_to_camera):
    """
    Convert 4x4 transformation matrix from a source system to COLMAP coordinate system,
    accounting for world-to-camera vs camera-to-world transformation.
    
    Args:
        source_matrix: 4x4 numpy array representing transformation in source coordinates
        is_world_to_camera: boolean indicating if input is world-to-camera transform
        
    Returns:
        4x4 numpy array representing camera-to-world transformation in COLMAP coordinates
    """

        # Create coordinate conversion matrix
    match str(source_coord_name).lower():
        case 'arkit':
            # ARKit to COLMAP conversion:
            # X_colmap =  X_arkit
            # Y_colmap = -Y_arkit
            # Z_colmap = -Z_arkit
            transform = np.array([
                [ 1,  0,  0, 0],
                [ 0, -1,  0, 0],
                [ 0,  0, -1, 0],
                [ 0,  0,  0, 1]
            ])
        case 'arcore':
            # ARCore uses a right-handed coordinate system with Y pointing up
            transform = np.array([
                [ 1,  0,  0, 0],
                [ 0,  1,  0, 0],
                [ 0,  0, -1, 0],
                [ 0,  0,  0, 1]
            ])
        case 'opengl':
            # OpenGL uses a right-handed coordinate system where:
            # X points right
            # Y points up
            # Z points out of the screen (towards the viewer)
            transform = np.array([
                [ 1,  0,  0, 0],
                [ 0, -1,  0, 0],
                [ 0,  0, -1, 0],
                [ 0,  0,  0, 1]
            ])
        case 'opencv': #identity
            # Right-handed coordinate system
            # X points right
            # Y points down
            # Z points forward (away from the camera)
            transform = np.array([
                [ 1,  0,  0, 0],
                [ 0,  1,  0, 0],
                [ 0,  0,  1, 0],
                [ 0,  0,  0, 1]
            ])
        case 'ros':
            # ROS's X (forward) becomes COLMAP's Z
            # ROS's Y (left) becomes COLMAP's -X
            # ROS's Z (up) becomes COLMAP's -Y
            transform = np.array([
                [ 0,  0,  1, 0],
                [-1,  0,  0, 0],
                [ 0, -1,  0, 0],
                [ 0,  0,  0, 1]
            ])
        case _:
            raise RuntimeError(f"""Input pose coordinate name {source_coord_name} not currently supported.
                               Only arkit, arcore, opengl, opencv, and ros are supported.""")

    if is_world_to_camera:
        # First apply coordinate change, then invert
        temp_matrix = transform @ source_matrix @ transform.T
        
        # Extract rotation and translation
        rot = temp_matrix[:3, :3]
        trans = temp_matrix[:3, 3]
        
        # Invert transform
        inv_rot = rot.T
        inv_trans = -inv_rot @ trans
        
        # Create COLMAP camera-to-world matrix
        colmap_matrix = np.eye(4)
        colmap_matrix[:3, :3] = inv_rot
        colmap_matrix[:3, 3] = inv_trans
    else:
        # If already camera-to-world, just apply coordinate change
        colmap_matrix = transform @ source_matrix @ transform.T
    
    return colmap_matrix

def normalize_quaternion_from_matrix_stable(matrix):
    """
    Extract rotation from transformation matrix, convert to quaternion,
    normalize it with better numerical stability, and return new transformation matrix.
    
    Args:
        matrix: 4x4 transformation matrix as numpy array
        
    Returns:
        4x4 transformation matrix with normalized rotation component
    """
    # Extract 3x3 rotation matrix
    rotation_matrix = matrix[:3, :3]
    
    # Convert to quaternion
    rot = Rotation.from_matrix(rotation_matrix)
    quat = rot.as_quat()
    
    # Get magnitude squared
    mag_squared = np.sum(quat * quat)
    
    # Normalize quaternion with better numerical stability
    if abs(1.0 - mag_squared) < 2.107342e-8:
        # If already very close to normalized, use first order approximation
        quat = quat * (2.0 / (1.0 + mag_squared))
    else:
        quat = quat / np.sqrt(mag_squared)
    
    # Convert back to rotation matrix
    normalized_rot = Rotation.from_quat(quat)
    normalized_matrix = matrix.copy()
    normalized_matrix[:3, :3] = normalized_rot.as_matrix()
    
    return normalized_matrix

def write_cameras_file(cameras, output_dir):
    """
    Write the Colmap cameras.txt file with camera intrinsic params
        0 = 'w': frame['w'],
        1 = 'h': frame['h'],
        2 = 'fl_x': frame['fl_x'],
        3 = 'cx': frame['cx'],
        4 = 'cy': frame['cy']
    """
    cameras_path = os.path.join(output_dir, 'cameras.txt')
    print(f"Writing cameras file to {cameras_path}")
    with open(cameras_path, 'w', encoding="utf-8") as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')

        for idx, camera in enumerate(cameras, 1):
            params = [camera['fl_x'], camera['cx'], camera['cy'], 0.0]
            f.write(f'{idx} SIMPLE_RADIAL {camera["w"]} {camera["h"]} {" ".join(map(str, params))}\n')
    print(f"Wrote {len(cameras)} cameras to file")

def write_images_file(image_names, poses, output_dir, source_coord_name, pose_is_world_to_cam):
    """
    Write images.txt file that Colmap needs for camera poses for each image
    """
    images_path = os.path.join(output_dir, 'images.txt')
    print(f"Writing images file to {images_path}")
    with open(images_path, 'w', encoding="utf-8") as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        normalized_poses, center, scale = normalize_poses(poses)
        for idx, (name, pose) in enumerate(zip(image_names, normalized_poses), 1):
            # Normalize quaternion and convert pose to colmap coordinates
            r = normalize_quaternion_from_matrix_stable(pose)
            colmap_matrix = pose_to_colmap_matrix(r, source_coord_name, pose_is_world_to_cam)
            R = Rotation.from_matrix(colmap_matrix[:3, :3])
            R_quat = R.as_quat()
            R_colmap = [R_quat[3], R_quat[0], R_quat[1], R_quat[2]] # Colmap expects scalar first

            # For translation
            t_colmap = colmap_matrix[:3, 3]

            # Extract just the filename without the 'images/' prefix
            image_name = os.path.basename(name)

            # Write the camera pose
            f.write(f'{idx} {R_colmap[0]} {R_colmap[1]} {R_colmap[2]} {R_colmap[3]} {t_colmap[0]} {t_colmap[1]} {t_colmap[2]} {1} {image_name}\n')
            # Empty second line for points2D
            f.write('\n')
    print(f"Wrote {len(image_names)} images to file")

def create_empty_points3d_file(output_dir):
    """
    Create an empty points3D.txt file so Colmap can use to populate point cloud during traingulation
    """
    points3d_path = os.path.join(output_dir, 'points3D.txt')
    print(f"Creating empty points3D file at {points3d_path}")
    with open(points3d_path, 'w', encoding="utf-8") as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')

def get_colmap_image_order(database_path):
    """
    Query the COLMAP database to get the order of images as processed by COLMAP
    """
    with sqlite3.connect(database_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT name 
            FROM images 
            ORDER BY image_id
            """
        )
        #cursor.execute("SELECT name FROM images ORDER BY image_id")
        image_names = [row[0] for row in cursor.fetchall()]
    return image_names

def update_colmap_db_with_pose_priors(colmap_db_path, images_txt_path):
    """
    Update a COLMAP database with pose priors from images.txt file.
    
    Args:
        colmap_db_path (str): Path to the COLMAP database file
        images_txt_path (str): Path to the COLMAP images.txt file containing pose information
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if files exist
        if not os.path.exists(colmap_db_path):
            print(f"Error: COLMAP database file not found at {colmap_db_path}")
            return False
            
        if not os.path.exists(images_txt_path):
            print(f"Error: images.txt file not found at {images_txt_path}")
            return False
        
        # Parse images.txt to extract pose information
        image_poses = {}
        with open(images_txt_path, 'r') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or not line:
                i += 1
                continue
                
            # Parse image line
            # Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            parts = line.split()
            if len(parts) >= 10:
                image_id = int(parts[0])
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                name = parts[9]
                
                # Store the pose information
                image_poses[image_id] = {
                    'quaternion': [qw, qx, qy, qz],
                    'translation': [tx, ty, tz],
                    'name': name
                }
                
                # Skip the next line which contains feature information
                i += 2
            else:
                i += 1
        
        # Connect to the COLMAP database
        conn = sqlite3.connect(colmap_db_path)
        cursor = conn.cursor()
        
        # First, clear any existing pose priors
        cursor.execute(
            """
            DELETE FROM pose_priors
            """
        )
        #cursor.execute("DELETE FROM pose_priors")
        
        # Update the pose_priors table with the pose information
        for image_id, pose_data in image_poses.items():
            # Get translation
            tx, ty, tz = pose_data['translation']
            
            # Convert quaternion to rotation matrix
            qw, qx, qy, qz = pose_data['quaternion']
            
            # Create rotation matrix from quaternion
            R = np.zeros((3, 3))
            R[0, 0] = 1 - 2 * qy**2 - 2 * qz**2
            R[0, 1] = 2 * qx * qy - 2 * qz * qw
            R[0, 2] = 2 * qx * qz + 2 * qy * qw
            R[1, 0] = 2 * qx * qy + 2 * qz * qw
            R[1, 1] = 1 - 2 * qx**2 - 2 * qz**2
            R[1, 2] = 2 * qy * qz - 2 * qx * qw
            R[2, 0] = 2 * qx * qz - 2 * qy * qw
            R[2, 1] = 2 * qy * qz + 2 * qx * qw
            R[2, 2] = 1 - 2 * qx**2 - 2 * qy**2
            
            # COLMAP uses a different convention: camera-to-world vs world-to-camera
            # We need to invert the transformation
            R_inv = R.transpose()
            t_inv = -R_inv @ np.array([tx, ty, tz])
            
            # Create position blob (translation vector)
            position = np.array(t_inv, dtype=np.float64)
            position_blob = position.tobytes()
            
            # Create a default identity covariance matrix (low uncertainty)
            # Small values indicate high confidence in the pose
            covariance = np.eye(3, dtype=np.float64) * 0.01
            covariance_blob = covariance.tobytes()
            
            # Use coordinate system 1 (COLMAP world coordinate system)
            coordinate_system = 1
            
            # Insert into pose_priors table
            cursor.execute("""
                INSERT OR REPLACE INTO pose_priors 
                (image_id, position, coordinate_system, position_covariance)
                VALUES (?, ?, ?, ?)
            """, (image_id, position_blob, coordinate_system, covariance_blob))
            
            # Check if the insert was successful
            if cursor.rowcount == 0:
                print(f"Warning: Failed to insert pose prior for image ID {image_id}")
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        print(f"Successfully updated {len(image_poses)} images with pose priors")
        return True
        
    except Exception as e:
        print(f"Error updating COLMAP database with pose priors: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    try:
        # Create Argument Parser with Rich Formatter
        parser = argparse.ArgumentParser(
            prog='process-pose-transforms',
            description='Extract an archive with prior poses and images for SfM'
        )

        # Define the Arguments
        parser.add_argument(
            '-i', '--input_transform_path',
            required=True,
            default=None,
            action='store',
            help='Input transforms.json path to use for processing pose priors'
        )

        parser.add_argument(
            '-c', '--source_coord_name',
            required=True,
            default=None,
            action='store',
            help='The source coordinate name: arkit, arcore, opengl, opencv, or ros'
        )

        parser.add_argument(
            '-p', '--pose_is_world_to_cam',
            required=True,
            default=None,
            action='store',
            help="""Whether the input poses are in world-to-camera (true),
            otherwise input poses will be camera-to-world (false)"""
        )

        parser.add_argument(
            '-t', '--use_transforms_file',
            required=True,
            default=None,
            action='store',
            help="""Whether to use transforms.json file for poses or
            already existing colmap model text files"""
        )

        args = parser.parse_args()

        source_coord_name = args.source_coord_name
        pose_is_world_to_cam = args.pose_is_world_to_cam
        source_transforms_path = args.input_transform_path
        use_transforms_file = args.use_transforms_file

        workspace_path = os.path.dirname(source_transforms_path)
        colmap_db_path = os.path.join(workspace_path, "database.db")
        image_path = os.path.join(workspace_path, "images")
        sparse_path = os.path.join(workspace_path, "sparse", "0")

        if str(use_transforms_file).lower() == "false":
            sync_images_txt_with_db(colmap_db_path, image_path, sparse_path)
        else:
            colmap_image_names = get_colmap_image_order(colmap_db_path)
            image_names = os.listdir(image_path)
            image_names = sorted(image_names)

            # Transform the poses and process them
            transforms_data = load_transforms(source_transforms_path)
            poses, image_names, cameras = process_transforms(transforms_data)
            print(f"Length of poses list: {len(poses)}")

            # Reorder poses and cameras to match COLMAP's order
            name_to_idx = {name: idx for idx, name in enumerate(image_names)}
            reordered_poses = []
            reordered_cameras = []

            for img_name in colmap_image_names:
                idx = name_to_idx[img_name]
                reordered_poses.append(poses[idx])
                reordered_cameras.append(cameras[idx])

            # Write reconstruction files with correct order
            write_cameras_file(reordered_cameras, sparse_path),
            write_images_file(
                colmap_image_names,
                reordered_poses,
                sparse_path,
                source_coord_name,
                pose_is_world_to_cam
            )
            create_empty_points3d_file(sparse_path)
        # Using spatial matcher matcher for pose priors requires colmap db to be updated with poses
        update_colmap_db_with_pose_priors(colmap_db_path, os.path.join(sparse_path, "images.txt"))
    except Exception as e:
        raise RuntimeError(f"Error converting pose priors: {e}") from e