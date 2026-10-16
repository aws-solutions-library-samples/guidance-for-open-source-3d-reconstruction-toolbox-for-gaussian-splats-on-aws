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
            WHERE type='table'
            ORDER BY name
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
            
            # Validate table name to prevent SQL injection
            if not table_name.replace('_', '').replace('-', '').isalnum():
                print(f"Skipping table with invalid name: {table_name}")
                continue
            
            # Get table schema using PRAGMA with safe identifier
            # PRAGMA doesn't support parameterized queries, so we validate and escape the table name
            safe_table_name = table_name.replace('"', '""')
            cursor.execute('PRAGMA table_info("' + safe_table_name + '")')  # nosemgrep: sqlalchemy-execute-raw-query,formatted-sql-query
            columns = cursor.fetchall()
            
            # Print column information
            print(f"{'Column Name':<20} {'Type':<15} {'NotNull':<8} {'Default Value':<15} {'Primary Key'}")
            print(f"{'-' * 70}")
            for col in columns:
                col_id, name, type_, not_null, default_val, primary_key = col
                print(f"{name:<20} {type_:<15} {not_null:<8} {str(default_val):<15} {primary_key}")
            
            # Print a few sample rows if the table has data
            # Additional validation: ensure table name is safe for SQL
            if not all(c.isalnum() or c in '_-' for c in table_name):
                print(f"Skipping unsafe table name: {table_name}")
                continue
            
            # Use safe table name with proper escaping
            safe_table_name = table_name.replace('"', '""')
            try:
                cursor.execute('SELECT COUNT(*) FROM "' + safe_table_name + '"')  # nosemgrep: formatted-sql-query,sqlalchemy-execute-raw-query
                count = cursor.fetchone()[0]
                
                if count > 0:
                    print(f"\nSample data ({min(3, count)} rows):")
                    cursor.execute('SELECT * FROM "' + safe_table_name + '" LIMIT 3')  # nosemgrep: formatted-sql-query,sqlalchemy-execute-raw-query
                    rows = cursor.fetchall()
                    for row in rows:
                        print(row)
            except sqlite3.Error as e:
                print(f"Error querying table {table_name}: {e}")
                continue
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
        images_bin_path = sparse_path / 'images.bin'
        if images_bin_path.exists():
            # Remove COLMAP 4.x rig/frame files before conversion —
            # they cause RigId crashes in both pycolmap and the CLI
            for _rig in ("rigs.bin", "frames.bin",
                         "rigs.txt", "frames.txt"):
                _rp = sparse_path / _rig
                if _rp.exists():
                    _rp.unlink()
                    print(f"Removed {_rig} before conversion")
            converted = False
            try:
                import pycolmap
                recon = pycolmap.Reconstruction(str(sparse_path))
                recon.write_text(str(sparse_path))
                converted = True
                print(
                    f"Converted COLMAP binary to text in"
                    f" {sparse_path} (pycolmap)"
                )
            except Exception as pyc_err:
                print(
                    f"pycolmap conversion failed ({pyc_err}),"
                    " trying colmap model_converter CLI"
                )
                import subprocess
                subprocess.run(
                    [
                        "colmap", "model_converter",
                        "--input_path", str(sparse_path),
                        "--output_path", str(sparse_path),
                        "--output_type", "TXT",
                    ],
                    check=True,
                )
                converted = True
                print(
                    f"Converted COLMAP binary to text in"
                    f" {sparse_path} (CLI)"
                )
            if converted:
                for _stale in (
                    "images.bin", "cameras.bin", "points3D.bin",
                    "rigs.bin", "frames.bin",
                    "rigs.txt", "frames.txt",
                ):
                    _bp = sparse_path / _stale
                    if _bp.exists():
                        _bp.unlink()
                print(
                    "Removed binary/rig model files"
                )
        if not images_txt_path.exists():
            raise FileNotFoundError(
                f"images.txt not found: {images_txt_path}"
            )
    
    # Create backup of original images.txt
    backup_path = images_txt_path.with_suffix('.txt.backup')
    shutil.copy2(images_txt_path, backup_path)

    # Read database: image_id, name, camera_id for each image
    db_images: dict[int, tuple[str, int]] = {}
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT image_id, name, camera_id "
            "FROM images ORDER BY image_id"
        )
        for image_id, name, camera_id in cursor.fetchall():
            db_images[image_id] = (name, camera_id)

    # Build name → db info lookup
    db_name_to_info: dict[str, tuple[int, int]] = {}
    for image_id, (name, camera_id) in db_images.items():
        db_name_to_info[name] = (image_id, camera_id)

    # Parse images.txt — index by image NAME (not numeric ID)
    model_entries: dict[str, list[str]] = {}
    with open(images_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            model_name = parts[9]
            second_line = lines[i + 1].rstrip() if i + 1 < len(lines) else ""
            model_entries[model_name] = [line, second_line]
            i += 2
        else:
            i += 1

    # Build a basename fallback index for model entries whose names
    # include subdirectory paths (e.g. "face_00/scan_000.jpg").
    # First match wins when multiple model entries share a basename.
    basename_to_model: dict[str, str] = {}
    for model_name in model_entries:
        bn = os.path.basename(model_name)
        if bn != model_name and bn not in basename_to_model:
            basename_to_model[bn] = model_name

    # Write images.txt matching model poses to database IDs by name
    matched = 0
    matched_by_basename = 0
    skipped_db = 0
    with open(images_txt_path, 'w', encoding='utf-8') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for db_id in sorted(db_images.keys()):
            db_name, db_cam_id = db_images[db_id]
            if db_name in model_entries:
                entry_key = db_name
            elif db_name in basename_to_model:
                entry_key = basename_to_model[db_name]
                matched_by_basename += 1
            else:
                skipped_db += 1
                continue
            parts = model_entries[entry_key][0].split()
            parts[0] = str(db_id)
            parts[8] = str(db_cam_id)
            parts[9] = db_name
            f.write(" ".join(parts) + "\n")
            f.write(model_entries[entry_key][1] + "\n")
            matched += 1

    # Write cameras.txt from database so camera IDs are consistent
    cameras_txt_path = sparse_path / "cameras.txt"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT camera_id, model, width, height, params "
            "FROM cameras ORDER BY camera_id"
        )
        db_cameras = cursor.fetchall()
    if db_cameras:
        _MODEL_NAMES = {
            0: "SIMPLE_PINHOLE", 1: "PINHOLE",
            2: "SIMPLE_RADIAL", 3: "RADIAL",
            4: "OPENCV", 5: "OPENCV_FISHEYE",
        }
        with open(cameras_txt_path, 'w', encoding='utf-8') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            for cam_id, model_id, w, h, params_blob in db_cameras:
                model_name = _MODEL_NAMES.get(model_id, f"UNKNOWN_{model_id}")
                params = np.frombuffer(params_blob, dtype=np.float64)
                params_str = " ".join(str(p) for p in params)
                f.write(f"{cam_id} {model_name} {w} {h} {params_str}\n")
        print(f"Wrote {len(db_cameras)} cameras from database to cameras.txt")

    print(f"Synchronized images.txt: {matched} matched by name"
          f" ({matched_by_basename} via basename fallback), "
          f"{skipped_db} db images not in model, "
          f"{len(model_entries) - matched} model images not in db")
    print(f"Original images.txt backed up to: {backup_path}")

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
    
    root_cam_model = transforms_json.get('camera_model')

    for frame in frames:
        transform = np.array(frame['transform_matrix'])
        poses.append(transform)

        # Extract just the filename part and add extension
        base_name = os.path.basename(frame['file_path'])
        image_name = base_name + img_ext
        image_names.append(image_name)

        src = frame if 'fl_x' in frame else transforms_json
        cam = {
            'w': frame.get('w', transforms_json.get('w')),
            'h': frame.get('h', transforms_json.get('h')),
            'fl_x': src['fl_x'],
            'fl_y': src.get('fl_y', src['fl_x']),
            'cx': src['cx'],
            'cy': src['cy'],
        }
        for k in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'):
            v = frame.get(k, transforms_json.get(k))
            if v is not None:
                cam[k] = v
        cam['camera_model'] = frame.get('camera_model', root_cam_model)
        cameras.append(cam)
    
    print(f"Processed {len(frames)} frames")
    return poses, image_names, cameras

def load_transforms(transforms_path):
    """
    Load the json file with the intrinsic and extrinsic parameters
    """
    print(f"Loading transforms from {transforms_path}")
    with open(transforms_path, 'r', encoding='utf-8') as f:
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

def get_camera_model_from_db(database_path):
    """
    Query the COLMAP database to get the camera model used during feature extraction.
    Returns the model name (e.g., 'SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'PINHOLE', 'RADIAL').
    """
    with sqlite3.connect(database_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT model 
            FROM cameras 
            LIMIT 1
            """
        )
        result = cursor.fetchone()
        if result:
            # COLMAP stores model as integer, map to string
            model_id = result[0]
            model_map = {
                0: 'SIMPLE_PINHOLE',
                1: 'PINHOLE',
                2: 'SIMPLE_RADIAL',
                3: 'RADIAL',
                4: 'OPENCV',
                5: 'OPENCV_FISHEYE',
                6: 'FULL_OPENCV',
                7: 'FOV',
                8: 'SIMPLE_RADIAL_FISHEYE',
                9: 'RADIAL_FISHEYE',
                10: 'THIN_PRISM_FISHEYE'
            }
            return model_map.get(model_id, 'SIMPLE_PINHOLE')
    return 'SIMPLE_PINHOLE'

def _determine_camera_model(camera):
    """Determine COLMAP camera model from intrinsic fields."""
    explicit = camera.get('camera_model')
    if explicit:
        model_map = {
            'OPENCV': 'OPENCV',
            'OPENCV_FISHEYE': 'OPENCV_FISHEYE',
            'PINHOLE': 'PINHOLE',
            'SIMPLE_PINHOLE': 'SIMPLE_PINHOLE',
            'SIMPLE_RADIAL': 'SIMPLE_RADIAL',
            'RADIAL': 'RADIAL',
        }
        mapped = model_map.get(explicit.upper())
        if mapped:
            return mapped

    k1 = float(camera.get('k1', 0))
    k2 = float(camera.get('k2', 0))
    p1 = float(camera.get('p1', 0))
    p2 = float(camera.get('p2', 0))
    has_dist = any(v != 0 for v in (k1, k2, p1, p2))
    fl_x = camera['fl_x']
    fl_y = camera.get('fl_y', fl_x)
    if has_dist:
        return 'OPENCV'
    if fl_x != fl_y:
        return 'PINHOLE'
    return 'PINHOLE'


def _camera_params_list(camera, model):
    """Build COLMAP parameter list for a camera model."""
    fl_x = camera['fl_x']
    fl_y = camera.get('fl_y', fl_x)
    cx = camera['cx']
    cy = camera['cy']
    k1 = float(camera.get('k1', 0))
    k2 = float(camera.get('k2', 0))
    p1 = float(camera.get('p1', 0))
    p2 = float(camera.get('p2', 0))

    if model == 'SIMPLE_PINHOLE':
        return [fl_x, cx, cy]
    if model == 'PINHOLE':
        return [fl_x, fl_y, cx, cy]
    if model == 'SIMPLE_RADIAL':
        return [fl_x, cx, cy, k1]
    if model == 'RADIAL':
        return [fl_x, cx, cy, k1, k2]
    if model == 'OPENCV':
        return [fl_x, fl_y, cx, cy, k1, k2, p1, p2]
    if model == 'OPENCV_FISHEYE':
        k3 = float(camera.get('k3', 0))
        k4 = float(camera.get('k4', 0))
        return [fl_x, fl_y, cx, cy, k1, k2, k3, k4]
    return [fl_x, fl_y, cx, cy]


def write_cameras_file(cameras, output_dir, database_path=None):
    """Write COLMAP cameras.txt from per-frame intrinsics."""
    cameras_path = os.path.join(output_dir, 'cameras.txt')
    print(f"Writing cameras file to {cameras_path}")
    with open(cameras_path, 'w', encoding="utf-8") as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')

        for idx, camera in enumerate(cameras, 1):
            model = _determine_camera_model(camera)
            params = _camera_params_list(camera, model)
            params_str = " ".join(str(p) for p in params)
            f.write(
                f'{idx} {model} {camera["w"]} {camera["h"]}'
                f' {params_str}\n'
            )
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
            f.write(f'{idx} {R_colmap[0]} {R_colmap[1]} {R_colmap[2]} {R_colmap[3]} {t_colmap[0]} {t_colmap[1]} {t_colmap[2]} {idx} {image_name}\n')
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

def clear_rig_tables(db_path: str) -> None:
    """Drop COLMAP 4.x rig tables from the database.

    COLMAP 4.x ``feature_extractor`` creates rig/frame tables even for
    single-camera datasets.  If these tables *exist* (even empty),
    ``point_triangulator`` enters the rig code path and crashes when
    the text model files disagree with auto-generated frame
    assignments.  Dropping the tables entirely forces COLMAP to fall
    back to the non-rig loading path.
    """
    rig_tables = ("frame_data", "frames", "rig_sensors", "rigs")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing = {r[0] for r in cursor.fetchall()}
        dropped: list[str] = []
        for tbl in rig_tables:
            if tbl in existing:
                cursor.execute(f"DROP TABLE {tbl}")
                dropped.append(tbl)
        conn.commit()
        conn.close()
        if dropped:
            print(
                "Dropped COLMAP 4.x rig tables to avoid"
                f" triangulator crash: {dropped}"
            )
    except Exception as exc:
        print(f"Warning: could not drop rig tables: {exc}")


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
        with open(images_txt_path, 'r', encoding='utf-8') as f:
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

        # DEBUG: print full schema of pose_priors before doing anything
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        all_tables = [r[0] for r in cursor.fetchall()]
        print(f"[POSE_PRIOR_DEBUG] All tables in DB: {all_tables}")
        cursor.execute("PRAGMA table_info(pose_priors)")
        schema_rows = cursor.fetchall()
        print(f"[POSE_PRIOR_DEBUG] pose_priors PRAGMA rows: {schema_rows}")

        # First, clear any existing pose priors
        cursor.execute("DELETE FROM pose_priors")

        # Detect schema variant:
        #   COLMAP 4.x rig: corr_data_id (references image_id), corr_sensor_type=0
        #   COLMAP 4.x early: image_id FK
        #   COLMAP 3.x: image_name text PK
        cursor.execute("PRAGMA table_info(pose_priors)")
        pose_prior_cols = {row[1] for row in cursor.fetchall()}
        print(f"[POSE_PRIOR_DEBUG] pose_prior_cols: {pose_prior_cols}")

        if not pose_prior_cols:
            print("[POSE_PRIOR_DEBUG] pose_priors table empty — creating 4.x schema")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pose_priors (
                    image_id INTEGER PRIMARY KEY NOT NULL,
                    position BLOB,
                    coordinate_system INTEGER NOT NULL,
                    position_covariance BLOB,
                    FOREIGN KEY(image_id) REFERENCES images(image_id)
                        ON DELETE CASCADE
                )
            """)
            pose_prior_cols = {'image_id'}

        use_corr_schema = "corr_data_id" in pose_prior_cols
        use_image_id_schema = "image_id" in pose_prior_cols
        schema_label = (
            "corr_data_id (4.x rig)" if use_corr_schema
            else "image_id (4.x)" if use_image_id_schema
            else "image_name (3.x)"
        )
        print(f"[POSE_PRIOR_DEBUG] schema={schema_label}")

        # Build name -> (image_id, camera_id) lookup for 4.x schemas
        name_to_image = {}
        if use_corr_schema or use_image_id_schema:
            cursor.execute("SELECT image_id, camera_id, name FROM images")
            for row in cursor.fetchall():
                name_to_image[row[2]] = {
                    'image_id': row[0],
                    'camera_id': row[1],
                }

        for image_id, pose_data in image_poses.items():
            tx, ty, tz = pose_data['translation']
            qw, qx, qy, qz = pose_data['quaternion']

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

            R_inv = R.transpose()
            t_inv = -R_inv @ np.array([tx, ty, tz])

            position_blob = np.array(t_inv, dtype=np.float64).tobytes()
            covariance_blob = (np.eye(3, dtype=np.float64) * 0.01).tobytes()
            coordinate_system = 1

            if use_corr_schema:
                # COLMAP 4.x rig schema: corr_data_id = image_id,
                # corr_sensor_type = 0 (camera), corr_sensor_id = camera_id
                img_info = name_to_image.get(pose_data['name'])
                if img_info is None:
                    print(f"Warning: '{pose_data['name']}' not in DB, skipping")
                    continue
                cursor.execute("""
                    INSERT INTO pose_priors
                    (corr_data_id, corr_sensor_type, corr_sensor_id,
                     position, coordinate_system, position_covariance)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (  # nosemgrep
                    img_info['image_id'], 0, img_info['camera_id'],
                    position_blob, coordinate_system, covariance_blob,
                ))
            elif use_image_id_schema:
                # COLMAP 4.x early: image_id PK
                img_info = name_to_image.get(pose_data['name'])
                if img_info is None:
                    print(f"Warning: '{pose_data['name']}' not in DB, skipping")
                    continue
                cursor.execute("""
                    INSERT OR REPLACE INTO pose_priors
                    (image_id, position, coordinate_system, position_covariance)
                    VALUES (?, ?, ?, ?)
                """, (  # nosemgrep
                    img_info['image_id'],
                    position_blob, coordinate_system, covariance_blob,
                ))
            else:
                # COLMAP 3.x: image_name text PK
                cursor.execute("""
                    INSERT OR REPLACE INTO pose_priors
                    (image_name, position, coordinate_system,
                     position_covariance)
                    VALUES (?, ?, ?, ?)
                """, (  # nosemgrep
                    pose_data['name'],
                    position_blob, coordinate_system, covariance_blob,
                ))
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        print(f"Successfully updated {len(image_poses)} images with pose priors")
        return True
        
    except Exception as e:
        print(f"Warning (non-fatal): Could not update pose priors in COLMAP database: {str(e)}")
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

            # Debug: Print available names and what we're looking for
            print(f"Available image names from transforms: {sorted(image_names)[:5]}...")  # First 5
            print(f"COLMAP image names: {sorted(colmap_image_names)[:5]}...")  # First 5
            print(f"Looking for: {colmap_image_names[0] if colmap_image_names else 'None'}")
            
            for img_name in colmap_image_names:
                if img_name not in name_to_idx:
                    print(f"WARNING: Image '{img_name}' not found in transforms data, trying base name match...")
                    base_name = os.path.splitext(img_name)[0]
                    found_match = False
                    for transform_name in name_to_idx.keys():
                        if os.path.splitext(transform_name)[0] == base_name:
                            print(f"Found match: {transform_name} for {img_name}")
                            idx = name_to_idx[transform_name]
                            reordered_poses.append(poses[idx])
                            reordered_cameras.append(cameras[idx])
                            found_match = True
                            break
                    if not found_match:
                        print("ERROR: No match found for", img_name)
                        raise KeyError("Image '" + img_name + "' not found")
                else:
                    idx = name_to_idx[img_name]
                    reordered_poses.append(poses[idx])
                    reordered_cameras.append(cameras[idx])

            # Write reconstruction files with correct order
            write_cameras_file(reordered_cameras, sparse_path, colmap_db_path),
            write_images_file(
                colmap_image_names,
                reordered_poses,
                sparse_path,
                source_coord_name,
                pose_is_world_to_cam
            )
            create_empty_points3d_file(sparse_path)
            # Remove binary files so COLMAP loads the text files we just wrote
            # and to avoid rig-format conflicts with COLMAP 4.x
            for _bin in (
                "images.bin", "cameras.bin", "points3D.bin",
                "rigs.bin", "frames.bin",
                "rigs.txt", "frames.txt",
            ):
                _bp = os.path.join(sparse_path, _bin)
                if os.path.exists(_bp):
                    os.remove(_bp)
                    print(f"Removed {_bin} to avoid rig-format conflicts")
            # Rewrite cameras.txt and images.txt so IDs match the
            # database (which has a single shared camera).  The poses
            # written above are preserved; only the numeric IDs change.
            sync_images_txt_with_db(colmap_db_path, image_path, sparse_path)
        # Using spatial matcher for pose priors requires colmap db to be updated with poses
        update_colmap_db_with_pose_priors(colmap_db_path, os.path.join(sparse_path, "images.txt"))
        clear_rig_tables(colmap_db_path)
    except Exception as e:
        raise RuntimeError("Error converting pose priors: " + str(e)) from e