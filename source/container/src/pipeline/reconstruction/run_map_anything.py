# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Run Map-Anything to recover poses and point cloud from images"""

import argparse
import subprocess
import sys
import shutil
import os
from pathlib import Path

def run_map_anything(scene_dir: str, skip_point2d: bool = False, voxel_size: str = "0.01"):
    """
    Run Map-Anything SfM pipeline
    
    Args:
        scene_dir: Path to scene directory containing images
        skip_point2d: Skip Point2D backprojection for faster export
        voxel_size: Explicit voxel size in meters
    """
    # Set PYTHONPATH to use map-anything specific pycolmap 3.10.0
    env = os.environ.copy()
    mapanything_pycolmap = "/opt/mapanything_pycolmap"
    if os.path.exists(mapanything_pycolmap):
        env['PYTHONPATH'] = f"{mapanything_pycolmap}:{env.get('PYTHONPATH', '')}"
        print(f"Using map-anything pycolmap 3.10.0 from: {mapanything_pycolmap}")
    
    images_dir = os.path.join(scene_dir, "images")
    output_dir = os.path.join(scene_dir, "sparse")
    
    cmd = [
        "python", "map-anything/scripts/demo_colmap.py",
        f"--images_dir={images_dir}",
        f"--output_dir={output_dir}"
    ]
    
    if skip_point2d:
        cmd.append("--skip_point2d")
    
    cmd.append(f"--voxel_size={voxel_size}")
    
    print(f"Running Map-Anything: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, env=env)
    
    # Debug: Check what was created
    sparse_dir = Path(scene_dir) / "sparse"
    sparse_0_dir = sparse_dir / "0"
    nested_sparse_dir = sparse_dir / "sparse"
    
    print(f"\n=== DEBUG: Map-Anything Output ===")
    print(f"Scene dir: {scene_dir}")
    print(f"Sparse dir exists: {sparse_dir.exists()}")
    if sparse_dir.exists():
        print(f"Files in sparse/: {[f.name for f in sparse_dir.iterdir()]}")
    
    # MapAnything outputs to scene_dir/sparse/sparse/, we need it in scene_dir/sparse/0/
    if nested_sparse_dir.exists():
        # Create sparse/0 if it doesn't exist
        sparse_0_dir.mkdir(parents=True, exist_ok=True)
        
        # Move COLMAP files from sparse/sparse/ to sparse/0/
        colmap_files = ['cameras.bin', 'images.bin', 'points3D.bin']
        for filename in colmap_files:
            src = nested_sparse_dir / filename
            if src.exists():
                dst = sparse_0_dir / filename
                shutil.move(str(src), str(dst))
                print(f"  Moved {filename} from sparse/sparse/ to sparse/0/")
        
        # Handle points.ply
        points_ply = nested_sparse_dir / "points.ply"
        if points_ply.exists():
            sparse_ply = sparse_0_dir / "sparse.ply"
            shutil.copy(str(points_ply), str(sparse_ply))
            print(f"  Copied points.ply to sparse/0/sparse.ply")
        
        # Remove empty nested sparse directory
        if nested_sparse_dir.exists() and not any(nested_sparse_dir.iterdir()):
            nested_sparse_dir.rmdir()
            print(f"  Removed empty nested sparse directory")
    
    if sparse_0_dir.exists():
        files = list(sparse_0_dir.iterdir())
        print(f"Files in sparse/0/: {[f.name for f in files]}")
    
    # Map-anything saves processed images to sparse/images
    # Replace original images with processed ones to match camera parameters
    processed_images_dir = sparse_dir / "images"
    original_images_dir = Path(scene_dir) / "images"
    
    if processed_images_dir.exists() and processed_images_dir != original_images_dir:
        print(f"\nReplacing original images with map-anything processed images")
        # Backup original images
        backup_dir = Path(scene_dir) / "images_original"
        if original_images_dir.exists():
            shutil.move(str(original_images_dir), str(backup_dir))
            print(f"  Backed up original images to: {backup_dir}")
        
        # Move processed images to expected location
        shutil.move(str(processed_images_dir), str(original_images_dir))
        print(f"  Moved processed images to: {original_images_dir}")
        print(f"  Image count: {len(list(original_images_dir.glob('*.[jp][pn]g')))}")
    
    print(f"=== END DEBUG ===")
    
    return result.returncode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Map-Anything SfM pipeline")
    parser.add_argument("--scene_dir", type=str, required=True, help="Path to scene directory")
    parser.add_argument("--skip_point2d", action="store_true", default=False, help="Skip Point2D backprojection for faster export")  
    parser.add_argument("--voxel_size", type=str, default="0.01", help="Explicit voxel size in meters")  
    args = parser.parse_args()
    
    sys.exit(run_map_anything(args.scene_dir, args.skip_point2d, args.voxel_size))