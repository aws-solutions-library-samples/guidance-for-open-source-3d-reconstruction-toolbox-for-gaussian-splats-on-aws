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

def run_map_anything(scene_dir: str, memory_efficient_inference: bool = True, use_ba: bool = False):
    """
    Run Map-Anything SfM pipeline
    
    Args:
        scene_dir: Path to scene directory containing images
        memory_efficient_inference: Use memory efficient inference (slower but handles more images)
        use_ba: Use bundle adjustment for refinement
    """
    # Set PYTHONPATH to use map-anything specific pycolmap 3.10.0
    env = os.environ.copy()
    mapanything_pycolmap = "/opt/mapanything_pycolmap"
    if os.path.exists(mapanything_pycolmap):
        env['PYTHONPATH'] = f"{mapanything_pycolmap}:{env.get('PYTHONPATH', '')}"
        print(f"Using map-anything pycolmap 3.10.0 from: {mapanything_pycolmap}")
    
    cmd = [
        "python", "map-anything/scripts/demo_colmap.py",
        f"--scene_dir={scene_dir}",
        "--use_ba",
        "--max_query_pts=2048", #orig 4096,2048
        "--query_frame_num=6", #orig 8,5
        "--shared_camera"
    ]
    
    if memory_efficient_inference:
        cmd.append("--memory_efficient_inference")
    
    if use_ba:
        cmd.append("--use_ba")
    
    print(f"Running Map-Anything: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        if use_ba:
            print(f"Map-Anything failed with bundle adjustment, retrying without BA...")
            cmd.remove("--use_ba")
            result = subprocess.run(cmd, check=True, env=env)
        else:
            raise
    
    # Debug: Check what was created
    sparse_dir = Path(scene_dir) / "sparse"
    sparse_0_dir = sparse_dir / "0"
    
    print(f"\n=== DEBUG: Map-Anything Output ===")
    print(f"Scene dir: {scene_dir}")
    print(f"Sparse dir exists: {sparse_dir.exists()}")
    if sparse_dir.exists():
        print(f"Files in sparse/: {list(sparse_dir.iterdir())}")
    
    print(f"Sparse/0 dir exists: {sparse_0_dir.exists()}")
    if sparse_0_dir.exists():
        files = list(sparse_0_dir.iterdir())
        print(f"Files in sparse/0/: {[f.name for f in files]}")
        for f in files:
            if f.is_file():
                print(f"  {f.name}: {f.stat().st_size} bytes")
    
    # Move COLMAP files from sparse/ to sparse/0/ if needed
    colmap_files = ['cameras.bin', 'images.bin', 'points3D.bin']
    files_in_root = [f for f in colmap_files if (sparse_dir / f).exists()]
    
    if files_in_root:
        print(f"\nMoving COLMAP files from sparse/ to sparse/0/")
        os.makedirs(sparse_0_dir, exist_ok=True)
        for filename in colmap_files:
            src = sparse_dir / filename
            dst = sparse_0_dir / filename
            if src.exists():
                shutil.move(str(src), str(dst))
                print(f"  Moved {filename}")
        
        # Verify files were moved
        print(f"\nAfter move - Files in sparse/0/: {[f.name for f in sparse_0_dir.iterdir()]}")
        
        # Verify image names in COLMAP model match actual images
        try:
            import pycolmap
            reconstruction = pycolmap.Reconstruction(str(sparse_0_dir))
            images_dir = Path(scene_dir) / "images"
            actual_images = set([f.name for f in images_dir.glob("*.[jp][pn]g")])
            colmap_images = set([img.name for img in reconstruction.images.values()])
            
            print(f"\nImage verification:")
            print(f"  Actual images in images/: {len(actual_images)}")
            print(f"  Images in COLMAP model: {len(colmap_images)}")
            
            missing = colmap_images - actual_images
            if missing:
                print(f"  WARNING: {len(missing)} images in COLMAP model not found in images/ directory")
                print(f"  Sample missing: {list(missing)[:3]}")
                print(f"  Sample actual: {list(actual_images)[:3]}")
        except Exception as e:
            print(f"  Could not verify image names: {e}")
    
    images_dir = Path(scene_dir) / "images"
    if images_dir.exists():
        image_files = list(images_dir.glob("*.[jp][pn]g"))
        print(f"Images in images/: {len(image_files)} files")
        if image_files:
            print(f"  Sample: {image_files[0].name}")
    print(f"=== END DEBUG ===")
    
    return result.returncode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Map-Anything SfM pipeline")
    parser.add_argument("--scene_dir", type=str, required=True, help="Path to scene directory")
    parser.add_argument("--memory_efficient_inference", action="store_true", default=True, help="Use memory efficient inference")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use bundle adjustment")
    
    args = parser.parse_args()
    
    sys.exit(run_map_anything(args.scene_dir, args.memory_efficient_inference, args.use_ba))