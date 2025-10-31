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
    cmd = [
        "python", "map-anything/scripts/demo_colmap.py",
        f"--scene_dir={scene_dir}",
        "--shared_camera"
    ]
    
    if memory_efficient_inference:
        cmd.append("--memory_efficient_inference")
    
    if use_ba:
        cmd.append("--use_ba")
    
    print(f"Running Map-Anything: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
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