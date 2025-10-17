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
Run VGGT and bundle adjustment using Colmap
"""

import os
import time
import shutil
import argparse
import subprocess

def main():
    # Start total script timer
    script_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Run VGGT on all images in folder")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--conf_thres_value", type=float, default=0.0)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    
    # Use input_dir as both input and output
    output_dir = args.input_dir
    
    # Look for images folder within the workspace
    images_folder = os.path.join(args.input_dir, "images")
    if not os.path.exists(images_folder):
        print(f"No images folder found in {args.input_dir}")
        return
    
    # Get all image files from images folder
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    all_images = [f for f in os.listdir(images_folder) if f.lower().endswith(image_extensions)]
    all_images.sort()
    
    if not all_images:
        print(f"No images found in {images_folder}")
        return
    
    print(f"Found {len(all_images)} total images")
    
    # Clean up any previous outputs in input directory
    final_sparse_dir = os.path.join(output_dir, "sparse")
    if os.path.exists(final_sparse_dir):
        shutil.rmtree(final_sparse_dir)
    
    print(f"Processing {len(all_images)} images in place")
    
    # Start VGGT timer
    vggt_start_time = time.time()

    # Run VGGT on all images
    print("Running VGGT on all images...")
    demo_cmd = [
        "python", "vggt/demo_colmap.py",
        "--scene_dir", output_dir,
        "--conf_thres_value", str(args.conf_thres_value),
        "--max_query_pts", "2048",
        "--shared_camera"#,
        #"--use_ba"
    ]
    
    try:
        subprocess.run(demo_cmd, check=True, cwd=os.getcwd())
        vggt_time = time.time() - vggt_start_time
        print(f"VGGT completed in {vggt_time:.2f}s ({vggt_time/60:.1f}m)")
    except subprocess.CalledProcessError as e:
        print(f"VGGT failed: {e}")
        return
    
    # Move COLMAP files to sparse/0/ subdirectory
    sparse_dir = os.path.join(output_dir, "sparse")
    sparse_0_dir = os.path.join(sparse_dir, "0")
    if not os.path.exists(sparse_0_dir):
        os.makedirs(sparse_0_dir)
        
        # Move all files to sparse/0/
        for file in os.listdir(sparse_dir):
            if os.path.isfile(os.path.join(sparse_dir, file)):
                src = os.path.join(sparse_dir, file)
                dst = os.path.join(sparse_0_dir, file)
                shutil.move(src, dst)
        print("Moved COLMAP files to sparse/0/")
    
    print("VGGT reconstruction completed")
    
    # Run training if requested
    if args.train:
        print("Starting nerfstudio training...")
        
        # Convert COLMAP to nerfstudio
        print("Converting COLMAP to nerfstudio format...")
        colmap_convert_cmd = ["python", "/mnt/efs/colmap_to_nerfstudio_cam.py", "-d", output_dir]
        try:
            subprocess.run(colmap_convert_cmd, check=True)
            print("COLMAP conversion completed")
        except subprocess.CalledProcessError as e:
            print(f"COLMAP conversion failed: {e}")
            return
        
        # Run nerfstudio training
        print("Starting splatfacto training...")
        train_cmd = [
            "ns-train", "splatfacto",
            "--max-num-iterations", "15000",
            "--viewer.quit-on-train-completion=True",
            "--timestamp", "train-stage-1",
            "--data", output_dir
        ]
        try:
            subprocess.run(train_cmd, check=True)
            print("Training completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Training failed: {e}")
            return
        
        # Export Splat
        export_cmd = [
            "ns-export",
            "gaussian-splat",
            "--load-config", "outputs/unnamed/splatfacto/train-stage-1/config.yml",
            "--output-dir", os.path.join(output_dir, "exports")
        ]
        try:
            subprocess.run(export_cmd, check=True)
            print("Export completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Export failed: {e}")
            return
    
    # Calculate and display total script time
    total_script_time = time.time() - script_start_time
    print(f"\nTotal script elapsed time: {total_script_time:.2f}s ({total_script_time/60:.1f}m)")

if __name__ == "__main__":
    main()
