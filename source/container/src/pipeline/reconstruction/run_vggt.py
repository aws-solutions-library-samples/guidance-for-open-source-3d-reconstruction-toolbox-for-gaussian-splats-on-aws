import torch
import argparse
import os
import re
import shutil
import subprocess
import numpy as np
import json

def resample_images(image_paths, max_count):
    """Take the first max_count images to avoid filename mismatches."""
    if len(image_paths) <= max_count:
        return image_paths
    
    def extract_number(path):
        filename = os.path.basename(path)
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0
    
    sorted_paths = sorted(image_paths, key=extract_number)
    return sorted_paths[:max_count]

def main():
    parser = argparse.ArgumentParser(description="Run VGGT on images and export COLMAP")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to workspace directory (should contain 'images' folder)")
    parser.add_argument("--max_images", type=int, default=25, help="Maximum number of images to process (default: 25)")
    parser.add_argument("--conf_thres_value", type=float, default=0.0, help="Confidence threshold for depth filtering (default: 0.0)")
    parser.add_argument("--use_ba", action="store_true", help="Run bundle adjustment (default: False)")
    parser.add_argument("--train", action="store_true", help="Run nerfstudio training after VGGT processing")
    args = parser.parse_args()
    
    # Get all image files from the images subdirectory
    images_input_dir = os.path.join(args.input_dir, "images")
    if not os.path.exists(images_input_dir):
        print(f"Error: 'images' folder not found in {args.input_dir}")
        return
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_names = [os.path.join(images_input_dir, f) for f in os.listdir(images_input_dir) 
                   if f.lower().endswith(image_extensions)]
    image_names.sort()
    
    if not image_names:
        print(f"No images found in {images_input_dir}")
        return
    
    print(f"Found {len(image_names)} images")
    
    # Resample if too many images
    if len(image_names) > args.max_images:
        image_names = resample_images(image_names, args.max_images)
        print(f"Resampled to {len(image_names)} images")
    
    # Use input_dir as workspace
    workspace_dir = args.input_dir
    
    # Clean up any previous outputs
    transforms_json = os.path.join(workspace_dir, "transforms.json")
    if os.path.exists(transforms_json):
        os.remove(transforms_json)
        print("Removed previous transforms.json")
    
    sparse_dir = os.path.join(workspace_dir, "sparse")
    if os.path.exists(sparse_dir):
        shutil.rmtree(sparse_dir)
        print("Removed previous sparse directory")
    
    # Images are already in the correct location
    images_dir = images_input_dir
    print(f"Using {len(image_names)} images from {images_dir}")
    
    # Clear GPU memory before running VGGT
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Run VGGT
    print("Running VGGT COLMAP demo...")
    demo_cmd = [
        "python", "vggt/demo_colmap.py", 
        "--scene_dir", workspace_dir,
        "--conf_thres_value", str(args.conf_thres_value),
        #"--max_query_pts", "256",
        #"--query_frame_num", "5",
        "--shared_camera"
    ]
    
    if args.use_ba:
        demo_cmd.append("--use_ba")
    
    try:
        result = subprocess.run(demo_cmd, check=True, cwd=os.getcwd(), capture_output=True, text=True)
        print("VGGT COLMAP processing completed")
    except subprocess.CalledProcessError as e:
        print(f"VGGT COLMAP processing failed: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"STDERR: {e.stderr}")
        return
    
    # Aggressive GPU memory cleanup after VGGT
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        print(f"GPU memory cleared after VGGT. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
    # Check if COLMAP files were created and move to sparse/0/ subdirectory
    sparse_dir = os.path.join(workspace_dir, "sparse")
    if os.path.exists(sparse_dir):
        colmap_files = os.listdir(sparse_dir)
        print(f"COLMAP files created: {colmap_files}")
        
        # Move COLMAP files to sparse/0/ subdirectory as expected by conversion script
        sparse_0_dir = os.path.join(sparse_dir, "0")
        if not os.path.exists(sparse_0_dir):
            os.makedirs(sparse_0_dir)
            
            # Move .bin files to sparse/0/
            for file in colmap_files:
                if file.endswith('.bin'):
                    src = os.path.join(sparse_dir, file)
                    dst = os.path.join(sparse_0_dir, file)
                    shutil.move(src, dst)
            print("Moved COLMAP files to sparse/0/ subdirectory")
    else:
        print("Warning: No sparse directory created")
        return
    
    # Run nerfstudio training if requested
    if args.train:
        print("Starting nerfstudio training...")
        
        # Run colmap_to_nerfstudio_cam.py
        print("Converting COLMAP to nerfstudio format...")
        colmap_convert_cmd = ["python", "/mnt/efs/colmap_to_nerfstudio_cam.py", "-d", workspace_dir]
        try:
            result = subprocess.run(colmap_convert_cmd, check=True, cwd=os.getcwd(), capture_output=True, text=True)
            print("COLMAP conversion completed")
        except subprocess.CalledProcessError as e:
            print(f"COLMAP conversion failed: {e}")
            return
        
        # Check if transforms.json was created
        transforms_path = os.path.join(workspace_dir, "transforms.json")
        if not os.path.exists(transforms_path):
            print(f"Error: transforms.json not found at {transforms_path}")
            return
            
        # Fix transforms.json to match actual images
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        
        # Get actual image files in directory
        actual_images = set(os.listdir(images_dir))
        
        # Filter frames to only include existing images
        original_frames = len(transforms.get('frames', []))
        valid_frames = []
        for frame in transforms.get('frames', []):
            image_name = os.path.basename(frame['file_path'])
            if image_name in actual_images:
                valid_frames.append(frame)
        
        # If insufficient valid frames, try to fix the file paths
        if len(valid_frames) <= 1 and transforms.get('frames'):
            print("Fixing file paths to match actual images...")
            actual_images_list = sorted(list(actual_images))
            valid_frames = []
            
            # Map each transform frame to an actual image by index
            for i, frame in enumerate(transforms['frames']):
                if i < len(actual_images_list):
                    frame['file_path'] = f"./images/{actual_images_list[i]}"
                    valid_frames.append(frame)
            print(f"Fixed {len(valid_frames)} frame paths")
        
        transforms['frames'] = valid_frames
        print(f"Final transforms.json: {original_frames} -> {len(transforms['frames'])} frames")
        
        # Check if we have enough frames for training
        if len(transforms['frames']) < 3:
            print(f"Error: Only {len(transforms['frames'])} valid frames found. Need at least 3 for training.")
            return
        
        # Check and fix camera pose scaling
        if transforms['frames']:
            positions = []
            for frame in transforms['frames']:
                transform_matrix = frame['transform_matrix']
                position = [transform_matrix[0][3], transform_matrix[1][3], transform_matrix[2][3]]
                positions.append(position)
            
            positions = np.array(positions)
            max_distance = np.max(np.abs(positions))
            
            # If all cameras are too close to origin, scale them up
            if max_distance < 0.1:
                scale_factor = 10.0
                print(f"Scaling camera poses by factor {scale_factor}")
                
                for frame in transforms['frames']:
                    frame['transform_matrix'][0][3] *= scale_factor
                    frame['transform_matrix'][1][3] *= scale_factor
                    frame['transform_matrix'][2][3] *= scale_factor
        
        # Save the fixed transforms.json
        with open(transforms_path, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        print("Fixed transforms.json saved")
        
        # Run nerfstudio training
        print("Starting splatfacto training...")
        train_cmd = [
            "ns-train", "splatfacto",
            "--max-num-iterations", "1000",
            "--viewer.quit-on-train-completion=True",
            "--timestamp", "train-stage-1",
            "--data", workspace_dir
        ]
        try:
            subprocess.run(train_cmd, check=True)
            print("Training completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Training failed: {e}")
            return
        
        # Export Splat
        workspace_name = os.path.basename(workspace_dir)
        config_path = f"outputs/{workspace_name}/splatfacto/train-stage-1/config.yml"
        export_cmd = [
            "ns-export",
            "gaussian-splat",
            "--load-config", config_path,
            "--output-dir",  "exports"
        ]
        try:
            subprocess.run(export_cmd, check=True)
            print("Export completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Export failed: {e}")
            return
    
    print("Processing complete!")

if __name__ == "__main__":
    main()