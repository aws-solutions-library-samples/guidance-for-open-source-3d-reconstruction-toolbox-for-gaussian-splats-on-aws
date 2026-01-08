#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Demo script to get MapAnything outputs in COLMAP format. Optionally can also run BA on outputs.

Reference: VGGT (https://github.com/facebookresearch/vggt/blob/main/demo_colmap.py)
"""

import argparse
import os
import glob
import copy

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.image import load_images, rgb
from mapanything.utils.geometry import closed_form_pose_inverse, depthmap_to_world_frame
from mapanything.third_party.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
)
from mapanything.third_party.track_predict import predict_tracks

def get_parser():
    parser = argparse.ArgumentParser(description="Memory Efficient MapAnything COLMAP Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--apache", action="store_true", default=True, help="Use Apache 2.0 licensed model")
    parser.add_argument("--memory_efficient_inference", action="store_true", default=True, help="Use memory efficient inference")
    parser.add_argument("--conf_thres_value", type=float, default=0.0, help="Confidence threshold for depth filtering")
    parser.add_argument("--shared_camera", action="store_true", default=True, help="Use shared camera for all images")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use bundle adjustment for reconstruction")
    parser.add_argument("--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for BA")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument("--fine_tracking", action="store_true", default=True, help="Use fine tracking")
    # Quality improvement options
    parser.add_argument("--apply_mask", action="store_true", default=True, help="Apply masking to dense geometry outputs")
    parser.add_argument("--mask_edges", action="store_true", default=True, help="Remove edge artifacts using normals and depth")
    parser.add_argument("--apply_confidence_mask", action="store_true", default=True, help="Filter low-confidence regions")
    parser.add_argument("--confidence_percentile", type=int, default=25, help="Remove bottom N percentile confidence pixels")
    parser.add_argument("--use_bf16", action="store_true", default=True, help="Use bfloat16 precision (better quality, more memory)")
    return parser

def create_pixel_coordinate_grid(num_frames, height, width):
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]
    
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))
    
    f_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))
    
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)
    return points_xyf

def write_poses_to_images_txt(sparse_dir, extrinsics, image_names, shared_camera=False):
    """Manually write camera poses to images.txt in COLMAP format"""
    def rotation_matrix_to_quaternion(R):
        """Convert rotation matrix to quaternion [w, x, y, z]"""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return np.array([w, x, y, z])
    
    images_txt_path = os.path.join(sparse_dir, "images.txt")
    with open(images_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(extrinsics)}, mean observations per image: 0\n")
        
        for i, (extrinsic, image_name) in enumerate(zip(extrinsics, image_names)):
            # Extract rotation and translation
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            
            # Convert rotation matrix to quaternion (w, x, y, z)
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R)
            
            # Use camera ID 1 for shared camera, otherwise use image ID
            camera_id = 1 if shared_camera else i + 1
            
            # Write image line
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {camera_id} {image_name}\n")
            f.write("\n")  # empty points2D line

def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    true_indices = np.flatnonzero(mask)
    if true_indices.size <= max_trues:
        return mask
    
    sampled_indices = np.random.choice(true_indices, size=max_trues, replace=False)
    limited_flat_mask = np.zeros(mask.size, dtype=bool)
    limited_flat_mask[sampled_indices] = True
    return limited_flat_mask.reshape(mask.shape)

def get_original_image_coords(image_paths, target_size=518):
    """Get original image coordinates for rescaling"""
    original_coords = []
    for image_path in image_paths:
        img = Image.open(image_path)
        width, height = img.size
        
        # Calculate padding and scaling (same logic as load_images)
        max_dim = max(width, height)
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2
        scale = target_size / max_dim
        
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale
        
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))
    
    return np.array(original_coords)

def rename_colmap_recons_and_rescale_camera(reconstruction, image_paths, original_coords, img_size, shared_camera=False):
    rescale_camera = True
    
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = os.path.basename(image_paths[pyimageid - 1])
        
        if rescale_camera:
            # Rescale camera parameters
            pred_params = copy.deepcopy(pycamera.params)
            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp
            
            pycamera.params = pred_params
            pycamera.width = int(real_image_size[0])
            pycamera.height = int(real_image_size[1])
        
        if shared_camera:
            # If shared_camera, all images share the same camera
            rescale_camera = False
    
    return reconstruction

def main():
    args = get_parser().parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    if args.apache:
        model_name = "facebook/map-anything-apache"
        print("Loading Apache 2.0 licensed MapAnything model...")
    else:
        model_name = "facebook/map-anything"
        print("Loading CC-BY-NC 4.0 licensed MapAnything model...")
    
    model = MapAnything.from_pretrained(model_name).to(device)
    
    # Get image paths
    image_dir = os.path.join(args.scene_dir, "images")
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"Loading {len(image_paths)} images from: {image_dir}")
    
    # Load images using the memory-efficient approach
    views = load_images(image_dir)
    print(f"Loaded {len(views)} views")
    
    # Get original coordinates for rescaling
    original_coords = get_original_image_coords(image_paths)
    
    # Run inference with quality settings
    print("Running inference...")
    amp_dtype = "bf16" if args.use_bf16 and torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    
    outputs = model.infer(
        views, 
        memory_efficient_inference=args.memory_efficient_inference,
        use_amp=True,
        amp_dtype=amp_dtype,
        apply_mask=args.apply_mask,
        mask_edges=args.mask_edges,
        apply_confidence_mask=args.apply_confidence_mask,
        confidence_percentile=args.confidence_percentile,
    )
    print("Inference complete!")
    
    # Process outputs to COLMAP format
    all_extrinsics = []
    all_intrinsics = []
    all_depth_maps = []
    all_depth_confs = []
    all_pts3d = []
    all_masks = []
    
    for pred in outputs:
        # Extract data
        depthmap_torch = pred["depth_z"][0].squeeze(-1)
        intrinsics_torch = pred["intrinsics"][0]
        camera_pose_torch = pred["camera_poses"][0]
        
        # Compute 3D points
        pts3d, valid_mask = depthmap_to_world_frame(depthmap_torch, intrinsics_torch, camera_pose_torch)
        
        # Extract enhanced masks for better quality
        base_mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = base_mask & valid_mask.cpu().numpy()
        
        # Apply additional quality filters if available
        if "non_ambiguous_mask" in pred:
            non_ambiguous = pred["non_ambiguous_mask"][0].cpu().numpy().astype(bool)
            mask = mask & non_ambiguous
            
        # Apply confidence-based filtering if enabled
        if args.apply_confidence_mask:
            conf = pred["conf"][0].cpu().numpy()
            conf_threshold = np.percentile(conf[mask], args.confidence_percentile)
            conf_mask = conf >= conf_threshold
            mask = mask & conf_mask
        
        # Convert to numpy
        extrinsic = closed_form_pose_inverse(pred["camera_poses"])[0].cpu().numpy()
        intrinsic = intrinsics_torch.cpu().numpy()
        depth_map = depthmap_torch.cpu().numpy()
        depth_conf = pred["conf"][0].cpu().numpy()
        pts3d_np = pts3d.cpu().numpy()
        
        all_extrinsics.append(extrinsic)
        all_intrinsics.append(intrinsic)
        all_depth_maps.append(depth_map)
        all_depth_confs.append(depth_conf)
        all_pts3d.append(pts3d_np)
        all_masks.append(mask)
    
    # Stack arrays
    all_extrinsics = np.stack(all_extrinsics)
    all_intrinsics = np.stack(all_intrinsics)
    all_depth_maps = np.stack(all_depth_maps)
    all_depth_confs = np.stack(all_depth_confs)
    all_pts3d = np.stack(all_pts3d)
    all_masks = np.stack(all_masks)
    
    # Create COLMAP reconstruction
    print("Converting to COLMAP format...")
    
    if args.use_ba:
        # Bundle adjustment path with reduced parameters for memory efficiency
        from torchvision import transforms as tvf
        from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
        
        # Use reduced resolution for memory efficiency
        img_load_resolution = 800  # Reduced from 1024
        images_for_tracking = []
        
        img_norm = IMAGE_NORMALIZATION_DICT[model.encoder.data_norm_type]
        img_transform = tvf.Compose([
            tvf.ToTensor(), 
            tvf.Normalize(mean=img_norm.mean, std=img_norm.std)
        ])
        
        for image_path in image_paths:
            img = Image.open(image_path).convert('RGB')
            width, height = img.size
            max_dim = max(width, height)
            left = (max_dim - width) // 2
            top = (max_dim - height) // 2
            
            square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
            square_img.paste(img, (left, top))
            square_img = square_img.resize((img_load_resolution, img_load_resolution), Image.Resampling.BICUBIC)
            
            img_tensor = img_transform(square_img)
            images_for_tracking.append(img_tensor)
        
        images_tensor = torch.stack(images_for_tracking).to(device)
        
        # Rescale intrinsics for tracking resolution
        mapanything_fixed_resolution = 518
        scale = img_load_resolution / mapanything_fixed_resolution
        all_intrinsics[:, :2, :] *= scale
        
        image_size = np.array([img_load_resolution, img_load_resolution])
        
        # Use reduced parameters for memory efficiency
        reduced_max_query_pts = min(args.max_query_pts, 2048)
        reduced_query_frame_num = min(args.query_frame_num, 5)
        
        print(f"Using reduced BA parameters: max_query_pts={reduced_max_query_pts}, query_frame_num={reduced_query_frame_num}")
        
        with torch.amp.autocast("cuda", dtype=torch.float16):
            # Predict tracks using VGGSfM tracker with reduced parameters
            pred_tracks, pred_vis_scores, pred_confs, points_3d_ba, points_rgb_ba = predict_tracks(
                images_tensor,
                conf=all_depth_confs,
                points_3d=all_pts3d,
                max_query_pts=reduced_max_query_pts,
                query_frame_num=reduced_query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )
            torch.cuda.empty_cache()
        
        track_mask = pred_vis_scores > args.vis_thresh
        
        # Create COLMAP reconstruction with tracks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d_ba,
            all_extrinsics,
            all_intrinsics,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=args.shared_camera,
            camera_type="SIMPLE_PINHOLE" if args.shared_camera else "PINHOLE",
            points_rgb=points_rgb_ba,
        )
        
        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")
        
        # Bundle Adjustment
        print("Running bundle adjustment...")
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)
        
        reconstruction_resolution = img_load_resolution
        
    else:
        # Feed-forward only path (original logic)
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 100000
        mapanything_fixed_resolution = 518
        
        num_frames, height, width, _ = all_pts3d.shape
        image_size = np.array([mapanything_fixed_resolution, mapanything_fixed_resolution])
        
        # Create RGB colors for points
        points_rgb_list = []
        for i, pred in enumerate(outputs):
            img_no_norm = pred["img_no_norm"][0].cpu().numpy()
            points_rgb_list.append(img_no_norm)
        
        points_rgb = np.stack(points_rgb_list)
        points_rgb = (points_rgb * 255).astype(np.uint8)
        
        # Create pixel coordinate grid
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
        
        # Apply enhanced confidence filtering
        if args.apply_confidence_mask:
            # Use percentile-based filtering for better quality
            valid_confs = all_depth_confs[all_masks]
            if len(valid_confs) > 0:
                conf_threshold = np.percentile(valid_confs, args.confidence_percentile)
                conf_mask = all_depth_confs >= max(conf_threshold, conf_thres_value)
            else:
                conf_mask = all_depth_confs >= conf_thres_value
        else:
            conf_mask = all_depth_confs >= conf_thres_value
            
        # Combine with existing masks for better quality
        final_mask = conf_mask & all_masks
        final_mask = randomly_limit_trues(final_mask, max_points_for_colmap)
        
        points_3d_filtered = all_pts3d[final_mask]
        points_xyf_filtered = points_xyf[final_mask]
        points_rgb_filtered = points_rgb[final_mask]
        
        # Create COLMAP reconstruction
        reconstruction, extrinsics_for_images, shared_camera_flag = batch_np_matrix_to_pycolmap_wo_track(
            points_3d_filtered,
            points_xyf_filtered,
            points_rgb_filtered,
            all_extrinsics,
            all_intrinsics,
            image_size,
            shared_camera=args.shared_camera,
            camera_type="PINHOLE",
        )
        
        reconstruction_resolution = mapanything_fixed_resolution
    
    # Rescale cameras to match original image dimensions
    for camera_id in reconstruction.cameras:
        camera = reconstruction.cameras[camera_id]
        # Get original image size for this camera
        # For shared camera, use first image; otherwise use corresponding image
        img_idx = 0 if args.shared_camera else (camera_id - 1)
        real_image_size = original_coords[img_idx, -2:]
        
        # Rescale camera parameters
        resize_ratio = max(real_image_size) / reconstruction_resolution
        camera.params = camera.params * resize_ratio
        
        # Update principal point to image center
        real_pp = real_image_size / 2
        camera.params[-2:] = real_pp
        
        # Update camera dimensions
        camera.width = int(real_image_size[0])
        camera.height = int(real_image_size[1])
    
    # Save reconstruction
    print(f"Saving reconstruction to {args.scene_dir}/sparse")
    sparse_dir = os.path.join(args.scene_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    
    # Write everything as text files (cameras now have correct rescaled parameters)
    reconstruction.write_text(sparse_dir)
    
    # Manually overwrite images.txt with poses
    write_poses_to_images_txt(
        sparse_dir, 
        extrinsics_for_images, 
        [os.path.basename(p) for p in image_paths],
        shared_camera_flag
    )
    print("Wrote images.txt manually")
    
    # Convert all text files to binary format using COLMAP
    import subprocess
    try:
        subprocess.run([
            "colmap", "model_converter",
            "--input_path", sparse_dir,
            "--output_path", sparse_dir,
            "--output_type", "BIN"
        ], check=True, capture_output=True)
        print("Converted to binary format")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to convert to binary: {e.stderr.decode() if e.stderr else str(e)}")
    
    print("COLMAP reconstruction saved successfully!")
    
    # Save point cloud
    if not args.use_ba:
        trimesh.PointCloud(points_3d_filtered, colors=points_rgb_filtered).export(
            os.path.join(sparse_dir, "sparse.ply")
        )
    
    print("Reconstruction complete!")


if __name__ == "__main__":
    main()