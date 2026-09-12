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

"""E57 to COLMAP Dataset Converter

Converts a pre-registered E57 point cloud (single merged scan with XYZ+RGB) into
a COLMAP-compatible dataset that can be fed directly into the Gaussian Splat pipeline:

    output_dir/
    ├── images/          # Rendered color PNG per virtual camera
    ├── depth_images/    # Rendered depth PNG (uint16, mm) per virtual camera
    └── colmap/
        └── sparse/
            └── 0/
                ├── cameras.txt
                ├── images.txt
                └── points3D.txt   # Subsampled colored point cloud

Virtual cameras are placed on a horizontal grid at multiple heights, all looking
inward toward the scene centroid.  The point cloud is projected through each
camera to produce color and depth renders.

Usage:
    python e57_to_colmap.py -i /path/to/scan.e57 -o /path/to/output \
        [--grid_xy 5] [--heights 1] [--image_wh 1920 1080] \
        [--fov_deg 90] [--max_points 2000000] [--depth_scale 1000]

Author: @eecorn
Date: 2025-06-01
Version: 1.0
"""

import os
import argparse
import time
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Point cloud I/O
# ---------------------------------------------------------------------------

def load_e57_points(e57_path: str, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Load XYZ and RGB from an E57 file, subsampling to at most max_points."""
    try:
        import pye57
    except ImportError as exc:
        raise ImportError("pye57 is required: pip install pye57") from exc

    e57 = pye57.E57(e57_path)
    print(f"E57 scan count: {e57.scan_count}")

    all_xyz, all_rgb = [], []
    for i in range(e57.scan_count):
        header = e57.get_header(i)
        print(f"  Scan {i}: {header.point_count:,} points")
        data = e57.read_scan_raw(i, ignore_unsupported_fields=True)

        xyz = np.column_stack([
            data["cartesianX"].astype(np.float32),
            data["cartesianY"].astype(np.float32),
            data["cartesianZ"].astype(np.float32),
        ])

        # Apply scanner pose if present
        R = np.array(header.rotation_matrix, dtype=np.float64)
        t = np.array(header.translation, dtype=np.float64).reshape(3)
        if not (np.allclose(R, np.eye(3)) and np.allclose(t, 0)):
            xyz = (R @ xyz.T).T + t

        if "colorRed" in data:
            rgb = np.column_stack([
                data["colorRed"].astype(np.uint8),
                data["colorGreen"].astype(np.uint8),
                data["colorBlue"].astype(np.uint8),
            ])
        else:
            gray = (data.get("intensity", np.ones(len(xyz), dtype=np.float32)) * 200).astype(np.uint8)
            rgb = np.column_stack([gray, gray, gray])

        all_xyz.append(xyz)
        all_rgb.append(rgb)

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)

    if len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
        print(f"Subsampled to {max_points:,} points")
    else:
        print(f"Loaded {len(xyz):,} points")

    return xyz, rgb


# ---------------------------------------------------------------------------
# Virtual camera placement
# ---------------------------------------------------------------------------

def place_cameras(xyz: np.ndarray, grid_xy: int, n_heights: int,
                  fov_deg: float, w: int, h: int) -> list[dict]:
    """
    Place virtual cameras on a horizontal grid at multiple heights, all
    looking toward the scene centroid.

    Returns a list of dicts with keys: R (3x3), t (3,), fx, fy, cx, cy, w, h
    """
    centroid = xyz.mean(axis=0)
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()

    # Margin so cameras sit just outside the bounding box
    margin = 0.15
    x_range = (x_max - x_min) * (1 + margin)
    y_range = (y_max - y_min) * (1 + margin)

    fx = fy = (w / 2) / np.tan(np.radians(fov_deg / 2))
    cx, cy = w / 2.0, h / 2.0

    cameras = []
    xs = np.linspace(x_min - (x_max - x_min) * margin / 2,
                     x_max + (x_max - x_min) * margin / 2, grid_xy)
    ys = np.linspace(y_min - (y_max - y_min) * margin / 2,
                     y_max + (y_max - y_min) * margin / 2, grid_xy)

    # Heights: evenly spaced between z_min+10% and z_max-10%
    z_pad = (z_max - z_min) * 0.1
    zs = np.linspace(z_min + z_pad, z_max - z_pad, max(1, n_heights))

    for z_cam in zs:
        for x_cam in xs:
            for y_cam in ys:
                cam_pos = np.array([x_cam, y_cam, z_cam], dtype=np.float64)

                # Look-at: forward = centroid - cam_pos
                forward = centroid - cam_pos
                dist = np.linalg.norm(forward)
                if dist < 1e-6:
                    continue
                forward /= dist

                # World up = +Z; if forward is nearly parallel to Z, use +Y
                world_up = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(forward, world_up)) > 0.99:
                    world_up = np.array([0.0, 1.0, 0.0])

                # OpenCV convention: X right, Y down, Z forward (into scene)
                # right = forward × world_up  (points right when looking forward)
                right = np.cross(forward, world_up)
                right /= np.linalg.norm(right)
                # down = forward × right  (Y points down in OpenCV)
                down = np.cross(forward, right)
                down /= np.linalg.norm(down)

                # Camera-to-world: columns are right, down, forward
                R_c2w = np.column_stack([right, down, forward])  # 3x3
                R_cw = R_c2w.T                                    # world-to-camera
                t_cw = -R_cw @ cam_pos                            # world-to-camera translation

                cameras.append({
                    "R": R_cw, "t": t_cw,
                    "pos": cam_pos,
                    "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                    "w": w, "h": h,
                })

    print(f"Placed {len(cameras)} virtual cameras "
          f"({grid_xy}x{grid_xy} grid × {n_heights} height(s))")
    return cameras


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_camera(xyz: np.ndarray, rgb: np.ndarray, cam: dict,
                  depth_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Project point cloud through a pinhole camera and rasterize color + depth.

    Returns:
        color_img: (H, W, 3) uint8 BGR
        depth_img: (H, W) uint16  (depth in mm when depth_scale=1000)
    """
    w, h = cam["w"], cam["h"]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    R, t = cam["R"], cam["t"]

    # Transform points to camera space
    pts_cam = (R @ xyz.T).T + t  # (N, 3)

    # Keep only points in front of camera
    valid = pts_cam[:, 2] > 0.01
    pts_cam = pts_cam[valid]
    pts_rgb = rgb[valid]

    if len(pts_cam) == 0:
        return (np.zeros((h, w, 3), dtype=np.uint8),
                np.zeros((h, w), dtype=np.uint16))

    # Project
    u = (pts_cam[:, 0] * fx / pts_cam[:, 2] + cx).astype(np.float32)
    v = (pts_cam[:, 1] * fy / pts_cam[:, 2] + cy).astype(np.float32)
    z = pts_cam[:, 2].astype(np.float32)

    # Clip to image bounds
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[in_bounds], v[in_bounds], z[in_bounds]
    pts_rgb = pts_rgb[in_bounds]

    ui = u.astype(np.int32)
    vi = v.astype(np.int32)

    # Z-buffer: keep nearest point per pixel
    depth_buf = np.full((h, w), np.inf, dtype=np.float32)
    color_buf = np.zeros((h, w, 3), dtype=np.uint8)

    # Sort by depth descending so nearest overwrites
    order = np.argsort(-z)
    ui, vi, z = ui[order], vi[order], z[order]
    pts_rgb = pts_rgb[order]

    depth_buf[vi, ui] = z
    color_buf[vi, ui] = pts_rgb[:, ::-1]  # RGB -> BGR for OpenCV

    # Convert depth to uint16 (mm)
    depth_valid = np.isfinite(depth_buf)
    depth_mm = np.zeros((h, w), dtype=np.uint16)
    depth_mm[depth_valid] = np.clip(
        depth_buf[depth_valid] * depth_scale, 0, 65535
    ).astype(np.uint16)

    return color_buf, depth_mm


# ---------------------------------------------------------------------------
# COLMAP file writers
# ---------------------------------------------------------------------------

def write_cameras_txt(cameras: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, cam in enumerate(cameras, 1):
            f.write(f"{i} PINHOLE {cam['w']} {cam['h']} "
                    f"{cam['fx']} {cam['fy']} {cam['cx']} {cam['cy']}\n")
    print(f"Wrote {len(cameras)} cameras → {out_path}")


def write_images_txt(cameras: list[dict], out_path: str) -> None:
    from scipy.spatial.transform import Rotation as R_
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, cam in enumerate(cameras, 1):
            rot = R_.from_matrix(cam["R"])
            qx, qy, qz, qw = rot.as_quat()  # scipy: x,y,z,w
            tx, ty, tz = cam["t"]
            name = f"{i:05d}.png"
            f.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {name}\n")
            f.write("\n")
    print(f"Wrote {len(cameras)} image poses → {out_path}")


def write_points3d_txt(xyz: np.ndarray, rgb: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i, (pt, col) in enumerate(zip(xyz, rgb), 1):
            f.write(f"{i} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{col[0]} {col[1]} {col[2]} 0.0\n")
    print(f"Wrote {len(xyz):,} points → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(e57_path: str, output_dir: str, grid_xy: int, n_heights: int,
            image_wh: tuple[int, int], fov_deg: float,
            max_points: int, depth_scale: float) -> None:

    t0 = time.time()
    w, h = image_wh

    images_dir = os.path.join(output_dir, "images")
    depth_dir = os.path.join(output_dir, "depth_images")
    sparse_dir = os.path.join(output_dir, "colmap", "sparse", "0")
    for d in [images_dir, depth_dir, sparse_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Load point cloud
    print("\n[1/4] Loading E57 point cloud...")
    xyz, rgb = load_e57_points(e57_path, max_points)

    # 2. Place virtual cameras
    print("\n[2/4] Placing virtual cameras...")
    cameras = place_cameras(xyz, grid_xy, n_heights, fov_deg, w, h)

    # 3. Render images
    print(f"\n[3/4] Rendering {len(cameras)} images ({w}x{h})...")
    for i, cam in enumerate(cameras, 1):
        color, depth = render_camera(xyz, rgb, cam, depth_scale)
        cv2.imwrite(os.path.join(images_dir, f"{i:05d}.png"), color)
        cv2.imwrite(os.path.join(depth_dir, f"{i:05d}.depth.png"), depth)
        if i % 10 == 0 or i == len(cameras):
            print(f"  {i}/{len(cameras)}")

    # 4. Write COLMAP files
    print("\n[4/4] Writing COLMAP sparse model...")
    write_cameras_txt(cameras, os.path.join(sparse_dir, "cameras.txt"))
    write_images_txt(cameras, os.path.join(sparse_dir, "images.txt"))

    # Subsample further for points3D if needed (COLMAP text files get large)
    pts_limit = min(len(xyz), 5_000_000)
    if len(xyz) > pts_limit:
        idx = np.random.choice(len(xyz), pts_limit, replace=False)
        xyz_out, rgb_out = xyz[idx], rgb[idx]
    else:
        xyz_out, rgb_out = xyz, rgb
    write_points3d_txt(xyz_out, rgb_out, os.path.join(sparse_dir, "points3D.txt"))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  →  {output_dir}")
    print(f"  images/       : {len(cameras)} color PNGs")
    print(f"  depth_images/ : {len(cameras)} depth PNGs (uint16, scale={depth_scale})")
    print(f"  colmap/sparse/0/cameras.txt  : {len(cameras)} cameras (PINHOLE)")
    print(f"  colmap/sparse/0/images.txt   : {len(cameras)} poses")
    print(f"  colmap/sparse/0/points3D.txt : {len(xyz_out):,} colored points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="e57-to-colmap",
        description="Convert a pre-registered E57 point cloud to a COLMAP dataset"
    )
    parser.add_argument("-i", "--input", required=True, help="Path to .e57 file")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--grid_xy", type=int, default=5,
                        help="Camera grid size per axis (default: 5 → 25 cameras per height)")
    parser.add_argument("--heights", type=int, default=3,
                        help="Number of camera height levels (default: 3)")
    parser.add_argument("--image_wh", type=int, nargs=2, default=[1920, 1080],
                        metavar=("W", "H"), help="Rendered image resolution (default: 1920 1080)")
    parser.add_argument("--fov_deg", type=float, default=90.0,
                        help="Horizontal field of view in degrees (default: 90)")
    parser.add_argument("--max_points", type=int, default=5_000_000,
                        help="Max points to load from E57 (default: 5000000)")
    parser.add_argument("--depth_scale", type=float, default=1000.0,
                        help="Depth scale factor: depth_mm = depth_m * scale (default: 1000)")

    args = parser.parse_args()

    convert(
        e57_path=os.path.realpath(args.input),
        output_dir=os.path.realpath(args.output),
        grid_xy=args.grid_xy,
        n_heights=args.heights,
        image_wh=tuple(args.image_wh),
        fov_deg=args.fov_deg,
        max_points=args.max_points,
        depth_scale=args.depth_scale,
    )
