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

"""Generates monocular normals and scale-aligned depth maps for dn-splatter/ags-mesh training."""

import argparse
import os
import shutil
import sys
from pathlib import Path


def convert_lidar_bins(data_dir):
    """Convert raw ARKit LiDAR depth .bin files to uint16 mm PNGs in depth/.

    Input:  <data_dir>/depths/<stem>_depth[_smoothed].bin — headerless little-endian
            float32 depth in meters, row-major, at the LiDAR's native resolution
            (256x192 on current iPhones/iPads).
    Output: <data_dir>/depth/<image_stem>.png — uint16 millimeters, same size and
            orientation as the matching RGB image, named after the (flattened)
            image file so the depth_file_path injection matches it by stem.

    ARKit stores depth in the sensor's landscape orientation. When the RGB images
    are portrait, the depth map is rotated 90 degrees clockwise to match (verified
    against COLMAP points: ~3 cm median error clockwise vs ~1 m counter-clockwise).
    Set LIDAR_DEPTH_ROTATION=ccw|none|cw to override.

    Returns the number of converted maps (0 if there is nothing to convert).
    """
    import re
    import numpy as np
    import cv2

    src_dir = os.path.join(data_dir, "depths")
    images_dir = os.path.join(data_dir, "images")
    out_dir = os.path.join(data_dir, "depth")
    if not os.path.isdir(src_dir) or not os.path.isdir(images_dir):
        return 0
    if os.path.isdir(out_dir) and any(f.lower().endswith(".png") for f in os.listdir(out_dir)):
        print(f"[LiDAR depth] {out_dir} already has PNGs, skipping .bin conversion")
        return 0

    bins = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".bin"))
    if not bins:
        return 0

    # Map lidar stem ("scan_000") -> image filename. Images may have been flattened
    # from images/<subdir>/scan_000.jpg to images/<subdir>_scan_000.jpg.
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    by_stem = {os.path.splitext(f)[0]: f for f in image_files}

    def _find_image(lidar_stem):
        if lidar_stem in by_stem:
            return by_stem[lidar_stem]
        matches = [f for s, f in by_stem.items() if s.endswith("_" + lidar_stem)]
        return matches[0] if len(matches) == 1 else None

    rotation = os.environ.get("LIDAR_DEPTH_ROTATION", "auto").lower()
    os.makedirs(out_dir, exist_ok=True)
    converted, skipped = 0, []
    ranges = []
    for fname in bins:
        lidar_stem = re.sub(r"_depth(_smoothed)?$", "", os.path.splitext(fname)[0])
        img_name = _find_image(lidar_stem)
        if img_name is None:
            skipped.append(f"{fname} (no matching image)")
            continue
        img = cv2.imread(os.path.join(images_dir, img_name), cv2.IMREAD_UNCHANGED)
        if img is None:
            skipped.append(f"{fname} (unreadable image {img_name})")
            continue
        img_h, img_w = img.shape[:2]

        raw = np.fromfile(os.path.join(src_dir, fname), dtype="<f4")
        # Infer the sensor shape: landscape with the RGB aspect ratio, falling back
        # to the standard ARKit 256x192.
        long_side, short_side = max(img_w, img_h), min(img_w, img_h)
        h = int(round(np.sqrt(raw.size * short_side / long_side)))
        w = raw.size // h if h else 0
        if h * w != raw.size:
            h, w = 192, 256
        if h * w != raw.size:
            skipped.append(f"{fname} ({raw.size} floats is not a recognised depth size)")
            continue
        depth = raw.reshape(h, w)

        portrait_rgb = img_h > img_w
        landscape_depth = w > h
        rot = rotation
        if rot == "auto":
            rot = "cw" if portrait_rgb and landscape_depth else "none"
        if rot == "cw":
            depth = np.rot90(depth, -1)
        elif rot == "ccw":
            depth = np.rot90(depth, 1)

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth < 0] = 0.0
        if depth.shape != (img_h, img_w):
            # Nearest-neighbour avoids blending foreground/background depth at edges.
            depth = cv2.resize(np.ascontiguousarray(depth), (img_w, img_h),
                               interpolation=cv2.INTER_NEAREST)
        depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        out_name = os.path.splitext(img_name)[0] + ".png"
        cv2.imwrite(os.path.join(out_dir, out_name), depth_mm)
        valid = depth[depth > 0]
        if valid.size:
            ranges.append((float(valid.min()), float(valid.max())))
        converted += 1

    if converted:
        lo = min(r[0] for r in ranges) if ranges else 0.0
        hi = max(r[1] for r in ranges) if ranges else 0.0
        print(f"[LiDAR depth] Converted {converted}/{len(bins)} .bin maps -> {out_dir} "
              f"(rotation={rot}, range {lo:.2f}-{hi:.2f} m)")
    for s in skipped[:5]:
        print(f"[LiDAR depth] WARNING: skipped {s}")
    if len(skipped) > 5:
        print(f"[LiDAR depth] WARNING: ... and {len(skipped) - 5} more skipped")
    return converted


# Ensures colmap/sparse symlink exists for depth alignment script
def ensure_colmap_sparse_link(data_dir):
    """Create colmap/sparse symlink if sparse/ exists but colmap/sparse does not."""
    sparse_path = os.path.join(data_dir, "sparse")
    colmap_sparse_path = os.path.join(data_dir, "colmap", "sparse")
    if os.path.exists(sparse_path) and not os.path.exists(colmap_sparse_path):
        colmap_dir = os.path.join(data_dir, "colmap")
        os.makedirs(colmap_dir, exist_ok=True)
        os.symlink(os.path.abspath(sparse_path), colmap_sparse_path)
        print(f"Created symlink: {colmap_sparse_path} -> {sparse_path}")


# Removes non-image files, flattens image subdirectories, and removes unreadable images.
# dn-splatter's normals_from_pretrain.py and depth_from_pretrain.py iterate images/ flat
# and crash on subdirs (IsADirectoryError) or unreadable files (cv2 empty assertion).
def clean_images_dir(images_dir):
    """Flatten image subdirs, remove non-image files, and remove OpenCV-unreadable images."""
    import cv2
    valid_extensions = {'.png', '.jpg', '.jpeg'}
    removed = 0
    flattened = 0
    invalid = 0

    # Pass 1: flatten subdirs and remove non-image files
    for entry in list(os.listdir(images_dir)):
        fpath = os.path.join(images_dir, entry)
        if os.path.isdir(fpath):
            for fname in os.listdir(fpath):
                if os.path.splitext(fname)[1].lower() in valid_extensions:
                    src = os.path.join(fpath, fname)
                    dst = os.path.join(images_dir, f"{entry}_{fname}")
                    os.rename(src, dst)
                    flattened += 1
            shutil.rmtree(fpath)
            print(f"Flattened subdir {entry}: moved {flattened} images")
        elif os.path.isfile(fpath) and os.path.splitext(entry)[1].lower() not in valid_extensions:
            os.remove(fpath)
            removed += 1

    # Pass 2: remove any image-extension file that OpenCV cannot decode
    for entry in list(os.listdir(images_dir)):
        fpath = os.path.join(images_dir, entry)
        if os.path.isfile(fpath) and os.path.splitext(entry)[1].lower() in valid_extensions:
            img = cv2.imread(fpath)
            if img is None:
                print(f"WARNING: Removing unreadable image: {fpath}")
                os.remove(fpath)
                invalid += 1

    print(f"clean_images_dir: flattened={flattened}, removed_non_image={removed}, removed_invalid={invalid}, "
          f"remaining={len([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])}")


def _generate_sfm_depths_pycolmap(data_dir: str, colmap_sparse: str) -> None:
    """Generate per-image SFM depth maps from a COLMAP reconstruction using pycolmap 4.x.

    Uses pycolmap.Reconstruction which correctly handles COLMAP 4.x rig format
    (rigs.bin/frames.bin) that breaks the legacy read_images_binary approach.
    Outputs to <data_dir>/sfm_depths/ as .npy files.
    """
    import numpy as np
    import pycolmap

    recon = pycolmap.Reconstruction(colmap_sparse)
    print(f"[SFM depths] Loaded: {recon.num_images()} images, "
          f"{recon.num_points3D()} 3D points")

    if recon.num_points3D() == 0:
        raise ValueError("Reconstruction has 0 3D points — cannot generate SFM depths")

    sfm_depths_dir = os.path.join(data_dir, "sfm_depths")
    os.makedirs(sfm_depths_dir, exist_ok=True)

    # COLMAP stores image names relative to the image root, e.g.
    # "face_iphone/scan_000.jpg". clean_images_dir() flattens such files to
    # images/face_iphone_scan_000.jpg, and mono_depth is named from the flattened
    # file. Derive SFM depth stems the same way so both sets match; otherwise the
    # reconcile step deletes every real SFM depth and keeps only zero stubs.
    def _flat_stem(colmap_name):
        return "_".join(Path(os.path.splitext(colmap_name)[0]).parts)

    # Build a stem->camera map so stubs have the correct (H, W).
    # Fall back to reading the image file if the camera lookup fails.
    stem_to_hw = {}
    for image_id, image in recon.images.items():
        cam = recon.cameras[image.camera_id] if image.camera_id in recon.cameras else None
        if cam is not None:
            stem = _flat_stem(image.name)
            stem_to_hw[stem] = (cam.height, cam.width)

    # Pre-create zero depth maps for every image so the count always matches mono_depth.
    # Images with no triangulated points will keep their zero map; images with points
    # will have their map overwritten below.
    images_dir = os.path.join(data_dir, "images")
    if os.path.isdir(images_dir):
        for fname in os.listdir(images_dir):
            if os.path.splitext(fname)[1].lower() in {".png", ".jpg", ".jpeg"}:
                stem = os.path.splitext(fname)[0]
                zero_path = os.path.join(sfm_depths_dir, stem + ".npy")
                if not os.path.exists(zero_path):
                    if stem in stem_to_hw:
                        H, W = stem_to_hw[stem]
                    else:
                        # Fall back to reading image dimensions
                        try:
                            import cv2 as _cv2
                            img = _cv2.imread(os.path.join(images_dir, fname))
                            H, W = img.shape[:2] if img is not None else (1, 1)
                        except Exception:
                            H, W = 1, 1
                    np.save(zero_path, np.zeros((H, W), dtype=np.float32))

    # All 3D points as one array, plus an id->row lookup for observation-based images.
    point_ids = np.fromiter(recon.points3D.keys(), dtype=np.int64)
    all_xyz = np.stack([np.asarray(recon.points3D[int(pid)].xyz) for pid in point_ids])
    id_to_row = {int(pid): i for i, pid in enumerate(point_ids)}

    def _project(cam, pts_cam):
        """Project camera-space points (N,3, z>0) to pixel coords (N,2)."""
        xy = pts_cam[:, :2] / pts_cam[:, 2:3]
        try:
            uv = np.asarray(cam.img_from_cam(xy), dtype=np.float64)
            if uv.shape == xy.shape:
                return uv
        except Exception:
            pass
        # Fallback: pinhole intrinsics (exact for PINHOLE / SIMPLE_PINHOLE).
        K = np.asarray(cam.calibration_matrix(), dtype=np.float64)
        return xy @ K[:2, :2].T + K[:2, 2]

    generated = 0
    filled_counts = []
    all_depth_samples = []
    used_all_points = 0
    for image_id, image in recon.images.items():
        has_pose = image.has_pose() if callable(image.has_pose) else image.has_pose
        if not has_pose:
            continue
        cam = recon.cameras[image.camera_id]
        W, H = cam.width, cam.height
        depth_map = np.zeros((H, W), dtype=np.float32)

        cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        R = np.asarray(cfw.rotation.matrix(), dtype=np.float64)
        t = np.asarray(cfw.translation, dtype=np.float64)

        # Prefer the image's own 2D-3D observations (they encode visibility).
        # Models built from known poses (e.g. ARKit/iPhone captures) often have
        # points3D but no per-image keypoints; in that case project every 3D point
        # and resolve occlusion with a coarse depth test below.
        obs_rows = []
        for p2d in image.points2D:
            has3d = p2d.has_point3D() if callable(p2d.has_point3D) else p2d.has_point3D
            if has3d and int(p2d.point3D_id) in id_to_row:
                obs_rows.append(id_to_row[int(p2d.point3D_id)])
        if obs_rows:
            xyz = all_xyz[np.asarray(obs_rows)]
        else:
            xyz = all_xyz
            used_all_points += 1

        pts_cam = xyz @ R.T + t
        pts_cam = pts_cam[pts_cam[:, 2] > 1e-6]
        if len(pts_cam):
            uv = _project(cam, pts_cam)
            u = np.round(uv[:, 0]).astype(np.int64)
            v = np.round(uv[:, 1]).astype(np.int64)
            z = pts_cam[:, 2].astype(np.float32)
            inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u, v, z = u[inb], v[inb], z[inb]

            if not obs_rows and len(z):
                # Without visibility info, background points can land on pixels
                # where the foreground surface is simply unsampled. Reject points
                # much deeper than the nearest point in their 16x16-pixel block.
                B = 16
                bw = (W + B - 1) // B
                blk = (v // B) * bw + (u // B)
                blk_min = np.full(((H + B - 1) // B) * bw, np.inf, dtype=np.float32)
                np.minimum.at(blk_min, blk, z)
                keep = z <= blk_min[blk] * 1.15
                u, v, z = u[keep], v[keep], z[keep]

            if len(z):
                # Per-pixel z-buffer: keep the nearest point for each pixel.
                zbuf = np.full(H * W, np.inf, dtype=np.float32)
                np.minimum.at(zbuf, v * W + u, z)
                zbuf[np.isinf(zbuf)] = 0.0
                depth_map = zbuf.reshape(H, W)

        filled = int((depth_map > 0).sum())
        filled_counts.append(filled)
        if filled:
            all_depth_samples.append(depth_map[depth_map > 0][:: max(1, filled // 2000)])

        stem = _flat_stem(image.name)
        np.save(os.path.join(sfm_depths_dir, stem + ".npy"), depth_map)
        generated += 1

    print(f"[SFM depths] Generated {generated} depth maps in {sfm_depths_dir}")
    if used_all_points:
        print(f"[SFM depths] {used_all_points} images had no 2D-3D observations; "
              f"projected all 3D points with occlusion filtering")
    if filled_counts:
        print(f"[SFM depths] Filled pixels per image: min={min(filled_counts)} "
              f"median={int(np.median(filled_counts))} max={max(filled_counts)}")
    if all_depth_samples:
        _d = np.concatenate(all_depth_samples)
        print(f"[SFM depths] Depth range: p5={np.percentile(_d, 5):.3f} "
              f"median={np.median(_d):.3f} p95={np.percentile(_d, 95):.3f} "
              f"(alignment uses 0.1-10.0; {100 * ((_d > 0.1) & (_d < 10)).mean():.1f}% in range)")
    else:
        print("[SFM depths] WARNING: every SFM depth map is empty — depth alignment will be invalid")


def main():
    parser = argparse.ArgumentParser(description="Generate normals and aligned depth for dn-splatter")
    parser.add_argument("--data-dir", required=True, help="Path to dataset root")
    parser.add_argument("--normal-format", default="dsine", choices=["omnidata", "dsine"],
                        help="Normal estimation network to use")
    parser.add_argument("--skip-normals", action="store_true", help="Skip normal generation")
    parser.add_argument("--skip-depth", action="store_true", help="Skip depth alignment")
    parser.add_argument("--has-sensor-depth", action="store_true",
                        help="Sensor depth images are in depth/ dir; skip mono depth generation")
    parser.add_argument("--generate-depth-masks", action="store_true",
                        help="Generate depth-normal consistency masks (required for ags-mesh)")
    args = parser.parse_args()

    data_dir = args.data_dir
    images_dir = os.path.join(data_dir, "images")

    # Clean non-image files
    if os.path.exists(images_dir):
        clean_images_dir(images_dir)

    # Normalize image dimensions after flattening — mixed sizes crash depth_from_pretrain.py
    # (torch.stack requires identical tensor shapes). Must run after clean_images_dir flattens
    # subdirectories so all images are in the flat images/ directory.
    if os.path.exists(images_dir):
        import cv2 as _cv2
        _img_files = [f for f in os.listdir(images_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                      and os.path.isfile(os.path.join(images_dir, f))]
        if _img_files:
            _dims = set()
            for _f in _img_files:
                _img = _cv2.imread(os.path.join(images_dir, _f))
                if _img is not None:
                    _dims.add((_img.shape[1], _img.shape[0]))
            if len(_dims) > 1:
                print(f"Mixed image dimensions detected {_dims} — normalizing to smallest common size")
                _target_w = min(d[0] for d in _dims)
                _target_h = min(d[1] for d in _dims)
                _resized = 0
                for _f in _img_files:
                    _fp = os.path.join(images_dir, _f)
                    _img = _cv2.imread(_fp)
                    if _img is None:
                        continue
                    _h, _w = _img.shape[:2]
                    if _w != _target_w or _h != _target_h:
                        _target_ratio = _target_w / _target_h
                        _img_ratio = _w / _h
                        if _img_ratio > _target_ratio:
                            _crop_w = int(_h * _target_ratio)
                            _x = (_w - _crop_w) // 2
                            _img = _img[:, _x:_x + _crop_w]
                        elif _img_ratio < _target_ratio:
                            _crop_h = int(_w / _target_ratio)
                            _y = (_h - _crop_h) // 2
                            _img = _img[_y:_y + _crop_h, :]
                        _img = _cv2.resize(_img, (_target_w, _target_h), interpolation=_cv2.INTER_AREA)
                        _cv2.imwrite(_fp, _img)
                        _resized += 1
                print(f"Normalized {_resized}/{len(_img_files)} images to {_target_w}x{_target_h}")

                # Resize masks to match the new image dimensions
                _masks_dir = os.path.join(data_dir, "masks")
                if os.path.isdir(_masks_dir):
                    _mask_files = [f for f in os.listdir(_masks_dir)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                                   and os.path.isfile(os.path.join(_masks_dir, f))]
                    _mask_resized = 0
                    for _mf in _mask_files:
                        _mp = os.path.join(_masks_dir, _mf)
                        _mask = _cv2.imread(_mp, _cv2.IMREAD_GRAYSCALE)
                        if _mask is None:
                            continue
                        _mh, _mw = _mask.shape[:2]
                        if _mw != _target_w or _mh != _target_h:
                            _mask = _cv2.resize(_mask, (_target_w, _target_h),
                                                interpolation=_cv2.INTER_NEAREST)
                            _cv2.imwrite(_mp, _mask)
                            _mask_resized += 1
                    if _mask_resized:
                        print(f"Resized {_mask_resized} masks to {_target_w}x{_target_h}")

            else:
                # Uniform dimensions — still resize masks if they don't match
                _target_w, _target_h = next(iter(_dims))
                _masks_dir = os.path.join(data_dir, "masks")
                if os.path.isdir(_masks_dir):
                    _mask_files = [f for f in os.listdir(_masks_dir)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                                   and os.path.isfile(os.path.join(_masks_dir, f))]
                    _mask_resized = 0
                    for _mf in _mask_files:
                        _mp = os.path.join(_masks_dir, _mf)
                        _mask = _cv2.imread(_mp, _cv2.IMREAD_GRAYSCALE)
                        if _mask is None:
                            continue
                        _mh, _mw = _mask.shape[:2]
                        if _mw != _target_w or _mh != _target_h:
                            _mask = _cv2.resize(_mask, (_target_w, _target_h),
                                                interpolation=_cv2.INTER_NEAREST)
                            _cv2.imwrite(_mp, _mask)
                            _mask_resized += 1
                    if _mask_resized:
                        print(f"Resized {_mask_resized} masks to {_target_w}x{_target_h}")

                # Update transforms.json intrinsics to match the new image dimensions
                _transforms_path = os.path.join(data_dir, "transforms.json")
                if os.path.isfile(_transforms_path):
                    import json as _json
                    with open(_transforms_path, 'r') as _f:
                        _tdata = _json.load(_f)
                    _cam_w = _tdata.get('w', _target_w)
                    _cam_h = _tdata.get('h', _target_h)
                    if _cam_w != _target_w or _cam_h != _target_h:
                        _sx = _target_w / _cam_w
                        _sy = _target_h / _cam_h
                        _tdata['w'] = _target_w
                        _tdata['h'] = _target_h
                        for _k in ('fl_x', 'cx'):
                            if _k in _tdata:
                                _tdata[_k] = _tdata[_k] * _sx
                        if 'fl_y' in _tdata:
                            _tdata['fl_y'] = _tdata['fl_y'] * _sy
                        if 'cy' in _tdata:
                            _tdata['cy'] = _tdata['cy'] * _sy
                        for _frame in _tdata.get('frames', []):
                            for _k in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy'):
                                if _k in _frame:
                                    _s = _sx if _k in ('w', 'fl_x', 'cx') else _sy
                                    _frame[_k] = int(_frame[_k] * _s) if _k in ('w', 'h') else _frame[_k] * _s
                        with open(_transforms_path, 'w') as _f:
                            _json.dump(_tdata, _f, indent=4)
                        print(f"Updated transforms.json intrinsics to {_target_w}x{_target_h}")

    # If images/ is empty, search for images in common alternative locations
    # (e.g. colmap/images/ when the zip was a COLMAP reconstruction export)
    if not os.path.isdir(images_dir) or not any(
        f.lower().endswith(('.png', '.jpg', '.jpeg'))
        for f in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, f))
    ):
        for alt in ['colmap/images', 'colmap/images/images']:
            alt_path = os.path.join(data_dir, alt)
            if os.path.isdir(alt_path) and any(
                f.lower().endswith(('.png', '.jpg', '.jpeg'))
                for f in os.listdir(alt_path)
                if os.path.isfile(os.path.join(alt_path, f))
            ):
                print(f"images/ is empty, copying images from {alt_path}")
                os.makedirs(images_dir, exist_ok=True)
                for f in os.listdir(alt_path):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        shutil.copy2(os.path.join(alt_path, f), os.path.join(images_dir, f))
                break

    # Ensure colmap/sparse structure for depth alignment
    ensure_colmap_sparse_link(data_dir)

    # Ensure sparse.ply exists for dn-splatter's normal_nerfstudio.py (load_3D_points=True).
    # colmap_to_nerfstudio_cam.py writes the ply_file_path into transforms.json but does not
    # create the file itself. Generate it from points3D.bin/txt using pycolmap if missing.
    def ensure_sparse_ply(data_dir):
        for sparse_subdir in ["sparse/0", "colmap/sparse/0"]:
            sparse_dir = os.path.join(data_dir, sparse_subdir)
            if not os.path.isdir(sparse_dir):
                continue
            ply_path = os.path.join(sparse_dir, "sparse.ply")
            if os.path.isfile(ply_path):
                return  # already exists
            points3d_bin = os.path.join(sparse_dir, "points3D.bin")
            points3d_txt = os.path.join(sparse_dir, "points3D.txt")
            if os.path.isfile(points3d_bin) or os.path.isfile(points3d_txt):
                try:
                    import pycolmap
                    reconstruction = pycolmap.Reconstruction(sparse_dir)
                    reconstruction.export_PLY(ply_path)
                    print(f"Generated sparse.ply with {reconstruction.num_points3D()} points at {ply_path}")
                    return
                except Exception as e:
                    print(f"WARNING: Could not generate sparse.ply: {e}")

    ensure_sparse_ply(data_dir)

    # Ensure sparse/0/ mirrors colmap/sparse/0/ so NerfStudio's default colmap_path
    # (colmap/sparse/0) and any ply_file_path entries in transforms.json that point
    # to sparse/0/sparse.ply both resolve correctly.
    _colmap_s0 = os.path.join(data_dir, "colmap", "sparse", "0")
    _sparse_s0 = os.path.join(data_dir, "sparse", "0")
    if os.path.isdir(_colmap_s0) and os.listdir(_colmap_s0):
        os.makedirs(_sparse_s0, exist_ok=True)
        for _f in os.listdir(_colmap_s0):
            _src = os.path.join(_colmap_s0, _f)
            _dst = os.path.join(_sparse_s0, _f)
            if not os.path.exists(_dst):
                os.symlink(_src, _dst)
        print(f"Linked colmap/sparse/0/ files into sparse/0/ ({len(os.listdir(_colmap_s0))} files)")

    # Patch dn_model.py: combined_depth_normalized is only set inside
    # 'if sensor_depth in batch' but referenced unconditionally at line 956.
    # Fix: initialize it right after combined_depth is first assigned.
    def _patch_dn_model_depth():
        path = "/opt/ml/code/dn-splatter/dn_splatter/dn_model.py"
        try:
            with open(path, 'r') as f:
                src = f.read()
            if 'combined_depth_normalized' in src and \
               'combined_depth = (\n            predicted_depth  # a placeholder' in src and \
               'combined_depth_normalized = combined_depth / combined_depth.max()' in src:
                patched = src.replace(
                    'combined_depth = (\n            predicted_depth  # a placeholder if no sensor depth is available\n        )',
                    'combined_depth = (\n            predicted_depth  # a placeholder if no sensor depth is available\n        )\n        combined_depth_normalized = combined_depth / (combined_depth.max() + 1e-8)  # default; overwritten if sensor_depth present'
                )
                if patched != src:
                    with open(path, 'w') as f:
                        f.write(patched)
                    print(f"Patched dn_model.py combined_depth_normalized initialization")
                else:
                    print("WARNING: dn_model.py combined_depth patch did not apply (text not found)")
            else:
                print("INFO: dn_model.py combined_depth already patched or has different structure")
        except Exception as e:
            print(f"WARNING: Could not patch dn_model.py: {e}")
    _patch_dn_model_depth()

    # Sanitize transforms.json: normals_from_pretrain.py reads file_path entries directly
    # and passes them to cv2.imread — it does NOT scan images/ on disk. After clean_images_dir
    # flattens subdirs (e.g. images/face_00/x.jpg -> images/face_00_x.jpg), the old paths in
    # transforms.json no longer exist, causing cv2.imread to return None -> cvtColor crash.
    # Fix: rewrite stale file_path entries to their flattened equivalents, drop unreadable ones.
    def sanitize_transforms_json(data_dir):
        import json
        import cv2 as _cv2
        transforms_path = os.path.join(data_dir, "transforms.json")
        if not os.path.exists(transforms_path):
            return
        with open(transforms_path, "r") as f:
            meta = json.load(f)
        frames = meta.get("frames", [])
        clean_frames = []
        rewritten = 0
        dropped = 0
        for frame in frames:
            fp = frame.get("file_path", "")
            abs_path = os.path.join(data_dir, fp)
            if not os.path.exists(abs_path):
                # Try to find the flattened equivalent: images/<subdir>_<filename>
                parts = Path(fp).parts  # e.g. ('images', 'face_00', 'frame.jpg')
                if len(parts) >= 3:
                    flat_name = "_".join(parts[1:])  # face_00_frame.jpg
                    flat_path = os.path.join(data_dir, parts[0], flat_name)
                    if os.path.exists(flat_path):
                        frame["file_path"] = str(Path(parts[0]) / flat_name)
                        abs_path = flat_path
                        rewritten += 1
                    else:
                        print(f"WARNING: dropping frame, file not found: {abs_path}")
                        dropped += 1
                        continue
                else:
                    print(f"WARNING: dropping frame, file not found: {abs_path}")
                    dropped += 1
                    continue
            if _cv2.imread(abs_path) is None:
                print(f"WARNING: dropping frame, unreadable image: {abs_path}")
                dropped += 1
                continue
            clean_frames.append(frame)
        meta["frames"] = clean_frames
        with open(transforms_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"sanitize_transforms_json: rewritten={rewritten}, dropped={dropped}, "
              f"remaining={len(clean_frames)}")

    sanitize_transforms_json(data_dir)

    # Patch normal_nerfstudio.py: replace natsorted with sorted for normal file paths.
    # natsorted can order files differently than the image list, causing normals from
    # one image to be applied to a different image, severely degrading quality.
    def _patch_normal_nerfstudio():
        path = "/opt/ml/code/dn-splatter/dn_splatter/data/normal_nerfstudio.py"
        try:
            with open(path, 'r') as f:
                src = f.read()
            patched = src.replace(
                'return natsorted(glob.glob(f"{self.normal_save_dir}/*.png"))',
                'return sorted(glob.glob(f"{self.normal_save_dir}/*.png"))'
            )
            # Also fix mono depth path ordering for the same reason
            patched = patched.replace(
                'depth_paths = natsorted(\n            glob.glob(f"{self.config.data}/mono_depth/*_aligned.npy")',
                'depth_paths = sorted(\n            glob.glob(f"{self.config.data}/mono_depth/*_aligned.npy")'
            ).replace(
                'depth_paths = natsorted(glob.glob(f"{self.config.data}/mono_depth/*.npy"))',
                'depth_paths = sorted(glob.glob(f"{self.config.data}/mono_depth/*.npy"))'
            )
            if patched != src:
                with open(path, 'w') as f:
                    f.write(patched)
                print("Patched normal_nerfstudio.py: natsorted -> sorted for normal file paths")
            # Evict cached module so the patch takes effect
            import sys
            for mod in list(sys.modules.keys()):
                if 'normal_nerfstudio' in mod:
                    del sys.modules[mod]
        except Exception as e:
            print(f"WARNING: Could not patch normal_nerfstudio.py: {e}")
    _patch_normal_nerfstudio()

    # Generate monocular normals
    if not args.skip_normals:
        print(f"Generating {args.normal_format} normals for {data_dir}")
        from dn_splatter.scripts.normals_from_pretrain import NormalsFromPretrained
        normals_generator = NormalsFromPretrained(
            data_dir=Path(data_dir),
            normal_format=args.normal_format,
        )
        normals_generator.main()

    # Convert raw ARKit LiDAR depth (depths/<stem>_depth*.bin, float32 meters at the
    # sensor's native resolution, e.g. 256x192) into depth/<image_stem>.png uint16 mm
    # at RGB resolution, so the sensor-depth injection below picks it up.
    convert_lidar_bins(data_dir)

    # Inject depth_file_path into transforms.json only when sensor depth is valid
    # uint16 mm PNGs (cv2.IMREAD_ANYDEPTH gives uint16, x0.001 -> meters in plausible range).
    # ARKit .depth.png files are uint8 confidence maps and are NOT valid sensor depth.
    # When valid sensor depth is present, depth_file_path enables EdgeAwareLogL1 supervision.
    # Otherwise training uses scale-aligned mono depth via mono_depth/*_aligned.npy.
    import json as _json
    _depth_dir = os.path.join(data_dir, "depth")
    if not os.path.isdir(_depth_dir):
        _depth_dir = os.path.join(data_dir, "depth_images")
    _transforms_path = os.path.join(data_dir, "transforms.json")
    if os.path.isdir(_depth_dir) and os.path.isfile(_transforms_path):
        # Flatten depth subdirs to match flattened images/ convention:
        # depth/face_00/pano_008.png -> depth/face_00_pano_008.png
        for _entry in list(os.listdir(_depth_dir)):
            _entry_path = os.path.join(_depth_dir, _entry)
            if os.path.isdir(_entry_path):
                for _fname in os.listdir(_entry_path):
                    if os.path.splitext(_fname)[1].lower() in {'.png', '.jpg', '.npy'}:
                        _src = os.path.join(_entry_path, _fname)
                        _dst = os.path.join(_depth_dir, f"{_entry}_{_fname}")
                        os.rename(_src, _dst)
                shutil.rmtree(_entry_path)
                print(f"Flattened depth subdir {_entry}")

        _depth_files = {os.path.splitext(f)[0].lower(): f
                        for f in os.listdir(_depth_dir)
                        if f.lower().endswith(('.png', '.jpg', '.npy'))}
        if _depth_files:
            # Validate format: sample one file and check it's uint16 PNG with plausible metric range
            _sample_file = os.path.join(_depth_dir, next(iter(_depth_files.values())))
            _valid_sensor_depth = False
            try:
                import cv2 as _cv2
                _sample = _cv2.imread(_sample_file, _cv2.IMREAD_ANYDEPTH)
                if _sample is not None and _sample.dtype == 'uint16':
                    _meters_max = float(_sample.max()) * 0.001
                    _meters_min = float(_sample[_sample > 0].min()) * 0.001 if (_sample > 0).any() else 0
                    # Plausible scene depth: 0.1m to 1000m
                    if _meters_min >= 0.1 and _meters_max <= 1000.0:
                        _valid_sensor_depth = True
                        print(f"Sensor depth validated: uint16 PNG, range [{_meters_min:.2f}, {_meters_max:.2f}]m")
                    else:
                        print(f"WARNING: Sensor depth rejected — range [{_meters_min:.3f}, {_meters_max:.3f}]m is not plausible metric depth. "
                              f"Expected uint16 mm values (e.g. 2500 = 2.5m). Falling back to mono depth.")
                else:
                    print(f"WARNING: Sensor depth rejected — expected uint16 PNG but got dtype={getattr(_sample, 'dtype', 'None')}. "
                          f"Falling back to mono depth.")
            except Exception as _e:
                print(f"WARNING: Could not validate sensor depth: {_e}. Falling back to mono depth.")

            if _valid_sensor_depth:
                with open(_transforms_path, "r") as _f:
                    _transforms = _json.load(_f)
                _injected = 0
                _depth_dir_name = os.path.basename(_depth_dir)
                for _frame in _transforms.get("frames", []):
                    if "depth_file_path" not in _frame:
                        _img = _frame.get("file_path", "")
                        _stem = os.path.splitext(os.path.basename(_img))[0].lower()
                        if _stem in _depth_files:
                            _frame["depth_file_path"] = f"{_depth_dir_name}/{_depth_files[_stem]}"
                            _injected += 1
                        else:
                            _num = ''.join(filter(str.isdigit, _stem))
                            _matched = next((k for k in _depth_files if ''.join(filter(str.isdigit, k)) == _num and _num), None)
                            if _matched:
                                _frame["depth_file_path"] = f"{_depth_dir_name}/{_depth_files[_matched]}"
                                _injected += 1
                if _injected > 0:
                    with open(_transforms_path, "w") as _f:
                        _json.dump(_transforms, _f, indent=2)
                    print(f"Injected depth_file_path into {_injected} frames from {_depth_dir_name}/")
                    # Write marker so training knows to use EdgeAwareLogL1 for sensor depth
                    open(os.path.join(data_dir, ".sensor_depth_valid"), 'w').close()
                    # dn-splatter loads sensor depth only when every frame has a
                    # depth_file_path; otherwise it needs mono depth for all frames.
                    _frames = _transforms.get("frames", [])
                    _sensor_depth_all_frames = bool(_frames) and all(
                        "depth_file_path" in _fr for _fr in _frames
                    )
                    if not _sensor_depth_all_frames:
                        print(f"WARNING: sensor depth matched only {_injected}/{len(_frames)} frames; "
                              f"mono depth will still be generated.")
                else:
                    print(f"WARNING: Could not match sensor depth files to transforms.json frames. Falling back to mono depth.")

            else:
                # Sensor depth rejected (e.g. uint8 ARKit maps).
                # After alignment, convert mono_depth/*_aligned.npy -> uint16 mm PNGs
                # so dn-splatter can use them as proper sensor depth supervision.
                print(f"INFO: Sensor depth rejected. Will convert aligned mono depth to uint16 mm PNGs after alignment.")
                _convert_mono_to_sensor = True
        else:
            print(f"INFO: No depth files found in {_depth_dir}. Using mono depth.")

    # Monkey-patch ZoeDepth model_io to use strict=False, fixing timm>=0.9
    # relative_position_index buffer key mismatch with the ZoeD_M12_N checkpoint.
    # torch.hub downloads ZoeDepth on first torch.hub.load call, so we pre-trigger
    # the download with source_only=True, patch the file, then the real load proceeds.
    def _patch_zoedepth_model_io():
        try:
            import torch.hub as _hub
            import glob as _glob
            # Pre-download ZoeDepth repo (no model weights, just source) so we can patch.
            # skip_validation=True avoids the GitHub fork-check HTTP call that fails in
            # network-restricted SageMaker/Batch environments (HTTP 504).
            _hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=False,
                      trust_repo=True, skip_validation=True)
        except Exception:
            pass  # may fail without weights; we only need the source on disk
        # Pre-download MiDaS so its beit.py is on disk before we patch it.
        try:
            _hub.load("intel-isl/MiDaS", "DPT_BEiT_L_512", pretrained=False,
                      trust_repo=True, skip_validation=True)
        except Exception:
            pass
        try:
            hub_dir = _hub.get_dir()
            candidates = _glob.glob(
                os.path.join(hub_dir, "isl-org_ZoeDepth*", "zoedepth", "models", "model_io.py")
            )
            for path in candidates:
                with open(path, "r") as f:
                    src = f.read()
                patched = src.replace(
                    "model.load_state_dict(state)\n",
                    "model.load_state_dict(state, strict=False)\n",
                )
                if patched != src:
                    with open(path, "w") as f:
                        f.write(patched)
                    print(f"Patched ZoeDepth model_io: {path}")
            # Evict cached in-memory copies of the patched module so the next
            # torch.hub.load re-imports from the patched file rather than the
            # stale in-memory version loaded during the pretrained=False call.
            for mod_name in list(sys.modules.keys()):
                if "zoedepth" in mod_name or "model_io" in mod_name:
                    del sys.modules[mod_name]
        except Exception as e:
            print(f"WARNING: Could not patch ZoeDepth model_io: {e}")
        # Patch MiDaS beit.py: timm>=0.9 renamed Block.drop_path -> drop_path1/drop_path2.
        # The MiDaS hub code calls self.drop_path(...) which no longer exists.
        try:
            hub_dir = _hub.get_dir()
            midas_beit_candidates = _glob.glob(
                os.path.join(hub_dir, "intel-isl_MiDaS*", "midas", "backbones", "beit.py")
            )
            for path in midas_beit_candidates:
                with open(path, "r") as f:
                    src = f.read()
                patched = src
                # Replace both occurrences: drop_path( -> drop_path1(
                # The MiDaS block_forward uses drop_path for both attn and mlp paths;
                # timm 0.9+ splits these into drop_path1 and drop_path2 respectively.
                patched = patched.replace(
                    "self.drop_path(self.attn(",
                    "self.drop_path1(self.attn("
                ).replace(
                    "self.drop_path(self.gamma_1",
                    "self.drop_path1(self.gamma_1"
                ).replace(
                    "self.drop_path(self.mlp(",
                    "self.drop_path2(self.mlp("
                ).replace(
                    "self.drop_path(self.gamma_2",
                    "self.drop_path2(self.gamma_2"
                )
                if patched != src:
                    with open(path, "w") as f:
                        f.write(patched)
                    print(f"Patched MiDaS beit.py: {path}")
            for mod_name in list(sys.modules.keys()):
                if "midas" in mod_name or "beit" in mod_name:
                    del sys.modules[mod_name]
        except Exception as e:
            print(f"WARNING: Could not patch MiDaS beit.py: {e}")

    # Generate scale-aligned mono depth (ZoeDepth + SFM alignment) only when it will
    # be used. When valid sensor depth covers every frame, dn-splatter trains on it
    # and never reads mono_depth/, so skip the ~1.3 GB download and several minutes.
    if locals().get('_sensor_depth_all_frames', False) and not args.skip_depth:
        print("Valid sensor depth found for every frame; skipping ZoeDepth mono depth generation.")
    elif not args.skip_depth:
        # Monkey-patch torch.hub.load to always pass skip_validation=True so the
        # GitHub fork-check HTTP call (which fails in network-restricted environments
        # with HTTP 504) is bypassed for all subsequent hub.load calls, including
        # those made inside dn-splatter's depth_from_pretrain.py.
        # Also force trust_repo=True so torch.hub never blocks on the interactive
        # "Do you trust this repository? (y/N)" prompt for untrusted repos such as
        # intel-isl/MiDaS (pulled transitively by ZoeDepth) — that prompt hangs
        # non-interactive Batch/SageMaker runs and requires a manual "y" locally.
        import torch.hub as _hub
        _orig_hub_load = _hub.load
        def _hub_load_no_validate(*a, **kw):
            kw.setdefault("skip_validation", True)
            kw.setdefault("trust_repo", True)
            return _orig_hub_load(*a, **kw)
        _hub.load = _hub_load_no_validate
        # Also patch torch.hub.load globally so depth_from_pretrain.py
        # (which imports torch.hub directly) also gets skip_validation=True.
        import torch
        torch.hub.load = _hub_load_no_validate
        _patch_zoedepth_model_io()
        colmap_sparse = os.path.join(data_dir, "colmap", "sparse", "0")
        if os.path.exists(colmap_sparse):
            print(f"Generating SFM depths from {colmap_sparse} using pycolmap")
            try:
                _generate_sfm_depths_pycolmap(data_dir, colmap_sparse)
                from dn_splatter.scripts.align_depth import ColmapToAlignedMonoDepths
                # Run mono depth generation first (skip_mono_depth_creation=False,
                # skip_alignment=True) so we know the exact set of mono_depth files
                # before attempting alignment — this avoids the count mismatch assert.
                mono_depth_dir = os.path.join(data_dir, "mono_depth")
                mono_already_done = os.path.isdir(mono_depth_dir) and \
                    any(f.endswith(".npy") for f in os.listdir(mono_depth_dir))
                if not mono_already_done:
                    depth_aligner_mono = ColmapToAlignedMonoDepths(
                        data=Path(data_dir),
                        skip_colmap_to_depths=True,
                        skip_mono_depth_creation=False,
                        skip_alignment=True,
                    )
                    depth_aligner_mono.main()
                else:
                    print(f"INFO: mono_depth already exists ({mono_depth_dir}), skipping re-generation.")
                # Reconcile sfm_depths to exactly match mono_depth stems.
                # align_depth.py asserts len(sfm)==len(filtered_mono) where
                # filtered_mono keeps only items whose stem appears in sfm_name.
                # Any mismatch in either direction triggers the assert.
                import numpy as np
                sfm_depths_dir = os.path.join(data_dir, "sfm_depths")
                mono_stems = {os.path.splitext(f)[0] for f in os.listdir(mono_depth_dir) if f.endswith(".npy")}
                sfm_stems = {os.path.splitext(f)[0] for f in os.listdir(sfm_depths_dir) if f.endswith(".npy")}
                # Add zero placeholders for mono stems missing from sfm.
                # Always derive shape from mono_depth — sfm placeholders may be (1,1) stubs.
                for stem in mono_stems - sfm_stems:
                    _ref_mono = os.path.join(mono_depth_dir, stem + '.npy')
                    _shape = np.load(_ref_mono).shape[:2]
                    np.save(os.path.join(sfm_depths_dir, stem + '.npy'), np.zeros(_shape, dtype=np.float32))
                    print(f"[SFM depths] Added zero placeholder for {stem}")
                # Remove sfm files that have no corresponding mono_depth
                for stem in sfm_stems - mono_stems:
                    os.remove(os.path.join(sfm_depths_dir, stem + ".npy"))
                    print(f"[SFM depths] Removed unmatched sfm file for {stem}")
                # Now run alignment only — both sets are guaranteed equal size.
                depth_aligner = ColmapToAlignedMonoDepths(
                    data=Path(data_dir),
                    skip_colmap_to_depths=True,
                    skip_mono_depth_creation=True,
                    skip_alignment=False,
                )
                depth_aligner.main()
            except Exception as depth_err:
                print(f"WARNING: SFM depth alignment failed ({depth_err}). "
                      f"Falling back to mono depth without scale alignment.")
                import traceback; traceback.print_exc()
                from dn_splatter.scripts.align_depth import ColmapToAlignedMonoDepths
                mono_depth_dir = os.path.join(data_dir, "mono_depth")
                mono_already_done = os.path.isdir(mono_depth_dir) and \
                    any(f.endswith(".npy") for f in os.listdir(mono_depth_dir))
                if not mono_already_done:
                    _img_dir = os.path.join(data_dir, "images")
                    if os.path.isdir(_img_dir) and any(
                        f.lower().endswith(('.png', '.jpg', '.jpeg'))
                        for f in os.listdir(_img_dir)
                    ):
                        depth_aligner = ColmapToAlignedMonoDepths(
                            data=Path(data_dir),
                            skip_colmap_to_depths=True,
                            skip_mono_depth_creation=False,
                            skip_alignment=True,
                        )
                        depth_aligner.main()
                    else:
                        print(f"WARNING: images/ is empty, skipping mono depth generation")
                else:
                    print(f"INFO: mono_depth already exists ({mono_depth_dir}), skipping re-generation.")
        else:
            print(f"WARNING: No colmap/sparse/0 found at {colmap_sparse}, skipping depth alignment")
            from dn_splatter.scripts.align_depth import ColmapToAlignedMonoDepths
            depth_aligner = ColmapToAlignedMonoDepths(
                data=Path(data_dir),
                skip_colmap_to_depths=True,
                skip_mono_depth_creation=False,
                skip_alignment=True,
            )
            depth_aligner.main()

    # Generate depth-normal consistency masks for AGS-Mesh
    # Only run when sensor depth exists — depth_normal_consistency.py requires
    # depth_file_path in every transforms.json frame, which is only injected
    # when a depth/ or depth_images/ directory is present.
    # Convert aligned mono depth to uint16 mm PNGs and inject as depth_file_path
    # when sensor depth was rejected (e.g. uint8 ARKit maps).
    # This gives dn-splatter proper metric depth supervision from the SFM-aligned mono depth.
    if locals().get('_convert_mono_to_sensor', False):
        import numpy as np
        import cv2 as _cv2
        _mono_depth_dir = os.path.join(data_dir, "mono_depth")
        _sensor_out_dir = os.path.join(data_dir, "depth_sensor")
        os.makedirs(_sensor_out_dir, exist_ok=True)
        _aligned_files = sorted([f for f in os.listdir(_mono_depth_dir) if f.endswith('_aligned.npy')])
        _converted = 0
        for _af in _aligned_files:
            _stem = _af.replace('_aligned.npy', '')
            _depth_m = np.load(os.path.join(_mono_depth_dir, _af)).squeeze().astype(np.float32)
            # Convert meters to uint16 mm, clip to valid uint16 range
            _depth_mm = np.clip(_depth_m * 1000.0, 0, 65535).astype(np.uint16)
            _cv2.imwrite(os.path.join(_sensor_out_dir, f"{_stem}.png"), _depth_mm)
            _converted += 1
        print(f"Converted {_converted} aligned mono depth maps to uint16 mm PNGs in depth_sensor/")
        # Inject depth_file_path into transforms.json
        if _converted > 0 and os.path.isfile(_transforms_path):
            with open(_transforms_path, 'r') as _f:
                _transforms = _json.load(_f)
            _injected = 0
            for _frame in _transforms.get('frames', []):
                if 'depth_file_path' not in _frame:
                    _img_stem = os.path.splitext(os.path.basename(_frame.get('file_path', '')))[0]
                    _png = os.path.join(_sensor_out_dir, f"{_img_stem}.png")
                    if os.path.exists(_png):
                        _frame['depth_file_path'] = f"depth_sensor/{_img_stem}.png"
                        _injected += 1
            if _injected > 0:
                with open(_transforms_path, 'w') as _f:
                    _json.dump(_transforms, _f, indent=2)
                print(f"Injected depth_file_path into {_injected} frames from depth_sensor/")
                open(os.path.join(data_dir, '.sensor_depth_valid'), 'w').close()

    # Only generate depth-normal consistency masks when valid sensor depth was confirmed
    # (depth_file_path entries injected, marked by .sensor_depth_valid).
    # depth_normal_consistency.py requires depth_file_path in every frame.
    _sensor_depth_valid_marker = os.path.join(data_dir, ".sensor_depth_valid")
    if args.generate_depth_masks and os.path.exists(_sensor_depth_valid_marker):
        print(f"Generating depth-normal consistency masks for {data_dir}")
        from dn_splatter.scripts.depth_normal_consistency import DepthNormalConsistency
        mask_generator = DepthNormalConsistency(
            data_dir=Path(data_dir),
            transforms_name="transforms.json",
            normal_format=args.normal_format,
        )
        mask_generator.main()

        # Resize generated masks to match actual image dimensions.
        # DepthNormalConsistency outputs masks at the normal estimator's native
        # resolution (e.g. 1536x1536 square) which may differ from the RGB images.
        import cv2 as _cv2
        _masks_dir = os.path.join(data_dir, "masks")
        if os.path.isdir(_masks_dir) and os.path.isdir(images_dir):
            _sample_img = next(
                (f for f in os.listdir(images_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                 and os.path.isfile(os.path.join(images_dir, f))), None
            )
            if _sample_img is not None:
                _ref = _cv2.imread(os.path.join(images_dir, _sample_img))
                if _ref is not None:
                    _img_h, _img_w = _ref.shape[:2]
                    _mask_resized = 0
                    for _mf in os.listdir(_masks_dir):
                        if not _mf.lower().endswith(('.png', '.jpg', '.jpeg')):
                            continue
                        _mp = os.path.join(_masks_dir, _mf)
                        _mask = _cv2.imread(_mp, _cv2.IMREAD_GRAYSCALE)
                        if _mask is None:
                            continue
                        if _mask.shape[0] != _img_h or _mask.shape[1] != _img_w:
                            _mask = _cv2.resize(_mask, (_img_w, _img_h),
                                                interpolation=_cv2.INTER_NEAREST)
                            _cv2.imwrite(_mp, _mask)
                            _mask_resized += 1
                    if _mask_resized:
                        print(f"Resized {_mask_resized} depth-normal masks to {_img_w}x{_img_h}")

    # Final mask resize pass: runs unconditionally at the end after all image
    # processing is complete. Reads actual image dimensions fresh from disk so
    # it works regardless of whether images were in subdirs, mixed sizes, or
    # already uniform when the earlier resize blocks ran.
    import cv2 as _cv2
    _masks_dir = os.path.join(data_dir, "masks")
    if os.path.isdir(_masks_dir) and os.path.isdir(images_dir):
        _img_sample = next(
            (f for f in os.listdir(images_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))
             and os.path.isfile(os.path.join(images_dir, f))), None
        )
        if _img_sample is not None:
            _ref = _cv2.imread(os.path.join(images_dir, _img_sample))
            if _ref is not None:
                _img_h, _img_w = _ref.shape[:2]
                _mask_resized = 0
                for _mf in os.listdir(_masks_dir):
                    if not _mf.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    _mp = os.path.join(_masks_dir, _mf)
                    _mask = _cv2.imread(_mp, _cv2.IMREAD_GRAYSCALE)
                    if _mask is None:
                        continue
                    if _mask.shape[0] != _img_h or _mask.shape[1] != _img_w:
                        _mask = _cv2.resize(_mask, (_img_w, _img_h),
                                            interpolation=_cv2.INTER_NEAREST)
                        _cv2.imwrite(_mp, _mask)
                        _mask_resized += 1
                if _mask_resized:
                    print(f"Final mask resize: {_mask_resized} masks resized to {_img_w}x{_img_h}")

    print("DN-Splatter pre-processing complete.")


if __name__ == "__main__":
    main()
