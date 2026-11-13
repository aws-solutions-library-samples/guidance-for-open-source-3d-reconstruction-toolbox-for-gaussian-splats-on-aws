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

"""Refine the camera poses of an uploaded COLMAP model with bundle adjustment.

Meant for pose-only exports (e.g. ARKit poses + LiDAR points) that are trained
as-is (RUN_RECON=false). Such models have no 2D-3D tracks, so bundle adjustment
needs correspondences first. The known poses are used throughout:

1. Extract SIFT features with the model's own intrinsics (and masks, if any).
2. Match only image pairs that are near each other according to the known
   poses (no exhaustive matching, no pose search).
3. Triangulate points with the input poses held fixed.
4. Bundle-adjust poses and points. Intrinsics and the rig (sensor_from_rig)
   stay fixed.
5. Align the result back to the input camera centres with a similarity
   transform, so the metric (LiDAR) frame and scale are kept.
6. Copy only the refined poses into the input model. Its cameras and rigs are
   unchanged, and per-frame depth maps still match their images.
7. If ARKit depth maps (depths/*.bin) are present, rebuild points3D from them
   with the refined poses, since the input cloud was fused with the old poses.
   Otherwise (or with --no-refuse-lidar) points3D is kept unchanged.

If the model already has tracks, steps 1-3 are skipped. The original model is
backed up to <dataset>/bundle_adjust/original/ and a report is written to
<dataset>/bundle_adjust/report.json. If any step fails or the result does not
pass the quality checks, the original model is kept and the job continues.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

TAG = "[Bundle-Adjust]"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def log(msg):
    print(f"{TAG} {msg}", flush=True)


# --------------------------------------------------------------------------
# Dataset layout
# --------------------------------------------------------------------------

def find_model(dataset):
    """Return (model_dir, fmt) for colmap/sparse/0 or sparse/0; fmt is 'bin' or 'txt'."""
    for rel in ("colmap/sparse/0", "sparse/0"):
        d = os.path.join(dataset, rel)
        for fmt in ("bin", "txt"):
            if all(os.path.isfile(os.path.join(d, f"{f}.{fmt}")) for f in ("cameras", "images", "points3D")):
                return d, fmt
    return None, None


def find_dir(dataset, *candidates):
    for rel in candidates:
        d = os.path.join(dataset, rel)
        if os.path.isdir(d):
            return d
    return None


def find_mask(mask_dir, image_name):
    """Masks follow <image name>.png (COLMAP) or <stem>.png; return the path or None."""
    stem = os.path.splitext(image_name)[0]
    for cand in (image_name + ".png", image_name, stem + ".png"):
        p = os.path.join(mask_dir, cand)
        if os.path.isfile(p):
            return p
    return None


def stage_masks(mask_dir, image_dir, names, out_dir):
    """Write a mask for every image to <out_dir>/<image name>.png, the name COLMAP
    expects. Usable masks are copied. Images whose mask is missing, the wrong
    size or all white get an all-white (keep everything) mask, because COLMAP
    extracts no features for an image without a mask once mask_path is set.
    Returns the number of real masks, or 0 if none (then mask_path is not used)."""
    import cv2
    real, staged = 0, []
    for name in names:
        img_path = os.path.join(image_dir, name)
        if not os.path.isfile(img_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        src = find_mask(mask_dir, name)
        m = cv2.imread(src, cv2.IMREAD_GRAYSCALE) if src else None
        usable = m is not None and m.shape == (h, w) and m.min() <= 127
        if usable:
            real += 1
        else:
            m = np.full((h, w), 255, np.uint8)
        staged.append((name, m))
    if real == 0:
        return 0
    for name, m in staged:
        dst = os.path.join(out_dir, name + ".png")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, m)
    return real


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def has_pose(obj):
    """has_pose is a property in some pycolmap versions and a method in others."""
    v = obj.has_pose
    return bool(v() if callable(v) else v)


def poses(rec):
    """{image_id: (R cam_from_world, centre)} for images with a pose."""
    out = {}
    for iid, img in rec.images.items():
        if not has_pose(img):
            continue
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        out[iid] = (np.asarray(cfw.rotation.matrix()), np.asarray(img.projection_center()))
    return out


def rotation_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def pose_pairs(rec, pose, k, max_angle_deg):
    """Pairs of images likely to overlap: for each image, its k nearest cameras
    (by centre) whose viewing directions differ by less than max_angle_deg.
    Images in the same rig frame are skipped (fixed relative pose, little overlap)."""
    ids = sorted(pose)
    C = np.array([pose[i][1] for i in ids])
    D = np.array([pose[i][0][2] for i in ids])  # camera z axis in world
    frame = {i: rec.images[i].frame_id for i in ids}
    cos_max = np.cos(np.radians(max_angle_deg))
    pairs = set()
    for a, ia in enumerate(ids):
        order = np.argsort(np.linalg.norm(C - C[a], axis=1))
        n = 0
        for b in order:
            ib = ids[b]
            if ib == ia or frame[ib] == frame[ia] or D[a] @ D[b] < cos_max:
                continue
            pairs.add((min(ia, ib), max(ia, ib)))
            n += 1
            if n >= k:
                break
    return pairs


def bearing_vectors(camera, xy):
    """Unit rays in the camera frame; works past 90 deg for fisheye cameras.
    pycolmap < 4.1 (the container has 4.0.4) has no cam_ray_from_img: there,
    cam_from_img gives normalized image coordinates (x, y) and the ray is
    (x, y, 1), which covers every pinhole camera and fisheye rays below 90 deg."""
    if hasattr(camera, "cam_ray_from_img"):
        rays = np.asarray(camera.cam_ray_from_img(xy), float)
    else:
        nrm = np.asarray(camera.cam_from_img(np.asarray(xy, float)), float)
        if nrm.ndim == 2 and nrm.shape[1] == 3:  # already rays
            rays = nrm
        else:
            rays = np.c_[nrm.reshape(-1, 2), np.ones(len(nrm))]
        rays[~np.isfinite(rays).all(axis=1)] = np.nan
    return rays / np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)


def verify_with_poses(pycolmap, rec, db_path, pairs, max_err_deg, min_inliers=15):
    """Keep matches whose angular epipolar error under the input poses is below
    max_err_deg and store them as the pair's verified (CALIBRATED) geometry.
    The tolerance has to be larger than the pose error being corrected."""
    cfw = {}
    for iid, img in rec.images.items():
        p = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        cfw[iid] = (np.asarray(p.rotation.matrix()), np.asarray(p.translation))
    sin_max = np.sin(np.radians(max_err_deg))
    kept = 0
    with pycolmap.Database.open(db_path) as db:
        kps = {}

        def keypoints(iid):
            if iid not in kps:
                kps[iid] = np.asarray(db.read_keypoints(iid))[:, :2].astype(float)
            return kps[iid]

        for a, b in pairs:
            if not db.exists_matches(a, b):
                continue
            m = np.asarray(db.read_matches(a, b))
            if a > b:  # matches are stored for the (smaller, larger) id order
                a, b = b, a
            if len(m) < min_inliers:
                continue
            fa = bearing_vectors(rec.images[a].camera, keypoints(a)[m[:, 0]])
            fb = bearing_vectors(rec.images[b].camera, keypoints(b)[m[:, 1]])
            Ra, ta = cfw[a]
            Rb, tb = cfw[b]
            R = Rb @ Ra.T
            t = tb - R @ ta
            if np.linalg.norm(t) < 1e-6:
                continue
            t = t / np.linalg.norm(t)
            n = np.cross(t, (R @ fa.T).T)  # epipolar plane normals in camera b
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            inl = m[np.abs(np.sum(n * fb, axis=1)) < sin_max]
            if len(inl) < min_inliers:
                continue
            geom = pycolmap.TwoViewGeometry()
            geom.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
            geom.inlier_matches = np.ascontiguousarray(inl, dtype=np.uint32)
            if db.exists_two_view_geometry(a, b):
                db.update_two_view_geometry(a, b, geom)
            else:
                db.write_two_view_geometry(a, b, geom)
            kept += 1
    return kept, len(pairs)


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------

def build_database(pycolmap, rec, db_path):
    """Create a database whose cameras, rigs, frames and images use the model's
    IDs, so feature extraction keeps the model's intrinsics and triangulation
    can match database and model images."""
    if os.path.exists(db_path):
        os.remove(db_path)
    with pycolmap.Database.open(db_path) as db:
        for cam in rec.cameras.values():
            db.write_camera(cam, use_camera_id=True)
        for rig in rec.rigs.values():
            db.write_rig(rig, use_rig_id=True)
        for img in rec.images.values():
            db_img = pycolmap.Image(name=img.name, camera_id=img.camera_id, image_id=img.image_id)
            db.write_image(db_img, use_image_id=True)
        for frame in rec.frames.values():
            db.write_frame(frame, use_frame_id=True)


def triangulate(pycolmap, rec, dataset, image_dir, work, args):
    db_path = os.path.join(work, "database.db")
    build_database(pycolmap, rec, db_path)
    names = [img.name for img in rec.images.values()]

    reader = pycolmap.ImageReaderOptions()
    mask_dir = find_dir(dataset, "masks", "colmap/masks")
    if mask_dir:
        staged = os.path.join(work, "masks")
        n = stage_masks(mask_dir, image_dir, names, staged)
        if n:
            reader.mask_path = staged
        log(f"Masks: {n}/{len(names)} usable from {mask_dir}"
            + (" (all-white masks written for the rest)" if 0 < n < len(names) else ""))
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = args.max_features
    # One CPU SIFT extractor per thread; the default (all cores) aborted on a
    # 32-thread host, so cap it.
    extraction.num_threads = args.num_threads
    log(f"Extracting features for {len(names)} images (model intrinsics kept)")
    pycolmap.extract_features(db_path, image_dir, image_names=names, reader_options=reader,
                              extraction_options=extraction)

    pairs = pose_pairs(rec, poses(rec), args.neighbors, args.max_pair_angle)
    pair_file = os.path.join(work, "pairs.txt")
    with open(pair_file, "w") as f:
        for a, b in sorted(pairs):
            f.write(f"{rec.images[a].name} {rec.images[b].name}\n")
    log(f"Matching {len(pairs)} image pairs chosen from the known poses")
    pairing = pycolmap.ImportedPairingOptions()
    pairing.match_list_path = pair_file
    matching = pycolmap.FeatureMatchingOptions()
    # COLMAP's RANSAC verification rejected every pair with a wide-angle fisheye
    # (X5 lens) in testing, although the matches agreed with the known poses.
    # The poses are known, so verify matches against them instead (below).
    matching.skip_geometric_verification = True
    pycolmap.match_image_pairs(db_path, matching_options=matching, pairing_options=pairing)
    kept, total = verify_with_poses(pycolmap, rec, db_path, pairs, args.max_epipolar_deg)
    log(f"Pose-guided verification: kept {kept}/{total} pairs "
        f"(epipolar error < {args.max_epipolar_deg} deg with the input poses)")

    tri_dir = os.path.join(work, "triangulated")
    os.makedirs(tri_dir, exist_ok=True)
    seed = pycolmap.Reconstruction(rec)  # copy; the input keeps its points3D
    # The triangulator runs its own bundle adjustment with the frame poses fixed.
    # Its defaults refine sensor_from_rig, which moved X5 tiles by up to 13 deg
    # in testing, so keep the rig and intrinsics fixed here too.
    tri_opts = pycolmap.IncrementalPipelineOptions()
    for name in ("ba_refine_sensor_from_rig", "ba_refine_focal_length",
                 "ba_refine_principal_point", "ba_refine_extra_params"):
        if hasattr(tri_opts, name):
            setattr(tri_opts, name, False)
    log("Triangulating with the input poses, rig and intrinsics fixed")
    return pycolmap.triangulate_points(seed, db_path, image_dir, tri_dir, clear_points=True,
                                       options=tri_opts, refine_intrinsics=False)


def ba_options(pycolmap, refine_intrinsics):
    opts = pycolmap.BundleAdjustmentOptions()
    for name, value in (
        ("refine_focal_length", refine_intrinsics),
        ("refine_principal_point", False),
        ("refine_extra_params", refine_intrinsics),
        ("refine_sensor_from_rig", False),
        ("refine_rig_from_world", True),
        ("refine_points3D", True),
    ):
        if hasattr(opts, name):  # option names differ between pycolmap versions
            setattr(opts, name, value)
    return opts


# --------------------------------------------------------------------------
# LiDAR re-fusion
# --------------------------------------------------------------------------

def find_lidar_maps(dataset, rec):
    """Pair ARKit depth maps (depths/<scan>_depth[_smoothed].bin) with model images.
    Rig sessions reuse the scan name for every camera, so prefer the iPhone image."""
    import re
    depth_dir = find_dir(dataset, "depths", "colmap/depths")
    if depth_dir is None:
        return depth_dir, []
    by_stem = {}
    for img in rec.images.values():
        by_stem.setdefault(os.path.splitext(os.path.basename(img.name))[0], []).append(img)
    pairs = []
    for f in sorted(os.listdir(depth_dir)):
        if not f.lower().endswith(".bin"):
            continue
        stem = re.sub(r"_depth(_smoothed)?$", "", os.path.splitext(f)[0])
        cands = by_stem.get(stem, [])
        if len(cands) > 1:
            cands = [i for i in cands if "iphone" in i.name.lower()]
        if len(cands) == 1:
            pairs.append((os.path.join(depth_dir, f), cands[0].image_id))
    return depth_dir, pairs


def load_lidar(path, img_w, img_h, rotation):
    """Headerless little-endian float32 metres at the sensor resolution (256x192 on
    current iPhones), stored landscape; rotate to the image orientation."""
    raw = np.fromfile(path, dtype="<f4")
    long_side, short_side = max(img_w, img_h), min(img_w, img_h)
    h = int(round(np.sqrt(raw.size * short_side / long_side)))
    w = raw.size // h if h else 0
    if h * w != raw.size:
        h, w = 192, 256
    if h * w != raw.size:
        return None
    d = raw.reshape(h, w)
    if rotation == "cw":
        d = np.rot90(d, -1)
    elif rotation == "ccw":
        d = np.rot90(d, 1)
    d = np.nan_to_num(np.ascontiguousarray(d), nan=0.0, posinf=0.0, neginf=0.0)
    d[d < 0] = 0.0
    return d


def backproject(rec, image_id, depth, max_depth, color_img=None, mask=None):
    """World points (and colours) from one z-depth map using the model's pose and
    intrinsics. Pixels at depth discontinuities (flying pixels) are dropped."""
    import cv2
    img = rec.images[image_id]
    cam = img.camera
    dh, dw = depth.shape
    sx, sy = cam.width / dw, cam.height / dh
    # Reject pixels whose 3x3 neighbourhood spans more than 5% of their depth.
    k = np.ones((3, 3), np.uint8)
    spread = cv2.dilate(depth, k) - cv2.erode(depth, k)
    ok = (depth > 0.05) & (depth < max_depth) & (spread < 0.05 * depth)
    if mask is not None:
        ok &= cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST) > 127
    v, u = np.nonzero(ok)
    if len(u) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    xy = np.c_[(u + 0.5) * sx, (v + 0.5) * sy]  # COLMAP pixel centres
    nrm = np.asarray(cam.cam_from_img(xy), float)
    z = depth[v, u][:, None]
    X_cam = np.c_[nrm * z, z]
    cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    R, t = np.asarray(cfw.rotation.matrix()), np.asarray(cfw.translation)
    X_world = (X_cam - t) @ R  # R^T (X - t)
    if color_img is not None:
        px = np.clip(xy.astype(int), 0, [color_img.shape[1] - 1, color_img.shape[0] - 1])
        rgb = color_img[px[:, 1], px[:, 0], ::-1]  # BGR -> RGB
    else:
        rgb = np.full((len(X_world), 3), 128, np.uint8)
    return X_world, rgb


def voxel_downsample(xyz, rgb, target):
    """Average points per voxel, choosing the voxel size so about `target` remain."""
    def run(size):
        keys = np.floor(xyz / size).astype(np.int64)
        _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        P = np.zeros((len(counts), 3))
        C = np.zeros((len(counts), 3))
        np.add.at(P, inv, xyz)
        np.add.at(C, inv, rgb.astype(float))
        return P / counts[:, None], (C / counts[:, None]).astype(np.uint8)

    size = 0.01
    for _ in range(4):  # surface samples: count ~ 1 / size^2
        P, C = run(size)
        if abs(len(P) - target) < 0.1 * target:
            break
        size *= np.sqrt(len(P) / max(target, 1))
    return P, C, size


def refuse_lidar(pycolmap, out, original, dataset, image_dir, args):
    """Rebuild points3D from the ARKit depth maps with the refined poses.
    Returns a dict for the report (status 'refused' or the reason it was skipped)."""
    import cv2
    from scipy.spatial import cKDTree

    depth_dir, maps = find_lidar_maps(dataset, out)
    if not maps:
        return {"status": "skipped", "reason": "no depths/*.bin maps matched to model images"}

    old = np.array([p.xyz for p in original.points3D.values()]) if original.num_points3D() else None
    mask_dir = find_dir(dataset, "masks", "colmap/masks")

    # Pick the depth-map rotation that best reproduces the input cloud on a few
    # frames (with the input poses, which built that cloud).
    def sample_error(rotation, sample):
        errs = []
        tree = cKDTree(old)
        for path, iid in sample:
            cam = original.images[iid].camera
            d = load_lidar(path, cam.width, cam.height, rotation)
            if d is None:
                continue
            X, _ = backproject(original, iid, d, args.max_lidar_depth)
            if len(X):
                errs.append(np.median(tree.query(X[:: max(1, len(X) // 2000)])[0]))
        return float(np.median(errs)) if errs else np.inf

    rotation = os.environ.get("LIDAR_DEPTH_ROTATION", "auto").lower()
    if rotation == "auto":
        if old is not None and len(old):
            sample = maps[:: max(1, len(maps) // 8)][:8]
            scores = {r: sample_error(r, sample) for r in ("cw", "ccw", "none")}
            rotation = min(scores, key=scores.get)
            log(f"LiDAR depth rotation: {rotation} (median distance to input cloud: "
                + ", ".join(f"{r} {s * 100:.1f} cm" for r, s in scores.items()) + ")")
            if scores[rotation] > 0.10:
                return {"status": "skipped", "rotation_scores_m": scores,
                        "reason": f"depth maps do not reproduce the input cloud "
                                  f"({scores[rotation] * 100:.1f} cm median); kept input points"}
        else:
            cam = out.images[maps[0][1]].camera
            rotation = "cw" if cam.height > cam.width else "none"

    xyz, rgb = [], []
    for path, iid in maps:
        img = out.images[iid]
        d = load_lidar(path, img.camera.width, img.camera.height, rotation)
        if d is None:
            continue
        color = cv2.imread(os.path.join(image_dir, img.name), cv2.IMREAD_COLOR)
        mask = None
        if mask_dir:
            mp = find_mask(mask_dir, img.name)
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if mp else None
            if m is not None and m.min() <= 127:
                mask = m
        X, C = backproject(out, iid, d, args.max_lidar_depth, color, mask)
        xyz.append(X)
        rgb.append(C)
    if not xyz or sum(len(x) for x in xyz) == 0:
        return {"status": "skipped", "reason": "no valid depth pixels"}
    xyz, rgb = np.concatenate(xyz), np.concatenate(rgb)
    target = original.num_points3D() if original.num_points3D() else 300000
    P, C, voxel = voxel_downsample(xyz, rgb, target)

    info = {"status": "refused", "rotation": rotation, "depth_maps": len(maps),
            "raw_points": int(len(xyz)), "points": int(len(P)), "voxel_m": float(voxel)}
    if old is not None and len(old):
        d_new = cKDTree(old).query(P[:: max(1, len(P) // 20000)])[0]
        info["median_distance_to_input_cloud_m"] = float(np.median(d_new))

    for pid in list(out.points3D.keys()):
        out.delete_point3D(pid)
    for x, c in zip(P, C):
        out.add_point3D(x, pycolmap.Track(), c)
    log(f"Re-fused LiDAR: {len(maps)} depth maps, {len(xyz)} samples -> {len(P)} points "
        f"(voxel {voxel * 100:.1f} cm), median distance to the input cloud "
        f"{info.get('median_distance_to_input_cloud_m', float('nan')) * 100:.1f} cm")
    return info


def write_model(rec, model_dir, fmt):
    had = {f: os.path.isfile(os.path.join(model_dir, f"{f}.{fmt}")) for f in ("rigs", "frames")}
    if fmt == "bin":
        rec.write_binary(model_dir)
    else:
        rec.write_text(model_dir)
    # Keep 3.x-format inputs in 3.x format: older readers (e.g. gsplat's pycolmap)
    # ignore rig files, but they should not appear where there were none.
    for f, existed in had.items():
        p = os.path.join(model_dir, f"{f}.{fmt}")
        if not existed and os.path.isfile(p):
            os.remove(p)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dataset", required=True, help="Dataset root (DATASET_PATH)")
    ap.add_argument("--refine-intrinsics", action="store_true",
                    help="Also refine focal length and distortion (principal point stays fixed)")
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--num-threads", type=int, default=min(8, os.cpu_count() or 1),
                    help="Feature extraction threads")
    ap.add_argument("--neighbors", type=int, default=20, help="Matched neighbours per image")
    ap.add_argument("--max-pair-angle", type=float, default=75.0,
                    help="Only match cameras whose viewing directions differ by less than this")
    ap.add_argument("--max-epipolar-deg", type=float, default=2.0,
                    help="Match verification tolerance against the input poses; must exceed their error")
    ap.add_argument("--min-image-observations", type=int, default=30,
                    help="Only update poses of images with at least this many triangulated observations")
    ap.add_argument("--min-observations-per-image", type=float, default=100.0,
                    help="Skip triangulation when the model already has this many tracks per image")
    ap.add_argument("--max-rotation-deg", type=float, default=5.0,
                    help="Keep the original if the median camera rotation change exceeds this")
    ap.add_argument("--max-shift-ratio", type=float, default=0.05,
                    help="Keep the original if the median centre shift exceeds this fraction of the camera-path extent")
    ap.add_argument("--no-refuse-lidar", dest="refuse_lidar", action="store_false",
                    help="Keep the input points3D instead of rebuilding them from depths/*.bin "
                         "with the refined poses")
    ap.add_argument("--max-lidar-depth", type=float, default=5.0,
                    help="Ignore LiDAR depth beyond this many metres when re-fusing")
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()

    import pycolmap

    out_dir = os.path.join(args.dataset, "bundle_adjust")
    work = os.path.join(out_dir, "work")
    os.makedirs(work, exist_ok=True)
    report = {}

    def finish(status, reason, **extra):
        report.update(status=status, reason=reason, **extra)
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        log(f"{status}: {reason}")

    model_dir, fmt = find_model(args.dataset)
    image_dir = find_dir(args.dataset, "images", "colmap/images")
    if model_dir is None or image_dir is None:
        finish("skipped", "no COLMAP model (colmap/sparse/0 or sparse/0) or images/ folder found")
        return

    rec = pycolmap.Reconstruction(model_dir)
    before = poses(rec)
    n_obs = int(rec.compute_num_observations())
    report.update(model_dir=model_dir, num_images=len(rec.images), num_posed=len(before),
                  input_observations=n_obs)
    log(f"Model {model_dir}: {len(rec.images)} images, {rec.num_points3D()} points, {n_obs} observations")
    if len(before) < 3:
        finish("skipped", "fewer than 3 posed images")
        return

    backup = os.path.join(out_dir, "original")
    if not os.path.isdir(backup):
        shutil.copytree(model_dir, backup)

    # Steps 1-3: correspondences from the known poses (skipped if tracks exist).
    if n_obs >= args.min_observations_per_image * len(before):
        log("Model already has tracks; running bundle adjustment on them directly")
        tri = pycolmap.Reconstruction(rec)
    else:
        try:
            tri = triangulate(pycolmap, rec, args.dataset, image_dir, work, args)
        except Exception as e:
            finish("kept_original", f"feature extraction/matching/triangulation failed: {e}")
            return
    tri_obs = int(tri.compute_num_observations())
    err_before = float(tri.compute_mean_reprojection_error())
    report.update(triangulated_points=int(tri.num_points3D()), triangulated_observations=tri_obs)
    log(f"Tracks: {tri.num_points3D()} points, {tri_obs} observations, "
        f"mean reprojection error {err_before:.3f} px with the input poses")
    if tri_obs < 50 * len(before):
        finish("kept_original", f"too few observations to refine poses ({tri_obs} for {len(before)} images)")
        return

    # Step 4: bundle adjustment.
    try:
        pycolmap.bundle_adjustment(tri, ba_options(pycolmap, args.refine_intrinsics))
    except Exception as e:
        finish("kept_original", f"bundle adjustment failed: {e}")
        return
    err_after = float(tri.compute_mean_reprojection_error())

    # Only images with enough observations were actually refined. Frames without
    # any such image keep their input pose (BA could not move them, and the
    # alignment below must not move them either).
    refined = poses(tri)
    ids = sorted(i for i in set(before) & set(refined)
                 if tri.images[i].num_points3D >= args.min_image_observations)
    update_frames = {tri.images[i].frame_id for i in ids}
    ids = sorted(i for i in set(before) & set(refined) if tri.images[i].frame_id in update_frames)
    unchanged = sorted(rec.images[i].name for i in before if i not in set(ids))
    report.update(refined_images=len(ids), unchanged_images=len(unchanged))
    if unchanged:
        groups = {}
        for n in unchanged:
            groups[n.split("/")[0]] = groups.get(n.split("/")[0], 0) + 1
        log(f"Keeping input poses for {len(unchanged)} images with fewer than "
            f"{args.min_image_observations} observations: {groups}")
        report["unchanged_by_folder"] = groups
    if len(ids) < 3:
        finish("kept_original", "fewer than 3 images have enough observations to refine")
        return

    # Step 5: align back to the input camera centres (BA leaves position,
    # rotation and scale free).
    sim3 = pycolmap.estimate_sim3d(np.array([refined[i][1] for i in ids]),
                                   np.array([before[i][1] for i in ids]))
    if sim3 is None:
        finish("kept_original", "could not align the refined model back to the input frame")
        return
    tri.transform(sim3)

    # Step 6 (prepare): copy only the refined frame poses into the input model,
    # which keeps its cameras, rigs and points3D. Checks below use these poses.
    out = pycolmap.Reconstruction(rec)
    for fid, frame in out.frames.items():
        if fid in update_frames and fid in tri.frames and has_pose(tri.frames[fid]):
            frame.rig_from_world = tri.frames[fid].rig_from_world
    refined = poses(out)
    # Sanity check: with the rig fixed, the copied frame poses must reproduce the
    # bundle-adjusted image poses.
    tri_poses = poses(tri)
    mismatch = max(rotation_deg(refined[i][0] @ tri_poses[i][0].T) for i in ids if i in tri_poses)
    report["rig_pose_mismatch_deg"] = float(mismatch)
    if mismatch > 0.1:
        finish("kept_original", f"rig changed during refinement (pose mismatch {mismatch:.3f} deg)")
        return

    rot = np.array([rotation_deg(refined[i][0] @ before[i][0].T) for i in ids])
    shift = np.array([np.linalg.norm(refined[i][1] - before[i][1]) for i in ids])
    C = np.array([before[i][1] for i in ids])
    extent = float(np.linalg.norm(C.max(0) - C.min(0))) or 1.0
    report.update(
        reprojection_error_px={"input_poses": err_before, "refined": err_after},
        rotation_change_deg={"median": float(np.median(rot)), "max": float(rot.max())},
        centre_shift={"median": float(np.median(shift)), "max": float(shift.max()),
                      "camera_path_extent": extent},
        alignment_scale=float(sim3.scale),
        per_image={rec.images[i].name: {"rotation_deg": float(r), "shift": float(s)}
                   for i, r, s in zip(ids, rot, shift)},
    )
    log(f"Reprojection error {err_before:.3f} -> {err_after:.3f} px; pose change median "
        f"{np.median(rot):.2f} deg / {np.median(shift):.4f}, max {rot.max():.2f} deg / {shift.max():.4f}; "
        f"alignment scale {sim3.scale:.4f}")

    if err_after > err_before:
        finish("kept_original", f"reprojection error did not improve ({err_before:.3f} -> {err_after:.3f} px)")
        return
    if np.median(rot) > args.max_rotation_deg or np.median(shift) > args.max_shift_ratio * extent:
        finish("kept_original", f"poses moved implausibly far (median {np.median(rot):.2f} deg, "
                                f"{np.median(shift):.4f} shift)")
        return

    # Step 7: the fused LiDAR cloud was built with the input poses, so rebuild it
    # from the per-frame depth maps with the refined poses (gsplat-depth uses these
    # points for its depth loss; every model uses them for initialization).
    if args.refuse_lidar:
        try:
            report["lidar_refusion"] = refuse_lidar(pycolmap, out, rec, args.dataset, image_dir, args)
        except Exception as e:
            report["lidar_refusion"] = {"status": "skipped", "reason": f"failed: {e}"}
        if report["lidar_refusion"]["status"] != "refused":
            log(f"LiDAR re-fusion skipped: {report['lidar_refusion']['reason']}")
            # Keep the input points exactly as they were.
            out = pycolmap.Reconstruction(rec)
            for fid, frame in out.frames.items():
                if fid in update_frames and fid in tri.frames and has_pose(tri.frames[fid]):
                    frame.rig_from_world = tri.frames[fid].rig_from_world

    # Step 6: write the input model with the refined poses.
    try:
        write_model(out, model_dir, fmt)
    except Exception as e:
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.copytree(backup, model_dir)
        finish("kept_original", f"writing the refined model failed ({e}); restored the original")
        return
    finish("refined", f"refined poses written to {model_dir} (original in {backup})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # never fail the job: training proceeds on the original model
        log(f"kept_original: unexpected error: {e}")
    sys.exit(0)
