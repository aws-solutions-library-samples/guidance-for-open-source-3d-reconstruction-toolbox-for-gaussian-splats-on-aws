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

Meant for pose-only exports (e.g. ARKit poses + LiDAR points, or AtlasMobile
iPhone + Insta360 rigs) that are trained as-is (RUN_RECON=false). Such models
have no 2D-3D tracks, so bundle adjustment needs correspondences first. The
known poses are used throughout:

1. Extract SIFT features with the model's own intrinsics (and masks, if any).
   With a COLMAP 4 command-line tool on the PATH (the container's colmap),
   extraction and matching run on the GPU and read WebP directly. Otherwise
   pycolmap runs them on the CPU; it cannot decode WebP, so WebP images are read
   from lossless PNG copies made a batch at a time (--png-batch) and deleted after
   each batch.
2. Match only image pairs that are near each other according to the known
   poses (no exhaustive matching, no pose search). Perspective cameras get
   COLMAP's two-view geometric verification, which does not depend on the
   input poses. Models with a fisheye camera are verified against the input
   poses instead (RANSAC rejected the wide-angle fisheye pairs in testing),
   within --max-epipolar-deg.
3. Triangulate points with the input poses held fixed and loose thresholds
   (--triangulation-max-error, 20 px). Images whose input pose is off by more
   than a few pixels (ARKit drift, an uncalibrated 360-camera mount) keep their
   observations, so step 4 can move them. With COLMAP's default 4 px only the
   observations that already agree with the input poses survive, and bundle
   adjustment cannot correct anything larger.
4. Robust bundle adjustment (Cauchy loss) of the frame poses and points, then
   drop observations above 8 px and adjust again, then 4 px. Intrinsics and the
   rig (sensor_from_rig) stay fixed, so every rig frame (e.g. all tiles of one
   360 shot) moves as one.
5. Put the result back in the input frame: rotation from the camera
   orientations and translation from the camera centres. The scale comes from
   the LiDAR when ARKit depth maps (depths/*.bin) are present (median LiDAR
   depth / triangulated depth), so the depth maps and the re-fused cloud agree
   with the refined geometry; otherwise from a similarity transform on the
   camera centres. (On an iPhone + X5 session the ARKit camera path implied a
   scale 3.7% off the LiDAR's.) A short or nearly straight camera path does not
   fix the rotation of that transform; the rotation then comes from the camera
   orientations.
6. Copy only the refined poses into the input model. Its cameras and rigs are
   unchanged, and per-frame depth maps still match their images.
7. If ARKit depth maps are present, rebuild points3D from them with the refined
   poses, since the input cloud was fused with the old poses: high-confidence
   pixels when raw/iphone/<scan>/confidence.bin is uploaded, no depth edges,
   masked pixels left out, voxels seen by at least two depth samples.
   Otherwise (or with --no-refuse-lidar) points3D is kept unchanged.

A model without rig files whose views belong to one rig (e.g. Rooster archive B:
one camera, one frame per image, six view folders cut from the same fisheye
frames) is first grouped into rig frames from the rig_config.json next to it
(COLMAP rig_configurator format), so the views of one capture move together.
The refined poses are written back in the input's layout.

If the model already has tracks, steps 1-3 are skipped. The original model is
backed up to <dataset>/bundle_adjust/original/ and a report is written to
<dataset>/bundle_adjust/report.json. If any step fails or the result does not
pass the quality checks, the original model is kept and the job continues.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

# One BLAS thread per caller: pycolmap's CPU matcher calls BLAS from one thread per core,
# and a multi-threaded OpenBLAS ran out of buffers there ("BLAS : Bad memory unallocation").
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
try:
    from image_formats import IMAGE_EXTS, pycolmap_can_read, write_png_copy  # /opt/ml/code/image_formats.py (on PYTHONPATH)
except ImportError:  # run from the repo: the module is in src/pipeline/
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from image_formats import IMAGE_EXTS, pycolmap_can_read, write_png_copy

TAG = "[Bundle-Adjust]"


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


def stage_masks(mask_dir, image_dir, names, out_dir, out_names=None, sizes=None):
    """Write a mask for every image to <out_dir>/<image name>.png, the name COLMAP
    expects. Usable masks are copied. Images whose mask is missing, the wrong
    size or all white get an all-white (keep everything) mask, because COLMAP
    extracts no features for an image without a mask once mask_path is set.
    out_names: the names the images have in the database, when they differ from
    names (PNG copies of WebP images); masks are still looked up by names.
    sizes: {name: (width, height)} from the model, so images are not decoded
    just for their size. Masks are written one at a time (a 9672-image upload
    held in memory took ~25 GB). Returns the number of real masks, or 0 if none
    (the staged folder is then removed and mask_path is not used)."""
    import cv2
    out_of = dict(zip(names, out_names)) if out_names else {}
    real = 0
    for name in names:
        img_path = os.path.join(image_dir, name)
        if not os.path.isfile(img_path):
            continue
        size = sizes.get(name) if sizes else None
        if size is None:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape[:2]
        else:
            w, h = int(size[0]), int(size[1])
        src = find_mask(mask_dir, name)
        m = cv2.imread(src, cv2.IMREAD_GRAYSCALE) if src else None
        usable = m is not None and m.shape == (h, w) and m.min() <= 127
        if usable:
            real += 1
        else:
            m = np.full((h, w), 255, np.uint8)
        dst = os.path.join(out_dir, out_of.get(name, name) + ".png")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, m)
    if real == 0:
        shutil.rmtree(out_dir, ignore_errors=True)
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
    """Rotation angle of R in degrees. atan2(|w|, trace - 1) with w = vee(R - R^T) stays
    precise near zero, where arccos((trace - 1) / 2) has a floor of ~0.04 deg for poses
    stored as float32 quaternions (AtlasMobile)."""
    R = np.asarray(R, float)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return float(np.degrees(np.arctan2(np.linalg.norm(w), np.trace(R) - 1.0)))


# Camera models that are not perspective projections (two-view RANSAC verification
# rejected wide-angle fisheye pairs in testing, so those are verified against the poses).
FISHEYE_MODELS = {"SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE",
                  "THIN_PRISM_FISHEYE", "RAD_TAN_THIN_PRISM_FISHEYE", "FOV"}


def has_fisheye(rec):
    return any(str(getattr(c, "model_name", c.model)).split(".")[-1] in FISHEYE_MODELS
               for c in rec.cameras.values())


def verified_pairs(db_path, min_inliers=15):
    """Number of image pairs with a verified two-view geometry of at least min_inliers."""
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        return int(con.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows >= ?",
                               (min_inliers,)).fetchone()[0])
    finally:
        con.close()


def filter_observations(pycolmap, rec, max_error_px, min_tri_angle_deg):
    """Delete observations above max_error_px and points below min_tri_angle_deg.
    Returns the number of observations removed."""
    return int(pycolmap.ObservationManager(rec).filter_all_points3D(max_error_px, min_tri_angle_deg))


def align_to_input(pycolmap, before, refined, ids, scale=None):
    """Sim3 input_from_refined: rotation from the camera orientations (chordal mean
    of R_in^T R_refined), then the scale (unless given) and the translation from the
    camera centres by least squares with that rotation. A similarity fitted to the
    camera centres alone (Umeyama) determines the rotation poorly when the camera
    path is short or nearly straight: on a 16-frame Rooster segment it left the
    whole refined model rotated by 3.2 deg against the input poses and points3D."""
    M = sum(before[i][0].T @ refined[i][0] for i in ids)
    U, _, Vt = np.linalg.svd(M)
    G = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    Cr = np.array([refined[i][1] for i in ids])
    Ci = np.array([before[i][1] for i in ids])
    if scale is None:
        Cr_c = (Cr - Cr.mean(axis=0)) @ G.T
        Ci_c = Ci - Ci.mean(axis=0)
        den = float((Cr_c ** 2).sum())
        scale = float((Ci_c * Cr_c).sum() / den) if den > 0 else 0.0
        if not np.isfinite(scale) or scale <= 0:
            return None
    T = (Ci - scale * Cr @ G.T).mean(axis=0)
    return pycolmap.Sim3d(float(scale), pycolmap.Rotation3d(G), T)


def rig_frame_report(before_rec, after_rec):
    """How the refinement moved multi-image rig frames (e.g. the 8 tiles of one 360 shot)
    relative to the single-image frame of the same capture (the image with the same file
    name in another folder, e.g. face_iphone/scan_014.jpg for the tiles face_0*/scan_014.jpg).
    Per frame M = R_after * R_after_ref^T * R_before_ref * R_before^T: the change of the
    frame's rotation relative to its reference image, in rig axes (for AtlasMobile the iPhone
    camera: x right, y down, z forward). Drift corrections common to both cancel. The mean of
    M over the frames is the 360 camera's mount error; the spread around it is per shot
    (capture timing or motion). None when fewer than 3 frames have a reference."""
    single = {}  # file name -> [single-image frame id]
    multi = []
    for fid, fr in before_rec.frames.items():
        ids = [d.id for d in fr.data_ids]
        if len(ids) == 1:
            single.setdefault(os.path.basename(before_rec.images[ids[0]].name), []).append(fid)
        elif len(ids) > 1:
            multi.append((fid, ids))

    def rot(rec, fid):
        fr = rec.frames.get(fid) if hasattr(rec.frames, "get") else rec.frames[fid]
        if fr is None or not has_pose(fr):
            return None
        return np.asarray(fr.rig_from_world.rotation.matrix())

    Ms, used = [], []
    for fid, ids in multi:
        refs = sorted({r for n in {os.path.basename(before_rec.images[i].name) for i in ids}
                       for r in single.get(n, [])})
        if len(refs) != 1:
            continue
        rb, ra = rot(before_rec, fid), rot(after_rec, fid)
        rb_ref, ra_ref = rot(before_rec, refs[0]), rot(after_rec, refs[0])
        if rb is None or ra is None or rb_ref is None or ra_ref is None:
            continue
        Ms.append(ra @ ra_ref.T @ rb_ref @ rb.T)
        used.append(os.path.basename(before_rec.images[ids[0]].name))
    if len(Ms) < 3:
        return None
    U, _, Vt = np.linalg.svd(np.sum(Ms, axis=0))
    mean = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    spread = np.array([rotation_deg(M @ mean.T) for M in Ms])
    # rotation vector of the mean (rig axes), degrees
    angle = rotation_deg(mean)
    w = np.array([mean[2, 1] - mean[1, 2], mean[0, 2] - mean[2, 0], mean[1, 0] - mean[0, 1]])
    axis = w / np.linalg.norm(w) if np.linalg.norm(w) > 1e-15 else np.zeros(3)
    return {"frames": len(Ms), "common_rotation_deg": angle,
            "common_rotation_vector_deg_rig_axes": (axis * angle).tolist(),
            "per_frame_spread_deg": {"median": float(np.median(spread)), "max": float(spread.max())},
            "per_frame": {n: float(s) for n, s in zip(used, spread)}}


def centre_fit_rotation_deg(sim3, before, refined, ids, n_frames):
    """Uncertainty (deg) of the rotation of a similarity fitted to the camera
    centres: the RMS centre residual over the spread of the centres across their
    main axis (a rotation about that axis moves the centres least), averaged over
    the n_frames independent camera positions. A short or nearly straight path
    cannot fix that rotation: a 16-frame Rooster segment (spread 0.13 m across the
    path) gives 1.6 deg, and its centre fit was 3.2 deg off; the 112-frame sample
    gives 0.1 deg. For a LiDAR rig the centre fit is the right reference
    otherwise, because the LiDAR map follows the LiDAR's positions, not the camera
    orientations (which carry any camera-to-LiDAR rotation error)."""
    Cr = np.array([refined[i][1] for i in ids])
    Ci = np.array([before[i][1] for i in ids])
    R = np.asarray(sim3.rotation.matrix())
    res = Ci - (float(sim3.scale) * Cr @ R.T + np.asarray(sim3.translation))
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    spread = np.linalg.svd(Ci - Ci.mean(axis=0), compute_uv=False) / np.sqrt(len(Ci))
    lever = float(np.hypot(spread[1], spread[2])) if len(spread) >= 3 else 0.0
    if lever <= 0:
        return 90.0
    return float(np.degrees(np.arctan2(rms, lever)) / np.sqrt(max(int(n_frames), 1)))


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


def covisible_pairs(rec, mask_dir, k, min_score=0.08, max_points=80000, cell_px=16, seed=0):
    """Pairs of images that see the same part of the model's points3D (e.g. the fused
    LiDAR cloud), from the known poses: each image's visible points (z-buffer on cell_px
    cells, inside its mask), pair score = shared points / visible points of the image
    that sees more of them; for each image its best k pairs above min_score. Images in
    the same rig frame are skipped. Unlike pose_pairs (nearest camera centres) this
    pairs a 360 tile with the phone images that look at the same wall, wherever they
    were taken. Returns None when the model has no points."""
    import cv2
    from scipy.sparse import csr_matrix

    if rec.num_points3D() == 0:
        return None
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(rec.point3D_ids()), dtype=np.int64)
    if len(pids) > max_points:
        pids = rng.choice(pids, max_points, replace=False)
    X = np.array([rec.points3D[int(p)].xyz for p in pids], dtype=np.float64)
    ids = sorted(i for i, img in rec.images.items() if has_pose(img))
    rows, cols = [], []
    for r, iid in enumerate(ids):
        img = rec.images[iid]
        cam = img.camera
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        Xc = X @ np.asarray(cfw.rotation.matrix()).T + np.asarray(cfw.translation)
        front = np.flatnonzero(Xc[:, 2] > 0.05)
        if not len(front):
            continue
        uv = np.asarray(cam.img_from_cam(Xc[front]), dtype=np.float64).reshape(-1, 2)
        w, h = int(cam.width), int(cam.height)
        ok = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < w) & (uv[:, 1] < h)
        sel, uv, z = front[ok], uv[ok], Xc[front[ok], 2]
        if mask_dir:
            mp = find_mask(mask_dir, img.name)
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if mp else None
            if m is not None:
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                keep = m[uv[:, 1].astype(int), uv[:, 0].astype(int)] > 127
                sel, uv, z = sel[keep], uv[keep], z[keep]
        if not len(sel):
            continue
        gw, gh = (w + cell_px - 1) // cell_px, (h + cell_px - 1) // cell_px
        cell = (uv[:, 1] // cell_px).astype(np.int64) * gw + (uv[:, 0] // cell_px).astype(np.int64)
        zbuf = np.full(gw * gh, np.inf)
        np.minimum.at(zbuf, cell, z)
        zmin = cv2.erode(zbuf.reshape(gh, gw).astype(np.float32), np.ones((3, 3), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=float(np.inf)).reshape(-1)
        vis = sel[z <= zmin[cell] * 1.05 + 0.02]
        rows.append(np.full(len(vis), r))
        cols.append(vis)
    if not rows:
        return set()
    # float32: the dense image x image score matrices of a 9672-image upload stay
    # around 1 GB (shared-point counts are small integers, exact in float32).
    V = csr_matrix((np.ones(sum(len(c) for c in cols), np.float32),
                    (np.concatenate(rows), np.concatenate(cols))), shape=(len(ids), len(pids)))
    S = (V @ V.T).toarray()
    n = np.maximum(np.diag(S), 1)
    score = np.minimum(S / n[:, None], S / n[None, :])
    np.fill_diagonal(score, 0)
    frame = np.array([rec.images[i].frame_id for i in ids])
    score[frame[:, None] == frame[None, :]] = 0
    pairs = set()
    for a in range(len(ids)):
        for b in np.argsort(-score[a])[:k]:
            if score[a, b] >= min_score:
                pairs.add((min(ids[a], ids[b]), max(ids[a], ids[b])))
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


def extract_from_png_copies(pycolmap, db_path, image_dir, names, db_name, work, reader, extraction,
                            batch, threads):
    """pycolmap.extract_features for images pycolmap cannot decode (WebP): each
    batch of images is written as lossless PNG copies under its database names
    (db_name[n], e.g. y000/000000.webp.png), features are extracted, and the
    copies are deleted, so at most one batch sits on disk. Images pycolmap can
    read are hard-linked (or copied) next to them."""
    from concurrent.futures import ThreadPoolExecutor
    copy_dir = os.path.join(work, "pycolmap_images")

    def stage(n):
        src, dst = os.path.join(image_dir, n), os.path.join(copy_dir, db_name[n])
        if db_name[n] != n:
            if not write_png_copy(src, dst):
                raise RuntimeError(f"cannot read {src}")
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    batch = max(1, int(batch))
    for i in range(0, len(names), batch):
        chunk = names[i:i + batch]
        with ThreadPoolExecutor(max(1, int(threads))) as ex:
            list(ex.map(stage, chunk))
        pycolmap.extract_features(db_path, copy_dir, image_names=[db_name[n] for n in chunk],
                                  reader_options=reader, extraction_options=extraction)
        shutil.rmtree(copy_dir, ignore_errors=True)


def colmap_cli(name):
    """Path of a COLMAP 4 (or later) command-line tool for feature extraction and
    matching, or None. It runs both on the GPU and reads WebP through OpenImageIO
    (pycolmap does neither in the container). The database is written by pycolmap
    4, whose schema a 3.x tool cannot use, so older tools are skipped."""
    if not name or str(name).lower() == "none":
        return None
    path = shutil.which(name) or (name if os.path.isfile(name) else None)
    if not path:
        return None
    try:
        r = subprocess.run([path, "help"], capture_output=True, text=True, timeout=120)
        m = re.search(r"COLMAP (\d+)\.(\d+)", (r.stdout or "") + (r.stderr or ""))
    except (OSError, subprocess.SubprocessError):
        m = None
    if not m or int(m.group(1)) < 4:
        log(f"Not using {path} for features and matching: needs COLMAP 4 or later"
            + (f" (found {m.group(0)})" if m else ""))
        return None
    return path


def run_colmap_cli(colmap, attempts, log_path):
    """Run the colmap tool with the first option set that succeeds (GPU, then CPU).
    A failed attempt leaves the database usable: extraction skips images that
    already have features, and matching skips pairs that already have matches.
    The output goes to log_path. Returns the index of the attempt that succeeded."""
    tail = ""
    for n, cmd in enumerate(attempts):
        with open(log_path, "a", errors="replace") as f:
            f.write(f"\n$ colmap {' '.join(cmd)}\n")
            f.flush()
            r = subprocess.run([colmap] + cmd, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode == 0:
            return n
        with open(log_path, errors="replace") as f:
            tail = f.read()[-2000:]
        log(f"colmap {cmd[0]} failed (exit code {r.returncode})"
            + ("; retrying on the CPU" if n + 1 < len(attempts) else ""))
    raise RuntimeError(f"colmap {attempts[0][0]} failed: {' '.join(tail.split())[-600:]}")


def images_with_keypoints(db_path):
    """Number of images in the database that have keypoints."""
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        return int(con.execute("SELECT COUNT(*) FROM keypoints WHERE rows > 0").fetchone()[0])
    finally:
        con.close()


def triangulate(pycolmap, rec, dataset, image_dir, work, args, colmap="auto"):
    db_path = os.path.join(work, "database.db")
    names = [img.name for img in rec.images.values()]
    # Feature extraction and matching run in the colmap tool when there is one
    # (GPU; it reads WebP). pycolmap runs them on the CPU and cannot decode WebP:
    # those images then get lossless PNG copies named <name>.png, and the
    # database, the pair list and the triangulation model use these names; the
    # result is renamed back before it is returned.
    if colmap == "auto":
        colmap = colmap_cli(args.colmap)
    use_gpu = args.gpu != "off"
    for f in (db_path + "-wal", db_path + "-shm"):
        if os.path.exists(f):
            os.remove(f)
    shutil.rmtree(os.path.join(work, "masks"), ignore_errors=True)
    if colmap:
        db_name = {n: n for n in names}
    else:
        db_name = {n: (n if pycolmap_can_read(n) else n + ".png") for n in names}
    renamed = [n for n in names if db_name[n] != n]
    rec_db = rec
    if renamed:
        rec_db = pycolmap.Reconstruction(rec)  # copy; rec keeps the real names
        for img in rec_db.images.values():
            img.name = db_name[img.name]
    build_database(pycolmap, rec_db, db_path)

    reader = pycolmap.ImageReaderOptions()
    mask_dir = find_dir(dataset, "masks", "colmap/masks")
    staged = None
    if mask_dir:
        staged = os.path.join(work, "masks")
        sizes = {img.name: (rec.cameras[img.camera_id].width, rec.cameras[img.camera_id].height)
                 for img in rec.images.values()}
        n = stage_masks(mask_dir, image_dir, names, staged, out_names=[db_name[x] for x in names],
                        sizes=sizes)
        if n:
            reader.mask_path = staged
        else:
            staged = None
        log(f"Masks: {n}/{len(names)} usable from {mask_dir}"
            + (" (all-white masks written for the rest)" if 0 < n < len(names) else ""))
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = args.max_features
    # One CPU SIFT extractor per thread; the default (all cores) aborted on a
    # 32-thread host, so cap it.
    extraction.num_threads = args.num_threads
    log(f"Extracting features for {len(names)} images (model intrinsics kept)")
    t0 = time.time()
    if colmap:
        image_list = os.path.join(work, "images.txt")
        with open(image_list, "w") as f:
            f.write("\n".join(names) + "\n")
        base = ["feature_extractor", "--database_path", db_path, "--image_path", image_dir,
                "--image_list_path", image_list, "--SiftExtraction.max_num_features", str(args.max_features)]
        if staged:
            base += ["--ImageReader.mask_path", staged]
        attempts = ([base + ["--FeatureExtraction.use_gpu", "1"]] if use_gpu else []) + [
            base + ["--FeatureExtraction.use_gpu", "0", "--FeatureExtraction.num_threads", str(args.num_threads)]]
        k = run_colmap_cli(colmap, attempts, os.path.join(work, "colmap_features.log"))
        how = f"colmap, {'GPU' if use_gpu and k == 0 else 'CPU'}"
    elif renamed:
        exts = sorted({os.path.splitext(n)[1].lower() for n in renamed})
        log(f"pycolmap cannot decode {', '.join(exts)}: reading {len(renamed)} images from lossless "
            f"PNG copies, {args.png_batch} at a time")
        extract_from_png_copies(pycolmap, db_path, image_dir, names, db_name, work, reader, extraction,
                                args.png_batch, args.num_threads)
        how = "pycolmap, CPU"
    else:
        pycolmap.extract_features(db_path, image_dir, image_names=names, reader_options=reader,
                                  extraction_options=extraction)
        how = "pycolmap, CPU"
    n_kp = images_with_keypoints(db_path)
    log(f"Features for {n_kp}/{len(names)} images in {time.time() - t0:.0f} s ({how})")
    if colmap and n_kp < 0.5 * len(names):
        # e.g. a COLMAP build whose OpenImageIO has no WebP plugin (the Windows
        # 4.2.1 release): start over with pycolmap, which reads PNG copies.
        log(f"colmap read only {n_kp}/{len(names)} images (see colmap_features.log); "
            f"extracting and matching with pycolmap on the CPU instead")
        return triangulate(pycolmap, rec, dataset, image_dir, work, args, colmap=None)
    if n_kp == 0:
        raise RuntimeError("no features were extracted")

    pairs = covisible_pairs(rec, mask_dir, args.covisible_pairs) if args.covisible_pairs > 0 else None
    if pairs:
        log(f"Pairs from the known poses and the model's points: {len(pairs)} image pairs that see "
            f"the same points (best {args.covisible_pairs} per image)")
    else:
        pairs = pose_pairs(rec, poses(rec), args.neighbors, args.max_pair_angle)
    pair_file = os.path.join(work, "pairs.txt")
    with open(pair_file, "w") as f:
        for a, b in sorted(pairs):
            f.write(f"{rec_db.images[a].name} {rec_db.images[b].name}\n")
    log(f"Matching {len(pairs)} image pairs chosen from the known poses")
    # COLMAP's RANSAC verification rejected every pair with a wide-angle fisheye
    # (X5 lens) in testing, although the matches agreed with the known poses.
    # The poses are known, so fisheye matches are verified against them (below).
    fisheye = has_fisheye(rec)
    t0 = time.time()
    if colmap:
        base = ["matches_importer", "--database_path", db_path, "--match_list_path", pair_file,
                "--match_type", "pairs"]
        if fisheye:
            base += ["--FeatureMatching.skip_geometric_verification", "1"]
        attempts = ([base + ["--FeatureMatching.use_gpu", "1"]] if use_gpu else []) + [
            base + ["--FeatureMatching.use_gpu", "0"]]
        k = run_colmap_cli(colmap, attempts, os.path.join(work, "colmap_matching.log"))
        how = f"colmap, {'GPU' if use_gpu and k == 0 else 'CPU'}"
    else:
        pairing = pycolmap.ImportedPairingOptions()
        pairing.match_list_path = pair_file
        matching = pycolmap.FeatureMatchingOptions()
        if fisheye:
            matching.skip_geometric_verification = True
        pycolmap.match_image_pairs(db_path, matching_options=matching, pairing_options=pairing)
        how = "pycolmap, CPU"
    log(f"Matched in {time.time() - t0:.0f} s ({how})")
    if fisheye:
        kept, total = verify_with_poses(pycolmap, rec, db_path, pairs, args.max_epipolar_deg)
        log(f"Pose-guided verification (fisheye camera in the model): kept {kept}/{total} pairs "
            f"(epipolar error < {args.max_epipolar_deg} deg with the input poses)")
    else:
        # Perspective cameras: COLMAP's own two-view verification. It does not
        # depend on the input poses, so matches of badly posed images (e.g. 360
        # tiles of an uncalibrated mount) are kept for the refinement to use.
        log(f"Two-view geometric verification (independent of the input poses): "
            f"{verified_pairs(db_path)}/{len(pairs)} pairs with >= 15 inliers")

    tri_dir = os.path.join(work, "triangulated")
    os.makedirs(tri_dir, exist_ok=True)
    seed = pycolmap.Reconstruction(rec_db)  # copy; the input keeps its points3D
    # The triangulator runs its own bundle adjustment with the frame poses fixed.
    # Its defaults refine sensor_from_rig, which moved X5 tiles by up to 13 deg
    # in testing, so keep the rig and intrinsics fixed here too.
    tri_opts = pycolmap.IncrementalPipelineOptions()
    for name in ("ba_refine_sensor_from_rig", "ba_refine_focal_length",
                 "ba_refine_principal_point", "ba_refine_extra_params"):
        if hasattr(tri_opts, name):
            setattr(tri_opts, name, False)
    # Loose thresholds: with the poses fixed, COLMAP's defaults (4 px) drop every
    # observation of an image whose input pose is a few pixels off, and bundle
    # adjustment then has nothing to correct it with. The robust adjustment in
    # main() removes the observations that stay far off.
    loose = float(args.triangulation_max_error)
    tri_o = tri_opts.triangulation
    tri_o.merge_max_reproj_error = loose
    tri_o.complete_max_reproj_error = loose
    tri_o.create_max_angle_error = max(tri_o.create_max_angle_error, 4.0)
    tri_o.continue_max_angle_error = max(tri_o.continue_max_angle_error, 4.0)
    tri_opts.triangulation = tri_o
    mapper = tri_opts.mapper
    mapper.filter_max_reproj_error = loose
    mapper.filter_min_tri_angle = 1.0
    tri_opts.mapper = mapper
    log(f"Triangulating with the input poses, rig and intrinsics fixed "
        f"(observations kept up to {loose:g} px)")
    # The triangulator reads the images only to colour the points, which are
    # discarded here. pycolmap cannot read WebP (and the PNG copies are gone), so
    # it would warn once per image.
    glog = getattr(pycolmap, "logging", None) if any(not pycolmap_can_read(n) for n in names) else None
    level = getattr(glog, "minloglevel", None)
    if level is not None:
        glog.minloglevel = 2  # errors only
    try:
        tri = pycolmap.triangulate_points(seed, db_path, image_dir, tri_dir, clear_points=True,
                                          options=tri_opts, refine_intrinsics=False)
    finally:
        if level is not None:
            glog.minloglevel = level
    if renamed:
        real_name = {img.image_id: img.name for img in rec.images.values()}
        for img in tri.images.values():
            img.name = real_name[img.image_id]
    return tri


def ba_options(pycolmap, refine_intrinsics, loss_scale=None):
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
    if loss_scale:
        # Cauchy loss (scale in px): observations of images that start far off do not
        # dominate the first rounds. pycolmap 4.x keeps it in options.ceres.
        ceres = getattr(opts, "ceres", None)
        holder = opts if ceres is None else ceres
        if hasattr(holder, "loss_function_type") and hasattr(pycolmap, "LossFunctionType"):
            holder.loss_function_type = pycolmap.LossFunctionType.CAUCHY
            holder.loss_function_scale = float(loss_scale)
            if ceres is not None:
                opts.ceres = holder
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


def voxel_downsample(xyz, rgb, target, min_count=1):
    """Average points per voxel, choosing the voxel size so about `target` remain.
    Voxels with fewer than min_count samples (e.g. one stray depth pixel) are dropped."""
    def run(size):
        keys = np.floor(xyz / size).astype(np.int64)
        _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        P = np.zeros((len(counts), 3))
        C = np.zeros((len(counts), 3))
        np.add.at(P, inv, xyz)
        np.add.at(C, inv, rgb.astype(float))
        keep = counts >= min_count
        return P[keep] / counts[keep, None], (C[keep] / counts[keep, None]).astype(np.uint8)

    size = 0.01
    for _ in range(4):  # surface samples: count ~ 1 / size^2
        P, C = run(size)
        if abs(len(P) - target) < 0.1 * target:
            break
        size *= np.sqrt(len(P) / max(target, 1))
    return P, C, size


def load_confidence(dataset, depth_path, depth_shape, rotation):
    """ARKit depth confidence (0 low, 1 medium, 2 high) for a depth map, read from the uploaded
    session's raw/iphone/<scan>/confidence.bin; None when it is not there."""
    import re
    stem = re.sub(r"_depth(_smoothed)?$", "", os.path.splitext(os.path.basename(depth_path))[0])
    p = os.path.join(dataset, "raw", "iphone", stem, "confidence.bin")
    if not os.path.isfile(p):
        return None
    raw = np.fromfile(p, dtype=np.uint8)
    h, w = depth_shape if rotation == "none" else depth_shape[::-1]
    if raw.size != h * w:
        return None
    c = raw.reshape(h, w)
    if rotation == "cw":
        c = np.rot90(c, -1)
    elif rotation == "ccw":
        c = np.rot90(c, 1)
    return np.ascontiguousarray(c)


def prepare_lidar(original, dataset, args):
    """Pair the ARKit depth maps with model images and find their rotation to the image
    orientation (the one that best reproduces the input cloud with the input poses).
    Returns {"maps": [(path, image_id)], "rotation": ...} or {"maps": [], "reason": ...}."""
    from scipy.spatial import cKDTree

    _, maps = find_lidar_maps(dataset, original)
    if not maps:
        return {"maps": [], "reason": "no depths/*.bin maps matched to model images"}
    old = np.array([p.xyz for p in original.points3D.values()]) if original.num_points3D() else None

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

    info = {"maps": maps}
    rotation = os.environ.get("LIDAR_DEPTH_ROTATION", "auto").lower()
    if rotation == "auto":
        if old is not None and len(old):
            sample = maps[:: max(1, len(maps) // 8)][:8]
            scores = {r: sample_error(r, sample) for r in ("cw", "ccw", "none")}
            rotation = min(scores, key=scores.get)
            info["rotation_scores_m"] = scores
            log(f"LiDAR depth rotation: {rotation} (median distance to input cloud: "
                + ", ".join(f"{r} {s * 100:.1f} cm" for r, s in scores.items()) + ")")
            if scores[rotation] > 0.10:
                return {"maps": [], "rotation_scores_m": scores,
                        "reason": f"depth maps do not reproduce the input cloud "
                                  f"({scores[rotation] * 100:.1f} cm median)"}
        else:
            cam = original.images[maps[0][1]].camera
            rotation = "cw" if cam.height > cam.width else "none"
    info["rotation"] = rotation
    return info


def lidar_depth_ok(dataset, path, d, rotation, max_depth, edge_rel):
    """Depth pixels to use: in range, no depth edge (3x3 spread < edge_rel of the depth) and,
    when the confidence map was uploaded, ARKit's high confidence. -> (mask, has_confidence)"""
    import cv2
    k = np.ones((3, 3), np.uint8)
    spread = cv2.dilate(d, k) - cv2.erode(d, k)
    ok = (d > 0.15) & (d < max_depth) & (spread < edge_rel * d)
    conf = load_confidence(dataset, path, d.shape, rotation)
    if conf is not None and conf.shape == d.shape:
        ok &= conf >= 2
    return ok, conf is not None


def lidar_scale(rec, lidar, dataset, args, edge_rel=0.03, min_image_samples=30, max_factor_dev=0.05):
    """Metres per model unit: median LiDAR depth / triangulated depth at the observations of
    the images with depth maps (points seen by at least 3 images). ARKit's depth scale also
    varies a little from frame to frame, so each image with enough samples gets its own factor
    (its median ratio / the global one, within +-max_factor_dev) for the re-fusion.
    Returns dict(scale (None with fewer than 200 samples), samples, iqr, factors={image_id: f})."""
    ratios, per_image = [], {}
    for path, iid in lidar["maps"]:
        if iid not in rec.images or not has_pose(rec.images[iid]):
            continue
        img = rec.images[iid]
        cam = img.camera
        d = load_lidar(path, cam.width, cam.height, lidar["rotation"])
        if d is None:
            continue
        ok, _ = lidar_depth_ok(dataset, path, d, lidar["rotation"], args.max_lidar_depth, edge_rel)
        obs = [(p.xy, rec.points3D[p.point3D_id].xyz) for p in img.points2D
               if p.has_point3D() and rec.points3D[p.point3D_id].track.length() >= 3]
        if not obs:
            continue
        xy = np.array([o[0] for o in obs], float)
        X = np.array([o[1] for o in obs], float)
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        z = X @ np.asarray(cfw.rotation.matrix()).T + np.asarray(cfw.translation)
        z = z[:, 2]
        dh, dw = d.shape
        j = np.clip((xy[:, 0] * dw / cam.width).astype(int), 0, dw - 1)
        i = np.clip((xy[:, 1] * dh / cam.height).astype(int), 0, dh - 1)
        sel = ok[i, j] & (z > 1e-6)
        ratios.append(d[i[sel], j[sel]] / z[sel])
        per_image[iid] = ratios[-1]
    r = np.concatenate(ratios) if ratios else np.zeros(0)
    if len(r) < 200:
        return {"scale": None, "samples": int(len(r)), "iqr": None, "factors": {}}
    s = float(np.median(r))
    factors = {iid: float(np.clip(np.median(v) / s, 1 - max_factor_dev, 1 + max_factor_dev))
               for iid, v in per_image.items() if len(v) >= min_image_samples}
    return {"scale": s, "samples": int(len(r)), "factors": factors,
            "iqr": (float(np.percentile(r, 25)), float(np.percentile(r, 75)))}


def refuse_lidar(pycolmap, out, original, dataset, image_dir, args, lidar):
    """Rebuild points3D from the ARKit depth maps with the refined poses (lidar: prepare_lidar()).
    Returns a dict for the report (status 'refused' or the reason it was skipped)."""
    import cv2
    from scipy.spatial import cKDTree

    if not lidar.get("maps"):
        return {"status": "skipped", "reason": lidar.get("reason", "no depth maps"),
                **({"rotation_scores_m": lidar["rotation_scores_m"]} if "rotation_scores_m" in lidar else {})}
    maps, rotation = lidar["maps"], lidar["rotation"]
    old = np.array([p.xyz for p in original.points3D.values()]) if original.num_points3D() else None
    mask_dir = find_dir(dataset, "masks", "colmap/masks")

    xyz, rgb, n_conf = [], [], 0
    for path, iid in maps:
        img = out.images[iid]
        d = load_lidar(path, img.camera.width, img.camera.height, rotation)
        if d is None:
            continue
        ok, has_conf = lidar_depth_ok(dataset, path, d, rotation, args.max_lidar_depth, 0.05)
        n_conf += has_conf
        # Per-frame depth factor from the scale fit (ARKit's depth scale drifts ~1%).
        d = np.where(ok, d / lidar.get("factors", {}).get(iid, 1.0), 0.0).astype(np.float32)
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
    # Keep voxels seen by at least two depth samples: one stray pixel makes no surface.
    P, C, voxel = voxel_downsample(xyz, rgb, target, min_count=2)

    info = {"status": "refused", "rotation": rotation, "depth_maps": len(maps),
            "confidence_maps": int(n_conf), "raw_points": int(len(xyz)), "points": int(len(P)),
            "voxel_m": float(voxel)}
    if old is not None and len(old):
        d_new = cKDTree(old).query(P[:: max(1, len(P) // 20000)])[0]
        info["median_distance_to_input_cloud_m"] = float(np.median(d_new))

    for pid in list(out.points3D.keys()):
        out.delete_point3D(pid)
    for x, c in zip(P, C):
        out.add_point3D(x, pycolmap.Track(), c)
    log(f"Re-fused LiDAR: {len(maps)} depth maps ({n_conf} with confidence), {len(xyz)} samples "
        f"-> {len(P)} points (voxel {voxel * 100:.1f} cm, >= 2 samples each), median distance to "
        f"the input cloud {info.get('median_distance_to_input_cloud_m', float('nan')) * 100:.1f} cm")
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


def cam_from_world(img):
    """Image pose (a method in pycolmap 4.x, a property in some earlier versions)."""
    return img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world


def rig_from_config(pycolmap, rec, dataset):
    """Group the images of a model without rig files into rig frames, from the
    rig_config.json (COLMAP rig_configurator format) next to it. Rooster archive B
    is such a model: one camera, one frame per image, six view folders cut from
    the same two fisheye frames. The rig model has one camera per view folder (the
    model's intrinsics), one rig frame per capture (the same file name in every
    view folder), sensor_from_rig from the file (the mean over the captures where
    the file has none) and the reference view's pose as the frame pose, so bundle
    adjustment moves the views of one capture together. It must reproduce every
    input image pose, otherwise it is not used. Returns (rig model, info) or
    (None, reason); reason is None when there is no rig_config.json."""
    path = next((p for p in (os.path.join(dataset, "rig_config.json"),
                             os.path.join(dataset, "colmap", "rig_config.json")) if os.path.isfile(p)), None)
    if path is None:
        return None, None
    images = list(rec.images.values())
    # (rec.rigs is not touched: pycolmap 4.0.4 on Windows crashes on it, COLMAP #3798)
    if len({img.frame_id for img in images}) < len(images):
        return None, "the model already groups its images into rig frames"
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        return None, f"cannot read {path}: {e}"
    rigs = cfg if isinstance(cfg, list) else [cfg]
    sensors = rigs[0].get("cameras", []) if len(rigs) == 1 and isinstance(rigs[0], dict) else []
    prefixes = [str(s.get("image_prefix", "")) for s in sensors]
    if len(sensors) < 2 or not all(prefixes) or len(set(prefixes)) != len(prefixes):
        return None, "rig_config.json does not describe one rig of two or more cameras with their own image_prefix"
    refs = [k for k, s in enumerate(sensors) if s.get("ref_sensor")]
    ref = refs[0] if refs else 0
    views = [{} for _ in sensors]  # per view: capture key (name after the prefix) -> image
    for img in images:
        k = max((k for k, p in enumerate(prefixes) if img.name.startswith(p)),
                key=lambda j: len(prefixes[j]), default=None)
        if k is None:
            return None, f"{img.name} is in none of the rig_config.json image_prefix folders"
        views[k][img.name[len(prefixes[k]):]] = img
    keys = sorted(views[ref])
    if any(sorted(v) != keys for v in views):
        return None, "the view folders do not hold the same captures"
    cams = []
    for v in views:
        ids = {img.camera_id for img in v.values()}
        if len(ids) != 1:
            return None, "the images of one view folder use different cameras"
        cams.append(rec.cameras[ids.pop()])

    def pose(img):
        p = cam_from_world(img)
        return np.asarray(p.rotation.matrix()), np.asarray(p.translation)

    sensor_from_rig, source = [], []
    for k, s in enumerate(sensors):
        q, t = s.get("cam_from_rig_rotation"), s.get("cam_from_rig_translation")
        if k == ref:
            sensor_from_rig.append((np.eye(3), np.zeros(3)))
            source.append("reference")
        elif q is not None and t is not None:  # quaternion w, x, y, z
            R = np.asarray(pycolmap.Rotation3d(np.array([q[1], q[2], q[3], q[0]], float)).matrix())
            sensor_from_rig.append((R, np.asarray(t, float)))
            source.append("rig_config.json")
        else:  # cam_from_rig = cam_from_world * inverse(ref_from_world), averaged
            Rs, ts = [], []
            for key in keys:
                Rv, tv = pose(views[k][key])
                Rr, tr = pose(views[ref][key])
                Rs.append(Rv @ Rr.T)
                ts.append(tv - Rv @ Rr.T @ tr)
            U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
            sensor_from_rig.append((U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt, np.mean(ts, axis=0)))
            source.append("estimated from the poses")

    rig_rec = pycolmap.Reconstruction()
    for k, cam in enumerate(cams):
        model = getattr(cam, "model_name", None) or str(cam.model).split(".")[-1]
        rig_rec.add_camera(pycolmap.Camera(model=model, width=int(cam.width), height=int(cam.height),
                                           params=list(cam.params), camera_id=k + 1))

    def sensor(k):
        return pycolmap.sensor_t(pycolmap.SensorType.CAMERA, k + 1)

    rig = pycolmap.Rig()
    rig.rig_id = 1
    rig.add_ref_sensor(sensor(ref))
    for k, (R, t) in enumerate(sensor_from_rig):
        if k != ref:
            rig.add_sensor(sensor(k), pycolmap.Rigid3d(pycolmap.Rotation3d(R), t))
    rig_rec.add_rig(rig)
    for fid, key in enumerate(keys, start=1):
        frame = pycolmap.Frame()
        frame.frame_id = fid
        frame.rig_id = 1
        frame.rig_from_world = cam_from_world(views[ref][key])
        for k in range(len(sensors)):
            frame.add_data_id(pycolmap.data_t(sensor(k), views[k][key].image_id))
        rig_rec.add_frame(frame)
        for k in range(len(sensors)):
            src = views[k][key]
            img = pycolmap.Image(name=src.name, camera_id=k + 1, image_id=src.image_id)
            img.frame_id = fid
            rig_rec.add_image(img)
    for p in rec.points3D.values():
        rig_rec.add_point3D(p.xyz, pycolmap.Track(), p.color)

    # The rig has to reproduce the input poses (same rig, same units).
    worst_r, worst_c = 0.0, 0.0
    for img in images:
        Ri, ti = pose(img)
        Rr, tr = pose(rig_rec.images[img.image_id])
        worst_r = max(worst_r, rotation_deg(Rr @ Ri.T))
        worst_c = max(worst_c, float(np.linalg.norm(Ri.T @ ti - Rr.T @ tr)))
    offsets = [float(np.linalg.norm(t)) for _, t in sensor_from_rig]
    C = np.array([-pose(img)[0].T @ pose(img)[1] for img in images])
    tol_c = 0.1 * max(offsets) if max(offsets) > 0 else 1e-6 * (float(np.linalg.norm(C.max(0) - C.min(0))) or 1.0)
    info = {"source": path, "views": [p.rstrip("/") for p in prefixes], "reference_view": prefixes[ref].rstrip("/"),
            "frames": len(keys), "sensor_from_rig": dict(zip([p.rstrip("/") for p in prefixes], source)),
            "max_pose_error_deg": worst_r, "max_centre_error": worst_c}
    if worst_r > 0.05 or worst_c > tol_c:
        return None, (f"rig_config.json does not reproduce the model's poses (up to {worst_r:.3f} deg, "
                      f"{worst_c:.4f} centre offset)")
    return rig_rec, info


def to_input_layout(pycolmap, rec_in, refined, new_points=False):
    """The input model (one frame per image) with the image poses of `refined`, a
    rig model from rig_from_config with the same image IDs. Each input frame holds
    one image whose camera is the frame's reference sensor, so the frame pose is
    the image pose. new_points: also take refined's points3D (a re-fused cloud)."""
    out = pycolmap.Reconstruction(rec_in)
    image_of_frame = {img.frame_id: img.image_id for img in out.images.values()}
    for fid, frame in out.frames.items():
        frame.rig_from_world = cam_from_world(refined.images[image_of_frame[fid]])
    worst = max(rotation_deg(np.asarray(cam_from_world(img).rotation.matrix())
                             @ np.asarray(cam_from_world(refined.images[img.image_id]).rotation.matrix()).T)
                for img in out.images.values())
    if worst > 0.01:
        raise RuntimeError(f"the input model's frames do not hold one image each (pose mismatch {worst:.3f} deg)")
    if new_points:
        for pid in list(out.point3D_ids()):
            out.delete_point3D(pid)
        for p in refined.points3D.values():
            out.add_point3D(p.xyz, pycolmap.Track(), p.color)
    return out


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
    ap.add_argument("--png-batch", type=int, default=256,
                    help="Images pycolmap cannot decode (WebP) are read from lossless PNG copies made "
                         "this many at a time")
    ap.add_argument("--covisible-pairs", type=int, default=40,
                    help="Pair each image with the (up to) this many images that see the same model points "
                         "(0: nearest camera centres only, --neighbors)")
    ap.add_argument("--neighbors", type=int, default=20,
                    help="Matched neighbours per image when the model has no points (or --covisible-pairs 0)")
    ap.add_argument("--max-pair-angle", type=float, default=75.0,
                    help="Only match cameras whose viewing directions differ by less than this")
    ap.add_argument("--max-epipolar-deg", type=float, default=4.0,
                    help="Models with a fisheye camera: match verification tolerance against the input "
                         "poses; must exceed their error (an uncalibrated X5 mount was 2.7 deg off)")
    ap.add_argument("--triangulation-max-error", type=float, default=20.0,
                    help="Keep observations up to this reprojection error (px) when triangulating with "
                         "the input poses; must exceed their error")
    ap.add_argument("--filter-max-errors", default="8,4",
                    help="Bundle adjustment rounds: after each, drop observations above this error (px)")
    ap.add_argument("--no-lidar-scale", dest="lidar_scale", action="store_false",
                    help="Scale the result to the input camera centres even when depth maps are present")
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
    ap.add_argument("--colmap", default="colmap",
                    help="COLMAP 4 command-line tool for feature extraction and matching (GPU, reads WebP); "
                         "'none' to use pycolmap on the CPU")
    ap.add_argument("--gpu", choices=("on", "off"), default="on",
                    help="Use the GPU in the colmap tool (it falls back to the CPU if the GPU run fails)")
    ap.add_argument("--max-centre-rotation-deg", type=float, default=0.2,
                    help="Without LiDAR scale, align with the camera centres only if they fix the rotation "
                         "to this many degrees (else the rotation comes from the camera orientations)")
    ap.add_argument("--no-rig-config", dest="rig_config", action="store_false",
                    help="Do not group the images into rig frames from rig_config.json")
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
    rec_in = rec
    n_obs = int(rec.compute_num_observations())
    report.update(model_dir=model_dir, num_images=len(rec.images), input_observations=n_obs)
    log(f"Model {model_dir}: {len(rec.images)} images, {rec.num_points3D()} points, {n_obs} observations")
    # A model without rig files but with rig_config.json (Rooster archive B): refine
    # rig frames, so the views of one capture move together; written back below in
    # the input's layout.
    if args.rig_config and n_obs < args.min_observations_per_image * max(len(rec.images), 1):
        try:
            rig_rec, rig_info = rig_from_config(pycolmap, rec, args.dataset)
        except Exception as e:
            rig_rec, rig_info = None, f"failed: {e}"
        if rig_rec is not None:
            rec = rig_rec
            report["rig_from_config"] = rig_info
            log(f"Rig from {rig_info['source']}: {len(rig_info['views'])} views "
                f"({', '.join(rig_info['views'])}, reference {rig_info['reference_view']}), "
                f"{rig_info['frames']} rig frames; reproduces the input poses to "
                f"{rig_info['max_pose_error_deg']:.4f} deg / {rig_info['max_centre_error']:.5f}")
        elif rig_info:
            report["rig_from_config"] = {"status": "not used", "reason": rig_info}
            log(f"rig_config.json not used ({rig_info}); refining each image on its own")
    before = poses(rec)
    report["num_posed"] = len(before)
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

    # Step 4: robust bundle adjustment; after each round drop the observations that
    # stay far off (mismatches, moving objects), then adjust again.
    try:
        limits = [float(x) for x in str(args.filter_max_errors).split(",") if x.strip()]
        rounds = []
        for k, max_err in enumerate(limits):
            pycolmap.bundle_adjustment(tri, ba_options(pycolmap, args.refine_intrinsics,
                                                       loss_scale=2.0 if k == 0 else 1.0))
            n_f = filter_observations(pycolmap, tri, max_err, 1.0 if k == 0 else 1.5)
            rounds.append({"max_error_px": max_err, "filtered_observations": n_f,
                           "mean_error_px": float(tri.compute_mean_reprojection_error())})
            log(f"Bundle adjustment round {k + 1}: dropped {n_f} observations above {max_err:g} px; "
                f"mean reprojection error {rounds[-1]['mean_error_px']:.3f} px")
        pycolmap.bundle_adjustment(tri, ba_options(pycolmap, args.refine_intrinsics, loss_scale=1.0))
        report["ba_rounds"] = rounds
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

    # Step 5: put the result back in the input frame (BA leaves position, rotation and
    # scale free). With depth maps the scale comes from the LiDAR, so the depth maps and
    # the re-fused cloud agree with the refined geometry; the input camera centres can
    # imply another scale (3.7% off on an iPhone + X5 session).
    lidar = prepare_lidar(rec, args.dataset, args) if (args.refuse_lidar or args.lidar_scale) \
        else {"maps": [], "reason": "not requested"}
    sim3, scale_info = None, {"source": "camera centres"}
    if args.lidar_scale and lidar.get("maps"):
        try:
            ls = lidar_scale(tri, lidar, args.dataset, args)
        except Exception as e:
            ls = {"scale": None, "samples": 0, "iqr": None, "factors": {}}
            log(f"LiDAR scale failed ({e}); scaling to the camera centres")
        s, n_s, iqr = ls["scale"], ls["samples"], ls["iqr"]
        lidar_ids = sorted({iid for _, iid in lidar["maps"]} & set(ids))
        if s is not None and len(lidar_ids) >= 3:
            sim3 = align_to_input(pycolmap, before, refined, lidar_ids, s)
            lidar["factors"] = ls["factors"]
            fv = np.array(list(ls["factors"].values())) if ls["factors"] else np.ones(1)
            scale_info = {"source": "lidar", "samples": n_s, "lidar_per_unit_p25_p75": iqr,
                          "per_image_factor_p10_p50_p90": [float(x) for x in np.percentile(fv, [10, 50, 90])]}
            log(f"Scale from the LiDAR: {s:.4f} ({n_s} depth samples, p25-p75 {iqr[0]:.4f}-{iqr[1]:.4f}); "
                f"rotation and translation from {len(lidar_ids)} cameras with depth maps; per-image depth "
                f"factors p10/p50/p90 {' / '.join(f'{x:.4f}' for x in np.percentile(fv, [10, 50, 90]))}")
        else:
            log(f"LiDAR scale not used ({n_s} depth samples); scaling to the camera centres")
    centre_fit = pycolmap.estimate_sim3d(np.array([refined[i][1] for i in ids]),
                                         np.array([before[i][1] for i in ids]))
    if centre_fit is not None:
        scale_info["camera_centre_scale"] = float(centre_fit.scale)
    if sim3 is None:
        # No LiDAR scale: a similarity on the camera centres (for a LiDAR rig the
        # map follows the LiDAR positions, so the centres are the reference), unless
        # the path is too short or straight to fix its rotation; the rotation then
        # comes from the camera orientations (see centre_fit_rotation_deg).
        unc = (centre_fit_rotation_deg(centre_fit, before, refined, ids, len(update_frames))
               if centre_fit is not None else 90.0)
        scale_info["centre_fit_rotation_deg"] = unc
        if centre_fit is not None and unc <= args.max_centre_rotation_deg:
            sim3 = centre_fit
        else:
            sim3 = align_to_input(pycolmap, before, refined, ids)
            scale_info["source"] = "camera orientations and centres"
            log(f"The camera centres fix the alignment's rotation only to about {unc:.2f} deg (short or "
                f"nearly straight path); taking the rotation from the camera orientations")
    if sim3 is None:
        finish("kept_original", "could not align the refined model back to the input frame")
        return
    scale_info["scale"] = float(sim3.scale)
    if scale_info["source"] == "lidar" and scale_info.get("camera_centre_scale"):
        # LiDAR depth vs the input camera path (both metric in ARKit): with the LiDAR
        # scale the refined path is this much larger (> 1) or smaller than the input's.
        gap = float(sim3.scale) / scale_info["camera_centre_scale"]
        scale_info["lidar_vs_camera_path"] = gap
        if abs(gap - 1.0) > 0.03:
            log(f"The LiDAR depth and the input camera path disagree on scale by {100 * (gap - 1):+.1f}% "
                f"(LiDAR {sim3.scale:.4f}, camera centres {scale_info['camera_centre_scale']:.4f} per model unit). "
                f"The LiDAR scale is used, so the depth maps agree with the refined geometry and the camera "
                f"path changes size by that much")
    report["alignment"] = scale_info
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
    # 360 shots vs the image of the same capture: a mount error is common to all shots,
    # a timing error differs from shot to shot.
    try:
        rf = rig_frame_report(rec, out)
    except Exception as e:  # diagnostics only
        rf = None
        log(f"Rig frame report failed: {e}")
    if rf:
        report["rig_frames"] = rf
        v = rf["common_rotation_vector_deg_rig_axes"]
        log(f"Rig frames vs the image of the same capture ({rf['frames']} frames): common rotation "
            f"{rf['common_rotation_deg']:.2f} deg (rotation vector {v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f} deg in rig "
            f"axes): the frames' mount on the rig was off by this. Per-frame spread median "
            f"{rf['per_frame_spread_deg']['median']:.2f}, max {rf['per_frame_spread_deg']['max']:.2f} deg "
            f"(capture timing or motion)")

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
        f"alignment scale {sim3.scale:.4f} ({scale_info['source']})")
    by_folder = {}
    for i, r_, s_ in zip(ids, rot, shift):
        name = rec.images[i].name
        by_folder.setdefault(name.split("/")[0] if "/" in name else ".", []).append((r_, s_))
    report["pose_change_by_folder"] = {
        g: {"images": len(v), "rotation_deg_median": float(np.median([a for a, _ in v])),
            "shift_median": float(np.median([b for _, b in v]))} for g, v in sorted(by_folder.items())}
    log("Pose change by folder (median): " + ", ".join(
        f"{g} {d['rotation_deg_median']:.2f} deg / {d['shift_median']:.3f}"
        for g, d in report["pose_change_by_folder"].items()))

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
            report["lidar_refusion"] = refuse_lidar(pycolmap, out, rec, args.dataset, image_dir, args, lidar)
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
    if rec is not rec_in:  # rig frames from rig_config.json: back to the input's layout
        try:
            out = to_input_layout(pycolmap, rec_in, out,
                                  new_points=report.get("lidar_refusion", {}).get("status") == "refused")
        except Exception as e:
            finish("kept_original", f"could not write the rig poses back to the input model: {e}")
            return
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
