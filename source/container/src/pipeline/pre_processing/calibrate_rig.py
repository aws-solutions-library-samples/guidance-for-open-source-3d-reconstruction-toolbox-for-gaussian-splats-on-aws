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

"""Re-estimate the camera rig (sensor_from_rig) of a multi-camera COLMAP 4 export
from the session's own images (CALIBRATE_RIG).

Built for mobile exports where a 360 camera (e.g. Insta360 X5 cube faces or fisheye
lenses) is mounted to an iPhone. The iPhone is the rig's reference sensor and its
poses (ARKit) are trusted; the mounting between the two cameras (the rig) comes from
an in-app calibration that is re-done every run and can be off by a few degrees and
centimetres.

Steps:
  1. Find a COLMAP model with rigs.bin + frames.bin and at least one non-reference
     sensor with images. Otherwise exit 0 without changing anything ("skipped").
  2. Pick the non-reference sensors whose masks leave at least --min-unmasked of the
     image visible (heavily masked faces rarely register and only add SfM time).
  3. Run an unconstrained COLMAP SfM on the reference images + the picked sensors,
     with intrinsics fixed to the export's cameras.
  4. Align the SfM to the exported reference poses (Sim3 on camera centres).
  5. Fit ONE rotation and ONE centre offset that correct every non-reference sensor
     (they are rigidly attached to the same 360 camera body).
  6. Quality gate. If it fails, keep the original poses ("flagged") and exit 0.
  7. Otherwise rewrite rigs.bin and the non-reference poses in images.bin. Originals
     are backed up to <dataset>/calibrate_rig/original/.

A JSON report is written to <dataset>/calibrate_rig/report.json in every case.
The COLMAP binaries are read and written directly (no pycolmap dependency).
"""

import argparse
import glob
import json
import os
import shutil
import struct
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

TAG = "[Calibrate-Rig]"

# COLMAP camera model id -> (name, number of params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
SENSOR_CAMERA = 0


def log(msg):
    print(f"{TAG} {msg}", flush=True)


# --------------------------------------------------------------------------
# COLMAP binary I/O
# --------------------------------------------------------------------------

def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id, model = struct.unpack("<Ii", f.read(8))
            w, h = struct.unpack("<QQ", f.read(16))
            name, n_par = CAMERA_MODELS[model]
            params = struct.unpack(f"<{n_par}d", f.read(8 * n_par))
            cams[cam_id] = {"model": name, "width": w, "height": h, "params": list(params)}
    return cams


def read_images(path):
    """Return {image_id: dict(name, camera_id, R (cam_from_world), t, offset)}.
    'offset' is the byte offset of the qvec so poses can be patched in place."""
    imgs = {}
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    n = struct.unpack_from("<Q", data, pos)[0]; pos += 8
    for _ in range(n):
        img_id = struct.unpack_from("<I", data, pos)[0]; pos += 4
        pose_off = pos
        q = struct.unpack_from("<4d", data, pos); pos += 32
        t = struct.unpack_from("<3d", data, pos); pos += 24
        cam_id = struct.unpack_from("<I", data, pos)[0]; pos += 4
        end = data.index(b"\x00", pos)
        name = data[pos:end].decode("utf-8"); pos = end + 1
        n2d = struct.unpack_from("<Q", data, pos)[0]; pos += 8 + 24 * n2d
        imgs[img_id] = {"name": name, "camera_id": cam_id, "R": quat_to_R(q),
                        "t": np.array(t), "offset": pose_off}
    return imgs


def write_image_poses(path, updates):
    """Patch qvec/tvec of images in place. updates = {offset: (R, t)}."""
    with open(path, "rb") as f:
        data = bytearray(f.read())
    for off, (R, t) in updates.items():
        struct.pack_into("<4d", data, off, *R_to_quat(R))
        struct.pack_into("<3d", data, off + 32, *[float(v) for v in t])
    with open(path, "wb") as f:
        f.write(bytes(data))


def read_rigs(path):
    rigs = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            rig_id, n_sensors = struct.unpack("<II", f.read(8))
            ref = None
            sensors = []
            if n_sensors > 0:
                ref = struct.unpack("<iI", f.read(8))
                for _ in range(n_sensors - 1):
                    s_type, s_id = struct.unpack("<iI", f.read(8))
                    has_pose = struct.unpack("<B", f.read(1))[0]
                    R = t = None
                    if has_pose:
                        q = struct.unpack("<4d", f.read(32))
                        t = np.array(struct.unpack("<3d", f.read(24)))
                        R = quat_to_R(q)
                    sensors.append({"type": s_type, "id": s_id, "R": R, "t": t})
            rigs.append({"rig_id": rig_id, "ref": ref, "sensors": sensors})
    return rigs


def write_rigs(path, rigs):
    out = bytearray(struct.pack("<Q", len(rigs)))
    for rig in rigs:
        n_sensors = (1 if rig["ref"] is not None else 0) + len(rig["sensors"])
        out += struct.pack("<II", rig["rig_id"], n_sensors)
        if rig["ref"] is not None:
            out += struct.pack("<iI", *rig["ref"])
        for s in rig["sensors"]:
            out += struct.pack("<iI", s["type"], s["id"])
            if s["R"] is None:
                out += struct.pack("<B", 0)
            else:
                out += struct.pack("<B", 1)
                out += struct.pack("<4d", *R_to_quat(s["R"]))
                out += struct.pack("<3d", *[float(v) for v in s["t"]])
    with open(path, "wb") as f:
        f.write(bytes(out))


def read_frames(path):
    frames = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            frame_id, rig_id = struct.unpack("<II", f.read(8))
            q = struct.unpack("<4d", f.read(32))
            t = np.array(struct.unpack("<3d", f.read(24)))
            n_data = struct.unpack("<I", f.read(4))[0]
            data = [struct.unpack("<iIQ", f.read(16)) for _ in range(n_data)]
            frames.append({"frame_id": frame_id, "rig_id": rig_id,
                           "R": quat_to_R(q), "t": t, "data": data})
    return frames


def quat_to_R(q_wxyz):
    w, x, y, z = q_wxyz
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def R_to_quat(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    return [float(w), float(x), float(y), float(z)]


def centre(R, t):
    return -R.T @ t


# --------------------------------------------------------------------------
# Dataset discovery
# --------------------------------------------------------------------------

def find_rig_models(dataset):
    out = []
    for rel in ("colmap/sparse/0", "sparse/0"):
        d = os.path.join(dataset, rel)
        if all(os.path.isfile(os.path.join(d, f))
               for f in ("cameras.bin", "images.bin", "rigs.bin", "frames.bin")):
            out.append(d)
    return out


def find_dir(dataset, *candidates):
    for rel in candidates:
        d = os.path.join(dataset, rel)
        if os.path.isdir(d):
            return d
    return None


def mask_file(mask_dir, image_name):
    if mask_dir is None:
        return None
    stem = os.path.splitext(image_name)[0]
    for cand in (image_name + ".png", stem + ".png", image_name):
        p = os.path.join(mask_dir, cand)
        if os.path.isfile(p):
            return p
    return None


def unmasked_fraction(path):
    import cv2
    m = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if m is None:
        return 1.0
    return float((m > 127).mean())


# --------------------------------------------------------------------------
# COLMAP CLI (works with COLMAP 3.9 and 4.x option names)
# --------------------------------------------------------------------------

def _db_files(db):
    return [db + s for s in ("", "-wal", "-shm")]


def run_colmap(colmap_bin, args, fallbacks=(), db=None):
    """Run colmap; on failure retry with each fallback arg list (option names differ
    between COLMAP 3.x and 4.x, and GPU steps can fail where CPU works).
    If db is given, the database is restored to its pre-attempt state before each
    retry so a crashed attempt cannot leave half-written rows behind."""
    last = ""
    snap = None
    if db:
        snap = db + ".snapshot"
        for f, s in zip(_db_files(db), _db_files(snap)):
            if os.path.exists(f):
                shutil.copy2(f, s)
            elif os.path.exists(s):
                os.remove(s)
    for n, attempt in enumerate([list(args)] + [list(a) for a in fallbacks]):
        if n and snap:
            for f, s in zip(_db_files(db), _db_files(snap)):
                if os.path.exists(f):
                    os.remove(f)
                if os.path.exists(s):
                    shutil.copy2(s, f)
        r = subprocess.run([colmap_bin] + attempt, capture_output=True, text=True)
        if r.returncode == 0:
            if n:
                log(f"colmap {args[0]} succeeded on fallback attempt {n} (last options: {' '.join(attempt[-2:])})")
            return
        last = (r.stderr or r.stdout or "")[-1500:]
    raise RuntimeError(f"colmap {args[0]} failed: {last}")


def run_sfm(colmap_bin, work, image_dir, groups, mask_root, use_gpu, max_features):
    """groups: list of (camera dict, [image names]). Returns {name: (R, C)}."""
    db = os.path.join(work, "database.db")
    sparse = os.path.join(work, "sparse")
    os.makedirs(sparse, exist_ok=True)
    gpu = "1" if use_gpu else "0"
    for i, (cam, names) in enumerate(groups):
        lst = os.path.join(work, f"list_{i}.txt")
        with open(lst, "w") as f:
            f.write("\n".join(names))
        base = ["feature_extractor", "--database_path", db, "--image_path", image_dir,
                "--image_list_path", lst,
                "--ImageReader.camera_model", cam["model"],
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_params", ",".join(f"{p:.6f}" for p in cam["params"]),
                "--SiftExtraction.max_num_features", str(max_features)]
        if mask_root:
            base += ["--ImageReader.mask_path", mask_root]
        run_colmap(colmap_bin, base + ["--FeatureExtraction.use_gpu", gpu],
                   fallbacks=[base + ["--SiftExtraction.use_gpu", gpu],
                              base + ["--FeatureExtraction.use_gpu", "0"],
                              base + ["--SiftExtraction.use_gpu", "0"]], db=db)
    base = ["exhaustive_matcher", "--database_path", db]
    # GPU matching can fail on large sessions (e.g. "max_num_matches > 0" when the
    # GPU matcher cannot reserve memory), so fall back to CPU matching.
    run_colmap(colmap_bin, base + ["--FeatureMatching.use_gpu", gpu],
               fallbacks=[base + ["--SiftMatching.use_gpu", gpu],
                          base + ["--FeatureMatching.use_gpu", "0"],
                          base + ["--SiftMatching.use_gpu", "0"]], db=db)
    base = ["mapper", "--database_path", db, "--image_path", image_dir, "--output_path", sparse]
    fixed = ["--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
             "--Mapper.ba_refine_extra_params", "0"]
    run_colmap(colmap_bin, base + fixed, fallbacks=[base])
    best = None
    for m in sorted(glob.glob(os.path.join(sparse, "*"))):
        p = os.path.join(m, "images.bin")
        if os.path.isfile(p):
            imgs = read_images(p)
            if best is None or len(imgs) > len(best):
                best = imgs
    if not best:
        return {}
    return {v["name"]: (v["R"], centre(v["R"], v["t"])) for v in best.values()}


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------

def umeyama(src, dst):
    """Similarity dst ≈ s * R @ src + t."""
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    E[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ E @ Vt
    s = np.trace(np.diag(sig) @ E) / (S ** 2).sum(1).mean()
    return s, R, md - s * R @ ms


def robust_rotation_mean(rots, n_iter=3):
    keep = np.ones(len(rots), bool)
    for _ in range(n_iter):
        m = Rotation.from_matrix(np.array(rots)[keep]).mean()
        err = np.array([np.degrees((m.inv() * Rotation.from_matrix(r)).magnitude()) for r in rots])
        thr = max(1.0, 3.0 * np.median(err[keep]))
        new = err <= thr
        if new.sum() < 3 or (new == keep).all():
            break
        keep = new
    return Rotation.from_matrix(np.array(rots)[keep]).mean().as_matrix(), keep


def angle_deg(R):
    return float(np.degrees(Rotation.from_matrix(R).magnitude()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dataset", required=True, help="Dataset root (DATASET_PATH)")
    ap.add_argument("--colmap", default="colmap", help="COLMAP executable")
    ap.add_argument("--min-unmasked", type=float, default=0.30,
                    help="Use non-reference sensors whose images are at least this fraction unmasked")
    ap.add_argument("--min-faces", type=int, default=10, help="Gate: minimum registered non-reference images")
    ap.add_argument("--max-residual-deg", type=float, default=1.5,
                    help="Gate: maximum median rotation error after the correction")
    ap.add_argument("--max-align-cm", type=float, default=5.0,
                    help="Gate: maximum median reference-camera centre error of the SfM alignment")
    ap.add_argument("--max-correction-deg", type=float, default=10.0)
    ap.add_argument("--max-correction-cm", type=float, default=25.0)
    ap.add_argument("--max-features", type=int, default=8000)
    ap.add_argument("--gpu", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--keep-work", action="store_true", help="Keep the SfM working directory")
    args = ap.parse_args()

    dataset = os.path.abspath(args.dataset)
    out_dir = os.path.join(dataset, "calibrate_rig")
    os.makedirs(out_dir, exist_ok=True)
    report = {"status": None, "reason": None, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def finish(status, reason=None, code=0):
        report["status"], report["reason"] = status, reason
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        if status == "applied":
            log(f"applied: {reason}")
        elif status == "skipped":
            log(f"skipped (no changes): {reason}")
        else:
            log(f"{status.upper()} - keeping the original poses: {reason}")
        sys.exit(code)

    try:
        models = find_rig_models(dataset)
        if not models:
            finish("skipped", "no COLMAP model with rigs.bin + frames.bin")
        model = models[0]
        report["model"] = os.path.relpath(model, dataset)
        cams = read_cameras(os.path.join(model, "cameras.bin"))
        imgs = read_images(os.path.join(model, "images.bin"))
        rigs = read_rigs(os.path.join(model, "rigs.bin"))
        frames = read_frames(os.path.join(model, "frames.bin"))
        if len(rigs) != 1 or rigs[0]["ref"] is None:
            finish("skipped", f"expected one rig with a reference sensor, found {len(rigs)}")
        rig = rigs[0]
        ref_cam = rig["ref"][1]
        sensors = {s["id"]: s for s in rig["sensors"]
                   if s["type"] == SENSOR_CAMERA and s["R"] is not None}
        if not sensors:
            finish("skipped", "rig has no non-reference camera sensors with a pose")

        # image id -> frame
        frame_of = {}
        for fr in frames:
            for s_type, s_id, d_id in fr["data"]:
                if s_type == SENSOR_CAMERA:
                    frame_of[d_id] = fr
        by_cam = {}
        for iid, im in imgs.items():
            by_cam.setdefault(im["camera_id"], []).append(iid)
        if not any(by_cam.get(c) for c in sensors):
            finish("skipped", "no images for the non-reference sensors")
        if len(by_cam.get(ref_cam, [])) < 5:
            finish("skipped", "fewer than 5 reference-sensor images")

        image_dir = find_dir(dataset, "images", "colmap/images")
        mask_dir = find_dir(dataset, "masks", "colmap/masks")
        if image_dir is None:
            finish("skipped", "no images/ directory")
        missing = [imgs[i]["name"] for i in imgs if not os.path.isfile(os.path.join(image_dir, imgs[i]["name"]))]
        if missing:
            finish("skipped", f"{len(missing)} images in the model are missing on disk, e.g. {missing[0]}")

        # ---- pick sensors by mask coverage
        coverage = {}
        for c in sensors:
            fr_ = [unmasked_fraction(mask_file(mask_dir, imgs[i]["name"]))
                   if mask_file(mask_dir, imgs[i]["name"]) else 1.0 for i in by_cam.get(c, [])]
            if fr_:
                coverage[c] = float(np.mean(fr_))
        picked = [c for c, v in coverage.items() if v >= args.min_unmasked]
        report["sensor_unmasked_fraction"] = {
            os.path.dirname(imgs[by_cam[c][0]]["name"]) or str(c): round(v, 3) for c, v in coverage.items()}
        report["sensors_used"] = [os.path.dirname(imgs[by_cam[c][0]]["name"]) or str(c) for c in picked]
        log(f"model {report['model']}: {len(imgs)} images, reference camera {ref_cam}, "
            f"{len(sensors)} other sensors; using {report['sensors_used']} (>= {args.min_unmasked:.0%} unmasked)")
        if not picked:
            finish("flagged", "no non-reference sensor is sufficiently unmasked")

        # ---- SfM
        work = os.path.join(out_dir, "work")
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        mask_root = None
        if mask_dir:
            import cv2
            mask_root = os.path.join(work, "masks")
            for c in picked:
                for i in by_cam[c]:
                    mf = mask_file(mask_dir, imgs[i]["name"])
                    if mf:
                        m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
                        dst = os.path.join(mask_root, imgs[i]["name"] + ".png")
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        cv2.imwrite(dst, ((m > 127) * 255).astype(np.uint8))
        groups = [(cams[ref_cam], sorted(imgs[i]["name"] for i in by_cam[ref_cam]))]
        groups += [(cams[c], sorted(imgs[i]["name"] for i in by_cam[c])) for c in picked]
        use_gpu = args.gpu == "on"
        if args.gpu == "auto":
            use_gpu = shutil.which("nvidia-smi") is not None
        t0 = time.time()
        sfm = run_sfm(args.colmap, work, image_dir, groups, mask_root, use_gpu, args.max_features)
        report["sfm_seconds"] = round(time.time() - t0, 1)
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)

        # ---- align SfM to the exported reference poses
        name_to_id = {im["name"]: iid for iid, im in imgs.items()}
        ref_ids = [i for i in by_cam[ref_cam] if imgs[i]["name"] in sfm]
        report["reference_registered"] = f"{len(ref_ids)}/{len(by_cam[ref_cam])}"
        if len(ref_ids) < max(5, len(by_cam[ref_cam]) // 2):
            finish("flagged", f"only {report['reference_registered']} reference images registered in SfM")
        A = np.array([sfm[imgs[i]["name"]][1] for i in ref_ids])
        B = np.array([centre(imgs[i]["R"], imgs[i]["t"]) for i in ref_ids])
        s, Rw, tw = umeyama(A, B)
        align_err = np.linalg.norm((s * (Rw @ A.T)).T + tw - B, axis=1) * 100
        align_rot = [angle_deg(sfm[imgs[i]["name"]][0] @ Rw.T @ imgs[i]["R"].T) for i in ref_ids]
        report["reference_alignment"] = {"centre_cm_median": round(float(np.median(align_err)), 2),
                                         "centre_cm_max": round(float(align_err.max()), 2),
                                         "rotation_deg_median": round(float(np.median(align_rot)), 3)}

        # ---- per non-reference image: measured correction
        Qs, ds, used = [], [], []
        for c in picked:
            S = sensors[c]
            c_old = centre(S["R"], S["t"])
            for i in by_cam[c]:
                nm = imgs[i]["name"]
                if nm not in sfm or i not in frame_of:
                    continue
                fr = frame_of[i]
                R_true = sfm[nm][0] @ Rw.T
                C_true = s * Rw @ sfm[nm][1] + tw
                Qs.append(S["R"].T @ R_true @ fr["R"].T)
                ds.append((fr["R"] @ C_true + fr["t"]) - c_old)
                used.append(i)
        report["registered_non_reference"] = len(used)
        if len(used) < args.min_faces:
            finish("flagged", f"only {len(used)} non-reference images registered (need {args.min_faces})")

        before = np.array([angle_deg(q) for q in Qs])
        Q, keep = robust_rotation_mean(Qs)
        d = np.median(np.array(ds)[keep], axis=0)
        after = np.array([angle_deg(Q.T @ q) for q in Qs])
        pos_before = np.array([np.linalg.norm(x) for x in ds]) * 100
        pos_after = np.array([np.linalg.norm(x - d) for x in ds]) * 100
        rv = Rotation.from_matrix(Q).as_rotvec()
        report["correction"] = {
            "rotation_deg": round(angle_deg(Q), 3),
            "rotvec_deg_rig_frame": [round(float(v), 3) for v in np.degrees(rv)],
            "quaternion_wxyz": [round(v, 6) for v in R_to_quat(Q)],
            "centre_shift_cm_rig_frame": [round(float(v) * 100, 2) for v in d],
            "outliers_rejected": int((~keep).sum()),
        }
        report["error_before"] = {"rotation_deg_median": round(float(np.median(before)), 3),
                                  "rotation_deg_max": round(float(before.max()), 3),
                                  "centre_cm_median": round(float(np.median(pos_before)), 2)}
        report["error_after"] = {"rotation_deg_median": round(float(np.median(after)), 3),
                                 "rotation_deg_max": round(float(after.max()), 3),
                                 "centre_cm_median": round(float(np.median(pos_after)), 2)}
        log(f"{len(used)} non-reference images: rotation error {np.median(before):.2f} -> "
            f"{np.median(after):.2f} deg (median); centre {np.median(pos_before):.1f} -> "
            f"{np.median(pos_after):.1f} cm; correction {angle_deg(Q):.2f} deg, "
            f"{np.linalg.norm(d) * 100:.1f} cm")

        # ---- gate
        reasons = []
        if np.median(align_err) > args.max_align_cm:
            reasons.append(f"reference alignment {np.median(align_err):.1f} cm > {args.max_align_cm} cm")
        if np.median(after) > args.max_residual_deg:
            reasons.append(f"rotation error after correction {np.median(after):.2f} deg > {args.max_residual_deg} deg")
        if angle_deg(Q) > args.max_correction_deg:
            reasons.append(f"correction {angle_deg(Q):.1f} deg > {args.max_correction_deg} deg")
        if np.linalg.norm(d) * 100 > args.max_correction_cm:
            reasons.append(f"centre shift {np.linalg.norm(d) * 100:.1f} cm > {args.max_correction_cm} cm")
        if np.median(after) >= np.median(before):
            reasons.append("correction does not reduce the error")
        if reasons:
            finish("flagged", "; ".join(reasons))

        # ---- apply to every model directory holding the same rig
        backup_root = os.path.join(out_dir, "original")
        written = []
        for mdir in models:
            m_imgs = read_images(os.path.join(mdir, "images.bin"))
            m_rigs = read_rigs(os.path.join(mdir, "rigs.bin"))
            m_frames = read_frames(os.path.join(mdir, "frames.bin"))
            m_sens = {x["id"]: x for x in m_rigs[0]["sensors"] if x["R"] is not None}
            same = len(m_rigs) == 1 and all(
                k in m_sens and np.allclose(m_sens[k]["R"], v["R"], atol=1e-6)
                and np.allclose(m_sens[k]["t"], v["t"], atol=1e-6) for k, v in sensors.items())
            if not same:
                log(f"WARNING: {os.path.relpath(mdir, dataset)} has a different rig; not modified")
                continue
            bdir = os.path.join(backup_root, os.path.relpath(mdir, dataset))
            os.makedirs(bdir, exist_ok=True)
            for fn in ("images.bin", "rigs.bin"):
                if not os.path.exists(os.path.join(bdir, fn)):
                    shutil.copy2(os.path.join(mdir, fn), os.path.join(bdir, fn))
            for sx in m_rigs[0]["sensors"]:
                if sx["R"] is None or sx["type"] != SENSOR_CAMERA:
                    continue
                c_new = centre(sx["R"], sx["t"]) + d
                sx["R"] = sx["R"] @ Q
                sx["t"] = -sx["R"] @ c_new
            new_sens = {x["id"]: x for x in m_rigs[0]["sensors"] if x["R"] is not None}
            fr_of = {}
            for fr in m_frames:
                for s_type, s_id, d_id in fr["data"]:
                    if s_type == SENSOR_CAMERA:
                        fr_of[d_id] = fr
            updates = {}
            for iid, im in m_imgs.items():
                sx = new_sens.get(im["camera_id"])
                fr = fr_of.get(iid)
                if sx is None or fr is None:
                    continue
                updates[im["offset"]] = (sx["R"] @ fr["R"], sx["R"] @ fr["t"] + sx["t"])
            write_rigs(os.path.join(mdir, "rigs.bin"), m_rigs)
            write_image_poses(os.path.join(mdir, "images.bin"), updates)
            written.append({"model": os.path.relpath(mdir, dataset), "images_updated": len(updates)})
        report["written"] = written
        if not written:
            finish("flagged", "no model directory could be updated")
        finish("applied", f"rotation {angle_deg(Q):.2f} deg, centre shift {np.linalg.norm(d) * 100:.1f} cm; "
                          f"error {np.median(before):.2f} -> {np.median(after):.2f} deg median")
    except SystemExit:
        raise
    except Exception as e:  # never break the job; keep the original poses
        import traceback
        report["traceback"] = traceback.format_exc()
        # Restore anything already rewritten so the dataset is left untouched.
        backup_root = os.path.join(out_dir, "original")
        for bfile in glob.glob(os.path.join(backup_root, "**", "*.bin"), recursive=True):
            target = os.path.join(dataset, os.path.relpath(bfile, backup_root))
            try:
                shutil.copy2(bfile, target)
            except OSError:
                pass
        finish("error", f"{type(e).__name__}: {e}", code=1)


if __name__ == "__main__":
    main()
