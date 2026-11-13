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

"""Remove blurry images from an image set, a pose-prior zip or a COLMAP model.

Sharpness is the variance of the Laplacian on a grayscale copy (long side
1024 px), ignoring masked pixels. Each image is compared with the median of
its neighbours in name order (capture order) within the same folder, since
some scenes are softer than others; an image is dropped when it is below
--ratio of that median. At most --max-drop of each group is dropped.
Multi-view names such as f01713_y000.jpg are grouped per view (the name
without its frame number, f_y000), so a sky-facing view is never compared
with a ground-facing one.

Dropped images are moved to <dataset>/blur_filter/removed/ and removed from
everything that refers to them, so later steps see a consistent dataset:
  - masks/ (all naming variants), depth/, depth_images/, depths/<stem>_depth*.bin
  - COLMAP models (images.bin/txt, points3D tracks, frames) in sparse/ or colmap/sparse/
  - COLMAP databases (*.db): images, keypoints, descriptors, matches,
    two-view geometries, frame data, empty frames and pose priors
  - transforms.json / transforms-in.json frames
A report is written to <dataset>/blur_filter/report.json.
"""

import argparse
import glob
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile

import cv2
import numpy as np

TAG = "[Blur-Filter]"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
SKIP_DIRS = {"blur_filter", "mask_previews", "bundle_adjust", "calibrate_rig", "temp"}
MAX_IMAGE_ID = 2147483647  # COLMAP pair_id = MAX_IMAGE_ID * id1 + id2 with id1 < id2


def log(msg):
    print(f"{TAG} {msg}", flush=True)


# --------------------------------------------------------------------------
# Sharpness
# --------------------------------------------------------------------------

def find_mask(masks_dir, rel):
    if not masks_dir:
        return None
    for name in (rel + ".png", os.path.splitext(rel)[0] + ".png", rel):
        p = os.path.join(masks_dir, name)
        if os.path.isfile(p):
            return p
    return None


def sharpness(path, mask_path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    s = min(1.0, 1024 / max(h, w))
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    keep = None
    if mask_path:
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            keep = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
            keep = cv2.erode(keep.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # skip mask edges
    vals = lap[keep] if keep is not None and keep.sum() > 0.05 * keep.size else lap.ravel()
    return float(vals.var())


MIN_VIEW_GROUP = 8


def view_groups(folder, rels):
    """Split one folder into per-view groups by removing the first digit run
    from the file stem (the frame number): f01713_y000.jpg -> f_y000, so each
    yaw of a multi-view capture is compared only with the same yaw in other
    frames. Single-camera names (frame_00001.jpg) end up in one group. Falls
    back to the whole folder when any group is too small to compare."""
    by_view = {}
    for rel in rels:
        stem = os.path.splitext(os.path.basename(rel))[0]
        by_view.setdefault(re.sub(r"\d+", "", stem, count=1), []).append(rel)
    if len(by_view) <= 1 or min(len(v) for v in by_view.values()) < MIN_VIEW_GROUP:
        return {folder: rels}
    return {f"{folder}/{view}" if folder else view: v for view, v in by_view.items()}


def choose_blurry(scores, window, ratio, max_drop):
    """scores: list of (rel, score) in capture order for one folder."""
    n = len(scores)
    if n < 5:
        return []
    vals = np.array([s for _, s in scores])
    rel_scores = []
    for i in range(n):
        lo, hi = max(0, i - window // 2), min(n, i + window // 2 + 1)
        neigh = np.delete(vals[lo:hi], i - lo)
        med = np.median(neigh) if len(neigh) else vals[i]
        rel_scores.append(vals[i] / med if med > 0 else 1.0)
    rel_scores = np.array(rel_scores)
    cand = [i for i in np.argsort(rel_scores) if rel_scores[i] < ratio]
    cand = cand[: int(max_drop * n)]
    return [(scores[i][0], float(scores[i][1]), float(rel_scores[i])) for i in sorted(cand)]


# --------------------------------------------------------------------------
# COLMAP models (edited as text so 3.x and 4.x models, bin or txt, work alike)
# --------------------------------------------------------------------------

def find_models(dataset):
    out = []
    for root, dirs, files in os.walk(dataset):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and d not in ("images", "masks")]
        for fmt in ("bin", "txt"):
            if f"images.{fmt}" in files and f"cameras.{fmt}" in files:
                out.append((root, fmt))
                break
    return out


def is_image_header(tokens):
    if len(tokens) < 10:
        return False
    try:
        int(tokens[0])
        [float(t) for t in tokens[1:8]]
        int(tokens[8])
    except ValueError:
        return False
    return tokens[-1].lower().endswith(IMAGE_EXTS)


def edit_text_model(d, removed_names):
    """Drop images by name; fix points3D tracks, points2D refs and frames. Returns removed image ids."""
    path = os.path.join(d, "images.txt")
    lines = open(path, encoding="utf-8").read().splitlines()
    comments = [l for l in lines if l.startswith("#")]
    body = [l for l in lines if not l.startswith("#")]
    entries, i = [], 0
    while i < len(body):
        tok = body[i].split()
        if not is_image_header(tok):
            i += 1
            continue
        header, pts = body[i], ""
        if i + 1 < len(body) and not is_image_header(body[i + 1].split()):
            pts = body[i + 1]  # may be empty (no 2D points)
            i += 1
        entries.append((int(tok[0]), " ".join(tok[9:]), header, pts))
        i += 1
    removed_ids = {iid for iid, name, _, _ in entries if name in removed_names}
    if not removed_ids:
        return set()

    # points3D: drop removed observations; drop points that fall below 2 observations
    # (points without any track, e.g. LiDAR points, are kept).
    dead_points = set()
    p3 = os.path.join(d, "points3D.txt")
    if os.path.isfile(p3):
        out = []
        for l in open(p3, encoding="utf-8").read().splitlines():
            if l.startswith("#") or not l.strip():
                out.append(l)
                continue
            t = l.split()
            head, track = t[:8], t[8:]
            pairs = [(track[k], track[k + 1]) for k in range(0, len(track) - 1, 2)]
            kept = [p for p in pairs if int(p[0]) not in removed_ids]
            if pairs and len(kept) < 2 and len(kept) < len(pairs):
                dead_points.add(head[0])
                continue
            out.append(" ".join(head + [x for p in kept for x in p]))
        open(p3, "w", encoding="utf-8").write("\n".join(out) + "\n")

    with open(path, "w", encoding="utf-8") as f:
        for c in comments:
            f.write(c + "\n")
        for iid, name, header, pts in entries:
            if iid in removed_ids:
                continue
            if dead_points and pts:
                t = pts.split()
                for k in range(2, len(t), 3):
                    if t[k] in dead_points:
                        t[k] = "-1"
                pts = " ".join(t)
            f.write(header + "\n" + pts + "\n")

    fr = os.path.join(d, "frames.txt")
    if os.path.isfile(fr):
        out = []
        for l in open(fr, encoding="utf-8").read().splitlines():
            if l.startswith("#") or not l.strip():
                out.append(l)
                continue
            t = l.split()
            head, data = t[:9], t[10:]
            triples = [data[k:k + 3] for k in range(0, len(data), 3)]
            kept = [x for x in triples if not (x[0] == "CAMERA" and int(x[2]) in removed_ids)]
            if not kept:
                continue
            out.append(" ".join(head + [str(len(kept))] + [v for x in kept for v in x]))
        open(fr, "w", encoding="utf-8").write("\n".join(out) + "\n")
    return removed_ids


def edit_model(d, fmt, removed_names):
    if fmt == "txt":
        return edit_text_model(d, removed_names)
    import pycolmap
    present = {f for f in ("rigs.bin", "frames.bin") if os.path.isfile(os.path.join(d, f))}
    tmp = tempfile.mkdtemp()
    try:
        pycolmap.Reconstruction(d).write_text(tmp)
        ids = edit_text_model(tmp, removed_names)
        if ids:
            pycolmap.Reconstruction(tmp).write_binary(d)
            for f in ("rigs.bin", "frames.bin"):
                if f not in present and os.path.isfile(os.path.join(d, f)):
                    os.remove(os.path.join(d, f))
        return ids
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# COLMAP database
# --------------------------------------------------------------------------

def clean_database(db_path, removed_names):
    con = sqlite3.connect(db_path)
    try:
        tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        if "images" not in tables:
            return None
        names = list(removed_names)
        ids = set()
        for k in range(0, len(names), 500):
            chunk = names[k:k + 500]
            q = "select image_id from images where name in (%s)" % ",".join("?" * len(chunk))
            ids.update(r[0] for r in con.execute(q, chunk))
        if not ids:
            return 0
        id_list = sorted(ids)
        marks = ",".join("?" * len(id_list))
        for t in ("keypoints", "descriptors", "images"):
            if t in tables:
                con.execute(f"delete from {t} where image_id in ({marks})", id_list)
        for t in ("matches", "two_view_geometries"):
            if t in tables:
                bad = [pid for (pid,) in con.execute(f"select pair_id from {t}")
                       if (pid // MAX_IMAGE_ID) in ids or (pid % MAX_IMAGE_ID) in ids]
                for k in range(0, len(bad), 500):
                    chunk = bad[k:k + 500]
                    con.execute(f"delete from {t} where pair_id in (%s)" % ",".join("?" * len(chunk)), chunk)
        if "frame_data" in tables:  # sensor_type 0 = CAMERA
            con.execute(f"delete from frame_data where sensor_type = 0 and data_id in ({marks})", id_list)
            if "frames" in tables:
                con.execute("delete from frames where frame_id not in (select frame_id from frame_data)")
        if "pose_priors" in tables:
            cols = {r[1] for r in con.execute("pragma table_info(pose_priors)")}
            if "corr_data_id" in cols:
                con.execute(f"delete from pose_priors where corr_sensor_type = 0 and corr_data_id in ({marks})",
                            id_list)
            elif "image_id" in cols:
                con.execute(f"delete from pose_priors where image_id in ({marks})", id_list)
        con.commit()
        return len(ids)
    finally:
        con.close()


# --------------------------------------------------------------------------
# Other references
# --------------------------------------------------------------------------

def move(src, dst_root, dataset):
    if not os.path.exists(src):
        return False
    dst = os.path.join(dst_root, os.path.relpath(src, dataset))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    return True


def remove_related_files(dataset, rel, removed_root):
    stem = os.path.splitext(rel)[0]
    flat = stem.replace("/", "_")
    base = os.path.basename(stem)
    n = 0
    for mdir in ("masks", "colmap/masks"):
        for name in (rel + ".png", stem + ".png", rel):
            n += move(os.path.join(dataset, mdir, name), removed_root, dataset)
    for ddir in ("depth", "depth_images", "colmap/depth", "colmap/depth_images"):
        for name in (stem + ".png", flat + ".png", stem + ".npy"):
            n += move(os.path.join(dataset, ddir, name), removed_root, dataset)
    for ddir in ("depths", "colmap/depths"):
        for p in glob.glob(os.path.join(dataset, ddir, glob.escape(base) + "_depth*.bin")):
            n += move(p, removed_root, dataset)
    return n


def edit_transforms(path, removed_rels):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    targets = {r.lower() for r in removed_rels} | {os.path.basename(r).lower() for r in removed_rels}

    def is_removed(fp):
        fp = fp.replace("\\", "/").lower()
        if fp.startswith("./"):
            fp = fp[2:]
        if fp.startswith("images/"):
            fp = fp[len("images/"):]
        return fp in targets
    kept = [fr for fr in frames if not is_removed(fr.get("file_path", ""))]
    if len(kept) != len(frames):
        data["frames"] = kept
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    return len(frames) - len(kept)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dataset", required=True, help="Dataset root (DATASET_PATH)")
    ap.add_argument("-i", "--images-dir", default="", help="Images folder (default <dataset>/images)")
    ap.add_argument("--database", action="append", default=[],
                    help="Extra COLMAP database to clean (any *.db under the dataset is cleaned too)")
    ap.add_argument("--ratio", type=float, default=0.5,
                    help="Drop an image whose sharpness is below this fraction of its neighbours' median")
    ap.add_argument("--window", type=int, default=15, help="Neighbourhood size in capture order")
    ap.add_argument("--max-drop", type=float, default=0.15, help="Max fraction dropped per folder")
    ap.add_argument("--dry-run", action="store_true", help="Score and report only")
    args = ap.parse_args()

    dataset = os.path.abspath(args.dataset)
    images_dir = args.images_dir or os.path.join(dataset, "images")
    if not os.path.isdir(images_dir) and os.path.isdir(os.path.join(dataset, "colmap", "images")):
        images_dir = os.path.join(dataset, "colmap", "images")
    out_dir = os.path.join(dataset, "blur_filter")
    removed_root = os.path.join(out_dir, "removed")
    os.makedirs(out_dir, exist_ok=True)
    masks_dir = next((os.path.join(dataset, m) for m in ("masks", "colmap/masks")
                      if os.path.isdir(os.path.join(dataset, m))), None)

    folders = {}
    for root, _, files in os.walk(images_dir):
        for f in sorted(files):
            if f.lower().endswith(IMAGE_EXTS):
                rel = os.path.relpath(os.path.join(root, f), images_dir).replace("\\", "/")
                folders.setdefault(os.path.dirname(rel), []).append(rel)
    groups = {}
    for folder, rels in folders.items():
        for key, members in view_groups(folder, rels).items():
            groups[key] = members
    n_images = sum(len(v) for v in groups.values())
    log(f"Scoring {n_images} images in {len(groups)} group(s) under {images_dir}")

    report = {"images": n_images, "ratio": args.ratio, "window": args.window, "max_drop": args.max_drop,
              "folders": {}, "removed": []}
    removed = []
    for folder, rels in sorted(groups.items()):
        scores = []
        for rel in sorted(rels):
            s = sharpness(os.path.join(images_dir, rel), find_mask(masks_dir, rel))
            if s is not None:
                scores.append((rel, s))
        drop = choose_blurry(scores, args.window, args.ratio, args.max_drop)
        report["folders"][folder or "."] = {"images": len(scores), "removed": len(drop),
                                            "median_sharpness": float(np.median([s for _, s in scores]))
                                            if scores else 0.0}
        removed += drop
        log(f"  {folder or '.'}: {len(drop)}/{len(scores)} blurry")
    report["removed"] = [{"image": r, "sharpness": s, "relative_to_neighbours": q} for r, s, q in removed]

    removed_rels = [r for r, _, _ in removed]
    if args.dry_run or not removed_rels:
        report["status"] = "dry_run" if args.dry_run else "nothing_removed"
        json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=2)
        log(f"{report['status']}: {len(removed_rels)} image(s) would be removed")
        return

    # 1. Images and per-image files.
    related = 0
    for rel in removed_rels:
        move(os.path.join(images_dir, rel), removed_root, dataset)
        related += remove_related_files(dataset, rel, removed_root)

    # 2. COLMAP models (image names are relative to the images folder). Each model
    #    is backed up first and restored if editing fails.
    names = set(removed_rels)
    report["models"] = {}
    for d, fmt in find_models(dataset):
        backup = os.path.join(out_dir, "original_models", os.path.relpath(d, dataset))
        if not os.path.isdir(backup):
            shutil.copytree(d, backup)
        try:
            ids = edit_model(d, fmt, names)
            report["models"][os.path.relpath(d, dataset)] = len(ids)
            if ids:
                log(f"  model {os.path.relpath(d, dataset)}: removed {len(ids)} images")
        except Exception as e:
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(backup, d)
            report["models"][os.path.relpath(d, dataset)] = f"failed: {e}"
            log(f"  WARNING: could not edit model {d} (restored the original): {e}")

    # 3. COLMAP databases.
    dbs = {os.path.abspath(p) for p in args.database if os.path.isfile(p)}
    for root, dirs, files in os.walk(dataset):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and x not in ("images", "masks")]
        dbs.update(os.path.join(root, f) for f in files if f.endswith(".db"))
    report["databases"] = {}
    for db in sorted(dbs):
        db_backup = os.path.join(out_dir, "original_databases", os.path.basename(db))
        if not os.path.isfile(db_backup):
            os.makedirs(os.path.dirname(db_backup), exist_ok=True)
            shutil.copy2(db, db_backup)
        try:
            n = clean_database(db, names)
            if n is not None:
                report["databases"][db] = n
                log(f"  database {db}: removed {n} images and their features/matches")
        except Exception as e:
            report["databases"][db] = f"failed: {e}"
            log(f"  WARNING: could not clean database {db}: {e}")

    # 4. transforms.json files.
    report["transforms"] = {}
    for name in ("transforms.json", "transforms-in.json"):
        p = os.path.join(dataset, name)
        if os.path.isfile(p):
            report["transforms"][name] = edit_transforms(p, removed_rels)

    report["status"] = "removed"
    report["related_files_moved"] = related
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=2)
    log(f"Removed {len(removed_rels)}/{n_images} blurry images ({related} related files); "
        f"originals in {removed_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # never fail the job: continue with the unfiltered dataset
        log(f"WARNING: blur filter failed, dataset left as is: {e}")
    sys.exit(0)
