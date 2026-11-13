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

"""GPS position priors for COLMAP's pose_prior_mapper.

COLMAP's feature_extractor reads GPS from EXIF itself, but the pipeline
rewrites images with OpenCV (resize, autoscale, rotation), which drops EXIF.
So the GPS is read once, right after the images are extracted, and written
into the COLMAP database after feature extraction for images that lost it.

Commands:
  probe  --input <zip|dir>        Sample images (and priors.csv) for GPS; prints
                                  one line: GPS_PROBE {"sampled":..,"with_gps":..}
  read   --images-dir D --dataset-dir R --output gps_priors.json
                                  EXIF GPS, then <R>/priors.csv (image,lat,lon,alt_m)
                                  for images without EXIF GPS. Also keeps the
                                  camera and suggested GPS std from <R>/manifest.json.
  write  --database DB --priors gps_priors.json
                                  Writes WGS84 pose priors (lat, lon, alt) with a
                                  diagonal covariance for images that have none.
"""

import argparse
import csv
import io
import json
import math
import os
import sys
import zipfile

TAG = "[GPS-Priors]"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
GPS_IFD = 0x8825
DEFAULT_STD_M = {"x": 5.0, "y": 5.0, "z": 10.0}


def log(msg):
    print(f"{TAG} {msg}", flush=True)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def _to_deg(dms):
    try:
        d, m, s = (float(v) for v in dms)
    except TypeError:
        return float(dms)
    return d + m / 60.0 + s / 3600.0


def gps_from_exif(fp):
    """(lat, lon, alt) from an image path or file object, or None."""
    from PIL import Image
    try:
        with Image.open(fp) as img:
            gps = img.getexif().get_ifd(GPS_IFD)
    except Exception:
        return None
    if not gps or 2 not in gps or 4 not in gps:
        return None
    try:
        lat = _to_deg(gps[2])
        lon = _to_deg(gps[4])
        if str(gps.get(1, "N")).upper().startswith("S"):
            lat = -lat
        if str(gps.get(3, "E")).upper().startswith("W"):
            lon = -lon
        alt = float(gps[6]) if 6 in gps else 0.0
        ref = gps.get(5, 0)
        if isinstance(ref, bytes):
            ref = ref[0] if ref else 0
        if int(ref) == 1:
            alt = -alt
    except Exception:
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)) or (lat == 0.0 and lon == 0.0):
        return None
    if abs(lat) > 90 or abs(lon) > 180:
        return None
    return lat, lon, alt


def read_priors_csv(path):
    """{image name: (lat, lon, alt)} from a CSV with image/lat/lon[/alt] columns."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {k.strip().lower(): v for k, v in row.items() if k}
            name = row.get("image") or row.get("name") or row.get("filename")
            lat = row.get("lat") or row.get("latitude")
            lon = row.get("lon") or row.get("lng") or row.get("longitude")
            alt = row.get("alt_m") or row.get("alt") or row.get("altitude") or "0"
            if not name or lat in (None, "") or lon in (None, ""):
                continue
            try:
                out[name.replace("\\", "/")] = (float(lat), float(lon), float(alt))
            except ValueError:
                continue
    return out


def read_manifest(path):
    """Camera intrinsics and suggested GPS std from a session manifest.json (optional)."""
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
    except Exception as e:
        log(f"WARNING: could not read {path}: {e}")
        return None, None
    camera = None
    cam = m.get("camera") if isinstance(m, dict) else None
    if isinstance(cam, dict) and cam.get("model") and cam.get("width") and cam.get("fx") is not None:
        camera = {k: cam[k] for k in ("model", "width", "height", "fx", "fy", "cx", "cy") if k in cam}
    std = None
    gps = m.get("gps") if isinstance(m, dict) else None
    if isinstance(gps, dict) and isinstance(gps.get("suggested_prior_std_m"), dict):
        s = gps["suggested_prior_std_m"]
        try:
            std = {k: float(s[k]) for k in ("x", "y", "z")}
        except (KeyError, TypeError, ValueError):
            std = None
    return camera, std


def extent_m(positions):
    """Approximate horizontal/vertical extent in metres (equirectangular)."""
    lats = [p[0] for p in positions]
    lons = [p[1] for p in positions]
    alts = [p[2] for p in positions]
    lat0 = math.radians(sum(lats) / len(lats))
    r = 6378137.0
    dx = math.radians(max(lons) - min(lons)) * r * math.cos(lat0)
    dy = math.radians(max(lats) - min(lats)) * r
    return round(dx, 1), round(dy, 1), round(max(alts) - min(alts), 1)


def cmd_read(args):
    images_dir = args.images_dir
    rels = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                rels.append(os.path.relpath(os.path.join(root, f), images_dir).replace("\\", "/"))
    rels.sort()
    csv_priors = read_priors_csv(os.path.join(args.dataset_dir, "priors.csv"))
    csv_by_base = {os.path.basename(k): v for k, v in csv_priors.items()}
    priors, n_exif, n_csv = {}, 0, 0
    for rel in rels:
        p = gps_from_exif(os.path.join(images_dir, rel))
        if p is not None:
            n_exif += 1
        else:
            p = csv_priors.get(rel)
            if p is None and "/" in rel:
                # Rig layout <camera>/<frame>.jpg listed as <frame>_<camera>.jpg
                folder, base = rel.rsplit("/", 1)
                stem, ext = os.path.splitext(base)
                p = csv_by_base.get(f"{stem}_{folder.replace('/', '_')}{ext}")
            if p is None:
                p = csv_by_base.get(os.path.basename(rel))
            if p is not None:
                n_csv += 1
        if p is not None:
            priors[rel] = list(p)
    camera, std = read_manifest(os.path.join(args.dataset_dir, "manifest.json"))
    coverage = len(priors) / len(rels) if rels else 0.0
    out = {
        "num_images": len(rels),
        "num_with_gps": len(priors),
        "from_exif": n_exif,
        "from_priors_csv": n_csv,
        "coverage": round(coverage, 4),
        "std_m": std,
        "camera": camera,
        "priors": priors,
    }
    if priors:
        out["extent_m"] = dict(zip(("east", "north", "up"), extent_m(list(priors.values()))))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log(f"{len(priors)}/{len(rels)} images have GPS ({n_exif} EXIF, {n_csv} priors.csv); "
        f"coverage {coverage:.0%}" + (f"; extent {out['extent_m']}" if priors else ""))
    if camera:
        log(f"manifest camera: {camera}")
    if std:
        log(f"manifest GPS std (m): {std}")
    log(f"wrote {args.output}")


def cmd_probe(args):
    """Quick build-time check: does the upload carry GPS?"""
    sampled, with_gps, has_csv = 0, 0, False
    src = args.input
    try:
        if os.path.isfile(src) and src.lower().endswith(".zip"):
            with zipfile.ZipFile(src) as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                norm = [n.replace("\\", "/") for n in names]
                has_csv = any(os.path.basename(n).lower() == "priors.csv" for n in norm)
                imgs = [n for n, nn in zip(names, norm)
                        if nn.lower().endswith(IMAGE_EXTS) and
                        not any(p.lower() in ("mask", "masks") for p in nn.split("/")[:-1]) and
                        "__macosx" not in nn.lower()]
                for n in _spread(sorted(imgs), args.samples):
                    sampled += 1
                    with zf.open(n) as fh:
                        if gps_from_exif(io.BytesIO(fh.read())) is not None:
                            with_gps += 1
        elif os.path.isdir(src):
            imgs = []
            for root, dirs, files in os.walk(src):
                dirs[:] = [d for d in dirs if d.lower() not in ("mask", "masks")]
                imgs += [os.path.join(root, f) for f in files if f.lower().endswith(IMAGE_EXTS)]
                has_csv = has_csv or "priors.csv" in (f.lower() for f in files)
            for p in _spread(sorted(imgs), args.samples):
                sampled += 1
                if gps_from_exif(p) is not None:
                    with_gps += 1
        elif os.path.isfile(src) and src.lower().endswith(IMAGE_EXTS):
            sampled = 1
            with_gps = int(gps_from_exif(src) is not None)
    except Exception as e:
        log(f"WARNING: probe failed: {e}")
    print("GPS_PROBE " + json.dumps({"sampled": sampled, "with_gps": with_gps,
                                     "has_priors_csv": has_csv}), flush=True)


def _spread(items, k):
    if len(items) <= k:
        return items
    step = len(items) / float(k)
    return [items[int(i * step)] for i in range(k)]


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def cmd_write(args):
    import numpy as np
    import pycolmap

    with open(args.priors, encoding="utf-8") as f:
        data = json.load(f)
    priors = data.get("priors", {})
    by_name = {k: v for k, v in priors.items()}
    by_base = {os.path.basename(k): v for k, v in priors.items()}
    by_stem = {os.path.splitext(os.path.basename(k))[0]: v for k, v in priors.items()}
    std = data.get("std_m") or DEFAULT_STD_M
    if args.std:
        std = {"x": args.std[0], "y": args.std[1], "z": args.std[2]}
    cov = np.diag([std["x"] ** 2, std["y"] ** 2, std["z"] ** 2]).astype(np.float64)

    written = existing = missing = 0
    with pycolmap.Database.open(args.database) as db:
        have = set()
        for pp in db.read_all_pose_priors():
            try:
                if pp.corr_data_id.sensor_id.type == pycolmap.SensorType.CAMERA and pp.has_position():
                    have.add(int(pp.corr_data_id.id))
            except Exception:
                continue
        for img in db.read_all_images():
            if int(img.image_id) in have:
                existing += 1
                continue
            name = img.name.replace("\\", "/")
            p = by_name.get(name) or by_base.get(os.path.basename(name)) or \
                by_stem.get(os.path.splitext(os.path.basename(name))[0])
            if p is None:
                missing += 1
                continue
            prior = pycolmap.PosePrior(
                position=np.array(p[:3], dtype=np.float64),
                position_covariance=cov,
                coordinate_system=pycolmap.PosePriorCoordinateSystem.WGS84,
            )
            prior.corr_data_id = img.data_id
            db.write_pose_prior(prior)
            written += 1
        total = db.num_pose_priors()
    log(f"pose priors: wrote {written}, already present {existing} (EXIF read by COLMAP), "
        f"no GPS {missing}; database now has {total}; std (m) {std}")
    print("GPS_PRIORS_IN_DB " + json.dumps({"written": written, "existing": existing,
                                            "missing": missing, "total": total}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--input", required=True)
    p.add_argument("--samples", type=int, default=24)
    r = sub.add_parser("read")
    r.add_argument("--images-dir", required=True)
    r.add_argument("--dataset-dir", required=True)
    r.add_argument("--output", required=True)
    w = sub.add_parser("write")
    w.add_argument("--database", required=True)
    w.add_argument("--priors", required=True)
    w.add_argument("--std", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="Override GPS std in metres (default: manifest value, else 5 5 10)")
    args = ap.parse_args()
    {"probe": cmd_probe, "read": cmd_read, "write": cmd_write}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
