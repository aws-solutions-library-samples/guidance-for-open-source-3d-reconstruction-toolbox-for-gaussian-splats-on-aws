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

"""Dataset normalization — auto-detect and convert vendor-specific dataset layouts
into the standard pipeline format before zip extraction proceeds.

Supported patterns:
  - SplatKing / ARKit LiDAR: COLMAP_Text_Model/ wrapper + sensor_data/ with
    float32 .bin depth maps and confidence maps. Flattens layout, converts
    depth to uint16 PNG in depths/, sets PRESERVE_SCENE_SCALE.
  - Generic COLMAP wrapper: any single subdirectory wrapping sparse/ + images/
    with a non-standard name (already handled by caller, but detected here for
    logging clarity).

Returns a NormalizationResult describing what was found and what config
overrides the pipeline should apply.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class NormalizationResult:
    """Describes what the normalizer found and what the pipeline should do."""
    # Whether the layout was modified
    modified: bool = False
    # Human-readable description of what was detected
    detected_pattern: Optional[str] = None
    # Config overrides to apply (key -> value strings, same as config dict)
    config_overrides: dict = field(default_factory=dict)
    # Log messages to emit
    messages: list = field(default_factory=list)


def _log(result: NormalizationResult, msg: str) -> None:
    result.messages.append(msg)
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Pattern: SplatKing / ARKit LiDAR
# ---------------------------------------------------------------------------

def _is_splatking(path: str) -> bool:
    """Detect SplatKing export: COLMAP_Text_Model/ dir + sensor_data/ dir."""
    return (
        os.path.isdir(os.path.join(path, "COLMAP_Text_Model")) and
        os.path.isdir(os.path.join(path, "sensor_data"))
    )


def _convert_arkit_depth_bin(
    bin_path: str,
    conf_path: str,
    meta: dict,
    out_png_path: str,
    target_w: int,
    target_h: int,
    min_confidence: int = 1,
) -> bool:
    """Convert a single ARKit float32 depth .bin + confidence .bin to uint16 PNG.

    Args:
        bin_path: path to *_depth.bin (float32, row-major)
        conf_path: path to *_confidence.bin (uint8, 0=low/1=med/2=high)
        meta: parsed JSON metadata dict for this frame
        out_png_path: output uint16 PNG path
        target_w/target_h: resize to match the RGB image resolution
        min_confidence: mask out pixels below this confidence level (0/1/2)

    Returns True on success.
    """
    aux = {a["type"]: a for a in meta.get("auxiliaryOutputs", [])}
    depth_meta = aux.get("depth", {})
    w = depth_meta.get("width", 256)
    h = depth_meta.get("height", 192)

    raw = np.frombuffer(open(bin_path, "rb").read(), dtype=np.float32)
    if raw.size != w * h:
        return False
    depth = raw.reshape(h, w).copy()

    if os.path.exists(conf_path):
        conf_meta = aux.get("depthConfidence", {})
        cw = conf_meta.get("width", w)
        ch = conf_meta.get("height", h)
        conf_raw = np.frombuffer(open(conf_path, "rb").read(), dtype=np.uint8)
        if conf_raw.size == cw * ch:
            conf = conf_raw.reshape(ch, cw)
            if conf.shape != depth.shape:
                conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_NEAREST)
            depth[conf < min_confidence] = 0.0

    # Upscale to RGB image resolution
    if (w, h) != (target_w, target_h):
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # Convert meters -> millimeters -> uint16
    depth_mm = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    cv2.imwrite(out_png_path, depth_mm)
    return True


def _normalize_splatking(extract_source: str, result: NormalizationResult) -> None:
    """Flatten SplatKing layout and convert LiDAR depth bins to uint16 PNG."""
    result.detected_pattern = "SplatKing/ARKit LiDAR"
    _log(result, "Detected SplatKing/ARKit LiDAR dataset — normalizing layout")

    colmap_dir = os.path.join(extract_source, "COLMAP_Text_Model")
    sensor_dir = os.path.join(extract_source, "sensor_data")

    # --- Flatten COLMAP_Text_Model/ -> root ---
    for item in os.listdir(colmap_dir):
        src = os.path.join(colmap_dir, item)
        dst = os.path.join(extract_source, item)
        if item == "README.txt":
            continue
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        shutil.move(src, dst)
    shutil.rmtree(colmap_dir)
    _log(result, "  Flattened COLMAP_Text_Model/ to dataset root")

    # --- Determine RGB image resolution from first image ---
    images_dir = os.path.join(extract_source, "images")
    img_files = sorted(f for f in os.listdir(images_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not img_files:
        _log(result, "  WARNING: no images found, skipping depth conversion")
        return

    sample_img = cv2.imread(os.path.join(images_dir, img_files[0]))
    if sample_img is None:
        _log(result, "  WARNING: could not read sample image, skipping depth conversion")
        return
    target_h, target_w = sample_img.shape[:2]
    _log(result, f"  RGB resolution: {target_w}x{target_h}")

    # --- Convert depth bins ---
    depths_dir = os.path.join(extract_source, "depths")
    os.makedirs(depths_dir, exist_ok=True)

    json_files = sorted(f for f in os.listdir(sensor_dir) if f.endswith(".json"))
    converted, skipped = 0, 0
    for jf in json_files:
        stem = jf[:-5]  # strip .json
        bin_path = os.path.join(sensor_dir, f"{stem}_depth.bin")
        conf_path = os.path.join(sensor_dir, f"{stem}_confidence.bin")
        out_png = os.path.join(depths_dir, f"{stem}.png")

        if not os.path.exists(bin_path):
            skipped += 1
            continue

        try:
            meta = json.load(open(os.path.join(sensor_dir, jf)))
        except Exception:
            skipped += 1
            continue

        ok = _convert_arkit_depth_bin(
            bin_path, conf_path, meta, out_png, target_w, target_h,
            min_confidence=1,
        )
        if ok:
            converted += 1
        else:
            skipped += 1

    _log(result, f"  Converted {converted}/{converted + skipped} depth maps -> depths/")

    # Remove sensor_data — no longer needed
    shutil.rmtree(sensor_dir)

    result.modified = True
    result.config_overrides["PRESERVE_SCENE_SCALE"] = "true"
    result.config_overrides["USE_POSE_PRIOR_COLMAP_MODEL_FILES"] = "true"
    _log(result, "  Config overrides: PRESERVE_SCENE_SCALE=true, USE_POSE_PRIOR_COLMAP_MODEL_FILES=true")

    # Convert text COLMAP model to binary so colmap_to_nerfstudio_cam.py can read it
    sparse_0 = os.path.join(extract_source, "sparse", "0")
    has_txt = all(os.path.exists(os.path.join(sparse_0, f))
                  for f in ("cameras.txt", "images.txt", "points3D.txt"))
    has_bin = os.path.exists(os.path.join(sparse_0, "cameras.bin"))
    if has_txt and not has_bin:
        import subprocess
        cmd = [
            "colmap", "model_converter",
            "--input_path", sparse_0,
            "--output_path", sparse_0,
            "--output_type", "BIN",
        ]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0:
            _log(result, "  Converted COLMAP text model to binary format")
        else:
            _log(result, f"  WARNING: COLMAP model_converter failed: {r.stderr.decode()[:200]}")


# ---------------------------------------------------------------------------
# Pattern: Generic COLMAP wrapper subdirectory
# Detects any single subdirectory that itself contains sparse/ + images/
# but has a non-standard name (e.g. "COLMAP_Export/", "reconstruction/").
# The caller already handles the single-subdir case for known names, but
# this catches deeper nesting.
# ---------------------------------------------------------------------------

def _is_wrapped_colmap(path: str) -> Optional[str]:
    """Return the wrapper subdir path if path contains a single subdir with sparse/+images/, else None."""
    entries = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e))]
    if len(entries) != 1:
        return None
    candidate = os.path.join(path, entries[0])
    has_images = os.path.isdir(os.path.join(candidate, "images"))
    has_sparse = os.path.isdir(os.path.join(candidate, "sparse"))
    has_transforms = os.path.isfile(os.path.join(candidate, "transforms.json"))
    if has_images and (has_sparse or has_transforms):
        return candidate
    return None


def _normalize_wrapped_colmap(extract_source: str, wrapper: str, result: NormalizationResult) -> None:
    """Flatten a generic wrapper subdirectory to extract_source root."""
    wrapper_name = os.path.basename(wrapper)
    result.detected_pattern = f"Wrapped COLMAP ({wrapper_name}/)"
    _log(result, f"Detected wrapped COLMAP layout ({wrapper_name}/) — flattening to root")
    for item in os.listdir(wrapper):
        src = os.path.join(wrapper, item)
        dst = os.path.join(extract_source, item)
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        shutil.move(src, dst)
    shutil.rmtree(wrapper)
    result.modified = True
    _log(result, f"  Flattened {wrapper_name}/ to dataset root")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def normalize(extract_source: str) -> NormalizationResult:
    """Auto-detect and normalize the dataset at extract_source in-place.

    Called after zip extraction, before the pipeline checks for images/sparse/.
    Returns a NormalizationResult with any config overrides to apply.
    """
    result = NormalizationResult()

    if _is_splatking(extract_source):
        _normalize_splatking(extract_source, result)
        return result

    wrapper = _is_wrapped_colmap(extract_source)
    if wrapper:
        _normalize_wrapped_colmap(extract_source, wrapper, result)
        return result

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Normalize a dataset directory in-place")
    parser.add_argument("path", help="Path to extracted dataset directory")
    args = parser.parse_args()
    r = normalize(args.path)
    print(f"Pattern: {r.detected_pattern}")
    print(f"Modified: {r.modified}")
    print(f"Config overrides: {r.config_overrides}")
