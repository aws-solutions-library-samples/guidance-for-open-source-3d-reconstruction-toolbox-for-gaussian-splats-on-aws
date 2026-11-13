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

"""Shared helpers for the generated training masks (lens flare, sky).

Convention: white (255) = keep, black (0) = ignore. Each image gets its mask
written under two names so every consumer finds it:
  masks/<image name>.png   e.g. f_001.jpg.png  (COLMAP --ImageReader.mask_path)
  masks/<image stem>.png   e.g. f_001.png      (nerfstudio --masks-path, gsplat)
New masks are AND-ed with any mask already present, so steps can be chained
and user-supplied masks are kept.
"""

import os

import cv2
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def list_images(images_dir):
    """Image paths relative to images_dir (recursive), sorted."""
    out = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                out.append(os.path.relpath(os.path.join(root, f), images_dir).replace("\\", "/"))
    return sorted(out)


def mask_names(rel_image):
    stem = os.path.splitext(rel_image)[0]
    return [rel_image + ".png", stem + ".png"]


def read_existing(masks_dir, rel_image, shape):
    """Existing keep-mask (bool, image size) or None."""
    for name in mask_names(rel_image) + [rel_image]:
        p = os.path.join(masks_dir, name)
        if os.path.isfile(p):
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape != shape:
                m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            return m > 127
    return None


def write_keep_mask(masks_dir, rel_image, keep):
    """AND `keep` (bool HxW) with any existing mask and write it under both names."""
    existing = read_existing(masks_dir, rel_image, keep.shape)
    if existing is not None:
        keep = keep & existing
    out = keep.astype(np.uint8) * 255
    for name in mask_names(rel_image):
        p = os.path.join(masks_dir, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        cv2.imwrite(p, out)
    return keep


def save_preview(debug_dir, rel_image, image_bgr, removed, color):
    """Write a half-size overlay of the removed region for review."""
    if not debug_dir:
        return
    vis = image_bgr.copy()
    vis[removed] = (0.45 * vis[removed] + 0.55 * np.array(color)).astype(np.uint8)
    h, w = vis.shape[:2]
    s = 960 / max(h, w)
    if s < 1:
        vis = cv2.resize(vis, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    p = os.path.join(debug_dir, os.path.splitext(rel_image)[0] + ".jpg")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, vis)
