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

"""Image file formats accepted as training/reconstruction input, and how the
pipeline writes images, in one place.

Dependency-free so every pipeline script can import it (the container puts this
file in /opt/ml/code, which is on PYTHONPATH). Masks stay PNG: an image
000000.webp has the mask 000000.webp.png (or 000000.png), as for .jpg inputs.

Every image the pipeline writes keeps full quality:
  * JPEG at quality 100 with 4:4:4 chroma. OpenCV defaults to quality 95 and
    Pillow to quality 75, both with 4:2:0 chroma.
  * WebP lossless. Pillow writes WebP LOSSY by default (quality 80). OpenCV is
    lossless without a quality parameter, but a quality <= 100 makes it lossy.
  * PNG is lossless at any compression level.
Write with imwrite() (OpenCV), or pass pil_save_kwargs(path) to PIL's save().

pycolmap (the 4.0.4 wheel in the container, and later wheels) decodes PNG, JPEG
and TIFF but not WebP: its feature extraction gets no image for a .webp file. Code
that hands images to pycolmap (extract_features, triangulate_points,
incremental_mapping) must give it lossless PNG copies of WebP images
(pycolmap_can_read(), write_png_copy()). The colmap command-line tool reads WebP.
"""

import os

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
# Image files the pycolmap wheels can decode (OpenImageIO built with PNG, JPEG and
# TIFF only; checked in the pycolmap 4.0.4 manylinux wheel).
PYCOLMAP_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
JPEG_QUALITY = 100


def is_image_file(name):
    """True if name has one of IMAGE_EXTS (case-insensitive)."""
    return str(name).lower().endswith(IMAGE_EXTS)


def is_jpeg(path):
    return str(path).lower().endswith((".jpg", ".jpeg"))


def is_webp(path):
    return str(path).lower().endswith(".webp")


def pycolmap_can_read(name):
    """True if pycolmap can decode the image file name (judged by its extension)."""
    return str(name).lower().endswith(PYCOLMAP_EXTS)


def _set_param(params, key, value):
    """cv2.imwrite params (flat key, value list) with key set to value."""
    out = []
    for k, v in zip(params[0::2], params[1::2]):
        if k != key:
            out += [k, v]
    return out + [key, value]


def cv2_write_params(path, params=None):
    """cv2.imwrite params for path that keep full quality: lossless .webp
    (quality > 100), and for .jpg/.jpeg quality 100 with 4:4:4 chroma (when this
    OpenCV has the sampling-factor flag). Other params are kept; a lower JPEG or
    WebP quality passed in params is replaced."""
    import cv2
    p = list(params) if params else []
    if is_webp(path):
        p = _set_param(p, cv2.IMWRITE_WEBP_QUALITY, 101)
    elif is_jpeg(path):
        p = _set_param(p, cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY)
        sampling = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
        s444 = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444", None)
        if sampling is not None and s444 is not None:
            p = _set_param(p, sampling, s444)
    return p


def imwrite(path, img, params=None):
    """cv2.imwrite at full quality (see cv2_write_params). Returns cv2.imwrite's result."""
    import cv2
    return cv2.imwrite(str(path), img, cv2_write_params(path, params))


def pil_save_kwargs(path):
    """Extra PIL Image.save() kwargs for path that keep full quality: lossless
    .webp, and .jpg/.jpeg at quality 100 with 4:4:4 chroma (subsampling=0)."""
    if is_webp(path):
        return {"lossless": True, "quality": 100, "method": 4}
    if is_jpeg(path):
        return {"quality": JPEG_QUALITY, "subsampling": 0}
    return {}


def write_png_copy(src, dst):
    """Write a lossless PNG copy of the image file src to dst (every pixel kept,
    alpha too), e.g. for pycolmap, which cannot decode WebP. Returns True on success."""
    import cv2
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    os.makedirs(os.path.dirname(os.path.abspath(str(dst))), exist_ok=True)
    return cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def list_images(directory, recursive=False):
    """Sorted image paths in directory (relative to it when recursive)."""
    if not os.path.isdir(directory):
        return []
    if not recursive:
        return sorted(f for f in os.listdir(directory)
                      if is_image_file(f) and os.path.isfile(os.path.join(directory, f)))
    return sorted(os.path.relpath(os.path.join(r, f), directory)
                  for r, _, fs in os.walk(directory) for f in fs if is_image_file(f))
