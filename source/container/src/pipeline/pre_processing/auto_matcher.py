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

"""
Analyze image overlap to recommend the best COLMAP feature matcher.

Sampling strategy:
    1. Sample consecutive pairs (stride-1) evenly across the dataset
    2. Sample stride-2 and stride-3 pairs to detect ordering even with gaps
    3. Sample distant pairs (stride = N/4) to measure unordered overlap
    4. Use MEDIAN overlap (not mean) to resist outliers
    5. Compute "sequential signal" = fraction of consecutive pairs with
       overlap above a minimum threshold — tolerates scattered bad frames

Decision logic:
    spatial     - GPS coordinates in EXIF or pose priors in use
    sequential  - >=70% of consecutive pairs have meaningful overlap
    vocab       - >1000 images with moderate overlap (unordered large set)
    exhaustive  - low overlap or small unordered set (most reliable fallback)

Usage:
    python auto_matcher.py -i <image_dir> [--pose-priors]
    Prints analysis to stdout. Final line: MATCHER=<method>
"""

import argparse
import os
import statistics
import sys

import cv2

# Sampling
SAMPLE_PAIRS = 30                 # pairs per stride level
MATCH_RATIO_TEST = 0.75           # Lowe's ratio test threshold
ORB_FEATURES = 1000               # ORB features per image
RESIZE_DIM = 800                  # resize long edge for speed

# Decision thresholds
SEQUENTIAL_SIGNAL_THRESH = 0.70   # fraction of consecutive pairs with overlap
OVERLAP_FLOOR = 0.08              # minimum overlap to count as "has overlap"
SEQUENTIAL_MEDIAN_THRESH = 0.20   # median consecutive overlap for sequential
LOW_OVERLAP_THRESH = 0.10         # below this median -> exhaustive
VOCAB_IMAGE_THRESH = 1000         # image count above which vocab is preferred


def get_image_files(image_dir):
    exts = ('.png', '.jpg', '.jpeg')
    return sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(exts) and os.path.isfile(os.path.join(image_dir, f))
    ])


def has_gps_exif(image_files, sample_count=8):
    """Check if sampled images contain GPS EXIF lat/lon."""
    try:
        from PIL import Image
        from PIL.ExifTags import IFD
    except ImportError:
        return False

    step = max(1, len(image_files) // sample_count)
    for idx in range(0, len(image_files), step):
        if idx >= len(image_files):
            break
        try:
            img = Image.open(image_files[idx])
            exif = img.getexif()
            if not exif:
                continue
            gps_ifd = exif.get_ifd(IFD.GPSInfo)
            if gps_ifd and 2 in gps_ifd and 4 in gps_ifd:
                print(f"GPS EXIF detected in {os.path.basename(image_files[idx])}")
                return True
        except Exception:
            continue
    return False


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > RESIZE_DIM:
        scale = RESIZE_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def overlap_ratio(img1, img2, orb, bf):
    """ORB match ratio between two images."""
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0.0
    matches = bf.knnMatch(des1, des2, k=2)
    good = sum(1 for m in matches if len(m) == 2 and m[0].distance < MATCH_RATIO_TEST * m[1].distance)
    total = max(len(kp1), len(kp2))
    return good / total if total > 0 else 0.0


def sample_pairs_at_stride(image_files, stride, num_pairs):
    """Return evenly-spaced (idx, idx+stride) index pairs."""
    n = len(image_files)
    valid = n - stride
    if valid <= 0:
        return []
    step = max(1, valid // num_pairs)
    pairs = []
    for idx in range(0, valid, step):
        pairs.append((idx, idx + stride))
        if len(pairs) >= num_pairs:
            break
    return pairs


def compute_overlaps(image_files, pairs):
    """Compute overlap ratios for a list of index pairs."""
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    ratios = []
    cache = {}
    for a, b in pairs:
        if a not in cache:
            cache[a] = load_gray(image_files[a])
        if b not in cache:
            cache[b] = load_gray(image_files[b])
        img_a, img_b = cache[a], cache[b]
        if img_a is None or img_b is None:
            ratios.append(0.0)
            continue
        ratios.append(overlap_ratio(img_a, img_b, orb, bf))
        # Keep cache bounded
        if len(cache) > 80:
            oldest = min(cache.keys())
            del cache[oldest]
    return ratios


def analyze(image_files):
    """Run multi-stride overlap analysis."""
    n = len(image_files)

    # Consecutive pairs (stride 1, 2, 3)
    pairs_s1 = sample_pairs_at_stride(image_files, 1, SAMPLE_PAIRS)
    pairs_s2 = sample_pairs_at_stride(image_files, 2, SAMPLE_PAIRS // 2)
    pairs_s3 = sample_pairs_at_stride(image_files, 3, SAMPLE_PAIRS // 2)

    # Distant pairs (stride = N/4)
    distant_stride = max(n // 4, 2)
    pairs_distant = sample_pairs_at_stride(image_files, distant_stride, SAMPLE_PAIRS // 2)

    overlaps_s1 = compute_overlaps(image_files, pairs_s1)
    overlaps_s2 = compute_overlaps(image_files, pairs_s2)
    overlaps_s3 = compute_overlaps(image_files, pairs_s3)
    overlaps_distant = compute_overlaps(image_files, pairs_distant)

    # Combine stride-1/2/3 for sequential signal
    all_consecutive = overlaps_s1 + overlaps_s2 + overlaps_s3

    # Sequential signal: fraction of consecutive pairs with meaningful overlap
    if all_consecutive:
        signal = sum(1 for r in all_consecutive if r >= OVERLAP_FLOOR) / len(all_consecutive)
        median_consecutive = statistics.median(all_consecutive)
    else:
        signal = 0.0
        median_consecutive = 0.0

    median_distant = statistics.median(overlaps_distant) if overlaps_distant else 0.0

    return {
        'median_consecutive': median_consecutive,
        'median_distant': median_distant,
        'sequential_signal': signal,
        'samples_s1': len(overlaps_s1),
        'samples_s2': len(overlaps_s2),
        'samples_s3': len(overlaps_s3),
        'samples_distant': len(overlaps_distant),
    }


def recommend(num_images, stats, has_gps, has_pose_priors):
    signal = stats['sequential_signal']
    med_con = stats['median_consecutive']

    if has_gps or has_pose_priors:
        reason = "GPS EXIF detected" if has_gps else "pose priors enabled"
        return "spatial", reason

    if signal >= SEQUENTIAL_SIGNAL_THRESH and med_con >= SEQUENTIAL_MEDIAN_THRESH:
        return "sequential", (
            f"sequential signal {signal:.0%} (>={SEQUENTIAL_SIGNAL_THRESH:.0%}), "
            f"median overlap {med_con:.2f}"
        )

    if num_images > VOCAB_IMAGE_THRESH and med_con >= LOW_OVERLAP_THRESH:
        return "vocab", (
            f"large dataset ({num_images} imgs), "
            f"moderate median overlap {med_con:.2f}"
        )

    if med_con < LOW_OVERLAP_THRESH:
        return "exhaustive", f"low median overlap {med_con:.2f}"

    if num_images > VOCAB_IMAGE_THRESH:
        return "vocab", f"{num_images} images with median overlap {med_con:.2f}"

    return "exhaustive", f"default for {num_images} images, median overlap {med_con:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Auto-detect best SfM matcher")
    parser.add_argument("-i", "--image-dir", required=True)
    parser.add_argument("--pose-priors", action="store_true")
    args = parser.parse_args()

    image_files = get_image_files(args.image_dir)
    if not image_files:
        print("No images found, defaulting to exhaustive")
        print("MATCHER=exhaustive")
        return

    num_images = len(image_files)
    print(f"Analyzing {num_images} images for overlap...")

    gps = has_gps_exif(image_files)
    stats = analyze(image_files)

    print(f"Stride-1 pairs sampled: {stats['samples_s1']}")
    print(f"Stride-2 pairs sampled: {stats['samples_s2']}")
    print(f"Stride-3 pairs sampled: {stats['samples_s3']}")
    print(f"Distant pairs sampled:  {stats['samples_distant']}")
    print(f"Median consecutive overlap: {stats['median_consecutive']:.3f}")
    print(f"Median distant overlap:     {stats['median_distant']:.3f}")
    print(f"Sequential signal:          {stats['sequential_signal']:.1%}")
    print(f"GPS EXIF: {gps}, Pose priors: {args.pose_priors}")

    method, reason = recommend(num_images, stats, gps, args.pose_priors)
    print(f"Recommended: {method} ({reason})")
    print(f"MATCHER={method}")


if __name__ == "__main__":
    main()