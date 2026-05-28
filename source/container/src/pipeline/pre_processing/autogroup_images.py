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
Autogroup images by filename prefix pattern and keep only the selected group.

Groups images by their leading alphabetic prefix (e.g. "DJI_0001.jpg" -> "dji",
"2X0A1234.jpg" -> "2x0a", "IMG_0001.jpg" -> "img"). When AUTOGROUP_TARGET_NAME
is provided and matches a group prefix, that group is selected. Otherwise the
group with the most images is used.

Phase 1: Each group is reconstructed separately by filtering to a single group.
Phase 2 (future): More advanced multi-group reconstruction schemes.

Usage:
    python autogroup_images.py -i <image_dir> [-t <target_name>]
"""

import argparse
import os
import re

IMAGE_EXTS = ('.png', '.jpg', '.jpeg')


def extract_prefix(filename):
    """Extract the leading alphanumeric prefix before the first underscore/dash/dot separator.

    Examples:
        DJI_0001.jpg   -> dji
        2X0A1234.jpg   -> 2x0a
        IMG_0001.jpg   -> img
        frame_001.png  -> frame
        photo001.jpg   -> photo
    """
    stem = os.path.splitext(filename)[0]
    match = re.match(r'^([A-Za-z0-9]*[A-Za-z])', stem)
    if match:
        return match.group(1).lower()
    return stem.lower()


def group_images(image_dir):
    """Group image files by their prefix pattern. Returns dict of prefix -> list of filenames."""
    groups = {}
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        if not os.path.isfile(os.path.join(image_dir, fname)):
            continue
        prefix = extract_prefix(fname)
        groups.setdefault(prefix, []).append(fname)
    return groups


def select_group(groups, target_name):
    """Select the target group by name match, or fall back to the largest group."""
    if target_name:
        target_lower = target_name.lower()
        for prefix, files in groups.items():
            if target_lower in prefix or prefix in target_lower:
                print(f"Matched target '{target_name}' to group '{prefix}' ({len(files)} images)")
                return prefix, files

        print(f"Target '{target_name}' did not match any group, using largest group")

    largest_prefix = max(groups, key=lambda k: len(groups[k]))
    print(f"Selected largest group '{largest_prefix}' ({len(groups[largest_prefix])} images)")
    return largest_prefix, groups[largest_prefix]


def flatten_subdirectories(image_dir):
    """Move all images from subdirectories to the root and remove empty subdirs."""
    moved = 0
    for root, dirs, files in os.walk(image_dir, topdown=False):
        if root == image_dir:
            continue
        for fname in files:
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            src = os.path.join(root, fname)
            dst = os.path.join(image_dir, fname)
            if os.path.exists(dst):
                # Avoid collision: prefix with subdirectory name
                subdir_name = os.path.basename(root)
                dst = os.path.join(image_dir, f"{subdir_name}_{fname}")
            os.rename(src, dst)
            moved += 1
        # Remove directory if empty
        try:
            os.rmdir(root)
        except OSError:
            pass
    if moved:
        print(f"Flattened {moved} images from subdirectories into {image_dir}")


def main():
    parser = argparse.ArgumentParser(description="Autogroup images by prefix pattern")
    parser.add_argument("-i", "--image-dir", required=True, help="Path to images directory")
    parser.add_argument("-t", "--target-name", default="", help="Preferred group prefix name")
    args = parser.parse_args()

    # Flatten any subdirectories first so all images are at the root level
    flatten_subdirectories(args.image_dir)

    # Remove non-image files that may have been copied from the input folder
    for fname in os.listdir(args.image_dir):
        fpath = os.path.join(args.image_dir, fname)
        if os.path.isfile(fpath) and not fname.lower().endswith(IMAGE_EXTS):
            os.remove(fpath)
            print(f"Removed non-image file: {fname}")

    groups = group_images(args.image_dir)
    if len(groups) <= 1:
        count = sum(len(v) for v in groups.values())
        print(f"Only {len(groups)} group(s) found ({count} images), no autogrouping needed")
        return

    print(f"Found {len(groups)} image groups:")
    for prefix, files in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"  '{prefix}': {len(files)} images (e.g. {files[0]})")

    selected_prefix, keep_files = select_group(groups, args.target_name)
    keep_set = set(keep_files)

    removed = 0
    for prefix, files in groups.items():
        if prefix == selected_prefix:
            continue
        for fname in files:
            os.remove(os.path.join(args.image_dir, fname))
            removed += 1

    print(f"Kept {len(keep_files)} images (group '{selected_prefix}'), removed {removed}")


if __name__ == "__main__":
    main()