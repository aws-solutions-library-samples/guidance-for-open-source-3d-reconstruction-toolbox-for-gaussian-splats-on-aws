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

"""Object (human) removal for camera-rig uploads: images/<camera>/<frame>.jpg.

remove_background.py, remove_object_using_mask.py and erase_object_using_mask.py
only read images directly inside one folder. This script runs the human
segmentation once per camera folder, then removes ("remove": pixels set to black,
alpha 0 for PNGs) or inpaints ("erase": AttentiveEraser) the detected people.

The rig layout is preserved: every image keeps its folder, file name, extension and
size, because the rig configurator matches images by camera folder and frame name.
Images are only replaced after every camera folder was processed, so a failure
leaves images/ unchanged.

Work files are written to --work-dir:
  seg/<camera>/<stem>.png          remove_background output (RGBA, alpha = person)
  person_masks/<camera>/<stem>.png binary masks of the frames that were changed (white = person)
  originals/<camera>/<file>        the original version of every replaced image
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

# mask_io lives one folder up (pre_processing/), in the repo and in the container.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from mask_io import read_existing  # noqa: E402

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
SKIP_DIRS = {'refined_masks'}
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

log = logging.getLogger("remove_object_rig")


def find_camera_dirs(images_dir):
    """Relative paths of every folder under images_dir that directly holds images."""
    cam_dirs = []
    for root, dirs, files in os.walk(images_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        if any(f.lower().endswith(IMAGE_EXTS) for f in files):
            cam_dirs.append(os.path.relpath(root, images_dir))
    return sorted(cam_dirs)


def person_mask(seg_path, min_area_fraction, min_blob_px, keep=None):
    """Binary person mask (uint8, 255 = person) from a remove_background output,
    or None when no person blob is large enough (same test as panorama_sfm.py).
    keep: optional bool mask of pixels to consider (False = already masked, e.g. an
    uploaded rig mask). u2net_human_seg also segments large dark foreground objects
    such as the rig, so those pixels are removed before the size test; otherwise
    every frame would count as containing a person."""
    seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
    if seg is None:
        return None
    if seg.ndim == 3 and seg.shape[2] == 4:
        alpha = seg[:, :, 3]
    elif seg.ndim == 3:
        alpha = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    else:
        alpha = seg
    if keep is not None:
        if keep.shape != alpha.shape:
            keep = cv2.resize(keep.astype(np.uint8), (alpha.shape[1], alpha.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
        alpha = np.where(keep, alpha, 0).astype(alpha.dtype)
    if alpha.max() == 0:
        return None
    binary = (alpha > 12).astype(np.uint8)  # > 0.05 * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed)
    min_px = max(min_blob_px, int(alpha.size * min_area_fraction))
    if n_labels <= 1 or stats[1:, cv2.CC_STAT_AREA].max() < min_px:
        return None
    return np.where(alpha > 10, 255, 0).astype(np.uint8)


def black_out(image, mask):
    """Set the person pixels to black (and transparent for 4-channel images)."""
    out = image.copy()
    person = mask > 0
    if out.ndim == 2:
        out[person] = 0
    else:
        out[person, :3] = 0
        if out.shape[2] == 4:
            out[person, 3] = 0
    return out


def run(cmd):
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)  # nosemgrep: dangerous-subprocess-use-audit


def main():
    parser = argparse.ArgumentParser(prog='remove-object-rig', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images-dir', required=True, help='images/ folder with camera sub-folders')
    parser.add_argument('--work-dir', required=True, help='Folder for intermediate files')
    parser.add_argument('--action', choices=['remove', 'erase'], default='remove')
    parser.add_argument('-m', '--model', default='u2net_human_seg', help='remove_background.py model')
    parser.add_argument('-nt', '--num_threads', default='1')
    parser.add_argument('-ng', '--num_gpus', default='0')
    parser.add_argument('-mp', '--model_path', default='stable-diffusion-xl-base-1.0',
                        help='SDXL model folder (erase only)')
    parser.add_argument('-pp', '--pipeline_path', default=None, help='AttentiveEraser pipeline (erase only)')
    parser.add_argument('-gpu', '--use_gpu', default='true')
    parser.add_argument('-method', '--method', default='SIP', choices=['SIP', 'DIP'])
    parser.add_argument('-log', '--log_level', default='info')
    parser.add_argument('--min-area-fraction', type=float, default=0.003,
                        help='Smallest person blob, as a fraction of the image area')
    parser.add_argument('--min-blob-px', type=int, default=1000, help='Smallest person blob in pixels')
    parser.add_argument('--masks-dir', default='',
                        help='Uploaded keep-masks (white = keep, e.g. the rig masked black). Already '
                             'masked pixels are left out of the person masks.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.log_level.lower() == 'debug' else logging.INFO,
                        format="%(levelname)s %(message)s")

    images_dir = os.path.abspath(args.images_dir)
    work_dir = os.path.abspath(args.work_dir)
    if not os.path.isdir(images_dir):
        raise SystemExit(f"Images folder {images_dir} does not exist")
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    seg_root = os.path.join(work_dir, 'seg')
    mask_root = os.path.join(work_dir, 'person_masks')
    erase_in_root = os.path.join(work_dir, 'erase_in')
    out_root = os.path.join(work_dir, 'out')
    orig_root = os.path.join(work_dir, 'originals')

    cam_dirs = find_camera_dirs(images_dir)
    if not cam_dirs:
        log.warning("No images found under %s; nothing to do", images_dir)
        return
    log.info("Object removal (%s) for %d camera folder(s): %s", args.action, len(cam_dirs), cam_dirs)

    # 1) Segment people and build the results in work_dir (images/ is not touched yet).
    replacements = []  # (relative image path, processed file)
    total = 0
    for rel in cam_dirs:
        src_dir = os.path.join(images_dir, rel)
        seg_dir = os.path.join(seg_root, rel)
        mask_dir = os.path.join(mask_root, rel)
        out_dir = os.path.join(out_root, rel)
        for d in (seg_dir, mask_dir, out_dir):
            os.makedirs(d, exist_ok=True)

        files = sorted(f for f in os.listdir(src_dir)
                       if os.path.isfile(os.path.join(src_dir, f)) and f.lower().endswith(IMAGE_EXTS))
        total += len(files)

        # The detector only looks outside the uploaded masks (e.g. the rig): masked
        # pixels are filled with the image's mean unmasked colour before segmenting,
        # so the rig is neither detected nor competes with people for saliency.
        keeps = {}
        det_dir = src_dir
        if args.masks_dir:
            det_dir = os.path.join(work_dir, 'seg_input', rel)
            os.makedirs(det_dir, exist_ok=True)
            n_masked = 0
            for name in files:
                image = cv2.imread(os.path.join(src_dir, name), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                rel_image = name if rel == '.' else f"{rel}/{name}".replace('\\', '/')
                keep = read_existing(args.masks_dir, rel_image, image.shape[:2])
                if keep is not None and not keep.all() and keep.any():
                    keeps[name] = keep
                    image = image.copy()
                    image[~keep] = image[keep].mean(axis=0).astype(np.uint8)
                    n_masked += 1
                cv2.imwrite(os.path.join(det_dir, name), image)
            log.info("%s: %d of %d frames have an uploaded mask; masked pixels hidden from the detector",
                     rel, n_masked, len(files))

        run([sys.executable, os.path.join(SCRIPT_DIR, 'remove_background.py'),
             '-i', det_dir, '-o', seg_dir, '-m', args.model,
             '-nt', str(args.num_threads), '-ng', str(args.num_gpus)])

        changed = []
        for name in files:
            stem = os.path.splitext(name)[0]
            mask = person_mask(os.path.join(seg_dir, stem + '.png'),
                               args.min_area_fraction, args.min_blob_px, keeps.get(name))
            if mask is None:
                continue
            image = cv2.imread(os.path.join(src_dir, name), cv2.IMREAD_UNCHANGED)
            if image is None or image.shape[:2] != mask.shape[:2]:
                log.warning("Skipping %s/%s: unreadable image or mask size mismatch", rel, name)
                continue
            cv2.imwrite(os.path.join(mask_dir, stem + '.png'), mask)
            changed.append(name)
            if args.action == 'remove':
                cv2.imwrite(os.path.join(out_dir, name), black_out(image, mask))
        log.info("%s: %d of %d frames contain people", rel, len(changed), len(files))

        if args.action == 'erase' and changed:
            erase_in = os.path.join(erase_in_root, rel)
            os.makedirs(erase_in, exist_ok=True)
            for name in changed:
                shutil.copy2(os.path.join(src_dir, name), os.path.join(erase_in, name))
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, 'erase_object_using_mask.py'),
                   '-id', erase_in, '-md', mask_dir, '-od', out_dir,
                   '-mp', args.model_path, '-gpu', str(args.use_gpu),
                   '-log', args.log_level, '-method', args.method]
            if args.pipeline_path:
                cmd.extend(['-pp', args.pipeline_path])
            run(cmd)

        for name in changed:
            processed = os.path.join(out_dir, name)
            if os.path.isfile(processed):
                replacements.append((os.path.join(rel, name), processed))
            else:
                log.warning("No processed output for %s/%s; keeping the original", rel, name)

    # 2) Replace the processed frames in place (same folder, name and size).
    replaced = 0
    for rel_path, processed in replacements:
        target = os.path.join(images_dir, rel_path)
        orig = cv2.imread(target, cv2.IMREAD_UNCHANGED)
        new = cv2.imread(processed, cv2.IMREAD_UNCHANGED)
        if orig is None or new is None or orig.shape[:2] != new.shape[:2]:
            log.warning("Keeping the original %s: processed image is missing or has another size", rel_path)
            continue
        backup = os.path.join(orig_root, rel_path)
        os.makedirs(os.path.dirname(backup), exist_ok=True)
        shutil.copy2(target, backup)
        shutil.copyfile(processed, target)
        replaced += 1
    log.info("Object removal replaced %d of %d images (originals in %s)", replaced, total, orig_root)


if __name__ == '__main__':
    main()
