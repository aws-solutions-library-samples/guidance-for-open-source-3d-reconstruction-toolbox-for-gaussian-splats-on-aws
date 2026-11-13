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

"""Mask lens flare and veiling glare so 3DGS training ignores it.

Flare moves with the camera while the scene does not, so excluding it from
the loss lets the other views fill the area in. No learned model is used
(no licence or download): the detector is classical image processing.

Veiling glare is light scattered in the lens around a very bright source (the
sun). It adds a smooth haze on top of the scene, so in the glare even the
darkest parts of the scene (shadows, foliage gaps) are lifted, and the lift
falls off smoothly with distance from the source in every direction.

A bright sunlit surface is different: its own colour is bright, but it has
shadows and dark gaps, its edges are sharp, and its brightness stops at a
sharp boundary (wall to grass) instead of fading. Absolute brightness is
therefore not used; the detector looks at the scene's dark level instead.

1. Dark level E: the local minimum of the darkest colour channel (a lower
   envelope). It is low in normal scenes, also on white walls with any shadow
   or texture, and high inside glare.
2. Candidate sources: clipped blobs (every channel near white), with fragments
   split by branches merged. A source must have a soft edge (glare blurs it);
   clipped walls, windows and signs have hard edges.
3. Veil test per source: in rings of growing distance, the dark level in every
   direction (8 sectors) must stay --veil-lift above the dark level far from
   the source, out to at least --min-reach of the image diagonal, and fall off
   with distance. Directions into the image border or the sky (which is
   already masked) are not tested.
4. The source must be in or next to the sky (the sky mask from --sky-dir, or
   without it bright, smooth, blue pixels), or the whole image must be lifted
   (--min-black-lift): a hazy, glare-filled frame.
5. Mask = the source plus the lifted pixels within the veil radius, with a
   small margin. AND-ed into masks/ (and --also-masks-dir), see mask_io.py.

Faint ghost reflections (rings, polygons) away from the source and flare
without a clipped source are not detected.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mask_io import list_images, read_existing, save_preview, write_keep_mask  # noqa: E402

WORK_SIZE = 640  # long side of the working resolution
N_SECTORS = 8
# Outer ring radii, as fractions of the image diagonal (measured from the source edge).
RINGS = (0.01, 0.03, 0.06, 0.10, 0.15, 0.22, 0.32)


def candidate_sources(dark, args):
    """Clipped blobs (every channel near white): [(bool mask, on_border)]."""
    h, w = dark.shape
    core = (dark > args.core_threshold).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # The sun behind a tree is split by branches into many fragments.
    k = max(3, int(args.merge * max(h, w))) | 1
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(core)
    out = []
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        if area < args.min_core_area * h * w or area > args.max_core_area * h * w:
            continue
        x, y, bw, bh = st[i, :4]
        out.append((lab == i, bool(x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1)))
    # Brightest and largest first, so the strongest sources are tested within the limit.
    out.sort(key=lambda c: -int(c[0].sum()))
    return out[:args.max_sources]


def veil_reach(src, on_border, near_sky, E, valid, sec_of, diag, args):
    """(reach as a fraction of the diagonal, dark level far away, distance map).
    reach = 0 when the dark level is not lifted all around the source."""
    d = cv2.distanceTransform((~src).astype(np.uint8), cv2.DIST_L2, 5)
    far = (d > 0.35 * diag) & valid
    ref = float(np.percentile(E[far], 20)) if far.sum() > 500 else float(np.percentile(E[valid], 10))
    ys, xs = np.nonzero(src)
    sec = sec_of(ys.mean(), xs.mean())
    # Directions that run into the border or into the masked sky cannot be tested.
    need = 3 if (on_border or near_sky) else 6
    reach, prof = 0.0, []
    for r0, r1 in zip((0.0,) + RINGS[:-1], RINGS):
        ring = (d > r0 * diag) & (d <= r1 * diag) & valid
        lifts = [float(np.percentile(E[ring & (sec == s)], 25)) - ref
                 for s in range(N_SECTORS) if (ring & (sec == s)).sum() >= 15]
        if len(lifts) < need or min(lifts) < args.veil_lift:
            break
        prof.append(float(np.median(lifts)))
        reach = r1
    # Glare falls off with distance; a lift that grows outwards is a surface.
    if any(b > a + args.falloff_tolerance for a, b in zip(prof, prof[1:])):
        reach = 0.0
    return reach, ref, d


def detect_flare(img_bgr, args, keep=None, sky=None):
    """Bool mask (full resolution) of flare / glare pixels, and its image fraction.
    keep: optional bool mask of pixels to analyse (False = already masked).
    sky: optional bool sky mask; a source must then be in or next to the sky."""
    H, W = img_bgr.shape[:2]
    s = min(1.0, WORK_SIZE / max(H, W))
    sm = cv2.resize(img_bgr, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    h, w = sm.shape[:2]
    diag = float(np.hypot(h, w))
    lum = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dark = sm.min(axis=2).astype(np.float32) / 255.0  # high only where all channels are lifted
    E = cv2.GaussianBlur(cv2.erode(dark, np.ones((7, 7), np.uint8)), (0, 0), 2)
    valid = np.ones((h, w), bool) if keep is None else \
        cv2.resize(keep.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    none = np.zeros((H, W), bool), 0.0
    if valid.mean() < 0.05:
        return none
    g = cv2.GaussianBlur(lum, (0, 0), 1.0)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    sky_s = None
    if sky is not None:
        sky_s = cv2.resize(sky.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        # No sky step: a stand-in that only tells whether a source is next to clear
        # sky (bright, smooth, blue). Overcast skies are not found; such sources then
        # need the whole-image haze test.
        lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.float32)
        sky_s = (lum > 0.5) & (lab[:, :, 2] < 128 - args.blue_sky_b) & \
            (cv2.blur(grad, (5, 5)) < 0.05)
        sky_s = cv2.morphologyEx(sky_s.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    sky_dist = None
    if sky_s.mean() >= 0.01:
        sky_dist = cv2.distanceTransform((~sky_s).astype(np.uint8), cv2.DIST_L2, 5)
    yy, xx = np.mgrid[:h, :w]

    def sec_of(cy, cx):
        ang = np.arctan2(yy - cy, xx - cx)
        return ((ang + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS

    flare = np.zeros((h, w), bool)
    for src, on_border in candidate_sources(dark, args):
        d0 = cv2.distanceTransform((~src).astype(np.uint8), cv2.DIST_L2, 5)
        # Glare blurs the edge of the light source; clipped walls and windows are sharp.
        if float(np.percentile(grad[(d0 > 0) & (d0 <= 3)], 75)) > args.max_edge:
            continue
        eq_r = np.sqrt(src.sum() / np.pi)
        near_sky = sky_dist is not None and \
            float(sky_dist[src].min()) <= max(args.sky_reach * min(h, w), 1.5 * eq_r)
        # Only a real (already masked) sky removes test directions.
        reach, ref, d = veil_reach(src, on_border, near_sky and sky is not None, E, valid, sec_of, diag, args)
        if reach < args.min_reach:
            continue
        # The source must be the sun: in or next to the sky, or (no sky mask, or the
        # sun just outside it) the whole image is hazy. A sunlit scene with deep
        # shadows and no veil has a far dark level near 0.
        if not near_sky and ref < args.min_black_lift:
            continue
        flare |= src | ((d <= reach * diag) & (E - ref >= args.mask_lift))
    if not flare.any():
        return none

    # Back to full resolution, with a margin for the soft veil edge.
    k = max(3, int(args.grow * min(h, w)) | 1)
    flare = cv2.dilate(flare.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    flare = cv2.resize(flare, (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return flare, float(flare.mean())


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--images-dir", required=True)
    ap.add_argument("-m", "--masks-dir", required=True, help="Masks are written (and merged) here")
    ap.add_argument("--debug-dir", default="", help="Write overlay previews here")
    ap.add_argument("--sky-dir", default="",
                    help="Sky masks from the sky step (<stem>.png, 255 = sky). When given, a light source "
                         "must be in or next to the sky (the sun)")
    ap.add_argument("--sky-reach", type=float, default=0.05,
                    help="How close to the sky a light source must be, as a fraction of the short side")
    ap.add_argument("--blue-sky-b", type=float, default=12.0,
                    help="Without --sky-dir: clear sky next to a source is found as bright, smooth pixels "
                         "with Lab b* below -this (OpenCV units, 0-255 scale)")
    ap.add_argument("--also-masks-dir", default="",
                    help="Also AND the flare into these keep-masks (e.g. masks_no_sky/ for transparent sky)")
    ap.add_argument("--core-threshold", type=float, default=0.96,
                    help="Min channel value (0-1) for a clipped light-source pixel")
    ap.add_argument("--merge", type=float, default=0.015,
                    help="Merge clipped fragments closer than this (fraction of the long side)")
    ap.add_argument("--min-core-area", type=float, default=0.0001,
                    help="Smallest light source, as a fraction of the image")
    ap.add_argument("--max-core-area", type=float, default=0.05,
                    help="Largest light source, as a fraction of the image (bigger = bright surface)")
    ap.add_argument("--max-sources", type=int, default=12, help="Candidate sources tested per image")
    ap.add_argument("--max-edge", type=float, default=0.30,
                    help="Max edge sharpness of a light source (75th pct gradient of luminance on its "
                         "edge); glare-blurred sources are soft, clipped surfaces sharp")
    ap.add_argument("--veil-lift", type=float, default=0.12,
                    help="The dark level must be this much (0-1) above the dark level far from the "
                         "source in every direction")
    ap.add_argument("--min-reach", type=float, default=0.10,
                    help="The veil must reach at least this far, as a fraction of the image diagonal")
    ap.add_argument("--falloff-tolerance", type=float, default=0.05,
                    help="How much the veil may rise again with distance (0-1)")
    ap.add_argument("--min-black-lift", type=float, default=0.08,
                    help="A source that is not next to the sky (or without --sky-dir) needs the dark level "
                         "far from it to be at least this (the whole image is hazy)")
    ap.add_argument("--mask-lift", type=float, default=0.09,
                    help="Inside the veil radius, mask pixels whose dark level is this much above the "
                         "far dark level")
    ap.add_argument("--grow", type=float, default=0.02,
                    help="Safety margin around the detected flare, as a fraction of the short image side")
    ap.add_argument("--max-fraction", type=float, default=0.35,
                    help="Ignore a detection covering more than this fraction (likely a bright scene)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    images = list_images(args.images_dir)
    print(f"[Lens-Flare] {len(images)} images in {args.images_dir}")
    hit, total = 0, []
    for rel in images:
        img = cv2.imread(os.path.join(args.images_dir, rel), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[Lens-Flare] WARNING: unreadable {rel}")
            continue
        # Only analyse pixels that are not already masked (sky, uploaded rig masks).
        keep = read_existing(args.masks_dir, rel, img.shape[:2])
        sky = None
        if args.sky_dir:
            _sp = os.path.join(args.sky_dir, os.path.splitext(rel)[0] + ".png")
            _sm = cv2.imread(_sp, cv2.IMREAD_GRAYSCALE) if os.path.isfile(_sp) else None
            sky = (cv2.resize(_sm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127) \
                if _sm is not None else np.zeros(img.shape[:2], bool)
        flare, frac = detect_flare(img, args, keep, sky)
        if frac > args.max_fraction:
            print(f"[Lens-Flare] {rel}: detection covers {frac:.0%}, ignoring (likely bright scene, not flare)")
            flare[:] = False
            frac = 0.0
        write_keep_mask(args.masks_dir, rel, ~flare)
        if args.also_masks_dir and frac > 0:
            write_keep_mask(args.also_masks_dir, rel, ~flare)
        if frac > 0:
            hit += 1
            total.append(frac)
            save_preview(args.debug_dir, rel, img, flare, (0, 220, 255))
    print(f"[Lens-Flare] Flare found in {hit}/{len(images)} images"
          + (f", median {np.median(total):.1%} of the image masked" if total else ""))


if __name__ == "__main__":
    main()
