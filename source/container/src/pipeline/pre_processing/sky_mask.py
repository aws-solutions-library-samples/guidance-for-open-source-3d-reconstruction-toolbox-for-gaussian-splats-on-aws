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

"""Mask the sky so 3DGS training does not fill it with floaters.

Models (all Apache-2.0, code and weights, loaded with Hugging Face transformers):
  - Grounding DINO (IDEA-Research/grounding-dino-base) finds "sky" boxes.
  - SAM 2.1 (facebook/sam2.1-hiera-large) segments the sky inside each box.
    Falls back to SAM (facebook/sam-vit-huge) on older transformers.
  - Depth Anything V2 Small (depth-anything/Depth-Anything-V2-Small-hf) gives a
    relative depth map. Only the Small variant is Apache-2.0; Base/Large are
    CC-BY-NC-4.0 and must not be used here.

SAM alone has two failure modes:
  - False positives: whole tree crowns, car bodies, pale walls marked as sky,
    because its 256x256 mask is smooth and Grounding DINO's "sky" box covers them.
  - Coarse edges: the mask edge sits a few pixels off the true boundary and the
    sky seen between branches stays unmasked, so it trains into a blue/white
    outline around objects.

Refinement at full resolution:
  1. Seeds: pixels that SAM calls sky AND are at the far end of the depth map
     AND are smooth, bright and not green/yellow-green (foliage). These seeds
     give a local sky colour.
  2. Candidates: near SAM's sky, far by depth (local minimum, so gaps between
     twigs count), with SAM probability > --min-prob. A candidate is sky when its
     colour is within --color-tol of the local sky colour, or (next to that core)
     when it is a mix of sky and the local foreground colour (--edge-alpha).
  3. Components: the main sky must contain seeds (and touch the top unless
     --no-require-top). Sky cells enclosed by branches are added when SAM calls
     them sky and they lie within reach of accepted sky. Small sky-coloured gaps
     in textured surroundings (foliage) are added too.
  4. Optional LiDAR veto (--depth-dir): pixels with a measured depth are not sky.
  5. A small dilation so sky-coloured edge pixels are excluded too.

Without the depth model (--no-mono-depth) the depth tests pass everywhere and
only colour, brightness and SAM decide.

Considered and rejected: OpenDroneMap SkyRemoval (AGPL-3.0); SAM 3 (SAM
License, not OSI-permissive).

The result is AND-ed into masks/ (see mask_io.py); sky pixels become black.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mask_io import list_images, read_existing, save_preview, write_keep_mask  # noqa: E402

DINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM2_MODEL = "facebook/sam2.1-hiera-large"
SAM1_MODEL = "facebook/sam-vit-huge"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


class SkySegmenter:
    def __init__(self, device, box_threshold, text_threshold):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        self.device = device
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.dino_proc = AutoProcessor.from_pretrained(DINO_MODEL)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL).to(device).eval()
        try:
            from transformers import Sam2Model, Sam2Processor
            self.sam_proc = Sam2Processor.from_pretrained(SAM2_MODEL)
            self.sam = Sam2Model.from_pretrained(SAM2_MODEL).to(device).eval()
            self.sam_name = SAM2_MODEL
        except ImportError:
            from transformers import SamModel, SamProcessor
            self.sam_proc = SamProcessor.from_pretrained(SAM1_MODEL)
            self.sam = SamModel.from_pretrained(SAM1_MODEL).to(device).eval()
            self.sam_name = SAM1_MODEL
        print(f"[Sky] Models: {DINO_MODEL} + {self.sam_name} on {device}")

    def __call__(self, pil_image):
        torch = self.torch
        W, H = pil_image.size
        inp = self.dino_proc(images=pil_image, text="sky.", return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.dino(**inp)
        det = self.dino_proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[(H, W)])[0]
        boxes = [[float(v) for v in b] for b in det["boxes"].tolist()]
        if not boxes:
            return np.zeros((H, W), np.float32), 0
        sin = self.sam_proc(images=pil_image, input_boxes=[boxes], return_tensors="pt").to(self.device)
        with torch.no_grad():
            sout = self.sam(**sin, multimask_output=False)
        masks = self.sam_proc.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"].cpu(),
                                                 binarize=False)[0]
        prob = torch.sigmoid(masks.float()).reshape(-1, H, W).max(0).values.numpy()
        return prob.astype(np.float32), len(boxes)


class MonoDepth:
    """Relative inverse depth (disparity, larger = nearer) from Depth Anything V2 Small."""

    def __init__(self, device, res):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch, self.device, self.res = torch, device, res
        self.proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        self.model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(device).eval()
        print(f"[Sky] Depth model: {DEPTH_MODEL} at {res}px")

    def __call__(self, pil_image):
        torch = self.torch
        W, H = pil_image.size
        with torch.no_grad():
            # The default 518 px input blurs depth across thin branches; ~1036 px
            # keeps sky gaps between twigs at the far end.
            inp = self.proc(images=pil_image, return_tensors="pt",
                            size={"height": self.res, "width": self.res}).to(self.device)
            d = self.model(**inp).predicted_depth
            d = torch.nn.functional.interpolate(d[None].float(), size=(H, W), mode="bicubic",
                                                align_corners=False)[0, 0]
        return d.cpu().numpy().astype(np.float32)


def find_depth(depth_dir, rel):
    if not depth_dir:
        return None
    stem = os.path.splitext(rel)[0]
    for cand in (stem + ".png", stem.replace("/", "_") + ".png"):
        p = os.path.join(depth_dir, cand)
        if os.path.isfile(p):
            return p
    return None


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _near(mask, r):
    """Pixels within r of mask (a disk dilation; distance transform is much faster
    than cv2.dilate for large r)."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) <= r


def _local_mean(lab, sel, k, step=4):
    """Mean Lab colour of the selected pixels in a k x k window (and their density),
    computed at 1/step resolution for speed."""
    H, W = sel.shape
    h, w = max(1, H // step), max(1, W // step)
    s = cv2.resize(sel.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    sl = cv2.resize(lab * sel[:, :, None], (w, h), interpolation=cv2.INTER_AREA)
    kk = max(3, k // step) | 1
    ws = cv2.blur(s, (kk, kk))
    m = np.stack([cv2.blur(sl[:, :, c], (kk, kk)) for c in range(3)], -1)
    m = cv2.resize(m / np.maximum(ws, 1e-6)[:, :, None], (W, H), interpolation=cv2.INTER_LINEAR)
    ws = cv2.resize(ws, (W, H), interpolation=cv2.INTER_LINEAR)
    return m, ws


def _multiscale_mean(lab, sel, short, scales):
    """Local mean colour of sel; coarser windows fill in where sel is sparse."""
    out = have = None
    for f in scales:
        m, w = _local_mean(lab, sel, int(f * short))
        if out is None:
            out, have = m, w > 0.02
        else:
            out[~have] = m[~have]
            have |= w > 1e-3
    if (~have).any() and sel.any():
        out[~have] = lab[sel].mean(0)
    return out


def _gradient(L, sigma):
    g = cv2.GaussianBlur(L, (0, 0), sigma)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def refine(img_bgr, prob, disp, args, depth_path=None):
    """Full-resolution sky mask from SAM's probability and a relative disparity map."""
    H, W = img_bgr.shape[:2]
    short = min(H, W)
    none = np.zeros((H, W), bool)
    lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)  # CIELAB units
    L = lab[:, :, 0] * 2.55  # 0-255 scale for the gradient thresholds
    A, B = lab[:, :, 1], lab[:, :, 2]
    p = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)

    if disp is not None:
        d = cv2.resize(disp, (W, H), interpolation=cv2.INTER_LINEAR)
        # Relative disparity is affine-ambiguous: map the far end (sky) to 0, near to 1.
        d_lo, d_hi = np.percentile(d, [0.5, 99])
        dn = (d - d_lo) / max(d_hi - d_lo, 1e-6)
        # Depth is blurry at thin structures: the local minimum keeps sky seen
        # between twigs "far".
        r = max(1, int(args.depth_min_win * short)) | 1
        far = cv2.erode(dn, np.ones((r, r), np.uint8)) < args.depth_far
    else:
        dn = np.zeros((H, W), np.float32)
        far = np.ones((H, W), bool)

    # Hue between yellow and green with some chroma: foliage, grass. Blue/cyan skies
    # have b* < 0, overcast skies little chroma, sunsets a* > 0.
    greenish = (B > 0) & (A < 2) & (np.hypot(A, B) > args.green_chroma)

    # 1. Seeds.
    sam_sky = p > args.seed_prob
    if not sam_sky.any():
        return none
    # The sky is the farthest thing in view, so seeds must be close to the lowest
    # disparity inside SAM's sky (distant tree lines and walls are a bit nearer).
    d_sky = float(np.percentile(dn[sam_sky], 2))
    seed_depth = min(args.seed_depth, d_sky + args.seed_depth_margin)
    # Sky is one of the brightest parts of a daylight exposure, so dark smooth
    # foliage cannot seed the sky colour. Per-pixel gradient (no window) so the
    # interiors of small gaps between branches still qualify.
    bright = L >= args.seed_min_rel_l * np.percentile(L, 95)
    seeds = sam_sky & (dn < seed_depth) & (_gradient(L, 1.5) < args.seed_max_grad) & bright & ~greenish
    seeds = cv2.erode(seeds.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if seeds.mean() < args.min_seed_frac:
        return none

    # 2. Candidates and colour tests.
    mu_s = _multiscale_mean(lab, seeds, short, (0.05, 0.15, 0.5))
    dd = lab - mu_s
    # Brighter than the local sky estimate is still sky; darker separates branches.
    dd[:, :, 0] = np.where(dd[:, :, 0] > 0, 0.5 * dd[:, :, 0], dd[:, :, 0])
    dist_s = np.linalg.norm(dd, axis=-1)
    reach = max(3, int(args.reach * short))
    near = _near(p > 0.5, reach)
    fg = ((p < 0.2) | (dn > 2 * args.depth_far) | (dist_s > 3 * args.color_tol)) & ~seeds
    mu_f = _multiscale_mean(lab, fg, short, (0.03, 0.1, 0.5))
    diff = mu_s - mu_f
    n2 = (diff ** 2).sum(-1)
    alpha = ((lab - mu_f) * diff).sum(-1) / np.maximum(n2, 1e-6)
    # Distance from the foreground-sky colour line: the pixel must be a mix of the two.
    resid = np.linalg.norm((lab - mu_f) - alpha[:, :, None] * diff, axis=-1)
    mixed = (np.sqrt(n2) > args.edge_min_contrast) & (resid < args.edge_max_residual) & \
            (alpha >= args.edge_alpha)
    cand = near & far & (p > args.min_prob)
    core = (cand & (dist_s < args.color_tol) & ~greenish) | seeds
    core = cv2.medianBlur(core.astype(np.uint8), 3) > 0
    band = _near(core, max(2, int(args.edge_band * short)))
    sky = core | (cand & mixed & band)

    # 3. Components.
    n, cc, st, _ = cv2.connectedComponentsWithStats(sky.astype(np.uint8), connectivity=8)
    area = st[:, cv2.CC_STAT_AREA]
    has_seed = np.zeros(n, bool)
    has_seed[np.unique(cc[seeds])] = True
    mean_p = np.bincount(cc.ravel(), weights=p.ravel(), minlength=n) / np.maximum(area, 1)
    # Sky cells enclosed by branches: SAM calls them sky and every pixel passed the
    # colour and depth tests, but they hold no seed.
    cells = ~has_seed & (mean_p > args.cell_prob) & (area <= args.cell_max_area * short * short)
    joinable = has_seed | cells
    keep = has_seed.copy()
    if args.require_top:
        top = np.zeros(n, bool)
        top[np.unique(cc[: max(2, int(0.02 * H))])] = True
        keep &= top
    keep[0] = joinable[0] = False
    # Branches split the sky into many components: accept joinable components within
    # reach of accepted sky, repeatedly (chains of cells).
    for _ in range(args.join_iters):
        if not keep.any():
            break
        touched = np.bincount(cc[_near(keep[cc], reach)], minlength=n) > 0
        new = joinable & ~keep & touched
        if not new.any():
            break
        keep |= new
    main = keep[cc]
    # Small sky-coloured gaps (railings, foliage) not joined above: keep only those
    # near the sky and surrounded by texture, not lying on a flat pale wall.
    small = (area <= args.hole_max_area * short * short) & ~keep
    small[0] = False
    holes = small[cc]
    if holes.any():
        tex = cv2.blur(_gradient(L, 1.0), (max(3, int(0.004 * short)) | 1,) * 2)
        n2_, cc2 = cv2.connectedComponents(holes.astype(np.uint8), connectivity=8)
        ring_lab = cv2.dilate(cc2.astype(np.float32), np.ones((7, 7), np.uint8)).astype(np.int32)
        ring_lab[holes | main] = 0
        sums = np.bincount(ring_lab.ravel(), weights=tex.ravel(), minlength=n2_)
        cnts = np.bincount(ring_lab.ravel(), minlength=n2_)
        ok = np.zeros(n2_, bool)
        ok[1:] = sums[1:] / np.maximum(cnts[1:], 1) > args.hole_min_texture
        holes = ok[cc2] & holes & _near(main, reach)
    out = main | holes

    # 4. LiDAR veto, 5. grow.
    k = max(1, int(args.grow * short))
    if k > 1:
        out = cv2.dilate(out.astype(np.uint8), _disk(k // 2)) > 0
    if depth_path:
        dl = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if dl is not None:
            dl = cv2.resize(dl, (W, H), interpolation=cv2.INTER_NEAREST)
            out &= ~(dl > 0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--images-dir", required=True)
    ap.add_argument("-m", "--masks-dir", required=True, help="Masks are written (and merged) here")
    ap.add_argument("--depth-dir", default="", help="Optional uint16 depth PNGs; measured pixels are never sky")
    ap.add_argument("--debug-dir", default="", help="Write overlay previews here")
    ap.add_argument("--sky-dir", default="",
                    help="Also write sky-only masks here (<stem>.png, 255 = sky) for transparent-sky training")
    ap.add_argument("--nosky-masks-dir", default="",
                    help="With --sky-dir: write the keep-masks from before this step here (<stem>.png)")
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--no-mono-depth", dest="mono_depth", action="store_false",
                    help=f"Do not use {DEPTH_MODEL} (more false positives on trees, cars, walls)")
    ap.add_argument("--depth-res", type=int, default=1036,
                    help="Depth model input size (multiple of 14); higher keeps thin branches sharper")
    ap.add_argument("--no-require-top", dest="require_top", action="store_false",
                    help="Keep sky regions that do not touch the top edge (e.g. fisheye/360 images)")
    g = ap.add_argument_group("depth gate (disparity normalised: 0 = farthest, 1 = nearest)")
    g.add_argument("--depth-far", type=float, default=0.15, help="Sky candidates must be farther than this")
    g.add_argument("--depth-min-win", type=float, default=0.01,
                   help="Window of the local depth minimum, as a fraction of the short image side")
    g.add_argument("--seed-depth", type=float, default=0.10, help="Seeds must be farther than this")
    g.add_argument("--seed-depth-margin", type=float, default=0.05,
                   help="...and within this of the farthest depth inside SAM's sky")
    g = ap.add_argument_group("seeds")
    g.add_argument("--seed-prob", type=float, default=0.5, help="SAM sky probability for seeds")
    g.add_argument("--seed-max-grad", type=float, default=20.0, help="Seeds must be smooth (gradient of L, 0-255)")
    g.add_argument("--seed-min-rel-l", type=float, default=0.75,
                   help="Seeds must be at least this times the image's 95th-percentile lightness")
    g.add_argument("--green-chroma", type=float, default=8.0,
                   help="Yellow-green/green pixels with more chroma than this are never seeds or core sky")
    g.add_argument("--min-seed-frac", type=float, default=0.0005,
                   help="No sky in the image if seeds cover less than this fraction")
    g = ap.add_argument_group("colour and edges")
    g.add_argument("--reach", type=float, default=0.08,
                   help="How far from SAM's sky to look, as a fraction of the short image side")
    g.add_argument("--color-tol", type=float, default=8.0, help="Max Lab distance from the local sky colour")
    g.add_argument("--min-prob", type=float, default=0.03, help="Sky needs at least this SAM probability")
    g.add_argument("--edge-band", type=float, default=0.004,
                   help="Mixed sky/foreground pixels are sky only within this of the core sky")
    g.add_argument("--edge-alpha", type=float, default=0.35,
                   help="A mixed pixel is sky when its colour is this far (0-1) from the local "
                        "foreground towards the sky colour; lower masks more of the edge")
    g.add_argument("--edge-min-contrast", type=float, default=8.0,
                   help="Mixing test only where sky and foreground colours differ by more than this (Lab)")
    g.add_argument("--edge-max-residual", type=float, default=10.0,
                   help="Mixing test only for colours this close to the sky-foreground line (Lab)")
    g = ap.add_argument_group("components")
    g.add_argument("--cell-prob", type=float, default=0.5,
                   help="Seedless sky components (cells between branches) need this mean SAM probability")
    g.add_argument("--cell-max-area", type=float, default=0.05, help="Largest cell, fraction of short_side^2")
    g.add_argument("--join-iters", type=int, default=6, help="Rounds of joining cells to the sky")
    g.add_argument("--hole-max-area", type=float, default=0.002,
                   help="Largest stand-alone sky-coloured gap, fraction of short_side^2")
    g.add_argument("--hole-min-texture", type=float, default=6.0,
                   help="Gaps must be surrounded by texture (foliage), not a flat surface")
    ap.add_argument("--grow", type=float, default=0.0015,
                    help="Final dilation, as a fraction of the short image side")
    ap.add_argument("--max-fraction", type=float, default=0.9,
                    help="Ignore a detection covering more than this fraction of the image")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    import torch
    from PIL import Image
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    images = list_images(args.images_dir)
    print(f"[Sky] {len(images)} images in {args.images_dir}")
    seg = SkySegmenter(device, args.box_threshold, args.text_threshold)
    depth = MonoDepth(device, args.depth_res) if args.mono_depth else None

    fracs, t0 = [], time.time()
    for n, rel in enumerate(images, 1):
        path = os.path.join(args.images_dir, rel)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[Sky] WARNING: unreadable {rel}")
            continue
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        prob, n_boxes = seg(pil)
        if n_boxes:
            disp = depth(pil) if depth is not None else None
            sky = refine(img, prob, disp, args, find_depth(args.depth_dir, rel))
        else:
            sky = np.zeros(img.shape[:2], bool)
        frac = float(sky.mean())
        if frac > args.max_fraction:
            print(f"[Sky] {rel}: sky covers {frac:.0%}, ignoring this detection")
            sky[:] = False
            frac = 0.0
        if args.sky_dir:
            # Sky-only masks (255 = sky) and the keep-masks from before this step
            # (lens flare / uploaded masks), so trainers that support it can learn
            # the sky as transparent instead of ignoring it.
            stem = os.path.splitext(rel)[0]
            before = read_existing(args.masks_dir, rel, sky.shape)
            for d, m in ((args.sky_dir, sky), (args.nosky_masks_dir, before)):
                if not d:
                    continue
                p = os.path.join(d, stem + ".png")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                img_out = (m.astype(np.uint8) * 255) if m is not None else np.full(sky.shape, 255, np.uint8)
                cv2.imwrite(p, img_out)
        write_keep_mask(args.masks_dir, rel, ~sky)
        fracs.append(frac)
        if frac > 0:
            save_preview(args.debug_dir, rel, img, sky, (255, 80, 0))
        if n % 25 == 0 or n == len(images):
            print(f"[Sky] {n}/{len(images)} images, {(time.time() - t0) / n:.1f} s/image")
    with_sky = [f for f in fracs if f > 0]
    print(f"[Sky] Sky found in {len(with_sky)}/{len(images)} images"
          + (f", median {np.median(with_sky):.1%} of the image masked" if with_sky else ""))


if __name__ == "__main__":
    main()
