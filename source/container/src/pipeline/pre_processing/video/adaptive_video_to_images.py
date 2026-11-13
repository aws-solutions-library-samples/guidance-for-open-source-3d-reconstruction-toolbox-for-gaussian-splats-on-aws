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

"""Motion-adaptive video frame extraction, optionally keeping the sharpest frame.

1. Keyframes: the adaptive frame extractor (github.com/morishuz/adaptive-frame-
   extractor, MIT) tracks camera motion with dense optical flow and picks a new
   keyframe once the image has moved enough. Fast motion gives dense keyframes,
   slow motion or a stopped camera gives few. Only its keyframes.csv (source
   frame numbers) is used.
2. Sharpness (--sharp): each keyframe gets a block reaching halfway to its
   neighbours, capped at +-window seconds. Every frame in the block is scored with
   sharp-frame-extractor's Tenengrad analyzer (the scorer behind the regular
   "Filter blurry images" video extraction) and the sharpest one is kept. So the
   spacing follows the motion and each spot gets its sharpest frame.
3. Cap: at most --num-frames frames; extra keyframes are thinned evenly.

Frames are decoded once, in order, with OpenCV (auto-rotation on), and written
as frame_00000.png ... in capture order. If the adaptive extractor fails or
finds fewer than 2 keyframes, frames are sampled at regular intervals instead
(still with the sharpness step when --sharp is on). A report is written to
<work-dir>/report.json.
"""

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

log = logging.getLogger("adaptive_video_to_images")
SCORE_LONG_SIDE = 1280  # sharpness is ranked at this size (faster, same ranking)


# --------------------------------------------------------------------------
# Sharpness: sharp-frame-extractor's Tenengrad analyzer, with a local fallback
# --------------------------------------------------------------------------

def make_scorer():
    """Function frames(N,H,W,3 uint8) -> scores, using sharp-frame-extractor's
    TenengradFrameAnalyzer when it is installed."""
    try:
        from sharp_frame_extractor.analyzer.frame_analyzer_base import FrameAnalyzerTask
        from sharp_frame_extractor.analyzer.tenegrad_frame_analyzer import TenengradFrameAnalyzer
        analyzer = TenengradFrameAnalyzer()

        def score(frames):
            return list(analyzer.process(FrameAnalyzerTask(0, None), frames).scores)
        log.info("Sharpness: sharp-frame-extractor TenengradFrameAnalyzer")
        return score
    except Exception as e:  # package missing or API changed
        log.warning(f"sharp-frame-extractor analyzer unavailable ({e}); using the same "
                    f"Tenengrad measure locally")

    def score(frames):
        out = []
        for f in frames:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = g.shape
            k = cv2.getGaussianKernel(h, max(1.0, 0.22 * h)) @ cv2.getGaussianKernel(w, max(1.0, 0.22 * w)).T
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            out.append(float(((gx * gx + gy * gy) * (k / k.sum())).sum()))
        return out
    return score


def small(frame):
    h, w = frame.shape[:2]
    s = SCORE_LONG_SIDE / max(h, w)
    return cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame


# --------------------------------------------------------------------------
# Adaptive keyframes
# --------------------------------------------------------------------------

def profile_threshold(profile):
    """trigger.main_threshold_analysis_px of a profile (motion needed for a new keyframe)."""
    with open(profile, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("main_threshold_analysis_px:"):
                return float(line.split(":", 1)[1].split("#")[0])
    return 100.0


def scaled_profile(profile, threshold, out_path):
    """Copy of a profile with another keyframe motion threshold."""
    with open(profile, encoding="utf-8") as f:
        lines = f.readlines()
    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.strip().startswith("main_threshold_analysis_px:"):
                indent = line[:len(line) - len(line.lstrip())]
                line = f"{indent}main_threshold_analysis_px: {threshold:.3f}\n"
            f.write(line)
    return out_path


def auto_keyframes(binary, profile, video, work_dir, start_frame, max_frames, target):
    """Adaptive keyframes with the motion threshold tuned so the count lands near
    target (the count is roughly inversely proportional to the threshold)."""
    threshold = profile_threshold(profile)
    best = None
    for attempt in range(3):
        prof = scaled_profile(profile, threshold, os.path.join(work_dir, f"auto_profile_{attempt}.yaml"))
        keys = adaptive_keyframes(binary, prof, video, work_dir, start_frame, max_frames)
        if not keys:
            return best[1] if best else None
        log.info(f"Auto density: threshold {threshold:.1f} px -> {len(keys)} keyframes (target {target})")
        # Prefer the run closest to the target without going far below it.
        if best is None or abs(len(keys) - target) < abs(len(best[1]) - target):
            best = (threshold, keys)
        if 0.85 * target <= len(keys) <= 1.3 * target:
            break
        threshold = float(np.clip(threshold * len(keys) / target, 4.0, 400.0))
    return best[1]


def adaptive_keyframes(binary, profile, video, work_dir, start_frame, max_frames):
    """Source frame numbers of the adaptive keyframes, or None on failure."""
    out_dir = os.path.join(work_dir, "frame_extractor")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [binary, video, "--output-dir", out_dir, "--start-frame", str(start_frame)]
    if profile:
        cmd += ["--config", profile]
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]
    log.info("Running: " + " ".join(cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)  # nosemgrep: dangerous-subprocess-use-audit
    except FileNotFoundError:
        log.warning(f"Adaptive frame extractor not found: {binary}")
        return None
    if res.returncode != 0:
        log.warning(f"Adaptive frame extractor failed (exit {res.returncode}): {res.stderr.strip()[-1500:]}")
        return None
    for line in res.stderr.splitlines():
        if line.startswith(("Processing ", "Finished:", "warning:")):
            log.info(f"frame-extractor: {line}")
    run_dir = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
    if not os.path.isdir(run_dir):
        subs = [os.path.join(out_dir, d) for d in os.listdir(out_dir)
                if os.path.isdir(os.path.join(out_dir, d))]
        run_dir = max(subs, key=os.path.getmtime) if subs else ""
    manifest = os.path.join(run_dir, "keyframes.csv")
    if not os.path.isfile(manifest):
        log.warning(f"Adaptive frame extractor wrote no keyframes.csv in {out_dir}")
        return None
    with open(manifest, newline="", encoding="utf-8") as f:
        frames = sorted({int(r["source_frame_index"]) for r in csv.DictReader(f)
                         if r.get("source_frame_index", "").strip()})
    # Its keyframe images are not needed (frames are decoded below).
    shutil.rmtree(os.path.join(run_dir, "keyframes"), ignore_errors=True)
    return frames


def thin(frames, n):
    if n <= 0 or len(frames) <= n:
        return list(frames)
    idx = np.unique(np.round(np.linspace(0, len(frames) - 1, n)).astype(int))
    return [frames[i] for i in idx]


def blocks_for(keys, window, first, last):
    """[(lo, hi, key)] per keyframe: halfway to the neighbours, at most +-window."""
    out = []
    for i, k in enumerate(keys):
        lo = k - window if i == 0 else max(k - window, (keys[i - 1] + k) // 2 + 1)
        hi = k + window if i == len(keys) - 1 else min(k + window, (k + keys[i + 1]) // 2)
        out.append((max(first, lo), min(last, hi), k))
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--video_path", required=True)
    ap.add_argument("-o", "--output_dir", required=True)
    ap.add_argument("-n", "--num_frames", type=int, required=True, help="Maximum number of frames")
    ap.add_argument("--density", default="auto", choices=["auto", "low", "medium", "high"],
                    help="auto: tune the motion threshold so about --num-frames keyframes are "
                         "picked (spacing still follows the motion); low/medium/high: the "
                         "extractor's fixed profiles (count depends on how much the camera moves)")
    ap.add_argument("--sharp", action="store_true",
                    help="Keep the sharpest frame in each keyframe's block (blurry frame removal)")
    ap.add_argument("--window", type=float, default=0.25,
                    help="Max block half-width in seconds for --sharp")
    ap.add_argument("--binary", default="frame-extractor", help="Adaptive frame extractor CLI")
    ap.add_argument("--profiles-dir", default="/opt/frame-extractor/configs",
                    help="Folder with low.yaml / medium.yaml / high.yaml")
    ap.add_argument("--work-dir", default="", help="Intermediate files (default: <output>/../adaptive_frames)")
    ap.add_argument("-s", "--start-time", type=float, default=0.0)
    ap.add_argument("-e", "--stop-time", type=float, default=-1.0, help="-1 = end")
    ap.add_argument("-ll", "--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="[Adaptive-Frames] %(message)s")

    video = args.video_path
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video {video}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = max(0, int(round(args.start_time * fps)))
    last = total - 1
    if args.stop_time and args.stop_time > 0:
        last = min(last, int(round(args.stop_time * fps)))
    if last <= first:
        raise SystemExit(f"Empty frame range {first}-{last} (video has {total} frames at {fps:.2f} fps)")
    work_dir = args.work_dir or os.path.join(os.path.dirname(os.path.abspath(args.output_dir)), "adaptive_frames")
    os.makedirs(work_dir, exist_ok=True)
    log.info(f"Video: {total} frames at {fps:.2f} fps; using frames {first}-{last}; "
             f"max {args.num_frames} frames; density {args.density}; sharpest-in-block {args.sharp}")

    profile = os.path.join(args.profiles_dir, f"{'medium' if args.density == 'auto' else args.density}.yaml")
    max_frames = last - first + 1 if last < total - 1 else None
    if not os.path.isfile(profile):
        log.warning(f"Profile {profile} not found; using the extractor's built-in defaults")
        keys = adaptive_keyframes(args.binary, "", video, work_dir, first, max_frames)
    elif args.density == "auto":
        keys = auto_keyframes(args.binary, profile, video, work_dir, first, max_frames, args.num_frames)
    else:
        keys = adaptive_keyframes(args.binary, profile, video, work_dir, first, max_frames)
        if keys and len(keys) < 0.25 * args.num_frames:
            log.warning(f"Only {len(keys)} keyframes with the '{args.density}' profile "
                        f"(max {args.num_frames}); the camera moves little. Use density 'auto' "
                        f"or 'high' for more frames")
    method = "adaptive"
    if keys:
        keys = [k for k in keys if first <= k <= last]
    if not keys or len(keys) < 2:
        method = "regular_fallback"
        n = max(2, args.num_frames)
        keys = sorted({int(round(x)) for x in np.linspace(first, last, n)})
        log.warning(f"Using {len(keys)} regularly spaced frames instead of adaptive keyframes")
    n_adaptive = len(keys)
    keys = thin(keys, args.num_frames)
    if len(keys) < n_adaptive:
        log.info(f"Adaptive extractor picked {n_adaptive} keyframes; thinned evenly to {len(keys)}")
    else:
        log.info(f"{len(keys)} keyframes ({method})")

    window = int(round(args.window * fps)) if args.sharp else 0
    blocks = blocks_for(keys, window, first, last)
    score = make_scorer() if args.sharp else None

    # Decode once, in order. Frames outside every block are only grabbed.
    os.makedirs(args.output_dir, exist_ok=True)
    picked, bi, best = [], 0, None
    pos = 0
    if blocks[0][0] > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, blocks[0][0])
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    while bi < len(blocks):
        lo, hi, key = blocks[bi]
        if pos < lo:
            if not cap.grab():
                break
            pos += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if score is not None:
            s = score(np.ascontiguousarray(small(frame))[None])[0]
            if best is None or s > best[0]:
                best = (s, pos, frame)
        elif pos == key:
            best = (0.0, pos, frame)
        pos += 1
        if pos > hi:
            if best is not None:
                name = f"frame_{len(picked):05d}.png"
                cv2.imwrite(os.path.join(args.output_dir, name), best[2])
                picked.append({"image": name, "keyframe": key, "frame": best[1],
                               "offset": best[1] - key, "sharpness": round(float(best[0]), 3)})
            bi, best = bi + 1, None
    cap.release()
    if not picked:
        raise SystemExit("No frames were extracted")

    moved = [p for p in picked if p["offset"] != 0]
    log.info(f"Wrote {len(picked)} frames to {args.output_dir}"
             + (f"; sharpest frame differs from the keyframe for {len(moved)} "
                f"(median shift {int(np.median([abs(p['offset']) for p in moved])) if moved else 0} frames)"
                if args.sharp else ""))
    gaps = np.diff([p["frame"] for p in picked]) / fps
    if len(gaps):
        log.info(f"Spacing between frames: median {np.median(gaps):.2f} s, min {gaps.min():.2f} s, "
                 f"max {gaps.max():.2f} s")
    with open(os.path.join(work_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"method": method, "density": args.density, "sharp": args.sharp,
                   "fps": fps, "frame_range": [first, last], "keyframes_found": n_adaptive,
                   "frames_written": len(picked), "frames": picked}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
