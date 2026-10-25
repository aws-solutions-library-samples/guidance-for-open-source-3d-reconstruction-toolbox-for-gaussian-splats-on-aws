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

"""Trains, evaluates, and renders a Faster-GS (NeRFICG) Gaussian splat from a COLMAP dataset."""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import yaml

PINHOLE_MODELS = ("PINHOLE", "SIMPLE_PINHOLE")
METHOD_NAME = "FasterGS"
STAGE_DIR_NAME = "faster_gs_input"

# Headless containers do not ship the NeRFICG GUI (ICGui), so create_config.py omits the
# TRAINING.GUI block, but the FasterGS trainer reads TRAINING.GUI.ACTIVATE unconditionally.
HEADLESS_GUI_BLOCK = {
    "ACTIVATE": False,
    "RENDER_INTERVAL": 10,
    "GUI_STATUS_ENABLED": False,
    "GUI_STATUS_INTERVAL": 20,
    "SKIP_GUI_SETUP": True,
    "FPS_ROLLING_AVERAGE_SIZE": 100,
}


# Prints a prefixed log line and flushes so it shows up in the pipeline log immediately.
def log(msg):
    print(f"[faster-gs] {msg}", flush=True)


# Runs a NeRFICG script as a subprocess from the NeRFICG root; exits the wrapper on failure.
def run_nerficg_script(nerficg_root, script, script_args):
    cmd = [sys.executable, os.path.join("scripts", script)] + list(script_args)
    log(f"Running: {' '.join(cmd)} (cwd={nerficg_root})")
    result = subprocess.run(cmd, cwd=nerficg_root, check=False)
    if result.returncode != 0:
        log(f"ERROR: {script} exited with code {result.returncode}")
        sys.exit(1)


# Returns the first COLMAP sparse model directory (sparse/0 or colmap/sparse/0) that contains cameras.
def find_sparse_model(data_dir):
    for rel in (("sparse", "0"), ("colmap", "sparse", "0")):
        candidate = os.path.join(data_dir, *rel)
        if any(os.path.isfile(os.path.join(candidate, f)) for f in ("cameras.bin", "cameras.txt")):
            return candidate
    return None


# Returns the set of COLMAP camera model names used by a sparse model.
def read_camera_models(sparse_dir):
    import pycolmap
    reconstruction = pycolmap.Reconstruction(sparse_dir)
    models = set()
    for camera in reconstruction.cameras.values():
        model = camera.model
        models.add(model.name if hasattr(model, "name") else str(model))
    return models, len(reconstruction.images)


# Moves the .bin/.txt files the COLMAP undistorter writes into sparse/ down into sparse/0/.
def normalize_undistorted_layout(stage_dir):
    sparse_dir = os.path.join(stage_dir, "sparse")
    sparse_0 = os.path.join(sparse_dir, "0")
    os.makedirs(sparse_0, exist_ok=True)
    for name in os.listdir(sparse_dir):
        src = os.path.join(sparse_dir, name)
        if os.path.isfile(src):
            shutil.move(src, os.path.join(sparse_0, name))


# Builds the NeRFICG input folder (images/ + sparse/0/). The Faster-GS rasterizer is
# pinhole-only and silently ignores lens distortion, so non-pinhole models are undistorted.
def stage_dataset(data_dir, image_dir):
    sparse_dir = find_sparse_model(data_dir)
    if sparse_dir is None:
        log(f"ERROR: no COLMAP model found under {data_dir}/sparse/0 or {data_dir}/colmap/sparse/0")
        sys.exit(1)
    if not os.path.isdir(image_dir):
        log(f"ERROR: image directory not found: {image_dir}")
        sys.exit(1)

    models, num_images = read_camera_models(sparse_dir)
    log(f"COLMAP model: {sparse_dir} | cameras: {sorted(models)} | registered images: {num_images}")

    stage_dir = os.path.join(data_dir, STAGE_DIR_NAME)
    if os.path.islink(stage_dir) or os.path.isfile(stage_dir):
        os.remove(stage_dir)
    elif os.path.isdir(stage_dir):
        shutil.rmtree(stage_dir)

    if models and all(m in PINHOLE_MODELS for m in models):
        # Pinhole already: link the existing data so nothing is duplicated on disk
        os.makedirs(os.path.join(stage_dir, "sparse"), exist_ok=True)
        os.symlink(os.path.abspath(image_dir), os.path.join(stage_dir, "images"))
        os.symlink(os.path.abspath(sparse_dir), os.path.join(stage_dir, "sparse", "0"))
        log("Cameras are pinhole; linked images/ and sparse/0/ without undistortion")
    else:
        log(f"Non-pinhole cameras {sorted(models)} detected; undistorting to PINHOLE")
        cmd = [
            "colmap", "image_undistorter",
            "--image_path", image_dir,
            "--input_path", sparse_dir,
            "--output_path", stage_dir,
            "--output_type", "COLMAP",
        ]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            log(f"ERROR: colmap image_undistorter failed with code {result.returncode}")
            sys.exit(1)
        normalize_undistorted_layout(stage_dir)
        new_models, _ = read_camera_models(os.path.join(stage_dir, "sparse", "0"))
        log(f"Undistorted cameras: {sorted(new_models)}")
    return stage_dir


# Estimates whether all training images fit comfortably in VRAM (PRELOADING_LEVEL 2) or should stay in RAM (1).
def choose_preloading_level(image_dir, num_images, image_scale):
    try:
        import torch
        from PIL import Image
        sample = next(
            (os.path.join(r, f) for r, _, fs in os.walk(image_dir, followlinks=True)
             for f in fs if f.lower().endswith((".png", ".jpg", ".jpeg"))),
            None,
        )
        if sample is None or not torch.cuda.is_available():
            return 1
        with Image.open(sample) as img:
            width, height = img.size
        # float32 RGBA per image, scaled
        image_bytes = width * height * 4 * 4 * (image_scale ** 2)
        total_bytes = image_bytes * num_images
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        level = 2 if total_bytes < 0.35 * vram_bytes else 1
        log(f"Image cache estimate {total_bytes / 1e9:.1f} GB vs {vram_bytes / 1e9:.1f} GB VRAM -> PRELOADING_LEVEL {level}")
        return level
    except Exception as e:  # pylint: disable=broad-except
        log(f"Could not estimate image cache size ({e}); using PRELOADING_LEVEL 1")
        return 1


# Recursively merges override values into a nested config dict.
def deep_merge(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict):
            if not isinstance(base.get(key), dict):
                base[key] = {}
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# Builds the YAML overrides that map toolbox settings onto the Faster-GS config.
def build_overrides(args, dataset_path, preloading_level):
    max_steps = int(args.max_steps)
    # Upstream defaults assume 30k steps; scale densification/ordering schedules with MAX_STEPS
    ratio = max_steps / 30000.0
    overrides = {
        "GLOBAL": {"GPU_INDICES": [0], "FILTER_WARNINGS": True},
        "DATASET": {
            "PATH": dataset_path,
            "IMAGE_SCALE_FACTOR": float(args.image_scale),
            # Keep the COLMAP world frame so the toolbox's post-processing rotations apply
            "APPLY_PCA": False,
            "NORMALIZE_CUBE": None,
            "NORMALIZE_RECENTER": False,
            "TEST_STEP": int(args.test_step),
        },
        "TRAINING": {
            "MODEL_NAME": args.model_name,
            "NUM_ITERATIONS": max_steps,
            "WANDB": {"ACTIVATE": False},
            "GUI": dict(HEADLESS_GUI_BLOCK),
            "DATA": {"PRELOADING_LEVEL": preloading_level},
            "BACKUP": {
                "FINAL_CHECKPOINT": True,
                "RENDER_TESTSET": int(args.test_step) > 0,
                "RENDER_TRAINSET": False,
                "RENDER_VALSET": False,
                "INTERMEDIATE_RENDERINGS": False,
                "TRAINING_STATE": False,
            },
            "DENSIFICATION_END_ITERATION": max(1000, int(14900 * ratio)),
            "MORTON_ORDERING_END_ITERATION": max(1000, int(15000 * ratio)),
            "DENSIFICATION_USE_ABS_GRAD": bool(args.absgrad),
            "OPTIMIZER": {
                "LEARNING_RATE_MEANS_MAX_STEPS": max_steps,
            },
        },
    }
    if not args.no_speedy:
        # Speedy-Splat pruning: ~2x faster training and ~10x fewer Gaussians on the bench scene.
        # Upstream schedule (start 6k, end 30k, every 3k) is scaled to MAX_STEPS; the config
        # comment recommends opacity lr 0.05 with Speedy-Splat.
        overrides["TRAINING"]["SPEEDYSPLAT_PRUNING"] = {
            "USE": True,
            "START_ITERATION": max(500, int(6000 * ratio)),
            "END_ITERATION": max_steps,
            "INTERVAL": max(100, int(3000 * ratio)),
        }
        overrides["TRAINING"]["OPTIMIZER"]["LEARNING_RATE_OPACITIES"] = 0.05
    isp = (args.isp or "none").lower()
    if isp == "ppisp":
        overrides["MODEL"] = {"PPISP": {"USE": True}}
    elif isp == "bilagrid":
        log("Bilateral grid is not available in Faster-GS; using PPISP instead")
        overrides["MODEL"] = {"PPISP": {"USE": True}}
    return overrides


# Returns the newest NeRFICG run directory for a model name, or None.
def find_latest_run(nerficg_root, model_name):
    runs = glob.glob(os.path.join(nerficg_root, "output", METHOD_NAME, f"{model_name}_*"))
    runs = [r for r in runs if os.path.isdir(r)]
    return max(runs, key=os.path.getmtime) if runs else None


# Counts vertices in a PLY file header without loading the whole file.
def count_ply_vertices(ply_path):
    with open(ply_path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return 0


# Trains Faster-GS end to end: stage data, create/patch config, train, export PLY, archive the run.
def cmd_train(args):
    data_dir = os.path.abspath(args.data_dir)
    image_dir = os.path.abspath(args.image_dir or os.path.join(data_dir, "images"))
    nerficg_root = os.path.abspath(args.nerficg_root)

    stage_dir = stage_dataset(data_dir, image_dir)
    _, num_images = read_camera_models(os.path.join(stage_dir, "sparse", "0"))
    preloading_level = choose_preloading_level(os.path.join(stage_dir, "images"), num_images, float(args.image_scale))

    # Generate a full default config for FasterGS + Colmap, then apply toolbox overrides
    run_nerficg_script(nerficg_root, "create_config.py", ["-m", METHOD_NAME, "-d", "Colmap", "-o", args.model_name])
    generated_cfg = os.path.join(nerficg_root, "configs", f"{args.model_name}.yaml")
    with open(generated_cfg, "r") as f:
        cfg = yaml.safe_load(f) or {}
    deep_merge(cfg, build_overrides(args, stage_dir, preloading_level))
    cfg_path = os.path.join(data_dir, "faster_gs_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    log(f"Wrote config: {cfg_path}")

    # Remove stale runs with the same name so the newest run is unambiguous
    for stale in glob.glob(os.path.join(nerficg_root, "output", METHOD_NAME, f"{args.model_name}_*")):
        shutil.rmtree(stale, ignore_errors=True)

    run_nerficg_script(nerficg_root, "train.py", ["-c", cfg_path])
    run_dir = find_latest_run(nerficg_root, args.model_name)
    if run_dir is None:
        log("ERROR: training finished but no NeRFICG run directory was found")
        sys.exit(1)
    log(f"Run directory: {run_dir}")

    run_nerficg_script(nerficg_root, "convert_to_ply.py", ["-d", run_dir])
    final_ply = os.path.join(run_dir, "final.ply")
    if not os.path.isfile(final_ply) or count_ply_vertices(final_ply) == 0:
        log(f"ERROR: exported PLY missing or empty: {final_ply}")
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(args.ply_out)), exist_ok=True)
    shutil.copy2(final_ply, args.ply_out)
    log(f"Exported {count_ply_vertices(final_ply):,} Gaussians -> {args.ply_out}")

    # Move the run (checkpoint, config, test renders/metrics) into the dataset so it is archived
    run_out = os.path.abspath(args.run_dir_out)
    if os.path.exists(run_out):
        shutil.rmtree(run_out)
    shutil.move(run_dir, run_out)
    log(f"Archived run directory -> {run_out}")


# Parses the summary line NeRFICG writes at the end of metrics_8bit.txt ("PSNR:x SSIM:y LPIPS:z").
def parse_metrics_file(path):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for line in reversed(lines):
        pairs = dict(re.findall(r"([A-Za-z]+):([-+0-9.eE]+)", line))
        if {"PSNR", "SSIM", "LPIPS"} <= pairs.keys():
            return {"psnr": float(pairs["PSNR"]), "ssim": float(pairs["SSIM"]), "lpips": float(pairs["LPIPS"])}
    return None


# Converts the test-set metrics written during training into the toolbox metrics.json format.
def cmd_metrics(args):
    candidates = glob.glob(os.path.join(args.run_dir, "test_*", "metrics_8bit.txt"))
    if not candidates:
        log(f"No test-set metrics found in {args.run_dir} (TEST_STEP=0 trains on all images); skipping")
        return
    # Highest iteration wins
    candidates.sort(key=lambda p: int(re.search(r"test_(\d+)", p).group(1)) if re.search(r"test_(\d+)", p) else -1)
    metrics = parse_metrics_file(candidates[-1])
    if metrics is None:
        log(f"Could not parse metrics from {candidates[-1]}; skipping")
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": metrics}, f, indent=2)
    log(f"PSNR {metrics['psnr']:.4f} | SSIM {metrics['ssim']:.4f} | LPIPS {metrics['lpips']:.4f} -> {args.output}")


# Renders the training views in capture order so Images-to-Video can build render.mp4.
def cmd_render(args):
    nerficg_root = os.path.abspath(args.nerficg_root)
    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isfile(os.path.join(run_dir, "checkpoints", "final.pt")):
        log(f"ERROR: no final checkpoint in {run_dir}")
        sys.exit(1)
    shutil.rmtree(os.path.join(run_dir, "inference"), ignore_errors=True)
    run_nerficg_script(nerficg_root, "inference.py", ["-d", run_dir, "-s", args.subset])
    frames = glob.glob(os.path.join(run_dir, "inference", f"{args.subset}_*", "rgb", "*.png"))
    log(f"Rendered {len(frames)} frames to {os.path.join(run_dir, 'inference')}")


# Parses CLI arguments for the train, metrics, and render subcommands.
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train Faster-GS and export a PLY")
    train.add_argument("--data-dir", required=True, help="Dataset root containing images/ and sparse/0 (or colmap/sparse/0)")
    train.add_argument("--image-dir", default=None, help="Image directory (default: <data-dir>/images)")
    train.add_argument("--ply-out", required=True, help="Destination for the exported splat PLY")
    train.add_argument("--run-dir-out", required=True, help="Where to move the NeRFICG run directory after training")
    train.add_argument("--max-steps", type=int, default=30000, help="Training iterations")
    train.add_argument("--test-step", type=int, default=8, help="Hold out every Nth image for metrics (0 = train on all)")
    train.add_argument("--image-scale", type=float, default=1.0, help="Image scale factor")
    train.add_argument("--isp", default="none", choices=["none", "ppisp", "bilagrid"], help="Photometric compensation mode")
    train.add_argument("--absgrad", action="store_true", help="Use absolute-gradient densification (AbsGS)")
    train.add_argument("--no-speedy", action="store_true", help="Disable Speedy-Splat pruning (upstream defaults: more Gaussians, slower)")
    train.add_argument("--model-name", default="faster_gs", help="NeRFICG model name")
    train.add_argument("--nerficg-root", default="/opt/ml/code/nerficg", help="NeRFICG install root")

    metrics = sub.add_parser("metrics", help="Write toolbox metrics.json from a finished run")
    metrics.add_argument("--run-dir", required=True, help="Archived NeRFICG run directory")
    metrics.add_argument("--output", required=True, help="Output metrics.json path")

    render = sub.add_parser("render", help="Render frames for the trajectory video")
    render.add_argument("--run-dir", required=True, help="Archived NeRFICG run directory")
    render.add_argument("--subset", default="train", help="Dataset subset or NeRFICG trajectory to render")
    render.add_argument("--nerficg-root", default="/opt/ml/code/nerficg", help="NeRFICG install root")
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    if ARGS.command == "train":
        cmd_train(ARGS)
    elif ARGS.command == "metrics":
        cmd_metrics(ARGS)
    elif ARGS.command == "render":
        cmd_render(ARGS)
