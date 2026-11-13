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
This script is the main entry point into training a gaussian splat from a
given set of images or video.

It uses a pipeline class to create and configure a pipeline containing a
list of components (each containing a script command with parameters). Each
pipeline component will get executed sequentially and there can be infinite
amount of components that can be chained together. The component types can be
pre-processing, reconstruction, training, or post-processing based on the component function use.
The scripts for components are ordered by task type under the pipeline directory
such as post_processing, training, etc.

Component Types = ['PRE_PROCESSING', 'RECONSTRUCTION', 'TRAINING', 'POST_PROCESSING']
Component Environments = ['EXECUTABLE', 'PYTHON']
                            _________________________________________________________________________
                            |                           EXAMPLE PIPELINE                             |
                            |  __________________     __________________     __________________      |
                            |  |                 |    |                 |    |                 |     |
                            |  |   COMPONENT 1   |    |   COMPONENT 2   |    |   COMPONENT N   |     |   
(.mp4,.mov,.zip,.tar.gz)o>-----| (PRE_PROCESS):  |----|     (RECON):    |----|  (COMP_TYPE):   |--//---->o[.ply,.spz,.sog,.usdz,.mp4,.png,.tar.gz]
                            |  | VIDEO-TO-IMAGES |    |     COLMAP      |    |  DO-SOMETHING   |     |
                            |  |     SCRIPT      |    |     SCRIPT      |    |   EXECUTABLE    |     |
                            |  |_________________|    |_________________|    |_________________|     |
                            |                                                                        |
                            |________________________________________________________________________|

ERROR CODES
700, "Required environment variables not set. Check that the payload has the required fields"
705, "Configuration not supported. Only pose prior transform json or pose prior colmap model files can be enabled, not both."
710, "Improper file type given for prior pose transformations. Only '.zip' is supported."
715, "Issue transforming pose to colmap component"
720, "Issue creating video to images component"
730, "Issue creating background removal component"
735, "Issue creating spherical image component"
740, "Issue creating human subject removal component"
745, "Reconstruction Software name given not implemented"
750, "Issue creating the reconstruction component"
755, "Issue creating the Colmap to Nerfstudio component"
760, "Trainer specified does not match proper configuration"
766, "Gaussian splat training diverged: all Gaussians are NaN/Inf (convergence failure)"
, "Issue exporting splat from NerfStudio"
771, "Issue calculating metrics"
775, "Issue rendering trajectory video"
776, "Issue extracting video thumbnail"
777, "Issue cleaning point cloud"
780, "Issue cropping splat bounding box"
781, "Issue removing PLY comments"
782, "Issue creating derivative ply files"
783, "Issue transforming coordinates"
784, "Issue mirroring PLY"
785, "Issue rotating PLY"
786, "Issue converting ply to SOG"
787, "Issue converting ply to USDZ"
788, "Issue converting ply to SPZ"
795, "General error running the pipeline"
800, "Issue generating or uploading collision voxel data"
801, "Issue generating or uploading LOD SOG bundle"
802, "Issue creating mesh extraction component"
804, "Issue creating GS mesh extraction component"
805, "Issue uploading GS mesh GLB to S3"
803, "Issue uploading mesh GLB to S3"
804, "Issue creating depth enhancement component"
"""

import re
import os
import sys
import ast
import time
import json
import math
import boto3
import torch
import shutil
import zipfile
import multiprocessing
import subprocess
import json as _json
import time as time_module
import threading as _threading
from pathlib import Path
from PIL import Image
from pipeline import Pipeline, Status, ComponentEnvironment, ComponentType, Component
from utils import (
    read_camera_params_from_file, validate_input_media,
    load_config, obj_to_glb, count_up_to, untar_gz, process_images,
    select_largest_colmap_model, create_tarball, has_alpha_channel,
    cleanup_dataset, cleanup_cuda_memory, validate_and_resize_images,
    extract_images_from_zip_temp, resize_images_to_common_dimensions,
    normalize_uploaded_masks,
    setup_local_debug, copy_to_local_output, print_container_version_info,
    update_dynamodb_metrics, update_component_phase_completion,
    parse_3dgrut_metrics_from_log, parse_gsplat_metrics_from_log,
    send_task_success, send_task_failure, send_task_heartbeat,
    flatten_images_for_gsplat, add_projected_observations_for_gsplat,
    remove_fully_masked_images_for_gsplat,
    rebuild_points3d_tracks_for_gsplat
)


def world_frame_dataparser_args(config, user_supplied_model=False):
    """nerfstudio dataparser args that keep the input world frame.

    When the user supplies the camera poses and PRESERVE_SCENE_SCALE is on, the splat
    should come out in that world frame. The poses can be supplied as pose priors
    (transforms.json or COLMAP model files, e.g. ARKit) or as a ready COLMAP model with
    reconstruction skipped; such a model may itself have been built from priors, which
    we can't tell, so it is treated the same way. The nerfstudio dataparsers otherwise
    re-orient the scene (orientation_method='up') and move its origin to the mean camera
    position (center_method='poses'), so the output no longer lines up with the source
    coordinates even though its scale is metric.
    """
    preserve = config.get('PRESERVE_SCENE_SCALE', 'false').lower() == 'true'
    priors = config.get('USE_POSE_PRIOR_TRANSFORM_JSON', 'false') == 'true' or \
        config.get('USE_POSE_PRIOR_COLMAP_MODEL_FILES', 'false') == 'true'
    if preserve and (priors or user_supplied_model):
        source = "pose priors" if priors else "uploaded COLMAP model"
        print(f"PRESERVE_SCENE_SCALE with {source}: keeping the input world frame "
              f"(--orientation-method none --center-method none)")
        return ["--orientation-method", "none", "--center-method", "none"]
    return []


# MCMC Gaussian cap estimate (splatfacto-mcmc, gsplat-depth-mcmc, multi-GPU gsplat mcmc).
# MCMC starts with one Gaussian per SfM point and only adds Gaussians up to the cap,
# so the cap must leave room to grow above the point count.
MCMC_CAP_POINTS_FACTOR = 2.5      # target at least this many Gaussians per SfM point
MCMC_CAP_HARD_MAX = 10_000_000    # same as the Gradio input maximum
MCMC_VRAM_USABLE_FRACTION = 0.85  # leave headroom for fragmentation
MCMC_VRAM_RESERVED_GB = 2.5       # CUDA context, SSIM/bilateral grid, eval renders
# Training memory per million Gaussians, in GB: parameters + gradients + Adam state
# (59 floats each at SH degree 3, ~1 KB per Gaussian) plus rasterization buffers that
# grow with the training image size. Conservative estimate, not a measurement.
MCMC_GB_PER_MILLION_BASE = 1.2
MCMC_GB_PER_MILLION_PER_MEGAPIXEL = 0.8


def count_sfm_points(dataset_path):
    """Number of 3D points in the COLMAP model training will start from, or 0."""
    import struct
    for d in (os.path.join(dataset_path, "colmap", "sparse", "0"),
              os.path.join(dataset_path, "sparse", "0"),
              os.path.join(dataset_path, "colmap", "sparse"),
              os.path.join(dataset_path, "sparse")):
        b, t = os.path.join(d, "points3D.bin"), os.path.join(d, "points3D.txt")
        try:
            if os.path.isfile(b) and os.path.getsize(b) >= 8:
                with open(b, "rb") as f:
                    return struct.unpack("<Q", f.read(8))[0]
            if os.path.isfile(t):
                with open(t) as f:
                    return sum(1 for line in f if line.strip() and not line.startswith("#"))
        except (OSError, struct.error, ValueError):
            continue
    return 0


def gpu_total_memory_gb():
    """Smallest per-GPU memory in GB from nvidia-smi (does not create a CUDA context), or 0."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        vals = [float(v) for v in out.split() if v.strip()]
        return min(vals) / 1024.0 if vals else 0.0
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def estimate_mcmc_cap(user_cap, num_points, num_images, megapixels, gpu_gb, images_on_gpu):
    """Return (cap, explanation). user_cap is honoured as a minimum; the cap is raised to
    MCMC_CAP_POINTS_FACTOR x the SfM points, but not past what fits in GPU memory."""
    floor = int(num_points * MCMC_CAP_POINTS_FACTOR)
    mem_cap = MCMC_CAP_HARD_MAX
    mem_note = "GPU memory unknown"
    if gpu_gb > 0:
        image_cache_gb = num_images * megapixels * 1e6 * 3 / 1e9 if images_on_gpu else 0.0
        per_million = MCMC_GB_PER_MILLION_BASE + MCMC_GB_PER_MILLION_PER_MEGAPIXEL * megapixels
        budget = gpu_gb * MCMC_VRAM_USABLE_FRACTION - MCMC_VRAM_RESERVED_GB - image_cache_gb
        mem_cap = max(0, int(budget / per_million * 1e6))
        mem_note = (f"GPU {gpu_gb:.1f} GB fits ~{mem_cap:,} ({per_million:.1f} GB per million at "
                    f"{megapixels:.1f} MP, {image_cache_gb:.1f} GB image cache)")
    cap = min(max(user_cap, min(floor, mem_cap)), MCMC_CAP_HARD_MAX)
    note = (f"MCMC Gaussian cap {cap:,} (setting {user_cap:,}; {num_points:,} SfM points x "
            f"{MCMC_CAP_POINTS_FACTOR} = {floor:,}; {mem_note})")
    if cap < floor:
        note += ". Cap is below the target, so MCMC has little room to add Gaussians"
    if gpu_gb > 0 and user_cap > mem_cap:
        note += ". The setting is above the GPU memory estimate and may run out of memory"
    return cap, note


def build_transparent_sky_images(images_dir, sky_dir, out_dir):
    """Write RGBA copies of the training images with alpha = 0 in the sky.

    Files keep their original relative names (e.g. f_001.jpg) so they match the
    COLMAP model, but hold PNG data (loaders detect the format from the content).
    sky_dir has <stem>.png masks with 255 = sky; images without one get alpha 255.
    Returns (number of images, fraction of sky pixels).
    """
    import cv2
    import numpy as np
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    n, sky_px, total_px = 0, 0, 0
    for root, _, files in os.walk(images_dir):
        for f in files:
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            src = os.path.join(root, f)
            rel = os.path.relpath(src, images_dir)
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            _sky_path = os.path.join(sky_dir, os.path.splitext(rel)[0] + ".png")
            sky = cv2.imread(_sky_path, cv2.IMREAD_GRAYSCALE) if os.path.isfile(_sky_path) else None
            if sky is not None:
                if sky.shape != img.shape[:2]:
                    sky = cv2.resize(sky, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                is_sky = sky > 127
                img[:, :, 3] = np.where(is_sky, 0, img[:, :, 3])
                sky_px += int(is_sky.sum())
            total_px += img.shape[0] * img.shape[1]
            ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not ok:
                raise RuntimeError(f"could not encode {rel}")
            dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(buf.tobytes())
            n += 1
    return n, (sky_px / total_px if total_px else 0.0)


def safe_extract_zip(zip_file_path, dest_path):
    """Extract a zip with zip-slip protection and Windows path separators fixed.

    Some Windows zip tools (e.g. Windows PowerShell 5.1 Compress-Archive) store
    entry names with backslashes ('scene\\colmap\\sparse\\0\\cameras.bin'). On
    Linux, zipfile treats those as one flat file name, so the folder structure
    is lost. Normalize them to '/' before extracting.
    """
    real_dest = os.path.realpath(dest_path)
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        members = zip_ref.infolist()
        fixed = 0
        for info in members:
            if "\\" in info.filename:
                info.filename = info.filename.replace("\\", "/")
                fixed += 1
            entry_path = os.path.realpath(os.path.join(dest_path, info.filename))
            if not entry_path.startswith(real_dest + os.sep):
                raise ValueError(f"Zip slip detected: {info.filename}")
        zip_ref.extractall(dest_path, members=members)
    return fixed


def run_mapper_checked(pipeline, i, db_path, config, local_debug, log):
    """Run the COLMAP mapper component and check its result.

    In LOCAL_DEBUG mode the database lives in /tmp and is lost with the container,
    so it is copied to <dataset>/database_failed_mapper.db when the mapping fails
    or is partial. The mapper can "succeed" with a few frames and almost no points,
    and training then fails much later (e.g. splatfacto's k-nearest-neighbour init
    on 2 points), so stop here with a clear error when fewer than 10% of the images
    registered or the model has fewer than 1000 points; below 50% only warn."""
    def _keep_db(reason):
        if local_debug and os.path.isfile(db_path):
            _keep = os.path.join(config['DATASET_PATH'], 'database_failed_mapper.db')
            try:
                shutil.copy2(db_path, _keep)
                log.error(f"{reason}; COLMAP database kept at {_keep}")
            except Exception as _copy_err:
                log.warning(f"Could not keep the COLMAP database: {_copy_err}")

    try:
        pipeline.run_component(i)
    except Exception:
        _keep_db("Mapper failed")
        raise

    try:
        import sqlite3 as _sql
        import pycolmap as _pc
        _con = _sql.connect(db_path)
        _n_db = _con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        _con.close()
        _args = pipeline.components[i].args
        _out = next((_args[_j + 1] for _j, _a in enumerate(_args[:-1]) if _a == "--output_path"), None)
        _best = (0, 0, None)
        if _out and os.path.isdir(_out):
            for _m in sorted(os.listdir(_out)):
                _mp = os.path.join(_out, _m)
                if os.path.isfile(os.path.join(_mp, "images.bin")):
                    _r = _pc.Reconstruction(_mp)
                    _best = max(_best, (_r.num_reg_images(), _r.num_points3D(), _m))
    except Exception as _chk_err:
        log.warning(f"Could not check the mapper result: {_chk_err}")
        return
    log.info(f"Mapper result: {_best[0]}/{_n_db} images registered, {_best[1]} points "
             f"(model {_best[2]})")
    if not _n_db:
        return
    _msg = f"the mapper registered only {_best[0]} of {_n_db} images with {_best[1]} points"
    if _best[0] < 0.1 * _n_db or _best[1] < 1000:
        # Too small to train on: stop here instead of failing inside training.
        _keep_db(f"SfM failed: {_msg}")
        raise RuntimeError(f"SfM failed: {_msg}")
    if _best[0] < 0.5 * _n_db:
        log.warning(f"Partial reconstruction: {_msg}")
        _keep_db(f"Partial reconstruction: {_msg}")


def read_gps_priors(script, image_dir, dataset_dir, output, log):
    """Read EXIF GPS (or priors.csv) into gps_priors.json before any step rewrites
    the images (OpenCV drops EXIF). Returns the parsed JSON or None."""
    result = subprocess.run(
        [sys.executable, script, 'read', '--images-dir', image_dir,
         '--dataset-dir', dataset_dir, '--output', output],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        log.info(line)
    if result.returncode != 0 or not os.path.isfile(output):
        log.warning(f"Reading GPS priors failed: {result.stderr.strip()[-1000:]}")
        return None
    with open(output, encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    ##################################
    # INITIALIZATION
    ##################################
    try:
        container_start = time_module.time()
        print(f"=== CONTAINER STARTUP TIMING ===")
        print(f"Container started at: {time_module.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Print version information at startup
        print_container_version_info()
        
        version_info_done = time_module.time()
        print(f"Version info completed in: {version_info_done - container_start:.1f}s")

        # Open config with default values
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        config_names = list(config.keys())
        config_values = list(config.values())
        config = load_config(config_names, config_values)
        
        # Sanity check on environment vars/constants
        if config['DATASET_PATH'] == "" or config['CODE_PATH'] == "" or \
            config['UUID'] == "" or config['FILENAME'] == "" or \
            (config['LOCAL_DEBUG'] == "false" and (config['S3_INPUT'] == "" or config['S3_OUTPUT'] == "")):
            error_message = """Error Code 700: Required environment variables not set.
                Check that the payload has the required fields"""
            raise RuntimeError(error_message)

        # A directory input may be given with a trailing slash (e.g. "scene/").
        # Output names are built from os.path.splitext(FILENAME)[0], which keeps the
        # slash and produces paths like "<uuid>/scene/.mp4", so normalize it once here.
        config['FILENAME'] = config['FILENAME'].rstrip('/\\')
        
        # Setup path constants
        OUTPUT_TAR_PATH = "/opt/ml/model/model.tar.gz"
        OUTPUT_DATASET_PATH = "/opt/ml/model/dataset"
        TRAIN_EXPERIMENT_NAME = "train-stage-1"
        RESUME_TRAIN_EXPERIMENT_NAME = "train-stage-2"
        EVAL_METRIC_FOLDER = "/opt/ml/model/dataset/eval"
        EVAL_METRIC_PATH = "/opt/ml/model/dataset/eval/metrics.json"
        IS_BATCH = 'AWS_BATCH_JOB_ID' in os.environ
        TASK_TOKEN = config.get('TASK_TOKEN', '')
        GPU_MAX_IMAGES = 500 # est at 4k
        MAP_ANYTHING_MAX_IMAGES = 100 # for memory efficient mode
        REFINE_STEPS_SPLATFACTO = max(24000, int(config['MAX_STEPS']))
        REFINE_STEPS_3DGRUT = max(12000, int(config['MAX_STEPS']))
        ENABLE_MULTI_GPU = "false"
        LOCAL_DEBUG = os.environ.get('LOCAL_DEBUG', config.get('LOCAL_DEBUG', 'false')).lower() == 'true'
        ENABLE_TASK_TOKEN_CALLBACK = IS_BATCH and bool(TASK_TOKEN) and not LOCAL_DEBUG
        # Faster-GS (NeRFICG) is a standalone trainer: it never routes through gsplat depth loss
        # or nerfstudio, trains on a single GPU, and outputs in the raw COLMAP world frame.
        IS_FASTER_GS = config['MODEL'] == 'faster-gs'
        FASTER_GS_MODELS_DIR = os.path.join(config['DATASET_PATH'], "faster_gs_models")
        # gsplat simple_trainer with sparse COLMAP-point depth loss. gsplat-depth uses
        # the "default" densification strategy, gsplat-depth-mcmc the "mcmc" strategy
        # (capped at NUM_GAUSSIANS).
        IS_GSPLAT_DEPTH = config['MODEL'] in ('gsplat-depth', 'gsplat-depth-mcmc')
        GENERATE_COLLISION = config.get('GENERATE_COLLISION', 'false').lower() == 'true'
        GENERATE_LOD = config.get('GENERATE_LOD', 'false').lower() == 'true'
        # Single mesh switch: dn-splatter/ags-mesh models use their native extractor,
        # other splat models use the Gaussian-splat TSDF extractor (see Extract-Mesh).
        GENERATE_MESH = config.get('GENERATE_MESH', 'true').lower() == 'true'
        ENABLE_ABSGRAD = config.get('ENABLE_ABSGRAD', 'false').lower() == 'true'
        REFINE_POSES = config.get('REFINE_POSES', 'false').lower() == 'true'
        ENHANCE_DEPTH = config.get('ENHANCE_DEPTH', 'false').lower() == 'true'
        CALIBRATE_RIG = config.get('CALIBRATE_RIG', 'false').lower() == 'true'
        BUNDLE_ADJUST_COLMAP_MODEL = config.get('BUNDLE_ADJUST_COLMAP_MODEL', 'false').lower() == 'true'
        # Generated training masks (masks/, white = keep): lens flare first, then sky.
        REMOVE_LENS_FLARE = config.get('REMOVE_LENS_FLARE', 'false').lower() == 'true'
        REMOVE_SKY = config.get('REMOVE_SKY', 'false').lower() == 'true'
        # Video input: motion-adaptive keyframes (adaptive_video_to_images.py) instead of
        # regular intervals. Combines with FILTER_BLURRY_IMAGES (sharpest frame per block).
        ADAPTIVE_FRAME_EXTRACTION = config.get('ADAPTIVE_FRAME_EXTRACTION', 'false').lower() == 'true'
        ADAPTIVE_FRAME_DENSITY = config.get('ADAPTIVE_FRAME_DENSITY', 'auto').lower()
        if ADAPTIVE_FRAME_DENSITY not in ('auto', 'low', 'medium', 'high'):
            ADAPTIVE_FRAME_DENSITY = 'auto'

        # Collision voxelization requires metric world scale to produce accurate collision geometry.
        # Force PRESERVE_SCENE_SCALE on whenever GENERATE_COLLISION is enabled.
        if GENERATE_COLLISION and config.get('PRESERVE_SCENE_SCALE', 'false').lower() != 'true':
            config['PRESERVE_SCENE_SCALE'] = 'true'
            print("GENERATE_COLLISION is enabled — forcing PRESERVE_SCENE_SCALE=true for accurate collision geometry")

        # Check if video or zip of images given
        VIDEO = validate_input_media(config['FILENAME'])

        if IS_BATCH:
            # AWS Batch environment setup
            os.environ['SM_MODEL_DIR'] = '/tmp/model'
            os.environ['SM_CHANNEL_TRAIN'] = '/tmp/input/train'
            os.environ['SM_CHANNEL_MODEL'] = '/tmp/input/model'
            os.environ['SM_OUTPUT_DATA_DIR'] = '/tmp/output'
            os.environ['MODEL_PATH'] = '/tmp/input/model'
            
            # Create directories
            os.makedirs('/tmp/model', exist_ok=True)
            os.makedirs('/tmp/input/train', exist_ok=True)
            os.makedirs('/tmp/input/model', exist_ok=True)
            os.makedirs('/tmp/output', exist_ok=True)
            
            # Download input data from S3 (skip if LOCAL_DEBUG)
            if not LOCAL_DEBUG:
                s3_client = boto3.client('s3')
            else:
                s3_client = None
            
            # Parse S3 paths from environment variables
            if s3_client:
                s3_input = os.environ.get('S3_INPUT', '')
                s3_model = os.environ.get('MODEL_INPUT', '')

            if s3_input.startswith('s3://'):
                bucket, key = s3_input[5:].split('/', 1)
                filename = os.path.basename(key)
                # Download to both locations for compatibility
                local_path_train = os.path.join('/tmp/input/train', filename)
                local_path_data = os.path.join('/opt/ml/input/data/train', filename)
                os.makedirs('/opt/ml/input/data/train', exist_ok=True)
                s3_client.download_file(bucket, key, local_path_train)
                s3_client.download_file(bucket, key, local_path_data)

            if s3_client and s3_model.startswith('s3://'):
                bucket, key = s3_model[5:].split('/', 1)
                local_path = '/tmp/input/model/models.tar.gz'
                try:
                    s3_client.download_file(bucket, key, local_path)
                    print(f"Downloaded models.tar.gz from s3://{bucket}/{key}")
                except Exception as _e:
                    print(f"Could not download models.tar.gz from s3://{bucket}/{key}: {_e}. Skipping.")
            elif s3_client and not s3_model and s3_input.endswith('model.tar.gz'):
                # MODEL_INPUT not provided but S3_INPUT is a model.tar.gz (resume job) —
                # derive the models.tar.gz path from the same S3 prefix as S3_INPUT
                input_bucket, input_key = s3_input[5:].split('/', 1)
                models_key = '/'.join(input_key.split('/')[:-2]) + '/models.tar.gz'
                local_path = '/tmp/input/model/models.tar.gz'
                try:
                    s3_client.download_file(input_bucket, models_key, local_path)
                    print(f"Downloaded models.tar.gz from s3://{input_bucket}/{models_key}")
                except Exception as _e:
                    print(f"Could not download models.tar.gz from s3://{input_bucket}/{models_key}: {_e}. Skipping.")

        # Unpack the sam2 models
        models_start = time_module.time()
        print(f"Starting model extraction at: {time_module.strftime('%Y-%m-%d %H:%M:%S')}")
        untar_gz(os.path.join(os.environ["MODEL_PATH"], "models.tar.gz"), os.environ["MODEL_PATH"])

        # Unpack all models from S3 - OPTIMIZED: Extract only needed files
        models_archive = os.path.join(os.environ["MODEL_PATH"], "models.tar.gz")
        
        # Check if models already extracted (for warm containers)
        u2net_dst = os.path.expanduser("~/.u2net")
        if not os.path.exists(u2net_dst):
            untar_gz(models_archive, os.environ["MODEL_PATH"])
        else:
            print(f"Models already extracted, skipping extraction")
        
        # Move models to expected locations
        # Move U2NET models to home directory
        u2net_src = os.path.join(os.environ["MODEL_PATH"], ".u2net")
        u2net_dst = os.path.expanduser("~/.u2net")
        if os.path.exists(u2net_src):
            os.makedirs(os.path.dirname(u2net_dst), exist_ok=True)
            shutil.move(u2net_src, u2net_dst)
        
        # Move PyTorch models to cache directory
        torch_cache_src = os.path.join(os.environ["MODEL_PATH"], ".cache")
        torch_cache_dst = os.path.expanduser("~/.cache")
        if os.path.exists(torch_cache_src):
            os.makedirs(torch_cache_dst, exist_ok=True)
            shutil.copytree(torch_cache_src, torch_cache_dst, dirs_exist_ok=True)
        
        # Move vocab tree to code directory
        vocab_src = os.path.join(os.environ["MODEL_PATH"], "vocab_tree_flickr100K_words32K.bin")
        vocab_dst = os.path.join(config['CODE_PATH'], "vocab_tree_flickr100K_words32K.bin")
        if os.path.exists(vocab_src):
            shutil.move(vocab_src, vocab_dst)
        
        # Move Stable Diffusion XL model to dataset directory
        sd_model_src = os.path.join(os.environ["MODEL_PATH"], "stable-diffusion-xl-base-1.0")
        sd_model_dst = os.path.join(config['DATASET_PATH'], "stable-diffusion-xl-base-1.0")
        if os.path.exists(sd_model_src):
            shutil.move(sd_model_src, sd_model_dst)
        
        models_done = time_module.time()
        print(f"Model extraction completed in: {models_done - models_start:.1f}s")
        print(f"Total startup time before pipeline: {models_done - container_start:.1f}s")

        # Instantiate Pipeline Session
        pipeline = Pipeline(
            name="3DGS-Pipeline",
            uuid=config['UUID'],
            num_threads=str(multiprocessing.cpu_count()),
            num_gpus=str(torch.cuda.device_count()),
            log_verbosity=config['LOG_VERBOSITY']
        )
        log = pipeline.session.log

        # Store the full list of GPUs
        if int(pipeline.config.num_gpus)>0:
            os.environ['CUDA_VISIBLE_DEVICES'] = count_up_to(int(pipeline.config.num_gpus))
            USE_GPU = "true"
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
            USE_GPU = "false"

        pipeline.session.status = Status.INIT
        
        # Now setup local debug with proper logger
        LOCAL_DEBUG = setup_local_debug(config, log)
        
        # Clean up previous output in local mode
        if LOCAL_DEBUG:
            workflow_output = os.path.join(os.path.dirname(config['CODE_PATH']), 'workflow-output')
            # Only remove UUID directory from workflow-output to prevent duplicate files
            uuid_output_dir = os.path.join(workflow_output, config['UUID'])
            if os.path.exists(uuid_output_dir):
                # Remove all files and subdirectories
                for item in os.listdir(uuid_output_dir):
                    item_path = os.path.join(uuid_output_dir, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                log.info(f"Cleaned previous output files in: {uuid_output_dir}")
            else:
                # Create the directory if it doesn't exist
                os.makedirs(uuid_output_dir, exist_ok=True)
        
        log.info(f"Successfully extracted {os.path.join(os.environ['MODEL_PATH'], 'models.tar.gz')} \
                 to {os.environ['MODEL_PATH']}")
        log.info(f"Pipeline status changed to {pipeline.session.status}")
    except Exception as e:
        error_message = f"""Required environment variables not set.
            Check that the payload has the required fields: {e}"""
        print(f"ERROR 700: {error_message}")
        sys.exit(1)

    # Options and Defaults
    log.info(f"UUID: {config['UUID']}")
    log.info(f"Dataset Path: {config['DATASET_PATH']}")
    log.info(f"Filename: {config['FILENAME']}")
    log.info(f"S3 Input Path: {config['S3_INPUT']}")
    log.info(f"S3 Output Path: {config['S3_OUTPUT']}")
    log.info(f"Execution mode: {'AWS Batch (ECS)' if IS_BATCH else 'SageMaker'}")
    if IS_BATCH:
        log.info(f"  Batch Job ID: {os.environ.get('AWS_BATCH_JOB_ID', 'N/A')}")
    log.info(f"  Model: {config['MODEL']}")
    log.info(f"  Is Video?: {VIDEO}")
    log.info(f"  Run reconstruction: {config['RUN_RECON'] == 'true'}")
    log.info(f"  Run training: {config['RUN_TRAIN'] == 'true'}")
    log.info(f"  Resume training: {config['RUN_RECON'] == 'false' and config['RUN_TRAIN'] == 'true'}")
    log.info(f"  Only export: {config['RUN_RECON'] == 'false' and config['RUN_TRAIN'] == 'false'}")
    # SQLite on EFS: EFS does not support POSIX file locking which SQLite requires.
    # Redirect both temp files and the COLMAP database itself to local storage.
    os.environ['SQLITE_TMPDIR'] = '/tmp'

    # Ensure we have an /images directory in dataset path for Colmap/Glomap
    image_path = os.path.join(config['DATASET_PATH'], "images")
    if not os.path.isdir(image_path):
        log.info(f"Creating '/images' directory in {config['DATASET_PATH']}")
        os.makedirs(image_path, exist_ok=True)

    # Ensure we have a /sparse directory in dataset path for NerfStudio
    sparse_path = os.path.join(config['DATASET_PATH'], "sparse")
    sparse_model_path = os.path.join(sparse_path, "0")

    # Remove dangling symlink from prior 3DGRUT run before creating directory
    if os.path.islink(sparse_path) and not os.path.exists(sparse_path):
        os.remove(sparse_path)

    if not os.path.isdir(sparse_path):
        log.info(f"Creating '/sparse/0' directory in {config['DATASET_PATH']}")
        os.makedirs(sparse_model_path, exist_ok=True)

    # Create the output directory for pre-processing
    filter_output_dir = os.path.join(config['DATASET_PATH'], "filtered_images")
    if not os.path.isdir(filter_output_dir):
        os.makedirs(filter_output_dir, exist_ok=True)

    # Create the output directory for export
    output_path = os.path.join(config['DATASET_PATH'], "exports")
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Create the masked background directory
    mask_bg_output_dir = os.path.join(config['DATASET_PATH'], "masked_bg_images")
    if not os.path.isdir(mask_bg_output_dir):
        os.makedirs(mask_bg_output_dir, exist_ok=True)

    # Create the masked human directory
    mask_human_output_dir = os.path.join(config['DATASET_PATH'], "masked_human_images")
    if not os.path.isdir(mask_human_output_dir):
        os.makedirs(mask_human_output_dir, exist_ok=True)

    # Setup Colmap and Nerfstudio directories
    # In LOCAL_DEBUG mode (typically EFS-mounted), place COLMAP database on local
    # storage to avoid SQLite locking protocol errors (EFS lacks POSIX file locks).
    if LOCAL_DEBUG:
        colmap_db_path = os.path.join('/tmp', 'colmap_database.db')
    else:
        colmap_db_path = os.path.join(config['DATASET_PATH'], "database.db")
    os.environ['COLMAP_DB_PATH'] = colmap_db_path
    transforms_in_path = os.path.join(config['DATASET_PATH'], "transforms-in.json")
    transforms_out_path = os.path.join(config['DATASET_PATH'], "transforms.json")
    colmap_vocab_path = os.path.join(config['CODE_PATH'], "vocab_tree_flickr100K_words32K.bin")

    if config['MODEL'] in ("splatfacto", "splatfacto-big", "splatfacto-mcmc"):
        model = "splatfacto"
    elif config['MODEL'] in ("dn-splatter", "dn-splatter-big", "ags-mesh"):
        model = config['MODEL']
    elif config['MODEL'] in ("gsplat-depth", "gsplat-depth-mcmc"):
        model = "splatfacto"
    else:
        model = config['MODEL']
    
    # Determine model directory name based on model type
    model_dir_name = "nerfstudio_models"
    if config['MODEL'] == "3dgrt" or config['MODEL'] == "3dgrut":
        model_dir_name = "3dgrut_models"
    
    model_config_path = os.path.join(config['CODE_PATH'], "outputs", "unnamed", model, TRAIN_EXPERIMENT_NAME, "config.yml")
    model_ckpt_path = os.path.join(config['CODE_PATH'], "outputs", "unnamed", model, TRAIN_EXPERIMENT_NAME, model_dir_name)
    ply_path = os.path.join(output_path, "splat.ply")
    sog_path = os.path.join(output_path, "splat.sog")
    spz_path = os.path.join(output_path, "spz.spz")
    usdz_path = os.path.join(output_path, "splat.usdz")
    orig_ply_path = os.path.join(output_path, "orig.ply")
    spz_ply_path = os.path.join(output_path, "spz.ply")
    usdz_ply_path = os.path.join(output_path, "usdz.ply")
    sog_ply_path = os.path.join(output_path, "sog.ply")
    collision_mesh_path = os.path.join(output_path, "collision_mesh.ply")
    voxel_path = os.path.join(output_path, "splat.voxel.json")
    lod_dir = os.path.join(output_path, "lod")

    # For spherical, will have 6 views per 360 image using cube faces so will be 6x images
    config['MAX_NUM_IMAGES'] = str(int(config['MAX_NUM_IMAGES']))

    # Setup paths
    input_filename_extension = os.path.splitext(config['FILENAME'])[1]
    input_file_path = os.path.join(config['DATASET_PATH'], config['FILENAME'])
    current_dir_path = os.path.dirname(os.path.realpath(__file__))
    os.environ['PYTHONPATH'] = f"{current_dir_path}:{os.environ.get('PYTHONPATH', '')}"

    ##################################
    # DETECT AND EXTRACT MODEL.TAR.GZ
    ##################################
    # Check if input is a model.tar.gz file for resuming training OR if resuming training look for model.tar.gz
    model_tar_found = False
    colmap_zip_found = False
    resume_training_active = False
    
    if config['FILENAME'].endswith('model.tar.gz') or config['FILENAME'].endswith('.tar.gz'):
        model_tar_found = True
    elif config['RUN_RECON'] == 'false' and config['RUN_TRAIN'] == 'true':
        # Check if this is a COLMAP reconstruction zip (not a model archive)
        if config['FILENAME'].endswith('.zip'):
            colmap_zip_found = True
            log.info(f"Detected potential COLMAP reconstruction zip: {config['FILENAME']}")
        else:
            # Check if FILENAME.zip exists (user may have omitted the extension)
            zip_candidate = config['FILENAME'] + '.zip'
            if os.path.exists(os.path.join(config['DATASET_PATH'], zip_candidate)):
                config['FILENAME'] = zip_candidate
                colmap_zip_found = True
                log.info(f"Found matching zip file: {zip_candidate}")
            # Check if FILENAME is a directory with reconstruction data already extracted
            elif config['FILENAME'].strip('/') and \
                    os.path.isdir(os.path.join(config['DATASET_PATH'], config['FILENAME'])):
                # Unzipped COLMAP export: same layouts as the zip (images/ + sparse/,
                # or colmap/images + colmap/sparse, optionally inside one sub-folder).
                dir_path = os.path.join(config['DATASET_PATH'], config['FILENAME'])
                _subs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
                _roots = [dir_path] + ([os.path.join(dir_path, _subs[0])] if len(_subs) == 1 else [])
                if any((os.path.isdir(os.path.join(_r, 'images')) or os.path.isdir(os.path.join(_r, 'colmap', 'images')))
                       and (os.path.isdir(os.path.join(_r, 'sparse')) or os.path.isdir(os.path.join(_r, 'colmap', 'sparse')))
                       for _r in _roots):
                    colmap_zip_found = True
                    log.info(f"Detected pre-extracted COLMAP reconstruction directory: {config['FILENAME']}")
                else:
                    log.warning(f"Folder {config['FILENAME']} has no images/ + sparse/ "
                                f"(or colmap/images + colmap/sparse); not a COLMAP model")
            # Check if a COLMAP model exists directly in DATASET_PATH (already extracted).
            # images/ and sparse/0/ are created empty at startup, so require real files.
            elif any(os.path.isfile(os.path.join(config['DATASET_PATH'], 'sparse', '0', _f))
                     for _f in ('cameras.bin', 'cameras.txt')) and \
                    any(_fs for _r, _d, _fs in os.walk(os.path.join(config['DATASET_PATH'], 'images'))):
                colmap_zip_found = True
                log.info(f"Detected COLMAP reconstruction data already in dataset directory")
            else:
                _avail = sorted(os.listdir(config['DATASET_PATH']))
                raise RuntimeError(
                    pipeline.report_error(
                        706,
                        f"No COLMAP model found for training without reconstruction: FILENAME="
                        f"'{config['FILENAME']}' is not a .zip or folder in {config['DATASET_PATH']} "
                        f"(contents: {_avail})"))
    elif config['RUN_RECON'] == 'true' and config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true' and \
            config['FILENAME'].endswith('.zip'):
        # Zip contains images + COLMAP pose priors — extract and normalize before reconstruction
        colmap_zip_found = True
        log.info(f"Detected pose-prior zip for reconstruction: {config['FILENAME']}")
    elif config['RUN_RECON'] == 'false' or config['RUN_TRAIN'] == 'false':
        # Look for model.tar.gz in dataset directory for resume training or export-only
        for file in os.listdir(config['DATASET_PATH']):
            if file.endswith('model.tar.gz') or file == 'model.tar.gz':
                config['FILENAME'] = file
                model_tar_found = True
                log.info(f"Found model archive for resume training/export: {file}")
                break
    
    colmap_zip_needs_conversion = False  # True when zip has sparse/ but no transforms.json
    ZIP_HAS_MASKS = False  # True when zip contains a masks/ directory
    if colmap_zip_found:
        log.info(f"Processing COLMAP reconstruction: {config['FILENAME']}")
        zip_path = os.path.join(config['DATASET_PATH'], config['FILENAME'])
        
        # Handle zip file extraction. An unzipped folder takes the same path: it is
        # copied to temp/ (the input folder is left unchanged) and processed like the
        # extracted zip, so the colmap/ prefix layout, masks and depths are handled.
        _is_dir_input = os.path.isdir(zip_path) and bool(config['FILENAME'].strip('/'))
        if (config['FILENAME'].endswith('.zip') and os.path.exists(zip_path)) or _is_dir_input:
            temp_path = os.path.join(config['DATASET_PATH'], 'temp')
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
            if _is_dir_input:
                shutil.copytree(zip_path, temp_path)
                log.info(f"Copied COLMAP folder {zip_path} to {temp_path}")
            else:
                _fixed = safe_extract_zip(zip_path, temp_path)
                if _fixed:
                    log.warning(f"Zip used Windows '\\' path separators in {_fixed} entries; normalized to '/'")
            
            # Check if extracted content is in a subdirectory
            temp_contents = os.listdir(temp_path)
            log.info(f"Zip extracted contents (top-level): {temp_contents}")
            if len(temp_contents) == 1 and os.path.isdir(os.path.join(temp_path, temp_contents[0])):
                extract_source = os.path.join(temp_path, temp_contents[0])
                log.info(f"Single subdir detected, extract_source={extract_source}, contents={os.listdir(extract_source)[:10]}")
            else:
                extract_source = temp_path

            # Auto-detect and normalize vendor-specific dataset layouts
            import sys as _sys
            _norm_dir = os.path.join(current_dir_path, 'pre_processing')
            if _norm_dir not in _sys.path:
                _sys.path.insert(0, _norm_dir)
            from normalize_dataset import normalize as _normalize_dataset
            _norm_result = _normalize_dataset(extract_source)
            if _norm_result.modified:
                log.info(f"Dataset normalized: pattern='{_norm_result.detected_pattern}'")
                for _k, _v in _norm_result.config_overrides.items():
                    config[_k] = _v
                    log.info(f"  Config override applied: {_k}={_v}")
            
            # Verify required COLMAP structure exists
            # Support both flat layout (images/, sparse/) and colmap-prefix
            # layout (colmap/images/, colmap/sparse/) produced by AtlasMobile.
            has_images = os.path.exists(os.path.join(extract_source, 'images')) or \
                        os.path.exists(os.path.join(extract_source, 'colmap', 'images'))
            has_sparse = os.path.exists(os.path.join(extract_source, 'sparse')) or \
                        os.path.exists(os.path.join(extract_source, 'colmap', 'sparse'))
            has_transforms = os.path.exists(os.path.join(extract_source, 'transforms.json'))
            
            if has_images and (has_sparse or has_transforms):
                log.info(f"Valid COLMAP reconstruction found: images={has_images}, sparse={has_sparse}, transforms={has_transforms}")
                if has_sparse and not has_transforms:
                    colmap_zip_needs_conversion = True
                    log.info("No transforms.json in zip - Colmap-to-Nerfstudio conversion will be required")

                # Detect masks directory in the zip (flat or under colmap/)
                has_masks = os.path.exists(os.path.join(extract_source, 'masks')) or \
                           os.path.exists(os.path.join(extract_source, 'colmap', 'masks'))
                if has_masks:
                    log.info("Masks directory detected in zip — will enable mask training")
                
                # Drop reconstruction folders left by an earlier run in the same dataset
                # directory (LOCAL_DEBUG reuses workflow-input). A stale colmap/sparse/0
                # would otherwise win over the new input's sparse/ (the move below is
                # skipped when colmap/sparse/0 is populated), and stale images/ would
                # block the colmap/images hoist.
                _src_items = set(os.listdir(extract_source))
                _stale = []
                if 'colmap' not in _src_items and os.path.isdir(os.path.join(config['DATASET_PATH'], 'colmap')):
                    _stale.append('colmap')
                if 'images' not in _src_items and os.path.isdir(os.path.join(extract_source, 'colmap', 'images')):
                    _stale.append('images')
                for _item in _stale:
                    shutil.rmtree(os.path.join(config['DATASET_PATH'], _item))
                    log.warning(f"Removed {_item}/ left in {config['DATASET_PATH']} by an earlier run")

                # Move contents to dataset directory
                for item in os.listdir(extract_source):
                    src = os.path.join(extract_source, item)
                    dst = os.path.join(config['DATASET_PATH'], item)
                    if os.path.exists(dst):
                        if os.path.isdir(dst):
                            shutil.rmtree(dst)
                        else:
                            os.remove(dst)
                    shutil.move(src, dst)

                # Hoist images/masks/depths from colmap/ to root when the
                # zip uses the colmap-prefix layout (colmap/images/ etc.).
                _colmap_dir = os.path.join(config['DATASET_PATH'], 'colmap')
                if os.path.isdir(_colmap_dir):
                    for _subdir in ('images', 'masks', 'depths'):
                        _src = os.path.join(_colmap_dir, _subdir)
                        _dst = os.path.join(config['DATASET_PATH'], _subdir)
                        # images/ (and sparse/0/) are created as empty placeholders at
                        # startup, so an existing but file-less folder must not block
                        # the hoist; otherwise models that skip the later copy step
                        # (e.g. gsplat-depth) train with an empty images/ folder.
                        _dst_empty = os.path.isdir(_dst) and not any(
                            _fs for _r, _d, _fs in os.walk(_dst)
                        )
                        if os.path.isdir(_src) and (not os.path.exists(_dst) or _dst_empty):
                            if _dst_empty:
                                shutil.rmtree(_dst)
                            shutil.move(_src, _dst)
                            log.info(
                                f"Hoisted colmap/{_subdir}/ to"
                                f" {_subdir}/"
                            )

                if has_masks:
                    ZIP_HAS_MASKS = True
                    masks_dir = os.path.join(config['DATASET_PATH'], 'masks')
                    # The zip contains masks in COLMAP convention: <image_filename>.png
                    # e.g. scan_001_view02.png.png (flat) or face_00/pano_002.png.png (panorama)
                    # NerfStudio --masks-path expects the image filename unchanged:
                    # e.g. scan_001_view02.png or face_00/pano_002.png
                    # Walk the full masks tree to handle both flat and subdirectory layouts.
                    for root, dirs, files in os.walk(masks_dir):
                        for mask_file in files:
                            mask_stem, mask_ext = os.path.splitext(mask_file)
                            inner_stem, inner_ext = os.path.splitext(mask_stem)
                            if inner_ext and inner_ext == mask_ext:
                                os.rename(
                                    os.path.join(root, mask_file),
                                    os.path.join(root, mask_stem)
                                )
                    log.info(f"Masks renamed from COLMAP to NerfStudio convention in: {masks_dir}")
                    # Fill blank (all-white) masks for images that have no mask
                    _filled = 0
                    for _root, _dirs, _files in os.walk(image_path):
                        _rel = os.path.relpath(_root, image_path)
                        for _img_file in _files:
                            if not _img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                continue
                            _mask_rel = os.path.join(_rel, _img_file) if _rel != '.' else _img_file
                            _mask_target = os.path.join(masks_dir, os.path.splitext(_mask_rel)[0] + '.png')
                            if not os.path.isfile(_mask_target):
                                os.makedirs(os.path.dirname(_mask_target), exist_ok=True)
                                _sample = Image.open(os.path.join(_root, _img_file))
                                Image.new('L', _sample.size, 255).save(_mask_target)
                                _sample.close()
                                _filled += 1
                    if _filled:
                        log.info(f"Generated {_filled} blank masks for images without existing masks")
                    # If transforms.json already exists (no conversion needed), inject mask_path now
                    # so gsplat can find the masks without relying on the Colmap-to-Nerfstudio step.
                    transforms_path = os.path.join(config['DATASET_PATH'], 'transforms.json')
                    if has_transforms and os.path.isfile(transforms_path):
                        with open(transforms_path, 'r') as _f:
                            _data = _json.load(_f)
                        _injected = 0
                        for _frame in _data.get('frames', []):
                            # Remove any pre-existing mask_path from external pipeline — we re-validate below
                            _frame.pop('mask_path', None)
                            _img = _frame.get('file_path', '')
                            if _img.startswith('images/'):
                                _img = _img[len('images/'):]
                            elif _img.startswith('./images/'):
                                _img = _img[len('./images/'):]
                            else:
                                _img = os.path.basename(_img)
                            _mask_file = os.path.join(masks_dir, _img)
                            if os.path.isfile(_mask_file):
                                _frame['mask_path'] = f'masks/{_img}'
                                _injected += 1
                        with open(transforms_path, 'w') as _f:
                            _json.dump(_data, _f, indent=4)
                        log.info(f"Injected mask_path into {_injected} frames in existing transforms.json")
                
                # Clean up temp directory
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path)

                # Validate and rescale transforms.json intrinsics to match actual image dimensions.
                # Some pipelines (e.g. SplatKing) write camera intrinsics at capture resolution
                # but store images at a different resolution, causing nerfstudio to crash.
                _transforms_path = os.path.join(config['DATASET_PATH'], 'transforms.json')
                if os.path.isfile(_transforms_path):
                    _img_files = [f for f in os.listdir(image_path)
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if _img_files:
                        with Image.open(os.path.join(image_path, _img_files[0])) as _img:
                            _actual_w, _actual_h = _img.size
                        with open(_transforms_path, 'r') as _f:
                            _tdata = _json.load(_f)
                        _cam_w = _tdata.get('w', _actual_w)
                        _cam_h = _tdata.get('h', _actual_h)
                        if _cam_w != _actual_w or _cam_h != _actual_h:
                            _sx = _actual_w / _cam_w
                            _sy = _actual_h / _cam_h
                            log.info(f"Rescaling transforms.json intrinsics: "
                                     f"{_cam_w}x{_cam_h} -> {_actual_w}x{_actual_h} "
                                     f"(scale {_sx:.4f}, {_sy:.4f})")
                            _tdata['w'] = _actual_w
                            _tdata['h'] = _actual_h
                            for _k in ('fl_x', 'cx'):
                                if _k in _tdata:
                                    _tdata[_k] = _tdata[_k] * (_sx if _k != 'cy' else _sy)
                            if 'fl_y' in _tdata:
                                _tdata['fl_y'] = _tdata['fl_y'] * _sy
                            if 'cy' in _tdata:
                                _tdata['cy'] = _tdata['cy'] * _sy
                            if 'cx' in _tdata:
                                _tdata['cx'] = _tdata['cx'] * _sx
                            for _frame in _tdata.get('frames', []):
                                for _k in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy'):
                                    if _k in _frame:
                                        _s = _sx if _k in ('w', 'fl_x', 'cx') else _sy
                                        _frame[_k] = int(_frame[_k] * _s) if _k in ('w', 'h') else _frame[_k] * _s
                            with open(_transforms_path, 'w') as _f:
                                _json.dump(_tdata, _f, indent=4)
                            log.info(f"Rescaled transforms.json intrinsics to match actual image size")
            else:
                log.warning(f"Zip does not contain valid COLMAP reconstruction structure")
                colmap_zip_found = False
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path)
        
        # Handle directory input - move contents to dataset root if in a subdirectory
        elif os.path.isdir(zip_path):
            for item in os.listdir(zip_path):
                src = os.path.join(zip_path, item)
                dst = os.path.join(config['DATASET_PATH'], item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)
            log.info(f"Moved reconstruction data from {zip_path} to dataset root")
            if os.path.exists(os.path.join(config['DATASET_PATH'], 'masks')):
                ZIP_HAS_MASKS = True
                masks_dir = os.path.join(config['DATASET_PATH'], 'masks')
                for root, dirs, files in os.walk(masks_dir):
                    for mask_file in files:
                        mask_stem, mask_ext = os.path.splitext(mask_file)
                        inner_stem, inner_ext = os.path.splitext(mask_stem)
                        if inner_ext and inner_ext == mask_ext:
                            os.rename(
                                os.path.join(root, mask_file),
                                os.path.join(root, mask_stem)
                            )
                log.info("Masks renamed from COLMAP to NerfStudio convention in directory input")
                _filled = 0
                for _root, _dirs, _files in os.walk(image_path):
                    _rel = os.path.relpath(_root, image_path)
                    for _img_file in _files:
                        if not _img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            continue
                        _mask_rel = os.path.join(_rel, _img_file) if _rel != '.' else _img_file
                        _mask_target = os.path.join(masks_dir, os.path.splitext(_mask_rel)[0] + '.png')
                        if not os.path.isfile(_mask_target):
                            os.makedirs(os.path.dirname(_mask_target), exist_ok=True)
                            _sample = Image.open(os.path.join(_root, _img_file))
                            Image.new('L', _sample.size, 255).save(_mask_target)
                            _sample.close()
                            _filled += 1
                if _filled:
                    log.info(f"Generated {_filled} blank masks for images without existing masks")

        # The named input is missing: fail now instead of training on the empty
        # images/ and sparse/0 placeholders created at startup.
        elif config['FILENAME'].endswith('.zip'):
            _avail = sorted(os.listdir(config['DATASET_PATH']))
            raise RuntimeError(
                pipeline.report_error(
                    706,
                    f"Input zip not found: {zip_path} does not exist "
                    f"(contents of {config['DATASET_PATH']}: {_avail})"))

        # Ensure colmap/sparse structure exists for NerfStudio (not needed for 3DGRUT or depth loss)
        if colmap_zip_found and config['MODEL'] not in ('3dgrt', '3dgut', 'faster-gs') and not IS_GSPLAT_DEPTH:
            colmap_sparse_0 = os.path.join(config['DATASET_PATH'], 'colmap', 'sparse', '0')
            colmap_sparse_0_populated = os.path.isdir(colmap_sparse_0) and bool(os.listdir(colmap_sparse_0))
            if not colmap_sparse_0_populated:
                if os.path.exists(os.path.join(config['DATASET_PATH'], 'sparse')):
                    # Only move sparse/ if it actually has files in its subdirectories
                    _sparse = os.path.join(config['DATASET_PATH'], 'sparse')
                    _sparse_has_files = any(
                        os.listdir(os.path.join(_sparse, d))
                        for d in os.listdir(_sparse)
                        if os.path.isdir(os.path.join(_sparse, d))
                    ) if os.listdir(_sparse) else False
                    if _sparse_has_files:
                        colmap_dir = os.path.join(config['DATASET_PATH'], 'colmap')
                        os.makedirs(colmap_dir, exist_ok=True)
                        _dst = os.path.join(colmap_dir, 'sparse')
                        if os.path.exists(_dst):
                            shutil.rmtree(_dst)
                        shutil.move(_sparse,  _dst)
                        log.info(f"Moved sparse/ to colmap/sparse/ for NerfStudio compatibility")
                    else:
                        log.info("sparse/ is empty placeholder, skipping move to colmap/sparse/")
                # Also check if colmap/sparse/0 exists in the extracted zip content
                # (zip may have colmap/sparse/0 directly without a top-level sparse/)
                _colmap_sparse_src = os.path.join(config['DATASET_PATH'], 'colmap', 'sparse', '0')
                if os.path.isdir(_colmap_sparse_src) and os.listdir(_colmap_sparse_src):
                    log.info(f"Found populated colmap/sparse/0 from zip: {os.listdir(_colmap_sparse_src)[:5]}")
            else:
                log.info(f"colmap/sparse/0 already populated ({len(os.listdir(colmap_sparse_0))} files), skipping sparse/ move")
            
            log.info(f"Successfully processed COLMAP reconstruction from {config['FILENAME']}")
            log.info(f"Dataset contents: {os.listdir(config['DATASET_PATH'])}")
            # Verify colmap/sparse/0 has required files
            sparse_0 = os.path.join(config['DATASET_PATH'], 'colmap', 'sparse', '0')
            if os.path.exists(sparse_0):
                log.info(f"colmap/sparse/0 contents: {os.listdir(sparse_0)}")
            else:
                log.warning(f"colmap/sparse/0 does not exist after extraction")
            # If images/ is empty but colmap/images/ has files, copy them over
            _img_dir = os.path.join(config['DATASET_PATH'], 'images')
            _colmap_img_dir = os.path.join(config['DATASET_PATH'], 'colmap', 'images')
            _img_empty = not os.path.isdir(_img_dir) or not any(
                _f.lower().endswith(('.png', '.jpg', '.jpeg'))
                for _r, _d, _fs in os.walk(_img_dir)
                for _f in _fs
            )
            if _img_empty and os.path.isdir(_colmap_img_dir):
                shutil.copytree(
                    _colmap_img_dir, _img_dir, dirs_exist_ok=True,
                )
                _copied = sum(
                    1 for _ in os.walk(_img_dir)
                    for _f in _[2]
                    if _f.lower().endswith(('.png', '.jpg', '.jpeg'))
                )
                log.info(f"Copied {_copied} images from colmap/images/ to images/")
            # If transforms.json is missing but transforms-in.json exists, use it
            _transforms = os.path.join(config['DATASET_PATH'], 'transforms.json')
            _transforms_in = os.path.join(config['DATASET_PATH'], 'transforms-in.json')
            if not os.path.exists(_transforms) and os.path.exists(_transforms_in):
                shutil.copy2(_transforms_in, _transforms)
                log.info(f"Copied transforms-in.json to transforms.json")

    if model_tar_found:
        log.info(f"Detected model archive: {config['FILENAME']} for resuming training")
        model_tar_path = os.path.join(config['DATASET_PATH'], config['FILENAME'])
        if os.path.exists(model_tar_path):
            log.info(f"Extracting {model_tar_path} to {config['CODE_PATH']}")
            untar_gz(model_tar_path, config['CODE_PATH'])
            
            # Debug: List what was extracted
            log.info(f"Contents of {config['CODE_PATH']} after extraction: {os.listdir(config['CODE_PATH'])}")
            
            # Handle dataset extraction - ensure complete dataset is moved
            dataset_dir = os.path.join(config['CODE_PATH'], 'dataset')
            if os.path.exists(dataset_dir):
                log.info(f"Contents of extracted dataset: {os.listdir(dataset_dir)}")
                # Check if files are under dataset/train/ (new structure)
                train_dir = os.path.join(dataset_dir, 'train')
                if os.path.exists(train_dir):
                    log.info(f"Detected new structure with dataset/train/. Contents: {os.listdir(train_dir)}")
                    # Move contents from dataset/train/ to DATASET_PATH
                    if os.path.exists(config['DATASET_PATH']) and config['DATASET_PATH'] != train_dir:
                        shutil.rmtree(config['DATASET_PATH'])
                    shutil.move(train_dir, config['DATASET_PATH'])
                    log.info(f"Moved dataset from {train_dir} to {config['DATASET_PATH']}")
                    # Clean up empty dataset directory
                    if os.path.exists(dataset_dir):
                        shutil.rmtree(dataset_dir)
                else:
                    # Old structure: dataset/ contains files directly
                    log.info(f"Detected old structure with dataset/ containing files directly")
                    if os.path.exists(config['DATASET_PATH']) and config['DATASET_PATH'] != dataset_dir:
                        shutil.rmtree(config['DATASET_PATH'])
                    shutil.move(dataset_dir, config['DATASET_PATH'])
                    log.info(f"Moved entire dataset from {dataset_dir} to {config['DATASET_PATH']}")
            
            # Move model directory and config.yml to proper output directory structure for resume training
            model_dir_name = "nerfstudio_models"
            if config['MODEL'] == "3dgrt" or config['MODEL'] == "3dgut":
                model_dir_name = "3dgrut_models"
            
            # Check both old and new locations for model files
            model_src_dir = os.path.join(config['DATASET_PATH'], model_dir_name)
            config_yml_src = os.path.join(config['DATASET_PATH'], 'config.yml')
            
            log.info(f"Looking for model files at: {model_src_dir}")
            log.info(f"Model directory exists: {os.path.exists(model_src_dir)}")
            if os.path.exists(config['DATASET_PATH']):
                log.info(f"DATASET_PATH contents: {os.listdir(config['DATASET_PATH'])}")
            
            if os.path.exists(model_src_dir):
                # Ensure the output directory structure exists
                os.makedirs(os.path.dirname(model_ckpt_path), exist_ok=True)

                # For 3dgrut models, we need to find the actual checkpoint file, not use the directory
                if config['MODEL'] == "3dgrt" or config['MODEL'] == "3dgut":
                    # Look for ckpt_last.pt first, then fall back to .ckpt files
                    ckpt_last_file = os.path.join(model_src_dir, "ckpt_last.pt")
                    if os.path.exists(ckpt_last_file):
                        # Use ckpt_last.pt
                        dst_ckpt_path = os.path.join(model_ckpt_path, "ckpt_last.pt")
                        os.makedirs(model_ckpt_path, exist_ok=True)
                        shutil.copy2(ckpt_last_file, dst_ckpt_path)
                        model_ckpt_path = dst_ckpt_path
                        log.info(f"Moved 3dgrut checkpoint from {ckpt_last_file} to {dst_ckpt_path}")
                    else:
                        # Find other checkpoint files
                        ckpt_files = [f for f in os.listdir(model_src_dir) if f.endswith('.ckpt')]
                        if ckpt_files:
                            # Use the first checkpoint file found
                            ckpt_file = ckpt_files[0]
                            src_ckpt_path = os.path.join(model_src_dir, ckpt_file)
                            dst_ckpt_path = os.path.join(model_ckpt_path, ckpt_file)
                            
                            # Create the destination directory and copy the checkpoint file
                            os.makedirs(model_ckpt_path, exist_ok=True)
                            shutil.copy2(src_ckpt_path, dst_ckpt_path)
                            
                            # Update model_ckpt_path to point to the actual checkpoint file for 3dgrut
                            model_ckpt_path = dst_ckpt_path
                            log.info(f"Moved 3dgrut checkpoint from {src_ckpt_path} to {dst_ckpt_path}")
                        else:
                            log.warning(f"No checkpoint files found in {model_src_dir}")
                            # Move the entire directory as fallback
                            shutil.move(model_src_dir, model_ckpt_path)
                            log.info(f"Moved 3dgrut_models directory from {model_src_dir} to {model_ckpt_path}")
                else:
                    # For splatfacto models, keep files in dataset directory
                    if config['MODEL'] in ["splatfacto", "splatfacto-big", "splatfacto-mcmc"]:
                        log.info(f"Keeping nerfstudio_models in dataset directory for splatfacto: {model_src_dir}")
                        # Don't move - files are already in correct location
                    else:
                        # For other models, move to code directory
                        shutil.move(model_src_dir, model_ckpt_path)
                        log.info(f"Moved nerfstudio_models from {model_src_dir} to {model_ckpt_path}")
            if os.path.exists(config_yml_src): # only for Nerfstudio
                # For splatfacto models, keep config.yml in dataset directory and update it there
                if config['MODEL'] in ["splatfacto", "splatfacto-big", "splatfacto-mcmc"]:
                    log.info(f"Keeping nerfstudio_models and config.yml in dataset directory for splatfacto resume: {model_src_dir}")
                    ckpt_files = sorted([f for f in os.listdir(model_src_dir) if f.endswith('.ckpt')])
                    log.info(f"Splatfacto resume checkpoint files found at: {ckpt_files}")
                else:
                    # For other models, move to code directory
                    os.makedirs(os.path.dirname(model_config_path), exist_ok=True)
                    shutil.move(config_yml_src, model_config_path)
                    log.info(f"Moved config.yml from {config_yml_src} to {model_config_path}")
            
            log.info("Dataset path after moving: ")
            if os.path.exists(config['DATASET_PATH']):
                log.info(", ".join(os.listdir(config['DATASET_PATH'])))
                # Check for nested directories that might contain the model files
                for item in os.listdir(config['DATASET_PATH']):
                    item_path = os.path.join(config['DATASET_PATH'], item)
                    if os.path.isdir(item_path):
                        log.info(f"Contents of {item}/: {os.listdir(item_path)}")
            else:
                log.error(f"DATASET_PATH does not exist: {config['DATASET_PATH']}")
            

            # Log checkpoint and config locations for debugging
            dataset_models_path = os.path.join(config['DATASET_PATH'], "nerfstudio_models")
            dataset_config_path = os.path.join(config['DATASET_PATH'], "config.yml")
            log.info(f"Checkpoint directory exists: {os.path.exists(dataset_models_path)}")
            log.info(f"Config file exists: {os.path.exists(dataset_config_path)}")
            if os.path.exists(dataset_models_path):
                ckpt_files = [f for f in os.listdir(dataset_models_path) if f.endswith('.ckpt')]
                log.info(f"Checkpoint files: {ckpt_files}")
                if ckpt_files:
                    log.info(f"Latest checkpoint: {sorted(ckpt_files)[-1]}")
            
            # Verify images directory exists and has content
            if os.path.exists(image_path):
                image_files = os.listdir(image_path)
                log.info(f"Found {len(image_files)} files in images directory")
                if image_files:
                    log.info(f"Sample image files: {image_files[:5]}")
            else:
                log.warning(f"Images directory not found at {image_path}")
            
            # Find original media file for proper output naming
            media_extensions = ('.mov', '.mp4', '.zip')
            original_filename_found = False
            for file in os.listdir(config['DATASET_PATH']):
                if file.lower().endswith(media_extensions):
                    config['FILENAME'] = file
                    log.info(f"Found original media file: {file}")
                    original_filename_found = True
                    break
            
            # If no media file found, keep the archive name but warn user
            if not original_filename_found:
                log.warning(f"No original media file found in dataset. Using archive name for output: {config['FILENAME']}")

            # Ensure we remove previous exports (but preserve in local debug)
            if not LOCAL_DEBUG and os.path.exists(output_path):
                shutil.rmtree(output_path)
            if not os.path.isdir(output_path):
                os.makedirs(output_path, exist_ok=True)
            log.info(f"Successfully extracted and organized model archive for resume training")
        
        # Final verification for splatfacto models
        if config['MODEL'] in ["splatfacto", "splatfacto-big", "splatfacto-mcmc"]:
            dataset_models_path = os.path.join(config['DATASET_PATH'], "nerfstudio_models")
            dataset_config_path = os.path.join(config['DATASET_PATH'], "config.yml")
            log.info(f"Final check - Checkpoint dir: {os.path.exists(dataset_models_path)}, Config: {os.path.exists(dataset_config_path)}")
            if os.path.exists(dataset_models_path):
                ckpt_files = [f for f in os.listdir(dataset_models_path) if f.endswith('.ckpt')]
                log.info(f"Available checkpoints: {ckpt_files}")
        
        # Verify checkpoint files and config are in correct location
        if config['MODEL'] in ["splatfacto", "splatfacto-big", "splatfacto-mcmc"]:
            dataset_models_path = os.path.join(config['DATASET_PATH'], "nerfstudio_models")
            dataset_config_path = os.path.join(config['DATASET_PATH'], "config.yml")
            if os.path.exists(dataset_models_path):
                log.info(f"Splatfacto checkpoint files found at: {dataset_models_path}")
            else:
                log.warning(f"Splatfacto checkpoint files not found at expected location: {dataset_models_path}")
            if os.path.exists(dataset_config_path):
                log.info(f"Splatfacto config file found at: {dataset_config_path}")
                # Log first few lines of config for debugging
                try:
                    with open(dataset_config_path, 'r') as f:
                        first_lines = ''.join(f.readlines()[:5])
                    log.info(f"Config file preview: {first_lines[:200]}...")
                except Exception as e:
                    log.warning(f"Could not read config file: {e}")
            else:
                log.warning(f"Splatfacto config file not found at expected location: {dataset_config_path}")
        else:
            if os.path.exists(model_ckpt_path):
                log.info(f"Checkpoint files found at: {model_ckpt_path}")
            else:
                log.warning(f"Checkpoint files not found at expected location: {model_ckpt_path}")
            if os.path.exists(model_config_path):
                log.info(f"Config file found at: {model_config_path}")
            else:
                log.warning(f"Config file not found at expected location: {model_config_path}")

    ##################################
    # CONFIGURE MULTI-GPU DISTRIBUTED TRAINING
    ##################################
    if int(pipeline.config.num_gpus) > 1 and IS_FASTER_GS:
        log.info(f"{pipeline.config.num_gpus} GPUs detected, but Faster-GS trains on a single GPU (GPU 0); multi-GPU disabled")
    elif int(pipeline.config.num_gpus) > 1:
        ENABLE_MULTI_GPU = "true"
        #os.environ['MAX_JOBS'] = '4'
        log.info(f"Multi-GPU enabled: {pipeline.config.num_gpus} GPUs (gsplat distributed training)")
        # Read SageMaker resource config for multi-container setup
        resource_config_path = '/opt/ml/input/config/resourceconfig.json'
        if os.path.exists(resource_config_path):
            with open(resource_config_path, 'r') as f:
                resource_config = json.load(f)
            
            hosts = resource_config.get('hosts', ['local-host'])
            current_host = resource_config.get('current_host', 'localhost')
            network_interface = resource_config.get('network_interface_name', 'eth0')
            
            log.info(f"DEBUG: Resource config - hosts: {hosts}, current: {current_host}, interface: {network_interface}")
            
            # Check if this is single-instance multi-GPU (only one host) or multi-container
            if len(hosts) == 1:
                # Single instance multi-GPU - use localhost
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['WORLD_SIZE'] = '1'
                os.environ['RANK'] = '0'
                os.environ['LOCAL_RANK'] = '0'
                log.info(f"""DEBUG: Single instance multi-GPU -
                         MASTER_ADDR={os.environ['MASTER_ADDR']},
                         MASTER_PORT={os.environ['MASTER_PORT']},
                         WORLD_SIZE={os.environ['WORLD_SIZE']}
                         """)
            else:
                # Multi-container setup - use first host
                os.environ['MASTER_ADDR'] = hosts[0]
                os.environ['MASTER_PORT'] = '29500'
                os.environ['WORLD_SIZE'] = str(len(hosts))
                os.environ['RANK'] = str(hosts.index(current_host))
                os.environ['LOCAL_RANK'] = '0'
                log.info(f"""DEBUG: Multi-container setup -
                         MASTER_ADDR={os.environ['MASTER_ADDR']},
                         MASTER_PORT={os.environ['MASTER_PORT']},
                         WORLD_SIZE={os.environ['WORLD_SIZE']},
                         RANK={os.environ['RANK']}
                         """)
        else:
            # Single instance multi-GPU setup (no resource config)
            # Set MASTER_ADDR/PORT since Dockerfile commented them out in gsplat/distributed.py
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'
            log.info(f"DEBUG: Single instance multi-GPU - set MASTER_ADDR=127.0.0.1, MASTER_PORT=29500 for {pipeline.config.num_gpus} GPUs")
    else:
        log.info("Single GPU setup, no distributed training configuration needed")

    ##################################
    # GPS POSE PRIORS (detection):
    # Images with GPS (EXIF, or a priors.csv in the zip) reconstructed with the
    # spatial matcher and no other pose priors use COLMAP pose_prior_mapper, which
    # keeps the camera positions close to the GPS positions (metric, ENU frame).
    # GLOMAP/hierarchical mappers ignore GPS, so COLMAP is used for these jobs.
    ##################################
    GPS_PRIORS = False
    gps_priors_json = os.path.join(config['DATASET_PATH'], "gps_priors.json")
    gps_priors_script = os.path.join(current_dir_path, 'pre_processing', 'gps_priors.py')
    try:
        if config['RUN_RECON'] == 'true' and config['MATCHING_METHOD'] == 'spatial' and \
                config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'false' and \
                config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'false' and \
                config['SPHERICAL_CAMERA'] != 'true' and VIDEO is False and \
                not colmap_zip_found and not model_tar_found and os.path.exists(input_file_path):
            _probe = subprocess.run(
                [sys.executable, gps_priors_script, 'probe', '--input', input_file_path],
                capture_output=True, text=True, timeout=600
            )
            _probe_info = None
            for _line in _probe.stdout.splitlines():
                if _line.startswith('GPS_PROBE '):
                    _probe_info = json.loads(_line[len('GPS_PROBE '):])
            if _probe_info is None:
                log.warning(f"GPS probe produced no result: {_probe.stderr.strip()[-500:]}")
            else:
                log.info(f"GPS probe: {_probe_info}")
                if _probe_info['has_priors_csv'] or (
                        _probe_info['sampled'] and _probe_info['with_gps'] >= 0.5 * _probe_info['sampled']):
                    GPS_PRIORS = True
                    if config['RECON_SOFTWARE_NAME'] != 'colmap':
                        log.info(f"GPS pose priors: using COLMAP pose_prior_mapper instead of "
                                 f"'{config['RECON_SOFTWARE_NAME']}' (it ignores GPS)")
                        config['RECON_SOFTWARE_NAME'] = 'colmap'
                    if config.get('AUTO_MAPPER', 'false') == 'true':
                        log.info("GPS pose priors: AUTO_MAPPER disabled (pose_prior_mapper is required)")
                        config['AUTO_MAPPER'] = 'false'
                    if config.get('AUTO_MATCHER', 'false') == 'true':
                        log.info("GPS pose priors: AUTO_MATCHER disabled (keeping the spatial matcher)")
                        config['AUTO_MATCHER'] = 'false'
                    log.info("GPS pose priors enabled: spatial matcher + COLMAP pose_prior_mapper")
                else:
                    log.info("Spatial matching without enough GPS in the images; using the standard mapper")
    except Exception as e:
        log.warning(f"GPS probe failed ({e}); using the standard mapper")
        GPS_PRIORS = False

    # Masks uploaded with a plain image zip (masks/<image>.png, white = keep):
    # route them to COLMAP feature extraction and to training.
    # A rig_config.json (COLMAP rig layout: images/<camera>/<frame>.jpg, same frame
    # name in every camera folder) makes COLMAP solve one pose per frame with the
    # cameras' relative poses taken from the config (see the rig_configurator step).
    UPLOADED_IMAGE_MASKS = False
    RIG_CONFIG = False
    RIG_NUM_CAMERAS = 1
    RIG_HAS_INTRINSICS = False  # every rig camera lists camera_model_name + camera_params
    rig_config_path = os.path.join(config['DATASET_PATH'], "rig_config.json")
    try:
        if config['RUN_RECON'] == 'true' and not colmap_zip_found and not model_tar_found and \
                config['SPHERICAL_CAMERA'] != 'true' and VIDEO is False and \
                ((input_file_path.lower().endswith('.zip') and os.path.isfile(input_file_path)) or
                 os.path.isdir(input_file_path)):
            # Same detection for an uploaded zip and an unzipped folder (local runs).
            if os.path.isdir(input_file_path):
                _entries = [os.path.relpath(os.path.join(_r, _f), input_file_path).replace('\\', '/')
                            for _r, _, _fs in os.walk(input_file_path) for _f in _fs]
                _zf = None

                def _read_entry(_n):
                    with open(os.path.join(input_file_path, _n), 'rb') as _fh:
                        return _fh.read()
            else:
                _zf = zipfile.ZipFile(input_file_path)
                _entries = _zf.namelist()
                _read_entry = _zf.read
            try:
                for _n in _entries:
                    _parts = _n.replace('\\', '/').split('/')
                    if _parts[-1].lower().endswith(('.png', '.jpg', '.jpeg')) and \
                            any(_p.lower() in ('mask', 'masks') for _p in _parts[:-1]):
                        UPLOADED_IMAGE_MASKS = True
                    if _parts[-1].lower() == 'rig_config.json':
                        RIG_CONFIG = True
                        try:
                            _rc = json.loads(_read_entry(_n).decode('utf-8'))
                            RIG_NUM_CAMERAS = max(1, max(len(_r.get('cameras', [])) for _r in _rc))
                            _rig_cams = [_c for _r in _rc for _c in _r.get('cameras', [])]
                            RIG_HAS_INTRINSICS = bool(_rig_cams) and all(
                                _c.get('camera_model_name') and _c.get('camera_params') for _c in _rig_cams)
                        except Exception as _rc_err:
                            log.warning(f"Could not read rig_config.json ({_rc_err})")
            finally:
                if _zf is not None:
                    _zf.close()
            if UPLOADED_IMAGE_MASKS:
                ZIP_HAS_MASKS = True
                log.info("Image zip contains masks/: using them for COLMAP features and training")
            if RIG_CONFIG:
                if config['RECON_SOFTWARE_NAME'] not in ('colmap', 'glomap'):
                    log.info(f"Camera rig: using COLMAP instead of '{config['RECON_SOFTWARE_NAME']}'")
                    config['RECON_SOFTWARE_NAME'] = 'colmap'
                if config.get('AUTO_MAPPER', 'false') == 'true':
                    log.info("Camera rig: AUTO_MAPPER disabled")
                    config['AUTO_MAPPER'] = 'false'
                log.info("Image zip contains rig_config.json: camera sub-folders are kept and "
                         "COLMAP solves one pose per rig frame")
    except Exception as e:
        log.warning(f"Could not inspect the zip for masks/rig config ({e})")

    ##################################
    # PRE-PROCESS COMPONENT:
    # Pose Transform for Reconstruction
    #################################
    try:
        if config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true' and config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true' and \
            config['RUN_RECON'] == 'true':
            raise RuntimeError(
                pipeline.report_error(
                    705,
                    f"""Configuration not supported.
                    Only pose prior transform json or pose prior colmap model files can be enabled, not both."""
                )
            )
        if (config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true' or config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true') and \
            config['RUN_RECON'] == 'true':
            if VIDEO is False and input_filename_extension.lower() == ".zip":
                if config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true':
                    use_transforms = "true"
                else:
                    use_transforms = "false"

                args = [
                    "-i", input_file_path,
                    "-t", use_transforms
                ]
                pipeline.create_component(
                    name="ExtractPosesImgs",
                    comp_type=ComponentType.PRE_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="reconstruction/extract_poses_imgs.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
            else:
                raise RuntimeError(
                    pipeline.report_error(
                        710,
                        f"""Improper file type {input_filename_extension} given for prior pose transformations.
                        Only '.zip' is supported."""
                    )
                )
    except Exception as e:
        error_message = f"Issue transforming pose to colmap component: {e}"
        pipeline.report_error(715, error_message)

    ##################################
    # PRE-PROCESS COMPONENT:
    # Video to Images
    ##################################
    try:
        if VIDEO is True and config['REMOVE_BACKGROUND'] == "true" and config['BACKGROUND_REMOVAL_MODEL'] == "sam2" and \
            config['RUN_RECON'] == 'true':
                # SAM2 BACKGROUND REMOVAL COMPONENT
                args = [
                    "-i", input_file_path,
                    "-o", image_path,
                    "-n", config['MAX_NUM_IMAGES'],
                    "-mt", config['MASK_THRESHOLD']
                ]
                pipeline.create_component(
                    name="RemoveBackground",
                    comp_type=ComponentType.PRE_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="sam/remove_background_sam2.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
        elif VIDEO is False and config['BACKGROUND_REMOVAL_MODEL'] == "sam2" and config['REMOVE_BACKGROUND']=="true" and \
            config['RUN_RECON'] == 'true':
            sys.exit("Error: SAM2 Background removal is only supported for video input")
        else:
            # Video frame selection (both settings can be combined):
            #   ADAPTIVE_FRAME_EXTRACTION  FILTER_BLURRY_IMAGES  frames
            #   false                      false                 regular intervals (simple)
            #   false                      true                  sharpest per regular block
            #   true                       false                 motion-adaptive keyframes
            #   true                       true                  sharpest frame per adaptive block
            # MAX_NUM_IMAGES caps the count in every mode.
            if ADAPTIVE_FRAME_EXTRACTION and config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'false' and \
               config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'false' and config['RUN_RECON'] == 'true':
                args = [
                    "-i", input_file_path,
                    "-o", image_path,
                    "-n", str(int(config['MAX_NUM_IMAGES'])),
                    "--density", ADAPTIVE_FRAME_DENSITY,
                    "--work-dir", os.path.join(config['DATASET_PATH'], "adaptive_frames"),
                    "-ll", config['LOG_VERBOSITY'].upper(),
                    "-s", config['VIDEO_START_TIME']
                ]
                if config['FILTER_BLURRY_IMAGES'] == "true":
                    args.append("--sharp")
                video_stop_time = str(config['VIDEO_STOP_TIME']).strip() if config['VIDEO_STOP_TIME'] is not None else ""
                if video_stop_time and video_stop_time.lower() not in ['none', 'null', '', 'nan', '-1']:
                    try:
                        if float(video_stop_time) > 0:
                            args.extend(["-e", video_stop_time])
                    except ValueError:
                        pass
                log.info(f"Video frames: adaptive keyframes (density {ADAPTIVE_FRAME_DENSITY}"
                         f"{', sharpest frame per block' if config['FILTER_BLURRY_IMAGES'] == 'true' else ''})")
                pipeline.create_component(
                    name="VideoToImages",
                    comp_type=ComponentType.PRE_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="pre_processing/video/adaptive_video_to_images.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
            # Use sharp-frame-extractor if blur filtering is enabled, otherwise use simple extraction
            elif config['FILTER_BLURRY_IMAGES'] == "true" and config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'false' and \
               config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'false':
                # Add 8% to max images to compensate for blur filtering reduction
                adjusted_max_images = str(int(int(config['MAX_NUM_IMAGES']) * 1.08))
                args = [
                    "-i", input_file_path,
                    "-o", image_path,
                    "-n", adjusted_max_images,
                    "-ll", config['LOG_VERBOSITY'].upper(),
                    "-s", config['VIDEO_START_TIME']
                ]
                video_stop_time = str(config['VIDEO_STOP_TIME']).strip() if config['VIDEO_STOP_TIME'] is not None else ""
                if video_stop_time and video_stop_time.lower() not in ['none', 'null', '', 'nan', '-1']:
                    try:
                        if float(video_stop_time) > 0:
                            args.extend(["-e", video_stop_time])
                    except ValueError:
                        pass
                pipeline.create_component(
                    name="VideoToImages",
                    comp_type=ComponentType.PRE_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="pre_processing/video/sharp_video_to_images.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
            elif config['FILTER_BLURRY_IMAGES'] == "false" and config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'false' and \
               config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'false'and config['RUN_RECON'] == 'true':
                # Standard extraction without blur filtering
                num_imgs = int(config['MAX_NUM_IMAGES'])
                args = [
                    "-i", input_file_path,
                    "-o", image_path,
                    "-n", str(num_imgs),
                    "-nw", str(pipeline.config.num_threads),
                    "-ll", config['LOG_VERBOSITY'].upper(),
                    "-st", config['VIDEO_START_TIME']
                ]
                
                video_stop_time = str(config['VIDEO_STOP_TIME']).strip() if config['VIDEO_STOP_TIME'] is not None else ""
                if video_stop_time and video_stop_time.lower() not in ['none', 'null', '', 'nan', '-1']:
                    try:
                        if float(video_stop_time) > 0:
                            args.extend(["-et", video_stop_time])
                    except ValueError:
                        pass
                
                pipeline.create_component(
                    name="VideoToImages",
                    comp_type=ComponentType.PRE_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="pre_processing/video/simple_video_to_images.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
    except Exception as e:
        error_message = f"Issue creating video to images component: {e}"
        pipeline.report_error(720, error_message)

    ##################################
    # PRE-PROCESS COMPONENT:
    # Remove blurry images from image inputs (zip/folder of images, pose-prior zip,
    # COLMAP model zip). Video input is already filtered by sharp-frame-extractor.
    # Runs right after the images are extracted, and also removes the dropped
    # images from COLMAP models, COLMAP databases, transforms.json, masks and depth.
    ##################################
    try:
        if config['FILTER_BLURRY_IMAGES'] == "true" and VIDEO is False and not model_tar_found:
            pipeline.create_component(
                name="Filter-Blurry-Images",
                comp_type=ComponentType.PRE_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="pre_processing/blur_filter.py",
                args=["--dataset", config['DATASET_PATH'], "--images-dir", image_path,
                      "--database", colmap_db_path],
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue creating blurry image filter component: {e}"
        pipeline.report_error(721, error_message)

    ##################################
    # PRE-PROCESS COMPONENT:
        # Autogroup Images by Prefix
    ##################################
    try:
        if config.get('AUTOGROUP_IMAGES', 'false') == 'true' and config['RUN_RECON'] == 'true' and \
                config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'false' and \
                config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'false':
            log.info(f"Creating AutogroupImages component: target_name={config.get('AUTOGROUP_TARGET_NAME', '')}")
            args = [
                "-i", image_path,
                "-t", config.get('AUTOGROUP_TARGET_NAME', '')
            ]
            pipeline.create_component(
                name="AutogroupImages",
                comp_type=ComponentType.PRE_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="pre_processing/autogroup_images.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue creating autogroup images component: {e}"
        pipeline.report_error(723, error_message)

    ##################################
    # PRE-PROCESS COMPONENT:
    # Autoscale Dataset Resolution/Count
    ##################################
    try:
        if config.get('AUTOSCALE_DATASET', 'false') == 'true' and config['RUN_RECON'] == 'true':
            autoscale_mode = config.get('AUTOSCALE_DATASET_MODE', 'resize').upper()
            log.info(f"Creating AutoscaleDataset component: mode={autoscale_mode}")
            args = [
                "-i", image_path,
                "-m", autoscale_mode
            ]
            pipeline.create_component(
                name="AutoscaleDataset",
                comp_type=ComponentType.PRE_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="pre_processing/autoscale_dataset.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue creating autoscale dataset component: {e}"
        pipeline.report_error(722, error_message)

    ##################################
    # PRE-PROCESS COMPONENT:
    # Rotate Portrait Images
    ##################################
    try:
        if config['RUN_RECON'] == 'true' and (config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'false' and \
           config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'false'):
            log.info(f"Creating RotatePortraitImages component for image_path: {image_path}")
            args = [
                "-i", image_path,
                "-d", config['DATASET_PATH']
            ]
            pipeline.create_component(
                name="RotatePortraitImages",
                comp_type=ComponentType.PRE_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="pre_processing/video/rotate_portrait_images.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
            log.info("RotatePortraitImages component created successfully")
    except Exception as e:
        error_message = f"Issue creating rotate portrait images component: {e}"
        pipeline.report_error(721, error_message)

    ##################################
    # PRE-PROCESS COMPONENT:
    # Remove Background
    ##################################
    try:
        if config['REMOVE_BACKGROUND'] == "true" and config['BACKGROUND_REMOVAL_MODEL'] != "sam2" and \
            config['RUN_RECON'] == "true":
            bg_removal_model = "u2net"

            args = [
                "-i", image_path,
                "-o", image_path,
                "-nt", str(pipeline.config.num_threads),
                "-ng", str(pipeline.config.num_gpus),
                "-m", bg_removal_model
            ]

            pipeline.create_component(
                name="RemoveBackground",
                comp_type=ComponentType.PRE_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="pre_processing/segmentation/remove_background.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=True
            )
    except Exception as e:
        error_message = f"Issue creating background removal component: {e}"
        pipeline.report_error(730, error_message)

    ##################################
    # PRE-PROCESS COMPONENT:
    # Spherical Image Processing
    ##################################
    try:
        if config['SPHERICAL_CAMERA'] == "true" and config['RUN_RECON'] == "true":
            if config['MATCHING_METHOD'] == "vocab":
                method = "vocabtree"
            else:
                method = config['MATCHING_METHOD']
            faces_to_remove = config['SPHERICAL_CUBE_FACES_TO_REMOVE'].strip()
            # Compute SIFT feature count scaled to input image resolution
            REF_MAX_IMAGE_SIZE = 3200
            REF_MAX_NUM_FEATURES = 8192
            sample_images = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            _max_dim = REF_MAX_IMAGE_SIZE
            if sample_images:
                with Image.open(os.path.join(image_path, sample_images[0])) as _img:
                    _max_dim = max(_img.width, _img.height)
                sift_max_num_features = min(int(REF_MAX_NUM_FEATURES * (_max_dim / REF_MAX_IMAGE_SIZE) ** 2), 65536)
            else:
                sift_max_num_features = REF_MAX_NUM_FEATURES
            log.info(f"Spherical SIFT: max_num_features={sift_max_num_features}, enhanced={config['ENABLE_ENHANCED_FEATURE_EXTRACTION']}")
            args = [
                "--input_image_path", image_path,
                "--output_path", config['DATASET_PATH'],
                "--matcher", method,
                "--max_num_features", str(sift_max_num_features),
            ]
            if config['ENABLE_ENHANCED_FEATURE_EXTRACTION'] == "true":
                args.append("--enhanced_feature_extraction")
            if faces_to_remove and faces_to_remove != '[]':
                try:
                    faces_list = ast.literal_eval(faces_to_remove)
                    exclude_str = ",".join(f.strip().lower() for f in faces_list)
                except (ValueError, SyntaxError):
                    exclude_str = faces_to_remove.strip("[]").replace("'", "").replace('"', "")
                args.extend(["--exclude_faces", exclude_str])
                log.info(f"Spherical: excluding cubemap faces: {exclude_str}")

            if config['REMOVE_OBJECT'] == "true":
                # "person": Grounding DINO + SAM 2.1 (person_segmentation.py)
                bg_removal_model = "person"
                args.extend(["--remove_object",
                             "--object_action", config['OBJECT_REMOVAL_ACTION'],
                             "-m", bg_removal_model,
                             "-nt", str(pipeline.config.num_threads),
                             "-ng", str(pipeline.config.num_gpus),
                             "-gpu", str(USE_GPU)
                             ])
            
            pipeline.create_component(
                name="PanoramaSfM",
                comp_type=ComponentType.RECONSTRUCTION,
                comp_environ=ComponentEnvironment.PYTHON,
                command="spherical/panorama_sfm.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue creating spherical image component: {e}"
        pipeline.report_error(735, error_message)

    ##################################
    # PRE-PROCESS COMPONENT:
    # Remove Objects
    ##################################
    PERSON_MASKS = False  # True when the removed people are written to masks/
    try:
        # Skip object removal if using spherical camera (handled in panorama_sfm.py).
        # Camera-rig uploads (images/<camera>/<frame>.jpg) use remove_object_rig.py,
        # which runs the same segmentation + remove/erase per camera folder and keeps
        # every file's folder, name and size so the rig layout still matches.
        if config['REMOVE_OBJECT'] == "true" and config['RUN_RECON'] == "true" and \
                config['SPHERICAL_CAMERA'] != "true":
            bg_removal_model = None
            # OBJECT REMOVAL COMPONENT FOR HUMAN
            try:
                objects_list = ast.literal_eval(config['OBJECT_REMOVAL_OBJECTS'])
                if "human" in [obj.lower() for obj in objects_list]:
                    # Grounding DINO + SAM 2.1 person masks (person_segmentation.py)
                    bg_removal_model = "person"
            except (ValueError, SyntaxError):
                # Fallback to string check if parsing fails
                if "human" in config['OBJECT_REMOVAL_OBJECTS'].lower():
                    bg_removal_model = "person"
            if bg_removal_model is not None and RIG_CONFIG:
                _rig_action = "remove" if config['OBJECT_REMOVAL_ACTION'] == "remove" else "erase"
                log.info(f"Camera rig: object removal ({_rig_action}) runs per camera folder")
                args = [
                    "--images-dir", image_path,
                    "--work-dir", os.path.join(config['DATASET_PATH'], "object_removal_rig"),
                    "--action", _rig_action,
                    "-m", bg_removal_model,
                    "-nt", str(pipeline.config.num_threads),
                    "-ng", str(pipeline.config.num_gpus),
                    "-mp", os.path.join(config['DATASET_PATH'], "stable-diffusion-xl-base-1.0"),
                    "-pp", os.path.join(config['CODE_PATH'], "AttentiveEraser", "pipelines",
                                        "pipeline_stable_diffusion_xl_attentive_eraser.py"),
                    "-gpu", str(USE_GPU),
                    "-log", config['LOG_VERBOSITY'],
                    "-method", "SIP",
                    # Uploaded rig masks: the detector must not see the rig (it can be
                    # segmented as a person), so already-masked pixels are left out.
                    "--masks-dir", os.path.join(config['DATASET_PATH'], "masks")
                ]
                pipeline.create_component(
                    name="RemoveObject-Rig",
                    comp_type=ComponentType.PRE_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="pre_processing/segmentation/remove_object_rig.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
                # Masks of the frames that were changed: <camera>/<stem>.png, white = person
                _person_dir = os.path.join(config['DATASET_PATH'], "object_removal_rig", "person_masks")
            elif bg_removal_model is not None:
                args = [
                    "-i", image_path,
                    "-o", filter_output_dir,
                    "-nt", str(pipeline.config.num_threads),
                    "-ng", str(pipeline.config.num_gpus),
                    "-m", bg_removal_model
                ]
                pipeline.create_component(
                    name="RemoveObject",
                    comp_type=ComponentType.PRE_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="pre_processing/segmentation/remove_background.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
                if config['OBJECT_REMOVAL_ACTION'] == "remove":
                    args = [
                        "-oi", image_path,
                        "-om", filter_output_dir,
                        "-od", mask_human_output_dir
                    ]
                    pipeline.create_component(
                        name="RemoveHumanSubjectMask",
                        comp_type=ComponentType.PRE_PROCESSING,
                        comp_environ=ComponentEnvironment.PYTHON,
                        command="pre_processing/segmentation/remove_object_using_mask.py",
                        args=args,
                        cwd=current_dir_path,
                        requires_gpu=False
                    )
                else: # eraser
                    args = [
                        "-id", image_path,
                        "-md", filter_output_dir,
                        "-mp", os.path.join(config['DATASET_PATH'], "stable-diffusion-xl-base-1.0"),
                        "-pp", os.path.join(config['CODE_PATH'], "AttentiveEraser", "pipelines", "pipeline_stable_diffusion_xl_attentive_eraser.py"),
                        "-gpu", USE_GPU,
                        "-log", config['LOG_VERBOSITY'],
                        "-method", "SIP" # or DIP
                    ]
                    pipeline.create_component(
                        name="EraseObject",
                        comp_type=ComponentType.PRE_PROCESSING,
                        comp_environ=ComponentEnvironment.PYTHON,
                        command="pre_processing/segmentation/erase_object_using_mask.py",
                        args=args,
                        cwd=current_dir_path,
                        requires_gpu=True
                    )
                # remove_background output: <stem>.png RGBA, alpha = person
                _person_dir = filter_output_dir
            if bg_removal_model is not None and config['OBJECT_REMOVAL_ACTION'] == "remove":
                # OBJECT_REMOVAL_ACTION=remove masks the people out: they are added to
                # the training masks (masks/, white = keep), so COLMAP extracts no
                # features there and the trainers ignore those pixels. With "erase" the
                # inpainted pixels are trained on, so no masks are written. Runs before
                # the sky / lens flare steps, which AND their masks with these.
                pipeline.create_component(
                    name="Person-Masks",
                    comp_type=ComponentType.PRE_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="pre_processing/segmentation/write_person_masks.py",
                    args=["--images-dir", image_path,
                          "--person-dir", _person_dir,
                          "--masks-dir", os.path.join(config['DATASET_PATH'], "masks"),
                          "--debug-dir", os.path.join(config['DATASET_PATH'], "mask_previews", "person")],
                    cwd=current_dir_path,
                    requires_gpu=False
                )
                PERSON_MASKS = True
                ZIP_HAS_MASKS = True
    except Exception as e:
        error_message = f"Issue creating human subject removal component: {e}"
        pipeline.report_error(740, error_message)

    ##################################
    # PRE_PROCESSING COMPONENTS:
    # Mask the sky, then lens flare, so COLMAP and training ignore those pixels.
    # Sky runs first so the flare detector only analyses the pixels that are left
    # (bright clouds are not flare). Created after the image-editing steps (video
    # extraction, autoscale, rotation, background/human removal) so the masks match
    # the final images. Masks are written to masks/ (white = keep) and AND-ed with
    # any uploaded masks.
    ##################################
    GENERATED_MASKS = PERSON_MASKS  # person masks from object removal (above)
    try:
        if (REMOVE_LENS_FLARE or REMOVE_SKY) and config['SPHERICAL_CAMERA'] == "true":
            log.warning("REMOVE_LENS_FLARE/REMOVE_SKY are not supported with SPHERICAL_CAMERA=true "
                        "(the panorama step writes its own cube-face masks); skipping")
        elif REMOVE_LENS_FLARE or REMOVE_SKY:
            _gen_masks_dir = os.path.join(config['DATASET_PATH'], "masks")
            _mask_debug_dir = os.path.join(config['DATASET_PATH'], "mask_previews")
            if REMOVE_SKY:
                _sky_args = ["--images-dir", image_path, "--masks-dir", _gen_masks_dir,
                             "--debug-dir", os.path.join(_mask_debug_dir, "sky"),
                             "--sky-dir", os.path.join(config['DATASET_PATH'], "sky_masks"),
                             "--nosky-masks-dir", os.path.join(config['DATASET_PATH'], "masks_no_sky")]
                for _dn in ("depth", "depth_images"):
                    if os.path.isdir(os.path.join(config['DATASET_PATH'], _dn)):
                        _sky_args.extend(["--depth-dir", os.path.join(config['DATASET_PATH'], _dn)])
                        break
                pipeline.create_component(
                    name="Remove-Sky",
                    comp_type=ComponentType.PRE_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="pre_processing/sky_mask.py",
                    args=_sky_args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
            if REMOVE_LENS_FLARE:
                _flare_args = ["--images-dir", image_path, "--masks-dir", _gen_masks_dir,
                               "--debug-dir", os.path.join(_mask_debug_dir, "lens_flare")]
                if REMOVE_SKY:
                    # The sky masks tell the detector where the sun can be (clipped
                    # white walls are not light sources). Transparent-sky training uses
                    # masks_no_sky/ (written by the sky step), so add the flare there too.
                    _flare_args.extend(["--sky-dir", os.path.join(config['DATASET_PATH'], "sky_masks"),
                                        "--also-masks-dir",
                                        os.path.join(config['DATASET_PATH'], "masks_no_sky")])
                pipeline.create_component(
                    name="Remove-Lens-Flare",
                    comp_type=ComponentType.PRE_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="pre_processing/lens_flare_mask.py",
                    args=_flare_args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
            # Downstream consumers (COLMAP mask_path, nerfstudio --masks-path, gsplat,
            # 3DGRUT) use the same masks/ folder as uploaded masks.
            GENERATED_MASKS = True
            ZIP_HAS_MASKS = True
    except Exception as e:
        error_message = f"Issue creating lens flare / sky mask components: {e}"
        pipeline.report_error(807, error_message)

    ##################################
    # PRE_PROCESSING COMPONENT:
    # Re-estimate the multi-camera rig (e.g. iPhone + Insta360 X5) from the
    # session's images. Runs before anything reads the poses. It only changes the
    # dataset when the upload has COLMAP rig files and the fit passes its quality
    # gate; otherwise the original poses are kept (see calibrate_rig/report.json).
    ##################################
    try:
        if CALIBRATE_RIG:
            pipeline.create_component(
                name="Calibrate-Rig",
                comp_type=ComponentType.PRE_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="pre_processing/calibrate_rig.py",
                args=["--dataset", config['DATASET_PATH'], "--gpu", "on" if USE_GPU == "true" else "off"],
                cwd=current_dir_path,
                requires_gpu=True
            )
    except Exception as e:
        error_message = f"Issue creating rig calibration component: {e}"
        pipeline.report_error(805, error_message)

    ##################################
    # PRE_PROCESSING COMPONENT:
    # Refine the poses of an uploaded COLMAP model (e.g. ARKit poses + LiDAR) with
    # bundle adjustment before training. Only for a COLMAP model that is trained
    # as-is (RUN_RECON=false). Runs after Calibrate-Rig because the rig is kept
    # fixed. Features are matched between pose neighbours to build tracks when
    # the model has none; only the poses change (see bundle_adjust/report.json).
    ##################################
    try:
        if BUNDLE_ADJUST_COLMAP_MODEL:
            if colmap_zip_found and config['RUN_RECON'] == 'false':
                pipeline.create_component(
                    name="Bundle-Adjust-Colmap-Model",
                    comp_type=ComponentType.PRE_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="pre_processing/bundle_adjust_model.py",
                    args=["--dataset", config['DATASET_PATH']],
                    cwd=current_dir_path,
                    requires_gpu=False
                )
            elif config['RUN_RECON'] == 'true' and (
                    config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true' or
                    config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true'):
                log.info("BUNDLE_ADJUST_COLMAP_MODEL=true with pose priors: COLMAP bundle_adjuster "
                         "runs after point_triangulator")
            else:
                log.warning("BUNDLE_ADJUST_COLMAP_MODEL=true only applies to an uploaded COLMAP model "
                            "with reconstruction disabled (RUN_RECON=false), or to pose priors "
                            "with reconstruction enabled; skipping")
    except Exception as e:
        error_message = f"Issue creating COLMAP model bundle adjustment component: {e}"
        pipeline.report_error(806, error_message)

    ##################################
    # PRE_PROCESSING COMPONENT:
    # Enhance sparse LiDAR depth images using PromptDA
    ##################################
    try:
        if ENHANCE_DEPTH:
            # Depth dir is resolved at runtime in the pipeline execution loop
            # because depth files may be extracted from a zip after this build phase.
            pipeline.create_component(
                name="Enhance-Depth",
                comp_type=ComponentType.PRE_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="pre_processing/enhance_depth.py",
                args=[
                    "--depth-dir", os.path.join(config['DATASET_PATH'], "depth"),
                    "--images-dir", os.path.join(config['DATASET_PATH'], "images"),
                    "--masks-dir", os.path.join(config['DATASET_PATH'], "masks"),
                ],
                cwd=current_dir_path,
                requires_gpu=True
            )
    except Exception as e:
        error_message = f"Issue creating depth enhancement component: {e}"
        pipeline.report_error(804, error_message)

    ##################################
    # RECONSTRUCTION COMPONENT:
    # Images to Point Cloud
    ##################################
    try:
        if config['RUN_RECON'] == "true":
            if config['SPHERICAL_CAMERA'] == "true":
                log.info("Using spherical camera processing with panorama_sfm.py")
            elif config['RECON_SOFTWARE_NAME'] == "colmap" or config['RECON_SOFTWARE_NAME'] == "glomap" or \
                config['RECON_SOFTWARE_NAME'] == "hloc":
                # Scale SIFT extraction to image resolution
                REF_MAX_IMAGE_SIZE = 3200
                REF_MAX_NUM_FEATURES = 8192
                sample_images = [f for f in os.listdir(image_path)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if sample_images:
                    with Image.open(os.path.join(image_path, sample_images[0])) as img:
                        sift_max_image_size = max(img.width, img.height)
                    scale_factor = (sift_max_image_size / REF_MAX_IMAGE_SIZE) ** 2
                    sift_max_num_features = min(int(REF_MAX_NUM_FEATURES * scale_factor), 65536)
                else:
                    sift_max_image_size = REF_MAX_IMAGE_SIZE
                    sift_max_num_features = REF_MAX_NUM_FEATURES
                log.info(f"SIFT extraction: max_image_size={sift_max_image_size}, "
                         f"max_num_features={sift_max_num_features}")

                # FEATURE EXTRACTOR COMPONENT
                args = [
                    "feature_extractor",
                    "--database_path", colmap_db_path,
                    "--image_path", image_path,
                    # Rig: one camera per camera folder (images/<camera>/<frame>.jpg)
                    "--ImageReader.single_camera_per_folder" if RIG_CONFIG else "--ImageReader.single_camera", "1",
                    "--FeatureExtraction.max_image_size", str(sift_max_image_size),
                    "--SiftExtraction.max_num_features", str(sift_max_num_features),
                    "--FeatureExtraction.use_gpu", "1"
                ]
                # Faster-GS's rasterizer is pinhole-only, so solve SfM with PINHOLE cameras
                if ENABLE_MULTI_GPU == "true" or IS_FASTER_GS or \
                    config['MODEL'] == "3dgut" or config['MODEL'] == "3dgrt":
                    if config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == "false" and \
                        config['USE_POSE_PRIOR_TRANSFORM_JSON'] == "false":
                        args.extend([
                            "--ImageReader.camera_model", "PINHOLE"
                        ])

                if config['ENABLE_ENHANCED_FEATURE_EXTRACTION'] == "true":
                    args.extend([
                        "--SiftExtraction.estimate_affine_shape", "1"
                        #"--SiftExtraction.domain_size_pooling", "1"
                    ])
                # Lens flare / sky masks: no SIFT features in masked pixels
                # (COLMAP reads masks/<image name>.png).
                if GENERATED_MASKS or UPLOADED_IMAGE_MASKS:
                    args.extend([
                        "--ImageReader.mask_path", os.path.join(config['DATASET_PATH'], "masks")
                    ])

                if config['LOG_VERBOSITY'] == "error":
                    args.extend([
                        "--log_level", "1"
                    ])
                pipeline.create_component(
                    name="ColmapSfM-Feature-Extractor",
                    comp_type=ComponentType.RECONSTRUCTION,
                    comp_environ=ComponentEnvironment.EXECUTABLE,
                    command="colmap",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
                # Camera rig: group the images of each frame (same file name in every
                # camera folder) into one rig frame with the cameras' relative poses
                # (and intrinsics, if given) from rig_config.json. Must run before
                # matching. Images are matched by the config's image_prefix values.
                if RIG_CONFIG:
                    pipeline.create_component(
                        name="ColmapSfM-Rig-Configurator",
                        comp_type=ComponentType.RECONSTRUCTION,
                        comp_environ=ComponentEnvironment.EXECUTABLE,
                        command="colmap",
                        args=["rig_configurator", "--database_path", colmap_db_path,
                              "--rig_config_path", rig_config_path],
                        cwd=current_dir_path,
                        requires_gpu=False
                    )
                # GPS priors read before the images were rewritten (EXIF lost) go into
                # the database for the spatial matcher and pose_prior_mapper.
                if GPS_PRIORS:
                    pipeline.create_component(
                        name="Write-GPS-Priors",
                        comp_type=ComponentType.RECONSTRUCTION,
                        comp_environ=ComponentEnvironment.PYTHON,
                        command="pre_processing/gps_priors.py",
                        args=["write", "--database", colmap_db_path, "--priors", gps_priors_json],
                        cwd=current_dir_path,
                        requires_gpu=False
                    )

                # Account for image name ordering and colmap database ordering when using pose priors
                # Perform the pose coordinate conversions or modify existing colmap model text files
                if config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true' or config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true':
                    if config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true':
                        use_transforms = "true"
                    else:
                        use_transforms = "false"
                    args = [
                        "-i", transforms_in_path,
                        "-c", config['SOURCE_COORD_NAME'],
                        "-p", config['POSE_IS_WORLD_TO_CAM'],
                        "-t", use_transforms
                    ]
                    pipeline.create_component(
                        name="ProcessPoseTransforms",
                        comp_type=ComponentType.RECONSTRUCTION,
                        comp_environ=ComponentEnvironment.PYTHON,
                        command="reconstruction/process_pose_transforms.py",
                        args=args,
                        cwd=current_dir_path,
                        requires_gpu=False
                    )

                # FEATURE MATCHER COMPONENT
                # Only use the sequential matcher if the images are in sequential order (e.g. video source)
                #if VIDEO is True:
                if config['MATCHING_METHOD'] == "sequential":
                    args = [
                        "sequential_matcher",
                        "--database_path",  colmap_db_path,
                        "--SequentialMatching.quadratic_overlap", "1"
                    ]
                    args.extend([
                        "--SequentialMatching.overlap", "10",
                        "--SequentialMatching.loop_detection", "1",
                        "--SequentialMatching.loop_detection_period", config['MAX_NUM_IMAGES'],
                        "--SequentialMatching.loop_detection_num_images", config['MAX_NUM_IMAGES'],
                        "--SequentialMatching.vocab_tree_path", colmap_vocab_path,
                        "--FeatureMatching.use_gpu", "1"
                    ])
                elif config['MATCHING_METHOD'] == "spatial":
                    args = [
                        "spatial_matcher",
                        "--database_path", colmap_db_path,
                        "--SpatialMatching.ignore_z", "0"
                    ]
                    if RIG_CONFIG:
                        # All cameras of a rig frame share one GPS position, so the
                        # default 50 nearest images would only reach ~50/N other
                        # frames (about +-2 m along a drive for 6 cameras) and give
                        # short-baseline pairs only. Keep the reach of 50 frames, and
                        # skip same-frame pairs (zero baseline, fixed by the rig).
                        args.extend([
                            "--SpatialMatching.max_num_neighbors", str(50 * RIG_NUM_CAMERAS),
                            "--FeatureMatching.skip_image_pairs_in_same_frame", "1"
                        ])
                elif config['MATCHING_METHOD'] == "vocab":
                    args = [
                        "vocab_tree_matcher",
                        "--database_path", colmap_db_path,
                        "--VocabTreeMatching.num_images", str(math.ceil(float(config['MAX_NUM_IMAGES'])/3)),
                        "--VocabTreeMatching.vocab_tree_path", colmap_vocab_path
                    ]
                # Otherwise run the exhaustive matcher which usually takes longer
                else:
                    args = [
                        "exhaustive_matcher",
                        "--database_path", colmap_db_path,
                        "--ExhaustiveMatching.block_size", config['MAX_NUM_IMAGES']
                    ]
                if config['LOG_VERBOSITY'] == "error":
                    args.extend([
                        "--log_level", "1"
                    ])
                pipeline.create_component(
                    name="ColmapSfM-Feature-Matcher",
                    comp_type=ComponentType.RECONSTRUCTION,
                    comp_environ=ComponentEnvironment.EXECUTABLE,
                    command="colmap",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )

                if config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == "true" or config['USE_POSE_PRIOR_TRANSFORM_JSON'] == "true":
                    # TRIANGULATION COMPONENT
                    # Known poses: follow the COLMAP FAQ ("Reconstruct sparse/dense model
                    # from known camera poses"): triangulate with the poses and intrinsics
                    # held fixed. Poses/intrinsics are only refined (bundle_adjuster below)
                    # when BUNDLE_ADJUST_COLMAP_MODEL=true.
                    args = [
                        'point_triangulator',
                        '--database_path', colmap_db_path,
                        '--image_path', image_path,
                        '--input_path', sparse_model_path,
                        '--output_path', sparse_model_path,
                        '--refine_intrinsics', "1" if BUNDLE_ADJUST_COLMAP_MODEL else "0",
                        '--Mapper.multiple_models', "0"
                    ]
                    if config['LOG_VERBOSITY'] == "error":
                        args.extend([
                            "--log_level", "1"
                        ])
                    pipeline.create_component(
                        name="ColmapSfM-Triangulator",
                        comp_type=ComponentType.RECONSTRUCTION,
                        comp_environ=ComponentEnvironment.EXECUTABLE,
                        command="colmap",
                        args=args,
                        cwd=current_dir_path,
                        requires_gpu=False
                    )
                    if BUNDLE_ADJUST_COLMAP_MODEL:
                        args = [
                            'bundle_adjuster',
                            '--input_path', sparse_model_path,
                            '--output_path', sparse_model_path,
                            '--BundleAdjustment.refine_principal_point', '0'
                        ]
                        pipeline.create_component(
                            name="ColmapSfM-Ba",
                            comp_type=ComponentType.RECONSTRUCTION,
                            comp_environ=ComponentEnvironment.EXECUTABLE,
                            command="colmap",
                            args=args,
                            cwd=current_dir_path,
                            requires_gpu=False
                        )
                    else:
                        log.info("Known poses: triangulating with fixed poses and intrinsics "
                                 "(COLMAP FAQ); set BUNDLE_ADJUST_COLMAP_MODEL=true to refine them")
                else:
                    # MAPPER COMPONENT
                    if config['RECON_SOFTWARE_NAME'] == "colmap" :
                        # GPS: pose_prior_mapper (same Mapper options; the GPS std and
                        # covariance overwrite are added at runtime from gps_priors.json,
                        # and it falls back to mapper if the database has no priors).
                        args = [
                            "pose_prior_mapper" if GPS_PRIORS else "mapper",
                            "--database_path", colmap_db_path,
                            "--image_path", image_path,
                            "--output_path", sparse_path,
                            "--Mapper.multiple_models", "0"
                        ]
                        if RIG_CONFIG:
                            # Keep the cameras' relative poses from rig_config.json fixed
                            # (COLMAP rig workflow for known rigs): BA moves whole frames.
                            args.extend(["--Mapper.ba_refine_sensor_from_rig", "0"])
                            if RIG_HAS_INTRINSICS and "--Mapper.ba_refine_focal_length" not in args:
                                # Exact intrinsics in the rig config (rig_configurator
                                # writes them to the database): keep them fixed.
                                args.extend([
                                    "--Mapper.ba_refine_focal_length", "0",
                                    "--Mapper.ba_refine_principal_point", "0",
                                    "--Mapper.ba_refine_extra_params", "0"
                                ])
                        if config.get('PRESERVE_SCENE_SCALE', 'false') == 'true':
                            pass  # Scale is preserved via auto_scale_poses=False in nerfstudio dataparser
                        if config.get('ENABLE_FL_METRIC', 'false') == 'true':
                            # User provided a known metric focal length — lock it in during BA
                            # so the reconstruction stays in true metric scale
                            args.extend([
                                "--Mapper.ba_refine_focal_length", "0",
                                "--Mapper.ba_refine_principal_point", "0",
                                "--Mapper.ba_refine_extra_params", "0"
                            ])
                        if config['LOG_VERBOSITY'] == "error":
                            args.extend([
                                "--log_level", "1"
                            ])
                        if RIG_CONFIG:
                            # Rig runs use CPU bundle adjustment (COLMAP's reference rig
                            # workflow). With GPU BA, rig reconstructions collapsed after
                            # repeated "Unable to perform dense Cholesky factorization".
                            log.info("Camera rig: using CPU bundle adjustment in the mapper")
                        elif int(pipeline.config.num_gpus) > 0:
                            args.extend(["--Mapper.ba_use_gpu", "1"])
                            if len([f for f in os.listdir(image_path)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) >= 250:
                                args.extend(["--Mapper.ba_global_backend", "CASPAR"])
                        pipeline.create_component(
                            name="ColmapSfM-Mapper",
                            comp_type=ComponentType.RECONSTRUCTION,
                            comp_environ=ComponentEnvironment.EXECUTABLE,
                            command="colmap",
                            cwd=current_dir_path,
                            args=args,
                            requires_gpu=False
                        )
                    elif config['RECON_SOFTWARE_NAME'] == "glomap":
                        args = [
                            "view_graph_calibrator",
                            "--database_path", colmap_db_path
                        ]
                        pipeline.create_component(
                            name="GlomapSfM-ViewGraph",
                            comp_type=ComponentType.RECONSTRUCTION,
                            comp_environ=ComponentEnvironment.EXECUTABLE,
                            command="colmap",
                            cwd=current_dir_path,
                            args=args,
                            requires_gpu=False
                        )
                        args = [
                            "global_mapper",
                            "--database_path", colmap_db_path,
                            "--image_path", image_path,
                            "--output_path", sparse_path
                        ]
                        pipeline.create_component(
                            name="GlomapSfM-Mapper",
                            comp_type=ComponentType.RECONSTRUCTION,
                            comp_environ=ComponentEnvironment.EXECUTABLE,
                            command="colmap",
                            cwd=current_dir_path,
                            args=args,
                            requires_gpu=False
                        )
                    else: #hloc
                        args = [
                            "hierarchical_mapper",
                            "--database_path", colmap_db_path,
                            "--image_path", image_path,
                            "--output_path", sparse_path
                        ]
                        pipeline.create_component(
                            name="HlocSfM-Mapper",
                            comp_type=ComponentType.RECONSTRUCTION,
                            comp_environ=ComponentEnvironment.EXECUTABLE,
                            command="colmap",
                            cwd=current_dir_path,
                            args=args,
                            requires_gpu=False
                        )
                        args = [
                            'point_triangulator',
                            '--database_path', colmap_db_path,
                            '--image_path', image_path,
                            '--input_path', sparse_model_path,
                            '--output_path', sparse_model_path,
                            '--refine_intrinsics', "1",
                            '--Mapper.multiple_models', "0"
                        ]
                        pipeline.create_component(
                            name="HlocSfM-Tri",
                            comp_type=ComponentType.RECONSTRUCTION,
                            comp_environ=ComponentEnvironment.EXECUTABLE,
                            command="colmap",
                            cwd=current_dir_path,
                            args=args,
                            requires_gpu=False
                        )
                # IMAGE UNDISTORTER
                # Run undistorter for multi-GPU or when using 3DGRUT with pose priors
                # For depth loss: convert cameras.bin to PINHOLE in-place instead —
                # the undistorter renames images in images.bin which breaks point_indices
                if IS_GSPLAT_DEPTH and ENABLE_MULTI_GPU == "false":
                    pipeline.create_component(
                        name="Convert-Cameras-To-Pinhole",
                        comp_type=ComponentType.RECONSTRUCTION,
                        comp_environ=ComponentEnvironment.PYTHON,
                        command="reconstruction/convert_cameras_to_pinhole.py",
                        args=["-s", sparse_model_path],
                        cwd=current_dir_path,
                        requires_gpu=False
                    )
                if ENABLE_MULTI_GPU == "true" or \
                    (config['MODEL'] == "3dgut" or config['MODEL'] == "3dgrt") and \
                    (config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true' or config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true'):
                    args = [
                        "image_undistorter",
                        "--image_path", image_path,
                        "--input_path", sparse_model_path,
                        "--output_path", sparse_model_path,
                        "--output_type", "COLMAP"
                    ]
                    if config['LOG_VERBOSITY'] == "error":
                        args.extend([
                            "--log_level", "1"
                        ])
                    pipeline.create_component(
                        name="ColmapSfM-Image-Undistorter",
                        comp_type=ComponentType.RECONSTRUCTION,
                        comp_environ=ComponentEnvironment.EXECUTABLE,
                        command="colmap",
                        args=args,
                        cwd=current_dir_path,
                        requires_gpu=False
                    )
                    
                    # Update cameras.txt to PINHOLE model after undistortion
                    if (config['MODEL'] == "3dgut" or config['MODEL'] == "3dgrt") and \
                       (config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true' or config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true'):
                        args = [
                            "-s", sparse_model_path
                        ]
                        pipeline.create_component(
                            name="UpdateCameraModel",
                            comp_type=ComponentType.RECONSTRUCTION,
                            comp_environ=ComponentEnvironment.PYTHON,
                            command="reconstruction/update_camera_model.py",
                            args=args,
                            cwd=current_dir_path,
                            requires_gpu=False
                        )
                elif ENABLE_MULTI_GPU == "false" and \
                    (config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true' or config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true'):
                    undist_output = os.path.join(
                        config['DATASET_PATH'], '_undistorted'
                    )
                    args = [
                        "image_undistorter",
                        "--image_path", image_path,
                        "--input_path", sparse_model_path,
                        "--output_path", undist_output,
                        "--output_type", "COLMAP"
                    ]
                    if config['LOG_VERBOSITY'] == "error":
                        args.extend(["--log_level", "1"])
                    pipeline.create_component(
                        name="ColmapSfM-Image-Undistorter",
                        comp_type=ComponentType.RECONSTRUCTION,
                        comp_environ=ComponentEnvironment.EXECUTABLE,
                        command="colmap",
                        args=args,
                        cwd=current_dir_path,
                        requires_gpu=False
                    )
            elif config['RECON_SOFTWARE_NAME'] == "map_anything": #MapAnything
                args = [
                    "--scene_dir", config['DATASET_PATH'],
                    "--skip_point2d",
                    "--voxel_size", "0.01"
                ]
                pipeline.create_component(
                    name="Map-Anything",
                    comp_type=ComponentType.RECONSTRUCTION,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="reconstruction/run_map_anything.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
            else:
                pipeline.report_error(
                    745, f"Reconstruction software not implemented yet:{config['RECON_SOFTWARE_NAME']}"
                )
        else:
            log.info("Reconstruction configured to be skipped...skipping reconstruction")
            # 3DGRUT handles all COLMAP camera models natively (PINHOLE,
            # SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV, OPENCV_FISHEYE),
            # so no undistortion step is needed for 3dgut.
            if colmap_zip_found and config['MODEL'] == '3dgrt':
                undist_output = os.path.join(config['DATASET_PATH'], '_undistorted')
                undist_args = [
                    "image_undistorter",
                    "--image_path", image_path,
                    "--input_path", sparse_model_path,
                    "--output_path", undist_output,
                    "--output_type", "COLMAP"
                ]
                if config['LOG_VERBOSITY'] == "error":
                    undist_args.extend(["--log_level", "1"])
                pipeline.create_component(
                    name="ColmapSfM-Image-Undistorter",
                    comp_type=ComponentType.RECONSTRUCTION,
                    comp_environ=ComponentEnvironment.EXECUTABLE,
                    command="colmap",
                    args=undist_args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
                log.info("Added undistorter for 3DGRT with pre-existing COLMAP reconstruction")
    except Exception as e:
        error_message = f"Issue creating the reconstruction component: {e}"
        pipeline.report_error(750, error_message)

    ##################################
    # RECONSTRUCTION COMPONENT:
    # Point Cloud, Images, and Poses to NerfStudio format
    ##################################
    try:
        if config['RUN_TRAIN'] == "true" and config['RUN_RECON'] == "true":
            if config['RECON_SOFTWARE_NAME'] == "colmap" or config['RECON_SOFTWARE_NAME'] == "glomap" or \
                config['RECON_SOFTWARE_NAME'] == "map_anything" or config['RECON_SOFTWARE_NAME'] == "hloc":
                args = ["--data_dir", config['DATASET_PATH']]
                pipeline.create_component(
                    name="Colmap-to-Nerfstudio",
                    comp_type=ComponentType.RECONSTRUCTION,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="training/colmap_to_nerfstudio_cam.py",
                    cwd=current_dir_path,
                    args=args,
                    requires_gpu=False
                )
            else:
                pipeline.report_error(
                    750,
                    f"Reconstruction software name given not implemented:{config['RECON_SOFTWARE_NAME']}"
                )
        elif colmap_zip_needs_conversion and config['MODEL'] not in ('3dgrt', '3dgut', 'faster-gs'):
            # zip had COLMAP sparse data but no transforms.json — run conversion now
            log.info("colmap_zip_needs_conversion=True: adding Colmap-to-Nerfstudio component")
            args = ["--data_dir", config['DATASET_PATH']]
            pipeline.create_component(
                name="Colmap-to-Nerfstudio",
                comp_type=ComponentType.RECONSTRUCTION,
                comp_environ=ComponentEnvironment.PYTHON,
                command="training/colmap_to_nerfstudio_cam.py",
                cwd=current_dir_path,
                args=args,
                requires_gpu=False
            )
        elif (config['RUN_TRAIN'] == "true"
              and config['MODEL'] not in ('3dgrt', '3dgut', 'faster-gs')
              and not os.path.exists(os.path.join(config['DATASET_PATH'], "transforms.json"))):
            # A COLMAP model is present if cameras.bin OR cameras.txt exists at
            # either the colmap/sparse/0 or sparse/0 layout. The downstream
            # converter (colmap_to_nerfstudio_cam.py -> colmap_to_json) accepts
            # both the binary and text formats, so the detection here must too —
            # otherwise a .txt-format or sparse/0-only model is silently skipped
            # and the dn-splatter normal-nerfstudio dataparser later fails with
            # FileNotFoundError: transforms.json.
            _has_sparse = any(
                os.path.isfile(os.path.join(config['DATASET_PATH'], *_loc, _f))
                for _loc in (("colmap", "sparse", "0"), ("sparse", "0"))
                for _f in ("cameras.bin", "cameras.txt")
            )
            if _has_sparse:
                log.info("transforms.json missing but COLMAP model found — adding conversion component")
                args = ["--data_dir", config['DATASET_PATH']]
                pipeline.create_component(
                    name="Colmap-to-Nerfstudio",
                    comp_type=ComponentType.RECONSTRUCTION,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="training/colmap_to_nerfstudio_cam.py",
                    cwd=current_dir_path,
                    args=args,
                    requires_gpu=False
                )
            else:
                log.info("Not configured to perform SfM reconstruction...skipping dataset conversion.")
        else:
            log.info("Not configured to perform SfM reconstruction...skipping dataset conversion.")
    except Exception as e:
        error_message = f"Issue creating the Colmap to Nerfstudio component: {e}"
        pipeline.report_error(755, error_message)

    ##################################
    # PRE-PROCESS COMPONENT:
    # Generate normals and aligned depth for DN-Splatter / AGS-Mesh
    ##################################
    try:
        if config['MODEL'] in ("dn-splatter", "dn-splatter-big", "ags-mesh") and config['RUN_TRAIN'] == "true":
            depth_dir = os.path.join(config['DATASET_PATH'], "depth")
            if not os.path.isdir(depth_dir):
                depth_dir = os.path.join(config['DATASET_PATH'], "depth_images")
            has_sensor_depth = os.path.isdir(depth_dir) and any(
                f.lower().endswith(('.png', '.jpg', '.npy'))
                for f in os.listdir(depth_dir)
            ) if os.path.isdir(depth_dir) else False
            args = [
                "--data-dir", config['DATASET_PATH'],
                "--normal-format", "dsine",
            ]
            if has_sensor_depth:
                log.info(f"DN-Splatter preprocess: sensor depth found in {depth_dir}, will inject depth_file_path entries")
            if config['MODEL'] == "ags-mesh":
                args.append("--generate-depth-masks")
            pipeline.create_component(
                name="DN-Splatter-Preprocess",
                comp_type=ComponentType.TRAINING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="training/dn_splatter_preprocess.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=True
            )
    except Exception as e:
        error_message = f"Issue creating DN-Splatter pre-processing component: {e}"
        pipeline.report_error(756, error_message)

    ##################################
    # TRAINING COMPONENT:
    # Point Cloud, Images, and Poses to 3D Gaussian Splat
    ##################################
    try:
        if config['RUN_TRAIN'] == "true" or (config['RUN_RECON'] == "false" and config['RUN_TRAIN'] == "false"):
            if config['RECON_SOFTWARE_NAME'] == "glomap" or config['RECON_SOFTWARE_NAME'] == "colmap" or \
                config['RECON_SOFTWARE_NAME'] == "map_anything" or config['RECON_SOFTWARE_NAME'] == "hloc":
                data_model = "colmap"
            # Single GPU gsplat with depth loss — override model choice and use gsplat simple_trainer
            if IS_GSPLAT_DEPTH and ENABLE_MULTI_GPU == "false":
                isp_mode = config.get('THREED_ISP', 'none').lower()
                if config['MODEL'] == "gsplat-depth-mcmc":
                    model = "mcmc"
                else:
                    model = "default"
                log.info(f"GSplat depth-loss strategy: {model}")
                args = [
                    model,
                    "--max_steps", str(int(config['MAX_STEPS'])),
                    "--result-dir", output_path,
                    "--data_factor", "1",
                    "--steps_scaler", "1.0",
                    "--disable_viewer",
                    "--eval_steps", str(int(config['MAX_STEPS'])),
                    "--depth-loss",
                    "--depth_lambda", "1e-3",
                    "--data-dir", config['DATASET_PATH']
                ]
                if config.get('PRESERVE_SCENE_SCALE', 'false').lower() == 'true':
                    args.extend(["--no-normalize-world-space", "--depth_lambda", "1e-4"])
                if model == "mcmc":
                    num_gaussians = int(config.get('NUM_GAUSSIANS', '1000000'))
                    # simple_trainer's MCMC options live under Config.strategy
                    args.extend(["--strategy.cap-max", str(num_gaussians)])
                if isp_mode == "ppisp":
                    args.extend(["--post_processing", "ppisp", "--batch_size", "1"])
                elif isp_mode == "bilagrid":
                    args.extend(["--post_processing", "bilateral_grid"])
                if ENABLE_ABSGRAD:
                    args.append("--strategy.absgrad")
                # REFINE_POSES: gsplat's per-image pose correction (pose_opt, off by default)
                if REFINE_POSES:
                    args.append("--pose_opt")
                log.info(f"GSplat refine poses: {REFINE_POSES}")
                pipeline.create_component(
                    name="Train",
                    comp_type=ComponentType.TRAINING,
                    comp_environ=ComponentEnvironment.EXECUTABLE,
                    command="/opt/ml/code/training/run_gsplat_trainer.sh",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
                ckpt_dir = os.path.join(output_path, "ckpts")
                eval_args = [
                    model,
                    "--disable_viewer",
                    "--data_factor", "1",
                    "--depth-loss",
                    "--data-dir", config['DATASET_PATH'],
                    "--result-dir", output_path,
                    "--ckpt", os.path.join(ckpt_dir, "ckpt_*.pt")
                ]
                if config.get('PRESERVE_SCENE_SCALE', 'false').lower() == 'true':
                    eval_args.append("--no-normalize-world-space")
                if isp_mode == "ppisp":
                    eval_args.extend(["--post_processing", "ppisp", "--batch_size", "1"])
                elif isp_mode == "bilagrid":
                    eval_args.extend(["--post_processing", "bilateral_grid"])
                pipeline.create_component(
                    name="GSplat-Metrics",
                    comp_type=ComponentType.TRAINING,
                    comp_environ=ComponentEnvironment.EXECUTABLE,
                    command="/opt/ml/code/training/run_gsplat_trainer.sh",
                    args=eval_args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
            # Single GPU dn-splatter / ags-mesh
            elif ENABLE_MULTI_GPU == "false" and \
                config['MODEL'] in ("dn-splatter", "dn-splatter-big", "ags-mesh"):
                # Detect whether sensor depth images are provided in a depth/ directory.
                # Mode 1 (sensor depth): depth/ dir present → use EdgeAwareLogL1, normal-supervision=depth
                # Mode 2 (mono depth):   no depth/ dir    → use PearsonDepth,       normal-supervision=mono
                depth_dir = os.path.join(config['DATASET_PATH'], "depth")
                if not os.path.isdir(depth_dir):
                    depth_dir = os.path.join(config['DATASET_PATH'], "depth_images")
                has_sensor_depth = False
                if os.path.isdir(depth_dir):
                    _depth_samples: list[str] = []
                    for _root, _dirs, _files in os.walk(depth_dir):
                        for _f in _files:
                            if _f.lower().endswith(('.png', '.jpg', '.npy')):
                                _depth_samples.append(
                                    os.path.join(_root, _f)
                                )
                            if len(_depth_samples) >= 3:
                                break
                        if _depth_samples:
                            break
                    if _depth_samples:
                        import cv2 as _cv2_check
                        _s = _cv2_check.imread(
                            _depth_samples[0],
                            _cv2_check.IMREAD_ANYDEPTH
                        )
                        if _s is not None and str(_s.dtype) == 'uint16':
                            _mm_max = float(_s.max()) * 0.001
                            _mm_min = float(_s[_s > 0].min()) * 0.001 if (_s > 0).any() else 0
                            has_sensor_depth = 0.1 <= _mm_min and _mm_max <= 1000.0
                # Also check depth_sensor/ or depth_images/ after preprocess
                if not has_sensor_depth:
                    for _dname in ("depth_sensor", "depth_images", "depth"):
                        _ds_dir = os.path.join(config['DATASET_PATH'], _dname)
                        if os.path.isdir(_ds_dir):
                            for _r2, _d2, _f2 in os.walk(_ds_dir):
                                if any(_fn.endswith('.png') for _fn in _f2):
                                    has_sensor_depth = True
                                    break
                        if has_sensor_depth:
                            break
                log.info(f"DN-Splatter depth mode: {'sensor (valid uint16)' if has_sensor_depth else 'mono (no valid sensor depth)'}")

                args = [
                    config['MODEL'],
                    "--viewer.quit-on-train-completion=True",
                    "--timestamp", TRAIN_EXPERIMENT_NAME,
                    "--pipeline.model.use-depth-loss", "True",
                    "--pipeline.model.use-normal-loss", "True",
                    "--max-num-iterations", str(int(config['MAX_STEPS'])),
                ]
                # EdgeAwareLogL1 is numerically stable for both sensor and mono depth.
                # PearsonDepth can produce NaN gradients when depth variance is near zero.
                depth_loss_type = "EdgeAwareLogL1"
                depth_lambda = "0.2" if config['MODEL'] == "ags-mesh" else "0.3"
                # Always use mono normals from dsine — they provide stronger supervision
                # on planar surfaces than depth-derived normals (which are self-referencing).
                normal_supervision = "mono"
                # Stop densification at ~67% of training to allow a polish phase
                stop_split_at = int(int(config['MAX_STEPS']) * 0.67)
                log.info(f"DN-Splatter depth loss: {depth_loss_type}, normal-supervision: {normal_supervision}, stop-split-at: {stop_split_at}")
                normal_lambda = "0.2" if config['MODEL'] == "ags-mesh" else "0.1"
                args.extend([
                    "--pipeline.model.depth-lambda", depth_lambda,
                    "--pipeline.model.depth-loss-type", depth_loss_type,
                    "--pipeline.model.normal-supervision", normal_supervision,
                    "--pipeline.model.normal-lambda", normal_lambda,
                    "--pipeline.model.use-normal-tv-loss", "True",
                    "--pipeline.model.stop-split-at", str(stop_split_at),
                    # dn-splatter defaults use_absgrad=True; pass ENABLE_ABSGRAD explicitly so
                    # the pipeline setting applies here like it does for the other models.
                    "--pipeline.model.use-absgrad", "True" if ENABLE_ABSGRAD else "False",
                    # REFINE_POSES: jointly optimize a per-image rotation + translation
                    # correction (SO3xR3) to absorb small pose errors such as ARKit drift.
                    "--pipeline.model.camera-optimizer.mode", "SO3xR3" if REFINE_POSES else "off",
                ])
                # THREED_ISP=bilagrid: per-image bilateral grid color correction during
                # training. PPISP is gsplat-only, so it falls back to the bilateral grid
                # here, matching the Splatfacto behavior.
                dn_isp_mode = config.get('THREED_ISP', 'none').lower()
                dn_use_bilagrid = dn_isp_mode in ("bilagrid", "ppisp")
                if dn_isp_mode == "ppisp":
                    log.info("PPISP not supported with dn-splatter (nerfstudio), using Bilateral-Grid instead")
                args.extend(["--pipeline.model.use-bilateral-grid", "True" if dn_use_bilagrid else "False"])
                log.info(f"DN-Splatter absgrad: {ENABLE_ABSGRAD}, refine poses: {REFINE_POSES}, "
                         f"bilateral grid: {dn_use_bilagrid}")
                if config['LOG_VERBOSITY'] != "debug":
                    args.extend([
                        "--logging.local-writer.enable", "False",
                        "--logging.profiler", "none"
                    ])
                args.extend([
                    "normal-nerfstudio",
                    "--data", config['DATASET_PATH'],
                    "--load-3D-points", "True",
                    "--load-normals", "True",
                    "--normal-format", "dsine",
                    "--load-depths", "True",
                ])
                # normal-nerfstudio defaults to auto_scale_poses=True, which divides all
                # camera positions by the largest one (scene squeezed into ~[-1, 1]). The
                # sensor depth is scaled by the same factor, so training is consistent, but
                # the exported splat is no longer metric. Honour PRESERVE_SCENE_SCALE like
                # the splatfacto branch does, and keep the pose-prior world frame too.
                if config.get('PRESERVE_SCENE_SCALE', 'false').lower() == 'true':
                    args.extend(["--auto-scale-poses", "False"])
                    log.info("DN-Splatter: PRESERVE_SCENE_SCALE=true -> --auto-scale-poses False (metric output)")
                args.extend(world_frame_dataparser_args(config, user_supplied_model=colmap_zip_found))
                if has_sensor_depth:
                    # Sensor depth is loaded via transforms.json depth_file_path entries;
                    # normal-nerfstudio dataparser does not accept --depth-mode as a CLI arg.
                    pass
                if config['MODEL'] == "ags-mesh":
                    # Only load confidence masks if valid sensor depth was injected
                    _marker = os.path.join(config['DATASET_PATH'], ".sensor_depth_valid")
                    if os.path.exists(_marker):
                        args.extend(["--load-depth-confidence-masks", "True"])
                pipeline.create_component(
                    name="Train",
                    comp_type=ComponentType.TRAINING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="training/run_dn_splatter_wrapper.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
            # Single GPU gsplat
            elif ENABLE_MULTI_GPU == "false" and not IS_FASTER_GS and not IS_GSPLAT_DEPTH and \
                config['MODEL'] != "3dgut" and config['MODEL'] != "3dgrt":
                args = [
                    config['MODEL'],
                    "--viewer.quit-on-train-completion=True"
                ]
                if config['LOG_VERBOSITY'] != "debug":
                    args.extend([
                        "--logging.local-writer.enable", "False",
                        "--logging.profiler", "none"
                    ])
                if config['MODEL'] == "nerfacto":
                    args.extend([
                        "--timestamp", TRAIN_EXPERIMENT_NAME,
                        "--pipeline.model.predict-normals", "True",
                        "--max-num-iterations", str(config['MAX_STEPS']),
                    ])
                elif config['MODEL'] in ("splatfacto", "splatfacto-big", "splatfacto-mcmc"):
                    if config['RUN_RECON'] == "false" and not colmap_zip_found: # Resume training
                        dataset_models_path = os.path.join(config['DATASET_PATH'], "nerfstudio_models")
                        has_ckpt = os.path.exists(dataset_models_path) and any(
                            f.endswith('.ckpt') for f in os.listdir(dataset_models_path)
                        ) if os.path.exists(dataset_models_path) else False

                        if has_ckpt:
                            resume_training_active = True
                            model_ckpt_path = os.path.join(config['CODE_PATH'], "outputs", "unnamed", model, RESUME_TRAIN_EXPERIMENT_NAME, model_dir_name)
                            model_config_path = os.path.join(config['CODE_PATH'], "outputs", "unnamed", model, RESUME_TRAIN_EXPERIMENT_NAME, "config.yml")
                            additional_steps = max(int(config['MAX_STEPS']), 9000)
                            ckpt_files = sorted([f for f in os.listdir(dataset_models_path) if f.endswith('.ckpt')])
                            ckpt_step = 0
                            if ckpt_files:
                                _m = re.search(r'step-(\d+)\.ckpt$', ckpt_files[-1])
                                if _m:
                                    ckpt_step = int(_m.group(1)) + 1
                            total_iterations = ckpt_step + additional_steps
                            stop_split_at = ckpt_step + int(additional_steps * 0.67)
                            args.extend([
                                "--load-dir", dataset_models_path,
                                "--load-scheduler", "False",
                                "--timestamp", RESUME_TRAIN_EXPERIMENT_NAME,
                                "--pipeline.model.use-scale-regularization", "True",
                                "--pipeline.model.use-bilateral-grid", "True",
                                "--pipeline.model.stop-split-at", str(stop_split_at),
                                "--max-num-iterations", str(total_iterations),
                                "--steps-per-save", str(total_iterations),
                            ])
                            if config['MODEL'] == "splatfacto-mcmc":
                                num_gaussians = int(config.get('NUM_GAUSSIANS', '1000000'))
                                args.extend(["--pipeline.model.max-gs-num", str(num_gaussians)])
                            log.info(f"Resume training: loading weights from {dataset_models_path}, fresh optimizer/scheduler, steps {ckpt_step}-{total_iterations}, densify until {stop_split_at}")
                    else:
                        # Initial training (either RUN_RECON=true or colmap_zip_found)
                        isp_mode = config.get('THREED_ISP', 'none').lower()
                        args.extend([
                        "--timestamp", TRAIN_EXPERIMENT_NAME,
                        "--pipeline.model.use-scale-regularization", "True",
                        "--max-num-iterations", str(int(config['MAX_STEPS']))
                    ])
                        if config['MODEL'] == "splatfacto-mcmc":
                            num_gaussians = int(config.get('NUM_GAUSSIANS', '1000000'))
                            args.extend([
                                "--pipeline.model.max-gs-num", str(num_gaussians),
                                "--pipeline.model.stop-split-at", str(int(config['MAX_STEPS'])),
                            ])
                        if isp_mode == "bilagrid":
                            args.extend(["--pipeline.model.use-bilateral-grid", "True"])
                        elif isp_mode == "ppisp":
                            log.info("PPISP not supported with Splatfacto (nerfstudio), using Bilateral-Grid instead")
                            args.extend(["--pipeline.model.use-bilateral-grid", "True"])
                        if ENABLE_ABSGRAD:
                            args.extend(["--pipeline.model.use-absgrad", "True"])
                        # REFINE_POSES: per-image rotation + translation correction (SO3xR3).
                        # Initial training only: a resumed checkpoint has no pose_adjustment
                        # tensor when it was trained with the optimizer off.
                        args.extend([
                            "--pipeline.model.camera-optimizer.mode",
                            "SO3xR3" if REFINE_POSES else "off",
                        ])
                        log.info(f"Splatfacto refine poses: {REFINE_POSES}")
                elif config['MODEL'] == "splatfacto-w-light":
                    if config['RUN_RECON'] == "false": # Resume training
                        if os.path.exists(model_ckpt_path):
                            args.extend([
                                "--load-dir", model_ckpt_path,
                                "--load-scheduler", "False",
                            ])
                            log.info(f"Stage-2 training: loading weights from {model_ckpt_path} but starting fresh training schedule")
                        args.extend([
                            "--timestamp", RESUME_TRAIN_EXPERIMENT_NAME,
                            "--max-num-iterations", str(REFINE_STEPS_SPLATFACTO)
                        ])
                    else:
                        args.extend([
                            "--timestamp", TRAIN_EXPERIMENT_NAME,
                            "--max-num-iterations", str(config['MAX_STEPS'])
                        ])
                    args.extend([
                        "--pipeline.model.enable-alpha-loss=True",
                        "--pipeline.model.enable-robust-mask=True"
                    ])
                    if config['REMOVE_BACKGROUND'] == "true":
                        args.extend([
                            "--pipeline.model.enable-bg-model=False"
                        ])
                    else:
                        args.extend([
                            "--pipeline.model.enable-bg-model=True"
                        ])
                else:
                    pipeline.report_error(765, "Trainer specified does not match proper configuration")

                # Set command based on model type
                if config['MODEL'] == "splatfacto-w-light":
                    command = "/opt/ml/code/training/run_splatfacto_w_wrapper.py"
                else:
                    command = "ns-train"

                auto_scale_value = "True" if config.get('PRESERVE_SCENE_SCALE', 'false') == 'false' else "False"
                args.extend([
                    data_model,
                    "--data", config['DATASET_PATH'],
                    "--downscale-factor", "1",
                    "--auto-scale-poses", auto_scale_value
                ])
                if auto_scale_value == "True":
                    args.extend(["--center-method", "poses"])
                args.extend(world_frame_dataparser_args(config, user_supplied_model=colmap_zip_found))
                if ZIP_HAS_MASKS and config['SPHERICAL_CAMERA'] != "true":
                    masks_path = os.path.join(config['DATASET_PATH'], 'masks')
                    args.extend(["--masks-path", masks_path])
                    log.info(f"Enabling mask training from zip: {masks_path}")
                pipeline.create_component(
                    name="Train",
                    comp_type=ComponentType.TRAINING,
                    comp_environ=ComponentEnvironment.EXECUTABLE,
                    command=command,
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
            # Multi-gpu gsplat
            elif ENABLE_MULTI_GPU == "true" and not IS_FASTER_GS and \
                config['MODEL'] != "3dgut" and config['MODEL'] != "3dgrt":
                #multi-gpu, use gsplat training strategy
                num_gpus = int(pipeline.config.num_gpus)
                batch_size = 1
                steps_scaler = 1.0 / num_gpus  # Scale by number of GPUs only
                #steps_scaler = 0.96*(num_gpus*batch_size)**(-1.689)
                if config['MODEL'] in ("splatfacto-mcmc", "gsplat-depth-mcmc"):
                    model = "mcmc"
                else:
                    model = "default"
                isp_mode = config.get('THREED_ISP', 'none').lower()
                depth_loss_flag = "--depth-loss" if IS_GSPLAT_DEPTH else "--no-depth-loss"
                args = [
                    model,
                    "--max_steps", str(int(config['MAX_STEPS'])),
                    "--result-dir", output_path,
                    "--data_factor", "1",
                    "--steps_scaler", str(steps_scaler),
                    "--disable_viewer",
                    "--packed",
                    "--eval_steps", str(int(config['MAX_STEPS'])),
                    #depth_loss_flag,
                    "--data-dir", config['DATASET_PATH']
                ]
                if IS_GSPLAT_DEPTH:
                    args.extend(["--depth_lambda", "1e-3"])
                if config.get('PRESERVE_SCENE_SCALE', 'false').lower() == 'true':
                    args.extend(["--no-normalize-world-space", "--depth_lambda", "1e-4"])
                if model == "mcmc":
                    num_gaussians = int(config.get('NUM_GAUSSIANS', '1000000'))
                    # simple_trainer's MCMC options live under Config.strategy
                    args.extend(["--strategy.cap-max", str(num_gaussians)])
                if isp_mode == "ppisp":
                    log.info("PPISP not supported with multi-GPU gsplat (requires single GPU), skipping")
                elif isp_mode == "bilagrid":
                    log.info("Bilateral grid not supported with multi-GPU gsplat (requires single GPU), skipping")
                if ENABLE_ABSGRAD:
                    args.append("--strategy.absgrad")
                # REFINE_POSES: gsplat's per-image pose correction (pose_opt, off by default)
                if REFINE_POSES:
                    args.append("--pose_opt")
                log.info(f"Multi-GPU gsplat refine poses: {REFINE_POSES}")
                pipeline.create_component(
                    name="Train",
                    comp_type=ComponentType.TRAINING,
                    comp_environ=ComponentEnvironment.EXECUTABLE,
                    command="/opt/ml/code/training/run_gsplat_trainer.sh",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
                # Create metrics component for multi-GPU gsplat after training
                ckpt_dir = os.path.join(output_path, "ckpts")
                eval_args = [
                    model,
                    "--disable_viewer",
                    "--data_factor", "1",
                    depth_loss_flag,
                    "--data-dir", config['DATASET_PATH'],
                    "--result-dir", output_path,
                    "--ckpt", os.path.join(ckpt_dir, "ckpt_*.pt")
                ]
                if config.get('PRESERVE_SCENE_SCALE', 'false').lower() == 'true':
                    eval_args.append("--no-normalize-world-space")
                if isp_mode == "bilagrid":
                    log.info("Bilateral grid not supported with multi-GPU gsplat (requires single GPU), skipping eval post_processing")
                pipeline.create_component(
                    name="GSplat-Metrics",
                    comp_type=ComponentType.TRAINING,
                    comp_environ=ComponentEnvironment.EXECUTABLE,
                    command="/opt/ml/code/training/run_gsplat_trainer.sh",
                    args=eval_args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
            # Faster-GS (NeRFICG): stages images + sparse/0 (undistorting to PINHOLE if
            # needed), trains, and exports the PLY straight to ply_path. Resume from a
            # model.tar.gz retrains from the archived dataset (no checkpoint continuation).
            elif IS_FASTER_GS:
                if config['RUN_RECON'] == "false" and not colmap_zip_found:
                    log.info("Faster-GS does not support checkpoint resume; training from the archived dataset")
                args = [
                    "train",
                    "--data-dir", config['DATASET_PATH'],
                    "--image-dir", image_path,
                    "--ply-out", ply_path,
                    "--run-dir-out", FASTER_GS_MODELS_DIR,
                    "--max-steps", str(int(config['MAX_STEPS'])),
                    "--isp", config.get('THREED_ISP', 'none').lower(),
                    "--model-name", "faster_gs",
                ]
                if ENABLE_ABSGRAD:
                    args.append("--absgrad")
                pipeline.create_component(
                    name="Train",
                    comp_type=ComponentType.TRAINING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="training/run_faster_gs.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
            # 3DGRUT
            elif config['MODEL'] == "3dgut" or config['MODEL'] == "3dgrt":
                args = [
                    "--config-name", f"apps/colmap_{config['MODEL']}_mcmc.yaml",
                    f"path={config['DATASET_PATH']}",
                    f"out_dir={output_path}",
                    "dataset.downsample_factor=1",
                    "optimizer.type=selective_adam",
                    "export_ply.enabled=true"
                ]
                if config['LOG_VERBOSITY'] != "error":
                    args.append("model.print_stats=true")
                else:
                    args.append("model.print_stats=false")
                if config['RUN_RECON'] == "false" and not colmap_zip_found: # 3dgrut resume training
                    # Validate checkpoint exists and is readable
                    # Check if model_ckpt_path already points to the checkpoint file
                    if os.path.isfile(model_ckpt_path):
                        threedgrut_ckpt_file = model_ckpt_path
                    else:
                        threedgrut_ckpt_file = os.path.join(model_ckpt_path, "ckpt_last.pt")
                    
                    if os.path.exists(threedgrut_ckpt_file):
                        # Read global_step from checkpoint so n_iterations = existing + new steps.
                        # The trainer loops for n_epochs = ceil(n_iterations / dataset_size) and
                        # skips steps where global_step >= n_iterations, so passing only
                        # REFINE_STEPS_3DGRUT would make it stop immediately after loading.
                        try:
                            _ckpt = torch.load(threedgrut_ckpt_file, weights_only=False, map_location="cpu")
                            _ckpt_global_step = int(_ckpt.get("global_step", 0))
                            del _ckpt
                        except Exception as _e:
                            log.warning(f"Could not read global_step from checkpoint: {_e}. Defaulting to 0.")
                            _ckpt_global_step = 0
                        _total_iterations = _ckpt_global_step + REFINE_STEPS_3DGRUT
                        log.info(f"3DGRUT resume: checkpoint global_step={_ckpt_global_step}, "
                                 f"adding {REFINE_STEPS_3DGRUT} steps -> n_iterations={_total_iterations}")
                        args.extend([
                            f"experiment_name={RESUME_TRAIN_EXPERIMENT_NAME}",
                            f"resume={threedgrut_ckpt_file}",
                            f"n_iterations={_total_iterations}",
                            f"scheduler.positions.max_steps={_total_iterations}",
                        ])
                    else:
                        log.warning(
                            f"3DGRUT checkpoint not found: {threedgrut_ckpt_file}"
                            " — falling back to fresh training"
                        )
                        args.extend([
                            f"experiment_name={TRAIN_EXPERIMENT_NAME}",
                            f"n_iterations={str(config['MAX_STEPS'])}",
                            f"scheduler.positions.max_steps={str(config['MAX_STEPS'])}",
                        ])
                else:
                    args.extend([
                        f"experiment_name={TRAIN_EXPERIMENT_NAME}",
                        f"n_iterations={str(config['MAX_STEPS'])}",
                        f"scheduler.positions.max_steps={str(config['MAX_STEPS'])}",
                    ])
                isp_mode = config.get('THREED_ISP', 'none').lower()
                if isp_mode in ("bilagrid", "ppisp"):
                    args.append(f"+model.isp.type={isp_mode}")

                # train.py is patched with DataLoader fix for Batch at build time
                pipeline.create_component(
                    name="Train",
                    comp_type=ComponentType.TRAINING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="3dgrut/train.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
            else:
                error_message = f"Trainer specified does not match proper configuration: {e}"
                pipeline.report_error(760, error_message)
    except Exception as e:
        error_message = f"Issue running the training session stage: {e}"
        pipeline.report_error(765, error_message)

    ##################################
    # TRAINING COMPONENT:
    # Transform checkpoints splat training to .ply
    ##################################
    try:
        # Faster-GS exports its PLY inside the Train component (run_faster_gs.py train)
        if config['MODEL'] != "3dgut" and config['MODEL'] != "3dgrt" and not IS_GSPLAT_DEPTH and not IS_FASTER_GS:
            if ENABLE_MULTI_GPU == "true":
                ckpt_dir = os.path.join(output_path, "ckpts")
                args = [
                    ckpt_dir,
                    ply_path
                ]
                pipeline.create_component(
                    name="Nerfstudio-Export",
                    comp_type=ComponentType.TRAINING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="post_processing/gsplat_pt_to_ply.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
            else:
                if config['MODEL'] == "nerfacto":
                    # Geometry
                    args = [
                        "poisson",
                        "--load-config", model_config_path,
                        "--output-dir", output_path
                    ]
                    pipeline.create_component(
                        name="Nerfstudio-Export",
                        comp_type=ComponentType.TRAINING,
                        comp_environ=ComponentEnvironment.EXECUTABLE,
                        command="ns-export",
                        args=args,
                        cwd=current_dir_path,
                        requires_gpu=False
                    )
                    # Texture
                    args = [
                        "--load-config", model_config_path,
                        "--input-mesh-filename", os.path.join(output_path, "poisson_mesh.ply"),
                        "--output-dir", os.path.join(output_path, "textured")
                    ]
                    pipeline.create_component(
                        name="Nerfstudio-Export-Nerfacto",
                        comp_type=ComponentType.TRAINING,
                        comp_environ=ComponentEnvironment.PYTHON,
                        command="nerfstudio/nerfstudio/scripts/texture.py",
                        args=args,
                        cwd=current_dir_path,
                        requires_gpu=True
                    )
                elif config['MODEL'] == "splatfacto-w-light": 
                    # Use Python wrapper to load gsplat==1.4.0 for splatfacto-w-light export
                    args = [
                        "--load_config", model_config_path,
                        "--output_dir", output_path,
                        "--camera_idx", "0"
                    ]
                    pipeline.create_component(
                        name="Nerfstudio-Export",
                        comp_type=ComponentType.TRAINING,
                        comp_environ=ComponentEnvironment.PYTHON,
                        command="training/run_splatfacto_w_export.py",
                        args=args,
                        cwd=current_dir_path,
                        requires_gpu=True
                    )
                else:
                    # Use correct output path for resume training
                    if resume_training_active:
                        # Resume training - use config from outputs directory (written by nerfstudio after training)
                        config_path = f"outputs/unnamed/{model}/{RESUME_TRAIN_EXPERIMENT_NAME}/config.yml"
                        args = [
                            "gaussian-splat",
                            "--load-config", config_path,
                            "--output-dir", output_path
                        ]
                        log.info(f"Resume training export using config: {config_path}")
                    else:
                        # Initial training - use config from outputs directory
                        config_path = f"outputs/unnamed/{model}/{TRAIN_EXPERIMENT_NAME}/config.yml"
                        args = [
                            "gaussian-splat",
                            "--load-config", config_path,
                            "--output-dir", output_path
                        ]
                    pipeline.create_component(
                        name="Nerfstudio-Export",
                        comp_type=ComponentType.TRAINING,
                        comp_environ=ComponentEnvironment.EXECUTABLE,
                        command="ns-export",
                        args=args,
                        cwd=current_dir_path,
                        requires_gpu=True
                    )
        else:
            log.info("Not configured to output a Gaussian Splat using Nerfstudio...skipping export.")
    except Exception as e:
        error_message = f"Issue exporting splat from NerfStudio: {e}"
        pipeline.report_error(770, error_message)

    ##################################
    # TRAINING COMPONENT:
    # Export PLY from gsplat depth loss checkpoint
    ##################################
    try:
        if IS_GSPLAT_DEPTH and config['MODEL'] != "3dgut" and config['MODEL'] != "3dgrt":
            ckpt_dir = os.path.join(output_path, "ckpts")
            pipeline.create_component(
                name="Nerfstudio-Export",
                comp_type=ComponentType.TRAINING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="post_processing/gsplat_pt_to_ply.py",
                args=[ckpt_dir, ply_path],
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue exporting PLY from gsplat depth loss checkpoint: {e}"
        pipeline.report_error(770, error_message)

    ##################################
    # TRAINING COMPONENT:
    # Generate evaluation metrics
    ##################################
    try:
        # Nerfstudio models (non-multi-GPU)
        if ENABLE_MULTI_GPU == "false":
            if not IS_GSPLAT_DEPTH and (config['MODEL'] in ("splatfacto", "splatfacto-big", "splatfacto-mcmc", "nerfacto", "splatfacto-w-light", "dn-splatter", "dn-splatter-big", "ags-mesh")):
                if resume_training_active:
                    # Resume training - use config from dataset directory for splatfacto models
                    if config['MODEL'] in ["splatfacto", "splatfacto-big", "splatfacto-mcmc"]:
                        config_path = f"outputs/unnamed/splatfacto/{RESUME_TRAIN_EXPERIMENT_NAME}/config.yml"
                    else:
                        # For splatfacto-w-light, use the train-stage-2 config from outputs
                        config_path = f"outputs/unnamed/splatfacto-w-light/{RESUME_TRAIN_EXPERIMENT_NAME}/config.yml"
                else:
                    if config['MODEL'] == "splatfacto-w-light":
                        model = "splatfacto-w-light"
                    elif config['MODEL'] in ("dn-splatter", "dn-splatter-big", "ags-mesh"):
                        model = config['MODEL']
                    else:
                        model = "splatfacto"
                    config_path = f"outputs/unnamed/{model}/{TRAIN_EXPERIMENT_NAME}/config.yml"
                args = [
                    "--load-config", config_path,
                    "--output-path", EVAL_METRIC_PATH
                ]
                # Use wrapper for splatfacto-w-light
                if config['MODEL'] == "splatfacto-w-light":
                    command = "/opt/ml/code/training/run_splatfacto_w_eval.py"
                    comp_environ = ComponentEnvironment.EXECUTABLE
                else:
                    command = "ns-eval"
                    comp_environ = ComponentEnvironment.EXECUTABLE
                pipeline.create_component(
                    name="Nerfstudio-Metrics",
                    comp_type=ComponentType.TRAINING,
                    comp_environ=comp_environ,
                    command=command,
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
            elif config['MODEL'] == "3dgrt" or config['MODEL'] == "3dgut":
                # 3DGRUT
                args = [
                    "--checkpoint", os.path.join(config['DATASET_PATH'], "3dgrut_models", "ckpt_last.pt"),
                    "--out-dir", EVAL_METRIC_FOLDER,
                ]
                pipeline.create_component(
                    name="3DGRUT-Metrics",
                    comp_type=ComponentType.TRAINING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="3dgrut/render.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
            elif IS_FASTER_GS:
                # Converts the held-out test-set metrics NeRFICG writes during training into metrics.json
                pipeline.create_component(
                    name="FasterGS-Metrics",
                    comp_type=ComponentType.TRAINING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="training/run_faster_gs.py",
                    args=["metrics", "--run-dir", FASTER_GS_MODELS_DIR, "--output", EVAL_METRIC_PATH],
                    cwd=current_dir_path,
                    requires_gpu=False
                )
    except Exception as e:
        error_message = f"Issue calculating metrics: {e}"
        pipeline.report_error(771, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Transform splat result to trajectory video
    ##################################
    try:
        if config['ENABLE_VIDEO_EXPORT'] == "true" and ENABLE_MULTI_GPU == "false":
            if not IS_GSPLAT_DEPTH and (config['MODEL'] in ("nerfacto", "splatfacto", "splatfacto-mcmc", "splatfacto-big", "splatfacto-w-light", "dn-splatter", "dn-splatter-big", "ags-mesh")):
                model = "splatfacto"
                if config['MODEL'] == "splatfacto-w-light":
                    model = "splatfacto-w-light"
                if config['MODEL'] == "nerfacto":
                    model = "nerfacto"
                if config['MODEL'] in ("dn-splatter", "dn-splatter-big", "ags-mesh"):
                    model = config['MODEL']
                # Use correct output path for resume training
                if resume_training_active:
                    train_stage = RESUME_TRAIN_EXPERIMENT_NAME
                    if LOCAL_DEBUG:
                        config_path = os.path.join(config['DATASET_PATH'], "config.yml")
                    else:
                        config_path = f"outputs/unnamed/{model}/{train_stage}/config.yml"
                else:
                    train_stage = TRAIN_EXPERIMENT_NAME
                    config_path = f"outputs/unnamed/{model}/{train_stage}/config.yml"
                args = [
                    "interpolate",
                    "--load-config", config_path,
                    "--output-path", os.path.join(output_path, "render.mp4"),
                    "--frame-rate", "24",
                    "--interpolation-steps", "10"
                ]
                # Use wrapper for splatfacto-w-light
                if config['MODEL'] == "splatfacto-w-light":
                    command = "/opt/ml/code/training/run_splatfacto_w_render.py"
                else:
                    command = "ns-render"
                pipeline.create_component(
                    name="Ply-to-Video",
                    comp_type=ComponentType.POST_PROCESSING,
                    comp_environ=ComponentEnvironment.EXECUTABLE,
                    command=command,
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
            elif config['MODEL'] == "3dgut" or config['MODEL'] == "3dgrt":
                args = [
                    "--checkpoint", os.path.join(config['DATASET_PATH'], "3dgrut_models", "ckpt_last.pt"),
                    "--out-dir", output_path,
                ]
                pipeline.create_component(
                    name="Ply-Export-Images",
                    comp_type=ComponentType.POST_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="3dgrut/render.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
                # Convert rendered images to video
                args = [
                    "-i", output_path,
                    "-o", os.path.join(output_path, "render.mp4"),
                    "-r", "2"
                ]
                pipeline.create_component(
                    name="Images-to-Video",
                    comp_type=ComponentType.POST_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="post_processing/images_to_video.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
            elif IS_FASTER_GS:
                # Render the training views in capture order, then encode them to render.mp4
                pipeline.create_component(
                    name="FasterGS-Render-Images",
                    comp_type=ComponentType.POST_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="training/run_faster_gs.py",
                    args=["render", "--run-dir", FASTER_GS_MODELS_DIR, "--subset", "train"],
                    cwd=current_dir_path,
                    requires_gpu=True
                )
                pipeline.create_component(
                    name="Images-to-Video",
                    comp_type=ComponentType.POST_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="post_processing/images_to_video.py",
                    args=[
                        "-i", os.path.join(FASTER_GS_MODELS_DIR, "inference"),
                        "-o", os.path.join(output_path, "render.mp4"),
                        "-r", "24"
                    ],
                    cwd=current_dir_path,
                    requires_gpu=False
                )
    except Exception as e:
        error_message = f"Issue rendering trajectory video: {e}"
        pipeline.report_error(775, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Extract video thumbnail
    ##################################
    try:
        if config['ENABLE_VIDEO_EXPORT'] == "true":
            video_path = os.path.join(output_path, "render.mp4")
            args = [
                "-i", video_path
            ]
            pipeline.create_component(
                name="Extract-Video-Thumbnail",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="post_processing/extract_video_thumbnail.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue extracting video thumbnail: {e}"
        pipeline.report_error(776, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Export VIDEO to S3
    ##################################
    try:
        if config['ENABLE_VIDEO_EXPORT'] == "true":
            args = ["s3", "cp"]
            args.extend([
                os.path.join(output_path, "render.mp4"),
                f"{config['S3_OUTPUT']}/{config['UUID']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.mp4"
            ])
            pipeline.create_component(
                name="S3-Export-Video",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.EXECUTABLE,
                command="aws",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue uploading asset to S3: {e}"
        pipeline.report_error(790, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Upload video thumbnail to S3
    ##################################
    try:
        if config['ENABLE_VIDEO_EXPORT'] == "true":
            thumbnail_path = os.path.join(output_path, "render_thumbnail.png")
            args = ["s3", "cp"]
            args.extend([
                thumbnail_path,
                f"{config['S3_OUTPUT']}/{config['UUID']}/render_thumbnail.png"
            ])
            pipeline.create_component(
                name="S3-Export-Thumbnail",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.EXECUTABLE,
                command="aws",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue uploading asset to S3: {e}"
        pipeline.report_error(790, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Clean point cloud - remove outlier noise from point cloud
    ##################################
    try:
        if config['CLEAN_SPLAT'] == "true":
            if config['MODEL'] != "nerfacto":
                args = [
                    ply_path,
                    ply_path,
                    "--filter-floaters",
                    "-w"
                ]
                pipeline.create_component(
                    name="Clean-Point-Cloud",
                    comp_type=ComponentType.POST_PROCESSING,
                    comp_environ=ComponentEnvironment.EXECUTABLE,
                    command="splat-transform",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
    except Exception as e:
        error_message = f"Issue cleaning point cloud: {e}"
        pipeline.report_error(777, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Crop splat bounds
    ##################################
    try:
        # Apply refinement of output bounds to remove noise if configured
        if config['CROP_OUTPUT_BOUNDS'] == "true" and config['MODEL'] != "nerfacto":   
            args = [
                ply_path,
                ply_path,
                "--log-level", config['LOG_VERBOSITY'].upper(),
                "--mode", config['CROP_MODE']
            ]
            pipeline.create_component(
                name="Crop-Splat",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="post_processing/crop_splat.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue cropping splat bounding box: {e}"
        pipeline.report_error(780, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Remove PLY comments - remove comments for SPZ compatibility
    ##################################
    try:
        if config['MODEL'] != "nerfacto":
            args = [
                "-i", ply_path
            ]
            pipeline.create_component(
                name="Remove-PLY-Comments",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="post_processing/remove_ply_comments.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue removing PLY comments: {e}"
        pipeline.report_error(781, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Create derivative ply files for other exports in order to transform coords for each
    ##################################
    try:
        if config['MODEL'] != "nerfacto":
            args = ["-i", ply_path, "--orig-ply", orig_ply_path]
            if config['ENABLE_SPZ'] == "true":
                args.extend(["--spz-ply", spz_ply_path])
            if config['ENABLE_SOG'] == "true":
                args.extend(["--sog-ply", sog_ply_path])
            if config['ENABLE_USDZ'] == "true":
                args.extend(["--usdz-ply", usdz_ply_path])
            pipeline.create_component(
                name="Create-Derivative-Plys",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="post_processing/create_derivative_plys.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue creating derivative ply files: {e}"
        pipeline.report_error(782, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Transform coordinates for PLY export
    ##################################
    try:
        if config['MODEL'] != "nerfacto":
            args = [
                "-i", orig_ply_path,
                "-o", orig_ply_path,
                "--target", config['PLY_COORDS']
            ]
            pipeline.create_component(
                name="Transform-Coords-Ply",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="post_processing/coordinate_systems.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue transforming coordinates: {e}"
        pipeline.report_error(783, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # PLY rotation (compensate for model coordinate space)
    ##################################
    try:
        if config['MODEL'] != "nerfacto":
            # Faster-GS, like 3DGRUT, outputs in the raw COLMAP world frame (y-down)
            if config['MODEL'] not in ("3dgut", "3dgrt", "faster-gs"):
                # gsplat-depth uses --no-normalize-world-space (raw COLMAP/OpenCV space)
                # nerfstudio models use OpenGL-normalized space
                rotation = '-90,0,0' if IS_GSPLAT_DEPTH else '270,0,180'
            else:
                # 3DGRUT outputs need 180° on right and up axes for PlayCanvas viewer
                rotation = '180,180,0'
            if rotation:
                args = [
                    orig_ply_path,
                    orig_ply_path,
                    f"--rotate={rotation}",
                    '--overwrite'
                ]
                pipeline.create_component(
                    name="Ply-Rotation",
                    comp_type=ComponentType.POST_PROCESSING,
                    comp_environ=ComponentEnvironment.EXECUTABLE,
                    command="splat-transform",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
    except Exception as e:
        error_message = f"Issue rotating PLY: {e}"
        pipeline.report_error(785, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Export PLY to S3
    ##################################
    try:
        args = ["s3", "cp"]
        if config['MODEL'] == "nerfacto":
            glb_path = os.path.join(output_path, "textured", f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.glb")
            args.extend([
                glb_path,
                f"{config['S3_OUTPUT']}/{config['UUID']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.glb"
            ])
        else:
            args.extend([
                orig_ply_path,
                f"{config['S3_OUTPUT']}/{config['UUID']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.ply"
            ])
        pipeline.create_component(
            name="S3-Export-Ply",
            comp_type=ComponentType.POST_PROCESSING,
            comp_environ=ComponentEnvironment.EXECUTABLE,
            command="aws",
            args=args,
            cwd=current_dir_path,
            requires_gpu=False
        )
    except Exception as e:
        error_message = f"Issue uploading asset to S3: {e}"
        pipeline.report_error(790, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Transform coordinates for SOG export
    ##################################
    try:
        if config['MODEL'] != "nerfacto" and config['ENABLE_SOG'] == "true":
            args = [
                "-i", sog_ply_path,
                "-o", sog_ply_path,
                "--target", config['SOG_COORDS']
            ]
            pipeline.create_component(
                name="Transform-Coords-Sog",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="post_processing/coordinate_systems.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue transforming coordinates: {e}"
        pipeline.report_error(783, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Rotate PLY for SOG
    ##################################
    try:
        if config['ENABLE_SOG'] == "true" and config['MODEL'] != "nerfacto":
            if config['MODEL'] not in ("3dgut", "3dgrt", "faster-gs"):
                # gsplat-depth uses --no-normalize-world-space (raw COLMAP/OpenCV space)
                # nerfstudio models use OpenGL-normalized space
                rotation = '-90,0,0' if IS_GSPLAT_DEPTH else '270,0,180'
            else:
                # 3DGRUT outputs need 180° on right and up axes for PlayCanvas viewer
                rotation = '180,180,0'
            if rotation:
                args = [
                    sog_ply_path,
                    sog_ply_path,
                    f"--rotate={rotation}",
                    '--overwrite'
                ]
                pipeline.create_component(
                    name="Rotate-PLY-For-SOG",
                    comp_type=ComponentType.POST_PROCESSING,
                    comp_environ=ComponentEnvironment.EXECUTABLE,
                    command="splat-transform",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
    except Exception as e:
        error_message = f"Issue rotating PLY: {e}"
        pipeline.report_error(785, error_message)
    
    ##################################
    # POST-PROCESS COMPONENT:
    # Transform .ply to sog for compressed web viewing
    ##################################
    try:
        if config['MODEL'] != "nerfacto" and config['ENABLE_SOG'] == "true":
            args = [
                "-i", sog_ply_path,
                "-o", sog_path,
                "-w"
            ]
            pipeline.create_component(
                name="SOG-Export",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="post_processing/convert_ply_to_sog.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue converting ply to SOG: {e}"
        pipeline.report_error(786, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Export SOG to S3
    ##################################
    try:
        if config['ENABLE_SOG'] == "true" and config['MODEL'] != "nerfacto":
            args = ["s3", "cp"]
            args.extend([
                sog_path,
                f"{config['S3_OUTPUT']}/{config['UUID']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.sog"
            ])
            pipeline.create_component(
                name="S3-Export-Sog",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.EXECUTABLE,
                command="aws",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
        else:
            log.info(
                "Not configured to output a SOG...skipping upload splat to S3."
                "Check the archive file for reconstruction results"
            )
    except Exception as e:
        error_message = f"Issue uploading asset to S3: {e}"
        pipeline.report_error(790, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Transform coordinates for USDZ export
    ##################################
    try:
        if config['MODEL'] != "nerfacto" and config['ENABLE_USDZ'] == "true":
            args = [
                "-i", usdz_ply_path,
                "-o", usdz_ply_path,
                "--target", config['USDZ_COORDS']
            ]
            pipeline.create_component(
                name="Transform-Coords-Sog",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="post_processing/coordinate_systems.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue transforming coordinates: {e}"
        pipeline.report_error(783, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Transform .ply to USDZ for compressed web viewing
    ##################################
    try:
        if config['ENABLE_USDZ'] == "true" and config['MODEL'] != "nerfacto":
            # Set PYTHONPATH to include 3dgrut directory
            threedgrut_path = os.path.join(current_dir_path, "3dgrut")
            os.environ['PYTHONPATH'] = f"{threedgrut_path}:{os.environ.get('PYTHONPATH', '')}"
            args = [
                usdz_ply_path,
                "--output_file", usdz_path,
            ]
            pipeline.create_component(
                name="USDZ-Export",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="3dgrut/threedgrut/export/scripts/ply_to_usd.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=True
            )
    except Exception as e:
        error_message = f"Issue converting ply to USDZ: {e}"
        pipeline.report_error(787, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Add collision mesh to USDZ
    ##################################
    try:
        if config['ENABLE_USDZ'] == "true" and config['MODEL'] != "nerfacto":
            threedgrut_path = os.path.join(current_dir_path, "3dgrut")
            os.environ['PYTHONPATH'] = f"{threedgrut_path}:{os.environ.get('PYTHONPATH', '')}"
            # Step 1: generate convex hull triangle mesh from the splat point positions
            pipeline.create_component(
                name="Create-Collision-Mesh",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="post_processing/create_collision_mesh.py",
                args=["-i", usdz_ply_path, "-o", collision_mesh_path],
                cwd=current_dir_path,
                requires_gpu=False
            )
            # Step 2: embed the mesh into the USDZ as invisible collision geometry
            usdz_tmp_path = os.path.join(output_path, "splat_with_mesh.usdz")
            pipeline.create_component(
                name="USDZ-Add-Collision-Mesh",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="3dgrut/threedgrut/export/scripts/add_mesh_to_usdz.py",
                args=[
                    "--input_usdz", usdz_path,
                    "--output_usdz", usdz_tmp_path,
                    "--mesh_ply", collision_mesh_path,
                    "--set_collision",
                    "--set_invisible"
                ],
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue adding collision mesh to USDZ: {e}"
        pipeline.report_error(787, error_message)
    
    ##################################
    # POST-PROCESS COMPONENT:
    # Export USDZ to S3
    ##################################
    try:
        if config['ENABLE_USDZ'] == "true" and config['MODEL'] != "nerfacto":
            args = ["s3", "cp"]
            args.extend([
                usdz_path,
                f"{config['S3_OUTPUT']}/{config['UUID']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.usdz"
            ])
            pipeline.create_component(
                name="S3-Export-Usdz",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.EXECUTABLE,
                command="aws",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue uploading asset to S3: {e}"
        pipeline.report_error(790, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Transform coordinates for SPZ export
    ##################################
    try:
        if config['MODEL'] != "nerfacto" and config['ENABLE_SPZ'] == "true":
            is_gsplat = ENABLE_MULTI_GPU == "true" or IS_GSPLAT_DEPTH
            preserve_scale = config.get('PRESERVE_SCENE_SCALE', 'false').lower() == 'true'
            # For normalized gsplat, spz.ply is copied from orig.ply (already coord-transformed
            # and rotated), so skip the coord transform step entirely.
            if not (is_gsplat and not preserve_scale):
                args = [
                    "-i", spz_ply_path,
                    "-o", spz_ply_path,
                    "--target", config['SPZ_COORDS']
                ]
                pipeline.create_component(
                    name="Transform-Coords-Sog",
                    comp_type=ComponentType.POST_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="post_processing/coordinate_systems.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
    except Exception as e:
        error_message = f"Issue transforming coordinates: {e}"
        pipeline.report_error(783, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # PLY rotation (compensate for model SPZ coordinate system)
    ##################################
    try:
        if config['MODEL'] != "nerfacto" and config['ENABLE_SPZ'] == "true":
            if config['MODEL'] not in ("3dgut", "3dgrt", "faster-gs"):
                is_lhyu = config.get('SPZ_COORDS', 'rhyu') == 'lhyu'
                preserve_scale = config.get('PRESERVE_SCENE_SCALE', 'false').lower() == 'true'
                if (ENABLE_MULTI_GPU == "true" or IS_GSPLAT_DEPTH) and preserve_scale:
                    # gsplat raw COLMAP space (--no-normalize-world-space)
                    rotation = '-90,0,180' if is_lhyu else '-90,0,0'
                elif ENABLE_MULTI_GPU == "true" or IS_GSPLAT_DEPTH:
                    # gsplat normalized: spz.ply copied from orig.ply (rhyu space)
                    # Apply 180,0,0 to match the orientation Babylon.js SPZ expects
                    rotation = '180,0,0'
                else:
                    # nerfstudio OpenGL-space
                    rotation = '90,0,180' if is_lhyu else '270,0,0'
            else:
                # 3DGRUT: 180° on up axis for PlayCanvas viewer
                rotation = '0,180,0'
            if rotation:
                args = [
                    spz_ply_path,
                    spz_ply_path,
                    f"--rotate={rotation}",
                ]
                # When PRESERVE_SCENE_SCALE is true the splat is in real-world metric scale.
                # Read the dataparser_scale from nerfstudio config.yml and apply its inverse
                # so the SPZ viewer sees a normalized (~1-unit) scene instead of a zoomed-out one.
                preserve_scale = config.get('PRESERVE_SCENE_SCALE', 'false').lower() == 'true'
                if preserve_scale and config['MODEL'] not in ('3dgut', '3dgrt', 'nerfacto', 'faster-gs'):
                    import yaml as _yaml
                    _config_yml = os.path.join(
                        config['CODE_PATH'], 'outputs', 'unnamed',
                        config['MODEL'], TRAIN_EXPERIMENT_NAME, 'config.yml'
                    )
                    _dp_scale = None
                    if os.path.exists(_config_yml):
                        try:
                            with open(_config_yml, 'r') as _f:
                                _ns_cfg = _yaml.safe_load(_f)
                            _dp_scale = (_ns_cfg.get('pipeline', {}).get('datamanager', {})
                                         .get('dataparser', {}).get('scale_factor'))
                        except Exception as _e:
                            log.warning(f"Could not read dataparser scale from config.yml: {_e}")
                    if _dp_scale and float(_dp_scale) > 0:
                        args.append(f"--scale={float(_dp_scale):.6f}")
                        log.info(f"SPZ preserve_scene_scale: applying --scale={_dp_scale} from config.yml")
                args.append('--overwrite')
                pipeline.create_component(
                    name="Spz-Ply-Rotation",
                    comp_type=ComponentType.POST_PROCESSING,
                    comp_environ=ComponentEnvironment.EXECUTABLE,
                    command="splat-transform",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
    except Exception as e:
        error_message = f"Issue rotating PLY: {e}"
        pipeline.report_error(785, error_message)
    
    ##################################
    # POST-PROCESS COMPONENT:
    # Transform PLY to compressed SPZ splat file
    ##################################
    try:
        if config['MODEL'] != "nerfacto" and config['ENABLE_SPZ'] == "true":
            args = [
                spz_ply_path,
                spz_path
            ]
            pipeline.create_component(
                name="Ply-to-Spz",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.EXECUTABLE,
                command="ply_to_spz",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue converting ply to SPZ: {e}"
        pipeline.report_error(788, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Export SPZ to S3
    ##################################
    try:
        if config['MODEL'] != "nerfacto" and config['ENABLE_SPZ'] == "true":
            args = ["s3", "cp", "--content-type", "application/octet-stream"]
            if config['MODEL'] != "nerfacto":
                args.extend([
                    spz_path,
                    f"{config['S3_OUTPUT']}/{config['UUID']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.spz"
                ])
            pipeline.create_component(
                name="S3-Export-Spz",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.EXECUTABLE,
                command="aws",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue uploading asset to S3: {e}"
        pipeline.report_error(790, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Generate collision voxel data from splat
    ##################################
    try:
        if GENERATE_COLLISION and config['MODEL'] != "nerfacto":
            args = [
                "-i", orig_ply_path,
                "-o", voxel_path,
                "--scene-type", config.get('COLLISION_SCENE_TYPE', 'outdoor'),
                "--seed-pos", config.get('COLLISION_SEED_POS', '0,0,0'),
            ]
            pipeline.create_component(
                name="Generate-Collision",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="post_processing/generate_collision.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=True
            )
    except Exception as e:
        error_message = f"Issue creating collision generation component: {e}"
        pipeline.report_error(800, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Export collision zip to S3
    ##################################
    try:
        if GENERATE_COLLISION and config['MODEL'] != "nerfacto":
            base_name = str(os.path.splitext(config['FILENAME'])[0]).lower()
            collision_zip_path = os.path.join(output_path, f"{base_name}-collision.zip")
            pipeline.create_component(
                name="S3-Export-Collision",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.EXECUTABLE,
                command="aws",
                args=["s3", "cp", collision_zip_path,
                      f"{config['S3_OUTPUT']}/{config['UUID']}/{base_name}-collision.zip"],
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue uploading collision data to S3: {e}"
        pipeline.report_error(800, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Generate streamed LOD SOG bundle from splat
    ##################################
    try:
        if GENERATE_LOD and config['MODEL'] != "nerfacto":
            args = [
                "-i", orig_ply_path,
                "--output-dir", lod_dir,
            ]
            pipeline.create_component(
                name="Generate-LOD",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="post_processing/generate_lod.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue creating LOD generation component: {e}"
        pipeline.report_error(801, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Export LOD zip to S3
    ##################################
    try:
        if GENERATE_LOD and config['MODEL'] != "nerfacto":
            base_name = str(os.path.splitext(config['FILENAME'])[0]).lower()
            lod_zip_path = os.path.join(output_path, f"{base_name}-lod.zip")
            pipeline.create_component(
                name="S3-Export-LOD",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.EXECUTABLE,
                command="aws",
                args=["s3", "cp", lod_zip_path,
                      f"{config['S3_OUTPUT']}/{config['UUID']}/{base_name}-lod.zip"],
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue uploading LOD bundle to S3: {e}"
        pipeline.report_error(801, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Extract mesh (GENERATE_MESH). Both extractors write <output_path>/mesh.glb,
    # which the single S3-Export-Mesh component below uploads.
    #   - dn-splatter / dn-splatter-big / ags-mesh: native gs-mesh o3dtsdf extractor
    #   - other splat models: Gaussian-splat TSDF extractor with texture baking
    #   - nerfacto / 3dgrt / 3dgut / faster-gs: not supported
    ##################################
    MESH_DN_MODELS = ("dn-splatter", "dn-splatter-big", "ags-mesh")
    # faster-gs: Extract-Mesh-GS expects a nerfstudio-space PLY + transforms.json; Faster-GS
    # exports in the raw COLMAP frame without transforms.json, so it is not supported yet.
    MESH_UNSUPPORTED_MODELS = ("nerfacto", "3dgrt", "3dgut", "faster-gs")
    try:
        if GENERATE_MESH and config['MODEL'] in MESH_DN_MODELS:
            mesh_ply_path = os.path.join(output_path, "mesh.ply")
            mesh_glb_path = os.path.join(output_path, "mesh.glb")
            model_config_path = f"outputs/unnamed/{config['MODEL']}/{TRAIN_EXPERIMENT_NAME}/config.yml"
            mesh_method = "o3dtsdf"
            args = [
                "--config-path", model_config_path,
                "--output-ply", mesh_ply_path,
                "--output-glb", mesh_glb_path,
                "--method", mesh_method,
                "--poisson-depth", "9",
                "--total-points", "2000000",
                "--voxel-size", config.get('MESH_VOXEL_SIZE', '0.004'),
                "--sdf-trunc", config.get('MESH_SDF_TRUNC', '0.02'),
            ]
            pipeline.create_component(
                name="Extract-Mesh",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="post_processing/extract_mesh.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=True
            )
        elif GENERATE_MESH and config['MODEL'] not in MESH_UNSUPPORTED_MODELS:
            transforms_path = os.path.join(config['DATASET_PATH'], "transforms.json")
            args = [
                "--ply", ply_path,
                "--transforms", transforms_path,
                "--output-dir", output_path,
                "--name", "mesh",
            ]
            pipeline.create_component(
                name="Extract-Mesh-GS",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.PYTHON,
                command="post_processing/extract_mesh_gs.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=True
            )
        elif GENERATE_MESH:
            log.info(f"GENERATE_MESH: mesh extraction is not supported for model {config['MODEL']}, skipping")
    except Exception as e:
        error_message = f"Issue creating mesh extraction component: {e}"
        pipeline.report_error(802, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Export mesh GLB to S3
    ##################################
    try:
        if GENERATE_MESH and config['MODEL'] not in MESH_UNSUPPORTED_MODELS:
            mesh_glb_path = os.path.join(output_path, "mesh.glb")
            base_name = str(os.path.splitext(config['FILENAME'])[0]).lower()
            pipeline.create_component(
                name="S3-Export-Mesh",
                comp_type=ComponentType.POST_PROCESSING,
                comp_environ=ComponentEnvironment.EXECUTABLE,
                command="aws",
                args=["s3", "cp", mesh_glb_path,
                      f"{config['S3_OUTPUT']}/{config['UUID']}/{base_name}.mesh.glb"],
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue uploading mesh GLB to S3: {e}"
        pipeline.report_error(803, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Create and upload model.tar.gz archive to S3
    ##################################
    try:
        args = [
            "s3"
        ]
        if IS_BATCH:
            args.extend([
                "cp",
                OUTPUT_TAR_PATH,
                f"{config['S3_OUTPUT']}/{config['UUID']}/output/model.tar.gz"
            ])
        else:
            args.extend(["ls"])
        pipeline.create_component(
            name="S3-Export-Archive",
            comp_type=ComponentType.POST_PROCESSING,
            comp_environ=ComponentEnvironment.EXECUTABLE,
            command="aws",
            args=args,
            cwd=current_dir_path,
            requires_gpu=False
        )
    except Exception as e:
        error_message = f"Issue uploading asset to S3: {e}"
        pipeline.report_error(790, error_message)

    ##################################
    # RUN THE PIPELINE W/ COMPONENTS LOGIC
    ##################################
    try:
        pipeline.session.status = Status.RUNNING
        log.info(f"Pipeline status changed to {pipeline.session.status}")
        start_time = int(time.time())

        # Start background heartbeat thread for long-running Batch jobs.
        # Fires every 6 hours so Step Functions never hits HeartbeatSeconds=172800.
        # Only active when IS_BATCH=True, TASK_TOKEN is set, and not LOCAL_DEBUG.
        if ENABLE_TASK_TOKEN_CALLBACK:
            _heartbeat_stop = _threading.Event()
            def _heartbeat_loop():
                # Send an immediate heartbeat on startup, then every 6 hours
                send_task_heartbeat(TASK_TOKEN, log)
                while not _heartbeat_stop.wait(timeout=21600):
                    send_task_heartbeat(TASK_TOKEN, log)
            _heartbeat_thread = _threading.Thread(target=_heartbeat_loop, daemon=True)
            _heartbeat_thread.start()
            log.info("Heartbeat thread started (interval=6h, max=48h)")

        # Initialize phase tracking
        pipeline.session.comp_group_names = [member.name for member in ComponentType]
        log.info(f"Component groups: {pipeline.session.comp_group_names}")
        pipeline.session.comp_group_elapsed_time = []
        pipeline.session.comp_start_names = []
        phase_start_times = {}
        phase_started = {comp_type: False for comp_type in ComponentType}
        last_phase = None
        ddb_table_name = os.environ.get('DDB_TABLE_NAME')
        log.info(f"DDB_TABLE_NAME from environment: {ddb_table_name}")
        
        i = 0
        while i < len(pipeline.components):
            component = pipeline.components[i]
            log.info(f"Running component: {component.name}")
            
            # Track phase start times - only when component actually runs
            if not phase_started[component.comp_type]:
                # Mark phase as started before checking if component will run
                phase_started[component.comp_type] = True
                
                # Write completion of previous phase if exists and DDB table is configured
                if last_phase is not None and ddb_table_name and last_phase in phase_start_times:
                    phase_elapsed = int(time.time()) - phase_start_times[last_phase]
                    log.info(f"Writing phase completion for {last_phase}: {phase_elapsed}s to DDB table {ddb_table_name}")
                    update_component_phase_completion(
                        uuid=config['UUID'],
                        table_name=ddb_table_name,
                        phase_name=last_phase,
                        elapsed_time=phase_elapsed,
                        log=log
                    )
                elif last_phase is not None and not ddb_table_name:
                    log.debug(f"Skipping DynamoDB update for {last_phase} - no table name configured")
                
                pipeline.session.comp_start_names.append(component.name)
                phase_start_times[component.comp_type.name] = int(time.time())
                log.info(f"{component.comp_type.name} started")
                last_phase = component.comp_type.name
            match component.name:
                case "DN-Splatter-Preprocess":
                    # --has-sensor-depth is no longer used by dn_splatter_preprocess.py.
                    # Sensor depth validation and depth_file_path injection happens
                    # internally in preprocess. Just run the component.
                    pipeline.run_component(i)
                case "VideoToImages":
                    if config['RUN_RECON'] == "false" and config['RUN_TRAIN'] == "false":
                        continue
                    video_found = False
                    # Check if input is a directory with files (for folder input mode)
                    IS_FOLDER_INPUT = os.path.isdir(input_file_path) and len(os.listdir(input_file_path)) > 0
                    # Skip if COLMAP reconstruction was already provided
                    if colmap_zip_found:
                        log.info("Skipping VideoToImages - COLMAP reconstruction already provided")
                        i += 1
                        continue
                    if (VIDEO is False and config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true') or \
                        (VIDEO is False and config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true'):
                        continue
                    else:
                        if (VIDEO is True and config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'false') or \
                            (VIDEO is True and config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'false'):
                            pipeline.run_component(i)
                            # Update VIDEO flag back to False after processing video to images
                            VIDEO = False
                        elif VIDEO is False and config['RUN_RECON'] == "false" and config['RUN_TRAIN'] == "true":
                            # Resume training - extract model.tar.gz
                            log.info("Detected resume training...")
                        elif IS_FOLDER_INPUT:
                            # Folder input - copy directly to images directory
                            log.info(f"Detected folder input: {input_file_path}")
                            _structured_folder = image_path != input_file_path and (
                                RIG_CONFIG or UPLOADED_IMAGE_MASKS or
                                os.path.isdir(os.path.join(input_file_path, 'images')))
                            if _structured_folder:
                                # Unzipped upload (images/, masks/, rig_config.json, manifest.json,
                                # priors.csv): handle it exactly like the zip. Work on a copy so
                                # the input folder is left unchanged.
                                temp_path = os.path.join(config['DATASET_PATH'], 'temp')
                                if os.path.exists(temp_path):
                                    shutil.rmtree(temp_path)
                                shutil.copytree(input_file_path, temp_path)
                                if os.path.exists(image_path):
                                    shutil.rmtree(image_path)
                                extract_images_from_zip_temp(
                                    temp_path, image_path, config['DATASET_PATH'], log,
                                    keep_subfolders=RIG_CONFIG
                                )
                                shutil.rmtree(temp_path)
                                if RIG_CONFIG and not os.path.isfile(rig_config_path):
                                    log.warning("rig_config.json was not found; "
                                                "the cameras are solved without a rig")
                            elif image_path != input_file_path:
                                if os.path.exists(image_path):
                                    shutil.rmtree(image_path)
                                shutil.copytree(input_file_path, image_path)
                                log.info(f"Copied {len(os.listdir(image_path))} files from folder to images directory")
                            else:
                                log.info(f"Using {len(os.listdir(image_path))} files from folder (already in place)")
                            if GPS_PRIORS:
                                for _kf in ('manifest.json', 'priors.csv'):
                                    _src_kf = os.path.join(input_file_path, _kf)
                                    if os.path.isfile(_src_kf) and not os.path.exists(
                                            os.path.join(config['DATASET_PATH'], _kf)):
                                        shutil.copy2(_src_kf, config['DATASET_PATH'])
                                read_gps_priors(gps_priors_script, image_path, config['DATASET_PATH'],
                                                gps_priors_json, log)
                            if config.get('AUTOSCALE_DATASET', 'false') != 'true':
                                validate_and_resize_images(image_path, config, log, pipeline)
                                resize_images_to_common_dimensions(image_path)
                        else: # Archive of images or archive with a video
                            # unzip archive into temp directory
                            temp_path = os.path.join(config['DATASET_PATH'], 'temp')
                            _fixed = safe_extract_zip(input_file_path, temp_path)
                            if _fixed:
                                log.warning(f"Zip used Windows '\\' path separators in {_fixed} entries; normalized to '/'")
                            
                            # Check for video files first (.mov, .mp4)
                            all_files = []
                            for root, dirs, files in os.walk(temp_path):
                                for file in files:
                                    all_files.append(os.path.join(root, file))
                                    if file.lower().endswith(('.mov', '.mp4')):
                                        # Found video - move to dataset root and update VIDEO flag
                                        video_path = os.path.join(root, file)
                                        new_video_path = os.path.join(config['DATASET_PATH'], file)
                                        shutil.move(video_path, new_video_path)
                                        config['FILENAME'] = file
                                        VIDEO = True
                                        video_found = True
                                        log.info(f"Found video in zip: {file}, updated VIDEO flag to True")
                                        break
                                if video_found:
                                    break
                            if not video_found:
                                # No video found - only applies when run_recon == true.
                                if config['RUN_RECON'] == 'true':
                                    extract_images_from_zip_temp(
                                        temp_path, image_path, config['DATASET_PATH'], log,
                                        keep_subfolders=RIG_CONFIG
                                    )
                                    if RIG_CONFIG and not os.path.isfile(rig_config_path):
                                        log.warning("rig_config.json was not extracted; "
                                                    "the cameras are solved without a rig")
                                    # Before any resize/rotation rewrites the images.
                                    if GPS_PRIORS:
                                        read_gps_priors(gps_priors_script, image_path,
                                                        config['DATASET_PATH'], gps_priors_json, log)
                                else:
                                    # run_recon == false: pre-processing already done,
                                    # move contents as-is (original behaviour)
                                    temp_dir_input = os.listdir(temp_path)[0]
                                    if os.path.isdir(os.path.join(temp_path, temp_dir_input)):
                                        if os.path.exists(image_path):
                                            shutil.rmtree(image_path)
                                        os.rename(os.path.join(temp_path, temp_dir_input), image_path)
                                    else:
                                        for filename in os.listdir(temp_path):
                                            shutil.move(
                                                os.path.join(temp_path, filename),
                                                os.path.join(image_path, filename)
                                            )
                            
                            # Clean up temp directory
                            if os.path.exists(temp_path):
                                shutil.rmtree(temp_path)
                        if config['RUN_RECON'] == "true" and config['RUN_TRAIN'] == "true" and \
                                config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'false' and \
                                config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'false' and \
                                not colmap_zip_found:
                            # Only process images if no video was found and not resuming/skipping recon
                            if not video_found:
                                if config.get('AUTOSCALE_DATASET', 'false') != 'true':
                                    validate_and_resize_images(image_path, config, log, pipeline)
                                    resize_images_to_common_dimensions(image_path)
                            else:
                                # Video found - update component args with correct video path and run
                                log.info("Video found in zip - running VideoToImages component")
                                # Update the input path in component args to point to extracted video
                                new_video_path = os.path.join(config['DATASET_PATH'], config['FILENAME'])
                                for j, arg in enumerate(component.args):
                                    if arg == "-i" and j + 1 < len(component.args):
                                        component.args[j + 1] = new_video_path
                                        log.info(f"Updated video input path to: {new_video_path}")
                                        break
                                pipeline.run_component(i)
                case "RemoveHumanSubjectMask":
                    # REMOVE HUMAN SUBJECT CONDITIONAL COMPONENT
                    # Run Component
                    pipeline.run_component(i)
                    # Rename the masked image directory to images, but only if it holds
                    # every image: replacing images/ with an incomplete (e.g. empty)
                    # output deleted the whole dataset.
                    def _count_imgs(_d):
                        return sum(1 for _r, _, _fs in os.walk(_d) for _f in _fs
                                   if _f.lower().endswith(('.png', '.jpg', '.jpeg')))
                    _n_in, _n_out = _count_imgs(image_path), _count_imgs(mask_human_output_dir)
                    if _n_in and _n_out >= _n_in:
                        shutil.rmtree(image_path)
                        os.rename(mask_human_output_dir, image_path)
                        log.info("All images successfully processed with human subject remover")
                    else:
                        log.warning(f"Human subject remover produced {_n_out} of {_n_in} images; "
                                    f"keeping the original images")
                case "ColmapSfM-Feature-Extractor":
                    # COLMAP FEATURE EXTRACTOR CONDITIONAL COMPONENT
                    log.info("Using standard COLMAP feature extraction")
                    # Uploaded masks: both file names, at the final image size
                    # (resize/autoscale/rotation may have run since extraction).
                    if UPLOADED_IMAGE_MASKS:
                        normalize_uploaded_masks(image_path, os.path.join(config['DATASET_PATH'], 'masks'), log)
                    # Session manifest camera (GPS jobs): exact shared intrinsics, scaled
                    # to the current image size and kept fixed in the mapper.
                    _gps_camera = None
                    if GPS_PRIORS and os.path.isfile(gps_priors_json) and \
                            config.get('ENABLE_FL_METRIC', 'false') != 'true' and \
                            config.get('ENABLE_FL_HEURISTIC', 'false') != 'true':
                        try:
                            with open(gps_priors_json, encoding='utf-8') as _f:
                                _cam = (json.load(_f) or {}).get('camera')
                            _imgs = sorted(os.path.relpath(os.path.join(_r, f), image_path)
                                           for _r, _, _fs in os.walk(image_path) for f in _fs
                                           if f.lower().endswith(('.png', '.jpg', '.jpeg')))
                            if _cam and _imgs and str(_cam.get('model', '')).upper() in ('PINHOLE', 'SIMPLE_PINHOLE'):
                                with Image.open(os.path.join(image_path, _imgs[0])) as _img:
                                    _w, _h = _img.size
                                _sx = _w / float(_cam['width'])
                                _sy = _h / float(_cam.get('height', _cam['width']))
                                if abs(_sx - _sy) < 1e-3:
                                    _fx = float(_cam['fx']) * _sx
                                    _fy = float(_cam.get('fy', _cam['fx'])) * _sy
                                    _cx = float(_cam.get('cx', _cam['width'] / 2.0)) * _sx
                                    _cy = float(_cam.get('cy', _cam.get('height', _cam['width']) / 2.0)) * _sy
                                    _gps_camera = ("PINHOLE", f"{_fx:.6f},{_fy:.6f},{_cx:.6f},{_cy:.6f}")
                                    log.info(f"Manifest camera scaled to {_w}x{_h}: PINHOLE {_gps_camera[1]}")
                                else:
                                    log.warning(f"Manifest camera {_cam['width']}x{_cam.get('height')} does not "
                                                f"match the image aspect ({_w}x{_h}); COLMAP estimates intrinsics")
                        except Exception as _e:
                            log.warning(f"Could not use the manifest camera: {_e}")
                    # Camera rig: use the rig config's intrinsics from the start when all
                    # cameras share one model and parameters (e.g. pinhole slices of a 360
                    # camera). Otherwise COLMAP starts from SIMPLE_RADIAL; rig_configurator
                    # then overwrites the cameras that list camera_model_name/camera_params.
                    if RIG_CONFIG and os.path.isfile(rig_config_path):
                        try:
                            with open(rig_config_path, encoding='utf-8') as _f:
                                _rig_cams = [_c for _r in json.load(_f) for _c in _r.get('cameras', [])]
                            _rig_models = {(str(_c.get('camera_model_name', '')).upper(),
                                            tuple(float(_p) for _p in _c.get('camera_params') or []))
                                           for _c in _rig_cams}
                            if len(_rig_models) == 1:
                                _m, _ps = next(iter(_rig_models))
                                if _m and _ps:
                                    _gps_camera = (_m, ",".join(f"{_p:.6f}" for _p in _ps))
                                    log.info(f"Camera rig: feature extraction with the rig config's "
                                             f"camera {_m} {_gps_camera[1]}")
                        except Exception as _e:
                            log.warning(f"Could not read the rig config cameras: {_e}")
                    if _gps_camera is not None:
                        try:
                            idx = component.args.index("--ImageReader.camera_model")
                            component.args.pop(idx)
                            component.args.pop(idx)
                        except ValueError:
                            pass
                        component.args.extend([
                            "--ImageReader.camera_model", _gps_camera[0],
                            "--ImageReader.camera_params", _gps_camera[1]
                        ])
                        for _c in pipeline.components:
                            if _c.name == "ColmapSfM-Mapper" and "--Mapper.ba_refine_focal_length" not in _c.args:
                                _c.args.extend([
                                    "--Mapper.ba_refine_focal_length", "0",
                                    "--Mapper.ba_refine_principal_point", "0",
                                    "--Mapper.ba_refine_extra_params", "0"
                                ])
                    elif config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == "true" or \
                        config['USE_POSE_PRIOR_TRANSFORM_JSON'] == "true":
                        # Pose priors: use the intrinsics from the model files
                        # Prefer cameras.bin/txt from existing COLMAP reconstruction
                        camera_params = read_camera_params_from_file(
                            os.path.join(sparse_model_path, "cameras.txt")
                        )
                        if camera_params is None and config['USE_POSE_PRIOR_TRANSFORM_JSON'] == "true":
                            # Fall back to transforms.json intrinsics
                            try:
                                _tf_path = transforms_in_path
                                if not os.path.exists(_tf_path):
                                    _tf_path = transforms_out_path
                                if os.path.exists(_tf_path):
                                    import json as _json_tf
                                    with open(_tf_path, 'r') as _f:
                                        _tf = _json_tf.load(_f)
                                    _src = _tf
                                    if 'fl_x' not in _src and _tf.get('frames'):
                                        _src = _tf['frames'][0]
                                    _fl_x = _src.get('fl_x')
                                    _fl_y = _src.get('fl_y', _fl_x)
                                    _cx = _src.get('cx')
                                    _cy = _src.get('cy')
                                    if _fl_x is not None and _cx is not None:
                                        _k1 = float(_src.get('k1', 0))
                                        _k2 = float(_src.get('k2', 0))
                                        _p1 = float(_src.get('p1', 0))
                                        _p2 = float(_src.get('p2', 0))
                                        if any(v != 0 for v in (_k1, _k2, _p1, _p2)):
                                            _cam = "OPENCV"
                                            _ps = f"{_fl_x},{_fl_y},{_cx},{_cy},{_k1},{_k2},{_p1},{_p2}"
                                        else:
                                            _cam = "PINHOLE"
                                            _ps = f"{_fl_x},{_fl_y},{_cx},{_cy}"
                                        camera_params = {'model': _cam, 'params_str': _ps}
                                        log.info(f"Camera from transforms.json: {_cam}")
                            except Exception as _e:
                                log.warning(f"Could not read camera from transforms.json: {_e}")
                        if camera_params is not None:
                            component.args.extend([
                                "--ImageReader.camera_model", camera_params['model'],
                                "--ImageReader.camera_params", camera_params['params_str']
                            ])
                            log.info(f"Feature extractor camera: {camera_params['model']}")
                        else:
                            log.warning("Could not read camera params — using COLMAP defaults")
                    elif config.get('ENABLE_FL_METRIC', 'false') == 'true':
                        # Convert metric focal length from mm to pixels using the 35mm-equivalent
                        # formula: focal_px = (focal_mm / 36.0) * image_width_px
                        # This avoids needing the physical sensor size and correctly accounts
                        # for any image resizing (e.g. 4K downscale) since _w is read from
                        # the actual image on disk after pre-processing.
                        try:
                            img_files = [f for f in os.listdir(image_path)
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                            if img_files:
                                with Image.open(os.path.join(image_path, img_files[0])) as _img:
                                    _w, _h = _img.size
                                fl_mm = float(config.get('FL_METRIC_VALUE', '24'))
                                focal = round((fl_mm / 36.0) * _w)
                                cx, cy = _w // 2, _h // 2
                                log.info(f"ENABLE_FL_METRIC: fl={fl_mm}mm -> focal={focal}px "
                                         f"(image {_w}x{_h}, formula: ({fl_mm}/36)*{_w})")
                                try:
                                    idx = component.args.index("--ImageReader.camera_model")
                                    component.args.pop(idx)
                                    component.args.pop(idx)
                                except ValueError:
                                    pass
                                component.args.extend([
                                    "--ImageReader.camera_model", "SIMPLE_PINHOLE",
                                    "--ImageReader.camera_params", f"{focal},{cx},{cy}"
                                ])
                        except Exception as _e:
                            log.warning(f"ENABLE_FL_METRIC focal length failed, "
                                        f"falling back to COLMAP defaults: {_e}")
                    elif config.get('ENABLE_FL_HEURISTIC', 'false') == 'true':
                        # Apply focal length heuristic: f = multiplier * max(width, height)
                        try:
                            img_files = [f for f in os.listdir(image_path)
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                            if img_files:
                                with Image.open(os.path.join(image_path, img_files[0])) as _img:
                                    _w, _h = _img.size
                                focal = round(float(config.get('FL_HEURISTIC_VALUE', '1.2'))* max(_w, _h))
                                cx, cy = _w // 2, _h // 2
                                log.info(f"ENABLE_FL_HEURISTIC: using focal={focal}, cx={cx}, cy={cy} "
                                         f"(image {_w}x{_h})")
                                try:
                                    idx = component.args.index("--ImageReader.camera_model")
                                    component.args.pop(idx)
                                    component.args.pop(idx)
                                except ValueError:
                                    pass
                                component.args.extend([
                                    "--ImageReader.camera_model", "SIMPLE_PINHOLE",
                                    "--ImageReader.camera_params", f"{focal},{cx},{cy}"
                                ])
                        except Exception as _e:
                            log.warning(f"PRESERVE_SCENE_SCALE focal heuristic failed, "
                                        f"falling back to COLMAP defaults: {_e}")
                    pipeline.run_component(i)
                    # AUTO_MAPPER: override mapper components based on actual image count
                    if config.get('AUTO_MAPPER', 'false') == 'true' and config['RUN_RECON'] == 'true':
                        num_imgs = len([f for f in os.listdir(image_path)
                                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        if num_imgs < 600:
                            target_mapper = 'glomap'
                        elif num_imgs <= 5000:
                            target_mapper = 'colmap'
                        else:
                            target_mapper = 'hloc'
                        log.info(f"AUTO_MAPPER: {num_imgs} images -> selecting '{target_mapper}' mapper")
                        if target_mapper != config['RECON_SOFTWARE_NAME']:
                            config['RECON_SOFTWARE_NAME'] = target_mapper
                            # Update DynamoDB to reflect the auto-selected mapper
                            if ddb_table_name:
                                try:
                                    import boto3 as _boto3
                                    _region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
                                    _ddb = _boto3.resource('dynamodb', region_name=_region)
                                    _table = _ddb.Table(ddb_table_name)
                                    _table.update_item(
                                        Key={os.environ.get('DDB_KEY_NAME', 'uuid'): config['UUID']},
                                        UpdateExpression='SET reconSoftwareName = :v',
                                        ExpressionAttributeValues={':v': target_mapper}
                                    )
                                    log.info(f"AUTO_MAPPER: updated DDB reconSoftwareName to '{target_mapper}'")
                                except Exception as _ddb_err:
                                    log.warning(f"AUTO_MAPPER: failed to update DDB: {_ddb_err}")
                            # Remove existing mapper/viewgraph components
                            pipeline.components = [
                                c for c in pipeline.components
                                if 'Mapper' not in c.name and 'ViewGraph' not in c.name
                                    and c.name not in ('HlocSfM-Tri')
                            ]
                            pipeline.config.num_components = len(pipeline.components)
                            # Insert new mapper components after the last matcher component
                            insert_idx = max(
                                (j for j, c in enumerate(pipeline.components)
                                 if 'Matcher' in c.name or 'Undistorter' in c.name),
                                default=i
                            ) + 1
                            new_components = []
                            if target_mapper == 'glomap':
                                new_components.append(Component(
                                    name='GlomapSfM-ViewGraph',
                                    comp_type=ComponentType.RECONSTRUCTION,
                                    comp_environ=ComponentEnvironment.EXECUTABLE,
                                    command='colmap',
                                    args=['view_graph_calibrator', '--database_path', colmap_db_path],
                                    cwd=current_dir_path, requires_gpu=False
                                ))
                                new_components.append(Component(
                                    name='GlomapSfM-Mapper',
                                    comp_type=ComponentType.RECONSTRUCTION,
                                    comp_environ=ComponentEnvironment.EXECUTABLE,
                                    command='colmap',
                                    args=['global_mapper', '--database_path', colmap_db_path,
                                          '--image_path', image_path, '--output_path', sparse_path],
                                    cwd=current_dir_path, requires_gpu=False
                                ))
                            elif target_mapper == 'colmap':
                                mapper_args = [
                                    'mapper', '--database_path', colmap_db_path,
                                    '--image_path', image_path, '--output_path', sparse_path,
                                    '--Mapper.multiple_models', '0'
                                ]
                                if config['LOG_VERBOSITY'] == 'error':
                                    mapper_args.extend(['--log_level', '1'])
                                if int(pipeline.config.num_gpus) > 0:
                                    mapper_args.extend(['--Mapper.ba_use_gpu', '1'])
                                    if num_imgs >= 250:
                                        mapper_args.extend(['--Mapper.ba_global_backend', 'CASPAR'])
                                new_components.append(Component(
                                    name='ColmapSfM-Mapper',
                                    comp_type=ComponentType.RECONSTRUCTION,
                                    comp_environ=ComponentEnvironment.EXECUTABLE,
                                    command='colmap',
                                    args=mapper_args,
                                    cwd=current_dir_path, requires_gpu=False
                                ))
                            else:  # hloc
                                new_components.append(Component(
                                    name='HlocSfM-Mapper',
                                    comp_type=ComponentType.RECONSTRUCTION,
                                    comp_environ=ComponentEnvironment.EXECUTABLE,
                                    command='colmap',
                                    args=['hierarchical_mapper', '--database_path', colmap_db_path,
                                          '--image_path', image_path, '--output_path', sparse_path],
                                    cwd=current_dir_path, requires_gpu=False
                                ))
                                new_components.append(Component(
                                    name='HlocSfM-Tri',
                                    comp_type=ComponentType.RECONSTRUCTION,
                                    comp_environ=ComponentEnvironment.EXECUTABLE,
                                    command='colmap',
                                    args=['point_triangulator', '--database_path', colmap_db_path,
                                          '--image_path', image_path, '--input_path', sparse_model_path,
                                          '--output_path', sparse_model_path, '--refine_intrinsics', '1',
                                          '--Mapper.multiple_models', '0'],
                                    cwd=current_dir_path, requires_gpu=False
                                ))
                            for idx_offset, comp in enumerate(new_components):
                                pipeline.components.insert(insert_idx + idx_offset, comp)
                            pipeline.config.num_components = len(pipeline.components)
                            log.info(f"AUTO_MAPPER: replaced mapper with {[c.name for c in new_components]}")
                    # AUTO_MATCHER: analyze image overlap to select best feature matcher
                    if config.get('AUTO_MATCHER', 'false') == 'true' and config['RUN_RECON'] == 'true':
                        has_pose_priors = config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true' or \
                                         config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true'
                        matcher_cmd = [
                            sys.executable, os.path.join(current_dir_path, 'pre_processing', 'auto_matcher.py'),
                            '-i', image_path
                        ]
                        if has_pose_priors:
                            matcher_cmd.append('--pose-priors')
                        try:
                            matcher_result = subprocess.run(
                                matcher_cmd, capture_output=True, text=True, check=True,
                                cwd=current_dir_path
                            )
                            log.info(matcher_result.stdout.strip())
                            # Parse MATCHER= line from output
                            target_matcher = None
                            auto_matcher_overlap = None
                            for line in matcher_result.stdout.strip().split('\n'):
                                if line.startswith('MATCHER='):
                                    target_matcher = line.split('=', 1)[1].strip()
                                elif 'Median consecutive overlap:' in line:
                                    try:
                                        auto_matcher_overlap = float(line.split(':')[-1].strip())
                                    except ValueError:
                                        pass
                            if target_matcher and target_matcher != config['MATCHING_METHOD']:
                                log.info(f"AUTO_MATCHER: overriding '{config['MATCHING_METHOD']}' -> '{target_matcher}'")
                                config['MATCHING_METHOD'] = target_matcher
                                # Update DynamoDB to reflect the auto-selected matcher
                                if ddb_table_name:
                                    try:
                                        import boto3 as _boto3
                                        _region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
                                        _ddb = _boto3.resource('dynamodb', region_name=_region)
                                        _table = _ddb.Table(ddb_table_name)
                                        _table.update_item(
                                            Key={os.environ.get('DDB_KEY_NAME', 'uuid'): config['UUID']},
                                            UpdateExpression='SET matchingMethod = :v',
                                            ExpressionAttributeValues={':v': target_matcher}
                                        )
                                        log.info(f"AUTO_MATCHER: updated DDB matchingMethod to '{target_matcher}'")
                                    except Exception as _ddb_err:
                                        log.warning(f"AUTO_MATCHER: failed to update DDB: {_ddb_err}")
                                # Find and replace the existing matcher component
                                for j, c in enumerate(pipeline.components):
                                    if c.name == 'ColmapSfM-Feature-Matcher':
                                        if target_matcher == 'sequential':
                                            c.args = [
                                                'sequential_matcher',
                                                '--database_path', colmap_db_path,
                                                '--SequentialMatching.quadratic_overlap', '1',
                                                '--SequentialMatching.overlap', '10',
                                                '--SequentialMatching.loop_detection', '1',
                                                '--SequentialMatching.loop_detection_period', config['MAX_NUM_IMAGES'],
                                                '--SequentialMatching.loop_detection_num_images', config['MAX_NUM_IMAGES'],
                                                '--SequentialMatching.vocab_tree_path', colmap_vocab_path
                                            ]
                                        elif target_matcher == 'spatial':
                                            c.args = [
                                                'spatial_matcher',
                                                '--database_path', colmap_db_path,
                                                '--SpatialMatching.ignore_z', '0'
                                            ]
                                        elif target_matcher == 'vocab':
                                            c.args = [
                                                'vocab_tree_matcher',
                                                '--database_path', colmap_db_path,
                                                '--VocabTreeMatching.num_images', str(math.ceil(float(config['MAX_NUM_IMAGES']) / 3)),
                                                '--VocabTreeMatching.vocab_tree_path', colmap_vocab_path
                                            ]
                                        else:  # exhaustive
                                            c.args = [
                                                'exhaustive_matcher',
                                                '--database_path', colmap_db_path,
                                                '--ExhaustiveMatching.block_size', config['MAX_NUM_IMAGES']
                                            ]
                                        if config['LOG_VERBOSITY'] == 'error':
                                            c.args.extend(['--log_level', '1'])
                                        log.info(f"AUTO_MATCHER: updated matcher component to '{target_matcher}'")
                                        break
                            elif target_matcher:
                                log.info(f"AUTO_MATCHER: keeping current matcher '{config['MATCHING_METHOD']}'")
                        except Exception as e:
                            log.warning(f"AUTO_MATCHER: analysis failed ({e}), keeping '{config['MATCHING_METHOD']}'")
                case "ColmapSfM-Mapper" if component.args and component.args[0] == "pose_prior_mapper":
                    # GPS: use pose_prior_mapper only when enough images have a prior
                    # in the database; otherwise run the standard mapper.
                    import sqlite3 as _sql
                    _n_priors, _n_imgs, _gps = 0, 0, {}
                    try:
                        _con = _sql.connect(colmap_db_path)
                        _n_priors = _con.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0]
                        _n_imgs = _con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
                        _con.close()
                    except Exception as _e:
                        log.warning(f"Could not count pose priors in {colmap_db_path}: {_e}")
                    if os.path.isfile(gps_priors_json):
                        with open(gps_priors_json, encoding='utf-8') as _f:
                            _gps = json.load(_f) or {}
                    if _n_imgs and _n_priors >= 0.5 * _n_imgs:
                        _std = _gps.get('std_m') or {'x': 5.0, 'y': 5.0, 'z': 10.0}
                        component.args.extend([
                            "--overwrite_priors_covariance", "1",
                            "--prior_position_std_x", str(_std['x']),
                            "--prior_position_std_y", str(_std['y']),
                            "--prior_position_std_z", str(_std['z']),
                            "--use_robust_loss_on_prior_position", "1"
                        ])
                        log.info(f"GPS pose_prior_mapper: {_n_priors}/{_n_imgs} images have GPS priors; "
                                 f"std (m) x={_std['x']} y={_std['y']} z={_std['z']}")
                    else:
                        component.args[0] = "mapper"
                        log.warning(f"Only {_n_priors}/{_n_imgs} images have GPS priors; "
                                    f"running the standard mapper instead of pose_prior_mapper")
                    run_mapper_checked(pipeline, i, colmap_db_path, config, LOCAL_DEBUG, log)
                case "ColmapSfM-Mapper":
                    run_mapper_checked(pipeline, i, colmap_db_path, config, LOCAL_DEBUG, log)
                case "ColmapSfM-Triangulator" | "HlocSfM-Tri":
                    import sqlite3 as _sql
                    try:
                        _conn = _sql.connect(colmap_db_path)
                        _cur = _conn.cursor()
                        _cur.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        )
                        _existing = {r[0] for r in _cur.fetchall()}
                        _dropped = []
                        for _tbl in ("frame_data", "frames", "rig_sensors", "rigs"):
                            if _tbl in _existing:
                                _cur.execute(f"DROP TABLE {_tbl}")
                                _dropped.append(_tbl)
                        _conn.commit()
                        _conn.close()
                        if _dropped:
                            log.info(
                                "Dropped COLMAP 4.x rig tables before"
                                f" triangulator: {_dropped}"
                            )
                    except Exception as _rig_err:
                        log.warning(f"Could not drop rig tables: {_rig_err}")
                    for _rf in ("rigs.txt", "frames.txt",
                                "rigs.bin", "frames.bin"):
                        _rfp = os.path.join(sparse_model_path, _rf)
                        if os.path.exists(_rfp):
                            os.remove(_rfp)
                            log.info(f"Removed {_rf} from model dir")
                    pipeline.run_component(i)
                case "ColmapSfM-Image-Undistorter":
                    pipeline.run_component(i)
                    # Move undistorted output into place
                    undist_dir = os.path.join(config['DATASET_PATH'], '_undistorted')
                    if os.path.isdir(undist_dir):
                        undist_sparse = os.path.join(undist_dir, 'sparse')
                        undist_images = os.path.join(undist_dir, 'images')
                        if os.path.isdir(undist_sparse):
                            for f in os.listdir(undist_sparse):
                                shutil.copy2(
                                    os.path.join(undist_sparse, f),
                                    os.path.join(sparse_model_path, f),
                                )
                            log.info(f"Copied undistorted model into {sparse_model_path}")
                            # Also update colmap/sparse/0 so colmap-to-nerfstudio
                            # reads the undistorted model
                            _colmap_sparse = os.path.join(
                                config['DATASET_PATH'], 'colmap', 'sparse', '0'
                            )
                            if os.path.isdir(_colmap_sparse):
                                for f in os.listdir(undist_sparse):
                                    shutil.copy2(
                                        os.path.join(undist_sparse, f),
                                        os.path.join(_colmap_sparse, f),
                                    )
                                log.info(
                                    f"Copied undistorted model into {_colmap_sparse}"
                                )
                        if os.path.isdir(undist_images):
                            for f in os.listdir(undist_images):
                                shutil.move(os.path.join(undist_images, f),
                                            os.path.join(image_path, f))
                            log.info(f"Moved undistorted images into {image_path}")
                        shutil.rmtree(undist_dir)
                case "Colmap-to-Nerfstudio":
                    # Ensure we use the largest Colmap model if multiple found
                    if config['RECON_SOFTWARE_NAME'] == "colmap" or config['RECON_SOFTWARE_NAME'] == "glomap" \
                        or config['RECON_SOFTWARE_NAME'] == "hloc":
                        select_largest_colmap_model(sparse_path)
                    # Move existing transforms.json to transforms-in.json when using pose priors
                    # This ensures colmap-to-nerfstudio creates fresh transforms.json from updated COLMAP data
                    if config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true' or \
                        config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true':
                        if os.path.exists(transforms_out_path):
                            log.info(f"Moving {transforms_out_path} to {transforms_in_path} to preserve original")
                            shutil.move(transforms_out_path, transforms_in_path)
                    pipeline.run_component(i)
                    # Copy COLMAP database back to dataset path for archiving
                    if LOCAL_DEBUG and os.path.exists(colmap_db_path):
                        dataset_db = os.path.join(config['DATASET_PATH'], 'database.db')
                        shutil.copy2(colmap_db_path, dataset_db)
                        log.info(f"Copied COLMAP database from {colmap_db_path} to {dataset_db}")
                case "Nerfstudio-Export":
                    if LOCAL_DEBUG and config['RUN_RECON'] == "false" and config['RUN_TRAIN'] == "false":
                        continue
                    try:
                        pipeline.run_component(i)
                    except RuntimeError as e:
                        captured = (e.args[1] if len(e.args) > 1 else '') or ''
                        component_out = getattr(component, 'output', '') or ''
                        combined = captured + component_out
                        if 'NaN/Inf' in combined or 'All tensors must be numpy arrays' in combined:
                            pipeline.report_error(766, "Gaussian splat training diverged: all Gaussians are NaN/Inf. "
                                "Try reducing the learning rate, increasing the number of images, "
                                "or improving image quality/coverage.")
                        raise
                    # Clean up CUDA memory after export
                    cleanup_cuda_memory()
                case "Train":
                    if LOCAL_DEBUG and config['RUN_RECON'] == "false" and config['RUN_TRAIN'] == "false":
                        continue
                    # TRAINING CONDITIONAL COMPONENT
                    if not os.path.exists(config['DATASET_PATH']):
                        log.error(f"CRITICAL: Dataset missing before training: {config['DATASET_PATH']}")
                        raise RuntimeError(f"Dataset disappeared: {config['DATASET_PATH']}")
                    log.info(f"Dataset contents before training: {os.listdir(config['DATASET_PATH'])}")
                    # For dn-splatter/ags-mesh: re-evaluate sensor depth at runtime and patch
                    # training args if the registration-time detection was wrong (e.g. depth dir
                    # was inside the zip and not yet extracted when args were built).
                    if config['MODEL'] in ("dn-splatter", "dn-splatter-big", "ags-mesh") and ENABLE_MULTI_GPU == "false":
                        # Re-evaluate sensor depth at runtime — depth_sensor/ is created by
                        # preprocess after pipeline creation, so the build-time detection may
                        # have been wrong (uint8 files present but not yet converted).
                        _has_sensor_depth = False
                        for _dname in ("depth_sensor", "depth_images", "depth"):
                            _dd = os.path.join(config['DATASET_PATH'], _dname)
                            if os.path.isdir(_dd):
                                for _r, _d, _fls in os.walk(_dd):
                                    if any(_fn.endswith('.png') for _fn in _fls):
                                        _has_sensor_depth = True
                                        break
                            if _has_sensor_depth:
                                break
                        # Always use mono normals — dsine provides stronger planar supervision
                        _normal_supervision = "mono"
                        log.info(f"DN-Splatter Train runtime depth mode: {'sensor' if _has_sensor_depth else 'mono'}")
                        log.info(f"DN-Splatter Train runtime normal-supervision: {_normal_supervision}")
                        # Patch normal-supervision in component args
                        _args = component.args
                        for _j, _arg in enumerate(_args):
                            if _arg == "--pipeline.model.normal-supervision" and _j + 1 < len(_args):
                                _args[_j + 1] = _normal_supervision
                                break
                    # Check image count and resolution at runtime to configure color correction
                    # and image caching. Count recursively: spherical runs use pano_camera*/
                    # subdirs and rig/colmap-prefix zips use images/<camera>/ subdirs. A flat
                    # listdir counted 0 for those, so the disk-cache guard below never fired
                    # and nerfstudio cached 500+ images in RAM until the OOM killer hit (-9).
                    image_files = [
                        os.path.join(root, f)
                        for root, _, files in os.walk(image_path)
                        for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ]
                    num_images = len(image_files)
                    log.info(f"Train runtime: {num_images} images under {image_path}")
                    is_4k_or_higher = False
                    if image_files:
                        sample_img = Image.open(image_files[0])
                        is_4k_or_higher = sample_img.width >= 3840 or sample_img.height >= 2160
                    
                    # Disable color correction metrics for large/high-res datasets
                    disable_color_correction = num_images > GPU_MAX_IMAGES or is_4k_or_higher
                    if disable_color_correction and config['MODEL'] in ['splatfacto', 'splatfacto-big', 'splatfacto-mcmc']:
                        log.info(f"Disabling color corrected metrics: num_images={num_images}, is_4k={is_4k_or_higher}")
                        # Find and replace color correction args
                        for j, arg in enumerate(component.args):
                            if arg == "--pipeline.model.color-corrected-metrics" and j+1 < len(component.args):
                                component.args[j+1] = "False"
                    # Transparent sky (REMOVE_SKY with splatfacto models): train the sky as
                    # alpha = 0 instead of ignoring it. Splatfacto composites RGBA targets
                    # with its random background, so anything rendered in the sky is
                    # pushed to transparent and the sky is left empty in the splat.
                    # The sky must not also be in the ignore masks (masked pixels have no
                    # loss), so training uses masks_no_sky/ (lens flare + uploaded masks).
                    if REMOVE_SKY and component.command == "ns-train" and "colmap" in component.args and \
                            config['MODEL'] in ("splatfacto", "splatfacto-big", "splatfacto-mcmc"):
                        _sky_dir = os.path.join(config['DATASET_PATH'], "sky_masks")
                        _nosky_dir = os.path.join(config['DATASET_PATH'], "masks_no_sky")
                        if os.path.isdir(_sky_dir):
                            try:
                                _n_rgba, _sky_px = build_transparent_sky_images(
                                    image_path, _sky_dir, os.path.join(config['DATASET_PATH'], "images_rgba"))
                                component.args.extend(["--images-path", "images_rgba"])
                                if "--masks-path" in component.args and os.path.isdir(_nosky_dir):
                                    component.args[component.args.index("--masks-path") + 1] = _nosky_dir
                                log.info(f"Transparent sky: {_n_rgba} RGBA training images in images_rgba/ "
                                         f"({_sky_px:.1%} of pixels are sky, trained as alpha=0); "
                                         f"ignore masks without sky from {_nosky_dir}")
                            except Exception as _sky_err:
                                log.warning(f"Transparent sky setup failed, sky stays ignored instead: {_sky_err}")
                        else:
                            log.warning("REMOVE_SKY: no sky_masks/ found (sky step failed?); sky is not removed")
                    # Set the image cache to disk if there are a lot of images to prevent OOM
                    if config['MODEL'] not in ("nerfacto", "3dgrt", "3dgut", "faster-gs") and ENABLE_MULTI_GPU == "false":
                        if num_images > GPU_MAX_IMAGES:
                            if "colmap" in component.args:
                                index = component.args.index("colmap")
                                if config['MODEL'] != "splatfacto-w-light":
                                    component.args.insert(index, "disk")
                                else:
                                    component.args.insert(index, "cpu")
                                component.args.insert(index, "--pipeline.datamanager.cache-images")
                                # Reduce dataloader workers to avoid IndexError with disk caching
                                component.args.insert(index, "0")
                                component.args.insert(index, "--pipeline.datamanager.dataloader-num-workers")
                    # MCMC models: set the Gaussian cap from the data and the GPU instead of
                    # using NUM_GAUSSIANS as is (a cap at or below the SfM point count leaves
                    # MCMC no room to add Gaussians). NUM_GAUSSIANS is honoured as a minimum.
                    _cap_flag = next((f for f in ("--pipeline.model.max-gs-num", "--strategy.cap-max")
                                      if f in component.args), None)
                    if _cap_flag and component.args.index(_cap_flag) + 1 < len(component.args):
                        try:
                            _ci = component.args.index(_cap_flag) + 1
                            _user_cap = int(float(component.args[_ci]))
                            _mp = 0.0
                            if image_files:
                                with Image.open(image_files[0]) as _im0:
                                    _mp = _im0.width * _im0.height / 1e6
                            _cap, _note = estimate_mcmc_cap(
                                _user_cap, count_sfm_points(config['DATASET_PATH']), num_images, _mp,
                                gpu_total_memory_gb(),
                                images_on_gpu="--pipeline.datamanager.cache-images" not in component.args
                                and component.command == "ns-train")
                            component.args[_ci] = str(_cap)
                            log.info(_note)
                        except Exception as _cap_err:
                            log.warning(f"MCMC cap estimate failed, using NUM_GAUSSIANS as is: {_cap_err}")
                    if ENABLE_MULTI_GPU == "false":
                        # Faster-GS reads sparse/0 (or colmap/sparse/0) directly, so skip the nerfstudio move
                        if config['MODEL'] not in ("3dgut", "3dgrt", "faster-gs") and not IS_GSPLAT_DEPTH and \
                                config['MODEL'] not in ("dn-splatter", "dn-splatter-big", "ags-mesh"):
                            if config['RECON_SOFTWARE_NAME'] == "colmap" or config['RECON_SOFTWARE_NAME'] == "glomap" or \
                                config['RECON_SOFTWARE_NAME'] == "map_anything" or config['RECON_SOFTWARE_NAME'] == "hloc":
                                # Ensure colmap/sparse structure exists for NerfStudio
                                log.info('Running Training...')
                                sparse_path_out = os.path.join(config['DATASET_PATH'], "colmap", "sparse")
                                if colmap_zip_found and os.path.exists(sparse_path_out):
                                    # colmap/sparse/ already populated from zip extraction — skip move
                                    log.info(f"colmap/sparse/ already populated from zip, skipping sparse/ move")
                                elif os.path.exists(sparse_path) and os.listdir(sparse_path) and \
                                        any(os.listdir(os.path.join(sparse_path, d))
                                            for d in os.listdir(sparse_path)
                                            if os.path.isdir(os.path.join(sparse_path, d))):
                                    # sparse/ has content - move it to colmap/sparse/
                                    if os.path.exists(sparse_path_out):
                                        if os.path.islink(sparse_path_out):
                                            os.unlink(sparse_path_out)
                                        else:
                                            shutil.rmtree(sparse_path_out)
                                    log.info(f"Moving sparse point cloud from {sparse_path} to {sparse_path_out}")
                                    os.makedirs(os.path.dirname(sparse_path_out), exist_ok=True)
                                    shutil.move(sparse_path, sparse_path_out)
                                elif not os.path.exists(sparse_path_out):
                                    log.error(f"No sparse reconstruction found at {sparse_path} or {sparse_path_out}")
                                    raise RuntimeError("No sparse reconstruction data found for training")
                                # Verify colmap/sparse/0 has required files
                                sparse_0 = os.path.join(sparse_path_out, "0")
                                if os.path.exists(sparse_0):
                                    sparse_files = os.listdir(sparse_0)
                                    log.info(f"colmap/sparse/0 contents: {sparse_files}")
                                    if not any(f in sparse_files for f in ['cameras.bin', 'cameras.txt']):
                                        log.error(f"cameras.bin/txt missing in {sparse_0}")
                                else:
                                    log.warning(f"colmap/sparse/0 does not exist, contents of colmap/sparse: {os.listdir(sparse_path_out) if os.path.exists(sparse_path_out) else 'N/A'}")                              
                                # nerfstudio opens <data>/<images-path>/<name> for every image in
                                # the model; check them all now so a model/image mismatch fails
                                # with the names instead of a FileNotFoundError mid-startup.
                                if os.path.exists(sparse_0):
                                    _img_root = os.path.join(config['DATASET_PATH'], component.args[component.args.index("--images-path") + 1]) \
                                        if "--images-path" in component.args else image_path
                                    try:
                                        import pycolmap as _pc
                                        _names = [im.name for im in _pc.Reconstruction(sparse_0).images.values()]
                                    except Exception as _rec_err:
                                        _names = []
                                        log.warning(f"Could not read {sparse_0} to check image names: {_rec_err}")
                                    _missing = [n for n in _names if not os.path.isfile(os.path.join(_img_root, n))]
                                    if _missing:
                                        _on_disk = sorted(os.path.relpath(os.path.join(_r, _f), _img_root)
                                                          for _r, _d, _fs in os.walk(_img_root) for _f in _fs)
                                        _dirs = sorted({os.path.dirname(n) or '.' for n in _names})
                                        raise RuntimeError(pipeline.report_error(
                                            706,
                                            f"{len(_missing)} of {len(_names)} images in the COLMAP model {sparse_0} "
                                            f"are not under {_img_root} (model sub-folders: {_dirs[:10]}). "
                                            f"Missing e.g. {_missing[:5]}; on disk e.g. {_on_disk[:5]} "
                                            f"({len(_on_disk)} files). Image names in images.bin must match "
                                            f"the paths under images/ exactly."))
                                    log.info(f"All {len(_names)} model images found under {_img_root}")
                                # Splatfacto starts each Gaussian with scale log(mean distance to its 3
                                # nearest SfM points). Four or more identical points give log(0) = -inf,
                                # which turns those Gaussians NaN within the first steps; MCMC never prunes
                                # them and they eventually turn the whole splat NaN (error 766). Train from
                                # a copy of the model without exact duplicate points. For splatfacto-mcmc
                                # also drop isolated points, which start as huge Gaussians that the MCMC
                                # noise step throws far out of the scene. The original model is unchanged.
                                if os.path.exists(sparse_0) and component.command == "ns-train" and \
                                        "colmap" in component.args and "--colmap-path" not in component.args:
                                    try:
                                        import numpy as _np
                                        import pycolmap as _pc
                                        from sklearn.neighbors import NearestNeighbors as _NN
                                        _rec = _pc.Reconstruction(sparse_0)
                                        _ids = _np.array(list(_rec.points3D.keys()))
                                        _xyz = _np.array([_rec.points3D[int(_i)].xyz for _i in _ids])
                                        _keep = _np.zeros(len(_ids), dtype=bool)
                                        if len(_ids):
                                            _keep[_np.unique(_xyz, axis=0, return_index=True)[1]] = True
                                        _n_dup = int((~_keep).sum())
                                        _n_iso = 0
                                        if config['MODEL'] == "splatfacto-mcmc" and _keep.sum() > 4:
                                            _k = _NN(n_neighbors=4).fit(_xyz[_keep]).kneighbors(_xyz[_keep])[0][:, 1:].mean(1)
                                            _iso = _k > 20.0 * _np.median(_k)
                                            _n_iso = int(_iso.sum())
                                            _keep[_np.where(_keep)[0][_iso]] = False
                                        if _n_dup or _n_iso:
                                            for _i in _ids[~_keep]:
                                                _rec.delete_point3D(int(_i))
                                            _train_model = os.path.join(sparse_path_out, "0_train")
                                            if os.path.exists(_train_model):
                                                shutil.rmtree(_train_model)
                                            os.makedirs(_train_model)
                                            _rec.write(_train_model)
                                            component.args.extend(["--colmap-path", os.path.relpath(_train_model, config['DATASET_PATH'])])
                                            log.info(f"Training from {_train_model}: removed {_n_dup:,} duplicate and {_n_iso:,} "
                                                     f"isolated SfM points of {len(_ids):,} (duplicates start as log(0) scale "
                                                     f"Gaussians, which turn NaN)")
                                    except Exception as _dedup_err:
                                        log.warning(f"Could not clean SfM points before training, using {sparse_0} as is: {_dedup_err}")
                        else: # 3dgrut
                            _sample_img = next(
                                (f for f in os.listdir(image_path)
                                 if os.path.isfile(os.path.join(image_path, f))
                                 and f.lower().endswith(('.png', '.jpg', '.jpeg'))),
                                None
                            )
                            if _sample_img and has_alpha_channel(os.path.join(image_path, _sample_img)):
                                process_images(image_path)
                    try:
                        # For gsplat-depth: fix point_indices KeyError caused by rig/subdir image names.
                        # gsplat's colmap.py keys point_indices by the image name in images.bin
                        # (e.g. 'face_01/pano_011.png'). After clean_images_dir flattens subdirs,
                        # the actual files are 'face_01_pano_011.png'. Flatten images.bin to match.
                        if IS_GSPLAT_DEPTH and ENABLE_MULTI_GPU == "false":
                            # Prepare the model gsplat will read: it loads <data-dir>/sparse/0
                            # when that holds a model; colmap/sparse/0 is only copied there
                            # (below) when sparse/0 is empty. Editing the other copy left
                            # gsplat with unflattened names (KeyError 'face_iphone/...').
                            _root_s0 = os.path.join(config['DATASET_PATH'], "sparse", "0")
                            _colmap_s0 = os.path.join(config['DATASET_PATH'], "colmap", "sparse", "0")
                            _root_has_model = any(os.path.isfile(os.path.join(_root_s0, _f))
                                                  for _f in ("cameras.bin", "cameras.txt"))
                            _sparse_0 = _root_s0 if _root_has_model or not os.path.exists(_colmap_s0) \
                                else _colmap_s0
                            log.info(f"gsplat-depth: preparing COLMAP model {_sparse_0}")
                            if os.path.exists(_sparse_0):
                                # Pose-only exports (ARKit poses + LiDAR points3D) have no
                                # points2D/tracks, so gsplat's depth loss gets no depths.
                                # Link each image to the points3D it sees (before the image
                                # names are flattened, so masks still match).
                                add_projected_observations_for_gsplat(
                                    _sparse_0, os.path.join(config['DATASET_PATH'], 'masks'), log)
                                flatten_images_for_gsplat(image_path, _sparse_0, log)
                                rebuild_points3d_tracks_for_gsplat(_sparse_0, log)
                                # Flatten masks/ to match flattened image names.
                                # patch_gsplat.py looks up masks by imdata[k].name which is now
                                # 'face_00_pano_001.png' after flattening, so masks must also be flat.
                                _masks_dir = os.path.join(config['DATASET_PATH'], 'masks')
                                if os.path.isdir(_masks_dir):
                                    for _sub in list(os.listdir(_masks_dir)):
                                        _sub_path = os.path.join(_masks_dir, _sub)
                                        if os.path.isdir(_sub_path):
                                            for _mfile in os.listdir(_sub_path):
                                                _src = os.path.join(_sub_path, _mfile)
                                                _dst = os.path.join(_masks_dir, f"{_sub}_{_mfile}")
                                                os.rename(_src, _dst)
                                            shutil.rmtree(_sub_path)
                                    log.info(f"Flattened masks/ directory for gsplat-depth")
                                    # Remove images where the mask covers all pixels
                                    remove_fully_masked_images_for_gsplat(
                                        image_path, _sparse_0, _masks_dir, log
                                    )
                        # 3DGRUT and gsplat's simple_trainer (gsplat-depth) read the
                        # COLMAP model from sparse/0/, but the colmap-prefix zip layout
                        # puts it at colmap/sparse/0/, and an empty sparse/0/
                        # placeholder is created at startup. Copy the files over
                        # (after flattening above, so the flattened images.bin is used).
                        if config['MODEL'] == '3dgut' or (
                            IS_GSPLAT_DEPTH and ENABLE_MULTI_GPU == "false"
                        ):
                            _s0 = os.path.join(
                                config['DATASET_PATH'], 'sparse', '0',
                            )
                            _cs0 = os.path.join(
                                config['DATASET_PATH'],
                                'colmap', 'sparse', '0',
                            )
                            _s0_empty = (
                                not os.path.isdir(_s0)
                                or not os.listdir(_s0)
                            )
                            if _s0_empty and os.path.isdir(_cs0):
                                os.makedirs(_s0, exist_ok=True)
                                # gsplat only needs per-image poses. Leave out COLMAP 4
                                # rig files: pycolmap derives image poses from them when
                                # present, so a rig that is incomplete (images without a
                                # frame) or inconsistent with images.bin breaks loading.
                                # Without them pycolmap uses the images.bin poses.
                                _skip_rig = config['MODEL'] != '3dgut'
                                _skipped = []
                                for _sf in os.listdir(_cs0):
                                    if _skip_rig and _sf in ('rigs.bin', 'frames.bin',
                                                             'rigs.txt', 'frames.txt'):
                                        _skipped.append(_sf)
                                        continue
                                    shutil.copy2(
                                        os.path.join(_cs0, _sf),
                                        os.path.join(_s0, _sf),
                                    )
                                log.info(
                                    "Copied COLMAP model from"
                                    " colmap/sparse/0/ to sparse/0/"
                                    f" for {config['MODEL']}"
                                    + (f" (left out {', '.join(_skipped)})" if _skipped else "")
                                )

                        # 3DGRUT expects masks as <stem>_mask.png next
                        # to each image, not in a separate masks/ dir.
                        if ZIP_HAS_MASKS and config['MODEL'] == '3dgut':
                            _masks_dir = os.path.join(
                                config['DATASET_PATH'], 'masks',
                            )
                            _copied_masks = 0
                            for _mr, _md, _mf in os.walk(image_path):
                                _mrel = os.path.relpath(_mr, image_path)
                                for _img_f in _mf:
                                    if not _img_f.lower().endswith(
                                        ('.png', '.jpg', '.jpeg')
                                    ):
                                        continue
                                    _stem = os.path.splitext(_img_f)[0]
                                    _mask_cand = os.path.join(
                                        _masks_dir,
                                        _mrel if _mrel != '.' else '',
                                        _img_f,
                                    )
                                    if not os.path.isfile(_mask_cand):
                                        _mask_cand = os.path.join(
                                            _masks_dir,
                                            _mrel if _mrel != '.' else '',
                                            _stem + '.png',
                                        )
                                    if os.path.isfile(_mask_cand):
                                        shutil.copy2(
                                            _mask_cand,
                                            os.path.join(
                                                _mr,
                                                _stem + '_mask.png',
                                            ),
                                        )
                                        _copied_masks += 1
                            log.info(
                                f"Copied {_copied_masks} masks"
                                " into 3DGRUT convention"
                                " (<image>_mask.png)"
                            )

                        # Clean up CUDA memory before training
                        cleanup_cuda_memory()
                        pipeline.run_component(i)
                        # Wait for subprocess to fully release GPU memory
                        log.info("Training complete. Waiting for GPU memory release...")
                        time.sleep(15)
                        cleanup_cuda_memory()
                    except Exception as e:
                        error_str = str(e)
                        if "cuda" in error_str.lower() and "assert" in error_str.lower():
                            log.error(f"CUDA assertion error detected: {error_str}")
                            log.error("Consider reducing the number of Gaussians or using more conservative training parameters.")
                        raise e
                    # Copy checkpoint and config to output directory immediately after training
                    # so they are available for debugging if export crashes
                    if ENABLE_MULTI_GPU == "false" and config['MODEL'] not in ["3dgut", "3dgrt", "faster-gs"]:
                        try:
                            output_ckpt_dst = os.path.join(output_path, "nerfstudio_models")
                            output_config_dst = os.path.join(output_path, "config.yml")
                            if os.path.exists(model_ckpt_path) and not os.path.exists(output_ckpt_dst):
                                shutil.copytree(model_ckpt_path, output_ckpt_dst)
                                log.info(f"Copied checkpoint to output for crash recovery: {output_ckpt_dst}")
                            if os.path.exists(model_config_path) and not os.path.exists(output_config_dst):
                                shutil.copy2(model_config_path, output_config_dst)
                                log.info(f"Copied config to output for crash recovery: {output_config_dst}")
                        except Exception as copy_err:
                            log.warning(f"Could not copy checkpoint/config to output: {copy_err}")

                    # Copy the output ply and checkpoint over to where we expect it (keep originals for export)
                    if ENABLE_MULTI_GPU == "false":
                        if config['MODEL'] == "3dgut" or config['MODEL'] == "3dgrt":
                            root_exp_dir = os.path.join(output_path, TRAIN_EXPERIMENT_NAME)
                            if config['RUN_RECON'] == "false" and not colmap_zip_found:
                                root_exp_dir = os.path.join(output_path, RESUME_TRAIN_EXPERIMENT_NAME)
                            if LOCAL_DEBUG:
                                # Find the actual experiment directory (with timestamp) - LOCAL_DEBUG only
                                if os.path.exists(root_exp_dir):
                                    exp_dirs = os.listdir(root_exp_dir)
                                    if exp_dirs:
                                        exp_dir = exp_dirs[0]  # Use the first (and likely only) directory
                                        ply_source = os.path.join(root_exp_dir, exp_dir, "export_last.ply")
                                        if os.path.exists(ply_source):
                                            shutil.move(ply_source, ply_path)
                                            log.info(f"Successfully moved PLY from {ply_source} to {ply_path}")
                                        else:
                                            log.error(f"PLY file not found at {ply_source}")
                                            # List available files for debugging
                                            available_files = os.listdir(os.path.join(root_exp_dir, exp_dir))
                                            log.error(f"Available files: {available_files}")
                                    else:
                                        log.error(f"No experiment directories found in {root_exp_dir}")
                                else:
                                    log.error(f"Root experiment directory not found: {root_exp_dir}")
                            else:
                                # Original behavior for non-LOCAL_DEBUG
                                exp_dir = os.listdir(root_exp_dir)[0]
                                shutil.move(os.path.join(root_exp_dir, exp_dir, "export_last.ply"), ply_path)
                        if config['MODEL'] == "3dgut" or config['MODEL'] == "3dgrt":
                            dest_dir = os.path.join(config['DATASET_PATH'], "3dgrut_models")
                            base_dir = os.path.join(config['DATASET_PATH'], 'exports', TRAIN_EXPERIMENT_NAME)
                            if config['RUN_RECON'] == "false" and not colmap_zip_found:
                                base_dir = os.path.join(config['DATASET_PATH'], 'exports', RESUME_TRAIN_EXPERIMENT_NAME)
                            src_dir = os.path.join(base_dir, sorted(os.listdir(base_dir))[-1])
                            os.makedirs(dest_dir, exist_ok=True)
                            shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
                case "Nerfstudio-Export-Nerfacto":
                    # NERFSTUDIO NERFACTO EXPORT CONDITIONAL COMPONENT
                    pipeline.run_component(i)
                    obj_to_glb(
                        os.path.join(output_path, "textured", "mesh.obj"),
                        os.path.join(output_path, "textured", f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.glb")
                    )
                case "USDZ-Add-Collision-Mesh":
                    usdz_tmp_path = os.path.join(output_path, "splat_with_mesh.usdz")
                    pipeline.run_component(i)
                    if os.path.exists(usdz_tmp_path):
                        shutil.move(usdz_tmp_path, usdz_path)
                        log.info(f"Replaced {usdz_path} with collision mesh version")
                    else:
                        log.warning(f"USDZ-Add-Collision-Mesh output not found at {usdz_tmp_path}, keeping original")
                case "Extract-Video-Thumbnail":
                    # For multi-GPU, copy auto-generated video before extracting thumbnail
                    if (ENABLE_MULTI_GPU == "true" or IS_GSPLAT_DEPTH) and config['ENABLE_VIDEO_EXPORT'] == "true":
                        videos_dir = os.path.join(output_path, "videos")
                        if os.path.exists(videos_dir):
                            video_files = sorted([f for f in os.listdir(videos_dir) if f.startswith('traj_') and f.endswith('.mp4')])
                            if video_files:
                                src_video = os.path.join(videos_dir, video_files[-1])
                                dst_video = os.path.join(output_path, "render.mp4")
                                shutil.copy2(src_video, dst_video)
                                log.info(f"Copied gsplat trajectory video from {src_video} to {dst_video}")
                            else:
                                log.warning(f"No trajectory video found in {videos_dir}")
                        else:
                            log.warning(f"Videos directory not found: {videos_dir}")
                    pipeline.run_component(i)
                case "S3-Export-Archive":
                    log.info("DEBUG: Handling S3-Export-Archive component...")
                    if LOCAL_DEBUG:
                        if config['MODEL'] == "nerfacto":
                            glb_path = os.path.join(output_path, "textured", f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.glb")
                            copy_to_local_output(glb_path, config, f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.glb", log)
                        else:
                            copy_to_local_output(ply_path, config, f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.ply", log)

                    log.info("DEBUG: Continuing after PLY copy/upload...")
                    # Copy nerfstudio_models and config.yml to dataset directory for archive
                    if config['RUN_TRAIN'] == "true" and ENABLE_MULTI_GPU != "true":
                        if config['MODEL'] not in ("3dgut", "3dgrt", "faster-gs"):
                            # Copy nerfstudio_models directory to dataset
                            dataset_models_path = os.path.join(config['DATASET_PATH'], "nerfstudio_models")
                            # For splatfacto initial training, copy from outputs directory
                            if config['MODEL'] in ["splatfacto", "splatfacto-big", "splatfacto-mcmc"] and config['RUN_RECON'] == "true":
                                if os.path.exists(model_ckpt_path) and not os.path.exists(dataset_models_path):
                                    shutil.copytree(model_ckpt_path, dataset_models_path)
                                    log.info(f"Copied nerfstudio_models from outputs to dataset: {dataset_models_path}")
                            elif os.path.exists(model_ckpt_path):
                                # For other models or resume training
                                if os.path.exists(dataset_models_path):
                                    shutil.rmtree(dataset_models_path)
                                shutil.copytree(model_ckpt_path, dataset_models_path)
                                log.info(f"Copied nerfstudio_models to dataset: {dataset_models_path}")
                            
                            # Log checkpoint files for debugging
                            if os.path.exists(dataset_models_path):
                                ckpt_files = [f for f in os.listdir(dataset_models_path) if f.endswith('.ckpt')]
                                log.info(f"Checkpoint files in archive: {ckpt_files}")
                            
                            # Copy config.yml to dataset
                            dataset_config_path = os.path.join(config['DATASET_PATH'], "config.yml")
                            if config['MODEL'] in ["splatfacto", "splatfacto-big", "splatfacto-mcmc"] and config['RUN_RECON'] == "true":
                                if os.path.exists(model_config_path) and not os.path.exists(dataset_config_path):
                                    shutil.copy2(model_config_path, dataset_config_path)
                                    log.info(f"Copied config.yml from outputs to dataset: {dataset_config_path}")
                            elif os.path.exists(model_config_path):
                                shutil.copy2(model_config_path, dataset_config_path)
                                log.info(f"Copied config.yml to dataset: {dataset_config_path}")

                    if not os.path.exists(config['DATASET_PATH']):
                        log.error(f"CRITICAL: Dataset missing before cleanup: {config['DATASET_PATH']}")
                        raise RuntimeError(f"Dataset disappeared before cleanup: {config['DATASET_PATH']}")
                    # Transparent-sky RGBA copies are only needed while training/evaluating;
                    # keep them out of model.tar.gz (sky_masks/ stays, so they can be rebuilt).
                    _rgba_dir = os.path.join(config['DATASET_PATH'], "images_rgba")
                    if os.path.isdir(_rgba_dir):
                        shutil.rmtree(_rgba_dir, ignore_errors=True)
                        log.info("Removed images_rgba/ (transparent-sky training copies) before archiving")
                    if IS_FASTER_GS:
                        # Drop the Faster-GS staging folder (symlinks or an undistorted image copy)
                        # and the video frames so model.tar.gz does not carry duplicate images.
                        # rmtree removes symlinks without following them.
                        for _fgs_tmp in (os.path.join(config['DATASET_PATH'], "faster_gs_input"),
                                         os.path.join(FASTER_GS_MODELS_DIR, "inference")):
                            if os.path.lexists(_fgs_tmp):
                                if os.path.islink(_fgs_tmp):
                                    os.unlink(_fgs_tmp)
                                else:
                                    shutil.rmtree(_fgs_tmp, ignore_errors=True)
                                log.info(f"Removed Faster-GS temporary data from archive: {_fgs_tmp}")
                    cleanup_dataset(config['DATASET_PATH'])
                    if not os.path.exists(config['DATASET_PATH']):
                        log.error(f"CRITICAL: Dataset missing after cleanup: {config['DATASET_PATH']}")
                        raise RuntimeError(f"Dataset disappeared after cleanup: {config['DATASET_PATH']}")
                    if LOCAL_DEBUG:
                        # Create tarball for local debug mode
                        log.info(f"Creating tarball for local debug mode...")
                        local_tar_path = os.path.join(config['S3_OUTPUT'], config['UUID'], 'output', 'model.tar.gz')
                        log.info(f"Tarball path: {local_tar_path}")
                        os.makedirs(os.path.dirname(local_tar_path), exist_ok=True)
                        create_tarball(config['DATASET_PATH'], local_tar_path, "dataset")
                        log.info(f"Created model.tar.gz archive for local debug: {local_tar_path}")
                        # Copy metrics.json if it exists
                        if os.path.exists(EVAL_METRIC_PATH):
                            metrics_dest = os.path.join(config['S3_OUTPUT'], config['UUID'], 'eval', 'metrics.json')
                            os.makedirs(os.path.dirname(metrics_dest), exist_ok=True)
                            shutil.copy2(EVAL_METRIC_PATH, metrics_dest)
                            log.info(f"Copied metrics.json to: {metrics_dest}")
                    else:
                        # Copy result over to where SM expects it
                        if os.path.exists(EVAL_METRIC_PATH):
                            metrics_dest = os.path.join(config['DATASET_PATH'], 'eval', 'metrics.json')
                            os.makedirs(os.path.dirname(metrics_dest), exist_ok=True)
                            shutil.copy2(EVAL_METRIC_PATH, metrics_dest)
                            log.info(f"Copied metrics.json into dataset for archive: {metrics_dest}")
                        if not IS_BATCH:
                            log.info(f"Moving dataset to where SageMaker expects it...")
                            shutil.move(config['DATASET_PATH'], OUTPUT_DATASET_PATH)
                    
                    log.info(f"Successful pipeline result generation located at "
                            f"{config['S3_OUTPUT']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.*")
                    if IS_BATCH and not LOCAL_DEBUG:
                        # S3 ARCHIVE UPLOAD COMPONENT
                        # Copy nerfstudio_models and config.yml to dataset directory for archive
                        if config['RUN_TRAIN'] == "true" and ENABLE_MULTI_GPU != "true":
                            if config['MODEL'] not in ("3dgut", "3dgrt", "faster-gs"):
                                dataset_source = config['DATASET_PATH']
                                # Copy nerfstudio_models directory to dataset
                                dataset_models_path = os.path.join(dataset_source, "nerfstudio_models")
                                if os.path.exists(model_ckpt_path):
                                    # Remove existing directory first to ensure clean copy
                                    if os.path.exists(dataset_models_path):
                                        shutil.rmtree(dataset_models_path)
                                    shutil.copytree(model_ckpt_path, dataset_models_path)
                                    log.info(f"Copied nerfstudio_models to dataset: {dataset_models_path}")
                                    
                                    # Log checkpoint files for debugging
                                    if os.path.exists(dataset_models_path):
                                        ckpt_files = [f for f in os.listdir(dataset_models_path) if f.endswith('.ckpt')]
                                        log.info(f"Checkpoint files in archive: {ckpt_files}")
                                
                                # Copy config.yml to dataset
                                dataset_config_path = os.path.join(dataset_source, "config.yml")
                                if os.path.exists(model_config_path):
                                    shutil.copy2(model_config_path, dataset_config_path)
                                    log.info(f"Copied config.yml to dataset: {dataset_config_path}")
                        # For Batch, always use DATASET_PATH as source since we copied files there
                        dataset_source = config['DATASET_PATH']
                        if os.path.exists(EVAL_METRIC_PATH):
                            metrics_dest = os.path.join(dataset_source, 'eval', 'metrics.json')
                            os.makedirs(os.path.dirname(metrics_dest), exist_ok=True)
                            shutil.copy2(EVAL_METRIC_PATH, metrics_dest)
                            log.info(f"Copied metrics.json into dataset for Batch archive: {metrics_dest}")
                        cleanup_dataset(dataset_source)
                        # For Batch, create archive with dataset/train structure
                        create_tarball(dataset_source, OUTPUT_TAR_PATH, "dataset/train")
                        log.info(f"Created model.tar.gz archive from {dataset_source} with dataset/train structure")
                        if not LOCAL_DEBUG:
                            os.makedirs(os.path.dirname(OUTPUT_TAR_PATH), exist_ok=True)
                            pipeline.run_component(i)
                            log.info(f"Uploaded model.tar.gz to {config['S3_OUTPUT']}/{config['UUID']}/output/model.tar.gz")
                        # Upload archive to S3
                        os.makedirs(os.path.dirname(OUTPUT_TAR_PATH), exist_ok=True)
                        pipeline.run_component(i)
                        log.info(f"Uploaded model.tar.gz to {config['S3_OUTPUT']}/{config['UUID']}/output/model.tar.gz")
                case "FasterGS-Render-Images":
                    # Video is an optional output: a render failure must not fail the job
                    try:
                        pipeline.run_component(i)
                    except Exception as _render_err:
                        log.warning(f"Faster-GS render failed, continuing without render.mp4: {_render_err}")
                case "Images-to-Video" if IS_FASTER_GS:
                    _frames = list(Path(FASTER_GS_MODELS_DIR, "inference").rglob("*.png"))
                    if not _frames:
                        log.info("Skipping Images-to-Video: no Faster-GS frames were rendered")
                        i += 1
                        continue
                    try:
                        pipeline.run_component(i)
                    except Exception as _video_err:
                        log.warning(f"Faster-GS video encoding failed, continuing without render.mp4: {_video_err}")
                case "Nerfstudio-Metrics" | "3DGRUT-Metrics" | "GSplat-Metrics" | "FasterGS-Metrics":
                    # For GSplat-Metrics, force single GPU and find checkpoint files
                    if component.name == "GSplat-Metrics":
                        original_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                        log.info("Running GSplat metrics with single GPU (CUDA_VISIBLE_DEVICES=0)")
                        
                        # Find and add checkpoint files to args
                        ckpt_dir = os.path.join(output_path, "ckpts")
                        if os.path.exists(ckpt_dir):
                            ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt') and 'rank' in f])
                            if ckpt_files:
                                latest_num = max([int(f.split('_')[1]) for f in ckpt_files])
                                latest_ckpts = [os.path.join(ckpt_dir, f) for f in ckpt_files if f.startswith(f'ckpt_{latest_num}_')]
                                # Replace wildcard in args with actual checkpoint files
                                new_args = []
                                for arg in component.args:
                                    if 'ckpt_*.pt' in arg:
                                        new_args.extend(latest_ckpts)
                                    else:
                                        new_args.append(arg)
                                component.args = new_args
                                log.info(f"Found {len(latest_ckpts)} checkpoint files for evaluation")
                    
                    try:
                        pipeline.run_component(i)
                    except RuntimeError as _metrics_err:
                        log.warning(f"Nerfstudio-Metrics failed (non-fatal): {_metrics_err}")
                    
                    # Restore original CUDA setting
                    if component.name == "GSplat-Metrics":
                        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda
                    
                    # Log evaluation metrics after component runs
                    if os.path.exists(EVAL_METRIC_PATH):
                        try:
                            with open(EVAL_METRIC_PATH, 'r') as f:
                                metrics_data = json.load(f)
                            results = metrics_data.get('results', {})
                            # Handle both splatfacto (psnr) and dn-splatter (rgb_psnr) key formats
                            psnr = results.get('psnr', results.get('rgb_psnr', None))
                            ssim = results.get('ssim', results.get('rgb_ssim', None))
                            lpips = results.get('lpips', results.get('rgb_lpips', None))
                            psnr_str = f"{psnr:.4f}" if psnr is not None else 'N/A'
                            ssim_str = f"{ssim:.4f}" if ssim is not None else 'N/A'
                            lpips_str = f"{lpips:.4f}" if lpips is not None else 'N/A'
                            log.info(f"Evaluation Metrics - PSNR: {psnr_str}, SSIM: {ssim_str}, LPIPS: {lpips_str}")
                        except Exception as e:
                            log.warning(f"Could not read evaluation metrics: {e}")
                    elif component.name == "3DGRUT-Metrics":
                        # 3DGRUT render.py writes its own metrics.json under EVAL_METRIC_FOLDER
                        # Find the most recently written metrics.json in any subdirectory
                        threedgrut_metrics = None
                        for metrics_file in sorted(Path(EVAL_METRIC_FOLDER).rglob("metrics.json")):
                            try:
                                with open(metrics_file, 'r') as f:
                                    raw = json.load(f)
                                # 3DGRUT format: {"mean_psnr": x, "mean_ssim": x, "mean_lpips": x, ...}
                                if "mean_psnr" in raw:
                                    threedgrut_metrics = {
                                        "psnr": float(raw["mean_psnr"]),
                                        "ssim": float(raw["mean_ssim"]),
                                        "lpips": float(raw["mean_lpips"])
                                    }
                                    break
                            except Exception as _e:
                                log.warning(f"Could not read 3DGRUT metrics from {metrics_file}: {_e}")
                        if threedgrut_metrics:
                            os.makedirs(os.path.dirname(EVAL_METRIC_PATH), exist_ok=True)
                            with open(EVAL_METRIC_PATH, 'w') as f:
                                json.dump({"results": threedgrut_metrics}, f, indent=2)
                            log.info(f"Evaluation Metrics - PSNR: {threedgrut_metrics['psnr']:.4f}, SSIM: {threedgrut_metrics['ssim']:.4f}, LPIPS: {threedgrut_metrics['lpips']:.4f}")
                        elif hasattr(component, 'output') and component.output:
                            # Fallback: parse from log output
                            parse_3dgrut_metrics_from_log(component.output, EVAL_METRIC_PATH)
                            if os.path.exists(EVAL_METRIC_PATH):
                                with open(EVAL_METRIC_PATH, 'r') as f:
                                    metrics_data = json.load(f)
                                results = metrics_data.get('results', {})
                                log.info(f"Evaluation Metrics - PSNR: {results.get('psnr', 'N/A'):.4f}, SSIM: {results.get('ssim', 'N/A'):.4f}, LPIPS: {results.get('lpips', 'N/A'):.4f}")
                    elif component.name == "GSplat-Metrics":
                        if hasattr(component, 'output') and component.output:
                            parse_gsplat_metrics_from_log(component.output, EVAL_METRIC_PATH)
                            if os.path.exists(EVAL_METRIC_PATH):
                                with open(EVAL_METRIC_PATH, 'r') as f:
                                    metrics_data = json.load(f)
                                results = metrics_data.get('results', {})
                                log.info(f"Evaluation Metrics - PSNR: {results.get('psnr', 'N/A'):.4f}, SSIM: {results.get('ssim', 'N/A'):.4f}, LPIPS: {results.get('lpips', 'N/A'):.4f}")
                case "S3-Export-Video" | "S3-Export-Spz" | "S3-Export-Usdz" | "S3-Export-Thumbnail" | "S3-Export-Ply" | "S3-Export-Sog":
                    # Skip S3 upload gracefully if the source file doesn't exist
                    # (e.g. video/thumbnail skipped when dn-splatter sensor-depth mode runs)
                    if component.name in ("S3-Export-Video", "S3-Export-Thumbnail"):
                        _src = component.args[2] if len(component.args) > 2 else ""
                        if not os.path.exists(_src):
                            log.info(f"Skipping {component.name}: source file not found: {_src}")
                            i += 1
                            continue
                    if LOCAL_DEBUG:
                        if component.name == "S3-Export-Video":
                            copy_to_local_output(os.path.join(output_path, "render.mp4"), config, 
                                               f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.mp4", log)
                        elif component.name == "S3-Export-Spz":
                            copy_to_local_output(spz_path, config, 
                                               f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.spz", log)
                        elif component.name == "S3-Export-Usdz":
                            copy_to_local_output(usdz_path, config, 
                                               f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.usdz", log)
                        elif component.name == "S3-Export-Thumbnail":
                            copy_to_local_output(os.path.join(output_path, "render_thumbnail.png"), config, 
                                               "render_thumbnail.png", log)
                        elif component.name == "S3-Export-Ply":
                            if config['MODEL'] == "nerfacto":
                                glb_path = os.path.join(output_path, "textured", f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.glb")
                                copy_to_local_output(glb_path, config, f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.glb", log)
                            else:
                                copy_to_local_output(orig_ply_path, config, f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.ply", log)
                        elif component.name == "S3-Export-Sog":
                            copy_to_local_output(sog_path, config, f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.sog", log)
                    else:
                        pipeline.run_component(i)
                case "S3-Export-Archive":
                    if LOCAL_DEBUG:
                        # Archive was already created in the first S3-Export-Archive case above
                        log.info("Archive already handled in LOCAL_DEBUG mode")
                    else:
                        pipeline.run_component(i)
                case "Ply-Rotation":
                    pipeline.run_component(i)
                    # For normalized gsplat, orig.ply is now correctly oriented.
                    # Copy it to spz.ply so SPZ starts from the same correct orientation
                    # instead of the pre-rotation splat.ply copy.
                    if (ENABLE_MULTI_GPU == "true" or IS_GSPLAT_DEPTH) and \
                            config.get('PRESERVE_SCENE_SCALE', 'false').lower() != 'true' and \
                            config['ENABLE_SPZ'] == "true" and os.path.exists(orig_ply_path):
                        shutil.copy2(orig_ply_path, spz_ply_path)
                        log.info(f"Copied rotated orig.ply to spz.ply for gsplat SPZ")
                case "Extract-Mesh-GS":
                    # Mesh extraction is an optional extra output: a failure here must
                    # not fail a job whose splat outputs were already produced. The
                    # dn-splatter extractor (Extract-Mesh) already exits 0 on failure.
                    try:
                        pipeline.run_component(i)
                    except Exception as _mesh_err:
                        log.warning(f"Mesh extraction failed, continuing without mesh.glb: {_mesh_err}")
                case "S3-Export-Mesh":
                    # Skip gracefully if mesh.glb wasn't produced (extraction failed or was skipped)
                    _mesh_glb = os.path.join(output_path, "mesh.glb")
                    if not os.path.exists(_mesh_glb):
                        log.info(f"Skipping S3-Export-Mesh: mesh.glb not found (mesh extraction was skipped)")
                        i += 1
                        continue
                    if LOCAL_DEBUG:
                        copy_to_local_output(_mesh_glb, config,
                                             f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.mesh.glb", log)
                    else:
                        pipeline.run_component(i)
                case "S3-Export-Collision":
                    # Zip all collision files into a single archive before uploading
                    base_name = str(os.path.splitext(config['FILENAME'])[0]).lower()
                    collision_zip_path = os.path.join(output_path, f"{base_name}-collision.zip")
                    try:
                        import zipfile as _zipfile
                        with _zipfile.ZipFile(collision_zip_path, 'w', _zipfile.ZIP_DEFLATED) as _zf:
                            for _ext in [".voxel.json", ".voxel.bin", ".collision.glb"]:
                                _src = os.path.join(output_path, f"splat{_ext}")
                                if os.path.exists(_src):
                                    _zf.write(_src, arcname=f"splat{_ext}")
                        log.info(f"Created collision zip: {collision_zip_path} ({os.path.getsize(collision_zip_path)} bytes)")
                    except Exception as _e:
                        log.warning(f"Could not create collision zip: {_e}")
                    pipeline.run_component(i)
                case "S3-Export-LOD":
                    # Zip all LOD files into a single archive before uploading
                    base_name = str(os.path.splitext(config['FILENAME'])[0]).lower()
                    lod_zip_path = os.path.join(output_path, f"{base_name}-lod.zip")
                    try:
                        import zipfile as _zipfile
                        with _zipfile.ZipFile(lod_zip_path, 'w', _zipfile.ZIP_DEFLATED) as _zf:
                            if os.path.isdir(lod_dir):
                                for _root, _dirs, _files in os.walk(lod_dir):
                                    for _fname in _files:
                                        _src = os.path.join(_root, _fname)
                                        _arcname = os.path.relpath(_src, lod_dir)
                                        _zf.write(_src, arcname=_arcname)
                        log.info(f"Created LOD zip: {lod_zip_path} ({os.path.getsize(lod_zip_path)} bytes)")
                    except Exception as _e:
                        log.warning(f"Could not create LOD zip: {_e}")
                    pipeline.run_component(i)
                case "Clean-Point-Cloud":
                    # splat-transform --filter-floaters default voxel size is 0.05m.
                    # For large scenes this creates too many voxels and SIGSEGV from
                    # WebGPU buffer limits. Compute safe voxel size from PLY bbox.
                    # Safe limit ~2M voxels empirically; scale up voxel size if needed.
                    try:
                        import struct as _struct
                        _ply = component.args[0]  # input PLY path
                        _voxel_size = 0.05  # default
                        if os.path.isfile(_ply):
                            try:
                                import numpy as _np
                                with open(_ply, 'rb') as _pf:
                                    _header = b''
                                    while True:
                                        _line = _pf.readline()
                                        _header += _line
                                        if _line.strip() == b'end_header':
                                            break
                                    _header_str = _header.decode('ascii', errors='ignore')
                                    _num_verts = 0
                                    _props = []
                                    for _hl in _header_str.splitlines():
                                        if _hl.startswith('element vertex'):
                                            _num_verts = int(_hl.split()[-1])
                                        elif _hl.startswith('property float x') or _hl.startswith('property float32 x'):
                                            _props.append('x')
                                        elif _hl.startswith('property float y') or _hl.startswith('property float32 y'):
                                            _props.append('y')
                                        elif _hl.startswith('property float z') or _hl.startswith('property float32 z'):
                                            _props.append('z')
                                    if _num_verts > 0 and len(_props) >= 3:
                                        # Sample up to 50K gaussians to estimate bbox
                                        _step = max(1, _num_verts // 50000)
                                        _row_bytes = 4 * 3  # at minimum x,y,z floats — estimate stride
                                        # Read all vertex data and stride through it
                                        _raw = _pf.read()
                                        # Estimate bytes per vertex from total data
                                        _stride = len(_raw) // _num_verts if _num_verts else 1
                                        _xs, _ys, _zs = [], [], []
                                        for _vi in range(0, _num_verts, _step):
                                            _off = _vi * _stride
                                            if _off + 12 <= len(_raw):
                                                _x, _y, _z = _struct.unpack_from('<fff', _raw, _off)
                                                if not (_np.isnan(_x) or _np.isinf(_x)):
                                                    _xs.append(_x); _ys.append(_y); _zs.append(_z)
                                        if _xs:
                                            _extent_x = max(_xs) - min(_xs)
                                            _extent_y = max(_ys) - min(_ys)
                                            _extent_z = max(_zs) - min(_zs)
                                            # Target ~2M voxels max
                                            _target_voxels = 2_000_000
                                            _min_size = (_extent_x * _extent_y * _extent_z / _target_voxels) ** (1/3)
                                            _voxel_size = max(0.05, round(_min_size * 20) / 20)  # round to nearest 0.05m
                                            _est_voxels = int((_extent_x / _voxel_size) * (_extent_y / _voxel_size) * (_extent_z / _voxel_size))
                                            log.info(f"Clean-Point-Cloud: scene {_extent_x:.1f}x{_extent_y:.1f}x{_extent_z:.1f}m, "
                                                     f"voxel_size={_voxel_size}m (~{_est_voxels//1000}K voxels)")
                                            if _voxel_size > 0.05:
                                                # Update args with explicit voxel size
                                                _ff_idx = component.args.index('--filter-floaters')
                                                component.args[_ff_idx] = f'--filter-floaters={_voxel_size}'
                            except Exception as _bbox_err:
                                log.warning(f"Clean-Point-Cloud: bbox estimation failed ({_bbox_err}), using default 0.05m")
                        pipeline.run_component(i)
                    except RuntimeError as _clean_err:
                        log.warning(f"Clean-Point-Cloud failed (non-fatal, skipping): {_clean_err}")
                case "Generate-LOD":
                    try:
                        pipeline.run_component(i)
                    except RuntimeError as _lod_err:
                        log.warning(f"Generate-LOD failed (non-fatal, skipping): {_lod_err}")
                case "Calibrate-Rig":
                    # Non-fatal: calibrate_rig.py keeps (or restores) the original
                    # poses whenever it cannot produce a trustworthy rig.
                    try:
                        pipeline.run_component(i)
                    except RuntimeError as _rig_err:
                        log.warning(f"Calibrate-Rig failed (non-fatal, original poses kept): {_rig_err}")
                    _rig_report = os.path.join(config['DATASET_PATH'], "calibrate_rig", "report.json")
                    if os.path.isfile(_rig_report):
                        try:
                            with open(_rig_report) as _rf:
                                _rr = _json.load(_rf)
                            log.info(f"Calibrate-Rig: {_rr.get('status')} - {_rr.get('reason')}")
                        except Exception:
                            pass
                case "Bundle-Adjust-Colmap-Model":
                    # Non-fatal: bundle_adjust_model.py keeps (or restores) the
                    # original model whenever the refinement fails its checks.
                    try:
                        pipeline.run_component(i)
                    except RuntimeError as _ba_err:
                        log.warning(f"Bundle-Adjust-Colmap-Model failed (non-fatal, original poses kept): {_ba_err}")
                    _ba_report = os.path.join(config['DATASET_PATH'], "bundle_adjust", "report.json")
                    if os.path.isfile(_ba_report):
                        try:
                            with open(_ba_report) as _bf:
                                _br = _json.load(_bf)
                            log.info(f"Bundle-Adjust-Colmap-Model: {_br.get('status')} - {_br.get('reason')}")
                        except Exception:
                            pass
                case "Enhance-Depth":
                    # Resolve depth dir at runtime — files may have been extracted from zip
                    _rt_depth_dir = os.path.join(config['DATASET_PATH'], "depth")
                    if not os.path.isdir(_rt_depth_dir):
                        _rt_depth_dir = os.path.join(config['DATASET_PATH'], "depth_images")
                    # Raw ARKit LiDAR (depths/*.bin) is converted to depth/*.png at
                    # training time by dn_splatter_preprocess; convert it now so
                    # PromptDA can enhance it (the dense 256x192 map is PromptDA's
                    # native prompt).
                    _bins_dir = os.path.join(config['DATASET_PATH'], "depths")
                    if not os.path.isdir(_rt_depth_dir) and os.path.isdir(_bins_dir) and any(
                        f.lower().endswith(".bin") for f in os.listdir(_bins_dir)
                    ):
                        try:
                            _train_dir = os.path.join(current_dir_path, 'training')
                            if _train_dir not in sys.path:
                                sys.path.insert(0, _train_dir)
                            from dn_splatter_preprocess import convert_lidar_bins
                            _n = convert_lidar_bins(config['DATASET_PATH'])
                            log.info(f"Enhance-Depth: converted {_n} ARKit LiDAR .bin maps to depth/")
                            _rt_depth_dir = os.path.join(config['DATASET_PATH'], "depth")
                        except Exception as _e:
                            log.warning(f"Enhance-Depth: LiDAR .bin conversion failed: {_e}")
                    if config['MODEL'] not in ("dn-splatter", "dn-splatter-big", "ags-mesh"):
                        log.warning(f"ENHANCE_DEPTH=true, but MODEL={config['MODEL']} does not train on "
                                    "depth maps (only dn-splatter/dn-splatter-big/ags-mesh do); "
                                    "the enhanced depth will not be used")
                    if os.path.isdir(_rt_depth_dir):
                        # Patch the --depth-dir arg with the resolved path
                        try:
                            _dd_idx = component.args.index("--depth-dir")
                            component.args[_dd_idx + 1] = _rt_depth_dir
                        except ValueError:
                            pass
                        pipeline.run_component(i)
                    else:
                        log.warning("ENHANCE_DEPTH=true but no depth/, depth_images/ or depths/*.bin found, skipping")
                case "Filter-Blurry-Images":
                    # A zip that turned out to hold a video was already filtered by
                    # sharp-frame-extractor in VideoToImages.
                    if globals().get('video_found', False):
                        log.info("Filter-Blurry-Images: input zip contained a video (already filtered), skipping")
                    else:
                        try:
                            pipeline.run_component(i)
                        except RuntimeError as _blur_err:
                            log.warning(f"Filter-Blurry-Images failed (non-fatal, all images kept): {_blur_err}")
                case "Remove-Lens-Flare" | "Remove-Sky":
                    # Non-fatal: if masking fails, train without these masks rather than
                    # losing the job. Any image left without a mask gets an all-white one,
                    # because nerfstudio's --masks-path expects a mask for every image.
                    try:
                        pipeline.run_component(i)
                    except RuntimeError as _mask_err:
                        log.warning(f"{component.name} failed (non-fatal, continuing without it): {_mask_err}")
                    _masks_dir = os.path.join(config['DATASET_PATH'], "masks")
                    _filled = 0
                    for _r, _d, _fs in os.walk(image_path):
                        for _f in _fs:
                            if not _f.lower().endswith(('.png', '.jpg', '.jpeg')):
                                continue
                            _rel = os.path.relpath(os.path.join(_r, _f), image_path)
                            _cands = [os.path.join(_masks_dir, _rel + ".png"),
                                      os.path.join(_masks_dir, os.path.splitext(_rel)[0] + ".png")]
                            if all(os.path.isfile(_c) for _c in _cands):
                                continue
                            with Image.open(os.path.join(_r, _f)) as _im:
                                _blank = Image.new("L", _im.size, 255)
                            for _c in _cands:
                                if not os.path.isfile(_c):
                                    os.makedirs(os.path.dirname(_c), exist_ok=True)
                                    _blank.save(_c)
                            _filled += 1
                    if _filled:
                        log.info(f"{component.name}: wrote all-white masks for {_filled} images without one")
                case "Ply-to-Video":
                    # ns-render interpolate follows the eval cameras by default. With fewer
                    # than 2 eval images (small datasets) there is nothing to interpolate
                    # between and it fails with "need at least one array to stack", so
                    # retry along the training cameras instead.
                    try:
                        pipeline.run_component(i)
                    except RuntimeError as _vid_err:
                        if "--pose-source" in component.args:
                            raise
                        log.warning(f"Ply-to-Video failed on the eval camera path ({_vid_err}); "
                                    "retrying with --pose-source train")
                        component.args.extend(["--pose-source", "train"])
                        pipeline.run_component(i)
                case _: # Default case, run Component
                    pipeline.run_component(i)
                    # After autoscale runs, normalize image dimensions
                    if component.name == "AutoscaleDataset":
                        resize_images_to_common_dimensions(image_path)

            i += 1

        pipeline.session.status = Status.STOP
        log.info(f"Pipeline status changed to {pipeline.session.status}")
        end_time = int(time.time())
        total_time = end_time - start_time
        
        # Calculate actual phase durations only for phases that ran
        phase_durations = {}
        phase_names = ['PRE_PROCESSING', 'RECONSTRUCTION', 'TRAINING', 'POST_PROCESSING']
        
        for i, phase_name in enumerate(phase_names):
            if phase_name in phase_start_times:
                # Find the next phase that actually ran
                next_phase_time = end_time
                for next_phase in phase_names[i+1:]:
                    if next_phase in phase_start_times:
                        next_phase_time = phase_start_times[next_phase]
                        break
                phase_durations[phase_name] = next_phase_time - phase_start_times[phase_name]
        
        # Write final phase completion if exists, hasn't been written yet, and DDB table is configured
        if last_phase and ddb_table_name and last_phase in phase_start_times:
            final_phase_elapsed = end_time - phase_start_times[last_phase]
            phase_durations[last_phase] = final_phase_elapsed
            log.info(f"Writing final phase completion for {last_phase}: {final_phase_elapsed}s")
            update_component_phase_completion(
                uuid=config['UUID'],
                table_name=ddb_table_name,
                phase_name=last_phase,
                elapsed_time=final_phase_elapsed,
                log=log
            )
        elif last_phase and not ddb_table_name:
            log.debug(f"Skipping final DynamoDB update for {last_phase} - no table name configured")
        
        # Update comp_group_elapsed_time with actual durations (only for phases that ran)
        pipeline.session.comp_group_elapsed_time = [phase_durations.get(name, 0) for name in pipeline.session.comp_group_names]
        
        log.info(f"Total Pipeline Time: {total_time}s")
        log.info(f"Phase durations: {phase_durations}")
        matcher_info = config['MATCHING_METHOD']
        if 'auto_matcher_overlap' in dir() and auto_matcher_overlap is not None:
            matcher_info += f" (overlap: {auto_matcher_overlap:.1%})"
        log.info(f"Mapper: {config['RECON_SOFTWARE_NAME']}  |  Matcher: {matcher_info}  |  Model: {config['MODEL']}")
        # Dataset summary: image count, resolution, autogroup prefix
        try:
            summary_imgs = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            summary_res = 'N/A'
            if summary_imgs:
                _s = Image.open(os.path.join(image_path, summary_imgs[0]))
                summary_res = f"{_s.width}x{_s.height}"
            summary_group = config.get('AUTOGROUP_TARGET_NAME', '') if config.get('AUTOGROUP_IMAGES', 'false') == 'true' else 'off'
            summary_autoscale = config.get('AUTOSCALE_DATASET_MODE', 'resize').upper() if config.get('AUTOSCALE_DATASET', 'false') == 'true' else 'off'
            log.info(f"Dataset: {len(summary_imgs)} images @ {summary_res}  |  Autogroup: {summary_group}  |  Autoscale: {summary_autoscale}")
        except Exception:
            pass
        # Log splat summary (gaussian count, SH bands) using splat-transform --summary
        try:
            _summary_ply = orig_ply_path if os.path.exists(orig_ply_path) else ply_path
            if os.path.exists(_summary_ply):
                # splat-transform 3.x replaced --summary with --info (structure) and
                # --stats (per-column statistics). Log --info for readability.
                _summary_result = subprocess.run(
                    ["splat-transform", _summary_ply, "--info", "text", "null"],
                    capture_output=True, text=True
                )
                _summary_out = _summary_result.stdout + _summary_result.stderr
                log.info(f"Splat Summary:\n{_summary_out}")
                # Read gaussian count and SH bands from the PLY header for DynamoDB,
                # so the metrics don't depend on splat-transform's output format.
                _splat_metrics = {}
                _f_rest = 0
                with open(_summary_ply, 'rb') as _ph:
                    for _raw in _ph:
                        _line = _raw.decode('ascii', 'ignore').strip()
                        if _line.startswith('element vertex '):
                            _splat_metrics['gaussian_count'] = int(_line.split()[-1])
                        elif _line.startswith('property') and _line.split()[-1].startswith('f_rest_'):
                            _f_rest += 1
                        elif _line == 'end_header':
                            break
                # f_rest_* holds 3 * ((bands + 1)^2 - 1) coefficients: 0/9/24/45 -> 0/1/2/3 bands
                _splat_metrics['sh_bands'] = {0: 0, 9: 1, 24: 2, 45: 3}.get(_f_rest, 0)
                if _splat_metrics and ddb_table_name:
                    update_dynamodb_metrics(
                        uuid=config['UUID'],
                        table_name=ddb_table_name,
                        metrics=_splat_metrics,
                        log=log
                    )
        except Exception as _e:
            log.warning(f"Could not collect splat summary metrics: {_e}")
        # Update DynamoDB with final timing information only if table is configured
        if ddb_table_name:
            log.info(f"Updating DynamoDB metrics for UUID {config['UUID']}")
            update_dynamodb_metrics(
                uuid=config['UUID'],
                table_name=ddb_table_name,
                comp_group_elapsed_time=pipeline.session.comp_group_elapsed_time,
                log=log
            )
        else:
            log.debug("Skipping DynamoDB metrics update - no table name configured")
        
        # Extract and update training metrics if available
        if os.path.exists(EVAL_METRIC_PATH):
            try:
                with open(EVAL_METRIC_PATH, 'r') as f:
                    metrics_data = json.load(f)
                results = metrics_data.get('results', {})
                training_metrics = {
                    'psnr': float(results.get('psnr', results.get('rgb_psnr', 0))),
                    'ssim': float(results.get('ssim', results.get('rgb_ssim', 0))),
                    'lpips': float(results.get('lpips', results.get('rgb_lpips', 0)))
                }
                if ddb_table_name:
                    log.info(f"Updating training metrics in DynamoDB for UUID {config['UUID']}")
                    update_dynamodb_metrics(
                        uuid=config['UUID'],
                        table_name=ddb_table_name,
                        metrics=training_metrics,
                        log=log
                    )
                else:
                    log.debug("Skipping training metrics DynamoDB update - no table name configured")
            except Exception as e:
                log.warning(f"Could not extract/update training metrics: {e}")

        ##################################
        # STEP FUNCTIONS TASK TOKEN CALLBACK
        # Notify Step Functions of successful completion when using waitForTaskToken
        # pattern. Skipped in LOCAL_DEBUG mode and when no token is present.
        ##################################
        if ENABLE_TASK_TOKEN_CALLBACK:
            _heartbeat_stop.set()  # stop the heartbeat thread before calling back
            send_task_success(
                task_token=TASK_TOKEN,
                output={"status": "SUCCEEDED", "uuid": config['UUID']},
                log=log
            )
    except Exception as e:
        error_message = f"General error running the pipeline: {e}"
        pipeline.report_error(795, error_message)
        # Read token and debug flag directly from env in case the crash happened
        # before TASK_TOKEN, LOCAL_DEBUG, or ENABLE_TASK_TOKEN_CALLBACK were assigned.
        # LOCAL_DEBUG check prevents spurious SendTaskFailure calls during local runs.
        _task_token = locals().get('TASK_TOKEN') or os.environ.get('TASK_TOKEN', '')
        _local_debug = locals().get('LOCAL_DEBUG') or os.environ.get('LOCAL_DEBUG', 'false').lower() == 'true'
        _is_batch = 'AWS_BATCH_JOB_ID' in os.environ
        if _is_batch and bool(_task_token) and not _local_debug:
            send_task_failure(
                task_token=_task_token,
                error="PipelineError",
                cause=error_message,
                log=locals().get('log')
            )
