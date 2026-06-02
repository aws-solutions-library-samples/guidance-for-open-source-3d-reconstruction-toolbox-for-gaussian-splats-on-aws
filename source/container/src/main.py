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
765, "Issue running the training session stage"
770, "Issue exporting splat from NerfStudio"
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
    setup_local_debug, copy_to_local_output, print_container_version_info,
    update_dynamodb_metrics, update_component_phase_completion,
    parse_3dgrut_metrics_from_log, parse_gsplat_metrics_from_log,
    send_task_success, send_task_failure, send_task_heartbeat
)

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
        ENABLE_DEPTH_LOSS = config.get('ENABLE_DEPTH_LOSS', 'false').lower() == 'true'

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
    os.environ['PYTORCH_CUDA_ALLOC_CONF']= 'expandable_segments:True'
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
    transforms_in_path = os.path.join(config['DATASET_PATH'], "transforms-in.json")
    transforms_out_path = os.path.join(config['DATASET_PATH'], "transforms.json")
    colmap_vocab_path = os.path.join(config['CODE_PATH'], "vocab_tree_flickr100K_words32K.bin")

    if config['MODEL'] == "splatfacto" or config['MODEL'] == "splatfacto-big" or config['MODEL'] == "splatfacto-mcmc":
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
            elif os.path.isdir(os.path.join(config['DATASET_PATH'], config['FILENAME'])):
                dir_path = os.path.join(config['DATASET_PATH'], config['FILENAME'])
                has_images = os.path.exists(os.path.join(dir_path, 'images'))
                has_sparse = os.path.exists(os.path.join(dir_path, 'sparse')) or \
                            os.path.exists(os.path.join(dir_path, 'colmap', 'sparse'))
                if has_images and has_sparse:
                    colmap_zip_found = True
                    log.info(f"Detected pre-extracted COLMAP reconstruction directory: {config['FILENAME']}")
            # Check if sparse/ exists directly in DATASET_PATH (already extracted)
            elif os.path.exists(os.path.join(config['DATASET_PATH'], 'sparse')) and \
                 os.path.exists(os.path.join(config['DATASET_PATH'], 'images')):
                colmap_zip_found = True
                log.info(f"Detected COLMAP reconstruction data already in dataset directory")
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
        
        # Handle zip file extraction
        if config['FILENAME'].endswith('.zip') and os.path.exists(zip_path):
            # Extract zip to temp directory
            temp_path = os.path.join(config['DATASET_PATH'], 'temp')
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Validate entries to prevent zip slip attacks
                for entry in zip_ref.namelist():
                    entry_path = os.path.realpath(os.path.join(temp_path, entry))
                    if not entry_path.startswith(os.path.realpath(temp_path) + os.sep):
                        raise ValueError(f"Zip slip detected: {entry}")
                zip_ref.extractall(temp_path)
            
            # Check if extracted content is in a subdirectory
            temp_contents = os.listdir(temp_path)
            if len(temp_contents) == 1 and os.path.isdir(os.path.join(temp_path, temp_contents[0])):
                extract_source = os.path.join(temp_path, temp_contents[0])
            else:
                extract_source = temp_path
            
            # Verify required COLMAP structure exists
            has_images = os.path.exists(os.path.join(extract_source, 'images'))
            has_sparse = os.path.exists(os.path.join(extract_source, 'sparse')) or \
                        os.path.exists(os.path.join(extract_source, 'colmap', 'sparse'))
            has_transforms = os.path.exists(os.path.join(extract_source, 'transforms.json'))
            
            if has_images and (has_sparse or has_transforms):
                log.info(f"Valid COLMAP reconstruction found: images={has_images}, sparse={has_sparse}, transforms={has_transforms}")
                if has_sparse and not has_transforms:
                    colmap_zip_needs_conversion = True
                    log.info("No transforms.json in zip - Colmap-to-Nerfstudio conversion will be required")

                # Detect masks directory in the zip
                has_masks = os.path.exists(os.path.join(extract_source, 'masks'))
                if has_masks:
                    log.info("Masks directory detected in zip — will enable mask training")
                
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
                    # If transforms.json already exists (no conversion needed), inject mask_path now
                    # so gsplat can find the masks without relying on the Colmap-to-Nerfstudio step.
                    transforms_path = os.path.join(config['DATASET_PATH'], 'transforms.json')
                    if has_transforms and os.path.isfile(transforms_path):
                        with open(transforms_path, 'r') as _f:
                            _data = _json.load(_f)
                        _injected = 0
                        for _frame in _data.get('frames', []):
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
        
        # Ensure colmap/sparse structure exists for NerfStudio (not needed for 3DGRUT or depth loss)
        if colmap_zip_found and config['MODEL'] not in ('3dgrt', '3dgut') and not ENABLE_DEPTH_LOSS:
            if not os.path.exists(os.path.join(config['DATASET_PATH'], 'colmap', 'sparse')):
                if os.path.exists(os.path.join(config['DATASET_PATH'], 'sparse')):
                    colmap_dir = os.path.join(config['DATASET_PATH'], 'colmap')
                    os.makedirs(colmap_dir, exist_ok=True)
                    shutil.move(os.path.join(config['DATASET_PATH'], 'sparse'), 
                               os.path.join(colmap_dir, 'sparse'))
                    log.info(f"Moved sparse/ to colmap/sparse/ for NerfStudio compatibility")
            
            log.info(f"Successfully processed COLMAP reconstruction from {config['FILENAME']}")
            log.info(f"Dataset contents: {os.listdir(config['DATASET_PATH'])}")
            # Verify colmap/sparse/0 has required files
            sparse_0 = os.path.join(config['DATASET_PATH'], 'colmap', 'sparse', '0')
            if os.path.exists(sparse_0):
                log.info(f"colmap/sparse/0 contents: {os.listdir(sparse_0)}")
            else:
                log.warning(f"colmap/sparse/0 does not exist after extraction")

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
                    log.info(f"Keeping config.yml in dataset directory for splatfacto: {config_yml_src}")
                    
                    # Update config.yml for resume training directly in dataset directory using text replacement
                    try:
                        with open(config_yml_src, 'r') as f:
                            config_content = f.read()

                        # Find latest checkpoint for load_checkpoint field
                        ckpt_files = sorted([f for f in os.listdir(model_src_dir) if f.endswith('.ckpt')])
                        if ckpt_files:
                            latest_ckpt = os.path.join(model_src_dir, ckpt_files[-1])
                            
                            # Update load_checkpoint in config
                            config_content = re.sub(
                                r'load_checkpoint: null',
                                f'load_checkpoint: !!python/object/apply:pathlib.PosixPath\n  - {latest_ckpt}',
                                config_content
                            )
                        
                        # Compute total iterations = checkpoint step + additional steps
                        # so nerfstudio continues from where it left off rather than restarting
                        ckpt_step = 0
                        if ckpt_files:
                            _m = re.search(r'step-(\d+)\.ckpt$', ckpt_files[-1])
                            if _m:
                                ckpt_step = int(_m.group(1)) + 1  # +1 because step is 0-indexed
                        additional_steps = max(int(config['MAX_STEPS']), 9000)
                        total_iterations = ckpt_step + additional_steps
                        log.info(f"Resume training: checkpoint_step={ckpt_step}, additional_steps={additional_steps}, total_iterations={total_iterations}")
                        
                        # Update max_num_iterations using text replacement
                        config_content = re.sub(r'max_num_iterations: \d+', f'max_num_iterations: {total_iterations}', config_content)

                        # Update timestamp using text replacement
                        config_content = re.sub(r'timestamp: [^\n]+', f'timestamp: {RESUME_TRAIN_EXPERIMENT_NAME}', config_content)
                        
                        # Set load_scheduler to false
                        config_content = re.sub(r'load_scheduler: \w+', 'load_scheduler: false', config_content)
                        
                        # Update dataset path using text replacement
                        config_content = re.sub(r'data: !!python/object/apply:pathlib\.PosixPath\s*\n\s*-[^\n]*(?:\n\s*-[^\n]*)*', 
                                              f'data: !!python/object/apply:pathlib.PosixPath\n      - {config["DATASET_PATH"]}', 
                                              config_content, flags=re.MULTILINE)
                        
                        # Update checkpoint path to point to dataset directory
                        checkpoint_path = os.path.join(config['DATASET_PATH'], 'nerfstudio_models')
                        config_content = re.sub(r'outputs/unnamed/splatfacto/[^/]+/nerfstudio_models', 
                                              checkpoint_path.replace('\\', '/'),
                                              config_content)
                        
                        with open(config_yml_src, 'w') as f:
                            f.write(config_content)
                        
                        log.info(f"""
                                Updated config.yml in dataset directory:
                                load_checkpoint={latest_ckpt if ckpt_files else 'null'},
                                max_num_iterations={total_iterations},
                                timestamp={RESUME_TRAIN_EXPERIMENT_NAME},
                                data_path={config['DATASET_PATH']}
                                """)
                    except Exception as e:
                        log.warning(f"Failed to update config.yml: {e}")
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
    if int(pipeline.config.num_gpus) > 1:
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
            # Use sharp-frame-extractor if blur filtering is enabled, otherwise use simple extraction
            if config['FILTER_BLURRY_IMAGES'] == "true" and config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'false' and \
               config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'false':
                # Add 8% to max images to compensate for blur filtering reduction
                adjusted_max_images = str(int(int(config['MAX_NUM_IMAGES']) * 1.08))
                args = [
                    "-i", input_file_path,
                    "-o", image_path,
                    "-n", adjusted_max_images,
                    "-ll", config['LOG_VERBOSITY'].upper()
                ]
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
        # Autogroup Images by Prefix
    ##################################
    try:
        if config.get('AUTOGROUP_IMAGES', 'false') == 'true' and config['RUN_RECON'] == 'true':
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
            model = "u2net"

            args = [
                "-i", image_path,
                "-o", image_path,
                "-nt", str(pipeline.config.num_threads),
                "-ng", str(pipeline.config.num_gpus),
                "-m", model
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
            args = [
                "--input_image_path", image_path,
                "--output_path", config['DATASET_PATH'],
                "--matcher", method
            ]
            if faces_to_remove and faces_to_remove != '[]':
                args.append("--remove_faces")

            if config['REMOVE_OBJECT'] == "true":
                model = "u2net_human_seg"
                try:
                    objects_list = ast.literal_eval(config['OBJECT_REMOVAL_OBJECTS'])
                    if "human" in [obj.lower() for obj in objects_list]:
                        model = "u2net_human_seg"
                except (ValueError, SyntaxError):
                    if "human" in config['OBJECT_REMOVAL_OBJECTS'].lower():
                        model = "u2net_human_seg"
                args.extend(["--remove_object",
                             "--object_action", config['OBJECT_REMOVAL_ACTION'],
                             "-m", model,
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
    try:
        # Skip object removal if using spherical camera (handled in panorama_sfm.py)
        if config['REMOVE_OBJECT'] == "true" and config['RUN_RECON'] == "true" and config['SPHERICAL_CAMERA'] != "true":
            model = None
            # OBJECT REMOVAL COMPONENT FOR HUMAN
            try:
                objects_list = ast.literal_eval(config['OBJECT_REMOVAL_OBJECTS'])
                if "human" in [obj.lower() for obj in objects_list]:
                    model = "u2net_human_seg"
            except (ValueError, SyntaxError):
                # Fallback to string check if parsing fails
                if "human" in config['OBJECT_REMOVAL_OBJECTS'].lower():
                    model = "u2net_human_seg"
            if model is not None:
                args = [
                    "-i", image_path,
                    "-o", filter_output_dir,
                    "-nt", str(pipeline.config.num_threads),
                    "-ng", str(pipeline.config.num_gpus),
                    "-m", model
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
    except Exception as e:
        error_message = f"Issue creating human subject removal component: {e}"
        pipeline.report_error(740, error_message)

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
                # FEATURE EXTRACTOR COMPONENT
                args = [
                    "feature_extractor",
                    "--database_path", colmap_db_path,
                    "--image_path", image_path,
                    "--ImageReader.single_camera", "1"
                ]
                if ENABLE_MULTI_GPU == "true" or \
                    config['MODEL'] == "3dgut" or config['MODEL'] == "3dgrt":
                    if config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == "false":
                        args.extend([
                            "--ImageReader.camera_model", "PINHOLE"
                        ])

                if config['ENABLE_ENHANCED_FEATURE_EXTRACTION'] == "true":
                    args.extend([
                        "--SiftExtraction.estimate_affine_shape", "1"
                        #"--SiftExtraction.domain_size_pooling", "1"
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
                        "--SequentialMatching.vocab_tree_path", colmap_vocab_path
                    ])
                elif config['MATCHING_METHOD'] == "spatial":
                    args = [
                        "spatial_matcher",
                        "--database_path", colmap_db_path,
                        "--SpatialMatching.ignore_z", "0"
                    ]
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
                    args = [
                        #'pose_prior_mapper',
                        'point_triangulator',
                        '--database_path', colmap_db_path,
                        '--image_path', image_path,
                        '--input_path', sparse_model_path,
                        '--output_path', sparse_model_path,
                        '--refine_intrinsics', "1",
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
                    # MAPPER COMPONENT
                    if config['RECON_SOFTWARE_NAME'] == "colmap" :
                        args = [
                            "mapper",
                            "--database_path", colmap_db_path,
                            "--image_path", image_path,
                            "--output_path", sparse_path,
                            "--Mapper.multiple_models", "0"
                        ]
                        if config.get('PRESERVE_SCENE_SCALE', 'false') == 'true':
                            args.extend([
                                "--Mapper.ba_refine_focal_length", "0",
                                "--Mapper.ba_refine_principal_point", "0",
                                "--Mapper.ba_refine_extra_params", "0"
                            ])
                        if config['LOG_VERBOSITY'] == "error":
                            args.extend([
                                "--log_level", "1"
                            ])
                        if int(pipeline.config.num_gpus) > 0:
                            args.extend(["--Mapper.ba_use_gpu", "1"])
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
                if ENABLE_DEPTH_LOSS and ENABLE_MULTI_GPU == "false":
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
            # 3DGRUT requires undistorted images (PINHOLE only) — run undistorter on pre-existing COLMAP
            if colmap_zip_found and config['MODEL'] in ('3dgrt', '3dgut'):
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
                log.info("Added undistorter for 3DGRUT with pre-existing COLMAP reconstruction")
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
        elif colmap_zip_needs_conversion and config['MODEL'] not in ('3dgrt', '3dgut'):
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
        else:
            log.info("Not configured to perform SfM reconstruction...skipping dataset conversion.")
    except Exception as e:
        error_message = f"Issue creating the Colmap to Nerfstudio component: {e}"
        pipeline.report_error(755, error_message)

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
            if ENABLE_DEPTH_LOSS and ENABLE_MULTI_GPU == "false":
                if config['MODEL'] == "splatfacto-mcmc":
                    model = "mcmc"
                else:
                    model = "default"
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
                pipeline.create_component(
                    name="GSplat-Metrics",
                    comp_type=ComponentType.TRAINING,
                    comp_environ=ComponentEnvironment.EXECUTABLE,
                    command="/opt/ml/code/training/run_gsplat_trainer.sh",
                    args=eval_args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
            # Single GPU gsplat
            elif ENABLE_MULTI_GPU == "false" and \
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
                elif config['MODEL'] == "splatfacto" or config['MODEL'] == "splatfacto-big" or \
                    config['MODEL'] == "splatfacto-mcmc":
                    if config['RUN_RECON'] == "false" and not colmap_zip_found: # Resume training
                       # For splatfacto resume training, config.yml was already updated during extraction
                        dataset_config_path = os.path.join(config['DATASET_PATH'], "config.yml")
                        dataset_models_path = os.path.join(config['DATASET_PATH'], "nerfstudio_models")
                        has_config = os.path.exists(dataset_config_path)
                        has_ckpt = os.path.exists(dataset_models_path) and any(
                            f.endswith('.ckpt') for f in os.listdir(dataset_models_path)
                        ) if os.path.exists(dataset_models_path) else False
                        
                        if has_config and has_ckpt:
                            resume_training_active = True
                            # Update model paths to point to the resume training output directory
                            model_ckpt_path = os.path.join(config['CODE_PATH'], "outputs", "unnamed", model, RESUME_TRAIN_EXPERIMENT_NAME, model_dir_name)
                            model_config_path = os.path.join(config['CODE_PATH'], "outputs", "unnamed", model, RESUME_TRAIN_EXPERIMENT_NAME, "config.yml")
                            args.extend([
                                "--load-config", dataset_config_path,
                            ])
                            log.info(f"Resume training using config: {dataset_config_path}")
                            log.info(f"Resume training: max_num_iterations set to checkpoint_step + additional_steps in config.yml")
                    else:
                        # Initial training (either RUN_RECON=true or colmap_zip_found)
                        isp_mode = config.get('THREED_ISP', 'none').lower()
                        args.extend([
                        "--timestamp", TRAIN_EXPERIMENT_NAME,
                        "--pipeline.model.use-scale-regularization", "True",
                        "--max-num-iterations", str(int(config['MAX_STEPS']))
                    ])
                        if isp_mode == "bilagrid":
                            args.extend(["--pipeline.model.use-bilateral-grid", "True"])
                        elif isp_mode == "ppisp":
                            log.info("PPISP not currently supported with Splatfacto, using Bilateral-Grid instead")
                            args.extend(["--pipeline.model.use-bilateral-grid", "True"])
                            #args.extend(["--pipeline.model.use-ppisp", "True"])
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
                # When resuming via --load-config the dataparser subcommand is already
                # encoded in config.yml; re-specifying it causes ns-train to ignore the checkpoint.
                is_splatfacto_resume = (
                    config['RUN_RECON'] == "false" and
                    not colmap_zip_found and
                    config['MODEL'] in ("splatfacto", "splatfacto-big", "splatfacto-mcmc")
                )
                if not is_splatfacto_resume:
                    args.extend([
                        data_model,
                        "--data", config['DATASET_PATH'],
                        "--downscale-factor", "1",
                        "--auto-scale-poses", auto_scale_value
                    ])
                    if auto_scale_value == "True":
                         args.extend(["--center-method", "poses"]) #poses,focus,none
                    if ZIP_HAS_MASKS:
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
            elif ENABLE_MULTI_GPU == "true" and \
                config['MODEL'] != "3dgut" and config['MODEL'] != "3dgrt":
                #multi-gpu, use gsplat training strategy
                num_gpus = int(pipeline.config.num_gpus)
                batch_size = 1
                #steps_scaler = 1.0 / num_gpus  # Scale by number of GPUs only
                steps_scaler = 0.9576*(num_gpus*batch_size)**(-1.689)
                if config['MODEL'] == "splatfacto-mcmc":
                    model = "mcmc"
                else:
                    model = "default"
                isp_mode = config.get('THREED_ISP', 'none').lower()
                depth_loss_flag = "--depth-loss" if ENABLE_DEPTH_LOSS else "--no-depth-loss"
                args = [
                    model,
                    "--max_steps", str(int(config['MAX_STEPS'])),
                    "--result-dir", output_path,
                    "--data_factor", "1",
                    "--steps_scaler", str(steps_scaler),
                    "--disable_viewer",
                    #"--packed",  # TODO: upstream gsplat bug - --packed causes cudaErrorIllegalAddress
                    #              # with NCCL all_reduce in multi-GPU distributed training.
                    #              # Re-enable once fixed: https://github.com/nerfstudio-project/gsplat/issues/910
                    "--eval_steps", str(int(config['MAX_STEPS'])),
                    depth_loss_flag,
                    "--data-dir", config['DATASET_PATH']
                ]
                if ENABLE_DEPTH_LOSS:
                    args.extend(["--depth_lambda", "1e-3"])
                if config.get('PRESERVE_SCENE_SCALE', 'false').lower() == 'true':
                    args.extend(["--no-normalize-world-space", "--depth_lambda", "1e-4"])
                if isp_mode == "bilagrid" or isp_mode == "ppisp":
                    log.info(f"ISP mode '{isp_mode}' not supported with multi-GPU gsplat, skipping")
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
                pipeline.create_component(
                    name="GSplat-Metrics",
                    comp_type=ComponentType.TRAINING,
                    comp_environ=ComponentEnvironment.EXECUTABLE,
                    command="/opt/ml/code/training/run_gsplat_trainer.sh",
                    args=eval_args,
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
                        log.error(f"3DGRUT checkpoint not found: {threedgrut_ckpt_file}")
                        raise RuntimeError(f"3DGRUT checkpoint missing: {threedgrut_ckpt_file}")
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
        if config['MODEL'] != "3dgut" and config['MODEL'] != "3dgrt" and not ENABLE_DEPTH_LOSS:
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
                        config_path = f"outputs/unnamed/splatfacto/{RESUME_TRAIN_EXPERIMENT_NAME}/config.yml"
                        args = [
                            "gaussian-splat",
                            "--load-config", config_path,
                            "--output-dir", output_path
                        ]
                        log.info(f"Resume training export using config: {config_path}")
                    else:
                        # Initial training - use config from outputs directory
                        config_path = f"outputs/unnamed/splatfacto/{TRAIN_EXPERIMENT_NAME}/config.yml"
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
        if ENABLE_DEPTH_LOSS and config['MODEL'] != "3dgut" and config['MODEL'] != "3dgrt":
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
            if not ENABLE_DEPTH_LOSS and (config['MODEL'] == "splatfacto" or config['MODEL'] == "splatfacto-big" or \
                config['MODEL'] == "splatfacto-mcmc" or config['MODEL'] == "nerfacto" or config['MODEL'] == "splatfacto-w-light"):
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
    except Exception as e:
        error_message = f"Issue calculating metrics: {e}"
        pipeline.report_error(771, error_message)

    ##################################
    # POST-PROCESS COMPONENT:
    # Transform splat result to trajectory video
    ##################################
    try:
        if config['ENABLE_VIDEO_EXPORT'] == "true" and ENABLE_MULTI_GPU == "false":
            if not ENABLE_DEPTH_LOSS and (config['MODEL'] == "nerfacto" or config['MODEL'] == "splatfacto" or config['MODEL'] == "splatfacto-mcmc" or \
                config['MODEL'] == "splatfacto-big" or config['MODEL'] == "splatfacto-w-light"):
                model = "splatfacto"
                if config['MODEL'] == "splatfacto-w-light":
                    model = "splatfacto-w-light"
                if config['MODEL'] == "nerfacto":
                    model = "nerfacto"
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
                    "--frame-rate", "5"
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
                    "--output", ply_path,
                    "--level", "low",
                    "--min-cluster", "100",
                    "--no-confirm"
                ]
                pipeline.create_component(
                    name="Clean-Point-Cloud",
                    comp_type=ComponentType.POST_PROCESSING,
                    comp_environ=ComponentEnvironment.PYTHON,
                    command="post_processing/clean_point_cloud.py",
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
            if config['MODEL'] not in ("3dgut", "3dgrt"):
                preserve_scale = config.get('PRESERVE_SCENE_SCALE', 'false').lower() == 'true'
                if (ENABLE_MULTI_GPU == "true" or ENABLE_DEPTH_LOSS) and preserve_scale:
                    # gsplat with --no-normalize-world-space: raw COLMAP space
                    # coord transform alone is sufficient, no additional rotation needed
                    rotation = None
                else:
                    # nerfstudio OR normalized gsplat (align_principal_axes makes it OpenGL-like)
                    rotation = '270,0,0'
                if rotation:
                    args = [
                        orig_ply_path,
                        orig_ply_path,
                        f"--rotate={rotation}",
                        '-w'
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
            if config['MODEL'] != "3dgut" and config['MODEL'] != "3dgrt":
                preserve_scale = config.get('PRESERVE_SCENE_SCALE', 'false').lower() == 'true'
                if (ENABLE_MULTI_GPU == "true" or ENABLE_DEPTH_LOSS) and preserve_scale:
                    rotation = None  # raw COLMAP: coord transform alone sufficient
                else:
                    rotation = '270,0,0'  # nerfstudio or normalized gsplat
                if rotation:
                    args = [
                        sog_ply_path,
                        sog_ply_path,
                        f"--rotate={rotation}",
                        '-w'
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
            is_gsplat = ENABLE_MULTI_GPU == "true" or ENABLE_DEPTH_LOSS
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
            if config['MODEL'] not in ("3dgut", "3dgrt"):
                is_lhyu = config.get('SPZ_COORDS', 'rhyu') == 'lhyu'
                preserve_scale = config.get('PRESERVE_SCENE_SCALE', 'false').lower() == 'true'
                if (ENABLE_MULTI_GPU == "true" or ENABLE_DEPTH_LOSS) and preserve_scale:
                    # gsplat raw COLMAP space (--no-normalize-world-space)
                    rotation = '-90,0,180' if is_lhyu else '-90,0,0'
                elif ENABLE_MULTI_GPU == "true" or ENABLE_DEPTH_LOSS:
                    # gsplat normalized: spz.ply copied from orig.ply (rhyu space)
                    # Apply 180,0,0 to match the orientation Babylon.js SPZ expects
                    rotation = '180,0,0'
                else:
                    # nerfstudio OpenGL-space
                    rotation = '90,0,180' if is_lhyu else '270,0,0'
            else:
                rotation = '180,0,0'
            if rotation:
                args = [
                    spz_ply_path,
                    spz_ply_path,
                    f"--rotate={rotation}",
                    '-w'
                ]
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
        
        for i in range(0, pipeline.config.num_components, 1):
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
                            if image_path != input_file_path:
                                if os.path.exists(image_path):
                                    shutil.rmtree(image_path)
                                shutil.copytree(input_file_path, image_path)
                                log.info(f"Copied {len(os.listdir(image_path))} files from folder to images directory")
                            else:
                                log.info(f"Using {len(os.listdir(image_path))} files from folder (already in place)")
                            if config.get('AUTOSCALE_DATASET', 'false') != 'true':
                                validate_and_resize_images(image_path, config, log, pipeline)
                                resize_images_to_common_dimensions(image_path)
                        else: # Archive of images or archive with a video
                            # unzip archive into temp directory
                            temp_path = os.path.join(config['DATASET_PATH'], 'temp')
                            with zipfile.ZipFile(input_file_path, "r") as zip_ref:
                                # Validate entries to prevent zip slip attacks
                                for entry in zip_ref.namelist():
                                    entry_path = os.path.realpath(os.path.join(temp_path, entry))
                                    if not entry_path.startswith(os.path.realpath(temp_path) + os.sep):
                                        raise ValueError(f"Zip slip detected: {entry}")
                                zip_ref.extractall(temp_path)
                            
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
                                        temp_path, image_path, config['DATASET_PATH'], log
                                    )
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
                    # Rename the masked image directory to images
                    shutil.rmtree(image_path)
                    os.rename(mask_human_output_dir, image_path)
                    log.info("All images successfully processed with human subject remover")
                case "ColmapSfM-Feature-Extractor":
                    # COLMAP FEATURE EXTRACTOR CONDITIONAL COMPONENT
                    log.info("Using standard COLMAP feature extraction")
                    # If using pose prior, use the intrinsics from the txt file
                    if config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == "true":
                        camera_params = read_camera_params_from_file(os.path.join(sparse_model_path, "cameras.txt"))
                        component.args.extend([
                            "--ImageReader.camera_model", camera_params['model'],
                            "--ImageReader.camera_params", camera_params['params_str']
                        ])
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
                case "ColmapSfM-Image-Undistorter":
                    pipeline.run_component(i)
                    # For 3DGRUT with pre-existing COLMAP: move undistorted output into place
                    undist_dir = os.path.join(config['DATASET_PATH'], '_undistorted')
                    if config['MODEL'] in ('3dgrt', '3dgut') and os.path.isdir(undist_dir):
                        undist_sparse = os.path.join(undist_dir, 'sparse')
                        undist_images = os.path.join(undist_dir, 'images')
                        if os.path.isdir(undist_sparse):
                            # Replace sparse/0/ contents with undistorted model
                            for f in os.listdir(undist_sparse):
                                shutil.move(os.path.join(undist_sparse, f),
                                            os.path.join(sparse_model_path, f))
                            log.info(f"Moved undistorted model into {sparse_model_path}")
                        if os.path.isdir(undist_images):
                            # Replace images with undistorted versions
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
                    pipeline.run_component(i)
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
                    # Check image count and resolution at runtime to configure color correction
                    image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    num_images = len(image_files)
                    is_4k_or_higher = False
                    if image_files:
                        sample_img = Image.open(os.path.join(image_path, image_files[0]))
                        is_4k_or_higher = sample_img.width >= 3840 or sample_img.height >= 2160
                    
                    # Disable color correction metrics for large/high-res datasets
                    disable_color_correction = num_images > GPU_MAX_IMAGES or is_4k_or_higher
                    if disable_color_correction and config['MODEL'] in ['splatfacto', 'splatfacto-big', 'splatfacto-mcmc']:
                        log.info(f"Disabling color corrected metrics: num_images={num_images}, is_4k={is_4k_or_higher}")
                        # Find and replace color correction args
                        for j, arg in enumerate(component.args):
                            if arg == "--pipeline.model.color-corrected-metrics" and j+1 < len(component.args):
                                component.args[j+1] = "False"
                    # Set the image cache to disk if there are a lot of images to prevent OOM
                    if config['MODEL'] != "nerfacto" and config['MODEL'] != "3dgrt" and config['MODEL'] != "3dgut" and ENABLE_MULTI_GPU == "false":
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
                    if ENABLE_MULTI_GPU == "false":
                        if config['MODEL'] != "3dgut" and config['MODEL'] != "3dgrt" and not ENABLE_DEPTH_LOSS:
                            if config['RECON_SOFTWARE_NAME'] == "colmap" or config['RECON_SOFTWARE_NAME'] == "glomap" or \
                                config['RECON_SOFTWARE_NAME'] == "map_anything" or config['RECON_SOFTWARE_NAME'] == "hloc":
                                # Ensure colmap/sparse structure exists for NerfStudio
                                log.info('Running Training...')
                                sparse_path_out = os.path.join(config['DATASET_PATH'], "colmap", "sparse")
                                if os.path.exists(sparse_path) and os.listdir(sparse_path):
                                    # sparse/ has content - move it to colmap/sparse/
                                    if os.path.exists(sparse_path_out):
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
                        else: # 3dgrut
                            if has_alpha_channel(os.path.join(image_path, os.listdir(image_path)[0])):
                                process_images(image_path)
                    try:
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
                    if ENABLE_MULTI_GPU == "false" and config['MODEL'] not in ["3dgut", "3dgrt"]:
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
                    if (ENABLE_MULTI_GPU == "true" or ENABLE_DEPTH_LOSS) and config['ENABLE_VIDEO_EXPORT'] == "true":
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
                        if config['MODEL'] != "3dgut" and config['MODEL'] != "3dgrt":
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
                            if config['MODEL'] != "3dgut" and config['MODEL'] != "3dgrt":
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
                case "Nerfstudio-Metrics" | "3DGRUT-Metrics" | "GSplat-Metrics":
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
                    
                    pipeline.run_component(i)
                    
                    # Restore original CUDA setting
                    if component.name == "GSplat-Metrics":
                        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda
                    
                    # Log evaluation metrics after component runs
                    if os.path.exists(EVAL_METRIC_PATH):
                        try:
                            with open(EVAL_METRIC_PATH, 'r') as f:
                                metrics_data = json.load(f)
                            results = metrics_data.get('results', {})
                            log.info(f"Evaluation Metrics - PSNR: {results.get('psnr', 'N/A'):.4f}, SSIM: {results.get('ssim', 'N/A'):.4f}, LPIPS: {results.get('lpips', 'N/A'):.4f}")
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
                    if (ENABLE_MULTI_GPU == "true" or ENABLE_DEPTH_LOSS) and \
                            config.get('PRESERVE_SCENE_SCALE', 'false').lower() != 'true' and \
                            config['ENABLE_SPZ'] == "true" and os.path.exists(orig_ply_path):
                        shutil.copy2(orig_ply_path, spz_ply_path)
                        log.info(f"Copied rotated orig.ply to spz.ply for gsplat SPZ")
                case _: # Default case, run Component
                    pipeline.run_component(i)
                    # After autoscale runs, normalize image dimensions
                    if component.name == "AutoscaleDataset":
                        resize_images_to_common_dimensions(image_path)

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
                    'psnr': float(results.get('psnr', 0)),
                    'ssim': float(results.get('ssim', 0)),
                    'lpips': float(results.get('lpips', 0))
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
