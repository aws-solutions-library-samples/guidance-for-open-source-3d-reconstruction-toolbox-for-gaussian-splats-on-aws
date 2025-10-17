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
loader, filter, transform, renderer, or exporter based on the component function use.
The scripts for components are ordered by task type under the pipeline directory
such as image_processing, segmentation, post_processing
                    _________________________________________________________________________
                    |                           EXAMPLE PIPELINE                             |
                    |  __________________     __________________     __________________      |
                    |  |                 |    |                 |    |                 |     |
                    |  |   COMPONENT 1   |    |   COMPONENT 2   |    |   COMPONENT N   |     |   
(.mp4,.mov,.zip)o>-----|  (TRANSFORM):   |----|    (FILTER):    |----|  (COMP_TYPE):   |--//---->o[.ply,.spz,.sog,.mp4,.png]
                    |  | VIDEO-TO-IMAGES |    | FILTER-BLUR-IMG |    |  DO-SOMETHING   |     |
                    |  |     SCRIPT      |    |     SCRIPT      |    |     SCRIPT      |     |
                    |  |_________________|    |_________________|    |_________________|     |
                    |                                                                        |
                    |________________________________________________________________________|

ERROR CODES
700, "Required environment variables not set. Check that the payload has the required fields"
705, "Configuration not supported. Only pose prior transform json or pose prior colmap model files can be enabled, not both."
710, "Improper file type given for prior pose transformations. Only '.zip' is supported."
715, "Issue transforming pose to colmap component"
720, "Issue creating video to images component"
725, "Issue creating remove blurry images component"
730, "Issue creating background removal component"
735, "Issue creating spherical image component"
740, "Issue creating human subject removal component"
745, "SfM Software name given not implemented"
750, "Issue creating the SfM component"
755, "Issue creating the Colmap to Nerfstudio component"
760, "Trainer specified does not match proper configuration"
765, "Issue running the training session stage"
770, "Issue exporting splat from NerfStudio"
775, "Issue rendering trajectory video"
776, "Issue extracting video thumbnail"
777, "Issue converting images to video"
778, "Issue uploading thumbnail to S3"
780, "Issue cropping splat bounding box"
781, "Issue cleaning PLY file"
782, "Issue rotating splat"
783, "Issue converting ply to SOGS"
784, "Issue mirroring the splat"
785, "Issue creating compressed spz splat"
790, "Issue uploading asset to S3"
795, "General error running the pipeline"
"""

import re
import os
import sys
import ast
import cv2
import time
import json
import math
import boto3
import torch
import shutil
import zipfile
import subprocess
import torchvision
import multiprocessing
from pipeline import Pipeline, Status, ComponentEnvironment, ComponentType
from utils import (
    read_camera_params_from_file, validate_input_media,
    load_config, obj_to_glb, count_up_to, untar_gz, process_images,
    select_largest_colmap_model, create_tarball, has_alpha_channel,
    cleanup_dataset, cleanup_cuda_memory, validate_and_resize_images
)

if __name__ == "__main__":
    ##################################
    # INITIALIZATION
    ##################################
    try:
        # Print version information at startup
        print("=== CONTAINER VERSION INFORMATION ===")
        print(f"  Python: {sys.version.split()[0]}")
        
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0 and 'release' in result.stdout:
                cuda_version = result.stdout.split('release ')[1].split(',')[0]
                print(f"  CUDA: {cuda_version}")
            else:
                print("  CUDA: Not found")
        except:
            print("  CUDA: Not found")
        
        try:
            print(f"  PyTorch: {torch.__version__}")
        except:
            print("  PyTorch: Not found")
        
        try:
            print(f"  TorchVision: {torchvision.__version__}")
        except:
            print("  TorchVision: Not found")
        
        try:
            result = subprocess.run(['colmap', '-h'], capture_output=True, text=True)
            if result.returncode == 0 and 'COLMAP' in result.stdout:
                colmap_version = result.stdout.split('\n')[0].split()[1]
                print(f"  COLMAP: {colmap_version}")
            else:
                print("  COLMAP: Not found")
        except:
            print("  COLMAP: Not found")
        
        try:
            result = subprocess.run(['glomap', '-h'], capture_output=True, text=True)
            if result.returncode == 0 and 'GLOMAP' in result.stdout:
                glomap_version = result.stdout.split('\n')[0].split()[1]
                print(f"  Glomap: {glomap_version}")
            else:
                print("  Glomap: Not found")
        except:
            print("  Glomap: Not found")
        
        print("=== END VERSION INFORMATION ===")
        print()

        # Setup path constants
        OUTPUT_TAR_PATH = "/opt/ml/model/model.tar.gz"
        OUTPUT_DATASET_PATH = "/opt/ml/model/dataset"
        TRAIN_EXPERIMENT_NAME = "train-stage-1"
        RESUME_TRAIN_EXPERIMENT_NAME = "train-stage-2"
        IS_BATCH = 'AWS_BATCH_JOB_ID' in os.environ
        GPU_MAX_IMAGES = 500 # est at 4k
        REFINE_STEPS_SPLATFACTO = 30000
        REFINE_STEPS_3DGRUT = 6000

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
            
            # Download input data from S3
            s3_client = boto3.client('s3')
            
            # Parse S3 paths from environment variables
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
            
            if s3_model.startswith('s3://'):
                bucket, key = s3_model[5:].split('/', 1)
                local_path = '/tmp/input/model/models.tar.gz'
                s3_client.download_file(bucket, key, local_path)

        # Open config with default values
        with open("config.json", encoding="utf-8") as f:
            config = json.load(f)
        config_names = list(config.keys())
        config_values = list(config.values())
        config = load_config(config_names, config_values)

        # Sanity check on environment vars/constants
        if config['DATASET_PATH'] == "" or config['CODE_PATH'] == "" or \
            config['UUID'] == "" or config['S3_INPUT'] == "" or \
                config['S3_OUTPUT'] == "" or config['FILENAME'] == "":
            error_message = """Error Code 700: Required environment variables not set.
                Check that the payload has the required fields"""
            raise RuntimeError(error_message)
        
        # Unpack the sam2 models
        untar_gz(os.path.join(os.environ["MODEL_PATH"], "models.tar.gz"), os.environ["MODEL_PATH"])

        # Unpack all models from S3
        models_archive = os.path.join(os.environ["MODEL_PATH"], "models.tar.gz")
        untar_gz(models_archive, os.environ["MODEL_PATH"])
        
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

        # Instantiate Pipeline Session
        pipeline = Pipeline(
            name="3DGS-Pipeline",
            uuid=config['UUID'],
            num_threads=str(multiprocessing.cpu_count()),
            num_gpus=str(torch.cuda.device_count()),
            log_verbosity=config['LOG_VERBOSITY']
        )
        log = pipeline.session.log
        pipeline.session.status = Status.INIT
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
    
    # Log system information for debugging
    log.info(f"System Information:")
    log.info(f"  Python: {sys.version.split()[0]}")
    log.info(f"  PyTorch: {torch.__version__}")
    log.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"  GPU: {torch.cuda.get_device_name()}")
        log.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    log.info(f"  Model: {config['MODEL']}")
    log.info(f"  Resume training: {config['RUN_SFM'] == 'false'}")

    # Ensure we have an /images directory in dataset path for Colmap/Glomap
    image_path = os.path.join(config['DATASET_PATH'], "images")
    if not os.path.isdir(image_path):
        log.info(f"Creating '/images' directory in {config['DATASET_PATH']}")
        os.makedirs(image_path, exist_ok=True)

    # Ensure we have a /sparse directory in dataset path for NerfStudio
    sparse_path = os.path.join(config['DATASET_PATH'], "sparse")
    sparse_model_path = os.path.join(sparse_path, "0")
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
    spz_path = os.path.join(output_path, "splat.spz")

    # For spherical, will have 6 views per 360 image using cube faces so will be 6x images
    config['MAX_NUM_IMAGES'] = str(int(config['MAX_NUM_IMAGES']))

    input_filename_extension = os.path.splitext(config['FILENAME'])[1]
    input_file_path = os.path.join(config['DATASET_PATH'], config['FILENAME'])
    current_dir_path = os.path.dirname(os.path.realpath(__file__))

    # Prevent DataLoader shared memory issues with large datasets
    #os.environ['PYTORCH_DATALOADER_NUM_WORKERS'] = '0'
    # Fix PyTorch 2.6+ weights_only default for checkpoint loading
    os.environ['PYTHONPATH'] = f"{current_dir_path}:{os.environ.get('PYTHONPATH', '')}"
    # CUDA debugging and memory management
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable synchronous CUDA for better error reporting
    #os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
    # Additional shared memory configuration for batch jobs
    #if IS_BATCH:
    #    os.environ['OMP_NUM_THREADS'] = '1'

    # Store the full list of GPUs
    if int(pipeline.config.num_gpus)>0:
        os.environ['CUDA_VISIBLE_DEVICES'] = count_up_to(int(pipeline.config.num_gpus))
        USE_GPU = "true"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        USE_GPU = "false"
    # Check if video or zip of images given
    VIDEO = validate_input_media(config['FILENAME'])
    log.info(f"Is Video?: {VIDEO}")
    config['ENABLE_MULTI_GPU']="false"

    ##################################
    # DETECT AND EXTRACT MODEL.TAR.GZ
    ##################################
    # Check if input is a model.tar.gz file for resuming training
    if config['FILENAME'].endswith('model.tar.gz') or config['FILENAME'].endswith('.tar.gz'):
        log.info(f"Detected model archive: {config['FILENAME']} for resuming training")
        model_tar_path = os.path.join(config['DATASET_PATH'], config['FILENAME'])
        if os.path.exists(model_tar_path):
            #ply_path = os.path.join(config['CODE_PATH'], "resume_exports", "splat.ply")
            log.info(f"Extracting {model_tar_path} to {config['CODE_PATH']}")
            untar_gz(model_tar_path, config['CODE_PATH'])
            print("Extracted resume files: ")
            print(", ".join(os.listdir(config['CODE_PATH'])))
            
            # Handle dataset extraction - ensure complete dataset is moved
            dataset_dir = os.path.join(config['CODE_PATH'], 'dataset')
            if os.path.exists(dataset_dir):
                print("Dataset directory contents: ")
                print(", ".join(os.listdir(dataset_dir)))
                
                # Clear existing dataset path and move entire dataset
                if os.path.exists(config['DATASET_PATH']):
                    shutil.rmtree(config['DATASET_PATH'])
                shutil.move(dataset_dir, config['DATASET_PATH'])
                log.info(f"Moved entire dataset from {dataset_dir} to {config['DATASET_PATH']}")
            
            # Move model directory and config.yml to proper output directory structure for resume training
            model_dir_name = "nerfstudio_models"
            if config['MODEL'] == "3dgrt" or config['MODEL'] == "3dgut":
                model_dir_name = "3dgrut_models"
            model_src_dir = os.path.join(config['DATASET_PATH'], model_dir_name)
            config_yml_src = os.path.join(config['DATASET_PATH'], 'config.yml')
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
                        
                        # Update max_num_iterations using text replacement
                        config_content = re.sub(r'max_num_iterations: \d+', f'max_num_iterations: {REFINE_STEPS_SPLATFACTO}', config_content)
                        
                        # Update timestamp using text replacement
                        config_content = re.sub(r'timestamp: [^\n]+', f'timestamp: {RESUME_TRAIN_EXPERIMENT_NAME}', config_content)
                        
                        # Update dataset path using text replacement
                        config_content = re.sub(r'data: !!python/object/apply:pathlib\.PosixPath\s*\n\s*-[^\n]*(?:\n\s*-[^\n]*)*', 
                                              f'data: !!python/object/apply:pathlib.PosixPath\n      - {config["DATASET_PATH"]}', 
                                              config_content, flags=re.MULTILINE)
                        
                        with open(config_yml_src, 'w') as f:
                            f.write(config_content)
                        
                        log.info(f"""
                                Updated config.yml in dataset directory:
                                max_num_iterations={REFINE_STEPS_SPLATFACTO},
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
            log.info(", ".join(os.listdir(config['DATASET_PATH'])))
            
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
            for file in os.listdir(config['DATASET_PATH']):
                if file.lower().endswith(media_extensions):
                    config['FILENAME'] = file
                    log.info(f"Found original media file: {file}")
                    break

            # Ensure we remove previous exports
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
        config['ENABLE_MULTI_GPU'] = "true"
        #os.environ['NCCL_DEBUG'] = 'INFO'
        #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        # Read SageMaker resource config for multi-container setup
        resource_config_path = '/opt/ml/input/config/resourceconfig.json'
        if os.path.exists(resource_config_path):
            with open(resource_config_path, 'r') as f:
                resource_config = json.load(f)
            
            hosts = resource_config.get('hosts', ['local-host'])
            current_host = resource_config.get('current_host', 'localhost')
            network_interface = resource_config.get('network_interface_name', 'eth0')
            
            log.info(f"DEBUG: Resource config - hosts: {hosts}, current: {current_host}, interface: {network_interface}")
            
            # Set distributed training environment variables
            os.environ['MASTER_ADDR'] = hosts[0]  # First host is master
            os.environ['MASTER_PORT'] = '29500'  # Use standard PyTorch distributed port
            os.environ['WORLD_SIZE'] = str(len(hosts))
            os.environ['RANK'] = str(hosts.index(current_host))
            os.environ['LOCAL_RANK'] = '0'  # Single GPU per container
            
            log.info(f"""DEBUG: Multi-container setup -
                     MASTER_ADDR={os.environ['MASTER_ADDR']},
                     MASTER_PORT={os.environ['MASTER_PORT']},
                     WORLD_SIZE={os.environ['WORLD_SIZE']},
                     RANK={os.environ['RANK']}
                     """)
        else:
            # Single instance multi-GPU setup
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'  # Use standard PyTorch distributed port
            os.environ['WORLD_SIZE'] = '1'
            os.environ['RANK'] = '0'
            os.environ['LOCAL_RANK'] = '0'
            
            log.info(f"""DEBUG: Single instance multi-GPU -
                     MASTER_ADDR={os.environ['MASTER_ADDR']},
                     MASTER_PORT={os.environ['MASTER_PORT']},
                     WORLD_SIZE={os.environ['WORLD_SIZE']}
                     """)
    else:
        log.info("DEBUG: Single GPU setup, no distributed training configuration needed")

    ##################################
    # TRANSFORM COMPONENT:
    # Pose Transform for SfM
    #################################
    try:
        if config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true' and \
            config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true' and \
            config['RUN_SFM'] == 'true':
            raise RuntimeError(
                pipeline.report_error(
                    705,
                    f"""Configuration not supported.
                    Only pose prior transform json or pose prior colmap model files can be enabled, not both."""
                )
            )
        if (config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true' or \
            config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true') and config['RUN_SFM'] == 'true':
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
                    comp_type=ComponentType.transform,
                    comp_environ=ComponentEnvironment.python,
                    command="sfm/extract_poses_imgs.py",
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
    # TRANSFORM COMPONENT:
    # Video to Images
    ##################################
    try:
        if VIDEO is True and config['REMOVE_BACKGROUND'] == "true" and \
                config['BACKGROUND_REMOVAL_MODEL'] == "sam2" and config['RUN_SFM'] == 'true':
                # SAM2 BACKGROUND REMOVAL COMPONENT
                args = [
                    "-i", input_file_path,
                    "-o", image_path,
                    "-n", config['MAX_NUM_IMAGES'],
                    "-mt", config['MASK_THRESHOLD']
                ]
                pipeline.create_component(
                    name="RemoveBackground",
                    comp_type=ComponentType.filter,
                    comp_environ=ComponentEnvironment.python,
                    command="sam/remove_background_sam2.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
        elif VIDEO is False and config['BACKGROUND_REMOVAL_MODEL'] == "sam2" and \
            config['REMOVE_BACKGROUND']=="true" and config['RUN_SFM'] == 'true':
            sys.exit("Error: SAM2 Background removal is only supported for video input")
        else: # Just extract the frames, remove background later
            args = [
                "-i", input_file_path,
                "-o", image_path,
                "-n", config['MAX_NUM_IMAGES'],
                "-nw", pipeline.config.num_threads,
                "-ll", config['LOG_VERBOSITY'].upper(),
                "-st", config['VIDEO_START_TIME']
            ]
            # Debug logging for VIDEO_STOP_TIME
            log.info(f"DEBUG: VIDEO_STOP_TIME value: '{config['VIDEO_STOP_TIME']}' (type: {type(config['VIDEO_STOP_TIME'])})")
            
            # Only add end time if it's not None/none/empty
            video_stop_time = str(config['VIDEO_STOP_TIME']).strip() if config['VIDEO_STOP_TIME'] is not None else ""
            log.info(f"DEBUG: Processed video_stop_time: '{video_stop_time}'")
            
            if video_stop_time and video_stop_time.lower() not in ['none', 'null', '', 'nan']:
                log.info(f"DEBUG: Adding -et parameter with value: '{video_stop_time}'")
                args.extend(["-et", video_stop_time])
            else:
                log.info(f"DEBUG: Skipping -et parameter - VIDEO_STOP_TIME is: '{video_stop_time}'")
            
            log.info(f"DEBUG: Final video processing args: {args}")
            pipeline.create_component(
                name="VideoToImages",
                comp_type=ComponentType.transform,
                comp_environ=ComponentEnvironment.python,
                command="video_processing/simple_video_to_images.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue creating video to images component: {e}"
        pipeline.report_error(720, error_message)

    ##################################
    # FILTER COMPONENT:
    # Remove Blurry Images
    ##################################
    try:
        # REMOVE BLURRY IMAGES COMPONENT
        # Skip blur filtering when using pose priors to maintain correspondence
        if config['FILTER_BLURRY_IMAGES'] == "true" and \
           config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'false' and \
           config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'false' and \
           config['RUN_SFM'] == 'true':
            # For zip archives, count images and use a percentage-based approach
            if VIDEO is False and input_filename_extension.lower() == ".zip":
                # Count the number of images in the directory
                image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
                image_files = [f for f in os.listdir(image_path) 
                              if os.path.isfile(os.path.join(image_path, f)) 
                              and any(f.lower().endswith(ext) for ext in image_extensions)]
                total_images = len(image_files)
                
                # Initialize num_to_keep with a default value
                num_to_keep = 300  # Default value
                
                # If we have images, set num_frames_target to 90% of total (adjust as needed)
                if total_images > 0:
                    num_to_keep = max(1, int(total_images * 0.9))
                    log.info(f"Filtering blurry images from zip archive: keeping {num_to_keep} out of {total_images} images")
                else:
                    log.warning(f"No images found in {image_path}, using default target of {num_to_keep}")

                args = [
                    "-I", image_path,
                    "-r", "30",
                    "-n", str(num_to_keep),
                    "-O", image_path
                ]
            else:
                # For videos, use the MAX_NUM_IMAGES parameter as before
                args = [
                    "-I", image_path,
                    "-r", "30",
                    "-n", str(config['MAX_NUM_IMAGES']),
                    "-O", image_path,
                    "--log-level", config['LOG_VERBOSITY'].upper()
                ]

            if config['LOG_VERBOSITY'] == "debug":
                args.extend(["-v"])
                
            pipeline.create_component(
                name="RemoveBlurryImages",
                comp_type=ComponentType.transform,
                comp_environ=ComponentEnvironment.python,
                command="image_processing/filter_blurry_images.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
        elif config['FILTER_BLURRY_IMAGES'] == "true":
            log.info("Skipping blur filtering because pose priors are enabled - must maintain image correspondence or resuming training")
    except Exception as e:
        error_message = f"Issue creating remove blurry images component: {e}"
        pipeline.report_error(725, error_message)

    ##################################
    # FILTER COMPONENT:
    # Remove Background
    ##################################
    try:
        if config['REMOVE_BACKGROUND'] == "true" and config['BACKGROUND_REMOVAL_MODEL'] != "sam2" and \
            config['RUN_SFM'] == "true":
            model = "u2net"

            args = [
                "-i", image_path,
                "-o", image_path,
                "-nt", pipeline.config.num_threads,
                "-ng", pipeline.config.num_gpus,
                "-m", model
            ]

            pipeline.create_component(
                name="RemoveBackground",
                comp_type=ComponentType.filter,
                comp_environ=ComponentEnvironment.python,
                command="segmentation/remove_background.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=True
            )
    except Exception as e:
        error_message = f"Issue creating background removal component: {e}"
        pipeline.report_error(730, error_message)

    ##################################
    # TRANSFORM COMPONENT:
    # Spherical Image Processing
    ##################################
    try:
        if config['SPHERICAL_CAMERA'] == "true" and config['RUN_SFM'] == "true":
            if config['MATCHING_METHOD'] == "vocab":
                method = "vocabtree"
            else:
                method = config['MATCHING_METHOD']
            args = [
                "--input_image_path", image_path,
                "--output_path", config['DATASET_PATH'],
                "--matcher", method #"sequential", "exhaustive", "vocabtree", "spatial"
            ]
            
            # Add cube face removal parameter
            faces_to_remove = config['SPHERICAL_CUBE_FACES_TO_REMOVE'].strip()
            if faces_to_remove and faces_to_remove != '[]':
                args.append("--remove_faces")
            
            if config['REMOVE_OBJECT'] == "true":
                # Determine model for human detection
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
                             "-nt", pipeline.config.num_threads,
                             "-ng", pipeline.config.num_gpus,
                             "-gpu", USE_GPU
                             ])
            
            pipeline.create_component(
                name="PanoramaSfM",
                comp_type=ComponentType.transform,
                comp_environ=ComponentEnvironment.python,
                command="spherical/panorama_sfm.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue creating spherical image component: {e}"
        pipeline.report_error(735, error_message)

    ##################################
    # FILTER COMPONENT:
    # Remove Objects
    ##################################
    try:
        if config['REMOVE_OBJECT'] == "true" and config['RUN_SFM'] == "true":
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
                    "-nt", pipeline.config.num_threads,
                    "-ng", pipeline.config.num_gpus,
                    "-m", model
                ]
                pipeline.create_component(
                    name="RemoveObject",
                    comp_type=ComponentType.filter,
                    comp_environ=ComponentEnvironment.python,
                    command="segmentation/remove_background.py",
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
                        comp_type=ComponentType.transform,
                        comp_environ=ComponentEnvironment.python,
                        command="segmentation/remove_object_using_mask.py",
                        args=args,
                        cwd=current_dir_path,
                        requires_gpu=False
                    )
                else: # eraser
                    args = [
                        "-id", image_path,
                        "-md", filter_output_dir,
                        "-mp", os.path.join(config['DATASET_PATH'], "stable-diffusion-xl-base-1.0"),
                        "-pp", "/opt/ml/AttentiveEraser/pipelines/pipeline_stable_diffusion_xl_attentive_eraser.py",
                        "-gpu", USE_GPU,
                        "-log", config['LOG_VERBOSITY'],
                        "-method", "SIP" #DIP
                    ]
                    pipeline.create_component(
                        name="EraseObject",
                        comp_type=ComponentType.transform,
                        comp_environ=ComponentEnvironment.python,
                        command="segmentation/erase_object_using_mask.py",
                        args=args,
                        cwd=current_dir_path,
                        requires_gpu=True
                    )
    except Exception as e:
        error_message = f"Issue creating human subject removal component: {e}"
        pipeline.report_error(740, error_message)

    ##################################
    # TRANSFORM COMPONENT:
    # Images to Point Cloud
    ##################################
    try:
        if config['RUN_SFM'] == "true":
            if config['SPHERICAL_CAMERA'] == "true":
                log.info("Using spherical camera processing with panorama_sfm.py")
            elif config['SFM_SOFTWARE_NAME'] == "colmap" or config['SFM_SOFTWARE_NAME'] == "glomap":
                # FEATURE EXTRACTOR COMPONENT
                args = [
                    "feature_extractor",
                    "--database_path", colmap_db_path,
                    "--image_path", image_path,
                    "--ImageReader.single_camera", "1",
                    "--SiftExtraction.num_threads", pipeline.config.num_threads#,
                ]
                if config['ENABLE_MULTI_GPU'] == "true" or \
                    config['MODEL'] == "3dgut" or config['MODEL'] == "3dgrt":
                    if config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == "false":
                        args.extend([
                            "--ImageReader.camera_model", "PINHOLE"
                        ])

                if config['ENABLE_ENHANCED_FEATURE_EXTRACTION'] == "true":
                    args.extend([
                        "--SiftExtraction.estimate_affine_shape", "1",
                        "--SiftExtraction.domain_size_pooling", "1"
                    ])

                if config['LOG_VERBOSITY'] == "error":
                    args.extend([
                        "--log_level", "1"
                    ])
                pipeline.create_component(
                    name="ColmapSfM-Feature-Extractor",
                    comp_type=ComponentType.filter,
                    comp_environ=ComponentEnvironment.executable,
                    command="colmap",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )

                # Account for image name ordering and colmap database ordering when using pose priors
                # Perform the pose coordinate conversions or modify existing colmap model text files
                if config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true' or \
                    config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true':
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
                        comp_type=ComponentType.transform,
                        comp_environ=ComponentEnvironment.python,
                        command="sfm/process_pose_transforms.py",
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
                        "--SiftMatching.num_threads", pipeline.config.num_threads,
                        "--SequentialMatching.quadratic_overlap", "1",
                        "--SiftMatching.guided_matching", "0"
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
                        "--SpatialMatching.ignore_z", "0",
                        "--SiftMatching.num_threads", pipeline.config.num_threads#,
                    ]
                elif config['MATCHING_METHOD'] == "vocab":
                    args = [
                        "vocab_tree_matcher",
                        "--database_path", colmap_db_path,
                        "--SiftMatching.guided_matching", "1",
                        "--VocabTreeMatching.num_images", str(math.ceil(float(config['MAX_NUM_IMAGES'])/3)),
                        "--VocabTreeMatching.vocab_tree_path", colmap_vocab_path,
                        "--SiftMatching.num_threads", pipeline.config.num_threads#,
                    ]
                # Otherwise run the exhaustive matcher which usually takes longer
                else:
                    args = [
                        "exhaustive_matcher",
                        "--database_path", colmap_db_path,
                        "--SiftMatching.guided_matching", "1",
                        "--ExhaustiveMatching.block_size", config['MAX_NUM_IMAGES'],
                        "--SiftMatching.num_threads", pipeline.config.num_threads
                    ]
                if config['LOG_VERBOSITY'] == "error":
                    args.extend([
                        "--log_level", "1"
                    ])
                pipeline.create_component(
                    name="ColmapSfM-Feature-Matcher",
                    comp_type=ComponentType.filter,
                    comp_environ=ComponentEnvironment.executable,
                    command="colmap",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )

                if config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == "true" or \
                    config['USE_POSE_PRIOR_TRANSFORM_JSON'] == "true":
                    # TRIANGULATION COMPONENT
                    args = [
                        'point_triangulator',
                        '--database_path', colmap_db_path,
                        '--image_path', image_path,
                        '--input_path', sparse_model_path,
                        '--output_path', sparse_model_path,
                        '--refine_intrinsics', "1",
                        '--Mapper.multiple_models', "0",
                        '--Mapper.num_threads', pipeline.config.num_threads
                    ]
                    if config['LOG_VERBOSITY'] == "error":
                        args.extend([
                            "--log_level", "1"
                        ])
                    pipeline.create_component(
                        name="ColmapSfM-Triangulator",
                        comp_type=ComponentType.transform,
                        comp_environ=ComponentEnvironment.executable,
                        command="colmap",
                        args=args,
                        cwd=current_dir_path,
                        requires_gpu=False
                    )
                else:
                    # MAPPER COMPONENT
                    if config['SFM_SOFTWARE_NAME'] == "colmap" :
                        args = [
                            "mapper",
                            "--database_path", colmap_db_path,
                            "--image_path", image_path,
                            "--output_path", sparse_path,
                            "--Mapper.multiple_models", "0",
                            "--Mapper.num_threads", pipeline.config.num_threads
                        ]
                        if config['LOG_VERBOSITY'] == "error":
                            args.extend([
                                "--log_level", "1"
                            ])
                        pipeline.create_component(
                            name="ColmapSfM-Mapper",
                            comp_type=ComponentType.transform,
                            comp_environ=ComponentEnvironment.executable,
                            command="colmap",
                            cwd=current_dir_path,
                            args=args,
                            requires_gpu=False
                        )
                    else: # Glomap
                        args = [
                            "mapper",
                            "--database_path", colmap_db_path,
                            "--image_path", image_path,
                            "--output_path", sparse_path
                        ]

                        pipeline.create_component(
                            name="GlomapSfM-Mapper",
                            comp_type=ComponentType.transform,
                            comp_environ=ComponentEnvironment.executable,
                            command="glomap",
                            cwd=current_dir_path,
                            args=args,
                            requires_gpu=False
                        )
                # IMAGE UNDISTORTER
                # Run undistorter for multi-GPU or when using 3DGRUT with pose priors (to convert SIMPLE_RADIAL to PINHOLE)
                if config['ENABLE_MULTI_GPU'] == "true" or \
                   (config['MODEL'] == "3dgut" or config['MODEL'] == "3dgrt") and \
                    (config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true' or \
                    config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true'):
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
                        comp_type=ComponentType.transform,
                        comp_environ=ComponentEnvironment.executable,
                        command="colmap",
                        args=args,
                        cwd=current_dir_path,
                        requires_gpu=False
                    )
                    
                    # Update cameras.txt to PINHOLE model after undistortion
                    if (config['MODEL'] == "3dgut" or config['MODEL'] == "3dgrt") and \
                       (config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true' or \
                        config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true'):
                        args = [
                            "-s", sparse_model_path
                        ]
                        pipeline.create_component(
                            name="UpdateCameraModel",
                            comp_type=ComponentType.transform,
                            comp_environ=ComponentEnvironment.python,
                            command="sfm/update_camera_model.py",
                            args=args,
                            cwd=current_dir_path,
                            requires_gpu=False
                        )
            elif config['SFM_SOFTWARE_NAME'] == "vggt":
                args = [
                    "--input_dir", config['DATASET_PATH']
                ]
                pipeline.create_component(
                    name="Vggt-Ba",
                    comp_type=ComponentType.transform,
                    comp_environ=ComponentEnvironment.python,
                    command="sfm/run_vggt.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
            else:
                raise RuntimeError(
                    pipeline.report_error(
                        745, f"SfM Software not implemented yet:{config['SFM_SOFTWARE_NAME']}"
                    )
                )
        else:
            log.info("SfM configured to be skipped...skipping SfM")
    except Exception as e:
        error_message = f"Issue creating the SfM component: {e}"
        pipeline.report_error(750, error_message)

    ##################################
    # TRANSFORM COMPONENT:
    # Point Cloud, Images, and Poses to NerfStudio format
    ##################################
    try:
        if config['RUN_TRAIN'] == "true" and config['RUN_SFM'] == "true":
            if config['SFM_SOFTWARE_NAME'] == "colmap" or config['SFM_SOFTWARE_NAME'] == "glomap" or \
                config['SFM_SOFTWARE_NAME'] == "vggt":
                args = ["--data_dir", config['DATASET_PATH']]
                pipeline.create_component(
                    name="Colmap-to-Nerfstudio",
                    comp_type=ComponentType.transform,
                    comp_environ=ComponentEnvironment.python,
                    command="training/colmap_to_nerfstudio_cam.py",
                    cwd=current_dir_path,
                    args=args,
                    requires_gpu=False
                )
            else:
                raise RuntimeError(
                    pipeline.report_error(
                        750,
                        f"SfM Software name given not implemented:{config['SFM_SOFTWARE_NAME']}"
                    )
                )
        else:
            log.info("Not configured to output a Gaussian Splat...skipping dataset conversion.")
    except Exception as e:
        error_message = f"Issue creating the Colmap to Nerfstudio component: {e}"
        pipeline.report_error(755, error_message)

    ##################################
    # TRANSFORM COMPONENT:
    # Point Cloud, Images, and Poses to 3D Gaussian Splat
    ##################################
    try:
        if config['RUN_TRAIN'] == "true":
            if config['SFM_SOFTWARE_NAME'] == "glomap" or config['SFM_SOFTWARE_NAME'] == "colmap" or \
                config['SFM_SOFTWARE_NAME'] == "vggt":
                data_model = "colmap"
            # Single GPU gsplat
            if config['ENABLE_MULTI_GPU'] == "false" and \
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
                    if config['RUN_SFM'] == "false": # Resume training
                        # Check if we have extracted models in dataset path
                        # Files should already be in correct location from extraction phase
                        # Just validate they exist and add load arguments
                        # For splatfacto resume training, use dataset paths
                        dataset_models_path = os.path.join(config['DATASET_PATH'], "nerfstudio_models")
                        dataset_config_path = os.path.join(config['DATASET_PATH'], "config.yml")
                        
                        if os.path.exists(dataset_models_path) and os.path.exists(dataset_config_path):
                            args.extend([
                                "--timestamp", RESUME_TRAIN_EXPERIMENT_NAME,
                                "--load-dir", dataset_models_path,
                                "--load-config", dataset_config_path,
                                "--load-scheduler", "False",
                                "--max-num-iterations", str(REFINE_STEPS_SPLATFACTO)
                            ])
                            log.info(f"Resume training using checkpoints at: {dataset_models_path}")
                            log.info(f"Resume training using config at: {dataset_config_path}")
                            log.info(f"Resume training with {REFINE_STEPS_SPLATFACTO} iterations")
                        else:
                            log.error(f"Checkpoint files not found at: {dataset_models_path} (exists: {os.path.exists(dataset_models_path)})")
                            log.error(f"Config file not found at: {dataset_config_path} (exists: {os.path.exists(dataset_config_path)})")
                            if os.path.exists(dataset_models_path):
                                log.error(f"Contents of checkpoint dir: {os.listdir(dataset_models_path)}")
                            raise RuntimeError(f"Cannot resume training - checkpoint files missing")
                    else:
                        args.extend([
                        "--timestamp", TRAIN_EXPERIMENT_NAME,
                        "--pipeline.model.use-scale-regularization", "True",
                        "--max-num-iterations", str(int(int(config['MAX_STEPS'])))
                    ])
                elif config['MODEL'] == "splatfacto-w-light":
                    if config['RUN_SFM'] == "false": # Resume training
                        if os.path.exists(model_ckpt_path):
                            args.extend([
                                "--load-dir", model_ckpt_path,
                                "--load-scheduler", "False"
                            ])
                        args.extend([
                            "--timestamp", RESUME_TRAIN_EXPERIMENT_NAME,
                            "--pipeline.model.continue-cull-post-densification", "False",
                            "--pipeline.model.cull-alpha-thresh", "0.005",
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
                    raise RuntimeError(pipeline.report_error(765, "Trainer specified does not match proper configuration"))

                args.extend([
                    data_model,
                    "--data", config['DATASET_PATH'],
                    "--downscale-factor", "1",
                ])

                pipeline.create_component(
                    name="Train",
                    comp_type=ComponentType.transform,
                    comp_environ=ComponentEnvironment.executable,
                    command="ns-train",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
            # Multi-gpu gsplat
            elif config['ENABLE_MULTI_GPU'] == "true" and \
                config['MODEL'] != "3dgut" and config['MODEL'] != "3dgrt":
                #multi-gpu, use gsplat training strategy
                batch_size = 1  # Keep batch size small for memory efficiency
                step_scaler = float(1/(int(pipeline.config.num_gpus)*batch_size))
                if config['MODEL'] == "splatfacto-mcmc":
                    model = "mcmc"
                else:
                    model = "default"
                args = [
                    model,
                    "--max_steps", str(int(int(config['MAX_STEPS']))),
                    "--result-dir", output_path,
                    "--data_factor", "1",
                    "--steps_scaler", str(step_scaler),
                    "--disable_viewer",
                    "--packed",
                    "--batch-size", str(batch_size),
                    "--data-dir", config['DATASET_PATH']
                ]
                pipeline.create_component(
                    name="Train",
                    comp_type=ComponentType.transform,
                    comp_environ=ComponentEnvironment.python,
                    command="gsplat/examples/simple_trainer.py",
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
                if config['RUN_SFM'] == "false": # 3dgrut resume training
                    # Validate checkpoint exists and is readable
                    if os.path.exists(model_ckpt_path):
                        args.extend([
                            f"experiment_name={RESUME_TRAIN_EXPERIMENT_NAME}",
                            f"resume={model_ckpt_path}",
                            f"n_iterations={REFINE_STEPS_3DGRUT}",
                            f"scheduler.positions.max_steps={REFINE_STEPS_3DGRUT}",
                        ])
                    else:
                        log.error(f"3DGRUT checkpoint not found: {model_ckpt_path}")
                        raise RuntimeError(f"3DGRUT checkpoint missing: {model_ckpt_path}")
                else:
                    args.extend([
                        f"experiment_name={TRAIN_EXPERIMENT_NAME}",
                        f"n_iterations={str(config['MAX_STEPS'])}",
                        f"scheduler.positions.max_steps={str(config['MAX_STEPS'])}",
                    ])
                pipeline.create_component(
                    name="Train",
                    comp_type=ComponentType.transform,
                    comp_environ=ComponentEnvironment.python,
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
    # EXPORT COMPONENT:
    # Export .ply from splat training
    ##################################
    try:
        if config['RUN_TRAIN'] == "true" and config['MODEL'] != "3dgut" and config['MODEL'] != "3dgrt":
            if config['ENABLE_MULTI_GPU'] == "true":
                ckpt_dir = os.path.join(output_path, "ckpts")
                args = [
                    ckpt_dir,
                    ply_path
                ]
                pipeline.create_component(
                    name="Nerfstudio-Export",
                    comp_type=ComponentType.exporter,
                    comp_environ=ComponentEnvironment.python,
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
                        comp_type=ComponentType.exporter,
                        comp_environ=ComponentEnvironment.executable,
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
                        comp_type=ComponentType.exporter,
                        comp_environ=ComponentEnvironment.python,
                        command="nerfstudio/nerfstudio/scripts/texture.py",
                        args=args,
                        cwd=current_dir_path,
                        requires_gpu=True
                    )
                elif config['MODEL'] == "splatfacto-w-light": 
                    args = [
                        "--load_config", model_config_path,
                        "--output_dir", output_path, #export_output_path,
                        "--camera_idx", "0" #str(math.ceil(float(config['MAX_NUM_IMAGES'])/2))
                    ]
                    pipeline.create_component(
                        name="Nerfstudio-Export",
                        comp_type=ComponentType.exporter,
                        comp_environ=ComponentEnvironment.python,
                        command="splatfacto-w/export_script.py",
                        args=args,
                        cwd=current_dir_path,
                        requires_gpu=True
                    )
                else:
                    # Use correct output path for resume training
                    if config['RUN_SFM'] == "false":
                    #    export_output_path = os.path.join(config['CODE_PATH'], "resume_exports")
                        train_stage = RESUME_TRAIN_EXPERIMENT_NAME
                    else:
                    #    export_output_path = output_path
                        train_stage = TRAIN_EXPERIMENT_NAME
                    args = [
                        "gaussian-splat",
                        "--load-config", f"outputs/unnamed/splatfacto/{train_stage}/config.yml",
                        "--output-dir", output_path #export_output_path
                    ]
                    pipeline.create_component(
                        name="Nerfstudio-Export",
                        comp_type=ComponentType.exporter,
                        comp_environ=ComponentEnvironment.executable,
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
    # EXPORT COMPONENT:
    # Export trajectory video of splat result
    ##################################
    try:
        if config['ENABLE_VIDEO_EXPORT'] == "true":
            if config['MODEL'] == "nerfacto" or config['MODEL'] == "splatfacto" or \
                config['MODEL'] == "splatfacto-big" or config['MODEL'] == "splatfacto-w-light" or \
                config['MODEL'] == "splatfacto-mcmc":
                model = "splatfacto"
                if config['MODEL'] == "splatfacto-w-light":
                    model = "splatfacto-w-light"
                if config['MODEL'] == "nerfacto":
                    model = "nerfacto"
                # Use correct output path for resume training
                if config['RUN_SFM'] == "false":
                    train_stage = RESUME_TRAIN_EXPERIMENT_NAME
                else:
                    train_stage = TRAIN_EXPERIMENT_NAME
                args = [
                    "interpolate",
                    "--load-config", f"outputs/unnamed/{model}/{train_stage}/config.yml",
                    "--output-path", os.path.join(output_path, "render.mp4"),
                    "--frame-rate", "10"
                ]
                pipeline.create_component(
                    name="Ply-Export-Video",
                    comp_type=ComponentType.exporter,
                    comp_environ=ComponentEnvironment.executable,
                    command="ns-render",
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
                    comp_type=ComponentType.exporter,
                    comp_environ=ComponentEnvironment.python,
                    command="3dgrut/render.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
                # Convert rendered images to video
                args = [
                    "-i", output_path,
                    "-o", os.path.join(output_path, "render.mp4"),
                    "-r", "10"
                ]
                pipeline.create_component(
                    name="Images-To-Video",
                    comp_type=ComponentType.transform,
                    comp_environ=ComponentEnvironment.python,
                    command="post_processing/images_to_video.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
    except Exception as e:
        error_message = f"Issue rendering trajectory video: {e}"
        pipeline.report_error(775, error_message)

    ##################################
    # TRANSFORM COMPONENT:
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
                comp_type=ComponentType.transform,
                comp_environ=ComponentEnvironment.python,
                command="post_processing/extract_video_thumbnail.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue extracting video thumbnail: {e}"
        pipeline.report_error(776, error_message)

    ##################################
    # EXPORT COMPONENT:
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
                comp_type=ComponentType.exporter,
                comp_environ=ComponentEnvironment.executable,
                command="aws",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue uploading thumbnail to S3: {e}"
        pipeline.report_error(790, error_message)

    ##################################
    # FILTER COMPONENT:
    # Crop splat bounds
    ##################################
    try:
        # Apply refinement of output bounds to remove noise if configured
        if config['REFINE_OUTPUT_BOUNDS'] == "true" and config['MODEL'] != "nerfacto":   
            args = [
                ply_path,
                ply_path,
                "--log-level", config['LOG_VERBOSITY'].upper(),
                "--mode", config['REFINEMENT_MODE'] #rigid_body, environment
            ]
            pipeline.create_component(
                name="Refine-BBox",
                comp_type=ComponentType.transform,
                comp_environ=ComponentEnvironment.python,
                command="post_processing/refine_bounding_box.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue cropping splat bounding box: {e}"
        pipeline.report_error(780, error_message)

    ##################################
    # TRANSFORM COMPONENT:
    # Clean PLY file - remove comments for SPZ compatibility
    ##################################
    try:
        if config['MODEL'] != "nerfacto":
            args = [
                "-i", ply_path
            ]
            pipeline.create_component(
                name="Clean-PLY",
                comp_type=ComponentType.transform,
                comp_environ=ComponentEnvironment.python,
                command="post_processing/clean_ply.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue cleaning PLY file: {e}"
        pipeline.report_error(781, error_message)

    ##################################
    # Transform COMPONENT:
    # Rotate splat for pre-sog export
    ##################################
    try:
        if config['ROTATE_SPLAT'] == "true" and config['MODEL'] != "nerfacto" and \
            config['ENABLE_SOGS'] == "true": # and config['MODEL'] != "splatfacto-mcmc":
            args = [
                "-i", ply_path,
                "-o", ply_path,
                "--rotations"
            ]
            if str(config['MODEL']).lower() != "3dgut" and \
                str(config['MODEL']).lower() != "3dgrt":
                # Apply standard rotation for non-3dgrt models
                args.append("x:270,y:0,z:0")
            else:
                args.append("x:180,y:0,z:0")
            pipeline.create_component(
                name="Rotation-Pre-Sog",
                comp_type=ComponentType.transform,
                comp_environ=ComponentEnvironment.python,
                command="post_processing/rotate_splat.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue rotating splat: {e}"
        pipeline.report_error(782, error_message)

    ##################################
    # Export COMPONENT: SOGS
    # Export sog from the .ply for compressed web viewing
    ##################################
    try:
        if config['MODEL'] != "nerfacto" and config['ENABLE_SOGS'] == "true": # and config['MODEL'] != "splatfacto-mcmc":
            args = [
                "-i", ply_path,
                "-o", sog_path,
                "-w"
            ]
            pipeline.create_component(
                name="SOGS-Export",
                comp_type=ComponentType.exporter,
                comp_environ=ComponentEnvironment.python,
                command="post_processing/convert_ply_to_sog.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue converting ply to SOGS: {e}"
        pipeline.report_error(783, error_message)

    ##################################
    # Transform COMPONENT:
    # Rotate splat for post-sog export
    ##################################
    try:
        if config['ROTATE_SPLAT'] == "true" and config['MODEL'] != "nerfacto" and \
            config['ENABLE_SOGS'] == "true": # and config['MODEL'] != "splatfacto-mcmc":
            args = [
                "-i", ply_path,
                "-o", ply_path,
                "--rotations", "x:-270, y:0, z:0"
            ]
            pipeline.create_component(
                name="Rotation-Post-Sog",
                comp_type=ComponentType.transform,
                comp_environ=ComponentEnvironment.python,
                command="post_processing/rotate_splat.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue rotating splat: {e}"
        pipeline.report_error(782, error_message)

    ##################################
    # TRANSFORM COMPONENT:
    # Rotate splat - pre-SPZ (SPZ module has built in rotation around X-Y)
    ##################################
    try:
        # Apply pre-rotation if configured
        if config['ROTATE_SPLAT'] == "true" and config['MODEL'] != "nerfacto" and config['ENABLE_SPZ'] == "true":
            args = [
                "-i", ply_path,
                "--rotations"
            ]
            if str(config['MODEL']).lower() != "3dgut" and \
                str(config['MODEL']).lower() != "3dgrt":
                # Apply standard rotation for non-3dgrt models
                args.append("x:270,y:180,z:0")
            else:
                args.append("x:180,y:180,z:0")
            pipeline.create_component(
                name="Rotation-Pre-SPZ",
                comp_type=ComponentType.transform,
                comp_environ=ComponentEnvironment.python,
                command="post_processing/rotate_splat.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue rotating splat: {e}"
        pipeline.report_error(782, error_message)

    ##################################
    # TRANSFORM COMPONENT:
    # Mirror splat - pre-SPZ (SPZ module has built in mirror around X-Y)
    ##################################
    try:
        if config['ROTATE_SPLAT'] == "true" and config['MODEL'] != "nerfacto" and config['ENABLE_SPZ'] == "true":
            args = [
                "--input", ply_path,
                "--axis", "x"  # Mirror along X-axis to compensate for SPZ built-in flip
            ]
            pipeline.create_component(
                name="Mirror-Pre-SPZ",
                comp_type=ComponentType.transform,
                comp_environ=ComponentEnvironment.python,
                command="post_processing/mirror_splat.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue mirroring the splat: {e}"
        pipeline.report_error(784, error_message)

    ##################################
    # EXPORT COMPONENT:
    # Export compressed SPZ splat file
    ##################################
    try:
        if config['MODEL'] != "nerfacto" and config['ENABLE_SPZ'] == "true":
            args = [
                ply_path
            ]
            pipeline.create_component(
                name="Spz-Export",
                comp_type=ComponentType.exporter,
                comp_environ=ComponentEnvironment.executable,
                command="splat_converter",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue creating compressed spz splat: {e}"
        pipeline.report_error(785, error_message)

    ##################################
    # TRANSFORM COMPONENT:
    # Rotate splat - post-SPZ
    ##################################
    try:
        # Apply post-rotation if configured
        if config['ROTATE_SPLAT'] == "true" and config['MODEL'] != "nerfacto" and config['ENABLE_SPZ'] == "true":
            args = [
                "-i", ply_path,
                "--rotations", "x:180,y:180,z:0"
            ]
            pipeline.create_component(
                name="Rotate-Post-SPZ",
                comp_type=ComponentType.transform,
                comp_environ=ComponentEnvironment.python,
                command="post_processing/rotate_splat.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue rotating splat: {e}"
        pipeline.report_error(782, error_message)

    ##################################
    # TRANSFORM COMPONENT: MIRROR SPLAT
    # Mirror splat - post-SPZ (SPZ module has built in mirror around X-Y)
    ##################################
    try:
        if config['ROTATE_SPLAT'] == "true" and config['MODEL'] != "nerfacto" and config['ENABLE_SPZ'] == "true":
            args = [
                "--input", ply_path,
                "--axis", "x"  # Mirror along X-axis to compensate for SPZ built-in flip
            ]
            pipeline.create_component(
                name="Mirror-Post-SPZ",
                comp_type=ComponentType.transform,
                comp_environ=ComponentEnvironment.python,
                command="post_processing/mirror_splat.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue mirroring the splat: {e}"
        pipeline.report_error(784, error_message)

    ##################################
    # EXPORT COMPONENT:
    # Export VIDEO to S3
    ##################################
    try:
        if config['ENABLE_VIDEO_EXPORT'] == "true" and config['MODEL'] != "nerfacto":
            args = ["s3", "cp"]
            args.extend([
                os.path.join(output_path, "render.mp4"),
                f"{config['S3_OUTPUT']}/{config['UUID']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.mp4"
            ])
            pipeline.create_component(
                name="S3-Export-Video",
                comp_type=ComponentType.exporter,
                comp_environ=ComponentEnvironment.executable,
                command="aws",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue uploading asset to S3: {e}"
        pipeline.report_error(790, error_message)

    ##################################
    # EXPORT COMPONENT:
    # Export SPZ to S3
    ##################################
    try:
        if config['RUN_TRAIN'] == "true" and config['MODEL'] != "nerfacto" and config['ENABLE_SPZ'] == "true":
            args = ["s3", "cp", "--content-type", "application/octet-stream"]
            if config['MODEL'] != "nerfacto":
                args.extend([
                    spz_path,
                    f"{config['S3_OUTPUT']}/{config['UUID']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.spz"
                ])
            pipeline.create_component(
                name="S3-Export-Spz",
                comp_type=ComponentType.exporter,
                comp_environ=ComponentEnvironment.executable,
                command="aws",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
        else:
            log.info(
                "Not configured to output a Gaussian Splat...skipping upload splat to S3."
                "Check the archive file for SfM results"
            )
    except Exception as e:
        error_message = f"Issue uploading asset to S3: {e}"
        pipeline.report_error(790, error_message)

    ##################################
    # EXPORT COMPONENT:
    # Export SOGS to S3
    ##################################
    try:
        if config['RUN_TRAIN'] == "true" and config['ENABLE_SOGS'] == "true" and \
            config['MODEL'] != "nerfacto": # and config['MODEL'] != "splatfacto-mcmc":
            args = ["s3", "cp"]
            args.extend([
                sog_path,
                f"{config['S3_OUTPUT']}/{config['UUID']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.sog"
            ])
            pipeline.create_component(
                name="S3-Export-Sog",
                comp_type=ComponentType.exporter,
                comp_environ=ComponentEnvironment.executable,
                command="aws",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
        else:
            log.info(
                "Not configured to output a SOGS...skipping upload splat to S3."
                "Check the archive file for SfM results"
            )
    except Exception as e:
        error_message = f"Issue uploading asset to S3: {e}"
        pipeline.report_error(790, error_message)

    ##################################
    # EXPORT COMPONENT:
    # Export PLY to S3
    ##################################
    try:
        if config['RUN_TRAIN'] == "true":
            args = ["s3", "cp"]
            if config['MODEL'] == "nerfacto":
                glb_path = os.path.join(output_path, "textured", f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.glb")
                args.extend([
                    glb_path,
                    f"{config['S3_OUTPUT']}/{config['UUID']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.glb"
                ])
            else:
                # Use correct path for resume training
                #if config['RUN_SFM'] == "false":
                #    ply_path = os.path.join(config['CODE_PATH'], "resume_exports", "splat.ply")
                args.extend([
                    ply_path,
                    f"{config['S3_OUTPUT']}/{config['UUID']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.ply"
                ])

            pipeline.create_component(
                name="S3-Export-Ply",
                comp_type=ComponentType.exporter,
                comp_environ=ComponentEnvironment.executable,
                command="aws",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
        else:
            log.info(
                "Not configured to output a Gaussian Splat...skipping upload splat to S3."
                "Check the archive file for SfM results"
            )
    except Exception as e:
        error_message = f"Issue uploading asset to S3: {e}"
        pipeline.report_error(790, error_message)

    ##################################
    # EXPORT COMPONENT:
    # Create and upload model.tar.gz archive to S3
    ##################################
    try:
        if IS_BATCH:
            args = [
                "s3", "cp",
                OUTPUT_TAR_PATH,
                f"{config['S3_OUTPUT']}/{config['UUID']}/output/model.tar.gz"
            ]
            pipeline.create_component(
                name="S3-Export-Archive",
                comp_type=ComponentType.exporter,
                comp_environ=ComponentEnvironment.executable,
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
        image_proc_time = None
        image_proc_end_time = None
        sfm_time = None
        sfm_end_time = None
        training_time = None
        for i in range(0, pipeline.config.num_components, 1):
            component = pipeline.components[i]
            log.info(f"Running component: {component.name}")
            match component.name:
                case "VideoToImages":
                    # VIDEO-TO-IMAGES CONDITIONAL COMPONENT
                    # Initialize video_found variable
                    video_found = False
                    
                    if (VIDEO is False and config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true') or \
                        (VIDEO is False and config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true'):
                        continue
                    else:
                        if (VIDEO is True and config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'false') or \
                            (VIDEO is True and config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'false'):
                            pipeline.run_component(i)
                            # Update VIDEO flag back to False after processing video to images
                            VIDEO = False
                        elif VIDEO is False and config['RUN_SFM'] == "false" and config['RUN_TRAIN'] == "true":
                            # Resume training - extract model.tar.gz
                            log.info("Detected resume training...")
                        else: # Archive of images or archive with a video
                            # unzip archive into temp directory
                            temp_path = os.path.join(config['DATASET_PATH'], 'temp')
                            with zipfile.ZipFile(input_file_path,"r") as zip_ref:
                                zip_ref.extractall(temp_path)  # nosemgrep: dangerous-tarfile-extractall
                            
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
                                # No video found, process as images
                                temp_dir_input = os.listdir(temp_path)[0]
                                if os.path.isdir(os.path.join(temp_path, temp_dir_input)): # Archive has a directory
                                    log.info("Moving directory from {temp_path} to {temp_dir_input}")
                                    # Remove existing images directory if it exists
                                    if os.path.exists(image_path):
                                        shutil.rmtree(image_path)
                                    os.rename(
                                        os.path.join(temp_path, temp_dir_input),
                                        image_path
                                    )
                                else: # Archive has files, not folder
                                    # Get all items in the source directory
                                    files = os.listdir(temp_path)
                                    # Move each item to the destination
                                    for filename in files:
                                        source_path = os.path.join(temp_path, filename)
                                        destination_path = os.path.join(image_path, filename)
                                        # Move the file
                                        shutil.move(source_path, destination_path)
                            
                            # Clean up temp directory
                            if os.path.exists(temp_path):
                                shutil.rmtree(temp_path)
                        if config['RUN_SFM'] == "true" and config['RUN_TRAIN'] == "true":
                            # Only process images if no video was found
                            if not video_found:
                                validate_and_resize_images(image_path, config, log, pipeline)
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
                case "PanoramaSfM":
                    # PANORAMA SFM COMPONENT
                    log.info("Running panorama SfM processing")
                    pipeline.run_component(i)
                case "ColmapSfM-Feature-Extractor":
                    # COLMAP FEATURE EXTRACTOR CONDITIONAL COMPONENT
                    log.info("Using standard COLMAP feature extraction")
                    current_time = int(time.time())
                    image_proc_time = current_time - start_time
                    image_proc_end_time = current_time  # Store the timestamp when image processing completes
                    log.info(f"Time to process images: {image_proc_time}s")

                    # If using pose prior, use the intrinsics from the txt file
                    if config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == "true":
                        camera_params = read_camera_params_from_file(os.path.join(sparse_model_path, "cameras.txt"))
                        component.args.extend([
                            "--ImageReader.camera_model", camera_params['model'],
                            "--ImageReader.camera_params", camera_params['params_str']
                        ])
                    # Only use GPU if not too many images
                    use_gpu = "0"
                    num_images = len(os.listdir(image_path))
                    if USE_GPU == "true" and num_images <= GPU_MAX_IMAGES and config['ENABLE_MULTI_GPU'] == "false":
                        use_gpu = "1"
                    component.args.extend([
                        "--SiftExtraction.use_gpu", "0" #use_gpu
                    ])
                    pipeline.run_component(i)
                case "ColmapSfM-Feature-Matcher":
                    # Only use GPU if not too many images
                    use_gpu = "0"
                    num_images = len(os.listdir(image_path))
                    if USE_GPU == "true" and num_images <= GPU_MAX_IMAGES:
                        use_gpu = "1"
                    component.args.extend([
                        "--SiftMatching.use_gpu", "0" #use_gpu
                    ])
                    pipeline.run_component(i)
                case "Colmap-to-Nerfstudio":
                    # Ensure we use the largest Colmap model if multiple found
                    if config['SFM_SOFTWARE_NAME'] == "colmap" or config['SFM_SOFTWARE_NAME'] == "glomap":
                        select_largest_colmap_model(sparse_path)
                    # Move existing transforms.json to transforms-in.json when using pose priors
                    # This ensures colmap-to-nerfstudio creates fresh transforms.json from updated COLMAP data
                    if config['USE_POSE_PRIOR_TRANSFORM_JSON'] == 'true' or \
                        config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'] == 'true':
                        if os.path.exists(transforms_out_path):
                            log.info(f"Moving {transforms_out_path} to {transforms_in_path} to preserve original")
                            shutil.move(transforms_out_path, transforms_in_path)
                    pipeline.run_component(i)
                case "Vggt-Ba":
                    cleanup_cuda_memory()
                    pipeline.run_component(i)
                case "Train":
                    # TRAIN CONDITIONAL COMPONENT
                    current_time = int(time.time())
                    if image_proc_end_time is None:
                        image_proc_end_time = 0
                    sfm_time = current_time - image_proc_end_time  # Calculate time since image processing completed
                    sfm_end_time = current_time  # Store the timestamp when SfM completes
                    log.info(f"Time for SfM: {sfm_time}s")
                    if config['ENABLE_MULTI_GPU'] == "false":
                        if config['MODEL'] != "3dgut" and config['MODEL'] != "3dgrt":
                            if config['SFM_SOFTWARE_NAME'] == "colmap" or config['SFM_SOFTWARE_NAME'] == "glomap" or \
                                config['SFM_SOFTWARE_NAME'] == "vggt":
                                # Move the sparse point cloud from sparse/0/* to colmap/sparse/*
                                log.info('Running Training...')
                                if config['RUN_SFM'] == "true":
                                    sparse_path_out = os.path.join(config['DATASET_PATH'], "colmap", "sparse")
                                    # Remove existing destination if it exists
                                    if os.path.exists(sparse_path_out):
                                        shutil.rmtree(sparse_path_out)
                                    # Ensure parent directory exists
                                    os.makedirs(os.path.dirname(sparse_path_out), exist_ok=True)
                                    shutil.move(sparse_path, sparse_path_out)
                                
                                # Set the image cache to disk if there are a lot of images to prevent OOM
                                # Set the max number of thread workers to 0 to prevent shared memory issues
                                if config['MODEL'] != "nerfacto":
                                    num_images = len(os.listdir(image_path))
                                    if num_images > GPU_MAX_IMAGES or config['RUN_SFM'] == "false":
                                        index = component.args.index("colmap")
                                        if index != -1:
                                            component.args.insert(index, "disk")
                                            component.args.insert(index, "--pipeline.datamanager.cache-images")
                                            #component.args.insert(index, "0")
                                            #component.args.insert(index, "--pipeline.datamanager.max-thread-workers")
                        else: # 3dgrut
                            if has_alpha_channel(os.path.join(image_path, os.listdir(image_path)[0])):
                                process_images(image_path)
                    elif config['ENABLE_MULTI_GPU'] == "true":
                        # Special handling for gsplat multi-GPU training - copy data to SageMaker location
                        # Extract dataset path from args
                        dataset_path = None
                        for j, arg in enumerate(component.args):
                            if arg == "--data-dir" and j + 1 < len(component.args):
                                dataset_path = component.args[j + 1]
                                break
                        
                        if dataset_path:
                            sagemaker_data_path = '/opt/ml/input/data/train'
                            os.makedirs(sagemaker_data_path, exist_ok=True)
                            
                            # Copy required data to SageMaker expected location
                            for item in ['sparse', 'images', 'transforms.json']:
                                if item == 'sparse':
                                    src_path = os.path.join(dataset_path, 'colmap', 'sparse')
                                else:
                                    src_path = os.path.join(dataset_path, item)
                                dst_path = os.path.join(sagemaker_data_path, item)
                                
                                if os.path.exists(src_path) and not os.path.exists(dst_path):
                                    if os.path.isdir(src_path):
                                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                                        log.info(f"Copied directory {src_path} to {dst_path}")
                                    else:
                                        shutil.copy2(src_path, dst_path)
                                        log.info(f"Copied file {src_path} to {dst_path}")
                            
                            # Update the data-dir argument to point to SageMaker location
                            for j, arg in enumerate(component.args):
                                if arg == "--data-dir":
                                    component.args[j + 1] = sagemaker_data_path
                                    break
                    try:
                        # Clean up CUDA memory before training
                        cleanup_cuda_memory()
                        pipeline.run_component(i)
                    except Exception as e:
                        error_str = str(e)
                        if "cuda" in error_str.lower() and "assert" in error_str.lower():
                            log.error(f"CUDA assertion error detected: {error_str}")
                            log.error("Consider reducing the number of Gaussians or using more conservative training parameters.")
                        raise e

                    # Copy the output ply and checkpoint over to where we expect it (keep originals for export)
                    if config['ENABLE_MULTI_GPU'] != "true":
                        if config['MODEL'] == "3dgut" or config['MODEL'] == "3dgrt":
                            root_exp_dir = os.path.join(output_path, TRAIN_EXPERIMENT_NAME)
                            if config['RUN_SFM'] == "false":
                                root_exp_dir = os.path.join(output_path, RESUME_TRAIN_EXPERIMENT_NAME)
                            exp_dir = os.listdir(root_exp_dir)[0]
                            shutil.move(os.path.join(root_exp_dir, exp_dir, "export_last.ply"), ply_path)
                        if config['MODEL'] == "3dgut" or config['MODEL'] == "3dgrt":
                            dest_dir = os.path.join(config['DATASET_PATH'], "3dgrut_models")
                            base_dir = os.path.join(config['DATASET_PATH'], 'exports', TRAIN_EXPERIMENT_NAME)
                            if config['RUN_SFM'] == "false":
                                base_dir = os.path.join(config['DATASET_PATH'], 'exports', RESUME_TRAIN_EXPERIMENT_NAME)
                            src_dir = os.path.join(base_dir, sorted(os.listdir(base_dir))[-1])
                            os.makedirs(dest_dir, exist_ok=True)
                            shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
                case "Nerfstudio-Export":
                    # NERFSTUDIO EXPORT CONDITIONAL COMPONENT
                    # For resume training, update config path to use the recreated checkpoint location
                    if config['RUN_SFM'] == "false":
                        if os.path.exists(model_config_path):
                            for j, arg in enumerate(component.args):
                                if arg == "--load-config" or arg == "--load_config":
                                    component.args[j + 1] = model_config_path
                                    log.info(f"Updated export config path to: {model_config_path}")
                                    break
                        else:
                            log.warning(f"Config file not found at {model_config_path}, export may fail")
                    pipeline.run_component(i)
                    current_time = int(time.time())
                    training_time = current_time - sfm_end_time  # Calculate actual training time
                    log.info(f"Time to train: {training_time}s")
                    # Clean up CUDA memory after training
                    cleanup_cuda_memory()
                case "Nerfstudio-Export-Nerfacto":
                    # NERFSTUDIO NERFACTO EXPORT CONDITIONAL COMPONENT
                    pipeline.run_component(i)
                    obj_to_glb(
                        os.path.join(output_path, "textured", "mesh.obj"),
                        os.path.join(output_path, "textured", f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.glb")
                    )
                case "S3-Export-Ply":
                    # S3 UPLOAD CONDITIONAL COMPONENT
                    log.info("Uploading output to S3")
                    pipeline.run_component(i)
                    
                    # Copy nerfstudio_models and config.yml to dataset directory for archive
                    if config['RUN_TRAIN'] == "true" and config['ENABLE_MULTI_GPU'] != "true":
                        if config['MODEL'] != "3dgut" and config['MODEL'] != "3dgrt":
                            # Copy nerfstudio_models directory to dataset
                            dataset_models_path = os.path.join(config['DATASET_PATH'], "nerfstudio_models")
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
                            dataset_config_path = os.path.join(config['DATASET_PATH'], "config.yml")
                            if os.path.exists(model_config_path):
                                shutil.copy2(model_config_path, dataset_config_path)
                                log.info(f"Copied config.yml to dataset: {dataset_config_path}")

                    # Clean up the dataset
                    cleanup_dataset(config['DATASET_PATH'])

                    # Copy result over to where SM expects it
                    shutil.move(config['DATASET_PATH'], OUTPUT_DATASET_PATH)
                    log.info(f"Successful pipeline result generation located at \
                            {config['S3_OUTPUT']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.*")
                case "S3-Export-Archive": # Batch only
                    # S3 ARCHIVE UPLOAD COMPONENT
                    # Copy nerfstudio_models and config.yml to dataset directory for archive
                    if config['RUN_TRAIN'] == "true" and config['ENABLE_MULTI_GPU'] != "true":
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
                    dataset_source = OUTPUT_DATASET_PATH if os.path.exists(OUTPUT_DATASET_PATH) else config['DATASET_PATH']
                    cleanup_dataset(dataset_source)
                    create_tarball(dataset_source, OUTPUT_TAR_PATH, "dataset")
                    log.info(f"Created model.tar.gz archive from {dataset_source}")
                    pipeline.run_component(i)
                case "Ply-Export-Video":
                    pipeline.run_component(i)
                    if config['MODEL'] == "3dgut" or config['MODEL'] == "3dgrt":
                        print("***OUTPUT_FILES:***")
                        folders = sorted(os.listdir(os.path.join(output_path, TRAIN_EXPERIMENT_NAME)))
                        folder = folders[-1]
                        folder_ = os.path.join(output_path, TRAIN_EXPERIMENT_NAME, folder)
                        subfolder = next((f for f in os.listdir(folder_) if f.startswith('ours_') and os.path.isdir(os.path.join(folder_, f))), None)
                        print(folder)
                        print(folder_)
                        print(subfolder)
                        print(", ".join(os.listdir(os.path.join(folder_, subfolder, "renders"))))
                case "SOGS-Export":
                    # Add CPU flag if we have many images
                    if os.path.exists(image_path):
                        # Count actual image files, excluding mask files and handling subdirectories
                        num_images = 0
                        if config['MODEL'] in ["3dgrt", "3dgut"]:
                            # For 3DGRUT models, exclude .png.png mask files
                            for f in os.listdir(image_path):
                                if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.endswith('.png.png'):
                                    num_images += 1
                        else:
                            # For other models, count images in subdirectories too
                            for root, dirs, files in os.walk(image_path):
                                for f in files:
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                                        num_images += 1

                        if num_images > GPU_MAX_IMAGES and "-c" not in component.args:
                            component.args.append("-c")
                            log.info(f"Added CPU flag for SOG conversion due to {num_images} images (>{GPU_MAX_IMAGES})")
                    
                    pipeline.run_component(i)
                case "S3-Export-Sog":
                    # Check if SOG file exists before attempting S3 upload
                    if os.path.exists(sog_path) and os.path.getsize(sog_path) > 0:
                        pipeline.run_component(i)
                    else:
                        log.info("SOG file not found or empty, skipping S3 upload")
                case _: # Default case, run Component
                    pipeline.run_component(i)
        pipeline.session.status = Status.STOP
        log.info(f"Pipeline status changed to {pipeline.session.status}")
        current_time = int(time.time())
        total_time = current_time - start_time
        log.info(f"Total Time: {total_time}s")
    except Exception as e:
        error_message = f"General error running the pipeline: {e}"
        pipeline.report_error(795, error_message)

