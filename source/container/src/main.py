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

""" This script is the main entry point into training a gaussian splat from a
given set of images or video """

"""
ERROR CODES
700, "Error reading camera parameters from file
705, "Input file type not supported. Only .mp3, .mp4, .mov, and .zip with .png or .jpeg/.jpg files are supported for input"
710, "Required environment variables not set. Check that the payload has the required fields"
715, "Configuration not supported. Only pose prior transform json or pose prior colmap model files can be enabled, not both."
720, "Improper file type given for prior pose transformations. Only '.zip' is supported."
725, "Issue transforming pose to Colmap component"
730, "Issue creating video to images component"
735, "Issue creating spherical image component"
740, "Issue creating background removal component"
745, "Issue creating human subject removal component"
750, "SfM Software name given not implemented"
755, "Issue creating the SfM component"
760, "Issue creating the Colmap to Nerfstudio component"
765, "Model not supported"
767, "Trainer specified does not match proper configuration"
770, "Issue running the training session, stage 1"
780, "Issue exporting splat from NerfStudio"
781, "Issue rotating splat before SPZ conversion"
783, "Issue creating compressed spz splat"
784, "Issue rotating splat after SPZ conversion"
785, "Issue uploading asset to S3"
790, "The archive doesn't contain supported image files .jpg, .jpeg, or .png"
795, "General error running the pipeline"
"""

import os
import sys
import time
import json
import math
import torch
import shutil
import logging
import zipfile
import multiprocessing
import trimesh
import cv2
import numpy as np
import tarfile
from tqdm import tqdm as tqdm_func
from PIL import Image
from pipeline import Pipeline, Status, ComponentEnvironment, ComponentType

def resize_to_4k(image_path, orientation="landscape"):
    """
    Resize image based on orientation:
    - For landscape: resize to 4k width (3840 pixels) if original width is larger
    - For portrait: resize to 4k height (2160 pixels) if original height is larger
    
    Args:
        image_path: string path to the image
        orientation: string, either "landscape" or "portrait"
        
    Returns:
        numpy array: resized image if dimension exceeded 4k threshold, original image otherwise
    """
    image = cv2.imread(image_path)

    # Get current dimensions
    height, width = image.shape[:2]
    
    # Determine if the image is portrait or landscape if not specified
    if orientation.lower() == "auto":
        orientation = "portrait" if height > width else "landscape"
    
    # Set target dimension based on orientation
    if orientation.lower() == "portrait":
        # For portrait, we check height against 4K height (2160 pixels)
        target_dimension = 2160
        current_dimension = height
        is_exceeding = height > target_dimension
    else:  # landscape
        # For landscape, we check width against 4K width (3840 pixels)
        target_dimension = 3840
        current_dimension = width
        is_exceeding = width > target_dimension
    
    # Only resize if the relevant dimension exceeds the 4K threshold
    if is_exceeding:
        aspect_ratio = width / height
        
        if orientation.lower() == "portrait":
            # Calculate new width to maintain aspect ratio
            new_width = int(target_dimension * aspect_ratio)
            new_height = target_dimension
        else:  # landscape
            # Calculate new height to maintain aspect ratio
            new_height = int(target_dimension / aspect_ratio)
            new_width = target_dimension
        
        # Choose interpolation method based on whether we're shrinking or enlarging
        if (orientation.lower() == "portrait" and height > new_height) or \
           (orientation.lower() == "landscape" and width > new_width):
            # If shrinking
            resized = cv2.resize(image, (new_width, new_height), 
                               interpolation=cv2.INTER_AREA)
        else:
            # If enlarging
            resized = cv2.resize(image, (new_width, new_height), 
                               interpolation=cv2.INTER_CUBIC)
        return resized
    
    # Return original image if no resize needed
    return image

def read_camera_params_from_file(cameras_txt_path):
    """Read camera parameters from cameras.txt file"""
    try:
        with open(cameras_txt_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
                
            # Parse camera line
            # Format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            parts = line.split()
            if len(parts) >= 5:
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                
                # Get the parameters - they might be comma-separated or space-separated
                params_str = ' '.join(parts[4:])

                # First, normalize the input by replacing commas with spaces
                normalized_params = params_str.replace(',', ' ')

                # Split by whitespace to get individual parameters
                param_list = normalized_params.split()

                # Join the parameters with commas to create the final comma-separated list
                comma_separated = ','.join(param_list)

                # Return the first camera entry
                return {
                    'id': camera_id,
                    'model': model,
                    'width': width,
                    'height': height,
                    'params_str': comma_separated
                }
        
        return None
    except Exception as e:
        print(f"Error Code 700: error reading camera parameters from file: {str(e)}")
        return None
        
def validate_input_media(filename: str)->bool:
    """
    # Validation Check if single images or video is input
    """
    ext = str(os.path.splitext(filename)[1]).lower()
    if ext == ".mp4" or ext == ".mov":
        return True
    elif ext == ".zip":
        return False
    else:
        error = """Error Code 705: Input file type not supported. Only .mp3, .mp4, .mov, and
            .zip with .png or .jpeg/.jpg files are supported for input"""
        raise RuntimeError(error)

def load_config(config_names: list, config_values: list)->dict:
    """
    # Load configuration from environment variables into a dict
    """
    for i, config_name in enumerate(config_names):
        if config_name in os.environ:
            config_values[i] = os.environ[config_name]

    conf = dict(zip(config_names, config_values))
    return conf

def obj_to_glb(obj_path: str, glb_path: str)->None:
    """
        Export the obj and material as a .glb file
    """
    mesh = trimesh.load(
        obj_path,
        file_type='obj',
        process=False,
        force='mesh',
        skip_texture=False,
        split_object=False,
        group_material=False
    )
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh = mesh.apply_transform(rot)
    trimesh.load(
        file_obj=mesh,
        file_type='obj'
    ).export(glb_path, file_type='glb')

def count_up_to(n):
    """
        Convert an integer to a list of numbers as string
    """
    return ','.join(str(i) for i in range(n))

def untar_gz(file_path, extract_path='.'):
    """
    Extracts a .tar.gz file.

    Args:
        file_path (str): The path to the .tar.gz file.
        extract_path (str, optional): The directory to extract to. Defaults to current directory.
    """
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(extract_path)
        print(f"Successfully extracted '{file_path}' to '{extract_path}'")
    except FileNotFoundError:
        print(f"Error: File not found: '{file_path}'")
    except tarfile.ReadError:
        print(f"Error: Could not open '{file_path}' with read mode 'r:gz'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def has_alpha_channel(image_path):
    """
    Check if an image has an alpha channel.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        bool: True if the image has an alpha channel, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            return img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)
    except Exception as e:
        print(f"Error checking alpha channel in {image_path}: {str(e)}")
        return False

def process_images(input_dir, output_dir=None):
    """
    Process RGBA images by:
    1. Converting them to RGB
    2. Creating mask files from the alpha channel
    
    Args:
        input_dir: Directory containing RGBA images
        output_dir: Directory to save processed images (if None, will use input_dir)
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    for image_file in tqdm_func(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, image_file)
        
        # Open the image
        img = Image.open(input_path)
        
        # Check if the image has an alpha channel
        if img.mode == 'RGBA':
            # Split into RGB and alpha
            rgb = img.convert('RGB')
            alpha = img.split()[3]
            
            # Save RGB image (overwrite original)
            output_rgb_path = os.path.join(output_dir, image_file)
            rgb.save(output_rgb_path)
            
            # Save alpha as mask
            base_name = os.path.splitext(image_file)[0]
            mask_file = f"{base_name}_mask.png"
            output_mask_path = os.path.join(output_dir, mask_file)
            alpha.save(output_mask_path)
            
            print(f"Processed {image_file} -> RGB + mask")
        else:
            print(f"Skipping {image_file} - not in RGBA format")

if __name__ == "__main__":
    try:
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
            error_message = """Error Code 710: Required environment variables not set.
                Check that the payload has the required fields"""
            raise RuntimeError(error_message)
        
        # Unpack the sam2 models
        untar_gz(os.path.join(os.environ["MODEL_PATH"], "models.tar.gz"), os.environ["MODEL_PATH"])

        # Instantiate Pipeline Session
        pipeline = Pipeline(
            name="3DGS-Pipeline",
            uuid=config['UUID'],
            num_threads=str(multiprocessing.cpu_count()),
            num_gpus=str(torch.cuda.device_count()),
            log_verbosity=config['LOG_VERBOSITY'])
        log = pipeline.session.log
        pipeline.session.status = Status.INIT
        log.info(f"Pipeline status changed to {pipeline.session.status}")
    except Exception as e:
        error_message = f"""Required environment variables not set.
            Check that the payload has the required fields: {e}"""
        pipeline.report_error(710, error_message)

    # Options and Defaults
    log.info(f"UUID: {config['UUID']}")
    log.info(f"Dataset Path: {config['DATASET_PATH']}")
    log.info(f"Filename: {config['FILENAME']}")
    log.info(f"S3 Input Path: {config['S3_INPUT']}")
    log.info(f"S3 Output Path: {config['S3_OUTPUT']}")

    # Check if video or zip of images given
    VIDEO = validate_input_media(config['FILENAME'])
    log.info(f"Is Video?: {VIDEO}")

    # Ensure we have an /images directory in dataset path for Colmap/Glomap
    image_path = os.path.join(config['DATASET_PATH'], "images")
    if not os.path.isdir(image_path):
        log.info(f"Creating '/images' directory in {config['DATASET_PATH']}")
        os.makedirs(image_path, exist_ok=True)

    # Ensure we have a /sparse directory in dataset path for NerfStudio
    sparse_path = os.path.join(config['DATASET_PATH'], "sparse")
    if not os.path.isdir(sparse_path):
        log.info(f"Creating '/sparse/0' directory in {config['DATASET_PATH']}")
        os.makedirs(os.path.join(sparse_path, "0"), exist_ok=True)

    # Create the output directory for pre-processing
    filter_output_dir = os.path.join(config['DATASET_PATH'], "filtered_images")
    if not os.path.isdir(filter_output_dir):
        os.makedirs(filter_output_dir, exist_ok=True)

    # Create the output directory for export
    output_path = os.path.join(config['DATASET_PATH'], "exports")
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    colmap_db_path = os.path.join(config['DATASET_PATH'], "database.db")
    transforms_in_path = os.path.join(config['DATASET_PATH'], "transforms-in.json")
    transforms_out_path = os.path.join(config['DATASET_PATH'], "transforms.json")
    spherical_num_images = int(float(config['MAX_NUM_IMAGES'])/float(6))
    if str(config['SPHERICAL_CAMERA']).lower() == "true": # 6 views per 360 image using cube faces
        num_images = spherical_num_images
    else:
        num_images = int(float(config['MAX_NUM_IMAGES']))
    head, extension = os.path.splitext(str(config['FILENAME']))
    os.environ['CUDA_VISIBLE_DEVICES'] = count_up_to(int(pipeline.config.num_gpus))
    input_file_path = os.path.join(config['DATASET_PATH'], config['FILENAME'])

    current_dir_path = os.path.dirname(os.path.realpath(__file__))

    ##################################
    # TRANSFORM: Pose Transform for SfM
    ##################################
    try:
        if str(config['USE_POSE_PRIOR_TRANSFORM_JSON']).lower() == 'true' and \
            str(config['USE_POSE_PRIOR_COLMAP_MODEL_FILES']).lower() == 'true':
            raise RuntimeError(
                pipeline.report_error(
                    715,
                    f"""Configuration not supported.
                    Only pose prior transform json or pose prior colmap model files can be enabled, not both."""
                )
            )
        if str(config['USE_POSE_PRIOR_TRANSFORM_JSON']).lower() == 'true' or \
            str(config['USE_POSE_PRIOR_COLMAP_MODEL_FILES']).lower() == 'true':
            if VIDEO is False and extension.lower() == ".zip":
                if str(config['USE_POSE_PRIOR_TRANSFORM_JSON']).lower() == 'true':
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
                        720,
                        f"""Improper file type {extension} given for prior pose transformations.
                        Only '.zip' is supported."""
                    )
                )
    except Exception as e:
        error_message = f"Issue transforming pose to colmap component: {e}"
        pipeline.report_error(725, error_message)

    ##################################
    # TRANSFORM: Video to Images
    ##################################
    try:
        if VIDEO is True:
            if str(config['REMOVE_BACKGROUND']).lower() == "true" and \
                str(config['BACKGROUND_REMOVAL_MODEL']).lower() == "sam2":
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
            else: # Just extract the frames, remove background later
                args = [
                    "-i", input_file_path,
                    "-o", image_path,
                    "-n", config['MAX_NUM_IMAGES']
                ]
                pipeline.create_component(
                    name="VideoToImages",
                    comp_type=ComponentType.transform,
                    comp_environ=ComponentEnvironment.python,
                    command="video_processing/simple_video_to_images.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=False
                )
        elif VIDEO is False and str(config['BACKGROUND_REMOVAL_MODEL']).lower() == "sam2" and \
            str(config['REMOVE_BACKGROUND']).lower()=="true":
            sys.exit("Error: SAM2 Background removal is only supported for video input")
        else:
            args = [
            ]
            pipeline.create_component(
                name="VideoToImages",
                comp_type=ComponentType.transform,
                comp_environ=ComponentEnvironment.python,
                command="",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )

    except Exception as e:
        error_message = f"Issue creating video to images component: {e}"
        pipeline.report_error(730, error_message)

    ##################################
    # FILTER: Remove Blurry Images
    ##################################
    try:
        # REMOVE BLURRY IMAGES COMPONENT
        if str(config['FILTER_BLURRY_IMAGES']).lower() == "true":
            args = [
                    "-I", image_path,
                    "-r", "30",
                    "-n", str(config['MAX_NUM_IMAGES']),
                    "-O", image_path
            ]

            if str(config['LOG_VERBOSITY'].lower() == "debug"):
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

    except Exception as e:
        error_message = f"Issue creating remove blurry images component: {e}"
        pipeline.report_error(730, error_message)

    ##################################
    # FILTER: Remove Background
    ##################################
    try:
        if config['REMOVE_BACKGROUND'].lower() == "true" and \
            config['BACKGROUND_REMOVAL_MODEL'].lower() != "sam2":
            # BACKGROUND REMOVAL COMPONENT
            if config['BACKGROUND_REMOVAL_MODEL'].lower() == "u2net_human":
                model = "u2net_human_seg"
            else:
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
        pipeline.report_error(740, error_message)

    ##################################
    # Filter: Spherical image
    # into cubemap and perspective images
    ##################################
    try:
        if config['SPHERICAL_CAMERA'].lower() == "true":
            # VIDEO TO IMAGES COMPONENT
            args = [
                "-d", image_path,
                "-ossfo", config['OPTIMIZE_SEQUENTIAL_SPHERICAL_FRAME_ORDER'],
                "-rf", config['SPHERICAL_CUBE_FACES_TO_REMOVE']
            ]
            pipeline.create_component(
                name="SphericaltoPerspective",
                comp_type=ComponentType.filter,
                comp_environ=ComponentEnvironment.python,
                command="spherical/equirectangular_to_perspective.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue creating spherical image component: {e}"
        pipeline.report_error(735, error_message)

    ##################################
    # FILTER: Remove Human Subject
    ##################################
    try:
        if config['REMOVE_HUMAN_SUBJECT'].lower() == "true":
            model = "u2net_human_seg"
            # HUMAN SUBJECT REMOVAL COMPONENT
            if os.path.isdir(image_path): # Process by image
                # Create the output directory
                filter_output_dir = os.path.join(config['DATASET_PATH'],"filtered_images")
                if not os.path.isdir(filter_output_dir):
                    os.makedirs(filter_output_dir, exist_ok=True)
                if not os.path.isdir(f"{config['DATASET_PATH']}/masked_images"):
                    os.makedirs(f"{config['DATASET_PATH']}/masked_images", exist_ok=True)
            else:
                pipeline.report_error(740, f"Images directory {image_path} doesn't exist")
            args = [
                "-oi", image_path,
                "-om", image_path,
                "-od", f"{config['DATASET_PATH']}/masked_images"
            ]
            pipeline.create_component(
                name="RemoveHumanSubject",
                comp_type=ComponentType.transform,
                comp_environ=ComponentEnvironment.python,
                command="segmentation/remove_object_using_mask.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue creating human subject removal component: {e}"
        pipeline.report_error(745, error_message)

    ##################################
    # TRANSFORM: Images to Point Cloud
    ##################################
    ### COLMAP SfM
    try:
        if config['RUN_SFM'].lower() == "true":
            if str(config['SFM_SOFTWARE_NAME']).lower() == "colmap" or \
                str(config['SFM_SOFTWARE_NAME']).lower() == "glomap":
                # FEATURE EXTRACTOR COMPONENT
                args = [
                    "feature_extractor",
                    "--database_path", colmap_db_path,
                    "--image_path", image_path,
                    "--ImageReader.single_camera", "1",
                    "--SiftExtraction.num_threads", pipeline.config.num_threads#,
                ]
                if str(config['ENABLE_MULTI_GPU']).lower() == "true" or \
                    str(config['MODEL']).lower() != "3dgut" or \
                    str(config['MODEL']).lower() != "3dgrt":
                    args.extend([
                        "--ImageReader.camera_model", "PINHOLE"
                    ])

                if config['ENABLE_ENHANCED_FEATURE_EXTRACTION'].lower() == "true":
                    args.extend([
                        "--SiftExtraction.estimate_affine_shape", "1",
                        "--SiftExtraction.domain_size_pooling", "1"
                    ])
                if config['SPHERICAL_CAMERA'].lower() == "true":
                    args.extend([
                        "--SiftExtraction.first_octave", "0",
                        "--SiftExtraction.max_num_orientations", "3"
                    ])
                if config['LOG_VERBOSITY'].lower() == "error":
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
                if str(config['USE_POSE_PRIOR_TRANSFORM_JSON']).lower() == 'true' or \
                    str(config['USE_POSE_PRIOR_COLMAP_MODEL_FILES']).lower() == 'true':
                    if str(config['USE_POSE_PRIOR_TRANSFORM_JSON']).lower() == 'true':
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
                if config['MATCHING_METHOD'].lower() == "sequential":
                    args = [
                        "sequential_matcher",
                        "--database_path",  colmap_db_path,
                        "--SiftMatching.num_threads", pipeline.config.num_threads,
                        "--SequentialMatching.quadratic_overlap", "1",
                        "--SiftMatching.guided_matching", "0"
                    ]
                    if config['SPHERICAL_CAMERA'].lower() == "true":
                        args.extend([
                            "--SequentialMatching.overlap", "10",
                            "--SequentialMatching.loop_detection", "0"
                        ])   
                    else:
                        args.extend([
                            "--SequentialMatching.overlap", "10",
                            "--SequentialMatching.loop_detection", "1",
                            "--SequentialMatching.loop_detection_period", config['MAX_NUM_IMAGES'],
                            "--SequentialMatching.loop_detection_num_images", config['MAX_NUM_IMAGES'],
                            "--SequentialMatching.vocab_tree_path", f"{config['CODE_PATH']}/vocab_tree_flickr100K_words32K.bin"
                        ])
                elif config['MATCHING_METHOD'].lower() == "spatial":
                    args = [
                        "spatial_matcher",
                        "--database_path", colmap_db_path,
                        "--SpatialMatching.ignore_z", "0",
                        "--SiftMatching.num_threads", pipeline.config.num_threads#,
                    ]
                elif config['MATCHING_METHOD'].lower() == "vocab":
                    args = [
                        "vocab_tree_matcher",
                        "--database_path", colmap_db_path,
                        "--SiftMatching.guided_matching", "1",
                        "--VocabTreeMatching.num_images", str(math.ceil(float(config['MAX_NUM_IMAGES'])/3)),
                        "--VocabTreeMatching.vocab_tree_path", f"{config['CODE_PATH']}/vocab_tree_flickr100K_words32K.bin",
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
                if config['LOG_VERBOSITY'].lower() == "error":
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

                if str(config['USE_POSE_PRIOR_COLMAP_MODEL_FILES']).lower() == "true" or \
                    str(config['USE_POSE_PRIOR_TRANSFORM_JSON']).lower() == "true":
                    # TRIANGULATION COMPONENT
                    args = [
                        'point_triangulator',
                        '--database_path', colmap_db_path,
                        '--image_path', image_path,
                        '--input_path', os.path.join(sparse_path, "0"),
                        '--output_path', os.path.join(sparse_path, "0"),
                        '--refine_intrinsics', "1",
                        '--Mapper.multiple_models', "0",
                        '--Mapper.num_threads', pipeline.config.num_threads
                    ]
                    if config['LOG_VERBOSITY'].lower() == "error":
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
                    if str(config['SFM_SOFTWARE_NAME']).lower() == "colmap" :
                        args = [
                            "mapper",
                            "--database_path", colmap_db_path,
                            "--image_path", image_path,
                            "--output_path", sparse_path,
                            "--Mapper.multiple_models", "0",
                            "--Mapper.num_threads", pipeline.config.num_threads
                        ]
                        if config['LOG_VERBOSITY'].lower() == "error":
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
                    else:
                        args = [
                            "mapper",
                            "--database_path", colmap_db_path,
                            "--image_path", image_path,
                            "--output_path", sparse_path
                        ]
                        if config['LOG_VERBOSITY'].lower() == "error":
                            args.extend([
                                "--log_level", "1"
                            ])
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
                if str(config['ENABLE_MULTI_GPU']).lower() == "true":
                    args = [
                        "image_undistorter",
                        "--image_path", image_path,
                        "--input_path", os.path.join(sparse_path, "0"),
                        "--output_path", os.path.join(sparse_path, "0"),
                        "--output_type", "COLMAP"
                    ]
                    if config['LOG_VERBOSITY'].lower() == "error":
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
            else:
                raise RuntimeError(
                    pipeline.report_error(
                        750, f"SfM Software not implemented yet:{str(config['SFM_SOFTWARE_NAME']).lower()}"
                    )
                )
        else:
            log.info("SfM configured to be skipped...skipping SfM")
    except Exception as e:
        error_message = f"Issue creating the SfM component: {e}"
        pipeline.report_error(755, error_message)

    ##################################
    # TRANSFORM: Point Cloud, Images,
    # and Poses to NerfStudio format
    ##################################
    # COLMAP TO NERFSTUDIO CONVERSION COMPONENT
    try:
        if config['GENERATE_SPLAT'].lower() == "true":
            if str(config['SFM_SOFTWARE_NAME']).lower() == "colmap" or \
                str(config['SFM_SOFTWARE_NAME']).lower() == "glomap":
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
                        f"SfM Software name given not implemented:{str(config['SFM_SOFTWARE_NAME']).lower()}"
                    )
                )
        else:
            log.info("Not configured to output a Gaussian Splat...skipping dataset conversion.")
    except Exception as e:
        error_message = f"Issue creating the Colmap to Nerfstudio component: {e}"
        pipeline.report_error(760, error_message)

    ##################################
    # TRANSFORM: Point Cloud, Images,
    # and Poses to 3D Gaussian Splat
    ##################################
    # CREATE NERFSTUDIO STAGE 1 COMPONENT
    try:
        if config['GENERATE_SPLAT'].lower() == "true":
            if config['SFM_SOFTWARE_NAME'].lower() == "glomap" or \
                config['SFM_SOFTWARE_NAME'].lower() == "colmap":
                data_model = "colmap"
            # Single GPU gsplat
            if str(config['ENABLE_MULTI_GPU']).lower() == "false" and \
                str(config['MODEL']).lower() != "3dgut" and \
                str(config['MODEL']).lower() != "3dgrt":
                args = [
                    config['MODEL'],
                    "--timestamp", "train-stage-1",
                    "--viewer.quit-on-train-completion=True"
                ]
                if str(config['LOG_VERBOSITY'].lower() != "debug"):
                    args.extend([
                        "--logging.local-writer.enable", "False",
                        "--logging.profiler", "none"
                    ])
                if config['MODEL'] == "nerfacto":
                    args.extend([
                        "--pipeline.model.predict-normals", "True",
                        "--max-num-iterations", str(config['MAX_STEPS']),
                    ])
                elif config['MODEL'] == "splatfacto" or \
                    config['MODEL'] == "splatfacto-big" or \
                    config['MODEL'] == "splatfacto-mcmc":
                    args.extend([
                        "--pipeline.model.use_scale_regularization=True",
                        "--max-num-iterations", str(int(int(config['MAX_STEPS'])))
                    ])
                elif config['MODEL'] == "splatfacto-w-light":
                    args.extend([
                        "--pipeline.model.enable-bg-model=True",
                        "--pipeline.model.enable-alpha-loss=True",
                        "--pipeline.model.enable-robust-mask=True",
                        "--max-num-iterations", str(config['MAX_STEPS']),
                    ])
                else:
                    raise RuntimeError(pipeline.report_error(765, "Model not supported"))

                args.extend([
                    data_model,
                    "--data", f"{config['DATASET_PATH']}",
                    "--downscale-factor", "1",
                ])

                pipeline.create_component(
                    name="Train-Stage1",
                    comp_type=ComponentType.transform,
                    comp_environ=ComponentEnvironment.executable,
                    command="ns-train",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
            # Multi-gpu gsplat
            elif str(config['ENABLE_MULTI_GPU']).lower() == "true" and \
                str(config['MODEL']).lower() != "3dgut" and \
                str(config['MODEL']).lower() != "3dgrt":
                #multi-gpu, use gsplat training strategy
                batch_size = 1
                step_scaler = float(1/(int(pipeline.config.num_gpus)*batch_size))
                if config['MODEL'] == "splatfacto-mcmc":
                    model = "mcmc"
                else:
                    model = "default"
                args = [
                    model,
                    "--max_steps", str(config['MAX_STEPS']),
                    "--result-dir", output_path,
                    "--data_factor", "1",
                    "--steps_scaler", str(step_scaler),
                    "--disable_viewer",
                    "--packed",
                    "--batch-size", str(batch_size),
                    "--data-dir", config['DATASET_PATH']
                ]
                pipeline.create_component(
                    name="Train-Stage1",
                    comp_type=ComponentType.transform,
                    comp_environ=ComponentEnvironment.python,
                    command="gsplat/examples/simple_trainer.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
            # 3DGRUT
            elif str(config['MODEL']).lower() == "3dgut" or \
                str(config['MODEL']).lower() == "3dgrt":
                os.makedirs(os.path.join(config['DATASET_PATH'], '3dgrut', 'runs'), exist_ok=True)
                args = [
                    "--config-name", f"apps/colmap_{str(config['MODEL']).lower()}_mcmc.yaml",
                    f"path={config['DATASET_PATH']}",
                    f"out_dir={os.path.join(config['DATASET_PATH'], '3dgrut', 'runs')}",
                    f"experiment_name={config['UUID']}",
                    f"n_iterations={str(config['MAX_STEPS'])}",
                    f"scheduler.positions.max_steps={str(config['MAX_STEPS'])}",
                    "dataset.downsample_factor=1",
                    "export_ply.enabled=true",
                    f"model.print_stats=true"
                ]
                pipeline.create_component(
                    name="Train-Stage1",
                    comp_type=ComponentType.transform,
                    comp_environ=ComponentEnvironment.python,
                    command="3dgrut/train.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
            else:
                error_message = f"Trainer specified does not match proper configuration: {e}"
                pipeline.report_error(765, error_message)
        else:
            log.info("Not configured to output a Gaussian Splat...skipping training stage 1.")
    except Exception as e:
        error_message = f"Issue running the training session stage 1: {e}"
        pipeline.report_error(770, error_message)

    ##################################
    # EXPORT:
    # CREATE NERFSTUDIO EXPORT COMPONENT
    ##################################
    try:
        if config['GENERATE_SPLAT'].lower() == "true" and \
            str(config['MODEL']).lower() != "3dgut" and \
            str(config['MODEL']).lower() != "3dgrt":
            if str(config['ENABLE_MULTI_GPU']).lower() == "true":
                ckpt_dir = os.path.join(output_path, "ckpts")
                args = [
                    ckpt_dir,
                    os.path.join(output_path, "splat.ply")
                ]
                pipeline.create_component(
                    name="Nerfstudio-Export",
                    comp_type=ComponentType.exporter,
                    comp_environ=ComponentEnvironment.python,
                    command="training/gsplat_pt_to_ply.py",
                    args=args,
                    cwd=current_dir_path,
                    requires_gpu=True
                )
            else:
                if config['MODEL'] == "nerfacto":
                    # Geometry
                    args = [
                        "poisson",
                        "--load-config", "outputs/train/nerfacto/train-stage-1/config.yml",
                        "--output-dir", output_path
                    ]
                    pipeline.create_component(
                        name="Nerfstudio-Export",
                        comp_type=ComponentType.exporter,
                        comp_environ=ComponentEnvironment.executable,
                        command="ns-export", args=args, requires_gpu=True
                    )
                    # Texture
                    args = [
                        "--load-config", "outputs/train/nerfacto/train-stage-1/config.yml",
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
                        "--load_config", "outputs/unnamed/splatfacto-w-light/train-stage-1/config.yml",
                        "--output_dir", output_path,
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
                    args = [
                        "gaussian-splat",
                        "--load-config", "outputs/unnamed/splatfacto/train-stage-1/config.yml",
                        "--output-dir", output_path
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
        pipeline.report_error(780, error_message)

    ##################################
    # TRANSFORM:
    # ROTATE SPLAT - PRE SPZ (SPZ MODULE HAS BUILT IN ROTATION around X-Y)
    ##################################
    try:
        # Apply pre-rotation if configured
        if str(config['ROTATE_SPLAT']).lower() == "true":
            # Apply the original pre-rotation values to work with the SPZ converter
            args = [
                "-i", os.path.join(output_path, "splat.ply"),
                "--rotations", "x:270,y:180,z:0"
            ]
            pipeline.create_component(
                name="Rotation-Pre-SPZ",
                comp_type=ComponentType.transform,
                comp_environ=ComponentEnvironment.python,
                command="training/rotate_splat_simple.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue rotating splat before SPZ conversion: {e}"
        pipeline.report_error(781, error_message)

    ##################################
    # EXPORT:
    # CREATE SPZ EXPORT COMPONENT
    ##################################
    try:
        if config['MODEL'] != "nerfacto":
            args = [
                os.path.join(output_path, 'splat.ply')
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
        pipeline.report_error(783, error_message)

    ##################################
    # TRANSFORM:
    # ROTATE SPLAT POST SPZ
    ##################################
    try:
        # Apply pre-rotation if configured
        if str(config['ROTATE_SPLAT']).lower() == "true":
            # Apply the original pre-rotation values to work with the SPZ converter
            args = [
                "-i", os.path.join(output_path, "splat.ply"),
                "--rotations", "x:180,y:180,z:0"
            ]
            pipeline.create_component(
                name="Rotation-Post-SPZ",
                comp_type=ComponentType.transform,
                comp_environ=ComponentEnvironment.python,
                command="training/rotate_splat_simple.py",
                args=args,
                cwd=current_dir_path,
                requires_gpu=False
            )
    except Exception as e:
        error_message = f"Issue rotating splat after SPZ conversion: {e}"
        pipeline.report_error(784, error_message)

    ##################################
    # EXPORT:
    # CREATE EXPORT SPZ COMPONENT TO S3
    ##################################
    try:
        if config['GENERATE_SPLAT'].lower() == "true":
            args = ["s3", "cp"]
            if config['MODEL'] != "nerfacto":
                args.extend([
                    os.path.join(output_path, "splat.spz"),
                    f"{config['S3_OUTPUT']}/{config['UUID']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.spz"
                ])
            pipeline.create_component(
                name="S3-Export1",
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
        pipeline.report_error(785, error_message)

    ##################################
    # EXPORT:
    # CREATE EXPORT SPLAT COMPONENT TO S3
    ##################################
    try:
        if config['GENERATE_SPLAT'].lower() == "true":
            args = ["s3", "cp"]
            if config['MODEL'] == "nerfacto":
                args.extend([
                    os.path.join(output_path, "textured", f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.glb"),
                    f"{config['S3_OUTPUT']}/{config['UUID']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.glb"
                ])
            else:
                args.extend([
                    os.path.join(output_path, "splat.ply"),
                    f"{config['S3_OUTPUT']}/{config['UUID']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.ply"
                ])

            pipeline.create_component(
                name="S3-Export2",
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
        pipeline.report_error(785, error_message)

    ##################################
    # RUN THE PIPELINE
    ##################################
    try:
        pipeline.session.status = Status.RUNNING
        log.info(f"Pipeline status changed to {pipeline.session.status}")
        start_time = int(time.time())
        image_proc_time = None
        sfm_time = None
        training_time = None
        for i in range(0, pipeline.config.num_components, 1):
            component = pipeline.components[i]
            log.info(f"Running component: {component.name}")
            match component.name:
                case "VideoToImages":
                    # VIDEO-TO-IMAGES CONDITIONAL COMPONENT
                    if (VIDEO is False and str(config['USE_POSE_PRIOR_COLMAP_MODEL_FILES']).lower() == 'true') or \
                        (VIDEO is False and str(config['USE_POSE_PRIOR_TRANSFORM_JSON']).lower() == 'true'):
                        continue
                    else:
                        if (VIDEO is True and str(config['USE_POSE_PRIOR_COLMAP_MODEL_FILES']).lower() == 'false') or \
                            (VIDEO is True and str(config['USE_POSE_PRIOR_TRANSFORM_JSON']).lower() == 'false'):
                            pipeline.run_component(i)
                        else:
                            # unzip archive of images into /images directory
                            temp_path = os.path.join(config['DATASET_PATH'], 'temp')
                            with zipfile.ZipFile(input_file_path,"r") as zip_ref:
                                zip_ref.extractall(temp_path)
                            temp_dir_input = os.listdir(temp_path)[0]
                            if os.path.isdir(os.path.join(temp_path, temp_dir_input)): # Archive has a directory
                                log.info("Moving directory from {temp_path} to {temp_dir_input}")
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
                            filenames = os.listdir(image_path)
                            for filename in filenames:
                                filepath = os.path.join(image_path, filename)
                                logging.info(f"Resizing image: {filepath}")
                                resize_to_4k(filepath)
                            head, first_file_ext = os.path.splitext(filenames[0])
                            if first_file_ext == ".png" or \
                                first_file_ext == ".jpeg" or first_file_ext == ".jpg":
                                logging.info("Found images in archive.")
                            else:
                                pipeline.report_error(
                                    790,
                                    """The archive doesn't contain supported image files
                                    .jpg, .jpeg, or .png"""
                                )
                case "RemoveHumanSubject":
                    # REMOVE HUMAN SUBJECT CONDITIONAL COMPONENT
                    # Run Component
                    pipeline.run_component(i)
                    # Rename the masked image directory to images
                    shutil.rmtree(image_path)
                    os.rename(f"{config['DATASET_PATH']}/masked_images", image_path)
                    log.info("All images successfully processed with human subject remover")
                case "ColmapSfM-Feature-Extractor":
                    # COLMAP FEATURE EXTRACTOR CONDITIONAL COMPONENT
                    current_time = int(time.time())
                    image_proc_time = current_time - start_time
                    log.info(f"Time to prExtractorocess images: {image_proc_time}s")
                    # If using pose prior, use the intrinsics from the txt file
                    if config['USE_POSE_PRIOR_COLMAP_MODEL_FILES'].lower() == "true":
                        camera_params = read_camera_params_from_file(os.path.join(sparse_path, "0", "cameras.txt"))
                        component.args.extend([
                            "--ImageReader.camera_model", camera_params['model'],
                            "--ImageReader.camera_params", camera_params['params_str']
                        ])
                    # Only use GPU if not too many images
                    num_images = len(os.listdir(image_path))
                    if num_images > 500:
                        use_gpu = "0"
                    else:
                        use_gpu = "1"
                    component.args.extend([
                        "--SiftExtraction.use_gpu", use_gpu
                    ])
                    pipeline.run_component(i)
                case "ColmapSfM-Feature-Matcher":
                    # Only use GPU if not too many images
                    num_images = len(os.listdir(image_path))
                    if num_images > 500:
                        use_gpu = "0"
                    else:
                        use_gpu = "1"
                    component.args.extend([
                        "--SiftMatching.use_gpu", use_gpu
                    ])
                    pipeline.run_component(i)
                case "Train-Stage1":
                    # TRAIN-STAGE1 CONDITIONAL COMPONENT
                    if (str(config['MODEL']).lower() != "3dgut" and str(config['MODEL']).lower() != "3dgrt"):
                        if (str(config['SFM_SOFTWARE_NAME']).lower() == "colmap" or \
                            str(config['SFM_SOFTWARE_NAME']).lower() == "glomap"):
                            # Move the sparse point cloud from sparse/0/* to colmap/sparse/*
                            log.info('Running Training...')
                            current_time = int(time.time())
                            print(f"Current time: {current_time}")
                            print(f"Image Proc time: {image_proc_time}")
                            print(f"Start time: {start_time}")
                            sfm_time = current_time - image_proc_time
                            log.info(f"Time for SfM: {sfm_time}s")
                            sparse_path_out = os.path.join(config['DATASET_PATH'], "colmap", "sparse")
                            shutil.copytree(sparse_path, sparse_path_out)
                            # Set the image cache to disk if there are a lot of images to prevent OOM
                            num_images = len(os.listdir(image_path))
                            if num_images > 500:
                                index = component.args.index("colmap")
                                if index != -1:
                                    component.args.insert(index, "disk")
                                    component.args.insert(index, "--pipeline.datamanager.cache-images")
                    else:
                        # Preprocess 3dgrut images if they have a mask
                        if str(config['MODEL']).lower() == "3dgut" or \
                            str(config['MODEL']).lower() == "3dgrt":
                            print("Using 3D-GRUT")
                            if has_alpha_channel(os.path.join(image_path, os.listdir(image_path)[0])):
                                print("Has alpha channel...")
                                process_images(image_path)
                            else:
                                print("No alpha channel...")
                    pipeline.run_component(i)
                case "Nerfstudio-Export":
                    # NERFSTUDIO EXPORT CONDITIONAL COMPONENT
                    pipeline.run_component(i)
                    current_time = int(time.time())
                    training_time = current_time - sfm_time
                    log.info(f"Time to train: {training_time}s")
                case "Nerfstudio-Export-Nerfacto":
                    # NERFSTUDIO NERFACTO EXPORT CONDITIONAL COMPONENT
                    pipeline.run_component(i)
                    obj_to_glb(
                        os.path.join(output_path, "textured", "mesh.obj"),
                        os.path.join(output_path, "textured", f"{str(os.path.splitext(config['FILENAME'])[0]).lower()}.glb")
                    )
                case "Spz-Export":
                    # If using 3dgrut, move the output splat over to where we expect it
                    if str(config['MODEL']).lower() == "3dgut" or \
                        str(config['MODEL']).lower() == "3dgrt":
                        root_exp_dir = os.path.join(config['DATASET_PATH'], "3dgrut", "runs", str(config['UUID']))
                        exp_dir = os.listdir(root_exp_dir)[0]
                        shutil.move(os.path.join(root_exp_dir, exp_dir, "export_last.ply"), os.path.join(output_path, "splat.ply"))
                    pipeline.run_component(i)
                case "S3-Export2":
                    # S3 UPLOAD CONDITIONAL COMPONENT
                    log.info("Uploading asset to S3")
                    pipeline.run_component(i)
                    # Copy result over to where SM expects it
                    shutil.copytree(config['DATASET_PATH'], "/opt/ml/model/dataset")
                    log.info(f"Successful pipeline result generation located at \
                            {config['S3_OUTPUT']}/{str(os.path.splitext(config['FILENAME'])[0]).lower()}.*")
                case _: # Default case, run Component
                    pipeline.run_component(i)
        pipeline.session.status = Status.STOP
        log.info(f"Pipeline status changed to {pipeline.session.status}")
        current_time = int(time.time())
        total_time = current_time - start_time
        log.info(f"Total Time: {total_time}s")
    except Exception as e:
        error_message = f"Error running the pipeline: {e}"
        pipeline.report_error(795, error_message)
