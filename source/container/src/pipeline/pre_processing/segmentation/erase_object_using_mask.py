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
#

"""
Object Eraser Script using AttentiveEraser
https://github.com/Anonym0u3/AttentiveEraser/blob/master/notebook/Attentive_Eraser_XL.ipynb

This script processes a directory of images to erase objects using corresponding mask files
and the AttentiveEraser diffusion model. It supports flexible mask naming patterns and
can upscale results back to original image dimensions.

Features:
- Batch processing of image directories
- Flexible mask file naming (exact match, _mask suffix, .png extension variations)
- Dimension validation between images and masks
- Processing at 1024x1024 for AttentiveEraser compatibility with upscaling to original size
- Optional difference image generation to show changed pixels
- GPU/CPU support with proper device handling
- Configurable model paths and pipeline parameters

Usage:
    python erase_object_using_mask.py -id /path/to/images -md /path/to/masks [options]

Required Arguments:
    -id, --input_dir: Directory containing input images
    -md, --mask_dir: Directory containing mask images

Optional Arguments:
    -od, --output_dir: Output directory (defaults to input_dir)
    -mp, --model_path: Diffusion model path
    -pp, --pipeline_path: AttentiveEraser pipeline path
    -s, --seed: Random seed for reproducibility
    -gpu, --use_gpu: Enable GPU processing (default: true)
    -diff, --save_diff: Save difference images (default: false)
    -log, --log_level: Logging level (info/debug/error)
"""

import os
import sys

# Disable xformers before importing any diffusers/torch modules
os.environ['XFORMERS_DISABLED'] = '1'
os.environ['XFORMERS_MORE_DETAILS'] = '0'

import torch
import argparse
import logging
import cv2
import numpy as np
from PIL import ImageFilter, Image
from rich.logging import RichHandler
from diffusers import DDIMScheduler, DiffusionPipeline
import torch.nn.functional as Functional
from torchvision.utils import save_image
from diffusers.utils import load_image
from torchvision.transforms.functional import to_tensor, gaussian_blur, to_pil_image

def preprocess_image(image_path, width, height, device, dtype):
    image = to_tensor((load_image(image_path)))
    image = image.unsqueeze_(0).float() * 2 - 1  # [0,1] --> [-1,1]
    if image.shape[1] != 3:
        image = image.expand(-1, 3, -1, -1)
    image = Functional.interpolate(image, (1024, 1024))
    image = image.to(dtype).to(device)
    return image

def preprocess_mask(mask_path, width, height, device, dtype, save_refined=False, refined_dir=None, filename=None):
    # Load mask - extract alpha channel from RGBA (remove_background output)
    mask_img = Image.open(mask_path)
    if mask_img.mode == 'RGBA':
        # Alpha channel is the clean human mask from rembg
        mask_np = np.array(mask_img)[:, :, 3].astype(np.float32) / 255.0
    else:
        mask_np = np.array(mask_img.convert('L')).astype(np.float32) / 255.0

    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), 'L')
    mask = to_tensor(mask_pil).unsqueeze_(0).float()
    mask = Functional.interpolate(mask, (1024, 1024))
    mask = gaussian_blur(mask, kernel_size=(3, 3))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = mask.to(dtype).to(device)

    if save_refined and refined_dir and filename:
        refined_path = os.path.join(refined_dir, filename)
        # Keep only the largest blob for a clean single-region refined mask
        mask_np = mask.squeeze().cpu().float().numpy()
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        binary = (mask_uint8 > 10).astype(np.uint8)
        n_labels, labels = cv2.connectedComponents(binary)
        if n_labels > 1:
            largest = max(range(1, n_labels), key=lambda i: np.sum(labels == i))
            single_blob = np.where(labels == largest, 255, 0).astype(np.uint8)
            # Scale dilation with blob size so bleed is proportional to human size
            blob_size = np.sum(labels == largest)
            blob_radius = int(np.sqrt(blob_size / np.pi))  # approximate radius
            dilation_px = max(3, min(20, blob_radius // 10))  # 1/10th of radius, clamped 3-20px
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px*2+1, dilation_px*2+1))
            single_blob = cv2.dilate(single_blob, kernel, iterations=1)
            cv2.imwrite(refined_path, single_blob)
        else:
            cv2.imwrite(refined_path, mask_uint8)

    return mask

def make_redder(img, mask, increase_factor=0.4):
    img_redder = img.clone()
    mask_expanded = mask.expand_as(img)
    img_redder[0][mask_expanded[0] == 1] = torch.clamp(img_redder[0][mask_expanded[0] == 1] + increase_factor, 0, 1)
    return img_redder

def is_mask_black(mask_path, threshold=0.001):
    """Check if mask is effectively black before any processing"""
    try:
        mask_img = Image.open(mask_path)
        
        # Handle different mask formats
        if mask_img.mode == 'RGBA':
            # For RGBA masks, check alpha channel first
            mask_array = np.array(mask_img)
            if mask_array.shape[2] == 4:  # Has alpha channel
                alpha = mask_array[:, :, 3]
                if np.max(alpha) > 0:
                    mask_values = alpha / 255.0
                else:
                    mask_values = np.array(mask_img.convert('L')) / 255.0
            else:
                mask_values = np.array(mask_img.convert('L')) / 255.0
        else:
            mask_values = np.array(mask_img.convert('L')) / 255.0
        
        # Simulate the blob detection logic from preprocess_mask to predict if mask will become black
        max_value = np.max(mask_values)
        non_zero_pixels = np.count_nonzero(mask_values > threshold)
        total_pixels = mask_values.size
        
        # First check: completely empty mask
        if max_value < threshold or non_zero_pixels == 0:
            return True, max_value, non_zero_pixels, total_pixels
        
        # Second check: simulate blob detection filtering
        # Apply similar thresholding as in preprocess_mask
        binary_mask = (mask_values > 0.05).astype(np.uint8)
        
        # Use OpenCV to find connected components
        num_labels, labels = cv2.connectedComponents(binary_mask)
        
        if num_labels <= 1:  # No components found
            return True, max_value, non_zero_pixels, total_pixels
        
        # Check if any blob meets the minimum size requirements (similar to preprocess_mask)
        min_area_percentage = 0.003  # 0.3% minimum for perspective crop blobs
        min_required_pixels = int(total_pixels * min_area_percentage)
        
        valid_blobs = 0
        for i in range(1, num_labels):
            component_size = np.sum(labels == i)
            if component_size >= min_required_pixels:
                valid_blobs += 1
        
        # If no valid blobs, mask will become black after preprocessing
        if valid_blobs == 0:
            return True, max_value, non_zero_pixels, total_pixels
        
        # Additional check: very sparse masks that will likely be filtered out
        mask_density = non_zero_pixels / total_pixels
        if mask_density < 0.0001:  # Less than 0.01% of pixels are non-zero
            return True, max_value, non_zero_pixels, total_pixels
        
        return False, max_value, non_zero_pixels, total_pixels
        
    except Exception as e:
        print(f"Error checking mask {mask_path}: {e}")
        return True, 0, 0, 0  # Assume black if error

def process_single_image(image_file, mask_path, args, pipeline, device, dtype, refined_mask_dir, log):
    input_path = os.path.join(args.input_dir, image_file)
    output_path = os.path.join(args.output_dir, image_file)
    base_name = os.path.splitext(image_file)[0]
    ext = os.path.splitext(image_file)[1]

    try:
        img = Image.open(input_path)
        mask_img = Image.open(mask_path)
        orig_width, orig_height = img.size
        mask_width, mask_height = mask_img.size

        if orig_width != mask_width or orig_height != mask_height:
            log.warning(f"Dimension mismatch for {image_file}: image({orig_width}x{orig_height}) vs mask({mask_width}x{mask_height}), skipping")
            return

        width, height = 1024, 1024
        seed = int(args.seed)
        generator = torch.Generator('cuda').manual_seed(seed) if str(args.use_gpu).lower() == "true" else torch.Generator().manual_seed(seed)
        source_image = preprocess_image(input_path, width, height, device, dtype)
        
        # Always create the refined mask
        mask = preprocess_mask(mask_path, width, height, device, dtype, save_refined=True, refined_dir=refined_mask_dir, filename=os.path.basename(mask_path))

        # Check if mask is black after preprocessing - skip AttentiveEraser but keep refined mask
        mask_max = torch.max(mask).item() if mask is not None else 0
        if mask_max < 0.001:
            log.info(f"Black mask for {image_file} (filtered out after preprocessing), copying original")
            Image.open(input_path).save(output_path)
            return

        # Process with AttentiveEraser and save result directly
        result_image = run_attentive_eraser(source_image, mask, args, pipeline, generator, height, width)
        
        # Resize back to original dimensions if needed and save
        if orig_width != 1024 or orig_height != 1024:
            result_pil = to_pil_image(result_image.clamp(0, 1))
            result_pil = result_pil.resize((orig_width, orig_height), Image.LANCZOS)
            result_pil.save(output_path)
        else:
            save_image(result_image.clamp(0, 1), output_path)
        
        # Save diff if requested
        if str(args.save_diff).lower() == "true":
            save_difference_image(input_path, final_image, args.output_dir, base_name, ext, orig_width, orig_height, device)
        
        log.info(f"Processed {image_file}")
        
    except Exception as e:
        import traceback
        log.warning(f"Error processing {image_file}: {e}")
        log.debug(traceback.format_exc())

def run_attentive_eraser(source_image, mask, args, pipeline, generator, height, width):
    """Run AttentiveEraser inference - SIP or DIP method"""
    if args.method == "DIP":
        # DIP: invert the source image first, then run with inverted latents
        strength = 0.8
        num_inference_steps = 50
        END_STEP = int(strength * num_inference_steps)
        start_code, latents_list = pipeline.invert(
            source_image,
            prompt="",
            guidance_scale=1,
            num_inference_steps=END_STEP,
            return_intermediates=True,
            generator=generator,
        )
        # Cast latents and ensure pipeline is back in fp16 after inversion
        pipe_dtype = next(pipeline.unet.parameters()).dtype
        start_code = start_code.to(pipe_dtype)
        latents_list = [l.to(pipe_dtype) for l in latents_list]
        pipeline.unet.to(pipe_dtype)  # ensure UNet is fp16 after inversion
        image = pipeline(
            prompt="",
            image=source_image,
            mask_image=mask,
            height=height,
            width=width,
            AAS=True,
            strength=strength,
            rm_guidance_scale=9,
            ss_steps=9,
            ss_scale=0.3,
            AAS_start_step=0,
            AAS_start_layer=34,
            AAS_end_layer=70,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=1,
            latents=start_code,
            x0_latents=latents_list[0],
            record_list=list(reversed(latents_list)),
        ).images[0]
    else:  # SIP
        image = pipeline(
            prompt="",
            image=source_image,
            mask_image=mask,
            height=height,
            width=width,
            AAS=True,
            strength=0.8,
            rm_guidance_scale=9,
            ss_steps=9,
            ss_scale=0.3,
            AAS_start_step=0,
            AAS_start_layer=34,
            AAS_end_layer=70,
            num_inference_steps=50,
            generator=generator,
            guidance_scale=1,
        ).images[0]
    # Convert PIL output to tensor [0,1]
    return to_tensor(image).to(source_image.device)

def apply_blending(source_image, mask, result_image, orig_width, orig_height, device):
    """Apply custom blending and resize to original dimensions"""
    img_tensor = (source_image * 0.5 + 0.5).squeeze(0)
    mask_red = mask.squeeze(0)
    img_redder = make_redder(img_tensor, mask_red)
    pil_mask = to_pil_image(mask.squeeze(0))
    mask_blurred = to_tensor(pil_mask).unsqueeze_(0).to(device=mask.device, dtype=mask.dtype)
    mask_f = 1 - (1 - mask) * (1 - mask_blurred)

    image_1 = result_image.unsqueeze(0)
    out_tile = mask_f * image_1 + (1 - mask_f) * (source_image * 0.5 + 0.5)

    if orig_width != 1024 or orig_height != 1024:
        out_tile = Functional.interpolate(out_tile, (orig_height, orig_width), mode='bicubic', align_corners=False)

    return out_tile

def save_difference_image(input_path, final_image, output_dir, base_name, ext, orig_width, orig_height, device):
    """Save difference image"""
    input_tensor = to_tensor(Image.open(input_path).convert('RGB')).unsqueeze(0).to(device)
    if orig_width != 1024 or orig_height != 1024:
        input_tensor = Functional.interpolate(input_tensor, (orig_height, orig_width), mode='bilinear', align_corners=False)
    diff_image = torch.abs(final_image - input_tensor)
    diff_path = os.path.join(output_dir, f"{base_name}_diff{ext}")
    save_image(diff_image.squeeze(0), diff_path)

def process_batch_images(batch_pairs, args, pipeline, device, dtype, refined_mask_dir, log):
    """Process images individually"""
    for image_file, mask_path in batch_pairs:
        process_single_image(image_file, mask_path, args, pipeline, device, dtype, refined_mask_dir, log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='object-eraser', description='Erase objects in an image given an mask')
    parser.add_argument('-id', '--input_dir', required=True, help='The directory containing images to perform the eraser on')
    parser.add_argument('-md', '--mask_dir', required=True, help='The directory containing image masks')
    parser.add_argument('-od', '--output_dir', required=False, default=None, help='The output directory for erased images')
    parser.add_argument('-mp', '--model_path', required=False, default="stable-diffusion-xl-base-1.0", help='The diffusion model path to use')
    parser.add_argument('-pp', '--pipeline_path', required=False, default=None, help='The pipeline path to use')
    parser.add_argument('-s', '--seed', required=False, default="123", help='The random seed to use')
    parser.add_argument('-gpu', '--use_gpu', required=False, default="true", help='Whether to use GPU or not for inference')
    parser.add_argument('-log', '--log_level', required=False, default='info', help='Level of logs to write to stdout')
    parser.add_argument('-diff', '--save_diff', required=False, default="false", help='Whether to save difference images')
    parser.add_argument('-method', '--method', required=False, default="DIP", choices=["SIP", "DIP"], help='Processing method: SIP or DIP (default: DIP)')
    parser.add_argument('-bs', '--batch_size', required=False, default=2, type=int, help='Batch size for GPU processing (default: 2)')
    args = parser.parse_args()

    level = logging.INFO
    if str(args.log_level).lower() == "debug":
        level = logging.DEBUG
    elif str(args.log_level).lower() == "error":
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(message)s", handlers=[RichHandler()])
    log = logging.getLogger()

    try:
        if args.output_dir is None:
            args.output_dir = args.input_dir
        else:
            os.makedirs(args.output_dir, exist_ok=True)

        # Create refined masks directory
        refined_mask_dir = os.path.join(args.output_dir, 'refined_masks')
        os.makedirs(refined_mask_dir, exist_ok=True)

        if str(args.use_gpu).lower() == "true":
            torch.cuda.set_device(0)

        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        os.chdir(parent_dir)

        if not os.path.isdir(args.input_dir):
            raise ValueError(f"Input directory {args.input_dir} does not exist")
        if not os.path.isdir(args.mask_dir):
            raise ValueError(f"Mask directory {args.mask_dir} does not exist")

        image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            raise ValueError(f"No valid image files found in {args.input_dir}")

        mask_files = [f for f in os.listdir(args.mask_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        log.info(f"Found {len(image_files)} image files")
        log.info(f"Found {len(mask_files)} mask files")
        log.info(f"Sample image files: {image_files[:5]}")
        log.info(f"Sample mask files: {mask_files[:5]}")
        log.info(f"Image directory: {args.input_dir}")
        log.info(f"Mask directory: {args.mask_dir}")

        dtype = torch.float16
        dtype_ = "fp16"
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

        # Set pipeline path based on method and provided argument
        if args.pipeline_path and os.path.exists(args.pipeline_path):
            pipeline_path = args.pipeline_path
        else:
            # Fallback to relative path construction
            # AttentiveEraser is cloned to /opt/ml/code/AttentiveEraser
            code_path = "/opt/ml/code"
            if args.method == "DIP":
                pipeline_path = os.path.join(code_path, "AttentiveEraser", "pipelines", "pipeline_stable_diffusion_xl_attentive_eraser_inversion.py")
            else:  # SIP
                pipeline_path = os.path.join(code_path, "AttentiveEraser", "pipelines", "pipeline_stable_diffusion_xl_attentive_eraser.py")

        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

        # Check if local model exists, otherwise use HuggingFace identifier
        model_path = args.model_path
        use_local_model = os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "model_index.json"))

        if not use_local_model:
            log.info(f"Local model not found at {model_path}, using HuggingFace identifier")
            model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            log.info(f"Using local model at {model_path}")
        
        log.info(f"Using pipeline: {pipeline_path}")

        # Load pipeline exactly like test.py - use custom_pipeline with local path
        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            custom_pipeline=pipeline_path,
            scheduler=scheduler,
            variant=dtype_,
            use_safetensors=True,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        pipeline.to(device)
        pipeline.enable_attention_slicing()
        # Patch GroupNorm to handle fp16/fp32 dtype mismatches (PyTorch 2.9 compatibility)
        import torch.nn as _nn
        _orig_gn = _nn.GroupNorm.forward
        def _patched_gn(self, x):
            if x.dtype != self.weight.dtype:
                x = x.to(self.weight.dtype)
            return _orig_gn(self, x)
        _nn.GroupNorm.forward = _patched_gn

        # Disable flash attention - causes NaN with fp16 on PyTorch >= 2.0 + CUDA 12.x
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        # Pre-filter images that have valid, non-empty masks
        valid_image_mask_pairs = []
        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]
            ext = os.path.splitext(image_file)[1]
            mask_patterns = [f"{base_name}_mask.png", f"output_{base_name}.png", image_file, f"{base_name}_mask{ext}", f"{base_name}.png"]

            mask_path = None
            for pattern in mask_patterns:
                potential_mask = os.path.join(args.mask_dir, pattern)
                if os.path.isfile(potential_mask):
                    mask_path = potential_mask
                    break

            if mask_path is None:
                log.warning(f"No mask found for {image_file}, copying original")
                # Copy original image to output
                input_path = os.path.join(args.input_dir, image_file)
                output_path = os.path.join(args.output_dir, image_file)
                Image.open(input_path).save(output_path)
                continue

            # Add all valid mask pairs - black mask detection happens during processing
            valid_image_mask_pairs.append((image_file, mask_path))

        if not valid_image_mask_pairs:
            log.info("No images with valid masks found, all originals copied")
            sys.exit(0)

        log.info(f"Found {len(valid_image_mask_pairs)} images with valid masks for processing")
        log.info(f"Using device: {device}")
        log.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

        log.info(f"Processing images individually")

        # Process images individually
        for image_file, mask_path in valid_image_mask_pairs:
            process_single_image(image_file, mask_path, args, pipeline, device, dtype, refined_mask_dir, log)

        log.info(f"Completed processing {len(valid_image_mask_pairs)} images with inference, {len(image_files) - len(valid_image_mask_pairs)} images copied as originals")

    except Exception as e:
        log.error(f"Error running object eraser script: {e}")
        raise e
