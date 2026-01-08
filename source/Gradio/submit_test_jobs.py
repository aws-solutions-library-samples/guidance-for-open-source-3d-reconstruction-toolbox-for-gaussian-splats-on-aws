#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Test automation script to submit multiple 3D reconstruction jobs

import boto3
import json
import argparse
import os
from pathlib import Path
from datetime import datetime

def upload_input_files(s3, bucket_name, job_id, input_prefix, media_prefix, input_file):
    """Upload input media file to S3"""
    file_path = Path(input_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    s3_key = f"{input_prefix}/{job_id}/{media_prefix}/{file_path.name}"
    s3.upload_file(str(file_path), bucket_name, s3_key)
    return s3_key

def submit_test_jobs(base_config, test_variations, bucket_name, input_prefix, media_prefix, aws_region, dry_run=False):
    """Submit multiple test jobs with parameter variations"""
    
    if not dry_run:
        s3 = boto3.client('s3', region_name=aws_region)
    
    submitted_jobs = []
    
    for i, test_case in enumerate(test_variations):
        config = base_config.copy()
        config.update(test_case.get('params', {}))
        
        job_id = f"test-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{i:02d}"
        config['job_id'] = job_id
        
        if dry_run:
            print(f"[DRY RUN] Would submit {job_id}:")
            print(f"  Changes: {test_case.get('params', {})}")
            print(f"  Input: {test_case.get('input_file', 'N/A')}")
        else:
            # Upload input file if specified
            if 'input_file' in test_case:
                try:
                    s3_key = upload_input_files(s3, bucket_name, job_id, input_prefix, media_prefix, test_case['input_file'])
                    print(f"  ✓ Uploaded: {s3_key}")
                except Exception as e:
                    print(f"  ✗ Upload failed: {e}")
                    continue
            
            # Upload job.json (triggers workflow)
            s3.put_object(
                Bucket=bucket_name,
                Key=f"{input_prefix}/{job_id}/job.json",
                Body=json.dumps(config, indent=2)
            )
            print(f"✓ Submitted {job_id}: {test_case.get('params', {})}")
        
        submitted_jobs.append(job_id)
    
    return submitted_jobs

def main():
    parser = argparse.ArgumentParser(description='Submit test jobs for 3D reconstruction')
    parser.add_argument('--stack-id', default='uamxwvf1', help='Stack unique identifier')
    parser.add_argument('--prefix', default='workflow-input', help='S3 input prefix')
    parser.add_argument('--media-prefix', default='media-input', help='S3 media prefix')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--dry-run', action='store_true', help='Print jobs without submitting')
    args = parser.parse_args()
    
    bucket = f'3dgs-bucket-{args.stack_id}'
    
    # Base configuration matching SharedState defaults
    base_config = {
        "instance": "ml.g5.4xlarge",
        "use_spot_instance": "false",
        "sfm": "glomap",
        "model": "splatfacto",
        "max_steps": 15000,
        "max_images": 300,
        "filter_blurry": "true",
        "training_enable": "true",
        "sfm_enable": "true",
        "enhanced_feature": "false",
        "matching_method": "sequential",
        "use_colmap_model": "false",
        "use_transform_json": "false",
        "spherical_enable": "false",
        "enable_spz": "true",
        "enable_sog": "true",
        "enable_usdz": "true",
        "remove_bg": "false",
        "remove_objects": "false",
        "rotate_splat": "true",
        "crop_output_bounds": "false",
        "log_verbosity": "info"
    }
    
    # Test variations - customize these for your tests
    # Each test case has 'params' (config overrides) and optional 'input_file' (local path)
    test_variations = [
        # Test 1: Different SfM method
        {
            "params": {"sfm": "colmap", "model": "splatfacto"},
            "input_file": "test_data/video1.mp4"  # Optional: path to input file
        },
        
        # Test 2: Larger instance with more steps
        {
            "params": {"instance": "ml.g5.8xlarge", "max_steps": 20000},
            "input_file": "test_data/video2.mp4"
        },
        
        # Test 3: Spot instance
        {
            "params": {"use_spot_instance": "true"},
            "input_file": "test_data/images.zip"
        },
        
        # Test 4: Different model
        {
            "params": {"model": "splatfacto-big", "max_steps": 20000},
            "input_file": "test_data/video3.mp4"
        },
        
        # Test 5: Background removal
        {
            "params": {"remove_bg": "true"},
            "input_file": "test_data/object.mp4"
        },
        
        # Test 6: More images, no blur filter
        {
            "params": {"max_images": 500, "filter_blurry": "false"},
            "input_file": "test_data/video4.mp4"
        },
        
        # Test 7: Spherical images
        {
            "params": {"spherical_enable": "true"},
            "input_file": "test_data/360_video.mp4"
        },
        
        # Test 8: Minimal outputs (no input file needed if already in S3)
        {
            "params": {"enable_spz": "false", "enable_sog": "false", "enable_usdz": "false"}
        }
    ]
    
    print(f"{'='*60}")
    print(f"3D Reconstruction Test Job Submission")
    print(f"{'='*60}")
    print(f"Stack ID: {args.stack_id}")
    print(f"Bucket: {bucket}")
    print(f"Input Prefix: {args.prefix}")
    print(f"Media Prefix: {args.media_prefix}")
    print(f"Region: {args.region}")
    print(f"Jobs to submit: {len(test_variations)}")
    print(f"{'='*60}\n")
    
    jobs = submit_test_jobs(
        base_config,
        test_variations,
        bucket,
        args.prefix,
        args.media_prefix,
        args.region,
        args.dry_run
    )
    
    print(f"\n{'='*60}")
    if args.dry_run:
        print(f"DRY RUN: {len(jobs)} jobs would be submitted")
    else:
        print(f"SUCCESS: {len(jobs)} jobs submitted")
        print(f"\nJob IDs:")
        for job_id in jobs:
            print(f"  - {job_id}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
