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

""" Wrapper to download models (SAM2, U2NET, Stable Diffusion XL, COLMAP vocab tree, PyTorch models), archive them, and push to S3 for container """

import os
import tempfile
import shutil
import boto3
import tarfile
import json
import time
# import subprocess  # Not needed for direct downloads
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# CloudFormation custom resource response function
def send_response(event, context, response_status, response_data, physical_resource_id=None):
    response_body = {
        'Status': response_status,
        'Reason': f'See the details in CloudWatch Log Stream: {context.log_stream_name}',
        'PhysicalResourceId': physical_resource_id or context.log_stream_name,
        'StackId': event['StackId'],
        'RequestId': event['RequestId'],
        'LogicalResourceId': event['LogicalResourceId'],
        'Data': response_data
    }
    
    response_body_json = json.dumps(response_body)
    
    print(f"Response body: {response_body_json}")
    
    headers = {
        'content-type': '',
        'content-length': str(len(response_body_json))
    }
    
    try:
        # Validate URL scheme and format for security
        response_url = event.get('ResponseURL', '')
        if not response_url or not isinstance(response_url, str):
            raise ValueError("Invalid or missing ResponseURL")
        
        parsed_url = urlparse(response_url)
        if parsed_url.scheme not in ('http', 'https'):
            raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme}. Only HTTP and HTTPS are allowed.")
        
        # Additional validation to prevent malicious URLs
        if not parsed_url.netloc:
            raise ValueError("Invalid URL: missing network location")
        
        req = Request(response_url, data=response_body_json.encode('utf-8'), headers=headers, method='PUT')
        with urlopen(req, timeout=30) as response:  # nosemgrep: dynamic-urllib-use-detected
            print(f"Status code: {response.status}")
            print(f"Status message: {response.reason}")
        return True
    except Exception as e:
        print(f"Error sending response: {str(e)}")
        return False

# Function to download file with progress reporting
def download_with_progress(url, destination):
    print(f"Starting download from {url}")
    start_time = time.time()
    
    # Validate URL scheme for security
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ('http', 'https'):
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme}. Only HTTP and HTTPS are allowed.")
    
    # Create headers to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Use urllib for file download - URL validated above
    req = Request(url, headers=headers)
    with urlopen(req, timeout=30) as response:  # nosemgrep: dynamic-urllib-use-detected
        file_size = int(response.headers.get('Content-Length', 0))
        print(f"File size: {file_size / (1024 * 1024):.2f} MB")
        
        # Download the file with progress reporting
        downloaded = 0
        last_report = 0
        block_size = 8192 * 64  # 512KB blocks
        
        with open(destination, 'wb') as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                downloaded += len(chunk)
                f.write(chunk)
                
                # Report progress every 5%
                if file_size > 0:
                    progress = downloaded / file_size * 100
                    if progress - last_report >= 5:
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed / (1024 * 1024) if elapsed > 0 else 0
                        print(f"Downloaded {downloaded / (1024 * 1024):.2f} MB of {file_size / (1024 * 1024):.2f} MB ({progress:.1f}%) - {speed:.2f} MB/s")
                        last_report = progress // 5 * 5
    
    elapsed = time.time() - start_time
    print(f"Download completed in {elapsed:.2f} seconds")
    return destination

def download_u2net_models(temp_dir):
    u2net_dir = os.path.join(temp_dir, '.u2net')
    os.makedirs(u2net_dir, exist_ok=True)
    
    # Download U2NET model parts and combine
    parts = ['u2aa', 'u2ab', 'u2ac', 'u2ad']
    part_files = []
    
    for part in parts:
        url = f"https://github.com/nadermx/backgroundremover/raw/main/models/{part}"
        part_path = os.path.join(u2net_dir, f"u2net.pth.{part}")
        download_with_progress(url, part_path)
        part_files.append(part_path)
    
    # Combine parts
    u2net_path = os.path.join(u2net_dir, 'u2net.pth')
    with open(u2net_path, 'wb') as outfile:
        for part_file in part_files:
            with open(part_file, 'rb') as infile:
                outfile.write(infile.read())
            os.remove(part_file)
    
    # Download u2netp model
    u2netp_url = "https://github.com/nadermx/backgroundremover/raw/main/models/u2netp.pth"
    u2netp_path = os.path.join(u2net_dir, 'u2netp.pth')
    download_with_progress(u2netp_url, u2netp_path)
    
    # Download U2NET human segmentation model parts and combine
    human_parts = ['u2haa', 'u2hab', 'u2hac', 'u2had']
    human_part_files = []
    
    for part in human_parts:
        url = f"https://github.com/nadermx/backgroundremover/raw/main/models/{part}"
        part_path = os.path.join(u2net_dir, f"u2net_human_seg.pth.{part}")
        download_with_progress(url, part_path)
        human_part_files.append(part_path)
    
    # Combine human model parts
    u2net_human_path = os.path.join(u2net_dir, 'u2net_human_seg.pth')
    with open(u2net_human_path, 'wb') as outfile:
        for part_file in human_part_files:
            with open(part_file, 'rb') as infile:
                outfile.write(infile.read())
            os.remove(part_file)
    
    return u2net_dir

def handler(event, context):
    print(f"Received event: {json.dumps(event)}")
    response_data = {}
    
    # Handle CloudFormation custom resource events
    if 'RequestType' in event:
        if event['RequestType'] == 'Delete':
            send_response(event, context, 'SUCCESS', response_data)
            return
    # For direct Lambda invocation (Terraform), proceed with model download
    
    try:
        # Get bucket name from event properties (CloudFormation) or environment variables (Terraform)
        s3_bucket = None
        if 'ResourceProperties' in event:
            s3_bucket = event['ResourceProperties'].get('BucketName')
        if not s3_bucket:
            s3_bucket = os.environ.get('S3_BUCKET_NAME')
        if not s3_bucket:
            raise Exception("No S3 bucket name provided in event properties or environment variables")
        
        print(f"Using S3 bucket: {s3_bucket}")
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {temp_dir}")
        
        # Download SAM2 model
        sam2_filename = "sam2.1_hiera_large.pt"
        sam2_path = os.path.join(temp_dir, sam2_filename)
        sam2_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        print("Downloading SAM2 model")
        download_with_progress(sam2_url, sam2_path)
        
        # Download U2NET models
        print("Downloading U2NET models")
        u2net_dir = download_u2net_models(temp_dir)
        
        # Download COLMAP vocab tree
        vocab_filename = "vocab_tree_flickr100K_words32K.bin"
        vocab_path = os.path.join(temp_dir, vocab_filename)
        vocab_url = "https://github.com/ZachMckennedyFWig/ColmapFaissVocabTrees/raw/main/vocab_tree_flickr100K_words32K.bin"
        print("Downloading COLMAP vocab tree")
        download_with_progress(vocab_url, vocab_path)
        
        # Download PyTorch models
        torch_models_dir = os.path.join(temp_dir, '.cache', 'torch', 'hub', 'checkpoints')
        os.makedirs(torch_models_dir, exist_ok=True)
        
        torch_models = [
            ("https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth", "mobilenet_v3_large-8738ca79.pth"),
            ("https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth", "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"),
            ("https://download.pytorch.org/models/vgg16-397923af.pth", "vgg16-397923af.pth"),
            ("https://download.pytorch.org/models/alexnet-owt-7be5be79.pth", "alexnet-owt-7be5be79.pth")
        ]
        
        for url, filename in torch_models:
            model_path = os.path.join(torch_models_dir, filename)
            print(f"Downloading {filename}")
            download_with_progress(url, model_path)
        
        # Create placeholder for Stable Diffusion XL - will be downloaded at runtime
        print("Creating Stable Diffusion XL placeholder")
        sd_model_dir = os.path.join(temp_dir, 'stable-diffusion-xl-base-1.0')
        os.makedirs(sd_model_dir, exist_ok=True)
        
        # Create a placeholder file to indicate SDXL should be downloaded at runtime
        with open(os.path.join(sd_model_dir, 'download_at_runtime.txt'), 'w') as f:
            f.write('This model will be downloaded at runtime to avoid Lambda storage limits')
        
        # Create tar.gz archive
        archive_path = os.path.join(temp_dir, 'models.tar.gz')
        print(f"Creating archive at {archive_path}")
        
        with tarfile.open(archive_path, "w:gz") as tar:
            print(f"Adding {sam2_path} to archive")
            tar.add(sam2_path, arcname=sam2_filename)
            
            print(f"Adding U2NET models to archive")
            tar.add(u2net_dir, arcname='.u2net')
            
            print(f"Adding vocab tree to archive")
            tar.add(vocab_path, arcname=vocab_filename)
            
            print(f"Adding PyTorch models to archive")
            tar.add(os.path.join(temp_dir, '.cache'), arcname='.cache')
            
            print(f"Adding Stable Diffusion XL placeholder to archive")
            tar.add(sd_model_dir, arcname='stable-diffusion-xl-base-1.0')
        
        # Verify the archive was created
        if os.path.exists(archive_path):
            archive_size = os.path.getsize(archive_path)
            print(f"Created archive size: {archive_size / (1024 * 1024):.2f} MB")
            if archive_size < 1024 * 1024:  # Less than 1MB
                raise Exception(f"Created archive is too small: {archive_size} bytes")
        else:
            raise Exception("Archive was not created")
        
        # Upload only the archive to S3
        print(f"Uploading archive to s3://{s3_bucket}/models/models.tar.gz")
        s3_client = boto3.client('s3')
        
        # Ensure models directory exists
        try:
            s3_client.head_object(Bucket=s3_bucket, Key='models/')
        except:
            s3_client.put_object(Bucket=s3_bucket, Key='models/', Body=b'')
        
        s3_client.upload_file(archive_path, s3_bucket, 'models/models.tar.gz')
        print(f"Successfully uploaded models.tar.gz to S3")
        
        response_data['Message'] = 'Essential models archive uploaded successfully. SDXL will be downloaded at runtime.'
        response_data['ArchiveLocation'] = f's3://{s3_bucket}/models/models.tar.gz'
        print(f"SUCCESS: {response_data['Message']}")
        
        # Send CloudFormation response if this is a custom resource
        if 'RequestType' in event:
            send_response(event, context, 'SUCCESS', response_data)
        
        return response_data
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        response_data['Error'] = str(e)
        
        # Send CloudFormation response if this is a custom resource
        if 'RequestType' in event:
            send_response(event, context, 'FAILED', response_data)
        
        raise e
    finally:
        # Clean up
        if 'temp_dir' in locals():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Error cleaning up: {str(e)}")