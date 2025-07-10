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

""" Download the SAM2 model, archive it, and push to S3 for container """

import os
import tempfile
import shutil
import boto3
import tarfile
import json
import time
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

def download_with_progress(destination):
    # Use hardcoded HTTPS URL for security - prevents file:// scheme vulnerabilities
    # URL is not user-controllable to avoid arbitrary file access
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    print(f"Starting download from {url}")
    start_time = time.time()
    
    # Create headers to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Use urllib for file download - hardcoded HTTPS URL prevents file:// vulnerabilities
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

def handler(event, context):
    print(f"Received event: {json.dumps(event)}")
    
    try:
        s3_bucket = os.environ['S3_BUCKET_NAME']
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {temp_dir}")
        
        # Define model file path
        model_filename = "sam2.1_hiera_large.pt"
        model_path = os.path.join(temp_dir, model_filename)
        
        # Download the model file directly
        print("Downloading model")
        download_with_progress(model_path)
        
        # Verify the file was downloaded
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"Downloaded model file size: {file_size / (1024 * 1024):.2f} MB")
            if file_size < 1024 * 1024:  # Less than 1MB
                raise Exception(f"Downloaded file is too small: {file_size} bytes")
        else:
            raise Exception("Model file was not downloaded")
        
        # Create tar.gz archive
        archive_path = os.path.join(temp_dir, 'models.tar.gz')
        print(f"Creating archive at {archive_path}")
        
        with tarfile.open(archive_path, "w:gz") as tar:
            print(f"Adding {model_path} to archive")
            tar.add(model_path, arcname=model_filename)
        
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
        s3_client.upload_file(archive_path, s3_bucket, 'models/models.tar.gz')
        
        return {
            'statusCode': 200,
            'body': json.dumps('Models archive uploaded successfully')
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }
    finally:
        # Clean up
        if 'temp_dir' in locals():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Error cleaning up: {str(e)}")
