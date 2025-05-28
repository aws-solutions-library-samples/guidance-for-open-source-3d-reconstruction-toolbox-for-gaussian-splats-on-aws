import os
import tempfile
import shutil
import boto3
import urllib.request
import tarfile
import json
import urllib.parse
import time

def download_with_progress(url, destination):
    print(f"Starting download from {url}")
    start_time = time.time()
    
    # Create a request with a user agent to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    req = urllib.request.Request(url, headers=headers)
    
    # Open the URL and get file size
    with urllib.request.urlopen(req) as response:
        file_size = int(response.info().get('Content-Length', 0))
        print(f"File size: {file_size / (1024 * 1024):.2f} MB")
        
        # Download the file with progress reporting
        downloaded = 0
        last_report = 0
        block_size = 8192 * 64  # 512KB blocks
        
        with open(destination, 'wb') as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                
                downloaded += len(buffer)
                f.write(buffer)
                
                # Report progress every 5%
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
        
        # Define model URL and file path
        model_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        model_filename = "sam2.1_hiera_large.pt"
        model_path = os.path.join(temp_dir, model_filename)
        
        # Download the model file directly
        print(f"Downloading model from {model_url}")
        download_with_progress(model_url, model_path)
        
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
