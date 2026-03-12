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
# A Gradio interface and server to submit and view splats

import os
import re
import html
import uuid
import json
import boto3
import time
import threading
import gradio as gr
import boto3.s3.transfer
from refine_splat import refine_splat

# In Gradio 6, subclassing creates a "custom component" that fails to load.
# Use gr.Column directly as an alias instead of subclassing.
Modal = gr.Column

_UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)

def _validate_uuid(value: str) -> str:
    """Validate that value is a well-formed UUID to prevent injection."""
    if not _UUID_RE.match(str(value)):
        raise ValueError(f"Invalid UUID format")
    return str(value)

print(f"Gradio Version: {gr.__version__}")

class SharedState:
    def __init__(self):
        # !!! vvv COMPLETE BELOW vvv !!!
        self.aws_region = "us-east-1"
        self.stack_unique_id = ""
        # !!! ^^^ COMPLETE ABOVE ^^^ !!!
        self.stack_unique_id = os.environ.get('STACK_UNIQUE_ID', self.stack_unique_id)
        self.s3_bucket = f"3dgs-bucket-{self.stack_unique_id}"
        self.ddb_table_name = f"3dgs-table-{self.stack_unique_id}"
        self.s3_input = "workflow-input"
        self.s3_output = "workflow-output"
        self.media_input = "media-input"
        self.instance = "ml.g5.4xlarge"
        self.use_spot_instance = "true"  # Default to Batch for faster startup
        self.sfm = "glomap"
        self.model = "splatfacto"
        self.faces = "[]"
        self.bg_model = "u2net"
        self.filter_blurry = "true"
        self.max_images = 300
        self.sfm_enable = "true"
        self.enhanced_feature = "false"
        self.matching_method = "sequential"
        self.use_colmap_model = "false"
        self.use_transform_json = "false"
        self.training_enable = "true"
        self.max_steps = 15000

        self.spherical_enable = "false"
        self.enable_spz = "true"
        self.enable_sog = "true"
        self.enable_usdz = "true"
        self.ply_coords = "rhyu"
        self.spz_coords = "rhyu"
        self.sog_coords = "rhyu"
        self.usdz_coords = "rhyu"
        self.remove_bg = "false"
        self.remove_objects = "false"
        self.object_removal_action = "erase"
        self.objects_to_remove = []
        self.source_coordinate = "arkit"
        self.pose_world_to_cam = "true"
        self.log_verbosity = "info"
        self.mask_threshold = 0.6
        self.model_3d = None
        self.crop_output_bounds = "false"
        self.crop_mode = "environment"
        self.clean_splat = "false"
        self.video_start_time = 0.0
        self.video_stop_time = None
        self.preserve_scene_scale = "false"
        self.isp_3d = "none"

# Create a singleton instance
shared_state = SharedState()

# Fetch current pricing on startup
print("Fetching current AWS pricing...")
# Pricing data will be set after function definition

def check_aws_credentials():
    try:
        s3 = boto3.client('s3')
        s3.list_buckets()
        print("AWS credentials are valid and working")
    except Exception as e:
        print(f"AWS credentials error: {str(e)}")

def refresh_aws_credentials(access_key, secret_key, session_token):
    try:
        # Set environment variables with new credentials
        os.environ['AWS_ACCESS_KEY_ID'] = access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
        os.environ['AWS_SESSION_TOKEN'] = session_token
        
        # Create new session using provided credentials
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token
        )
        
        # Test the credentials
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        
        # Update the default session
        boto3.setup_default_session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token
        )
        
        return f"AWS credentials refreshed successfully. Account: {identity['Account']}"
        
    except Exception as e:
        return f"Error refreshing credentials: {str(e)}"

def parse_aws_credentials(creds_string):
    try:
        # Initialize variables
        access_key = None
        secret_key = None
        session_token = None
        
        # Split the string by spaces
        parts = creds_string.strip().split()
        
        # Process each part
        for part in parts:
            #print(f"DEBUG: Processing part: {part[:15]}...")  # Show first 15 chars for debugging
            
            if part.startswith('$Env:AWS_ACCESS_KEY_ID='):
                access_key = part.split('=', 1)[1].strip('"').strip("'")
                #print("DEBUG: Found access key")
            elif part.startswith('$Env:AWS_SECRET_ACCESS_KEY='):
                secret_key = part.split('=', 1)[1].strip('"').strip("'")
                #print("DEBUG: Found secret key")
            elif part.startswith('$Env:AWS_SESSION_TOKEN='):
                session_token = part.split('=', 1)[1].strip('"').strip("'")
                #print("DEBUG: Found session token")
        
        #print("\nDEBUG: Final parsed values:")
        #print(f"Access Key present: {bool(access_key)}")
        #print(f"Secret Key present: {bool(secret_key)}")
        #print(f"Session Token present: {bool(session_token)}")
                
        # Verify all credentials are present and not empty
        if not all([
            access_key and access_key.strip(),
            secret_key and secret_key.strip(),
            session_token and session_token.strip()
        ]):
            missing = []
            if not (access_key and access_key.strip()):
                missing.append("AWS_ACCESS_KEY_ID")
            if not (secret_key and secret_key.strip()):
                missing.append("AWS_SECRET_ACCESS_KEY")
            if not (session_token and session_token.strip()):
                missing.append("AWS_SESSION_TOKEN")
            return f"Error: Missing or empty credentials: {', '.join(missing)}"
            
        # Call the refresh credentials function with parsed values
        return refresh_aws_credentials(access_key, secret_key, session_token)
        
    except Exception as e:
        print(f"DEBUG: Exception occurred: {str(e)}")
        return f"Error parsing credentials: {str(e)}"

def get_thumbnail_url(job_id):
    """Get thumbnail URL for a job if it exists - optimized version"""
    try:
        bucket_name = shared_state.s3_bucket
        output_prefix = shared_state.s3_output or "workflow-output"
        thumbnail_key = f"{output_prefix}/{job_id}/render_thumbnail.png"
        
        # Generate presigned URL without checking if file exists
        # The browser will handle 404s gracefully
        return generate_presigned_url(bucket_name, thumbnail_key)
    except Exception as e:
        print(f"Error getting thumbnail for {job_id}: {e}")
        return None

def refresh_s3_contents():
    """Refresh contents from DynamoDB and return grouped data by job ID - optimized"""
    try:
        refresh_id = time.time()
        print(f"\n=== Refreshing Job Contents from DynamoDB (ID: {refresh_id}) ===")
        dynamodb = boto3.resource('dynamodb', region_name=shared_state.aws_region)
        table = dynamodb.Table(shared_state.ddb_table_name)
        
        response = table.scan()
        
        print(f"DEBUG: Found {len(response.get('Items', []))} total items in DynamoDB")
        
        # Group files by job ID from DynamoDB
        jobs_dict = {}
        for item in response.get('Items', []):
            job_id = item.get('uuid', 'unknown')
            status = item.get('uuidStatus', 'unknown')
            
            # Make status check case-insensitive and flexible
            status_lower = str(status).lower()
            if status_lower not in ['complete', 'completed']:
                continue
            
            job_id = item['uuid']
            output_files = item.get('outputFiles', [])
            
            if not output_files:
                continue
            
            last_modified = item.get('endTimestamp', item.get('startTimestamp', ''))
            
            # Generate thumbnail URL without S3 API calls
            bucket_name = shared_state.s3_bucket
            output_prefix = shared_state.s3_output or "workflow-output"
            thumbnail_key = f"{output_prefix}/{job_id}/render_thumbnail.png"
            thumbnail_url = generate_presigned_url(bucket_name, thumbnail_key)
            
            jobs_dict[job_id] = {
                'files': [],
                'last_modified': last_modified,
                'thumbnail_url': thumbnail_url
            }
            
            for file_info in output_files:
                size_bytes = file_info.get('size', 0)
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes/1024:.1f} KB"
                else:
                    size_str = f"{size_bytes/(1024*1024):.1f} MB"
                
                jobs_dict[job_id]['files'].append({
                    'filename': file_info['filename'],
                    'size': size_str,
                    'last_modified': last_modified
                })
        
        # Convert to display format with grouped rows
        files_data = []
        for job_id, job_data in jobs_dict.items():
            # Create thumbnail HTML
            thumbnail_html = ""
            if job_data['thumbnail_url']:
                thumbnail_html = f'<a href="{job_data["thumbnail_url"]}" download="{job_id}_thumbnail.png" style="display:inline-block;"><img src="{job_data["thumbnail_url"]}" style="width:60px;height:60px;object-fit:cover;border-radius:4px;cursor:pointer;" alt="Thumbnail" loading="lazy" title="Click to download" onerror="this.parentElement.style.display=\'none\'"/></a>'
            
            # Add job header row with thumbnail
            files_data.append([
                f"📁 {job_id[:8]}...",
                f"Job ({len(job_data['files'])} files)",
                "",
                job_data['last_modified'],
                thumbnail_html
            ])
            
            # Add individual file rows with indentation
            for file_info in job_data['files']:
                files_data.append([
                    job_id,  # Keep full job_id for selection
                    f"  └─ {file_info['filename']}",  # Indent filename
                    file_info['size'],
                    file_info['last_modified'],
                    ""  # Empty thumbnail for individual files
                ])
        
        # Sort by last modified date
        files_data.sort(key=lambda x: x[3], reverse=True)
        
        print(f"Found {len([f for f in files_data if not f[1].startswith('Job')])} files in {len(jobs_dict)} jobs")
        print(f"=== End Refresh (ID: {refresh_id}) ===\n")
        
        return files_data
        
    except Exception as e:
        print(f"Error refreshing S3 contents: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def preview_json(s3_bucket_name, s3_input_prefix, s3_output_prefix, video_file, 
                instance_type, sfm_software, training_model, cube_faces_remove, bg_removal_model,
                filter_blurry, max_images, sfm_enable, enhanced_feature, matching_method, use_colmap_model,
                use_transform_json, training_enable, max_steps, spherical_enable, remove_bg, remove_objects,
                object_removal_action, objects_to_remove, source_coordinate, pose_world_to_cam, log_verbosity, mask_threshold, 
                crop_output_bounds, crop_mode, clean_splat, enable_spz, enable_sog, video_start_time, video_stop_time, preserve_scene_scale):
    unique_uuid = uuid.uuid4()
    original_filename = os.path.basename(video_file) if video_file else "No file selected"
    
    # Create filename with basename_uuid.ext format to avoid conflicts
    if video_file:
        file_name, file_extension = os.path.splitext(original_filename)
        media_filename = f"{file_name}_{str(unique_uuid)}{file_extension}"
    else:
        media_filename = "No file selected"

    file_contents = {
        "uuid": str(unique_uuid),
        "instanceType": instance_type.strip(),
        "useSpotInstance": shared_state.use_spot_instance,
        "logVerbosity": log_verbosity,
        "s3": {
            "bucketName": s3_bucket_name,
            "inputPrefix": s3_input_prefix,
            "inputKey": media_filename,
            "outputPrefix": s3_output_prefix
        },
        "videoProcessing": {
            "maxNumImages": str(max_images),
            "videoStartTime": video_start_time if video_start_time is not None else None,
            "videoStopTime": video_stop_time if video_stop_time is not None else None,
            "filterBlurryImages": filter_blurry == "true"
        },
        "reconstruction": {
            "enable": sfm_enable == "true",
            "softwareName": sfm_software,
            "posePriors": {
                "usePosePriorColmapModelFiles": use_colmap_model == "true",
                "usePosePriorTransformJson": {
                    "enable": use_transform_json == "true",
                    "sourceCoordinateName": source_coordinate,
                    "poseIsWorldToCam": pose_world_to_cam == "true"
                }
            },
            "enableEnhancedFeatureExtraction": enhanced_feature == "true",
            "matchingMethod": matching_method
        },
        "training": {
            "enable": training_enable == "true",
            "maxSteps": str(max_steps),
            "model": training_model,
            "preserveSceneScale": preserve_scene_scale == "true",
            "3dIsp": shared_state.isp_3d
        },
        "postProcessing": {
            "cropOutputBounds": crop_output_bounds == "true" if isinstance(crop_output_bounds, str) else crop_output_bounds,
            "cropMode": crop_mode if isinstance(crop_mode, str) else "environment",
            "cleanSplat": clean_splat == "true" if isinstance(clean_splat, str) else clean_splat,
            "enableSpz": enable_spz == "true",
            "enableSog": enable_sog == "true",
            "enableUsdz": shared_state.enable_usdz == "true",
            "plyCoords": shared_state.ply_coords,
            "spzCoords": shared_state.spz_coords,
            "sogCoords": shared_state.sog_coords,
            "usdzCoords": shared_state.usdz_coords
        },
        "sphericalCamera": {
            "enable": spherical_enable == "true",
            "cubeFacesToRemove": cube_faces_remove
        },
        "segmentation": {
            "backgroundRemoval": {
                "enable": remove_bg == "true",
                "model": bg_removal_model,
                "maskThreshold": str(mask_threshold)
            },
            "objectRemoval": {
                "enable": remove_objects == "true",
                "action": object_removal_action,
                "objects": str(objects_to_remove)
            }
        }
    }
    return json.dumps(file_contents, indent=2)

def generate_splat(s3_bucket_name, s3_input_prefix, s3_output_prefix, file_obj, 
                  instance_type, sfm_software, training_model, cube_faces_remove, 
                  bg_removal_model, filter_blurry, max_images, 
                  sfm_enable, enhanced_feature, matching_method, use_colmap_model,
                  use_transform_json, training_enable, max_steps, 
                  spherical_enable, remove_bg, remove_objects, source_coordinate, 
                  pose_world_to_cam, log_verbosity, mask_threshold, enable_spz, enable_sog, media_input_prefix="media-input"):
    try:
        session = boto3.Session()
        s3 = session.client('s3')
        unique_uuid = uuid.uuid4()

        # Get actual values from Gradio components
        s3_bucket_name = getattr(s3_bucket_name, 'value', s3_bucket_name)
        s3_input_prefix = getattr(s3_input_prefix, 'value', s3_input_prefix)
        s3_output_prefix = getattr(s3_output_prefix, 'value', s3_output_prefix)
        instance_type = getattr(instance_type, 'value', instance_type)
        sfm_software = getattr(sfm_software, 'value', sfm_software)
        training_model = getattr(training_model, 'value', training_model)
        cube_faces_remove = getattr(cube_faces_remove, 'value', cube_faces_remove)
        bg_removal_model = getattr(bg_removal_model, 'value', bg_removal_model)
        filter_blurry = getattr(filter_blurry, 'value', filter_blurry)
        max_images = getattr(max_images, 'value', max_images)
        sfm_enable = getattr(sfm_enable, 'value', sfm_enable)
        enhanced_feature = getattr(enhanced_feature, 'value', enhanced_feature)
        matching_method = getattr(matching_method, 'value', matching_method)
        use_colmap_model = getattr(use_colmap_model, 'value', use_colmap_model)
        use_transform_json = getattr(use_transform_json, 'value', use_transform_json)
        training_enable = getattr(training_enable, 'value', training_enable)
        max_steps = getattr(max_steps, 'value', max_steps)
        spherical_enable = getattr(spherical_enable, 'value', spherical_enable)
        enable_spz = getattr(enable_spz, 'value', enable_spz)
        enable_sog = getattr(enable_sog, 'value', enable_sog)
        remove_bg = getattr(remove_bg, 'value', remove_bg)
        remove_objects = getattr(remove_objects, 'value', remove_objects)
        source_coordinate = getattr(source_coordinate, 'value', source_coordinate)
        pose_world_to_cam = getattr(pose_world_to_cam, 'value', pose_world_to_cam)
        log_verbosity = getattr(log_verbosity, 'value', log_verbosity)
        mask_threshold = getattr(mask_threshold, 'value', mask_threshold)
        media_input_prefix = getattr(media_input_prefix, 'value', media_input_prefix)

        # Step 1: Upload the video file to media-input prefix with basename_uuid.ext format
        original_filename = os.path.basename(file_obj.name)
        file_name, file_extension = os.path.splitext(original_filename)
        filename = f"{file_name}_{str(unique_uuid)}{file_extension}"
        video_key = f"{media_input_prefix}/{filename}"
        
        print(f"Uploading video to s3://{s3_bucket_name}/{video_key}")
        s3.upload_file(
            Filename=file_obj.name,
            Bucket=s3_bucket_name,
            Key=video_key
        )

        # Step 2: Create the job JSON with correct media-input prefix
        job_config = {
            "uuid": str(unique_uuid),
            "instanceType": instance_type.strip(),
            "useSpotInstance": "false",
            "logVerbosity": log_verbosity,
            "s3": {
                "bucketName": s3_bucket_name,
                "inputPrefix": media_input_prefix,  # Use media_input_prefix instead of s3_input_prefix
                "inputKey": filename,
                "outputPrefix": s3_output_prefix
            },
            "videoProcessing": {
                "maxNumImages": str(max_images),
                "filterBlurryImages": filter_blurry == "true"
            },
            "reconstruction": {
                "enable": sfm_enable == "true",
                "softwareName": sfm_software,
                "posePriors": {
                    "usePosePriorColmapModelFiles": use_colmap_model == "true",
                    "usePosePriorTransformJson": {
                        "enable": use_transform_json == "true",
                        "sourceCoordinateName": source_coordinate,
                        "poseIsWorldToCam": pose_world_to_cam == "true"
                    }
                },
                "enableEnhancedFeatureExtraction": enhanced_feature == "true",
                "matchingMethod": matching_method
            },
            "training": {
                "enable": training_enable == "true",
                "maxSteps": str(max_steps),
                "model": training_model,
                "preserveSceneScale": shared_state.preserve_scene_scale == "true",
                "3dIsp": shared_state.isp_3d
            },
            "postProcessing": {
                "cropOutputBounds": shared_state.crop_output_bounds == "true",
                "cropMode": shared_state.crop_mode,
                "cleanSplat": shared_state.clean_splat == "true",
                "enableSpz": enable_spz == "true",
                "enableSog": enable_sog == "true",
                "enableUsdz": shared_state.enable_usdz == "true",
                "plyCoords": shared_state.ply_coords,
                "spzCoords": shared_state.spz_coords,
                "sogCoords": shared_state.sog_coords,
                "usdzCoords": shared_state.usdz_coords
            },
            "sphericalCamera": {
                "enable": spherical_enable == "true",
                "cubeFacesToRemove": cube_faces_remove if isinstance(cube_faces_remove, list) else []
            },
            "segmentation": {
                "backgroundRemoval": {
                    "enable": remove_bg == "true",
                    "model": bg_removal_model,
                    "maskThreshold": str(mask_threshold)
                },
                "objectRemoval": {
                    "enable": remove_objects == "true",
                    "action": "erase",
                    "objects": "['human']"
                }
            }
        }

        # Step 3: Upload the job JSON to workflow-input prefix
        job_json_key = f"{s3_input_prefix}/{unique_uuid}.json"
        print(f"Uploading job configuration to s3://{s3_bucket_name}/{job_json_key}")
        
        # Convert job config to JSON string
        job_json = json.dumps(job_config, indent=4)
        
        # Upload JSON using put_object
        s3.put_object(
            Bucket=s3_bucket_name,
            Key=job_json_key,
            Body=job_json.encode('utf-8'),
            ContentType='application/json'
        )

        return f"Successfully uploaded video and job configuration.\nVideo: {filename}\nJob ID: {unique_uuid}"

    except Exception as e:
        return f"Error processing file: {str(e)}"

def create_upload_aws_tab():
    with gr.Tab("Upload Media"):
        with gr.Row():
            with gr.Column():
                video_file = gr.File(
                    label="Upload Media File",
                    file_types=[".mp4", ".MP4", ".mov", ".MOV", ".zip", ".ZIP"]
                )
                output = gr.Textbox(label="Output", lines=20)
                upload_button = gr.Button("Upload to AWS (Submit Job)", variant="primary", elem_classes=["orange-button"])

                def upload_to_aws(video_file):
                    try:
                        if video_file is None:
                            return "Please upload a media file first."
                        
                        session = boto3.Session()
                        s3 = session.client('s3')
                        unique_uuid = uuid.uuid4()
                        
                        # Use shared_state values
                        bucket_name = shared_state.s3_bucket
                        
                        # 1. Upload the video file with multipart and add UUID to basename to avoid conflicts
                        original_filename = os.path.basename(video_file.name)
                        file_name, file_extension = os.path.splitext(original_filename)
                        filename = f"{file_name}_{str(unique_uuid)}{file_extension}"
                        video_key = f"media-input/{filename}"
                        
                        # Configure the transfer config for multipart upload
                        config = boto3.s3.transfer.TransferConfig(
                            multipart_threshold=1024 * 1024 * 8,  # 8MB
                            max_concurrency=10,  # Number of concurrent threads
                            multipart_chunksize=1024 * 1024 * 8,  # 8MB per part
                            use_threads=True
                        )
                        
                        # Create a callback to monitor upload progress
                        class ProgressPercentage:
                            def __init__(self, filename):
                                self._filename = filename
                                self._size = float(os.path.getsize(filename))
                                self._seen_so_far = 0
                                self._lock = threading.Lock()

                            def __call__(self, bytes_amount):
                                with self._lock:
                                    self._seen_so_far += bytes_amount
                                    percentage = (self._seen_so_far / self._size) * 100
                                    print(f"\rUploading {self._filename}: {percentage:.2f}%", end='', flush=True)

                        # Upload file with progress callback
                        print(f"Starting multipart upload for {filename}...")
                        s3_transfer = boto3.s3.transfer.S3Transfer(s3, config)
                        s3_transfer.upload_file(
                            video_file.name,
                            bucket_name,
                            video_key,
                            callback=ProgressPercentage(video_file.name)
                        )
                        
                        print("\nVideo upload complete!")
                        
                        # 2. Create job configuration JSON with the renamed file
                        job_config = {
                            "uuid": str(unique_uuid),
                            "instanceType": shared_state.instance.strip(),
                            "useSpotInstance": shared_state.use_spot_instance,
                            "logVerbosity": shared_state.log_verbosity,
                            "s3": {
                                "bucketName": bucket_name,
                                "inputPrefix": "media-input",
                                "inputKey": filename,
                                "outputPrefix": shared_state.s3_output
                            },
                            "videoProcessing": {
                                "maxNumImages": str(shared_state.max_images),
                                "videoStartTime": shared_state.video_start_time if shared_state.video_start_time is not None else None,
                                "videoStopTime": shared_state.video_stop_time if shared_state.video_stop_time is not None else None,
                                "filterBlurryImages": shared_state.filter_blurry == "true"
                            },
                            "reconstruction": {
                                "enable": shared_state.sfm_enable == "true",
                                "softwareName": shared_state.sfm,
                                "posePriors": {
                                    "usePosePriorColmapModelFiles": shared_state.use_colmap_model == "true",
                                    "usePosePriorTransformJson": {
                                        "enable": shared_state.use_transform_json == "true",
                                        "sourceCoordinateName": shared_state.source_coordinate,
                                        "poseIsWorldToCam": shared_state.pose_world_to_cam == "true"
                                    }
                                },
                                "enableEnhancedFeatureExtraction": shared_state.enhanced_feature == "true",
                                "matchingMethod": shared_state.matching_method
                            },
                            "training": {
                                "enable": shared_state.training_enable == "true",
                                "maxSteps": str(shared_state.max_steps),
                                "model": shared_state.model,
                                "preserveSceneScale": shared_state.preserve_scene_scale == "true",
                                "3dIsp": shared_state.isp_3d
                            },
                            "postProcessing": {
                                "cropOutputBounds": shared_state.crop_output_bounds == "true",
                                "cropMode": shared_state.crop_mode,
                                "cleanSplat": shared_state.clean_splat == "true",
                                "enableSpz": shared_state.enable_spz == "true",
                                "enableSog": shared_state.enable_sog == "true",
                                "enableUsdz": shared_state.enable_usdz == "true",
                                "plyCoords": shared_state.ply_coords,
                                "spzCoords": shared_state.spz_coords,
                                "sogCoords": shared_state.sog_coords,
                                "usdzCoords": shared_state.usdz_coords
                            },
                            "sphericalCamera": {
                                "enable": shared_state.spherical_enable == "true",
                                "cubeFacesToRemove": shared_state.faces if isinstance(shared_state.faces, list) else []
                            },
                            "segmentation": {
                                "backgroundRemoval": {
                                    "enable": shared_state.remove_bg == "true",
                                    "model": shared_state.bg_model,
                                    "maskThreshold": str(shared_state.mask_threshold)
                                },
                                "objectRemoval": {
                                    "enable": shared_state.remove_objects == "true",
                                    "action": shared_state.object_removal_action,
                                    "objects": "['human']"
                                }
                            }
                        }
                        
                        # 3. Upload job JSON to workflow-input
                        job_json_key = f"{shared_state.s3_input}/{unique_uuid}.json"
                        job_json = json.dumps(job_config, indent=4)
                        
                        s3.put_object(
                            Bucket=bucket_name,
                            Key=job_json_key,
                            Body=job_json.encode('utf-8'),
                            ContentType='application/json'
                        )
                        
                        return f"Successfully uploaded video and job configuration.\nVideo: {file_name}\nJob ID: {unique_uuid}"
                        
                    except Exception as e:
                        return f"Error uploading files: {str(e)}"

                upload_button.click(
                    fn=upload_to_aws,
                    inputs=[video_file],
                    outputs=[output]
                )

def create_aws_configuration_tab():
    with gr.Tab("AWS Configuration"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Compute Settings")
                instance = gr.Dropdown(
                    label="Instance Type",
                    choices=[
                        "ml.g5.4xlarge", # (1x24GB VRAM, 16 vCPUs, 128 GB RAM)
                        "ml.g5.8xlarge", # (1x24GB VRAM, 32 vCPUs, 256 GB RAM)
                        "ml.g5.12xlarge", # (4x24GB VRAM, 48 vCPUs, 192 GB RAM)
                        "ml.g6.4xlarge", # (1x24GB VRAM, 16 vCPUs, 64 GB RAM)
                        "ml.g6.8xlarge", # (1x24GB VRAM, 32 vCPUs, 128 GB RAM)
                        "ml.g6e.4xlarge"], # (1x48GB VRAM, 16 vCPUs, 128 GB RAM)
                    value=shared_state.instance
                )
                use_spot_instance = gr.Radio(
                    label="Compute Type",
                    choices=[("AWS Batch (Spot Instances - Up to 50% cost savings)", "true"), ("SageMaker (On-Demand)", "false")],
                    value=shared_state.use_spot_instance,
                    info="Batch uses spot instances for significant cost savings but may have longer queue times"
                )
                gr.HTML("<hr style='margin: 15px 0; border: none; border-top: 2px solid #ddd;'>")
                gr.Markdown("### AWS Settings")
                aws_region = gr.Textbox(label="AWS Region", value=shared_state.aws_region)
                ddb_table_name = gr.Textbox(label="DynamoDB Table Name", value=shared_state.ddb_table_name)
                s3_bucket = gr.Textbox(label="S3 Bucket Name", value=shared_state.s3_bucket)
                s3_input = gr.Textbox(label="S3 Input Prefix", value=shared_state.s3_input)
                s3_output = gr.Textbox(label="S3 Output Prefix", value="workflow-output")
                media_input = gr.Textbox(label="Media Input Prefix", value="media-input")

                def update_shared_state(region, ddb_table, bucket, input_prefix, output_prefix, media_prefix, inst, spot):
                    shared_state.aws_region = region
                    shared_state.ddb_table_name = ddb_table
                    shared_state.s3_bucket = bucket
                    shared_state.s3_input = input_prefix
                    shared_state.s3_output = output_prefix
                    shared_state.media_input = media_prefix
                    shared_state.instance = inst
                    shared_state.use_spot_instance = spot
                    return "AWS configuration updated"

                # Immediately update shared_state when values change
                def update_instance_type(inst):
                    shared_state.instance = inst
                
                def update_spot_instance(spot):
                    shared_state.use_spot_instance = spot
                
                # Update shared_state immediately when values change
                instance.change(
                    fn=update_instance_type,
                    inputs=[instance]
                )
                
                use_spot_instance.change(
                    fn=update_spot_instance,
                    inputs=[use_spot_instance]
                )
                
                # Update shared state when any value changes
                for component in [aws_region, ddb_table_name, s3_bucket, s3_input, s3_output, media_input]:
                    component.change(
                        fn=update_shared_state,
                        inputs=[aws_region, ddb_table_name, s3_bucket, s3_input, s3_output, media_input, instance, use_spot_instance],
                        outputs=[gr.Textbox(label="Status", visible=False)]
                    )

def create_advanced_settings_tab():
    # Define helper function first
    def get_saved_configs():
        configs_dir = os.path.join(os.path.dirname(__file__), "configs")
        if not os.path.exists(configs_dir):
            return []
        return [f[:-5] for f in os.listdir(configs_dir) if f.endswith('.json')]
    
    with gr.Tab("Advanced Settings"):
        # Configuration Presets Section with visual separation
        with gr.Group():
            gr.HTML("<div style='border-bottom: 2px solid #e0e0e0; margin-bottom: 15px; padding-bottom: 10px;'><h3 style='margin: 0;'>⚙️ Configuration Presets</h3></div>")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        config_name = gr.Textbox(label="Configuration Name", placeholder="Enter preset name")
                        save_config_btn = gr.Button("Save Config", size="sm")
                        load_config_dropdown = gr.Dropdown(label="Load Config", choices=get_saved_configs(), interactive=True, min_width=200)
                        load_config_btn = gr.Button("Load Config", size="sm")
                    config_status = gr.Textbox(label="Status", value="", interactive=False)
        
        # Separator between presets and settings
        gr.HTML("<hr style='margin: 20px 0; border: none; border-top: 1px solid #ddd;'>")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### General")
                log_verbosity = gr.Dropdown(
                    label="Log Verbosity",
                    choices=["info", "warning", "error"],
                    value="info"
                )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Video Processing")
                filter_blurry = gr.Radio(
                label="Filter Blurry Images",
                choices=["true", "false"],
                value="true"
                )
                max_images = gr.Number(
                label="Max Images",
                value=300,
                minimum=1,
                maximum=4999
                )
                video_start_time = gr.Number(
                    label="Video Start Time (seconds)",
                    value=0.0,
                    minimum=0.0,
                    info="Start time in seconds for video processing"
                )
                video_stop_time = gr.Number(
                    label="Video Stop Time (seconds)",
                    value=None,
                    minimum=0.0,
                    info="End time in seconds (leave zero/blank for full video)"
                )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Spherical Camera")
                spherical_enable = gr.Radio(
                    label="Enable Spherical Camera",
                    choices=["true", "false"],
                    value="false"
                )
                faces_options = ["down", "up", "front", "back", "left", "right"]
                faces = gr.CheckboxGroup(
                    label="Cube Faces to Remove",
                    choices=faces_options,
                    value=[]
                )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Segmentation")
                with gr.Group():
                    gr.HTML("<h4 style='margin: 0 0 10px 0; padding: 8px; background: rgba(0,0,0,0.05); border-radius: 4px; border-left: 3px solid #007acc;'>🎭 Background Removal</h4>")
                    remove_bg = gr.Radio(
                        label="Enable Background Removal",
                        choices=["true", "false"],
                        value="false"
                    )
                    bg_model = gr.Dropdown(
                        label="Background Removal Model",
                        choices=["u2net", "sam2"],
                        value="u2net"
                    )
                    mask_threshold = gr.Slider(
                        label="SAM2 Mask Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.6,
                        step=0.01,
                        info="If object doesn't have large contrast from background, use lower number like 0.38"
                    )
                with gr.Group():
                    gr.HTML("<h4 style='margin: 0 0 10px 0; padding: 8px; background: rgba(0,0,0,0.05); border-radius: 4px; border-left: 3px solid #ff6b35;'>🎯 Object Removal</h4>")
                    remove_objects = gr.Radio(
                        label="Enable Object Removal",
                        choices=["true", "false"],
                        value="false"
                    )
                    object_removal_action = gr.Radio(
                        label="Object Removal Action",
                        choices=["erase", "remove"],
                        value="erase"
                    )
                    objects_to_remove = gr.CheckboxGroup(
                        label="Objects to Remove",
                        choices=["human"],
                        value=[]
                    )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Reconstruction")
                # Reconstruction Settings
                sfm_enable = gr.Radio(
                    label="Enable Reconstruction",
                    choices=["true", "false"],
                    value="true"
                )
                sfm = gr.Dropdown(
                    label="Reconstruction Software",
                    choices=["colmap", "glomap", "map_anything"],
                    value="glomap"
                )
            with gr.Column():
                gr.Markdown("### Colmap Settings")
                enhanced_feature = gr.Radio(
                    label="Colmap Enhanced Feature Extraction",
                    choices=["true", "false"],
                    value="false"
                )
                matching_method = gr.Dropdown(
                    label="Colmap Matching Method",
                    choices=["sequential", "exhaustive", "vocab", "spatial"],
                    value="sequential"
                )
            with gr.Column():
                gr.Markdown("### Pose Priors-Colmap")
                use_colmap_model = gr.Radio(
                    label="Use COLMAP Model",
                    choices=["true", "false"],
                    value="false"
                )
            with gr.Column():
                gr.Markdown("### Pose Priors-Transform JSON")
                use_transform_json = gr.Radio(
                    label="Use Transform JSON",
                    choices=["true", "false"],
                    value="false"
                )
                source_coordinate = gr.Dropdown(
                    label="Source Coordinate",
                    choices=["arkit", "arcore", "opengl", "opencv", "ros"],
                    value="arkit"
                )
                pose_world_to_cam = gr.Radio(
                    label="Pose is World to Camera",
                    choices=["true", "false"],
                    value="true"
                )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Training")
                training_enable = gr.Radio(
                    label="Enable Training",
                    choices=["true", "false"],
                    value="true"
                )
                max_steps = gr.Number(
                    label="Max Steps",
                    value=15000,
                    minimum=5,
                    maximum=100000
                )
                model = gr.Dropdown(
                    label="Training Model",
                    choices=[
                        "splatfacto",
                        "splatfacto-big",
                        "splatfacto-mcmc",
                        "splatfacto-w-light",
                        "3dgut",
                        "3dgrt",
                        "nerfacto"
                    ],
                    value="splatfacto"
                )
                preserve_scene_scale = gr.Radio(
                    label="Preserve Scene Scale",
                    choices=["true", "false"],
                    value="false"
                )
                isp_3d = gr.Dropdown(
                    label="3D ISP Mode",
                    choices=[
                        ("None", "none"),
                        ("Bilateral Grid (bilagrid)", "bilagrid"),
                        ("Per-Pixel ISP (ppisp)", "ppisp")
                    ],
                    value="none",
                    info="Image signal processing for splatfacto, gsplat multi-GPU, and 3DGRUT. Not applicable to nerfacto."
                )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Post Processing")
                crop_output_bounds = gr.Radio(
                    label="Crop Output Bounds",
                    choices=["true", "false"],
                    value="false"
                )
                crop_mode = gr.Dropdown(
                    label="Crop Mode",
                    choices=["environment", "rigid_body"],
                    value="environment"
                )
                clean_splat = gr.Radio(
                    label="Clean Splat (Remove Noise)",
                    choices=["true", "false"],
                    value="false"
                )
                enable_spz = gr.Radio(
                    label="Enable SPZ Export",
                    choices=["true", "false"],
                    value="true"
                )
                enable_sog = gr.Radio(
                    label="Enable SOG Export",
                    choices=["true", "false"],
                    value="true"
                )
                enable_usdz = gr.Radio(
                    label="Enable USDZ Export",
                    choices=["true", "false"],
                    value="true"
                )
                ply_coords = gr.Dropdown(
                    label="PLY Coordinate System",
                    choices=[
                        ("Right-Hand, Y-Up (playcanvas)", "rhyu"),
                        ("Left-Hand, Y-Up (babylon.js)", "lhyu"),
                        ("Right-Hand, Z-Up (blender)", "rhzu"),
                        ("Left-Hand, Z-Up (unreal)", "lhzu")
                    ],
                    value="rhyu"
                )
                spz_coords = gr.Dropdown(
                    label="SPZ Coordinate System",
                    choices=[
                        ("Right-Hand, Y-Up (playcanvas)", "rhyu"),
                        ("Left-Hand, Y-Up (babylon.js)", "lhyu"),
                        ("Right-Hand, Z-Up (blender)", "rhzu"),
                        ("Left-Hand, Z-Up (unreal)", "lhzu")
                    ],
                    value="rhyu"
                )
                sog_coords = gr.Dropdown(
                    label="SOG Coordinate System",
                    choices=[
                        ("Right-Hand, Y-Up (playcanvas)", "rhyu"),
                        ("Left-Hand, Y-Up (babylon.js)", "lhyu"),
                        ("Right-Hand, Z-Up (blender)", "rhzu"),
                        ("Left-Hand, Z-Up (unreal)", "lhzu")
                    ],
                    value="rhyu"
                )
                usdz_coords = gr.Dropdown(
                    label="USDZ Coordinate System",
                    choices=[
                        ("Right-Hand, Y-Up (playcanvas)", "rhyu"),
                        ("Left-Hand, Y-Up (babylon.js)", "lhyu"),
                        ("Right-Hand, Z-Up (blender)", "rhzu"),
                        ("Left-Hand, Z-Up (unreal)", "lhzu")
                    ],
                    value="rhyu"
                )

                def update_advanced_settings(*args):
                    # Update shared state with all advanced settings
                    (shared_state.sfm, shared_state.model, shared_state.faces, 
                     shared_state.bg_model, shared_state.filter_blurry,
                     shared_state.max_images, shared_state.video_start_time, shared_state.video_stop_time, shared_state.sfm_enable, 
                     shared_state.enhanced_feature, shared_state.matching_method,
                     shared_state.use_colmap_model, shared_state.use_transform_json,
                     shared_state.training_enable, shared_state.max_steps, shared_state.enable_spz, shared_state.enable_sog, shared_state.enable_usdz,
                     shared_state.crop_output_bounds, shared_state.crop_mode, shared_state.clean_splat,
                     shared_state.spherical_enable,
                     shared_state.remove_bg, shared_state.remove_objects,
                     shared_state.object_removal_action, shared_state.objects_to_remove, shared_state.source_coordinate, shared_state.pose_world_to_cam,
                     shared_state.log_verbosity, shared_state.mask_threshold, shared_state.ply_coords, shared_state.spz_coords, shared_state.sog_coords, shared_state.usdz_coords, shared_state.preserve_scene_scale, shared_state.isp_3d) = args
                    return "Advanced settings updated"

                # Get all advanced settings components after they're defined
                advanced_components = [
                    sfm, model, faces, bg_model, filter_blurry,
                    max_images, video_start_time, video_stop_time, sfm_enable, enhanced_feature, matching_method,
                    use_colmap_model, use_transform_json, training_enable,
                    max_steps, enable_spz, enable_sog, enable_usdz,
                    crop_output_bounds, crop_mode, clean_splat,
                    spherical_enable, remove_bg, remove_objects,
                    object_removal_action, objects_to_remove, source_coordinate, pose_world_to_cam,
                    log_verbosity, mask_threshold, ply_coords, spz_coords, sog_coords, usdz_coords, preserve_scene_scale, isp_3d
                ]
                
                def save_configuration(config_name, *settings):
                    if not config_name.strip():
                        return "Please enter a configuration name", gr.update()
                    
                    config_data = {
                        'sfm': settings[0],
                        'model': settings[1], 
                        'faces': settings[2],
                        'bg_model': settings[3],
                        'filter_blurry': settings[4],
                        'max_images': settings[5],
                        'video_start_time': settings[6],
                        'video_stop_time': settings[7],
                        'sfm_enable': settings[8],
                        'enhanced_feature': settings[9],
                        'matching_method': settings[10],
                        'use_colmap_model': settings[11],
                        'use_transform_json': settings[12],
                        'training_enable': settings[13],
                        'max_steps': settings[14],
                        'enable_spz': settings[15],
                        'enable_sog': settings[16],
                        'enable_usdz': settings[17],
                        'crop_output_bounds': settings[18],
                        'crop_mode': settings[19],
                        'clean_splat': settings[20],
                        'spherical_enable': settings[21],
                        'remove_bg': settings[22],
                        'remove_objects': settings[23],
                        'object_removal_action': settings[24],
                        'objects_to_remove': settings[25],
                        'source_coordinate': settings[26],
                        'pose_world_to_cam': settings[27],
                        'log_verbosity': settings[28],
                        'mask_threshold': settings[29],
                        'ply_coords': settings[30],
                        'spz_coords': settings[31],
                        'sog_coords': settings[32],
                        'usdz_coords': settings[33],
                        'preserve_scene_scale': settings[34],
                        'isp_3d': settings[35]
                    }
                    
                    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
                    os.makedirs(configs_dir, exist_ok=True)
                    safe_name = re.sub(r'[^\w\-. ]', '', config_name.strip())[:64]
                    config_file = os.path.join(configs_dir, f"{safe_name}.json")
                    if not os.path.realpath(config_file).startswith(os.path.realpath(configs_dir) + os.sep):
                        return "Error: Invalid configuration name", gr.update()
                    
                    try:
                        with open(config_file, 'w') as f:
                            json.dump(config_data, f, indent=2)
                        return f"Configuration '{config_name}' saved successfully", gr.update(choices=get_saved_configs())
                    except Exception as e:
                        return f"Error saving configuration: {str(e)}", gr.update()

                
                def load_configuration(config_name):
                    if not config_name:
                        return ["Please select a configuration"] + [gr.update() for _ in range(34)]
                    
                    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
                    safe_name = re.sub(r'[^\w\-. ]', '', config_name)[:64]
                    config_file = os.path.join(configs_dir, f"{safe_name}.json")
                    if not os.path.realpath(config_file).startswith(os.path.realpath(configs_dir) + os.sep):
                        return ["Error: Invalid configuration name"] + [gr.update() for _ in range(34)]
                    
                    try:
                        with open(config_file, 'r') as f:
                            config_data = json.load(f)
                        
                        # Update shared_state immediately when loading
                        shared_state.sfm = config_data.get('sfm', 'glomap')
                        shared_state.model = config_data.get('model', 'splatfacto')
                        shared_state.faces = config_data.get('faces', [])
                        shared_state.bg_model = config_data.get('bg_model', 'u2net')
                        shared_state.filter_blurry = config_data.get('filter_blurry', 'true')
                        shared_state.max_images = config_data.get('max_images', 300)
                        shared_state.video_start_time = config_data.get('video_start_time', 0.0)
                        shared_state.video_stop_time = config_data.get('video_stop_time', None)
                        shared_state.sfm_enable = config_data.get('sfm_enable', 'true')
                        shared_state.enhanced_feature = config_data.get('enhanced_feature', 'false')
                        shared_state.matching_method = config_data.get('matching_method', 'sequential')
                        shared_state.use_colmap_model = config_data.get('use_colmap_model', 'false')
                        shared_state.use_transform_json = config_data.get('use_transform_json', 'false')
                        shared_state.training_enable = config_data.get('training_enable', 'true')
                        shared_state.max_steps = config_data.get('max_steps', 15000)
                        shared_state.enable_spz = config_data.get('enable_spz', 'true')
                        shared_state.enable_sog = config_data.get('enable_sog', 'true')
                        shared_state.enable_usdz = config_data.get('enable_usdz', 'true')
                        shared_state.crop_output_bounds = config_data.get('crop_output_bounds', 'false')
                        shared_state.crop_mode = config_data.get('crop_mode', 'environment')
                        shared_state.clean_splat = config_data.get('clean_splat', 'false')
                        shared_state.spherical_enable = config_data.get('spherical_enable', 'false')
                        shared_state.remove_bg = config_data.get('remove_bg', 'false')
                        shared_state.remove_objects = config_data.get('remove_objects', 'false')
                        raw_action = config_data.get('object_removal_action', 'erase')
                        shared_state.object_removal_action = raw_action if raw_action in ('erase', 'remove') else 'erase'
                        shared_state.objects_to_remove = config_data.get('objects_to_remove', [])
                        shared_state.source_coordinate = config_data.get('source_coordinate', 'arkit')
                        shared_state.pose_world_to_cam = config_data.get('pose_world_to_cam', 'true')
                        shared_state.log_verbosity = config_data.get('log_verbosity', 'info')
                        shared_state.mask_threshold = config_data.get('mask_threshold', 0.6)
                        shared_state.ply_coords = config_data.get('ply_coords', 'rhyu')
                        shared_state.spz_coords = config_data.get('spz_coords', 'rhyu')
                        shared_state.sog_coords = config_data.get('sog_coords', 'rhyu')
                        shared_state.usdz_coords = config_data.get('usdz_coords', 'rhyu')
                        shared_state.preserve_scene_scale = config_data.get('preserve_scene_scale', 'false')
                        shared_state.isp_3d = config_data.get('isp_3d', 'none')
                        
                        return [
                            f"Configuration '{config_name}' loaded successfully",
                            config_data.get('sfm', 'glomap'),
                            config_data.get('model', 'splatfacto'),
                            config_data.get('faces', []),
                            config_data.get('bg_model', 'u2net'),
                            config_data.get('filter_blurry', 'true'),
                            config_data.get('max_images', 300),
                            config_data.get('video_start_time', 0.0),
                            config_data.get('video_stop_time', None),
                            config_data.get('sfm_enable', 'true'),
                            config_data.get('enhanced_feature', 'false'),
                            config_data.get('matching_method', 'sequential'),
                            config_data.get('use_colmap_model', 'false'),
                            config_data.get('use_transform_json', 'false'),
                            config_data.get('training_enable', 'true'),
                            config_data.get('max_steps', 15000),
                            config_data.get('enable_spz', 'true'),
                            config_data.get('enable_sog', 'true'),
                            config_data.get('enable_usdz', 'true'),
                            config_data.get('crop_output_bounds', 'false'),
                            config_data.get('crop_mode', 'environment'),
                            config_data.get('clean_splat', 'false'),
                            config_data.get('spherical_enable', 'false'),
                            config_data.get('remove_bg', 'false'),
                            config_data.get('remove_objects', 'false'),
                            config_data.get('object_removal_action', 'erase') if config_data.get('object_removal_action', 'erase') in ('erase', 'remove') else 'erase',
                            config_data.get('objects_to_remove', []),
                            config_data.get('source_coordinate', 'arkit'),
                            config_data.get('pose_world_to_cam', 'true'),
                            config_data.get('log_verbosity', 'info'),
                            config_data.get('mask_threshold', 0.6),
                            config_data.get('ply_coords', 'rhyu'),
                            config_data.get('spz_coords', 'rhyu'),
                            config_data.get('sog_coords', 'rhyu'),
                            config_data.get('usdz_coords', 'rhyu'),
                            config_data.get('preserve_scene_scale', 'false'),
                            config_data.get('isp_3d', 'none')
                        ]
                    except Exception as e:
                        return [f"Error loading configuration: {str(e)}"] + [gr.update() for _ in range(35)]
                
                # Wire up save/load buttons
                save_config_btn.click(
                    fn=save_configuration,
                    inputs=[config_name] + advanced_components,
                    outputs=[config_status, load_config_dropdown]
                )
                
                load_config_btn.click(
                    fn=load_configuration,
                    inputs=[load_config_dropdown],
                    outputs=[config_status] + advanced_components
                )

                # Update shared state when any value changes
                for component in advanced_components:
                    component.change(
                        fn=update_advanced_settings,
                        inputs=advanced_components,
                        outputs=[gr.Textbox(label="Status", visible=False)]
                    )

def on_select(evt: gr.SelectData, data):
    """Handle row selection in the files table with improved error handling"""
    try:
        if not hasattr(evt, 'index') or evt.index is None or len(evt.index) == 0:
            #print("[DEBUG] No index in selection event")
            raise ValueError("Invalid selection event - no index")
            
        row_idx = evt.index[0]
        
        if hasattr(data, 'values') and hasattr(data.values, 'tolist'):
            data_list = data.values.tolist()
        elif isinstance(data, list):
            data_list = data
        else:
            #print(f"[DEBUG] Unexpected data type: {type(data)}")
            raise ValueError(f"Unexpected data type: {type(data)}")
            
        if not data_list or row_idx >= len(data_list):
            raise ValueError("Invalid selection")
            
        selected_row = data_list[row_idx]
        #print(f"[DEBUG] Selected row data: {selected_row}")
        
        if not selected_row or len(selected_row) < 2:
            raise ValueError("Invalid row data structure")
        
        # Check if this is a job header row (starts with folder icon)
        if selected_row[1].startswith("Job ("):
            # This is a job header row - disable action buttons
            job_id = selected_row[0].replace("📁 ", "").replace("...", "")
            job_metadata = get_job_metadata(job_id)
            return [
                None,  # No file selected
                gr.update(interactive=False, value="Select a file to download"),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                job_metadata
            ]
        
        # Check if this is an individual file row (starts with tree characters)
        if selected_row[1].startswith("  └─ "):
            # Extract the actual filename by removing the tree characters
            actual_filename = selected_row[1].replace("  └─ ", "")
            # Create a clean row for processing
            clean_row = [selected_row[0], actual_filename, selected_row[2], selected_row[3]]
            
            job_metadata = get_job_metadata(selected_row[0])
            
            return [
                clean_row,  # Clean row data for processing
                gr.update(interactive=True, value="Download Selected"),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                job_metadata
            ]
        
        # Fallback for any other row type
        job_metadata = get_job_metadata(selected_row[0])
        return [
            selected_row,
            gr.update(interactive=True, value="Download Selected"),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            job_metadata
        ]
        
    except Exception as e:
        import traceback
        print(f"[DEBUG] Error in selection handler: {str(e)}")
        traceback.print_exc()
        return [
            None,
            gr.update(interactive=False, value="Download Selected"),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "Select a job to view metadata"
        ]

def handle_view_with_progress(selected_row):
    """Handle view button click with progress bar"""
    try:
        if not selected_row:
            return gr.update(value=None), "No file selected", ""
        
        bucket_name = shared_state.s3_bucket
        output_prefix = shared_state.s3_output or "workflow-output"
        
        job_id = selected_row[0]
        filename = selected_row[1]
        file_key = f"{output_prefix}/{job_id}/{filename}"
        
        # Check if this is the currently loaded model
        current_url = getattr(shared_state, 'current_model_url', None)
        current_key = getattr(shared_state, 'current_model_key', None)
        
        if current_key == file_key:
            # Model is already loaded, don't show progress bar
            return gr.update(value=current_url), f"Already loaded: {filename}", ""
        
        # Get file size information and validate file exists
        file_size_mb = None
        size_info = ""
        try:
            s3_client = boto3.client('s3')
            response = s3_client.head_object(Bucket=bucket_name, Key=file_key)
            file_size = response['ContentLength']
            file_size_mb = file_size / (1024 * 1024)
            size_info = f" ({file_size_mb:.1f} MB)"
            
            # Check if file size is reasonable (not empty or corrupted)
            if file_size < 1000:  # Less than 1KB is likely corrupted
                return gr.update(value=None), f"Error: File {filename} appears to be corrupted (too small)", ""
                
        except Exception as e:
            print(f"Error getting file size: {str(e)}")
            return gr.update(value=None), f"Error accessing file: {str(e)}", ""
        
        # Download file to temp directory for Gradio 6 Model3D compatibility
        # (Model3D no longer accepts external URLs, needs local file paths)
        import tempfile
        temp_dir = os.path.join(tempfile.gettempdir(), "gradio_3d_cache")
        os.makedirs(temp_dir, exist_ok=True)
        local_path = os.path.join(temp_dir, f"{job_id}_{filename}")
        
        if not os.path.exists(local_path):
            print(f"Downloading {file_key} to {local_path}...")
            s3_client = boto3.client('s3')
            s3_client.download_file(bucket_name, file_key, local_path)
            print(f"Downloaded to {local_path}")

        # Store current model info
        shared_state.current_model_url = local_path
        shared_state.current_model_key = file_key
        
        return gr.update(value=local_path), f"Loading {filename}...", ""
        
    except Exception as e:
        error_msg = f"Error viewing file: {str(e)}"
        print(f"[DEBUG] Error in handle_view_with_progress: {error_msg}")
        import traceback
        traceback.print_exc()
        return gr.update(value=None), error_msg, ""
        
        # Create a unique ID for this model
        model_id = f"{job_id}_{filename.replace('.', '_')}"
        
        # Track loaded models in shared state
        if not hasattr(shared_state, 'loaded_models'):
            shared_state.loaded_models = set()
            
        # Only show progress bar for models not yet loaded
        if file_key not in shared_state.loaded_models:
            shared_state.loaded_models.add(file_key)
            
            # Create progress bar HTML
            progress_html = f"""
            <div style="margin: 10px 0;">
                <div style="background: #555; border-radius: 10px; overflow: hidden; height: 20px;">
                    <div style="background: linear-gradient(90deg, #4CAF50, #45a049); height: 100%; width: 0%; animation: loading {estimated_time}s ease-out forwards;"></div>
                </div>
                <div style="text-align: center; margin-top: 5px; font-size: 14px;">Loading {filename}{size_info}... (~{estimated_time:.0f}s estimated)</div>
            </div>
            <style>
            @keyframes loading {{
                0% {{ width: 0%; }}
                30% {{ width: 40%; }}
                60% {{ width: 70%; }}
                90% {{ width: 90%; }}
                100% {{ width: 100%; }}
            }}
            </style>
            """
        else:
            # Empty progress HTML if already loaded
            progress_html = ""
        
        # Estimate loading time based on file size  
        # Model based on actual loading times:
        # 12MB=6sec, 31MB=9sec, 180MB=75sec, 236MB=105sec, 448MB=220sec
        if file_size_mb is None:
            estimated_time = 10
        else:
            # Quadratic model: time = 0.001x² + 0.3x + 3
            estimated_time = 0.001 * (file_size_mb ** 2) + 0.3 * file_size_mb + 3
            
        # Only show progress bar when the View button is clicked
        # Check if we're navigating between tabs by looking at the referrer
        progress_html = f"""
        <div style="margin: 10px 0;">
            <div style="background: #555; border-radius: 10px; overflow: hidden; height: 20px;">
                <div style="background: linear-gradient(90deg, #4CAF50, #45a049); height: 100%; width: 0%; animation: loading {estimated_time}s ease-out forwards;"></div>
            </div>
            <div style="text-align: center; margin-top: 5px; font-size: 14px;">Loading {filename}{size_info}... (~{estimated_time:.0f}s estimated)</div>
        </div>
        <style>
        @keyframes loading {{
            0% {{ width: 0%; }}
            30% {{ width: 40%; }}
            60% {{ width: 70%; }}
            90% {{ width: 90%; }}
            100% {{ width: 100%; }}
        }}
        </style>
        <script>
        (function() {{
            // Check if this is a tab navigation by looking at document.referrer
            const isTabNavigation = document.referrer.includes(window.location.origin);
            
            // If this is tab navigation, hide the progress bar
            if (isTabNavigation) {{
                // Find all progress bars and hide them
                const progressBars = document.querySelectorAll('div[style*="margin: 10px 0;"]');
                progressBars.forEach(bar => {{
                    bar.style.display = 'none';
                }});
            }}
        }})();
        </script>
        """
        
        # Create a unique ID for this model
        model_id = f"{job_id}_{filename.replace('.', '_')}"
        
        # Return all three required outputs
        return gr.update(value=presigned_url), f"Loading {filename}...", progress_html
        
    except Exception as e:
        error_msg = f"Error viewing file: {str(e)}"
        print(f"[DEBUG] Error in handle_view_with_progress: {error_msg}")
        import traceback
        traceback.print_exc()
        return gr.update(value=None), error_msg, ""

def handle_view(selected_row):
    """Handle view button click"""
    result = handle_view_with_progress(selected_row)
    return result[0], result[1]

def create_playcanvas_sog_viewer(presigned_url, filename):
    """Create PlayCanvas SOG viewer using exact working method"""
    # Fetch the SOG file and convert to base64 (like the working version)
    import requests
    import base64
    try:
        response = requests.get(presigned_url)
        file_data = base64.b64encode(response.content).decode('utf-8')
        file_size = f"{len(response.content):,} bytes"
        
        # Store the data globally and trigger the viewer
        viewer_html = f"""
        <div id="sog-container" style="height: 900px; background: #1a1a1a; border: 1px solid #444; display: flex; align-items: center; justify-content: center; color: white;">Loading SOG viewer...</div>
        <script>
        window.sogData = {{
            fileData: '{file_data}',
            fileName: '{html.escape(filename)}',
            fileSize: '{html.escape(file_size)}'
        }};
        window.sogLoaded = false;
        </script>
        """
        return viewer_html
        
    except Exception as e:
        return f"<div style='color: red;'>Error loading SOG file: {str(e)}</div>"

def handle_view_multi(selected_row):
    """Handle view button click for 3D models, SOG files, and videos"""
    try:
        if not selected_row:
            return gr.update(value=None), "No file selected", gr.update(value=None), "No file selected", gr.update(value=None), "No file selected"
        
        filename = selected_row[1]
        
        # Download splat files server-side to avoid browser CORS restrictions on S3 presigned URLs
        if filename.lower().endswith(('.sog', '.ply')):
            import requests, base64
            bucket_name = shared_state.s3_bucket
            output_prefix = shared_state.s3_output or "workflow-output"
            job_id = selected_row[0]
            file_key = f"{output_prefix}/{job_id}/{filename}"
            # Skip re-download if already cached from a prior table-click
            current_key = getattr(shared_state, 'current_model_key', None)
            cached_data = getattr(shared_state, 'current_model_data', None)
            if current_key == file_key and cached_data:
                file_data = cached_data
            else:
                presigned_url = generate_presigned_url(bucket_name, file_key)
                if not presigned_url:
                    return (gr.update(value=None), "", gr.update(value=""), "Error generating URL", gr.update(value=None), "", gr.update(value=""))
                try:
                    response = requests.get(presigned_url)
                    file_data = base64.b64encode(response.content).decode('utf-8')
                except Exception as e:
                    return (gr.update(value=None), "", gr.update(value=""), f"Error: {e}", gr.update(value=None), "", gr.update(value=""))
                shared_state.current_model_key = file_key
                shared_state.current_model_data = file_data
            return (gr.update(value=None), "", gr.update(value=json.dumps({"data": file_data, "filename": filename, "ts": __import__('time').time()})), f"Loading {filename}...", gr.update(value=None), "", gr.update(value=""))

        elif filename.lower().endswith('.spz'):
            import requests, base64
            bucket_name = shared_state.s3_bucket
            output_prefix = shared_state.s3_output or "workflow-output"
            job_id = selected_row[0]
            file_key = f"{output_prefix}/{job_id}/{filename}"
            current_key = getattr(shared_state, 'current_model_key', None)
            cached_data = getattr(shared_state, 'current_model_data', None)
            if current_key == file_key and cached_data:
                file_data = cached_data
            else:
                presigned_url = generate_presigned_url(bucket_name, file_key)
                if not presigned_url:
                    return (gr.update(value=None), "", gr.update(value=""), "Error generating URL", gr.update(value=None), "", gr.update(value=""))
                try:
                    response = requests.get(presigned_url)
                    file_data = base64.b64encode(response.content).decode('utf-8')
                except Exception as e:
                    return (gr.update(value=None), "", gr.update(value=""), f"Error: {e}", gr.update(value=None), "", gr.update(value=""))
                shared_state.current_model_key = file_key
                shared_state.current_model_data = file_data
            payload = json.dumps({"data": file_data, "filename": filename, "ts": __import__('time').time()})
            return (gr.update(value=None), "", gr.update(value=""), gr.update(value=""), gr.update(value=None), "", gr.update(value=payload))

        elif filename.lower().endswith('.mp4'):
            bucket_name = shared_state.s3_bucket
            output_prefix = shared_state.s3_output or "workflow-output"
            job_id = selected_row[0]
            file_key = f"{output_prefix}/{job_id}/{filename}"
            
            presigned_url = generate_presigned_url(bucket_name, file_key)
            if presigned_url:
                return (
                    gr.update(value=None), "Video selected - check Video Preview tab",
                    gr.update(value=""), gr.update(value=""),
                    gr.update(value=presigned_url), f"Loaded video: {filename}",
                    gr.update(value="")
                )
            else:
                return (
                    gr.update(value=None), "Error loading video",
                    gr.update(value=""), gr.update(value=""),
                    gr.update(value=None), "Error generating video URL",
                    gr.update(value="")
                )
        else:
            result = handle_view_with_progress(selected_row)
            return (
                result[0], result[1],
                gr.update(value=""), gr.update(value=""),
                gr.update(value=None), "3D model selected - check 3D Model Viewer tab",
                gr.update(value="")
            )

    except Exception as e:
        error_msg = f"Error viewing file: {str(e)}"
        return (
            gr.update(value=None), error_msg,
            gr.update(value=""), gr.update(value=f"Error: {error_msg}"),
            gr.update(value=None), error_msg,
            gr.update(value="")
        )

def fetch_current_pricing():
    """Fetch current AWS pricing using the Pricing API"""
    try:
        pricing_client = boto3.client('pricing', region_name='us-east-1')
        
        instance_types = ['ml.g5.4xlarge', 'ml.g5.8xlarge', 'ml.g5.12xlarge', 'ml.g6.4xlarge', 'ml.g6.8xlarge', 'ml.g6e.4xlarge']
        pricing_data = {}
        
        for instance_type in instance_types:
            try:
                response = pricing_client.get_products(
                    ServiceCode='AmazonSageMaker',
                    Filters=[
                        {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                        {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': 'US East (N. Virginia)'},
                        {'Type': 'TERM_MATCH', 'Field': 'productFamily', 'Value': 'ML Instance'}
                    ]
                )
                
                if response['PriceList']:
                    price_data = json.loads(response['PriceList'][0])
                    terms = price_data['terms']['OnDemand']
                    for term_key in terms:
                        price_dimensions = terms[term_key]['priceDimensions']
                        for dim_key in price_dimensions:
                            price_per_hour = float(price_dimensions[dim_key]['pricePerUnit']['USD'])
                            pricing_data[instance_type] = price_per_hour
                            break
                        break
            except Exception as e:
                print(f"Error fetching price for {instance_type}: {e}")
                # Fallback to static pricing
                fallback_prices = {
                    'ml.g5.4xlarge': 1.624, 'ml.g5.8xlarge': 3.248, 'ml.g5.12xlarge': 6.496,
                    'ml.g6.4xlarge': 1.624, 'ml.g6.8xlarge': 3.248, 'ml.g6e.4xlarge': 1.624
                }
                pricing_data[instance_type] = fallback_prices.get(instance_type, 1.624)
        
        return pricing_data
    except Exception as e:
        print(f"Error fetching pricing data: {e}")
        # Return fallback pricing
        return {
            'ml.g5.4xlarge': 1.624, 'ml.g5.8xlarge': 3.248, 'ml.g5.12xlarge': 6.496,
            'ml.g6.4xlarge': 1.624, 'ml.g6.8xlarge': 3.248, 'ml.g6e.4xlarge': 1.624
        }

# Now call the function after it's defined
shared_state.pricing_data = fetch_current_pricing()
print(f"Pricing data loaded: {shared_state.pricing_data}")

def check_aws_credentials():
    try:
        s3 = boto3.client('s3')
        s3.list_buckets()
        print("AWS credentials are valid and working")
    except Exception as e:
        print(f"AWS credentials error: {str(e)}")

def estimate_job_cost(instance_type, duration_seconds, is_spot=False):
    """Estimate the cost of running a job based on instance type and duration"""
    # Use cached pricing data
    pricing_data = getattr(shared_state, 'pricing_data', {})
    
    if instance_type in pricing_data:
        hourly_rate = pricing_data[instance_type]
        if is_spot:
            hourly_rate *= 0.3  # 70% discount for spot instances
        
        hours = duration_seconds / 3600
        estimated_cost = hourly_rate * hours
        
        return f"${estimated_cost:.3f}"
    else:
        return "N/A"

def get_job_metadata(job_id):
    """Fetch job metadata from DynamoDB"""
    try:
        if job_id == 'local':
            return "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>No metadata available for local files</div>"
        
        print(f"Fetching metadata for job_id: {job_id}")
        print(f"Using DynamoDB table: {shared_state.ddb_table_name}")
        
        dynamodb = boto3.resource('dynamodb', region_name=shared_state.aws_region)
        table_name = shared_state.ddb_table_name
        
        # First, check if table exists
        try:
            table = dynamodb.Table(table_name)
            table.load()  # This will raise an exception if table doesn't exist
            print(f"Table {table_name} exists in {shared_state.aws_region}")
        except Exception as table_error:
            print(f"Table error: {str(table_error)}")
            return f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>DynamoDB table '{table_name}' not found or not accessible in {shared_state.aws_region}</div>"
        
        response = table.get_item(Key={'uuid': _validate_uuid(job_id)})
        print(f"DynamoDB response: {response}")
        
        if 'Item' in response:
            item = response['Item']
            # Format metadata as a left-justified table
            sorted_keys = sorted([k for k in item.keys() if k != 'uuid'])
            
            table_rows = []
            
            # Calculate estimated cost if we have the necessary data
            estimated_cost = "N/A"
            if 'instanceType' in item:
                instance_type = str(item['instanceType'])
                is_spot = str(item.get('useSpotInstance', 'false')).lower() == 'true'
                
                try:
                    # Use componentGroupElapsedTime (actual processing time) instead of elapsedTimestamp (includes startup)
                    total_seconds = 0
                    if 'componentGroupElapsedTime' in item:
                        total_seconds = sum(float(t) for t in item['componentGroupElapsedTime'])
                    elif 'elapsedTimestamp' in item:
                        elapsed_str = str(item['elapsedTimestamp'])
                        if ':' in elapsed_str:
                            if 'day' in elapsed_str:
                                days_part, time_part = elapsed_str.split(', ')
                                days = int(days_part.split()[0])
                                time_parts = time_part.split(':')
                                hours = int(time_parts[0]) + (days * 24)
                                minutes = int(time_parts[1])
                                seconds = float(time_parts[2].split('.')[0])
                            else:
                                time_parts = elapsed_str.split(':')
                                if len(time_parts) >= 3:
                                    hours = int(time_parts[0])
                                    minutes = int(time_parts[1])
                                    seconds = float(time_parts[2].split('.')[0])
                                else:
                                    hours = minutes = seconds = 0
                            total_seconds = hours * 3600 + minutes * 60 + seconds
                    
                    if total_seconds > 0:
                        estimated_cost = estimate_job_cost(instance_type, total_seconds, is_spot)
                except Exception as e:
                    print(f"Error calculating cost: {e}")
            
            # Add estimated cost as first row if available
            if estimated_cost != "N/A":
                table_rows.append(f"<tr><td style='padding: 4px 8px; font-weight: bold; color: #333; border-bottom: 1px solid #ccc;'>estimatedCost</td><td style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>{estimated_cost}</td></tr>")
            
            for key in sorted_keys:
                value = item[key]
                # Don't truncate S3 paths and other important values
                if isinstance(value, str) and len(str(value)) > 100 and key not in ['s3', 'inputPrefix', 'outputPrefix', 'bucketName', 'inputKey']:
                    display_value = str(value)[:97] + "..."
                else:
                    display_value = str(value)
                table_rows.append(f"<tr><td style='padding: 4px 8px; font-weight: bold; color: #333; border-bottom: 1px solid #ccc; white-space: nowrap;'>{key}</td><td style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc; word-break: break-all; max-width: 400px;'>{display_value}</td></tr>")
            
            # Create HTML table with proper formatting and wider layout
            table_content = f"<table style='border-collapse: collapse; width: 100%; min-width: 600px;'>{''.join(table_rows)}</table>"
            return f"<h3 style='margin: 0 0 8px 0;'>Job Configuration</h3><div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333; line-height: 1.4; width: 100%; overflow-x: auto;'>{table_content}</div>"
        else:
            error_content = f"No metadata found for job ID: {html.escape(str(job_id))}<br/><br/>This could mean:<br/>- The job hasn't been processed yet<br/>- The job was created before metadata tracking<br/>- The job ID is incorrect"
            return f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333; line-height: 1.4;'>{error_content}</div>"
            
    except Exception as e:
        print(f"Exception in get_job_metadata: {str(e)}")
        import traceback
        traceback.print_exc()
        error_content = f"Error fetching metadata: {str(e)}"
        return f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333; line-height: 1.4;'>{error_content}</div>"

def add_to_favorites(selected_data):
    """Add currently selected item to favorites"""
    try:
        if not selected_data:
            return "No item selected"
        
        # Check if selected_data is a list or array
        if not isinstance(selected_data, (list, tuple)):
            return "Invalid selection format"
            
        # Extract filename and job_id from the selected data
        # selected_data format: [job_id, filename, size, last_modified]
        job_id = selected_data[0]  # First column is job_id
        filename = selected_data[1]  # Second column is filename
        
        # Use the job_id instead of generating a new UUID
        # Create a filename that includes both the original name and job UUID
        name, ext = os.path.splitext(filename)
        favorite_filename = f"{name}_{job_id}{ext}"
        
        # Create favorite data
        favorite = {
            'original_filename': filename,
            'filename': favorite_filename,
            'job_id': job_id,
            'uuid': job_id  # Use job_id as the UUID
        }
        
        # Save to favorites directory
        favorites_dir = os.path.join(os.path.dirname(__file__), "favorites")
        os.makedirs(favorites_dir, exist_ok=True)
        safe_favorite_filename = os.path.basename(favorite_filename)
        favorite_path = os.path.join(favorites_dir, safe_favorite_filename)
        if not os.path.realpath(favorite_path).startswith(os.path.realpath(favorites_dir) + os.sep):
            return "Error: Invalid filename"
        
        # Copy the file to favorites directory
        if job_id == 'local':
            # File is already local, just verify it exists
            # Validate filename to prevent path traversal
            safe_filename = os.path.basename(filename)
            if safe_filename != filename or '..' in filename:
                return "Error: Invalid filename"
            if not os.path.exists(os.path.join(favorites_dir, safe_filename)):
                return f"Error: File {safe_filename} not found in favorites directory"
        else:
            # Download from S3
            try:
                bucket_name = shared_state.s3_bucket
                output_prefix = shared_state.s3_output or "workflow-output"
                file_key = f"{output_prefix}/{job_id}/{filename}"
                
                s3_client = boto3.client('s3')
                s3_client.download_file(bucket_name, file_key, favorite_path)
                print(f"Downloaded favorite to: {favorite_path}")
            except Exception as e:
                return f"Error downloading file: {str(e)}"
        
        return f"Added {filename} to favorites"
        
    except Exception as e:
        print(f"Error in add_to_favorites: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error adding favorite: {str(e)}"

# This is the modified implementation of create_s3_browser_tab
def create_s3_browser_tab():
    with gr.Tab("Viewer & Library"):
        # Create favorites section first
        gr.Markdown("### Local Favorites")
        favorites = load_favorites()
        favorite_buttons = []
        
        with gr.Row(elem_classes="favorites-buttons-row"):
            if not favorites:
                gr.HTML('<div class="no-favorites-text">No favorites yet</div>')
            else:
                for favorite in favorites:
                    with gr.Column(scale=1, min_width=100):
                        display_name = favorite.get('display_name', favorite['filename'])
                        favorite_btn = gr.Button(
                            value=f"📌 {display_name}", 
                            elem_classes=["favorite-button"],
                            size="sm"
                        )
                        favorite_buttons.append((favorite_btn, favorite['path'], favorite['filename']))
        
        # Create hidden components for SOG data and filename tracking
        sog_file_data = gr.Textbox(visible=False)
        current_filename = gr.Textbox(visible=False)
        
        # Create viewer modal
        with Modal(visible=False, elem_id="viewer-modal-content") as viewer_modal:
            with gr.Row():
                gr.Markdown("### 3D Viewer")
                viewer_close_btn = gr.Button("✕ Close", size="sm", elem_classes=["close-button"])
            
            viewer = gr.Model3D(visible=False, label="3D Viewer")
            viewer_status = gr.Textbox(value="", visible=False, label="Viewer Status")
            with gr.Tabs():
                with gr.Tab("3D Splat Viewer"):
                    sog_viewer = gr.HTML(
                        value="<div id='sog-container' style='height: 700px; background: #1a1a1a; border: 1px solid #444; display: flex; align-items: center; justify-content: center; color: white;'>Select a .sog or .spz file to view</div>",
                        label="SOG Viewer"
                    )
                    sog_status = gr.Textbox(
                        label="SOG Viewer Status",
                        interactive=False,
                        show_label=True,
                        value="",
                        lines=3,
                        max_lines=5
                    )
                    spz_viewer = gr.Textbox(value="", visible=False, label="SPZ Data")

                with gr.Tab("Video Preview"):
                    video_viewer = gr.Video(
                        label="Trajectory Video",
                        height=700,
                        interactive=False,
                        format="mp4"
                    )
                    video_status = gr.Textbox(
                        label="Video Status",
                        interactive=False,
                        show_label=True,
                        value="",
                        lines=1,
                        max_lines=1
                    )
        
        # Connect favorite button handlers to open viewer modal
        def open_favorite_in_modal(path, name):
            favorites_dir = os.path.join(os.path.dirname(__file__), "favorites")
            if not os.path.realpath(path).startswith(os.path.realpath(favorites_dir) + os.sep):
                return (gr.update(), gr.update(), "", "", gr.update(), gr.update(), "", "", "")
            if name.lower().endswith('.sog'):
                import base64
                with open(path, 'rb') as f:
                    file_data = base64.b64encode(f.read()).decode('utf-8')
                payload = json.dumps({"data": file_data, "filename": name, "ts": __import__('time').time()})
                return (
                    gr.update(visible=True),
                    gr.update(value=None),
                    "",
                    payload,
                    gr.update(),
                    gr.update(value=None),
                    "",
                    name,
                    ""
                )
            elif name.lower().endswith('.spz'):
                import base64
                with open(path, 'rb') as f:
                    file_data = base64.b64encode(f.read()).decode('utf-8')
                payload = json.dumps({"data": file_data, "filename": name, "ts": __import__('time').time()})
                return (
                    gr.update(visible=True),
                    gr.update(value=None),
                    "",
                    "",
                    gr.update(),
                    gr.update(value=None),
                    "",
                    name,
                    payload
                )
            else:
                return (
                    gr.update(visible=True),
                    gr.update(value=path),
                    f"Loaded: {name}",
                    "",
                    gr.update(),
                    gr.update(value=None),
                    "",
                    name,
                    ""
                )
        
        for btn, path, name in favorite_buttons:
            btn.click(
                fn=lambda p=path, n=name: open_favorite_in_modal(p, n),
                inputs=[],
                outputs=[
                    viewer_modal,
                    viewer,
                    viewer_status,
                    sog_file_data,
                    sog_viewer,
                    video_viewer,
                    video_status,
                    current_filename,
                    spz_viewer
                ]
            )
        
        # Store the current model URL
        current_model = gr.State(None)

        # File browser section
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### S3 Contents")
                refresh_button = gr.Button(
                    "Refresh Input Contents", 
                    variant="primary",
                    size="sm"
                )
                with gr.Row():
                    download_btn = gr.Button(
                        "Download Selected", 
                        interactive=False,
                        size="sm"
                    )
                    add_favorite_btn = gr.Button(
                        "Add to Favorites", 
                        interactive=False,
                        size="sm"
                    )
                    refine_btn = gr.Button(
                        "Refine Splat (Resume Training)", 
                        interactive=False,
                        size="sm"
                    )
                    view_btn = gr.Button(
                        "View Selected", 
                        interactive=False,
                        size="sm"
                    )
                files_box = gr.Dataframe(
                    headers=["Job ID", "Filename", "Size", "Last Modified", "Thumbnail"],
                    interactive=False,
                    value=[],
                    visible=True,
                    elem_id="files_table",
                    datatype=["str", "str", "str", "str", "html"],
                    wrap=True
                )

                selected_data = gr.State(None)
        
        download_iframe = gr.HTML(visible=True)

        # Create refine modal dialog with custom CSS for popup
        with Modal(visible=False, elem_id="refine-modal-content") as refine_modal:
            gr.Markdown("### Refine Splat Settings", elem_classes="padded-markdown")
            gr.Markdown("Review and modify settings for refinement job:", elem_classes="padded-markdown")
            
            refine_instance = gr.Dropdown(
                label="Instance Type",
                choices=["ml.g5.4xlarge", "ml.g5.8xlarge", "ml.g5.12xlarge", "ml.g6.4xlarge", "ml.g6.8xlarge", "ml.g6e.4xlarge"],
                value=shared_state.instance
            )
            refine_compute = gr.Radio(
                label="Compute Type",
                choices=[("AWS Batch (Spot Instances - Up to 50% cost savings)", "true"), ("SageMaker (On-Demand)", "false")],
                value=shared_state.use_spot_instance
            )
            refine_crop = gr.Radio(
                label="Crop Output Bounds",
                choices=["true", "false"],
                value=shared_state.crop_output_bounds
            )
            refine_crop_mode = gr.Dropdown(
                label="Crop Mode",
                choices=["environment", "rigid_body"],
                value=shared_state.crop_mode
            )
            refine_isp_3d = gr.Dropdown(
                label="3D ISP Mode",
                choices=[
                    ("None", "none"),
                    ("Bilateral Grid (bilagrid)", "bilagrid"),
                    ("Per-Pixel ISP (ppisp)", "ppisp")
                ],
                value=shared_state.isp_3d,
                info="Not applicable to nerfacto."
            )
            refine_clean_splat = gr.Radio(
                label="Clean Splat (Remove Noise)",
                choices=["true", "false"],
                value=shared_state.clean_splat
            )
            refine_enable_spz = gr.Radio(
                label="Enable SPZ Export",
                choices=["true", "false"],
                value=shared_state.enable_spz
            )
            refine_enable_sog = gr.Radio(
                label="Enable SOG Export",
                choices=["true", "false"],
                value=shared_state.enable_sog
            )
            refine_enable_usdz = gr.Radio(
                label="Enable USDZ Export",
                choices=["true", "false"],
                value=shared_state.enable_usdz
            )
            refine_ply_coords = gr.Dropdown(
                label="PLY Coordinate System",
                choices=[
                    ("Right-Hand, Y-Up (playcanvas)", "rhyu"),
                    ("Left-Hand, Y-Up (babylon.js)", "lhyu"),
                    ("Right-Hand, Z-Up (blender)", "rhzu"),
                    ("Left-Hand, Z-Up (unreal)", "lhzu")
                ],
                value=shared_state.ply_coords
            )
            refine_spz_coords = gr.Dropdown(
                label="SPZ Coordinate System",
                choices=[
                    ("Right-Hand, Y-Up (playcanvas)", "rhyu"),
                    ("Left-Hand, Y-Up (babylon.js)", "lhyu"),
                    ("Right-Hand, Z-Up (blender)", "rhzu"),
                    ("Left-Hand, Z-Up (unreal)", "lhzu")
                ],
                value=shared_state.spz_coords
            )
            refine_sog_coords = gr.Dropdown(
                label="SOG Coordinate System",
                choices=[
                    ("Right-Hand, Y-Up (playcanvas)", "rhyu"),
                    ("Left-Hand, Y-Up (babylon.js)", "lhyu"),
                    ("Right-Hand, Z-Up (blender)", "rhzu"),
                    ("Left-Hand, Z-Up (unreal)", "lhzu")
                ],
                value=shared_state.sog_coords
            )
            refine_usdz_coords = gr.Dropdown(
                label="USDZ Coordinate System",
                choices=[
                    ("Right-Hand, Y-Up (playcanvas)", "rhyu"),
                    ("Left-Hand, Y-Up (babylon.js)", "lhyu"),
                    ("Right-Hand, Z-Up (blender)", "rhzu"),
                    ("Left-Hand, Z-Up (unreal)", "lhzu")
                ],
                value=shared_state.usdz_coords
            )
            
            with gr.Row():
                refine_cancel_btn = gr.Button("Cancel", variant="secondary")
                refine_submit_btn = gr.Button("Submit Refinement Job", variant="primary")
        
        # Create action status for both refine and favorites
        action_status = gr.Textbox(
            label="Action Status",
            value="",
            visible=True,
            lines=3
        )

        # Create cost display before using it
        cost_display = gr.HTML(
            value="<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Click 'Refresh Input Contents' to calculate costs</div>",
            label="Job Costs"
        )
        
        # Connect refresh button to refresh function with cost calculation
        refresh_button.click(
            fn=refresh_and_update,
            inputs=[],
            outputs=[files_box, selected_data, download_btn, add_favorite_btn, refine_btn, cost_display]
        )

        # Create metadata display AFTER the files table (moved to the bottom)
        metadata_display = gr.HTML(
            value="Select a job to view metadata",
            label="Job Configuration"
        )
        
        # Update the select event handler (no longer opens viewer automatically)
        files_box.select(
            fn=on_select,
            inputs=[files_box],
            outputs=[
                selected_data,
                download_btn,
                view_btn,
                add_favorite_btn,
                refine_btn,
                metadata_display
            ]
        )
        
        # View button opens the viewer modal
        def open_viewer_modal(selected_data_val):
            if not selected_data_val:
                return tuple([gr.update(visible=False)] + [gr.update() for _ in range(7)])
            
            viewer_result = handle_view_multi(selected_data_val)
            filename = selected_data_val[1].replace("  └─ ", "")
            return (
                                gr.update(visible=True),
                viewer_result[0],  # viewer
                viewer_result[1],  # viewer_status
                viewer_result[2],  # sog_file_data
                viewer_result[3],  # sog_status
                viewer_result[4],  # video_viewer
                viewer_result[5],  # video_status
                viewer_result[6],  # spz_viewer
                filename
            )
        
        view_btn.click(
            fn=open_viewer_modal,
            inputs=[selected_data],
            outputs=[
                viewer_modal,
                viewer,
                viewer_status,
                sog_file_data,
                sog_status,
                video_viewer,
                video_status,
                spz_viewer,
                current_filename
            ]
        )
        
        # Add JavaScript tab switching when filename changes
        current_filename.change(
            fn=None,
            inputs=[current_filename],
            js="""(filename) => {
                if(filename) {
                    const fname = filename.toLowerCase();
                    setTimeout(() => {
                        const allTabs = document.querySelectorAll('#viewer-modal-content button[role="tab"]');
                        allTabs.forEach((tab) => {
                            const tabText = tab.textContent.trim();
                            if(fname.endsWith('.mp4') && tabText.includes('Video Preview')) {
                                tab.click();
                            } else if(!fname.endsWith('.mp4') && tabText.includes('3D Splat Viewer')) {
                                tab.click();
                            }
                        });
                    }, 200);
                }
            }"""
        )
        
        viewer_close_btn.click(
            fn=lambda: gr.update(visible=False),
            outputs=[viewer_modal]
        )
        
        # Connect download button
        download_btn.click(
            fn=handle_download,
            inputs=[selected_data],
            outputs=[download_iframe]
        )
        download_iframe.change(
            fn=None,
            inputs=[download_iframe],
            outputs=None,
            js="""(url) => {
                if (url) { const a = document.createElement('a'); a.href = url; a.download = ''; document.body.appendChild(a); a.click(); document.body.removeChild(a); }
            }"""
        )
        
        # Favorite buttons already set filename in open_favorite_in_modal, no need for second handler
        
        # Store SOG data and trigger viewer
        # Connect add to favorites button
        add_to_favorites_btn.click(
            fn=add_to_favorites,
            inputs=[selected_data],
            outputs=[action_status]
        )
        
        # Show modal when refine button is clicked
        def show_refine_modal(selected_data):
            if not selected_data:
                return gr.update(visible=False), "No file selected"
            return gr.update(visible=True), ""
        
        refine_btn.click(
            fn=show_refine_modal,
            inputs=[selected_data],
            outputs=[refine_modal, action_status]
        )
        
        # Hide modal on cancel
        refine_cancel_btn.click(
            fn=lambda: gr.update(visible=False),
            outputs=[refine_modal]
        )
        
        # Submit refinement job with user selections
        def submit_refine_job(selected_data, instance, compute, crop, crop_mode, clean_splat, spz, sog, usdz, ply_coords, spz_coords, sog_coords, usdz_coords, isp_3d):
            result = refine_splat(selected_data, instance, compute, crop, crop_mode, clean_splat, spz, sog, usdz, ply_coords, spz_coords, sog_coords, usdz_coords, isp_3d)
            return gr.update(visible=False), result
        
        refine_submit_btn.click(
            fn=submit_refine_job,
            inputs=[selected_data, refine_instance, refine_compute, refine_crop, refine_crop_mode, refine_clean_splat, refine_enable_spz, refine_enable_sog, refine_enable_usdz, refine_ply_coords, refine_spz_coords, refine_sog_coords, refine_usdz_coords, refine_isp_3d],
            outputs=[refine_modal, action_status]
        )
        
        # Add SuperSplat link at the bottom of the viewer tab
        supersplat_link = gr.HTML(
            '<div style="text-align:center; margin:10px 0;"><a href="https://superspl.at/editor" target="_blank" style="display:inline-block; background:#f97316; color:white; padding:8px 16px; text-decoration:none; border-radius:6px; font-size:14px; font-weight:500;">🚀 Open SuperSplat Editor</a></div>'
        )
                

def get_job_output_files(job_id):
    """Get list of output files with metadata for a job from DynamoDB"""
    try:
        dynamodb = boto3.resource('dynamodb', region_name=shared_state.aws_region)
        table = dynamodb.Table(shared_state.ddb_table_name)
        response = table.get_item(Key={'uuid': _validate_uuid(job_id)})
        
        if 'Item' not in response:
            return []
        
        item = response['Item']
        output_files = item.get('outputFiles', [])
        
        return output_files
    except Exception as e:
        print(f"Error getting job output files: {e}")
        return []

def calculate_job_costs_from_jobs(jobs_data):
    """Calculate costs from job progress data"""
    try:
        if not jobs_data:
            return "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>No jobs to calculate costs for</div>"
        
        dynamodb = boto3.resource('dynamodb', region_name=shared_state.aws_region)
        table = dynamodb.Table(shared_state.ddb_table_name)
        
        cost_rows = []
        total_cost = 0
        total_duration = 0
        job_count = 0
        
        display_jobs = jobs_data[:50]
        
        for job_row in display_jobs:
            if len(job_row) < 1:
                continue
            job_id = job_row[0]
            try:
                response = table.get_item(Key={'uuid': job_id})
                if 'Item' in response:
                    item = response['Item']
                    instance_type = str(item.get('instanceType', 'ml.g5.4xlarge'))
                    elapsed_str = str(item.get('elapsedTimestamp', '0:00:00'))
                    is_spot = str(item.get('useSpotInstance', 'false')).lower() == 'true'
                    
                    total_seconds = 0
                    try:
                        # Use componentGroupElapsedTime if available
                        if 'componentGroupElapsedTime' in item:
                            total_seconds = sum(float(t) for t in item['componentGroupElapsedTime'])
                        elif ':' in elapsed_str:
                            if 'day' in elapsed_str:
                                days_part, time_part = elapsed_str.split(', ')
                                days = int(days_part.split()[0])
                                time_parts = time_part.split(':')
                                hours = int(time_parts[0]) + (days * 24)
                                minutes = int(time_parts[1])
                                seconds = float(time_parts[2].split('.')[0])
                            else:
                                time_parts = elapsed_str.split(':')
                                if len(time_parts) >= 3:
                                    hours = int(time_parts[0])
                                    minutes = int(time_parts[1])
                                    seconds = float(time_parts[2].split('.')[0])
                                else:
                                    hours = minutes = seconds = 0
                            total_seconds = hours * 3600 + minutes * 60 + seconds
                    except Exception as e:
                        print(f"Error parsing elapsed time '{elapsed_str}': {e}")
                        total_seconds = 0
                    
                    if total_seconds > 0:
                        total_duration += total_seconds
                        job_count += 1
                    
                    cost = estimate_job_cost(instance_type, total_seconds, is_spot)
                    if cost != "N/A":
                        cost_value = float(cost.replace('$', ''))
                        total_cost += cost_value
                    
                    compute_type = "Batch Spot" if is_spot else "SageMaker"
                    duration = f"{total_seconds//3600:.0f}h {(total_seconds%3600)//60:.0f}m" if total_seconds > 0 else "N/A"
                    
                    cost_rows.append(f"<tr><td style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>{job_id[:8]}...</td><td style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>{instance_type}</td><td style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>{compute_type}</td><td style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>{duration}</td><td style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>{cost}</td></tr>")
            except Exception as e:
                print(f"Error getting metadata for job {job_id}: {e}")
                continue
        
        if not cost_rows:
            return "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>No job cost data available</div>"
        
        avg_duration = total_duration / job_count if job_count > 0 else 0
        avg_duration_str = f"{avg_duration//3600:.0f}h {(avg_duration%3600)//60:.0f}m" if avg_duration > 0 else "N/A"
        
        display_note = f" (showing first 50 of {len(jobs_data)})" if len(jobs_data) > 50 else ""
        
        summary_rows = []
        summary_rows.append(f"<tr style='font-weight: bold; background-color: #f0f0f0;'><td colspan='4' style='padding: 4px 8px; color: #333; border-bottom: 2px solid #333;'>Total Cost</td><td style='padding: 4px 8px; color: #333; border-bottom: 2px solid #333;'>${total_cost:.3f}</td></tr>")
        summary_rows.append(f"<tr style='font-weight: bold; background-color: #f8f8f8;'><td colspan='3' style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>Average Duration</td><td style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>{avg_duration_str}</td><td style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>-</td></tr>")
        
        all_rows = summary_rows + cost_rows
        
        table_content = f"<div id='job-costs-table' style='max-height: 400px; overflow-y: auto; border: 1px solid #ccc;'><table style='border-collapse: collapse; width: 100%;'><thead style='position: sticky; top: 0; background-color: #f5f5f5;'><tr><th style='padding: 4px 8px; font-weight: bold; color: #333; border-bottom: 1px solid #ccc;'>Job ID</th><th style='padding: 4px 8px; font-weight: bold; color: #333; border-bottom: 1px solid #ccc;'>Instance</th><th style='padding: 4px 8px; font-weight: bold; color: #333; border-bottom: 1px solid #ccc;'>Compute</th><th style='padding: 4px 8px; font-weight: bold; color: #333; border-bottom: 1px solid #ccc;'>Duration</th><th style='padding: 4px 8px; font-weight: bold; color: #333; border-bottom: 1px solid #ccc;'>Cost</th></tr></thead><tbody>{''.join(all_rows)}</tbody></table></div>"
        
        return f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333; line-height: 1.4; display: inline-block;'>{table_content}</div>"
        
    except Exception as e:
        print(f"Error calculating job costs: {e}")
        return f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Error calculating costs: {str(e)}</div>"

def get_job_progress_data():
    """Query DynamoDB for job progress monitoring data"""
    try:
        import datetime
        dynamodb = boto3.resource('dynamodb', region_name=shared_state.aws_region)
        table = dynamodb.Table(shared_state.ddb_table_name)
        
        # Scan the table to get all jobs
        response = table.scan()
        jobs_data = []
        
        for item in response['Items']:
            job_id = str(item.get('uuid', 'N/A'))
            
            # Use uuidStatus field only
            status = str(item.get('uuidStatus', 'N/A'))
            
            # Normalize status string for display
            if status.lower() == 'complete':
                status = 'Complete'
            elif status.lower() == 'in-progress':
                status = 'In-Progress'
            
            # Truncate start time to show only date and hour:minute
            start_time_raw = str(item.get('startTimestamp', 'N/A'))
            if start_time_raw != 'N/A' and len(start_time_raw) > 16:
                # Format: 2025-01-15 10:30 (truncate seconds and microseconds)
                start_time = start_time_raw[:16]
            else:
                start_time = start_time_raw
            instance_type = str(item.get('instanceType', 'N/A'))
            is_spot = str(item.get('useSpotInstance', 'false')).lower() == 'true'
            compute_type = "Batch Spot" if is_spot else "SageMaker"
            
            # Extract model name from training config
            model_name = "N/A"
            if 'training' in item and 'model' in item['training']:
                model_name = str(item['training']['model'])
            elif 'model' in item:
                model_name = str(item['model'])
            
            # Extract reconstruction software from reconstruction config
            reconstruction_software = "N/A"
            if 'reconSoftwareName' in item:
                reconstruction_software = str(item['reconSoftwareName'])
            elif 'reconstruction' in item and 'softwareName' in item['reconstruction']:
                reconstruction_software = str(item['reconstruction']['softwareName'])
            
            # Extract media filename from DynamoDB and truncate
            # Prioritize 'originalMediaFilename' field for refined jobs, then 'filename'
            media_filename = "N/A"
            if 'originalMediaFilename' in item:
                media_filename = str(item['originalMediaFilename'])
            elif 'filename' in item and str(item['filename']) != 'model.tar.gz':
                media_filename = str(item['filename'])
            elif 's3Input' in item and 'model.tar.gz' in str(item['s3Input']):
                # For old refined jobs, extract parent job ID from s3Input path
                s3_input = str(item['s3Input'])
                # Format: s3://bucket/workflow-output/PARENT_JOB_ID/output/model.tar.gz
                parts = s3_input.split('/')
                if len(parts) >= 5:
                    parent_job_id = parts[-3]  # Get parent job ID
                    try:
                        parent_response = table.get_item(Key={'uuid': parent_job_id})
                        if 'Item' in parent_response:
                            parent_item = parent_response['Item']
                            if 'originalMediaFilename' in parent_item:
                                media_filename = str(parent_item['originalMediaFilename'])
                            elif 'filename' in parent_item and str(parent_item['filename']) != 'model.tar.gz':
                                media_filename = str(parent_item['filename'])
                            elif 's3' in parent_item and 'inputKey' in parent_item['s3']:
                                media_filename = str(parent_item['s3']['inputKey']).split('/')[-1]
                            elif 'inputKey' in parent_item:
                                media_filename = str(parent_item['inputKey']).split('/')[-1]
                    except:
                        pass
            elif 's3' in item and 'inputKey' in item['s3']:
                s3_key = str(item['s3']['inputKey'])
                media_filename = s3_key.split('/')[-1]
            elif 'inputKey' in item:
                s3_key = str(item['inputKey'])
                media_filename = s3_key.split('/')[-1]
            
            # Truncate media filename to 20 characters
            if len(media_filename) > 20:
                media_filename = media_filename[:17] + "..."
            
            # Calculate elapsed time
            elapsed_time = "N/A"
            if status.lower() == 'in-progress':
                try:
                    # Use startTimestamp from DynamoDB for in-progress jobs
                    start_timestamp = item.get('startTimestamp')
                    if start_timestamp:
                        # Parse timestamp format: 2025-08-12T05:55:43.553226
                        start_dt = datetime.datetime.fromisoformat(str(start_timestamp))
                        current_dt = datetime.datetime.now()
                        elapsed_delta = current_dt - start_dt
                        hours = int(elapsed_delta.total_seconds() // 3600)
                        minutes = int((elapsed_delta.total_seconds() % 3600) // 60)
                        elapsed_time = f"{hours}h {minutes}m"
                except Exception as e:
                    print(f"Error calculating elapsed time for {job_id}: {e}")
                    elapsed_time = "N/A"
            else:
                # Use elapsedTimestamp for completed jobs
                try:
                    elapsed_str = str(item.get('elapsedTimestamp', '0:00:00'))
                    total_seconds = 0
                    if ':' in elapsed_str:
                        if 'day' in elapsed_str:
                            days_part, time_part = elapsed_str.split(', ')
                            days = int(days_part.split()[0])
                            time_parts = time_part.split(':')
                            hours = int(time_parts[0]) + (days * 24)
                            minutes = int(time_parts[1])
                            seconds = float(time_parts[2].split('.')[0])
                        else:
                            time_parts = elapsed_str.split(':')
                            if len(time_parts) >= 3:
                                hours = int(time_parts[0])
                                minutes = int(time_parts[1])
                                seconds = float(time_parts[2].split('.')[0])
                            else:
                                hours = minutes = seconds = 0
                        total_seconds = hours * 3600 + minutes * 60 + seconds
                    
                    if total_seconds > 0:
                        hours = int(total_seconds // 3600)
                        minutes = int((total_seconds % 3600) // 60)
                        elapsed_time = f"{hours}h {minutes}m"
                    else:
                        elapsed_time = "N/A"
                except Exception as e:
                    print(f"Error calculating elapsed time for job {job_id}: {e}")
                    elapsed_time = "N/A"
            
            # Extract evaluation metrics
            metrics_str = "N/A"
            if 'evaluationMetrics' in item:
                metrics = item['evaluationMetrics']
                if isinstance(metrics, dict):
                    parts = []
                    if 'ssim' in metrics:
                        parts.append(f"SSIM: {float(metrics['ssim']):.3f}")
                    if 'lpips' in metrics:
                        parts.append(f"LPIPS: {float(metrics['lpips']):.3f}")
                    if 'psnr' in metrics:
                        parts.append(f"PSNR: {float(metrics['psnr']):.2f}")
                    metrics_str = " | ".join(parts) if parts else "N/A"
            
            jobs_data.append([
                job_id,              # Full job ID (UUID)
                media_filename,      # Media input filename
                status,
                start_time,
                compute_type,
                instance_type,
                model_name,          # Model name
                reconstruction_software,  # Reconstruction software
                elapsed_time,
                metrics_str          # Training metrics
            ])
        
        # Sort by start time (most recent first) - start time is now at index 3
        jobs_data.sort(key=lambda x: x[3] if len(x) > 3 else "", reverse=True)
        
        return jobs_data
        
    except Exception as e:
        print(f"Error fetching job progress data: {e}")
        return []

def calculate_job_costs(files_data):
    """Calculate costs for all jobs in the table"""
    try:
        if not files_data:
            return "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>No jobs to calculate costs for</div>"
        
        dynamodb = boto3.resource('dynamodb', region_name=shared_state.aws_region)
        table = dynamodb.Table(shared_state.ddb_table_name)
        
        cost_rows = []
        total_cost = 0
        total_duration = 0
        job_count = 0
        
        # Group files by job ID to avoid duplicates
        job_ids_seen = set()
        unique_jobs = []
        for file_row in files_data:
            if len(file_row) < 2:
                continue
            job_id = file_row[0]
            if job_id not in job_ids_seen:
                job_ids_seen.add(job_id)
                unique_jobs.append(file_row)
        
        # Limit to first 50 unique jobs for display
        display_files = unique_jobs[:50]
        
        for file_row in display_files:
            job_id = file_row[0]
            try:
                response = table.get_item(Key={'uuid': job_id})
                if 'Item' in response:
                    item = response['Item']
                    instance_type = str(item.get('instanceType', 'ml.g5.4xlarge'))
                    elapsed_str = str(item.get('elapsedTimestamp', '0:00:00'))
                    is_spot = str(item.get('useSpotInstance', 'false')).lower() == 'true'
                    
                    # Parse elapsed time
                    total_seconds = 0
                    try:
                        # Use componentGroupElapsedTime if available
                        if 'componentGroupElapsedTime' in item:
                            total_seconds = sum(float(t) for t in item['componentGroupElapsedTime'])
                        elif ':' in elapsed_str:
                            # Handle format like "1 day, 9:46:28" or "3 days, 9:46:28" or "0:46:28"
                            if 'day' in elapsed_str:
                                # Parse "X day(s), H:MM:SS" format
                                days_part, time_part = elapsed_str.split(', ')
                                days = int(days_part.split()[0])
                                time_parts = time_part.split(':')
                                hours = int(time_parts[0]) + (days * 24)
                                minutes = int(time_parts[1])
                                seconds = float(time_parts[2].split('.')[0])
                            else:
                                # Parse "H:MM:SS" format
                                time_parts = elapsed_str.split(':')
                                if len(time_parts) >= 3:
                                    hours = int(time_parts[0])
                                    minutes = int(time_parts[1])
                                    seconds = float(time_parts[2].split('.')[0])
                                else:
                                    hours = minutes = seconds = 0
                            total_seconds = hours * 3600 + minutes * 60 + seconds
                    except Exception as e:
                        print(f"Error parsing elapsed time '{elapsed_str}' for cost calculation: {e}")
                        total_seconds = 0
                    
                    if total_seconds > 0:
                        total_duration += total_seconds
                        job_count += 1
                    
                    cost = estimate_job_cost(instance_type, total_seconds, is_spot)
                    if cost != "N/A":
                        cost_value = float(cost.replace('$', ''))
                        total_cost += cost_value
                    
                    compute_type = "Batch Spot" if is_spot else "SageMaker"
                    duration = f"{total_seconds//3600:.0f}h {(total_seconds%3600)//60:.0f}m" if total_seconds > 0 else "N/A"
                    
                    cost_rows.append(f"<tr><td style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>{job_id[:8]}...</td><td style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>{instance_type}</td><td style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>{compute_type}</td><td style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>{duration}</td><td style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>{cost}</td></tr>")
            except Exception as e:
                print(f"Error getting metadata for job {job_id}: {e}")
                continue
        
        if not cost_rows:
            return "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>No job cost data available</div>"
        
        # Calculate average duration
        avg_duration = total_duration / job_count if job_count > 0 else 0
        avg_duration_str = f"{avg_duration//3600:.0f}h {(avg_duration%3600)//60:.0f}m" if avg_duration > 0 else "N/A"
        
        # Add note if showing limited results
        display_note = f" (showing first 50 of {len(files_data)})" if len(files_data) > 50 else ""
        
        # Create summary rows first
        summary_rows = []
        summary_rows.append(f"<tr style='font-weight: bold; background-color: #f0f0f0;'><td colspan='4' style='padding: 4px 8px; color: #333; border-bottom: 2px solid #333;'>Total Cost</td><td style='padding: 4px 8px; color: #333; border-bottom: 2px solid #333;'>${total_cost:.3f}</td></tr>")
        summary_rows.append(f"<tr style='font-weight: bold; background-color: #f8f8f8;'><td colspan='3' style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>Average Duration</td><td style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>{avg_duration_str}</td><td style='padding: 4px 8px; color: #333; border-bottom: 1px solid #ccc;'>-</td></tr>")
        
        # Combine summary rows first, then individual job rows
        all_rows = summary_rows + cost_rows
        
        table_content = f"<div id='job-costs-table' style='max-height: 400px; overflow-y: auto; border: 1px solid #ccc;'><table style='border-collapse: collapse; width: 100%;'><thead style='position: sticky; top: 0; background-color: #f5f5f5;'><tr><th style='padding: 4px 8px; font-weight: bold; color: #333; border-bottom: 1px solid #ccc;'>Job ID</th><th style='padding: 4px 8px; font-weight: bold; color: #333; border-bottom: 1px solid #ccc;'>Instance</th><th style='padding: 4px 8px; font-weight: bold; color: #333; border-bottom: 1px solid #ccc;'>Compute</th><th style='padding: 4px 8px; font-weight: bold; color: #333; border-bottom: 1px solid #ccc;'>Duration</th><th style='padding: 4px 8px; font-weight: bold; color: #333; border-bottom: 1px solid #ccc;'>Cost</th></tr></thead><tbody>{''.join(all_rows)}</tbody></table></div>"
        
        return f"""<h3 style='margin: 0 0 8px 0;'>Job Costs{display_note}</h3>
        <div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333; line-height: 1.4; display: inline-block;'>
        {table_content}
        </div>"""
        
        return f"""<h3 style='margin: 0 0 8px 0;'>Job Costs</h3>
        <div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333; line-height: 1.4; display: inline-block;'>
        {table_content}
        </div>"""
        
    except Exception as e:
        print(f"Error calculating job costs: {e}")
        return f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Error calculating costs: {str(e)}</div>"

def refresh_and_update():
    """Refresh S3 contents and update the DataFrame"""
    try:
        files = refresh_s3_contents()
        
        if isinstance(files, dict) and 'data' in files:
            files = files['data']
        
        # Ensure we always return a list, even if empty
        if files is None:
            files = []
        
        # Calculate job costs
        job_costs_html = calculate_job_costs(files)
            
        # Reset the selection state when refreshing data
        print("Refreshed data, resetting selection state")
        return files, None, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), job_costs_html
    except Exception as e:
        print(f"Error in refresh_and_update: {str(e)}")
        return [], None, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), "Error calculating costs"

def add_to_favorites_and_reload(selected_data):
    """Add to favorites and force a page reload"""
    status = add_to_favorites(selected_data)
    
    # Create a JavaScript snippet that will reload the page
    reload_html = """
    <script>
    // Force a complete page reload (not from cache)
    setTimeout(function() {
        window.location.href = window.location.href + '?t=' + new Date().getTime();
    }, 1000);
    </script>
    """
    
    gr.Info("Adding to favorites...the Favorites list will be updated when the local website is re-launched.")
    return gr.HTML(reload_html)


def handle_download(selected_data):
    try:
        if not selected_data:
            return ""
            
        job_id = selected_data[0]
        filename = selected_data[1]
        
        bucket_name = shared_state.s3_bucket
        output_prefix = shared_state.s3_output or "workflow-output"
        file_key = f"{output_prefix}/{job_id}/{filename}"
        
        s3_client = boto3.client('s3')
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': file_key,
                'ResponseContentDisposition': f'attachment; filename="{filename}"'
            },
            ExpiresIn=3600
        )
        return url
        
    except Exception as e:
        print(f"Error generating download URL: {str(e)}")
        return ""

def generate_presigned_url(bucket_name, key, expiration=3600):
    """Generate a presigned URL for downloading an S3 object"""
    try:
        s3_client = boto3.client('s3')
        
        # Set appropriate content type based on file extension
        params = {
            'Bucket': bucket_name,
            'Key': key
        }
        
        # For SPZ files, don't set any response headers to let S3 serve them naturally
        #if not key.lower().endswith('.spz'):
        params['ResponseContentType'] = 'application/octet-stream'
        
        url = s3_client.generate_presigned_url(
            'get_object',
            Params=params,
            ExpiresIn=expiration
        )
        #print(f"[DEBUG] Generated presigned URL for {key}")
        return url
    except Exception as e:
        print(f"Error generating presigned URL: {str(e)}")
        return None

def create_credentials_tab():
    with gr.Tab("AWS Credentials"):
        with gr.Column():
            gr.Markdown("### AWS Credentials")
            gr.Markdown("Paste your AWS credentials exactly as shown from PowerShell.\nAll environment variables should be on one line, separated by spaces:")
            aws_creds = gr.Textbox(
                label="AWS Credentials",
                placeholder='$Env:ISENGARD_PRODUCTION_ACCOUNT="123" $Env:AWS_ACCESS_KEY_ID="ASIA..." $Env:AWS_SECRET_ACCESS_KEY="abcd..." $Env:AWS_SESSION_TOKEN="IQoJ..."',
                value="",
                type="password",
                lines=1  # Single line input
            )
            gr.Markdown("""Tips:
    1. Copy and paste all variables from PowerShell
    2. Make sure there are spaces between each variable
    3. Keep everything on one line""")
            creds_submit_btn = gr.Button("Update Credentials", elem_classes="orange-button")
            creds_status = gr.Textbox(label="Credentials Status", value="")
    creds_submit_btn.click(
        parse_aws_credentials,
        inputs=[aws_creds],
        outputs=[creds_status]
    )

def cleanup_local_files():
    """Clean up local temporary files and favorites to free disk space"""
    try:
        import shutil
        import tempfile
        
        cleanup_report = []
        total_freed = 0
        
        # Clean Gradio temp files
        gradio_temp = "/tmp/gradio"
        if os.path.exists(gradio_temp):
            size_before = sum(os.path.getsize(os.path.join(dirpath, filename))
                            for dirpath, dirnames, filenames in os.walk(gradio_temp)
                            for filename in filenames) / (1024*1024)
            shutil.rmtree(gradio_temp)
            cleanup_report.append(f"Cleared Gradio temp files: {size_before:.1f} MB")
            total_freed += size_before
        
        # Skip cleaning favorites directory - preserve user's saved files
        
        cleanup_report.append(f"\nTotal space freed: {total_freed:.1f} MB")
        return "\n".join(cleanup_report)
        
    except Exception as e:
        return f"Error during cleanup: {str(e)}"

def create_debug_tab():
    # Combined Debug Tab
    with gr.Tab("🔧 Debug"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### AWS Credentials")
                gr.Markdown("Paste your AWS credentials exactly as shown from PowerShell.\nAll environment variables should be on one line, separated by spaces:")
                aws_creds = gr.Textbox(
                    label="AWS Credentials",
                    placeholder='$Env:ISENGARD_PRODUCTION_ACCOUNT="123" $Env:AWS_ACCESS_KEY_ID="ASIA..." $Env:AWS_SECRET_ACCESS_KEY="abcd..." $Env:AWS_SESSION_TOKEN="IQoJ..."',
                    value="",
                    type="password",
                    lines=1
                )
                gr.Markdown("""Tips:
    1. Copy and paste all variables from PowerShell
    2. Make sure there are spaces between each variable
    3. Keep everything on one line""")
                creds_submit_btn = gr.Button("Update Credentials", elem_classes="orange-button")
                creds_status = gr.Textbox(label="Credentials Status", value="")
            
            with gr.Column():
                gr.Markdown("### Debug Tools")
                preview_btn = gr.Button("Preview JSON", elem_classes="orange-button")
                cleanup_btn = gr.Button("🗑️ Clean Local Files", variant="secondary")
                debug_output = gr.Textbox(label="JSON Preview", lines=15)
                cleanup_output = gr.Textbox(label="Cleanup Results", lines=5)

                def preview_json_with_shared_state():
                    # Handle the faces value - convert from string to list if needed
                    faces = shared_state.faces
                    if not isinstance(faces, list):
                        try:
                            # Try to convert string to list if it's a string representation of a list
                            if isinstance(faces, str) and (faces.startswith('[') or faces.startswith('[')):
                                import ast
                                faces = ast.literal_eval(faces)
                            else:
                                faces = []
                        except:
                            faces = []
                    
                    return preview_json(
                        shared_state.s3_bucket,
                        shared_state.s3_input or "workflow-input",
                        shared_state.s3_output or "workflow-output",
                        None,  # video_file will be None for preview
                        shared_state.instance,
                        shared_state.sfm,
                        shared_state.model,
                        faces,
                        shared_state.bg_model,
                        shared_state.filter_blurry,
                        shared_state.max_images,
                        shared_state.sfm_enable,
                        shared_state.enhanced_feature,
                        shared_state.matching_method,
                        shared_state.use_colmap_model,
                        shared_state.use_transform_json,
                        shared_state.training_enable,
                        shared_state.max_steps,
                        shared_state.spherical_enable,
                        shared_state.remove_bg,
                        shared_state.remove_objects,
                        shared_state.object_removal_action,
                        shared_state.objects_to_remove,
                        shared_state.source_coordinate,
                        shared_state.pose_world_to_cam,
                        shared_state.log_verbosity,
                        shared_state.mask_threshold,
                        shared_state.crop_output_bounds,
                        shared_state.crop_mode,
                        shared_state.clean_splat,
                        shared_state.enable_spz,
                        shared_state.enable_sog,
                        shared_state.video_start_time,
                        shared_state.video_stop_time,
                        shared_state.preserve_scene_scale
                    )

        # Wire up the event handlers
        creds_submit_btn.click(
            parse_aws_credentials,
            inputs=[aws_creds],
            outputs=[creds_status]
        )
        
        preview_btn.click(
            fn=preview_json_with_shared_state,
            inputs=None,
            outputs=[debug_output]
        )
        
        cleanup_btn.click(
            fn=cleanup_local_files,
            inputs=[],
            outputs=[cleanup_output]
        )

def load_favorites():
    """Load favorite files from local directory"""
    favorites_dir = os.path.join(os.path.dirname(__file__), "favorites")
    os.makedirs(favorites_dir, exist_ok=True)
    
    favorites = []
    try:
        # List all files in the directory
        for file in os.listdir(favorites_dir):
            # Check for supported file types
            if file.endswith(('.ply', '.spz', '.glb', '.sog', '.usdz')):
                # Extract the original filename and UUID if possible
                parts = file.rsplit('_', 1)
                if len(parts) == 2:
                    name_part = parts[0]
                    uuid_part = parts[1].split('.')[0]  # Remove extension
                    display_name = f"{name_part}.{file.split('.')[-1]} ({uuid_part[:8]})"
                else:
                    display_name = file
                
                favorite = {
                    'filename': file,
                    'display_name': display_name,
                    'job_id': 'local',
                    'path': os.path.join(favorites_dir, file)
                }
                favorites.append(favorite)
                print(f"Found favorite: {file}, display as: {display_name}")
    except Exception as e:
        print(f"Error loading favorites: {str(e)}")
    
    return favorites

def create_combined_monitor_viewer_tab():
    with gr.Tab("Job Monitor & Viewer"):
        with gr.Column():
            # Favorites section at top
            gr.Markdown("### 📌 Favorites")
            favorites = load_favorites()
            favorite_buttons = []
            
            with gr.Row(elem_classes="favorites-buttons-row"):
                if not favorites:
                    gr.HTML('<div class="no-favorites-text">No favorites yet</div>')
                else:
                    for favorite in favorites:
                        with gr.Column(scale=1, min_width=100):
                            display_name = favorite.get('display_name', favorite['filename'])
                            favorite_btn = gr.Button(
                                value=f"📌 {display_name}", 
                                elem_classes=["favorite-button"],
                                size="sm"
                            )
                            favorite_buttons.append((favorite_btn, favorite['path'], favorite['filename']))
            
            gr.HTML("<hr style='margin: 20px 0; border: none; border-top: 2px solid #ddd;'>")
            
            # Job Progress Monitor section
            gr.Markdown("### 📊 Job Progress Monitor")
            gr.Markdown("Monitor jobs, view outputs, and refine completed jobs.")
            
            with gr.Row():
                refresh_jobs_btn = gr.Button(
                    "🔄 Refresh Jobs", 
                    variant="primary",
                    size="sm"
                )
                cancel_job_btn = gr.Button(
                    "🛑 Cancel Job",
                    interactive=False,
                    size="sm",
                    variant="stop"
                )
            
            jobs_table = gr.Dataframe(
                headers=["Job ID", "Media File", "Status", "Start Time", "Compute Type", "Instance Type", "Model", "Reconstruction Software", "Elapsed Time", "Evaluation Metrics"],
                interactive=False,
                value=[],
                visible=True,
                elem_id="jobs_monitor_table"
            )
            
            selected_job_data = gr.State(None)
            selected_file_data = gr.State(None)
            
            # Cancel job modal
            with Modal(visible=False, elem_id="cancel-modal-content") as cancel_modal:
                gr.Markdown("### ⚠️ Cancel Job", elem_classes="padded-markdown")
                cancel_job_info = gr.Markdown("Are you sure you want to cancel this job?")
                with gr.Row():
                    cancel_no_btn = gr.Button("No, Keep Running", variant="secondary")
                    cancel_yes_btn = gr.Button("Yes, Cancel Job", variant="stop")
            
            # Files modal - always in DOM, shown/hidden via JS (must be visible=True so children can be updated)
            with Modal(visible=True, elem_id="files-modal-content") as files_modal:
                with gr.Row():
                    gr.Markdown("### 📁 Job Output Files")
                    files_close_btn = gr.Button("✕ Close", size="sm", elem_classes=["close-button"])
                
                # Add thumbnail display
                job_thumbnail = gr.Image(
                    label="Job Thumbnail",
                    show_label=False,
                    height=200,
                    interactive=False
                )
                
                gr.Markdown("Click on a file to open it in the viewer.")
                
                with gr.Row():
                    refine_job_btn = gr.Button(
                        "🔧 Refine Job",
                        interactive=False,
                        size="sm",
                        variant="primary"
                    )
                
                files_table = gr.Dataframe(
                    headers=["Filename", "Size", "Type"],
                    interactive=False,
                    value=[],
                    visible=True,
                    elem_id="job_files_table"
                )
            
            # Phase progress and config in expandable sections
            with gr.Accordion("📈 Pipeline Progress", open=True) as phase_accordion:
                phase_progress_display = gr.HTML(
                    value="<div style='border: 3px solid #666; border-radius: 8px; padding: 15px; background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%); box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><h3 style='margin: 0 0 15px 0; color: #fff; border-bottom: 2px solid #666; padding-bottom: 8px;'>📊 Job Progress</h3><div style='color: #ccc;'>Select a job to view phase progress</div></div>",
                    elem_id="phase_progress_display"
                )
            
            with gr.Accordion("⚙️ Job Configuration", open=True) as config_accordion:
                job_config_display = gr.HTML(
                    value="<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Select a job to view configuration</div>",
                    elem_id="job_config_display"
                )
            
            # Refine modal and status
            with Modal(visible=False, elem_id="refine-modal-content") as refine_modal:
                gr.Markdown("### Refine Splat Settings", elem_classes="padded-markdown")
                gr.Markdown("Review and modify settings for refinement job:", elem_classes="padded-markdown")

                refine_instance = gr.Dropdown(
                    label="Instance Type",
                    choices=["ml.g5.4xlarge", "ml.g5.8xlarge", "ml.g5.12xlarge", "ml.g6.4xlarge", "ml.g6.8xlarge", "ml.g6e.4xlarge"],
                    value=shared_state.instance
                )
                refine_compute = gr.Radio(
                    label="Compute Type",
                    choices=[("AWS Batch (Spot Instances - Up to 50% cost savings)", "true"), ("SageMaker (On-Demand)", "false")],
                    value=shared_state.use_spot_instance
                )
                refine_crop = gr.Radio(
                    label="Crop Output Bounds",
                    choices=["true", "false"],
                    value=shared_state.crop_output_bounds
                )
                refine_crop_mode = gr.Dropdown(
                    label="Crop Mode",
                    choices=["environment", "rigid_body"],
                    value=shared_state.crop_mode
                )
                refine_isp_3d = gr.Dropdown(
                    label="3D ISP Mode",
                    choices=[
                        ("None", "none"),
                        ("Bilateral Grid (bilagrid)", "bilagrid"),
                        ("Per-Pixel ISP (ppisp)", "ppisp")
                    ],
                    value=shared_state.isp_3d,
                    info="Not applicable to nerfacto."
                )
                refine_clean_splat = gr.Radio(
                    label="Clean Splat (Remove Noise)",
                    choices=["true", "false"],
                    value=shared_state.clean_splat
                )
                refine_enable_spz = gr.Radio(
                    label="Enable SPZ Export",
                    choices=["true", "false"],
                    value=shared_state.enable_spz
                )
                refine_enable_sog = gr.Radio(
                    label="Enable SOG Export",
                    choices=["true", "false"],
                    value=shared_state.enable_sog
                )
                refine_enable_usdz = gr.Radio(
                    label="Enable USDZ Export",
                    choices=["true", "false"],
                    value=shared_state.enable_usdz
                )
                refine_ply_coords = gr.Dropdown(
                    label="PLY Coordinate System",
                    choices=[
                        ("Right-Hand, Y-Up (playcanvas)", "rhyu"),
                        ("Left-Hand, Y-Up (babylon.js)", "lhyu"),
                        ("Right-Hand, Z-Up (blender)", "rhzu"),
                        ("Left-Hand, Z-Up (unreal)", "lhzu")
                    ],
                    value=shared_state.ply_coords
                )
                refine_spz_coords = gr.Dropdown(
                    label="SPZ Coordinate System",
                    choices=[
                        ("Right-Hand, Y-Up (playcanvas)", "rhyu"),
                        ("Left-Hand, Y-Up (babylon.js)", "lhyu"),
                        ("Right-Hand, Z-Up (blender)", "rhzu"),
                        ("Left-Hand, Z-Up (unreal)", "lhzu")
                    ],
                    value=shared_state.spz_coords
                )
                refine_sog_coords = gr.Dropdown(
                    label="SOG Coordinate System",
                    choices=[
                        ("Right-Hand, Y-Up (playcanvas)", "rhyu"),
                        ("Left-Hand, Y-Up (babylon.js)", "lhyu"),
                        ("Right-Hand, Z-Up (blender)", "rhzu"),
                        ("Left-Hand, Z-Up (unreal)", "lhzu")
                    ],
                    value=shared_state.sog_coords
                )
                refine_usdz_coords = gr.Dropdown(
                    label="USDZ Coordinate System",
                    choices=[
                        ("Right-Hand, Y-Up (playcanvas)", "rhyu"),
                        ("Left-Hand, Y-Up (babylon.js)", "lhyu"),
                        ("Right-Hand, Z-Up (blender)", "rhzu"),
                        ("Left-Hand, Z-Up (unreal)", "lhzu")
                    ],
                    value=shared_state.usdz_coords
                )
                
                with gr.Row():
                    refine_cancel_btn = gr.Button("Cancel", variant="secondary")
                    refine_submit_btn = gr.Button("Submit Refinement Job", variant="primary")
            

            
            # Cost table at bottom
            gr.Markdown("### 💰 Job Costs")
            cost_display = gr.HTML(
                value="<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Click 'Refresh Jobs' to calculate costs</div>",
                label="Job Costs"
            )
            
            def refresh_jobs():
                """Refresh job monitoring data"""
                try:
                    jobs_data = get_job_progress_data()
                    return jobs_data, "<div style='border: 3px solid #666; border-radius: 8px; padding: 15px; background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%); box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><h3 style='margin: 0 0 15px 0; color: #fff; border-bottom: 2px solid #666; padding-bottom: 8px;'>📊 Job Progress</h3><div style='color: #ccc;'>Select a job to view phase progress</div></div>", "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Select a job to view configuration</div>"
                except Exception as e:
                    print(f"Error refreshing jobs: {e}")
                    return [], "<div style='border: 3px solid #666; border-radius: 8px; padding: 15px; background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%); box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><h3 style='margin: 0 0 15px 0; color: #fff; border-bottom: 2px solid #666; padding-bottom: 8px;'>📊 Job Progress</h3><div style='color: #ccc;'>Error loading jobs</div></div>", "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Error loading jobs</div>"
            
            def get_phase_progress_html(job_id):
                """Generate HTML for phase progress visualization"""
                try:
                    dynamodb = boto3.resource('dynamodb', region_name=shared_state.aws_region)
                    table = dynamodb.Table(shared_state.ddb_table_name)
                    response = table.get_item(Key={'uuid': _validate_uuid(job_id)})
                    
                    if 'Item' not in response:
                        return "<div style='border: 3px solid #666; border-radius: 8px; padding: 15px; background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%); box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><h3 style='margin: 0 0 15px 0; color: #fff; border-bottom: 2px solid #666; padding-bottom: 8px;'>📊 Job Progress</h3><div style='color: #ccc;'>No phase data available</div></div>"
                    
                    item = response['Item']
                    job_status = str(item.get('uuidStatus', '')).lower()
                    
                    phases = [
                        ('Pre-Processing', item.get('pre_processingElapsedTime')),
                        ('Reconstruction', item.get('reconstructionElapsedTime')),
                        ('Training', item.get('trainingElapsedTime')),
                        ('Post-Processing', item.get('post_processingElapsedTime'))
                    ]
                    
                    last_phase = str(item.get('lastUpdatedPhase', ''))
                    
                    bars = []
                    # Determine current in-progress phase based on elapsed times
                    current_phase = None
                    phase_list = [('Pre-Processing', item.get('pre_processingElapsedTime')),
                                  ('Reconstruction', item.get('reconstructionElapsedTime')),
                                  ('Training', item.get('trainingElapsedTime')),
                                  ('Post-Processing', item.get('post_processingElapsedTime'))]
                    
                    # Check if reconstruction is disabled for refined jobs
                    recon_disabled = 'runRecon' in item and str(item['runRecon']).lower() == 'false'
                    
                    # Find the current in-progress phase
                    for i, (name, elapsed) in enumerate(phase_list):
                        # Skip reconstruction if disabled
                        if i == 1 and recon_disabled:
                            continue
                        # If this phase hasn't started yet (None or missing)
                        if elapsed is None:
                            # Check if previous phase is complete or skipped
                            if i == 0:
                                # First phase not started
                                current_phase = name
                                break
                            elif i == 1 and recon_disabled:
                                # Reconstruction skipped, check if training should be current
                                continue
                            elif i > 0:
                                prev_elapsed = phase_list[i-1][1]
                                # If previous phase exists (not None), this is current
                                if prev_elapsed is not None or (i == 2 and recon_disabled and phase_list[0][1] is not None):
                                    current_phase = name
                                    break
                    
                    # Check if no phases have started
                    any_phase_started = any(elapsed is not None for _, elapsed in phase_list)
                    
                    # Check if all phases are complete (accounting for skipped phases)
                    if recon_disabled:
                        all_complete = all(elapsed is not None for i, (_, elapsed) in enumerate(phase_list) if i != 1)
                        # Use componentGroupElapsedTime if available, otherwise sum individual phases
                        if 'componentGroupElapsedTime' in item:
                            total_time = sum(float(t) for i, t in enumerate(item['componentGroupElapsedTime']) if i != 1)
                        else:
                            total_time = sum(float(elapsed) for i, (_, elapsed) in enumerate(phase_list) if elapsed is not None and i != 1)
                    else:
                        all_complete = all(elapsed is not None for _, elapsed in phase_list)
                        # Use componentGroupElapsedTime if available, otherwise sum individual phases
                        if 'componentGroupElapsedTime' in item:
                            total_time = sum(float(t) for t in item['componentGroupElapsedTime'])
                        else:
                            total_time = sum(float(elapsed) for _, elapsed in phase_list if elapsed is not None)
                    
                    # For older jobs without phase times, check if job is complete
                    if not any_phase_started and job_status in ['complete', 'completed']:
                        return f"<div style='border: 3px solid #666; border-radius: 8px; padding: 15px; background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%); box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><h3 style='margin: 0 0 15px 0; color: #fff; border-bottom: 2px solid #666; padding-bottom: 8px;'>📊 Job Progress</h3><div style='color: #ccc; text-align: center; padding: 20px;'>✅ Job completed (legacy format - no phase timing data)</div></div>"
                    
                    if not any_phase_started:
                        if job_status in ['failed', 'error']:
                            return f"<div style='border: 3px solid #666; border-radius: 8px; padding: 15px; background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%); box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><h3 style='margin: 0 0 15px 0; color: #fff; border-bottom: 2px solid #666; padding-bottom: 8px;'>📊 Job Progress</h3><div style='color: #ff6b6b; text-align: center; padding: 20px;'>❌ Job failed before processing started</div></div>"
                        return f"<div style='border: 3px solid #666; border-radius: 8px; padding: 15px; background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%); box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><h3 style='margin: 0 0 15px 0; color: #fff; border-bottom: 2px solid #666; padding-bottom: 8px;'>📊 Job Progress</h3><div style='color: #ccc; text-align: center; padding: 20px;'>⏳ Job pending - waiting to start</div></div>"
                    
                    for name, elapsed in phases:
                        # Skip reconstruction phase if disabled
                        if name == 'Reconstruction' and recon_disabled:
                            continue
                        
                        if elapsed is not None:
                            status = 'complete'
                            color = '#4CAF50'
                        elif current_phase and name == current_phase:
                            status = 'in-progress'
                            color = '#FFA500'
                        else:
                            status = 'pending'
                            color = '#999'
                        
                        time_str = f"{elapsed}s" if elapsed is not None else "-"
                        width_pct = '100%' if status == 'complete' else '50%' if status == 'in-progress' else '0%'
                        bars.append(f"<div style='margin: 8px 0;'><div style='display: flex; align-items: center;'><div style='width: 150px; font-weight: 500; color: #fff;'>{name}</div><div style='flex: 1; background: #555; height: 24px; border-radius: 4px; overflow: hidden; margin: 0 10px;'><div style='background: {color}; height: 100%; width: {width_pct}; transition: width 0.3s;'></div></div><div style='width: 80px; text-align: right; color: #fff;'>{time_str}</div></div></div>")
                    
                    if all_complete:
                        h = int(total_time//3600)
                        m = int((total_time%3600)//60)
                        s = int(total_time%60)
                        time_parts = []
                        if h > 0:
                            time_parts.append(f"{h}h")
                        if m > 0:
                            time_parts.append(f"{m}min")
                        if s > 0 or not time_parts:
                            time_parts.append(f"{s}s")
                        total_html = f"<div style='margin-top: 15px; padding-top: 15px; border-top: 1px solid #666; color: #fff; font-weight: bold; text-align: right;'>Total Elapsed Time: {' '.join(time_parts)}</div>"
                    else:
                        total_html = ""
                    return f"<div style='border: 3px solid #666; border-radius: 8px; padding: 15px; background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%); box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><h3 style='margin: 0 0 15px 0; color: #fff; border-bottom: 2px solid #666; padding-bottom: 8px;'>📊 Job Progress</h3>{''.join(bars)}{total_html}</div>"
                except Exception as e:
                    return f"<div style='border: 3px solid #666; border-radius: 8px; padding: 15px; background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%); box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><h3 style='margin: 0 0 15px 0; color: #fff; border-bottom: 2px solid #666; padding-bottom: 8px;'>📊 Job Progress</h3><div style='color: #ccc;'>Error loading phase data: {str(e)}</div></div>"
            
            def on_job_select(evt: gr.SelectData, data):
                """Handle job selection to show configuration and enable refine button"""
                try:
                    # Handle DataFrame or list data
                    if hasattr(data, 'empty'):
                        if data.empty:
                            return "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>No job data available</div>", "<div style='color: #666;'>No job selected</div>", None, gr.update(interactive=False)
                        data_list = data.values.tolist()
                    else:
                        if not data or len(data) == 0:
                            return "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>No job data available</div>", "<div style='color: #666;'>No job selected</div>", None, gr.update(interactive=False)
                        data_list = data
                    
                    row_idx = evt.index[0]
                    if row_idx >= len(data_list):
                        return "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Invalid selection</div>", "<div style='color: #666;'>Invalid selection</div>", None, gr.update(interactive=False)
                    
                    selected_row = data_list[row_idx]
                    if len(selected_row) < 3:
                        return "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Invalid job data</div>", "<div style='color: #666;'>Invalid job data</div>", None, gr.update(interactive=False)
                    
                    # Extract job ID and status
                    job_id = selected_row[0]
                    status = selected_row[2].lower() if len(selected_row) > 2 else ""
                    
                    # Enable refine button only for completed jobs
                    enable_refine = status in ['complete', 'completed']
                    
                    # Store selected job data for refine operation
                    job_data = [job_id, "model.tar.gz"]  # Format similar to Viewer & Library
                    
                    # Get full job metadata and phase progress
                    return get_phase_progress_html(job_id), get_job_metadata(job_id), job_data, gr.update(interactive=enable_refine)
                    
                except Exception as e:
                    print(f"Error in job selection: {e}")
                    return f"<div style='color: #666;'>Error: {str(e)}</div>", f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Error: {str(e)}</div>", None, gr.update(interactive=False)
            
            # Hidden components for viewer modal
            sog_file_data = gr.Textbox(visible=False)
            current_filename = gr.Textbox(visible=False)
            download_iframe = gr.HTML(visible=True)
            show_files_signal = gr.Textbox(value="", visible=False, elem_id="show-files-signal")
            # Dummy output to replace files_modal in outputs (since hoisted modals can't be updated by Gradio)
            files_modal_dummy = gr.Textbox(visible=False)
            
            # Viewer modal (visible=True so children can be updated; CSS hides it initially)
            with Modal(visible=True, elem_id="viewer-modal-content") as viewer_modal:
                with gr.Row():
                    gr.Markdown("### 3D Viewer")
                    viewer_close_btn = gr.Button("✕ Close", size="sm", elem_classes=["close-button"])
                
                with gr.Row():
                    # Left panel: File list and buttons
                    with gr.Column(scale=1, min_width=250):
                        gr.Markdown("#### Files")
                        file_selector = gr.Dropdown(
                            label="Select File",
                            choices=[],
                            interactive=True
                        )
                        with gr.Column():
                            download_file_btn = gr.Button("⬇️ Download", size="sm")
                            add_favorite_file_btn = gr.Button("⭐ Add to Favorites", size="sm")
                    
                    # Right panel: Viewer
                    with gr.Column(scale=3):
                        # Hidden dummy components for viewer/viewer_status (needed for output compatibility)
                        viewer_status = gr.Textbox(visible=False)
                        
                        with gr.Tabs():
                            with gr.Tab("3D Splat Viewer"):
                                sog_viewer = gr.HTML(
                                    value="<div id='sog-container-monitor' style='height: 700px; background: #1a1a1a; border: 1px solid #444; display: flex; align-items: center; justify-content: center; color: white;'>Select a file to view</div>",
                                    label="SOG Viewer"
                                )
                                sog_status = gr.Textbox(
                                    label="SOG Viewer Status",
                                    interactive=False,
                                    value="",
                                    lines=1,
                                    max_lines=1
                                )
                                viewer = gr.Model3D(visible=False, label="3D Viewer")
                                spz_viewer = gr.Textbox(value="", visible=False, label="SPZ Data")

                            with gr.Tab("Video Preview"):
                                video_viewer = gr.Video(
                                    label="Trajectory Video",
                                    height=700,
                                    interactive=False,
                                    format="mp4"
                                )
                                video_status = gr.Textbox(
                                    label="Video Status",
                                    interactive=False,
                                    value=""
                                )
            
            # Wire up event handlers
            def refresh_jobs_and_costs():
                jobs_data = get_job_progress_data()
                costs_html = calculate_job_costs_from_jobs(jobs_data)
                return (
                    jobs_data,
                    "<div style='border: 3px solid #666; border-radius: 8px; padding: 15px; background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%); box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><h3 style='margin: 0 0 15px 0; color: #fff; border-bottom: 2px solid #666; padding-bottom: 8px;'>📊 Job Progress</h3><div style='color: #ccc;'>Select a job to view phase progress</div></div>",
                    "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Select a job to view configuration</div>",
                    costs_html
                )
            
            refresh_jobs_btn.click(
                fn=refresh_jobs_and_costs,
                inputs=[],
                outputs=[jobs_table, phase_progress_display, job_config_display, cost_display]
            )
            
            def on_job_select_combined(evt: gr.SelectData, data):
                """Handle job selection - open files modal"""
                import time as _time
                try:
                    if hasattr(data, 'empty'):
                        if data.empty:
                            return None, gr.update(), [], gr.update(interactive=False), gr.update(interactive=False), "", "", None, f"hide:{_time.time()}"
                        data_list = data.values.tolist()
                    else:
                        if not data or len(data) == 0:
                            return None, gr.update(), [], gr.update(interactive=False), gr.update(interactive=False), "", "", None, f"hide:{_time.time()}"
                        data_list = data
                    
                    row_idx = evt.index[0]
                    if row_idx >= len(data_list):
                        return None, gr.update(), [], gr.update(interactive=False), gr.update(interactive=False), "", "", None, f"hide:{_time.time()}"
                    
                    selected_row = data_list[row_idx]
                    job_id = selected_row[0]
                    status = selected_row[2].lower() if len(selected_row) > 2 else ""
                    
                    print(f"Selected job: {job_id}, status: {status}")
                    
                    # Get job files from DynamoDB
                    job_files = get_job_output_files(job_id)
                    print(f"Found {len(job_files)} files for job {job_id}")
                    
                    # Create files table data with size and type
                    files_data = []
                    for file_info in job_files:
                        filename = file_info.get('filename', '')
                        size_bytes = file_info.get('size', 0)
                        
                        if size_bytes < 1024:
                            size_str = f"{size_bytes} B"
                        elif size_bytes < 1024 * 1024:
                            size_str = f"{size_bytes/1024:.1f} KB"
                        else:
                            size_str = f"{size_bytes/(1024*1024):.1f} MB"
                        
                        ext = filename.split('.')[-1].upper() if '.' in filename else 'Unknown'
                        files_data.append([filename, size_str, ext])
                    
                    # Enable refine button only for completed jobs
                    enable_refine = status in ['complete', 'completed']
                    # Enable cancel button only for in-progress jobs
                    enable_cancel = status == 'in-progress'
                    
                    job_data = [job_id, "model.tar.gz"]
                    
                    # Get phase progress and metadata
                    phase_html = get_phase_progress_html(job_id)
                    metadata_html = get_job_metadata(job_id)
                    
                    # Generate thumbnail URL
                    bucket_name = shared_state.s3_bucket
                    output_prefix = shared_state.s3_output or "workflow-output"
                    thumbnail_key = f"{output_prefix}/{job_id}/render_thumbnail.png"
                    thumbnail_url = generate_presigned_url(bucket_name, thumbnail_key)
                    
                    print(f"Phase HTML length: {len(phase_html)}, Metadata HTML length: {len(metadata_html)}")
                    
                    # Open files modal if there are files
                    # Append timestamp so signal always changes, ensuring .then() fires every click
                    import time as _time
                    ts = _time.time()
                    if len(files_data) > 0:
                        return (
                            job_data,
                            gr.update(),
                            files_data,
                            gr.update(interactive=enable_refine),
                            gr.update(interactive=enable_cancel),
                            phase_html,
                            metadata_html,
                            thumbnail_url,
                            f"show:{ts}"
                        )
                    else:
                        return (
                            job_data,
                            gr.update(),
                            [],
                            gr.update(interactive=enable_refine),
                            gr.update(interactive=enable_cancel),
                            phase_html,
                            metadata_html,
                            None,
                            f"hide:{ts}"
                        )
                except Exception as e:
                    print(f"Error in job selection: {e}")
                    import traceback
                    traceback.print_exc()
                    return None, gr.update(), [], gr.update(interactive=False), gr.update(interactive=False), "", "", None, f"hide:{_time.time()}"
            
            # Debug: add JS to verify select fires, then call Python handler
            jobs_table.select(
                fn=on_job_select_combined,
                inputs=[jobs_table],
                outputs=[selected_job_data, files_modal_dummy, files_table, refine_job_btn, cancel_job_btn, phase_progress_display, job_config_display, job_thumbnail, show_files_signal],
                js="""(...args) => { console.log('GRADIO6 DEBUG: jobs_table.select fired!', args); return args; }"""
            ).then(
                fn=None,
                inputs=[show_files_signal],
                js="""(sig) => { const el = document.getElementById('files-modal-content'); if(el) { if(sig && sig.startsWith('show')) { el.style.display = 'flex'; } else { el.style.display = 'none'; } } }"""
            )
            
            # Cancel job handlers
            def show_cancel_modal(selected_data):
                if not selected_data:
                    return gr.update(visible=False), ""
                job_id = selected_data[0][:8] if selected_data[0] else "Unknown"
                return (
                    gr.update(visible=True),
                    f"Are you sure you want to cancel job **{job_id}...**?"
                )
            
            def cancel_job_execution(selected_data):
                """Cancel a running job by stopping the state machine execution"""
                try:
                    if not selected_data:
                        return gr.update(elem_classes=["hide"]), "No job selected"
                    
                    job_id = selected_data[0]
                    
                    sfn_client = boto3.client('stepfunctions', region_name=shared_state.aws_region)
                    
                    # Look up the state machine ARN from SSM (same way Lambda does)
                    ssm_client = boto3.client('ssm', region_name=shared_state.aws_region)
                    ssm_param_name = f"3dgs-sfn-arn-{shared_state.stack_unique_id}"
                    state_machine_arn = ssm_client.get_parameter(Name=ssm_param_name, WithDecryption=True)['Parameter']['Value']
                    
                    # Find the running execution that contains this job UUID
                    execution_arn = None
                    paginator = sfn_client.get_paginator('list_executions')
                    for page in paginator.paginate(stateMachineArn=state_machine_arn, statusFilter='RUNNING'):
                        for ex in page['executions']:
                            if job_id in ex['name'] or job_id in ex.get('executionArn', ''):
                                execution_arn = ex['executionArn']
                                break
                        if execution_arn:
                            break
                    
                    if not execution_arn:
                        raise Exception(f"No running execution found for job {job_id[:8]}")
                    
                    sfn_client.stop_execution(
                        executionArn=execution_arn,
                        error='UserCancelled',
                        cause='Job cancelled by user from Gradio UI'
                    )
                    
                    # Update DynamoDB to mark as cancelled
                    dynamodb = boto3.resource('dynamodb', region_name=shared_state.aws_region)
                    table = dynamodb.Table(shared_state.ddb_table_name)
                    table.update_item(
                        Key={'uuid': _validate_uuid(job_id)},
                        UpdateExpression='SET uuidStatus = :status',
                        ExpressionAttributeValues={':status': 'Cancelled'}
                    )
                    
                    gr.Info(f"✅ Job {job_id[:8]}... cancelled successfully")
                    return gr.update(visible=False)
                    
                except Exception as e:
                    print(f"Error cancelling job: {e}")
                    import traceback
                    traceback.print_exc()
                    gr.Warning(f"❌ Error cancelling job: {str(e)}")
                    return gr.update(visible=False)
            
            cancel_job_btn.click(
                fn=show_cancel_modal,
                inputs=[selected_job_data],
                outputs=[cancel_modal, cancel_job_info]
            )
            
            cancel_no_btn.click(
                fn=lambda: gr.update(visible=False),
                outputs=[cancel_modal]
            )
            
            cancel_yes_btn.click(
                fn=cancel_job_execution,
                inputs=[selected_job_data],
                outputs=[cancel_modal]
            )
            
            # Close files modal - use JS only since element is hoisted to document.body
            files_close_btn.click(
                fn=None,
                js="""() => { const el = document.getElementById('files-modal-content'); if(el) el.style.display = 'none'; }"""
            )
            
            # Handle file selection from files table - automatically open viewer
            def on_file_select_and_view(evt: gr.SelectData, data, job_data):
                """Handle file selection - automatically open viewer with file"""
                try:
                    if job_data is None:
                        return tuple([None] + [gr.update()] * 2 + [gr.update()] * 8)
                    
                    if hasattr(data, 'empty'):
                        if data.empty:
                            return tuple([None] + [gr.update()] * 2 + [gr.update()] * 8)
                        data_list = data.values.tolist()
                    elif isinstance(data, list):
                        if len(data) == 0:
                            return tuple([None] + [gr.update()] * 10)
                        data_list = data
                    else:
                        return tuple([None] + [gr.update()] * 10)
                    
                    row_idx = evt.index[0]
                    if row_idx >= len(data_list):
                        return tuple([None] + [gr.update()] * 10)
                    
                    selected_file = data_list[row_idx]
                    filename = selected_file[0]
                    file_data = [job_data[0], filename]
                    
                    # Get all files for this job
                    job_id = job_data[0]
                    job_files = get_job_output_files(job_id)
                    file_choices = [f.get('filename', '') for f in job_files]
                    
                    # Load the selected file
                    selected_row = [job_id, filename]
                    result = handle_view_multi(selected_row)
                    
                    # JS handles modal show/hide - just return data updates
                    # Set both choices and value so the dropdown reflects the selected file.
                    # load_selected_file guards against re-downloading when data is already cached.
                    return (
                        file_data,
                        gr.update(),  # files_modal - JS handles hide
                        gr.update(),  # viewer_modal - JS handles show
                        gr.update(choices=file_choices, value=filename),
                        result[0],
                        result[1],
                        result[2],
                        result[3],
                        result[4],
                        result[5],
                        result[6],
                        filename
                    )
                except Exception as e:
                    print(f"Error in file selection: {e}")
                    import traceback
                    traceback.print_exc()
                    return tuple([None] + [gr.update()] * 11)
            
            # Two dummy textboxes to replace files_modal and viewer_modal in outputs
            files_modal_dummy2 = gr.Textbox(visible=False)
            viewer_modal_dummy = gr.Textbox(visible=False)
            
            files_table.select(
                fn=on_file_select_and_view,
                inputs=[files_table, selected_job_data],
                outputs=[selected_file_data, files_modal_dummy2, viewer_modal_dummy, file_selector, viewer, viewer_status, sog_file_data, sog_status, video_viewer, video_status, spz_viewer, current_filename],
                js="""(...args) => {
                    // Hide files modal and show viewer modal via style.display
                    const filesEl = document.getElementById('files-modal-content');
                    const viewerEl = document.getElementById('viewer-modal-content');
                    if(filesEl) filesEl.style.display = 'none';
                    if(viewerEl) {
                        viewerEl.style.display = 'flex';
                        setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
                    }
                    return args;
                }"""
            )
            
            # View button opens viewer modal from files modal with all job files
            def open_viewer_from_files_modal(file_data, job_data):
                if not file_data or not job_data:
                    return tuple([gr.update(elem_classes=["hide"])] * 2 + [gr.update(elem_classes=["hide"])] * 2 + [gr.update() for _ in range(8)])
                
                job_id = job_data[0]
                filename = file_data[1]
                
                job_files = get_job_output_files(job_id)
                file_choices = [f.get('filename', '') for f in job_files]
                
                selected_row = [job_id, filename]
                result = handle_view_multi(selected_row)
                
                return (
                    gr.update(elem_classes=["hide"]),  # Close files modal
                    gr.update(elem_classes=[]),   # Open viewer modal
                    gr.update(choices=file_choices, value=filename),
                    result[0],
                    result[1],
                    result[2],
                    result[3],
                    result[4],
                    result[5],
                    result[6],
                    filename
                )
            
            # File selector change loads the selected file AND updates selected_file_data
            def load_selected_file(filename, job_data):
                if not filename or not job_data:
                    return gr.update(), "", "", "", gr.update(), "", gr.update(), "", None
                job_id = job_data[0]
                output_prefix = shared_state.s3_output or "workflow-output"
                file_key = f"{output_prefix}/{job_id}/{filename}"
                current_key = getattr(shared_state, 'current_model_key', None)
                if current_key == file_key:
                    # Re-use cached data to re-trigger the viewer
                    cached_data = getattr(shared_state, 'current_model_data', None)
                    if cached_data:
                        payload = json.dumps({"data": cached_data, "filename": filename, "ts": __import__('time').time()})
                        return gr.update(), gr.update(), gr.update(value=payload), gr.update(), gr.update(), gr.update(), gr.update(), filename, [job_id, filename]
                    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), filename, [job_id, filename]
                # Clear cache so handle_view_multi re-downloads for the new file
                shared_state.current_model_key = None
                shared_state.current_model_data = None
                selected_row = [job_id, filename]
                result = handle_view_multi(selected_row)
                return result[0], result[1], result[2], result[3], result[4], result[5], result[6], filename, selected_row
            
            file_selector.change(
                fn=load_selected_file,
                inputs=[file_selector, selected_job_data],
                outputs=[viewer, viewer_status, sog_file_data, sog_status, video_viewer, video_status, spz_viewer, current_filename, selected_file_data]
            )

            spz_viewer.change(
                fn=None,
                inputs=[spz_viewer],
                outputs=None,
                js="""async (payload) => {
                    if(!payload) return;
                    let parsed; try { parsed = JSON.parse(payload); } catch(e) { return; }
                    const {data: fileData, filename: fileName} = parsed;
                    if(!fileData || !fileName) return;
                    window.spzData = {fileData, fileName};
                    let attempts = 0;
                    const tryLoad = () => {
                        if(window.createSPZViewer) {
                            window.createSPZViewer(fileData, fileName, 'S3 file');
                        } else if(attempts < 20) { attempts++; setTimeout(tryLoad, 250); }
                    };
                    setTimeout(tryLoad, 300);
                }"""
            )

            # Download button in left panel
            download_file_btn.click(
                fn=lambda filename, job_data: handle_download([job_data[0], filename]) if filename and job_data else "",
                inputs=[file_selector, selected_job_data],
                outputs=[download_iframe]
            )
            download_iframe.change(
                fn=None,
                inputs=[download_iframe],
                outputs=None,
                js="""(url) => {
                    if (url) { const a = document.createElement('a'); a.href = url; a.download = ''; document.body.appendChild(a); a.click(); document.body.removeChild(a); }
                }"""
            )
            
            # Add to favorites button in left panel
            def add_favorite_from_selector(filename, job_data):
                result = add_to_favorites([job_data[0], filename]) if filename and job_data else "No file selected"
                gr.Info(result)
            
            add_favorite_file_btn.click(
                fn=add_favorite_from_selector,
                inputs=[file_selector, selected_job_data]
            )
            
            # Close viewer modal via JS
            viewer_close_btn.click(
                fn=None,
                js="""() => { const el = document.getElementById('viewer-modal-content'); if(el) el.style.display = 'none'; }"""
            )
            
            # Refine modal handlers
            def show_refine_modal_monitor(selected_data):
                if not selected_data:
                    return gr.update(visible=False)
                return gr.update(visible=True)
            
            refine_job_btn.click(
                fn=show_refine_modal_monitor,
                inputs=[selected_job_data],
                outputs=[refine_modal]
            )
            
            refine_cancel_btn.click(
                fn=lambda: gr.update(visible=False),
                outputs=[refine_modal]
            )
            
            def submit_refine_job_monitor(selected_data, instance, compute, crop, crop_mode, clean_splat, spz, sog, usdz, ply_coords, spz_coords, sog_coords, usdz_coords, isp_3d):
                result = refine_splat(selected_data, instance, compute, crop, crop_mode, clean_splat, spz, sog, usdz, ply_coords, spz_coords, sog_coords, usdz_coords, isp_3d)
                gr.Info(result)
                return gr.update(visible=False)
            
            refine_submit_btn.click(
                fn=submit_refine_job_monitor,
                inputs=[selected_job_data, refine_instance, refine_compute, refine_crop, refine_crop_mode, refine_clean_splat, refine_enable_spz, refine_enable_sog, refine_enable_usdz, refine_ply_coords, refine_spz_coords, refine_sog_coords, refine_usdz_coords, refine_isp_3d],
                outputs=[refine_modal]
            )
            
            # Connect favorite buttons to viewer modal
            def open_favorite_in_modal(path, name):
                favorites_dir = os.path.join(os.path.dirname(__file__), "favorites")
                if not os.path.realpath(path).startswith(os.path.realpath(favorites_dir) + os.sep):
                    return (gr.update(), gr.update(), gr.update(), gr.update(), "", gr.update(), gr.update(), "", "", "")
                if name.lower().endswith('.sog'):
                    import base64
                    with open(path, 'rb') as f:
                        file_data = base64.b64encode(f.read()).decode('utf-8')
                    payload = json.dumps({"data": file_data, "filename": name, "ts": __import__('time').time()})
                    return (
                        gr.update(elem_classes=[]),
                        gr.update(choices=[name], value=name),
                        gr.update(value=None),
                        "",
                        payload,
                        gr.update(),
                        gr.update(value=None),
                        "",
                        name,
                        ""
                    )
                elif name.lower().endswith('.spz'):
                    import base64
                    with open(path, 'rb') as f:
                        file_data = base64.b64encode(f.read()).decode('utf-8')
                    payload = json.dumps({"data": file_data, "filename": name, "ts": __import__('time').time()})
                    return (
                        gr.update(elem_classes=[]),
                        gr.update(choices=[name], value=name),
                        gr.update(value=None),
                        "",
                        "",
                        gr.update(),
                        gr.update(value=None),
                        "",
                        name,
                        payload
                    )
                else:
                    return (
                        gr.update(elem_classes=[]),
                        gr.update(choices=[name], value=name),
                        gr.update(value=path),
                        f"Loaded: {name}",
                        "",
                        gr.update(),
                        gr.update(value=None),
                        "",
                        name,
                        ""
                    )
            
            # Dummy for favorites output (can't output to gr.Column modals in Gradio 6)
            viewer_modal_dummy2 = gr.Textbox(visible=False)
            
            for btn, path, name in favorite_buttons:
                btn.click(
                    fn=lambda p=path, n=name: open_favorite_in_modal(p, n),
                    inputs=[],
                    outputs=[viewer_modal_dummy2, file_selector, viewer, viewer_status, sog_file_data, sog_status, video_viewer, video_status, current_filename, spz_viewer],
                    js="""() => { const el = document.getElementById('viewer-modal-content'); if(el) el.style.display = 'flex'; }"""
                )
            
            # Add tab switching when current_filename changes - route all splat files to PlayCanvas
            current_filename.change(
                fn=None,
                inputs=[current_filename],
                js="""(filename) => {
                    if(filename) {
                        const fname = filename.toLowerCase();
                        setTimeout(() => {
                            const modalContent = document.querySelector('#viewer-modal-content');
                            if(modalContent) {
                                const allTabs = modalContent.querySelectorAll('button[role="tab"]');
                                allTabs.forEach((tab) => {
                                    const tabText = tab.textContent.trim();
                                    if(fname.endsWith('.mp4') && tabText.includes('Video Preview')) {
                                        tab.click();
                                    } else if(!fname.endsWith('.mp4') && (tabText.includes('3D Splat Viewer') || tabText.includes('SOG Viewer'))) {
                                        tab.click();
                                    }
                                });
                            }
                        }, 200);
                    }
                }"""
            )
            
            # Trigger PlayCanvas GSplat viewer when data changes (supports .sog, .spz, .ply)
            # Uses retry logic to wait for createSOGViewer to be defined after page load
            sog_file_data.change(
                None,
                inputs=[sog_file_data],
                outputs=None,
                js="""async (payload) => {
                    if(!payload) return;
                    let parsed; try { parsed = JSON.parse(payload); } catch(e) { return; }
                    const {data: fileData, filename: fileName} = parsed;
                    if(!fileData || !fileName) return;
                    const fname = fileName.toLowerCase();
                    if(!fname.endsWith('.sog') && !fname.endsWith('.ply')) return;
                    window.sogData = {fileData, fileName, fileSize: 'S3 file'};
                    window.sogLoaded = false;
                    let attempts = 0;
                    const tryLoad = () => {
                        console.log('[SPZ/SOG] tryLoad attempt', attempts, 'fname:', fname, 'createSPZViewer:', !!window.createSPZViewer, 'createSOGViewer:', !!window.createSOGViewer);
                        if(window.createSOGViewer) {
                            window.createSOGViewer(fileData, fileName, 'S3 file');
                            window.sogLoaded = true;
                        } else if(attempts < 20) { attempts++; setTimeout(tryLoad, 250); }
                    };
                    setTimeout(tryLoad, 800);
                }"""
            )
            
            gr.HTML(
                '<div style="text-align:center; margin:10px 0;"><a href="https://superspl.at/editor" target="_blank" style="display:inline-block; background:#f97316; color:white; padding:8px 16px; text-decoration:none; border-radius:6px; font-size:14px; font-weight:500;">🚀 Open SuperSplat Editor</a></div>'
            )

# Add the PlayCanvas JavaScript code globally
playcanvas_js = """
async () => {
    // Ensure modal ancestors don't create stacking contexts that break position:fixed
    const fixModalAncestors = () => {
        ['cancel-modal-content','files-modal-content','refine-modal-content','viewer-modal-content'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                let parent = el.parentElement;
                while (parent && parent !== document.body) {
                    const style = window.getComputedStyle(parent);
                    if (style.transform !== 'none' || style.willChange === 'transform' || style.contain === 'paint') {
                        parent.style.transform = 'none';
                        parent.style.willChange = 'auto';
                        parent.style.contain = 'none';
                    }
                    parent = parent.parentElement;
                }
            }
        });
    };
    // Initially hide modals that start visible=True (so Gradio can update their children)
    const hideModalsOnLoad = () => {
        ['files-modal-content', 'viewer-modal-content'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });
    };
    // Run immediately and after delays to catch late-rendering
    hideModalsOnLoad();
    setTimeout(hideModalsOnLoad, 50);
    setTimeout(hideModalsOnLoad, 200);
    setTimeout(hideModalsOnLoad, 500);
    setTimeout(hideModalsOnLoad, 1000);
    
    setTimeout(fixModalAncestors, 500);
    setTimeout(fixModalAncestors, 2000);

    // Load PlayCanvas SDK - try multiple methods for Gradio 6 compatibility
    if (!window.pc) {
        console.log('PlayCanvas not found on window.pc, attempting to load...');
        
        // Method 1: Dynamic script tag (works in most cases)
        try {
            const script = document.createElement('script');
            script.src = 'https://code.playcanvas.com/playcanvas-2.17.0.min.js';
            script.crossOrigin = 'anonymous';
            document.head.appendChild(script);
            await new Promise((resolve, reject) => {
                script.onload = () => { console.log('PlayCanvas loaded via dynamic script tag'); resolve(); };
                script.onerror = (e) => { console.error('PlayCanvas dynamic script failed:', e); reject(e); };
                setTimeout(() => reject(new Error('PlayCanvas load timeout')), 10000);
            });
        } catch (e) {
            console.warn('Dynamic script load failed, trying fetch+blob fallback:', e);
            // Method 2: Fetch + Blob URL (bypasses some CSP restrictions)
            try {
                const resp = await fetch('https://code.playcanvas.com/playcanvas-2.17.0.min.js');
                const text = await resp.text();
                const blob = new Blob([text], { type: 'application/javascript' });
                const blobUrl = URL.createObjectURL(blob);
                const script2 = document.createElement('script');
                script2.src = blobUrl;
                document.head.appendChild(script2);
                await new Promise((resolve) => { script2.onload = resolve; setTimeout(resolve, 2000); });
                URL.revokeObjectURL(blobUrl);
                console.log('PlayCanvas loaded via fetch+blob fallback');
            } catch (e2) {
                console.error('All PlayCanvas loading methods failed:', e2);
            }
        }
        
        // Final wait for window.pc to be defined
        let pcWaitAttempts = 0;
        while (!window.pc && pcWaitAttempts < 30) {
            await new Promise(resolve => setTimeout(resolve, 200));
            pcWaitAttempts++;
        }
    }
    
    if (!window.pc) {
        console.error('PlayCanvas SDK is not available after all loading attempts');
        return;
    }
    console.log('PlayCanvas SDK ready, version:', window.pc.version || 'unknown');
    
    globalThis.createSOGViewer = (fileData, fileName, fileSize) => {
        // Try monitor tab container first, then fall back to library tab container
        const container = document.getElementById('sog-container-monitor') || document.getElementById('sog-container');
        if (!container) return;
        
        // Force container AND its parent tab panel to be visible
        container.style.width = container.offsetWidth > 0 ? container.offsetWidth + 'px' : '800px';
        container.style.height = '700px';
        container.style.display = 'block';
        // Walk up the DOM and force any hidden tab panels to display
        let el = container.parentElement;
        while (el && el.id !== 'viewer-modal-content') {
            if (el.style.display === 'none' || window.getComputedStyle(el).display === 'none') {
                el.style.display = 'block';
            }
            el = el.parentElement;
        }
        
        container.innerHTML = '<canvas id="pc-canvas" style="width: 100%; height: 700px; margin: 0px; background: #1a1a1a;"></canvas>';
        const canvas = document.getElementById('pc-canvas');
        
        // Force canvas dimensions
        const containerWidth = container.offsetWidth || 800;
        const containerHeight = 700;
        canvas.width = containerWidth * (window.devicePixelRatio || 1);
        canvas.height = containerHeight * (window.devicePixelRatio || 1);
        canvas.style.width = containerWidth + 'px';
        canvas.style.height = containerHeight + 'px';
        
        const app = new pc.Application(canvas, {
            mouse: new pc.Mouse(canvas),
            touch: new pc.TouchDevice(canvas),
            keyboard: new pc.Keyboard(window),
            graphicsDeviceOptions: {
                antialias: true,
                alpha: false,
                preserveDrawingBuffer: false,
                preferWebGl2: true,
                powerPreference: "high-performance"
            }
        });
        
        // Configure high-resolution settings
        app.scene.clusteredLightingEnabled = false;
        app.autoRender = true;
        
        const pixelRatio = window.devicePixelRatio || 1;
        app.graphicsDevice.maxPixelRatio = pixelRatio;
        app.setCanvasFillMode(pc.FILLMODE_NONE);
        app.setCanvasResolution(pc.RESOLUTION_FIXED);
        
        // Handle resize for high DPI - only resize if container has real dimensions
        const handleResize = () => {
            const rect = container.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                const pixelRatio = window.devicePixelRatio || 1;
                canvas.width = rect.width * pixelRatio;
                canvas.height = rect.height * pixelRatio;
                canvas.style.width = rect.width + 'px';
                canvas.style.height = rect.height + 'px';
                app.graphicsDevice.setResolution(canvas.width, canvas.height);
            }
        };
        
        window.addEventListener('resize', handleResize);
        // Don't call handleResize() immediately - use the forced dimensions we already set above
        // The resize handler will recalculate when the tab becomes visible
        app.graphicsDevice.setResolution(canvas.width, canvas.height);
        
        app.start();
        
        const camera = new pc.Entity('Camera');
        camera.addComponent('camera', {
            clearColor: new pc.Color(0.1, 0.1, 0.1, 1.0),
            fov: 75,
            nearClip: 0.1,
            farClip: 1000
        });
        camera.setPosition(0, 2, 5);
        app.root.addChild(camera);
        
        const light = new pc.Entity('Light');
        light.addComponent('light', {
            type: 'directional',
            color: new pc.Color(1, 1, 1),
            intensity: 1
        });
        light.setEulerAngles(45, 30, 0);
        app.root.addChild(light);
        
        // Camera controls
        let isMouseDown = false;
        let isPanning = false;
        let lastMouseX = 0;
        let lastMouseY = 0;
        let cameraDistance = 10;
        let cameraYaw = 0;
        let cameraPitch = 0.3;
        const target = new pc.Vec3(0, 0, 0);

        const updateCameraPosition = () => {
            const x = target.x + cameraDistance * Math.sin(cameraYaw) * Math.cos(cameraPitch);
            const y = target.y + cameraDistance * Math.sin(cameraPitch);
            const z = target.z + cameraDistance * Math.cos(cameraYaw) * Math.cos(cameraPitch);
            camera.setPosition(x, y, z);
            camera.lookAt(target.x, target.y, target.z);
        };

        canvas.addEventListener('mousedown', (e) => {
            isMouseDown = true;
            isPanning = e.button === 2;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
            e.preventDefault();
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!isMouseDown) return;
            const deltaX = e.clientX - lastMouseX;
            const deltaY = e.clientY - lastMouseY;

            if (isPanning) {
                const panSpeed = cameraDistance * 0.001;
                const right = new pc.Vec3();
                const up = new pc.Vec3();
                const cameraTransform = camera.getWorldTransform();
                cameraTransform.getX(right);
                cameraTransform.getY(up);
                right.mulScalar(-deltaX * panSpeed);
                up.mulScalar(deltaY * panSpeed);
                target.add(right).add(up);
            } else {
                const rotateSpeed = 0.005;
                cameraYaw -= deltaX * rotateSpeed;  // Reversed direction
                cameraPitch = Math.max(-Math.PI/2 + 0.1, Math.min(Math.PI/2 - 0.1, cameraPitch + deltaY * rotateSpeed));  // Switched back up/down
            }

            updateCameraPosition();
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
            e.preventDefault();
        });

        canvas.addEventListener('mouseup', () => {
            isMouseDown = false;
            isPanning = false;
        });

        canvas.addEventListener('wheel', (e) => {
            const zoomSpeed = 0.001;
            const zoomDelta = e.deltaY * zoomSpeed * cameraDistance;
            cameraDistance = Math.max(0.001, Math.min(100, cameraDistance + zoomDelta));
            updateCameraPosition();
            e.preventDefault();
        });

        canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        updateCameraPosition();
        
        // Try to load actual SOG file
        try {
            let url;
            if (fileData.startsWith('blob:') || fileData.startsWith('http')) {
                url = fileData;
            } else {
                const binaryString = atob(fileData);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                const blob = new Blob([bytes], { type: 'application/octet-stream' });
                url = URL.createObjectURL(blob);
            }
            
            const asset = new pc.Asset(fileName, 'gsplat', {
                url: url,
                filename: fileName
            });
            
            asset.ready(() => {
                const entity = new pc.Entity('GaussianSplat');
                entity.addComponent('gsplat', {
                    asset: asset
                });
                app.root.addChild(entity);
                
                // Focus camera on splat
                //camera.setPosition(5, 2, 5);
                //camera.lookAt(0, 0, 0);
            });
            
            asset.on('error', (err) => {
                console.log('GSplat failed, showing placeholder:', err);
                // Fallback to green box
                const entity = new pc.Entity('SOGPlaceholder');
                entity.addComponent('render', { type: 'box' });
                const material = new pc.StandardMaterial();
                material.diffuse = new pc.Color(0.8, 0.2, 0.2); // Red for error
                entity.render.material = material;
                app.root.addChild(entity);
            });
            
            app.assets.add(asset);
            app.assets.load(asset);
            
        } catch (error) {
            console.log('Error processing SOG file:', error);
            // Fallback to green box
            const entity = new pc.Entity('SOGError');
            entity.addComponent('render', { type: 'box' });
            const material = new pc.StandardMaterial();
            material.diffuse = new pc.Color(0.2, 0.8, 0.2); // Green for fallback
            entity.render.material = material;
            app.root.addChild(entity);
        }
        

    };
    

}"""

def create_interface():
    # Create the main Gradio interface
    '''
            max-height: calc(80vh - 30px) !important;
            overflow-y: auto !important;
            overflow-x: auto !important;
    '''
    with gr.Blocks(js=playcanvas_js, title="Open Source 3D Reconstruction Toolbox for Gaussian Splats on AWS", theme=gr.themes.Ocean(), css="""
        /* Modal popup styles */
        #refine-modal-overlay {
            pointer-events: none !important;
        }
        #refine-modal-content {
            position: fixed !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            z-index: 20000 !important;
            background: #1e1e1e !important;
            color: #e0e0e0 !important;
            padding: 20px !important;
            border-radius: 8px !important;
            box-shadow: 0 0 0 9999px rgba(0,0,0,0.5), 0 4px 20px rgba(0,0,0,0.5) !important;
            border: 1px solid #444 !important;
            width: 600px !important;
            max-height: 80vh !important;
            overflow-x: hidden !important;
            overflow-y: auto !important;
            pointer-events: auto !important;
            white-space: normal !important;
            word-wrap: break-word !important;
        }
        #refine-modal-content .wrap,
        #refine-modal-content .options,
        #refine-modal-content ul[role="listbox"] {
            overflow: visible !important;
            z-index: 30000 !important;
            position: relative !important;
        }
        
        /* Cancel modal styles */
        #cancel-modal-overlay {
            pointer-events: none !important;
        }
        #cancel-modal-content {
            position: fixed !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            z-index: 1000 !important;
            background: #1e1e1e !important;
            color: #e0e0e0 !important;
            padding: 20px !important;
            border-radius: 8px !important;
            box-shadow: 0 0 0 5px #000, 0 4px 20px rgba(0,0,0,0.5) !important;
            border: 1px solid #444 !important;
            width: 400px !important;
            pointer-events: auto !important;
        }
        #refine-modal-content > div {
            padding: 15px !important;
        } 
        .padded-markdown {
            padding: 2px !important;
            margin: 2px !important;
            -ms-overflow-style: none !important;
            scrollbar-width: none !important;
        } 
        #refine-modal-content > gr-header {
            max-width: none !important; /* Remove max-width constraint from the header container */
            width: 90% !important;      /* Set a desired width for the modal header (adjust as needed) */
            margin-left: auto !important;
            margin-right: auto !important;
        }
        #refine-modal-content h3 {
            white-space: normal !important; /* Allow text to wrap naturally if it still exceeds the new width */
            word-wrap: break-word !important;
            max-width: 100% !important; /* Ensure the title uses all available space in its container */
        }

        /* Hide class for modal toggling via JS */
        .hide {
            display: none !important;
        }
        
        /* Files modal styles - initially hidden via CSS, shown via JS style.display='flex' which overrides this */
        #files-modal-content {
            display: none;
            position: fixed !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            z-index: 10000 !important;
            background: #1e1e1e !important;
            color: #e0e0e0 !important;
            padding: 20px !important;
            border-radius: 8px !important;
            box-shadow: 0 0 0 9999px rgba(0,0,0,0.5), 0 4px 20px rgba(0,0,0,0.5) !important;
            border: 1px solid #444 !important;
            width: 700px !important;
            max-height: 80vh !important;
            overflow-y: auto !important;
            pointer-events: auto !important;
        }
        
        /* Viewer modal styles - initially hidden via CSS, shown via JS style.display='flex' which overrides this */
        #viewer-modal-content {
            display: none;
            position: fixed !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            z-index: 1000 !important;
            background: #1e1e1e !important;
            color: #e0e0e0 !important;
            padding: 0px !important;
            border-radius: 8px !important;
            box-shadow: 0 0 0 9999px rgba(0,0,0,0.5), 0 4px 20px rgba(0,0,0,0.5) !important;
            border: 1px solid #444 !important;
            width: 90vw !important;
            max-width: 1200px !important;
            max-height: 90vh !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }
        #viewer-modal-content > * {
            padding: 20px !important;
        }
        #viewer-modal-content .tabs,
        #viewer-modal-content .tabitem,
        #viewer-modal-content [role="tabpanel"],
        #viewer-modal-content .tab-content,
        #viewer-modal-content > div,
        #viewer-modal-content .block,
        #viewer-modal-content .gap {
            background: #1e1e1e !important;
            color: #e0e0e0 !important;
        }
        #viewer-modal-content::-webkit-scrollbar {
            width: 8px;
        }
        #viewer-modal-content::-webkit-scrollbar-track {
            background: #1e1e1e !important;
        }
        #viewer-modal-content::-webkit-scrollbar-thumb {
            background: #444 !important;
            border-radius: 4px;
        }
        .close-button {
            margin-left: auto !important;
        }
        
        /* Page title styling with reddish-purple background */
        .gradio-container h1:first-of-type,
        .gradio-markdown h1:first-of-type,
        .markdown h1:first-of-type,
        h1:contains("Open Source 3D Reconstruction Toolbox") {
            background: linear-gradient(135deg, #8B0000, #4B0082) !important;
            color: white !important;
            padding: 15px 20px !important;
            border-radius: 8px !important;
            margin-bottom: 20px !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        }
        
        /* Alternative approach - target all h1 elements */
        h1 {
            background: linear-gradient(135deg, #8B0000, #4B0082) !important;
            color: white !important;
            padding: 15px 20px !important;
            border-radius: 8px !important;
            margin-bottom: 20px !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        }
        
        #viewer-container {
            width: 100%;
            height: 600px;
            background: #1a1a1a;
            position: relative;
            margin-top: 20px;
        }
        #renderCanvas {
            width: 100%;
            height: 100%;
            touch-action: none;
        }
        .logo-container {
            display: block;
            margin-left: auto;
            margin-right: 0;
            text-align: right;
        }
        
        /* Logo container styling */
        .logo-container {
            text-align: right;
        }
        
        #logo-column {
            margin-top: 15px;
        }
        
        /* Theme-based images */
        .theme-image-light {
            display: none;
        }
        
        .theme-image-dark {
            display: none;
        }
        
        /* Show appropriate image based on theme */
        @media (prefers-color-scheme: light) {
            .theme-image-light {
                display: block;
            }
            
            .theme-image-dark {
                display: none;
            }
        }
        
        @media (prefers-color-scheme: dark) {
            .theme-image-light {
                display: none;
            }
            
            .theme-image-dark {
                display: block;
            }
        }
    """) as interface:
        # PlayCanvas will be loaded by individual viewers when needed
        
        with gr.Row():
            with gr.Column():
                gr.HTML('<div><h1 style="background: linear-gradient(135deg, #8B0000, #4B0082); color: white; padding: 15px 20px; border-radius: 8px; margin-bottom: 0px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); box-shadow: 0 4px 8px rgba(0,0,0,0.2);">Open Source 3D Reconstruction Toolbox for Gaussian Splats on AWS</h1><div style="padding: 12px 16px; margin-top: 5px;">Generate and upload a metadata file and media (.mov, .mp4, .zip) for gaussian splat creation.<br>Browse and render generated splats in a local 3D web viewer.</div></div>')
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=1):
                        pass
                    with gr.Column(scale=0, elem_classes=["logo-container"], elem_id="logo-column"):
                        # Load logos directly from Gradio components and apply theme-based visibility
                        light_logo = gr.Image(
                            "../../assets/images/PoweredByAWS_horiz_RGB_1c_Gray850.png",
                            show_label=False,
                            container=False,
                            height=40,
                            width=None,
                            elem_classes=["theme-image-light"],
                            interactive=False,
                            sources=None,
                            buttons=None
                        )
                        dark_logo = gr.Image(
                            "../../assets/images/PoweredByAWS_horiz_RGB_1c_White.png",
                            show_label=False,
                            container=False,
                            height=40,
                            width=None,
                            elem_classes=["theme-image-dark"],
                            interactive=False,
                            sources=None,
                            buttons=[]
                        )
                        
                        # JavaScript to handle theme detection and switching
                        gr.HTML("""
                        <script>
                        // Add listener for theme changes if supported
                        if (window.matchMedia) {
                            const darkModeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
                            const lightModeMediaQuery = window.matchMedia('(prefers-color-scheme: light)');
                            
                            function updateTheme() {
                                const isDarkMode = darkModeMediaQuery.matches;
                                const isLightMode = lightModeMediaQuery.matches;
                                
                                document.querySelectorAll('.theme-image-dark').forEach(img => {
                                    img.style.display = isDarkMode ? 'block' : 'none';
                                });
                                
                                document.querySelectorAll('.theme-image-light').forEach(img => {
                                    img.style.display = isLightMode ? 'block' : 'none';
                                });
                            }
                            
                            // Set initial theme
                            updateTheme();
                            
                            // Listen for changes
                            darkModeMediaQuery.addEventListener('change', updateTheme);
                            lightModeMediaQuery.addEventListener('change', updateTheme);
                        }
                        </script>
                        """)

        with gr.Tabs():
            create_aws_configuration_tab()
            create_advanced_settings_tab()
            create_upload_aws_tab()
            create_combined_monitor_viewer_tab()
            create_debug_tab()
    return interface

# Modify your main execution code
if __name__ == "__main__":
    # Disable Hugging Face integration to avoid postMessage errors
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    check_aws_credentials()
    
    iface = create_interface()

    # Add favorites directory and temp 3D cache to allowed_paths
    import tempfile
    favorites_dir = os.path.join(os.path.dirname(__file__), "favorites")
    temp_3d_cache = os.path.join(tempfile.gettempdir(), "gradio_3d_cache")
    os.makedirs(temp_3d_cache, exist_ok=True)
    iface.launch(server_name="0.0.0.0", server_port=7861, share=False, allowed_paths=[favorites_dir, temp_3d_cache],
                 head="""<script src="https://code.playcanvas.com/playcanvas-2.17.0.min.js"></script>
<script>
// createSOGViewer defined in <head> so it's in the true global scope.
// In Gradio 6, the js= param on gr.Blocks runs in a sandboxed AsyncFunction scope
// where window/globalThis assignments don't persist to .change()/.then() JS callbacks.
// But <head> scripts run in the real global scope, so this function IS accessible everywhere.
window.createSOGViewer = function(fileData, fileName, fileSize) {
    if (!window.pc) { console.error('PlayCanvas not loaded'); return; }
    var container = document.getElementById('sog-container-monitor') || document.getElementById('sog-container');
    if (!container) { console.error('No sog-container found'); return; }

    // Force container visible
    container.style.display = 'block';
    var el = container.parentElement;
    while (el && el.id !== 'viewer-modal-content') {
        if (el.style.display === 'none' || window.getComputedStyle(el).display === 'none') el.style.display = 'block';
        el = el.parentElement;
    }

    var W = container.offsetWidth || 800;
    var H = 700;
    // Set explicit width/height attributes on canvas BEFORE creating pc.Application
    container.innerHTML = '<canvas id="pc-canvas" width="' + W + '" height="' + H + '" style="width:' + W + 'px;height:' + H + 'px;display:block;background:#1a1a1a;"></canvas>';
    var canvas = document.getElementById('pc-canvas');

    var app = new pc.Application(canvas, {
        mouse: new pc.Mouse(canvas),
        touch: new pc.TouchDevice(canvas),
        keyboard: new pc.Keyboard(window),
        graphicsDeviceOptions: { antialias: true, alpha: false, preferWebGl2: true, powerPreference: 'high-performance' }
    });
    app.setCanvasFillMode(pc.FILLMODE_NONE);
    app.setCanvasResolution(pc.RESOLUTION_AUTO);
    app.start();
    app.resizeCanvas(W, H);

    // Camera
    var camera = new pc.Entity('Camera');
    camera.addComponent('camera', { clearColor: new pc.Color(0.1,0.1,0.1,1), fov: 75, nearClip: 0.1, farClip: 1000 });
    camera.setPosition(0, 2, 5);
    app.root.addChild(camera);

    // Light
    var light = new pc.Entity('Light');
    light.addComponent('light', { type: 'directional', intensity: 1 });
    light.setEulerAngles(45, 30, 0);
    app.root.addChild(light);

    // Orbit controls
    var yaw = 0, pitch = 0.3, dist = 10;
    var target = new pc.Vec3(0,0,0);
    var updateCam = function() {
        camera.setPosition(
            target.x + dist * Math.sin(yaw) * Math.cos(pitch),
            target.y + dist * Math.sin(pitch),
            target.z + dist * Math.cos(yaw) * Math.cos(pitch)
        );
        camera.lookAt(target);
    };
    var dragging = false, panning = false, lx = 0, ly = 0;
    canvas.addEventListener('mousedown', function(e) { dragging = true; panning = e.button === 2; lx = e.clientX; ly = e.clientY; e.preventDefault(); });
    canvas.addEventListener('mouseup', function() { dragging = false; panning = false; });
    canvas.addEventListener('mousemove', function(e) {
        if (!dragging) return;
        var dx = e.clientX - lx, dy = e.clientY - ly;
        if (panning) {
            var ps = dist * 0.001;
            var r = new pc.Vec3(), u = new pc.Vec3();
            var ct = camera.getWorldTransform(); ct.getX(r); ct.getY(u);
            r.mulScalar(-dx * ps); u.mulScalar(dy * ps);
            target.add(r).add(u);
        } else {
            yaw -= dx * 0.005;
            pitch = Math.max(-1.5, Math.min(1.5, pitch + dy * 0.005));
        }
        updateCam(); lx = e.clientX; ly = e.clientY; e.preventDefault();
    });
    canvas.addEventListener('wheel', function(e) { dist = Math.max(0.01, Math.min(100, dist + e.deltaY * 0.001 * dist)); updateCam(); e.preventDefault(); });
    canvas.addEventListener('contextmenu', function(e) { e.preventDefault(); });
    updateCam();

    // Resize handler
    window.addEventListener('resize', function() {
        var rect = container.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) app.resizeCanvas(rect.width, H);
    });

    // Load GSplat
    try {
        var url;
        if (fileData.startsWith('blob:') || fileData.startsWith('http')) {
            url = fileData;
        } else {
            var bin = atob(fileData);
            var bytes = new Uint8Array(bin.length);
            for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
            var blob = new Blob([bytes], { type: 'application/octet-stream' });
            url = URL.createObjectURL(blob);
        }
        var asset = new pc.Asset(fileName, 'gsplat', { url: url, filename: fileName });
        asset.ready(function() {
            var entity = new pc.Entity('GaussianSplat');
            entity.addComponent('gsplat', { asset: asset });
            app.root.addChild(entity);
            console.log('GSplat loaded successfully');
        });
        asset.on('error', function(err) { console.error('GSplat error:', err); });
        app.assets.add(asset);
        app.assets.load(asset);
    } catch (error) { console.error('Error loading GSplat:', error); }
};
window.createSPZViewer = async function(fileData, fileName) {
    var container = document.getElementById('spz-container-monitor') || document.getElementById('sog-container-monitor') || document.getElementById('spz-container') || document.getElementById('sog-container');
    if (!container) return;
    container.innerHTML = '<canvas id="spz-canvas" style="width:100%;height:700px;"></canvas>';
    var canvas = document.getElementById('spz-canvas');
    canvas.width = (container.offsetWidth || 800) * (window.devicePixelRatio || 1);
    canvas.height = 700 * (window.devicePixelRatio || 1);
    canvas.style.width = (container.offsetWidth || 800) + 'px';
    canvas.style.height = '700px';
    function loadAndRender() {
        var engine = new BABYLON.Engine(canvas, true);
        var scene = new BABYLON.Scene(engine);
        scene.clearColor = new BABYLON.Color4(0.1, 0.1, 0.1, 1);
        var camera = new BABYLON.ArcRotateCamera('cam', -Math.PI/2, Math.PI/3, 5, BABYLON.Vector3.Zero(), scene);
        camera.attachControl(canvas, true);
        camera.wheelPrecision = 50;          // lower = faster zoom (default ~3), higher = slower
        camera.panningSensibility = 100;     // lower = faster pan (default ~1000), higher = slower
        camera.panningInertia = 0.9;
        camera.lowerRadiusLimit = 0.01;      // allow zooming very close (default ~0.1)
        new BABYLON.HemisphericLight('light', new BABYLON.Vector3(0,1,0), scene);
        var blobUrl;
        if (fileData.startsWith('blob:') || fileData.startsWith('http')) {
            blobUrl = fileData;
        } else {
            var bytes = new Uint8Array(atob(fileData).split('').map(function(c){return c.charCodeAt(0);}));
            var blob = new Blob([bytes], {type: 'application/octet-stream'});
            blobUrl = URL.createObjectURL(blob);
        }
        BABYLON.SceneLoader.Append('', blobUrl, scene, function() {
            scene.createDefaultCameraOrLight(true, true, true);
            if (!fileData.startsWith('blob:') && !fileData.startsWith('http')) URL.revokeObjectURL(blobUrl);
        }, null, null, '.spz');
        engine.runRenderLoop(function() { scene.render(); });
        window.addEventListener('resize', function() { engine.resize(); });
    }
    if (!window.BABYLON) {
        var s1 = document.createElement('script'); s1.src = 'https://cdn.babylonjs.com/babylon.js';
        document.head.appendChild(s1);
        s1.onload = function() {
            var s2 = document.createElement('script'); s2.src = 'https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js';
            document.head.appendChild(s2);
            s2.onload = loadAndRender;
        };
    } else { loadAndRender(); }
};
</script>""")
