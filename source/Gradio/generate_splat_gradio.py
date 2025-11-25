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
import uuid
import json
import boto3
import time
import threading
import gradio as gr
import boto3.s3.transfer
from refine_splat import refine_splat

print(f"Gradio Version: {gr.__version__}")

class SharedState:
    def __init__(self):
        self.aws_region = "us-east-1"
        self.stack_unique_id = "j6x8xn"
        self.s3_bucket = f"3dgs-bucket-{self.stack_unique_id}"
        self.ddb_table_name = f"3dgs-table-{self.stack_unique_id}"
        self.s3_input = "workflow-input"
        self.s3_output = "workflow-output"
        self.media_input = "media-input"
        self.instance = "ml.g5.4xlarge"
        self.use_spot_instance = "false"
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
        self.enable_sogs = "true"
        self.enable_usdz = "true"
        self.remove_bg = "false"
        self.remove_objects = "false"
        self.object_removal_action = "erase"
        self.objects_to_remove = []
        self.source_coordinate = "arkit"
        self.pose_world_to_cam = "true"
        self.log_verbosity = "info"
        self.mask_threshold = 0.6
        self.model_3d = None
        self.rotate_splat = "true"
        self.crop_output_bounds = "false"
        self.crop_mode = "environment"
        self.video_start_time = 0.0
        self.video_stop_time = None

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
        import os
        import boto3
        
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
        
        print("\nDEBUG: Final parsed values:")
        print(f"Access Key present: {bool(access_key)}")
        print(f"Secret Key present: {bool(secret_key)}")
        print(f"Session Token present: {bool(session_token)}")
                
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
    """Get thumbnail URL for a job if it exists"""
    try:
        s3_client = boto3.client('s3')
        bucket_name = shared_state.s3_bucket
        output_prefix = shared_state.s3_output or "workflow-output"
        thumbnail_key = f"{output_prefix}/{job_id}/render_thumbnail.png"
        
        # Check if thumbnail exists
        try:
            s3_client.head_object(Bucket=bucket_name, Key=thumbnail_key)
            return generate_presigned_url(bucket_name, thumbnail_key)
        except:
            # Try to find thumbnail with filename prefix
            try:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=f"{output_prefix}/{job_id}/"
                )
                if 'Contents' in response:
                    # Look for files ending with _thumbnail.png
                    for obj in response['Contents']:
                        if obj['Key'].endswith('_thumbnail.png'):
                            return generate_presigned_url(bucket_name, obj['Key'])
            except:
                pass
            return None
    except Exception as e:
        print(f"Error getting thumbnail for {job_id}: {e}")
        return None

def refresh_s3_contents():
    """Refresh contents from DynamoDB and return grouped data by job ID"""
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
            print(f"DEBUG: Job {job_id[:8]}... has status: '{status}' (type: {type(status)})")
            
            # Make status check case-insensitive and flexible
            status_lower = str(status).lower()
            if status_lower not in ['complete', 'completed']:
                print(f"DEBUG: Skipping job {job_id[:8]}... - status '{status}' doesn't match")
                continue
            
            print(f"DEBUG: Including job {job_id[:8]}... - status matches")
            
            job_id = item['uuid']
            output_files = item.get('outputFiles', [])
            print(f"DEBUG: Job {job_id[:8]}... has {len(output_files)} output files")
            print(f"DEBUG: Output files content: {output_files}")
            
            if not output_files:
                print(f"DEBUG: Skipping job {job_id[:8]}... - no output files")
                continue
            
            last_modified = item.get('endTimestamp', item.get('startTimestamp', ''))
            
            jobs_dict[job_id] = {
                'files': [],
                'last_modified': last_modified,
                'thumbnail_url': get_thumbnail_url(job_id)
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
                thumbnail_html = f'<a href="{job_data["thumbnail_url"]}" download="{job_id}_thumbnail.png" style="display:inline-block;"><img src="{job_data["thumbnail_url"]}" style="width:60px;height:60px;object-fit:cover;border-radius:4px;cursor:pointer;" alt="Thumbnail" loading="lazy" title="Click to download"/></a>'
            
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
                rotate_splat, crop_output_bounds, crop_mode, enable_spz, enable_sogs, video_start_time, video_stop_time):
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
        "useSpotInstance": "false",
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
            "videoStopTime": video_stop_time if video_stop_time is not None else None
        },
        "imageProcessing": {
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
            "model": training_model
        },
        "postProcessing": {
            "rotateSplat": rotate_splat == "true",
            "cropOutputBounds": crop_output_bounds == "true" if isinstance(crop_output_bounds, str) else crop_output_bounds,
            "cropMode": crop_mode if isinstance(crop_mode, str) else "environment",
            "enableSpz": enable_spz == "true",
            "enableSogs": enable_sogs == "true",
            "enableUsdz": "true"
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
                  pose_world_to_cam, log_verbosity, mask_threshold, enable_spz, enable_sogs, media_input_prefix="media-input", rotate_splat="babylon"):
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
        enable_sogs = getattr(enable_sogs, 'value', enable_sogs)
        remove_bg = getattr(remove_bg, 'value', remove_bg)
        remove_objects = getattr(remove_objects, 'value', remove_objects)
        source_coordinate = getattr(source_coordinate, 'value', source_coordinate)
        pose_world_to_cam = getattr(pose_world_to_cam, 'value', pose_world_to_cam)
        log_verbosity = getattr(log_verbosity, 'value', log_verbosity)
        mask_threshold = getattr(mask_threshold, 'value', mask_threshold)
        media_input_prefix = getattr(media_input_prefix, 'value', media_input_prefix)
        rotate_splat = getattr(rotate_splat, 'value', rotate_splat)

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
            },
            "imageProcessing": {
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
                "model": training_model
            },
            "postProcessing": {
                "rotateSplat": rotate_splat == "true",
                "cropOutputBounds": shared_state.crop_output_bounds == "true",
                "cropMode": shared_state.crop_mode,
                "enableSpz": enable_spz == "true",
                "enableSogs": enable_sogs == "true",
                "enableUsdz": shared_state.enable_usdz == "true"
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
                upload_button = gr.Button("Upload to AWS", variant="primary", elem_classes=["orange-button"])

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
                                "videoStopTime": shared_state.video_stop_time if shared_state.video_stop_time is not None else None
                            },
                            "imageProcessing": {
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
                                "model": shared_state.model
                            },
                            "postProcessing": {
                                "rotateSplat": shared_state.rotate_splat == "true",
                                "cropOutputBounds": shared_state.crop_output_bounds == "true",
                                "cropMode": shared_state.crop_mode,
                                "enableSpz": shared_state.enable_spz == "true",
                                "enableSogs": shared_state.enable_sogs == "true",
                                "enableUsdz": shared_state.enable_usdz == "true"
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
                gr.Markdown("### AWS Settings")
                aws_region = gr.Textbox(label="AWS Region", value=shared_state.aws_region)
                ddb_table_name = gr.Textbox(label="DynamoDB Table Name", value=shared_state.ddb_table_name)
                s3_bucket = gr.Textbox(label="S3 Bucket Name", value=shared_state.s3_bucket)
                s3_input = gr.Textbox(label="S3 Input Prefix", value=shared_state.s3_input)
                s3_output = gr.Textbox(label="S3 Output Prefix", value="workflow-output")
                media_input = gr.Textbox(label="Media Input Prefix", value="media-input")
                instance = gr.Dropdown(
                    label="Instance Type",
                    choices=[
                        "ml.g5.4xlarge",
                        "ml.g5.8xlarge",
                        #"ml.g5.12xlarge",
                        "ml.g6.4xlarge",
                        "ml.g6.8xlarge",
                        #"ml.g6.12xlarge",
                        "ml.g6e.4xlarge"],
                        #"ml.g6e.12xlarge"],
                    value=shared_state.instance
                )
                use_spot_instance = gr.Radio(
                    label="Compute Type",
                    choices=[("AWS Batch (Spot Instances - Up to 90% cost savings)", "true"), ("SageMaker (On-Demand)", "false")],
                    value=shared_state.use_spot_instance,
                    info="Batch uses spot instances for significant cost savings but may have longer queue times"
                )

                def update_shared_state(region, ddb_table, bucket, input_prefix, output_prefix, media_prefix, inst, spot):
                    #print(f"DEBUG: Updating shared_state.instance from '{shared_state.instance}' to '{inst}'")
                    #print(f"DEBUG: Updating shared_state.use_spot_instance from '{shared_state.use_spot_instance}' to '{spot}'")
                    shared_state.aws_region = region
                    shared_state.ddb_table_name = ddb_table
                    shared_state.s3_bucket = bucket
                    shared_state.s3_input = input_prefix
                    shared_state.s3_output = output_prefix
                    shared_state.media_input = media_prefix
                    shared_state.instance = inst
                    shared_state.use_spot_instance = spot
                    #print(f"DEBUG: Updated shared_state.instance to '{shared_state.instance}'")
                    return "AWS configuration updated"

                # Immediately update shared_state when values change
                def update_instance_type(inst):
                    #print(f"DEBUG: Instance type changed to: {inst}")
                    shared_state.instance = inst
                    #print(f"DEBUG: Shared state instance updated to: {shared_state.instance}")
                    #print(f"DEBUG: Main shared_state object ID: {id(shared_state)}")
                
                def update_spot_instance(spot):
                    #print(f"DEBUG: Spot instance setting changed to: {spot}")
                    shared_state.use_spot_instance = spot
                    #print(f"DEBUG: Shared state spot instance updated to: {shared_state.use_spot_instance}")
                    #print(f"DEBUG: Main shared_state object ID: {id(shared_state)}")
                
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
                gr.Markdown("### Image Processing")
                filter_blurry = gr.Radio(
                label="Filter Blurry Images",
                choices=["true", "false"],
                value="true"
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
                        step=0.01
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
                    choices=["colmap", "glomap", "vggt", "map_anything"],
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
                    minimum=1000,
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
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Post Processing")
                rotate_splat = gr.Radio(
                    label="Rotate Splat for Gradio Viewer",
                    choices=["true", "false"],
                    value="true"
                )
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
                enable_spz = gr.Radio(
                    label="Enable SPZ Export",
                    choices=["true", "false"],
                    value="true"
                )
                enable_sogs = gr.Radio(
                    label="Enable SOGS Export",
                    choices=["true", "false"],
                    value="true"
                )
                enable_usdz = gr.Radio(
                    label="Enable USDZ Export",
                    choices=["true", "false"],
                    value="true"
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


                def update_advanced_settings(*args):
                    # Update shared state with all advanced settings
                    (shared_state.sfm, shared_state.model, shared_state.faces, 
                     shared_state.bg_model, shared_state.filter_blurry,
                     shared_state.max_images, shared_state.video_start_time, shared_state.video_stop_time, shared_state.sfm_enable, 
                     shared_state.enhanced_feature, shared_state.matching_method,
                     shared_state.use_colmap_model, shared_state.use_transform_json,
                     shared_state.training_enable, shared_state.max_steps, shared_state.enable_spz, shared_state.enable_sogs, shared_state.enable_usdz,
                     shared_state.rotate_splat, shared_state.crop_output_bounds, shared_state.crop_mode,
                     shared_state.spherical_enable,
                     shared_state.remove_bg, shared_state.remove_objects,
                     shared_state.object_removal_action, shared_state.objects_to_remove, shared_state.source_coordinate, shared_state.pose_world_to_cam,
                     shared_state.log_verbosity, shared_state.mask_threshold) = args
                    return "Advanced settings updated"

                # Get all advanced settings components after they're defined
                advanced_components = [
                    sfm, model, faces, bg_model, filter_blurry,
                    max_images, video_start_time, video_stop_time, sfm_enable, enhanced_feature, matching_method,
                    use_colmap_model, use_transform_json, training_enable,
                    max_steps, enable_spz, enable_sogs, enable_usdz,
                    rotate_splat, crop_output_bounds, crop_mode,
                    spherical_enable, remove_bg, remove_objects,
                    object_removal_action, objects_to_remove, source_coordinate, pose_world_to_cam,
                    log_verbosity, mask_threshold
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
                        'enable_sogs': settings[16],
                        'enable_usdz': settings[17],
                        'rotate_splat': settings[18],
                        'crop_output_bounds': settings[19],
                        'crop_mode': settings[20],
                        'spherical_enable': settings[21],
                        'remove_bg': settings[22],
                        'remove_objects': settings[23],
                        'object_removal_action': settings[24],
                        'objects_to_remove': settings[25],
                        'source_coordinate': settings[26],
                        'pose_world_to_cam': settings[27],
                        'log_verbosity': settings[28],
                        'mask_threshold': settings[29]
                    }
                    
                    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
                    os.makedirs(configs_dir, exist_ok=True)
                    config_file = os.path.join(configs_dir, f"{config_name.strip()}.json")
                    
                    try:
                        with open(config_file, 'w') as f:
                            json.dump(config_data, f, indent=2)
                        return f"Configuration '{config_name}' saved successfully", gr.update(choices=get_saved_configs())
                    except Exception as e:
                        return f"Error saving configuration: {str(e)}", gr.update()

                
                def load_configuration(config_name):
                    if not config_name:
                        return ["Please select a configuration"] + [gr.update() for _ in range(30)]
                    
                    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
                    config_file = os.path.join(configs_dir, f"{config_name}.json")
                    
                    try:
                        with open(config_file, 'r') as f:
                            config_data = json.load(f)
                        
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
                            config_data.get('enable_sogs', 'true'),
                            config_data.get('enable_usdz', 'true'),
                            config_data.get('rotate_splat', 'true'),
                            config_data.get('crop_output_bounds', 'false'),
                            config_data.get('crop_mode', 'environment'),
                            config_data.get('spherical_enable', 'false'),
                            config_data.get('remove_bg', 'false'),
                            config_data.get('remove_objects', 'false'),
                            config_data.get('object_removal_action', 'erase'),
                            config_data.get('objects_to_remove', []),
                            config_data.get('source_coordinate', 'arkit'),
                            config_data.get('pose_world_to_cam', 'true'),
                            config_data.get('log_verbosity', 'info'),
                            config_data.get('mask_threshold', 0.6)
                        ]
                    except Exception as e:
                        return [f"Error loading configuration: {str(e)}"] + [gr.update() for _ in range(30)]
                
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
        
        # Generate a presigned URL for the file
        presigned_url = generate_presigned_url(bucket_name, file_key)
        
        if not presigned_url:
            return gr.update(value=None), "Error generating URL", ""
        
        # Add cache-busting parameter for SPZ files to avoid browser caching issues
        #if filename.lower().endswith('.spz'):
        #    import time
        #    cache_buster = int(time.time())
        #    separator = '&' if '?' in presigned_url else '?'
        #    presigned_url += f"{separator}cb={cache_buster}"
            
        # Store current model info
        shared_state.current_model_url = presigned_url
        shared_state.current_model_key = file_key
        
        # Estimate loading time based on file size
        estimated_time = file_size_mb * 0.5 if file_size_mb else 25
        
        return gr.update(value=presigned_url), f"Loading {filename}...", ""
        
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
                <div style="background: #f0f0f0; border-radius: 10px; overflow: hidden; height: 20px;">
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
            <div style="background: #f0f0f0; border-radius: 10px; overflow: hidden; height: 20px;">
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
            fileName: '{filename}',
            fileSize: '{file_size}'
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
        
        # Check if it's a SOG file
        if filename.lower().endswith('.sog'):
            bucket_name = shared_state.s3_bucket
            output_prefix = shared_state.s3_output or "workflow-output"
            job_id = selected_row[0]
            file_key = f"{output_prefix}/{job_id}/{filename}"
            
            presigned_url = generate_presigned_url(bucket_name, file_key)
            if presigned_url:
                # Fetch SOG data and use the working method
                import requests
                import base64
                try:
                    response = requests.get(presigned_url)
                    file_data = base64.b64encode(response.content).decode('utf-8')
                    file_size = f"{len(response.content):,} bytes"
                    file_info = f"Loaded: {filename} ({file_size})"
                except Exception as e:
                    file_data = ""
                    file_info = f"Error: {str(e)}"
                
                return (
                    gr.update(value=None), 
                    f"SOG file loaded: {filename}",
                    gr.update(value=file_data),
                    gr.update(value=file_info),
                    gr.update(value=None),
                    f"SOG file loaded - switch to SOG Viewer tab"
                )
            else:
                return (
                    gr.update(value=None), 
                    "Error loading SOG file",
                    gr.update(value="<div style='color: red;'>Error generating SOG file URL</div>"),
                    "Error generating SOG file URL",
                    gr.update(value=None),
                    "Error loading SOG file"
                )
        
        # Check if it's a video file
        elif filename.lower().endswith('.mp4'):
            bucket_name = shared_state.s3_bucket
            output_prefix = shared_state.s3_output or "workflow-output"
            job_id = selected_row[0]
            file_key = f"{output_prefix}/{job_id}/{filename}"
            
            presigned_url = generate_presigned_url(bucket_name, file_key)
            if presigned_url:
                #print(f"[DEBUG] Video URL generated: {presigned_url}")
                #print(f"[DEBUG] Video filename: {filename}")
                return (
                    gr.update(value=None), 
                    "Video selected - check Video Preview tab",
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(value=presigned_url),
                    f"Loaded video: {filename}"
                )
            else:
                return (
                    gr.update(value=None), 
                    "Error loading video",
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(value=None),
                    "Error generating video URL"
                )
        else:
            # Handle other 3D models (PLY, SPZ, GLB, etc.)
            result = handle_view_with_progress(selected_row)
            return (
                result[0], 
                result[1],
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=None),
                "3D model selected - check 3D Model Viewer tab"
            )
            
    except Exception as e:
        error_msg = f"Error viewing file: {str(e)}"
        return (
            gr.update(value=None), 
            error_msg,
            gr.update(value=""),
            gr.update(value=f"Error: {error_msg}"),
            gr.update(value=None),
            error_msg
        )

def fetch_current_pricing():
    """Fetch current AWS pricing using the Pricing API"""
    try:
        pricing_client = boto3.client('pricing', region_name='us-east-1')
        
        instance_types = ['ml.g5.4xlarge', 'ml.g5.8xlarge', 'ml.g6.4xlarge', 'ml.g6.8xlarge']
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
                    'ml.g5.4xlarge': 1.624, 'ml.g5.8xlarge': 3.248,
                    'ml.g6.4xlarge': 1.624, 'ml.g6.8xlarge': 3.248
                }
                pricing_data[instance_type] = fallback_prices.get(instance_type, 1.624)
        
        return pricing_data
    except Exception as e:
        print(f"Error fetching pricing data: {e}")
        # Return fallback pricing
        return {
            'ml.g5.4xlarge': 1.624, 'ml.g5.8xlarge': 3.248,
            'ml.g6.4xlarge': 1.624, 'ml.g6.8xlarge': 3.248
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
        
        response = table.get_item(Key={'uuid': job_id})
        print(f"DynamoDB response: {response}")
        
        if 'Item' in response:
            item = response['Item']
            # Format metadata as a left-justified table
            sorted_keys = sorted([k for k in item.keys() if k != 'uuid'])
            
            table_rows = []
            
            # Calculate estimated cost if we have the necessary data
            estimated_cost = "N/A"
            if 'instanceType' in item and 'elapsedTimestamp' in item:
                instance_type = str(item['instanceType'])
                elapsed_str = str(item['elapsedTimestamp'])
                is_spot = str(item.get('useSpotInstance', 'false')).lower() == 'true'
                
                try:
                    # Parse elapsed time (format: "H:MM:SS.microseconds" or "X days, H:MM:SS.microseconds")
                    if ':' in elapsed_str:
                        # Handle format like "3 days, 9:46:28.647185" or "0:46:28.647185"
                        if 'days' in elapsed_str:
                            # Parse "X days, H:MM:SS" format
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
                                seconds = float(time_parts[2].split('.')[0])  # Remove microseconds
                            else:
                                hours = minutes = seconds = 0
                        total_seconds = hours * 3600 + minutes * 60 + seconds
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
            error_content = f"No metadata found for job ID: {job_id}<br/><br/>This could mean:<br/>- The job hasn't been processed yet<br/>- The job was created before metadata tracking<br/>- The job ID is incorrect"
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
        
        #print(f"Debug - selected_data: {selected_data}")  # Debug print
        
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
        favorite_path = os.path.join(favorites_dir, favorite_filename)
        
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
        gr.Markdown("### Favorites")
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
        
        # Create tabbed interface for viewer types
        with gr.Tabs():
            with gr.Tab("3D Model Viewer (Babylon.js)"):
                viewer = gr.Model3D(
                    label="3D Viewer",
                    clear_color=[0.2, 0.2, 0.2, 1.0],
                    height=900,
                    interactive=True
                )

                viewer_status = gr.Textbox(
                    label="Viewer Status",
                    interactive=False,
                    show_label=True,
                    value=""
                )
            
            with gr.Tab("SOG Viewer (PlayCanvas)"):
                sog_viewer = gr.HTML(
                    value="<div id='sog-container' style='height: 900px; background: #1a1a1a; border: 1px solid #444; display: flex; align-items: center; justify-content: center; color: white;'>Select a .sog file to view</div><script>const observer = new MutationObserver(() => { const container = document.getElementById('sog-container'); if(container && container.offsetWidth > 0 && window.sogData && !window.sogLoaded) { globalThis.createSOGViewer(window.sogData.fileData, window.sogData.fileName, window.sogData.fileSize); window.sogLoaded = true; observer.disconnect(); } }); observer.observe(document.body, {childList: true, subtree: true, attributes: true});</script>",
                    label="SOG Viewer"
                )
                
                sog_status = gr.Textbox(
                    label="SOG Viewer Status",
                    interactive=False,
                    show_label=True,
                    value=""
                )
            
            with gr.Tab("Video Preview"):
                video_viewer = gr.Video(
                    label="Trajectory Video",
                    height=900,
                    interactive=False,
                    format="mp4"
                )
                
                video_status = gr.Textbox(
                    label="Video Status",
                    interactive=False,
                    show_label=True,
                    value=""
                )
                

        
        # Connect favorite button handlers for loading files
        for btn, path, name in favorite_buttons:
            btn.click(
                fn=lambda p=path, n=name: (p, f"Loaded local file: {n}"),
                inputs=[],
                outputs=[viewer, viewer_status]
            )
        
        # Store the current model URL
        current_model = gr.State(None)

        # File browser section
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Contents")
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

        # Create refine status before metadata display
        refine_status = gr.Textbox(
            label="Refine Status",
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
        
        # Update the select event handler
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
        
        # Connect download button
        download_btn.click(
            fn=handle_download,
            inputs=[selected_data],
            outputs=[download_iframe]
        )
        
        # Create hidden components for SOG data and filename tracking
        sog_file_data = gr.Textbox(visible=False)
        sog_file_info = gr.Textbox(visible=False)
        current_filename = gr.Textbox(visible=False)
        
        # Connect view button to handle 3D models, SOG files, and videos
        view_btn.click(
            fn=handle_view_multi,
            inputs=[selected_data],
            outputs=[viewer, viewer_status, sog_file_data, sog_file_info, video_viewer, video_status]
        )
        
        # Separate handler to update filename for tab switching
        view_btn.click(
            fn=lambda selected_row: selected_row[1] if selected_row else "",
            inputs=[selected_data],
            outputs=[current_filename]
        )
        
        # Add JavaScript-only click handler for tab switching using the filename
        current_filename.change(
            fn=None,
            inputs=[current_filename],
            js="""(filename) => {
                console.log('Tab switching JS triggered with filename:', filename);
                if(filename) {
                    const fname = filename.toLowerCase();
                    console.log('Processing filename:', fname);
                    
                    setTimeout(() => {
                        const allTabs = document.querySelectorAll('button[role="tab"]');
                        console.log('Found tabs:', allTabs.length);
                        
                        allTabs.forEach((tab, index) => {
                            const tabText = tab.textContent.trim();
                            console.log('Tab', index, ':', tabText);
                            
                            if(fname.endsWith('.mp4') && tabText.includes('Video Preview')) {
                                console.log('Clicking Video Preview tab');
                                tab.click();
                            } else if(fname.endsWith('.sog') && tabText.includes('SOG Viewer')) {
                                console.log('Clicking SOG Viewer tab');
                                tab.click();
                            } else if(!fname.endsWith('.sog') && !fname.endsWith('.mp4') && tabText.includes('3D Model Viewer')) {
                                console.log('Clicking 3D Model Viewer tab');
                                tab.click();
                            }
                        });
                    }, 300);
                }
            }"""
        )
        
        # Now connect favorites tab switching after current_filename is defined
        for btn, path, name in favorite_buttons:
            btn.click(
                fn=lambda n=name: n,
                inputs=[],
                outputs=[current_filename]
            )
        
        # Store SOG data and wait for tab to be visible
        sog_file_data.change(
            None,
            inputs=[sog_file_data, sog_file_info],
            outputs=None,
            js="(fileData, fileInfo) => { if(fileData && fileInfo.includes('Loaded:')) { const parts = fileInfo.split(' '); const fileName = parts[1]; const fileSize = parts[2] + ' ' + parts[3]; window.sogData = {fileData, fileName, fileSize}; window.sogLoaded = false; } }"
        )
        
        # Connect add to favorites button
        add_favorite_btn.click(
            fn=add_to_favorites,
            inputs=[selected_data],
            outputs=[gr.Textbox(visible=False)]
        )
        
        # Connect refine button - pass current shared_state values
        refine_btn.click(
            fn=lambda selected_data: refine_splat(selected_data, shared_state.instance, shared_state.use_spot_instance),
            inputs=[selected_data],
            outputs=[refine_status]
        )
        
        # Add SuperSplat link at the bottom of the viewer tab
        supersplat_link = gr.HTML(
            '<div style="text-align:center; margin:10px 0;"><a href="https://superspl.at/editor" target="_blank" style="display:inline-block; background:#f97316; color:white; padding:8px 16px; text-decoration:none; border-radius:6px; font-size:14px; font-weight:500;">🚀 Open SuperSplat Editor</a></div>'
        )
                

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
            
            start_time = str(item.get('startTimestamp', 'N/A'))
            instance_type = str(item.get('instanceType', 'N/A'))
            is_spot = str(item.get('useSpotInstance', 'false')).lower() == 'true'
            compute_type = "Batch Spot" if is_spot else "SageMaker"
            
            # Extract model name from training config
            model_name = "N/A"
            if 'training' in item and 'model' in item['training']:
                model_name = str(item['training']['model'])
            elif 'model' in item:
                model_name = str(item['model'])
            
            # Extract media filename from DynamoDB
            media_filename = "N/A"
            if 'filename' in item:
                media_filename = str(item['filename'])
            elif 's3' in item and 'inputKey' in item['s3']:
                s3_key = str(item['s3']['inputKey'])
                media_filename = s3_key.split('/')[-1]
            elif 'inputKey' in item:
                s3_key = str(item['inputKey'])
                media_filename = s3_key.split('/')[-1]
            
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
                # Use stored elapsed time for completed jobs
                elapsed_str = str(item.get('elapsedTimestamp', '0:00:00'))
                try:
                    if ':' in elapsed_str:
                        # Handle format like "3 days, 9:46:28.647185" or "0:46:28.647185"
                        if 'days' in elapsed_str:
                            # Parse "X days, H:MM:SS" format
                            days_part, time_part = elapsed_str.split(', ')
                            days = int(days_part.split()[0])
                            time_parts = time_part.split(':')
                            hours = int(time_parts[0]) + (days * 24)
                            minutes = int(time_parts[1])
                        else:
                            # Parse "H:MM:SS" format
                            time_parts = elapsed_str.split(':')
                            if len(time_parts) >= 3:
                                hours = int(time_parts[0])
                                minutes = int(time_parts[1])
                            else:
                                hours = minutes = 0
                        elapsed_time = f"{hours}h {minutes}m"
                    else:
                        elapsed_time = "N/A"
                except Exception as e:
                    print(f"Error parsing elapsed time '{elapsed_str}' for job {job_id}: {e}")
                    elapsed_time = "N/A"
            
            jobs_data.append([
                job_id,              # Full job ID (UUID)
                media_filename,      # Media input filename
                status,
                start_time,
                compute_type,
                instance_type,
                model_name,          # Model name
                elapsed_time
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
                        if ':' in elapsed_str:
                            # Handle format like "3 days, 9:46:28.647185" or "0:46:28.647185"
                            if 'days' in elapsed_str:
                                # Parse "X days, H:MM:SS" format
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
            return "No file selected"
            
        job_id = selected_data[0]  # First column is job ID
        filename = selected_data[1]  # Second column is filename
        
        # Generate the presigned URL
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
        
        # Return an iframe that will trigger the download
        return gr.HTML(f"""
            <iframe 
                src="{url}" 
                style="display: none;"
                onload="this.parentElement.removeChild(this)"
            ></iframe>
        """)
        
    except Exception as e:
        return f"Error downloading file: {str(e)}"

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
                        shared_state.rotate_splat,
                        shared_state.crop_output_bounds,
                        shared_state.crop_mode,
                        shared_state.enable_spz,
                        shared_state.enable_sogs,
                        shared_state.video_start_time,
                        shared_state.video_stop_time
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

def create_job_monitor_tab():
    with gr.Tab("Job Monitor"):
        with gr.Column():
            gr.Markdown("### Job Progress Monitor")
            gr.Markdown("Monitor the status and progress of your Gaussian splat reconstruction jobs.")
            
            refresh_jobs_btn = gr.Button(
                "Refresh Jobs", 
                variant="primary",
                size="sm"
            )
            
            jobs_table = gr.Dataframe(
                headers=["Job ID", "Media File", "Status", "Start Time", "Compute Type", "Instance Type", "Model", "Elapsed Time"],
                interactive=False,
                value=[],
                visible=True,
                elem_id="jobs_monitor_table"
            )
            
            # Job configuration display
            gr.Markdown("### Job Configuration")
            gr.Markdown("Select a job above to view its configuration details.")
            
            job_config_display = gr.HTML(
                value="<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Select a job to view configuration</div>",
                elem_id="job_config_display"
            )
            
            def refresh_jobs():
                """Refresh job monitoring data"""
                try:
                    jobs_data = get_job_progress_data()
                    return jobs_data, "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Select a job to view configuration</div>"
                except Exception as e:
                    print(f"Error refreshing jobs: {e}")
                    return [], "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Error loading jobs</div>"
            
            def on_job_select(evt: gr.SelectData, data):
                """Handle job selection to show configuration"""
                try:
                    # Handle DataFrame or list data
                    if hasattr(data, 'empty'):
                        if data.empty:
                            return "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>No job data available</div>"
                        data_list = data.values.tolist()
                    else:
                        if not data or len(data) == 0:
                            return "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>No job data available</div>"
                        data_list = data
                    
                    row_idx = evt.index[0]
                    if row_idx >= len(data_list):
                        return "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Invalid selection</div>"
                    
                    selected_row = data_list[row_idx]
                    if len(selected_row) < 1:
                        return "<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Invalid job data</div>"
                    
                    # Extract full job ID
                    job_id = selected_row[0]
                    
                    # Get full job metadata
                    return get_job_metadata(job_id)
                    
                except Exception as e:
                    print(f"Error in job selection: {e}")
                    return f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #e8e8e8; color: #333;'>Error: {str(e)}</div>"
            
            # Wire up event handlers
            refresh_jobs_btn.click(
                fn=refresh_jobs,
                inputs=[],
                outputs=[jobs_table, job_config_display]
            )
            
            jobs_table.select(
                fn=on_job_select,
                inputs=[jobs_table],
                outputs=[job_config_display]
            )

# Add the PlayCanvas JavaScript code globally
playcanvas_js = """
async () => {
    if (!window.pc) {
        const script = document.createElement('script');
        script.src = 'https://code.playcanvas.com/playcanvas-2.12.1.min.js';
        document.head.appendChild(script);
        await new Promise(resolve => script.onload = resolve);
    }
    
    globalThis.createSOGViewer = (fileData, fileName, fileSize) => {
        const container = document.getElementById('sog-container');
        if (!container) return;
        
        container.innerHTML = '<canvas id="pc-canvas" style="width: 100%; height: 900px; background: #1a1a1a;"></canvas>';
        const canvas = document.getElementById('pc-canvas');
        
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
        app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
        app.setCanvasResolution(pc.RESOLUTION_AUTO);
        
        // Handle resize for high DPI
        const handleResize = () => {
            const rect = container.getBoundingClientRect();
            const pixelRatio = window.devicePixelRatio || 1;
            canvas.width = rect.width * pixelRatio;
            canvas.height = rect.height * pixelRatio;
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';
            app.graphicsDevice.setResolution(canvas.width, canvas.height);
        };
        
        window.addEventListener('resize', handleResize);
        handleResize();
        
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
            cameraDistance = Math.max(0.5, Math.min(100, cameraDistance + zoomDelta));
            updateCameraPosition();
            e.preventDefault();
        });

        canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        updateCameraPosition();
        
        // Try to load actual SOG file
        try {
            const binaryString = atob(fileData);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            const blob = new Blob([bytes], { type: 'application/octet-stream' });
            const url = URL.createObjectURL(blob);
            
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
                camera.setPosition(5, 2, 5);
                camera.lookAt(0, 0, 0);
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
        
        const info = document.createElement('div');
        info.style.cssText = 'position: absolute; top: 10px; right: 10px; color: white; background: rgba(0,0,0,0.7); padding: 8px; border-radius: 4px; font-size: 12px; z-index: 1000;';
        //info.innerHTML = `PlayCanvas SOG Viewer<br>File: ${fileName}<br>Size: ${fileSize}<br>Status: SOG Loaded Successfully<br>Mouse: Rotate | Wheel: Zoom | Right-click: Pan`;
        container.style.position = 'relative';
        container.appendChild(info);
    };
    
    // Check for pending SOG data when tab becomes visible
    const checkSOGData = () => {
        if (window.sogData && !window.sogLoaded) {
            const container = document.getElementById('sog-container');
            if (container && container.offsetWidth > 0) {
                globalThis.createSOGViewer(window.sogData.fileData, window.sogData.fileName, window.sogData.fileSize);
                window.sogLoaded = true;
            }
        }
    };
    
    // Check periodically for SOG data
    setInterval(checkSOGData, 500);
}
"""

def create_interface():
    # Create the main Gradio interface
    with gr.Blocks(js=playcanvas_js, title="Open Source 3D Reconstruction Toolbox for Gaussian Splats on AWS", theme=gr.themes.Ocean(), css="""
        /* Add global tracking script */
        <script>
        // Global variable to track loaded models
        window.loadedModels = {};
        </script>
        
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
                            show_download_button=False,
                            show_label=False,
                            container=False,
                            height=40,
                            width=None,
                            show_fullscreen_button=False,
                            elem_classes=["theme-image-light"]
                        )
                        dark_logo = gr.Image(
                            "../../assets/images/PoweredByAWS_horiz_RGB_1c_White.png",
                            show_download_button=False,
                            show_label=False,
                            container=False,
                            height=40,
                            width=None,
                            show_fullscreen_button=False,
                            elem_classes=["theme-image-dark"]
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
            create_job_monitor_tab()
            create_s3_browser_tab()
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

    # Add favorites directory to allowed_paths
    favorites_dir = os.path.join(os.path.dirname(__file__), "favorites")
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False, allowed_paths=[favorites_dir])
