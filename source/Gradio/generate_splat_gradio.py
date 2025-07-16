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

print(f"Gradio Version: {gr.__version__}")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class SharedState:
    def __init__(self):
        self.s3_bucket = "3dgs-bucket-bqmst2"
        self.s3_input = "workflow-input"
        self.s3_output = "workflow-output"
        self.media_input = "media-input"
        self.instance = "ml.g5.4xlarge"
        self.sfm = "glomap"
        self.model = "splatfacto"
        self.faces = "[]"
        self.optimize = "true"
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
        self.enable_multi_gpu = "false"
        self.spherical_enable = "false"
        self.remove_bg = "false"
        self.remove_human = "false"
        self.source_coordinate = "arkit"
        self.pose_world_to_cam = "true"
        self.log_verbosity = "info"
        self.mask_threshold = 0.6
        self.model_3d = None
        self.rotate_splat = "true"

# Create a singleton instance
shared_state = SharedState()

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
            print(f"DEBUG: Processing part: {part[:15]}...")  # Show first 15 chars for debugging
            
            if part.startswith('$Env:AWS_ACCESS_KEY_ID='):
                access_key = part.split('=', 1)[1].strip('"').strip("'")
                print("DEBUG: Found access key")
            elif part.startswith('$Env:AWS_SECRET_ACCESS_KEY='):
                secret_key = part.split('=', 1)[1].strip('"').strip("'")
                print("DEBUG: Found secret key")
            elif part.startswith('$Env:AWS_SESSION_TOKEN='):
                session_token = part.split('=', 1)[1].strip('"').strip("'")
                print("DEBUG: Found session token")
        
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

def refresh_s3_contents():
    """Refresh S3 contents and return sorted data for .ply and .spz files"""
    try:
        refresh_id = time.time()  # Generate unique ID for this refresh operation
        print(f"\n=== Refreshing S3 Contents (ID: {refresh_id}) ===")
        s3_client = boto3.client('s3')
        bucket_name = shared_state.s3_bucket
        output_prefix = shared_state.s3_output or "workflow-output"
        
        # List objects in the bucket with the specified prefix
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=output_prefix
        )
        
        files_data = []
        if 'Contents' in response:
            for item in response['Contents']:
                # Skip if it's a directory
                if item['Key'].endswith('/'):
                    continue
                
                # Check if file is .ply or .spz
                if not (item['Key'].lower().endswith('.ply') or item['Key'].lower().endswith('.spz')):
                    continue
                    
                # Get the job ID from the path
                path_parts = item['Key'].split('/')
                if len(path_parts) >= 3:  # Ensure we have enough parts
                    job_id = path_parts[-2]  # Second to last part
                    filename = path_parts[-1]  # Last part
                    
                    # Format the size
                    size_bytes = item['Size']
                    if size_bytes < 1024:
                        size_str = f"{size_bytes} B"
                    elif size_bytes < 1024 * 1024:
                        size_str = f"{size_bytes/1024:.1f} KB"
                    else:
                        size_str = f"{size_bytes/(1024*1024):.1f} MB"
                    
                    # Format the last modified date
                    last_modified = item['LastModified'].strftime("%Y-%m-%d %H:%M:%S")
                    
                    files_data.append([
                        job_id,
                        filename,
                        size_str,
                        last_modified
                    ])
        
        # Sort the data by Last Modified column (index 3) in descending order
        files_data.sort(key=lambda x: x[3], reverse=True)
        
        print(f"Found {len(files_data)} .ply and .spz files")
        for i, file_data in enumerate(files_data):
            print(f"  [{i}]: {file_data[0]} - {file_data[1]}")
        print(f"=== End Refresh (ID: {refresh_id}) ===\n")
        
        return files_data
        
    except Exception as e:
        print(f"Error refreshing S3 contents: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def preview_json(s3_bucket_name, s3_input_prefix, s3_output_prefix, video_file, 
                instance_type, sfm_software, training_model, cube_faces_remove, optimize_sequential, bg_removal_model,
                filter_blurry, max_images, sfm_enable, enhanced_feature, matching_method, use_colmap_model,
                use_transform_json, training_enable, max_steps, enable_multi_gpu, spherical_enable, remove_bg, remove_human,
                source_coordinate, pose_world_to_cam, log_verbosity, mask_threshold, rotate_splat):
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
        "logVerbosity": log_verbosity,
        "s3": {
            "bucketName": s3_bucket_name,
            "inputPrefix": s3_input_prefix,
            "inputKey": media_filename,
            "outputPrefix": s3_output_prefix
        },
        "videoProcessing": {
            "maxNumImages": str(max_images),
        },
        "imageProcessing": {
            "filterBlurryImages": filter_blurry == "true"
        },
        "sfm": {
            "enable": sfm_enable == "true",
            "softwareName": sfm_software,
            "enableEnhancedFeatureExtraction": enhanced_feature == "true",
            "matchingMethod": matching_method,
            "posePriors": {
                "usePosePriorColmapModelFiles": use_colmap_model == "true",
                "usePosePriorTransformJson": {
                    "enable": use_transform_json == "true",
                    "sourceCoordinateName": source_coordinate,
                    "poseIsWorldToCam": pose_world_to_cam == "true"
                }
            }
        },
        "training": {
            "enable": training_enable == "true",
            "maxSteps": str(max_steps),
            "model": training_model,
            "enableMultiGpu": enable_multi_gpu == "true",
            "rotateSplat": rotate_splat == "true"
        },
        "sphericalCamera": {
            "enable": spherical_enable == "true",
            "cubeFacesToRemove": cube_faces_remove,
            "optimizeSequentialFrameOrder": optimize_sequential == "true"
        },
        "segmentation": {
            "removeBackground": remove_bg == "true",
            "backgroundRemovalModel": bg_removal_model,
            "maskThreshold": str(mask_threshold),
            "removeHumanSubject": remove_human == "true"
        }
    }
    
    return json.dumps(file_contents, indent=2)

def generate_splat(s3_bucket_name, s3_input_prefix, s3_output_prefix, file_obj, 
                  instance_type, sfm_software, training_model, cube_faces_remove, 
                  optimize_sequential, bg_removal_model, filter_blurry, max_images, 
                  sfm_enable, enhanced_feature, matching_method, use_colmap_model,
                  use_transform_json, training_enable, max_steps, enable_multi_gpu, 
                  spherical_enable, remove_bg, remove_human, source_coordinate, 
                  pose_world_to_cam, log_verbosity, mask_threshold, media_input_prefix="media-input", rotate_splat="babylon"):
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
        optimize_sequential = getattr(optimize_sequential, 'value', optimize_sequential)
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
        enable_multi_gpu = getattr(enable_multi_gpu, 'value', enable_multi_gpu)
        spherical_enable = getattr(spherical_enable, 'value', spherical_enable)
        remove_bg = getattr(remove_bg, 'value', remove_bg)
        remove_human = getattr(remove_human, 'value', remove_human)
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
            "sfm": {
                "enable": sfm_enable == "true",
                "softwareName": sfm_software,
                "enableEnhancedFeatureExtraction": enhanced_feature == "true",
                "matchingMethod": matching_method,
                "posePriors": {
                    "usePosePriorColmapModelFiles": use_colmap_model == "true",
                    "usePosePriorTransformJson": {
                        "enable": use_transform_json == "true",
                        "sourceCoordinateName": source_coordinate,
                        "poseIsWorldToCam": pose_world_to_cam == "true"
                    }
                }
            },
            "training": {
                "enable": training_enable == "true",
                "maxSteps": str(max_steps),
                "model": training_model,
                "enableMultiGpu": enable_multi_gpu == "true",
                "rotateSplat": rotate_splat == "true"
            },
            "sphericalCamera": {
                "enable": spherical_enable == "true",
                "cubeFacesToRemove": cube_faces_remove if isinstance(cube_faces_remove, list) else [],
                "optimizeSequentialFrameOrder": optimize_sequential == "true"
            },
            "segmentation": {
                "removeBackground": remove_bg == "true",
                "backgroundRemovalModel": bg_removal_model,
                "maskThreshold": str(mask_threshold),
                "removeHumanSubject": remove_human == "true"
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
                            "logVerbosity": shared_state.log_verbosity,
                            "s3": {
                                "bucketName": bucket_name,
                                "inputPrefix": "media-input",
                                "inputKey": filename,
                                "outputPrefix": shared_state.s3_output
                            },
                            "videoProcessing": {
                                "maxNumImages": str(shared_state.max_images),
                            },
                            "imageProcessing": {
                                "filterBlurryImages": shared_state.filter_blurry == "true"
                            },
                            "sfm": {
                                "enable": shared_state.sfm_enable == "true",
                                "softwareName": shared_state.sfm,
                                "enableEnhancedFeatureExtraction": shared_state.enhanced_feature == "true",
                                "matchingMethod": shared_state.matching_method,
                                "posePriors": {
                                    "usePosePriorColmapModelFiles": shared_state.use_colmap_model == "true",
                                    "usePosePriorTransformJson": {
                                        "enable": shared_state.use_transform_json == "true",
                                        "sourceCoordinateName": shared_state.source_coordinate,
                                        "poseIsWorldToCam": shared_state.pose_world_to_cam == "true"
                                    }
                                }
                            },
                            "training": {
                                "enable": shared_state.training_enable == "true",
                                "maxSteps": str(shared_state.max_steps),
                                "model": shared_state.model,
                                "enableMultiGpu": shared_state.enable_multi_gpu == "true",
                                "rotateSplat": shared_state.rotate_splat == "true"
                            },
                            "sphericalCamera": {
                                "enable": shared_state.spherical_enable == "true",
                                "cubeFacesToRemove": shared_state.faces if isinstance(shared_state.faces, list) else [],
                                "optimizeSequentialFrameOrder": shared_state.optimize == "true"
                            },
                            "segmentation": {
                                "removeBackground": shared_state.remove_bg == "true",
                                "backgroundRemovalModel": shared_state.bg_model,
                                "maskThreshold": str(shared_state.mask_threshold),
                                "removeHumanSubject": shared_state.remove_human == "true"
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
                        "ml.g6e.4xlarge",
                        "ml.g6e.8xlarge"],
                        #"ml.g6e.12xlarge"],
                    value="ml.g5.4xlarge"
                )

                def update_shared_state(bucket, input_prefix, output_prefix, media_prefix, inst):
                    shared_state.s3_bucket = bucket
                    shared_state.s3_input = input_prefix
                    shared_state.s3_output = output_prefix
                    shared_state.media_input = media_prefix
                    shared_state.instance = inst
                    return "AWS configuration updated"

                # Update shared state when any value changes
                for component in [s3_bucket, s3_input, s3_output, media_input, instance]:
                    component.change(
                        fn=update_shared_state,
                        inputs=[s3_bucket, s3_input, s3_output, media_input, instance],
                        outputs=[gr.Textbox(label="Status", visible=False)]
                    )

def create_advanced_settings_tab():
    with gr.Tab("Advanced Settings"):
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
                maximum=1000
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
                remove_bg = gr.Radio(
                    label="Remove Background",
                    choices=["true", "false"],
                    value="false"
                )
                bg_model = gr.Dropdown(
                    label="Background Removal Model",
                    choices=["sam2","u2net", "u2net-human"],
                    value="u2net"
                )
                remove_human = gr.Radio(
                    label="Remove Human",
                    choices=["true", "false"],
                    value="false"
                )
                mask_threshold = gr.Slider(
                    label="Mask Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.6,
                    step=0.01
                )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### SFM")
                # SFM Settings
                sfm_enable = gr.Radio(
                    label="Enable SFM",
                    choices=["true", "false"],
                    value="true"
                )
                sfm = gr.Dropdown(
                    label="SFM Software",
                    choices=["colmap", "glomap"],
                    value="glomap"
                )
                enhanced_feature = gr.Radio(
                    label="Enhanced Feature Extraction",
                    choices=["true", "false"],
                    value="false"
                )
                matching_method = gr.Dropdown(
                    label="Matching Method",
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
                        "3dgrt"
                    ],
                    value="splatfacto"
                )
                #enable_multi_gpu = gr.Radio(
                #    label="Enable Multi-GPU",
                #    choices=["true", "false"],
                #    value="false"
                #)
                rotate_splat = gr.Radio(
                    label="Rotate Splat for Gradio Viewer",
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
                optimize = gr.Radio(
                    label="Optimize Sequential Frame Order",
                    choices=["true", "false"],
                    value="true"
                )

                def update_advanced_settings(*args):
                    # Update shared state with all advanced settings
                    (shared_state.sfm, shared_state.model, shared_state.faces, 
                     shared_state.optimize, shared_state.bg_model, shared_state.filter_blurry,
                     shared_state.max_images, shared_state.sfm_enable, 
                     shared_state.enhanced_feature, shared_state.matching_method,
                     shared_state.use_colmap_model, shared_state.use_transform_json,
                     shared_state.training_enable, shared_state.max_steps,
                     #shared_state.enable_multi_gpu,
                     shared_state.spherical_enable,
                     shared_state.remove_bg, shared_state.remove_human,
                     shared_state.source_coordinate, shared_state.pose_world_to_cam,
                     shared_state.log_verbosity, shared_state.mask_threshold,
                     shared_state.rotate_splat) = args
                    return "Advanced settings updated"

                # Get all advanced settings components after they're defined
                advanced_components = [
                    sfm, model, faces, optimize, bg_model, filter_blurry,
                    max_images, sfm_enable, enhanced_feature, matching_method,
                    use_colmap_model, use_transform_json, training_enable,
                    max_steps, #enable_multi_gpu,
                    spherical_enable, remove_bg,
                    remove_human, source_coordinate, pose_world_to_cam,
                    log_verbosity, mask_threshold, rotate_splat
                ]

                # Update shared state when any value changes
                for component in advanced_components:
                    component.change(
                        fn=update_advanced_settings,
                        inputs=advanced_components,
                        outputs=[gr.Textbox(label="Status", visible=False)]
                    )

def on_select(evt: gr.SelectData, data):
    """Handle row selection in the files table"""
    try:
        row_idx = evt.index[0]
        selected_row = data[row_idx]
        print(f"[DEBUG] Selected row data: {selected_row}")
        # Return all four required outputs
        return [
            selected_row,  # for selected_data State
            gr.update(interactive=True),  # for download_btn
            gr.update(interactive=True),  # for view_btn
            gr.update(interactive=True)   # for add_favorite_btn
        ]
    except Exception as e:
        print(f"[DEBUG] Error in selection handler: {str(e)}")
        # Return all four required outputs even in case of error
        return [
            None,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False)
        ]

def handle_view(selected_row):
    """Handle view button click"""
    try:
        if not selected_row:
            return gr.update(value=None), "No file selected"
        
        bucket_name = shared_state.s3_bucket
        output_prefix = shared_state.s3_output or "workflow-output"
        
        job_id = selected_row[0]
        filename = selected_row[1]
        file_key = f"{output_prefix}/{job_id}/{filename}"
        
        # Get file size information
        file_size_mb = None
        try:
            s3_client = boto3.client('s3')
            response = s3_client.head_object(Bucket=bucket_name, Key=file_key)
            file_size = response['ContentLength']
            file_size_mb = file_size / (1024 * 1024)
            size_info = f" ({file_size_mb:.1f} MB)"
        except Exception as e:
            print(f"Error getting file size: {str(e)}")
            size_info = ""
        
        # Note: Removed artificial delay as it's not necessary for functionality
        
        # Generate a presigned URL for the file
        presigned_url = generate_presigned_url(bucket_name, file_key)
        
        if not presigned_url:
            return gr.update(value=None), "Error generating URL"
        
        # Return the URL for the Model3D component
        return gr.update(value=presigned_url), f"Loading {filename}{size_info}..."
        
    except Exception as e:
        error_msg = f"Error viewing file: {str(e)}"
        print(f"[DEBUG] Error in handle_view: {error_msg}")
        import traceback
        traceback.print_exc()
        return gr.update(value=None), error_msg

def add_to_favorites(selected_data):
    """Add currently selected item to favorites"""
    try:
        if not selected_data:
            return "No item selected"
        
        print(f"Debug - selected_data: {selected_data}")  # Debug print
        
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
            if not os.path.exists(os.path.join(favorites_dir, filename)):
                return f"Error: File {filename} not found in favorites directory"
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

def create_s3_browser_tab():
    with gr.Tab("3D Viewer & Library"):
        # Create viewer and status first (but don't render yet)
        viewer = gr.Model3D(
            label="3D Viewer",
            clear_color=[0.2, 0.2, 0.2, 1.0],
            height=900,
            interactive=True,
            render=False,
            #camera_position = (0, 0, 0) 
        )

        viewer_status = gr.Textbox(
            label="Viewer Status",
            interactive=False,
            show_label=True,
            value="",
            render=False
        )
        
        # Store the current model URL
        current_model = gr.State(None)

        # Create a container for favorites that can be updated
        favorites_container = gr.Column()
        
        # Function to update favorites UI
        def update_favorites_ui():
            favorites = load_favorites()
            with gr.Column() as new_container:
                gr.Markdown("### Favorites")
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
                                                        
                                # Create a click handler for this favorite
                                favorite_path = favorite['path']
                                favorite_name = favorite['filename']
                                
                                favorite_btn.click(
                                    fn=lambda p=favorite_path, n=favorite_name: (
                                        p,
                                        f"Loaded local file: {n}"
                                    ),
                                    inputs=[],
                                    outputs=[viewer, viewer_status]
                                )
            return new_container

        # Initial render of favorites
        with favorites_container:
            update_favorites_ui()

        # Now render the viewer
        viewer.render()
        viewer_status.render()

        def on_select(evt: gr.SelectData):
            try:
                # Create a timestamp to trace this specific selection event
                selection_time = time.time()
                print(f"\n=== DEBUG TABLE SELECTION {selection_time} ===")
                
                # Get the current data from the DataFrame
                current_data = files_box.value
                
                # Debug print to see what's in the table
                print(f"Type of files_box.value: {type(current_data)}")
                print(f"Event index: {evt.index}")
                
                # Always convert data to a consistent format
                if isinstance(current_data, dict) and 'data' in current_data:
                    current_data = current_data['data']
                    print(f"Extracted data from dictionary, new type: {type(current_data)}")
                
                # Convert to list if it's a numpy array or other sequence type
                if not isinstance(current_data, list) and hasattr(current_data, '__iter__'):
                    current_data = list(current_data)
                    print(f"Converted data to list")
                
                row_index = evt.index[0]
                print(f"Row index: {row_index}, Data length: {len(current_data) if current_data else 0}")
                
                if current_data and row_index < len(current_data):
                    selected_row = current_data[row_index]
                    print(f"Selected row data: {selected_row}")
                    print(f"Type of selected row: {type(selected_row)}")
                    
                    # Force selected_row to be a list if it's not already
                    if not isinstance(selected_row, list):
                        print(f"Converting selected_row to list")
                        selected_row = list(selected_row)
                        print(f"After conversion: {selected_row}")
                    
                    print(f"Enabling buttons for selection at {selection_time}")
                    return (
                        selected_row,
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True)
                    )
                else:
                    print(f"Invalid row index {row_index} or empty data")
                    if current_data:
                        print(f"Data length: {len(current_data)}")
                        print(f"First few items: {current_data[:3] if len(current_data) > 3 else current_data}")
                    print(f"Type of current_data: {type(current_data)}")
                    return (
                        None,
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                        gr.update(interactive=False)
                    )
                
            except Exception as e:
                print("\n=== Error Debug Information ===")
                print(f"Error in selection handler: {str(e)}")
                import traceback
                traceback.print_exc()
                print("=== End Error Information ===\n")
                return (
                    None,
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False)
                )
                
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
                    view_btn = gr.Button(
                        "View Selected", 
                        interactive=False,
                        size="sm"
                    )
                    add_favorite_btn = gr.Button(
                        "Add to Favorites", 
                        interactive=False,
                        size="sm"
                    )
                files_box = gr.Dataframe(
                    headers=["Job ID", "Filename", "Size", "Last Modified"],
                    interactive=False,
                    value=[],
                    visible=True,
                    elem_id="files_table"
                )

                selected_data = gr.State(None)

        download_iframe = gr.HTML(visible=True)
        
        # Update the select event handler
        files_box.select(
            fn=on_select,
            inputs=[],
            outputs=[
                selected_data,
                download_btn,
                view_btn,
                add_favorite_btn
            ]
        )

def refresh_and_update():
    """Refresh S3 contents and update the DataFrame"""
    try:
        files = refresh_s3_contents()
        
        if isinstance(files, dict) and 'data' in files:
            files = files['data']
        
        # Ensure we always return a list, even if empty
        if files is None:
            files = []
            
        # Reset the selection state when refreshing data
        print("Refreshed data, resetting selection state")
        return files, None, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
    except Exception as e:
        print(f"Error in refresh_and_update: {str(e)}")
        return [], None, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

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

def create_s3_browser_tab():
    with gr.Tab("3D Viewer & Library"):
        # Create viewer and status first (but don't render yet)
        viewer = gr.Model3D(
            label="3D Viewer",
            clear_color=[0.2, 0.2, 0.2, 1.0],
            height=900,
            interactive=True,
            render=False,
            #camera_position = (0, 0, 0) 
        )

        viewer_status = gr.Textbox(
            label="Viewer Status",
            interactive=False,
            show_label=True,
            value="",
            render=False
        )
        
        # Store the current model URL
        current_model = gr.State(None)

        # Create a container for favorites that can be updated
        favorites_container = gr.Column()
        
        # Function to update favorites UI
        def update_favorites_ui():
            favorites = load_favorites()
            with gr.Column() as new_container:
                gr.Markdown("### Favorites")
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
                                                        
                                # Create a click handler for this favorite
                                favorite_path = favorite['path']
                                favorite_name = favorite['filename']
                                
                                favorite_btn.click(
                                    fn=lambda p=favorite_path, n=favorite_name: (
                                        p,
                                        f"Loaded local file: {n}"
                                    ),
                                    inputs=[],
                                    outputs=[viewer, viewer_status]
                                )
            return new_container

        # Initial render of favorites
        with favorites_container:
            update_favorites_ui()

        # Now render the viewer
        viewer.render()
        viewer_status.render()


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
                    view_btn = gr.Button(
                        "View Selected", 
                        interactive=False,
                        size="sm"
                    )
                    add_favorite_btn = gr.Button(
                        "Add to Favorites", 
                        interactive=False,
                        size="sm"
                    )
                files_box = gr.Dataframe(
                    headers=["Job ID", "Filename", "Size", "Last Modified"],
                    interactive=False,
                    value=[],
                    visible=True,
                    elem_id="files_table"
                )

                selected_data = gr.State(None)

        download_iframe = gr.HTML(visible=True)
        
        # Replace the select event handler with a more direct approach
        def direct_select(evt: gr.SelectData, data):
            """Direct selection handler that uses the event data to get the selected row"""
            try:
                # Get row index from the event
                row_idx = evt.index[0]
                
                # Debug info
                print(f"\n=== DIRECT SELECT ===")
                print(f"Event index: {evt.index}")
                print(f"Data type: {type(data)}")
                print(f"Data length: {len(data) if hasattr(data, '__len__') else 'unknown'}")
                
                # Handle different data types
                if hasattr(data, 'empty') and not data.empty and row_idx < len(data):
                    # DataFrame case
                    selected_row = data.iloc[row_idx].tolist()
                    print(f"Selected row: {selected_row}")
                    return selected_row, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
                elif hasattr(data, '__len__') and len(data) > 0 and row_idx < len(data):
                    # List/array case
                    selected_row = data[row_idx]
                    print(f"Selected row: {selected_row}")
                    return selected_row, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
                else:
                    print(f"Invalid selection: row_idx={row_idx}, data_len={len(data) if hasattr(data, '__len__') else 0}")
                    return None, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
                    
            except Exception as e:
                print(f"Error in direct_select: {str(e)}")
                import traceback
                traceback.print_exc()
                return None, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
        
        # Connect the direct selection handler
        files_box.select(
            fn=direct_select,
            inputs=[files_box],  # Pass the current data directly
            outputs=[
                selected_data,
                download_btn,
                view_btn,
                add_favorite_btn
            ]
        )

        # Add a callback to log when the dataframe value changes
        files_box.change(
            fn=lambda data: print(f"\n=== DATA UPDATED ===\nNew data type: {type(data)}\nData length: {len(data) if hasattr(data, '__len__') else 'unknown'}\nFirst 2 items: {data[:2] if hasattr(data, '__getitem__') else 'N/A'}\n"),
            inputs=[files_box],
            outputs=[]
        )
        
        # Connect refresh button with a direct callback that ensures proper data setting
        def refresh_button_handler():
            # Get fresh data
            refresh_id = f"refresh_{time.time()}"
            print(f"\n=== REFRESH TRIGGERED (ID: {refresh_id}) ===")
            
            try:
                # Get fresh data from S3
                files_data = refresh_s3_contents()
                
                # Log the data we're about to set
                print(f"Data to set - Type: {type(files_data)}, Length: {len(files_data)}")
                for i, item in enumerate(files_data[:5]):  # Log first 5 items
                    print(f"Item {i}: {item}")
                
                # Make sure files_data is a list
                if not isinstance(files_data, list):
                    if hasattr(files_data, '__iter__'):
                        files_data = list(files_data)
                    else:
                        files_data = []
                
                # Sort the data to ensure consistent ordering
                # Sort by Last Modified column (index 3) in descending order
                files_data.sort(key=lambda x: x[3] if len(x) > 3 else "", reverse=True)
                
                # Force a clean list of lists structure
                clean_data = []
                for row in files_data:
                    if isinstance(row, list) and len(row) >= 4:
                        clean_data.append(row)
                
                print(f"Refresh complete [{refresh_id}], returning {len(clean_data)} items")
                
                # Return data directly for the dataframe
                return (
                    clean_data, 
                    None, 
                    gr.update(interactive=False), 
                    gr.update(interactive=False), 
                    gr.update(interactive=False)
                )
                
            except Exception as e:
                print(f"Error in refresh_button_handler: {str(e)}")
                import traceback
                traceback.print_exc()
                return (
                    [], 
                    None, 
                    gr.update(interactive=False), 
                    gr.update(interactive=False), 
                    gr.update(interactive=False)
                )
            
        refresh_button.click(
            fn=refresh_button_handler,
            inputs=[],
            outputs=[
                files_box, 
                selected_data, 
                download_btn, 
                view_btn, 
                add_favorite_btn
            ]
        )

        # SuperSplat link will be added in the UI layout
        
        # Connect other buttons
        download_btn.click(
            fn=handle_download,
            inputs=[selected_data],
            outputs=[download_iframe]
        )
        
        view_btn.click(
            fn=handle_view,
            inputs=[selected_data],
            outputs=[viewer, viewer_status],
            show_progress="full"  # Show full progress animation
        )
        
        # Update this to use the new function that updates the UI
        add_favorite_btn.click(
            fn=add_to_favorites_and_reload,
            inputs=[selected_data],
            outputs=[gr.HTML(visible=True)]  # Changed to HTML output
        )
        
        # Replace button with centered HTML link
        supersplat_link = gr.HTML(
            '<div style="text-align:center; margin:10px 0;"><a href="https://superspl.at/editor" target="_blank" style="display:inline-block; background:#f97316; color:white; padding:8px 16px; text-decoration:none; border-radius:6px; font-size:14px; font-weight:500;">🚀 Open SuperSplat Editor</a></div>'
        )

        # Initial load of S3 contents
        initial_files = refresh_s3_contents()
        if isinstance(initial_files, dict) and 'data' in initial_files:
            initial_files = initial_files['data']
        files_box.value = initial_files

        return files_box

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
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': key,
                'ResponseContentType': 'application/octet-stream'  # Force binary content type
            },
            ExpiresIn=expiration
        )
        print(f"[DEBUG] Generated presigned URL with content type for {key}")
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

def create_debug_tab():
    # Debug Tab
    with gr.Tab("Debug"):
        with gr.Row():
            with gr.Column():
                preview_btn = gr.Button("Preview JSON", elem_classes="orange-button")
                debug_output = gr.Textbox(label="JSON Preview", lines=20)

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
                        shared_state.instance or "ml.g5.4xlarge",
                        shared_state.sfm,
                        shared_state.model,
                        faces,
                        shared_state.optimize,
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
                        shared_state.enable_multi_gpu,
                        shared_state.spherical_enable,
                        shared_state.remove_bg,
                        shared_state.remove_human,
                        shared_state.source_coordinate,
                        shared_state.pose_world_to_cam,
                        shared_state.log_verbosity,
                        shared_state.mask_threshold,
                        shared_state.rotate_splat
                    )

            preview_btn.click(
                fn=preview_json_with_shared_state,
                inputs=None,
                outputs=[debug_output]
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
            if file.endswith(('.ply', '.spz', '.glb')):
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

def create_interface():
    # Create the main Gradio interface
    with gr.Blocks(title="Open Source 3D Reconstruction Toolbox for Gaussian Splats on AWS", theme=gr.themes.Ocean(), css="""
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
        # These images are not needed anymore as we're using the theme-based images in the layout
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Open Source 3D Reconstruction Toolbox for Gaussian Splats on AWS")
                gr.Markdown("Generate and upload a metadata file and media (.mov, .mp4, .zip) for gaussian splat creation.</br>Browse and render generated splats in a local 3D web viewer.")
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=1):
                        pass
                    with gr.Column(scale=0, elem_classes=["logo-container"]):
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
            create_s3_browser_tab()
            
            # Less frequently used tabs with visual distinction
            with gr.Tab("⚙️ AWS Credentials"):
                create_credentials_tab()
            
            with gr.Tab("🔧 Debug"):
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
