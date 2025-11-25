import boto3
import uuid
import json
import os

def refine_splat(selected_data, instance_type, use_spot_instance):
    """Resume training for a selected splat by creating a new job with RUN_SFM=false"""
    try:
        # Import shared_state for other configuration values
        from generate_splat_gradio import shared_state
        print(f"DEBUG: Received parameters - instance: {instance_type}, spot: {use_spot_instance}")
        
        if not selected_data:
            return "No file selected"
        
        job_id = selected_data[0]  # First column is job ID
        filename = selected_data[1]  # Second column is filename
        
        # Get job metadata from DynamoDB
        dynamodb = boto3.resource('dynamodb', region_name=shared_state.aws_region)
        table = dynamodb.Table(shared_state.ddb_table_name)
        
        response = table.get_item(Key={'uuid': job_id})
        if 'Item' not in response:
            return f"Error: Job {job_id} not found in database"
        
        selected_job = response['Item']
        #print(f"Selected job data: {selected_job}")
        
        # Extract model and max steps from selected job
        original_model = None
        original_max_steps = None
        
        if 'training' in selected_job and 'model' in selected_job['training']:
            original_model = selected_job['training']['model']
            original_max_steps = selected_job['training'].get('maxSteps')
        elif 'model' in selected_job:
            original_model = selected_job['model']
        elif 'MODEL' in selected_job:
            original_model = selected_job['MODEL']
        
        # Check for maxSteps in DynamoDB record
        if original_max_steps is None:
            original_max_steps = selected_job.get('maxSteps')
        
        # Generate new UUID for refinement job
        refine_uuid = uuid.uuid4()
        
        # Create refinement job config based on selected job
        print(f"DEBUG: Using parameters - instance: {instance_type}, spot: {use_spot_instance}")
        refine_config = {
            "uuid": str(refine_uuid),
            "instanceType": instance_type,
            "useSpotInstance": use_spot_instance,
            "logVerbosity": str(selected_job.get('logVerbosity', shared_state.log_verbosity)),
            "s3": {
                "bucketName": shared_state.s3_bucket,
                "inputPrefix": f"{str(selected_job.get('outputPrefix', shared_state.s3_output))}/{job_id}/output",
                "inputKey": "model.tar.gz",  # Use the model.tar.gz from selected job
                "outputPrefix": shared_state.s3_output
            },
            "videoProcessing": selected_job.get('videoProcessing', {
                "maxNumImages": str(shared_state.max_images),
                "videoStartTime": str(selected_job.get('videoProcessing', {}).get('videoStartTime', shared_state.video_start_time)),
                "videoStopTime" : str(selected_job.get('videoProcessing', {}).get('videoStopTime', shared_state.video_stop_time))
            }),
            "imageProcessing": selected_job.get('imageProcessing', {
                "filterBlurryImages": shared_state.filter_blurry == "true"
            }),
            "reconstruction": {
                "enable": False,  # Skip SFM for refinement
                "softwareName": str(selected_job.get('reconstruction', {}).get('softwareName', selected_job.get('sfm', {}).get('softwareName', shared_state.sfm))),
                "posePriors": selected_job.get('reconstruction', {}).get('posePriors', selected_job.get('sfm', {}).get('posePriors', {
                    "usePosePriorColmapModelFiles": shared_state.use_colmap_model == "true",
                    "usePosePriorTransformJson": {
                        "enable": shared_state.use_transform_json == "true",
                        "sourceCoordinateName": shared_state.source_coordinate,
                        "poseIsWorldToCam": shared_state.pose_world_to_cam == "true"
                    }
                })),
                "enableEnhancedFeatureExtraction": selected_job.get('reconstruction', {}).get('enableEnhancedFeatureExtraction', selected_job.get('sfm', {}).get('enableEnhancedFeatureExtraction', shared_state.enhanced_feature == "true")),
                "matchingMethod": str(selected_job.get('reconstruction', {}).get('matchingMethod', selected_job.get('sfm', {}).get('matchingMethod', shared_state.matching_method)))
            },
            "training": {
                "enable": True,  # Enable training for refinement
                "maxSteps": "20000", # This will be overwritten in container using contants set for max_steps for optimal training
                "model": str(original_model or shared_state.model),
                "enableMultiGpu": selected_job.get('training', {}).get('enableMultiGpu', False),

                **{k: v for k, v in selected_job.get('training', {}).items() if k not in ['enable', 'maxSteps', 'model', 'enableMultiGpu', 'refineSteps']}  # Preserve other parameters
            },
            "postProcessing": {
                "rotateSplat": selected_job.get('rotateSplat', shared_state.rotate_splat) == "True",
                "cropOutputBounds": selected_job.get('cropOutputBounds', selected_job.get('refineOutputBounds', shared_state.crop_output_bounds)) == "True",
                "cropMode": selected_job.get('cropMode', selected_job.get('refinementMode', shared_state.crop_mode)),
                "enableSpz": selected_job.get('enableSpz', shared_state.enable_spz) == "True",
                "enableSogs": selected_job.get('enableSogs', shared_state.enable_sogs) == "True",
                "enableUsdz": selected_job.get('enableUsdz', shared_state.enable_usdz) == "True"
            },
            "sphericalCamera": {
                "enable": selected_job.get('sphericalCamera') == 'True' if isinstance(selected_job.get('sphericalCamera'), str) else selected_job.get('sphericalCamera', {}).get('enable', shared_state.spherical_enable == "true"),
                "cubeFacesToRemove": shared_state.faces if isinstance(shared_state.faces, list) else []
            },
            "segmentation": selected_job.get('segmentation', {
                "backgroundRemoval": {
                    "enable": shared_state.remove_bg == "true",
                    "model": shared_state.bg_model,
                    "maskThreshold": str(shared_state.mask_threshold)
                },
                "objectRemoval": {
                    "enable": shared_state.remove_objects == "true",
                    "action": shared_state.object_removal_action,
                    "objects": str(shared_state.objects_to_remove)
                }
            })
        }
        
        # Upload refinement job JSON to workflow-input
        s3_client = boto3.client('s3')
        job_json_key = f"{shared_state.s3_input}/{refine_uuid}.json"
        job_json = json.dumps(refine_config, indent=4)
        
        s3_client.put_object(
            Bucket=shared_state.s3_bucket,
            Key=job_json_key,
            Body=job_json.encode('utf-8'),
            ContentType='application/json'
        )
        
        return f"Refinement job submitted successfully!\nSource Job: {job_id[:8]}...\nRefinement Job ID: {refine_uuid}\nThis will resume training from the selected job's model."
        
    except Exception as e:
        print(f"Error in refine_splat: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error submitting refinement job: {str(e)}"
