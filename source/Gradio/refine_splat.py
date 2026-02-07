import boto3
import uuid
import json
import os
from decimal import Decimal

def refine_splat(selected_data, instance_type, use_spot_instance, crop_bounds=None, crop_mode=None, enable_spz=None, enable_sog=None, enable_usdz=None, ply_coords=None, spz_coords=None, sog_coords=None, usdz_coords=None):
    """Resume training for a selected splat by creating a new job with RUN_SFM=false"""
    try:
        # Import shared_state for other configuration values
        from generate_splat_gradio import shared_state
        print(f"DEBUG: Received parameters - instance: {instance_type}, spot: {use_spot_instance}")
        
        # Use provided values or fall back to shared_state
        crop_bounds = crop_bounds if crop_bounds is not None else shared_state.crop_output_bounds
        crop_mode = crop_mode if crop_mode is not None else shared_state.crop_mode
        enable_spz = enable_spz if enable_spz is not None else shared_state.enable_spz
        enable_sog = enable_sog if enable_sog is not None else shared_state.enable_sog
        enable_usdz = enable_usdz if enable_usdz is not None else shared_state.enable_usdz
        ply_coords = ply_coords if ply_coords is not None else shared_state.ply_coords
        spz_coords = spz_coords if spz_coords is not None else shared_state.spz_coords
        sog_coords = sog_coords if sog_coords is not None else shared_state.sog_coords
        usdz_coords = usdz_coords if usdz_coords is not None else shared_state.usdz_coords
        
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
        
        # Get original filename from source job for tracking in UI
        # Priority: originalMediaFilename > s3.inputKey > filename > inputKey
        original_filename = None
        if 'originalMediaFilename' in selected_job:
            original_filename = selected_job['originalMediaFilename']
        elif 's3' in selected_job and 'inputKey' in selected_job['s3']:
            input_key = selected_job['s3']['inputKey']
            if input_key != 'model.tar.gz':
                original_filename = input_key.split('/')[-1]  # Get filename from path
        elif 'filename' in selected_job and selected_job['filename'] != 'model.tar.gz':
            original_filename = selected_job['filename']
        elif 'inputKey' in selected_job and selected_job['inputKey'] != 'model.tar.gz':
            original_filename = selected_job['inputKey'].split('/')[-1]
        
        # If still no filename, try to get from parent job if this is already a refined job
        if not original_filename or original_filename == 'model.tar.gz':
            s3_input = selected_job.get('s3', {}).get('inputPrefix', '')
            if '/output' in s3_input:
                parent_job_id = s3_input.split('/')[-2]
                try:
                    parent_response = table.get_item(Key={'uuid': parent_job_id})
                    if 'Item' in parent_response:
                        parent_job = parent_response['Item']
                        if 'originalMediaFilename' in parent_job:
                            original_filename = parent_job['originalMediaFilename']
                        elif 's3' in parent_job and 'inputKey' in parent_job['s3']:
                            original_filename = parent_job['s3']['inputKey'].split('/')[-1]
                except:
                    pass
        
        # Final fallback
        if not original_filename or original_filename == 'model.tar.gz':
            original_filename = f"refined_{job_id[:8]}.mp4"
        
        # Build refine config from original job, preserving nested structures
        refine_config = {
            "uuid": str(refine_uuid),
            "instanceType": instance_type,
            "useSpotInstance": use_spot_instance,
            "logVerbosity": str(selected_job.get('logVerbosity', shared_state.log_verbosity)),
            "originalMediaFilename": original_filename,
            "s3": {
                "bucketName": shared_state.s3_bucket,
                "inputPrefix": f"{str(selected_job.get('s3', {}).get('outputPrefix', shared_state.s3_output))}/{job_id}/output",
                "inputKey": "model.tar.gz",
                "outputPrefix": shared_state.s3_output
            },
            "videoProcessing": selected_job.get('videoProcessing', {
                "maxNumImages": str(selected_job.get('maxNumImages', shared_state.max_images)),
                "videoStartTime": selected_job.get('videoStartTime', shared_state.video_start_time),
                "videoStopTime": selected_job.get('videoStopTime', shared_state.video_stop_time),
                "filterBlurryImages": str(selected_job.get('filterBlurryImages', shared_state.filter_blurry)) == "True"
            }),
            "reconstruction": {
                "enable": False,
                "softwareName": str(selected_job.get('reconstruction', {}).get('softwareName', selected_job.get('reconSoftwareName', shared_state.sfm))),
                "posePriors": selected_job.get('reconstruction', {}).get('posePriors', {
                    "usePosePriorColmapModelFiles": str(selected_job.get('usePosePriorColmapModelFiles', shared_state.use_colmap_model)) == "True",
                    "usePosePriorTransformJson": {
                        "enable": str(selected_job.get('usePosePriorTransformJson', shared_state.use_transform_json)) == "True",
                        "sourceCoordinateName": str(selected_job.get('sourceCoordinateName', shared_state.source_coordinate)),
                        "poseIsWorldToCam": str(selected_job.get('poseIsWorldToCam', shared_state.pose_world_to_cam)) == "True"
                    }
                }),
                "enableEnhancedFeatureExtraction": str(selected_job.get('enableEnhancedFeatureExtraction', shared_state.enhanced_feature)) == "True",
                "matchingMethod": str(selected_job.get('matchingMethod', shared_state.matching_method))
            },
            "training": {
                "enable": True,
                "maxSteps": "20000",
                "model": str(original_model or shared_state.model),
                "preserveSceneScale": str(selected_job.get('preserveSceneScale', shared_state.preserve_scene_scale)) == "True"
            },
            "postProcessing": {
                "cropOutputBounds": crop_bounds == "true" if isinstance(crop_bounds, str) else crop_bounds,
                "cropMode": crop_mode,
                "enableSpz": enable_spz == "true" if isinstance(enable_spz, str) else enable_spz,
                "enableSog": enable_sog == "true" if isinstance(enable_sog, str) else enable_sog,
                "enableUsdz": enable_usdz == "true" if isinstance(enable_usdz, str) else enable_usdz,
                "plyCoords": ply_coords,
                "spzCoords": spz_coords,
                "sogCoords": sog_coords,
                "usdzCoords": usdz_coords
            },
            "sphericalCamera": {
                "enable": str(selected_job.get('sphericalCamera', shared_state.spherical_enable)) == "True",
                "cubeFacesToRemove": selected_job.get('sphericalCubeFacesToRemove', [])
            },
            "segmentation": selected_job.get('segmentation', {
                "backgroundRemoval": {
                    "enable": str(selected_job.get('removeBackground', shared_state.remove_bg)) == "True",
                    "model": str(selected_job.get('backgroundRemovalModel', shared_state.bg_model)),
                    "maskThreshold": str(selected_job.get('maskThreshold', shared_state.mask_threshold))
                },
                "objectRemoval": {
                    "enable": str(selected_job.get('removeObject', shared_state.remove_objects)) == "True",
                    "action": str(selected_job.get('objectRemovalAction', shared_state.object_removal_action)),
                    "objects": str(selected_job.get('objectRemovalObjects', shared_state.objects_to_remove))
                }
            })
        }
        
        print(f"DEBUG: Using parameters - instance: {instance_type}, spot: {use_spot_instance}")
        print(f"DEBUG: Original filename for tracking: {original_filename}")
        
        # Convert Decimal to float for JSON serialization
        def convert_decimals(obj):
            if isinstance(obj, list):
                return [convert_decimals(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: convert_decimals(v) for k, v in obj.items()}
            elif isinstance(obj, Decimal):
                return float(obj)
            return obj
        
        refine_config = convert_decimals(refine_config)
        
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
