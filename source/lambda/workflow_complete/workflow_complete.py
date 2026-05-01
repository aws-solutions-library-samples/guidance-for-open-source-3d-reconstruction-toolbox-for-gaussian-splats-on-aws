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

""" Lambda function invoked by the Step Functions state machine upon job completion (success or failure).
Updates the DynamoDB job record with end timestamp and status, retrieves CloudWatch logs to identify
errors, queries the Batch API for compute environment diagnostics when a job never started, and
sends an SNS notification to the user with job timing, output file locations, or error details. """

import os
import re
import json
import boto3
from datetime import datetime
from dateutil import parser

from botocore.exceptions import ClientError

_UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)

def _validate_uuid(value: str) -> str:
    """Validate that value is a well-formed UUID to prevent injection."""
    if not _UUID_RE.match(str(value)):
        raise ValueError(f"Invalid UUID format: {value!r}")
    return str(value)

def _sanitize_text(value: str) -> str:
    """Strip newline/carriage-return characters to prevent log/message injection."""
    return re.sub(r'[\r\n]', ' ', str(value))

def send_sns_notification(training_job_name, message, is_error=False):
    """Send an SNS notification about the training job status."""
    sns_client = boto3.client('sns')
    sns_topic_arn = os.environ.get('SNS_TOPIC_ARN')
    
    if not sns_topic_arn:
        print("SNS_TOPIC_ARN environment variable not set")
        return False
    
    try:
        # Get job details for more context
        sagemaker_client = boto3.client('sagemaker')
        job_details = sagemaker_client.describe_training_job(
            TrainingJobName=training_job_name
        )
        
        # Extract useful information
        status = job_details.get('TrainingJobStatus', 'Unknown')
        start_time = job_details.get('TrainingStartTime', 'Unknown')
        end_time = job_details.get('TrainingEndTime', 'Unknown')
        
        # Format times if they exist
        if isinstance(start_time, datetime):
            start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(end_time, datetime):
            end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create subject based on status
        if is_error:
            subject = f"❌ 3D Gaussian Splat Job Failed: {training_job_name}"
        else:
            subject = f"✅ 3D Gaussian Splat Job Completed: {training_job_name}"
        
        # Create message body
        body = f"""
Job Name: {training_job_name}
Status: {status}
Start Time: {start_time}
End Time: {end_time}

{message}
"""
        
        # Send the notification
        response = sns_client.publish(
            TopicArn=sns_topic_arn,
            Message=body,
            Subject=subject
        )
        
        print(f"SNS notification sent: {response['MessageId']}")
        return True
        
    except Exception as e:
        print(f"Error sending SNS notification: {str(e)}")
        return False

def check_for_timeout(training_job_name):
    """Check if the SageMaker training job timed out."""
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        response = sagemaker_client.describe_training_job(
            TrainingJobName=training_job_name
        )
        
        # Check if the job failed due to timeout
        if response['TrainingJobStatus'] == 'Failed':
            failure_reason = response.get('FailureReason', '')
            if 'timeout' in failure_reason.lower() or 'timed out' in failure_reason.lower():
                return True, f"Training job timed out: {failure_reason}"
            
        # Check if the job was stopped and exceeded max runtime
        if response['TrainingJobStatus'] == 'Stopped':
            # Check if the job was running for close to the max runtime
            start_time = response.get('TrainingStartTime')
            end_time = response.get('TrainingEndTime')
            
            if start_time and end_time:
                # Calculate duration in seconds
                duration = (end_time - start_time).total_seconds()
                max_runtime = response.get('StoppingCondition', {}).get('MaxRuntimeInSeconds', 0)
                
                # If duration is within 5 minutes of max runtime, likely a timeout
                if max_runtime > 0 and duration >= (max_runtime - 300):
                    return True, f"Training job likely timed out after running for {duration} seconds (max: {max_runtime})"
        
        return False, None
        
    except Exception as e:
        print(f"Error checking for timeout: {str(e)}")
        return False, None

def is_cuda_oom_failure(message):
    """Check if the message indicates a CUDA Out of Memory error or CUDA assertion failure."""
    oom_patterns = [
        'CUDA out of memory',
        'OutOfMemoryError',
        'torch.OutOfMemoryError',
        'CUDA error: device-side assert triggered',
        'IndexKernel.cu',
        'index out of bounds',
        'device-side assertions',
        'gsplat/strategy/ops.py',
        'torch.where(mask)[0]',
        'RuntimeError: CUDA error',
        '3dgrut_wrapper.py' and 'failed with return code -11',
        'Command \'/usr/local/bin/python 3dgrut_wrapper.py\'' and 'failed with return code -11'
    ]
    
    if any(pattern in message for pattern in oom_patterns):
        print(f"CUDA failure detected in message: {message[:100]}...")  # Debug log
        return True
    return False

def is_sfm_failure(message):
    """Check if the message indicates an SFM reconstruction failure."""
    # Check for various SFM failure patterns - exclude gsplat errors which are training issues
    sfm_patterns = [
        'glomap::ViewGraph::KeepLargestConnectedComponents',
        'Command \'glomap mapper\'' and 'failed with return code -11'
    ]
    
    if any(pattern in message for pattern in sfm_patterns):
        print(f"SFM failure detected in message: {message[:100]}...")  # Debug log
        return True
    return False

def get_cloudwatch_logs(training_job_name, is_batch_job=False, log_stream_name=None):
    logs_client = boto3.client('logs')
    
    try:
        # FIRST: Check if the job actually succeeded before scanning for errors
        if not is_batch_job:
            sagemaker_client = boto3.client('sagemaker')
            training_job = sagemaker_client.describe_training_job(
                TrainingJobName=training_job_name
            )
            
            # If job succeeded, return success immediately without scanning logs
            if training_job['TrainingJobStatus'] == 'Completed':
                return {
                    'status': 'SUCCESS',
                    'message': 'Training completed successfully'
                }
            
            # If job failed, continue to scan logs for error details
            if training_job['TrainingJobStatus'] != 'Failed':
                return {
                    'status': 'SUCCESS',
                    'message': f"Job status: {training_job['TrainingJobStatus']}"
                }
        
        # Only scan logs if job actually failed
        if is_batch_job and log_stream_name:
            # For Batch jobs, use the provided log stream name
            log_group_name = '/aws/batch/job'
            print(f"Looking for Batch logs in group: {log_group_name}, stream: {log_stream_name}")
            
            # Try to get the exact log stream first
            try:
                response = logs_client.describe_log_streams(
                    logGroupName=log_group_name,
                    logStreamNamePrefix=log_stream_name
                )
                print(f"Found {len(response.get('logStreams', []))} log streams")
            except Exception as e:
                print(f"Error accessing Batch log group {log_group_name}: {e}")
                return {
                    'status': 'ERROR',
                    'message': f"Unable to access Batch logs: {str(e)}"
                }
        else:
            # For SageMaker jobs, use the original logic
            response = logs_client.describe_log_streams(
                logGroupName='/aws/sagemaker/TrainingJobs',
                logStreamNamePrefix=training_job_name
            )

        error_messages = []
        found_error = False
        
        # Keywords that indicate an error
        error_indicators = [
            'ERROR',
            'Error',
            'error',
            'Exception',
            'exception',
            'Traceback',
            'terminate called',
            'failed',
            'Failed',
            'OutOfMemoryError',
            'CUDA out of memory'
        ]
        
            # Messages to ignore (false positives)
        ignore_messages = [
            'TensorFloat32 tensor cores',
            'libio_e57.so',
            'Linear solver failure',
            'CHOLMOD warning',
            'invalid',
            'socket.cpp',
            'Cannot assign requested address',
            'client socket has failed',
            'Downloading:',
            'download.pytorch.org',
            '/root/.cache/torch/hub/checkpoints',
            'UserWarning:',
            'Exception ignored in:',
            '_MultiProcessingDataLoaderIter.__del__',
            'DataLoader worker',
            'is killed by signal',
            'torch/utils/data/dataloader.py',
            '_shutdown_workers',
            'multiprocessing/process.py',
            'multiprocessing/popen_fork.py',
            'multiprocessing/connection.py',
            'selectors.py',
            '_utils/signal_handling.py',
            'OOM errors or segfault',
            'UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.',
            'PERFORMANCE WARNING:',
            'Pairs read done',
            'invalid / total number',
            'are invalid',
            'Filtered',
            'track_filter.cc',
            'colmap_converter.cc',
            'global_mapper.cc',
            'view_graph_manipulation.cc',
            'view_graph_calibration.cc',
            'relpose_filter.cc',
            'Feature matching',
            'Creating SIFT GPU feature matcher',
            'Generating sequential image pairs',
            'Generating image pairs with vocabulary tree',
            'Indexing image',
            'pairing.cc',
            'sift.cc',
            'misc.cc',
            'Exception ignored in atexit callback',
            'torch/multiprocessing/spawn.py',
            'ProcessRaisedException',
            'CUDA kernel errors might be asynchronously reported',
            'For debugging consider passing CUDA_LAUNCH_BLOCKING=1',
            'Distributed worker:',
            'Warning: image_path not found for reconstruction',
            'terminated with the following error',
            'Skipping the post-processing step due to the error above',
            'OK to ignore the error above',
            'Command \'ns-train splatfacto-mcmc',
            'torch.multinomial',
            'TORCH_USE_CUDA_DSA',
            'device-side assertions',
            'glomap::ViewGraph::KeepLargestConnectedComponents',
            'Failed to extract frame',
            'mean_reprojection_error',
            'vkCreateInstance failed with VK_ERROR_INCOMPATIBLE_DRIVER',
            'Warning: vkCreateInstance',
            'XDG_RUNTIME_DIR not set in the environment',
            'maxDynamicUniformBuffersPerPipelineLayout',
            'maxDynamicStorageBuffersPerPipelineLayout',
            'LOG_VERBOSITY:',
            '(type: str)'
        ]
        
        def should_ignore_message(message):
            if is_sfm_failure(message):
                print("Should not ignore SFM failure")  # Debug log
                return False
            # Check if message contains any of the ignore patterns
            if any(ignore_msg in message for ignore_msg in ignore_messages):
                return True
            
            # Add specific PyTorch multiprocessing patterns to ignore
            pytorch_ignore_patterns = [
                'ProcessRaisedException',
                'multiprocessing/spawn.py',
                'CUDA kernel errors might be asynchronously reported',
                'CUDA_LAUNCH_BLOCKING=1',
                'torch.multiprocessing',
                'process_context.join()',
                'terminated with the following error',
                '_wrap',
                'Distributed worker:',
                'Warning: image_path not found for reconstruction',
                'glomap::ViewGraph::KeepLargestConnectedComponents'
            ]
            
            if any(pattern in message for pattern in pytorch_ignore_patterns):
                return True
    
            # Specific check for DataLoader cleanup stack traces
            if ('DataLoader worker' in message and 
                ('killed by signal' in message or 
                 '_MultiProcessingDataLoaderIter' in message)):
                return True
            
            # Check for normal training progress indicators
            if any(x in message for x in ['loss=', 'it/s', '|']):
                return True
        
            return False
        
        log_group_name = '/aws/batch/job' if is_batch_job else '/aws/sagemaker/TrainingJobs'
        
        for stream in response.get('logStreams', []):
            events = logs_client.get_log_events(
                logGroupName=log_group_name,
                logStreamName=stream['logStreamName'],
                startFromHead=False
            )
            
            for event in events['events']:
                message = event['message']
                
                # Check for CUDA OOM failure first
                if is_cuda_oom_failure(message):
                    # Determine if it's a memory issue or assertion failure
                    if 'CUDA out of memory' in message or 'OutOfMemoryError' in message:
                        cuda_error_message = """
            ❌ CUDA Out of Memory Error

            The training process ran out of GPU memory during execution. This typically occurs when:

            1. Dataset Issues:
            - Too many high-resolution images
            - Images are too large for the selected GPU
            - Video resolution is too high

            2. Model Configuration:
            - Training steps set too high for available memory
            - Model complexity exceeds GPU capacity
            - Batch size too large

            3. Instance Type:
            - Selected instance has insufficient GPU memory
            - Multiple processes competing for GPU memory

            Recommendations:
            1. Reduce Dataset Size:
            - Limit max images to 200-300
            - Use lower resolution input (1080p or less)
            - Consider cropping or downscaling images

            2. Adjust Training Parameters:
            - Reduce max training steps
            - Use a simpler model (splatfacto instead of 3dgrt)
            - Enable background removal to reduce scene complexity

            3. Upgrade Instance:
            - Use ml.g5.8xlarge (32GB GPU memory)
            - Use ml.g6.8xlarge for newer GPU architecture
            - Consider ml.g6e.4xlarge for cost-effective option

            Technical Details:
            - Error: CUDA OutOfMemoryError during training
            - GPU Memory: Insufficient for current workload
            - Status: Process terminated due to memory exhaustion"""
                    else:
                        cuda_error_message = """
            ❌ CUDA Memory/Indexing Error

            The training process encountered a CUDA error during model execution. This typically occurs when:

            1. GPU Memory Issues:
            - Insufficient GPU memory for the model complexity
            - Memory fragmentation during training
            - Competing processes using GPU memory

            2. Model/Data Compatibility:
            - Dataset too complex for available GPU memory
            - Model requires more memory than available
            - Indexing errors due to memory constraints

            3. Instance Limitations:
            - Current GPU instance insufficient for training
            - Memory allocation failures during optimization

            Recommendations:
            1. Reduce Model Complexity:
            - Use simpler model (splatfacto instead of 3dgrt)
            - Reduce max training steps (try 15000-20000)
            - Enable background removal to simplify scene

            2. Optimize Dataset:
            - Limit max images to 200-250
            - Use lower resolution input (1080p or less)
            - Remove complex/cluttered scenes

            3. Upgrade Instance:
            - Use ml.g5.8xlarge (32GB GPU memory)
            - Use ml.g6.8xlarge for newer architecture
            - Consider ml.g6e.4xlarge for better memory handling

            Technical Details:
            - Error: CUDA device-side assertion failure
            - Component: Gaussian Splat training optimization
            - Status: Process terminated due to GPU memory/indexing error"""
                    error_messages.append(cuda_error_message)
                    found_error = True
                    # Continue collecting next 20 lines for context
                    continue
                
                # Check for SFM failure
                if is_sfm_failure(message):
                    sfm_error_message = """
            ❌ Structure from Motion (SFM) Reconstruction Failed

            The camera pose estimation process could not converge. This typically occurs when:

            1. Image Quality Issues:
            - Insufficient overlap between consecutive frames
            - Motion blur in images
            - Poor lighting conditions
            - Low image resolution

            2. Scene Characteristics:
            - Not enough distinctive features in the scene
            - Highly reflective or transparent surfaces
            - Uniform/textureless areas
            - Dynamic objects or movement in scene

            3. Camera Motion:
            - Too rapid camera movement
            - Large gaps in viewpoints
            - Irregular camera paths

            Recommendations:
            1. Image Capture:
            - Ensure 60-80% overlap between consecutive frames
            - Move camera slowly and steadily
            - Maintain consistent lighting
            - Capture higher resolution images
            - Avoid motion blur

            2. Scene Setup:
            - Add more distinctive features to the scene
            - Ensure adequate and consistent lighting
            - Avoid highly reflective surfaces
            - Remove moving objects if possible

            3. Processing:
            - Try reducing the number of input images
            - Consider using a different subset of images
            - Verify image quality before processing

            Technical Details:
            - Error: SFM reconstruction failure during Gaussian optimization
            - Component: torch.multinomial sampling in gsplat strategy
            - Status: Process terminated during training"""
                    error_messages.append(sfm_error_message)
                    found_error = True
                    # Continue collecting next 20 lines for context
                    continue

                # Skip messages that should be ignored
                #if any(ignore_msg in message for ignore_msg in ignore_messages):
                #    continue
                # Only proceed with normal error checking if not an SFM failure
                if not found_error:
                    # Skip messages that should be ignored
                    if should_ignore_message(message):
                        continue

  
                    # Check if any error indicators are present
                    if any(indicator in message for indicator in error_indicators):
                        # Double check it's not a false positive we want to ignore
                        if not should_ignore_message(message):
                            # Additional check for PyTorch-specific log patterns
                            if not (message.startswith('I') or 
                                message.startswith('W') or 
                                '[W' in message or 
                                'Exception ignored in:' in message):
                                error_messages.append(message.strip())
                                found_error = True
                                continue
                
                # If we found an error, collect the next lines for context (up to 50 total lines)
                if found_error and len(error_messages) < 50:
                    error_messages.append(message.strip())
            
            # If we found enough error context, stop processing more log streams
            if found_error and len(error_messages) >= 50:
                break

        if found_error:
            return {
                'status': 'ERROR',
                'message': '\n'.join(error_messages)
            }
        
        # Double check training job status
        sagemaker_client = boto3.client('sagemaker')
        training_job = sagemaker_client.describe_training_job(
            TrainingJobName=training_job_name
        )
        
        if training_job['TrainingJobStatus'] == 'Failed':
            return {
                'status': 'ERROR',
                'message': f"Job failed: {training_job.get('FailureReason', 'Unknown failure reason')}"
            }
        
        return {
            'status': 'SUCCESS',
            'message': 'No container errors found'
        }

    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f"Error fetching logs: {str(e)}"
        }

def get_training_metrics(training_job):
    """
    Extract relevant metrics from the training job
    """
    try:
        metrics = {
            'billableTimeInSeconds': training_job.get('BillableTimeInSeconds', 0),
            'trainingTimeInSeconds': training_job.get('TrainingTimeInSeconds', 0),
            'instanceType': training_job.get('ResourceConfig', {}).get('InstanceType', ''),
            'instanceCount': training_job.get('ResourceConfig', {}).get('InstanceCount', 0),
            'maxRuntimeInSeconds': training_job.get('StoppingCondition', {}).get('MaxRuntimeInSeconds', 0)
        }

        # Add any custom metrics from training
        if 'FinalMetricDataList' in training_job:
            for metric in training_job['FinalMetricDataList']:
                metrics[metric['MetricName']] = metric['Value']

        return metrics
    except Exception as e:
        raise RuntimeError(f"Error extracting metrics: {str(e)}") from e

def put_ddb_item(table, item):
    """
    # Put item in DynamoDB
    """
    try:
        table.put_item(Item=item)
        print(f"Created new workflow in DynamoDB: {item}")
    except ClientError as e:
        status = f"Error trying to add new value got {item} into the the DB {table}. Error: {e}"
        raise SystemError(status) from e

def get_ddb_item_value(table, key):
    """
    Get item value in DDB
    """
    try:
        # Get the value from the table using the key
        result = table.get_item(Key=key)
        print(f"Object {key} from DynamoDB {table} is {result}")
        return result
    except ClientError as e:
        status = f"Error getting object {key} value from the DynamoDB table {table}. Error: {e}"
        raise SystemError(status) from e

def update_ddb_item_value(table, key, update_expression, expression_attribute_values):
    """
    Update item value in DDB
    """
    try:
        update_result = table.update_item(
            Key=key,
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_attribute_values,
            ReturnValues='ALL_NEW'
        )
        print(f"Updated value for {key}: {update_result['Attributes']}")
    except ClientError as e:
        status = f"Error trying to update the item with key, update expression and update values. Error: {e}"
        raise SystemError(status) from e

def lambda_handler(event, context):
    """
    Main lambda event handler
    """
    try:
        print(event)
        # initialize boto3 clients 
        dynamodb = boto3.resource('dynamodb')
        sns_client = boto3.client("sns")
        table_name = os.environ['DDB_TABLE_NAME']
        table = dynamodb.Table(table_name)
        sns_topic_arn = os.environ['SNS_TOPIC_ARN']

        key = {
            'uuid': _validate_uuid(event['envVars']['UUID'])
        }

        # Update end time stamp in DynamoDB
        current_date = datetime.now()
        update_expression = 'SET endTimestamp = :stopTimestamp'
        expression_attribute_values = {':stopTimestamp': str(current_date)}
        update_ddb_item_value(table, key, update_expression, expression_attribute_values)

        # Update elapsed stamp in DynamoDB
        result_workflow = get_ddb_item_value(table, key)
        print(f"Result Workflow: {result_workflow}")
        start_date = result_workflow["Item"]["startTimestamp"]
        start_date = parser.parse(start_date)
        elapsed_time = str(current_date - start_date)
        update_expression = 'SET elapsedTimestamp = :elapsedTime'
        expression_attribute_values = {':elapsedTime': elapsed_time}
        update_ddb_item_value(table, key, update_expression, expression_attribute_values)

        # Check if there was an error in the previous state
        error = event.get('error', None)
        training_job_name = str(event['envVars']['UUID'])
        
        # Check if this is a Batch job
        is_batch_job = event.get('result', {}).get('JobId') is not None or event.get('envVars', {}).get('COMPUTE_TYPE') == 'batch'
        
        # Detect waitForTaskToken success: container sends {"status": "SUCCEEDED", "uuid": "..."}
        task_token_success = event.get('result', {}).get('status') == 'SUCCEEDED'
        
        # For successful Batch jobs, skip error checking
        if is_batch_job and event.get('status') == 'SUCCESS' and (event.get('result', {}).get('Status') == 'SUCCEEDED' or task_token_success):
            # Get timing information from Batch job result or Batch API
            batch_result = event.get('result', {})
            started_at = batch_result.get('StartedAt', 0)
            stopped_at = batch_result.get('StoppedAt', 0)
            submitted_at = batch_result.get('SubmittedAt', 0)

            # waitForTaskToken path: result has no timing fields — look up from Batch API via batchJobId in DynamoDB
            if not started_at:
                try:
                    ddb_item = get_ddb_item_value(table, key).get('Item', {})
                    batch_job_id = ddb_item.get('batchJobId')
                    if batch_job_id and batch_job_id != 'pending':
                        batch_client = boto3.client('batch')
                        job_desc = batch_client.describe_jobs(jobs=[batch_job_id])
                        jobs = job_desc.get('jobs', [])
                        if jobs:
                            started_at = jobs[0].get('startedAt', 0)
                            stopped_at = jobs[0].get('stoppedAt', 0)
                            submitted_at = jobs[0].get('createdAt', 0)
                            print(f"Retrieved timing from Batch API: started={started_at}, stopped={stopped_at}")
                except Exception as timing_err:
                    print(f"Could not retrieve timing from Batch API: {timing_err}")

            # Calculate durations
            if started_at and stopped_at:
                compute_time = int((stopped_at - started_at) / 1000)  # Convert ms to seconds
                compute_minutes = compute_time // 60
                compute_seconds = compute_time % 60
                compute_time_str = f"{compute_minutes}m {compute_seconds}s"
            else:
                compute_time_str = "Unknown"
                compute_time = 0
                
            if submitted_at and started_at:
                queue_time = int((started_at - submitted_at) / 1000)  # Convert ms to seconds
                queue_minutes = queue_time // 60
                queue_seconds = queue_time % 60
                queue_time_str = f"{queue_minutes}m {queue_seconds}s"
            else:
                queue_time_str = "Unknown"
                queue_time = 0

            # Calculate total elapsed time
            if compute_time_str != "Unknown" or queue_time_str != "Unknown":
                total_seconds = (compute_time if compute_time_str != "Unknown" else 0) + \
                                (queue_time if queue_time_str != "Unknown" else 0)
                total_minutes = total_seconds // 60
                total_secs = total_seconds % 60
                total_time_str = f"{total_minutes}m {total_secs}s"
            else:
                total_time_str = "Unknown"
            
            # List output files from S3 and store in DynamoDB
            s3_client = boto3.client('s3')
            output_files = []
            try:
                s3_output = event['envVars']['S3_OUTPUT']
                # Handle both s3://bucket/prefix and bucket/prefix formats
                if s3_output.startswith('s3://'):
                    s3_output = s3_output[5:]  # Remove s3://
                parts = s3_output.split('/', 1)
                bucket_name = parts[0]
                prefix_base = parts[1] if len(parts) > 1 else ''
                prefix = f"{prefix_base}/{event['envVars']['UUID']}/" if prefix_base else f"{event['envVars']['UUID']}/"
                response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
                for obj in response.get('Contents', []):
                    if obj['Key'].endswith(('.ply', '.spz', '.sog', '.usdz', '.mp4', '.glb')):
                        output_files.append({'filename': obj['Key'].split('/')[-1], 'size': obj['Size']})
            except Exception as e:
                print(f"Error listing output files: {e}")
            
            # Update DynamoDB status and output files
            update_expression = 'SET uuidStatus = :uuidStatus, outputFiles = :outputFiles'
            expression_attribute_values = {':uuidStatus': 'Complete', ':outputFiles': output_files}
            update_ddb_item_value(table, key, update_expression, expression_attribute_values)
            
            # Get additional job details from DynamoDB
            result_workflow = get_ddb_item_value(table, key)
            workflow_item = result_workflow.get("Item", {})
            
            # Extract model from training config
            model = "N/A"
            if 'training' in workflow_item and 'model' in workflow_item['training']:
                model = workflow_item['training']['model']
            elif 'model' in workflow_item:
                model = workflow_item['model']
            
            # Extract reconstruction software name
            recon_software = "N/A"
            if 'reconSoftwareName' in workflow_item:
                recon_software = workflow_item['reconSoftwareName']
            elif 'reconstruction' in workflow_item and 'softwareName' in workflow_item['reconstruction']:
                recon_software = workflow_item['reconstruction']['softwareName']
            
            # Extract s3Input
            s3_input = workflow_item.get('s3Input', 'N/A')
            
            # Extract instanceType
            instance_type = workflow_item.get('instanceType', 'N/A')
            
            # Determine if this is a refine job (runSfm == false)
            sfm_config = workflow_item.get('sfm', {})
            run_sfm = sfm_config.get('enable', True) if isinstance(sfm_config, dict) else True
            refine_job = not run_sfm
            
            # Send success notification
            message_text = f"""✅ Splat Processing Complete
            
File Processed Successfully: {_sanitize_text(event['envVars']['FILENAME'])}

📂 Output Location:
{event['envVars']['S3_OUTPUT']}/{event['envVars']['UUID']}

💻 Compute Method: AWS Batch (Spot Instances)

📋 Job Details:
• Model: {model}
• Reconstruction Software: {recon_software}
• S3 Input: {s3_input}
• Instance Type: {instance_type}

⏱️ Timing Details:
• Queue Time: {queue_time_str}
• Compute Time: {compute_time_str}
• Total Time: {total_time_str}

------------------------------------------
This is an automated message from the Splat Processing System"""

            sns_client.publish(
                TargetArn=sns_topic_arn,
                Message=message_text,
                Subject=_sanitize_text(f"✅ Splat Processing Complete: {event['envVars']['UUID']}"),
            )
            
            return {
                'statusCode': 200,
                'body': {
                    'status': 'Completed',
                    'computeMethod': 'AWS Batch',
                    'queueTime': queue_time_str,
                    'computeTime': compute_time_str
                }
            }

        # Skip timeout check for Batch jobs (handled differently)
        if not is_batch_job:
            # Check for timeout first
            is_timeout, timeout_message = check_for_timeout(training_job_name)
            if is_timeout:
                # Handle timeout as a failure
                error_message = f"""
        ❌ Training Job Timeout

        Your 3D Gaussian Splat job has timed out.

        {timeout_message}

        Possible reasons:
        1. The job exceeded the maximum allowed runtime
        2. The instance may have run out of memory
        3. The dataset might be too large for the selected instance type

        Recommendations:
        1. Try using a larger instance type
        2. Reduce the number of input images
        3. Decrease the maximum number of steps
        4. Check if your input media has any issues
        """
                # Send notification about timeout
                send_sns_notification(training_job_name, error_message, is_error=True)
                return {
                    'statusCode': 200,
                    'body': json.dumps('Timeout detected and notification sent')
                }

        # Only get SageMaker job details for non-Batch jobs
        if not is_batch_job:
            # Get the training job details
            sagemaker_client = boto3.client('sagemaker')
            response = sagemaker_client.describe_training_job(
                TrainingJobName=training_job_name
            )

            # Get container logs first
            container_logs = get_cloudwatch_logs(training_job_name)
        else:
            # For Batch jobs that reach here, they failed
            log_stream_name = None
            print(f"Batch job failed. Error: {error}")

            # 1. Try to extract log stream from the error cause (old Batch .sync path)
            if error:
                try:
                    cause_str = error.get('Cause', str(error)) if hasattr(error, 'get') else str(error)
                    if isinstance(cause_str, str) and cause_str.startswith('{'):
                        cause_data = json.loads(cause_str)
                        if 'Container' in cause_data and 'LogStreamName' in cause_data['Container']:
                            log_stream_name = cause_data['Container']['LogStreamName']
                        elif 'Attempts' in cause_data and len(cause_data['Attempts']) > 0:
                            attempt = cause_data['Attempts'][0]
                            if 'Container' in attempt and 'LogStreamName' in attempt['Container']:
                                log_stream_name = attempt['Container']['LogStreamName']
                except Exception as parse_error:
                    print(f"Error parsing batch error for logs: {parse_error}")

            # 2. waitForTaskToken path: look up batchJobId from DynamoDB then query Batch API
            if not log_stream_name:
                try:
                    ddb_item = get_ddb_item_value(table, key).get('Item', {})
                    batch_job_id = ddb_item.get('batchJobId')
                    if batch_job_id and batch_job_id != 'pending':
                        batch_client = boto3.client('batch')
                        job_desc = batch_client.describe_jobs(jobs=[batch_job_id])
                        jobs = job_desc.get('jobs', [])
                        if jobs:
                            job = jobs[0]
                            log_stream_name = job.get('container', {}).get('logStreamName')
                            print(f"Retrieved log stream from Batch API: {log_stream_name}")
                            # If still no log stream, job never started — diagnose why
                            if not log_stream_name:
                                job_status = job.get('status', 'UNKNOWN')
                                job_reason = job.get('statusReason', 'No reason provided')
                                print(f"Batch job {batch_job_id} status={job_status}, reason={job_reason}")
                                # Check compute environment and job queue health
                                ce_diagnostics = []
                                try:
                                    queue_name = job.get('jobQueue', '')
                                    queue_desc = batch_client.describe_job_queues(jobQueues=[queue_name])
                                    for q in queue_desc.get('jobQueues', []):
                                        q_state = q.get('state')
                                        q_status = q.get('status')
                                        q_reason = q.get('statusReason', '')
                                        ce_diagnostics.append(f"Job Queue '{q.get('jobQueueName')}': state={q_state}, status={q_status}, reason={q_reason}")
                                        for ce_order in q.get('computeEnvironmentOrder', []):
                                            ce_name = ce_order.get('computeEnvironment', '')
                                            ce_desc = batch_client.describe_compute_environments(computeEnvironments=[ce_name])
                                            for ce in ce_desc.get('computeEnvironments', []):
                                                ce_state = ce.get('state')
                                                ce_status = ce.get('status')
                                                ce_reason = ce.get('statusReason', '')
                                                ce_diagnostics.append(f"Compute Environment '{ce.get('computeEnvironmentName')}': state={ce_state}, status={ce_status}, reason={ce_reason}")
                                except Exception as ce_err:
                                    ce_diagnostics.append(f"Could not query compute environment: {ce_err}")
                                diag_str = '\n'.join(ce_diagnostics)
                                print(f"Batch infrastructure diagnostics:\n{diag_str}")
                                container_logs = {
                                    'status': 'ERROR',
                                    'message': (
                                        f"Batch job never started (no container launched).\n"
                                        f"Job ID: {batch_job_id}\n"
                                        f"Job status: {job_status}\n"
                                        f"Job reason: {job_reason}\n\n"
                                        f"Infrastructure diagnostics:\n{diag_str}\n\n"
                                        f"Common causes: compute environment INVALID after CDK update, "
                                        f"insufficient Spot capacity, or vCPU quota exceeded."
                                    )
                                }
                except Exception as batch_lookup_err:
                    print(f"Could not retrieve log stream from Batch API: {batch_lookup_err}")

            print(f"Log stream for error reporting: {log_stream_name}")

            if log_stream_name:
                container_logs = get_cloudwatch_logs(training_job_name, is_batch_job=True, log_stream_name=log_stream_name)
            elif 'container_logs' not in dir():
                container_logs = {'status': 'ERROR', 'message': 'Batch job failed - unable to retrieve logs (no log stream found)'}
            response = None

        # Check if container logs indicate an error
        if container_logs.get('status') == 'ERROR':
            raise RuntimeError(f"Container logs indicate error: {container_logs['message']}")

        # Process successful case
        output = {
            'statusCode': 200,
            'body': {
                'status': 'Completed',
                'metrics': {
                    'billableTimeInSeconds': response['BillableTimeInSeconds'],
                    'trainingTimeInSeconds': response['TrainingTimeInSeconds'],
                    'instanceType': response['ResourceConfig']['InstanceType'],
                    'instanceCount': response['ResourceConfig']['InstanceCount'],
                    'maxRuntimeInSeconds': response['StoppingCondition']['MaxRuntimeInSeconds']
                },
                'containerLogs': container_logs,
                'modelArtifacts': response['ModelArtifacts']['S3ModelArtifacts']
            }
        }

        # List output files from S3 and store in DynamoDB
        s3_client = boto3.client('s3')
        output_files = []
        try:
            s3_output = event['envVars']['S3_OUTPUT']
            # Handle both s3://bucket/prefix and bucket/prefix formats
            if s3_output.startswith('s3://'):
                s3_output = s3_output[5:]  # Remove s3://
            parts = s3_output.split('/', 1)
            bucket_name = parts[0]
            prefix_base = parts[1] if len(parts) > 1 else ''
            prefix = f"{prefix_base}/{event['envVars']['UUID']}/" if prefix_base else f"{event['envVars']['UUID']}/"
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            for obj in response.get('Contents', []):
                if obj['Key'].endswith(('.ply', '.spz', '.sog', '.usdz', '.mp4', '.glb')):
                    output_files.append({'filename': obj['Key'].split('/')[-1], 'size': obj['Size']})
        except Exception as e:
            print(f"Error listing output files: {e}")
        
        # Update DynamoDB status and output files
        update_expression = 'SET uuidStatus = :uuidStatus, outputFiles = :outputFiles'
        expression_attribute_values = {':uuidStatus': 'Complete', ':outputFiles': output_files}
        update_ddb_item_value(table, key, update_expression, expression_attribute_values)

        # Get additional job details from DynamoDB
        result_workflow = get_ddb_item_value(table, key)
        workflow_item = result_workflow.get("Item", {})
        
        # Extract model from training config
        model = "N/A"
        if 'training' in workflow_item and 'model' in workflow_item['training']:
            model = workflow_item['training']['model']
        elif 'model' in workflow_item:
            model = workflow_item['model']
        
        # Extract reconstruction software name
        recon_software = "N/A"
        if 'reconSoftwareName' in workflow_item:
            recon_software = workflow_item['reconSoftwareName']
        elif 'reconstruction' in workflow_item and 'softwareName' in workflow_item['reconstruction']:
            recon_software = workflow_item['reconstruction']['softwareName']
        
        # Extract s3Input
        s3_input = workflow_item.get('s3Input', 'N/A')
        
        # Extract instanceType
        instance_type = workflow_item.get('instanceType', 'N/A')
        
        # Determine if this is a refine job (runSfm == false)
        sfm_config = workflow_item.get('sfm', {})
        run_sfm = sfm_config.get('enable', True) if isinstance(sfm_config, dict) else True
        refine_job = not run_sfm
        
        # Determine compute type
        compute_type = "SageMaker (On-Demand)"
        
        # Format success message
        message_text = f"""✅ Splat Processing Complete
        
File Processed Successfully: {_sanitize_text(event['envVars']['FILENAME'])}

📂 Output Location:
{event['envVars']['S3_OUTPUT']}/{event['envVars']['UUID']}

💻 Compute Method: {compute_type}

📋 Job Details:
• Model: {model}
• Reconstruction Software: {recon_software}
• S3 Input: {s3_input}
• Instance Type: {instance_type}

📊 Processing Details:
{json.dumps(output, indent=2)}

------------------------------------------
This is an automated message from the Splat Processing System"""

        # Publish the success message
        response = sns_client.publish(
            TargetArn=sns_topic_arn,
            Message=message_text,
            Subject=_sanitize_text(f"✅ Splat Processing Complete: {event['envVars']['UUID']}"),
        )

        return output

    except Exception as e:
        # Update status in DynamoDB to reflect error
        update_expression = 'SET uuidStatus = :uuidStatus'
        expression_attribute_values = {':uuidStatus': 'Error'}
        update_ddb_item_value(table, key, update_expression, expression_attribute_values)

        # Always try to get container logs first
        # Check if this is a Batch job error
        is_batch_error = event.get('result', {}).get('JobId') is not None or event.get('envVars', {}).get('COMPUTE_TYPE') == 'batch'
        
        if is_batch_error:
            # Extract log stream name from the error details if available
            log_stream_name = None
            print(f"Batch error detected. Error object: {error}")
            
            if error:
                try:
                    if isinstance(error, dict) and 'Cause' in error:
                        cause_str = error['Cause']
                    elif hasattr(error, 'get') and error.get('Cause'):
                        cause_str = error.get('Cause')
                    else:
                        cause_str = str(error)
                    
                    print(f"Parsing cause string: {cause_str[:500]}...")
                    
                    if isinstance(cause_str, str) and cause_str.startswith('{'):
                        cause_data = json.loads(cause_str)
                        if 'Container' in cause_data and 'LogStreamName' in cause_data['Container']:
                            log_stream_name = cause_data['Container']['LogStreamName']
                        elif 'Attempts' in cause_data and len(cause_data['Attempts']) > 0:
                            attempt = cause_data['Attempts'][0]
                            if 'Container' in attempt and 'LogStreamName' in attempt['Container']:
                                log_stream_name = attempt['Container']['LogStreamName']
                    
                    print(f"Extracted log stream name: {log_stream_name}")
                except Exception as parse_error:
                    print(f"Error parsing batch error details: {parse_error}")
            
            # waitForTaskToken fallback: look up batchJobId from DynamoDB then Batch API
            if not log_stream_name:
                try:
                    ddb_item = get_ddb_item_value(table, key).get('Item', {})
                    batch_job_id = ddb_item.get('batchJobId')
                    if batch_job_id and batch_job_id != 'pending':
                        batch_client = boto3.client('batch')
                        job_desc = batch_client.describe_jobs(jobs=[batch_job_id])
                        jobs = job_desc.get('jobs', [])
                        if jobs:
                            job = jobs[0]
                            log_stream_name = job.get('container', {}).get('logStreamName')
                            print(f"Retrieved log stream from Batch API (except path): {log_stream_name}")
                            if not log_stream_name:
                                job_status = job.get('status', 'UNKNOWN')
                                job_reason = job.get('statusReason', 'No reason provided')
                                print(f"Batch job {batch_job_id} status={job_status}, reason={job_reason}")
                                ce_diagnostics = []
                                try:
                                    queue_name = job.get('jobQueue', '')
                                    queue_desc = batch_client.describe_job_queues(jobQueues=[queue_name])
                                    for q in queue_desc.get('jobQueues', []):
                                        q_state = q.get('state')
                                        q_status = q.get('status')
                                        q_reason = q.get('statusReason', '')
                                        ce_diagnostics.append(f"Job Queue '{q.get('jobQueueName')}': state={q_state}, status={q_status}, reason={q_reason}")
                                        for ce_order in q.get('computeEnvironmentOrder', []):
                                            ce_name = ce_order.get('computeEnvironment', '')
                                            ce_desc = batch_client.describe_compute_environments(computeEnvironments=[ce_name])
                                            for ce in ce_desc.get('computeEnvironments', []):
                                                ce_state = ce.get('state')
                                                ce_status = ce.get('status')
                                                ce_reason = ce.get('statusReason', '')
                                                ce_diagnostics.append(f"Compute Environment '{ce.get('computeEnvironmentName')}': state={ce_state}, status={ce_status}, reason={ce_reason}")
                                except Exception as ce_err:
                                    ce_diagnostics.append(f"Could not query compute environment: {ce_err}")
                                diag_str = '\n'.join(ce_diagnostics)
                                print(f"Batch infrastructure diagnostics:\n{diag_str}")
                                container_logs = {
                                    'status': 'ERROR',
                                    'message': (
                                        f"Batch job never started (no container launched).\n"
                                        f"Job ID: {batch_job_id}\n"
                                        f"Job status: {job_status}\n"
                                        f"Job reason: {job_reason}\n\n"
                                        f"Infrastructure diagnostics:\n{diag_str}\n\n"
                                        f"Common causes: compute environment INVALID after CDK update, "
                                        f"insufficient Spot capacity, or vCPU quota exceeded."
                                    )
                                }
                except Exception as batch_lookup_err:
                    print(f"Could not retrieve log stream from Batch API: {batch_lookup_err}")
            
            if log_stream_name:
                container_logs = get_cloudwatch_logs(training_job_name, is_batch_job=True, log_stream_name=log_stream_name)
            elif 'container_logs' not in dir():
                container_logs = {'status': 'ERROR', 'message': 'Batch job failed - unable to retrieve logs (no log stream found)'}
        else:
            container_logs = get_cloudwatch_logs(training_job_name)
        
        error_message = container_logs.get('message', '')
        
        # Handle error cases with detailed logging
        error_details = {
            'statusCode': 500,
            'body': {
                'status': 'Failed',
                'containerError': container_logs['message'] if container_logs.get('status') == 'ERROR' else 'No container errors found',
                'error': str(e) if not error else error
            }
        }
        
        # Check if this is an SFM failure for specialized messaging
        if container_logs.get('status') == 'ERROR' and 'Structure from Motion (SFM) Reconstruction Failed' in error_message:
            error_details['body']['error_type'] = 'SFM_FAILURE'
            subject_prefix = "⚠️ Structure from Motion (SFM) Processing Error"
        else:
            subject_prefix = "⚠️ Splat Processing Error"
            
        message_text = f"""{subject_prefix}

Failed to process file: {_sanitize_text(event['envVars']['FILENAME'])}

❌ Container Error Details:
{container_logs['message'][:50000] if container_logs.get('status') == 'ERROR' else 'Job failed: AlgorithmError: , exit code: 1'}

❌ Additional Error Information:
{json.dumps(error_details, indent=2)[:10000]}

------------------------------------------
This is an automated message from the Splat Processing System"""

        # Publish the error message
        response = sns_client.publish(
            TargetArn=sns_topic_arn,
            Message=message_text,
            Subject=_sanitize_text(f"{subject_prefix}: {event['envVars']['UUID']}"),
        )
