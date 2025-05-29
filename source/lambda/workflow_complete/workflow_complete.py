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

""" A sample script to receive unique metadata file uploaded to S3 for gaussian splat creation
and start the workflow """

import os
import json
import boto3
from datetime import datetime
from dateutil import parser

from botocore.exceptions import ClientError

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

def is_sfm_failure(message):
    """Check if the message indicates an SFM reconstruction failure."""
    if ('Command \'glomap mapper\'' in message and 
        'failed with return code -11' in message and 
        'glomap::ViewGraph::KeepLargestConnectedComponents' in message and 
        'SIGSEGV' in message):
        print("SFM failure detected!")  # Debug log
        return True
    return False

def get_cloudwatch_logs(training_job_name):
    logs_client = boto3.client('logs')
    
    try:
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
            'Failed'
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
            'glomap::ViewGraph::KeepLargestConnectedComponents'
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
        
        for stream in response.get('logStreams', []):
            events = logs_client.get_log_events(
                logGroupName='/aws/sagemaker/TrainingJobs',
                logStreamName=stream['logStreamName'],
                startFromHead=False
            )
            
            for event in events['events']:
                message = event['message']
                
                # Check for SFM failure first
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
            - Error: ViewGraph Component Connection Failure
            - Component: glomap::ViewGraph::KeepLargestConnectedComponents
            - Exit Code: -11
            - Status: Process terminated with SIGSEGV"""
                    error_messages.append(sfm_error_message)
                    found_error = True
                    break

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
                
                    if found_error and len(error_messages) < 15:
                        error_messages.append(message.strip())
            
            if found_error:
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
            'uuid': str(event['envVars']['UUID'])
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

        # Get the training job details
        sagemaker_client = boto3.client('sagemaker')
        response = sagemaker_client.describe_training_job(
            TrainingJobName=training_job_name
        )

        # Get container logs first
        container_logs = get_cloudwatch_logs(training_job_name)

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

        # Update DynamoDB status
        update_expression = 'SET uuidStatus = :uuidStatus'
        expression_attribute_values = {':uuidStatus': 'Complete'}
        update_ddb_item_value(table, key, update_expression, expression_attribute_values)

        # Format success message
        message_text = f"""✅ Splat Processing Complete
        
File Processed Successfully: {event['envVars']['FILENAME']}

📂 Output Location:
{event['envVars']['S3_OUTPUT']}/{event['envVars']['UUID']}

📊 Processing Details:
{json.dumps(output, indent=2)}

------------------------------------------
This is an automated message from the Splat Processing System"""

        # Publish the success message
        response = sns_client.publish(
            TargetArn=sns_topic_arn,
            Message=message_text,
            Subject=f"✅ Splat Processing Complete: {event['envVars']['UUID']}",
        )

        return output

    except Exception as e:
        # Update status in DynamoDB to reflect error
        update_expression = 'SET uuidStatus = :uuidStatus'
        expression_attribute_values = {':uuidStatus': 'Error'}
        update_ddb_item_value(table, key, update_expression, expression_attribute_values)

        # Always try to get container logs first
        container_logs = get_cloudwatch_logs(training_job_name)
        
        error_message = container_logs.get('message', '')
        
        # Check if this is an SFM failure
        if container_logs.get('status') == 'ERROR' and 'Structure from Motion (SFM) Reconstruction Failed' in error_message:
            error_details = {
                'statusCode': 500,
                'body': {
                    'status': 'Failed',
                    'error_type': 'SFM_FAILURE',
                    'containerError': error_message,
                    'error': str(e) if not error else error
                }
            }
            
            message_text = f"""⚠️ Structure from Motion (SFM) Processing Error

    Failed to process file: {event['envVars']['FILENAME']}

    {error_message}

    Additional Details:
    {json.dumps(error_details, indent=2)[:1000]}

    ------------------------------------------
    This is an automated message from the Splat Processing System"""
        else:
            # Handle other types of errors as before
            error_details = {
                'statusCode': 500,
                'body': {
                    'status': 'Failed',
                    'containerError': container_logs['message'] if container_logs.get('status') == 'ERROR' else 'No container errors found',
                    'error': str(e) if not error else error
                }
            }
            
            message_text = f"""⚠️ Splat Processing Error

    Failed to process file: {event['envVars']['FILENAME']}

    ❌ Container Error Details:
    {container_logs['message'][:50000] if container_logs.get('status') == 'ERROR' else 'No container errors found'}

    ❌ Additional Error Information:
    {json.dumps(error_details, indent=2)[:10000]}

    ------------------------------------------
    This is an automated message from the Splat Processing System"""

        # Publish the error message
        response = sns_client.publish(
            TargetArn=sns_topic_arn,
            Message=message_text,
            Subject=f"⚠️ Splat Processing Error: {event['envVars']['UUID']}",
        )