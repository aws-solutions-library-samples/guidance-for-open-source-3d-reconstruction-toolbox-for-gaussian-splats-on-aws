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

import json
import os

def lambda_handler(event, context):
    """
    Select appropriate Batch job definition and queue based on instance type
    """
    
    # Get default job definitions and queues
    default_large = os.environ.get('BATCH_JOB_DEFINITION_LARGE')
    default_xlarge = os.environ.get('BATCH_JOB_DEFINITION_XLARGE')
    default_queue = os.environ.get('BATCH_JOB_QUEUE')
    
    # Instance type to job definition mapping
    instance_type_mapping = {
        # Large instances (16 vCPUs) - Default
        'g5.4xlarge': os.environ.get('BATCH_JOB_DEFINITION_G5_4XLARGE', default_large),
        'g6.4xlarge': os.environ.get('BATCH_JOB_DEFINITION_G6_4XLARGE', default_large),
        'g6e.4xlarge': os.environ.get('BATCH_JOB_DEFINITION_G6E_4XLARGE', default_large),
        
        # Extra Large instances (32 vCPUs)
        'g5.8xlarge': os.environ.get('BATCH_JOB_DEFINITION_G5_8XLARGE', default_xlarge),
        'g6.8xlarge': os.environ.get('BATCH_JOB_DEFINITION_G6_8XLARGE', default_xlarge),

        # Multi-GPU instances
        'g5.12xlarge': os.environ.get('BATCH_JOB_DEFINITION_G5_12XLARGE', default_xlarge),
    }

    # Instance type to dedicated job queue mapping
    queue_mapping = {
        'g5.4xlarge':  os.environ.get('BATCH_JOB_QUEUE_G5_4XLARGE',  default_queue),
        'g5.8xlarge':  os.environ.get('BATCH_JOB_QUEUE_G5_8XLARGE',  default_queue),
        'g5.12xlarge': os.environ.get('BATCH_JOB_QUEUE_G5_12XLARGE', default_queue),
        'g6.4xlarge':  os.environ.get('BATCH_JOB_QUEUE_G6_4XLARGE',  default_queue),
        'g6.8xlarge':  os.environ.get('BATCH_JOB_QUEUE_G6_8XLARGE',  default_queue),
        'g6e.4xlarge': os.environ.get('BATCH_JOB_QUEUE_G6E',         default_queue),
    }
    
    try:
        # Extract instance type from the event
        instance_type = event.get('envVars', {}).get('INSTANCE_TYPE', 'ml.g5.4xlarge')
        
        print(f"DEBUG: Raw instance type from event: {instance_type}")
        print(f"DEBUG: Full event envVars: {event.get('envVars', {})}")
        
        # Remove 'ml.' prefix if present (for SageMaker instance types)
        if instance_type.startswith('ml.'):
            instance_type = instance_type[3:]
        
        print(f"DEBUG: Processed instance type: {instance_type}")
        print(f"DEBUG: Available instance type mappings: {list(instance_type_mapping.keys())}")
        
        # Select appropriate job definition
        selected_job_definition = instance_type_mapping.get(
            instance_type, 
            default_large  # Default to large
        )
        
        # Validate that we have a job definition
        if not selected_job_definition:
            error_msg = f"No job definition found for instance type {instance_type}. Available mappings: {list(instance_type_mapping.keys())}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        print(f"DEBUG: Selected job definition: {selected_job_definition}")
        
        # Return the event with the selected job definition and queue
        result = event.copy()
        result['selectedBatchJobDefinition'] = selected_job_definition
        result['selectedBatchJobQueue'] = queue_mapping.get(instance_type, default_queue)
        
        print(f"Selected job definition {selected_job_definition} for instance type {instance_type}")
        print(f"Selected queue {result['selectedBatchJobQueue']} for instance type {instance_type}")
        
        return result
        
    except Exception as e:
        print(f"Error selecting job definition: {str(e)}")
        # Return default job definition on error
        result = event.copy()
        result['selectedBatchJobDefinition'] = default_large or os.environ.get('BATCH_JOB_DEFINITION_LARGE', '')
        return result