# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import json
import os

def lambda_handler(event, context):
    """
    Select appropriate Batch job definition based on instance type
    """
    
    # Instance type to job definition mapping
    instance_type_mapping = {
        # Large instances (16 vCPUs) - Default
        'g5.4xlarge': os.environ.get('BATCH_JOB_DEFINITION_G5_4XLARGE', os.environ['BATCH_JOB_DEFINITION_LARGE']),
        'g6.4xlarge': os.environ['BATCH_JOB_DEFINITION_LARGE'],
        'g6e.4xlarge': os.environ['BATCH_JOB_DEFINITION_LARGE'],
        
        # Extra Large instances (32 vCPUs)
        'g5.8xlarge': os.environ['BATCH_JOB_DEFINITION_XLARGE'],
        'g6.8xlarge': os.environ['BATCH_JOB_DEFINITION_XLARGE'],
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
            os.environ['BATCH_JOB_DEFINITION_LARGE']  # Default to large
        )
        
        print(f"DEBUG: Selected job definition: {selected_job_definition}")
        
        # Return the event with the selected job definition
        result = event.copy()
        result['selectedBatchJobDefinition'] = selected_job_definition
        
        print(f"Selected job definition {selected_job_definition} for instance type {instance_type}")
        
        return result
        
    except Exception as e:
        print(f"Error selecting job definition: {str(e)}")
        # Return default job definition on error
        result = event.copy()
        result['selectedBatchJobDefinition'] = os.environ['BATCH_JOB_DEFINITION_LARGE']
        return result