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

"""Selects the Batch job definition and queue by instance type, then submits the Batch job with the Step Functions task token."""

import json
import os
import boto3

def _build_container_environment(env_vars: dict, task_token: str) -> list:
    # Build the Batch ContainerOverrides environment list from the envVars dict,
    # injecting the Step Functions task token so the container can call back on completion.
    env_list = [{"name": k, "value": str(v)} for k, v in env_vars.items()]
    env_list.append({"name": "TASK_TOKEN", "value": task_token})
    return env_list

def lambda_handler(event, context):
    """Select appropriate Batch job definition and queue based on instance type, then submit the job."""

    # Get default job definitions and queues
    default_large = os.environ.get('BATCH_JOB_DEFINITION_LARGE')
    default_xlarge = os.environ.get('BATCH_JOB_DEFINITION_XLARGE')
    default_queue = os.environ.get('BATCH_JOB_QUEUE')

    # Instance type to job definition mapping
    instance_type_mapping = {
        'g5.4xlarge':  os.environ.get('BATCH_JOB_DEFINITION_G5_4XLARGE',  default_large),
        'g6.4xlarge':  os.environ.get('BATCH_JOB_DEFINITION_G6_4XLARGE',  default_large),
        'g6e.4xlarge': os.environ.get('BATCH_JOB_DEFINITION_G6E_4XLARGE', default_large),
        'g5.8xlarge':  os.environ.get('BATCH_JOB_DEFINITION_G5_8XLARGE',  default_xlarge),
        'g6.8xlarge':  os.environ.get('BATCH_JOB_DEFINITION_G6_8XLARGE',  default_xlarge),
        'g5.12xlarge': os.environ.get('BATCH_JOB_DEFINITION_G5_12XLARGE', default_xlarge),
    }

    # Instance type to dedicated job queue mapping
    queue_mapping = {
        'g5.4xlarge':  os.environ.get('BATCH_JOB_QUEUE_G5_4XLARGE',  default_queue),
        'g5.8xlarge':  os.environ.get('BATCH_JOB_QUEUE_G5_8XLARGE',  default_queue),
        'g5.12xlarge': os.environ.get('BATCH_JOB_QUEUE_G5_12XLARGE', default_queue),
        'g6.4xlarge':  os.environ.get('BATCH_JOB_QUEUE_G6_4XLARGE',  default_queue),
        'g6.8xlarge':  os.environ.get('BATCH_JOB_QUEUE_G6_8XLARGE',  default_queue),
        'g6e.4xlarge': os.environ.get('BATCH_JOB_QUEUE_G6E',          default_queue),
    }

    # Extract the task token and the original state machine event from the payload
    task_token = event.get('taskToken', '')
    sfn_event = event.get('event', event)

    try:
        env_vars = sfn_event.get('envVars', {})
        instance_type = env_vars.get('INSTANCE_TYPE', 'g5.4xlarge')
        print(f"Instance type from event: {instance_type}")

        # Remove 'ml.' prefix if present (SageMaker instance type format)
        if instance_type.startswith('ml.'):
            instance_type = instance_type[3:]

        selected_job_definition = instance_type_mapping.get(instance_type, default_large)
        selected_queue = queue_mapping.get(instance_type, default_queue)

        if not selected_job_definition:
            raise ValueError(f"No job definition found for instance type {instance_type}")

        print(f"Selected job definition: {selected_job_definition} queue: {selected_queue}")

        # Check if a Batch job with this UUID is already running or starting up
        # to avoid submitting a duplicate when Step Functions retries after a heartbeat timeout
        # while the original container is still running.
        batch_client = boto3.client('batch')
        uuid = env_vars.get('UUID', '')
        if uuid:
            existing = batch_client.list_jobs(
                jobQueue=selected_queue,
                filters=[{'name': 'JOB_NAME', 'values': [uuid]}]
            )
            active_jobs = [
                j for j in existing.get('jobSummaryList', [])
                if j['status'] in ('SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING')
            ]
            if active_jobs:
                print(f"Job {uuid} already active ({active_jobs[0]['status']}), skipping duplicate submission")
                # Send task failure so Step Functions does not hang — the active job
                # holds the correct token and will call back when it finishes
                if task_token:
                    sfn_client = boto3.client('stepfunctions')
                    sfn_client.send_task_failure(
                        taskToken=task_token,
                        error='DuplicateJobSkipped',
                        cause=f'Job {uuid} is already {active_jobs[0]["status"]} — original container will send callback'
                    )
                return {"jobId": active_jobs[0]['jobId'], "skipped": True}

        # Check if the Step Functions execution is still running before submitting.
        # If it was aborted or timed out, the task token is already invalid — skip submission.
        if task_token:
            try:
                sfn_client = boto3.client('stepfunctions')
                # Decode the execution ARN from the task token context is not possible directly,
                # but we can validate the token by attempting a heartbeat.
                sfn_client.send_task_heartbeat(taskToken=task_token)
            except sfn_client.exceptions.InvalidToken:
                print(f"Task token is no longer valid — execution was aborted. Skipping Batch submission for UUID={uuid}")
                return {"skipped": True, "reason": "execution_aborted"}
            except sfn_client.exceptions.TaskTimedOut:
                print(f"Task token timed out — execution already moved on. Skipping Batch submission for UUID={uuid}")
                return {"skipped": True, "reason": "task_timed_out"}
            except Exception as _hb_err:
                # Heartbeat failed for another reason — proceed anyway
                print(f"Heartbeat check failed ({_hb_err}), proceeding with submission")

        # Submit the Batch job directly, passing the task token to the container.
        # Step Functions is already parked on the waitForTaskToken — the container
        # will call SendTaskSuccess/Failure when the pipeline completes.
        response = batch_client.submit_job(
            jobName=env_vars.get('UUID', 'gs-job'),
            jobQueue=selected_queue,
            jobDefinition=selected_job_definition,
            containerOverrides={
                'environment': _build_container_environment(env_vars, task_token)
            }
        )
        batch_job_id = response['jobId']
        print(f"Batch job submitted: UUID={uuid} batchJobId={batch_job_id}")
        print(f"To find logs: aws batch describe-jobs --jobs {batch_job_id} --query 'jobs[0].container.logStreamName' --output text")

        # Write the batchJobId to DynamoDB so it can be looked up by UUID.
        # Log stream is not yet available at submission time — it will be populated
        # once the container starts. Use 'pending' as placeholder.
        ddb_table = os.environ.get('DDB_TABLE_NAME', '')
        if ddb_table and uuid:
            try:
                ddb = boto3.resource('dynamodb').Table(ddb_table)
                ddb.update_item(
                    Key={'uuid': uuid},
                    UpdateExpression='SET batchJobId = :jid, batchLogStream = :ls',
                    ExpressionAttributeValues={':jid': batch_job_id, ':ls': 'pending'}
                )
            except Exception as ddb_err:
                print(f"Warning: could not write batchJobId to DynamoDB: {ddb_err}")

        # Return without calling SendTaskSuccess — the container does that on completion
        return {"jobId": batch_job_id}

    except Exception as e:
        print(f"Error submitting Batch job: {e}")
        # Send task failure so Step Functions does not hang waiting for a callback
        # that will never arrive
        if task_token:
            sfn_client = boto3.client('stepfunctions')
            sfn_client.send_task_failure(
                taskToken=task_token,
                error='BatchSubmitError',
                cause=str(e)[:32768]
            )
        raise