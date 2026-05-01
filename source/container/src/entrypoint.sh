#!/bin/bash
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
# Shell-level entrypoint wrapper for the 3DGS pipeline container.
# Runs main.py and catches any non-zero exit — including pre-Python crashes
# (OOM, GPU init failure, SyntaxError at import time) — then calls
# SendTaskFailure so Step Functions is never left stuck waiting.
# Only fires the callback when TASK_TOKEN is set and AWS_BATCH_JOB_ID is
# present (i.e. running as a Batch job, not SageMaker or local debug).

set -o pipefail

python /opt/ml/code/main.py "$@"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "Container exited with code ${EXIT_CODE} — sending SendTaskFailure to Step Functions"
    if [ -n "${TASK_TOKEN}" ] && [ -n "${AWS_BATCH_JOB_ID}" ]; then
        REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-east-1}}"
        CAUSE="Container exited with code ${EXIT_CODE}. Check CloudWatch log stream for details."
        aws stepfunctions send-task-failure \
            --task-token "${TASK_TOKEN}" \
            --error "ContainerExitError" \
            --cause "${CAUSE}" \
            --region "${REGION}" 2>/dev/null || \
            echo "Warning: send-task-failure call failed (token may be expired)"
    fi
    exit $EXIT_CODE
fi
