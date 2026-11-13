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

# Must be set before PyTorch's CUDA allocator initializes (first CUDA op).
# Reduces OOM errors on large scenes by allowing the allocator to grow
# segments incrementally rather than pre-allocating large contiguous blocks.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CUDA forward compatibility, only when the host driver needs it.
# The Dockerfile puts the cuda-compat libcuda first on LD_LIBRARY_PATH so the CUDA
# 13.2 runtime works on older host drivers. On a host whose driver is as new as
# (or newer than) the compat libcuda, loading it fails every CUDA call with
# error 803 "unsupported display driver / cuda driver combination" and PyTorch
# falls back to CPU. Following the AWS guidance for SageMaker containers, use
# the compat libraries only when the host driver is older than they are:
# https://docs.aws.amazon.com/sagemaker/latest/dg/container-nvidia-compliance.html
verlt() {
    [ "$1" = "$2" ] && return 1 || [ "$1" = "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" ]
}
CUDA_COMPAT_DIR=""
for _d in /usr/local/cuda/compat /usr/local/cuda-*/compat; do
    if [ -e "${_d}/libcuda.so.1" ]; then CUDA_COMPAT_DIR="${_d}"; break; fi
done
# Drop any compat entries inherited from the image ENV, then re-add if needed.
_ld=""
IFS=':' read -r -a _ld_parts <<< "${LD_LIBRARY_PATH}"
for _p in "${_ld_parts[@]}"; do
    case "${_p}" in
        ""|*/compat|*/compat/) ;;
        *) _ld="${_ld:+${_ld}:}${_p}" ;;
    esac
done
if [ -n "${CUDA_COMPAT_DIR}" ]; then
    CUDA_COMPAT_MAX_DRIVER_VERSION=$(readlink "${CUDA_COMPAT_DIR}/libcuda.so.1" | sed -n 's/^libcuda\.so\.//p')
    NVIDIA_DRIVER_VERSION=$(sed -n 's/^NVRM.*Kernel Module *\([0-9.]*\).*$/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)
    if [ -z "${NVIDIA_DRIVER_VERSION}" ] && command -v nvidia-smi >/dev/null 2>&1; then
        NVIDIA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')
    fi
    echo "Host NVIDIA driver: ${NVIDIA_DRIVER_VERSION:-unknown}; CUDA compat libcuda: ${CUDA_COMPAT_MAX_DRIVER_VERSION:-unknown} (${CUDA_COMPAT_DIR})"
    # cuda-compat-13-x only works on r580+ base drivers (NVIDIA forward-compatibility
    # matrix); on older hosts every CUDA call fails with error 803 whatever we do here.
    _drv_major="${NVIDIA_DRIVER_VERSION%%.*}"
    if [ -n "${_drv_major}" ] && [ "${_drv_major}" -lt 580 ] 2>/dev/null; then
        echo "ERROR: host NVIDIA driver ${NVIDIA_DRIVER_VERSION} is older than r580. This image uses CUDA 13.2, whose" \
             "forward-compat libraries need an r580+ driver, so the GPU will be unusable (CUDA error 803)." \
             "Run on a host with driver >= 580 (e.g. AWS Batch with a newer ECS GPU AMI) or build the image with CUDA 12.x."
    fi
    if [ -z "${NVIDIA_DRIVER_VERSION}" ] || [ -z "${CUDA_COMPAT_MAX_DRIVER_VERSION}" ] || \
            verlt "${NVIDIA_DRIVER_VERSION}" "${CUDA_COMPAT_MAX_DRIVER_VERSION}"; then
        echo "Using CUDA forward-compat libraries (host driver older than ${CUDA_COMPAT_MAX_DRIVER_VERSION} or unknown)"
        _ld="${CUDA_COMPAT_DIR}${_ld:+:${_ld}}"
    else
        echo "Host driver is new enough; not using CUDA forward-compat libraries"
    fi
fi
export LD_LIBRARY_PATH="${_ld}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

python /opt/ml/code/main.py "$@"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "Container exited with code ${EXIT_CODE} — sending SendTaskFailure to Step Functions"
    if [ -n "${TASK_TOKEN}" ] && [ -n "${AWS_BATCH_JOB_ID}" ]; then
        REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-east-1}}"
        CAUSE="Container exited with code ${EXIT_CODE}. Check CloudWatch log stream for details."
        for attempt in 1 2 3; do
            aws stepfunctions send-task-failure \
                --task-token "${TASK_TOKEN}" \
                --error "ContainerExitError" \
                --cause "${CAUSE}" \
                --region "${REGION}" && break
            echo "Warning: send-task-failure attempt ${attempt} failed, retrying..."
            sleep 5
        done
    fi
    exit $EXIT_CODE
fi
