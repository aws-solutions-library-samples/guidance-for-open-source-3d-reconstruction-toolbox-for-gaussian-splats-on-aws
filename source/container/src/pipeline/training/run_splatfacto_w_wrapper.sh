#!/bin/bash
# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Wrapper script to run splatfacto-w-light with gsplat==1.4.0
# This isolates the older gsplat version required by splatfacto-w-light

# Prepend splatfacto-w environment to PYTHONPATH (before global packages)
export PYTHONPATH="/opt/splatfacto_w_env:${PYTHONPATH}"

echo "Running splatfacto-w-light with gsplat==1.4.0 from /opt/splatfacto_w_env"
echo "PYTHONPATH: $PYTHONPATH"

# Execute ns-train with all passed arguments
exec ns-train "$@"
