#!/bin/bash
# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Wrapper script to run Map-Anything with dedicated pycolmap 3.10.0 environment

# Activate the map_anything virtual environment
source /opt/map_anything_env/bin/activate

# Run the map_anything script with all passed arguments
exec python reconstruction/run_map_anything.py "$@"
