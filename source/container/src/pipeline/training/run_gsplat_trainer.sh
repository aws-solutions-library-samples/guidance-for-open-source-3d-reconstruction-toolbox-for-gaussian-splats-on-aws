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
# Wrapper script to run gsplat trainer with the correct pycolmap version

# Prepend gsplat-specific pycolmap to PYTHONPATH so it's found first
export PYTHONPATH="/opt/gsplat_pycolmap:${PYTHONPATH}"

# Debug: Verify pycolmap version being used
echo "=== GSPLAT PYCOLMAP DEBUG ==="
echo "PYTHONPATH: $PYTHONPATH"
python -c "import sys; print('Python path:', sys.path[:3])"
python -c "import numpy; print('Numpy version:', numpy.__version__); print('Numpy location:', numpy.__file__)"
python -c "import pycolmap; print('PyColmap location:', pycolmap.__file__)"
echo "=== END DEBUG ==="

# Derive GPU count from CUDA_VISIBLE_DEVICES if set, otherwise default to 1
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=1
fi

# gsplat's cli() reads WORLD_SIZE from env to determine how many workers to spawn via mp.spawn.
# Unset RANK/LOCAL_RANK so spawned workers initialize their own rank assignments correctly.
export WORLD_SIZE="$NUM_GPUS"
unset RANK LOCAL_RANK

# Run gsplat trainer directly with python - cli() in distributed.py handles mp.spawn internally.
exec python /opt/ml/code/gsplat/examples/simple_trainer.py "$@"
