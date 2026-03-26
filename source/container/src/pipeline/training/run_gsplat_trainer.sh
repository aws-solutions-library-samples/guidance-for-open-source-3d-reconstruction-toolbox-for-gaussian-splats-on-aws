#!/bin/bash
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
