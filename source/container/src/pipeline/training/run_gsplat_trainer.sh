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

# Enable CUDA debugging for better error messages
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Run gsplat trainer with gsplat's pycolmap in PYTHONPATH
exec python /opt/ml/code/gsplat/examples/simple_trainer.py "$@"
