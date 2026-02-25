#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""
Wrapper to run splatfacto-w export with gsplat==1.4.0
"""

import sys
import os

# Prepend isolated environment to sys.path BEFORE any other imports
sys.path.insert(0, '/opt/splatfacto_w_env')

# Patch torch.load to use weights_only=False for compatibility
import torch
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# Now run the export script
if __name__ == "__main__":
    import runpy
    runpy.run_path('/opt/ml/code/splatfacto-w/export_script.py', run_name='__main__')
