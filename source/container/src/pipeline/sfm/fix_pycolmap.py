#!/usr/bin/env python3
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

"""
Fix pycolmap compatibility issue with camera_image_id attribute.
"""

import sys

def main():
    try:
        import pycolmap
        
        # Add missing camera_image_id attribute to Camera class
        if not hasattr(pycolmap._core.Camera, 'camera_image_id'):
            def get_camera_image_id(self):
                return getattr(self, 'image_id', 0)
            
            pycolmap._core.Camera.camera_image_id = property(get_camera_image_id)
            print("Applied pycolmap Camera.camera_image_id compatibility fix")
        else:
            print("pycolmap Camera.camera_image_id already exists")
            
    except Exception as e:
        print(f"Warning: Could not apply pycolmap fix: {e}")
        # Don't fail the pipeline for this fix
        pass

if __name__ == "__main__":
    main()