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

# A sample script to generate a unique metadata file and upload it to S3 for gaussian splat creation
import os
import uuid
import json
import boto3

s3 = boto3.client('s3')
unique_uuid = uuid.uuid4()
filename = f"./workflow-submissions/{str(unique_uuid)}.json"

if os.path.isdir("./workflow-submissions") == False:
    os.mkdir("./workflow-submissions")

"""
!!! Input bucket name and media filename to use for submitting job !!!
!!! UPLOAD MEDIA FILE TO THE S3 LOCATION BEFORE RUNNING THIS SCRIPT !!!
!!! S3 LOCATION: s3://<bucket-name>/<s3_input_prefix>/<media-filename> !!!
"""

s3_bucket_name = ""

s3_job_prefix = "workflow-input"
s3_input_prefix = "media-input"

s3_output_prefix = "workflow-output"
media_filename = ""
instance_type = "ml.g6.4xlarge" 

"""
!!! Change the input parameters for each option below !!!
"""
# Options Selections:
# instance_type: "ml.g5.4xlarge" or "ml.g5.8xlarge" or "ml.g5.12xlarge (multi-gpu)"
# logVerbosity: debug, info, error
# reconstruction.softwareName: "colmap" or "glomap" or "hloc" or "map_anything"
# reconstruction.matchingMethod: "sequential", "spatial", "vocab", "exhaustive"
# reconstruction.sourceCoordinateName: "arkit" or "arcore" or "opengl" or "opencv" or "ros"
# reconstruction.enableFlHeuristic: "true" or "false" (estimate focal length as multiplier * max(w,h))
# reconstruction.flHeuristicValue: e.g. "1.2" (coefficient for focal length heuristic)
# reconstruction.enableFlMetric: "true" or "false" (convert focal length mm to pixels via 35mm-equivalent formula)
# reconstruction.flMetricValue: e.g. "6.86" (focal length in mm from EXIF FocalLength tag, e.g. iPhone 14 Pro main = 6.86)
# training.model: "splatfacto" or "splatfacto-big" or "splatfacto-w-light" or "splatfacto-mcmc" or "3dgrt" or "3dgut" or "nerfacto"
# training.3dIsp: "none" or "bilagrid" or "ppisp"
# training.preserveSceneScale: "true" or "false"
# training.enableDepthLoss: "true" or "false"
# postProcessing.cropMode: "environment" or "rigid_object"
# postProcessing.enableVideoExport: "true" or "false"
# sphericalCamera.cubeFacesToRemove: "['back', 'down', 'front', 'left', 'right', 'up']" or "['']"

file_contents = {
    "uuid": str(unique_uuid),
    "instanceType": instance_type,
    "useSpotInstance": "false",
    "logVerbosity": "info",
    "s3": {
        "bucketName": s3_bucket_name,
        "inputPrefix": s3_input_prefix,
        "inputKey": media_filename,
        "outputPrefix": s3_output_prefix
    },
    "videoProcessing": {
        "maxNumImages": "300",
        "videoStartTime": "0",
        "videoStopTime": "None",
        "filterBlurryImages": "true"
    },
    "reconstruction": {
        "enable": "true",
        "softwareName": "glomap",
        "enableEnhancedFeatureExtraction": "false",
        "matchingMethod": "sequential",
        "enableFlHeuristic": "false",
        "flHeuristicValue": "1.2",
        "enableFlMetric": "false",
        "flMetricValue": "24",
        "posePriors": {
            "usePosePriorColmapModelFiles": "false",
            "usePosePriorTransformJson": {
                "enable": "false",
                "sourceCoordinateName": "arkit",
                "poseIsWorldToCam": "true",
            },
        }
    },
    "training": {
        "enable": "true",
        "maxSteps": "15000",
        "numGaussians": "1000000",
        "model": "splatfacto",
        "3dIsp": "none",
        "preserveSceneScale": "false",
        "enableDepthLoss": "false"
    },
    "postProcessing": {
        "cropOutputBounds": "false",
        "cropMode": "environment",
        "cleanSplat": "false",
        "enableSpz": "true",
        "enableSog": "true",
        "enableUsdz": "true",
        "enableVideoExport": "true",
        "generateCollision": "false",
        "collisionSceneType": "outdoor",
        "collisionSeedPos": "0,0,0",
        "generateLod": "false",
        "generateMesh": "true",
        "plyCoords": "rhyu",
        "spzCoords": "rhyu",
        "sogCoords": "rhyu",
        "usdzCoords": "rhyu"
    },
    "sphericalCamera": {
        "enable": "false",
        "cubeFacesToRemove": "['down', 'up']",
        "optimizeSequentialFrameOrder": "true"
    },
    "segmentation": {
        "backgroundRemoval": {
            "enable": "false",
            "model": "u2net", #"u2net", "sam2"
            "maskThreshold": "0.6",
        },
        "objectRemoval": {
            "enable": "false",
            "action": "erase", #remove
            "objects": "['human']",
        }
    }
}

try:
    file_out = open(filename, "w", encoding="utf-8")
    file_out.write(json.dumps(file_contents))
    file_out.close()
except Exception as e:
    print(f"Error saving output metadata file: {e}")
    raise e

try:
    s3.upload_file(
        Filename=filename,
        Bucket=s3_bucket_name,
        Key=f"{s3_job_prefix}/{unique_uuid}.json",
        ExtraArgs={
            "CacheControl":"no-cache",
            #"ServerSideEncryption": "aws:kms"
        }
    )
    print(f"""Successfully uploaded output metadata file: 
        {str(unique_uuid)}.json to s3://{s3_bucket_name}/{s3_input_prefix}""")
except Exception as e:
    print(f"Error uploading output metadata file: {e}")
    raise e
