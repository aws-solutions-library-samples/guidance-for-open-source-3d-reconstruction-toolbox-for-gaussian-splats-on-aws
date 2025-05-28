# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

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
instance_type = "ml.g6e.4xlarge" 

"""
!!! Change the input parameters for each option below !!!
"""
# Options Selections:
# instance_type: "ml.g5.4xlarge" or "ml.g5.8xlarge" or "ml.g5.12xlarge (multi-gpu)"
# logVerbosity: debug, info, error
# sfm.softwareName: "colmap" or "glomap"
# sfm.matchingMethod: "sequential", "spatial", "vocab", "exhaustive"
# sfm.sourceCoordinateName" "arkit" or "arcore" or "opengl" or "opencv" or "ros"
# training.model: "splatfacto" or "splatfacto-big" or "splatfacto-w-light" or "splatfacto-mcmc" or "nerfacto"
# sphericalCamera.cubeFacesToRemove: "['back', 'down', 'front', 'left', 'right', 'up']" or "['']"
# sphericalCamera.optimizeSequentialFrameOrder: "true" or "false"
# segmentation.backgroundRemovalModel: "object" or "human"
# training.rotateSplat: 'true' or 'false' (rotate output splat for gradio viewer)

file_contents = {
    "uuid": str(unique_uuid),
    "instanceType": instance_type,
    "logVerbosity": "info",
    "s3": {
        "bucketName": s3_bucket_name,
        "inputPrefix": s3_input_prefix,
        "inputKey": media_filename,
        "outputPrefix": s3_output_prefix
    },
    "videoProcessing": {
        "maxNumImages": "300",
    },
    "imageProcessing": {
        "filterBlurryImages": "true"
    },
    "sfm": {
        "enable": "true",
        "softwareName": "glomap",
        "enableEnhancedFeatureExtraction": "false",
        "matchingMethod": "sequential",
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
        "model": "splatfacto",
        "enableMultiGpu": "false",
        "rotateSplat": "true"
    },
    "sphericalCamera": {
        "enable": "false",
        "cubeFacesToRemove": "['down', 'up']",
        "optimizeSequentialFrameOrder": "true"
    },
    "segmentation": {
        "removeBackground": "false",
        "backgroundRemovalModel": "u2net", #"u2net", "u2net-human","sam2"
        "maskThreshold": "0.6", #0.6 #0.38
        "removeHumanSubject": "false"
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
