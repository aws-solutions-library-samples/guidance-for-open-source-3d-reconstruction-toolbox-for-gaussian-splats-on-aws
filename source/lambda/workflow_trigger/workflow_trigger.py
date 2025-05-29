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

""" A sample script to receive unique metadata file uploaded to S3 for gaussian splat creation
and start the workflow """

import os
import json
import boto3
from datetime import datetime
import urllib.parse
import traceback
from botocore.exceptions import ClientError

# initialize boto3 clients 
stepfunctions = boto3.client('stepfunctions')
ssm_client = boto3.client('ssm')
s3_client = boto3.client("s3")
dynamodb = boto3.resource('dynamodb')

def validate_config(config: dict):
    required_dict_props = {
        "uuid": None,
        "instanceType": None,
        "logVerbosity": None,
        "s3": {
            "bucketName": None,
            "inputPrefix": None,
            "inputKey": None,
            "outputPrefix": None
        },
        "videoProcessing": {
            "maxNumImages": None,
        },
        "imageProcessing": {
            "filterBlurryImages": None
        },
        "sfm": {
            "enable": None,
            "softwareName": None,
            "posePriors": {
                "usePosePriorColmapModelFiles": None,
                "usePosePriorTransformJson": {
                    "enable": None,
                    "sourceCoordinateName": None,
                    "poseIsWorldToCam": None,
                },
            },
            "enableEnhancedFeatureExtraction": None,
            "matchingMethod": None
        },
        "training": {
            "enable": None,
            "maxSteps": None,
            "model": None,
            "enableMultiGpu": None,
            "rotateSplat": None
        },
        "sphericalCamera": {
            "enable": None,
            "cubeFacesToRemove": None,
            "optimizeSequentialFrameOrder": None
        },
        "segmentation": {
            "removeBackground": None,
            "backgroundRemovalModel": None,
            "maskThreshold": None,
            "removeHumanSubject": None
        }
    }

    dict_keys = required_dict_props.keys()

    status_code = None
    body = None
    for key in dict_keys:
        if key not in config:
            raise RuntimeError(f"Required configuration property {prop} was not found.")
    print("Input validation passed...")

def lambda_handler(event, context):
    # Get the object from the S3 event and show its content type
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')

    # Get the State Machine ARN from the SSM Parameter Store
    param_name = os.environ['STATE_MACHINE_PARAM_NAME']
    response = ssm_client.get_parameter(Name=param_name, WithDecryption=True)
    state_machine_arn = response['Parameter']['Value']

    table_name = os.environ['DDB_TABLE_NAME']
    table = dynamodb.Table(table_name)
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    session = boto3.session.Session()
    region = session.region_name
    try:
        # Get the Job JSON content for the workflow
        s3_object_response = s3_client.get_object(Bucket=bucket_name, Key=key)
        file_content = s3_object_response["Body"].read().decode('utf-8')
        json_content = json.loads(file_content)
        print("JSON Object: " + str(json_content))
    except Exception as e:
        status = f"Error getting object {key} from bucket {bucket_name}. Make sure they exist and \
            your bucket is in the same region {region} as this function: {traceback.format_exc()}"
        raise Exception(status)

    # Ensure the required props were configured
    validate_config(json_content)

    if 'status_code' not in json_content:
        # Get the payload from the event
        try:
            uuid = json_content["uuid"]
            key = {
                'uuid': uuid,
            }
            # lookup item in DB using UUID
            try:
                result = table.get_item(Key=key)
            except ClientError as e:
                status = (310, f"Error trying to get the item using UUID. Error:  {e}")
                print(status)
                errorObj = {
                    "statusCode": status[0],
                    "statusMsg": status[1],
                    "errorMsg": str(e)
                }
                raise SystemError(json.dumps(errorObj))

            if 'Item' in result:
                # update uuid status if found
                update_expression = 'SET uuidStatus = :uuidStatus'
                expression_attribute_values = {':uuidStatus': 'In-Progress'}
                try:
                    update_result = table.update_item(
                        Key=key,
                        UpdateExpression=update_expression,
                        ExpressionAttributeValues=expression_attribute_values,
                        ReturnValues='ALL_NEW'
                    )
                    print(f"Updated UUID: {update_result['Attributes']}")
                except ClientError as e:
                    status = (312, f"Error trying to update the item with key, \
                            update expression and update values. Error: {e}")
                    print(status)
                    errorObj = {
                        "statusCode": status[0],
                        "statusMsg": status[1],
                        "errorMsg": str(e)
                    }
                    raise SystemError(json.dumps(errorObj))
            else:
                # entry doesn't exist, create new entry
                current_date = datetime.now()
                datestamp = str(current_date.isoformat())

                item = {
                    'uuid':  json_content["uuid"],
                    'status': 'In-Progress',
                    'startTimestamp': datestamp,
                    'instanceType': str(json_content['instanceType']),
                    "logVerbosity": str(json_content['logVerbosity']),
                    "s3Input": f"s3://{json_content["s3"]["bucketName"]}/{json_content["s3"]["inputPrefix"]}/{json_content["s3"]["inputKey"]}",
                    "s3Output": f"s3://{json_content["s3"]["bucketName"]}/{json_content["s3"]["outputPrefix"]}",
                    "filename": str(json_content["s3"]["inputKey"]),
                    "maxNumImages": str(json_content["videoProcessing"]["maxNumImages"]),
                    "filterBlurryImages": str(json_content["imageProcessing"]["filterBlurryImages"]),
                    "runSfm": str(json_content["sfm"]["enable"]),
                    "sfmSoftwareName": str(json_content["sfm"]["softwareName"]),
                    "usePosePriorColmapModelFiles": str(json_content["sfm"]["posePriors"]["usePosePriorColmapModelFiles"]),
                    "usePosePriorTransformJson": str(json_content["sfm"]["posePriors"]["usePosePriorTransformJson"]["enable"]),
                    "sourceCoordinateName": str(json_content["sfm"]["posePriors"]["usePosePriorTransformJson"]["sourceCoordinateName"]),
                    "poseIsWorldToCam": str(json_content["sfm"]["posePriors"]["usePosePriorTransformJson"]["poseIsWorldToCam"]),
                    "enableEnhancedFeatureExtraction": str(json_content["sfm"]["enableEnhancedFeatureExtraction"]),
                    "matchingMethod": str(json_content["sfm"]["matchingMethod"]),
                    "runTrain": str(json_content["training"]["enable"]),
                    "model": str(json_content["training"]["model"]),
                    "maxSteps": str(json_content["training"]["maxSteps"]),
                    "enableMultiGpu": str(json_content["training"]["enableMultiGpu"]),
                    "rotateSplat": str(json_content["training"]["rotateSplat"]),
                    "sphericalCamera": str(json_content["sphericalCamera"]["enable"]),
                    "sphericalCubeFacesToRemove": str(json_content["sphericalCamera"]["cubeFacesToRemove"]),
                    "optimizeSequentialSphericalFrameOrder": str(json_content["sphericalCamera"]["optimizeSequentialFrameOrder"]),
                    "removeBackground": str(json_content["segmentation"]["removeBackground"]),
                    "backgroundRemovalModel": str(json_content["segmentation"]["backgroundRemovalModel"]),
                    "maskThreshold": str(json_content["segmentation"]["maskThreshold"]),
                    "removeHumanSubject": str(json_content["segmentation"]["removeHumanSubject"])
                }

                try:
                    table.put_item(Item=item)
                    print(f"Created new record: {item}")
                except ClientError as e:
                    status = (311, f"Error trying to add new value into the the DB. Error: {e}")
                    print(status)
                    errorObj = {
                        "statusCode": status[0],
                        "statusMsg": status[1],
                        "errorMsg": str(e)
                    }
                    raise SystemError(json.dumps(errorObj))

            inputObj = {
                "stateMachine" : {
                    "timeout": 14400,
                    "instanceCount": 1,
                    "volumeSizeInGB": 25,
                    "ecrImageArn": os.environ["ECR_IMAGE_URI"],
                    "containerEntryPoint": ["python"],
                    "containerArgs": ["/opt/ml/code/main.py"],
                    "containerRoleArn": f"arn:aws:iam::{account_id}:role/{os.environ["CONTAINER_ROLE_NAME"]}",
                    "completeLambdaName": os.environ['LAMBDA_COMPLETE_NAME'],
                    "startTimestamp": datestamp
                },
                "envVars": {
                    "UUID": str(uuid),
                    "DATA_CHANNEL": "train",
                    "MODEL_INPUT": f"s3://{json_content["s3"]["bucketName"]}/models/models.tar.gz",
                    "S3_INPUT": f"s3://{json_content["s3"]["bucketName"]}/{json_content["s3"]["inputPrefix"]}/{json_content["s3"]["inputKey"]}",
                    "S3_OUTPUT": f"s3://{json_content["s3"]["bucketName"]}/{json_content["s3"]["outputPrefix"]}",
                    "FILENAME": str(json_content["s3"]["inputKey"]),
                    "INSTANCE_TYPE": str(json_content["instanceType"]),
                    "LOG_VERBOSITY": str(json_content["logVerbosity"]),
                    "MAX_NUM_IMAGES": str(json_content["videoProcessing"]["maxNumImages"]),
                    "FILTER_BLURRY_IMAGES": str(json_content["imageProcessing"]["filterBlurryImages"]),
                    "RUN_SFM": str(json_content["sfm"]["enable"]),
                    "SFM_SOFTWARE_NAME": str(json_content["sfm"]["softwareName"]),
                    "USE_POSE_PRIOR_COLMAP_MODEL_FILES": str(json_content["sfm"]["posePriors"]["usePosePriorColmapModelFiles"]),
                    "USE_POSE_PRIOR_TRANSFORM_JSON": str(json_content["sfm"]["posePriors"]["usePosePriorTransformJson"]["enable"]),
                    "SOURCE_COORD_NAME": str(json_content["sfm"]["posePriors"]["usePosePriorTransformJson"]["sourceCoordinateName"]),
                    "POSE_IS_WORLD_TO_CAM": str(json_content["sfm"]["posePriors"]["usePosePriorTransformJson"]["poseIsWorldToCam"]),
                    "ENABLE_ENHANCED_FEATURE_EXTRACTION": str(json_content["sfm"]["enableEnhancedFeatureExtraction"]),
                    "MATCHING_METHOD": str(json_content["sfm"]["matchingMethod"]),
                    "RUN_TRAIN": str(json_content["training"]["enable"]),
                    "MODEL": str(json_content["training"]["model"]),
                    "MAX_STEPS": str(json_content["training"]["maxSteps"]),
                    "ENABLE_MULTI_GPU": str(json_content["training"]["enableMultiGpu"]),
                    "ROTATE_SPLAT": str(json_content["training"]["rotateSplat"]),
                    "SPHERICAL_CAMERA": str(json_content["sphericalCamera"]["enable"]),
                    "SPHERICAL_CUBE_FACES_TO_REMOVE": str(json_content["sphericalCamera"]["cubeFacesToRemove"]),
                    "OPTIMIZE_SEQUENTIAL_SPHERICAL_FRAME_ORDER": str(json_content["sphericalCamera"]["optimizeSequentialFrameOrder"]),
                    "REMOVE_BACKGROUND": str(json_content["segmentation"]["removeBackground"]),
                    "BACKGROUND_REMOVAL_MODEL": str(json_content["segmentation"]["backgroundRemovalModel"]),
                    "MASK_THRESHOLD": str(json_content["segmentation"]["maskThreshold"]),
                    "REMOVE_HUMAN_SUBJECT": str(json_content["segmentation"]["removeHumanSubject"])
                },
                "sns": {
                    "topicArn": os.environ["SNS_TOPIC_ARN"],
                    "message": f"Your 3D reconstruction from file {str(json_content["s3"]["inputKey"])} is ready to view at \
                        {f"s3://{json_content["s3"]["bucketName"]}/{json_content["s3"]["outputPrefix"]}"}/{str(json_content["uuid"])}"
                }
            }
            print(f"Input Object is: {inputObj}")
        except Exception as e:
            status = (f"Error getting payload: {traceback.format_exc()}")
            raise Exception(status)

        # Start the step function workflow
        try:
            print("Starting Step Function Workflow")
            inputObj['stateMachine']['statusCode'] = 200
            # Send the message to start the State Machine
            response = stepfunctions.start_execution(
                stateMachineArn=state_machine_arn,
                name=uuid,
                input=json.dumps(inputObj))
        except Exception as e:
            status = f"Error starting the step function workflow: {traceback.format_exc()}"
            raise Exception(status)


        return {
            'statusCode': 200,
            'body': json.dumps(inputObj)
        }
    else:
        inputObj = {}
        inputObj['stateMachine']['statusCode'] = response['status_code']
        inputObj['stateMachine']['body'] = response['body']
        inputObj['sns']['topicArn'] = os.environ["SNS_TOPIC_ARN"]
        inputObj['sns']['message'] = f"There was an error in the input configuration: {str(response['body'])}"
        return {
            'statusCode': response['status_code'],
            'body': json.dumps(inputObj)
        }
