## Submitting a job through the backend

### Job Schema Structure
The job metadata JSON file must follow this structure:

```json
{
    "uuid": "<unique-job-id>",
    "instanceType": "ml.g5.4xlarge",
    "useSpotInstance": "false",
    "logVerbosity": "info",
    "originalMediaFilename": "<optional-original-filename>",
    "s3": {
        "bucketName": "<bucket-name>",
        "inputPrefix": "media-input",
        "inputKey": "<media-filename>",
        "outputPrefix": "workflow-output"
    },
    "videoProcessing": {
        "maxNumImages": "300",
        "videoStartTime": "0",
        "videoStopTime": "None",
        "filterBlurryImages": true
    },
    "reconstruction": {
        "enable": true,
        "softwareName": "glomap",
        "enableEnhancedFeatureExtraction": false,
        "matchingMethod": "sequential",
        "enableFlHeuristic": false,
        "flHeuristicValue": "1.2",
        "enableFlMetric": false,
        "flMetricValue": "24",
        "posePriors": {
            "usePosePriorColmapModelFiles": false,
            "usePosePriorTransformJson": {
                "enable": false,
                "sourceCoordinateName": "arkit",
                "poseIsWorldToCam": true
            }
        }
    },
    "training": {
        "enable": true,
        "maxSteps": "15000",
        "model": "splatfacto",
        "3dIsp": "none",
        "preserveSceneScale": false,
        "enableDepthLoss": false
    },
    "postProcessing": {
        "cropOutputBounds": false,
        "cropMode": "environment",
        "cleanSplat": false,
        "enableSpz": true,
        "enableSog": true,
        "enableUsdz": true,
        "enableVideoExport": true,
        "generateCollision": false,
        "collisionSceneType": "outdoor",
        "collisionSeedPos": "0,0,0",
        "generateLod": false,
        "generateMesh": true,
        "plyCoords": "rhyu",
        "spzCoords": "rhyu",
        "sogCoords": "rhyu",
        "usdzCoords": "rhyu"
    },
    "sphericalCamera": {
        "enable": false,
        "cubeFacesToRemove": []
    },
    "segmentation": {
        "backgroundRemoval": {
            "enable": false,
            "model": "u2net",
            "maskThreshold": "0.6"
        },
        "objectRemoval": {
            "enable": false,
            "action": "erase",
            "objects": "['human']"
        }
    }
}
```

### Utility to create JSON job file
The metadata file can be created manually following the structure documented above or created automatically and submitted using `source/generate-splat.py`. Modify the script contents to output a valid metadata file before uploading your media to s3.

### Submit Media and Metadata
Using the AWS Management Console or AWS CLI, follow the instructions below:
1. Choose an S3 prefix `{inputPrefix}` in `{bucketName}` for your media files and create a folder `{bucketName}/{inputPrefix}`. The `{bucketName}` is obtained from both the Terraform/CDK console output and the `outputs.json` file within the `deployment/terraform` or `deployment/cdk` directory.
2. Upload a video (.mp4 or .mov) into `{bucketName}/{inputPrefix}`
3. Submit a unique UUID metadata json file
      - Use utility to create and submit job
        - Open `source/generate_splat.py` in your favorite text editor
        - Fill in the top section of the script ensuring you enter the bucket name from the deployment configuration output and media filename in it and save it.
        - Open a shell session and run the script from a machine that has AWS access to `PUT` into the `{bucketName}/{inputPrefix}` location.
        ```bash
        python3 generate_splat.py
        ```
        OR
    - Manually create and submit metadata file
        - Create a file locally `uuid.json`
        - Open the `uuid.json` file in your favorite editor
        - Copy the metadata json block above and change the parameters to suit your use-case
        - Save the metadata file locally
        - Using AWS credentials, log into the AWS Console in the account and region specified in CDK/Terraform
        - Navigate to `{bucketName}/{s3TriggerKey}` in the AWS S3 console. `{s3TriggerKey}` is obtained from the CDK/Terraform configuration
        - Upload the `uuid.json` file into s3 at the `{bucketName}/upload-workflow` location
    > *Note: each metadata file needs to have a unique UUID both on the filename and inside of the json file*
4. Upon completion, SNS will send an email to the address provided in CDK/Terraform. A splat and training data will be exported to `{bucketName}/outputPrefix`
> You can follow the progress in the AWS Console under `Step Functions->State Machine`

> **Note (container changes):** Any change to `source/container/` (including `main.py`, `utils.py`, or component scripts) requires rebuilding and pushing the container image to ECR before resubmitting jobs. The infrastructure stack does not need to be redeployed for container-only changes.

> **Note (Spot/Batch jobs):** When `useSpotInstance` is `true`, the Step Functions execution parks in the `SetDefaultJobDefinition` state until the container completes and sends a callback. This is expected behaviour — the state machine is not stalled. Progress can be monitored in the AWS Batch console or via CloudWatch Logs at `/aws/batch/job`.
If you do not have the storage bucket name noted down, it can be found in the Terraform outputs.json file or CDK outputs file depending on your deployment method.
