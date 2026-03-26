# Guidance for Open Source 3D Reconstruction Toolbox for Gaussian Splats on AWS

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Cost](#cost)
5. [Security](#security)
6. [Deployment and User Guide](#deployment-and-user-guide)
7. [Next Steps](#next-steps)
8. [License](#license)
9. [Authors](#authors)

## Overview

The Open Source 3D Reconstruction Toolbox for Gaussian Splats provides an end-to-end, pipeline-based guidance on AWS to reconstruct 3D scenes or objects from images or video inputs. The infrastructure can be deployed via AWS Cloud Development Kit (CDK) or Terraform leveraging infrastructure-as-code.

Once deployed, the guidance features a full 3D reconstruction back-end system with the following customizable components or pipelines:

1. **Media Ingestion**: Process videos or collections of images as input
2. **Image Processing**: Automatic filtering, enhancement, and preparation of source imagery (e.g. background removal)
3. **Structure from Motion (SfM)**: Camera pose estimation and initial 3D point cloud generation
4. **Gaussian Splat Training**: Optimization of 3D Gaussian primitives to represent the scene using AI/ML
5. **Export & Delivery**: Generation of the final 3D asset in standard formats for easy viewing and notification via email

By deploying this guidance, users gain access to a flexible infrastructure that handles the entire 3D reconstruction process programmatically, from media upload to final 3D model delivery, while being highly modular through its componentized pipeline-based approach. This guidance addresses the significant challenges organizations face when trying to create photorealistic 3D content - traditionally a time-consuming, expensive, and technically complex process requiring specialized skills and equipment.

## Architecture

This guidance will:

- create the infrastructure required to create a Gaussian splat from a video or set of images
- create the mechanism to run the code and perform 3D reconstruction
- enable a user to create a 3D Gaussian splat using open source tools and AWS by uploading a video (.mp4 or .mov) or images (.png or .jpg) and metadata (.json) into S3
- provide a 3D viewer for viewing the photo-realistic effects and performant nature of Gaussian splats

### Architecture Diagram

<div align="center">
<img src="assets/images/gs-workflow-arch.PNG" width=70%> 
<br/>
<i>Figure 1: 3D Reconstruction Toolbox for Gaussian Splats on AWS Reference Architecture</i>
</div>

### Architecture Steps
1. User authenticates to [AWS Identity and Access Management (IAM)](https://aws.amazon.com/iam/) via AWS Tools and SDKs.
2. The input is uploaded to a dedicated [Amazon Simple Storage Service (S3)](https://aws.amazon.com/s3/)  job bucket location. This can be done using a Gradio interface and AWS Software Development Kit (SDK).
3. Optionally, the solution supports external job submission by uploading a ‘.JSON’ job configuration file and media into a designated S3 job bucket location. 
4. The job JSON file uploaded to the S3 job bucket will trigger an [Amazon Simple Notification Service (SNS)](https://aws.amazon.com/sns/) message that will invoke an initialization [AWS Lambda](https://aws.amazon.com/lambda/) function.
5. The job trigger **AWS Lambda** function will perform input validation and set appropriate variables for the [AWS Step Function State Machine](https://aws.amazon.com/step-functions/).
6. The workflow job record will be created in [Amazon DynamoDB](https://aws.amazon.com/dynamodb/) job table.
7. The job trigger **AWS Lambda** function will invoke an **AWS Step Functions State Machine** to handle the entire workflow job.
8. An [Amazon SageMaker](https://aws.amazon.com/sagemaker/) Training or [AWS Batch](https://aws.amazon.com/batch/) Job will be submitted synchronously using the state machine built-in wait until completion mechanism. 
9. The [Amazon Elastic Container Registry (ECR)](https://aws.amazon.com/ecr/) container image and S3 job bucket model artifacts will be used to spin up a new Graphics Processing Unit (GPU) container. The compute node instance type is determined by the job JSON configuration.
10. The GPU container will run the entire pipeline as an **Amazon SageMaker** training or **AWS Batch** job.
11. The job completion **AWS Lambda** function will complete the workflow job by updating the job metadata in **Amazon DynamoDB** and notifying the user via email upon completion using **Amazon SNS**.
12. Internal workflow parameters are stored in [AWS System Manager Parameter Store](https://docs.aws.amazon.com/systems-manager/latest/userguide/systems-manager-parameter-store.html) during guidance deployment to decouple the job trigger **AWS Lambda** function and the **AWS Step Function State Machine**.
13. [Amazon CloudWatch](https://aws.amazon.com/cloudwatch/) is used to monitor the training logs, surfacing errors to the user.

### AWS Services in this Guidance

| **AWS Service**                                                                    | Role |                                                                                                  |
| -----------------------------------------------------------------------------------| ---- | -------------------------------------------------------------------------------------------------|
| [Amazon Simple Storage Service (S3)](https://aws.amazon.com/pm/serv-s3/)           | Core | Host training models, job configurations, media, and generated assets                            |
| [AWS Lambda](https://aws.amazon.com/lambda/)                                       | Core | Run custom code to process requests                                                              |
| [Amazon Simple Notification Service (SNS)](https://aws.amazon.com/sns/)            | Core | Send completion status via notification to email                                                 |
| [AWS Step Functions](https://aws.amazon.com/step-functions/)                       | Core | Orchestrate the 3D reconstruction workflow                                                       |
| [Amazon DynamoDB](https://aws.amazon.com/dynamodb/)                                | Core | Store training job details and attributes                                                        |
| [Amazon SageMaker](https://aws.amazon.com/sagemaker/)                              | Core | Run 3D reconstruction pipeline processing on container using On-Demand instances                                          |
| [AWS Batch](https://aws.amazon.com/batch/)                              | Core | Run 3D reconstruction pipeline processing on container using Spot instances                                          |
| [Amazon Elastic Container Service](https://aws.amazon.com/ecs/)                    | Core | Orchestrate 3D reconstruction pipeline processing on containers                                       |
| [Amazon Elastic Container Registry](https://aws.amazon.com/ecr/)                   | Core | Image store for the custom created container                                                     |
| [Amazon CloudWatch](https://aws.amazon.com/cloudwatch/)                            | Core | Monitor logs and surface errors to SNS                                                           |
| [AWS Identity and Access Management](https://aws.amazon.com/iam/)                  | Core | Security access controls to run the workflow securely                                            |
| [AWS Cloud Development Kit](https://aws.amazon.com/cdk/)                           | Core | Cloud infrastructure as code for easy deployment                                                 |
| [Amazon Systems Manager Parameter Store](https://aws.amazon.com/systems-manager/)  | Core | Securely store infrastructure resource ids in Parameter Store to aid in deployment and execution |

### Custom GS Pipeline Container

In this project, there is only one Docker container that contains all of the 3D reconstruction tools for Gaussian Splatting. This container has a `Dockerfile`, `main.py`, and helper script files and open source libraries under the `source/container` directory. The main script processes each request from the SageMaker or Batch Training Job using ECS invoke message and saves the result to S3 upon successful completion.


Current features and open source libraries include:

| Category            | Component                                                    | Software                                     | Can Use GPU?      | Notes                                                                                                        |
| ------------------- | ------------------------------------------------------------ | -------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------ |
| Video Processing | Video to Images                                              | [OpenCV](https://github.com/opencv/opencv) [(Apache-2.0)](https://github.com/opencv/opencv?tab=Apache-2.0-1-ov-file#readme) | No | Extract frames (images) from video. Supports .mov/.mp4, perspective/equirectangular. Can set start/stop time. |
| Video Processing | Filter Blurry Images from Video                              | [Sharp Frame Extractor](https://github.com/cansik/sharp-frame-extractor) [(MIT)](https://github.com/cansik/sharp-frame-extractor?tab=MIT-1-ov-file) | No | Remove blurry images from a correlated set of frames in a video.                                      |
| Spherical Images | Equirectangular to Perspective                               | [OpenCV](https://github.com/opencv/opencv) [(Apache-2.0)](https://github.com/opencv/opencv?tab=Apache-2.0-1-ov-file#readme) | No | Convert spherical images into perspective images for processing. Remove cube map faces from the dataset.        |
| Segmentation     | Erase Objects                                                | [rembg](https://github.com/danielgatis/rembg/) [(MIT)](https://github.com/danielgatis/rembg/#MIT-1-ov-file), [Attentive-Eraser](https://github.com/Anonym0u3/AttentiveEraser) [(Apache-2.0)](https://github.com/Anonym0u3/AttentiveEraser#Apache-2.0-1-ov-file) | Yes | Use u2net to classify objects and erase them using diffusion/in-painting. Supports "human".                  |
| Segmentation     | Remove Objects                                               | [rembg](https://github.com/danielgatis/rembg/) [(MIT)](https://github.com/danielgatis/rembg/#MIT-1-ov-file) | Yes | Use u2net to classify objects and remove/mask them. Supports "human".                                        |
| Segmentation     | Remove Background                                            | [backgroundremover](https://github.com/nadermx/backgroundremover) [(MIT)](https://github.com/nadermx/backgroundremover?tab=MIT-1-ov-file#readme), [sam2](https://github.com/facebookresearch/sam2) [(Apache-2.0/BSD-3-Clause)](https://github.com/facebookresearch/sam2/blob/main/LICENSE) | Yes |Use u2net or SAM2 to detect objects (rigid body) and remove background. SAM2 only supports video.            |
| Reconstruction   | Images to Point Cloud/Poses - Incremental SfM                | [Colmap](https://github.com/colmap/colmap) [(BSD)](https://github.com/colmap/colmap?tab=License-1-ov-file#readme) | Yes |Supports Colmap. input video/images only or images + pose-priors.                                            |
| Reconstruction   | Images to Point Cloud/Poses - Global SfM                     | [Glomap](https://github.com/colmap/glomap) [(BSD-3-Clause)](https://github.com/colmap/glomap?tab=BSD-3-Clause-1-ov-file#readme) | Yes |Supports Glomap. input video/images only or images + pose-priors.                                            |
| Reconstruction   | Images to Point Cloud/Poses - Transformer                     | [MapAnything](https://github.com/facebookresearch/map-anything) [(Apache-2.0)](https://github.com/facebookresearch/map-anything#Apache-2.0-1-ov-file) | Yes | Supports MapAnything with GPU (limited to < 50 images)                                                 |
| Training         | Images, Point Cloud, & Poses to Gaussian Splat               | [NerfStudio](https://github.com/nerfstudio-project/nerfstudio) [(Apache-2.0)](https://github.com/nerfstudio-project/nerfstudio/tree/main?tab=Apache-2.0-1-ov-file#readme), [gsplat](https://github.com/nerfstudio-project/gsplat) [(Apache-2.0)](https://github.com/nerfstudio-project/gsplat?tab=Apache-2.0-1-ov-file#readme), [splatfacto-w](https://github.com/KevinXu02/splatfacto-w) [(Apache-2.0)](https://github.com/KevinXu02/splatfacto-w?tab=Apache-2.0-1-ov-file#readme), [3DGRUT](https://github.com/nv-tlabs/3dgrut) [(Apache-2.0)](https://github.com/nv-tlabs/3dgrut#Apache-2.0-1-ov-file) | Yes | Supports splatfacto, splatfacto-big, splatfacto-mcmc, splatfacto-w-light, 3dgrt, 3dgut, nerfacto              |
| Training         | 3D Image Signal Processing - Bilateral Grid, PPISP              | [PPISP](https://github.com/nv-tlabs/ppisp) [(Apache-2.0)](https://github.com/nv-tlabs/ppisp?tab=Apache-2.0-1-ov-file), [Bilateral Grid](https://github.com/yuehaowang/bilarf) [(Apache-2.0)](https://github.com/yuehaowang/bilarf?tab=Apache-2.0-1-ov-file) | Yes | Supports Bilateral Grid and PPISP            |
| Post Processing  | Export Gaussian Splat and Metadata                           | [Splat-Transform](https://github.com/playcanvas/splat-transform) [(MIT)](https://github.com/playcanvas/splat-transform#MIT-1-ov-file), [spz](https://github.com/nianticlabs/spz/tree/main) [(MIT)](https://github.com/nianticlabs/spz/tree/main#MIT-1-ov-file) | Yes | Supports outputs .ply, .spz, .sog, .usdz (beta), .mp4(splat render), .jpg(thumbnail), metrics(psnr, lpips, ssim)                                                                               |
| Post Processing  | Transform/Rotate all generated objects from native to viewer | [Splat-Transform](https://github.com/playcanvas/splat-transform) [(MIT)](https://github.com/playcanvas/splat-transform#MIT-1-ov-file) | Yes | Supports Gradio interface coordinate system             |
| Post Processing  | Crop all generated objects                                   | | Yes | Supports environments or rigid_objects to reduce noise.             |
| User Interface   | Submit jobs, view results                                    | [Gradio](https://github.com/gradio-app/gradio) [(Apache-2.0)](https://github.com/gradio-app/gradio#Apache-2.0-1-ov-file) | No |
| Web Viewer       | View 3D objects                        | [SuperSplat](https://github.com/playcanvas/supersplat) [(MIT)](https://github.com/playcanvas/supersplat?tab=MIT-1-ov-file#readme), [Babylon.js](https://github.com/BabylonJS/Babylon.js) [(Apache-2.0)](https://github.com/BabylonJS/Babylon.js#Apache-2.0-1-ov-file) | Yes |


>*Note: A Gaussian splat job can be submitted either through a [user interface](source/Gradio/Readme.md) or [back-end](source/readme.md) depending on your preference. Full deployment details can be found in the [Implementation Guide](https://aws-solutions-library-samples.github.io/compute/open-source-3d-reconstruction-toolbox-for-gaussian-splats-on-aws.html).*

## Prerequisites

### Third-party tools

- Git
- Docker
- Terraform (if choosing not to deploy infrastructure using CDK)

### AWS account requirements

An active AWS Account with IAM user or role with elevated permissions to deploy resources is required to deploy this guidance, along with either a local computer with appropriate AWS credentials to deploy the CDK or Terraform solution, or utilize an AWS EC2 workstation to build and deploy the CDK or Terraform solution. Please refer to the [Implementation Guide](https://aws-solutions-library-samples.github.io/compute/open-source-3d-reconstruction-toolbox-for-gaussian-splats-on-aws.html) for detailed instructions for deployment and running the guidance.

- EC2 (if choosing not to deploy infrastructure from your local computer)
- IAM roles with permissions
- CloudFormation
- ECR Image
- S3 Buckets
- DynamoDB Table
- Lambda Functions
- SageMaker Training Jobs
- Batch Jobs
- Step Functions State Machine
- CDK (Please refer to the [Implementation Guide](https://aws-solutions-library-samples.github.io/compute/open-source-3d-reconstruction-toolbox-for-gaussian-splats-on-aws.html) for detailed instructions for deployment and running the guidance.)

### Service limits

- [Service quotas](https://docs.aws.amazon.com/servicequotas/latest/userguide/intro.html) - increases can be requested via the AWS Management Console, AWS CLI, or AWS SDKs (see [Accessing Service Quotas](https://docs.aws.amazon.com/servicequotas/latest/userguide/intro.html#access))

- This solution runs SageMaker Training Jobs or Batch Jobs which uses a Docker container to run the training. This deployment guide walks through building a custom container image for SageMaker or Batch.
  - Depending on what instances you will be using to train on (configured during job submission, ml.g5.4xlarge is the default), you may need to adjust the SageMaker Training Jobs or Batch Jobs quota. This will be under the SageMaker service in Service Quotas named "training job usage" or AWS Batch Job "instance usage".
  - (Optional) You can optionally build and test this container locally (not running on AWS Services) on a GPU-enabled EC2 instance. If you plan to do this, increase the EC2 quota named "Running On-Demand G and VT instances" and/or "Running On-Demand P instances", depending on the instance family you plan to use, to a desired maximum number of vCPUs for running instances of the target family. Note, this is vCPUs NOT number of instances like the SageMaker Training Jobs quota.

## Cost

_You are responsible for the cost of the AWS services used while running this Guidance. As of May 2025, the cost for running this Guidance with the default settings in the default AWS Region (US East 1(N. Virginia)) is approximately $278.33 per month for processing 100 requests._

_We recommend creating a [Budget](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html) through [AWS Cost Explorer](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/) to help manage costs. Prices are subject to change. For full details, refer to the pricing webpage for each AWS service used in this Guidance._

### Cost Table

The following table provides a sample cost breakdown for deploying this Guidance with the default parameters in the US East (N. Virginia) Region for one month.

**Spot Instance (Batch/ECS)**
| AWS Service        | Dimensions                                                                       | Cost [USD]        |
| ------------------ | -------------------------------------------------------------------------------- | ----------------- |
| Amazon S3          | Standard feature storage (input=200MB, output=2.5GB)                             | $1.61/month       |
| Amazon S3          | Data transfer feature                                                            | $0.90/month       |
| Amazon DynamoDB    | Job table storage, 0.5MB per month, 1GB total, avg item size=825bytes            | $0.81/month       |
| AWS Lambda         | 2 invocations per job, 1.25s, 7.1s = 8.5s                                        | $0.01/month       |
| AWS Step Functions | State transitions per workflow = 5                                               | $0.01/month       |
| Amazon ECR         | Data storage, 15GB                                                               | $1.47/month       |
| Amazon SNS         | Email notifications, 1 per request                                               | $0.01/month       |
| Parameter Store    | Store 1 param                                                                    | $0.01/month       |
| Amazon CloudWatch  | Metrics, 1GB                                                                     | $0.50/month       |
| AWS CodeBuild      | Optional, $0.005 per minute, 100min free, only need build once, takes ~60mn      | -                 |
| Amazon ECS/Batch   | num_instance=1, num_hours_per_job=1, ml.g5.4xlarge, Volume_size_in_GB_per_job=15 | $68.00/month      |
| **TOTAL**          | (est. 100 requests)                                                              | **$73.33/month**  |

> *Note: Amazon SageMaker can be used instead of AWS Batch for on-demand processing, bypassing any wait queues for ~$142.33/month total*

## Security

When you build systems on AWS infrastructure, security responsibilities are shared between you and AWS. This [shared responsibility
model](https://aws.amazon.com/compliance/shared-responsibility-model/) reduces your operational burden because AWS operates, manages, and
controls the components including the host operating system, the virtualization layer, and the physical security of the facilities in
which the services operate. For more information about AWS security, visit [AWS Cloud Security](http://aws.amazon.com/security/).

- All data is encrypted at rest and at transit within the AWS Cloud services in this Guidance
- An Amazon S3 access logging bucket logs all access to the asset bucket
- Input validation on the job configuration will flag any misconfigurations in the json file
- Least priviledge access rights on service actions

**Considerations**

At the time of publishing (Mar 2026), the codebase was scanned using [Semgrep](https://semgrep.dev/), [Bandit](https://github.com/PyCQA/bandit), [Checkov](https://www.checkov.io/), [Gitleaks](https://github.com/gitleaks/gitleaks), [Grype](https://github.com/anchore/grype), and [Codeguru](https://aws.amazon.com/codeguru/) code security scanning tools. The following table outlines all security issues flagged as ERROR or CRITICAL with explanations.
| Level   | Classification  | Source       | Rule ID                             | Cause                                                                                                  | Explanation                                                                                                                                                              |
| ------- | --------------- | ------------ | ----------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Error   | False Positive  | Bandit       | B202 tarfile.extractall             | tarfile.extractall used without any validation. Please check and discard dangerous members             | The tarfile extraction in `utils.py` validates every member before extraction, explicitly rejecting absolute paths, `..` sequences, and symlinks pointing outside the extraction directory. Only safe members are passed to `extractall`. |
| Error   | False Positive  | Semgrep      | 590 dangerous-subprocess-use-audit  | Detected subprocess function 'run' without a static string. If this data can be controlled by a malicious actor, it may be an instance of command injection | The subprocess call is already validated - it uses a list of arguments (preventing shell injection) and all parameters are validated before use, making it safe from command injection attacks. |
| Error   | False Positive  | Semgrep      | 98 sqlalchemy-execute-raw-query     | Avoiding SQL string concatenation: untrusted input concatenated with raw SQL query can result in SQL Injection | The query is already validated with proper table name escaping, making it safe from SQL injection attacks.                                                                                                                               |
| Error   | False Positive  | Semgrep      | 93 sqlalchemy-execute-raw-query     | Avoiding SQL string concatenation: untrusted input concatenated with raw SQL query can result in SQL Injection | The query is already validated with proper table name escaping and validation, making it safe from SQL injection attacks.                                                                                                                   |
| Error   | False Positive  | Gitleaks     | generic-api-key — terraform/main.tf:54 | API Key found | This is not an API key but a random prefix string used for project component naming. No credentials are present. |
| Error   | False Positive  | Codeguru     | NoSQL Injection — workflow_complete.py:591,603,615 | Untrusted input used in DynamoDB operation without sanitization | The helper functions `put_ddb_item`, `get_ddb_item_value`, and `update_ddb_item_value` receive a key dict that is already validated via `_validate_uuid()` at the only call site in `lambda_handler`. Codeguru does not trace the validation through the call chain. |
| Error   | False Positive  | Codeguru     | XSS — workflow_complete.py:768,965 | `client.publish` using unsanitized user input | The SNS `Subject` field is sanitized with `_sanitize_text()`. The `Message` body field is plain text delivered via email/SMS and is not rendered as HTML in a browser context. The `FILENAME` value in the message body was additionally sanitized as a precaution. |
| Error   | False Positive  | Codeguru     | NoSQL Injection — generate_splat_gradio.py:1699,2276,3169,3556 | Untrusted input used in DynamoDB operation without sanitization | All four call sites pass `_validate_uuid(job_id)` as the DynamoDB key, which enforces strict UUID format validation before any database operation. Codeguru does not trace the inline validation wrapper. |
| Error   | False Positive  | Codeguru     | XSS — generate_splat_gradio.py:532,680 | Untrusted data incorporated into web page content without encoding | These return plain strings to `gr.Textbox` components. Gradio textboxes render output as escaped plain text, not as HTML, so no XSS risk exists at these output points. |
| Error   | False Positive  | Codeguru     | XSS — generate_splat_gradio.py:142 | Untrusted data incorporated into web page content | This is a `gr.Textbox` output returning a plain string. Gradio renders textbox content as escaped plain text, not HTML. No XSS risk exists. |
| Error   | False Positive  | Codeguru     | NoSQL Injection — refine_splat.py:44 | Untrusted input used in DynamoDB operation without sanitization | UUID format validation via `_UUID_RE.match()` is performed immediately before the `get_item` call. Codeguru does not trace the guard through the conditional block. |
| Error   | False Positive  | Bandit/Semgrep | B602 subprocess-shell-true — cdk.out/utils.py | subprocess call with shell=True identified | The `cdk.out/` directory contains generated CDK build artifacts, not source files. The source file `source/container/src/pipeline/utils.py` was fixed (shell=True removed). The artifact copies will be regenerated with the fix on the next `cdk synth` / `cdk deploy`. |
| Error   | False Positive  | Semgrep      | dangerous-subprocess-use-audit / dangerous-subprocess-use-tainted-env-args | subprocess 'run' without static string or with user-controlled data | These subprocess calls use a list of arguments (not a shell string), which prevents shell injection. All parameters are validated before use. This matches the existing Semgrep 590 false positive pattern already documented above. |
| Error   | False Positive  | Codeguru     | OS command injection — generate_splat_gradio.py:120,121,122 | Untrusted input used in system commands | These lines set `os.environ` variables for AWS credentials provided by the user in the Gradio credentials tab. No subprocess or shell command is invoked with these values; they are passed to `boto3.Session()` which handles them safely. |
| Error   | False Positive  | Codeguru     | Path traversal — generate_splat_gradio.py:591,1363,1368,1804,1815,1825,1911,1927,3814,3831 | User-controlled input in file paths | These paths construct S3 key strings or presigned URL parameters, not local filesystem paths. S3 key construction does not carry path traversal risk. Lines 1363/1368 use `os.path.basename()` and `os.path.realpath()` guards. |
| Error   | False Positive  | Codeguru     | Path traversal — container pipeline files (utils.py, colmap.py, remove_background.py, remove_background_sam2.py, erase_object_using_mask.py, filter_blurry_images.py, simple_video_to_images.py, sharp_video_to_images.py, rotate_portrait_images.py, images_to_video.py, extract_poses_imgs.py, update_camera_model.py, run_map_anything.py, demo_colmap_mapanything.py, demo_colmap_mapanything_.py, pipeline.py, and others) | User-controlled input in file paths across multiple pipeline modules | All file paths in the container pipeline originate from the job configuration, which is validated via `validate_input_media()` and `os.path.realpath()` checks in `main.py` before being passed to any pipeline module. The scanner does not trace validation through the call chain from `main.py` into the individual pipeline modules. |
| Error   | False Positive  | Codeguru     | Path traversal — model_deployment.py:105,127,136,141,142,150,159,164,165,282 | User-controlled input in file paths | All paths in `model_deployment.py` are constructed from `tempfile.mkdtemp()` (a system-generated secure temp directory) or from hardcoded model filenames. No user-supplied input is used to construct local file paths. |
| Error   | False Positive  | Codeguru     | Path traversal — index.py:104,178 | User-controlled input in file paths | Paths in `index.py` are constructed from S3 download targets using hardcoded filenames and a system temp directory. No user-supplied path components are used. |
| Error   | False Positive  | Codeguru     | Path traversal — post_deploy_stack.py:65,70, container_deployment.py:55, container_deployment_codebuild.py:52, lambdas.py:101 | User-controlled input in file paths | These CDK construct files use `os.path.join` with `__file__` (the script's own path) and static string literals to locate deployment assets. No user-supplied input influences these paths. |
| Error   | False Positive  | Codeguru     | Path traversal — patch_splatfactow.py:13,52 | User-controlled input in file paths | `patch_splatfactow.py` operates on hardcoded nerfstudio installation paths inside the container. No user-supplied input influences these paths. |
| Error   | False Positive  | Codeguru     | Path traversal — submit_test_jobs.py:17,22 | User-controlled input in file paths | `submit_test_jobs.py` is a developer test utility that reads local test fixture files using hardcoded relative paths. It is not part of the deployed guidance and is not exposed to user input. |
| Error   | False Positive  | Codeguru     | NoSQL Injection — utils.py:983,1027,1060,1070,1091,1092,1093,1102 | Untrusted input used in DynamoDB operation without sanitization | The `update_dynamodb_metrics` and `update_component_phase_completion` functions receive `uuid` from the job configuration, which is validated via `_validate_uuid()` in `main.py` before any pipeline function is called. Codeguru does not trace the validation through the call chain. |
| Error   | False Positive  | Codeguru     | NoSQL Injection — clean_point_cloud.py:81,112,127, clean_ply.py:44, remove_ply_comments.py:44 | Untrusted input used in NoSQL database operation without sanitization | None of these files contain any DynamoDB or NoSQL operations. The scanner incorrectly flagged `tree.query()` (a scipy KD-tree spatial query), `content.find()` (a bytes search), and PLY file parsing as NoSQL database calls. These are pure in-memory data processing operations with no database interaction. |
| Error   | False Positive  | Codeguru     | OS command injection — pipeline.py:97,103 | Untrusted input used in system commands | `pipeline.py` uses `subprocess.run()` with a validated list of arguments and explicit checks for dangerous characters before execution. No `shell=True` is used. Codeguru does not recognise the inline argument validation. |
| Error   | False Positive  | Codeguru     | OS command injection — filter_blurry_images.py:971 | Untrusted input used in system commands | The flagged line is inside the `main()` argument-parsing block. The only subprocess call in this file already uses `sys.executable` with a list of arguments and no `shell=True`, preventing shell injection. Codeguru incorrectly attributed the finding to the argument-parsing region. |
| Error   | False Positive  | Codeguru     | XSS — utils.py:103,128,130,132 | Untrusted data incorporated into web page content | These lines are `print()` statements and f-strings inside `rotate_single_image()` used for console logging only. No web framework or HTML rendering is involved; the output goes to stdout/CloudWatch logs, not to any browser context. |
| Error   | False Positive  | Codeguru     | XSS — patch_splatfactow.py:43 | Untrusted data incorporated into web page content | This line is a `print()` statement inside a file-patching utility that writes to stdout/CloudWatch logs only. No web framework or HTML rendering is involved. |
| Error   | False Positive  | Codeguru     | RAG/vector embedding — rotate_splat.py:104,189 | Code implements RAG with vector embeddings without access controls | These lines perform standard numpy quaternion arithmetic and spherical-harmonic coefficient rotation. There is no LLM, vector store, or retrieval-augmented generation involved. The scanner incorrectly matched on array indexing patterns. |
| Error   | False Positive  | Codeguru     | SQL Injection — process_pose_transforms.py:74,93,98 | SQL string concatenation may result in SQL injection | The table name is validated with `isalnum()` and `replace('"', '""')` escaping before use in PRAGMA and SELECT statements. Parameterized queries are used for all INSERT/DELETE operations. `nosemgrep` comments are present. |
| Error   | False Positive  | Codeguru     | B607 partial executable path — utils.py:869,904,921,931 | Starting a process with a partial executable path (`nvcc`, `df`, `colmap`, `glomap`) | These calls are inside `print_container_version_info()`, a diagnostic function that probes for installed system tools at container startup. The tool names are hardcoded string literals, not user-controlled input. No user data influences which executable is invoked. |
| Error   | False Positive  | Codeguru     | B607 partial executable path — demo_colmap_mapanything_.py:467 | Starting a process with a partial executable path `colmap` | This call invokes the system `colmap` binary with a hardcoded argument list for model format conversion. The path is a hardcoded string literal, not user-controlled input. |
| Error   | False Positive  | Codeguru     | Lambda reserved env var override — generate_splat_gradio.py:120,121,122 | Code overrides Lambda runtime reserved environment variables | The Gradio application runs as a standalone Python process on EC2, not inside a Lambda function. Setting `os.environ['AWS_ACCESS_KEY_ID']` etc. is the standard pattern for refreshing boto3 credentials in a long-running process and does not affect any Lambda runtime. |
| Error   | False Positive  | Codeguru     | Overly permissive managed policies — ecr.py:174, stepfunctions.py:290,293,295 | `AmazonSageMakerFullAccess`, `AWSStepFunctionsFullAccess`, `AmazonEventBridgeFullAccess` grant broad permissions | These managed policies are required for the guidance to function: `AmazonSageMakerFullAccess` is needed for the container role to submit and monitor SageMaker training jobs; `AWSStepFunctionsFullAccess` is needed for nested state machine execution; `AmazonEventBridgeFullAccess` is needed for the Step Functions EventBridge integration to monitor SageMaker/Batch job completion. No narrower AWS managed policies cover all required actions for these services. Inline justification comments have been added to the source. |
|

## Deployment and User Guide

For detailed guidance deployment steps and running the guidance as a user please see the [Implementation Guide](https://aws-solutions-library-samples.github.io/compute/open-source-3d-reconstruction-toolbox-for-gaussian-splats-on-aws.html)

## Next Steps

This robust framework for 3D reconstruction serves as a fundamental building block for scalable construction of 3D environments and content workflows. You can extend this solution in multiple ways: embed it into your web applications, integrate it with game engines for interactive experiences, or implement it in virtual production environments - these are just a few possibilities to support your requirements.

By leveraging other AWS services, you can further enhance your workflow to scale, share, and optimize your 3D reconstruction needs, whatever they might be.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](./LICENSE) file.

## Authors

- [Eric Cornwell](https://www.linkedin.com/in/eric-cornwell-2249543b/), Sr. Spatial Compute SA
- [Dario Macangano](https://www.linkedin.com/in/dario-macagnano-6b7562b9/), Sr. Worldwide Visual Compute SA
- [Stanford Lee](https://www.linkedin.com/in/stanfordlee/), Technical Account Manager (Spatial Computing TFC)
- [Daniel Zilberman](https://www.linkedin.com/in/danzilberman/), Sr. Specialist SA, Prototyping & Scaling

