# Guidance for Open Source 3D Reconstruction Toolbox for Gaussian Splats on AWS

This title correlates exactly to the Guidance it’s linked to, including its corresponding sample code repository. 


## Table of Contents (required)

List the top-level sections of the README template, along with a hyperlink to the specific section.

### Required

1. [Overview](#overview)
    - [Cost](#cost)
2. [Prerequisites](#prerequisites)
    - [Operating System](#operating-system)
3. [Deployment Steps](#deployment-steps)
4. [Deployment Validation](#deployment-validation)
5. [Running the Guidance](#running-the-guidance)
6. [Next Steps](#next-steps)
7. [Cleanup](#cleanup)

***Optional***

8. [FAQ, known issues, additional considerations, and limitations](#faq-known-issues-additional-considerations-and-limitations-optional)
9. [Revisions](#revisions-optional)
10. [Notices](#notices-optional)
11. [Authors](#authors-optional)

<!-- copied from the GitLab README -->
# Open Source 3D Reconstruction Toolbox for Gaussian Splats
This solution provides the infrastructure and open source code to reconstruct 3D scenes or objects (splats) from images or video. Under the hood, there is a 3D gaussian splatting workflow with various processing pipelines such as image processing/filtering (background removal), structure-from-motion (images-to-point-cloud), and gaussian splat training which uses traditional AI/ML approaches on a GPU. Both a Cloud Development Kit (CDK) and a Terraform infrastructure as code deployments are available and deploy a full backend system to 1/ enable a user to upload the media and json configuration file to S3 and 2/ output an email notification on completion of splat and assets.

## I. Summary

The Open Source 3D Reconstruction Toolbox for Gaussian Splats provides an end-to-end, pipeline-based solution on AWS to reconstruct 3D scenes or objects from images or video inputs. The infrastructure can be deployed via AWS Cloud Development Kit (CDK) or Terraform leveraging infrastructure-as-code. 

Once deployed, the solution features a full 3D reconstruction back-end system with the following customizable components or pipelines:

1. **Media Ingestion**: Process videos or collections of images as input
2. **Image Processing**: Automatic filtering, enhancement, and preparation of source imagery (e.g. background removal)
3. **Structure from Motion (SfM)**: Camera pose estimation and initial 3D point cloud generation
4. **Gaussian Splat Training**: Optimization of 3D Gaussian primitives to represent the scene using AI/ML
5. **Export & Delivery**: Generation of the final 3D asset in standard formats for easy viewing and notification via email

By deploying this solution, users gain access to a flexible infrastructure that handles the entire 3D reconstruction process programatically, from media upload to final 3D model delivery, while being highly modular through its componentized pipeline-based approach. This solution addresses the significant challenges organizations face when trying to create photorealistic 3D content - traditionally a time-consuming, expensive, and technically complex process requiring specialized skills and equipment.

<!-- ![](docs/media/gs-workflow-arch.png "Architecture Diagram") -->

1.	System administrator deploys solution to AWS account and region using AWS Cloud Development Kit or Terraform.
2.	Once solution is deployed in a specific AWS account and region, an authenticated user uploads the necessary configuration and input media into a dedicated Amazon Simple Storage Service (S3) bucket location. This can be done using a Gradio interface and AWS Software Development Kit (SDK).
3.	Optionally, the solution supports external job submission by uploading a ‘.json’ job configuration file and media into a designated S3 bucket location. This upload process could be manual through the AWS Management Console or could also be an external process depending on the use-case.
4.	The job json file upload to the bucket location will trigger an Amazon Simple Notification Service (SNS) message that will invoke an initialization AWS Lambda function.
5.	The initialization Lambda function will perform input validation and set appropriate variables for the state machine.
6.	The workflow job record will be created in Amazon DynamoDB job table.
7.	The initialization Lambda function will invoke an AWS Step Functions State Machine to handle the entire workflow job.
8.	If the configuration is successful, an Amazon SageMaker Training Job will be submitted synchronously using the state machine built-in wait until completion mechanism. Otherwise (jump to step 11), the completion Lambda function will handle the error, update the database and notify the user via an SNS email.
9.	The Amazon Elastic Container Registry (ECR) container image and S3 model artifacts will be used to spin up a new graphics processing unit (GPU) container. The instance type is determined by the job json configuration.
10.	The GPU container will run the entire pipeline.
11.	Upon job completion or error, a completion Lambda function will complete the workflow job by updating the job in DynamoDB and notifying the user via email upon completion using SNS.
12.	Internal workflow parameters are stored in Parameter Store during solution deployment to decouple services.
13.	Amazon CloudWatch is used to monitor the training logs, surfacing errors to th


This simple backend will:
   - create the infrastructure required to create a gaussian splat from a video or set of images
   - create the mechanism to run the code and perform 3D reconstruction
   - enable a user to create a 3D gaussian splat from the backend (no UI) using open source tools and AWS by uploading a video (.mp4 or .mov) or images (.png or .jpg) and metadata (.json) into S3

### Custom GS Pipeline Container
In this project, there is only one Docker container that contains all of the 3D reconstruction tools for Gaussian Splatting. This container has a `Dockerfile`, `main.py`, and helper script files and open source libraries under the `backend/container` directory. The main script processes each request from the SageMaker Training Job invoke message and saves the result to S3 upon successful completion. The list of open source libraries that make this project possible include:
- [NerfStudio](https://github.com/nerfstudio-project/nerfstudio) [(Apache-2.0)](https://github.com/nerfstudio-project/nerfstudio/tree/main?tab=Apache-2.0-1-ov-file#readme)
- [Glomap](https://github.com/colmap/glomap) [(BSD-3-Clause)](https://github.com/colmap/glomap?tab=BSD-3-Clause-1-ov-file#readme)
- [Colmap](https://github.com/colmap/colmap) [(BSD)](https://github.com/colmap/colmap?tab=License-1-ov-file#readme)
- [OpenCV](https://github.com/opencv/opencv) [(Apache-2.0)](https://github.com/opencv/opencv?tab=Apache-2.0-1-ov-file#readme)
- [gsplat](https://github.com/nerfstudio-project/gsplat) [(Apache-2.0)](https://github.com/nerfstudio-project/gsplat?tab=Apache-2.0-1-ov-file#readme)
- [backgroundremover](https://github.com/nadermx/backgroundremover) [(MIT)](https://github.com/nadermx/backgroundremover?tab=MIT-1-ov-file#readme)
- [splatfacto-w](https://github.com/KevinXu02/splatfacto-w) [(Apache-2.0)](https://github.com/KevinXu02/splatfacto-w?tab=Apache-2.0-1-ov-file#readme)

## II. Requirements



<!-- end of copy from GitLab FREADME -->
## Overview

1. Provide a brief overview explaining the what, why, or how of your Guidance. You can answer any one of the following to help you write this:

    - **Why did you build this Guidance?**
    - **What problem does this Guidance solve?**

2. Include the architecture diagram image, as well as the steps explaining the high-level overview and flow of the architecture. 
    - To add a screenshot, create an ‘assets/images’ folder in your repository and upload your screenshot to it. Then, using the relative file path, add it to your README. 

### Cost

This section is for a high-level cost estimate. Think of a likely straightforward scenario with reasonable assumptions based on the problem the Guidance is trying to solve. Provide an in-depth cost breakdown table in this section below ( you should use AWS Pricing Calculator to generate cost breakdown ).

Start this section with the following boilerplate text:

_You are responsible for the cost of the AWS services used while running this Guidance. As of <month> <year>, the cost for running this Guidance with the default settings in the <Default AWS Region (Most likely will be US East (N. Virginia)) > is approximately $<n.nn> per month for processing ( <nnnnn> records )._

Replace this amount with the approximate cost for running your Guidance in the default Region. This estimate should be per month and for processing/serving resonable number of requests/entities.

Suggest you keep this boilerplate text:
_We recommend creating a [Budget](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html) through [AWS Cost Explorer](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/) to help manage costs. Prices are subject to change. For full details, refer to the pricing webpage for each AWS service used in this Guidance._

### Sample Cost Table

**Note : Once you have created a sample cost table using AWS Pricing Calculator, copy the cost breakdown to below table and upload a PDF of the cost estimation on BuilderSpace. Do not add the link to the pricing calculator in the ReadMe.**

The following table provides a sample cost breakdown for deploying this Guidance with the default parameters in the US East (N. Virginia) Region for one month.

| AWS service  | Dimensions | Cost [USD] |
| ----------- | ------------ | ------------ |
| Amazon API Gateway | 1,000,000 REST API calls per month  | $ 3.50month |
| Amazon Cognito | 1,000 active users per month without advanced security feature | $ 0.00 |

## Prerequisites 

- Local computer with appropriate AWS credentials to deploy the CDK or Terraform solution
- **(Optional, but recommended)** Use an EC2 workstation to build and deploy the CDK or Terraform solution
    - Ensure your local computer has an SSH client (For Windows, [Putty](https://www.putty.org/) was tested)
    - Ensure your local computer has the NICE DCV client installed ([Windows](https://docs.aws.amazon.com/dcv/latest/userguide/client-windows.html), [MacOS](https://docs.aws.amazon.com/dcv/latest/userguide/client-mac.html), or [Linux](https://docs.aws.amazon.com/dcv/latest/userguide/client-linux.html))
    - A CloudFormation template is given [here](https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop) to spin up a fresh, full-featured Ubuntu desktop
        1. Prerequisites: Before you build the EC2 workstation stack, ensure the following resources are created in your AWS account and region of choice:
            - VPC
                - Follow [these instructions](https://docs.aws.amazon.com/vpc/latest/userguide/create-vpc.htm) if you do not have one. This will be where your EC2 will live. Ensure there is a public subnet available with internet access in order to pull the GitHub repositories.
            - Keypair
                - Follow [these instructions](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/create-key-pairs.html) if you do not have one. This is used to remote into the EC2 desktop.
            - Security Group
                - Follow [these instructions](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/creating-security-group.html) to create a security group. Enable inbound NiceDCV using TCP/UDP port 8443 and SSH using port 22. Ensure your source IP address is the resource for all entries.
                - For Inbound rules, add:
                    - Custom TCP, Port range=8443, source="My IP"
                    - Custom UDP, Port range=8443, source="My IP"
                    - SSH, Port range=22, source="My IP"
                - Record the security group Id for later
        2. Download the `deep-learning-ubuntu-desktop.yaml` file locally from the repo linked above
        3. Open the AWS Console and navigate to the CloudFormation console
        4. Select `Create stack` -> `With new resources`
        5. On `Create Stack` page, select:
            - Choose `an existing template`
            - Choose `Upload a template file`
            - Select the `deep-learning-ubuntu-desktop.yaml` file downloaded earlier
        6. On `Specify stack details` page, leave default values except for the following:
            - Stack Name: `YOUR-CHOICE`
            - AWSUbuntuAMIType: `UbuntuPro2204LTS`
            - DesktopAccessCIDR: `YOUR-PUBLIC-IP-ADDRESS/32`
            - DesktopInstanceType: `g4dn.2xlarge`
            - DesktopSecurityGroupId: `SG-ID-FROM-ABOVE`
            - DesktopVpcId: `VPC-ID-FROM-ABOVE`
            - DesktopVpcSubnetId: `PUBLIC-SUBNET-ID`
            - KeyName: `KEYNAME-FROM-ABOVE`
            - S3Bucket: `S3-BUCKET-WITH-MODELS`
        7. Submit and monitor the stack creation in the CloudFormation console
        8. On successful building of the stack, navigate to the EC2 console in the account and region the deployed stack is in
        9. Locate the instance just created using the `Stack Name` entered above, select the instance, and select `Actions->Security->Modify IAM Role`
        10. Record the current IAM role name
        11. Navigate to the IAM Console in a separate browser tab or window
        12. Under `Roles`, search for the role using the IAM role name identified above
        13. Select the role by clicking on its name
        14. In the permissions policies table, select `Add permissions->Attach policies`:
            - Attach the following AWS managed policies to the role
                - AmazonEC2ContainerRegistryFullAccess
                - AmazonS3FullAccess
                - AmazonSSMManagedInstanceCore
                - AWSCloudFormationFullAccess
                - IAMFullAccess
        15. SSH into the workstation using the EC2 public IP (found in the EC2 console), security group, and SSH terminal
        16. Once connected to the EC2 workstation, perform the following commands to update the OS and password
            ```bash
            sudo apt update
            sudo passwd ubuntu
            ```
        17. The EC2 will reboot automatically while updating is being performed in the background
        18. The EC2 setup is complete once the message `echo 'NICE DCV server is enabled!'` is shown when performing the following command
            ```bash
            tail /var/log/cloud-init-output.log
            ```
        19. Once the EC2 has the enabled NICE DCV message, use the NICE DCV client, EC2 public IP address, username `ubuntu` and Ubuntu password set earlier to remotely connect to the EC2 instance.
        20. Be sure to **not upgrade the OS** (even when prompted) as it will break critical packages. Only choose to enable security updates.
        21. Open the Visual Code program in the EC2 instance by locating it in the Application library



### Third-party tools (If applicable)

*List any installable third-party tools required for deployment.*


### AWS account requirements (If applicable)

*List out pre-requisites required on the AWS account if applicable, this includes enabling AWS regions, requiring ACM certificate.*

**Example:** “This deployment requires you have public ACM certificate available in your AWS account”

**Example resources:**
- ACM certificate 
- DNS record
- S3 bucket
- VPC
- IAM role with specific permissions
- Enabling a Region or service etc.


### aws cdk bootstrap (if sample code has aws-cdk)

<If using aws-cdk, include steps for account bootstrap for new cdk users.>

**Example blurb:** “This Guidance uses aws-cdk. If you are using aws-cdk for first time, please perform the below bootstrapping....”

### Service limits  (if applicable)

<Talk about any critical service limits that affect the regular functioning of the Guidance. If the Guidance requires service limit increase, include the service name, limit name and link to the service quotas page.>

### Supported Regions (if applicable)

<If the Guidance is built for specific AWS Regions, or if the services used in the Guidance do not support all Regions, please specify the Region this Guidance is best suited for>


## Deployment Steps (required)

Deployment steps must be numbered, comprehensive, and usable to customers at any level of AWS expertise. The steps must include the precise commands to run, and describe the action it performs.

* All steps must be numbered.
* If the step requires manual actions from the AWS console, include a screenshot if possible.
* The steps must start with the following command to clone the repo. ```git clone xxxxxxx```
* If applicable, provide instructions to create the Python virtual environment, and installing the packages using ```requirement.txt```.
* If applicable, provide instructions to capture the deployed resource ARN or ID using the CLI command (recommended), or console action.

 
**Example:**

1. Clone the repo using command ```git clone xxxxxxxxxx```
2. cd to the repo folder ```cd <repo-name>```
3. Install packages in requirements using command ```pip install requirement.txt```
4. Edit content of **file-name** and replace **s3-bucket** with the bucket name in your account.
5. Run this command to deploy the stack ```cdk deploy``` 
6. Capture the domain name created by running this CLI command ```aws apigateway ............```



## Deployment Validation  (required)

<Provide steps to validate a successful deployment, such as terminal output, verifying that the resource is created, status of the CloudFormation template, etc.>


**Examples:**

* Open CloudFormation console and verify the status of the template with the name starting with xxxxxx.
* If deployment is successful, you should see an active database instance with the name starting with <xxxxx> in        the RDS console.
*  Run the following CLI command to validate the deployment: ```aws cloudformation describe xxxxxxxxxxxxx```



## Running the Guidance (required)

<Provide instructions to run the Guidance with the sample data or input provided, and interpret the output received.> 

This section should include:

* Guidance inputs
* Commands to run
* Expected output (provide screenshot if possible)
* Output description

## Next Steps (required)

Provide suggestions and recommendations about how customers can modify the parameters and the components of the Guidance to further enhance it according to their requirements.

## Cleanup (required)

- Include detailed instructions, commands, and console actions to delete the deployed Guidance.
- If the Guidance requires manual deletion of resources, such as the content of an S3 bucket, please specify.



## FAQ, known issues, additional considerations, and limitations (optional)


**Known issues (optional)**

<If there are common known issues, or errors that can occur during the Guidance deployment, describe the issue and resolution steps here>


**Additional considerations (if applicable)**

<Include considerations the customer must know while using the Guidance, such as anti-patterns, or billing considerations.>

**Examples:**

- “This Guidance creates a public AWS bucket required for the use-case.”
- “This Guidance created an Amazon SageMaker notebook that is billed per hour irrespective of usage.”
- “This Guidance creates unauthenticated public API endpoints.”


Provide a link to the *GitHub issues page* for users to provide feedback.


**Example:** *“For any feedback, questions, or suggestions, please use the issues tab under this repo.”*

## Revisions (optional)

Document all notable changes to this project.

Consider formatting this section based on Keep a Changelog, and adhering to Semantic Versioning.

## Notices (optional)

*Customers are responsible for making their own independent assessment of the information in this Guidance. This Guidance: (a) is for informational purposes only, (b) represents AWS current product offerings and practices, which are subject to change without notice, and (c) does not create any commitments or assurances from AWS and its affiliates, suppliers or licensors. AWS products or services are provided “as is” without warranties, representations, or conditions of any kind, whether express or implied. AWS responsibilities and liabilities to its customers are controlled by AWS agreements, and this Guidance is not part of, nor does it modify, any agreement between AWS and its customers.*


## Authors (optional)

Standford Lee, Technical Account Manager (ANZ)
Eric Cornwell, Sr. Spatial Compute SA
Dario Macangano, Sr. WordlWide Visual Compute SA
Daniel Zilberman, Sr. SA AWS Technical Solutions
