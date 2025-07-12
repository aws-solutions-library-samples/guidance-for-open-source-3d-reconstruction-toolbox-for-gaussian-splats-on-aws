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

"""Post deployment stack to build the container and deploy model components"""

from stacks.components.container_deployment import ContainerDeployment
from aws_cdk import (
    Stack,
    Environment,
    CfnOutput,
    aws_iam as iam,
    aws_s3_deployment as s3deploy,
    aws_s3 as s3,
    aws_lambda as lambda_,
    Duration,
    CustomResource,
    Fn,
    custom_resources as cr,
    Size
)
from constructs import Construct
import os
import json

class GSWorkflowPostDeployStack(Stack):
    """Class for Post Deploy Infrastructure Stack"""
    def __init__(
            self,
            scope: Construct,
            id: str,
            env: Environment,
            config_data: dict,
            output_json_path: str,
            build_args: dict,
            dockerfile_path: str,
            **kwargs) -> None:
        super().__init__(scope, id, env=env, **kwargs)

        try:
            # Initialize Ids and Variables
            self.current_path = os.path.dirname(os.path.realpath(__file__))

            # Verify dockerfile path
            if not os.path.exists(dockerfile_path):
                raise ValueError(f"Dockerfile directory not found at: {dockerfile_path}")
            if not os.path.exists(os.path.join(dockerfile_path, "Dockerfile")):
                raise ValueError(f"Dockerfile not found in {dockerfile_path}")

            # Load and validate output data
            try:
                with open(output_json_path, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
                
                if 'GSWorkflowBaseStack' not in output_data:
                    raise KeyError("Base stack outputs not found")

                # Validate required outputs
                required_outputs = ['ECRRepoName', 'S3BucketName']
                for output in required_outputs:
                    if output not in output_data['GSWorkflowBaseStack']:
                        raise KeyError(f"Required output '{output}' not found in base stack outputs")

                # Log output data for debugging
                print(f"ECR Repo Name: {output_data['GSWorkflowBaseStack']['ECRRepoName']}")
                print(f"S3 Bucket Name: {output_data['GSWorkflowBaseStack']['S3BucketName']}")

            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in output file: {e}")

            # Create deployment role with required permissions
            deployment_role = iam.Role(
                self,
                "ContainerDeploymentRole",
                assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
                managed_policies=[
                    iam.ManagedPolicy.from_aws_managed_policy_name(
                        "service-role/AWSLambdaBasicExecutionRole"
                    )
                ]
            )

            # Add ECR permissions
            deployment_role.add_to_policy(
                iam.PolicyStatement(
                    actions=[
                        "ecr:GetAuthorizationToken",
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:GetRepositoryPolicy",
                        "ecr:DescribeRepositories",
                        "ecr:ListImages",
                        "ecr:DescribeImages",
                        "ecr:BatchGetImage",
                        "ecr:InitiateLayerUpload",
                        "ecr:UploadLayerPart",
                        "ecr:CompleteLayerUpload",
                        "ecr:PutImage"
                    ],
                    resources=[f"arn:aws:ecr:{env.region}:{env.account}:repository/{output_data['GSWorkflowBaseStack']['ECRRepoName']}"]
                )
            )

            # Container Deployment Construct
            container_deployment = ContainerDeployment(
                scope=self,
                id="ContainerDeployment",
                env=env,
                config_data=config_data,
                output_data=output_data['GSWorkflowBaseStack'],
                build_args=build_args,
                dockerfile_path=dockerfile_path
            )

            # Add outputs
            CfnOutput(
                self,
                "DeploymentBucketName",
                value=output_data['GSWorkflowBaseStack']['S3BucketName'],
                description="Name of the deployment bucket"
            )

            CfnOutput(
                self,
                "ECRRepositoryName",
                value=output_data['GSWorkflowBaseStack']['ECRRepoName'],
                description="Name of the ECR repository"
            )

            # Create a Lambda function to download and upload models
            # Use absolute path to avoid path resolution issues
            #lambda_code_path = os.path.join("/mnt/efs/gaussian_splats/backend/lambda/model_deployment")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            lambda_code_path = os.path.join(current_dir, "..", "..", "..", "source", "lambda", "model_deployment")
            
            # Use the existing index.py file that we've already created
            
            model_deployment_lambda = lambda_.Function(
                self,
                "ModelDeploymentLambda",
                runtime=lambda_.Runtime.PYTHON_3_9,
                handler="index.handler",
                timeout=Duration.minutes(15),
                memory_size=3072,  # Increased memory for large model download
                ephemeral_storage_size=Size.gibibytes(10),  # Add more storage for the large model file
                code=lambda_.Code.from_asset(lambda_code_path)
            )

            # Grant S3 permissions to the Lambda
            s3_bucket = s3.Bucket.from_bucket_name(
                self, 
                "ImportedBucket", 
                output_data['GSWorkflowBaseStack']['S3BucketName']
            )
            s3_bucket.grant_read_write(model_deployment_lambda)

            # Create a custom resource provider
            provider = cr.Provider(
                self,
                "ModelDeploymentProvider",
                on_event_handler=model_deployment_lambda
            )

            # Create a custom resource to trigger the model deployment
            model_deployment = CustomResource(
                self,
                "ModelDeployment",
                service_token=provider.service_token,
                properties={
                    "BucketName": output_data['GSWorkflowBaseStack']['S3BucketName'],
                    "Timestamp": str(os.path.getmtime(__file__)),  # Force update on code change
                    "DeploymentId": f"{id}-{env.account}-{env.region}"  # Ensure uniqueness
                }
            )

            # Add output for the models location
            CfnOutput(
                self,
                "ModelsArchiveLocation",
                value=f"s3://{output_data['GSWorkflowBaseStack']['S3BucketName']}/models/models.tar.gz",
                description="Location of the models archive in S3"
            )

        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"Error creating post-deploy stack: {e}")
            raise e
        except Exception as e:
            print(f"Unexpected error creating post-deploy stack: {e}")
            raise e

    @property
    def outputs(self):
        return {
            'DeploymentBucketName': self.get_output('DeploymentBucketName'),
            'ECRRepositoryName': self.get_output('ECRRepositoryName'),
            'ModelsArchiveLocation': self.get_output('ModelsArchiveLocation')
        }

    def get_output(self, output_name: str) -> str:
        """Helper method to get stack outputs"""
        return Fn.get_att(self.stack_name, f'Outputs.{output_name}')