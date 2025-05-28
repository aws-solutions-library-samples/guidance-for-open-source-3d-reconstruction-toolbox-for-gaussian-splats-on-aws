# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

"""Main construct to build and push the container resources to ECR"""

from aws_cdk import (
    Environment,
    RemovalPolicy,
    aws_ecr as ecr,
    aws_iam as iam,
    aws_ecr_assets as ecr_assets,
    CfnOutput
)
from aws_cdk.aws_ecr_assets import DockerImageAsset
import cdk_ecr_deployment
from constructs import Construct
import json
import os

class ContainerDeployment(Construct):
    """Class for Container Deployment Construct"""
    def __init__(
            self,
            scope: Construct,
            id: str,
            config_data: dict,
            output_data: dict,
            build_args: dict,
            dockerfile_path: str,
            env: Environment,
            **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        try:
            # Validate input parameters
            if not dockerfile_path or not os.path.exists(dockerfile_path):
                raise ValueError(f"Invalid dockerfile path: {dockerfile_path}")
            if not os.path.exists(os.path.join(dockerfile_path, "Dockerfile")):
                raise ValueError(f"Dockerfile not found in {dockerfile_path}")

            # Get ECR repository name
            print(output_data)
            # Check if output_data already has the GSWorkflowBaseStack key or is already the stack outputs
            if 'GSWorkflowBaseStack' in output_data:
                ecr_repo_name = output_data['GSWorkflowBaseStack']['ECRRepoName']
            else:
                ecr_repo_name = output_data['ECRRepoName']
                
            if not ecr_repo_name:
                raise ValueError("ECR repository name not found in output data")

            # Create deployment role with required permissions
            deployment_role = iam.Role(
                self,
                "ECRDeploymentRole",
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
                    resources=[f"arn:aws:ecr:{env.region}:{env.account}:repository/{ecr_repo_name}"]
                )
            )

            # Build and assign the docker image for ECR
            self.asset = ecr_assets.DockerImageAsset(
                self,
                "DockerImage",
                asset_name=ecr_repo_name,
                directory=dockerfile_path,
                build_args=build_args,
                platform=ecr_assets.Platform.LINUX_AMD64,
                #build_options={
                #    "platform": "linux/amd64"
                #}
            )

            # Copy image from cdk docker image asset to ECR
            self.deployment = cdk_ecr_deployment.ECRDeployment(
                self,
                "DeployDockerImage",
                src=cdk_ecr_deployment.DockerImageName(self.asset.image_uri),
                dest=cdk_ecr_deployment.DockerImageName(
                    f"{env.account}.dkr.ecr.{env.region}.amazonaws.com/{ecr_repo_name}:latest"
                ),
                role=deployment_role,
                memory_limit=512,
            )

            # Add dependencies
            if hasattr(self.asset, "node"):
                self.deployment.node.add_dependency(self.asset)

            # Add outputs
            CfnOutput(
                self,
                "DockerImageUri",
                value=self.asset.image_uri,
                description="URI of the built Docker image"
            )

            CfnOutput(
                self,
                "ECRRepositoryUri",
                value=f"{env.account}.dkr.ecr.{env.region}.amazonaws.com/{ecr_repo_name}",
                description="URI of the ECR repository"
            )

        except Exception as e:
            print(f"Error in ContainerDeployment construct: {e}")
            raise e

    @property
    def image_uri(self) -> str:
        """Return the URI of the deployed Docker image"""
        return self.asset.image_uri if hasattr(self, 'asset') else None

    @property
    def deployment_role(self) -> iam.Role:
        """Return the deployment role"""
        return self.deployment.role if hasattr(self, 'deployment') else None

    def add_to_role_policy(self, statement: iam.PolicyStatement):
        """Add additional permissions to the deployment role"""
        if self.deployment_role:
            self.deployment_role.add_to_policy(statement)
