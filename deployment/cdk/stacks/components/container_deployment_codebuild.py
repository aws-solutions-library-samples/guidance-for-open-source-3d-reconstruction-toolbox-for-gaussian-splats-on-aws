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

"""Container deployment using CodeBuild"""

from aws_cdk import (
    Environment,
    Duration,
    aws_ecr as ecr,
    aws_iam as iam,
    aws_codebuild as codebuild,
    aws_s3_assets as s3_assets,
    custom_resources as cr,
    CfnOutput
)
from constructs import Construct
import os

class ContainerDeploymentCodeBuild(Construct):
    """Class for Container Deployment using CodeBuild"""
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

        if not dockerfile_path or not os.path.exists(dockerfile_path):
            raise ValueError(f"Invalid dockerfile path: {dockerfile_path}")
        if not os.path.exists(os.path.join(dockerfile_path, "Dockerfile")):
            raise ValueError(f"Dockerfile not found in {dockerfile_path}")

        ecr_repo_name = output_data.get('ECRRepoName')
        if not ecr_repo_name:
            raise ValueError("ECR repository name not found in output data")

        self.image_uri_value = f"{env.account}.dkr.ecr.{env.region}.amazonaws.com/{ecr_repo_name}:latest"

        docker_asset = s3_assets.Asset(
            self,
            "DockerSourceAsset",
            path=dockerfile_path
        )

        build_project = codebuild.Project(
            self,
            "DockerBuildProject",
            project_name=f"docker-build-{ecr_repo_name}",
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                privileged=True,
                compute_type=codebuild.ComputeType.LARGE
            ),
            environment_variables={
                "AWS_ACCOUNT_ID": codebuild.BuildEnvironmentVariable(value=env.account),
                "AWS_REGION": codebuild.BuildEnvironmentVariable(value=env.region),
                "ECR_REPO_NAME": codebuild.BuildEnvironmentVariable(value=ecr_repo_name),
                "CODE_PATH": codebuild.BuildEnvironmentVariable(value=build_args.get('CODE_PATH', '/opt/ml/code')),
                "MODEL_PATH": codebuild.BuildEnvironmentVariable(value=build_args.get('MODEL_PATH', '/opt/ml/model')),
            },
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "pre_build": {
                        "commands": [
                            "echo Logging in to Amazon ECR...",
                            "aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com",
                            "echo Downloading source from S3...",
                            "aws s3 cp s3://$SOURCE_BUCKET/$SOURCE_KEY source.zip",
                            "unzip -o source.zip -d build_context",
                            "ls -la build_context/"
                        ]
                    },
                    "build": {
                        "commands": [
                            "echo Building Docker image...",
                            "cd build_context",
                            "docker build --build-arg CODE_PATH=$CODE_PATH --build-arg MODEL_PATH=$MODEL_PATH -t $ECR_REPO_NAME:latest .",
                            "docker tag $ECR_REPO_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest"
                        ]
                    },
                    "post_build": {
                        "commands": [
                            "echo Pushing Docker image to ECR...",
                            "docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest",
                            "echo Docker image pushed successfully"
                        ]
                    }
                }
            }),
            timeout=Duration.hours(3)
        )

        docker_asset.grant_read(build_project)

        ecr_repo = ecr.Repository.from_repository_name(
            self,
            "ExistingECRRepo",
            repository_name=ecr_repo_name
        )
        ecr_repo.grant_pull_push(build_project)

        build_project.add_to_role_policy(
            iam.PolicyStatement(
                actions=["ecr:GetAuthorizationToken"],
                resources=["*"]
            )
        )

        trigger_build = cr.AwsCustomResource(
            self,
            "TriggerDockerBuild",
            on_create=cr.AwsSdkCall(
                service="CodeBuild",
                action="startBuild",
                parameters={
                    "projectName": build_project.project_name,
                    "environmentVariablesOverride": [
                        {
                            "name": "SOURCE_BUCKET",
                            "value": docker_asset.s3_bucket_name,
                            "type": "PLAINTEXT"
                        },
                        {
                            "name": "SOURCE_KEY",
                            "value": docker_asset.s3_object_key,
                            "type": "PLAINTEXT"
                        }
                    ]
                },
                physical_resource_id=cr.PhysicalResourceId.of(f"docker-build-{ecr_repo_name}"),
                output_paths=["build.id"]
            ),
            on_update=cr.AwsSdkCall(
                service="CodeBuild",
                action="startBuild",
                parameters={
                    "projectName": build_project.project_name,
                    "environmentVariablesOverride": [
                        {
                            "name": "SOURCE_BUCKET",
                            "value": docker_asset.s3_bucket_name,
                            "type": "PLAINTEXT"
                        },
                        {
                            "name": "SOURCE_KEY",
                            "value": docker_asset.s3_object_key,
                            "type": "PLAINTEXT"
                        }
                    ]
                },
                physical_resource_id=cr.PhysicalResourceId.of(f"docker-build-{ecr_repo_name}"),
                output_paths=["build.id"]
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements([
                iam.PolicyStatement(
                    actions=["codebuild:StartBuild"],
                    resources=[build_project.project_arn]
                )
            ])
        )

        CfnOutput(
            self,
            "DockerImageUri",
            value=self.image_uri_value,
            description="URI of the built Docker image"
        )

        CfnOutput(
            self,
            "ECRRepositoryUri",
            value=f"{env.account}.dkr.ecr.{env.region}.amazonaws.com/{ecr_repo_name}",
            description="URI of the ECR repository"
        )

        CfnOutput(
            self,
            "CodeBuildProjectName",
            value=build_project.project_name,
            description="Name of the CodeBuild project"
        )

    @property
    def image_uri(self) -> str:
        return self.image_uri_value if hasattr(self, 'image_uri_value') else None

    @property
    def deployment_role(self) -> iam.Role:
        return None

    def add_to_role_policy(self, statement: iam.PolicyStatement):
        pass
