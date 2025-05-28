# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

"""Main construct to create an ECR Repository for a docker image"""

from aws_cdk import (
    aws_ecr as ecr,
    aws_iam as iam,
    Environment,
    RemovalPolicy,
)
from constructs import Construct

class Ecr(Construct):
    """Class for the ECR Construct"""
    def __init__(
            self,
            scope: Construct,
            id: str,
            env: Environment,
            ecr_repo_name: str,
            s3_bucket_name: str,
            container_role_name: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create the ECR repository
        self.repository = ecr.Repository(
            self, "EcrRepo",
            repository_name=ecr_repo_name,
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True,
            #auto_delete_images=True,
            image_scan_on_push=False,
            image_tag_mutability=ecr.TagMutability.MUTABLE
            # Using default encryption (S3-managed keys)
        )

        # Create the Container IAM Role
        self.container_role = self.create_container_role(
            env,
            self.repository,
            s3_bucket_name,
            container_role_name
        )

    def create_container_role(self, env, ecr_repo_name, s3_bucket_name, container_role_name) -> iam.Role:
        """Function to create the Container Iam Role"""
        # Define the IAM policy
        #container_policy_statement_ecr1 = iam.PolicyStatement(
        #    effect=iam.Effect.ALLOW,
        #    actions=["ecr:CreateRepository","ecr:GetAuthorizationToken"],
        #    resources=["*"]
        #)

        container_policy_statement_ecr2 = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "ecr:BatchCheckLayerAvailability",
                "ecr:BatchGetImage",
                "ecr:CompleteLayerUpload",
                "ecr:GetDownloadUrlForLayer",
                "ecr:InitiateLayerUpload",
                "ecr:PutImage",
                "ecr:UploadLayerPart",
                "ecr:CreateRepository",
                "ecr:GetAuthorizationToken"
            ],
            resources=[
                f"arn:aws:ecr:{env.region}:{env.account}:repository/{ecr_repo_name}"
            ]
        )

        container_policy_statement_s3 = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "s3:Abort*",
                "s3:DeleteObject*",
                "s3:GetBucket*",
                "s3:GetObject*",
                "s3:List*",
                "s3:PutObject",
                "s3:PutObjectLegalHold",
                "s3:PutObjectRetention",
                "s3:PutObjectTagging",
                "s3:PutObjectVersionTagging"
            ],
            resources=[
                f"arn:aws:s3:::{s3_bucket_name}",
                f"arn:aws:s3:::{s3_bucket_name}/*"
            ]
        )

        # Add KMS permissions for S3 and ECR encryption
        #container_policy_statement_kms = iam.PolicyStatement(
        #    effect=iam.Effect.ALLOW,
        #    actions=[
        #        "kms:Decrypt",
        #        "kms:DescribeKey",
        #        "kms:Encrypt",
        #        "kms:GenerateDataKey*",
        #        "kms:ReEncrypt*"
        #    ],
        #    resources=["*"]  # You can restrict this to specific KMS keys if known
        #)

        # Create the IAM policy
        container_policy = iam.Policy(self,
            "ContainerPolicy",
            #policy_name="ContainerPolicy",
            statements=[
                #container_policy_statement_ecr1,
                container_policy_statement_ecr2,
                container_policy_statement_s3,
                #container_policy_statement_kms
            ]
        )

        # Create an IAM role for the container
        container_role = iam.Role(self,
            "ContainerRole",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("ec2.amazonaws.com"),
                iam.ServicePrincipal("sagemaker.amazonaws.com")
            ),
            role_name=container_role_name,
            description="An IAM role for the Container",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonEC2ContainerRegistryFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSSMManagedInstanceCore")
            ]
        )
        container_role.attach_inline_policy(container_policy)
        return container_role
