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

"""Main stack to build the infrastructure and associated components"""

from stacks.components.s3 import S3
from stacks.components.ddb import Ddb
from stacks.components.lambdas import Lambda
from stacks.components.stepfunctions import Sfn
from stacks.components.ecr import Ecr
from stacks.components.sns import Sns
from aws_cdk import (
    aws_dynamodb as dynamodb,
    aws_lambda as lambda_,
    aws_ssm as ssm,
    aws_iam as iam,
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    Environment
)
from constructs import Construct
import random
import string
import os

class GSWorkflowBaseStack(Stack):
    """Class for Base Infrastructure Stack"""
    def __init__(
            self,
            scope: Construct,
            id: str,
            env: Environment,
            config_data: dict,
            **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Initialize Ids and Variables
        self.prefix = config_data['constructNamePrefix']
        self.s3_trigger_key = config_data['s3TriggerKey']
        self.random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        self.bucket_name = f"{self.prefix}-bucket-{self.random_id}"
        self.ecr_repo_name = f"{self.prefix}-ecr-repo-{self.random_id}"
        self.sfn_ssm_param_name = f"{self.prefix}-sfn-arn-{self.random_id}"
        self.container_role_name = f"{self.prefix}-container-role-{self.random_id}"
        self.ddb_table_name = f"{self.prefix}-ddb-table-{self.random_id}"
        self.state_machine_name = f"{self.prefix}-sfn-{self.random_id}"
        self.maintain_s3_objects_on_stack_deletion = config_data['maintainS3ObjectsOnStackDeletion']
        self.current_path = os.path.dirname(os.path.realpath(__file__))

        CfnOutput(self, 'Region', value=config_data['region'])

        # Create SNS topic
        sns = Sns(
            self,
            "NotificationConstruct",
            admin_email=config_data['adminEmail']
        )

        # Create outputs
        CfnOutput(self, "SnsTopicName", value=sns.sns_topic.topic_name)
        CfnOutput(self, "SnsTopicArn", value=sns.sns_topic.topic_arn)

        # ECR Construct
        self.ecr = Ecr(
            scope=self,
            id="EcrConstruct",
            env=env,
            ecr_repo_name=self.ecr_repo_name,
            s3_bucket_name=self.bucket_name,
            container_role_name=self.container_role_name
        )
        CfnOutput(self, "ECRRepoName", value=self.ecr.repository.repository_name)
        CfnOutput(self, "ContainerRoleArn", value=self.ecr.container_role.role_arn)

        sns_topic_arn = f"arn:aws:sns:{env.region}:{env.account}:{sns.sns_topic.topic_name}"
        sns_statement = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=["sns:Publish"],
            resources=[
                sns_topic_arn
                ]
        )

        sagemaker_statement = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=["sagemaker:DescribeTrainingJob"],
            resources=[
                "*"
            ]
        )

        cloudwatch_statement = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "logs:DescribeLogStreams",
                "logs:GetLogEvents"
            ],
            resources=[
                "*"
            ]
        )

        # Lambda: Workflow Complete Construct
        lambda_workflow_complete = Lambda(
            scope=self,
            id="LambdaWorkflowCompleteConstruct",
            env=env,
            runtime=lambda_.Runtime.PYTHON_3_12,
            code_path=os.path.join(
                self.current_path,
                "../../../source/lambda/workflow_complete"),
            main_function="lambda_handler",
            timeout=Duration.seconds(30),
            memory=128,
            storage=512,
            env_vars= {
                'DDB_TABLE_NAME': self.ddb_table_name,
                'SNS_TOPIC_ARN': sns.sns_topic.topic_arn
            },
            reserved_concurrent_executions=100,
            tracing=lambda_.Tracing.ACTIVE
        )
        CfnOutput(
            self,
            'LambdaWorkflowCompleteFunctionName',
            value=lambda_workflow_complete.lambda_function.function_name
        )

        # Lambda: Workflow Trigger Construct
        lambda_workflow_trigger = Lambda(
            scope=self,
            id="LambdaWorkflowTriggerConstruct",
            env=env,
            runtime=lambda_.Runtime.PYTHON_3_12,
            code_path=os.path.join(self.current_path,
                                "../../../source/lambda/workflow_trigger"),
            main_function="lambda_handler",
            timeout=Duration.seconds(30),
            memory=128,
            storage=512,
            env_vars= {
                'STATE_MACHINE_PARAM_NAME': self.sfn_ssm_param_name,
                'SNS_TOPIC_ARN': sns_topic_arn,
                'LAMBDA_COMPLETE_NAME': lambda_workflow_complete.lambda_function.function_name,
                'DDB_TABLE_NAME': self.ddb_table_name,
                'ECR_IMAGE_URI': self.ecr.repository.repository_uri,
                'CONTAINER_ROLE_NAME': self.container_role_name
                },
            reserved_concurrent_executions=100,
            tracing=lambda_.Tracing.ACTIVE
        )
        CfnOutput(
            self,
            'LambdaWorkflowTriggerFunctionName',
            value=lambda_workflow_trigger.lambda_function.function_name
        )

        # S3 Construct
        s3 = S3(
            scope=self,
            id="S3Construct",
            env=env,
            bucket_name=self.bucket_name,
            trigger_lambda_function=lambda_workflow_trigger.lambda_function,
            s3_trigger_key=self.s3_trigger_key,
            s3_trigger_extension=".json",
            maintain_s3_objects_on_stack_deletion=self.maintain_s3_objects_on_stack_deletion
        )
        CfnOutput(self, "S3BucketName", value=s3.bucket.bucket_name)

        # DynamoDB Construct
        ddb = Ddb(
            scope=self,
            id="DdbConstruct",
            env=env,
            ddb_table_name=self.ddb_table_name,
            partition_key="uuid",
            sort_key=None,
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST, #dynamodb.BillingMode.PROVISIONED
            removal_policy=RemovalPolicy.DESTROY,
            #provisioned_throughput={
            #    'read_capacity': 10,
            #    'write_capacity': 10,
            #    'min_capacity': 10,
            #    'max_capacity': 200,
            #    'target_utilization': 80
            #}
        )
        CfnOutput(self, 'DynamoDBTableName', value=ddb.table.table_name)

        # Step Functions Construct
        sfn = Sfn(
            scope=self,
            id="SfnConstruct",
            env=env,
            state_machine_name=self.state_machine_name,
            asl_code_path=os.path.join(self.current_path,
                                        "../../../source/state-machines/ASLdefinition.json"),
            workflow_trigger_lambda_arn=lambda_workflow_trigger.lambda_function.function_arn,
            workflow_complete_lambda_arn=lambda_workflow_complete.lambda_function.function_arn,
            ecr_repo_name=self.ecr_repo_name,
            container_role_name=self.container_role_name
        )
        CfnOutput(self, "StateMachineName", value=sfn.state_machine.state_machine_name)

        # Add the State Machine ARN to SSM Parameter store for lambda to grab during execution
        ssm_sfn_arn = ssm.StringParameter(
            self,
            "SfnArnParameter",
            parameter_name=self.sfn_ssm_param_name,
            string_value=sfn.state_machine.state_machine_arn,
        )
        CfnOutput(self, "SfnArnSsmParameterName", value=ssm_sfn_arn.parameter_name)

        ########### IAM Roles and Policies ##########
        # Define the DDB IAM policy statement
        dynamodb_statement = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "dynamodb:BatchGetItem",
                "dynamodb:BatchWriteItem",
                "dynamodb:ConditionCheckItem",
                "dynamodb:DeleteItem",
                "dynamodb:DescribeTable",
                "dynamodb:GetItem",
                "dynamodb:GetRecords",
                "dynamodb:GetShardIterator",
                "dynamodb:PutItem",
                "dynamodb:Query",
                "dynamodb:Scan",
                "dynamodb:UpdateItem"
                ],
            resources=[
                f"arn:aws:dynamodb:{env.region}:{env.account}:table/{self.ddb_table_name}"
            ]
        )

        # Add IAM Policy to Lambda Roles
        lambda_workflow_trigger.lambda_function.add_to_role_policy(
            statement=iam.PolicyStatement(
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

                        ],
                        effect=iam.Effect.ALLOW,
                        resources=[
                            f"arn:aws:s3:::{self.bucket_name}",
                            f"arn:aws:s3:::{self.bucket_name}/*",
                        ]
            )
        )

        lambda_workflow_trigger.lambda_function.add_to_role_policy(
            statement=iam.PolicyStatement(
                        actions=[
                            "ssm:GetParameters",
                            "ssm:GetParameter"
                        ],
                        effect=iam.Effect.ALLOW,
                        resources=[
                            f"arn:aws:ssm:{env.region}:{env.account}:parameter/{self.sfn_ssm_param_name}"
                        ]
            )
        )

        lambda_workflow_trigger.lambda_function.add_to_role_policy(
            statement=iam.PolicyStatement(
                        actions=[
                            "ssm:DescribeParameters"
                        ],
                        effect=iam.Effect.ALLOW,
                        resources=[
                            "*"
                        ]
            )
        )

        # TODO: Use specific SFN arn without cyclic reference
        lambda_workflow_trigger.lambda_function.add_to_role_policy(
            statement=iam.PolicyStatement(
                        actions=[
                            "states:StartExecution"
                        ],
                        effect=iam.Effect.ALLOW,
                        resources=[
                            f"arn:aws:states:{env.region}:{env.account}:stateMachine:*"
                        ]
            )
        )
        lambda_workflow_trigger.lambda_function.add_to_role_policy(
            statement=dynamodb_statement
        )
        lambda_workflow_complete.lambda_function.add_to_role_policy(
            statement=dynamodb_statement
        )
        lambda_workflow_complete.lambda_function.add_to_role_policy(
            statement=sns_statement
        )
        lambda_workflow_complete.lambda_function.add_to_role_policy(
            statement=sagemaker_statement
        )
        lambda_workflow_complete.lambda_function.add_to_role_policy(
            statement=cloudwatch_statement
        )
