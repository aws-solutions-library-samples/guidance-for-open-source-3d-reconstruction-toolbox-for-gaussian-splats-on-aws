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

"""Main construct to build a Step Functions State Machine"""

import os
from aws_cdk import (
    aws_iam as iam,
    aws_stepfunctions as sfn,
    Environment,
    aws_logs as logs,
)
from constructs import Construct

class Sfn(Construct):
    """Class for Step Functions State Machine Construct"""
    def __init__(
            self,
            scope: Construct,
            id: str,
            env: Environment,
            state_machine_name: str,
            asl_code_path: str,
            workflow_trigger_lambda_arn: str,
            workflow_complete_lambda_arn: str,
            ecr_repo_name: str,
            container_role_name: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.state_machine = self.create_state_machine(
            env,
            state_machine_name,
            asl_code_path,
            workflow_trigger_lambda_arn,
            workflow_complete_lambda_arn,
            ecr_repo_name,
            container_role_name
        )

    def create_state_machine(
            self,
            env,
            state_machine_name,
            asl_code_path,
            workflow_trigger_lambda_arn,
            workflow_complete_lambda_arn,
            ecr_repo_name,
            container_role_name) -> sfn.StateMachine:
        """Function to create the State Machine component"""

        # Define the state machine
        state_machine = sfn.StateMachine(
            self, "StateMachine",
            state_machine_name=state_machine_name,
            definition_body=sfn.DefinitionBody.from_file(path=asl_code_path),
            role=self.create_state_machine_role(
                env,
                workflow_trigger_lambda_arn,
                workflow_complete_lambda_arn,
                ecr_repo_name,
                container_role_name,
                state_machine_name
            ),
            tracing_enabled=True,
            logs=sfn.LogOptions(
                destination=logs.LogGroup(self, state_machine_name),  # Creates a new log group
                level=sfn.LogLevel.ALL,  # Log all events
                include_execution_data=True  # Include execution data in logs
    )
        )
        return state_machine

    def create_state_machine_role(
            self,
            env,
            workflow_trigger_lambda_arn,
            workflow_complete_lambda_arn,
            ecr_repo_name,
            container_role_name,
            state_machine_name) -> iam.Role:
        """Function to create the State Machine Iam Role"""
        # Define the Lambda Invoke IAM policy document
        lambda_invoke_statement = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=["lambda:InvokeFunction"],
            resources=[
                workflow_trigger_lambda_arn,
                workflow_complete_lambda_arn
            ]
        )

        sagemaker_training_statement = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "sagemaker:CreateTrainingJob",
                "sagemaker:DescribeTrainingJob",
                "sagemaker:StopTrainingJob",
                "sagemaker:ListTags"],
            resources=[
                f"arn:aws:sagemaker:{env.region}:{env.account}:training-job/*"
            ]
        )

        #sagemaker_tags_statement = iam.PolicyStatement(
        #    effect=iam.Effect.ALLOW,
        #    actions=[
        #        "sagemaker:ListTags"
        #        ],
        #    resources=[
        #        "*"
        #    ]
        #)

        container_role = iam.Role.from_role_name(
            self,
            id="ContainerRole",
            role_name=container_role_name
        )

        sagemaker_pass_role_statement = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "iam:PassRole"
                ],
            resources=[
                f"{container_role.role_arn}"
            ],
            conditions={
                "StringEquals": {
                    "iam:PassedToService": "sagemaker.amazonaws.com"
                }
            }
        )

        sagemaker_eventbridge_statement = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "events:PutRule",
                "events:PutTargets",
                "events:DeleteRule",
                "events:RemoveTargets",
                "events:DescribeRule",
                "events:EnableRule",
                "events:DisableRule",
                "events:CreateManagedRule",
                "events:DeleteManagedRule",
                "events:UpdateManagedRule"
                ],
            resources=[
                f"arn:aws:events:{env.region}:{env.account}:rule/*",
                f"arn:aws:events:{env.region}:{env.account}:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule",
                f"arn:aws:events:{env.region}:{env.account}:rule/StepFunctions*"
            ]
        )

        ecr_statement = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "ecr:BatchGetImage",
                "ecr:BatchCheckLayerAvailability",
                "ecr:CompleteLayerUpload",
                "ecr:GetDownloadUrlForLayer"
                ],
            resources=[
                f"arn:aws:ecr:{env.region}:{env.account}:repository/{ecr_repo_name}"
            ]
        )

        iam_statement = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "iam:PassRole",
                "iam:CreateRole",
                "iam:DeleteRole",
                "iam:GetRole",
                "iam:PutRolePolicy",
                "iam:DeleteRolePolicy"
            ],
            resources=[f"arn:aws:iam::{env.account}:role/*"]
        )

        # Add new statements for Step Functions
        step_functions_statement = iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "states:StartExecution",
                "states:StopExecution",
                "states:DescribeExecution"
            ],
            resources=[f"arn:aws:states:{env.region}:{env.account}:stateMachine:*"]
        )

        # Create the IAM role for the state machine
        role = iam.Role(
            self, "StateMachineExecutionRole",
            assumed_by=iam.ServicePrincipal("states.amazonaws.com"),
            role_name=f"{state_machine_name}-ExecutionRole",
            description="An IAM role for Step Functions",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"),
                # Changed this line - removed service-role prefix
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AWSStepFunctionsFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonEventBridgeFullAccess")
            ]
        )

        role.attach_inline_policy(
            iam.Policy(
            self, "StepFunctionsLambdaInvokePolicy",
            policy_name="StepFunctionsLambdaInvokePolicy",
            statements=[lambda_invoke_statement]
            )
        )

        role.attach_inline_policy(
            iam.Policy(
            self, "SageMakerTrainingPolicy",
            policy_name="SageMakerTrainingPolicy",
            statements=[
                sagemaker_training_statement,
                #sagemaker_tags_statement,
                sagemaker_pass_role_statement,
                sagemaker_eventbridge_statement,
                ecr_statement,
                iam_statement,
                step_functions_statement
                ]
            )
        )

        return role
