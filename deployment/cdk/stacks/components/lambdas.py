# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

"""Main construct to build a Lambda Function"""

import os
from aws_cdk import (
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_logs as logs,
    Environment, 
    Duration,
    Size
)
from constructs import Construct

class Lambda(Construct):
    """Class for Lambda Construct"""
    def __init__(
            self,
            scope: Construct,
            id: str,
            env: Environment,
            runtime: lambda_.Runtime,
            code_path: str,
            main_function: str,
            timeout: Duration = None,
            memory: int = 128,
            storage: int = 512,
            role: iam.Role = None,
            env_vars: dict[str, str] = None,
            reserved_concurrent_executions: int = None,
            tracing: lambda_.Tracing = lambda_.Tracing.ACTIVE, **kwargs) -> None :
        super().__init__(scope, id, **kwargs)

        # Create the Lambda Function
        self.lambda_function = self.create_lambda_function(
            id,
            runtime,
            code_path,
            main_function,
            timeout,
            memory,
            storage,
            role,
            env_vars,
            reserved_concurrent_executions,
            tracing
        )

    def create_lambda_function(
            self,
            id: str,
            runtime: lambda_.Runtime,
            code_path: str,
            main_function: str,
            timeout: Duration = None,
            memory: int = 128,
            storage: int = 512,
            role: iam.Role = None,
            envs: dict[str, str] = None,
            reserved_concurrent_executions: int = None,
            tracing: lambda_.Tracing = lambda_.Tracing.ACTIVE) -> lambda_.Function:
        """
        Function to create the Lambda Function with code signing and concurrency controls
        
        Args:
            id: Function identifier
            runtime: Lambda runtime
            code_path: Path to the function code
            main_function: Handler function name
            timeout: Function timeout
            memory: Memory allocation in MB
            storage: Ephemeral storage in MB
            role: IAM role for the function
            envs: Environment variables
            reserved_concurrent_executions: Number of concurrent executions to reserve
            tracing: X-Ray tracing mode (ACTIVE, PASS_THROUGH, or DISABLED)
        """
        lambda_handler = os.path.basename(code_path) + "." + main_function
        props = {
            'runtime': runtime,
            'handler': lambda_handler,
            'code': lambda_.Code.from_asset(os.path.join(os.path.dirname(__file__), code_path)),
            'timeout': timeout or Duration.seconds(3),
            'memory_size': memory,
            'ephemeral_storage_size': Size.mebibytes(storage),
            'environment': envs,
            'role': role,
            'reserved_concurrent_executions': reserved_concurrent_executions,
            'tracing': tracing,
            'log_retention': logs.RetentionDays.ONE_YEAR
        }

        lambda_function = lambda_.Function(self, f"{id}_FUNC", **props)
        return lambda_function
