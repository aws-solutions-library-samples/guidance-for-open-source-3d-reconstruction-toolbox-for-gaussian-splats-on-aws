#!/usr/bin/env python3
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

""" Main entry into CDK App to build infrastructure stack """

import os
import json
import aws_cdk as cdk
from stacks.infra_stack import GSWorkflowBaseStack
from stacks.post_deploy_stack import GSWorkflowPostDeployStack

app = cdk.App()

# Load the app configuration from the config.json file
try:
    with open("config.json", "r", encoding="utf-8") as config_file:
        config_data = json.load(config_file)
except Exception as e:
    print(f"Could not read the app configuration file. {e}")
    raise e

# Set CDK environment variables
environment = cdk.Environment(
    account=config_data['accountId'],
    region=config_data['region']
)

# Comply with SageMaker path definitions
current_path = os.path.dirname(os.path.realpath(__file__))
build_args = {'CODE_PATH':'/opt/ml/code','MODEL_PATH':'/opt/ml/model'} #,'--no-cache': 'true'}

# Handle deploying and destroying groups of stacks
select_all = False
bundling_stacks = app.node.try_get_context("aws:cdk:bundling-stacks")
is_destroy = app.node.try_get_context("destroy")
bootstrap = app.node.try_get_context("bootstrap")

# Check if bundling_stacks exists and contains "**"
if bundling_stacks and "**" in bundling_stacks:
    select_all = True

# Create the Base Stack
if is_destroy or select_all or bootstrap or bundling_stacks is None or (bundling_stacks and "GSWorkflowBaseStack" in bundling_stacks):
    print("Creating base stack...")
    base_stack = GSWorkflowBaseStack(
        scope=app,
        id="GSWorkflowBaseStack",
        config_data=config_data,
        env=environment,
        description="Guidance for Open Source 3D Reconstruction Toolbox for Gaussian Splats on AWS (SO9142)"
    )

# Always include post-deploy stack in the app definition, even during destroy
# Handle cases where bundling_stacks is None, empty list, or contains the stack name
if select_all or bundling_stacks is None or len(bundling_stacks) == 0 or (bundling_stacks and "GSWorkflowPostDeployStack" in bundling_stacks):
        print("Post-deploy stack condition is TRUE")
        try:
            print("Creating post-deploy stack...")
            outputs_path = os.path.join(current_path, "outputs.json")
            print(f"Looking for outputs file at: {outputs_path}")
            print(f"File exists: {os.path.exists(outputs_path)}")
            
            # Try to read existing outputs
            with open(outputs_path, "r") as f:
                output_data = json.load(f)
                print(f"Successfully loaded outputs data: {list(output_data.keys()) if output_data else 'empty'}")
            
            post_deploy_stack = GSWorkflowPostDeployStack(
                scope=app,
                id="GSWorkflowPostDeployStack",
                config_data=config_data,
                output_json_path=outputs_path,
                build_args=build_args,
                dockerfile_path=os.path.join(current_path, "../../source/container"),
                env=environment,
                description="Guidance for Open Source 3D Reconstruction Toolbox for Gaussian Splats on AWS (SO9142)"
            )
            print("Post-deploy stack created successfully")
            
            if 'base_stack' in locals():
                post_deploy_stack.add_dependency(base_stack)
                print("Added dependency on base stack")
        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: Could not create post-deploy stack due to missing outputs: {e}")
        except Exception as e:
            print(f"Error creating post-deploy stack: {str(e)}")
else:
    print("Post-deploy stack condition is FALSE")

app.synth()
