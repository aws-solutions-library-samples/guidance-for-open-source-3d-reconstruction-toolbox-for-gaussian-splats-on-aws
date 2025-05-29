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

"""Main construct to build DynamoDB database"""

from aws_cdk import (
    aws_dynamodb as ddb,
    RemovalPolicy,
    Environment
)
from constructs import Construct
from typing import Optional, Dict, Any

class Ddb(Construct):
    """Class for DynamoDB Construct"""
    def __init__(
            self,
            scope: Construct,
            id: str,
            env: Environment,
            ddb_table_name: str,
            partition_key: str,
            sort_key: str,
            billing_mode: ddb.BillingMode,
            removal_policy: RemovalPolicy,
            provisioned_throughput: Optional[Dict[str, int]] = None,
            **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Default provisioned throughput if not specified
        default_throughput = {
            'read_capacity': 5,
            'write_capacity': 5,
            'min_capacity': 5,
            'max_capacity': 100,
            'target_utilization': 70
        }

        # Use provided throughput values or defaults
        if provisioned_throughput:
            throughput = {**default_throughput, **provisioned_throughput}
        else:
            throughput = default_throughput

        # Base table configuration
        table_props = {
            'table_name': ddb_table_name,
            'partition_key': {
                'name': partition_key,
                'type': ddb.AttributeType.STRING
            },
            'billing_mode': billing_mode,
            'removal_policy': removal_policy,
            'point_in_time_recovery': True,
        }

        # Add sort key if provided
        if sort_key:
            table_props['sort_key'] = {
                'name': sort_key,
                'type': ddb.AttributeType.STRING
            }

        # Add read and write capacity if using PROVISIONED mode
        if billing_mode == ddb.BillingMode.PROVISIONED:
            table_props['read_capacity'] = throughput['read_capacity']
            table_props['write_capacity'] = throughput['write_capacity']

        # Create DynamoDB table
        self.table = ddb.Table(
            self,
            "AssetTable",
            **table_props
        )

        # Enable auto scaling for the base table if using PROVISIONED mode
        if billing_mode == ddb.BillingMode.PROVISIONED:
            self._configure_auto_scaling(
                min_capacity=throughput['min_capacity'],
                max_capacity=throughput['max_capacity'],
                target_utilization=throughput['target_utilization']
            )

        # Add GSI configuration
        gsi_props = {
            'index_name': "uuid-index",
            'partition_key': {
                'name': partition_key,
                'type': ddb.AttributeType.STRING
            }
        }

        # Add capacity settings for GSI if using PROVISIONED mode
        if billing_mode == ddb.BillingMode.PROVISIONED:
            gsi_props['read_capacity'] = throughput['read_capacity']
            gsi_props['write_capacity'] = throughput['write_capacity']

        # Add GSI
        gsi = self.table.add_global_secondary_index(**gsi_props)

        # Enable auto scaling for the GSI if using PROVISIONED mode
        if billing_mode == ddb.BillingMode.PROVISIONED:
            self._configure_gsi_auto_scaling(
                index_name="uuid-index",
                min_capacity=throughput['min_capacity'],
                max_capacity=throughput['max_capacity'],
                target_utilization=throughput['target_utilization']
            )

    def _configure_auto_scaling(
            self,
            min_capacity: int,
            max_capacity: int,
            target_utilization: int):
        """Configure auto scaling for the base table"""
        write_scaling = self.table.auto_scale_write_capacity(
            min_capacity=min_capacity,
            max_capacity=max_capacity
        )
        read_scaling = self.table.auto_scale_read_capacity(
            min_capacity=min_capacity,
            max_capacity=max_capacity
        )

        write_scaling.scale_on_utilization(
            target_utilization_percent=target_utilization
        )
        read_scaling.scale_on_utilization(
            target_utilization_percent=target_utilization
        )

    def _configure_gsi_auto_scaling(
            self,
            index_name: str,
            min_capacity: int,
            max_capacity: int,
            target_utilization: int):
        """Configure auto scaling for GSI"""
        write_scaling = self.table.auto_scale_global_secondary_index_write_capacity(
            index_name=index_name,
            min_capacity=min_capacity,
            max_capacity=max_capacity
        )
        read_scaling = self.table.auto_scale_global_secondary_index_read_capacity(
            index_name=index_name,
            min_capacity=min_capacity,
            max_capacity=max_capacity
        )

        write_scaling.scale_on_utilization(
            target_utilization_percent=target_utilization
        )
        read_scaling.scale_on_utilization(
            target_utilization_percent=target_utilization
        )
