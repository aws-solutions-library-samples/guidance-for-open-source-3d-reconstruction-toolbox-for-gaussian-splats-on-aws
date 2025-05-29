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

"""Main construct to build an SNS Topic"""

from constructs import Construct
from aws_cdk import (
    aws_sns as sns,
    aws_sns_subscriptions as subscriptions
)

class Sns(Construct):
    """Class for SNS Construct"""
    def __init__(
            self,
            scope: Construct,
            id: str,
            admin_email: str,
            **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create the SNS topic with default encryption
        self.sns_topic = sns.Topic(  # Changed from self.topic to self.sns_topic
            self,
            "NotificationTopic",
            display_name="3DGS Workflow Notifications"
        )

        # Add email subscription
        self.sns_topic.add_subscription(  # Updated to use sns_topic
            subscriptions.EmailSubscription(admin_email)
        )
