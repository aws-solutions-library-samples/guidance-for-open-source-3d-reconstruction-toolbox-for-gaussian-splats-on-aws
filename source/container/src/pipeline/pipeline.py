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

""" This script has the main classes for running a pipeline """

import os
import sys
import logging
import enum
import subprocess
import traceback
import shlex
from typing import List
from rich.logging import RichHandler

class ComponentType(enum.Enum):
    loader = 0
    filter = 1
    transform = 2
    renderer = 3
    exporter = 4

class ComponentEnvironment(enum.Enum):
    executable = 0
    python = 1

class Status(enum.Enum):
    ERROR = 0
    INIT = 1
    RUNNING = 2
    STOP = 3

class Component:
    """
    There can be multiple components of each type in a pipeline
    """
    def __init__(
            self,
            name: str,
            comp_type: ComponentType,
            comp_environ: ComponentEnvironment,
            command: str,
            args: List[str],
            cwd: str,
            requires_gpu: bool) -> None:
        self.name = name
        self.comp_type = comp_type
        self.comp_environ = comp_environ
        self.command = command
        self.args = args
        self.cwd = cwd
        self.requires_gpu = requires_gpu
    name: str
    comp_type: ComponentType
    comp_environ: ComponentEnvironment
    command: str
    args: List[str]
    cwd: str
    requires_gpu: bool

class Pipeline:
    """
    Pipeline is the main parent class to hold data pertaining to a processing session
    """
    def __init__(
            self,
            name: str,
            uuid: str,
            num_threads: int,
            num_gpus: int,
            log_verbosity: str) -> None:
        # Initialize class Members
        self.reset_class()
        self.config = self.Config(name, num_threads, num_gpus)
        self.session = self.Session(uuid)
        # Setup logging
        level = logging.INFO
        if log_verbosity.lower() == "debug":
            level = logging.DEBUG
        elif log_verbosity.lower() == "error":
            level = logging.ERROR
        logging.basicConfig(
            level = level, 
            format = "%(asctime)s %(message)s",
            datefmt = "%m/%d/%Y %I:%M:%S %p",
            handlers = [
                RichHandler(),
                logging.FileHandler(f"{self.config.name}-pipeline-log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    class Config:
        def __init__(
                self,
                name: str,
                num_threads: int,
                num_gpus: int) -> None:
            self.name = name
            self.num_threads = num_threads
            self.num_gpus = num_gpus
        name: str
        num_components: int
        num_threads: str
        num_gpus: str
    class Session:
        def __init__(
                self,
                uuid: str) -> None:
            self.log = logging.getLogger()
            self.uuid = uuid
        uuid: str
        status: dict
        current_comp_step: int
        log: logging
        progress: int
    config: Config
    session: Session
    components: List[Component]

    def reset_class(self)->None:
        """
        Reset the class, happens automatically on initialization
        """
        # Config
        self.Config.name = ""
        self.Config.num_components = 0
        self.Config.num_threads = "0"
        self.Config.num_gpus = "0"
        # Session
        self.Session.current_comp_step = 0
        self.Session.progress = 0
        self.Session.status = (0, Status.STOP) # Response code, status message
        self.Session.uuid = None
        self.components = []

    def create_component(
            self,
            name: str,
            comp_type: ComponentType,
            comp_environ: ComponentEnvironment,
            command: str,
            args: List[str],
            cwd: str,
            requires_gpu: bool)->None:
        """
        Create a pipeline component
        """
        component = Component(
            name=name,
            comp_type=comp_type,
            comp_environ=comp_environ,
            command=command,
            args=args,
            cwd=cwd,
            requires_gpu=requires_gpu)
        self.components.append(component)
        self.Config.num_components = self.Config.num_components + 1

    def run_component(self, index: int)->None:
        """
        Run a pipeline component
        """
        self.session.log.info(f"Running component {self.components[index].name}")
        
        # Create a copy of the current environment
        env = os.environ.copy()
        
        # Set up the command and arguments
        cmd_args = []
        
        # Add the command
        if self.components[index].comp_environ == ComponentEnvironment.python:
            cmd_args.append(sys.executable)
        cmd_args.append(self.components[index].command)
        
        # Add all arguments, handling environment variables specially
        for arg in self.components[index].args:
            # Validate argument is a string to prevent injection
            if not isinstance(arg, str):
                raise ValueError(f"Invalid argument type: {type(arg)}")
            
            # Additional validation: check for dangerous characters
            if any(char in arg for char in ['`', '$', '|', '&', ';', '>', '<']):
                raise ValueError(f"Potentially dangerous characters in argument: {arg}")
            
            if '=' in arg and arg.split('=')[0] == 'CUDA_VISIBLE_DEVICES':
                # This is an environment variable setting, not an argument
                env_value = arg.split('=')[1]
                # Validate environment variable value
                if not env_value.replace(',', '').replace(' ', '').isdigit():
                    raise ValueError(f"Invalid CUDA_VISIBLE_DEVICES value: {env_value}")
                env['CUDA_VISIBLE_DEVICES'] = env_value
            else:
                # Add argument directly - no shell escaping needed for subprocess.run() with list
                cmd_args.append(arg)
                
        self.session.log.info(f"Component command: {cmd_args}")
        if 'CUDA_VISIBLE_DEVICES' in env:
            self.session.log.info(f"Using CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
            
        try:
            # Subprocess call is secure: uses list format (not shell=True) with validated arguments
            result = subprocess.run(  # nosemgrep: dangerous-subprocess-use-audit
                cmd_args,
                check=True,
                cwd=self.components[index].cwd,
                env=env
            )
            self.session.log.info(result.stdout)
            if result.stderr:
                self.session.log.error(result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Command '{' '.join(e.cmd)}' failed with return code {e.returncode}")
            if e.stderr is not None:
                print(f"Error message: {e.stderr.strip()}")
            if e.stdout is not None:
                print(f"Output (if any): {e.stdout.strip()}")
            sys.exit(1)
        except Exception as e:
            self.session.log.error(f"An unexpected error occurred: {str(e)}")
            sys.exit(1)

    def report_error(self, response_code: int, message:str):
        """
        Report the latest error
        """
        self.session.status = (response_code, Status.ERROR)
        error = message+": "+traceback.format_exc()
        self.session.log.error(error)
        sys.exit(1)
