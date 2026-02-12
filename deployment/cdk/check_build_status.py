#!/usr/bin/env python3

import json
import sys
import time
import boto3
from pathlib import Path
from datetime import datetime

# Load config
config_path = Path(__file__).parent / "config.json"
with open(config_path) as f:
    config = json.load(f)

region = config["region"]

# Load outputs
outputs_path = Path(__file__).parent / "outputs.json"
if not outputs_path.exists():
    print("❌ outputs.json not found. Deploy base stack first.")
    sys.exit(1)

with open(outputs_path) as f:
    outputs = json.load(f)

ecr_repo = outputs["GSWorkflowBaseStack"]["ECRRepoName"]
project_name = f"docker-build-{ecr_repo}"

print(f"Checking CodeBuild project: {project_name}")
print(f"Region: {region}\n")

# Initialize boto3 client
codebuild = boto3.client("codebuild", region_name=region)

# Get latest build
try:
    response = codebuild.list_builds_for_project(projectName=project_name, sortOrder="DESCENDING")
    if not response["ids"]:
        print(f"❌ No builds found for project {project_name}")
        sys.exit(1)
    
    build_id = response["ids"][0]
    print(f"Latest Build ID: {build_id}\n")
    
    start_time = time.time()
    
    # Poll status
    while True:
        build_info = codebuild.batch_get_builds(ids=[build_id])["builds"][0]
        status = build_info["buildStatus"]
        phase = build_info.get("currentPhase", "N/A")
        
        elapsed = int(time.time() - start_time)
        elapsed_str = f"{elapsed // 60}m {elapsed % 60}s"
        
        print(f"Status: {status} | Phase: {phase} | Elapsed: {elapsed_str}")
        
        if status == "SUCCEEDED":
            total_time = int(time.time() - start_time)
            total_str = f"{total_time // 60}m {total_time % 60}s"
            print(f"\n✅ Build completed successfully!")
            print(f"Image: {ecr_repo}:latest")
            print(f"Total time: {total_str}")
            sys.exit(0)
        elif status in ["FAILED", "FAULT", "STOPPED", "TIMED_OUT"]:
            total_time = int(time.time() - start_time)
            total_str = f"{total_time // 60}m {total_time % 60}s"
            print(f"\n❌ Build failed with status: {status}")
            if "logs" in build_info and "deepLink" in build_info["logs"]:
                print(f"View logs: {build_info['logs']['deepLink']}")
            print(f"Total time: {total_str}")
            sys.exit(1)
        
        time.sleep(30)

except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
