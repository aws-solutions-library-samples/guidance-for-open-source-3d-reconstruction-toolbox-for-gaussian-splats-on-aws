# Local Debug Mode

This document explains how to run the 3D reconstruction with Gaussian Splatting container locally using the `LOCAL_DEBUG` mode.

## Overview

When `LOCAL_DEBUG` is enabled either as an environment variable or in the `source/container/src/config.json` file, the container uses local filesystem directories instead of S3 for input/output operations. This allows you to test and debug the pipeline locally without AWS infrastructure.

## Hardware Requirements

- **GPU**: CUDA-capable GPU with 24GB VRAM or more
- **Memory**: 64GB RAM or more
- **Storage**: 50GB free disk space minimum
- **Docker**: GPU support enabled (NVIDIA Container Toolkit with `nvidia-container-runtime`)

### Directory Structure

When `LOCAL_DEBUG` is enabled, the following local directories are used (default mount point: `/mnt/data`):

```
/mnt/data/
├── workflow-input/     # Replaces S3_INPUT - place your input media here
├── workflow-output/    # Replaces S3_OUTPUT - final outputs go here
└── models/             # Model files (u2net, vocab tree, etc.)
```

**Note:** `CODE_PATH` remains at `/opt/ml/code` (inside container) and does not change in local debug mode.

## Configuration

### Enable Local Debug Mode

- Option 1: Set `LOCAL_DEBUG=true` inside of `source/container/src/config.json` file

- Option 2: Set `LOCAL_DEBUG=true` as an environment variable:

  ```bash
  export LOCAL_DEBUG=true
  ```

- Option 3: Set `LOCAL_DEBUG=true` as an environment variable running the docker run command (see below)

> The `config.json` file provides defaults, but **environment variables override** config.json values. Feel free to override pipeline defaults using the environment variables.

## Building the Docker Container
> *NOTE: Any changes to the `config.json` or **base code** should follow with rebuilding the container to observe new changes to the container*
Before running locally, build the container image:

```bash
# Navigate to the container directory
cd source/container

# Build the Docker image
docker build -t 3dgs-container:latest .
```

This will create a container image tagged as `3dgs-container:latest` that includes all necessary dependencies for 3D reconstruction with Gaussian Splats.

## Running Locally with Docker

**Local debug mode uses environment variables** (same as SageMaker/Batch):

```bash
docker run -it \
  -e LOCAL_DEBUG=true \
  -e UUID=test-job-001 \
  -e FILENAME=input.mp4 \
  -v <YOUR-LOCAL-PATH>/guidance-for-open-source-3d-reconstruction-toolbox-for-gaussian-splats-on-aws/source/container:/mnt/data \
  -v <YOUR-LOCAL-PATH>/guidance-for-open-source-3d-reconstruction-toolbox-for-gaussian-splats-on-aws/source/container/models:/opt/ml/input/data/model \
  --entrypoint python \
  --runtime nvidia \
  --device=/dev/nvidia0:/dev/nvidia0 \
  --device=/dev/nvidiactl:/dev/nvidiactl \
  --device=/dev/nvidia-modeset:/dev/nvidia-modeset \
  --device=/dev/nvidia-uvm:/dev/nvidia-uvm \
  --device=/dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
  --device=/dev/nvidia-caps:/dev/nvidia-caps \
  --shm-size=8g \
  3dgs-container:latest \
  /opt/ml/code/main.py
```

> *NOTE: For a full list of pipeline environment variables, please see `source/container/src/config.json`*

### Complete Example

```bash
# Create local directories
cd /<YOUR-LOCAL-PATH>/guidance-for-open-source-3d-reconstruction-toolbox-for-gaussian-splats-on-aws/source/container
mkdir -p workflow-input workflow-output models

# Copy your input media (or folder of images)
cp your-video.mov workflow-input/

# Copy models (if not already in container)
# This is the models.tar.gz gzip file that gets built with CDK/Terraform
# If not deployed yet, see the code in
# /mnt/efs/guidance-for-open-source-3d-reconstruction-toolbox-for-gaussian-splats-on-aws/source/lambda/model_deployment
# to download and archive the model files locally.
cp -r <YOUR-LOCAL-PATH>/models/* models/

# Run container
# Use environment variables to override pipeline config.json file parameters
docker run -it \
  -e LOCAL_DEBUG=true \
  -e UUID=test-job-001 \
  -e FILENAME=input.mov \
  -v $(pwd):/mnt/data \
  -v $(pwd)/models:/opt/ml/input/data/model \
  --entrypoint python \
  --runtime nvidia \
  --device=/dev/nvidia0:/dev/nvidia0 \
  --device=/dev/nvidiactl:/dev/nvidiactl \
  --device=/dev/nvidia-modeset:/dev/nvidia-modeset \
  --device=/dev/nvidia-uvm:/dev/nvidia-uvm \
  --device=/dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
  --device=/dev/nvidia-caps:/dev/nvidia-caps \
  --shm-size=8g \
  3dgs-container:latest \
  /opt/ml/code/main.py

# Check results
ls -lh $(pwd)/workflow-output/job-*/
```

### Example Commands

#### Image Directory Input, Set SfM, Set Model, Set Max Steps
```bash
docker run -it \
  -e LOCAL_DEBUG=true \
  -e UUID=amazon-sphere-images-22k-splatfacto \
  -e FILENAME=amazon-sphere-images \
  -e RECON_SOFTWARE_NAME=colmap \
  -e MODEL=splatfacto \
  -e THREED_ISP=bilagrid \
  -e PRESERVE_SCENE_SCALE=true \
  -e MAX_STEPS=22000 \
  -v $(pwd):/mnt/data \
  -v $(pwd)/models:/opt/ml/input/data/model \
  --entrypoint python \
  --runtime nvidia \
  --device=/dev/nvidia0:/dev/nvidia0 \
  --device=/dev/nvidiactl:/dev/nvidiactl \
  --device=/dev/nvidia-modeset:/dev/nvidia-modeset \
  --device=/dev/nvidia-uvm:/dev/nvidia-uvm \
  --device=/dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
  --device=/dev/nvidia-caps:/dev/nvidia-caps \
  --shm-size=8g \
  3dgs-container:latest \
  /opt/ml/code/main.py

```

#### Resume Model Input, Set Max Steps, Set Export Options
```bash
docker run -it \
  -e LOCAL_DEBUG=true \
  -e UUID=amazon-sphere-images-30k-splatfacto \
  -e FILENAME=model.tar.gz \
  -e RUN_RECON=false \
  -e MAX_STEPS=30000 \
  -e ENABLE_USDZ=false \
  -e ENABLE_SPZ=false \
  -e CROP_OUTPUT_BOUNDS=true \
  -e CROP_MODE=large_scale \
  -v $(pwd):/mnt/data \
  -v $(pwd)/models:/opt/ml/input/data/model \
  --entrypoint python \
  --runtime nvidia \
  --device=/dev/nvidia0:/dev/nvidia0 \
  --device=/dev/nvidiactl:/dev/nvidiactl \
  --device=/dev/nvidia-modeset:/dev/nvidia-modeset \
  --device=/dev/nvidia-uvm:/dev/nvidia-uvm \
  --device=/dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
  --device=/dev/nvidia-caps:/dev/nvidia-caps \
  --shm-size=8g \
  3dgs-container:latest \
  /opt/ml/code/main.py

```


## How It Works

### Input Processing

Input files are read from `${LOCAL_MOUNT}/workflow-input/`

### Output Handling

Instead of uploading to S3, outputs are copied to:
```
${LOCAL_MOUNT}/workflow-output/${UUID}/
├── video-name.ply
├── video-name.spz
├── video-name.sog
├── video-name.usdz
├── video-name.mp4 (rendered video)
├── render_thumbnail.png
└── output/
    └── model.tar.gz (for resume training)
```

## Differences from AWS Mode

| Feature | AWS Mode | Local Debug Mode |
|---------|----------|------------------|
| Input Source | S3 bucket | Local directory |
| Output Destination | S3 bucket | Local directory |
| DATASET_PATH | `/opt/ml/input/data/train` | `${LOCAL_MOUNT}/workflow-input` |
| CODE_PATH | `/opt/ml/code` | `/opt/ml/code` (unchanged) |
| AWS Credentials | Required | Not required |
| Batch/SageMaker | Supported | Runs standalone |
| S3 downloads | Automatic | Manual file placement |
| Entrypoint | Set by SageMaker/Batch | Use `--entrypoint` flag |

## Troubleshooting

### CUDA "No Device" Errors During Pipeline

If you see `CUDA error: no CUDA-capable device is detected` or `RuntimeError: No CUDA GPUs are available` at various pipeline stages (SfM, training, export), the issue is typically how the GPU is passed through to the container. Using `--gpus all` can intermittently fail to expose NVIDIA device nodes to subprocesses.

**Option 1 (Recommended):** Use `--runtime nvidia` with explicit device passthrough as shown in the examples above.

If you get `unknown or invalid runtime name: nvidia`, register the runtime first:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Option 2:** Run the container in privileged mode (less secure, but simplest fix):
```bash
docker run -it --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  ...
```

**Option 3:** If you must use `--gpus all`, ensure your `/etc/docker/daemon.json` has the nvidia runtime configured as default:
```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
```
Then restart Docker: `sudo systemctl restart docker`

### GPU Driver Not Found Error

If you see `could not select device driver "" with capabilities: [[gpu]]`, install NVIDIA Container Toolkit:

```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --runtime nvidia --device=/dev/nvidia0:/dev/nvidia0 --device=/dev/nvidiactl:/dev/nvidiactl --device=/dev/nvidia-uvm:/dev/nvidia-uvm nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### Permission Issues

Ensure the mounted directory has proper permissions:
```bash
chmod -R 777 /tmp/gs-local
```

### Missing Models

If models are not found, ensure they're in `${LOCAL_MOUNT}/models/` or already bundled in the container at `/opt/ml/input/data/model/`.

### Output Not Found

Check that:
1. `LOCAL_DEBUG=true` is set
2. `UUID` environment variable is set
3. Output directory exists: `${LOCAL_MOUNT}/workflow-output/${UUID}/`


## Notes
- Local debug mode is intended for development and testing
- GPU is still required for training and certain processing steps
- Currently tested on Linux (Ubuntu 22.04)