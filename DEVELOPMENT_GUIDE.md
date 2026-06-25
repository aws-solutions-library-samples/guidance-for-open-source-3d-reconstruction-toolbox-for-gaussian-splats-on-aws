# Development Steering Guide

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Codebase Structure](#codebase-structure)
4. [Adding a New Pipeline Feature — End-to-End Checklist](#adding-a-new-pipeline-feature--end-to-end-checklist)
5. [Coding Standards and Rules](#coding-standards-and-rules)
6. [Parameter Propagation Reference](#parameter-propagation-reference)
7. [Documentation Rules](#documentation-rules)
8. [Container Dependency Rules](#container-dependency-rules)
9. [Local Debug Mode](#local-debug-mode)
10. [Error Code Registry](#error-code-registry)

---

## Repository Overview

This guidance deploys a fully automated, GPU-accelerated 3D reconstruction pipeline on AWS. A user uploads a video or image archive to S3; the system automatically runs Structure-from-Motion (SfM), Gaussian Splat training, and post-processing, then delivers the finished 3D assets back to S3 and notifies the user via email.

The pipeline is entirely containerized. All reconstruction logic lives inside a single Docker container (`source/container/`). The AWS infrastructure (Lambda, Step Functions, SageMaker/Batch, DynamoDB, SNS) is responsible only for orchestrating that container — it does not contain any 3D reconstruction logic itself.

---

## Architecture Deep Dive

### End-to-End Data Flow

```
User
 │
 ├─ uploads media (.mp4/.mov/.zip) + job JSON  ──► S3 (media-input/)
 │
 └─ S3 PUT event on workflow-input/*.json
         │
         ▼
    SNS Topic
         │
         ▼
    workflow_trigger Lambda
    ├─ validates job JSON schema
    ├─ writes/updates DynamoDB job record
    ├─ maps JSON fields → container env vars
    └─ starts Step Functions State Machine
              │
              ▼
    Step Functions (ASLdefinition.json)
    ├─ CheckComputeType
    │   ├─ useSpotInstance=true  → SetDefaultJobDefinition → InvokeBatchWorkflow (AWS Batch / Spot)
    │   └─ useSpotInstance=false → InvokeSageMakerWorkflow (SageMaker On-Demand)
    │
    ├─ [job runs synchronously — state machine waits]
    │
    ├─ SuccessHandler / ErrorHandler
    └─ Complete → workflow_complete Lambda
                  ├─ updates DynamoDB (status, timing, metrics)
                  └─ sends SNS email notification
              │
              ▼
    Container (ECR image)
    ├─ main.py  ← single entry point
    │   ├─ INITIALIZATION  (env vars, model extraction, pipeline setup)
    │   ├─ COMPONENT CREATION  (ordered try/except blocks, one per pipeline stage)
    │   └─ PIPELINE EXECUTION  (match/case loop over components)
    │
    ├─ pipeline/pipeline.py   ← Pipeline and Component classes
    ├─ pipeline/utils.py      ← shared utility functions
    └─ pipeline/<type>/<script>.py  ← individual component scripts
              │
              ▼
    S3 (workflow-output/<uuid>/)
    ├─ <name>.ply / .spz / .sog / .usdz
    ├─ render.mp4 / render_thumbnail.png
    ├─ output/model.tar.gz  (checkpoint archive for resume training)
    └─ eval/metrics.json
```

### Key Architectural Contracts

| Layer | Responsibility | Must NOT |
|---|---|---|
| Job JSON | Declare user intent | Contain AWS resource IDs |
| workflow_trigger Lambda | Validate JSON, map fields to env vars, start state machine | Contain pipeline logic |
| Step Functions | Route to SageMaker or Batch, wait for completion | Know about 3D reconstruction |
| Container `main.py` | Read env vars, build component list, execute pipeline | Call AWS APIs directly (use utils.py) |
| Component scripts | Perform one discrete processing task | Read env vars directly — receive args only via CLI |
| `utils.py` | Provide reusable container utilities | Contain pipeline orchestration logic |

### Compute Routing

The state machine routes jobs based on `useSpotInstance`:

- `false` → **SageMaker Training Job** (on-demand, no queue, higher cost)
- `true` → **AWS Batch** (Spot instances, potential queue wait, lower cost)

Both paths pass identical environment variables to the container. `main.py` detects which path it is running on via `IS_BATCH = 'AWS_BATCH_JOB_ID' in os.environ` and handles S3 download differences accordingly.

---

## Codebase Structure

```
source/
├── container/
│   ├── Dockerfile                        # Container build definition
│   ├── requirements.txt                  # Python dependencies
│   ├── LOCAL_DEBUG_README.md             # Local debug instructions
│   └── src/
│       ├── config.json                   # Default env var values
│       ├── main.py                       # Pipeline entry point
│       └── pipeline/
│           ├── pipeline.py               # Pipeline, Component, Status classes
│           ├── utils.py                  # Shared utility functions
│           ├── pre_processing/
│           │   ├── segmentation/         # Background/object removal scripts
│           │   └── video/                # Video-to-image extraction scripts
│           ├── reconstruction/           # SfM scripts (colmap, glomap, hloc, map_anything)
│           │   └── spherical/            # 360-camera SfM
│           ├── training/                 # Training wrappers and export scripts
│           └── post_processing/          # PLY manipulation, format conversion, S3 export
├── lambda/
│   ├── workflow_trigger/                 # Triggered by S3 PUT on job JSON
│   ├── workflow_complete/                # Triggered at end of state machine
│   └── job_definition_selector/         # Selects Batch job definition by instance type
├── state-machines/
│   └── ASLdefinition.json               # Step Functions ASL definition
├── Gradio/
│   └── generate_splat_gradio.py         # Gradio UI — mirrors job JSON fields
├── generate_splat.py                    # CLI job submission utility
└── readme.md                            # Backend submission guide
```

### `main.py` Internal Structure

`main.py` is divided into three clearly separated zones. **Do not mix concerns across zones.**

```
main.py
│
├── ZONE 1 — INITIALIZATION  (top of file)
│   ├─ Import statements
│   ├─ Environment variable loading via load_config()
│   ├─ Path constants (OUTPUT_TAR_PATH, EVAL_METRIC_PATH, etc.)
│   ├─ Feature flags derived from config (IS_BATCH, LOCAL_DEBUG, etc.)
│   ├─ AWS Batch S3 download (if IS_BATCH)
│   ├─ Model archive extraction
│   └─ Pipeline instantiation
│
├── ZONE 2 — COMPONENT CREATION  (middle of file)
│   ├─ One try/except block per pipeline stage
│   ├─ Each block calls pipeline.create_component(...)
│   ├─ Components are added in execution order
│   └─ Conditional logic determines which components are added
│
└── ZONE 3 — PIPELINE EXECUTION  (bottom of file)
    ├─ pipeline.session.status = Status.RUNNING
    ├─ for loop over pipeline.components
    ├─ match/case on component.name for special handling
    ├─ Default case: pipeline.run_component(i)
    └─ DynamoDB metrics update on completion
```

---

## Adding a New Pipeline Feature — End-to-End Checklist

Follow every step in order. Each step references the exact file to modify.

### Step 1 — Design the Job JSON Parameter

Decide the JSON key name, type, default value, and which section it belongs to (`videoProcessing`, `reconstruction`, `training`, `postProcessing`, `segmentation`, or a new section).

**Naming convention:** camelCase for JSON keys, UPPER_SNAKE_CASE for env vars.

Example: JSON key `myNewFeature` → env var `MY_NEW_FEATURE`.

---

### Step 2 — Update `source/generate_splat.py`

Add the new parameter to the `file_contents` dict with its default value. This is the canonical reference for the job JSON schema.

```python
"training": {
    ...
    "myNewFeature": "false",   # Add here with default
},
```

---

### Step 3 — Update `workflow_trigger` Lambda

File: `source/lambda/workflow_trigger/workflow_trigger.py`

**3a.** Add the field to `validate_config`'s `required_dict_props` dict under the correct section so missing fields are caught at submission time.

**3b.** Add the field to the DynamoDB `item` dict (the `else` branch that creates a new record):

```python
"myNewFeature": str(json_content["training"].get("myNewFeature", "false")),
```

**3c.** Add the env var mapping to `inputObj["envVars"]`:

```python
"MY_NEW_FEATURE": str(json_content["training"].get("myNewFeature", "false")),
```

---

### Step 4 — Update the Step Functions State Machine

File: `source/state-machines/ASLdefinition.json`

Add the env var to **both** compute paths so the container receives it regardless of whether SageMaker or Batch is used.

**In `InvokeBatchWorkflow` → `ContainerOverrides.Environment`:**
```json
{"Name": "MY_NEW_FEATURE", "Value.$": "$.envVars.MY_NEW_FEATURE"}
```

**In `InvokeSageMakerWorkflow` → `Environment`:**
```json
"MY_NEW_FEATURE.$": "$.envVars.MY_NEW_FEATURE"
```

---

### Step 5 — Update `config.json` (Container Defaults)

File: `source/container/src/config.json`

Add a default value so the container works in local debug mode without the env var being set:

```json
"MY_NEW_FEATURE": "false"
```

---

### Step 6 — Write the Component Script

Create a new script under the appropriate subdirectory of `source/container/src/pipeline/`:

| Feature type | Directory |
|---|---|
| Video/image pre-processing | `pipeline/pre_processing/video/` |
| Segmentation | `pipeline/pre_processing/segmentation/` |
| SfM / reconstruction | `pipeline/reconstruction/` |
| Training | `pipeline/training/` |
| Post-processing / export | `pipeline/post_processing/` |

**Every new script must:**

1. Include the MIT license header (see [Coding Standards](#coding-standards-and-rules))
2. Include a one-sentence module docstring describing what the script does
3. Use `argparse` for all inputs — never read env vars directly
4. Have a comment on every function explaining what it does
5. Be self-contained and runnable as a standalone CLI tool

---

### Step 7 — Add Utility Functions to `utils.py`

File: `source/container/src/pipeline/utils.py`

If the new feature requires helper logic that could be reused across scripts or needs access to AWS clients, add it to `utils.py`. Do **not** put utility functions directly in `main.py` or in component scripts.

Every function added to `utils.py` must have a comment explaining what it does.

---

### Step 8 — Register the Component in `main.py`

File: `source/container/src/main.py`

**8a. Read the config value** — `load_config()` automatically reads from env vars using the key names in `config.json`. Access it as `config['MY_NEW_FEATURE']`.

**8b. Add a feature flag** (if needed) near the top of the initialization zone:

```python
ENABLE_MY_NEW_FEATURE = config.get('MY_NEW_FEATURE', 'false').lower() == 'true'
```

**8c. Add a component creation block** in Zone 2, in the correct pipeline order. Follow the existing pattern exactly:

```python
##################################
# <COMPONENT_TYPE> COMPONENT:
# Brief description of what this does
##################################
try:
    if ENABLE_MY_NEW_FEATURE:
        args = [
            "-i", input_path,
            "-o", output_path,
        ]
        pipeline.create_component(
            name="My-New-Feature",
            comp_type=ComponentType.PRE_PROCESSING,   # or RECONSTRUCTION, TRAINING, POST_PROCESSING
            comp_environ=ComponentEnvironment.PYTHON,  # or EXECUTABLE
            command="pre_processing/my_new_feature.py",
            args=args,
            cwd=current_dir_path,
            requires_gpu=False  # True only if the script uses CUDA
        )
except Exception as e:
    error_message = f"Issue creating my new feature component: {e}"
    pipeline.report_error(8XX, error_message)  # Assign a new error code from the registry
```

**8d. Add a match/case handler** in Zone 3 if the component needs special pre/post-run logic (file moves, conditional skipping, etc.). If it just needs to run, the default `case _:` handler will call `pipeline.run_component(i)` automatically.

---

### Step 9 — Update the Gradio UI (if user-facing)

File: `source/Gradio/generate_splat_gradio.py`

Add a UI control for the new parameter and wire it into the job JSON construction function. Follow the existing pattern for the section the parameter belongs to.

---

### Step 10 — Update Documentation

All three documentation files must be updated together. Do not update one without the others.

| File | What to add |
|---|---|
| `UNDERSTANDING_CONFIGURATION.md` | Full description of the new parameter under the correct section, including type, valid values, and behavior |
| `source/readme.md` | Add the parameter to the Job Schema Structure JSON block |
| `README.md` | If a new open-source library was added, add a row to the Custom GS Pipeline Container feature table |

If the feature changes local debug behavior, also update `source/container/LOCAL_DEBUG_README.md`.

---

### Step 11 — Update the Container (if new dependency)

If the feature requires a new Python package or system tool:

1. Add it to `source/container/requirements.txt`
2. Add the install step to `source/container/Dockerfile` if it requires system-level installation
3. Add a row to the feature table in `README.md` with the library name, license, and GPU support status (see [Container Dependency Rules](#container-dependency-rules))

---

### Step 12 — Assign an Error Code

Add a new error code to the registry at the top of `main.py` and to the [Error Code Registry](#error-code-registry) section of this document. Error codes 700–799 are reserved for container pipeline errors.

---

## Coding Standards and Rules

### MIT License Header

Every new Python script must begin with this exact header, followed by a one-sentence module docstring:

```python
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

"""One sentence describing what this script does."""
```

### Function Comments

Every function must have a comment directly above or inside it explaining what it does. For short functions, a single line is sufficient. For complex functions, describe the inputs, outputs, and any side effects.

```python
# Converts a PLY file to the target coordinate system by applying a rotation matrix.
def transform_coordinates(input_path, output_path, target):
    ...
```

### Component Script Rules

- Accept all inputs via `argparse` — never `os.environ`
- Use `sys.exit(1)` on fatal errors, not `raise` (so the pipeline captures the exit code)
- Write progress to stdout/stderr — the pipeline captures this as component output
- Keep scripts focused on one task; split into multiple scripts if needed

### `utils.py` Rules

- Only add functions that are called from more than one place, or that require AWS SDK clients
- AWS client initialization belongs in `utils.py`, not in component scripts
- All functions must have a comment

### `main.py` Zone Rules

- **Zone 1 (Initialization):** constants, config loading, path setup, feature flags only
- **Zone 2 (Component Creation):** `pipeline.create_component()` calls only — no file I/O, no subprocess calls
- **Zone 3 (Execution):** `pipeline.run_component()` calls and inter-component file operations only

### Component Ordering in `main.py`

Components must be created in this order to match the pipeline execution sequence:

1. PRE_PROCESSING — pose extraction, video-to-images, image rotation, background/object removal
2. RECONSTRUCTION — feature extraction, matching, mapping, colmap-to-nerfstudio conversion
3. TRAINING — model training, PLY export, metrics
4. POST_PROCESSING — PLY manipulation, format conversion, S3 uploads

---

## Parameter Propagation Reference

This table shows the exact field name at each layer for a parameter. Use it as a template when adding new parameters.

| Layer | Location | Field name example |
|---|---|---|
| Job JSON | `source/generate_splat.py` | `"myNewFeature": "false"` |
| Job JSON schema | `UNDERSTANDING_CONFIGURATION.md` | `myNewFeature` |
| Job JSON schema | `source/readme.md` | `"myNewFeature": "false"` |
| Lambda validation | `workflow_trigger.py` → `validate_config` | `"myNewFeature": None` |
| DynamoDB record | `workflow_trigger.py` → `item` dict | `"myNewFeature": str(...)` |
| Lambda env var map | `workflow_trigger.py` → `inputObj["envVars"]` | `"MY_NEW_FEATURE": str(...)` |
| State machine (Batch) | `ASLdefinition.json` → `ContainerOverrides.Environment` | `{"Name": "MY_NEW_FEATURE", "Value.$": "$.envVars.MY_NEW_FEATURE"}` |
| State machine (SageMaker) | `ASLdefinition.json` → `Environment` | `"MY_NEW_FEATURE.$": "$.envVars.MY_NEW_FEATURE"` |
| Container default | `config.json` | `"MY_NEW_FEATURE": "false"` |
| Container config load | `main.py` → `load_config()` | `config['MY_NEW_FEATURE']` |
| Container feature flag | `main.py` Zone 1 | `ENABLE_MY_NEW_FEATURE = config.get(...) == 'true'` |
| Component creation | `main.py` Zone 2 | `if ENABLE_MY_NEW_FEATURE: pipeline.create_component(...)` |
| Gradio UI | `generate_splat_gradio.py` | UI control + JSON builder |

---

## Documentation Rules

### When to update each file

| Change | Files to update |
|---|---|
| New job JSON parameter | `UNDERSTANDING_CONFIGURATION.md`, `source/readme.md`, `source/generate_splat.py` |
| New open-source library in container | `README.md` (feature table) |
| New container dependency (pip/apt) | `source/container/requirements.txt`, `source/container/Dockerfile`, `README.md` |
| Change to local debug behavior | `source/container/LOCAL_DEBUG_README.md` |
| New pipeline component | `UNDERSTANDING_CONFIGURATION.md` (if user-configurable), `source/readme.md` |

### `UNDERSTANDING_CONFIGURATION.md` format

New parameters must follow the existing format:

```markdown
- **My new feature:** (type), description of what it does and when to use it.
    - Valid values: `"value1"`, `"value2"`
    - Default: `"false"`
```

### `README.md` feature table format

New open-source libraries must be added as a row in the Custom GS Pipeline Container table:

```markdown
| Category | Component description | [Library name](url) [(License)](license-url) | GPU? | Notes |
```

---

## Container Dependency Rules

When adding a new open-source library to the container:

1. **Check the license** — only Apache-2.0, MIT, BSD, and compatible licenses are acceptable. GPL is not acceptable.
2. **Add to `requirements.txt`** with a pinned version: `my-library==1.2.3`
3. **Add to `Dockerfile`** if system packages are required (apt-get, cmake builds, etc.)
4. **Add to `README.md`** feature table with: category, component name, library link, license link, GPU support, and a brief note
5. **Document the model files** in `source/container/LOCAL_DEBUG_README.md` if the library requires pre-downloaded model weights that must be included in `models.tar.gz`

---

## Local Debug Mode

Local debug mode allows running the full pipeline on a local GPU machine (e.g. an EC2 instance) without SageMaker or Batch. It is activated by setting `LOCAL_DEBUG=true` in the environment.

Key behaviors in local debug mode (controlled in `main.py` and `utils.py`):

- S3 downloads are skipped; input media is read from the local filesystem
- S3 uploads are replaced by local file copies to a `workflow-output/` directory
- The COLMAP database is placed in `/tmp/` to avoid SQLite locking issues on EFS mounts
- A `model.tar.gz` archive is created locally instead of being uploaded to S3
- Previous output for the same UUID is cleaned before each run

When adding a new component, check whether it has any behavior that differs in local debug mode (e.g. file paths, S3 operations) and add the appropriate `if LOCAL_DEBUG:` guard. Document any local debug setup requirements in `source/container/LOCAL_DEBUG_README.md`.

---

## Error Code Registry

Error codes 700–799 are reserved for container pipeline errors. When adding a new component, assign the next available code and add it here and to the docstring at the top of `main.py`.

| Code | Location | Description |
|---|---|---|
| 700 | Initialization | Required environment variables not set |
| 705 | Pose transform | Both pose prior modes enabled simultaneously (unsupported) |
| 710 | Pose transform | Wrong file type for pose priors (only .zip supported) |
| 715 | Pose transform | Issue transforming pose to colmap component |
| 720 | Video processing | Issue creating video to images component |
| 721 | Video processing | Issue creating rotate portrait images component |
| 730 | Segmentation | Issue creating background removal component |
| 735 | Segmentation | Issue creating spherical image component |
| 740 | Segmentation | Issue creating human subject removal component |
| 745 | Reconstruction | Reconstruction software name not implemented |
| 750 | Reconstruction | Issue creating the reconstruction component |
| 755 | Reconstruction | Issue creating the Colmap to Nerfstudio component |
| 760 | Training | Trainer specified does not match proper configuration |
| 765 | Training | Issue running the training session stage |
| 770 | Training | Issue exporting splat from NerfStudio |
| 771 | Training | Issue calculating metrics |
| 775 | Post-processing | Issue rendering trajectory video |
| 776 | Post-processing | Issue extracting video thumbnail |
| 777 | Post-processing | Issue cleaning point cloud |
| 780 | Post-processing | Issue cropping splat bounding box |
| 781 | Post-processing | Issue removing PLY comments |
| 782 | Post-processing | Issue creating derivative PLY files |
| 783 | Post-processing | Issue transforming coordinates |
| 784 | Post-processing | Issue mirroring PLY |
| 785 | Post-processing | Issue rotating PLY |
| 786 | Post-processing | Issue converting PLY to SOG |
| 787 | Post-processing | Issue converting PLY to USDZ |
| 788 | Post-processing | Issue converting PLY to SPZ |
| 790 | Post-processing | Issue uploading asset to S3 |
| 795 | General | General error running the pipeline |
| 800 | Post-processing | Issue generating or uploading collision voxel data |
| 801 | Post-processing | Issue generating or uploading LOD SOG bundle |
| 802–899 | Reserved | Available for new features |
