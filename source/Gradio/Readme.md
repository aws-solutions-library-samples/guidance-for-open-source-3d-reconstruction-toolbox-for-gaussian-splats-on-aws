This README provides a comprehensive guide for users to run the Gradio Interface for testing out Open Source 3D Reconstruction Toolbox for Gaussian Splats

This guide will help you to do the following:

1. Set up the required environment
2. Install necessary dependencies
3. Run the application
4. Understand the interface components
5. Troubleshoot common issues

# Generate Splat Gradio Interface

This Gradio interface will provides a web-based user interface for generating 3D Gaussian Splatting models using Gradio for testing purposes between your local machine and AWS.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Create and activate a virtual environment (recommended) or use same environment as infrastructure deployment:

```bash
python -m venv venv
source venv/bin/activate # On Windows, use: venv\Scripts\activate
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Configure the Application

Configure the Gradio Application to use the created bucket:
  - Open `generate_splat_gradio.py` in a text editor
  - Input the S3 bucket name into the `self.s3_bucket = ""` field
  - Save the file and exit

## Authenticate with AWS
- Confirm authenticated as the correct IAM principal and in the correct AWS account (see [get-caller-identity](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/sts/get-caller-identity.html) for more info).

    ```bash
    aws sts get-caller-identity
    ```

## Running the Application

- Start the Gradio interface:

  ```bash
  cd source/Gradio
  python generate_splat_gradio.py
  ```

- Open your web browser and navigate to the URL displayed in the terminal (typically `http://127.0.0.1:7860`)

## Interface Components

The interface is organized into several sections:

### Processing Components

- `AWS Workload`: For AWS Settings specific to the infrastructure and services.
- `Advanced Settings`: Settings to guide the processing of the gaussian splat
  Key Settings include:
  - `Max Number of Images`: Set the maximum number of images to process from a video (default: 300)
  - `Filter Blurry Images`: Toggle to filter out blurry images
  - `Background Removal`: Toggle to remove background of objects
  - `Use Pose Priors`: Enable/disable pose priors for camera positioning
  - `Select SfM Software and Parameters`: Optimize the 3D reconstruction initialization
  - `Select Splat Training Software and Parameters` : Use models and parameters suitable for your use-case
- `S3 Browser`: ability to interrogate the contents of the S3 bucket chosen for the pipeline and download and view assets from the S3 bucket
- `Debug:` a quick access to view the `.json` payload which will be sent with the job.

## How It Works

1. **Input**: Users upload video/images and configure processing parameters through the web interface.
2. **Processing**: The application processes the inputs and generates a configuration JSON that defines:
   - Image processing settings
   - Camera configuration
   - Model generation parameters
   - Video payload
   - Submission: Clicking the **Upload to AWS** button will generate the required `.json` config file and upload both the `.json` and the video to AWS. A unique UUID will be included in this file to differentiate the jobs.
   - **NB:** The upload sequencing to AWS is important, the video must be uploaded first followed by the `.json`. The S3 bucket has a S3 Trigger which is actioned by the file type `.json`. If the video has not been uploaded yet, you will receive an error the video file not available. The Gradio file already caters for this, but when building your own interface make sure to keep this in mind.
3. **Output**: The system generates a 3D Gaussian Splatting model based on the settings provided in the `.json`. and will be placed in your chosen output folder location as your S3 Output Prefix

## Tips for Best Results

- Start by capturing objects, recording a video while orbiting the object
- Ensure video taken has good consistent lighting and are in focus
- Provide sufficient overlap for better reconstruction
- Use consistent camera settings across all shots
- Avoid reflective or transparent surfaces for best results

## Troubleshooting

If you encounter package-related errors, try:

```bash
pip uninstall gradio gradio-client packaging -y
pip install packaging --upgrade
pip install gradio --upgrade
```
