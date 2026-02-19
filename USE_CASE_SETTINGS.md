# 3D Reconstruction Use-Case Settings Guide

**Recommendations:**
1. Image Capture:
- Ensure 60-80% overlap between consecutive frames
- Move camera slowly and steadily
- Maintain consistent lighting
- Capture higher resolution images
- Avoid motion blur

2. Scene Setup:
- Add more distinctive features to the scene
- Ensure adequate and consistent lighting
- Avoid highly reflective surfaces
- Remove moving objects if possible

3. Processing:
- Try reducing the number of input images
- Consider using a different subset of images
- Verify image quality before processing

4. General:
- Use video input when possible with sequential matching
- If pose data is available from the images, use pose-priors and spatial matching
- If your environment is featureless, use pose data to help SfM converge
- Colmap is an incremental mapper, while Glomap is a global mapper
    - Incremental SfM (COLMAP)/Sequential approach:
        - Summary: Starts with a pair of images, estimates their poses, then incrementally adds one image at a time
        - Process: Initialize → Add image → Bundle adjustment → Repeat
        - Advantages: More robust to outliers, handles challenging scenes better
        - Disadvantages: Slower, can accumulate drift over long sequences
    - Global SfM (GLOMAP)/Simultaneous approach:
        - Summary: Estimates all camera poses at once using global optimization
        - Process: Extract features → Match all pairs → Solve for all poses simultaneously
        - Advantages: Faster, no drift accumulation, better for well-connected image sets
        - Disadvantages: Less robust to outliers, requires good feature matches across many images

#### Spherical Camera
- When scanning outside-in, similar to scanning objects in an orbital path, a single monocular 4K camera is sufficient.
- When capturing spaces inside-out (environments, not objects), we recommend using a spherical camera to gather imagery in 360 degrees.
- Using a spherical camera will greatly increase the number of input images without manual work and enable SfM to more effectively converge.
- At the time of writing this, Colmap requires the input images to be in perspective. For this, we have implemented a robust algorithm that will automatically transform your equirectangular video/images into perspective images.
- It is sometimes handy to remove views from the 360 image due to possibly camera person holding the camera. The "remove faces" option allows you to mask a view from the cubemap so the feature will not be in the output.
- We have added a feature to optimize the cubemap views of the image sequence using connective images and view nodes to help SfM converge. Please see the `/source/container/src/pipeline/spherical/equirectangular_to_perspective.py` script for more details.
- Be careful enabling `optimizing cubemap views` and masking views other than `up` or `down` as the algorithm leans on the horizontal views for connectivity.

## General Camera Settings
- Turn off auto white balance, adjust as needed for entire scene and lock it
- Turn off HDR
- Turn off auto exposure
- Ensure there is good lighting
- Spherical camera recommendations
    - Ensure you are either behind or under the camera with an invisible selfie stick
    - Use interval recording instead of video to alleviate blur
    - Use at least 8k panomara images

## Input Methods
Perspective or spherical(equirectangular) images are supported for all methods below

### 1. Monocular Video
- .mp4 and .mov supported
- 4K at 60FPS recommended

### 2. Image Archive
- .zip supported
- 4K resolution recommended
- images must be archived into single folder and sequentially numbered

### 3. Image Archive w/ GPS Data
- .zip supported
- 4K resolution recommended
- images must be archived into single folder and sequentially numbered
- images must have EXIF GPS data attached to the image header

### 4. Image Archive w/ Pose Priors
- .zip supported
- 4K resolution recommended
- images must be archived into single folder and sequentially numbered
- archive must include pose priors in either two formats
    - colmap model
        - a colmap reconstruction model in the form of .bin or .txt files can be given
    - transform json
        - a json in the format of [NeRFCapture](https://github.com/jc211/NeRFCapture/tree/main)/InstantNGP can be given and coordinate transform used (e.g. OpenGL, ARKit, ARCore, etc.)

### 5. Image Archive w/ Pose Priors and Point Cloud Prior
- Not yet supported

### 6. Model Archive
- .tar.gz is supported
- a previous job can be resumed or used to export 3d assets

## General Scanning Techniques

# Outside-In
Use this technique to scan **objects**, or small contained spaces
- In this mode, usually the capture sis 

# Inside-Out


## Use Case Settings
### Large Scene Reconstruction

### Indoor Room Scanning

### Object Scanning

### Outdoor Scanning
