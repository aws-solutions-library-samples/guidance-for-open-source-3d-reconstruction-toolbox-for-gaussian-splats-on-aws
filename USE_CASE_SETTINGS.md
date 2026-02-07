# 3D Reconstruction Use-Case Settings Guide

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
