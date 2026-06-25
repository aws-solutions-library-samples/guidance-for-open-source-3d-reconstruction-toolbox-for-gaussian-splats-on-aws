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
- When capturing spaces inside-out (environments, not objects), we recommend using a spherical camera to gather imagery in 360 degrees with at least 5.7k with 8k being preferred.
- Using a spherical camera will greatly increase the number of input images without manual work and enable SfM to more effectively converge.
- At the time of writing this, Colmap requires the input images to be in perspective. For this, we have implemented a colmap sample algorithm that will automatically transform your equirectangular video/images into perspective images.
- It is sometimes handy to remove views from the 360 image due to possibly camera person holding the camera. The "remove faces" option allows you to mask a view from the cubemap so the feature will not be in the output. For example for capturing using an Insta360 X5 on a selfie stick, as long as the person capturing the video is at the other end of the stick, you can remove the bottom cubemap and it will remove the person capturing the 360 video. If this is your intention, make sure to enable Tilt Recovery in the stabilization options for FlowState. Disable Direction Lock.

## General Camera Settings
- Turn off auto white balance, adjust as needed for entire scene and lock it
- Turn off HDR
- Turn off auto exposure, adjust to balance 
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
- .zip supported
- colmap model
    - a colmap reconstruction model in the form of .bin or .txt files can be given
    - the points3d.txt file must contain your lidar point cloud and the points generated from pose-prior point triangulator in Z-forward, Y-down coordinate system

### 6. Splat Model Archive
- .tar.gz is supported
- a previous job can be resumed or used to export 3d assets

## General Scanning Techniques

# Outside-In (Object Scanning)
Use this technique to scan **objects**, or small contained spaces
- In this mode, usually the capture system is focused on a specific object or group of objects
- This can be thought of an observer on the outside of the object of interest, always looking into the object
- For adequate capturing, rotate around the object at three different elevations: bottom, middle, top. Be sure to close the capture loop by ending the session where you started
    - For rotating, you can use a turn-table to turn the object while capturing video of the object on a boom stand. If you do this, ensure you turn on the option to remove the background, otherwise the SfM model will not converge. If you don't have a turntable, just slowly walk around the object circling it.
- Usually 300 frames or 1 minute video should be plenty to reconstruct the object

# Inside-Out (Environment Scanning)
Use this technique to scan **environments**, or rooms
- In this mode, usually the capture system is focused on capturing an environment
- This can be thought of an observer on the inside of a room, always looking outward towards the walls
- For adequate capturing, use a 360 camera on a selfie stick or drone to capture either video or a series of images
- Usually 600 frames or 3 minute video should be plenty to reconstruct a room/environment

## Use Case Settings
### Object Scanning

### Indoor Room Scanning

### Outdoor Scanning

### Large Scene Reconstruction
