### Understanding the configuration and capabilities
This Guidance is built with a variety of open source tools that can be used for various use cases. Because of this, many options are contained in this Guidance. The following is a high-level overview of each option and its applicability:

- **Workflow input:**
    - Video (.mov or .mp4)
    - Archive (.zip) of images (.png or .jpeg)
    - Archive (.zip) of pose priors and images (transforms.json, colmap model files)
- **Workflow output:** .ply and .spz, archive of project files (images, point cloud, metadata)
- **UUID:** A unique identifier used by backend system to record individual requests in DynamoDB
- **Instance type:** The EC2 compute resource to use for the workflow. Currently, these instance types are tested and supported:
    - ml.g5.4xlarge (recommended for <500 4k images)
    - ml.g5.8xlarge (recommended for <500 4k images)
    - ml.g6e.4xlarge (for large datasets (e.g. >500 4k images or 3DGRT))
    - ml.g6e.8xlarge (for large datasets (e.g. >500 4k images or 3DGRT))
- **S3:**
    - **Bucket name:** The name of the S3 bucket that was deployed by CDK/Terraform. This is an output from the deployment.
    - **Input prefix:** The S3 prefix (initial directory minus the job prefix) for the input media
    - **Input key:** The S3 key for the input media
    - **Output prefix:** The S3 prefix (initial directory minus the job prefix) for the output files
    - **S3 Job prefix:** The S3 prefix (initial directory) for the job json files
- **Video processing:**
    - **Max number of images:** (integer), the maximum number of images to use when a video is given as input. If using a `.zip` file with images or pose priors, this parameter will be ignored.
- **Image processing:**
    - **Filter blurry images:** (boolean), whether to remove blurry images from the dataset. If using a `.zip` file with pose priors, this parameter will be ignored.
- **Structure from motion (SfM):**
    - **Enable:** true or false, Whether to enable SfM or not. Future plans will enable input of SfM output
    - **Software name:** colmap or glomap, software to use for the triangulation of the mapper
    - **Enable enhanced feature extraction:** true or false, whether to enable enhanced feature extraction which uses `estimate_affine_shape` and `domain_size_pooling` to enhance the feature matching
    - **Matching method:**
        - sequential (best for videos or images that share overlapping features)
        - spatial (best to use for pose priors to take spatial orientation into account)
        - vocab (best for large datasets that are not sequentially bound e.g. <1000 images)
        - exhaustive (only use this method if dataset struggles to converge with other methods)
    - **Pose priors:** in order to speed up reconstruction, camera poses associated with the images can be used as input. In particular, this feature accepts a `.zip` archive folder that has the same schema as [NerfCapture](https://github.com/jc211/NeRFCapture/tree/main){:target="_blank"} or can be a `.zip` archive folder that contains images and sparse directories with [Colmap model text files](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses){:target="_blank"}.

        > At this time, depth images are not used in the splat process.

        > ***Note: All image files must be sequentially named and padded (e.g. 001.png, 002.png, etc.)***

        - **Use pose prior Colmap model files:** see [Colmap model text files](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses){:target="_blank"}
            - The file schema for providing the already created Colmap model files looks like this
                <details>
                <summary><strong>Archive structure (.zip)</strong></summary>
                <pre style="background-color: #F6F6F6;">
                <code>
                archive.zip
                ├── images/
                    └── *.{png,jpg,jpeg}
                └── sparse/
                    ├── 0/
                        └── cameras.txt (camera intrinsics)
                        └── images.txt (camera extrinsics/poses)
                        └── points3D.txt (empty)
                </code>
                </pre>
                </details>
        - **Use pose prior transform JSON:** see [NerfCapture](https://github.com/jc211/NeRFCapture/tree/main){:target="_blank"}
            > Ensure the `.zip` contains both the `transforms.json` file and an `/images` directory with sequentially named RGB images.
            - **Enable:** (boolean), whether to enable using `transforms.json` file for pose priors.
            - **Source coordinate name:** ("arkit" or "arcore" or "opengl" or "opencv" or "ros"), the source coordinates used with pose priors
            - **Pose is world-to-camera:** (boolean), whether the source coordinates for pose priors are in world-to-camera (True) or camera-to-world (False)
            - The schema for `transforms.json` to input looks like this in case you would perform an alternate method to extract the image and poses.
                > ***Note:** Depth images do not need to be present in the `.zip` file, but the "depth_path" still needs to be filled so the extension is known. For example if image is `images/3.png`, then `depth_path=images/3.depth.png` and `file_path=images/3`. `timestamp` and `depth_images` are not currently used in the pipeline.*
                <details>
                <summary><strong>Sample <code>transforms.json</code> contents</strong></summary>
                <pre style="background-color: #F6F6F6;">
                <code>
                {
                    "frames": [
                        {
                            "fl_y": 1363.47,
                            "w": 1920,
                            "fl_x": 1363.47,
                            "cy": 728.95135,
                            "timestamp": 309075.09929275,
                            "depth_path": "images/3.depth.png",
                            "transform_matrix": [
                                [
                                    -0.98750305,
                                    0.08610178,
                                    -0.13199921,
                                    0.9487833
                                ],
                                [
                                    0.024557188,
                                    0.91140217,
                                    0.41078326,
                                    -0.5495963
                                ],
                                [
                                    0.1556736,
                                    0.40240824,
                                    -0.9021269,
                                    -0.4476293
                                ],
                                [
                                    0,
                                    0,
                                    0,
                                    1
                                ]
                            ],
                            "file_path": "images/3",
                            "cx": 956.5136,
                            "h": 1440
                        },
                        ...
                    ]
                }
                </code>
                </pre>
                </details>

            - Parameter definitions
                <details>
                <summary><strong>Parameter defintions</strong></summary>
                <pre style="background-color: #F6F6F6;">
                <code>
                "cx": camera sensor center on x-axis in pixels
                "cy": camera sensor center on y-axis in pixels
                "fl_x":  focal length on x-axis in pixels
                "fl_y": focal length on y-axis in pixels
                "w": camera resolution width in pixels
                "h": camera resolution height in pixels
                "file_path": this will look like this: "images/13" if image filename is images/13.png
                "depth_path": this will look like this: "images/13.depth.png" if image filename is images/13.depth.png (even if you dont have depth images, fill this in with correct extension)
                "transform_matrix": 4x4 pose matrix
                "timestamp": seconds (can be absolute or use epoch)
                </code>
                </pre>
                </details>

            - Archive structure (.zip)
                <details>
                <summary><strong>Archive structure (.zip)</strong></summary>
                <pre style="background-color: #F6F6F6;">
                <code>
                archive.zip
                ├── transforms.json
                └── images/
                    └── *.{png,jpg,jpeg}  # Image files, depth images need "depth" in file name
                </code>
                </pre>
                </details>

- **Training:**
    - **Enable:** true or false, whether to enable 3DGS training or not. Future plans will enable user to only perform SfM
    - **Maximum steps:** (integer), The maximum training steps to use while training the splat
    - **Splat model:** splatfacto, splatfacto-big, splatfacto-w-light, nerfacto, the GS model to use for training
        - Pointers:
            - **splatfacto:** a great, generalized model that is perfect to start with if you are unsure what model to choose
            - **splatfacto-big:** high quality model that should be used to output feature-rich scenes and objects. This will yield a larger .ply file.
            - **splatfacto-w-light:** if wanting to achieve superior quality, while still pruning unwanted gaussians, use this one. This model will take the longest time, but the output file size will be much less, while still upholding quality.
            - **splatfacto-mcmc:** the current SotA that balances small training time with high quality output
            - **nerfacto:** used for testing and comparisons between NeRF and GS. The output will be a NeRF. Beware, this model will need more images than GS in order to maintain higher quality.
            - **3dgut:** used for enabling Distorted Cameras and Secondary Rays in Gaussian Splatting. Great for fisheye camera input.
            - **3dgrt:** used for 3D Gaussian Ray Tracing and Fast Tracing of Particle Scenes. Great for highly detailed scenes at the cost of processing power and time
    - **Rotate splat:** true or false, whether to rotate the output splat for the Gradio 3D Model viewer coordinate system (set to true will rotate both the .ply and .spz)
- **Spherical camera:**
    - **Enable:** true or false, whether to enable 360 camera support or not
    - **Cube faces to remove:** "['back', 'down', 'front', 'left', 'right', 'up']", a list of cube faces to remove from the spherical image. This is great for cropping out people or objects from the 360 image. 
    > ***Note:** The above configurations were tested with an Insta360 ONE X2, exporting frame(s) in equirectangular format, 9:16 ratio, 5.7k resolution, and 30 frames per second. The captures were taken with the camera display aimed toward the person capturing the frame(s).*

    <div align="center">
    {% include image.html file="open_3drt_images/erp-views.png" alt="Equirectangular Cube Map Views" %}
    <i>Figure 12: Equirectangular cube map views from a spherical camera</i>
    </div>
    <p>
    <br>
    </p>
    <div align="center">
    {% include image.html file="open_3drt_images/erp-masked.png" alt="Filtered equirectangular image" %}
    <i>Figure 13: Example: Enable Spherical Camera = true, Cube Faces to Remove = "['back', 'down']"</i>
    </div>
    <p>
    <br>
    </p>

- **Segmentation:**
    - **Remove background:** true or false, whether to remove the background when input an object (not scene) or not
    - **Background removal model:** "u2net", "u2net-human", or "sam2" the background removal model to use
    > *Note: The sam2 model can only be used on video at this time*
    - **Remove human subject:** true or false, whether to remove humans from the scene or not. This can be combined with other removal methods such as background removal and cube face removal.

    <div align="center">
    {% include image.html file="open_3drt_images/background-removal-img.gif" alt="Background removal" %}
    <i>Figure 14: Background removal using SAM2</i>
    </div>