# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# This script will mask/remove parts of a 360/spherical/equirectangular image
# based on cubemap face structure. It will then reproject a directory of ERP images into
# perspective images that are masked (.png)

import os
import re
import cv2
import argparse
import torch
import numpy as np
from imageio.v2 import imread, imwrite
import Equirec2Cube
from PIL import Image
import py360convert
import ast
import subprocess
import multiprocessing
import shutil

def arg_as_list(s):
    v = ast.literal_eval(s)                                                    
    if type(v) is not list:                                                    
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def reverse_file_order(directory_path):
    """
    Reverses the order of sequentially named files in a directory.
    Example: 00000.png -> 00099.png, 00001.png -> 00098.png, etc.
    
    Args:
        directory_path (str): Path to the directory containing the files
    """
    try:
        # Get list of files and sort them
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        files.sort()
        
        # Create temporary directory
        temp_dir = os.path.join(directory_path, 'temp_reverse')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # Get the total number of files and adjust for zero-based naming
        total_files = len(files) - 1  # Subtract 1 to account for starting at 0
        width = len(files[0].split('.')[0])  # Get width of number portion
        
        # Rename files in reverse order to temp directory
        for i, filename in enumerate(files):
            name, ext = os.path.splitext(filename)
            new_name = str(total_files - i).zfill(width) + ext
            old_path = os.path.join(directory_path, filename)
            temp_path = os.path.join(temp_dir, new_name)
            shutil.copy2(old_path, temp_path)
            
        # Move files back to original directory
        for filename in os.listdir(temp_dir):
            temp_path = os.path.join(temp_dir, filename)
            new_path = os.path.join(directory_path, filename)
            shutil.move(temp_path, new_path)
            
        # Remove temporary directory
        os.rmdir(temp_dir)
        print("File renaming completed successfully!")
        
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise RuntimeError(f"An error occurred reversing file order: {str(e)}") from e

if __name__ == '__main__':
    # Create Argument Parser with Rich Formatter
    parser = argparse.ArgumentParser(
    prog='remove-360-face',
    description='Create a filter that will remove portions of a \
        spherical image based on a cubemap face selection regime'
    )

    # Define the Arguments
    parser.add_argument(
        '-d', '--data_dir',
        required=True,
        default=None,
        action='store',
        help='Target data directory for the ERP images')

    parser.add_argument(
        '-rf', '--remove_faces',
        type=arg_as_list,
        default=[],
        help="""A list of faces to remove. 
        Can be ["back", "down", "front", "left", "right", "up"]"""
    )

    parser.add_argument(
        '-ossfo', '--optimize_sequential_spherical_frame_order',
        required=False,
        default='true',
        action='store',
        help='Whether to enable optimization of spherical video frames to help solve SfM'
    )
    
    args = parser.parse_args()

    # Setup paths
    data_dir = str(args.data_dir)
    remove_face_list = args.remove_faces
    thread_count = multiprocessing.cpu_count()
    optimize_seq_spherical_frames = True
    if str(args.optimize_sequential_spherical_frame_order).lower() == "true":
        optimize_seq_spherical_frames = True
    else:
        optimize_seq_spherical_frames = False

    # If you need to use GPU to accelerate (especially for the need of converting many images)
    USE_GPU = True # set False to disable
    Image.MAX_IMAGE_PIXELS = 1000000000
    try:
        # Check that input directory exists
        if os.path.isdir(data_dir):
            # Get list of all files in data directory
            filenames = os.listdir(data_dir)
            filenames = sorted(filenames)
            if filenames is not None:

                for filename in filenames:
                    base_name, extension = os.path.splitext(filename)
                    if base_name[:5].lower() == "frame": # if using opencv frame extraction, will be in format "frame_298_01520.png"
                        img_num = base_name.split('_')[1] # only grab the new frame numbers
                    else:
                        img_num = os.path.basename(base_name) # this will be case for no image processing like deblur
                    if extension.lower() == ".jpeg" or \
                        extension.lower() == ".jpg" or \
                        extension.lower() == ".png":
                        orig_path = os.path.join(data_dir, f"{base_name}{extension}")

                        # Prepare images into separate sequential directories for
                        # reordering based on neighboring faces
                        new_dir = os.path.join(data_dir, img_num)

                        if not os.path.isdir(new_dir):
                            os.mkdir(new_dir)

                        # Move input file to its own directory
                        new_path = os.path.join(new_dir, f"{img_num}{extension}")

                        if not os.path.isfile(new_path):
                            print(f"Moving {orig_path} to {new_path}")
                            shutil.move(orig_path, new_path)

                        # Read the Equirectangular to Cubemap projection
                        img = imread(new_path, pilmode='RGBA')
                        #img = cv2.resize(, (2048, 1024), interpolation=cv2.INTER_AREA)
                        dims = img.shape

                        # Equirectangular to Cubemap
                        try:
                            # The parameters are equirectangular height/width, cubemap dim, and use GPU or not
                            e2c = Equirec2Cube.Equirec2Cube(dims[0], dims[1], int(float(dims[0])/2), CUDA=USE_GPU)

                            batch = torch.FloatTensor(img.astype(float)/255).permute(2, 0, 1)[None, ...]
                            if USE_GPU: batch = batch.cuda()
                            
                            # First convert the image to cubemap
                            cubemap_tensor = e2c(batch)
                            cubemap = cubemap_tensor.permute(0, 2, 3, 1).cpu().numpy()
                        except Exception as e:
                            raise RuntimeError(f"An error occurred during Equirectangular to Cubemap: {str(e)}") from e
                            #continue

                        # Now we save the cubemap to disk
                        order = ['right', 'down', 'left', 'back', 'front', 'up']
                        for i, term in enumerate(order):
                            face = (cubemap[i] * 255).astype(np.uint8)
                            if not os.path.isdir(f"{new_dir}/faces"):
                                os.mkdir(f"{new_dir}/faces")
                            print(f"Saving face {term} to {new_dir}/faces/{term}.png")
                            imwrite(f"{new_dir}/faces/{term}.png", face)

                        # Remove the unwanted faces
                        if len(remove_face_list) > 0:
                            if remove_face_list[0] != '' and remove_face_list[0] != "":
                                for remove_face in remove_face_list:
                                    # Create a transparent image and overwrite the face image
                                    img_height, img_width = int(float(dims[0])/2), int(float(dims[0])/2)
                                    n_channels = 4
                                    transparent_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)
                                    cubemap_face_filename = f"{new_dir}/faces/{str(remove_face).lower()}.png"
                                    # Save the image for visualization
                                    cv2.imwrite(cubemap_face_filename, transparent_img)

                        # Cubemap image faces to Equirectangular
                        cube_back = np.array(Image.open(f"{new_dir}/faces/back.png"))
                        cube_down = np.array(Image.open(f"{new_dir}/faces/down.png"))
                        cube_front = Image.open(f"{new_dir}/faces/front.png")
                        cube_left = np.array(Image.open(f"{new_dir}/faces/left.png"))
                        cube_right = Image.open(f"{new_dir}/faces/right.png")
                        cube_up = Image.open(f"{new_dir}/faces/up.png")

                        # Flip faces to correspond to mapping
                        flip_cube_front = np.array(cube_front.transpose(Image.FLIP_LEFT_RIGHT))
                        flip_cube_up = np.array(cube_up.transpose(Image.FLIP_TOP_BOTTOM))
                        flip_cube_right = np.array(cube_right.transpose(Image.FLIP_LEFT_RIGHT))

                        # Convert Cubemap to ERP
                        cube_dice = [cube_left, flip_cube_front, flip_cube_right, cube_back, flip_cube_up, cube_down]
                        try:
                            erp_img = py360convert.c2e(cubemap=cube_dice, h=dims[0], w= dims[1], cube_format='list')
                        except Exception as e:
                            raise RuntimeError(f"An error occurred converting cubemap to ERP: {str(e)}") from e

                        filtered_img_dir = f"{new_dir}/filtered_imgs"
                        if not os.path.isdir(filtered_img_dir):
                            os.mkdir(filtered_img_dir)
                        Image.fromarray(erp_img.astype(np.uint8)).save(f"{filtered_img_dir}/{img_num}.png")

                        # Generate "connective images" between change in views to increase sfm convergence
                        if optimize_seq_spherical_frames is True:
                            pers_img_dir = f"{filtered_img_dir}/pers_imgs"
                            if not os.path.isdir(pers_img_dir):
                                os.mkdir(pers_img_dir)

                            try:
                                # Run the converter script for ERP to perspective images
                                subprocess.run([
                                    "python", "spherical/360ImageConverterforColmap.py",
                                    "-i", filtered_img_dir,
                                    "-o", pers_img_dir,
                                    "--overlap", "0",
                                    "--fov", "90", "90",
                                    "--base_angle", "45", "45",
                                    "--resolution", str(960), str(960),
                                    "--threads", str(thread_count),
                                    "--exclude_v_angles", "90"
                                ], check=True)
                            except Exception as e:
                                raise RuntimeError(f"An error occurred converting ERP to perspective images: {str(e)}") from e

                # Reorder the image sequence to be primarily ordered by view across frames instead of inverse
                # Theoretically, this will allow better matching during SfM due to parallax effect in sequential frames
                # Only get subfolders in the input directory                # .../images/01
                subfolders = [ f.path for f in os.scandir(data_dir) if f.is_dir() ]
                subfolders = sorted(subfolders)
                view_dir_path_list = []

                # Index through subfolders and process each directory that has cube face photos
                for subfolder in subfolders:
                    # Get list of all images in the subdirectory
                    faces_subfolder = os.path.join(subfolder, "faces")
                    files = [os.path.join(path, name) for path, subdirs, files in os.walk(faces_subfolder) for name in files]
                    files = sorted(files)
                    # Index through images and copy them into destination directory
                    for img_file in files:
                        head, tail = os.path.splitext(img_file)
                        #./data\images\000\faces\back.png
                        view = os.path.basename(head)
                        path_parts = re.split(r"[/\\]", head)
                        path_parts.pop()
                        path_parts.pop()
                        img_num = f"{path_parts[len(path_parts)-1]}"
                        path_parts.pop()
                        path_parts.pop()

                        if path_parts[0] == '':
                            root_view_path = os.path.join("/", *path_parts, "views")
                        else:
                            root_view_path = os.path.join(*path_parts, "views")
                        if not os.path.isdir(root_view_path):
                            os.mkdir(root_view_path)
                        view_dir = os.path.join(root_view_path, view)

                        if view_dir not in view_dir_path_list:
                            view_dir_path_list.append(view_dir)
                        if not os.path.isdir(view_dir):
                            os.mkdir(view_dir)
                        filename_str = f"{img_num}{tail}"
                        dest_path = os.path.join(view_dir, filename_str)
                        try:
                            print(f"Moving {img_file} to {dest_path}")
                            shutil.move(img_file, dest_path)
                        except Exception as e:
                            raise RuntimeError(f"An error occurred moving cube face images: {str(e)}") from e

                image_folders = os.listdir(data_dir)
                view_num_len = len(image_folders[0])

                # Only get subfolders in the view directory                # .../views/back
                view_path = os.path.join(data_dir, "..", "views")
                view_subfolders = [ f.path for f in os.scandir(view_path) if f.is_dir() ]
                view_subfolders = sorted(view_subfolders)
                
                if optimize_seq_spherical_frames is True:
                    # Optimize the views
                    # Reverse order for particular views, add supplementary images between views
                    # Front
                    # Last front 45, RF
                    # Right (rev)
                    # Last right 45, RB
                    # Back
                    # Last back 45, LB
                    # Left (rev)
                    # Last left 45, LB
                    # Up
                    # Last front
                    # Down (rev)

                    for i, view_subfolder in enumerate(view_subfolders):
                        # Remove views that have been configured to be removed
                        if os.path.basename(view_subfolder) in remove_face_list:
                            shutil.rmtree(view_subfolder)
                        else:
                            view_images = [ f.path for f in os.scandir(view_subfolder) if f.is_file() ]
                            view_images = sorted(view_images)
                            file_count = len(view_images)

                            first_filename = view_images[0]
                            last_filename = view_images[file_count-1]

                            head_first_fn, tail = os.path.splitext(first_filename)
                            head_last_fn, tail = os.path.splitext(last_filename)

                            first_view = os.path.basename(head_first_fn)
                            first_view =  f"{int(first_view):0{view_num_len}d}"

                            last_view = os.path.basename(head_last_fn)
                            last_view =  f"{int(last_view):0{view_num_len}d}"

                            persp_image_path = ""
                            view = os.path.basename(os.path.normpath(view_subfolder))

                            if view == "up":
                                persp_image_path = os.path.join(view_path, "front", f"{last_view}.png")
                            elif view == "back":
                                persp_image_path = os.path.join(data_dir, last_view, "filtered_imgs", "pers_imgs", f"{last_view}_perspective_04.png") # 45 LB
                            elif view == "front":
                                persp_image_path = os.path.join(data_dir, last_view, "filtered_imgs", "pers_imgs", f"{last_view}_perspective_02.png") # 45 RF
                            elif view == "left":
                                reverse_file_order(view_subfolder)
                                persp_image_path = os.path.join(data_dir, first_view, "filtered_imgs", "pers_imgs", f"{first_view}_perspective_01.png") # 45 LB
                            elif view == "right":
                                reverse_file_order(view_subfolder)
                                persp_image_path = os.path.join(data_dir, first_view, "filtered_imgs", "pers_imgs", f"{first_view}_perspective_03.png") # 45 RB
                            elif view == "down":
                                reverse_file_order(view_subfolder)
                            destination_path = os.path.join(view_subfolder, f"{file_count:0{view_num_len}d}{tail}")
                            if persp_image_path != "":
                                if os.path.isfile(persp_image_path):
                                    print(f"Copying {persp_image_path} to {destination_path}")
                                    shutil.copy(persp_image_path, destination_path)
                                else:
                                    raise RuntimeError(f"Error: {persp_image_path} is not a valid file path.")

                # Only get subfolders in the view directory                # .../views/back
                view_path = os.path.join(data_dir, "..", "views")
                view_subfolders = [ f.path for f in os.scandir(view_path) if f.is_dir() ]
                print(f"Found {len(view_subfolders)} views in {view_path}")

                # Remove original image folder after moving them to view path
                shutil.rmtree(data_dir)
                os.mkdir(data_dir)

                # Front -> Right -> Back -> Left -> Up -> Down ->
                # Reorder the view path to coordinate with optimal view pattern
                for view_dir_path in view_subfolders:
                    rest, view = os.path.split(view_dir_path)
                    view_order = ""
                    print(f"Processing {view} view")
                    # Need to use match in order to map views to image order
                    match str(view).lower():
                        case "up":
                            if optimize_seq_spherical_frames is True:
                                view_order = "05"
                            else:
                                view_order = "06"
                        case "back":
                            if optimize_seq_spherical_frames is True:
                                view_order = "03"
                            else:
                                view_order = "02"
                        case "down":
                            if optimize_seq_spherical_frames is True:
                                view_order = "06"
                            else:
                                view_order = "05"
                        case "front":
                            if optimize_seq_spherical_frames is True:
                                view_order = "01"
                            else:
                                view_order = "01"
                        case "left":
                            if optimize_seq_spherical_frames is True:
                                view_order = "04"
                            else:
                                view_order = "03"
                        case "right":
                            if optimize_seq_spherical_frames is True:
                                view_order = "02"
                            else:
                                view_order = "04"
                        case _:
                            raise RuntimeError(f"Error: {view} is not a valid view.")
                    os.rename(view_dir_path, os.path.join(rest, view_order))

                root_path_parts = re.split(r"[/\\]", data_dir)
                root_path_parts.pop()
                if root_path_parts[0] == '':
                    view_path = os.path.join("/", *root_path_parts, "views")
                    img_path = os.path.join("/", *root_path_parts, "images")
                else:
                    view_path = os.path.join(*root_path_parts, "views")
                    img_path = os.path.join(*root_path_parts, "images")
                view_dirs = os.listdir(view_path)
                view_dirs = sorted(view_dirs)
                print(f"Found {len(view_dirs)} views in {view_path}")

                # Move the images from the "view" directory to the "images" directory
                i = 1
                for view_dir in view_dirs:
                    view_dir_path = os.path.join(view_path, view_dir)
                    img_filenames = os.listdir(view_dir_path)
                    img_filenames = sorted(img_filenames)
                    print(f"Found {len(img_filenames)} images in {view_dir_path}")
                    for img_filename in img_filenames:
                        input_img_filename_path = os.path.join(view_dir_path, img_filename)
                        head, extension = os.path.splitext(input_img_filename_path)
                        output_img_filename_path = os.path.join(img_path, f"{i:05d}{extension}")
                        print(f"Moving {input_img_filename_path} to {output_img_filename_path}")
                        shutil.move(input_img_filename_path, output_img_filename_path)
                        i = i + 1
                shutil.rmtree(view_path)
            else:
                print(f"No supported images present in {data_dir}.")
        else:
            print("Input directory is not valid")
    except Exception as e:
        raise RuntimeError("Error running spherical to perspective transformation. {e}") from e