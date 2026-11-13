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

""" Remove an object from a directory of images using a mask """

import numpy as np
import cv2
import argparse
import os
import shutil

def apply_binary_mask(image, mask):
    """
    Apply a binary mask with alpha channel to an image
    Args:
        image: Input image (BGR or BGRA format)
        mask: Binary mask with alpha channel (BGRA format)
    Returns:
        Masked image with alpha channel (BGRA format)
    """
    # Convert image to BGRA if it's not already
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image_alpha = np.ones(image.shape[:2], dtype=np.float32)
    else:
        # If image already has alpha channel, normalize it to 0-1
        image_alpha = image[:, :, 3] / 255.0

    # Ensure both images have the same dimensions
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask must have the same dimensions")

    # Extract alpha channel from mask and normalize to 0-1
    mask_alpha = mask[:, :, 3] / 255.0

    # Combine both alpha channels by multiplication
    combined_alpha = image_alpha * mask_alpha

    # Apply combined alpha channel to the image
    result = image.copy()
    result[:, :, 3] = (combined_alpha * 255).astype(np.uint8)

    # Apply the alpha to the RGB channels
    for i in range(3):
        result[:, :, i] = (image[:, :, i] * combined_alpha).astype(np.uint8)

    return result

def binary_to_alpha(image):
    """
    Convert a binary image into an image with an alpha channel
    """
    # Load the binary image
    binary_image = image

    # Create an alpha channel (initially all transparent)
    alpha_channel = np.zeros(binary_image.shape, dtype=np.uint8)

    # Set the alpha channel to 255 where the binary image is white
    alpha_channel[binary_image == 255] = 255

    # Create a 4-channel BGRA image
    bgra_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGRA)

    # Replace the alpha channel with the one we created
    bgra_image[:, :, 3] = alpha_channel

    # Save the image with alpha channel
    #cv2.imwrite('image_with_alpha.png', bgra_image)
    return bgra_image

def black_to_alpha(image):
    """
    Transform a black and white image to an image with an alpha channel
    """
    # Make a True/False mask of pixels whose BGR values sum to more than zero
    alpha = np.sum(image, axis=-1) > 0

    # Convert True/False to 0/255 and change type to "uint8" to match "na"
    alpha = np.uint8(alpha * 255)

    # Stack new alpha layer with existing image to go from BGR to BGRA,
    # i.e. 3 channels to 4 channels
    res = np.dstack((image, alpha))

    return res

if __name__ == "__main__":
    # Create Argument Parser with Rich Formatter
    parser = argparse.ArgumentParser(
    prog='remove-object-using-mask',
    description='Remove parts of an image using a mask and alpha'
    )

    # Define the Arguments
    parser.add_argument(
        '-oi', '--orig_img_path',
        required=True,
        default=None,
        action='store',
        help='The path to the original images to apply object removal on'
    )

    parser.add_argument(
        '-om', '--mask_img_path',
        required=True,
        default=None,
        action='store',
        help='The path to the mask images to use to remove the objects'
    )

    parser.add_argument(
        '-od', '--output_dir',
        required=True,
        default=None,
        action='store',
        help='The directory to store the result images'
    )

    args = parser.parse_args()

    orig_image_path = args.orig_img_path
    mask_image_path = args.mask_img_path
    output_path = args.output_dir

    if orig_image_path == output_path:
        raw_input_path = orig_image_path + '_raw'
        os.makedirs(raw_input_path, exist_ok=True)
        os.rename(orig_image_path, raw_input_path)
        orig_image_path = raw_input_path
    os.makedirs(output_path, exist_ok=True)
    orig_imgs_list = sorted(f for f in os.listdir(orig_image_path)
                            if os.path.isfile(os.path.join(orig_image_path, f))
                            and f.lower().endswith(('.jpg', '.jpeg', '.png')))

    def find_mask(filename):
        # remove_background writes <stem>.png; pair by name, not by listing order.
        for cand in (os.path.splitext(filename)[0] + ".png", filename + ".png", filename):
            p = os.path.join(mask_image_path, cand)
            if os.path.isfile(p):
                return p
        return None

    def person_alpha(mask_filepath):
        # Person mask (255 = person). remove_background writes RGBA with the person
        # in alpha and the image in RGB, so the alpha channel must be used.
        m = cv2.imread(mask_filepath, cv2.IMREAD_UNCHANGED)
        if m is None:
            return None
        if m.ndim == 3 and m.shape[2] == 4:
            return m[:, :, 3]
        return m if m.ndim == 2 else cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    n_changed = 0
    try:
        for i, orig_filename in enumerate(orig_imgs_list):
            orig_filepath = os.path.join(orig_image_path, orig_filename)
            output_filepath = os.path.join(output_path, orig_filename)
            mask_filepath = find_mask(orig_filename)
            alpha = person_alpha(mask_filepath) if mask_filepath else None
            if alpha is None or not (alpha > 10).any():
                # No person (or no mask): keep the original file (no re-encode), so
                # the output holds every image.
                shutil.copy2(orig_filepath, output_filepath)
                continue
            orig_img = cv2.imread(orig_filepath, cv2.IMREAD_UNCHANGED)
            if orig_img is None or orig_img.shape[:2] != alpha.shape[:2]:
                print(f"WARNING: {orig_filename}: unreadable or mask size mismatch; keeping the original")
                shutil.copy2(orig_filepath, output_filepath)
                continue
            keep = np.where(alpha > 10, 0, 255).astype(np.uint8)
            masked_img = apply_binary_mask(orig_img, binary_to_alpha(keep))
            cv2.imwrite(output_filepath, masked_img)
            n_changed += 1
        print(f"Removed people from {n_changed} of {len(orig_imgs_list)} images")
    except Exception as e:
        raise RuntimeError(f"Error removing object using mask: {e}") from e
