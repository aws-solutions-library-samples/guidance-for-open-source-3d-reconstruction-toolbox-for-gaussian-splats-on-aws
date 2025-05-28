"""
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
#
# Adapted from: https://github.com/argildotai/sam2removevideobackground

Remove the background of images given a video

"""

import os
import torch
import numpy as np
import cv2
import sys
import logging

# Import our compatibility module first to patch the cog module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam2.build_sam import build_sam2_video_predictor
from cog import BasePredictor, Input, Path
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from datetime import datetime
from pathlib import Path
import multiprocessing
from typing import Optional, Tuple
import argparse
import shutil
import json

class Predictor(BasePredictor):
    def setup(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.info("Starting setup")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

        self.checkpoint = f"{os.environ['MODEL_PATH']}/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        try:
            self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint)
            logging.info("SAM2 predictor built successfully")
        except Exception as e:
            logging.exception(f"Error building SAM2 predictor: {e}")
            raise

        # Load a pre-trained Faster R-CNN model for body detection
        self.body_detector = None
        #self.body_detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        #self.body_detector.eval()
        #self.body_detector.to(self.device)

        logging.info("Setup completed")

    def predict(
            self,
            input_video: Path = Input(description="Input video file"),
            bg_color: str = Input(description="Background color (hex code)", default="#00FF00"),
            num_frames: int = Input(description="Number of frames to extract", default=300),
            output_path: Path = Input(description="Output path for the processed video frames", default=None),
            mask_threshold: float = Input(description="Mask threshold (0.0-1.0)", default=0.5),
            tracking_mode: str = Input(description="Tracking mode ('body' or 'center')", default="center")
    ) -> bool:
        try:
            # Store the current mode
            self.current_mode = tracking_mode.lower()
            # With custom output directory and quality settings
            mp4_path = self.ensure_mp4_format(
                input_video_path=str(input_video),
            )

            input_video = Path(mp4_path)
            frames_dir = os.path.join(os.path.dirname(str(input_video)), 'extracted_frames')
            output_frames_dir = output_path
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(output_frames_dir, exist_ok=True)

            logging.info("Extracting frames...")
            frame_names = self.extract_frames(input_video, frames_dir, num_frames)
            
            # Convert hex color to BGR tuple here, and only once
            if isinstance(bg_color, str):
                bg_color = tuple(int(bg_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))[::-1]

            logging.info("Initializing video processing...")
            inference_state = self.predictor.init_state(video_path=frames_dir)
            first_frame_path = os.path.join(frames_dir, frame_names[0])
            first_frame = cv2.imread(first_frame_path)

            # Select tracking mode
            if self.current_mode == 'body':
                logging.info("Using body tracking mode")
                keypoints = self.detect_body_keypoints(first_frame)
                labels = np.ones(len(keypoints), dtype=np.int32)
            else:
                logging.info("Using center point tracking mode")
                keypoints, labels = self.detect_center_keypoints(first_frame)
                if mask_threshold == 0.5:
                    mask_threshold = 0.35  # Adjusted threshold for center mode


            logging.info(f"Using mask threshold: {mask_threshold}")
            logging.info(f"Number of keypoints: {len(keypoints)}")
            logging.info(f"Number of foreground points: {np.sum(labels == 1)}")
            logging.info(f"Number of background points: {np.sum(labels == 0)}")

            # Initialize with selected points
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=keypoints,
                labels=labels,
            )

            logging.info("Collecting masks through propagate_in_video...")
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: out_mask_logits[i].detach().cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # Pre-load and verify frames
            frames_dict = {}
            for idx in range(len(frame_names)):
                frame_path = os.path.join(frames_dir, f"{idx:05d}.jpg")
                frame = cv2.imread(frame_path)
                if frame is not None and frame.size > 0:
                    frames_dict[idx] = frame
                else:
                    logging.error(f"Failed to load frame {idx}")

            if not frames_dict:
                raise RuntimeError("No valid frames loaded")

            # Ensure output directory exists and is empty
            output_frames_dir = Path(output_path)
            if output_frames_dir.exists():
                logging.info(f"Clearing existing output directory: {output_frames_dir}")
                for file in output_frames_dir.glob("*.png"):
                    file.unlink()
                output_frames_dir.mkdir(parents=True, exist_ok=True)
                
                logging.info(f"Will save processed frames to: {output_frames_dir}")
                
                # Process frames in batches
                batch_size = 10
                frame_indices = list(frames_dict.keys())
                total_frames = len(frame_indices)
                processed_count = 0

                for i in range(0, total_frames, batch_size):
                    batch_indices = frame_indices[i:i + batch_size]
                    logging.info(f"Processing batch of {len(batch_indices)} frames")
                    
                    try:
                        processed = self.process_frame_batch(
                            frame_batch=batch_indices,
                            frames_dict=frames_dict,
                            output_frames_dir=output_frames_dir,
                            video_segments=video_segments,
                            bg_color=bg_color,
                            mask_threshold=mask_threshold
                        )
                        processed_count += processed
                        logging.info(f"Processed {processed_count}/{total_frames} frames")
                    except Exception as e:
                        logging.error(f"Error processing batch {i//batch_size}: {str(e)}")
                        continue

                if processed_count == 0:
                    raise RuntimeError("No frames were saved to output directory")

                return True

        except Exception as e:
            logging.error(f"Error in predict: {str(e)}")
            raise

    def extract_equidistant_frames(self, input_video, output_dir, desired_frames):
        """
        Extract equidistant frames from a video file using OpenCV.
        """
        # Convert desired_frames to int if it's not already
        desired_frames = int(str(desired_frames))
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Open video file with OpenCV
            cap = cv2.VideoCapture(str(input_video))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {input_video}")
                
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logging.info(f"Video properties: {total_frames} frames, {fps:.2f} fps, {duration:.2f} seconds")
            
            if duration <= 0:
                raise RuntimeError("Could not determine video duration")
                
            # Calculate the frame interval to achieve desired number of frames
            interval = max(1, total_frames // desired_frames)
            
            # Extract frames at calculated intervals
            count = 0
            frame_idx = 0
            extracted_frames = 0
            
            while count < total_frames and extracted_frames < desired_frames:
                # Set the position to the exact frame we want
                cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                ret, frame = cap.read()
                
                if ret:
                    frame_path = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
                    if cv2.imwrite(frame_path, frame):
                        frame_idx += 1
                        extracted_frames += 1
                    else:
                        logging.warning(f"Failed to save frame at position {count}")
                else:
                    logging.warning(f"Failed to read frame at position {count}")
                    
                # Move to next interval
                count += interval
                
            cap.release()
            
            logging.info(f"Successfully extracted {extracted_frames} frames")
            
            if extracted_frames != desired_frames:
                logging.warning(f"Number of extracted frames ({extracted_frames}) "
                            f"differs from requested frames ({desired_frames})")
                
            return extracted_frames
                
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            raise
    
    def clean_mask(self, mask, min_area_ratio=0.01, kernel_size=5):
        """
        Clean the mask by removing small components and filling holes
        """
        # Calculate minimum area based on total image size
        total_pixels = mask.shape[0] * mask.shape[1]
        min_area = int(total_pixels * min_area_ratio)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Create clean mask
        clean_mask = np.zeros_like(mask)
        
        # Keep only large enough components
        for label in range(1, num_labels):  # Start from 1 to skip background
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                clean_mask[labels == label] = 1
        
        # Create kernels for morphological operations
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Fill small holes
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
        
        # Optional: Remove small noise
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill any remaining holes
        contours, _ = cv2.findContours(clean_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Fill all contours
        for contour in contours:
            cv2.fillPoly(clean_mask, [contour], 1)
        
        return clean_mask

    def process_frame_batch(self, frame_batch, frames_dict, output_frames_dir, video_segments, bg_color, mask_threshold):
        """
        Process a batch of frames with transparent background
        """
        processed_count = 0
        
        for idx in frame_batch:
            try:
                frame = frames_dict[idx]
                if idx not in video_segments:
                    logging.warning(f"No mask found for frame {idx}")
                    continue

                mask_logits = video_segments[idx]
                if 1 not in mask_logits:
                    logging.warning(f"No valid mask logits for frame {idx}")
                    continue

                # Get mask and process it
                mask = mask_logits[1]
                mask = np.squeeze(mask)  # Remove single-dimensional entries
                
                # Debug raw mask values
                logging.info(f"Frame {idx} raw mask range: {mask.min():.3f} to {mask.max():.3f}")
                
                # Normalize the mask to [0, 1] range
                mask_min = mask.min()
                mask_max = mask.max()
                if mask_max > mask_min:
                    mask = (mask - mask_min) / (mask_max - mask_min)
                
                # Debug normalized mask values
                logging.info(f"Frame {idx} normalized mask range: {mask.min():.3f} to {mask.max():.3f}")
                
                # Threshold the mask
                binary_mask = (mask > mask_threshold).astype(np.uint8)
                
                # Invert the binary mask for center mode
                if self.current_mode == 'center':
                    binary_mask = 1 - binary_mask
                    logging.info(f"Frame {idx}: Inverted mask for center mode")
                
                # Debug binary mask
                non_zero = np.count_nonzero(binary_mask)
                total_pixels = binary_mask.size
                logging.info(f"Frame {idx} binary mask: {non_zero}/{total_pixels} pixels ({(non_zero/total_pixels)*100:.2f}%)")
                
                if not np.any(binary_mask):
                    logging.warning(f"Empty mask for frame {idx} after thresholding")
                    # Try adaptive thresholding
                    adaptive_threshold = np.percentile(mask, 75)  # Use 75th percentile as threshold
                    binary_mask = (mask > adaptive_threshold).astype(np.uint8)
                    if self.current_mode == 'center':
                        binary_mask = 1 - binary_mask
                    logging.info(f"Trying adaptive threshold: {adaptive_threshold:.3f}")
                    
                    if not np.any(binary_mask):
                        logging.warning("Still empty mask after adaptive threshold")
                        continue

                # Clean the mask
                binary_mask = self.clean_mask(
                    binary_mask,
                    min_area_ratio=0.001,  # Reduced minimum area ratio
                    kernel_size=3  # Reduced kernel size
                )

                # Convert BGR to BGRA (add alpha channel)
                result = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                
                # Set alpha channel based on mask (255 for foreground, 0 for background)
                result[:, :, 3] = binary_mask * 255
                
                # Set background pixels to transparent
                result[binary_mask == 0] = [0, 0, 0, 0]  # Fully transparent

                # Save the processed frame
                output_path = os.path.join(output_frames_dir, f'frame_{idx:05d}.png')
                if cv2.imwrite(output_path, result):
                    processed_count += 1
                    logging.info(f"Saved frame {idx}")
                else:
                    logging.error(f"Failed to save frame {idx}")

            except Exception as e:
                logging.error(f"Error processing frame {idx}: {str(e)}")
                continue

        return processed_count

    def extract_frames(self, input_video, frames_dir, num_frames):
        cap = cv2.VideoCapture(str(input_video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // num_frames)
        
        frame_names = []
        frame_idx = 0
        count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % frame_interval == 0:
                frame_path = os.path.join(frames_dir, f"{frame_idx:05d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_names.append(f"{frame_idx:05d}.jpg")
                frame_idx += 1
                
            count += 1
            if frame_idx >= num_frames:
                break
                
        cap.release()
        return frame_names

    def detect_center_keypoints(self, frame):
        """
        Create a robust pattern of keypoints to better define foreground and background
        """
        height, width = frame.shape[:2]
        
        # Calculate center point with upward shift
        center_x = width // 2
        center_y = height // 2
        upward_shift = height * 0.15  # 15% of height upward shift
        center_y = int(center_y - upward_shift)
        
        # Define foreground points (object region)
        foreground_points = [
            # Center points
            [center_x, center_y],                    # Center
            [center_x, center_y - height * 0.1],     # Upper center
            [center_x, center_y + height * 0.1],     # Lower center
            [center_x - width * 0.1, center_y],      # Left center
            [center_x + width * 0.1, center_y],      # Right center
            
            # Diamond pattern around center
            [center_x - width * 0.15, center_y - height * 0.15],  # Upper left
            [center_x + width * 0.15, center_y - height * 0.15],  # Upper right
            [center_x - width * 0.15, center_y + height * 0.15],  # Lower left
            [center_x + width * 0.15, center_y + height * 0.15],  # Lower right
        ]
        
        # Define background points
        background_points = [
            # Top edge points
            [0, 0],                     # Top-left corner
            [width//4, 0],              # Top-left quarter
            [width//2, 0],              # Top center
            [3*width//4, 0],           # Top-right quarter
            [width-1, 0],               # Top-right corner
            
            # Bottom edge points
            [0, height-1],              # Bottom-left corner
            [width//4, height-1],       # Bottom-left quarter
            [width//2, height-1],       # Bottom center
            [3*width//4, height-1],     # Bottom-right quarter
            [width-1, height-1],        # Bottom-right corner
            
            # Side points
            [0, height//4],             # Left side upper quarter
            [0, height//2],             # Left side middle
            [0, 3*height//4],          # Left side lower quarter
            [width-1, height//4],       # Right side upper quarter
            [width-1, height//2],       # Right side middle
            [width-1, 3*height//4],    # Right side lower quarter
            
            # Ground area points (to better separate ground from object)
            [width//4, height-height//4],     # Lower-left ground
            [width//2, height-height//4],     # Lower-center ground
            [3*width//4, height-height//4],   # Lower-right ground
        ]
        
        # Combine all points
        all_points = foreground_points + background_points
        
        # Create labels (1 for foreground, 0 for background)
        labels = np.zeros(len(all_points), dtype=np.int32)
        labels[:len(foreground_points)] = 1  # Mark foreground points
        
        # Convert to numpy array
        points = np.array(all_points, dtype=np.float32)
        
        # Debug information
        logging.info(f"Created {len(foreground_points)} foreground points and {len(background_points)} background points")
        logging.info(f"Center point at ({center_x}, {center_y}) with {upward_shift:.1f}px upward shift")
        
        return points, labels

    def detect_body_keypoints(self, frame):
        """
        Detect human body and return keypoints
        """
        # Initialize body detector if not already done
        if self.body_detector is None:
            self.body_detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            self.body_detector.eval()
            self.body_detector.to(self.device)

        # Convert frame to tensor
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.body_detector(frame_tensor)

        # Get the highest confidence person detection
        scores = predictions[0]['scores']
        boxes = predictions[0]['boxes']

        if len(scores) > 0:
            best_idx = torch.argmax(scores)
            box = boxes[best_idx].cpu().numpy()
            
            # Calculate keypoints from the bounding box
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            keypoints = np.array([
                [center_x, center_y - height*0.2],  # Upper center
                [center_x - width*0.2, center_y - height*0.2],  # Upper left
                [center_x + width*0.2, center_y - height*0.2],  # Upper right
                [center_x, center_y],  # Center
                [center_x, center_y - height*0.3],  # Top
            ], dtype=np.float32)

            return keypoints
        else:
            # If no person is detected, fall back to center point
            # Fallback to center points if no person detected
            logging.warning("No person detected, falling back to center points")
            #height, width = frame.shape[:2]
            #center = np.array([[width // 2, height // 2 + (height*0.3)]], dtype=np.float32)
            #return np.tile(center, (5, 1))  # Return 5 identical center points as fallback

            return self.detect_center_keypoints(frame)

    def remove_background(self, frame, mask, bg_color):
        """
        Remove background with transparency support
        Returns RGBA image with alpha channel
        """
        # Convert mask to proper format
        mask = mask.astype(np.float32)
        
        # Apply sigmoid to get probabilities
        mask = 1 / (1 + np.exp(-mask))
        
        # Threshold the mask
        mask = (mask > 0.5).astype(np.uint8)
        
        # Ensure mask has same dimensions as frame
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        
        # Create RGBA image
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        
        # Set alpha channel based on mask
        rgba[:, :, 3] = mask * 255
        
        # Create mask for background
        bg_mask = (mask == 0)
        
        # Set background pixels to transparent
        rgba[bg_mask, 3] = 0
        
        # Optional: Set background color
        if bg_color != (0, 0, 0):
            rgba[bg_mask, 0:3] = bg_color
        
        return rgba

    def clean_hair_area(self, original, processed, mask, bg_color):
        # Create a dilated mask to capture the hair edges
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
        hair_edge_mask = cv2.subtract(dilated_mask, mask)

        # Calculate the average color of the removed background
        bg_sample = cv2.bitwise_and(original, original, mask=cv2.bitwise_not(dilated_mask))
        bg_average = cv2.mean(bg_sample)[:3]

        # Create a color distance map
        color_distances = np.sqrt(np.sum((original.astype(np.float32) - bg_average) ** 2, axis=2))

        # Normalize color distances
        color_distances = (color_distances - color_distances.min()) / (color_distances.max() - color_distances.min())

        # Create an alpha mask based on color distance
        alpha = (1 - color_distances) * (hair_edge_mask / 255.0)
        alpha = np.clip(alpha, 0, 1)

        # Blend the hair edge area
        for c in range(3):
            processed[:, :, c] = processed[:, :, c] * (1 - alpha) + bg_color[c] * alpha

        return processed
    
    def get_video_info(self, video_path: str) -> Tuple[str, dict]:
        """
        Get video format and metadata using OpenCV
        """
        try:
            # Open video file with OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {video_path}")
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            
            # Convert fourcc integer to string representation
            fourcc_bytes = bytes([
                fourcc_int & 0xFF,
                (fourcc_int >> 8) & 0xFF,
                (fourcc_int >> 16) & 0xFF,
                (fourcc_int >> 24) & 0xFF
            ])
            fourcc = fourcc_bytes.decode('ascii', errors='replace').strip('\0')
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0
            
            # Determine format name based on extension
            _, ext = os.path.splitext(video_path)
            format_name = ext.lower().lstrip('.')
            
            # Create info dictionary similar to ffprobe output
            info = {
                "format": {
                    "format_name": format_name,
                    "duration": str(duration),
                    "filename": video_path,
                },
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": fourcc,
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "frame_count": frame_count,
                    }
                ]
            }
            
            cap.release()
            
            # For mp4/mov files, assume h264 codec
            if format_name in ['mp4', 'mov']:
                info["streams"][0]["codec_name"] = 'h264'
                
            return format_name, info
            
        except Exception as e:
            logging.error(f"Error getting video info: {str(e)}")
            raise

    def ensure_mp4_format(
        self,
        input_video_path: str,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Check if video can be opened with OpenCV and return the path.
        
        Args:
            input_video_path: Path to input video
            output_dir: Optional directory for output file (unused, kept for compatibility)
        
        Returns:
            Path to video
        """
        try:
            # Try to open the video with OpenCV to verify it's readable
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video file with OpenCV: {input_video_path}")
                
            # Get basic video info
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Log video information
            logging.info(f"Video dimensions: {width}x{height} at {fps:.2f} fps, {frame_count} frames")
            
            # Check if video properties are valid
            if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
                logging.warning("Video has invalid properties but will attempt to process anyway")
                
            cap.release()
            
            # Return the input path as we're not converting formats
            return input_video_path
            
        except Exception as e:
            logging.error(f"Error checking video format: {str(e)}")
            raise
    
    def refine_mask(self, mask, kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Clean up small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def calculate_optimal_chunk_size(self, total_frames, num_cores):
        """Calculate optimal chunk size based on number of frames and cores"""
        
        # Base minimum chunk size - avoid too small chunks to reduce overhead
        MIN_CHUNK_SIZE = 10
        # Base maximum chunk size - avoid too large chunks that could cause memory issues
        MAX_CHUNK_SIZE = 200
        
        # Calculate a base chunk size that scales with the total number of frames
        # Aim to have enough chunks to keep all cores busy with some extra chunks for load balancing
        target_chunks_per_core = 4  # Multiple chunks per core for better load balancing
        base_chunk_size = max(MIN_CHUNK_SIZE, total_frames // (num_cores * target_chunks_per_core))
        
        # Adjust chunk size based on total frames
        if total_frames < 100:
            # For small frame counts, use smaller chunks
            chunk_size = min(base_chunk_size, total_frames // 2)
        elif total_frames < 1000:
            # For medium frame counts, use the base calculation
            chunk_size = base_chunk_size
        else:
            # For large frame counts, cap the chunk size
            chunk_size = min(base_chunk_size, MAX_CHUNK_SIZE)
        
        # Ensure chunk size doesn't exceed total frames
        chunk_size = min(chunk_size, total_frames)
        
        return int(chunk_size)

    def check_and_remove_first_frame(self, output_dir: Path, threshold_ratio: float = 1.5) -> bool:
        """
        Check and remove first frame if it contains significantly more OR less non-transparent 
        pixels than the next frame.
        
        Args:
            output_dir: Directory containing the PNG frames
            threshold_ratio: Ratio threshold for comparing non-transparent pixels
                        (remove if ratio > threshold_ratio OR ratio < 1/threshold_ratio)
        
        Returns:
            bool: True if first frame was removed, False otherwise
        """
        try:
            # Get sorted list of PNG files
            frames = sorted([f for f in output_dir.glob('frame_*.png')])
            if len(frames) < 2:
                logging.warning("Not enough frames to compare")
                return False
                
            # Read first two frames
            first_frame = cv2.imread(str(frames[0]), cv2.IMREAD_UNCHANGED)
            second_frame = cv2.imread(str(frames[1]), cv2.IMREAD_UNCHANGED)
            
            if first_frame is None or second_frame is None:
                logging.error("Failed to read frames")
                return False
                
            # Count non-transparent pixels (alpha channel > 0)
            first_non_transparent = np.sum(first_frame[:,:,3] > 0)
            second_non_transparent = np.sum(second_frame[:,:,3] > 0)
            
            logging.info(f"First frame non-transparent pixels: {first_non_transparent}")
            logging.info(f"Second frame non-transparent pixels: {second_non_transparent}")
            
            # Calculate ratio
            if second_non_transparent == 0:
                ratio = float('inf')
            else:
                ratio = first_non_transparent / second_non_transparent
                
            logging.info(f"Ratio of non-transparent pixels: {ratio:.2f}")
            
            # Remove first frame if ratio is either too high or too low
            if ratio > threshold_ratio or ratio < (1.0 / threshold_ratio):
                reason = "high" if ratio > threshold_ratio else "low"
                logging.info(f"Removing first frame due to {reason} non-transparent pixel ratio ({ratio:.2f})")
                frames[0].unlink()
                return True
                
            return False
                
        except Exception as e:
            logging.error(f"Error in check_and_remove_first_frame: {str(e)}")
            return False


if __name__ == '__main__':
    # Create Argument Parser
    parser = argparse.ArgumentParser(
        prog='',
        description=''
    )

    # Define the Arguments
    parser.add_argument(
        '-i', '--input_file_path',
        required=True,
        default=None,
        action='store',
        help='Target data input for video file'
    )

    # Define the Arguments
    parser.add_argument(
        '-o', '--output_image_dir',
        required=True,
        default=None,
        action='store',
        help='Target data output for video frames'
    )

    # Define the Arguments
    parser.add_argument(
        '-n', '--num_frames',
        required=False,
        default="300",
        action='store',
        help='Target number of frames to extract from video frames'
    )

    # Define the Arguments
    parser.add_argument(
        '-mt', '--mask_threshold',
        required=False,
        default="0.6",
        action='store',
        help='Target threshold to use on mask. If object doesnt have large contrast from background, use lower number like 0.38'
    )

    args = parser.parse_args()

    input_path = args.input_file_path
    output_path = args.output_image_dir
    num_frames = int(args.num_frames)
    mask_threshold = float(args.mask_threshold)

    try:
        # Set multiprocessing start method to 'spawn'
        multiprocessing.set_start_method('spawn', force=True)

        # First create an instance of the Predictor class
        predictor = Predictor()

        # Record start time
        start_time = datetime.now()
        temp_path = None
        if input_path == output_path:
            temp_path = f"{input_path}_temp"
            os.rename(input_path, temp_path)
            input_path = temp_path
        # Call the setup method first (required)
        predictor.setup()

        # Then call predict with your input video and optional background color
        input_video_path = Path(input_path) #wolf.mp4
        output_path = Path(output_path)
        num_frames = num_frames + 1 # Initial image for body detection, use other frames for sam2

        # Call predict method
        output_video = predictor.predict(
            input_video=input_video_path,
            num_frames=num_frames,
            output_path=output_path,
            mask_threshold=mask_threshold,  # Adjust this value between 0.0 and 1.0 lion=0.38, angler-fish=0.6
            tracking_mode="body" # "center" "body"
        )

        # In the main execution, after predict() completes:
        if output_video:  # If processing was successful
            # Check and potentially remove first frame
            removed = predictor.check_and_remove_first_frame(
                output_path,
                threshold_ratio=1.5  # Adjust this value based on your needs
            )
            if removed:
                logging.info("First frame was removed due to excessive non-transparent pixels")

        # Calculate elapsed time
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        if temp_path is not None:
            shutil.rmtree(temp_path, ignore_errors=True)
        # output_video will be a Path object pointing to the processed video
        print(f"Processed video saved at: {output_video}")
        print(f"Total processing time: {elapsed_time}")
    except Exception as e:
        logging.exception("Error in main execution")
        raise
    finally:
        # Cleanup
        torch.cuda.empty_cache()
