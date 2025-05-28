#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Extract no-blurry frames from a video file, e.g., for NeRF.

It uses the variance of the Laplacian of each frame as a blur indicator.
First, it selects the best frame over each period_small frame period
(by default 1 second). Then, it selects the same number of frames (to
get num_frames_target frames in total) from each period_large period.
Each time a frame f is selected, frames from f-spacing/2 to f+spacing/2
are invalidated, where spacing = num_frames_total / num_frames_target.

Author: F. Devernay (deverf@)

Functions:
video_info(): Get number of frames and fps for a video file.
compute_laplacian_variance(): Compute the variance of the Laplacian
    for each frame in the video.
select_frames(): Select non-blurry frames.
extract_frames_opencv(): Extract frames to png using OpenCV.
"""

import logging
import math
import os
import shutil
from collections import deque
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple, Union

import cv2  # type: ignore
import numpy as np
import numpy.typing as npt
import subprocess

try:
    from natsort import natsorted  # type: ignore   # Only for sorting "naturally" input images
except ImportError:
    natsorted = None
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

_LOGGER = logging.getLogger(__name__)

# OpenCV has an issue with GoPro videos, because they contain other tracks than the video track
# (e.g., GPS data), and OpenCV does not skip those tracks.
# The solution proposed in https://stackoverflow.com/a/55644446 was to retry until OpenCV
# succeeds in reading the video track. We implemented that, but with a maximum number of retries,
# because we don't want to wait forever. The default value of 20 retries was chosen by testing
# on a GoPro video, which required 12 retries
_MAX_CVCAPREAD_RETRIES = 20


MaskMethodBasic = Literal["full", "center"]
MaskMethodU2Net = Literal["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta"]
MaskMethod = Union[MaskMethodBasic, MaskMethodU2Net]

IMAGE_SUFFIXES = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")

def reduce_images_to_target(image_dir: str, target_count: int) -> int:
    """
    Reduces the number of images in a directory to the target count by removing
    images in an equidistant pattern.
    
    Args:
        image_dir (str): Path to directory containing images
        target_count (int): Desired number of images to keep
        
    Returns:
        int: Number of images removed
    """
    # Define supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    # Get list of image files
    image_files = [
        f for f in os.listdir(image_dir) 
        if os.path.isfile(os.path.join(image_dir, f)) 
        and f.lower().endswith(image_extensions)
    ]
    
    current_count = len(image_files)
    
    # If current count is less than or equal to target, do nothing
    if current_count <= target_count:
        return 0
    
    # Sort files to ensure consistent behavior
    image_files.sort()
    
    # Calculate which indices to keep using numpy's linspace for even distribution
    if target_count > 1:
        # Use linspace to get evenly spaced indices
        indices_to_keep = set(int(idx) for idx in np.linspace(0, current_count-1, target_count))
    else:
        # If target is 1, just keep the middle image
        indices_to_keep = {current_count // 2}
    
    # Remove files that aren't at the kept indices
    removed_count = 0
    for idx, filename in enumerate(image_files):
        if idx not in indices_to_keep:
            file_path = os.path.join(image_dir, filename)
            try:
                os.remove(file_path)
                removed_count += 1
            except Exception as e:
                print(f"Error removing {filename}: {str(e)}")
    
    return removed_count

def rotate_images_in_directory(input_dir, degrees=90):
    """
    Rotates all images in a directory by 90 degrees clockwise using OpenCV
    and overwrites the original images.
    
    Args:
        input_dir (str): Path to directory containing images to rotate
        degrees (int, optional): Degrees to rotate. Default is 90 degrees clockwise.
    
    Returns:
        int: Number of images successfully rotated
    """
    # Define supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    # Counter for successfully rotated images
    success_count = 0
    
    # Get all files in the directory
    files = os.listdir(input_dir)
    
    for filename in files:
        # Check if file is an image
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(input_dir, filename)
            
            try:
                # Read the image
                img = cv2.imread(input_path)
                
                if img is None:
                    print(f"Warning: Could not read image {filename}")
                    continue
                
                # Rotate the image 90 degrees clockwise
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                
                # Save the rotated image, overwriting the original
                cv2.imwrite(input_path, rotated_img)
                success_count += 1
                print(f"Rotated and overwritten: {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print(f"Successfully rotated {success_count} images")
    return success_count


def video_info(video_path: Path) -> Tuple[float, int]:
    """Get number of frames and fps for a video file.

    Parameters
    ----------
    video_path : Path
        Path to the video file.


    Returns
    -------
    float
        frames per second.
    int
        number of frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fps, int(frame_count)


def compute_laplacian_variance_u2net(
    input_path: Union[Path, List[Path]],
    fps: Optional[float] = None,
    mask_method: MaskMethod = "silueta",
    decay_s: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Compute the Laplacian variance for every frame in a video, masked with salient
    object detection and temporal filtering on the salient object mask.

    Parameters
    ----------
    input_path : Union[Path,List[Path]]
        Path to the video file or images.
    model : str
        The name of a U2Net model to use for salient object detection.
        Models are stored by the rembg package in ~/.u2net
        One of ["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta"].
    decay_s : float
        Decay of the temporal filter (single pole recursive low-pass filter) in seconds.
        0 means no decay. Recommended value is 1.

    Returns
    -------
    NDArray[float32]
        1-D array of the Laplacian values
    """

    from PIL import Image
    from rembg import new_session  # type: ignore

    session = new_session(mask_method)

    input_is_images = isinstance(input_path, list)
    if input_is_images:
        input_path_name = "images"
        if fps is None:
            raise ValueError("fps must not be None if video_path is a list of images")
        assert isinstance(input_path, list)
        frame_count = len(input_path)
    else:
        input_path = Path(input_path)
        input_path_name = str(input_path)
        cap = cv2.VideoCapture(str(input_path))
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    _LOGGER.info(
        f"Computing Laplacian variance for {input_path_name}: fps={fps}, frame_count={frame_count}"
    )
    _LOGGER.info(f"Masking using U2Net with {mask_method} model with smoothing (decay={decay_s}s).")
    v = []

    # x parameter for the single pole recursive low-pass filter, see p. 322 of
    # https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch19.pdf
    if decay_s > 0.0:
        x = math.exp(-1 / (decay_s * fps))
        if int(x * 255) == 0:
            raise ValueError("decay ifs too long (max decay: {1/(fps*math.log(1./255))})")
    else:
        x = 0.0
    gray_prev = None
    mask_filtered_prev = None
    flow = None
    max_retries = 0
    for f in tqdm(range(frame_count)):
        """
        Loop invariant:
        - mask_filtered_prev contains the filtered mask from the previous frame.
        - if video_path is not None, gray_prev contains the previous grayscale frame.
        Loop iteration:
        - if video_path is None:
            - set gray to None
            - set mask_filtered_prev_warped to mask_filtered_prev (no warp)
        - else:
            - read current frame as gray
            - compute OF to previous frame (gray_prev)
            - compute backward warp map from flow
            - warp mask_filtered_prev to mask_filtered_prev_warped
        - read mask from current frame as mask
        - compute the filtered mask as mask_filtered = ((1 - x) * mask + x * mask_filtered_prev_warped
        - save mask_filtered
        - set mask_filtered_prev = mask_filtered, gray_prev = gray
        """
        if input_is_images:
            assert isinstance(input_path, list)
            frame = cv2.imread(str(input_path[f]), cv2.IMREAD_COLOR)
        else:
            retries = -1
            success = False
            while not success and f < frame_count and retries < _MAX_CVCAPREAD_RETRIES:
                success, frame = cap.read()
                retries += 1
            if not success:
                _LOGGER.info(
                    f"Can't read frame {f + 1} from {input_path}, even with {_MAX_CVCAPREAD_RETRIES} retries"
                )
            else:
                max_retries = max(max_retries, retries)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if x > 0.0 and gray_prev is not None:
            # we re-use flow here, but it's really optional
            flow = cv2.calcOpticalFlowFarneback(gray, gray_prev, flow, 0.5, 3, 15, 3, 5, 1.2, 0)
            # transform flow to a map for use with cv2.remap
            flow[:, :, 0] += np.arange(flow.shape[1])
            flow[:, :, 1] += np.arange(flow.shape[0])[:, np.newaxis]
            mask_filtered_prev_warped = cv2.remap(mask_filtered_prev, flow, None, cv2.INTER_LINEAR)

        # Predict salient object mask using U2-Net
        mask = np.array(session.predict(Image.fromarray(frame))[0])

        if x > 0.0 and mask_filtered_prev is not None:
            # single-pole recursive filter, see p. 322 of
            # https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch19.pdf
            mask_filtered = ((1 - x) * mask + x * mask_filtered_prev_warped).astype(np.uint8)
        else:
            mask_filtered = mask

        v.append(cv2.Laplacian(frame, cv2.CV_32F)[mask_filtered > 127].var())
        if x > 0.0:
            mask_filtered_prev = mask_filtered
            gray_prev = gray

    if not input_is_images:
        # cleanup (should not be necessary)
        cap.release()
    if max_retries > 0:
        _LOGGER.debug(f"Used {max_retries} retries to read frames")
    assert len(v) == frame_count

    return np.asarray(v, dtype=np.float32)


def compute_laplacian_variance_basic(
    input_path: Union[Path, List[Path]],
    mask_method: MaskMethod = "full",
    thread_count: int = -1,
) -> npt.NDArray[np.float32]:
    """Compute the Laplacian variance for every frame in a video.

    Parameters
    ----------
    input_path : Union[Path,List[Path]]
        Path to the video file or images.
    mask_method : str
        One of ["full", "center"].
        The name of a masking method to select in-focus frames:
        "full" means to use the full frame,
        "center" means to use the frame center (1/3 of each frame dimension).
    thread_count : int
        Number of threads to use for threaded execution. 0 means
        no threaded execution. -1 means to use os.cpu_count().

    Returns
    -------
    NDArray[float32]
        1-D array of the Laplacian values
    """

    class DummyTask:
        def __init__(self, data):
            self.data = data

        def ready(self):
            return True

        def get(self):
            return self.data

    def process_frame(frame):
        # some intensive computation...
        if mask_method == "full":
            return cv2.Laplacian(frame, cv2.CV_32F).var()
        elif mask_method == "center":
            return cv2.Laplacian(
                frame[
                    frame.shape[0] // 3 : frame.shape[0] * 2 // 3,
                    frame.shape[1] // 3 : frame.shape[1] * 2 // 3,
                    :,
                ],
                cv2.CV_32F,
            ).var()
        else:
            raise ValueError(f"Unknown mask_method {mask_method}")

    if mask_method not in ["full", "center"]:
        raise ValueError('Only supported values for mask_method are: "full", "center"')

    input_is_images = isinstance(input_path, list)
    if input_is_images:
        assert isinstance(input_path, list)
        frame_count = len(input_path)
        _LOGGER.info(f"Computing Laplacian variance for images: frame_count={frame_count}")

    else:
        input_path = Path(input_path)
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _LOGGER.info(
            f"Computing Laplacian variance for {input_path}: fps={fps}, frame_count={frame_count}"
        )
    v = []

    if thread_count == -1:
        cpu_count: Optional[int] = os.cpu_count()
        if cpu_count is not None:
            thread_count = cpu_count
    done_reading = False
    done_processing = False
    pending: Deque[Any] = deque()
    f = 0
    max_retries = 0
    try:
        if thread_count > 0:
            pool = ThreadPool(processes=thread_count)
        with logging_redirect_tqdm(loggers=[_LOGGER]), tqdm(
            total=frame_count, disable=not _LOGGER.isEnabledFor(logging.INFO)
        ) as pbar:
            while not done_processing:
                while pending and pending[0].ready():
                    res = pending.popleft().get()
                    if res is None:
                        _LOGGER.debug("done processing!")
                        done_processing = True
                    else:
                        v.append(res)
                        pbar.update(1)

                if not done_reading and (thread_count == 0 or len(pending) < thread_count):
                    if input_is_images:
                        success = f < frame_count
                        if success:
                            assert isinstance(input_path, list)
                            frame = cv2.imread(str(input_path[f]), cv2.IMREAD_COLOR)
                            f += 1
                    else:
                        retries = -1
                        success = False
                        while not success and f < frame_count and retries < _MAX_CVCAPREAD_RETRIES:
                            success, frame = cap.read()
                            retries += 1
                        if not success:
                            _LOGGER.info(
                                f"Can't read frame {f + 1} from {input_path}, even with {_MAX_CVCAPREAD_RETRIES} retries"
                            )
                        else:
                            f += 1
                            max_retries = max(max_retries, retries)
                    task: Any
                    if success:
                        if thread_count > 0:
                            task = pool.apply_async(process_frame, (frame.copy(),))
                        else:
                            task = DummyTask(
                                process_frame(
                                    frame,
                                )
                            )
                    else:
                        task = DummyTask(None)
                        done_reading = True
                        _LOGGER.debug("done reading!")
                    pending.append(task)
    finally:
        if thread_count > 0:
            pool.close()
            pool.terminate()
    if not input_is_images:
        # cleanup (should not be necessary)
        cap.release()
    if max_retries > 0:
        _LOGGER.debug(f"Used {max_retries} retries to read frames")
    #assert len(v) == frame_count

    return np.asarray(v, dtype=np.float32)


def compute_laplacian_variance(
    input_path: Union[Path, List[Path]],
    mask_method: MaskMethod = "full",
    decay_frames: float = 0.0,
    thread_count: int = -1,
) -> npt.NDArray[np.float32]:
    """Compute the Laplacian variance for every frame in a video.

    Parameters
    ----------
    input_path : Union[Path, List[Path]]
        Path to the video file or directory containingimages.
    fps : Optional[float]
        FPS, only necessary if the mask_method is using U2Net with decay_s
    mask_method : Optional[str]
        One of ["full", "center", "u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta"].
        The name of a masking method to select in-focus frames:
        "full" means to use the full frame,
        "center" means to use the frame center (1/3 of each frame dimension),
        other values are U2Net models for salient object detection.
        Models are stored by the rembg package in ~/.u2net
    decay_frames : float
        For U2Net-based masking, decay of the temporal filter (single pole recursive low-pass filter) in frames.
        0 means no temporal smoothing. Recommended value is 1.
    thread_count : int
        Number of threads to use for threaded execution. 0 means
        no threaded execution. -1 means to use os.cpu_count().

    Returns
    -------
    NDArray[float32]
        1-D array of the Laplacian values
    """

    if mask_method in ("u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta"):
        return compute_laplacian_variance_u2net(input_path, mask_method, decay_frames)
    if mask_method in ("full", "center"):
        return compute_laplacian_variance_basic(input_path, mask_method, thread_count)
    raise ValueError(f"Unknown mask_method {mask_method}")


def select_frames(
    video_num_frames: int,
    video_fps: float,
    num_frames_target: int,
    input_path_name: str,
    image_dir: str,
    laplacian_var: Optional[npt.NDArray[np.float32]] = None,
    period_small_s: float = 0.0,
    period_large_s: float = 0.0

) -> npt.NDArray[np.int32]:
    """Converts a video into a sequence of images.

    Parameters
    ----------
    video_num_frames : int
        Number of frames of the video.
    video_fps : float
        Frames per second in the video.
    num_frames_target : int
        Number of frames to extract.
    laplacian_var : NDArray[float32], optional
        Laplacian variance for each frame in the video. Can be
        computed by compute_laplacian_variance.
        If None, will do regular sampling with a fixed frame
        interval (as in nerfstudio 0.1.16), and the two periods
        need to be zero. Note that this may result in a different
        number of frames than num_frames_target.
    period_small_s: float
        Periods (in seconds) for initial frame selection. If zero or less than
        duration / num_frames_target, will be computed to give num_frames_target
        frames. 1 second is recommended.
    period_large_s: float
        Periods (in seconds) for secondary frame selection. The timeline is divided
        in intervals of this size, and the remaining frames are evenly distributed
        among those large periods. If zero, the whole video is considered as a single
        large period. 4 seconds is recommended.


    Returns
    -------
    NDArray[int]
        Sorted array of frame numbers to be extracted. Length may be
        different than num_frames_target.

    Raises
    ------
    ValueError
        Wrong parameter values are entered.
    """
    if laplacian_var is None and (period_small_s != 0 or period_large_s != 0):
        raise ValueError("period_small_s and period_large_s must be zero if laplacian_var is None")

    num_frames = video_num_frames
    if num_frames <= num_frames_target:
        _LOGGER.warning(
            f"Video has {num_frames} frames, which is less than the target {num_frames_target}."
        )
        num_frames_target = num_frames
    if laplacian_var is not None and len(laplacian_var) != num_frames:
        _LOGGER.info("""Using simple video to images script instead 
                     because laplacian_var does not have the same size as the number of frames in the video.""")
        try:
            subprocess.run([
                "python",
                "video_processing/simple_video_to_images.py",
                "-i", input_path_name,
                "-o", image_dir,
                "-n", str(num_frames_target)
            ], check=True)
            _LOGGER.info("Successfully ran video to image fallback script")
            exit(0)
        except Exception as e:
            raise RuntimeError("Error using video to image fallback script: {e}") from e
        
        #raise ValueError(
        #    "laplacian_var does not have the same size as the number of frames in the video."
        #)

    frames_selected = []
    spacing = num_frames // num_frames_target
    if laplacian_var is None:
        # nerfstudio-like frame selection
        return np.asarray(list(range(0, num_frames, spacing)), dtype=int)

    fps = video_fps

    # arrays representing which frames can still be selected,
    # their frame number, and their value
    valid_all = np.ones((num_frames,), dtype=bool)
    frames_all = np.asarray(range(num_frames))
    var_all = laplacian_var

    # rembg can lead to nans in var, let's just replace them with zeros so
    # calculation below still works - this means these frames won't be selected
    var_all[np.isnan(var_all)] = 0

    with logging_redirect_tqdm(loggers=[_LOGGER]), tqdm(
        total=num_frames_target, disable=not _LOGGER.isEnabledFor(logging.INFO)
    ) as pbar:
        # never fetch more than num_frames_target frames
        if period_small_s == 0:
            num_frames_sparse = num_frames_target
        else:
            num_frames_sparse = min(num_frames_target, int(num_frames / (fps * period_small_s)))
        for j in range(num_frames_sparse):
            # find the best frame in the range [fj,fjn[
            fj = int(j * num_frames / num_frames_sparse)
            fjn = int((j + 1) * num_frames / num_frames_sparse)
            vj = var_all[fj:fjn]
            f = fj + int(vj.argmax())
            assert var_all[f] == vj[np.asarray(vj).argmax()]
            frames_selected.append(f)
            pbar.update(1)
            # invalidate frames next to the selected one
            f_min = f - spacing // 2
            f_max = f_min + spacing
            f_min = max(0, f_min)
            f_max = min(num_frames, f_max)
            valid_all[f_min:f_max] = False

        _LOGGER.debug(f"Selected {len(frames_selected)} frames with a period of {period_small_s}")
        if len(frames_selected) >= num_frames_target:
            # They are sorted already
            return np.asarray(frames_selected, dtype=int)

        # now split into sections of approximate duration period_large_s
        if period_large_s == 0:
            num_period = 1
        else:
            num_period = math.ceil(num_frames / (fps * period_large_s))
        for j in range(num_period):
            # we still have num_period-j periods to go,
            # and num_frames_target - len(frames_selected) frames to go
            num_frames_period = int((num_frames_target - len(frames_selected)) / (num_period - j))
            _LOGGER.debug(f"Adding {num_frames_period} to the {j}th period of {period_large_s}")
            if num_frames_period <= 0:
                if len(frames_selected) >= num_frames_target:
                    # we're done already, break the 'for' loop
                    break
                # let's add at least 1 frame to this period
                # and continue
                num_frames_period = 1
            # find the best num_frames_period frames in the range [fj,fjn[
            fj = int(j * num_frames / num_period)
            fjn = int((j + 1) * num_frames / num_period)
            for k in range(num_frames_period):
                valid_in_period = valid_all[fj:fjn]
                values_in_period = var_all[fj:fjn]
                frames_in_period = frames_all[fj:fjn]
                values_valid = values_in_period[valid_in_period]
                frames_valid = frames_in_period[valid_in_period]
                if values_valid.size == 0:
                    # no more valid frames in the period
                    break
                best_p = np.argmax(values_valid)
                f = frames_valid[best_p]
                frames_selected.append(f)
                pbar.update(1)
                # invalidate frames next to the selected one
                f_min = f - spacing // 2
                f_max = f_min + spacing
                f_min = max(0, f_min)
                f_max = min(num_frames, f_max)
                valid_all[f_min:f_max] = False

    return np.asarray(sorted(frames_selected), dtype=int)


def create_image_dir(image_dir: Path, force: bool = True):
    """Creates the image directory if it doesn't exist, and optionally cleans it if not empty
    
    Parameters
    ----------
    image_dir : Path
        Path to the directory to create or clean
    force : bool, optional
        If True, automatically remove existing images; if False, raise error if images exist
    """
    if image_dir.is_dir():
        # Check for existing images in folder
        files_to_delete = [
            f for f in image_dir.iterdir() if f.name[0] != "." and f.suffix in IMAGE_SUFFIXES
        ]  # ignore hidden files
        if files_to_delete:
            if force:
                _LOGGER.info(f"Removing {len(files_to_delete)} existing images from {image_dir}")
                for file_path in files_to_delete:
                    try:
                        file_path.unlink()
                    except Exception as e:
                        _LOGGER.warning(f"Failed to delete {file_path}: {e}")
            else:
                raise FileExistsError(
                    f"{image_dir} contains {len(files_to_delete)} PNG or JPEG files, remove them before proceeding"
                )
    else:
        image_dir.mkdir(parents=True)


def copy_images(
    input_path: List[Path],
    image_dir: Path,
    frames: npt.NDArray[np.int32],
    pattern: str = "frame_{i:03d}_{f:05d}",
) -> None:
    """Converts a video into a sequence of PNG images using OpenCV.
    Not recommended for HDR video (BT.2020).

    Parameters
    ----------
    input_path : List[Path]
        Path to the images.
    image_dir : Path
        Path to the output directory. Created if non-existing.
        Existing images are deleted.
    frames : NDArray[int]
        Sorted frames (computed by select_frames).
    pattern : str
        Python string format used for generating image names. i is the image index, f
        is the frame number.

    Raises
    ------
    FileExistsError
        image_dir already exists and contains PNG files.
    """

    create_image_dir(image_dir, force=True)
    out_filename = str(image_dir / pattern)

    _LOGGER.info(f"Copying frames to {out_filename}.SUFFIX")

    for i, f in enumerate(tqdm(frames, disable=not _LOGGER.isEnabledFor(logging.INFO))):
        shutil.copy(str(input_path[f]), out_filename.format(f=f, i=i) + input_path[f].suffix)


def extract_frames_opencv(
    video_path: Path,
    image_dir: Path,
    frames: npt.NDArray[np.int32],
    pattern: str = "frame_{i:03d}_{f:05d}",
) -> None:
    """Converts a video into a sequence of PNG images using OpenCV.
    Not recommended for HDR video (BT.2020).

    Parameters
    ----------
    video_path : Path
        Path to the video file.
    image_dir : Path
        Path to the output directory. Created if non-existing.
        Existing images are deleted.
    frames : NDArray[int]
        Sorted frames (computed by select_frames).
    pattern : str
        Python string format used for generating image names. i is the image index, f
        is the frame number.

    Raises
    ------
    FileExistsError
        image_dir already exists and contains PNG files.
    """

    create_image_dir(image_dir, force=True)
    out_filename = str(image_dir / (pattern + ".png"))

    _LOGGER.info(f"Writing frames as {out_filename}")

    cap = cv2.VideoCapture(str(video_path))
    # Note: cap.set(cv2.CV_CAP_PROP_POS_FRAMES, f) is inaccurate if the video is not intra-only,
    # so we prefer grabbing all frames.

    with logging_redirect_tqdm(loggers=[_LOGGER]):
        prev_f = -1
        for i, f in enumerate(tqdm(frames, disable=not _LOGGER.isEnabledFor(logging.INFO))):
            assert f > prev_f
            for j in range(f - prev_f):
                res = cap.grab()
                assert res
            res, frame = cap.retrieve()
            filename = out_filename.format(f=f, i=i)
            cv2.imwrite(filename, frame)
            # _LOGGER.info(f"Wrote {filename}")
            prev_f = f
    # cleanup (should not be necessary)
    cap.release()

def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="""
    Extract no-blurry frames from a video file, e.g., for NeRF.

    It uses the variance of the Laplacian of each frame as a blur indicator.
    First, it selects the best frame over each period_small frame period
    (by default 1 second). Then, it selects the same number of frames (to
    get num_frames_target frames in total) from each period_large period.
    Each time a frame f is selected, frames from f-spacing/2 to f+spacing/2
    are invalidated, where spacing = num_frames_total / num_frames_target.
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-I",
        "--video-path",
        type=Path,
        help="Input video or a directory containing only individual image frames.",
    )
    parser.add_argument(
        "-r",
        "--frame-rate",
        type=float,
        default=None,
        help="Input frame rate (if input is a video, the video frame rate is ignored).",
    )
    if natsorted is not None:
        parser.add_argument(
            "-S",
            "--string-sort",
            action="store_true",
            help="Use string sorting to sort image filenames. The default is to use natsort's natural sorting (only applicable to images).",
        )
    parser.add_argument(
        "-l",
        "--list-frames",
        action="store_true",
        help="List the sorted images before selection, to check that sorting is right (only applicable to images).",
    )
    parser.add_argument(
        "-n",
        "--num-frames-target",
        type=int,
        default=300,
        help="Approximate number of frames to extract.",
    )
    parser.add_argument(
        "-p",
        "--period-small",
        type=float,
        default=1.0,
        help="""Period (s) to use to extract local non-blurred frames.
        Initially, one frame is selected per period_small interval.
        Set to 0 for the same behavior as nerfstudio 0.1.16 (regular sampling).""",
    )
    parser.add_argument(
        "-P",
        "--period-large",
        type=float,
        default=4.0,
        help="""Period (in s) to use to add remaining non-blurred frames.
        After selecting one frame per period_small interval,
        the remaining frames are selected per period_large interval.""",
    )
    parser.add_argument(
        "-O",
        "--image-dir",
        type=Path,
        default="images",
        help="Output image directory.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Delete PNG files from image_dir, if any.",
    )
    parser.add_argument(
        "-t",
        "--thread-count",
        type=int,
        default=-1,
        help="""Number of threads to use for threaded execution. 0 means
        no threaded execution. -1 means to use os.cpu_count()""",
    )
    parser.add_argument(
        "-m",
        "--mask-method",
        default="full",
        choices=[
            "full",
            "center",
            "u2net",
            "u2netp",
            "u2net_human_seg",
            "u2net_cloth_seg",
            "silueta",
        ],
        help="""The name of a masking method to select in-focus frames:
        "full" means to use the full frame,
        "center" means to use the frame center (1/3 of each frame dimension)
        - this is the recommended value when scanning an object.
        Other values are U2Net models for salient object detection.
        Models are stored by the rembg package in ~/.u2net
        u2net is 176.3MB, u2netp is 4.7MB, silueta is 43MB but performs almost as well as u2net.
        U2Net slows down the process by a ~10x factor.
        See https://github.com/danielgatis/rembg#installation to install rembg with the proper
        onnx runtime.""",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose execution.",
    )

    args = parser.parse_args(args)

    if args.verbose:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARNING
    logging.basicConfig(
        format="[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        level=logging_level,
    )

    video_path = args.video_path
    frame_rate = args.frame_rate
    list_frames = args.list_frames
    num_frames_target = args.num_frames_target
    period_small = args.period_small
    period_large = args.period_large
    image_dir = args.image_dir
    force = args.force
    thread_count = args.thread_count
    mask_method = args.mask_method

    input_is_images = video_path.is_dir()
    if natsorted is None:
        logging.warning(
            "natsort is not installed, only string sorting is available for image filenames"
        )
        string_sort = input_is_images
    else:
        string_sort = args.string_sort
    temp_path = None
    if input_is_images:
        print(f"Input is a directory: {video_path}")
        if frame_rate is None:
            parser.error("--frame-rate must be specified when video_path is a directory")
        if video_path == image_dir:
            temp_path = os.path.join(os.path.dirname(video_path), "temp_images")
            os.makedirs(temp_path, exist_ok=True)
            print(f"Moving {video_path} to {temp_path}...")
            os.rename(video_path, temp_path)
            video_path = temp_path
        video_path = [
            f for f in Path(temp_path).iterdir() if f.name[0] != "." and f.suffix in IMAGE_SUFFIXES
        ]  # ignore hidden files
        print(f"Number of images: {len(video_path)}")

        if string_sort:
            video_path = sorted(video_path)
        else:
            video_path = natsorted(video_path)
        if list_frames:
            return
    else:
        if frame_rate is not None:
            parser.error("Cannot use --frame-rate when input is a video")
        if string_sort:
            parser.error("Cannot use --string-sort when input is a video")
        if list_frames:
            parser.error("Cannot use --list-frames when input is a video")

    if image_dir.is_dir():
        # delete existing images in folder
        files_to_delete = [
            f for f in image_dir.iterdir() if f.name[0] != "." and f.suffix in IMAGE_SUFFIXES
        ]
        if files_to_delete:
            if force:
                logging.info(f"Deleting {len(files_to_delete)} PNGs and JPEGs from {image_dir}")
                with logging_redirect_tqdm(loggers=[_LOGGER]):
                    for img in tqdm(
                        files_to_delete, disable=not _LOGGER.isEnabledFor(logging.INFO)
                    ):
                        img.unlink()
            else:
                raise FileExistsError(
                    f"{image_dir} contains {len(files_to_delete)} PNG files, remove them before proceeding or use option -f/--force"
                )
    if input_is_images:
        input_path_name = "images"
        fps = frame_rate
        num_frames = len(video_path)
    else:
        input_path_name = str(video_path)
        fps, num_frames = video_info(video_path)
        if frame_rate is not None:
            fps = frame_rate
    logging.info(f"Processing video {input_path_name}: {num_frames} frames at {fps}FPS")
    if period_small == 0.0:
        logging.info("Using nerfstudio 0.1.16 (regular) frame sampling...")
        laplacian_var = None
    else:
        logging.info(
            f"Computing variance of Laplacian from video using mask method {mask_method}..."
        )
        laplacian_var = compute_laplacian_variance(
            video_path, mask_method=mask_method, thread_count=thread_count
        )
    logging.info("Selecting frames...")
    frames = select_frames(
        video_num_frames=num_frames,
        video_fps=fps,
        num_frames_target=num_frames_target,
        laplacian_var=laplacian_var,
        period_small_s=period_small,
        period_large_s=period_large,
        input_path_name=input_path_name,
        image_dir=image_dir
    )
    logging.info(f"Extracting {len(frames)} frames...")
    if input_is_images:
        copy_images(input_path=video_path, image_dir=image_dir, frames=frames)
    else:
        extract_frames_opencv(video_path=video_path, image_dir=image_dir, frames=frames)
    removed = reduce_images_to_target(image_dir, num_frames_target)
    print(f"Removed {removed} images to maintain image count of {str(num_frames_target)}")
    if temp_path is not None:
        shutil.rmtree(temp_path, ignore_errors=True)
    logging.info("Done!")

if __name__ == "__main__":
    main()
