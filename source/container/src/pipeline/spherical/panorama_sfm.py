"""
An example for running incremental SfM on 360 spherical panorama images.
"""

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import shutil

import cv2
import numpy as np
import PIL.ExifTags
import PIL.Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import pycolmap
from pycolmap import logging


@dataclass
class PanoRenderOptions:
    num_steps_yaw: int
    pitches_deg: Sequence[float]
    hfov_deg: float
    vfov_deg: float


PANO_RENDER_OPTIONS: dict[str, PanoRenderOptions] = {
    "overlapping": PanoRenderOptions(
        num_steps_yaw=4,
        pitches_deg=(-35.0, 0.0, 35.0),
        hfov_deg=90.0,
        vfov_deg=90.0,
    ),
    # Cubemap without top and bottom images.
    "non-overlapping": PanoRenderOptions(
        num_steps_yaw=4,
        pitches_deg=(0.0,),
        hfov_deg=90.0,
        vfov_deg=90.0,
    ),
}


def create_virtual_camera(
    pano_width: int,
    pano_height: int,
    hfov_deg: float,
    vfov_deg: float,
) -> pycolmap.Camera:
    """Create a virtual perspective camera."""
    image_width = int(pano_width * hfov_deg / 360)
    image_height = int(pano_height * vfov_deg / 180)
    focal = image_width / (2 * np.tan(np.deg2rad(hfov_deg) / 2))
    return pycolmap.Camera.create(
        0, "SIMPLE_PINHOLE", focal, image_width, image_height
    )


def get_virtual_camera_rays(camera: pycolmap.Camera) -> np.ndarray:
    size = (camera.width, camera.height)
    x, y = np.indices(size).astype(np.float32)
    xy = np.column_stack([x.ravel(), y.ravel()])
    # The center of the upper left most pixel has coordinate (0.5, 0.5)
    xy += 0.5
    xy_norm = camera.cam_from_img(xy)
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def spherical_img_from_cam(image_size, rays_in_cam: np.ndarray) -> np.ndarray:
    """Project rays into a 360 panorama (spherical) image."""
    if image_size[0] != image_size[1] * 2:
        raise ValueError("Only 360° panoramas are supported.")
    if rays_in_cam.ndim != 2 or rays_in_cam.shape[1] != 3:
        raise ValueError(f"{rays_in_cam.shape=} but expected (N,3).")
    r = rays_in_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u, v], -1) * image_size


def get_virtual_rotations(
    num_steps_yaw: int, pitches_deg: Sequence[float]
) -> Sequence[np.ndarray]:
    """Get the relative rotations of the virtual cameras w.r.t. the panorama."""
    # Assuming that the panos are approximately upright.
    cams_from_pano_r = []
    yaws = np.linspace(0, 360, num_steps_yaw, endpoint=False)
    for pitch_deg in pitches_deg:
        yaw_offset = (360 / num_steps_yaw / 2) if pitch_deg > 0 else 0
        for yaw_deg in yaws + yaw_offset:
            cam_from_pano_r = Rotation.from_euler(
                "XY", [-pitch_deg, -yaw_deg], degrees=True
            ).as_matrix()
            cams_from_pano_r.append(cam_from_pano_r)
    return cams_from_pano_r


def create_pano_rig_config(
    cams_from_pano_rotation: Sequence[np.ndarray], ref_idx: int = 0
) -> pycolmap.RigConfig:
    """Create a RigConfig for the given virtual rotations."""
    rig_cameras = []
    for idx, cam_from_pano_rotation in enumerate(cams_from_pano_rotation):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_rotation = (
                cam_from_pano_rotation @ cams_from_pano_rotation[ref_idx].T
            )
            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_rotation), np.zeros(3)
            )
        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=idx == ref_idx,
                image_prefix=f"pano_camera{idx}/",
                cam_from_rig=cam_from_rig,
            )
        )
    return pycolmap.RigConfig(cameras=rig_cameras)


class PanoProcessor:
    def __init__(
        self,
        pano_image_dir: Path,
        output_image_dir: Path,
        mask_dir: Path,
        render_options: PanoRenderOptions,
    ):
        self.render_options = render_options
        self.pano_image_dir = pano_image_dir
        self.output_image_dir = output_image_dir
        self.mask_dir = mask_dir

        self.cams_from_pano_rotation = get_virtual_rotations(
            num_steps_yaw=render_options.num_steps_yaw,
            pitches_deg=render_options.pitches_deg,
        )
        self.rig_config = create_pano_rig_config(self.cams_from_pano_rotation)

        # We assign each pano pixel to the virtual camera
        # with the closest camera center.
        self.cam_centers_in_pano = np.einsum(
            "nij,i->nj", self.cams_from_pano_rotation, [0, 0, 1]
        )

        self._lock = Lock()

        # These are initialized on the first pano image
        # to avoid recomputing the rays for each pano image.
        self._camera = None
        self._pano_size = None
        self._rays_in_cam = None

    def process(self, pano_name: str):
        pano_path = self.pano_image_dir / pano_name
        try:
            pano_image = PIL.Image.open(pano_path)
        except PIL.Image.UnidentifiedImageError:
            logging.info(f"Skipping file {pano_path} as it cannot be read.")
            return

        pano_exif = pano_image.getexif()
        pano_image = np.asarray(pano_image)
        gpsonly_exif = PIL.Image.Exif()
        gpsonly_exif[PIL.ExifTags.IFD.GPSInfo] = pano_exif.get_ifd(
            PIL.ExifTags.IFD.GPSInfo
        )

        pano_height, pano_width, *_ = pano_image.shape
        if pano_width != pano_height * 2:
            raise ValueError("Only 360° panoramas are supported.")

        with self._lock:
            if self._camera is None:  # First image, precompute rays once.
                self._camera = create_virtual_camera(
                    pano_width=pano_width,
                    pano_height=pano_height,
                    hfov_deg=self.render_options.hfov_deg,
                    vfov_deg=self.render_options.vfov_deg,
                )
                for rig_camera in self.rig_config.cameras:
                    rig_camera.camera = self._camera
                self._pano_size = (pano_width, pano_height)
                self._rays_in_cam = get_virtual_camera_rays(self._camera)
            else:  # Later images, verify consistent panoramas.
                if (pano_width, pano_height) != self._pano_size:
                    raise ValueError(
                        "Panoramas of different sizes are not supported."
                    )

        for cam_idx, cam_from_pano_r in enumerate(self.cams_from_pano_rotation):
            rays_in_pano = self._rays_in_cam @ cam_from_pano_r
            xy_in_pano = spherical_img_from_cam(self._pano_size, rays_in_pano)
            xy_in_pano = xy_in_pano.reshape(
                self._camera.width, self._camera.height, 2
            ).astype(np.float32)
            xy_in_pano -= 0.5  # COLMAP to OpenCV pixel origin.
            image = cv2.remap(
                pano_image,
                *np.moveaxis(xy_in_pano, [0, 1, 2], [2, 1, 0]),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
            # We define a mask such that each pixel of the panorama has its
            # features extracted only in a single virtual camera.
            closest_camera = np.argmax(
                rays_in_pano @ self.cam_centers_in_pano.T, -1
            )
            mask = (
                ((closest_camera == cam_idx) * 255)
                .astype(np.uint8)
                .reshape(self._camera.width, self._camera.height)
                .transpose()
            )
            
            # Log mask statistics for debugging
            mask_coverage = np.sum(mask > 0) / mask.size
            if cam_idx == 0:  # Only log for first camera to avoid spam
                logging.info(f"Mask coverage for {pano_name}: {mask_coverage:.3f}")

            image_name = (
                self.rig_config.cameras[cam_idx].image_prefix + pano_name
            )
            mask_name = f"{image_name}.png"

            image_path = self.output_image_dir / image_name
            image_path.parent.mkdir(exist_ok=True, parents=True)
            PIL.Image.fromarray(image).save(image_path, exif=gpsonly_exif)

            # Create mask with proper naming for perspective images
            mask_path = self.mask_dir / mask_name
            mask_path.parent.mkdir(exist_ok=True, parents=True)
            if not pycolmap.Bitmap.from_array(mask).write(str(mask_path)):
                raise RuntimeError(f"Cannot write {mask_path}")
            
            # Also create mask in flat structure for COLMAP compatibility
            # Use the same name as the perspective image
            flat_mask_name = image_name.replace('/', '_') + ".png"
            flat_mask_path = self.mask_dir / flat_mask_name
            if not pycolmap.Bitmap.from_array(mask).write(str(flat_mask_path)):
                raise RuntimeError(f"Cannot write {flat_mask_path}")


def render_perspective_images(
    pano_image_names: Sequence[str],
    pano_image_dir: Path,
    output_image_dir: Path,
    mask_dir: Path,
    render_options: PanoRenderOptions,
) -> pycolmap.RigConfig:
    processor = PanoProcessor(
        pano_image_dir, output_image_dir, mask_dir, render_options
    )

    num_panos = len(pano_image_names)
    max_workers = min(32, (os.cpu_count() or 2) - 1)

    with tqdm(total=num_panos) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as thread_pool:
            futures = [
                thread_pool.submit(processor.process, pano_name)
                for pano_name in pano_image_names
            ]
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    return processor.rig_config

def run(args):
    # Define the paths.
    image_dir = args.output_path / "images"
    mask_dir = args.output_path / "masks"
    image_dir.mkdir(exist_ok=True, parents=True)
    mask_dir.mkdir(exist_ok=True, parents=True)
    
    # Object removal setup
    if args.remove_object:
        filter_output_dir = args.output_path / "filtered_images"
        filter_output_dir.mkdir(exist_ok=True, parents=True)

    database_path = args.output_path / "database.db"
    if database_path.exists():
        database_path.unlink()

    rec_path = args.output_path / "sparse"
    rec_path.mkdir(exist_ok=True, parents=True)

    # Search for input images.
    pano_image_dir = args.input_image_path
    pano_image_names = sorted(
        p.relative_to(pano_image_dir).as_posix()
        for p in pano_image_dir.rglob("*")
        if not p.is_dir()
    )
    logging.info(f"Found {len(pano_image_names)} images in {pano_image_dir}.")


    
    # Determine render options based on cube face removal
    render_type = "non-overlapping" if hasattr(args, 'remove_faces') and args.remove_faces else "overlapping"
    
    rig_config = render_perspective_images(
        pano_image_names,
        pano_image_dir,
        image_dir,
        mask_dir,
        PANO_RENDER_OPTIONS[render_type],
    )
    
    # Object removal for panorama workflows
    if args.remove_object:
        logging.info(f"Performing object removal with action: {args.object_action}, model: {args.model}")
        mask_human_output_dir = args.output_path / "masked_human_images"
        mask_human_output_dir.mkdir(exist_ok=True, parents=True)
        filter_output_dir = args.output_path / "filtered_images"
        filter_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Process each pano camera subfolder separately
        for subfolder in image_dir.iterdir():
            if subfolder.is_dir() and subfolder.name.startswith('pano_camera'):
                logging.info(f"Processing subfolder: {subfolder.name}")
                
                # Create corresponding output directories
                subfolder_filter_output = filter_output_dir / subfolder.name
                subfolder_filter_output.mkdir(exist_ok=True, parents=True)
                
                subfolder_masked_output = mask_human_output_dir / subfolder.name
                subfolder_masked_output.mkdir(exist_ok=True, parents=True)
                
                # Generate object masks using remove_background.py
                mask_command = [
                    sys.executable, "-m", "segmentation.remove_background",
                    "-i", str(subfolder),
                    "-o", str(subfolder_filter_output),
                    "-nt", str(args.num_threads),
                    "-ng", str(args.num_gpus),
                    "-m", args.model
                ]
                logging.info(f"Running mask generation: {' '.join(mask_command)}")
                try:
                    subprocess.run(mask_command, check=True)
                    logging.info(f"Successfully generated masks for {subfolder.name}")
                except subprocess.CalledProcessError as e:
                    logging.error(f"Failed to generate masks: {e}")
                    continue
                
                # Apply object removal based on action
                if args.object_action == "remove":
                    # Remove objects using masks
                    remove_command = [
                        sys.executable, "-m", "segmentation.remove_object_using_mask",
                        "-oi", str(subfolder),
                        "-om", str(subfolder_filter_output),
                        "-od", str(subfolder_masked_output)
                    ]
                    logging.info(f"Running object removal: {' '.join(remove_command)}")
                    try:
                        subprocess.run(remove_command, check=True)
                        logging.info(f"Successfully removed objects from {subfolder.name}")
                    except subprocess.CalledProcessError as e:
                        logging.error(f"Failed to remove objects: {e}")
                        continue
                else:  # erase action
                    # Erase objects using masks
                    erase_command = [
                        sys.executable, "-m", "segmentation.erase_object_using_mask",
                        "-id", str(subfolder),
                        "-md", str(subfolder_filter_output),
                        "-mp", str(args.output_path / "stable-diffusion-xl-base-1.0"),
                        "-pp", str(Path(os.path.dirname(os.path.realpath(__file__))).parent.parent / 
                                   "AttentiveEraser" / "pipelines" / "pipeline_stable_diffusion_xl_attentive_eraser.py"),
                        "-gpu", args.use_gpu,
                        "-log", "info",
                        "-method", "SIP"
                    ]
                    logging.info(f"Running object erasing: {' '.join(erase_command)}")
                    try:
                        subprocess.run(erase_command, check=True)
                        logging.info(f"Successfully erased objects from {subfolder.name}")
                    except subprocess.CalledProcessError as e:
                        logging.error(f"Failed to erase objects: {e}")
                        continue
        
        # Use the processed images for further steps
        if args.object_action == "remove":
            # Create a backup of the original images
            original_images_backup = args.output_path / "original_images"
            if original_images_backup.exists():
                shutil.rmtree(original_images_backup)
            shutil.copytree(image_dir, original_images_backup)
            
            # Replace original images with masked versions
            for subfolder in mask_human_output_dir.iterdir():
                if subfolder.is_dir() and subfolder.name.startswith('pano_camera'):
                    target_dir = image_dir / subfolder.name
                    # Clear target directory and copy processed images
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    shutil.copytree(subfolder, target_dir)
            
            logging.info("Replaced original images with processed images (object removal)")
        
        # Count processed images in subdirectories
        processed_count = 0
        for subfolder in image_dir.iterdir():
            if subfolder.is_dir() and subfolder.name.startswith('pano_camera'):
                for img_file in subfolder.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        processed_count += 1
        logging.info(f"Found {processed_count} perspective images in {image_dir} subdirectories after processing")

    # Feature extraction - after object removal is complete
    pycolmap.set_random_seed(0)
    
    # Count perspective images for verification
    perspective_count = 0
    for subfolder in image_dir.iterdir():
        if subfolder.is_dir() and subfolder.name.startswith('pano_camera'):
            for img_file in subfolder.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    perspective_count += 1
    logging.info(f"Extracting features from {perspective_count} perspective images")
    
    pycolmap.extract_features(
        str(database_path),
        str(image_dir),
        reader_options={"mask_path": str(mask_dir)},
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
    )

    with pycolmap.Database(str(database_path)) as db:
        pycolmap.apply_rig_config([rig_config], db)

    if args.matcher == "sequential":
        pycolmap.match_sequential(
            str(database_path),
            matching_options=pycolmap.SequentialMatchingOptions(
                loop_detection=True
            ),
        )
    elif args.matcher == "exhaustive":
        pycolmap.match_exhaustive(str(database_path))
    elif args.matcher == "vocabtree":
        pycolmap.match_vocabtree(str(database_path))
    elif args.matcher == "spatial":
        pycolmap.match_spatial(str(database_path))
    else:
        logging.fatal(f"Unknown matcher: {args.matcher}")

    # Validate database before incremental mapping
    try:
        with pycolmap.Database(str(database_path)) as db:
            num_images = len(db.read_all_images())
            logging.info(f"Database validation: {num_images} images loaded")
            
            if num_images == 0:
                logging.error("No images found in database")
                return
    except Exception as e:
        logging.error(f"Database validation failed: {e}")
        return
    
    # Incremental mapping after all preprocessing is complete
    opts = pycolmap.IncrementalPipelineOptions(
        ba_refine_sensor_from_rig=False,
        ba_refine_focal_length=True,
        ba_refine_principal_point=False,
        ba_refine_extra_params=False,
        #min_num_matches=15,  # Reduce minimum matches for sparse data
        #abs_pose_min_num_inliers=12,  # Reduce inlier requirements
        #abs_pose_min_inlier_ratio=0.15,  # Lower inlier ratio
    )
    
    try:
        # Validate database has sufficient matches before mapping
        with pycolmap.Database(str(database_path)) as db:
            matches = db.read_all_matches()
            if isinstance(matches, dict):
                total_matches = sum(len(match_data) for match_data in matches.values())
            else:
                total_matches = len(matches) if matches else 0
            logging.info(f"Total feature matches in database: {total_matches}")
            
            if total_matches < 100:
                logging.warning("Very few matches found, mapping may fail")
        
        recs = pycolmap.incremental_mapping(
            str(database_path), str(image_dir), str(rec_path), opts
        )
        
        if not recs:
            logging.error("No reconstructions generated")
            return
            
        for idx, rec in recs.items():
            logging.info(f"#{idx} {rec.summary()}")
            
    except Exception as e:
        logging.error(f"Incremental mapping failed: {e}")
        # Don't re-raise to prevent container crash
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--matcher",
        default="sequential",
        choices=["sequential", "exhaustive", "vocabtree", "spatial"],
    )
    parser.add_argument(
        "--remove_object",
        action="store_true",
        help="Enable object removal (human detection and removal)"
    )
    parser.add_argument(
        "--object_action",
        default="erase",
        choices=["erase", "remove"],
        help="Action to perform on detected objects"
    )
    parser.add_argument(
        "-m", "--model",
        default="u2net_human_seg",
        help="Model to use for object detection"
    )
    parser.add_argument(
        "-nt", "--num_threads",
        type=int,
        default=32,
        help="Number of threads to use"
    )
    parser.add_argument(
        "-ng", "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "-gpu", "--use_gpu",
        default="true",
        help="Whether to use GPU"
    )
    parser.add_argument(
        "--remove_faces",
        action="store_true",
        help="Use non-overlapping render mode to exclude top/bottom faces"
    )

    run(parser.parse_args())
