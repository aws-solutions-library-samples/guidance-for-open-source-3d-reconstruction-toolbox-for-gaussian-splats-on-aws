# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Based on COLMAP 4.0.2 panorama_sfm.py example with extensions for object removal
# and cube face exclusion.

"""
Incremental SfM on 360 spherical panorama images with optional object removal.
"""

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal, TypeVar, cast

import cv2
import numpy as np
import numpy.typing as npt
import PIL.ExifTags
import PIL.Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import pycolmap
from pycolmap import logging

N = TypeVar("N", bound=int)
NDArrayNx2 = np.ndarray[tuple[N, Literal[2]], np.dtype[np.float64]]
NDArray3x1 = np.ndarray[tuple[Literal[3], Literal[1]], np.dtype[np.float64]]

VOCAB_TREE_PATH = "/opt/ml/code/vocab_tree_flickr100K_words32K.bin"


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


def check_for_valid_rigid_objects(
    mask_dir: Path, min_area_percentage: float = 0.01, min_blob_size: int = 1000
) -> bool:
    """Return True if any mask contains a significant rigid object."""
    mask_files = [f for f in mask_dir.iterdir() if f.suffix.lower() == ".png"]
    if not mask_files:
        return False
    for mask_file in mask_files:
        try:
            mask_img = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
            if mask_img is None:
                continue
            if mask_img.ndim == 3 and mask_img.shape[2] == 4:
                mask_values = mask_img[:, :, 3] / 255.0
            elif mask_img.ndim == 3:
                mask_values = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY) / 255.0
            else:
                mask_values = mask_img / 255.0
            if np.max(mask_values) < 0.001:
                continue
            binary_mask = (mask_values > 0.05).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            num_labels, labels = cv2.connectedComponents(binary_mask)
            if num_labels <= 1:
                continue
            total_pixels = mask_values.shape[0] * mask_values.shape[1]
            min_pixels = int(total_pixels * min_area_percentage)
            for i in range(1, num_labels):
                if np.sum(labels == i) >= min_blob_size and np.sum(labels == i) >= min_pixels:
                    logging.info(f"Found valid rigid object in {mask_file.name}")
                    return True
        except Exception as e:
            logging.warning(f"Error checking mask {mask_file}: {e}")
    logging.info(f"No valid rigid objects found in {len(mask_files)} masks")
    return False


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
    return pycolmap.Camera.create_from_model_id(
        camera_id=0,
        model=pycolmap.CameraModelId.SIMPLE_PINHOLE,
        focal_length=focal,
        width=image_width,
        height=image_height,
    )


def get_virtual_camera_rays(
    camera: pycolmap.Camera,
) -> npt.NDArray[np.floating]:
    size = (camera.width, camera.height)
    x, y = np.indices(size).astype(np.float32)
    xy: NDArrayNx2 = np.column_stack([x.ravel(), y.ravel()])
    xy += 0.5
    xy_norm: NDArrayNx2 = camera.cam_from_img(image_points=xy)
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def spherical_img_from_cam(
    image_size: tuple[int, int], rays_in_cam: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
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
) -> Sequence[npt.NDArray[np.floating]]:
    """Get the relative rotations of the virtual cameras w.r.t. the panorama."""
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
    cams_from_pano_rotation: Sequence[npt.NDArray[np.floating]],
    ref_idx: int = 0,
) -> pycolmap.RigConfig:
    """Create a RigConfig for the given virtual rotations."""
    rig_cameras = []
    zero_translation = cast(NDArray3x1, np.zeros((3, 1), dtype=np.float64))
    for idx, cam_from_pano_rotation in enumerate(cams_from_pano_rotation):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_rotation = (
                cam_from_pano_rotation @ cams_from_pano_rotation[ref_idx].T
            )
            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_rotation),
                zero_translation,
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

        self.cam_centers_in_pano = np.einsum(
            "nij,i->nj", self.cams_from_pano_rotation, [0, 0, 1]
        )

        self._lock = Lock()
        self._camera: pycolmap.Camera | None = None
        self._pano_size: tuple[int, int] | None = None
        self._rays_in_cam: npt.NDArray[np.floating] | None = None

    def process(self, pano_name: str) -> None:
        pano_path = self.pano_image_dir / pano_name
        try:
            pano_pil_image = PIL.Image.open(pano_path)
        except PIL.Image.UnidentifiedImageError:
            logging.info(f"Skipping file {pano_path} as it cannot be read.")
            return

        pano_exif = pano_pil_image.getexif()
        pano_image = np.asarray(pano_pil_image)
        gpsonly_exif = PIL.Image.Exif()
        gpsonly_exif[PIL.ExifTags.IFD.GPSInfo] = pano_exif.get_ifd(
            PIL.ExifTags.IFD.GPSInfo
        )

        pano_height, pano_width, *_ = pano_image.shape
        if pano_width != pano_height * 2:
            raise ValueError("Only 360° panoramas are supported.")

        with self._lock:
            if self._camera is None:
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
            else:
                if (pano_width, pano_height) != self._pano_size:
                    raise ValueError("Panoramas of different sizes are not supported.")

        for cam_idx, cam_from_pano_r in enumerate(self.cams_from_pano_rotation):
            assert self._rays_in_cam is not None
            rays_in_pano = self._rays_in_cam @ cam_from_pano_r
            xy_in_pano = spherical_img_from_cam(self._pano_size, rays_in_pano)
            xy_in_pano = xy_in_pano.reshape(
                self._camera.width, self._camera.height, 2
            ).astype(np.float32)
            xy_in_pano -= 0.5
            x_coords, y_coords = np.moveaxis(xy_in_pano, [0, 1, 2], [2, 1, 0])
            image = cv2.remap(
                pano_image,
                x_coords,
                y_coords,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
            closest_camera = np.argmax(
                rays_in_pano @ self.cam_centers_in_pano.T, -1
            )
            mask = (
                ((closest_camera == cam_idx) * 255)
                .astype(np.uint8)
                .reshape(self._camera.width, self._camera.height)
                .transpose()
            )

            image_name = self.rig_config.cameras[cam_idx].image_prefix + pano_name
            mask_name = f"{image_name}.png"

            image_path = self.output_image_dir / image_name
            image_path.parent.mkdir(exist_ok=True, parents=True)
            PIL.Image.fromarray(image).save(image_path, exif=gpsonly_exif)

            mask_path = self.mask_dir / mask_name
            mask_path.parent.mkdir(exist_ok=True, parents=True)
            if not pycolmap.Bitmap.from_array(mask).write(mask_path):
                raise RuntimeError(f"Cannot write {mask_path}")


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


def _apply_object_removal(args, image_dir: Path) -> None:
    """Run object removal on each pano_camera subfolder after perspective rendering."""
    logging.info(f"Performing object removal: action={args.object_action}, model={args.model}")
    mask_human_output_dir = args.output_path / "masked_human_images"
    filter_output_dir = args.output_path / "filtered_images"
    mask_human_output_dir.mkdir(exist_ok=True, parents=True)
    filter_output_dir.mkdir(exist_ok=True, parents=True)

    for subfolder in sorted(image_dir.iterdir()):
        if not (subfolder.is_dir() and subfolder.name.startswith("pano_camera")):
            continue
        logging.info(f"Processing {subfolder.name}")
        subfolder_filter = filter_output_dir / subfolder.name
        subfolder_masked = mask_human_output_dir / subfolder.name
        subfolder_filter.mkdir(exist_ok=True, parents=True)
        subfolder_masked.mkdir(exist_ok=True, parents=True)

        # Generate segmentation masks
        try:
            subprocess.run(
                [sys.executable, "-m", "pre_processing.segmentation.remove_background",
                 "-i", str(subfolder), "-o", str(subfolder_filter),
                 "-nt", str(args.num_threads), "-ng", str(args.num_gpus), "-m", args.model],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Mask generation failed for {subfolder.name}: {e}")
            continue

        if not check_for_valid_rigid_objects(subfolder_filter):
            logging.info(f"No significant objects in {subfolder.name}, skipping")
            continue

        if args.object_action == "remove":
            try:
                subprocess.run(
                    [sys.executable, "-m", "segmentation.remove_object_using_mask",
                     "-oi", str(subfolder), "-om", str(subfolder_filter),
                     "-od", str(subfolder_masked)],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                logging.error(f"Object removal failed for {subfolder.name}: {e}")
        else:  # erase
            pp_path = (
                Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
                / "AttentiveEraser" / "pipelines"
                / "pipeline_stable_diffusion_xl_attentive_eraser.py"
            )
            try:
                subprocess.run(
                    [sys.executable, "-m", "pre_processing.segmentation.erase_object_using_mask",
                     "-id", str(subfolder), "-md", str(subfolder_filter),
                     "-od", str(subfolder_masked),
                     "-mp", str(args.output_path / "stable-diffusion-xl-base-1.0"),
                     "-pp", str(pp_path),
                     "-gpu", args.use_gpu, "-log", "info", "-method", "SIP"],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                logging.error(f"Object erase failed for {subfolder.name}: {e}")

    # Replace perspective images with processed versions
    any_processed = any(
        any(f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')
            for f in (mask_human_output_dir / sf.name).iterdir())
        for sf in image_dir.iterdir()
        if sf.is_dir() and sf.name.startswith("pano_camera")
        and (mask_human_output_dir / sf.name).exists()
    )
    if not any_processed:
        logging.info("No objects processed, keeping original perspective images")
        return

    original_backup = args.output_path / "original_images"
    if original_backup.exists():
        shutil.rmtree(original_backup)
    shutil.copytree(image_dir, original_backup)

    if args.object_action == "remove":
        for subfolder in mask_human_output_dir.iterdir():
            if subfolder.is_dir() and subfolder.name.startswith("pano_camera"):
                target = image_dir / subfolder.name
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(subfolder, target)
    else:  # erase — copy inpainted images then resize all to match
        for subfolder in mask_human_output_dir.iterdir():
            if not (subfolder.is_dir() and subfolder.name.startswith("pano_camera")):
                continue
            target_dir = image_dir / subfolder.name
            # Inpainted images are written to the eraser output root (not refined_masks)
            for img_file in subfolder.iterdir():
                if not (img_file.is_file() and img_file.suffix.lower() in (".png", ".jpg", ".jpeg")):
                    continue
                img = cv2.imread(str(img_file))
                if img is not None and np.max(img) > 10:
                    # Match to original filename in target_dir by stem
                    target_file = target_dir / img_file.name
                    if not target_file.exists():
                        # Try matching by stem with different extension
                        for ext in (".jpg", ".jpeg", ".png"):
                            candidate = target_dir / (img_file.stem + ext)
                            if candidate.exists():
                                target_file = candidate
                                break
                    cv2.imwrite(str(target_file), img)
            logging.info(f"Copied non-black erased images into {subfolder.name}")

        # Resize ALL perspective images to 960x960 to match eraser output dimensions
        for subfolder in image_dir.iterdir():
            if not (subfolder.is_dir() and subfolder.name.startswith("pano_camera")):
                continue
            for img_file in subfolder.iterdir():
                if not (img_file.is_file() and img_file.suffix.lower() in (".png", ".jpg", ".jpeg")):
                    continue
                img = cv2.imread(str(img_file))
                if img is not None and (img.shape[0] != 960 or img.shape[1] != 960):
                    cv2.imwrite(str(img_file),
                                cv2.resize(img, (960, 960), interpolation=cv2.INTER_LANCZOS4))
            logging.info(f"Resized all images to 960x960 in {subfolder.name}")

    logging.info("Perspective images replaced with object-removed versions")


def run(args: argparse.Namespace) -> None:
    pycolmap.set_random_seed(0)

    image_dir = args.output_path / "images"
    mask_dir = args.output_path / "masks"
    image_dir.mkdir(exist_ok=True, parents=True)
    mask_dir.mkdir(exist_ok=True, parents=True)

    database_path = args.output_path / "database.db"
    if database_path.exists():
        database_path.unlink()

    rec_path = args.output_path / "sparse"
    rec_path.mkdir(exist_ok=True, parents=True)

    pano_image_dir = args.input_image_path
    # If ERP images are in the same dir as perspective output, back them up first
    if pano_image_dir == image_dir:
        erp_backup_dir = args.output_path / "erp_images_original"
        if erp_backup_dir.exists():
            shutil.rmtree(erp_backup_dir)
        shutil.move(str(pano_image_dir), str(erp_backup_dir))
        pano_image_dir = erp_backup_dir
        image_dir.mkdir(exist_ok=True, parents=True)

    pano_image_names = sorted(
        p.relative_to(pano_image_dir).as_posix()
        for p in pano_image_dir.rglob("*")
        if not p.is_dir()
    )
    logging.info(f"Found {len(pano_image_names)} ERP images in {pano_image_dir}.")

    render_type = "non-overlapping" if args.remove_faces else "overlapping"
    rig_config = render_perspective_images(
        pano_image_names,
        pano_image_dir,
        image_dir,
        mask_dir,
        PANO_RENDER_OPTIONS[render_type],
    )

    if args.remove_object:
        _apply_object_removal(args, image_dir)

    pycolmap.extract_features(
        database_path,
        image_dir,
        reader_options=pycolmap.ImageReaderOptions(mask_path=mask_dir),
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
    )

    with pycolmap.Database.open(database_path) as db:
        pycolmap.apply_rig_config([rig_config], db)

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.rig_verification = True
    matching_options.skip_image_pairs_in_same_frame = True

    if args.matcher == "sequential":
        loop_detection = os.path.exists(VOCAB_TREE_PATH)
        if not loop_detection:
            logging.warning(f"Vocab tree not found at {VOCAB_TREE_PATH}, disabling loop detection")
        pairing_opts = pycolmap.SequentialPairingOptions(loop_detection=loop_detection)
        if loop_detection:
            pairing_opts.vocab_tree_path = VOCAB_TREE_PATH
        pycolmap.match_sequential(
            database_path,
            pairing_options=pairing_opts,
            matching_options=matching_options,
        )
    elif args.matcher == "exhaustive":
        pycolmap.match_exhaustive(database_path, matching_options=matching_options)
    elif args.matcher == "vocabtree":
        pycolmap.match_vocabtree(database_path, matching_options=matching_options)
    elif args.matcher == "spatial":
        pycolmap.match_spatial(database_path, matching_options=matching_options)
    else:
        logging.fatal(f"Unknown matcher: {args.matcher}")

    opts = pycolmap.IncrementalPipelineOptions(
        ba_refine_sensor_from_rig=False,
        ba_refine_focal_length=False,
        ba_refine_principal_point=False,
        ba_refine_extra_params=False,
    )
    try:
        recs = pycolmap.incremental_mapping(database_path, image_dir, rec_path, opts)
        if not recs:
            logging.error("No reconstructions generated")
            return
        for idx, rec in recs.items():
            logging.info(f"#{idx} {rec.summary()}")
    except Exception as e:
        logging.error(f"Incremental mapping failed: {e}")


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
        "--remove_faces",
        action="store_true",
        help="Use non-overlapping render mode to exclude top/bottom cube faces",
    )
    parser.add_argument(
        "--remove_object",
        action="store_true",
        help="Enable object removal on perspective images after rendering",
    )
    parser.add_argument(
        "--object_action", default="erase", choices=["erase", "remove"],
    )
    parser.add_argument("-m", "--model", default="u2net_human_seg")
    parser.add_argument("-nt", "--num_threads", type=int, default=32)
    parser.add_argument("-ng", "--num_gpus", type=int, default=1)
    parser.add_argument("-gpu", "--use_gpu", default="true")
    run(parser.parse_args())
