# MIT License — see __init__.py
"""Camera loading from nerfstudio transforms.json format."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor


@dataclass
class Camera:
    """A single camera with intrinsics, extrinsics, and optionally an image."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: np.ndarray  # [4, 4] camera-to-world
    image_path: Optional[str] = None

    @property
    def w2c(self) -> np.ndarray:
        """World-to-camera [4, 4] (OpenGL convention)."""
        return np.linalg.inv(self.c2w)

    @property
    def c2w_opencv(self) -> np.ndarray:
        """Camera-to-world [4, 4] in OpenCV convention (+z forward, +y down).

        Nerfstudio stores c2w in OpenGL convention (-z forward, +y up).
        This flips columns 1 and 2 to get OpenCV.
        """
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        return self.c2w @ flip

    def intrinsic_matrix(self, device: str = "cuda") -> Tensor:
        """[3, 3] intrinsic matrix."""
        K = torch.zeros(3, 3, device=device)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        K[2, 2] = 1.0
        return K

    def viewmat(self, device: str = "cuda") -> Tensor:
        """[4, 4] world-to-camera for gsplat (OpenCV: +z forward, +y down).

        Nerfstudio c2w is OpenGL convention (-z forward, +y up).
        gsplat expects OpenCV convention, so we flip y and z rows.
        """
        w2c = self.w2c.copy()
        w2c[1, :] *= -1  # flip y
        w2c[2, :] *= -1  # flip z
        return torch.tensor(w2c, dtype=torch.float32, device=device)

    def load_image(self, device: str = "cuda") -> Tensor:
        """Load the image as [1, 3, H, W] float tensor in [0, 1]."""
        assert self.image_path is not None
        img = Image.open(self.image_path).convert("RGB")
        img = img.resize((self.width, self.height), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return (
            torch.from_numpy(arr)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )


def load_cameras_from_transforms(
    transforms_path: str,
    image_dir: Optional[str] = None,
    downscale: int = 1,
) -> List[Camera]:
    """Load cameras from nerfstudio transforms.json.

    Args:
        transforms_path: path to transforms.json.
        image_dir: override directory for images. If None, uses
            the directory containing transforms.json.
        downscale: downscale factor for resolution.

    Returns:
        List of Camera objects.
    """
    with open(transforms_path) as f:
        data = json.load(f)

    base_dir = Path(transforms_path).parent
    if image_dir is None:
        image_dir_path = base_dir
    else:
        image_dir_path = Path(image_dir)

    # Global intrinsics (may be overridden per-frame)
    global_fx = data.get("fl_x", data.get("focal_length", 0))
    global_fy = data.get("fl_y", global_fx)
    global_w = data.get("w", 0)
    global_h = data.get("h", 0)
    global_cx = data.get("cx", global_w / 2.0)
    global_cy = data.get("cy", global_h / 2.0)

    # Scene normalization transform applied during training.
    # Gaussians live in this transformed space, so cameras must too.
    applied_transform = data.get("applied_transform")
    if applied_transform is not None:
        at = np.array(applied_transform, dtype=np.float64)
        if at.shape == (3, 4):
            at = np.vstack([at, [0, 0, 0, 1]])
    else:
        at = None

    cameras = []
    for frame in data.get("frames", []):
        fx = frame.get("fl_x", global_fx) / downscale
        fy = frame.get("fl_y", global_fy) / downscale
        w = int(frame.get("w", global_w)) // downscale
        h = int(frame.get("h", global_h)) // downscale
        cx = frame.get("cx", global_cx) / downscale
        cy = frame.get("cy", global_cy) / downscale

        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        if c2w.shape[0] == 3:
            c2w = np.vstack([c2w, [0, 0, 0, 1]])

        if at is not None:
            c2w = at @ c2w

        file_path = frame.get("file_path", "")
        img_path = str(image_dir_path / file_path)

        if not os.path.isfile(img_path):
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
                candidate = img_path + ext
                if os.path.isfile(candidate):
                    img_path = candidate
                    break

        cameras.append(
            Camera(
                width=w,
                height=h,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                c2w=c2w,
                image_path=img_path if os.path.isfile(img_path) else None,
            )
        )

    return cameras
