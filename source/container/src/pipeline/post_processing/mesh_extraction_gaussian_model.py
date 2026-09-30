# MIT License — see __init__.py
"""Gaussian model: load from PLY, manage differentiable parameters."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData
from torch import Tensor

from mesh_extraction_utils import (
    build_covariance,
    build_covariance_inverse,
    quaternion_to_rotation_matrix,
)


class GaussianModel(nn.Module):
    """Differentiable 3D Gaussian representation.

    Stores Gaussian parameters as nn.Parameters for optimization.
    Loads from standard 3DGS PLY format (positions, SH, opacity,
    log-scales, quaternions).
    """

    def __init__(
        self,
        positions: Tensor,
        quaternions: Tensor,
        log_scales: Tensor,
        raw_opacities: Tensor,
        sh_dc: Tensor,
        sh_rest: Tensor,
    ) -> None:
        super().__init__()
        self._positions = nn.Parameter(positions)
        self._quaternions = nn.Parameter(quaternions)
        self._log_scales = nn.Parameter(log_scales)
        self._raw_opacities = nn.Parameter(raw_opacities)
        self._sh_dc = nn.Parameter(sh_dc)
        self._sh_rest = nn.Parameter(sh_rest)

    @classmethod
    def from_ply(cls, path: str, device: str = "cuda") -> GaussianModel:
        """Load a GaussianModel from a standard 3DGS PLY file."""
        plydata = PlyData.read(path)
        v = plydata["vertex"]

        positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)

        sh_dc = np.stack(
            [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1
        ).reshape(-1, 1, 3)

        rest_names = sorted(
            [p.name for p in v.properties if p.name.startswith("f_rest_")],
            key=lambda n: int(n.split("_")[-1]),
        )
        if rest_names:
            sh_rest_flat = np.stack([v[n] for n in rest_names], axis=-1)
            # PLY stores SH rest in channel-first order (3, n_coeffs);
            # transpose to (n_coeffs, 3) expected by the model.
            sh_rest = sh_rest_flat.reshape(len(positions), 3, -1).transpose(0, 2, 1)
        else:
            sh_rest = np.zeros((len(positions), 0, 3), dtype=np.float32)

        raw_opacities = v["opacity"].copy()

        log_scales = np.stack(
            [v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1
        )

        quaternions = np.stack(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1
        )

        def _t(arr: np.ndarray) -> Tensor:
            return torch.tensor(arr, dtype=torch.float32, device=device)

        return cls(
            positions=_t(positions),
            quaternions=_t(quaternions),
            log_scales=_t(log_scales),
            raw_opacities=_t(raw_opacities),
            sh_dc=_t(sh_dc),
            sh_rest=_t(sh_rest),
        )

    @classmethod
    def from_gsplat_checkpoint(
        cls, path: str, device: str = "cuda"
    ) -> GaussianModel:
        """Load from a gsplat .pt checkpoint (keys: means, quats, scales,
        opacities, sh0, shN)."""
        ckpt = torch.load(path, map_location=device, weights_only=True)
        splats = ckpt if "splats" not in ckpt else ckpt["splats"]

        positions = splats["means"].float()
        quaternions = splats["quats"].float()
        log_scales = splats["scales"].float()
        raw_opacities = splats["opacities"].float()
        sh_dc = splats["sh0"].float()
        sh_rest = splats.get(
            "shN", torch.zeros(len(positions), 0, 3, device=device)
        ).float()

        return cls(
            positions=positions,
            quaternions=quaternions,
            log_scales=log_scales,
            raw_opacities=raw_opacities,
            sh_dc=sh_dc,
            sh_rest=sh_rest,
        )

    # --- Derived properties ---

    @property
    def n_gaussians(self) -> int:
        return self._positions.shape[0]

    @property
    def positions(self) -> Tensor:
        return self._positions

    @property
    def scales(self) -> Tensor:
        """Exponentiated scales: [N, 3]."""
        return torch.exp(self._log_scales)

    @property
    def opacities(self) -> Tensor:
        """Sigmoid opacities: [N] in (0, 1)."""
        return torch.sigmoid(self._raw_opacities)

    @property
    def rotation_matrices(self) -> Tensor:
        """[N, 3, 3] rotation matrices from quaternions."""
        return quaternion_to_rotation_matrix(self._quaternions)

    @property
    def covariances(self) -> Tensor:
        """[N, 3, 3] covariance matrices."""
        return build_covariance(self.scales, self.rotation_matrices)

    @property
    def covariances_inverse(self) -> Tensor:
        """[N, 3, 3] inverse covariance matrices."""
        return build_covariance_inverse(self.scales, self.rotation_matrices)

    @property
    def normals(self) -> Tensor:
        """[N, 3] surface normals — the axis with the smallest scale."""
        R = self.rotation_matrices  # [N, 3, 3]
        s = self.scales  # [N, 3]
        min_idx = s.argmin(dim=-1)  # [N]
        normals = R[torch.arange(len(R), device=R.device), :, min_idx]
        return normals

    @property
    def min_scales(self) -> Tensor:
        """[N] the smallest scale per Gaussian."""
        return self.scales.min(dim=-1).values

    def sh_coefficients(self) -> Tensor:
        """Full SH coefficients [N, K, 3] (DC + rest)."""
        return torch.cat([self._sh_dc, self._sh_rest], dim=1)

    def spatial_extent(self) -> float:
        """Approximate scene radius (max distance from centroid)."""
        with torch.no_grad():
            centroid = self._positions.mean(dim=0)
            dists = (self._positions - centroid).norm(dim=-1)
            return dists.quantile(0.95).item()

    # --- Pruning ---

    @torch.no_grad()
    def prune(self, mask: Tensor) -> None:
        """Remove Gaussians where mask is True."""
        keep = ~mask
        self._positions = nn.Parameter(self._positions[keep])
        self._quaternions = nn.Parameter(self._quaternions[keep])
        self._log_scales = nn.Parameter(self._log_scales[keep])
        self._raw_opacities = nn.Parameter(self._raw_opacities[keep])
        self._sh_dc = nn.Parameter(self._sh_dc[keep])
        self._sh_rest = nn.Parameter(self._sh_rest[keep])

    # --- Export ---

    @torch.no_grad()
    def to_ply(self, path: str) -> None:
        """Save as standard 3DGS PLY."""
        from plyfile import PlyElement

        N = self.n_gaussians
        pos = self._positions.detach().cpu().numpy()
        dc = self._sh_dc.detach().cpu().numpy().reshape(N, 3)
        # Transpose back to channel-first order for PLY storage
        rest = self._sh_rest.detach().cpu().numpy().transpose(0, 2, 1).reshape(N, -1)
        opac = self._raw_opacities.detach().cpu().numpy()
        sc = self._log_scales.detach().cpu().numpy()
        rot = self._quaternions.detach().cpu().numpy()

        names = (
            ["x", "y", "z"]
            + ["f_dc_0", "f_dc_1", "f_dc_2"]
            + [f"f_rest_{i}" for i in range(rest.shape[1])]
            + ["opacity"]
            + ["scale_0", "scale_1", "scale_2"]
            + ["rot_0", "rot_1", "rot_2", "rot_3"]
        )
        arrays = [
            pos[:, 0], pos[:, 1], pos[:, 2],
            dc[:, 0], dc[:, 1], dc[:, 2],
            *[rest[:, i] for i in range(rest.shape[1])],
            opac,
            sc[:, 0], sc[:, 1], sc[:, 2],
            rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3],
        ]

        dtype = [(n, "f4") for n in names]
        arr = np.empty(N, dtype=dtype)
        for name, data in zip(names, arrays):
            arr[name] = data

        el = PlyElement.describe(arr, "vertex")
        PlyData([el]).write(path)
