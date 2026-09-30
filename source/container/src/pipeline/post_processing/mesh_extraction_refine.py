# MIT License — see __init__.py
"""Mesh refinement by binding Gaussians to mesh triangles.

Based on Section 3.5 of Guédon & Lepetit, CVPR 2024:
Gaussians are placed at fixed barycentric coordinates on mesh
triangles with 2D tangent-plane rotations. The mesh vertices and
Gaussian parameters are jointly optimized via rendering loss +
mesh normal consistency.
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from torch import Tensor

from mesh_extraction_cameras import Camera
from mesh_extraction_config import MeshExtractionConfig
from mesh_extraction_losses import photometric_loss
from mesh_extraction_utils import quaternion_to_rotation_matrix


# Fixed barycentric coordinates for placing Gaussians on triangles
BARY_CONFIGS = {
    1: torch.tensor([[1 / 3, 1 / 3, 1 / 3]]),
    3: torch.tensor([
        [1 / 2, 1 / 4, 1 / 4],
        [1 / 4, 1 / 2, 1 / 4],
        [1 / 4, 1 / 4, 1 / 2],
    ]),
    4: torch.tensor([
        [1 / 3, 1 / 3, 1 / 3],
        [2 / 3, 1 / 6, 1 / 6],
        [1 / 6, 2 / 3, 1 / 6],
        [1 / 6, 1 / 6, 2 / 3],
    ]),
    6: torch.tensor([
        [2 / 3, 1 / 6, 1 / 6],
        [1 / 6, 2 / 3, 1 / 6],
        [1 / 6, 1 / 6, 2 / 3],
        [1 / 6, 5 / 12, 5 / 12],
        [5 / 12, 1 / 6, 5 / 12],
        [5 / 12, 5 / 12, 1 / 6],
    ]),
}


class BoundGaussianModel(nn.Module):
    """Gaussians bound to mesh triangle surfaces.

    Each triangle has a fixed number of Gaussians at predetermined
    barycentric coordinates. Each Gaussian has:
    - 2D in-plane scale (tangent to triangle)
    - 2D complex rotation in the tangent plane
    - opacity
    - SH color coefficients

    The Gaussian 3D positions are derived from the mesh vertices
    and barycentric coordinates.
    """

    def __init__(
        self,
        vertices: Tensor,
        faces: Tensor,
        n_gaussians_per_triangle: int = 6,
        sh_degree: int = 0,
        surface_thickness: float = 1e-6,
    ) -> None:
        super().__init__()
        device = vertices.device
        F_count = faces.shape[0]
        G = n_gaussians_per_triangle

        self._vertices = nn.Parameter(vertices.float())
        self.register_buffer("_faces", faces.long())
        self.n_per_tri = G
        self.surface_thickness = surface_thickness

        bary = BARY_CONFIGS[G].to(device)
        self.register_buffer(
            "_bary", bary.unsqueeze(0).expand(F_count, -1, -1)
        )  # [F, G, 3]

        n_total = F_count * G
        self._log_scales_2d = nn.Parameter(
            torch.full((n_total, 2), -5.0, device=device)
        )
        # 2D complex rotation: (cos, sin) initialized to identity
        self._tangent_rotation = nn.Parameter(
            torch.zeros(n_total, 2, device=device)
        )
        self._tangent_rotation.data[:, 0] = 1.0

        self._raw_opacities = nn.Parameter(
            torch.full((n_total,), 2.0, device=device)
        )

        n_sh = (sh_degree + 1) ** 2
        self._sh_dc = nn.Parameter(
            torch.zeros(n_total, 1, 3, device=device)
        )
        if n_sh > 1:
            self._sh_rest = nn.Parameter(
                torch.zeros(n_total, n_sh - 1, 3, device=device)
            )
        else:
            self._sh_rest = nn.Parameter(
                torch.zeros(n_total, 0, 3, device=device)
            )

    @classmethod
    def from_mesh(
        cls,
        mesh: trimesh.Trimesh,
        n_per_triangle: int = 6,
        device: str = "cuda",
        surface_thickness: float = 1e-6,
    ) -> BoundGaussianModel:
        """Create from a trimesh mesh."""
        vertices = torch.tensor(
            mesh.vertices, dtype=torch.float32, device=device
        )
        faces = torch.tensor(
            mesh.faces, dtype=torch.long, device=device
        )
        return cls(
            vertices, faces,
            n_gaussians_per_triangle=n_per_triangle,
            surface_thickness=surface_thickness,
        )

    @property
    def n_gaussians(self) -> int:
        return self._faces.shape[0] * self.n_per_tri

    def _face_vertices(self) -> Tensor:
        """[F, 3, 3] vertices of each face."""
        return self._vertices[self._faces]

    def _face_normals(self) -> Tensor:
        """[F, 3] face normals (unit vectors)."""
        fv = self._face_vertices()
        e1 = fv[:, 1] - fv[:, 0]
        e2 = fv[:, 2] - fv[:, 0]
        n = torch.cross(e1, e2, dim=-1)
        return n / n.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def gaussian_centers(self) -> Tensor:
        """[N_total, 3] Gaussian positions from barycentric interpolation."""
        fv = self._face_vertices()  # [F, 3, 3]
        centers = torch.einsum("fgb,fbj->fgj", self._bary, fv)
        return centers.reshape(-1, 3)

    def gaussian_rotations(self) -> Tensor:
        """[N_total, 4] quaternions for each Gaussian.

        Builds a full 3D rotation from the face tangent frame +
        the learned 2D in-plane rotation.
        """
        fv = self._face_vertices()
        normals = self._face_normals()  # [F, 3]

        # Tangent frame: edge0 direction + cross product
        e0 = fv[:, 1] - fv[:, 0]
        t0 = e0 / e0.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        t1 = torch.cross(normals, t0, dim=-1)

        # Build rotation matrix for each face: columns = (t0, t1, normal)
        face_R = torch.stack([t0, t1, normals], dim=-1)  # [F, 3, 3]

        # Apply learned 2D rotation in tangent plane
        rot_2d = F.normalize(self._tangent_rotation, dim=-1)
        cos_a = rot_2d[:, 0].reshape(-1, 1, 1)
        sin_a = rot_2d[:, 1].reshape(-1, 1, 1)

        # 2D rotation matrix in tangent plane
        R2d = torch.zeros(
            self.n_gaussians, 3, 3, device=self._vertices.device
        )
        R2d[:, 0, 0] = cos_a.squeeze()
        R2d[:, 0, 1] = -sin_a.squeeze()
        R2d[:, 1, 0] = sin_a.squeeze()
        R2d[:, 1, 1] = cos_a.squeeze()
        R2d[:, 2, 2] = 1.0

        F_count = self._faces.shape[0]
        face_R_expanded = (
            face_R.unsqueeze(1)
            .expand(-1, self.n_per_tri, -1, -1)
            .reshape(-1, 3, 3)
        )
        full_R = face_R_expanded @ R2d  # [N_total, 3, 3]

        return _rotation_matrix_to_quaternion(full_R)

    def gaussian_scales(self) -> Tensor:
        """[N_total, 3] scales — 2D in-plane + tiny normal thickness."""
        s2d = torch.exp(self._log_scales_2d)
        thickness = torch.full(
            (self.n_gaussians, 1),
            self.surface_thickness,
            device=self._vertices.device,
        )
        return torch.cat([s2d, thickness], dim=-1)

    def mesh_normal_consistency_loss(self) -> Tensor:
        """Penalize angle between normals of adjacent faces.

        Standard mesh normal consistency (fully vectorized).
        """
        normals = self._face_normals()  # [F, 3]
        device = self._vertices.device

        edges = torch.cat([
            self._faces[:, [0, 1]],
            self._faces[:, [1, 2]],
            self._faces[:, [2, 0]],
        ], dim=0)  # [3F, 2]
        face_indices = torch.arange(
            self._faces.shape[0], device=device
        ).repeat(3)

        edges_sorted = edges.sort(dim=-1).values
        edge_keys = (
            edges_sorted[:, 0].long() * self._vertices.shape[0]
            + edges_sorted[:, 1].long()
        )

        sorted_order = edge_keys.argsort()
        edge_keys_s = edge_keys[sorted_order]
        face_indices_s = face_indices[sorted_order]

        # Shared edges: consecutive entries with same key
        same = edge_keys_s[:-1] == edge_keys_s[1:]
        if same.sum() == 0:
            return torch.tensor(0.0, device=device)

        f0 = face_indices_s[:-1][same]
        f1 = face_indices_s[1:][same]
        dot = (normals[f0] * normals[f1]).sum(dim=-1)
        return (1.0 - dot).mean()

    def export_mesh(self) -> trimesh.Trimesh:
        """Export the current mesh state as a trimesh."""
        v = self._vertices.detach().cpu().numpy()
        f = self._faces.detach().cpu().numpy()
        return trimesh.Trimesh(vertices=v, faces=f, process=False)


def _rotation_matrix_to_quaternion(R: Tensor) -> Tensor:
    """Convert rotation matrices to quaternions (w, x, y, z).

    Args:
        R: [..., 3, 3] rotation matrices.

    Returns:
        [..., 4] quaternions.
    """
    batch_shape = R.shape[:-2]
    m = R.reshape(-1, 3, 3)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    q = torch.zeros(m.shape[0], 4, device=R.device, dtype=R.dtype)

    s = torch.sqrt((trace + 1.0).clamp(min=1e-8)) * 2
    q[:, 0] = 0.25 * s
    q[:, 1] = (m[:, 2, 1] - m[:, 1, 2]) / s
    q[:, 2] = (m[:, 0, 2] - m[:, 2, 0]) / s
    q[:, 3] = (m[:, 1, 0] - m[:, 0, 1]) / s

    return F.normalize(q.reshape(*batch_shape, 4), dim=-1)


def _render_bound_gsplat(
    model: BoundGaussianModel,
    camera: Camera,
    device: str = "cuda",
    sh_degree: int = 0,
) -> tuple[Tensor, Tensor]:
    """Render from bound Gaussian model."""
    from gsplat import rasterization

    viewmat = camera.viewmat(device).unsqueeze(0)
    K = camera.intrinsic_matrix(device).unsqueeze(0)

    centers = model.gaussian_centers()
    quats = model.gaussian_rotations()
    scales = model.gaussian_scales()
    opacities = torch.sigmoid(model._raw_opacities)
    sh_coeffs = torch.cat([model._sh_dc, model._sh_rest], dim=1)

    renders, alphas, _ = rasterization(
        means=centers,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=sh_coeffs,
        viewmats=viewmat,
        Ks=K,
        width=camera.width,
        height=camera.height,
        sh_degree=sh_degree,
        render_mode="RGB",
        packed=True,
    )
    rendered = renders.permute(0, 3, 1, 2).clamp(0, 1)
    alphas = alphas.permute(0, 3, 1, 2) if alphas.dim() == 4 else alphas
    return rendered, alphas


def refine_mesh(
    mesh: trimesh.Trimesh,
    cameras: List[Camera],
    config: MeshExtractionConfig,
) -> trimesh.Trimesh:
    """Jointly optimize mesh vertices + bound Gaussians.

    Args:
        mesh: coarse mesh from Poisson reconstruction.
        cameras: training cameras with images.
        config: extraction configuration.

    Returns:
        Refined trimesh.
    """
    device = config.device
    trainable_cameras = [c for c in cameras if c.image_path is not None]
    if not trainable_cameras:
        print("WARNING: No training images found, skipping refinement")
        return mesh

    model = BoundGaussianModel.from_mesh(
        mesh,
        n_per_triangle=config.n_gaussians_per_triangle,
        device=device,
        surface_thickness=config.surface_mesh_thickness,
    )

    n_verts = model._vertices.shape[0]
    bbox_radius = (
        model._vertices.max(dim=0).values - model._vertices.min(dim=0).values
    ).norm().item() / 2

    optimizer = torch.optim.Adam([
        {"params": [model._vertices], "lr": config.lr_vertices},
        {"params": [model._log_scales_2d], "lr": 0.005},
        {"params": [model._tangent_rotation], "lr": 0.001},
        {"params": [model._raw_opacities], "lr": 0.05},
        {"params": [model._sh_dc], "lr": 0.0025},
        {"params": [model._sh_rest], "lr": 0.000125},
    ], eps=1e-15)

    n_iters = config.refinement_iterations
    print(
        f"Starting mesh refinement: {n_iters} iterations, "
        f"{model.n_gaussians} bound Gaussians on "
        f"{model._faces.shape[0]} triangles"
    )

    for step in range(n_iters):
        cam = random.choice(trainable_cameras)

        rendered, _ = _render_bound_gsplat(model, cam, device)
        gt_image = cam.load_image(device)

        loss_photo = photometric_loss(rendered, gt_image)

        loss_normal = model.mesh_normal_consistency_loss()
        loss = loss_photo + config.refinement_normal_weight * loss_normal

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 500 == 0 or step == n_iters - 1:
            print(
                f"  [refine iter {step}/{n_iters}] "
                f"loss={loss.item():.5f} "
                f"photo={loss_photo.item():.5f} "
                f"normal={loss_normal.item():.5f}"
            )

    print("Mesh refinement complete.")
    return model.export_mesh()
