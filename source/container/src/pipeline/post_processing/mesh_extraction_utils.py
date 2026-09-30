# MIT License — see __init__.py
"""Math utilities for Gaussian-to-mesh extraction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def quaternion_to_rotation_matrix(q: Tensor) -> Tensor:
    """Convert unit quaternions to 3x3 rotation matrices.

    Args:
        q: [..., 4] quaternions (w, x, y, z convention).

    Returns:
        [..., 3, 3] rotation matrices.
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)

    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def build_covariance(
    scales: Tensor, rotation_matrices: Tensor
) -> Tensor:
    """Build 3x3 covariance matrices from scales and rotations.

    Sigma = R @ S @ S^T @ R^T  where S = diag(scales).

    Args:
        scales: [..., 3] scale factors (already exponentiated).
        rotation_matrices: [..., 3, 3] rotation matrices.

    Returns:
        [..., 3, 3] covariance matrices.
    """
    S = torch.diag_embed(scales)  # [..., 3, 3]
    M = rotation_matrices @ S  # [..., 3, 3]
    return M @ M.transpose(-1, -2)


def build_covariance_inverse(
    scales: Tensor, rotation_matrices: Tensor
) -> Tensor:
    """Build inverse covariance: Sigma^-1 = R @ S^-2 @ R^T.

    Args:
        scales: [..., 3] scale factors (already exponentiated, must be > 0).
        rotation_matrices: [..., 3, 3] rotation matrices.

    Returns:
        [..., 3, 3] inverse covariance matrices.
    """
    S_inv_sq = torch.diag_embed(1.0 / (scales * scales).clamp(min=1e-6))
    return rotation_matrices @ S_inv_sq @ rotation_matrices.transpose(-1, -2)


def chunked_knn(
    query: Tensor,
    reference: Tensor,
    k: int,
    chunk_size: int = 50_000,
) -> tuple[Tensor, Tensor]:
    """K-nearest-neighbors via chunked L2 distance.

    Args:
        query: [M, 3] query points.
        reference: [N, 3] reference points.
        k: number of neighbors.
        chunk_size: process this many query points per chunk.

    Returns:
        distances: [M, k] squared L2 distances.
        indices: [M, k] indices into reference.
    """
    all_dists = []
    all_idxs = []
    for i in range(0, len(query), chunk_size):
        chunk = query[i : i + chunk_size]
        dists = torch.cdist(chunk, reference)  # [chunk, N]
        topk = dists.topk(k, dim=-1, largest=False)
        all_dists.append(topk.values)
        all_idxs.append(topk.indices)
    return torch.cat(all_dists, dim=0), torch.cat(all_idxs, dim=0)


def sample_points_in_gaussians(
    positions: Tensor,
    scales: Tensor,
    rotation_matrices: Tensor,
    opacities: Tensor,
    n_per_gaussian: int = 2,
    scale_factor: float = 1.5,
) -> tuple[Tensor, Tensor]:
    """Sample 3D points inside Gaussians, weighted by volume * opacity.

    Points are sampled as: x = mu + R @ (s * z * scale_factor), z ~ N(0,I)

    Args:
        positions: [N, 3] Gaussian centers.
        scales: [N, 3] scale factors.
        rotation_matrices: [N, 3, 3] rotation matrices.
        opacities: [N] sigmoid opacities.
        n_per_gaussian: samples per Gaussian.
        scale_factor: multiplier on the sampling envelope.

    Returns:
        points: [N * n_per_gaussian, 3] sampled points.
        gaussian_indices: [N * n_per_gaussian] which Gaussian each came from.
    """
    N = positions.shape[0]
    device = positions.device

    z = torch.randn(N, n_per_gaussian, 3, device=device)
    scaled_z = z * scales[:, None, :] * scale_factor  # [N, n, 3]

    rotated = torch.einsum("nij,nmj->nmi", rotation_matrices, scaled_z)
    points = positions[:, None, :] + rotated  # [N, n, 3]

    points = points.reshape(-1, 3)
    gaussian_indices = (
        torch.arange(N, device=device)
        .unsqueeze(1)
        .expand(-1, n_per_gaussian)
        .reshape(-1)
    )
    return points, gaussian_indices


def ssim(
    img1: Tensor, img2: Tensor, window_size: int = 11
) -> Tensor:
    """Compute mean SSIM between two images.

    Args:
        img1, img2: [B, C, H, W] images in [0, 1].
        window_size: size of the Gaussian window.

    Returns:
        Scalar mean SSIM.
    """
    C = img1.shape[1]
    sigma = 1.5
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]  # [ws, ws]
    window = window.expand(C, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)

    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu12
    )

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()
