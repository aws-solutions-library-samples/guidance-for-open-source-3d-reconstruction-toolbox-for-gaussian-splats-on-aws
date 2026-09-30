# MIT License — see __init__.py
from __future__ import annotations

"""Regularization losses for surface-aligned Gaussian splatting.

Implements the regularization terms described in Section 3.3 of
Guédon & Lepetit, CVPR 2024 (arXiv:2311.12775):
  - Entropy loss on opacities (standard technique)
  - Surface alignment loss (density matches target Gaussian profile)
  - Normal consistency loss (neighboring Gaussians agree on normal direction)
"""

import torch
from torch import Tensor

from mesh_extraction_density import evaluate_density
from mesh_extraction_utils import chunked_knn, ssim


def photometric_loss(
    rendered: Tensor,
    target: Tensor,
    l1_weight: float = 0.8,
    ssim_weight: float = 0.2,
) -> Tensor:
    """L1 + D-SSIM photometric loss (standard in 3DGS).

    Args:
        rendered: [B, C, H, W] rendered image.
        target: [B, C, H, W] ground truth.
        l1_weight: weight for L1 term.
        ssim_weight: weight for (1 - SSIM) term.
    """
    l1 = (rendered - target).abs().mean()
    ssim_val = ssim(rendered, target)
    return l1_weight * l1 + ssim_weight * (1.0 - ssim_val)


def entropy_loss(opacities: Tensor) -> Tensor:
    """Binary entropy loss on opacities.

    Pushes opacities toward 0 or 1 (fully transparent or opaque).
    Standard technique in neural opacity fields.

    L = mean( -alpha * log(alpha) - (1-alpha) * log(1-alpha) )
    """
    eps = 1e-7
    a = opacities.clamp(eps, 1.0 - eps)
    return (-a * a.log() - (1.0 - a) * (1.0 - a).log()).mean()


def surface_alignment_loss(
    sample_points: Tensor,
    sample_gaussian_indices: Tensor,
    positions: Tensor,
    normals: Tensor,
    scales: Tensor,
    covariances_inverse: Tensor,
    opacities: Tensor,
    k_neighbors: int = 16,
    chunk_size: int = 50_000,
) -> Tensor:
    """Surface alignment regularization (paper Section 3.3).

    For each sampled point near Gaussian i:
    1. Estimate the SDF as projection onto Gaussian's normal:
       sdf_est = dot(sample - center_i, normal_i)
    2. Compute the density that SHOULD exist at this SDF if
       the Gaussians define a proper surface:
       target_density = exp(-0.5 * sdf_est^2 / beta_i^2)
       where beta_i = min_scale of Gaussian i.
    3. Compute actual density from the Gaussian mixture.
    4. Loss = |actual_density - target_density|

    Args:
        sample_points: [M, 3] points sampled near Gaussians.
        sample_gaussian_indices: [M] index of the Gaussian each point came from.
        positions: [N, 3] Gaussian centers.
        normals: [N, 3] Gaussian normals (shortest axis).
        scales: [N, 3] Gaussian scales.
        covariances_inverse: [N, 3, 3] inverse covariances.
        opacities: [N] sigmoid opacities.
        k_neighbors: K for density evaluation.
        chunk_size: chunk size for density evaluation.

    Returns:
        Scalar loss.
    """
    source_pos = positions[sample_gaussian_indices]  # [M, 3]
    source_normals = normals[sample_gaussian_indices]  # [M, 3]
    source_min_scale = scales[sample_gaussian_indices].min(dim=-1).values

    diff = sample_points - source_pos
    sdf_est = (diff * source_normals).sum(dim=-1)  # [M]

    beta = source_min_scale.clamp(min=1e-7)
    target_density = torch.exp(-0.5 * (sdf_est / beta) ** 2)

    actual_density = evaluate_density(
        sample_points,
        positions,
        covariances_inverse,
        opacities,
        k_neighbors=k_neighbors,
        chunk_size=chunk_size,
    )

    # Density clamping with detached gradient: keeps gradients flowing
    # but bounds the magnitude when density exceeds 1.0
    clamped = torch.where(
        actual_density > 1.0,
        actual_density / (actual_density.detach() + 1e-5),
        actual_density,
    )

    return (clamped - target_density.detach()).abs().mean()


def normal_consistency_loss(
    sample_points: Tensor,
    sample_gaussian_indices: Tensor,
    positions: Tensor,
    normals: Tensor,
    scales: Tensor,
    opacities: Tensor,
    k_neighbors: int = 16,
    chunk_size: int = 50_000,
) -> Tensor:
    """Normal consistency regularization (paper Section 3.3).

    Encourages neighboring Gaussians to have consistent normals,
    weighted by their contribution to the local SDF gradient.

    For each sample point near Gaussian i, with K nearest neighbors j:
    1. Flip neighbor normals to agree with source normal.
    2. Weight: w_j = alpha_j * |dot(sample - center_j, normal_j)| / min_scale_j^2
    3. Normalize weights to sum to 1.
    4. Loss = ||normal_i - sum_j(w_j * normal_j)||^2

    Gradients flow only through normals, not weights (per paper).

    Args:
        sample_points: [M, 3] sampled points.
        sample_gaussian_indices: [M] source Gaussian indices.
        positions: [N, 3] Gaussian centers.
        normals: [N, 3] Gaussian normals.
        scales: [N, 3] Gaussian scales.
        opacities: [N] opacities.
        k_neighbors: K for neighbor lookup.
        chunk_size: chunk size.

    Returns:
        Scalar loss.
    """
    M = sample_points.shape[0]
    device = sample_points.device
    min_scales = scales.min(dim=-1).values  # [N]
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        chunk_pts = sample_points[start:end]
        chunk_src = sample_gaussian_indices[start:end]
        C = chunk_pts.shape[0]

        dists = torch.cdist(chunk_pts, positions)
        _, knn_idx = dists.topk(k_neighbors, dim=-1, largest=False)

        src_normals = normals[chunk_src]  # [C, 3]
        nn_normals = normals[knn_idx]  # [C, K, 3]
        nn_opacities = opacities[knn_idx]  # [C, K]
        nn_positions = positions[knn_idx]  # [C, K, 3]
        nn_min_scales = min_scales[knn_idx]  # [C, K]

        # Flip neighbor normals to agree with source
        dot_sign = (src_normals[:, None, :] * nn_normals).sum(dim=-1)
        flip = torch.where(dot_sign >= 0, 1.0, -1.0)
        nn_normals_aligned = nn_normals * flip.unsqueeze(-1)

        # Compute weights (detached — no gradient through weights)
        diff = chunk_pts[:, None, :] - nn_positions  # [C, K, 3]
        proj = (diff * nn_normals_aligned).sum(dim=-1).abs()  # [C, K]
        scale_sq = (nn_min_scales ** 2).clamp(min=1e-12)
        weights = (nn_opacities * proj / scale_sq).detach()  # [C, K]

        # Normalize
        w_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = weights / w_sum  # [C, K]

        # Weighted average normal
        avg_normal = (weights.unsqueeze(-1) * nn_normals_aligned).sum(
            dim=1
        )  # [C, 3]

        loss = ((src_normals - avg_normal) ** 2).sum(dim=-1)  # [C]
        total_loss = total_loss + loss.sum()
        count += C

    if count > 0:
        total_loss = total_loss / count
    return total_loss
