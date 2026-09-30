# MIT License — see __init__.py
"""Density and SDF field evaluation from Gaussian mixtures.

Based on the density field formulation in Section 3.2 of
Guédon & Lepetit, CVPR 2024 (arXiv:2311.12775).
"""

from __future__ import annotations

import torch
from torch import Tensor



def evaluate_density(
    query_points: Tensor,
    positions: Tensor,
    covariances_inverse: Tensor,
    opacities: Tensor,
    k_neighbors: int = 16,
    chunk_size: int = 50_000,
) -> Tensor:
    """Evaluate Gaussian mixture density at query points.

    d(x) = sum_k alpha_k * exp(-0.5 * (x-mu_k)^T Sigma_k^-1 (x-mu_k))

    where the sum is over the K nearest Gaussians.

    Args:
        query_points: [M, 3] points to evaluate.
        positions: [N, 3] Gaussian centers.
        covariances_inverse: [N, 3, 3] inverse covariance matrices.
        opacities: [N] sigmoid opacities.
        k_neighbors: number of nearest Gaussians to consider.
        chunk_size: process queries in chunks for memory.

    Returns:
        [M] density values.
    """
    M = query_points.shape[0]
    device = query_points.device
    densities = torch.zeros(M, device=device)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        chunk = query_points[start:end]

        dists = torch.cdist(chunk, positions)
        _, knn_idx = dists.topk(k_neighbors, dim=-1, largest=False)

        nn_pos = positions[knn_idx]  # [C, K, 3]
        nn_cov_inv = covariances_inverse[knn_idx]  # [C, K, 3, 3]
        nn_opac = opacities[knn_idx]  # [C, K]

        diff = chunk[:, None, :] - nn_pos  # [C, K, 3]

        # Mahalanobis distance: diff^T @ Sigma^-1 @ diff
        mahal = torch.einsum("ckj,ckjl,ckl->ck", diff, nn_cov_inv, diff)
        mahal = mahal.clamp(min=0.0, max=100.0)
        gauss_vals = torch.exp(-0.5 * mahal)  # [C, K]

        densities[start:end] = (nn_opac * gauss_vals).sum(dim=-1)

    return densities


def evaluate_density_with_grad(
    query_points: Tensor,
    positions: Tensor,
    covariances_inverse: Tensor,
    opacities: Tensor,
    k_neighbors: int = 16,
    chunk_size: int = 50_000,
) -> tuple[Tensor, Tensor]:
    """Evaluate density and its gradient at query points.

    Returns:
        density: [M] density values.
        grad: [M, 3] gradient of density w.r.t. query points.
    """
    M = query_points.shape[0]
    device = query_points.device
    densities = torch.zeros(M, device=device)
    gradients = torch.zeros(M, 3, device=device)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        chunk = query_points[start:end]

        dists = torch.cdist(chunk, positions)
        _, knn_idx = dists.topk(k_neighbors, dim=-1, largest=False)

        nn_pos = positions[knn_idx]
        nn_cov_inv = covariances_inverse[knn_idx]
        nn_opac = opacities[knn_idx]

        diff = chunk[:, None, :] - nn_pos  # [C, K, 3]

        mahal = torch.einsum("ckj,ckjl,ckl->ck", diff, nn_cov_inv, diff)
        mahal = mahal.clamp(min=0.0, max=100.0)
        gauss_vals = torch.exp(-0.5 * mahal)  # [C, K]
        weighted = nn_opac * gauss_vals  # [C, K]

        densities[start:end] = weighted.sum(dim=-1)

        # grad d(x) = sum_k -alpha_k * g_k * Sigma_k^-1 @ (x - mu_k)
        cov_inv_diff = torch.einsum(
            "ckij,ckj->cki", nn_cov_inv, diff
        )  # [C, K, 3]
        grad_contrib = -weighted.unsqueeze(-1) * cov_inv_diff
        gradients[start:end] = grad_contrib.sum(dim=1)

    return densities, gradients


def compute_beta(
    query_points: Tensor,
    positions: Tensor,
    scales: Tensor,
    k_neighbors: int = 16,
    chunk_size: int = 50_000,
) -> Tensor:
    """Compute local scale parameter beta(x).

    beta(x) = mean of min-scale over K nearest Gaussians.
    This is the local characteristic scale used in the
    density-to-SDF conversion.

    Args:
        query_points: [M, 3] points.
        positions: [N, 3] Gaussian centers.
        scales: [N, 3] Gaussian scales.
        k_neighbors: K for KNN.

    Returns:
        [M] beta values.
    """
    min_scales = scales.min(dim=-1).values  # [N]
    M = query_points.shape[0]
    device = query_points.device
    betas = torch.zeros(M, device=device)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        chunk = query_points[start:end]
        dists = torch.cdist(chunk, positions)
        _, knn_idx = dists.topk(k_neighbors, dim=-1, largest=False)
        nn_min_scales = min_scales[knn_idx]  # [C, K]
        betas[start:end] = nn_min_scales.mean(dim=-1)

    return betas


def density_to_sdf(
    density: Tensor,
    beta: Tensor,
    density_threshold: float = 0.5,
) -> Tensor:
    """Convert density to approximate SDF.

    f(x) = beta * (sqrt(-2 * log(d(x))) - sqrt(-2 * log(tau)))

    From Section 3.2 of the paper. This inverts the Gaussian
    density relationship to approximate a signed distance.

    Args:
        density: [M] density values (must be > 0).
        beta: [M] local scale parameters.
        density_threshold: tau, the surface-level density.

    Returns:
        [M] approximate SDF values.
    """
    eps = 1e-7
    d_clamped = density.clamp(min=eps)
    threshold_term = (
        -2.0 * torch.tensor(density_threshold + eps).log()
    ).sqrt()
    return beta * ((-2.0 * d_clamped.log()).sqrt() - threshold_term)
