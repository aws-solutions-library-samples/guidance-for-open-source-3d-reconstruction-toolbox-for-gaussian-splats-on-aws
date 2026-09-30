# MIT License — see __init__.py
from __future__ import annotations

"""Regularization training loop.

Continues training Gaussians with surface-alignment losses to flatten
them onto surfaces, enabling high-quality mesh extraction.

Based on the training procedure described in Section 3.3 of
Guédon & Lepetit, CVPR 2024 (arXiv:2311.12775).
"""

import random
from typing import List

import torch
from torch import Tensor

from mesh_extraction_cameras import Camera
from mesh_extraction_config import MeshExtractionConfig
from mesh_extraction_gaussian_model import GaussianModel
from mesh_extraction_losses import (
    entropy_loss,
    normal_consistency_loss,
    photometric_loss,
    surface_alignment_loss,
)
from mesh_extraction_utils import sample_points_in_gaussians


def _render_gsplat(
    model: GaussianModel,
    camera: Camera,
    device: str = "cuda",
    sh_degree: int = 3,
) -> tuple[Tensor, Tensor]:
    """Render an image using gsplat's rasterization.

    Returns:
        rendered: [1, 3, H, W] rendered RGB image.
        alphas: [1, 1, H, W] accumulated alpha.
    """
    from gsplat import rasterization

    viewmat = camera.viewmat(device).unsqueeze(0)  # [1, 4, 4]
    K = camera.intrinsic_matrix(device).unsqueeze(0)  # [1, 3, 3]

    sh_coeffs = model.sh_coefficients()  # [N, K, 3]
    actual_sh_degree = min(
        sh_degree,
        {1: 0, 4: 1, 9: 2, 16: 3}.get(sh_coeffs.shape[1], 0),
    )

    renders, alphas, _info = rasterization(
        means=model.positions,
        quats=model._quaternions,
        scales=torch.exp(model._log_scales),
        opacities=torch.sigmoid(model._raw_opacities),
        colors=sh_coeffs,
        viewmats=viewmat,
        Ks=K,
        width=camera.width,
        height=camera.height,
        sh_degree=actual_sh_degree,
        render_mode="RGB",
        packed=True,
    )
    # gsplat returns [C, H, W, 3] — transpose to [C, 3, H, W]
    rendered = renders.permute(0, 3, 1, 2).clamp(0, 1)
    alphas = alphas.permute(0, 3, 1, 2) if alphas.dim() == 4 else alphas
    return rendered, alphas


def _render_depth_gsplat(
    model: GaussianModel,
    camera: Camera,
    device: str = "cuda",
) -> Tensor:
    """Render a depth map by alpha-blending per-Gaussian camera-space z.

    Returns:
        depth: [H, W] depth map (camera-space z).
    """
    from gsplat import rasterization

    viewmat = camera.viewmat(device).unsqueeze(0)
    K = camera.intrinsic_matrix(device).unsqueeze(0)

    pos_h = torch.cat(
        [model.positions, torch.ones(model.n_gaussians, 1, device=device)],
        dim=-1,
    )
    cam_pos = (viewmat[0] @ pos_h.T).T  # [N, 4]
    depths_per_gaussian = cam_pos[:, 2:3].expand(-1, 3)  # [N, 3]

    renders, _, _ = rasterization(
        means=model.positions,
        quats=model._quaternions,
        scales=torch.exp(model._log_scales),
        opacities=torch.sigmoid(model._raw_opacities),
        colors=depths_per_gaussian,
        viewmats=viewmat,
        Ks=K,
        width=camera.width,
        height=camera.height,
        render_mode="RGB",
        sh_degree=None,
        packed=True,
    )
    return renders[0, :, :, 0]  # [H, W]


def _make_optimizer(
    model: GaussianModel, config: MeshExtractionConfig
) -> torch.optim.Adam:
    """Create the Adam optimizer with per-parameter learning rates."""
    spatial_lr = config.lr_position * model.spatial_extent()

    param_groups = [
        {"params": [model._positions], "lr": spatial_lr, "name": "positions"},
        {"params": [model._quaternions], "lr": config.lr_quaternion, "name": "quats"},
        {"params": [model._log_scales], "lr": config.lr_scale, "name": "scales"},
        {"params": [model._raw_opacities], "lr": config.lr_opacity, "name": "opacs"},
        {"params": [model._sh_dc], "lr": config.lr_sh_dc, "name": "sh_dc"},
        {"params": [model._sh_rest], "lr": config.lr_sh_rest, "name": "sh_rest"},
    ]
    return torch.optim.Adam(param_groups, eps=1e-15)


def _update_position_lr(
    optimizer: torch.optim.Adam,
    step: int,
    total_steps: int,
    config: MeshExtractionConfig,
    spatial_extent: float,
) -> None:
    """Exponential decay for position learning rate."""
    lr_init = config.lr_position * spatial_extent
    lr_final = lr_init * config.lr_position_final_factor
    t = step / max(total_steps, 1)
    lr = lr_init * (lr_final / lr_init) ** t
    for pg in optimizer.param_groups:
        if pg["name"] == "positions":
            pg["lr"] = lr


def regularize(
    model: GaussianModel,
    cameras: List[Camera],
    config: MeshExtractionConfig,
) -> GaussianModel:
    """Run regularization training to align Gaussians with surfaces.

    Training schedule:
    - Phase 1 (0 to prune_at_iter): photometric + entropy loss
    - Prune low-opacity Gaussians
    - Phase 2 (prune_at_iter to end): photometric + surface alignment + normal consistency

    Args:
        model: pretrained GaussianModel.
        cameras: training cameras with images.
        config: extraction configuration.

    Returns:
        The regularized GaussianModel (modified in-place).
    """
    device = config.device
    model = model.to(device)
    model.train()

    optimizer = _make_optimizer(model, config)
    spatial_extent = model.spatial_extent()
    n_iters = config.regularization_iterations
    trainable_cameras = [c for c in cameras if c.image_path is not None]

    if not trainable_cameras:
        print("WARNING: No training images found, skipping regularization")
        return model

    print(
        f"Starting regularization: {n_iters} iterations, "
        f"{model.n_gaussians} Gaussians, {len(trainable_cameras)} cameras"
    )

    for step in range(n_iters):
        _update_position_lr(optimizer, step, n_iters, config, spatial_extent)

        cam = random.choice(trainable_cameras)

        # --- Photometric loss ---
        rendered, _alphas = _render_gsplat(model, cam, device)
        gt_image = cam.load_image(device)
        loss_photo = photometric_loss(
            rendered, gt_image,
            l1_weight=config.photometric_l1_weight,
            ssim_weight=config.photometric_ssim_weight,
        )

        loss = loss_photo

        # --- Phase 1: Entropy regularization ---
        if config.entropy_start_iter <= step < config.entropy_end_iter:
            loss_ent = entropy_loss(model.opacities)
            loss = loss + config.entropy_weight * loss_ent

        # --- Prune low-opacity Gaussians ---
        if step == config.prune_at_iter:
            with torch.no_grad():
                mask = model.opacities < config.prune_opacity_threshold
                n_before = model.n_gaussians
                model.prune(mask)
                print(
                    f"  [iter {step}] Pruned {mask.sum().item()} "
                    f"Gaussians ({n_before} -> {model.n_gaussians})"
                )
                optimizer = _make_optimizer(model, config)

        # --- Phase 2: Surface alignment + normal consistency ---
        if step >= config.prune_at_iter:
            with torch.no_grad():
                positions = model.positions
                scales = model.scales
                rot_mats = model.rotation_matrices
                opacities_val = model.opacities

            sample_pts, sample_idxs = sample_points_in_gaussians(
                positions.detach(),
                scales.detach(),
                rot_mats.detach(),
                opacities_val.detach(),
                n_per_gaussian=config.n_samples_per_gaussian,
                scale_factor=config.sampling_scale_factor,
            )

            loss_align = surface_alignment_loss(
                sample_pts,
                sample_idxs,
                model.positions,
                model.normals,
                model.scales,
                model.covariances_inverse,
                model.opacities,
                k_neighbors=config.k_neighbors,
                chunk_size=config.chunk_size,
            )
            loss = loss + config.surface_alignment_weight * loss_align

            loss_normal = normal_consistency_loss(
                sample_pts,
                sample_idxs,
                model.positions,
                model.normals,
                model.scales,
                model.opacities,
                k_neighbors=config.k_neighbors,
                chunk_size=config.chunk_size,
            )
            loss = loss + config.normal_consistency_weight * loss_normal

        # --- Backward + step ---
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            if step % 500 == 0:
                print(f"  [iter {step}] NaN/Inf loss, skipping")
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            model._log_scales.clamp_(-15.0, -1.6)

        if step % 500 == 0 or step == n_iters - 1:
            parts = [f"iter {step}/{n_iters}", f"loss={loss.item():.5f}"]
            parts.append(f"photo={loss_photo.item():.5f}")
            if step >= config.prune_at_iter:
                parts.append(f"align={loss_align.item():.5f}")
                parts.append(f"normal={loss_normal.item():.5f}")
            print(f"  [{'] ['.join(parts)}]")

    model.eval()
    print(f"Regularization complete. {model.n_gaussians} Gaussians remain.")
    return model
