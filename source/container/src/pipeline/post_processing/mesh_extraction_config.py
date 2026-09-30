# MIT License — see __init__.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MeshExtractionConfig:
    """Configuration for the Gaussian-to-mesh extraction pipeline."""

    # --- Regularization training ---
    regularization_iterations: int = 8000
    entropy_start_iter: int = 0
    entropy_end_iter: int = 2000
    entropy_weight: float = 0.1
    prune_opacity_threshold: float = 0.5
    prune_at_iter: int = 2000
    surface_alignment_weight: float = 0.2
    normal_consistency_weight: float = 0.2
    photometric_l1_weight: float = 0.8
    photometric_ssim_weight: float = 0.2

    # --- Density / SDF ---
    density_threshold: float = 0.5
    k_neighbors: int = 16
    n_samples_per_gaussian: int = 2
    sampling_scale_factor: float = 1.5

    # --- Learning rates ---
    lr_position: float = 1.6e-4
    lr_position_final_factor: float = 0.01
    lr_opacity: float = 0.05
    lr_scale: float = 0.005
    lr_quaternion: float = 0.001
    lr_sh_dc: float = 0.0025
    lr_sh_rest: float = 0.000125

    # --- Surface extraction ---
    surface_levels: List[float] = field(
        default_factory=lambda: [0.1, 0.3, 0.5]
    )
    n_ray_samples: int = 21
    ray_range_sigmas: float = 3.0
    min_points_per_view: int = 100

    # --- Poisson reconstruction ---
    poisson_depth: int = 8
    poisson_density_quantile: float = 0.02
    decimation_target: int = 200_000

    # --- Mesh refinement ---
    enable_refinement: bool = False
    n_gaussians_per_triangle: int = 6
    refinement_iterations: int = 15000
    surface_mesh_thickness: float = 1e-6
    refinement_normal_weight: float = 0.1
    lr_vertices: float = 1e-4

    # --- General ---
    device: str = "cuda"
    batch_size: int = 1
    chunk_size: int = 5_000
