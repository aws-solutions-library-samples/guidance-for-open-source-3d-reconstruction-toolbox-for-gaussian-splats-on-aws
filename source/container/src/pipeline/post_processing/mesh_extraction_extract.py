# MIT License — see __init__.py
from __future__ import annotations

"""Surface point extraction and mesh reconstruction.

Implements the mesh extraction pipeline from Section 3.4 of
Guédon & Lepetit, CVPR 2024 (arXiv:2311.12775):
1. Render depth from each camera.
2. Backproject to 3D and sample along camera rays.
3. Evaluate density at samples.
4. Find level-set crossings via linear interpolation.
5. Compute normals from density gradient.
6. Run Poisson reconstruction (Kazhdan & Hoppe 2013).
7. Decimate mesh (Garland & Heckbert 1997).
"""

from typing import List, Optional

import numpy as np
import open3d as o3d
import torch
import trimesh
from torch import Tensor

from mesh_extraction_cameras import Camera
from mesh_extraction_config import MeshExtractionConfig
from mesh_extraction_density import evaluate_density, evaluate_density_with_grad
from mesh_extraction_gaussian_model import GaussianModel


def _backproject_depth(
    depth: Tensor,
    camera: Camera,
    device: str = "cuda",
) -> Tensor:
    """Backproject a depth map to world-space 3D points.

    Args:
        depth: [H, W] depth map.
        camera: Camera with intrinsics and extrinsics.

    Returns:
        [H*W, 3] world-space points.
    """
    H, W = depth.shape
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    x_cam = (u - camera.cx) / camera.fx * depth
    y_cam = (v - camera.cy) / camera.fy * depth
    z_cam = depth

    pts_cam = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(z_cam)], dim=-1)
    pts_cam = pts_cam.reshape(-1, 4)

    c2w = torch.tensor(camera.c2w_opencv, dtype=torch.float32, device=device)
    pts_world = (c2w @ pts_cam.T).T[:, :3]
    return pts_world


def _compute_ray_directions(
    camera: Camera,
    device: str = "cuda",
) -> Tensor:
    """Compute world-space ray directions for each pixel.

    Returns:
        [H*W, 3] normalized ray directions.
    """
    H, W = camera.height, camera.width
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    dirs_cam = torch.stack([
        (u - camera.cx) / camera.fx,
        (v - camera.cy) / camera.fy,
        torch.ones_like(u),
    ], dim=-1)  # [H, W, 3]
    dirs_cam = dirs_cam.reshape(-1, 3)

    c2w = torch.tensor(camera.c2w_opencv, dtype=torch.float32, device=device)
    R = c2w[:3, :3]
    dirs_world = (R @ dirs_cam.T).T
    dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return dirs_world


def extract_surface_points_from_camera(
    model: GaussianModel,
    camera: Camera,
    surface_level: float = 0.3,
    n_ray_samples: int = 21,
    range_sigmas: float = 3.0,
    device: str = "cuda",
    chunk_size: int = 50_000,
) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """Extract surface-level points from a single camera view.

    1. Render depth map from Gaussians.
    2. Backproject to get approximate surface locations.
    3. Sample along rays around the depth estimate.
    4. Evaluate density at all samples.
    5. Find level-set crossings via interpolation.
    6. Compute normals from density gradient.

    Args:
        model: regularized GaussianModel.
        camera: camera viewpoint.
        surface_level: density threshold for the surface.
        n_ray_samples: number of samples per ray.
        range_sigmas: sample range in units of local Gaussian scale.
        device: compute device.
        chunk_size: points per density evaluation chunk.

    Returns:
        points: [P, 3] surface points (or None if no crossings found).
        normals: [P, 3] surface normals.
        colors: [P, 3] approximate colors from nearest Gaussian SH DC.
    """
    from mesh_extraction_regularize import _render_depth_gsplat

    with torch.no_grad():
        depth_map = _render_depth_gsplat(model, camera, device)

    H, W = camera.height, camera.width
    depth_flat = depth_map.reshape(-1)  # [H*W]

    # Filter out invalid depths
    valid = depth_flat > 0.01
    if valid.sum() < 100:
        return None, None, None

    # Compute ray origins and directions (OpenCV convention)
    c2w = torch.tensor(camera.c2w_opencv, dtype=torch.float32, device=device)
    ray_origins = c2w[:3, 3].expand(H * W, 3)  # [H*W, 3]
    ray_dirs = _compute_ray_directions(camera, device)  # [H*W, 3]

    # Approximate the local scale at each depth point for sampling range
    pts_3d = _backproject_depth(depth_map, camera, device)

    # Find nearest Gaussian to each point for scale estimate
    with torch.no_grad():
        subsample = valid.nonzero(as_tuple=True)[0]
        if len(subsample) > 50_000:
            perm = torch.randperm(len(subsample), device=device)[:50_000]
            subsample = subsample[perm]

        sub_pts = pts_3d[subsample]
        sub_dirs = ray_dirs[subsample]
        sub_depths = depth_flat[subsample]

        # Convert z-depth to ray t (Euclidean distance along normalized ray)
        ray_z_component = sub_dirs[:, 2].abs().clamp(min=1e-6)
        sub_t = sub_depths / ray_z_component

        # Find nearest Gaussian to estimate local scale (chunked)
        nn_idx = torch.empty(len(sub_pts), dtype=torch.long, device=device)
        for ci in range(0, len(sub_pts), chunk_size):
            ce = min(ci + chunk_size, len(sub_pts))
            d = torch.cdist(sub_pts[ci:ce], model.positions)
            nn_idx[ci:ce] = d.argmin(dim=-1)
            del d
        local_scale = model.min_scales[nn_idx]  # [S]

        # Sample range: +/- range_sigmas * local_scale along ray
        sample_range = range_sigmas * local_scale  # [S]
        t_min = sub_t - sample_range  # [S]
        t_max = sub_t + sample_range  # [S]
        t_min = t_min.clamp(min=0.01)

        # Create sample points along each ray
        S = len(subsample)
        t_vals = torch.linspace(0, 1, n_ray_samples, device=device)
        t_vals = t_vals.unsqueeze(0).expand(S, -1)  # [S, n_ray_samples]
        t_samples = t_min.unsqueeze(-1) + t_vals * (
            t_max - t_min
        ).unsqueeze(-1)  # [S, n_ray_samples]

        ray_o = c2w[:3, 3].unsqueeze(0).expand(S, 3)
        sample_pts = (
            ray_o.unsqueeze(1)
            + sub_dirs.unsqueeze(1) * t_samples.unsqueeze(-1)
        )  # [S, n_ray_samples, 3]

        # Evaluate density at all sample points
        all_pts = sample_pts.reshape(-1, 3)
        all_density = evaluate_density(
            all_pts,
            model.positions,
            model.covariances_inverse,
            model.opacities,
            k_neighbors=min(16, model.n_gaussians),
            chunk_size=chunk_size,
        )
        density_grid = all_density.reshape(S, n_ray_samples)

        # Find level-set crossings
        above = density_grid >= surface_level
        below = density_grid < surface_level

        # Crossing: below[i] & above[i+1] (entering surface from camera)
        crossing = below[:, :-1] & above[:, 1:]
        has_crossing = crossing.any(dim=-1)

        if has_crossing.sum() == 0:
            return None, None, None

        # For rays with crossings, take the first crossing
        crossing_idx = crossing.float().argmax(dim=-1)  # [S]
        valid_rays = has_crossing

        ray_indices = valid_rays.nonzero(as_tuple=True)[0]
        cross_k = crossing_idx[ray_indices]

        d_before = density_grid[ray_indices, cross_k]
        d_after = density_grid[ray_indices, cross_k + 1]
        t_before = t_samples[ray_indices, cross_k]
        t_after = t_samples[ray_indices, cross_k + 1]

        # Linear interpolation for crossing point
        w = (surface_level - d_before) / (d_after - d_before + 1e-8)
        w = w.clamp(0, 1)
        t_cross = t_before + w * (t_after - t_before)

        surface_pts = (
            ray_o[ray_indices]
            + sub_dirs[ray_indices] * t_cross.unsqueeze(-1)
        )

        # Compute normals from density gradient
        _, grad = evaluate_density_with_grad(
            surface_pts,
            model.positions,
            model.covariances_inverse,
            model.opacities,
            k_neighbors=min(16, model.n_gaussians),
            chunk_size=chunk_size,
        )
        surface_normals = -grad / grad.norm(dim=-1, keepdim=True).clamp(
            min=1e-8
        )

        # Get colors from the rendered RGB image at surface pixel locations
        from mesh_extraction_regularize import _render_gsplat
        rendered, _ = _render_gsplat(model, camera, device)
        rendered_flat = rendered[0].permute(1, 2, 0).reshape(-1, 3)
        pixel_indices = subsample[ray_indices]
        colors = rendered_flat[pixel_indices].clamp(0, 1)

    return surface_pts, surface_normals, colors


def _extract_gaussian_center_points(
    model: GaussianModel,
    opacity_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract surface points directly from Gaussian centers.

    Uses high-opacity Gaussian centers as surface points, with normals
    from the smallest eigenvector direction. Colors from SH DC (base color).
    """
    with torch.no_grad():
        mask = model.opacities > opacity_threshold
        positions = model.positions[mask].cpu().numpy()
        normals_arr = model.normals[mask].cpu().numpy()

        sh_dc = model._sh_dc.detach()[mask, 0, :]
        SH_C0 = 0.28209479177387814
        colors = (0.5 + SH_C0 * sh_dc).clamp(0, 1).cpu().numpy()

    return positions, normals_arr, colors


def extract_all_surface_points(
    model: GaussianModel,
    cameras: List[Camera],
    config: MeshExtractionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract surface points from all cameras + Gaussian centers.

    Combines ray-marched level-set points (high precision near surfaces)
    with Gaussian center points (full scene coverage).

    Args:
        model: regularized GaussianModel.
        cameras: all training cameras.
        config: extraction config.

    Returns:
        points: [P, 3] numpy array.
        normals: [P, 3] numpy array.
        colors: [P, 3] numpy array (0-1 range).
    """
    device = config.device
    all_points = []
    all_normals = []
    all_colors = []

    # --- Gaussian center points for full coverage ---
    print("  Extracting Gaussian center points...")
    gc_pts, gc_normals, gc_colors = _extract_gaussian_center_points(
        model, opacity_threshold=0.3,
    )
    all_points.append(gc_pts)
    all_normals.append(gc_normals)
    all_colors.append(gc_colors)
    print(f"  Gaussian centers: {len(gc_pts)} points (opacity > 0.3)")

    # --- Ray-marched level-set points for precision ---
    print("  Extracting ray-marched surface points...")
    for i, cam in enumerate(cameras):
        for level in config.surface_levels:
            pts, norms, cols = extract_surface_points_from_camera(
                model, cam,
                surface_level=level,
                n_ray_samples=config.n_ray_samples,
                range_sigmas=config.ray_range_sigmas,
                device=device,
                chunk_size=config.chunk_size,
            )
            if pts is not None and len(pts) >= config.min_points_per_view:
                all_points.append(pts.cpu().numpy())
                all_normals.append(norms.cpu().numpy())
                all_colors.append(cols.cpu().numpy())

        if (i + 1) % 20 == 0:
            total = sum(len(p) for p in all_points)
            print(f"  Camera {i+1}/{len(cameras)}: {total} points so far")

    points = np.concatenate(all_points, axis=0)
    normals = np.concatenate(all_normals, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    print(f"  Total surface points: {len(points)}")
    return points, normals, colors


def poisson_reconstruct(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    poisson_depth: int = 8,
    density_quantile: float = 0.02,
    decimation_target: int = 200_000,
) -> trimesh.Trimesh:
    """Run Poisson surface reconstruction with quality enhancements.

    Pipeline: outlier removal -> Poisson -> density filter -> decimate
    -> vertex projection -> color transfer.

    Args:
        points: [P, 3] surface points.
        normals: [P, 3] surface normals.
        colors: [P, 3] colors (0-1).
        poisson_depth: octree depth for Poisson.
        density_quantile: remove vertices below this density quantile.
        decimation_target: target number of faces.

    Returns:
        trimesh.Trimesh with vertex colors.
    """
    # --- Statistical outlier removal ---
    print(f"  Removing outliers from {len(points)} points...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd_clean, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=20.0
    )
    print(f"  After outlier removal: {len(pcd_clean.points)} points "
          f"(removed {len(points) - len(pcd_clean.points)})")

    # --- Propagate consistent normal orientation ---
    print("  Propagating consistent normal orientation...")
    pcd_clean.orient_normals_consistent_tangent_plane(k=10)

    # Keep clean arrays for color transfer later
    clean_points = np.asarray(pcd_clean.points)
    clean_colors = np.asarray(pcd_clean.colors)

    # --- Poisson reconstruction ---
    print(f"  Running Poisson reconstruction (depth={poisson_depth})...")
    mesh_o3d, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_clean, depth=poisson_depth, n_threads=-1
        )
    )

    # Remove low-density vertices (gentle trimming)
    densities = np.asarray(densities)
    threshold = np.quantile(densities, density_quantile)
    vertices_to_remove = densities < threshold
    mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

    print(
        f"  Poisson mesh: {len(mesh_o3d.vertices)} vertices, "
        f"{len(mesh_o3d.triangles)} faces"
    )

    # --- Decimate ---
    if len(mesh_o3d.triangles) > decimation_target:
        print(f"  Decimating to {decimation_target} faces...")
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(
            target_number_of_triangles=decimation_target
        )
        print(
            f"  After decimation: {len(mesh_o3d.vertices)} vertices, "
            f"{len(mesh_o3d.triangles)} faces"
        )

    # --- Clean up ---
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_non_manifold_edges()

    # --- Color transfer: assign colors from nearest surface points ---
    mesh_verts = np.asarray(mesh_o3d.vertices)
    if len(clean_points) > 0 and len(mesh_verts) > 0:
        print("  Transferring colors from surface points...")
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(clean_points)
        kdtree = o3d.geometry.KDTreeFlann(source_pcd)

        vertex_colors_arr = np.zeros((len(mesh_verts), 3), dtype=np.float64)
        k_color = 5
        for i in range(len(mesh_verts)):
            _, idx, dist_sq = kdtree.search_knn_vector_3d(
                mesh_verts[i], k_color
            )
            dists = np.sqrt(np.array(dist_sq)).clip(min=1e-8)
            weights = 1.0 / dists
            weights /= weights.sum()
            for j in range(len(idx)):
                vertex_colors_arr[i] += weights[j] * clean_colors[idx[j]]

        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(
            vertex_colors_arr.clip(0, 1)
        )
        print("  Color transfer complete.")

    # --- Convert to trimesh ---
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)

    vertex_colors = None
    if mesh_o3d.has_vertex_colors():
        vc = np.asarray(mesh_o3d.vertex_colors)
        vertex_colors = (vc * 255).clip(0, 255).astype(np.uint8)
        alpha = np.full((len(vertex_colors), 1), 255, dtype=np.uint8)
        vertex_colors = np.hstack([vertex_colors, alpha])

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,
    )

    return mesh
