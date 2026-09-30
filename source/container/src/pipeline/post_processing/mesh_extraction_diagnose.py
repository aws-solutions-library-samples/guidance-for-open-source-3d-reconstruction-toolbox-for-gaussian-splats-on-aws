# MIT License — see __init__.py
"""Diagnostic script to verify gsplat rendering works."""
from __future__ import annotations

import torch

from mesh_extraction_cameras import load_cameras_from_transforms
from mesh_extraction_gaussian_model import GaussianModel
from mesh_extraction_regularize import _render_gsplat, _render_depth_gsplat


def diagnose(
    ply_path: str,
    transforms_path: str,
    downscale: int = 2,
) -> None:
    device = "cuda"

    print("Loading model...")
    model = GaussianModel.from_ply(ply_path, device=device)
    model.eval()
    print(f"  {model.n_gaussians} Gaussians")

    print("\nLoading cameras...")
    cameras = load_cameras_from_transforms(transforms_path, downscale=downscale)
    cam = cameras[0]
    print(f"  Camera 0: {cam.width}x{cam.height}")

    print("\n--- Test 1: RGB rendering ---")
    with torch.no_grad():
        rendered, alphas = _render_gsplat(model, cam, device)
    print(f"  rendered range: {rendered.min().item():.4f} to "
          f"{rendered.max().item():.4f}")
    print(f"  alpha mean: {alphas.mean().item():.4f}")

    print("\n--- Test 2: Depth rendering ---")
    with torch.no_grad():
        depth_map = _render_depth_gsplat(model, cam, device)
    print(f"  depth range: {depth_map.min().item():.4f} to "
          f"{depth_map.max().item():.4f}")
    depth_flat = depth_map.reshape(-1)
    n_valid = (depth_flat > 0.01).sum().item()
    print(f"  valid pixels (>0.01): {n_valid} / {depth_flat.numel()}")

    print("\n--- Test 3: Viewmat z-depth check ---")
    viewmat = cam.viewmat(device).unsqueeze(0)
    pos_h = torch.cat(
        [model.positions, torch.ones(model.n_gaussians, 1, device=device)],
        dim=-1,
    )
    cam_pos = (viewmat[0] @ pos_h.T).T
    z_vals = cam_pos[:, 2]
    print(f"  cam-space z: min={z_vals.min().item():.4f}, "
          f"max={z_vals.max().item():.4f}")
    print(f"  positive z: {(z_vals > 0).sum().item()}, "
          f"negative z: {(z_vals < 0).sum().item()}")

    if n_valid > 0:
        print("\n--- PASS: Rendering produces visible output ---")
    else:
        print("\n--- FAIL: No visible output ---")


if __name__ == "__main__":
    import sys
    diagnose(
        ply_path=sys.argv[1],
        transforms_path=sys.argv[2],
        downscale=int(sys.argv[3]) if len(sys.argv) > 3 else 2,
    )
