#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY

"""Extract a mesh from a trained dn-splatter/ags-mesh model using gs-mesh and export as GLB."""

import argparse
import os
import subprocess
import sys
import trimesh


def run_gs_mesh_sugar(config_path, output_dir, poisson_depth=9, total_points=2000000):
    """Run gs-mesh sugar-coarse to extract a mesh using SuGaR-style level-set + Poisson."""
    cmd = [
        "gs-mesh", "sugar-coarse",
        "--load-config", config_path,
        "--output-dir", output_dir,
        "--return-normal", "closest_gaussian",
        "--poisson-depth", str(poisson_depth),
        "--total-points", str(total_points),
    ]
    print(f"Running gs-mesh sugar-coarse: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def run_gs_mesh_tsdf(config_path, output_dir, voxel_size=None, sdf_trunc=None):
    """Run gs-mesh o3dtsdf to extract a mesh via TSDF fusion (fallback method)."""
    cmd = [
        "gs-mesh", "o3dtsdf",
        "--load-config", config_path,
        "--output-dir", output_dir,
    ]
    if voxel_size is not None:
        cmd.extend(["--voxel-size", str(voxel_size)])
    if sdf_trunc is not None:
        cmd.extend(["--sdf-trunc", str(sdf_trunc)])
    print(f"Running gs-mesh o3dtsdf: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def find_mesh_ply(output_dir):
    """Find the mesh PLY file produced by gs-mesh in the output directory."""
    for fname in os.listdir(output_dir):
        if fname.endswith(".ply") and "mesh" in fname.lower():
            return os.path.join(output_dir, fname)
    for fname in os.listdir(output_dir):
        if fname.endswith(".ply") and fname not in (
            "splat.ply", "orig.ply", "sog.ply", "spz.ply", "usdz.ply"
        ):
            return os.path.join(output_dir, fname)
    return None


def convert_ply_to_glb(input_ply, output_glb):
    """Convert a PLY mesh file to GLB format using Open3D + trimesh."""
    import numpy as np
    import open3d as o3d
    print(f"Converting {input_ply} to {output_glb}")
    o3d_mesh = o3d.io.read_triangle_mesh(input_ply)
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    # Remove NaN/Inf vertices — TSDF fusion can produce these at scene boundaries.
    # GLB accessor min/max become NaN and viewers reject the file as invalid JSON.
    valid_mask = np.isfinite(vertices).all(axis=1)
    if not valid_mask.all():
        n_invalid = int((~valid_mask).sum())
        print(f"Removing {n_invalid}/{len(vertices)} non-finite vertices")
        index_map = np.full(len(vertices), -1, dtype=np.int64)
        new_indices = np.arange(valid_mask.sum())
        index_map[valid_mask] = new_indices
        vertices = vertices[valid_mask]
        # Remap faces — drop any face referencing a removed vertex
        remapped = index_map[faces]
        face_valid = (remapped >= 0).all(axis=1)
        faces = remapped[face_valid]

    if len(faces) == 0:
        print("Warning: no valid faces remain after NaN removal, skipping GLB export")
        return

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if o3d_mesh.has_vertex_colors():
        colors_float = np.asarray(o3d_mesh.vertex_colors)
        if not valid_mask.all():
            colors_float = colors_float[valid_mask]
        colors_uint8 = (colors_float * 255).clip(0, 255).astype(np.uint8)
        alpha = np.full((len(colors_uint8), 1), 255, dtype=np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=np.hstack([colors_uint8, alpha])
        )
    scene = trimesh.scene.scene.Scene(geometry={'mesh': mesh})
    scene.export(output_glb, file_type="glb")
    print(f"GLB written: {output_glb} ({os.path.getsize(output_glb)} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract mesh from dn-splatter/ags-mesh model and export as GLB"
    )
    parser.add_argument("--config-path", required=True,
                        help="Path to nerfstudio config.yml from training")
    parser.add_argument("--output-ply", required=True, help="Output PLY mesh path")
    parser.add_argument("--output-glb", required=True, help="Output GLB mesh path")
    parser.add_argument("--method", choices=["sugar-coarse", "o3dtsdf"],
                        default="sugar-coarse",
                        help="Mesh extraction method (default: sugar-coarse)")
    parser.add_argument("--poisson-depth", type=int, default=9,
                        help="Poisson reconstruction depth (sugar-coarse only)")
    parser.add_argument("--total-points", type=int, default=2000000,
                        help="Total points to sample (sugar-coarse only)")
    parser.add_argument("--voxel-size", type=float, default=None,
                        help="TSDF voxel size in meters (o3dtsdf only)")
    parser.add_argument("--sdf-trunc", type=float, default=None,
                        help="TSDF truncation distance in meters (o3dtsdf only)")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_ply)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(args.config_path):
        print(f"Info: config not found, skipping mesh extraction: {args.config_path}")
        sys.exit(0)

    if args.method == "sugar-coarse":
        rc = run_gs_mesh_sugar(
            args.config_path, output_dir,
            poisson_depth=args.poisson_depth,
            total_points=args.total_points,
        )
        if rc != 0:
            print(f"sugar-coarse failed (rc={rc}), falling back to o3dtsdf")
            rc = run_gs_mesh_tsdf(
                args.config_path, output_dir,
                voxel_size=args.voxel_size or 0.004,
                sdf_trunc=args.sdf_trunc or 0.02,
            )
    else:
        rc = run_gs_mesh_tsdf(
            args.config_path, output_dir,
            voxel_size=args.voxel_size,
            sdf_trunc=args.sdf_trunc,
        )

    if rc != 0:
        print(f"gs-mesh failed with return code {rc} — skipping mesh extraction (non-fatal)")
        sys.exit(0)

    mesh_ply = find_mesh_ply(output_dir)
    if not mesh_ply or not os.path.exists(mesh_ply) or os.path.getsize(mesh_ply) == 0:
        print(f"Warning: no mesh PLY found in {output_dir}, skipping GLB conversion")
        sys.exit(0)

    if os.path.abspath(mesh_ply) != os.path.abspath(args.output_ply):
        import shutil
        shutil.move(mesh_ply, args.output_ply)

    convert_ply_to_glb(args.output_ply, args.output_glb)


if __name__ == "__main__":
    main()
