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

"""Extract a mesh from a trained dn-splatter/ags-mesh model using gs-mesh o3dtsdf and export as GLB."""

import argparse
import os
import subprocess
import sys
import trimesh


def run_gs_mesh(config_path, output_dir):
    """Run gs-mesh o3dtsdf to extract a mesh from the trained model checkpoint."""
    cmd = [
        "gs-mesh", "o3dtsdf",
        "--load-config", config_path,
        "--output-dir", output_dir,
    ]
    print(f"Running gs-mesh: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def find_mesh_ply(output_dir):
    """Find the mesh PLY file produced by gs-mesh in the output directory."""
    for fname in os.listdir(output_dir):
        if fname.endswith(".ply") and "mesh" in fname.lower():
            return os.path.join(output_dir, fname)
    # Fallback: any new PLY that isn't splat.ply or orig.ply
    for fname in os.listdir(output_dir):
        if fname.endswith(".ply") and fname not in ("splat.ply", "orig.ply", "sog.ply", "spz.ply", "usdz.ply"):
            return os.path.join(output_dir, fname)
    return None


def convert_ply_to_glb(input_ply, output_glb):
    """Convert a PLY mesh file to GLB format using Open3D + trimesh.
    Open3D preserves vertex colors from the TSDF fusion; trimesh exports to GLB."""
    import numpy as np
    import open3d as o3d
    print(f"Converting {input_ply} to {output_glb}")
    o3d_mesh = o3d.io.read_triangle_mesh(input_ply)
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if o3d_mesh.has_vertex_colors():
        colors_float = np.asarray(o3d_mesh.vertex_colors)  # float64 [0,1]
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
    parser = argparse.ArgumentParser(description="Extract mesh from dn-splatter model using gs-mesh o3dtsdf and export as GLB")
    parser.add_argument("--config-path", required=True, help="Path to nerfstudio config.yml from training")
    parser.add_argument("--output-ply", required=True, help="Output PLY mesh path")
    parser.add_argument("--output-glb", required=True, help="Output GLB mesh path")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_ply)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(args.config_path):
        print(f"Info: config not found, skipping mesh extraction: {args.config_path}")
        sys.exit(0)

    rc = run_gs_mesh(args.config_path, output_dir)
    if rc != 0:
        print(f"gs-mesh failed with return code {rc} — skipping mesh extraction (non-fatal)")
        sys.exit(0)

    mesh_ply = find_mesh_ply(output_dir)
    if not mesh_ply or not os.path.exists(mesh_ply) or os.path.getsize(mesh_ply) == 0:
        print(f"Warning: no mesh PLY found in {output_dir}, skipping GLB conversion")
        sys.exit(0)

    # Rename to expected output path if different
    if os.path.abspath(mesh_ply) != os.path.abspath(args.output_ply):
        import shutil
        shutil.move(mesh_ply, args.output_ply)

    convert_ply_to_glb(args.output_ply, args.output_glb)


if __name__ == "__main__":
    main()
