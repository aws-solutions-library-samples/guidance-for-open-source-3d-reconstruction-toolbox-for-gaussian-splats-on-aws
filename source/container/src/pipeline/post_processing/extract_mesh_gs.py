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

"""TSDF mesh extraction from Gaussian splats with texture baking.

Extracts a textured mesh (OBJ + GLB) from a trained Gaussian splat model
using TSDF fusion, PyMeshFix repair, xatlas UV unwrapping, and camera-
projection texture baking with configurable color grading.
Auto-detects scene vs object mode.
"""
import argparse
import os
import sys
import struct
import json as json_mod
import gc
from collections import defaultdict
from io import BytesIO

import numpy as np
import torch
import open3d as o3d
import open3d.core as o3c
import trimesh
import xatlas
from PIL import Image

# Container pipeline path for helper imports
_PIPELINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from post_processing.mesh_extraction_gaussian_model import GaussianModel
from post_processing.mesh_extraction_cameras import load_cameras_from_transforms
from post_processing.mesh_extraction_regularize import _render_gsplat, _render_depth_gsplat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract textured mesh from Gaussian splat model."
    )
    p.add_argument("--ply", required=True, help="Path to splat.ply")
    p.add_argument("--transforms", required=True, help="Path to transforms.json")
    p.add_argument("--output-dir", required=True,
                   help="Directory for output files (OBJ, GLB, texture PNG)")
    p.add_argument("--name", default="gs_mesh", help="Output filename stem")
    p.add_argument("--device", default="cuda", help="Torch device")
    p.add_argument("--no-applied-transform", action="store_true",
                   help="Skip applying applied_transform from transforms.json")

    g = p.add_argument_group("geometry")
    g.add_argument("--voxel-div", type=int, default=512)
    g.add_argument("--target-faces", type=int, default=None)
    g.add_argument("--smooth-iterations", type=int, default=10)
    g.add_argument("--smooth-lambda", type=float, default=0.5)
    g.add_argument("--solidify-offset", type=float, default=0.005)

    g = p.add_argument_group("texture")
    g.add_argument("--texture-size", type=int, default=4096)
    g.add_argument("--bg-color-thresh", type=int, default=25)

    g = p.add_argument_group("color grading")
    g.add_argument("--contrast", type=float, default=1.5)
    g.add_argument("--saturation", type=float, default=0.6)
    g.add_argument("--brightness", type=float, default=0.50)

    return p.parse_args()


# Globals set from CLI args
DEVICE: str = "cuda"
PLY: str = ""
TRANSFORMS: str = ""
OUT_OBJ: str = ""
OUT_GLB: str = ""
TEX_PATH: str = ""
VOXEL_DIV: int = 512
TARGET_FACES: int = 100_000
TEXTURE_SIZE: int = 4096
BG_COLOR_THRESH: int = 25
SMOOTH_ITERATIONS: int = 10
SMOOTH_LAMBDA: float = 0.5
SOLIDIFY_OFFSET: float = 0.005
CONTRAST: float = 1.5
SAT_BOOST: float = 0.6
BRIGHTNESS: float = 0.50
SCENE_MODE = None


def pr(msg):
    print(msg, flush=True)


def detect_bg_color(model, cams, device, n_sample=5):
    corner_colors = []
    with torch.no_grad():
        for i in range(0, len(cams), max(1, len(cams) // n_sample)):
            rgb, _ = _render_gsplat(model, cams[i], device=device)
            img = rgb[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            h, w = img.shape[:2]
            for r, c in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
                corner_colors.append(img[r, c])
    bg = np.median(corner_colors, axis=0)
    pr(f"Background: RGB({bg[0]:.3f}, {bg[1]:.3f}, {bg[2]:.3f})")
    return bg


def auto_detect_scene_mode(model, cams, bg_color, device, n_sample=8,
                            bg_thresh_frac=25.0 / 255.0, bg_pixel_thresh=0.40):
    coverages = []
    bg_fracs = []
    with torch.no_grad():
        for i in range(0, len(cams), max(1, len(cams) // n_sample)):
            cam = cams[i]
            rgb, alpha = _render_gsplat(model, cam, device=device)
            alpha_np = (alpha[0, 0].cpu().numpy() if alpha.dim() == 4
                        else alpha.cpu().numpy())
            coverages.append(float((alpha_np > 0.5).mean()))
            img = rgb[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            alpha_safe = alpha_np.clip(min=0.1)
            img_unpremult = (img / alpha_safe[:, :, None]).clip(0, 1)
            bg_diff = np.abs(img_unpremult - bg_color).max(axis=2)
            bg_fracs.append(float((bg_diff < bg_thresh_frac).mean()))

    med_cov = float(np.median(coverages))
    med_bg_frac = float(np.median(bg_fracs))
    pr(f"Auto-detect: coverage={med_cov*100:.1f}%, bg-frac={med_bg_frac*100:.1f}%")
    if med_cov < 0.85:
        pr("  -> Object (transparent background)")
        return False
    if med_bg_frac > bg_pixel_thresh:
        pr(f"  -> Object (opaque background, {med_bg_frac*100:.0f}% bg pixels)")
        return False
    pr("  -> Scene (high coverage, no distinct background)")
    return True


def estimate_object_extent(model, cams, bg_color, device, n_sample=8):
    pts_3d = []
    with torch.no_grad():
        for i in range(0, len(cams), max(1, len(cams) // n_sample)):
            cam = cams[i]
            depth = _render_depth_gsplat(model, cam, device=device)
            _, alpha = _render_gsplat(model, cam, device=device)
            alpha_np = (alpha[0, 0].cpu().numpy() if alpha.dim() == 4
                        else alpha.cpu().numpy())
            depth_np = depth.cpu().numpy()
            mask = (alpha_np > 0.5) & (depth_np > 0)
            if mask.sum() == 0:
                continue
            vs, us = np.where(mask)
            zs = depth_np[vs, us]
            xs = (us - cam.cx) / cam.fx * zs
            ys = (vs - cam.cy) / cam.fy * zs
            pts_cam = np.stack([xs, ys, zs], axis=1)
            c2w = cam.c2w_opencv
            pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]
            pts_3d.append(pts_world[::10])
    all_pts = np.concatenate(pts_3d)
    lo = np.percentile(all_pts, 2, axis=0)
    hi = np.percentile(all_pts, 98, axis=0)
    return float((hi - lo).max())


def estimate_scene_extent(model, cams, device, n_sample=12):
    pts_3d = []
    all_depths = []
    with torch.no_grad():
        for i in range(0, len(cams), max(1, len(cams) // n_sample)):
            cam = cams[i]
            depth = _render_depth_gsplat(model, cam, device=device)
            depth_np = depth.cpu().numpy()
            mask = depth_np > 0
            if mask.sum() == 0:
                continue
            all_depths.append(depth_np[mask][::10])
            vs, us = np.where(mask)
            zs = depth_np[vs, us]
            xs = (us - cam.cx) / cam.fx * zs
            ys = (vs - cam.cy) / cam.fy * zs
            pts_cam = np.stack([xs, ys, zs], axis=1)
            c2w = cam.c2w_opencv
            pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]
            pts_3d.append(pts_world[::10])
    all_pts = np.concatenate(pts_3d)
    lo = np.percentile(all_pts, 2, axis=0)
    hi = np.percentile(all_pts, 98, axis=0)
    full_extent = float((hi - lo).max())
    med_depth = float(np.median(np.concatenate(all_depths)))
    detail_extent = min(full_extent, med_depth * 8)
    pr(f"  Median depth: {med_depth:.2f}, full: {full_extent:.2f}, detail: {detail_extent:.2f}")
    return full_extent, detail_extent


def solidify_mesh(verts, faces, offset, vertex_normals=None):
    n_v = len(verts)
    n_f = len(faces)
    if vertex_normals is not None:
        back_verts = verts - vertex_normals * offset
    else:
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(fn, axis=1, keepdims=True).clip(min=1e-10)
        front_dir = (fn * areas).sum(axis=0)
        front_dir /= np.linalg.norm(front_dir) + 1e-10
        back_verts = verts - front_dir * offset
    back_faces = faces[:, ::-1] + n_v
    side_faces = []
    if vertex_normals is None:
        edge_count = defaultdict(list)
        for fi, face in enumerate(faces):
            for i in range(3):
                e = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_count[e].append((face[i], face[(i + 1) % 3]))
        for e, info in edge_count.items():
            if len(info) == 1:
                a, b = info[0]
                side_faces.append([a, b, b + n_v])
                side_faces.append([a, b + n_v, a + n_v])
    all_verts = np.vstack([verts, back_verts])
    parts = [faces, back_faces]
    if side_faces:
        parts.append(np.array(side_faces, dtype=faces.dtype))
    all_faces = np.vstack(parts)
    pr(f"  Solidified: {n_v}V->{len(all_verts)}V, {n_f}F->{len(all_faces)}F")
    return all_verts, all_faces


def dilate_texture(texture, valid_mask, iterations=8):
    filled = valid_mask.copy()
    for _ in range(iterations):
        expanded = np.zeros_like(filled)
        expanded[1:] |= filled[:-1]
        expanded[:-1] |= filled[1:]
        expanded[:, 1:] |= filled[:, :-1]
        expanded[:, :-1] |= filled[:, 1:]
        new_pixels = expanded & ~filled
        if not new_pixels.any():
            break
        ny, nx = np.where(new_pixels)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            sy = np.clip(ny + dy, 0, texture.shape[0] - 1)
            sx = np.clip(nx + dx, 0, texture.shape[1] - 1)
            mask = filled[sy, sx]
            texture[ny[mask], nx[mask]] = texture[sy[mask], sx[mask]]
        filled |= new_pixels
    return texture


def export_glb(path, verts, faces, uvs, normals, texture_img):
    buf = BytesIO()
    texture_img.save(buf, format="PNG")
    tex_png = buf.getvalue()
    pos_bytes = verts.astype(np.float32).tobytes()
    nrm_bytes = normals.astype(np.float32).tobytes()
    uv_bytes = uvs.astype(np.float32).tobytes()
    idx_bytes = faces.astype(np.uint32).tobytes()

    def align4(n):
        return n + (4 - n % 4) % 4

    bv_lengths = [len(pos_bytes), len(nrm_bytes), len(uv_bytes),
                  len(idx_bytes), len(tex_png)]
    offsets = []
    cur = 0
    for length in bv_lengths:
        offsets.append(cur)
        cur = align4(cur + length)
    total_bin = cur
    n_v = len(verts)
    pos_min = verts.astype(np.float32).min(axis=0).tolist()
    pos_max = verts.astype(np.float32).max(axis=0).tolist()
    gltf = {
        "asset": {"version": "2.0", "generator": "extract_mesh_gs"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {
            "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2,
        }, "indices": 3, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicFactor": 0.0, "roughnessFactor": 1.0,
        }, "doubleSided": True}],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987}],
        "images": [{"bufferView": 4, "mimeType": "image/png"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n_v,
             "type": "VEC3", "min": pos_min, "max": pos_max},
            {"bufferView": 1, "componentType": 5126, "count": n_v, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": n_v, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5125, "count": faces.size, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": bv_lengths[0], "target": 34962},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": bv_lengths[1], "target": 34962},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": bv_lengths[2], "target": 34962},
            {"buffer": 0, "byteOffset": offsets[3], "byteLength": bv_lengths[3], "target": 34963},
            {"buffer": 0, "byteOffset": offsets[4], "byteLength": bv_lengths[4]},
        ],
        "buffers": [{"byteLength": total_bin}],
    }
    json_str = json_mod.dumps(gltf, separators=(",", ":"))
    while len(json_str) % 4 != 0:
        json_str += " "
    json_bytes = json_str.encode("utf-8")
    bin_data = bytearray(total_bin)
    for i, data in enumerate([pos_bytes, nrm_bytes, uv_bytes, idx_bytes, tex_png]):
        bin_data[offsets[i]: offsets[i] + len(data)] = data
    glb_length = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, glb_length))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_data), 0x004E4942))
        f.write(bytes(bin_data))
    pr(f"GLB written: {glb_length} bytes, {n_v}V {len(faces)}F")


def main():
    global TRANSFORMS, SCENE_MODE
    pr(f"[ENV] open3d={o3d.__version__} torch={torch.__version__} "
       f"numpy={np.__version__}")
    m = GaussianModel.from_ply(PLY, device=DEVICE)
    m.eval()

    # Auto-detect whether applied_transform should be used
    skip_at = getattr(parse_args, '_skip_at', False)
    if not skip_at:
        with open(TRANSFORMS) as _f:
            _tdata = json_mod.load(_f)
        _has_at = "applied_transform" in _tdata
        if _has_at:
            import tempfile as _tmpmod
            _tdata_no_at = {k: v for k, v in _tdata.items() if k != "applied_transform"}
            _tmp = _tmpmod.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json_mod.dump(_tdata_no_at, _tmp)
            _tmp.close()
            cams_with = load_cameras_from_transforms(TRANSFORMS, downscale=4)
            cams_without = load_cameras_from_transforms(_tmp.name, downscale=4)
            os.unlink(_tmp.name)
            n_test = min(8, len(cams_with))
            test_idx = [int(i * len(cams_with) / n_test) for i in range(n_test)]
            covs_w, covs_wo = [], []
            with torch.no_grad():
                for ti in test_idx:
                    _, a_w = _render_gsplat(m, cams_with[ti], device=DEVICE)
                    _, a_wo = _render_gsplat(m, cams_without[ti], device=DEVICE)
                    covs_w.append((a_w.cpu().numpy().flatten() > 0.5).mean())
                    covs_wo.append((a_wo.cpu().numpy().flatten() > 0.5).mean())
            med_w = float(np.median(covs_w))
            med_wo = float(np.median(covs_wo))
            pr(f"Auto-detect applied_transform: with={med_w*100:.1f}%, without={med_wo*100:.1f}%")
            if med_wo > med_w * 1.5:
                pr("  -> Skipping applied_transform")
                skip_at = True
            else:
                pr("  -> Applying applied_transform")
            del cams_with, cams_without
            torch.cuda.empty_cache()
        else:
            pr("No applied_transform in transforms.json")

    if skip_at:
        with open(TRANSFORMS) as _f:
            _tdata = json_mod.load(_f)
        if "applied_transform" in _tdata:
            import tempfile as _tmpmod
            _tdata.pop("applied_transform")
            _tmp = _tmpmod.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json_mod.dump(_tdata, _tmp)
            _tmp.close()
            TRANSFORMS = _tmp.name

    cams_half = load_cameras_from_transforms(TRANSFORMS, downscale=2)
    cams_full = load_cameras_from_transforms(TRANSFORMS, downscale=1)
    bg_color = detect_bg_color(m, cams_half, DEVICE)
    bg_thresh = BG_COLOR_THRESH / 255.0

    if SCENE_MODE is None:
        SCENE_MODE = auto_detect_scene_mode(m, cams_half, bg_color, DEVICE,
                                             bg_thresh_frac=bg_thresh)
    else:
        pr(f"Mode: {'scene' if SCENE_MODE else 'object'} (manual override)")

    # Filter cameras
    if SCENE_MODE:
        pr("Scene mode: using all cameras")
        opaque_bg = False
        good_indices = list(range(len(cams_half)))
    else:
        pr("Filtering cameras by object coverage...")
        MIN_COVERAGE, MAX_COVERAGE = 0.02, 0.85
        coverages, good_indices = [], []
        with torch.no_grad():
            for i, cam in enumerate(cams_half):
                _, alpha = _render_gsplat(m, cam, device=DEVICE)
                alpha_np = (alpha[0, 0].cpu().numpy() if alpha.dim() == 4
                            else alpha.cpu().numpy())
                coverage = (alpha_np > 0.5).mean()
                coverages.append(coverage)
                if MIN_COVERAGE <= coverage <= MAX_COVERAGE:
                    good_indices.append(i)
                if (i + 1) % 50 == 0:
                    torch.cuda.empty_cache()
        median_cov = float(np.median(coverages))
        opaque_bg = median_cov > 0.85
        if opaque_bg:
            pr(f"  Opaque background (median {median_cov*100:.0f}%) — using all cameras")
            good_indices = list(range(len(cams_half)))
        else:
            pr(f"  {len(good_indices)}/{len(cams_half)} cameras passed filter")
            if len(good_indices) == 0:
                pr("  WARNING: No cameras passed filter, using all")
                good_indices = list(range(len(cams_half)))

    cams_half = [cams_half[i] for i in good_indices]
    cams_full = [cams_full[i] for i in good_indices]

    if SCENE_MODE:
        pr("Estimating scene extent...")
        full_extent, detail_extent = estimate_scene_extent(m, cams_half, DEVICE)
        obj_extent = full_extent
        voxel_extent = detail_extent
    else:
        pr("Estimating object extent...")
        obj_extent = estimate_object_extent(m, cams_half, bg_color, DEVICE)
        voxel_extent = obj_extent
    pr(f"Extent: {obj_extent:.4f} (voxel: {voxel_extent:.4f})")

    voxel_length = voxel_extent / VOXEL_DIV
    sdf_trunc = voxel_length * 4
    pr(f"Voxel: {voxel_length:.5f}, trunc: {sdf_trunc:.5f}")

    # TSDF integration using VoxelBlockGrid (tensor API).
    # ScalableTSDFVolume is broken in Open3D >= 0.20.0.
    _o3d_dev = o3c.Device('CPU:0')
    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=voxel_length,
        block_resolution=16,
        block_count=100000,
        device=_o3d_dev,
    )
    cam0 = cams_half[0]
    _K_np = np.array([[cam0.fx, 0, cam0.cx],
                      [0, cam0.fy, cam0.cy],
                      [0, 0, 1.0]], dtype=np.float64)
    _intrinsic_t = o3c.Tensor(_K_np)
    _depth_trunc = float(voxel_extent if SCENE_MODE else obj_extent * 3)

    pr(f"Integrating {len(cams_half)} cameras into TSDF...")
    with torch.no_grad():
        for i, cam in enumerate(cams_half):
            depth = _render_depth_gsplat(m, cam, device=DEVICE)
            rgb, alpha = _render_gsplat(m, cam, device=DEVICE)
            rgb_np = rgb[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            alpha_np = (alpha[0, 0].cpu().numpy() if alpha.dim() == 4
                        else alpha.cpu().numpy())
            alpha_safe = alpha_np.clip(min=0.1)
            rgb_unpremult = (rgb_np / alpha_safe[:, :, None]).clip(0, 1)
            if SCENE_MODE:
                depth[torch.from_numpy(alpha_np < 0.3).to(depth.device)] = 0.0
                depth[depth > voxel_extent] = 0.0
            elif opaque_bg:
                bg_diff = np.abs(rgb_unpremult - bg_color).max(axis=2)
                bg_mask = (bg_diff < bg_thresh) | (alpha_np < 0.1)
                depth[torch.from_numpy(bg_mask).to(depth.device)] = 0.0
            else:
                depth[torch.from_numpy(alpha_np < 0.5).to(depth.device)] = 0.0
            depth_np = depth.cpu().numpy().astype(np.float32)
            color_f = rgb_unpremult.astype(np.float32)
            depth_t = o3d.t.geometry.Image(o3c.Tensor(depth_np))
            color_t = o3d.t.geometry.Image(o3c.Tensor(color_f))
            w2c = np.ascontiguousarray(
                np.linalg.inv(cam.c2w_opencv), dtype=np.float64)
            extrinsic_t = o3c.Tensor(w2c)
            try:
                frustum_coords = volume.compute_unique_block_coordinates(
                    depth_t, _intrinsic_t, extrinsic_t, 1.0, _depth_trunc)
                volume.integrate(frustum_coords, depth_t, color_t,
                                 _intrinsic_t, extrinsic_t, 1.0, _depth_trunc)
            except RuntimeError:
                pass
            if (i + 1) % 40 == 0:
                pr(f"  {i + 1}/{len(cams_half)}")
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

    pr("Extracting mesh from TSDF...")
    mesh_t = volume.extract_triangle_mesh()
    mesh = mesh_t.to_legacy()
    del volume
    torch.cuda.empty_cache()
    gc.collect()
    mesh.compute_vertex_normals()
    pr(f"TSDF raw: {len(mesh.vertices)}V {len(mesh.triangles)}F")

    # Cleanup connected components
    if len(mesh.triangles) > 0:
        tri_cl, cl_sizes, _ = mesh.cluster_connected_triangles()
        tri_cl = np.asarray(tri_cl)
        cl_sizes = np.asarray(cl_sizes)
        if len(cl_sizes) > 1:
            if SCENE_MODE:
                min_cl = max(100, int(0.01 * len(mesh.triangles)))
                small = np.array([cl_sizes[c] < min_cl for c in tri_cl])
                mesh.remove_triangles_by_mask(small)
                mesh.remove_unreferenced_vertices()
                pr(f"Removed small components (< {min_cl} tris)")
            else:
                largest = cl_sizes.argmax()
                mesh.remove_triangles_by_mask(tri_cl != largest)
                mesh.remove_unreferenced_vertices()
                pr(f"Kept largest component")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    pr(f"Taubin smoothing ({SMOOTH_ITERATIONS} iters)...")
    mesh = mesh.filter_smooth_taubin(
        number_of_iterations=SMOOTH_ITERATIONS,
        lambda_filter=SMOOTH_LAMBDA,
        mu=-SMOOTH_LAMBDA - 0.01,
    )
    mesh.compute_vertex_normals()
    pr(f"After smoothing: {len(mesh.vertices)}V {len(mesh.triangles)}F")

    complex_geometry = len(mesh.triangles) > 1_000_000
    pre_target = TARGET_FACES * 3 if (complex_geometry and not SCENE_MODE) else TARGET_FACES
    if len(mesh.triangles) > pre_target:
        gc.collect()
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=pre_target)
        mesh.compute_vertex_normals()

    vert_colors = np.asarray(mesh.vertex_colors).astype(np.float64)
    has_vc = len(vert_colors) == len(mesh.vertices) and vert_colors.max() > 0
    if has_vc:
        vert_colors = np.power(vert_colors.clip(min=1e-4), 0.55).clip(0, 1)

    o3d_verts = np.asarray(mesh.vertices)
    o3d_faces = np.asarray(mesh.triangles)

    if SCENE_MODE:
        pr("Scene mode: skipping hole fill and MeshFix")
        front_verts = o3d_verts.astype(np.float64)
        front_faces = o3d_faces.astype(np.int64)
    else:
        n_orig_v = len(o3d_verts)
        tm_temp = trimesh.Trimesh(vertices=o3d_verts, faces=o3d_faces, process=False)
        n_before = len(tm_temp.faces)
        for _hf_pass in range(5):
            prev = len(tm_temp.faces)
            trimesh.repair.fill_holes(tm_temp)
            if len(tm_temp.faces) == prev:
                break
        pr(f"Hole fill: {n_before}F -> {len(tm_temp.faces)}F")
        if len(tm_temp.vertices) > n_orig_v and has_vc:
            n_new = len(tm_temp.vertices) - n_orig_v
            new_colors = np.full((n_new, 3), vert_colors.mean(axis=0))
            for vi in range(n_orig_v, len(tm_temp.vertices)):
                adj = np.any(tm_temp.faces == vi, axis=1)
                nbrs = set(tm_temp.faces[adj].ravel()) - {vi}
                orig_nbrs = [v for v in nbrs if v < n_orig_v]
                if orig_nbrs:
                    new_colors[vi - n_orig_v] = vert_colors[orig_nbrs].mean(axis=0)
            vert_colors = np.vstack([vert_colors, new_colors])
        front_verts = tm_temp.vertices.astype(np.float64)
        front_faces = tm_temp.faces.astype(np.int64)
        import pymeshfix
        pr("Running MeshFix...")
        mfix = pymeshfix.MeshFix(front_verts.astype(np.float32), front_faces.astype(np.int32))
        try:
            mfix.repair(verbose=False)
        except TypeError:
            mfix.repair()
        fixed_verts = np.array(mfix.v, dtype=np.float64)
        fixed_faces = np.array(mfix.f, dtype=np.int64)
        pr(f"MeshFix: {len(front_verts)}V/{len(front_faces)}F -> {len(fixed_verts)}V/{len(fixed_faces)}F")
        if has_vc and len(fixed_verts) > 0:
            from scipy.spatial import cKDTree
            tree = cKDTree(front_verts)
            _, idx = tree.query(fixed_verts, k=1)
            vert_colors = vert_colors[np.clip(idx, 0, len(vert_colors) - 1)]
        front_verts = fixed_verts
        front_faces = fixed_faces
        if complex_geometry and len(front_faces) > TARGET_FACES:
            pr(f"Decimating to {TARGET_FACES}F...")
            post_mesh = o3d.geometry.TriangleMesh()
            post_mesh.vertices = o3d.utility.Vector3dVector(front_verts)
            post_mesh.triangles = o3d.utility.Vector3iVector(front_faces.astype(np.int32))
            post_mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
            post_mesh = post_mesh.simplify_quadric_decimation(target_number_of_triangles=TARGET_FACES)
            post_mesh.compute_vertex_normals()
            front_verts = np.asarray(post_mesh.vertices).astype(np.float64)
            front_faces = np.asarray(post_mesh.triangles).astype(np.int64)
            vert_colors = np.asarray(post_mesh.vertex_colors).astype(np.float64)

    pr(f"Front mesh: {len(front_verts)}V {len(front_faces)}F")

    # Solidify
    if SCENE_MODE:
        pr("Scene mode: skipping solidify")
        solid_verts = front_verts
        solid_faces = front_faces
        solid_colors = vert_colors
    else:
        pr("Solidifying mesh...")
        solid_verts, solid_faces = solidify_mesh(front_verts, front_faces, SOLIDIFY_OFFSET)
        solid_colors = np.vstack([vert_colors, vert_colors])

    o3d_solid = o3d.geometry.TriangleMesh()
    o3d_solid.vertices = o3d.utility.Vector3dVector(solid_verts)
    o3d_solid.triangles = o3d.utility.Vector3iVector(solid_faces.astype(np.int32))
    o3d_solid.compute_vertex_normals()
    solid_normals = np.asarray(o3d_solid.vertex_normals).astype(np.float64)
    nrm_len = np.linalg.norm(solid_normals, axis=1, keepdims=True)
    zero_nrm = (nrm_len < 1e-8).ravel()
    if zero_nrm.any():
        solid_normals[zero_nrm] = [0.0, 0.0, 1.0]
        nrm_len[zero_nrm] = 1.0
    solid_normals /= nrm_len

    if SCENE_MODE:
        pr("Cleaning mesh topology for xatlas...")
        tm_clean = trimesh.Trimesh(vertices=solid_verts, faces=solid_faces, process=True)
        trimesh.repair.fix_normals(tm_clean)
        solid_verts = tm_clean.vertices.astype(np.float64)
        solid_faces = tm_clean.faces.astype(np.int64)
        from scipy.spatial import cKDTree as _cKD
        _tree = _cKD(np.asarray(o3d_solid.vertices))
        _, _idx = _tree.query(solid_verts, k=1)
        solid_colors = solid_colors[np.clip(_idx, 0, len(solid_colors) - 1)]
        o3d_solid2 = o3d.geometry.TriangleMesh()
        o3d_solid2.vertices = o3d.utility.Vector3dVector(solid_verts)
        o3d_solid2.triangles = o3d.utility.Vector3iVector(solid_faces.astype(np.int32))
        o3d_solid2.compute_vertex_normals()
        solid_normals = np.asarray(o3d_solid2.vertex_normals).astype(np.float64)
        nrm_len = np.linalg.norm(solid_normals, axis=1, keepdims=True)
        zero_nrm = (nrm_len < 1e-8).ravel()
        if zero_nrm.any():
            solid_normals[zero_nrm] = [0.0, 0.0, 1.0]
            nrm_len[zero_nrm] = 1.0
        solid_normals /= nrm_len
        pr(f"After cleanup: {len(solid_verts)}V {len(solid_faces)}F")

    # xatlas UV unwrap
    pr(f"Running xatlas on {len(solid_faces)} faces...")
    atlas = xatlas.Atlas()
    atlas.add_mesh(solid_verts.astype(np.float32), solid_faces.astype(np.uint32))
    atlas.generate()
    vmapping, new_faces, uvs = atlas[0]
    pr(f"xatlas: {len(vmapping)}V, {len(new_faces)}F")
    new_verts = solid_verts[vmapping]
    new_normals = solid_normals[vmapping]
    new_colors = solid_colors[vmapping]

    # Rasterize UV -> texel positions/normals/colors
    TEX = TEXTURE_SIZE
    pr(f"Rasterizing UV into {TEX}x{TEX}...")
    texel_pos = np.zeros((TEX, TEX, 3), dtype=np.float64)
    texel_nrm = np.zeros((TEX, TEX, 3), dtype=np.float64)
    texel_vc = np.zeros((TEX, TEX, 3), dtype=np.float64)
    texel_valid = np.zeros((TEX, TEX), dtype=bool)
    for start in range(0, len(new_faces), 5000):
        end = min(start + 5000, len(new_faces))
        for fi in range(end - start):
            f = new_faces[start + fi]
            uv_tri = uvs[f]
            pos_tri = new_verts[f]
            nrm_tri = new_normals[f]
            col_tri = new_colors[f]
            tx = uv_tri[:, 0] * (TEX - 1)
            ty = (1.0 - uv_tri[:, 1]) * (TEX - 1)
            tx_min = max(0, int(np.floor(tx.min())))
            tx_max = min(TEX - 1, int(np.ceil(tx.max())))
            ty_min = max(0, int(np.floor(ty.min())))
            ty_max = min(TEX - 1, int(np.ceil(ty.max())))
            if tx_max <= tx_min or ty_max <= ty_min:
                continue
            pxs = np.arange(tx_min, tx_max + 1)
            pys = np.arange(ty_min, ty_max + 1)
            gx, gy = np.meshgrid(pxs, pys)
            pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
            v0 = np.array([tx[0], ty[0]])
            e1 = np.array([tx[1] - tx[0], ty[1] - ty[0]])
            e2 = np.array([tx[2] - tx[0], ty[2] - ty[0]])
            d00, d01, d11 = np.dot(e1, e1), np.dot(e1, e2), np.dot(e2, e2)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-10:
                continue
            dp = pts - v0
            d20, d21 = dp @ e1, dp @ e2
            bv = (d11 * d20 - d01 * d21) / denom
            bw = (d00 * d21 - d01 * d20) / denom
            bu = 1.0 - bv - bw
            inside = (bu >= -0.001) & (bv >= -0.001) & (bw >= -0.001)
            if not inside.any():
                continue
            good = pts[inside].astype(int)
            bary = np.stack([bu[inside], bv[inside], bw[inside]], axis=1)
            texel_pos[good[:, 1], good[:, 0]] = bary @ pos_tri
            n3d = bary @ nrm_tri
            n3d /= np.linalg.norm(n3d, axis=1, keepdims=True).clip(min=1e-10)
            texel_nrm[good[:, 1], good[:, 0]] = n3d
            texel_vc[good[:, 1], good[:, 0]] = (bary @ col_tri).clip(0, 1)
            texel_valid[good[:, 1], good[:, 0]] = True
    pr(f"Valid texels: {texel_valid.sum()}")

    # Camera-based texture baking
    pr("Baking texture from camera projections...")
    valid_ys, valid_xs = np.where(texel_valid)
    n_texels = len(valid_ys)
    t_pos = texel_pos[valid_ys, valid_xs].astype(np.float32)
    t_nrm = texel_nrm[valid_ys, valid_xs].astype(np.float32)
    t_vc = texel_vc[valid_ys, valid_xs].astype(np.float32)
    t_cam_color = np.zeros((n_texels, 3), dtype=np.float32)
    t_cam_weight = np.zeros(n_texels, dtype=np.float32)
    depth_tol = obj_extent * 0.03
    CHUNK = 2_000_000
    gc.collect()
    pr(f"Projecting {n_texels} texels into {len(cams_full)} cameras...")
    with torch.no_grad():
        for ci, cam in enumerate(cams_full):
            rgb_t, alpha_t = _render_gsplat(m, cam, device=DEVICE)
            depth_t = _render_depth_gsplat(m, cam, device=DEVICE)
            rgb_np = rgb_t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            alpha_np = (alpha_t[0, 0].cpu().numpy() if alpha_t.dim() == 4
                        else alpha_t.cpu().numpy())
            depth_np = depth_t.cpu().numpy()
            del rgb_t, alpha_t, depth_t
            alpha_safe = alpha_np.clip(min=0.1)
            rgb_unpremult = (rgb_np / alpha_safe[:, :, None]).clip(0, 1)
            if SCENE_MODE:
                fg_mask = alpha_np > 0.3
            elif opaque_bg:
                bg_diff = np.abs(rgb_unpremult - bg_color).max(axis=2)
                fg_mask = (bg_diff > bg_thresh) & (alpha_np > 0.5)
            else:
                fg_mask = alpha_np > 0.5
            w2c = np.linalg.inv(cam.c2w_opencv).astype(np.float32)
            rot, tra = w2c[:3, :3], w2c[:3, 3]
            z = np.empty(n_texels, dtype=np.float32)
            u_i = np.empty(n_texels, dtype=np.int32)
            v_i = np.empty(n_texels, dtype=np.int32)
            for cs in range(0, n_texels, CHUNK):
                ce = min(cs + CHUNK, n_texels)
                vc = (rot @ t_pos[cs:ce].T).T + tra
                z[cs:ce] = vc[:, 2]
                zc = z[cs:ce].clip(min=1e-8)
                u_i[cs:ce] = np.round(vc[:, 0] / zc * cam.fx + cam.cx).astype(np.int32)
                v_i[cs:ce] = np.round(vc[:, 1] / zc * cam.fy + cam.cy).astype(np.int32)
            valid = ((z > 0.01) & (u_i >= 0) & (u_i < cam.width)
                     & (v_i >= 0) & (v_i < cam.height))
            vi = np.where(valid)[0]
            if len(vi) == 0:
                continue
            uu, vv = u_i[vi], v_i[vi]
            keep = ((alpha_np[vv, uu] > 0.7) & fg_mask[vv, uu]
                    & (np.abs(z[vi] - depth_np[vv, uu]) < depth_tol))
            vi = vi[keep]
            if len(vi) == 0:
                continue
            uu, vv = u_i[vi], v_i[vi]
            cam_pos = cam.c2w_opencv[:3, 3]
            view_dirs = cam_pos - t_pos[vi]
            view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True).clip(min=1e-8)
            dots = np.sum(t_nrm[vi] * view_dirs, axis=1).clip(min=0)
            weights = dots * alpha_np[vv, uu]
            t_cam_color[vi] += rgb_unpremult[vv, uu] * weights[:, None]
            t_cam_weight[vi] += weights
            if (ci + 1) % 40 == 0:
                pr(f"  Camera {ci + 1}/{len(cams_full)}: {(t_cam_weight > 0.01).sum()}/{n_texels} textured")
            if (ci + 1) % 50 == 0:
                torch.cuda.empty_cache()

    has_cam = t_cam_weight > 0.01
    n_cam = has_cam.sum()
    pr(f"Camera-textured: {n_cam}/{n_texels} ({100 * n_cam / max(n_texels, 1):.1f}%)")
    t_cam_final = np.zeros_like(t_cam_color)
    t_cam_final[has_cam] = t_cam_color[has_cam] / t_cam_weight[has_cam, None]

    if n_cam > 100:
        cam_med = np.median(np.mean(t_cam_final[has_cam], axis=1))
        vc_med = np.median(np.mean(t_vc[has_cam], axis=1))
        scale = cam_med / vc_med if vc_med > 0.01 else 1.0
        t_vc_matched = (t_vc * scale).clip(0, 1)
    else:
        t_vc_matched = t_vc
    t_final = np.where(has_cam[:, None], t_cam_final, t_vc_matched)

    # Color grading
    pr("Color grading...")
    lo = np.percentile(t_final[t_final > 0.01], 2)
    hi = np.percentile(t_final[t_final > 0.01], 98)
    if hi - lo > 0.01:
        np.clip(t_final, lo, hi, out=t_final)
        t_final -= lo
        t_final /= (hi - lo)
    t_final = t_final - 0.5
    t_final *= CONTRAST * 6
    np.exp(-t_final, out=t_final)
    t_final += 1.0
    np.divide(1.0, t_final, out=t_final)
    s_lo = 1.0 / (1.0 + np.exp(CONTRAST * 6 * 0.5))
    s_hi = 1.0 / (1.0 + np.exp(-CONTRAST * 6 * 0.5))
    t_final -= s_lo
    t_final /= (s_hi - s_lo)
    np.clip(t_final, 0, 1, out=t_final)
    lum = 0.299 * t_final[:, 0] + 0.587 * t_final[:, 1] + 0.114 * t_final[:, 2]
    for ch in range(3):
        t_final[:, ch] = lum + (t_final[:, ch] - lum) * SAT_BOOST
    np.clip(t_final, 0, 1, out=t_final)
    t_final *= BRIGHTNESS

    del t_cam_color, t_cam_weight, t_cam_final, t_vc, t_vc_matched
    del t_pos, t_nrm, texel_pos, texel_nrm, texel_vc

    texture = np.full((TEX, TEX, 3), 128, dtype=np.uint8)
    texture[valid_ys, valid_xs] = (t_final.clip(0, 1) * 255).astype(np.uint8)
    pr("Dilating texture seams...")
    texture = dilate_texture(texture, texel_valid)
    tex_img = Image.fromarray(texture)
    tex_img.save(TEX_PATH)
    pr(f"Saved texture: {TEX_PATH}")

    # Export OBJ
    pr("Exporting OBJ...")
    mtl_path = OUT_OBJ.replace(".obj", ".mtl")
    tex_name = os.path.basename(TEX_PATH)
    mtl_name = os.path.basename(mtl_path)
    with open(mtl_path, "w") as f:
        f.write("newmtl material0\nillum 0\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\n")
        f.write(f"map_Kd {tex_name}\n")
    with open(OUT_OBJ, "w") as f:
        f.write(f"mtllib {mtl_name}\nusemtl material0\n")
        for v in new_verts.astype(np.float32):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for n in new_normals.astype(np.float32):
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for face in new_faces:
            i0, i1, i2 = face[0] + 1, face[1] + 1, face[2] + 1
            f.write(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")
    pr(f"Saved: {OUT_OBJ}")

    # Export GLB
    pr("Exporting GLB...")
    glb_uvs = uvs.copy()
    glb_uvs[:, 1] = 1.0 - glb_uvs[:, 1]
    export_glb(OUT_GLB, new_verts, new_faces, glb_uvs, new_normals, tex_img)
    pr(f"Saved: {OUT_GLB}")
    pr("Done!")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    DEVICE = args.device
    PLY = args.ply
    TRANSFORMS = args.transforms
    parse_args._skip_at = args.no_applied_transform
    OUT_OBJ = os.path.join(args.output_dir, f"{args.name}.obj")
    OUT_GLB = os.path.join(args.output_dir, f"{args.name}.glb")
    TEX_PATH = os.path.join(args.output_dir, f"{args.name}_texture.png")
    SCENE_MODE = None  # always auto-detect
    VOXEL_DIV = args.voxel_div
    TARGET_FACES = args.target_faces if args.target_faces is not None else 100_000
    TEXTURE_SIZE = args.texture_size
    BG_COLOR_THRESH = args.bg_color_thresh
    SMOOTH_ITERATIONS = args.smooth_iterations
    SMOOTH_LAMBDA = args.smooth_lambda
    SOLIDIFY_OFFSET = args.solidify_offset
    CONTRAST = args.contrast
    SAT_BOOST = args.saturation
    BRIGHTNESS = args.brightness
    main()
