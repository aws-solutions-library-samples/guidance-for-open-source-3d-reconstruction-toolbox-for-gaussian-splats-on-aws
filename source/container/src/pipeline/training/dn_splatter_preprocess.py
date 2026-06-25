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

"""Generates monocular normals and scale-aligned depth maps for dn-splatter/ags-mesh training."""

import argparse
import os
import shutil
import sys
from pathlib import Path


# Ensures colmap/sparse symlink exists for depth alignment script
def ensure_colmap_sparse_link(data_dir):
    """Create colmap/sparse symlink if sparse/ exists but colmap/sparse does not."""
    sparse_path = os.path.join(data_dir, "sparse")
    colmap_sparse_path = os.path.join(data_dir, "colmap", "sparse")
    if os.path.exists(sparse_path) and not os.path.exists(colmap_sparse_path):
        colmap_dir = os.path.join(data_dir, "colmap")
        os.makedirs(colmap_dir, exist_ok=True)
        os.symlink(os.path.abspath(sparse_path), colmap_sparse_path)
        print(f"Created symlink: {colmap_sparse_path} -> {sparse_path}")


# Removes non-image files, flattens image subdirectories, and removes unreadable images.
# dn-splatter's normals_from_pretrain.py and depth_from_pretrain.py iterate images/ flat
# and crash on subdirs (IsADirectoryError) or unreadable files (cv2 empty assertion).
def clean_images_dir(images_dir):
    """Flatten image subdirs, remove non-image files, and remove OpenCV-unreadable images."""
    import cv2
    valid_extensions = {'.png', '.jpg', '.jpeg'}
    removed = 0
    flattened = 0
    invalid = 0

    # Pass 1: flatten subdirs and remove non-image files
    for entry in list(os.listdir(images_dir)):
        fpath = os.path.join(images_dir, entry)
        if os.path.isdir(fpath):
            for fname in os.listdir(fpath):
                if os.path.splitext(fname)[1].lower() in valid_extensions:
                    src = os.path.join(fpath, fname)
                    dst = os.path.join(images_dir, f"{entry}_{fname}")
                    os.rename(src, dst)
                    flattened += 1
            shutil.rmtree(fpath)
            print(f"Flattened subdir {entry}: moved {flattened} images")
        elif os.path.isfile(fpath) and os.path.splitext(entry)[1].lower() not in valid_extensions:
            os.remove(fpath)
            removed += 1

    # Pass 2: remove any image-extension file that OpenCV cannot decode
    for entry in list(os.listdir(images_dir)):
        fpath = os.path.join(images_dir, entry)
        if os.path.isfile(fpath) and os.path.splitext(entry)[1].lower() in valid_extensions:
            img = cv2.imread(fpath)
            if img is None:
                print(f"WARNING: Removing unreadable image: {fpath}")
                os.remove(fpath)
                invalid += 1

    print(f"clean_images_dir: flattened={flattened}, removed_non_image={removed}, removed_invalid={invalid}, "
          f"remaining={len([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])}")


def _generate_sfm_depths_pycolmap(data_dir: str, colmap_sparse: str) -> None:
    """Generate per-image SFM depth maps from a COLMAP reconstruction using pycolmap 4.x.

    Uses pycolmap.Reconstruction which correctly handles COLMAP 4.x rig format
    (rigs.bin/frames.bin) that breaks the legacy read_images_binary approach.
    Outputs to <data_dir>/sfm_depths/ as .npy files.
    """
    import numpy as np
    import pycolmap

    recon = pycolmap.Reconstruction(colmap_sparse)
    print(f"[SFM depths] Loaded: {recon.num_images()} images, "
          f"{recon.num_points3D()} 3D points")

    if recon.num_points3D() == 0:
        raise ValueError("Reconstruction has 0 3D points — cannot generate SFM depths")

    sfm_depths_dir = os.path.join(data_dir, "sfm_depths")
    os.makedirs(sfm_depths_dir, exist_ok=True)

    # Build a stem->camera map so stubs have the correct (H, W).
    # Fall back to reading the image file if the camera lookup fails.
    stem_to_hw = {}
    for image_id, image in recon.images.items():
        cam = recon.cameras[image.camera_id] if image.camera_id in recon.cameras else None
        if cam is not None:
            stem = os.path.splitext(os.path.basename(image.name))[0]
            stem_to_hw[stem] = (cam.height, cam.width)

    # Pre-create zero depth maps for every image so the count always matches mono_depth.
    # Images with no triangulated points will keep their zero map; images with points
    # will have their map overwritten below.
    images_dir = os.path.join(data_dir, "images")
    if os.path.isdir(images_dir):
        for fname in os.listdir(images_dir):
            if os.path.splitext(fname)[1].lower() in {".png", ".jpg", ".jpeg"}:
                stem = os.path.splitext(fname)[0]
                zero_path = os.path.join(sfm_depths_dir, stem + ".npy")
                if not os.path.exists(zero_path):
                    if stem in stem_to_hw:
                        H, W = stem_to_hw[stem]
                    else:
                        # Fall back to reading image dimensions
                        try:
                            import cv2 as _cv2
                            img = _cv2.imread(os.path.join(images_dir, fname))
                            H, W = img.shape[:2] if img is not None else (1, 1)
                        except Exception:
                            H, W = 1, 1
                    np.save(zero_path, np.zeros((H, W), dtype=np.float32))

    generated = 0
    for image_id, image in recon.images.items():
        if not image.has_pose:
            continue
        cam = recon.cameras[image.camera_id]
        W, H = cam.width, cam.height
        depth_map = np.zeros((H, W), dtype=np.float32)

        # Get the camera-from-world rotation and translation for depth computation
        pose = image.cam_from_world()  # Rigid3d
        R = pose.rotation.matrix()    # 3x3
        t = pose.translation           # (3,)

        for p2d in image.points2D:
            if not p2d.has_point3D:
                continue
            if p2d.point3D_id not in recon.points3D:
                continue
            p3d = recon.points3D[p2d.point3D_id]
            # Project to image coordinates
            uv = image.project_point(p3d.xyz)
            if uv is None:
                continue
            # Depth = Z in camera space = R[2,:] @ xyz + t[2]
            z = float(R[2] @ p3d.xyz + t[2])
            if z <= 0:
                continue
            u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
            if 0 <= u < W and 0 <= v < H:
                if depth_map[v, u] == 0 or z < depth_map[v, u]:
                    depth_map[v, u] = z

        stem = os.path.splitext(os.path.basename(image.name))[0]
        np.save(os.path.join(sfm_depths_dir, stem + ".npy"), depth_map)
        generated += 1

    print(f"[SFM depths] Generated {generated} depth maps in {sfm_depths_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate normals and aligned depth for dn-splatter")
    parser.add_argument("--data-dir", required=True, help="Path to dataset root")
    parser.add_argument("--normal-format", default="dsine", choices=["omnidata", "dsine"],
                        help="Normal estimation network to use")
    parser.add_argument("--skip-normals", action="store_true", help="Skip normal generation")
    parser.add_argument("--skip-depth", action="store_true", help="Skip depth alignment")
    parser.add_argument("--has-sensor-depth", action="store_true",
                        help="Sensor depth images are in depth/ dir; skip mono depth generation")
    parser.add_argument("--generate-depth-masks", action="store_true",
                        help="Generate depth-normal consistency masks (required for ags-mesh)")
    args = parser.parse_args()

    data_dir = args.data_dir
    images_dir = os.path.join(data_dir, "images")

    # Clean non-image files
    if os.path.exists(images_dir):
        clean_images_dir(images_dir)

    # Ensure colmap/sparse structure for depth alignment
    ensure_colmap_sparse_link(data_dir)

    # Ensure sparse.ply exists for dn-splatter's normal_nerfstudio.py (load_3D_points=True).
    # colmap_to_nerfstudio_cam.py writes the ply_file_path into transforms.json but does not
    # create the file itself. Generate it from points3D.bin/txt using pycolmap if missing.
    def ensure_sparse_ply(data_dir):
        for sparse_subdir in ["sparse/0", "colmap/sparse/0"]:
            sparse_dir = os.path.join(data_dir, sparse_subdir)
            if not os.path.isdir(sparse_dir):
                continue
            ply_path = os.path.join(sparse_dir, "sparse.ply")
            if os.path.isfile(ply_path):
                return  # already exists
            points3d_bin = os.path.join(sparse_dir, "points3D.bin")
            points3d_txt = os.path.join(sparse_dir, "points3D.txt")
            if os.path.isfile(points3d_bin) or os.path.isfile(points3d_txt):
                try:
                    import pycolmap
                    reconstruction = pycolmap.Reconstruction(sparse_dir)
                    reconstruction.export_PLY(ply_path)
                    print(f"Generated sparse.ply with {reconstruction.num_points3D()} points at {ply_path}")
                    return
                except Exception as e:
                    print(f"WARNING: Could not generate sparse.ply: {e}")

    ensure_sparse_ply(data_dir)

    # Patch dn_model.py: combined_depth_normalized is only set inside
    # 'if sensor_depth in batch' but referenced unconditionally at line 956.
    # Fix: initialize it right after combined_depth is first assigned.
    def _patch_dn_model_depth():
        path = "/opt/ml/code/dn-splatter/dn_splatter/dn_model.py"
        try:
            with open(path, 'r') as f:
                src = f.read()
            if 'combined_depth_normalized' in src and \
               'combined_depth = (\n            predicted_depth  # a placeholder' in src and \
               'combined_depth_normalized = combined_depth / combined_depth.max()' in src:
                patched = src.replace(
                    'combined_depth = (\n            predicted_depth  # a placeholder if no sensor depth is available\n        )',
                    'combined_depth = (\n            predicted_depth  # a placeholder if no sensor depth is available\n        )\n        combined_depth_normalized = combined_depth / (combined_depth.max() + 1e-8)  # default; overwritten if sensor_depth present'
                )
                if patched != src:
                    with open(path, 'w') as f:
                        f.write(patched)
                    print(f"Patched dn_model.py combined_depth_normalized initialization")
                else:
                    print("WARNING: dn_model.py combined_depth patch did not apply (text not found)")
            else:
                print("INFO: dn_model.py combined_depth already patched or has different structure")
        except Exception as e:
            print(f"WARNING: Could not patch dn_model.py: {e}")
    _patch_dn_model_depth()

    # Sanitize transforms.json: normals_from_pretrain.py reads file_path entries directly
    # and passes them to cv2.imread — it does NOT scan images/ on disk. After clean_images_dir
    # flattens subdirs (e.g. images/face_00/x.jpg -> images/face_00_x.jpg), the old paths in
    # transforms.json no longer exist, causing cv2.imread to return None -> cvtColor crash.
    # Fix: rewrite stale file_path entries to their flattened equivalents, drop unreadable ones.
    def sanitize_transforms_json(data_dir):
        import json
        import cv2 as _cv2
        transforms_path = os.path.join(data_dir, "transforms.json")
        if not os.path.exists(transforms_path):
            return
        with open(transforms_path, "r") as f:
            meta = json.load(f)
        frames = meta.get("frames", [])
        clean_frames = []
        rewritten = 0
        dropped = 0
        for frame in frames:
            fp = frame.get("file_path", "")
            abs_path = os.path.join(data_dir, fp)
            if not os.path.exists(abs_path):
                # Try to find the flattened equivalent: images/<subdir>_<filename>
                parts = Path(fp).parts  # e.g. ('images', 'face_00', 'frame.jpg')
                if len(parts) >= 3:
                    flat_name = "_".join(parts[1:])  # face_00_frame.jpg
                    flat_path = os.path.join(data_dir, parts[0], flat_name)
                    if os.path.exists(flat_path):
                        frame["file_path"] = str(Path(parts[0]) / flat_name)
                        abs_path = flat_path
                        rewritten += 1
                    else:
                        print(f"WARNING: dropping frame, file not found: {abs_path}")
                        dropped += 1
                        continue
                else:
                    print(f"WARNING: dropping frame, file not found: {abs_path}")
                    dropped += 1
                    continue
            if _cv2.imread(abs_path) is None:
                print(f"WARNING: dropping frame, unreadable image: {abs_path}")
                dropped += 1
                continue
            clean_frames.append(frame)
        meta["frames"] = clean_frames
        with open(transforms_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"sanitize_transforms_json: rewritten={rewritten}, dropped={dropped}, "
              f"remaining={len(clean_frames)}")

    sanitize_transforms_json(data_dir)

    # Generate monocular normals
    if not args.skip_normals:
        print(f"Generating {args.normal_format} normals for {data_dir}")
        from dn_splatter.scripts.normals_from_pretrain import NormalsFromPretrained
        normals_generator = NormalsFromPretrained(
            data_dir=Path(data_dir),
            normal_format=args.normal_format,
        )
        normals_generator.main()

    # Inject depth_file_path into transforms.json only when sensor depth is valid
    # uint16 mm PNGs (cv2.IMREAD_ANYDEPTH gives uint16, x0.001 -> meters in plausible range).
    # ARKit .depth.png files are uint8 confidence maps and are NOT valid sensor depth.
    # When valid sensor depth is present, depth_file_path enables EdgeAwareLogL1 supervision.
    # Otherwise training uses scale-aligned mono depth via mono_depth/*_aligned.npy.
    import json as _json
    _depth_dir = os.path.join(data_dir, "depth")
    if not os.path.isdir(_depth_dir):
        _depth_dir = os.path.join(data_dir, "depth_images")
    _transforms_path = os.path.join(data_dir, "transforms.json")
    if os.path.isdir(_depth_dir) and os.path.isfile(_transforms_path):
        # Flatten depth subdirs to match flattened images/ convention:
        # depth/face_00/pano_008.png -> depth/face_00_pano_008.png
        for _entry in list(os.listdir(_depth_dir)):
            _entry_path = os.path.join(_depth_dir, _entry)
            if os.path.isdir(_entry_path):
                for _fname in os.listdir(_entry_path):
                    if os.path.splitext(_fname)[1].lower() in {'.png', '.jpg', '.npy'}:
                        _src = os.path.join(_entry_path, _fname)
                        _dst = os.path.join(_depth_dir, f"{_entry}_{_fname}")
                        os.rename(_src, _dst)
                shutil.rmtree(_entry_path)
                print(f"Flattened depth subdir {_entry}")

        _depth_files = {os.path.splitext(f)[0].lower(): f
                        for f in os.listdir(_depth_dir)
                        if f.lower().endswith(('.png', '.jpg', '.npy'))}
        if _depth_files:
            # Validate format: sample one file and check it's uint16 PNG with plausible metric range
            _sample_file = os.path.join(_depth_dir, next(iter(_depth_files.values())))
            _valid_sensor_depth = False
            try:
                import cv2 as _cv2
                _sample = _cv2.imread(_sample_file, _cv2.IMREAD_ANYDEPTH)
                if _sample is not None and _sample.dtype == 'uint16':
                    _meters_max = float(_sample.max()) * 0.001
                    _meters_min = float(_sample[_sample > 0].min()) * 0.001 if (_sample > 0).any() else 0
                    # Plausible scene depth: 0.1m to 1000m
                    if _meters_min >= 0.1 and _meters_max <= 1000.0:
                        _valid_sensor_depth = True
                        print(f"Sensor depth validated: uint16 PNG, range [{_meters_min:.2f}, {_meters_max:.2f}]m")
                    else:
                        print(f"WARNING: Sensor depth rejected — range [{_meters_min:.3f}, {_meters_max:.3f}]m is not plausible metric depth. "
                              f"Expected uint16 mm values (e.g. 2500 = 2.5m). Falling back to mono depth.")
                else:
                    print(f"WARNING: Sensor depth rejected — expected uint16 PNG but got dtype={getattr(_sample, 'dtype', 'None')}. "
                          f"Falling back to mono depth.")
            except Exception as _e:
                print(f"WARNING: Could not validate sensor depth: {_e}. Falling back to mono depth.")

            if _valid_sensor_depth:
                with open(_transforms_path, "r") as _f:
                    _transforms = _json.load(_f)
                _injected = 0
                _depth_dir_name = os.path.basename(_depth_dir)
                for _frame in _transforms.get("frames", []):
                    if "depth_file_path" not in _frame:
                        _img = _frame.get("file_path", "")
                        _stem = os.path.splitext(os.path.basename(_img))[0].lower()
                        if _stem in _depth_files:
                            _frame["depth_file_path"] = f"{_depth_dir_name}/{_depth_files[_stem]}"
                            _injected += 1
                        else:
                            _num = ''.join(filter(str.isdigit, _stem))
                            _matched = next((k for k in _depth_files if ''.join(filter(str.isdigit, k)) == _num and _num), None)
                            if _matched:
                                _frame["depth_file_path"] = f"{_depth_dir_name}/{_depth_files[_matched]}"
                                _injected += 1
                if _injected > 0:
                    with open(_transforms_path, "w") as _f:
                        _json.dump(_transforms, _f, indent=2)
                    print(f"Injected depth_file_path into {_injected} frames from {_depth_dir_name}/")
                    # Write marker so training knows to use EdgeAwareLogL1 for sensor depth
                    open(os.path.join(data_dir, ".sensor_depth_valid"), 'w').close()
                else:
                    print(f"WARNING: Could not match sensor depth files to transforms.json frames. Falling back to mono depth.")

            else:
                print(f"INFO: Using scale-aligned mono depth (mono_depth/*_aligned.npy).")
        else:
            print(f"INFO: No depth files found in {_depth_dir}. Using mono depth.")

    # Monkey-patch ZoeDepth model_io to use strict=False, fixing timm>=0.9
    # relative_position_index buffer key mismatch with the ZoeD_M12_N checkpoint.
    # torch.hub downloads ZoeDepth on first torch.hub.load call, so we pre-trigger
    # the download with source_only=True, patch the file, then the real load proceeds.
    def _patch_zoedepth_model_io():
        try:
            import torch.hub as _hub
            import glob as _glob
            # Pre-download ZoeDepth repo (no model weights, just source) so we can patch.
            # skip_validation=True avoids the GitHub fork-check HTTP call that fails in
            # network-restricted SageMaker/Batch environments (HTTP 504).
            _hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=False,
                      trust_repo=True, skip_validation=True)
        except Exception:
            pass  # may fail without weights; we only need the source on disk
        # Pre-download MiDaS so its beit.py is on disk before we patch it.
        try:
            _hub.load("intel-isl/MiDaS", "DPT_BEiT_L_512", pretrained=False,
                      trust_repo=True, skip_validation=True)
        except Exception:
            pass
        try:
            hub_dir = _hub.get_dir()
            candidates = _glob.glob(
                os.path.join(hub_dir, "isl-org_ZoeDepth*", "zoedepth", "models", "model_io.py")
            )
            for path in candidates:
                with open(path, "r") as f:
                    src = f.read()
                patched = src.replace(
                    "model.load_state_dict(state)\n",
                    "model.load_state_dict(state, strict=False)\n",
                )
                if patched != src:
                    with open(path, "w") as f:
                        f.write(patched)
                    print(f"Patched ZoeDepth model_io: {path}")
            # Evict cached in-memory copies of the patched module so the next
            # torch.hub.load re-imports from the patched file rather than the
            # stale in-memory version loaded during the pretrained=False call.
            for mod_name in list(sys.modules.keys()):
                if "zoedepth" in mod_name or "model_io" in mod_name:
                    del sys.modules[mod_name]
        except Exception as e:
            print(f"WARNING: Could not patch ZoeDepth model_io: {e}")
        # Patch MiDaS beit.py: timm>=0.9 renamed Block.drop_path -> drop_path1/drop_path2.
        # The MiDaS hub code calls self.drop_path(...) which no longer exists.
        try:
            hub_dir = _hub.get_dir()
            midas_beit_candidates = _glob.glob(
                os.path.join(hub_dir, "intel-isl_MiDaS*", "midas", "backbones", "beit.py")
            )
            for path in midas_beit_candidates:
                with open(path, "r") as f:
                    src = f.read()
                patched = src
                # Replace both occurrences: drop_path( -> drop_path1(
                # The MiDaS block_forward uses drop_path for both attn and mlp paths;
                # timm 0.9+ splits these into drop_path1 and drop_path2 respectively.
                patched = patched.replace(
                    "self.drop_path(self.attn(",
                    "self.drop_path1(self.attn("
                ).replace(
                    "self.drop_path(self.gamma_1",
                    "self.drop_path1(self.gamma_1"
                ).replace(
                    "self.drop_path(self.mlp(",
                    "self.drop_path2(self.mlp("
                ).replace(
                    "self.drop_path(self.gamma_2",
                    "self.drop_path2(self.gamma_2"
                )
                if patched != src:
                    with open(path, "w") as f:
                        f.write(patched)
                    print(f"Patched MiDaS beit.py: {path}")
            for mod_name in list(sys.modules.keys()):
                if "midas" in mod_name or "beit" in mod_name:
                    del sys.modules[mod_name]
        except Exception as e:
            print(f"WARNING: Could not patch MiDaS beit.py: {e}")

    # Generate scale-aligned mono depth (always run regardless of sensor depth)
    if not args.skip_depth:
        # Monkey-patch torch.hub.load to always pass skip_validation=True so the
        # GitHub fork-check HTTP call (which fails in network-restricted environments
        # with HTTP 504) is bypassed for all subsequent hub.load calls, including
        # those made inside dn-splatter's depth_from_pretrain.py.
        import torch.hub as _hub
        _orig_hub_load = _hub.load
        def _hub_load_no_validate(*a, **kw):
            kw.setdefault("skip_validation", True)
            return _orig_hub_load(*a, **kw)
        _hub.load = _hub_load_no_validate
        _patch_zoedepth_model_io()
        colmap_sparse = os.path.join(data_dir, "colmap", "sparse", "0")
        if os.path.exists(colmap_sparse):
            print(f"Generating SFM depths from {colmap_sparse} using pycolmap")
            try:
                _generate_sfm_depths_pycolmap(data_dir, colmap_sparse)
                from dn_splatter.scripts.align_depth import ColmapToAlignedMonoDepths
                # Run mono depth generation first (skip_mono_depth_creation=False,
                # skip_alignment=True) so we know the exact set of mono_depth files
                # before attempting alignment — this avoids the count mismatch assert.
                mono_depth_dir = os.path.join(data_dir, "mono_depth")
                mono_already_done = os.path.isdir(mono_depth_dir) and \
                    any(f.endswith(".npy") for f in os.listdir(mono_depth_dir))
                if not mono_already_done:
                    depth_aligner_mono = ColmapToAlignedMonoDepths(
                        data=Path(data_dir),
                        skip_colmap_to_depths=True,
                        skip_mono_depth_creation=False,
                        skip_alignment=True,
                    )
                    depth_aligner_mono.main()
                else:
                    print(f"INFO: mono_depth already exists ({mono_depth_dir}), skipping re-generation.")
                # Reconcile sfm_depths to exactly match mono_depth stems.
                # align_depth.py asserts len(sfm)==len(filtered_mono) where
                # filtered_mono keeps only items whose stem appears in sfm_name.
                # Any mismatch in either direction triggers the assert.
                import numpy as np
                sfm_depths_dir = os.path.join(data_dir, "sfm_depths")
                mono_stems = {os.path.splitext(f)[0] for f in os.listdir(mono_depth_dir) if f.endswith(".npy")}
                sfm_stems = {os.path.splitext(f)[0] for f in os.listdir(sfm_depths_dir) if f.endswith(".npy")}
                # Add zero placeholders for mono stems missing from sfm.
                # Always derive shape from mono_depth — sfm placeholders may be (1,1) stubs.
                for stem in mono_stems - sfm_stems:
                    _ref_mono = os.path.join(mono_depth_dir, stem + '.npy')
                    _shape = np.load(_ref_mono).shape[:2]
                    np.save(os.path.join(sfm_depths_dir, stem + '.npy'), np.zeros(_shape, dtype=np.float32))
                    print(f"[SFM depths] Added zero placeholder for {stem}")
                # Remove sfm files that have no corresponding mono_depth
                for stem in sfm_stems - mono_stems:
                    os.remove(os.path.join(sfm_depths_dir, stem + ".npy"))
                    print(f"[SFM depths] Removed unmatched sfm file for {stem}")
                # Now run alignment only — both sets are guaranteed equal size.
                depth_aligner = ColmapToAlignedMonoDepths(
                    data=Path(data_dir),
                    skip_colmap_to_depths=True,
                    skip_mono_depth_creation=True,
                    skip_alignment=False,
                )
                depth_aligner.main()
            except Exception as depth_err:
                print(f"WARNING: SFM depth alignment failed ({depth_err}). "
                      f"Falling back to mono depth without scale alignment.")
                import traceback; traceback.print_exc()
                from dn_splatter.scripts.align_depth import ColmapToAlignedMonoDepths
                mono_depth_dir = os.path.join(data_dir, "mono_depth")
                mono_already_done = os.path.isdir(mono_depth_dir) and \
                    any(f.endswith(".npy") for f in os.listdir(mono_depth_dir))
                if not mono_already_done:
                    depth_aligner = ColmapToAlignedMonoDepths(
                        data=Path(data_dir),
                        skip_colmap_to_depths=True,
                        skip_mono_depth_creation=False,
                        skip_alignment=True,
                    )
                    depth_aligner.main()
                else:
                    print(f"INFO: mono_depth already exists ({mono_depth_dir}), skipping re-generation.")
        else:
            print(f"WARNING: No colmap/sparse/0 found at {colmap_sparse}, skipping depth alignment")
            from dn_splatter.scripts.align_depth import ColmapToAlignedMonoDepths
            depth_aligner = ColmapToAlignedMonoDepths(
                data=Path(data_dir),
                skip_colmap_to_depths=True,
                skip_mono_depth_creation=False,
                skip_alignment=True,
            )
            depth_aligner.main()

    # Generate depth-normal consistency masks for AGS-Mesh
    # Only run when sensor depth exists — depth_normal_consistency.py requires
    # depth_file_path in every transforms.json frame, which is only injected
    # when a depth/ or depth_images/ directory is present.
    # Only generate depth-normal consistency masks when valid sensor depth was confirmed
    # (depth_file_path entries injected, marked by .sensor_depth_valid).
    # depth_normal_consistency.py requires depth_file_path in every frame.
    _sensor_depth_valid_marker = os.path.join(data_dir, ".sensor_depth_valid")
    if args.generate_depth_masks and os.path.exists(_sensor_depth_valid_marker):
        print(f"Generating depth-normal consistency masks for {data_dir}")
        from dn_splatter.scripts.depth_normal_consistency import DepthNormalConsistency
        mask_generator = DepthNormalConsistency(
            data_dir=Path(data_dir),
            transforms_name="transforms.json",
            normal_format=args.normal_format,
        )
        mask_generator.main()

    print("DN-Splatter pre-processing complete.")


if __name__ == "__main__":
    main()
