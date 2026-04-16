#!/usr/bin/env python3
"""
Convert cameras.bin to PINHOLE model in-place.
Reads cameras.bin, strips distortion parameters, writes back as PINHOLE.
Does NOT touch images.bin or image files - image names are preserved.
This is used for gsplat depth loss which needs PINHOLE cameras but cannot
use the COLMAP undistorter (which renames images and breaks point_indices).
"""
import argparse
import os
import struct
import shutil


CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_NAMES = {v[0]: (k, v[1]) for k, v in CAMERA_MODEL_IDS.items()}

PINHOLE_MODEL_ID = 1
PINHOLE_NUM_PARAMS = 4  # fx, fy, cx, cy


def read_cameras_bin(path):
    cameras = {}
    with open(path, 'rb') as f:
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack('<I', f.read(4))[0]
            model_id = struct.unpack('<I', f.read(4))[0]
            width = struct.unpack('<Q', f.read(8))[0]
            height = struct.unpack('<Q', f.read(8))[0]
            num_params = CAMERA_MODEL_IDS.get(model_id, (None, 0))[1]
            params = list(struct.unpack(f'<{num_params}d', f.read(8 * num_params)))
            cameras[camera_id] = {
                'model_id': model_id,
                'model_name': CAMERA_MODEL_IDS.get(model_id, ('UNKNOWN', 0))[0],
                'width': width,
                'height': height,
                'params': params,
            }
    return cameras


def write_cameras_bin(path, cameras):
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(cameras)))
        for camera_id, cam in cameras.items():
            f.write(struct.pack('<I', camera_id))
            f.write(struct.pack('<I', cam['model_id']))
            f.write(struct.pack('<Q', cam['width']))
            f.write(struct.pack('<Q', cam['height']))
            f.write(struct.pack(f'<{len(cam["params"])}d', *cam['params']))


def to_pinhole(cam):
    """Convert any camera model to PINHOLE (fx, fy, cx, cy) by dropping distortion."""
    model = cam['model_name']
    p = cam['params']
    if model == 'PINHOLE':
        return p[:4]  # already fx, fy, cx, cy
    elif model == 'SIMPLE_PINHOLE':
        f, cx, cy = p[0], p[1], p[2]
        return [f, f, cx, cy]
    elif model == 'SIMPLE_RADIAL':
        f, cx, cy = p[0], p[1], p[2]  # p[3] = k1 (dropped)
        return [f, f, cx, cy]
    elif model == 'RADIAL':
        f, cx, cy = p[0], p[1], p[2]  # p[3]=k1, p[4]=k2 (dropped)
        return [f, f, cx, cy]
    elif model in ('OPENCV', 'FULL_OPENCV', 'THIN_PRISM_FISHEYE'):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]  # rest dropped
        return [fx, fy, cx, cy]
    else:
        # Fallback: assume first param is f or fx
        if len(p) >= 4:
            return [p[0], p[1], p[2], p[3]]
        elif len(p) >= 3:
            return [p[0], p[0], p[1], p[2]]
        return p[:4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sparse_path', required=True,
                        help='Path to sparse/0 directory containing cameras.bin')
    args = parser.parse_args()

    cameras_bin = os.path.join(args.sparse_path, 'cameras.bin')
    if not os.path.exists(cameras_bin):
        print(f"cameras.bin not found at {cameras_bin}, skipping")
        return

    cameras = read_cameras_bin(cameras_bin)

    already_pinhole = all(c['model_name'] == 'PINHOLE' for c in cameras.values())
    if already_pinhole:
        print("All cameras already PINHOLE, nothing to do")
        return

    # Backup
    shutil.copy2(cameras_bin, cameras_bin + '.backup')

    updated = {}
    for cid, cam in cameras.items():
        print(f"Camera {cid}: {cam['model_name']} {cam['width']}x{cam['height']} params={cam['params']}")
        pinhole_params = to_pinhole(cam)
        updated[cid] = {
            'model_id': PINHOLE_MODEL_ID,
            'model_name': 'PINHOLE',
            'width': cam['width'],
            'height': cam['height'],
            'params': pinhole_params,
        }
        print(f"  -> PINHOLE params={pinhole_params}")

    write_cameras_bin(cameras_bin, updated)
    print(f"Updated {cameras_bin} to PINHOLE model")


if __name__ == '__main__':
    main()
