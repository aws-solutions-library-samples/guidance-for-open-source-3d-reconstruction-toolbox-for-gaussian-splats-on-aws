"""Debug tool: write colorized depth previews for visual inspection.

For each image it writes one JPG with four panels:
  RGB | LiDAR depth (depth/*.png) | LiDAR over RGB | aligned mono depth (mono_depth/*_aligned.npy)
All depth panels share one color scale (turbo, near = blue, far = red); black = no depth.

Run inside the container (it has OpenCV), from source/container on the host:
  docker run --rm -v $(pwd):/mnt/data --entrypoint python \
    3dgs-container:latest /opt/ml/code/training/view_depth.py [--every 10] [--max-depth 6]

Output: workflow-input/depth_preview/
"""
import argparse
import os

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data", default="/mnt/data/workflow-input")
p.add_argument("--every", type=int, default=10, help="preview every Nth image")
p.add_argument("--max-depth", type=float, default=0.0,
               help="color scale max in meters (0 = 98th percentile of LiDAR depth)")
p.add_argument("--width", type=int, default=480, help="panel width in pixels")
args = p.parse_args()

img_dir = os.path.join(args.data, "images")
lidar_dir = os.path.join(args.data, "depth")
mono_dir = os.path.join(args.data, "mono_depth")
out_dir = os.path.join(args.data, "depth_preview")
os.makedirs(out_dir, exist_ok=True)

names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
names = names[:: max(1, args.every)]


def load_lidar(stem):
    f = os.path.join(lidar_dir, stem + ".png")
    d = cv2.imread(f, cv2.IMREAD_ANYDEPTH) if os.path.exists(f) else None
    return None if d is None else d.astype(np.float32) / 1000.0


def load_mono(stem):
    f = os.path.join(mono_dir, stem + "_aligned.npy")
    return np.load(f).squeeze().astype(np.float32) if os.path.exists(f) else None


# Shared color scale so panels are comparable across images.
vmax = args.max_depth
if vmax <= 0:
    samples = [d[d > 0][::50] for d in (load_lidar(os.path.splitext(n)[0]) for n in names) if d is not None]
    vmax = float(np.percentile(np.concatenate(samples), 98)) if samples else 5.0
print(f"Color scale: 0 - {vmax:.2f} m")


def colorize(depth, size):
    if depth is None:
        panel = np.zeros((size[1], size[0], 3), np.uint8)
        cv2.putText(panel, "missing", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return panel
    d = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap((np.clip(d / vmax, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[d <= 0.1] = 0  # pixels dn-splatter ignores (depth_tolerance)
    return color


def label(panel, text):
    cv2.putText(panel, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(panel, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return panel


for n in names:
    stem = os.path.splitext(n)[0]
    rgb = cv2.imread(os.path.join(img_dir, n))
    h, w = rgb.shape[:2]
    size = (args.width, int(round(h * args.width / w)))
    rgb_s = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)

    lidar = load_lidar(stem)
    lidar_c = colorize(lidar, size)
    overlay = cv2.addWeighted(rgb_s, 0.5, lidar_c, 0.5, 0)
    mono_c = colorize(load_mono(stem), size)

    stats = "missing" if lidar is None else \
        f"{lidar[lidar > 0].min():.2f}-{lidar.max():.2f} m, {100 * (lidar > 0.1).mean():.0f}% valid"
    row = np.hstack([label(rgb_s, stem), label(lidar_c, "LiDAR " + stats),
                     label(overlay, "LiDAR over RGB"), label(mono_c, "aligned mono")])
    cv2.imwrite(os.path.join(out_dir, stem + ".jpg"), row, [cv2.IMWRITE_JPEG_QUALITY, 90])

print(f"Wrote {len(names)} previews to {out_dir}")
