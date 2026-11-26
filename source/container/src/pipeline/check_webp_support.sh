#!/usr/bin/env bash
# Check WebP input support and full-quality image writes inside the container:
#   docker run --rm --entrypoint /opt/ml/code/check_webp_support.sh 3dgs-container:latest
# Required (exit 1 if missing): OpenImageIO with libwebp (COLMAP reads images
# through it), colmap feature_extractor on a .webp image, WebP decoding in PIL,
# imageio and OpenCV, and image_formats.py writing JPEG at quality 100 / 4:4:4
# and WebP lossless. Informational: pycolmap cannot decode WebP, so pipeline steps
# that use pycolmap (bundle_adjust_model.py, panorama_sfm.py) hand it PNG copies.
set -u
CODE_PATH="${CODE_PATH:-/opt/ml/code}"
export PYTHONPATH="${CODE_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
FAIL=0

echo "== OpenImageIO"
LIB="$(ls /usr/local/lib/libOpenImageIO.so* 2>/dev/null | head -n 1)"
if [ -n "${LIB}" ] && ldd "${LIB}" | grep -q libwebp; then
    echo "PASS  ${LIB} links libwebp"
else
    echo "FAIL  OpenImageIO without libwebp (${LIB:-not found})"
    FAIL=1
fi

echo "== Python image stack and image_formats.py"
python - "${WORK}" <<'PY' || FAIL=1
import os
import sys

import cv2
import imageio.v2 as iio
import numpy as np
from PIL import Image, JpegImagePlugin

import image_formats as F

work = sys.argv[1]
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}")


rng = np.random.default_rng(0)
img = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
# Blocks of texture, so SIFT finds features in the colmap test below.
img = cv2.resize(cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA), (256, 256),
                 interpolation=cv2.INTER_NEAREST)
src = os.path.join(work, "images", "test.webp")
os.makedirs(os.path.dirname(src), exist_ok=True)
Image.fromarray(img).save(src, lossless=True)
check("PIL reads lossless WebP", np.array_equal(np.array(Image.open(src).convert("RGB")), img))
check("imageio reads WebP", np.array_equal(np.asarray(iio.imread(src))[..., :3], img))
check("OpenCV reads WebP", np.array_equal(cv2.imread(src)[..., ::-1], img))

w = os.path.join(work, "out.webp")
F.imwrite(w, img[..., ::-1], [cv2.IMWRITE_WEBP_QUALITY, 90])
check("image_formats.imwrite keeps WebP lossless", np.array_equal(cv2.imread(w)[..., ::-1], img))
w2 = os.path.join(work, "out_pil.webp")
Image.fromarray(img).save(w2, **F.pil_save_kwargs(w2))
check("pil_save_kwargs keeps WebP lossless", np.array_equal(np.array(Image.open(w2).convert("RGB")), img))


def q100_444(path):
    im = Image.open(path)
    return all(v == 1 for t in im.quantization.values() for v in t) and JpegImagePlugin.get_sampling(im) == 0


j = os.path.join(work, "out.jpg")
F.imwrite(j, img[..., ::-1])
check("image_formats.imwrite writes JPEG at quality 100, 4:4:4", q100_444(j))
j2 = os.path.join(work, "out_pil.jpg")
Image.fromarray(img).save(j2, **F.pil_save_kwargs(j2))
check("pil_save_kwargs writes JPEG at quality 100, 4:4:4", q100_444(j2))

png = os.path.join(work, "copy", "test.webp.png")
check("write_png_copy makes a pixel-identical PNG",
      F.write_png_copy(src, png) and np.array_equal(cv2.imread(png)[..., ::-1], img))
try:
    import pycolmap
    readable = pycolmap.Bitmap.read(src, True) is not None
    print(f"INFO  pycolmap {pycolmap.__version__} decodes WebP: {readable} "
          f"(pipeline steps give pycolmap PNG copies of WebP images)")
    check("pycolmap reads the PNG copy", pycolmap.Bitmap.read(png, True) is not None)
except ImportError:
    print("INFO  pycolmap not importable")
sys.exit(0 if ok else 1)
PY

echo "== colmap feature_extractor on a WebP image"
if colmap feature_extractor --database_path "${WORK}/db.db" --image_path "${WORK}/images" \
        --FeatureExtraction.use_gpu 0 > "${WORK}/fe.log" 2>&1; then
    N="$(python -c "import sqlite3, sys; print(sqlite3.connect(sys.argv[1]).execute('select coalesce(sum(rows), 0) from keypoints').fetchone()[0])" "${WORK}/db.db")"
    if [ "${N}" -gt 0 ]; then
        echo "PASS  colmap extracted ${N} features from test.webp"
    else
        echo "FAIL  colmap read no features from test.webp"; tail -n 5 "${WORK}/fe.log"; FAIL=1
    fi
else
    echo "FAIL  colmap feature_extractor failed"; tail -n 5 "${WORK}/fe.log"; FAIL=1
fi

if [ "${FAIL}" -eq 0 ]; then echo "WebP support OK"; else echo "WebP support INCOMPLETE"; fi
exit "${FAIL}"
