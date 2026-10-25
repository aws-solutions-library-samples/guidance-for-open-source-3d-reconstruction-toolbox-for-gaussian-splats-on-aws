#!/bin/bash
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

# Builds models.tar.gz locally, mirroring what the model_deployment Lambda does.
# Output: ./models/models.tar.gz  (ready to mount as /opt/ml/input/data/model/ in local debug)
#
# Usage:
#   cd source/container
#   bash build_models_tar.sh
#
# Optional: upload to S3 after building
#   bash build_models_tar.sh --upload s3://your-bucket-name
#
# Optional: write models.tar.gz to a directory you own (recommended on shared
# filesystems like EFS to avoid permission problems):
#   bash build_models_tar.sh --output-dir /tmp/models-out

set -e

UPLOAD_BUCKET=""
OUTPUT_DIR_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --upload)
            UPLOAD_BUCKET="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR_OVERRIDE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUTPUT_DIR_OVERRIDE:-$SCRIPT_DIR/models}"
TMP_DIR="$(mktemp -d)"
ARCHIVE="$OUT_DIR/models.tar.gz"

# Validate the output directory is writable BEFORE downloading ~2 GB of models,
# so a permission problem fails fast with an actionable message instead of
# crashing on the final `tar` step after the long download.
ensure_writable_dir() {
    local dir="$1"
    if ! mkdir -p "$dir" 2>/dev/null; then
        echo "ERROR: cannot create output directory: $dir" >&2
        echo "  The parent directory is not writable by the current user." >&2
        print_permission_hint "$dir"
        exit 1
    fi
    if ! ( touch "$dir/.write_test" && rm -f "$dir/.write_test" ) 2>/dev/null; then
        local owner
        owner="$(stat -c '%U' "$dir" 2>/dev/null || echo unknown)"
        echo "ERROR: output directory is not writable: $dir" >&2
        echo "  Current user: $(id -un)   Directory owner: $owner" >&2
        echo "  This usually means the directory was created by a previous" >&2
        echo "  root/sudo or container run on a shared filesystem (e.g. EFS)." >&2
        print_permission_hint "$dir"
        exit 1
    fi
}

print_permission_hint() {
    local dir="$1"
    echo "  Fix it with either of:" >&2
    echo "    sudo chown -R \$(id -un):\$(id -gn) $dir" >&2
    echo "  or build to a directory you own:" >&2
    echo "    bash build_models_tar.sh --output-dir /tmp/models-out" >&2
    echo "  And run this script WITHOUT sudo so the directory stays user-owned." >&2
}

ensure_writable_dir "$OUT_DIR"

echo "=== Building models.tar.gz ==="
echo "Temp dir: $TMP_DIR"
echo "Output:   $ARCHIVE"

# Helper: download with progress
download() {
    local url="$1"
    local dest="$2"
    echo "Downloading: $url -> $dest"
    curl -L --progress-bar -o "$dest" "$url"
}

# SAM2
echo ""
echo "--- SAM2 ---"
download \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
    "$TMP_DIR/sam2.1_hiera_large.pt"

# U2NET (assembled from split parts)
echo ""
echo "--- U2NET ---"
U2NET_DIR="$TMP_DIR/.u2net"
mkdir -p "$U2NET_DIR"

for part in u2aa u2ab u2ac u2ad; do
    download \
        "https://github.com/nadermx/backgroundremover/raw/main/models/$part" \
        "$U2NET_DIR/u2net.pth.$part"
done
cat "$U2NET_DIR/u2net.pth.u2aa" \
    "$U2NET_DIR/u2net.pth.u2ab" \
    "$U2NET_DIR/u2net.pth.u2ac" \
    "$U2NET_DIR/u2net.pth.u2ad" > "$U2NET_DIR/u2net.pth"
rm "$U2NET_DIR/u2net.pth.u2aa" "$U2NET_DIR/u2net.pth.u2ab" \
   "$U2NET_DIR/u2net.pth.u2ac" "$U2NET_DIR/u2net.pth.u2ad"

download \
    "https://github.com/nadermx/backgroundremover/raw/main/models/u2netp.pth" \
    "$U2NET_DIR/u2netp.pth"

for part in u2haa u2hab u2hac u2had; do
    download \
        "https://github.com/nadermx/backgroundremover/raw/main/models/$part" \
        "$U2NET_DIR/u2net_human_seg.pth.$part"
done
cat "$U2NET_DIR/u2net_human_seg.pth.u2haa" \
    "$U2NET_DIR/u2net_human_seg.pth.u2hab" \
    "$U2NET_DIR/u2net_human_seg.pth.u2hac" \
    "$U2NET_DIR/u2net_human_seg.pth.u2had" > "$U2NET_DIR/u2net_human_seg.pth"
rm "$U2NET_DIR/u2net_human_seg.pth.u2haa" "$U2NET_DIR/u2net_human_seg.pth.u2hab" \
   "$U2NET_DIR/u2net_human_seg.pth.u2hac" "$U2NET_DIR/u2net_human_seg.pth.u2had"

# COLMAP vocab tree
echo ""
echo "--- COLMAP vocab tree ---"
download \
    "https://github.com/ZachMckennedyFWig/ColmapFaissVocabTrees/raw/main/vocab_tree_flickr100K_words32K.bin" \
    "$TMP_DIR/vocab_tree_flickr100K_words32K.bin"

# PyTorch hub checkpoints
echo ""
echo "--- PyTorch models ---"
TORCH_DIR="$TMP_DIR/.cache/torch/hub/checkpoints"
mkdir -p "$TORCH_DIR"

download "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth"          "$TORCH_DIR/mobilenet_v3_large-8738ca79.pth"
download "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" "$TORCH_DIR/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
download "https://download.pytorch.org/models/vgg16-397923af.pth"                        "$TORCH_DIR/vgg16-397923af.pth"
download "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"                  "$TORCH_DIR/alexnet-owt-7be5be79.pth"

# Stable Diffusion XL placeholder (downloaded at runtime inside container)
echo ""
echo "--- Stable Diffusion XL placeholder ---"
SD_DIR="$TMP_DIR/stable-diffusion-xl-base-1.0"
mkdir -p "$SD_DIR"
echo "This model will be downloaded at runtime to avoid storage limits" \
    > "$SD_DIR/download_at_runtime.txt"

# Package
echo ""
echo "--- Creating archive ---"
tar -czf "$ARCHIVE" \
    -C "$TMP_DIR" \
    "sam2.1_hiera_large.pt" \
    ".u2net" \
    "vocab_tree_flickr100K_words32K.bin" \
    ".cache" \
    "stable-diffusion-xl-base-1.0"

ARCHIVE_MB=$(du -m "$ARCHIVE" | cut -f1)
echo "Archive created: $ARCHIVE (${ARCHIVE_MB} MB)"

# Cleanup
rm -rf "$TMP_DIR"

# Optional S3 upload
if [[ -n "$UPLOAD_BUCKET" ]]; then
    echo ""
    echo "--- Uploading to s3://$UPLOAD_BUCKET/models/models.tar.gz ---"
    aws s3 cp "$ARCHIVE" "s3://$UPLOAD_BUCKET/models/models.tar.gz"
    echo "Upload complete."
fi

echo ""
echo "=== Done ==="
echo "To use locally, mount the models/ directory:"
echo "  -v \$(pwd)/models:/opt/ml/input/data/model"
