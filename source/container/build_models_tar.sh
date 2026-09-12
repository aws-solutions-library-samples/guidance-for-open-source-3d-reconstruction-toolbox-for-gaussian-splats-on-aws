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

set -e

UPLOAD_BUCKET=""
if [[ "$1" == "--upload" && -n "$2" ]]; then
    UPLOAD_BUCKET="$2"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/models"
TMP_DIR="$(mktemp -d)"
ARCHIVE="$OUT_DIR/models.tar.gz"

mkdir -p "$OUT_DIR"

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
