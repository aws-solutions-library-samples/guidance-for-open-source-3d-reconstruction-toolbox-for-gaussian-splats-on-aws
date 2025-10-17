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
#
# This file serves as the Dockerfile code that is used to build the container image for
# building splats using NerfStudio, GSplat, Colmap, etc

# nosemgrep: dockerfile-source-not-pinned
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Fixed versions
ENV PYTHON_VERSION="3.11.0"
ENV PYTHON_VERSION_="3.11"
ENV CUDA_VERSION="12.8"
ENV CUDA_VERSION_="128"
ENV TORCH_VERSION="2.7.1"
ENV TORCH_VISION_VERSION="0.22.1"
ENV CMAKE_VERSION="3.30.3"
ENV EIGEN_VERSION="3.4"
ENV CERES_VERSION="2.2.0"
ENV COLMAP_VERSION="3.12.4"
ENV GLOMAP_VERSION="1.1.0"

# Non-fixed versions (latest/dev/main)
# nerfstudio, gsplat, sogs, splatfacto-w, tiny-cuda-nn

# Only build for supported GPU architectures
#P3=7.0, G4dn=7.5, P4d=8.0, g5=8.6, g6/g6e=8.9
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9"
ENV TCNN_CUDA_ARCHITECTURES=70;75;80;86;89
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}

# Ensures that Python outputs everything that's printed directly to the terminal (so logs can be seen in real-time)
ENV PYTHONUNBUFFERED=TRUE

# Ensures Python doesn't try to write .pyc files to disk (useful for improving performance in some scenarios)
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Create code path ENV VAR for container
ARG CODE_PATH
ENV CODE_PATH="/opt/ml/code"

# Create model path ENV VAR for container
ARG MODEL_PATH
#ENV MODEL_PATH="/opt/ml/model"
ENV MODEL_PATH="/opt/ml/input/data/model"

# Create build argument for container
ARG DATASET_PATH
ENV DATASET_PATH="/opt/ml/input/data/train"

ARG LOCAL_DEBUG
ENV LOCAL_DEBUG=${LOCAL_DEBUG}

ARG UUID
ENV UUID=${UUID}

ARG S3_INPUT
ENV S3_INPUT=${S3_INPUT}

ARG S3_OUTPUT
ENV S3_OUTPUT=${S3_OUTPUT}

ARG FILENAME
ENV FILENAME=${FILENAME}

ENV PATH="/usr/local/bin:${PATH}"

# Disable conda by default - using system Python
ENV CONDA_AUTO_ACTIVATE_BASE=false

# Prepare the container environment and directories
RUN mkdir -p ${CODE_PATH}
WORKDIR ${CODE_PATH}

## Setup tzdata, update package index, upgrade packages and install needed software
RUN truncate -s0 /tmp/preseed.cfg && \
    (echo "tzdata tzdata/Areas select America" >> /tmp/preseed.cfg) && \
    (echo "tzdata tzdata/Zones/America select Los_Angeles" >> /tmp/preseed.cfg) && \
    debconf-set-selections /tmp/preseed.cfg && \
    rm -f /etc/timezone /etc/localtime && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true \
    apt-get install -y tzdata
## cleanup of files from setup
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN DEBIAN_FRONTEND=noninteractive apt-get clean && apt-get update -y && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    git \
    unzip \
    wget \
# Building from source for specific version    cmake \ 
    make \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
# Building from source for specific version    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
# Building from source for specific version    libceres-dev \
    libffi-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    gcc-10 g++-10 \
    libssl-dev \
    libbz2-dev \
    libexpat1-dev \
    zlib1g-dev \
    liblzma-dev \
    libreadline-dev \
    libncursesw5-dev \
    libdb-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install and upgrade pip and dependencies with retry logic
RUN pip install --upgrade pip setuptools wheel

# Install Python and make it the default
RUN cd /usr/src \
    && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xzf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations --with-openssl=/usr --prefix=/usr/local --with-system-ffi --with-system-expat --enable-loadable-sqlite-extensions \
    && make -j $(nproc) \
    && make altinstall \
    && cd /usr/src && rm -rf Python-${PYTHON_VERSION}*

# Create explicit symlinks
RUN ln -sf /usr/local/bin/python${PYTHON_VERSION_} /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/local/bin/pip${PYTHON_VERSION_} /usr/local/bin/pip3 \
    && ln -sf /usr/local/bin/pip3 /usr/local/bin/pip

# Update PATH and PYTHONPATH to ensure our Python is used first
ENV PATH=/usr/local/bin:${PATH}
ENV PYTHONPATH=/usr/local/lib/python${PYTHON_VERSION_}/site-packages:/usr/local/lib/python${PYTHON_VERSION_}/dist-packages:${CODE_PATH}
WORKDIR ${CODE_PATH}

# Install typing-extensions first to avoid naming conflict
RUN pip install typing-extensions>=4.10.0

# Install Torch for Cuda
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCH_VISION_VERSION} torchaudio==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_}

# Downgrade setuptools to <v70, otherwise will get ImportError: cannot import name 'packaging' from 'pkg_resources'
RUN pip install "setuptools<70.0.0" ninja boto3 packaging rich natsort trimesh onnxruntime onnxruntime-gpu

# Install tiny-cuda-nn latest stable
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch

# Install Cmake, Eigen, and Ceres for Glomap in single layer
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh \
    && mkdir /opt/cmake \
    && sh cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/opt/cmake --skip-license \
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && rm cmake-${CMAKE_VERSION}-linux-x86_64.sh \
    && git clone --single-branch --depth=1 --branch ${EIGEN_VERSION} https://gitlab.com/libeigen/eigen.git \
    && mkdir -p ${CODE_PATH}/eigen/build \
    && cd ${CODE_PATH}/eigen/build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="/usr/local/bin/eigen/${EIGEN_VERSION}" \
    && make install \
    && cd ${CODE_PATH} \
    && wget https://github.com/ceres-solver/ceres-solver/archive/refs/tags/${CERES_VERSION}.tar.gz \
    && tar -xvzf ${CERES_VERSION}.tar.gz && rm ${CERES_VERSION}.tar.gz && mkdir ceres-bin && cd ceres-bin \
    && cmake -DCUDA=ON -DCUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES}" -DEXPORT_BUILD_DIR=ON ../ceres-solver-${CERES_VERSION} && make -j3 && make install

ENV CMAKE_PREFIX_PATH="/usr/local/bin/eigen/${EIGEN_VERSION}"
ENV Eigen3_DIR="/usr/local/bin/eigen/${EIGEN_VERSION}"
ENV Ceres_DIR=${CODE_PATH}/ceres-bin
WORKDIR ${CODE_PATH}

# Install latest NerfStudio from source to get get full access to repo files for mcmc
RUN git clone https://github.com/nerfstudio-project/nerfstudio.git \
    && cd ${CODE_PATH}/nerfstudio \
    && find . -name "*.py" -exec sed -i 's/torch\.load(\([^,)]*\))/torch.load(\1, weights_only=False)/g' {} \; \
    && find . -name "*.py" -exec sed -i 's/torch\.load(\([^,)]*,\s*[^,)]*\))/torch.load(\1, weights_only=False)/g' {} \; \
    && sed -i '/pycolmap/d' pyproject.toml \
    && sed -i '/# TODO(1480) enable when pycolmap windows wheels are available/,+1d' pyproject.toml \
    && pip install -e . \
    && cd ${CODE_PATH}

# Install latest gsplat from source and copy repo for multi-gpu features
RUN git clone https://github.com/nerfstudio-project/gsplat.git --recursive \
    && cd ${CODE_PATH}/gsplat \
    && sed -i '/os.environ\["MASTER_ADDR"\] = "localhost"/s/^/# /' gsplat/distributed.py \
    && sed -i '/os.environ\["MASTER_PORT"\] = str(_find_free_port())/s/^/# /' gsplat/distributed.py \
    && pip install -e . \
    && pip install -r examples/requirements.txt \
    && cd ${CODE_PATH}

# Install latest Splatfacto-w (in the wild) model and download repo to expose splat export script
# Warning: This will down-grade gsplat to the version it supports. Future work will isolate dep versions
# from the functionality
RUN git clone https://github.com/KevinXu02/splatfacto-w \
    && pip install git+https://github.com/KevinXu02/splatfacto-w

# Install COLMAP from source
RUN wget https://github.com/colmap/colmap/archive/refs/tags/${COLMAP_VERSION}.zip \
    && unzip ${COLMAP_VERSION}.zip && rm ${COLMAP_VERSION}.zip \
    && cd colmap-${COLMAP_VERSION}/ && mkdir build && cd build \
    && cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES}" -DMARCH_NATIVE=OFF && ninja install \
    && cd ${CODE_PATH}/colmap-${COLMAP_VERSION} \
    && sed -i 's/cmake_minimum_required(VERSION [0-9]\+\.[0-9]\+)/cmake_minimum_required(VERSION 3.15)/' CMakeLists.txt \
    && sed -i '/\.value("CUDA", ceres::DenseLinearAlgebraLibraryType::CUDA);/c\          ;' src/pycolmap/estimators/ceres_bindings.cc \
    && python -m pip install . \
    && cd ${CODE_PATH}

# Download vocab tree as fallback if not provided via S3 models archive
RUN git clone https://github.com/ZachMckennedyFWig/ColmapFaissVocabTrees.git \
    && mv ${CODE_PATH}/ColmapFaissVocabTrees/vocab_tree_flickr100K_words32K.bin ${CODE_PATH} \
    && rm -rf ${CODE_PATH}/ColmapFaissVocabTrees

# Install Glomap and AWS CLI in single layer
RUN wget https://github.com/colmap/glomap/archive/refs/tags/${GLOMAP_VERSION}.tar.gz && \
    tar -xvzf ${GLOMAP_VERSION}.tar.gz && cd glomap-${GLOMAP_VERSION} && \
    mkdir build && cd build && \
    cmake .. -GNinja -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES}" -DMARCH_NATIVE=OFF -DCeres_DIR=${CODE_PATH}/ceres-bin && ninja && ninja install \
    && cd ${CODE_PATH} \
    && wget "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -O "awscliv2.zip" \
    && unzip awscliv2.zip && ./aws/install && rm -rf aws awscliv2.zip

# Install utilities and ensure numpy<v2 in order not to error
RUN pip install --no-cache-dir scipy 'numpy<2.0.0' Pillow>=10.3.0 timm einops omegaconf fvcore iopath rembg

# Install Background Removal and SAM2 in single layer with pip cache cleanup
RUN pip cache purge \
    && git clone https://github.com/nadermx/backgroundremover.git \
    && sed -i '53s/^        print/        #print/' ${CODE_PATH}/backgroundremover/backgroundremover/u2net/detect.py \
    && pip install --no-cache-dir -r ${CODE_PATH}/backgroundremover/requirements.txt \
    && git clone https://github.com/facebookresearch/sam2.git \
    && mv ${CODE_PATH}/sam2 ${CODE_PATH}/sam \
    && cd ${CODE_PATH}/sam \
    && sed -i 's/"torch>=2.5.1",/#"torch>=2.5.1",/' setup.py \
    && sed -i 's/"torchvision>=0.20.1",/#"torchvision>=0.20.1",/' setup.py \
    && sed -i '/^    "pillow>=9.4.0",$/a \    "cog>=0.14.12",' setup.py \
    && sed -i 's/"hydra-core>=1.3.2",/"hydra-core==1.3.2",/' setup.py \
    && pip install --no-deps --no-cache-dir -e . \
    && pip install --no-cache-dir hydra-core==1.3.2 cog>=0.14.12 opencv-python matplotlib \
    && cd ${CODE_PATH}

# U2NET models will be provided via S3 models archive
ENV U2NET_PATH=/root/.u2net/u2net.pth
ENV U2NETP_PATH=/root/.u2net/u2netp.pth

# Build NVIDIA 3DGRUT and Attentive Eraser in single layer
# Stable Diffusion XL model will be downloaded from HuggingFace at runtime if using eraser
RUN git clone --recursive https://github.com/nv-tlabs/3dgrut.git \
    && cd ${CODE_PATH}/3dgrut \
    && sed -i 's/checkpoint = torch.load(checkpoint_path)/checkpoint = torch.load(checkpoint_path, weights_only=False)/' threedgrut/render.py \
    && sed -i 's/checkpoint = torch.load(conf.resume)/checkpoint = torch.load(conf.resume, weights_only=False)/' threedgrut/trainer.py \
    && python -m pip install -r requirements.txt \
    && cd ${CODE_PATH} \
    && git clone https://github.com/Anonym0u3/AttentiveEraser.git \
    && cd ${CODE_PATH}/AttentiveEraser \
    && sed -i 's/^torchvision/#torchvision/' requirements.txt \
    && pip install -r requirements.txt \
    && cd ${CODE_PATH}

# Install Node.js, npm, and ffmpeg
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs ffmpeg

# Install PlayCanvas splat-transform globally for SOGS capability
RUN npm install -g @playcanvas/splat-transform

# Install VGGT
RUN git clone https://github.com/facebookresearch/vggt.git
WORKDIR ${CODE_PATH}/vggt
RUN sed -i '/^torch==/d' requirements.txt && \
    sed -i '/^torchvision==/d' requirements.txt && \
    sed -i '/^numpy==/d' requirements.txt && \
    pip install -r requirements.txt
RUN pip install git+https://github.com/cvg/LightGlue.git hydra-core
# Fix pycolmap 3.12 compatibility - replace camera_image_id with camera_id everywhere
RUN find . -name "*.py" -exec sed -i 's/camera_image_id/camera_id/g' {} \; && \
    find . -name "*.py" -exec sed -i 's/id=fidx + 1/image_id=fidx + 1/g' {} \;
WORKDIR ${CODE_PATH}

# Fix version compatibility issues with diffusers and transformers
RUN pip install --force-reinstall --no-cache-dir transformers==4.44.2 diffusers==0.30.3

# The main entrypoint and arguments are added into the container in the State Machine (e.g. ENTRYPOINT [ "python", "main.py" ])
# Copy over application files
RUN mkdir -p ${CODE_PATH}/sfm
RUN mkdir -p ${CODE_PATH}/training
RUN mkdir -p ${CODE_PATH}/spherical
RUN mkdir -p ${CODE_PATH}/segmentation
RUN mkdir -p ${CODE_PATH}/post_processing
RUN mkdir -p ${CODE_PATH}/video_processing
RUN mkdir -p ${CODE_PATH}/image_processing

COPY ./src/main.py                                                  ${CODE_PATH}
COPY ./src/config.json                                              ${CODE_PATH}
COPY ./src/pipeline/pipeline.py                                     ${CODE_PATH}
COPY ./src/pipeline/utils.py                                        ${CODE_PATH}
COPY ./src/pipeline/sfm/extract_poses_imgs.py                       ${CODE_PATH}/sfm
COPY ./src/pipeline/sfm/process_pose_transforms.py                  ${CODE_PATH}/sfm
COPY ./src/pipeline/sfm/update_camera_model.py                      ${CODE_PATH}/sfm
COPY ./src/pipeline/sfm/run_vggt.py                                 ${CODE_PATH}/sfm
COPY ./src/pipeline/training/colmap_to_nerfstudio_cam.py            ${CODE_PATH}/training
COPY ./src/pipeline/spherical/equirectangular_to_perspective.py     ${CODE_PATH}/spherical
COPY ./src/pipeline/spherical/Equirec2Cube.py                       ${CODE_PATH}/spherical
COPY ./src/pipeline/spherical/360ImageConverterforColmap.py         ${CODE_PATH}/spherical
COPY ./src/pipeline/spherical/py360convert.py                       ${CODE_PATH}/spherical
COPY ./src/pipeline/spherical/panorama_sfm.py                       ${CODE_PATH}/spherical
COPY ./src/pipeline/segmentation/remove_object_using_mask.py        ${CODE_PATH}/segmentation
COPY ./src/pipeline/segmentation/remove_background.py               ${CODE_PATH}/segmentation
COPY ./src/pipeline/segmentation/erase_object_using_mask.py         ${CODE_PATH}/segmentation
COPY ./src/pipeline/segmentation/generate_masks_from_filtered.py    ${CODE_PATH}/segmentation
COPY ./src/pipeline/segmentation/remove_background_sam2.py          ${CODE_PATH}/sam
COPY ./src/pipeline/image_processing/filter_blurry_images.py        ${CODE_PATH}/image_processing
COPY ./src/pipeline/video_processing/simple_video_to_images.py      ${CODE_PATH}/video_processing
COPY ./src/pipeline/post_processing/rotate_splat.py                 ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/mirror_splat.py                 ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/gsplat_pt_to_ply.py             ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/refine_bounding_box.py          ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/extract_center_cube.py          ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/refine_center_object.py         ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/convert_ply_to_sog.py           ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/spz                             ${CODE_PATH}/post_processing/spz
COPY ./src/pipeline/post_processing/transform_coordinates.py        ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/clean_ply.py                    ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/extract_video_thumbnail.py      ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/images_to_video.py              ${CODE_PATH}/post_processing

# Build the SPZ tool
WORKDIR ${CODE_PATH}/post_processing/spz/src
RUN mkdir build
WORKDIR ${CODE_PATH}/post_processing/spz/src/build
RUN cmake ..
RUN make
RUN make install
RUN ln -sf ${CODE_PATH}/post_processing/spz/src/build/splat_converter /usr/local/bin/splat_converter

# Verify installed versions match expected versions
RUN echo "=== VERSION VERIFICATION ===" \
    && echo "Expected versions:" \
    && echo "  Python: ${PYTHON_VERSION}" \
    && echo "  CUDA: ${CUDA_VERSION}" \
    && echo "  PyTorch: ${TORCH_VERSION}" \
    && echo "  TorchVision: ${TORCH_VISION_VERSION}" \
    && echo "  CMake: ${CMAKE_VERSION}" \
    && echo "  Eigen: ${EIGEN_VERSION}" \
    && echo "  Ceres: ${CERES_VERSION}" \
    && echo "  COLMAP: ${COLMAP_VERSION}" \
    && echo "  Glomap: ${GLOMAP_VERSION}" \
    && echo "" \
    && echo "Installed versions:" \
    && echo "  Python: $(python --version 2>&1 | cut -d' ' -f2)" \
    && echo "  CUDA: $(nvcc --version 2>/dev/null | grep 'release' | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/' || echo 'Not found')" \
    && echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not found')" \
    && echo "  TorchVision: $(python -c 'import torchvision; print(torchvision.__version__)' 2>/dev/null || echo 'Not found')" \
    && echo "  CMake: $(cmake --version 2>/dev/null | head -1 | cut -d' ' -f3 || echo 'Not found')" \
    && echo "  COLMAP: $(colmap -h 2>/dev/null | head -1 | cut -d' ' -f2 || echo 'Not found')" \
    && echo "  PyColmap: $(python -c 'import pycolmap; print(pycolmap.__version__)' 2>/dev/null || echo 'Not found')" \
    && echo "  Glomap: $(glomap -h 2>/dev/null | head -1 | cut -d' ' -f2 || echo 'Not found')" \
    && echo "  NerfStudio: $(python -c 'import nerfstudio; print(getattr(nerfstudio, "__version__", "installed"))' 2>/dev/null || echo 'Not found')" \
    && echo "  GSplat: $(python -c 'import gsplat; print(gsplat.__version__)' 2>/dev/null || echo 'Not found')" \
    && echo "=== END VERSION VERIFICATION ==="

WORKDIR ${CODE_PATH}
