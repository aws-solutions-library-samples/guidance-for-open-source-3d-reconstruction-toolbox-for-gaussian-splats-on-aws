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
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# Only build for supported GPU architectures
#P3=7.0, G4dn=7.5, P4d=8.0, g5=8.6, g6/g6e=8.9
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9"
ENV TCNN_CUDA_ARCHITECTURES=70;75;80;86;89
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib
ENV PATH=/usr/local/cuda-11.8/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}

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

# Install pre-dependencies
RUN apt-get update -y
RUN apt-get install wget -y
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libxnvctrl0_575.57.08-0ubuntu1_amd64.deb
RUN chmod +x ${CODE_PATH}/libxnvctrl0_575.57.08-0ubuntu1_amd64.deb
RUN apt install ./libxnvctrl0_575.57.08-0ubuntu1_amd64.deb -y
RUN rm -rf ${CODE_PATH}/libxnvctrl0_575.57.08-0ubuntu1_amd64.deb

RUN DEBIAN_FRONTEND=noninteractive apt-get clean && apt-get update -y && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    git \
    unzip \
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
    && rm -rf /var/lib/apt/lists/*

# Install and upgrade pip and dependencies
RUN pip install --upgrade pip

# Install Python v3.11 and make it the default
RUN cd /usr/src \
    && wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz \
    && tar -xzf Python-3.11.0.tgz \
    && cd Python-3.11.0 \
    && ./configure --enable-optimizations --with-openssl=/usr --prefix=/usr/local --with-system-ffi --with-system-expat --enable-loadable-sqlite-extensions \
    && make -j $(nproc) \
    && make altinstall \
    && cd /usr/src && rm -rf Python-3.11.0*

# Create explicit symlinks
RUN ln -sf /usr/local/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip3 \
    && ln -sf /usr/local/bin/pip3 /usr/local/bin/pip

# Update PATH and PYTHONPATH to ensure our Python is used first
ENV PATH=/usr/local/bin:${PATH}
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages:/usr/local/lib/python3.11/dist-packages:${CODE_PATH}
WORKDIR ${CODE_PATH}

# Install Torch for Cuda 11.8
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Downgrade setuptools to <v70, otherwise will get ImportError: cannot import name 'packaging' from 'pkg_resources'
RUN pip install "setuptools<70.0.0" ninja boto3 packaging rich natsort trimesh onnxruntime onnxruntime-gpu

# Install nerfacc v0.5.2 and tiny-cuda-nn latest stable
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch

# Install Cmake 3.30.3 for Glomap
RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.3/cmake-3.30.3-linux-x86_64.sh \
    && mkdir /opt/cmake \
    && sh cmake-3.30.3-linux-x86_64.sh --prefix=/opt/cmake --skip-license \
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && rm cmake-3.30.3-linux-x86_64.sh
WORKDIR ${CODE_PATH}

# Install Eigen 3.4.0 for Glomap
RUN git clone --single-branch --depth=1 --branch "3.4" https://gitlab.com/libeigen/eigen.git
RUN mkdir -p ${CODE_PATH}/eigen/build
WORKDIR ${CODE_PATH}/eigen/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="/usr/local/bin/eigen/3.4"
RUN make install
ENV CMAKE_PREFIX_PATH="/usr/local/bin/eigen/3.4"
ENV Eigen3_DIR="/usr/local/bin/eigen/3.4"
WORKDIR ${CODE_PATH}

# Install Ceres v2.0 for Glomap
RUN wget https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.0.0.tar.gz
RUN tar -xvzf 2.0.0.tar.gz && rm 2.0.0.tar.gz && mkdir ceres-bin && cd ceres-bin &&\
    cmake -DEXPORT_BUILD_DIR=ON ../ceres-solver-2.0.0 && make -j3 && make install
ENV Ceres_DIR=${CODE_PATH}/ceres-bin
WORKDIR ${CODE_PATH}

# Install COLMAP v3.12.0 from source
RUN wget https://github.com/colmap/colmap/archive/refs/tags/3.12.0.zip \
    && unzip 3.12.0.zip && rm 3.12.0.zip
RUN bash -c "cd colmap-3.12.0/ && mkdir build && cd build && cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES='${TCNN_CUDA_ARCHITECTURES}' -DMARCH_NATIVE=OFF && ninja install"
WORKDIR ${CODE_PATH}

# Download the vocab tree which is used by Colmap - FAISS converted
#RUN wget https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin
RUN git clone https://github.com/ZachMckennedyFWig/ColmapFaissVocabTrees.git
RUN mv ${CODE_PATH}/ColmapFaissVocabTrees/vocab_tree_flickr100K_words32K.bin ${CODE_PATH}

# Install Glomap v1.0.0 (see https://github.com/colmap/glomap/issues/55#issuecomment-2437082982)
RUN wget https://github.com/colmap/glomap/archive/refs/tags/1.0.0.tar.gz && \
    tar -xvzf 1.0.0.tar.gz && cd glomap-1.0.0 && \
    sed -i 's/GIT_TAG           b3691b791bcedccd5451621b2275a1df0d9dcdeb/GIT_TAG           master/' cmake/FindDependencies.cmake && \
    grep "GIT_TAG" cmake/FindDependencies.cmake && \
    mkdir build && cd build && \
    cmake .. -GNinja -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURES} -DCeres_DIR=${CODE_PATH}/ceres-bin && ninja && ninja install
WORKDIR ${CODE_PATH}

# Install latest NerfStudio from source to get get full access to repo files for mcmc
RUN git clone https://github.com/nerfstudio-project/nerfstudio.git
WORKDIR ${CODE_PATH}/nerfstudio
RUN pip install -e .
WORKDIR ${CODE_PATH}

# Install latest gsplat from source and copy repo for multi-gpu features
RUN git clone https://github.com/nerfstudio-project/gsplat.git --recursive
WORKDIR ${CODE_PATH}/gsplat
#RUN pip install -e .
RUN pip install -r examples/requirements.txt
WORKDIR ${CODE_PATH}
RUN pip install git+https://github.com/nerfstudio-project/gsplat.git

# Install latest Splatfacto-w (in the wild) model and download repo to expose splat export script
# Warning: This will down-grade gsplat to the version it supports. Future work will isolate dep versions
# from the functionality
RUN git clone https://github.com/KevinXu02/splatfacto-w
RUN pip install git+https://github.com/KevinXu02/splatfacto-w

# Install AWS CLI v2 latest
RUN wget "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -O "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws

# Install utilities and ensure numpy<v2 in order not to error
RUN pip install --no-cache-dir scipy 'numpy<2.0.0' Pillow>=10.3.0 timm einops omegaconf fvcore iopath

# Install the Background Removal Classic Tool
RUN git clone https://github.com/nadermx/backgroundremover.git
RUN pip install -r ${CODE_PATH}/backgroundremover/requirements.txt

# Install the SAM2 Segmentation Model and Code
RUN git clone https://github.com/facebookresearch/sam2.git
RUN mv ${CODE_PATH}/sam2 ${CODE_PATH}/sam
WORKDIR ${CODE_PATH}/sam

# Keep the NerfStudio torch version but make sure we install all required dependencies
RUN sed -i 's/"torch>=2.5.1",/#"torch>=2.5.1",/' setup.py && \
    sed -i 's/"torchvision>=0.20.1",/#"torchvision>=0.20.1",/' setup.py && \
    sed -i '/^    "pillow>=9.4.0",$/a \    "cog>=0.14.12",' setup.py && \
    sed -i 's/"hydra-core>=1.3.2",/"hydra-core==1.3.2",/' setup.py

# Install SAM2 dependencies
RUN pip install -e .
WORKDIR ${CODE_PATH}

# Build NVIDIA 3DGRUT
RUN git clone --recursive https://github.com/nv-tlabs/3dgrut.git
WORKDIR ${CODE_PATH}/3dgrut
RUN python -m pip install -r requirements.txt
WORKDIR ${CODE_PATH}

# The main entrypoint and arguments are added into the container in the State Machine (e.g. ENTRYPOINT [ "python", "main.py" ])
# Copy over application files
RUN mkdir -p ${CODE_PATH}/video_processing
RUN mkdir -p ${CODE_PATH}/image_processing
RUN mkdir -p ${CODE_PATH}/sfm
RUN mkdir -p ${CODE_PATH}/training
RUN mkdir -p ${CODE_PATH}/spherical
RUN mkdir -p ${CODE_PATH}/segmentation
RUN mkdir -p ${CODE_PATH}/post_processing

COPY ./src/main.py                                              ${CODE_PATH}
COPY ./src/config.json                                          ${CODE_PATH}
COPY ./src/pipeline/pipeline.py                                 ${CODE_PATH}
COPY ./src/pipeline/image_processing/filter_blurry_images.py    ${CODE_PATH}/image_processing
COPY ./src/pipeline/video_processing/simple_video_to_images.py  ${CODE_PATH}/video_processing
COPY ./src/pipeline/sfm/extract_poses_imgs.py                   ${CODE_PATH}/sfm
COPY ./src/pipeline/sfm/process_pose_transforms.py              ${CODE_PATH}/sfm
COPY ./src/pipeline/sfm/update_camera_model.py                  ${CODE_PATH}/sfm
COPY ./src/pipeline/training/colmap_to_nerfstudio_cam.py        ${CODE_PATH}/training
COPY ./src/pipeline/spherical/equirectangular_to_perspective.py ${CODE_PATH}/spherical
COPY ./src/pipeline/spherical/Equirec2Cube.py                   ${CODE_PATH}/spherical
COPY ./src/pipeline/spherical/360ImageConverterforColmap.py     ${CODE_PATH}/spherical
COPY ./src/pipeline/spherical/py360convert.py                   ${CODE_PATH}/spherical
COPY ./src/pipeline/segmentation/remove_object_using_mask.py    ${CODE_PATH}/segmentation
COPY ./src/pipeline/segmentation/remove_background.py           ${CODE_PATH}/segmentation
COPY ./src/pipeline/post_processing/rotate_splat.py             ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/mirror_splat.py             ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/gsplat_pt_to_ply.py         ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/spz                         ${CODE_PATH}/post_processing/spz
COPY ./src/pipeline/segmentation/remove_background_sam2.py      ${CODE_PATH}/sam

# Build the SPZ tool
WORKDIR ${CODE_PATH}/post_processing/spz/src
RUN mkdir build
WORKDIR ${CODE_PATH}/post_processing/spz/src/build
RUN cmake ..
RUN make
RUN make install
RUN ln -sf ${CODE_PATH}/post_processing/spz/src/build/splat_converter /usr/local/bin/splat_converter

WORKDIR ${CODE_PATH}
