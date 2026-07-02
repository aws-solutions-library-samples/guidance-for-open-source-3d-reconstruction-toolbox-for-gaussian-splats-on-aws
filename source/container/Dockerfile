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
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

# Pinned versions
ENV PYTHON_VERSION="3.12.0"
ENV PYTHON_VERSION_="3.12"
ENV CUDA_VERSION="13.0"
ENV CUDA_VERSION_="130"
ENV CUDSS_VERSION="0.7.1"
ENV TORCH_VERSION="2.9.1"
ENV TORCH_VISION_VERSION="0.24.1"
ENV CMAKE_VERSION="3.30.3"
ENV NODE_VERSION="22"
ENV EIGEN_VERSION="3.4"
ENV COLMAP_VERSION="4.0.4"
ENV SPLAT_TRANSFORM_VERSION="2.5.2"
ENV MAPANYTHING_VERSION="1.1.2"
ENV SPZ_VERSION="2.1.0"
ENV OPENEXR_VERSION="3.1.5"
ENV OPENIMAGEIO_VERSION="2.5.13.0"

# Pinned versions
ENV CERES_HASH="0ba987acaf9e8674070f116ed624edf017d2b630"
ENV NERFSTUDIO_HASH="50e0e3c70c775e89333256213363badbf074f29d"
ENV GSPLAT_HASH="4e52698e45eaaed929ed3a5065e96a688d085df6"
ENV THREEDGRUT_HASH="9846a28babb2171802907017c44d3adaa927f4e4"
ENV BGREMOVER_HASH="d6278fef141d1669e5fffbbc4cf185edfdc89cf9"
ENV SAM2_HASH="2b90b9f5ceec907a1c18123530e92e794ad901a4"

# Only build for supported GPU architectures
# G4dn=7.5, P4d=8.0, g5=8.6, g6/g6e=8.9
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9"
ENV TCNN_CUDA_ARCHITECTURES=75;80;86;89
ENV NINJA_JOBS=2
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}
ENV OPTIX_FORCE_DEPRECATED_LAUNCHER=1
ENV CUDA_MODULE_LOADING=LAZY

# Expose compute + graphics + utility caps so the NVIDIA Container Toolkit
# projects the host's NVIDIA driver libraries and Vulkan ICD into the
# container at runtime.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

# Ensures that Python outputs everything that's printed directly to the terminal (so logs can be seen in real-time)
ENV PYTHONUNBUFFERED=TRUE

# Ensures Python doesn't try to write .pyc files to disk (useful for improving performance in some scenarios)
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Create code path ENV VAR for container
ARG CODE_PATH
ENV CODE_PATH="/opt/ml/code"

# Create model path ENV VAR for container
ARG MODEL_PATH
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
    libopenimageio-dev \
# Building from source for specific version    libceres-dev \
    libffi-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    gcc-10 g++-10 \
    gfortran \
    libssl-dev \
    libbz2-dev \
    libexpat1-dev \
    zlib1g-dev \
    liblzma-dev \
    libreadline-dev \
    libncursesw5-dev \
    libdb-dev \
    curl \
    vulkan-tools \
    libvulkan1 \
    libvulkan-dev \
    mesa-vulkan-drivers \
    libgl1 \
    libglvnd0 \
    libglx0 \
    libegl1 \
    libxext6 \
    libboost-thread-dev \
    libboost-regex-dev \
    libtiff-dev \
    libjpeg-dev \
    libpng-dev \
    libraw-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python version and make it the default
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

# Install PyTorch for CUDA 13.0 (required after custom Python install)
RUN pip install torch==${TORCH_VERSION}+cu${CUDA_VERSION_} torchvision==${TORCH_VISION_VERSION}+cu${CUDA_VERSION_} torchaudio==${TORCH_VERSION}+cu${CUDA_VERSION_} --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_}

# Downgrade setuptools to <v70, otherwise will get ImportError: cannot import name 'packaging' from 'pkg_resources'
RUN pip install "typing-extensions>=4.10.0" "setuptools<70.0.0" wheel packaging ninja

# Install tiny-cuda-nn latest stable (requires torch to be installed first)
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch

# Install Cmake and Eigen
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh \
    && mkdir /opt/cmake \
    && sh cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/opt/cmake --skip-license \
    && ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && rm cmake-${CMAKE_VERSION}-linux-x86_64.sh \
    && git clone --single-branch --depth=1 --branch ${EIGEN_VERSION} https://gitlab.com/libeigen/eigen.git \
    && mkdir -p ${CODE_PATH}/eigen/build \
    && cd ${CODE_PATH}/eigen/build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="/usr/local/bin/eigen/${EIGEN_VERSION}" \
    && make install \
    && cd ${CODE_PATH}

# Install cuDSS library
RUN wget https://developer.download.nvidia.com/compute/cudss/${CUDSS_VERSION}/local_installers/cudss-local-repo-ubuntu2204-${CUDSS_VERSION}_${CUDSS_VERSION}-1_amd64.deb \
    && dpkg -i cudss-local-repo-ubuntu2204-${CUDSS_VERSION}_${CUDSS_VERSION}-1_amd64.deb \
    && cp /var/cudss-local-repo-ubuntu2204-${CUDSS_VERSION}/cudss-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && dpkg --force-overwrite -i \
        /var/cudss-local-repo-ubuntu2204-${CUDSS_VERSION}/libcudss0-cuda-13_${CUDSS_VERSION}.4-1_amd64.deb \
        /var/cudss-local-repo-ubuntu2204-${CUDSS_VERSION}/libcudss0-dev-cuda-13_${CUDSS_VERSION}.4-1_amd64.deb \
        /var/cudss-local-repo-ubuntu2204-${CUDSS_VERSION}/libcudss0-static-cuda-13_${CUDSS_VERSION}.4-1_amd64.deb \
    && apt-mark hold libcudss0-cuda-13 libcudss0-dev-cuda-13 libcudss0-static-cuda-13 \
    && echo "/usr/lib/x86_64-linux-gnu/libcudss/13" > /etc/ld.so.conf.d/cudss.conf \
    && ldconfig \
    && ln -sf /usr/include/libcudss/13/cudss.h /usr/include/cudss.h \
    && ln -sf /usr/include/libcudss/13/cudss_threading_interface.h /usr/include/cudss_threading_interface.h \
    && ln -sf /usr/include/libcudss/13/cudss_distributed_interface.h /usr/include/cudss_distributed_interface.h \
    && cd ${CODE_PATH}

# Build Ceres with cuDSS support
RUN git clone https://github.com/ceres-solver/ceres-solver.git --recursive \
    && cd ${CODE_PATH}/ceres-solver \
    && git reset --hard ${CERES_HASH} \
    && cd ${CODE_PATH} \
    && mkdir ceres-bin && cd ceres-bin \
    && cmake -DCUDA=ON -DUSE_CUDSS=ON -DCUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES}" -Dcudss_DIR=/usr/lib/x86_64-linux-gnu/libcudss/13/cmake/cudss -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF ../ceres-solver \
    && make -j3 && make install \
    && cd ${CODE_PATH}

ENV CMAKE_PREFIX_PATH="/usr/local/bin/eigen/${EIGEN_VERSION}"
ENV Eigen3_DIR="/usr/local/bin/eigen/${EIGEN_VERSION}"
ENV Ceres_DIR=${CODE_PATH}/ceres-bin
ENV cudss_DIR=/usr/lib/x86_64-linux-gnu/libcudss/13/cmake/cudss

# Install NerfStudio from source to get get full access to repo files
RUN git clone https://github.com/nerfstudio-project/nerfstudio.git \
    && cd ${CODE_PATH}/nerfstudio \
    && git reset --hard ${NERFSTUDIO_HASH} \
    && find . -name "*.py" -exec sed -i 's/torch\.load(\([^,)]*\))/torch.load(\1, weights_only=False)/g' {} \; \
    && find . -name "*.py" -exec sed -i 's/torch\.load(\([^,)]*,\s*[^,)]*\))/torch.load(\1, weights_only=False)/g' {} \; \
    && sed -i '/pycolmap/d' pyproject.toml \
    && sed -i '/gsplat/s/^/# /' pyproject.toml \
    && sed -i '/# TODO(1480) enable when pycolmap windows wheels are available/,+1d' pyproject.toml \
    && sed -i '/@torch_compile()/d' nerfstudio/models/splatfacto.py \
    && pip install -e . \
    && cd ${CODE_PATH}

# Install latest gsplat from source (for splatfacto, splatfacto-big, splatfacto-mcmc)
RUN git clone https://github.com/nerfstudio-project/gsplat.git --recursive \
    && cd ${CODE_PATH}/gsplat \
    && git reset --hard ${GSPLAT_HASH} \
    && sed -i '/os.environ\["MASTER_ADDR"\] = "localhost"/s/^/# /' gsplat/distributed.py \
    && sed -i '/os.environ\["MASTER_PORT"\] = str(_find_free_port())/s/^/# /' gsplat/distributed.py \
    && sed -i 's/point_indices = self\.parser\.point_indices\[image_name\]/point_indices = self.parser.point_indices.get(image_name, np.array([], dtype=np.int32))/' examples/datasets/colmap.py \
    && python setup.py build_ext --inplace \
    && python setup.py develop \
    && pip install -e libs/scene -e libs/stage \
    && pip install --no-build-isolation -r examples/requirements.txt \
    && pip install --force-reinstall imageio tqdm \
    && cd ${CODE_PATH}

COPY ./src/pipeline/training/patch_gsplat.py ${CODE_PATH}/
RUN python3 ${CODE_PATH}/patch_gsplat.py ${CODE_PATH} \
    && rm ${CODE_PATH}/patch_gsplat.py

# Install DN-Splatter (dn-splatter + ags-mesh models)
ENV DN_SPLATTER_HASH="main"
RUN git clone https://github.com/maturk/dn-splatter.git ${CODE_PATH}/dn-splatter \
    && cd ${CODE_PATH}/dn-splatter \
    && pip install -e . --no-deps \
    && pip install --no-cache-dir geffnet scikit-learn PyMCubes pymeshlab open3d \
    && cd ${CODE_PATH}

# Apply compatibility patches to dn-splatter for gsplat 1.5.x
# dn_model_patched.py replaces the cloned dn_model.py with a known-working version
# Cache-bust: bd4b5d4a0175861ed881c84384bb9e30
COPY src/pipeline/training/dn_model_patched.py ${CODE_PATH}/dn-splatter/dn_splatter/dn_model.py
COPY src/pipeline/training/patch_dn_splatter.py ${CODE_PATH}/
RUN python3 ${CODE_PATH}/patch_dn_splatter.py ${CODE_PATH}/dn-splatter \
    && rm ${CODE_PATH}/patch_dn_splatter.py
# Fix normal map loading order: natsorted causes normals to load in a different order
# than images when filenames are numeric (e.g. 1.png, 2.png ... 10.png, 100.png).
# Replacing with sorted() ensures consistent alphabetical/numeric ordering.
RUN sed -i 's/return natsorted(glob.glob(f"{self.normal_save_dir}\/\*.png"))/return sorted(glob.glob(f"{self.normal_save_dir}\/\*.png"))/' \
    ${CODE_PATH}/dn-splatter/dn_splatter/data/normal_nerfstudio.py

# Install Splatfacto-w with gsplat==1.4.0 to separate location
# Install with dependencies to get complete environment
RUN mkdir -p /opt/splatfacto_w_env \
    && git clone https://github.com/KevinXu02/splatfacto-w ${CODE_PATH}/splatfacto-w \
    && pip install --target=/opt/splatfacto_w_env --no-deps gsplat==1.4.0 \
    && pip install --target=/opt/splatfacto_w_env --upgrade git+https://github.com/KevinXu02/splatfacto-w

# Copy and make wrapper scripts executable
COPY src/pipeline/training/run_splatfacto_w_wrapper.py /opt/ml/code/training/run_splatfacto_w_wrapper.py
COPY src/pipeline/training/run_splatfacto_w_export.py /opt/ml/code/training/run_splatfacto_w_export.py
COPY src/pipeline/training/run_splatfacto_w_eval.py /opt/ml/code/training/run_splatfacto_w_eval.py
COPY src/pipeline/training/run_splatfacto_w_render.py /opt/ml/code/training/run_splatfacto_w_render.py
COPY src/pipeline/training/run_dn_splatter_wrapper.py /opt/ml/code/training/run_dn_splatter_wrapper.py
RUN chmod +x /opt/ml/code/training/run_splatfacto_w_wrapper.py \
    /opt/ml/code/training/run_splatfacto_w_export.py \
    /opt/ml/code/training/run_splatfacto_w_eval.py \
    /opt/ml/code/training/run_splatfacto_w_render.py \
    /opt/ml/code/training/run_dn_splatter_wrapper.py

# Install gsplat-specific pycolmap to separate location
RUN mkdir -p /opt/gsplat_pycolmap \
    && pip install --target=/opt/gsplat_pycolmap git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e

# Build OpenEXR + OpenImageIO from source (system libopenimageio-dev is built against old OpenEXR
# and its cmake config pins an incompatible version requirement for COLMAP 4.x)
RUN wget https://github.com/AcademySoftwareFoundation/openexr/archive/refs/tags/v${OPENEXR_VERSION}.tar.gz \
    && tar -xzf v${OPENEXR_VERSION}.tar.gz && rm v${OPENEXR_VERSION}.tar.gz \
    && cmake -S ${CODE_PATH}/openexr-${OPENEXR_VERSION} -B ${CODE_PATH}/openexr-${OPENEXR_VERSION}/build -GNinja \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DBUILD_TESTING=OFF -DOPENEXR_BUILD_TOOLS=OFF \
    && ninja -C ${CODE_PATH}/openexr-${OPENEXR_VERSION}/build -j${NINJA_JOBS} install \
    && rm -rf ${CODE_PATH}/openexr-${OPENEXR_VERSION} && ldconfig
RUN wget https://github.com/AcademySoftwareFoundation/OpenImageIO/archive/refs/tags/v${OPENIMAGEIO_VERSION}.tar.gz \
    && tar -xzf v${OPENIMAGEIO_VERSION}.tar.gz && rm v${OPENIMAGEIO_VERSION}.tar.gz \
    && cmake -S ${CODE_PATH}/OpenImageIO-${OPENIMAGEIO_VERSION} -B ${CODE_PATH}/OpenImageIO-${OPENIMAGEIO_VERSION}/build -GNinja \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DCMAKE_PREFIX_PATH=/usr/local \
       -DOIIO_BUILD_TESTS=OFF -DOIIO_BUILD_TOOLS=OFF -DUSE_PYTHON=OFF -DINSTALL_DOCS=OFF \
    && ninja -C ${CODE_PATH}/OpenImageIO-${OPENIMAGEIO_VERSION}/build -j${NINJA_JOBS} install \
    && rm -rf ${CODE_PATH}/OpenImageIO-${OPENIMAGEIO_VERSION} && ldconfig \
    && find /usr/local/lib/cmake -name 'OpenImageIOConfig.cmake'

# Install COLMAP from source
RUN wget https://github.com/colmap/colmap/archive/refs/tags/${COLMAP_VERSION}.zip \
    && unzip ${COLMAP_VERSION}.zip && rm ${COLMAP_VERSION}.zip \
    && cd colmap-${COLMAP_VERSION}/ && mkdir build && cd build \
    && cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES}" -DMARCH_NATIVE=OFF -DCMAKE_PREFIX_PATH=/usr/local -Dcudss_DIR=/usr/lib/x86_64-linux-gnu/libcudss/13/cmake/cudss && ninja -j${NINJA_JOBS} install \
    && cd ${CODE_PATH}/colmap-${COLMAP_VERSION} \
    && python -m pip install . \
    && cd ${CODE_PATH}

# Download vocab tree as fallback if not provided via S3 models archive
RUN git clone https://github.com/ZachMckennedyFWig/ColmapFaissVocabTrees.git \
    && mv ${CODE_PATH}/ColmapFaissVocabTrees/vocab_tree_flickr100K_words32K.bin ${CODE_PATH} \
    && rm -rf ${CODE_PATH}/ColmapFaissVocabTrees

# Install AWS CLI
RUN wget "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -O "awscliv2.zip" \
    && unzip awscliv2.zip && ./aws/install && rm -rf aws awscliv2.zip

# Install utilities and ensure numpy==1.26.4 for compatibility
RUN pip install --no-cache-dir scipy 'numpy==1.26.4' Pillow>=10.3.0 'timm>=1.0.0' einops omegaconf fvcore iopath rembg rich natsort trimesh onnxruntime onnxruntime-gpu boto3

# Install sharp-frame-extractor separately after scipy is available
RUN pip install --no-cache-dir sharp-frame-extractor

# Install Background Removal and SAM2 in single layer
RUN git clone https://github.com/nadermx/backgroundremover.git \
    && cd ${CODE_PATH}/backgroundremover \
    && git reset --hard ${BGREMOVER_HASH} \
    && sed -i '53s/^        print/        #print/' ${CODE_PATH}/backgroundremover/backgroundremover/u2net/detect.py \
    && sed -i 's/from moviepy import VideoFileClip/from moviepy.editor import VideoFileClip/' ${CODE_PATH}/backgroundremover/backgroundremover/bg.py \
    && pip install --no-cache-dir -r ${CODE_PATH}/backgroundremover/requirements.txt \
    && pip install --force-reinstall --no-cache-dir 'numpy==1.26.4' 'moviepy==1.0.3' \
    && git clone https://github.com/facebookresearch/sam2.git ${CODE_PATH}/sam \
    && cd ${CODE_PATH}/sam \
    && git reset --hard ${SAM2_HASH} \
    && sed -i 's/"torch>=2.5.1",/#"torch>=2.5.1",/' setup.py \
    && sed -i 's/"torchvision>=0.20.1",/#"torchvision>=0.20.1",/' setup.py \
    && sed -i '/^    "pillow>=9.4.0",$/a \    "cog>=0.14.12",' setup.py \
    && sed -i 's/"hydra-core>=1.3.2",/"hydra-core==1.3.2",/' setup.py \
    && pip install --no-deps --no-cache-dir -e . \
    && pip install --no-cache-dir hydra-core==1.3.2 cog>=0.14.12 opencv-python matplotlib \
    && pip install --force-reinstall --no-cache-dir 'numpy==1.26.4' \
    && cd ${CODE_PATH}

# U2NET models will be provided via S3 models archive
ENV U2NET_PATH=/root/.u2net/u2net.pth
ENV U2NETP_PATH=/root/.u2net/u2netp.pth

# Build NVIDIA 3DGRUT and Attentive Eraser in single layer
# Stable Diffusion XL model will be downloaded from HuggingFace at runtime if using eraser
RUN git clone --recursive https://github.com/nv-tlabs/3dgrut.git \
    && cd ${CODE_PATH}/3dgrut \
    && git reset --hard ${THREEDGRUT_HASH} \
    && sed -i 's/checkpoint = torch.load(checkpoint_path)/checkpoint = torch.load(checkpoint_path, weights_only=False)/' threedgrut/render.py \
    && sed -i 's/checkpoint = torch.load(conf.resume)/checkpoint = torch.load(conf.resume, weights_only=False)/' threedgrut/trainer.py \
    && sed -i '/fused-ssim/d' requirements.txt \
    && sed -i 's/^xformers/#xformers/' requirements.txt \
    && python -m pip install --no-build-isolation -r requirements.txt \
    && python -m pip install --no-build-isolation -e . \
    && cd ${CODE_PATH} \
    && git clone https://github.com/Anonym0u3/AttentiveEraser.git \
    && cd ${CODE_PATH}/AttentiveEraser \
    && sed -i 's/^torchvision/#torchvision/' requirements.txt \
    && sed -i 's/^transformers/#transformers/' requirements.txt \
    && sed -i 's/^xformers/#xformers/' requirements.txt \
    && sed -i 's/^diffusers/#diffusers/' requirements.txt \
    && pip install -r requirements.txt \
    && cd ${CODE_PATH} \
    && python3 - <<'PATCH_SCRIPT'
import urllib.request, re

# Download latest SIP pipeline from AttentiveEraser repo
urls = [
    ('https://raw.githubusercontent.com/Anonym0u3/AttentiveEraser/master/pipelines/pipeline_stable_diffusion_xl_attentive_eraser.py',
     '/opt/ml/code/AttentiveEraser/pipelines/pipeline_stable_diffusion_xl_attentive_eraser.py'),
    ('https://raw.githubusercontent.com/Anonym0u3/AttentiveEraser/master/pipelines/pipeline_stable_diffusion_xl_attentive_eraser_inversion.py',
     '/opt/ml/code/AttentiveEraser/pipelines/pipeline_stable_diffusion_xl_attentive_eraser_inversion.py'),
]
for url, path in urls:
    urllib.request.urlretrieve(url, path)
    print(f'Downloaded: {path}')

# Patch DIP inversion pipeline for PyTorch 2.9 + diffusers 0.37+ compatibility
with open('/opt/ml/code/AttentiveEraser/pipelines/pipeline_stable_diffusion_xl_attentive_eraser_inversion.py') as f:
    content = f.read()

# Fix 1: record_list dtype cast to prevent Half/Float mismatch
content = content.replace(
    'latents = (1 - mask) * self._record_list[i] + mask * latents',
    'latents = (1 - mask) * self._record_list[i].to(latents.dtype) + mask * latents'
)
# Fix 2: x0_latents dtype cast
content = content.replace(
    'latents = latents * mask + (1-mask) * self._x0_latents',
    'latents = latents * mask + (1-mask) * self._x0_latents.to(latents.dtype)'
)
# Fix 3: init_image cast - use VAE dtype instead of hardcoded float32
content = content.replace(
    'init_image = init_image.to(dtype=torch.float32)',
    'init_image = init_image.to(dtype=next(self.vae.parameters()).dtype)'
)

with open('/opt/ml/code/AttentiveEraser/pipelines/pipeline_stable_diffusion_xl_attentive_eraser_inversion.py', 'w') as f:
    f.write(content)
print('Patched DIP inversion pipeline for PyTorch 2.9 compatibility')
PATCH_SCRIPT

# Copy and apply patch for 3DGRUT to fix all CUDA extension import issues with v13
COPY ./src/pipeline/training/patch_tracer.py ${CODE_PATH}/
RUN python3 ${CODE_PATH}/patch_tracer.py \
    ${CODE_PATH}/3dgrut/threedgut_tracer/tracer.py \
    ${CODE_PATH}/3dgrut/threedgrt_tracer/tracer.py \
    ${CODE_PATH}/3dgrut/threedgrut/strategy/mcmc.py \
    ${CODE_PATH}/3dgrut/threedgrut/optimizers/__init__.py && \
    echo "Patches applied successfully" && \
    rm ${CODE_PATH}/patch_tracer.py

# Install Node.js, npm, and ffmpeg
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - \
    && apt-get install -y nodejs ffmpeg

# Install PlayCanvas splat-transform globally for SOG capability
RUN npm install -g @playcanvas/splat-transform@${SPLAT_TRANSFORM_VERSION}

# Install Map-Anything with pycolmap 3.10.0 in separate location
RUN mkdir -p /opt/mapanything_pycolmap \
    && pip install --target=/opt/mapanything_pycolmap --no-deps pycolmap==3.10.0 \
    && wget https://github.com/facebookresearch/map-anything/archive/refs/tags/v${MAPANYTHING_VERSION}.tar.gz \
    && tar -xzf v${MAPANYTHING_VERSION}.tar.gz \
    && mv map-anything-${MAPANYTHING_VERSION} map-anything \
    && rm v${MAPANYTHING_VERSION}.tar.gz
WORKDIR ${CODE_PATH}/map-anything
RUN pip install -e .
WORKDIR ${CODE_PATH}

# Install light glue
RUN pip install git+https://github.com/cvg/LightGlue.git hydra-core

# Remove xformers if LightGlue installed it
RUN pip uninstall -y xformers || true

# Reinstall pycolmap after map-anything to ensure ABI compatibility
# Map-anything may have installed incompatible dependencies that conflict with pycolmap
RUN pip install --force-reinstall --no-cache-dir --no-deps pycolmap==4.0.4 

# Install diffusers without torch dependencies to avoid overwriting
RUN pip install --no-cache-dir --no-deps diffusers && pip install --no-cache-dir accelerate safetensors regex requests transformers

# Final NumPy constraint to fix Numba compatibility (requires NumPy < 2.0)
# Pin Pillow < 10.0.0 last to prevent NerfStudio pil_to_numpy breakage (ImagingCore.setimage API change)
RUN pip install --force-reinstall --no-cache-dir 'numpy<2.0,>=1.26.4' 'moviepy==1.0.3' 'Pillow>=9.4.0,<10.0.0'

# Aggressively remove xformers completely after ALL installations
RUN pip uninstall -y xformers || true && \
    rm -rf /usr/local/lib/python3.12/site-packages/xformers* && \
    rm -rf /usr/local/lib/python3.12/site-packages/xFormers* && \
    find /usr/local -name "*xformers*" -exec rm -rf {} + 2>/dev/null || true && \
    python3 -c "import sys; sys.exit(0 if 'xformers' not in sys.modules else 1)" && \
    echo "xformers successfully removed"

# Build the SPZ tool (New version not supported with Gradio 3D Model Viewer)
RUN wget https://github.com/nianticlabs/spz/archive/refs/tags/v${SPZ_VERSION}.tar.gz \
    && tar -xzf v${SPZ_VERSION}.tar.gz \
    && mv spz-${SPZ_VERSION} spz \
    && rm v${SPZ_VERSION}.tar.gz
WORKDIR ${CODE_PATH}/spz
RUN pip install scikit-build-core nanobind && pip install . \
    && mkdir -p build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make ply_to_spz \
    && ln -sf ${CODE_PATH}/spz/build/ply_to_spz /usr/local/bin/ply_to_spz
WORKDIR ${CODE_PATH}

# Install gsplat-specific pycolmap for multi-GPU (MUST BE LAST to avoid being overwritten)
RUN mkdir -p /opt/gsplat_pycolmap \
    && pip install --target=/opt/gsplat_pycolmap --no-deps git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e \
    && rm -rf /opt/gsplat_pycolmap/numpy* /opt/gsplat_pycolmap/scipy* \
    && pip install --target=/opt/gsplat_pycolmap --upgrade 'numpy==1.26.4'

# The main entrypoint and arguments are added into the container in the State Machine (e.g. ENTRYPOINT [ "python", "main.py" ])
# Copy over application files
RUN mkdir -p ${CODE_PATH}/reconstruction
RUN mkdir -p ${CODE_PATH}/training
RUN mkdir -p ${CODE_PATH}/spherical
RUN mkdir -p ${CODE_PATH}/pre_processing/segmentation
RUN mkdir -p ${CODE_PATH}/pre_processing/video
RUN mkdir -p ${CODE_PATH}/post_processing

COPY ./src/entrypoint.sh                                                            ${CODE_PATH}
RUN chmod +x ${CODE_PATH}/entrypoint.sh
COPY ./src/main.py                                                                  ${CODE_PATH}
COPY ./src/config.json                                                              ${CODE_PATH}
COPY ./src/pipeline/pipeline.py                                                     ${CODE_PATH}
COPY ./src/pipeline/utils.py                                                        ${CODE_PATH}
COPY ./src/pipeline/reconstruction/extract_poses_imgs.py                            ${CODE_PATH}/reconstruction
COPY ./src/pipeline/reconstruction/process_pose_transforms.py                       ${CODE_PATH}/reconstruction
COPY ./src/pipeline/reconstruction/update_camera_model.py                           ${CODE_PATH}/reconstruction
COPY ./src/pipeline/reconstruction/convert_cameras_to_pinhole.py                    ${CODE_PATH}/reconstruction
COPY ./src/pipeline/reconstruction/run_map_anything.py                              ${CODE_PATH}/reconstruction
COPY ./src/pipeline/training/colmap_to_nerfstudio_cam.py                            ${CODE_PATH}/training
COPY ./src/pipeline/training/dn_splatter_preprocess.py                               ${CODE_PATH}/training
COPY ./src/pipeline/training/run_gsplat_trainer.sh                                  ${CODE_PATH}/training
RUN chmod +x ${CODE_PATH}/training/run_gsplat_trainer.sh
COPY ./src/pipeline/reconstruction/spherical/panorama_sfm.py                        ${CODE_PATH}/spherical
COPY ./src/pipeline/pre_processing/segmentation/remove_object_using_mask.py         ${CODE_PATH}/pre_processing/segmentation
COPY ./src/pipeline/pre_processing/segmentation/remove_background.py                ${CODE_PATH}/pre_processing/segmentation
COPY ./src/pipeline/pre_processing/segmentation/erase_object_using_mask.py          ${CODE_PATH}/pre_processing/segmentation
COPY ./src/pipeline/pre_processing/segmentation/remove_background_sam2.py           ${CODE_PATH}/sam
COPY ./src/pipeline/pre_processing/video/simple_video_to_images.py                  ${CODE_PATH}/pre_processing/video
COPY ./src/pipeline/pre_processing/video/sharp_video_to_images.py                   ${CODE_PATH}/pre_processing/video
COPY ./src/pipeline/pre_processing/video/rotate_portrait_images.py                  ${CODE_PATH}/pre_processing/video
COPY ./src/pipeline/pre_processing/autoscale_dataset.py                             ${CODE_PATH}/pre_processing
COPY ./src/pipeline/pre_processing/autogroup_images.py                              ${CODE_PATH}/pre_processing
COPY ./src/pipeline/pre_processing/auto_matcher.py                                  ${CODE_PATH}/pre_processing
COPY ./src/pipeline/post_processing/rotate_splat.py                                 ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/mirror_splat.py                                 ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/gsplat_pt_to_ply.py                             ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/crop_splat.py                                   ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/extract_center_cube.py                          ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/refine_center_object.py                         ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/coordinate_systems.py                           ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/remove_ply_comments.py                          ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/clean_point_cloud.py                            ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/extract_video_thumbnail.py                      ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/images_to_video.py                              ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/create_derivative_plys.py                       ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/convert_ply_to_sog.py                           ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/create_collision_mesh.py                        ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/generate_collision.py                           ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/generate_lod.py                                ${CODE_PATH}/post_processing
COPY ./src/pipeline/post_processing/extract_mesh.py                                ${CODE_PATH}/post_processing

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
    && echo "" \
    && echo "Installed versions:" \
    && echo "  Python: $(python --version 2>&1 | cut -d' ' -f2)" \
    && echo "  CUDA: $(nvcc --version 2>/dev/null | grep 'release' | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/' || echo 'Not found')" \
    && echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not found')" \
    && echo "  TorchVision: $(python -c 'import torchvision; print(torchvision.__version__)' 2>/dev/null || echo 'Not found')" \
    && echo "  CMake: $(cmake --version 2>/dev/null | head -1 | cut -d' ' -f3 || echo 'Not found')" \
    && echo "  COLMAP: $(colmap -h 2>/dev/null | head -1 | cut -d' ' -f2 || echo 'Not found')" \
    && echo "  PyColmap (base): $(python -c 'import pycolmap; print(pycolmap.__version__)' 2>/dev/null || echo 'Not found')" \
    && echo "  PyColmap (gsplat): $(PYTHONPATH=/opt/gsplat_pycolmap:$PYTHONPATH python -c 'import pycolmap; print(pycolmap.__version__)' 2>/dev/null || echo 'Not found')" \
    && echo "  NerfStudio: $(python -c 'import nerfstudio; print(getattr(nerfstudio, "__version__", "installed"))' 2>/dev/null || echo 'Not found')" \
    && echo "  GSplat: $(python -c 'import gsplat; print(gsplat.__version__)' 2>/dev/null || echo 'Not found')" \
    && echo "=== END VERSION VERIFICATION ==="


# Hot-patch layer: copied last so all slow dependency layers stay cached during
# active development. These overwrite the versions installed earlier in the build.
# Once stable, the earlier COPY can be removed and this layer kept as the canonical one.
#COPY src/pipeline/training/dn_model_patched.py          ${CODE_PATH}/dn-splatter/dn_splatter/dn_model.py
#COPY src/pipeline/training/dn_splatter_preprocess.py    ${CODE_PATH}/training/dn_splatter_preprocess.py
#COPY src/main.py                                        ${CODE_PATH}/main.py
