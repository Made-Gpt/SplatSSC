# Start from the NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables to avoid interaction during installations
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Singapore

# Install wget and curl
RUN apt-get -y update
RUN apt-get install -y --no-install-recommends \
    sudo \
    wget \
    curl \
    vim \
    git \
    cmake \
    libglm-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    openssh-server \
    openssh-client \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# for snap docker
# RUN ln -s /var/lib/snapd/hostfs/usr/bin/nvidia-smi /usr/local/bin/nvidia-smi

# Install Miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo PATH="/root/miniconda3/bin":$PATH >> .bashrc \
    && exec bash

# Add conda to PATH
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Create a conda environment
RUN conda create -y -n splatssc python=3.9 cmake=3.14.0 && \
    conda clean -afy

# Activate the environment
SHELL ["conda", "run", "-n", "splatssc", "/bin/bash", "-c"]

# Install PyTorch with CUDA support
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install other packages
RUN pip install numpy==1.26.4
RUN pip install open3d h5py einops
RUN pip install openmim==0.3.9
RUN mim install mmcv==2.0.1
RUN mim install mmdet==3.0.0
RUN mim install mmsegmentation==1.2.2
RUN mim install mmdet3d==1.1.1

RUN pip install spconv-cu117 timm torch-scatter

# Install additional requirements from requirements.txt
# COPY requirements.txt ./
# RUN pip install -r requirements.txt

# Set working directory
WORKDIR /SplatSSC

# Copy your application code
COPY . /SplatSSC

# Set the default command
CMD ["bash"]
