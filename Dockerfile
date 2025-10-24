# Dockerfile for PIM Matmul Benchmarks Development and Testing
# Ubuntu 22.04 with UPMEM SDK, build tools, and Python environment

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y build-essential doxygen \
    python3 python3-pip python3.7-dev \
    wget sudo git pkg-config gdb gdb-multiarch \
    libelf-dev libnuma-dev libgomp1 && \
    apt-get clean

# Install Python dependencies
RUN pip3 install --upgrade pip && pip3 install pyyaml psutil

# Set up workdir and copy project
WORKDIR /workspace
COPY . /workspace

# Install UPMEM SDK
RUN tar -xzf /workspace/lib/upmem.tar.gz -C /opt/

# Set environment variables for UPMEM SDK
ENV PKG_CONFIG_PATH="/opt/upmem-2023.2.0-Linux-x86_64/share/pkgconfig"
ENV PATH="/opt/upmem-2023.2.0-Linux-x86_64/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/upmem-2023.2.0-Linux-x86_64/lib"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/opt/upmem-2023.2.0-Linux-x86_64/lib"

SHELL ["/bin/bash", "-c"]

# Source the upmem_env.sh script
RUN source /opt/upmem-2023.2.0-Linux-x86_64/upmem_env.sh simulator

# Source the environment script
RUN source /workspace/source.me

