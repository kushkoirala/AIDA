# GPU-Accelerated Flight Dynamics Simulator
# 
# Based on NVIDIA NGC PyTorch container for optimal GPU performance
#
# Build:
#   docker build -t flight-dynamics .
#
# Run:
#   docker run --gpus all -it flight-dynamics
#
# Author: Kushal Koirala

# Use NVIDIA NGC PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL maintainer="Kushal Koirala <kush.koirala@gmail.com>"
LABEL description="GPU-Accelerated 6-DOF Flight Dynamics Simulator"
LABEL version="1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install additional dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    cupy-cuda12x \
    numpy \
    matplotlib \
    gymnasium \
    stable-baselines3

# Create working directory
WORKDIR /workspace/flight-dynamics

# Copy source code
COPY cuda/ ./cuda/
COPY python/ ./python/
COPY README.md ./

# Build CUDA library
WORKDIR /workspace/flight-dynamics/cuda
RUN make CUDA_ARCH=sm_80

# Set library path
ENV LD_LIBRARY_PATH=/workspace/flight-dynamics/cuda:${LD_LIBRARY_PATH}

# Install Python package
WORKDIR /workspace/flight-dynamics/python
RUN pip install -e .

# Return to main directory
WORKDIR /workspace/flight-dynamics

# Default command: run benchmarks
CMD ["python", "-c", "from flight_dynamics import benchmark; benchmark(10000, 1000)"]
