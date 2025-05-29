# Pre-reqs:
#  1. podman (shown below) or just docker
#     $ dnf install -y podman podman-docker
#  2. NVIDIA container toolkit
#     $ dnf install -y nvidia-container-toolkit
#
# Build:
#  $ cd ~/monarch
#  $ export TAG_NAME=$USER-dev
#  $ docker build --network=host \
#     -t monarch:$TAG_NAME \
#     -f Dockerfile .
#
# Build (with http proxy):
#  $ docker build --network=host \
#     --build-arg=http_proxy=$http_proxy \
#     --build-arg=https_proxy=$https_proxy \
#     -t monarch:$TAG_NAME \
#     -f Dockerfile .
#
ARG http_proxy
ARG https_proxy

FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel
WORKDIR /monarch

# export http proxy env vars if build-args are provided
RUN if [ -n "${http_proxy}" ]; then export http_proxy=${http_proxy}; fi && \
    if [ -n "${https_proxy}" ]; then export https_proxy=${https_proxy}; fi

# Install native dependencies
RUN apt-get update -y && \
    apt-get -y install curl clang liblzma-dev libunwind-dev

# Install Rust
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Install Python build deps
RUN pip install --no-cache-dir setuptools-rust

# Install Python deps as a separate layer to avoid rebuilding if deps do not change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install monarch
COPY . .
# currently monarch/pyproject.toml uses meta-internal build-backend, just use setup.py
RUN rm pyproject.toml
RUN cargo install --path monarch_hyperactor
RUN pip install .
