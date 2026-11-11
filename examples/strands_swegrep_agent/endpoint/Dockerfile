# Toolchain image for the local SkyRL Tinker endpoint: Ubuntu 22.04 (glibc 2.35, which SkyRL's pinned
# vllm-router wheel needs) with CUDA 13.0 at /usr/local/cuda-13.0, on any host with Docker and NVIDIA GPUs.
# The image holds only the OS toolchain; SkyRL source, its venv, caches and state live on
# bind-mounted host paths (see docker.sh) so rebuilding the image never reinstalls SkyRL.
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04@sha256:1c517d4fb96528c8999e14e6e6b16d7e2ff9ae8d194fd788d60fdc88e693e982

ARG UID=1000
ARG GID=1000
ARG USERNAME=skyrl
ARG HOME_DIR=/home/skyrl
ARG UV_VERSION=0.9.4

RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       git curl ca-certificates build-essential cmake ninja-build libnuma-dev kmod iproute2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Same user, UID and home directory as the host, so bind-mounted host paths resolve identically inside.
RUN groupadd -o -g "${GID}" skyrl && useradd -o -m -u "${UID}" -g "${GID}" -d "${HOME_DIR}" -s /bin/bash "${USERNAME}"
USER ${USERNAME}
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh \
    && mkdir -p "${HOME_DIR}/.cache/huggingface/hub"   # user-owned parents for the model bind mount
ENV PATH=${HOME_DIR}/.local/bin:${PATH}
WORKDIR ${HOME_DIR}
