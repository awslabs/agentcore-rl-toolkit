#!/usr/bin/env bash
# Run an endpoint script (setup.sh / serve.sh / any command) inside the Ubuntu 22.04 + CUDA 13.0
# toolchain container. Host paths are mounted at identical locations, so environment.sh is
# valid both inside and outside the container. Host networking: the endpoint listens on
# $TINKER_PORT of this host directly (no auth/TLS - keep the port closed in the security group).
#   bash endpoint/docker.sh build
#   bash endpoint/docker.sh bash setup.sh
#   bash endpoint/docker.sh bash serve.sh
# With DETACH_LOG=<file>, the container runs in the background (Docker keeps it alive; stop it with
# `docker stop $SKYRL_CONTAINER`) and its output is appended to that file.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/environment.sh"
MODEL_DIR="$(realpath "$MODEL_DIR")"
IMAGE="${SKYRL_IMAGE:-swegrep-skyrl-endpoint:cu130-ubuntu2204}"
NAME="${SKYRL_CONTAINER:-skyrl-endpoint}"

BUILD_ARGS=(--build-arg UID="$(id -u)" --build-arg GID="$(id -g)" --build-arg USERNAME="$(id -un)" --build-arg HOME_DIR="$HOME")
if [ "${1:-}" = build ]; then
  exec docker build -t "$IMAGE" "${BUILD_ARGS[@]}" "$HERE"
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" "${BUILD_ARGS[@]}" "$HERE"

mkdir -p "$SKYRL_HOME" "$SKYRL_STATE_DIR"
MODEL_REPO_DIR="$(dirname "$(dirname "$MODEL_DIR")")"   # .../models--Qwen--Qwen3.5-9B (snapshots symlink into blobs)
# huggingface_hub >= 1.x keeps large blobs in a hub-level shared store (hub/blobs/xx/...), which the
# repo's blobs/ symlink into; mount it too or the weights dangle inside the container.
SHARED_BLOBS="$(dirname "$MODEL_REPO_DIR")/blobs"
EXTRA_MOUNTS=(); [ -d "$SHARED_BLOBS" ] && EXTRA_MOUNTS=(-v "$SHARED_BLOBS:$SHARED_BLOBS:ro")
# --rm: the container is disposable; all state is on the mounted host paths. --init reaps the server's
# child processes, so `docker stop` cannot leave a zombie container behind.
OPTS=(--rm --init --name "$NAME"
  --gpus all --network host --ipc host --shm-size 64g
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576
  -e MODEL_ID -e MODEL_REVISION -e TINKER_PORT -e TINKER_HOST -e SKYRL_COMMIT -e HF_TOKEN
  -e MODEL_DIR -e SKYRL_HOME -e SKYRL_STATE_DIR -e BACKEND_CONFIG_FILE -e HOST
  -v "$HERE:$HERE"
  -v "$SKYRL_HOME:$SKYRL_HOME"
  -v "$SKYRL_STATE_DIR:$SKYRL_STATE_DIR"
  -v "$MODEL_REPO_DIR:$MODEL_REPO_DIR:ro" "${EXTRA_MOUNTS[@]}"
  -w "$HERE")
if [ -n "${DETACH_LOG:-}" ]; then
  docker run -d "${OPTS[@]}" "$IMAGE" "$@" >/dev/null
  # Follows the container's output until it exits.
  setsid nohup docker logs -f "$NAME" >>"$DETACH_LOG" 2>&1 </dev/null &
  exit 0
fi
TTY=(); [ -t 0 ] && TTY=(-it)
exec docker run "${TTY[@]}" "${OPTS[@]}" "$IMAGE" "$@"
