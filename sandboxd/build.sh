#!/usr/bin/env bash
# Cross-compile agentcore-sandboxd as a static Linux binary.
#
# Usage:
#   ./build.sh [--arch arm64|amd64] [--stage <dir>]
#
#   --arch   Target architecture (default: arm64 — AgentCore Runtime is arm64-only today).
#   --stage  Additionally copy the built binary into <dir> (e.g. an example's Docker
#            build context).
#
# Works on any host (including x86): Go cross-compiles natively, and if no Go
# toolchain is installed the build runs inside a golang container instead (the
# container runs natively on the host arch and cross-compiles — no qemu needed).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="arm64"
STAGE_DIR=""
GO_IMAGE="public.ecr.aws/docker/library/golang:1.24"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      ARCH="$2"
      shift 2
      ;;
    --arch=*)
      ARCH="${1#*=}"
      shift
      ;;
    --stage)
      STAGE_DIR="$2"
      shift 2
      ;;
    --stage=*)
      STAGE_DIR="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--arch arm64|amd64] [--stage <dir>]" >&2
      exit 1
      ;;
  esac
done

if [[ "$ARCH" != "arm64" && "$ARCH" != "amd64" ]]; then
  echo "Unsupported --arch '$ARCH' (expected arm64 or amd64)" >&2
  exit 1
fi

OUTPUT="dist/agentcore-sandboxd-linux-${ARCH}"

if command -v go >/dev/null 2>&1; then
  echo "Building with local Go toolchain ($(go version))..."
  (cd "$SCRIPT_DIR" && CGO_ENABLED=0 GOOS=linux GOARCH="$ARCH" go build -ldflags "-s -w" -o "$OUTPUT" .)
elif command -v docker >/dev/null 2>&1; then
  echo "No local Go toolchain found; building inside $GO_IMAGE..."
  docker run --rm -v "$SCRIPT_DIR":/src -w /src \
    -e CGO_ENABLED=0 -e GOOS=linux -e GOARCH="$ARCH" -e GOFLAGS=-buildvcs=false \
    "$GO_IMAGE" go build -ldflags "-s -w" -o "$OUTPUT" .
else
  echo "Neither 'go' nor 'docker' found on PATH." >&2
  echo "Install Go (https://go.dev/dl/) or Docker, then re-run." >&2
  exit 1
fi

echo "Built: $SCRIPT_DIR/$OUTPUT"

if [[ -n "$STAGE_DIR" ]]; then
  cp "$SCRIPT_DIR/$OUTPUT" "$STAGE_DIR/"
  echo "Staged: $STAGE_DIR/$(basename "$OUTPUT")"
fi
