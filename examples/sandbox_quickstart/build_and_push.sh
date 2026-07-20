#!/usr/bin/env bash
# Convenience wrapper: build the sandboxd binary, stage it here, and build/push
# the sandbox image to ECR in one command.
#
# Usage:
#   ./build_and_push.sh [tag]      # tag defaults to "sandbox-quickstart"
#
# Requires a .env at the repo root with ECR_REPO_NAME, AWS_REGION, AWS_ACCOUNT
# (see .env.example) — same as the other examples.
#
# NOTE: temporary scaffolding. This script owns no build or ECR logic — it only
# sequences ../../sandboxd/build.sh (binary) and
# ../../scripts/build_docker_image_and_push_to_ecr.sh (image + ECR plumbing).
# A future phase moves image wrapping into the SDK (SandboxEnv.from_image()),
# at which point this wrapper goes away.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TAG="${1:-sandbox-quickstart}"

"$REPO_ROOT/sandboxd/build.sh" --stage "$SCRIPT_DIR"

cd "$REPO_ROOT"
./scripts/build_docker_image_and_push_to_ecr.sh \
  --dockerfile=examples/sandbox_quickstart/Dockerfile \
  --tag="$TAG" \
  --context=examples/sandbox_quickstart
