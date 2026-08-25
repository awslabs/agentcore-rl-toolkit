#!/usr/bin/env bash
# Build ONE harness image and push to ECR as $HARNESS_REPO:<tag> (repo from .env).
#
#   build_push.sh strands           -> $HARNESS_REPO:strands
#   build_push.sh claude_code       -> $HARNESS_REPO:claude-code
#   build_push.sh all               -> both
#
# Each harness is a (Dockerfile + harness.py) pair in its own subdirectory; the
# build stages in the toolkit source and the local harbor_sandbox package (both
# imported by harness.py). All build for linux/arm64 (the default serverless
# MicroVM runtime); qemu binfmt handles the cross-build on this x86 host — the
# images are pure Python so emulated builds are fine.
#
# Optional 3rd arg overrides the image tag (default: the agent name). Use a
# versioned tag (e.g. claude-code-v1) to publish a NEW harness build WITHOUT
# clobbering the tag a live runtime is pulling from.
#   build_push.sh claude_code linux/arm64 claude-code-v1  -> $HARNESS_REPO:claude-code-v1
set -euo pipefail

HARNESS=${1:?usage: build_push.sh <strands|claude_code|all> [platform] [tag]}
PLATFORM=${2:-linux/arm64}
TAG_OVERRIDE=${3:-}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # examples/sandbox_harbor/harness
EXAMPLE="$(cd "$HERE/.." && pwd)"                          # examples/sandbox_harbor
TOOLKIT_SRC="$(cd "$HERE/../../.." && pwd)"                # repo root (this example lives inside it)

# Same .env the Python config reads (no hand-synced duplicate). Full-line comments
# + KEY=value only, so it sources cleanly. REPO mirrors config's HARNESS_REPO.
ENV_FILE="$EXAMPLE/harbor_sandbox/.env"
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE (copy harbor_sandbox/.env.example)"; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"
REPO="$HARNESS_REPO"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

if [ "$HARNESS" = "all" ]; then
  for h in strands claude_code; do
    "$0" "$h" "$PLATFORM"
  done
  exit 0
fi
[ -d "$HERE/$HARNESS" ] || { echo "unknown harness '$HARNESS'"; exit 1; }

TAG="${TAG_OVERRIDE:-${HARNESS//_/-}}"   # default claude_code -> claude-code; override for versioned builds
IMAGE="${REGISTRY}/${REPO}:${TAG}"
CTX="$HERE/$HARNESS"

# Stage clean copies of the toolkit source + the local harbor_sandbox package
# into the build context (no venv/git/caches). harbor_sandbox drops the wrapper/
# sandboxd binaries — those are for the image builder, not the harness runtime.
# NB: rsync ignores .gitignore, so the git-ignored .env IS copied in — intentional:
# it's how the harness container gets the real account/region.
echo ">> staging toolkit source into $HARNESS/_toolkit"
rm -rf "$CTX/_toolkit" "$CTX/harbor_sandbox"
mkdir -p "$CTX/_toolkit"
rsync -a --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
      --exclude '*.egg-info' --exclude 'examples' \
      "$TOOLKIT_SRC/" "$CTX/_toolkit/"
rsync -a --exclude '__pycache__' --exclude 'wrapper' \
      "$EXAMPLE/harbor_sandbox/" "$CTX/harbor_sandbox/"

# Ensure the ECR repo exists.
aws ecr describe-repositories --region "$REGION" --repository-names "$REPO" >/dev/null 2>&1 \
  || aws ecr create-repository --region "$REGION" --repository-name "$REPO" >/dev/null

echo ">> docker login to ECR"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

echo ">> building $IMAGE ($PLATFORM)"
docker build --platform "$PLATFORM" -t "$IMAGE" "$CTX"

echo ">> pushing $IMAGE"
docker push "$IMAGE"

rm -rf "$CTX/_toolkit" "$CTX/harbor_sandbox"
echo ">> done: $IMAGE"
