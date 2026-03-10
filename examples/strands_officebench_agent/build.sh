#!/bin/bash
set -e

# Build script for OfficeBench agent Docker image.
# Copies OfficeBench app scripts into the build context, then invokes
# the shared build_docker_image_and_push_to_ecr.sh script.
#
# Usage:
#   cd <repo-root>
#   ./examples/strands_officebench_agent/build.sh --tag=dev
#
# Requires:
#   OFFICEBENCH_DIR env var pointing to OfficeBench repo
#   (default: ~/OfficeBench)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OFFICEBENCH_DIR="${OFFICEBENCH_DIR:-${HOME}/OfficeBench}"
APPS_SRC="${OFFICEBENCH_DIR}/apps"
APPS_DST="${SCRIPT_DIR}/apps"

if [ ! -d "$APPS_SRC" ]; then
    echo "Error: OfficeBench apps not found at ${APPS_SRC}"
    echo "Set OFFICEBENCH_DIR env var to the OfficeBench repository path"
    exit 1
fi

# Copy OfficeBench apps into the build context
echo "Copying OfficeBench apps from ${APPS_SRC} -> ${APPS_DST}"
rm -rf "${APPS_DST}"
cp -r "${APPS_SRC}" "${APPS_DST}"

# Run the shared build script
cd "${REPO_ROOT}"
./scripts/build_docker_image_and_push_to_ecr.sh \
    --dockerfile=examples/strands_officebench_agent/Dockerfile \
    --context=examples/strands_officebench_agent \
    --additional-context=toolkit=. \
    "$@"

# Clean up copied apps
rm -rf "${APPS_DST}"
echo "Cleaned up temporary apps directory"
