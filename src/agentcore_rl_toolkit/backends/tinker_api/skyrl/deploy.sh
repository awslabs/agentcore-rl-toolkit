#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"
export TMPDIR="${TMPDIR:-${TMP:-${TEMP:-/tmp}}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$TMPDIR/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$TMPDIR/skyrl-endpoint-client}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$TMPDIR/skyrl-endpoint-pycache}"
mkdir -p "$TMPDIR"
# The same environment can run lifecycle commands: ./deploy.sh sky stop CLUSTER.
if [[ "${1:-}" == sky ]]; then
  exec uv run --project "$EXAMPLE_DIR" --frozen "$@"
fi
exec uv run --project "$EXAMPLE_DIR" --frozen python "$EXAMPLE_DIR/deploy.py" "$@"
