#!/usr/bin/env bash

set -eux

source /agent/.venv/bin/activate
# uvicorn directly, not `fastapi run`: the app is a Starlette app, and one process keeps
# a single async task registry for /ping to answer from.
exec opentelemetry-instrument uvicorn swe_agent_server.app:app --port 8080 --host 0.0.0.0
