#!/usr/bin/env bash

set -eux

source /agent/.venv/bin/activate
exec opentelemetry-instrument uvicorn swe_agent_server.app:app --port 9000 --host 0.0.0.0
