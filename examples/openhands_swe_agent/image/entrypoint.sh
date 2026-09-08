#!/usr/bin/env bash

set -eux

source /agent/.venv/bin/activate
# uvicorn directly, not `fastapi run`: swe_agent_server.app:app is a
# BedrockAgentCoreApp (a Starlette app), which fastapi-cli would announce as a
# FastAPI app with docs it does not serve. One process, so the app's single async
# task registry is the one /ping answers from.
exec opentelemetry-instrument uvicorn swe_agent_server.app:app --port 8080 --host 0.0.0.0
