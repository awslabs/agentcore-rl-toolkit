FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app

ENV UV_SYSTEM_PYTHON=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1

COPY . .
RUN uv pip install . aws-opentelemetry-distro==0.12.2

RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 8080
CMD ["opentelemetry-instrument", "python", "-m", "rl_app"]
