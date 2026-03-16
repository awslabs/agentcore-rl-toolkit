FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app

# All environment variables in one layer
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    AWS_REGION=us-west-2 \
    AWS_DEFAULT_REGION=us-west-2 \
    APPWORLD_ROOT=/appworld \
    APPWORLD_CACHE=/appworld/cache

# ---------- AppWorld specific requirements ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs curl && \
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . .

# Install local toolkit from build context, then install example deps
RUN --mount=type=bind,from=toolkit,source=.,target=/toolkit \
    uv pip install /toolkit && uv pip install .

RUN uv pip install aws-opentelemetry-distro==0.12.2

# Bake AppWorld data into image (as root, since appworld install writes to site-packages)
RUN appworld install && appworld download data --root /appworld

RUN useradd -m -u 1000 bedrock_agentcore && \
    chown -R bedrock_agentcore:bedrock_agentcore /appworld /app
USER bedrock_agentcore

EXPOSE 8080

CMD ["opentelemetry-instrument", "python", "-m", "rl_app"]
