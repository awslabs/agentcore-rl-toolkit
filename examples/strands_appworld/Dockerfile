FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    UV_GIT_LFS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    DOCKER_CONTAINER=1 \
    AWS_REGION=us-west-2 \
    AWS_DEFAULT_REGION=us-west-2 \
    APPWORLD_CACHE=/app/.appworld_cache

# Install git and git-lfs (needed for appworld git clone with LFS bundles)
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install toolkit from build context, then install example deps
# Note: uv pip install does NOT respect [tool.uv.sources], so appworld must be
# installed explicitly from git with UV_GIT_LFS=1 to fetch LFS bundle files.
# Once appworld is installed, uv pip install . will see it as satisfied and skip it.
RUN --mount=type=bind,from=toolkit,source=.,target=/toolkit \
    uv pip install /toolkit && \
    uv pip install "appworld @ git+https://github.com/stonybrooknlp/appworld.git@11e8d1832ffd564d3a87d30d6ad9a1e5997afda1" && \
    uv pip install .

# Install OpenTelemetry for ACR observability
RUN uv pip install aws-opentelemetry-distro==0.12.2

# Install AppWorld data at build time
RUN appworld install && appworld download data

# Make AppWorld data dirs writable for non-root user
RUN chmod -R a+rw /app || true
RUN APPWORLD_DATA=$(python -c "import appworld; import os; print(os.path.dirname(appworld.__file__))") && \
    chmod -R a+rw "$APPWORLD_DATA" || true

# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 9000
EXPOSE 8000
EXPOSE 8080

CMD ["opentelemetry-instrument", "python", "-m", "rl_app"]
