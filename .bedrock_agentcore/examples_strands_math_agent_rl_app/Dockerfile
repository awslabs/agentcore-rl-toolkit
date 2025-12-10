FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app

# All environment variables in one layer
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1 \
    AWS_REGION=us-west-2 \
    AWS_DEFAULT_REGION=us-west-2



COPY examples/strands_math_agent examples/strands_math_agent
# Install from pyproject.toml directory
RUN cd examples/strands_math_agent && uv pip install .




RUN uv pip install aws-opentelemetry-distro==0.12.2

# [TODO]: remove the following lines after being able to add this agentcore-rl-toolkit as
# a dependency in examples/strands_math_agent/pyproject.toml
COPY . .
RUN uv pip install --force-reinstall --no-deps .


# Signal that this is running in Docker for host binding logic
ENV DOCKER_CONTAINER=1

# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 9000
EXPOSE 8000
EXPOSE 8080

# Copy entire project (respecting .dockerignore)
COPY . .

# Use the full module path

CMD ["opentelemetry-instrument", "python", "-m", "rl_app"]
