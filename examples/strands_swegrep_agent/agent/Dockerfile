# Python 3.12.12, as in the image that trained the published model; uv pinned (the toolkit's
# pyproject needs a recent uv to parse).
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
COPY --from=ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 /uv /usr/local/bin/uv

WORKDIR /app

# Source + packaging (small, cache-friendly).
COPY pyproject.toml uv.lock README.md rl_app.py ./
COPY rollout ./rollout
COPY scoring ./scoring
# Every dependency from uv.lock, hash-checked; then this project alone.
RUN uv export --frozen --no-emit-project --no-dev --output-file /tmp/requirements.lock >/dev/null \
    && uv pip install --system --no-cache --require-hashes -r /tmp/requirements.lock \
    && uv pip install --system --no-cache --no-deps . \
    && rm /tmp/requirements.lock

ENV OTEL_METRICS_EXPORTER=none

EXPOSE 8080
# opentelemetry-instrument emits traces to CloudWatch/X-Ray (aws-opentelemetry-distro).
CMD ["opentelemetry-instrument", "python", "rl_app.py"]
