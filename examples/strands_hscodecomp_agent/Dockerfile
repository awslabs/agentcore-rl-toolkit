FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app

# All environment variables in one layer
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    AWS_REGION=us-west-2 \
    AWS_DEFAULT_REGION=us-west-2 \
    BYPASS_TOOL_CONSENT=true \
    STRANDS_NON_INTERACTIVE=true \
    HTS_DATA_DIR=/app/data

COPY . .

RUN uv pip install . aws-opentelemetry-distro==0.22.0

# ---------- HSCodeComp specific: the official HTS tables (about 23 MB) ----------
RUN python scripts/download_hts.py /app/data

RUN useradd -m -u 1000 bedrock_agentcore && \
    chown -R bedrock_agentcore:bedrock_agentcore /app
USER bedrock_agentcore

EXPOSE 8080

CMD ["opentelemetry-instrument", "python", "-m", "rl_app"]
