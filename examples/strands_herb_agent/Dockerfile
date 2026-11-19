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
    HERB_DIR=/app/herb \
    HERB_INDEX_DIR=/app/index

COPY . .

# PyStemmer < 3 (llama-index's BM25 retriever) has no arm64 wheels: it is compiled
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*

RUN uv pip install . aws-opentelemetry-distro==0.22.0

# ---------- HERB specific: the pinned HERB code and data (about 7 MB), from its repository ----------
RUN python scripts/fetch_herb.py /app/herb && test -f /app/index/embeddings.npy

RUN useradd -m -u 1000 bedrock_agentcore && \
    chown -R bedrock_agentcore:bedrock_agentcore /app
USER bedrock_agentcore

EXPOSE 8080

CMD ["opentelemetry-instrument", "python", "-m", "rl_app"]
