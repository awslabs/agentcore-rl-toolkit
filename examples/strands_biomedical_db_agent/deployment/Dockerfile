FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    TMPDIR=/var/tmp/biomedical TMP=/var/tmp/biomedical TEMP=/var/tmp/biomedical \
    GWAS_CATALOG_DUCKDB_EXTENSION_DIR=/opt/duckdb-extensions
RUN mkdir -p "$TMPDIR"
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
RUN uv sync --locked --no-dev --no-editable --extra aws --extra mcp --extra runtime --no-cache
ENV PATH="/opt/venv/bin:$PATH"
RUN python -c "import duckdb,os; c=duckdb.connect(); c.execute('SET extension_directory = ?', [os.environ['GWAS_CATALOG_DUCKDB_EXTENSION_DIR']]); c.execute('INSTALL httpfs; INSTALL aws;')"
COPY apps/ apps/

FROM base AS opentargets
EXPOSE 8080
CMD ["python", "-m", "apps.opentargets"]

FROM base AS gwas-catalog
EXPOSE 8080
CMD ["python", "-m", "apps.gwas_catalog"]

FROM base AS codex
RUN apt-get update && apt-get install -y --no-install-recommends bash ripgrep ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN uv sync --locked --no-dev --no-editable --extra aws --extra mcp --extra runtime --extra codex --no-cache
COPY integrations/codex/ integrations/codex/
ENV MAIN_AGENT_EXECUTION_MODE=container CODEX_HOME=/var/tmp/biomedical/codex
WORKDIR /workspace
EXPOSE 8080
CMD ["python", "/app/integrations/codex/app.py"]

FROM base AS mcp
EXPOSE 8000
CMD ["biomedical-mcp", "--transport", "streamable-http"]
