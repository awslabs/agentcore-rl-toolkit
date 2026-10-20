FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    GWAS_CATALOG_DUCKDB_EXTENSION_DIR=/opt/duckdb-extensions
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
RUN uv sync --locked --no-dev --no-editable --extra mcp --extra runtime --no-cache
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
RUN uv sync --locked --no-dev --no-editable --extra mcp --extra runtime --extra codex --no-cache
# The proxy uses MCP 1.x; keep it separate from this example's MCP 2.x environment.
RUN UV_TOOL_DIR=/opt/mcp-proxy UV_TOOL_BIN_DIR=/usr/local/bin \
    uv tool install --no-cache mcp-proxy-for-aws-cli==1.7.0
COPY integrations/codex/ integrations/codex/
ENV MAIN_AGENT_EXECUTION_MODE=container CODEX_HOME=/var/tmp/biomedical/codex
WORKDIR /workspace
EXPOSE 8080
CMD ["python", "/app/integrations/codex/app.py"]

FROM base AS mcp
EXPOSE 8000
CMD ["biomedical-mcp", "--transport", "streamable-http"]
