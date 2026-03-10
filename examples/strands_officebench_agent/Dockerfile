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

# ---------- OfficeBench specific requirements ----------
# System dependencies: LibreOffice (doc conversion), Tesseract (OCR), ImageMagick
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libreoffice \
    tesseract-ocr \
    libtesseract-dev \
    imagemagick \
    git \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create testbed directories
RUN mkdir -p /testbed/data /testbed/calendar /testbed/emails

# ----------

COPY . .
# OfficeBench app scripts are expected at /apps/ (copied into context by build.sh)
RUN if [ -d "apps" ]; then cp -r apps /apps; fi

# Install local toolkit from build context, then install example deps
RUN --mount=type=bind,from=toolkit,source=.,target=/toolkit \
    uv pip install /toolkit && uv pip install .

RUN uv pip install aws-opentelemetry-distro==0.12.2

# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore && \
    chown -R bedrock_agentcore:bedrock_agentcore /testbed /app
USER bedrock_agentcore

EXPOSE 9000
EXPOSE 8000
EXPOSE 8080

CMD ["opentelemetry-instrument", "python", "-m", "dev_app"]
