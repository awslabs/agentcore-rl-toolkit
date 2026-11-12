FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY pyproject.toml .
RUN uv pip install --system --no-cache -r pyproject.toml
COPY server.py .
RUN useradd --create-home --uid 10001 appuser
USER appuser
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "server.py"]
