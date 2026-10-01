FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt *.whl ./
RUN uv pip install --system --no-cache -r requirements.txt ./*.whl
COPY agent.py ./
USER 1000:1000
EXPOSE 8080
CMD ["python", "agent.py"]
