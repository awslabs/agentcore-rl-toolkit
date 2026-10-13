FROM public.ecr.aws/docker/library/python:3.12-slim
WORKDIR /app
ENV TMPDIR=/var/tmp PYTHONUNBUFFERED=1
COPY agentcore_rl_toolkit-*.whl ./
RUN pip install --no-cache-dir --no-compile \
    ./agentcore_rl_toolkit-*.whl \
    openai==3.24.0 tenacity==9.1.2 PyYAML==6.0.3
COPY app.py harness.py vanillux.py vanillux_prompts.yaml LICENSE NOTICE.md ./
EXPOSE 8080
CMD ["python", "app.py"]
