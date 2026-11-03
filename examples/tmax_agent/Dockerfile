FROM public.ecr.aws/docker/library/python:3.12-slim
WORKDIR /app
ENV TMPDIR=/var/tmp PYTHONUNBUFFERED=1 \
    MSWEA_GLOBAL_CONFIG_DIR=/var/tmp/mini-swe-agent MSWEA_SILENT_STARTUP=1
RUN pip install --no-cache-dir --no-compile \
    'agentcore-rl-toolkit[core]==0.2.0' \
    mini-swe-agent==2.4.6 \
    # litellm>=1.106 requires mcp for tool calls (BerriAI/litellm#46045).
    litellm==1.105.0 \
    aws-opentelemetry-distro==0.12.2
COPY app.py harness.py ./
EXPOSE 8080
CMD ["opentelemetry-instrument", "python", "app.py"]
