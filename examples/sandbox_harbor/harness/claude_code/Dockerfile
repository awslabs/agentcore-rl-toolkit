# tmax harness image — agent_type "claude_code" (Claude Code co-located in the
# task sandbox; this image only orchestrates — node + claude are bootstrapped
# INTO the task box at session start).
# Built for linux/arm64: runs on the default serverless MicroVM runtime.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AWS_REGION=us-west-2 \
    AWS_DEFAULT_REGION=us-west-2

WORKDIR /app

# Core server + AWS SDK first (cached layer). bedrock-agentcore provides the
# /ping + /invocations HTTP contract on :8080.
RUN pip install "bedrock-agentcore>=1.0.3" "boto3>=1.43.68"

# The toolkit source (staged by build_push.sh) provides SandboxClient + the
# @rollout_entrypoint SDK (AgentCoreRLApp).
COPY _toolkit /toolkit
RUN pip install /toolkit

# The local harbor_sandbox package (staged by build_push.sh) provides
# HarborSandboxClient; /app is on sys.path when `python harness.py` runs.
COPY harbor_sandbox /app/harbor_sandbox
COPY harness.py /app/harness.py

EXPOSE 8080
CMD ["python", "harness.py"]
