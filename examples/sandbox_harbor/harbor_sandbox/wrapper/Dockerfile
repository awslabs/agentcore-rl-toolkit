# Sandbox wrapper: layer the AgentCore sandboxd health shim onto a task env image.
# COPY-only (no RUN) -> no shell execution -> no qemu even when cross-building.
# SANDBOXD selects the shim arch (arm64 for the serverless microVM runtime);
# pass the matching --platform too. build.py always sets this explicitly.
ARG BASE
FROM ${BASE}

ARG SANDBOXD=agentcore-sandboxd-linux-arm64
COPY --chmod=0755 ${SANDBOXD} /opt/agentcore-sandbox/agentcore-sandboxd

EXPOSE 8080
CMD ["/opt/agentcore-sandbox/agentcore-sandboxd"]
