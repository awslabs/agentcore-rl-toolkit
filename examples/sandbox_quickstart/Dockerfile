# A shell IS required in the base image: the sandbox SDK runs commands via
# AgentCore Runtime's InvokeAgentRuntimeCommand API, which executes shell
# commands inside this container (scratch/distroless images would break exec).
FROM public.ecr.aws/docker/library/debian:bookworm-slim

# Prebuilt static binary — build it first with: ../../sandboxd/build.sh --stage .
# --chmod instead of a RUN chmod: RUN would execute arm64 /bin/sh, which needs
# qemu on x86 build hosts; with COPY-only layers no emulation is required.
COPY --chmod=0755 agentcore-sandboxd-linux-arm64 /opt/agentcore-sandbox/agentcore-sandboxd

WORKDIR /app
EXPOSE 8080
CMD ["/opt/agentcore-sandbox/agentcore-sandboxd"]
