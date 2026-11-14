# Debian provides the shell and tools the demo commands use.
FROM public.ecr.aws/docker/library/debian:bookworm-slim

# The E2B SDK runs commands as `user` in /home/user unless told otherwise.
RUN useradd --create-home --shell /bin/bash user

# envd serves E2B's process and filesystem APIs.
ADD --chmod=0755 https://storage.googleapis.com/e2b-artifact-binaries/envd/v0.10.2/arm64/envd /usr/local/bin/envd

# Build it first with: ../../sandboxd/build.sh --stage .
COPY --chmod=0755 agentcore-sandboxd-linux-arm64 /usr/local/bin/agentcore-sandboxd

# sandboxd answers AgentCore's /ping and /invocations on the serving port;
# envd listens beside it on a port in the runtime's allowedPorts.
EXPOSE 11000 10983
CMD ["/bin/sh", "-c", "envd -isnotfc -no-cgroups -port 10983 >/var/log/envd.log 2>&1 & exec agentcore-sandboxd --listen 0.0.0.0:11000"]
