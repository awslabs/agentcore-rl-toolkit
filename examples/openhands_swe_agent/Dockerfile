FROM ubuntu:22.04

RUN apt-get -q update && apt-get install -yq \
    zsh curl rsync vim jq \
    skopeo umoci \
    git make tmux build-essential

# The scripts below all come from ./image/, which is the whole set of scripts that
# execute in this image -- during the build (install.sh, install_graders.sh,
# copy.sh) or at runtime (entrypoint.sh, swe_unpack.sh). They are flattened into
# /agent as they are copied, so the paths the image and the agent use (/agent/
# entrypoint.sh, /agent/swe_unpack.sh in swe_agent_server.rollout) do not carry the
# repo's layout. Everything else beside them in examples/openhands_swe_agent/
# runs on the developer node or in the trainer and is not part of this image.
WORKDIR /agent
COPY ./image/install.sh ./install.sh
RUN ./install.sh
ENV PATH=/agent/uv:$PATH

COPY ./image/install_graders.sh ./install_graders.sh
RUN ./install_graders.sh

# The harness implements a wire protocol it does not own: the request/response
# types live in agentcore_rl_toolkit.rollout_session.wire, on the
# trainer's side of the boundary, so the caller and every harness agree on one
# definition. Supplied as a named build context (--build-context rlpkg=<repo
# root>) rather than by widening this image's context to the repo, which would
# invalidate the layers above on any unrelated change in the repo. Only the
# three paths below are copied -- the minimum hatchling builds a wheel from:
# pyproject.toml, the README its `readme =` metadata points at, and the src/
# layout package itself. copy.sh installs it with its declared dependencies:
# the toolkit's top-level __init__ imports .app eagerly, so importing
# agentcore_rl_toolkit.rollout_session.wire runs `import bedrock_agentcore` too,
# and a --no-deps install is a package that cannot be imported at all.
COPY --from=rlpkg pyproject.toml README.md /rl_toolkit/
COPY --from=rlpkg src/agentcore_rl_toolkit /rl_toolkit/src/agentcore_rl_toolkit/

COPY pyproject.toml ./
COPY ./image/copy.sh ./image/entrypoint.sh ./image/swe_unpack.sh ./
COPY src ./src/

RUN ./copy.sh
CMD ["/agent/entrypoint.sh"]
EXPOSE 8080
