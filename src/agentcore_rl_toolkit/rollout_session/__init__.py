"""Running one agent rollout in a container, and the protocol for talking to it.

This package is the boundary between the trainer and a task container. It owns
both halves of that boundary: :mod:`.wire` defines what is said over it, while
:mod:`.lifecycle` defines what a caller may ask of a rollout session (start
it, run a rollout in it, tear it down) and :mod:`.agentcore_session` /
:mod:`.docker_session` implement that against Bedrock AgentCore and a local
Docker daemon respectively. How the messages are carried is each session's own
business -- the AgentCore data plane in one case, HTTP in the other.

The protocol lives here rather than in an agent harness on purpose: it is what the
trainer requires of *any* agent it can drive, so a second harness implements it
instead of inheriting it from the first. See :mod:`.wire`.

Import layering, which is load-bearing and worth keeping:

- This ``__init__`` performs no imports at all, so importing one module here costs
  that module's own dependencies and nothing else. It is the same contract
  :mod:`..rollout_gateway` states, arrived at more cheaply: that package has
  re-exports and so needs a lazy ``__getattr__`` to keep the aiohttp-dependent ones
  off the import path, while there is nothing to re-export here.
- :mod:`.wire` needs only pydantic, and :mod:`.exception_utils` only the stdlib, so
  an agent harness can take the protocol it has to speak without the trainer's
  dependency tree -- which is the point of defining that protocol on this side of
  the boundary.
- Everything else -- :mod:`.lifecycle`, the two session implementations and
  :mod:`.factory` -- needs the ``[rollout]`` extra (aioboto3, backoff, aiohttp),
  mostly by way of :mod:`..aws_tools`. All of it is trainer- or host-side, so that is
  not a cost the agent container pays.
- Nothing in here imports Ray or reads config of its own, so the same modules run in
  a verl worker, in a task container and in a standalone script.
"""
