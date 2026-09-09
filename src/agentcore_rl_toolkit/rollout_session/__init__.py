"""Running one agent rollout in a container, and the protocol for talking to it.

:mod:`.wire` defines the messages, :mod:`.lifecycle` the session contract, and
:mod:`.agentcore_session` / :mod:`.docker_session` implement it against Bedrock
AgentCore and a local Docker daemon.

This ``__init__`` imports nothing, so an agent harness can take :mod:`.wire` (pydantic
only) without the trainer's ``[rollout]`` dependency tree.
"""
