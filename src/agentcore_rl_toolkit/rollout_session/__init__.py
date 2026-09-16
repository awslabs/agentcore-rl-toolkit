"""Running one agent rollout in a container, and the protocol for talking to it.

:mod:`.wire` defines the messages, :mod:`.lifecycle` the session contract, and
:mod:`.agentcore_http_session` / :mod:`.docker_session` implement it against Bedrock
AgentCore and a local Docker daemon. :mod:`.agentcore_s3_session` implements the same
contract over a different agent shape: an ``AgentCoreRLApp`` invoked once with its result
polled from S3, so an agent deployed for batch evaluation trains unchanged.

This ``__init__`` imports nothing, so an agent harness can take :mod:`.wire` (pydantic
only) without the trainer's ``[rollout]`` dependency tree.
"""
