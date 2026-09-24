"""Running one agent rollout in a container, and the protocol for talking to it.

This ``__init__`` imports nothing, so an agent harness can take :mod:`.wire` (pydantic
only) without the trainer's ``[rollout]`` dependency tree.
"""
