"""Concurrency limits for the rollout path: how many at once, in what order, how fast.

Each limit is a ``Protocol`` plus a process-local implementation; Ray is confined to
:mod:`.ray_adapters`. This ``__init__`` imports nothing, so Ray stays optional.
"""
