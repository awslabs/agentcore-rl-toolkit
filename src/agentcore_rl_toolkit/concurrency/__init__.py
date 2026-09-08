"""Shaping concurrency on the rollout path: how many at once, in what order, how fast.

Three separate limits, one module each: :mod:`.priority_semaphore` bounds how many
rollouts run at once, :mod:`.priority_assigner` decides which of the waiting ones
goes first, and :mod:`.rate_limiter` bounds the rate of new AgentCore sessions.
They are grouped here because a caller configures them together, and because they
share a shape: a ``Protocol`` for the interface plus a process-local implementation.

Each limit exists cluster-wide in a real run, as the local class run as a Ray actor.
That is the *only* reason Ray appears anywhere in here, and it is confined to
:mod:`.ray_adapters`; every module above is importable with Ray absent, and this
``__init__`` deliberately imports nothing so that importing the package does not
change that. Callers therefore split cleanly: whoever owns the cluster creates the
actors and wraps the handles via :mod:`.ray_adapters`, and everything downstream of
that sees only the protocols.
"""
