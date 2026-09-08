"""Shared AWS access: the boto3 plumbing and the thin per-service wrappers on it.

This is a *shared* package -- the rollout path and the deploy-time scripts both
import it -- so everything in here reads no config of its own by construction: no
config file, no environment, no ssh, no subprocesses. Callers pass in the ARNs,
table names and regions they want, which is what lets the same code serve a deploy
script (whose values came from a config file) and the trainer (where they arrived as
hydra config).

:mod:`.boto3_tools` is the floor: cached sync and async sessions, self-refreshing
self-assumed role credentials, ``tags_to_map``. On top of it sits one module per
service -- :mod:`.dynamodb_tools`, :mod:`.s3_tools`, :mod:`.ec2_tools` (plus IMDS,
for the identity of the host we are on) -- and, for bedrock-agentcore, two split by
what they act on and how often: :mod:`.agentcore_tools` is one session's lifecycle,
once per rollout, :mod:`.agentcore_control` the pool and runtime those sessions land
on, once per image roll.

Two modules here are policy rather than wrappers: :mod:`.ec2_monitor`, a long-lived
poller that maps running instances back to their sessions, and
:mod:`.persistent_dict`, a write-through dict whose ``Persister`` seam is in
principle store-agnostic. Both live here because their subject is entirely AWS in
practice, so leaving them outside would only have kept AWS-coupled modules at the
top level.

This ``__init__`` imports nothing, but unlike :mod:`..rollout_session` the package has
no base-install-only subset to enter through: :mod:`.boto3_tools` imports aioboto3 and
everything above it imports :mod:`.boto3_tools`, so any module here needs the
``[rollout]`` extra.
"""
