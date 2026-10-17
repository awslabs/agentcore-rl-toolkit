"""Shared AWS access: boto3 session plumbing plus thin per-service wrappers on it.

Nothing here reads config of its own -- callers pass in ARNs, table names and regions --
so the same code serves deploy scripts and the trainer. Dependencies are included
in the base SDK installation.
"""
