"""Exceptions raised by harbor_sandbox — gathered here so every module's failure
modes are visible in one place."""
from __future__ import annotations


class ValidationError(ValueError):
    """Malformed benchmark id ('org/name' expected)."""


class ImageNotFoundError(RuntimeError):
    """No conventional ECR tag for (task, arch) — task unknown or not built yet."""
