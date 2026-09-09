"""Small shared helpers for the agent server."""

import traceback


def exc_to_full_string(exc: BaseException) -> str:
    return "".join(traceback.TracebackException.from_exception(exc).format())


def clean_metrics(metrics: dict[str, float | int | None]) -> dict[str, float]:
    """Drop unmeasured (None) metrics and cast the rest to float.

    Keeps the trainer from reducing placeholder values.
    """
    return {k: float(v) for k, v in metrics.items() if v is not None}
