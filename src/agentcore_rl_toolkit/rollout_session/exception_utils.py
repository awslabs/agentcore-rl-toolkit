"""Formatting helpers for exceptions that cross a process or session boundary."""

import traceback


def exception_to_string(exc: BaseException) -> str:
    """Format ``exc`` as a full traceback string, including local variables.

    Frame locals are captured because a container-side failure reports back only this
    string. They are rendered with ``repr()``, so never use this where a frame could
    hold a credential.
    """
    tb_exc = traceback.TracebackException.from_exception(
        exc,
        capture_locals=True,
    )
    return "".join(tb_exc.format(chain=True))


def root_cause(exc: BaseException) -> BaseException:
    """The deepest exception in ``exc``'s cause/context chain -- ``exc`` itself if it has none.

    A cleanup failure raised while another exception was in flight is what propagates, so
    the outermost exception is the one a caller sees while the interesting one sits below
    it. Python builds acyclic chains, but ``__context__`` is writable, so this stops if it
    comes back round.
    """
    seen = {id(exc)}
    while True:
        below = exc.__cause__ or exc.__context__
        if below is None or id(below) in seen:
            return exc
        seen.add(id(below))
        exc = below


def describe_with_root_cause(exc: BaseException) -> str:
    """``exc`` for a one-line log, naming what it was raised over when that differs.

    Use in the message itself, not as a substitute for ``exc_info``: a teardown error is
    unreadable on its own -- see :meth:`AgentCoreHttpSession.__aexit__` -- and the summary
    line is what anyone triaging a burst of failures actually scans.
    """
    root = root_cause(exc)
    if root is exc:
        return f"{exc}"
    return f"{exc} [raised over {type(root).__name__}: {root}]"
