"""Formatting helpers for exceptions that cross a process or session boundary.

It sits in this package because that boundary is a container rollout's: the formatted
string is stored on the failed rollout's record. Move it up to the package root once
something outside ``rollout_session`` needs it.
"""

import traceback


def exception_to_string(exc: BaseException) -> str:
    """Format ``exc`` as a full traceback string, including local variables.

    ``capture_locals=True`` is the point of this helper over a plain
    :func:`traceback.format_exception`: a rollout that fails inside a container reports
    back only what we serialize here, so the frame locals are usually the only record
    of which task, session or endpoint the failure was about. Chained causes come along
    too, so an error re-raised from a wrapper still carries the original.

    The cost is that locals are rendered with ``repr()``, so anything held in a failing
    frame ends up in the stored string -- do not use this where a frame could hold a
    credential.
    """
    tb_exc = traceback.TracebackException.from_exception(
        exc,
        capture_locals=True,
    )
    return "".join(tb_exc.format(chain=True))
