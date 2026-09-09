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
