"""Botocore freezegun compatibility patch.

AppWorld uses freezegun to freeze time for task simulation. This conflicts with
botocore's datetime usage for AWS request signing. This module provides:

1. patch_botocore() - runtime monkey-patch for local dev
2. The file itself can be copied over botocore/compat.py at Docker build time
"""

import ctypes
import ctypes.util
import logging

logger = logging.getLogger(__name__)

# Load the C standard library for clock_gettime
_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
CLOCK_REALTIME = 0


class _Timespec(ctypes.Structure):
    _fields_ = [
        ("tv_sec", ctypes.c_long),
        ("tv_nsec", ctypes.c_long),
    ]


def _system_time_real() -> float:
    """Return the real Unix timestamp using clock_gettime, bypassing freezegun."""
    t = _Timespec()
    if _libc.clock_gettime(CLOCK_REALTIME, ctypes.byref(t)) != 0:
        raise OSError("clock_gettime failed")
    return t.tv_sec + t.tv_nsec / 1e9


def patch_botocore():
    """Monkey-patch botocore.compat.get_current_datetime to use real wall-clock time.

    This is a fallback for local development. In Docker, the full compat.py is
    copied over botocore's compat.py at build time.
    """
    try:
        import datetime

        import botocore.compat

        def get_current_datetime(remove_tzinfo=True):
            ts = _system_time_real()
            dt = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
            if remove_tzinfo:
                dt = dt.replace(tzinfo=None)
            return dt

        botocore.compat.get_current_datetime = get_current_datetime
        logger.info("Patched botocore.compat.get_current_datetime for freezegun compatibility")
    except (ImportError, AttributeError) as e:
        logger.warning("Could not patch botocore: %s", e)
