"""screen's ten tools: ORCS SQL without the hit flag, four direct-query tools of common/online.py and the
five of online_extra.py.
"""

from __future__ import annotations

from ..common import online
from . import online_extra, orcs

OFFICIAL_FROM_MIXED = ("query_ensembl", "query_uniprot", "query_stringdb", "query_quickgo")
TOOL_SPECS = [
    *orcs.TOOL_SPECS,
    *[s for s in online.OFFICIAL_SPECS if s["function"]["name"] in OFFICIAL_FROM_MIXED],
    *online_extra.EXTRA_SPECS,
]
NAMES = [spec["function"]["name"] for spec in TOOL_SPECS]
if len(NAMES) != len(set(NAMES)) or len(NAMES) != 10:
    raise RuntimeError(f"Unexpected screen mixed tool pool: {NAMES}")


def invoke(name: str, arguments: dict) -> dict:
    if name == "query_orcs":
        return orcs.invoke(name, arguments)
    if name in OFFICIAL_FROM_MIXED:
        return online.invoke(name, arguments)
    if name in online_extra.EXTRA_NAMES:
        return online_extra.invoke(name, arguments)
    return {"status": "unknown_tool", "name": name}
