"""Read-only SQL over the fixed BioGRID ORCS 2.0.18 snapshot, hit flag removed.

The scores.hit flag is not in the index, so significance has to be judged from
the scores and each screen's own criteria.
The model writes the query; nothing is matched, ranked or selected here.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time

from ..common.paths import DATA

ORCS = DATA / "screen/orcs_nohit.sqlite"
TABLES = {"screens", "scores"}
SOURCE = "BioGRID ORCS 2.0.18 local snapshot"
MAX_QUERY_CHARS = 8000
MAX_ROWS = 100
MAX_RESULT_BYTES = 48_000
MAX_QUERY_SECONDS = 15.0
# Index-backed lookups need a few thousand VM steps; a full scan of the
# 26M-row scores table cannot finish inside the time limit either way.
MAX_SQL_STEPS = 50_000_000
BACKEND_FAILURES = (
    "unable to open",
    "disk i/o",
    "database is locked",
    "malformed",
    "file is not a database",
    "out of memory",
)


def _read_only_authorizer(action: int, arg1: str | None, arg2: str | None, *_):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ and arg1 in TABLES:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION:
        # Extension loading stays disabled on the connection.
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def query_orcs(query: str, offset: int = 0) -> dict:
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= MAX_QUERY_CHARS:
        return {"status": "invalid_query", "source": SOURCE}
    statement = query.strip().removesuffix(";").strip()
    if not re.match(r"(?is)^(SELECT|WITH)\b", statement):
        return {
            "status": "invalid_query",
            "source": SOURCE,
            "error": "Supply one read-only SELECT query.",
        }
    if type(offset) is not int or not 0 <= offset <= 100_000:  # noqa: E721
        return {"status": "invalid_offset", "source": SOURCE}
    started = time.monotonic()
    try:
        con = sqlite3.connect(f"file:{ORCS}?mode=ro&immutable=1", uri=True, timeout=1.0)
    except sqlite3.Error as exc:
        return {"status": "backend_error", "source": SOURCE, "error": str(exc)}
    steps = 0
    stopped = False

    def check_budget() -> int:
        nonlocal steps, stopped
        steps += 1000
        stopped = steps > MAX_SQL_STEPS or time.monotonic() - started > MAX_QUERY_SECONDS
        return int(stopped)

    try:
        con.execute("PRAGMA query_only=ON")
        con.set_authorizer(_read_only_authorizer)
        con.set_progress_handler(check_budget, 1000)
        # Wrapping gives every query the same explicit pagination contract.
        cursor = con.execute(
            f"SELECT * FROM ({statement}) AS submitted_query LIMIT ? OFFSET ?",
            (MAX_ROWS + 1, offset),
        )
        columns = [column[0] for column in cursor.description]
        fetched = cursor.fetchall()
        rows = []
        for values in fetched[:MAX_ROWS]:
            row = dict(zip(columns, values, strict=False))
            prospective = {
                "columns": columns,
                "rows": [*rows, row],
                "source": SOURCE,
                "status": "ok",
            }
            if len(json.dumps(prospective, ensure_ascii=False).encode()) > MAX_RESULT_BYTES:
                if not rows:
                    return {
                        "status": "result_too_large",
                        "source": SOURCE,
                        "error": "One row exceeds the result budget; select fewer columns.",
                    }
                break
            rows.append(row)
        return {
            "status": "ok",
            "source": SOURCE,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "offset": offset,
            "has_more": len(fetched) > len(rows),
            "next_offset": offset + len(rows) if len(fetched) > len(rows) else None,
            "seconds": round(time.monotonic() - started, 3),
        }
    except sqlite3.Error as exc:
        message = str(exc)
        if stopped:
            status = "query_budget_exceeded"
        elif any(marker in message.lower() for marker in BACKEND_FAILURES):
            status = "backend_error"
        else:
            # Everything else (unknown names, syntax, denied tables, overflow)
            # is caused by the submitted query, not by the snapshot.
            status = "query_error"
        return {"status": status, "source": SOURCE, "error": message}
    finally:
        con.close()


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "query_orcs",
            "description": (
                "BioGRID ORCS 2.0.18 CRISPR screen database, fixed local snapshot. "
                "Table screens has one row per published screen with the native "
                "ORCS metadata columns screen_id, source_id, source_type, author, "
                "screen_name, scores_size, full_size, full_size_available, "
                "number_of_hits, analysis, significance_indicator, "
                "significance_criteria, throughput, screen_type, screen_format, "
                "experimental_setup, duration, condition_name, condition_dosage, "
                "moi, library, library_type, library_methodology, enzyme, "
                "cell_line, cell_type, phenotype, score_col_count, "
                "score_1_type ... score_5_type, organism_id, organism_official, "
                "notes, source, screen_rationale. Table scores has one row per "
                "gene record reported by a screen: screen_id, symbol, aliases, "
                "score_1 ... score_5. All values are text; "
                "scores.symbol is stored upper-case. Indexes: screens(screen_id), "
                "scores(screen_id, symbol), scores(symbol). A gene with no row in "
                "a screen was not reported by that screen. Write one SQLite "
                "SELECT (WITH is allowed). Results include source, status and "
                "pagination; use offset to continue."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQLite SELECT query."},
                    "offset": {"type": "integer", "minimum": 0},
                },
                "required": ["query"],
            },
        },
    },
]
_FUNCTIONS = {"query_orcs": query_orcs}


def invoke(name: str, arguments: dict) -> dict:
    function = _FUNCTIONS.get(name)
    if function is None:
        return {"status": "unknown_tool", "name": name}
    try:
        return function(**arguments)
    except (TypeError, ValueError) as exc:
        return {"status": "invalid_arguments", "error": str(exc)}
