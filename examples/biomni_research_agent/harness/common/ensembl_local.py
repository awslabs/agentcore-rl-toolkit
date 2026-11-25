"""Ensembl release 116 (GRCh38) REST routes served from the local SQLite file prepare_data.py builds.

Served for human: /lookup/id/{gene id}, /lookup/symbol/{species}/{symbol} (with expand=1, the same gene body without a
Transcript array), /xrefs/symbol/{species}/{symbol}, /xrefs/id/{gene id} and /sequence/id/{id} for type cds, cdna and
protein. /variation/{species}/{id} and /overlap/region/{species}/{region} return None, and online.py sends them to
rest.ensembl.org. Every other request (VEP, LD, other species, genomic sequence, an id the release does not carry)
gets a fixed 404, so the tool never reaches the live service for it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

HOST = "rest.ensembl.org"
SPECIES = {"human", "homo_sapiens"}
SEQUENCE_TYPES = {"cds", "cdna", "protein"}
MISS = (
    "Request is outside the Ensembl strict-local projection. Served routes: "
    "/lookup/id/{id}, /lookup/symbol/{species}/{symbol}, /variation/{species}/{id}, "
    "/overlap/region/{species}/{region}, /sequence/id/{id}."
)


def _parameters(query: str) -> dict[str, str]:
    # Ensembl accepts both ';' and '&' between parameters.
    return dict(parse_qsl(query.replace(";", "&"), keep_blank_values=True))


class EnsemblLocal:
    def __init__(self, path: Path):
        self.path = path
        self._state = threading.local()

    def _db(self) -> sqlite3.Connection:
        db = getattr(self._state, "db", None)
        if db is None:
            if not self.path.is_file():
                raise FileNotFoundError(f"{self.path} is missing; run prepare_data.py --only ensembl")
            db = sqlite3.connect(f"file:{self.path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
            self._state.db = db
        return db

    def get(self, url: str) -> tuple[int, object] | None:
        """(HTTP status, JSON body), or None for a variation or overlap request, which goes online."""
        parts = [part for part in urlsplit(url).path.split("/") if part]
        if (parts[:1] == ["variation"] and len(parts) == 3) or (parts[:2] == ["overlap", "region"] and len(parts) == 4):
            return None
        return self._served(url) or (404, {"error": MISS})

    def _served(self, url: str) -> tuple[int, object] | None:
        parsed = urlsplit(url)
        if (parsed.hostname or "").lower() != HOST:
            return None
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        params = _parameters(parsed.query)
        params.pop("content-type", None)
        route, args = parts[:2], parts[2:]
        if route == ["lookup", "id"] and len(args) == 1 and set(params) <= {"expand"}:
            return self._one("SELECT body FROM genes WHERE stable_id = ?", args[0])
        if route == ["lookup", "symbol"] and len(args) == 2 and set(params) <= {"expand"}:
            if args[0].lower() not in SPECIES:
                return None
            return self._one(
                "SELECT body FROM genes JOIN gene_symbols USING (stable_id) WHERE symbol_lower = ?", args[1].lower()
            )
        if route == ["xrefs", "symbol"] and len(args) == 2 and set(params) <= {"object_type", "db_type", "external_db"}:
            if args[0].lower() not in SPECIES:
                return None
            rows = (
                self._db()
                .execute(
                    "SELECT stable_id FROM xref_symbols WHERE symbol_lower = ? ORDER BY stable_id",
                    (args[1].casefold(),),
                )
                .fetchall()
            )
            return (200, [{"type": "gene", "id": row[0]} for row in rows]) if rows else None
        if route == ["xrefs", "id"] and len(args) == 1 and set(params) <= {"all_levels", "external_db", "db_type"}:
            return self._one("SELECT body FROM gene_xrefs WHERE stable_id = ?", args[0])
        if route == ["sequence", "id"] and len(args) == 1 and set(params) <= {"type", "multiple_sequences"}:
            # The model sometimes writes type=cds;content-type=application/json as one value.
            kind = params.get("type", "genomic").split(";")[0].strip()
            if kind not in SEQUENCE_TYPES:
                return None
            row = (
                self._db()
                .execute(
                    "SELECT molecule, version, seq FROM sequences WHERE stable_id = ? AND type = ?", (args[0], kind)
                )
                .fetchone()
            )
            if row is None:
                return None
            molecule, version, seq = row
            return 200, {
                "molecule": molecule,
                "id": args[0],
                "query": args[0],
                "version": version,
                "desc": None,
                "seq": seq,
            }
        return None

    def _one(self, sql: str, key: str) -> tuple[int, object] | None:
        row = self._db().execute(sql, (key,)).fetchone()
        return (200, json.loads(row[0])) if row else None
