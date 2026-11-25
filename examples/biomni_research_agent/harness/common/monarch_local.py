"""Monarch entity lookups served from the local SQLite file prepare_data.py builds from a dated Monarch KG release.

Served: GET /v3/api/entity/{curie}, shaped like the live response: the dump's node record plus has_phenotype,
causal_gene, causes_disease and inheritance from the KG edges. A missing entity answers the live 404 body. Every other
route (search, associations, semsim, histopheno) returns None, and online.py sends it to api.monarchinitiative.org:
those are ranked or computed by Monarch's own services, which the KG dump does not reproduce.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from urllib.parse import unquote, urlsplit

HOST = "api.monarchinitiative.org"
_ENTITY = re.compile(r"^/v3/api/entity/(?P<prefix>[A-Za-z][A-Za-z0-9._-]*):(?P<local>[A-Za-z0-9._-]+)/?$")
# The model writes ncbigene:1100 and NCBIGene:1100 interchangeably; the dump spells prefixes one way.
_PREFIX = {
    p.casefold(): p
    for p in ("HGNC", "MONDO", "HP", "OMIM", "NCBIGene", "ENSEMBL", "MGI", "ZFIN", "Orphanet", "MP", "GO", "UBERON")
}
_PREFIX.update({"mim": "OMIM", "orpha": "Orphanet"})
# Fields the live endpoint computes from its own index (closures, counts, UI links); not in the dump, left null.
_ABSENT = (
    "has_phenotype_closure",
    "has_phenotype_closure_label",
    "has_descendant",
    "has_descendant_label",
    "has_descendant_count",
    "node_relationships",
    "node_hierarchy",
    "association_counts",
    "mappings",
    "external_links",
    "provided_by_link",
    "cross_species_term_clique",
    "uri",
)


class MonarchLocal:
    def __init__(self, path: Path):
        self.path = path
        self._state = threading.local()

    def _db(self) -> sqlite3.Connection:
        db = getattr(self._state, "db", None)
        if db is None:
            if not self.path.is_file():
                raise FileNotFoundError(f"{self.path} is missing; run prepare_data.py --only monarch")
            db = sqlite3.connect(f"file:{self.path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
            self._state.db = db
        return db

    def get(self, url: str) -> tuple[int, object] | None:
        """(HTTP status, JSON body) for an entity lookup, else None."""
        parsed = urlsplit(url)
        match = _ENTITY.match(unquote(parsed.path))
        if (parsed.hostname or "").lower() != HOST or match is None:
            return None
        prefix = _PREFIX.get(match["prefix"].casefold(), match["prefix"])
        db = self._db()
        row = db.execute("SELECT payload FROM nodes WHERE id = ?", (f"{prefix}:{match['local']}",)).fetchone()
        if row is None:
            return 404, {"detail": "Entity not found"}
        entity = json.loads(row[0])
        node = entity["id"]
        phenotypes = self._objects(db, "biolink:has_phenotype", node)
        entity["has_phenotype"] = phenotypes or None
        entity["has_phenotype_label"] = self._labels(db, phenotypes) or None
        entity["has_phenotype_count"] = len(phenotypes) or None
        # causal_gene: genes that cause this disease (subjects of its incoming causes edges); causes_disease: the
        # diseases this gene causes (objects of its outgoing causes edges). Both embed whole node records, as live.
        entity["causal_gene"] = self._nodes(db, self._subjects(db, "biolink:causes", node), "biolink:Gene")
        entity["causes_disease"] = self._nodes(db, self._objects(db, "biolink:causes", node), "biolink:Disease")
        inheritance = self._nodes(db, self._objects(db, "biolink:has_mode_of_inheritance", node))
        entity["inheritance"] = inheritance[0] if inheritance else None
        for field in _ABSENT:
            entity.setdefault(field, None)
        return 200, entity

    @staticmethod
    def _objects(db: sqlite3.Connection, predicate: str, subject: str) -> list[str]:
        sql = "SELECT object FROM edges WHERE predicate = ? AND subject = ?"
        return [row[0] for row in db.execute(sql, (predicate, subject))]

    @staticmethod
    def _subjects(db: sqlite3.Connection, predicate: str, obj: str) -> list[str]:
        sql = "SELECT DISTINCT subject FROM edges WHERE predicate = ? AND object = ?"
        return [row[0] for row in db.execute(sql, (predicate, obj))]

    @staticmethod
    def _labels(db: sqlite3.Connection, ids: list[str]) -> list[str]:
        names = dict(db.execute(f"SELECT id, name FROM nodes WHERE id IN ({','.join('?' * len(ids))})", ids))
        return [names.get(i) or i for i in ids]

    @staticmethod
    def _nodes(db: sqlite3.Connection, ids: list[str], category: str | None = None) -> list[dict] | None:
        if not ids:
            return None
        rows = db.execute(f"SELECT payload FROM nodes WHERE id IN ({','.join('?' * len(ids))})", ids)
        items = [json.loads(row[0]) for row in rows]
        return [item for item in items if category is None or item.get("category") == category] or None
