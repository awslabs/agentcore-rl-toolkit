"""seqqa's tools: generic sequence primitives, related official tools and reference routing.

No tool accepts a question, answer, file path, or executable code. The official
module is vendored without edits; documented corrections live in this adapter.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
from functools import lru_cache

import jsonschema
import requests
from Bio import Entrez, SeqIO

from ..common.online import NCBI_HOST, NCBI_KEY_INTERVAL, _limit
from ..common.online import ncbi_keys as _keys
from ..common.paths import DATA
from . import extra_tools as extra
from . import official, primitives

REFS = DATA / "seqqa/refs"
MAX_REPLY_BYTES = 48_000
MAX_SEQUENCE = 100_000
_KEY_LOCK = threading.Lock()
_KEY_INDEX = os.getpid()


def _redact(text):
    for key in _keys():
        text = text.replace(key, "[REDACTED]")
    return re.sub(r"(?i)(api_key[=:])[^&\s]+", r"\1[REDACTED]", text)


def _rate_limit(key):
    # The same stamp files as common/online.py, so dbqa, screen and seqqa share each key's rate budget.
    _limit(f"{NCBI_HOST}_sha256_{hashlib.sha256(key.encode()).hexdigest()[:12]}", NCBI_KEY_INTERVAL)


class _EntrezProxy:
    """Use bounded requests and a per-request key; never mutate Bio.Entrez."""

    email = "biomni-research-agent@example.org"
    read = staticmethod(Entrez.read)

    def _request(self, method, params):
        global _KEY_INDEX
        keys = _keys()
        with _KEY_LOCK:
            key = keys[_KEY_INDEX % len(keys)]
            _KEY_INDEX += 1
        _rate_limit(key)
        params = dict(params, email=self.email, tool="biomni-research-agent", api_key=key)
        with requests.get(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{method}.fcgi",
            params=params,
            timeout=(5, 30),
            stream=True,
        ) as response:
            response.raise_for_status()
            chunks = []
            size = 0
            for chunk in response.iter_content(8192):
                size += len(chunk)
                if size > 16_000_000:
                    raise ValueError("Reference response exceeded 16 MB")
                chunks.append(chunk)
            body = b"".join(chunks)
        if params.get("retmode") == "text":
            return io.StringIO(body.decode("utf-8"))
        return io.BytesIO(body)

    def esearch(self, **params):
        return self._request("esearch", params)

    def efetch(self, **params):
        return self._request("efetch", params)


official.Entrez = _EntrezProxy()


@lru_cache(maxsize=2)
def _reference(filename):
    return SeqIO.read(REFS / filename, "genbank")


def _local_organism(organism):
    normalized = re.sub(r"[\s._-]+", "", organism).lower()
    return normalized in {
        "ecoli",
        "escherichiacoli",
        "ecolik12",
        "escherichiacolik12",
        "ecolik12mg1655",
        "escherichiacolik12mg1655",
        "escherichiacolistrk12substrmg1655",
        "nc0009133",
    }


def get_gene_coding_sequence(gene_name, organism, email=None):
    if not _local_organism(organism):
        return official.get_gene_coding_sequence(gene_name, organism, email)
    genome = _reference("EcoliK12-NC_000913.3.gbwithparts")
    primary = [
        feature
        for feature in genome.features
        if feature.type == "CDS"
        and gene_name.casefold()
        in {name.casefold() for name in feature.qualifiers.get("gene", []) + feature.qualifiers.get("locus_tag", [])}
    ]
    sequences = []
    # Exact primary identifiers take priority over potentially colliding aliases.
    for feature in primary or genome.features:
        if feature.type != "CDS":
            continue
        names = feature.qualifiers.get("gene", []) + feature.qualifiers.get("locus_tag", [])
        for synonyms in feature.qualifiers.get("gene_synonym", []):
            names.extend(part.strip() for part in synonyms.split(";"))
        if gene_name.casefold() in {name.casefold() for name in names}:
            sequences.append(
                {
                    "refseq_id": genome.id,
                    "sequence": str(feature.extract(genome.seq)),
                }
            )
    if not sequences:
        raise ValueError(f"Gene {gene_name!r} is absent from local {genome.id}")
    return {
        "explanation": "sequences contains refseq_id and strand-correct CDS sequence.",
        "sequences": sequences,
    }


def get_plasmid_sequence(identifier, is_addgene=None):
    if is_addgene is not True and identifier.strip().lower() in {
        "puc19",
        "m77789",
        "m77789.2",
    }:
        record = _reference("pUC19-M77789.2.gb")
        return {
            "explanation": "source, identifier and complete plasmid sequence.",
            "source": "NCBI",
            "identifier": record.id,
            "sequence": str(record.seq),
        }
    # Addgene is outside the fixed local snapshot; keep the official route but
    # bound its otherwise unbounded requests.get call.
    return official.get_plasmid_sequence(identifier, is_addgene)


class _RequestsProxy:
    @staticmethod
    def get(*args, **kwargs):
        kwargs.setdefault("timeout", (5, 30))
        return requests.get(*args, **kwargs)


official.requests = _RequestsProxy()


def _normalize(value):
    if hasattr(value, "_asdict"):
        return {key: _normalize(val) for key, val in value._asdict().items()}
    if isinstance(value, dict):
        return {key: _normalize(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize(val) for val in value]
    return value


def _redact_value(value):
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


S = {"type": "string", "minLength": 1, "maxLength": MAX_SEQUENCE}
B = {"type": "boolean"}
NAMES = {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 100}, "minItems": 1, "maxItems": 100}


def _spec(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SPECS = primitives.specs(_spec) + [
    _spec(
        "get_gene_coding_sequence",
        "Retrieve gene CDS sequences by gene name and organism. "
        "E. coli defaults to the fixed K-12 MG1655 reference; other organisms use NCBI.",
        {"gene_name": S, "organism": S, "email": {"type": ["string", "null"]}},
        ["gene_name", "organism"],
    ),
    _spec(
        "get_plasmid_sequence",
        "Retrieve a plasmid sequence by name or accession, or Addgene ID. "
        "Uses a fixed local reference when available. is_addgene can force the source.",
        {"identifier": S, "is_addgene": {"type": ["boolean", "null"]}},
        ["identifier"],
    ),
]
FUNCTIONS = {
    **primitives.FUNCTIONS,
    "get_gene_coding_sequence": get_gene_coding_sequence,
    "get_plasmid_sequence": get_plasmid_sequence,
    **extra.FUNCTIONS,
}
TOOL_SPECS.extend(extra.specs(_spec))
TOOL_SPECS.sort(key=lambda spec: spec["function"]["name"])
VALIDATORS = {
    spec["function"]["name"]: jsonschema.Draft202012Validator(spec["function"]["parameters"]) for spec in TOOL_SPECS
}


def invoke(name, arguments):
    backend = "local"
    try:
        if name not in FUNCTIONS:
            return {"status": "unknown_tool", "error": f"Unknown tool: {name}"}
        VALIDATORS[name].validate(arguments)
        if name in extra.EXTRA_NAMES:
            arguments = extra.prepare(name, arguments)
        if name in extra.ONLINE_NAMES:
            backend = "online"
        if name == "get_gene_coding_sequence" and not _local_organism(arguments["organism"]):
            backend = "online"
        if name == "get_plasmid_sequence" and (
            arguments.get("is_addgene") is True
            or arguments["identifier"].strip().lower() not in {"puc19", "m77789", "m77789.2"}
        ):
            backend = "online"
        result = _normalize(FUNCTIONS[name](**arguments))
        if name in extra.ONLINE_NAMES:
            # verbose=False may return the official formatted result directly.
            status = result.get("status", "ok") if isinstance(result, dict) else "ok"
            result = _redact_value(result)
            if status != "ok":
                return {"status": status, "source": name, "backend_source": backend, "result": result}
        if result is None:
            return {"status": "not_found", "source": name, "backend_source": backend}
        encoded = json.dumps(result, ensure_ascii=False)
        if len(encoded.encode()) > MAX_REPLY_BYTES:
            return {
                "status": "result_too_large",
                "source": name,
                "backend_source": backend,
                "error": "Result exceeds 48 KB; narrow the request.",
            }
        return {"status": "ok", "source": name, "backend_source": backend, "result": result}
    except jsonschema.ValidationError as exc:
        # Do not echo user sequences in schema errors.
        error = f"Invalid parameter at {list(exc.path)}: {exc.validator}"
        status = "invalid_arguments"
    except (ValueError, TypeError, AttributeError) as exc:
        error, status = str(exc), "invalid_arguments"
    except (requests.RequestException, OSError) as exc:
        error, status = str(exc), "backend_error"
    except Exception as exc:
        # Official Entrez helpers wrap transport exceptions in RuntimeError.
        cause = exc.__cause__
        status = "backend_error" if isinstance(cause, (requests.RequestException, OSError)) else "tool_error"
        error = f"{type(exc).__name__}: {exc}"
    return {"status": status, "source": name, "backend_source": backend, "error": _redact(error)[:1000]}
