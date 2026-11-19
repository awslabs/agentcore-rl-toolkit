"""The CompLit environment: arxiv_search and read_paper, ported from upstream complit/tools/arxiv.py
(dayoon-ko/CompLit@2654a93f, MIT): the same queries, result formatting, full-text extraction and truncation.

- arxiv_search(query, max_results=10): a broad query (upstream: all:<query>) and, when the query has at least 2
  content words, a strict AND of its first 4, interleaved strict-first, as upstream. Both run over a local index of
  a frozen arXiv metadata snapshot (build_index.py), not the arXiv API, which rate-limits shared egress addresses:
  the broad query is an OR of the query's words, ranked by BM25 over title, authors, abstract, comments,
  journal-ref and categories (porter stemming); the strict query requires all its terms, so the ranking is not
  arXiv's own. Queries run in worker processes (COMPLIT_SEARCH_WORKERS): SQLite queries in threads of one process
  run one at a time.
- read_paper(arxiv_id): the paper's ar5iv HTML as text (the <article> or ltx_document body), cut at 24,000
  characters. Requests start at least COMPLIT_AR5IV_MIN_INTERVAL seconds apart (upstream does not pace ar5iv), and
  pages are cached on disk by URL (COMPLIT_CACHE_DIR), so every session and model reads the same text. HTTP 429 and
  5xx are retried with backoff; if they persist, InfraError (the episode is voided, not shown to the model). A
  timeout or another transport error is returned to the model with upstream's message, as upstream.
"""

import fcntl
import hashlib
import http.client
import json
import multiprocessing
import os
import re
import sqlite3
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from itertools import zip_longest
from pathlib import Path

INDEX = Path(os.environ.get("COMPLIT_INDEX", "index/arxiv.sqlite"))
CACHE = Path(os.environ.get("COMPLIT_CACHE_DIR", "cache"))
AR5IV_MIN_INTERVAL = float(os.environ.get("COMPLIT_AR5IV_MIN_INTERVAL", "1"))
SEARCH_WORKERS = int(os.environ.get("COMPLIT_SEARCH_WORKERS", "10"))
AR5IV_URL = "https://ar5iv.labs.arxiv.org/html/{}"
MAX_PAPER_CHARS = 24000
THROTTLE = Path(tempfile.gettempdir()) / "complit_ar5iv.throttle"
RETRY_DELAYS = (5, 15, 30, 60)
RETRYABLE = frozenset({429, 500, 502, 503, 504})
UA = "paper-search-eval/1.0 (benchmark evaluation)"
DUAL_QUERY = True
AND_TERMS = 4
_QUERY_STOPWORDS = frozenset(
    """a an the of for and or to in on with using via based by from that
this these those is are was were be been we our their its it as at into over under between within
can could should would may might will shall do does did not no than then there here which who whom
whose what when where why how paper method approach model models study work research propose
proposed novel new results result show shows shown also more most such other others only very""".split()
)


class InfraError(Exception):
    """arXiv / ar5iv unavailable after retries: not the model's fault."""


def _request(url: str, timeout: int):
    """(status, body, content type) of one request; a transport failure (timeout, connection) raises."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace"), resp.headers.get("content-type", "")
    except urllib.error.HTTPError as e:
        return e.code, "", ""


def _pace() -> None:
    """Wait until COMPLIT_AR5IV_MIN_INTERVAL has passed since the last request started on this node."""
    interval = AR5IV_MIN_INTERVAL
    with open(THROTTLE, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            lock.seek(0)
            prev = float(lock.read().strip() or 0)
            time.sleep(max(0.0, interval - (time.time() - prev)))
            lock.seek(0)
            lock.truncate()
            lock.write(repr(time.time()))
            lock.flush()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _get(url: str, timeout: int):
    """(body, content type) of url: from the cache, else paced requests, retried on 429 / 5xx. A 404 is cached as an
    empty body; another client error raises ValueError and a transport failure OSError (both reported to the model);
    429 / 5xx that persist through the retries are infrastructure (InfraError)."""
    path = CACHE / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
    if path.exists():
        hit = json.loads(path.read_text())
        return hit["body"], hit["content_type"]
    CACHE.mkdir(parents=True, exist_ok=True)
    for delay in (*RETRY_DELAYS, None):
        _pace()
        status, body, ctype = _request(url, timeout)
        if status not in RETRYABLE:
            break
        if delay is None:
            raise InfraError(f"{url} failed after retries (HTTP {status})")
        time.sleep(delay)
    if status not in (200, 404):
        raise ValueError(f"HTTP Error {status}")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "url": url,
                "status": status,
                "content_type": ctype,
                "body": body,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
    )
    tmp.replace(path)
    return body, ctype


def _words(query: str) -> list:
    return re.findall(r"\w+", query)


def _match(terms: list, op: str) -> str:
    return f" {op} ".join('"' + t.replace('"', "") + '"' for t in terms)


def _search(match: str, max_results: int) -> list:
    """The top entries of an FTS5 query, BM25-ranked: [{"id", "title", "published", "authors", "summary"}]."""
    if not match:
        return []
    con = sqlite3.connect(f"file:{INDEX}?mode=ro&immutable=1", uri=True)
    try:
        rows = con.execute(
            "SELECT arxiv_id, title, published, names, abstract FROM papers WHERE papers MATCH ? ORDER BY rank LIMIT ?",
            (match, max_results),
        ).fetchall()
    finally:
        con.close()
    return [
        {"id": r[0], "title": r[1], "published": r[2], "authors": [a for a in r[3].split("; ") if a], "summary": r[4]}
        for r in rows
    ]


_POOL, _POOL_LOCK = None, threading.Lock()


def _pool() -> ProcessPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ProcessPoolExecutor(SEARCH_WORKERS, mp_context=multiprocessing.get_context("spawn"))
        return _POOL


def _and_query(query: str, k: int = AND_TERMS) -> list:
    seen, terms = set(), []
    for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", query.lower()):
        if t in _QUERY_STOPWORDS or len(t) <= 2 or t in seen:
            continue
        seen.add(t)
        terms.append(t)
        if len(terms) == k:
            break
    return terms if len(terms) >= 2 else []


def _interleave(primary: list, secondary: list, limit: int) -> list:
    merged, seen = [], set()
    for a, b in zip_longest(primary, secondary):
        for e in (a, b):
            if e is None:
                continue
            if e["id"] in seen:
                continue
            seen.add(e["id"])
            merged.append(e)
            if len(merged) >= limit:
                return merged
    return merged


def _format_entries(entries: list) -> str:
    if not entries:
        return "No results found. Try different or broader search terms."
    blocks = []
    for idx, e in enumerate(entries, 1):
        authors = e["authors"]
        authors_str = ", ".join(authors[:6]) + (" et al." if len(authors) > 6 else "")
        summary = e["summary"]
        if len(summary) > 500:
            summary = summary[:500] + "..."
        blocks.append(
            f"[{idx}] {e['title']}\n    arxiv_id: {e['id']} | year: {e['published'][:4]}\n"
            f"    authors: {authors_str}\n    abstract: {summary}"
        )
    return "\n\n".join(blocks)


def _run_queries(broad: str, strict: str, max_results: int):
    global _POOL
    try:
        jobs = [_pool().submit(_search, m, max_results) for m in (broad, strict)]
        return [j.result() for j in jobs]
    except BrokenProcessPool as e:  # a worker died: infrastructure, and a fresh pool next time
        with _POOL_LOCK:
            _POOL = None
        raise InfraError(f"search worker failed: {e}") from None


def arxiv_search(query: str, max_results: int = 10) -> str:
    max_results = max(1, min(int(max_results), 25))
    strict = _and_query(query) if DUAL_QUERY else []
    broad, strict_entries = _run_queries(
        _match(_words(query), "OR"), _match(strict, "AND") if strict else "", max_results
    )
    if strict_entries:
        return _format_entries(_interleave(strict_entries, broad, max_results))
    return _format_entries(broad)


def _html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    root = soup.find("article") or soup.find("div", class_="ltx_document") or soup.body or soup
    return " ".join(root.get_text(" ").split())


def read_paper(arxiv_id: str) -> str:
    if not arxiv_id or not str(arxiv_id).strip():
        return "No arxiv_id provided. Pass an id from arxiv_search results."
    aid = re.sub(r"v\d+$", "", str(arxiv_id).strip())
    try:
        html, ctype = _get(AR5IV_URL.format(aid), timeout=60)
    except (OSError, ValueError, http.client.HTTPException) as e:  # upstream's _read_error
        return f"Failed to read paper {arxiv_id} ({type(e).__name__}: {e}). Try again or pick another candidate."
    head = html[:2000]
    missing = "This paper does not exist" in head or "Conversion failed" in head
    usable = html and "html" in ctype.lower() and not missing
    text = _html_to_text(html) if usable else ""
    if not text:
        return (
            f"Full text for {arxiv_id} is unavailable on ar5iv. Rely on the abstract from arxiv_search, or try "
            "another candidate."
        )
    return text[:MAX_PAPER_CHARS] + "\n...[truncated]" if len(text) > MAX_PAPER_CHARS else text
