"""Build the arxiv_search index from a frozen arXiv metadata snapshot.

The snapshot is librarian-bots/arxiv-metadata-snapshot (the arXiv OAI metadata, CC0) at a pinned revision: its 11
parquet files are downloaded to DATA_DIR if missing. The index is one SQLite FTS5 table (porter stemming, BM25) over
the fields arXiv's `all:` prefix searches: title, authors, abstract, comments, journal-ref and categories. Each row
also keeps what the arXiv API returns for an entry: the id with its latest version, the first version's date and the
author names. About 15 minutes and 8 GB; replaces an existing index.

    python build_index.py --data-dir /data/arxiv-metadata --out /data/index/arxiv.sqlite
"""

import argparse
import email.utils
import sqlite3
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

SNAPSHOT = (
    "https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot/resolve/"
    "c428226b689a0e17db2cd55265133b9eca15d97f/data/train-{:05d}-of-00011.parquet"
)
COLUMNS = ["id", "authors", "title", "comments", "journal-ref", "categories", "abstract", "versions", "authors_parsed"]


def clean(v) -> str:
    return "" if v in (None, "None") else " ".join(str(v).split())


def row(r: dict) -> tuple:
    versions = r["versions"] or []
    first = versions[0]["created"] if versions else ""
    published = email.utils.parsedate_to_datetime(first).strftime("%Y-%m-%dT%H:%M:%SZ") if first else ""
    latest = versions[-1]["version"] if versions else ""
    names = "; ".join(" ".join(p for p in (a[1], a[0], a[2]) if p) for a in (r["authors_parsed"] or []))
    return (
        r["id"] + latest,
        published,
        names,
        clean(r["title"]),
        clean(r["authors"]),
        clean(r["abstract"]),
        clean(r["comments"]),
        clean(r["journal-ref"]),
        clean(r["categories"]),
    )


def download(data_dir: Path) -> list:
    data_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(11):
        path = data_dir / f"train-{i:05d}-of-00011.parquet"
        if not path.exists():
            print(f"downloading {path.name}", flush=True)
            urllib.request.urlretrieve(SNAPSHOT.format(i), path.with_suffix(".part"))
            path.with_suffix(".part").replace(path)
        files.append(path)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    files = download(a.data_dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute(
        "CREATE VIRTUAL TABLE papers USING fts5(arxiv_id UNINDEXED, published UNINDEXED, names UNINDEXED, "
        "title, authors, abstract, comments, journal_ref, categories, tokenize='porter unicode61')"
    )
    t0, n = time.time(), 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=50000, columns=COLUMNS):
            rows = [row(r) for r in batch.to_pylist()]
            con.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?)", rows)
            n += len(rows)
        con.commit()
        print(f"{f.name}: {n} papers, {time.time() - t0:.0f} s", flush=True)
    con.execute("INSERT INTO papers(papers) VALUES('optimize')")
    con.commit()
    con.close()
    tmp.replace(a.out)
    print(f"{a.out}: {n} papers, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
