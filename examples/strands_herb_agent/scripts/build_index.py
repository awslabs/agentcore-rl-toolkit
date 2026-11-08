"""Build the retrieval index from the downloaded HERB data: HERB's documents, SentenceSplitter() nodes, and their
Cohere Embed v4 embeddings (about 34,500 nodes; Bedrock calls only).

    python scripts/build_index.py herb index
"""

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from llama_index.core.node_parser import SentenceSplitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import retrieval  # noqa: E402


def main(herb: str, out: str):
    herb, out = Path(herb), Path(out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    docs = retrieval.build_documents(herb / "data" / "products")
    nodes = SentenceSplitter().get_nodes_from_documents(docs)
    print(f"{len(docs)} documents, {len(nodes)} nodes", flush=True)
    with open(out / "nodes.jsonl", "w") as f:
        for n in nodes:
            rel = {k.value: v.model_dump() for k, v in n.relationships.items()}
            f.write(json.dumps({"id_": n.node_id, "text": n.text, "relationships": rel}, ensure_ascii=False) + "\n")
    texts = [n.get_content() for n in nodes]
    batches = [texts[i : i + retrieval.BATCH] for i in range(0, len(texts), retrieval.BATCH)]
    with ThreadPoolExecutor(4) as ex:
        emb = np.concatenate(list(ex.map(lambda b: retrieval.embed(b, "search_document"), batches)))
    np.save(out / "embeddings.npy", emb)
    manifest = {
        "embedding_model": retrieval.EMBED_MODEL,
        "dimension": int(emb.shape[1]),
        "documents": len(docs),
        "nodes": len(nodes),
        "splitter": "SentenceSplitter() defaults (1024/200)",
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "nodes.jsonl", out / "embeddings.npy")
        },
        "seconds": round(time.time() - t0),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) > 2 else ("herb", "index")))
