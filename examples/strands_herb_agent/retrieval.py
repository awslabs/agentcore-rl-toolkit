"""HERB's unstructured retrieval (`retrieve_all`), following its react.py:

- documents as HERB builds them: top-level Slack messages, documents, meeting transcripts and meeting chats (PRs and
  URLs are reachable only through the structured tools);
- llama_index `SentenceSplitter()` with its defaults (1024 tokens, overlap 200);
- the ensemble retriever: BM25 top 20, then dense top 20, union deduplicated by node id, BM25 first;
- the tool output: the nodes' content (MetadataMode.LLM) joined with blank lines, for the query string llama_index's
  RetrieverTool builds ("input is <query>\n").

The dense embeddings are Cohere Embed v4 on Bedrock (1536 dimensions; documents and queries with their input
types) in place of OpenAI text-embedding-3-large. The index (`scripts/build_index.py`) holds the nodes and their
embeddings.
"""

import json
import os
import time
from pathlib import Path

import boto3
import numpy as np
from botocore.config import Config

EMBED_MODEL = "us.cohere.embed-v4:0"
DIMENSION = 1536
BATCH = 96  # texts per request
TOP_K = 20
INDEX_DIR = Path(os.environ.get("HERB_INDEX_DIR", Path(__file__).parent / "index"))

_bedrock = None


def embed(texts: list[str], input_type: str) -> np.ndarray:
    """Unit-normalized embeddings; input_type is search_document or search_query."""
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-west-2"),
            config=Config(read_timeout=120, retries={"max_attempts": 6, "mode": "adaptive"}, max_pool_connections=64),
        )
    out = []
    for i in range(0, len(texts), BATCH):
        body = {
            "texts": texts[i : i + BATCH],
            "input_type": input_type,
            "embedding_types": ["float"],
            "output_dimension": DIMENSION,
            "truncate": "RIGHT",
        }
        for attempt in range(10):  # the per-minute token quota: back off and retry
            try:
                r = _bedrock.invoke_model(modelId=EMBED_MODEL, body=json.dumps(body))
                break
            except _bedrock.exceptions.ThrottlingException:
                if attempt == 9:
                    raise
                time.sleep(min(60, 2**attempt))
        out.extend(json.loads(r["body"].read())["embeddings"]["float"])
    v = np.asarray(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def build_documents(products_dir: Path) -> list:
    from llama_index.core import Document

    docs = []
    for product_file in sorted(Path(products_dir).glob("*.json")):
        d = json.loads(product_file.read_text())
        for m in d.get("slack", []):
            u = m["Message"]["User"]
            content = (
                f"Message ID: {u['utterranceID']}\nChannel Name: {m['Channel']['name']}\n"
                f"Timestamp: {u['timestamp']}\n\n{u['userId']}: {u['text']}"
            )
            docs.append(Document(text=content, id_=f"slack::{u['utterranceID']}"))
        for x in d.get("documents", []):
            content = (
                f"Doc ID: {x['id']}\nLink: {x['document_link']}\n{x['date']}\nAuthor: {x['author']}\n\n"
                f"{x['type']}\n{x['content']}"
            )
            docs.append(Document(text=content, id_=f"document::{x['id']}"))
        for x in d.get("meeting_transcripts", []):
            content = f"Meeting ID: {x['id']}\n{x['date']}\nParticipants: {x['participants']}\n\n{x['transcript']}"
            docs.append(Document(text=content, id_=f"transcript::{x['id']}"))
        for x in d.get("meeting_chats", []):
            content = f"Meeting ID: {x['id']}\n\nChats:\n{x['text']}"
            docs.append(Document(text=content, id_=f"chat::{x['id']}"))
    return docs


class HerbIndex:
    def __init__(self, index_dir: Path = INDEX_DIR):
        from llama_index.core.schema import TextNode
        from llama_index.retrievers.bm25 import BM25Retriever

        d = Path(index_dir)
        self.nodes = [TextNode(**json.loads(line)) for line in (d / "nodes.jsonl").read_text().splitlines()]
        self.emb = np.load(d / "embeddings.npy")
        self.bm25 = BM25Retriever.from_defaults(nodes=self.nodes, similarity_top_k=TOP_K)

    def retrieve(self, query: str) -> list:
        bm = self.bm25.retrieve(query)
        q = embed([query], "search_query")[0]
        top = np.argsort(-(self.emb @ q))[:TOP_K]
        seen, out = set(), []
        for n in [x.node for x in bm] + [self.nodes[i] for i in top]:
            if n.node_id not in seen:
                seen.add(n.node_id)
                out.append(n)
        return out

    def retrieve_all(self, input: str) -> str:  # noqa: A002 - the argument name of HERB's tool
        from llama_index.core.schema import MetadataMode

        return "".join(n.get_content(MetadataMode.LLM) + "\n\n" for n in self.retrieve(f"input is {input}\n"))
