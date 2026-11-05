"""CompLit's two tools, served by the environment service (env/serve.py) at COMPLIT_ENV_URL.

Names, descriptions and parameters are upstream's (complit/tools/arxiv.py, dayoon-ko/CompLit@2654a93f, MIT). An
unreachable service or a 503 (ar5iv unavailable after retries) is recorded in `state`, so the app stops the episode
and fails the rollout instead of showing the error to the model.
"""

import json
import os
import urllib.error
import urllib.request

from strands import tool

ENV_URL = os.environ.get("COMPLIT_ENV_URL", "http://127.0.0.1:18731")

SEARCH = {
    "description": "Search arXiv for papers matching a query. Returns a numbered list of matching papers with title, "
    "arxiv_id, year, authors and a truncated abstract.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text search terms (keywords, concepts, author names, etc.). "
                "Searches titles, abstracts, authors and other fields.",
            },
            "max_results": {"type": "integer", "description": "Number of results to return (default 10, keep modest)."},
        },
        "required": ["query"],
    },
}
READ = {
    "description": "Read the full text of one arXiv paper to verify it against the description. Use this once you have "
    "a promising candidate from arxiv_search: fetch its full text and check whether it actually satisfies the "
    "constraints in the description before reporting it. Returns the paper's full text (title, abstract, body, "
    "tables), truncated if long.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "arxiv_id": {
                "type": "string",
                "description": 'The arXiv id of the paper (e.g. "2106.09063" or "2106.09063v4"), as shown in '
                "arxiv_search results.",
            }
        },
        "required": ["arxiv_id"],
    },
}


def make_tools(state: dict) -> list:
    def call(name: str, args: dict) -> str:
        request = urllib.request.Request(
            f"{ENV_URL}/{name}", data=json.dumps(args).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read())["output"]
        except urllib.error.HTTPError as e:
            error = json.loads(e.read() or b"{}").get("error", str(e))
            if e.code == 400:
                return f"Error: {error}"
            state["infra_error"] = error
        except OSError as e:
            state["infra_error"] = f"environment service unreachable: {e}"
        return "The tool is unavailable."

    @tool(name="arxiv_search", **SEARCH)
    def arxiv_search(query: str, max_results: int = 10) -> str:
        return call("arxiv_search", {"query": query, "max_results": max_results})

    @tool(name="read_paper", **READ)
    def read_paper(arxiv_id: str) -> str:
        return call("read_paper", {"arxiv_id": arxiv_id})

    return [arxiv_search, read_paper]
