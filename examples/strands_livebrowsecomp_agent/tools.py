"""LiveBrowseComp's tools: `search` and `visit` on the live web, as in the paper (Appendix C).

- search: the AgentCore Gateway Web Search Tool (LIVEBROWSECOMP_GATEWAY_URL, IAM authorization), 10 results.
- visit: an HTTP fetch with text extraction (HTML, PDF via pypdf, JSON as text), then a summary toward the agent's
  goal by a fixed summarizer model (gpt-oss-120b on Bedrock, reasoning dropped). The summarizer is part of the
  environment, the same for every policy. Wikipedia is fetched live unless WIKIPEDIA_SERVICE_URL names a service
  that serves it from local dumps (GET <service>/fetch?url=<url> -> {"text"}). Live Wikipedia often refuses
  AgentCore Runtime's addresses; the refusal goes to the model like any page error.

Leakage control: huggingface.co and the benchmark's site are excluded from search; results and pages of its
repository and paper, or that hold its canary string, are dropped. Tool output is cut at 40,000 characters. Rate
limiting (outside Wikimedia) and a failing gateway or summarizer are infrastructure: recorded in `state`, so the app
fails the rollout instead of showing the model.
"""

import io
import json
import os
import re
import time
from urllib.parse import quote, urlparse

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.config import Config
from bs4 import BeautifulSoup
from strands import tool

GATEWAY_URL = os.environ.get("LIVEBROWSECOMP_GATEWAY_URL", "")
WIKIPEDIA_SERVICE_URL = os.environ.get("WIKIPEDIA_SERVICE_URL", "")
BLOCKED_DOMAINS = ["huggingface.co", "livebrowsecomp.github.io"]
BLOCKED_URL_PREFIXES = [
    "github.com/livebrowsecomp",
    "arxiv.org/abs/2605.28721",
    "arxiv.org/html/2605.28721",
    "arxiv.org/pdf/2605.28721",
    "alphaxiv.org/abs/2605.28721",
]
BLOCKED_SUBSTRINGS = ["livebrowsecomp:7a3f9c21"]  # the dataset's canary string, lowercase as matched
MAX_QUERY_CHARS = 200  # the Web Search Tool rejects longer queries
MAX_TOOL_CHARS = 40000
SUMMARIZER = "openai.gpt-oss-120b-1:0"
SUMMARIZER_MAX_TOKENS = 12288  # reasoning + summary (gpt-oss counts both)
MAX_PAGE_CHARS = 150_000  # page text given to the summarizer (~40K tokens)
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
WIKI_UA = "agentcore-rl-toolkit-example/0.1 (research benchmark evaluation; python-httpx)"  # Wikimedia UA policy
WIKIMEDIA = ("wikipedia.org", "wikimedia.org", "wikidata.org", "wiktionary.org", "wikisource.org")

SUMMARY_PROMPT = """You are reading a web page on behalf of a research agent. Extract the information on the page that is relevant to the agent's goal.

Goal: {goal}
URL: {url}

Page content:
<<<
{text}
>>>

Instructions:
- Report everything on the page that is relevant to the goal. Quote key facts, names, numbers, and dates exactly as they appear on the page.
- If the relevant information is a list or a table, or the goal asks for one, reproduce every relevant item or row in full. Do not drop, merge, or abbreviate items.
- Use only the page content; do not add outside knowledge or guesses.
- If the page contains nothing relevant to the goal, say so and briefly describe what the page contains.{truncated_note}"""  # noqa: E501

_bedrock = boto3.client("bedrock-runtime", config=Config(retries={"max_attempts": 8, "mode": "adaptive"}))


class InfraError(Exception):
    """A rate-limited site, the search gateway or the summarizer is unavailable: not the model's fault."""


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def on(host: str, domains) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def blocked_url(url: str) -> bool:
    bare = re.sub(r"^https?://(www\.)?", "", (url or "").lower())
    return on(host_of(url), BLOCKED_DOMAINS) or any(bare.startswith(p) for p in BLOCKED_URL_PREFIXES)


def blocked_text(text: str) -> bool:
    return any(s in (text or "").lower() for s in BLOCKED_SUBSTRINGS)


def gateway(method: str, params: dict) -> dict | int:
    """One SigV4-signed MCP call to the gateway -> its JSON-RPC result, or the HTTP status of a client error."""
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    request = AWSRequest(method="POST", url=GATEWAY_URL, data=body, headers=headers)
    SigV4Auth(boto3.Session().get_credentials(), "bedrock-agentcore", host_of(GATEWAY_URL).split(".")[-3]).add_auth(
        request
    )
    try:
        r = httpx.post(GATEWAY_URL, content=body, headers=dict(request.headers), timeout=60)
    except httpx.HTTPError as e:
        raise InfraError(f"search gateway unreachable ({type(e).__name__})") from None
    if r.status_code == 429 or r.status_code >= 500:
        raise InfraError(f"search HTTP {r.status_code}: {r.text[:200]}")
    if r.status_code != 200:
        return r.status_code
    try:
        reply = r.json()
    except ValueError:
        raise InfraError(f"search returned a malformed response: {r.text[:120]!r}") from None
    if "error" in reply:
        raise InfraError(f"search error: {str(reply['error'])[:200]}")
    return reply["result"]


_search_tool = None


def search_tool_name() -> str:
    """The gateway's Web Search Tool, named <target>___WebSearch."""
    global _search_tool
    if _search_tool is None:
        listed = gateway("tools/list", {})
        if isinstance(listed, int):  # e.g. 403: the caller may not invoke the gateway
            raise InfraError(f"search gateway tools/list HTTP {listed}")
        _search_tool = next(t["name"] for t in listed["tools"] if t["name"].endswith("___WebSearch"))
    return _search_tool


def search(query: str) -> str:
    if len(query) > MAX_QUERY_CHARS:  # the backend's own validation error, returned as an API would
        return f"Error: the search query is too long ({len(query)} characters; at most {MAX_QUERY_CHARS})."
    arguments = {"query": query, "maxResults": 10, "filters": {"domainFilter": {"exclude": BLOCKED_DOMAINS}}}
    result = gateway("tools/call", {"name": search_tool_name(), "arguments": arguments})
    if isinstance(result, int):
        return f"Error: search failed (HTTP {result})."
    try:
        content = result["content"][0]["text"]
        if content.startswith("DependencyFailedException"):  # the search backend failed, e.g. its own 429
            raise InfraError(f"search dependency failed: {content[:200]}")
        payload = json.loads(content)
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise InfraError(f"search returned a malformed result ({type(e).__name__})") from None
    results = [
        r
        for r in payload.get("results", [])
        if not blocked_url(r.get("url", "")) and not blocked_text(f"{r.get('title', '')} {r.get('text', '')}")
    ]
    if not results:
        return "No results found."
    return "\n\n".join(
        f"[{i}] {r.get('title', '')}\nURL: {r.get('url', '')}\n{r.get('text', '')}" for i, r in enumerate(results, 1)
    )


def html_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        t.decompose()
    return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip() or "Error: no extractable text on this page."


def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    try:
        text = "\n".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(data)).pages)
    except Exception as e:  # noqa: BLE001 - a malformed PDF is a property of the page, not infra
        return f"Error: could not read the PDF ({type(e).__name__})."
    return re.sub(r"\n{3,}", "\n\n", text).strip() or "Error: no extractable text in this PDF."


def fetch(url: str) -> str:
    """Page text, or a string starting with 'Error:' (a property of the page, returned to the model). A 429 is
    retried with backoff (4-32 s) and then raised as infrastructure. Wikimedia sites answer bursts, and often cloud
    addresses such as AgentCore Runtime's, with 403 or 429: retried twice, then returned to the model."""
    host = host_of(url)
    if WIKIPEDIA_SERVICE_URL and on(host, ("wikipedia.org",)):
        try:
            r = httpx.get(f"{WIKIPEDIA_SERVICE_URL}/fetch?url={quote(url, safe='')}", timeout=60)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise InfraError(f"Wikipedia service: {type(e).__name__}") from None
        return r.json()["text"]
    wiki = on(host, WIKIMEDIA)
    attempts = 3 if wiki else 5
    for attempt in range(attempts):
        try:
            r = httpx.get(url, headers={"User-Agent": WIKI_UA if wiki else UA}, timeout=30, follow_redirects=True)
        except httpx.TimeoutException:
            return "Error: request timed out."
        except httpx.HTTPError as e:
            return f"Error: could not fetch the page ({type(e).__name__})."
        if not (r.status_code == 429 or (wiki and r.status_code == 403)):
            break
        if attempt == attempts - 1:
            if wiki:
                return f"Error: HTTP {r.status_code}."
            raise InfraError(f"fetch rate-limited (HTTP {r.status_code}): {url[:120]}")
        time.sleep(4 * 2**attempt)
    if r.status_code >= 400:
        return f"Error: HTTP {r.status_code}."
    ctype = r.headers.get("content-type", "").lower()
    if "pdf" in ctype or r.content[:5] == b"%PDF-":
        return pdf_text(r.content)
    if "json" in ctype:
        try:
            return json.dumps(r.json(), ensure_ascii=False, indent=1)
        except ValueError:
            return r.text
    if "html" not in ctype and "text" not in ctype:
        return f"Error: unsupported content type ({ctype})."
    return html_text(r.text)


def summarize(url: str, goal: str, text: str) -> str:
    note = "\n- The page was truncated; mention that the summary covers only the first part of the page."
    prompt = SUMMARY_PROMPT.format(
        goal=goal, url=url, text=text[:MAX_PAGE_CHARS], truncated_note=note if len(text) > MAX_PAGE_CHARS else ""
    )
    try:
        r = _bedrock.converse(
            modelId=SUMMARIZER,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": SUMMARIZER_MAX_TOKENS},
        )
    except _bedrock.exceptions.ValidationException as e:  # e.g. a page the model rejects: a property of the page
        return f"Error: the page could not be summarized ({str(e)[:200]})."
    except Exception as e:  # noqa: BLE001
        raise InfraError(f"summarizer: {type(e).__name__}: {str(e)[:200]}") from None
    return "".join(b["text"] for b in r["output"]["message"]["content"] if "text" in b).strip()  # drop reasoning


def visit(url: str, goal: str) -> str:
    if blocked_url(url):
        return "Error: this URL is not accessible."
    text = fetch(url)
    if text.startswith("Error:"):
        return text
    if blocked_text(text):
        return "Error: this URL is not accessible."
    summary = summarize(url, goal, text)
    return f"URL: {url}\n\n{summary}" if summary else "Error: the page could not be summarized."


def make_tools(state: dict) -> list:
    def guarded(fn, *args) -> str:
        try:
            out = fn(*args)
        except InfraError as e:
            state["infra_error"] = str(e)
            return "The tool is unavailable."
        return out[:MAX_TOOL_CHARS] + "\n[output truncated]" if len(out) > MAX_TOOL_CHARS else out

    @tool(
        name="search",
        description="Perform a web search. Returns up to 10 results, each with its URL and a text snippet.",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    )
    def search_tool(query: str) -> str:
        return guarded(search, query)

    @tool(
        name="visit",
        description="Open a web page (HTML, PDF, or JSON) and return its content summarized toward the given "
        "information goal.",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL of the page to open."},
                "goal": {
                    "type": "string",
                    "description": "The specific information you want to obtain from this page.",
                },
            },
            "required": ["url", "goal"],
        },
    )
    def visit_tool(url: str, goal: str) -> str:
        return guarded(visit, url, goal)

    return [search_tool, visit_tool]
