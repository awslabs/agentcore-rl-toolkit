"""HSCodeComp's tools: web_search, web_extract and the three official HTS tools (hts_tools.py).

- web_search: the AgentCore Gateway Web Search Tool (HSCODECOMP_GATEWAY_URL, IAM authorization), 5 results.
  Leakage control: huggingface.co is excluded by the request's domain filter; URLs under github.com/AIDC-AI (the
  benchmark's repository) are dropped here, since the filter matches whole domains.
- web_extract: an HTTP fetch with text extraction (HTML, PDF via pypdf, JSON as text). CROSS ruling pages are a
  JavaScript app, so ruling URLs are served from the CROSS JSON API, which returns the full ruling text.

Tool output is cut at 20,000 characters. A failing gateway (HTTP 429 / 5xx, a malformed reply) is recorded in
`state`, so the app stops the episode and fails the rollout instead of showing the error to the model.
"""

import io
import json
import os
import re
from urllib.parse import urlparse

import boto3
import hts_tools
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from bs4 import BeautifulSoup
from strands import tool

GATEWAY_URL = os.environ.get("HSCODECOMP_GATEWAY_URL", "")
EXCLUDE_DOMAINS = ["huggingface.co"]
EXCLUDE_URL_PREFIXES = ["https://github.com/aidc-ai/", "http://github.com/aidc-ai/", "https://www.github.com/aidc-ai/"]
MAX_RESULTS = 5
MAX_TOOL_CHARS = 20000
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
CROSS_RULING = re.compile(r"rulings\.cbp\.gov/(?:ruling|api/ruling)/([A-Za-z0-9]+)")


class InfraError(Exception):
    """The search gateway or the CROSS API is unavailable: not the model's fault."""


def gateway(method: str, params: dict) -> dict | int:
    """One SigV4-signed MCP call to the gateway -> its JSON-RPC result, or the HTTP status of a client error."""
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    request = AWSRequest(method="POST", url=GATEWAY_URL, data=body, headers=headers)
    region = urlparse(GATEWAY_URL).hostname.split(".")[-3]
    SigV4Auth(boto3.Session().get_credentials(), "bedrock-agentcore", region).add_auth(request)
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
        names = [t["name"] for t in listed["tools"]]
        _search_tool = next(n for n in names if n.endswith("___WebSearch"))
    return _search_tool


def blocked(url: str) -> bool:
    u = (url or "").lower()
    host = urlparse(u).hostname or ""
    return any(host == d or host.endswith("." + d) for d in EXCLUDE_DOMAINS) or any(
        u.startswith(p) for p in EXCLUDE_URL_PREFIXES
    )


def web_search(query: str) -> str:
    arguments = {"query": query, "maxResults": MAX_RESULTS, "filters": {"domainFilter": {"exclude": EXCLUDE_DOMAINS}}}
    result = gateway("tools/call", {"name": search_tool_name(), "arguments": arguments})
    if isinstance(result, int):
        return f"Error: search failed (HTTP {result})."
    try:
        payload = json.loads(result["content"][0]["text"])
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise InfraError(f"search returned a malformed result ({type(e).__name__})") from None
    results = [r for r in payload.get("results", []) if not blocked(r.get("url", ""))]
    if not results:
        return "No results found."
    return "\n\n".join(
        f"[{i}] {r.get('title', '')}\nURL: {r.get('url', '')}\n{r.get('text', '')}" for i, r in enumerate(results, 1)
    )


def web_extract(url: str) -> str:
    if blocked(url):
        return "Error: this URL is not accessible."
    m = CROSS_RULING.search(url)
    try:
        if m:
            r = httpx.get(
                f"https://rulings.cbp.gov/api/ruling/{m.group(1)}",
                headers={"User-Agent": UA},
                timeout=30,
                follow_redirects=True,
            )
            if r.status_code == 429 or r.status_code >= 500:
                raise InfraError(f"CROSS HTTP {r.status_code}")
            if r.status_code != 200:
                return f"Error: ruling {m.group(1)} not found."
            text = (r.json() or {}).get("text", "")
            return re.sub(r"[\r\f]+", "\n", text).strip() or "Error: empty ruling."
        r = httpx.get(url, headers={"User-Agent": UA}, timeout=30, follow_redirects=True)
    except httpx.TimeoutException:
        return "Error: request timed out."
    except httpx.HTTPError as e:
        return f"Error: could not fetch the page ({type(e).__name__})."
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
    soup = BeautifulSoup(r.text, "lxml")
    for t in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        t.decompose()
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip()
    return text or "Error: no extractable text on this page."


def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    try:
        text = "\n".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(data)).pages)
    except Exception as e:  # noqa: BLE001 - a malformed PDF is a property of the page, not infra
        return f"Error: could not read the PDF ({type(e).__name__})."
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or "Error: no extractable text in this PDF."


def make_tools(state: dict) -> list:
    def guarded(fn, *args) -> str:
        try:
            out = fn(*args)
        except InfraError as e:
            state["infra_error"] = str(e)
            return "The tool is unavailable."
        return out[:MAX_TOOL_CHARS] + "\n[output truncated]" if len(out) > MAX_TOOL_CHARS else out

    @tool(
        name="web_search",
        description="Search the web. Returns the top results with title, URL, and an excerpt of the page text.",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    )
    def web_search_tool(query: str) -> str:
        return guarded(web_search, query)

    @tool(
        name="web_extract",
        description="Fetch a web page by URL and return its main text content.",
        inputSchema={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to fetch."}},
            "required": ["url"],
        },
    )
    def web_extract_tool(url: str) -> str:
        return guarded(web_extract, url)

    def spec(schema: dict) -> dict:
        return {"name": schema["name"], "description": schema["description"], "inputSchema": schema["parameters"]}

    @tool(**spec(hts_tools.HTS_SUBCATEGORIES_SCHEMA))
    def get_hs_subcategories(parent_code: str) -> str:
        return guarded(hts_tools.get_hs_subcategories, parent_code)

    @tool(**spec(hts_tools.HTS_CODE_INFO_SCHEMA))
    def get_hs_code_info(hs_code: str) -> str:
        return guarded(hts_tools.get_hs_code_info, hs_code)

    @tool(**spec(hts_tools.HTS_VALIDATE_SCHEMA))
    def validate_hs_code(hs_code: str) -> str:
        return guarded(hts_tools.validate_hs_code, hs_code)

    return [web_search_tool, web_extract_tool, get_hs_subcategories, get_hs_code_info, validate_hs_code]
