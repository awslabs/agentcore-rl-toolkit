"""The Wikipedia dumps service: Wikipedia pages from local Kiwix ZIM dumps over HTTP, for text_browser_view.

GET /fetch?url=<a Wikipedia URL> returns 200 {"text"}: what the live page would give as text (wikipedia_dump.py),
"Error: HTTP 404." for a page the dump does not have, "Error: HTTP 400." for an invalid parameter. A language
without a dump returns 404, so the agent fetches that page live. GET /health returns the languages served. The
service only reads the dumps. Run it on a host in the AgentCore Runtime's VPC, bound to that host's private address
(README).

    python serve.py --host 10.0.1.23 --zim en=/data/wikipedia_en_all_maxi_2026-08.zim
"""

import argparse
import json
import re
import signal
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup
from wikipedia_dump import Dumps

HOST = re.compile(r"^([a-z][a-z-]*)\.(?:m\.)?wikipedia\.org$")
dumps: Dumps


def html_text(html: str) -> str:
    """The agent's page text (tools.html_text)."""
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        t.decompose()
    return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip() or "Error: no extractable text on this page."


class Server(ThreadingHTTPServer):
    request_queue_size = 256  # many concurrent sessions; the default backlog is 5
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            return self._send(200, {"languages": sorted(dumps.paths)})
        if u.path != "/fetch":
            return self._send(404, {"error": "not found"})
        url = (parse_qs(u.query).get("url") or [""])[0]
        m = HOST.match((urlparse(url).hostname or "").lower())
        if not m or m.group(1) not in dumps.paths:
            return self._send(404, {"error": "no dump for this host"})
        try:
            text = dumps.fetch(m.group(1), url)
        except ValueError:  # an invalid parameter, e.g. a non-numeric limit, as the live site answers it
            text = "Error: HTTP 400."
        self._send(200, {"text": text})

    def _send(self, code, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def main():
    global dumps
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="the address to bind: the host's private address, or 127.0.0.1")
    ap.add_argument("--port", type=int, default=18732)
    ap.add_argument("--zim", action="append", required=True, metavar="LANG=PATH", help="a dump, e.g. en=/data/en.zim")
    a = ap.parse_args()
    dumps = Dumps(dict(z.split("=", 1) for z in a.zim), html_text)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    Server((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
