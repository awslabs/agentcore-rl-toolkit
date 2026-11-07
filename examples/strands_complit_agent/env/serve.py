"""The CompLit environment service: arxiv_search and read_paper over HTTP, for the agent app's tools.

POST /arxiv_search {"query", "max_results"} and POST /read_paper {"arxiv_id"} return {"output": <the tool's text>}.
An infrastructure failure (ar5iv unavailable after retries) returns 503 with {"error"}; invalid arguments 400.
GET /health returns the number of indexed papers. The service only reads public data (the arXiv metadata index
and ar5iv pages). Run it on a host in the AgentCore Runtime's VPC, bound to that host's private address (README).

    COMPLIT_INDEX=/data/index/arxiv.sqlite COMPLIT_CACHE_DIR=/data/cache python serve.py --host 10.0.1.23
"""

import argparse
import json
import signal
import sqlite3
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import arxiv_tools

TOOLS = {"/arxiv_search": arxiv_tools.arxiv_search, "/read_paper": arxiv_tools.read_paper}


class Server(ThreadingHTTPServer):
    request_queue_size = 256  # many concurrent sessions; the default backlog is 5
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        fn = TOOLS.get(self.path)
        if fn is None:
            return self._send(404, {"error": f"unknown tool {self.path}"})
        try:
            args = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            output = fn(**args)
        except arxiv_tools.InfraError as e:
            return self._send(503, {"error": str(e)})
        except (TypeError, ValueError) as e:
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})
        self._send(200, {"output": output})

    def do_GET(self):
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        con = sqlite3.connect(f"file:{arxiv_tools.INDEX}?mode=ro&immutable=1", uri=True)
        try:
            self._send(200, {"papers": con.execute("SELECT max(rowid) FROM papers").fetchone()[0]})
        finally:
            con.close()

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="the address to bind: the host's private address, or 127.0.0.1")
    ap.add_argument("--port", type=int, default=18731)
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # exit normally, so the search workers are shut down
    Server((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
