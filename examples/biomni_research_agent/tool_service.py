"""Tool service: runs one task family's tools next to the environment data and the NCBI keys.

The agent on AgentCore Runtime runs the episode loop and sends every tool call here.

    POST /invoke  {"name", "arguments", "session"}  -> {"result": <the tool's result>}
    POST /close   {"session"}                        (gwas: closes that episode's GWAS Catalog SQL backend)
    GET  /health                                      -> {"ok": true, "family": ...}

The service binds to loopback by default. For the Runtime, bind it to the node's private IP and allow its port
only from the Runtime's security group.

    python tool_service.py <dbqa|screen|seqqa|gwas> --port <port> [--host <private IP>]
"""

import argparse
import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FAMILIES = ("dbqa", "screen", "seqqa", "gwas")


def make_entry(family: str):
    """The family's tool entry point: (name, arguments, session) -> JSON-serializable result."""
    tools = importlib.import_module(f"harness.{family}.tools")
    if family != "gwas":
        from harness.common import online

        online.ncbi_keys()  # fail at startup, not on the first NCBI call
        return lambda name, arguments, session: tools.invoke(name, arguments), lambda session: None

    from harness.gwas.harness import sql_backend

    backends, lock = {}, threading.Lock()

    def entry(name, arguments, session):
        if name == "query_open_targets_graphql":
            return list(tools.ot_invoke(arguments))
        if name != "query_gwas_catalog_sql":
            return ["error", f"Error: unknown tool {name}", {"error": f"unknown tool {name}"}]
        with lock:
            if session not in backends:
                backends[session] = sql_backend()
            backend = backends[session]
        return list(tools.gwas_invoke(backend, arguments))

    def close(session):
        with lock:
            backend = backends.pop(session, None)
        if backend is not None:
            backend.close()

    sql_backend().close()  # verify the GWAS Catalog files at startup
    return entry, close


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    entry, close = make_entry(args.family)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self.send(200, {"ok": True, "family": args.family})
            self.send(404, {"error": "not found"})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/close":
                close(request.get("session"))
                return self.send(200, {"ok": True})
            if self.path != "/invoke":
                return self.send(404, {"error": "not found"})
            try:
                result = entry(request["name"], request["arguments"], request.get("session"))
            except Exception as exc:  # noqa: BLE001  the tools catch their own errors; this is a service fault
                return self.send(500, {"error": f"{type(exc).__name__}: {exc}"})
            self.send(200, {"result": result})

    print(json.dumps({"family": args.family, "listening": f"http://{args.host}:{args.port}"}), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":  # the GWAS SQL backend's spawn workers re-import this file: keep everything in main()
    main()
