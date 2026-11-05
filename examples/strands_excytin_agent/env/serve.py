"""The ExCyTIn environment service: execute_sql over HTTP, for the agent app's tool.

POST /execute_sql {"incident", "query"} returns {"output"}: the query's result on that incident's database, as
upstream's execute[...] action (secgym/excytin_env.py): MAX_EXECUTION_TIME 30 s; when a result's string form is
longer than 100,000 characters and it has more than 15 rows, only the first 15 rows are shown, with the number
retrieved. A query error (a bad query, a denied statement, the time limit) is the output. An unavailable database
returns 503 with {"error"}; an unknown incident or invalid arguments 400. GET /health returns the incidents served.
The service connects as the incident's reader (db.py), which may only read that incident. Run it on a host in the
AgentCore Runtime's VPC, bound to that host's private address (README).

    python serve.py --host 10.0.1.23 --data /data/excytin
"""

import argparse
import json
import signal
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import db
import pymysql

MAX_ENTRY_RETURN = 15
MAX_STR_LEN = 100_000
LOST = frozenset({2003, 2006, 2013})  # cannot connect, server gone away, lost connection during query
DATA: Path


def longer_than(rows: list, limit: int) -> bool:
    """len(str(rows)) > limit, without building the string of a large result."""
    n = 2 + 2 * max(len(rows) - 1, 0)  # the brackets and the ", " separators
    for r in rows:
        n += len(repr(r))
        if n > limit:
            return True
    return False


def execute_sql(incident: int, query: str) -> str:
    """The tool's output; raises ConnectionError when the database is unavailable."""
    try:
        con = db.connect(DATA, incident)
    except pymysql.err.OperationalError as e:
        raise ConnectionError(f"database unavailable: {e}") from None
    try:
        with con.cursor() as cur:
            cur.execute("SET SESSION MAX_EXECUTION_TIME=30000")
            cur.execute(query)
            rows = list(cur.fetchall())
    except (pymysql.err.InterfaceError, pymysql.err.OperationalError) as e:
        if isinstance(e, pymysql.err.InterfaceError) or e.args[0] in LOST:
            raise ConnectionError(f"database connection lost: {e}") from None
        return f"Error: {type(e).__name__}: {e}"
    except pymysql.err.MySQLError as e:
        return f"Error: {type(e).__name__}: {e}"
    finally:
        con.close()
    if len(rows) > MAX_ENTRY_RETURN and longer_than(rows, MAX_STR_LEN):
        return f"Retrieved {len(rows)} entries. Displaying first {MAX_ENTRY_RETURN} entries.\n{rows[:MAX_ENTRY_RETURN]}"
    return str(rows)


class Server(ThreadingHTTPServer):
    request_queue_size = 256  # many concurrent sessions; the default backlog is 5
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/execute_sql":
            return self._send(404, {"error": f"unknown tool {self.path}"})
        try:
            args = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            incident, query = int(args["incident"]), str(args["query"])
            if incident not in db.INCIDENTS:
                raise ValueError(f"unknown incident {incident}")
        except (KeyError, TypeError, ValueError) as e:
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})
        try:
            output = execute_sql(incident, query)
        except ConnectionError as e:
            return self._send(503, {"error": str(e)})
        self._send(200, {"output": output})

    def do_GET(self):
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        self._send(200, {"incidents": db.INCIDENTS})

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
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="the address to bind: the host's private address, or 127.0.0.1")
    ap.add_argument("--port", type=int, default=18733)
    ap.add_argument("--data", type=Path, required=True, help="the DATA_DIR of db.py")
    a = ap.parse_args()
    DATA = a.data
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    Server((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
