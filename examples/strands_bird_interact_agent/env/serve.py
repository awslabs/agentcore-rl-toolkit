"""The BIRD-Interact environment service: one a-Interact episode per session, for the agent app's tools.

    POST /start {"session_id", "instance_id"}           -> {"observation"}: the question and the budget
    POST /act   {"session_id", "action", "args"}        -> {"observation", "done"}; action null: a turn without one
    POST /end   {"session_id"}                          -> {"phase1", "phase2", "reward"}; the session is closed
    GET  /health                                        -> {"tasks"}

The gold SQL and test cases stay here: the service merges upstream's public tasks (Hugging Face
birdsql/bird-interact-lite, pinned revision) with the gold file you obtained from the authors (--gold), in memory,
and never returns them. Each session has its own database copy, dropped at /end or after an idle hour. An
unavailable database or user simulator returns 503 with {"error"}; an unknown session, task or action 400. Run it
on a host in the AgentCore Runtime's VPC, bound to that host's private address (README).

    python serve.py --host 10.0.1.23 --data /data/bird --gold /data/bird_interact_gt_kg_testcases_1008.jsonl
"""

import argparse
import json
import re
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import bird_env
import db
import psycopg2
from huggingface_hub import snapshot_download

REPO, REVISION = "birdsql/bird-interact-lite", "f7881a9c2b9630cc4fc13b0c39279740b0a2fd87"
PATIENCE = 6  # upstream's official runs (run_batch_experiments.sh)
IDLE_SECONDS = 3600
SESSION_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")


class Store:
    """The tasks, the databases' schema / column meanings / knowledge, and the open sessions."""

    def __init__(self, data: Path, gold: Path):
        self.data = data
        public = Path(snapshot_download(REPO, repo_type="dataset", revision=REVISION))
        golds = {g["instance_id"]: g for g in map(json.loads, filter(str.strip, gold.read_text().splitlines()))}
        self.tasks = {}
        for r in map(json.loads, (public / "bird_interact_data.jsonl").read_text().splitlines()):
            g = golds.get(r["instance_id"])
            if g is None:
                continue
            for key in ("sol_sql", "external_knowledge", "test_cases"):  # combine_public_with_gt.py
                r[key] = g.get(key, [])
                r.setdefault("follow_up", {})[key] = (g.get("follow_up") or {}).get(key, [])
            self.tasks[r["instance_id"]] = r
        self.info = {}
        for name in db.DATABASES:
            d = public / name
            kb = [json.loads(line) for line in (d / f"{name}_kb.jsonl").read_text().splitlines() if line.strip()]
            self.info[name] = {
                "schema": (d / f"{name}_schema.txt").read_text(),
                "column_meanings": {
                    k.lower(): v for k, v in json.loads((d / f"{name}_column_meaning_base.json").read_text()).items()
                },
                "kb": {k["knowledge"]: k for k in kb},
            }
        self.prompts = json.loads((Path(__file__).parent / "user_simulator_prompts.json").read_text())
        self.sessions, self.lock = {}, threading.Lock()

    def db_info(self, name: str) -> dict:
        return self.info[name]

    def fresh_copy(self, ep: bird_env.Episode):
        """A connection to the episode's database copy, recreated from its template (upstream resets the database
        before every execute and submit)."""
        try:
            admin = db.connect(self.data)
            admin.autocommit = True
            with admin.cursor() as cur:
                cur.execute(f'DROP DATABASE IF EXISTS "{ep.copy}" WITH (FORCE)')
                cur.execute(f'CREATE DATABASE "{ep.copy}" TEMPLATE "{ep.db}_template"')
            admin.close()
            return db.connect(self.data, ep.copy)
        except psycopg2.Error as e:
            raise bird_env.InfraError(f"database reset: {e}") from None

    def drop(self, ep: bird_env.Episode) -> None:
        try:
            admin = db.connect(self.data)
            admin.autocommit = True
            with admin.cursor() as cur:
                cur.execute(f'DROP DATABASE IF EXISTS "{ep.copy}" WITH (FORCE)')
            admin.close()
        except psycopg2.Error:
            pass

    def reap(self) -> None:
        """Close sessions idle for an hour (an agent that never called /end)."""
        while True:
            time.sleep(300)
            with self.lock:
                stale = [s for s, ep in self.sessions.items() if time.time() - ep.used > IDLE_SECONDS]
                eps = [self.sessions.pop(s) for s in stale]
            for ep in eps:
                self.drop(ep)


STORE: Store


class Server(ThreadingHTTPServer):
    request_queue_size = 256  # many concurrent sessions; the default backlog is 5
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            sid = str(body["session_id"])
            if not SESSION_ID.match(sid):
                raise ValueError("invalid session_id")
            if self.path == "/start":
                return self._send(200, self.start(sid, str(body["instance_id"])))
            ep = STORE.sessions.get(sid)
            if ep is None:
                raise ValueError(f"unknown session {sid}")
            if self.path == "/act":
                with ep.lock:
                    ep.used = time.time()
                    observation = ep.act(body["action"], body.get("args") or {})
                return self._send(200, {"observation": observation, "done": ep.done})
            if self.path == "/end":
                with STORE.lock:
                    STORE.sessions.pop(sid, None)
                STORE.drop(ep)
                return self._send(
                    200, {"phase1": ep.phase_passed[1], "phase2": ep.phase_passed[2], "reward": ep.reward}
                )
            return self._send(404, {"error": f"unknown endpoint {self.path}"})
        except bird_env.InfraError as e:
            return self._send(503, {"error": str(e)})
        except (KeyError, TypeError, ValueError) as e:
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})

    def start(self, sid: str, instance_id: str) -> dict:
        record = STORE.tasks.get(instance_id)
        if record is None:
            raise ValueError(f"unknown task {instance_id}")
        ep = bird_env.Episode(record, STORE, PATIENCE, STORE.prompts, copy=f"ep_{sid.replace('-', '')[:32]}")
        with STORE.lock:
            if sid in STORE.sessions:
                raise ValueError(f"session {sid} exists")
            STORE.sessions[sid] = ep
        return {"observation": ep.opening()}

    def do_GET(self):
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        self._send(200, {"tasks": len(STORE.tasks), "sessions": len(STORE.sessions)})

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
    global STORE
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="the address to bind: the host's private address, or 127.0.0.1")
    ap.add_argument("--port", type=int, default=18734)
    ap.add_argument("--data", type=Path, required=True, help="the DATA_DIR of db.py")
    ap.add_argument("--gold", type=Path, required=True, help="the gold file from the authors (kept private)")
    a = ap.parse_args()
    STORE = Store(a.data, a.gold)
    threading.Thread(target=STORE.reap, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    Server((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
