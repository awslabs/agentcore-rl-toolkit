"""The BIRD-Interact Lite databases: upstream's PostgreSQL 14 image (env/docker-compose.yml, pinned by digest), which
loads the 18 databases on first start as templates <db>_template, published on 127.0.0.1 only. Each episode works on
its own copy of its database (bird_env.py). The superuser password is generated into DATA_DIR/secrets/ (mode 0600)
instead of the image's default.

    python db.py start DATA_DIR    create the container (first run: loads the databases, a few minutes) or start it
    python db.py check DATA_DIR    the 18 templates and their table counts
    python db.py stop              remove the container (the volume, and so the loaded data, is kept)
"""

import secrets
import subprocess
import sys
import time
from pathlib import Path

import psycopg2

IMAGE = "shawnxxh/bird-interact-postgresql@sha256:1ae45d7aa5d64dd8eb82e4058f56b4b9625d5035b9b9dc0d2afa0295d9d3053c"
CONTAINER, VOLUME = "bird-interact-postgresql", "bird-interact-postgresql-data"
PORT = 35432
USER = "root"  # upstream's superuser; the agent's SQL runs as it, as upstream
DATABASES = (
    "alien", "archeology", "credit", "cross_db", "crypto", "cybermarket", "disaster", "fake", "gaming",
    "insider", "mental", "museum", "news", "polar", "robot", "solar", "vaccine", "virtual",
)  # fmt: skip


def secret(data: Path) -> str:
    p = data / "secrets" / "postgres"
    if not p.exists():
        p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        p.touch(mode=0o600)
        p.write_text(secrets.token_urlsafe(24))
    return p.read_text().strip()


def connect(data: Path, dbname: str = "postgres"):
    return psycopg2.connect(
        host="127.0.0.1", port=PORT, user=USER, password=secret(data), dbname=dbname, connect_timeout=10
    )


def docker(*args, check=True) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check).stdout.strip()


def templates(data: Path) -> set[str]:
    con = connect(data)
    try:
        with con.cursor() as cur:
            cur.execute("SELECT datname FROM pg_database WHERE datistemplate")
            return {r[0] for r in cur.fetchall()}
    finally:
        con.close()


def start(data: Path) -> None:
    state = docker("ps", "-a", "--filter", f"name=^{CONTAINER}$", "--format", "{{.State}}", check=False)
    if state and state != "running":
        docker("start", CONTAINER)
    elif not state:
        secret(data)
        docker(
            "run", "-d", "--name", CONTAINER, "--restart", "unless-stopped",
            "-p", f"127.0.0.1:{PORT}:5432",  # loopback only
            "-e", f"POSTGRES_USER={USER}",
            "-e", "POSTGRES_PASSWORD=",  # the image sets upstream's default password; an empty one defers to the file
            "-e", "POSTGRES_PASSWORD_FILE=/run/secrets/postgres",
            "-e", "TZ=Asia/Hong_Kong",  # as upstream
            "-v", f"{(data / 'secrets/postgres').resolve()}:/run/secrets/postgres:ro",
            "-v", f"{VOLUME}:/var/lib/postgresql/data",
            IMAGE, "-c", "max_connections=300", "-c", "shared_buffers=256MB",
        )  # fmt: skip
    need = {f"{db}_template" for db in DATABASES}
    for _ in range(900):  # the templates are marked last, so all of them means the load finished
        try:
            if need <= templates(data):
                print(f"{CONTAINER} ready on 127.0.0.1:{PORT}")
                return
        except psycopg2.OperationalError:
            pass
        if docker("inspect", "-f", "{{.State.Status}}", CONTAINER, check=False) == "exited":
            sys.exit(f"{CONTAINER} exited:\n{docker('logs', '--tail', '40', CONTAINER, check=False)}")
        time.sleep(2)
    sys.exit(f"{CONTAINER} not ready after 30 minutes")


def check(data: Path) -> None:
    missing = {f"{db}_template" for db in DATABASES} - templates(data)
    for db in DATABASES:
        con = connect(data, f"{db}_template")
        with con.cursor() as cur:
            cur.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'")
            print(f"{db}: {cur.fetchone()[0]} tables")
        con.close()
    if missing:
        sys.exit(f"missing templates: {sorted(missing)}")


if __name__ == "__main__":
    cmd, data = sys.argv[1] if len(sys.argv) > 1 else "", Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if cmd == "stop":
        docker("rm", "-f", CONTAINER, check=False)
    elif cmd in ("start", "check") and data:
        {"start": start, "check": check}[cmd](data)
    else:
        sys.exit(__doc__)
