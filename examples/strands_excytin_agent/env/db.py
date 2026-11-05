"""The ExCyTIn incident databases: one MySQL 9.0 container (as upstream) holding the 8 incidents as databases
incident_<id>, published on 127.0.0.1 only. The service connects to incident <id> as `reader_<id>`, which may only
SELECT (and SHOW VIEW) on that incident's database, so no episode can read another incident (the held-out ones
included). The data is upstream's release (Hugging Face anandmudgerikar/excytin-bench, data.zip, pinned revision):
its sql_files, with the database renamed per incident and the admin user and the loads of absent macOS ._* files
removed, load its csv_files, mounted read-only. The passwords are generated into DATA_DIR/secrets/ (mode 0600).

    python db.py start DATA_DIR    download the data and create the container (first run: a few minutes), or start it
    python db.py check DATA_DIR    table and row counts per incident, as its reader, and that it can only read those
    python db.py stop              remove the container (the volume, and so the loaded data, is kept)
"""

import re
import secrets
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pymysql
from huggingface_hub import hf_hub_download

REPO, REVISION = "anandmudgerikar/excytin-bench", "8bc9ce1f97f8a6880c815dfba76d88651ddb761a"
IMAGE, CONTAINER, VOLUME = "mysql:9.0", "excytin-mysql", "excytin-mysql-data"
PORT = 33306
INCIDENTS = (5, 34, 38, 39, 55, 134, 166, 322)
ADMIN = (  # the release's admin user, dropped
    "CREATE USER 'admin'@'%' IDENTIFIED BY 'admin';\n\n"
    "GRANT ALL PRIVILEGES ON *.* TO 'admin'@'%';\n\n"
    "FLUSH PRIVILEGES;\n"
)


def secret(data: Path, name: str) -> str:
    p = data / "secrets" / name
    if not p.exists():
        p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        p.touch(mode=0o600)
        p.write_text(secrets.token_urlsafe(24))
    return p.read_text().strip()


def docker(*args, check=True) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check).stdout.strip()


def prepare(data: Path) -> None:
    """The release, unzipped into DATA_DIR/data/, and the init scripts in DATA_DIR/init/."""
    if not (data / "data/sql_files").is_dir():
        with zipfile.ZipFile(hf_hub_download(REPO, "data.zip", repo_type="dataset", revision=REVISION)) as z:
            z.extractall(data)
    init = data / "init"
    init.mkdir(exist_ok=True)
    for inc in INCIDENTS:
        sql = (data / f"data/sql_files/incident_{inc}.sql").read_text()
        assert sql.startswith(ADMIN), f"incident_{inc}.sql: unexpected header"
        sql = sql[len(ADMIN) :].replace("env_monitor_db", f"incident_{inc}")
        # the release's scripts also load macOS metadata (._*.csv) in split-table folders, absent from the release
        sql = re.sub(r"LOAD DATA INFILE '[^']*/\._[^']*'.*?IGNORE 1 ROWS;\n", "", sql, flags=re.S)
        sql = sql.replace("LOAD DATA INFILE '/var/lib/mysql-files/", f"LOAD DATA INFILE '/data/csv/incident_{inc}/")
        (init / f"10_incident_{inc}.sql").write_text(sql)
    readers = "".join(
        f"CREATE USER 'reader_{i}'@'%' IDENTIFIED BY '{secret(data, 'reader')}';\n"
        f"GRANT SELECT, SHOW VIEW ON `incident_{i}`.* TO 'reader_{i}'@'%';\n"
        for i in INCIDENTS
    )
    (init / "90_readers.sql").write_text(f"{readers}FLUSH PRIVILEGES;\n")
    for f in init.glob("*.sql"):
        f.chmod(0o644)  # read by the container's mysql user


def connect(data: Path, incident: int):
    return pymysql.connect(
        host="127.0.0.1",
        port=PORT,
        user=f"reader_{incident}",
        password=secret(data, "reader"),
        database=f"incident_{incident}",
        connect_timeout=10,
        autocommit=True,
    )


def start(data: Path) -> None:
    state = docker("ps", "-a", "--filter", f"name=^{CONTAINER}$", "--format", "{{.State}}", check=False)
    if state and state != "running":
        docker("start", CONTAINER)
    elif not state:
        prepare(data)
        secret(data, "root")
        docker(
            "run", "-d", "--name", CONTAINER, "--restart", "unless-stopped",
            "-p", f"127.0.0.1:{PORT}:3306",  # loopback only
            "-e", "MYSQL_ROOT_PASSWORD_FILE=/run/secrets/root",
            "-v", f"{(data / 'secrets/root').resolve()}:/run/secrets/root:ro",
            "-v", f"{(data / 'data/csv_files').resolve()}:/data/csv:ro",
            "-v", f"{(data / 'init').resolve()}:/docker-entrypoint-initdb.d:ro",
            "-v", f"{VOLUME}:/var/lib/mysql",
            IMAGE, "--secure-file-priv=",  # LOAD DATA from the read-only mount; the readers have no FILE privilege
        )  # fmt: skip
    for _ in range(900):  # the readers are created last, so a reader login means the load finished
        try:
            connect(data, INCIDENTS[-1]).close()
            print(f"{CONTAINER} ready on 127.0.0.1:{PORT}")
            return
        except pymysql.err.MySQLError:
            pass
        if docker("inspect", "-f", "{{.State.Status}}", CONTAINER, check=False) == "exited":
            sys.exit(f"{CONTAINER} exited:\n{docker('logs', '--tail', '40', CONTAINER, check=False)}")
        time.sleep(2)
    sys.exit(f"{CONTAINER} not ready after 30 minutes")


def check(data: Path) -> None:
    for inc in INCIDENTS:
        con = connect(data, inc)
        with con.cursor() as cur:
            cur.execute("SHOW TABLES")
            tables = [r[0] for r in cur.fetchall()]
            rows = 0
            for t in tables:
                cur.execute(f"SELECT COUNT(*) FROM `{t}`")
                rows += cur.fetchone()[0]
            denied = []
            other = next(i for i in INCIDENTS if i != inc)
            for probe in ("CREATE TABLE probe_write (a INT)", f"SELECT COUNT(*) FROM incident_{other}.SecurityAlert"):
                try:
                    cur.execute(probe)
                except pymysql.err.MySQLError:
                    denied.append(probe)
        con.close()
        print(f"incident_{inc}: {len(tables)} tables, {rows} rows, only its own readable: {len(denied) == 2}")
        if len(denied) != 2:
            sys.exit("a reader must only read its own incident")


if __name__ == "__main__":
    cmd, data = sys.argv[1] if len(sys.argv) > 1 else "", Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if cmd == "stop":
        docker("rm", "-f", CONTAINER, check=False)
    elif cmd in ("start", "check") and data:
        {"start": start, "check": check}[cmd](data)
    else:
        sys.exit(__doc__)
