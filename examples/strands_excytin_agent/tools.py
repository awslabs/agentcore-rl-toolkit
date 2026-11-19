"""ExCyTIn's one tool, execute_sql, served by the environment service (env/serve.py) at EXCYTIN_ENV_URL.

The query runs on the task's incident database only, read-only, as upstream's execute[...] action
(secgym/excytin_env.py, microsoft/SecRL@a1b234f7, MIT). An unreachable service or a 503 (the database unavailable)
is recorded in `state`, so the app stops the episode and fails the rollout instead of showing the error to the model.
Tool output is cut at 100,000 characters.
"""

import json
import os
import urllib.error
import urllib.request

from strands import tool

ENV_URL = os.environ.get("EXCYTIN_ENV_URL", "http://127.0.0.1:18733")
MAX_TOOL_CHARS = 100_000

SQL = {
    "description": "Execute a SQL query on the MySQL database of security logs and return the result rows. For "
    "example, SHOW TABLES, DESCRIBE table_name, or a SELECT query.",
    "inputSchema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The SQL query to execute."}},
        "required": ["query"],
    },
}


def make_tools(state: dict, incident: int) -> list:
    @tool(name="execute_sql", **SQL)
    def execute_sql(query: str) -> str:
        request = urllib.request.Request(
            f"{ENV_URL}/execute_sql",
            data=json.dumps({"incident": incident, "query": query}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                out = json.loads(response.read())["output"]
        except urllib.error.HTTPError as e:
            state["infra_error"] = json.loads(e.read() or b"{}").get("error", str(e))
            return "The tool is unavailable."
        except OSError as e:
            state["infra_error"] = f"environment service unreachable: {e}"
            return "The tool is unavailable."
        return out[:MAX_TOOL_CHARS] + "\n[output truncated]" if len(out) > MAX_TOOL_CHARS else out

    return [execute_sql]
