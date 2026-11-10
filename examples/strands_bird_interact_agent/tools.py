"""BIRD-Interact's actions as tools, served by the environment service (env/serve.py) at BIRD_ENV_URL.

The names, inputs and costs are upstream's a-Interact actions (bird_interact_agent/experiments/utils/prompts.py,
BIRD-Interact@451fe2c, MIT); the service keeps the budget, the phase and the gold, and appends the budget note to
each observation. An unreachable service or a 503 (the database or the user simulator unavailable) is recorded in
`state`, so the app stops the episode and fails the rollout instead of showing the error to the model.
"""

import json
import os
import urllib.error
import urllib.request

from strands import tool

ENV_URL = os.environ.get("BIRD_ENV_URL", "http://127.0.0.1:18734")


def post(state: dict, path: str, body: dict) -> dict | None:
    request = urllib.request.Request(
        f"{ENV_URL}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        state["infra_error"] = json.loads(e.read() or b"{}").get("error", str(e))
    except OSError as e:
        state["infra_error"] = f"environment service unreachable: {e}"
    return None


def act(state: dict, action: str | None, **args) -> str:
    """One action's observation; records when the task is finished."""
    out = post(state, "/act", {"session_id": state["session_id"], "action": action, "args": args})
    if out is None:
        return "The tool is unavailable."
    state["done"] = out["done"]
    return out["observation"]


def make_tools(state: dict) -> list:
    @tool
    def execute(sql: str) -> str:
        """Interact with the PostgreSQL database: returns the fetched result. Cost: 1.

        Args:
            sql: PSQL command to execute. Could contain multiple commands separated by semicolon.
        """
        return act(state, "execute", sql=sql)

    @tool
    def get_schema() -> str:
        """Get the schema of the database: the DDL with demo data. Cost: 1."""
        return act(state, "get_schema")

    @tool
    def get_all_column_meanings() -> str:
        """Get the meaning of all columns in the database. Cost: 1."""
        return act(state, "get_all_column_meanings")

    @tool
    def get_column_meaning(table_name: str, column_name: str) -> str:
        """Get the meaning of a column. Cost: 0.5.

        Args:
            table_name: name of the table to get column meaning.
            column_name: name of the column to get meaning.
        """
        return act(state, "get_column_meaning", table_name=table_name, column_name=column_name)

    @tool
    def get_all_external_knowledge_names() -> str:
        """Get all external knowledge names. Cost: 0.5."""
        return act(state, "get_all_external_knowledge_names")

    @tool
    def get_knowledge_definition(knowledge_name: str) -> str:
        """Get external knowledge by name: its definition. Cost: 0.5.

        Args:
            knowledge_name: name of the external knowledge to get definition.
        """
        return act(state, "get_knowledge_definition", knowledge_name=knowledge_name)

    @tool
    def get_all_knowledge_definitions() -> str:
        """Get all external knowledge names with definitions. Cost: 1."""
        return act(state, "get_all_knowledge_definitions")

    @tool
    def ask(question: str) -> str:
        """Ask the user for clarification, one question at a time: returns the user's reply. Cost: 2.

        Args:
            question: question to ask user for clarification.
        """
        return act(state, "ask", question=question)

    @tool
    def submit(sql: str) -> str:
        """Submit the SQL to the user, who tests it and gives feedback. Cost: 3.

        Args:
            sql: SQL to submit to the user. Could contain multiple commands separated by semicolon.
        """
        return act(state, "submit", sql=sql)

    return [
        execute,
        get_schema,
        get_all_column_meanings,
        get_column_meaning,
        get_all_external_knowledge_names,
        get_knowledge_definition,
        get_all_knowledge_definitions,
        ask,
        submit,
    ]
