"""Biomni research agent: one task episode per invocation, tools on the tool service, reward from harness metric.

Payload:
- task_name: one of TASK_FAMILIES
- prompt: the question
- answer: the label (optional; without it the reward is 0)
- _config: model config with base_url, model_id and optional api_key

The episode's sampling, system prompt, tools and budgets are the harness's. Each task family's tools are served by
tool_service.py at the URL in TOOL_SERVICE_<FAMILY> (e.g. TOOL_SERVICE_DBQA).
"""

import hashlib
import logging
import os
import uuid
from types import SimpleNamespace

import requests

from agentcore_rl_toolkit import AgentCoreRuntimeApp
from harness.dbqa import harness as dbqa
from harness.gwas import harness as gwas
from harness.screen import harness as screen
from harness.seqqa import harness as seqqa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HARNESS = {"dbqa": dbqa, "screen": screen, "seqqa": seqqa, "gwas": gwas}
TASK_FAMILIES = {task: family for family, module in HARNESS.items() for task in module.TASKS}
TOOL_TIMEOUT = 900

app = AgentCoreRuntimeApp()
http = requests.Session()


def tool_client(family: str, session: str):
    """The family's tool entry point, backed by its tool service; a service fault is a backend error."""
    url = os.environ[f"TOOL_SERVICE_{family.upper()}"].rstrip("/")

    def invoke(name: str, arguments: dict):
        try:
            response = http.post(
                url + "/invoke",
                json={"name": name, "arguments": arguments, "session": session},
                timeout=TOOL_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()["result"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            error = f"tool service: {type(exc).__name__}: {exc}"
            if family == "gwas":
                return "backend_error", f"Error: {error}", {"error": error}
            return {"status": "backend_error", "error": error}
        return tuple(result) if family == "gwas" else result

    return url, invoke


@app.entrypoint
def invoke_agent(payload: dict, context):
    config = payload["_config"]
    task = payload["task_name"]
    family = TASK_FAMILIES[task]
    module = HARNESS[family]
    prompt = payload["prompt"]
    row = SimpleNamespace(
        task_name=task,
        task_instance_id=payload.get("task_instance_id") or hashlib.sha256(prompt.encode()).hexdigest()[:12],
        prompt=prompt,
        answer=payload.get("answer"),
    )
    # During training the rollout gateway keys trajectory capture off the api-key slot.
    client = requests.Session()
    client.headers["Authorization"] = f"Bearer {config.get('api_key') or 'EMPTY'}"
    session = uuid.uuid4().hex
    url, invoke = tool_client(family, session)
    try:
        record = module.run_one(client, config["base_url"], config["model_id"], row, task, module.SYSTEM[task], invoke)
    finally:
        if family == "gwas":
            try:
                http.post(url + "/close", json={"session": session}, timeout=60)
            except requests.RequestException:
                logger.warning("could not close the GWAS SQL session %s", session)
    logger.info("%s %s: %s, reward %s", task, row.task_instance_id, record["terminal_reason"], record["score"])
    return {"reward": record["score"], "answer": record["parsed_answer"], "response": record["final_response"]}


if __name__ == "__main__":
    app.run()
