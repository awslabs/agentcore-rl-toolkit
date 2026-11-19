"""HERB's tools: `retrieve_all` (retrieval.py) and the 8 structured tools of HERB's metadata_search_functions.py
(employee, role, customer, PR and URL lookups; upstream.py), with their names, descriptions and parameters.

Tool output is cut at 200,000 characters. A tool error caused by the call (e.g. an unknown ID) is shown to the model;
an unavailable Bedrock model (the embeddings, or the helper LLM some structured tools call) is recorded in `state`,
so the app stops the episode and fails the rollout instead of showing the error to the model.
"""

import botocore.exceptions
import upstream
from llama_index.core.tools import FunctionTool
from strands.tools.tools import PythonAgentTool

MAX_TOOL_CHARS = 200_000
INFRA_ERRORS = (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError, RuntimeError)


def make_tools(state: dict, index) -> list:
    def agent_tool(name: str, description: str, parameters: dict, fn) -> PythonAgentTool:
        def call(tool_use, **_):
            try:
                out = str(fn(**(tool_use.get("input") or {})))
            except INFRA_ERRORS as e:
                state["infra_error"] = f"{name}: {type(e).__name__}: {e}"
                out = "The tool is unavailable."
            except Exception as e:  # noqa: BLE001 - a bad call is the model's, as in HERB's agent
                out = f"Error: {type(e).__name__}: {e}"
            if len(out) > MAX_TOOL_CHARS:
                out = out[:MAX_TOOL_CHARS] + "\n[output truncated]"
            return {"toolUseId": tool_use["toolUseId"], "status": "success", "content": [{"text": out}]}

        spec = {"name": name, "description": description, "inputSchema": {"json": parameters}}
        return PythonAgentTool(name, spec, call)

    retrieve = {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}
    tools = [agent_tool("retrieve_all", upstream.retrieve_all_description(), retrieve, index.retrieve_all)]
    for t in vars(upstream.metadata_tools()).values():
        if isinstance(t, FunctionTool):
            tools.append(agent_tool(t.metadata.name, t.metadata.description, t.metadata.get_parameters_dict(), t.fn))
    return tools
