"""GWAS Catalog training entrypoint on AgentCore Runtime."""

import asyncio

from agentcore_rl_toolkit import AgentCoreRuntimeApp

from apps.models import InvocationRequest
from apps.reward import ExactAnswerReward
from biomedical_db_agent.gwas_catalog.agent import create_agent
from biomedical_db_agent.gwas_catalog.sql_backend import SnapshotSQLBackend

app = AgentCoreRuntimeApp()
reward_fn = ExactAnswerReward()


@app.entrypoint
async def invoke(payload: dict) -> dict:
    request = InvocationRequest.model_validate(payload)
    backend = await asyncio.to_thread(SnapshotSQLBackend)
    with backend:
        agent = create_agent(backend)
        response = await agent.answer(request.query, request.config)
    return {"response": response.model_dump(), "reward": reward_fn(response, request.expected_answers)}


if __name__ == "__main__":
    app.run()
