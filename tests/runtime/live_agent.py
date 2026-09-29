"""Deploy this agent from the checkout to run test_live.py; no model or S3 needed."""

import asyncio
import uuid

from agentcore_rl_toolkit import AgentCoreRuntimeApp

app = AgentCoreRuntimeApp()
instance_id = str(uuid.uuid4())
call_count = 0


@app.entrypoint
async def invoke(payload, context):
    global call_count
    call_count += 1
    count = call_count
    await asyncio.sleep(min(float(payload.get("delay", 0)), 180))
    if payload.get("fail"):
        raise ValueError("intentional test failure")
    return {
        "value": payload.get("value"),
        "reward": 1.0,
        "session_id": context.session_id,
        "instance_id": instance_id,
        "call_count": count,
    }


if __name__ == "__main__":
    app.run()
