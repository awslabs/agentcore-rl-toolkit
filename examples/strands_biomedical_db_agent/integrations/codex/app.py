"""HTTP conversations backed by a native Codex thread and four MCP tools."""

import asyncio
import logging
from contextlib import asynccontextmanager

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from codex_agent import Settings, configure, prepare_schemas, run_turn, start_thread
from openai_codex import AsyncCodex
from starlette.exceptions import HTTPException

from biomedical_db_agent.contracts import DatabaseRequest

logger = logging.getLogger(__name__)
thread = None
codex = None
settings = None
turn_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_app):
    global thread, codex, settings
    settings = Settings.from_env()
    config = configure(settings)
    await asyncio.to_thread(prepare_schemas)
    async with AsyncCodex(config=config) as codex:
        async with asyncio.timeout(settings.timeout_seconds):
            thread = await start_thread(codex)
        try:
            yield
        finally:
            thread = None


app = BedrockAgentCoreApp(lifespan=lifespan)


@app.entrypoint
async def invoke(payload: dict) -> dict:
    global thread
    request = DatabaseRequest.model_validate(payload)
    async with turn_lock:
        if thread is None:
            raise HTTPException(503, "This conversation has ended; start a new session")
        try:
            return await run_turn(thread, request.query, settings.timeout_seconds)
        except BaseException:
            # Keep an interrupted/failed turn from silently continuing a damaged conversation.
            thread = None
            await codex.close()
            logger.exception("Codex turn failed")
            raise


if __name__ == "__main__":
    app.run()
