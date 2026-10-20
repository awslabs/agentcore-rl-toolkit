"""Own database resources for the lifetime of a local process or Runtime session."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import ExitStack, asynccontextmanager

from .agent import DatabaseAgent
from .contracts import AgentName
from .gwas_catalog.agent import create_agent as create_gwas_agent
from .gwas_catalog.sql_backend import SnapshotSQLBackend
from .opentargets.agent import create_agent as create_opentargets_agent


@asynccontextmanager
async def open_agents(names: Sequence[AgentName]) -> AsyncIterator[dict[AgentName, DatabaseAgent]]:
    with ExitStack() as stack:
        agents: dict[AgentName, DatabaseAgent] = {}
        for name in dict.fromkeys(names):
            if name == "opentargets":
                agents[name] = await asyncio.to_thread(create_opentargets_agent)
            elif name == "gwas_catalog":
                backend = stack.enter_context(SnapshotSQLBackend())
                agents[name] = create_gwas_agent(backend)
            else:
                raise ValueError(f"Unknown database agent: {name}")
        yield agents
