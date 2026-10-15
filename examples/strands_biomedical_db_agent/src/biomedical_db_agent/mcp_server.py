"""Two database agents exposed as MCP tools."""

import argparse
from contextlib import asynccontextmanager

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.types import ToolAnnotations

from .config import ModelConfig
from .contracts import DatabaseResponse
from .service import open_agents


@asynccontextmanager
async def lifespan(_server):
    configs = {name: ModelConfig.from_env(name) for name in ("opentargets", "gwas_catalog")}
    async with open_agents(list(configs)) as agents:
        yield agents, configs


server = MCPServer("biomedical-agents", lifespan=lifespan)
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True)


@server.tool(annotations=READ_ONLY)
async def ask_opentargets_agent(query: str, ctx: Context) -> DatabaseResponse:
    """Delegate a self-contained Open Targets evidence task.

    Include phenotypes, variants or genes, all candidates, evidence requirements,
    and output constraints. Returns an answer with supporting evidence and uncertainty.
    """
    agents, configs = ctx.request_context.lifespan_context
    return await agents["opentargets"].answer(query, configs["opentargets"])


@server.tool(annotations=READ_ONLY)
async def ask_gwas_catalog_agent(query: str, ctx: Context) -> DatabaseResponse:
    """Delegate a self-contained GWAS Catalog causal-gene task.

    Include the phenotype, candidate genes, and available variant/study context.
    Returns an answer with supporting evidence and uncertainty.
    """
    agents, configs = ctx.request_context.lifespan_context
    return await agents["gwas_catalog"].answer(query, configs["gwas_catalog"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    args = parser.parse_args()
    if args.transport == "streamable-http":
        server.run(transport="streamable-http", host="0.0.0.0", port=8000, stateless_http=True)
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
