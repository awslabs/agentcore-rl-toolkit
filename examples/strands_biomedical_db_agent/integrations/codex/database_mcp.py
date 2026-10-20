"""Raw database tools available alongside the two subagents."""

from contextlib import asynccontextmanager

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.types import ToolAnnotations

from biomedical_db_agent.gwas_catalog.sql_backend import SnapshotSQLBackend
from biomedical_db_agent.opentargets.graphql_transport import query_open_targets_graphql as graphql_query


@asynccontextmanager
async def lifespan(_server):
    with SnapshotSQLBackend() as backend:
        yield backend


server = MCPServer("biomedical-databases", lifespan=lifespan)
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True)


@server.tool(annotations=READ_ONLY, structured_output=False)
def query_open_targets_graphql(query: str, variables: dict | None = None) -> dict:
    """Execute read-only Open Targets GraphQL. Read schemas/opentargets.graphql first.

    Returns data/errors unchanged. Use bounded fields and page sizes.
    """
    return graphql_query(query=query, variables=variables)


@server.tool(annotations=READ_ONLY, structured_output=False)
def query_gwas_catalog_sql(query: str, ctx: Context, parameters: dict | None = None) -> dict:
    """Execute read-only DuckDB SQL. Read schemas/gwas_catalog.sql first.

    Bind $parameters by name. Returns columns, rows, errors, and snapshot metadata.
    Limit results to 200 rows and 64 KB; narrow queries when a limit is exceeded.
    """
    return ctx.request_context.lifespan_context.execute(query, parameters)


if __name__ == "__main__":
    server.run(transport="stdio")
