"""One Codex conversation per dedicated container."""

import asyncio
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import tomli_w
from openai_codex import ApprovalMode, AsyncCodex, CodexConfig

from biomedical_db_agent.gwas_catalog.sql_backend import GENERATED
from biomedical_db_agent.opentargets.schema import load_schema

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    model: str
    region: str
    effort: str
    timeout_seconds: float

    @classmethod
    def from_env(cls):
        timeout = float(os.getenv("MAIN_AGENT_TIMEOUT_SECONDS", "1800"))
        if timeout <= 0:
            raise ValueError("MAIN_AGENT_TIMEOUT_SECONDS must be positive")
        return cls(
            model=os.environ["MAIN_AGENT_MODEL"],
            region=os.getenv("AWS_REGION", "us-west-2"),
            effort=os.getenv("MAIN_AGENT_EFFORT", "high"),
            timeout_seconds=timeout,
        )


def configure(settings: Settings) -> CodexConfig:
    if os.getenv("MAIN_AGENT_EXECUTION_MODE") != "container":
        raise RuntimeError("Run the Codex integration in its dedicated container")
    config = tomllib.loads((ROOT / "codex.toml").read_text())
    config["model"] = settings.model
    config["model_reasoning_effort"] = settings.effort
    config["model_providers"] = {"amazon-bedrock-runtime": {"aws": {"region": settings.region}}}
    temp_env = {key: os.environ[key] for key in ("TMPDIR", "TMP", "TEMP") if key in os.environ}
    config["shell_environment_policy"]["set"] = temp_env
    forwarded = sorted(
        set(temp_env)
        | {"AWS_REGION"}
        | {
            key
            for key in os.environ
            if key.startswith(("AWS_", "OPEN_TARGETS_", "OPENTARGETS_MODEL_", "GWAS_", "DATABASE_AGENT_"))
        }
    )
    config["mcp_servers"]["database"].update(
        command=sys.executable,
        args=[str(ROOT / "database_mcp.py")],
        env_vars=forwarded,
    )
    subagents = config["mcp_servers"]["subagents"]
    subagents.update(
        command=sys.executable,
        args=["-m", "biomedical_db_agent.mcp_server"],
        env_vars=forwarded,
    )
    if arn := os.getenv("BIOMEDICAL_RUNTIME_ARN"):
        region = arn.split(":")[3]
        endpoint = (
            f"https://bedrock-agentcore.{region}.amazonaws.com"
            f"/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"
        )
        subagents.update(
            command="mcp-proxy-for-aws-cli",
            args=[
                endpoint,
                "--service",
                "bedrock-agentcore",
                "--region",
                region,
                "--connect-timeout",
                "10",
                "--timeout",
                "660",
                "--read-timeout",
                "660",
                "--tool-timeout",
                "660",
            ],
        )
    directory = Path(os.environ["CODEX_HOME"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.toml").write_text(tomli_w.dumps(config))
    return CodexConfig(cwd=str(Path.cwd()), env={"AWS_REGION": settings.region, "SHELL": "/bin/bash"})


def prepare_schemas():
    directory = Path.cwd() / "schemas"
    directory.mkdir(exist_ok=True)
    (directory / "opentargets.graphql").write_text(load_schema())
    (directory / "gwas_catalog.sql").write_bytes((GENERATED / "gwas_catalog_schema.sql").read_bytes())


async def start_thread(codex: AsyncCodex):
    return await codex.thread_start(
        approval_mode=ApprovalMode.deny_all,
        base_instructions=(ROOT / "prompt.md").read_text(),
        cwd=str(Path.cwd()),
        ephemeral=True,
    )


async def run_turn(thread, query: str, timeout_seconds: float) -> dict:
    async with asyncio.timeout(timeout_seconds):
        result = await thread.run(query)
    if result.status.value != "completed" or result.error is not None:
        raise RuntimeError(f"Codex turn failed: {result.error or result.status.value}")
    if result.final_response is None:
        raise RuntimeError("Codex completed without a final response")
    return {"response": result.final_response, "thread_id": thread.id}
