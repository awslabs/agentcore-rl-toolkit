"""This recipe's own ``config.toml`` and the lookups over it.

Recipe-level facts (image, pool, table, bucket) live here; the developer's cluster facts
stay in ``.env``. AgentCore ARNs are deliberately not config -- :func:`resolve_env` looks
them up by name so no two scripts can disagree about which runtime they mean.
"""

import tomllib
from pathlib import Path
from typing import NamedTuple

from agentcore_rl_toolkit.aws_tools.agentcore_control import (
    find_agent_runtime,
    find_capacity_provider,
)

RECIPE_DIR = Path(__file__).resolve().parent
# The toolkit repo root, handed to docker as the ``rlpkg`` build context: the directory
# holding ``pyproject.toml`` and ``src/agentcore_rl_toolkit``.
TOOLKIT_ROOT = RECIPE_DIR.parents[1]
CONFIG_PATH = RECIPE_DIR / "config.toml"
EXAMPLE_PATH = RECIPE_DIR / "config.example.toml"

# Everything this recipe's scripts write that is not checked in: task parquets, eval
# reports, rollout dumps. Absolute, so it is the same directory whatever the cwd.
LOCAL_DIR = RECIPE_DIR / "local"


def dataset_parquet(dataset_name: str) -> Path:
    """Where a named dataset's parquet lives -- the convention the builder and the eval
    that reads it as a slice both go through. A filtered cut is a different file and
    still wants an explicit ``--output``.
    """
    return LOCAL_DIR / f"{dataset_name}.parquet"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """The whole of ``config.toml``. Raises on a missing file: the example is checked in
    but the real one is not, so "no config.toml" is the expected first-run state.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"no config at {path}; copy {EXAMPLE_PATH} to {CONFIG_PATH.name} and fill in your account's values"
        )
    return tomllib.loads(path.read_text())


class EcrNamespace(NamedTuple):
    """An ECR registry host and the path under it, taken apart: the deploy needs the
    account (to scope the execution role's ECR grants) and the region, not the string.
    """

    account_id: str
    region: str
    path: str


def _ecr_namespace(setting: str, value: str) -> EcrNamespace:
    """Parse one ``<account>.dkr.ecr.<region>.amazonaws.com/<path>`` setting, strictly:
    every looser reading fails much later and further away.
    """
    host, _, path = value.strip("/").partition("/")
    fields = host.split(".")
    if (
        not path
        or ":" in path
        or len(fields) != 6
        or fields[1:3] != ["dkr", "ecr"]
        or fields[4:] != ["amazonaws", "com"]
    ):
        raise ValueError(
            f"{setting} {value!r} is not an ECR namespace; expected "
            f"'<account>.dkr.ecr.<region>.amazonaws.com/<path>' with no tag"
        )
    return EcrNamespace(account_id=fields[0], region=fields[3], path=path)


def agent_repository(config: dict) -> EcrNamespace:
    """The repository the *agent* image is pushed to and pulled from.

    ``agentcore.docker_repo`` without the tag a deploy appends, which is the granularity
    the execution role's ECR read grant wants: one grant covers every tag.
    """
    return _ecr_namespace("docker_repo", config["agentcore"]["docker_repo"])


def cache_prefix(config: dict) -> str:
    """The ECR prefix the *task* images are cached under, e.g. ``docker-hub``.

    A bare rule name, not a registry path: the rule is necessarily in this account and in
    ``agentcore.region``, so a session pulls from the region it runs in.
    """
    prefix = config["docker_hub"]["cache_prefix"].strip("/")
    if not prefix or ":" in prefix or "amazonaws.com" in prefix:
        raise ValueError(
            f"cache_prefix {prefix!r} is not a pull through cache prefix; expected just "
            f"the rule's name, e.g. 'docker-hub' -- the registry and the region it lives "
            f"in are this account and agentcore.region, not something to state here"
        )
    return prefix


class Storage(NamedTuple):
    """Where a run's records go, and the region they go to."""

    dynamodb_table: str
    rollout_output_s3: str
    region: str


def storage(config: dict) -> Storage:
    """The ``[storage]`` table, with ``agentcore.region`` filled in as the region the
    records live in.
    """
    section = config["storage"]
    return Storage(
        dynamodb_table=section["dynamodb_table"],
        rollout_output_s3=section["rollout_output_s3"],
        region=config["agentcore"]["region"],
    )


def task_image_namespace(config: dict) -> str:
    """The registry namespace an eval tells the agent to pull task images from.

    Assembled from this account, ``agentcore.region`` and :func:`cache_prefix` rather than
    stated, so it is necessarily the cache rule the same deploy created and granted.
    """
    region = config["agentcore"]["region"]
    account_id = agent_repository(config).account_id
    return f"{account_id}.dkr.ecr.{region}.amazonaws.com/{cache_prefix(config)}"


async def resolve_env(config: dict) -> dict:
    """The ``env`` mapping an eval run is handed, keyed by the same names the training
    side resolves from its own config. A missing runtime or pool is an error here rather
    than a later ``None`` in a boto3 call: it means the deploy has not been run.
    """
    agentcore = config["agentcore"]
    records = storage(config)
    region_name = agentcore["region"]

    runtime_name = agentcore["runtime_name"]
    provider_name = agentcore["capacity_provider"]["name"]
    runtime = await find_agent_runtime(region_name, runtime_name)
    provider = await find_capacity_provider(region_name, provider_name)
    if runtime is None:
        raise RuntimeError(f"no agent runtime named {runtime_name!r} in {region_name}; run ./deploy.py")
    if provider is None:
        raise RuntimeError(f"no capacity provider named {provider_name!r} in {region_name}; run ./deploy.py")

    return {
        "aws_region": records.region,
        "agentcore_runtime_arn": runtime["agentRuntimeArn"],
        "agentcore_capacity_provider_arn": provider["capacityProviderArn"],
        "agent_dynamodb_table": records.dynamodb_table,
        "rollout_output_s3": records.rollout_output_s3,
    }
