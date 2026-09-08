"""This recipe's own config: ``config.toml``, and what has to be looked up from it.

The counterpart to the cluster-side config a developer keeps in their ``.env``.
The split is by *whose* fact it is: which image this agent runs, on which pool,
against which table and bucket are properties of the recipe, so they travel with
it and every script in this directory reads them from here. ``.env`` keeps the
facts about the developer's own cluster.

The AgentCore ARNs are deliberately *not* config. They are looked up by name in
:func:`resolve_env`, so no two scripts can disagree about which runtime they mean --
a stale pasted ARN is exactly the drift this avoids.
"""

import tomllib
from pathlib import Path
from typing import NamedTuple

from agentcore_rl_toolkit.aws_tools.agentcore_control import (
    find_agent_runtime,
    find_capacity_provider,
)

RECIPE_DIR = Path(__file__).resolve().parent
# The toolkit repo's root: this recipe is ``examples/<recipe>/``, so two levels up.
# Named here because ``deploy.py`` hands it to docker as the ``rlpkg`` build context,
# and the image copies the wire protocol package out of it -- so what this has to point
# at is the directory holding ``pyproject.toml`` and ``src/agentcore_rl_toolkit``.
TOOLKIT_ROOT = RECIPE_DIR.parents[1]
CONFIG_PATH = RECIPE_DIR / "config.toml"
EXAMPLE_PATH = RECIPE_DIR / "config.example.toml"

# Everything this recipe's scripts write that is not checked in: task parquets,
# eval reports, rollout dumps. Under this directory rather than the repo's top-level
# ``local`` because they are artifacts of this recipe, and absolute so it is the same
# directory whichever one a script was launched from.
LOCAL_DIR = RECIPE_DIR / "local"


def dataset_parquet(dataset_name: str) -> Path:
    """Where a named dataset's parquet lives.

    A convention rather than a path each script states, because the script that builds
    the parquet and the eval that reads it as a slice have to agree on it. Naming it by
    the dataset key means a build of the whole dataset is found by name; a filtered cut
    is a different file and still wants an explicit ``--output``, since its name is the
    filters, which this cannot know.
    """
    return LOCAL_DIR / f"{dataset_name}.parquet"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """The whole of ``config.toml``.

    Raises on a missing file rather than defaulting: the example config is checked
    in but the real one is not, so "no config.toml" is the expected first-run state
    and deserves a message that says what to copy.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"no config at {path}; copy {EXAMPLE_PATH} to {CONFIG_PATH.name} and fill in your account's values"
        )
    return tomllib.loads(path.read_text())


class EcrNamespace(NamedTuple):
    """An ECR registry host and the path under it, taken apart.

    ``agentcore.docker_repo`` is the one setting written this way --
    ``<account>.dkr.ecr.<region>.amazonaws.com/<repository>`` -- because it is what
    docker is handed. It is parsed because the deploy needs the parts and not the
    string: the account and the repository are what the execution role's ECR grant
    is scoped to, and the account is also the registry the task image cache is
    built in (:func:`task_image_namespace`).
    """

    account_id: str
    region: str
    path: str


def _ecr_namespace(setting: str, value: str) -> EcrNamespace:
    """Parse one ``<account>.dkr.ecr.<region>.amazonaws.com/<path>`` setting.

    Strictly, rather than defaulting anything: every looser reading fails much
    later and further away. A missing path would scope an ECR grant to the whole
    registry, a non-ECR host has no account to scope one to at all, and a tag left
    on the end would be pushed as part of the repository name.
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

    ``agentcore.docker_repo`` without the tag a deploy appends -- so the path is one
    repository, and every tag of the agent image lives in it. That is the granularity
    the execution role's ECR read grant wants: tags are not part of a repository ARN,
    so one grant covers the tag a runtime is on today and the one ``--image-tag`` rolls
    tomorrow, without naming either.
    """
    return _ecr_namespace("docker_repo", config["agentcore"]["docker_repo"])


def cache_prefix(config: dict) -> str:
    """The ECR prefix the *task* images are cached under, e.g. ``docker-hub``.

    ``docker_hub.cache_prefix`` is the whole of what the config says about the pull
    through cache, and it is a bare prefix rather than a registry path because the
    registry is not a choice: the rule is made in this account (there is nowhere else
    it could be) and in ``agentcore.region``, so that a session pulls its task image
    from the region it runs in. What is left is the rule's name, which is a choice --
    an account can hold several rules, and this says which one is ours.

    Both deploy-time uses read it from here so they cannot disagree: it is the prefix
    the rule is created on, and it scopes the execution role's ECR grants.
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
    """Where a run's records go, and the region they go to.

    The region is not a third setting: the session table and the dump bucket are the
    ones AgentCore's sessions write from, so they are read in ``agentcore.region`` like
    everything else here. That is what makes it safe to reach the table later with
    nothing but this config.
    """

    dynamodb_table: str
    rollout_output_s3: str
    region: str


def storage(config: dict) -> Storage:
    """The ``[storage]`` table, with the region the records live in filled in.

    Read by every script here, whether it writes a run's records (through
    :func:`resolve_env`) or reads them back, so that none of them can end up looking in
    a different place -- which, for a reader, shows up as a run that appears never to
    have happened.
    """
    section = config["storage"]
    return Storage(
        dynamodb_table=section["dynamodb_table"],
        rollout_output_s3=section["rollout_output_s3"],
        region=config["agentcore"]["region"],
    )


def task_image_namespace(config: dict) -> str:
    """The registry namespace an eval tells the agent to pull task images from.

    The cache assembled back into the string the agent joins onto a task's
    ``docker_image_uri``, from the three parts that decide it: this account, the region
    the sessions run in, and :func:`cache_prefix`. Built rather than stated so that the
    namespace an eval pulls from is necessarily the rule the same deploy created and
    granted -- the failure it replaces is a pasted namespace naming a cache in another
    region, which pulls fine (it exists) and slowly.

    The account is the one ``agentcore.docker_repo`` names, since a deploy already
    insists that registry is the deploying account's own, and it is the only account
    any of this recipe's images live in.
    """
    region = config["agentcore"]["region"]
    account_id = agent_repository(config).account_id
    return f"{account_id}.dkr.ecr.{region}.amazonaws.com/{cache_prefix(config)}"


async def resolve_env(config: dict) -> dict:
    """The ``env`` mapping an eval run is handed.

    That mapping is keyed by the same names the training side resolves out of its own
    config; this is the recipe's side of the same shape, with the two ARNs looked up
    from the names in ``[agentcore]`` rather than restated. A runtime or pool that does
    not exist yet is an error here, not a later ``None`` in a boto3 call: it means the
    deploy has not been run.
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
