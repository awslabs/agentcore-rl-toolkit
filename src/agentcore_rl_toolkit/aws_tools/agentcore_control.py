"""AgentCore *control plane*: the capacity provider and the runtime on it.

Two resources, provisioned once per image roll rather than once per rollout:
a **capacity provider**, which is the EC2 pool sessions land on, and an **agent
runtime**, which says what image those sessions run and which role they assume.

Every function here takes what it needs as arguments and reads no config of its
own, which is what lets unrelated callers share them: one passes values out of a
``config.toml``, another out of the environment.

The asymmetry in the ``ensure_*`` pair is the API's, not ours: a runtime can be
repointed at a new image, so :func:`ensure_agentcore_runtime` creates or updates as
needed, whereas a capacity provider's compute configuration is immutable
(``UpdateCapacityProvider`` accepts only a description), so
:func:`ensure_capacity_provider` can only report the one already there. Changing an
instance type means a new provider under a new name.

A pool also needs two IAM roles, built here out of AWS's own managed policies by
:func:`ensure_capacity_provider_roles`, because they say nothing about any
particular agent. The *runtime's* execution role -- what an agent's code runs as --
stays the caller's, since only the caller knows what its agent may do.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session
from agentcore_rl_toolkit.aws_tools.iam_control import (
    current_account_id,
    ensure_instance_profile,
    ensure_role,
)

logger = logging.getLogger(__name__)

# How long an idle *session* on a runtime survives, and its hard cap. Sessions are
# the unit of work and the rollout session tears one down when a rollout ends, so these
# only bound a session that was abandoned.
RUNTIME_LIFECYCLE = {
    "idleRuntimeSessionTimeout": 3600,
    "maxLifetime": 3600,
}

# The same for an *instance* in the pool: the provider reaps one that no session
# has claimed, so a burst of rollouts does not leave the pool warm for hours.
INSTANCE_LIFECYCLE = {
    "idleInstanceTimeout": 3600,
    "maxLifetime": 3600,
}


@asynccontextmanager
async def _control_client(region_name: str):
    async with (await get_aioboto3_session()).client("bedrock-agentcore-control", region_name=region_name) as acrc:  # type: ignore
        yield acrc


async def _wait_ready(describe, what: str) -> dict:
    """Poll ``describe`` until the resource is READY, raising if it failed.

    Both resources report a terminal ``*_FAILED`` status, and waiting on one
    forever is the difference between a deploy that says what went wrong and one
    that looks like a hang.
    """
    while True:
        await asyncio.sleep(3)
        state = await describe()
        status = state["status"]
        if status == "READY":
            return state
        if status.endswith("_FAILED"):
            reason = state.get("failureReason") or state.get("statusReason") or "no reason given"
            raise RuntimeError(f"{what} is {status}: {reason}")
        logger.info("waiting for %s: %s", what, status)


# --- capacity provider roles ------------------------------------------------

# AgentCore assumes the *operator* role (the console calls it the infrastructure
# role) to launch, tag, network and reap the EC2 instances behind a pool. Its policy
# is AWS's, not ours, and deliberately so: the actions are the ones the feature
# needs, they are already scoped by the `bedrock-agentcore:capacity-provider-id`
# request tag and the `ec2:ManagedResourceOperator` condition key rather than by
# anything we could say about our own resources, and AWS has already added actions to
# it once (August 2026, PassRole). A hand-written copy would be one that goes stale
# into an outage.
OPERATOR_ROLE_NAME = "AmazonBedrockAgentCoreCapacityProviderDefaultOperatorRole"
OPERATOR_POLICY_ARN = "arn:aws:iam::aws:policy/BedrockAgentCoreRuntimeInstancesOperatorRolePolicy"
OPERATOR_DESCRIPTION = "AgentCore capacity provider operator (infrastructure) role"

# The *instance* role is carried by each instance in the pool, and it grants exactly
# one action -- `bedrock-agentcore:PutSystemLogEvents`, so that AgentCore can collect
# the instance's system logs. It is not how an agent gets its permissions; that is
# the runtime's execution role, whose credentials are vended into the session
# separately. So there is nothing to add here, and anything added would be a grant to
# every process on an instance that sessions share.
#
# The name is not a free choice. The operator policy's `iam:PassRole` is scoped to
# roles named with the `AmazonBedrockAgentCoreCapacityProviderDefaultInstanceRole`
# prefix, so a role named anything else cannot be passed to EC2 by an operator role
# holding that policy -- instance launches simply fail. The instance profile takes the
# same name, which is what the console does.
INSTANCE_ROLE_NAME = "AmazonBedrockAgentCoreCapacityProviderDefaultInstanceRole"
INSTANCE_POLICY_ARN = "arn:aws:iam::aws:policy/BedrockAgentCoreRuntimeInstancesInstanceRolePolicy"
INSTANCE_DESCRIPTION = "AgentCore capacity provider instance role (system logs only)"


def _operator_trust_policy(account_id: str) -> dict:
    """Who may assume the operator role: AgentCore, on behalf of this account only.

    The ``aws:SourceAccount`` condition is the confused deputy guard the Instances
    security guidance asks for -- without it, this role's ARN is enough for AgentCore
    acting for *any* account to launch EC2 in ours. It is not conditioned on a source
    ARN as well: the first thing to assume this role is ``CreateCapacityProvider``,
    which happens before the capacity provider whose ARN would be matched exists.

    If a ``CreateCapacityProvider`` ever fails having been unable to assume the role,
    this condition is the first thing to suspect -- AWS documents neither this trust
    policy nor which condition keys its assume-role call carries.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AssumeByAgentCore",
                "Effect": "Allow",
                "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {"StringEquals": {"aws:SourceAccount": account_id}},
            },
        ],
    }


def _instance_trust_policy() -> dict:
    """Who may assume the instance role: EC2, which is the only thing that can.

    No account condition here, unlike :func:`_operator_trust_policy`: a role reaches
    an instance through an instance profile, and a profile in this account can only be
    attached to an instance in this account.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AssumeByEC2",
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            },
        ],
    }


async def ensure_capacity_provider_roles(region_name: str) -> tuple[str, str]:
    """Reconcile a pool's two roles; return the operator ARN and the instance profile ARN.

    The pair :func:`create_capacity_provider` asks for. Neither role is a choice a
    caller makes -- the names are fixed (see :data:`INSTANCE_ROLE_NAME` for why one of
    them has to be) and each carries only the AWS managed policy written for its job,
    so this is one call with no arguments beyond the region. What it buys over letting
    the console create them is that a deploy is the single place they come from.

    Both ARNs are part of a capacity provider's immutable compute configuration, so
    calling this cannot move an existing pool onto different roles: that takes a new
    provider name.
    """
    account_id = await current_account_id(region_name)

    operator_role_arn = await ensure_role(
        role_name=OPERATOR_ROLE_NAME,
        trust_policy=_operator_trust_policy(account_id),
        managed_policy_arns=[OPERATOR_POLICY_ARN],
        description=OPERATOR_DESCRIPTION,
        region_name=region_name,
    )

    await ensure_role(
        role_name=INSTANCE_ROLE_NAME,
        trust_policy=_instance_trust_policy(),
        managed_policy_arns=[INSTANCE_POLICY_ARN],
        description=INSTANCE_DESCRIPTION,
        region_name=region_name,
    )
    # The pool is given the profile, never the role: an instance can only be handed a
    # role through one.
    instance_profile_arn = await ensure_instance_profile(
        profile_name=INSTANCE_ROLE_NAME,
        role_name=INSTANCE_ROLE_NAME,
        region_name=region_name,
    )

    return operator_role_arn, instance_profile_arn


# --- capacity provider ------------------------------------------------------


async def find_capacity_provider(region_name: str, name: str) -> dict | None:
    """The capacity provider called ``name``, or ``None``.

    Looked up by name rather than by ARN because the name is what a caller's
    config can state ahead of time -- the ARN only exists once we have created it.
    """
    async with _control_client(region_name) as acrc:
        paginator = acrc.get_paginator("list_capacity_providers")
        async for page in paginator.paginate():
            for cp in page["capacityProviders"]:
                if cp["name"] == name:
                    return cp
    return None


async def create_capacity_provider(
    region_name: str,
    name: str,
    operator_role_arn: str,
    instance_profile_arn: str,
    subnets: list[str],
    security_groups: list[str],
    instance_type: str = "c5.large",
    root_throughput: int = 600,
    ssh_key_name: str | None = None,
) -> str:
    """Create the EC2 pool and return its ARN once it is READY.

    ``ssh_key_name`` is optional, as it is in the API: a pool created without one
    launches instances nobody holds a key for. Nothing about a rollout needs it --
    sessions are reached through the AgentCore data plane, not the instance -- so
    what it buys is a shell on a pool instance when a rollout misbehaves in a way
    only the host can explain, which is how the EBS throughput work was done. What
    it costs is that the key is a standing way onto a machine that runs other
    people's task containers.

    Like everything else in the compute configuration it cannot be added later, so a
    pool made without a key can only gain one under a new ``name``. That makes
    leaving it out the right default for a pool meant to be left alone, and worth
    setting for one being brought up to investigate something.
    """
    launch_parameters = {
        "operatingSystem": "LINUX_X86_64",
        "instanceRequirements": {"allowedInstanceTypes": [instance_type]},
        "capacityReservationSpecification": {"capacityReservationPreference": "open"},
        "instanceProfileArn": instance_profile_arn,
    }
    # Omitted rather than passed as None or "": the API rejects both, and an empty
    # string is what a config key left blank yields.
    if ssh_key_name:
        launch_parameters["sshKeyName"] = ssh_key_name

    async with _control_client(region_name) as acrc:
        resp = await acrc.create_capacity_provider(
            name=name,
            permissionsConfiguration={"capacityProviderOperatorRoleArn": operator_role_arn},
            computeConfiguration={
                "ec2Configuration": {
                    "launchTemplateSource": {"launchParameters": launch_parameters},
                    "vpcConfiguration": {
                        "subnets": subnets,
                        "securityGroups": security_groups,
                    },
                    "volumes": [],
                    "lifecycleConfiguration": INSTANCE_LIFECYCLE,
                    # throughput is capped per instance type (600 is the c5.large maximum), IOPS must be at least 4x
                    # https://docs.aws.amazon.com/ec2/latest/instancetypes/co.html
                    "rootVolume": {
                        "volumeType": "gp3",
                        "freeSpaceGiB": 100,
                        "throughput": root_throughput,
                        "iops": root_throughput * 4,
                    },
                }
            },
        )
        provider_id = resp["capacityProviderId"]
        await _wait_ready(
            lambda: acrc.get_capacity_provider(capacityProviderId=provider_id),
            f"capacity provider {name}",
        )

    return resp["capacityProviderArn"]


async def ensure_capacity_provider(region_name: str, name: str, **kwargs) -> str:
    """The ARN of the capacity provider called ``name``, creating it if absent.

    An existing provider is reported and left alone: its compute configuration
    cannot be updated, so there is nothing to reconcile and silently accepting a
    changed ``instance_type`` would be a lie. Use a new name for a new shape.
    """
    existing = await find_capacity_provider(region_name, name)
    if existing is not None:
        logger.info(
            "capacity provider %s already exists (%s), leaving it alone; "
            "its compute configuration is immutable, so change the name to change the pool",
            name,
            existing["status"],
        )
        return existing["capacityProviderArn"]

    logger.info("creating capacity provider %s", name)
    return await create_capacity_provider(region_name=region_name, name=name, **kwargs)


# --- agent runtime ----------------------------------------------------------


async def find_agent_runtime(region_name: str, name: str) -> dict | None:
    """The agent runtime called ``name``, or ``None``."""
    async with _control_client(region_name) as acrc:
        paginator = acrc.get_paginator("list_agent_runtimes")
        async for page in paginator.paginate():
            for rt in page["agentRuntimes"]:
                if rt["agentRuntimeName"] == name:
                    return rt
    return None


async def create_agentcore_runtime(
    runtime_name: str,
    region_name: str,
    image_uri: str,
    role_arn: str,
    capacity_provider_arn: str,
):
    async with _control_client(region_name) as acrc:
        resp = await acrc.create_agent_runtime(
            agentRuntimeName=runtime_name,
            agentRuntimeArtifact={"containerConfiguration": {"containerUri": image_uri}},
            roleArn=role_arn,
            capacityProviderConfiguration={"capacityProviderArn": capacity_provider_arn},
            lifecycleConfiguration=RUNTIME_LIFECYCLE,
        )
        runtime_id = resp["agentRuntimeId"]
        await _wait_ready(
            lambda: acrc.get_agent_runtime(agentRuntimeId=runtime_id),
            f"runtime {runtime_name}",
        )

    return resp


async def update_agentcore_runtime(
    runtime_arn: str,
    image_uri: str,
    role_arn: str,
    capacity_provider_arn: str,
):
    runtime_id = runtime_arn.split("/")[-1]
    region_name = runtime_arn.split(":")[3]

    async with _control_client(region_name) as acrc:
        await acrc.update_agent_runtime(
            agentRuntimeId=runtime_id,
            agentRuntimeArtifact={"containerConfiguration": {"containerUri": image_uri}},
            roleArn=role_arn,
            capacityProviderConfiguration={"capacityProviderArn": capacity_provider_arn},
            lifecycleConfiguration=RUNTIME_LIFECYCLE,
        )
        return await _wait_ready(
            lambda: acrc.get_agent_runtime(agentRuntimeId=runtime_id),
            f"runtime {runtime_id}",
        )


async def cleanup_agentcore_runtime(runtime_arn: str) -> str:
    """Delete every version of the runtime except the one its endpoint serves.

    Each update leaves its predecessor behind, and the versions are a quota, so
    this runs after every update rather than being an occasional chore.
    """
    runtime_id = runtime_arn.split("/")[-1]
    region_name = runtime_arn.split(":")[3]

    async with _control_client(region_name) as acrc:
        endpoints = await acrc.list_agent_runtime_endpoints(agentRuntimeId=runtime_id)
        current_version = endpoints["runtimeEndpoints"][0]["liveVersion"]

        paginator = acrc.get_paginator("list_agent_runtime_versions")
        iterator = paginator.paginate(agentRuntimeId=runtime_id, PaginationConfig={"MaxItems": 1000, "PageSize": 100})
        async for page in iterator:
            for rv in page["agentRuntimes"]:
                if rv["agentRuntimeVersion"] != current_version:
                    logger.info("deleting runtime %s version %s", runtime_id, rv["agentRuntimeVersion"])
                    await acrc.delete_agent_runtime(
                        agentRuntimeId=runtime_id, agentRuntimeVersion=rv["agentRuntimeVersion"]
                    )

    return current_version


async def ensure_agentcore_runtime(
    runtime_name: str,
    region_name: str,
    image_uri: str,
    role_arn: str,
    capacity_provider_arn: str,
) -> str:
    """The ARN of the runtime called ``runtime_name``, pointed at ``image_uri``.

    Creates it if absent, otherwise repoints the existing one and deletes the
    version it just superseded -- so re-running this after a fresh image build is
    the whole roll, and running it twice over is a no-op beyond one new version.
    """
    existing = await find_agent_runtime(region_name, runtime_name)
    if existing is None:
        logger.info("creating runtime %s on %s", runtime_name, image_uri)
        resp = await create_agentcore_runtime(
            runtime_name=runtime_name,
            region_name=region_name,
            image_uri=image_uri,
            role_arn=role_arn,
            capacity_provider_arn=capacity_provider_arn,
        )
        return resp["agentRuntimeArn"]

    runtime_arn = existing["agentRuntimeArn"]
    logger.info("updating runtime %s (version %s) to %s", runtime_name, existing["agentRuntimeVersion"], image_uri)
    await update_agentcore_runtime(
        runtime_arn=runtime_arn,
        image_uri=image_uri,
        role_arn=role_arn,
        capacity_provider_arn=capacity_provider_arn,
    )
    await cleanup_agentcore_runtime(runtime_arn)
    return runtime_arn
