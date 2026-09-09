"""AgentCore control plane: the capacity provider (EC2 pool) and the runtime on it,
plus the pool's two IAM roles. Provisioned once per image roll, not per rollout.

A runtime can be repointed at a new image, but a capacity provider's compute
configuration is immutable -- changing e.g. the instance type means a new provider name.
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

# Only bounds abandoned sessions; a rollout normally tears its own session down.
RUNTIME_LIFECYCLE = {
    "idleRuntimeSessionTimeout": 3600,
    "maxLifetime": 3600,
}

# The same for an unclaimed instance in the pool.
INSTANCE_LIFECYCLE = {
    "idleInstanceTimeout": 3600,
    "maxLifetime": 3600,
}


@asynccontextmanager
async def _control_client(region_name: str):
    async with (await get_aioboto3_session()).client("bedrock-agentcore-control", region_name=region_name) as acrc:  # type: ignore
        yield acrc


async def _wait_ready(describe, what: str) -> dict:
    """Poll ``describe`` until the resource is READY, raising on a terminal ``*_FAILED``."""
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

# AgentCore assumes the operator (a.k.a. infrastructure) role to launch, tag, network
# and reap the pool's EC2 instances. Use AWS's managed policy: a hand-written copy goes
# stale (AWS added PassRole to it in August 2026).
OPERATOR_ROLE_NAME = "AmazonBedrockAgentCoreCapacityProviderDefaultOperatorRole"
OPERATOR_POLICY_ARN = "arn:aws:iam::aws:policy/BedrockAgentCoreRuntimeInstancesOperatorRolePolicy"
OPERATOR_DESCRIPTION = "AgentCore capacity provider operator (infrastructure) role"

# Carried by each pool instance and shared by all sessions on it; grants system logging
# only. An agent's own permissions come from the runtime's execution role instead.
# The name is fixed: the operator policy's `iam:PassRole` is scoped to this prefix, so a
# differently named role cannot be passed to EC2 and instance launches fail.
INSTANCE_ROLE_NAME = "AmazonBedrockAgentCoreCapacityProviderDefaultInstanceRole"
INSTANCE_POLICY_ARN = "arn:aws:iam::aws:policy/BedrockAgentCoreRuntimeInstancesInstanceRolePolicy"
INSTANCE_DESCRIPTION = "AgentCore capacity provider instance role (system logs only)"


def _operator_trust_policy(account_id: str) -> dict:
    """Who may assume the operator role: AgentCore, on behalf of this account only.

    ``aws:SourceAccount`` is the confused-deputy guard; there is deliberately no source
    ARN condition, since ``CreateCapacityProvider`` assumes the role before the provider
    it would match exists. AWS documents neither this trust policy nor the condition keys
    its assume-role call carries, so suspect this first if creation fails on AssumeRole.
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
    """Who may assume the instance role: EC2. No account condition needed -- an instance
    profile in this account can only be attached to an instance in this account.
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

    Both ARNs are part of a provider's immutable compute configuration, so this cannot
    move an existing pool onto different roles.
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
    # The pool is given the profile, never the role: that is the only way EC2 takes one.
    instance_profile_arn = await ensure_instance_profile(
        profile_name=INSTANCE_ROLE_NAME,
        role_name=INSTANCE_ROLE_NAME,
        region_name=region_name,
    )

    return operator_role_arn, instance_profile_arn


# --- capacity provider ------------------------------------------------------


async def find_capacity_provider(region_name: str, name: str) -> dict | None:
    """The capacity provider called ``name``, or ``None``. By name, since a caller's
    config can state that ahead of time while the ARN only exists after creation.
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

    ``ssh_key_name`` is only for debugging a pool from its host; it cannot be added
    later, and it is a standing way onto a machine running shared task containers.
    """
    launch_parameters = {
        "operatingSystem": "LINUX_X86_64",
        "instanceRequirements": {"allowedInstanceTypes": [instance_type]},
        "capacityReservationSpecification": {"capacityReservationPreference": "open"},
        "instanceProfileArn": instance_profile_arn,
    }
    # Omitted rather than passed as None or "" (a blank config key): the API rejects both.
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

    An existing provider is left alone: its compute configuration is immutable, so use a
    new name for a new shape.
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

    Each update leaves its predecessor behind and versions are a quota, so this runs
    after every update.
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

    Creates it if absent, otherwise repoints the existing one and deletes the version it
    superseded.
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
