"""IAM *control plane*: create-or-update a role from the policies you hand it.

Every policy document is the caller's. :func:`ensure_role` is declarative -- what you
pass is the role's whole permission set afterwards, so any inline policy or managed
attachment not named in the call is removed (and logged).
"""

import asyncio
import json
import logging
from collections.abc import Sequence

import botocore.exceptions

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session

logger = logging.getLogger(__name__)

# IAM is eventually consistent: a brand-new role is not yet assumable, and handing it
# straight to another service fails with "Unable to assume role". Only on create.
ROLE_PROPAGATION_DELAY_SECONDS = 12


async def _iam_client(region_name: str | None = None):
    """An IAM client. ``region_name`` only picks the endpoint; IAM itself is global."""
    return (await get_aioboto3_session()).client("iam", region_name=region_name)  # type: ignore


async def current_account_id(region_name: str | None = None) -> str:
    """The account the ambient credentials are in, per STS."""
    async with (await get_aioboto3_session()).client("sts", region_name=region_name) as sts:  # type: ignore
        return (await sts.get_caller_identity())["Account"]


async def find_role(role_name: str, region_name: str | None = None) -> dict | None:
    """The role called ``role_name``, or ``None``."""
    async with await _iam_client(region_name) as iam:
        try:
            return (await iam.get_role(RoleName=role_name))["Role"]
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] == "NoSuchEntity":
                return None
            raise


async def ensure_role(
    role_name: str,
    trust_policy: dict,
    policies: dict[str, dict] | None = None,
    managed_policy_arns: Sequence[str] = (),
    description: str = "",
    region_name: str | None = None,
) -> str:
    """The ARN of the role called ``role_name``, reconciled to the given policies.

    ``policies`` maps inline policy name to document. Anything on the role but not
    named here is removed; see the module docstring.
    """
    policies = policies or {}
    existing = await find_role(role_name, region_name)

    async with await _iam_client(region_name) as iam:
        if existing is None:
            logger.info("creating role %s", role_name)
            existing = (
                await iam.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(trust_policy),
                    Description=description,
                )
            )["Role"]
            created = True
        else:
            logger.info("updating role %s", role_name)
            await iam.update_assume_role_policy(RoleName=role_name, PolicyDocument=json.dumps(trust_policy))
            await iam.update_role(RoleName=role_name, Description=description)
            created = False

        for policy_name, document in policies.items():
            logger.info("putting inline policy %s on %s", policy_name, role_name)
            await iam.put_role_policy(
                RoleName=role_name,
                PolicyName=policy_name,
                PolicyDocument=json.dumps(document),
            )

        paginator = iam.get_paginator("list_role_policies")
        async for page in paginator.paginate(RoleName=role_name):
            for policy_name in page["PolicyNames"]:
                if policy_name in policies:
                    continue
                # Loud: this takes a permission away from the role.
                logger.warning(
                    "deleting inline policy %s from %s: it is not in the deployed document",
                    policy_name,
                    role_name,
                )
                await iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)

        # Attaching an already-attached policy is a no-op, so no membership check.
        for policy_arn in managed_policy_arns:
            logger.info("attaching managed policy %s to %s", policy_arn, role_name)
            await iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)

        paginator = iam.get_paginator("list_attached_role_policies")
        async for page in paginator.paginate(RoleName=role_name):
            for attached in page["AttachedPolicies"]:
                if attached["PolicyArn"] in managed_policy_arns:
                    continue
                logger.warning(
                    "detaching managed policy %s from %s: it is not one the deploy attaches",
                    attached["PolicyArn"],
                    role_name,
                )
                await iam.detach_role_policy(RoleName=role_name, PolicyArn=attached["PolicyArn"])

    if created:
        logger.info(
            "waiting %ss for role %s to propagate before anything assumes it",
            ROLE_PROPAGATION_DELAY_SECONDS,
            role_name,
        )
        await asyncio.sleep(ROLE_PROPAGATION_DELAY_SECONDS)

    return existing["Arn"]


async def find_instance_profile(profile_name: str, region_name: str | None = None) -> dict | None:
    """The instance profile called ``profile_name``, or ``None``."""
    async with await _iam_client(region_name) as iam:
        try:
            return (await iam.get_instance_profile(InstanceProfileName=profile_name))["InstanceProfile"]
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] == "NoSuchEntity":
                return None
            raise


async def ensure_instance_profile(
    profile_name: str,
    role_name: str,
    region_name: str | None = None,
) -> str:
    """The ARN of the instance profile called ``profile_name``, holding ``role_name``.

    A profile holds at most one role, so any other role found in it is removed. The
    role must already exist; create it with :func:`ensure_role` first.
    """
    existing = await find_instance_profile(profile_name, region_name)

    async with await _iam_client(region_name) as iam:
        if existing is None:
            logger.info("creating instance profile %s", profile_name)
            existing = (await iam.create_instance_profile(InstanceProfileName=profile_name))["InstanceProfile"]
            created = True
        else:
            created = False

        held = [role["RoleName"] for role in existing["Roles"]]
        for other in held:
            if other == role_name:
                continue
            logger.warning(
                "removing role %s from instance profile %s: the deploy puts %s in it",
                other,
                profile_name,
                role_name,
            )
            await iam.remove_role_from_instance_profile(InstanceProfileName=profile_name, RoleName=other)
        if role_name not in held:
            logger.info("adding role %s to instance profile %s", role_name, profile_name)
            await iam.add_role_to_instance_profile(InstanceProfileName=profile_name, RoleName=role_name)

    if created:
        # Slower to propagate than roles; a launch that races it fails with
        # "Invalid IAM Instance Profile name".
        logger.info(
            "waiting %ss for instance profile %s to propagate before EC2 is given it",
            ROLE_PROPAGATION_DELAY_SECONDS,
            profile_name,
        )
        await asyncio.sleep(ROLE_PROPAGATION_DELAY_SECONDS)

    return existing["Arn"]
