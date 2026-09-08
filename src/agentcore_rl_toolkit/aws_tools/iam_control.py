"""IAM *control plane*: create-or-update a role from the policies you hand it.

The same shape as the other control-plane modules here: a resource a deploy has to
bring into existence before it can point anything at it. This module holds only the
create/update mechanics; every policy document is the caller's, so nothing here
knows what a role is *for*.

:func:`ensure_role` is declarative: what you pass is the role's whole permission set
afterwards, including the *absence* of anything you did not pass. An inline policy
already on the role and not named in the call is deleted, and so is an attachment of
a managed policy the call does not list -- the alternative is a role whose real
permissions are the union of the checked-in document and whatever a past console
session left behind, the drift that makes "what can this role actually do?"
unanswerable without an API call. Both are logged individually so a surprise shows
up in the deploy output.

The two kinds of grant are not alike, which is worth keeping in mind when deciding
where a permission belongs. An inline document is *ours*: what we decided this role
may do, scopeable to our own resources. An attached AWS managed policy is *AWS's*:
the vendor's answer to what a role in one of their features needs, and it changes
under us when they add an action. For a role whose job is defined by an AWS feature
-- the operator and instance roles of an AgentCore capacity provider -- that is the
point: better to inherit the next permission the feature needs than discover it as
an outage.
"""

import asyncio
import json
import logging
from collections.abc import Sequence

import botocore.exceptions

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session

logger = logging.getLogger(__name__)

# IAM is eventually consistent, and a role is not immediately assumable by the
# service principal its freshly written trust policy names. The next thing a deploy
# does is hand the role to another service, which rejects one it cannot yet assume
# ("Unable to assume role"), so a brand-new role gets a pause before it is used.
# Only on create: an update leaves an already-propagated role in place.
ROLE_PROPAGATION_DELAY_SECONDS = 12


async def _iam_client(region_name: str | None = None):
    """An IAM client. ``region_name`` only picks the endpoint; IAM itself is global."""
    return (await get_aioboto3_session()).client("iam", region_name=region_name)  # type: ignore


async def current_account_id(region_name: str | None = None) -> str:
    """The account the ambient credentials are in.

    Every ARN in a policy document needs it, and asking STS is better than making
    the caller restate in config something the credentials already know -- a
    mismatch between the two would grant on one account while deploying to another.
    """
    async with (await get_aioboto3_session()).client("sts", region_name=region_name) as sts:  # type: ignore
        return (await sts.get_caller_identity())["Account"]


async def find_role(role_name: str, region_name: str | None = None) -> dict | None:
    """The role called ``role_name``, or ``None``.

    By name rather than ARN, like every other ``find_*`` here: the name is what a
    caller's config can state before the role exists.
    """
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

    Creates the role if absent, otherwise rewrites its trust policy, description,
    inline policies and managed policy attachments in place -- so the first deploy
    and every later one are the same call, and running it twice over changes
    nothing. ``policies`` maps inline policy name to policy document, and
    ``managed_policy_arns`` lists the managed policies to attach; see the module
    docstring on why any inline policy or attachment *not* named here is removed,
    and on when to reach for which.
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
                # Loud, because this is the one thing here that takes a permission
                # away: it means the role carried a grant that the checked-in
                # document does not, and the document wins.
                logger.warning(
                    "deleting inline policy %s from %s: it is not in the deployed document",
                    policy_name,
                    role_name,
                )
                await iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)

        # Attaching one that is already attached is a no-op, so this needs no
        # membership check -- unlike the detach below, which does.
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

    An instance profile is the wrapper EC2 needs in order to give a role to an
    instance: nothing but the profile can be handed to a launch, and a profile holds
    at most one role. So this creates the profile if absent and then makes
    ``role_name`` the role in it, swapping out a different one if it finds it --
    which keeps the same declarative promise as :func:`ensure_role`, at the one
    resource where IAM's shape forces a second call to keep it.

    The role must already exist; create it with :func:`ensure_role` first.
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
        # Instance profiles propagate to EC2 more slowly than roles do to STS, and
        # a launch that races it fails with "Invalid IAM Instance Profile name".
        logger.info(
            "waiting %ss for instance profile %s to propagate before EC2 is given it",
            ROLE_PROPAGATION_DELAY_SECONDS,
            profile_name,
        )
        await asyncio.sleep(ROLE_PROPAGATION_DELAY_SECONDS)

    return existing["Arn"]
