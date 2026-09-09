#!/usr/bin/env python

"""Build this recipe's agent image and put it on AgentCore -- the whole roll, one command.

Builds and tests the image, pushes it, reconciles the IAM roles, brings up the ECR pull
through cache and the DynamoDB session table, then creates or updates the capacity
provider and the runtime. Everything is looked up by name, so the first deploy and every
later image roll are the same invocation:

    ./deploy.py
    ./deploy.py --image-tag r12      # build and roll to a one-off tag
    ./deploy.py --skip-build         # repoint an existing image only
    ./deploy.py --skip-tests         # push what it built without testing it

An existing capacity provider is reported and left alone (its compute configuration is
immutable -- rename it in the config to reshape the pool). The execution role is
reconciled to ``iam_policy.py``, reverting grants made by hand. The session table only
accumulates, and the S3 dump bucket is deliberately not created here.
"""

import argparse
import asyncio
import logging
import shlex
from pathlib import Path

import iam_policy
from config import (
    CONFIG_PATH,
    RECIPE_DIR,
    TOOLKIT_ROOT,
    agent_repository,
    cache_prefix,
    load_config,
)
from verify_image import verify_image

from agentcore_rl_toolkit.aws_tools.agentcore_control import (
    ensure_agentcore_runtime,
    ensure_capacity_provider,
    ensure_capacity_provider_roles,
)
from agentcore_rl_toolkit.aws_tools.dynamodb_control import ensure_session_table
from agentcore_rl_toolkit.aws_tools.ecr_control import (
    ensure_docker_hub_cache_rule,
    ensure_registry_secret,
    find_pull_through_cache_rule,
    validate_pull_through_cache_rule,
)
from agentcore_rl_toolkit.aws_tools.iam_control import current_account_id, ensure_role

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="deploy config (default: %(default)s)",
    )
    parser.add_argument(
        "--image-tag",
        help="override the config's image_tag, for building and rolling a one-off tag",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="deploy the image that is already in the registry, without rebuilding it",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="push the image it built without running its unit tests in it first",
    )
    return parser.parse_args()


async def run(*argv: str) -> None:
    """Run a command with its output on ours, raising if it fails."""
    logger.info("+ %s", shlex.join(argv))
    proc = await asyncio.create_subprocess_exec(*argv)
    if await proc.wait() != 0:
        raise RuntimeError(f"command failed: {shlex.join(argv)}")


async def build_image(image_uri: str) -> None:
    """Build this recipe's ``Dockerfile`` as the local tag ``image_uri``.

    The toolkit repo root goes in as the named context ``rlpkg`` so the Dockerfile can copy
    the wire protocol package out of it without widening the context to the whole repo.
    """
    await run(
        "docker",
        "build",
        "-t",
        image_uri,
        "--build-context",
        f"rlpkg={TOOLKIT_ROOT}",
        str(RECIPE_DIR),
    )


async def push_image(image_uri: str) -> None:
    """Push the image the build and the tests just produced; registry auth is docker's own."""
    await run("docker", "push", image_uri)


async def ensure_execution_role(config: dict, region_name: str, account_id: str) -> str:
    """Reconcile the session execution role to ``iam_policy`` and return its ARN.

    Its ECR grants are scoped to the two repositories the config already names, so they
    cannot disagree with the image a runtime is pointed at or the cache it pulls from. The
    agent repository must be in the deploying account, since that is whose registry the
    grant ARNs name.
    """
    agentcore = config["agentcore"]
    agent = agent_repository(config)
    if agent.account_id != account_id:
        raise RuntimeError(
            f"docker_repo names registry {agent.account_id} but these credentials are "
            f"for account {account_id}: the execution role's ECR grants, and the task "
            f"image cache, are the deploying account's own registry"
        )

    return await ensure_role(
        role_name=agentcore["execution_role_name"],
        trust_policy=iam_policy.trust_policy(account_id, region_name),
        policies={
            iam_policy.POLICY_NAME: iam_policy.permissions_policy(
                account_id, region_name, cache_prefix(config), agent.path
            )
        },
        description=iam_policy.DESCRIPTION,
        region_name=region_name,
    )


async def ensure_pull_through_cache(config: dict, region_name: str) -> None:
    """Bring up the Docker Hub pull through cache the *task* images are pulled from.

    The rule and its Secrets Manager secret, both in ``region_name`` (a cache is regional).
    ``docker_hub.secret_name`` is what makes the rule ours to manage: without it the rule is
    only validated, the mode for a cache someone else owns.
    """
    docker_hub = config["docker_hub"]
    prefix = cache_prefix(config)
    secret_name = docker_hub.get("secret_name")

    if secret_name is None:
        if await find_pull_through_cache_rule(prefix, region_name) is None:
            raise RuntimeError(
                f"no pull through cache rule on {prefix!r} in {region_name}, and no "
                f"secret_name in [docker_hub] to create one with: task images are pulled "
                f"through that prefix, so give the secret holding a Docker Hub token"
            )
        logger.info(
            "no secret_name in [docker_hub]: leaving the %s pull through cache rule in %s "
            "as it is, and only checking that it works",
            prefix,
            region_name,
        )
        await validate_pull_through_cache_rule(prefix, region_name)
        return

    credential_arn = await ensure_registry_secret(
        secret_name=secret_name,
        region_name=region_name,
        username=docker_hub.get("username"),
        access_token=docker_hub.get("access_token"),
    )
    await ensure_docker_hub_cache_rule(
        prefix=prefix,
        region_name=region_name,
        credential_arn=credential_arn,
    )


async def main():
    args = parse_args()
    config = load_config(args.config)
    agentcore = config["agentcore"]
    region_name = agentcore["region"]
    image_uri = agentcore["docker_repo"] + ":" + (args.image_tag or agentcore["image_tag"])

    if args.skip_build:
        # Nothing built locally means nothing local to test either.
        logger.info("skipping build and tests, deploying %s as it is in the registry", image_uri)
    else:
        await build_image(image_uri)
        if args.skip_tests:
            logger.info("skipping the unit tests in %s, pushing it untested", image_uri)
        else:
            await verify_image(image_uri)
        await push_image(image_uri)

    # Roles and pool first: a runtime is created naming an execution role and a READY pool.
    account_id = await current_account_id(region_name)
    execution_role_arn = await ensure_execution_role(config, region_name, account_id)
    await ensure_pull_through_cache(config, region_name)
    # Written with ambient credentials by the eval and the trainer, not with the execution
    # role, which is why nothing in iam_policy mentions this table.
    await ensure_session_table(config["storage"]["dynamodb_table"], region_name)
    operator_role_arn, instance_profile_arn = await ensure_capacity_provider_roles(region_name)
    capacity_provider_arn = await ensure_capacity_provider(
        region_name=region_name,
        operator_role_arn=operator_role_arn,
        instance_profile_arn=instance_profile_arn,
        **agentcore["capacity_provider"],
    )

    runtime_arn = await ensure_agentcore_runtime(
        runtime_name=agentcore["runtime_name"],
        region_name=region_name,
        image_uri=image_uri,
        role_arn=execution_role_arn,
        capacity_provider_arn=capacity_provider_arn,
    )

    # Printed as .env assignments because that is where they go: the trainer reads them as
    # hydra interpolations. evaluate.py does not need them.
    print(f"agent_iam_role_arn={execution_role_arn}")
    print(f"agentcore_capacity_provider_arn={capacity_provider_arn}")
    print(f"agentcore_runtime_arn={runtime_arn}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    asyncio.run(main())
