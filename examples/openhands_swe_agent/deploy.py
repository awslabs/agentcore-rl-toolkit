#!/usr/bin/env python

"""Build this recipe's agent image and put it on AgentCore -- the whole roll, one command.

Six steps that used to be separate scripts or hand edits: build the image, test it,
push it, reconcile the IAM roles, bring up the ECR pull through cache the task images
arrive by and the DynamoDB table the rollouts are recorded in, then create or update
the capacity provider, then create or update the runtime. Everything here is looked
up by name and does whichever of create/update is called for, so the first deploy
and every later image roll are the same invocation:

    ./deploy.py
    ./deploy.py --image-tag r12      # build and roll to a one-off tag
    ./deploy.py --skip-build         # repoint an existing image only
    ./deploy.py --skip-tests         # push what it built without testing it

The build comes first so that a failed build cannot leave the runtime pointed at a
tag that does not exist -- and the harness's own unit tests run in the image between
the build and the push (:mod:`verify_image`), so neither can a failing harness.

The AWS resources differ in what "already exists" allows. A role and a runtime can
both be rewritten in place, so an existing one is updated -- and for the runtime,
the version it superseded is deleted. A capacity provider's compute configuration
is immutable, so an existing one is reported and left alone -- change
``capacity_provider.name`` in the config to get a differently shaped pool, and the
runtime is repointed at it. Its operator role and instance profile are part of that
immutable configuration too, so a *change* to either only reaches a pool created
under a new name.

The pull through cache is in the same region as everything else here, and that is the
point of it being here: a cache is regional, so the rule and the Docker Hub secret
behind it are made in ``agentcore.region``, where the sessions that pull task images
run. A second region means a second deploy of this recipe against it -- the cache
follows the runtime rather than being pointed anywhere independently.

The session table is the one resource here that is neither built nor pointed at
anything: it accumulates. Every rollout of every run this recipe has ever done is an
item in it, so a deploy only ever creates it or adds the index the analysis needs --
see :mod:`~agentcore_rl_toolkit.aws_tools.dynamodb_control` for what a mismatch it
refuses to fix looks like. The S3 bucket the rollout dumps go to is *not* created
here, and deliberately: a bucket is a name in a global namespace with a lifecycle
policy and a retention decision behind it, not a per-recipe resource.

The execution role is *reconciled*, not merely created: ``iam_policy.py``
is its whole permission set afterwards, so a grant added by hand in the console is
reverted by the next deploy. That is the point -- it is what makes the checked-in
document a truthful answer to what a rollout container may do. The pool's own two
roles are brought up as well, but they are not the recipe's to describe and carry
AWS's managed policies instead (``aws_tools/agentcore_control.py``).

Config is the recipe's own ``config.toml`` (see :mod:`config`) and nothing else -- in
particular not the developer's ``.env``: which image this agent runs on which pool is
a property of the recipe, so it travels with the recipe. The ARNs printed at the end
are for the trainer, which still reads them from ``.env``.
"""

import argparse
import asyncio
import logging
import shlex
from pathlib import Path

# This recipe's own modules, imported as top-level names: these scripts are
# entrypoints run from this directory (``./deploy.py``), and ``conftest.py`` puts the
# same directory on ``sys.path`` so the tests resolve them identically.
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
    """Run a command with its output on ours, raising if it fails.

    The echo and the raise are the ``set -eux`` this replaced: docker's own output
    is the interesting part, so it is inherited rather than captured.
    """
    logger.info("+ %s", shlex.join(argv))
    proc = await asyncio.create_subprocess_exec(*argv)
    if await proc.wait() != 0:
        raise RuntimeError(f"command failed: {shlex.join(argv)}")


async def build_image(image_uri: str) -> None:
    """Build this recipe's ``Dockerfile`` as the local tag ``image_uri``.

    The build context is this directory, plus the toolkit repo's root as the named
    context ``rlpkg`` -- the Dockerfile copies the wire protocol package out of it
    without widening the context to the whole repo, which would invalidate its layers
    on any unrelated change.

    Building and pushing are separate because the unit tests run between them
    (:mod:`verify_image`): what is tested is the image the push then publishes, and a
    harness that fails its tests never reaches the registry, let alone a runtime.
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
    """Push the image the build and the tests just produced.

    Registry auth is docker's own -- an ambient ECR credential helper, or a prior
    ``docker login`` -- not this script's.
    """
    await run("docker", "push", image_uri)


async def ensure_execution_role(config: dict, region_name: str, account_id: str) -> str:
    """Reconcile the session execution role to ``iam_policy`` and return its ARN.

    The two repositories its ECR grants are scoped to are the two the config already
    names: ``agentcore.docker_repo`` for the agent image and ``docker_hub.cache_prefix``
    for the task images. Neither is restated here -- a grant that could disagree with
    the image a runtime is actually pointed at, or with the cache the rollouts
    actually pull from, is the one thing this reconcile must not be able to do.

    The agent repository has to be in the deploying account, because that is whose
    registry the ARNs name -- and it is also the registry the task image cache is
    assembled from (``config.task_image_namespace``), so a mismatch here would be a
    namespace nothing pulls from either. A repository in another account never worked
    here, the grant having always been scoped to one account, so this says so instead
    of writing a policy that cannot authorise the pull it is for.
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

    The rule and the Secrets Manager secret it authenticates with, both in
    ``region_name``: a cache is regional, and this is the region whose sessions pull
    through it.

    ``docker_hub.secret_name`` is what makes the rule ours to manage. With it, the
    rule is created if absent and repointed at that secret otherwise, and the
    credentials in the config are the secret's value -- see
    :func:`~agentcore_rl_toolkit.aws_tools.ecr_control.ensure_registry_secret` for what
    naming the secret without credentials does. Without it the rule is left exactly as
    it is and only validated, which is the mode for a cache someone else owns: the
    recipe still needs a working rule on that prefix, so this reports what it skipped
    and fails on a prefix that has no rule at all rather than passing silently.
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
        # Nothing was built here, so there is nothing local to test either: the image
        # named is one the registry already has, tested by whichever deploy built it.
        logger.info("skipping build and tests, deploying %s as it is in the registry", image_uri)
    else:
        await build_image(image_uri)
        if args.skip_tests:
            logger.info("skipping the unit tests in %s, pushing it untested", image_uri)
        else:
            await verify_image(image_uri)
        await push_image(image_uri)

    # The roles and the pool first: the runtime is created naming an execution role
    # and a pool, so each has to exist -- and the pool be READY -- before we can ask
    # for a runtime at all. The pool's own two roles come before the pool for the
    # same reason.
    account_id = await current_account_id(region_name)
    execution_role_arn = await ensure_execution_role(config, region_name, account_id)
    # The task images' side of the same permission: the role may populate the cache,
    # and this is the cache it populates.
    await ensure_pull_through_cache(config, region_name)
    # Where the rollouts are recorded, in the region they are recorded from: the eval
    # and the trainer both write session items with the ambient credentials, not with
    # the execution role, which is why nothing in iam_policy mentions this table.
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

    # Printed as .env assignments because that is where they are going: the
    # trainer reads them as hydra interpolations seeded from the project .env.
    # evaluate.py does not need them -- it resolves the runtime and the pool from
    # the names in config.toml, and never assumes the role itself.
    print(f"agent_iam_role_arn={execution_role_arn}")
    print(f"agentcore_capacity_provider_arn={capacity_provider_arn}")
    print(f"agentcore_runtime_arn={runtime_arn}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    asyncio.run(main())
