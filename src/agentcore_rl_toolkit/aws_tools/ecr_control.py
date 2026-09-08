"""ECR *control plane*: the pull through cache the task images arrive by.

The same shape as the other control-plane modules here: a resource a deploy has to
bring into existence before it can point a rollout at it, looked up by name, created
if absent, updated if present, and reading no config of its own -- every value is
the caller's.

A **pull through cache rule** mirrors an upstream registry under a prefix in this
account's ECR registry, so that a pull of ``<prefix>/<upstream path>`` populates the
mirror on first request and is served from ECR afterwards. That is how *task* images
-- one SWE-bench or SWE-Gym environment each -- are pulled: straight from Docker
Hub, a training run's few thousand pulls meet the rate limit long before the run is
done, and a throttled pull surfaces as an aborted rollout rather than as anything
mentioning Docker Hub.

A rule is regional, so a caller needs one per region it pulls task images in.

**The secret.** Docker Hub is not one of the upstreams ECR will mirror
anonymously, so a rule for it names a Secrets Manager secret holding a Docker Hub
username and access token. ECR is particular about that secret in three ways, all
of them easier to know than to debug: its name must begin with
``ecr-pullthroughcache/``, its value must be JSON with exactly the keys
``username`` and ``accessToken``, and it must live in the same account and region
as the rule. :func:`ensure_registry_secret` writes one of the right shape and
:data:`SECRET_NAME_PREFIX` is checked rather than assumed, because the failure for
a wrongly named secret is an ``InvalidParameterException`` on the rule, not on the
secret.
"""

import json
import logging
from contextlib import asynccontextmanager

import botocore.exceptions

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session

logger = logging.getLogger(__name__)

# Docker Hub as ECR names it. ``upstreamRegistry`` is the well-known name that
# decides how ECR authenticates and rewrites paths; ``upstreamRegistryUrl`` is the
# host it actually talks to. Both are part of a rule's immutable identity -- only
# the credential can be updated -- so they are constants here rather than arguments:
# this recipe caches Docker Hub, and a rule for some other upstream is a different
# rule under a different prefix.
DOCKER_HUB_UPSTREAM_REGISTRY = "docker-hub"
DOCKER_HUB_UPSTREAM_URL = "registry-1.docker.io"

# The name prefix ECR requires of a secret it will read credentials from. See the
# module docstring.
SECRET_NAME_PREFIX = "ecr-pullthroughcache/"

SECRET_DESCRIPTION = "Docker Hub credentials for an ECR pull through cache rule"


@asynccontextmanager
async def _ecr_client(region_name: str):
    async with (await get_aioboto3_session()).client("ecr", region_name=region_name) as ecr:  # type: ignore
        yield ecr


@asynccontextmanager
async def _secrets_client(region_name: str):
    async with (await get_aioboto3_session()).client("secretsmanager", region_name=region_name) as sm:  # type: ignore
        yield sm


# --- the credential secret --------------------------------------------------


async def find_secret(secret_name: str, region_name: str) -> dict | None:
    """The secret called ``secret_name``, or ``None``.

    By name rather than ARN, like the other ``find_*`` in this package: Secrets
    Manager appends six random characters to the ARN of every secret, so the name
    is the only handle a caller's config can state before the secret exists.
    """
    async with _secrets_client(region_name) as sm:
        try:
            return await sm.describe_secret(SecretId=secret_name)
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] == "ResourceNotFoundException":
                return None
            raise


async def ensure_registry_secret(
    secret_name: str,
    region_name: str,
    username: str | None = None,
    access_token: str | None = None,
) -> str:
    """The ARN of the secret called ``secret_name``, holding the given credentials.

    Creates it if absent and puts a new version if the credentials differ from
    what it already holds, so a deploy that changes nothing writes nothing -- a
    version per deploy would be churn in the one place where the history is
    credentials.

    ``username`` and ``access_token`` are optional *together*, and leaving them out
    means "use the secret that is already there": the rule is still repointed at
    it, but its value is not rewritten. That is the mode for a deployer who would
    rather not keep an access token in a config file, and it is why a missing
    secret with no credentials to put in it is an error naming both -- there is
    nothing this can do about it, and the rule would fail to validate a moment
    later with a message about the rule instead.
    """
    if not secret_name.startswith(SECRET_NAME_PREFIX):
        raise ValueError(
            f"secret name {secret_name!r} cannot back a pull through cache rule: "
            f"ECR only reads credentials from a secret named {SECRET_NAME_PREFIX}*"
        )
    if (username is None) != (access_token is None):
        raise ValueError(
            "a registry credential needs both a username and an access token, or "
            f"neither to keep the value already in {secret_name!r}"
        )

    existing = await find_secret(secret_name, region_name)
    # ECR reads the value by these key names; see the module docstring.
    secret_string = json.dumps({"username": username, "accessToken": access_token})

    async with _secrets_client(region_name) as sm:
        if existing is None:
            if access_token is None:
                raise RuntimeError(
                    f"no secret named {secret_name!r} in {region_name} and no credentials "
                    f"to create it with: give a username and an access_token in the config"
                )
            logger.info("creating secret %s", secret_name)
            created = await sm.create_secret(
                Name=secret_name,
                Description=SECRET_DESCRIPTION,
                SecretString=secret_string,
            )
            return created["ARN"]

        secret_arn = existing["ARN"]
        if access_token is None:
            logger.info("secret %s already exists; leaving its value alone", secret_name)
            return secret_arn

        current = await sm.get_secret_value(SecretId=secret_arn)
        if current.get("SecretString") == secret_string:
            logger.info("secret %s already holds these credentials", secret_name)
        else:
            logger.info("putting a new version of secret %s", secret_name)
            await sm.put_secret_value(SecretId=secret_arn, SecretString=secret_string)
        return secret_arn


# --- the pull through cache rule --------------------------------------------


async def find_pull_through_cache_rule(prefix: str, region_name: str) -> dict | None:
    """The pull through cache rule on ``prefix`` in this account's registry, or ``None``."""
    async with _ecr_client(region_name) as ecr:
        try:
            described = await ecr.describe_pull_through_cache_rules(ecrRepositoryPrefixes=[prefix])
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] == "PullThroughCacheRuleNotFoundException":
                return None
            raise
    rules = described["pullThroughCacheRules"]
    return rules[0] if rules else None


async def validate_pull_through_cache_rule(prefix: str, region_name: str) -> None:
    """Raise unless ECR can reach the upstream registry with the rule's credential.

    Worth a call on every deploy because of what a bad credential looks like
    otherwise: the rule exists, so a pull of an image the cache already holds still
    works, and only an image nobody has pulled before fails -- as
    ``name unknown: The repository with name ... does not exist``, from inside a
    rollout container, which reads as a broken dataset rather than as an expired
    access token. This is the one place that can be turned into a deploy-time error.
    """
    async with _ecr_client(region_name) as ecr:
        result = await ecr.validate_pull_through_cache_rule(ecrRepositoryPrefix=prefix)
    if not result["isValid"]:
        raise RuntimeError(
            f"pull through cache rule {prefix} in {region_name} does not work: "
            f"{result.get('failure') or 'no reason given'} -- if the credentials are "
            f"stale, put a current Docker Hub access token in the config and deploy again"
        )
    logger.info("pull through cache rule %s in %s validates", prefix, region_name)


async def ensure_docker_hub_cache_rule(
    prefix: str,
    region_name: str,
    credential_arn: str,
) -> str:
    """Reconcile the Docker Hub cache rule on ``prefix``; return its ECR prefix.

    Creates the rule if absent and otherwise repoints it at ``credential_arn`` if
    that is not already what it names -- the one part of a rule that can be
    updated, which is what makes a rotated credential a redeploy rather than a
    delete. An existing rule on the same prefix for a *different* upstream is an
    error: ECR cannot move it, and silently caching a different registry under the
    prefix the task images are pulled from is the worst available outcome.

    Deleting a rule leaves the repositories it created behind, so recreating one
    under the same prefix picks the existing mirror back up rather than starting
    cold.
    """
    existing = await find_pull_through_cache_rule(prefix, region_name)

    async with _ecr_client(region_name) as ecr:
        if existing is None:
            logger.info(
                "creating pull through cache rule %s -> %s in %s",
                prefix,
                DOCKER_HUB_UPSTREAM_URL,
                region_name,
            )
            await ecr.create_pull_through_cache_rule(
                ecrRepositoryPrefix=prefix,
                upstreamRegistry=DOCKER_HUB_UPSTREAM_REGISTRY,
                upstreamRegistryUrl=DOCKER_HUB_UPSTREAM_URL,
                credentialArn=credential_arn,
            )
        else:
            upstream = existing.get("upstreamRegistryUrl")
            if upstream != DOCKER_HUB_UPSTREAM_URL:
                raise RuntimeError(
                    f"pull through cache rule {prefix} in {region_name} already caches "
                    f"{upstream}, not {DOCKER_HUB_UPSTREAM_URL}; a rule's upstream cannot be "
                    f"updated, so either use another prefix or delete the rule first"
                )
            if existing.get("credentialArn") == credential_arn:
                logger.info(
                    "pull through cache rule %s in %s already names %s",
                    prefix,
                    region_name,
                    credential_arn,
                )
            else:
                logger.info(
                    "repointing pull through cache rule %s in %s at %s",
                    prefix,
                    region_name,
                    credential_arn,
                )
                await ecr.update_pull_through_cache_rule(ecrRepositoryPrefix=prefix, credentialArn=credential_arn)

    await validate_pull_through_cache_rule(prefix, region_name)
    return prefix
