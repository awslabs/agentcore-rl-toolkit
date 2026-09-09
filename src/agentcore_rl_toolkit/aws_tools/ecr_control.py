"""ECR *control plane*: create-or-update the pull through cache the task images arrive by.

A pull through cache rule mirrors Docker Hub under a prefix in this account's regional
ECR registry, so a run's thousands of task-image pulls do not hit Docker Hub's rate
limit. Docker Hub needs credentials, and ECR demands its Secrets Manager secret be
named ``ecr-pullthroughcache/*``, hold JSON with exactly ``username``/``accessToken``,
and live in the rule's own account and region.
"""

import json
import logging
from contextlib import asynccontextmanager

import botocore.exceptions

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session

logger = logging.getLogger(__name__)

# Part of a rule's immutable identity (only its credential can be updated), hence
# constants rather than arguments.
DOCKER_HUB_UPSTREAM_REGISTRY = "docker-hub"
DOCKER_HUB_UPSTREAM_URL = "registry-1.docker.io"

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

    By name, not ARN: Secrets Manager suffixes every ARN with six random characters,
    so the name is the only handle a caller's config can state up front.
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

    Creates it if absent and puts a new version only if the credentials differ, so a
    no-op deploy writes no version. ``username`` and ``access_token`` are optional
    *together*, meaning "keep the value already there".
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

    Worth calling on every deploy: otherwise a stale credential only shows up as a
    ``name unknown`` pull failure inside a rollout, which reads as a broken dataset.
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

    Creates the rule if absent, else repoints it at ``credential_arn`` -- the only
    updatable part. An existing rule on the same prefix for a different upstream is
    an error, since ECR cannot move it.
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
