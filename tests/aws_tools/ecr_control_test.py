#!/usr/bin/env python
"""Unit tests for the ECR pull through cache control plane: which deploys write, which
are no-ops, and the shapes ECR is strict about. Fake ECR and Secrets Manager, no AWS.
"""

import asyncio
import json
import unittest
from contextlib import asynccontextmanager
from unittest import mock

import botocore.exceptions

from agentcore_rl_toolkit.aws_tools import ecr_control

MODULE = "agentcore_rl_toolkit.aws_tools.ecr_control"
REGION = "us-west-2"
PREFIX = "docker-hub"
SECRET_NAME = "ecr-pullthroughcache/swe"
SECRET_ARN = f"arn:aws:secretsmanager:{REGION}:123456789012:secret:{SECRET_NAME}-AbCdEf"
OTHER_ARN = f"arn:aws:secretsmanager:{REGION}:123456789012:secret:{SECRET_NAME}-Zz9999"


def credential(username: str = "someone", token: str = "dckr_pat_current") -> str:
    """A secret value in the shape ECR reads."""
    return json.dumps({"username": username, "accessToken": token})


def client_error(code: str, operation: str) -> botocore.exceptions.ClientError:
    return botocore.exceptions.ClientError({"Error": {"Code": code, "Message": ""}}, operation)


class FakeSecretsManager:
    """Secrets Manager, one secret deep."""

    def __init__(self, value: str | None = None):
        self.value = value
        self.created: list[dict] = []
        self.put: list[str] = []

    async def describe_secret(self, SecretId):
        if self.value is None:
            raise client_error("ResourceNotFoundException", "DescribeSecret")
        return {"ARN": SECRET_ARN, "Name": SECRET_NAME}

    async def get_secret_value(self, SecretId):
        return {"ARN": SECRET_ARN, "SecretString": self.value}

    async def create_secret(self, Name, Description, SecretString):
        self.created.append({"Name": Name, "SecretString": SecretString})
        self.value = SecretString
        return {"ARN": SECRET_ARN, "Name": Name}

    async def put_secret_value(self, SecretId, SecretString):
        self.put.append(SecretString)
        self.value = SecretString
        return {"ARN": SECRET_ARN, "VersionId": "v2"}


class FakeEcr:
    """ECR's pull through cache rules, keyed by prefix as the real API keys them."""

    def __init__(self, rules: dict | None = None, is_valid: bool = True, empty_describe: bool = False):
        self.rules = dict(rules or {})
        self.is_valid = is_valid
        # Unknown prefix reported as an empty list instead of an error; both are handled.
        self.empty_describe = empty_describe
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self.validated: list[str] = []

    async def describe_pull_through_cache_rules(self, ecrRepositoryPrefixes):
        found = [self.rules[p] for p in ecrRepositoryPrefixes if p in self.rules]
        if not found and not self.empty_describe:
            raise client_error("PullThroughCacheRuleNotFoundException", "DescribePullThroughCacheRules")
        return {"pullThroughCacheRules": found}

    async def create_pull_through_cache_rule(self, **kwargs):
        self.created.append(kwargs)
        self.rules[kwargs["ecrRepositoryPrefix"]] = dict(kwargs)
        return dict(kwargs)

    async def update_pull_through_cache_rule(self, ecrRepositoryPrefix, credentialArn):
        self.updated.append({"ecrRepositoryPrefix": ecrRepositoryPrefix, "credentialArn": credentialArn})
        self.rules[ecrRepositoryPrefix]["credentialArn"] = credentialArn
        return self.rules[ecrRepositoryPrefix]

    async def validate_pull_through_cache_rule(self, ecrRepositoryPrefix):
        self.validated.append(ecrRepositoryPrefix)
        return {
            "ecrRepositoryPrefix": ecrRepositoryPrefix,
            "isValid": self.is_valid,
            "failure": None if self.is_valid else "unauthorized: incorrect username or password",
        }


def existing_rule(credential_arn: str = SECRET_ARN, upstream: str | None = None) -> dict:
    return {
        "ecrRepositoryPrefix": PREFIX,
        "upstreamRegistryUrl": upstream or ecr_control.DOCKER_HUB_UPSTREAM_URL,
        "upstreamRegistry": "docker-hub",
        "credentialArn": credential_arn,
    }


@asynccontextmanager
async def fake_client(client):
    yield client


def patch_clients(ecr=None, secrets=None):
    return mock.patch.multiple(
        MODULE,
        _ecr_client=lambda region_name: fake_client(ecr),
        _secrets_client=lambda region_name: fake_client(secrets),
    )


class RegistrySecretTest(unittest.TestCase):
    def test_creates_the_secret_with_the_keys_ecr_reads(self):
        secrets = FakeSecretsManager(value=None)
        with patch_clients(secrets=secrets):
            arn = asyncio.run(
                ecr_control.ensure_registry_secret(
                    SECRET_NAME,
                    REGION,
                    username="someone",
                    access_token="dckr_pat_new",
                )
            )
        self.assertEqual(arn, SECRET_ARN)
        self.assertEqual(len(secrets.created), 1)
        # The key names are ECR's: anything else is stored fine but rejected by the rule.
        self.assertEqual(
            json.loads(secrets.created[0]["SecretString"]),
            {"username": "someone", "accessToken": "dckr_pat_new"},
        )

    def test_unchanged_credentials_write_no_new_version(self):
        secrets = FakeSecretsManager(value=credential())
        with patch_clients(secrets=secrets):
            asyncio.run(
                ecr_control.ensure_registry_secret(
                    SECRET_NAME,
                    REGION,
                    username="someone",
                    access_token="dckr_pat_current",
                )
            )
        self.assertEqual(secrets.put, [])
        self.assertEqual(secrets.created, [])

    def test_rotated_credentials_write_a_new_version(self):
        secrets = FakeSecretsManager(value=credential())
        with patch_clients(secrets=secrets):
            asyncio.run(
                ecr_control.ensure_registry_secret(
                    SECRET_NAME,
                    REGION,
                    username="someone",
                    access_token="dckr_pat_rotated",
                )
            )
        self.assertEqual(len(secrets.put), 1)
        self.assertEqual(json.loads(secrets.put[0])["accessToken"], "dckr_pat_rotated")

    def test_no_credentials_keeps_the_existing_value(self):
        """A deployer with no token on disk adopts the secret rather than rewriting it."""
        secrets = FakeSecretsManager(value=credential())
        with patch_clients(secrets=secrets):
            arn = asyncio.run(ecr_control.ensure_registry_secret(SECRET_NAME, REGION))
        self.assertEqual(arn, SECRET_ARN)
        self.assertEqual(secrets.put, [])
        self.assertEqual(secrets.value, credential())

    def test_no_credentials_and_no_secret_is_an_error(self):
        secrets = FakeSecretsManager(value=None)
        with patch_clients(secrets=secrets), self.assertRaises(RuntimeError) as caught:
            asyncio.run(ecr_control.ensure_registry_secret(SECRET_NAME, REGION))
        self.assertIn("access_token", str(caught.exception))

    def test_half_a_credential_is_an_error(self):
        secrets = FakeSecretsManager(value=credential())
        with patch_clients(secrets=secrets), self.assertRaises(ValueError):
            asyncio.run(
                ecr_control.ensure_registry_secret(
                    SECRET_NAME,
                    REGION,
                    username="someone",
                )
            )

    def test_a_name_ecr_cannot_read_is_rejected_before_it_is_written(self):
        """Caught here rather than as an InvalidParameterException on the rule."""
        secrets = FakeSecretsManager(value=None)
        with patch_clients(secrets=secrets), self.assertRaises(ValueError) as caught:
            asyncio.run(
                ecr_control.ensure_registry_secret(
                    "swe/dockerhub",
                    REGION,
                    username="someone",
                    access_token="t",
                )
            )
        self.assertIn(ecr_control.SECRET_NAME_PREFIX, str(caught.exception))
        self.assertEqual(secrets.created, [])


class CacheRuleTest(unittest.TestCase):
    def test_creates_the_rule_for_docker_hub(self):
        ecr = FakeEcr()
        with patch_clients(ecr=ecr):
            asyncio.run(ecr_control.ensure_docker_hub_cache_rule(PREFIX, REGION, SECRET_ARN))
        self.assertEqual(
            ecr.created,
            [
                {
                    "ecrRepositoryPrefix": PREFIX,
                    "upstreamRegistry": ecr_control.DOCKER_HUB_UPSTREAM_REGISTRY,
                    "upstreamRegistryUrl": ecr_control.DOCKER_HUB_UPSTREAM_URL,
                    "credentialArn": SECRET_ARN,
                }
            ],
        )
        self.assertEqual(ecr.validated, [PREFIX])

    def test_an_unknown_prefix_reported_as_an_empty_list_is_still_a_create(self):
        ecr = FakeEcr(empty_describe=True)
        with patch_clients(ecr=ecr):
            asyncio.run(ecr_control.ensure_docker_hub_cache_rule(PREFIX, REGION, SECRET_ARN))
        self.assertEqual(len(ecr.created), 1)

    def test_an_existing_rule_on_the_same_secret_is_left_alone(self):
        ecr = FakeEcr(rules={PREFIX: existing_rule()})
        with patch_clients(ecr=ecr):
            asyncio.run(ecr_control.ensure_docker_hub_cache_rule(PREFIX, REGION, SECRET_ARN))
        self.assertEqual(ecr.created, [])
        self.assertEqual(ecr.updated, [])
        # Still validated: the rule is unchanged, the credential inside it may not be.
        self.assertEqual(ecr.validated, [PREFIX])

    def test_a_rule_on_another_secret_is_repointed(self):
        """Lets a rotated credential be a redeploy rather than a delete."""
        ecr = FakeEcr(rules={PREFIX: existing_rule(credential_arn=OTHER_ARN)})
        with patch_clients(ecr=ecr):
            asyncio.run(ecr_control.ensure_docker_hub_cache_rule(PREFIX, REGION, SECRET_ARN))
        self.assertEqual(ecr.created, [])
        self.assertEqual(ecr.updated, [{"ecrRepositoryPrefix": PREFIX, "credentialArn": SECRET_ARN}])

    def test_a_rule_for_another_upstream_is_an_error(self):
        ecr = FakeEcr(rules={PREFIX: existing_rule(upstream="ghcr.io")})
        with patch_clients(ecr=ecr), self.assertRaises(RuntimeError) as caught:
            asyncio.run(ecr_control.ensure_docker_hub_cache_rule(PREFIX, REGION, SECRET_ARN))
        self.assertIn("ghcr.io", str(caught.exception))
        self.assertEqual(ecr.updated, [])

    def test_a_rule_that_does_not_validate_fails_the_deploy(self):
        """A stale token, caught here instead of as a missing image mid-rollout."""
        ecr = FakeEcr(rules={PREFIX: existing_rule()}, is_valid=False)
        with patch_clients(ecr=ecr), self.assertRaises(RuntimeError) as caught:
            asyncio.run(ecr_control.ensure_docker_hub_cache_rule(PREFIX, REGION, SECRET_ARN))
        self.assertIn("unauthorized", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
