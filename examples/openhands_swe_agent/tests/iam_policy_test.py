#!/usr/bin/env python
"""Unit tests for the execution role policy and the config it is scoped from.

One property in two halves: a rollout container can read the agent repository and the
pull-through cache namespace, and nothing else. No AWS.
"""

import unittest

# The recipe directory is on ``sys.path`` via its ``conftest.py``.
import iam_policy
from config import agent_repository, cache_prefix, task_image_namespace

ACCOUNT = "123456789012"
REGION = "us-west-2"
CONFIG = {
    "agentcore": {
        "region": REGION,
        "docker_repo": f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/agents/swe",
    },
    "docker_hub": {"cache_prefix": "docker-hub"},
}


def statement(policy: dict, sid: str) -> dict:
    return next(s for s in policy["Statement"] if s["Sid"] == sid)


def as_list(value) -> list:
    """A policy element that may be a string or a list of them, as a list."""
    return value if isinstance(value, list) else [value]


class NamespaceTest(unittest.TestCase):
    def test_reads_the_agent_repository(self):
        agent = agent_repository(CONFIG)
        self.assertEqual(
            (agent.account_id, agent.region, agent.path),
            (ACCOUNT, REGION, "agents/swe"),
        )

    def test_the_task_image_namespace_is_assembled_from_the_parts(self):
        self.assertEqual(cache_prefix(CONFIG), "docker-hub")
        self.assertEqual(
            task_image_namespace(CONFIG),
            f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/docker-hub",
        )

    def test_a_tag_left_on_the_repo_is_rejected(self):
        """The deploy appends ``:<tag>``, so one here would be part of the name."""
        config = {"agentcore": {"docker_repo": f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/agents:r70"}}
        with self.assertRaises(ValueError):
            agent_repository(config)

    def test_a_namespace_that_is_not_ecr_is_rejected(self):
        for repo in (
            "docker.io/library",
            f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com",
            "user",
        ):
            with self.subTest(docker_repo=repo), self.assertRaises(ValueError):
                agent_repository({"agentcore": {"docker_repo": repo}})


class PermissionsPolicyTest(unittest.TestCase):
    def setUp(self):
        # Scoped off the config exactly as the deploy scopes it.
        self.policy = iam_policy.permissions_policy(
            ACCOUNT, REGION, cache_prefix(CONFIG), agent_repository(CONFIG).path
        )

    def test_reads_only_the_agent_image_and_the_cache(self):
        # Region wildcarded (caches are per region), repository not.
        self.assertEqual(
            statement(self.policy, "ECRImageAccess")["Resource"],
            [
                f"arn:aws:ecr:*:{ACCOUNT}:repository/agents/swe",
                f"arn:aws:ecr:*:{ACCOUNT}:repository/docker-hub/*",
            ],
        )

    def test_no_ecr_grant_reaches_a_third_repository(self):
        # Stated over every statement so a newly added ECR grant is covered too.
        # ``GetAuthorizationToken`` is the documented exception: no resource but ``*``.
        allowed = {
            f"arn:aws:ecr:*:{ACCOUNT}:repository/agents/swe",
            f"arn:aws:ecr:*:{ACCOUNT}:repository/docker-hub/*",
        }
        for statement_ in self.policy["Statement"]:
            actions = as_list(statement_["Action"])
            if statement_["Sid"] == "ECRTokenAccess" or not any(a.startswith("ecr:") for a in actions):
                continue
            self.assertLessEqual(set(as_list(statement_["Resource"])), allowed, statement_["Sid"])

    def test_the_namespace_the_eval_pulls_from_is_one_the_role_may_read(self):
        """The two halves joined: what evaluate.py points a rollout at is granted."""
        registry, _, path = task_image_namespace(CONFIG).partition("/")
        self.assertTrue(registry.startswith(f"{ACCOUNT}."), registry)
        self.assertIn(
            f"arn:aws:ecr:*:{ACCOUNT}:repository/{path}/*",
            statement(self.policy, "ECRImageAccess")["Resource"],
        )

    def test_populating_the_cache_stays_inside_the_cache(self):
        """``CreateRepository`` is a cache grant, not a registry-wide one."""
        self.assertEqual(
            statement(self.policy, "ECRPullThroughCache")["Resource"],
            [f"arn:aws:ecr:*:{ACCOUNT}:repository/docker-hub/*"],
        )


if __name__ == "__main__":
    unittest.main()
