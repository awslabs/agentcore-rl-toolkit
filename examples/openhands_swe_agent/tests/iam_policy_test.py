#!/usr/bin/env python
"""Unit tests for the execution role's document and the config it is scoped from.

Both halves of one property: a rollout container can read the *two* repositories
this recipe uses and nothing else. The narrowing is the part worth a regression
guard, because widening it back to ``repository/*`` breaks nothing and would never
be noticed -- whereas narrowing it too far is noticed immediately, as an AgentCore
session that cannot pull the agent image (see the module docstring of
``iam_policy.py`` for why that grant has to stay).

The config half is here because the scoping is only as good as the parse: the
repository names come out of ``docker_repo`` and ``docker_hub.cache_prefix``, and a
setting this cannot read is one that would otherwise be granted as-is. It also
covers the other direction -- that the namespace an eval pulls task images from is
assembled out of the same two settings, so it cannot name a cache the grant misses.

No AWS.
"""

import unittest

# The recipe's modules under the same names its own scripts use -- the recipe
# directory is on ``sys.path`` via its ``conftest.py``.
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
        """The account off ``docker_repo``, the region off ``[agentcore]``, the prefix."""
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
            "vdanylo",
        ):
            with self.subTest(docker_repo=repo), self.assertRaises(ValueError):
                agent_repository({"agentcore": {"docker_repo": repo}})

    def test_a_cache_prefix_that_is_a_whole_namespace_is_rejected(self):
        """The setting it replaced was a URI, so pasting one back has to be an error."""
        for prefix in (
            f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/docker-hub",
            "docker-hub:latest",
            "",
        ):
            with self.subTest(cache_prefix=prefix), self.assertRaises(ValueError):
                cache_prefix({"docker_hub": {"cache_prefix": prefix}})


class PermissionsPolicyTest(unittest.TestCase):
    def setUp(self):
        # Scoped off the config exactly as the deploy scopes it.
        self.policy = iam_policy.permissions_policy(
            ACCOUNT, REGION, cache_prefix(CONFIG), agent_repository(CONFIG).path
        )

    def test_reads_only_the_agent_image_and_the_cache(self):
        # Region wildcarded, repository not: the caches are per region, the
        # repositories are the two this recipe has.
        self.assertEqual(
            statement(self.policy, "ECRImageAccess")["Resource"],
            [
                f"arn:aws:ecr:*:{ACCOUNT}:repository/agents/swe",
                f"arn:aws:ecr:*:{ACCOUNT}:repository/docker-hub/*",
            ],
        )

    def test_no_ecr_grant_reaches_a_third_repository(self):
        """The guard against quietly widening any ECR grant back to the registry.

        Stated over every statement rather than over the two above, so that a new
        ECR grant is covered by it the moment someone adds one. ``GetAuthorizationToken``
        is the documented exception: it takes no resource but ``*``.
        """
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
        """The two halves, joined: what evaluate.py points a rollout at is granted.

        Now that the namespace is assembled rather than stated, this is what stands in
        for the pasted string agreeing with the grant -- both come off ``docker_repo``
        and ``cache_prefix``, so the only way they can disagree is a change to one of
        the two derivations.
        """
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
