#!/usr/bin/env python
"""Tests for the config -> rollout session factory: backends resolved by import path."""

import unittest

from agentcore_rl_toolkit.aws_tools.persistent_dict import NullPersister, PersistentDict
from agentcore_rl_toolkit.rollout_session.agentcore_a2a_session import AgentCoreA2ASession
from agentcore_rl_toolkit.rollout_session.docker_a2a_session import DockerA2ASession
from agentcore_rl_toolkit.rollout_session.factory import make_session, require, resolve_backend

SESSION_ID = "verl_" + "0" * 32
RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/agent-abc"
CAPACITY_PROVIDER_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:capacity-provider/pool-abc"


def state() -> PersistentDict:
    return PersistentDict(persister=NullPersister())


class OutOfTreeSession:
    """A backend the factory has never heard of: nothing registers it."""

    def __init__(self, session_id, session_state, *, endpoint):
        self.session_id = session_id
        self.session_state = session_state
        self.endpoint = endpoint

    @classmethod
    def from_config(cls, session_id, cfg, session_state):
        (endpoint,) = require(cfg, cls, "endpoint")
        return cls(session_id, session_state, endpoint=endpoint)


class NotABackend:
    pass


HERE = __name__


class ResolveTest(unittest.TestCase):
    def test_a_class_anywhere_is_a_backend_by_its_import_path(self):
        meta = state()
        session = make_session(SESSION_ID, {"backend": f"{HERE}.OutOfTreeSession", "endpoint": "http://h:1"}, meta)
        self.assertIsInstance(session, OutOfTreeSession)
        self.assertEqual((session.session_id, session.endpoint), (SESSION_ID, "http://h:1"))
        self.assertIs(session.session_state, meta)

    def test_other_backends_keys_are_ignored(self):
        cfg = {"backend": f"{HERE}.OutOfTreeSession", "endpoint": "http://h:1", "agent_image_uri": "img"}
        self.assertEqual(make_session(SESSION_ID, cfg, state()).endpoint, "http://h:1")

    def test_a_missing_key_names_the_backend_and_the_key(self):
        with self.assertRaises(ValueError) as caught:
            make_session(SESSION_ID, {"backend": f"{HERE}.OutOfTreeSession", "endpoint": None}, state())
        self.assertIn("OutOfTreeSession", str(caught.exception))
        self.assertIn("endpoint", str(caught.exception))

    def test_a_bare_name_asks_for_an_import_path(self):
        with self.assertRaises(ValueError) as caught:
            resolve_backend("agentcore_a2a")
        self.assertIn("import path", str(caught.exception))
        self.assertIn("AgentCoreA2ASession", str(caught.exception))

    def test_unresolvable_paths_say_what_is_missing(self):
        for path, expected in (
            ("no_such_package.Session", "cannot import 'no_such_package'"),
            (f"{HERE}.NoSuchSession", "has no 'NoSuchSession'"),
            (f"{HERE}.NotABackend", "from_config"),
        ):
            with self.subTest(path=path), self.assertRaises(ValueError) as caught:
                resolve_backend(path)
            self.assertIn(expected, str(caught.exception))


class A2AFromConfigTest(unittest.TestCase):
    def test_agentcore_reads_its_keys(self):
        cfg = {
            "backend": "agentcore_rl_toolkit.rollout_session.agentcore_a2a_session.AgentCoreA2ASession",
            "agentcore_runtime_arn": RUNTIME_ARN,
            "capacity_provider_arn": CAPACITY_PROVIDER_ARN,
        }
        session = make_session(SESSION_ID, cfg, state())
        self.assertIsInstance(session, AgentCoreA2ASession)
        self.assertEqual((session.runtime_arn, session.capacity_provider_arn), (RUNTIME_ARN, CAPACITY_PROVIDER_ARN))

    def test_agentcore_needs_a_capacity_provider(self):
        with self.assertRaises(ValueError) as caught:
            AgentCoreA2ASession.from_config(SESSION_ID, {"agentcore_runtime_arn": RUNTIME_ARN}, state())
        self.assertIn("capacity_provider_arn", str(caught.exception))

    def test_docker_reads_its_keys(self):
        cfg = {
            "backend": "agentcore_rl_toolkit.rollout_session.docker_a2a_session.DockerA2ASession",
            "agent_image_uri": "repo:tag",
            "docker_iam_role_arn": "arn:aws:iam::123456789012:role/agent",
            "docker_log_group": "/agents",
            "docker_log_region": "us-west-2",
        }
        session = make_session(SESSION_ID, cfg, state())
        self.assertIsInstance(session, DockerA2ASession)
        self.assertEqual(
            (session.agent_image_uri, session.iam_role_arn, session.log_group, session.log_region),
            ("repo:tag", "arn:aws:iam::123456789012:role/agent", "/agents", "us-west-2"),
        )


if __name__ == "__main__":
    unittest.main()
