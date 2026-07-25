"""Prove the hydra-instantiation ctor contract: the YAML entry's extra keys
(including `name`) are passed as kwargs and must be swallowed, exactly as
verl's AgentLoopWorker does via hydra.utils.instantiate."""

from unittest.mock import MagicMock, patch

import pytest

from .conftest import FakeLLMServerClient, FakeTokenizer, make_trainer_config

hydra = pytest.importorskip("hydra")


def test_instantiate_from_yaml_entry():
    from omegaconf import OmegaConf

    from agentcore_rl_toolkit.backends.experimental.verl.agent_loop import AgentCoreAgentLoop

    entry = OmegaConf.create(
        {
            "name": "agentcore_agent",
            "_target_": "agentcore_rl_toolkit.backends.experimental.verl.agent_loop.AgentCoreAgentLoop",
            "agent_runtime_arn": "arn:aws:bedrock-agentcore:us-west-2:123:runtime/test",
            "s3_bucket": "test-bucket",
            "max_rollout_time": 60,
            "gateway_bind_host": "127.0.0.1",
            "gateway_public_host": "127.0.0.1",
        }
    )

    with patch("agentcore_rl_toolkit.backends.experimental.verl.agent_loop.RolloutClient"):
        loop = hydra.utils.instantiate(
            config=entry,
            trainer_config=make_trainer_config(),
            server_manager=FakeLLMServerClient(),
            tokenizer=FakeTokenizer(),
            processor=None,
            dataset_cls=None,
            data_config=MagicMock(config={}),
            tools=MagicMock(),  # verl passes ToolListWrap
        )

    assert isinstance(loop, AgentCoreAgentLoop)
    assert loop.max_rollout_time == 60
