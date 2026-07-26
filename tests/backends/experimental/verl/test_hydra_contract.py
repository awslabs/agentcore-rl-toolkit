"""Prove the hydra-instantiation ctor contract: verl's AgentLoopWorker calls
``hydra.utils.instantiate`` on the YAML registry entry, passing the entry's
extra keys (including ``name``) as kwargs plus its ``DictConfigWrap``/
``ToolListWrap`` wrappers — all of which the constructor must swallow."""

from unittest.mock import patch

import hydra
from omegaconf import DictConfig, OmegaConf
from verl.experimental.agent_loop.agent_loop import ToolListWrap

from .conftest import FakeLLMServerClient, FakeTokenizer, make_data_config, make_trainer_config


def test_instantiate_from_yaml_entry():
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
            data_config=make_data_config(),
            tools=ToolListWrap([]),
        )

    assert isinstance(loop, AgentCoreAgentLoop)
    assert loop.max_rollout_time == 60
    # AgentLoopBase unwrapped the DictConfigWrap wrappers
    assert loop.config.trainer.use_v1 is True
    assert isinstance(loop.data_config, DictConfig)
