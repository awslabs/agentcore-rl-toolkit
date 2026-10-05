"""The EC2 monitor follows the backend config's capacity provider, whatever the backend is."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agentcore_rl_toolkit.backends.experimental.verl import rollout_session_resources as resources

pytestmark = pytest.mark.asyncio

CAPACITY_PROVIDER_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:capacity-provider/pool-abc"


def config(**backend):
    return SimpleNamespace(
        rollout_session_backend={"backend": "some.module.Session", **backend},
        ec2_monitor_poll_interval=60,
        dynamodb_table="sessions",
        aws_region="us-west-2",
    )


async def test_a_backend_with_a_capacity_provider_is_monitored():
    with patch.object(resources, "start_ec2_monitor", AsyncMock(return_value="monitor")) as start:
        assert await resources._start_ec2_monitor(config(capacity_provider_arn=CAPACITY_PROVIDER_ARN)) == "monitor"
    start.assert_awaited_once_with(CAPACITY_PROVIDER_ARN, "sessions", poll_interval=60, region_name="us-west-2")


async def test_a_backend_without_one_is_not():
    with patch.object(resources, "start_ec2_monitor", AsyncMock()) as start:
        assert await resources._start_ec2_monitor(config()) is None
    start.assert_not_awaited()
