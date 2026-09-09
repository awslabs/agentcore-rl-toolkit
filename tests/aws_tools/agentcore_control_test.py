#!/usr/bin/env python
"""Unit tests for the capacity provider's create payload, validated against
botocore's model of ``CreateCapacityProvider``. Fake client, no AWS.
"""

import asyncio
import unittest
from contextlib import asynccontextmanager
from unittest import mock

import botocore.session
from botocore.validate import ParamValidator

from agentcore_rl_toolkit.aws_tools import agentcore_control

MODULE = "agentcore_rl_toolkit.aws_tools.agentcore_control"
REGION = "us-west-2"
OPERATOR_ROLE_ARN = "arn:aws:iam::123456789012:role/Operator"
INSTANCE_PROFILE_ARN = "arn:aws:iam::123456789012:instance-profile/Instances"

CREATE_INPUT = (
    botocore.session.get_session()
    .get_service_model("bedrock-agentcore-control")
    .operation_model("CreateCapacityProvider")
    .input_shape
)


class FakeControlPlane:
    def __init__(self):
        self.params: dict = {}

    async def create_capacity_provider(self, **kwargs):
        self.params = kwargs
        return {"capacityProviderId": "cp-1", "capacityProviderArn": "arn:aws:bedrock-agentcore:cp"}


async def _ready(describe, what):
    """``_wait_ready`` without the polling."""
    return {"status": "READY"}


def create(**kwargs) -> dict:
    """Run ``create_capacity_provider`` against a fake and return the sent params."""
    fake = FakeControlPlane()

    @asynccontextmanager
    async def client(region_name):
        yield fake

    with mock.patch.multiple(MODULE, _control_client=client, _wait_ready=_ready):
        asyncio.run(
            agentcore_control.create_capacity_provider(
                region_name=REGION,
                name="pool",
                operator_role_arn=OPERATOR_ROLE_ARN,
                instance_profile_arn=INSTANCE_PROFILE_ARN,
                subnets=["subnet-1"],
                security_groups=["sg-1"],
                **kwargs,
            )
        )
    return fake.params


def launch_parameters(params: dict) -> dict:
    return params["computeConfiguration"]["ec2Configuration"]["launchTemplateSource"]["launchParameters"]


class CapacityProviderPayloadTest(unittest.TestCase):
    def assert_valid(self, params: dict) -> None:
        report = ParamValidator().validate(params, CREATE_INPUT)
        self.assertFalse(report.has_errors(), report.generate_report())

    def test_no_ssh_key_omits_the_field(self):
        params = create()
        self.assertNotIn("sshKeyName", launch_parameters(params))
        self.assert_valid(params)

    def test_a_blank_ssh_key_is_omitted_too(self):
        params = create(ssh_key_name="")
        self.assertNotIn("sshKeyName", launch_parameters(params))
        self.assert_valid(params)

    def test_an_ssh_key_is_passed_through(self):
        params = create(ssh_key_name="some-key")
        self.assertEqual(launch_parameters(params)["sshKeyName"], "some-key")
        self.assert_valid(params)

    def test_the_root_volume_keeps_iops_at_four_times_throughput(self):
        """4x throughput is EC2's floor for gp3."""
        root = create(root_throughput=750)["computeConfiguration"]["ec2Configuration"]["rootVolume"]
        self.assertEqual((root["throughput"], root["iops"]), (750, 3000))

    def test_the_instance_type_is_the_pool_shape_asked_for(self):
        params = create(instance_type="c5.xlarge")
        self.assertEqual(launch_parameters(params)["instanceRequirements"]["allowedInstanceTypes"], ["c5.xlarge"])


if __name__ == "__main__":
    unittest.main()
