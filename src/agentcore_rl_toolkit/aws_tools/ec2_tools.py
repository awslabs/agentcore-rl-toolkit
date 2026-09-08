"""Reading this host's identity from IMDS, and AgentCore's instances from the EC2 API.

Both halves are needed on the rollout path -- the trainer stamps its instance type
onto the run, and the EC2 monitor maps running instances back to their sessions --
so this module reads no config of its own and shells out to nothing: no config
file, no environment, no ssh, no subprocesses. Anything that needs a shell on a
cluster host belongs beside its caller instead, not here.
"""

import logging
import urllib.request

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session, tags_to_map

logger = logging.getLogger(__name__)

_IMDS_TOKEN_URL = "http://169.254.169.254/latest/api/token"
_IMDS_META_URL = "http://169.254.169.254/latest/meta-data"


def _imds(path: str, timeout: float = 2.0) -> str:
    """Fetch a metadata value via IMDSv2. Raises on failure."""
    token = (
        urllib.request.urlopen(
            urllib.request.Request(
                _IMDS_TOKEN_URL,
                method="PUT",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            ),
            timeout=timeout,
        )
        .read()
        .decode()
    )

    return (
        urllib.request.urlopen(
            urllib.request.Request(
                f"{_IMDS_META_URL}/{path}",
                headers={"X-aws-ec2-metadata-token": token},
            ),
            timeout=timeout,
        )
        .read()
        .decode()
        .strip()
    )


def get_current_instance_type(default: str = "unknown") -> str:
    """Return this host's EC2 instance type via the instance metadata service.

    The instance type is read directly from IMDSv2, which requires no IAM
    permissions. (The EC2 ``DescribeInstances`` API is intentionally avoided:
    the HyperPod execution role is not authorized to call it.) Returns
    ``default`` if the host is not on EC2 or the lookup fails, so callers never
    need to handle exceptions.
    """
    try:
        return _imds("instance-type") or default
    except Exception as e:
        logger.warning("Failed to retrieve EC2 instance type from IMDS: %s", e)
        return default


def instance_to_session(i):
    return tags_to_map(i["Tags"]).get("bedrock-agentcore:runtime-session-id")


async def find_running_instances(capacity_provider_arn: str, session_prefix: str | None = None):
    region_name = capacity_provider_arn.split(":")[3]
    session = await get_aioboto3_session()
    async with session.client("ec2", region_name=region_name) as ec2:  # type: ignore
        paginator = ec2.get_paginator("describe_instances")
        pages = paginator.paginate(
            Filters=[
                {"Name": "tag:aws:ec2:managed-launch", "Values": ["agentcore-runtime-instance"]},
                {
                    "Name": "tag:bedrock-agentcore:capacity-provider-id",
                    "Values": [capacity_provider_arn.split("/")[-1]],
                },
                {"Name": "instance-state-name", "Values": ["running"]},
            ]
        )
        instances = [
            i
            async for page in pages
            for r in page["Reservations"]
            for i in r["Instances"]
            if session_prefix is None
            or tags_to_map(i["Tags"]).get("bedrock-agentcore:runtime-session-id", "").startswith(session_prefix)
        ]
    return instances
