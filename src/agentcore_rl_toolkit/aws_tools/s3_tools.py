"""S3 helpers taking ``s3://`` URIs."""

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session


async def upload_object(s3_uri: str, region_name: str, data: bytes):
    path = s3_uri.split("s3://")[1]
    bucket, key = path.split("/")[0], "/".join(path.split("/")[1:])
    session = await get_aioboto3_session()
    async with session.resource("s3", region_name=region_name) as s3:
        s3_object = await s3.Object(bucket, key)
        await s3_object.put(Body=data)
