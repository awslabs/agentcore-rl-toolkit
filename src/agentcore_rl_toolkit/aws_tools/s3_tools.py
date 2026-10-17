"""S3 helpers taking ``s3://`` URIs."""

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_async_session


async def upload_object(s3_uri: str, region_name: str, data: bytes):
    path = s3_uri.split("s3://")[1]
    bucket, key = path.split("/")[0], "/".join(path.split("/")[1:])
    session = await get_async_session()
    async with session.create_client("s3", region_name=region_name) as s3:
        await s3.put_object(Bucket=bucket, Key=key, Body=data)
