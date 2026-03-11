from pydantic import BaseModel


class InvocationRequest(BaseModel):
    task_uri: str  # S3 URI to task config JSON (e.g. s3://bucket/officebench/1-1/config.json)
    testbed_uri: str | None = None  # S3 URI to testbed tar.gz (if task has data files)
