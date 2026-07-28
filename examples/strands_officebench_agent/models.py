from pydantic import BaseModel


class InvocationRequest(BaseModel):
    task_uri: str  # S3 URI to task config JSON (e.g. s3://bucket/officebench/1-1/config.json)
    testbed_uri: str | None = None  # S3 URI to testbed tar.gz (if task has data files)


class EvaluationCheck(BaseModel):
    # Each check names an evaluation function and its args. `function` is looked up
    # against an allowlist (EVAL_FUNCTIONS in reward.py) at scoring time, so an
    # unknown name scores 0.0 rather than executing anything.
    function: str
    args: dict = {}


class TaskConfig(BaseModel):
    # Schema for the task config JSON loaded from the (untrusted) task_uri S3 object.
    # `task` is the natural-language instruction handed to an agent that holds
    # shell + ALL_TOOLS, so we require it to be a string here — this guarantees a
    # non-string/content-block value can never reach the agent, and that required
    # fields are present before the rollout runs. Extra keys are ignored.
    #
    # Trust boundary: validating the shape does NOT make the *content* of `task`
    # safe — it is free-form text driving a powerful agent. Task URIs must come
    # from a trusted bucket you control; do not point this agent at task configs
    # from untrusted principals.
    task: str
    evaluation: list[EvaluationCheck] = []
    username: str | None = None
    date: str | None = None
    weekday: str | None = None
    time: str | None = None
