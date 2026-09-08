import datetime as dt
from decimal import Decimal, Inexact, Rounded, localcontext
from typing import Any, AsyncIterator, Mapping

from boto3.dynamodb.conditions import Key
from boto3.dynamodb.types import DYNAMODB_CONTEXT
from pydantic import BaseModel

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session
from agentcore_rl_toolkit.aws_tools.dynamodb_control import (
    EXPERIMENT_INDEX,
    EXPERIMENT_PARTITION_KEY,
    EXPERIMENT_SORT_KEY,
    SESSION_KEY,
)


def to_dynamodb(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return to_dynamodb(value.model_dump())

    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()

    if isinstance(value, dict):
        return {k: to_dynamodb(v) for k, v in value.items()}

    if isinstance(value, float):
        with localcontext(DYNAMODB_CONTEXT) as ctx:
            ctx.traps[Inexact] = False
            ctx.traps[Rounded] = False
            return ctx.create_decimal(Decimal(value))

    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_dynamodb(v) for v in value]

    return value


def from_dynamodb(value: Any) -> Any:
    """:func:`to_dynamodb` undone for numbers: ``Decimal`` back to ``int``/``float``.

    DynamoDB has one numeric type, and boto3 hands it back as a ``Decimal`` -- which
    is neither an ``int`` nor a ``float``. So any reduction that picks its fields by
    ``isinstance(value, (int, float))``, which is how both the eval report and the
    trainer's metric mixin decide what is summarizable, sees *no* numbers at all in a
    record read back out of the table, and silently reports nothing. Integral values
    come back as ``int`` and the rest as ``float``, so a rollout record loaded from
    DynamoDB reduces exactly as the in-memory one it was written from did.
    """
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)

    if isinstance(value, dict):
        return {k: from_dynamodb(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [from_dynamodb(v) for v in value]

    return value


async def update_dict(
    table,
    key: dict[str, Any],
    updates: Mapping[str, Any],
    **kwargs,
):
    if not updates:
        raise ValueError("updates must not be empty")

    names = {}
    values = {}
    assignments = []

    updates = to_dynamodb(updates)

    for i, (field, value) in enumerate(updates.items()):
        # exclude key updates
        if field in key:
            continue

        name = f"#f{i}"
        value_name = f":v{i}"

        names[name] = field
        values[value_name] = value
        assignments.append(f"{name} = {value_name}")

    return await table.update_item(
        Key=key,
        UpdateExpression="SET " + ", ".join(assignments),
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=values,
        **kwargs,
    )


async def paginate_table(
    operation,
    **kwargs,
) -> AsyncIterator[dict[str, Any]]:
    while True:
        response = await operation(**kwargs)
        yield response

        last_key = response.get("LastEvaluatedKey")
        if last_key is None:
            break

        kwargs["ExclusiveStartKey"] = last_key


async def get_sessions_for_experiment_run(
    experiment_name: str,
    experiment_start_at: str = "",
    *,
    table_name: str,
    region_name: str | None = None,
) -> list[dict]:
    """Every session item of one experiment, through the ``experiment_sessions`` index.

    ``experiment_start_at`` narrows the query to a single *run* of the experiment: the
    sort key is ``"<experiment_start_at>:<session_id>"``, so a run is a
    ``begins_with`` prefix on it rather than a scan. Empty means every run the name
    ever had. ``region_name`` of ``None`` takes the session's default region.
    """
    sessions = []

    key_condition = Key(EXPERIMENT_PARTITION_KEY).eq(experiment_name)
    if experiment_start_at is not None and len(experiment_start_at) > 0:
        key_condition &= Key(EXPERIMENT_SORT_KEY).begins_with(experiment_start_at)

    async with (await get_aioboto3_session()).resource("dynamodb", region_name=region_name) as ddb:
        table = await ddb.Table(table_name)
        pages = paginate_table(
            table.query,
            IndexName=EXPERIMENT_INDEX,
            KeyConditionExpression=key_condition,
        )
        async for page in pages:
            sessions.extend(page["Items"])
    return sessions


async def get_session(
    session_id: str,
    *,
    table_name: str,
    region_name: str | None = None,
) -> dict:
    async with (await get_aioboto3_session()).resource("dynamodb", region_name=region_name) as ddb:
        table = await ddb.Table(table_name)
        query = await table.query(KeyConditionExpression=Key(SESSION_KEY).eq(session_id))
        return query["Items"][0]
