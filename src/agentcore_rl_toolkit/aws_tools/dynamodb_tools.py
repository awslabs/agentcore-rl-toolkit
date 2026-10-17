"""DynamoDB data plane: value conversion, item updates, and the session queries.

The table and its ``experiment_sessions`` index are defined in :mod:`.dynamodb_control`.
"""

import datetime as dt
from decimal import Decimal, Inexact, Rounded, localcontext
from typing import Any, AsyncIterator, Mapping

from boto3.dynamodb.types import DYNAMODB_CONTEXT, TypeDeserializer, TypeSerializer
from pydantic import BaseModel

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_async_session
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

    boto3 returns every number as a ``Decimal``, which is neither an ``int`` nor a
    ``float``, so consumers that select fields by ``isinstance(value, (int, float))``
    silently see no numbers at all in a record read back out of the table.
    """
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)

    if isinstance(value, dict):
        return {k: from_dynamodb(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [from_dynamodb(v) for v in value]

    return value


async def update_dict(
    client,
    table_name: str,
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

    response = await client.update_item(
        TableName=table_name,
        Key=_serialize_item(key),
        UpdateExpression="SET " + ", ".join(assignments),
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=_serialize_item(values),
        **kwargs,
    )
    if "Attributes" in response:
        response["Attributes"] = _deserialize_item(response["Attributes"])
    return response


def _serialize_item(item: Mapping[str, Any]) -> dict:
    serializer = TypeSerializer()
    return {key: serializer.serialize(to_dynamodb(value)) for key, value in item.items()}


def _deserialize_item(item: dict) -> dict:
    deserializer = TypeDeserializer()
    return {key: deserializer.deserialize(value) for key, value in item.items()}


async def paginate_table(
    operation,
    **kwargs,
) -> AsyncIterator[dict[str, Any]]:
    while True:
        response = await operation(**kwargs)
        yield response

        last_key = response.get("LastEvaluatedKey")
        if not last_key:
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

    ``experiment_start_at`` is a ``begins_with`` prefix on the sort key that narrows the
    query to a single run; empty means every run of that name.
    """
    sessions = []

    key_condition = "#experiment = :experiment"
    names = {"#experiment": EXPERIMENT_PARTITION_KEY}
    values = {":experiment": experiment_name}
    if experiment_start_at:
        key_condition += " AND begins_with(#started, :started)"
        names["#started"] = EXPERIMENT_SORT_KEY
        values[":started"] = experiment_start_at

    async with (await get_async_session()).create_client("dynamodb", region_name=region_name) as ddb:
        pages = paginate_table(
            ddb.query,
            TableName=table_name,
            IndexName=EXPERIMENT_INDEX,
            KeyConditionExpression=key_condition,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=_serialize_item(values),
        )
        async for page in pages:
            sessions.extend(_deserialize_item(item) for item in page["Items"])
    return sessions


async def get_session(
    session_id: str,
    *,
    table_name: str,
    region_name: str | None = None,
) -> dict:
    async with (await get_async_session()).create_client("dynamodb", region_name=region_name) as ddb:
        query = await ddb.query(
            TableName=table_name,
            KeyConditionExpression="#session = :session",
            ExpressionAttributeNames={"#session": SESSION_KEY},
            ExpressionAttributeValues=_serialize_item({":session": session_id}),
        )
        return _deserialize_item(query["Items"][0])
