"""DynamoDB value conversion and pagination at the async client boundary."""

import datetime as dt
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentcore_rl_toolkit.aws_tools import dynamodb_tools


@pytest.mark.asyncio
async def test_update_converts_values_in_both_directions():
    client = AsyncMock()
    client.update_item.return_value = {"Attributes": {"reward": {"N": "0.5"}}}
    response = await dynamodb_tools.update_dict(
        client,
        "sessions",
        {"session_id": "s1"},
        {"session_id": "ignored", "reward": 0.5, "details": {"date": dt.date(2026, 10, 17), "done": True}},
        ReturnValues="ALL_NEW",
    )
    sent = client.update_item.await_args.kwargs
    assert sent["TableName"] == "sessions"
    assert sent["Key"] == {"session_id": {"S": "s1"}}
    assignments = (part.strip().split(" = ") for part in sent["UpdateExpression"].removeprefix("SET ").split(","))
    written = {
        sent["ExpressionAttributeNames"][name]: sent["ExpressionAttributeValues"][value] for name, value in assignments
    }
    assert written == {
        "reward": {"N": "0.5"},
        "details": {"M": {"date": {"S": "2026-10-17"}, "done": {"BOOL": True}}},
    }
    assert response["Attributes"] == {"reward": Decimal("0.5")}
    assert isinstance(response["Attributes"]["reward"], Decimal)


@pytest.mark.asyncio
async def test_experiment_query_reads_all_pages_and_decodes_items(monkeypatch):
    cursor = {"session_id": {"S": "s1"}}
    client = AsyncMock()
    client.query.side_effect = [
        {"Items": [{"session_id": {"S": "s1"}, "reward": {"N": "0.5"}}], "LastEvaluatedKey": cursor},
        {"Items": [{"session_id": {"S": "s2"}, "reward": {"N": "1"}}], "LastEvaluatedKey": {}},
    ]
    session = MagicMock()
    session.create_client.return_value.__aenter__.return_value = client
    monkeypatch.setattr(dynamodb_tools, "get_async_session", AsyncMock(return_value=session))

    result = await dynamodb_tools.get_sessions_for_experiment_run(
        "experiment-1", "2026-10", table_name="sessions", region_name="us-west-2"
    )

    first, second = (call.kwargs for call in client.query.await_args_list)
    assert "ExclusiveStartKey" not in first
    assert second == {**first, "ExclusiveStartKey": cursor}
    assert first["TableName"] == "sessions"
    assert first["IndexName"] == dynamodb_tools.EXPERIMENT_INDEX
    assert "begins_with" in first["KeyConditionExpression"]
    assert list(first["ExpressionAttributeValues"].values()) == [{"S": "experiment-1"}, {"S": "2026-10"}]
    assert result == [
        {"session_id": "s1", "reward": Decimal("0.5")},
        {"session_id": "s2", "reward": Decimal("1")},
    ]
    assert all(isinstance(item["reward"], Decimal) for item in result)
