#!/usr/bin/env python
"""Unit tests for the rollout session table's control plane.

The table is the one resource in a deploy that *accumulates*: every rollout this
recipe has ever run is an item in it, so what these tests are about is which deploys
touch it. A fresh account gets the whole table and its index in one call, a deploy
against the table that is already right writes nothing, and a table that predates the
index gains it -- while a table keyed differently, or an index keyed differently, is
an error rather than a create call that DynamoDB would reject or, worse, a silent
mismatch the analysis queries fail on much later.

The create payload is validated against botocore's own model of ``CreateTable``,
which is a stronger statement than any assertion about our own dict; the clients are
fakes and the model is read from the local service definition, so: no AWS.
"""

import asyncio
import unittest
from contextlib import asynccontextmanager
from unittest import mock

import botocore.exceptions
import botocore.session
from botocore.validate import ParamValidator

from agentcore_rl_toolkit.aws_tools import dynamodb_control

MODULE = "agentcore_rl_toolkit.aws_tools.dynamodb_control"
REGION = "us-west-2"
TABLE = "swe_agent_sessions"
TABLE_ARN = f"arn:aws:dynamodb:{REGION}:123456789012:table/{TABLE}"

DYNAMODB = botocore.session.get_session().get_service_model("dynamodb")
CREATE_INPUT = DYNAMODB.operation_model("CreateTable").input_shape
UPDATE_INPUT = DYNAMODB.operation_model("UpdateTable").input_shape


def table(key_schema=None, indexes=None, billing_mode="PAY_PER_REQUEST", status="ACTIVE") -> dict:
    """A ``describe_table`` payload, as much of one as the code under test reads."""
    described = {
        "TableName": TABLE,
        "TableArn": TABLE_ARN,
        "TableStatus": status,
        "KeySchema": key_schema if key_schema is not None else dynamodb_control.TABLE_KEY_SCHEMA,
        "BillingModeSummary": {"BillingMode": billing_mode},
    }
    if indexes is not None:
        described["GlobalSecondaryIndexes"] = indexes
    return described


def index(key_schema=None, status="ACTIVE") -> dict:
    return {
        "IndexName": dynamodb_control.EXPERIMENT_INDEX,
        "IndexStatus": status,
        "KeySchema": key_schema if key_schema is not None else dynamodb_control.INDEX_KEY_SCHEMA,
        "Projection": {"ProjectionType": "ALL"},
    }


class FakeDynamoDb:
    """One table deep, and CREATING until the second describe.

    The status transition is here rather than patched out because
    :func:`_wait_table_active` is the part that decides a deploy's "done": it has to
    keep polling while either the table or an index is still coming up, and a fake
    that is ACTIVE on the first describe would never exercise that.
    """

    def __init__(self, described: dict | None = None):
        self.described = described
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self.describes = 0

    async def describe_table(self, TableName):
        if self.described is None:
            raise botocore.exceptions.ClientError(
                {"Error": {"Code": "ResourceNotFoundException", "Message": ""}}, "DescribeTable"
            )
        self.describes += 1
        described = dict(self.described)
        if self.describes == 1:
            return {"Table": described}
        # Whatever was coming up is up by the second poll.
        described["TableStatus"] = "ACTIVE"
        if "GlobalSecondaryIndexes" in described:
            described["GlobalSecondaryIndexes"] = [
                {**gsi, "IndexStatus": "ACTIVE"} for gsi in described["GlobalSecondaryIndexes"]
            ]
        return {"Table": described}

    async def create_table(self, **kwargs):
        self.created.append(kwargs)
        self.described = table(
            key_schema=kwargs["KeySchema"],
            indexes=[{**gsi, "IndexStatus": "CREATING"} for gsi in kwargs["GlobalSecondaryIndexes"]],
            status="CREATING",
        )
        self.describes = 0
        return {"TableDescription": self.described}

    async def update_table(self, **kwargs):
        self.updated.append(kwargs)
        creates = [u["Create"] for u in kwargs.get("GlobalSecondaryIndexUpdates", []) if "Create" in u]
        self.described = table(
            indexes=[{**gsi, "IndexStatus": "CREATING"} for gsi in creates],
        )
        self.describes = 0
        return {"TableDescription": self.described}


def ensure(fake: FakeDynamoDb) -> str:
    """Run ``ensure_session_table`` against ``fake``, polling without waiting."""

    @asynccontextmanager
    async def client(region_name):
        yield fake

    with mock.patch.multiple(MODULE, _dynamodb_client=client, POLL_INTERVAL=0):
        return asyncio.run(dynamodb_control.ensure_session_table(TABLE, REGION))


class SessionTableTest(unittest.TestCase):
    def assert_valid(self, params: dict, shape) -> None:
        report = ParamValidator().validate(params, shape)
        self.assertFalse(report.has_errors(), report.generate_report())

    def test_a_missing_table_is_created_with_the_index_on_it(self):
        fake = FakeDynamoDb()
        self.assertEqual(ensure(fake), TABLE_ARN)

        params = fake.created[0]
        self.assert_valid(params, CREATE_INPUT)
        self.assertEqual(params["TableName"], TABLE)
        self.assertEqual(params["BillingMode"], "PAY_PER_REQUEST")
        self.assertEqual(params["KeySchema"], [{"AttributeName": "session_id", "KeyType": "HASH"}])
        self.assertEqual(
            params["GlobalSecondaryIndexes"][0]["KeySchema"],
            [
                {"AttributeName": "experiment_name", "KeyType": "HASH"},
                {"AttributeName": "experiment_start_at_session_id", "KeyType": "RANGE"},
            ],
        )

    def test_every_key_attribute_is_declared_as_a_string(self):
        """A key attribute missing from AttributeDefinitions is a rejected CreateTable."""
        fake = FakeDynamoDb()
        ensure(fake)
        params = fake.created[0]
        keys = {key["AttributeName"] for key in params["KeySchema"]}
        keys |= {key["AttributeName"] for key in params["GlobalSecondaryIndexes"][0]["KeySchema"]}
        self.assertEqual({d["AttributeName"] for d in params["AttributeDefinitions"]}, keys)
        self.assertEqual({d["AttributeType"] for d in params["AttributeDefinitions"]}, {"S"})

    def test_creation_waits_for_the_index_to_come_up_too(self):
        """A query against an index still CREATING fails, so the deploy waits for it."""
        fake = FakeDynamoDb()
        ensure(fake)
        self.assertGreater(fake.describes, 1)

    def test_a_table_that_is_already_right_is_left_alone(self):
        fake = FakeDynamoDb(table(indexes=[index()]))
        self.assertEqual(ensure(fake), TABLE_ARN)
        self.assertEqual((fake.created, fake.updated), ([], []))

    def test_an_index_still_backfilling_is_waited_for(self):
        """What an interrupted earlier deploy leaves behind: right schema, not yet usable."""
        fake = FakeDynamoDb(table(indexes=[index(status="CREATING")]))
        self.assertEqual(ensure(fake), TABLE_ARN)
        self.assertEqual((fake.created, fake.updated), ([], []))
        self.assertGreater(fake.describes, 1)

    def test_a_table_without_the_index_gains_it(self):
        """The one update DynamoDB allows here, and the one a deploy should make."""
        fake = FakeDynamoDb(table())
        self.assertEqual(ensure(fake), TABLE_ARN)
        self.assertEqual(fake.created, [])

        params = fake.updated[0]
        self.assert_valid(params, UPDATE_INPUT)
        created = params["GlobalSecondaryIndexUpdates"][0]["Create"]
        self.assertEqual(created["IndexName"], "experiment_sessions")
        self.assertEqual(created["KeySchema"], dynamodb_control.INDEX_KEY_SCHEMA)
        # The index's key attributes have to be declared in the same call.
        self.assertLessEqual(
            {key["AttributeName"] for key in created["KeySchema"]},
            {d["AttributeName"] for d in params["AttributeDefinitions"]},
        )

    def test_a_differently_keyed_table_is_an_error(self):
        """Not a create call: the key schema is fixed for the table's lifetime."""
        fake = FakeDynamoDb(table(key_schema=[{"AttributeName": "rollout_id", "KeyType": "HASH"}]))
        with self.assertRaises(RuntimeError):
            ensure(fake)
        self.assertEqual((fake.created, fake.updated), ([], []))

    def test_a_differently_keyed_index_is_an_error(self):
        """An index cannot be re-keyed either, and the analysis queries need these keys."""
        wrong = index(key_schema=[{"AttributeName": "experiment_name", "KeyType": "HASH"}])
        fake = FakeDynamoDb(table(indexes=[wrong]))
        with self.assertRaises(RuntimeError):
            ensure(fake)
        self.assertEqual((fake.created, fake.updated), ([], []))

    def test_a_provisioned_table_is_reported_but_not_switched(self):
        """Somebody's capacity decision, and switching it silently is not a deploy's call."""
        fake = FakeDynamoDb(table(indexes=[index()], billing_mode="PROVISIONED"))
        with self.assertLogs(MODULE, level="WARNING") as logs:
            self.assertEqual(ensure(fake), TABLE_ARN)
        self.assertIn("PROVISIONED", "\n".join(logs.output))
        self.assertEqual(fake.updated, [])


if __name__ == "__main__":
    unittest.main()
