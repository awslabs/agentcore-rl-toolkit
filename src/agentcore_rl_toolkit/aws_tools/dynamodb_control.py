"""DynamoDB control plane: the session table every rollout is recorded in, one item per
rollout keyed by ``session_id``. Records are schemaless apart from the keys below.

The ``experiment_sessions`` index is what makes "all the sessions of one run" a
``begins_with`` query rather than a table scan. Billing is on-demand because writes come
in bursts and a throttled write here is an incomplete rollout record.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import botocore.exceptions

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session

logger = logging.getLogger(__name__)

SESSION_KEY = "session_id"

# The sort key is the compound ``"<experiment_start_at>:<session_id>"``: unique per item
# while still ordering and prefixing by run.
EXPERIMENT_INDEX = "experiment_sessions"
EXPERIMENT_PARTITION_KEY = "experiment_name"
EXPERIMENT_SORT_KEY = "experiment_start_at_session_id"

BILLING_MODE = "PAY_PER_REQUEST"

POLL_INTERVAL = 5

TABLE_KEY_SCHEMA = [{"AttributeName": SESSION_KEY, "KeyType": "HASH"}]

INDEX_KEY_SCHEMA = [
    {"AttributeName": EXPERIMENT_PARTITION_KEY, "KeyType": "HASH"},
    {"AttributeName": EXPERIMENT_SORT_KEY, "KeyType": "RANGE"},
]

# Only the key attributes need declaring; the rest of a record is schemaless.
ATTRIBUTE_DEFINITIONS = [
    {"AttributeName": name, "AttributeType": "S"}
    for name in (SESSION_KEY, EXPERIMENT_PARTITION_KEY, EXPERIMENT_SORT_KEY)
]

EXPERIMENT_INDEX_SCHEMA = {
    "IndexName": EXPERIMENT_INDEX,
    "KeySchema": INDEX_KEY_SCHEMA,
    "Projection": {"ProjectionType": "ALL"},
}


@asynccontextmanager
async def _dynamodb_client(region_name: str):
    async with (await get_aioboto3_session()).client("dynamodb", region_name=region_name) as ddb:  # type: ignore
        yield ddb


def _key_names(key_schema: list[dict]) -> list[tuple[str, str]]:
    return [(key["AttributeName"], key["KeyType"]) for key in key_schema]


async def find_table(table_name: str, region_name: str) -> dict | None:
    """The description of the table, or ``None`` if it does not exist."""
    async with _dynamodb_client(region_name) as ddb:
        try:
            return (await ddb.describe_table(TableName=table_name))["Table"]
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] == "ResourceNotFoundException":
                return None
            raise


async def _wait_table_active(ddb, table_name: str) -> dict:
    """Poll until the table *and* every index on it are ACTIVE; return the description.

    The table's own status is not enough: it reads ACTIVE while a new index backfills, and
    querying an index in ``CREATING`` fails.
    """
    while True:
        described = (await ddb.describe_table(TableName=table_name))["Table"]
        statuses = [described["TableStatus"]] + [
            index["IndexStatus"] for index in described.get("GlobalSecondaryIndexes", [])
        ]
        if all(status == "ACTIVE" for status in statuses):
            return described
        logger.info("waiting for table %s: %s", table_name, ", ".join(statuses))
        await asyncio.sleep(POLL_INTERVAL)


async def ensure_session_table(table_name: str, region_name: str) -> str:
    """The ARN of the rollout session table called ``table_name``, creating it if absent.

    Creates the table with :data:`EXPERIMENT_INDEX`, or adds that index to a table that
    predates it -- the one update DynamoDB allows here -- and waits for the result to be
    queryable. Raises if the table or index is keyed differently, since neither can be
    re-keyed on a live table. Everything else (tags, encryption, TTL, billing mode) is
    left as it is.
    """
    existing = await find_table(table_name, region_name)

    async with _dynamodb_client(region_name) as ddb:
        if existing is None:
            logger.info("creating dynamodb table %s in %s", table_name, region_name)
            await ddb.create_table(
                TableName=table_name,
                BillingMode=BILLING_MODE,
                KeySchema=TABLE_KEY_SCHEMA,
                AttributeDefinitions=ATTRIBUTE_DEFINITIONS,
                GlobalSecondaryIndexes=[EXPERIMENT_INDEX_SCHEMA],
            )
            return (await _wait_table_active(ddb, table_name))["TableArn"]

        if _key_names(existing["KeySchema"]) != _key_names(TABLE_KEY_SCHEMA):
            raise RuntimeError(
                f"dynamodb table {table_name} in {region_name} is keyed by "
                f"{_key_names(existing['KeySchema'])}, not by {_key_names(TABLE_KEY_SCHEMA)}: "
                f"a table's key schema cannot be changed, so point the config at another "
                f"table name or delete this one"
            )

        index = next(
            (i for i in existing.get("GlobalSecondaryIndexes", []) if i["IndexName"] == EXPERIMENT_INDEX),
            None,
        )
        if index is None:
            logger.info(
                "adding the %s index to dynamodb table %s in %s; it backfills every item "
                "already in the table, and this waits for that",
                EXPERIMENT_INDEX,
                table_name,
                region_name,
            )
            await ddb.update_table(
                TableName=table_name,
                AttributeDefinitions=ATTRIBUTE_DEFINITIONS,
                GlobalSecondaryIndexUpdates=[{"Create": EXPERIMENT_INDEX_SCHEMA}],
            )
            return (await _wait_table_active(ddb, table_name))["TableArn"]

        if _key_names(index["KeySchema"]) != _key_names(INDEX_KEY_SCHEMA):
            raise RuntimeError(
                f"index {EXPERIMENT_INDEX} on dynamodb table {table_name} in {region_name} "
                f"is keyed by {_key_names(index['KeySchema'])}, not by "
                f"{_key_names(INDEX_KEY_SCHEMA)}: an index cannot be re-keyed, so delete "
                f"the index and deploy again to have it recreated"
            )

        billing_mode = existing.get("BillingModeSummary", {}).get("BillingMode")
        if billing_mode not in (None, BILLING_MODE):
            logger.warning(
                "dynamodb table %s in %s is %s, not %s; leaving it as it is, but a run's "
                "writes arrive in bursts and provisioned capacity throttles them",
                table_name,
                region_name,
                billing_mode,
                BILLING_MODE,
            )

        logger.info("dynamodb table %s in %s already has the session schema", table_name, region_name)
        # Still waited on: an index created by an earlier or interrupted deploy may still
        # be backfilling, and a returned table must be queryable.
        await _wait_table_active(ddb, table_name)
        return existing["TableArn"]
