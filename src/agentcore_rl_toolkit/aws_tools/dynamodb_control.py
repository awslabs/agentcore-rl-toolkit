"""DynamoDB *control plane*: the session table every rollout is recorded in.

Looked up by name, created if absent, reconciled as far as the API allows, reading
no config of its own -- the same shape as the other control-plane modules here.
Writing and querying the records is the data plane, elsewhere.

One item per rollout, keyed by ``session_id`` and mutated as the rollout
progresses, which is what makes an in-flight run observable and a finished one
analysable. Nothing here declares the item's fields -- DynamoDB does not ask, and
they differ between writers -- so the whole schema is the keys below.

**The index is the point.** A ``session_id`` is a uuid, so it answers "how did
*this* rollout go" and nothing else. Every analysis question is instead "all the
sessions of one run", and that is what ``experiment_sessions`` is for: partitioned
by ``experiment_name``, sorted by ``"<experiment_start_at>:<session_id>"``, so one
run of an experiment is a ``begins_with`` query on the sort key rather than a scan
of every rollout ever recorded. It projects ALL because the readers want whole
records, and a table this size must never be scanned to find a hundred rows.

**On-demand billing**, because the write pattern is all burst: a training step or an
eval run updates thousands of items within a couple of minutes and then writes
nothing at all until the next one. Provisioned capacity sized for the burst is paid
for while idle, and capacity sized for the average throttles exactly when a run is
recording -- and a throttled write here is a rollout whose record is incomplete.

What can be reconciled is narrower than what can be created: a table's key schema is
fixed for its lifetime, and a secondary index can be *added* to a live table (with a
backfill) but not re-keyed. So :func:`ensure_session_table` creates the whole thing,
adds a missing index to a table that predates it, and raises on anything else that
disagrees rather than pretending to have fixed it.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import botocore.exceptions

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session

logger = logging.getLogger(__name__)

# The table's own key: one item per rollout session.
SESSION_KEY = "session_id"

# The index the analysis reads a run through, and its two keys. The sort key is
# ``"<experiment_start_at>:<session_id>"`` -- a compound of the run's start time and
# the session -- so that it is unique per item while still ordering and prefixing by
# run, which is what makes one run of a repeated experiment a range query.
EXPERIMENT_INDEX = "experiment_sessions"
EXPERIMENT_PARTITION_KEY = "experiment_name"
EXPERIMENT_SORT_KEY = "experiment_start_at_session_id"

BILLING_MODE = "PAY_PER_REQUEST"

# Seconds between describe calls while waiting for the table or an index to come up.
POLL_INTERVAL = 5

TABLE_KEY_SCHEMA = [{"AttributeName": SESSION_KEY, "KeyType": "HASH"}]

INDEX_KEY_SCHEMA = [
    {"AttributeName": EXPERIMENT_PARTITION_KEY, "KeyType": "HASH"},
    {"AttributeName": EXPERIMENT_SORT_KEY, "KeyType": "RANGE"},
]

# Only the key attributes are declared, and all three are strings. Everything else a
# record holds is schemaless, which is why the writers can add a field without this
# module hearing about it.
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
    """A key schema as comparable pairs, in the order the keys are declared in."""
    return [(key["AttributeName"], key["KeyType"]) for key in key_schema]


async def find_table(table_name: str, region_name: str) -> dict | None:
    """The description of the table called ``table_name``, or ``None``."""
    async with _dynamodb_client(region_name) as ddb:
        try:
            return (await ddb.describe_table(TableName=table_name))["Table"]
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] == "ResourceNotFoundException":
                return None
            raise


async def _wait_table_active(ddb, table_name: str) -> dict:
    """Poll until the table *and* every index on it are ACTIVE; return the description.

    The table's own status is not enough: a newly created index reports the table
    ACTIVE while it backfills, and a query against an index in ``CREATING`` fails.
    Waiting for both is what makes a deploy that returns mean the analysis queries
    work, and it is why adding an index to a large table takes as long as it does --
    the backfill reads every existing item.
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

    Creates the table with :data:`EXPERIMENT_INDEX` on it, or adds that index to a
    table that has everything else -- the one update DynamoDB allows here, and the
    one worth making, since a table created before the index exists is one every
    analysis query has to scan. Both wait for the result to be usable rather than
    merely accepted.

    Raises if an existing table is keyed differently, or if the index is there under
    different keys: neither can be changed on a live table, so the choices are a
    different table name or deleting this one, and that is a decision about somebody's
    recorded runs rather than something a deploy should take.

    Leaves everything the recipe has no opinion about alone -- tags, encryption,
    deletion protection, TTL, point-in-time recovery. A billing mode that is not
    on-demand is reported rather than changed: it is somebody's capacity decision, and
    switching it is not a call a deploy of an agent should make silently.
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
        # Still waited on: an index added by a deploy that was interrupted, or by the
        # one before this in a shared account, backfills long after the call that
        # created it returned. Otherwise this function's promise -- that a table it
        # returns can be queried -- would hold only on the paths that write.
        await _wait_table_active(ddb, table_name)
        return existing["TableArn"]
