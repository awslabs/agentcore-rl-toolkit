"""The read side of a run's rollout records: loading them out of the session table
(:func:`load_runs`) and reducing them to the numbers a run is judged by
(:func:`summarize`). A report file is a snapshot of this, not the place the numbers live.

The reduction names almost no fields on purpose -- only ``reward``/``resolved`` and
``task_id``, because pass@k is a cross-sample question generic reduction cannot express.
"""

import logging
from typing import NamedTuple

from agentcore_rl_toolkit.aws_tools.dynamodb_tools import (
    from_dynamodb,
    get_sessions_for_experiment_run,
)

logger = logging.getLogger(__name__)


class Run(NamedTuple):
    """One run of one experiment: its start timestamp and its rollout records.

    An experiment name is not unique -- re-running one appends another run to the same
    partition -- so a run is the (name, start) pair, i.e. the index's sort key prefix.
    """

    experiment_name: str
    experiment_start_at: str
    rows: list[dict]


async def load_runs(
    experiment_name: str,
    *,
    table_name: str,
    region_name: str | None = None,
    experiment_start_at: str | None = None,
) -> list[Run]:
    """Every run of ``experiment_name`` in the session table, oldest first.

    ``experiment_start_at`` is a prefix on the run's timestamp, so it can pin one run or
    narrow to a day. Rows come back with DynamoDB's ``Decimal``s converted, since a
    record whose numbers are Decimals reduces to nothing. Raises when the name matches
    nothing, rather than reporting zero rollouts for what is usually a typo.
    """
    sessions = await get_sessions_for_experiment_run(
        experiment_name,
        experiment_start_at or "",
        table_name=table_name,
        region_name=region_name,
    )
    if not sessions:
        raise RuntimeError(
            f"no rollout records for experiment {experiment_name!r}"
            + (f" starting {experiment_start_at!r}" if experiment_start_at else "")
            + f" in dynamodb table {table_name}: the name is the experiment_name an "
            f"eval or a training run recorded its sessions under"
        )

    by_run: dict[str, list[dict]] = {}
    for session in sessions:
        row = from_dynamodb(session)
        by_run.setdefault(str(row.get("experiment_start_at", "")), []).append(row)

    return [Run(experiment_name, start_at, rows) for start_at, rows in sorted(by_run.items())]


async def load_run(
    experiment_name: str,
    *,
    table_name: str,
    region_name: str | None = None,
    experiment_start_at: str | None = None,
) -> Run:
    """The latest run of ``experiment_name``, of however many the table holds.

    Warns (with the timestamps to pin an older one by) when there was a choice: silently
    analysing a different run than the caller meant is the failure worth being loud about.
    """
    runs = await load_runs(
        experiment_name,
        table_name=table_name,
        region_name=region_name,
        experiment_start_at=experiment_start_at,
    )
    if len(runs) > 1:
        logger.warning(
            "experiment %s has %d runs in the table (%s); using the latest, %s -- "
            "pass an experiment_start_at prefix to pick another",
            experiment_name,
            len(runs),
            ", ".join(f"{run.experiment_start_at} ({len(run.rows)} rollouts)" for run in runs),
            runs[-1].experiment_start_at,
        )
    return runs[-1]


def percentiles(values: list, ps=(50, 90, 99)) -> dict:
    """Nearest-rank percentiles + count/mean/min/max/sum; dependency-free, None-safe."""
    values = [v for v in values if v is not None]
    if not values:
        return {f"p{p}": None for p in ps} | {"count": 0, "mean": None, "min": None, "max": None, "sum": None}
    ordered = sorted(values)
    out = {}
    for p in ps:
        k = max(0, min(len(ordered) - 1, int(round((p / 100) * len(ordered) + 0.5)) - 1))
        out[f"p{p}"] = ordered[k]
    out["count"] = len(ordered)
    out["mean"] = sum(ordered) / len(ordered)
    out["min"] = ordered[0]
    out["max"] = ordered[-1]
    out["sum"] = sum(ordered)
    return out


def numeric_stats(rows: list[dict]) -> dict:
    """Summarize every numeric field present in any rollout row.

    Aggregates ``int``/``float`` (bool included, so a flag reduces to its rate) and skips
    everything else, since a row is the session meta and carries datetimes, arns and ids
    too -- the trainer's rule, so no field name needs declaring here. A field is
    summarized only over the rollouts that reported it; ``count`` says how many.
    """
    per_field: dict[str, list[float]] = {}
    for row in rows:
        for name, value in row.items():
            if isinstance(value, (int, float)):
                per_field.setdefault(name, []).append(float(value))
    return {name: percentiles(values) for name, values in sorted(per_field.items())}


def summarize(rows: list[dict], k: int | None = None) -> dict:
    """Reduce per-rollout records to pass@k plus a summary of every numeric field.

    ``k`` defaults to the largest group of samples any task has, so a run loaded from the
    table reports the same pass@k the eval did from its config. Stats cover non-aborted
    rollouts only (a container that never ran shouldn't drag the distributions), but
    aborted ones still count in the pass@k denominator.
    """
    by_task: dict = {}
    for row in rows:
        by_task.setdefault(row.get("task_id"), []).append(row)

    tasks_passed = sum(1 for samples in by_task.values() if any(s.get("resolved") for s in samples))
    num_tasks = len(by_task)

    ok = [row for row in rows if not row.get("aborted")]
    rewards = [row["reward"] for row in ok if row.get("reward") is not None]

    return {
        "counts": {
            "rollouts": len(rows),
            "rollouts_ok": len(ok),
            "rollouts_aborted": sum(1 for row in rows if row.get("aborted")),
            "tasks": num_tasks,
        },
        "reward": {
            "pass_at_k_k": k if k is not None else max((len(s) for s in by_task.values()), default=0),
            "pass_at_k": (tasks_passed / num_tasks) if num_tasks else None,
            "tasks_passed": tasks_passed,
            "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
        },
        "metrics": numeric_stats(ok),
    }
