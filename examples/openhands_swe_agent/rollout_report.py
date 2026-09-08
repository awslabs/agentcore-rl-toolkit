"""What a run's rollout records mean: loading them back, and reducing them to stats.

The session table is the record of a run -- one item per rollout, written through as
the rollout progresses. This module is the read side of that: it loads a run's items
back out (:func:`load_runs`) and reduces them to the numbers a run is judged by
(:func:`summarize`).

Both halves are here because they are the same claim made twice. An eval writes a
report file when it finishes, and that file used to be the only way to see any of
this -- which made every question about a run answerable exactly once, at the moment
it ended, and not at all if the driver died on the last rollout, or if the question
came up later, or if the run was a training run rather than an eval. The records were
in DynamoDB the whole time. So the report is now a *snapshot* of what this module
computes, from the same rows, rather than the place the numbers live.

The reduction knows almost no field names on purpose. A rollout record is the
session's whole meta -- reward, the session's metrics, every lifecycle timing span,
the token counts -- flat, so :func:`numeric_stats` summarizes whatever is numeric and
a metric added anywhere upstream shows up here without being declared. Only
``reward``/``resolved`` and ``task_id`` are named, because pass@k is a cross-sample
question that generic reduction cannot express.
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

    An experiment name is not unique -- re-running the same grid entry appends
    another run to the same partition -- so a run is the (name, start) pair, which
    is exactly what the index's sort key prefix is.
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

    ``experiment_start_at`` is a prefix on the run's timestamp, so it can pin one run
    exactly or narrow to a day. The rows come back with DynamoDB's ``Decimal``s
    converted (:func:`~agentcore_rl_toolkit.aws_tools.dynamodb_tools.from_dynamodb`),
    because a record whose numbers are Decimals reduces to nothing at all.

    Raises when the name matches no records: the alternative is a report of zero
    rollouts, which reads like a run that did nothing rather than like a typo.
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

    Latest rather than "the one run" because a name can be reused, and rather than an
    error because the last run of an experiment is nearly always the one meant -- the
    earlier ones are the attempts that were re-run. It says so when it had a choice,
    with the timestamps to pin an older one by, since silently analysing a different
    run than the caller had in mind is the failure worth being loud about.
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

    The reduction rule is the trainer's -- see
    ``AgentLoopMetricsMixin._agent_loop_extra_field_metrics``, which reduces the
    ``agent_loop/*`` metrics the same way: aggregate ``int``/``float`` values (bool
    included, so a flag reduces to its rate) and skip everything else, because a
    rollout row is the session meta and carries non-numeric fields too (datetimes,
    arns, ids) that would otherwise crash the arithmetic. No field name appears
    here, so a new session metric, timing span or token stat lands in the report
    without touching this function.

    Each field is summarized only over the rollouts that reported it: absent or
    ``None`` is excluded rather than counted as 0, and ``count`` says how many
    rollouts contributed.
    """
    per_field: dict[str, list[float]] = {}
    for row in rows:
        for name, value in row.items():
            if isinstance(value, (int, float)):
                per_field.setdefault(name, []).append(float(value))
    return {name: percentiles(values) for name, values in sorted(per_field.items())}


def summarize(rows: list[dict], k: int | None = None) -> dict:
    """Reduce per-rollout records to pass@k plus a summary of every numeric field.

    ``k`` is the samples-per-task the pass@k is over. Observed from the rows when not
    given -- the largest group of samples any task has -- so that a run loaded from
    the session table reports the same pass@k as the eval that produced it did from
    its own config, without the config having to be recorded anywhere.

    Stats are taken over non-aborted rollouts only, so a container that never ran
    doesn't drag the latency and token distributions. Aborted rollouts still count in
    the pass@k denominator: they are tasks the run failed to answer, not tasks it was
    not asked.
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
