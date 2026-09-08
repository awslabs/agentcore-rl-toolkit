#!/usr/bin/env python

"""Report the stats of a run -- any run -- from the rollout records in DynamoDB.

The same numbers an eval prints when it finishes, computed the same way
(:mod:`rollout_report`), but from the session table rather than from a report file.
That is the point of this script existing
separately: every rollout persists its own record as it goes, so a run's results are
readable while it is still in flight, after its driver has exited, for a run whose
driver died before writing anything, and for a *training* run, which writes the same
records and never wrote a report at all.

    ./swe_agent/analyze.py eval_none_n1_gym_r5
    ./swe_agent/analyze.py eval_none_n1_gym_r5 eval_none_n1_bench   # one report each
    ./swe_agent/analyze.py eval_none_n1_gym_r5 --runs               # which runs exist
    ./swe_agent/analyze.py eval_none_n1_gym_r5 --start-at 2026-09-03T19
    ./swe_agent/analyze.py eval_none_n1_gym_r5 --output /tmp/r5.json

An experiment name is what a run recorded its sessions under -- ``experiment_name``
in the eval grid, ``trainer.experiment_name`` for a training run -- and it is not
unique: re-running one appends another run to the same partition. So the latest run
wins by default, ``--runs`` lists what is there, and ``--start-at`` pins an older one
by a prefix of its start timestamp.

The table and its region come from this recipe's ``config.toml`` (``[storage]``), so
this needs AWS credentials that can read that table, and nothing else -- no parquet,
no S3, no runtime.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

import polars as pl
from config import load_config, storage
from rollout_report import Run, load_run, load_runs, summarize

logger = logging.getLogger(__name__)

# The stats columns worth a terminal's width, in the order they read in. Every field
# `numeric_stats` produces has all of them; `sum` is dropped from the table because it
# is meaningful for a handful of fields (token counts) and misleading for the rest.
STAT_COLUMNS = ["count", "mean", "p50", "p90", "p99", "min", "max"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "experiments",
        nargs="+",
        help="experiment names, as the run recorded them in the session table",
    )
    parser.add_argument(
        "--start-at",
        help="prefix of the run's start timestamp, to analyze a run other than the "
        "latest of an experiment that was run more than once",
    )
    parser.add_argument(
        "--runs",
        action="store_true",
        help="list the runs each experiment name has in the table and stop, without loading or reducing any of them",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the report as json (counts, reward, metrics and every "
        "rollout record); one file per experiment, so this is a directory "
        "unless exactly one experiment is named",
    )
    args = parser.parse_args()
    if args.output is not None and len(args.experiments) > 1 and args.output.suffix == ".json":
        parser.error("--output names one file but several experiments were given; pass a directory instead")
    return args


def print_summary(run: Run, report: dict) -> None:
    """The header, then a row per numeric field the rollouts carried."""
    counts, reward = report["counts"], report["reward"]
    print(
        f"\n{run.experiment_name}  run {run.experiment_start_at}\n"
        f"  rollouts {counts['rollouts']} (ok {counts['rollouts_ok']}, "
        f"aborted {counts['rollouts_aborted']}) over {counts['tasks']} tasks\n"
        f"  pass@{reward['pass_at_k_k']} {_number(reward['pass_at_k'])}  "
        f"tasks_passed {reward['tasks_passed']}  "
        f"mean_reward {_number(reward['mean_reward'])}"
    )

    table = pl.from_dicts(
        [
            {"metric": name} | {column: stats[column] for column in STAT_COLUMNS}
            for name, stats in report["metrics"].items()
        ],
        schema={"metric": pl.String} | {column: pl.Float64 for column in STAT_COLUMNS},
    )
    with pl.Config(
        tbl_rows=-1,
        tbl_width_chars=200,
        fmt_str_lengths=60,
        float_precision=3,
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    ):
        print(table)
    print(
        "one row per numeric field the rollout records carried, over the "
        f"{counts['rollouts_ok']} rollouts that were not aborted."
    )


def _number(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def report_path(output: Path, experiment_name: str, single: bool) -> Path:
    """Where ``--output`` puts this experiment's json."""
    if single and output.suffix == ".json":
        return output
    return output / f"{experiment_name}.json"


async def analyze(experiment_name: str, args, table_name: str, region_name: str) -> None:
    """Load one experiment's run, print its stats, and write its json if asked."""
    run = await load_run(
        experiment_name,
        table_name=table_name,
        region_name=region_name,
        experiment_start_at=args.start_at,
    )
    # No `k`: the rows are all there is here, so pass@k is over the samples the run
    # actually recorded rather than over an `n` from a config this script never sees.
    report = summarize(run.rows)
    print_summary(run, report)

    if args.output is not None:
        path = report_path(args.output, experiment_name, single=len(args.experiments) == 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "experiment_name": run.experiment_name,
            "experiment_start_at": run.experiment_start_at,
            **report,
            "rollouts": run.rows,
        }
        path.write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {path}")


async def print_runs(experiment_name: str, args, table_name: str, region_name: str) -> None:
    """Every run the name has, so an older one can be named with ``--start-at``."""
    runs = await load_runs(
        experiment_name,
        table_name=table_name,
        region_name=region_name,
        experiment_start_at=args.start_at,
    )
    print(f"\n{experiment_name}: {len(runs)} run(s)")
    for run in runs:
        aborted = sum(1 for row in run.rows if row.get("aborted"))
        print(f"  {run.experiment_start_at}  {len(run.rows)} rollouts, {aborted} aborted")


async def main() -> None:
    args = parse_args()
    records = storage(load_config())
    logger.info(
        "reading rollout records from dynamodb table %s in %s",
        records.dynamodb_table,
        records.region,
    )

    for experiment_name in args.experiments:
        step = print_runs if args.runs else analyze
        await step(experiment_name, args, records.dynamodb_table, records.region)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    asyncio.run(main())
