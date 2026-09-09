#!/usr/bin/env python

"""Build the verl task parquet for a SWE dataset -- optionally only the tasks worth training on.

Without filters, every dataset instance becomes one row (prompt, eval script, task image).
Each ``--*-experiment`` filter is a rule over the per-instance pass rate of a prior batch
eval, read from the session table: drop tasks the empty patch already resolves, drop tasks
the gold patch cannot fix, keep only tasks with a non-zero GRPO advantage.

    ./swe_agent/preprocess.py
    ./swe_agent/preprocess.py --output swe_agent/local/swegym_challenged.parquet \\
        --noop-experiment eval_none_n1_gym_noop_r6 \\
        --oracle-experiment eval_none_n1_gym_r5 \\
        --model-experiment eval_qwen.qwen3-coder-30b-a3b-instruct_n4_gym_r5@2026-09-03

An experiment name resolves to its latest run unless ``name@<start-at prefix>`` pins one.
Writing the parquet also writes ``<output>.lineage.json`` beside it; ``--dry-run`` reports
the same filtration without building either.
"""

import argparse
import asyncio
import datetime as dt
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import datasets
import polars as pl
from config import Storage, dataset_parquet, load_config, storage
from polars import col as c
from rollout_report import Run, load_run

logger = logging.getLogger(__name__)

# Which dataset to convert. Switching also means pointing --swebench-path at the matching
# harness checkout: SWE-Gym's eval scripts come from its own fork.
DATASETS = {
    "swegym": ("SWE-Gym/SWE-Gym", "train", "xingyaoww"),
    "swebench": ("SWE-bench/SWE-bench_Verified", "test", "swebench"),
}

DATASET_NAME = "swegym"
DATA_SOURCE, SPLIT, DOCKER_NAMESPACE = DATASETS[DATASET_NAME]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output",
        type=Path,
        default=dataset_parquet(DATASET_NAME),
        help="parquet to write (the trainer's data.train_files) (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the filtration and the number of tasks that would be written, then "
        "stop before generating the eval scripts and the parquet",
    )
    parser.add_argument(
        "--swebench-path",
        type=Path,
        help="checkout to import the swebench harness from, for its eval scripts "
        "(SWE-Gym needs its own fork, not upstream SWE-bench)",
    )
    parser.add_argument(
        "--noop-experiment",
        metavar="NAME[@START_AT]",
        help="run of the noop agent; drops tasks it already resolves",
    )
    parser.add_argument(
        "--oracle-experiment",
        metavar="NAME[@START_AT]",
        help="run of the oracle agent; drops tasks it fails to resolve",
    )
    parser.add_argument(
        "--model-experiment",
        metavar="NAME[@START_AT]",
        help="run of the policy; keeps only tasks inside the pass-rate window",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.25,
        help="lower edge of the --model-experiment window (default: %(default)s)",
    )
    parser.add_argument(
        "--max-pass-rate",
        type=float,
        default=0.75,
        help="upper edge of the --model-experiment window (default: %(default)s)",
    )
    return parser.parse_args()


# --- the filters ---------------------------------------------------------------


@dataclass(frozen=True)
class TaskFilter:
    """One filter: a rule over one eval run's per-instance pass rate.

    ``keep`` is evaluated against the dataset's instances left-joined onto that run's pass
    rates, so it sees a null ``pass_rate`` for every instance the run did not grade.
    """

    stage: str
    experiment: str
    keep: pl.Expr
    rule: str
    # What ``keep`` does with an instance the run neither graded nor aborted, for the warning.
    unmeasured: str


def reliable(rule: pl.Expr) -> pl.Expr:
    """``rule``, and only for a task every one of whose rollouts completed.

    A null abort count means the run has nothing on the task at all, which is not an abort --
    what that means is left to ``rule``.
    """
    return (c("aborted").fill_null(0) == 0) & rule


def build_filters(args) -> list[TaskFilter]:
    """The filters the arguments asked for, in the order they are applied.

    Order does not change the result -- the rules are independent -- only how the printed
    table reads.
    """
    filters = []
    if args.noop_experiment:
        filters.append(
            TaskFilter(
                stage="already_solved",
                experiment=args.noop_experiment,
                # Null (never graded) is kept: no observed noop pass is no evidence of triviality.
                keep=reliable(c("pass_rate").fill_null(0.0) <= 0),
                rule="noop pass rate == 0",
                unmeasured="kept",
            )
        )
    if args.oracle_experiment:
        filters.append(
            TaskFilter(
                stage="unsolvable",
                experiment=args.oracle_experiment,
                # Every graded oracle rollout has to pass, and at least one has to exist.
                keep=reliable(c("pass_rate").fill_null(-1.0) >= 1),
                rule="oracle pass rate == 1",
                unmeasured="dropped",
            )
        )
    if args.model_experiment:
        filters.append(
            TaskFilter(
                stage="pass_rate",
                experiment=args.model_experiment,
                keep=reliable(c("pass_rate").fill_null(-1.0).is_between(args.min_pass_rate, args.max_pass_rate)),
                rule=f"model pass rate in [{args.min_pass_rate}, {args.max_pass_rate}]",
                unmeasured="dropped",
            )
        )
    return filters


def as_float(value) -> float | None:
    """A reward as a float (DynamoDB hands it over as a ``Decimal``), or None if ungraded."""
    return None if value is None else float(value)


def parse_experiment(spec: str) -> tuple[str, str | None]:
    """``name``, or ``name@<start-at prefix>`` to pin one run of a repeated experiment.

    An experiment name is not unique in the session table; without a prefix the latest run
    wins.
    """
    name, _, start_at = spec.partition("@")
    return name, start_at or None


async def load_rollouts(spec: str, records: Storage) -> Run:
    """The run of ``spec`` to filter by, as its rollout records, straight from the session
    table rather than from a report its driver may never have written.
    """
    experiment, start_at = parse_experiment(spec)
    run = await load_run(
        experiment,
        table_name=records.dynamodb_table,
        region_name=records.region,
        experiment_start_at=start_at,
    )
    logger.info(
        "experiment %s: %d rollouts of the run started %s",
        experiment,
        len(run.rows),
        run.experiment_start_at,
    )
    return run


def pass_rates(rollouts: list[dict]) -> pl.DataFrame:
    """Per-instance pass rate of one run: ``instance_id``, ``rollouts``, ``aborted``, ``pass_rate``.

    The pass rate is polars' mean reward, which skips nulls -- so exactly the graded rollouts.
    """
    rows = [
        {
            "instance_id": str(r["instance_id"]).lower(),
            "reward": as_float(r.get("reward")),
            "aborted": bool(r.get("aborted")),
        }
        for r in rollouts
        if r.get("instance_id")
    ]
    if not rows:
        raise RuntimeError("rollouts carry no instance_id: not a batch evaluation run?")

    return (
        pl.from_dicts(
            rows,
            schema={
                "instance_id": pl.String,
                "reward": pl.Float64,
                "aborted": pl.Boolean,
            },
        )
        .group_by("instance_id")
        .agg(
            pl.len().alias("rollouts"),
            c("aborted").sum().alias("aborted"),
            c("reward").mean().alias("pass_rate"),
        )
    )


# Stated rather than inferred: the integer columns are null on the bracketing rows.
TABLE_SCHEMA = {
    "stage": pl.String,
    "experiment": pl.String,
    "tasks": pl.Int64,
    "graded": pl.Int64,
    "aborted": pl.Int64,
    "rejected": pl.Int64,
    "dropped": pl.Int64,
    "remaining": pl.Int64,
}


async def apply_filters(
    instance_ids: list[str],
    filters: list[TaskFilter],
    records: Storage,
) -> tuple[list[str], pl.DataFrame, list[dict]]:
    """The instances that survive every filter, how they thinned out, and by what.

    ``rejected`` is what a filter refuses out of the whole dataset, independent of the
    others; ``dropped`` is what it removed from what the earlier filters had left. The third
    return value is that accounting plus the rule and the run behind it, for the lineage
    file -- built here because this is the only place that knows which run a name resolved to.
    """
    instances = pl.DataFrame({"instance_id": instance_ids}, schema={"instance_id": pl.String})
    total = instances.height
    kept = instances

    rows = [{"stage": "dataset", "experiment": f"{DATA_SOURCE}:{SPLIT}", "remaining": total}]
    provenance = []
    for task_filter in filters:
        run = await load_rollouts(task_filter.experiment, records)
        verdict = instances.join(pass_rates(run.rows), on="instance_id", how="left").select(
            "instance_id",
            keep=task_filter.keep,
            graded=c("pass_rate").is_not_null(),
            aborted=c("aborted").fill_null(0) > 0,
            # Neither graded nor aborted: the run has nothing at all on the task.
            unmeasured=c("pass_rate").is_null() & (c("aborted").fill_null(0) == 0),
        )

        unmeasured = int(verdict["unmeasured"].sum())
        if unmeasured:
            # Filtering by a run that covered a different slice of the dataset is how a
            # subset silently comes out far too small.
            logger.warning(
                "experiment %s has neither a pass rate nor an abort for %d of %d tasks, "
                "so they are %s: the run did not cover them",
                task_filter.experiment,
                unmeasured,
                verdict.height,
                task_filter.unmeasured,
            )

        before = kept.height
        kept = kept.join(verdict.filter(c("keep")), on="instance_id", how="semi")
        rows.append(
            {
                "stage": task_filter.stage,
                "experiment": run.experiment_name,
                "tasks": verdict.height,
                "graded": int(verdict["graded"].sum()),
                "aborted": int(verdict["aborted"].sum()),
                "rejected": total - int(verdict["keep"].sum()),
                "dropped": before - kept.height,
                "remaining": kept.height,
            }
        )
        provenance.append(
            rows[-1]
            | {
                "rule": task_filter.rule,
                "experiment_start_at": run.experiment_start_at,
                "run_rollouts": len(run.rows),
                "unmeasured": unmeasured,
                "unmeasured_tasks": task_filter.unmeasured,
            }
        )

    rows.append({"stage": "output", "experiment": "", "remaining": kept.height})
    return (
        kept["instance_id"].to_list(),
        pl.from_dicts(rows, schema=TABLE_SCHEMA),
        provenance,
    )


def print_filtration(filters: list[TaskFilter], table: pl.DataFrame) -> None:
    """The filtration report: what each filter keeps, then what it cost."""
    for task_filter in filters:
        print(f"{task_filter.stage}: keep if {task_filter.rule} and no rollout aborted  [{task_filter.experiment}]")

    with pl.Config(
        tbl_rows=-1,
        tbl_width_chars=200,
        fmt_str_lengths=60,
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    ):
        print(table)
    print(
        "tasks = dataset instances the run was asked about; graded = those it scored at "
        "least once;\naborted = those with at least one aborted rollout, all of which the "
        "filter rejects;\nrejected = those the filter refuses on its own; dropped = those "
        "it removed that earlier filters kept."
    )


def lineage_path(output: Path) -> Path:
    """``<output>.lineage.json``, beside the parquet it describes."""
    return output.with_suffix(output.suffix + ".lineage.json")


def write_lineage(
    output: Path,
    args,
    provenance: list[dict],
    table: pl.DataFrame,
    num_tasks: int,
    records: Storage,
) -> Path:
    """Record what this parquet is, next to it, and return where that went.

    ``experiment_start_at`` per filter is what makes the build reproducible: an experiment
    name resolves to its latest run, which is a moving target.
    """
    lineage = {
        "output": str(output),
        "written_at": dt.datetime.now().isoformat(),
        "tasks": num_tasks,
        "dataset": {
            "source": DATA_SOURCE,
            "split": SPLIT,
            "docker_namespace": DOCKER_NAMESPACE,
        },
        "session_table": {"name": records.dynamodb_table, "region": records.region},
        "pass_rate_window": {"min": args.min_pass_rate, "max": args.max_pass_rate},
        "filters": provenance,
        "filtration": table.to_dicts(),
    }
    path = lineage_path(output)
    path.write_text(json.dumps(lineage, indent=2, default=str))
    return path


# --- the conversion ------------------------------------------------------------


def load_make_test_spec(swebench_path: Path | None):
    """The harness's ``make_test_spec``, from whichever checkout is on the path.

    SWE-Gym's fork keeps it in the ``swebench.harness.test_spec`` module; upstream has since
    turned that name into a package with the function one level deeper.
    """
    if swebench_path is not None:
        sys.path.append(str(swebench_path))
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ModuleNotFoundError:
        from swebench.harness.test_spec import make_test_spec
    return make_test_spec


def format_docker_image_uri(
    instance_id: str,
    docker_namespace: str,
) -> str:
    # swebench/sweb.eval.x86_64.django_1776_django-11333:latest
    # xingyaoww/sweb.eval.x86_64.pandas-dev_s_pandas-51976:latest
    repo, name = instance_id.split("__")
    official_image_name = docker_namespace.rstrip("/")
    separator = "1776" if "swebench" in docker_namespace else "s"
    official_image_name += f"/sweb.eval.x86_64.{repo}_{separator}_{name}:latest".lower()
    return official_image_name


def process_fn(example, idx, make_test_spec):
    data = {
        "data_source": DATA_SOURCE,  # used to find the reward function definition
        "prompt": [  # conversation to load into context, harness will add system instructions
            {
                "role": "user",
                "content": example["problem_statement"],
            }
        ],
        "extra_info": {
            "index": idx,  # used to identify tasks
        },
        "task_id": example["instance_id"],
        "eval_script": make_test_spec(example).eval_script,
        "docker_image_uri": format_docker_image_uri(example["instance_id"].lower(), DOCKER_NAMESPACE),
        "repo_path": "/testbed",
    }
    return data


async def main():
    args = parse_args()
    records = storage(load_config())

    dataset = datasets.load_dataset(DATA_SOURCE)[SPLIT].map(lambda x: x | {"instance_id": x["instance_id"].lower()})

    filters = build_filters(args)
    kept_ids, table, provenance = await apply_filters(dataset["instance_id"], filters, records)
    print_filtration(filters, table)

    if filters:
        keep = set(kept_ids)
        dataset = dataset.filter(lambda x: x["instance_id"] in keep)

    # Stopping here skips the slow part: an eval script per task, and the harness import.
    if args.dry_run:
        print(f"dry run: would write {dataset.num_rows} tasks to {args.output}")
        return

    make_test_spec = load_make_test_spec(args.swebench_path)

    # After the filtering, so extra_info.index numbers this parquet's rows, not the
    # unfiltered dataset's.
    verl_dataset = dataset.map(
        function=process_fn,
        with_indices=True,
        fn_kwargs={"make_test_spec": make_test_spec},
    )
    # A fresh checkout has no ``local``, and an hour of generation must not die on that.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    verl_dataset.to_parquet(args.output)
    print(f"wrote {verl_dataset.num_rows} tasks to {args.output}")
    # After the parquet, so a lineage file never describes one that was not written.
    print(f"wrote {write_lineage(args.output, args, provenance, table, verl_dataset.num_rows, records)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    asyncio.run(main())
