"""Build the benchmark task files from their public sources (pinned Hugging Face revisions).

    python data/build_benchmarks.py [--check]

--check fails unless both files match the sha256 in data/manifests/expected.json, and unless no
benchmark task shares a repository or base commit with the training data (build that first, with
data/build_training_data.py).

Writes data/benchmarks/:
  swe-explore.jsonl        848 SWE-Explore tasks, in the order of bench.final.public.jsonl. The task
                           (repo, commit, problem statement) comes from SWE-bench Verified / Pro /
                           Multilingual; gold_files are SWE-Explore's read_core_files.
  contextbench-lite.jsonl  500 ContextBench Lite tasks (contextbench_verified.parquet), repo taken from
                           repo_url; gold_files are the files of the published gold_context.

  <name>.gold.json         each benchmark's published gold, {instance_id: [[path, start, end], ...]},
                           which eval/score.py scores against.

gold_lines in the task files holds the line-level gold ({path: [[start, end], ...]}, repo-relative
paths only) so the agent's in-episode reward runs under the training reward settings; it plays no
part in the benchmark scores.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/benchmarks"
SWE_EXPLORE = (
    "SWE-Explore-Bench/SWE-Explore-Bench",
    "bdb0ae45d7c337d9e1dc3ebfe2a0af6bc7c1fbd9",
    "bench.final.public.jsonl",
)
CONTEXTBENCH = (
    "Contextbench/ContextBench",
    "c2855792b006af41c67202d33883fb9d46362853",
    "data/contextbench_verified.parquet",
)
SWE_BENCH = {
    "verified": ("SWE-bench/SWE-bench_Verified", "78f471bf655a3137b2e8a75af1501690ec009ec3",
                 ["data/test-00000-of-00001.parquet"]),
    "multilingual": ("SWE-bench/SWE-bench_Multilingual", "846e647b9f33c0b51b739d005d13d85493c9af09",
                     ["data/test-00000-of-00001.parquet"]),
    "pro": ("ScaleAI/SWE-bench_Pro", "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50",
            ["data/default/test-00000-of-00001.parquet", "data/v1/test-00000-of-00001.parquet"]),
}  # fmt: skip


def download(repo_id, revision, filename):
    return hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision)


def task(instance_id, repo, base_commit, problem_statement, gold_files, regions, data_source):
    gold_lines = {}
    for path, start, end in regions:
        if not path.startswith("/"):  # ContextBench gold can include reproduction scripts outside the repo
            gold_lines.setdefault(path, []).append([start, end])
    return {"instance_id": instance_id, "repo": repo, "base_commit": base_commit,
            "problem_statement": problem_statement, "gold_files": gold_files, "gold_lines": json.dumps(gold_lines),
            "data_source": data_source, "in_validation": False}  # fmt: skip


def swe_explore():
    sources = {}
    for dataset, (repo_id, revision, files) in SWE_BENCH.items():
        for name in files:
            columns = ["instance_id", "repo", "base_commit", "problem_statement"]
            for row in pq.read_table(download(repo_id, revision, name), columns=columns).to_pylist():
                # SWE-bench Pro ids carry an "instance_" prefix that SWE-Explore drops.
                sources.setdefault((dataset, row["instance_id"].removeprefix("instance_")), row)
    rows = []
    for line in Path(download(*SWE_EXPLORE)).read_text().splitlines():
        bench = json.loads(line)
        source = sources[(bench["dataset"], bench["instance_id"])]
        truth = bench["ground_truth"]
        rows.append(
            task(
                bench["instance_id"],
                source["repo"],
                source["base_commit"],
                source["problem_statement"],
                truth["read_core_files"],
                [(g["path"], g["start"], g["end"]) for g in truth["read_core_regions"]],
                f"swex_{bench['dataset']}",
            )
        )
    return rows


def canonical(path, repo):
    """ContextBench gold paths are /workspace/<repo slug>__<version>/<relative>."""
    prefix = "/workspace/" + repo.replace("/", "__") + "__"
    if path.startswith(prefix):
        root, sep, relative = path[len(prefix) :].partition("/")
        if root and sep:
            return relative
    return path


def contextbench_gold(row):
    """A ContextBench row's repository and gold spans, with paths made repo-relative."""
    repo = re.sub(r"(\.git)?/?$", "", row["repo_url"].split("github.com/")[-1])
    return repo, [(canonical(g["file"], repo), g["start_line"], g["end_line"]) for g in json.loads(row["gold_context"])]


def published_gold():
    """Both benchmarks' published gold: {benchmark: {instance_id: [(path, start, end), ...]}}."""
    swe_explore = {}
    for line in Path(download(*SWE_EXPLORE)).read_text().splitlines():
        bench = json.loads(line)
        regions = bench["ground_truth"]["read_core_regions"]
        swe_explore[bench["instance_id"]] = [(g["path"], g["start"], g["end"]) for g in regions]
    rows = pq.read_table(download(*CONTEXTBENCH)).to_pylist()
    contextbench = {row["instance_id"]: contextbench_gold(row)[1] for row in rows}
    return {"swe-explore": swe_explore, "contextbench-lite": contextbench}


def contextbench_lite():
    rows = []
    for row in pq.read_table(download(*CONTEXTBENCH)).to_pylist():
        repo, gold = contextbench_gold(row)
        files = list(dict.fromkeys(path for path, _, _ in gold))
        rows.append(
            task(
                row["instance_id"],
                repo,
                row["base_commit"],
                row["problem_statement"],
                files,
                gold,
                f"ctx_{row['source'].lower()}",
            )
        )
    return rows


def training_overlap(rows):
    """Repositories and base commits a benchmark shares with data/train.jsonl and data/val.jsonl."""
    training = [
        json.loads(line)
        for split in ("train", "val")
        for line in (ROOT / f"data/{split}.jsonl").read_text().splitlines()
    ]
    repos = {r["repo"].lower() for r in training} & {r["repo"].lower() for r in rows}
    commits = {r["base_commit"] for r in training} & {r["base_commit"] for r in rows}
    return sorted(repos), sorted(commits)


def main():
    check = "--check" in sys.argv[1:]
    published = json.loads((ROOT / "data/manifests/expected.json").read_text())
    if check and not (ROOT / "data/train.jsonl").exists():
        sys.exit("--check compares against the training data: run data/build_training_data.py first")
    OUT.mkdir(exist_ok=True)
    for name, rows, expected in [
        ("swe-explore", swe_explore(), 848),
        ("contextbench-lite", contextbench_lite(), 500),
    ]:
        assert len(rows) == len({r["instance_id"] for r in rows}) == expected, (name, len(rows))
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{name}: {len(rows)} tasks -> {OUT / f'{name}.jsonl'}")
        digest = hashlib.sha256((OUT / f"{name}.jsonl").read_bytes()).hexdigest()
        if check and digest != published[f"{name}.jsonl"]["sha256"]:
            sys.exit(f"{name}.jsonl: sha256 {digest} differs from data/manifests/expected.json")
        if check and any(overlap := training_overlap(rows)):
            sys.exit(f"{name} shares (repos, commits) {overlap} with the training data")
    for name, gold in published_gold().items():
        (OUT / f"{name}.gold.json").write_text(json.dumps(gold) + "\n")


if __name__ == "__main__":
    main()
