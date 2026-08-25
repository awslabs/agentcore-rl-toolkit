"""Build payload-only MigrationBench parquet files from prepared S3 metadata.

Run ``examples/strands_migration_agent/preprocess.py`` first to upload repository
tarballs and metadata. Training excludes repositories without tests; validation
uses every prepared test repository.

The payload shape is the agent's invoke contract (``models.InvocationRequest`` in
that example) and is also built by its ``eval_utils.prepare_payload`` for batch
evaluation — the example is a standalone uv project, not importable from the
trainer env, so a contract change has to be applied in both places.
"""

import argparse
import concurrent.futures
import json
import os

import boto3
import pandas as pd
from botocore.exceptions import ClientError

# Formatted by the agent (rl_app.py) once it knows the container-local repo path;
# it is deliberately not resolved here.
PROMPT_TEMPLATE = "Please help migrate this repo: {repo_path}. There are {num_tests} test cases in it."


def list_repo_prefixes(s3, bucket: str, prefix: str) -> list[str]:
    """Immediate sub-folder names under an S3 prefix (one per repo)."""
    paginator = s3.get_paginator("list_objects_v2")
    names = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            names.append(cp["Prefix"][len(prefix) :].rstrip("/"))
    return names


def fetch_metadata(s3, bucket: str, key: str) -> dict | None:
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def build_split(
    s3,
    bucket: str,
    prefix: str,
    *,
    require_tests: bool,
    require_maximal_migration: bool,
    use_dependency_search_tool: bool,
    apply_static_update: bool,
) -> tuple[list[dict], int]:
    folders = list_repo_prefixes(s3, bucket, prefix)
    print(f"  {prefix}: {len(folders)} repos in S3")

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        metas = pool.map(lambda f: (f, fetch_metadata(s3, bucket, f"{prefix}{f}/metadata.json")), folders)
        pairs = [(f, m) for f, m in metas if m is not None]

    rows, skipped = [], len(folders) - len(pairs)
    for folder, meta in sorted(pairs):
        num_tests = meta.get("num_test_cases", 0)
        if require_tests and num_tests <= 0:
            skipped += 1
            continue
        rows.append(
            {
                "payload": {
                    "prompt": PROMPT_TEMPLATE,
                    "repo_uri": f"s3://{bucket}/{prefix}{folder}/{folder}.tar.gz",
                    "metadata_uri": f"s3://{bucket}/{prefix}{folder}/metadata.json",
                    "require_maximal_migration": require_maximal_migration,
                    "use_dependency_search_tool": use_dependency_search_tool,
                    "apply_static_update": apply_static_update,
                },
            }
        )
    return rows, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket-name", required=True, help="Bucket the agent's preprocess.py uploaded to")
    parser.add_argument("--output-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--require-maximal-migration", action="store_true", help="Require dependencies at their latest Java 17 versions"
    )
    parser.add_argument(
        "--use-dependency-search-tool", action="store_true", help="Give the agent the version-lookup tool"
    )
    parser.add_argument(
        "--apply-static-update", action="store_true", help="Pre-update pom.xml dependencies before the agent starts"
    )
    args = parser.parse_args()

    s3 = boto3.client("s3")
    os.makedirs(args.output_dir, exist_ok=True)

    for prefix, require_tests, out_name in [
        ("tars/train/", True, "migrationbench_agent_train.parquet"),
        ("tars/test/", False, "migrationbench_agent_test.parquet"),
    ]:
        rows, skipped = build_split(
            s3,
            args.s3_bucket_name,
            prefix,
            require_tests=require_tests,
            require_maximal_migration=args.require_maximal_migration,
            use_dependency_search_tool=args.use_dependency_search_tool,
            apply_static_update=args.apply_static_update,
        )
        if not rows:
            raise SystemExit(
                f"No usable repos under s3://{args.s3_bucket_name}/{prefix} — run the agent example's "
                "preprocess.py against this bucket first (see this file's docstring)."
            )
        out_path = os.path.join(args.output_dir, out_name)
        pd.DataFrame(rows, columns=["payload"]).to_parquet(out_path, index=False)
        print(f"wrote {out_path}: {len(rows)} rows ({skipped} skipped)")


if __name__ == "__main__":
    main()
