"""Build sandboxd overlays from reviewed, downloaded Harbor task bundles."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def command(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def build(context, repository, tag):
    uri = f"{repository}:{tag}"
    ecr = boto3.client("ecr", region_name=repository.split(".")[3])
    try:
        ecr.describe_images(repositoryName=repository.split("/", 1)[1], imageIds=[{"imageTag": tag}])
        print(f"Reusing {uri}", flush=True)
        return uri
    except ecr.exceptions.ImageNotFoundException:
        pass
    command(
        "docker",
        "buildx",
        "build",
        "--platform",
        "linux/arm64",
        "--provenance=false",
        "--load",
        "-t",
        uri,
        str(context),
    )
    command("docker", "push", uri)
    return uri


def prepare_task(task_dir, binary, repository, bucket, s3, scratch):
    config = tomllib.loads((task_dir / "task.toml").read_text())
    source = json.loads((task_dir / "SOURCE.json").read_text())
    context = scratch / task_dir.name
    # The build context contains only environment/, never instruction/tests/solution.
    shutil.copytree(task_dir / "environment", context)
    shutil.copy2(binary, context / "agentcore-sandboxd")
    with (context / "Dockerfile").open("a") as dockerfile:
        dockerfile.write(
            "\nENV TMPDIR=/var/tmp\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \\\n"
            "    && rm -rf /var/lib/apt/lists/*\n"
            "COPY --chmod=755 agentcore-sandboxd /usr/local/bin/agentcore-sandboxd\n"
            'EXPOSE 8080\nCMD ["/usr/local/bin/agentcore-sandboxd"]\n'
        )
    digest = hashlib.sha256()
    for file in sorted(context.rglob("*")):
        if file.is_file():
            digest.update(str(file.relative_to(context)).encode())
            digest.update(file.read_bytes())
    image = build(context, repository, f"{task_dir.name}-{digest.hexdigest()[:12]}")
    archive = scratch / f"{task_dir.name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(task_dir / "tests", arcname="tests")
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    key = f"tmax/verifiers/{task_dir.name}/{archive_hash}.tar.gz"
    s3.upload_file(str(archive), bucket, key)
    return {
        "task_id": task_dir.name,
        "source": source,
        "image_uri": image,
        "instruction": (task_dir / "instruction.md").read_text(),
        "verifier_key": key,
        "verifier_sha256": archive_hash,
        "agent_timeout": config["agent"]["timeout_sec"],
        "verifier_timeout": int(config["verifier"]["timeout_sec"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("tasks", type=Path, nargs="+")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    s3 = boto3.client("s3", region_name=config["verifier_region"])
    with tempfile.TemporaryDirectory(prefix="tmax-build-") as tmp:
        scratch = Path(tmp)
        binary = scratch / "agentcore-sandboxd"
        command(
            "go",
            "build",
            "-ldflags",
            "-s -w",
            "-o",
            str(binary),
            ".",
            cwd=ROOT / "sandboxd",
            env={**os.environ, "GOOS": "linux", "GOARCH": "arm64", "CGO_ENABLED": "0"},
        )
        harness = scratch / "harness"
        harness.mkdir()
        for name in ("Dockerfile", "app.py", "harness.py"):
            shutil.copy2(HERE / name, harness / name)
        digest = hashlib.sha256()
        for file in sorted(harness.iterdir()):
            digest.update(file.name.encode())
            digest.update(file.read_bytes())
        manifest = {
            "harness_image": build(harness, config["repository"], f"harness-{digest.hexdigest()[:12]}"),
            "tasks": [],
        }
        for task in args.tasks:
            manifest["tasks"].append(
                prepare_task(task, binary, config["repository"], config["verifier_bucket"], s3, scratch)
            )
            args.output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
