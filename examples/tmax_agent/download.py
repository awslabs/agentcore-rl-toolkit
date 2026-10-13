"""Download pinned Harbor tasks using Harbor's official task client.

Run in a scratch environment with harbor==0.23.0 installed.
"""

import argparse
import asyncio
import json
from pathlib import Path

from harbor.models.task.id import PackageTaskId
from harbor.tasks.client import TaskClient


async def download(subset, output):
    tasks = subset["tasks"]
    ids = [
        PackageTaskId(
            org=task["package"]["org"]["name"], name=task["package"]["name"], ref=f"sha256:{task['content_hash']}"
        )
        for task in tasks
    ]
    await TaskClient().download_tasks(ids, output_dir=output, export=True)
    for task in tasks:
        (output / task["package"]["name"] / "SOURCE.json").write_text(json.dumps(task, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(download(json.loads(args.subset.read_text()), args.output))
