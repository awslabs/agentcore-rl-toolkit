#!/usr/bin/env python

"""Run this recipe's server-side unit tests inside the agent image just built.

``deploy.py`` calls this between the build and the push -- the image is the only place
these tests' imports resolve. Neither the tests nor pytest are baked in: the tests are
bind-mounted read-only and pytest is installed into the container's venv for the run.
"""

import argparse
import asyncio
import logging
import shlex
from pathlib import Path

logger = logging.getLogger(__name__)

RECIPE_DIR = Path(__file__).resolve().parent
TESTS_DIR = RECIPE_DIR / "tests"

CONTAINER_TESTS_DIR = "/agent/tests"

# Excluded because they cover the host side of the recipe (the deploy's execution role and
# the eval harness), none of which is in the image. Naming the exceptions rather than the
# inclusions means a new server-side test file runs here just by existing.
HOST_ONLY_TESTS = (
    "iam_policy_test.py",
    "rollout_batch_test.py",
    "rollout_report_test.py",
)


def pytest_argv() -> list[str]:
    """The pytest invocation, as run inside the container."""
    return [
        "pytest",
        CONTAINER_TESTS_DIR,
        *(f"--ignore={CONTAINER_TESTS_DIR}/{name}" for name in HOST_ONLY_TESTS),
        "-q",
        # No .pytest_cache: the container is deleted when the run ends.
        "-p",
        "no:cacheprovider",
    ]


def container_script() -> str:
    """The shell the container runs: venv, pytest, tests. ``exec`` so pytest's exit status
    is the container's.
    """
    return "\n".join(
        [
            "set -eu",
            "source /agent/.venv/bin/activate",
            "uv pip install --quiet pytest",
            f"exec {shlex.join(pytest_argv())}",
        ]
    )


def docker_argv(image_uri: str) -> list[str]:
    """The ``docker run`` that executes :func:`container_script` in ``image_uri``.

    No ``--entrypoint`` override needed: the image sets ``CMD``, so the command given here
    replaces the agent server outright.
    """
    return [
        "docker",
        "run",
        "--rm",
        "--volume",
        f"{TESTS_DIR}:{CONTAINER_TESTS_DIR}:ro",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        image_uri,
        "bash",
        "-c",
        container_script(),
    ]


async def verify_image(image_uri: str) -> None:
    """Run the server-side tests in ``image_uri``, raising if any of them fails."""
    argv = docker_argv(image_uri)
    logger.info("+ %s", shlex.join(argv))
    proc = await asyncio.create_subprocess_exec(*argv)
    if await proc.wait() != 0:
        raise RuntimeError(
            f"the unit tests failed in {image_uri}, which was therefore not pushed: "
            f"see the pytest output above. Re-run them on their own with "
            f"./verify_image.py {image_uri}, or deploy with --skip-tests to push anyway"
        )
    logger.info("unit tests passed in %s", image_uri)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "image",
        help="the image to run the tests in, as docker resolves it: a local tag or a registry reference",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    asyncio.run(verify_image(parse_args().image))
