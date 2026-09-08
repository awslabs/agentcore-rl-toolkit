#!/usr/bin/env python

"""Run this recipe's server-side unit tests inside the agent image just built.

``deploy.py`` calls :func:`verify_image` between the build and the push, so the
artifact that reaches a runtime is the artifact these tests ran in, and a harness
that no longer imports cannot be deployed at all -- the failure is a raise, before
the push, rather than a session that dies on its first rollout.

Inside the image is the *only* place these tests run. They import
``swe_agent_server`` and, through it, ``agentcore_rl_toolkit.rollout_session.wire``,
openhands and strands: that set of packages exists in one environment, the one
``image/copy.sh`` builds at ``/agent/.venv``, and nowhere on the developer node or in
the toolkit's own dev environment. Running them on the host is what has kept them
dead (``uv run pytest examples/openhands_swe_agent/tests`` collects them and fails on
the imports), so the container is not a convenience here, it is the environment.

Two things are deliberately *not* baked into the image:

* the tests themselves, which are bind-mounted read-only at ``/agent/tests`` for the
  duration of the run. A shipped rollout container has no business carrying its own
  test suite, and mounting also means editing a test does not cost an image build.
* pytest, installed into the container's venv as the first thing the run does. Same
  reason: a test dependency in a build layer is a test dependency in the artifact.
  It is installed by name here rather than declared as a ``[test]`` extra in this
  recipe's ``pyproject.toml`` because that file is copied into the image and is what
  ``copy.sh`` installs from -- an extra there is a line the shipped image reads, and
  the one place this dependency is needed is the throwaway container below.

Runnable on its own against any image reference, which is the way to re-run the
tests against an image already built:

    ./verify_image.py 5211...dkr.ecr.us-west-2.amazonaws.com/vdanylo:swe_agent_server_r70
"""

import argparse
import asyncio
import logging
import shlex
from pathlib import Path

logger = logging.getLogger(__name__)

RECIPE_DIR = Path(__file__).resolve().parent
TESTS_DIR = RECIPE_DIR / "tests"

# Beside /agent/src and /agent/pyproject.toml, where the harness itself lives. The
# mount is read-only: the container must not be able to write into the developer's
# checkout, least of all root-owned ``__pycache__`` directories.
CONTAINER_TESTS_DIR = "/agent/tests"

# Everything in tests/ runs in the container except these, which cover the *host* side
# of the recipe -- the deploy's execution role (``config``, ``iam_policy``) and the eval
# harness (``rollout_batch``, ``rollout_report``, and through them aws_tools and
# concurrency). None of that is in the image, which carries the agent server and nothing
# that drives it. They run in the toolkit's own dev environment instead, where this
# recipe's ``conftest.py`` puts the recipe directory on ``sys.path`` for them. Naming the
# exceptions rather than listing the included files means a new server-side test file is
# run here just by existing.
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
        # No .pytest_cache: nothing reads it between runs of a container that is
        # deleted when the run ends.
        "-p",
        "no:cacheprovider",
    ]


def container_script() -> str:
    """The shell the container runs: venv, pytest, tests.

    ``exec`` so pytest's exit status is the container's, which is what
    :func:`verify_image` reads. The venv activation is the same one
    ``image/entrypoint.sh`` does for the server -- these tests run against the
    interpreter the harness runs on, not a second one.
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

    No ``--entrypoint`` override is needed: the image sets ``CMD`` and not
    ``ENTRYPOINT``, so a command given here replaces the agent server outright.
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
    """Run the server-side tests in ``image_uri``, raising if any of them fails.

    Output is inherited rather than captured: a failing assertion is the reason to
    read this at all, so it goes to the terminal as pytest wrote it.
    """
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
