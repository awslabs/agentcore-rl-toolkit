import json
import logging
import os
import random
import socket
from contextlib import contextmanager
from typing import Any, Optional

from appworld import AppWorld
from managed_process import ManagedProcess
from strands import tool

logger = logging.getLogger(__name__)


def cap_string(json_str: str, max_length: int = 7200) -> str:
    """Truncate a string to max_length, appending a truncation notice."""
    if len(json_str) > max_length:
        suffix = "... (truncated)"
        return json_str[: max_length - len(suffix)] + suffix
    return json_str


def get_free_port(start_port: int = 6000, max_port: int = 65535) -> int:
    """Scan from start_port upward until a free port is found."""
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return port
            except OSError:
                continue
    raise RuntimeError("No free ports available in range!")


@contextmanager
def get_world_context(task_id: str, port: int = 8000):
    """Get AppWorld context for a specific task with remote APIs."""
    remote_environment_url = f"http://localhost:{port}"
    with AppWorld(
        task_id=task_id,
        remote_environment_url=remote_environment_url,
        experiment_name=f"strands_rl_{task_id}/{port}",
    ) as world:
        yield world


class AppWorldExecutor:
    """Maintains AppWorld state and provides a structured execute tool."""

    def __init__(self):
        self.world: Optional[Any] = None
        self._execution_count: int = 0

    def set_world(self, world):
        """Set the current AppWorld instance and reset execution count."""
        self.world = world
        self._execution_count = 0

    def get_execute_tool(self):
        """Return an execute tool function that uses the current world instance."""

        @tool
        def execute(code: str) -> str:
            """Execute Python code in the AppWorld environment.

            Args:
                code: Python code to execute in AppWorld

            Returns:
                JSON string with execution output and task completion status
            """
            if self.world is None:
                return json.dumps({"error": "No AppWorld instance set", "success": False})

            self._execution_count += 1

            try:
                output = self.world.execute(code)
                return cap_string(
                    json.dumps(
                        {
                            "output": output,
                            "task_completed": self.world.task_completed(),
                            "execution_count": self._execution_count,
                            "success": True,
                        }
                    )
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                return cap_string(
                    json.dumps(
                        {
                            "error_type": type(e).__name__,
                            "error": str(e),
                            "execution_count": self._execution_count,
                            "success": False,
                        }
                    )
                )

        return execute


def execute_appworld(agent, task_id: str, appworld_executor: AppWorldExecutor) -> dict:
    """Full lifecycle: start server, create world, invoke agent, evaluate, stop server.

    Returns the evaluation dict (contains 'passes', 'num_tests', etc.).
    """
    # Determine appworld binary path
    appworld_execute_env = os.environ.get("APPWORLD_EXECUTE_ENV", "")
    if appworld_execute_env:
        appworld_path = os.path.join(appworld_execute_env, "bin/appworld")
    else:
        appworld_path = "appworld"

    # Find a free port with random offset to reduce collisions under concurrency
    port_min = 6000 + random.randint(0, 100) * 10
    port = get_free_port(port_min)
    logger.info("Using AppWorld environment server port: %d", port)

    shell_cmd = [appworld_path, "serve", "environment", "--port", str(port)]
    appworld_server_process = ManagedProcess(shell_cmd, "Uvicorn running on", 30)

    try:
        appworld_server_process.start()
        logger.info("Started AppWorld environment server on port %d", port)

        with get_world_context(task_id, port) as world:
            user_message = (
                f"Using these APIs, now generate code to solve the actual task:\n"
                f"Today's date is: {world.task.datetime}\n"
                f"My name is: {world.task.supervisor.first_name} {world.task.supervisor.last_name}. "
                f"My personal email is {world.task.supervisor.email} "
                f"and phone number is {world.task.supervisor.phone_number}.\n"
                f"Task: {world.task.instruction}"
            )
            appworld_executor.set_world(world)
            agent(user_message)
            world.save()
            evaluation: dict = world.evaluate().to_dict()
            result = {"id": world.task.id, **evaluation}
    finally:
        appworld_server_process.stop()
        logger.info("Stopped AppWorld environment server on port %d", port)

    return result
