"""Local test script for OfficeBench agent — bypasses ACR/S3."""

import argparse
import json
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TESTBED_DIR = "/tmp/testbed"

SYSTEM_PROMPT = (
    "You are an AI office assistant for user {username}. "
    "Today is {date} ({weekday}). The current time is {time}.\n\n"
    "Help solve the office task using the available tools.\n"
    "Task files are in {testbed}/data/. Calendar files are in {testbed}/calendar/. "
    "Email files are in {testbed}/emails/.\n\n"
    "Important:\n"
    "- Use absolute paths starting with {testbed}/ for all file operations.\n"
    "- When creating new files, place them under {testbed}/data/ unless specified otherwise.\n"
    "- When the task is complete, stop calling tools and summarize what you did."
)


def setup_local_testbed(task_dir: str):
    """Set up testbed from local OfficeBench task directory."""
    if os.path.exists(TESTBED_DIR):
        shutil.rmtree(TESTBED_DIR)

    testbed_src = os.path.join(task_dir, "testbed")
    if os.path.exists(testbed_src):
        shutil.copytree(testbed_src, TESTBED_DIR)
        logger.info(f"Copied testbed from {testbed_src}")
    else:
        os.makedirs(TESTBED_DIR, exist_ok=True)
        logger.info("No testbed data, created empty directory")

    # Ensure standard directories exist
    os.makedirs(os.path.join(TESTBED_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(TESTBED_DIR, "calendar"), exist_ok=True)
    os.makedirs(os.path.join(TESTBED_DIR, "emails"), exist_ok=True)

    # Copy reference/ files for evaluate_exact_match
    reference_src = os.path.join(task_dir, "reference")
    if os.path.exists(reference_src):
        shutil.copytree(reference_src, os.path.join(TESTBED_DIR, "reference"))
        logger.info(f"Copied reference/ from {reference_src}")

    # Create cache/ snapshot of original data for evaluate_diff_contain_text
    data_src = os.path.join(TESTBED_DIR, "data")
    if os.path.exists(data_src) and os.listdir(data_src):
        cache_data_dir = os.path.join(TESTBED_DIR, "cache", "data")
        shutil.copytree(data_src, cache_data_dir)
        logger.info("Created cache/ snapshot of original data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, default="1-1", help="Task ID (e.g. 1-1)")
    parser.add_argument("--subtask_id", type=str, default="0", help="Subtask ID (e.g. 0, 1, 2)")
    parser.add_argument(
        "--model_id",
        type=str,
        default="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help="Bedrock model ID",
    )
    args = parser.parse_args()

    # Configure local paths BEFORE importing tools (they read env at import time)
    officebench_dir = os.environ.get("OFFICEBENCH_DIR", os.path.expanduser("~/OfficeBench"))
    os.environ["OFFICEBENCH_APPS_DIR"] = os.path.join(officebench_dir, "apps")
    os.environ["OFFICEBENCH_TESTBED_DIR"] = TESTBED_DIR
    os.environ["BYPASS_TOOL_CONSENT"] = "true"

    from reward import OfficeBenchReward
    from strands import Agent
    from strands.agent.conversation_manager import NullConversationManager
    from strands.models import BedrockModel
    from strands_tools import shell
    from tools import ALL_TOOLS

    task_dir = os.path.join(officebench_dir, "tasks", args.task_id)
    config_path = os.path.join(task_dir, "subtasks", f"{args.subtask_id}.json")

    with open(config_path) as f:
        task_config = json.load(f)

    logger.info(f"Task {args.task_id}/{args.subtask_id}: {task_config['task']}")

    setup_local_testbed(task_dir)

    model = BedrockModel(model_id=args.model_id)

    system_prompt = SYSTEM_PROMPT.format(
        username=task_config.get("username", "User"),
        date=task_config.get("date", "unknown"),
        weekday=task_config.get("weekday", "unknown"),
        time=task_config.get("time", "unknown"),
        testbed=TESTBED_DIR,
    )

    agent = Agent(
        model=model,
        tools=[shell, *ALL_TOOLS],
        system_prompt=system_prompt,
        conversation_manager=NullConversationManager(),
    )

    response = agent(task_config["task"])
    response_text = response.message["content"][0]["text"]
    logger.info(f"Agent response: {response_text}")

    reward_fn = OfficeBenchReward()
    reward = reward_fn(testbed_dir=TESTBED_DIR, evaluation_config=task_config["evaluation"])
    logger.info(f"Reward: {reward}")

    for root, _dirs, files in os.walk(TESTBED_DIR):
        for f in files:
            path = os.path.join(root, f)
            logger.info(f"  Testbed file: {path} ({os.path.getsize(path)} bytes)")

    print(f"\nResult: {'PASS' if reward == 1.0 else 'FAIL'} (reward={reward})")


if __name__ == "__main__":
    main()
