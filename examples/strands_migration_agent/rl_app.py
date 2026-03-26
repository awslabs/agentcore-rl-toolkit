import logging
import time

from dotenv import load_dotenv
from java_migration_agent.tools.dependency_tools import search_dependency_version
from models import InvocationRequest, RepoMetaData
from reward import MigrationReward
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import editor, shell
from utils import load_metadata_from_s3, load_repo_from_s3, setup_repo_environment

from agentcore_rl_toolkit import AgentCoreRLApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = AgentCoreRLApp()

load_dotenv()

system_prompt = (
    "You are a coding agent that helps to migrate repos written in Java8 to Java17. "
    + "To successfully migrate the repo, your goal is to:\n"
    + "- Get `mvn clean verify` to pass without errors after migrating to Java17.\n"
    + "- Make sure the major version of all compiled .class files is 61 (Java17).\n"
    + "- Pass all tests. Preserve the number of test cases as well as their "
    + "functional equivalence as the original repo in Java8, which means no additional "
    + "test should be ignored, skipped or disabled for the purpose of this migration.\n"
    + "Do not perform any work outside the repository folder the user provides.\n"
    + "Rules:\n"
    + "- Always use the `-ntp` flag with Maven to suppress download logs.\n"
    + "- Always pipe Maven output through `tail -n 100` to limit output size. "
    + "Example: mvn -ntp clean verify 2>&1 | tail -n 100\n"
    + "- If you need to see earlier output, run a separate command with `head -n 100`.\n"
    + "- When you have finished the task, generate a paragraph summarizing the changes you made "
    + "without using any tools.\n"
)

reward_fn = MigrationReward()


@app.rollout_entrypoint
def invoke_agent(payload: dict):
    base_url = payload["_rollout"]["base_url"]
    model_id = payload["_rollout"]["model_id"]
    agent_type = payload.get("agent_type", "baseline")
    params = payload["_rollout"].get("sampling_params", {})
    tools = [shell, editor]

    request = InvocationRequest(**payload)
    prompt = system_prompt
    if request.require_maximal_migration:
        prompt += (
            "\nYou should update all dependencies in the `pom.xml` file to their latest versions that support Java 17."
        )
    if agent_type == "rag":
        prompt += (
            "\nYou have access to a dependency version lookup tool. When updating dependencies "
            "in pom.xml:\n"
            "1. Use the search_dependency_version tool to look up the recommended Java 17 "
            "compatible version for each dependency\n"
            "2. If a dependency is not found in the database, use your knowledge to select "
            "an appropriate version\n"
            "3. Update all dependencies to their Java 17 compatible versions"
        )
        tools.append(search_dependency_version)
    elif agent_type == "hybrid":
        prompt += (
            "\nDependencies in the `pom.xml` file have been updated to their "
            "latest versions that support Java 17, but these changes might introduce "
            "compatibility issues in the codebase. Please fix any such issues in your "
            "migration. Do not downgrade the dependency versions back to their JDK 8 "
            "compatible versions."
        )

    model = OpenAIModel(client_args={"api_key": "EMPTY", "base_url": base_url}, model_id=model_id, params=params)

    agent = Agent(
        model=model,
        tools=tools,
        system_prompt=prompt,
    )

    metadata = RepoMetaData(**load_metadata_from_s3(request.metadata_uri))

    start_time = time.time()
    repo_path = load_repo_from_s3(request.repo_uri)
    load_duration = time.time() - start_time
    logger.info(f"Loaded repo into: {repo_path} (took {load_duration:.2f}s)")

    start_time = time.time()
    setup_repo_environment(repo_path, agent_type)
    setup_duration = time.time() - start_time
    logger.info(f"Finished repo setup for: {repo_path} (took {setup_duration:.2f}s)")

    user_input = request.prompt.format(
        repo_path=repo_path,
        num_tests=metadata.num_test_cases,
    )
    logger.info(f"User input: {user_input}")

    response = agent(user_input)

    logger.info(f'Model response: {response.message["content"][0]["text"]}')

    reward = reward_fn(
        repo_dir=repo_path,
        original_num_tests=metadata.num_test_cases,
        original_commit_id=metadata.base_commit,
        require_maximal_migration=request.require_maximal_migration,
    )

    return {"rewards": reward}


if __name__ == "__main__":
    app.run()
