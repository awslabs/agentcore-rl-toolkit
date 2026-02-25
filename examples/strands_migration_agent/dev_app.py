import logging
import time

from dotenv import load_dotenv
from models import InvocationRequest, RepoMetaData
from reward import MigrationReward
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands_tools import editor, shell
from utils import load_metadata_from_s3, load_repo_from_s3, setup_repo_environment

from agentcore_rl_toolkit import AgentCoreRLApp
from agentcore_rl_toolkit.frameworks.strands.vllm_model import vLLMModel

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
    + "Tips:\n- When executing maven commands, try to keep the logs concise by filtering "
    + "out unuseful information. For example, use the `-ntp` flag (e.g., mvn -ntp clean verify) "
    + "to suppress download chatters; use `tail` or `head` to get logs progressively, "
    + "if needed (e.g. mvn -ntp clean verify | tail -n 50)."
)

reward_fn = MigrationReward()


@app.rollout_entrypoint
def invoke_agent(payload: dict):
    base_url = payload["_rollout"]["base_url"]
    model_id = payload["_rollout"]["model_id"]
    params = payload["_rollout"].get("sampling_params", {})

    model = vLLMModel(client_args={"api_key": "EMPTY", "base_url": base_url}, model_id=model_id, params=params)

    agent = Agent(
        model=model,
        tools=[shell, editor],
        system_prompt=system_prompt,
        conversation_manager=NullConversationManager(),
    )

    request = InvocationRequest(**payload)
    metadata = RepoMetaData(**load_metadata_from_s3(request.metadata_uri))

    start_time = time.time()
    repo_path = load_repo_from_s3(request.repo_uri)
    load_duration = time.time() - start_time
    logger.info(f"Loaded repo into: {repo_path} (took {load_duration:.2f}s)")

    start_time = time.time()
    setup_repo_environment(repo_path)
    setup_duration = time.time() - start_time
    logger.info(f"Finished repo setup for: {repo_path} (took {setup_duration:.2f}s)")

    user_input = request.prompt.format(
        repo_path=repo_path,
        num_tests=metadata.num_test_cases,
    )
    logger.info(f"User input: {user_input}")

    response = agent(user_input)

    logger.info(f'Model response: {response.message["content"][0]["text"]}')

    rollout_data = model.get_token_data()

    reward = reward_fn(
        repo_dir=repo_path,
        original_num_tests=metadata.num_test_cases,
        original_commit_id=metadata.base_commit,
        require_maximal_migration=request.require_maximal_migration,
    )

    return {"rollout_data": rollout_data, "rewards": reward}


if __name__ == "__main__":
    app.run()
