import logging
from typing import Any, Optional

from tau2.data_model.tasks import EnvAssertion, EnvFunctionCall

from agentcore_rl_toolkit import RewardFunction

logger = logging.getLogger(__name__)


class TauBenchReward(RewardFunction):
    """
    Computes tau-bench reward based on task's reward_basis.

    Args:
        get_environment_func: dict mapping domain name -> environment constructor callable
    """

    def __init__(self, get_environment_func: dict):
        self.get_environment_func = get_environment_func

    def __call__(
        self, env: Any, task: dict, domain: str, messages: Optional[list[dict]] = None, **kwargs: Any
    ) -> tuple[float, dict]:
        """
        Compute reward for a completed tau-bench rollout.

        Args:
            env: The post-rollout Environment (already mutated by agent's tool calls)
            task: Task dict from the payload (raw JSON, not Pydantic)
            domain: Domain name ("airline", "retail", "telecom")
            messages: Global message list in Strands format (for COMMUNICATE and ACTION checking)

        Returns:
            tuple: (reward: float, reward_info: dict)
                - reward: scalar in [0, 1] based on reward_basis
                - reward_info: detailed breakdown with db_check, communicate_checks,
                  env_assertion_checks, action_checks, reward_basis, reward_breakdown
        """
        eval_criteria = task.get("evaluation_criteria") or {}
        reward_basis = eval_criteria.get("reward_basis")
        # Default to DB only
        if not reward_basis:
            reward_basis = ["DB"]

        reward = 1.0
        reward_breakdown = {}
        reward_info = {"reward_basis": reward_basis}

        if "DB" in reward_basis:
            db_reward, db_check = self._compute_db_reward(env, task, domain)
            reward *= db_reward
            reward_breakdown["DB"] = db_reward
            reward_info["db_check"] = db_check

        if "COMMUNICATE" in reward_basis:
            comm_reward, comm_checks = self._compute_communicate_reward(eval_criteria, messages or [])
            reward *= comm_reward
            reward_breakdown["COMMUNICATE"] = comm_reward
            reward_info["communicate_checks"] = comm_checks

        if "ENV_ASSERTION" in reward_basis:
            env_reward, env_checks = self._compute_env_assertion_reward(env, eval_criteria)
            reward *= env_reward
            reward_breakdown["ENV_ASSERTION"] = env_reward
            reward_info["env_assertion_checks"] = env_checks

        # ACTION axis: count toward reward when listed in reward_basis;
        # otherwise still compute and surface as diagnostic-only.
        action_checks = self._compute_action_checks(eval_criteria, messages or [])
        reward_info["action_checks"] = action_checks
        if "ACTION" in reward_basis:
            action_reward = (
                1.0
                if action_checks and all(c["action_match"] for c in action_checks)
                else (1.0 if not action_checks else 0.0)
            )
            reward *= action_reward
            reward_breakdown["ACTION"] = action_reward
            logger.info("ACTION reward: %s", action_reward)

        reward_info["reward"] = reward
        reward_info["reward_breakdown"] = reward_breakdown
        return reward, reward_info

    def _compute_db_reward(self, env: Any, task: dict, domain: str) -> tuple[float, dict]:
        """
        Compare DB hashes between predicted env (live) and gold env (fresh + golden actions).

        The live `env` was mutated during the rollout via use_tool/sync_tools, so it already
        represents the predicted environment state — no need to replay the trajectory.

        Returns:
            tuple: (reward: float, db_check: dict)
        """
        eval_criteria = task.get("evaluation_criteria") or {}
        golden_actions = eval_criteria.get("actions") or []

        # Build gold environment: init + golden actions
        get_environment = self.get_environment_func[domain]
        gold_env = get_environment()

        # Replay initialization
        initial_state = task.get("initial_state") or {}
        for action in initial_state.get("initialization_actions") or []:
            gold_env.run_env_function_call(EnvFunctionCall(**action))

        # Run golden actions
        for action in golden_actions:
            try:
                gold_env.make_tool_call(
                    tool_name=action["name"],
                    requestor=action.get("requestor", "assistant"),
                    **action.get("arguments", {}),
                )
            except Exception as e:
                logger.warning("Gold action %s failed: %s", action["name"], e)

        # Compare agent DB hashes
        gold_agent_hash = gold_env.get_db_hash()
        pred_agent_hash = env.get_db_hash()
        agent_match = gold_agent_hash == pred_agent_hash

        # Compare user DB hashes (only telecom has user_tools; others return None)
        gold_user_hash = gold_env.get_user_db_hash()
        pred_user_hash = env.get_user_db_hash()
        if gold_user_hash is not None and pred_user_hash is not None:
            user_match = gold_user_hash == pred_user_hash
        else:
            user_match = True

        db_reward = 1.0 if (agent_match and user_match) else 0.0
        db_check = {
            "db_match": agent_match and user_match,
            "db_reward": db_reward,
            "agent_db_match": agent_match,
            "user_db_match": user_match,
        }
        logger.info("DB reward: %s (agent_match=%s, user_match=%s)", db_reward, agent_match, user_match)
        return db_reward, db_check

    def _compute_communicate_reward(self, eval_criteria: dict, messages: list[dict]) -> tuple[float, list[dict]]:
        """
        Check if agent communicated required info strings to the user.

        Matches tau-bench's evaluator_communicate.py: case-insensitive substring check
        with commas stripped from agent text.

        Returns:
            tuple: (reward: float, checks: list[dict])
        """
        communicate_info = eval_criteria.get("communicate_info") or []
        if not communicate_info:
            return 1.0, []

        # Collect all assistant text blocks from the conversation
        assistant_texts = []
        for m in messages:
            if m.get("role") == "assistant" or m.get("from") == "assistant":
                for block in m.get("content", []):
                    if "text" in block:
                        assistant_texts.append(block["text"].lower().replace(",", ""))

        checks = []
        all_met = True
        for info_str in communicate_info:
            found = any(info_str.lower() in text for text in assistant_texts)
            checks.append({"info": info_str, "met": found})
            if not found:
                all_met = False
                logger.warning("COMMUNICATE failed: '%s' not found in assistant responses", info_str)

        comm_reward = 1.0 if all_met else 0.0
        passed = sum(1 for c in checks if c["met"])
        logger.info("COMMUNICATE reward: %s (%d/%d passed)", comm_reward, passed, len(checks))
        return comm_reward, checks

    def _compute_env_assertion_reward(self, env: Any, eval_criteria: dict) -> tuple[float, list[dict]]:
        """
        Run env_assertions on the post-rollout environment.
        Reward is the product of all assertion results (all-or-nothing per assertion).

        Returns:
            tuple: (reward: float, checks: list[dict])
        """
        env_assertions = eval_criteria.get("env_assertions") or []
        if not env_assertions:
            return 1.0, []

        checks = []
        reward = 1.0
        for assertion_dict in env_assertions:
            assertion = EnvAssertion(**assertion_dict)
            success = env.run_env_assertion(assertion, raise_assertion_error=False)
            checks.append(
                {
                    "func_name": assertion.func_name,
                    "arguments": assertion.arguments,
                    "met": success,
                }
            )
            if not success:
                reward *= 0.0
                logger.warning("ENV_ASSERTION failed: %s(%s)", assertion.func_name, assertion.arguments)

        passed = sum(1 for c in checks if c["met"])
        logger.info("ENV_ASSERTION reward: %s (%d/%d passed)", reward, passed, len(checks))
        return reward, checks

    def _compute_action_checks(self, eval_criteria: dict, messages: list[dict]) -> list[dict]:
        """
        Check if agent made the expected tool calls (ACTION matching).
        Not used for reward — saved as diagnostics only.

        Returns:
            list[dict]: One entry per golden action with action details and match result.
        """
        golden_actions = eval_criteria.get("actions") or []
        if not golden_actions:
            return []

        # Extract tool calls from Strands-format messages, keyed by requestor.
        # Telecom user-side tools (e.g. toggle_airplane_mode) are emitted by the
        # user simulator and tagged from="user"; assistant-side tools are tagged
        # from="assistant". A golden action's `requestor` selects the matching pool.
        pred_tool_calls_by_requestor = {"assistant": [], "user": []}
        for m in messages:
            requestor = m.get("from")
            if requestor not in pred_tool_calls_by_requestor:
                continue
            for block in m.get("content", []):
                if "toolUse" in block:
                    tu = block["toolUse"]
                    pred_tool_calls_by_requestor[requestor].append(
                        {
                            "name": tu["name"],
                            "arguments": tu.get("input", {}),
                        }
                    )

        checks = []
        for gold_action in golden_actions:
            gold_name = gold_action["name"]
            gold_args = gold_action.get("arguments", {})
            compare_args = gold_action.get("compare_args")
            gold_requestor = gold_action.get("requestor", "assistant")
            candidate_pool = pred_tool_calls_by_requestor.get(gold_requestor, [])

            found = False
            for pred in candidate_pool:
                if pred["name"] != gold_name:
                    continue
                # Filter arguments by compare_args (None = check all pred args)
                if compare_args is None:
                    cmp_keys = pred["arguments"].keys()
                else:
                    cmp_keys = compare_args
                if len(cmp_keys) == 0:
                    found = True
                    break
                pred_filtered = {k: v for k, v in pred["arguments"].items() if k in cmp_keys}
                gold_filtered = {k: v for k, v in gold_args.items() if k in cmp_keys}
                if pred_filtered == gold_filtered:
                    found = True
                    break

            checks.append(
                {
                    "action": gold_action,
                    "action_match": found,
                    "action_reward": 1.0 if found else 0.0,
                }
            )

        passed = sum(1 for c in checks if c["action_match"])
        logger.info("ACTION checks (diagnostic): %d/%d matched", passed, len(checks))
        return checks
