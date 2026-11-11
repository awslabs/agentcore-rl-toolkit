# SWE-Grep agent (AgentCore container)

The code deployed to Amazon Bedrock AgentCore Runtime. Each task is one AgentCore session: the trainer
(or `../eval/full_benchmark.py`) sends a task row, and the agent searches a checkout of
`repo@base_commit` with three tools, answers with code regions and returns its reward.

| File | |
|---|---|
| `rl_app.py` | The entrypoint. Returns `{"reward", "metrics", "regions", "messages"}`, or `{"reward": 0, "error"}` if the model was never reached. `eval/claude_bench.py` calls its `run_episode` directly. |
| `rollout/agent_prompts.py` | The system prompt and the final-answer nudge. |
| `rollout/repo_tools.py` | `grep(pattern, path)`, `glob(pattern)` and `read(path, start_line, end_line)` over the checkout. |
| `rollout/turn_cap.py` | Stops the search after 15 tool-calling turns or at 44k context tokens, then asks for the answer. |
| `rollout/custom_openai_model.py` | Sends earlier turns back as generated: reasoning kept and tool names restored. |
| `rollout/repo_store.py` | Downloads the checkout from S3 (uploaded by `../scripts/prepare_corpus.py`); the runtime has no internet access. |
| `scoring/` | Answer parsing and the reward. `../eval/` and `../data/` import it too. |

The prompt, tool limits and turn budget are the ones the published model was trained with; changing
them changes what the policy sees.

## Answer

The final message lists regions, most useful first:

```text
<answer>
src/router.py:120-165
tests/test_router.py:40-68
</answer>
```

Each entry is a repository-relative path and a 1-indexed inclusive range (`path:line` is one line).
Entries for missing files are dropped as invalid, and ranges past the end of a file are clipped.

## Reward

```text
model never reached                   ->  0     (an infrastructure error; a penalty would skew the group)
no valid region, rollout crashed      -> -1
no valid region                       -> -0.3
otherwise                             ->  F-beta(line precision, line recall)
                                          - 0.05 per distinct invalid region (at most 0.3)
```

Precision and recall count distinct lines against the task's `gold_lines`. Beta comes from the
recipe's `beta` (2 for the final model), which `../cdk/deploy.sh` sets on the runtime as `SWEGREP_BETA`.
