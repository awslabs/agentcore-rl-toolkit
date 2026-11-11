# SWE-Grep

Reinforcement learning for code search. SWE-Grep trains Qwen3.5-9B to find the code that needs to
change for a GitHub issue. The agent searches a repository with `grep`, `glob` and `read`, then answers
with ranked file regions.

The agent runs on AWS Bedrock AgentCore. Training uses the
[AgentCore RL toolkit](https://github.com/awslabs/agentcore-rl-toolkit) against a local
[SkyRL](https://github.com/NovaSky-AI/SkyRL) Tinker endpoint.

## Results

Line-level F1, precision and recall, with one attempt per task:

| Model | SWE-Explore F1 | P | R | ContextBench Lite F1 | P | R |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B base | 17.5 | 58.5 | 14.1 | 28.0 | 37.0 | 36.8 |
| Claude Sonnet 5 | 22.6 | 47.0 | 21.4 | 28.5 | 25.2 | 54.8 |
| Claude Sonnet 5.5 | 19.1 | 61.2 | 14.4 | 36.8 | 41.9 | 51.0 |
| **Qwen3.5-9B + RL** | **30.5** | 77.7 | 24.7 | 36.0 | 39.1 | 53.1 |

RL adds +13.0 F1 on SWE-Explore and +8.0 on ContextBench Lite. To reproduce the Claude rows, see
[Benchmark Claude](#benchmark-claude).

## Recipe

| | |
|---|---|
| Data | 1,098 training and 100 validation issues from SWE-rebench and SWE-rebench-V2, in 18 languages, sharing no repository or commit with either benchmark |
| Labels | Reference patch hunks ±20 lines, source files only |
| Reward | Line-level F-β (β = 2), −0.3 for no answer, small penalty for invalid regions |
| Training | LoRA rank 64, learning rate 8e-5, 12 tasks × 16 rollouts, 30 steps, 64k context, 16 agent turns |
| Evaluation | [SWE-Explore](https://arxiv.org/abs/2606.07297) (848 tasks) and [ContextBench](https://huggingface.co/datasets/Contextbench/ContextBench) Lite (500 tasks) |

## Requirements

- 8 NVIDIA GPUs with 80 GB each (tested on 8× H100)
- Docker with the NVIDIA container toolkit
- [uv](https://docs.astral.sh/uv/), Node.js and `jq`
- An AWS account with Bedrock AgentCore, with this host in a VPC the stack can attach to

## Quickstart

Run every command from `examples/strands_swegrep_agent`.

```bash
# 1. Install
uv sync --frozen --extra merge

# 2. Build the data; both commands fail unless the hashes match
uv run python data/build_training_data.py --check
uv run python data/build_benchmarks.py --check

# 3. Deploy the agent with the final recipe's settings (writes .env)
cp cdk/config.example.yaml cdk/config.yaml    # account, region, VPC, security group; see cdk/README.md
bash cdk/deploy.sh

# 4. Upload repository snapshots; rerun if GitHub throttles, finished ones are skipped
uv run python scripts/prepare_corpus.py --workers 16

# 5. Install and start the SkyRL endpoint (set HF_TOKEN for a more reliable model download)
bash scripts/endpoint.sh setup
bash scripts/endpoint.sh start

# 6. One-step test run (a few minutes; prints PASS or FAIL)
bash scripts/test_run.sh

# 7. Train, export, merge and benchmark (several hours)
bash scripts/run_recipe.sh recipes/final.json final
```

Results are written to `evals/final/eval/RESULTS.txt`. If step 7 stops, rerun it: it skips the stages that
finished, and training continues from its last checkpoint (saved every 5 steps).

Set `WANDB_API_KEY` in `.env` to log to W&B.

The SkyRL endpoint listens on 127.0.0.1:18080; set `TINKER_PORT` (exported or in `.env`) to change the port.

## Compare with the base model

Benchmark the base model; its scores go to `evals/base/eval/RESULTS.txt`, next to the recipe run's
`evals/final/eval/RESULTS.txt`:

```bash
bash eval/benchmark_base.sh
```

## Ablations

Each recipe in `recipes/ablations/` changes one setting of `final.json`. The reward's beta is set on the deployed
agent, so redeploy it with the ablation's recipe before training:

```bash
bash cdk/deploy.sh -c recipe=recipes/ablations/beta1.json
bash scripts/run_recipe.sh recipes/ablations/beta1.json beta1
```

`raw-patch.json` trains on the raw patch hunks; build its data first with
`uv run python data/build_training_data.py --window 0`. Afterwards, run `bash cdk/deploy.sh` again to restore
the final recipe's beta.

## Benchmark Claude

`eval/claude_bench.py` runs the same agent on this host with Claude on Amazon Bedrock as the model; nothing
else differs. It needs Bedrock access to the model. All 1,348 tasks with Sonnet 5.5 cost about $60 at list
prices and take under an hour.

```bash
bash eval/benchmark_claude.sh                                   # Sonnet 5.5
bash eval/benchmark_claude.sh global.anthropic.claude-sonnet-5  # or any other Bedrock model ID
```

Scores go to `evals/<model>/eval/RESULTS.txt`, for example `evals/claude-sonnet-5-5/eval/RESULTS.txt`.
Rerunning resumes and retries failed tasks.

## Layout

| Path | Purpose |
|---|---|
| `agent/` | AgentCore agent: tools, prompt, answer parser and reward ([details](agent/README.md)) |
| `cdk/` | AWS stack ([details](cdk/README.md)) |
| `data/` | Data builders and manifests |
| `endpoint/` | SkyRL image and serve scripts |
| `eval/` | Benchmark harness, scorer and the Claude runner |
| `recipes/` | `final.json` and the ablations |
| `scripts/` | Endpoint control, training, corpus build, adapter export and merge |
| `swegrep_train.py` | The toolkit's Tinker loop plus recipes, the agent's metrics and resume |
