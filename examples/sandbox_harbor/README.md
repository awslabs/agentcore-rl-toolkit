# Harbor on AgentCore Runtime

Harbor support easy evaluation on popular benchmarks on infra such as Daytona, Modal, LangSmith, Blaxel, and Novita Sandbox:
```
harbor run -d "<dataset@version>" -m "<model>" -a "<agent>"
```
This example shows how to enable similar, convenient evaluation on Harbor benchmarks on AgentCore Runtime.

one command that evaluates a whole [Harbor](https://harborframework.com) benchmark on AgentCore Runtime:

```bash
uv run python bench.py --benchmark tmax/TMax-15K-Harbor --task-root ./tasks \
                       --agent claude-code --model us.anthropic.claude-sonnet-4-6
```

That runs one rollout per task: for each task the agent gets its own fresh
sandbox, does the work, and is graded by the task's own tests. You get a solve
rate and one result record per task.

## How it fits together

A Harbor benchmark is a folder of tasks. Each task ships a container image, an
instruction, and hidden tests. Evaluating it on AgentCore Runtime takes three
pieces, all in this folder:

- **`harbor_sandbox/`** — turns each task into a runnable sandbox. `build.py`
  packages every task image and pushes it to ECR; `HarborSandboxClient` creates
  (and later removes) a task's runtime on demand.
- **`harness/`** — the agent. Two interchangeable ones: `strands` and
  `claude-code` (Claude Code co-located in the box). A harness is deployed once
  as a long-lived runtime and reused for every task.
- **`bench.py`** — the entrypoint above. It hands each task to the harness,
  collects results, and prints the summary.

## Setup

```bash
uv sync    # installs this example (and its harbor_sandbox package) into ./.venv
cp harbor_sandbox/.env.example harbor_sandbox/.env   # then fill in your account/region/bucket
```

`.env` holds your account-specific values (git-ignored, never committed); it is
read by `harbor_sandbox/config.py` and sourced by `harness/build_push.sh`.

## Steps

1. **Build the task images** (once per benchmark):

   ```bash
   uv run python -m harbor_sandbox.build --task-root ./tasks \
          --benchmark tmax/TMax-15K-Harbor --arch arm64
   ```

2. **Deploy an agent harness** (once):

   ```bash
   (cd harness && ./build_push.sh claude_code)
   uv run python harness/deploy_harness.py --agent claude-code --image-tag claude-code
   ```

3. **Run the benchmark**:

   ```bash
   uv run python bench.py --benchmark tmax/TMax-15K-Harbor --task-root ./tasks \
          --agent claude-code --model us.anthropic.claude-sonnet-4-6
   ```

## Runtime

Tasks run on the serverless arm64 microVM substrate — fast cold start, PUBLIC
network. Build the task images for arm64 (`--arch arm64`); that is the substrate
this example targets end to end. For now, AgentCore Runtime microVM only support arm64, with upcoming support on x86 microVM, we'll update to support once the infra is available.

## Notes

- The harness needs no credentials injected: each task runtime uses its own IAM
  role.
- By default a task's runtime is removed right after its rollout, so a large
  benchmark only ever holds a handful of runtimes at once. Alternatively, one can just raise up AgentCore Runtime quota, and keep the runtime deployed to achieve a faster start.
- Restrict a run with `--tasks` / `--exclude` (a comma list or `@file`) and
  `--limit`.
- `bench.py` reads tasks from `--task-root`; if that dir is empty it auto-pulls
  them from the Harbor registry, which needs the optional `harbor` package
  (`uv sync --extra harbor`). With the tasks already on disk this is skipped.

Known limits on the terminal-bench-2 benchmark are tracked in [TODO.md](TODO.md).
