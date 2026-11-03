# Setup

## 1. Create a Python 3.12 venv

```bash
cd /path/to/agentcore-rl-toolkit
uv venv --python 3.12
```

## 2. Activate it

```bash
source .venv/bin/activate
```

## 3. Install dependencies

From the repo root (the install script clones `Megatron-LM` and `slime` there):

```bash
bash src/agentcore_rl_toolkit/backends/slime/scripts/install_slime.sh
```

## 4. Configure

```bash
cd src/agentcore_rl_toolkit/backends/slime/examples/math_agent
cp config.yaml.example config.yaml
# edit config.yaml: set rollout_session_backend.agentcore_runtime_arn, etc.
```

## 5. Train

```bash
bash train.sh
```

## Score centering

slime implements score centering ([arXiv:2609.20807](https://arxiv.org/abs/2609.20807))
behind `--use-score-centering`. With `--rollout-top-p 1` it centers each token's score over
the sampler's top-`--score-centering-top-k` head and approximates the remaining tail. The
gateway requests that head from SGLang for every model turn, and `generate` hands it to
slime with the rest of the `Sample`. slime's exact top-p mode (`--rollout-top-p` below 1) is
not supported through the gateway.

Train with [`train_score_centering.sh`](train_score_centering.sh) instead of `train.sh`. It
adds slime's score centering flags (`--use-score-centering --pg-loss-type reinforce` with
TIS) to the same colocated setup.

slime's fully-async rollout (`--flush-cache-interval 0` on separate rollout GPUs) cannot
run with in-training eval at slime `540ea919`: eval and the fully-async worker share
`GenerateState`'s `asyncio.Semaphore` from two event loops, and the worker fails the job
at the first eval. Drop the `--eval-*` flags to use it. Its default `--flush-cache-interval 1`
also aborts in-flight SGLang requests at each weight sync, and the agent continues from the
truncated turn.

Each turn's sampling params must match the rollout config (`--rollout-temperature`,
`--rollout-top-p 1`, `--rollout-top-k -1`, no penalties). The gateway rejects a turn
whose agent overrides `temperature`, `top_p`, or `top_k` with HTTP 400.
