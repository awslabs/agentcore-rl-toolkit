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

## Score function centering

[`score_centering.py`](../../integration/score_centering.py) implements
[Tinker's StableReinforce](https://tinker-docs.thinkingmachines.ai/tinker/losses/stable-reinforce/)
loss. SGLang draws the samples through the `sglang_token_samples` plugin, registered as a
`sglang.srt.plugins` entry point (reinstall the package after pulling).

Train with [`train_sfc.sh`](train_sfc.sh) instead of `train.sh`. It runs slime's `train_async.py` (separate
`TRAINER_GPUS` and `ROLLOUT_GPUS`) through
`python3 -m agentcore_rl_toolkit.backends.slime.train`, which registers the
`--sfc-num-auxiliary-token-samples`, `--sfc-kl-beta`, and `--sfc-loss-agg-mode` flags, and
sets the loss, the init hook, and `AGENTCORE_SGLANG_TOKEN_SAMPLES=1` for the SGLang engines.

Sampling must be untruncated (`--rollout-top-p 1.0`, `--rollout-top-k -1`, and no
`top_p`/`top_k` from the agent); the gateway fails turns sampled otherwise.
