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
bash src/agentcore_rl_toolkit/backends/experimental/slime/scripts/install_slime.sh
```

## 4. Configure

```bash
cd src/agentcore_rl_toolkit/backends/experimental/slime/examples/math_agent
cp config.yaml.example config.yaml
# edit config.yaml: set acr_agent_runtime_arn, s3_bucket, etc.
```

## 5. Train

```bash
bash train.sh
```
