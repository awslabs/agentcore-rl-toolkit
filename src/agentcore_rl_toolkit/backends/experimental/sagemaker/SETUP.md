# SageMaker Backend — Setup Guide

## 0. Deploy your agent to AgentCore Runtime (ACR)

Build and deploy your `rl_app.py` agent to Bedrock AgentCore Runtime. See `examples/strands_*` under the repository's root directory. Read their `REAME.md` files to build and deploy the agents to ACR. In this instruction, we take `examples/strands_math_agent` as an example.

## 1. Install agentcore-rl-toolkit and SageMaker training SDK

```bash
cd /path/to/agentcore-rl-toolkit
uv venv --python=3.12
source .venv/bin/activate

uv pip install sagemaker-train
uv pip install -e ".[gateway]"
uv pip install transformers==5.12
```

## 2. Prepare training dataset
Scripts for preparing AgentCore-compatible dataset are in `src/agentcore_rl_toolkit/backends/experimental/sagemaker/prepare_datasets`. For example, to prepare a `gsm8k` dataset, run:
```bash
python src/agentcore_rl_toolkit/backends/experimental/verl/examples/math_agent/preprocess_gsm8k.py \
    --output-dir /path/to/data/gsm8k
```



## 3. Set up config

```bash
cp config.yaml.example config.yaml
# fill in: role_arn, s3_output_path, model_package_group_arn, s3_bucket,
#          base_model_arn, agent_runtime_arn, dataset_path, eval_dataset_path
```

## 4. Run the training

```bash
cd src/agentcore_rl_toolkit/backends/experimental/sagemaker
python train_grpo.py --config config.yaml
```
