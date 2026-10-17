# SageMaker Backend — Setup Guide

> **Networking:** this loop hosts the rollout gateway on the machine you run it on
> and hands the agent an `http://<private-ip>:<gateway_port>/v1` `base_url`, so every
> model call is an inbound connection from ACR to this host. Run it on an EC2 instance
> in a VPC, deploy the ACR runtime into that same VPC (`--vpc --subnets --security-groups`),
> and allow inbound TCP on `gateway_port`. **Running from a laptop does not work** — the
> runtime has no route to it, and rollouts silently score `0.0`. See the "Networking"
> section of the [SageMaker backend setup guide](../../../../../docs/site/src/content/docs/guides/sagemaker-backend-setup.md).

## 0. Deploy your agent to AgentCore Runtime (ACR)

Build and deploy an `AgentCoreRuntimeApp` agent to Bedrock AgentCore Runtime; `AgentCoreRLApp` (`rl_app.py`) agents are not supported by this backend. In this instruction, we take `examples/strands_math_agent/runtime_app.py` as an example; see the "Run Runtime App (No S3)" section of its `README.md` to deploy it.

## 1. Install agentcore-rl-toolkit and SageMaker training SDK

```bash
cd /path/to/agentcore-rl-toolkit
uv venv --python=3.12
source .venv/bin/activate

uv pip install sagemaker-train
uv pip install -e ".[rollout,gateway]"
uv pip install transformers==5.12
```

## 2. Prepare training dataset
Scripts for preparing AgentCore-compatible dataset are in `src/agentcore_rl_toolkit/backends/experimental/sagemaker/prepare_datasets`. For example, to prepare a `gsm8k` dataset, run:
```bash
python src/agentcore_rl_toolkit/backends/experimental/sagemaker/prepare_datasets/prepare_gsm8k.py \
    --output-dir /path/to/data/gsm8k
```

This writes `gsm8k_agent_{train,test}.parquet` — point `dataset_path` / `eval_dataset_path`
at them in step 3.

## 3. Set up config

```bash
cp config.yaml.example config.yaml
# fill in: role_arn, s3_output_path, model_package_group_arn,
#          base_model_arn, agent_runtime_arn, dataset_path, eval_dataset_path
```

## 4. Run the training

```bash
cd src/agentcore_rl_toolkit/backends/experimental/sagemaker
python train_grpo.py --config config.yaml
```
