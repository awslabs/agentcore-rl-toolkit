from dataclasses import dataclass
from typing import Literal

import yaml


@dataclass
class Config:
    # --- SageMaker / model ---
    region: str = "us-east-1"
    base_model_arn: str = ""
    role_arn: str = ""
    s3_output_path: str = ""
    model_package_group_arn: str = ""
    lora_rank: int = 32
    lora_alpha: int = 64
    resume_from_model_package: str = ""

    # --- ACR / rollout ---
    agent_runtime_arn: str = ""
    s3_bucket: str = ""
    exp_id: str = ""
    max_rollout_time: float = 1800.0
    acr_tps_limit: int = 5
    gateway_port: int = 0

    # --- RL training ---
    batch_size: int = 32
    responses_per_prompt: int = 4
    learning_rate: float = 1e-5
    max_context_tokens: int = 32768
    max_new_tokens: int = 2048
    temperature: float = 1.0
    epochs: int = 1
    save_every: int = -1
    loss: Literal["ppo", "importance_sampling", "cispo"] = "ppo"
    save_weights_at_end: bool = False

    # --- Dataset ---
    dataset_path: str = ""
    max_prompts: int = 0

    # --- Evaluation ---
    eval_dataset_path: str = ""
    eval_every: int = 0
    eval_max_prompts: int = 200
    eval_temperature: float = 0.0

    # --- Logging ---
    wandb_project: str = ""
    wandb_run_name: str = ""
    rollout_log_path: str = ""  # JSONL file for per-rollout diagnostics; auto-named if empty


def load_config(path: str) -> Config:
    with open(path) as f:
        raw: dict = yaml.safe_load(f) or {}

    fields = Config.__dataclass_fields__
    return Config(**{k: v for k, v in raw.items() if k in fields})
