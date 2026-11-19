"""llm_completion for the official evaluation: Claude Sonnet 4.5 on Bedrock, temperature 0, 10,240 output tokens
(the official default_eval_config: GPT-4.1, temperature 0, 10,240 tokens). It serves the primary-key and column
alignment and the llm_judge columns."""

import types

import boto3
from botocore.config import Config

MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
_bedrock = boto3.client("bedrock-runtime", config=Config(retries={"max_attempts": 8, "mode": "adaptive"}))


def llm_completion(messages, tools=None, model_config_name=None):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    prompt = "\n\n".join(m["content"] for m in messages)
    r = _bedrock.converse(
        modelId=MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 10240, "temperature": 0},
    )
    return types.SimpleNamespace(content="".join(b["text"] for b in r["output"]["message"]["content"] if "text" in b))
