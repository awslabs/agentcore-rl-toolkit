"""Drop-in replacement for HERB's utils.get_gpt4_response, backed by Claude Haiku 5.5 on Bedrock.

Haiku 5.5 rejects the temperature parameter, so sampling uses the service default (the official GPT-4o call
used 0.00001).
The official helper calls GPT-4o (temperature 0.00001; JSON mode when response_format="json") for the structured
tools' entity extraction, the finish_agent check, and the evaluation judge. Haiku has no JSON mode, so the JSON
object is taken from the reply (code fences stripped). To stand in for JSON mode, json requests add a system
instruction asking for a valid JSON object and are retried once if the reply does not parse; as in the official
helper, a reply that still does not parse returns None.
"""

import ast
import json
import re
import time

import boto3
from botocore.config import Config

MODEL = "us.anthropic.claude-haiku-5-5"
JSON_SYSTEM = "Return only a valid JSON object (double-quoted keys and strings), with no other text."
_client = boto3.client(
    "bedrock-runtime",
    region_name="us-west-2",
    config=Config(read_timeout=300, retries={"max_attempts": 1}, max_pool_connections=64),
)


def _parse_json(text):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            return None
        for parse in (json.loads, ast.literal_eval):  # the prompts show single-quoted pseudo-JSON
            try:
                v = parse(m.group(0))
                return v if isinstance(v, dict) else None
            except (ValueError, SyntaxError):
                continue
        return None


def get_gpt4_response(messages, model_name=None, temperature=0.00001, response_format="text"):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
    }
    if response_format == "json":
        body["system"] = JSON_SYSTEM
    parse_retries = 1
    for attempt in range(6):
        try:
            r = _client.invoke_model(modelId=MODEL, body=json.dumps(body))
            out = json.loads(r["body"].read())
            text = "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
            if response_format != "json":
                return text.strip()
            v = _parse_json(text)
            if v is None and parse_retries > 0:
                parse_retries -= 1
                continue
            return v
        except _client.exceptions.ThrottlingException:
            time.sleep(5 * (attempt + 1))
        except Exception:  # noqa: BLE001
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("helper LLM: retries exhausted")
