---
title: app
description: AgentCoreRLApp and the @rollout_entrypoint decorator.
sidebar:
  order: 0
---


_module: `agentcore_rl_toolkit.app`_

## `class AgentCoreRLApp(BedrockAgentCoreApp)`

**Constructor**

```python
AgentCoreRLApp()
```

### Methods

#### `rollout_entrypoint(func)`

Decorator for RL training that handles asyncio.create_task and result saving automatically.

This decorator:
1. Handles both sync and async user functions using BedrockAgentCoreApp's infrastructure
2. Automatically saves the returned dict to S3 when S3 config is present
3. Handles errors and saves error results for client notification
4. Returns immediately with {"status": "processing"} for non-blocking behavior

The return value must be a JSON-serializable dict when S3 save is configured.
Any dict structure is accepted — there are no required keys. Common patterns:
- RL training: {"rollout_data": [...], "rewards": [...]}
- Evaluation: {"rewards": [...], "metrics": {...}}
- Custom: {"summary": "...", "artifacts": {...}}

Serialization note: saved via json.dumps() → S3 as application/json.
Supported types: str, int, float, bool, None, list, dict.
Non-serializable values (custom objects, bytes, datetime, numpy arrays, etc.)
will trigger the error path and an error file will be saved to S3.

Reserved keys: `save_result` injects SDK metadata into the saved JSON.
See `save_result` docstring for the full list of reserved keys.

**Parameters**

- `func`: The user function that handles agent logic and result collection

**Returns**

- : Decorated function registered as entrypoint

#### `save_result(result: dict, rollout_config: dict, result_key: str, payload: dict = None)`

Save result data to S3.

The result dict is saved as-is with metadata added for correlation and debugging.
Any JSON-serializable dict is accepted — there are no required keys.

Reserved keys — the SDK injects the following keys into the saved JSON.
Avoid using these in your return dict to prevent unexpected overwrites:

- `status_code`: Set to 200 if not already present in the user dict.
- `stop_reason`: Set to `"end_turn"` if not already present.
- `input_id`: Always overwritten with the value from rollout config.
- `s3_bucket`: Always overwritten with the value from rollout config.
- `result_key`: Always overwritten with the computed S3 key.
- `payload`: Always overwritten with the original request payload.

**Parameters**

- `result` *(dict)*: The result data to save (any JSON-serializable dict)

- `rollout_config` *(dict)*: Rollout configuration dict containing:
- s3_bucket: S3 bucket name
- exp_id: Experiment ID for organizing data
- input_id: id for discriminating different input data examples

- `payload` *(dict)* — default `None`: Original request payload (included in saved result for debugging)

- `result_key` *(str)*: S3 key for the result (computed externally for consistency)

### Attributes

- `s3_client`
