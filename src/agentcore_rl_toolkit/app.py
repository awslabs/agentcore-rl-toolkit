import asyncio
import json
import logging
import traceback
import uuid
from dataclasses import dataclass
from functools import wraps

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp

_S3_CONFIG_FIELDS = ("exp_id", "input_id", "s3_bucket")


@dataclass
class RolloutConfig:
    """Rollout configuration for rollout collection and storage."""

    exp_id: str
    input_id: str
    s3_bucket: str

    @classmethod
    def from_dict(cls, data: dict) -> "RolloutConfig":
        """Create RolloutConfig from dictionary with validation."""
        try:
            return cls(
                exp_id=data["exp_id"],
                input_id=data["input_id"],
                s3_bucket=data["s3_bucket"],
            )
        except KeyError as e:
            raise ValueError(f"Missing required rollout config field: {e}") from e


class AgentCoreRLApp(BedrockAgentCoreApp):
    def __init__(self):
        super().__init__()
        self.s3_client = boto3.client("s3")

    def save_result(self, result: dict, rollout_config: dict, result_key: str, payload: dict = None):
        """
        Save result data to S3.

        The result dict is saved as-is with metadata added for correlation and debugging.
        Any JSON-serializable dict is accepted — there are no required keys.

        Reserved keys — the SDK injects the following keys into the saved JSON.
        Avoid using these in your return dict to prevent unexpected overwrites:
            - ``status_code``: Set to 200 if not already present in the user dict.
            - ``stop_reason``: Set to ``"end_turn"`` if not already present.
            - ``input_id``: Always overwritten with the value from rollout config.
            - ``s3_bucket``: Always overwritten with the value from rollout config.
            - ``result_key``: Always overwritten with the computed S3 key.
            - ``payload``: Always overwritten with the original request payload.

        Args:
            result: The result data to save (any JSON-serializable dict)
            rollout_config: Rollout configuration dict containing:
                - s3_bucket: S3 bucket name
                - exp_id: Experiment ID for organizing data
                - input_id: id for discriminating different input data examples
            payload: Original request payload (included in saved result for debugging)
            result_key: S3 key for the result (computed externally for consistency)
        """
        # Validate and extract rollout configuration
        try:
            config = RolloutConfig.from_dict(rollout_config)
        except ValueError as e:
            logging.error(f"Invalid rollout configuration: {e}")
            raise

        if "status_code" not in result:
            result["status_code"] = 200

        if "stop_reason" not in result:
            result["stop_reason"] = "end_turn"

        # Include metadata for correlation and debugging
        result["input_id"] = config.input_id
        result["s3_bucket"] = config.s3_bucket
        result["result_key"] = result_key

        # Include full payload for debugging (with _rollout config for reproducibility)
        if payload is not None:
            result["payload"] = payload

        # Save to S3
        try:
            self.s3_client.put_object(
                Bucket=config.s3_bucket,
                Key=result_key,
                Body=json.dumps(result, indent=2),
                ContentType="application/json",
            )
            logging.info(f"Stored complete results at {result_key}")
        except Exception as e:
            logging.error(f"Failed to store results in S3: {e}")
            raise

    def rollout_entrypoint(self, func):
        """
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

        Reserved keys: ``save_result`` injects SDK metadata into the saved JSON.
        See ``save_result`` docstring for the full list of reserved keys.

        Usage:
            @app.rollout_entrypoint
            def invoke_agent(payload, context):  # Can be sync or async
                # Framework-specific rollout collection
                result = collect_result(...)
                return result  # Automatically saved!

        Args:
            func: The user function that handles agent logic and result collection

        Returns:
            Decorated function registered as entrypoint
        """

        async def rollout_background_task(payload, context, result_key):
            """Background task that does the actual agent work and result saving."""
            rollout_dict = payload.get("_rollout")

            # Register with async task tracking system for logging and ping status
            task_id = self.add_async_task(f"{func.__name__}")

            try:
                # Use BedrockAgentCoreApp's _invoke_handler for sync/async compatibility
                # This automatically runs sync functions in thread pool to avoid blocking
                result = await self._invoke_handler(func, context, self._takes_context(func), payload)

                # Save result to S3 if S3 config is present
                if result_key:
                    if not isinstance(result, dict):
                        raise ValueError(
                            f"Return value must be a dict when S3 save is configured, got {type(result).__name__}"
                        )
                    self.save_result(
                        result=result,
                        rollout_config=rollout_dict,
                        payload=payload,
                        result_key=result_key,
                    )
                    logging.info(f"Result saved for function: {func.__name__}")

                return result

            except BaseException as e:
                # Save error result for client notification when S3 is configured.
                # Uses BaseException to also catch CancelledError, GeneratorExit, etc.
                # that can arise from task cancellation or deep async generator unwinding.
                if result_key:
                    try:
                        error_result = {
                            "status_code": 500,
                            "stop_reason": str(e),
                            "traceback": traceback.format_exc(),
                        }
                        self.save_result(
                            result=error_result,
                            rollout_config=rollout_dict,
                            payload=payload,
                            result_key=result_key,
                        )
                        logging.error(f"Error result saved for function: {func.__name__}: {e}")
                    except Exception:
                        logging.error(f"Failed to save error result for function: {func.__name__}", exc_info=True)
                raise
            finally:
                # Complete the async task for logging and ping status
                self.complete_async_task(task_id)

        @wraps(func)
        async def rollout_entrypoint_wrapper(payload, context):
            """Entrypoint that starts background task and returns immediately."""
            rollout_dict = payload.get("_rollout")

            # Validate required fields before launching background task.
            # ValueError propagates to base class, which returns HTTP 500.
            result_key = None
            rollout_config = None
            if rollout_dict is not None and any(f in rollout_dict for f in _S3_CONFIG_FIELDS):
                rollout_config = RolloutConfig.from_dict(rollout_dict)
                # session_id comes from ACR's HTTP header (set via runtimeSessionId),
                # fall back to UUID for local testing.
                session_id = context.session_id if context.session_id else str(uuid.uuid4())
                result_key = f"{rollout_config.exp_id}/{rollout_config.input_id}/{session_id}.json"

            # Start background task without waiting
            asyncio.create_task(rollout_background_task(payload, context, result_key))

            # Return result location so client can poll S3 for completion
            if rollout_config:
                return {
                    "status": "processing",
                    "s3_bucket": rollout_config.s3_bucket,
                    "result_key": result_key,
                }
            return {"status": "processing"}

        # Remove __wrapped__ so inspect.signature() sees the wrapper's actual signature
        # (payload, context) instead of the user function's signature. This ensures
        # BedrockAgentCoreApp._takes_context() correctly passes context to this wrapper.
        del rollout_entrypoint_wrapper.__wrapped__

        # Register using existing BedrockAgentCoreApp entrypoint infrastructure
        return self.entrypoint(rollout_entrypoint_wrapper)
