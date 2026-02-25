import asyncio
import json
import logging
import os
from dataclasses import dataclass
from functools import wraps

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp


@dataclass
class RolloutConfig:
    """Rollout configuration for rollout collection and storage."""

    exp_id: str
    session_id: str
    input_id: str
    s3_bucket: str

    @classmethod
    def from_dict(cls, data: dict) -> "RolloutConfig":
        """Create RolloutConfig from dictionary with validation."""
        try:
            return cls(
                exp_id=data["exp_id"],
                session_id=data["session_id"],
                input_id=data["input_id"],
                s3_bucket=data["s3_bucket"],
            )
        except KeyError as e:
            raise ValueError(f"Missing required rollout config field: {e}") from e


class AgentCoreRLApp(BedrockAgentCoreApp):
    def __init__(self):
        super().__init__()
        self.s3_client = boto3.client("s3")

    def create_openai_compatible_model(self, **kwargs):
        """Create an OpenAI-compatible model for this framework.

        Optional: Override in framework-specific subclasses, or create model directly
        in your entrypoint (see examples/strands_migration_agent/dev_app.py).

        Args:
            **kwargs: Framework-specific model parameters

        Returns:
            Framework-specific model instance configured for vLLM server

        Raises:
            NotImplementedError: If called without override. Create model directly instead.
        """
        raise NotImplementedError(
            "create_openai_compatible_model() is optional. "
            "Either override in a subclass or create your model directly in the entrypoint."
        )

    def _get_model_config(self):
        """Get and validate model configuration from environment."""
        base_url = os.getenv("BASE_URL")
        model_id = os.getenv("MODEL_ID")

        if not base_url or not model_id:
            raise ValueError(
                "Missing required environment variables: BASE_URL, MODEL_ID. " "Make sure to call load_dotenv()."
            )

        return base_url, model_id

    def _validate_and_normalize_rollout(self, rollout_dict: dict) -> dict:
        """
        Validate and normalize rollout data structure.

        Ensures the return value from user functions has the expected format:
        {"rollout_data": [...], "rewards": [...]}

        Args:
            rollout_dict: Dictionary returned from user function

        Returns:
            Normalized rollout dictionary with validated structure

        Raises:
            ValueError: If structure is invalid or rewards don't match rollout length
        """
        # Require both fields to exist
        if "rollout_data" not in rollout_dict:
            raise ValueError("Return value must include 'rollout_data' field")
        if "rewards" not in rollout_dict:
            raise ValueError("Return value must include 'rewards' field")

        rollout_data = rollout_dict["rollout_data"]
        rewards = rollout_dict["rewards"]

        # Validate rollout_data
        if not isinstance(rollout_data, list) or len(rollout_data) == 0:
            raise ValueError("rollout_data must be a list with length >= 1")

        # Normalize rewards to list if not already
        if not isinstance(rewards, list):
            rewards = [rewards]

        # Validate rewards length
        if len(rewards) != 1 and len(rewards) != len(rollout_data):
            raise ValueError(
                f"rewards must be length 1 (outcome reward) or "
                f"match rollout_data length {len(rollout_data)} (per-step reward)"
            )

        # Update with normalized rewards
        rollout_dict["rewards"] = rewards
        return rollout_dict

    def save_rollout(self, rollout_data: dict, rollout_config: dict, payload: dict = None, result_key: str = None):
        """
        Save rollout data to S3.

        Args:
            rollout_data: The prepared rollout data
            rollout_config: Rollout configuration dict containing:
                - s3_bucket: S3 bucket name
                - exp_id: Experiment ID for organizing data
                - session_id: Session id for the current task
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

        # Use provided result_key or compute it
        if result_key is None:
            result_key = f"{config.exp_id}/{config.input_id}_{config.session_id}.json"

        if "status_code" not in rollout_data:
            rollout_data["status_code"] = 200

        if "stop_reason" not in rollout_data:
            rollout_data["stop_reason"] = "end_turn"

        # Include metadata for correlation and debugging
        rollout_data["input_id"] = config.input_id
        rollout_data["s3_bucket"] = config.s3_bucket
        rollout_data["result_key"] = result_key

        # Include full payload for debugging (with _rollout config for reproducibility)
        if payload is not None:
            rollout_data["payload"] = payload

        # Save to S3
        try:
            self.s3_client.put_object(
                Bucket=config.s3_bucket,
                Key=result_key,
                Body=json.dumps(rollout_data, indent=2),
                ContentType="application/json",
            )
            logging.info(f"Stored complete results at {result_key}")
        except Exception as e:
            logging.error(f"Failed to store results in S3: {e}")
            raise

    def rollout_entrypoint(self, func):
        """
        Decorator for RL training that handles asyncio.create_task and rollout saving automatically.

        This decorator:
        1. Handles both sync and async user functions using BedrockAgentCoreApp's infrastructure
        2. Automatically saves rollout data when user returns it
        3. Handles errors and saves error rollouts for client notification
        4. Returns immediately with {"status": "processing"} for non-blocking behavior

        Usage:
            @app.rollout_entrypoint
            def invoke_agent(payload, context):  # Can be sync or async
                # Framework-specific rollout collection
                rollout_data = collect_rollout(...)
                return rollout_data  # Automatically saved!

        Args:
            func: The user function that handles agent logic and rollout collection

        Returns:
            Decorated function registered as entrypoint
        """

        async def rollout_background_task(payload, context, result_key):
            """Background task that does the actual agent work and rollout saving."""
            rollout_dict = payload.get("_rollout")

            # Register with async task tracking system for logging and ping status
            task_id = self.add_async_task(f"{func.__name__}")

            try:
                # Use BedrockAgentCoreApp's _invoke_handler for sync/async compatibility
                # This automatically runs sync functions in thread pool to avoid blocking
                result = await self._invoke_handler(func, context, self._takes_context(func), payload)

                # If this is an RL training run, validate and normalize the rollout structure
                if rollout_dict:
                    if not isinstance(result, dict):
                        raise ValueError("RL training runs must return a dictionary")
                    result = self._validate_and_normalize_rollout(result)

                # Save rollout data if we have training config
                if isinstance(result, dict) and rollout_dict:
                    self.save_rollout(
                        rollout_data=result,
                        rollout_config=rollout_dict,
                        payload=payload,
                        result_key=result_key,
                    )
                    logging.info(f"Rollout data saved for function: {func.__name__}")

                return result

            except Exception as e:
                # Always save error rollout for client notification
                if rollout_dict:
                    error_rollout = {"status_code": 500, "stop_reason": str(e)}
                    self.save_rollout(
                        rollout_data=error_rollout,
                        rollout_config=rollout_dict,
                        payload=payload,
                        result_key=result_key,
                    )
                    logging.error(f"Error rollout saved for function: {func.__name__}: {e}")
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
            if rollout_dict is not None:
                rollout_config = RolloutConfig.from_dict(rollout_dict)
                result_key = f"{rollout_config.exp_id}/{rollout_config.input_id}_{rollout_config.session_id}.json"

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
