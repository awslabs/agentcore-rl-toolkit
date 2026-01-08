import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp


@dataclass
class TrainingConfig:
    """Training configuration for rollout collection and storage."""

    exp_id: str
    session_id: str
    input_id: str
    sqs_url: str
    s3_bucket: str

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        """Create TrainingConfig from dictionary with validation."""
        try:
            return cls(
                exp_id=data["exp_id"],
                session_id=data["session_id"],
                input_id=data["input_id"],
                sqs_url=data["sqs_url"],
                s3_bucket=data["s3_bucket"],
            )
        except KeyError as e:
            raise ValueError(f"Missing required training config field: {e}") from e


class AgentCoreRLApp(BedrockAgentCoreApp, ABC):
    def __init__(self):
        super().__init__()
        self.s3_client = boto3.client("s3")
        self.sqs_client = boto3.client("sqs")

    @abstractmethod
    def create_openai_compatible_model(self, **kwargs):
        """Create an OpenAI-compatible model for this framework.

        Must be implemented by framework-specific subclasses.

        Args:
            **kwargs: Framework-specific model parameters

        Returns:
            Framework-specific model instance configured for vLLM server
        """
        pass

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

    def save_rollout_and_notify(self, rollout_data: dict, training_config: dict):
        """
        Save rollout data to S3 and notify SQS queue.

        Args:
            rollout_data: The prepared rollout data
            training_config: Training configuration dict containing:
                - s3_bucket: S3 bucket name
                - sqs_url: SQS queue URL for notifications
                - exp_id: Experiment ID for organizing data
                - session_id: Session id for the current task
                - input_id: id for discriminating different input data examples
        """
        # Validate and extract training configuration
        try:
            config = TrainingConfig.from_dict(training_config)
        except ValueError as e:
            logging.error(f"Invalid training configuration: {e}")
            raise

        result_key = f"{config.exp_id}/{config.input_id}_{config.session_id}.json"

        if "status_code" not in rollout_data:
            rollout_data["status_code"] = 200

        if "stop_reason" not in rollout_data:
            rollout_data["stop_reason"] = "end_turn"

        # Return the input id identifying rollouts of the same input data (prompt) example
        # for advantage computation.
        rollout_data["input_id"] = config.input_id

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

        # Send SQS notification (mimic S3 notification format)
        try:
            sqs_message = {
                "Records": [
                    {
                        "eventSource": "rollout:collector",
                        "eventName": "ObjectCreated:Put",
                        "eventTime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "s3": {"bucket": {"name": config.s3_bucket}, "object": {"key": result_key}},
                    }
                ]
            }

            self.sqs_client.send_message(QueueUrl=config.sqs_url, MessageBody=json.dumps(sqs_message))
            logging.info(f"Sent SQS notification for {result_key}")
        except Exception as e:
            logging.error(f"Failed to send SQS notification for {result_key}: {e}")
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

        async def rollout_background_task(payload, context):
            """Background task that does the actual agent work and rollout saving."""
            training_config = payload.get("_training")

            # Register with async task tracking system for logging and ping status
            task_id = self.add_async_task(f"{func.__name__}")

            try:
                # Use BedrockAgentCoreApp's _invoke_handler for sync/async compatibility
                # This automatically runs sync functions in thread pool to avoid blocking
                result = await self._invoke_handler(func, context, self._takes_context(func), payload)

                # If this is an RL training run, validate and normalize the rollout structure
                if training_config:
                    if not isinstance(result, dict):
                        raise ValueError("RL training runs must return a dictionary")
                    result = self._validate_and_normalize_rollout(result)

                # Save rollout data if we have training config
                if isinstance(result, dict) and training_config:
                    self.save_rollout_and_notify(rollout_data=result, training_config=training_config)
                    logging.info(f"Rollout data saved for function: {func.__name__}")

                return result

            except Exception as e:
                # Always save error rollout for client notification
                if training_config:
                    error_rollout = {"status_code": 500, "stop_reason": str(e)}
                    self.save_rollout_and_notify(rollout_data=error_rollout, training_config=training_config)
                    logging.error(f"Error rollout saved for function: {func.__name__}: {e}")
                raise
            finally:
                # Complete the async task for logging and ping status
                self.complete_async_task(task_id)

        @wraps(func)
        async def rollout_entrypoint_wrapper(payload, context):
            """Entrypoint that starts background task and returns immediately."""
            # Start background task without waiting
            asyncio.create_task(rollout_background_task(payload, context))
            return {"status": "processing"}

        # Remove __wrapped__ so inspect.signature() sees the wrapper's actual signature
        # (payload, context) instead of the user function's signature. This ensures
        # BedrockAgentCoreApp._takes_context() correctly passes context to this wrapper.
        del rollout_entrypoint_wrapper.__wrapped__

        # Register using existing BedrockAgentCoreApp entrypoint infrastructure
        return self.entrypoint(rollout_entrypoint_wrapper)
