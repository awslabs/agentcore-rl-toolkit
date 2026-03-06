"""Tests for the @rollout_entrypoint decorator."""

import inspect
import json
import time
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from agentcore_rl_toolkit import AgentCoreRLApp


def test_wrapper_signature_has_context():
    """Test that the wrapper's signature includes (payload, context) for BedrockAgentCoreApp."""
    app = AgentCoreRLApp()

    @app.rollout_entrypoint
    async def my_handler(payload: dict):
        return {"rollout_data": [], "rewards": [0]}

    wrapper = app.handlers["main"]
    params = list(inspect.signature(wrapper).parameters.keys())

    assert len(params) == 2
    assert params[0] == "payload"
    assert params[1] == "context"


def test_wrapper_preserves_function_name():
    """Test that @wraps preserves the original function name."""
    app = AgentCoreRLApp()

    @app.rollout_entrypoint
    async def my_custom_handler(payload: dict):
        return {"rollout_data": [], "rewards": [0]}

    wrapper = app.handlers["main"]
    assert wrapper.__name__ == "my_custom_handler"


def test_entrypoint_with_payload_only():
    """Test that user function with signature (payload) works."""
    app = AgentCoreRLApp()

    @app.rollout_entrypoint
    async def handler(payload: dict):
        return {"rollout_data": [{"test": True}], "rewards": [1.0]}

    client = TestClient(app)
    response = client.post("/invocations", json={"prompt": "test"})

    assert response.status_code == 200
    assert response.json() == {"status": "processing"}


def test_entrypoint_with_payload_and_context():
    """Test that user function with signature (payload, context) works."""
    app = AgentCoreRLApp()

    @app.rollout_entrypoint
    async def handler(payload: dict, context):
        return {"rollout_data": [{"session": context.session_id}], "rewards": [1.0]}

    client = TestClient(app)
    response = client.post(
        "/invocations",
        json={"prompt": "test"},
        headers={"X-Amz-Bedrock-AgentCore-Session-Id": "session-123"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "processing"}


def test_entrypoint_with_sync_handler():
    """Test that sync user function works."""
    app = AgentCoreRLApp()

    @app.rollout_entrypoint
    def handler(payload: dict):
        return {"rollout_data": [{"sync": True}], "rewards": [1.0]}

    client = TestClient(app)
    response = client.post("/invocations", json={"prompt": "test"})

    assert response.status_code == 200
    assert response.json() == {"status": "processing"}


def test_response_includes_result_location_with_rollout_config():
    """Test that response includes s3_bucket and result_key when _rollout config is provided."""
    app = AgentCoreRLApp()

    @app.rollout_entrypoint
    async def handler(payload: dict):
        return {"rollout_data": [{"test": True}], "rewards": [1.0]}

    client = TestClient(app)
    response = client.post(
        "/invocations",
        json={
            "prompt": "test",
            "_rollout": {
                "exp_id": "exp-123",
                "input_id": "input-789",
                "s3_bucket": "my-bucket",
            },
        },
        headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "sess-456"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "processing"
    assert result["s3_bucket"] == "my-bucket"
    assert result["result_key"] == "exp-123/input-789/sess-456.json"


def test_response_without_rollout_config():
    """Test that response is minimal when no _rollout config is provided."""
    app = AgentCoreRLApp()

    @app.rollout_entrypoint
    async def handler(payload: dict):
        return {"rollout_data": [{"test": True}], "rewards": [1.0]}

    client = TestClient(app)
    response = client.post("/invocations", json={"prompt": "test"})

    assert response.status_code == 200
    result = response.json()
    assert result == {"status": "processing"}
    assert "s3_bucket" not in result
    assert "result_key" not in result


def test_entrypoint_accepts_empty_rollout_config():
    """Test that _rollout: {} (no S3 fields) returns HTTP 200 with minimal response."""
    app = AgentCoreRLApp()

    @app.rollout_entrypoint
    async def handler(payload: dict):
        return {"rollout_data": [{"test": True}], "rewards": [1.0]}

    client = TestClient(app)
    response = client.post(
        "/invocations",
        json={"prompt": "test", "_rollout": {}},
    )

    assert response.status_code == 200
    result = response.json()
    assert result == {"status": "processing"}
    assert "s3_bucket" not in result
    assert "result_key" not in result


def test_entrypoint_accepts_model_only_rollout_config():
    """Test that _rollout with only base_url/model_id (no S3 fields) returns HTTP 200."""
    app = AgentCoreRLApp()

    @app.rollout_entrypoint
    async def handler(payload: dict):
        return {"rollout_data": [{"test": True}], "rewards": [1.0]}

    client = TestClient(app)
    response = client.post(
        "/invocations",
        json={
            "prompt": "test",
            "_rollout": {
                "base_url": "http://localhost:8000/v1",
                "model_id": "my-model",
            },
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result == {"status": "processing"}
    assert "s3_bucket" not in result
    assert "result_key" not in result


def test_entrypoint_with_rewards_only():
    """Test that returning only rewards (no rollout_data) works."""
    app = AgentCoreRLApp()

    @app.rollout_entrypoint
    async def handler(payload: dict):
        return {"rewards": [1.0, 0.5]}

    client = TestClient(app)
    response = client.post("/invocations", json={"prompt": "test"})

    assert response.status_code == 200
    assert response.json() == {"status": "processing"}


def test_entrypoint_with_custom_fields_only():
    """Test that returning arbitrary custom fields (no rollout_data or rewards) works."""
    app = AgentCoreRLApp()

    @app.rollout_entrypoint
    async def handler(payload: dict):
        return {"metrics": {"accuracy": 0.95}, "summary": "All tests passed"}

    client = TestClient(app)
    response = client.post("/invocations", json={"prompt": "test"})

    assert response.status_code == 200
    assert response.json() == {"status": "processing"}


def test_entrypoint_with_rollout_data_rewards_and_extra_fields():
    """Test that returning rollout_data + rewards + extra fields works."""
    app = AgentCoreRLApp()

    @app.rollout_entrypoint
    async def handler(payload: dict):
        return {
            "rollout_data": [{"tokens": [1, 2, 3]}],
            "rewards": [1.0],
            "metrics": {"latency_ms": 150},
            "agent_version": "v2",
        }

    client = TestClient(app)
    response = client.post("/invocations", json={"prompt": "test"})

    assert response.status_code == 200
    assert response.json() == {"status": "processing"}


@pytest.mark.parametrize("missing_field", ["exp_id", "input_id", "s3_bucket"])
def test_entrypoint_rejects_partial_s3_config(missing_field):
    """Test that providing some but not all S3 fields returns HTTP 500."""
    app = AgentCoreRLApp()

    @app.rollout_entrypoint
    async def handler(payload: dict):
        return {"rollout_data": [{"test": True}], "rewards": [1.0]}

    complete_config = {
        "exp_id": "exp-123",
        "input_id": "input-789",
        "s3_bucket": "my-bucket",
    }
    incomplete_config = {k: v for k, v in complete_config.items() if k != missing_field}

    client = TestClient(app)
    response = client.post(
        "/invocations",
        json={"prompt": "test", "_rollout": incomplete_config},
        headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "sess-456"},
    )

    assert response.status_code == 500
    error_msg = response.json()["error"]
    assert "Missing required rollout config field" in error_msg
    assert missing_field in error_msg


# --- S3 save path tests (mocked S3) ---

_ROLLOUT_CONFIG = {
    "exp_id": "exp-001",
    "input_id": "input-001",
    "s3_bucket": "test-bucket",
}


def _make_app_with_mock_s3(handler_fn):
    """Create an AgentCoreRLApp with mocked S3 client and register the handler."""
    app = AgentCoreRLApp()
    app.s3_client = MagicMock()

    app.rollout_entrypoint(handler_fn)

    return app


def _invoke_and_wait_for_s3(app, payload, *, timeout=5.0):
    """POST to /invocations and wait for the background task to call put_object."""
    client = TestClient(app)
    response = client.post(
        "/invocations",
        json=payload,
        headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "sess-test"},
    )

    # Poll until the background task calls put_object
    deadline = time.monotonic() + timeout
    while app.s3_client.put_object.call_count == 0 and time.monotonic() < deadline:
        time.sleep(0.05)

    return response


def test_s3_save_rewards_only():
    """Test that a rewards-only dict is saved to S3 with SDK metadata."""

    async def handler(payload: dict):
        return {"rewards": [1.0]}

    app = _make_app_with_mock_s3(handler)
    response = _invoke_and_wait_for_s3(app, {"prompt": "test", "_rollout": _ROLLOUT_CONFIG})

    assert response.status_code == 200
    assert app.s3_client.put_object.call_count == 1

    call_kwargs = app.s3_client.put_object.call_args[1]
    assert call_kwargs["Bucket"] == "test-bucket"
    assert call_kwargs["Key"] == "exp-001/input-001/sess-test.json"

    body = json.loads(call_kwargs["Body"])
    # User field preserved
    assert body["rewards"] == [1.0]
    # SDK metadata injected
    assert body["status_code"] == 200
    assert body["stop_reason"] == "end_turn"
    assert body["input_id"] == "input-001"
    assert body["s3_bucket"] == "test-bucket"
    assert body["result_key"] == "exp-001/input-001/sess-test.json"
    assert body["payload"]["prompt"] == "test"


def test_s3_save_custom_fields():
    """Test that arbitrary custom fields are saved to S3 with SDK metadata."""

    async def handler(payload: dict):
        return {"metrics": {"accuracy": 0.95}, "summary": "All passed"}

    app = _make_app_with_mock_s3(handler)
    response = _invoke_and_wait_for_s3(app, {"prompt": "eval", "_rollout": _ROLLOUT_CONFIG})

    assert response.status_code == 200
    assert app.s3_client.put_object.call_count == 1

    body = json.loads(app.s3_client.put_object.call_args[1]["Body"])
    # User fields preserved
    assert body["metrics"] == {"accuracy": 0.95}
    assert body["summary"] == "All passed"
    # SDK metadata present
    assert body["status_code"] == 200
    assert body["input_id"] == "input-001"
    assert "result_key" in body


def test_s3_save_backward_compat_rollout_and_rewards():
    """Test that the classic rollout_data + rewards structure is saved correctly."""

    async def handler(payload: dict):
        return {"rollout_data": [{"tokens": [10, 20, 30]}], "rewards": [1.0, 0.5]}

    app = _make_app_with_mock_s3(handler)
    response = _invoke_and_wait_for_s3(app, {"prompt": "math", "_rollout": _ROLLOUT_CONFIG})

    assert response.status_code == 200
    assert app.s3_client.put_object.call_count == 1

    body = json.loads(app.s3_client.put_object.call_args[1]["Body"])
    # User fields preserved
    assert body["rollout_data"] == [{"tokens": [10, 20, 30]}]
    assert body["rewards"] == [1.0, 0.5]
    # SDK metadata present
    assert body["status_code"] == 200
    assert body["stop_reason"] == "end_turn"
    assert body["input_id"] == "input-001"
    assert body["s3_bucket"] == "test-bucket"
