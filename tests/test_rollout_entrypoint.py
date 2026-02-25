"""Tests for the @rollout_entrypoint decorator."""

import inspect

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
                "session_id": "sess-456",
                "input_id": "input-789",
                "s3_bucket": "my-bucket",
            },
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "processing"
    assert result["s3_bucket"] == "my-bucket"
    assert result["result_key"] == "exp-123/input-789_sess-456.json"


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


@pytest.mark.parametrize("missing_field", ["exp_id", "session_id", "input_id", "s3_bucket"])
def test_entrypoint_rejects_partial_s3_config(missing_field):
    """Test that providing some but not all S3 fields returns HTTP 500."""
    app = AgentCoreRLApp()

    @app.rollout_entrypoint
    async def handler(payload: dict):
        return {"rollout_data": [{"test": True}], "rewards": [1.0]}

    complete_config = {
        "exp_id": "exp-123",
        "session_id": "sess-456",
        "input_id": "input-789",
        "s3_bucket": "my-bucket",
    }
    incomplete_config = {k: v for k, v in complete_config.items() if k != missing_field}

    client = TestClient(app)
    response = client.post(
        "/invocations",
        json={"prompt": "test", "_rollout": incomplete_config},
    )

    assert response.status_code == 500
    error_msg = response.json()["error"]
    assert "Missing required rollout config field" in error_msg
    assert missing_field in error_msg
