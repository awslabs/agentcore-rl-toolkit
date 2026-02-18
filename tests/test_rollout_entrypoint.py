"""Tests for the @rollout_entrypoint decorator."""

import inspect

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


def test_response_includes_result_location_with_training_config():
    """Test that response includes s3_bucket and result_key when _training config is provided."""
    app = AgentCoreRLApp()

    @app.rollout_entrypoint
    async def handler(payload: dict):
        return {"rollout_data": [{"test": True}], "rewards": [1.0]}

    client = TestClient(app)
    response = client.post(
        "/invocations",
        json={
            "prompt": "test",
            "_training": {
                "exp_id": "exp-123",
                "session_id": "sess-456",
                "input_id": "input-789",
                "s3_bucket": "my-bucket",
                "sqs_url": "https://sqs.us-east-1.amazonaws.com/123/queue",
            },
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "processing"
    assert result["s3_bucket"] == "my-bucket"
    assert result["result_key"] == "exp-123/input-789_sess-456.json"


def test_response_without_training_config():
    """Test that response is minimal when no _training config is provided."""
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
