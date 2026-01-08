"""Tests for the @rollout_entrypoint decorator."""

import inspect

from starlette.testclient import TestClient

from agentcore_rl_toolkit import AgentCoreRLApp


class MockAgentCoreRLApp(AgentCoreRLApp):
    """Minimal concrete implementation for testing."""

    def create_openai_compatible_model(self, **kwargs):
        return None


def test_wrapper_signature_has_context():
    """Test that the wrapper's signature includes (payload, context) for BedrockAgentCoreApp."""
    app = MockAgentCoreRLApp()

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
    app = MockAgentCoreRLApp()

    @app.rollout_entrypoint
    async def my_custom_handler(payload: dict):
        return {"rollout_data": [], "rewards": [0]}

    wrapper = app.handlers["main"]
    assert wrapper.__name__ == "my_custom_handler"


def test_entrypoint_with_payload_only():
    """Test that user function with signature (payload) works."""
    app = MockAgentCoreRLApp()

    @app.rollout_entrypoint
    async def handler(payload: dict):
        return {"rollout_data": [{"test": True}], "rewards": [1.0]}

    client = TestClient(app)
    response = client.post("/invocations", json={"prompt": "test"})

    assert response.status_code == 200
    assert response.json() == {"status": "processing"}


def test_entrypoint_with_payload_and_context():
    """Test that user function with signature (payload, context) works."""
    app = MockAgentCoreRLApp()

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
    app = MockAgentCoreRLApp()

    @app.rollout_entrypoint
    def handler(payload: dict):
        return {"rollout_data": [{"sync": True}], "rewards": [1.0]}

    client = TestClient(app)
    response = client.post("/invocations", json={"prompt": "test"})

    assert response.status_code == 200
    assert response.json() == {"status": "processing"}
