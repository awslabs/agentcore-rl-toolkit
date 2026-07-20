"""Tests for SandboxClient and Sandbox with mocked boto3."""

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, EventStreamError

from agentcore_rl_toolkit.sandbox import Sandbox, SandboxClient, SandboxProtocolError

FAKE_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/sandbox-test"
FAKE_SESSION_ID = "a" * 40


def mock_streaming_body(data: dict) -> io.BytesIO:
    """Create a mock StreamingBody-like object for /invocations responses."""
    return io.BytesIO(json.dumps(data).encode())


def make_start_response(body: dict = None) -> dict:
    return {"response": mock_streaming_body(body if body is not None else {"status": "ok", "state": "busy"})}


def make_command_response(deltas: list = None, exit_code: int = 0, status: str = "COMPLETED") -> dict:
    """Build a fake invoke_agent_runtime_command response with an eventstream.

    Botocore eventstreams are iterables of event dicts; a plain list suffices.
    """
    events = [{"chunk": {"contentStart": {}}}]
    for delta in deltas or []:
        events.append({"chunk": {"contentDelta": delta}})
    events.append({"chunk": {"contentStop": {"exitCode": exit_code, "status": status}}})
    return {"stream": events}


def make_client_and_mock(start_response: dict = None):
    """Create a SandboxClient with a mocked bedrock-agentcore client."""
    with patch("agentcore_rl_toolkit.sandbox.client.boto3") as mock_boto3:
        mock_acr = MagicMock()
        mock_boto3.client.return_value = mock_acr
        client = SandboxClient(runtime_arn=FAKE_ARN)
    if start_response is not None:
        mock_acr.invoke_agent_runtime.return_value = start_response
    return client, mock_acr


class TestSandboxClientInit:
    def test_region_parsed_from_arn(self):
        client, _ = make_client_and_mock()
        assert client.region == "us-west-2"

    def test_explicit_region_wins(self):
        with patch("agentcore_rl_toolkit.sandbox.client.boto3"):
            client = SandboxClient(runtime_arn=FAKE_ARN, region="eu-west-1")
        assert client.region == "eu-west-1"

    def test_invalid_arn_raises(self):
        with pytest.raises(ValueError, match="Invalid ARN format"):
            SandboxClient(runtime_arn="not-an-arn")

    def test_default_qualifier(self):
        client, _ = make_client_and_mock()
        assert client.qualifier == "DEFAULT"

    def test_boto3_client_config(self):
        with patch("agentcore_rl_toolkit.sandbox.client.boto3") as mock_boto3:
            SandboxClient(runtime_arn=FAKE_ARN, max_retry_attempts=7)
        args, kwargs = mock_boto3.client.call_args
        assert args == ("bedrock-agentcore",)
        assert kwargs["region_name"] == "us-west-2"
        assert kwargs["config"].retries == {"max_attempts": 7, "mode": "adaptive"}


class TestStart:
    def test_start_invokes_runtime_with_start_action(self):
        client, mock_acr = make_client_and_mock(make_start_response())
        sandbox = client.start()
        kwargs = mock_acr.invoke_agent_runtime.call_args.kwargs
        assert kwargs["agentRuntimeArn"] == FAKE_ARN
        assert kwargs["qualifier"] == "DEFAULT"
        assert json.loads(kwargs["payload"]) == {"action": "start"}
        assert isinstance(sandbox, Sandbox)

    def test_generated_session_id_long_enough(self):
        client, mock_acr = make_client_and_mock(make_start_response())
        sandbox = client.start()
        assert len(sandbox.session_id) >= 33
        assert mock_acr.invoke_agent_runtime.call_args.kwargs["runtimeSessionId"] == sandbox.session_id

    def test_explicit_session_id_honored(self):
        client, _ = make_client_and_mock(make_start_response())
        sandbox = client.start(session_id=FAKE_SESSION_ID)
        assert sandbox.session_id == FAKE_SESSION_ID

    def test_extra_response_fields_ignored(self):
        client, _ = make_client_and_mock(make_start_response({"status": "ok", "state": "busy", "future": "field"}))
        client.start()  # no raise

    def test_non_json_response_raises(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime.return_value = {"response": io.BytesIO(b"<html>not json</html>")}
        with pytest.raises(SandboxProtocolError, match="non-JSON"):
            client.start()

    def test_error_status_response_raises(self):
        client, _ = make_client_and_mock(make_start_response({"status": "error", "error": "boom"}))
        with pytest.raises(SandboxProtocolError, match="Unexpected sandbox start response"):
            client.start()

    def test_client_error_propagates(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "InvokeAgentRuntime"
        )
        with pytest.raises(ClientError):
            client.start()


class TestAttach:
    def test_attach_makes_no_api_call(self):
        client, mock_acr = make_client_and_mock()
        sandbox = client.attach(FAKE_SESSION_ID)
        assert sandbox.session_id == FAKE_SESSION_ID
        mock_acr.invoke_agent_runtime.assert_not_called()

    def test_short_session_id_raises(self):
        client, _ = make_client_and_mock()
        with pytest.raises(ValueError, match="33-256 characters"):
            client.attach("too-short")


class TestExec:
    def test_happy_path_accumulates_output(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime_command.return_value = make_command_response(
            deltas=[{"stdout": "hello "}, {"stdout": "world"}, {"stderr": "warn"}],
        )
        result = client.attach(FAKE_SESSION_ID).exec("echo hello world")
        assert result.exit_code == 0
        assert result.stdout == "hello world"
        assert result.stderr == "warn"
        assert result.timed_out is False

    def test_nonzero_exit_is_data(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime_command.return_value = make_command_response(
            deltas=[{"stderr": "tests failed"}], exit_code=1
        )
        result = client.attach(FAKE_SESSION_ID).exec("pytest -q")
        assert result.exit_code == 1
        assert result.timed_out is False

    def test_timed_out_is_data_with_partial_output(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime_command.return_value = make_command_response(
            deltas=[{"stdout": "partial"}], exit_code=-1, status="TIMED_OUT"
        )
        result = client.attach(FAKE_SESSION_ID).exec("sleep 999", timeout=1)
        assert result.timed_out is True
        assert result.exit_code == -1
        assert result.stdout == "partial"

    def test_mixed_delta_in_one_event(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime_command.return_value = make_command_response(
            deltas=[{"stdout": "out", "stderr": "err"}],
        )
        result = client.attach(FAKE_SESSION_ID).exec("cmd")
        assert result.stdout == "out"
        assert result.stderr == "err"

    def test_timeout_forwarded_in_body(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime_command.return_value = make_command_response()
        client.attach(FAKE_SESSION_ID).exec("cmd", timeout=600)
        body = mock_acr.invoke_agent_runtime_command.call_args.kwargs["body"]
        assert body == {"command": "/bin/sh -c 'cmd'", "timeout": 600}

    def test_timeout_omitted_when_none(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime_command.return_value = make_command_response()
        client.attach(FAKE_SESSION_ID).exec("cmd")
        body = mock_acr.invoke_agent_runtime_command.call_args.kwargs["body"]
        assert body == {"command": "/bin/sh -c 'cmd'"}

    def test_cwd_and_env_composed(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime_command.return_value = make_command_response()
        client.attach(FAKE_SESSION_ID).exec("pytest -q", cwd="/app", env={"FOO": "bar"})
        body = mock_acr.invoke_agent_runtime_command.call_args.kwargs["body"]
        assert body["command"] == "/bin/sh -c 'cd /app && export FOO=bar; pytest -q'"

    def test_client_level_shell(self):
        with patch("agentcore_rl_toolkit.sandbox.client.boto3") as mock_boto3:
            mock_acr = MagicMock()
            mock_boto3.client.return_value = mock_acr
            client = SandboxClient(runtime_arn=FAKE_ARN, shell="/bin/bash")
        mock_acr.invoke_agent_runtime_command.return_value = make_command_response()
        client.attach(FAKE_SESSION_ID).exec("cmd")
        body = mock_acr.invoke_agent_runtime_command.call_args.kwargs["body"]
        assert body["command"] == "/bin/bash -c 'cmd'"

    def test_per_exec_shell_overrides_client(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime_command.return_value = make_command_response()
        client.attach(FAKE_SESSION_ID).exec("cmd", shell="/bin/bash")
        body = mock_acr.invoke_agent_runtime_command.call_args.kwargs["body"]
        assert body["command"] == "/bin/bash -c 'cmd'"

    def test_session_and_arn_forwarded(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime_command.return_value = make_command_response()
        client.attach(FAKE_SESSION_ID).exec("cmd")
        kwargs = mock_acr.invoke_agent_runtime_command.call_args.kwargs
        assert kwargs["agentRuntimeArn"] == FAKE_ARN
        assert kwargs["runtimeSessionId"] == FAKE_SESSION_ID
        assert kwargs["qualifier"] == "DEFAULT"

    def test_error_event_in_stream_raises(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime_command.return_value = {
            "stream": [{"throttlingException": {"message": "too fast"}}]
        }
        with pytest.raises(SandboxProtocolError, match="Error event in command stream"):
            client.attach(FAKE_SESSION_ID).exec("cmd")

    def test_eventstream_error_propagates(self):
        client, mock_acr = make_client_and_mock()

        def raising_stream():
            yield {"chunk": {"contentStart": {}}}
            raise EventStreamError({"Error": {"Code": "InternalServerException", "Message": "boom"}}, "ResponseStream")

        mock_acr.invoke_agent_runtime_command.return_value = {"stream": raising_stream()}
        with pytest.raises(EventStreamError):
            client.attach(FAKE_SESSION_ID).exec("cmd")

    def test_stream_without_content_stop_raises(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime_command.return_value = {
            "stream": [{"chunk": {"contentStart": {}}}, {"chunk": {"contentDelta": {"stdout": "x"}}}]
        }
        with pytest.raises(SandboxProtocolError, match="without a result"):
            client.attach(FAKE_SESSION_ID).exec("cmd")

    def test_client_error_propagates(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime_command.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "no session"}}, "InvokeAgentRuntimeCommand"
        )
        with pytest.raises(ClientError):
            client.attach(FAKE_SESSION_ID).exec("cmd")


class TestTerminate:
    def test_terminate_sends_stop_then_stops_session(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime.return_value = make_start_response({"status": "ok", "state": "healthy"})
        client.attach(FAKE_SESSION_ID).terminate()
        payload = json.loads(mock_acr.invoke_agent_runtime.call_args.kwargs["payload"])
        assert payload == {"action": "stop"}
        kwargs = mock_acr.stop_runtime_session.call_args.kwargs
        assert kwargs["agentRuntimeArn"] == FAKE_ARN
        assert kwargs["runtimeSessionId"] == FAKE_SESSION_ID
        assert kwargs["clientToken"]

    def test_stop_action_failure_still_stops_session(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime.side_effect = ClientError(
            {"Error": {"Code": "RuntimeClientError", "Message": "gone"}}, "InvokeAgentRuntime"
        )
        client.attach(FAKE_SESSION_ID).terminate()  # no raise
        mock_acr.stop_runtime_session.assert_called_once()

    def test_stop_session_failure_swallowed(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime.return_value = make_start_response({"status": "ok", "state": "healthy"})
        mock_acr.stop_runtime_session.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "already gone"}}, "StopRuntimeSession"
        )
        client.attach(FAKE_SESSION_ID).terminate()  # no raise

    def test_terminate_idempotent(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime.return_value = make_start_response({"status": "ok", "state": "healthy"})
        sandbox = client.attach(FAKE_SESSION_ID)
        sandbox.terminate()
        sandbox.terminate()
        assert mock_acr.stop_runtime_session.call_count == 1
        assert mock_acr.invoke_agent_runtime.call_count == 1


class TestContextManager:
    def test_exit_terminates(self):
        client, mock_acr = make_client_and_mock(make_start_response())
        with client.start() as sandbox:
            pass
        assert sandbox._terminated
        mock_acr.stop_runtime_session.assert_called_once()

    def test_exception_propagates_and_terminates(self):
        client, mock_acr = make_client_and_mock(make_start_response())
        with pytest.raises(RuntimeError, match="boom"):
            with client.start():
                raise RuntimeError("boom")
        mock_acr.stop_runtime_session.assert_called_once()
