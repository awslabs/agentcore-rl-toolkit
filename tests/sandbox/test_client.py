"""Tests for SandboxClient and Sandbox with mocked boto3."""

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from agentcore_rl_toolkit.sandbox import ExecError, ExecHandle, Sandbox, SandboxClient, SandboxProtocolError

FAKE_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/sandbox-test"
FAKE_SESSION_ID = "a" * 40


def mock_streaming_body(data: dict) -> io.BytesIO:
    """Create a mock StreamingBody-like object for /invocations responses."""
    return io.BytesIO(json.dumps(data).encode())


def make_start_response(body: dict = None) -> dict:
    return {"response": mock_streaming_body(body if body is not None else {"status": "ok", "state": "busy"})}


def set_execution_response(mock_acr, status="completed", result=None, error=None):
    """Return a fresh body for each request, using its actual invocation ID."""

    def respond(**kwargs):
        request = json.loads(kwargs["payload"])["_agentcore_runtime"]
        response = {"version": 1, "invocation_id": request["invocation_id"], "status": status}
        if status == "completed":
            if error is not None:
                response["error"] = error
            else:
                response["result"] = result or {
                    "exit_code": 0,
                    "stdout": "hello",
                    "stderr": "warn",
                    "timed_out": False,
                }
        return {"response": mock_streaming_body(response)}

    mock_acr.invoke_agent_runtime.side_effect = respond


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
    def test_foreground_uses_managed_invocation(self):
        client, mock_acr = make_client_and_mock()
        set_execution_response(mock_acr)
        result = client.attach(FAKE_SESSION_ID).exec("echo hello")
        assert (result.exit_code, result.stdout, result.stderr, result.timed_out) == (0, "hello", "warn", False)
        kwargs = mock_acr.invoke_agent_runtime.call_args.kwargs
        assert kwargs["agentRuntimeArn"] == FAKE_ARN
        assert kwargs["runtimeSessionId"] == FAKE_SESSION_ID
        assert kwargs["qualifier"] == "DEFAULT"
        payload = json.loads(kwargs["payload"])
        assert payload["command"] == "echo hello"
        assert payload["shell"] == "/bin/sh"
        assert payload["_agentcore_runtime"]["version"] == 1
        assert payload["_agentcore_runtime"]["operation"] == "start"
        assert payload["_agentcore_runtime"]["background"] is False
        assert payload["_agentcore_runtime"]["invocation_id"]
        mock_acr.invoke_agent_runtime_command.assert_not_called()

    @pytest.mark.parametrize("exit_code,timed_out", [(3, False), (-1, True)])
    def test_nonzero_and_timeout_are_data(self, exit_code, timed_out):
        client, mock_acr = make_client_and_mock()
        set_execution_response(
            mock_acr,
            result={
                "exit_code": exit_code,
                "stdout": "partial",
                "stderr": "",
                "timed_out": timed_out,
                "stdout_truncated": True,
                "stderr_truncated": False,
            },
        )
        result = client.attach(FAKE_SESSION_ID).exec("cmd")
        assert result.exit_code == exit_code
        assert result.timed_out is timed_out
        assert result.stdout == "partial"
        assert result.stdout_truncated is True
        assert result.stderr_truncated is False

    def test_cwd_env_shell_and_timeout(self):
        client, mock_acr = make_client_and_mock()
        set_execution_response(mock_acr)
        client.attach(FAKE_SESSION_ID).exec(
            "printf '%s' \"$FOO\"", timeout=600, cwd="/app", env={"FOO": "a b"}, shell="/bin/bash"
        )
        payload = json.loads(mock_acr.invoke_agent_runtime.call_args.kwargs["payload"])
        assert payload["command"] == "cd /app && export FOO='a b' && printf '%s' \"$FOO\""
        assert payload["shell"] == "/bin/bash"
        assert payload["timeout"] == 600

    def test_client_shell_and_default_timeout(self):
        client, mock_acr = make_client_and_mock()
        client.shell = "/bin/bash"
        set_execution_response(mock_acr)
        client.attach(FAKE_SESSION_ID).exec("cmd")
        payload = json.loads(mock_acr.invoke_agent_runtime.call_args.kwargs["payload"])
        assert payload["shell"] == "/bin/bash"
        assert "timeout" not in payload

    def test_background_returns_handle_without_waiting(self):
        client, mock_acr = make_client_and_mock()
        set_execution_response(mock_acr, status="in_progress")
        handle = client.attach(FAKE_SESSION_ID).exec("cmd", background=True, invocation_id="saved-id")
        assert isinstance(handle, ExecHandle)
        assert handle.invocation_id == "saved-id"
        assert handle.session_id == FAKE_SESSION_ID
        assert mock_acr.invoke_agent_runtime.call_count == 1
        assert json.loads(mock_acr.invoke_agent_runtime.call_args.kwargs["payload"])["_agentcore_runtime"]["background"]

    def test_network_failure_retains_handle_and_original_error(self):
        client, mock_acr = make_client_and_mock()
        error = ClientError(
            {"Error": {"Code": "InternalServerException", "Message": "lost response"}}, "InvokeAgentRuntime"
        )
        mock_acr.invoke_agent_runtime.side_effect = error
        with pytest.raises(ExecError) as caught:
            client.attach(FAKE_SESSION_ID).exec("cmd")
        assert caught.value.__cause__ is error
        handle = caught.value.handle
        payload = json.loads(mock_acr.invoke_agent_runtime.call_args.kwargs["payload"])
        assert handle.invocation_id == payload["_agentcore_runtime"]["invocation_id"]
        set_execution_response(mock_acr)
        assert handle.result().stdout == "hello"
        assert (
            json.loads(mock_acr.invoke_agent_runtime.call_args.kwargs["payload"])["_agentcore_runtime"]["operation"]
            == "get"
        )

    @pytest.mark.parametrize(
        "body", [b"not json", b'{"status":"ok"}', b'{"version":1,"invocation_id":"wrong","status":"completed"}']
    )
    def test_bad_or_old_daemon_response(self, body):
        client, mock_acr = make_client_and_mock()
        stream = io.BytesIO(body)
        mock_acr.invoke_agent_runtime.return_value = {"response": stream}
        with pytest.raises(ExecError) as caught:
            client.attach(FAKE_SESSION_ID).exec("cmd")
        assert isinstance(caught.value.__cause__, SandboxProtocolError)
        assert stream.closed


class TestExecHandle:
    def test_reattach_is_local_and_get_contains_no_command(self):
        client, mock_acr = make_client_and_mock()
        handle = client.attach(FAKE_SESSION_ID).get_exec("saved-id")
        mock_acr.invoke_agent_runtime.assert_not_called()
        set_execution_response(mock_acr, status="in_progress")
        assert handle.status() == "in_progress"
        assert json.loads(mock_acr.invoke_agent_runtime.call_args.kwargs["payload"]) == {
            "_agentcore_runtime": {"version": 1, "operation": "get", "invocation_id": "saved-id"}
        }

    def test_wait_timeout_keeps_execution_and_session(self):
        client, mock_acr = make_client_and_mock()
        set_execution_response(mock_acr, status="in_progress")
        handle = client.attach(FAKE_SESSION_ID).exec("cmd", background=True)
        with pytest.raises(TimeoutError):
            handle.result(timeout=0)
        mock_acr.stop_runtime_session.assert_not_called()
        set_execution_response(mock_acr)
        assert handle.result(timeout=0).stdout == "hello"
        operations = [
            json.loads(c.kwargs["payload"])["_agentcore_runtime"]["operation"]
            for c in mock_acr.invoke_agent_runtime.call_args_list
        ]
        assert operations == ["start", "get", "get"]

    def test_result_polls_only_get(self):
        client, mock_acr = make_client_and_mock()
        handle = client.attach(FAKE_SESSION_ID).get_exec("poll")
        pending = {"version": 1, "invocation_id": "poll", "status": "in_progress"}
        complete = {
            **pending,
            "status": "completed",
            "result": {"exit_code": 0, "stdout": "done", "stderr": "", "timed_out": False},
        }
        mock_acr.invoke_agent_runtime.side_effect = [
            {"response": mock_streaming_body(pending)},
            {"response": mock_streaming_body(complete)},
        ]
        with patch("agentcore_rl_toolkit.sandbox.client.time.sleep") as sleep:
            assert handle.result(timeout=10).stdout == "done"
        sleep.assert_called_once_with(0.1)
        for call in mock_acr.invoke_agent_runtime.call_args_list:
            assert json.loads(call.kwargs["payload"])["_agentcore_runtime"]["operation"] == "get"

    def test_polling_does_not_start_another_request_after_deadline(self):
        client, mock_acr = make_client_and_mock()
        set_execution_response(mock_acr, status="in_progress")
        handle = client.attach(FAKE_SESSION_ID).get_exec("deadline")
        with (
            patch("agentcore_rl_toolkit.sandbox.client.time.monotonic", side_effect=[0, 0, 0.1]),
            patch("agentcore_rl_toolkit.sandbox.client.time.sleep"),
            pytest.raises(TimeoutError),
        ):
            handle.result(timeout=0.1)
        mock_acr.invoke_agent_runtime.assert_called_once()

    @pytest.mark.parametrize("status", ["interrupted", "not_found"])
    def test_unrecoverable_state_does_not_restart(self, status):
        client, mock_acr = make_client_and_mock()
        set_execution_response(mock_acr, status=status)
        handle = client.attach(FAKE_SESSION_ID).get_exec("old")
        assert handle.status() == status
        with pytest.raises(ExecError, match=status) as caught:
            handle.result()
        assert caught.value.handle is handle
        mock_acr.stop_runtime_session.assert_not_called()

    def test_persisted_execution_error(self):
        client, mock_acr = make_client_and_mock()
        set_execution_response(mock_acr, error={"code": "execution_failed", "message": "missing shell"})
        handle = client.attach(FAKE_SESSION_ID).get_exec("error")
        with pytest.raises(ExecError, match="missing shell"):
            handle.result()

    @pytest.mark.parametrize("invocation_id", ["", "../escape", "bad/id", "a" * 129])
    def test_invalid_id(self, invocation_id):
        client, mock_acr = make_client_and_mock()
        with pytest.raises(ValueError, match="invocation_id"):
            client.attach(FAKE_SESSION_ID).get_exec(invocation_id)
        mock_acr.invoke_agent_runtime.assert_not_called()

    def test_handle_cannot_query_terminated_session(self):
        client, mock_acr = make_client_and_mock(make_start_response())
        sandbox = client.attach(FAKE_SESSION_ID)
        handle = sandbox.get_exec("test")
        sandbox.terminate()
        calls = mock_acr.invoke_agent_runtime.call_count
        with pytest.raises(RuntimeError, match="is terminated"):
            handle.result()
        assert mock_acr.invoke_agent_runtime.call_count == calls


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
        assert kwargs["qualifier"] == "DEFAULT"
        assert kwargs["clientToken"]

    def test_terminate_forwards_custom_qualifier(self):
        """A non-DEFAULT qualifier must reach stop_runtime_session, not just start/exec."""
        with patch("agentcore_rl_toolkit.sandbox.client.boto3") as mock_boto3:
            mock_acr = MagicMock()
            mock_boto3.client.return_value = mock_acr
            client = SandboxClient(runtime_arn=FAKE_ARN, qualifier="prod-endpoint")
        mock_acr.invoke_agent_runtime.return_value = make_start_response({"status": "ok", "state": "healthy"})
        client.attach(FAKE_SESSION_ID).terminate()
        assert mock_acr.invoke_agent_runtime.call_args.kwargs["qualifier"] == "prod-endpoint"
        assert mock_acr.stop_runtime_session.call_args.kwargs["qualifier"] == "prod-endpoint"

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


class TestExecAfterTerminate:
    def test_exec_after_terminate_raises(self):
        client, mock_acr = make_client_and_mock()
        mock_acr.invoke_agent_runtime.return_value = make_start_response({"status": "ok", "state": "healthy"})
        sandbox = client.attach(FAKE_SESSION_ID)
        sandbox.terminate()
        with pytest.raises(RuntimeError, match="is terminated"):
            sandbox.exec("echo hi")
        mock_acr.invoke_agent_runtime_command.assert_not_called()

    def test_exec_after_context_exit_raises(self):
        client, _ = make_client_and_mock(make_start_response())
        with client.start() as sandbox:
            pass
        with pytest.raises(RuntimeError, match="is terminated"):
            sandbox.exec("echo hi")


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
