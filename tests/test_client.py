"""Tests for RolloutClient, RolloutFuture, and BatchResult."""

import io
import json
import time
from concurrent.futures import CancelledError
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from agentcore_rl_toolkit import BatchItem, BatchResult, RolloutClient, RolloutFuture


def mock_streaming_body(data: dict) -> io.BytesIO:
    """Create a mock StreamingBody-like object for ACR responses."""
    return io.BytesIO(json.dumps(data).encode())


class TestRolloutFuture:
    """Tests for RolloutFuture S3 polling behavior."""

    def test_done_returns_true_when_object_exists(self):
        """Test done() returns True when S3 HEAD succeeds."""
        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {}

        future = RolloutFuture(
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            result_key="exp/input_session.json",
        )

        assert future.done() is True
        assert future._done is True
        mock_s3.head_object.assert_called_once_with(Bucket="test-bucket", Key="exp/input_session.json")

    def test_done_returns_false_on_404(self):
        """Test done() returns False when S3 returns 404."""
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}},
            "HeadObject",
        )

        future = RolloutFuture(
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            result_key="exp/input_session.json",
        )

        assert future.done() is False
        assert future._done is False

    def test_done_updates_backoff_on_404(self):
        """Test done() increases poll interval after 404."""
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}},
            "HeadObject",
        )

        future = RolloutFuture(
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            result_key="exp/input_session.json",
            initial_interval=1.0,
            backoff_factor=2.0,
            max_interval=10.0,
        )

        assert future._poll_interval == 1.0
        # Each done() call polls S3 HEAD; on 404, it increases the backoff interval
        future.done()
        assert future._poll_interval == 2.0  # 1.0 * 2.0
        future.done()
        assert future._poll_interval == 4.0  # 2.0 * 2.0
        future.done()
        assert future._poll_interval == 8.0  # 4.0 * 2.0
        future.done()
        assert future._poll_interval == 10.0  # capped at max_interval

    def test_done_caches_result(self):
        """Test done() only calls S3 once after success."""
        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {}

        future = RolloutFuture(
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            result_key="exp/input_session.json",
        )

        assert future.done() is True
        assert future.done() is True
        assert future.done() is True
        mock_s3.head_object.assert_called_once()

    def test_done_raises_on_non_404_error(self):
        """Test done() raises on non-404 errors."""
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}},
            "HeadObject",
        )

        future = RolloutFuture(
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            result_key="exp/input_session.json",
        )

        with pytest.raises(ClientError):
            future.done()

    def test_result_returns_parsed_json(self):
        """Test result() fetches and parses S3 object."""
        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {}

        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"rollout_data": [], "rewards": [1.0]}).encode()
        mock_s3.get_object.return_value = {"Body": mock_body}

        future = RolloutFuture(
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            result_key="exp/input_session.json",
        )

        result = future.result()

        assert result == {"rollout_data": [], "rewards": [1.0]}
        mock_s3.get_object.assert_called_once_with(Bucket="test-bucket", Key="exp/input_session.json")

    def test_result_caches_result(self):
        """Test result() only fetches once."""
        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {}

        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"data": "test"}).encode()
        mock_s3.get_object.return_value = {"Body": mock_body}

        future = RolloutFuture(
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            result_key="exp/input_session.json",
        )

        result1 = future.result()
        result2 = future.result()

        assert result1 == result2
        mock_s3.get_object.assert_called_once()

    def test_time_until_next_poll(self):
        """Test time_until_next_poll() returns correct value."""
        mock_s3 = MagicMock()

        future = RolloutFuture(
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            result_key="exp/input_session.json",
            initial_interval=1.0,
        )

        # Before any poll, should be ready immediately
        assert future.time_until_next_poll() <= 0

        # Simulate a poll that returned 404
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}},
            "HeadObject",
        )
        future.done()

        # Just after poll, should wait ~1.5s (initial_interval * backoff_factor)
        wait_time = future.time_until_next_poll()
        assert wait_time > 0
        assert wait_time <= future._poll_interval

    def test_ready_to_poll(self):
        """Test ready_to_poll() returns True when interval elapsed."""
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}},
            "HeadObject",
        )

        future = RolloutFuture(
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            result_key="exp/input_session.json",
            initial_interval=0.1,  # Short interval for testing
        )

        # Before any poll
        assert future.ready_to_poll() is True

        # Right after poll
        future.done()
        assert future.ready_to_poll() is False

        # After waiting
        time.sleep(0.2)
        assert future.ready_to_poll() is True

    def test_cancel_stops_session(self):
        """Test cancel() calls stop_runtime_session with correct args and returns True."""
        mock_s3 = MagicMock()
        mock_acr = MagicMock()

        future = RolloutFuture(
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            result_key="exp/input_session.json",
            session_id="sess-12345678-abcd",
            agentcore_client=mock_acr,
            agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:agent/test",
        )

        assert future.cancel() is True
        assert future.cancelled is True
        mock_acr.stop_runtime_session.assert_called_once_with(
            agentRuntimeArn="arn:aws:bedrock-agentcore:us-west-2:123:agent/test",
            runtimeSessionId="sess-12345678-abcd",
        )

    def test_cancel_is_idempotent(self):
        """Test second cancel() is a no-op and API is called only once."""
        mock_s3 = MagicMock()
        mock_acr = MagicMock()

        future = RolloutFuture(
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            result_key="exp/input_session.json",
            session_id="sess-12345678-abcd",
            agentcore_client=mock_acr,
            agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:agent/test",
        )

        assert future.cancel() is True
        assert future.cancel() is False
        mock_acr.stop_runtime_session.assert_called_once()

    def test_cancel_without_client(self):
        """Test cancel() returns False when agentcore_client is None."""
        mock_s3 = MagicMock()

        future = RolloutFuture(
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            result_key="exp/input_session.json",
        )

        assert future.cancel() is False
        assert future.cancelled is True

    def test_cancel_handles_exception(self):
        """Test cancel() returns False when API raises, still marks cancelled."""
        mock_s3 = MagicMock()
        mock_acr = MagicMock()
        mock_acr.stop_runtime_session.side_effect = Exception("Service unavailable")

        future = RolloutFuture(
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            result_key="exp/input_session.json",
            session_id="sess-12345678-abcd",
            agentcore_client=mock_acr,
            agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:agent/test",
        )

        assert future.cancel() is False
        assert future.cancelled is True

    def test_done_returns_true_after_cancel(self):
        """Test done() returns True after cancel (cancelled is a terminal state)."""
        mock_s3 = MagicMock()

        future = RolloutFuture(
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            result_key="exp/input_session.json",
        )

        future.cancel()
        assert future.done() is True
        # S3 should not be polled after cancellation
        mock_s3.head_object.assert_not_called()

    def test_result_raises_after_cancel(self):
        """Test result() raises CancelledError after cancel."""
        mock_s3 = MagicMock()

        future = RolloutFuture(
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            result_key="exp/input_session.json",
        )

        future.cancel()
        with pytest.raises(CancelledError):
            future.result()


class TestRolloutClient:
    """Tests for RolloutClient."""

    def test_init_sets_attributes(self):
        """Test RolloutClient initializes correctly."""
        with patch("agentcore_rl_toolkit.client.boto3"):
            client = RolloutClient(
                agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:agent/test",
                s3_bucket="test-bucket",
                exp_id="exp-001",
                tps_limit=10,
                base_url="http://localhost:8000",
                model_id="test-model",
                temperature=0.7,
            )

            assert client.agent_runtime_arn == "arn:aws:bedrock-agentcore:us-west-2:123:agent/test"
            assert client.s3_bucket == "test-bucket"
            assert client.exp_id == "exp-001"
            assert client.region == "us-west-2"  # Inferred from ARN
            assert client.tps_limit == 10
            assert client.base_url == "http://localhost:8000"
            assert client.model_id == "test-model"
            assert client.extra_config == {"temperature": 0.7}

    def test_parse_region_from_arn_valid(self):
        """Test _parse_region_from_arn extracts region correctly."""
        assert (
            RolloutClient._parse_region_from_arn("arn:aws:bedrock-agentcore:us-west-2:123456789012:agent/my-agent")
            == "us-west-2"
        )
        assert RolloutClient._parse_region_from_arn("arn:aws:bedrock-agentcore:eu-west-2:123:agent/test") == "eu-west-2"
        assert (
            RolloutClient._parse_region_from_arn("arn:aws-cn:bedrock-agentcore:cn-north-1:123:agent/test")
            == "cn-north-1"
        )

    def test_parse_region_from_arn_invalid(self):
        """Test _parse_region_from_arn raises on invalid ARNs."""
        with pytest.raises(ValueError, match="Invalid ARN format"):
            RolloutClient._parse_region_from_arn("not-an-arn")
        with pytest.raises(ValueError, match="Invalid ARN format"):
            RolloutClient._parse_region_from_arn("arn:aws:service")
        with pytest.raises(ValueError, match="Invalid ARN format"):
            RolloutClient._parse_region_from_arn("arn:aws:service::account:resource")  # Empty region

    def test_invoke_returns_future(self):
        """Test invoke() returns a RolloutFuture."""
        with patch("agentcore_rl_toolkit.client.boto3") as mock_boto3:
            mock_acr = MagicMock()
            mock_s3 = MagicMock()
            mock_boto3.client.side_effect = lambda service, **kwargs: (
                mock_acr if service == "bedrock-agentcore" else mock_s3
            )

            # Mock ACR response with StreamingBody-like object
            mock_acr.invoke_agent_runtime.return_value = {
                "response": mock_streaming_body(
                    {"status": "processing", "s3_bucket": "test-bucket", "result_key": "exp/key.json"}
                )
            }

            client = RolloutClient(
                agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:agent/test",
                s3_bucket="test-bucket",
                exp_id="exp-001",
            )

            future = client.invoke({"prompt": "test"})

            assert isinstance(future, RolloutFuture)
            assert future.s3_bucket == "test-bucket"
            assert future.result_key == "exp/key.json"

    def test_invoke_builds_training_config(self):
        """Test invoke() correctly builds _training config."""
        with patch("agentcore_rl_toolkit.client.boto3") as mock_boto3:
            mock_acr = MagicMock()
            mock_s3 = MagicMock()
            mock_boto3.client.side_effect = lambda service, **kwargs: (
                mock_acr if service == "bedrock-agentcore" else mock_s3
            )

            mock_acr.invoke_agent_runtime.return_value = {
                "response": mock_streaming_body({"status": "processing", "s3_bucket": "bucket", "result_key": "key"})
            }

            client = RolloutClient(
                agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:agent/test",
                s3_bucket="test-bucket",
                exp_id="exp-001",
                base_url="http://localhost:8000",
                model_id="test-model",
                temperature=0.7,
            )

            client.invoke({"prompt": "test"}, session_id="sess-1", input_id="input-1")

            # Check the payload sent to ACR
            call_args = mock_acr.invoke_agent_runtime.call_args
            payload = json.loads(call_args.kwargs["payload"])

            assert payload["prompt"] == "test"
            assert payload["_training"]["exp_id"] == "exp-001"
            assert payload["_training"]["session_id"] == "sess-1"
            assert payload["_training"]["input_id"] == "input-1"
            assert payload["_training"]["s3_bucket"] == "test-bucket"
            assert payload["_training"]["base_url"] == "http://localhost:8000"
            assert payload["_training"]["model_id"] == "test-model"
            assert payload["_training"]["temperature"] == 0.7

    def test_invoke_future_has_session_id(self):
        """Test invoke() returns future with session_id and agentcore_client set."""
        with patch("agentcore_rl_toolkit.client.boto3") as mock_boto3:
            mock_acr = MagicMock()
            mock_s3 = MagicMock()
            mock_boto3.client.side_effect = lambda service, **kwargs: (
                mock_acr if service == "bedrock-agentcore" else mock_s3
            )

            mock_acr.invoke_agent_runtime.return_value = {
                "response": mock_streaming_body(
                    {"status": "processing", "s3_bucket": "test-bucket", "result_key": "exp/key.json"}
                )
            }

            client = RolloutClient(
                agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:agent/test",
                s3_bucket="test-bucket",
                exp_id="exp-001",
            )

            future = client.invoke({"prompt": "test"}, session_id="my-session")

            assert future.session_id == "my-session"
            assert future.agentcore_client is mock_acr
            assert future.agent_runtime_arn == "arn:aws:bedrock-agentcore:us-west-2:123:agent/test"

    def test_run_batch_returns_batch_result(self):
        """Test run_batch() returns a BatchResult."""
        with patch("agentcore_rl_toolkit.client.boto3"):
            client = RolloutClient(
                agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:agent/test",
                s3_bucket="test-bucket",
                exp_id="exp-001",
            )

            payloads = [{"prompt": "q1"}, {"prompt": "q2"}]
            result = client.run_batch(payloads, max_concurrent_sessions=50)

            assert isinstance(result, BatchResult)
            assert result.payloads == payloads
            assert result.max_concurrent == 50


class TestBatchResult:
    """Tests for BatchResult."""

    def test_batch_yields_all_results(self):
        """Test iterating over BatchResult yields all results."""
        with patch("agentcore_rl_toolkit.client.boto3") as mock_boto3:
            mock_acr = MagicMock()
            mock_s3 = MagicMock()
            mock_boto3.client.side_effect = lambda service, **kwargs: (
                mock_acr if service == "bedrock-agentcore" else mock_s3
            )

            # Track invoke calls to generate unique keys
            invoke_count = [0]

            def mock_invoke(*args, **kwargs):
                invoke_count[0] += 1
                return {
                    "response": mock_streaming_body(
                        {
                            "status": "processing",
                            "s3_bucket": "bucket",
                            "result_key": f"key{invoke_count[0]}.json",
                        }
                    )
                }

            mock_acr.invoke_agent_runtime.side_effect = mock_invoke
            mock_s3.head_object.return_value = {}

            # Return different results for each key
            def mock_get_object(Bucket, Key):
                idx = int(Key.replace("key", "").replace(".json", ""))
                body = MagicMock()
                body.read.return_value = json.dumps({"result": idx}).encode()
                return {"Body": body}

            mock_s3.get_object.side_effect = mock_get_object

            client = RolloutClient(
                agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:agent/test",
                s3_bucket="test-bucket",
                exp_id="exp-001",
                tps_limit=1000,  # High limit to speed up test
            )

            payloads = [{"prompt": "q1"}, {"prompt": "q2"}, {"prompt": "q3"}]

            # Stream results as they complete
            items = []
            for item in client.run_batch(payloads, max_concurrent_sessions=10):
                assert isinstance(item, BatchItem)
                assert item.success is True
                assert item.error is None
                items.append(item)

            assert len(items) == 3
            # Results may not be in order since they complete as they're polled
            result_values = sorted([item.result["result"] for item in items])
            assert result_values == [1, 2, 3]
            # All successful items should have a positive elapsed time
            for item in items:
                assert item.elapsed is not None
                assert item.elapsed >= 0

    def test_batch_continues_on_invocation_error(self):
        """Test that batch continues processing when one invocation fails."""
        with patch("agentcore_rl_toolkit.client.boto3") as mock_boto3:
            mock_acr = MagicMock()
            mock_s3 = MagicMock()
            mock_boto3.client.side_effect = lambda service, **kwargs: (
                mock_acr if service == "bedrock-agentcore" else mock_s3
            )

            # First call succeeds, second fails, third succeeds
            invoke_count = [0]

            def mock_invoke(*args, **kwargs):
                invoke_count[0] += 1
                if invoke_count[0] == 2:
                    raise Exception("ACR invocation failed")
                return {
                    "response": mock_streaming_body(
                        {
                            "status": "processing",
                            "s3_bucket": "bucket",
                            "result_key": f"key{invoke_count[0]}.json",
                        }
                    )
                }

            mock_acr.invoke_agent_runtime.side_effect = mock_invoke
            mock_s3.head_object.return_value = {}

            def mock_get_object(Bucket, Key):
                idx = int(Key.replace("key", "").replace(".json", ""))
                body = MagicMock()
                body.read.return_value = json.dumps({"result": idx}).encode()
                return {"Body": body}

            mock_s3.get_object.side_effect = mock_get_object

            client = RolloutClient(
                agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:agent/test",
                s3_bucket="test-bucket",
                exp_id="exp-001",
                tps_limit=1000,
            )

            payloads = [{"prompt": "q1"}, {"prompt": "q2"}, {"prompt": "q3"}]

            # Stream results, separating successes and errors
            successes = []
            errors = []
            for item in client.run_batch(payloads, max_concurrent_sessions=10):
                if item.success:
                    successes.append(item)
                else:
                    errors.append(item)

            assert len(successes) == 2
            assert len(errors) == 1

            # Error item should have index and error message
            assert errors[0].index == 1
            assert "ACR invocation failed" in errors[0].error
            assert errors[0].elapsed == 0.0

            # Successful items should have positive elapsed time
            for item in successes:
                assert item.elapsed is not None
                assert item.elapsed >= 0

    def test_batch_times_out_slow_requests(self):
        """Test that batch times out requests that exceed the timeout."""
        with patch("agentcore_rl_toolkit.client.boto3") as mock_boto3:
            mock_acr = MagicMock()
            mock_s3 = MagicMock()
            mock_boto3.client.side_effect = lambda service, **kwargs: (
                mock_acr if service == "bedrock-agentcore" else mock_s3
            )

            mock_acr.invoke_agent_runtime.return_value = {
                "response": mock_streaming_body(
                    {
                        "status": "processing",
                        "s3_bucket": "bucket",
                        "result_key": "key1.json",
                    }
                )
            }

            # S3 HEAD always returns 404 (result never ready)
            mock_s3.head_object.side_effect = ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
            )

            client = RolloutClient(
                agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:agent/test",
                s3_bucket="test-bucket",
                exp_id="exp-001",
                tps_limit=1000,
            )

            payloads = [{"prompt": "q1"}]

            # Use very short timeout for testing
            items = list(client.run_batch(payloads, max_concurrent_sessions=1, timeout=0.1))

            assert len(items) == 1
            assert items[0].success is False
            assert "Timeout" in items[0].error
            assert items[0].index == 0
            assert items[0].elapsed >= 0.1

    def test_batch_timeout_cancels_session(self):
        """Test that batch timeout triggers stop_runtime_session call."""
        with patch("agentcore_rl_toolkit.client.boto3") as mock_boto3:
            mock_acr = MagicMock()
            mock_s3 = MagicMock()
            mock_boto3.client.side_effect = lambda service, **kwargs: (
                mock_acr if service == "bedrock-agentcore" else mock_s3
            )

            mock_acr.invoke_agent_runtime.return_value = {
                "response": mock_streaming_body(
                    {
                        "status": "processing",
                        "s3_bucket": "bucket",
                        "result_key": "key1.json",
                    }
                )
            }

            # S3 HEAD always returns 404 (result never ready)
            mock_s3.head_object.side_effect = ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
            )

            client = RolloutClient(
                agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:agent/test",
                s3_bucket="test-bucket",
                exp_id="exp-001",
                tps_limit=1000,
            )

            payloads = [{"prompt": "q1"}]

            # Use very short timeout for testing
            items = list(client.run_batch(payloads, max_concurrent_sessions=1, timeout=0.1))

            assert len(items) == 1
            assert items[0].success is False
            assert "Timeout" in items[0].error

            # Verify stop_runtime_session was called
            mock_acr.stop_runtime_session.assert_called_once()
            call_kwargs = mock_acr.stop_runtime_session.call_args.kwargs
            assert call_kwargs["agentRuntimeArn"] == "arn:aws:bedrock-agentcore:us-west-2:123:agent/test"
            assert "runtimeSessionId" in call_kwargs
