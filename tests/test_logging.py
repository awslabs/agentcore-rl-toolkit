"""Tests for the structured logging module."""

import json
import logging
from unittest.mock import patch

import pytest

from agentcore_rl_toolkit.logging import CorrelatedFormatter, configure_logging


@pytest.fixture(autouse=True)
def reset_root_logger():
    """Reset root logger state before each test."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    if hasattr(root, "_art_logging_configured"):
        delattr(root, "_art_logging_configured")
    yield
    root.handlers = original_handlers
    root.setLevel(original_level)
    if hasattr(root, "_art_logging_configured"):
        delattr(root, "_art_logging_configured")


class TestCorrelatedFormatter:
    def test_outputs_valid_json(self):
        formatter = CorrelatedFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "hello world"
        assert parsed["logger"] == "test.logger"
        assert "timestamp" in parsed

    @patch("bedrock_agentcore.runtime.BedrockAgentCoreContext.get_session_id", return_value="sess-123")
    @patch("bedrock_agentcore.runtime.BedrockAgentCoreContext.get_request_id", return_value="req-456")
    def test_includes_session_and_request_id(self, mock_req, mock_sess):
        formatter = CorrelatedFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="msg", args=(), exc_info=None
        )

        parsed = json.loads(formatter.format(record))

        assert parsed["sessionId"] == "sess-123"
        assert parsed["requestId"] == "req-456"

    @patch("bedrock_agentcore.runtime.BedrockAgentCoreContext.get_session_id", return_value=None)
    @patch("bedrock_agentcore.runtime.BedrockAgentCoreContext.get_request_id", return_value=None)
    def test_omits_ids_when_no_context(self, mock_req, mock_sess):
        formatter = CorrelatedFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="msg", args=(), exc_info=None
        )

        parsed = json.loads(formatter.format(record))

        assert "sessionId" not in parsed
        assert "requestId" not in parsed

    def test_includes_exception_info(self):
        formatter = CorrelatedFormatter()

        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0, msg="failed", args=(), exc_info=exc_info
        )

        parsed = json.loads(formatter.format(record))

        assert parsed["errorType"] == "ValueError"
        assert parsed["errorMessage"] == "test error"
        assert isinstance(parsed["stackTrace"], list)
        assert len(parsed["stackTrace"]) > 0


class TestConfigureLogging:
    def test_attaches_handler_to_root(self):
        configure_logging()

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, CorrelatedFormatter)

    def test_sets_root_level(self):
        configure_logging(level=logging.DEBUG)

        assert logging.getLogger().level == logging.DEBUG

    def test_idempotent(self):
        configure_logging()
        configure_logging()
        configure_logging()

        assert len(logging.getLogger().handlers) == 1

    def test_clears_existing_handlers(self):
        logging.basicConfig(level=logging.INFO)
        assert len(logging.getLogger().handlers) >= 1

        configure_logging()

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, CorrelatedFormatter)

    def test_suppresses_sdk_logger_propagation(self):
        configure_logging()

        sdk_logger = logging.getLogger("bedrock_agentcore.app")
        assert sdk_logger.propagate is False
