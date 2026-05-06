import json
import logging
import traceback
from datetime import datetime, timezone


class CorrelatedFormatter(logging.Formatter):
    """JSON formatter that injects sessionId and requestId from ACR request context."""

    def format(self, record):
        from bedrock_agentcore.runtime import BedrockAgentCoreContext

        log_entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        request_id = BedrockAgentCoreContext.get_request_id()
        if request_id:
            log_entry["requestId"] = request_id

        session_id = BedrockAgentCoreContext.get_session_id()
        if session_id:
            log_entry["sessionId"] = session_id

        if record.exc_info and record.exc_info[0]:
            log_entry["errorType"] = record.exc_info[0].__name__
            log_entry["errorMessage"] = str(record.exc_info[1])
            log_entry["stackTrace"] = traceback.format_exception(*record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def configure_logging(level=logging.INFO):
    """Attach CorrelatedFormatter to the root logger for automatic sessionId injection.

    Idempotent — safe to call multiple times.
    """
    root = logging.getLogger()
    if getattr(root, "_art_logging_configured", False):
        return

    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(CorrelatedFormatter())
    root.addHandler(handler)
    # Use the more verbose level: honors user's basicConfig(level=DEBUG) even though we default to INFO.
    # Root default is WARNING (30); min(30, 20) = INFO; min(10, 20) = DEBUG.
    root.setLevel(min(root.level, level))

    # The SDK's bedrock_agentcore.app logger already has its own JSON handler.
    # Suppress propagation to avoid duplicate output.
    logging.getLogger("bedrock_agentcore.app").propagate = False

    root._art_logging_configured = True
