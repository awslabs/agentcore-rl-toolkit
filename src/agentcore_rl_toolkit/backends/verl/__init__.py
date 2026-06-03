import logging
import os

from agentcore_rl_toolkit.backends.verl.config import (
    AgentCoreConfig,
    AgentCoreFSDPActorConfig,
    AgentCoreMcoreActorConfig,
    AgentCoreRolloutConfig,
)
from agentcore_rl_toolkit.backends.verl.dataset import AgentCoreDataset
from agentcore_rl_toolkit.backends.verl.loop_manager import AgentCoreLoopManager
from agentcore_rl_toolkit.backends.verl.trainer import AgentCoreTrainer


def _configure_package_logger() -> None:
    """Attach a dedicated handler to this package's logger with propagation off.

    Loggers in this package have no logger handler of their own, so their records
    could be silently dropped. Giving the package logger its own handler and setting
    ``propagate = False`` keep our logs flowing regardless of how other libraries
    reconfigure the root logger.
    """
    pkg_logger = logging.getLogger(__name__)
    if any(getattr(h, "_agentcore_verl_handler", False) for h in pkg_logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(asctime)s:%(message)s"))
    handler._agentcore_verl_handler = True
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
    pkg_logger.propagate = False


_configure_package_logger()

__all__ = [
    "AgentCoreConfig",
    "AgentCoreDataset",
    "AgentCoreFSDPActorConfig",
    "AgentCoreLoopManager",
    "AgentCoreMcoreActorConfig",
    "AgentCoreRolloutConfig",
    "AgentCoreTrainer",
]
