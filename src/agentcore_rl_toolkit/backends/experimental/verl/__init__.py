"""What every verl worker needs, imported via ``VERL_USE_EXTERNAL_MODULES``.

Agent loops are star-imported so their registration decorators run in every rollout worker.
"""

from .rollout_session_agent_loop import *  # noqa: F403
