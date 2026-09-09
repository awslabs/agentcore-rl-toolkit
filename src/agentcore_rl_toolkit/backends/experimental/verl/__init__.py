"""What every verl worker needs, imported via ``VERL_USE_EXTERNAL_MODULES``.

The agent loops are star-imported here because their ``@register_agent_loop``
decorators must have run in every rollout worker.
"""

from .rollout_session_agent_loop import *  # noqa: F403
