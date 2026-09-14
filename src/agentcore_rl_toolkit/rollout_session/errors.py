"""The one exception type that separates a broken run from a failed rollout.

A rollout stops for two very different reasons, and a trainer must treat them
differently:

* **A failed rollout** -- a container that never came up, a timeout, an agent handler
  that raised. Expected at some rate, so what the agent loop reports for it is a policy
  choice (``on_rollout_failure``), and one that a healthy run survives either way.
* **A contract violation** -- a dataset with no ``payload`` column, a row with no
  ``task_id``, a reward function returning a string. These recur on *every* rollout of
  the run, so absorbing them would spend a whole training job producing nothing trainable
  and one warning per step.

Only the second kind always raises out of the loop, whatever the failure policy says --
which is what stops a run. Raising it is therefore a deliberate assertion that retrying
cannot help.

:class:`ValueError` remains the base so existing callers (and tests) that catch
``ValueError`` are unaffected.
"""


class RolloutContractError(ValueError):
    """A rollout could not be attempted, or scored, because the setup is wrong.

    Raise it for conditions no rollout of this run can satisfy; never for a rollout that
    merely failed.
    """


__all__ = ["RolloutContractError"]
