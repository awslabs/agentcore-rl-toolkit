from pydantic import BaseModel


class InvocationRequest(BaseModel):
    # `prompt` is typed `str` on purpose: it is passed straight to the agent, and
    # pydantic rejects any non-string value (e.g. a list of content blocks) before
    # the agent runs. Do NOT relax this to `str | list` / `Any` — accepting a
    # `toolUse` content block here would let a caller bypass model invocation and
    # dispatch a tool directly (Strands event-loop dispatch, H1-3679111).
    prompt: str
    # Ground truth for reward computation (RL entrypoint only); optional so the
    # production entrypoint can accept the same payload shape.
    answer: str | None = None
