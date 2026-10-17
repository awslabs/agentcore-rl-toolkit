"""Public SDK entry points, loaded when first accessed."""

from importlib import import_module

_EXPORTS = {
    "AgentCoreRLApp": ".app",
    "AgentCoreRuntimeApp": ".runtime.app",
    "AgentCoreHttpClient": ".runtime.client",
    "InvocationHandle": ".runtime.client",
    "InvocationError": ".runtime.client",
    "RewardFunction": ".reward_function",
    "RolloutClient": ".client",
    "RolloutFuture": ".client",
    "BatchResult": ".client",
    "AsyncBatchResult": ".client",
    "BatchItem": ".client",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
