"""HTTP invocation lifecycle for AgentCore Runtime applications."""

from importlib import import_module

_EXPORTS = {
    "AgentCoreRuntimeApp": ".app",
    "AgentCoreHttpClient": ".client",
    "InvocationHandle": ".client",
    "InvocationError": ".client",
    "sagemaker_mtrl": ".sagemaker_mtrl",
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
