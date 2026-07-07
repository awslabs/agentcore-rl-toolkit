"""HTTP adapters for agent rollouts (Chat Completions / Anthropic Messages)."""

from .anthropic import AnthropicAdapter
from .common import BaseAdapter
from .openai import OpenAIAdapter

__all__ = ["AnthropicAdapter", "BaseAdapter", "OpenAIAdapter"]
