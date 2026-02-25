from ...app import AgentCoreRLApp


class StrandsAgentCoreRLApp(AgentCoreRLApp):
    def create_openai_compatible_model(self, provider_model_id=None, capture_tokens=False, **kwargs):
        """
        Create Strands model that's compatible with the OpenAI format.

        When ``provider_model_id`` is provided, LiteLLM model will be used
        (for evaluation against cloud providers).

        When ``capture_tokens=True``, an :class:`RLLMRemoteModel` is
        returned.  This model calls rLLM's ``/v1/model_response`` endpoint
        and captures token-level data (``prompt_ids``, ``completion_ids``,
        ``logprobs``) needed for RL training.  Call
        ``model.get_model_outputs()`` after each episode to retrieve the
        stored outputs.

        When ``capture_tokens=False`` (the default), a standard
        ``OpenAIModel`` is returned that uses ``/v1/chat/completions``.

        :param provider_model_id: Model ID for cloud providers (bedrock,
            anthropic, openai, etc.).  Leave ``None`` to use BASE_URL.
        :param capture_tokens: If ``True``, return an
            :class:`RLLMRemoteModel` that captures token-level data.
        """
        if provider_model_id:
            try:
                from strands.models.litellm import LiteLLMModel
            except ImportError:
                raise ImportError(
                    "Strands not installed. Install with: uv pip install strands-agents[litellm]"
                ) from None
            return LiteLLMModel(model_id=provider_model_id, **kwargs)

        base_url, model_id = self._get_model_config()

        if capture_tokens:
            from .rllm_model import RLLMRemoteModel

            return RLLMRemoteModel(base_url=base_url, model_id=model_id, **kwargs)

        try:
            from strands.models.openai import OpenAIModel
        except ImportError:
            raise ImportError(
                "Strands not installed. Install with: uv pip install strands-agents[openai]"
            ) from None

        return OpenAIModel(client_args={"api_key": "dummy", "base_url": base_url}, model_id=model_id, **kwargs)
