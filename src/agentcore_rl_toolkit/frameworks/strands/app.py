from ...app import AgentCoreRLApp


class StrandsAgentCoreRLApp(AgentCoreRLApp):
    def create_openai_compatible_model(self, provider_model_id=None, **kwargs):
        """
        Create Strands model that's compatible with the OpenAI format. When provider_model_id
        is provided, LiteLLM model will be used. Otherwise, an OpenAI compatible model with
        base_url and model_id will be used.

        :param provider_model_id: Provide this parameter when using cloud providers (bedrock,
        anthropic, openai, etc.) that does not use a base_url. Example: Otherwise, leave it to None.
        """
        try:
            from strands.models.openai import OpenAIModel
        except ImportError:
            raise ImportError("Strands not installed. Install with: " "uv pip install strands-agents[openai]") from None

        if not provider_model_id:
            base_url, model_id = self._get_model_config()
            return OpenAIModel(client_args={"api_key": "dummy", "base_url": base_url}, model_id=model_id, **kwargs)

        try:
            from strands.models.litellm import LiteLLMModel
        except ImportError:
            raise ImportError(
                "Strands not installed. Install with: " "uv pip install strands-agents[litellm]"
            ) from None

        return LiteLLMModel(model_id=provider_model_id, **kwargs)
