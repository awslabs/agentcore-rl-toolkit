from ...app import AgentCoreRLApp


class StrandsAgentCoreRLApp(AgentCoreRLApp):
    def create_openai_compatible_model(self, **kwargs):
        """Create Strands OpenAI-compatible model for vLLM/SGLang server."""
        try:
            from strands.models.openai import OpenAIModel
        except ImportError:
            raise ImportError("Strands not installed. Install with: uv pip install strands-agents[openai]") from None

        base_url, model_id = self._get_model_config()

        return OpenAIModel(client_args={"api_key": "dummy", "base_url": base_url}, model_id=model_id, **kwargs)
