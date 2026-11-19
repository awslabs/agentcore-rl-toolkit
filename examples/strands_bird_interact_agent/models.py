from pydantic import BaseModel, ConfigDict, Field


class InvocationRequest(BaseModel):
    """One BIRD-Interact task. Its question, gold SQL and test cases stay on the environment service."""

    model_config = ConfigDict(populate_by_name=True)

    instance_id: str
    config: dict = Field(alias="_config")  # model_id, and base_url (OpenAI-compatible) or bedrock_fields (Bedrock)
