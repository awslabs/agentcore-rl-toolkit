from pydantic import BaseModel, ConfigDict, Field


class InvocationRequest(BaseModel):
    """One Seal-Hard task. Typed fields reject non-string input (e.g. toolUse blocks) before it reaches the agent."""

    model_config = ConfigDict(populate_by_name=True)

    question: str
    gold: str = ""  # the reference answer, for the judge
    config: dict = Field(alias="_config")  # model_id, and base_url (OpenAI-compatible) or bedrock_fields (Bedrock)
