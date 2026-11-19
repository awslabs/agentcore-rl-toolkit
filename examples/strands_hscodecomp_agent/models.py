from pydantic import BaseModel, ConfigDict, Field


class InvocationRequest(BaseModel):
    """One HSCodeComp task. Typed fields reject non-string input (e.g. toolUse blocks) before it reaches the agent."""

    model_config = ConfigDict(populate_by_name=True)

    question: str  # the product listing and the classification instruction
    gold: str = ""  # the 10-digit HTS code
    config: dict = Field(alias="_config")  # model_id, and base_url (OpenAI-compatible) or bedrock_fields (Bedrock)
