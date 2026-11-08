from pydantic import BaseModel, ConfigDict, Field


class InvocationRequest(BaseModel):
    """One HERB question. Typed fields reject non-string input (e.g. toolUse blocks) before it reaches the agent."""

    model_config = ConfigDict(populate_by_name=True)

    question: str
    type: str  # content | person | company | url | pr (answerable), or unanswerable
    gold: str | list[str] = ""  # the reference: a paragraph (content) or a set of items; empty when unanswerable
    config: dict = Field(alias="_config")  # model_id, and base_url (OpenAI-compatible) or bedrock_fields (Bedrock)
