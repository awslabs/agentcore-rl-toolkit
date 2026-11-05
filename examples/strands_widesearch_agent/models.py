from pydantic import BaseModel, ConfigDict, Field


class InvocationRequest(BaseModel):
    """One WideSearch task. Typed fields reject non-string input (e.g. toolUse blocks) before it reaches the agent."""

    model_config = ConfigDict(populate_by_name=True)

    instance_id: str
    question: str  # the query
    evaluation: dict  # the official evaluation spec: unique columns, required columns, per-column metrics
    gold_csv: str  # the gold table
    config: dict = Field(alias="_config")  # model_id, and base_url (OpenAI-compatible) or bedrock_fields (Bedrock)
