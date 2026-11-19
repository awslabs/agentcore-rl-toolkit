from pydantic import BaseModel, ConfigDict, Field


class InvocationRequest(BaseModel):
    """One ExCyTIn question. Typed fields reject non-string input (e.g. toolUse blocks) before it reaches the agent."""

    model_config = ConfigDict(populate_by_name=True)

    incident: int  # the incident whose database the agent queries
    context: str = ""
    question: str
    gold: str = ""  # the answer, for the judge
    solution: list[str] = []  # the gold solution's steps, for the judge's partial credit
    config: dict = Field(alias="_config")  # model_id, and base_url (OpenAI-compatible) or bedrock_fields (Bedrock)
