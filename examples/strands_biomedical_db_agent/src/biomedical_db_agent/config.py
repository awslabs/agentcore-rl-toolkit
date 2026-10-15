"""Model configuration, supplied by the deployment or ART rollout payload."""

import json
import os
from typing import Any

from pydantic import BaseModel, Field, SecretStr

from .contracts import AgentName


class ModelConfig(BaseModel):
    base_url: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    api_key: SecretStr = SecretStr("EMPTY")
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    turn_limit: int = Field(default=20, gt=0)
    timeout_seconds: float = Field(default=600, gt=0)

    @classmethod
    def from_env(cls, name: AgentName) -> "ModelConfig":
        prefix = {"opentargets": "OPENTARGETS_MODEL", "gwas_catalog": "GWAS_MODEL"}[name]
        return cls(
            base_url=os.environ[f"{prefix}_BASE_URL"],
            model_id=os.environ[f"{prefix}_ID"],
            api_key=os.getenv(f"{prefix}_API_KEY", "EMPTY"),
            sampling_params=json.loads(os.getenv(f"{prefix}_SAMPLING_PARAMS", "{}")),
            turn_limit=int(os.getenv("DATABASE_AGENT_TURN_LIMIT", "20")),
            timeout_seconds=float(os.getenv("DATABASE_AGENT_TIMEOUT_SECONDS", "600")),
        )
