from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

FindingType = Literal[
    "prompt_injection_success",
    "boundary_violation",
    "hallucination",
    "inappropriate_response",
    "refusal_failure",
]

Severity = Literal["critical", "high", "medium", "low"]


class FindingCreateRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=128)
    run_id: str | None = Field(default=None, max_length=128)
    persona_type: str = Field(min_length=1, max_length=64)
    finding_type: FindingType
    severity: Severity
    prompt_vector: str | None = Field(default=None, max_length=4096)
    agent_response_excerpt: str | None = Field(default=None, max_length=4096)


class FindingResponse(BaseModel):
    id: str
    project_id: str
    run_id: str | None
    persona_type: str
    finding_type: str
    severity: str
    prompt_vector: str | None
    agent_response_excerpt: str | None
    created_at: datetime


class FindingListResponse(BaseModel):
    items: list[FindingResponse]
    total: int
