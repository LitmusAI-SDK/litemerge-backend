from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, Field


class RunCreateRequest(BaseModel):
    project_id: str = Field(min_length=3, max_length=128)
    test_suite: Literal["standard", "adversarial", "full"] = "standard"
    fail_threshold: int = Field(default=70, ge=0, le=100)
    notify_webhook: AnyHttpUrl | None = None
    # Optional overrides — when set, these take precedence over the suite defaults.
    persona_ids: list[str] | None = Field(
        default=None,
        description="Explicit persona IDs to run (e.g. ['p1','p4']). Overrides test_suite mapping.",
        max_length=20,
    )
    turns_per_session: int | None = Field(
        default=None,
        ge=1,
        le=30,
        description="Number of conversation turns per persona session. Defaults to 8.",
    )


class RunCreateResponse(BaseModel):
    run_id: str
    status: Literal["queued"]
    status_url: str
    estimated_duration_s: int


class SessionStatus(BaseModel):
    persona_id: str
    persona_name: str | None = None
    persona_type: str | None = None
    status: Literal["in_progress", "completed", "failed"]
    turns_completed: int


class RunStatusResponse(BaseModel):
    run_id: str
    status: Literal["queued", "running", "evaluating", "complete", "failed"]
    score: int | None = None
    passed: bool | None = None
    report_url: str | None = None
    summary: dict | None = None
    session_statuses: list[SessionStatus] = []


class RunListItem(BaseModel):
    run_id: str
    project_id: str
    test_suite: Literal["standard", "adversarial", "full"]
    status: Literal["queued", "running", "evaluating", "complete", "failed"]
    score: int | None = None
    passed: bool | None = None
    fail_threshold: int
    created_at: str
    completed_at: str | None = None
