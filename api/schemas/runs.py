from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, Field


class RunCreateRequest(BaseModel):
    project_id: str = Field(min_length=3, max_length=128)
    test_suite: Literal["standard", "adversarial", "full"] = "standard"
    fail_threshold: int = Field(default=70, ge=0, le=100)
    notify_webhook: AnyHttpUrl | None = None


class RunCreateResponse(BaseModel):
    run_id: str
    status: Literal["queued"]
    status_url: str
    estimated_duration_s: int


class RunStatusResponse(BaseModel):
    run_id: str
    status: Literal["queued", "running", "evaluating", "complete", "failed"]
    score: int | None = None
    passed: bool | None = None
    report_url: str | None = None
    summary: dict | None = None
