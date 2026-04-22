from datetime import datetime

from pydantic import BaseModel


class FindingTypeSummary(BaseModel):
    finding_type: str
    count: int
    by_severity: dict[str, int]


class ReportSessionStatus(BaseModel):
    persona_id: str
    persona_name: str | None = None
    persona_type: str | None = None
    status: str
    turns_completed: int


class RunReportResponse(BaseModel):
    run_id: str
    status: str
    score: int | None
    passed: bool | None
    fail_threshold: int
    test_suite: str
    created_at: datetime
    completed_at: datetime | None
    summary: dict | None
    findings_count: int
    findings_by_severity: dict[str, int]
    findings_by_type: list[FindingTypeSummary]
    findings: list[dict]
    session_statuses: list[ReportSessionStatus] = []
