"""Reports routes — structured post-run evaluation reports."""

import logging

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Request, status

from api.schemas.reports import FindingTypeSummary, RunReportResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])

# Severity display order for consistent output
_SEVERITY_ORDER = ["critical", "high", "medium", "low"]


def _finding_to_dict(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "project_id": doc.get("project_id"),
        "run_id": doc.get("run_id"),
        "persona_type": doc.get("persona_type"),
        "finding_type": doc.get("finding_type"),
        "severity": doc.get("severity"),
        "prompt_vector": doc.get("prompt_vector"),
        "agent_response_excerpt": doc.get("agent_response_excerpt"),
        "created_at": doc.get("created_at"),
    }


@router.get("/{run_id}", response_model=RunReportResponse)
async def get_report(run_id: str, request: Request) -> RunReportResponse:
    """Return a structured evaluation report for a completed run.

    Includes:
    - Run metadata (status, score, pass/fail, suite, timestamps)
    - Aggregated finding counts by severity and type
    - Full findings list ordered critical → high → medium → low
    """
    run_doc = await request.app.state.db["runs"].find_one({"run_id": run_id})
    if not run_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Run not found"
        )

    score = run_doc.get("score")
    fail_threshold = run_doc.get("fail_threshold", 70)
    passed: bool | None = None
    if score is not None:
        passed = score >= fail_threshold

    # ── Fetch all findings for this run ──────────────────────────────────────
    severity_sort = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    cursor = (
        request.app.state.db["findings"]
        .find({"run_id": run_id})
        .sort("created_at", -1)
    )
    raw_findings = await cursor.to_list(length=None)

    # Sort in memory: critical → high → medium → low, then newest-first within tier
    raw_findings.sort(key=lambda f: severity_sort.get(f.get("severity", "low"), 3))

    # ── Aggregate by severity ─────────────────────────────────────────────────
    findings_by_severity: dict[str, int] = {s: 0 for s in _SEVERITY_ORDER}
    for f in raw_findings:
        sev = f.get("severity", "low")
        findings_by_severity[sev] = findings_by_severity.get(sev, 0) + 1

    # ── Aggregate by finding type ─────────────────────────────────────────────
    type_map: dict[str, dict[str, int]] = {}
    for f in raw_findings:
        ft = f.get("finding_type", "unknown")
        sev = f.get("severity", "low")
        if ft not in type_map:
            type_map[ft] = {s: 0 for s in _SEVERITY_ORDER}
        type_map[ft][sev] = type_map[ft].get(sev, 0) + 1

    findings_by_type = [
        FindingTypeSummary(
            finding_type=ft,
            count=sum(counts.values()),
            by_severity=counts,
        )
        for ft, counts in sorted(type_map.items())
    ]

    return RunReportResponse(
        run_id=run_doc["run_id"],
        status=run_doc["status"],
        score=score,
        passed=passed,
        fail_threshold=fail_threshold,
        test_suite=run_doc.get("test_suite", "standard"),
        created_at=run_doc["created_at"],
        completed_at=run_doc.get("completed_at"),
        summary=run_doc.get("summary"),
        findings_count=len(raw_findings),
        findings_by_severity=findings_by_severity,
        findings_by_type=findings_by_type,
        findings=[_finding_to_dict(f) for f in raw_findings],
    )
