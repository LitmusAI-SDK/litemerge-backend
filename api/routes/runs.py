from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, status

from api.schemas.runs import RunCreateRequest, RunCreateResponse, RunStatusResponse
from core.config import settings
from core.security import generate_run_id

router = APIRouter(prefix="/runs", tags=["runs"])

ESTIMATED_DURATION_BY_SUITE = {
    "standard": 180,
    "adversarial": 240,
    "full": 360,
}


@router.post("", response_model=RunCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_run(payload: RunCreateRequest, request: Request) -> RunCreateResponse:
    api_key_record = getattr(request.state, "api_key_record", {})
    allowed_projects = api_key_record.get("project_ids", [])
    if allowed_projects and payload.project_id not in allowed_projects:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key is not authorized for this project",
        )

    run_id = generate_run_id()
    now = datetime.now(timezone.utc)
    estimated_duration_s = ESTIMATED_DURATION_BY_SUITE[payload.test_suite]

    run_doc = {
        "run_id": run_id,
        "project_id": payload.project_id,
        "test_suite": payload.test_suite,
        "fail_threshold": payload.fail_threshold,
        "notify_webhook": str(payload.notify_webhook) if payload.notify_webhook else None,
        "status": "queued",
        "estimated_duration_s": estimated_duration_s,
        "score": None,
        "summary": None,
        "created_at": now,
        "updated_at": now,
    }

    await request.app.state.db["runs"].insert_one(run_doc)

    return RunCreateResponse(
        run_id=run_id,
        status="queued",
        status_url=(
            f"{settings.public_api_base_url.rstrip('/')}/v1/runs/{run_id}"
        ),
        estimated_duration_s=estimated_duration_s,
    )


@router.get("/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str, request: Request) -> RunStatusResponse:
    run_doc = await request.app.state.db["runs"].find_one({"run_id": run_id})
    if not run_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    score = run_doc.get("score")
    fail_threshold = run_doc.get("fail_threshold")
    passed = None
    if score is not None and fail_threshold is not None:
        passed = score >= fail_threshold

    report_url = None
    if run_doc.get("status") == "complete":
        report_url = f"{settings.public_api_base_url.rstrip('/')}/v1/reports/{run_id}"

    return RunStatusResponse(
        run_id=run_doc["run_id"],
        status=run_doc["status"],
        score=score,
        passed=passed,
        report_url=report_url,
        summary=run_doc.get("summary"),
    )
