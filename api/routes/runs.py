"""Runs routes — create, status poll, and SSE live stream."""

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from api.schemas.runs import (
    RunCreateRequest,
    RunCreateResponse,
    RunListItem,
    RunStatusResponse,
    SessionStatus,
)
from core.config import settings
from core.security import generate_run_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])

ESTIMATED_DURATION_BY_SUITE = {
    "standard": 180,
    "adversarial": 240,
    "full": 360,
}

# SSE keepalive interval in seconds — prevents proxy/browser timeout on idle runs
_SSE_KEEPALIVE_S = 15


@router.get("", response_model=list[RunListItem])
async def list_runs(
    request: Request, project_id: str | None = None
) -> list[RunListItem]:
    """List runs, optionally filtered by project_id.

    Returns runs in descending creation order (newest first).
    Scoped to projects the API key is authorized for when the key has
    explicit project restrictions.
    """
    api_key_record = getattr(request.state, "api_key_record", {})
    allowed_projects = api_key_record.get("project_ids", [])

    query: dict = {}
    if project_id:
        if allowed_projects and project_id not in allowed_projects:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key is not authorized for this project",
            )
        query["project_id"] = project_id
    elif allowed_projects:
        query["project_id"] = {"$in": allowed_projects}

    cursor = request.app.state.db["runs"].find(query).sort("created_at", -1).limit(200)
    docs = await cursor.to_list(length=200)

    items: list[RunListItem] = []
    for doc in docs:
        score = doc.get("score")
        fail_threshold = doc.get("fail_threshold", 70)
        passed: bool | None = None
        if score is not None:
            passed = score >= fail_threshold

        created_at = doc.get("created_at")
        completed_at = doc.get("completed_at")

        items.append(
            RunListItem(
                run_id=doc["run_id"],
                project_id=doc.get("project_id", ""),
                test_suite=doc.get("test_suite", "standard"),
                status=doc["status"],
                score=score,
                passed=passed,
                fail_threshold=fail_threshold,
                created_at=created_at.isoformat()
                if hasattr(created_at, "isoformat")
                else str(created_at),
                completed_at=completed_at.isoformat()
                if completed_at and hasattr(completed_at, "isoformat")
                else (str(completed_at) if completed_at else None),
            )
        )
    return items


@router.post("", response_model=RunCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_run(payload: RunCreateRequest, request: Request) -> RunCreateResponse:
    api_key_record = getattr(request.state, "api_key_record", {})
    allowed_projects = api_key_record.get("project_ids", [])
    if allowed_projects and payload.project_id not in allowed_projects:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key is not authorized for this project",
        )

    project_doc = await request.app.state.db["projects"].find_one(
        {"_id": payload.project_id}, {"_id": 1}
    )
    if not project_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{payload.project_id}' not found",
        )

    run_id = generate_run_id()
    now = datetime.now(timezone.utc)
    estimated_duration_s = ESTIMATED_DURATION_BY_SUITE[payload.test_suite]

    run_doc = {
        "run_id": run_id,
        "project_id": payload.project_id,
        "test_suite": payload.test_suite,
        "fail_threshold": payload.fail_threshold,
        "notify_webhook": str(payload.notify_webhook)
        if payload.notify_webhook
        else None,
        "status": "queued",
        "estimated_duration_s": estimated_duration_s,
        "score": None,
        "summary": None,
        "created_at": now,
        "updated_at": now,
    }

    await request.app.state.db["runs"].insert_one(run_doc)

    # Dispatch to Celery worker — non-blocking, worker picks up from Redis queue
    try:
        from worker.tasks import process_run

        process_run.delay(run_id)
        logger.info("Dispatched run %s to Celery", run_id)
    except Exception:
        logger.exception("Failed to dispatch run %s to Celery", run_id)
        # Don't fail the API response — run is queued in DB, worker will pick it up
        # when Celery recovers, or operator can re-trigger manually.

    return RunCreateResponse(
        run_id=run_id,
        status="queued",
        status_url=(f"{settings.public_api_base_url.rstrip('/')}/v1/runs/{run_id}"),
        estimated_duration_s=estimated_duration_s,
    )


@router.get("/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str, request: Request) -> RunStatusResponse:
    run_doc = await request.app.state.db["runs"].find_one({"run_id": run_id})
    if not run_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Run not found"
        )

    score = run_doc.get("score")
    fail_threshold = run_doc.get("fail_threshold")
    passed = None
    if score is not None and fail_threshold is not None:
        passed = score >= fail_threshold

    report_url = None
    if run_doc.get("status") == "complete":
        report_url = f"{settings.public_api_base_url.rstrip('/')}/v1/reports/{run_id}"

    # Fetch per-session statuses from chat_logs for partial-result visibility
    session_statuses = await _fetch_session_statuses(request.app.state.db, run_id)

    return RunStatusResponse(
        run_id=run_doc["run_id"],
        status=run_doc["status"],
        score=score,
        passed=passed,
        report_url=report_url,
        summary=run_doc.get("summary"),
        session_statuses=session_statuses,
    )


@router.get("/{run_id}/sessions")
async def get_run_sessions(run_id: str, request: Request) -> list[dict]:
    """Return full conversation logs for every persona session in a run.

    Each element represents one persona session and includes:
    - persona_id, persona_name, persona_type, status, turns_completed
    - turns: list of {role, content, turn_index} message pairs
    """
    run_doc = await request.app.state.db["runs"].find_one(
        {"run_id": run_id}, {"_id": 0, "run_id": 1, "project_id": 1}
    )
    if not run_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Run not found"
        )

    api_key_record = getattr(request.state, "api_key_record", {})
    allowed_projects = api_key_record.get("project_ids", [])
    if allowed_projects and run_doc.get("project_id") not in allowed_projects:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden"
        )

    cursor = request.app.state.db["chat_logs"].find(
        {"run_id": run_id},
        {
            "_id": 0,
            "persona_id": 1,
            "persona_name": 1,
            "persona_type": 1,
            "status": 1,
            "turns": 1,
        },
    )
    sessions: list[dict] = []
    async for doc in cursor:
        raw_status = doc.get("status", "in_progress")
        if raw_status not in ("in_progress", "completed", "failed"):
            raw_status = "in_progress"

        turns = []
        for i, turn in enumerate(doc.get("turns", [])):
            turns.append(
                {
                    "turn_index": i,
                    "persona_message": turn.get("persona_message")
                    or turn.get("user")
                    or "",
                    "agent_response": turn.get("agent_response")
                    or turn.get("assistant")
                    or "",
                }
            )

        sessions.append(
            {
                "persona_id": doc.get("persona_id", ""),
                "persona_name": doc.get("persona_name"),
                "persona_type": doc.get("persona_type"),
                "status": raw_status,
                "turns_completed": len(turns),
                "turns": turns,
            }
        )

    return sessions


@router.get("/{run_id}/stream")
async def stream_run_events(run_id: str, request: Request) -> StreamingResponse:
    """SSE endpoint — streams live simulation progress events to the dashboard.

    Events are JSON objects published by SimulationRun to Redis channel
    "run:{run_id}". The client receives them as SSE data frames.

    The stream terminates automatically when a terminal event is received
    (run_complete) or when the client disconnects.
    """
    run_doc = await request.app.state.db["runs"].find_one(
        {"run_id": run_id}, {"status": 1}
    )
    if not run_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Run not found"
        )

    # If already terminal, return the current state immediately and close
    if run_doc["status"] in ("complete", "failed"):

        async def _immediate():
            yield _sse_event(
                {
                    "event": "run_complete",
                    "run_id": run_id,
                    "status": run_doc["status"],
                }
            )

        return StreamingResponse(_immediate(), media_type="text/event-stream")

    redis = request.app.state.redis

    async def _event_generator():
        pubsub = redis.pubsub()
        channel = f"run:{run_id}"
        await pubsub.subscribe(channel)
        try:
            while True:
                if await request.is_disconnected():
                    break

                # Non-blocking read with a short timeout so we can send keepalives
                msg = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=_SSE_KEEPALIVE_S,
                )

                if msg is None:
                    # No new message — send keepalive comment to prevent timeout
                    yield ": keepalive\n\n"
                    continue

                if msg["type"] == "message":
                    data_str = msg["data"]
                    yield _sse_event(json.loads(data_str))

                    # Close stream on terminal event
                    try:
                        payload = json.loads(data_str)
                        if payload.get("event") == "run_complete":
                            break
                    except (json.JSONDecodeError, AttributeError):
                        pass

        except asyncio.TimeoutError:
            yield ": keepalive\n\n"
        except Exception:
            logger.exception("SSE stream error for run %s", run_id)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse_event(payload: dict) -> str:
    """Format a dict as an SSE data frame."""
    return f"data: {json.dumps(payload)}\n\n"


async def _fetch_session_statuses(db, run_id: str) -> list[SessionStatus]:
    """Return per-session progress from chat_logs for the given run."""
    cursor = db["chat_logs"].find(
        {"run_id": run_id},
        {
            "persona_id": 1,
            "persona_name": 1,
            "persona_type": 1,
            "status": 1,
            "turns": 1,
        },
    )
    statuses: list[SessionStatus] = []
    async for doc in cursor:
        raw_status = doc.get("status", "in_progress")
        # Normalise any legacy values to our schema literals
        if raw_status not in ("in_progress", "completed", "failed"):
            raw_status = "in_progress"
        statuses.append(
            SessionStatus(
                persona_id=doc.get("persona_id", ""),
                persona_name=doc.get("persona_name"),
                persona_type=doc.get("persona_type"),
                status=raw_status,
                turns_completed=len(doc.get("turns", [])),
            )
        )
    return statuses
