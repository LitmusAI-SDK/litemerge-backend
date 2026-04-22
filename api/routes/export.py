"""Export routes — cross-run conversation exports for persona analysis."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request, status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["export"])


@router.get("/conversations")
async def export_persona_conversations(
    request: Request,
    persona_id: str = Query(description="Persona ID to export (e.g. 'p1')"),
) -> dict:
    """Export every conversation turn for a given persona across all runs and projects.

    Returns a single JSON document containing:
    - export_metadata: who/what/when was exported
    - sessions: one entry per run where this persona ran, each with the project's
      company_context so reviewers know what domain was being tested

    Useful for longitudinal analysis — e.g. "how does Maria behave across all
    client deployments?" without being scoped to a single run.

    API key project scoping is respected: keys with project restrictions will
    only see sessions from their permitted projects.
    """
    api_key_record = getattr(request.state, "api_key_record", {})
    allowed_projects = api_key_record.get("project_ids", [])

    # ── Fetch all chat_logs for this persona ─────────────────────────────────
    chat_query: dict = {"persona_id": persona_id}
    if allowed_projects:
        chat_query["project_id"] = {"$in": allowed_projects}

    cursor = (
        request.app.state.db["chat_logs"]
        .find(
            chat_query,
            {
                "_id": 0,
                "run_id": 1,
                "project_id": 1,
                "persona_id": 1,
                "persona_name": 1,
                "persona_type": 1,
                "status": 1,
                "turns": 1,
            },
        )
        .sort("_id", 1)
    )

    raw_sessions = await cursor.to_list(length=None)

    if not raw_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No conversation data found for persona '{persona_id}'",
        )

    # ── Collect unique project_ids so we can batch-fetch company profiles ─────
    project_ids = list(
        {s.get("project_id") for s in raw_sessions if s.get("project_id")}
    )
    project_cursor = request.app.state.db["projects"].find(
        {"_id": {"$in": project_ids}},
        {"_id": 1, "name": 1, "agent_endpoint": 1, "company_context": 1},
    )
    project_map: dict[str, dict] = {}
    async for pdoc in project_cursor:
        project_map[str(pdoc["_id"])] = {
            "project_id": str(pdoc["_id"]),
            "project_name": pdoc.get("name"),
            "agent_endpoint": pdoc.get("agent_endpoint"),
            "company_context": pdoc.get("company_context") or "",
        }

    # ── Collect unique run_ids so we can batch-fetch run metadata ─────────────
    run_ids = list({s.get("run_id") for s in raw_sessions if s.get("run_id")})
    run_cursor = request.app.state.db["runs"].find(
        {"run_id": {"$in": run_ids}},
        {"run_id": 1, "test_suite": 1, "status": 1, "score": 1, "created_at": 1},
    )
    run_map: dict[str, dict] = {}
    async for rdoc in run_cursor:
        created_at = rdoc.get("created_at")
        run_map[rdoc["run_id"]] = {
            "run_id": rdoc["run_id"],
            "test_suite": rdoc.get("test_suite"),
            "status": rdoc.get("status"),
            "score": rdoc.get("score"),
            "run_created_at": created_at.isoformat()
            if hasattr(created_at, "isoformat")
            else str(created_at),
        }

    # ── Build session list ────────────────────────────────────────────────────
    persona_name: str | None = None
    persona_type: str | None = None
    total_turns = 0

    sessions: list[dict] = []
    for doc in raw_sessions:
        if not persona_name and doc.get("persona_name"):
            persona_name = doc["persona_name"]
        if not persona_type and doc.get("persona_type"):
            persona_type = doc["persona_type"]

        turns = []
        for i, turn in enumerate(doc.get("turns", [])):
            turns.append(
                {
                    "turn_index": i,
                    "persona_message": turn.get("persona_message")
                    or turn.get("persona_turn")
                    or turn.get("user")
                    or "",
                    "agent_response": turn.get("agent_response")
                    or turn.get("assistant")
                    or "",
                }
            )
        total_turns += len(turns)

        raw_status = doc.get("status", "in_progress")
        if raw_status not in ("in_progress", "completed", "failed"):
            raw_status = "in_progress"

        run_id = doc.get("run_id", "")
        project_id = str(doc.get("project_id", ""))

        sessions.append(
            {
                **run_map.get(run_id, {"run_id": run_id}),
                "project": project_map.get(project_id, {"project_id": project_id}),
                "persona_id": doc.get("persona_id", ""),
                "persona_name": doc.get("persona_name"),
                "persona_type": doc.get("persona_type"),
                "status": raw_status,
                "turns_completed": len(turns),
                "turns": turns,
            }
        )

    return {
        "export_metadata": {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "persona_id": persona_id,
            "persona_name": persona_name,
            "persona_type": persona_type,
            "total_sessions": len(sessions),
            "total_turns": total_turns,
        },
        "sessions": sessions,
    }
