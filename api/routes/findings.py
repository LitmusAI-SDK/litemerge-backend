"""Findings routes — seed, list, and manage KB findings."""

import logging

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Query, Request, status

from api.schemas.findings import (
    FindingCreateRequest,
    FindingListResponse,
    FindingResponse,
)
from kb.writer import KBWriter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/findings", tags=["findings"])

_writer = KBWriter()


def _doc_to_response(doc: dict) -> FindingResponse:
    return FindingResponse(
        id=str(doc["_id"]),
        project_id=doc["project_id"],
        run_id=doc.get("run_id"),
        persona_type=doc["persona_type"],
        finding_type=doc["finding_type"],
        severity=doc["severity"],
        prompt_vector=doc.get("prompt_vector"),
        agent_response_excerpt=doc.get("agent_response_excerpt"),
        created_at=doc["created_at"],
    )


@router.post("", response_model=FindingResponse, status_code=status.HTTP_201_CREATED)
async def create_finding(
    payload: FindingCreateRequest, request: Request
) -> FindingResponse:
    """Manually seed a finding into the KB.

    Useful for testing persona prompt enrichment before the evaluation engine
    (Phase 6) writes findings automatically post-run.
    """
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

    inserted_id = await _writer.write_finding(
        request.app.state.db,
        project_id=payload.project_id,
        run_id=payload.run_id,
        persona_type=payload.persona_type,
        finding_type=payload.finding_type,
        severity=payload.severity,
        prompt_vector=payload.prompt_vector,
        agent_response_excerpt=payload.agent_response_excerpt,
    )

    doc = await request.app.state.db["findings"].find_one(
        {"_id": ObjectId(inserted_id)}
    )
    return _doc_to_response(doc)


@router.get("", response_model=FindingListResponse)
async def list_findings(
    request: Request,
    project_id: str = Query(..., min_length=1, description="Filter by project ID"),
    persona_type: str | None = Query(
        default=None, description="Filter by persona type"
    ),
    severity: str | None = Query(default=None, description="Filter by severity"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> FindingListResponse:
    """List findings for a project. Supports filtering by persona_type and severity."""
    api_key_record = getattr(request.state, "api_key_record", {})
    allowed_projects = api_key_record.get("project_ids", [])
    if allowed_projects and project_id not in allowed_projects:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key is not authorized for this project",
        )

    query: dict = {"project_id": project_id}
    if persona_type:
        query["persona_type"] = persona_type
    if severity:
        query["severity"] = severity

    total = await request.app.state.db["findings"].count_documents(query)
    cursor = (
        request.app.state.db["findings"]
        .find(query)
        .sort("created_at", -1)
        .skip(offset)
        .limit(limit)
    )

    items: list[FindingResponse] = []
    async for doc in cursor:
        items.append(_doc_to_response(doc))

    return FindingListResponse(items=items, total=total)
