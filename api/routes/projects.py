from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, status
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from api.schemas.projects import (
    AMBER_LATENCY_MS,
    AuthConfigInput,
    AuthConfigPublic,
    PreflightResponse,
    ProjectCreateRequest,
    ProjectListResponse,
    ProjectPatchRequest,
    ProjectResponse,
)
from caller.agent_caller import create_agent_caller
from core.crypto import encrypt_secret
from core.security import generate_project_id

_PREFLIGHT_PROBE = "Hello, can you help me?"

router = APIRouter(prefix="/projects", tags=["projects"])


def _auth_storage(config: AuthConfigInput) -> dict:
    return {
        "type": config.type,
        "value_encrypted": encrypt_secret(config.value) if config.value else None,
        "header_name": config.header_name,
    }


def _to_project_response(project_doc: dict) -> ProjectResponse:
    auth_config = project_doc.get("auth_config", {})

    return ProjectResponse(
        id=project_doc["_id"],
        name=project_doc["name"],
        agent_endpoint=project_doc["agent_endpoint"],
        auth_config=AuthConfigPublic(
            type=auth_config.get("type", "none"),
            header_name=auth_config.get("header_name"),
            has_value=bool(auth_config.get("value_encrypted")),
        ),
        owner_id=project_doc["owner_id"],
        schema_hints=project_doc.get("schema_hints"),
        created_at=project_doc["created_at"],
        updated_at=project_doc["updated_at"],
    )


def _ensure_project_access(project_id: str, api_key_record: dict) -> None:
    allowed_projects = api_key_record.get("project_ids", [])
    if allowed_projects and project_id not in allowed_projects:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key is not authorized for this project",
        )


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    payload: ProjectCreateRequest, request: Request
) -> ProjectResponse:
    now = datetime.now(timezone.utc)
    project_id = payload.id or generate_project_id()
    project_doc = {
        "_id": project_id,
        "name": payload.name,
        "agent_endpoint": str(payload.agent_endpoint),
        "auth_config": _auth_storage(payload.auth_config),
        "owner_id": payload.owner_id,
        "schema_hints": payload.schema_hints,
        "created_at": now,
        "updated_at": now,
    }

    try:
        await request.app.state.db["projects"].insert_one(project_doc)
    except DuplicateKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Project with id '{project_id}' already exists",
        ) from exc

    api_key_record = getattr(request.state, "api_key_record", {})
    allowed_projects = api_key_record.get("project_ids", [])

    if allowed_projects:
        await request.app.state.db["api_keys"].update_one(
            {"_id": api_key_record["_id"]},
            {
                "$addToSet": {"project_ids": project_id},
                "$set": {"updated_at": now},
            },
        )

    return _to_project_response(project_doc)


@router.get("", response_model=ProjectListResponse)
async def list_projects(request: Request) -> ProjectListResponse:
    api_key_record = getattr(request.state, "api_key_record", {})
    allowed_projects = api_key_record.get("project_ids", [])

    query: dict = {}
    if allowed_projects:
        query["_id"] = {"$in": allowed_projects}

    cursor = request.app.state.db["projects"].find(query).sort("created_at", -1)
    items = [_to_project_response(doc) async for doc in cursor]
    return ProjectListResponse(items=items)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def patch_project(
    project_id: str, payload: ProjectPatchRequest, request: Request
) -> ProjectResponse:
    api_key_record = getattr(request.state, "api_key_record", {})
    _ensure_project_access(project_id, api_key_record)

    update_fields: dict = {}

    if payload.name is not None:
        update_fields["name"] = payload.name
    if payload.agent_endpoint is not None:
        update_fields["agent_endpoint"] = str(payload.agent_endpoint)
    if payload.owner_id is not None:
        update_fields["owner_id"] = payload.owner_id
    if payload.schema_hints is not None:
        update_fields["schema_hints"] = payload.schema_hints
    if payload.auth_config is not None:
        update_fields["auth_config"] = _auth_storage(payload.auth_config)

    update_fields["updated_at"] = datetime.now(timezone.utc)

    updated = await request.app.state.db["projects"].find_one_and_update(
        {"_id": project_id},
        {"$set": update_fields},
        return_document=ReturnDocument.AFTER,
    )

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )

    return _to_project_response(updated)


@router.post("/{project_id}/preflight", response_model=PreflightResponse)
async def preflight_project(project_id: str, request: Request) -> PreflightResponse:
    """Send one neutral probe to the customer's agent and report connectivity health.

    Returns:
        green  — 200 reply received, latency under 2 s.
        amber  — 200 reply received but latency >= 2 s (slow but reachable).
        red    — timeout, non-200, or unparseable/missing reply field.
    """
    api_key_record = getattr(request.state, "api_key_record", {})
    _ensure_project_access(project_id, api_key_record)

    project_doc = await request.app.state.db["projects"].find_one({"_id": project_id})
    if not project_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )

    caller = create_agent_caller(project_doc)
    result = await caller.send(
        message=_PREFLIGHT_PROBE,
        session_id=f"litmusai-preflight-{project_id}",
        history=[],
    )

    if result.error:
        return PreflightResponse(
            status="red", latency_ms=result.latency_ms, error=result.error
        )

    if result.latency_ms >= AMBER_LATENCY_MS:
        return PreflightResponse(status="amber", latency_ms=result.latency_ms)

    return PreflightResponse(status="green", latency_ms=result.latency_ms)
