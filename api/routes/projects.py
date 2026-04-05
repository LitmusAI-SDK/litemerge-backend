from fastapi import APIRouter

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("")
async def list_projects() -> dict[str, list[dict[str, str]]]:
    return {"items": []}
