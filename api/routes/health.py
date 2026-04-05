from fastapi import APIRouter, Request

from api.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    mongodb_status = "ok" if getattr(request.app.state, "db_ready", False) else "degraded"

    return HealthResponse(
        status="ok",
        services={
            "mongodb": mongodb_status,
            "redis": "configured",
        },
    )
