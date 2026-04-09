from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from core.security import hash_api_key


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, exempt_paths: list[str]) -> None:
        super().__init__(app)
        self.exempt_paths = {path.rstrip("/") or "/" for path in exempt_paths}

    def _is_exempt(self, path: str) -> bool:
        normalized_path = path.rstrip("/") or "/"
        if normalized_path in self.exempt_paths:
            return True
        if normalized_path.startswith("/docs") or normalized_path.startswith("/redoc"):
            return True
        return False

    def _extract_bearer_token(self, request: Request) -> str | None:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None
        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        return parts[1]

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or self._is_exempt(request.url.path):
            return await call_next(request)

        if not getattr(request.app.state, "db_ready", False):
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Database is unavailable. Check startup logs and dependencies."
                },
            )

        token = self._extract_bearer_token(request)
        if not token:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Missing or invalid Authorization header. Expected: Bearer <token>"
                },
            )

        key_hash = hash_api_key(token)
        api_key_record = await request.app.state.db["api_keys"].find_one(
            {"key_hash": key_hash, "is_active": True}
        )
        if not api_key_record:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

        request.state.api_key_record = api_key_record
        return await call_next(request)
