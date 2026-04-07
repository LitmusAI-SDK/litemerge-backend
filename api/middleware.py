from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from core.security import hash_api_key


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, exempt_paths: list[str], api_key_header: str) -> None:
        super().__init__(app)
        self.exempt_paths = {path.rstrip("/") or "/" for path in exempt_paths}
        self.api_key_header = api_key_header

    def _is_exempt(self, path: str) -> bool:
        normalized_path = path.rstrip("/") or "/"
        if normalized_path in self.exempt_paths:
            return True
        if normalized_path.startswith("/docs") or normalized_path.startswith("/redoc"):
            return True
        return False

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

        api_key = request.headers.get(self.api_key_header)
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": f"Missing {self.api_key_header} header"},
            )

        key_hash = hash_api_key(api_key)
        api_key_record = await request.app.state.db["api_keys"].find_one(
            {"key_hash": key_hash, "is_active": True}
        )
        if not api_key_record:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

        request.state.api_key_record = api_key_record
        return await call_next(request)
