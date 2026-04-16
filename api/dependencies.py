from fastapi import Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Declares Bearer auth in OpenAPI so Swagger UI can send Authorization header.
# Runtime token validation remains in ApiKeyAuthMiddleware.
bearer_auth = HTTPBearer(
    auto_error=False,
    scheme_name="BearerAuth",
    description=(
        "Paste the API key token value only. Swagger will send "
        "Authorization: Bearer <token>."
    ),
)


async def require_bearer_for_docs(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_auth),
) -> HTTPAuthorizationCredentials | None:
    return credentials
