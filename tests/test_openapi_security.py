from fastapi.testclient import TestClient

from api.main import app


def test_openapi_declares_bearer_auth_scheme() -> None:
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    security_schemes = schema["components"]["securitySchemes"]

    assert "BearerAuth" in security_schemes
    bearer = security_schemes["BearerAuth"]
    assert bearer["type"] == "http"
    assert bearer["scheme"] == "bearer"


def test_openapi_marks_protected_projects_route_with_bearer_auth() -> None:
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    projects_get = schema["paths"]["/v1/projects"]["get"]
    assert {"BearerAuth": []} in projects_get["security"]


def test_openapi_does_not_mark_health_route_as_protected() -> None:
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    health_get = schema["paths"]["/v1/health"]["get"]
    assert "security" not in health_get
