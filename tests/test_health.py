from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint_returns_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "ok"
    assert "mongodb" in payload["services"]
    assert "redis" in payload["services"]
