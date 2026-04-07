"""Integration tests for POST /v1/projects/{id}/preflight.

The FastAPI app is started with TestClient (lifespan runs but MongoDB is
unavailable in test environments).  After entering the client context we
replace app.state.db with a per-test mock and force db_ready = True so the
auth middleware proceeds normally.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from caller.agent_caller import CallerResult
from core.security import hash_api_key

BOOTSTRAP_KEY = "lmai_dev_key"
TEST_PROJECT_ID = "proj_test123"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _project_doc(project_id: str = TEST_PROJECT_ID) -> dict:
    return {
        "_id": project_id,
        "agent_endpoint": "https://agent.example.com/chat",
        "auth_config": {"type": "none"},
        "schema_hints": None,
    }


def _api_key_record() -> dict:
    return {
        "_id": "key-hash-id",
        "key_hash": hash_api_key(BOOTSTRAP_KEY),
        "is_active": True,
        "project_ids": [],  # empty = unrestricted
        "scopes": ["*"],
    }


def _mock_db(project_doc: dict | None = "default") -> MagicMock:
    """Build a mock Motor database whose collections return controllable values."""
    if project_doc == "default":
        project_doc = _project_doc()

    projects_col = MagicMock()
    projects_col.find_one = AsyncMock(return_value=project_doc)

    api_keys_col = MagicMock()
    api_keys_col.find_one = AsyncMock(return_value=_api_key_record())

    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(
        side_effect=lambda name: {
            "projects": projects_col,
            "api_keys": api_keys_col,
        }.get(name, MagicMock())
    )
    return mock_db


@pytest.fixture
def client_and_db():
    """Yield (TestClient, mock_db) with db_ready forced True."""
    db = _mock_db()
    with TestClient(app) as client:
        app.state.db = db
        app.state.db_ready = True
        yield client, db


def _caller_result(
    reply: str | None = "Hello, how can I help you today?",
    latency_ms: float = 450.0,
    status_code: int = 200,
    raw_body: str = '{"reply":"Hello, how can I help you today?"}',
    error: str | None = None,
) -> CallerResult:
    return CallerResult(
        reply=reply,
        latency_ms=latency_ms,
        status_code=status_code,
        raw_body=raw_body,
        error=error,
    )


# ---------------------------------------------------------------------------
# Green / amber / red status tests
# ---------------------------------------------------------------------------


def test_preflight_green(client_and_db) -> None:
    client, _ = client_and_db
    result = _caller_result(latency_ms=450.0)

    with patch("api.routes.projects.AgentCaller") as MockCaller:
        MockCaller.return_value.send = AsyncMock(return_value=result)
        response = client.post(
            f"/v1/projects/{TEST_PROJECT_ID}/preflight",
            headers={"x-api-key": BOOTSTRAP_KEY},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "green"
    assert data["latency_ms"] == pytest.approx(450.0)
    assert data["error"] is None


def test_preflight_amber_slow_response(client_and_db) -> None:
    """Latency >= 2000 ms with a valid reply → amber."""
    client, _ = client_and_db
    result = _caller_result(latency_ms=2500.0)

    with patch("api.routes.projects.AgentCaller") as MockCaller:
        MockCaller.return_value.send = AsyncMock(return_value=result)
        response = client.post(
            f"/v1/projects/{TEST_PROJECT_ID}/preflight",
            headers={"x-api-key": BOOTSTRAP_KEY},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "amber"
    assert data["latency_ms"] == pytest.approx(2500.0)
    assert data["error"] is None


def test_preflight_amber_boundary_exactly_2000ms(client_and_db) -> None:
    """Exactly 2000 ms → amber (boundary is inclusive)."""
    client, _ = client_and_db
    result = _caller_result(latency_ms=2000.0)

    with patch("api.routes.projects.AgentCaller") as MockCaller:
        MockCaller.return_value.send = AsyncMock(return_value=result)
        response = client.post(
            f"/v1/projects/{TEST_PROJECT_ID}/preflight",
            headers={"x-api-key": BOOTSTRAP_KEY},
        )

    assert response.json()["status"] == "amber"


def test_preflight_red_timeout(client_and_db) -> None:
    client, _ = client_and_db
    result = _caller_result(
        reply=None,
        latency_ms=30000.0,
        status_code=0,
        raw_body="",
        error="Request timed out after 30.0s",
    )

    with patch("api.routes.projects.AgentCaller") as MockCaller:
        MockCaller.return_value.send = AsyncMock(return_value=result)
        response = client.post(
            f"/v1/projects/{TEST_PROJECT_ID}/preflight",
            headers={"x-api-key": BOOTSTRAP_KEY},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "red"
    assert "timed out" in data["error"]


def test_preflight_red_non_200(client_and_db) -> None:
    client, _ = client_and_db
    result = _caller_result(
        reply=None,
        latency_ms=120.0,
        status_code=503,
        raw_body='{"detail":"Service unavailable"}',
        error="Agent returned HTTP 503",
    )

    with patch("api.routes.projects.AgentCaller") as MockCaller:
        MockCaller.return_value.send = AsyncMock(return_value=result)
        response = client.post(
            f"/v1/projects/{TEST_PROJECT_ID}/preflight",
            headers={"x-api-key": BOOTSTRAP_KEY},
        )

    data = response.json()
    assert data["status"] == "red"
    assert "503" in data["error"]


def test_preflight_red_bad_reply_field(client_and_db) -> None:
    client, _ = client_and_db
    result = _caller_result(
        reply=None,
        latency_ms=200.0,
        status_code=200,
        raw_body='{"answer":"wrong field name"}',
        error="Response missing expected reply field 'reply'",
    )

    with patch("api.routes.projects.AgentCaller") as MockCaller:
        MockCaller.return_value.send = AsyncMock(return_value=result)
        response = client.post(
            f"/v1/projects/{TEST_PROJECT_ID}/preflight",
            headers={"x-api-key": BOOTSTRAP_KEY},
        )

    data = response.json()
    assert data["status"] == "red"
    assert data["error"] is not None


# ---------------------------------------------------------------------------
# AgentCaller construction — verify project doc is passed through
# ---------------------------------------------------------------------------


def test_preflight_passes_project_doc_to_agent_caller(client_and_db) -> None:
    """AgentCaller must be instantiated with the full project doc from DB."""
    client, _ = client_and_db
    result = _caller_result()

    with patch("api.routes.projects.AgentCaller") as MockCaller:
        MockCaller.return_value.send = AsyncMock(return_value=result)
        client.post(
            f"/v1/projects/{TEST_PROJECT_ID}/preflight",
            headers={"x-api-key": BOOTSTRAP_KEY},
        )

    project_doc_arg = MockCaller.call_args[0][0]
    assert project_doc_arg["_id"] == TEST_PROJECT_ID
    assert project_doc_arg["agent_endpoint"] == "https://agent.example.com/chat"


def test_preflight_probe_uses_preflight_session_id(client_and_db) -> None:
    """session_id sent to the agent should be clearly identifiable as a preflight probe."""
    client, _ = client_and_db
    result = _caller_result()

    with patch("api.routes.projects.AgentCaller") as MockCaller:
        MockCaller.return_value.send = AsyncMock(return_value=result)
        client.post(
            f"/v1/projects/{TEST_PROJECT_ID}/preflight",
            headers={"x-api-key": BOOTSTRAP_KEY},
        )

    _, send_kwargs = MockCaller.return_value.send.call_args
    assert "preflight" in send_kwargs["session_id"]
    assert send_kwargs["history"] == []


# ---------------------------------------------------------------------------
# Auth / access control
# ---------------------------------------------------------------------------


def test_preflight_404_when_project_not_found() -> None:
    db = _mock_db(project_doc=None)  # DB returns None for find_one
    with TestClient(app) as client:
        app.state.db = db
        app.state.db_ready = True
        response = client.post(
            "/v1/projects/nonexistent/preflight",
            headers={"x-api-key": BOOTSTRAP_KEY},
        )
    assert response.status_code == 404


def test_preflight_403_when_api_key_not_scoped_to_project() -> None:
    """API key with a project_ids restriction that excludes the target project → 403."""
    db = _mock_db()
    # Override api_keys to return a key scoped to a different project
    restricted_key = {**_api_key_record(), "project_ids": ["proj_other"]}
    db["api_keys"].find_one = AsyncMock(return_value=restricted_key)

    with TestClient(app) as client:
        app.state.db = db
        app.state.db_ready = True
        response = client.post(
            f"/v1/projects/{TEST_PROJECT_ID}/preflight",
            headers={"x-api-key": BOOTSTRAP_KEY},
        )
    assert response.status_code == 403


def test_preflight_401_when_api_key_missing() -> None:
    db = _mock_db()
    with TestClient(app) as client:
        app.state.db = db
        app.state.db_ready = True
        response = client.post(f"/v1/projects/{TEST_PROJECT_ID}/preflight")
    assert response.status_code == 401
