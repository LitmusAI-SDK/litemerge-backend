"""Unit tests for simulation.agent_caller.SimulationAgentCaller.

Tests the wrapper layer: AgentResponse shape, delegation to base caller,
and mock failure injection.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from caller.agent_caller import AgentResponse, AgentRetriableError, SimulationAgentCaller
from core.crypto import encrypt_secret


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project(
    endpoint: str = "https://agent.example.com/chat",
    auth_type: str = "none",
    auth_value: str | None = None,
    header_name: str | None = None,
    schema_hints: dict | None = None,
) -> dict:
    auth_config: dict = {"type": auth_type}
    if auth_value:
        auth_config["value_encrypted"] = encrypt_secret(auth_value)
    if header_name:
        auth_config["header_name"] = header_name
    return {
        "agent_endpoint": endpoint,
        "auth_config": auth_config,
        "schema_hints": schema_hints,
    }


def _mock_http_response(status_code: int = 200, body: dict | str | None = None):
    if body is None:
        body = {"reply": "Hello from agent!"}
    raw = json.dumps(body) if isinstance(body, dict) else body

    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.text = raw
    if isinstance(body, dict):
        mock_response.json.return_value = body
    else:
        mock_response.json.side_effect = ValueError("not JSON")

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    return mock_cm, mock_client


# ---------------------------------------------------------------------------
# Delegation tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_bearer_auth_injects_authorization_header() -> None:
    project = _make_project(auth_type="bearer", auth_value="my-token-123")
    caller = SimulationAgentCaller(project, mock=False)

    mock_cm, mock_client = _mock_http_response()
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        resp = await caller.send("Hi", "sess_1", [])

    assert resp.reply == "Hello from agent!"
    sent_headers = mock_client.post.call_args[1]["headers"]
    assert sent_headers["Authorization"] == "Bearer my-token-123"


@pytest.mark.anyio
async def test_schema_hint_remaps_message_key() -> None:
    project = _make_project(schema_hints={"message": "user_query", "reply": "answer"})
    caller = SimulationAgentCaller(project, mock=False)

    mock_cm, mock_client = _mock_http_response(body={"answer": "remapped reply"})
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        resp = await caller.send("Hi", "sess_1", [])

    assert resp.reply == "remapped reply"
    sent_body = mock_client.post.call_args[1]["json"]
    assert "user_query" in sent_body
    assert "message" not in sent_body


@pytest.mark.anyio
async def test_session_header_always_present() -> None:
    project = _make_project()
    caller = SimulationAgentCaller(project, mock=False)

    mock_cm, mock_client = _mock_http_response()
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        await caller.send("Hi", "sess_xyz", [])

    sent_headers = mock_client.post.call_args[1]["headers"]
    assert sent_headers["X-LitmusAI-Session"] == "sess_xyz"


@pytest.mark.anyio
async def test_non_200_returns_none_reply() -> None:
    project = _make_project()
    caller = SimulationAgentCaller(project, mock=False)

    mock_cm, _ = _mock_http_response(status_code=503, body={"detail": "unavailable"})
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        resp = await caller.send("Hi", "sess_1", [])

    assert isinstance(resp, AgentResponse)
    assert resp.reply is None
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Mock mode tests
# ---------------------------------------------------------------------------

def test_mock_mode_produces_503_and_429() -> None:
    """Over 200 mock calls, at least one 429 and one 503 must appear."""
    project = _make_project()
    caller = SimulationAgentCaller(project, mock=True)

    status_codes: list[int] = []
    for _ in range(200):
        resp = caller._mock_response()
        status_codes.append(resp.status_code)

    assert 429 in status_codes, "Expected at least one 429 in 200 mock calls"
    assert 503 in status_codes, "Expected at least one 503 in 200 mock calls"


def test_mock_mode_200_reply_is_non_empty() -> None:
    project = _make_project()
    caller = SimulationAgentCaller(project, mock=True)

    # Generate mock responses until we get a 200
    for _ in range(50):
        resp = caller._mock_response()
        if resp.status_code == 200:
            assert resp.reply is not None
            assert len(resp.reply) > 0
            return

    pytest.skip("Did not encounter a 200 response in 50 tries (unlikely but possible)")


def test_agent_retriable_error_is_exception() -> None:
    err = AgentRetriableError("status 503")
    assert isinstance(err, Exception)
    assert "503" in str(err)
