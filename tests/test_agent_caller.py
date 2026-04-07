"""Unit tests for caller.AgentCaller.

httpx.AsyncClient is mocked so no real network calls are made.
"""

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from caller.agent_caller import AgentCaller
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


def _mock_http_response(
    status_code: int = 200,
    body: dict | str | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Return (mock_client_cm, mock_client) configured to produce the given response."""
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
# Successful call tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_send_no_auth_success() -> None:
    project = _make_project(auth_type="none")
    caller = AgentCaller(project)

    mock_cm, mock_client = _mock_http_response()
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        result = await caller.send(message="Hi", session_id="sess_1", history=[])

    assert result.ok
    assert result.reply == "Hello from agent!"
    assert result.status_code == 200
    assert result.error is None
    assert result.latency_ms >= 0

    _, kwargs = mock_client.post.call_args
    assert "Authorization" not in mock_client.post.call_args[1].get("headers", {})


@pytest.mark.anyio
async def test_send_bearer_auth() -> None:
    project = _make_project(auth_type="bearer", auth_value="my-token-123")
    caller = AgentCaller(project)

    mock_cm, mock_client = _mock_http_response()
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        result = await caller.send(message="Hi", session_id="sess_1", history=[])

    assert result.ok
    sent_headers = mock_client.post.call_args[1]["headers"]
    assert sent_headers["Authorization"] == "Bearer my-token-123"


@pytest.mark.anyio
async def test_send_apikey_auth() -> None:
    project = _make_project(auth_type="apikey", auth_value="key-xyz", header_name="X-Custom-Key")
    caller = AgentCaller(project)

    mock_cm, mock_client = _mock_http_response()
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        result = await caller.send(message="Hi", session_id="sess_1", history=[])

    assert result.ok
    sent_headers = mock_client.post.call_args[1]["headers"]
    assert sent_headers["X-Custom-Key"] == "key-xyz"
    assert "Authorization" not in sent_headers


@pytest.mark.anyio
async def test_send_basic_auth() -> None:
    project = _make_project(auth_type="basic", auth_value="user:pass")
    caller = AgentCaller(project)

    mock_cm, mock_client = _mock_http_response()
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        result = await caller.send(message="Hi", session_id="sess_1", history=[])

    assert result.ok
    sent_headers = mock_client.post.call_args[1]["headers"]
    expected = "Basic " + base64.b64encode(b"user:pass").decode()
    assert sent_headers["Authorization"] == expected


@pytest.mark.anyio
async def test_send_session_id_header() -> None:
    project = _make_project()
    caller = AgentCaller(project)

    mock_cm, mock_client = _mock_http_response()
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        await caller.send(message="Hi", session_id="sess_xyz", history=[])

    sent_headers = mock_client.post.call_args[1]["headers"]
    assert sent_headers["X-LitmusAI-Session"] == "sess_xyz"


@pytest.mark.anyio
async def test_send_default_body_fields() -> None:
    project = _make_project()
    caller = AgentCaller(project)
    history = [{"role": "user", "content": "hey"}, {"role": "assistant", "content": "yo"}]

    mock_cm, mock_client = _mock_http_response()
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        await caller.send(message="next turn", session_id="sess_1", history=history)

    sent_body = mock_client.post.call_args[1]["json"]
    assert sent_body["message"] == "next turn"
    assert sent_body["session_id"] == "sess_1"
    assert sent_body["conversation_history"] == history


@pytest.mark.anyio
async def test_send_schema_hints_remap_fields() -> None:
    project = _make_project(
        schema_hints={
            "message": "input",
            "session_id": "conv_id",
            "conversation_history": "history",
            "reply": "response",
        }
    )
    caller = AgentCaller(project)

    mock_cm, mock_client = _mock_http_response(body={"response": "Remapped reply"})
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        result = await caller.send(message="Hi", session_id="sess_1", history=[])

    assert result.ok
    assert result.reply == "Remapped reply"

    sent_body = mock_client.post.call_args[1]["json"]
    assert "input" in sent_body
    assert "conv_id" in sent_body
    assert "history" in sent_body
    assert "message" not in sent_body


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_send_timeout() -> None:
    import httpx as _httpx

    project = _make_project()
    caller = AgentCaller(project, timeout_s=5.0)

    mock_cm = AsyncMock()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=_httpx.TimeoutException("timed out"))
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        result = await caller.send(message="Hi", session_id="sess_1", history=[])

    assert not result.ok
    assert result.reply is None
    assert result.status_code == 0
    assert "timed out" in result.error.lower()
    assert result.latency_ms == pytest.approx(5000.0)


@pytest.mark.anyio
async def test_send_request_error() -> None:
    import httpx as _httpx

    project = _make_project()
    caller = AgentCaller(project)

    mock_cm = AsyncMock()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        side_effect=_httpx.ConnectError("Connection refused")
    )
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        result = await caller.send(message="Hi", session_id="sess_1", history=[])

    assert not result.ok
    assert result.status_code == 0
    assert result.error is not None


@pytest.mark.anyio
async def test_send_non_200_response() -> None:
    project = _make_project()
    caller = AgentCaller(project)

    mock_cm, _ = _mock_http_response(status_code=503, body={"detail": "unavailable"})
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        result = await caller.send(message="Hi", session_id="sess_1", history=[])

    assert not result.ok
    assert result.reply is None
    assert result.status_code == 503
    assert "503" in result.error


@pytest.mark.anyio
async def test_send_invalid_json_response() -> None:
    project = _make_project()
    caller = AgentCaller(project)

    mock_cm, _ = _mock_http_response(status_code=200, body="not json at all")
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        result = await caller.send(message="Hi", session_id="sess_1", history=[])

    assert not result.ok
    assert result.reply is None
    assert result.status_code == 200
    assert "json" in result.error.lower()


@pytest.mark.anyio
async def test_send_missing_reply_field() -> None:
    project = _make_project()
    caller = AgentCaller(project)

    mock_cm, _ = _mock_http_response(body={"answer": "oops, wrong field"})
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        result = await caller.send(message="Hi", session_id="sess_1", history=[])

    assert not result.ok
    assert result.reply is None
    assert result.status_code == 200
    assert "reply" in result.error


@pytest.mark.anyio
async def test_send_reply_not_string() -> None:
    project = _make_project()
    caller = AgentCaller(project)

    mock_cm, _ = _mock_http_response(body={"reply": 42})
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        result = await caller.send(message="Hi", session_id="sess_1", history=[])

    assert not result.ok
    assert result.reply is None
    assert "reply" in result.error


@pytest.mark.anyio
async def test_raw_body_captured_on_error() -> None:
    project = _make_project()
    caller = AgentCaller(project)

    mock_cm, _ = _mock_http_response(status_code=400, body={"detail": "bad request"})
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=mock_cm):
        result = await caller.send(message="Hi", session_id="sess_1", history=[])

    assert result.raw_body != ""
    assert "bad request" in result.raw_body
