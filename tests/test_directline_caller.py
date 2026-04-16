"""Unit tests for DirectLineAgentCaller and the create_agent_caller() factory.

All HTTP calls are mocked so no real network traffic is made.
asyncio.sleep is patched to keep tests fast.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from caller.agent_caller import (
    AgentCaller,
    DirectLineAgentCaller,
    SimulationAgentCaller,
    _DirectLineError,
    create_agent_caller,
)
from core.crypto import encrypt_secret


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_directline_project(
    base_url: str = "https://directline.botframework.com",
    secret: str = "dl-channel-secret",
) -> dict:
    """Build a minimal project document for a DirectLine-enabled project."""
    return {
        "agent_endpoint": base_url,
        "auth_config": {
            "type": "bearer",
            "value_encrypted": encrypt_secret(secret),
        },
        "schema_hints": {"caller_type": "directline"},
    }


def _make_standard_project(
    endpoint: str = "https://agent.example.com/chat",
) -> dict:
    return {
        "agent_endpoint": endpoint,
        "auth_config": {"type": "none"},
        "schema_hints": {},
    }


def _make_http_cm(
    method: str,
    status: int,
    body: dict | None = None,
    json_error: bool = False,
) -> tuple[AsyncMock, AsyncMock]:
    """Return (context_manager, inner_client) for one httpx.AsyncClient call."""
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.text = json.dumps(body) if body is not None else ""
    if json_error:
        mock_resp.json.side_effect = ValueError("not valid json")
    else:
        mock_resp.json.return_value = body if body is not None else {}

    mock_client = AsyncMock()
    if method == "get":
        mock_client.get = AsyncMock(return_value=mock_resp)
    else:
        mock_client.post = AsyncMock(return_value=mock_resp)

    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    return mock_cm, mock_client


def _conv_body(
    conversation_id: str = "conv-abc123",
    token: str = "session-token-xyz",
) -> dict:
    return {"conversationId": conversation_id, "token": token}


def _activities_body(
    bot_text: str | None = None,
    watermark: str = "1",
) -> dict:
    activities = []
    if bot_text is not None:
        activities.append(
            {
                "type": "message",
                "from": {"id": "bot", "role": "bot"},
                "text": bot_text,
            }
        )
    return {"activities": activities, "watermark": watermark}


# ---------------------------------------------------------------------------
# _create_conversation tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_create_conversation_stores_state() -> None:
    """POST /conversations with secret → _conversation_id, _token set."""
    project = _make_directline_project(secret="dl-channel-secret")
    caller = DirectLineAgentCaller(project)

    cm, client = _make_http_cm("post", 200, _conv_body("conv-123", "tok-abc"))
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=cm):
        await caller._create_conversation()

    assert caller._conversation_id == "conv-123"
    assert caller._token == "tok-abc"
    assert caller._watermark is None

    sent_headers = client.post.call_args[1]["headers"]
    assert sent_headers["Authorization"] == "Bearer dl-channel-secret"


@pytest.mark.anyio
async def test_create_conversation_missing_secret_raises_value_error() -> None:
    """_get_secret raises ValueError when value_encrypted is absent; propagates from _create_conversation."""
    project = {
        "agent_endpoint": "https://directline.botframework.com",
        "auth_config": {"type": "bearer"},  # no value_encrypted
        "schema_hints": {"caller_type": "directline"},
    }
    caller = DirectLineAgentCaller(project)

    with pytest.raises(ValueError, match="missing value_encrypted"):
        await caller._create_conversation()


@pytest.mark.anyio
async def test_create_conversation_non_200_raises() -> None:
    """Non-200 from POST /conversations raises _DirectLineError."""
    caller = DirectLineAgentCaller(_make_directline_project())

    cm, _ = _make_http_cm("post", 403, {"error": "Unauthorized"})
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=cm):
        with pytest.raises(_DirectLineError) as exc_info:
            await caller._create_conversation()

    assert exc_info.value.status_code == 403


@pytest.mark.anyio
async def test_create_conversation_invalid_json_raises() -> None:
    """Non-JSON response from POST /conversations raises _DirectLineError."""
    caller = DirectLineAgentCaller(_make_directline_project())

    cm, _ = _make_http_cm("post", 200, json_error=True)
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=cm):
        with pytest.raises(_DirectLineError):
            await caller._create_conversation()


@pytest.mark.anyio
async def test_create_conversation_missing_fields_raises() -> None:
    """Response missing conversationId or token raises _DirectLineError."""
    caller = DirectLineAgentCaller(_make_directline_project())

    cm, _ = _make_http_cm("post", 200, {"watermark": "0"})  # no conversationId/token
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=cm):
        with pytest.raises(_DirectLineError) as exc_info:
            await caller._create_conversation()

    assert "conversationId" in str(exc_info.value) or "token" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _post_activity tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_post_activity_sends_correct_payload() -> None:
    """POST activity includes X-LitmusAI-Session header and text field."""
    project = _make_directline_project()
    caller = DirectLineAgentCaller(project)
    caller._conversation_id = "conv-abc"
    caller._token = "sess-tok"

    cm, client = _make_http_cm("post", 200, {"id": "act|000001"})
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=cm):
        await caller._post_activity("Hello bot", "session_p1")

    sent_headers = client.post.call_args[1]["headers"]
    assert sent_headers["X-LitmusAI-Session"] == "session_p1"
    assert sent_headers["Authorization"] == "Bearer sess-tok"

    sent_body = client.post.call_args[1]["json"]
    assert sent_body["type"] == "message"
    assert sent_body["text"] == "Hello bot"


@pytest.mark.anyio
async def test_post_activity_non_200_raises() -> None:
    """Non-200 from POST activity raises _DirectLineError."""
    caller = DirectLineAgentCaller(_make_directline_project())
    caller._conversation_id = "conv-abc"
    caller._token = "tok"

    cm, _ = _make_http_cm("post", 429, {"error": "Too Many Requests"})
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=cm):
        with pytest.raises(_DirectLineError) as exc_info:
            await caller._post_activity("hi", "sess_1")

    assert exc_info.value.status_code == 429


# ---------------------------------------------------------------------------
# _poll_for_reply tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_poll_returns_bot_reply_immediately() -> None:
    """First GET returns bot activity → reply extracted, watermark advanced."""
    caller = DirectLineAgentCaller(_make_directline_project())
    caller._conversation_id = "conv-abc"
    caller._token = "sess-tok"
    caller._watermark = "0"

    cm, _ = _make_http_cm(
        "get", 200, _activities_body("Hello from bot!", watermark="2")
    )
    with (
        patch("caller.agent_caller.httpx.AsyncClient", return_value=cm),
        patch("caller.agent_caller.asyncio.sleep"),
    ):
        reply, raw_body = await caller._poll_for_reply()

    assert reply == "Hello from bot!"
    assert caller._watermark == "2"
    assert raw_body != ""


@pytest.mark.anyio
async def test_poll_retries_until_reply_appears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First N polls have no bot activity; reply found on N+1."""
    monkeypatch.setattr(DirectLineAgentCaller, "MAX_POLL_ATTEMPTS", 4)

    caller = DirectLineAgentCaller(_make_directline_project())
    caller._conversation_id = "conv-abc"
    caller._token = "sess-tok"

    # 3 empty polls, then 1 with bot reply
    empty = _activities_body(bot_text=None, watermark="1")
    reply_body = _activities_body(bot_text="Finally here!", watermark="2")

    cms = [_make_http_cm("get", 200, empty)[0] for _ in range(3)]
    cms.append(_make_http_cm("get", 200, reply_body)[0])

    with (
        patch("caller.agent_caller.httpx.AsyncClient", side_effect=cms),
        patch("caller.agent_caller.asyncio.sleep"),
    ):
        reply, _ = await caller._poll_for_reply()

    assert reply == "Finally here!"


@pytest.mark.anyio
async def test_poll_timeout_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exhausting MAX_POLL_ATTEMPTS without a bot reply returns (None, raw_body)."""
    monkeypatch.setattr(DirectLineAgentCaller, "MAX_POLL_ATTEMPTS", 3)

    caller = DirectLineAgentCaller(_make_directline_project())
    caller._conversation_id = "conv-abc"
    caller._token = "sess-tok"

    empty = _activities_body(bot_text=None)
    cms = [_make_http_cm("get", 200, empty)[0] for _ in range(3)]

    with (
        patch("caller.agent_caller.httpx.AsyncClient", side_effect=cms),
        patch("caller.agent_caller.asyncio.sleep"),
    ):
        reply, _ = await caller._poll_for_reply()

    assert reply is None


@pytest.mark.anyio
async def test_poll_watermark_sent_in_request() -> None:
    """Watermark is included as query param in GET activities."""
    caller = DirectLineAgentCaller(_make_directline_project())
    caller._conversation_id = "conv-abc"
    caller._token = "sess-tok"
    caller._watermark = "5"

    cm, client = _make_http_cm("get", 200, _activities_body("hi"))
    with (
        patch("caller.agent_caller.httpx.AsyncClient", return_value=cm),
        patch("caller.agent_caller.asyncio.sleep"),
    ):
        await caller._poll_for_reply()

    _, kwargs = client.get.call_args
    assert kwargs.get("params", {}).get("watermark") == "5"


# ---------------------------------------------------------------------------
# send() integration tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_send_initializes_conversation_on_first_call() -> None:
    """`send()` calls `_create_conversation` when _conversation_id is None."""
    project = _make_directline_project()
    caller = DirectLineAgentCaller(project)

    async def fake_create():
        caller._conversation_id = "conv-1"
        caller._token = "tok-1"

    with (
        patch.object(caller, "_create_conversation", fake_create),
        patch.object(caller, "_post_activity", AsyncMock()),
        patch.object(
            caller,
            "_poll_for_reply",
            AsyncMock(return_value=("Bot says hi", '{"activities":[]}')),
        ),
    ):
        result = await caller.send("Hello", "sess_p1", [])

    assert result.ok
    assert result.reply == "Bot says hi"
    assert caller._conversation_id == "conv-1"


@pytest.mark.anyio
async def test_send_skips_create_conversation_on_subsequent_calls() -> None:
    """`send()` does NOT call `_create_conversation` when already initialized."""
    project = _make_directline_project()
    caller = DirectLineAgentCaller(project)
    caller._conversation_id = "conv-existing"
    caller._token = "tok-existing"

    create_mock = AsyncMock()
    with (
        patch.object(caller, "_create_conversation", create_mock),
        patch.object(caller, "_post_activity", AsyncMock()),
        patch.object(
            caller,
            "_poll_for_reply",
            AsyncMock(return_value=("Reply", "raw")),
        ),
    ):
        result = await caller.send("Turn 2", "sess_p1", [])

    create_mock.assert_not_called()
    assert result.ok


@pytest.mark.anyio
async def test_send_create_conversation_error_returns_caller_result() -> None:
    """Error in `_create_conversation` surfaces as CallerResult.error, not raised."""
    caller = DirectLineAgentCaller(_make_directline_project())

    async def fail_create():
        raise _DirectLineError("Unauthorized", status_code=401, raw_body="Forbidden")

    with patch.object(caller, "_create_conversation", fail_create):
        result = await caller.send("Hello", "sess_p1", [])

    assert not result.ok
    assert result.reply is None
    assert result.status_code == 401
    assert "Unauthorized" in result.error


@pytest.mark.anyio
async def test_send_post_activity_error_returns_caller_result() -> None:
    """Error in `_post_activity` (e.g. 429) surfaces as CallerResult.error."""
    caller = DirectLineAgentCaller(_make_directline_project())
    caller._conversation_id = "conv-1"
    caller._token = "tok"

    async def fail_post(msg, sid):
        raise _DirectLineError(
            "Too Many Requests", status_code=429, raw_body='{"error":"rate limited"}'
        )

    with patch.object(caller, "_post_activity", fail_post):
        result = await caller.send("Hello", "sess_p1", [])

    assert not result.ok
    assert result.status_code == 429
    assert result.reply is None


@pytest.mark.anyio
async def test_send_poll_timeout_returns_error() -> None:
    """Polling timeout surfaces as CallerResult.error (reply is None)."""
    caller = DirectLineAgentCaller(_make_directline_project())
    caller._conversation_id = "conv-1"
    caller._token = "tok"

    with (
        patch.object(caller, "_post_activity", AsyncMock()),
        patch.object(caller, "_poll_for_reply", AsyncMock(return_value=(None, ""))),
    ):
        result = await caller.send("Hello", "sess_p1", [])

    assert not result.ok
    assert result.reply is None
    assert result.error is not None
    assert "timeout" in result.error.lower() or "did not reply" in result.error.lower()


@pytest.mark.anyio
async def test_send_value_error_from_get_secret_returns_caller_result() -> None:
    """Missing value_encrypted surfaces as CallerResult.error (not a raised exception)."""
    project = {
        "agent_endpoint": "https://directline.botframework.com",
        "auth_config": {"type": "bearer"},  # no value_encrypted
        "schema_hints": {"caller_type": "directline"},
    }
    caller = DirectLineAgentCaller(project)

    result = await caller.send("Hello", "sess_p1", [])

    assert not result.ok
    assert result.reply is None
    assert result.status_code == 0
    assert "missing value_encrypted" in result.error.lower()


@pytest.mark.anyio
async def test_send_refreshes_token_on_401_from_post_activity() -> None:
    """401 from _post_activity triggers one-shot conversation refresh and retry."""
    caller = DirectLineAgentCaller(_make_directline_project())
    caller._conversation_id = "conv-old"
    caller._token = "tok-expired"

    refresh_called = []
    post_calls = []

    async def fake_refresh():
        refresh_called.append(True)
        caller._conversation_id = "conv-new"
        caller._token = "tok-fresh"

    async def fail_then_succeed(msg, sid):
        post_calls.append(sid)
        if len(post_calls) == 1:
            raise _DirectLineError("Unauthorized", status_code=401, raw_body="")

    with (
        patch.object(caller, "_create_conversation", fake_refresh),
        patch.object(caller, "_post_activity", fail_then_succeed),
        patch.object(
            caller, "_poll_for_reply", AsyncMock(return_value=("Fresh reply", ""))
        ),
    ):
        result = await caller.send("Hello", "sess_p1", [])

    assert result.ok
    assert result.reply == "Fresh reply"
    assert len(refresh_called) == 1
    assert len(post_calls) == 2  # first (expired), then retry (fresh)


@pytest.mark.anyio
async def test_send_returns_error_when_refresh_also_fails() -> None:
    """If the one-shot token refresh also fails, CallerResult.error is returned."""
    caller = DirectLineAgentCaller(_make_directline_project())
    caller._conversation_id = "conv-old"
    caller._token = "tok-expired"

    async def fail_post(msg, sid):
        raise _DirectLineError("Unauthorized", status_code=401, raw_body="")

    async def fail_refresh():
        raise _DirectLineError("Service unavailable", status_code=503, raw_body="")

    with (
        patch.object(caller, "_create_conversation", fail_refresh),
        patch.object(caller, "_post_activity", fail_post),
    ):
        result = await caller.send("Hello", "sess_p1", [])

    assert not result.ok
    assert result.status_code == 503


@pytest.mark.anyio
async def test_poll_401_clears_state_and_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """401 during polling resets _conversation_id/_token so next send() re-authenticates."""
    monkeypatch.setattr(DirectLineAgentCaller, "MAX_POLL_ATTEMPTS", 3)

    caller = DirectLineAgentCaller(_make_directline_project())
    caller._conversation_id = "conv-abc"
    caller._token = "tok-expired"

    cm, _ = _make_http_cm("get", 401, {"error": "Unauthorized"})
    with patch("caller.agent_caller.httpx.AsyncClient", return_value=cm):
        reply, _ = await caller._poll_for_reply()

    assert reply is None
    assert caller._conversation_id is None
    assert caller._token is None


# ---------------------------------------------------------------------------
# create_agent_caller() factory tests
# ---------------------------------------------------------------------------


def test_factory_returns_directline_caller() -> None:
    """`create_agent_caller` returns DirectLineAgentCaller when hint is set."""
    project = _make_directline_project()
    caller = create_agent_caller(project)
    assert isinstance(caller, DirectLineAgentCaller)


def test_factory_returns_standard_caller_without_hint() -> None:
    """`create_agent_caller` returns AgentCaller when no caller_type hint."""
    project = _make_standard_project()
    caller = create_agent_caller(project)
    assert isinstance(caller, AgentCaller)


def test_factory_returns_standard_caller_for_unknown_caller_type() -> None:
    """Unknown caller_type value falls back to AgentCaller."""
    project = _make_standard_project()
    project["schema_hints"] = {"caller_type": "something_unknown"}
    caller = create_agent_caller(project)
    assert isinstance(caller, AgentCaller)


def test_factory_passes_timeout_to_directline() -> None:
    """Timeout is forwarded to DirectLineAgentCaller."""
    project = _make_directline_project()
    caller = create_agent_caller(project, timeout_s=15.0)
    assert isinstance(caller, DirectLineAgentCaller)
    assert caller._timeout_s == 15.0


# ---------------------------------------------------------------------------
# SimulationAgentCaller DirectLine routing test
# ---------------------------------------------------------------------------


def test_simulation_agent_caller_uses_directline_caller() -> None:
    """SimulationAgentCaller creates a DirectLineAgentCaller for directline projects."""
    project = _make_directline_project()
    sim_caller = SimulationAgentCaller(project, mock=False)
    assert isinstance(sim_caller._caller, DirectLineAgentCaller)


def test_simulation_agent_caller_uses_standard_caller_by_default() -> None:
    """SimulationAgentCaller creates an AgentCaller for standard projects."""
    project = _make_standard_project()
    sim_caller = SimulationAgentCaller(project, mock=False)
    assert isinstance(sim_caller._caller, AgentCaller)
