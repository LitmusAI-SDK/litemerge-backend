"""AgentCaller — sends persona turns to the customer's agent endpoint.

Usage::

    caller = AgentCaller(project_doc)
    result = await caller.send(
        message="Hello, can you help me?",
        session_id="sess_abc",
        history=[],
    )
    # result.reply, result.latency_ms, result.status_code, result.raw_body, result.error

DirectLine usage::

    # Mode A — channel secret (preferred when available):
    # schema_hints = {"caller_type": "directline"}
    # agent_endpoint = "https://directline.botframework.com"
    # auth_config.value = long-lived DirectLine channel secret.
    #
    # Mode B — captured conversation (no channel secret available, e.g. Air India):
    # schema_hints = {
    #     "caller_type": "directline",
    #     "directline_conversation_id": "<conversationId captured from browser>",
    # }
    # auth_config.value = the per-conversation Bearer token captured from the
    # widget's Authorization header.  Token will expire (~1h) and cannot be
    # refreshed without the channel secret — capture a fresh one when it does.
    caller = DirectLineAgentCaller(project_doc)
    result = await caller.send(message="Hello", session_id="sess_abc", history=[])
"""

import asyncio
import base64
import os
import random
import time
from dataclasses import dataclass, field

import httpx

from core.crypto import decrypt_secret

DEFAULT_TIMEOUT_S: float = 30.0

# Canonical field names → default wire names
_FIELD_DEFAULTS: dict[str, str] = {
    "message": "message",
    "session_id": "session_id",
    "conversation_history": "conversation_history",
    "reply": "reply",
}


@dataclass
class CallerResult:
    """Outcome of a single call to the customer's agent endpoint."""

    reply: str | None
    latency_ms: float
    status_code: int
    raw_body: str
    error: str | None = field(default=None)

    @property
    def ok(self) -> bool:
        return self.error is None and self.reply is not None


class AgentCaller:
    """Sends a single persona message to the customer's agent and captures the result.

    Args:
        project_config: A project document dict as stored in MongoDB.  Must
            contain ``agent_endpoint``, ``auth_config``, and optionally
            ``schema_hints``.
        timeout_s: Per-request timeout in seconds (default 30).
    """

    def __init__(
        self, project_config: dict, timeout_s: float = DEFAULT_TIMEOUT_S
    ) -> None:
        self._endpoint: str = project_config["agent_endpoint"]
        self._auth_config: dict = project_config.get("auth_config") or {}
        self._schema_hints: dict[str, str] = project_config.get("schema_hints") or {}
        self._timeout_s = timeout_s

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(self, session_id: str) -> dict[str, str]:
        headers: dict[str, str] = {
            "X-LitmusAI-Session": session_id,
        }

        auth_type = self._auth_config.get("type", "none")
        if auth_type == "none":
            return headers

        value_encrypted = self._auth_config.get("value_encrypted")
        if not value_encrypted:
            return headers

        secret = decrypt_secret(value_encrypted)

        if auth_type == "bearer":
            headers["Authorization"] = f"Bearer {secret}"
        elif auth_type == "basic":
            encoded = base64.b64encode(secret.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        return headers

    def _build_body(
        self,
        message: str,
        session_id: str,
        history: list[dict],
    ) -> dict:
        field_map = {**_FIELD_DEFAULTS, **self._schema_hints}
        return {
            field_map["message"]: message,
            field_map["session_id"]: session_id,
            field_map["conversation_history"]: history,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def send(
        self,
        message: str,
        session_id: str,
        history: list[dict],
    ) -> CallerResult:
        """POST one persona turn to the customer's agent endpoint.

        Returns a :class:`CallerResult` regardless of outcome — errors are
        surfaced via ``result.error`` rather than raised, so callers can log
        and continue without crashing a simulation run.
        """
        headers = self._build_headers(session_id)
        body = self._build_body(message, session_id, history)

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(self._endpoint, json=body, headers=headers)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except httpx.TimeoutException:
            latency_ms = self._timeout_s * 1000.0
            return CallerResult(
                reply=None,
                latency_ms=latency_ms,
                status_code=0,
                raw_body="",
                error=f"Request timed out after {self._timeout_s}s",
            )
        except httpx.RequestError as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return CallerResult(
                reply=None,
                latency_ms=latency_ms,
                status_code=0,
                raw_body="",
                error=f"Request error: {exc}",
            )

        raw_body = response.text
        status_code = response.status_code

        if status_code != 200:
            return CallerResult(
                reply=None,
                latency_ms=latency_ms,
                status_code=status_code,
                raw_body=raw_body,
                error=f"Agent returned HTTP {status_code}",
            )

        try:
            data = response.json()
        except Exception:
            return CallerResult(
                reply=None,
                latency_ms=latency_ms,
                status_code=status_code,
                raw_body=raw_body,
                error="Response body is not valid JSON",
            )

        reply_field = self._schema_hints.get("reply", _FIELD_DEFAULTS["reply"])
        reply = data.get(reply_field)

        if not isinstance(reply, str):
            return CallerResult(
                reply=None,
                latency_ms=latency_ms,
                status_code=status_code,
                raw_body=raw_body,
                error=f"Response missing expected reply field '{reply_field}'",
            )

        return CallerResult(
            reply=reply,
            latency_ms=latency_ms,
            status_code=status_code,
            raw_body=raw_body,
        )


# ---------------------------------------------------------------------------
# DirectLine adapter — two-step send/poll flow for Bot Framework endpoints
# ---------------------------------------------------------------------------


class DirectLineAgentCaller:
    """Implements the Microsoft DirectLine v3 protocol for Bot Framework bots.

    DirectLine requires three HTTP operations per persona turn:
    1. (First turn only) POST /v3/directline/conversations — creates a new
       conversation and returns a short-lived session token.
    2. POST /v3/directline/conversations/{id}/activities — sends the message;
       returns only an activity ID (not the bot reply).
    3. GET /v3/directline/conversations/{id}/activities?watermark=... — poll
       until a bot-role message activity appears.

    Session state (``_conversation_id``, ``_token``, ``_watermark``) is held
    in-memory on the instance.  Because ``SimulationAgentCaller`` creates one
    instance per ``PersonaSession``, the state persists across all turns of a
    session but is discarded when the session ends.

    Args:
        project_config: Full project document from MongoDB.  ``agent_endpoint``
            must be the DirectLine base URL (e.g.
            ``https://directline.botframework.com``).  ``auth_config.value``
            must be the long-lived DirectLine channel secret.
        timeout_s: Per-request HTTP timeout in seconds (default 30).
    """

    POLL_INTERVAL_S: float = 1.0
    MAX_POLL_ATTEMPTS: int = 30  # 30 s total wait before giving up
    _CONVERSATIONS_PATH: str = "/v3/directline/conversations"

    def __init__(
        self, project_config: dict, timeout_s: float = DEFAULT_TIMEOUT_S
    ) -> None:
        self._base_url: str = str(project_config["agent_endpoint"]).rstrip("/")
        self._auth_config: dict = project_config.get("auth_config") or {}
        schema_hints: dict = project_config.get("schema_hints") or {}
        self._timeout_s = timeout_s
        # Session state — populated on first send()
        self._conversation_id: str | None = None
        self._token: str | None = None
        self._watermark: str | None = None

        # Captured-conversation mode: operator supplied a conversationId they
        # pulled from the browser.  Treat auth_config.value as the per-conversation
        # Bearer token (NOT a channel secret) and skip _create_conversation.
        captured_conv_id = schema_hints.get("directline_conversation_id")
        if captured_conv_id:
            self._conversation_id = str(captured_conv_id)
            try:
                self._token = self._get_secret()
            except ValueError:
                # Leave _token as None — send() will surface the error cleanly.
                self._token = None
            self._captured_token_mode = True
        else:
            self._captured_token_mode = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_secret(self) -> str:
        """Decrypt and return the DirectLine channel secret."""
        value_encrypted = self._auth_config.get("value_encrypted")
        if not value_encrypted:
            raise ValueError("DirectLine auth_config is missing value_encrypted")
        return decrypt_secret(value_encrypted)

    async def _create_conversation(self) -> None:
        """POST /v3/directline/conversations to obtain conversationId and token.

        Uses the long-lived channel secret for authentication.  On success,
        populates ``_conversation_id`` and ``_token``.
        """
        url = f"{self._base_url}{self._CONVERSATIONS_PATH}"
        secret = self._get_secret()
        headers = {
            "Authorization": f"Bearer {secret}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(url, headers=headers, json={})

        if response.status_code not in (200, 201):
            raise _DirectLineError(
                f"Failed to create DirectLine conversation: HTTP {response.status_code}",
                status_code=response.status_code,
                raw_body=response.text,
            )

        try:
            data = response.json()
        except Exception as exc:
            raise _DirectLineError(
                "DirectLine conversation response is not valid JSON",
                status_code=response.status_code,
                raw_body=response.text,
            ) from exc

        conversation_id = data.get("conversationId")
        token = data.get("token")
        if not conversation_id or not token:
            raise _DirectLineError(
                "DirectLine conversation response missing conversationId or token",
                status_code=response.status_code,
                raw_body=response.text,
            )

        self._conversation_id = conversation_id
        self._token = token
        self._watermark = None

    async def _post_activity(self, message: str, session_id: str) -> None:
        """POST a message activity to the conversation.

        Raises:
            _DirectLineError: on any non-200/201 response.
        """
        url = (
            f"{self._base_url}{self._CONVERSATIONS_PATH}"
            f"/{self._conversation_id}/activities"
        )
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "X-LitmusAI-Session": session_id,
        }
        body = {
            "type": "message",
            "from": {"id": session_id},
            "text": message,
        }
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(url, headers=headers, json=body)

        if response.status_code not in (200, 201):
            raise _DirectLineError(
                f"DirectLine POST activity failed: HTTP {response.status_code}",
                status_code=response.status_code,
                raw_body=response.text,
            )

    async def _poll_for_reply(self) -> tuple[str | None, str]:
        """Poll GET activities until a bot message activity appears.

        Returns:
            ``(reply_text, raw_body)`` where ``reply_text`` is ``None`` if no
            bot reply was found within ``MAX_POLL_ATTEMPTS`` polls.
        """
        url = (
            f"{self._base_url}{self._CONVERSATIONS_PATH}"
            f"/{self._conversation_id}/activities"
        )
        headers = {"Authorization": f"Bearer {self._token}"}
        last_raw_body = ""

        for attempt in range(self.MAX_POLL_ATTEMPTS):
            params: dict[str, str] = {}
            if self._watermark is not None:
                params["watermark"] = self._watermark

            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.get(url, headers=headers, params=params)

            last_raw_body = response.text

            if response.status_code in (401, 403):
                # Auth token expired mid-poll.  In channel-secret mode, clear
                # state so next send() recreates the conversation.  In
                # captured-token mode we cannot refresh — leave state intact
                # and let the next send() surface the expiry cleanly.
                if not self._captured_token_mode:
                    self._conversation_id = None
                    self._token = None
                break

            if response.status_code != 200:
                # Other transient error — keep trying.
                if attempt < self.MAX_POLL_ATTEMPTS - 1:
                    await asyncio.sleep(self.POLL_INTERVAL_S)
                continue

            try:
                data = response.json()
            except Exception:
                if attempt < self.MAX_POLL_ATTEMPTS - 1:
                    await asyncio.sleep(self.POLL_INTERVAL_S)
                continue

            new_watermark = data.get("watermark")
            if new_watermark is not None:
                self._watermark = str(new_watermark)

            bot_messages = [
                a
                for a in data.get("activities", [])
                if a.get("type") == "message" and a.get("from", {}).get("role") == "bot"
            ]

            if bot_messages:
                reply = bot_messages[-1].get("text") or ""
                return reply, last_raw_body

            if attempt < self.MAX_POLL_ATTEMPTS - 1:
                await asyncio.sleep(self.POLL_INTERVAL_S)

        return None, last_raw_body

    # ------------------------------------------------------------------
    # Public interface — matches AgentCaller.send() signature
    # ------------------------------------------------------------------

    async def send(
        self,
        message: str,
        session_id: str,
        history: list[dict],  # noqa: ARG002 — DirectLine bots are stateful server-side
    ) -> CallerResult:
        """Execute a full DirectLine turn: initialise session if needed, post
        activity, then poll for the bot reply.

        Returns a :class:`CallerResult` regardless of outcome — errors surface
        via ``result.error`` rather than being raised.
        """
        t0 = time.perf_counter()

        # Captured-token mode: conversation was pre-populated from schema_hints.
        # If the token decryption failed in __init__, surface that now.
        if self._captured_token_mode and self._token is None:
            return CallerResult(
                reply=None,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                status_code=0,
                raw_body="",
                error=(
                    "DirectLine captured-token mode requires auth_config.value_encrypted"
                    " (the Bearer token captured from the widget)."
                ),
            )

        # Step 1: create conversation on the first turn
        if self._conversation_id is None:
            try:
                await self._create_conversation()
            except ValueError as exc:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return CallerResult(
                    reply=None,
                    latency_ms=latency_ms,
                    status_code=0,
                    raw_body="",
                    error=str(exc),
                )
            except _DirectLineError as exc:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return CallerResult(
                    reply=None,
                    latency_ms=latency_ms,
                    status_code=exc.status_code,
                    raw_body=exc.raw_body,
                    error=exc.args[0],
                )
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return CallerResult(
                    reply=None,
                    latency_ms=latency_ms,
                    status_code=0,
                    raw_body="",
                    error=f"Error creating DirectLine conversation: {exc}",
                )

        # Step 2: post the message activity (with one-shot token refresh on 401/403)
        try:
            await self._post_activity(message, session_id)
        except _DirectLineError as exc:
            # In captured-token mode we cannot refresh (no channel secret) —
            # surface 401/403 directly with a clear hint.
            if self._captured_token_mode and exc.status_code in (401, 403):
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return CallerResult(
                    reply=None,
                    latency_ms=latency_ms,
                    status_code=exc.status_code,
                    raw_body=exc.raw_body,
                    error=(
                        "DirectLine captured token expired or rejected. Capture a"
                        " fresh Bearer token from the widget and update auth_config."
                    ),
                )
            if exc.status_code not in (401, 403):
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return CallerResult(
                    reply=None,
                    latency_ms=latency_ms,
                    status_code=exc.status_code,
                    raw_body=exc.raw_body,
                    error=exc.args[0],
                )
            # Token expired — re-create the conversation once and retry
            try:
                await self._create_conversation()
                await self._post_activity(message, session_id)
            except (
                ValueError,
                _DirectLineError,
                httpx.TimeoutException,
                httpx.RequestError,
            ) as retry_exc:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                if isinstance(retry_exc, _DirectLineError):
                    return CallerResult(
                        reply=None,
                        latency_ms=latency_ms,
                        status_code=retry_exc.status_code,
                        raw_body=retry_exc.raw_body,
                        error=retry_exc.args[0],
                    )
                return CallerResult(
                    reply=None,
                    latency_ms=latency_ms,
                    status_code=0,
                    raw_body="",
                    error=f"Token refresh failed: {retry_exc}",
                )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return CallerResult(
                reply=None,
                latency_ms=latency_ms,
                status_code=0,
                raw_body="",
                error=f"Error posting DirectLine activity: {exc}",
            )

        # Step 3: poll for bot reply
        reply, raw_body = await self._poll_for_reply()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if reply is None:
            return CallerResult(
                reply=None,
                latency_ms=latency_ms,
                status_code=200,
                raw_body=raw_body,
                error="DirectLine bot did not reply within the polling timeout",
            )

        return CallerResult(
            reply=reply,
            latency_ms=latency_ms,
            status_code=200,
            raw_body=raw_body,
        )


class _DirectLineError(Exception):
    """Internal sentinel for DirectLine HTTP errors."""

    def __init__(self, message: str, status_code: int, raw_body: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.raw_body = raw_body


# ---------------------------------------------------------------------------
# Factory — returns the right caller based on schema_hints.caller_type
# ---------------------------------------------------------------------------


def create_agent_caller(
    project_config: dict,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> "AgentCaller | DirectLineAgentCaller":
    """Return the appropriate caller for the given project configuration.

    If ``schema_hints.caller_type == "directline"``, returns a
    :class:`DirectLineAgentCaller`; otherwise returns a standard
    :class:`AgentCaller`.
    """
    schema_hints: dict = project_config.get("schema_hints") or {}
    if schema_hints.get("caller_type") == "directline":
        return DirectLineAgentCaller(project_config, timeout_s=timeout_s)
    return AgentCaller(project_config, timeout_s=timeout_s)


# ---------------------------------------------------------------------------
# Simulation layer — retry sentinel, response shape, and mock injection
# ---------------------------------------------------------------------------


class AgentRetriableError(Exception):
    """Raised when the agent returns a non-200 response, signalling tenacity to retry."""


@dataclass
class AgentResponse:
    """Uniform response shape used by the simulation session loop."""

    reply: str | None
    status_code: int
    latency_ms: float
    raw_body: str


_USE_MOCK_AGENT: bool = os.getenv("USE_MOCK_AGENT", "false").lower() == "true"


class SimulationAgentCaller:
    """Wraps AgentCaller (or DirectLineAgentCaller) with mock failure injection
    and a uniform AgentResponse shape.

    The correct underlying caller is selected automatically via
    :func:`create_agent_caller` based on ``schema_hints.caller_type``.

    Args:
        project_config: Full project document from MongoDB.
        timeout_s: Per-request HTTP timeout in seconds.
        mock: Override USE_MOCK_AGENT env var (useful in tests).
    """

    def __init__(
        self,
        project_config: dict,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        mock: bool | None = None,
    ) -> None:
        self._caller: AgentCaller | DirectLineAgentCaller = create_agent_caller(
            project_config, timeout_s=timeout_s
        )
        self._mock = mock if mock is not None else _USE_MOCK_AGENT

    async def send(
        self,
        message: str,
        session_id: str,
        history: list[dict],
    ) -> AgentResponse:
        if self._mock:
            return self._mock_response()

        result = await self._caller.send(message, session_id, history)
        return AgentResponse(
            reply=result.reply,
            status_code=result.status_code,
            latency_ms=result.latency_ms,
            raw_body=result.raw_body,
        )

    @staticmethod
    def _mock_response() -> AgentResponse:
        roll = random.random()
        if roll < 0.80:
            return AgentResponse(
                reply="This is a mock agent response.",
                status_code=200,
                latency_ms=50.0,
                raw_body='{"reply": "This is a mock agent response."}',
            )
        elif roll < 0.90:
            return AgentResponse(
                reply=None,
                status_code=429,
                latency_ms=20.0,
                raw_body='{"error": "Too Many Requests"}',
            )
        else:
            return AgentResponse(
                reply=None,
                status_code=503,
                latency_ms=20.0,
                raw_body='{"error": "Service Unavailable"}',
            )
