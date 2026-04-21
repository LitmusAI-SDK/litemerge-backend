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

TMobile usage::

    # T-Mobile /self-service-flex/v1/info-bot-chat endpoint.
    # Turn 1 body: {"userInput": "..."}.
    # Turn 2+ body: {"userInput": "...", "conversationId": "<id from turn 1 response>"}.
    # Reply field: response["content"].
    # schema_hints = {"caller_type": "tmobile"}
    # agent_endpoint = "https://<host>/self-service-flex/v1/info-bot-chat"
    # auth_config = bearer token (JWT).
    caller = TMobileAgentCaller(project_doc)
    result = await caller.send(message="Hello", session_id="sess_abc", history=[])
"""

import asyncio
import base64
import logging
import os
import random
import time
from dataclasses import dataclass, field

import httpx

from core.crypto import decrypt_secret

logger = logging.getLogger(__name__)

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
    _TOKEN_REFRESH_PATH: str = "/v3/directline/tokens/refresh"

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
        print(f"[DIRECTLINE] _create_conversation url={url}", flush=True)
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(url, headers=headers, json={})
        print(f"[DIRECTLINE] _create_conversation status={response.status_code} body={response.text[:300]!r}", flush=True)

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

    async def _refresh_token(self) -> None:
        """POST /v3/directline/tokens/refresh to extend the current token.

        Used in captured-token mode instead of re-creating the conversation.
        Updates ``_token`` in place; ``_conversation_id`` and ``_watermark``
        are preserved.

        Raises:
            _DirectLineError: if the refresh request fails.
        """
        url = f"{self._base_url}{self._TOKEN_REFRESH_PATH}"
        headers = {"Authorization": f"Bearer {self._token}"}
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(url, headers=headers)

        if response.status_code not in (200, 201):
            raise _DirectLineError(
                f"DirectLine token refresh failed: HTTP {response.status_code}."
                " Capture a fresh token from the widget and update auth_config.",
                status_code=response.status_code,
                raw_body=response.text,
            )

        try:
            data = response.json()
        except Exception as exc:
            raise _DirectLineError(
                "DirectLine token refresh response is not valid JSON",
                status_code=response.status_code,
                raw_body=response.text,
            ) from exc

        new_token = data.get("token")
        if not new_token:
            raise _DirectLineError(
                "DirectLine token refresh response missing token field",
                status_code=response.status_code,
                raw_body=response.text,
            )

        self._token = new_token

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
        print(f"[DIRECTLINE] _post_activity url={url} conv={self._conversation_id}", flush=True)
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(url, headers=headers, json=body)
        print(f"[DIRECTLINE] _post_activity status={response.status_code} body={response.text[:300]!r}", flush=True)

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
                if self._captured_token_mode:
                    # Try token refresh and retry the poll immediately.
                    try:
                        await self._refresh_token()
                        continue
                    except _DirectLineError:
                        break
                else:
                    # Channel-secret mode: clear state so next send() recreates.
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

            activities = data.get("activities", [])
            # logger.warning(
            #     "DirectLine poll %d/%d conv=%s watermark=%s: %d activities from.ids=%s types=%s roles=%s",
            #     attempt + 1, self.MAX_POLL_ATTEMPTS, self._conversation_id, self._watermark,
            #     len(activities), [a.get("from", {}).get("id") for a in activities],
            #     [a.get("type") for a in activities], [a.get("from", {}).get("role") for a in activities],
            # )
            print(
                f"[DIRECTLINE] poll {attempt + 1}/{self.MAX_POLL_ATTEMPTS}"
                f" conv={self._conversation_id} watermark={self._watermark}"
                f" activities={len(activities)}"
                f" from.ids={[a.get('from', {}).get('id') for a in activities]}"
                f" types={[a.get('type') for a in activities]}"
                f" roles={[a.get('from', {}).get('role') for a in activities]}",
                flush=True,
            )

            bot_messages = [
                a
                for a in activities
                if a.get("type") == "message"
                and (
                    a.get("from", {}).get("role") == "bot"
                    or not a.get("from", {}).get("id", "").startswith("dl_")
                )
                and (a.get("text") or "").strip()
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
            # In captured-token mode, refresh via /tokens/refresh then retry once.
            if self._captured_token_mode and exc.status_code in (401, 403):
                try:
                    await self._refresh_token()
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
                # Refresh + retry succeeded — fall through to poll
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
# TMobile adapter — stateful REST caller for T-Mobile info-bot-chat endpoint
# ---------------------------------------------------------------------------


class TMobileAgentCaller:
    """Caller for the T-Mobile /self-service-flex/v1/info-bot-chat endpoint.

    T-Mobile requires a set of fixed proprietary headers on every request plus
    a pre-supplied ``conversationId`` (captured from a real browser session).
    All of these are configured via ``schema_hints``:

    Required schema_hints keys:
        ``tmobile_conversation_id``   — conversationId captured from the browser
        ``tmobile_session_id``        — session-id header value
        ``tmobile_interaction_id``    — interaction-id header value
        ``tmobile_activity_id``       — activity-id header value (can reuse interaction-id)
        ``tmobile_x_auth_originator`` — x-auth-originator header JWT
        ``tmobile_workflow_id``       — workflow-id (e.g. "UPGRADE")
        ``tmobile_sub_workflow_id``   — sub-workflow-id (e.g. "Contact Us")

    auth_config must be a bearer token (the Authorization JWT).

    The reply is at ``response["content"]``.
    """

    # Fixed headers that don't change between sessions
    _STATIC_HEADERS: dict[str, str] = {
        "Content-Type": "application/json",
        "channel-id": "WEB",
        "application-id": "SDLCC0603210",
        "origin-application-id": "SDLCC0603210",
        "ai-chat-lang": "en",
        "Referer": "https://www.t-mobile.com/contact-us",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/147.0.0.0 Safari/537.36"
        ),
        "sec-ch-ua": '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
    }

    def __init__(
        self, project_config: dict, timeout_s: float = DEFAULT_TIMEOUT_S
    ) -> None:
        self._endpoint: str = project_config["agent_endpoint"]
        self._auth_config: dict = project_config.get("auth_config") or {}
        self._schema_hints: dict = project_config.get("schema_hints") or {}
        self._timeout_s = timeout_s
        # Pre-supplied conversationId from schema_hints — used from turn 1
        self._conversation_id: str | None = self._schema_hints.get("tmobile_conversation_id") or None

    def _build_headers(self) -> dict[str, str]:
        import uuid
        headers: dict[str, str] = dict(self._STATIC_HEADERS)

        # Authorization JWT
        value_encrypted = self._auth_config.get("value_encrypted")
        if value_encrypted:
            secret = decrypt_secret(value_encrypted)
            headers["Authorization"] = f"Bearer {secret}"
            headers["x-auth-originator"] = self._schema_hints.get("tmobile_x_auth_originator") or secret

        # Per-session fixed headers from schema_hints
        if sid := self._schema_hints.get("tmobile_session_id"):
            headers["session-id"] = sid
        if iid := self._schema_hints.get("tmobile_interaction_id"):
            headers["interaction-id"] = iid
        if wid := self._schema_hints.get("tmobile_workflow_id"):
            headers["workflow-id"] = wid
        if swid := self._schema_hints.get("tmobile_sub_workflow_id"):
            headers["sub-workflow-id"] = swid

        # Per-request generated headers
        headers["service-transaction-id"] = str(uuid.uuid4())
        headers["activity-id"] = str(uuid.uuid4())
        headers["timestamp"] = __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")

        return headers

    def _build_body(self, message: str) -> dict:
        body: dict = {"userInput": message}
        if self._conversation_id is not None:
            body["conversationId"] = self._conversation_id
        return body

    async def send(
        self,
        message: str,
        session_id: str,  # noqa: ARG002 — T-Mobile bot is stateful server-side
        history: list[dict],  # noqa: ARG002
    ) -> CallerResult:
        headers = self._build_headers()
        body = self._build_body(message)

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

        # Capture conversationId from the first response.
        if self._conversation_id is None:
            conv_id = data.get("conversationId")
            if conv_id:
                self._conversation_id = str(conv_id)

        reply = data.get("content")
        if not isinstance(reply, str):
            return CallerResult(
                reply=None,
                latency_ms=latency_ms,
                status_code=status_code,
                raw_body=raw_body,
                error="Response missing expected reply field 'content'",
            )

        return CallerResult(
            reply=reply,
            latency_ms=latency_ms,
            status_code=status_code,
            raw_body=raw_body,
        )


# ---------------------------------------------------------------------------
# Factory — returns the right caller based on schema_hints.caller_type
# ---------------------------------------------------------------------------


def create_agent_caller(
    project_config: dict,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> "AgentCaller | DirectLineAgentCaller | TMobileAgentCaller":
    """Return the appropriate caller for the given project configuration.

    Selects based on ``schema_hints.caller_type``:
    - ``"directline"`` → :class:`DirectLineAgentCaller`
    - ``"tmobile"`` → :class:`TMobileAgentCaller`
    - anything else → :class:`AgentCaller`
    """
    schema_hints: dict = project_config.get("schema_hints") or {}
    caller_type = schema_hints.get("caller_type")
    if caller_type == "directline":
        return DirectLineAgentCaller(project_config, timeout_s=timeout_s)
    if caller_type == "tmobile":
        return TMobileAgentCaller(project_config, timeout_s=timeout_s)
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
        self._caller: AgentCaller | DirectLineAgentCaller | TMobileAgentCaller = create_agent_caller(
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
        print(
            f"[AGENT_CALLER] session={session_id} ok={result.ok}"
            f" status={result.status_code} error={result.error!r}"
            f" raw_body={result.raw_body[:300]!r}",
            flush=True,
        )
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
