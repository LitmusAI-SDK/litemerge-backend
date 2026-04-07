"""AgentCaller — sends persona turns to the customer's agent endpoint.

Usage::

    caller = AgentCaller(project_doc)
    result = await caller.send(
        message="Hello, can you help me?",
        session_id="sess_abc",
        history=[],
    )
    # result.reply, result.latency_ms, result.status_code, result.raw_body, result.error
"""

import base64
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

    def __init__(self, project_config: dict, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
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
        elif auth_type == "apikey":
            header_name = self._auth_config.get("header_name") or "X-Api-Key"
            headers[header_name] = secret
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
# Simulation layer — retry sentinel, response shape, and mock injection
# ---------------------------------------------------------------------------

import os
import random


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
    """Wraps AgentCaller with mock failure injection and a uniform AgentResponse shape.

    Args:
        project_config: Full project document from MongoDB.
        timeout_s: Per-request HTTP timeout in seconds.
        mock: Override USE_MOCK_AGENT env var (useful in tests).
    """

    def __init__(
        self,
        project_config: dict,
        timeout_s: int = 30,
        mock: bool | None = None,
    ) -> None:
        self._caller = AgentCaller(project_config, timeout_s=float(timeout_s))
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
