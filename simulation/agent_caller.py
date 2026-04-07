"""SimulationAgentCaller — thin wrapper around caller.AgentCaller.

Adds:
  - AgentResponse dataclass (reply, status_code, latency_ms, raw_body)
  - AgentRetriableError sentinel for tenacity retry logic
  - Mock failure injection controlled by USE_MOCK_AGENT env var

Mock probability distribution (USE_MOCK_AGENT=true):
  0.80 → 200 OK with a generic reply string
  0.10 → 429 Too Many Requests
  0.10 → 503 Service Unavailable
"""

import os
import random
from dataclasses import dataclass

from caller.agent_caller import AgentCaller as _BaseAgentCaller

USE_MOCK_AGENT: bool = os.getenv("USE_MOCK_AGENT", "false").lower() == "true"


class AgentRetriableError(Exception):
    """Raised when the agent returns a non-200 response, signalling tenacity to retry."""


@dataclass
class AgentResponse:
    reply: str | None
    status_code: int
    latency_ms: float
    raw_body: str


class SimulationAgentCaller:
    """Wraps the base AgentCaller with mock injection and a uniform AgentResponse shape.

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
        self._caller = _BaseAgentCaller(project_config, timeout_s=float(timeout_s))
        self._mock = mock if mock is not None else USE_MOCK_AGENT

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
