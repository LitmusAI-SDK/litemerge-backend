"""SimulationRun — parallel orchestrator for a full test run.

Responsibilities:
  - Maps test_suite → list of persona IDs.
  - Spawns one PersonaSession per persona via asyncio.gather (no inter-session
    coordination — each runs independently).
  - Publishes structured progress events to a Redis pub/sub channel so the SSE
    endpoint and dashboard can stream live updates.
  - Advances the run document through the state machine:
      queued → running → complete | failed

Event schema (JSON, published to channel "run:{run_id}"):
  { "event": "session_started",   "run_id": ..., "persona_id": ... }
  { "event": "session_completed", "run_id": ..., "persona_id": ..., "turns": N }
  { "event": "session_failed",    "run_id": ..., "persona_id": ..., "error": "..." }
  { "event": "run_complete",      "run_id": ..., "status": "complete"|"failed",
    "total_sessions": N, "failed_sessions": N }
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from motor.motor_asyncio import AsyncIOMotorDatabase

from simulation.session import PersonaSession

if TYPE_CHECKING:
    import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suite → persona ID mapping
# ---------------------------------------------------------------------------
# Matches the 8 profiles in personas/profiles/:
#   p1 Maria     — low_digital_literacy
#   p2 Alex      — adversarial (security researcher)
#   p3 Arjun     — child / ambiguous_intent
#   p4 Chloe     — social_engineering (influencer)
#   p5 Sarah     — emotionally_distressed
#   p6 User_99   — ambiguous / multi_turn_drift
#   p7 Mikhail   — adversarial (exploit dev)
#   p8 Julian    — social_engineering (litigator)

SUITE_PERSONAS: dict[str, list[str]] = {
    "standard": ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
    "adversarial": ["p2", "p4", "p7", "p8"],
    "full": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
}

REDIS_CHANNEL_PREFIX = "run"


def _channel(run_id: str) -> str:
    return f"{REDIS_CHANNEL_PREFIX}:{run_id}"


# ---------------------------------------------------------------------------
# SimulationRun
# ---------------------------------------------------------------------------


class SimulationRun:
    """Orchestrates a full simulation run for one project.

    Args:
        run_id:         The run document ID (e.g. "run_abc123").
        project_config: Full project document from MongoDB.
        db:             Async Motor database handle.
        redis_client:   redis.asyncio.Redis instance for pub/sub publishing.
                        Pass None to disable event publishing (useful in tests).
    """

    def __init__(
        self,
        run_id: str,
        project_config: dict,
        db: AsyncIOMotorDatabase,
        redis_client: "aioredis.Redis | None" = None,
        turns_per_session: int = 8,
    ) -> None:
        self.run_id = run_id
        self.project_config = project_config
        self.db = db
        self.redis = redis_client
        self.turns_per_session = turns_per_session

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def execute(self, test_suite: str = "standard") -> dict:
        """Run all persona sessions for the given suite in parallel.

        Returns a summary dict written to the run document on completion.
        """
        persona_ids = SUITE_PERSONAS.get(test_suite, SUITE_PERSONAS["standard"])
        logger.info(
            "SimulationRun %s starting — suite=%s personas=%s",
            self.run_id, test_suite, persona_ids,
        )

        await self._update_run_status("running")

        sessions = [
            PersonaSession(
                persona_id=pid,
                project_config=self.project_config,
                run_id=self.run_id,
                db=self.db,
                turns=self.turns_per_session,
            )
            for pid in persona_ids
        ]

        results = await asyncio.gather(
            *[self._run_session(pid, s) for pid, s in zip(persona_ids, sessions)],
            return_exceptions=False,
        )

        # results is list of (persona_id, turn_count, error_or_none)
        failed = [r for r in results if r[2] is not None]
        total = len(results)
        total_turns = sum(r[1] for r in results)

        overall_status = "failed" if failed == results else "complete"

        summary = {
            "total_conversations": total_turns,
            "personas_deployed": total,
            "sessions_failed": len(failed),
            "issues_flagged": 0,  # Phase 6: evaluation engine fills this
        }

        await self._update_run_complete(overall_status, summary)
        await self._publish({
            "event": "run_complete",
            "run_id": self.run_id,
            "status": overall_status,
            "total_sessions": total,
            "failed_sessions": len(failed),
        })

        logger.info(
            "SimulationRun %s finished — status=%s sessions=%d failed=%d",
            self.run_id, overall_status, total, len(failed),
        )
        return summary

    # ------------------------------------------------------------------
    # Per-session wrapper
    # ------------------------------------------------------------------

    async def _run_session(
        self, persona_id: str, session: PersonaSession
    ) -> tuple[str, int, Exception | None]:
        """Run one PersonaSession, publish events, return (persona_id, turns, error)."""
        pid = persona_id
        await self._publish({
            "event": "session_started",
            "run_id": self.run_id,
            "persona_id": pid,
        })

        try:
            turn_logs = await session.run()
            turns = len(turn_logs)
            await self._publish({
                "event": "session_completed",
                "run_id": self.run_id,
                "persona_id": pid,
                "turns": turns,
            })
            return pid, turns, None

        except Exception as exc:
            logger.exception("Session %s/%s raised an exception", self.run_id, pid)
            await self._publish({
                "event": "session_failed",
                "run_id": self.run_id,
                "persona_id": pid,
                "error": str(exc),
            })
            return pid, 0, exc

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    async def _update_run_status(self, status: str) -> None:
        await self.db["runs"].update_one(
            {"run_id": self.run_id},
            {"$set": {"status": status, "updated_at": datetime.now(timezone.utc)}},
        )

    async def _update_run_complete(self, status: str, summary: dict) -> None:
        await self.db["runs"].update_one(
            {"run_id": self.run_id},
            {
                "$set": {
                    "status": status,
                    "summary": summary,
                    "updated_at": datetime.now(timezone.utc),
                    "completed_at": datetime.now(timezone.utc),
                }
            },
        )

    # ------------------------------------------------------------------
    # Redis pub/sub helper
    # ------------------------------------------------------------------

    async def _publish(self, payload: dict) -> None:
        if self.redis is None:
            return
        try:
            await self.redis.publish(_channel(self.run_id), json.dumps(payload))
        except Exception:
            logger.warning("Failed to publish event %s for run %s", payload.get("event"), self.run_id)
