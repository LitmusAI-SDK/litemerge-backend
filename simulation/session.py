"""PersonaSession — main simulation loop with checkpoint/resume and resilience.

Design notes:
  - Standalone module-level retry helpers avoid re-creating decorated functions
    on every loop iteration.
  - Each turn is persisted immediately after completion so partial runs are
    fully visible and resumable.
  - Gemini CachedContent resources are always cleaned up in a finally block.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random,
)

from core.config import settings
from llm.caller import agenerate
from llm.models import LLMResponse
from personas.engine import PersonaEngine
from personas.loader import PersonaProfile
from simulation.agent_caller import AgentResponse, AgentRetriableError, SimulationAgentCaller
from simulation.scrubbing import Scrubber

# ---------------------------------------------------------------------------
# Standalone retried helpers (module-level — decorated once at import time)
# ---------------------------------------------------------------------------

@retry(
    wait=wait_random(min=settings.retry_min_wait_s, max=settings.retry_max_wait_s),
    stop=stop_after_attempt(settings.retry_max_attempts),
    reraise=True,
)
async def _retried_llm_call(
    system_prompt: str,
    history: list[dict],
    session_meta: dict,
) -> LLMResponse:
    return await agenerate(system_prompt, history, session_meta)


@retry(
    wait=wait_random(min=settings.retry_min_wait_s, max=settings.retry_max_wait_s),
    stop=stop_after_attempt(settings.retry_max_attempts),
    retry=retry_if_exception_type(AgentRetriableError),
    reraise=True,
)
async def _retried_agent_call(
    sim_caller: SimulationAgentCaller,
    message: str,
    session_id: str,
    history: list[dict],
) -> AgentResponse:
    resp = await sim_caller.send(message, session_id, history)
    if resp.reply is None:
        raise AgentRetriableError(
            f"Agent returned non-200 response (status {resp.status_code})"
        )
    return resp


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TurnLog:
    turn_index: int
    persona_turn: str
    agent_response: str | None
    latency_ms: float
    created_at: datetime


@dataclass
class PersonaSession:
    persona_id: str           # e.g. "p1" → loads personas/profiles/p1.md
    project_config: dict      # full project document from MongoDB
    run_id: str
    db: AsyncIOMotorDatabase
    turns: int = 8            # default 8; configurable per run

    # Allow injecting a profiles_dir for tests
    _profiles_dir: str | None = field(default=None, repr=False)

    @property
    def session_id(self) -> str:
        return f"{self.run_id}_{self.persona_id}"

    # ------------------------------------------------------------------
    # Main execution entry point
    # ------------------------------------------------------------------

    async def run(self) -> list[TurnLog]:
        """Execute the full persona simulation loop.

        Steps:
          1. Checkpoint resume — skip already-completed turns.
          2. Build system prompt via PersonaEngine.
          3. Initialise Gemini cache if applicable.
          4. Conversation loop — LLM turn → agent call → persist → repeat.
          5. Cleanup Gemini cache in finally block.
        """
        # ── Step 1: Checkpoint resume ───────────────────────────────────────
        start_turn, history = await self._load_checkpoint()
        if start_turn is None:
            # Session already completed — return early (idempotent)
            return []

        # ── Step 2: Build system prompt ─────────────────────────────────────
        domain_context = {
            "product_name": self.project_config.get("name", ""),
            "user_type": self.project_config.get("user_type", ""),
            "domain_vocabulary": self.project_config.get("domain_vocabulary", ""),
            "application_domain": self.project_config.get("domain", ""),
        }
        engine = PersonaEngine(profiles_dir=self._profiles_dir)
        system_prompt, profile = engine.build_prompt(
            persona_id=self.persona_id,
            domain_context=domain_context,
            raw_kb_findings=[],  # Phase 5: replace with live findings
        )

        # ── Step 3: Initialise Gemini cache if applicable ───────────────────
        gemini_manager = None
        gemini_cache_name = None

        if settings.llm_provider == "gemini":
            from llm.gemini_cache_manager import GeminiCacheManager
            gemini_manager = GeminiCacheManager(system_prompt, self.session_id)
            gemini_cache_name = await gemini_manager.get_or_create()

        # ── Step 4: Initialise remaining dependencies ───────────────────────
        agent_caller = SimulationAgentCaller(self.project_config)
        scrubber = Scrubber(_extract_secrets(self.project_config))
        results: list[TurnLog] = []

        # ── Step 5: Conversation loop ───────────────────────────────────────
        try:
            for i in range(start_turn, self.turns):
                session_meta = {
                    "turn_index": i,
                    "session_id": self.session_id,
                    "gemini_cache_name": gemini_cache_name,
                }

                # LLM persona turn
                try:
                    llm_response = await _retried_llm_call(system_prompt, history, session_meta)
                except Exception:
                    await _mark_session(self.db, self.session_id, status="failed")
                    break

                persona_message = scrubber.scrub(llm_response.content)

                # Agent call
                t0 = time.monotonic()
                try:
                    agent_resp = await _retried_agent_call(
                        agent_caller, persona_message, self.session_id, history
                    )
                except (AgentRetriableError, RetryError):
                    latency_ms = (time.monotonic() - t0) * 1000
                    agent_resp = AgentResponse(
                        reply=None, status_code=0, latency_ms=latency_ms, raw_body=""
                    )
                latency_ms = (time.monotonic() - t0) * 1000

                # Scrub + update history
                clean_reply = scrubber.scrub(agent_resp.reply or "")
                history.append({"role": "user", "content": persona_message})
                history.append({"role": "assistant", "content": clean_reply})

                # Build TurnLog
                turn = TurnLog(
                    turn_index=i,
                    persona_turn=persona_message,
                    agent_response=clean_reply or None,
                    latency_ms=latency_ms,
                    created_at=datetime.now(timezone.utc),
                )
                results.append(turn)

                # Persist turn immediately (safe for partial runs)
                await _upsert_turn(
                    self.db, self.session_id, self.run_id,
                    self.project_config, profile, turn, llm_response,
                )

                # Early exit on agent failure
                if agent_resp.reply is None:
                    await _mark_session(self.db, self.session_id, status="failed")
                    break

            else:
                await _mark_session(self.db, self.session_id, status="completed")

        finally:
            if gemini_manager:
                await gemini_manager.delete()

        return results

    # ------------------------------------------------------------------
    # Checkpoint helper
    # ------------------------------------------------------------------

    async def _load_checkpoint(self) -> tuple[int | None, list[dict]]:
        """Return (start_turn, history) from any existing partial run.

        Returns (None, []) if the session is already completed (caller should exit).
        Returns (0, []) if no prior turns exist.
        """
        doc = await self.db["chat_logs"].find_one({"session_id": self.session_id})
        if doc is None:
            return 0, []

        if doc.get("status") == "completed":
            return None, []

        existing_turns: list[dict] = doc.get("turns", [])
        history: list[dict] = []
        for t in existing_turns:
            history.append({"role": "user", "content": t.get("persona_turn", "")})
            history.append({"role": "assistant", "content": t.get("agent_response") or ""})

        return len(existing_turns), history


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

async def _upsert_turn(
    db: AsyncIOMotorDatabase,
    session_id: str,
    run_id: str,
    project_config: dict,
    profile: PersonaProfile,
    turn: TurnLog,
    llm_response: LLMResponse,
) -> None:
    await db["chat_logs"].update_one(
        {"session_id": session_id},
        {
            "$set": {
                "run_id": run_id,
                "project_id": project_config.get("_id"),
                "persona_id": profile.persona_id,
                "persona_type": profile.persona_type,
                "persona_name": profile.name,
                "status": "in_progress",
            },
            "$push": {"turns": _turn_to_dict(turn)},
            "$inc": {
                "total_usage.prompt_tokens": llm_response.prompt_tokens,
                "total_usage.completion_tokens": llm_response.completion_tokens,
                "total_usage.total_tokens": llm_response.total_tokens,
                "total_usage.cache_read_tokens": llm_response.cache_read_tokens,
                "total_usage.cache_write_tokens": llm_response.cache_write_tokens,
            },
        },
        upsert=True,
    )


async def _mark_session(
    db: AsyncIOMotorDatabase,
    session_id: str,
    status: str,
) -> None:
    await db["chat_logs"].update_one(
        {"session_id": session_id},
        {"$set": {"status": status, "completed_at": datetime.now(timezone.utc)}},
    )


def _turn_to_dict(turn: TurnLog) -> dict:
    return {
        "turn_index": turn.turn_index,
        "persona_turn": turn.persona_turn,
        "agent_response": turn.agent_response,
        "latency_ms": turn.latency_ms,
        "created_at": turn.created_at,
    }


def _extract_secrets(project_config: dict) -> list[str]:
    """Decrypt all auth_config secret values for Scrubber construction."""
    auth_config = project_config.get("auth_config") or {}
    value_encrypted = auth_config.get("value_encrypted")
    if not value_encrypted:
        return []
    try:
        from core.crypto import decrypt_secret
        return [decrypt_secret(value_encrypted)]
    except Exception:
        return []
