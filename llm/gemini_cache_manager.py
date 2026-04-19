"""Manages Gemini CachedContent resources for per-session context caching.

Usage pattern in PersonaSession.run():

    manager = GeminiCacheManager(system_prompt, session_id)
    cache_name = await manager.get_or_create()   # None if prompt too short
    session_meta["gemini_cache_name"] = cache_name

    # ... run all turns ...

    await manager.delete()                        # no-op if cache_name is None
"""

import asyncio
import logging
import random

import httpx
from core.config import settings

logger = logging.getLogger(__name__)

# Gemini requires at least 32 768 tokens to create a CachedContent resource.
# Estimate: 1 token ≈ 4 chars.  Skip caching for prompts below this threshold.
_MIN_CACHE_CHARS = 32_768 * 4

GEMINI_CACHE_API = "https://generativelanguage.googleapis.com/v1beta/cachedContents"
TTL_SECONDS = 3600  # 1 hour; set to expected max session duration

_RETRY_DELAYS = (5.0, 15.0, 45.0)  # backoff steps on 429


class GeminiCacheManager:
    def __init__(self, system_prompt: str, session_id: str):
        self.system_prompt = system_prompt
        self.session_id = session_id
        self._cache_name: str | None = None

    async def get_or_create(self) -> str | None:
        """Create a CachedContent resource containing the system prompt.

        Returns the resource name, or None if the prompt is too short to cache
        (Gemini requires ≥32 768 tokens).  Callers must handle None gracefully.
        Idempotent: if already created, returns the existing name.
        """
        if self._cache_name:
            return self._cache_name

        if len(self.system_prompt) < _MIN_CACHE_CHARS:
            logger.debug(
                "Session %s: system prompt too short for Gemini caching "
                "(%d chars < %d threshold) — skipping cache",
                self.session_id,
                len(self.system_prompt),
                _MIN_CACHE_CHARS,
            )
            return None

        # litellm model string is "gemini/gemini-1.5-flash"; strip prefix for REST API
        bare_model = settings.llm_model.replace("gemini/", "")

        payload = {
            "model": f"models/{bare_model}",
            "systemInstruction": {"parts": [{"text": self.system_prompt}]},
            "ttl": f"{TTL_SECONDS}s",
            "displayName": f"litmusai-{self.session_id}",
        }
        # Jitter on first attempt so parallel sessions don't all fire at once
        await asyncio.sleep(random.uniform(0, 2.0))

        for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    GEMINI_CACHE_API,
                    json=payload,
                    params={"key": settings.gemini_api_key},
                    timeout=15,
                )
            if r.status_code == 429 and delay is not None:
                retry_after = r.headers.get("Retry-After")
                wait = (
                    float(retry_after)
                    if retry_after
                    else delay + random.uniform(0, 3.0)
                )
                await asyncio.sleep(wait)
                continue
            if r.status_code == 400:
                logger.warning(
                    "Session %s: Gemini cache creation returned 400 — "
                    "running without cache. Response: %s",
                    self.session_id,
                    r.text[:400],
                )
                return None
            r.raise_for_status()
            self._cache_name = r.json()["name"]
            return self._cache_name

        # Final attempt after all retries exhausted
        r.raise_for_status()  # type: ignore[possibly-undefined]
        self._cache_name = r.json()["name"]
        return self._cache_name

    async def delete(self) -> None:
        """Delete the CachedContent resource after the session ends.

        Should be called in a finally block to avoid orphaned billable resources.
        """
        if not self._cache_name:
            return
        async with httpx.AsyncClient() as client:
            await client.delete(
                f"{GEMINI_CACHE_API}/{self._cache_name}",
                params={"key": settings.gemini_api_key},
                timeout=10,
            )
        self._cache_name = None
