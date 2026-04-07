"""Manages Gemini CachedContent resources for per-session context caching.

Usage pattern in PersonaSession.run():

    manager = GeminiCacheManager(system_prompt, session_id)
    cache_name = await manager.get_or_create()   # before turn 1
    session_meta["gemini_cache_name"] = cache_name

    # ... run all turns ...

    await manager.delete()                        # after session ends or on error
"""

import httpx
from core.config import settings

GEMINI_CACHE_API = "https://generativelanguage.googleapis.com/v1beta/cachedContents"
TTL_SECONDS = 3600  # 1 hour; set to expected max session duration


class GeminiCacheManager:
    def __init__(self, system_prompt: str, session_id: str):
        self.system_prompt = system_prompt
        self.session_id = session_id
        self._cache_name: str | None = None

    async def get_or_create(self) -> str:
        """Create a CachedContent resource containing the system prompt.

        Returns the resource name used in subsequent generation calls.
        Idempotent: if already created, returns the existing name.
        """
        if self._cache_name:
            return self._cache_name

        # litellm model string is "gemini/gemini-1.5-flash"; strip prefix for REST API
        bare_model = settings.llm_model.replace("gemini/", "")

        payload = {
            "model": f"models/{bare_model}",
            "systemInstruction": {"parts": [{"text": self.system_prompt}]},
            "ttl": f"{TTL_SECONDS}s",
            "displayName": f"litmusai-{self.session_id}",
        }
        async with httpx.AsyncClient() as client:
            r = await client.post(
                GEMINI_CACHE_API,
                json=payload,
                params={"key": settings.gemini_api_key},
                timeout=15,
            )
            r.raise_for_status()
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
