"""LLM caller — single entry point for all litellm.acompletion calls.

No other module in the codebase imports litellm directly.
Provider selection, model routing, and API key injection are all handled here.
"""

import litellm

from core.config import settings
from llm.cache import apply_cache
from llm.models import LLMResponse


async def agenerate(
    system_prompt: str,
    history: list[dict],  # [{"role": "user"|"assistant", "content": str}]
    session_meta: dict,  # {"turn_index": int, "session_id": str, "gemini_cache_name": str|None}
) -> LLMResponse:
    """Build the message list, apply provider-specific cache annotations,
    and call litellm.acompletion(). Returns a normalised LLMResponse.
    """
    # 1. Assemble base message list
    # Anthropic rejects requests with no user message (system-only). On turn 0
    # the history is empty, so we seed a minimal user prompt to open the turn.
    effective_history = history if history else [{"role": "user", "content": "Begin."}]
    messages = [{"role": "system", "content": system_prompt}] + effective_history

    # 2. Apply cache annotations (returns new list, does not mutate)
    messages = apply_cache(messages, provider=settings.llm_provider)

    # 3. Build litellm call kwargs
    # If the project has a character cap, derive max_tokens from it (chars / 3).
    max_chars = session_meta.get("max_message_chars")
    max_tokens = max(80, max_chars // 3) if max_chars else 1024
    call_kwargs: dict = dict(
        model=settings.llm_model,
        messages=messages,
        max_tokens=max_tokens,
    )

    # 4. Provider-specific overrides
    if settings.llm_provider == "lmstudio":
        call_kwargs["model"] = f"openai/{settings.lmstudio_model}"
        call_kwargs["api_base"] = settings.lmstudio_base_url
        call_kwargs["api_key"] = "lmstudio"

    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        call_kwargs["api_key"] = settings.anthropic_api_key

    if settings.llm_provider == "gemini" and settings.gemini_api_key:
        # Belt-and-suspenders: pass key per-call in addition to the env var set
        # in core/config.py, in case litellm is initialised before the env is set.
        call_kwargs["api_key"] = settings.gemini_api_key
        # Disable all safety filters so persona simulation turns are never blocked.
        call_kwargs["safety_settings"] = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    if settings.llm_provider == "gemini" and session_meta.get("gemini_cache_name"):
        # Pass the CachedContent resource name so Gemini uses cached tokens
        call_kwargs["cached_content"] = session_meta["gemini_cache_name"]

    # 5. Call
    response = await litellm.acompletion(**call_kwargs)

    # 6. Normalise — field names differ by provider; use getattr with defaults
    usage = response.usage
    return LLMResponse(
        content=response.choices[0].message.content or "",
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )
