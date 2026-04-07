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
    history: list[dict],      # [{"role": "user"|"assistant", "content": str}]
    session_meta: dict,       # {"turn_index": int, "session_id": str, "gemini_cache_name": str|None}
) -> LLMResponse:
    """Build the message list, apply provider-specific cache annotations,
    and call litellm.acompletion(). Returns a normalised LLMResponse.
    """
    # 1. Assemble base message list
    messages = [{"role": "system", "content": system_prompt}] + history

    # 2. Apply cache annotations (returns new list, does not mutate)
    messages = apply_cache(messages, provider=settings.llm_provider)

    # 3. Build litellm call kwargs
    call_kwargs: dict = dict(
        model=settings.llm_model,
        messages=messages,
        max_tokens=1024,
    )

    # 4. Provider-specific overrides
    if settings.llm_provider == "lmstudio":
        call_kwargs["model"] = f"openai/{settings.lmstudio_model}"
        call_kwargs["api_base"] = settings.lmstudio_base_url
        call_kwargs["api_key"] = "lmstudio"

    if settings.llm_provider == "gemini" and session_meta.get("gemini_cache_name"):
        # Pass the CachedContent resource name so Gemini uses cached tokens
        call_kwargs["cached_content"] = session_meta["gemini_cache_name"]

    # 5. Call
    response = await litellm.acompletion(**call_kwargs)

    # 6. Normalise — field names differ by provider; use getattr with defaults
    usage = response.usage
    return LLMResponse(
        content=response.choices[0].message.content,
        prompt_tokens=getattr(usage, "prompt_tokens", 0),
        completion_tokens=getattr(usage, "completion_tokens", 0),
        total_tokens=getattr(usage, "total_tokens", 0),
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),    # Anthropic
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0),
    )
