"""Per-provider cache annotation logic.

The cacheable static block is always messages[0] (the system prompt). It is
identical on every turn within a session, so it is cacheable from turn 1.

Each provider uses a different mechanism; all branching lives here and
nowhere else in the codebase.
"""

import copy
from typing import Literal

Provider = Literal["anthropic", "gemini", "openai", "lmstudio"]


def apply_cache(messages: list[dict], provider: Provider) -> list[dict]:
    """Annotate messages with provider-specific cache control markers.

    Returns a new list (never mutates input). The cacheable block is
    messages[0] (system prompt). History turns are never annotated —
    they grow each turn and caching them would cause cache misses.
    """
    messages = copy.deepcopy(messages)

    match provider:
        case "anthropic":
            return _apply_anthropic_cache(messages)
        case "gemini":
            return _apply_gemini_cache(messages)
        case "openai":
            # OpenAI prompt caching is automatic — no annotation needed.
            # Prefix stability is guaranteed because system prompt is always
            # first and never changes within a session.
            return messages
        case "lmstudio":
            # Local model via LM Studio — no caching layer.
            return messages
        case _:
            return messages


def _apply_anthropic_cache(messages: list[dict]) -> list[dict]:
    """Anthropic requires cache_control: {"type": "ephemeral"} on the last
    content block of the system message to cache everything up to that point.

    Minimum cacheable token count: 1024.
    The annotation must appear on every API call; Anthropic does not persist
    the marker across requests.
    """
    system_msg = messages[0]
    if system_msg.get("role") != "system":
        return messages

    content = system_msg["content"]

    if isinstance(content, str):
        # Wrap plain string in block format Anthropic expects
        system_msg["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    elif isinstance(content, list) and content:
        # Attach cache_control to the last block only
        content[-1]["cache_control"] = {"type": "ephemeral"}

    return messages


def _apply_gemini_cache(messages: list[dict]) -> list[dict]:
    """Gemini context caching is a separate resource lifecycle managed by
    GeminiCacheManager. apply_cache() only marks the system message with a
    hint so the caller knows it is the cache target.

    The actual CachedContent resource name is passed into agenerate() via
    session_meta["gemini_cache_name"] and injected as cached_content= in
    the litellm call kwargs.
    """
    system_msg = messages[0]
    if system_msg.get("role") == "system":
        system_msg["_gemini_cache_hint"] = True
    return messages
