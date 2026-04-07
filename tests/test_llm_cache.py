"""Unit tests for llm.cache.apply_cache().

All tests are pure — no LLM calls, no network, no external deps.
"""

import copy

import pytest

from llm.cache import apply_cache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _messages(system_content="You are a helpful assistant."):
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "Hello"},
    ]


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def test_anthropic_wraps_system_content_in_block() -> None:
    messages = apply_cache(_messages(), provider="anthropic")
    system_content = messages[0]["content"]
    assert isinstance(system_content, list)
    assert system_content[0]["type"] == "text"


def test_anthropic_cache_control_is_ephemeral() -> None:
    messages = apply_cache(_messages(), provider="anthropic")
    last_block = messages[0]["content"][-1]
    assert last_block["cache_control"] == {"type": "ephemeral"}


def test_anthropic_does_not_mutate_input() -> None:
    original = _messages()
    original_copy = copy.deepcopy(original)
    apply_cache(original, provider="anthropic")
    assert original == original_copy


def test_anthropic_existing_list_content_gets_cache_control() -> None:
    """If system content is already a list, cache_control is added to last block."""
    msgs = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "block one"}, {"type": "text", "text": "block two"}],
        },
        {"role": "user", "content": "Hi"},
    ]
    result = apply_cache(msgs, provider="anthropic")
    blocks = result[0]["content"]
    assert "cache_control" not in blocks[0]
    assert blocks[-1]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def test_gemini_hint_added_to_system_message() -> None:
    messages = apply_cache(_messages(), provider="gemini")
    assert messages[0].get("_gemini_cache_hint") is True


def test_gemini_does_not_mutate_input() -> None:
    original = _messages()
    original_copy = copy.deepcopy(original)
    apply_cache(original, provider="gemini")
    assert original == original_copy


# ---------------------------------------------------------------------------
# OpenAI / LM Studio / unknown
# ---------------------------------------------------------------------------

def test_openai_messages_returned_unchanged() -> None:
    original = _messages()
    result = apply_cache(original, provider="openai")
    # Content should not be modified (no cache_control, no hints)
    assert result[0]["content"] == original[0]["content"]


def test_lmstudio_messages_returned_unchanged() -> None:
    original = _messages()
    result = apply_cache(original, provider="lmstudio")
    assert result[0]["content"] == original[0]["content"]


def test_unknown_provider_no_crash() -> None:
    original = _messages()
    result = apply_cache(original, provider="completely_unknown")  # type: ignore[arg-type]
    assert result[0]["content"] == original[0]["content"]
