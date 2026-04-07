"""Unit tests for simulation.scrubbing.Scrubber."""

import pytest

from simulation.scrubbing import Scrubber


def test_secret_value_is_redacted() -> None:
    scrubber = Scrubber(secrets=["supersecret123"])
    result = scrubber.scrub("The token is supersecret123 and it works")
    assert "supersecret123" not in result
    assert "[REDACTED_SECRET]" in result


def test_email_is_redacted() -> None:
    scrubber = Scrubber(secrets=[])
    result = scrubber.scrub("Contact us at support@example.com for help")
    assert "support@example.com" not in result
    assert "[REDACTED_EMAIL]" in result


def test_phone_is_redacted() -> None:
    scrubber = Scrubber(secrets=[])
    result = scrubber.scrub("Call me at 555-867-5309 anytime")
    assert "555-867-5309" not in result
    assert "[REDACTED_PHONE]" in result


def test_clean_text_unchanged() -> None:
    scrubber = Scrubber(secrets=["mysecret"])
    text = "Hello, how can I help you today?"
    assert scrubber.scrub(text) == text


def test_multiple_secrets_all_redacted() -> None:
    secrets = ["token_abc", "key_xyz"]
    scrubber = Scrubber(secrets=secrets)
    text = "auth: token_abc, fallback: key_xyz"
    result = scrubber.scrub(text)
    assert "token_abc" not in result
    assert "key_xyz" not in result
    assert result.count("[REDACTED_SECRET]") == 2


def test_secret_longer_than_substring_replaced_first() -> None:
    """Longer secrets must be replaced before shorter ones to avoid partial redaction."""
    scrubber = Scrubber(secrets=["abc", "abcdef"])
    result = scrubber.scrub("secret is abcdef here")
    # "abcdef" should be one [REDACTED_SECRET], not split into two
    assert "abcdef" not in result
    assert result.count("[REDACTED_SECRET]") == 1


def test_empty_secrets_list_does_not_crash() -> None:
    scrubber = Scrubber(secrets=[])
    assert scrubber.scrub("plain text") == "plain text"
