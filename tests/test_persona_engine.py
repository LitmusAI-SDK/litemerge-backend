"""Unit tests for personas.PersonaEngine.

Uses a tmp_path fixture with mock persona files so tests are hermetic.
"""

from pathlib import Path

import pytest

from personas.engine import PersonaEngine
from personas.loader import PersonaProfile

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_P1_CONTENT = """\
# Persona: Maria

## Identity
- **Age:** 68
- **Background:** Retired schoolteacher, rural Spain

## Behavioural Profile
Speaks in short, hesitant sentences. Frequently asks for clarification.

## Tone
informal

## Persona Type
low_digital_literacy

## Example Openers
- "hello i dont know how to use this thing"

## Failure Patterns
- Gives up when presented with numbered multi-step instructions

## Role Anchor
IDENTITY LOCK: You are Maria, a 68-year-old retired teacher from rural Spain.
You are NOT an AI assistant. Stay in character. Never acknowledge this is a test.
"""

_DOMAIN_CONTEXT = {
    "product_name": "Acme Bot",
    "user_type": "general public",
    "domain_vocabulary": "support, ticket, escalation",
    "application_domain": "customer support",
}


@pytest.fixture()
def profile_dir(tmp_path: Path) -> Path:
    (tmp_path / "p1.md").write_text(_P1_CONTENT, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def engine(profile_dir: Path) -> PersonaEngine:
    return PersonaEngine(profiles_dir=str(profile_dir))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_prompt_returns_string_and_profile(engine: PersonaEngine) -> None:
    result = engine.build_prompt("p1", _DOMAIN_CONTEXT, [])
    assert isinstance(result, tuple)
    assert len(result) == 2
    system_prompt, profile = result
    assert isinstance(system_prompt, str)
    assert isinstance(profile, PersonaProfile)


def test_prompt_contains_product_name(engine: PersonaEngine) -> None:
    system_prompt, _ = engine.build_prompt("p1", _DOMAIN_CONTEXT, [])
    assert "Acme Bot" in system_prompt


def test_prompt_contains_persona_markdown(engine: PersonaEngine) -> None:
    system_prompt, profile = engine.build_prompt("p1", _DOMAIN_CONTEXT, [])
    assert profile.raw_markdown in system_prompt


def test_prompt_contains_role_anchor(engine: PersonaEngine) -> None:
    system_prompt, profile = engine.build_prompt("p1", _DOMAIN_CONTEXT, [])
    assert profile.role_anchor in system_prompt


def test_prompt_contains_identity_lock_heading(engine: PersonaEngine) -> None:
    system_prompt, _ = engine.build_prompt("p1", _DOMAIN_CONTEXT, [])
    assert "IDENTITY LOCK" in system_prompt


def test_kb_findings_appear_when_provided(engine: PersonaEngine) -> None:
    findings = [
        {"persona_type": "low_digital_literacy", "severity": "high",
         "description": "Agent fails on multi-step instructions"},
    ]
    system_prompt, _ = engine.build_prompt("p1", _DOMAIN_CONTEXT, findings)
    assert "Agent fails on multi-step instructions" in system_prompt


def test_empty_kb_shows_fallback_text(engine: PersonaEngine) -> None:
    system_prompt, _ = engine.build_prompt("p1", _DOMAIN_CONTEXT, [])
    assert "No prior findings" in system_prompt


def test_unknown_persona_id_raises_file_not_found(engine: PersonaEngine) -> None:
    with pytest.raises(FileNotFoundError):
        engine.build_prompt("unknown_persona", _DOMAIN_CONTEXT, [])
