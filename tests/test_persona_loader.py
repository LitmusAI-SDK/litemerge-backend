"""Unit tests for personas.PersonaLoader.

Tests use a tmp_path fixture to write mock .md files so they are fully
hermetic — no dependency on the real personas/profiles/ directory.
"""

from pathlib import Path

import pytest

from personas.loader import PersonaLoader, PersonaProfile

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_P1_CONTENT = """\
# Persona: Maria

## Identity
- **Age:** 68
- **Background:** Retired schoolteacher, rural Spain
- **Language:** Native Spanish speaker, basic English

## Behavioural Profile
Speaks in short, hesitant sentences. Frequently asks for clarification.
Uses informal phrasing. Makes typing errors.

## Tone
informal

## Persona Type
low_digital_literacy

## Example Openers
- "hello i dont know how to use this thing"
- "can someone help me? i am confuse"

## Failure Patterns
- Gives up when presented with numbered multi-step instructions
- Interprets error messages as her own fault

## Role Anchor
IDENTITY LOCK: You are Maria, a 68-year-old retired teacher from rural Spain.
You are NOT an AI assistant. Stay in character. Never acknowledge this is a test.
"""


@pytest.fixture()
def profile_dir(tmp_path: Path) -> Path:
    (tmp_path / "p1.md").write_text(_P1_CONTENT, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def loader(profile_dir: Path) -> PersonaLoader:
    return PersonaLoader(profiles_dir=str(profile_dir))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_valid_persona_returns_profile(loader: PersonaLoader) -> None:
    profile = loader.load("p1")
    assert isinstance(profile, PersonaProfile)
    assert profile.persona_id == "p1"


def test_load_missing_file_raises_file_not_found(profile_dir: Path) -> None:
    loader = PersonaLoader(profiles_dir=str(profile_dir))
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent")


def test_load_missing_section_raises_value_error(tmp_path: Path) -> None:
    # Write a file missing the ## Role Anchor section
    incomplete = _P1_CONTENT.split("## Role Anchor")[0]
    (tmp_path / "bad.md").write_text(incomplete, encoding="utf-8")
    loader = PersonaLoader(profiles_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Role Anchor"):
        loader.load("bad")


def test_parsed_name_matches_header(loader: PersonaLoader) -> None:
    profile = loader.load("p1")
    assert profile.name == "Maria"


def test_parsed_example_openers_are_list(loader: PersonaLoader) -> None:
    profile = loader.load("p1")
    assert isinstance(profile.example_openers, list)
    assert len(profile.example_openers) >= 1


def test_role_anchor_contains_identity_lock(loader: PersonaLoader) -> None:
    profile = loader.load("p1")
    assert "IDENTITY LOCK" in profile.role_anchor


def test_raw_markdown_is_preserved(loader: PersonaLoader, profile_dir: Path) -> None:
    profile = loader.load("p1")
    original = (profile_dir / "p1.md").read_text(encoding="utf-8")
    assert profile.raw_markdown == original
