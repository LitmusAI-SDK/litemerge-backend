"""PersonaLoader — reads and parses persona profile .md files.

File format contract:
  # Persona: <Name>

  ## Identity
  - **Key:** Value

  ## Behavioural Profile
  <free text>

  ## Tone
  <single word/phrase>

  ## Persona Type
  <slug e.g. low_digital_literacy>

  ## Example Openers
  - "opener text"

  ## Failure Patterns
  - pattern description

  ## Role Anchor
  IDENTITY LOCK: ...
"""

import re
from dataclasses import dataclass
from pathlib import Path

from core.config import settings

_REQUIRED_SECTIONS = [
    "Identity",
    "Behavioural Profile",
    "Tone",
    "Persona Type",
    "Example Openers",
    "Failure Patterns",
    "Role Anchor",
]


@dataclass
class PersonaProfile:
    persona_id: str
    name: str
    identity: dict            # raw key-value pairs from ## Identity section
    behavioral_profile: str   # full text of ## Behavioural Profile section
    tone: str
    persona_type: str         # slug e.g. "low_digital_literacy", "adversarial"
    example_openers: list[str]
    failure_patterns: list[str]
    role_anchor: str
    raw_markdown: str         # full original file content (injected into master prompt)


class PersonaLoader:
    """Reads and parses a persona profile .md file into a PersonaProfile.

    Args:
        profiles_dir: Override the default profiles directory (useful in tests).
    """

    def __init__(self, profiles_dir: str | None = None) -> None:
        self._dir = Path(profiles_dir or settings.persona_profiles_dir)

    def load(self, persona_id: str) -> PersonaProfile:
        """Read and parse personas/{persona_id}.md.

        Called once per session at startup — not cached at module level.

        Raises:
            FileNotFoundError: if the .md file does not exist.
            ValueError: if any required ## section is missing or malformed.
        """
        path = self._dir / f"{persona_id}.md"
        if not path.exists():
            raise FileNotFoundError(f"Persona profile not found: {path}")

        raw = path.read_text(encoding="utf-8")
        return self._parse(persona_id, raw)

    # ------------------------------------------------------------------
    # Internal parser
    # ------------------------------------------------------------------

    def _parse(self, persona_id: str, raw: str) -> PersonaProfile:
        # Extract name from top-level heading
        name_match = re.search(r"^#\s+Persona:\s+(.+)$", raw, re.MULTILINE)
        if not name_match:
            raise ValueError(
                f"Missing '# Persona: Name' header in {persona_id}.md"
            )
        name = name_match.group(1).strip()

        # Split into sections by ## headings
        sections: dict[str, str] = {}
        current_heading: str | None = None
        current_lines: list[str] = []

        for line in raw.split("\n"):
            if line.startswith("## "):
                if current_heading is not None:
                    sections[current_heading] = "\n".join(current_lines).strip()
                current_heading = line[3:].strip()
                current_lines = []
            elif current_heading is not None:
                current_lines.append(line)

        if current_heading is not None:
            sections[current_heading] = "\n".join(current_lines).strip()

        # Validate required sections
        missing = [s for s in _REQUIRED_SECTIONS if s not in sections]
        if missing:
            raise ValueError(
                f"Missing required sections in {persona_id}.md: {missing}"
            )

        return PersonaProfile(
            persona_id=persona_id,
            name=name,
            identity=self._parse_identity(sections["Identity"]),
            behavioral_profile=sections["Behavioural Profile"],
            tone=self._first_line(sections["Tone"]),
            persona_type=self._first_line(sections["Persona Type"]),
            example_openers=self._parse_bullet_list(sections["Example Openers"]),
            failure_patterns=self._parse_bullet_list(sections["Failure Patterns"]),
            role_anchor=sections["Role Anchor"],
            raw_markdown=raw,
        )

    @staticmethod
    def _first_line(text: str) -> str:
        for line in text.split("\n"):
            line = line.strip()
            if line:
                return line
        return ""

    @staticmethod
    def _parse_identity(text: str) -> dict:
        identity: dict[str, str] = {}
        for line in text.split("\n"):
            match = re.match(r"-\s+\*\*(.+?):\*\*\s+(.+)", line)
            if match:
                identity[match.group(1).strip()] = match.group(2).strip()
        return identity

    @staticmethod
    def _parse_bullet_list(text: str) -> list[str]:
        items: list[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                item = line[2:].strip().strip('"')
                if item:
                    items.append(item)
        return items
