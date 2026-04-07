"""PersonaEngine — single entry point wiring loader → kb_filter → master prompt."""

from personas.kb_filter import filter_findings
from personas.loader import PersonaLoader, PersonaProfile
from personas.master_prompt import render


class PersonaEngine:
    """Orchestrates persona loading, KB filtering, and prompt rendering.

    Args:
        profiles_dir: Override the default profiles directory (useful in tests).
    """

    def __init__(self, profiles_dir: str | None = None) -> None:
        self._loader = PersonaLoader(profiles_dir=profiles_dir)

    def build_prompt(
        self,
        persona_id: str,
        domain_context: dict,
        raw_kb_findings: list[dict],  # unfiltered; engine filters internally
    ) -> tuple[str, PersonaProfile]:
        """Load the persona .md file, filter KB findings, and render the master prompt.

        Returns:
            (system_prompt_string, profile)
            Callers receive the profile to access metadata (name, persona_type,
            etc.) without re-parsing the file.

        Raises:
            FileNotFoundError: for unknown persona_id.
            ValueError: for malformed .md files.
        """
        profile = self._loader.load(persona_id)
        filtered = filter_findings(raw_kb_findings, profile.persona_type)
        system_prompt = render(profile, domain_context, filtered)
        return system_prompt, profile
