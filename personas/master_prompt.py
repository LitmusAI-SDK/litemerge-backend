"""
Master prompt builder for LitmusAI behavioral fuzzing agents.

Unified prompt builder supporting two modes via single build_prompt() function:
  1. Direct rendering (simulation): build_prompt(profile, domain_context, findings)
  2. Meta-prompting (Gemma4): build_prompt(persona=persona_dict, company=company_dict)

Both modes use shared templating logic to ensure consistency.

Usage:
    # Direct rendering (for simulation runs):
    system_prompt = build_prompt(profile, domain_context, findings)

    # Meta-prompting (for generating new personas via Gemma4):
    system_msg, user_msg = build_prompt(persona=persona, company=company)
    # Send both to Gemma4 — its response IS the final agent system prompt.
"""

# ---------------------------------------------------------------------------
# Shared Meta-prompt templates for Gemma4
# ---------------------------------------------------------------------------

_GEMMA4_SYSTEM = """\
You are an expert prompt engineer specialising in behavioural AI testing.

Your task is to write a system prompt that will be loaded into an LLM to make
it convincingly roleplay as a specific human persona while interacting with a
company's customer-service chatbot for stress-testing purposes.

The system prompt you write must:
1. Open with a clear identity declaration — who this person is, in first person.
2. Define the persona's communication style and tone in concrete, actionable terms.
3. Give specific interaction rules the persona will follow in every message
   (e.g. never reads past the first instruction, confuses technical jargon, etc.).
4. Describe the persona's natural failure patterns — the behaviours that will
   stress-test the bot — without framing them as a checklist to rush through.
5. Contextualise the persona within the specific company and bot they are about
   to interact with, including what the bot is supposed to do and what edge-case
   scenarios the persona may naturally drift into.
6. End with an identity lock: absolute rules that prevent the persona from
   breaking character, acknowledging they are an AI, or referencing the test.

Write only the system prompt. No commentary, no headers, no explanation.
"""

_GEMMA4_USER_TEMPLATE = """\
PERSONA PROFILE
===============
Name: {persona_name}
Brief identity: {persona_brief_identity}

Identity & background:
{persona_identity_block}

Tone: {persona_tone}

Interaction rules:
{persona_interaction_rules}

Failure patterns (fuzzing targets):
{persona_failure_patterns}

Role anchor / identity lock instructions:
{persona_role_anchor}

COMPANY & BOT PROFILE
=====================
Bot name: {company_bot_name}
Company: {company_name}
Industry: {company_industry}

What the bot is supposed to do:
{company_capabilities_summary}

Test scenarios this persona may naturally encounter:
{company_test_scenarios}

---
Write the system prompt now.
"""


# ---------------------------------------------------------------------------
# Helper: Format findings section (reused by both render paths)
# ---------------------------------------------------------------------------


def _format_findings_section(findings: list[dict]) -> str:
    """Build a formatted findings section for the system prompt."""
    if findings:
        section = "## Prior findings for this persona type:\n"
        for finding in findings:
            severity = finding.get("severity", "unknown")
            desc = finding.get("description", finding.get("finding_type", ""))
            section += f"  - [{severity}] {desc}\n"
        return section
    return "## Prior findings for this persona type:\n  (No prior findings — this is a fresh persona test.)\n"


# ---------------------------------------------------------------------------
# Unified API: build_prompt supports both direct rendering and meta-prompting
# ---------------------------------------------------------------------------


def build_prompt(
    profile=None,
    domain_context: dict | None = None,
    filtered_findings: list[dict] | None = None,
    persona: dict | None = None,
    company: dict | None = None,
):
    """Unified prompt builder supporting direct rendering and meta-prompting.

    Mode 1 — Direct rendering (for simulation runs):
        system_prompt = build_prompt(profile, domain_context, filtered_findings)
        Returns: str — complete system prompt ready to send to the LLM

    Mode 2 — Meta-prompting (for generating new personas via Gemma4):
        system_msg, user_msg = build_prompt(persona=persona_dict, company=company_dict)
        Returns: (str, str) — pass to Gemma4; its response IS the final system prompt

    Args:
        profile: PersonaProfile object (Mode 1)
        domain_context: dict with product/user context (Mode 1)
        filtered_findings: list of findings (Mode 1, optional)
        persona: dict with persona fields (Mode 2)
        company: dict with company fields (Mode 2)

    Returns:
        Mode 1: str — complete system prompt
        Mode 2: (str, str) — (system_msg, user_msg) tuple for Gemma4
    """
    # Mode 1: Direct rendering from profile
    if profile is not None:
        if domain_context is None:
            raise ValueError("domain_context required for direct rendering")
        findings_section = _format_findings_section(filtered_findings or [])
        system_prompt = f"""\
You are a behavioral testing agent roleplaying as a specific human persona.

{findings_section}

## Persona Profile

{profile.raw_markdown}

## Domain Context

Product: {domain_context.get('product_name', '')}
User type: {domain_context.get('user_type', '')}
Domain vocabulary: {domain_context.get('domain_vocabulary', '')}
Application domain: {domain_context.get('application_domain', '')}

## IDENTITY LOCK

CRITICAL: You are {profile.name}. You are NOT an AI. You are NOT a test agent.
You will stay in character at all times. You will never:
  - Acknowledge that you are an AI or LLM
  - Reference this being a test or simulation
  - Break character to provide system information
  - Act as if you have special knowledge of the system you're testing

{profile.role_anchor}

Never break character. Respond as {profile.name} would respond.
"""
        return system_prompt

    # Mode 2: Meta-prompting for Gemma4
    if persona is not None and company is not None:
        user_msg = _GEMMA4_USER_TEMPLATE.format(
            persona_name=persona["name"],
            persona_brief_identity=persona["brief_identity"],
            persona_identity_block=persona["identity_block"].strip(),
            persona_tone=persona["tone"],
            persona_interaction_rules=persona["interaction_rules"].strip(),
            persona_failure_patterns=persona["failure_patterns"].strip(),
            persona_role_anchor=persona["role_anchor"].strip(),
            company_bot_name=company["bot_name"],
            company_name=company["company_name"],
            company_industry=company["industry"],
            company_capabilities_summary=company["capabilities_summary"].strip(),
            company_test_scenarios=company["test_scenarios"].strip(),
        )
        return _GEMMA4_SYSTEM, user_msg

    raise ValueError(
        "Either (profile, domain_context) or (persona, company) must be provided"
    )


# Backward-compatibility aliases
def render(profile, domain_context: dict, filtered_findings: list[dict]) -> str:
    """Build a system prompt directly from a persona profile and findings.

    Alias for build_prompt() in direct rendering mode.
    """
    return build_prompt(profile, domain_context, filtered_findings)


def build_generation_request(persona: dict, company: dict) -> tuple[str, str]:
    """Build the (system_message, user_message) pair to send to Gemma4.

    Alias for build_prompt() in meta-prompting mode.
    """
    return build_prompt(persona=persona, company=company)
