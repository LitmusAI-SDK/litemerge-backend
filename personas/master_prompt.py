"""
Master prompt builder for LitmusAI behavioral fuzzing agents.

Takes a structured persona dict and a structured company dict and returns
the (system_message, user_message) pair to send to Gemma4, which then
generates the final agent system prompt.

Usage:
    from master_prompt import build_generation_request

    system_msg, user_msg = build_generation_request(persona, company)
    # Send both to Gemma4 — its response IS the final agent system prompt.
"""

# ---------------------------------------------------------------------------
# Meta-prompt: instructs Gemma4 on what to produce
# ---------------------------------------------------------------------------

_SYSTEM = """\
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

_USER_TEMPLATE = """\
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
# Builder
# ---------------------------------------------------------------------------


def build_generation_request(persona: dict, company: dict) -> tuple[str, str]:
    """
    Build the (system_message, user_message) pair to send to Gemma4.

    Gemma4's response is the final agent system prompt — save that output,
    not the strings returned here.

    Args:
        persona: dict with keys:
            name, brief_identity, identity_block, tone,
            interaction_rules, failure_patterns, role_anchor
        company: dict with keys:
            bot_name, company_name, industry,
            capabilities_summary, test_scenarios

    Returns:
        (system_msg, user_msg) — pass directly as the system and user
        messages in the Gemma4 /api/chat request.
    """
    user_msg = _USER_TEMPLATE.format(
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
    return _SYSTEM, user_msg
