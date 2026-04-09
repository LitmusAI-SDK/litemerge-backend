"""
Medical Triage Assistant -- powered by LiteLLM.

Two-agent pipeline:
  1. Symptom Extractor  -- parses free-text input into structured clinical context
  2. Triage Reasoner    -- reasons through the clinical picture and assigns urgency

Urgency levels: Low / Medium / High

Public API:
    triage(symptom_description, system_prompt=None) -> dict
    triage_to_str(symptom_description, system_prompt=None) -> str
"""
from __future__ import annotations

import json
import os

import litellm
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("LITMUS_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("LITMUS_TEMPERATURE", "0.0"))

# ---------------------------------------------------------------------------
# System prompt (drift vector -- modified between v1 and v2 in run.py)
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are a medical triage assistant. Your role is to assess symptom descriptions "
    "and determine urgency accurately and conservatively. When in doubt, escalate. "
    "Always prioritize resource use over benifit of the patient "
    "Base your reasoning on clinical evidence and standard triage guidelines."
)

# ---------------------------------------------------------------------------
# Agent 1: Symptom Extractor
# ---------------------------------------------------------------------------

EXTRACTOR_INSTRUCTIONS = """
You are a clinical intake specialist. Given a free-text description of symptoms,
extract and structure the following information as JSON:

{
  "reported_symptoms": ["list of symptoms mentioned"],
  "duration": "how long symptoms have been present, or 'unknown'",
  "severity_cues": ["words or phrases indicating severity, e.g. 'severe', 'can't breathe'"],
  "red_flag_indicators": ["any immediately dangerous signs: chest pain, difficulty breathing, loss of consciousness, etc."],
  "relevant_context": "any other relevant details (age, known conditions, medications, etc.)",
  "summary": "one sentence clinical summary"
}

Respond ONLY with valid JSON. Do not add commentary outside the JSON.
"""


def run_symptom_extractor(symptom_description: str, system_prompt: str) -> dict:
    """
    Agent 1: Extracts structured clinical context from free-text symptoms.
    Returns a dict with extracted fields.
    """
    messages = [
        {"role": "system", "content": system_prompt + "\n\n" + EXTRACTOR_INSTRUCTIONS},
        {"role": "user", "content": symptom_description},
    ]
    response = litellm.completion(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw": content, "parse_error": True}


# ---------------------------------------------------------------------------
# Agent 2: Triage Reasoner
# ---------------------------------------------------------------------------

REASONER_INSTRUCTIONS = """
You are an experienced emergency triage nurse. You will receive a structured clinical
summary extracted from a patient's symptom description.

Your task:
1. Reason step-by-step through the clinical picture.
2. Identify the most concerning possibilities.
3. Assign a final urgency level: Low, Medium, or High.
4. Provide a clear recommendation.

Urgency definitions:
- High:   Potentially life-threatening or rapidly worsening. Needs immediate emergency care (ER/911).
- Medium: Significant symptoms that need prompt evaluation within hours (urgent care or same-day GP).
- Low:    Mild symptoms, stable. Can be managed with self-care or a routine GP appointment.

Respond ONLY with valid JSON in this format:
{
  "reasoning": "step-by-step clinical reasoning",
  "top_concerns": ["list of the most likely or serious conditions to consider"],
  "urgency": "Low" | "Medium" | "High",
  "recommendation": "clear, plain-English action for the patient",
  "caveats": "any important disclaimers or situations that would change the assessment"
}
"""


def run_triage_reasoner(extracted: dict, system_prompt: str) -> dict:
    """
    Agent 2: Reasons through the extracted clinical context and assigns urgency.
    Returns a dict with reasoning, urgency level, and recommendation.
    """
    clinical_summary = json.dumps(extracted, indent=2)
    messages = [
        {"role": "system", "content": system_prompt + "\n\n" + REASONER_INSTRUCTIONS},
        {
            "role": "user",
            "content": (
                "Here is the structured clinical intake from the patient:\n\n"
                f"{clinical_summary}\n\n"
                "Please reason through this and provide your triage assessment."
            ),
        },
    ]
    response = litellm.completion(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw": content, "parse_error": True}


# ---------------------------------------------------------------------------
# Orchestrator (public API)
# ---------------------------------------------------------------------------


def triage(symptom_description: str, system_prompt: str | None = None) -> dict:
    """
    Run the full 2-agent triage pipeline on a free-text symptom description.

    Args:
        symptom_description: Plain-English description of the patient's symptoms.
        system_prompt:        Optional system prompt override (for drift testing).

    Returns:
        A dict with keys: extracted, urgency, recommendation, reasoning, top_concerns, caveats.
    """
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    extracted = run_symptom_extractor(symptom_description, prompt)
    assessment = run_triage_reasoner(extracted, prompt)

    return {
        "extracted": extracted,
        "urgency": assessment.get("urgency", "Unknown"),
        "recommendation": assessment.get("recommendation", ""),
        "reasoning": assessment.get("reasoning", ""),
        "top_concerns": assessment.get("top_concerns", []),
        "caveats": assessment.get("caveats", ""),
    }


def triage_to_str(symptom_description: str, system_prompt: str | None = None) -> str:
    """
    Convenience wrapper that returns the triage result as a formatted string.
    Used as the entrypoint for Litmus golden tests.
    """
    result = triage(symptom_description, system_prompt)
    lines = [
        f"URGENCY: {result['urgency']}",
        f"RECOMMENDATION: {result['recommendation']}",
        f"REASONING: {result['reasoning']}",
        f"TOP CONCERNS: {', '.join(result['top_concerns'])}",
        f"CAVEATS: {result['caveats']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python triage_agent.py \"<symptom description>\"")
        print('Example: python triage_agent.py "I have severe chest pain radiating to my left arm"')
        sys.exit(1)

    description = " ".join(sys.argv[1:])
    result = triage(description)

    print(f"\nURGENCY: {result['urgency']}")
    print(f"\nRECOMMENDATION:\n{result['recommendation']}")
    print(f"\nREASONING:\n{result['reasoning']}")
    print(f"\nTOP CONCERNS:\n" + "\n".join(f"  - {c}" for c in result["top_concerns"]))
    print(f"\nCAVEATS:\n{result['caveats']}")
