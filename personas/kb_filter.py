"""KB relevance filter — Phase 5 slot, wired now with empty list.

Phase 5: session.py passes [] here. Replace with live DB findings then.
"""


def filter_findings(
    findings: list[dict],
    persona_type: str,
    max_findings: int = 3,
) -> list[dict]:
    """Include a finding if:
      - finding["persona_type"] == persona_type, OR
      - finding["severity"] == "critical"

    Returns at most max_findings items.
    """
    relevant = [
        f
        for f in findings
        if f.get("persona_type") == persona_type or f.get("severity") == "critical"
    ]
    return relevant[:max_findings]
