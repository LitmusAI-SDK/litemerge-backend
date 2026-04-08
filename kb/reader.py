"""KBReader — fetches relevant findings from MongoDB to enrich persona prompts.

Called by PersonaSession before each simulation session so that personas can
probe known weaknesses discovered in prior runs.

Filtering logic mirrors personas/kb_filter.py:
  - Include findings whose persona_type matches the current persona, OR
  - Include any finding with severity == "critical" (always relevant).
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorDatabase

_SEVERITY_RANK: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


class KBReader:
    """Reads findings from the KB for a given project and persona type."""

    async def get_findings(
        self,
        db: AsyncIOMotorDatabase,
        project_id: str,
        *,
        persona_type: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Return up to `limit` findings for the project, sorted by severity.

        If persona_type is provided, only findings that match that type OR have
        severity "critical" are returned (so critical findings are always surfaced).

        Results are sorted critical → high → medium → low, then newest-first
        within the same severity tier.
        """
        if persona_type:
            query: dict = {
                "project_id": project_id,
                "$or": [
                    {"persona_type": persona_type},
                    {"severity": "critical"},
                ],
            }
        else:
            query = {"project_id": project_id}

        cursor = (
            db["findings"]
            .find(query, {"_id": 0})
            .sort("created_at", -1)
            .limit(limit)
        )

        findings: list[dict] = []
        async for doc in cursor:
            findings.append(doc)

        # Sort by severity rank (critical first), stable so created_at order preserved
        # within the same tier.
        findings.sort(key=lambda f: _SEVERITY_RANK.get(f.get("severity", "low"), 3))
        return findings
