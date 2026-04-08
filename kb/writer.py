"""KBWriter — persists flagged findings from simulation sessions to MongoDB.

Called by the evaluation engine (Phase 6) and the findings API endpoint.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from motor.motor_asyncio import AsyncIOMotorDatabase

FindingType = Literal[
    "prompt_injection_success",
    "boundary_violation",
    "hallucination",
    "inappropriate_response",
    "refusal_failure",
]

Severity = Literal["critical", "high", "medium", "low"]


class KBWriter:
    """Writes flagged behavior findings to the findings collection."""

    async def write_finding(
        self,
        db: AsyncIOMotorDatabase,
        *,
        project_id: str,
        run_id: str | None,
        persona_type: str,
        finding_type: FindingType,
        severity: Severity,
        prompt_vector: str | None = None,
        agent_response_excerpt: str | None = None,
    ) -> str:
        """Insert a finding document. Returns the inserted document ID as a string."""
        doc = {
            "project_id": project_id,
            "run_id": run_id,
            "persona_type": persona_type,
            "finding_type": finding_type,
            "severity": severity,
            "prompt_vector": prompt_vector,
            "agent_response_excerpt": agent_response_excerpt,
            "created_at": datetime.now(timezone.utc),
        }
        result = await db["findings"].insert_one(doc)
        return str(result.inserted_id)
