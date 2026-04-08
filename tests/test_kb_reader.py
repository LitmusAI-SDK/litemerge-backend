"""Unit tests for KBReader."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from kb.reader import KBReader


def _make_finding(persona_type: str, severity: str, created_offset: int = 0) -> dict:
    return {
        "project_id": "proj-1",
        "run_id": "run-abc",
        "persona_type": persona_type,
        "finding_type": "boundary_violation",
        "severity": severity,
        "prompt_vector": None,
        "agent_response_excerpt": None,
        "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
    }


def _mock_cursor(docs: list[dict]):
    """Return an async iterable mock for Motor cursor."""
    cursor = MagicMock()
    cursor.sort = MagicMock(return_value=cursor)
    cursor.limit = MagicMock(return_value=cursor)

    async def _aiter(self):
        for doc in docs:
            yield doc

    cursor.__aiter__ = _aiter
    return cursor


def _mock_db(docs: list[dict]) -> MagicMock:
    db = MagicMock()
    db["findings"].find = MagicMock(return_value=_mock_cursor(docs))
    return db


@pytest.mark.anyio
async def test_get_findings_returns_list(mock_db=None):
    docs = [_make_finding("adversarial", "high")]
    db = _mock_db(docs)

    reader = KBReader()
    result = await reader.get_findings(db, "proj-1", persona_type="adversarial")

    assert isinstance(result, list)
    assert len(result) == 1


@pytest.mark.anyio
async def test_get_findings_severity_sort():
    """Critical findings should appear before high, high before medium."""
    docs = [
        _make_finding("adversarial", "medium"),
        _make_finding("adversarial", "critical"),
        _make_finding("adversarial", "high"),
    ]
    db = _mock_db(docs)

    reader = KBReader()
    result = await reader.get_findings(db, "proj-1", persona_type="adversarial")

    severities = [r["severity"] for r in result]
    assert severities == ["critical", "high", "medium"]


@pytest.mark.anyio
async def test_get_findings_with_persona_type_uses_or_query():
    """Querying with persona_type should send an $or query to include critical findings."""
    db = _mock_db([])

    reader = KBReader()
    await reader.get_findings(db, "proj-1", persona_type="adversarial")

    call_query = db["findings"].find.call_args[0][0]
    assert "$or" in call_query
    assert {"persona_type": "adversarial"} in call_query["$or"]
    assert {"severity": "critical"} in call_query["$or"]


@pytest.mark.anyio
async def test_get_findings_without_persona_type_uses_simple_query():
    """Querying without persona_type should use a simple project_id filter."""
    db = _mock_db([])

    reader = KBReader()
    await reader.get_findings(db, "proj-1")

    call_query = db["findings"].find.call_args[0][0]
    assert "$or" not in call_query
    assert call_query == {"project_id": "proj-1"}


@pytest.mark.anyio
async def test_get_findings_empty_db_returns_empty_list():
    db = _mock_db([])

    reader = KBReader()
    result = await reader.get_findings(db, "proj-1", persona_type="adversarial")

    assert result == []
