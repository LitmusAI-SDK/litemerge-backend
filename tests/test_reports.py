"""Tests for GET /v1/reports/{run_id}."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from bson import ObjectId
from fastapi.testclient import TestClient

from api.main import app
from core.security import hash_api_key

BOOTSTRAP_KEY = "lmai_dev_key"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api_key_record() -> dict:
    return {
        "_id": "key-id",
        "key_hash": hash_api_key(BOOTSTRAP_KEY),
        "is_active": True,
        "project_ids": [],
    }


def _make_run_doc(
    run_id="run_abc",
    status="complete",
    score=85,
    fail_threshold=70,
    test_suite="standard",
):
    now = datetime.now(timezone.utc)
    return {
        "run_id": run_id,
        "project_id": "proj_1",
        "test_suite": test_suite,
        "fail_threshold": fail_threshold,
        "notify_webhook": None,
        "status": status,
        "score": score,
        "summary": {
            "total_conversations": 56,
            "personas_deployed": 7,
            "sessions_failed": 0,
            "issues_flagged": 2,
        },
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }


def _make_finding(finding_type="hallucination", severity="high", run_id="run_abc"):
    return {
        "_id": ObjectId(),
        "project_id": "proj_1",
        "run_id": run_id,
        "persona_type": "adversarial",
        "finding_type": finding_type,
        "severity": severity,
        "prompt_vector": "some prompt",
        "agent_response_excerpt": "some response",
        "created_at": datetime.now(timezone.utc),
    }


def _setup_db(app, run_doc, findings):
    """Replace app.state.db with per-collection mocks.

    Uses __getitem__ side_effect (same pattern as test_preflight.py) so that
    the auth middleware's db["api_keys"].find_one() returns a valid key record.
    """
    runs_col = MagicMock()
    runs_col.find_one = AsyncMock(return_value=run_doc)

    api_keys_col = MagicMock()
    api_keys_col.find_one = AsyncMock(return_value=_api_key_record())

    mock_cursor = MagicMock()
    mock_cursor.sort = MagicMock(return_value=mock_cursor)
    mock_cursor.to_list = AsyncMock(return_value=findings)
    findings_col = MagicMock()
    findings_col.find = MagicMock(return_value=mock_cursor)

    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(
        side_effect=lambda name: {
            "runs": runs_col,
            "api_keys": api_keys_col,
            "findings": findings_col,
        }.get(name, MagicMock())
    )
    app.state.db = mock_db
    app.state.db_ready = True
    return mock_db


# ---------------------------------------------------------------------------
# GET /v1/reports/{run_id}
# ---------------------------------------------------------------------------


def test_get_report_returns_200_for_complete_run():
    run_doc = _make_run_doc()
    findings = [_make_finding()]

    with TestClient(app) as client:
        _setup_db(app, run_doc, findings)
        resp = client.get(
            "/v1/reports/run_abc",
            headers={"Authorization": f"Bearer {BOOTSTRAP_KEY}"},
        )

    assert resp.status_code == 200


def test_get_report_returns_404_when_run_not_found():
    with TestClient(app) as client:
        _setup_db(app, None, [])
        resp = client.get(
            "/v1/reports/no_such_run",
            headers={"Authorization": f"Bearer {BOOTSTRAP_KEY}"},
        )

    assert resp.status_code == 404


def test_get_report_score_and_passed_fields():
    run_doc = _make_run_doc(score=85, fail_threshold=70)

    with TestClient(app) as client:
        _setup_db(app, run_doc, [])
        resp = client.get(
            "/v1/reports/run_abc",
            headers={"Authorization": f"Bearer {BOOTSTRAP_KEY}"},
        )

    body = resp.json()
    assert body["score"] == 85
    assert body["passed"] is True
    assert body["fail_threshold"] == 70


def test_get_report_passed_false_when_below_threshold():
    run_doc = _make_run_doc(score=50, fail_threshold=70)

    with TestClient(app) as client:
        _setup_db(app, run_doc, [])
        resp = client.get(
            "/v1/reports/run_abc",
            headers={"Authorization": f"Bearer {BOOTSTRAP_KEY}"},
        )

    body = resp.json()
    assert body["passed"] is False


def test_get_report_passed_none_when_score_is_none():
    run_doc = _make_run_doc(score=None, fail_threshold=70)

    with TestClient(app) as client:
        _setup_db(app, run_doc, [])
        resp = client.get(
            "/v1/reports/run_abc",
            headers={"Authorization": f"Bearer {BOOTSTRAP_KEY}"},
        )

    body = resp.json()
    assert body["passed"] is None


def test_get_report_findings_count():
    run_doc = _make_run_doc()
    findings = [
        _make_finding("hallucination", "high"),
        _make_finding("boundary_violation", "critical"),
        _make_finding("hallucination", "medium"),
    ]

    with TestClient(app) as client:
        _setup_db(app, run_doc, findings)
        resp = client.get(
            "/v1/reports/run_abc",
            headers={"Authorization": f"Bearer {BOOTSTRAP_KEY}"},
        )

    body = resp.json()
    assert body["findings_count"] == 3


def test_get_report_findings_by_severity_aggregation():
    run_doc = _make_run_doc()
    findings = [
        _make_finding("hallucination", "critical"),
        _make_finding("boundary_violation", "high"),
        _make_finding("hallucination", "high"),
        _make_finding("refusal_failure", "low"),
    ]

    with TestClient(app) as client:
        _setup_db(app, run_doc, findings)
        resp = client.get(
            "/v1/reports/run_abc",
            headers={"Authorization": f"Bearer {BOOTSTRAP_KEY}"},
        )

    body = resp.json()
    sev = body["findings_by_severity"]
    assert sev["critical"] == 1
    assert sev["high"] == 2
    assert sev["medium"] == 0
    assert sev["low"] == 1


def test_get_report_findings_by_type_aggregation():
    run_doc = _make_run_doc()
    findings = [
        _make_finding("hallucination", "high"),
        _make_finding("hallucination", "medium"),
        _make_finding("boundary_violation", "critical"),
    ]

    with TestClient(app) as client:
        _setup_db(app, run_doc, findings)
        resp = client.get(
            "/v1/reports/run_abc",
            headers={"Authorization": f"Bearer {BOOTSTRAP_KEY}"},
        )

    body = resp.json()
    by_type = {t["finding_type"]: t for t in body["findings_by_type"]}

    assert "hallucination" in by_type
    assert by_type["hallucination"]["count"] == 2

    assert "boundary_violation" in by_type
    assert by_type["boundary_violation"]["count"] == 1
    assert by_type["boundary_violation"]["by_severity"]["critical"] == 1


def test_get_report_includes_run_metadata():
    run_doc = _make_run_doc(test_suite="adversarial")

    with TestClient(app) as client:
        _setup_db(app, run_doc, [])
        resp = client.get(
            "/v1/reports/run_abc",
            headers={"Authorization": f"Bearer {BOOTSTRAP_KEY}"},
        )

    body = resp.json()
    assert body["run_id"] == "run_abc"
    assert body["status"] == "complete"
    assert body["test_suite"] == "adversarial"
    assert body["summary"]["personas_deployed"] == 7


def test_get_report_findings_sorted_critical_first():
    run_doc = _make_run_doc()
    findings = [
        _make_finding("hallucination", "low"),
        _make_finding("boundary_violation", "critical"),
        _make_finding("refusal_failure", "medium"),
    ]

    with TestClient(app) as client:
        _setup_db(app, run_doc, findings)
        resp = client.get(
            "/v1/reports/run_abc",
            headers={"Authorization": f"Bearer {BOOTSTRAP_KEY}"},
        )

    body = resp.json()
    severities = [f["severity"] for f in body["findings"]]
    assert severities[0] == "critical"
    assert severities[-1] == "low"


def test_get_report_no_findings_returns_zeros():
    run_doc = _make_run_doc()

    with TestClient(app) as client:
        _setup_db(app, run_doc, [])
        resp = client.get(
            "/v1/reports/run_abc",
            headers={"Authorization": f"Bearer {BOOTSTRAP_KEY}"},
        )

    body = resp.json()
    assert body["findings_count"] == 0
    assert body["findings"] == []
    assert body["findings_by_type"] == []
    assert all(v == 0 for v in body["findings_by_severity"].values())
