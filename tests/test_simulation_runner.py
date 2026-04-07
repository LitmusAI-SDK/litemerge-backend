"""Tests for simulation/runner.py — SimulationRun orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from simulation.runner import SUITE_PERSONAS, SimulationRun


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_db(run_doc=None, project_doc=None):
    """Build a mock Motor database with pre-configured find_one responses."""
    db = MagicMock()
    db["runs"].update_one = AsyncMock()
    db["runs"].find_one = AsyncMock(return_value=run_doc)
    db["projects"].find_one = AsyncMock(return_value=project_doc)
    return db


def _project_config(project_id="proj_1"):
    return {
        "_id": project_id,
        "name": "Test Product",
        "agent_endpoint": "http://agent.example.com/chat",
        "auth_config": {"type": "none"},
    }


def _make_runner(
    run_id="run_test",
    suite_override=None,
    redis=None,
    session_side_effect=None,
):
    db = _make_db()
    runner = SimulationRun(
        run_id=run_id,
        project_config=_project_config(),
        db=db,
        redis_client=redis,
    )
    return runner, db


# ---------------------------------------------------------------------------
# Suite → persona mapping
# ---------------------------------------------------------------------------


def test_standard_suite_personas():
    assert set(SUITE_PERSONAS["standard"]) == {"p1", "p2", "p3", "p4", "p5", "p6", "p7"}


def test_adversarial_suite_personas():
    assert set(SUITE_PERSONAS["adversarial"]) == {"p2", "p4", "p7", "p8"}


def test_full_suite_personas():
    assert set(SUITE_PERSONAS["full"]) == {
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
    }


def test_full_is_superset_of_standard():
    assert set(SUITE_PERSONAS["standard"]).issubset(set(SUITE_PERSONAS["full"]))


# ---------------------------------------------------------------------------
# SimulationRun.execute — happy path
# ---------------------------------------------------------------------------


def _session_factory(turn_count=3, raises=None):
    """Return a side_effect callable that yields fresh session mocks per call."""

    def factory(*_, **__):
        m = MagicMock()
        if raises:
            m.run = AsyncMock(side_effect=raises)
        else:
            m.run = AsyncMock(return_value=[MagicMock()] * turn_count)
        return m

    return factory


@pytest.mark.anyio
async def test_execute_standard_runs_7_sessions():
    """execute() with standard suite should spin up exactly 7 PersonaSessions."""
    runner, db = _make_runner()

    with patch(
        "simulation.runner.PersonaSession", side_effect=_session_factory(3)
    ) as mock_cls:
        summary = await runner.execute(test_suite="standard")

    assert mock_cls.call_count == 7
    assert summary["personas_deployed"] == 7
    assert summary["total_conversations"] == 21  # 7 sessions × 3 turns
    assert summary["sessions_failed"] == 0


@pytest.mark.anyio
async def test_execute_adversarial_suite():
    runner, db = _make_runner()

    with patch(
        "simulation.runner.PersonaSession", side_effect=_session_factory(5)
    ) as mock_cls:
        summary = await runner.execute(test_suite="adversarial")

    assert mock_cls.call_count == 4
    assert summary["personas_deployed"] == 4


@pytest.mark.anyio
async def test_execute_full_suite():
    runner, db = _make_runner()

    with patch(
        "simulation.runner.PersonaSession", side_effect=_session_factory(2)
    ) as mock_cls:
        _summary = await runner.execute(test_suite="full")

    assert mock_cls.call_count == 8


# ---------------------------------------------------------------------------
# Run document state transitions
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_execute_sets_running_then_complete():
    """Run doc should go queued→running (at start) then complete (at end)."""
    runner, db = _make_runner()

    session_mock = MagicMock()
    session_mock.run = AsyncMock(return_value=[])

    with patch("simulation.runner.PersonaSession", return_value=session_mock):
        await runner.execute(test_suite="standard")

    calls = db["runs"].update_one.call_args_list
    statuses = [c[0][1]["$set"]["status"] for c in calls]
    assert statuses[0] == "running"
    assert statuses[-1] in ("complete", "failed")


@pytest.mark.anyio
async def test_execute_all_failed_marks_run_failed():
    """If every session raises, overall status should be 'failed'."""
    runner, db = _make_runner()

    session_mock = MagicMock()
    session_mock.run = AsyncMock(side_effect=RuntimeError("agent down"))

    with patch("simulation.runner.PersonaSession", return_value=session_mock):
        summary = await runner.execute(test_suite="standard")

    assert summary["sessions_failed"] == 7

    calls = db["runs"].update_one.call_args_list
    final_status = calls[-1][0][1]["$set"]["status"]
    assert final_status == "failed"


# ---------------------------------------------------------------------------
# Redis event publishing
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_events_published_on_session_complete():
    """Expect session_started, session_completed, and run_complete events."""
    mock_redis = MagicMock()
    mock_redis.publish = AsyncMock()

    runner, _ = _make_runner(redis=mock_redis)

    session_mock = MagicMock()
    session_mock.run = AsyncMock(return_value=[MagicMock()])  # 1 turn

    with patch("simulation.runner.PersonaSession", return_value=session_mock):
        await runner.execute(test_suite="adversarial")  # 4 personas

    published_events = [
        __import__("json").loads(call[0][1])
        for call in mock_redis.publish.call_args_list
    ]
    event_types = [e["event"] for e in published_events]

    assert event_types.count("session_started") == 4
    assert event_types.count("session_completed") == 4
    assert event_types.count("run_complete") == 1


@pytest.mark.anyio
async def test_no_redis_does_not_raise():
    """SimulationRun with redis_client=None must not raise."""
    runner, _ = _make_runner(redis=None)

    session_mock = MagicMock()
    session_mock.run = AsyncMock(return_value=[])

    with patch("simulation.runner.PersonaSession", return_value=session_mock):
        summary = await runner.execute(test_suite="adversarial")

    assert "personas_deployed" in summary


@pytest.mark.anyio
async def test_failed_session_publishes_session_failed_event():
    mock_redis = MagicMock()
    mock_redis.publish = AsyncMock()

    runner, _ = _make_runner(redis=mock_redis)

    session_mock = MagicMock()
    session_mock.run = AsyncMock(side_effect=RuntimeError("timeout"))

    with patch("simulation.runner.PersonaSession", return_value=session_mock):
        await runner.execute(test_suite="adversarial")

    published_events = [
        __import__("json").loads(call[0][1])
        for call in mock_redis.publish.call_args_list
    ]
    failed_events = [e for e in published_events if e["event"] == "session_failed"]
    assert len(failed_events) == 4
    assert all("error" in e for e in failed_events)


# ---------------------------------------------------------------------------
# Unknown suite falls back to standard
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_unknown_suite_falls_back_to_standard():
    runner, _ = _make_runner()

    session_mock = MagicMock()
    session_mock.run = AsyncMock(return_value=[])

    with patch(
        "simulation.runner.PersonaSession", return_value=session_mock
    ) as mock_cls:
        await runner.execute(test_suite="unknown_suite")

    assert mock_cls.call_count == 7  # standard = 7 personas
