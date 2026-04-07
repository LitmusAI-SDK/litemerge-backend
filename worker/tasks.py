"""Celery tasks for LitmusAI background job execution.

process_run is the main task — it's dispatched by POST /v1/runs and drives
the full simulation lifecycle inside a Celery worker process.

Because PersonaSession and SimulationRun are async, we wrap execution in
asyncio.run() here. Celery workers are sync; the async event loop is created
fresh per task invocation.
"""

import asyncio

from celery.utils.log import get_task_logger

from worker.celery_app import celery_app

logger = get_task_logger(__name__)


@celery_app.task(name="worker.ping")
def ping() -> str:
    return "pong"


@celery_app.task(name="worker.process_run", bind=True, max_retries=0)
def process_run(self, run_id: str) -> dict:
    """Execute a full simulation run asynchronously inside a Celery worker.

    Steps:
      1. Open a dedicated MongoDB connection (no shared app.state in workers).
      2. Open a Redis client for pub/sub event publishing.
      3. Fetch the run document to get project_id and test_suite.
      4. Fetch the project document for full config.
      5. Execute SimulationRun.execute() via asyncio.run().
      6. Return summary dict (stored as Celery task result).

    On any unhandled exception the run document is marked "failed" before
    re-raising so the API can surface a terminal state.
    """
    logger.info("process_run received run_id=%s", run_id)
    return asyncio.run(_execute(run_id))


async def _execute(run_id: str) -> dict:
    """Async body of process_run — runs inside asyncio.run()."""
    from motor.motor_asyncio import AsyncIOMotorClient
    import redis.asyncio as aioredis

    from core.config import settings
    from simulation.runner import SimulationRun

    mongo_client = AsyncIOMotorClient(settings.mongodb_url)
    db = mongo_client[settings.mongodb_db]
    redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)

    try:
        # Fetch run doc
        run_doc = await db["runs"].find_one({"run_id": run_id})
        if run_doc is None:
            logger.error("process_run: run_id=%s not found in DB", run_id)
            return {"run_id": run_id, "status": "failed", "error": "run not found"}

        project_id = run_doc["project_id"]
        test_suite = run_doc.get("test_suite", "standard")

        # Fetch project doc
        project_doc = await db["projects"].find_one({"_id": project_id})
        if project_doc is None:
            logger.error(
                "process_run: project_id=%s not found for run=%s", project_id, run_id
            )
            await _mark_failed(db, run_id)
            return {"run_id": run_id, "status": "failed", "error": "project not found"}

        runner = SimulationRun(
            run_id=run_id,
            project_config=project_doc,
            db=db,
            redis_client=redis_client,
        )

        summary = await runner.execute(test_suite=test_suite)
        logger.info("process_run complete run_id=%s summary=%s", run_id, summary)
        return {"run_id": run_id, "status": "complete", "summary": summary}

    except Exception as exc:
        logger.exception("process_run unhandled error for run_id=%s", run_id)
        await _mark_failed(db, run_id)
        raise exc

    finally:
        await redis_client.aclose()
        mongo_client.close()


async def _mark_failed(db, run_id: str) -> None:
    from datetime import datetime, timezone

    try:
        await db["runs"].update_one(
            {"run_id": run_id},
            {"$set": {"status": "failed", "updated_at": datetime.now(timezone.utc)}},
        )
    except Exception:
        pass
