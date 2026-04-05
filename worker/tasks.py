from celery.utils.log import get_task_logger

from worker.celery_app import celery_app

logger = get_task_logger(__name__)


@celery_app.task(name="worker.ping")
def ping() -> str:
    return "pong"


@celery_app.task(name="worker.process_run")
def process_run(run_id: str) -> dict[str, str]:
    logger.info("Received run %s for processing", run_id)
    return {"run_id": run_id, "status": "queued"}
