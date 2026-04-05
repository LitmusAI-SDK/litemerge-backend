from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from api.middleware import ApiKeyAuthMiddleware
from api.routes import health, projects, reports, runs
from core.config import settings
from db.migrations import run_migrations
from db.mongodb import bootstrap_api_key, create_mongo_client, get_database, init_indexes

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_ready = False
    app.state.startup_error = None

    app.state.mongo_client = create_mongo_client(settings.mongodb_url)
    app.state.db = get_database(app.state.mongo_client, settings.mongodb_db)

    try:
        applied_migrations = await run_migrations(app.state.db)
        if applied_migrations:
            logger.info("Applied MongoDB migrations: %s", ", ".join(applied_migrations))

        await init_indexes(app.state.db)
        await bootstrap_api_key(app.state.db, settings.bootstrap_api_key)
        app.state.db_ready = True
    except Exception as exc:
        app.state.startup_error = str(exc)
        logger.exception("Database startup setup failed")

    yield

    app.state.mongo_client.close()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    ApiKeyAuthMiddleware,
    exempt_paths=settings.auth_exempt_paths,
    api_key_header=settings.api_key_header,
)

app.include_router(health.router, prefix="/v1")
app.include_router(runs.router, prefix="/v1")
app.include_router(projects.router, prefix="/v1")
app.include_router(reports.router, prefix="/v1")
