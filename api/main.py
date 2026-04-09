from contextlib import asynccontextmanager
import logging

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import ApiKeyAuthMiddleware
from api.routes import findings, health, projects, reports, runs
from core.config import settings
from db.migrations import run_migrations
from db.mongodb import (
    bootstrap_api_key,
    create_mongo_client,
    get_database,
    init_indexes,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_ready = False
    app.state.startup_error = None

    app.state.mongo_client = create_mongo_client(settings.mongodb_url)
    app.state.db = get_database(app.state.mongo_client, settings.mongodb_db)

    # Redis async client — used by SSE stream endpoint for pub/sub subscribe
    app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)

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

    await app.state.redis.aclose()
    app.state.mongo_client.close()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    ApiKeyAuthMiddleware,
    exempt_paths=settings.auth_exempt_paths,
)

app.include_router(health, prefix="/v1")
app.include_router(runs, prefix="/v1")
app.include_router(projects, prefix="/v1")
app.include_router(reports, prefix="/v1")
app.include_router(findings, prefix="/v1")
