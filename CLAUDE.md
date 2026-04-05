# CLAUDE.md

## Project Overview
- Product: LitmusAI behavioral testing platform.
- Goal: run persona-driven simulations against customer-deployed AI agents, score behavior, and gate CI/CD.
- Core architecture: hosted LitmusAI backend calls customer agent endpoints, evaluates outcomes, and exposes reports.

## Workspace Structure
- backend: FastAPI + Celery + MongoDB + Redis services.
- frontend: Next.js dashboard (scaffold present, implementation pending).
- Litmus- Emergence 33967dbd4e178077ad5de285f401ce09.md: authoritative implementation reference document.

## Product Flow (High Level)
1. Customer CI calls LitmusAI `POST /v1/runs`.
2. Litmus backend orchestrates simulation sessions.
3. Backend calls customer agent endpoint with persona conversations.
4. Evaluation engine scores run and generates artifacts.
5. CI polls status or receives webhook, then passes/fails pipeline.

## Backend Implementation Status
- Completed: Phase 1A
  - FastAPI shell and route structure under `/v1`.
  - Health endpoint.
  - Runs create/status endpoints.
  - API key middleware (`x-api-key`).
  - Celery worker scaffold.
  - Docker compose stack (`api`, `worker`, `mongo`, `redis`).
- Completed: Phase 1B
  - Mongo migration runner bootstrapped at app startup.
  - Singleton settings document for DB schema metadata:
    - Collection: `settings`
    - `_id`: `litmusai_settings`
    - Tracks: `db_version`, `last_migration`, `applied_migrations`.
  - Project CRUD:
    - `POST /v1/projects`
    - `GET /v1/projects`
    - `PATCH /v1/projects/{id}`
  - Auth config model with types:
    - `bearer`, `apikey`, `basic`, `none`
  - Auth secrets stored as encrypted strings in Mongo.
  - API key project scoping integrated into project listing and run authorization.

## MongoDB Migration Strategy
- Migration scripts are stored in `backend/db/versions`.
- Each migration must define:
  - `VERSION = <int>`
  - `async def upgrade(db) -> None`
- Startup executes pending migrations in ascending version order.
- Existing migrations:
  - `v0001_init_settings.py`
  - `v0002_projects_and_api_keys.py`

## Data Modeling Approach
- API request/response validation uses Pydantic models under `backend/api/schemas`.
- MongoDB persistence uses plain document shapes and migration scripts; no SQLAlchemy or Alembic ORM models are used.

## Current Key Data Collections
- `settings`: singleton migration/version metadata.
- `api_keys`: hashed key records, scopes, project access list.
- `projects`: project configuration (endpoint, owner, auth config, schema hints).
- `runs`: async run lifecycle records.

## Environment Variables (Backend)
- `MONGODB_URL`
- `MONGODB_DB`
- `REDIS_URL`
- `PUBLIC_API_BASE_URL`
- `API_KEY_HEADER`
- `BOOTSTRAP_API_KEY`
- `SECRETS_ENCRYPTION_KEY`
- `AUTH_EXEMPT_PATHS`

## Common Backend Commands
- Start stack: `docker compose up --build`
- Run tests: `python -m pytest -q`

## Next Recommended Phase
- Phase 2A: Agent caller module
  - apply auth config to outbound requests
  - request/response schema mapping
  - latency and error capture
- Phase 2B: Pre-flight route
  - `POST /v1/projects/{id}/preflight`
  - return green/amber/red status + latency + error details

## Notes for Future Sessions
- This implementation intentionally uses MongoDB instead of Postgres.
- Keep all DB shape/index changes in migration scripts, not ad-hoc startup logic.
- Preserve API key scoping behavior when adding new routes that reference `project_id`.