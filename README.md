# litemerge-backend

Phase 1A backend scaffold for LitmusAI, adapted to MongoDB.

## What this includes

- FastAPI app with v1 routers:
	- `/v1/health`
	- `/v1/runs`
	- `/v1/projects`
	- `/v1/reports`
- MongoDB persistence for API keys and runs
- MongoDB migration runner using scripts in `alembic/versions`
- Singleton `settings` document to track DB schema version metadata
- API key middleware (`x-api-key`) checked against Mongo `api_keys`
- Celery + Redis worker scaffold
- Docker Compose stack for local development

## Local quick start

```bash
docker compose up --build
```

API is available at `http://localhost:8000`.

## Bootstrap API key

The app seeds one development key at startup via `BOOTSTRAP_API_KEY`.

Default key in compose: `lmai_dev_key`

Use it in requests:

```http
x-api-key: lmai_dev_key
```

## Trigger a run

```bash
curl -X POST http://localhost:8000/v1/runs \
	-H "Content-Type: application/json" \
	-H "x-api-key: lmai_dev_key" \
	-d '{
		"project_id": "proj_abc123",
		"test_suite": "standard",
		"fail_threshold": 70
	}'
```

## Notes

- This is Phase 1A scaffold. Project CRUD/auth-config management from Session 1B is not implemented yet.
- The `alembic/versions` folder is used as a MongoDB migration script directory.

## MongoDB migrations

- Migration files live in `alembic/versions`.
- Each migration script must define:
	- `VERSION = <int>`
	- `async def upgrade(db) -> None`
- On API startup, pending migrations are applied in ascending `VERSION` order.
- Migration state is stored in one singleton document:
	- Collection: `settings`
	- `_id`: `litmusai_settings`
	- Fields include `db_version`, `last_migration`, and `applied_migrations`.
