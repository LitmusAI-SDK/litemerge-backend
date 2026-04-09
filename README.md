# litemerge-backend

Phase 1A + 1B backend scaffold for LitmusAI, adapted to MongoDB.

## What this includes

- FastAPI app with v1 routers:
	- `/v1/health`
	- `/v1/runs`
	- `/v1/projects`
	- `/v1/reports`
- MongoDB persistence for API keys and runs
- MongoDB migration runner using scripts in `db/versions`
- Singleton `settings` document to track DB schema version metadata
- Project CRUD (`POST /v1/projects`, `GET /v1/projects`, `PATCH /v1/projects/{id}`)
- Encrypted storage for project auth config secrets (`auth_config.value`)
- Bearer token middleware (`Authorization: Bearer <token>`) checked against Mongo `api_keys`
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
Authorization: Bearer lmai_dev_key
```

Auth config encryption uses:

```bash
SECRETS_ENCRYPTION_KEY=replace-with-a-long-random-secret
```

## Trigger a run

```bash
curl -X POST http://localhost:8000/v1/runs \
	-H "Content-Type: application/json" \
	-H "Authorization: Bearer lmai_dev_key" \
	-d '{
		"project_id": "proj_abc123",
		"test_suite": "standard",
		"fail_threshold": 70
	}'
```

## Notes

- The `db/versions` folder is used as a MongoDB migration script directory.

## MongoDB migrations

- Migration files live in `db/versions`.
- Each migration script must define:
	- `VERSION = <int>`
	- `async def upgrade(db) -> None`
- On API startup, pending migrations are applied in ascending `VERSION` order.
- Migration state is stored in one singleton document:
	- Collection: `settings`
	- `_id`: `litmusai_settings`
	- Fields include `db_version`, `last_migration`, and `applied_migrations`.
