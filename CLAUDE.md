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
- Completed: Phase 2A
  - `backend/caller/` package with `AgentCaller` class and `CallerResult` dataclass.
  - `AgentCaller(project_doc)` accepts a MongoDB project document directly.
  - Auth applied per type: `bearer` → `Authorization: Bearer`, `apikey` → custom header, `basic` → `Authorization: Basic` (base64), `none` → no auth headers.
  - Request body field names mapped via `schema_hints` (default: `message`, `session_id`, `conversation_history`); response `reply` field also remappable.
  - `X-LitmusAI-Session: {session_id}` passthrough header always sent.
  - 30s default timeout; errors (timeout, network, non-200, bad JSON, missing reply) surface via `result.error`, never raised.
  - `CallerResult.ok` property: True only when `reply` is populated and `error` is None.
  - 14 unit tests in `tests/test_agent_caller.py` covering all auth types, schema remapping, and all error paths.
- Completed: Phase 2B
  - `POST /v1/projects/{id}/preflight` — sends one neutral probe to the customer's agent via `AgentCaller`.
  - Status classification: `green` (200 + reply + latency < 2000 ms), `amber` (200 + reply + latency >= 2000 ms), `red` (any error).
  - Response: `{ status, latency_ms, error? }`.
  - Full auth middleware chain applies (API key lookup, project access check, 404 if project missing).
  - Session ID sent as `litmusai-preflight-{project_id}` so customers can filter probe traffic in their logs.
  - 11 integration tests in `tests/test_preflight.py`.

## Known schema_hints Gaps (to address in Phase 3)
`AgentCaller` currently handles flat field-name remapping only. Three cases are deferred:
1. **Nested response extraction** — e.g. `choices[0].message.content` (OpenAI-style). Needs dot/index path support in `schema_hints["reply"]`.
2. **Stateful endpoints** — agents that track history server-side reject `conversation_history` in the body. Needs a `send_history: false` hint.
3. **Extra required body fields** — some APIs require fixed params (`model`, `temperature`). Needs a `static_body_fields: dict` hint in project config.

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

## Tooling — venv and uv
- The Python virtualenv lives at `backend/venv/`. Always use it — never the system Python.
- **Run tests:** `venv/Scripts/python -m pytest -q` (run from `backend/`)
- **Install a package:** `uv pip install --python venv/Scripts/python <package>` (run from `backend/`)
- **Install all deps:** `uv pip install -r requirements.txt --python venv/Scripts/python`
- **Start stack:** `docker compose up --build`
- Async tests use `@pytest.mark.anyio` — `trio` is not installed, only `asyncio` backend works.
- `tests/conftest.py` patches DB startup functions so `TestClient` starts without MongoDB.

## Next Recommended Phase
- Phase 3A: Persona template system
  - Seven persona dataclasses (low_literacy, non_native, adversarial, distressed, domain_expert, ambiguous, multi_turn_drift)
  - `generate(persona_type, domain_context, kb_findings) → system_prompt`
  - Domain context injection + KB findings appended as probing hints
  - LLM interface via LiteLLM (agreed at Phase 2A completion)

## Testing Notes
- Run tests: `venv/Scripts/python -m pytest -q`
- Async tests use `pytest.mark.anyio` + `anyio_backend = "asyncio"` fixture in `tests/conftest.py` (trio not installed).

## Notes for Future Sessions
- This implementation intentionally uses MongoDB instead of Postgres.
- Keep all DB shape/index changes in migration scripts, not ad-hoc startup logic.
- Preserve API key scoping behavior when adding new routes that reference `project_id`.