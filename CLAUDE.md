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
2. Litmus backend orchestrates simulation sessions via Celery.
3. `SimulationRun` spawns parallel `PersonaSession` instances, each driving an LLM persona against the customer's agent.
4. Evaluation engine scores run and generates artifacts (Phase 6).
5. CI polls `GET /v1/runs/{id}` or streams `GET /v1/runs/{id}/stream` (SSE), then passes/fails pipeline.

## Backend Implementation Status

### Completed: Phase 1A
- FastAPI shell and route structure under `/v1`.
- Health endpoint, runs create/status endpoints, API key middleware (`x-api-key`).
- Celery worker scaffold, Docker compose stack (`api`, `worker`, `mongo`, `redis`).

### Completed: Phase 1B
- Mongo migration runner bootstrapped at app startup.
- Singleton settings document (`settings` collection, `_id: litmusai_settings`).
- Project CRUD: `POST /v1/projects`, `GET /v1/projects`, `PATCH /v1/projects/{id}`.
- Auth config model: `bearer`, `apikey`, `basic`, `none`. Secrets encrypted in Mongo.
- API key project scoping on project listing and run authorization.

### Completed: Phase 2A
- `caller/agent_caller.py` — single canonical module containing:
  - `AgentCaller` + `CallerResult` (HTTP layer, used by preflight)
  - `SimulationAgentCaller` + `AgentResponse` + `AgentRetriableError` (simulation layer, wraps AgentCaller)
- Auth applied per type; field names mapped via `schema_hints`; `X-LitmusAI-Session` header always sent.
- 30s default timeout; all errors surface via result fields, never raised.
- `USE_MOCK_AGENT=true` env var enables mock failure injection (80% 200, 10% 429, 10% 503).

### Completed: Phase 2B
- `POST /v1/projects/{id}/preflight` — neutral probe, returns `{ status: green|amber|red, latency_ms, error? }`.
- `green`: 200 + reply + latency < 2000 ms. `amber`: slow but valid. `red`: any error.

### Completed: Phase 3 (Persona Engine + Conversation Runner)
- 8 persona profiles as `.md` files in `personas/profiles/` (p1–p8).
- `PersonaLoader` parses `.md` → `PersonaProfile` dataclass.
- `PersonaEngine.build_prompt(persona_id, domain_context, raw_kb_findings)` → `(system_prompt, profile)`.
- `llm/` package: `agenerate()` via LiteLLM (Gemini, Anthropic, OpenAI, LM Studio). Per-provider context caching in `llm/cache.py`; Gemini CachedContent lifecycle in `llm/gemini_cache_manager.py`.
- `PersonaSession.run()` — N-turn loop: LLM persona turn → agent call → scrub → persist → repeat. Checkpoint/resume from `chat_logs`. Default 8 turns.
- `simulation/scrubbing.py` — `Scrubber` redacts auth secrets and PII before any DB write.
- Migration `v0003_chat_logs.py` — indexes on `chat_logs` collection.

### Completed: Phase 4 (Simulation Engine + Orchestrator + SSE)
- `simulation/runner.py` — `SimulationRun`: maps test suite → persona list, runs sessions in parallel via `asyncio.gather`, publishes Redis pub/sub events.
- Suite → persona mapping: `standard` = p1–p7, `adversarial` = p2/p4/p7/p8, `full` = p1–p8.
- `worker/tasks.py` — `process_run(run_id)` Celery task: opens own Mongo+Redis connections, fetches run+project docs, calls `SimulationRun.execute()` via `asyncio.run()`.
- `POST /v1/runs` now dispatches `process_run.delay(run_id)` after inserting the run doc.
- Run state machine: `queued → running → complete | failed` (evaluating state reserved for Phase 6).
- `GET /v1/runs/{id}` returns `session_statuses[]` (per-persona progress from `chat_logs`).
- `GET /v1/runs/{id}/stream` — SSE endpoint; subscribes to Redis channel `run:{run_id}`; streams `session_started`, `session_completed`, `session_failed`, `run_complete` events; 15s keepalive.
- Redis async client (`redis.asyncio`) added to FastAPI app lifespan.

## Known schema_hints Gaps (deferred)
1. **Nested response extraction** — e.g. `choices[0].message.content`. Needs dot/index path support in `schema_hints["reply"]`.
2. **Stateful endpoints** — agents that track history server-side. Needs `send_history: false` hint.
3. **Extra required body fields** — fixed params like `model`, `temperature`. Needs `static_body_fields: dict` hint.

## MongoDB Migration Strategy
- Scripts in `backend/db/versions/`. Each defines `VERSION = <int>` and `async def upgrade(db)`.
- Startup runs pending migrations in ascending version order.
- Existing: `v0001_init_settings.py`, `v0002_projects_and_api_keys.py`, `v0003_chat_logs.py`.

## Current Key Data Collections
- `settings`: singleton migration/version metadata.
- `api_keys`: hashed key records, scopes, project access list.
- `projects`: project configuration (endpoint, owner, auth config, schema hints).
- `runs`: async run lifecycle records (status, suite, score, summary).
- `chat_logs`: per-session turn logs (persona_id, turns[], status, token usage).

## Key Architectural Decisions (deviations from original spec)
- **MongoDB** instead of Postgres + SQLAlchemy. Custom migration scripts instead of Alembic.
- **No CrewAI/AutoGen** — parallelism via native `asyncio.gather` + `PersonaSession`.
- **LiteLLM** instead of Anthropic-only — supports Gemini, Anthropic, OpenAI, LM Studio.
- **File-based persona profiles** (`.md`) instead of dataclasses — 8 profiles vs 7 planned.
- `/orchestrator` and `/simulation` merged into `simulation/` — split if orchestration grows.

## Environment Variables (Backend)
- `MONGODB_URL`, `MONGODB_DB`, `REDIS_URL`
- `PUBLIC_API_BASE_URL`, `API_KEY_HEADER`, `BOOTSTRAP_API_KEY`, `SECRETS_ENCRYPTION_KEY`, `AUTH_EXEMPT_PATHS`
- `LLM_PROVIDER`, `LLM_MODEL`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
- `LMSTUDIO_BASE_URL`, `LMSTUDIO_MODEL`
- `USE_MOCK_AGENT` — set `true` to bypass real HTTP calls in simulation (dev/test only)

## Tooling — venv and uv
- The Python virtualenv lives at `backend/venv/`. Always use it — never the system Python.
- **Run tests:** `venv/Scripts/python.exe -m pytest -q` (run from `backend/`)
- **Install a package:** `uv pip install --python venv/Scripts/python.exe <package>` (run from `backend/`)
- **Install all deps:** `uv pip install -r requirements.txt --python venv/Scripts/python.exe`
- **Start stack:** `docker compose up --build`
- Async tests use `@pytest.mark.anyio` — only `asyncio` backend (trio not installed).
- `tests/conftest.py` patches Mongo startup functions AND `aioredis.from_url` so `TestClient` starts without live infrastructure.

## Next Recommended Phase
- Phase 5: Vulnerability KB store + read/write
  - `findings` collection: `project_id`, `run_id`, `persona_type`, `prompt_vector`, `agent_response_excerpt`, `finding_type`, `severity`
  - `KBWriter`: called post-run by evaluation engine, writes flagged behaviors
  - `KBReader`: called by `PersonaEngine` pre-session, returns top-N findings for project (wired but empty in Phase 3)
  - Finding types: `prompt_injection_success`, `boundary_violation`, `hallucination`, `inappropriate_response`, `refusal_failure`
  - Severity: `critical | high | medium | low`
  - Migration `v0004_findings.py`

## Notes for Future Sessions
- MongoDB instead of Postgres — intentional. No SQLAlchemy anywhere.
- Keep all DB shape/index changes in migration scripts, not ad-hoc startup logic.
- Preserve API key scoping behavior when adding new routes that reference `project_id`.
- `caller/agent_caller.py` is the single source for all agent-calling logic — do not create new caller modules.
- Phase 5 KB read path slot is already wired in `PersonaSession` as `raw_kb_findings=[]` — replace with live `KBReader` call.
- `evaluating` run status is reserved in the schema for Phase 6; runner currently skips straight to `complete`.
