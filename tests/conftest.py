from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


@pytest.fixture(autouse=True)
def mock_db_startup():
    """Prevent TestClient lifespan from blocking on real MongoDB connections.

    Patches run_migrations, init_indexes, and bootstrap_api_key to no-ops so
    that TestClient(app) starts instantly in environments without MongoDB.
    app.state.db_ready stays False by default; tests that need routes behind
    the auth middleware must flip it to True and supply a mock db themselves.
    """
    with (
        patch("api.main.run_migrations", new=AsyncMock(return_value=[])),
        patch("api.main.init_indexes", new=AsyncMock()),
        patch("api.main.bootstrap_api_key", new=AsyncMock()),
    ):
        yield
