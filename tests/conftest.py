from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


@pytest.fixture(autouse=True)
def mock_db_startup():
    """Prevent TestClient lifespan from blocking on real MongoDB/Redis connections.

    Patches run_migrations, init_indexes, bootstrap_api_key, and aioredis.from_url
    to no-ops so that TestClient(app) starts instantly without live infrastructure.
    """
    mock_redis = MagicMock()
    mock_redis.aclose = AsyncMock()

    with (
        patch("api.main.run_migrations", new=AsyncMock(return_value=[])),
        patch("api.main.init_indexes", new=AsyncMock()),
        patch("api.main.bootstrap_api_key", new=AsyncMock()),
        patch("api.main.aioredis.from_url", return_value=mock_redis),
    ):
        yield
