from db.migrations import discover_migrations


def test_initial_migration_is_discoverable() -> None:
    migrations = discover_migrations()

    assert migrations
    assert migrations[0].version == 1
    assert migrations[0].name == "v0001_init_settings"
