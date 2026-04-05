from db.migrations import discover_migrations


def test_initial_migration_is_discoverable() -> None:
    migrations = discover_migrations()

    assert migrations
    assert migrations[0].version == 1
    assert migrations[0].name == "v0001_init_settings"


def test_migrations_are_version_sorted() -> None:
    migrations = discover_migrations()

    versions = [migration.version for migration in migrations]
    assert versions == sorted(versions)
    assert versions[-1] == 2
