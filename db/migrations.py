from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import inspect
from pathlib import Path
from typing import Awaitable, Callable

from motor.motor_asyncio import AsyncIOMotorDatabase

SETTINGS_COLLECTION = "settings"
SETTINGS_DOC_ID = "litmusai_settings"
MIGRATIONS_DIR = Path(__file__).resolve().parent / "versions"


@dataclass(frozen=True)
class MigrationScript:
    version: int
    name: str
    upgrade: Callable[[AsyncIOMotorDatabase], Awaitable[None]]


def _load_migration_script(path: Path) -> MigrationScript:
    module_name = f"mongo_migration_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load migration module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    version = getattr(module, "VERSION", None)
    upgrade = getattr(module, "upgrade", None)

    if not isinstance(version, int):
        raise RuntimeError(f"Migration {path.name} must define integer VERSION")

    if not inspect.iscoroutinefunction(upgrade):
        raise RuntimeError(f"Migration {path.name} must define async upgrade(db)")

    return MigrationScript(version=version, name=path.stem, upgrade=upgrade)


def discover_migrations(migrations_dir: Path = MIGRATIONS_DIR) -> list[MigrationScript]:
    if not migrations_dir.exists():
        return []

    scripts = [
        _load_migration_script(path)
        for path in sorted(migrations_dir.glob("*.py"))
        if not path.name.startswith("_")
    ]
    scripts.sort(key=lambda script: script.version)

    seen_versions: set[int] = set()
    for script in scripts:
        if script.version in seen_versions:
            raise RuntimeError(f"Duplicate migration version detected: {script.version}")
        seen_versions.add(script.version)

    return scripts


def _base_settings_document(now: datetime) -> dict:
    return {
        "singleton_key": "global",
        "db_version": 0,
        "last_migration": None,
        "applied_migrations": [],
        "meta": {},
        "created_at": now,
        "updated_at": now,
    }


async def ensure_settings_document(db: AsyncIOMotorDatabase) -> None:
    now = datetime.now(timezone.utc)
    await db[SETTINGS_COLLECTION].update_one(
        {"_id": SETTINGS_DOC_ID},
        {"$setOnInsert": _base_settings_document(now)},
        upsert=True,
    )


async def run_migrations(db: AsyncIOMotorDatabase) -> list[str]:
    await ensure_settings_document(db)

    settings = await db[SETTINGS_COLLECTION].find_one({"_id": SETTINGS_DOC_ID})
    current_version = int(settings.get("db_version", 0)) if settings else 0

    applied: list[str] = []
    for script in discover_migrations():
        if script.version <= current_version:
            continue

        await script.upgrade(db)
        now = datetime.now(timezone.utc)
        await db[SETTINGS_COLLECTION].update_one(
            {"_id": SETTINGS_DOC_ID},
            {
                "$set": {
                    "db_version": script.version,
                    "last_migration": script.name,
                    "updated_at": now,
                },
                "$addToSet": {"applied_migrations": script.name},
            },
            upsert=True,
        )

        current_version = script.version
        applied.append(script.name)

    return applied
