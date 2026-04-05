from datetime import datetime, timezone

VERSION = 1
DESCRIPTION = "Create singleton settings document for MongoDB schema metadata"

SETTINGS_COLLECTION = "settings"
SETTINGS_DOC_ID = "litmusai_settings"


async def upgrade(db) -> None:
    now = datetime.now(timezone.utc)
    await db[SETTINGS_COLLECTION].update_one(
        {"_id": SETTINGS_DOC_ID},
        {
            "$setOnInsert": {
                "singleton_key": "global",
                "db_version": 0,
                "last_migration": None,
                "applied_migrations": [],
                "meta": {},
                "created_at": now,
            },
            "$set": {"updated_at": now},
        },
        upsert=True,
    )
