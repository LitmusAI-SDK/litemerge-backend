from datetime import datetime, timezone

VERSION = 2
DESCRIPTION = "Add projects and api_keys indexes plus baseline key fields"


async def upgrade(db) -> None:
    now = datetime.now(timezone.utc)

    await db["projects"].create_index(
        [("owner_id", 1), ("created_at", -1)], name="owner_created_at_idx"
    )
    await db["projects"].create_index("name", name="project_name_idx")

    await db["api_keys"].create_index("key_hash", unique=True, name="uniq_api_key_hash")
    await db["api_keys"].create_index("project_ids", name="api_key_project_ids_idx")

    await db["api_keys"].update_many(
        {},
        [
            {
                "$set": {
                    "project_ids": {"$ifNull": ["$project_ids", []]},
                    "scopes": {"$ifNull": ["$scopes", []]},
                    "is_active": {"$ifNull": ["$is_active", True]},
                    "created_at": {"$ifNull": ["$created_at", now]},
                    "updated_at": now,
                }
            }
        ],
    )
