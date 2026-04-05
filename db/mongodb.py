from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from core.security import hash_api_key


def create_mongo_client(mongodb_url: str) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(mongodb_url, serverSelectionTimeoutMS=5000)


def get_database(client: AsyncIOMotorClient, db_name: str) -> AsyncIOMotorDatabase:
    return client[db_name]


async def init_indexes(db: AsyncIOMotorDatabase) -> None:
    await db["api_keys"].create_index("key_hash", unique=True, name="uniq_api_key_hash")
    await db["runs"].create_index("run_id", unique=True, name="uniq_run_id")
    await db["runs"].create_index(
        [("project_id", 1), ("created_at", -1)], name="project_created_at_idx"
    )


async def bootstrap_api_key(db: AsyncIOMotorDatabase, raw_api_key: str) -> None:
    if not raw_api_key:
        return

    key_hash = hash_api_key(raw_api_key)
    now = datetime.now(timezone.utc)

    await db["api_keys"].update_one(
        {"key_hash": key_hash},
        {
            "$setOnInsert": {
                "key_hash": key_hash,
                "name": "bootstrap-local-key",
                "project_ids": [],
                "scopes": ["runs:create", "runs:read"],
                "is_active": True,
                "created_at": now,
            }
        },
        upsert=True,
    )
