VERSION = 3
DESCRIPTION = "Add indexes for chat_logs collection"


async def upgrade(db) -> None:
    await db["chat_logs"].create_index(
        "run_id",
        name="chat_logs_run_id_idx",
    )
    await db["chat_logs"].create_index(
        [("run_id", 1), ("persona_id", 1)],
        unique=True,
        name="chat_logs_run_persona_idx",
    )
    # Optional TTL — uncomment to activate automatic log expiry after 90 days
    # await db["chat_logs"].create_index(
    #     "completed_at",
    #     expireAfterSeconds=7776000,
    #     name="chat_logs_ttl_idx",
    # )
