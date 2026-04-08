VERSION = 4
DESCRIPTION = "Add findings collection with indexes for KB read/write"


async def upgrade(db) -> None:
    # project_id + created_at — primary listing query (list findings by project, newest first)
    await db["findings"].create_index(
        [("project_id", 1), ("created_at", -1)],
        name="findings_project_created_idx",
    )
    # project_id + persona_type — KBReader query (fetch findings matching a persona type)
    await db["findings"].create_index(
        [("project_id", 1), ("persona_type", 1)],
        name="findings_project_persona_idx",
    )
    # project_id + severity — KBReader priority filter (always include critical findings)
    await db["findings"].create_index(
        [("project_id", 1), ("severity", 1)],
        name="findings_project_severity_idx",
    )
    # run_id — link findings back to the run that produced them
    await db["findings"].create_index(
        "run_id",
        name="findings_run_id_idx",
    )
    # finding_type — aggregate by type for reports (Phase 6)
    await db["findings"].create_index(
        "finding_type",
        name="findings_type_idx",
    )
