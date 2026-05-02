from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiosqlite
from pydantic import BaseModel

from app.storage.database import initialize_database

TERMINAL_STATUSES = {"completed", "error", "cancelled", "interrupted"}
UNFINISHED_STATUSES = {"queued", "running", "cancelling"}
IMAGE_CLASSIFICATION_RUN_TYPES = (
    "image_classification_inference",
    "image_classification_training",
)


class ActiveRunExistsError(ValueError):
    def __init__(self, active_run: RunRecord) -> None:
        self.active_run = active_run
        super().__init__(
            "Another image classification task is already active: "
            f"{active_run.run_id} ({active_run.run_type}). "
            "Cancel it or wait for it to finish."
        )


class RunCreate(BaseModel):
    run_type: str
    status: str = "queued"
    request: dict[str, object]
    hardware_backend: str
    model_ref: str | None = None
    input_path: str | None = None
    output_path: str | None = None
    total_items: int = 0
    processed_items: int = 0


class RunRecord(BaseModel):
    run_id: str
    run_type: str
    status: str
    request: dict[str, object]
    hardware_backend: str
    model_ref: str | None
    input_path: str | None
    output_path: str | None
    total_items: int
    processed_items: int
    error_message: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_run(row: aiosqlite.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        run_type=row["run_type"],
        status=row["status"],
        request=json.loads(row["request_json"]),
        hardware_backend=row["hardware_backend"],
        model_ref=row["model_ref"],
        input_path=row["input_path"],
        output_path=row["output_path"],
        total_items=row["total_items"],
        processed_items=row["processed_items"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


def _placeholders(values: tuple[str, ...] | set[str]) -> str:
    return ", ".join("?" for _ in values)


async def create_run(database_path: Path, create: RunCreate) -> RunRecord:
    await initialize_database(database_path)
    run_id = str(uuid4())
    now = utc_now()
    async with aiosqlite.connect(database_path) as connection:
        await connection.execute(
            """
            INSERT INTO runs (
                run_id, run_type, status, request_json, hardware_backend, model_ref,
                input_path, output_path, total_items, processed_items, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                create.run_type,
                create.status,
                json.dumps(create.request, sort_keys=True),
                create.hardware_backend,
                create.model_ref,
                create.input_path,
                create.output_path,
                create.total_items,
                create.processed_items,
                now,
                now,
            ),
        )
        await connection.commit()

    run = await get_run(database_path, run_id)
    if run is None:
        raise RuntimeError("Created run could not be loaded.")
    return run


async def get_active_run(
    database_path: Path,
    run_types: tuple[str, ...] = IMAGE_CLASSIFICATION_RUN_TYPES,
) -> RunRecord | None:
    await initialize_database(database_path)
    async with aiosqlite.connect(database_path) as connection:
        connection.row_factory = aiosqlite.Row
        query = f"""
            SELECT *
            FROM runs
            WHERE run_type IN ({_placeholders(run_types)})
              AND status IN ({_placeholders(UNFINISHED_STATUSES)})
            ORDER BY created_at ASC
            LIMIT 1
            """
        async with connection.execute(
            query,
            (*run_types, *UNFINISHED_STATUSES),
        ) as cursor:
            row = await cursor.fetchone()

    if row is None:
        return None
    return _row_to_run(row)


async def create_run_with_single_active_lock(
    database_path: Path,
    create: RunCreate,
    run_types: tuple[str, ...] = IMAGE_CLASSIFICATION_RUN_TYPES,
) -> RunRecord:
    await initialize_database(database_path)
    run_id = str(uuid4())
    now = utc_now()

    async with aiosqlite.connect(database_path) as connection:
        connection.row_factory = aiosqlite.Row
        await connection.execute("BEGIN IMMEDIATE")
        try:
            active_query = f"""
                SELECT *
                FROM runs
                WHERE run_type IN ({_placeholders(run_types)})
                  AND status IN ({_placeholders(UNFINISHED_STATUSES)})
                ORDER BY created_at ASC
                LIMIT 1
                """
            async with connection.execute(
                active_query,
                (*run_types, *UNFINISHED_STATUSES),
            ) as cursor:
                active_row = await cursor.fetchone()

            if active_row is not None:
                await connection.rollback()
                raise ActiveRunExistsError(_row_to_run(active_row))

            await connection.execute(
                """
                INSERT INTO runs (
                    run_id, run_type, status, request_json, hardware_backend, model_ref,
                    input_path, output_path, total_items, processed_items, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    create.run_type,
                    create.status,
                    json.dumps(create.request, sort_keys=True),
                    create.hardware_backend,
                    create.model_ref,
                    create.input_path,
                    create.output_path,
                    create.total_items,
                    create.processed_items,
                    now,
                    now,
                ),
            )
            await connection.commit()
        except Exception:
            if connection.in_transaction:
                await connection.rollback()
            raise

    run = await get_run(database_path, run_id)
    if run is None:
        raise RuntimeError("Created run could not be loaded.")
    return run


async def get_run(database_path: Path, run_id: str) -> RunRecord | None:
    await initialize_database(database_path)
    async with aiosqlite.connect(database_path) as connection:
        connection.row_factory = aiosqlite.Row
        async with connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)) as cursor:
            row = await cursor.fetchone()

    if row is None:
        return None
    return _row_to_run(row)


async def update_run_status(
    database_path: Path,
    run_id: str,
    status: str,
    total_items: int | None = None,
    processed_items: int | None = None,
    error_message: str | None = None,
    output_path: str | None = None,
) -> RunRecord:
    now = utc_now()
    assignments = ["status = ?", "updated_at = ?"]
    values: list[object] = [status, now]

    if status == "running":
        assignments.append("started_at = COALESCE(started_at, ?)")
        values.append(now)
    if status in TERMINAL_STATUSES:
        assignments.append("completed_at = COALESCE(completed_at, ?)")
        values.append(now)
    if total_items is not None:
        assignments.append("total_items = ?")
        values.append(total_items)
    if processed_items is not None:
        assignments.append("processed_items = ?")
        values.append(processed_items)
    if error_message is not None:
        assignments.append("error_message = ?")
        values.append(error_message)
    if output_path is not None:
        assignments.append("output_path = ?")
        values.append(output_path)

    values.append(run_id)
    await initialize_database(database_path)
    async with aiosqlite.connect(database_path) as connection:
        await connection.execute(
            f"UPDATE runs SET {', '.join(assignments)} WHERE run_id = ?",
            values,
        )
        await connection.commit()

    run = await get_run(database_path, run_id)
    if run is None:
        raise ValueError(f"Run not found: {run_id}")
    return run


async def append_run_log(database_path: Path, run_id: str, message: str) -> None:
    await initialize_database(database_path)
    async with aiosqlite.connect(database_path) as connection:
        await connection.execute(
            "INSERT INTO run_logs (run_id, message) VALUES (?, ?)",
            (run_id, message),
        )
        await connection.commit()


async def list_run_logs(database_path: Path, run_id: str) -> list[str]:
    await initialize_database(database_path)
    async with aiosqlite.connect(database_path) as connection:
        async with connection.execute(
            "SELECT message FROM run_logs WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ) as cursor:
            rows = await cursor.fetchall()

    return [row[0] for row in rows]


async def mark_unfinished_runs_interrupted(database_path: Path) -> None:
    await initialize_database(database_path)
    now = utc_now()
    async with aiosqlite.connect(database_path) as connection:
        await connection.execute(
            """
            UPDATE runs
            SET status = 'interrupted',
                error_message = 'Task was interrupted by backend shutdown.',
                updated_at = ?,
                completed_at = COALESCE(completed_at, ?)
            WHERE status IN ('queued', 'running', 'cancelling')
            """,
            (now, now),
        )
        await connection.commit()
