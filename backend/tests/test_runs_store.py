from app.storage.database import initialize_database
from app.storage.runs_store import (
    RunCreate,
    append_run_log,
    create_run,
    get_run,
    list_run_logs,
    mark_unfinished_runs_interrupted,
    update_run_status,
)


async def _create_sample_run(database_path):
    await initialize_database(database_path)
    return await create_run(
        database_path,
        RunCreate(
            run_type="image_classification_inference",
            status="queued",
            request={"model_ref": "local-model"},
            hardware_backend="cpu",
            model_ref="local-model",
            input_path="/tmp/input",
            output_path="/tmp/output",
        ),
    )


def test_runs_store_persists_status_and_logs(tmp_path):
    database_path = tmp_path / "galaxy.sqlite3"

    import anyio

    run = anyio.run(_create_sample_run, database_path)
    anyio.run(append_run_log, database_path, run.run_id, "started")
    anyio.run(
        update_run_status,
        database_path,
        run.run_id,
        "completed",
        3,
        3,
        None,
        "/tmp/output",
    )

    stored = anyio.run(get_run, database_path, run.run_id)
    logs = anyio.run(list_run_logs, database_path, run.run_id)

    assert stored is not None
    assert stored.status == "completed"
    assert stored.total_items == 3
    assert stored.processed_items == 3
    assert logs == ["started"]


def test_unfinished_runs_are_marked_interrupted_on_startup(tmp_path):
    database_path = tmp_path / "galaxy.sqlite3"

    import anyio

    run = anyio.run(_create_sample_run, database_path)
    anyio.run(mark_unfinished_runs_interrupted, database_path)

    stored = anyio.run(get_run, database_path, run.run_id)

    assert stored is not None
    assert stored.status == "interrupted"
    assert stored.error_message == "Task was interrupted by backend shutdown."
