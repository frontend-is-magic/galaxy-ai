import os
import time

import anyio

from app.runtime.run_executor import RunExecutor
from app.storage.runs_store import RunCreate, create_run, get_run, list_run_logs


def progress_target(**kwargs):
    kwargs["progress_callback"](1)
    kwargs["progress_callback"](2)
    return {"total_items": 2, "processed_items": 2, "output_path": "/tmp/out"}


def short_blocking_target(**_kwargs):
    time.sleep(0.2)
    return {}


def marker_target(**kwargs):
    kwargs["marker_path"].write_text("executed", encoding="utf-8")
    return {}


def wait_for_marker_target(**kwargs):
    deadline = time.monotonic() + 2
    while not kwargs["release_path"].exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    return {}


def blocked_target(**_kwargs):
    while True:
        time.sleep(1)


def hard_exit_target(**_kwargs):
    os._exit(11)


def test_run_executor_progress_callback_updates_processed_items(tmp_path):
    database_path = tmp_path / "galaxy.sqlite3"
    run = anyio.run(
        create_run,
        database_path,
        RunCreate(
            run_type="image_classification_inference",
            request={},
            hardware_backend="cpu",
            model_ref="local-model",
            total_items=2,
        ),
    )
    executor = RunExecutor(database_path)
    assert executor._mp_context.get_start_method() == "spawn"

    try:
        executor.submit(run.run_id, progress_target, {})
        for _ in range(50):
            stored = anyio.run(get_run, database_path, run.run_id)
            if stored is not None and stored.status == "completed":
                break
            time.sleep(0.01)
    finally:
        executor.shutdown()

    stored = anyio.run(get_run, database_path, run.run_id)
    assert stored is not None
    assert stored.status == "completed"
    assert stored.processed_items == 2


def test_run_executor_cancelled_queued_run_does_not_execute_target(tmp_path):
    database_path = tmp_path / "galaxy.sqlite3"
    first_run = anyio.run(
        create_run,
        database_path,
        RunCreate(
            run_type="image_classification_inference",
            request={},
            hardware_backend="cpu",
        ),
    )
    queued_run = anyio.run(
        create_run,
        database_path,
        RunCreate(
            run_type="image_classification_training",
            request={},
            hardware_backend="cpu",
        ),
    )
    executor = RunExecutor(database_path, max_workers=1)
    queued_marker = tmp_path / "queued-target-executed"

    try:
        executor.submit(first_run.run_id, short_blocking_target, {})
        executor.submit(queued_run.run_id, marker_target, {"marker_path": queued_marker})
        for _ in range(50):
            stored = anyio.run(get_run, database_path, first_run.run_id)
            if stored is not None and stored.status == "running":
                break
            time.sleep(0.01)
        status = anyio.run(executor.cancel, queued_run.run_id)
        for _ in range(100):
            stored = anyio.run(get_run, database_path, queued_run.run_id)
            if stored is not None and stored.status == "cancelled":
                break
            time.sleep(0.01)
    finally:
        executor.shutdown()

    stored = anyio.run(get_run, database_path, queued_run.run_id)
    assert status in {"cancelling", "cancelled"}
    assert stored is not None
    assert stored.status == "cancelled"
    assert not queued_marker.exists()


def test_run_executor_cancel_is_idempotent_for_cancelling_run(tmp_path):
    database_path = tmp_path / "galaxy.sqlite3"
    run = anyio.run(
        create_run,
        database_path,
        RunCreate(
            run_type="image_classification_inference",
            request={},
            hardware_backend="cpu",
        ),
    )
    executor = RunExecutor(database_path)
    release_run = tmp_path / "release-run"

    try:
        executor.submit(run.run_id, wait_for_marker_target, {"release_path": release_run})
        first_status = anyio.run(executor.cancel, run.run_id)
        second_status = anyio.run(executor.cancel, run.run_id)
        release_run.write_text("release", encoding="utf-8")
        for _ in range(50):
            stored = anyio.run(get_run, database_path, run.run_id)
            if stored is not None and stored.status == "cancelled":
                break
            time.sleep(0.01)
    finally:
        executor.shutdown()

    logs = anyio.run(list_run_logs, database_path, run.run_id)
    assert first_status == "cancelling"
    assert second_status == "cancelling"
    assert logs.count("Cancellation requested.") == 1


def test_run_executor_cancel_terminates_blocked_running_target(tmp_path):
    database_path = tmp_path / "galaxy.sqlite3"
    run = anyio.run(
        create_run,
        database_path,
        RunCreate(
            run_type="image_classification_inference",
            request={},
            hardware_backend="cpu",
        ),
    )
    executor = RunExecutor(database_path)

    try:
        executor.submit(run.run_id, blocked_target, {})
        for _ in range(50):
            stored = anyio.run(get_run, database_path, run.run_id)
            if stored is not None and stored.status == "running":
                break
            time.sleep(0.01)
        status = anyio.run(executor.cancel, run.run_id)
        for _ in range(50):
            stored = anyio.run(get_run, database_path, run.run_id)
            if stored is not None and stored.status == "cancelled":
                break
            time.sleep(0.01)
    finally:
        executor.shutdown()

    stored = anyio.run(get_run, database_path, run.run_id)
    assert status in {"cancelling", "cancelled"}
    assert stored is not None
    assert stored.status == "cancelled"


def test_run_executor_reports_hard_process_exit_with_crash_log(tmp_path):
    database_path = tmp_path / "galaxy.sqlite3"
    run = anyio.run(
        create_run,
        database_path,
        RunCreate(
            run_type="image_classification_training",
            request={},
            hardware_backend="cpu",
        ),
    )
    executor = RunExecutor(database_path)

    try:
        executor.submit(run.run_id, hard_exit_target, {})
        for _ in range(100):
            stored = anyio.run(get_run, database_path, run.run_id)
            if stored is not None and stored.status == "error":
                break
            time.sleep(0.01)
    finally:
        executor.shutdown()

    stored = anyio.run(get_run, database_path, run.run_id)
    logs = anyio.run(list_run_logs, database_path, run.run_id)
    assert stored is not None
    assert stored.status == "error"
    assert stored.error_message == "Run process exited with code 11."
    assert any("Crash log:" in log for log in logs)
    assert (database_path.parent / "run-crashes" / f"{run.run_id}.log").exists()
