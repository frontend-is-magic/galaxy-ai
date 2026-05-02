from __future__ import annotations

import asyncio
import faulthandler
import multiprocessing
import queue
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from app.storage.runs_store import (
    TERMINAL_STATUSES,
    append_run_log,
    get_run,
    update_run_progress,
    update_run_status,
)

RunCallable = Callable[..., dict[str, Any]]


def _run_target_process(
    database_path: Path,
    run_id: str,
    target: RunCallable,
    kwargs: dict[str, Any],
    cancel_event: Any,
    result_queue: Any,
    crash_log_path: Path,
) -> None:
    crash_log_path.parent.mkdir(parents=True, exist_ok=True)
    crash_log = crash_log_path.open("w", encoding="utf-8")
    faulthandler.enable(file=crash_log, all_threads=True)

    def log_callback(message: str) -> None:
        asyncio.run(append_run_log(database_path, run_id, message))

    def progress_callback(processed_items: int, total_items: int | None = None) -> None:
        asyncio.run(
            update_run_progress(
                database_path,
                run_id,
                processed_items,
                total_items,
            )
        )

    try:
        result = target(
            **kwargs,
            cancel_event=cancel_event,
            log_callback=log_callback,
            progress_callback=progress_callback,
        )
        result_queue.put({"status": "ok", "result": result})
    except BaseException as exc:  # noqa: BLE001
        result_queue.put({"status": "error", "error": str(exc)})
    finally:
        faulthandler.disable()
        crash_log.close()


class RunExecutor:
    def __init__(self, database_path: Path, max_workers: int = 1) -> None:
        self.database_path = database_path
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="galaxy-run")
        self._mp_context = multiprocessing.get_context("spawn")
        self._cancel_events: dict[str, Any] = {}
        self._processes: dict[str, Any] = {}
        self._lock = Lock()

    def submit(self, run_id: str, target: RunCallable, kwargs: dict[str, Any]) -> None:
        cancel_event = self._mp_context.Event()
        with self._lock:
            self._cancel_events[run_id] = cancel_event
        self._pool.submit(self._run, run_id, target, kwargs, cancel_event)

    async def cancel(self, run_id: str) -> str:
        run = await get_run(self.database_path, run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        if run.status in TERMINAL_STATUSES:
            return run.status
        if run.status == "cancelling":
            return run.status

        with self._lock:
            cancel_event = self._cancel_events.get(run_id)
            if cancel_event is not None:
                cancel_event.set()
            process = self._processes.get(run_id)

        await append_run_log(self.database_path, run_id, "Cancellation requested.")
        if process is not None and process.is_alive():
            process.terminate()
        if cancel_event is None:
            updated = await update_run_status(
                self.database_path,
                run_id,
                "cancelled",
                error_message="Task was cancelled.",
            )
            await append_run_log(self.database_path, run_id, "Run cancelled.")
            return updated.status

        updated = await update_run_status(self.database_path, run_id, "cancelling")
        return updated.status

    def shutdown(self) -> None:
        with self._lock:
            processes = list(self._processes.values())
            cancel_events = list(self._cancel_events.values())
        for cancel_event in cancel_events:
            cancel_event.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _run(
        self,
        run_id: str,
        target: RunCallable,
        kwargs: dict[str, Any],
        cancel_event: Any,
    ) -> None:
        try:
            if cancel_event.is_set():
                asyncio.run(
                    update_run_status(
                        self.database_path,
                        run_id,
                        "cancelled",
                        error_message="Task was cancelled.",
                    )
                )
                asyncio.run(append_run_log(self.database_path, run_id, "Run cancelled."))
                return

            asyncio.run(update_run_status(self.database_path, run_id, "running"))
            asyncio.run(append_run_log(self.database_path, run_id, "Run started."))

            result_queue = self._mp_context.Queue()
            crash_log_path = self.database_path.parent / "run-crashes" / f"{run_id}.log"
            process = self._mp_context.Process(
                target=_run_target_process,
                args=(
                    self.database_path,
                    run_id,
                    target,
                    kwargs,
                    cancel_event,
                    result_queue,
                    crash_log_path,
                ),
            )
            with self._lock:
                self._processes[run_id] = process
            process.start()

            while process.is_alive():
                process.join(timeout=0.1)
                if cancel_event.is_set() and process.is_alive():
                    process.terminate()

            if cancel_event.is_set():
                process.join(timeout=1)
                if process.is_alive():
                    kill = getattr(process, "kill", None)
                    if kill is not None:
                        kill()
                    else:
                        process.terminate()
                    process.join(timeout=1)
                cancelled_result: dict[str, Any] = {}
                try:
                    cancelled_payload = result_queue.get_nowait()
                    if cancelled_payload["status"] == "ok":
                        cancelled_result = cancelled_payload["result"]
                except queue.Empty:
                    pass
                asyncio.run(
                    update_run_status(
                        self.database_path,
                        run_id,
                        "cancelled",
                        cancelled_result.get("total_items"),
                        cancelled_result.get("processed_items"),
                        "Task was cancelled.",
                        cancelled_result.get("output_path"),
                    )
                )
                asyncio.run(append_run_log(self.database_path, run_id, "Run cancelled."))
                return

            try:
                payload = result_queue.get_nowait()
            except queue.Empty:
                asyncio.run(
                    append_run_log(self.database_path, run_id, f"Crash log: {crash_log_path}")
                )
                payload = {
                    "status": "error",
                    "error": f"Run process exited with code {process.exitcode}.",
                }

            if payload["status"] == "ok":
                result = payload["result"]
                asyncio.run(
                    update_run_status(
                        self.database_path,
                        run_id,
                        "completed",
                        result.get("total_items"),
                        result.get("processed_items"),
                        None,
                        result.get("output_path"),
                    )
                )
                asyncio.run(append_run_log(self.database_path, run_id, "Run completed."))
            else:
                asyncio.run(
                    update_run_status(
                        self.database_path,
                        run_id,
                        "error",
                        error_message=payload["error"],
                    )
                )
                asyncio.run(
                    append_run_log(self.database_path, run_id, f"Run failed: {payload['error']}")
                )
        except Exception as exc:
            if cancel_event.is_set():
                asyncio.run(
                    update_run_status(
                        self.database_path,
                        run_id,
                        "cancelled",
                        error_message="Task was cancelled.",
                    )
                )
                asyncio.run(append_run_log(self.database_path, run_id, "Run cancelled."))
            else:
                asyncio.run(
                    update_run_status(
                        self.database_path,
                        run_id,
                        "error",
                        error_message=str(exc),
                    )
                )
                asyncio.run(append_run_log(self.database_path, run_id, f"Run failed: {exc}"))
        finally:
            with self._lock:
                self._cancel_events.pop(run_id, None)
                self._processes.pop(run_id, None)
