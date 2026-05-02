from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from typing import Any

from app.storage.runs_store import (
    TERMINAL_STATUSES,
    append_run_log,
    get_run,
    update_run_status,
)

RunCallable = Callable[..., dict[str, Any]]


class RunExecutor:
    def __init__(self, database_path: Path, max_workers: int = 1) -> None:
        self.database_path = database_path
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="galaxy-run")
        self._cancel_events: dict[str, Event] = {}
        self._lock = Lock()

    def submit(self, run_id: str, target: RunCallable, kwargs: dict[str, Any]) -> None:
        cancel_event = Event()
        with self._lock:
            self._cancel_events[run_id] = cancel_event
        self._pool.submit(self._run, run_id, target, kwargs, cancel_event)

    async def cancel(self, run_id: str) -> str:
        run = await get_run(self.database_path, run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        if run.status in TERMINAL_STATUSES:
            return run.status

        with self._lock:
            cancel_event = self._cancel_events.get(run_id)
            if cancel_event is not None:
                cancel_event.set()

        await append_run_log(self.database_path, run_id, "Cancellation requested.")
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
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _run(
        self,
        run_id: str,
        target: RunCallable,
        kwargs: dict[str, Any],
        cancel_event: Event,
    ) -> None:
        try:
            asyncio.run(update_run_status(self.database_path, run_id, "running"))
            asyncio.run(append_run_log(self.database_path, run_id, "Run started."))

            def log_callback(message: str) -> None:
                asyncio.run(append_run_log(self.database_path, run_id, message))

            result = target(**kwargs, cancel_event=cancel_event, log_callback=log_callback)
            if cancel_event.is_set():
                asyncio.run(
                    update_run_status(
                        self.database_path,
                        run_id,
                        "cancelled",
                        result.get("total_items"),
                        result.get("processed_items"),
                        "Task was cancelled.",
                        result.get("output_path"),
                    )
                )
                asyncio.run(append_run_log(self.database_path, run_id, "Run cancelled."))
            else:
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
