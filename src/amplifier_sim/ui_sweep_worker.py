"""Isolated, resumable worker for one Streamlit parameter sweep."""

from __future__ import annotations

import argparse
import copy
import multiprocessing
import os
import pickle
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from queue import Empty
from typing import Any, Callable

import numpy as np

from .json_contract import dumps_json_portable
from .solver import AmplifierSimulator
from .sweep import (SweepGridRun, SweepRun, run_parameter_grid, run_parameter_sweep,
                    request_uses_continuation, sweep_x_indices)
from .ui_solve_worker import (
    _atomic_text,
    _continuation_state_from_archive_descriptor,
    _watch_manager_heartbeat,
    apply_ui_worker_resource_limits,
)


_PARALLEL_BASE_CONFIG: dict[str, Any] | None = None
_PARALLEL_REQUEST: dict[str, Any] | None = None
_PARALLEL_ADJUSTMENTS: list[dict[str, Any]] | None = None
_PARALLEL_POINTS_DIR: Path | None = None
_PARALLEL_PROGRESS_QUEUE: Any | None = None

_PARALLEL_PROGRESS_METRICS = {
    "thermal_optical_iteration",
    "thermal_optical_total",
    "fixed_point_iteration",
    "fixed_point_total",
    "pulse_index",
    "pulse_total",
    "iteration",
    "iterations",
}


def _atomic_pickle(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if not str(key).startswith("_")
    }


def _point_worker_count(request: dict[str, Any], total: int) -> int:
    """Return a bounded point-process count while preserving state continuation."""

    raw = request.get("point_workers", 1)
    if isinstance(raw, bool):
        raise ValueError("sweep point_workers must be an integer from 1 to 3")
    try:
        workers = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sweep point_workers must be an integer from 1 to 3"
        ) from exc
    if workers != raw or not 1 <= workers <= 3:
        raise ValueError("sweep point_workers must be an integer from 1 to 3")
    if request_uses_continuation(request):
        return 1
    return min(workers, max(int(total), 1))


def _parallel_progress_event(event: dict[str, Any]) -> dict[str, Any]:
    """Keep parallel live progress scalar-only and cheap to transfer."""

    compact = {
        key: copy.deepcopy(event[key])
        for key in (
            "sequence",
            "stage",
            "message",
            "state",
            "overall_fraction",
            "current",
            "total",
            "elapsed_s",
        )
        if key in event
    }
    metrics = event.get("metrics")
    if isinstance(metrics, dict):
        compact["metrics"] = {
            key: copy.deepcopy(value)
            for key, value in metrics.items()
            if key in _PARALLEL_PROGRESS_METRICS
        }
    return compact


def _task_record_context(
    task: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    contextual = _public_record(record)
    contextual["point_index"] = int(task["point_index"])
    if task["dimension"] == 2:
        contextual.update(
            {
                "parameter_x_index": int(task["x_index"]),
                "parameter_y_index": int(task["y_index"]),
                "parameter_x_value": task["value_x"],
                "parameter_y_value": task["value_y"],
            }
        )
    else:
        contextual["parameter_value"] = task["value_x"]
    return contextual


def _parallel_point_tasks(request: dict[str, Any]) -> list[dict[str, Any]]:
    dimension = int(request.get("dimension", 1))
    parameter_x = request["parameter"]
    values_x = np.asarray(request["values"], dtype=float)
    parameter_y = request.get("secondary_parameter")
    values_y = np.asarray(request.get("secondary_values", []), dtype=float)
    y_iterator: list[tuple[int | None, float | None]] = (
        list(enumerate(values_y)) if dimension == 2 else [(None, None)]
    )
    tasks: list[dict[str, Any]] = []
    for y_index, raw_y in y_iterator:
        value_y: int | float | None = None
        if dimension == 2 and parameter_y is not None and raw_y is not None:
            value_y = (
                int(round(float(raw_y)))
                if parameter_y.integer
                else float(raw_y)
            )
        for x_index, raw_x in enumerate(values_x):
            value_x: int | float = (
                int(round(float(raw_x)))
                if parameter_x.integer
                else float(raw_x)
            )
            tasks.append(
                {
                    "point_index": len(tasks),
                    "dimension": dimension,
                    "x_index": x_index,
                    "y_index": y_index,
                    "value_x": value_x,
                    "value_y": value_y,
                }
            )
    return tasks


def _initialize_parallel_point_worker(
    base_config: dict[str, Any],
    request: dict[str, Any],
    adjustments: list[dict[str, Any]],
    points_dir: str,
    progress_queue: Any,
) -> None:
    global _PARALLEL_BASE_CONFIG, _PARALLEL_REQUEST
    global _PARALLEL_ADJUSTMENTS, _PARALLEL_POINTS_DIR
    global _PARALLEL_PROGRESS_QUEUE

    _PARALLEL_BASE_CONFIG = copy.deepcopy(base_config)
    _PARALLEL_REQUEST = copy.deepcopy(request)
    _PARALLEL_ADJUSTMENTS = copy.deepcopy(adjustments)
    _PARALLEL_POINTS_DIR = Path(points_dir)
    _PARALLEL_PROGRESS_QUEUE = progress_queue


def _execute_parallel_point(task: dict[str, Any]) -> dict[str, Any]:
    """Run and persist one full-fidelity point inside an isolated process."""

    if (
        _PARALLEL_BASE_CONFIG is None
        or _PARALLEL_REQUEST is None
        or _PARALLEL_ADJUSTMENTS is None
        or _PARALLEL_POINTS_DIR is None
        or _PARALLEL_PROGRESS_QUEUE is None
    ):
        raise RuntimeError("parallel sweep worker was not initialized")
    request = _PARALLEL_REQUEST
    total = len(_parallel_point_tasks(request))
    point_number = int(task["point_index"]) + 1

    def point_progress(
        _local_point: int,
        _local_total: int,
        record: dict[str, Any],
        event: dict[str, Any],
    ) -> None:
        _PARALLEL_PROGRESS_QUEUE.put(
            {
                "event_type": "point_progress",
                "point": point_number,
                "total": total,
                "record": _task_record_context(task, record),
                "event": _parallel_progress_event(event),
            }
        )

    _PARALLEL_PROGRESS_QUEUE.put(
        {
            "event_type": "point_progress",
            "point": point_number,
            "total": total,
            "record": _task_record_context(
                task, {"status": "execution_failed"}
            ),
            "event": {
                "stage": "initialization",
                "message": "Preparing parallel sweep point",
                "state": "running",
                "overall_fraction": 0.0,
                "current": 0,
                "total": 1,
                "metrics": {},
            },
        }
    )

    if task["dimension"] == 2:
        local_run = run_parameter_grid(
            _PARALLEL_BASE_CONFIG,
            request["parameter"],
            [task["value_x"]],
            request["secondary_parameter"],
            [task["value_y"]],
            request["metric_keys"],
            point_progress=point_progress,
            progress_mode="milestones",
        )
    else:
        local_run = run_parameter_sweep(
            _PARALLEL_BASE_CONFIG,
            request["parameter"],
            [task["value_x"]],
            request["metric_keys"],
            point_progress=point_progress,
            progress_mode="milestones",
        )
    record = dict(local_run.records[0])
    record["point_index"] = int(task["point_index"])
    if task["dimension"] == 2:
        record.update(
            {
                "parameter_x_index": int(task["x_index"]),
                "parameter_y_index": int(task["y_index"]),
                "parameter_x_value": task["value_x"],
                "parameter_y_value": task["value_y"],
            }
        )
    else:
        record["parameter_value"] = task["value_x"]
    return _persist_point(
        _PARALLEL_POINTS_DIR,
        int(task["point_index"]),
        record,
        _PARALLEL_ADJUSTMENTS,
    )


def _drain_parallel_progress(
    progress_queue: Any,
    emit: Callable[[dict[str, Any]], None],
    *,
    wait_s: float = 0.0,
) -> int:
    drained = 0
    while True:
        try:
            wrapper = (
                progress_queue.get(timeout=wait_s)
                if drained == 0 and wait_s > 0.0
                else progress_queue.get_nowait()
            )
        except Empty:
            return drained
        emit(wrapper)
        drained += 1


def _run_parallel_points(
    base_config: dict[str, Any],
    request: dict[str, Any],
    adjustments: list[dict[str, Any]],
    points_dir: Path,
    workers: int,
    emit: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    tasks = _parallel_point_tasks(request)
    records: list[dict[str, Any] | None] = [None] * len(tasks)
    context = multiprocessing.get_context("spawn")
    progress_queue = context.Queue()
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_parallel_point_worker,
        initargs=(
            base_config,
            request,
            adjustments,
            str(points_dir),
            progress_queue,
        ),
    )
    futures = {
        executor.submit(_execute_parallel_point, task): task for task in tasks
    }
    pending = set(futures)
    try:
        while pending:
            _drain_parallel_progress(progress_queue, emit)
            done, pending = wait(
                pending,
                timeout=0.25,
                return_when=FIRST_COMPLETED,
            )
            _drain_parallel_progress(progress_queue, emit)
            for future in done:
                task = futures[future]
                record = future.result()
                point_index = int(task["point_index"])
                records[point_index] = record
                emit(
                    {
                        "event_type": "point_complete",
                        "point": point_index + 1,
                        "total": len(tasks),
                        "record": record,
                    }
                )
        # Queue feeder threads can deliver the last scalar status just after a
        # future returns. Completed-point events remain authoritative.
        _drain_parallel_progress(progress_queue, emit, wait_s=0.1)
    except BaseException:
        for future in pending:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        _drain_parallel_progress(progress_queue, emit)
        progress_queue.close()
        progress_queue.join_thread()
    if any(record is None for record in records):
        raise RuntimeError("parallel sweep completed without every point record")
    return [record for record in records if record is not None]


def _persist_point(
    points_dir: Path,
    point_index: int,
    record: dict[str, Any],
    compatibility_adjustments: list[dict[str, Any]],
) -> dict[str, Any]:
    result = record.get("_result")
    config = copy.deepcopy(record.get("_config", {}))
    persisted_record = _public_record(record)
    persisted_record["ui_compatibility_adjustments"] = copy.deepcopy(
        compatibility_adjustments
    )
    prefix = points_dir / f"{point_index:06d}"
    if result is not None:
        _atomic_pickle(prefix.with_suffix(".result.pkl"), result)
    # The metadata marker is written last.  Its presence therefore means the
    # point is complete and safe to expose through a partial download.
    _atomic_pickle(
        prefix.with_suffix(".meta.pkl"),
        {"record": persisted_record, "config": config},
    )
    return persisted_record


def _run_sequential_points(
    base_config: dict[str, Any],
    request: dict[str, Any],
    adjustments: list[dict[str, Any]],
    points_dir: Path,
    total: int,
    emit: Callable[[dict[str, Any]], None],
    *,
    initial_continuation_state: Any | None,
) -> list[dict[str, Any]]:
    """Preserve the original ordered full-result sweep path exactly."""

    records: list[dict[str, Any]] = []
    dimension = int(request.get("dimension", 1))
    values_x = np.asarray(request["values"], dtype=float)
    values_y = np.asarray(request.get("secondary_values", []), dtype=float)
    pump_continuation_state = initial_continuation_state
    continuation_enabled = request_uses_continuation(request)
    continuation_source = None
    point_index = 0
    y_iterator: list[tuple[int | None, float | None]] = (
        list(enumerate(values_y)) if dimension == 2 else [(None, None)]
    )
    for y_index, value_y in y_iterator:
        for x_index in sweep_x_indices(len(values_x), y_index or 0, continuation_enabled):
            value_x = values_x[x_index]
            point_number = point_index + 1

            def point_progress(
                _local_point: int,
                _local_total: int,
                record: dict[str, Any],
                event: dict[str, Any],
            ) -> None:
                display_record = _public_record(record)
                display_record["point_index"] = point_index
                if "continuation" in display_record:
                    display_record["continuation"]["source_point"] = continuation_source
                if dimension == 2:
                    display_record.update(
                        {
                            "parameter_x_index": x_index,
                            "parameter_y_index": y_index,
                        }
                    )
                emit(
                    {
                        "event_type": "point_progress",
                        "point": point_number,
                        "total": total,
                        "record": display_record,
                        "event": event,
                    }
                )

            if dimension == 2:
                local_run = run_parameter_grid(
                    base_config,
                    request["parameter"],
                    [value_x],
                    request["secondary_parameter"],
                    [value_y],
                    request["metric_keys"],
                    point_progress=point_progress,
                    progress_mode="milestones",
                    initial_continuation_state=pump_continuation_state,
                )
            else:
                local_run = run_parameter_sweep(
                    base_config,
                    request["parameter"],
                    [value_x],
                    request["metric_keys"],
                    point_progress=point_progress,
                    progress_mode="milestones",
                    initial_continuation_state=pump_continuation_state,
                )
            record = dict(local_run.records[0])
            if continuation_enabled:
                pump_continuation_state = record.get("_continuation_state")
                details = record.get("continuation", record.get("pump_continuation", {}))
                details["source_point"] = continuation_source
                if pump_continuation_state is None:
                    continuation_source = None
                if details.get("state_exported"):
                    continuation_source = dict(point_index=point_index, x_index=x_index,
                                               y_index=y_index, x_value=float(value_x),
                                               y_value=None if value_y is None else float(value_y))
            record["point_index"] = point_index
            if dimension == 2:
                record.update(
                    {
                        "parameter_x_index": x_index,
                        "parameter_y_index": y_index,
                    }
                )
            persisted = _persist_point(
                points_dir, point_index, record, adjustments
            )
            records.append(persisted)
            emit(
                {
                    "event_type": "point_complete",
                    "point": point_number,
                    "total": total,
                    "record": persisted,
                }
            )
            point_index += 1
    return records


def run_worker(job_dir: Path) -> int:
    with (job_dir / "request.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    base_config = payload["config"]
    apply_ui_worker_resource_limits(base_config)
    request = payload["request"]
    adjustments = list(payload.get("compatibility_adjustments", []))
    points_dir = job_dir / "points"
    points_dir.mkdir(exist_ok=True)

    watchdog = threading.Thread(
        target=_watch_manager_heartbeat,
        args=(job_dir / "manager.heartbeat",),
        kwargs={"terminate_process_group": True},
        daemon=True,
        name="amplifier-sweep-parent-watchdog",
    )
    watchdog.start()
    progress_stream = (job_dir / "progress.jsonl").open(
        "a", encoding="utf-8", buffering=1
    )

    def emit(event: dict[str, Any]) -> None:
        progress_stream.write(
            dumps_json_portable(event, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        progress_stream.flush()

    dimension = int(request.get("dimension", 1))
    values_x = np.asarray(request["values"], dtype=float)
    values_y = np.asarray(request.get("secondary_values", []), dtype=float)
    total = len(values_x) * (len(values_y) if dimension == 2 else 1)
    point_workers = _point_worker_count(request, total)
    pump_continuation_state: Any | None = None
    if (
        request_uses_continuation(request)
        and payload.get("initial_continuation_archive") is not None
    ):
        continuation_validator = AmplifierSimulator(copy.deepcopy(base_config))
        pump_continuation_state = _continuation_state_from_archive_descriptor(
            payload["initial_continuation_archive"],
            continuation_validator,
        )
    try:
        if point_workers > 1:
            records = _run_parallel_points(
                base_config,
                request,
                adjustments,
                points_dir,
                point_workers,
                emit,
            )
        else:
            records = _run_sequential_points(
                base_config,
                request,
                adjustments,
                points_dir,
                total,
                emit,
                initial_continuation_state=pump_continuation_state,
            )

        if dimension == 2:
            summary = SweepGridRun(
                request["parameter"].path,
                request["secondary_parameter"].path,
                values_x,
                values_y,
                tuple(records),
            )
        else:
            summary = SweepRun(
                request["parameter"].path,
                values_x,
                tuple(records),
            )
        _atomic_pickle(job_dir / "summary.pkl", summary)
        _atomic_text(job_dir / "outcome.json", {"status": "completed"})
        return 0
    except BaseException as exc:
        _atomic_text(
            job_dir / "outcome.json",
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1
    finally:
        progress_stream.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    arguments = parser.parse_args()
    return run_worker(arguments.job_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
