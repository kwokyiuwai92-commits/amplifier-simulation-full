"""Isolated worker process for one Streamlit single-point solve.

The browser process owns lifecycle control while this module owns only the
physical calculation.  Keeping Streamlit out of the worker means ordinary UI
reruns cannot interrupt or mutate an in-flight solve.
"""

from __future__ import annotations

import argparse
import os
import pickle
import signal
import threading
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .json_contract import dumps_json_portable
from .result_archive import (
    ArchivePoint,
    diagnostic_record_for_result,
    load_result_archive_point,
    python_warning_record,
    write_result_archive,
)
from .solver import AmplifierSimulator, SimulationContinuationState


def _continuation_state_from_result(
    result: Any,
    simulator: AmplifierSimulator,
) -> SimulationContinuationState:
    """Convert one archived result into a validated numerical warm start.

    Result archives intentionally store physical results rather than private
    solver objects.  The reported population is therefore used as a nearby
    periodic-map guess; the receiving simulator still closes every equation
    (including any newly enabled free-carrier state) before reporting a result.
    """

    if list(result.level_names) != list(simulator.levels.names):
        raise ValueError(
            "warm-start result level names do not match the current configuration"
        )
    for axis in ("x", "y", "z"):
        archived = np.asarray(getattr(result.grid, axis), dtype=float)
        current = np.asarray(getattr(simulator.grid, axis), dtype=float)
        if archived.shape != current.shape or not np.allclose(
            archived,
            current,
            rtol=64.0 * np.finfo(float).eps,
            atol=64.0
            * np.finfo(float).eps
            * max(
                float(np.max(np.abs(current))),
                float(np.ptp(current)),
                np.finfo(float).tiny,
            ),
        ):
            raise ValueError(
                f"warm-start result {axis}-grid does not match the current configuration"
            )
    stress = result.stress or {}
    return SimulationContinuationState(
        periodic_boundary_population_m3=np.asarray(
            result.population_m3, dtype=float
        ).copy(),
        thermal_delta_n=np.asarray(result.delta_n, dtype=float).copy(),
        response_temperature_K=np.asarray(
            result.temperature_K, dtype=float
        ).copy(),
        thermal_temperature_K=np.asarray(
            result.temperature_K, dtype=float
        ).copy(),
        front_surface_sag_m=(
            None
            if stress.get("front_surface_sag_m") is None
            else np.asarray(stress["front_surface_sag_m"], dtype=float).copy()
        ),
        back_surface_sag_m=(
            None
            if stress.get("back_surface_sag_m") is None
            else np.asarray(stress["back_surface_sag_m"], dtype=float).copy()
        ),
    )


def _install_archive_continuation(
    simulator: AmplifierSimulator,
    descriptor: dict[str, Any] | None,
) -> None:
    state = _continuation_state_from_archive_descriptor(descriptor, simulator)
    if state is not None:
        simulator.set_continuation_state(state)


def _continuation_state_from_archive_descriptor(
    descriptor: dict[str, Any] | None,
    simulator: AmplifierSimulator,
) -> SimulationContinuationState | None:
    """Load one archived point as a validated continuation state."""

    if descriptor is None:
        return None
    point = load_result_archive_point(
        Path(str(descriptor["archive_path"])),
        int(descriptor["point_index"]),
    )
    if point.result is None:
        raise ValueError("the selected warm-start archive point has no full result")
    return _continuation_state_from_result(point.result, simulator)


def apply_ui_worker_resource_limits(config: dict[str, Any]) -> None:
    """Apply an explicitly requested FFT/kernel cap inside a UI worker."""

    raw_limit = os.environ.get("AMPLIFIER_SIM_UI_FFT_WORKERS")
    if raw_limit is None:
        return
    try:
        limit = max(int(raw_limit), 1)
    except ValueError:
        return
    performance = config.setdefault("performance", {})
    configured_fft = int(performance.get("fft_workers", 0))
    performance["fft_workers"] = (
        limit if configured_fft == 0 else min(configured_fft, limit)
    )
    transient = (
        config.setdefault("seed", {})
        .setdefault("coherent", {})
        .setdefault("transient_gain", {})
    )
    configured_kernel = int(
        transient.get("single_transition_kernel_workers", 0)
    )
    transient["single_transition_kernel_workers"] = (
        limit if configured_kernel == 0 else min(configured_kernel, limit)
    )


def _atomic_text(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        dumps_json_portable(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _autosave_result_archive(
    *,
    output_directory: Path,
    job_id: str,
    config_digest: str,
    config: dict[str, Any],
    result: Any,
    python_warnings: list[dict[str, Any]],
    compatibility_adjustments: list[dict[str, Any]],
) -> Path:
    """Atomically persist one complete UI result outside the job directory."""

    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    final_path = output_directory / (
        f"amplifier_result_{timestamp}_{job_id[:8]}.ampres.zip"
    )
    if final_path.exists():
        raise FileExistsError(f"refusing to replace existing archive {final_path}")
    temporary_path = output_directory / f".{final_path.name}.partial"
    temporary_path.unlink(missing_ok=True)
    record = diagnostic_record_for_result(
        result,
        python_warnings=python_warnings,
        ui_compatibility_adjustments=compatibility_adjustments,
        solve_job_id=job_id,
        solve_config_sha256=config_digest,
    )
    try:
        with temporary_path.open("xb") as destination:
            write_result_archive(
                destination,
                (ArchivePoint(record=record, config=config, result=result),),
                run_kind="single",
                base_config=config,
                run_metadata={
                    "source": "single_run_autosave",
                    "ui_compatibility_adjustments": compatibility_adjustments,
                },
            )
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, final_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return final_path


def _watch_manager_heartbeat(
    heartbeat_path: Path,
    *,
    maximum_age_s: float = 15.0,
    terminate_process_group: bool = False,
) -> None:
    """Hard-exit if the owning Streamlit server disappears.

    ``os._exit`` is intentional: it also works while the numerical main
    thread is inside a long native FFT call and cannot process Python signals.
    """

    try:
        observed_mtime_ns: int | None = heartbeat_path.stat().st_mtime_ns
    except OSError:
        observed_mtime_ns = None
    last_refresh_monotonic_s = time.monotonic()
    last_check_monotonic_s = last_refresh_monotonic_s
    while True:
        time.sleep(1.0)
        now_monotonic_s = time.monotonic()
        try:
            current_mtime_ns: int | None = heartbeat_path.stat().st_mtime_ns
        except OSError:
            current_mtime_ns = None
        (
            observed_mtime_ns,
            last_refresh_monotonic_s,
            last_check_monotonic_s,
            stale,
        ) = _advance_heartbeat_watchdog(
            observed_mtime_ns=observed_mtime_ns,
            last_refresh_monotonic_s=last_refresh_monotonic_s,
            last_check_monotonic_s=last_check_monotonic_s,
            current_mtime_ns=current_mtime_ns,
            now_monotonic_s=now_monotonic_s,
            maximum_age_s=maximum_age_s,
        )
        if stale:
            if (
                terminate_process_group
                and os.name == "posix"
                and hasattr(os, "killpg")
                and hasattr(signal, "SIGKILL")
            ):
                try:
                    # Managed workers start in their own session.  A parallel
                    # sweep's child processes inherit this group, so killing
                    # the group prevents expensive orphan point solves when
                    # the owning Streamlit server disappears abruptly.
                    os.killpg(os.getpgrp(), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            os._exit(74)


def _advance_heartbeat_watchdog(
    *,
    observed_mtime_ns: int | None,
    last_refresh_monotonic_s: float,
    last_check_monotonic_s: float,
    current_mtime_ns: int | None,
    now_monotonic_s: float,
    maximum_age_s: float,
) -> tuple[int | None, float, float, bool]:
    """Advance the parent-watchdog state without wall-clock false positives.

    A long gap between watchdog iterations means the host process was
    suspended or starved.  Give the manager one complete grace interval after
    that gap instead of treating time spent asleep as evidence that the
    Streamlit process disappeared.
    """

    if now_monotonic_s - last_check_monotonic_s > maximum_age_s:
        last_refresh_monotonic_s = now_monotonic_s
    if current_mtime_ns != observed_mtime_ns:
        observed_mtime_ns = current_mtime_ns
        last_refresh_monotonic_s = now_monotonic_s
    stale = now_monotonic_s - last_refresh_monotonic_s > maximum_age_s
    return (
        observed_mtime_ns,
        last_refresh_monotonic_s,
        now_monotonic_s,
        stale,
    )


def run_worker(job_dir: Path) -> int:
    request_path = job_dir / "request.pkl"
    progress_path = job_dir / "progress.jsonl"
    heartbeat_path = job_dir / "manager.heartbeat"
    result_path = job_dir / "result.pkl"
    outcome_path = job_dir / "outcome.json"

    with request_path.open("rb") as stream:
        request = pickle.load(stream)
    config = request["config"]
    apply_ui_worker_resource_limits(config)

    watchdog = threading.Thread(
        target=_watch_manager_heartbeat,
        args=(heartbeat_path,),
        daemon=True,
        name="amplifier-parent-watchdog",
    )
    watchdog.start()

    progress_stream = progress_path.open("a", encoding="utf-8", buffering=1)

    def report(event: dict[str, Any]) -> None:
        progress_stream.write(
            dumps_json_portable(event, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        progress_stream.flush()

    caught_warnings: list[Any] = []
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            simulator = AmplifierSimulator(
                config,
                progress_callback=report,
                progress_mode="milestones",
            )
            _install_archive_continuation(
                simulator,
                request.get("initial_continuation_archive"),
            )
            result = simulator.run()
        payload = {
            "result": result,
            "python_warnings": [
                python_warning_record(item) for item in caught_warnings
            ],
        }
        temporary_result = result_path.with_suffix(".pkl.tmp")
        with temporary_result.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_result, result_path)
        archive_path: Path | None = None
        archive_error: str | None = None
        archive_output_directory = request.get("archive_output_directory")
        if archive_output_directory is not None:
            try:
                archive_path = _autosave_result_archive(
                    output_directory=Path(str(archive_output_directory)),
                    job_id=str(request["job_id"]),
                    config_digest=str(request["config_digest"]),
                    config=config,
                    result=result,
                    python_warnings=payload["python_warnings"],
                    compatibility_adjustments=list(
                        request.get("compatibility_adjustments", [])
                    ),
                )
            except Exception as exc:
                archive_error = f"{type(exc).__name__}: {exc}"
        _atomic_text(
            outcome_path,
            {
                "status": "completed",
                "archive_path": (
                    None if archive_path is None else str(archive_path)
                ),
                "archive_error": archive_error,
            },
        )
        return 0
    except BaseException as exc:
        _atomic_text(
            outcome_path,
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "python_warnings": [
                    python_warning_record(item) for item in caught_warnings
                ],
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
