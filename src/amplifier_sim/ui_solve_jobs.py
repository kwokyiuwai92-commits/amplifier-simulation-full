"""Lifecycle manager for Streamlit-owned background solve processes."""

from __future__ import annotations

import atexit
import copy
import hashlib
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .json_contract import dumps_json_portable
from .ui_capacity import release_ui_worker, reserve_ui_worker


ACTIVE_STATUSES = frozenset({"running", "paused"})
TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "browser_disconnected"}
)
_PULSE_SUMMARY_STAGES = frozenset(
    {"pulse_train", "pulse_map", "reduced_pulse_map"}
)
_MAX_NON_PULSE_EVENTS = 64
_CURRENT_ONLY_PROGRESS_METRICS = frozenset(
    {
        "spectrum_preview_wavelength_m",
        "spectrum_preview_relative",
        "output_fwhm_s",
        "compressed_fwhm_s",
    }
)


@dataclass(frozen=True)
class SolveJobView:
    job_id: str
    session_id: str
    status: str
    pid: int | None
    config_snapshot: dict[str, Any]
    config_digest: str
    compatibility_adjustments: list[dict[str, Any]]
    events: tuple[dict[str, Any], ...]
    started_at_epoch_s: float
    finished_at_epoch_s: float | None
    result: Any | None = None
    python_warnings: tuple[dict[str, Any], ...] = ()
    archive_path: str | None = None
    archive_error: str | None = None
    failure: dict[str, Any] | None = None
    cancellation_reason: str | None = None

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES


@dataclass
class _SolveJob:
    job_id: str
    session_id: str
    process: subprocess.Popen[bytes]
    job_dir: Path
    config_snapshot: dict[str, Any]
    config_digest: str
    compatibility_adjustments: list[dict[str, Any]]
    status: str = "running"
    events: list[dict[str, Any]] = field(default_factory=list)
    progress_offset: int = 0
    started_at_epoch_s: float = field(default_factory=time.time)
    finished_at_epoch_s: float | None = None
    result: Any | None = None
    python_warnings: list[dict[str, Any]] = field(default_factory=list)
    archive_path: str | None = None
    archive_error: str | None = None
    failure: dict[str, Any] | None = None
    cancellation_reason: str | None = None
    disconnected_since_monotonic_s: float | None = None
    outcome_loaded: bool = False


_LOCK = threading.RLock()
_JOBS: dict[str, _SolveJob] = {}
_SUPERVISOR_STARTED = False
_SUPERVISOR_STOP = threading.Event()


def _streamlit_session_is_active(session_id: str) -> bool:
    """Read Streamlit's WebSocket session registry without browser timers."""

    try:
        from streamlit.runtime.runtime import Runtime

        if not Runtime.exists():
            return True
        return bool(Runtime.instance().is_active_session(session_id))
    except Exception:
        # Do not cancel a physical solve merely because an internal status API
        # changed. The worker's parent heartbeat still prevents true orphans.
        return True


def _touch_manager_heartbeat(job: _SolveJob) -> None:
    try:
        (job.job_dir / "manager.heartbeat").touch(exist_ok=True)
    except OSError:
        pass


def _progress_outer_iteration(event: dict[str, Any]) -> int | None:
    value = dict(event.get("metrics", {})).get("thermal_optical_iteration")
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _pulse_traversal_key(event: dict[str, Any]) -> tuple[str, int | None] | None:
    metrics = dict(event.get("metrics", {}))
    stage = str(event.get("stage", "unknown"))
    if stage not in _PULSE_SUMMARY_STAGES:
        return None
    if metrics.get("pulse_index", metrics.get("periodic_map_cycle")) is None:
        return None
    if metrics.get("output_energy_J") is None:
        return None
    return stage, _progress_outer_iteration(event)


def _progress_detail_key(
    event: dict[str, Any],
) -> tuple[str, int | None, str | None]:
    metrics = dict(event.get("metrics", {}))
    substage = metrics.get("seed_substage", metrics.get("phase"))
    return (
        str(event.get("stage", "unknown")),
        _progress_outer_iteration(event),
        None if substage is None else str(substage),
    )


def _compact_live_progress_events(
    events: list[dict[str, Any]],
    incoming: dict[str, Any],
) -> None:
    """Keep the live UI small while the append-only JSONL remains complete.

    Every completed pulse in the current outer traversal is retained.  Axial
    cells, substeps, Pump steps, and thermal iterations are represented only by
    their newest event for that stage/outer-iteration pair.  Spectrum previews
    and pulse widths are current-state products, so only their newest copy is
    kept in memory.
    """

    event = copy.deepcopy(incoming)
    incoming_metrics = dict(event.get("metrics", {}))
    replaced_metrics = _CURRENT_ONLY_PROGRESS_METRICS.intersection(
        incoming_metrics
    )
    if replaced_metrics:
        for stored in events:
            stored_metrics = stored.get("metrics")
            if not isinstance(stored_metrics, dict):
                continue
            for key in replaced_metrics:
                stored_metrics.pop(key, None)

    traversal_key = _pulse_traversal_key(event)
    if traversal_key is not None:
        prior_traversal = next(
            (
                key
                for stored in reversed(events)
                if (key := _pulse_traversal_key(stored)) is not None
            ),
            None,
        )
        if prior_traversal is not None and prior_traversal != traversal_key:
            # Previous traversals remain in progress.jsonl on disk.  The live
            # browser receives only the complete current traversal.
            events[:] = [
                stored
                for stored in events
                if _pulse_traversal_key(stored) is None
            ]
        events.append(event)
    else:
        detail_key = _progress_detail_key(event)
        events[:] = [
            stored
            for stored in events
            if _pulse_traversal_key(stored) is not None
            or _progress_detail_key(stored) != detail_key
        ]
        events.append(event)

    non_pulse_indices = [
        index
        for index, stored in enumerate(events)
        if _pulse_traversal_key(stored) is None
    ]
    excess = len(non_pulse_indices) - _MAX_NON_PULSE_EVENTS
    if excess > 0:
        discarded = set(non_pulse_indices[:excess])
        events[:] = [
            stored for index, stored in enumerate(events) if index not in discarded
        ]


def _read_progress_locked(job: _SolveJob) -> None:
    path = job.job_dir / "progress.jsonl"
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as stream:
            stream.seek(job.progress_offset)
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line:
                    job.progress_offset = position
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    job.progress_offset = position
                    break
                _compact_live_progress_events(job.events, event)
                job.progress_offset = stream.tell()
    except OSError:
        return


def _load_outcome_locked(job: _SolveJob) -> None:
    if job.outcome_loaded:
        return
    outcome_path = job.job_dir / "outcome.json"
    result_path = job.job_dir / "result.pkl"
    if not outcome_path.exists():
        return
    try:
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(outcome, dict):
        job.status = "failed"
        job.failure = {
            "status": "execution_failed",
            "error": "Background worker outcome must be a JSON object",
            "error_type": "WorkerOutcomeError",
            "error_message": "invalid outcome.json root",
            "traceback": _worker_log_tail(job),
            "python_warnings": [],
        }
        job.outcome_loaded = True
        job.finished_at_epoch_s = time.time()
        return
    status = str(outcome.get("status", "failed"))
    if status == "completed":
        raw_archive_path = outcome.get("archive_path")
        job.archive_path = (
            None if raw_archive_path is None else str(raw_archive_path)
        )
        raw_archive_error = outcome.get("archive_error")
        job.archive_error = (
            None if raw_archive_error is None else str(raw_archive_error)
        )
        if job.archive_path is not None and not Path(job.archive_path).is_file():
            job.archive_error = (
                job.archive_error
                or f"autosaved archive is missing: {job.archive_path}"
            )
            job.archive_path = None
        if not result_path.exists():
            job.status = "failed"
            job.failure = {
                "status": "execution_failed",
                "error": "Background worker reported completion without result.pkl",
                "error_type": "WorkerOutcomeError",
                "error_message": "missing result.pkl",
                "traceback": _worker_log_tail(job),
                "python_warnings": [],
            }
        else:
            try:
                with result_path.open("rb") as stream:
                    payload = pickle.load(stream)
                if not isinstance(payload, dict) or "result" not in payload:
                    raise ValueError("result.pkl must contain a result mapping")
            except Exception as exc:
                job.status = "failed"
                job.failure = {
                    "status": "execution_failed",
                    "error": f"Result handoff failed: {type(exc).__name__}: {exc}",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": "",
                    "python_warnings": [],
                }
            else:
                job.result = payload["result"]
                job.python_warnings = list(payload.get("python_warnings", []))
                job.status = "completed"
    elif status == "failed":
        job.status = "failed"
        job.failure = dict(outcome)
    else:
        job.status = "failed"
        job.failure = {
            "status": "execution_failed",
            "error": f"Background worker reported unsupported status {status!r}",
            "error_type": "WorkerOutcomeError",
            "error_message": f"unsupported outcome status {status!r}",
            "traceback": _worker_log_tail(job),
            "python_warnings": [],
        }
    job.outcome_loaded = True
    job.finished_at_epoch_s = time.time()


def _poll_locked(job: _SolveJob) -> None:
    _read_progress_locked(job)
    return_code = job.process.poll()
    if return_code is None:
        return
    release_ui_worker(job.job_id)
    if job.status in {"cancelled", "browser_disconnected"}:
        job.finished_at_epoch_s = job.finished_at_epoch_s or time.time()
        return
    _load_outcome_locked(job)
    if not job.outcome_loaded:
        job.status = "failed"
        job.finished_at_epoch_s = time.time()
        job.failure = {
            "status": "execution_failed",
            "error": f"Background worker exited with code {return_code}",
            "error_type": "WorkerProcessError",
            "error_message": f"exit code {return_code}",
            "traceback": _worker_log_tail(job),
            "python_warnings": [],
        }
        job.outcome_loaded = True


def _worker_log_tail(job: _SolveJob, maximum_chars: int = 12000) -> str:
    path = job.job_dir / "worker.log"
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-maximum_chars:]
    except OSError:
        return ""


def _view_locked(job: _SolveJob) -> SolveJobView:
    return SolveJobView(
        job_id=job.job_id,
        session_id=job.session_id,
        status=job.status,
        pid=job.process.pid,
        config_snapshot=copy.deepcopy(job.config_snapshot),
        config_digest=job.config_digest,
        compatibility_adjustments=copy.deepcopy(job.compatibility_adjustments),
        events=tuple(copy.deepcopy(job.events)),
        started_at_epoch_s=job.started_at_epoch_s,
        finished_at_epoch_s=job.finished_at_epoch_s,
        result=job.result,
        python_warnings=tuple(copy.deepcopy(job.python_warnings)),
        archive_path=job.archive_path,
        archive_error=job.archive_error,
        failure=copy.deepcopy(job.failure),
        cancellation_reason=job.cancellation_reason,
    )


def _kill_locked(job: _SolveJob, *, status: str, reason: str) -> None:
    if job.process.poll() is None:
        if job.status == "paused" and hasattr(signal, "SIGCONT"):
            try:
                _signal_worker_group(job, signal.SIGCONT)
            except ProcessLookupError:
                pass
        try:
            if hasattr(signal, "SIGKILL"):
                _signal_worker_group(job, signal.SIGKILL)
            else:
                job.process.kill()
        except ProcessLookupError:
            pass
        try:
            job.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
    job.status = status
    release_ui_worker(job.job_id)
    job.cancellation_reason = reason
    job.finished_at_epoch_s = time.time()
    _read_progress_locked(job)


def _signal_worker_group(job: _SolveJob, requested_signal: int) -> None:
    """Signal the isolated worker session, including any numerical children."""

    if os.name == "posix" and hasattr(os, "killpg"):
        os.killpg(os.getpgid(job.process.pid), requested_signal)
        return
    os.kill(job.process.pid, requested_signal)


def _supervise_once(
    *,
    session_is_active: Callable[[str], bool] = _streamlit_session_is_active,
    now_monotonic_s: float | None = None,
) -> None:
    now = time.monotonic() if now_monotonic_s is None else now_monotonic_s
    with _LOCK:
        for session_id, job in list(_JOBS.items()):
            _poll_locked(job)
            if job.status not in ACTIVE_STATUSES:
                continue
            _touch_manager_heartbeat(job)
            if session_is_active(job.session_id):
                job.disconnected_since_monotonic_s = None
                continue
            if job.disconnected_since_monotonic_s is None:
                job.disconnected_since_monotonic_s = now
            # A browser is only a viewer/controller.  Temporary WebSocket loss,
            # a sleeping laptop, or a slow Plotly render must never terminate a
            # multi-hour physical solve.  The manager heartbeat still stops an
            # orphan worker if the Streamlit server itself disappears.


def _supervisor_loop() -> None:
    while not _SUPERVISOR_STOP.wait(1.0):
        _supervise_once()


def _heartbeat_loop(
    jobs: dict[str, Any],
    stop: threading.Event,
    touch: Callable[[Any], None],
) -> None:
    """Keep workers alive while result loading/archiving holds manager locks."""

    while not stop.wait(1.0):
        # A shallow dictionary snapshot does not borrow the lifecycle lock:
        # archive compression and large pickle loads may hold it for minutes.
        for job in jobs.copy().values():
            if job.status in ACTIVE_STATUSES:
                touch(job)


def _ensure_supervisor() -> None:
    global _SUPERVISOR_STARTED
    with _LOCK:
        if _SUPERVISOR_STARTED:
            return
        threading.Thread(
            target=_heartbeat_loop,
            args=(_JOBS, _SUPERVISOR_STOP, _touch_manager_heartbeat),
            daemon=True,
            name="amplifier-solve-heartbeat",
        ).start()
        thread = threading.Thread(
            target=_supervisor_loop,
            daemon=True,
            name="amplifier-solve-supervisor",
        )
        thread.start()
        _SUPERVISOR_STARTED = True


def start_solve_job(
    session_id: str,
    config: dict[str, Any],
    *,
    project_dir: Path,
    compatibility_adjustments: list[dict[str, Any]] | None = None,
    initial_continuation_archive: dict[str, Any] | None = None,
    archive_output_directory: Path | None = None,
) -> SolveJobView:
    """Start one immutable-config worker for a connected browser session."""

    snapshot = copy.deepcopy(config)
    continuation_descriptor: dict[str, Any] | None = None
    if initial_continuation_archive is not None:
        archive_path = Path(
            str(initial_continuation_archive["archive_path"])
        ).expanduser().resolve()
        point_index = int(initial_continuation_archive["point_index"])
        if not archive_path.is_file():
            raise FileNotFoundError(
                f"warm-start result archive does not exist: {archive_path}"
            )
        if point_index < 0:
            raise ValueError("warm-start archive point index must be non-negative")
        continuation_descriptor = {
            "archive_path": str(archive_path),
            "point_index": point_index,
        }
    digest = hashlib.sha256(
        dumps_json_portable(snapshot, sort_keys=True).encode("utf-8")
    ).hexdigest()
    with _LOCK:
        existing = _JOBS.get(session_id)
        if existing is not None:
            _poll_locked(existing)
            if existing.status in ACTIVE_STATUSES:
                raise RuntimeError("this browser session already owns an active solve")
            _cleanup_locked(existing)
            _JOBS.pop(session_id, None)

        job_id = uuid.uuid4().hex
        job_dir = Path(tempfile.mkdtemp(prefix=f"amplifier-solve-{job_id[:8]}-"))
        request_path = job_dir / "request.pkl"
        request: dict[str, Any] = {
            "config": snapshot,
            "job_id": job_id,
            "config_digest": digest,
            "compatibility_adjustments": copy.deepcopy(
                compatibility_adjustments or []
            ),
        }
        if archive_output_directory is not None:
            request["archive_output_directory"] = str(
                Path(archive_output_directory).expanduser().resolve()
            )
        if continuation_descriptor is not None:
            request["initial_continuation_archive"] = continuation_descriptor
        with request_path.open("wb") as stream:
            pickle.dump(request, stream, protocol=pickle.HIGHEST_PROTOCOL)
        (job_dir / "manager.heartbeat").touch()
        log_stream = (job_dir / "worker.log").open("ab", buffering=0)
        environment = os.environ.copy()
        source_dir = str(Path(project_dir).resolve() / "src")
        prior_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_dir
            if not prior_pythonpath
            else source_dir + os.pathsep + prior_pythonpath
        )
        native_threads = environment.get("AMPLIFIER_SIM_UI_NATIVE_THREADS")
        if native_threads is not None:
            try:
                native_threads = str(max(int(native_threads), 1))
            except ValueError:
                native_threads = None
        if native_threads is not None:
            for variable in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                environment[variable] = native_threads
            environment["AMPLIFIER_SIM_UI_FFT_WORKERS"] = native_threads
        try:
            reserve_ui_worker(job_id)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "amplifier_sim.ui_solve_worker",
                    "--job-dir",
                    str(job_dir),
                ],
                cwd=str(Path(project_dir).resolve()),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            release_ui_worker(job_id)
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        finally:
            log_stream.close()
        job = _SolveJob(
            job_id=job_id,
            session_id=str(session_id),
            process=process,
            job_dir=job_dir,
            config_snapshot=snapshot,
            config_digest=digest,
            compatibility_adjustments=copy.deepcopy(
                compatibility_adjustments or []
            ),
        )
        _JOBS[session_id] = job
        _ensure_supervisor()
        return _view_locked(job)


def get_solve_job(session_id: str) -> SolveJobView | None:
    with _LOCK:
        job = _JOBS.get(session_id)
        if job is None:
            return None
        _touch_manager_heartbeat(job)
        _poll_locked(job)
        return _view_locked(job)


def pause_solve_job(session_id: str) -> SolveJobView:
    with _LOCK:
        job = _JOBS[session_id]
        _poll_locked(job)
        if job.status != "running":
            return _view_locked(job)
        if not hasattr(signal, "SIGSTOP"):
            raise RuntimeError("process pause is not supported on this platform")
        try:
            _signal_worker_group(job, signal.SIGSTOP)
        except ProcessLookupError:
            _poll_locked(job)
            return _view_locked(job)
        job.status = "paused"
        return _view_locked(job)


def resume_solve_job(session_id: str) -> SolveJobView:
    with _LOCK:
        job = _JOBS[session_id]
        _poll_locked(job)
        if job.status != "paused":
            return _view_locked(job)
        if not hasattr(signal, "SIGCONT"):
            raise RuntimeError("process resume is not supported on this platform")
        try:
            _signal_worker_group(job, signal.SIGCONT)
        except ProcessLookupError:
            _poll_locked(job)
            return _view_locked(job)
        job.status = "running"
        return _view_locked(job)


def cancel_solve_job(
    session_id: str,
    *,
    reason: str = "Cancelled explicitly by the user.",
) -> SolveJobView:
    with _LOCK:
        job = _JOBS[session_id]
        _poll_locked(job)
        if job.status in ACTIVE_STATUSES:
            _kill_locked(job, status="cancelled", reason=reason)
        return _view_locked(job)


def release_solve_job(session_id: str) -> SolveJobView | None:
    """Detach one terminal job after the UI has persisted its outcome."""

    with _LOCK:
        job = _JOBS.get(session_id)
        if job is None:
            return None
        _poll_locked(job)
        if job.status in ACTIVE_STATUSES:
            raise RuntimeError("cannot release an active solve job")
        view = _view_locked(job)
        _JOBS.pop(session_id, None)
        _cleanup_locked(job)
        return view


def _cleanup_locked(job: _SolveJob) -> None:
    try:
        shutil.rmtree(job.job_dir)
    except OSError:
        pass


def _shutdown_all_jobs() -> None:
    _SUPERVISOR_STOP.set()
    with _LOCK:
        for job in list(_JOBS.values()):
            if job.status in ACTIVE_STATUSES:
                _kill_locked(job, status="cancelled", reason="UI server shutdown")
            _cleanup_locked(job)
        _JOBS.clear()


atexit.register(_shutdown_all_jobs)
