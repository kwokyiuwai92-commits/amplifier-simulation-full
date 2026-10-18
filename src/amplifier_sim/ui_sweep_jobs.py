"""Lifecycle manager for pauseable Streamlit parameter-sweep workers."""

from __future__ import annotations

import atexit
import copy
import hashlib
import hmac
import io
import json
import os
import pickle
import secrets
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
from typing import Any, BinaryIO

from .json_contract import dumps_json_portable
from .result_archive import ArchivePointFile, write_result_archive_from_point_files
from .sweep import SWEEP_METRICS, request_uses_continuation, validate_sweep_parameter_pair, sweep_parameter_inactive_reason
from .ui_capacity import release_ui_worker, reserve_ui_worker
from .ui_solve_jobs import (
    ACTIVE_STATUSES,
    _compact_live_progress_events,
    _heartbeat_loop,
    _streamlit_session_is_active,
)


@dataclass(frozen=True)
class SweepJobView:
    job_id: str
    session_id: str
    status: str
    pid: int | None
    config_snapshot: dict[str, Any]
    config_digest: str
    request_snapshot: dict[str, Any]
    compatibility_adjustments: list[dict[str, Any]]
    completed_records: tuple[dict[str, Any], ...]
    current_point: int | None
    total_points: int
    current_record: dict[str, Any] | None
    events: tuple[dict[str, Any], ...]
    active_points: tuple[dict[str, Any], ...]
    started_at_epoch_s: float
    finished_at_epoch_s: float | None
    summary: Any | None = None
    failure: dict[str, Any] | None = None
    cancellation_reason: str | None = None

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES


@dataclass
class _SweepJob:
    job_id: str
    session_id: str
    process: subprocess.Popen[bytes] | None
    job_dir: Path
    config_snapshot: dict[str, Any]
    config_digest: str
    request_snapshot: dict[str, Any]
    compatibility_adjustments: list[dict[str, Any]]
    total_points: int
    status: str = "running"
    completed_records: list[dict[str, Any]] = field(default_factory=list)
    current_point: int | None = None
    current_record: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    active_records: dict[int, dict[str, Any]] = field(default_factory=dict)
    active_events: dict[int, list[dict[str, Any]]] = field(
        default_factory=dict
    )
    progress_offset: int = 0
    started_at_epoch_s: float = field(default_factory=time.time)
    finished_at_epoch_s: float | None = None
    summary: Any | None = None
    failure: dict[str, Any] | None = None
    cancellation_reason: str | None = None
    outcome_loaded: bool = False
    full_archive_written_at_epoch_s: float | None = None
    preserve_after_browser_disconnect: bool = False


_LOCK = threading.RLock()
_JOBS: dict[str, _SweepJob] = {}
_SUPERVISOR_STARTED = False
_SUPERVISOR_STOP = threading.Event()

_JOB_METADATA_FILE = "job.json"
_FULL_ARCHIVE_MARKER = "full-archive-written.json"


def _sweep_storage_root() -> Path:
    """Use a private namespace per work directory, never the global temp pool."""
    work = Path(os.environ.get("AMPLIFIER_SIM_WORK_DIR", str(Path(__file__).resolve().parents[2]))).expanduser().resolve()
    namespace = hashlib.sha256(os.fsencode(work)).hexdigest()[:32]
    root = Path(tempfile.gettempdir()) / ("amplifier-private-" + namespace)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("Sweep storage must be a private directory")
    if os.name == "posix" and (root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077):
        raise RuntimeError("Sweep storage ownership or permissions are unsafe")
    return root.resolve()


def _sweep_storage_key(root: Path) -> bytes:
    """Authenticate local requests before recovery can deserialize them."""
    key_path = root / "request-auth.key"
    if not key_path.exists():
        # Publish a complete key atomically; simultaneous managers never
        # overwrite an existing key or observe a partially written key.
        with tempfile.NamedTemporaryFile(dir=root, prefix="key-", mode="wb") as stream:
            stream.write(secrets.token_bytes(32))
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(stream.name, key_path)
            except FileExistsError:
                pass
    if key_path.is_symlink() or not key_path.is_file():
        raise RuntimeError("Invalid sweep authentication key")
    if os.name == "posix" and (key_path.stat().st_uid != os.getuid() or key_path.stat().st_mode & 0o077):
        raise RuntimeError("Sweep key ownership or permissions are unsafe")
    key = key_path.read_bytes()
    if len(key) != 32:
        raise RuntimeError("Invalid sweep authentication key length")
    return key


def _other_manager_is_alive(pid: Any) -> bool:
    if type(pid) is not int or not 1 < pid <= 2**31 - 1:
        return True  # Missing/legacy provenance never grants recovery authority.
    if pid == os.getpid():
        return False
    if os.name != "posix":
        return True  # Do not use POSIX liveness probes on another platform.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _archive_was_written(job: _SweepJob) -> bool:
    return (
        job.full_archive_written_at_epoch_s is not None
        or (job.job_dir / _FULL_ARCHIVE_MARKER).is_file()
    )


def _touch_heartbeat(job: _SweepJob) -> None:
    try:
        (job.job_dir / "manager.heartbeat").touch(exist_ok=True)
    except OSError:
        pass


def _point_index(record: dict[str, Any], point: int) -> int:
    return int(record.get("point_index", max(point - 1, 0)))


def _completed_point_indices(job: _SweepJob) -> set[int]:
    return {
        _point_index(record, index + 1)
        for index, record in enumerate(job.completed_records)
    }


def _store_completed_record(job: _SweepJob, record: dict[str, Any], point: int) -> None:
    point_index = _point_index(record, point)
    replaced = False
    for index, existing in enumerate(job.completed_records):
        if _point_index(existing, index + 1) == point_index:
            job.completed_records[index] = record
            replaced = True
            break
    if not replaced:
        job.completed_records.append(record)
    job.completed_records.sort(
        key=lambda item: _point_index(item, point_index + 1)
    )


def _refresh_legacy_current_point(job: _SweepJob) -> None:
    """Expose one representative point to old serial UI consumers."""

    if job.active_records:
        point = min(job.active_records)
        job.current_point = point
        job.current_record = copy.deepcopy(job.active_records[point])
        job.events = copy.deepcopy(job.active_events.get(point, []))
    elif job.completed_records:
        record = job.completed_records[-1]
        job.current_point = _point_index(record, len(job.completed_records)) + 1
        job.current_record = copy.deepcopy(record)
        job.events = []


def _read_progress_locked(job: _SweepJob) -> None:
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
                    wrapper = json.loads(line)
                except json.JSONDecodeError:
                    job.progress_offset = position
                    break
                event_type = str(wrapper.get("event_type", ""))
                point = int(wrapper.get("point", 0) or 0)
                if event_type == "point_progress":
                    if point - 1 in _completed_point_indices(job):
                        job.progress_offset = stream.tell()
                        continue
                    record = copy.deepcopy(wrapper.get("record", {}))
                    job.active_records[point] = record
                    point_events = job.active_events.setdefault(point, [])
                    event = wrapper.get("event")
                    if isinstance(event, dict):
                        _compact_live_progress_events(point_events, event)
                    _refresh_legacy_current_point(job)
                elif event_type == "point_complete":
                    record = copy.deepcopy(wrapper.get("record", {}))
                    _store_completed_record(job, record, point)
                    job.active_records.pop(point, None)
                    job.active_events.pop(point, None)
                    _refresh_legacy_current_point(job)
                job.progress_offset = stream.tell()
    except OSError:
        return


def _load_outcome_locked(job: _SweepJob) -> None:
    if job.outcome_loaded:
        return
    outcome_path = job.job_dir / "outcome.json"
    if not outcome_path.exists():
        return
    try:
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(outcome, dict):
        job.status = "failed"
        job.failure = {
            "error": "Background sweep outcome must be a JSON object",
            "error_type": "WorkerOutcomeError",
            "error_message": "invalid outcome.json root",
            "traceback": _worker_log_tail(job),
        }
        job.finished_at_epoch_s = time.time()
        job.outcome_loaded = True
        return
    if str(outcome.get("status")) == "completed":
        try:
            with (job.job_dir / "summary.pkl").open("rb") as stream:
                job.summary = pickle.load(stream)
        except Exception as exc:
            job.status = "failed"
            job.failure = {
                "error": f"Sweep handoff failed: {type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": "",
            }
        else:
            job.status = "completed"
            job.active_records.clear()
            job.active_events.clear()
    else:
        job.status = "failed"
        job.failure = dict(outcome)
    job.finished_at_epoch_s = time.time()
    job.outcome_loaded = True


def _worker_log_tail(job: _SweepJob, maximum_chars: int = 12000) -> str:
    try:
        return (job.job_dir / "worker.log").read_text(
            encoding="utf-8", errors="replace"
        )[-maximum_chars:]
    except OSError:
        return ""


def _poll_locked(job: _SweepJob) -> None:
    _read_progress_locked(job)
    if job.process is None:
        _load_outcome_locked(job)
        return
    return_code = job.process.poll()
    if return_code is None:
        return
    release_ui_worker(job.job_id)
    if job.status == "cancelled":
        job.finished_at_epoch_s = job.finished_at_epoch_s or time.time()
        return
    _load_outcome_locked(job)
    if not job.outcome_loaded:
        job.status = "failed"
        job.finished_at_epoch_s = time.time()
        job.failure = {
            "error": f"Background sweep worker exited with code {return_code}",
            "error_type": "WorkerProcessError",
            "error_message": f"exit code {return_code}",
            "traceback": _worker_log_tail(job),
        }
        job.outcome_loaded = True


def _view_locked(job: _SweepJob) -> SweepJobView:
    active_points = tuple(
        {
            "point": point,
            "record": copy.deepcopy(job.active_records[point]),
            "events": tuple(copy.deepcopy(job.active_events.get(point, []))),
        }
        for point in sorted(job.active_records)
    )
    return SweepJobView(
        job_id=job.job_id,
        session_id=job.session_id,
        status=job.status,
        pid=job.process.pid if job.process is not None else None,
        config_snapshot=copy.deepcopy(job.config_snapshot),
        config_digest=job.config_digest,
        request_snapshot=copy.deepcopy(job.request_snapshot),
        compatibility_adjustments=copy.deepcopy(job.compatibility_adjustments),
        completed_records=tuple(copy.deepcopy(job.completed_records)),
        current_point=job.current_point,
        total_points=job.total_points,
        current_record=copy.deepcopy(job.current_record),
        events=tuple(copy.deepcopy(job.events)),
        active_points=active_points,
        started_at_epoch_s=job.started_at_epoch_s,
        finished_at_epoch_s=job.finished_at_epoch_s,
        summary=job.summary,
        failure=copy.deepcopy(job.failure),
        cancellation_reason=job.cancellation_reason,
    )


def _signal_group(job: _SweepJob, requested_signal: int) -> None:
    if job.process is None:
        raise ProcessLookupError("recovered sweep worker is no longer running")
    if os.name == "posix" and hasattr(os, "killpg"):
        os.killpg(os.getpgid(job.process.pid), requested_signal)
    else:
        os.kill(job.process.pid, requested_signal)


def _kill_locked(job: _SweepJob, reason: str) -> None:
    if job.process is None:
        raise RuntimeError("cannot cancel a recovered terminal sweep")
    if job.process.poll() is None:
        if job.status == "paused" and hasattr(signal, "SIGCONT"):
            try:
                _signal_group(job, signal.SIGCONT)
            except ProcessLookupError:
                pass
        try:
            if hasattr(signal, "SIGKILL"):
                _signal_group(job, signal.SIGKILL)
            else:
                job.process.kill()
        except ProcessLookupError:
            pass
        try:
            job.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
    _read_progress_locked(job)
    job.status = "cancelled"
    release_ui_worker(job.job_id)
    job.cancellation_reason = reason
    job.finished_at_epoch_s = time.time()


def _supervise_once() -> None:
    with _LOCK:
        for owner, job in list(_JOBS.items()):
            _poll_locked(job)
            browser_active = _streamlit_session_is_active(owner)
            if job.status in ACTIVE_STATUSES:
                # The browser is only a viewer.  Keep the task alive across
                # tab closure/reconnect; the worker watchdog still exits if
                # this server process stops touching its heartbeat.
                if not browser_active:
                    job.preserve_after_browser_disconnect = True
                _touch_heartbeat(job)
            elif not browser_active and _archive_was_written(job):
                # A terminal job is disposable only after a full archive was
                # built successfully.  Browser loss alone must never destroy
                # the only full-fidelity copy of a completed sweep.
                _JOBS.pop(owner, None)
                shutil.rmtree(job.job_dir, ignore_errors=True)


def _supervisor_loop() -> None:
    while not _SUPERVISOR_STOP.wait(1.0):
        _supervise_once()


def _ensure_supervisor() -> None:
    global _SUPERVISOR_STARTED
    with _LOCK:
        if _SUPERVISOR_STARTED:
            return
        threading.Thread(
            target=_heartbeat_loop,
            args=(_JOBS, _SUPERVISOR_STOP, _touch_heartbeat),
            daemon=True,
            name="amplifier-sweep-heartbeat",
        ).start()
        threading.Thread(
            target=_supervisor_loop,
            daemon=True,
            name="amplifier-sweep-supervisor",
        ).start()
        _SUPERVISOR_STARTED = True


def _total_points(request: dict[str, Any]) -> int:
    count = len(request["values"])
    if int(request.get("dimension", 1)) == 2:
        count *= len(request["secondary_values"])
    return count


def _normalize_point_workers(request: dict[str, Any]) -> int:
    """Validate point parallelism and retain ordered optical warm starts."""

    raw = request.get("point_workers", 1)
    if isinstance(raw, bool):
        raise ValueError("point_workers must be an integer from 1 to 3")
    try:
        requested = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("point_workers must be an integer from 1 to 3") from exc
    if requested != raw or not 1 <= requested <= 3:
        raise ValueError("point_workers must be an integer from 1 to 3")
    pump_continuation = request_uses_continuation(request)
    actual = 1 if pump_continuation else min(requested, _total_points(request))
    request["point_workers_requested"] = requested
    request["point_workers"] = actual
    request["pump_continuation_serialized"] = pump_continuation
    request["continuation_serialized"] = pump_continuation
    request["continuation_traversal"] = "snake" if pump_continuation and int(request.get("dimension", 1)) == 2 else "input_order"
    return actual


def start_sweep_job(
    session_id: str,
    config: dict[str, Any],
    request: dict[str, Any],
    *,
    project_dir: Path,
    compatibility_adjustments: list[dict[str, Any]] | None = None,
    initial_continuation_archive: dict[str, Any] | None = None,
) -> SweepJobView:
    snapshot = copy.deepcopy(config)
    request_snapshot = copy.deepcopy(request)
    for key in ("parameter", "secondary_parameter"):
        parameter = request_snapshot.get(key)
        if parameter is not None:
            reason = sweep_parameter_inactive_reason(snapshot, parameter)
            if reason is not None:
                raise ValueError(f"Cannot sweep {parameter.path}: {reason}")
    if int(request_snapshot.get("dimension", 1)) == 2:
        validate_sweep_parameter_pair(
            snapshot, request_snapshot["parameter"], request_snapshot["secondary_parameter"]
        )
    point_workers = _normalize_point_workers(request_snapshot)
    digest = hashlib.sha256(
        dumps_json_portable(snapshot, sort_keys=True).encode("utf-8")
    ).hexdigest()
    with _LOCK:
        existing = _JOBS.get(session_id)
        if existing is not None:
            _poll_locked(existing)
            if existing.status in ACTIVE_STATUSES:
                raise RuntimeError("this browser session already owns an active sweep")
            if not existing.completed_records:
                # A terminal sweep with no completed point has no recoverable
                # result to archive.  Requiring a download here deadlocks the
                # UI because no archive link can be rendered.  Any sweep with
                # even one durable point still follows the strict archive gate.
                shutil.rmtree(existing.job_dir, ignore_errors=True)
                _JOBS.pop(session_id, None)
                existing = None
            elif not _archive_was_written(existing):
                raise RuntimeError(
                    "the previous completed sweep still awaits a full archive "
                    "download; download it before starting another sweep"
                )
            else:
                shutil.rmtree(existing.job_dir, ignore_errors=True)
                _JOBS.pop(session_id, None)

        job_id = uuid.uuid4().hex
        storage = _sweep_storage_root()
        storage_key = _sweep_storage_key(storage)
        job_dir = Path(tempfile.mkdtemp(prefix=f"amplifier-sweep-{job_id[:8]}-", dir=storage))
        with (job_dir / "request.pkl").open("wb") as stream:
            pickle.dump(
                {
                    "config": snapshot,
                    "request": request_snapshot,
                    "compatibility_adjustments": copy.deepcopy(
                        compatibility_adjustments or []
                    ),
                    "initial_continuation_archive": copy.deepcopy(
                        initial_continuation_archive
                    ),
                },
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        (job_dir / _JOB_METADATA_FILE).write_text(
            dumps_json_portable(
                {
                    "job_id": job_id,
                    "config_digest": digest,
                    "started_at_epoch_s": time.time(),
                    "point_workers": point_workers,
                    "manager_pid": os.getpid(),
                    "storage_version": 2,
                    "request_hmac_sha256": hmac.new(storage_key, (job_dir / "request.pkl").read_bytes(), hashlib.sha256).hexdigest(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (job_dir / "manager.heartbeat").touch()
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
        log_stream = (job_dir / "worker.log").open("ab", buffering=0)
        try:
            reserve_ui_worker(job_id)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "amplifier_sim.ui_sweep_worker",
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
        job = _SweepJob(
            job_id=job_id,
            session_id=str(session_id),
            process=process,
            job_dir=job_dir,
            config_snapshot=snapshot,
            config_digest=digest,
            request_snapshot=request_snapshot,
            compatibility_adjustments=copy.deepcopy(
                compatibility_adjustments or []
            ),
            total_points=_total_points(request_snapshot),
        )
        _JOBS[session_id] = job
        _ensure_supervisor()
        return _view_locked(job)


def get_sweep_job(session_id: str) -> SweepJobView | None:
    with _LOCK:
        job = _JOBS.get(session_id)
        if job is None:
            return None
        _touch_heartbeat(job)
        _poll_locked(job)
        return _view_locked(job)


def pause_sweep_job(session_id: str) -> SweepJobView:
    with _LOCK:
        job = _JOBS[session_id]
        _poll_locked(job)
        if job.status == "running":
            if not hasattr(signal, "SIGSTOP"):
                raise RuntimeError("process pause is not supported on this platform")
            try:
                _signal_group(job, signal.SIGSTOP)
            except ProcessLookupError:
                _poll_locked(job)
            else:
                job.status = "paused"
        return _view_locked(job)


def resume_sweep_job(session_id: str) -> SweepJobView:
    with _LOCK:
        job = _JOBS[session_id]
        _poll_locked(job)
        if job.status == "paused":
            if not hasattr(signal, "SIGCONT"):
                raise RuntimeError("process resume is not supported on this platform")
            try:
                _signal_group(job, signal.SIGCONT)
            except ProcessLookupError:
                _poll_locked(job)
            else:
                job.status = "running"
        return _view_locked(job)


def cancel_sweep_job(
    session_id: str, *, reason: str = "Cancelled explicitly by the user."
) -> SweepJobView:
    with _LOCK:
        job = _JOBS[session_id]
        _poll_locked(job)
        if job.status in ACTIVE_STATUSES:
            _kill_locked(job, reason)
        return _view_locked(job)


def release_sweep_job(session_id: str) -> SweepJobView | None:
    """Remove one terminal job after its archive is no longer needed."""

    with _LOCK:
        job = _JOBS.get(session_id)
        if job is None:
            return None
        _poll_locked(job)
        if job.status in ACTIVE_STATUSES:
            raise RuntimeError("cannot release an active sweep job")
        view = _view_locked(job)
        _JOBS.pop(session_id, None)
        shutil.rmtree(job.job_dir, ignore_errors=True)
        return view


def _recover_terminal_orphans_locked() -> list[_SweepJob]:
    """Serialize ownership claims across processes sharing a work directory."""
    root = _sweep_storage_root()
    with (root / "recovery.lock").open("a") as guard:
        if os.name == "posix":
            import fcntl
            fcntl.flock(guard, fcntl.LOCK_EX)
        try:
            return _recover_terminal_orphans_in_storage(root)
        finally:
            if os.name == "posix":
                fcntl.flock(guard, fcntl.LOCK_UN)


def _recover_terminal_orphans_in_storage(temporary_root: Path) -> list[_SweepJob]:
    """Recover completed, unarchived sweeps after a manager/session reload.

    Finished directories are never deleted merely because the in-memory job
    registry was lost.  A terminal orphan is removed only when its durable
    marker proves that a full archive was already built successfully.
    """

    storage_key = _sweep_storage_key(temporary_root)
    registered = {job.job_dir.resolve() for job in _JOBS.values()}
    recovered: list[_SweepJob] = []
    for candidate in temporary_root.glob("amplifier-sweep-*"):
        try:
            if candidate.is_symlink():
                continue
            resolved = candidate.resolve(strict=True)
            stat = resolved.stat()
            if os.name == "posix" and (stat.st_uid != os.getuid() or stat.st_mode & 0o077):
                continue
        except OSError:
            continue
        if (
            resolved.parent != temporary_root
            or candidate.is_symlink()
            or not resolved.name.startswith("amplifier-sweep-")
            or resolved in registered
        ):
            continue
        try:
            metadata_path = resolved / _JOB_METADATA_FILE
            request_path = resolved / "request.pkl"
            if metadata_path.is_symlink() or request_path.is_symlink():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (not isinstance(metadata, dict) or metadata.get("storage_version") != 2
                    or _other_manager_is_alive(metadata.get("manager_pid"))):
                continue
            request_bytes = request_path.read_bytes()
            expected = metadata.get("request_hmac_sha256")
            if not isinstance(expected, str) or not hmac.compare_digest(
                    expected, hmac.new(storage_key, request_bytes, hashlib.sha256).hexdigest()):
                continue
        except (OSError, ValueError, TypeError):
            continue
        outcome_path = resolved / "outcome.json"
        try:
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(outcome, dict) or str(outcome.get("status")) not in {"completed", "failed"}:
            continue
        if (resolved / _FULL_ARCHIVE_MARKER).is_file():
            shutil.rmtree(resolved, ignore_errors=True)
            continue
        try:
            # Deserialize the exact bytes authenticated above, not a second
            # file read which could have changed after verification.
            payload = pickle.loads(request_bytes)
            snapshot = copy.deepcopy(payload["config"])
            request = copy.deepcopy(payload["request"])
            digest = str(
                metadata.get("config_digest")
                or hashlib.sha256(
                    dumps_json_portable(snapshot, sort_keys=True).encode("utf-8")
                ).hexdigest()
            )
            job = _SweepJob(
                job_id=str(
                    metadata.get("job_id")
                    or f"recovered-{resolved.name.removeprefix('amplifier-sweep-')}"
                ),
                session_id="",
                process=None,
                job_dir=resolved,
                config_snapshot=snapshot,
                config_digest=digest,
                request_snapshot=request,
                compatibility_adjustments=copy.deepcopy(
                    payload.get("compatibility_adjustments", [])
                ),
                total_points=_total_points(request),
                started_at_epoch_s=float(
                    metadata.get("started_at_epoch_s")
                    or resolved.stat().st_mtime
                ),
            )
            _poll_locked(job)
        except Exception:
            # An unreadable directory is preserved for manual diagnosis.  A
            # recovery failure is never permission to destroy result data.
            continue
        if job.outcome_loaded and not job.completed_records:
            # Empty terminal orphan directories cannot produce an archive and
            # are never shown by the UI, so retaining them would permanently
            # block the next sweep after session recovery.
            shutil.rmtree(resolved, ignore_errors=True)
            continue
        if job.outcome_loaded:
            # Publish the ownership claim while holding the interprocess lock.
            # A second manager sees our live PID before it can load the request.
            metadata["manager_pid"] = os.getpid()
            claim = resolved / ("owner-" + uuid.uuid4().hex + ".tmp")
            claim.write_text(dumps_json_portable(metadata), encoding="utf-8")
            os.replace(claim, metadata_path)
            recovered.append(job)
    return recovered


def reconcile_sweep_session(session_id: str) -> SweepJobView | None:
    """Reattach a detached or disk-recovered unarchived sweep.

    Browser closure or refresh may create a new Streamlit session id.  A lone
    active job whose prior WebSocket is gone is transferred to the new session.
    Completed jobs remain recoverable across browser and manager reloads until
    a full archive has been written successfully.
    """

    normalized_session_id = str(session_id)
    with _LOCK:
        current = _JOBS.get(normalized_session_id)
        if current is None:
            detached_recoverable: list[tuple[str, _SweepJob]] = []
            for owner, job in list(_JOBS.items()):
                _poll_locked(job)
                if (
                    not _archive_was_written(job)
                    and not _streamlit_session_is_active(owner)
                ):
                    detached_recoverable.append((owner, job))
            for recovered in _recover_terminal_orphans_locked():
                detached_recoverable.append(("", recovered))
            if len(detached_recoverable) == 1:
                prior_owner, current = detached_recoverable[0]
                if prior_owner:
                    _JOBS.pop(prior_owner, None)
                current.session_id = normalized_session_id
                current.preserve_after_browser_disconnect = False
                _JOBS[normalized_session_id] = current
            elif detached_recoverable:
                # Prefer the newest unarchived result without deleting any
                # older candidate.  This mirrors the most recent UI task.
                prior_owner, current = max(
                    detached_recoverable,
                    key=lambda item: item[1].started_at_epoch_s,
                )
                if prior_owner:
                    _JOBS.pop(prior_owner, None)
                current.session_id = normalized_session_id
                current.preserve_after_browser_disconnect = False
                _JOBS[normalized_session_id] = current

        for owner, job in list(_JOBS.items()):
            _poll_locked(job)
            if (
                owner != normalized_session_id
                and job.status not in ACTIVE_STATUSES
                and _archive_was_written(job)
            ):
                _JOBS.pop(owner, None)
                shutil.rmtree(job.job_dir, ignore_errors=True)

        current = _JOBS.get(normalized_session_id)
        if current is None:
            return None
        _touch_heartbeat(current)
        _poll_locked(current)
        return _view_locked(current)


def build_sweep_job_archive(session_id: str) -> bytes:
    """Build the compatibility bytes payload through the streaming writer."""

    payload = io.BytesIO()
    write_sweep_job_archive(session_id, payload)
    return payload.getvalue()


def write_sweep_job_archive(session_id: str, destination: BinaryIO) -> None:
    """Build an archive from exactly the points durably completed so far.

    The manager lock remains held through the local file reads performed by
    the archive builder.  ``release_sweep_job`` therefore cannot delete the
    job directory between path discovery and archive construction.
    """

    with _LOCK:
        job = _JOBS[session_id]
        _poll_locked(job)
        completed_count = len(job.completed_records)
        if completed_count == 0:
            raise ValueError("no completed sweep points are available")
        point_files: list[ArchivePointFile] = []
        completed_indices = sorted(
            _point_index(record, index + 1)
            for index, record in enumerate(job.completed_records)
        )
        for point_index in completed_indices:
            prefix = job.job_dir / "points" / f"{point_index:06d}"
            with prefix.with_suffix(".meta.pkl").open("rb") as stream:
                metadata = pickle.load(stream)
            result_path = prefix.with_suffix(".result.pkl")
            point_files.append(
                ArchivePointFile(
                    record=metadata["record"],
                    config=metadata["config"],
                    result_pickle_path=result_path if result_path.exists() else None,
                )
            )
        config = copy.deepcopy(job.config_snapshot)
        request = copy.deepcopy(job.request_snapshot)
        status = job.status
        total_points = job.total_points
        adjustments = copy.deepcopy(job.compatibility_adjustments)
        parameter = request["parameter"]
        parameters = [
            {
                "path": parameter.path,
                "label": parameter.label,
                "unit": parameter.unit,
                "display_scale": parameter.display_scale,
                "value_field": "parameter_x_value" if int(request.get("dimension", 1)) == 2 else "parameter_value",
                "values": list(request["values"]),
                "spacing": request.get("spacing", "linear"),
            }
        ]
        if int(request.get("dimension", 1)) == 2:
            secondary = request["secondary_parameter"]
            parameters.append(
                {
                    "path": secondary.path,
                    "label": secondary.label,
                    "unit": secondary.unit,
                    "display_scale": secondary.display_scale,
                    "value_field": "parameter_y_value",
                    "values": list(request["secondary_values"]),
                    "spacing": request.get("secondary_spacing", "linear"),
                }
            )
        write_result_archive_from_point_files(
            destination,
            point_files,
            run_kind=(
                "sweep_2d"
                if int(request.get("dimension", 1)) == 2
                else "sweep"
            ),
            base_config=config,
            run_metadata={
                "parameters": parameters,
                "metric_keys": list(request["metric_keys"]),
                "metrics": {
                    key: {
                        "label": SWEEP_METRICS[key].label,
                        "unit": SWEEP_METRICS[key].unit,
                        "display_scale": SWEEP_METRICS[key].display_scale,
                    }
                    for key in request["metric_keys"]
                },
                "completed_points": completed_count,
                "requested_points": total_points,
                "completed_point_indices": completed_indices,
                "point_workers": int(request.get("point_workers", 1)),
                "partial": completed_count < total_points or status != "completed",
                "job_status_at_download": status,
                "ui_compatibility_adjustments": adjustments,
            },
        )
        if status not in ACTIVE_STATUSES:
            # A failed/cancelled sweep cannot produce its uncomputed points.
            # Archiving every durable terminal point satisfies the retention
            # gate too; active partial downloads must never do so.
            written_at = time.time()
            (job.job_dir / _FULL_ARCHIVE_MARKER).write_text(
                dumps_json_portable(
                    {"written_at_epoch_s": written_at},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            job.full_archive_written_at_epoch_s = written_at


def _shutdown_all_jobs() -> None:
    _SUPERVISOR_STOP.set()
    with _LOCK:
        for job in list(_JOBS.values()):
            if job.status in ACTIVE_STATUSES:
                _kill_locked(job, "UI server shutdown")
            if _archive_was_written(job):
                shutil.rmtree(job.job_dir, ignore_errors=True)
        _JOBS.clear()


atexit.register(_shutdown_all_jobs)
