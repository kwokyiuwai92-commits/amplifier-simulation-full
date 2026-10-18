"""Lightweight, opt-in timing for numerically important solver operators.

The profiler is deliberately observational: it records wall-clock duration and
input-array sizes, but never retains arrays or changes execution order.  The
collector is execution-context local, so concurrent simulator threads do not
overwrite one another's enable flag or timing ledger.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import platform
import time
from typing import Any, Callable, Iterator, TypeVar

import numpy as np

try:  # ``resource`` is unavailable on Windows.
    import resource as _resource
except ImportError:  # pragma: no cover - exercised on non-POSIX platforms
    _resource = None


_enabled: ContextVar[bool] = ContextVar(
    "amplifier_sim_operation_profiler_enabled", default=False
)
_operations: ContextVar[dict[str, dict[str, Any]] | None] = ContextVar(
    "amplifier_sim_operation_profile", default=None
)
_profile_started_s: ContextVar[float] = ContextVar(
    "amplifier_sim_operation_profile_started_s", default=time.perf_counter()
)

_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])


def configure_operation_profiler(config: dict[str, Any] | None = None) -> None:
    """Enable or disable operator timing from ``performance`` configuration."""

    _enabled.set(bool((config or {}).get("operator_profiling", False)))


def reset_operation_profiler() -> None:
    """Start an empty profile for one simulator run."""

    _operations.set({})
    _profile_started_s.set(time.perf_counter())


def operation_profiler_enabled() -> bool:
    return _enabled.get()


def _array_observation(value: Any) -> tuple[int, int, tuple[int, ...] | None]:
    if value is None:
        return 0, 0, None
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return 0, 0, None
    return int(array.nbytes), int(array.size), tuple(int(item) for item in array.shape)


def record_operation(
    name: str,
    elapsed_s: float,
    *,
    array: Any = None,
    bytes_processed: int | None = None,
) -> None:
    """Record one completed operation without retaining its input."""

    if not _enabled.get():
        return
    observed_bytes, elements, shape = _array_observation(array)
    byte_count = observed_bytes if bytes_processed is None else max(int(bytes_processed), 0)
    operations = _operations.get()
    if operations is None:
        operations = {}
        _operations.set(operations)
    entry = operations.setdefault(
        str(name),
        {
            "call_count": 0,
            "total_time_s": 0.0,
            "maximum_call_time_s": 0.0,
            "observed_input_bytes": 0,
            "maximum_input_elements": 0,
            "largest_input_shape": None,
        },
    )
    entry["call_count"] += 1
    entry["total_time_s"] += max(float(elapsed_s), 0.0)
    entry["maximum_call_time_s"] = max(
        float(entry["maximum_call_time_s"]), max(float(elapsed_s), 0.0)
    )
    entry["observed_input_bytes"] += byte_count
    if elements > int(entry["maximum_input_elements"]):
        entry["maximum_input_elements"] = elements
        entry["largest_input_shape"] = None if shape is None else list(shape)


@contextmanager
def profile_operation(
    name: str,
    *,
    array: Any = None,
    bytes_processed: int | None = None,
) -> Iterator[None]:
    """Measure one operator call when profiling is enabled."""

    if not _enabled.get():
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        record_operation(
            name,
            time.perf_counter() - started,
            array=array,
            bytes_processed=bytes_processed,
        )


def profiled_operation(name: str) -> Callable[[_CallableT], _CallableT]:
    """Decorate a substantial operator without changing its signature."""

    def decorate(function: _CallableT) -> _CallableT:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not _enabled.get():
                return function(*args, **kwargs)
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                record_operation(name, time.perf_counter() - started)

        return wrapped  # type: ignore[return-value]

    return decorate


def _peak_resident_set_bytes() -> int | None:
    if _resource is None:
        return None
    try:
        maximum_rss = int(
            _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        )
    except (OSError, ValueError):
        return None
    # Darwin reports bytes; Linux and the BSDs used by CI report KiB.
    if platform.system() == "Darwin":
        return maximum_rss
    return maximum_rss * 1024


def operation_profile_snapshot(*, total_run_elapsed_s: float | None = None) -> dict[str, Any]:
    """Return a JSON-safe profile ordered by descending inclusive time."""

    elapsed = (
        max(float(total_run_elapsed_s), 0.0)
        if total_run_elapsed_s is not None
        else max(time.perf_counter() - _profile_started_s.get(), 0.0)
    )
    raw_operations = _operations.get() or {}
    ordered = sorted(
        raw_operations.items(),
        key=lambda item: (-float(item[1]["total_time_s"]), item[0]),
    )
    operations: dict[str, dict[str, Any]] = {}
    for name, raw in ordered:
        count = int(raw["call_count"])
        total = float(raw["total_time_s"])
        operations[name] = {
            **raw,
            "mean_call_time_s": total / max(count, 1),
            "fraction_of_run_wall_time": total / max(elapsed, np.finfo(float).tiny),
        }
    return {
        "enabled": _enabled.get(),
        "run_wall_time_s": elapsed,
        "peak_process_resident_set_bytes": _peak_resident_set_bytes(),
        "peak_memory_scope": "process_lifetime_high_water_mark",
        "timing_scope": (
            "inclusive operator wall time; nested entries overlap and must not be summed"
        ),
        "operations": operations,
    }
