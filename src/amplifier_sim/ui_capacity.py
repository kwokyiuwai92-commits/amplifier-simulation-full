"""Process-wide capacity guard for managed Streamlit numerical workers."""

from __future__ import annotations

import os
import threading


_LOCK = threading.Lock()
_ACTIVE_JOB_IDS: set[str] = set()


def reserve_ui_worker(job_id: str) -> None:
    """Reserve one server-wide worker slot or fail before spawning."""

    raw_limit = os.environ.get("AMPLIFIER_SIM_UI_MAX_ACTIVE_JOBS", "2")
    try:
        limit = max(int(raw_limit), 1)
    except ValueError:
        limit = 2
    with _LOCK:
        if len(_ACTIVE_JOB_IDS) >= limit:
            raise RuntimeError(
                "shared UI worker capacity is full "
                f"({len(_ACTIVE_JOB_IDS)}/{limit}); wait for an active solve "
                "or sweep to finish, or raise AMPLIFIER_SIM_UI_MAX_ACTIVE_JOBS "
                "after sizing host memory"
            )
        _ACTIVE_JOB_IDS.add(str(job_id))


def release_ui_worker(job_id: str) -> None:
    """Release a reservation; repeated terminal polling is harmless."""

    with _LOCK:
        _ACTIVE_JOB_IDS.discard(str(job_id))


def active_ui_worker_count() -> int:
    with _LOCK:
        return len(_ACTIVE_JOB_IDS)
