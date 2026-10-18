"""Result-recording policy shared by configuration, solvers, and the UI.

The policy deliberately separates *what physics is enabled* from *how densely
diagnostics are retained*.  In particular, detailed recording aggregates all
adaptive optical microsteps into one row per physical crystal z cell; it never
retains a field-sized array for every microstep.
"""

from __future__ import annotations

import copy
from typing import Any

from .validation import require_exact_integer


RESULT_RECORDING_LEVELS = ("simple", "medium", "detailed")


def diagnostic_observers_enabled(propagator: Any) -> bool:
    """Final-only observers are armed solely by the reported Seed entry."""
    recording = getattr(propagator, "_result_recording", None)
    if recording is None:
        recording = getattr(propagator, "simulation", {}).get("result_recording", {})
    return (recording.get("diagnostics_timing", "all_propagations") != "final_output"
            or bool(getattr(propagator, "physics", {}).get("_final_output_diagnostics", False)))

RESULT_RECORDING_DEFAULTS: dict[str, Any] = {
    "level": "simple",
    "iteration_history": True,
    "diagnostics_timing": "all_propagations",
    "longitudinal": {
        "enabled": True,
        "z_stride": 1,
        "beam": True,
        "gain": True,
        "nonlinear": True,
        "modes": True,
        "m2_checkpoints": True,
        "m2_z_stride": 5,
        "intensity_snapshots": False,
        "intensity_z_stride": 1,
    },
}


def resolve_result_recording(simulation: dict[str, Any]) -> dict[str, Any]:
    """Return the effective density policy without mutating the configuration."""

    configured = simulation.get("result_recording", {})
    merged = copy.deepcopy(RESULT_RECORDING_DEFAULTS)
    if isinstance(configured, dict):
        for key, value in configured.items():
            if key == "longitudinal" and isinstance(value, dict):
                merged["longitudinal"].update(value)
            else:
                merged[key] = value

    level = str(merged.get("level", "simple"))
    medium_or_higher = level in {"medium", "detailed"}
    detailed = level == "detailed"
    longitudinal = merged["longitudinal"]
    longitudinal_enabled = detailed and bool(longitudinal.get("enabled", True))
    return {
        "schema_version": 1,
        "level": level,
        "diagnostics_timing": str(merged.get("diagnostics_timing", "all_propagations")),
        "iteration_history": medium_or_higher
        and bool(merged.get("iteration_history", True)),
        "longitudinal": {
            "enabled": longitudinal_enabled,
            "z_stride": int(longitudinal.get("z_stride", 1)),
            "beam": longitudinal_enabled
            and bool(longitudinal.get("beam", True)),
            "gain": longitudinal_enabled
            and bool(longitudinal.get("gain", True)),
            "nonlinear": longitudinal_enabled
            and bool(longitudinal.get("nonlinear", True)),
            "modes": longitudinal_enabled
            and bool(longitudinal.get("modes", True)),
            "m2_checkpoints": longitudinal_enabled
            and bool(longitudinal.get("m2_checkpoints", True)),
            "m2_z_stride": int(longitudinal.get("m2_z_stride", 5)),
            "intensity_snapshots": longitudinal_enabled
            and bool(longitudinal.get("intensity_snapshots", False)),
            "intensity_z_stride": int(longitudinal.get("intensity_z_stride", 1)),
            "adaptive_substeps_aggregated_to_physical_z_cells": True,
            "retains_field_arrays": longitudinal_enabled
            and bool(longitudinal.get("intensity_snapshots", False)),
            "retains_complex_field_arrays": False,
        },
    }


def validate_result_recording(config: Any) -> None:
    """Validate the public ``simulation.result_recording`` object."""

    if not isinstance(config, dict):
        raise ValueError("simulation.result_recording must be an object")
    level = str(config.get("level", "simple"))
    if level not in RESULT_RECORDING_LEVELS:
        raise ValueError(
            "simulation.result_recording.level must be simple, medium, or detailed"
        )
    if not isinstance(config.get("iteration_history", True), bool):
        raise ValueError(
            "simulation.result_recording.iteration_history must be boolean"
        )
    if config.get("diagnostics_timing", "all_propagations") not in {"all_propagations", "final_output"}:
        raise ValueError("simulation.result_recording.diagnostics_timing must be all_propagations or final_output")
    longitudinal = config.get("longitudinal", {})
    if not isinstance(longitudinal, dict):
        raise ValueError(
            "simulation.result_recording.longitudinal must be an object"
        )
    for key in (
        "enabled",
        "beam",
        "gain",
        "nonlinear",
        "modes",
        "m2_checkpoints",
        "intensity_snapshots",
    ):
        if not isinstance(longitudinal.get(key, True), bool):
            raise ValueError(
                f"simulation.result_recording.longitudinal.{key} must be boolean"
            )
    require_exact_integer(
        longitudinal.get("z_stride", 1),
        name="simulation.result_recording.longitudinal.z_stride",
        minimum=1,
    )
    require_exact_integer(
        longitudinal.get("m2_z_stride", 5),
        name="simulation.result_recording.longitudinal.m2_z_stride",
        minimum=1,
    )
    require_exact_integer(
        longitudinal.get("intensity_z_stride", 1),
        name="simulation.result_recording.longitudinal.intensity_z_stride",
        minimum=1,
    )
