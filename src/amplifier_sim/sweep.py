"""Reusable one-parameter sweep engine with per-point failure isolation."""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import traceback
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .result_archive import python_warning_record
from .validation import require_exact_integer


SweepPointProgress = Callable[
    [int, int, dict[str, Any], dict[str, Any]],
    None,
]


@dataclass(frozen=True)
class SweepParameter:
    label: str
    path: str
    unit: str
    display_scale: float = 1.0
    integer: bool = False
    fallback_start: float | None = None
    fallback_stop: float | None = None


@dataclass(frozen=True)
class SweepMetric:
    label: str
    unit: str
    display_scale: float = 1.0


@dataclass(frozen=True)
class SweepRun:
    parameter_path: str
    parameter_values: np.ndarray
    records: tuple[dict[str, Any], ...]

    @property
    def successful_records(self) -> tuple[dict[str, Any], ...]:
        """Compatibility alias: only physically valid points are successful."""

        return self.valid_records

    @property
    def valid_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            record
            for record in self.records
            if record.get("status", "valid" if not record.get("error") else "execution_failed")
            in {"valid", "valid_with_warnings"}
        )

    @property
    def warning_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            record
            for record in self.records
            if record.get("status") == "valid_with_warnings"
        )

    @property
    def invalid_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            record
            for record in self.records
            if record.get("status") == "execution_completed_but_invalid"
        )

    @property
    def failed_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            record for record in self.records if record.get("status") == "execution_failed"
        )


@dataclass(frozen=True)
class SweepGridRun:
    parameter_x_path: str
    parameter_y_path: str
    parameter_x_values: np.ndarray
    parameter_y_values: np.ndarray
    records: tuple[dict[str, Any], ...]

    @property
    def successful_records(self) -> tuple[dict[str, Any], ...]:
        return self.valid_records

    @property
    def valid_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            record
            for record in self.records
            if record.get("status", "valid" if not record.get("error") else "execution_failed")
            in {"valid", "valid_with_warnings"}
        )

    @property
    def warning_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            record
            for record in self.records
            if record.get("status") == "valid_with_warnings"
        )

    @property
    def invalid_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            record
            for record in self.records
            if record.get("status") == "execution_completed_but_invalid"
        )

    @property
    def failed_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            record for record in self.records if record.get("status") == "execution_failed"
        )


def _selected_metrics(
    result: Any, selected: tuple[str, ...]
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    # Keep the established extractor seam available to custom integrations
    # while deriving diagnostic validity independently.
    metrics = extract_standard_metrics(result)
    validity = standard_metric_validity(result, metrics)
    return (
        {key: float(metrics[key]) for key in selected},
        {key: dict(validity[key]) for key in selected},
    )


def _complete_sweep_record(
    record: dict[str, Any],
    result: Any,
    selected: tuple[str, ...],
) -> None:
    """Attach metrics and the shared result-validity report to one point."""

    validation = result.summary.get("validation", {})
    record["validation_mode"] = str(validation.get("mode", "exploratory"))
    record["validation_enforced"] = bool(validation.get("enforced", False))
    record["validation_status"] = str(
        validation.get(
            "status",
            "valid" if validation.get("valid", True) else "invalid",
        )
    )
    record["invalid"] = not bool(validation.get("valid", True))
    record["validation_failures"] = list(validation.get("failures", []))
    record["validation_warnings"] = list(validation.get("warnings", []))
    record["solver_warnings"] = list(result.summary.get("warnings", []))
    signal = result.summary.get("signal", {})
    record["requested_signal_model"] = str(
        signal.get("requested_model", signal.get("model", "unknown"))
    )
    record["executed_signal_model"] = str(
        signal.get("executed_model", signal.get("model", "unknown"))
    )
    record["signal_fallback_active"] = bool(
        signal.get(
            "fallback_active",
            record["executed_signal_model"] != record["requested_signal_model"],
        )
    )
    record["output_M2_quantity_kind"] = (
        signal.get("output_beam_quality", {}) or {}
    ).get("reported_quantity_kind")

    metrics, metric_validity = _selected_metrics(result, selected)
    record.update(metrics)
    record["metric_validity"] = metric_validity
    record["invalid_metrics"] = sorted(
        key for key, status in metric_validity.items() if not bool(status["valid"])
    )

    if record["invalid"]:
        record["status"] = "execution_completed_but_invalid"
    elif record["validation_warnings"]:
        record["status"] = "valid_with_warnings"
    else:
        record["status"] = "valid"


SWEEP_PARAMETERS: tuple[SweepParameter, ...] = (
    SweepParameter("泵浦功率", "pump.power_W", "W"),
    SweepParameter("Pump 中心波长", "pump.wavelength_m", "nm", display_scale=1e9),
    SweepParameter("Seed 中心波长", "seed.wavelength_m", "nm", display_scale=1e9),
    SweepParameter("Pump LASER→首元件距离", "derived.pump_source_distance_m", "mm", display_scale=1e3),
    SweepParameter("Seed LASER→首元件距离", "derived.seed_source_distance_m", "mm", display_scale=1e3),
    SweepParameter("Pump 前一元件→首个 GAIN 距离", "derived.pump_gain_distance_m", "mm", display_scale=1e3),
    SweepParameter("Seed 前一元件→首个 GAIN 距离", "derived.seed_gain_distance_m", "mm", display_scale=1e3),
    SweepParameter("Pump x/y 腰斑统一缩放", "derived.pump_waist_xy_scale", "×", fallback_start=0.5, fallback_stop=1.5),
    SweepParameter("Seed x/y 腰斑统一缩放", "derived.seed_waist_xy_scale", "×", fallback_start=0.5, fallback_stop=1.5),
    SweepParameter("重复频率", "simulation.repetition_rate_Hz", "Hz", fallback_start=1e3, fallback_stop=1e5),
    SweepParameter("种子脉冲能量", "seed.pulse_energy_J", "J"),
    SweepParameter("种子 x 腰斑", "seed.waist_x_m", "µm", display_scale=1e6),
    SweepParameter("种子 y 腰斑", "seed.waist_y_m", "µm", display_scale=1e6),
    SweepParameter("泵浦 x 腰斑", "pump.waist_x_m", "µm", display_scale=1e6),
    SweepParameter("泵浦 y 腰斑", "pump.waist_y_m", "µm", display_scale=1e6),
    SweepParameter("晶体长度", "grid.length_z_m", "mm", display_scale=1e3),
    SweepParameter("晶体折射率", "material.refractive_index", ""),
    SweepParameter("掺杂离子密度", "material.ion_density_m3", "m⁻³"),
    SweepParameter("泵浦吸收截面", "energy_levels.pump_absorption_cross_section_m2", "m²"),
    SweepParameter("信号发射截面", "energy_levels.signal_emission_cross_section_m2", "m²"),
    SweepParameter("上能级寿命", "energy_levels.upper_state_lifetime_s", "s"),
    SweepParameter("多程晶体命中次数", "simulation.multipass.pass_count", "次", integer=True, fallback_start=2, fallback_stop=5),
)


SWEEP_METRICS: dict[str, SweepMetric] = {
    "signal_output_energy_J": SweepMetric("输出种子能量", "µJ", 1e6),
    "signal_output_average_power_W": SweepMetric("输出平均功率", "W"),
    "signal_energy_gain": SweepMetric("总能量增益", "×"),
    "signal_output_M2_x": SweepMetric("报告的 Seed M² x（见每点方法）", ""),
    "signal_output_M2_y": SweepMetric("报告的 Seed M² y（见每点方法）", ""),
    "pump_absorbed_power_W": SweepMetric("吸收泵浦功率", "W"),
    "pump_absorption_fraction": SweepMetric("泵浦吸收比例", "%", 100.0),
    "primary_signal_upper_fraction_mean": SweepMetric("泵后主信号上能级占比（平均）", "%", 100.0),
    "primary_signal_upper_fraction_max": SweepMetric("泵后主信号上能级占比（最大）", "%", 100.0),
    "primary_signal_transition_inversion_mean_m3": SweepMetric("泵后主信号跃迁反转密度（平均）", "m⁻³"),
    "primary_signal_transition_inversion_max_m3": SweepMetric("泵后主信号跃迁反转密度（最大）", "m⁻³"),
    # Compatibility aliases retained for old sweep presets.  The labels now
    # state the exact reservoir/transition scope rather than implying a
    # unique upper state in an arbitrary custom level scheme.
    "after_pump_upper_fraction_mean": SweepMetric("泵后主信号上能级占比（平均，兼容名）", "%", 100.0),
    "after_pump_upper_fraction_max": SweepMetric("泵后主信号上能级占比（最大，兼容名）", "%", 100.0),
    "after_pump_inversion_mean_m3": SweepMetric("泵后主信号跃迁反转密度（平均，兼容名）", "m⁻³"),
    "after_pump_inversion_max_m3": SweepMetric("泵后主信号跃迁反转密度（最大，兼容名）", "m⁻³"),
    "heat_total_W": SweepMetric("总热负载", "W"),
    "heat_max_W_m3": SweepMetric("最大热功率密度", "W/m³"),
    "temperature_max_K": SweepMetric("晶体最高温度", "K"),
    "temperature_rise_max_K": SweepMetric("晶体最大温升", "K"),
    "extracted_net_J": SweepMetric("种子净提取能量", "µJ", 1e6),
    "extracted_positive_J": SweepMetric("正提取能量", "µJ", 1e6),
    "gain_max_m_inv": SweepMetric("最终最大局部增益", "m⁻¹"),
}

LEGACY_SWEEP_METRIC_KEYS = {
    "after_pump_upper_fraction_mean",
    "after_pump_upper_fraction_max",
    "after_pump_inversion_mean_m3",
    "after_pump_inversion_max_m3",
}


def get_config_value(config: dict[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for part in dotted_path.split("."):
        value = value[part]
    return value


def set_config_value(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    target: Any = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def _configured_beam_path(config: dict[str, Any], beam_kind: str) -> list[dict[str, Any]]:
    return config["optical_train"][beam_kind]


def _first_gain_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first physical gain encounter without coupling later passes."""

    return next((row for row in rows if row.get("type") == "gain_medium"), None)


def get_sweep_parameter_value(config: dict[str, Any], parameter: SweepParameter) -> float:
    path = parameter.path
    if path == "derived.pump_source_distance_m":
        rows = _configured_beam_path(config, "pump")
        return float(rows[0].get("distance_before_m", 0.0)) if rows else 0.0
    if path == "derived.seed_source_distance_m":
        rows = _configured_beam_path(config, "seed")
        return float(rows[0].get("distance_before_m", 0.0)) if rows else 0.0
    if path in {"derived.pump_gain_distance_m", "derived.seed_gain_distance_m"}:
        beam_kind = "pump" if ".pump_" in path else "seed"
        gain = _first_gain_row(_configured_beam_path(config, beam_kind))
        return float(gain.get("distance_before_m", 0.0)) if gain is not None else 0.0
    if path in {"derived.pump_waist_xy_scale", "derived.seed_waist_xy_scale"}:
        return 1.0
    return float(get_config_value(config, path))


def apply_sweep_parameter(config: dict[str, Any], parameter: SweepParameter, value: int | float) -> None:
    reason = sweep_parameter_inactive_reason(config, parameter)
    if reason is not None:
        raise ValueError(f"Cannot sweep {parameter.path}: {reason}")
    path = parameter.path
    if path in {"derived.pump_source_distance_m", "derived.seed_source_distance_m"}:
        beam_kind = "pump" if ".pump_" in path else "seed"
        rows = _configured_beam_path(config, beam_kind)
        if not rows:
            rows.append({"type": "gain_medium", "name": f"{beam_kind}_gain"})
        rows[0]["distance_before_m"] = float(value)
        return
    if path in {"derived.pump_gain_distance_m", "derived.seed_gain_distance_m"}:
        beam_kind = "pump" if ".pump_" in path else "seed"
        rows = _configured_beam_path(config, beam_kind)
        gain = _first_gain_row(rows)
        if gain is None:
            gain = {
                "type": "gain_medium",
                "name": f"{beam_kind}_gain",
                "distance_before_m": 0.0,
            }
            rows.append(gain)
        gain["distance_before_m"] = float(value)
        return
    if path in {"derived.pump_waist_xy_scale", "derived.seed_waist_xy_scale"}:
        beam_kind = "pump" if ".pump_" in path else "seed"
        factor = float(value)
        config[beam_kind]["waist_x_m"] = float(config[beam_kind]["waist_x_m"]) * factor
        config[beam_kind]["waist_y_m"] = float(config[beam_kind]["waist_y_m"]) * factor
        return
    set_config_value(config, path, value)


def sweep_parameter_inactive_reason(
    config: dict[str, Any], parameter: SweepParameter,
) -> str | None:
    """Identify inputs replaced by a different, explicitly configured source.

    A flat scan of a discarded scalar is not a physical sensitivity result.
    Keep the source table intact and require a scan of its actual input instead.
    """
    path = parameter.path
    levels = config.get("energy_levels", {})
    for role in ("pump", "signal"):
        for channel in ("absorption", "emission"):
            if path != f"energy_levels.{role}_{channel}_cross_section_m2":
                continue
            if levels.get("model") == "custom":
                return "custom optical transitions supply their own cross sections"
            if levels.get(f"{role}_cross_section_spectrum"):
                return f"{role}_cross_section_spectrum replaces this scalar; edit the spectral data explicitly"
            response = config.get("material", {}).get("spectroscopic_response", {})
            law = response.get("cross_sections", {}).get(f"{role}_{channel}", {})
            if response.get("mode", "off") != "off" and law.get("model") == "absolute_temperature_wavelength_table":
                return "an absolute temperature/wavelength table replaces this scalar"
    if path in {"pump.waist_x_m", "pump.waist_y_m", "derived.pump_waist_xy_scale"}:
        mode = config.get("pump", {}).get("source_definition", {}).get("mode", "waist_m2")
        if mode != "waist_m2":
            return f"pump.source_definition ({mode}) supplies the source radii; sweep its radius fields instead"
    if path == "seed.pulse_energy_J":
        seed = config.get("seed", {})
        spectrum = seed.get("spectrum", {})
        if (seed.get("pulse_definition", {}).get("domain", "independent") != "time"
                and spectrum.get("type") == "file"
                and spectrum.get("value_mode", "relative") in {"energy_per_bin_J", "energy_density_J_per_nm"}):
            return "the absolute-energy spectrum file supplies the pulse energy"
    return None


def validate_sweep_parameter_pair(
    config: dict[str, Any], parameter_x: SweepParameter, parameter_y: SweepParameter,
) -> None:
    """Reject two labels that write the very same physical input."""
    if parameter_x.path == parameter_y.path:
        raise ValueError("Two-dimensional sweep parameters must be different")
    paths = {parameter_x.path, parameter_y.path}
    for beam in ("pump", "seed"):
        aliases = {f"derived.{beam}_source_distance_m", f"derived.{beam}_gain_distance_m"}
        if paths == aliases:
            rows = _configured_beam_path(config, beam)
            if not rows or rows[0].get("type") == "gain_medium":
                raise ValueError(
                    f"Both sweep axes write the same {beam} source-to-gain distance; "
                    "select independent parameters or insert an intervening optical element"
                )


def make_sweep_values(
    start: float,
    stop: float,
    count: int,
    spacing: str = "linear",
    integer: bool = False,
) -> np.ndarray:
    if count < 1:
        raise ValueError("Sweep point count must be positive")
    if spacing == "log":
        if start <= 0.0 or stop <= 0.0:
            raise ValueError("Logarithmic sweep endpoints must be positive")
        values = np.geomspace(start, stop, count)
    elif spacing == "linear":
        values = np.linspace(start, stop, count)
    else:
        raise ValueError("Sweep spacing must be linear or log")
    if integer:
        values = np.unique(np.maximum(np.rint(values).astype(int), 1))
    return values


def extract_standard_metrics_with_validity(
    result: Any,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Return metrics plus independent per-metric validity contracts.

    A completed simulation can be physically valid even when one diagnostic,
    such as sampled M², is unavailable.  Conversely, finite numbers from an
    invalid run remain diagnostic values and do not become valid results.
    """

    summary = result.summary
    density = (
        np.sum(result.population_after_pump_m3, axis=0)
        if getattr(result, "dopant_density_m3", None) is None
        else np.asarray(result.dopant_density_m3, dtype=float)
    )
    upper_name = summary["population"]["upper_level"]
    upper_index = result.level_names.index(upper_name)
    upper_fraction = np.divide(
        result.population_after_pump_m3[upper_index],
        density,
        out=np.zeros_like(density),
        where=density > 0.0,
    )
    crystal = (
        np.ones(result.grid.shape, dtype=bool)
        if getattr(result, "crystal_mask", None) is None
        else np.asarray(result.crystal_mask, dtype=bool)
    )
    if not np.any(crystal):
        raise ValueError("Sweep metrics require at least one active crystal cell")
    dopant_active = crystal & np.isfinite(density) & (density > 0.0)
    if not np.any(dopant_active):
        raise ValueError(
            "Population sweep metrics require at least one doped crystal cell"
        )
    cell_volumes = np.asarray(getattr(result.grid, "cell_volumes", np.ones(result.grid.shape)))
    doped_volumes = cell_volumes[dopant_active]
    pump_input = float(summary["pump"].get("grid_input_power_W", 0.0))
    pump_absorbed = float(summary["pump"].get("absorbed_power_W", 0.0))
    reference = float(summary["thermal"]["reference_temperature_K"])
    output_quality = summary.get("signal", {}).get("output_beam_quality", {})
    quality_valid = bool(output_quality.get("valid", False))
    m2_x = output_quality.get("M2_x")
    m2_y = output_quality.get("M2_y")
    metrics = {
        "signal_output_energy_J": float(summary["signal"]["system_output_energy_J"]),
        "signal_output_average_power_W": float(summary["signal"].get("system_output_average_power_W") or 0.0),
        "signal_energy_gain": float(summary["signal"]["energy_gain_source_to_output"]),
        "signal_output_M2_x": float(np.nan if m2_x is None else m2_x),
        "signal_output_M2_y": float(np.nan if m2_y is None else m2_y),
        "pump_absorbed_power_W": pump_absorbed,
        "pump_absorption_fraction": pump_absorbed / max(pump_input, np.finfo(float).tiny),
        "primary_signal_upper_fraction_mean": float(np.average(upper_fraction[dopant_active], weights=doped_volumes)),
        "primary_signal_upper_fraction_max": float(np.max(upper_fraction[dopant_active])),
        "primary_signal_transition_inversion_mean_m3": float(
            np.average(result.inversion_after_pump_m3[dopant_active], weights=doped_volumes)
        ),
        "primary_signal_transition_inversion_max_m3": float(
            np.max(result.inversion_after_pump_m3[dopant_active])
        ),
        "heat_total_W": float(summary["thermal"]["heat_total_W"]),
        "heat_max_W_m3": float(np.max(result.heat_W_m3[crystal])),
        "temperature_max_K": float(np.max(result.temperature_K[crystal])),
        "temperature_rise_max_K": float(
            np.max(result.temperature_K[crystal]) - reference
        ),
        "extracted_net_J": float(summary["extraction"]["net_J"]),
        "extracted_positive_J": float(summary["extraction"]["positive_J"]),
        "gain_max_m_inv": float(np.max(result.gain_m_inv[crystal])),
    }
    metrics["after_pump_upper_fraction_mean"] = metrics[
        "primary_signal_upper_fraction_mean"
    ]
    metrics["after_pump_upper_fraction_max"] = metrics[
        "primary_signal_upper_fraction_max"
    ]
    metrics["after_pump_inversion_mean_m3"] = metrics[
        "primary_signal_transition_inversion_mean_m3"
    ]
    metrics["after_pump_inversion_max_m3"] = metrics[
        "primary_signal_transition_inversion_max_m3"
    ]
    validity: dict[str, dict[str, Any]] = {}
    for key, value in metrics.items():
        valid = bool(np.isfinite(value))
        validity[key] = {
            "valid": valid,
            "reason": None if valid else "metric is non-finite",
        }
    for key in ("signal_output_M2_x", "signal_output_M2_y"):
        if not quality_valid:
            validity[key] = {
                "valid": False,
                "reason": (
                    "output beam-quality diagnostic is invalid; enlarge/refine "
                    "the sampled optical window before interpreting M²"
                ),
            }
    return metrics, validity


def extract_standard_metrics(result: Any) -> dict[str, float]:
    """Compatibility metric-only view used by convergence and old clients."""

    return extract_standard_metrics_with_validity(result)[0]


def standard_metric_validity(
    result: Any, metrics: dict[str, float]
) -> dict[str, dict[str, Any]]:
    """Build per-metric validity for built-in or external metric extractors."""

    validity = {
        key: {
            "valid": bool(np.isfinite(value)),
            "reason": None if np.isfinite(value) else "metric is non-finite",
        }
        for key, value in metrics.items()
    }
    output_quality = (
        getattr(result, "summary", {})
        .get("signal", {})
        .get("output_beam_quality")
    )
    if isinstance(output_quality, dict) and not bool(
        output_quality.get("valid", False)
    ):
        for key in ("signal_output_M2_x", "signal_output_M2_y"):
            if key in validity:
                validity[key] = {
                    "valid": False,
                    "reason": (
                        "output beam-quality diagnostic is invalid; enlarge/refine "
                        "the sampled optical window before interpreting M²"
                    ),
                }
    return validity


_compact_worker_base_config: dict[str, Any] | None = None
_compact_worker_metrics: tuple[str, ...] = ()
_compact_worker_fft_workers = 1


def _coherent_kernel_workers_for_processes(
    config: dict[str, Any], process_workers: int
) -> int:
    configured = int(
        config.get("seed", {})
        .get("coherent", {})
        .get("transient_gain", {})
        .get("single_transition_kernel_workers", 0)
    )
    return 1 if int(process_workers) > 1 and configured == 0 else configured


def _initialize_compact_sweep_worker(
    base_config: dict[str, Any],
    selected: tuple[str, ...],
    fft_workers: int,
    process_workers: int,
) -> None:
    global _compact_worker_base_config, _compact_worker_metrics
    global _compact_worker_fft_workers
    _compact_worker_base_config = copy.deepcopy(base_config)
    _compact_worker_metrics = tuple(selected)
    _compact_worker_fft_workers = int(fft_workers)
    transient_gain = (
        _compact_worker_base_config.setdefault("seed", {})
        .setdefault("coherent", {})
        .setdefault("transient_gain", {})
    )
    configured_kernel_workers = _coherent_kernel_workers_for_processes(
        _compact_worker_base_config, process_workers
    )
    if configured_kernel_workers > 0:
        transient_gain["single_transition_kernel_workers"] = (
            configured_kernel_workers
        )


def _execute_compact_sweep_task(
    payload: tuple[
        dict[str, Any],
        tuple[tuple[SweepParameter, int | float], ...],
    ],
) -> dict[str, Any]:
    """Run one isolated point and return no volumetric SimulationResult."""

    from .solver import AmplifierSimulator

    record, assignments = payload
    if _compact_worker_base_config is None:
        raise RuntimeError("compact sweep worker was not initialized")
    local_config = copy.deepcopy(_compact_worker_base_config)
    local_config.setdefault("performance", {})["fft_workers"] = (
        _compact_worker_fft_workers
    )
    local_config["performance"]["operator_profiling"] = False
    caught_warnings: list[Any] = []
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            for parameter, assigned in assignments:
                apply_sweep_parameter(local_config, parameter, assigned)
            result = AmplifierSimulator(local_config).run()
        _complete_sweep_record(record, result, _compact_worker_metrics)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc)
        record["traceback"] = traceback.format_exc()
        record["status"] = "execution_failed"
    record["python_warnings"] = [
        python_warning_record(warning) for warning in caught_warnings
    ]
    record["result_retention"] = "compact_metrics_validation_only"
    return record


def _run_compact_sweep_tasks(
    base_config: dict[str, Any],
    selected: tuple[str, ...],
    tasks: list[
        tuple[
            dict[str, Any],
            tuple[tuple[SweepParameter, int | float], ...],
        ]
    ],
    *,
    workers: int,
    fft_workers_per_process: int,
    progress: Callable[[int, int, dict[str, Any]], None] | None,
) -> list[dict[str, Any]]:
    worker_count = require_exact_integer(
        workers, name="sweep.workers", minimum=1
    )
    fft_workers = require_exact_integer(
        fft_workers_per_process,
        name="sweep.fft_workers_per_process",
        minimum=1,
    )
    worker_count = min(worker_count, max(len(tasks), 1))
    if worker_count == 1:
        _initialize_compact_sweep_worker(
            base_config, selected, fft_workers, worker_count
        )
        iterator = map(_execute_compact_sweep_task, tasks)
        executor = None
    else:
        context = multiprocessing.get_context("spawn")
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_compact_sweep_worker,
            initargs=(base_config, selected, fft_workers, worker_count),
        )
        iterator = executor.map(_execute_compact_sweep_task, tasks)
    records: list[dict[str, Any]] = []
    try:
        for completed, record in enumerate(iterator, start=1):
            record["execution"] = {
                "process_workers": worker_count,
                "fft_workers_per_process": fft_workers,
                "coherent_kernel_workers_per_process": (
                    _coherent_kernel_workers_for_processes(
                        base_config, worker_count
                    )
                ),
                "base_config_transfer": "initialized_once_per_worker",
            }
            records.append(record)
            if progress is not None:
                progress(completed, len(tasks), record)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return records


def run_compact_parameter_sweep(
    base_config: dict[str, Any],
    parameter: SweepParameter,
    values: Iterable[float],
    metric_keys: Iterable[str] | None = None,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    *,
    workers: int = 1,
    fft_workers_per_process: int = 1,
) -> SweepRun:
    """Run a metric-only sweep in isolated processes without large-array IPC.

    Use :func:`run_parameter_sweep` when every point must retain its full field
    result for an archive.  This compact variant is intended for coarse
    screening; selected points can then be rerun with the full solver contract.
    """

    selected = tuple(metric_keys or SWEEP_METRICS)
    unknown = set(selected) - set(SWEEP_METRICS)
    if unknown:
        raise ValueError(f"Unknown sweep metrics: {sorted(unknown)}")
    point_values = np.asarray(list(values), dtype=float)
    tasks = []
    for index, raw_value in enumerate(point_values):
        assigned: int | float = (
            int(round(float(raw_value)))
            if parameter.integer
            else float(raw_value)
        )
        tasks.append(
            (
                {
                    "point_index": index,
                    "parameter_value": assigned,
                    "status": "execution_failed",
                    "error": None,
                    "error_type": None,
                    "error_message": None,
                    "traceback": None,
                },
                ((parameter, assigned),),
            )
        )
    records = _run_compact_sweep_tasks(
        base_config,
        selected,
        tasks,
        workers=workers,
        fft_workers_per_process=fft_workers_per_process,
        progress=progress,
    )
    return SweepRun(parameter.path, point_values, tuple(records))


def run_compact_parameter_grid(
    base_config: dict[str, Any],
    parameter_x: SweepParameter,
    values_x: Iterable[float],
    parameter_y: SweepParameter,
    values_y: Iterable[float],
    metric_keys: Iterable[str] | None = None,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    *,
    workers: int = 1,
    fft_workers_per_process: int = 1,
) -> SweepGridRun:
    """Run a two-dimensional metric-only grid with process isolation."""

    validate_sweep_parameter_pair(base_config, parameter_x, parameter_y)
    selected = tuple(metric_keys or SWEEP_METRICS)
    unknown = set(selected) - set(SWEEP_METRICS)
    if unknown:
        raise ValueError(f"Unknown sweep metrics: {sorted(unknown)}")
    x_values = np.asarray(list(values_x), dtype=float)
    y_values = np.asarray(list(values_y), dtype=float)
    tasks = []
    point_index = 0
    for y_index, raw_y in enumerate(y_values):
        assigned_y: int | float = (
            int(round(float(raw_y)))
            if parameter_y.integer
            else float(raw_y)
        )
        for x_index, raw_x in enumerate(x_values):
            assigned_x: int | float = (
                int(round(float(raw_x)))
                if parameter_x.integer
                else float(raw_x)
            )
            tasks.append(
                (
                    {
                        "point_index": point_index,
                        "parameter_x_index": x_index,
                        "parameter_y_index": y_index,
                        "parameter_x_value": assigned_x,
                        "parameter_y_value": assigned_y,
                        "status": "execution_failed",
                        "error": None,
                        "error_type": None,
                        "error_message": None,
                        "traceback": None,
                    },
                    (
                        (parameter_x, assigned_x),
                        (parameter_y, assigned_y),
                    ),
                )
            )
            point_index += 1
    records = _run_compact_sweep_tasks(
        base_config,
        selected,
        tasks,
        workers=workers,
        fft_workers_per_process=fft_workers_per_process,
        progress=progress,
    )
    return SweepGridRun(
        parameter_x.path,
        parameter_y.path,
        x_values,
        y_values,
        tuple(records),
    )


# Only optical input parameters which leave the physical/material grid intact.
CONTINUATION_PARAMETER_PATHS = frozenset({
    "pump.power_W",
    "pump.waist_x_m", "pump.waist_y_m", "seed.waist_x_m", "seed.waist_y_m",
    "derived.pump_waist_xy_scale", "derived.seed_waist_xy_scale",
    "derived.pump_gain_distance_m", "derived.seed_gain_distance_m",
    "derived.pump_source_distance_m", "derived.seed_source_distance_m",
})


def supports_sweep_continuation(*parameters: SweepParameter) -> bool:
    """All varied axes must preserve the state grid and material definition."""
    return bool(parameters) and all(
        getattr(parameter, "path", None) in CONTINUATION_PARAMETER_PATHS
        for parameter in parameters
    )


def request_uses_continuation(request: dict[str, Any]) -> bool:
    parameters = [request.get("parameter")]
    if int(request.get("dimension", 1)) == 2:
        parameters.append(request.get("secondary_parameter"))
    return supports_sweep_continuation(*parameters)


def sweep_x_indices(count: int, row: int, continuation: bool) -> range:
    """Snake traversal keeps consecutive rows adjacent, without moving axes."""
    return range(count - 1, -1, -1) if continuation and row % 2 else range(count)


def _state_is_converged(result: Any) -> bool:
    summary = getattr(result, "summary", {})
    # Numerical convergence and overall diagnostic validity are distinct.
    return all(section.get("converged") is None or bool(section["converged"]) for section in (
        summary.get("multiphysics", {}), summary.get("population", {}),
        summary.get("thermal", {}),
    ))


def run_parameter_sweep(
    base_config: dict[str, Any],
    parameter: SweepParameter,
    values: Iterable[float],
    metric_keys: Iterable[str] | None = None,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    point_progress: SweepPointProgress | None = None,
    progress_mode: str = "detailed",
    initial_continuation_state: Any | None = None,
    continuation_enabled: bool | None = None,
    _additional_assignments: tuple[tuple[SweepParameter, float], ...] = (),
) -> SweepRun:
    """Run simulations with point-level failure isolation.

    ``progress`` reports one completed point. ``point_progress`` forwards the
    same structured solver events used by a single run while a point is still
    executing. Keeping the two callbacks separate lets callers update a
    lightweight sweep summary without discarding the current point's detailed
    physical progress. Supported optical-input sweeps continue from the
    preceding converged point; the first point uses the configured initializer.
    """

    # Lazy import prevents importing the full simulator when a UI only needs
    # sweep metadata or value generation.
    from .solver import AmplifierSimulator

    selected = tuple(metric_keys or SWEEP_METRICS)
    unknown = set(selected) - set(SWEEP_METRICS)
    if unknown:
        raise ValueError(f"Unknown sweep metrics: {sorted(unknown)}")
    point_values = np.asarray(list(values), dtype=float)
    records: list[dict[str, Any]] = []
    total = len(point_values)
    pump_continuation_enabled = supports_sweep_continuation(
        parameter, *(p for p, _ in _additional_assignments)
    ) and continuation_enabled is not False
    continuation_state = (
        initial_continuation_state if pump_continuation_enabled else None
    )
    continuation_source_value: int | float | None = None
    for index, raw_value in enumerate(point_values):
        # Keep only the newest transferable state. Results already retain their
        # own physical fields; keeping one extra state per historical point
        # would needlessly duplicate several full 3-D arrays.
        if records:
            records[-1].pop("_continuation_state", None)
        assigned: int | float = int(round(float(raw_value))) if parameter.integer else float(raw_value)
        local_config = copy.deepcopy(base_config)
        record: dict[str, Any] = {
            "point_index": index,
            "parameter_value": assigned,
            "status": "execution_failed",
            "error": None,
            "error_type": None,
            "error_message": None,
            "traceback": None,
            "pump_continuation": {
                "enabled": pump_continuation_enabled,
                "used": False,
                "source": None,
                "source_parameter_value": continuation_source_value,
                "state_exported": False,
                "state_retained_after_point_error": False,
            },
        }
        # Keep the historical key readable by existing archive consumers.
        record["continuation"] = record["pump_continuation"]
        caught_warnings: list[Any] = []
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                apply_sweep_parameter(local_config, parameter, assigned)
                for extra_parameter, extra_value in _additional_assignments:
                    apply_sweep_parameter(local_config, extra_parameter, extra_value)
                if point_progress is None:
                    simulator = AmplifierSimulator(local_config)
                else:

                    def report_point(event: dict[str, Any]) -> None:
                        point_progress(index + 1, total, record, event)

                    simulator_kwargs: dict[str, Any] = {
                        "progress_callback": report_point,
                    }
                    if progress_mode != "detailed":
                        simulator_kwargs["progress_mode"] = progress_mode
                    simulator = AmplifierSimulator(
                        local_config, **simulator_kwargs
                    )
                if pump_continuation_enabled and continuation_state is not None:
                    try:
                        simulator.set_continuation_state(continuation_state)
                    except (ValueError, TypeError) as exc:
                        record["continuation"]["fallback_reason"] = str(exc)
                        continuation_state = None
                        continuation_source_value = None
                    else:
                        record["continuation"].update(
                            used=True,
                            source=("most_recent_completed_pump_point"
                                    if parameter.path == "pump.power_W"
                                    else "most_recent_completed_sweep_point"),
                        )
                result = simulator.run()
            _complete_sweep_record(record, result, selected)
            record["_result"] = result
            if pump_continuation_enabled:
                export_state = getattr(
                    simulator, "export_continuation_state", None
                )
                if callable(export_state) and _state_is_converged(result):
                    continuation_state = export_state(result)
                    continuation_source_value = assigned
                    record["_continuation_state"] = continuation_state
                    record["pump_continuation"]["state_exported"] = True
                elif continuation_state is not None:
                    record["_continuation_state"] = continuation_state
                    record["continuation"]["state_retained_after_unconverged_point"] = True
                else:
                    continuation_source_value = None
        except Exception as exc:  # point-level isolation is intentional
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["error_type"] = type(exc).__name__
            record["error_message"] = str(exc)
            record["traceback"] = traceback.format_exc()
            record["status"] = "execution_failed"
            if pump_continuation_enabled and continuation_state is not None:
                # A bad/stalled numerical point has no new state to export,
                # but it must not erase the last completed Pump operating
                # point.  The next point may continue from that older state.
                record["_continuation_state"] = continuation_state
                record["pump_continuation"][
                    "state_retained_after_point_error"
                ] = True
        record["python_warnings"] = [
            python_warning_record(warning) for warning in caught_warnings
        ]
        record["_config"] = local_config
        records.append(record)
        if progress is not None:
            progress(index + 1, total, record)
    return SweepRun(parameter.path, point_values, tuple(records))


def run_parameter_grid(
    base_config: dict[str, Any],
    parameter_x: SweepParameter,
    values_x: Iterable[float],
    parameter_y: SweepParameter,
    values_y: Iterable[float],
    metric_keys: Iterable[str] | None = None,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    point_progress: SweepPointProgress | None = None,
    progress_mode: str = "detailed",
    initial_continuation_state: Any | None = None,
) -> SweepGridRun:
    """Run a Cartesian grid in snake order when both axes allow continuation.

    Coordinates always retain their original indices. Only the latest state is
    retained; failures cannot replace it. Unsupported axes use independent runs.
    """
    validate_sweep_parameter_pair(base_config, parameter_x, parameter_y)
    selected = tuple(metric_keys or SWEEP_METRICS)
    unknown = set(selected) - set(SWEEP_METRICS)
    if unknown:
        raise ValueError(f"Unknown sweep metrics: {sorted(unknown)}")
    x_values = np.asarray(list(values_x), dtype=float)
    y_values = np.asarray(list(values_y), dtype=float)
    enabled = supports_sweep_continuation(parameter_x, parameter_y)
    state = initial_continuation_state if enabled else None
    source = None
    records = []
    total = len(x_values) * len(y_values)
    for yi, raw_y in enumerate(y_values):
        y = int(round(float(raw_y))) if parameter_y.integer else float(raw_y)
        for xi in sweep_x_indices(len(x_values), yi, enabled):
            if records:
                records[-1].pop("_continuation_state", None)
            x = int(round(float(x_values[xi]))) if parameter_x.integer else float(x_values[xi])
            index = len(records)
            coordinates = dict(point_index=index, parameter_x_index=xi,
                               parameter_y_index=yi, parameter_x_value=x, parameter_y_value=y)
            def report(_point, _total, record, event):
                record.update(coordinates)
                record["continuation"]["source_coordinates"] = source
                if point_progress is not None:
                    point_progress(index + 1, total, record, event)
            # The one-point engine applies X then Y to a fresh base copy and
            # isolates parameter/solver failures for both axes.
            run = run_parameter_sweep(
                base_config, parameter_x, [x], selected,
                point_progress=report if point_progress is not None else None,
                progress_mode=progress_mode, initial_continuation_state=state,
                continuation_enabled=enabled,
                _additional_assignments=((parameter_y, y),),
            )
            record = run.records[0]
            record.update(coordinates)
            record["continuation"]["source_coordinates"] = source
            if record["continuation"]["state_exported"]:
                source = dict(x_index=xi, y_index=yi, x_value=x, y_value=y)
            state = record.get("_continuation_state")
            if state is None:
                source = None
            records.append(record)
            if progress is not None:
                progress(index + 1, total, record)
    return SweepGridRun(parameter_x.path, parameter_y.path, x_values, y_values, tuple(records))
