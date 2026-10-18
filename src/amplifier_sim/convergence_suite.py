"""Automated numerical-refinement studies for the main solver observables."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Iterable

import numpy as np

from .config import resolve_physics, resolve_signal_model
from .sweep import standard_metric_validity


DEFAULT_CONVERGENCE_METRICS: tuple[str, ...] = (
    "signal_output_energy_J",
    "pump_absorbed_power_W",
    "temperature_max_K",
    "extracted_net_J",
    "gain_max_m_inv",
)


@dataclass(frozen=True)
class ConvergenceAxis:
    name: str
    label: str
    applicability: Callable[[dict[str, Any]], bool]
    refine: Callable[[dict[str, Any], float], dict[str, Any]]
    interpretation: str = "refinement"


def _scaled_count(value: Any, factor: float, *, minimum: int = 1) -> int:
    return max(int(round(float(value) * factor)), minimum)


def _physical_axis(key: str) -> Callable[[dict[str, Any], float], dict[str, Any]]:
    def refine(config: dict[str, Any], factor: float) -> dict[str, Any]:
        config["grid"][key] = _scaled_count(config["grid"][key], factor, minimum=2)
        updated = {f"grid.{key}": int(config["grid"][key])}
        if key in {"nx", "ny"} and config["grid"].get("spacing_xy") == "refined_region":
            region = config["grid"]["refined_region"]
            region[key] = _scaled_count(region[key], factor, minimum=2)
            updated[f"grid.refined_region.{key}"] = int(region[key])
        return updated

    return refine


def _optical_window_applicable(config: dict[str, Any]) -> bool:
    return bool(config["grid"].get("optical", {}).get("enabled", False)) or (
        resolve_physics(config).get("optical_propagation")
        in {"abcd", "fft", "hybrid_abcd_fft"}
    )


def _refine_optical_window(
    config: dict[str, Any], factor: float
) -> dict[str, Any]:
    base_size_x = float(config["grid"]["size_x_m"])
    base_size_y = float(config["grid"]["size_y_m"])
    geometry = config["material"].setdefault("geometry", {})
    if str(geometry.get("shape", "cuboid")) == "cylinder":
        if geometry.get("diameter_m") is None:
            geometry["diameter_m"] = min(base_size_x, base_size_y)
    else:
        if geometry.get("width_x_m") is None:
            geometry["width_x_m"] = base_size_x
        if geometry.get("height_y_m") is None:
            geometry["height_y_m"] = base_size_y
    optical = config["grid"].get("optical", {})
    target = optical if optical.get("enabled", False) else config["grid"]
    original_target_x = float(target["size_x_m"])
    original_target_y = float(target["size_y_m"])
    if target is optical:
        # The independently sampled optical window is embedded in the fixed
        # physical host.  Growing it may use available host margin, but may
        # never silently resize/coarsen the thermal/population/FEM grid.
        center_x = float(target.get("center_x_m", 0.0))
        center_y = float(target.get("center_y_m", 0.0))
        host_center_x = float(config["grid"].get("center_x_m", 0.0))
        host_center_y = float(config["grid"].get("center_y_m", 0.0))
        maximum_x = max(base_size_x - 2.0 * abs(center_x - host_center_x), 0.0)
        maximum_y = max(base_size_y - 2.0 * abs(center_y - host_center_y), 0.0)
        target["size_x_m"] = min(original_target_x * factor, maximum_x)
        target["size_y_m"] = min(original_target_y * factor, maximum_y)
        # Change only the represented domain: preserve the original dx/dy so
        # the independent sampling axis remains a genuinely separate study.
        target["nx"] = _scaled_count(
            target["nx"], target["size_x_m"] / original_target_x, minimum=2
        )
        target["ny"] = _scaled_count(
            target["ny"], target["size_y_m"] / original_target_y, minimum=2
        )
    else:
        # Without a separate optical grid this is necessarily a joint-domain
        # extension.  Preserve physical dx/dy by increasing Nx/Ny with size.
        target["size_x_m"] = original_target_x * factor
        target["size_y_m"] = original_target_y * factor
        target["nx"] = _scaled_count(target["nx"], factor, minimum=2)
        target["ny"] = _scaled_count(target["ny"], factor, minimum=2)
    prefix = "grid.optical" if target is optical else "grid"
    updated = {
        f"{prefix}.size_x_m": float(target["size_x_m"]),
        f"{prefix}.size_y_m": float(target["size_y_m"]),
    }
    if target is optical:
        updated.update(
            {
                f"{prefix}.nx": int(target["nx"]),
                f"{prefix}.ny": int(target["ny"]),
                "refinement_contract": "independent_optical_window_with_fixed_physical_grid",
                "transverse_sampling_contract": "preserve_baseline_dx_dy",
            }
        )
        updated["window_capped_by_physical_domain"] = bool(
            original_target_x * factor > maximum_x
            or original_target_y * factor > maximum_y
        )
    else:
        updated.update(
            {
                "grid.nx": int(target["nx"]),
                "grid.ny": int(target["ny"]),
                "refinement_contract": "joint_domain_extension_at_fixed_physical_dx_dy",
            }
        )
    if str(geometry.get("shape", "cuboid")) == "cylinder":
        updated["material.geometry.diameter_m"] = float(geometry["diameter_m"])
    else:
        updated["material.geometry.width_x_m"] = float(geometry["width_x_m"])
        updated["material.geometry.height_y_m"] = float(geometry["height_y_m"])
    return updated


def _refine_optical_sampling(
    config: dict[str, Any], factor: float
) -> dict[str, Any]:
    optical = config["grid"].get("optical", {})
    target = optical if optical.get("enabled", False) else config["grid"]
    target["nx"] = _scaled_count(target["nx"], factor, minimum=2)
    target["ny"] = _scaled_count(target["ny"], factor, minimum=2)
    prefix = "grid.optical" if target is optical else "grid"
    return {
        f"{prefix}.nx": int(target["nx"]),
        f"{prefix}.ny": int(target["ny"]),
    }


def _independent_pump_grid_applicable(config: dict[str, Any]) -> bool:
    return str(config["grid"].get("pump_optical", {}).get("mode", "shared")) == "custom"


def _refine_pump_optical_window(
    config: dict[str, Any], factor: float
) -> dict[str, Any]:
    pump = config["grid"]["pump_optical"]
    physical = config["grid"]
    original_x = float(pump["size_x_m"])
    original_y = float(pump["size_y_m"])
    host_x = float(physical["size_x_m"])
    host_y = float(physical["size_y_m"])
    offset_x = abs(
        float(pump.get("center_x_m", 0.0))
        - float(physical.get("center_x_m", 0.0))
    )
    offset_y = abs(
        float(pump.get("center_y_m", 0.0))
        - float(physical.get("center_y_m", 0.0))
    )
    maximum_x = max(host_x - 2.0 * offset_x, 0.0)
    maximum_y = max(host_y - 2.0 * offset_y, 0.0)
    pump["size_x_m"] = min(original_x * factor, maximum_x)
    pump["size_y_m"] = min(original_y * factor, maximum_y)
    pump["nx"] = _scaled_count(
        pump["nx"], pump["size_x_m"] / original_x, minimum=2
    )
    pump["ny"] = _scaled_count(
        pump["ny"], pump["size_y_m"] / original_y, minimum=2
    )
    return {
        "grid.pump_optical.size_x_m": float(pump["size_x_m"]),
        "grid.pump_optical.size_y_m": float(pump["size_y_m"]),
        "grid.pump_optical.nx": int(pump["nx"]),
        "grid.pump_optical.ny": int(pump["ny"]),
        "refinement_contract": "independent_pump_window_at_preserved_dx_dy",
        "window_capped_by_physical_domain": bool(
            original_x * factor > maximum_x
            or original_y * factor > maximum_y
        ),
    }


def _refine_pump_optical_sampling(
    config: dict[str, Any], factor: float
) -> dict[str, Any]:
    pump = config["grid"]["pump_optical"]
    pump["nx"] = _scaled_count(pump["nx"], factor, minimum=2)
    pump["ny"] = _scaled_count(pump["ny"], factor, minimum=2)
    return {
        "grid.pump_optical.nx": int(pump["nx"]),
        "grid.pump_optical.ny": int(pump["ny"]),
    }


def _refine_temporal_sampling(
    config: dict[str, Any], factor: float
) -> dict[str, Any]:
    seed = config["seed"]
    updated: dict[str, Any] = {}
    for section, key, minimum in (
        (seed["temporal"], "sample_count", 2),
        (seed["pulse_definition"], "fft_sample_count", 8),
        (seed["coherent"], "sample_count", 8),
    ):
        section[key] = _scaled_count(section[key], factor, minimum=minimum)
    pulse_definition = seed["pulse_definition"]
    time_samples = pulse_definition.get(
        "solver_time_sample_count",
        pulse_definition.get("solver_sample_count", 1),
    )
    pulse_definition["solver_time_sample_count"] = _scaled_count(
        time_samples, factor, minimum=1
    )
    updated["seed.temporal.sample_count"] = int(seed["temporal"]["sample_count"])
    updated["seed.pulse_definition.fft_sample_count"] = int(
        seed["pulse_definition"]["fft_sample_count"]
    )
    updated["seed.coherent.sample_count"] = int(seed["coherent"]["sample_count"])
    updated["seed.pulse_definition.solver_time_sample_count"] = int(
        pulse_definition["solver_time_sample_count"]
    )
    return updated


def _refine_spectral_sampling(
    config: dict[str, Any], factor: float
) -> dict[str, Any]:
    seed = config["seed"]
    seed["spectrum"]["sample_count"] = _scaled_count(
        seed["spectrum"]["sample_count"], factor, minimum=1
    )
    pulse_definition = seed["pulse_definition"]
    spectral_samples = pulse_definition.get(
        "solver_spectral_sample_count",
        pulse_definition.get("solver_sample_count", 1),
    )
    pulse_definition["solver_spectral_sample_count"] = _scaled_count(
        spectral_samples, factor, minimum=1
    )
    return {
        "seed.spectrum.sample_count": int(seed["spectrum"]["sample_count"]),
        "seed.pulse_definition.solver_spectral_sample_count": int(
            pulse_definition["solver_spectral_sample_count"]
        ),
    }


def _coherent_slices_applicable(config: dict[str, Any]) -> bool:
    return (
        resolve_signal_model(config) == "coherent_cpa"
        and bool(config["seed"]["coherent"]["transient_gain"].get("enabled", False))
    )


def _refine_coherent_slices(
    config: dict[str, Any], factor: float
) -> dict[str, Any]:
    transient = config["seed"]["coherent"]["transient_gain"]
    transient["slice_count"] = _scaled_count(
        transient["slice_count"], factor, minimum=2
    )
    transient["maximum_slice_count"] = max(
        _scaled_count(transient["maximum_slice_count"], factor, minimum=2),
        int(transient["slice_count"]),
    )
    return {
        "seed.coherent.transient_gain.slice_count": int(transient["slice_count"]),
        "seed.coherent.transient_gain.maximum_slice_count": int(
            transient["maximum_slice_count"]
        ),
    }


def _raytrace_applicable(config: dict[str, Any]) -> bool:
    return resolve_physics(config).get("pump_propagation") == "stochastic_raytrace"


def _refine_ray_count(config: dict[str, Any], factor: float) -> dict[str, Any]:
    raytrace = config["pump"]["raytrace"]
    raytrace["ray_count"] = _scaled_count(raytrace["ray_count"], factor, minimum=100)
    return {"pump.raytrace.ray_count": int(raytrace["ray_count"])}


def _vary_ray_seed(config: dict[str, Any], factor: float) -> dict[str, Any]:
    raytrace = config["pump"]["raytrace"]
    base = int(raytrace["random_seed"])
    offset = max(int(round((factor - 1.0) * 104729.0)), 0)
    raytrace["random_seed"] = base + offset
    return {"pump.raytrace.random_seed": int(raytrace["random_seed"])}


def _fem_applicable(config: dict[str, Any]) -> bool:
    return (
        str(config["material"]["mechanical"].get("stress_model", "none"))
        in {"thermoelastic_fem", "anisotropic_thermoelastic_fem"}
    )


def _refine_fem_mesh(config: dict[str, Any], factor: float) -> dict[str, Any]:
    updated: dict[str, Any] = {}
    for key in ("nx", "ny", "nz"):
        updated.update(_physical_axis(key)(config, factor))
    fem = config["material"]["mechanical"]["thermoelastic_fem"]
    required_budget = int(config["grid"]["nx"]) * int(config["grid"]["ny"]) * int(
        config["grid"]["nz"]
    )
    fem["maximum_active_elements"] = max(
        int(fem["maximum_active_elements"]), required_budget
    )
    updated["material.mechanical.thermoelastic_fem.maximum_active_elements"] = int(
        fem["maximum_active_elements"]
    )
    return updated


def _ase_spectral_applicable(config: dict[str, Any]) -> bool:
    return (
        str(config.get("ase", {}).get("model", "off"))
        in {"six_ray", "angular_discrete_ordinates"}
        and len(config["ase"].get("wavelengths_m", [])) >= 2
    )


def _refine_ase_spectrum(config: dict[str, Any], factor: float) -> dict[str, Any]:
    ase = config["ase"]
    wavelengths = np.asarray(ase["wavelengths_m"], dtype=float)
    old_weights = np.asarray(ase.get("spectral_weights", []), dtype=float)
    if old_weights.size == 0:
        old_weights = np.ones_like(wavelengths)
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    old_weights = old_weights[order]
    count = _scaled_count(len(wavelengths), factor, minimum=2)
    if count == len(wavelengths):
        refined = wavelengths.copy()
        weights = old_weights / float(np.sum(old_weights))
    else:
        old_edges = np.empty(len(wavelengths) + 1)
        old_edges[1:-1] = 0.5 * (wavelengths[:-1] + wavelengths[1:])
        old_edges[0] = wavelengths[0] - 0.5 * (
            wavelengths[1] - wavelengths[0]
        )
        old_edges[-1] = wavelengths[-1] + 0.5 * (
            wavelengths[-1] - wavelengths[-2]
        )
        new_edges = np.linspace(old_edges[0], old_edges[-1], count + 1)
        refined = 0.5 * (new_edges[:-1] + new_edges[1:])
        normalized = old_weights / float(np.sum(old_weights))
        weights = np.zeros(count, dtype=float)
        for source in range(len(wavelengths)):
            source_width = float(old_edges[source + 1] - old_edges[source])
            first = max(
                0,
                int(np.searchsorted(new_edges, old_edges[source], side="right") - 1),
            )
            target = first
            while target < count and new_edges[target] < old_edges[source + 1]:
                overlap = max(
                    0.0,
                    min(float(old_edges[source + 1]), float(new_edges[target + 1]))
                    - max(float(old_edges[source]), float(new_edges[target])),
                )
                weights[target] += normalized[source] * overlap / source_width
                target += 1
        weights /= float(np.sum(weights))
    ase["wavelengths_m"] = refined.tolist()
    ase["spectral_weights"] = weights.tolist()
    return {"ase.spectral_sample_count": count}


def _ase_angular_applicable(config: dict[str, Any]) -> bool:
    return str(config.get("ase", {}).get("model", "off")) == (
        "angular_discrete_ordinates"
    )


def _refine_ase_directions(
    config: dict[str, Any], factor: float
) -> dict[str, Any]:
    angular = config["ase"]["angular"]
    count = _scaled_count(angular["direction_count"], factor, minimum=8)
    if count % 2:
        count += 1
    angular["direction_count"] = count
    return {"ase.angular.direction_count": count}


CONVERGENCE_AXES: dict[str, ConvergenceAxis] = {
    "physical_nx": ConvergenceAxis(
        "physical_nx", "Physical grid Nx", lambda config: True, _physical_axis("nx")
    ),
    "physical_ny": ConvergenceAxis(
        "physical_ny", "Physical grid Ny", lambda config: True, _physical_axis("ny")
    ),
    "physical_nz": ConvergenceAxis(
        "physical_nz", "Physical grid Nz", lambda config: True, _physical_axis("nz")
    ),
    "optical_window": ConvergenceAxis(
        "optical_window",
        "Seed FFT transverse window",
        _optical_window_applicable,
        _refine_optical_window,
    ),
    "optical_sampling": ConvergenceAxis(
        "optical_sampling",
        "Seed FFT transverse sampling",
        _optical_window_applicable,
        _refine_optical_sampling,
    ),
    "pump_optical_window": ConvergenceAxis(
        "pump_optical_window",
        "Independent Pump FFT transverse window",
        _independent_pump_grid_applicable,
        _refine_pump_optical_window,
    ),
    "pump_optical_sampling": ConvergenceAxis(
        "pump_optical_sampling",
        "Independent Pump FFT transverse sampling",
        _independent_pump_grid_applicable,
        _refine_pump_optical_sampling,
    ),
    "temporal_sampling": ConvergenceAxis(
        "temporal_sampling",
        "Seed temporal/Fourier sampling",
        lambda config: bool(config["seed"]["temporal"].get("resolved", False))
        or str(config["simulation"].get("signal_model", "auto"))
        in {"spectral_time", "josa_transport", "coherent_cpa"},
        _refine_temporal_sampling,
    ),
    "spectral_sampling": ConvergenceAxis(
        "spectral_sampling",
        "Seed spectral sampling",
        lambda config: str(config["seed"]["spectrum"].get("type", "monochromatic"))
        != "monochromatic"
        or str(config["simulation"].get("signal_model", "auto"))
        in {"spectral_time", "josa_transport", "coherent_cpa"},
        _refine_spectral_sampling,
    ),
    "coherent_slices": ConvergenceAxis(
        "coherent_slices",
        "Coherent transient-gain soft window",
        _coherent_slices_applicable,
        _refine_coherent_slices,
    ),
    "pump_ray_count": ConvergenceAxis(
        "pump_ray_count",
        "Stochastic Pump ray count",
        _raytrace_applicable,
        _refine_ray_count,
    ),
    "pump_ray_seed": ConvergenceAxis(
        "pump_ray_seed",
        "Stochastic Pump random-seed sensitivity (deprecated axis name)",
        _raytrace_applicable,
        _vary_ray_seed,
        interpretation="sensitivity",
    ),
    "pump_ray_seed_sensitivity": ConvergenceAxis(
        "pump_ray_seed_sensitivity",
        "Stochastic Pump random-seed sensitivity",
        _raytrace_applicable,
        _vary_ray_seed,
        interpretation="sensitivity",
    ),
    "fem_mesh": ConvergenceAxis(
        "fem_mesh",
        "Thermoelastic FEM voxel mesh",
        _fem_applicable,
        _refine_fem_mesh,
    ),
    "ase_spectral_sampling": ConvergenceAxis(
        "ase_spectral_sampling",
        "ASE spectral sampling",
        _ase_spectral_applicable,
        _refine_ase_spectrum,
    ),
    "ase_angular_directions": ConvergenceAxis(
        "ase_angular_directions",
        "ASE angular directions",
        _ase_angular_applicable,
        _refine_ase_directions,
    ),
}

DEFAULT_CONVERGENCE_AXES: tuple[str, ...] = tuple(
    name for name in CONVERGENCE_AXES if name != "pump_ray_seed"
)


def _relative_change(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), np.finfo(float).tiny)


def _sampling_statistics(values: np.ndarray) -> dict[str, Any]:
    samples = np.asarray(values, dtype=float)
    if samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("stochastic statistics require finite samples")
    mean = float(np.mean(samples))
    standard_deviation = (
        float(np.std(samples, ddof=1)) if samples.size > 1 else None
    )
    standard_error = None if standard_deviation is None else standard_deviation / math.sqrt(float(samples.size))
    interval = None
    method = "unavailable_insufficient_samples"
    if standard_error is not None:
        try:
            from scipy.stats import t
        except ImportError:
            method = "unavailable_scipy_required"
        else:
            half_width = float(t.ppf(0.975, samples.size - 1)) * standard_error
            interval = [mean - half_width, mean + half_width]
            method = "student_t_independent_approximately_normal_replicates"
    return {
        "sample_count": int(samples.size),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "confidence_level": 0.95,
        "confidence_interval_95": interval,
        "confidence_interval_method": method,
    }


def _config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolved_discretization(
    result: Any, config: dict[str, Any]
) -> dict[str, Any]:
    """Return the mesh/sampling state actually reported by the solver."""

    summary = getattr(result, "summary", {}) or {}
    signal = summary.get("signal", {}) or {}
    pump = summary.get("pump", {}) or {}
    ase = summary.get("ase", {}) or {}
    optical = config["grid"].get("optical", {})
    optical_target = optical if optical.get("enabled", False) else config["grid"]
    pump_mode = str(config["grid"].get("pump_optical", {}).get("mode", "shared"))
    pump_target = (
        optical_target
        if pump_mode == "shared"
        else config["grid"]
        if pump_mode == "physical"
        else config["grid"]["pump_optical"]
    )
    shape = summary.get("grid_shape_nz_ny_nx")
    if shape is None:
        shape = [
            int(config["grid"]["nz"]),
            int(config["grid"]["ny"]),
            int(config["grid"]["nx"]),
        ]
    return {
        "physical_shape_nz_ny_nx": list(shape),
        "optical_shape_ny_nx": [
            int(optical_target["ny"]),
            int(optical_target["nx"]),
        ],
        "optical_window_size_yx_m": [
            float(optical_target["size_y_m"]),
            float(optical_target["size_x_m"]),
        ],
        "pump_optical_mode": pump_mode,
        "pump_optical_shape_ny_nx": [
            int(pump_target["ny"]),
            int(pump_target["nx"]),
        ],
        "pump_optical_window_size_yx_m": [
            float(pump_target["size_y_m"]),
            float(pump_target["size_x_m"]),
        ],
        "temporal_samples": signal.get("temporal_samples"),
        "spectral_samples": signal.get("spectral_samples"),
        "coherent_transient_requested_slice_count": config["seed"]
        .get("coherent", {})
        .get("transient_gain", {})
        .get("slice_count"),
        "coherent_transient_largest_slice_count_used": (
            (signal.get("cpa") or {})
            .get("transient_gain", {})
            .get("largest_slice_count_used")
        ),
        "pump_ray_count": pump.get("ray_count"),
        "pump_ray_random_seed": config["pump"].get("raytrace", {}).get(
            "random_seed"
        ),
        "pump_maximum_used_substeps_per_z_cell": pump.get(
            "maximum_used_substeps_per_z_cell"
        ),
        "ase_direction_count": ase.get("direction_count"),
        "ase_spectral_samples": (
            None
            if ase.get("wavelengths_m") is None
            else len(ase.get("wavelengths_m", []))
        ),
    }


def run_convergence_suite(
    base_config: dict[str, Any],
    *,
    axes: Iterable[str] | None = None,
    factors: Iterable[float] = (1.0, 1.5, 2.0),
    metric_keys: Iterable[str] = DEFAULT_CONVERGENCE_METRICS,
    relative_tolerance: float = 0.01,
    simulator_factory: Callable[[dict[str, Any]], Any] | None = None,
    metric_extractor: Callable[[Any], dict[str, float]] | None = None,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Refine one numerical axis at a time and compare reported observables.

    The base configuration is never mutated.  A failed or strictly-invalid run
    remains in the report and cannot be labelled converged.
    """

    selected_axes = tuple(axes or DEFAULT_CONVERGENCE_AXES)
    unknown_axes = set(selected_axes) - set(CONVERGENCE_AXES)
    if unknown_axes:
        raise ValueError(f"Unknown convergence axes: {sorted(unknown_axes)}")
    selected_metrics = tuple(metric_keys)
    if not selected_metrics:
        raise ValueError("At least one convergence metric is required")
    levels = tuple(float(value) for value in factors)
    if len(levels) < 2 or any(not np.isfinite(value) or value <= 0.0 for value in levels):
        raise ValueError("Convergence factors need at least two positive finite values")
    if any(levels[index] >= levels[index + 1] for index in range(len(levels) - 1)):
        raise ValueError("Convergence factors must be strictly increasing")
    if relative_tolerance <= 0.0 or not np.isfinite(relative_tolerance):
        raise ValueError("relative_tolerance must be positive and finite")

    if simulator_factory is None or metric_extractor is None:
        from .solver import AmplifierSimulator
        from .sweep import extract_standard_metrics

        simulator_factory = simulator_factory or AmplifierSimulator
        metric_extractor = metric_extractor or extract_standard_metrics

    applicable = [
        CONVERGENCE_AXES[name]
        for name in selected_axes
        if CONVERGENCE_AXES[name].applicability(base_config)
    ]
    total = len(applicable) * len(levels)
    completed = 0
    cache: dict[str, dict[str, Any]] = {}
    axis_reports: list[dict[str, Any]] = []

    def evaluate(local: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        digest = _config_digest(local)
        if digest not in cache:
            try:
                result = simulator_factory(local)
                result = result.run() if hasattr(result, "run") else result
                metrics = metric_extractor(result)
                missing = set(selected_metrics) - set(metrics)
                if missing:
                    raise KeyError(f"Metric extractor omitted {sorted(missing)}")
                selected_values = {
                    key: float(metrics[key]) for key in selected_metrics
                }
                validity = standard_metric_validity(result, selected_values)
                nonfinite = [key for key in selected_values
                             if not bool(validity[key]["valid"])]
                if nonfinite:
                    raise ValueError(
                        f"Convergence metrics are non-finite or invalid: {sorted(nonfinite)}"
                    )
                validation = getattr(result, "summary", {}).get(
                    "validation", {"valid": True, "failures": []}
                )
                signal = getattr(result, "summary", {}).get("signal", {}) or {}
                cache[digest] = {
                    "executed_signal_model": str(signal.get("executed_model", signal.get("model", "unknown"))),
                    "metrics": selected_values,
                    "validation": validation,
                    "resolved_discretization": _resolved_discretization(
                        result, local
                    ),
                }
            except Exception as exc:  # each refinement remains inspectable
                cache[digest] = {"error": f"{type(exc).__name__}: {exc}"}
        return digest, cache[digest]

    for axis in (CONVERGENCE_AXES[name] for name in selected_axes):
        if not axis.applicability(base_config):
            axis_reports.append(
                {
                    "axis": axis.name,
                    "label": axis.label,
                    "interpretation": axis.interpretation,
                    "status": "not_applicable",
                    "records": [],
                }
            )
            continue
        records: list[dict[str, Any]] = []
        effective_states: dict[str, float] = {}
        for factor in levels:
            local = copy.deepcopy(base_config)
            settings = axis.refine(local, factor)
            digest = _config_digest(local)
            record: dict[str, Any] = {
                "factor": factor,
                "settings": settings,
                "config_sha256": digest,
                "status": "complete",
                "error": None,
            }
            replicate_configs = [local]
            if axis.name == "pump_ray_count":
                replicate_configs = []
                base_seed = int(local["pump"]["raytrace"]["random_seed"])
                for offset in (0, 104729, 209759):
                    replicate = copy.deepcopy(local)
                    replicate["pump"]["raytrace"]["random_seed"] = base_seed + offset
                    replicate_configs.append(replicate)
                record["stochastic_replicate_seeds"] = [
                    int(item["pump"]["raytrace"]["random_seed"])
                    for item in replicate_configs
                ]
            evaluated = [evaluate(item) for item in replicate_configs]
            failures = [cached["error"] for _, cached in evaluated if "error" in cached]
            if failures:
                record["status"] = "error"
                record["error"] = "; ".join(failures)
            else:
                payloads = [cached for _, cached in evaluated]
                record["executed_signal_models"] = sorted({payload["executed_signal_model"] for payload in payloads})
                metric_samples = {
                    key: np.asarray(
                        [payload["metrics"][key] for payload in payloads], dtype=float
                    )
                    for key in selected_metrics
                }
                record["metrics"] = {
                    key: float(np.mean(values))
                    for key, values in metric_samples.items()
                }
                if len(payloads) > 1:
                    record["stochastic_statistics"] = {
                        key: _sampling_statistics(values)
                        for key, values in metric_samples.items()
                    }
                record["validation"] = {
                    "valid": all(
                        bool(payload["validation"].get("valid", True))
                        for payload in payloads
                    ),
                    "replicates": [
                        copy.deepcopy(payload["validation"])
                        for payload in payloads
                    ],
                }
                record["resolved_discretization"] = copy.deepcopy(
                    payloads[0]["resolved_discretization"]
                )
                if len(payloads) > 1:
                    record["resolved_discretization"].update(
                        {
                            "pump_ray_random_seeds": record[
                                "stochastic_replicate_seeds"
                            ],
                            "stochastic_replicate_count": len(payloads),
                        }
                    )
                if not bool(record["validation"].get("valid", True)):
                    record["status"] = "invalid"
                else:
                    effective_digest = hashlib.sha256(
                        json.dumps(
                            record["resolved_discretization"],
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        ).encode("utf-8")
                    ).hexdigest()
                    if effective_digest in effective_states:
                        record["status"] = "no_effective_refinement"
                        record["duplicates_factor"] = effective_states[
                            effective_digest
                        ]
                    else:
                        effective_states[effective_digest] = factor
            records.append(record)
            completed += 1
            if progress is not None:
                progress(completed, total, copy.deepcopy(record))

        complete_records = [
            record for record in records if record["status"] == "complete"
        ]
        for index, record in enumerate(complete_records):
            if index == 0:
                record["relative_change_from_previous"] = None
            else:
                previous = complete_records[index - 1]["metrics"]
                record["relative_change_from_previous"] = {
                    key: _relative_change(record["metrics"][key], previous[key])
                    for key in selected_metrics
                }
        assessed = complete_records[-1] if len(complete_records) >= 3 else None
        latest_changes = (
            assessed.get("relative_change_from_previous") if assessed else None
        )
        assessed_changes = (
            [
                record["relative_change_from_previous"]
                for record in complete_records[-2:]
            ]
            if len(complete_records) >= 3
            else []
        )
        converged = (
            None
            if len(assessed_changes) < 2
            else all(
                value <= relative_tolerance
                for change in assessed_changes
                for value in change.values()
            )
        )
        status = "complete"
        if any(record["status"] == "error" for record in records):
            status = "error"
            converged = None
        elif any(record["status"] == "invalid" for record in records):
            status = "invalid"
            converged = None
        elif len({model for record in records for model in record.get("executed_signal_models", [])}) > 1:
            status = "model_changed"
            converged = None
        elif axis.interpretation == "sensitivity":
            status = (
                "sensitivity_complete"
                if len(complete_records) >= 3
                else "insufficient_valid_levels"
            )
            converged = None
        elif converged is False:
            status = "not_converged"
        elif converged is None:
            status = "insufficient_valid_levels"
        axis_reports.append(
            {
                "axis": axis.name,
                "label": axis.label,
                "interpretation": axis.interpretation,
                "status": status,
                "converged": converged,
                "relative_tolerance": relative_tolerance,
                "latest_relative_changes": latest_changes,
                "assessed_successive_relative_changes": assessed_changes,
                "effective_level_count": len(complete_records),
                "minimum_effective_level_count": 3,
                "statistical_summary": (
                    {
                        key: _sampling_statistics(
                            np.asarray(
                                [record["metrics"][key] for record in complete_records],
                                dtype=float,
                            )
                        )
                        for key in selected_metrics
                    }
                    if axis.interpretation == "sensitivity" and complete_records
                    else None
                ),
                "duplicate_level_count": sum(
                    record["status"] == "no_effective_refinement"
                    for record in records
                ),
                "records": records,
            }
        )

    assessed_axes = [
        axis for axis in axis_reports if axis["status"] != "not_applicable"
    ]
    all_converged = bool(assessed_axes) and all(
        (
            axis["status"] == "sensitivity_complete"
            if axis.get("interpretation") == "sensitivity"
            else axis.get("converged") is True and axis["status"] == "complete"
        )
        for axis in assessed_axes
    )
    return {
        "status": "converged" if all_converged else "review_required",
        "all_assessed_axes_converged": all_converged,
        "base_config_sha256": _config_digest(base_config),
        "factors": list(levels),
        "metrics": list(selected_metrics),
        "relative_tolerance": relative_tolerance,
        "axes": axis_reports,
        "limitations": [
            "Axes are refined independently; this is not a full Cartesian uncertainty study.",
            "The default contract compares selected scalar observables, not full-field norms; supply a custom metric_extractor for application-specific field-derived scalars.",
            "Six-ray ASE has a fixed angular quadrature; angular-order convergence is available only for angular_discrete_ordinates.",
            "FEM refinement uses the active physical voxel mesh; it does not create an external adaptive finite-element mesh.",
        ],
    }
