"""Top-level orchestration of independently replaceable amplifier submodels."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .ase import ASEResult, ASETransport
from .commercial_app_fast import (
    CommercialAppFastAveragePowerSignalPropagator,
    CommercialAppFastSignalPropagator,
)
from .average_signal import AveragePowerSignalPropagator
from .spectral_volume import concatenate_spectral_intensities
from .beams import make_beam
from .compute_backend import compute_backend_info, compute_backend_warnings, configure_compute_backend
from .constants import LIGHT_SPEED
from .config import resolve_physics, resolve_signal_model, validate_config
from .field_diagnostics import parabolic_field_diagnostics
from .diagnostics import transition_inversion_m3
from .doping import dopant_density_field
from .dispersion import material_refractive_index
from .dispersion import material_group_index
from .energy_ledger import build_energy_ledger
from .event_energy import EventEnergyLedger, ase_source_interface_diagnostics
from .errors import OPTIONAL_BACKEND_FAILURES, OPTIONAL_MODEL_FAILURES
from .fixed_point import SafeguardedAnderson
from .grid import (
    Grid3D,
    optical_grid_from_config,
    pump_optical_grid_from_config,
)
from .grid_coupling import (
    CoupledAveragePowerSignalPropagator,
    CoupledPumpPropagator,
    CoupledSignalPropagator,
    GridCoupler,
)
from .geometry import crystal_geometry_summary, crystal_volume_mask
from .levels import LevelSystem
from .josa_modes import JOSA_FULL_3D_MODE
from .multipass import (
    compile_multipass_plan,
    sequential_plan_components,
    uses_causal_clock,
)
from .optical_system import gain_medium_path_cosine
from .optics import (
    OpticalTrace,
    split_gain_path,
    transverse_boundary_energy_fractions,
    transverse_intensity_moments,
)
from .optional_signal import ResilientSignalPropagator, optional_fallback_origin
from .physics_validity import pulse_relaxation_validity
from .profiling import (
    configure_operation_profiler,
    operation_profile_snapshot,
    reset_operation_profiler,
)
from .profiles import PumpWaveform, periodic_pump_waveform
from .result_recording import resolve_result_recording
from .flight_time import interpass_dark_time_s, optical_train_flight_time
from .pump import PumpPropagationResult, PumpPropagator, pump_metric_spec
from .pump_heat import (
    SPECTRAL_HEAT_MODELS,
    lifetime_nonradiative_heat,
    pump_component_ledger_applicability,
    spectral_active_heat_fraction,
)
from .signal import SignalPropagationResult, SignalPropagator, _log_expm1, frantz_nodvik
from .spectral_signal import IncoherentHGSignalPropagator, SpectralSignalPropagator
from .spectral_reporting import (
    spectral_bin_coordinate_for_result,
    spectral_density_from_bin_energy,
)
from .run_context import RunContext
from .spatial_modes import spatial_mode_summary
from .stress import make_stress_solver
from .strict_validation import (
    build_required_diagnostic_contract,
    build_validation_report,
)
from .thermal import ThermalSolver


def _has_nonzero_phase_variation(samples: Any, *, real_imag: bool = False) -> bool:
    """Return whether tabulated complex data carries more than a global phase."""

    try:
        rows = np.asarray(samples, dtype=float)
    except (TypeError, ValueError):
        return True
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] < 3:
        return False
    if real_imag:
        amplitude = np.hypot(rows[:, 1], rows[:, 2])
        active = amplitude > np.finfo(float).tiny
        if np.count_nonzero(active) < 2:
            return False
        phase = np.unwrap(np.angle(rows[active, 1] + 1j * rows[active, 2]))
    else:
        phase = np.unwrap(rows[:, 2])
    if not np.all(np.isfinite(phase)):
        return True
    scale = max(float(np.max(np.abs(phase))), 1.0)
    return bool(float(np.ptp(phase)) > 64.0 * np.finfo(float).eps * scale)


def _runtime_optional_fallback(result: SignalPropagationResult | None) -> bool:
    """Return whether a requested Seed backend failed during this call."""

    return bool(
        result is not None
        and optional_fallback_origin(result) == "runtime"
    )


def _phase_insensitive_dispersion_sources(config: dict[str, Any]) -> list[str]:
    """List configured phase terms that require coherent temporal reassembly."""

    sources: list[str] = []
    seed = config.get("seed", {})
    coherent = seed.get("coherent", {})
    high_order_keys = ("gdd_s2", "tod_s3", "fod_s4")
    for section_name in ("phase", "stretcher", "compressor"):
        section = coherent.get(section_name, {})
        if any(abs(float(section.get(key, 0.0))) > 0.0 for key in high_order_keys):
            sources.append(f"seed.coherent.{section_name}")
    spectral_phase = coherent.get("spectral_phase", {})
    if bool(spectral_phase.get("enabled", False)):
        sources.append("seed.coherent.spectral_phase")
    if abs(float(seed.get("chirp_s2", 0.0))) > 0.0:
        sources.append("seed.chirp_s2")
    for section_name in ("temporal", "spectrum"):
        section = seed.get(section_name, {})
        if (
            str(section.get("type", "")) == "file"
            and _has_nonzero_phase_variation(section.get("samples", []))
        ):
            sources.append(f"seed.{section_name}.samples.phase")
    input_mode = str(coherent.get("input_mode", "coefficients"))
    if input_mode == "phase_retrieval":
        sources.append("seed.coherent.phase_retrieval")
    elif input_mode == "complex_spectrum":
        complex_cfg = coherent.get("complex_spectrum", {})
        if _has_nonzero_phase_variation(
            complex_cfg.get("samples", []),
            real_imag=str(complex_cfg.get("format", "amplitude_phase"))
            == "real_imag",
        ):
            sources.append("seed.coherent.complex_spectrum")
    material_dispersion = config.get("material", {}).get("dispersion", {})
    if any(
        abs(float(material_dispersion.get(f"{key}_per_m", 0.0))) > 0.0
        for key in high_order_keys
    ):
        sources.append("material.dispersion")
    for train_name, elements in (
        ("optical_train.seed", config.get("optical_train", {}).get("seed", [])),
        (
            "optical_train.between_pass",
            config.get("optical_train", {}).get("between_pass", []),
        ),
    ):
        for index, element in enumerate(elements or []):
            dispersion = (
                element.get("dispersion", {})
                if isinstance(element, dict)
                else {}
            )
            if any(
                abs(float(dispersion.get(key, 0.0))) > 0.0
                for key in high_order_keys
            ):
                sources.append(f"{train_name}[{index}].dispersion")
    return list(dict.fromkeys(sources))


def _output_beam_quality_validity(
    *,
    sampled_valid: bool,
    forced_model_valid: bool,
    model_used: str,
    cpa_diagnostics: dict[str, Any] | None,
    strict_config: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Gate a reported sampled M2 by the numerical data that produced it."""

    valid = bool(forced_model_valid or sampled_valid)
    reasons: list[str] = []
    if not valid:
        reasons.append("output_plane_spatial_second_moment_invalid")
    if str(model_used) == "spectral_time_causal_multipass":
        reasons.append("output_incoherent_channels_not_retained")
        return False, reasons
    if str(model_used) != "coherent_cpa":
        return valid, reasons

    cpa = cpa_diagnostics or {}
    sampling = cpa.get("sampling") or {}
    if sampling and not bool(sampling.get("adequate", False)):
        valid = False
        reasons.append("coherent_input_sampling_contract_failed")
    spectral_boundary = cpa.get(
        "maximum_output_spectral_boundary_energy_fraction"
    )
    spectral_limit = min(
        float(cpa.get("spectral_boundary_warning_fraction", 1.0)),
        float(
            strict_config.get(
                "output_spectral_boundary_energy_fraction",
                1e-6,
            )
        ),
    )
    if (
        spectral_boundary is not None
        and float(spectral_boundary) > spectral_limit
    ):
        valid = False
        reasons.append("coherent_output_spectral_fft_boundary_failed")
    if str(cpa.get("fidelity_mode", "")) in {
        "axisymmetric_kerr",
        "axisymmetric_transient_gain",
    }:
        axisymmetric = cpa.get("axisymmetric_reduction") or {}
        projection_fraction = axisymmetric.get(
            "total_input_projection_energy_fraction"
        )
        projection_loss_limit = float(
            strict_config.get(
                "maximum_axisymmetric_projection_loss_fraction",
                1e-3,
            )
        )
        if (
            projection_fraction is None
            or 1.0 - float(projection_fraction) > projection_loss_limit
        ):
            valid = False
            reasons.append("axisymmetric_input_projection_failed")
        dynamic_omitted_fraction = float(
            axisymmetric.get(
                "maximum_dynamic_field_omitted_energy_fraction",
                0.0,
            )
        )
        if dynamic_omitted_fraction > projection_loss_limit:
            valid = False
            reasons.append(
                "axisymmetric_dynamic_azimuthal_closure_failed"
            )
        boundary = axisymmetric.get(
            "maximum_energy_weighted_outer_annulus_fraction",
            axisymmetric.get("maximum_outer_annulus_energy_fraction"),
        )
        boundary_limit = float(
            strict_config.get(
                "maximum_axisymmetric_boundary_energy_fraction",
                1e-4,
            )
        )
        if boundary is None or float(boundary) > boundary_limit:
            valid = False
            reasons.append("axisymmetric_radial_boundary_failed")
        asymmetry = max(
            float(
                axisymmetric.get(
                    "maximum_population_azimuthal_residual",
                    float("inf"),
                )
            ),
            float(
                axisymmetric.get(
                    "maximum_delta_n_azimuthal_residual",
                    float("inf"),
                )
            ),
            float(
                axisymmetric.get(
                    "maximum_free_carrier_azimuthal_residual",
                    0.0,
                )
            ),
            float(
                axisymmetric.get(
                    "maximum_cheap_m2_material_fit_residual",
                    0.0,
                )
            ),
        )
        asymmetry_limit = float(
            strict_config.get("maximum_axisymmetric_azimuthal_residual", 5e-2)
        )
        if asymmetry > asymmetry_limit:
            valid = False
            reasons.append("axisymmetric_material_symmetry_failed")
        realizability_correction = float(
            axisymmetric.get(
                "maximum_population_realizability_correction_fraction",
                0.0,
            )
        )
        realizability_correction_limit = float(
            strict_config.get(
                "maximum_axisymmetric_population_realizability_correction_fraction",
                5e-2,
            )
        )
        if realizability_correction > realizability_correction_limit:
            valid = False
            reasons.append(
                "axisymmetric_population_realizability_failed"
            )
        positivity_redistribution = float(
            axisymmetric.get(
                "maximum_population_positivity_redistribution_fraction",
                0.0,
            )
        )
        positivity_redistribution_limit = float(
            strict_config.get(
                "maximum_axisymmetric_population_positivity_redistribution_fraction",
                5e-2,
            )
        )
        if positivity_redistribution > positivity_redistribution_limit:
            valid = False
            reasons.append(
                "axisymmetric_population_positivity_redistribution_failed"
            )
        norm_correction = axisymmetric.get(
            "maximum_diffraction_norm_correction"
        )
        norm_limit = float(
            strict_config.get(
                "maximum_axisymmetric_diffraction_norm_correction",
                1e-3,
            )
        )
        if norm_correction is None or float(norm_correction) > norm_limit:
            valid = False
            reasons.append("axisymmetric_diffraction_norm_failed")
        gain_span = axisymmetric.get(
            "maximum_gain_bin_relative_frequency_span"
        )
        gain_span_limit = float(
            strict_config.get(
                "maximum_axisymmetric_gain_bin_relative_frequency_span",
                5e-2,
            )
        )
        if gain_span is None or float(gain_span) > gain_span_limit:
            valid = False
            reasons.append("axisymmetric_gain_spectral_resolution_failed")
        gain_correction = axisymmetric.get(
            "maximum_gain_reconstruction_normalization_correction"
        )
        gain_correction_limit = float(
            strict_config.get(
                "maximum_axisymmetric_gain_reconstruction_normalization_correction",
                2e-2,
            )
        )
        if (
            gain_correction is None
            or float(gain_correction) > gain_correction_limit
        ):
            valid = False
            reasons.append("axisymmetric_gain_reconstruction_failed")
        mapping_error = axisymmetric.get("population_mapping_relative_error")
        mapping_limit = float(
            strict_config.get(
                "axisymmetric_population_mapping_relative_tolerance",
                1e-10,
            )
        )
        if mapping_error is None or float(mapping_error) > mapping_limit:
            valid = False
            reasons.append("axisymmetric_population_mapping_failed")
    return valid, list(dict.fromkeys(reasons))


@dataclass
class SimulationContinuationState:
    """Numerical state that can initialize a nearby parameter point.

    The state is deliberately limited to fields that are fixed-point unknowns
    or linear-solver guesses.  It never substitutes a previous result for the
    new solve; the receiving simulator still closes every configured equation.
    """

    periodic_boundary_population_m3: np.ndarray | None = None
    thermal_delta_n: np.ndarray | None = None
    response_temperature_K: np.ndarray | None = None
    thermal_temperature_K: np.ndarray | None = None
    front_surface_sag_m: np.ndarray | None = None
    back_surface_sag_m: np.ndarray | None = None


@dataclass
class SimulationResult:
    grid: Grid3D
    population_m3: np.ndarray
    population_after_pump_m3: np.ndarray
    inversion_after_pump_m3: np.ndarray
    pump_intensity_W_m2: np.ndarray
    pump_absorbed_W_m3: np.ndarray
    signal_fluence_J_m2: np.ndarray
    gain_m_inv: np.ndarray
    initial_active_gain_m_inv: np.ndarray
    remaining_active_gain_m_inv: np.ndarray
    passive_loss_m_inv: np.ndarray
    net_gain_m_inv: np.ndarray
    heat_W_m3: np.ndarray
    extracted_energy_J_m3: np.ndarray
    temperature_K: np.ndarray
    delta_n: np.ndarray
    stress: dict[str, np.ndarray]
    optical_planes: dict[str, np.ndarray]
    level_names: list[str]
    summary: dict[str, Any]
    wavelengths_m: np.ndarray | None = None
    input_spectrum_J: np.ndarray | None = None
    output_spectrum_J: np.ndarray | None = None
    time_s: np.ndarray | None = None
    input_temporal_energy_J: np.ndarray | None = None
    output_temporal_energy_J: np.ndarray | None = None
    frequency_Hz: np.ndarray | None = None
    input_spectral_phase_rad: np.ndarray | None = None
    output_spectral_phase_rad: np.ndarray | None = None
    compressed_spectral_phase_rad: np.ndarray | None = None
    compressed_temporal_power_W: np.ndarray | None = None
    cpa_diagnostics: dict[str, Any] | None = None
    crystal_mask: np.ndarray | None = None
    optical_grid: Grid3D | None = None
    pump_optical_grid: Grid3D | None = None
    optical_plane_coordinates: dict[
        str, tuple[np.ndarray, np.ndarray]
    ] | None = None
    pump_active_absorbed_W_m3: np.ndarray | None = None
    pump_passive_absorbed_W_m3: np.ndarray | None = None
    signal_fluence_last_pass_J_m2: np.ndarray | None = None
    signal_fluence_cumulative_J_m2: np.ndarray | None = None
    signal_fluence_per_pass_J_m2: np.ndarray | None = None
    dopant_density_m3: np.ndarray | None = None
    nonlinear_loss_energy_J_m3: np.ndarray | None = None
    ase_intensity_W_m2: np.ndarray | None = None
    signal_passive_loss_energy_J_m3: np.ndarray | None = None
    output_jones_field: np.ndarray | None = None
    pump_wavelengths_m: np.ndarray | None = None
    pump_spectral_intensity_W_m2: np.ndarray | None = None
    pump_spectral_active_absorbed_W_m3: np.ndarray | None = None
    pump_spectral_passive_absorbed_W_m3: np.ndarray | None = None
    longitudinal_intensity_W_m2: np.ndarray | None = None
    longitudinal_intensity_area_m2: np.ndarray | None = None
    longitudinal_temporal_power_W: np.ndarray | None = None
    temporal_bin_edges_s: np.ndarray | None = None
    spectral_bin_edges_Hz: np.ndarray | None = None

    def save(self, output_dir: str | Path) -> None:
        from .output import save_result

        save_result(self, Path(output_dir))


class AmplifierSimulator:
    """Coordinate one mutable simulation run.

    Instances retain backend diagnostics, material environments, fixed-point
    history, and the most recent fields. They are intentionally not thread
    safe or re-entrant; parallel sweeps must create one instance per point.
    """

    thread_safe = False
    reentrant = False

    @property
    def _last_ase_result(self) -> ASEResult | None:
        return self.run_context.last_ase_result

    @_last_ase_result.setter
    def _last_ase_result(self, value: ASEResult | None) -> None:
        self.run_context.last_ase_result = value

    @property
    def _ase_warnings(self) -> list[str]:
        return self.run_context.ase_warnings

    @_ase_warnings.setter
    def _ase_warnings(self, value: list[str]) -> None:
        self.run_context.ase_warnings = value

    @property
    def _last_spectral_heat_stats(self) -> dict[str, Any]:
        return self.run_context.spectral_heat_stats

    @_last_spectral_heat_stats.setter
    def _last_spectral_heat_stats(self, value: dict[str, Any]) -> None:
        self.run_context.spectral_heat_stats = value

    @property
    def _last_event_energy_stats(self) -> dict[str, Any]:
        return self.run_context.event_energy_stats

    @_last_event_energy_stats.setter
    def _last_event_energy_stats(self, value: dict[str, Any]) -> None:
        self.run_context.event_energy_stats = value

    @property
    def _last_lifetime_heat_stats(self) -> dict[str, Any]:
        return self.run_context.lifetime_heat_stats

    @_last_lifetime_heat_stats.setter
    def _last_lifetime_heat_stats(self, value: dict[str, Any]) -> None:
        self.run_context.lifetime_heat_stats = value

    @property
    def _last_pump_component_applicability(self) -> dict[str, Any]:
        return self.run_context.pump_component_applicability

    @_last_pump_component_applicability.setter
    def _last_pump_component_applicability(
        self,
        value: dict[str, Any],
    ) -> None:
        self.run_context.pump_component_applicability = value

    @property
    def _last_periodic_boundary_population(self) -> np.ndarray | None:
        value = self.run_context.fixed_point_history.get("periodic_boundary_population")
        return None if value is None else np.asarray(value, dtype=float)

    @_last_periodic_boundary_population.setter
    def _last_periodic_boundary_population(
        self, value: np.ndarray | None
    ) -> None:
        if value is None:
            self.run_context.fixed_point_history.pop(
                "periodic_boundary_population", None
            )
        else:
            self.run_context.fixed_point_history[
                "periodic_boundary_population"
            ] = np.asarray(value, dtype=float).copy()

    def __init__(
        self,
        config: dict[str, Any],
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_mode: str = "detailed",
    ):
        # The simulator owns a configuration snapshot.  Validation performs a
        # small amount of canonicalization (notably for Pump source aliases),
        # so both that mutation and every later solver reference must be
        # isolated from the caller's still-mutable mapping.
        config = copy.deepcopy(config)
        validate_config(config)
        configure_compute_backend(config.get("performance", {}))
        configure_operation_profiler(config.get("performance", {}))
        self.config = config
        self._progress_callback = progress_callback
        self._progress_mode = str(progress_mode)
        if self._progress_mode not in {"detailed", "milestones"}:
            raise ValueError("progress_mode must be 'detailed' or 'milestones'")
        self._progress_ranges: list[tuple[float, float]] = [(0.0, 1.0)]
        self._progress_started_s = time.perf_counter()
        self._progress_sequence = 0
        self._last_progress_fraction = 0.0
        self.run_context = RunContext()
        self._continuation_state: SimulationContinuationState | None = None
        self._continuation_thermal_guess_consumed = False
        self._reduced_pulse_map_initializer_consumed = False
        self.config_sha256 = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()
        self.grid = Grid3D.from_config(config["grid"])
        self.optical_grid = optical_grid_from_config(self.grid, config["grid"])
        self.pump_optical_grid = pump_optical_grid_from_config(
            self.grid,
            config["grid"],
            self.optical_grid,
        )
        self.grid_coupler = (
            GridCoupler.build(self.grid, self.optical_grid)
            if self.optical_grid is not self.grid
            else None
        )
        self.pump_grid_coupler = (
            self.grid_coupler
            if self.pump_optical_grid is self.optical_grid
            else (
                GridCoupler.build(self.grid, self.pump_optical_grid)
                if self.pump_optical_grid is not self.grid
                else None
            )
        )
        self.material = config["material"]
        self.crystal_mask = crystal_volume_mask(self.grid, self.material)
        self._thermal_solver: ThermalSolver | None = None
        self._stress_solver: Any | None = None
        self.dopant_density_m3, self.doping_profile_summary = dopant_density_field(
            self.grid, self.material
        )
        self.pump_cfg = config["pump"]
        self.seed_cfg = config["seed"]
        self.sim_cfg = config["simulation"]
        self.result_recording = resolve_result_recording(self.sim_cfg)
        self.physics = resolve_physics(config)
        self._last_periodic_boundary_population: np.ndarray | None = None
        self._last_spectral_heat_stats: dict[str, Any] = {"enabled": False}
        self._last_event_energy_stats: dict[str, Any] = {"enabled": False}
        self._last_lifetime_heat_stats: dict[str, Any] = {"enabled": False}
        self._last_pump_component_applicability: dict[str, Any] = {
            "applicable": None,
            "reason": "not_evaluated",
        }
        # Preserve the original gain-only field for old configurations unless a
        # new optical choice was made explicitly.
        if config.get("physics", {}).get("optical_propagation", "auto") == "auto" and self.sim_cfg.get("signal_propagation") == "gain_only":
            self.physics["optical_propagation"] = "off"
        self.levels = LevelSystem.from_config(config["energy_levels"], self.material)
        self.levels.set_environment(
            float(self.material.get("operating_temperature_K", 293.15)),
            self.dopant_density_m3,
        )
        self.ase_solver = ASETransport(
            self.grid,
            self.levels,
            self.material,
            config.get("ase", {}),
            float(self.seed_cfg["wavelength_m"]),
            self.crystal_mask,
        )
        # ``None`` means ASE is absent from the coupled operating point.  A
        # synthetic disabled ASEResult used to be stored here, which left all
        # fields numerically zero but incorrectly reported ``ase_coupled=true``
        # and made downstream heat/diagnostic code treat ASE as active.
        self._last_ase_result: ASEResult | None = None
        self._ase_warnings: list[str] = []
        self._ase_execution: dict[str, Any] = {
            "requested": bool(self.ase_solver.enabled),
            "executed": False,
            "disabled_after_failure": False,
            "disabled_reason": None,
        }
        self.pump_beam = make_beam(self.pump_cfg)
        train = config.get("optical_train", {})
        pump_pre, pump_gain, _ = split_gain_path(train.get("pump", []))
        self.pump_gain_marker = pump_gain
        gain_markers: list[dict[str, Any]] = []
        between_segments: list[list[dict[str, Any]]] = []
        bench_warnings: list[str] = []
        seed_pre, seed_gain, seed_post = split_gain_path(
            train.get("seed", [])
        )
        resolved_signal_model = resolve_signal_model(config)
        multipass_cfg = self.sim_cfg.get("multipass", {})
        causal_multipass = uses_causal_clock(
            self.sim_cfg, self.seed_cfg, resolved_signal_model
        )
        if bool(multipass_cfg.get("enabled", False)) and not causal_multipass:
            plan = compile_multipass_plan(
                multipass_cfg,
                seed_gain,
                crystal_length_m=float(config["grid"]["length_z_m"]),
                crystal_refractive_index=material_refractive_index(
                    self.material,
                    float(self.seed_cfg["wavelength_m"]),
                    role="signal",
                ),
                # Sequential backends add the simulation-level delay when
                # evaluating each route. Do not also encode it in the route.
                global_extra_delay_s=0.0,
            )
            gain_markers, between_segments = sequential_plan_components(plan)
            bench_warnings = [*plan.warnings, *plan.assumptions]
        pump_solver = PumpPropagator(
            self.pump_optical_grid,
            self.material,
            self.pump_cfg,
            self.pump_beam,
            self.levels,
            self.physics,
            pump_pre,
            pump_gain,
        )
        self.pump_solver = (
            CoupledPumpPropagator(pump_solver, self.pump_grid_coupler)
            if self.pump_grid_coupler is not None
            else pump_solver
        )
        self.signal_model = resolved_signal_model
        signal_args = (
            self.optical_grid,
            self.material,
            self.seed_cfg,
            config["energy_levels"],
            self.levels,
            self.sim_cfg,
            self.physics,
            seed_pre,
            seed_gain,
            seed_post,
            list(train.get("between_pass", [])),
            gain_markers,
            between_segments,
            bench_warnings,
        )
        average_signal_class = (
            CommercialAppFastAveragePowerSignalPropagator
            if resolved_signal_model == "commercial_app_fast"
            else AveragePowerSignalPropagator
        )
        commercial_app_axisymmetric = (
            resolved_signal_model == "commercial_app_fast"
            and self.seed_cfg.get("commercial_app_fast", {}).get("spatial_backend", "cartesian") == "axisymmetric"
        )
        if commercial_app_axisymmetric:
            from .axisymmetric_commercial_app import AxisymmetricCommercialAppFastSignalPropagator
            average_signal_class = AxisymmetricCommercialAppFastSignalPropagator
        spectral_axisymmetric = (
            resolved_signal_model == "spectral_time"
            and self.seed_cfg.get("spectral_time", {}).get("spatial_backend", "cartesian") == "axisymmetric"
        )
        if spectral_axisymmetric:
            from .axisymmetric_spectral import AxisymmetricSpectralSignalPropagator
            if causal_multipass:
                raise ValueError("Spectral Time axisymmetric does not support causal multipass routing")
            average_signal_class = AxisymmetricSpectralSignalPropagator
        if (
            resolved_signal_model == "coherent_cpa"
            and self.seed_cfg.get("spatial_modes", {}).get("model") != "incoherent_hg"
            and self.seed_cfg.get("coherent", {}).get("fidelity_mode")
            in {"axisymmetric_kerr", "axisymmetric_transient_gain"}
        ):
            from .axisymmetric_average import AxisymmetricAveragePowerSignalPropagator

            average_signal_class = AxisymmetricAveragePowerSignalPropagator
        if (
            resolved_signal_model == "coherent_cpa"
            and self.seed_cfg.get("spatial_modes", {}).get("model") == "incoherent_hg"
            and self.seed_cfg.get("coherent", {}).get("fidelity_mode") == "axisymmetric_kerr"
        ):
            from .axisymmetric_hg_coherent import AxisymmetricHGAveragePowerSignalPropagator

            average_signal_class = AxisymmetricHGAveragePowerSignalPropagator
        average_signal_solver = average_signal_class(*signal_args)
        self.average_signal_solver = (
            CoupledAveragePowerSignalPropagator(
                average_signal_solver,
                self.grid_coupler,
                self.crystal_mask,
            )
            if self.grid_coupler is not None
            else average_signal_solver
        )
        self.reduced_pulse_map_signal_solver: Any | None = None
        self.reduced_pulse_map_settings: dict[str, Any] = {}
        if resolved_signal_model == "commercial_app_fast":
            signal_class = AxisymmetricCommercialAppFastSignalPropagator if commercial_app_axisymmetric else CommercialAppFastSignalPropagator
            self.signal_solver = signal_class(*signal_args)
        elif spectral_axisymmetric:
            self.signal_solver = AxisymmetricSpectralSignalPropagator(*signal_args)
        elif (
            resolved_signal_model != "coherent_cpa"
            and self.seed_cfg.get("spatial_modes", {}).get("model")
            == "incoherent_hg"
        ):
            self.signal_solver = IncoherentHGSignalPropagator(*signal_args)
        elif resolved_signal_model == "coherent_cpa":
            periodic_cfg = self.sim_cfg.get("periodic_steady_state", {})
            periodic_method = str(periodic_cfg.get("method", "pulse_map"))
            pulse_map_initialization = str(
                periodic_cfg.get("pulse_map_initialization", "auto")
            )
            if (
                periodic_method == "reduced_pulse_map"
                or pulse_map_initialization == "reduced_pulse_map"
            ):
                reduced_cfg = dict(periodic_cfg.get("reduced_pulse_map", {}))
                reduced_model = str(
                    reduced_cfg.get("model", "phase_insensitive")
                )
                temporal_bins = int(reduced_cfg.get("temporal_bins", 1))
                spectral_bins = int(reduced_cfg.get("spectral_bins", 33))
                proxy_seed_cfg = copy.deepcopy(self.seed_cfg)
                proxy_definition = proxy_seed_cfg.setdefault(
                    "pulse_definition", {}
                )
                # The periodic proxy intentionally transports one integrated
                # pulse packet with a controlled spectral quadrature. Derive
                # it from the coherent spectrum even when the user's legacy
                # phase-insensitive configuration declared independent time
                # and spectrum profiles.
                proxy_definition["domain"] = "spectrum"
                proxy_definition["solver_time_sample_count"] = temporal_bins
                proxy_definition["solver_spectral_sample_count"] = spectral_bins
                if reduced_model == "axisymmetric_coherent_kerr":
                    proxy_coherent = proxy_seed_cfg.setdefault("coherent", {})
                    coherent_sample_count = int(
                        reduced_cfg.get("coherent_sample_count", 1024)
                    )
                    proxy_coherent["sample_count"] = coherent_sample_count
                    proxy_axisymmetric = proxy_coherent.setdefault(
                        "axisymmetric", {}
                    )
                    proxy_axisymmetric["radial_bins"] = int(
                        reduced_cfg.get("axisymmetric_radial_bins", 24)
                    )
                    proxy_axisymmetric["gain_spectral_bins"] = int(
                        reduced_cfg.get(
                            "axisymmetric_gain_spectral_bins", 33
                        )
                    )
                proxy_seed_cfg.setdefault("joint_time_frequency", {}).update(
                    {"model": "separable", "weights": []}
                )
                proxy_seed_cfg.setdefault("caustic_diagnostics", {})[
                    "enabled"
                ] = False
                proxy_physics = copy.deepcopy(self.physics)
                proxy_physics["record_optical_planes"] = False
                proxy_physics["_suppress_output_diagnostics"] = True
                proxy_args = (
                    self.optical_grid,
                    self.material,
                    proxy_seed_cfg,
                    config["energy_levels"],
                    self.levels,
                    self.sim_cfg,
                    proxy_physics,
                    seed_pre,
                    seed_gain,
                    seed_post,
                    list(train.get("between_pass", [])),
                    gain_markers,
                    between_segments,
                    bench_warnings,
                )
                if reduced_model == "axisymmetric_coherent_kerr":
                    from .axisymmetric_coherent import (
                        AxisymmetricCoherentSignalPropagator,
                    )

                    self.reduced_pulse_map_signal_solver = (
                        AxisymmetricCoherentSignalPropagator(*proxy_args)
                    )
                    self.reduced_pulse_map_settings = {
                        "temporal_bins": coherent_sample_count,
                        "spectral_bins": coherent_sample_count,
                        "radial_bins": int(proxy_axisymmetric["radial_bins"]),
                        "azimuthal_fidelity": str(
                            proxy_axisymmetric.get(
                                "azimuthal_fidelity", "m0"
                            )
                        ),
                        "angular_samples": int(
                            proxy_axisymmetric.get("angular_samples", 12)
                        ),
                        "gain_spectral_bins": int(
                            proxy_axisymmetric["gain_spectral_bins"]
                        ),
                        "field_model": (
                            "axisymmetric_coherent_frequency_time_radius"
                            if str(
                                proxy_axisymmetric.get(
                                    "azimuthal_fidelity", "m0"
                                )
                            )
                            == "m0"
                            else "finite_azimuthal_coherent_frequency_time_radius"
                        ),
                    }
                else:
                    self.reduced_pulse_map_signal_solver = (
                        SpectralSignalPropagator(*proxy_args)
                    )
                    self.reduced_pulse_map_settings = {
                        "temporal_bins": temporal_bins,
                        "spectral_bins": spectral_bins,
                        "field_model": "phase_insensitive_spectral_spatial_whole_pulse",
                    }
            failure_policy = str(self.seed_cfg.get("coherent", {}).get("failure_policy", "error"))
            explicit_hg_cpa = (
                self.seed_cfg.get("spatial_modes", {}).get("model")
                == "incoherent_hg"
            )
            fallback = (
                IncoherentHGSignalPropagator(*signal_args)
                if explicit_hg_cpa
                else SpectralSignalPropagator(*signal_args)
            )
            try:
                # Lazy import keeps all non-CPA workflows independent from the
                # optional coherent solver and its dependencies.
                fidelity_mode = str(
                    self.seed_cfg.get("coherent", {}).get(
                        "fidelity_mode", "configured"
                    )
                )
                if explicit_hg_cpa and fidelity_mode == "axisymmetric_kerr":
                    from .axisymmetric_hg_coherent import AxisymmetricHGCoherentSignalPropagator

                    primary = AxisymmetricHGCoherentSignalPropagator(*signal_args)
                elif explicit_hg_cpa:
                    from .coherent_signal import (
                        IncoherentHGCoherentSignalPropagator,
                    )

                    primary = IncoherentHGCoherentSignalPropagator(
                        *signal_args
                    )
                elif fidelity_mode in {
                    "axisymmetric_kerr",
                    "axisymmetric_transient_gain",
                }:
                    from .axisymmetric_coherent import (
                        AxisymmetricCoherentSignalPropagator,
                    )

                    primary = AxisymmetricCoherentSignalPropagator(*signal_args)
                else:
                    from .coherent_signal import CoherentSignalPropagator

                    primary = CoherentSignalPropagator(*signal_args)
                initial_failure = None
            except OPTIONAL_BACKEND_FAILURES as exc:
                if failure_policy == "error":
                    raise
                primary = None
                initial_failure = f"{type(exc).__name__}: {exc}"
            if failure_policy == "error":
                self.signal_solver = primary
            else:
                self.signal_solver = ResilientSignalPropagator(primary, fallback, initial_failure)
        elif resolved_signal_model == "josa_transport":
            josa_mode = str(self.seed_cfg.get("josa", {}).get("mode", "retarded_pc"))
            failure_policy = (
                "error"
                if josa_mode == JOSA_FULL_3D_MODE
                else str(self.seed_cfg.get("josa", {}).get("failure_policy", "error"))
            )
            fallback = SpectralSignalPropagator(*signal_args)
            try:
                from .josa_signal import JosaTransportSignalPropagator

                primary = JosaTransportSignalPropagator(*signal_args)
                initial_failure = None
            except OPTIONAL_BACKEND_FAILURES as exc:
                if failure_policy == "error":
                    raise
                primary = None
                initial_failure = f"{type(exc).__name__}: {exc}"
            if failure_policy == "error":
                self.signal_solver = primary
            else:
                self.signal_solver = ResilientSignalPropagator(
                    primary,
                    fallback,
                    initial_failure,
                    feature_name="josa_transport",
                    fallback_model="josa_transport_fallback",
                    diagnostic_target="gain_transport",
                )
        else:
            if (
                resolved_signal_model == "spectral_time"
                and causal_multipass
            ):
                from .causal_multipass_signal import (
                    CausalMultipassSpectralSignalPropagator,
                )

                signal_class = CausalMultipassSpectralSignalPropagator
            else:
                signal_class = (
                    SpectralSignalPropagator
                    if resolved_signal_model == "spectral_time"
                    else SignalPropagator
                )
            self.signal_solver = signal_class(*signal_args)
        if self.grid_coupler is not None:
            self.signal_solver = CoupledSignalPropagator(
                self.signal_solver,
                self.grid_coupler,
                self.crystal_mask,
            )
            if self.reduced_pulse_map_signal_solver is not None:
                self.reduced_pulse_map_signal_solver = CoupledSignalPropagator(
                    self.reduced_pulse_map_signal_solver,
                    self.grid_coupler,
                    self.crystal_mask,
                )
        self._seed_progress_details_enabled = False
        self._install_signal_transport_progress()

    def set_continuation_state(
        self,
        state: SimulationContinuationState,
    ) -> None:
        """Install a validated warm start exported by a nearby solved point."""

        if not isinstance(state, SimulationContinuationState):
            raise TypeError("state must be a SimulationContinuationState")

        def volume_field(
            value: np.ndarray | None,
            name: str,
            *,
            positive: bool = False,
        ) -> np.ndarray | None:
            if value is None:
                return None
            field = np.asarray(value, dtype=float)
            if field.shape != self.grid.shape or not np.all(
                np.isfinite(field)
            ):
                raise ValueError(
                    f"continuation {name} must be finite and match the physical grid"
                )
            if positive and float(np.min(field)) <= 0.0:
                raise ValueError(f"continuation {name} must be strictly positive")
            return field.copy()

        population = state.periodic_boundary_population_m3
        if population is not None:
            population = np.asarray(population, dtype=float)
            expected_shape = (len(self.levels.names), *self.grid.shape)
            if population.shape != expected_shape or not np.all(
                np.isfinite(population)
            ):
                raise ValueError(
                    "continuation periodic population must be finite and match "
                    "the level/grid shape"
                )
            population = population.copy()

        def surface_field(
            value: np.ndarray | None,
            name: str,
        ) -> np.ndarray | None:
            if value is None:
                return None
            field = np.asarray(value, dtype=float)
            if field.shape != self.grid.shape[-2:] or not np.all(
                np.isfinite(field)
            ):
                raise ValueError(
                    f"continuation {name} must be finite and match the transverse grid"
                )
            return field.copy()

        self._continuation_state = SimulationContinuationState(
            periodic_boundary_population_m3=population,
            thermal_delta_n=volume_field(
                state.thermal_delta_n, "thermal_delta_n"
            ),
            response_temperature_K=volume_field(
                state.response_temperature_K,
                "response_temperature_K",
                positive=True,
            ),
            thermal_temperature_K=volume_field(
                state.thermal_temperature_K,
                "thermal_temperature_K",
                positive=True,
            ),
            front_surface_sag_m=surface_field(
                state.front_surface_sag_m, "front_surface_sag_m"
            ),
            back_surface_sag_m=surface_field(
                state.back_surface_sag_m, "back_surface_sag_m"
            ),
        )
        self._continuation_thermal_guess_consumed = False

    def export_continuation_state(
        self,
        result: SimulationResult,
    ) -> SimulationContinuationState:
        """Return copies of the converged state needed by the next Pump point."""

        if result.grid.shape != self.grid.shape:
            raise ValueError("result grid does not match this simulator")
        front = result.stress.get("front_surface_sag_m")
        back = result.stress.get("back_surface_sag_m")
        periodic_population = self._last_periodic_boundary_population
        return SimulationContinuationState(
            periodic_boundary_population_m3=(
                None
                if periodic_population is None
                else periodic_population.copy()
            ),
            thermal_delta_n=np.asarray(result.delta_n, dtype=float).copy(),
            response_temperature_K=np.asarray(
                result.temperature_K, dtype=float
            ).copy(),
            thermal_temperature_K=np.asarray(
                result.temperature_K, dtype=float
            ).copy(),
            front_surface_sag_m=(
                None if front is None else np.asarray(front, dtype=float).copy()
            ),
            back_surface_sag_m=(
                None if back is None else np.asarray(back, dtype=float).copy()
            ),
        )

    def _install_signal_transport_progress(self) -> None:
        """Attach one observer to every possible primary/fallback Seed backend."""

        stack = [
            self.signal_solver,
            self.average_signal_solver,
            self.reduced_pulse_map_signal_solver,
        ]
        visited: set[int] = set()
        while stack:
            target = stack.pop()
            if target is None or id(target) in visited:
                continue
            visited.add(id(target))
            setattr(
                target,
                "_transport_progress_callback",
                self._signal_transport_progress,
            )
            for attribute in ("inner", "primary", "fallback"):
                child = getattr(target, attribute, None)
                if child is not None:
                    stack.append(child)

    def _signal_transport_progress(self, event: dict[str, Any]) -> None:
        if (
            not self._seed_progress_details_enabled
            or getattr(self, "_progress_mode", "detailed") == "milestones"
        ):
            return
        phase = str(event.get("phase", "crystal_transport"))
        current = int(event.get("current", 0))
        total = int(event.get("total", 0))
        transport_metrics = {
            **dict(getattr(self, "_seed_progress_context", {})),
            **dict(event.get("metrics", {})),
        }
        transport_metrics.update(
            {
                "seed_substage": phase,
                "backend": event.get("backend"),
            }
        )
        if phase == "spectral_source_setup":
            transport_metrics.update(
                {
                    "source_channel_index": current,
                    "source_channel_total": total,
                }
            )
            wavelength = transport_metrics.get("source_channel_wavelength_m")
            wavelength_text = (
                ""
                if wavelength is None
                else f", wavelength {float(wavelength) * 1e9:.2f} nm"
            )
            self._emit_progress(
                "seed_propagation",
                (
                    f"Preparing Seed spectral channel {current}/{total}"
                    f"{wavelength_text}"
                ),
                fraction=float(event.get("fraction", 0.0)),
                current=current,
                total=total,
                metrics=transport_metrics,
            )
            return
        if phase == "output_postprocess":
            self._emit_progress(
                "seed_propagation",
                f"Building Seed output diagnostics {current}/{total}",
                fraction=float(event.get("fraction", 0.0)),
                current=current,
                total=total,
                metrics=transport_metrics,
            )
            return
        pass_index = int(event.get("pass_index", 1))
        pass_total = int(event.get("pass_total", 1))
        cells_per_pass = max(total // max(pass_total, 1), 1)
        cell_target = transport_metrics.get("crystal_cell_target")
        cell_in_pass = (
            int(cell_target)
            if cell_target is not None
            else max(current - (pass_index - 1) * cells_per_pass, 0)
        )
        z_index = event.get("z_index")
        z_position_m = None
        if z_index is not None and 0 <= int(z_index) < len(self.grid.z):
            z_position_m = float(self.grid.z[int(z_index)])
        transport_metrics.update(
            {
                "pass": f"{pass_index}/{pass_total}",
                "pass_index": pass_index,
                "pass_total": pass_total,
                "crystal_cell": cell_in_pass,
                "crystal_cells": cells_per_pass,
                "z_index": z_index,
                "z_position_m": z_position_m,
                "backend": event.get("backend"),
            }
        )
        self._emit_progress(
            "seed_propagation",
            (
                f"Seed axial transport: pass {pass_index}/{pass_total}, "
                f"crystal cell {current}/{total}"
            ),
            fraction=float(event.get("fraction", 0.0)),
            current=current,
            total=total,
            metrics=transport_metrics,
        )

    def _population_progress_metrics(
        self, population: np.ndarray
    ) -> dict[str, float]:
        """Return cheap scalar population observables for live progress only."""

        values = np.asarray(population, dtype=float)
        density_source = getattr(self, "dopant_density_m3", None)
        level_names = getattr(getattr(self, "levels", None), "names", None)
        if density_source is None or level_names is None:
            return {}
        density = np.asarray(density_source, dtype=float)
        active = density > np.finfo(float).tiny
        if values.shape[1:] != self.grid.shape or not np.any(active):
            return {}
        fractions = np.divide(
            values,
            density[None, ...],
            out=np.zeros_like(values),
            where=density[None, ...] > np.finfo(float).tiny,
        )
        weights = np.asarray(getattr(self.grid, "cell_volumes", np.ones(self.grid.shape)))[active]
        upper_indices = [
            index
            for index, name in enumerate(level_names)
            if str(name) == "upper" or str(name).startswith("upper[")
        ]
        if upper_indices:
            upper = np.sum(fractions[upper_indices], axis=0)[active]
            return {
                "upper_population_fraction_mean": float(np.average(upper, weights=weights)),
                "upper_population_fraction_max": float(np.max(upper)),
            }
        ground_indices = [
            index
            for index, name in enumerate(level_names)
            if str(name) == "ground" or str(name).startswith("ground[")
        ]
        if not ground_indices:
            return {}
        excited = 1.0 - np.sum(fractions[ground_indices], axis=0)[active]
        return {
            "excited_population_fraction_mean": float(np.average(excited, weights=weights)),
            "excited_population_fraction_max": float(np.max(excited)),
        }

    def _signal_progress_metrics(
        self,
        result: SignalPropagationResult,
        *,
        include_spectrum_preview: bool = False,
    ) -> dict[str, Any]:
        """Extract already-computed pulse observables without retaining fields."""

        metrics: dict[str, Any] = {
            "output_energy_J": float(result.system_output_energy_J),
        }
        repetition_rate = float(self.sim_cfg.get("repetition_rate_Hz", 0.0))
        if repetition_rate > 0.0:
            metrics["output_average_power_W"] = (
                float(result.system_output_energy_J) * repetition_rate
            )
        if result.output_spectrum_J is not None and result.wavelengths_m is not None:
            spectrum = np.asarray(result.output_spectrum_J, dtype=float)
            wavelengths = np.asarray(result.wavelengths_m, dtype=float)
            if spectrum.size and wavelengths.shape == spectrum.shape:
                try:
                    density = spectral_density_from_bin_energy(
                        wavelengths,
                        np.maximum(spectrum, 0.0),
                        bin_coordinate=spectral_bin_coordinate_for_result(
                            str(result.model_used),
                            self.seed_cfg,
                        ) if result.spectral_bin_edges_Hz is None else "frequency",
                        bin_edges=result.spectral_bin_edges_Hz,
                    )
                except ValueError:
                    pass
                else:
                    preview_wavelength = np.asarray(
                        density["wavelength_m"], dtype=float
                    )
                    preview_density = np.asarray(
                        density["normalized_wavelength_density"],
                        dtype=float,
                    )
                    metrics["spectral_peak_wavelength_m"] = float(
                        preview_wavelength[int(np.argmax(preview_density))]
                    )
                    metrics["spectral_peak_basis"] = "wavelength_spectral_density"
                    if include_spectrum_preview:
                        maximum_points = 65
                        if len(preview_wavelength) > maximum_points:
                            indices = np.unique(
                                np.rint(
                                    np.linspace(
                                        0,
                                        len(preview_wavelength) - 1,
                                        maximum_points,
                                    )
                                ).astype(int)
                            )
                            preview_wavelength = preview_wavelength[indices]
                            preview_density = preview_density[indices]
                        metrics["spectrum_preview_wavelength_m"] = (
                            preview_wavelength.tolist()
                        )
                        metrics["spectrum_preview_relative"] = (
                            preview_density.tolist()
                        )
        cpa = result.cpa_diagnostics or {}
        for source, target in (
            ("output_peak_power_W", "output_peak_power_W"),
            ("compressed_peak_power_W", "compressed_peak_power_W"),
            ("output_fwhm_s", "output_fwhm_s"),
            ("compressed_fwhm_s", "compressed_fwhm_s"),
            ("maximum_cumulative_B_integral_rad", "b_integral_rad"),
            (
                "maximum_single_step_nonlinear_phase_rad",
                "maximum_nonlinear_substep_rad",
            ),
            (
                "maximum_temporal_boundary_energy_fraction",
                "temporal_boundary_fraction",
            ),
            (
                "maximum_output_spectral_boundary_energy_fraction",
                "spectral_boundary_fraction",
            ),
            (
                "maximum_local_nonlinear_intensity_W_m2",
                "maximum_local_nonlinear_intensity_W_m2",
            ),
        ):
            if cpa.get(source) is not None:
                metrics[target] = cpa[source]
        strong_field = cpa.get("strong_field", {})
        if isinstance(strong_field, dict):
            for source in (
                "multiphoton_absorbed_energy_J",
                "free_carrier_absorbed_energy_J",
                "maximum_peak_carrier_density_m3",
            ):
                value = strong_field.get(source)
                if value is not None:
                    metrics[source] = value
        if result.output_field_diagnostics is not None:
            field_diagnostics = copy.deepcopy(result.output_field_diagnostics)
            metrics["output_field_diagnostics"] = field_diagnostics
            quality = field_diagnostics.get("beam_quality", {})
            finite_quality = field_diagnostics.get(
                "beam_quality_finite_difference", {}
            )
            if quality.get("m2_x") is not None:
                metrics["output_M2_x"] = quality["m2_x"]
            if quality.get("m2_y") is not None:
                metrics["output_M2_y"] = quality["m2_y"]
            if finite_quality.get("m2_x") is not None:
                metrics["output_M2_x_finite_difference"] = finite_quality["m2_x"]
            if finite_quality.get("m2_y") is not None:
                metrics["output_M2_y_finite_difference"] = finite_quality["m2_y"]
        for key in (
            "b_integral_distribution",
            "b_integral_per_pass",
            "nonlinear_phase_sampling",
            "kerr_self_focusing_screen",
            "local_operator",
        ):
            value = cpa.get(key)
            if value:
                metrics[key] = copy.deepcopy(value)
        self_focusing = cpa.get("kerr_self_focusing_screen", {})
        if isinstance(self_focusing, dict):
            ratio = self_focusing.get("peak_to_critical_power_ratio")
            if ratio is not None:
                metrics["peak_to_critical_power_ratio"] = ratio
        compact_passes: list[dict[str, Any]] = []
        for row in result.passes:
            compact = {
                key: copy.deepcopy(row[key])
                for key in (
                    "pass",
                    "direction",
                    "gain_input_J",
                    "gain_output_J",
                    "crystal_exit_J",
                    "crystal_aperture_loss_J",
                    "scalar_diagnostics",
                )
                if row.get(key) is not None
            }
            if compact:
                compact_passes.append(compact)
        if compact_passes:
            metrics["pass_diagnostics"] = compact_passes
        gain_transport = result.gain_transport or {}
        recording_contract = gain_transport.get("result_recording")
        if recording_contract:
            metrics["result_recording"] = copy.deepcopy(recording_contract)
        longitudinal_profile = gain_transport.get("longitudinal_profile")
        if (
            self.result_recording["level"] == "detailed"
            and longitudinal_profile
        ):
            metrics["longitudinal_profile"] = copy.deepcopy(
                longitudinal_profile
            )
        return metrics

    def _record_nested_population_history(
        self,
        target: list[dict[str, Any]],
        statistics: dict[str, Any],
        *,
        outer_iteration: int,
        outer_stage: str,
    ) -> None:
        """Retain scalar inner histories that outer fixed points used to drop."""

        if not bool(self.result_recording["iteration_history"]):
            return
        history = statistics.get("convergence_history", [])
        if not isinstance(history, list):
            return
        detailed = self.result_recording["level"] == "detailed"
        for inner_index, source in enumerate(history, start=1):
            if not isinstance(source, dict):
                continue
            row = copy.deepcopy(source)
            if not detailed:
                for key in (
                    "longitudinal_profile",
                    "spectrum_preview_wavelength_m",
                    "spectrum_preview_relative",
                ):
                    row.pop(key, None)
            row["outer_iteration"] = int(outer_iteration)
            row["outer_stage"] = str(outer_stage)
            row["inner_iteration"] = int(
                source.get("cycle", source.get("iteration", inner_index))
            )
            target.append(row)

    def _cycle_gain_overlap_metrics(
        self,
        pre_seed_population: np.ndarray,
        result: SignalPropagationResult,
    ) -> dict[str, Any]:
        """Return compact gain-shape diagnostics weighted by sampled Seed fluence."""

        fluence_source = (
            result.fluence_last_pass_J_m2
            if result.fluence_last_pass_J_m2 is not None
            else result.fluence_J_m2
        )
        fluence = np.asarray(fluence_source, dtype=float)
        if fluence.shape != self.grid.shape:
            return {
                "gain_overlap_diagnostics": {
                    "applicable": False,
                    "reason": "signal fluence and population grids differ",
                }
            }
        pre_gain = np.asarray(
            self._signal_active_gain(pre_seed_population),
            dtype=float,
        )
        post_gain = np.asarray(
            self._remaining_override_gain(result)
            if self.config["energy_levels"].get("gain_override_m_inv")
            is not None
            else self._signal_active_gain(result.population_m3),
            dtype=float,
        )
        weights = np.maximum(fluence, 0.0) * self.grid.cell_volumes
        total_weight = float(np.sum(weights))
        if total_weight <= np.finfo(float).tiny:
            return {
                "gain_overlap_diagnostics": {
                    "applicable": False,
                    "reason": "sampled signal fluence is zero",
                }
            }

        def weighted_gain_stats(values: np.ndarray) -> tuple[float, float]:
            mean = float(np.sum(values * weights) / total_weight)
            standard_deviation = float(
                np.sqrt(
                    max(
                        float(np.sum((values - mean) ** 2 * weights) / total_weight),
                        0.0,
                    )
                )
            )
            return mean, standard_deviation

        pre_mean, pre_standard_deviation = weighted_gain_stats(pre_gain)
        post_mean, post_standard_deviation = weighted_gain_stats(post_gain)
        transverse_fluence = np.sum(
            np.maximum(fluence, 0.0) * self.grid.z_widths[:, None, None],
            axis=0,
        )
        fluence_weight = transverse_fluence * self.grid.cell_areas
        fluence_total = max(float(np.sum(fluence_weight)), np.finfo(float).tiny)
        xx, yy = self.grid.transverse_mesh
        signal_center_x = float(np.sum(fluence_weight * xx) / fluence_total)
        signal_center_y = float(np.sum(fluence_weight * yy) / fluence_total)

        positive_gain_projection = np.sum(
            np.maximum(pre_gain, 0.0) * self.grid.z_widths[:, None, None],
            axis=0,
        )
        positive_gain_weight = positive_gain_projection * self.grid.cell_areas
        positive_gain_total = float(np.sum(positive_gain_weight))
        if positive_gain_total > np.finfo(float).tiny:
            gain_center_x = float(
                np.sum(positive_gain_weight * xx) / positive_gain_total
            )
            gain_center_y = float(
                np.sum(positive_gain_weight * yy) / positive_gain_total
            )
            offset_x = signal_center_x - gain_center_x
            offset_y = signal_center_y - gain_center_y
        else:
            gain_center_x = None
            gain_center_y = None
            offset_x = None
            offset_y = None
        return {
            "gain_overlap_diagnostics": {
                "applicable": True,
                "method": "last_pass_fluence_weighted_gain_and_positive_gain_centroid",
                "pre_seed_gain_mean_m_inv": pre_mean,
                "pre_seed_gain_standard_deviation_m_inv": pre_standard_deviation,
                "pre_seed_gain_coefficient_of_variation": float(
                    pre_standard_deviation
                    / max(abs(pre_mean), np.finfo(float).tiny)
                ),
                "post_seed_gain_mean_m_inv": post_mean,
                "post_seed_gain_standard_deviation_m_inv": post_standard_deviation,
                "signal_fluence_centroid_x_m": signal_center_x,
                "signal_fluence_centroid_y_m": signal_center_y,
                "positive_gain_centroid_x_m": gain_center_x,
                "positive_gain_centroid_y_m": gain_center_y,
                "signal_minus_gain_centroid_x_m": offset_x,
                "signal_minus_gain_centroid_y_m": offset_y,
                "signal_gain_centroid_offset_m": (
                    None
                    if offset_x is None or offset_y is None
                    else float(np.hypot(offset_x, offset_y))
                ),
                "crystal_aperture_loss_energy_J": float(
                    result.crystal_aperture_loss_energy_J
                ),
            }
        }

    @contextmanager
    def _progress_scope(self, start: float, stop: float) -> Iterator[None]:
        """Map local progress in ``[0, 1]`` into a parent solve interval."""

        ranges = getattr(self, "_progress_ranges", None)
        if not ranges:
            # Unit-level callers may deliberately construct a simulator with
            # ``__new__`` to exercise one numerical routine in isolation.
            yield
            return
        parent_start, parent_stop = ranges[-1]
        parent_span = parent_stop - parent_start
        local_start = min(max(float(start), 0.0), 1.0)
        local_stop = min(max(float(stop), local_start), 1.0)
        self._progress_ranges.append(
            (
                parent_start + parent_span * local_start,
                parent_start + parent_span * local_stop,
            )
        )
        try:
            yield
        finally:
            self._progress_ranges.pop()

    def _emit_progress(
        self,
        stage: str,
        message: str,
        *,
        fraction: float | None = None,
        current: int | None = None,
        total: int | None = None,
        metrics: dict[str, Any] | None = None,
        state: str = "running",
    ) -> None:
        """Publish a lightweight, UI-independent solver progress event.

        The callback is observational: a rendering failure must never alter a
        physical solve.  Nested fixed points use :meth:`_progress_scope` so the
        top-level bar remains monotonic even when an inner loop restarts.
        """

        callback = getattr(self, "_progress_callback", None)
        if callback is None:
            return
        if (
            getattr(self, "_progress_mode", "detailed") == "milestones"
            and not self._is_progress_milestone(
                stage=stage,
                message=message,
                fraction=fraction,
                current=current,
                metrics=metrics,
                state=state,
            )
        ):
            return
        overall: float | None = None
        if fraction is not None:
            local = min(max(float(fraction), 0.0), 1.0)
            start, stop = self._progress_ranges[-1]
            overall = start + (stop - start) * local
            overall = max(self._last_progress_fraction, min(overall, 1.0))
            self._last_progress_fraction = overall
        self._progress_sequence += 1
        event_metrics = {
            **dict(getattr(self, "_outer_progress_context", {})),
            **copy.deepcopy(metrics or {}),
        }
        event = {
            "sequence": self._progress_sequence,
            "stage": str(stage),
            "message": str(message),
            "state": str(state),
            "overall_fraction": overall,
            "current": None if current is None else int(current),
            "total": None if total is None else int(total),
            "elapsed_s": max(time.perf_counter() - self._progress_started_s, 0.0),
            "metrics": event_metrics,
        }
        try:
            callback(event)
        except Exception:
            # Progress is a diagnostic side channel. Disable a broken consumer
            # for the remainder of the run instead of corrupting solver state.
            self._progress_callback = None

    def _is_progress_milestone(
        self,
        *,
        stage: str,
        message: str,
        fraction: float | None,
        current: int | None,
        metrics: dict[str, Any] | None,
        state: str,
    ) -> bool:
        """Return whether a progress event is useful to a low-frequency UI.

        Milestone mode deliberately reports completed physical/fixed-point work,
        not axial cells, adaptive substeps, linear iterations, or Pump waveform
        samples.  It changes only the observational side channel; all numerical
        loops and convergence tests execute unchanged.
        """

        del message
        values = dict(metrics or {})
        stage_name = str(stage)
        if str(state) != "running" or stage_name == "complete":
            return True
        if stage_name == "seed_propagation":
            if values.get("seed_substage") is not None:
                return False
            # Hidden pulse-map/train propagations are represented by their
            # completed pulse summary instead of duplicate start/end events.
            if getattr(self, "_seed_progress_context", {}):
                return False
            return fraction is None or float(fraction) in {0.0, 1.0}
        if stage_name in {"pump_waveform", "thermal_solver"}:
            return False
        if stage_name == "population_steady_state":
            return fraction is not None and float(fraction) >= 1.0
        if stage_name == "thermal_optical_fixed_point":
            return current == 0 or "residual" in values
        if stage_name == "cycle_average":
            return (
                current == 0
                or "output_power_W" in values
                or values.get("converged") is not None
            )
        if stage_name in {"pulse_map", "reduced_pulse_map", "pulse_train"}:
            return (
                current == 0
                or values.get("output_energy_J") is not None
                or values.get("converged") is not None
            )
        return True

    def run(self) -> SimulationResult:
        self._progress_started_s = time.perf_counter()
        configure_compute_backend(self.config.get("performance", {}))
        configure_operation_profiler(self.config.get("performance", {}))
        reset_operation_profiler()
        self._progress_ranges = [(0.0, 1.0)]
        self._progress_sequence = 0
        self._last_progress_fraction = 0.0
        self._emit_progress(
            "initialization",
            "Preparing grids, material state, and solver backends",
            fraction=0.0,
        )
        self.run_context.reset()
        self._continuation_thermal_guess_consumed = False
        self._reduced_pulse_map_initializer_consumed = False
        # Periodic/transient operating-point solvers may arm a coherent
        # free-carrier boundary state for the final reported Seed pulse.
        # Never inherit such a one-shot state from an earlier top-level run.
        self._pending_seed_initial_free_carrier_density_m3 = None
        if self._thermal_solver is not None:
            self._thermal_solver.reset_initial_guess()
        if self._stress_solver is not None:
            reset_stress_guess = getattr(
                self._stress_solver, "reset_initial_guess", None
            )
            if reset_stress_guess is not None:
                reset_stress_guess()
        self._configure_endface_surface_sag(None)
        self.levels.reset_nonlinear_diagnostics()
        self.levels.reset_population_correction_diagnostics()
        self.levels.reset_signal_event_diagnostics()
        self.levels.reset_implicit_linear_solver_diagnostics()
        self.levels.reset_static_cross_section_cache_diagnostics()
        self._ase_execution = {
            "requested": bool(self.ase_solver.enabled),
            "executed": False,
            "disabled_after_failure": False,
            "disabled_reason": None,
        }
        self.levels.response.range_guard.reset()
        self.levels.set_environment(
            float(self.material.get("operating_temperature_K", 293.15)),
            self.dopant_density_m3,
        )
        coupling_model = str(
            self.config.get("multiphysics", {}).get("coupling_model", "one_way")
        )
        if coupling_model == "thermal_optical_fixed_point":
            self._emit_progress(
                "thermal_optical_fixed_point",
                "Starting coupled Pump, population, and thermal fixed point",
                fraction=0.02,
            )
            with self._progress_scope(0.02, 0.68):
                (
                    population,
                    population_stats,
                    pump_result,
                    heat,
                    temperature,
                    thermal_stats,
                    stress,
                    delta_n,
                    coupling_stats,
                ) = self._run_thermal_optical_fixed_point()
        else:
            zero_delta_n = np.zeros(self.grid.shape, dtype=float)
            continuation_population = (
                None
                if self._continuation_state is None
                else self._continuation_state.periodic_boundary_population_m3
            )
            self._emit_progress(
                "operating_point",
                "Solving Pump propagation and population operating point",
                fraction=0.02,
            )
            with self._progress_scope(0.02, 0.43):
                population, population_stats, pump_result = self._operating_point(
                    zero_delta_n,
                    bool(self.physics["record_optical_planes"]),
                    periodic_initial_population=continuation_population,
                    periodic_initialization_source="pump_sweep_previous_point",
                )
            self._emit_progress(
                "thermal_response",
                "Mapping absorbed power into heat, stress, and refractive index",
                fraction=0.43,
            )
            with self._progress_scope(0.43, 0.62):
                heat, temperature, thermal_stats, stress, delta_n = self._thermal_response(
                    pump_result,
                    population=population,
                )
            coupling_stats = {
                "model": "one_way",
                "converged": None,
                "iterations": 1,
                "feedback_to_pump": False,
                "feedback_to_population": False,
                "feedback_to_seed": bool(
                    self.physics["refractive_index_model"] == "thermal_3d"
                    and thermal_stats.get("enabled", False)
                ),
                "thermal_field_applied_to_seed": bool(
                    self.physics["refractive_index_model"] == "thermal_3d"
                    and thermal_stats.get("enabled", False)
                ),
                "seed_to_thermal_feedback": False,
                "pump_sweep_continuation_used": self._continuation_state
                is not None,
                "pump_sweep_population_warm_start_used": bool(
                    self._continuation_state is not None
                    and self._continuation_state.periodic_boundary_population_m3
                    is not None
                ),
                "pump_sweep_thermal_solver_warm_start_used": bool(
                    self._continuation_state is not None
                    and self._continuation_state.thermal_temperature_K is not None
                ),
                "description": (
                    "Pump and populations are solved without thermal aberration; the resulting "
                    "thermal delta-n is applied once to the reported Seed propagation."
                ),
            }
        coupling_stats.setdefault(
            "result_recording",
            copy.deepcopy(self.result_recording),
        )
        reference = float(self.material.get("reference_temperature_K", 293.15))
        stress_solver = self._stress_backend()
        population_after_pump = population.copy()
        inversion_after_pump = transition_inversion_m3(
            population_after_pump,
            self.levels.signal.upper,
            self.levels.signal.lower,
        )
        seed_heat_mode = str(
            self.config.get("multiphysics", {}).get("seed_heat_coupling", "off")
        )
        suppress_probe_diagnostics = seed_heat_mode == "one_way_postprocess"
        pre_seed_free_carrier = getattr(
            self,
            "_pending_seed_initial_free_carrier_density_m3",
            None,
        )
        if pre_seed_free_carrier is not None:
            pre_seed_free_carrier = np.asarray(
                pre_seed_free_carrier, dtype=float
            ).copy()
            # A thermal-optical fixed-point closure may already have run a
            # final hidden Seed probe.  Coherent carrier boundary states are
            # intentionally one-shot, so that probe consumes the state armed
            # by the periodic operating-point solve.  Re-arm the identical
            # pre-Seed state here before the user-facing propagation.  In the
            # one-way path this is idempotent and keeps one common dispatch
            # contract for both coupling modes.
            self._set_seed_initial_free_carrier(pre_seed_free_carrier)
        previous_suppression = self.physics.get("_suppress_output_diagnostics")
        if suppress_probe_diagnostics:
            self.physics["_suppress_output_diagnostics"] = True
        self._configure_signal_photoelastic_tensor(stress)
        self._configure_endface_surface_sag(stress)
        self._emit_progress(
            "seed_propagation",
            f"Propagating Seed with {self.signal_model}",
            fraction=0.69 if coupling_model == "thermal_optical_fixed_point" else 0.63,
        )
        try:
            with self._progress_scope(
                0.69 if coupling_model == "thermal_optical_fixed_point" else 0.63,
                0.82,
            ):
                signal_result = self._propagate_seed_with_progress(
                    population,
                    delta_n,
                    record_planes=bool(self.physics["record_optical_planes"]),
                )
                self._guard_reported_seed_backend_consistency(
                    signal_result,
                    population_stats,
                    seed_heat_mode,
                )
        finally:
            if suppress_probe_diagnostics:
                if previous_suppression is None:
                    self.physics.pop("_suppress_output_diagnostics", None)
                else:
                    self.physics["_suppress_output_diagnostics"] = previous_suppression
        population = signal_result.population_m3
        if seed_heat_mode == "one_way_postprocess":
            self._emit_progress(
                "seed_heat_postprocess",
                "Recomputing thermal field from irreversible Seed loss",
                fraction=0.82,
            )
            seed_heat = self._seed_irreversible_heat(signal_result)
            with self._progress_scope(0.82, 0.88):
                heat, temperature, thermal_stats, stress, delta_n = self._thermal_response(
                    pump_result,
                    seed_heat,
                    population=population_after_pump,
                )
            self._configure_signal_photoelastic_tensor(stress)
            self._configure_endface_surface_sag(stress)
            # Re-run from the same pre-Seed population.  The first pass is a
            # thermal source probe only and must never deplete the population
            # a second time.  Coherent FCA also consumes a one-shot entrance
            # carrier boundary state, so re-arm that same state before the
            # reported pass instead of silently restarting it from zero.
            if pre_seed_free_carrier is not None:
                self._set_seed_initial_free_carrier(pre_seed_free_carrier)
            with self._progress_scope(0.88, 0.92):
                signal_result = self._propagate_seed_with_progress(
                    population_after_pump.copy(),
                    delta_n,
                    record_planes=bool(self.physics["record_optical_planes"]),
                )
                self._guard_reported_seed_backend_consistency(
                    signal_result,
                    population_stats,
                    seed_heat_mode,
                )
            population = signal_result.population_m3
            coupling_stats["seed_to_thermal_feedback"] = True
            coupling_stats["seed_heat_one_way_postprocess"] = True
            coupling_stats["seed_heat_self_consistent"] = False
            coupling_stats["seed_heat_closure_iterations"] = 1
            seed_irreversible_heat_W = float(
                self.grid.integrate_volume(seed_heat)
            )
            coupling_stats["seed_irreversible_heat_W"] = (
                seed_irreversible_heat_W
            )
            # Backward-compatible result key retained for existing archives.
            # It historically included optional passive Seed absorption too,
            # despite the narrower name.
            coupling_stats["seed_nonlinear_heat_W"] = (
                seed_irreversible_heat_W
            )
            coupling_stats["description"] = (
                "Pump and populations are first solved without Seed-deposited "
                "heat. One Seed probe supplies irreversible loss to a single "
                "thermal update, and the reported Seed is then propagated once "
                "through that updated field. This is one Picard update, not a "
                "self-consistent Seed-thermal fixed point."
            )
        if population_stats.get("dynamics") == "transient_pulse_train":
            pulse_total = int(population_stats.get("pulse_train_cycles", 1))
            final_pulse_metrics = {
                "pulse_index": pulse_total,
                "pulse_total": pulse_total,
                "repetition_rate_Hz": float(
                    population_stats.get(
                        "repetition_rate_Hz",
                        self.sim_cfg.get("repetition_rate_Hz", 0.0),
                    )
                ),
                "seed_pulse_applied": True,
                **self._signal_progress_metrics(
                    signal_result,
                    include_spectrum_preview=True,
                ),
                **self._population_progress_metrics(population),
            }
            pulse_history = population_stats.get("pulse_history")
            if isinstance(pulse_history, list) and pulse_history:
                final_pulse_metrics = {
                    **dict(pulse_history[-1]),
                    **final_pulse_metrics,
                }
                final_pulse_metrics.pop("awaiting_reported_seed_pulse", None)
                pulse_history[-1] = copy.deepcopy(final_pulse_metrics)
            self._emit_progress(
                "pulse_train",
                f"Reported Seed pulse {pulse_total}/{pulse_total} completed",
                current=pulse_total,
                total=pulse_total,
                metrics=final_pulse_metrics,
            )
        self._emit_progress(
            "diagnostics",
            "Building energy ledgers, validity checks, and output diagnostics",
            fraction=0.88 if seed_heat_mode != "one_way_postprocess" else 0.92,
        )
        initial_active_gain = self._signal_active_gain(population_after_pump)
        if self.config["energy_levels"].get("gain_override_m_inv") is None:
            remaining_active_gain = self._signal_active_gain(population)
        else:
            remaining_active_gain = self._remaining_override_gain(signal_result)
        passive_loss = (
            float(self.material.get("signal_passive_loss_m_inv", 0.0))
            * self.crystal_mask
        )
        gain = (remaining_active_gain - passive_loss) * self.crystal_mask
        signal_result.initial_active_gain_m_inv = initial_active_gain.copy()
        signal_result.remaining_active_gain_m_inv = remaining_active_gain.copy()
        signal_result.passive_loss_m_inv = passive_loss.copy()
        signal_result.net_gain_m_inv = gain.copy()
        density = self.dopant_density_m3
        if population_stats.get("dynamics") == "cycle_averaged_steady_state":
            pulse_jump = self._population_residual_relative(
                population, population_after_pump
            )
            population_stats["final_pulse_population_jump_relative"] = pulse_jump
            reported_average_power = (
                float(signal_result.system_output_energy_J)
                * float(self.sim_cfg.get("repetition_rate_Hz", 0.0))
            )
            probe_average_power = float(
                population_stats.get("average_probe_output_power_W", 0.0)
            )
            population_stats["reported_pulse_output_average_power_W"] = reported_average_power
            population_stats["reported_vs_average_probe_relative_difference"] = abs(
                reported_average_power - probe_average_power
            ) / max(
                abs(reported_average_power),
                abs(probe_average_power),
                np.finfo(float).tiny,
            )
        elif population_stats.get("dynamics") == "reduced_periodic_steady_state":
            proxy_post_population = getattr(
                self,
                "_last_periodic_boundary_population",
                None,
            )
            if proxy_post_population is None:
                raise RuntimeError(
                    "reduced pulse map did not retain its post-Seed boundary population"
                )
            proxy_output_energy = float(
                population_stats.get("reduced_proxy_output_energy_J", 0.0)
            )
            reported_output_energy = float(signal_result.system_output_energy_J)
            population_stats[
                "reported_vs_reduced_proxy_population_relative_difference"
            ] = self._population_residual_relative(
                population,
                np.asarray(proxy_post_population, dtype=float),
            )
            population_stats[
                "reported_vs_reduced_proxy_output_relative_difference"
            ] = abs(reported_output_energy - proxy_output_energy) / max(
                abs(reported_output_energy),
                abs(proxy_output_energy),
                np.finfo(float).tiny,
            )
            # Preserve the historical key for archive readers; the configured
            # final propagation may itself be the axisymmetric coherent mode.
            population_stats["reported_full_field_output_energy_J"] = (
                reported_output_energy
            )
            population_stats["reported_configured_output_energy_J"] = (
                reported_output_energy
            )
        expected_density = density
        population_error = float(np.max(np.abs(np.sum(population, axis=0) - expected_density)))
        active = self.crystal_mask & (density > 0.0)
        if not np.any(active):
            raise ValueError("The configured dopant profile contains no positive-density crystal cells")
        local_population_error = np.abs(np.sum(population, axis=0) - expected_density)
        population_error_relative = float(
            np.max(local_population_error[active] / density[active])
        )
        transition_inversions: list[dict[str, Any]] = []
        for transition in self.levels.signal_transitions:
            after_pump_transition_inversion = transition_inversion_m3(
                population_after_pump,
                transition.upper,
                transition.lower,
            )
            final_transition_inversion = transition_inversion_m3(
                population,
                transition.upper,
                transition.lower,
            )
            transition_inversions.append(
                {
                    "name": transition.name,
                    "ensemble": transition.ensemble,
                    "lower_level": self.levels.names[transition.lower],
                    "upper_level": self.levels.names[transition.upper],
                    "after_pump_min_m3": float(
                        np.min(after_pump_transition_inversion[active])
                    ),
                    "after_pump_max_m3": float(
                        np.max(after_pump_transition_inversion[active])
                    ),
                    "final_min_m3": float(
                        np.min(final_transition_inversion[active])
                    ),
                    "final_max_m3": float(
                        np.max(final_transition_inversion[active])
                    ),
                }
            )
        ensemble_inventory = self.levels.ensemble_inventory_diagnostics(
            population, density
        )
        ensemble_inventory["after_pump"] = (
            self.levels.ensemble_inventory_diagnostics(
                population_after_pump, density
            )
        )
        reported_wavelengths = (
            np.asarray(signal_result.wavelengths_m, dtype=float)
            if signal_result.wavelengths_m is not None
            else np.asarray(
                [float(self.seed_cfg["wavelength_m"])], dtype=float
            )
        )
        if len(reported_wavelengths) > 129:
            reported_wavelengths = reported_wavelengths[
                np.unique(
                    np.linspace(
                        0,
                        len(reported_wavelengths) - 1,
                        129,
                        dtype=int,
                    )
                )
            ]
        spectral_active_gain = (
            np.broadcast_to(remaining_active_gain, (len(reported_wavelengths), *self.grid.shape))
            if self.config["energy_levels"].get("gain_override_m_inv") is not None
            else self.levels.signal_gain_spectrum_m_inv(
                population,
                reported_wavelengths,
            )
        )
        spectral_gain = (
            spectral_active_gain
            - float(self.material.get("signal_passive_loss_m_inv", 0.0))
            * self.crystal_mask[None, ...]
        )
        crystal_cell_volumes = self.grid.cell_volumes[self.crystal_mask]
        frequency_gain_summary = [
            {
                "wavelength_m": float(wavelength),
                "frequency_Hz": float(LIGHT_SPEED / wavelength),
                "net_gain_min_m_inv": float(
                    np.min(spectral_gain[index][self.crystal_mask])
                ),
                "net_gain_max_m_inv": float(
                    np.max(spectral_gain[index][self.crystal_mask])
                ),
                "net_gain_mean_m_inv": float(
                    np.average(spectral_gain[index][self.crystal_mask],
                               weights=crystal_cell_volumes)
                ),
            }
            for index, wavelength in enumerate(reported_wavelengths)
        ]
        maximum_dark_outflow_rate_s = (
            self.levels.maximum_dark_linear_outflow_rate_s(self.grid.shape)
        )
        validity, validity_warnings = pulse_relaxation_validity(
            self.config,
            signal_result.model_used or self.signal_model,
            signal_result.cpa_diagnostics,
            self.physics,
            actual_fastest_dark_linear_relaxation_s=(
                None
                if maximum_dark_outflow_rate_s <= 0.0
                else 1.0 / maximum_dark_outflow_rate_s
            ),
        )
        warnings = self._diagnostics(
            pump_result.stats,
            signal_result.source_grid_energy_J,
            [
                *pump_result.warnings,
                *signal_result.warnings,
                *((signal_result.caustic_diagnostics or {}).get("warnings", [])),
                *population_stats.get("average_probe_warnings", []),
                *self._ase_warnings,
                *self._material_library_warnings(),
                *compute_backend_warnings(),
                *validity_warnings,
            ],
        )
        if coupling_model == "thermal_optical_fixed_point" and not bool(
            coupling_stats.get("converged", False)
        ):
            warnings.append(
                "The thermal-optical fixed-point calculation did not return a fully converged "
                "state: at least one of delta-n, absorbed Pump power, population, spectroscopy "
                "temperature, or the inner thermal solve missed its convergence criterion."
            )
        if population_stats.get("dynamics") == "cycle_averaged_steady_state":
            pulse_jump = float(population_stats["final_pulse_population_jump_relative"])
            if not bool(population_stats.get("converged", False)):
                warnings.append(
                    "Cycle-average Pump+Seed steady state reached its iteration limit before "
                    "both population and output-power residuals met the requested tolerance."
                )
            if pulse_jump > 1e-2:
                warnings.append(
                    "Cycle-average steady state predicts that one reported Seed pulse changes a local "
                    f"level population by {pulse_jump:.3%}; use the exact pulse-map steady state before "
                    "interpreting this high-depletion result."
                )
            probe_difference = float(
                population_stats["reported_vs_average_probe_relative_difference"]
            )
            if probe_difference > 5e-2:
                warnings.append(
                    "The final selected pulse solver and the cycle-average linear probe differ in "
                    f"output average power by {probe_difference:.3%}; inspect single-pulse depletion, "
                    "coherent/thermal propagation, and prefer the exact pulse map when needed."
                )
        elif population_stats.get("dynamics") == "reduced_periodic_steady_state":
            reduced_population_difference = float(
                population_stats[
                    "reported_vs_reduced_proxy_population_relative_difference"
                ]
            )
            reduced_output_difference = float(
                population_stats[
                    "reported_vs_reduced_proxy_output_relative_difference"
                ]
            )
            strict_cfg = self.sim_cfg.get("strict_validation", {})
            reduced_population_limit = float(
                strict_cfg.get(
                    "reduced_pulse_map_population_difference_relative_tolerance",
                    1e-2,
                )
            )
            reduced_output_limit = float(
                strict_cfg.get(
                    "reduced_pulse_map_output_difference_relative_tolerance",
                    5e-2,
                )
            )
            if reduced_population_difference > reduced_population_limit:
                warnings.append(
                    "The reduced discrete pulse map and the reported configured "
                    "pulse differ in post-pulse population by "
                    f"{reduced_population_difference:.3%}; refine the proxy or "
                    "use the exact pulse map before quantitative interpretation."
                )
            if reduced_output_difference > reduced_output_limit:
                warnings.append(
                    "The reduced discrete pulse map and the reported configured "
                    "pulse differ in output energy by "
                    f"{reduced_output_difference:.3%}; refine the proxy or use "
                    "the exact pulse map before quantitative interpretation."
                )
        initializer = population_stats.get("initializer", {})
        if initializer.get("failure"):
            warnings.append(
                "The optional cycle-average initializer failed and the exact periodic map safely "
                f"continued from the original population ({initializer['failure']})."
            )
        population_acceleration = population_stats.get("acceleration", {})
        if (
            population_acceleration.get("enabled", False)
            and population_acceleration.get("memory_limited", False)
            and int(population_acceleration.get("effective_history_depth", 0)) == 0
        ):
            warnings.append(
                "Periodic/average fixed-point Anderson acceleration was disabled by its history-memory "
                "limit; the original fixed-point iteration remained active."
            )
        coupling_acceleration = coupling_stats.get("acceleration", {})
        if (
            coupling_acceleration.get("enabled", False)
            and coupling_acceleration.get("memory_limited", False)
            and int(coupling_acceleration.get("effective_history_depth", 0)) == 0
        ):
            warnings.append(
                "Thermal-optical Anderson acceleration was disabled by its history-memory limit; "
                "the relaxed fixed-point iteration remained active."
            )
        population_dynamics = str(population_stats.get("dynamics", ""))
        if (
            population_dynamics in {"steady_state", "periodic_steady_state"}
            and population_stats.get("converged") is False
        ):
            if population_dynamics == "periodic_steady_state":
                warnings.append(
                    "The exact Pump-period/Seed-pulse map reached its cycle limit before "
                    "population and output energy met the requested periodic steady-state tolerance."
                )
            else:
                warnings.append(
                    "The stationary Pump/population iteration reached its iteration limit before "
                    "the local population met the convergence tolerance."
                )
        if bool(thermal_stats.get("enabled", False)) and thermal_stats.get(
            "converged"
        ) is False:
            warnings.append(
                "The thermal solver reached its iteration/time-step limit without meeting its "
                "configured convergence criterion; temperature, stress, and thermal lens may be inaccurate."
            )
        if self._uses_cw_average_for_configured_pump_waveform():
            warnings.append(
                "A non-CW Pump waveform is being represented by its configured average power in "
                "the stationary population solve. Pulse-shape dynamics are not resolved; select "
                "transient population dynamics or periodic steady state when modulation matters."
            )
        if not self.physics["pump_depletion"] and pump_result.stats["absorbed_power_W"] > pump_result.stats["grid_input_power_W"]:
            warnings.append("Pump depletion is off, so integrated absorbed power can exceed incident power in optically thick cases.")
        if self.physics["refractive_index_model"] == "thermal_3d" and not thermal_stats["enabled"]:
            warnings.append("Thermal refractive-index coupling is selected but the thermal solver is off; delta-n is zero.")
        seed_m2 = max(
            float(self.seed_cfg.get("m2_x", 1.0)),
            float(self.seed_cfg.get("m2_y", 1.0)),
        )
        seed_phase_screen_configured = any(
            element.get("type") == "phase_screen"
            for element in (
                self.config.get("optical_train", {}).get("seed", [])
                + self.config.get("optical_train", {}).get("between_pass", [])
            )
        )
        if (
            seed_m2 > 1.0 + 1e-12
            and self.physics["optical_propagation"] in {"fft", "hybrid_abcd_fft"}
            and (
                self.physics["refractive_index_model"] == "thermal_3d"
                or self.physics.get("nonlinear_phase_model", "off") != "off"
                or seed_phase_screen_configured
            )
        ):
            warnings.append(
                "Seed M2>1 is propagated as an effective diffraction factor while thermal/Kerr/user phase screens use physical phase in radians. "
                "This is an engineering surrogate, not a self-consistent modal complex field; use M2=1 or an explicit modal model for quantitative phase-screen coupling."
            )
        pump_m2 = max(
            float(self.pump_cfg.get("m2_x", 1.0)),
            float(self.pump_cfg.get("m2_y", 1.0)),
        )
        if (
            pump_m2 > 1.0 + 1e-12
            and self.physics["pump_propagation"] in {"fft", "hybrid_abcd_fft"}
            and self.physics["refractive_index_model"] == "thermal_3d"
        ):
            warnings.append(
                "Pump M2>1 FFT propagation combines an effective diffraction factor with a physical-wavelength thermal phase screen. "
                "This surrogate preserves first-order M2 geometry but is not a resolved incoherent modal ensemble; use stochastic rays or an explicit modal model for quantitative thermal deflection."
            )
        heat_fraction = self.material["thermal"].get("heat_fraction", "quantum_defect")
        (
            resolved_active_heat_fraction,
            heat_reference_wavelength,
            heat_reference_source,
        ) = self._resolved_active_heat_fraction()
        pump_quantum_efficiency = float(
            self.levels.pump.quantum_efficiency
        )
        if (
            thermal_stats["enabled"]
            and self.material["thermal"].get("heat_source_model", "pump_component_ledger")
            == "pump_component_ledger"
            and heat_fraction
            in {
                "quantum_defect",
                "lifetime_quantum_defect",
                "scaled_quantum_defect",
                "spectral_quantum_defect",
                "spectral_scaled_quantum_defect",
            }
            and pump_result.stats["absorbed_power_W"] > np.finfo(float).tiny
            and resolved_active_heat_fraction <= 0.0
        ):
            if heat_fraction in {
                "quantum_defect",
                "lifetime_quantum_defect",
                "spectral_quantum_defect",
            }:
                warnings.append(
                    "Quantum-defect heating is zero after applying pump quantum efficiency; check pump and effective fluorescence wavelengths "
                    "or provide an explicit thermal heat_fraction."
                )
            else:
                warnings.append(
                    "Scaled quantum-defect Pump heating resolved to zero; check the Pump/reference wavelengths and multiplier "
                    "or provide an explicit thermal heat_fraction."
                )
        if (
            heat_fraction
            in {
                "quantum_defect",
                "lifetime_quantum_defect",
                "scaled_quantum_defect",
            }
            and len(self.pump_solver.spectrum.coordinate) > 1
            and self.material["thermal"].get("heat_source_model", "pump_component_ledger")
            == "pump_component_ledger"
            and not self._last_spectral_heat_stats.get("enabled", False)
        ):
            warnings.append(
                "The Pump absorption solver integrates the configured spectrum, but the active heat fraction currently uses the Pump "
                "centre wavelength. Use a narrow Pump spectrum or an explicit measured heat fraction for quantitative broadband heating."
            )
        thermo_optic_diagnostics = self._thermo_optic_diagnostics(
            delta_n,
            gain,
            signal_result,
            coupling_stats,
        )
        warnings.extend(
            thermo_optic_diagnostics.get("field_diagnostics", {}).get("warnings", [])
        )
        optical_trace = OpticalTrace()
        optical_trace.extend(pump_result.trace, "pump_")
        optical_trace.extend(signal_result.trace, "seed_")
        gain_path_samples = self._gain_path_moment_samples(
            pump_result,
            signal_result,
        )
        propagation_by_beam = {
            "pump": str(self.physics.get("pump_propagation", "off")),
            "seed": str(self.physics.get("optical_propagation", "off")),
        }
        periodic_fft_modes_by_beam = {
            "pump": {"fft", "hybrid_abcd_fft"},
            # Seed "abcd" is analytic only outside the crystal; the gain
            # medium itself still uses sampled split-step FFT propagation.
            "seed": {"abcd", "fft", "hybrid_abcd_fft"},
        }
        executed_coherent_fidelity = str(
            (signal_result.cpa_diagnostics or {}).get("fidelity_mode", "")
        )
        axisymmetric_seed_transport = executed_coherent_fidelity in {
            "axisymmetric_kerr",
            "axisymmetric_transient_gain",
        }
        periodic_fft_applies = {
            "pump": propagation_by_beam["pump"]
            in periodic_fft_modes_by_beam["pump"],
            "seed": (
                self.signal_model != "commercial_app_fast"
                and not axisymmetric_seed_transport
                and propagation_by_beam["seed"]
                in periodic_fft_modes_by_beam["seed"]
            ),
        }
        native_optical_boundary_diagnostics = {
            "pump": pump_result.stats.get(
                "optical_grid_boundary_diagnostics"
            ),
            "seed": (signal_result.grid_coupling or {}).get(
                "optical_grid_boundary_diagnostics"
            ),
        }
        boundary_diagnostics = self._gain_path_boundary_diagnostics(
            gain_path_samples,
            periodic_fft_applies=periodic_fft_applies,
            native_optical_diagnostics=native_optical_boundary_diagnostics,
        )
        recorded_plane_boundary_diagnostics = (
            self._recorded_plane_boundary_diagnostics(optical_trace.summaries)
        )
        for beam_kind, diagnostic in boundary_diagnostics["beams"].items():
            if (
                diagnostic.get("at_risk", False)
                and diagnostic.get("periodic_fft_applies", False)
            ):
                warnings.append(
                    f"{beam_kind.capitalize()} sampled field has "
                    f"{float(diagnostic['maximum_boundary_energy_fraction']):.3%} "
                    "of its energy in the outer 5% of the physical optical "
                    "window width on each side; enlarge the optical window "
                    "to avoid periodic FFT wrap-around."
                )
        for beam_kind, diagnostic in recorded_plane_boundary_diagnostics[
            "beams"
        ].items():
            if diagnostic.get("at_risk", False):
                warnings.append(
                    f"Recorded {beam_kind.capitalize()} plane "
                    f"{diagnostic['plane']!r} has "
                    f"{float(diagnostic['maximum_boundary_energy_fraction']):.3%} "
                    "of its sampled energy in the outer 5% of that plane's "
                    "physical window width on each side; enlarge the relevant "
                    "Pump or Seed diagnostic window before interpreting its "
                    "profile or sampled M2."
                )
        seed_plane_summaries = [
            plane
            for plane in optical_trace.summaries
            if str(plane.get("name", "")).startswith("seed_")
        ]
        output_beam_quality = next(
            (
                plane
                for plane in reversed(seed_plane_summaries)
                if "system_output" in str(plane.get("name", ""))
            ),
            seed_plane_summaries[-1] if seed_plane_summaries else {},
        )
        raw_output_m2_x = output_beam_quality.get("M2_x_sampled")
        raw_output_m2_y = output_beam_quality.get("M2_y_sampled")
        transport_m2_x = float(self.seed_cfg.get("m2_x", 1.0))
        transport_m2_y = float(self.seed_cfg.get("m2_y", 1.0))
        constrained_gaussian_output = bool(
            signal_result.gain_transport
            and signal_result.gain_transport.get("backend")
            == "gaussian_mode_overlap"
            and not signal_result.gain_transport.get(
                "downstream_spatial_reshaping", False
            )
        )
        commercial_app_beam_quality_output = bool(
            signal_result.gain_transport
            and signal_result.gain_transport.get("backend")
            == "commercial_app_fast_gaussian_duct"
            and signal_result.gain_transport.get("output_effective_M2_x")
            is not None
            and signal_result.gain_transport.get("output_effective_M2_y")
            is not None
        )
        commercial_app_output_m2_x = (
            None
            if not commercial_app_beam_quality_output
            else float(
                signal_result.gain_transport["output_effective_M2_x"]
            )
        )
        commercial_app_output_m2_y = (
            None
            if not commercial_app_beam_quality_output
            else float(
                signal_result.gain_transport["output_effective_M2_y"]
            )
        )
        commercial_app_quality_closure = (
            None
            if not commercial_app_beam_quality_output
            else str(
                signal_result.gain_transport.get(
                    "beam_quality_closure", "configured_m2"
                )
            )
        )
        explicit_incoherent_modal_output = bool(
            signal_result.gain_transport
            and signal_result.gain_transport.get("incoherent_spatial_modes", False)
        )
        if commercial_app_beam_quality_output:
            output_m2_quantity_kind = (
                "commercial_app_projection_residual_effective_m2_proxy"
                if commercial_app_quality_closure == "projection_residual"
                else "commercial_app_configured_transport_m2_contract"
            )
        elif constrained_gaussian_output:
            output_m2_quantity_kind = "configured_gaussian_modal_m2_contract"
        elif explicit_incoherent_modal_output:
            output_m2_quantity_kind = "sampled_incoherent_second_moment_m2"
        elif not (
            math.isclose(transport_m2_x, 1.0, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(
                transport_m2_y, 1.0, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            output_m2_quantity_kind = (
                "sampled_distortion_times_effective_transport_m2_proxy"
            )
        else:
            output_m2_quantity_kind = "sampled_complex_field_phase_space_m2"
        input_energy = signal_result.source_grid_energy_J
        output_energy = signal_result.system_output_energy_J
        seed_boundary = signal_result.source_spatio_spectral or {}
        model_used = signal_result.model_used or self.signal_model
        if model_used == "spectral_time_causal_multipass":
            output_m2_quantity_kind = "unavailable_incoherent_channels_not_retained"
        backend_specialization = (
            model_used
            if model_used == "incoherent_hg_shared_reservoir"
            else None
        )
        executed_model = (
            self.signal_model
            if backend_specialization is not None
            else model_used
        )
        fallback_active = bool(str(model_used).endswith("_fallback"))
        output_beam_quality_valid, output_beam_quality_invalid_reasons = (
            _output_beam_quality_validity(
                sampled_valid=bool(
                    output_beam_quality.get("beam_quality_valid", False)
                ),
                forced_model_valid=bool(
                    constrained_gaussian_output or commercial_app_beam_quality_output
                ),
                model_used=model_used,
                cpa_diagnostics=signal_result.cpa_diagnostics,
                strict_config=self.sim_cfg.get("strict_validation", {}),
            )
        )
        configured_coherent_phase_sources = (
            _phase_insensitive_dispersion_sources(self.config)
            if model_used != "coherent_cpa"
            else []
        )
        commercial_app_temporal_closure = (
            (signal_result.gain_transport or {}).get("temporal_closure")
            if model_used == "commercial_app_fast"
            else None
        )
        commercial_app_one_dimensional_coherent_envelope = bool(
            commercial_app_temporal_closure
            in {
                "one_dimensional_effective_area_spm",
                "one_dimensional_effective_area_strong_field",
            }
        )
        # Commercial App effective-area mode is spatially reduced, but it is not
        # phase-insensitive: its one-dimensional complex envelope applies the
        # configured source/stretcher/material/train/compressor phase.  Keep
        # the strict phase-insensitive failure for fixed-duration/CW Commercial App and
        # the genuinely intensity-only backends only.
        phase_insensitive_phase_sources = (
            []
            if commercial_app_one_dimensional_coherent_envelope
            else configured_coherent_phase_sources
        )
        phase_insensitive_phase_requested = bool(
            phase_insensitive_phase_sources
        )
        if configured_coherent_phase_sources:
            if model_used == "commercial_app_fast":
                if commercial_app_one_dimensional_coherent_envelope:
                    warnings.append(
                        "Higher-order spectral phase was configured at "
                        + ", ".join(configured_coherent_phase_sources)
                        + "; Commercial App carries it in a one-dimensional effective-area "
                        "SPM/dispersion envelope and reports the output/compressed "
                        "pulse, but does not resolve spatially varying spectral phase. "
                        "Use coherent_cpa for space-time coupled compressor predictions."
                    )
                elif commercial_app_temporal_closure == "cw_average_intensity":
                    warnings.append(
                        "Higher-order spectral phase was configured at "
                        + ", ".join(configured_coherent_phase_sources)
                        + "; Commercial App CW uses average intensity and does not construct "
                        "a coherent pulse, so pulse broadening and compressor "
                        "diagnostics are not defined."
                    )
                else:
                    warnings.append(
                        "Higher-order spectral phase was configured at "
                        + ", ".join(configured_coherent_phase_sources)
                        + "; Commercial App fixed-duration does not propagate these phase terms. "
                        "Only phase already included in the constructed input pulse can affect its one fixed "
                        "effective duration; material/train/compressor phase does not produce output reshaping. "
                        "Use coherent_cpa for output duration, SPM spectrum, or compressed peak power."
                    )
            else:
                warnings.append(
                    "Higher-order spectral phase was configured at "
                    + ", ".join(configured_coherent_phase_sources)
                    + " while the selected Seed backend is phase-insensitive; "
                    "spectral labels remain available, but coherent temporal broadening/compression is not resolved. "
                    "Use coherent_cpa for pulse-duration, peak-power, or compressor predictions."
                )
        extracted_energy = signal_result.extracted_energy_J_m3
        if extracted_energy is None:
            extracted_energy = np.zeros(self.grid.shape, dtype=float)
        net_extracted_energy = float(self.grid.integrate_volume(extracted_energy))
        positive_extracted_energy = float(self.grid.integrate_volume(np.maximum(extracted_energy, 0.0)))
        absorbed_signal_energy = float(-self.grid.integrate_volume(np.minimum(extracted_energy, 0.0)))
        nonlinear_loss_energy = (
            np.zeros(self.grid.shape, dtype=float)
            if signal_result.nonlinear_loss_energy_J_m3 is None
            else np.asarray(signal_result.nonlinear_loss_energy_J_m3, dtype=float)
        )
        nonlinear_loss_total = float(
            self.grid.integrate_volume(nonlinear_loss_energy)
        )
        passive_loss_energy = (
            np.zeros(self.grid.shape, dtype=float)
            if signal_result.passive_loss_energy_J_m3 is None
            else np.asarray(signal_result.passive_loss_energy_J_m3, dtype=float)
        )
        passive_loss_total = float(self.grid.integrate_volume(passive_loss_energy))
        repetition_rate = float(self.sim_cfg.get("repetition_rate_Hz", 0.0))
        event_energy_summary = dict(self._last_event_energy_stats)
        event_energy_summary["ase_source_interface"] = (
            ase_source_interface_diagnostics(
                event_energy_summary,
                (
                    self._last_ase_result.stats
                    if self._last_ase_result is not None
                    else {"enabled": False}
                ),
            )
        )
        summary = {
            "name": self.sim_cfg.get("name", "amplifier"),
            "model": self.config["energy_levels"]["model"],
            "level_names": self.levels.names,
            "grid_shape_nz_ny_nx": list(self.grid.shape),
            "grid_coupling": self._grid_coupling_summary(),
            "performance": compute_backend_info(),
            "result_recording": copy.deepcopy(self.result_recording),
            "feature_status": {
                "single_transition_core": "stable",
                "multi_optical_transition": (
                    "beta_strict_audited"
                    if len(self.levels.pump_transitions) > 1
                    or len(self.levels.signal_transitions) > 1
                    else "not_enabled"
                ),
                "inhomogeneous_ensemble": (
                    "beta_strict_audited"
                    if self.levels.ensemble_contract.get("enabled", False)
                    else "not_enabled"
                ),
                "event_energy_ledger": (
                    "beta_strict_audited"
                    if bool(
                        self.config["energy_levels"]
                        .get("event_energy_ledger", {})
                        .get("enabled", False)
                    )
                    else "not_enabled"
                ),
                "multi_entity_thermal_conduction": (
                    "beta_grid_convergence_required"
                    if bool(
                        self.material.get("thermal", {}).get(
                            "entities", []
                        )
                    )
                    else "not_enabled"
                ),
                "excluded_scope": [
                    "CAD contact geometry",
                    "coolant CFD",
                    "mount/foil/solder contact mechanics",
                    "preload, friction, debonding, and pressure-dependent contact resistance",
                ],
            },
            "crystal_geometry": crystal_geometry_summary(self.grid, self.material),
            "doping_profile": self.doping_profile_summary,
            "spectroscopic_response": {
                **self.levels.response.summary(),
                "static_cross_section_cache": (
                    self.levels.static_cross_section_cache_diagnostics()
                ),
            },
            "physics_validity": validity,
            "ase": (
                {
                    **self._last_ase_result.stats,
                    **self._ase_execution,
                }
                if self._last_ase_result is not None
                else {
                    "enabled": False,
                    "model": str(
                        self.config.get("ase", {}).get("model", "off")
                    ),
                    **self._ase_execution,
                }
            ),
            "material_library": self.material.get("library", {}),
            "configuration_provenance": self._configuration_provenance(),
            "save_vtk": bool(self.sim_cfg.get("save_vtk", True)),
            "physics": self.physics,
            "multiphysics": coupling_stats,
            "pump": pump_result.stats,
            "signal": {
                "configured_source_energy_J": float(self.seed_cfg["pulse_energy_J"]),
                "grid_source_energy_J": input_energy,
                "grid_source_energy_semantics": (
                    "legacy_seed_ledger_input_before_named_external_losses"
                ),
                "analytic_gain_entrance_energy_J": seed_boundary.get(
                    "analytic_gain_entrance_energy_J"
                ),
                "sampled_gain_entrance_energy_J": seed_boundary.get(
                    "sampled_gain_entrance_energy_J"
                ),
                "gain_entrance_capture_fraction": seed_boundary.get(
                    "gain_entrance_capture_fraction"
                ),
                "minimum_wavelength_gain_entrance_capture_fraction": (
                    seed_boundary.get(
                        "minimum_wavelength_gain_entrance_capture_fraction"
                    )
                ),
                "numerical_window_truncation_J": seed_boundary.get(
                    "numerical_window_truncation_J"
                ),
                "physical_external_loss_J": seed_boundary.get(
                    "physical_external_loss_J"
                ),
                "gain_entrance_loss_partition_identity": seed_boundary.get(
                    "loss_partition_identity"
                ),
                "system_output_energy_J": output_energy,
                "configured_source_average_power_W": (
                    float(self.seed_cfg["pulse_energy_J"]) * repetition_rate if repetition_rate > 0.0 else None
                ),
                "grid_source_average_power_W": input_energy * repetition_rate if repetition_rate > 0.0 else None,
                "system_output_average_power_W": (
                    output_energy * repetition_rate if repetition_rate > 0.0 else None
                ),
                "energy_gain_source_to_output": output_energy / max(input_energy, np.finfo(float).tiny),
                "passes": signal_result.passes,
                "propagation": self.physics["optical_propagation"],
                "model": model_used,
                "requested_model": self.signal_model,
                "executed_model": executed_model,
                "backend_specialization": backend_specialization,
                "fallback_active": fallback_active,
                "phase_insensitive_spectral_phase_requested": (
                    phase_insensitive_phase_requested
                ),
                "phase_insensitive_spectral_phase_sources": (
                    phase_insensitive_phase_sources
                ),
                "configured_coherent_spectral_phase_sources": (
                    configured_coherent_phase_sources
                ),
                "spectral_phase_resolution": (
                    "full_space_time"
                    if model_used == "coherent_cpa"
                    else "one_dimensional_effective_area"
                    if commercial_app_one_dimensional_coherent_envelope
                    else "none"
                ),
                "temporal_type": str(
                    self.seed_cfg.get("temporal", {}).get("type", "gaussian")
                ),
                "gain_source": (
                    "configured_saturable_override_reservoir"
                    if self.config["energy_levels"].get("gain_override_m_inv")
                    is not None
                    else "physical_level_population"
                ),
                "intrapulse_relaxation_enabled": bool(
                    (signal_result.gain_transport or {}).get(
                        "include_relaxation_during_pulse", False
                    )
                    or (
                        (signal_result.cpa_diagnostics or {}).get(
                            "transient_gain", {}
                        )
                        or {}
                    ).get("include_relaxation_during_pulse", False)
                ),
                "between_pass_population_relaxation": any(
                    bool(item.get("between_pass_population_relaxation", False))
                    for item in signal_result.passes
                ),
                "gain_saturation": bool(self.physics["gain_saturation"]),
                "spectral_samples": 0 if signal_result.wavelengths_m is None else len(signal_result.wavelengths_m),
                "temporal_samples": 0 if signal_result.time_s is None else len(signal_result.time_s),
                "cpa": signal_result.cpa_diagnostics,
                "gain_transport": signal_result.gain_transport,
                "grid_coupling": signal_result.grid_coupling,
                "spatial_modes": spatial_mode_summary(self.seed_cfg),
                "source_spatio_spectral": signal_result.source_spatio_spectral,
                "polarization": signal_result.polarization_diagnostics
                or {
                    "model": "scalar",
                    "birefringence_model": "off",
                },
                "output_beam_quality": {
                    "valid": output_beam_quality_valid,
                    "invalid_reasons": output_beam_quality_invalid_reasons,
                    "reported_quantity_kind": output_m2_quantity_kind,
                    "method": (
                        "unavailable_incoherent_channels_not_retained"
                        if model_used == "spectral_time_causal_multipass"
                        else (
                            (
                                (
                                    "commercial_app_gaussian_q_with_projection_residual_second_moment_proxy"
                                    if commercial_app_quality_closure == "projection_residual"
                                    else "commercial_app_gaussian_q_configured_m2_contract"
                                )
                                if commercial_app_beam_quality_output
                                else "constrained_gaussian_mode_with_sampled_complex_field_audit"
                            )
                            if constrained_gaussian_output or commercial_app_beam_quality_output
                            else (
                                "sampled_incoherent_hg_channel_second_moments"
                                if explicit_incoherent_modal_output
                                else "sampled_complex_field_with_effective_transport_scaling"
                            )
                        )
                    ),
                    "raw_sampled_M2_x": raw_output_m2_x,
                    "raw_sampled_M2_y": raw_output_m2_y,
                    "input_transport_M2_x": transport_m2_x,
                    "input_transport_M2_y": transport_m2_y,
                    "M2_x": (
                        commercial_app_output_m2_x
                        if commercial_app_beam_quality_output
                        else transport_m2_x
                        if constrained_gaussian_output
                        else (
                            None
                            if raw_output_m2_x is None
                            else float(raw_output_m2_x)
                            * (1.0 if explicit_incoherent_modal_output else transport_m2_x)
                        )
                    ),
                    "M2_y": (
                        commercial_app_output_m2_y
                        if commercial_app_beam_quality_output
                        else transport_m2_y
                        if constrained_gaussian_output
                        else (
                            None
                            if raw_output_m2_y is None
                            else float(raw_output_m2_y)
                            * (1.0 if explicit_incoherent_modal_output else transport_m2_y)
                        )
                    ),
                    "plane": output_beam_quality.get("name"),
                    "scope": (
                        (
                            signal_result.gain_transport.get(
                                "beam_quality_scope",
                                "Commercial App Gaussian-q beam-quality proxy.",
                            )
                            if commercial_app_beam_quality_output
                            else (
                                "The Gaussian-overlap backend projects onto a moment-matched Gaussian after each "
                                "crystal cell, so reported M2 is the configured modal contract. raw_sampled_M2_* "
                                "remains an audit of finite-window phase sampling and is not substituted for that contract."
                            )
                        )
                        if constrained_gaussian_output or commercial_app_beam_quality_output
                        else (
                            "Independently propagated HG channels are treated as mutually incoherent in the "
                            "sampled second moments, so no additional effective-M2 multiplier is applied."
                            if explicit_incoherent_modal_output
                            else (
                                "For configured M2=1 this is the ISO-style coherent complex-field second moment. "
                                "For effective/incoherent M2 sources, the sampled distortion factor is multiplied "
                                "by the configured transport M2 and remains an effective scalar diagnostic."
                            )
                        )
                    ),
                },
                "caustic_diagnostics": signal_result.caustic_diagnostics
                or {
                    "enabled": False,
                    "application": "diagnostic_only",
                    "affects_propagation": False,
                },
                "nonlinear_loss_energy_J": nonlinear_loss_total,
                "passive_loss_energy_J": passive_loss_total,
                "crystal_aperture_loss_energy_J": float(
                    signal_result.crystal_aperture_loss_energy_J
                ),
                "external_loss_ledger": list(
                    signal_result.external_loss_ledger or []
                ),
                "volume_recording": self.sim_cfg.get(
                    "signal_volume_recording", "last_and_cumulative"
                ),
            },
            "optical_train": {
                "plane_count": len(optical_trace.summaries),
                "planes": optical_trace.summaries,
                "gain_path_sample_count": len(gain_path_samples),
                "gain_path_samples": gain_path_samples,
                "boundary_window_diagnostics": boundary_diagnostics,
                "recorded_plane_boundary_diagnostics": (
                    recorded_plane_boundary_diagnostics
                ),
                "note": (
                    "ABCD is Gaussian-only; FFT is scalar sequential physical optics, not a non-sequential CAD ray tracer. "
                    "Gain-path D4-sigma samples are transverse moments of each stored optical z bin. "
                    "Pump intensity is a cell average located at the bin centre; Seed fluence is the "
                    "post-split-step field located at the bin exit. Pump is always available and Seed "
                    "covers every stored pass or otherwise the last pass."
                ),
                "multipass": copy.deepcopy(
                    self.sim_cfg.get("multipass", {"enabled": False})
                ),
            },
            "population": {
                **population_stats,
                "initial_fraction_contract": dict(
                    self.levels.initial_population_diagnostics
                ),
                "decay_contract": dict(
                    self.levels.decay_contract_diagnostics
                ),
                "nonlinear_solver": dict(self.levels.nonlinear_diagnostics),
                "signal_event_limiter": dict(
                    self.levels.signal_event_diagnostics
                ),
                "numerical_population_correction": dict(
                    self.levels.population_correction_diagnostics
                ),
                "steady_state_solver": dict(
                    self.levels.steady_state_diagnostics
                ),
                "implicit_linear_solver": dict(
                    self.levels.implicit_linear_solver_diagnostics
                ),
                "ensemble_inventory": ensemble_inventory,
                "signal_transition_inversions": transition_inversions,
                "primary_signal_transition": (
                    self.levels.signal.name
                ),
                "primary_signal_upper_level": self.levels.names[
                    self.levels.signal.upper
                ],
                "upper_level": self.levels.names[self.levels.signal.upper],
                "upper_fraction_min": float(
                    np.min(population[self.levels.signal.upper][active] / density[active])
                ),
                "upper_fraction_max": float(
                    np.max(population[self.levels.signal.upper][active] / density[active])
                ),
                "conservation_max_abs_m3": population_error,
                "conservation_max_relative": population_error_relative,
                "after_pump_upper_fraction_min": float(
                    np.min(population_after_pump[self.levels.signal.upper][active] / density[active])
                ),
                "after_pump_upper_fraction_max": float(
                    np.max(population_after_pump[self.levels.signal.upper][active] / density[active])
                ),
                "after_pump_inversion_min_m3": float(np.min(inversion_after_pump[active])),
                "after_pump_inversion_max_m3": float(np.max(inversion_after_pump[active])),
                "primary_transition_inversion_after_pump_min_m3": float(
                    np.min(inversion_after_pump[active])
                ),
                "primary_transition_inversion_after_pump_max_m3": float(
                    np.max(inversion_after_pump[active])
                ),
                "compatibility_aliases": {
                    "upper_level": "primary_signal_upper_level",
                    "after_pump_inversion_min_m3": (
                        "primary_transition_inversion_after_pump_min_m3"
                    ),
                    "after_pump_inversion_max_m3": (
                        "primary_transition_inversion_after_pump_max_m3"
                    ),
                },
            },
            "extraction": {
                "definition": "signed active-transition energy transfer; positive is extraction and passive loss is excluded",
                "net_J": net_extracted_energy,
                "positive_J": positive_extracted_energy,
                "active_absorption_J": absorbed_signal_energy,
                "minimum_J_m3": float(np.min(extracted_energy)),
                "maximum_J_m3": float(np.max(extracted_energy)),
            },
            "gain": {
                # The computational box can contain air cells around a
                # cylindrical crystal.  Those zeros are not material gain and
                # must not dilute or clip the reported crystal statistics.
                "min_m_inv": float(np.min(gain[self.crystal_mask])),
                "max_m_inv": float(np.max(gain[self.crystal_mask])),
                "mean_m_inv": float(np.average(gain[self.crystal_mask],
                                              weights=crystal_cell_volumes)),
                "mean_basis": "crystal_volume_weighted",
                "field_contract": {
                    "gain_m_inv": "compatibility alias for final net_gain_m_inv",
                    "initial_active_gain_m_inv": "active gain immediately before the reported Seed propagation",
                    "remaining_active_gain_m_inv": "active population/override reservoir after the reported Seed propagation",
                    "passive_loss_m_inv": "configured passive Seed loss coefficient",
                    "net_gain_m_inv": "remaining active gain minus passive loss",
                },
                "initial_active_min_m_inv": float(
                    np.min(initial_active_gain[self.crystal_mask])
                ),
                "initial_active_max_m_inv": float(
                    np.max(initial_active_gain[self.crystal_mask])
                ),
                "remaining_active_min_m_inv": float(
                    np.min(remaining_active_gain[self.crystal_mask])
                ),
                "remaining_active_max_m_inv": float(
                    np.max(remaining_active_gain[self.crystal_mask])
                ),
                "passive_loss_m_inv": float(
                    self.material.get("signal_passive_loss_m_inv", 0.0)
                ),
                "gain_source": (
                    "configured_saturable_override_reservoir"
                    if self.config["energy_levels"].get("gain_override_m_inv")
                    is not None
                    else "physical_level_population"
                ),
                "frequency_resolved_net_gain": frequency_gain_summary,
                "frequency_resolved_sample_count": len(
                    frequency_gain_summary
                ),
                "frequency_resolved_sampling": (
                    "all propagated wavelengths"
                    if signal_result.wavelengths_m is None
                    or len(signal_result.wavelengths_m) <= 129
                    else "129 uniformly indexed propagated wavelength samples"
                ),
            },
            "thermal": {
                **thermal_stats,
                # Report the backend that actually ran.  ``configured`` is an
                # internal selector meaning "follow material.thermal" and is
                # not itself a thermal model; exposing it in a result snapshot
                # made a cold run look thermally ambiguous.
                "model": self._executed_thermal_model(thermal_stats),
                "configured_model": self.physics["thermal_model"],
                "heat_source_model": self.material["thermal"].get(
                    "heat_source_model", "pump_component_ledger"
                ),
                "configured_active_heat_fraction": heat_fraction,
                "active_heat_fraction_model": (
                    heat_fraction if isinstance(heat_fraction, str) else "explicit_fraction"
                ),
                "resolved_active_heat_fraction": resolved_active_heat_fraction,
                "extra_heat_fraction": float(
                    self.material["thermal"].get("extra_heat_fraction", 0.0)
                ),
                "stokes_shift_multiplier": float(
                    self.material["thermal"].get("stokes_shift_multiplier", 1.0)
                ),
                "pump_spectral_heat": self._last_spectral_heat_stats,
                "pump_component_ledger_applicability": (
                    self._last_pump_component_applicability
                ),
                "event_energy_ledger": event_energy_summary,
                "lifetime_nonradiative_heat": (
                    self._last_lifetime_heat_stats
                ),
                "heat_reference_wavelength_m": heat_reference_wavelength,
                "heat_reference_source": heat_reference_source,
                "pump_quantum_efficiency": pump_quantum_efficiency,
                "pump_transition_quantum_efficiencies": {
                    transition.name: float(
                        transition.quantum_efficiency
                    )
                    for transition in self.levels.pump_transitions
                },
                "pump_active_absorbed_power_W": pump_result.stats.get(
                    "active_absorbed_power_W"
                ),
                "pump_passive_absorbed_power_W": pump_result.stats.get(
                    "passive_absorbed_power_W"
                ),
                "ase_passive_heat_W": (
                    0.0
                    if self._last_ase_result is None
                    else float(
                        self.grid.integrate_volume(
                            self._last_ase_result.passive_heat_W_m3
                        )
                    )
                ),
                "seed_irreversible_heat_coupling": seed_heat_mode,
                # Backward-compatible alias; this coupling also owns optional
                # passive Seed absorption and is therefore not only nonlinear.
                "seed_nonlinear_heat_coupling": seed_heat_mode,
                "seed_nonlinear_loss_power_W": (
                    nonlinear_loss_total * repetition_rate
                    if repetition_rate > 0.0
                    else None
                ),
                "seed_passive_absorption_power_W": (
                    passive_loss_total * repetition_rate
                    if repetition_rate > 0.0
                    else None
                ),
                "reference_temperature_K": reference,
                "heat_total_W": float(self.grid.integrate_volume(heat)),
                "temperature_min_K": float(np.min(temperature)),
                "temperature_max_K": float(np.max(temperature)),
                **thermo_optic_diagnostics,
            },
            "stress": {
                "model": self.material["mechanical"].get("stress_model", "none"),
                "approximation": stress_solver.approximation,
                "von_mises_max_Pa": float(np.max(stress["von_mises_Pa"])),
                "equilibrium_residual_relative": (
                    None
                    if "fem_equilibrium_residual_relative" not in stress
                    else float(stress["fem_equilibrium_residual_relative"])
                ),
                "active_elements": (
                    None
                    if "fem_active_elements" not in stress
                    else int(stress["fem_active_elements"])
                ),
                "active_nodes": (
                    None
                    if "fem_active_nodes" not in stress
                    else int(stress["fem_active_nodes"])
                ),
                "linear_iterations": (
                    None
                    if "fem_linear_iterations" not in stress
                    else int(stress["fem_linear_iterations"])
                ),
                "operator_assembly_reused": (
                    None
                    if "fem_operator_assembly_reused" not in stress
                    else bool(stress["fem_operator_assembly_reused"])
                ),
                "factorization_reused": (
                    None
                    if "fem_factorization_reused" not in stress
                    else bool(stress["fem_factorization_reused"])
                ),
                "preconditioner_reused": (
                    None
                    if "fem_preconditioner_reused" not in stress
                    else bool(stress["fem_preconditioner_reused"])
                ),
                "operator_assembly_time_s": (
                    None
                    if "fem_operator_assembly_time_s" not in stress
                    else float(stress["fem_operator_assembly_time_s"])
                ),
                "factorization_time_s": (
                    None
                    if "fem_factorization_time_s" not in stress
                    else float(stress["fem_factorization_time_s"])
                ),
                "linear_solve_time_s": (
                    None
                    if "fem_linear_solve_time_s" not in stress
                    else float(stress["fem_linear_solve_time_s"])
                ),
                "axial_displacement_peak_to_valley_m": (
                    None
                    if "axial_displacement_peak_to_valley_m" not in stress
                        else float(stress["axial_displacement_peak_to_valley_m"])
                ),
                "endface_sag_peak_to_valley_m": (
                    None
                    if "endface_sag_peak_to_valley_m" not in stress
                    else float(stress["endface_sag_peak_to_valley_m"])
                ),
                "endface_optical_path_peak_to_valley_m": (
                    None
                    if "endface_optical_path_difference_m" not in stress
                    else float(
                        np.ptp(
                            np.asarray(
                                stress["endface_optical_path_difference_m"],
                                dtype=float,
                            )
                        )
                    )
                ),
                "endface_deformation_model": self.material["mechanical"]
                .get("endface_deformation", {})
                .get("model", "off"),
                "photoelastic_birefringence_max": (
                    None
                    if "photoelastic_birefringence_magnitude" not in stress
                    else float(
                        np.max(stress["photoelastic_birefringence_magnitude"])
                    )
                ),
                "tensor_voigt_order": (
                    ", ".join(
                        self.material["mechanical"]
                        .get("anisotropic", {})
                        .get(
                            "voigt_order",
                            ["xx", "yy", "zz", "xy", "yz", "xz"],
                        )
                    )
                    if self.material["mechanical"].get("stress_model")
                    == "anisotropic_thermoelastic_fem"
                    else None
                ),
                "voigt_shear_convention": (
                    self.material["mechanical"]
                    .get("anisotropic", {})
                    .get(
                        "voigt_shear_convention",
                        "engineering_strain_tensor_stress",
                    )
                    if self.material["mechanical"].get("stress_model")
                    == "anisotropic_thermoelastic_fem"
                    else None
                ),
                "orientation_matrix_material_to_lab": (
                    None
                    if "photoelastic_orientation_matrix_material_to_lab" not in stress
                    else np.asarray(
                        stress["photoelastic_orientation_matrix_material_to_lab"],
                        dtype=float,
                    ).tolist()
                ),
                "boundary_condition": (
                    None
                    if self.material["mechanical"].get("stress_model", "none")
                    not in {"thermoelastic_fem", "anisotropic_thermoelastic_fem"}
                    else (
                        "fixed_faces"
                        if self.material["mechanical"]
                        .get("thermoelastic_fem", {})
                        .get("fixed_faces", [])
                        else "traction_free_with_minimal_rigid_body_stabilization"
                    )
                ),
                "photoelastic_coupling": (
                    self.material["mechanical"]
                    .get("thermoelastic_fem", {})
                    .get("photoelastic_coupling", "off")
                    if self.material["mechanical"].get("stress_model", "none")
                    in {"thermoelastic_fem", "anisotropic_thermoelastic_fem"}
                    else None
                ),
            },
            "assumptions": self._assumptions(
                model_used,
                signal_result.cpa_diagnostics,
                signal_result.gain_transport,
            ),
            "warnings": list(dict.fromkeys(warnings)),
        }
        try:
            level_energies, level_energy_source = self.levels.reference_level_energies_J(
                float(self.pump_cfg["wavelength_m"]),
                float(self.seed_cfg["wavelength_m"]),
            )
            stored_after_pump = float(
                self.grid.integrate_volume(
                    np.einsum("i,i...->...", level_energies, population_after_pump)
                )
            )
            stored_after_seed = float(
                self.grid.integrate_volume(
                    np.einsum("i,i...->...", level_energies, population)
                )
            )
            summary["stored_level_energy"] = {
                "available": True,
                "source": level_energy_source,
                "level_names": list(self.levels.names),
                "level_energies_J": level_energies.tolist(),
                "after_pump_J": stored_after_pump,
                "after_seed_J": stored_after_seed,
                "seed_step_change_J": stored_after_seed - stored_after_pump,
            }
        except ValueError as exc:
            summary["stored_level_energy"] = {
                "available": False,
                "reason": str(exc),
            }
        summary["energy_ledger"] = build_energy_ledger(summary)
        summary["strict_diagnostic_contract"] = (
            build_required_diagnostic_contract(summary)
        )
        validation = build_validation_report(
            summary,
            mode=str(self.sim_cfg.get("validation_mode", "exploratory")),
            settings=self.sim_cfg.get("strict_validation", {}),
        )
        summary["validation"] = validation
        if not validation["valid"]:
            validation_warning = (
                f"Result validity audit found {validation['failure_count']} failed "
                f"contract(s); mode={validation['mode']}. Inspect summary.validation "
                "before using this point quantitatively."
            )
            summary["warnings"] = list(
                dict.fromkeys([*summary["warnings"], validation_warning])
            )
        elif validation.get("warning_count", 0):
            advisory = (
                "Result validity audit passed all blocking contracts but found "
                f"{validation['warning_count']} diagnostic-quality warning(s). "
                "Inspect summary.validation.warnings before interpreting the "
                "affected diagnostic products."
            )
            summary["warnings"] = list(
                dict.fromkeys([*summary["warnings"], advisory])
            )
        planes = optical_trace.planes if self.physics["record_optical_planes"] else {}
        result = SimulationResult(
            grid=self.grid,
            population_m3=population,
            population_after_pump_m3=population_after_pump,
            inversion_after_pump_m3=inversion_after_pump,
            pump_intensity_W_m2=pump_result.intensity_W_m2,
            pump_absorbed_W_m3=pump_result.absorbed_W_m3,
            signal_fluence_J_m2=signal_result.fluence_J_m2,
            gain_m_inv=gain,
            initial_active_gain_m_inv=initial_active_gain,
            remaining_active_gain_m_inv=remaining_active_gain,
            passive_loss_m_inv=passive_loss,
            net_gain_m_inv=gain.copy(),
            heat_W_m3=heat,
            extracted_energy_J_m3=extracted_energy,
            temperature_K=temperature,
            delta_n=delta_n,
            stress=stress,
            optical_planes=planes,
            optical_plane_coordinates=(
                {
                    name: (
                        optical_trace.plane_coordinates[name][0].copy(),
                        optical_trace.plane_coordinates[name][1].copy(),
                    )
                    for name in planes
                    if name in optical_trace.plane_coordinates
                }
                if planes
                else {}
            ),
            level_names=self.levels.names,
            summary=summary,
            wavelengths_m=signal_result.wavelengths_m,
            input_spectrum_J=signal_result.input_spectrum_J,
            output_spectrum_J=signal_result.output_spectrum_J,
            time_s=signal_result.time_s,
            input_temporal_energy_J=signal_result.input_temporal_energy_J,
            output_temporal_energy_J=signal_result.output_temporal_energy_J,
            frequency_Hz=signal_result.frequency_Hz,
            temporal_bin_edges_s=signal_result.temporal_bin_edges_s,
            spectral_bin_edges_Hz=signal_result.spectral_bin_edges_Hz,
            input_spectral_phase_rad=signal_result.input_spectral_phase_rad,
            output_spectral_phase_rad=signal_result.output_spectral_phase_rad,
            compressed_spectral_phase_rad=signal_result.compressed_spectral_phase_rad,
            compressed_temporal_power_W=signal_result.compressed_temporal_power_W,
            cpa_diagnostics=signal_result.cpa_diagnostics,
            longitudinal_intensity_W_m2=signal_result.longitudinal_intensity_W_m2,
            longitudinal_intensity_area_m2=signal_result.longitudinal_intensity_area_m2,
            longitudinal_temporal_power_W=signal_result.longitudinal_temporal_power_W,
            crystal_mask=self.crystal_mask,
            optical_grid=self.optical_grid if self.grid_coupler is not None else None,
            pump_optical_grid=(
                self.pump_optical_grid
                if self.pump_grid_coupler is not None
                else None
            ),
            pump_active_absorbed_W_m3=pump_result.active_absorbed_W_m3,
            pump_passive_absorbed_W_m3=pump_result.passive_absorbed_W_m3,
            signal_fluence_last_pass_J_m2=signal_result.fluence_last_pass_J_m2,
            signal_fluence_cumulative_J_m2=signal_result.fluence_cumulative_J_m2,
            signal_fluence_per_pass_J_m2=signal_result.fluence_per_pass_J_m2,
            dopant_density_m3=density.copy(),
            nonlinear_loss_energy_J_m3=nonlinear_loss_energy,
            ase_intensity_W_m2=(
                None
                if self._last_ase_result is None
                else np.sum(self._last_ase_result.spectral_intensity_W_m2, axis=0)
            ),
            signal_passive_loss_energy_J_m3=passive_loss_energy,
            output_jones_field=signal_result.output_jones_field,
            pump_wavelengths_m=pump_result.wavelengths_m,
            pump_spectral_intensity_W_m2=pump_result.spectral_intensity_W_m2,
            pump_spectral_active_absorbed_W_m3=(
                pump_result.spectral_active_absorbed_W_m3
            ),
            pump_spectral_passive_absorbed_W_m3=(
                pump_result.spectral_passive_absorbed_W_m3
            ),
        )
        summary["performance"]["operator_profile"] = operation_profile_snapshot(
            total_run_elapsed_s=time.perf_counter() - self._progress_started_s
        )
        self._emit_progress(
            "complete",
            "Simulation and validity audit completed",
            fraction=1.0,
            metrics={
                "output_energy_J": float(signal_result.system_output_energy_J),
                "validation_valid": bool(validation.get("valid", False)),
                "warning_count": len(summary.get("warnings", [])),
            },
            state="complete",
        )
        return result

    def _gain_path_moment_samples(
        self,
        pump_result: PumpPropagationResult,
        signal_result: SignalPropagationResult,
    ) -> list[dict[str, Any]]:
        """Measure D4-sigma on every stored crystal z bin without saving fields."""

        samples: list[dict[str, Any]] = []
        pump_cosine = gain_medium_path_cosine(self.pump_gain_marker)
        samples.extend(
            self._volume_d4sigma_samples(
                pump_result.intensity_W_m2,
                beam_kind="pump",
                pass_index=1,
                direction=str(
                    pump_result.stats.get(
                        "direction",
                        self.pump_cfg.get("direction", "forward"),
                    )
                ),
                effective_path_length_m=float(
                    self.grid.z_edges[-1] - self.grid.z_edges[0]
                )
                / pump_cosine,
                quantity="intensity_W_m2",
                sample_offset_fraction=0.5,
                sample_location="optical_z_bin_cell_average",
            )
        )

        pass_rows = list(signal_result.passes)
        per_pass = signal_result.fluence_per_pass_J_m2
        if per_pass is not None and len(per_pass) == len(pass_rows):
            signal_volumes = [np.asarray(volume, dtype=float) for volume in per_pass]
            pass_indices = list(range(1, len(signal_volumes) + 1))
        else:
            last_volume = (
                signal_result.fluence_last_pass_J_m2
                if signal_result.fluence_last_pass_J_m2 is not None
                else signal_result.fluence_J_m2
            )
            signal_volumes = [np.asarray(last_volume, dtype=float)]
            pass_indices = [max(len(pass_rows), 1)]
        for volume, pass_index in zip(signal_volumes, pass_indices, strict=True):
            pass_summary = (
                pass_rows[pass_index - 1]
                if 0 <= pass_index - 1 < len(pass_rows)
                else {}
            )
            samples.extend(
                self._volume_d4sigma_samples(
                    volume,
                    beam_kind="seed",
                    pass_index=pass_index,
                    direction=str(
                        pass_summary.get(
                            "direction",
                            self.seed_cfg.get("direction", "forward"),
                        )
                    ),
                    effective_path_length_m=float(
                        pass_summary.get(
                            "effective_path_length_m",
                            self.grid.z_edges[-1] - self.grid.z_edges[0],
                        )
                    ),
                    quantity="fluence_J_m2",
                    sample_offset_fraction=1.0,
                    sample_location="optical_z_bin_exit_after_split_step",
                )
            )
        return samples

    def _volume_d4sigma_samples(
        self,
        values: np.ndarray,
        *,
        beam_kind: str,
        pass_index: int,
        direction: str,
        effective_path_length_m: float,
        quantity: str,
        sample_offset_fraction: float,
        sample_location: str,
    ) -> list[dict[str, Any]]:
        array = np.asarray(values, dtype=float)
        if array.shape != self.grid.shape:
            return []
        if direction not in {"forward", "backward"}:
            direction = "forward"
        indices = (
            list(range(len(self.grid.z)))
            if direction == "forward"
            else list(reversed(range(len(self.grid.z))))
        )
        step = float(effective_path_length_m) / max(len(indices), 1)
        offset = float(sample_offset_fraction)
        if not 0.0 <= offset <= 1.0:
            raise ValueError("sample_offset_fraction must be in [0, 1]")
        samples: list[dict[str, Any]] = []
        for path_index, iz in enumerate(indices):
            moments = transverse_intensity_moments(array[iz], self.grid)
            boundary = transverse_boundary_energy_fractions(array[iz], self.grid)
            samples.append(
                {
                    "beam_kind": beam_kind,
                    "pass_index": int(pass_index),
                    "direction": direction,
                    "z_bin_index": int(iz),
                    "gain_relative_path_m": (path_index + offset) * step,
                    "sample_location": sample_location,
                    quantity: moments["total"],
                    "centroid_x_m": moments["centroid_x_m"],
                    "centroid_y_m": moments["centroid_y_m"],
                    "radius_x_D4sigma_m": moments["radius_x_D4sigma_m"],
                    "radius_y_D4sigma_m": moments["radius_y_D4sigma_m"],
                    "covariance_xy_m2": moments["covariance_xy_m2"],
                    "radius_major_D4sigma_m": moments[
                        "radius_major_D4sigma_m"
                    ],
                    "radius_minor_D4sigma_m": moments[
                        "radius_minor_D4sigma_m"
                    ],
                    "major_axis_angle_deg": moments["major_axis_angle_deg"],
                    **boundary,
                }
            )
        return samples

    @staticmethod
    def _gain_path_boundary_diagnostics(
        samples: list[dict[str, Any]],
        *,
        warning_threshold: float = 1e-3,
        periodic_fft_applies: dict[str, bool] | None = None,
        native_optical_diagnostics: dict[str, dict[str, Any] | None]
        | None = None,
    ) -> dict[str, Any]:
        """Summarize numerical-window occupancy for sampled gain paths.

        A dual-grid run uses the native optical volume captured before
        conservative backprojection; the remapped physical volume is not an
        FFT-window diagnostic.  Boundary occupancy is invalidating only for a
        beam whose selected backend actually uses a periodic transverse FFT on
        the sampled path.  Other backends retain the measurement as a
        diagnostic without mislabelling a physical crystal edge as numerical
        wrap-around.
        """

        beams: dict[str, Any] = {}
        for beam_kind in ("pump", "seed"):
            rows = [
                row
                for row in samples
                if row.get("beam_kind") == beam_kind
                and np.isfinite(float(row.get("boundary_energy_fraction", 0.0)))
            ]
            native = (
                None
                if native_optical_diagnostics is None
                else native_optical_diagnostics.get(beam_kind)
            )
            native_fraction = (
                None
                if not isinstance(native, dict)
                else native.get("maximum_boundary_energy_fraction")
            )
            if native_fraction is not None and np.isfinite(
                float(native_fraction)
            ):
                maximum = native
                maximum_fraction = float(native_fraction)
                diagnostic_grid = str(
                    native.get("diagnostic_grid", "native_optical")
                )
                diagnostic_source = "pre_remap_native_optical_volume"
            else:
                if not rows:
                    continue
                maximum = max(
                    rows,
                    key=lambda row: float(
                        row.get("boundary_energy_fraction", 0.0)
                    ),
                )
                maximum_fraction = float(
                    maximum.get("boundary_energy_fraction", 0.0)
                )
                diagnostic_grid = "shared_physical_optical"
                diagnostic_source = "gain_path_volume"
            beams[beam_kind] = {
                "maximum_boundary_energy_fraction": maximum_fraction,
                "at_risk": maximum_fraction > warning_threshold,
                "periodic_fft_applies": (
                    True
                    if periodic_fft_applies is None
                    else bool(periodic_fft_applies.get(beam_kind, False))
                ),
                "pass_index": int(maximum.get("pass_index", 1)),
                "z_bin_index": int(maximum.get("z_bin_index", 0)),
                "sample_location": maximum.get("sample_location"),
                "diagnostic_grid": diagnostic_grid,
                "diagnostic_source": diagnostic_source,
                "grid_nx": maximum.get("grid_nx"),
                "grid_ny": maximum.get("grid_ny"),
                "grid_window_x_m": maximum.get("grid_window_x_m"),
                "grid_window_y_m": maximum.get("grid_window_y_m"),
            }
        return {
            "method": "backend_native_outer_physical_window_energy_fraction",
            "boundary_region_fraction": 0.05,
            "boundary_region_basis": "physical_window_width_each_side",
            "warning_threshold": float(warning_threshold),
            "beams": beams,
        }

    @staticmethod
    def _recorded_plane_boundary_diagnostics(
        summaries: list[dict[str, Any]],
        *,
        warning_threshold: float = 1e-2,
    ) -> dict[str, Any]:
        """Flag stored plane plots whose finite window limits shape or M2."""

        beams: dict[str, Any] = {}
        for beam_kind in ("pump", "seed"):
            prefix = f"{beam_kind}_"
            rows = [
                row
                for row in summaries
                if str(row.get("name", "")).startswith(prefix)
                and np.isfinite(
                    float(row.get("boundary_energy_fraction", 0.0))
                )
            ]
            if not rows:
                continue
            maximum = max(
                rows,
                key=lambda row: float(
                    row.get("boundary_energy_fraction", 0.0)
                ),
            )
            maximum_fraction = float(
                maximum.get("boundary_energy_fraction", 0.0)
            )
            beams[beam_kind] = {
                "maximum_boundary_energy_fraction": maximum_fraction,
                "at_risk": maximum_fraction > warning_threshold,
                "plane": maximum.get("name"),
                "grid_nx": maximum.get("grid_nx"),
                "grid_ny": maximum.get("grid_ny"),
                "grid_window_x_m": maximum.get("grid_window_x_m"),
                "grid_window_y_m": maximum.get("grid_window_y_m"),
            }
        return {
            "method": "outer_physical_window_energy_fraction",
            "boundary_region_fraction": 0.05,
            "boundary_region_basis": "physical_window_width_each_side",
            "warning_threshold": float(warning_threshold),
            "beams": beams,
        }

    def _operating_point(
        self,
        delta_n: np.ndarray,
        record_planes: bool,
        periodic_initial_population: np.ndarray | None = None,
        periodic_initialization_source: str = "thermal_outer_warm_start",
    ) -> tuple[np.ndarray, dict[str, Any], PumpPropagationResult]:
        """Solve Pump/population dynamics for one prescribed thermo-optic field."""

        self._emit_progress(
            "operating_point",
            "Preparing Pump and population operating point",
            fraction=0.0,
        )
        previous_suppression = self.physics.get("_suppress_output_diagnostics")
        self.physics["_suppress_output_diagnostics"] = True
        try:
            pulse_train = self._run_pulse_train_population(
                delta_n,
                record_planes,
                periodic_initial_population=periodic_initial_population,
                periodic_initialization_source=periodic_initialization_source,
            )
            if pulse_train is not None:
                self._emit_progress(
                    "operating_point",
                    "Pump and periodic population operating point completed",
                    fraction=1.0,
                    metrics={
                        "population_iterations": int(pulse_train[1].get("iterations", 0)),
                    },
                )
                return pulse_train
            self._emit_progress(
                "population",
                "Solving population state",
                fraction=0.08,
            )
            population, population_stats = self._prepare_population(delta_n)
            self._emit_progress(
                "pump_propagation",
                "Propagating Pump through the configured optical path",
                fraction=0.72,
            )
            pump_result = self.pump_solver.propagate(
                population,
                record_planes,
                delta_n=delta_n,
            )
            sampled_pump = self.run_context.fixed_point_history.pop(
                "transient_sampled_pump_result", None
            )
            if sampled_pump is not None:
                sampled_pump.trace = pump_result.trace
                sampled_pump.warnings = list(
                    dict.fromkeys(
                        [*sampled_pump.warnings, *pump_result.warnings]
                    )
                )
                pump_result = sampled_pump
            self._emit_progress(
                "operating_point",
                "Pump and population operating point completed",
                fraction=1.0,
                metrics={
                    "population_iterations": int(population_stats.get("iterations", 0)),
                    "absorbed_pump_power_W": float(
                        pump_result.stats.get("absorbed_power_W", 0.0)
                    ),
                },
            )
            return population, population_stats, pump_result
        finally:
            if previous_suppression is None:
                self.physics.pop("_suppress_output_diagnostics", None)
            else:
                self.physics["_suppress_output_diagnostics"] = previous_suppression

    def _population_residual_relative(
        self,
        current: np.ndarray,
        previous: np.ndarray,
    ) -> float:
        difference = np.max(
            np.abs(np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)),
            axis=0,
        )
        active = self.dopant_density_m3 > 0.0
        if not np.any(active):
            return 0.0
        return float(np.max(difference[active] / self.dopant_density_m3[active]))

    def _population_fractions(self, population: np.ndarray) -> np.ndarray:
        """Return dimensionless level fractions for fixed-point acceleration."""

        density = self.dopant_density_m3
        return np.divide(
            np.asarray(population, dtype=float),
            density[None, ...],
            out=np.zeros_like(np.asarray(population, dtype=float)),
            where=density[None, ...] > np.finfo(float).tiny,
        )

    def _population_from_fraction_candidate(
        self,
        candidate: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Project an accelerated fraction field onto each voxel simplex."""

        fractions = np.asarray(candidate, dtype=float)
        expected_shape = (len(self.levels.names), *self.grid.shape)
        if fractions.shape != expected_shape:
            raise ValueError("accelerated population fractions have the wrong shape")
        active = self.dopant_density_m3 > np.finfo(float).tiny
        if np.any(active):
            active_values = fractions[:, active]
            violation_values = [
                max(0.0, -float(np.min(active_values))),
                max(0.0, float(np.max(active_values)) - 1.0),
                float(np.max(np.abs(np.sum(active_values, axis=0) - 1.0))),
            ]
            for component, target_fraction in zip(
                self.levels.conservation_components(),
                self.levels.conservation_component_fractions(),
            ):
                violation_values.append(
                    float(
                        np.max(
                            np.abs(
                                np.sum(active_values[component], axis=0)
                                - target_fraction
                            )
                        )
                    )
                )
            violation = max(violation_values)
        else:
            violation = 0.0
        projected = np.maximum(fractions, 0.0)
        for component, target_fraction in zip(
            self.levels.conservation_components(),
            self.levels.conservation_component_fractions(),
        ):
            component_total = np.sum(projected[component], axis=0)
            missing = (
                active
                & (target_fraction > np.finfo(float).tiny)
                & (component_total <= np.finfo(float).tiny)
            )
            if np.any(missing):
                projected[component[0], missing] = target_fraction
        projected = self.levels.project_component_inventories(
            projected,
            np.where(active, 1.0, 0.0),
        )
        projected[:, ~active] = 0.0
        projection_rms = float(
            np.sqrt(np.mean((projected - fractions) ** 2))
        )
        population = projected * self.dopant_density_m3[None, ...]
        return population, violation, projection_rms

    def _accelerated_population_update(
        self,
        accelerator: SafeguardedAnderson,
        iteration: int,
        population: np.ndarray,
        mapped_population: np.ndarray,
        maximum_fraction_violation: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply Anderson in population-fraction space with physical safeguards."""

        current_fraction = self._population_fractions(population)
        mapped_fraction = self._population_fractions(mapped_population)
        decision = accelerator.propose(iteration, current_fraction, mapped_fraction)
        accepted = decision.accepted
        reason = decision.reason
        violation = 0.0
        projection_rms = 0.0
        if accepted:
            candidate, violation, projection_rms = (
                self._population_from_fraction_candidate(decision.state)
            )
            if violation > maximum_fraction_violation:
                accelerator.reject_last_step("population_constraint_violation")
                accepted = False
                reason = "population_constraint_violation"
                candidate = np.asarray(mapped_population, dtype=float).copy()
        else:
            candidate = np.asarray(mapped_population, dtype=float).copy()
        return candidate, {
            "anderson_accepted": accepted,
            "anderson_reason": reason,
            "anderson_history_depth": decision.history_depth,
            "anderson_predicted_residual_ratio": decision.predicted_residual_ratio,
            "anderson_restarted": decision.restarted,
            "population_fraction_constraint_violation": violation,
            "population_fraction_projection_rms": projection_rms,
        }

    def _thermal_response(
        self,
        pump_result: PumpPropagationResult,
        additional_heat_W_m3: np.ndarray | None = None,
        population: np.ndarray | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        dict[str, Any],
        dict[str, np.ndarray],
        np.ndarray,
    ]:
        """Map absorbed Pump power to heat, temperature, stress, and delta-n."""

        self._emit_progress(
            "thermal_response",
            "Building the volumetric heat source",
            fraction=0.0,
        )
        heat = self._heat_load(pump_result, population)
        if additional_heat_W_m3 is not None:
            additional = np.asarray(additional_heat_W_m3, dtype=float)
            if additional.shape != self.grid.shape:
                raise ValueError("additional thermal source must match the physical grid")
            if not np.all(np.isfinite(additional)):
                raise ValueError("additional thermal source must be finite")
            additional_scale = max(float(np.max(np.abs(additional))), 1.0)
            if float(np.min(additional)) < (
                -64.0 * np.finfo(float).eps * additional_scale
            ):
                raise ValueError(
                    "additional irreversible thermal source must be non-negative"
                )
            heat = heat + np.maximum(additional, 0.0) * self.crystal_mask
        self._emit_progress(
            "thermal_solver",
            "Solving the configured thermal field",
            fraction=0.25,
        )
        with self._progress_scope(0.25, 0.72):
            temperature, thermal_stats = self._solve_temperature(heat)
        reference = float(self.material.get("reference_temperature_K", 293.15))
        self._emit_progress(
            "stress_solver",
            "Computing stress, photoelastic response, and end-face deformation",
            fraction=0.72,
        )
        stress_solver = self._stress_backend()
        stress = stress_solver.solve_on_material(
            temperature,
            reference,
            self.crystal_mask,
        )
        endface_cfg = self.material.get("mechanical", {}).get(
            "endface_deformation", {}
        )
        if endface_cfg.get("model", "off") == "distributed_opd":
            front = np.asarray(
                stress.get("front_surface_displacement_z_m"), dtype=float
            )
            back = np.asarray(
                stress.get("back_surface_displacement_z_m"), dtype=float
            )
            if front.shape != self.grid.shape[-2:] or back.shape != self.grid.shape[-2:]:
                raise RuntimeError(
                    "endface deformation feedback requires front/back FEM surface displacement maps"
                )
            transverse_mask = np.any(self.crystal_mask, axis=0)
            if (
                not np.all(np.isfinite(front[transverse_mask]))
                or not np.all(np.isfinite(back[transverse_mask]))
            ):
                raise RuntimeError(
                    "endface deformation feedback contains non-finite FEM "
                    "surface displacement inside the crystal aperture"
                )
            front = np.where(transverse_mask, front, 0.0)
            back = np.where(transverse_mask, back, 0.0)
            stress["front_surface_sag_m"] = front
            stress["back_surface_sag_m"] = back
            sag = back - front
            outside_index = float(
                endface_cfg.get("outside_refractive_index", 1.0)
            )
            crystal_index = material_refractive_index(
                self.material,
                float(self.seed_cfg["wavelength_m"]),
                role="signal",
            )
            optical_path = (
                (crystal_index - outside_index)
                * sag
            )
            # Retain a centre-wavelength OPD only as a backward-compatible
            # diagnostic.  Optical propagation consumes the two raw surfaces
            # at each wavelength; this field is never folded into bulk delta-n.
            stress["endface_optical_path_difference_m"] = optical_path
            stress["endface_feedback_reference_wavelength_m"] = np.asarray(
                float(self.seed_cfg["wavelength_m"])
            )
            stress["endface_feedback_operator"] = np.asarray(
                "wavelength_and_beam_role_specific_surface_phase"
            )
        delta_n = np.zeros(self.grid.shape, dtype=float)
        if (
            self.physics["refractive_index_model"] == "thermal_3d"
            and bool(thermal_stats.get("enabled", False))
        ):
            delta_n = float(self.material.get("dn_dT_per_K", 0.0)) * (
                temperature - reference
            )
            delta_n += np.asarray(stress.get("delta_n_stress", 0.0), dtype=float)
            delta_n *= self.crystal_mask
        self._configure_endface_surface_sag(stress)
        self._emit_progress(
            "thermal_response",
            "Thermal and thermo-optic response completed",
            fraction=1.0,
            metrics={
                "thermal_iterations": int(thermal_stats.get("iterations", 0) or 0),
                "thermal_converged": thermal_stats.get("converged"),
                "peak_temperature_K": float(np.max(temperature)),
            },
        )
        return heat, temperature, thermal_stats, stress, delta_n

    def _configure_signal_photoelastic_tensor(
        self, stress: dict[str, np.ndarray]
    ) -> None:
        """Route FEM transverse index tensors to Jones-capable Seed solvers."""

        self._configure_signal_photoelastic_tensor_fields(
            self._photoelastic_tensor_fields(stress)
        )

    @staticmethod
    def _photoelastic_tensor_fields(
        stress: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray] | None:

        names = {
            "xx": "delta_n_photoelastic_xx",
            "yy": "delta_n_photoelastic_yy",
            "xy": "delta_n_photoelastic_xy",
        }
        return (
            {
                key: np.asarray(stress[source], dtype=float)
                for key, source in names.items()
            }
            if all(source in stress for source in names.values())
            else None
        )

    def _configure_signal_photoelastic_tensor_fields(
        self, fields: dict[str, np.ndarray] | None
    ) -> None:
        for solver in (
            self.signal_solver,
            self.average_signal_solver,
            self.reduced_pulse_map_signal_solver,
        ):
            if solver is None:
                continue
            setter = getattr(solver, "set_photoelastic_tensor", None)
            if setter is not None:
                setter(fields)

    def _configure_endface_surface_sag(
        self, stress: dict[str, np.ndarray] | None
    ) -> None:
        endface_cfg = self.material.get("mechanical", {}).get(
            "endface_deformation", {}
        )
        enabled = bool(
            stress is not None
            and endface_cfg.get("model", "off") == "distributed_opd"
            and "front_surface_sag_m" in stress
            and "back_surface_sag_m" in stress
        )
        front = None if not enabled else np.asarray(stress["front_surface_sag_m"])
        back = None if not enabled else np.asarray(stress["back_surface_sag_m"])
        for solver in (
            self.pump_solver,
            self.signal_solver,
            self.average_signal_solver,
            self.reduced_pulse_map_signal_solver,
        ):
            if solver is None:
                continue
            setter = getattr(solver, "set_endface_surface_sag", None)
            if setter is not None:
                setter(front, back)

    @staticmethod
    def _guard_reported_seed_backend_consistency(
        result: SignalPropagationResult,
        population_stats: dict[str, Any],
        seed_heat_mode: str,
    ) -> None:
        """Reject a mid-run backend change after a stateful closure.

        An initialization fallback is consistent because every Seed event uses
        that backend. A runtime fallback is also valid for an isolated pulse.
        It is not the same physical map after earlier Seed-depletion events, a
        periodic closure, or a Seed-to-thermal closure has already been solved.
        """

        if not _runtime_optional_fallback(result):
            return
        dynamics = str(population_stats.get("dynamics", ""))
        prior_seed_events = int(
            population_stats.get(
                "seed_depletion_events_before_reported_pulse",
                0,
            )
        )
        stateful_population = dynamics in {
            "periodic_steady_state",
            "reduced_periodic_steady_state",
        }
        seed_thermal_closure = str(seed_heat_mode) in {
            "fixed_point",
            "one_way_postprocess",
        }
        if stateful_population or prior_seed_events > 0 or seed_thermal_closure:
            raise RuntimeError(
                "the requested Seed backend failed during the reported pulse "
                "after a stateful population/thermal closure; returning the "
                "fallback pulse would mix different physical propagation maps"
            )

    def _propagate_seed_current_mode(
        self,
        population: np.ndarray,
        delta_n: np.ndarray,
        *,
        record_planes: bool | None = None,
    ) -> SignalPropagationResult:
        """Dispatch every reported and hidden Seed solve through one contract."""

        self._emit_progress(
            "seed_propagation",
            f"Starting {getattr(self, 'signal_model', 'configured')} Seed propagation",
            fraction=0.0,
        )
        if str(self.seed_cfg.get("temporal", {}).get("type", "gaussian")) == "cw":
            result = self.average_signal_solver.propagate_average(
                population,
                delta_n,
                float(self.sim_cfg["repetition_rate_Hz"]),
                (
                    bool(self.physics["record_optical_planes"])
                    if record_planes is None
                    else bool(record_planes)
                ),
            )
            self._emit_progress(
                "seed_propagation",
                "Average-power Seed propagation completed",
                fraction=1.0,
                metrics={
                    "output_energy_J": getattr(result, "system_output_energy_J", None)
                },
            )
            return result
        if record_planes is None:
            result = self.signal_solver.propagate(population, delta_n)
            self._emit_progress(
                "seed_propagation",
                "Seed propagation completed",
                fraction=1.0,
                metrics={
                    "output_energy_J": getattr(result, "system_output_energy_J", None)
                },
            )
            return result
        targets = [
            candidate
            for candidate in (
                self.signal_solver,
                getattr(self.signal_solver, "primary", None),
                getattr(self.signal_solver, "fallback", None),
            )
            if candidate is not None and hasattr(candidate, "physics")
        ]
        previous = [
            bool(target.physics.get("record_optical_planes", False))
            for target in targets
        ]
        try:
            for target in targets:
                target.physics["record_optical_planes"] = bool(record_planes)
            result = self.signal_solver.propagate(population, delta_n)
        finally:
            for target, value in zip(targets, previous, strict=True):
                target.physics["record_optical_planes"] = value
        self._emit_progress(
            "seed_propagation",
            "Seed propagation and optical-plane recording completed",
            fraction=1.0,
            metrics={
                **self._signal_progress_metrics(
                    result,
                    include_spectrum_preview=True,
                ),
                **self._population_progress_metrics(result.population_m3),
            },
        )
        return result

    @contextmanager
    def _reported_seed_diagnostics(self) -> Iterator[None]:
        """Arm observers across wrappers only for the actual reported pass.

        The first one-way heat-source probe is not a reported pass. All inner
        maps, thermal retries and cycle-average calls remain unarmed by default.
        """
        enabled = not bool(self.physics.get("_suppress_output_diagnostics", False))
        pending = [self.signal_solver, self.average_signal_solver]
        visited: set[int] = set()
        physics_objects = {id(self.physics): self.physics}
        while pending:
            target = pending.pop()
            if target is None or id(target) in visited:
                continue
            visited.add(id(target))
            physics = getattr(target, "physics", None)
            if isinstance(physics, dict):
                physics_objects[id(physics)] = physics
            pending.extend(getattr(target, name, None) for name in ("inner", "primary", "fallback", "radial"))
        missing = object()
        previous = []
        final_only = self.result_recording.get("diagnostics_timing") == "final_output"
        try:
            for physics in physics_objects.values():
                values = {"_final_output_diagnostics": enabled}
                if final_only:
                    values["_record_cycle_scalar_diagnostics"] = enabled
                for key, value in values.items():
                    previous.append((physics, key, physics.get(key, missing)))
                    physics[key] = value
            yield
        finally:
            for physics, key, value in reversed(previous):
                if value is missing:
                    physics.pop(key, None)
                else:
                    physics[key] = value

    def _propagate_seed_with_progress(
        self,
        population: np.ndarray,
        delta_n: np.ndarray,
        *,
        record_planes: bool | None = None,
    ) -> SignalPropagationResult:
        """Expose axial cell progress only for the user-facing final Seed pass."""

        previous = self._seed_progress_details_enabled
        self._seed_progress_details_enabled = True
        try:
            with self._reported_seed_diagnostics():
                return self._propagate_seed_current_mode(
                    population,
                    delta_n,
                    record_planes=record_planes,
                )
        finally:
            self._seed_progress_details_enabled = previous

    def _run_thermal_optical_fixed_point(
        self,
    ) -> tuple[
        np.ndarray,
        dict[str, Any],
        PumpPropagationResult,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
        dict[str, np.ndarray],
        np.ndarray,
        dict[str, Any],
    ]:
        """Iterate Pump/population/thermal optics to a relaxed fixed point.

        Seed irreversible heating is included only when its explicit fixed-point
        switch is enabled.  The loop closes Pump/ASE/population, heat, mechanical
        stress, delta-n and Pump propagation without reusing a reported Seed run
        or depositing the same loss twice.
        """

        cfg = self.config.get("multiphysics", {})
        iteration_scheme = cfg.get("iteration_scheme", "nested")
        joint_mode = iteration_scheme in {"joint", "joint_frozen"}
        frozen_probe_mode = iteration_scheme == "joint_frozen"
        joint_cycles = int(cfg.get("joint_cycles_per_iteration", 1))
        probe_cycles = int(cfg.get("frozen_probe_cycles", 3))
        previous_joint_map_state = getattr(self, "_joint_map_state", None)
        if frozen_probe_mode:
            self._joint_map_state = {}
        frozen_probe_verified = False
        minimum = int(cfg.get("minimum_iterations", 2))
        maximum = int(cfg.get("maximum_iterations", 20))
        tolerance = float(cfg.get("relative_tolerance", 1e-4))
        relaxation = float(cfg.get("relaxation", 0.5))
        continuation = self._continuation_state
        delta_n = (
            np.zeros(self.grid.shape, dtype=float)
            if continuation is None or continuation.thermal_delta_n is None
            else continuation.thermal_delta_n.copy()
        )
        response_mode = self.levels.response.mode
        response_temperature = (
            np.full(
                self.grid.shape,
                float(self.material.get("operating_temperature_K", 293.15)),
                dtype=float,
            )
            if continuation is None
            or continuation.response_temperature_K is None
            else continuation.response_temperature_K.copy()
        )
        seed_heat_mode = str(cfg.get("seed_heat_coupling", "off"))
        previous_absorbed: float | None = None
        previous_population: np.ndarray | None = None
        previous_thermal_temperature: np.ndarray | None = None
        previous_seed_heat: np.ndarray | None = None
        previous_seed_output_energy_J: float | None = None
        history: list[dict[str, Any]] = []
        nested_population_history: list[dict[str, Any]] = []
        seed_probe_history: list[dict[str, Any]] = []
        runtime_optional_fallback_iterations: list[int] = []
        nonconverged_population_iterations: list[int] = []
        nonconverged_thermal_iterations: list[int] = []
        converged = False
        closure_pending = False
        closure_attempts = 0
        closure_checks: list[dict[str, Any]] = []
        accepted_thermal_state = None
        accepted_thermal_iteration: int | None = None
        accelerator = SafeguardedAnderson(cfg.get("acceleration", {}))
        delta_n_scale: float | None = None
        temperature_scale: float | None = None
        periodic_enabled = bool(
            self.sim_cfg.get("periodic_steady_state", {}).get("enabled", False)
        )
        periodic_population_guess: np.ndarray | None = (
            None
            if continuation is None
            or continuation.periodic_boundary_population_m3 is None
            else continuation.periodic_boundary_population_m3.copy()
        )
        joint_previous_output = None
        joint_stable_count = 0
        warm_start_uses = 0
        tensor_fields: dict[str, np.ndarray] | None = None
        surface_sag_enabled = bool(
            self.material.get("mechanical", {})
            .get("endface_deformation", {})
            .get("model", "off")
            == "distributed_opd"
        )
        current_front_sag = (
            np.zeros(self.grid.shape[-2:], dtype=float)
            if continuation is None or continuation.front_surface_sag_m is None
            else continuation.front_surface_sag_m.copy()
        )
        current_back_sag = (
            np.zeros(self.grid.shape[-2:], dtype=float)
            if continuation is None or continuation.back_surface_sag_m is None
            else continuation.back_surface_sag_m.copy()
        )
        if surface_sag_enabled:
            self._configure_endface_surface_sag(
                {
                    "front_surface_sag_m": current_front_sag,
                    "back_surface_sag_m": current_back_sag,
                }
            )
        if (
            str(
                self.seed_cfg.get("polarization", {}).get(
                    "birefringence_model", "off"
                )
            )
            == "tensor_photoelastic"
        ):
            # The first fixed-point probe precedes the first mechanical solve.
            # A tensor-Jones backend nevertheless requires a complete field
            # stack, so use the physically neutral zero-stress iterate rather
            # than dropping the tensor operator for iteration one.
            tensor_fields = {
                key: np.zeros(self.grid.shape, dtype=float)
                for key in ("xx", "yy", "xy")
            }

        original_record = bool(self.physics.get("record_optical_planes", False))
        original_suppression = self.physics.get("_suppress_output_diagnostics")
        previous_outer_progress_context = dict(
            getattr(self, "_outer_progress_context", {})
        )
        self._emit_progress(
            "thermal_optical_fixed_point",
            "Beginning thermal-optical fixed-point iterations",
            fraction=0.0,
            current=0,
            total=maximum,
            metrics={"target_residual": tolerance},
        )
        original_progress_depth = len(self._progress_ranges)
        try:
            # A closure check is a full inner solve followed by a thermal map.
            # Failed checks consume the remaining outer-iteration budget.  Keep
            # the historical allowance of one final check after that budget.
            # Only checks retain Pump planes; the reported Seed is replayed
            # from the same pre-Seed state and the same accepted thermal fields.
            self.physics["record_optical_planes"] = False
            self.physics["_suppress_output_diagnostics"] = True
            for iteration in range(1, maximum + 2):
                is_closure = closure_pending
                self.physics["record_optical_planes"] = (
                    original_record if is_closure else False
                )
                if is_closure:
                    closure_attempts += 1
                    self._emit_progress(
                        "thermal_optical_closure",
                        "Checking the internally converged state against a new thermal solve",
                        fraction=(iteration - 1) / (maximum + 1),
                        current=iteration - 1,
                        total=maximum + 1,
                    )
                self._outer_progress_context = {
                    "thermal_optical_iteration": iteration,
                    "thermal_optical_total": maximum + 1,
                }
                self._emit_progress(
                    "thermal_optical_fixed_point",
                    (
                        f"Thermal-optical closure check at evaluation {iteration}"
                        if is_closure
                        else f"Thermal-optical iteration {iteration}/{maximum}"
                    ),
                    fraction=(iteration - 1) / max(maximum + 1, 1),
                    current=iteration - 1,
                    total=maximum + 1,
                )
                parent_start, parent_stop = self._progress_ranges[-1]
                parent_span = parent_stop - parent_start
                self._progress_ranges.append(
                    (
                        parent_start
                        + parent_span * (iteration - 1) / max(maximum + 1, 1),
                        parent_start
                        + parent_span * iteration / max(maximum + 1, 1),
                    )
                )
                if response_mode == "local_3d":
                    self.levels.set_environment(
                        response_temperature, self.dopant_density_m3
                    )
                self._configure_signal_photoelastic_tensor_fields(tensor_fields)
                # Joint packets advance the configured pulse count per heat
                # update; final verification follows the selected policy.
                periodic_cfg = self.sim_cfg["periodic_steady_state"]
                saved_limits = {key: periodic_cfg[key] for key in ("minimum_cycles", "maximum_cycles", "consecutive_cycles", "acceleration")}
                if frozen_probe_mode:
                    self._joint_map_state["fixed_probe"] = is_closure
                if joint_mode and not is_closure:
                    periodic_cfg.update(minimum_cycles=joint_cycles, maximum_cycles=joint_cycles, consecutive_cycles=1)
                elif frozen_probe_mode and is_closure:
                    periodic_cfg.update(minimum_cycles=1, maximum_cycles=probe_cycles, consecutive_cycles=probe_cycles,
                                        acceleration={"method": "off"})
                try:
                    population, iteration_population_stats, pump = self._operating_point(
                        delta_n,
                        original_record if is_closure else False,
                        periodic_initial_population=periodic_population_guess,
                        periodic_initialization_source=(
                            "pump_sweep_previous_point"
                            if iteration == 1
                            and continuation is not None
                            and continuation.periodic_boundary_population_m3 is not None
                            else "thermal_outer_warm_start"
                        ),
                    )
                finally:
                    periodic_cfg.update(saved_limits)
                self._record_nested_population_history(
                    nested_population_history,
                    iteration_population_stats,
                    outer_iteration=iteration,
                    outer_stage="final_closure" if is_closure else "fixed_point",
                )
                population_solver_converged = bool(
                    iteration_population_stats.get("converged", True)
                )
                if not population_solver_converged and iteration <= maximum:
                    nonconverged_population_iterations.append(iteration)
                if periodic_enabled:
                    next_guess = self._last_periodic_boundary_population
                    if next_guess is not None:
                        periodic_population_guess = next_guess.copy()
                        if iteration > 1:
                            warm_start_uses += 1
                if frozen_probe_mode and is_closure:
                    # Verify the pulse map on the accepted, unchanged thermal
                    # state. This policy deliberately does not evaluate a new
                    # thermal map after the pulse verification.
                    self._progress_ranges.pop()
                    frozen_probe_verified = population_solver_converged
                    closure_checks.append({
                        "iteration": iteration, "policy": "frozen_thermal_pulse_probe",
                        "converged": frozen_probe_verified,
                        "thermal_recomputed": False,
                        "probe_cycles": iteration_population_stats.get("cycles_completed"),
                        "population_residual_relative": iteration_population_stats.get("final_population_residual_relative"),
                        "output_residual_relative": iteration_population_stats.get("final_output_residual_relative"),
                        "target_residual": periodic_cfg["relative_tolerance"],
                    })
                    if frozen_probe_verified or iteration > maximum:
                        converged = frozen_probe_verified
                        break
                    closure_pending = False
                    joint_stable_count = 0
                    joint_previous_output = None
                    continue
                seed_probe = (
                    self._propagate_seed_current_mode(
                        population.copy(), delta_n, record_planes=False
                    )
                    if seed_heat_mode == "fixed_point"
                    else None
                )
                runtime_fallback = _runtime_optional_fallback(seed_probe)
                if runtime_fallback and is_closure:
                    raise RuntimeError(
                        "the requested Seed backend failed during the final "
                        "thermal-optical closure probe; fallback heat would not "
                        "represent the physical map iterated above"
                    )
                if runtime_fallback:
                    runtime_optional_fallback_iterations.append(iteration)
                seed_heat = self._seed_irreversible_heat(seed_probe)
                seed_output_energy_J = (
                    None
                    if seed_probe is None
                    else float(seed_probe.system_output_energy_J)
                )
                if (
                    seed_probe is not None
                    and bool(self.result_recording["iteration_history"])
                ):
                    seed_probe_history.append(
                        {
                            "outer_iteration": int(iteration),
                            "outer_stage": "final_closure" if is_closure else "fixed_point",
                            **self._signal_progress_metrics(seed_probe),
                        }
                    )
                (
                    iteration_heat,
                    thermal_temperature,
                    iteration_thermal_stats,
                    iteration_stress,
                    thermal_delta_n,
                ) = self._thermal_response(
                    pump,
                    seed_heat,
                    population=population,
                )
                self._progress_ranges.pop()
                thermal_solver_converged = bool(
                    not iteration_thermal_stats.get("enabled", False)
                    or iteration_thermal_stats.get("converged", False)
                )
                if not thermal_solver_converged and iteration <= maximum:
                    nonconverged_thermal_iterations.append(iteration)
                candidate_tensor_fields = self._photoelastic_tensor_fields(
                    iteration_stress
                )
                if surface_sag_enabled:
                    candidate_front_sag = np.asarray(
                        iteration_stress["front_surface_sag_m"], dtype=float
                    )
                    candidate_back_sag = np.asarray(
                        iteration_stress["back_surface_sag_m"], dtype=float
                    )
                    sag_scale = max(
                        float(np.max(np.abs(candidate_front_sag))),
                        float(np.max(np.abs(candidate_back_sag))),
                        float(np.max(np.abs(current_front_sag))),
                        float(np.max(np.abs(current_back_sag))),
                        1e-15,
                    )
                    surface_sag_residual = max(
                        float(
                            np.max(np.abs(candidate_front_sag - current_front_sag))
                            / sag_scale
                        ),
                        float(
                            np.max(np.abs(candidate_back_sag - current_back_sag))
                            / sag_scale
                        ),
                    )
                else:
                    candidate_front_sag = current_front_sag
                    candidate_back_sag = current_back_sag
                    surface_sag_residual = 0.0
                if seed_heat_mode != "fixed_point":
                    tensor_residual = 0.0
                elif candidate_tensor_fields is None and tensor_fields is None:
                    tensor_residual = 0.0
                elif candidate_tensor_fields is None or tensor_fields is None:
                    tensor_residual = float("inf")
                else:
                    tensor_scale = max(
                        *(
                            float(np.max(np.abs(values)))
                            for values in candidate_tensor_fields.values()
                        ),
                        *(
                            float(np.max(np.abs(values)))
                            for values in tensor_fields.values()
                        ),
                        1e-15,
                    )
                    tensor_residual = max(
                        float(
                            np.max(
                                np.abs(
                                    candidate_tensor_fields[key]
                                    - tensor_fields[key]
                                )
                            )
                            / tensor_scale
                        )
                        for key in candidate_tensor_fields
                    )
                scale = max(
                    float(np.max(np.abs(thermal_delta_n))),
                    float(np.max(np.abs(delta_n))),
                    1e-15,
                )
                delta_residual = float(
                    np.max(np.abs(thermal_delta_n - delta_n)) / scale
                )
                absorbed = float(pump.stats["absorbed_power_W"])
                absorbed_residual = (
                    float("inf")
                    if previous_absorbed is None
                    else abs(absorbed - previous_absorbed)
                    / max(abs(absorbed), abs(previous_absorbed), np.finfo(float).tiny)
                )
                population_residual = (
                    float("inf")
                    if previous_population is None
                    else self._population_residual_relative(population, previous_population)
                )
                spectroscopy_temperature_residual = (
                    float(
                        np.max(np.abs(thermal_temperature - response_temperature))
                        / max(
                            float(np.max(np.abs(thermal_temperature))),
                            float(np.max(np.abs(response_temperature))),
                            1.0,
                        )
                    )
                    if response_mode == "local_3d"
                    else 0.0
                )
                thermal_temperature_residual = (
                    float("inf")
                    if previous_thermal_temperature is None
                    else float(
                        np.max(
                            np.abs(
                                thermal_temperature
                                - previous_thermal_temperature
                            )
                        )
                        / max(
                            float(np.max(np.abs(thermal_temperature))),
                            float(
                                np.max(
                                    np.abs(previous_thermal_temperature)
                                )
                            ),
                            1.0,
                        )
                    )
                )
                seed_heat_residual = (
                    0.0
                    if seed_heat_mode != "fixed_point"
                    else (
                        float("inf")
                        if previous_seed_heat is None
                        else float(
                            np.max(np.abs(seed_heat - previous_seed_heat))
                            / max(
                                float(np.max(np.abs(seed_heat))),
                                float(np.max(np.abs(previous_seed_heat))),
                                np.finfo(float).tiny,
                            )
                        )
                    )
                )
                seed_output_residual = (
                    0.0
                    if seed_heat_mode != "fixed_point"
                    else (
                        float("inf")
                        if previous_seed_output_energy_J is None
                        or seed_output_energy_J is None
                        else abs(
                            seed_output_energy_J
                            - previous_seed_output_energy_J
                        )
                        / max(
                            abs(seed_output_energy_J),
                            abs(previous_seed_output_energy_J),
                            np.finfo(float).tiny,
                        )
                    )
                )
                residual = max(
                    delta_residual,
                    absorbed_residual,
                    population_residual,
                    spectroscopy_temperature_residual,
                    thermal_temperature_residual,
                    seed_heat_residual,
                    seed_output_residual,
                    tensor_residual,
                    surface_sag_residual,
                )
                valid_inner_map = bool(
                    not runtime_fallback
                    and (population_solver_converged or (joint_mode and not is_closure))
                    and thermal_solver_converged
                )
                if not valid_inner_map:
                    # This probe may seed the next relaxed thermal iterate, but
                    # a runtime backend change or any unconverged inner solve
                    # is not an evaluation of the requested coupled map. It
                    # cannot close the residual or share quasi-Newton/history
                    # baselines with the next valid evaluation.
                    residual = float("inf")
                joint_history = iteration_population_stats.get("convergence_history", [])
                joint_output = float(joint_history[-1]["output_energy_J"]) if joint_history and "output_energy_J" in joint_history[-1] else None
                joint_output_residual = (
                    float("inf") if joint_previous_output is None or joint_output is None
                    else abs(joint_output - joint_previous_output) / max(abs(joint_output), abs(joint_previous_output), np.finfo(float).tiny)
                )
                joint_population_residual = float(iteration_population_stats.get("final_population_residual_relative", float("inf")))
                joint_tolerance = float(periodic_cfg["relative_tolerance"])
                joint_required = int((cfg.get("joint_consecutive_iterations") or periodic_cfg["consecutive_cycles"]))
                joint_carrier_residual = (
                    float(iteration_population_stats.get("final_free_carrier_convergence_residual_relative", 0.0))
                    if frozen_probe_mode and self._free_carrier_convergence_gate_enabled() else 0.0
                )
                joint_pass = (joint_output_residual <= joint_tolerance
                              and joint_population_residual <= joint_tolerance
                              and joint_carrier_residual <= joint_tolerance
                              and residual <= tolerance)
                joint_stable_count = joint_stable_count + 1 if joint_pass else 0
                joint_ready = joint_stable_count >= joint_required
                joint_previous_output = joint_output
                iteration_record = {
                    "iteration": iteration,
                    "joint_output_residual": joint_output_residual,
                    "joint_population_residual": joint_population_residual,
                    "free_carrier_convergence_gate_enabled": self._free_carrier_convergence_gate_enabled(),
                    "joint_carrier_residual_used": joint_carrier_residual,
                    "joint_stable_count": joint_stable_count,
                    "joint_ready": joint_ready,
                    "runtime_optional_fallback": runtime_fallback,
                    "joint_single_pulse": joint_mode and joint_cycles == 1 and not is_closure,
                    "population_solver_converged": (
                        population_solver_converged
                    ),
                    "delta_n_residual_relative": delta_residual,
                    "absorbed_pump_residual_relative": absorbed_residual,
                    "population_residual_relative": population_residual,
                    "spectroscopy_temperature_residual_relative": spectroscopy_temperature_residual,
                    "thermal_temperature_residual_relative": thermal_temperature_residual,
                    "seed_heat_residual_relative": seed_heat_residual,
                    "seed_output_energy_residual_relative": seed_output_residual,
                    "photoelastic_tensor_residual_relative": tensor_residual,
                    "endface_surface_sag_residual_relative": surface_sag_residual,
                    "thermal_solver_converged": thermal_solver_converged,
                    "thermal_solver_residual_K": float(
                        iteration_thermal_stats.get("residual_K", 0.0)
                    ),
                    "absorbed_pump_power_W": absorbed,
                    "delta_n_peak_to_valley": float(np.ptp(thermal_delta_n)),
                    "seed_irreversible_heat_W": float(
                        self.grid.integrate_volume(seed_heat)
                    ),
                    "seed_nonlinear_heat_W": float(
                        self.grid.integrate_volume(seed_heat)
                    ),
                    "seed_output_energy_J": seed_output_energy_J,
                }
                iteration_record["outer_stage"] = (
                    "final_closure" if is_closure else "fixed_point"
                )
                if is_closure:
                    closure_checks.append({
                        **iteration_record,
                        "residual": residual,
                        "converged": bool(valid_inner_map and residual <= tolerance),
                    })
                    if (valid_inner_map and residual <= tolerance) or iteration > maximum:
                        converged = bool(valid_inner_map and residual <= tolerance)
                        break
                    # A failed verification is an ordinary coupled-map step;
                    # it must be counted and relaxed, not silently discarded.
                    closure_pending = False
                history.append(iteration_record)
                self._emit_progress(
                    "thermal_optical_fixed_point",
                    f"Thermal-optical iteration {iteration}/{maximum} completed",
                    fraction=iteration / max(maximum + 1, 1),
                    current=iteration,
                    total=maximum,
                    metrics={
                        "residual": residual,
                        "target_residual": tolerance,
                        "absorbed_pump_power_W": absorbed,
                        "delta_n_residual": delta_residual,
                        "population_residual": population_residual,
                        "thermal_solver_converged": thermal_solver_converged,
                        "joint_single_pulse": joint_mode and joint_cycles == 1 and not is_closure,
                        "population_solver_converged": (
                            population_solver_converged
                        ),
                    },
                )
                if valid_inner_map:
                    previous_absorbed = absorbed
                    previous_population = population.copy()
                    previous_thermal_temperature = thermal_temperature.copy()
                    previous_seed_heat = seed_heat.copy()
                    previous_seed_output_energy_J = seed_output_energy_J
                else:
                    # The returned fields remain useful as a damped numerical
                    # guess, but a following valid iteration must establish a
                    # fresh, like-for-like closure baseline before it can stop.
                    previous_absorbed = None
                    previous_population = None
                    previous_thermal_temperature = None
                    previous_seed_heat = None
                    previous_seed_output_energy_J = None
                if (
                    (
                        iteration >= minimum
                        and (not joint_mode or joint_ready)
                        and residual <= tolerance
                        and valid_inner_map
                    )
                    or iteration == maximum
                ):
                    if frozen_probe_mode and not (
                        iteration >= minimum and joint_ready and valid_inner_map and residual <= tolerance
                    ):
                        # The budget is not evidence of convergence. Preserve
                        # the last applied thermal state for diagnostics only.
                        accepted_thermal_state = (
                            iteration_heat.copy(), thermal_temperature.copy(),
                            copy.deepcopy(iteration_thermal_stats),
                            {key: value.copy() for key, value in iteration_stress.items()},
                            thermal_delta_n.copy(),
                        )
                        accepted_thermal_iteration = iteration
                        break
                    history[-1].update({
                        "anderson_accepted": False,
                        "anderson_reason": (
                            "closure_verification_scheduled" if valid_inner_map
                            else "runtime_optional_fallback_history_reset" if runtime_fallback
                            else "unconverged_population_history_reset" if not population_solver_converged
                            else "unconverged_thermal_history_reset"
                        ),
                        "anderson_history_depth": 0,
                        "anderson_predicted_residual_ratio": None,
                        "anderson_restarted": not valid_inner_map,
                    })
                    if not valid_inner_map:
                        accelerator.reset()
                    accepted_thermal_state = (
                        iteration_heat.copy(),
                        thermal_temperature.copy(),
                        copy.deepcopy(iteration_thermal_stats),
                        {key: value.copy() for key, value in iteration_stress.items()},
                        thermal_delta_n.copy(),
                    )
                    accepted_thermal_iteration = iteration
                    delta_n = thermal_delta_n
                    tensor_fields = (
                        None
                        if candidate_tensor_fields is None
                        else {
                            key: value.copy()
                            for key, value in candidate_tensor_fields.items()
                        }
                    )
                    if response_mode == "local_3d":
                        response_temperature = thermal_temperature
                    current_front_sag = candidate_front_sag.copy()
                    current_back_sag = candidate_back_sag.copy()
                    if surface_sag_enabled:
                        self._configure_endface_surface_sag(iteration_stress)
                    closure_pending = True
                    continue
                base_delta_n = (
                    (1.0 - relaxation) * delta_n
                    + relaxation * thermal_delta_n
                )
                if response_mode == "local_3d":
                    base_response_temperature = (
                        (1.0 - relaxation) * response_temperature
                        + relaxation * thermal_temperature
                    )
                else:
                    base_response_temperature = response_temperature
                if candidate_tensor_fields is None:
                    base_tensor_fields = None
                elif tensor_fields is None:
                    base_tensor_fields = {
                        key: relaxation * value
                        for key, value in candidate_tensor_fields.items()
                    }
                else:
                    base_tensor_fields = {
                        key: (1.0 - relaxation) * tensor_fields[key]
                        + relaxation * value
                        for key, value in candidate_tensor_fields.items()
                    }
                if delta_n_scale is None:
                    delta_n_scale = max(
                        float(np.max(np.abs(thermal_delta_n))), 1e-12
                    )
                state_parts = [(delta_n / delta_n_scale).ravel()]
                mapped_parts = [(base_delta_n / delta_n_scale).ravel()]
                if response_mode == "local_3d":
                    reference_temperature = float(
                        self.material.get("reference_temperature_K", 293.15)
                    )
                    if temperature_scale is None:
                        temperature_scale = max(
                            float(
                                np.max(
                                    np.abs(
                                        thermal_temperature
                                        - reference_temperature
                                    )
                                )
                            ),
                            1.0,
                        )
                    state_parts.append(
                        (
                            (response_temperature - reference_temperature)
                            / temperature_scale
                        ).ravel()
                    )
                    mapped_parts.append(
                        (
                            (base_response_temperature - reference_temperature)
                            / temperature_scale
                        ).ravel()
                    )
                state_vector = np.concatenate(state_parts)
                mapped_vector = np.concatenate(mapped_parts)
                if not valid_inner_map:
                    accelerator.reset()
                    acceleration_step = None
                    acceleration_accepted = False
                    acceleration_reason = (
                        "runtime_optional_fallback_history_reset"
                        if runtime_fallback
                        else (
                            "unconverged_population_history_reset"
                            if not population_solver_converged
                            else "unconverged_thermal_history_reset"
                        )
                    )
                else:
                    acceleration_step = accelerator.propose(
                        iteration, state_vector, mapped_vector
                    )
                    acceleration_accepted = acceleration_step.accepted
                    acceleration_reason = acceleration_step.reason
                if acceleration_accepted:
                    assert acceleration_step is not None
                    cell_count = delta_n.size
                    candidate_delta_n = (
                        acceleration_step.state[:cell_count].reshape(delta_n.shape)
                        * delta_n_scale
                    )
                    if response_mode == "local_3d":
                        assert temperature_scale is not None
                        candidate_temperature = (
                            acceleration_step.state[cell_count:].reshape(
                                response_temperature.shape
                            )
                            * temperature_scale
                            + reference_temperature
                        )
                    else:
                        candidate_temperature = base_response_temperature
                    if (
                        not np.all(np.isfinite(candidate_delta_n))
                        or not np.all(np.isfinite(candidate_temperature))
                        or float(np.min(candidate_temperature)) <= 0.0
                    ):
                        accelerator.reject_last_step("thermal_state_constraint_violation")
                        acceleration_accepted = False
                        acceleration_reason = "thermal_state_constraint_violation"
                if acceleration_accepted:
                    delta_n = candidate_delta_n
                    response_temperature = candidate_temperature
                else:
                    delta_n = base_delta_n
                    response_temperature = base_response_temperature
                tensor_fields = base_tensor_fields
                if surface_sag_enabled:
                    current_front_sag = (
                        (1.0 - relaxation) * current_front_sag
                        + relaxation * candidate_front_sag
                    )
                    current_back_sag = (
                        (1.0 - relaxation) * current_back_sag
                        + relaxation * candidate_back_sag
                    )
                    self._configure_endface_surface_sag(
                        {
                            "front_surface_sag_m": current_front_sag,
                            "back_surface_sag_m": current_back_sag,
                        }
                    )
                history[-1].update(
                    {
                        "anderson_accepted": acceleration_accepted,
                        "anderson_reason": acceleration_reason,
                        "anderson_history_depth": (
                            0
                            if acceleration_step is None
                            else acceleration_step.history_depth
                        ),
                        "anderson_predicted_residual_ratio": (
                            None
                            if acceleration_step is None
                            else acceleration_step.predicted_residual_ratio
                        ),
                        "anderson_restarted": (
                            not valid_inner_map
                            if acceleration_step is None
                            else acceleration_step.restarted
                        ),
                    }
                )
        finally:
            if frozen_probe_mode:
                if previous_joint_map_state is None:
                    del self._joint_map_state
                else:
                    self._joint_map_state = previous_joint_map_state
            del self._progress_ranges[original_progress_depth:]
            self._outer_progress_context = previous_outer_progress_context
            self.physics["record_optical_planes"] = original_record
            if original_suppression is None:
                self.physics.pop("_suppress_output_diagnostics", None)
            else:
                self.physics["_suppress_output_diagnostics"] = original_suppression

        assert accepted_thermal_state is not None
        population_stats = iteration_population_stats
        pump_result = pump
        final_seed_heat = seed_heat
        final_delta_residual = delta_residual
        final_absorbed_residual = absorbed_residual
        final_population_residual = population_residual
        final_spectroscopy_temperature_residual = spectroscopy_temperature_residual
        final_thermal_temperature_residual = thermal_temperature_residual
        final_seed_heat_residual = seed_heat_residual
        final_seed_output_residual = seed_output_residual
        final_tensor_residual = tensor_residual
        final_surface_sag_residual = surface_sag_residual
        final_thermal_solver_converged = thermal_solver_converged
        final_population_solver_converged = population_solver_converged
        heat, temperature, thermal_stats, stress, final_delta_n = accepted_thermal_state
        # These fields are the previous complete thermal solution, actually
        # consumed by the final inner solve. The new solution only verifies
        # closure; its diagnostics must not masquerade as the saved fields'.
        thermal_stats["coupled_field_source"] = "accepted_inner_map_thermal_state"
        thermal_stats["coupled_field_iteration"] = accepted_thermal_iteration
        thermal_stats["closure_verification"] = {
            "applied_to_reported_seed": False,
            "converged": thermal_solver_converged,
            "heat_total_W": float(self.grid.integrate_volume(iteration_heat)),
            "temperature_min_K": float(np.min(thermal_temperature)),
            "temperature_max_K": float(np.max(thermal_temperature)),
            "residual_K": float(iteration_thermal_stats.get("residual_K", 0.0)),
            "delta_n_residual_relative": final_delta_residual,
            "spectroscopy_temperature_residual_relative": final_spectroscopy_temperature_residual,
        }
        if frozen_probe_mode:
            thermal_stats["closure_verification"] = {
                "policy": "frozen_thermal_pulse_probe",
                "thermal_recomputed": False,
                "applied_to_reported_seed": False,
                "pulse_probe_converged": frozen_probe_verified,
            }
        if response_mode == "local_3d":
            self.levels.set_environment(temperature, self.dopant_density_m3)
        population_feedback = self.physics["population_dynamics"] != "frozen"
        pump_thermal_refraction = not (
            self.physics["pump_propagation"] == "stochastic_raytrace"
            and not bool(
                self.pump_cfg.get("raytrace", {}).get("thermal_refraction", True)
            )
        )
        coupling_stats: dict[str, Any] = {
            "model": "thermal_optical_fixed_point",
            "converged": converged,
            "iterations": len(history),
            "closure_policy": "frozen_thermal_pulse_probe" if frozen_probe_mode else "verify_then_accept_inner_state_or_retry",
            "final_thermal_closure_recomputed": not frozen_probe_mode,
            "frozen_probe_cycles": probe_cycles if frozen_probe_mode else None,
            "joint_cycles_per_iteration": joint_cycles if joint_mode else None,
            "joint_consecutive_iterations": int((cfg.get("joint_consecutive_iterations") or self.sim_cfg["periodic_steady_state"]["consecutive_cycles"])) if joint_mode else None,
            "closure_attempts": closure_attempts,
            "closure_checks": closure_checks,
            "coupled_map_evaluations": iteration,
            "maximum_coupled_map_evaluations": maximum + 1,
            "accepted_thermal_iteration": accepted_thermal_iteration,
            "reported_state_matches_final_inner_map": (not frozen_probe_mode or bool(closure_checks)),
            "minimum_iterations": minimum,
            "maximum_iterations": maximum,
            "relative_tolerance": tolerance,
            "relaxation": relaxation,
            "acceleration": accelerator.diagnostics(),
            "runtime_optional_fallback_iterations": list(
                runtime_optional_fallback_iterations
            ),
            "nonconverged_population_iterations": list(
                nonconverged_population_iterations
            ),
            "nonconverged_thermal_iterations": list(
                nonconverged_thermal_iterations
            ),
            "periodic_operating_point_warm_start_uses": warm_start_uses,
            "pump_sweep_continuation_used": continuation is not None,
            "pump_sweep_population_warm_start_used": bool(
                continuation is not None
                and continuation.periodic_boundary_population_m3 is not None
            ),
            "pump_sweep_thermal_optical_warm_start_used": bool(
                continuation is not None
                and continuation.thermal_delta_n is not None
                and continuation.response_temperature_K is not None
            ),
            "final_delta_n_closure_residual_relative": final_delta_residual,
            "final_absorbed_pump_closure_residual_relative": final_absorbed_residual,
            "final_population_closure_residual_relative": final_population_residual,
            "final_spectroscopy_temperature_residual_relative": final_spectroscopy_temperature_residual,
            "final_thermal_temperature_residual_relative": final_thermal_temperature_residual,
            "final_seed_heat_residual_relative": final_seed_heat_residual,
            "final_seed_output_energy_residual_relative": final_seed_output_residual,
            "final_photoelastic_tensor_residual_relative": final_tensor_residual,
            "final_endface_surface_sag_residual_relative": (
                final_surface_sag_residual
            ),
            "thermal_solver_converged": final_thermal_solver_converged,
            "joint_single_pulse": joint_mode and joint_cycles == 1 and not is_closure,
            "iteration_scheme": iteration_scheme,
            "population_solver_converged": (
                final_population_solver_converged
            ),
            "thermal_solver_residual_K": float(thermal_stats.get("residual_K", 0.0)),
            "feedback_to_pump": pump_thermal_refraction,
            "feedback_to_population": population_feedback,
            "feedback_to_seed": True,
            "thermal_field_applied_to_seed": True,
            "seed_to_thermal_feedback": seed_heat_mode == "fixed_point",
            "seed_heat_self_consistent": bool(
                seed_heat_mode == "fixed_point" and converged
            ),
            "seed_heat_closure_iterations": (
                len(history) if seed_heat_mode == "fixed_point" else 0
            ),
            "seed_irreversible_heat_W": float(
                self.grid.integrate_volume(final_seed_heat)
            ),
            # Backward-compatible alias; see seed_irreversible_heat_W.
            "seed_nonlinear_heat_W": float(
                self.grid.integrate_volume(final_seed_heat)
            ),
            "convergence_history": history,
            "nested_population_convergence_history": nested_population_history,
            "seed_probe_history": seed_probe_history,
            "result_recording": copy.deepcopy(self.result_recording),
            "local_spectroscopy_feedback": response_mode == "local_3d",
            "history": history,
            "scope": (
                "Pump absorption, heat and stress/dn-dT are iterated; "
                + (
                    "the Pump propagation responds to thermal refraction; "
                    if pump_thermal_refraction
                    else "the stochastic Pump rays intentionally remain cold; "
                )
                + ("population is re-solved on every iterate; " if population_feedback else "population is intentionally frozen; ")
                +
                "the reported Seed reuses the final inner map's accepted thermal state; "
                + (
                    "the final pulse probe holds the thermal state fixed without recomputing heat; "
                    if frozen_probe_mode else
                    "the next thermal solution verifies closure and is not applied to that Seed; "
                )
                + (
                    "irreversible coherent Seed loss is included as average heat; "
                    if seed_heat_mode == "fixed_point"
                    else "Seed-deposited heat is outside this loop; "
                )
                + (
                    "local temperature/concentration spectroscopy is re-evaluated; "
                    if response_mode == "local_3d"
                    else "local temperature-dependent spectroscopy is disabled; "
                )
                + (
                    "stationary ASE is re-solved with population; "
                    if self.ase_solver.enabled
                    else "ASE is disabled; "
                )
                + "the selected mechanical stress proxy is recomputed from temperature and contributes to delta-n."
            ),
        }
        if frozen_probe_mode:
            last_joint_residuals = {
                key: value for key, value in coupling_stats.items()
                if key.startswith("final_") and "closure_residual" in key
            }
            coupling_stats["last_joint_thermal_residuals"] = last_joint_residuals
            for key in last_joint_residuals:
                coupling_stats[key] = None
            coupling_stats["scope"] += " Final acceptance uses a frozen-thermal pulse probe; the thermal map is not recomputed after that probe."
            coupling_stats["seed_heat_self_consistent"] = False
        self._emit_progress(
            "thermal_optical_fixed_point",
            ("Joint iteration and frozen-thermal pulse verification completed"
             if frozen_probe_mode else
             "Thermal-optical fixed point and final closure check completed"),
            fraction=1.0,
            current=len(history),
            total=maximum,
            metrics={
                "converged": converged,
                "residual": max(
                    final_delta_residual,
                    final_absorbed_residual,
                    final_population_residual,
                    final_spectroscopy_temperature_residual,
                    final_thermal_temperature_residual,
                    final_seed_heat_residual,
                    final_seed_output_residual,
                    final_tensor_residual,
                    final_surface_sag_residual,
                ),
                "target_residual": tolerance,
            },
        )
        return (
            population,
            population_stats,
            pump_result,
            heat,
            temperature,
            thermal_stats,
            stress,
            final_delta_n,
            coupling_stats,
        )

    def _thermo_optic_diagnostics(
        self,
        delta_n: np.ndarray,
        gain_m_inv: np.ndarray,
        signal_result: SignalPropagationResult,
        coupling_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Summarize the phase screen that is actually passed to the seed solver."""

        active = self.crystal_mask > 0.0
        values = np.asarray(delta_n, dtype=float)[active]
        minimum = float(np.min(values)) if values.size else 0.0
        maximum = float(np.max(values)) if values.size else 0.0
        single_axial_optical_path = np.sum(
            np.asarray(delta_n, dtype=float)
            * self.crystal_mask
            * self.grid.z_widths[:, None, None],
            axis=0,
        )
        axial_length = float(np.sum(self.grid.z_widths))
        pass_path_scales = [
            float(row.get("effective_path_length_m", axial_length))
            / max(axial_length, np.finfo(float).tiny)
            for row in signal_result.passes
        ] or [1.0]
        accumulated_path_scale = float(np.sum(pass_path_scales))
        optical_path = single_axial_optical_path * accumulated_path_scale
        transverse_mask = np.any(active, axis=0)
        source_spatio = signal_result.source_spatio_spectral or {}
        wavelength_samples = source_spatio.get("wavelength_samples", [])
        entrance = (
            wavelength_samples[0].get("gain_entrance", {})
            if wavelength_samples
            else source_spatio.get("gain_entrance", {})
        )
        seed_entrance_radius = (
            entrance.get("radius_x_m"),
            entrance.get("radius_y_m"),
        )
        diagnostic_config = copy.deepcopy(
            self.material.get("field_diagnostics", {})
        )
        commercial_app_thermal_lens = self.seed_cfg.get("commercial_app_fast", {}).get(
            "thermal_lens", {}
        )
        transport_thermal_lens = (
            (getattr(signal_result, "gain_transport", None) or {}).get(
                "thermal_lens", {}
            )
        )
        linked_commercial_app_local_fit = (
            getattr(signal_result, "model_used", None) == "commercial_app_fast"
            and str(commercial_app_thermal_lens.get("model", "sampled_phase"))
            == "local_parabolic"
        )
        linked_sampled_local_fit = bool(
            str(transport_thermal_lens.get("model", "sampled_phase"))
            == "local_parabolic"
            and transport_thermal_lens.get("affects_propagation", False)
            and transport_thermal_lens.get("application")
            == "sampled_field_propagation"
        )
        linked_local_fit = linked_commercial_app_local_fit or linked_sampled_local_fit
        if linked_local_fit:
            # The reported thin-lens diagnostic must use the same pupil that
            # actually drives propagation so the two cannot silently drift.
            active_thermal_lens = (
                commercial_app_thermal_lens
                if linked_commercial_app_local_fit
                else transport_thermal_lens
            )
            diagnostic_config.update(
                {
                    "enabled": True,
                    "fit_radius_mode": "manual",
                    "manual_radius_x_m": float(
                        active_thermal_lens["radius_x_m"]
                        if active_thermal_lens.get("radius_x_m") is not None
                        else active_thermal_lens["fit_radius_x_m"]
                    ),
                    "manual_radius_y_m": float(
                        active_thermal_lens["radius_y_m"]
                        if active_thermal_lens.get("radius_y_m") is not None
                        else active_thermal_lens["fit_radius_y_m"]
                    ),
                    "center_x_m": active_thermal_lens.get(
                        "center_x_m",
                        active_thermal_lens.get("fit_center_x_m"),
                    ),
                    "center_y_m": active_thermal_lens.get(
                        "center_y_m",
                        active_thermal_lens.get("fit_center_y_m"),
                    ),
                }
            )
        geometry = self.material.get("geometry", {})
        if diagnostic_config.get("center_x_m") is None:
            diagnostic_config["center_x_m"] = float(
                geometry.get("center_x_m", 0.0)
            )
        if diagnostic_config.get("center_y_m") is None:
            diagnostic_config["center_y_m"] = float(
                geometry.get("center_y_m", 0.0)
            )
        field_diagnostics = parabolic_field_diagnostics(
            delta_n,
            gain_m_inv,
            self.grid,
            self.crystal_mask,
            diagnostic_config,
            seed_entrance_radius,
            float(self.seed_cfg["wavelength_m"]),
        )
        field_diagnostics["single_axial_pass"] = True
        field_diagnostics["reported_pass_path_scales"] = pass_path_scales
        field_diagnostics["reported_accumulated_path_scale"] = accumulated_path_scale
        if linked_local_fit:
            field_diagnostics["linked_transport_model"] = (
                "commercial_app_fast.local_parabolic"
                if linked_commercial_app_local_fit
                else "sampled_field.local_parabolic"
            )
            field_diagnostics["transport_window_synchronized"] = True
            field_diagnostics["scope"] = (
                "The reported parabolic fit uses the same configured ellipse as "
                "the local quadratic thermal lens that affects propagation."
            )
        focal_x: float | None = None
        focal_y: float | None = None
        if (
            np.count_nonzero(transverse_mask) >= 6
            and np.ptp(optical_path[transverse_mask]) > 0.0
        ):
            xx, yy = self.grid.transverse_mesh
            x = xx[transverse_mask]
            y = yy[transverse_mask]
            scale_x = max(float(np.ptp(x)), np.finfo(float).tiny)
            scale_y = max(float(np.ptp(y)), np.finfo(float).tiny)
            x = (x - np.mean(x)) / scale_x
            y = (y - np.mean(y)) / scale_y
            design = np.column_stack(
                [
                    np.ones(np.count_nonzero(transverse_mask)),
                    x, y, x ** 2, y ** 2, x * y,
                ]
            )
            weights = np.sqrt(self.grid.cell_areas[transverse_mask])
            coefficients, _, rank, _ = np.linalg.lstsq(
                design * weights[:, None],
                optical_path[transverse_mask] * weights,
                rcond=None,
            )
            if rank == design.shape[1]:
                curvature_x = float(coefficients[3]) / scale_x ** 2
                curvature_y = float(coefficients[4]) / scale_y ** 2
                if abs(curvature_x) > np.finfo(float).eps:
                    focal_x = -1.0 / (2.0 * curvature_x)
                if abs(curvature_y) > np.finfo(float).eps:
                    focal_y = -1.0 / (2.0 * curvature_y)
        if field_diagnostics.get("enabled", False):
            focal_x = field_diagnostics.get("estimated_thin_lens_focal_x_m")
            focal_y = field_diagnostics.get("estimated_thin_lens_focal_y_m")
            if focal_x is not None:
                focal_x = float(focal_x) / accumulated_path_scale
            if focal_y is not None:
                focal_y = float(focal_y) / accumulated_path_scale
        single_phase_ptv = 2.0 * math.pi * float(
            np.ptp(single_axial_optical_path[transverse_mask])
        ) / float(self.seed_cfg["wavelength_m"]) if np.any(transverse_mask) else 0.0
        phase_ptv = 2.0 * math.pi * float(np.ptp(optical_path[transverse_mask])) / float(
            self.seed_cfg["wavelength_m"]
        ) if np.any(transverse_mask) else 0.0
        return {
            "signal_refractive_index": material_refractive_index(
                self.material, float(self.seed_cfg["wavelength_m"]), role="signal"
            ),
            "pump_refractive_index": material_refractive_index(
                self.material, float(self.pump_cfg["wavelength_m"]), role="pump"
            ),
            "delta_n_min": minimum,
            "delta_n_max": maximum,
            "delta_n_peak_to_valley": maximum - minimum,
            "thermo_optic_phase_peak_to_valley_rad": phase_ptv,
            "thermo_optic_phase_peak_to_valley_single_axial_pass_rad": single_phase_ptv,
            "thermo_optic_accumulated_geometric_path_scale": accumulated_path_scale,
            "thermo_optic_per_pass_geometric_path_scales": pass_path_scales,
            "thermo_optic_path_diagnostic_scope": (
                "straight-axis equivalent with each pass weighted by its configured geometric path length; "
                "the propagated field remains authoritative when walk-off or diffraction samples different transverse delta-n"
            ),
            "estimated_thin_lens_focal_x_m": focal_x,
            "estimated_thin_lens_focal_y_m": focal_y,
            "field_diagnostics": field_diagnostics,
            "phase_screen_applied_to_seed": self.physics["refractive_index_model"] == "thermal_3d",
            "focusing_observable_in_current_model": self.physics["optical_propagation"] != "off",
            "coupling_scope": (
                (coupling_stats or {}).get("scope")
                or (coupling_stats or {}).get("description")
                or "post-pump Seed propagation"
            ),
        }

    def _run_pulse_train_population(
        self,
        delta_n: np.ndarray | None = None,
        record_planes: bool | None = None,
        periodic_initial_population: np.ndarray | None = None,
        periodic_initialization_source: str = "thermal_outer_warm_start",
    ) -> tuple[np.ndarray, dict[str, Any], PumpPropagationResult] | None:
        thermal_delta_n = (
            np.zeros(self.grid.shape, dtype=float)
            if delta_n is None
            else np.asarray(delta_n, dtype=float)
        )
        if thermal_delta_n.shape != self.grid.shape:
            raise ValueError("multiphysics delta_n must match the physical crystal grid")
        should_record = (
            bool(self.physics["record_optical_planes"])
            if record_planes is None
            else bool(record_planes)
        )
        repetition_rate = float(self.sim_cfg.get("repetition_rate_Hz", 0.0))
        density = self.dopant_density_m3
        population = self.levels.initial_population(density, self.grid.shape)
        periodic_cfg = self.sim_cfg.get("periodic_steady_state", {})
        periodic_enabled = bool(periodic_cfg.get("enabled", False))
        initialization_source = "configured_initial_population"
        if periodic_enabled and periodic_initial_population is not None:
            proposed = np.asarray(periodic_initial_population, dtype=float)
            if proposed.shape != population.shape or not np.all(np.isfinite(proposed)):
                raise ValueError(
                    "periodic steady-state warm-start population must be finite and match the level/grid shape"
                )
            population, violation, _ = self._population_from_fraction_candidate(
                self._population_fractions(proposed)
            )
            maximum_violation = float(
                periodic_cfg.get("maximum_population_fraction_violation", 0.05)
            )
            if violation > maximum_violation:
                raise ValueError(
                    "periodic steady-state warm-start population violates non-negativity or local conservation"
                )
            initialization_source = str(periodic_initialization_source)
        if repetition_rate > 0.0 and periodic_enabled:
            method = str(periodic_cfg.get("method", "pulse_map"))
            if method == "cycle_average":
                return self._run_cycle_averaged_steady_state(
                    population,
                    density,
                    repetition_rate,
                    periodic_cfg,
                    thermal_delta_n,
                    should_record,
                    initialization_source=initialization_source,
                )
            if method == "reduced_pulse_map":
                exact_cfg = copy.deepcopy(periodic_cfg)
                exact_cfg["method"] = "pulse_map"
                exact_cfg["pulse_map_initialization"] = "reduced_pulse_map"
                return self._run_to_periodic_steady_state(
                    population,
                    density,
                    repetition_rate,
                    exact_cfg,
                    thermal_delta_n,
                    should_record,
                    initialization_source=initialization_source,
                )
            if method != "pulse_map":
                raise ValueError(f"Unknown periodic steady-state method {method!r}")
            return self._run_to_periodic_steady_state(
                population,
                density,
                repetition_rate,
                periodic_cfg,
                thermal_delta_n,
                should_record,
                initialization_source=initialization_source,
            )
        if self.physics["population_dynamics"] != "transient":
            return None
        if repetition_rate <= 0.0:
            temporal_type = str(self.pump_cfg.get("temporal", {}).get("type", "cw"))
            duration = float(self.sim_cfg.get("pump_duration_s", 0.0))
            if temporal_type == "cw" or duration <= 0.0:
                return None
            waveform_cfg = copy.deepcopy(self.pump_cfg.get("temporal", {}))
            waveform_cfg["period_s"] = duration
            waveform_cfg["sample_count"] = int(self.sim_cfg["pump_time_steps"])
            waveform = periodic_pump_waveform(
                waveform_cfg,
                float(self.pump_cfg["power_W"]),
                0.0,
                int(self.sim_cfg["pump_time_steps"]),
            )
            population, pump_result = self._advance_sampled_pump(
                population,
                waveform,
                should_record,
                thermal_delta_n,
            )
            return population, {
                "dynamics": "transient_waveform",
                "iterations": len(waveform.power_W),
                "simulated_pump_duration_s": duration,
                "waveform": temporal_type,
                "converged": None,
            }, pump_result
        cycles = int(self.sim_cfg.get("pulse_train_cycles", 1))
        pulse_train_initializer = str(
            self.sim_cfg.get(
                "pulse_train_initialization",
                "configured_initial_population",
            )
        )
        initializer_stats: dict[str, Any] = {
            "requested": pulse_train_initializer,
            "attempted": False,
            "used": False,
            "reason": "configured_initial_population",
        }
        if pulse_train_initializer in {"cycle_average", "auto"}:
            initializer_cfg = copy.deepcopy(periodic_cfg)
            initializer_cfg["pulse_map_initialization"] = (
                "cycle_average"
                if pulse_train_initializer == "cycle_average"
                else "auto"
            )
            population, initialization_source, initializer_stats = (
                self._pulse_map_initial_guess(
                    population,
                    density,
                    repetition_rate,
                    initializer_cfg,
                    thermal_delta_n,
                    initialization_source,
                )
            )
        cycle_timing = self._pulse_map_cycle_timing(repetition_rate)
        configured_first_pulse_delay = self.sim_cfg.get(
            "first_pulse_delay_s"
        )
        first_pre_seed_pump_duration_s = (
            float(cycle_timing["population_dark_duration_s"])
            if configured_first_pulse_delay is None
            else float(configured_first_pulse_delay)
        )
        first_pre_seed_free_carrier_duration_s = (
            float(cycle_timing["free_carrier_dark_duration_s"])
            if configured_first_pulse_delay is None
            else float(configured_first_pulse_delay)
        )
        last_pump: PumpPropagationResult | None = None
        free_carrier_enabled = self._free_carrier_periodic_state_enabled()
        free_carrier_state = np.zeros(self.grid.shape, dtype=float)
        pre_seed_free_carrier = free_carrier_state.copy()
        pulse_history: list[dict[str, Any]] = []
        self._emit_progress(
            "pulse_train",
            "Beginning configured transient Pump-period / Seed-pulse train",
            fraction=0.0,
            current=0,
            total=cycles,
            metrics={
                "pulse_total": cycles,
                "repetition_rate_Hz": repetition_rate,
            },
        )
        for cycle in range(cycles):
            pre_seed_pump_duration_s = (
                first_pre_seed_pump_duration_s
                if cycle == 0
                else float(cycle_timing["population_dark_duration_s"])
            )
            pre_seed_free_carrier_duration_s = (
                first_pre_seed_free_carrier_duration_s
                if cycle == 0
                else float(cycle_timing["free_carrier_dark_duration_s"])
            )
            cycle_start_fraction = cycle / max(cycles, 1)
            cycle_stop_fraction = (cycle + 1) / max(cycles, 1)
            pump_stop_fraction = cycle_start_fraction + 0.32 * (
                cycle_stop_fraction - cycle_start_fraction
            )
            seed_stop_fraction = cycle_start_fraction + 0.94 * (
                cycle_stop_fraction - cycle_start_fraction
            )
            with self._progress_scope(
                cycle_start_fraction,
                pump_stop_fraction,
            ):
                population, last_pump = self._advance_one_pump_period(
                    population,
                    record_planes=should_record and cycle == cycles - 1,
                    delta_n=thermal_delta_n,
                    dark_duration_s=pre_seed_pump_duration_s,
                )
            cycle_ase = self._last_ase_result
            if free_carrier_enabled:
                pre_seed_free_carrier = self._decay_free_carrier_state(
                    free_carrier_state,
                    pre_seed_free_carrier_duration_s,
                )
                self._set_seed_initial_free_carrier(pre_seed_free_carrier)
            pulse_metrics: dict[str, Any] = {
                "pulse_index": cycle + 1,
                "pulse_total": cycles,
                "repetition_rate_Hz": repetition_rate,
                "pulse_time_s": (
                    first_pre_seed_pump_duration_s
                    + cycle * cycle_timing["repetition_period_s"]
                ),
                "pre_seed_pump_duration_s": pre_seed_pump_duration_s,
                "pre_seed_free_carrier_decay_duration_s": (
                    pre_seed_free_carrier_duration_s
                    if free_carrier_enabled
                    else None
                ),
                "absorbed_pump_power_W": float(
                    last_pump.stats.get("absorbed_power_W", 0.0)
                ),
                "ase_interpulse_diagnostics": (
                    None
                    if cycle_ase is None
                    else {
                        "model": cycle_ase.stats.get("model"),
                        "escaped_power_W": float(cycle_ase.escaped_power_W),
                        "energy_ledger_W": copy.deepcopy(
                            cycle_ase.stats.get("energy_ledger_W", {})
                        ),
                    }
                ),
                **self._population_progress_metrics(population),
            }
            if cycle + 1 < cycles:
                with self._progress_scope(
                    pump_stop_fraction,
                    seed_stop_fraction,
                ):
                    previous_seed_context = getattr(
                        self, "_seed_progress_context", {}
                    )
                    self._seed_progress_context = {
                        "pulse_index": cycle + 1,
                        "pulse_total": cycles,
                    }
                    try:
                        trial = self._propagate_seed_with_progress(
                            population,
                            thermal_delta_n,
                            record_planes=False,
                        )
                    finally:
                        self._seed_progress_context = previous_seed_context
                if _runtime_optional_fallback(trial):
                    raise RuntimeError(
                        "the requested Seed backend failed inside the transient "
                        "pulse train; continuing with a fallback pulse would mix "
                        "different depletion maps within one physical history"
                    )
                population = trial.population_m3
                pulse_metrics.update(
                    {
                        "seed_pulse_applied": True,
                        **self._signal_progress_metrics(
                            trial,
                            include_spectrum_preview=bool(
                                cycle == 0 or (cycle + 1) % 5 == 0
                            ),
                        ),
                        **self._population_progress_metrics(population),
                    }
                )
                if free_carrier_enabled:
                    if trial.free_carrier_density_m3 is None:
                        raise RuntimeError(
                            "coherent pulse-train propagation did not export its free-carrier state"
                        )
                    free_carrier_state = np.asarray(
                        trial.free_carrier_density_m3, dtype=float
                    ).copy()
            else:
                pulse_metrics["seed_pulse_applied"] = False
                pulse_metrics["awaiting_reported_seed_pulse"] = True
            pulse_history.append(copy.deepcopy(pulse_metrics))
            self._emit_progress(
                "pulse_train",
                f"Transient pulse period {cycle + 1}/{cycles} completed",
                fraction=cycle_stop_fraction,
                current=cycle + 1,
                total=cycles,
                metrics=pulse_metrics,
            )
        if free_carrier_enabled:
            # The top-level reported Seed pulse follows the final Pump period.
            self._set_seed_initial_free_carrier(pre_seed_free_carrier)
        assert last_pump is not None
        return population, {
            "dynamics": "transient_pulse_train",
            "iterations": cycles * int(self.pump_cfg.get("temporal", {}).get("sample_count", self.sim_cfg["pump_time_steps_per_cycle"])),
            "pulse_train_cycles": cycles,
            "repetition_rate_Hz": repetition_rate,
            "simulated_duration_s": (
                first_pre_seed_pump_duration_s
                + float(cycle_timing["seed_resolved_elapsed_time_s"])
                + (cycles - 1) * float(cycle_timing["repetition_period_s"])
            ),
            "first_pulse_delay_s": configured_first_pulse_delay,
            "first_pulse_time_s": first_pre_seed_pump_duration_s,
            "first_pulse_timing": (
                "configured_delay_from_initial_population"
                if configured_first_pulse_delay is not None
                else "legacy_one_dark_window"
            ),
            "cycle_timing": cycle_timing,
            "seed_depletion_events_before_reported_pulse": max(cycles - 1, 0),
            "initialization_source": initialization_source,
            "pulse_train_initializer": initializer_stats,
            "pulse_history": pulse_history,
            "ase_coupled": self._last_ase_result is not None,
            "ase_interpulse_coupling": (
                "quasi_stationary_post_seed_boundary_held_through_pump_dark_interval"
                if self._last_ase_result is not None
                else "disabled"
            ),
            "ase_refreshes_per_cycle": (
                1 if self._last_ase_result is not None else 0
            ),
            "ase_during_resolved_seed_pulse": (
                "omitted"
                if self._last_ase_result is not None
                else "not_applicable"
            ),
            "free_carrier_state_carried_between_pulses": free_carrier_enabled,
            "reported_pulse_initial_free_carrier_density_max_m3": float(
                np.max(pre_seed_free_carrier)
            ),
            "converged": None,
        }, last_pump

    def _uses_cw_average_for_configured_pump_waveform(self) -> bool:
        """Return whether a shaped Pump is intentionally reduced to average CW power."""

        temporal = self.pump_cfg.get("temporal", {})
        if str(temporal.get("type", "cw")) == "cw":
            return False
        repetition_rate = float(self.sim_cfg.get("repetition_rate_Hz", 0.0))
        periodic_enabled = bool(
            self.sim_cfg.get("periodic_steady_state", {}).get("enabled", False)
        )
        waveform_is_sampled = bool(
            (repetition_rate > 0.0 and periodic_enabled)
            or (
                self.physics["population_dynamics"] == "transient"
                and (
                    repetition_rate > 0.0
                    or float(self.sim_cfg.get("pump_duration_s", 0.0)) > 0.0
                )
            )
        )
        return not waveform_is_sampled

    def _run_cycle_averaged_steady_state(
        self,
        population: np.ndarray,
        density: np.ndarray,
        repetition_rate: float,
        cfg: dict[str, Any],
        delta_n: np.ndarray,
        record_planes: bool,
        *,
        initialization_source: str = "configured_initial_population",
    ) -> tuple[np.ndarray, dict[str, Any], PumpPropagationResult]:
        """Solve a high-repetition Pump+Seed average-rate fixed point.

        Unlike the exact pulse map, an iteration here is a nonlinear
        self-consistency update of stationary equations, not one elapsed laser
        period.  The Seed is represented by average spectral photon flux
        ``pulse_energy * repetition_rate``. This is the cycle-averaged amplifier
        physics and is valid only when one pulse causes a small population
        jump.
        """

        if self.levels.nonlinear:
            raise ValueError(
                "cycle-average steady state currently supports linear level transitions only; "
                "use the exact pulse-map steady state for upconversion or cross-relaxation"
            )
        minimum = int(cfg.get("average_minimum_iterations", 2))
        maximum = int(cfg.get("average_maximum_iterations", 40))
        tolerance = float(cfg.get("relative_tolerance", 1e-5))
        relaxation = float(cfg.get("average_relaxation", 0.5))
        accelerator = SafeguardedAnderson(cfg.get("acceleration", {}))
        maximum_fraction_violation = float(
            cfg.get("maximum_population_fraction_violation", 0.05)
        )
        configured_pump_power = float(self.pump_cfg["power_W"])
        waveform = periodic_pump_waveform(
            self.pump_cfg.get("temporal", {}),
            configured_pump_power,
            repetition_rate,
            int(self.sim_cfg["pump_time_steps_per_cycle"]),
        )
        absolute_file = bool(
            str(self.pump_cfg.get("temporal", {}).get("type", "cw")) == "file"
            and str(
                self.pump_cfg.get("temporal", {}).get("value_mode", "relative")
            )
            == "absolute_W"
        )
        pump_scale = (
            waveform.average_power_W
            if configured_pump_power == 0.0 and absolute_file
            else 0.0
            if configured_pump_power == 0.0
            else waveform.average_power_W / configured_pump_power
        )
        previous_output: float | None = None
        history: list[dict[str, Any]] = []
        converged = False
        final_pump: PumpPropagationResult | None = None
        final_probe: SignalPropagationResult | None = None
        self._emit_progress(
            "cycle_average",
            "Beginning cycle-averaged Pump + Seed fixed point",
            fraction=0.0,
            current=0,
            total=maximum,
            metrics={
                "target_residual": tolerance,
                "iteration_kind": (
                    "pulse_map_initializer"
                    if initialization_source
                    == "pulse_map_cycle_average_initializer"
                    else "cycle_average_fixed_point"
                ),
                "initialization_source": initialization_source,
                "fixed_point_iteration": 0,
                "fixed_point_total": maximum,
                "pump_average_power_W": float(waveform.average_power_W),
                "seed_input_average_power_W": (
                    float(self.seed_cfg["pulse_energy_J"]) * repetition_rate
                ),
            },
        )
        for iteration in range(1, maximum + 1):
            iteration_start_fraction = (
                0.92 * (iteration - 1) / max(maximum, 1)
            )
            iteration_stop_fraction = 0.92 * iteration / max(maximum, 1)
            seed_start_fraction = iteration_start_fraction + 0.2 * (
                iteration_stop_fraction - iteration_start_fraction
            )
            self._emit_progress(
                "cycle_average",
                (
                    "Cycle-average Pump probe for fixed-point iteration "
                    f"{iteration} (limit {maximum})"
                ),
                fraction=iteration_start_fraction,
                current=iteration - 1,
                total=maximum,
                metrics={
                    "fixed_point_iteration": iteration,
                    "fixed_point_total": maximum,
                    "seed_substage": "cycle_average_pump_probe",
                },
            )
            pump = self.pump_solver.propagate(
                population,
                False,
                power_scale=pump_scale,
                delta_n=delta_n,
            )
            with self._progress_scope(
                seed_start_fraction,
                iteration_stop_fraction,
            ):
                previous_seed_details = self._seed_progress_details_enabled
                previous_seed_context = getattr(
                    self, "_seed_progress_context", {}
                )
                self._seed_progress_details_enabled = True
                self._seed_progress_context = {
                    "fixed_point_iteration": iteration,
                    "fixed_point_total": maximum,
                }
                try:
                    probe = self.average_signal_solver.propagate_average(
                        population,
                        delta_n,
                        repetition_rate,
                        False,
                    )
                finally:
                    self._seed_progress_details_enabled = previous_seed_details
                    self._seed_progress_context = previous_seed_context
            spectral_intensity = probe.average_spectral_intensity_W_m2
            if spectral_intensity is None or probe.wavelengths_m is None:
                raise RuntimeError("cycle-average Seed probe did not return spectral intensity")
            ase = self._solve_ase(population)
            if ase is not None:
                stationary_spectral_intensity = concatenate_spectral_intensities(
                    [spectral_intensity, ase.spectral_intensity_W_m2]
                )
                stationary_wavelengths = np.concatenate(
                    [probe.wavelengths_m, ase.wavelengths_m]
                )
            else:
                stationary_spectral_intensity = spectral_intensity
                stationary_wavelengths = probe.wavelengths_m
            stationary = self.levels.steady_state(
                density,
                pump.intensity_W_m2,
                float(self.pump_cfg["wavelength_m"]),
                pump_wavelengths_m=self.pump_solver.spectrum.coordinate,
                pump_power_weights=self.pump_solver.spectrum.weights,
                pump_spectral_intensity_W_m2=pump.spectral_intensity_W_m2,
                signal_spectral_intensity_W_m2=stationary_spectral_intensity,
                signal_wavelengths_m=stationary_wavelengths,
            )
            stationary *= self.crystal_mask[None, ...]
            base_update = (
                (1.0 - relaxation) * population + relaxation * stationary
            )
            population_residual = self._population_residual_relative(
                stationary, population
            )
            updated, acceleration_row = self._accelerated_population_update(
                accelerator,
                iteration,
                population,
                base_update,
                maximum_fraction_violation,
            )
            output_power = float(probe.system_output_energy_J) * repetition_rate
            if previous_output is None:
                output_residual = float("inf")
            else:
                output_residual = abs(output_power - previous_output) / max(
                    abs(output_power),
                    abs(previous_output),
                    np.finfo(float).tiny,
                )
            history.append(
                {
                    "iteration": iteration,
                    "average_output_power_W": output_power,
                    "population_residual_relative": population_residual,
                    "output_residual_relative": output_residual,
                    **acceleration_row,
                }
            )
            self._emit_progress(
                "cycle_average",
                (
                    f"Cycle-average fixed-point iteration {iteration} completed "
                    f"(limit {maximum})"
                ),
                fraction=0.92 * iteration / max(maximum, 1),
                current=iteration,
                total=maximum,
                metrics={
                    "population_residual": population_residual,
                    "output_residual": output_residual,
                    "target_residual": tolerance,
                    "output_power_W": output_power,
                    "iteration_kind": (
                        "pulse_map_initializer"
                        if initialization_source
                        == "pulse_map_cycle_average_initializer"
                        else "cycle_average_fixed_point"
                    ),
                    "initialization_source": initialization_source,
                    "fixed_point_iteration": iteration,
                    "fixed_point_total": maximum,
                    **self._population_progress_metrics(updated),
                },
            )
            previous_output = output_power
            final_pump = pump
            final_probe = probe
            if iteration >= minimum and max(population_residual, output_residual) <= tolerance:
                converged = True
                break
            population = updated

        if final_pump is None or final_probe is None:
            raise RuntimeError("cycle-average steady-state loop did not execute")

        # Re-evaluate both fluxes on the converged population.  This is a
        # stationary diagnostic pass and does not advance physical time.
        self._emit_progress(
            "cycle_average",
            "Beginning final cycle-average closure probe",
            fraction=0.92,
            current=len(history),
            total=maximum,
            metrics={
                "seed_substage": "cycle_average_closure_pump",
                "fixed_point_iteration": len(history),
                "fixed_point_total": maximum,
                "closure_probe": True,
            },
        )
        final_pump = self.pump_solver.propagate(
            population,
            record_planes,
            power_scale=pump_scale,
            delta_n=delta_n,
        )
        with self._progress_scope(0.94, 0.995):
            previous_seed_details = self._seed_progress_details_enabled
            previous_seed_context = getattr(self, "_seed_progress_context", {})
            self._seed_progress_details_enabled = True
            self._seed_progress_context = {
                "fixed_point_iteration": len(history),
                "fixed_point_total": maximum,
                "closure_probe": True,
            }
            try:
                final_probe = self.average_signal_solver.propagate_average(
                    population,
                    delta_n,
                    repetition_rate,
                    False,
                )
            finally:
                self._seed_progress_details_enabled = previous_seed_details
                self._seed_progress_context = previous_seed_context
        final_ase = self._solve_ase(population)
        final_spectral_intensity = final_probe.average_spectral_intensity_W_m2
        if final_spectral_intensity is None or final_probe.wavelengths_m is None:
            raise RuntimeError(
                "cycle-average final closure probe did not return spectral intensity"
            )
        if final_ase is not None:
            closure_spectral_intensity = concatenate_spectral_intensities(
                [final_spectral_intensity, final_ase.spectral_intensity_W_m2]
            )
            closure_wavelengths = np.concatenate(
                [final_probe.wavelengths_m, final_ase.wavelengths_m]
            )
        else:
            closure_spectral_intensity = final_spectral_intensity
            closure_wavelengths = final_probe.wavelengths_m
        closure_stationary = self.levels.steady_state(
            density,
            final_pump.intensity_W_m2,
            float(self.pump_cfg["wavelength_m"]),
            pump_wavelengths_m=self.pump_solver.spectrum.coordinate,
            pump_power_weights=self.pump_solver.spectrum.weights,
            pump_spectral_intensity_W_m2=final_pump.spectral_intensity_W_m2,
            signal_spectral_intensity_W_m2=closure_spectral_intensity,
            signal_wavelengths_m=closure_wavelengths,
        )
        closure_stationary *= self.crystal_mask[None, ...]
        final_population_residual = self._population_residual_relative(
            closure_stationary,
            population,
        )
        final_output_power = (
            float(final_probe.system_output_energy_J) * repetition_rate
        )
        final_output_residual = abs(
            final_output_power - float(previous_output)
        ) / max(
            abs(final_output_power),
            abs(float(previous_output)),
            np.finfo(float).tiny,
        )
        converged = bool(
            len(history) >= minimum
            and max(final_population_residual, final_output_residual) <= tolerance
        )
        history[-1]["final_closure_population_residual_relative"] = (
            final_population_residual
        )
        history[-1]["final_closure_output_residual_relative"] = (
            final_output_residual
        )
        self._last_periodic_boundary_population = population.copy()
        self._emit_progress(
            "cycle_average",
            "Cycle-average fixed point and closure probe completed",
            fraction=1.0,
            current=len(history),
            total=maximum,
            metrics={
                "converged": converged,
                "population_residual": final_population_residual,
                "output_residual": final_output_residual,
                "target_residual": tolerance,
                "output_power_W": final_output_power,
                "iteration_kind": (
                    "pulse_map_initializer"
                    if initialization_source
                    == "pulse_map_cycle_average_initializer"
                    else "cycle_average_fixed_point"
                ),
                "initialization_source": initialization_source,
                "fixed_point_iteration": len(history),
                "fixed_point_total": maximum,
                **self._population_progress_metrics(population),
            },
        )
        return population, {
            "dynamics": "cycle_averaged_steady_state",
            "method": "stationary_average_photon_flux",
            "iterations": len(history),
            "converged": converged,
            "repetition_rate_Hz": repetition_rate,
            "equivalent_seed_input_average_power_W": (
                float(self.seed_cfg["pulse_energy_J"]) * repetition_rate
            ),
            "average_probe_output_power_W": (
                float(final_probe.system_output_energy_J) * repetition_rate
            ),
            "average_probe_warnings": list(final_probe.warnings),
            "average_probe_transport": dict(final_probe.gain_transport or {}),
            "ase_coupled": self._last_ase_result is not None,
            "relative_tolerance": tolerance,
            "relaxation": relaxation,
            "initialization_source": initialization_source,
            "acceleration": accelerator.diagnostics(),
            "final_population_residual_relative": final_population_residual,
            "final_output_residual_relative": final_output_residual,
            "convergence_history": history,
            "approximation": (
                "one Seed pulse per period is replaced by its average spectral photon flux; "
                "the final selected pulse solver is then evaluated once on the stationary population"
            ),
        }, final_pump

    def _maximum_population_memory_lifetime_s(self) -> float:
        """Return a conservative lifetime scale for periodic-map stiffness."""

        configured = [
            float(self.levels.cfg.get("upper_state_lifetime_s", 0.0)),
            float(self.levels.cfg.get("pump_level_lifetime_s", 0.0)),
            float(self.levels.cfg.get("lower_level_lifetime_s", 0.0)),
        ]
        configured.extend(
            1.0 / float(decay.rate_s)
            for decay in self.levels.decays
            if float(decay.rate_s) > 0.0
        )
        finite = [value for value in configured if np.isfinite(value) and value > 0.0]
        return max(finite, default=0.0)

    def _pulse_map_initial_guess(
        self,
        population: np.ndarray,
        density: np.ndarray,
        repetition_rate: float,
        cfg: dict[str, Any],
        delta_n: np.ndarray,
        initialization_source: str,
    ) -> tuple[np.ndarray, str, dict[str, Any]]:
        """Optionally initialize the exact map with an accelerated fixed point.

        ``reduced_pulse_map`` is an initializer, never a reported periodic
        solution.  It may itself use the cycle-average fixed point, then hands
        its post-Seed boundary population to the configured exact pulse map.
        The reduced initializer is consumed once per top-level run; later
        thermal-optical outer iterations warm-start from the preceding exact
        map instead of repeating the proxy.
        """

        requested = str(cfg.get("pulse_map_initialization", "auto"))
        lifetime_periods = repetition_rate * self._maximum_population_memory_lifetime_s()
        diagnostics: dict[str, Any] = {
            "requested": requested,
            "attempted": False,
            "used": False,
            "reason": "not_requested",
            "maximum_lifetime_periods": lifetime_periods,
        }
        if requested == "reduced_pulse_map":
            if self._reduced_pulse_map_initializer_consumed:
                diagnostics["reason"] = "reduced_pulse_map_initializer_already_consumed"
                return population, initialization_source, diagnostics
            if self.reduced_pulse_map_signal_solver is None:
                diagnostics["reason"] = "reduced_pulse_map_solver_unavailable"
                return population, initialization_source, diagnostics
            self._reduced_pulse_map_initializer_consumed = True
            reduced_cfg = copy.deepcopy(cfg)
            reduced_cfg["method"] = "pulse_map"
            reduced_cfg["pulse_map_initialization"] = "cycle_average"
            diagnostics["attempted"] = True
            diagnostics["progress_fraction"] = 0.45
            try:
                with self._progress_scope(0.0, 0.45):
                    _, reduced_stats, _ = self._run_to_periodic_steady_state(
                        population.copy(),
                        density,
                        repetition_rate,
                        reduced_cfg,
                        delta_n,
                        False,
                        initialization_source=initialization_source,
                        reduced_seed_proxy=True,
                    )
                reduced_boundary = self._last_periodic_boundary_population
                if reduced_boundary is None:
                    raise RuntimeError(
                        "reduced pulse-map initializer did not retain its post-Seed boundary population"
                    )
            except (
                ValueError,
                RuntimeError,
                FloatingPointError,
                np.linalg.LinAlgError,
            ) as exc:
                diagnostics.update(
                    {
                        "reason": "reduced_pulse_map_failed_fallback_to_exact_map",
                        "failure": f"{type(exc).__name__}: {exc}",
                    }
                )
                return population, initialization_source, diagnostics
            diagnostics.update(
                {
                    "used": True,
                    "reason": "reduced_pulse_map_completed",
                    "converged": bool(reduced_stats.get("converged", False)),
                    "iterations": int(reduced_stats.get("cycles_completed", 0)),
                    "final_population_residual_relative": float(
                        reduced_stats.get(
                            "final_population_residual_relative", float("inf")
                        )
                    ),
                    "final_output_residual_relative": float(
                        reduced_stats.get(
                            "final_output_residual_relative", float("inf")
                        )
                    ),
                    "reduced_map": reduced_stats,
                }
            )
            return (
                np.asarray(reduced_boundary, dtype=float).copy(),
                "reduced_pulse_map_initializer",
                diagnostics,
            )
        if initialization_source in {
            "thermal_outer_warm_start",
            "pump_sweep_previous_point",
        }:
            diagnostics["reason"] = f"{initialization_source}_preferred"
            return population, initialization_source, diagnostics
        if requested == "initial_population":
            return population, initialization_source, diagnostics
        if self.levels.nonlinear:
            diagnostics["reason"] = "nonlinear_level_processes_not_supported"
            return population, initialization_source, diagnostics
        if self.levels.cfg.get("gain_override_m_inv") is not None:
            diagnostics["reason"] = "gain_override_has_no_physical_average_reservoir"
            return population, initialization_source, diagnostics
        if self._free_carrier_periodic_state_enabled():
            carrier = self.material.get("strong_field", {}).get(
                "free_carrier", {}
            )
            residual_fraction = math.exp(
                -1.0
                / (
                    repetition_rate
                    * float(carrier.get("lifetime_s", 1e-9))
                )
            )
            residual_limit = float(
                cfg.get(
                    "maximum_unmodeled_free_carrier_residual_fraction",
                    1e-4,
                )
            )
            if residual_fraction > residual_limit:
                diagnostics.update(
                    {
                        "reason": (
                            "cycle_average_initializer_cannot_carry_"
                            "free_carrier_memory"
                        ),
                        "free_carrier_residual_fraction": residual_fraction,
                        "free_carrier_residual_limit": residual_limit,
                    }
                )
                return population, initialization_source, diagnostics
        if requested == "auto":
            threshold = float(
                cfg.get(
                    "cycle_average_initializer_minimum_lifetime_periods", 20.0
                )
            )
            if lifetime_periods < threshold:
                diagnostics["reason"] = "map_not_stiff_enough_for_initializer"
                diagnostics["automatic_threshold_lifetime_periods"] = threshold
                return population, initialization_source, diagnostics
        try:
            diagnostics["attempted"] = True
            with self._progress_scope(0.0, 0.15):
                initialized, stats, _ = self._run_cycle_averaged_steady_state(
                    population.copy(),
                    density,
                    repetition_rate,
                    cfg,
                    delta_n,
                    False,
                    initialization_source="pulse_map_cycle_average_initializer",
                )
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as exc:
            diagnostics.update(
                {
                    "reason": "cycle_average_failed_fallback_to_initial_population",
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
            return population, initialization_source, diagnostics
        diagnostics.update(
            {
                "used": True,
                "reason": "cycle_average_completed",
                "converged": bool(stats.get("converged", False)),
                "iterations": int(stats.get("iterations", 0)),
                "final_population_residual_relative": float(
                    stats.get("final_population_residual_relative", float("inf"))
                ),
                "final_output_residual_relative": float(
                    stats.get("final_output_residual_relative", float("inf"))
                ),
            }
        )
        return initialized, "cycle_average_initializer", diagnostics

    def _run_to_periodic_steady_state(
        self,
        population: np.ndarray,
        density: np.ndarray,
        repetition_rate: float,
        cfg: dict[str, Any],
        delta_n: np.ndarray,
        record_planes: bool,
        *,
        initialization_source: str = "configured_initial_population",
        reduced_seed_proxy: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any], PumpPropagationResult]:
        """Iterate a Pump-period/Seed-depletion map to a periodic fixed point.

        ``reduced_seed_proxy`` replaces only the repeated configured Seed solve
        with the selected reduced phase-insensitive or axisymmetric coherent
        propagator. Pump-period dynamics, FN saturation, local population
        depletion, and the fixed-point convergence test remain discrete. The
        caller still performs one configured full-field propagation from the
        converged pre-Seed boundary state and validates the proxy defect
        afterwards.
        """

        map_signal_solver = (
            self.reduced_pulse_map_signal_solver
            if reduced_seed_proxy
            else self.signal_solver
        )
        if map_signal_solver is None:
            raise RuntimeError(
                "reduced_pulse_map requested but its Seed proxy was not initialized"
            )
        map_stage = "reduced_pulse_map" if reduced_seed_proxy else "pulse_map"
        map_label = "Reduced" if reduced_seed_proxy else "Exact"

        self._emit_progress(
            "pulse_map_initialization",
            (
                "Preparing the reduced discrete pulse map and cycle-average initializer"
                if reduced_seed_proxy
                else "Preparing the exact pulse map and optional cycle-average initializer"
            ),
            fraction=0.0,
            metrics={
                "initialization_requested": str(
                    cfg.get("pulse_map_initialization", "auto")
                )
            },
        )
        population, initialization_source, initializer_stats = (
            self._pulse_map_initial_guess(
                population,
                density,
                repetition_rate,
                cfg,
                delta_n,
                initialization_source,
            )
        )
        pulse_progress_start = float(
            initializer_stats.get(
                "progress_fraction",
                0.15 if bool(initializer_stats.get("attempted", False)) else 0.0,
            )
        )
        self._emit_progress(
            "pulse_map_initialization",
            "Pulse-map initialization completed",
            fraction=pulse_progress_start,
            metrics={
                "initialization_source": initialization_source,
                "initializer_used": bool(initializer_stats.get("used", False)),
                "initializer_converged": initializer_stats.get("converged"),
                "initializer_iterations": initializer_stats.get("iterations"),
                "population_residual": initializer_stats.get(
                    "final_population_residual_relative"
                ),
                "output_residual": initializer_stats.get(
                    "final_output_residual_relative"
                ),
                **self._population_progress_metrics(population),
            },
        )
        reduced_convergence_cfg = (
            dict(cfg.get("reduced_pulse_map", {}))
            if reduced_seed_proxy
            else {}
        )
        minimum = int(cfg.get("minimum_cycles", 3))
        maximum = int(cfg.get("maximum_cycles", 1000))
        tolerance = float(
            reduced_convergence_cfg.get(
                "relative_tolerance",
                cfg.get("relative_tolerance", 1e-5),
            )
        )
        required = int(
            reduced_convergence_cfg.get(
                "consecutive_cycles",
                cfg.get("consecutive_cycles", 3),
            )
        )
        expected_open_ended_cycles = max(minimum + required - 1, 3)

        def open_ended_cycle_fraction(completed_cycles: int) -> float:
            # The configured maximum is a safety cap, not a prediction.  Using
            # (for example) 5/1000 as UI progress made a healthy converging map
            # appear frozen.  An asymptotic estimate advances visibly while
            # reserving the final jump for an actually converged/finished map.
            return min(
                0.94,
                1.0
                - math.exp(
                    -max(int(completed_cycles), 0)
                    / float(expected_open_ended_cycles)
                ),
            )
        joint_map_state = getattr(self, "_joint_map_state", None)
        previous_output: float | None = (
            None if joint_map_state is None else joint_map_state.get("output_energy_J")
        )
        stable_count = 0
        converged = False
        last_pump: PumpPropagationResult | None = None
        last_pre_seed = population.copy()
        history: list[dict[str, Any]] = []
        runtime_optional_fallback_cycles: list[int] = []
        cycles_completed = 0
        accelerator = SafeguardedAnderson(cfg.get("acceleration", {}))
        maximum_fraction_violation = float(
            cfg.get("maximum_population_fraction_violation", 0.05)
        )
        last_post_seed = population.copy()
        free_carrier_enabled = self._free_carrier_periodic_state_enabled()
        cycle_timing = self._pulse_map_cycle_timing(repetition_rate)
        free_carrier_state = (
            np.zeros(self.grid.shape, dtype=float)
            if joint_map_state is None or "free_carrier" not in joint_map_state
            else joint_map_state["free_carrier"].copy()
        )
        last_pre_seed_free_carrier = free_carrier_state.copy()
        last_post_seed_free_carrier = free_carrier_state.copy()
        previous_free_carrier_absorbed_energy_J: float | None = (
            None if joint_map_state is None else joint_map_state.get("carrier_absorbed_energy_J")
        )
        carrier_convergence_metric = str(
            cfg.get("free_carrier_convergence", {}).get(
                "metric", "optical_effect"
            )
        )
        self._emit_progress(
            map_stage,
            f"Beginning {map_label.lower()} Pump-period / Seed-pulse map",
            fraction=pulse_progress_start,
            current=0,
            total=maximum,
            metrics={
                "target_residual": tolerance,
                "consecutive_cycles_required": required,
                "initialization_source": initialization_source,
            },
        )
        record_targets = [
            candidate
            for candidate in (
                map_signal_solver,
                getattr(map_signal_solver, "primary", None),
                getattr(map_signal_solver, "fallback", None),
            )
            if candidate is not None and hasattr(candidate, "physics")
        ]
        original_record = [bool(target.physics.get("record_optical_planes", False)) for target in record_targets]
        original_cycle_scalar_diagnostics = [
            target.physics.get("_record_cycle_scalar_diagnostics")
            for target in record_targets
        ]
        try:
            for target in record_targets:
                target.physics["record_optical_planes"] = False
                target.physics["_record_cycle_scalar_diagnostics"] = True
            for cycle in range(1, maximum + 1):
                cycle_start_fraction = (
                    pulse_progress_start
                    + (0.96 - pulse_progress_start)
                    * open_ended_cycle_fraction(cycle - 1)
                )
                cycle_stop_fraction = (
                    pulse_progress_start
                    + (0.96 - pulse_progress_start)
                    * open_ended_cycle_fraction(cycle)
                )
                pump_stop_fraction = cycle_start_fraction + 0.32 * (
                    cycle_stop_fraction - cycle_start_fraction
                )
                seed_stop_fraction = cycle_start_fraction + 0.94 * (
                    cycle_stop_fraction - cycle_start_fraction
                )
                cycle_start_population = population.copy()
                cycle_start_free_carrier = free_carrier_state.copy()
                with self._progress_scope(
                    cycle_start_fraction,
                    pump_stop_fraction,
                ):
                    population, last_pump = self._advance_one_pump_period(
                        population,
                        record_planes=False,
                        delta_n=delta_n,
                        dark_duration_s=cycle_timing[
                            "population_dark_duration_s"
                        ],
                    )
                cycle_ase = self._last_ase_result
                last_pre_seed = population.copy()
                if free_carrier_enabled:
                    last_pre_seed_free_carrier = self._decay_free_carrier_state(
                        cycle_start_free_carrier,
                        cycle_timing["free_carrier_dark_duration_s"],
                    )
                    self._set_seed_initial_free_carrier(
                        last_pre_seed_free_carrier
                    )
                with self._progress_scope(
                    pump_stop_fraction,
                    seed_stop_fraction,
                ):
                    previous_seed_context = getattr(
                        self, "_seed_progress_context", {}
                    )
                    self._seed_progress_context = {
                        "pulse_index": cycle,
                        "pulse_total": maximum,
                        "periodic_map_cycle": cycle,
                        "periodic_map_model": map_stage,
                    }
                    try:
                        if reduced_seed_proxy:
                            previous_details = self._seed_progress_details_enabled
                            self._seed_progress_details_enabled = True
                            try:
                                trial = map_signal_solver.propagate(
                                    last_pre_seed.copy(),
                                    delta_n,
                                )
                            finally:
                                self._seed_progress_details_enabled = previous_details
                        else:
                            trial = self._propagate_seed_with_progress(
                                last_pre_seed.copy(),
                                delta_n,
                                record_planes=False,
                            )
                    finally:
                        self._seed_progress_context = previous_seed_context
                runtime_fallback = _runtime_optional_fallback(trial)
                if runtime_fallback:
                    runtime_optional_fallback_cycles.append(cycle)
                    if free_carrier_enabled:
                        raise RuntimeError(
                            "the requested coherent Seed backend failed inside "
                            "a free-carrier pulse map; its fallback does not "
                            "represent the same carried carrier state"
                        )
                post_seed = trial.population_m3
                last_post_seed = post_seed.copy()
                if free_carrier_enabled:
                    if trial.free_carrier_density_m3 is None:
                        raise RuntimeError(
                            "coherent pulse-map propagation did not return its free-carrier state"
                        )
                    last_post_seed_free_carrier = np.asarray(
                        trial.free_carrier_density_m3, dtype=float
                    ).copy()
                    carrier_convergence = self._free_carrier_convergence_metrics(
                        last_post_seed_free_carrier,
                        cycle_start_free_carrier,
                        trial,
                        previous_free_carrier_absorbed_energy_J=(
                            previous_free_carrier_absorbed_energy_J
                        ),
                        metric=carrier_convergence_metric,
                    )
                else:
                    carrier_convergence = {
                        "field_residual_relative": 0.0,
                        "optical_effect_residual_relative": 0.0,
                        "convergence_residual_relative": 0.0,
                        "absorption_energy_change_fraction": 0.0,
                        "optical_depth_change": 0.0,
                        "phase_change_rad": 0.0,
                        "free_carrier_absorbed_energy_J": None,
                    }
                free_carrier_field_residual = float(
                    carrier_convergence["field_residual_relative"]
                )
                free_carrier_convergence_residual = float(
                    carrier_convergence["convergence_residual_relative"]
                )
                output = float(trial.system_output_energy_J)
                population_residual = self._population_residual_relative(
                    post_seed, cycle_start_population
                )
                if runtime_fallback or previous_output is None:
                    output_residual = float("inf")
                else:
                    output_residual = abs(output - float(previous_output)) / max(
                        abs(output), abs(float(previous_output)), np.finfo(float).tiny
                    )
                if runtime_fallback:
                    # The fallback map is allowed to provide a new numerical
                    # guess, but it is not evidence of convergence of the
                    # requested map and must not enter Anderson history.
                    accelerator.reset()
                    next_population = np.asarray(post_seed, dtype=float).copy()
                    acceleration_row = {
                        "anderson_accepted": False,
                        "anderson_reason": (
                            "runtime_optional_fallback_history_reset"
                        ),
                        "anderson_history_depth": 0,
                        "anderson_predicted_residual_ratio": None,
                        "anderson_restarted": True,
                        "population_fraction_constraint_violation": 0.0,
                        "population_fraction_projection_rms": 0.0,
                    }
                else:
                    next_population, acceleration_row = (
                        self._accelerated_population_update(
                            accelerator,
                            cycle,
                            cycle_start_population,
                            post_seed,
                            maximum_fraction_violation,
                        )
                    )
                # A periodic state is not closed while an optically active
                # carrier reservoir is still drifting, even if population and
                # output energy happen to change little in one cycle.  Use the
                # configured carrier metric as an optional third gate. Explicit
                # opt-out preserves both carrier evolution and raw diagnostics.
                residual = max(
                    population_residual,
                    output_residual,
                    (
                        free_carrier_convergence_residual
                        if free_carrier_enabled and self._free_carrier_convergence_gate_enabled()
                        else 0.0
                    ),
                )
                if runtime_fallback:
                    residual = float("inf")
                cycle_signal_metrics = self._signal_progress_metrics(trial)
                cycle_gain_overlap_metrics = self._cycle_gain_overlap_metrics(
                    last_pre_seed,
                    trial,
                )
                cycle_pump_diagnostics = {
                    "method": "one_pump_period_preceding_this_seed_pulse",
                    **{
                        key: copy.deepcopy(last_pump.stats[key])
                        for key in (
                            "configured_power_W",
                            "grid_source_power_W",
                            "grid_input_power_W",
                            "output_power_W",
                            "absorbed_power_W",
                            "active_absorbed_power_W",
                            "passive_absorbed_power_W",
                            "crystal_power_balance_residual_W",
                            "source_to_exit_balance_residual_W",
                            "optical_grid_boundary_diagnostics",
                        )
                        if last_pump.stats.get(key) is not None
                    },
                    "maximum_intensity_W_m2": float(
                        np.max(last_pump.intensity_W_m2)
                    ),
                }
                history.append(
                    {
                        "cycle": cycle,
                        "cycle_scalar_diagnostics_schema_version": 1,
                        "cycle_scalar_diagnostics_scope": (
                            "post_seed_state_after_one_pump_period_and_one_seed_pulse"
                        ),
                        "cycle_scalar_diagnostics_retain_field_arrays": False,
                        "cycle_field_diagnostics_recorded": bool(
                            trial.output_field_diagnostics is not None
                        ),
                        "pump_cycle_diagnostics": cycle_pump_diagnostics,
                        "ase_interpulse_diagnostics": (
                            None
                            if cycle_ase is None
                            else {
                                "model": cycle_ase.stats.get("model"),
                                "escaped_power_W": float(
                                    cycle_ase.escaped_power_W
                                ),
                                "energy_ledger_W": copy.deepcopy(
                                    cycle_ase.stats.get("energy_ledger_W", {})
                                ),
                            }
                        ),
                        "runtime_optional_fallback": runtime_fallback,
                        "output_energy_J": output,
                        "population_residual_relative": population_residual,
                        "output_residual_relative": output_residual,
                        # Preserve the historical field max-norm under its
                        # original portable-result key.  New convergence uses
                        # the explicitly named optical-effect residual below.
                        "free_carrier_residual_relative": free_carrier_field_residual,
                        "free_carrier_convergence_residual_relative": (
                            free_carrier_convergence_residual
                        ),
                        "free_carrier_optical_effect_residual_relative": float(
                            carrier_convergence[
                                "optical_effect_residual_relative"
                            ]
                        ),
                        "free_carrier_absorption_energy_change_fraction": float(
                            carrier_convergence[
                                "absorption_energy_change_fraction"
                            ]
                        ),
                        "free_carrier_optical_depth_change": float(
                            carrier_convergence["optical_depth_change"]
                        ),
                        "free_carrier_phase_change_rad": float(
                            carrier_convergence["phase_change_rad"]
                        ),
                        "pre_seed_free_carrier_density_max_m3": float(
                            np.max(last_pre_seed_free_carrier)
                        ),
                        "post_seed_free_carrier_density_max_m3": float(
                            np.max(last_post_seed_free_carrier)
                        ),
                        **cycle_signal_metrics,
                        **cycle_gain_overlap_metrics,
                        **self._population_progress_metrics(post_seed),
                        **acceleration_row,
                    }
                )
                if (
                    not runtime_fallback
                    and cycle >= minimum
                    and residual <= tolerance
                ):
                    stable_count += 1
                else:
                    stable_count = 0
                self._emit_progress(
                    map_stage,
                    f"{map_label} periodic map cycle {cycle}/{maximum}",
                    fraction=cycle_stop_fraction,
                    current=cycle,
                    total=maximum,
                    metrics={
                        "population_residual": population_residual,
                        "output_residual": output_residual,
                        "free_carrier_convergence_gate_enabled": self._free_carrier_convergence_gate_enabled(),
                        "free_carrier_residual": (
                            free_carrier_convergence_residual
                        ),
                        "free_carrier_field_residual": (
                            free_carrier_field_residual
                        ),
                        "free_carrier_optical_depth_change": float(
                            carrier_convergence["optical_depth_change"]
                        ),
                        "free_carrier_phase_change_rad": float(
                            carrier_convergence["phase_change_rad"]
                        ),
                        "target_residual": tolerance,
                        "stable_cycles": stable_count,
                        "consecutive_cycles_required": required,
                        "pulse_index": cycle,
                        "pulse_total": maximum,
                        "output_energy_J": output,
                        "periodic_map_model": map_stage,
                        "runtime_optional_fallback": runtime_fallback,
                        "pump_cycle_diagnostics": cycle_pump_diagnostics,
                        **(
                            self.reduced_pulse_map_settings
                            if reduced_seed_proxy
                            else {}
                        ),
                        **(
                            self._signal_progress_metrics(
                                trial,
                                include_spectrum_preview=True,
                            )
                            if bool(
                                cycle == 1
                                or cycle % 5 == 0
                                or stable_count >= required
                            )
                            else cycle_signal_metrics
                        ),
                        **cycle_gain_overlap_metrics,
                        **self._population_progress_metrics(post_seed),
                    },
                )
                cycles_completed = cycle
                if stable_count >= required:
                    converged = True
                    break
                if joint_map_state is not None and joint_map_state.get("fixed_probe") and residual > tolerance:
                    break
                population = next_population
                free_carrier_state = last_post_seed_free_carrier.copy()
                current_carrier_energy = carrier_convergence.get(
                    "free_carrier_absorbed_energy_J"
                )
                if current_carrier_energy is not None and not runtime_fallback:
                    previous_free_carrier_absorbed_energy_J = float(
                        current_carrier_energy
                    )
                elif runtime_fallback:
                    previous_free_carrier_absorbed_energy_J = None
                previous_output = None if runtime_fallback else output
        finally:
            for target, value, scalar_value in zip(
                record_targets,
                original_record,
                original_cycle_scalar_diagnostics,
                strict=True,
            ):
                target.physics["record_optical_planes"] = value
                if scalar_value is None:
                    target.physics.pop("_record_cycle_scalar_diagnostics", None)
                else:
                    target.physics["_record_cycle_scalar_diagnostics"] = scalar_value
        if last_pump is None:
            raise RuntimeError("periodic steady-state loop did not execute")
        if joint_map_state is not None:
            joint_map_state.update(
                output_energy_J=None if runtime_fallback else output,
                free_carrier=last_post_seed_free_carrier.copy(),
                carrier_absorbed_energy_J=None if runtime_fallback else carrier_convergence.get("free_carrier_absorbed_energy_J"),
            )
        self._last_periodic_boundary_population = last_post_seed.copy()
        if free_carrier_enabled:
            # The reported pulse must start from the same pre-Seed carrier
            # boundary state used by the converged map, not from zero.
            self._set_seed_initial_free_carrier(last_pre_seed_free_carrier)
        if record_planes:
            # Convergence cycles intentionally suppress large optical-plane
            # traces.  Reconstruct one representative average-power Pump trace
            # after convergence without advancing the population again; keep
            # the already accumulated period-average intensity and absorption.
            configured_power = float(self.pump_cfg["power_W"])
            waveform = periodic_pump_waveform(
                self.pump_cfg.get("temporal", {}),
                configured_power,
                repetition_rate,
                int(self.sim_cfg["pump_time_steps_per_cycle"]),
            )
            representative_scale = (
                waveform.average_power_W
                if configured_power == 0.0
                and str(self.pump_cfg.get("temporal", {}).get("type", "cw"))
                == "file"
                and str(
                    self.pump_cfg.get("temporal", {}).get(
                        "value_mode", "relative"
                    )
                )
                == "absolute_W"
                else 0.0
                if configured_power == 0.0
                else waveform.average_power_W / configured_power
            )
            traced_pump = self.pump_solver.propagate(
                last_pre_seed,
                True,
                power_scale=representative_scale,
                delta_n=delta_n,
            )
            last_pump.trace = traced_pump.trace
            last_pump.warnings = list(
                dict.fromkeys([*last_pump.warnings, *traced_pump.warnings])
            )
        final_record = history[-1]
        acceleration_stats = accelerator.diagnostics()
        numerical_root_iterations = bool(
            initializer_stats.get("used", False)
            or initialization_source
            in {
                "thermal_outer_warm_start",
                "pump_sweep_previous_point",
                "reduced_pulse_map_initializer",
            }
            or int(acceleration_stats.get("accepted_steps", 0)) > 0
            or runtime_optional_fallback_cycles
        )
        self._emit_progress(
            map_stage,
            f"{map_label} periodic map completed",
            fraction=1.0,
            current=cycles_completed,
            total=maximum,
            metrics={
                "converged": converged,
                "population_residual": final_record[
                    "population_residual_relative"
                ],
                "output_residual": final_record["output_residual_relative"],
                "free_carrier_residual": final_record[
                    "free_carrier_convergence_residual_relative"
                ],
                "free_carrier_field_residual": final_record[
                    "free_carrier_residual_relative"
                ],
                "target_residual": tolerance,
                "stable_cycles": stable_count,
                "consecutive_cycles_required": required,
                "pulse_index": cycles_completed,
                "pulse_total": maximum,
                "periodic_map_model": map_stage,
                **(
                    self.reduced_pulse_map_settings
                    if reduced_seed_proxy
                    else {}
                ),
                **self._signal_progress_metrics(
                    trial,
                    include_spectrum_preview=True,
                ),
                **self._population_progress_metrics(last_post_seed),
            },
        )
        return last_pre_seed, {
            "dynamics": (
                "reduced_periodic_steady_state"
                if reduced_seed_proxy
                else "periodic_steady_state"
            ),
            "method": map_stage,
            "seed_map_model": (
                self.reduced_pulse_map_settings.get("field_model")
                if reduced_seed_proxy
                else "configured_full_seed_solver"
            ),
            "seed_map_settings": (
                dict(self.reduced_pulse_map_settings)
                if reduced_seed_proxy
                else {}
            ),
            "iterations": cycles_completed
            * int(self.pump_cfg.get("temporal", {}).get("sample_count", self.sim_cfg["pump_time_steps_per_cycle"])),
            "cycles_completed": cycles_completed,
            "map_evaluations": cycles_completed,
            "repetition_rate_Hz": repetition_rate,
            "simulated_duration_s": cycles_completed / repetition_rate,
            "simulated_duration_is_physical": bool(
                not reduced_seed_proxy and not numerical_root_iterations
            ),
            "cycle_timing": cycle_timing,
            "seed_depletion_events_before_reported_pulse": (
                0 if reduced_seed_proxy else max(cycles_completed - 1, 0)
            ),
            "runtime_optional_fallback_cycles": list(
                runtime_optional_fallback_cycles
            ),
            "reduced_proxy_output_energy_J": (
                float(final_record["output_energy_J"])
                if reduced_seed_proxy
                else None
            ),
            "converged": converged,
            "relative_tolerance": tolerance,
            "consecutive_cycles_required": required,
            "final_population_residual_relative": final_record["population_residual_relative"],
            "final_output_residual_relative": final_record["output_residual_relative"],
            "free_carrier_periodic_state_enabled": free_carrier_enabled,
            "final_free_carrier_residual_relative": final_record[
                "free_carrier_residual_relative"
            ],
            "free_carrier_convergence_metric": carrier_convergence_metric,
            "free_carrier_convergence_gate_enabled": self._free_carrier_convergence_gate_enabled(),
            "free_carrier_convergence_relative_tolerance": tolerance,
            "final_free_carrier_convergence_residual_relative": final_record[
                "free_carrier_convergence_residual_relative"
            ],
            "final_free_carrier_optical_effect_residual_relative": final_record[
                "free_carrier_optical_effect_residual_relative"
            ],
            "pre_seed_free_carrier_density_max_m3": float(
                np.max(last_pre_seed_free_carrier)
            ),
            "post_seed_free_carrier_density_max_m3": float(
                np.max(last_post_seed_free_carrier)
            ),
            "free_carrier_interpulse_decay_contract": (
                "exact lifetime decay over repetition period minus Seed-resolved "
                "intrapulse/interpass time; the Seed backend advances the remainder"
                if free_carrier_enabled
                else "not_applicable"
            ),
            "initialization_source": initialization_source,
            "initializer": initializer_stats,
            "acceleration": acceleration_stats,
            "ase_coupled": self._last_ase_result is not None,
            "ase_interpulse_coupling": (
                "quasi_stationary_post_seed_boundary_held_through_pump_dark_interval"
                if self._last_ase_result is not None
                else "disabled"
            ),
            "ase_refreshes_per_cycle": (
                1 if self._last_ase_result is not None else 0
            ),
            "ase_during_resolved_seed_pulse": (
                "omitted"
                if self._last_ase_result is not None
                else "not_applicable"
            ),
            "convergence_history": history,
        }, last_pump

    def _free_carrier_periodic_state_enabled(self) -> bool:
        material = getattr(self, "material", {}) or {}
        carrier = material.get("strong_field", {}).get("free_carrier", {})
        configured = bool(
            str(carrier.get("model", "off")) == "rate_equation"
            and getattr(self, "signal_model", "") == "coherent_cpa"
            and getattr(self, "physics", {}).get(
                "nonlinear_loss_model", "off"
            )
            == "strong_field_time_domain"
        )
        if not configured:
            return False
        solver = getattr(self, "signal_solver", None)
        capability = getattr(
            solver,
            "periodic_free_carrier_state_enabled",
            None,
        )
        if capability is not None:
            return bool(capability())
        return bool(
            getattr(
                solver,
                "resolved_intrapulse_free_carrier_time_s",
                None,
            )
            is not None
            and getattr(solver, "set_initial_free_carrier_density", None)
            is not None
        )

    def _set_seed_initial_free_carrier(
        self, density_m3: np.ndarray | None
    ) -> None:
        self._pending_seed_initial_free_carrier_density_m3 = (
            None
            if density_m3 is None
            else np.asarray(density_m3, dtype=float).copy()
        )
        setter = getattr(
            self.signal_solver, "set_initial_free_carrier_density", None
        )
        if setter is None:
            if density_m3 is not None and np.any(density_m3):
                raise RuntimeError(
                    "Selected Seed backend cannot accept a nonzero free-carrier boundary state"
                )
            return
        setter(density_m3)

    def _decay_free_carrier_state(
        self, density_m3: np.ndarray, delay_s: float
    ) -> np.ndarray:
        carrier = self.material.get("strong_field", {}).get("free_carrier", {})
        lifetime = float(carrier.get("lifetime_s", 0.0))
        if lifetime <= 0.0:
            raise ValueError("free-carrier lifetime must be positive")
        return np.asarray(density_m3, dtype=float) * math.exp(
            -max(float(delay_s), 0.0) / lifetime
        )

    @staticmethod
    def _free_carrier_state_residual(
        mapped: np.ndarray, previous: np.ndarray
    ) -> float:
        mapped_array = np.asarray(mapped, dtype=float)
        previous_array = np.asarray(previous, dtype=float)
        return float(
            np.max(np.abs(mapped_array - previous_array))
            / max(
                float(np.max(np.abs(mapped_array))),
                float(np.max(np.abs(previous_array))),
                np.finfo(float).tiny,
            )
        )

    def _free_carrier_convergence_gate_enabled(self) -> bool:
        """Optional stopping gate only; carrier physics/diagnostics stay active."""
        return bool(self.sim_cfg["periodic_steady_state"].get(
            "free_carrier_convergence", {}
        ).get("enabled", True))

    def _free_carrier_convergence_metrics(
        self,
        mapped: np.ndarray,
        previous: np.ndarray,
        result: SignalPropagationResult,
        *,
        previous_free_carrier_absorbed_energy_J: float | None,
        metric: str,
    ) -> dict[str, float | None]:
        """Measure periodic carrier convergence by its optical consequence.

        A pure relative max norm can remain large when an optically negligible
        carrier hotspot moves by one cell.  The default metric instead bounds
        the maximum change in FCA optical depth, carrier-induced phase, and
        pulse-integrated FCA energy.  The legacy field residual is retained in
        every history row for strict numerical diagnostics.
        """

        mapped_array = np.asarray(mapped, dtype=float)
        previous_array = np.asarray(previous, dtype=float)
        field_residual = self._free_carrier_state_residual(
            mapped_array, previous_array
        )
        maximum_density_change_m3 = float(
            np.max(np.abs(mapped_array - previous_array))
        )
        strong_field = self.material.get("strong_field", {})
        carrier = strong_field.get("free_carrier", {})
        absorption_cross_section_m2 = float(
            carrier.get("absorption_cross_section_m2", 0.0)
        )
        refractive_index_m3 = float(carrier.get("refractive_index_m3", 0.0))
        axial_length_m = float(self.grid.z_edges[-1] - self.grid.z_edges[0])
        # Bound the effect accumulated along the complete executed route,
        # including oblique paths and repeated visits to the same crystal.
        pass_rows = getattr(result, "passes", None) or [{}]
        length_m = sum(
            float(row.get("effective_path_length_m", axial_length_m))
            for row in pass_rows
        )
        optical_depth_change = (
            absorption_cross_section_m2
            * maximum_density_change_m3
            * length_m
        )
        wavelength_m = float(self.seed_cfg.get("wavelength_m", 1.0e-6))
        phase_change_rad = (
            2.0
            * math.pi
            / wavelength_m
            * abs(refractive_index_m3)
            * maximum_density_change_m3
            * length_m
        )

        cpa = result.cpa_diagnostics or {}
        strong_diagnostics = cpa.get("strong_field", {})
        current_absorbed_energy: float | None = None
        if isinstance(strong_diagnostics, dict):
            value = strong_diagnostics.get("free_carrier_absorbed_energy_J")
            if value is not None and math.isfinite(float(value)):
                current_absorbed_energy = float(value)
        absorption_energy_change_fraction = 0.0
        if (
            current_absorbed_energy is not None
            and previous_free_carrier_absorbed_energy_J is not None
        ):
            energy_scale = max(
                abs(float(result.system_output_energy_J)),
                abs(float(result.source_grid_energy_J)),
                np.finfo(float).tiny,
            )
            absorption_energy_change_fraction = abs(
                current_absorbed_energy
                - float(previous_free_carrier_absorbed_energy_J)
            ) / energy_scale

        optical_effect_residual = max(
            float(optical_depth_change),
            float(phase_change_rad),
            float(absorption_energy_change_fraction),
        )
        convergence_residual = (
            field_residual if metric == "field" else optical_effect_residual
        )
        return {
            "field_residual_relative": field_residual,
            "optical_effect_residual_relative": optical_effect_residual,
            "convergence_residual_relative": convergence_residual,
            "absorption_energy_change_fraction": (
                absorption_energy_change_fraction
            ),
            "optical_depth_change": optical_depth_change,
            "phase_change_rad": phase_change_rad,
            "free_carrier_absorbed_energy_J": current_absorbed_energy,
        }

    def _advance_one_pump_period(
        self,
        population: np.ndarray,
        record_planes: bool,
        delta_n: np.ndarray | None = None,
        *,
        dark_duration_s: float | None = None,
    ) -> tuple[np.ndarray, PumpPropagationResult]:
        # Optical ASE transport equilibrates much faster than the population
        # changes over one repetition period.  Solve it from the post-Seed
        # boundary population and hold that quasi-stationary photon field
        # during the following Pump/dark interval.  The discrete Seed solver
        # runs afterwards and never receives this field, so ASE is deliberately
        # omitted during the short resolved Seed pulse.
        ase = self._solve_ase(population)
        configured_power = float(self.pump_cfg["power_W"])
        full_waveform = periodic_pump_waveform(
            self.pump_cfg.get("temporal", {}),
            configured_power,
            float(self.sim_cfg["repetition_rate_Hz"]),
            int(self.sim_cfg["pump_time_steps_per_cycle"]),
        )
        waveform = self._pump_waveform_window(
            full_waveform,
            full_waveform.period_s
            if dark_duration_s is None
            else float(dark_duration_s),
        )
        return self._advance_sampled_pump(
            population,
            waveform,
            record_planes,
            delta_n,
            signal_spectral_intensity_W_m2=(
                None if ase is None else ase.spectral_intensity_W_m2
            ),
            signal_wavelengths_m=(
                None if ase is None else ase.wavelengths_m
            ),
        )

    @staticmethod
    def _pump_waveform_window(
        waveform: PumpWaveform,
        duration_s: float,
    ) -> PumpWaveform:
        """Restrict a normalized period waveform to the pre-Seed dark window."""

        duration = float(duration_s)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("Pulse-map dark duration must be positive and finite")
        if duration > waveform.period_s * (1.0 + 64.0 * np.finfo(float).eps):
            raise ValueError("Pulse-map dark duration cannot exceed the repetition period")
        if math.isclose(
            duration,
            waveform.period_s,
            rel_tol=64.0 * np.finfo(float).eps,
            abs_tol=0.0,
        ):
            return waveform
        count = len(waveform.power_W)
        dt = duration / count
        time = (np.arange(count, dtype=float) + 0.5) * dt
        # Samples are finite-bin mean powers. Interpolating those means at
        # new centres moves pulse edges and changes deposited energy. Rebin
        # the cumulative energy instead, integrating old/new cell overlaps.
        old_edges = np.arange(count + 1, dtype=float) * waveform.dt_s
        old_energy = np.concatenate(([0.0], np.cumsum(waveform.power_W) * waveform.dt_s))
        new_edges = np.arange(count + 1, dtype=float) * dt
        power = np.diff(np.interp(new_edges, old_edges, old_energy)) / dt
        return PumpWaveform(
            time,
            power,
            dt,
            duration,
            normalization_period_s=waveform.period_s,
        )

    def _pulse_map_cycle_timing(
        self,
        repetition_rate_Hz: float,
    ) -> dict[str, Any]:
        """Partition one period for every state advanced by the Seed solver.

        Ion populations and coherent free carriers need not resolve the same
        portion of an FFT window.  Population dynamics use the physical pulse
        support selected by their gain backend, whereas the carrier rate
        equation currently traverses the complete coherent time window.  One
        shared ``dark_duration_s`` therefore cannot close both state clocks.
        """

        period = 1.0 / float(repetition_rate_Hz)
        solver: Any = self.signal_solver
        while hasattr(solver, "inner"):
            solver = solver.inner
        if isinstance(solver, ResilientSignalPropagator):
            solver = solver._active_solver()
        passes = (
            len(getattr(solver, "gain_markers", []))
            or int(self.sim_cfg.get("passes", 1))
        )
        gain_markers = list(getattr(solver, "gain_markers", []))
        default_marker = getattr(solver, "gain_marker", {})
        between_default = list(
            getattr(solver, "between_pass_elements", [])
        )
        between_segments = list(
            getattr(solver, "between_pass_segments", [])
        )
        interpass_delay = 0.0
        interpass_rows: list[dict[str, Any]] = []
        for pass_index in range(max(passes - 1, 0)):
            marker = (
                gain_markers[pass_index]
                if pass_index < len(gain_markers)
                else default_marker
            )
            elements = (
                between_segments[pass_index]
                if pass_index < len(between_segments)
                else between_default
            )
            outside_n = float(marker.get("outside_refractive_index", 1.0))
            counter_route_getter = getattr(
                solver, "_return_route_elements", None
            )
            counter_flight_getter = getattr(
                solver, "_route_flight_time", None
            )
            if (
                pass_index == 0
                and counter_route_getter is not None
                and counter_flight_getter is not None
            ):
                elements = counter_route_getter()
                flight = counter_flight_getter(elements, outside_n)
            else:
                flight = optical_train_flight_time(
                    elements,
                    initial_refractive_index=outside_n,
                    initial_group_refractive_index=marker.get("outside_group_refractive_index"),
                    extra_delay_s=float(
                        self.sim_cfg.get(
                            "unmodeled_between_pass_delay_s", 0.0
                        )
                    ),
                )
            crystal_delay = (
                material_group_index(
                    self.material,
                    float(self.seed_cfg["wavelength_m"]),
                    role="signal",
                )
                * float(np.sum(self.grid.z_widths))
                / gain_medium_path_cosine(marker)
                / LIGHT_SPEED
            )
            resolved = flight.delay_s + crystal_delay
            interpass_delay += resolved
            interpass_rows.append(
                {
                    "after_pass": pass_index + 1,
                    "external_delay_s": flight.delay_s,
                    "crystal_group_delay_s": crystal_delay,
                    "resolved_delay_s": resolved,
                }
            )

        population_intrapulse_per_pass = 0.0
        population_intrapulse_reason = "not_resolved"
        free_carrier_enabled = self._free_carrier_periodic_state_enabled()
        total_population_time_getter = getattr(
            solver,
            "resolved_seed_population_time_s",
            None,
        )
        population_seed_resolved_override: float | None = None
        if total_population_time_getter is not None:
            population_seed_resolved_override = float(
                total_population_time_getter()
            )
            if (
                not math.isfinite(population_seed_resolved_override)
                or population_seed_resolved_override < 0.0
            ):
                raise ValueError(
                    "Resolved total Seed population time must be finite and non-negative"
                )
        population_time_getter = getattr(
            solver,
            "resolved_intrapulse_population_time_s",
            None,
        )
        if (
            population_seed_resolved_override is None
            and population_time_getter is not None
        ):
            population_intrapulse_per_pass = float(
                population_time_getter()
            )
        if (
            not math.isfinite(population_intrapulse_per_pass)
            or population_intrapulse_per_pass < 0.0
        ):
            raise ValueError(
                "Resolved Seed population time must be finite and non-negative"
            )
        if population_intrapulse_per_pass > 0.0:
            population_intrapulse_reason = (
                "josa_intrapulse_level_relaxation"
                if self.signal_model == "josa_transport"
                else "commercial_app_intrapulse_level_relaxation"
                if self.signal_model == "commercial_app_fast"
                else "coherent_intrapulse_level_relaxation"
            )
        elif (
            population_seed_resolved_override is not None
            and population_seed_resolved_override > 0.0
        ):
            population_intrapulse_reason = "causal_multipass_absolute_time_clock"

        free_carrier_intrapulse_per_pass = 0.0
        if free_carrier_enabled:
            carrier_time_getter = getattr(
                solver,
                "resolved_intrapulse_free_carrier_time_s",
                None,
            )
            if carrier_time_getter is None:
                raise RuntimeError(
                    "Selected Seed backend cannot report its resolved free-carrier time"
                )
            free_carrier_intrapulse_per_pass = float(carrier_time_getter())
            if (
                not math.isfinite(free_carrier_intrapulse_per_pass)
                or free_carrier_intrapulse_per_pass <= 0.0
            ):
                raise ValueError(
                    "Resolved Seed free-carrier time must be positive and finite"
                )

        population_intrapulse_total = population_intrapulse_per_pass * passes
        free_carrier_intrapulse_total = (
            free_carrier_intrapulse_per_pass * passes
        )
        population_interpass_dark = sum(
            interpass_dark_time_s(row["resolved_delay_s"], population_intrapulse_per_pass)
            for row in interpass_rows
        )
        carrier_interpass_dark = sum(
            interpass_dark_time_s(row["resolved_delay_s"], free_carrier_intrapulse_per_pass)
            for row in interpass_rows
        )
        population_seed_resolved = (
            population_interpass_dark + population_intrapulse_total
            if population_seed_resolved_override is None
            else population_seed_resolved_override
        )
        if population_seed_resolved_override is not None:
            population_intrapulse_total = max(
                population_seed_resolved_override - interpass_delay,
                0.0,
            )
            population_intrapulse_per_pass = (
                population_intrapulse_total / max(passes, 1)
            )
        free_carrier_seed_resolved = (
            carrier_interpass_dark + free_carrier_intrapulse_total
            if free_carrier_enabled
            else 0.0
        )

        def require_room(name: str, resolved: float) -> None:
            tolerance = 64.0 * np.finfo(float).eps * max(
                abs(period),
                abs(resolved),
                np.finfo(float).tiny,
            )
            if resolved >= period - tolerance:
                raise ValueError(
                    f"Resolved Seed {name} intrapulse/interpass time must be "
                    "shorter than the repetition period; got "
                    f"Seed={resolved:.6g} s and period={period:.6g} s."
                )

        require_room("population", population_seed_resolved)
        if free_carrier_enabled:
            require_room("free-carrier", free_carrier_seed_resolved)

        population_dark = period - population_seed_resolved
        free_carrier_dark = (
            period - free_carrier_seed_resolved
            if free_carrier_enabled
            else population_dark
        )
        seed_resolved = max(
            population_seed_resolved,
            free_carrier_seed_resolved,
        )
        intrapulse_per_pass = max(
            population_intrapulse_per_pass,
            free_carrier_intrapulse_per_pass,
        )
        intrapulse_total = max(
            population_intrapulse_total,
            free_carrier_intrapulse_total,
        )
        intrapulse_reason = (
            "population_and_free_carrier_resolved_on_separate_clocks"
            if population_intrapulse_per_pass > 0.0 and free_carrier_enabled
            else "free_carrier_rate_equation"
            if free_carrier_enabled
            else population_intrapulse_reason
        )
        return {
            "repetition_period_s": period,
            "seed_resolved_elapsed_time_s": seed_resolved,
            # Compatibility alias: Pump evolution acts on the ion population.
            "dark_duration_s": population_dark,
            "population_dark_duration_s": population_dark,
            "free_carrier_dark_duration_s": free_carrier_dark,
            "population_seed_resolved_elapsed_time_s": (
                population_seed_resolved
            ),
            "free_carrier_seed_resolved_elapsed_time_s": (
                free_carrier_seed_resolved if free_carrier_enabled else None
            ),
            "intrapulse_time_per_pass_s": intrapulse_per_pass,
            "intrapulse_total_s": intrapulse_total,
            "intrapulse_reason": intrapulse_reason,
            "population_intrapulse_time_per_pass_s": (
                population_intrapulse_per_pass
            ),
            "population_intrapulse_total_s": population_intrapulse_total,
            "population_intrapulse_reason": population_intrapulse_reason,
            "free_carrier_intrapulse_time_per_pass_s": (
                free_carrier_intrapulse_per_pass
                if free_carrier_enabled
                else None
            ),
            "free_carrier_intrapulse_total_s": (
                free_carrier_intrapulse_total
                if free_carrier_enabled
                else None
            ),
            "interpass_resolved_time_s": interpass_delay,
            "population_interpass_dark_time_s": (
                population_interpass_dark if population_seed_resolved_override is None else None
            ),
            "free_carrier_interpass_dark_time_s": (
                carrier_interpass_dark if free_carrier_enabled else None
            ),
            "interpass_segments": interpass_rows,
            "time_closure_residual_s": (
                period - population_dark - population_seed_resolved
            ),
            "population_time_closure_residual_s": (
                period - population_dark - population_seed_resolved
            ),
            "free_carrier_time_closure_residual_s": (
                period - free_carrier_dark - free_carrier_seed_resolved
                if free_carrier_enabled
                else None
            ),
        }

    def _advance_sampled_pump(
        self,
        population: np.ndarray,
        waveform: Any,
        record_planes: bool,
        delta_n: np.ndarray | None = None,
        *,
        signal_spectral_intensity_W_m2: np.ndarray | None = None,
        signal_wavelengths_m: np.ndarray | None = None,
    ) -> tuple[np.ndarray, PumpPropagationResult]:
        if (signal_spectral_intensity_W_m2 is None) != (
            signal_wavelengths_m is None
        ):
            raise ValueError(
                "interpulse signal spectral intensity and wavelengths must be provided together"
            )
        configured_power = float(self.pump_cfg["power_W"])
        normalization_period_s = getattr(
            waveform, "normalization_period_s", None
        )
        period_average_scale = (
            waveform.period_s
            / (
                waveform.period_s
                if normalization_period_s is None
                else float(normalization_period_s)
            )
        )
        period_average_power_W = float(
            getattr(
                waveform,
                "period_average_power_W",
                float(waveform.average_power_W) * period_average_scale,
            )
        )
        average_intensity = np.zeros(self.grid.shape, dtype=float)
        average_absorbed = np.zeros(self.grid.shape, dtype=float)
        average_active_absorbed = np.zeros(self.grid.shape, dtype=float)
        average_passive_absorbed = np.zeros(self.grid.shape, dtype=float)
        average_spectral_intensity: np.ndarray | None = None
        average_spectral_active: np.ndarray | None = None
        average_spectral_passive: np.ndarray | None = None
        # Every additive power ledger entry must describe the same period-average
        # state as the returned intensity and absorption fields.  Keeping a short
        # allow-list here previously left aperture/transport balances attached to
        # the final instantaneous sample.
        scalar_totals: dict[str, float] = {}
        worst_maxima: dict[str, float] = {}
        worst_minima: dict[str, float] = {}
        triggered_flags: dict[str, bool] = {}
        warnings: list[str] = []
        last_sample: PumpPropagationResult | None = None
        event_ledger: EventEnergyLedger | None = None
        event_destination_average: dict[str, np.ndarray] | None = None
        event_rate_average: dict[str, np.ndarray] | None = None
        event_scalar_average: dict[str, float] = {}
        event_scalar_maximum: dict[str, float] = {}
        event_energy_rate_average: dict[str, float] = {}
        last_event_stats: dict[str, Any] | None = None
        if (
            str(
                getattr(self, "material", {}).get("thermal", {}).get(
                    "heat_source_model", "pump_component_ledger"
                )
            )
            == "event_energy_ledger"
        ):
            event_ledger = EventEnergyLedger(
                self.levels,
                self.config["energy_levels"].get(
                    "event_energy_ledger", {}
                ),
                float(self.pump_cfg["wavelength_m"]),
                float(self.seed_cfg["wavelength_m"]),
            )
        absolute_file = bool(
            str(self.pump_cfg.get("temporal", {}).get("type", "cw")) == "file"
            and str(
                self.pump_cfg.get("temporal", {}).get(
                    "value_mode", "relative"
                )
            )
            == "absolute_W"
        )
        sample_total = len(waveform.power_W)
        progress_stride = max(sample_total // 100, 1)
        self._emit_progress(
            "pump_waveform",
            "Beginning sampled Pump waveform propagation",
            fraction=0.0,
            current=0,
            total=sample_total,
        )
        for sample_index, instantaneous_power in enumerate(
            waveform.power_W, start=1
        ):
            if configured_power == 0.0:
                scale = float(instantaneous_power) if absolute_file else 0.0
            else:
                scale = float(instantaneous_power) / configured_power
            pump = self.pump_solver.propagate(
                population,
                False,
                power_scale=scale,
                delta_n=delta_n,
            )
            last_sample = pump
            average_intensity += pump.intensity_W_m2 / len(waveform.power_W)
            average_absorbed += pump.absorbed_W_m3 / len(waveform.power_W)
            if pump.spectral_intensity_W_m2 is not None:
                if average_spectral_intensity is None:
                    average_spectral_intensity = np.zeros_like(
                        pump.spectral_intensity_W_m2
                    )
                average_spectral_intensity += (
                    pump.spectral_intensity_W_m2 / len(waveform.power_W)
                )
            if pump.spectral_active_absorbed_W_m3 is not None:
                if average_spectral_active is None:
                    average_spectral_active = np.zeros_like(
                        pump.spectral_active_absorbed_W_m3
                    )
                average_spectral_active += (
                    pump.spectral_active_absorbed_W_m3 / len(waveform.power_W)
                )
            if pump.spectral_passive_absorbed_W_m3 is not None:
                if average_spectral_passive is None:
                    average_spectral_passive = np.zeros_like(
                        pump.spectral_passive_absorbed_W_m3
                    )
                average_spectral_passive += (
                    pump.spectral_passive_absorbed_W_m3
                    / len(waveform.power_W)
                )
            if pump.active_absorbed_W_m3 is not None:
                average_active_absorbed += (
                    pump.active_absorbed_W_m3 / len(waveform.power_W)
                )
            if pump.passive_absorbed_W_m3 is not None:
                average_passive_absorbed += (
                    pump.passive_absorbed_W_m3 / len(waveform.power_W)
                )
            for key, value in pump.stats.items():
                if value is None:
                    continue
                if isinstance(value, (bool, np.bool_)):
                    triggered_flags[key] = bool(
                        triggered_flags.get(key, False) or bool(value)
                    )
                    continue
                if (
                    isinstance(value, (str, bytes))
                    or not np.isscalar(value)
                ):
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if not np.isfinite(numeric):
                    continue
                is_maximum = bool(
                    key.startswith(("maximum_", "max_"))
                    or "_maximum_" in key
                    or key.endswith(("_maximum", "_max"))
                )
                is_minimum = bool(
                    key.startswith(("minimum_", "min_"))
                    or "_minimum_" in key
                    or key.endswith(("_minimum", "_min"))
                    or "capture_fraction" in key
                )
                # Semantic extrema take precedence over physical units: a
                # ``maximum_*_W`` diagnostic is still a worst-case maximum,
                # not a period-average power ledger entry.
                if is_maximum:
                    worst_maxima[key] = max(
                        worst_maxima.get(key, -float("inf")), numeric
                    )
                elif is_minimum:
                    worst_minima[key] = min(
                        worst_minima.get(key, float("inf")), numeric
                    )
                elif key.endswith("_W"):
                    metric_spec = pump_metric_spec(key)
                    if metric_spec is None:
                        raise RuntimeError(
                            "Pump waveform aggregation received scalar power "
                            f"diagnostic {key!r} without an explicit reducer contract"
                        )
                    if metric_spec.temporal_reduction != "mean":
                        raise RuntimeError(
                            f"Unsupported Pump temporal reducer for {key!r}: "
                            f"{metric_spec.temporal_reduction!r}"
                        )
                    scalar_totals[key] = scalar_totals.get(key, 0.0) + numeric / len(
                        waveform.power_W
                    )
                elif key.endswith(("_W_m2", "_W_m3")):
                    # These are local densities/intensities rather than scalar
                    # power-ledger terms. Their current waveform contract is a
                    # time average and is named explicitly here, not inferred
                    # from a generic ``*_W`` suffix.
                    scalar_totals[key] = scalar_totals.get(key, 0.0) + numeric / len(
                        waveform.power_W
                    )
                elif key.endswith("_relative") or key.endswith("_fraction"):
                    # Residuals and loss fractions are conservative worst-case
                    # diagnostics unless recomputed from an aggregate ledger
                    # below.
                    worst_maxima[key] = max(
                        worst_maxima.get(key, -float("inf")), numeric
                    )
            warnings.extend(pump.warnings)
            population_before = population
            step_events = {} if event_ledger is not None else None
            population = self.levels.advance_implicit(
                population,
                waveform.dt_s,
                pump.intensity_W_m2,
                float(self.pump_cfg["wavelength_m"]),
                pump_wavelengths_m=self.pump_solver.spectrum.coordinate,
                pump_power_weights=self.pump_solver.spectrum.weights,
                pump_spectral_intensity_W_m2=pump.spectral_intensity_W_m2,
                signal_spectral_intensity_W_m2=(
                    signal_spectral_intensity_W_m2
                ),
                signal_wavelengths_m=signal_wavelengths_m,
                step_events=step_events,
            )
            if event_ledger is not None:
                assert step_events is not None
                event_wavelengths, event_intensity = self._pump_event_spectral_fields(pump)
                # Linear kinetics uses backward Euler, so event counts are
                # dt * rate(N_linear_end). Nonlinear counts come from the
                # actual budget-limited substeps, before any renormalization.
                event = event_ledger.evaluate(
                    step_events["linear_population_m3"],
                    pump_wavelengths_m=event_wavelengths,
                    pump_spectral_intensity_W_m2=event_intensity,
                    grid=self.grid,
                    nonlinear_event_counts_m3=step_events["nonlinear_event_counts_m3"],
                    step_duration_s=waveform.dt_s,
                )
                storage_change = np.einsum(
                    "i,i...->...", event_ledger.level_energies_J,
                    population - population_before,
                ) / waveform.dt_s
                actual_storage_rate = float(self.grid.integrate_volume(storage_change))
                event.stats["finite_step_level_storage_derivative_W"] = actual_storage_rate
                if signal_spectral_intensity_W_m2 is None:
                    event.stats["finite_step_kinetic_storage_residual_W"] = (
                        event.stats["modeled_level_storage_derivative_W"] - actual_storage_rate
                    )
                event.stats["finite_step_storage_audit_scope"] = (
                    "Pump, linear decays and integrated nonlinear events; "
                    "external interpulse Signal exchange is excluded from the kinetic residual"
                )
                if pump.active_absorbed_W_m3 is not None:
                    event.stats["pump_transport_kinetic_mismatch_W"] = (
                        float(self.grid.integrate_volume(pump.active_absorbed_W_m3))
                        - event.stats["pump_absorbed_optical_power_W"]
                        + event.stats["pump_stimulated_optical_power_W"]
                    )
                if event_destination_average is None:
                    event_destination_average = {
                        name: np.zeros_like(values)
                        for name, values in event.destination_W_m3.items()
                    }
                    event_rate_average = {
                        name: np.zeros_like(values)
                        for name, values in event.event_rate_m3_s.items()
                    }
                time_weight = 1.0 / len(waveform.power_W)
                for name in event_destination_average:
                    event_destination_average[name] += event.destination_W_m3[name] * time_weight
                assert event_rate_average is not None
                for name in event_rate_average:
                    event_rate_average[name] += event.event_rate_m3_s[name] * time_weight
                for key, value in event.stats.items():
                    if isinstance(value, (bool, str, bytes)) or not np.isscalar(value):
                        continue
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if not np.isfinite(numeric):
                        continue
                    if key.startswith("maximum_"):
                        event_scalar_maximum[key] = max(
                            event_scalar_maximum.get(key, -float("inf")), numeric
                        )
                    elif key.endswith("_W"):
                        event_scalar_average[key] = (
                            event_scalar_average.get(key, 0.0) + numeric * time_weight
                        )
                for name, value in event.stats.get("event_energy_rate_W", {}).items():
                    event_energy_rate_average[name] = (
                        event_energy_rate_average.get(name, 0.0) + float(value) * time_weight
                    )
                last_event_stats = event.stats
            if (
                getattr(self, "_progress_mode", "detailed") == "detailed"
                and (
                    sample_index == sample_total
                    or sample_index % progress_stride == 0
                )
            ):
                self._emit_progress(
                    "pump_waveform",
                    f"Pump waveform sample {sample_index}/{sample_total}",
                    fraction=sample_index / max(sample_total, 1),
                    current=sample_index,
                    total=sample_total,
                    metrics={
                        "instantaneous_power_W": float(instantaneous_power),
                        "absorbed_pump_power_W": float(
                            pump.stats.get("absorbed_power_W", 0.0)
                        ),
                        "pump_output_power_W": float(
                            pump.stats.get("output_power_W", 0.0)
                        ),
                        "pump_time_s": float(sample_index * waveform.dt_s),
                        **self._population_progress_metrics(population),
                    },
                )
        if last_sample is None:
            raise RuntimeError("sampled Pump waveform contains no time samples")
        average_intensity *= period_average_scale
        average_absorbed *= period_average_scale
        average_active_absorbed *= period_average_scale
        average_passive_absorbed *= period_average_scale
        if average_spectral_intensity is not None:
            average_spectral_intensity *= period_average_scale
        if average_spectral_active is not None:
            average_spectral_active *= period_average_scale
        if average_spectral_passive is not None:
            average_spectral_passive *= period_average_scale
        scalar_totals = {
            key: value * period_average_scale
            for key, value in scalar_totals.items()
        }
        event_scalar_average = {
            key: value * period_average_scale
            for key, value in event_scalar_average.items()
        }
        event_energy_rate_average = {
            key: value * period_average_scale
            for key, value in event_energy_rate_average.items()
        }
        if event_destination_average is not None:
            for values in event_destination_average.values():
                values *= period_average_scale
        if event_rate_average is not None:
            for values in event_rate_average.values():
                values *= period_average_scale
        if record_planes:
            representative_scale = (
                period_average_power_W
                if configured_power == 0.0 and absolute_file
                else 0.0
                if configured_power == 0.0
                else period_average_power_W / configured_power
            )
            representative = self.pump_solver.propagate(
                population,
                True,
                power_scale=representative_scale,
                delta_n=delta_n,
            )
        else:
            # Intensity/absorption and scalar statistics are replaced by the
            # period averages below.  Reusing the final sample avoids one full
            # extra optical propagation on every hidden convergence cycle.
            representative = last_sample
        representative.intensity_W_m2 = average_intensity
        representative.absorbed_W_m3 = average_absorbed
        representative.active_absorbed_W_m3 = average_active_absorbed
        representative.passive_absorbed_W_m3 = average_passive_absorbed
        representative.spectral_intensity_W_m2 = average_spectral_intensity
        representative.spectral_active_absorbed_W_m3 = average_spectral_active
        representative.spectral_passive_absorbed_W_m3 = average_spectral_passive
        if event_destination_average is not None and last_event_stats is not None:
            representative.event_energy_heat_W_m3 = (
                event_destination_average["heat"]
            )
            averaged_event_stats = dict(last_event_stats)
            averaged_event_stats.update(event_scalar_average)
            averaged_event_stats.update(event_scalar_maximum)
            averaged_event_stats.update(
                {
                    "destination_power_W": {
                        name: float(self.grid.integrate_volume(values))
                        for name, values in event_destination_average.items()
                    },
                    "event_count_rate_s": {
                        name: float(self.grid.integrate_volume(values))
                        for name, values in (event_rate_average or {}).items()
                    },
                    "event_energy_rate_W": event_energy_rate_average,
                    "heat_power_W": float(
                        self.grid.integrate_volume(
                            event_destination_average["heat"]
                        )
                    ),
                    "temporal_applicability": (
                        "time_bin_backward_euler_event_counts_period_average"
                    ),
                    "time_step_s": float(waveform.dt_s),
                    "time_step_count": int(len(waveform.power_W)),
                    "integration_duration_s": float(
                        waveform.dt_s * len(waveform.power_W)
                    ),
                    "population_sampling": (
                        "linear_bin_end_and_actual_nonlinear_event_extents"
                    ),
                }
            )
            destination_scale = sum(
                abs(float(value))
                for value in averaged_event_stats[
                    "destination_power_W"
                ].values()
            )
            averaged_event_stats["unresolved_destination_fraction"] = (
                abs(
                    float(
                        averaged_event_stats["destination_power_W"].get(
                            "unresolved", 0.0
                        )
                    )
                )
                / max(destination_scale, np.finfo(float).tiny)
            )
            identity_scale = max(
                abs(
                    float(
                        averaged_event_stats.get(
                            "pump_absorbed_optical_power_W", 0.0
                        )
                    )
                ),
                abs(
                    float(
                        averaged_event_stats.get(
                            "pump_stimulated_optical_power_W", 0.0
                        )
                    )
                ),
                abs(
                    float(
                        averaged_event_stats.get(
                            "modeled_level_storage_derivative_W", 0.0
                        )
                    )
                ),
                destination_scale,
                np.finfo(float).tiny,
            )
            averaged_event_stats["full_identity_residual_relative"] = (
                abs(
                    float(
                        averaged_event_stats.get(
                            "full_identity_residual_W", 0.0
                        )
                    )
                )
                / identity_scale
            )
            representative.event_energy_stats = averaged_event_stats
        representative.stats.update(scalar_totals)
        representative.stats.update(worst_maxima)
        representative.stats.update(worst_minima)
        representative.stats.update(triggered_flags)
        representative.stats.update(
            {
                "configured_power_W": configured_power,
                "average_power_W": period_average_power_W,
                "active_window_average_power_W": waveform.average_power_W,
                "peak_power_W": float(np.max(waveform.power_W)),
                "waveform": str(self.pump_cfg.get("temporal", {}).get("type", "cw")),
                "period_s": (
                    waveform.period_s
                    if normalization_period_s is None
                    else float(normalization_period_s)
                ),
                "integrated_dark_window_s": waveform.period_s,
                "period_average_scale": period_average_scale,
                "time_steps_per_cycle": len(waveform.power_W),
                "absorbed_power_W": float(self.grid.integrate_volume(average_absorbed)),
                "power_ledger_time_basis": "period_average",
                "temporal_diagnostic_reduction": {
                    "power": "arithmetic_period_average",
                    "maximum_and_residual": "maximum_over_time_bins",
                    "minimum": "minimum_over_time_bins",
                    "boolean_status": "logical_or_over_time_bins",
                    "ledger_relative_residual": "recomputed_after_aggregation",
                },
            }
        )
        input_power = float(representative.stats.get("grid_input_power_W", 0.0))
        crystal_residual = representative.stats.get(
            "crystal_power_balance_residual_W"
        )
        if crystal_residual is not None:
            representative.stats["crystal_power_balance_residual_relative"] = abs(
                float(crystal_residual)
            ) / max(abs(input_power), np.finfo(float).tiny)
        source_residual = representative.stats.get("source_to_exit_balance_residual_W")
        average_source_power = float(
            representative.stats.get("grid_source_power_W", waveform.average_power_W)
        )
        if source_residual is not None:
            representative.stats["source_to_exit_balance_residual_relative"] = abs(
                float(source_residual)
            ) / max(abs(average_source_power), np.finfo(float).tiny)
        entrance_before = representative.stats.get(
            "gain_entrance_power_before_crystal_aperture_W"
        )
        if entrance_before is not None:
            representative.stats["gain_entrance_crystal_aperture_transmission"] = (
                input_power
                / max(abs(float(entrance_before)), np.finfo(float).tiny)
            )
        if "computed_source_to_crystal_efficiency" in representative.stats:
            representative.stats["computed_source_to_crystal_efficiency"] = (
                input_power
                / max(abs(average_source_power), np.finfo(float).tiny)
            )
        representative.warnings = list(dict.fromkeys([*representative.warnings, *warnings]))
        return population, representative

    def _prepare_population(
        self,
        delta_n: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        density = self.dopant_density_m3
        population = self.levels.initial_population(density, self.grid.shape)
        mode = self.physics["population_dynamics"]
        if mode == "frozen":
            if self.ase_solver.enabled:
                raise ValueError(
                    "enabled ASE transport requires a population solver; frozen populations cannot "
                    "be depleted or clamped by ASE"
                )
            return population, {"dynamics": mode, "iterations": 0, "converged": True}
        if mode == "steady_state":
            residual = float("inf")
            iterations = 0
            iteration_limit = int(self.physics["steady_state_iterations"])
            ase_relaxation = 1.0
            if self.ase_solver.enabled:
                iteration_limit = max(
                    iteration_limit,
                    int(self.config.get("ase", {}).get("maximum_population_iterations", 12)),
                )
                ase_relaxation = float(
                    self.config.get("ase", {}).get("population_relaxation", 0.5)
                )
            self._emit_progress(
                "population_steady_state",
                "Beginning stationary population iteration",
                fraction=0.0,
                current=0,
                total=iteration_limit,
            )
            for iterations in range(1, iteration_limit + 1):
                pump = self.pump_solver.propagate(
                    population,
                    False,
                    delta_n=delta_n,
                )
                ase = self._solve_ase(population)
                updated = self.levels.steady_state(
                    density,
                    pump.intensity_W_m2,
                    float(self.pump_cfg["wavelength_m"]),
                    pump_wavelengths_m=self.pump_solver.spectrum.coordinate,
                    pump_power_weights=self.pump_solver.spectrum.weights,
                    pump_spectral_intensity_W_m2=pump.spectral_intensity_W_m2,
                    signal_spectral_intensity_W_m2=(
                        ase.spectral_intensity_W_m2 if ase is not None else None
                    ),
                    signal_wavelengths_m=(
                        ase.wavelengths_m if ase is not None else None
                    ),
                )
                updated *= self.crystal_mask[None, ...]
                # Test the physical fixed-point map before damping: arbitrarily
                # small relaxation must not manufacture convergence.
                residual = self._population_residual_relative(updated, population)
                if self.ase_solver.enabled:
                    updated = (
                        (1.0 - ase_relaxation) * population
                        + ase_relaxation * updated
                    )
                population = updated
                self._emit_progress(
                    "population_steady_state",
                    f"Stationary population iteration {iterations}/{iteration_limit}",
                    fraction=0.95 * iterations / max(iteration_limit, 1),
                    current=iterations,
                    total=iteration_limit,
                    metrics={
                        "residual": residual,
                        "target_residual": 1e-7,
                    },
                )
                if residual < 1e-7:
                    break
            final_ase = self._solve_ase(population)
            self._emit_progress(
                "population_steady_state",
                "Stationary population solve completed",
                fraction=1.0,
                current=iterations,
                total=iteration_limit,
                metrics={
                    "residual": residual,
                    "target_residual": 1e-7,
                    "converged": residual < 1e-7,
                },
            )
            return population, {
                "dynamics": mode,
                "iterations": iterations,
                "converged": residual < 1e-7,
                "iteration_residual_relative": residual,
                "ase_coupled": final_ase is not None,
            }
        if self.ase_solver.enabled:
            raise ValueError(
                "enabled ASE transport requires population_dynamics=steady_state, "
                "a repetition-rate pulse train, or an enabled periodic steady-state "
                "method; single-shot transient Pump/ASE co-propagation is not implemented"
            )
        duration = float(self.sim_cfg["pump_duration_s"])
        steps = int(self.sim_cfg["pump_time_steps"])
        completed = 0
        # The preparation interval also contains dark level kinetics at zero Pump.
        if duration > 0.0:
            waveform_cfg = copy.deepcopy(
                self.pump_cfg.get("temporal", {})
            )
            waveform_cfg.update(
                {
                    "period_s": duration,
                    "sample_count": steps,
                }
            )
            waveform = periodic_pump_waveform(
                waveform_cfg,
                float(self.pump_cfg["power_W"]),
                0.0,
                steps,
            )
            population, sampled_pump = self._advance_sampled_pump(
                population,
                waveform,
                False,
                delta_n,
            )
            completed = steps
            self.run_context.fixed_point_history[
                "transient_sampled_pump_result"
            ] = sampled_pump
        return population, {
            "dynamics": mode,
            "iterations": completed,
            "simulated_pump_duration_s": duration if completed else 0.0,
            "converged": None,
        }

    def _solve_ase(self, population: np.ndarray) -> ASEResult | None:
        """Run the optional ASE closure without silently changing failures."""

        if not self.ase_solver.enabled:
            return None
        if bool(self._ase_execution.get("disabled_after_failure", False)):
            return None
        try:
            result = self.ase_solver.solve(population)
        except OPTIONAL_MODEL_FAILURES as exc:
            policy = str(self.config.get("ase", {}).get("failure_policy", "error"))
            if policy == "error":
                raise
            reason = f"{type(exc).__name__}: {exc}"
            self._ase_warnings.append(
                f"ASE transport closure failed ({reason}); ASE was disabled for this operating point."
            )
            self._ase_execution.update(
                {
                    "executed": False,
                    "disabled_after_failure": True,
                    "disabled_reason": reason,
                }
            )
            self._last_ase_result = None
            return None
        self._last_ase_result = result
        self._ase_execution.update(
            {
                "executed": True,
                "disabled_after_failure": False,
                "disabled_reason": None,
            }
        )
        self._ase_warnings.extend(result.warnings)
        return result

    def _solve_temperature(self, heat: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if self._thermal_solver is None:
            thermal_cfg = copy.deepcopy(self.material["thermal"])
            mode = self.physics["thermal_model"]
            if mode == "off":
                thermal_cfg["enabled"] = False
            elif mode == "steady_3d":
                thermal_cfg["enabled"] = True
                thermal_cfg.setdefault("transient", {})["enabled"] = False
            elif mode == "transient_3d":
                thermal_cfg["enabled"] = True
                thermal_cfg.setdefault("transient", {})["enabled"] = True
            self._thermal_solver = ThermalSolver(
                self.grid,
                thermal_cfg,
                self.crystal_mask,
                self.material.get("geometry", {}),
            )
        if not self._continuation_thermal_guess_consumed:
            continuation_temperature = (
                None
                if self._continuation_state is None
                else self._continuation_state.thermal_temperature_K
            )
            if continuation_temperature is not None:
                self._thermal_solver.set_initial_guess(continuation_temperature)
            self._continuation_thermal_guess_consumed = True

        def report(event: dict[str, Any]) -> None:
            phase = str(event.get("phase", "steady"))
            current = event.get("current")
            total = event.get("total")
            if phase == "transient_time_step":
                message = f"Thermal time step {current}/{total}"
            else:
                message = f"Thermal linear iteration {current}/{total}"
            metrics = {
                key: value
                for key, value in event.items()
                if key not in {"current", "total", "fraction"}
            }
            self._emit_progress(
                "thermal_solver",
                message,
                fraction=float(event.get("fraction", 0.0)),
                current=None if current is None else int(current),
                total=None if total is None else int(total),
                metrics=metrics,
            )

        return self._thermal_solver.solve(
            heat,
            progress_callback=(
                report
                if getattr(self, "_progress_mode", "detailed") == "detailed"
                else None
            ),
        )

    @staticmethod
    def _executed_thermal_model(thermal_stats: dict[str, Any]) -> str:
        """Name the thermal backend represented by one immutable result."""

        if not bool(thermal_stats.get("enabled", False)):
            return "off"
        return (
            "transient_3d"
            if bool(thermal_stats.get("transient", False))
            else "steady_3d"
        )

    def _stress_backend(self) -> Any:
        if self._stress_solver is None:
            self._stress_solver = make_stress_solver(
                self.material["mechanical"],
                self.grid,
                self.material.get("geometry", {}),
            )
        return self._stress_solver

    def _pump_event_spectral_fields(
        self, pump_result: PumpPropagationResult
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use the same Pump spectrum as the population rate equations.

        Shared spatial transport omits a spectral volume but still represents
        every source wavelength. Reconstruct its power-weighted intensities;
        a single field here must not turn a broadband Pump into a line source.
        """

        if pump_result.spectral_intensity_W_m2 is not None:
            if pump_result.wavelengths_m is None:
                raise RuntimeError("spectral Pump intensities require wavelengths")
            return (
                np.asarray(pump_result.wavelengths_m, dtype=float),
                np.asarray(pump_result.spectral_intensity_W_m2, dtype=float),
            )
        spectrum = self.pump_solver.spectrum
        intensity = np.asarray(pump_result.intensity_W_m2, dtype=float)
        weights = np.asarray(spectrum.weights, dtype=float).reshape(
            (-1,) + (1,) * intensity.ndim
        )
        return (
            np.asarray(spectrum.coordinate, dtype=float),
            weights * intensity[None, ...],
        )

    def _heat_load(
        self,
        pump_result: PumpPropagationResult,
        population: np.ndarray | None = None,
    ) -> np.ndarray:
        def validated_heat_component(
            values: np.ndarray,
            *,
            name: str,
            expected_shape: tuple[int, ...],
            non_negative: bool,
        ) -> np.ndarray:
            array = np.asarray(values, dtype=float)
            if array.shape != expected_shape:
                raise RuntimeError(
                    f"{name} must have shape {expected_shape}; got {array.shape}"
                )
            if not np.all(np.isfinite(array)):
                raise RuntimeError(f"{name} must be finite")
            if not non_negative:
                return array
            scale = max(float(np.max(np.abs(array))), 1.0)
            if float(np.min(array)) < -64.0 * np.finfo(float).eps * scale:
                raise RuntimeError(f"{name} must be non-negative")
            return np.maximum(array, 0.0)

        thermal = self.material["thermal"]
        source_model = str(
            thermal.get("heat_source_model", "pump_component_ledger")
        )
        fraction_value, reference_wavelength, _ = self._resolved_active_heat_fraction()
        configured_fraction = thermal.get("heat_fraction", "quantum_defect")
        spectral_fraction_model = (
            str(configured_fraction)
            if configured_fraction in SPECTRAL_HEAT_MODELS
            else (
                "spectral_quantum_defect"
                if configured_fraction
                in {"quantum_defect", "lifetime_quantum_defect"}
                and len(self.levels.pump_transitions) > 1
                else "spectral_scaled_quantum_defect"
                if configured_fraction == "scaled_quantum_defect"
                and len(self.levels.pump_transitions) > 1
                else None
            )
        )
        fraction_field: float | np.ndarray = fraction_value
        spectrally_resolved_component_heat: np.ndarray | None = None
        self._last_spectral_heat_stats = {"enabled": False}
        self._last_event_energy_stats = {"enabled": False}
        self._last_lifetime_heat_stats = {"enabled": False}
        self._last_pump_component_applicability = {
            "applicable": None,
            "reason": (
                "different_heat_source_model"
                if source_model != "pump_component_ledger"
                else "not_evaluated"
            ),
        }
        if source_model == "event_energy_ledger":
            if population is None:
                raise RuntimeError(
                    "event_energy_ledger heat requires the operating population"
                )
            if pump_result.event_energy_heat_W_m3 is not None:
                event_heat = validated_heat_component(
                    pump_result.event_energy_heat_W_m3,
                    name="Pump event-energy heat density",
                    expected_shape=self.grid.shape,
                    non_negative=False,
                )
                self._last_event_energy_stats = dict(
                    pump_result.event_energy_stats or {}
                )
            else:
                wavelengths, spectral_intensity = (
                    self._pump_event_spectral_fields(pump_result)
                )
                ledger = EventEnergyLedger(
                    self.levels,
                    self.config["energy_levels"].get(
                        "event_energy_ledger", {}
                    ),
                    float(self.pump_cfg["wavelength_m"]),
                    float(self.seed_cfg["wavelength_m"]),
                )
                event_result = ledger.evaluate(
                    population,
                    pump_wavelengths_m=wavelengths,
                    pump_spectral_intensity_W_m2=np.asarray(
                        spectral_intensity,
                        dtype=float,
                    ),
                    grid=self.grid,
                )
                event_heat = validated_heat_component(
                    event_result.heat_W_m3,
                    name="Pump event-energy heat density",
                    expected_shape=self.grid.shape,
                    non_negative=False,
                )
                self._last_event_energy_stats = event_result.stats
                self._last_event_energy_stats[
                    "temporal_applicability"
                ] = "stationary_operating_point_snapshot"
            passive_heat = (
                validated_heat_component(
                    pump_result.passive_absorbed_W_m3,
                    name="Pump passive absorbed-power density",
                    expected_shape=self.grid.shape,
                    non_negative=True,
                )
                if pump_result.passive_absorbed_W_m3 is not None
                else np.zeros(self.grid.shape, dtype=float)
            )
            ase_passive_heat = (
                np.zeros(self.grid.shape, dtype=float)
                if self._last_ase_result is None
                else validated_heat_component(
                    self._last_ase_result.passive_heat_W_m3,
                    name="ASE passive heat density",
                    expected_shape=self.grid.shape,
                    non_negative=True,
                )
            )
            return (
                event_heat + passive_heat + ase_passive_heat
            ) * self.crystal_mask
        if spectral_fraction_model is not None:
            if population is None:
                raise RuntimeError(
                    "spectrally resolved Pump heating requires the operating population"
                )
            if (
                pump_result.spectral_active_absorbed_W_m3 is not None
                and len(self.levels.pump_transitions) == 1
            ):
                wavelengths = np.asarray(
                    pump_result.wavelengths_m, dtype=float
                )
                if spectral_fraction_model == "spectral_quantum_defect":
                    fractions = np.maximum(
                        0.0,
                        1.0
                        - float(self.levels.pump.quantum_efficiency)
                        * wavelengths
                        / reference_wavelength,
                    )
                else:
                    fractions = float(
                        thermal.get("stokes_shift_multiplier", 1.0)
                    ) * np.maximum(0.0, 1.0 - wavelengths / reference_wavelength)
                fractions += float(thermal.get("extra_heat_fraction", 0.0))
                if np.any((fractions < 0.0) | (fractions > 1.0)):
                    raise ValueError(
                        "Every spectrally resolved active Pump heat fraction must lie in [0, 1]"
                    )
                active_heat = np.sum(
                    np.maximum(
                        validated_heat_component(
                            pump_result.spectral_active_absorbed_W_m3,
                            name="Spectral Pump active absorbed-power density",
                            expected_shape=(len(wavelengths), *self.grid.shape),
                            non_negative=False,
                        ),
                        0.0,
                    )
                    * fractions[:, None, None, None],
                    axis=0,
                )
                passive_heat = (
                    validated_heat_component(
                        pump_result.passive_absorbed_W_m3,
                        name="Pump passive absorbed-power density",
                        expected_shape=self.grid.shape,
                        non_negative=True,
                    )
                    if pump_result.spectral_passive_absorbed_W_m3 is None
                    else np.sum(
                        validated_heat_component(
                            pump_result.spectral_passive_absorbed_W_m3,
                            name="Spectral Pump passive absorbed-power density",
                            expected_shape=(len(wavelengths), *self.grid.shape),
                            non_negative=True,
                        ),
                        axis=0,
                    )
                )
                spectrally_resolved_component_heat = active_heat + passive_heat
                self._last_spectral_heat_stats = {
                    "enabled": True,
                    "method": "transported_per_wavelength_absorption_ledger",
                    "wavelengths_m": wavelengths.tolist(),
                    "active_heat_fractions": fractions.tolist(),
                    "spatial_spectrum_resolved": True,
                }
            else:
                fraction_field, self._last_spectral_heat_stats = (
                    spectral_active_heat_fraction(
                        self.levels,
                        population,
                        self.pump_solver.spectrum.coordinate,
                        self.pump_solver.spectrum.weights,
                        reference_wavelength,
                        spectral_fraction_model,
                        float(thermal.get("extra_heat_fraction", 0.0)),
                        float(thermal.get("stokes_shift_multiplier", 1.0)),
                        self.crystal_mask,
                        spectral_active_absorbed_W_m3=(
                            pump_result.spectral_active_absorbed_W_m3
                        ),
                    )
                )
                if configured_fraction not in SPECTRAL_HEAT_MODELS:
                    self._last_spectral_heat_stats[
                        "automatic_multi_transition_upgrade"
                    ] = (
                        f"{configured_fraction} was evaluated per Pump "
                        "transition to preserve distinct quantum efficiencies "
                        "and level populations"
                    )
        if source_model == "pump_component_ledger":
            if (
                pump_result.active_absorbed_W_m3 is None
                or pump_result.passive_absorbed_W_m3 is None
            ):
                raise RuntimeError(
                    "pump_component_ledger requires active/passive Pump absorption components"
                )
            component_wavelengths = (
                np.asarray(pump_result.wavelengths_m, dtype=float)
                if pump_result.wavelengths_m is not None
                else np.asarray(
                    self.pump_solver.spectrum.coordinate,
                    dtype=float,
                )
            )
            self._last_pump_component_applicability = (
                pump_component_ledger_applicability(
                    self.levels,
                    component_wavelengths,
                    self.grid.shape,
                    pump_result.stats.get(
                        "minimum_active_pump_absorption_m_inv"
                    ),
                    negative_absorption_tolerance_m_inv=float(
                        self.sim_cfg.get("strict_validation", {}).get(
                            "pump_component_negative_active_absorption_tolerance_m_inv",
                            1e-12,
                        )
                    ),
                    level_gap_relative_tolerance=float(
                        self.sim_cfg.get("strict_validation", {}).get(
                            "pump_component_level_gap_relative_tolerance",
                            1e-3,
                        )
                    ),
                )
            )
            # Active absorption first creates excitation and therefore uses
            # the configured quantum-defect/extra fraction. Passive host or
            # impurity absorption is non-radiative here and becomes heat in full.
            heat = (
                spectrally_resolved_component_heat
                if spectrally_resolved_component_heat is not None
                else (
                    np.maximum(
                        validated_heat_component(
                            pump_result.active_absorbed_W_m3,
                            name="Pump active absorbed-power density",
                            expected_shape=self.grid.shape,
                            non_negative=False,
                        ),
                        0.0,
                    )
                    * fraction_field
                    + validated_heat_component(
                        pump_result.passive_absorbed_W_m3,
                        name="Pump passive absorbed-power density",
                        expected_shape=self.grid.shape,
                        non_negative=True,
                    )
                )
            )
            if configured_fraction == "lifetime_quantum_defect":
                radiative_lifetime = self.config["energy_levels"].get(
                    "upper_state_radiative_lifetime_s"
                )
                if radiative_lifetime is None:
                    raise RuntimeError(
                        "lifetime_quantum_defect requires an explicit "
                        "upper-state radiative lifetime"
                    )
                nonradiative_heat, lifetime_stats = (
                    lifetime_nonradiative_heat(
                        self.levels,
                        population,
                        float(radiative_lifetime),
                        reference_wavelength,
                        self.crystal_mask,
                    )
                )
                lifetime_stats["heat_power_W"] = float(
                    self.grid.integrate_volume(nonradiative_heat)
                )
                self._last_lifetime_heat_stats = lifetime_stats
                heat = heat + nonradiative_heat
        else:
            raise ValueError(
                "material.thermal.heat_source_model must be "
                "pump_component_ledger or event_energy_ledger"
            )
        ase_passive_heat = (
            np.zeros(self.grid.shape, dtype=float)
            if self._last_ase_result is None
            else validated_heat_component(
                self._last_ase_result.passive_heat_W_m3,
                name="ASE passive heat density",
                expected_shape=self.grid.shape,
                non_negative=True,
            )
        )
        return (heat + ase_passive_heat) * self.crystal_mask

    def _seed_irreversible_heat(
        self,
        signal_result: SignalPropagationResult | None,
    ) -> np.ndarray:
        """Convert irreversible Seed losses to average heat exactly once.

        Population dark-relaxation heat is intentionally excluded until it
        has an event-resolved accumulation path; strict validation rejects
        that combination when the event-energy ledger is selected.
        """

        if signal_result is None:
            return np.zeros(self.grid.shape, dtype=float)

        def validated_loss(
            values: np.ndarray | None,
            *,
            name: str,
        ) -> np.ndarray:
            if values is None:
                return np.zeros(self.grid.shape, dtype=float)
            array = np.asarray(values, dtype=float)
            if array.shape != self.grid.shape:
                raise RuntimeError(f"{name} must match the physical grid")
            if not np.all(np.isfinite(array)):
                raise RuntimeError(f"{name} must be finite")
            scale = max(float(np.max(np.abs(array))), 1.0)
            if float(np.min(array)) < -64.0 * np.finfo(float).eps * scale:
                raise RuntimeError(f"{name} must be non-negative")
            return np.maximum(array, 0.0)

        fraction = float(
            self.material.get("strong_field", {}).get(
                "deposit_fraction_as_heat", 1.0
            )
        )
        nonlinear = validated_loss(
            signal_result.nonlinear_loss_energy_J_m3,
            name="Seed nonlinear-loss energy density",
        ) * fraction
        passive = (
            validated_loss(
                signal_result.passive_loss_energy_J_m3,
                name="Seed passive-loss energy density",
            )
            if bool(
                self.material.get("thermal", {}).get(
                    "include_seed_passive_absorption_heat", False
                )
            )
            else np.zeros(self.grid.shape, dtype=float)
        )
        repetition_rate = float(self.sim_cfg.get("repetition_rate_Hz", 0.0))
        if repetition_rate <= 0.0:
            if np.any(nonlinear > 0.0) or np.any(passive > 0.0):
                raise ValueError(
                    "Seed irreversible-loss heating requires simulation.repetition_rate_Hz > 0"
                )
            return np.zeros(self.grid.shape, dtype=float)
        return (nonlinear + passive) * repetition_rate * self.crystal_mask

    def _resolved_active_heat_fraction(self) -> tuple[float, float, str]:
        thermal = self.material["thermal"]
        configured = thermal.get("heat_fraction", "quantum_defect")
        library = self.material.get("library", {})
        fluorescence_enabled = bool(
            library.get("effective_fluorescence_wavelength_enabled", False)
        )
        fluorescence_wavelength = (
            library.get("effective_fluorescence_wavelength_m")
            if fluorescence_enabled
            else None
        )
        reference_wavelength = float(
            fluorescence_wavelength or self.seed_cfg["wavelength_m"]
        )
        reference_source = (
            "effective_fluorescence_wavelength"
            if fluorescence_wavelength
            else "seed_wavelength_compatibility_fallback"
        )
        if configured == "quantum_defect":
            quantum_efficiency = float(
                self.levels.pump.quantum_efficiency
            )
            fraction_value = max(
                0.0,
                1.0
                - quantum_efficiency
                * float(self.pump_cfg["wavelength_m"])
                / reference_wavelength,
            )
        elif configured == "lifetime_quantum_defect":
            quantum_efficiency = float(
                self.levels.pump.quantum_efficiency
            )
            fraction_value = max(
                0.0,
                1.0
                - quantum_efficiency
                * float(self.pump_cfg["wavelength_m"])
                / reference_wavelength,
            )
            reference_source += "_plus_lifetime_rate_decomposition"
        elif configured == "scaled_quantum_defect":
            stokes_multiplier = float(thermal.get("stokes_shift_multiplier", 1.0))
            fraction_value = max(
                0.0,
                1.0 - float(self.pump_cfg["wavelength_m"]) / reference_wavelength,
            ) * stokes_multiplier
        elif configured in SPECTRAL_HEAT_MODELS:
            wavelengths = np.asarray(
                self.pump_solver.spectrum.coordinate, dtype=float
            )
            weights = np.asarray(self.pump_solver.spectrum.weights, dtype=float)
            weights = weights / np.sum(weights)
            if configured == "spectral_quantum_defect":
                fractions = np.maximum(
                    0.0,
                    1.0
                    - float(self.levels.pump.quantum_efficiency)
                    * wavelengths
                    / reference_wavelength,
                )
            else:
                fractions = float(
                    thermal.get("stokes_shift_multiplier", 1.0)
                ) * np.maximum(0.0, 1.0 - wavelengths / reference_wavelength)
            fraction_value = float(np.sum(weights * fractions))
            reference_source = "spectral_power_weighted_nominal"
        else:
            fraction_value = float(configured)
            reference_source = "explicit_heat_fraction"
        fraction_value += float(thermal.get("extra_heat_fraction", 0.0))
        if not 0.0 <= fraction_value <= 1.0:
            raise ValueError("Total thermal heat fraction must be in [0, 1]")
        return fraction_value, reference_wavelength, reference_source

    def _signal_active_gain(self, population: np.ndarray) -> np.ndarray:
        override = self.config["energy_levels"].get("gain_override_m_inv")
        if override is not None:
            return np.full(self.grid.shape, float(override)) * self.crystal_mask
        gain = self.levels.signal_gain_m_inv(
            population,
            0.0,
            float(self.seed_cfg["wavelength_m"]),
        )
        return gain * self.crystal_mask

    def _remaining_override_gain(
        self, signal_result: SignalPropagationResult
    ) -> np.ndarray:
        """Read the consumed override reservoir from the backend that ran."""

        use_fallback = str(signal_result.model_used or "").endswith("_fallback")

        def reservoir(solver: Any) -> np.ndarray | None:
            if solver is None:
                return None
            if hasattr(solver, "coupler") and hasattr(solver, "inner"):
                optical = reservoir(solver.inner)
                return (
                    None
                    if optical is None
                    else np.asarray(solver.coupler.to_physical(optical), dtype=float)
                )
            if hasattr(solver, "primary") and hasattr(solver, "fallback"):
                selected = solver.fallback if use_fallback else solver.primary
                value = reservoir(selected)
                return value if value is not None else reservoir(solver.fallback)
            transient = getattr(solver, "transient_gain", None)
            transient_value = getattr(transient, "_override_gain_m_inv", None)
            if transient_value is not None:
                return np.asarray(transient_value, dtype=float).copy()
            value = getattr(solver, "_override_gain_m_inv", None)
            return None if value is None else np.asarray(value, dtype=float).copy()

        remaining = reservoir(self.signal_solver)
        if remaining is None or remaining.shape != self.grid.shape:
            raise RuntimeError(
                "The selected saturable gain_override backend did not expose its remaining active-gain reservoir"
            )
        if not np.all(np.isfinite(remaining)):
            raise RuntimeError("Remaining gain_override reservoir contains non-finite values")
        return remaining * self.crystal_mask

    def _diagnostics(self, pump_stats: dict[str, Any], signal_source_J: float, warnings: list[str]) -> list[str]:
        result = list(warnings)
        nonlinear_diagnostics = self.levels.nonlinear_diagnostics
        if int(nonlinear_diagnostics.get("cap_limited_calls", 0)) > 0:
            result.append(
                "The adaptive nonlinear ion-kinetics solver exhausted maximum_substeps in "
                f"{int(nonlinear_diagnostics['cap_limited_calls'])} call(s); its configured "
                "fractional-depletion target was not met. Increase "
                "energy_levels.nonlinear_solver.maximum_substeps and check convergence."
            )
        correction = self.levels.population_correction_diagnostics
        maximum_negative_fraction = float(
            correction.get("maximum_negative_population_fraction", 0.0)
        )
        maximum_renormalization = float(
            correction.get(
                "maximum_conservation_error_before_renormalization_relative",
                0.0,
            )
        )
        if max(maximum_negative_fraction, maximum_renormalization) > 1e-8:
            result.append(
                "The population solver required a numerical non-negativity or ion-conservation "
                "correction above 1e-8 relative magnitude "
                f"(negative={maximum_negative_fraction:.3e}, "
                f"renormalization={maximum_renormalization:.3e}). Reduce the time step, "
                "inspect nonlinear rates, and do not treat the corrected run as converged "
                "without a refinement check."
            )
        provenance = self._configuration_provenance()
        if not provenance["material_record_traceable"]:
            result.append(
                "The configured named material has no imported record hash. Its numeric values are runnable, "
                "but the material source cannot be reproduced from the name alone; apply a material-library "
                "record or preserve an explicit parameter snapshot with the setup."
            )
        if not provenance.get("material_snapshot_matches_runtime", True):
            result.append(
                "Runtime material/level values differ from the imported material snapshot: "
                + ", ".join(provenance.get("material_snapshot_mismatches", {}))
                + ". The run remains valid as a user override, but it is no longer an unchanged database reproduction."
            )
        if not provenance.get("setup_material_hash_matches", True):
            result.append(
                "setup_provenance.material_record_sha256 does not match the embedded material record; "
                "the setup may have been combined with a different material file."
            )
        source_verification = provenance.get("setup_source_verification")
        if source_verification == "mismatch":
            result.append(
                "setup_provenance.source_file exists but its SHA-256 does not match source_sha256; "
                "the external setup source has changed since this JSON was aligned."
            )
        elif source_verification == "unavailable":
            result.append(
                "setup_provenance.source_file is unavailable, so the stored source SHA-256 could not be "
                "rechecked. The embedded JSON remains runnable but external-source traceability is incomplete."
            )
        geometry = self.material.get("geometry", {})
        grid_width = float(self.grid.x_edges[-1] - self.grid.x_edges[0])
        grid_height = float(self.grid.y_edges[-1] - self.grid.y_edges[0])
        if geometry.get("shape", "cuboid") == "cuboid":
            configured_width = geometry.get("width_x_m")
            configured_height = geometry.get("height_y_m")
            width = float(
                grid_width if configured_width is None else configured_width
            )
            height = float(
                grid_height if configured_height is None else configured_height
            )
            if width > grid_width or height > grid_height:
                result.append("The cuboid crystal is larger than the transverse numerical window and is truncated by the grid.")
        else:
            configured_diameter = geometry.get("diameter_m")
            diameter = float(
                min(grid_width, grid_height)
                if configured_diameter is None
                else configured_diameter
            )
            if diameter > min(grid_width, grid_height):
                result.append("The cylindrical crystal diameter exceeds the transverse numerical window and is truncated by the grid.")
        doping = self.doping_profile_summary
        if bool(doping.get("enabled", False)):
            clipped = int(doping.get("raw_clipped_cell_count", 0))
            if clipped:
                result.append(
                    f"The dopant profile reached minimum_raw_fraction in {clipped} active cells; "
                    "check linear-gradient amplitudes or treat this clipping as an explicit model choice."
                )
            direction_cell_size = {
                "x_min": float(self.grid.x_widths[0]),
                "x_max": float(self.grid.x_widths[-1]),
                "y_min": float(self.grid.y_widths[0]),
                "y_max": float(self.grid.y_widths[-1]),
                "z_min": float(self.grid.z_widths[0]),
                "z_max": float(self.grid.z_widths[-1]),
                "radial_outer": min(
                    float(np.min(self.grid.x_widths)),
                    float(np.min(self.grid.y_widths)),
                ),
            }
            under_resolved: list[str] = []
            for component in doping.get("components", []):
                component_type = component.get("type", "surface_exponential")
                if component_type not in {"surface_exponential", "axis_symmetric"}:
                    continue
                direction = str(component.get("direction", "z_min"))
                if component_type == "axis_symmetric":
                    decay_length = float(component.get("transition_length_m", 0.0))
                    direction = {
                        "x": "x_min",
                        "y": "y_min",
                        "z": "z_min",
                        "radial_xy": "radial_outer",
                    }.get(direction, direction)
                else:
                    decay_length = float(component.get("decay_length_m", 0.0))
                local_cell = direction_cell_size.get(direction)
                if local_cell is not None and decay_length < 2.0 * local_cell:
                    under_resolved.append(
                        f"{direction}: L/cell={decay_length / local_cell:.3g}"
                    )
            if under_resolved:
                result.append(
                    "Dopant exponential profiles are sampled by fewer than two cells in their depth direction ("
                    + ", ".join(under_resolved)
                    + "); refine the corresponding axes before quantitative use."
                )
        sampling = {
            "pump source waist x": float(self.pump_cfg["waist_x_m"]) / float(np.min(self.pump_optical_grid.x_widths)),
            "pump source waist y": float(self.pump_cfg["waist_y_m"]) / float(np.min(self.pump_optical_grid.y_widths)),
            "seed source waist x": float(self.seed_cfg["waist_x_m"]) / float(np.min(self.optical_grid.x_widths)),
            "seed source waist y": float(self.seed_cfg["waist_y_m"]) / float(np.min(self.optical_grid.y_widths)),
        }
        for name, cells in sampling.items():
            if cells < 3.0:
                result.append(f"{name} is sampled by only {cells:.2f} cells per radius; refine the transverse grid.")
        configured_pump = float(self.pump_cfg["power_W"])
        if configured_pump > 0:
            capture = pump_stats["grid_source_power_W"] / configured_pump
            if not 0.98 <= capture <= 1.02:
                result.append(f"Pump source plane captures {capture:.3%} of configured power; enlarge or refine the grid.")
        configured_seed = float(self.seed_cfg["pulse_energy_J"])
        if configured_seed > 0:
            capture = signal_source_J / configured_seed
            if not 0.98 <= capture <= 1.02:
                result.append(f"Seed source plane captures {capture:.3%} of configured energy; enlarge or refine the grid.")
        return result

    def _grid_coupling_summary(self) -> dict[str, Any]:
        single_grid = {"enabled": False, "method": "single_grid"}
        seed = (
            self.grid_coupler.summary()
            if self.grid_coupler is not None
            else dict(single_grid)
        )
        pump = (
            self.pump_grid_coupler.summary()
            if self.pump_grid_coupler is not None
            else dict(single_grid)
        )
        # Keep the historical flat fields for existing result readers.  Seed
        # is preferred when present; otherwise a Pump-only custom grid is no
        # longer incorrectly reported as a single-grid run.
        preferred = seed if seed["enabled"] else pump
        grid_cfg = self.config["grid"]
        physical_center = [
            float(grid_cfg.get("center_x_m", 0.0)),
            float(grid_cfg.get("center_y_m", 0.0)),
        ]
        refinement_raw = [
            grid_cfg.get("refinement_center_x_m"),
            grid_cfg.get("refinement_center_y_m"),
        ]
        physical_refinement_center = [
            physical_center[index]
            if configured is None
            else float(configured)
            for index, configured in enumerate(refinement_raw)
        ]
        return {
            **preferred,
            "enabled": bool(seed["enabled"] or pump["enabled"]),
            "seed": seed,
            "pump": pump,
            "shared_optical_grid": bool(
                self.pump_optical_grid is self.optical_grid
            ),
            "physical_spacing_xy": str(
                grid_cfg.get("spacing_xy", "uniform")
            ),
            "physical_refinement_active": (
                str(grid_cfg.get("spacing_xy", "uniform"))
                in {"center_stretched", "refined_region"}
            ),
            "physical_refined_region": (
                copy.deepcopy(grid_cfg.get("refined_region", {}))
                if grid_cfg.get("spacing_xy") == "refined_region" else None
            ),
            "physical_grid_center_xy_m": physical_center,
            "physical_refinement_center_xy_m": physical_refinement_center,
            "physical_refinement_center_independent_xy": [
                configured is not None for configured in refinement_raw
            ],
        }

    def _configuration_provenance(self) -> dict[str, Any]:
        library = self.material.get("library", {})
        record_hash = library.get("record_sha256")
        setup_provenance = copy.deepcopy(self.config.get("setup_provenance", {}))
        snapshot = copy.deepcopy(library.get("applied_parameter_snapshot", {}))
        levels = self.config["energy_levels"]
        current = {
            "ion_density_m3": self.material.get("ion_density_m3"),
            "refractive_index": self.material.get("refractive_index"),
            "pump_refractive_index": self.material.get("pump_refractive_index"),
            "dn_dT_per_K": self.material.get("dn_dT_per_K"),
            "pump_absorption_cross_section_m2": levels.get(
                "pump_absorption_cross_section_m2"
            ),
            "pump_emission_cross_section_m2": levels.get(
                "pump_emission_cross_section_m2"
            ),
            "signal_absorption_cross_section_m2": levels.get(
                "signal_absorption_cross_section_m2"
            ),
            "signal_emission_cross_section_m2": levels.get(
                "signal_emission_cross_section_m2"
            ),
            "upper_state_lifetime_s": levels.get("upper_state_lifetime_s"),
            "pump_quantum_efficiency": levels.get("pump_quantum_efficiency"),
        }
        mismatches: dict[str, dict[str, Any]] = {}
        for key, expected in snapshot.items():
            actual = current.get(key)
            if expected is None and actual is None:
                continue
            try:
                matches = bool(
                    np.isclose(
                        float(actual),
                        float(expected),
                        rtol=1e-12,
                        atol=0.0,
                    )
                )
            except (TypeError, ValueError):
                matches = actual == expected
            if not matches:
                mismatches[key] = {"snapshot": expected, "runtime": actual}
        material_name = str(self.material.get("name", "user_material"))
        generic_name = material_name.strip().lower() in {
            "",
            "user_material",
            "custom",
            "unnamed",
        }
        source_file = setup_provenance.get("source_file")
        expected_source_hash = setup_provenance.get("source_sha256")
        runtime_source_hash = None
        source_read_error = None
        if source_file and expected_source_hash:
            try:
                source_path = Path(str(source_file)).expanduser()
                if source_path.is_file():
                    runtime_source_hash = hashlib.sha256(
                        source_path.read_bytes()
                    ).hexdigest()
                    source_verification = (
                        "match"
                        if runtime_source_hash == expected_source_hash
                        else "mismatch"
                    )
                else:
                    source_verification = "unavailable"
            except OSError as exc:
                source_verification = "unavailable"
                source_read_error = str(exc)
        else:
            source_verification = "not_configured"
        return {
            "config_sha256": self.config_sha256,
            "schema_policy": self.config.get(
                "schema_policy", "strict_core_with_plugin_extras"
            ),
            "simulator_thread_safe": self.thread_safe,
            "simulator_reentrant": self.reentrant,
            "parallel_sweep_contract": "independent_instance_per_point",
            "material_name": material_name,
            "material_record_traceable": bool(record_hash) or generic_name,
            "material_record_sha256": record_hash,
            "material_record_id": library.get("record_id"),
            "material_source": copy.deepcopy(library.get("source", {})),
            "doping_percent": library.get("doping_percent"),
            "operating_temperature_K": library.get("operating_temperature_K"),
            "applied_parameter_snapshot": snapshot,
            "material_snapshot_matches_runtime": not mismatches,
            "material_snapshot_mismatches": mismatches,
            "setup_provenance": setup_provenance,
            "setup_source_verification": source_verification,
            "setup_source_runtime_sha256": runtime_source_hash,
            "setup_source_read_error": source_read_error,
            "setup_material_hash_matches": bool(
                not setup_provenance.get("material_record_sha256")
                or setup_provenance.get("material_record_sha256") == record_hash
            ),
            "note": (
                "The config digest identifies the fully merged runtime input; the material-record digest "
                "identifies the normalized imported database record before doping/temperature evaluation. "
                "When an external setup source and digest are configured, the source file is rechecked at runtime."
            ),
        }

    def _material_library_warnings(self) -> list[str]:
        library = self.material.get("library", {})
        warnings = list(library.get("warnings", []))
        if library.get("effective_fluorescence_wavelength_enabled", False):
            warnings = [
                warning
                for warning in warnings
                if "disabled for heating until explicitly enabled" not in warning
            ]
        return warnings

    def _assumptions(
        self,
        model_used: str | None = None,
        cpa_diagnostics: dict[str, Any] | None = None,
        gain_transport: dict[str, Any] | None = None,
    ) -> list[str]:
        actual_model = model_used or self.signal_model
        polarization_model = str(
            self.seed_cfg.get("polarization", {}).get("model", "scalar")
        )
        mechanical_model = str(
            self.material.get("mechanical", {}).get("stress_model", "none")
        )
        if mechanical_model == "thermoelastic_fem":
            mechanical_assumption = (
                "Thermal stress uses the optional 3D small-strain isotropic thermoelastic FEM. "
                "It solves force balance with traction-free or explicitly fixed faces and reports "
                "displacement plus the full symmetric stress/strain components. Contact mechanics, "
                "fracture statistics, coating/heat-sink mechanics and local deformed-interface normals "
                "are outside this model; optional endface feedback is a distributed OPD closure."
            )
        elif mechanical_model == "anisotropic_thermoelastic_fem":
            mechanical_assumption = (
                "Thermal stress uses 3D small-strain anisotropic thermoelastic FEM with rotated "
                "stiffness and expansion tensors. Optional rotated photoelastic tensors drive local "
                "Jones retardance. Contact/coating/heat-sink mechanics, fracture statistics and "
                "surface-normal ray deflection remain outside scope; endface feedback, when enabled, "
                "uses the FEM front/back sag as a distributed equivalent OPD."
            )
        elif mechanical_model == "isotropic_proxy":
            mechanical_assumption = (
                "Thermal stress uses isotropic_proxy, which is not a force-balanced FEM and does "
                "not calculate displacement, tensor photoelastic birefringence, or fracture margins."
            )
        else:
            mechanical_assumption = "Thermo-mechanical stress and displacement solving are disabled."
        assumptions = [
            (
                f"Signal model is {actual_model}; Seed polarization is propagated as a two-component Jones field "
                "in fixed laboratory x/y axes. Diffraction and scalar gain act independently and identically on both components."
                if polarization_model == "jones"
                else f"Signal model is {actual_model}; fields are scalar and polarization is not propagated as a Jones vector."
            ),
            f"External seed propagation is {self.physics['optical_propagation']}; pump propagation is {self.physics['pump_propagation']}.",
            f"Population dynamics is {self.physics['population_dynamics']}; gain saturation is {self.physics['gain_saturation']}.",
            (
                "Commercial App Fast advances independent HG Gaussian-q channels; modes at one wavelength "
                "share one local population update and are summed incoherently."
                if actual_model == "commercial_app_fast"
                and self.seed_cfg.get("spatial_modes", {}).get("model")
                == "incoherent_hg"
                else "FFT propagation is paraxial with periodic transverse boundaries; independently propagated "
                "HG modes share one population reservoir and are summed incoherently."
                if self.seed_cfg.get("spatial_modes", {}).get("model") == "incoherent_hg"
                else "FFT propagation is paraxial with periodic transverse boundaries; effective-M2 remains the inexpensive reduced-order path."
            ),
            (
                "Heat source model is "
                f"{self.material['thermal'].get('heat_source_model', 'pump_component_ledger')}; "
                "ASE passive loss and Seed irreversible loss enter heat only when their explicit coupling switches are enabled; "
                "fluorescence escape is not a separate photon-transport solve."
            ),
            (
                "Radial/tangential thermal photoelastic birefringence uses the configured plane-stress coefficients; "
                "the photoelastic temperature is referenced to the local axis T(r,z)-T(0,z), so a uniform temperature "
                "rise produces no stress retardance. It is an axisymmetric reciprocal local Jones retarder and is not "
                "a force-balanced FEM, square-boundary/crystal-cut tensor model, end-face-deformation, or fracture model."
                if self.seed_cfg.get("polarization", {}).get("birefringence_model", "off")
                == "radial_tangential_plane_stress"
                else mechanical_assumption
            ),
        ]
        if (
            str(self.config.get("multiphysics", {}).get("coupling_model", "one_way"))
            == "thermal_optical_fixed_point"
        ):
            assumptions.append(
                "Thermal-optical fixed point iterates sampled-field Pump propagation, population, "
                "Pump/ASE heat, temperature/stress and delta-n; optional local spectroscopy and "
                "irreversible Seed heat follow their explicit coupling switches. Mechanics remain one-way."
            )
        else:
            assumptions.append(
                "One-way coupling solves Pump/population first and applies the resulting thermal delta-n only to the reported Seed propagation."
            )
        if self.ase_solver.enabled:
            if str(self.config.get("ase", {}).get("model")) == "angular_discrete_ordinates":
                assumptions.append(
                    "ASE uses arbitrary-angle stationary discrete ordinates and a matrix-free reflected-boundary "
                    "eigenvalue. Reflection is resolved on Cartesian external faces; curved embedded TIR, scattering "
                    "and coherent wave-optical resonator modes require a body-fitted/noise-seeded backend."
                )
            else:
                assumptions.append(
                    "ASE uses the retained fast six-direction discrete-ordinates screen; effective face reflectivity "
                    "closes axis-aligned parasitic loops, but arbitrary oblique/TIR paths are not resolved."
                )
        if str(self.pump_cfg.get("spectrum", {}).get("spatial_transport", "shared")) == "per_wavelength":
            assumptions.append(
                "Pump spectrum is propagated wavelength by wavelength with independent diffraction, absorption and "
                "optical routing; all bins share the same local population reservoir."
            )
        else:
            assumptions.append(
                "Broadband Pump uses the retained shared-spatial-field reduced model unless spectrum.spatial_transport=per_wavelength is selected."
            )
        if self.levels.nonlinear:
            assumptions.append(
                "Configured nonlinear ion-transfer processes update population, but the generic Pump-fraction heat model "
                "does not yet derive transition-resolved multiphonon/upconversion heat from level energies."
            )
        if self.levels.response.enabled:
            assumptions.append(
                "Temperature/concentration spectroscopy corrections use only user-provided laws/tables and clamp table "
                "extrapolation to endpoint data; no undocumented host-specific coefficients are inserted."
            )
        if self.physics.get("nonlinear_loss_model") in {
            "strong_field_time_domain",
            "strong_field_effective_area",
        }:
            assumptions.append(
                "Strong-field loss integrates dI/dz=-beta*I^K analytically per z step; optional carrier generation, "
                "FCA and carrier refraction are scalar local rate equations. The Commercial App effective-area variant applies "
                "the Gaussian transverse Kth-moment closure rather than a sampled x-y field."
            )
        if self.grid_coupler is not None or self.pump_grid_coupler is not None:
            assumptions.append(
                "Pump and Seed optics may use separate uniform transverse grids; finite-volume cell-overlap remapping couples "
                "optical source terms, population changes, and thermal delta-n to the full crystal grid."
            )
        if bool(self.sim_cfg.get("multipass", {}).get("enabled", False)) and not uses_causal_clock(
            self.sim_cfg, self.seed_cfg, self.signal_model
        ):
            assumptions.append(
                "The unified multipass hit/transition plan is executed sequentially by this backend; it is paraxial and does not resolve simultaneous crystal occupancy or mechanical mount collisions."
            )
        library = self.material.get("library", {})
        if library:
            assumptions.append(
                f"Imported concentration/temperature-dependent material data were evaluated at "
                f"{float(library.get('doping_percent', 0.0)):g}% and "
                f"{float(library.get('operating_temperature_K', 293.15)):g} K before the run; "
                + (
                    "the configured local response tables/laws are additionally re-evaluated at each voxel temperature."
                    if self.levels.response.mode == "local_3d"
                    else "the imported snapshot itself remains fixed unless local_3d response data are configured."
                )
            )
            assumptions.append(
                f"Material database anisotropy is projected onto the selected {library.get('polarization', 'x')} axis; "
                + (
                    "the Jones propagator then applies the separately configured radial/tangential thermal retarder."
                    if polarization_model == "jones"
                    else "the propagated optical field remains scalar."
                )
            )
            if library.get("rate_system", {}).get("usable", True):
                assumptions.append(
                    "Imported proprietary rate-line topology is provenance metadata only; the active runtime uses the selected built-in/custom LevelSystem transitions."
                )
            elif library.get("rate_system"):
                assumptions.append(
                    "The imported proprietary rate-line topology is quarantined and unusable because its species metadata failed validation; the runtime uses only the explicitly configured LevelSystem."
                )
        if actual_model == "commercial_app_fast":
            transport = gain_transport or {}
            temporal_closure = str(
                transport.get(
                    "temporal_closure", "fixed_input_effective_duration"
                )
            )
            if bool(transport.get("incoherent_hg_modes", False)):
                assumptions.append(
                    "Commercial App fast expands the configured incoherent Hermite-Gaussian basis into independent q/energy channels. "
                    "At each wavelength and ordered time bin, modal fluences are combined before one shared local population "
                    "update and the output is distributed back by local modal input fraction."
                )
                assumptions.append(
                    "Each retained HG mode keeps its own beam size, curvature and centroid while thermal phase, Kerr phase "
                    "and gain-guided distortion are projected onto that mode's second moments; complete x-y complex-field "
                    "diffraction and coherent interference between modes are not represented."
                )
            elif temporal_closure in {
                "one_dimensional_effective_area_spm",
                "one_dimensional_effective_area_strong_field",
            }:
                assumptions.append(
                    "Commercial App fast keeps wavelength-resolved gain and physical population exchange in every crystal cell and "
                    "collapses gain saturation to one whole-pulse fluence packet; a separate one-dimensional coherent "
                    "effective-area envelope retains configured phase, material/element dispersion, SPM broadening, "
                    "output duration and compressor diagnostics."
                )
                assumptions.append(
                    "Thermal phase, spatial Kerr lensing and gain-guided structure are power-preservingly projected onto "
                    "axis-aligned astigmatic Gaussian centroid/radius/tilt/curvature moments. The one-dimensional envelope "
                    "does not resolve radial spectral broadening or other space-time coupling."
                )
                if temporal_closure == "one_dimensional_effective_area_strong_field":
                    assumptions.append(
                        "The Commercial App L3.5 strong-field closure applies multiphoton loss, causal carrier generation, FCA and "
                        "carrier refraction to the one-dimensional envelope using the Gaussian transverse Kth moment. "
                        "Its inter-pulse carrier state is a local scalar fixed point; radial carrier gradients and a "
                        "spatial Seed-heat deposition map require coherent L4/L4.5 validation."
                    )
            elif temporal_closure == "cw_average_intensity":
                assumptions.append(
                    "Commercial App fast keeps wavelength-resolved gain and physical population exchange in every crystal cell; "
                    "CW Kerr lensing uses transported average intensity and no artificial pulse envelope, SPM spectrum, "
                    "pulse duration or compressor result is constructed."
                )
                assumptions.append(
                    "Thermal phase, Kerr phase and gain-guided structure are power-preservingly projected onto axis-aligned "
                    "astigmatic Gaussian centroid/radius/tilt/curvature moments."
                )
            elif temporal_closure == "time_binned_intrapulse_population_relaxation":
                assumptions.append(
                    "Commercial App fast uses ordered front-to-tail gain-time bins. Each bin sees the population left by earlier "
                    "bins and the configured dark level kinetics are advanced between bins; retarded propagation time is reduced."
                )
                assumptions.append(
                    "All time bins share one transverse Gaussian-q state per wavelength; thermal phase, Kerr phase and "
                    "gain-guided structure retain only centroid/radius/tilt/curvature moments."
                )
            else:
                assumptions.append(
                    "Commercial App fast keeps wavelength-resolved gain and physical population exchange in every crystal cell, "
                    "but collapses the pulse to one whole-fluence packet and uses a fixed input effective duration for peak intensity."
                )
                assumptions.append(
                    "Thermal phase, Kerr phase and gain-guided structure are power-preservingly projected onto axis-aligned "
                    "astigmatic Gaussian centroid/radius/tilt/curvature moments; output SPM spectrum, pulse duration and compressor peak power are not predicted."
                )
            if transport.get("minimum_gaussian_projection_overlap", 1.0) < 0.99:
                assumptions.append(
                    "The reported Gaussian projection overlap indicates material non-Gaussian content was discarded; use coherent_cpa/FFT for a quantitative wavefront or hot-spot result."
                )
        elif actual_model == "coherent_cpa":
            transient = (cpa_diagnostics or {}).get("transient_gain", {})
            if transient.get("status") == "applied":
                assumptions.append(
                    "Coherent intrapulse saturation uses ordered overlapping soft time windows; each window receives wavelength-dependent frequency-domain gain and commits stimulated population changes before the next window."
                )
                assumptions.append(
                    "This is a rate-equation/SVEA model, not Maxwell-Bloch: optical polarization, finite dephasing, coherent Rabi dynamics, dynamic intrapulse ASE, and Raman response are not included."
                )
            else:
                assumptions.append(
                    "Coherent phase, dispersion, diffraction, gain narrowing, and optional Kerr phase are retained; intrapulse gain saturation uses the fast local whole-fluence approximation."
                )
        elif actual_model == "spectral_time_causal_multipass":
            transport = (gain_transport or {}).get("multipass", {})
            assumptions.append(
                "Finite multipass Seed transport compiles the selected template "
                "into crystal hits and external transitions, then advances all "
                "active pass channels on one absolute-time CFL grid. Spatially "
                "and temporally coincident fluences share one local saturated "
                "population event; overlap is not a user-selected correction."
            )
            assumptions.append(
                "The solver represents one injected pulse, not coexistence of "
                "successive source pulses. It uses a shared reference group "
                "velocity and a bounded common path-cosine approximation "
                f"(reported span {float(transport.get('relative_path_cosine_span', 0.0)):.6g}); "
                "Nz/time-window and transverse-grid convergence remain required."
            )
        elif actual_model == "josa_transport":
            transport = gain_transport or {}
            if transport.get("mode") == JOSA_FULL_3D_MODE:
                assumptions.append(
                    "JOSA full_3d_pc resolves x/y FFT diffraction, z transport, ordered time bins, wavelength bins, "
                    "spectral group-delay walk-off, signed in-crystal incidence walk-off, physical population depletion, "
                    "Pump FFT/depletion, three-dimensional heat flow and the resulting Seed thermal phase."
                )
                assumptions.append(
                    "The group-delay remap conserves time-bin energy but collapses multiple incoherent source-bin "
                    "contributions to one local phase-moment field before later FFT diffraction; full_3d_pc is therefore "
                    "a phase-closure approximation, not strict incoherent-channel propagation. Mutually coherent spectral "
                    "phase, GDD/TOD pulse compression and Kerr SPM belong to coherent_cpa."
                )
            else:
                assumptions.append(
                    "Seed gain uses the selectable JOSA B photon-transport backend in characteristic coordinates; "
                    "the Explicit/PC local update replaces Frantz-Nodvik while configured x/y diffraction and optical routing are retained."
                )
            assumptions.append(
                "The JOSA transport branch is a population-rate/SVEA model, not Maxwell-Bloch; optical polarization, "
                "finite dephasing, coherent Rabi dynamics, dynamic intrapulse ASE, and Raman response are not included. "
                "Stationary six-direction ASE may still clamp the pre-pulse population when its separate module is enabled."
            )
        elif actual_model == "josa_transport_fallback":
            assumptions.append(
                "The optional JOSA transport backend failed safely; this result uses the legacy spectral-time Frantz-Nodvik solver."
            )
        elif actual_model == "spectral_time_fallback":
            assumptions.append("The optional coherent CPA module failed safely; this result uses the phase-insensitive spectral-time fallback.")
        return assumptions

    @staticmethod
    def _log_expm1(value: np.ndarray) -> np.ndarray:
        return _log_expm1(value)

    @staticmethod
    def _frantz_nodvik(fluence: np.ndarray, gain_exponent: np.ndarray, saturation: float) -> np.ndarray:
        return frantz_nodvik(fluence, gain_exponent, saturation)
