"""Pump propagation strategies, separated from population and signal solvers."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .beams import BeamProfile, make_beam
from .dispersion import material_refractive_index
from .grid import Grid3D
from .geometry import (
    crystal_aperture_clipping_required,
    crystal_transverse_mask,
)
from .levels import LevelSystem
from .optical_system import (
    ExternalField,
    additive_source_gain_angle_warning,
    apply_endface_sag_phase,
    build_external_field,
    gain_interface,
    gain_interface_element,
    gain_medium_axis_angles_rad,
    gain_medium_path_cosine,
    gain_surface_power_transmission,
    refract_axis_angles_rad,
)
from .optics import (
    OpticalTrace,
    angular_spectrum,
    gaussian_abcd_to_plane,
    interface_power_transmission,
    shift_field,
    source_field,
    super_gaussian_abcd_to_plane,
)
from .profiles import spectral_profile
from .profiling import profiled_operation
from .pump_raytrace import RayEntrance, build_ray_entrance, propagate_ray_volume
from .pump_source import canonicalize_pump_source
from .pass_routing import resolve_pump_direction


@dataclass
class PumpPropagationResult:
    intensity_W_m2: np.ndarray
    absorbed_W_m3: np.ndarray
    stats: dict[str, Any]
    trace: OpticalTrace
    warnings: list[str]
    active_absorbed_W_m3: np.ndarray | None = None
    passive_absorbed_W_m3: np.ndarray | None = None
    wavelengths_m: np.ndarray | None = None
    spectral_intensity_W_m2: np.ndarray | None = None
    spectral_active_absorbed_W_m3: np.ndarray | None = None
    spectral_passive_absorbed_W_m3: np.ndarray | None = None
    event_energy_heat_W_m3: np.ndarray | None = None
    event_energy_stats: dict[str, Any] | None = None


@dataclass(frozen=True)
class PumpMetricSpec:
    """Declare how one scalar Pump diagnostic combines across independent bins."""

    spectral_reduction: str
    temporal_reduction: str


_ADDITIVE_PUMP_POWER_METRICS = {
    "absorbed_power_W",
    "active_absorbed_power_W",
    "configured_power_W",
    "crystal_power_balance_residual_W",
    "crystal_side_aperture_loss_W",
    "crystal_transport_accounted_power_W",
    "external_train_loss_W",
    "external_train_output_power_W",
    "gain_entrance_analytic_power_W",
    "gain_entrance_crystal_aperture_loss_W",
    "gain_entrance_power_before_crystal_aperture_W",
    "gain_entrance_sampled_power_W",
    "grid_input_power_W",
    "grid_source_power_W",
    "nominal_power_before_geometric_efficiency_W",
    "optical_grid_absorbed_power_W",
    "output_power_W",
    "passive_absorbed_power_W",
    "prescribed_caustic_unresolved_redistribution_power_W",
    "ray_power_truncation_loss_W",
    "source_to_exit_accounted_power_W",
    "source_to_exit_balance_residual_W",
    "source_unit_power_W",
}


PUMP_METRIC_SPECS: dict[str, PumpMetricSpec] = {
    name: PumpMetricSpec("sum", "mean")
    for name in sorted(_ADDITIVE_PUMP_POWER_METRICS)
}


def pump_metric_spec(name: str) -> PumpMetricSpec | None:
    """Return an explicit reducer contract; names never imply semantics."""

    return PUMP_METRIC_SPECS.get(str(name))


class PumpPropagator:
    """Select prescribed, Gaussian, sampled-wave, or stochastic-ray transport."""

    def __init__(
        self,
        grid: Grid3D,
        material: dict[str, Any],
        pump_cfg: dict[str, Any],
        beam: BeamProfile,
        levels: LevelSystem,
        physics: dict[str, Any],
        pre_elements: list[dict[str, Any]],
        gain_marker: dict[str, Any],
    ):
        self.grid = grid
        self.material = material
        self.cfg = pump_cfg
        self.beam = beam
        self.levels = levels
        self.physics = physics
        self.pre_elements = pre_elements
        self.gain_marker = gain_marker
        absorption_transport = self.cfg.get("absorption_transport", {})
        self.absorption_transport_model = str(
            absorption_transport.get("model", "population_dependent")
        )
        fixed_active_coefficient = absorption_transport.get(
            "fixed_active_coefficient_m_inv"
        )
        self.fixed_active_absorption_m_inv = (
            None
            if fixed_active_coefficient is None
            else float(fixed_active_coefficient)
        )
        self.spectrum = spectral_profile(
            self.cfg.get("spectrum", {}),
            float(self.cfg["wavelength_m"]),
            float(self.cfg.get("spectral_bandwidth_m", 0.0)),
        )
        self.crystal_mask = crystal_transverse_mask(grid, material)
        self.clip_crystal_configured = bool(
            material.get("geometry", {}).get("clip_optical_field", True)
        )
        self.clip_crystal = crystal_aperture_clipping_required(
            material, self.crystal_mask
        )
        self._fft_external_cache: ExternalField | None = None
        self._hybrid_external_cache: tuple[ExternalField, dict[str, Any]] | None = None
        self._raytrace_external_cache: RayEntrance | None = None
        self._endface_surface_sag_m: dict[str, np.ndarray | None] = {
            "front": None,
            "back": None,
        }

    def set_endface_surface_sag(
        self,
        front_surface_sag_m: np.ndarray | None,
        back_surface_sag_m: np.ndarray | None,
    ) -> None:
        self._endface_surface_sag_m = {
            "front": (
                None
                if front_surface_sag_m is None
                else np.asarray(front_surface_sag_m, dtype=float).copy()
            ),
            "back": (
                None
                if back_surface_sag_m is None
                else np.asarray(back_surface_sag_m, dtype=float).copy()
            ),
        }

    def _source_power(self, power_scale: float) -> float:
        configured = float(self.cfg["power_W"])
        absolute_file = bool(
            str(self.cfg.get("temporal", {}).get("type", "cw")) == "file"
            and str(
                self.cfg.get("temporal", {}).get("value_mode", "relative")
            )
            == "absolute_W"
        )
        # For an absolute-W file, ``power_scale`` carries watts when the
        # nominal configuration is zero.  This avoids silently erasing a real
        # waveform while retaining the historical dimensionless scale for all
        # other source definitions.
        source = (
            float(power_scale)
            if absolute_file and configured == 0.0
            else configured * float(power_scale)
        )
        return source * float(self.cfg.get("geometric_efficiency", 1.0))

    @profiled_operation("pump.propagate")
    def propagate(
        self,
        population: np.ndarray,
        record_planes: bool = False,
        power_scale: float = 1.0,
        delta_n: np.ndarray | None = None,
    ) -> PumpPropagationResult:
        self._minimum_total_absorption_m_inv = np.inf
        self._maximum_total_absorption_m_inv = -np.inf
        self._minimum_active_absorption_m_inv = np.inf
        self.levels.record_spectral_support(
            "pump", self.spectrum.coordinate, self.spectrum.weights
        )
        mode = self.physics["pump_propagation"]
        if (
            str(self.cfg.get("spectrum", {}).get("spatial_transport", "shared"))
            == "per_wavelength"
            and len(self.spectrum.coordinate) > 1
        ):
            result = self._spectrally_resolved(
                population,
                record_planes,
                power_scale,
                delta_n,
            )
        elif mode == "prescribed_caustic":
            result = self._prescribed(population, record_planes, power_scale, delta_n)
        elif mode in {"abcd", "super_gaussian_abcd"}:
            result = self._abcd(
                population,
                record_planes,
                power_scale,
                delta_n,
                preserve_super_gaussian=mode == "super_gaussian_abcd",
            )
        elif mode == "fft":
            result = self._fft(population, record_planes, power_scale, delta_n)
        elif mode == "hybrid_abcd_fft":
            result = self._hybrid_abcd_fft(
                population,
                record_planes,
                power_scale,
                delta_n,
            )
        elif mode == "stochastic_raytrace":
            result = self._stochastic_raytrace(
                population,
                record_planes,
                power_scale,
                delta_n,
            )
        else:
            raise ValueError(f"Unknown pump propagation mode {mode!r}")
        result.stats.setdefault(
            "spectral_spatial_transport",
            "shared"
            if len(self.spectrum.coordinate) > 1
            else "monochromatic",
        )
        result.stats.setdefault(
            "spectral_spatial_sample_count", len(self.spectrum.coordinate)
        )
        result.stats["absorption_transport_model"] = (
            self.absorption_transport_model
        )
        result.stats["fixed_active_absorption_m_inv"] = (
            self.fixed_active_absorption_m_inv
            if self.absorption_transport_model
            == "fixed_active_coefficient"
            else None
        )
        angle_warning = additive_source_gain_angle_warning(
            self.cfg,
            [self.gain_marker],
            "Pump",
        )
        if angle_warning is not None:
            result.warnings = list(dict.fromkeys([*result.warnings, angle_warning]))
        minimum_absorption = float(
            getattr(self, "_minimum_total_absorption_m_inv", np.inf)
        )
        maximum_absorption = float(
            getattr(self, "_maximum_total_absorption_m_inv", -np.inf)
        )
        path_length = float(result.stats.get(
            "crystal_transport_step_length_m",
            self.grid.dz / gain_medium_path_cosine(self.gain_marker),
        ))
        nominal_maximum_gain_exponent = (
            0.0
            if not math.isfinite(minimum_absorption)
            else max(
                -minimum_absorption * path_length,
                0.0,
            )
        )
        nominal_maximum_absorption_optical_depth = (
            0.0
            if not math.isfinite(maximum_absorption)
            else max(maximum_absorption * path_length, 0.0)
        )
        result.stats["nominal_maximum_pump_gain_exponent"] = (
            nominal_maximum_gain_exponent
        )
        minimum_active_absorption = float(
            getattr(self, "_minimum_active_absorption_m_inv", np.inf)
        )
        result.stats["minimum_active_pump_absorption_m_inv"] = (
            None
            if not math.isfinite(minimum_active_absorption)
            else minimum_active_absorption
        )
        result.stats["nominal_maximum_pump_absorption_optical_depth"] = (
            nominal_maximum_absorption_optical_depth
        )
        resolved_absorption_optical_depth = (
            float(
                result.stats.get(
                    "maximum_pump_ray_absorption_optical_depth",
                    nominal_maximum_absorption_optical_depth,
                )
            )
            if mode == "stochastic_raytrace"
            else nominal_maximum_absorption_optical_depth
        )
        result.stats["maximum_pump_absorption_optical_depth"] = (
            resolved_absorption_optical_depth
        )
        result.stats["pump_absorption_optical_depth_source"] = (
            "actual_ray_segments"
            if mode == "stochastic_raytrace"
            else "nominal_gain_marker_path"
        )
        result.stats["positive_pump_exponent_clipping_active"] = bool(
            nominal_maximum_gain_exponent > 50.0
        )
        if nominal_maximum_gain_exponent > 50.0:
            result.warnings = list(
                dict.fromkeys(
                    [
                        *result.warnings,
                        "A negative Pump absorption coefficient produced a nominal single-cell "
                        "gain exponent above 50, so the exponential overflow guard is active. "
                        "Refine the transport grid and verify whether Pump stimulated emission "
                        "belongs in the selected level model.",
                    ]
                )
            )
        if resolved_absorption_optical_depth > 1.0:
            result.warnings = list(
                dict.fromkeys(
                    [
                        *result.warnings,
                        "The maximum Pump absorption optical depth alpha*ds exceeds 1 in a "
                        "single longitudinal cell. Beer-Lambert power loss remains exact for "
                        "the piecewise-constant coefficient, but population/temperature variation "
                        "inside that cell is unresolved; refine Nz for quantitative work.",
                    ]
                )
            )
        return result

    def _spectrally_resolved(
        self,
        population: np.ndarray,
        record_planes: bool,
        power_scale: float,
        delta_n: np.ndarray | None,
    ) -> PumpPropagationResult:
        """Propagate every Pump wavelength as an independent spatial field.

        The population is held fixed during this optical solve, exactly as it
        is for the legacy shared-field transport.  Population iterations then
        consume the returned local spectral intensity, so all wavelength bins
        share the same ion reservoir without assuming a spatially constant
        spectrum.
        """

        wavelengths = np.asarray(self.spectrum.coordinate, dtype=float)
        weights = np.asarray(self.spectrum.weights, dtype=float)
        spectral_intensity: list[np.ndarray] = []
        spectral_active: list[np.ndarray] = []
        spectral_passive: list[np.ndarray] = []
        child_results: list[PumpPropagationResult] = []
        trace = OpticalTrace()
        warnings: list[str] = []
        minimum_absorption = np.inf
        maximum_absorption = -np.inf
        minimum_active_absorption = np.inf

        for index, (wavelength, weight) in enumerate(
            zip(wavelengths, weights, strict=True)
        ):
            local_cfg = copy.deepcopy(self.cfg)
            local_cfg["wavelength_m"] = float(wavelength)
            absolute_zero_nominal = bool(
                float(self.cfg["power_W"]) == 0.0
                and str(self.cfg.get("temporal", {}).get("type", "cw"))
                == "file"
                and str(
                    self.cfg.get("temporal", {}).get(
                        "value_mode", "relative"
                    )
                )
                == "absolute_W"
            )
            local_cfg["power_W"] = (
                float(weight)
                if absolute_zero_nominal
                else float(self.cfg["power_W"]) * float(weight)
            )
            if absolute_zero_nominal:
                # The parent power scale is an absolute instantaneous watt
                # value; each spectral child turns it into its weighted share
                # through an ordinary dimensionless scale.
                local_cfg["temporal"]["value_mode"] = "relative"
            local_cfg["spectrum"] = {
                "type": "monochromatic",
                "bandwidth_fwhm_m": 0.0,
                "sample_count": 1,
                "span_fwhm": 1.0,
                "order": 1.0,
                "samples": [],
                "spatial_transport": "shared",
            }
            # ``waist_divergence`` and ``emitter_na`` define a wavelength-
            # independent native angular spread.  Their field-transport M2 is
            # derived from that angle and therefore scales as 1 / wavelength.
            # Recompute the derived solver fields for every spectral child;
            # retaining the centre-wavelength M2 would instead make the child
            # divergence scale spuriously with wavelength.
            canonicalize_pump_source(local_cfg)
            child = PumpPropagator(
                self.grid,
                self.material,
                local_cfg,
                make_beam(local_cfg),
                self.levels,
                self.physics,
                self.pre_elements,
                self.gain_marker,
            )
            child.set_endface_surface_sag(
                self._endface_surface_sag_m["front"],
                self._endface_surface_sag_m["back"],
            )
            local_result = child.propagate(
                population,
                record_planes,
                power_scale=power_scale,
                delta_n=delta_n,
            )
            child_results.append(local_result)
            spectral_intensity.append(local_result.intensity_W_m2)
            spectral_active.append(
                np.zeros_like(local_result.absorbed_W_m3)
                if local_result.active_absorbed_W_m3 is None
                else local_result.active_absorbed_W_m3
            )
            spectral_passive.append(
                np.zeros_like(local_result.absorbed_W_m3)
                if local_result.passive_absorbed_W_m3 is None
                else local_result.passive_absorbed_W_m3
            )
            if record_planes:
                trace.extend(local_result.trace, f"lambda_{index:03d}_")
            warnings.extend(local_result.warnings)
            minimum_absorption = min(
                minimum_absorption,
                float(
                    getattr(
                        child,
                        "_minimum_total_absorption_m_inv",
                        np.inf,
                    )
                ),
            )
            maximum_absorption = max(
                maximum_absorption,
                float(
                    getattr(
                        child,
                        "_maximum_total_absorption_m_inv",
                        -np.inf,
                    )
                ),
            )
            minimum_active_absorption = min(
                minimum_active_absorption,
                float(
                    getattr(
                        child,
                        "_minimum_active_absorption_m_inv",
                        np.inf,
                    )
                ),
            )

        spectral_intensity_array = np.stack(spectral_intensity, axis=0)
        spectral_active_array = np.stack(spectral_active, axis=0)
        spectral_passive_array = np.stack(spectral_passive, axis=0)
        active = np.sum(spectral_active_array, axis=0)
        passive = np.sum(spectral_passive_array, axis=0)
        absorbed = active + passive
        intensity = np.sum(spectral_intensity_array, axis=0)

        stats = copy.deepcopy(child_results[0].stats)
        observed_power_keys = {
            key
            for result in child_results
            for key, value in result.stats.items()
            if key.endswith("_W")
            and value is not None
            and not isinstance(value, bool)
            and np.isscalar(value)
            and np.isfinite(float(value))
        }
        undeclared_power_keys = sorted(
            key for key in observed_power_keys if pump_metric_spec(key) is None
        )
        if undeclared_power_keys:
            raise RuntimeError(
                "Pump spectral transport produced scalar power diagnostics "
                "without an explicit reducer contract: "
                f"{undeclared_power_keys}"
            )
        additive_keys = {
            key
            for key in observed_power_keys
            if pump_metric_spec(key) is not None
            and pump_metric_spec(key).spectral_reduction == "sum"
        }
        for key in additive_keys:
            stats[key] = float(
                sum(float(result.stats.get(key, 0.0) or 0.0) for result in child_results)
            )
        maximum_keys = {
            "crystal_transport_step_length_m",
            "maximum_pump_absorption_optical_depth",
            "nominal_maximum_pump_absorption_optical_depth",
            "maximum_pump_ray_absorption_optical_depth",
            "maximum_positive_pump_ray_gain_exponent",
            "maximum_used_substeps_per_z_cell",
            "maximum_external_ray_angle_deg",
            "maximum_launch_ray_angle_deg",
            "deposition_cells_below_30_samples_fraction",
            "median_poisson_relative_noise_proxy",
            "external_transmission_monte_carlo_relative_standard_error",
        }
        minimum_keys = {
            "minimum_crystal_capture_fraction",
            "gain_entrance_capture_fraction",
            "deposition_samples_minimum",
            "deposition_effective_samples_minimum",
            "entrance_window_power_fraction",
            "source_display_capture_fraction",
        }
        boolean_or_keys = {
            key
            for result in child_results
            for key, value in result.stats.items()
            if isinstance(value, (bool, np.bool_))
        }
        for key in maximum_keys:
            values = [
                float(result.stats[key])
                for result in child_results
                if result.stats.get(key) is not None
                and np.isscalar(result.stats[key])
                and np.isfinite(float(result.stats[key]))
            ]
            if values:
                stats[key] = max(values)
        for key in minimum_keys:
            values = [
                float(result.stats[key])
                for result in child_results
                if result.stats.get(key) is not None
                and np.isscalar(result.stats[key])
                and np.isfinite(float(result.stats[key]))
            ]
            if values:
                stats[key] = min(values)
        for key in boolean_or_keys:
            stats[key] = any(bool(result.stats.get(key, False)) for result in child_results)
        analytic_entrance = stats.get("gain_entrance_analytic_power_W")
        sampled_entrance = stats.get("gain_entrance_sampled_power_W")
        if analytic_entrance not in (None, 0.0) and sampled_entrance is not None:
            stats["gain_entrance_capture_fraction"] = float(sampled_entrance) / max(
                float(analytic_entrance), np.finfo(float).tiny
            )
        stats.update(
            {
                "spectral_spatial_transport": "per_wavelength",
                "spectral_spatial_sample_count": len(wavelengths),
                "spectral_wavelengths_m": wavelengths.tolist(),
                "spectral_source_power_weights": weights.tolist(),
                "spectral_grid_input_power_W": [
                    float(result.stats.get("grid_input_power_W", 0.0))
                    for result in child_results
                ],
                "spectral_output_power_W": [
                    float(result.stats.get("output_power_W", 0.0))
                    for result in child_results
                ],
                "spectral_absorbed_power_W": [
                    float(result.stats.get("absorbed_power_W", 0.0))
                    for result in child_results
                ],
                "spectral_transport_stats": [
                    copy.deepcopy(result.stats) for result in child_results
                ],
                "spectral_diagnostic_reducers": {
                    "schema": {
                        key: {
                            "spectral": spec.spectral_reduction,
                            "temporal": spec.temporal_reduction,
                        }
                        for key, spec in sorted(PUMP_METRIC_SPECS.items())
                    },
                    "worst_case_maximum": sorted(maximum_keys),
                    "worst_case_minimum": sorted(minimum_keys),
                    "boolean_status": "logical_or",
                    "gain_entrance_capture_fraction": (
                        "aggregate_sampled_power/aggregate_analytic_power"
                    ),
                },
            }
        )
        for residual_key, scale_key in (
            ("crystal_power_balance_residual_W", "grid_input_power_W"),
            ("source_to_exit_balance_residual_W", "grid_source_power_W"),
        ):
            if stats.get(residual_key) is not None:
                stats[residual_key.replace("_W", "_relative")] = abs(
                    float(stats[residual_key])
                ) / max(abs(float(stats.get(scale_key, 0.0))), np.finfo(float).tiny)
        stats["absorption_component_closure_relative"] = abs(
            float(self.grid.integrate_volume(active + passive - absorbed))
        ) / max(
            abs(float(self.grid.integrate_volume(absorbed))),
            np.finfo(float).tiny,
        )
        self._minimum_total_absorption_m_inv = minimum_absorption
        self._maximum_total_absorption_m_inv = maximum_absorption
        self._minimum_active_absorption_m_inv = minimum_active_absorption
        return PumpPropagationResult(
            intensity_W_m2=intensity,
            absorbed_W_m3=absorbed,
            stats=stats,
            trace=trace,
            warnings=list(dict.fromkeys(warnings)),
            active_absorbed_W_m3=active,
            passive_absorbed_W_m3=passive,
            wavelengths_m=wavelengths.copy(),
            spectral_intensity_W_m2=spectral_intensity_array,
            spectral_active_absorbed_W_m3=spectral_active_array,
            spectral_passive_absorbed_W_m3=spectral_passive_array,
        )

    def _path(self) -> tuple[list[int], str, str]:
        direction = resolve_pump_direction(
            str(self.cfg.get("direction", "forward")),
            str(self.gain_marker.get("direction", "auto")),
        )
        if direction == "forward":
            return list(range(len(self.grid.z))), direction, "front"
        if direction == "backward":
            return list(reversed(range(len(self.grid.z)))), direction, "back"
        raise ValueError("pump.direction must be forward or backward")

    def _absorption(self, population_slice: np.ndarray, z_index: int | None = None) -> np.ndarray:
        total, _, _ = self._absorption_components(
            population_slice, z_index
        )
        return total

    def _absorption_components(
        self,
        population_slice: np.ndarray,
        z_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.absorption_transport_model == "fixed_active_coefficient":
            if self.fixed_active_absorption_m_inv is None:
                raise RuntimeError(
                    "fixed Pump absorption transport requires an explicit "
                    "active coefficient"
                )
            active = (
                self.fixed_active_absorption_m_inv * self.crystal_mask
            )
        else:
            active = self.levels.pump_absorption_spectral_m_inv(
                population_slice,
                0.0,
                self.spectrum.coordinate,
                self.spectrum.weights,
                z_index,
            ) * self.crystal_mask
        passive = (
            float(self.material.get("pump_passive_absorption_m_inv", 0.0))
            * self.crystal_mask
        )
        total = active + passive
        self._minimum_total_absorption_m_inv = min(
            float(getattr(self, "_minimum_total_absorption_m_inv", np.inf)),
            float(np.min(total)),
        )
        self._maximum_total_absorption_m_inv = max(
            float(getattr(self, "_maximum_total_absorption_m_inv", -np.inf)),
            float(np.max(total)),
        )
        self._minimum_active_absorption_m_inv = min(
            float(
                getattr(
                    self,
                    "_minimum_active_absorption_m_inv",
                    np.inf,
                )
            ),
            float(np.min(active)),
        )
        return total, active, passive

    @staticmethod
    def _decompose_absorbed_power_density(
        local_intensity: np.ndarray,
        alpha_total: np.ndarray,
        alpha_active: np.ndarray,
        alpha_passive: np.ndarray,
        path_length_m: float,
        cell_depth_m: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cell_depth = path_length_m if cell_depth_m is None else float(cell_depth_m)
        # Large negative Beer-Lambert exponents safely underflow towards zero
        # and must not be clipped: doing so leaves a false exp(-50) leakage in
        # strongly absorbing cells.  Only Pump gain needs an overflow guard.
        optical_depth = alpha_total * path_length_m
        exponent = np.minimum(-optical_depth, 50.0)
        transmission = np.exp(exponent)
        # ``-expm1(-alpha*ds)`` retains all significant digits at
        # transparency.  Forming the component sources through the exact
        # longitudinal intensity integral also avoids alpha_component/alpha
        # amplification when active gain nearly cancels passive absorption.
        removed_fraction = -np.expm1(exponent)
        phi = np.ones_like(optical_depth, dtype=float)
        finite_depth = np.abs(optical_depth) > 1e-8
        np.divide(
            removed_fraction,
            optical_depth,
            out=phi,
            where=finite_depth,
        )
        if np.any(~finite_depth):
            small = optical_depth[~finite_depth]
            phi[~finite_depth] = (
                1.0 - 0.5 * small + small * small / 6.0
            )
        path_scale = float(path_length_m) / cell_depth
        mean_path_intensity = local_intensity * phi * path_scale
        active = mean_path_intensity * alpha_active
        passive = mean_path_intensity * alpha_passive
        total = active + passive
        return transmission, total, active, passive

    @staticmethod
    def _cell_average_intensity(
        entrance_intensity: np.ndarray,
        alpha_total: np.ndarray,
        absorbed_power_density: np.ndarray,
        path_length_m: float | None = None,
        cell_depth_m: float | None = None,
    ) -> np.ndarray:
        """Return the Beer-Lambert cell-average Pump intensity.

        Population rate equations live at finite-volume cell centres, so they
        must consume the longitudinal average intensity rather than the value
        just before the cell attenuation operator.  Since
        ``absorbed_power_density = alpha * <I>``, this form is exact for the
        piecewise-constant absorption coefficient used by a propagation step
        and has the correct zero-absorption limit.
        """

        if path_length_m is None or cell_depth_m is None:
            finite = np.abs(alpha_total) > 1e-14
            return np.where(
                finite,
                np.divide(
                    absorbed_power_density,
                    alpha_total,
                    out=np.zeros_like(alpha_total),
                    where=finite,
                ),
                entrance_intensity,
            )
        optical_depth = np.asarray(alpha_total, dtype=float) * float(path_length_m)
        exponent = np.minimum(-optical_depth, 50.0)
        phi = np.ones_like(optical_depth)
        finite_depth = np.abs(optical_depth) > 1e-8
        np.divide(
            -np.expm1(exponent),
            optical_depth,
            out=phi,
            where=finite_depth,
        )
        if np.any(~finite_depth):
            small = optical_depth[~finite_depth]
            phi[~finite_depth] = 1.0 - 0.5 * small + small * small / 6.0
        return (
            np.asarray(entrance_intensity, dtype=float)
            * phi
            * float(path_length_m)
            / float(cell_depth_m)
        )

    def _prescribed(
        self,
        population: np.ndarray,
        record_planes: bool,
        power_scale: float,
        delta_n: np.ndarray | None,
    ) -> PumpPropagationResult:
        indices, direction, side = self._path()
        intensity = np.zeros(self.grid.shape, dtype=float)
        absorbed = np.zeros_like(intensity)
        active_absorbed = np.zeros_like(intensity)
        passive_absorbed = np.zeros_like(intensity)
        transmission_map = np.ones((len(self.grid.y), len(self.grid.x)), dtype=float)
        x, y = self.grid.transverse_mesh
        power = self._source_power(power_scale)
        n_out = float(self.gain_marker.get("outside_refractive_index", self.cfg.get("source_refractive_index", 1.0)))
        n_gain = material_refractive_index(self.material, float(self.cfg["wavelength_m"]), role="pump")
        model = str(self.gain_marker.get(f"{side}_model", self.gain_marker.get("model", self.physics["interface_model"])))
        specified = gain_surface_power_transmission(
            self.gain_marker,
            side,
            float(self.cfg["wavelength_m"]),
        )
        internal_x, internal_y = gain_medium_axis_angles_rad(self.gain_marker)
        angle_x, angle_y = refract_axis_angles_rad(
            internal_x,
            internal_y,
            n_before=n_gain,
            n_after=n_out,
        )
        incidence_angle_deg = math.degrees(
            math.atan(math.hypot(math.tan(angle_x), math.tan(angle_y)))
        )
        entrance_transmission = interface_power_transmission(
            n_out,
            n_gain,
            model,
            specified,
            incidence_angle_deg,
            str(self.gain_marker.get("polarization", "unpolarized")),
        )
        # The prescribed caustic is evaluated inside the crystal.  Its vacuum
        # wavelength therefore acquires the material-index Rayleigh length,
        # independently of the refractive index at the configured source plane.
        in_crystal_cfg = dict(self.cfg)
        in_crystal_cfg["propagation_refractive_index"] = n_gain
        in_crystal_beam = make_beam(in_crystal_cfg)
        entrance_ideal = power * entrance_transmission * in_crystal_beam.density(x, y, 0.0)
        entrance_field = np.sqrt(np.maximum(entrance_ideal, 0.0)).astype(complex)
        if self.clip_crystal:
            entrance_field *= self.crystal_mask
        input_power = float(self.grid.integrate_transverse(np.abs(entrance_field) ** 2))
        path_cosine = gain_medium_path_cosine(self.gain_marker)
        step_length = self.grid.dz / path_cosine
        for path_index, iz in enumerate(indices):
            distance = (path_index + 0.5) * step_length
            axial_distance = (path_index + 0.5) * self.grid.dz
            ideal = power * entrance_transmission * in_crystal_beam.density(
                x - axial_distance * math.tan(internal_x),
                y - axial_distance * math.tan(internal_y),
                distance,
            )
            local = ideal * transmission_map
            if self.clip_crystal:
                local *= self.crystal_mask
            alpha, alpha_active, alpha_passive = self._absorption_components(
                population[:, iz], iz
            )
            local_transmission, total_source, active_source, passive_source = (
                self._decompose_absorbed_power_density(
                    local,
                    alpha,
                    alpha_active,
                    alpha_passive,
                    step_length,
                    self.grid.dz,
                )
            )
            intensity[iz] = self._cell_average_intensity(
                local, alpha, total_source, step_length, self.grid.dz
            )
            absorbed[iz] = total_source
            active_absorbed[iz] = active_source
            passive_absorbed[iz] = passive_source
            if self.physics["pump_depletion"]:
                transmission_map *= local_transmission
        exit_ideal = power * entrance_transmission * in_crystal_beam.density(
            x - len(self.grid.z) * self.grid.dz * math.tan(internal_x),
            y - len(self.grid.z) * self.grid.dz * math.tan(internal_y),
            len(self.grid.z) * step_length,
        )
        exit_field = np.sqrt(np.maximum(exit_ideal * transmission_map, 0.0)).astype(complex)
        if self.clip_crystal:
            exit_field *= self.crystal_mask
        output_power = float(self.grid.integrate_transverse(np.abs(exit_field) ** 2))
        source = source_field(self.cfg, power, self.grid)
        source_display_power = float(self.grid.integrate_transverse(np.abs(source) ** 2))
        warnings: list[str] = []
        if delta_n is not None and np.any(delta_n):
            warnings.append(
                "prescribed_caustic does not propagate the Pump through a 3D thermal phase; "
                "use Pump FFT or hybrid_abcd_fft for thermal-optical feedback."
            )
        if power > 0.0 and source_display_power < 0.98 * power:
            warnings.append(
                "Prescribed Pump caustic uses an analytic infinite-plane source: the source-plane display window "
                f"contains {source_display_power / power:.3%}, but no power was clipped unless an "
                "aperture is configured."
            )
        trace = OpticalTrace()
        if record_planes:
            trace.add("source", source, self.grid, "power_W")
            trace.add("gain_entrance", entrance_field, self.grid, "power_W", complex_field_available=False)
            trace.add("gain_exit", exit_field, self.grid, "power_W", complex_field_available=False)
        if self.pre_elements:
            warnings.append(
                "pump_propagation=prescribed_caustic ignores pump source-to-gain "
                "optical elements; use abcd, fft, hybrid_abcd_fft, or "
                "stochastic_raytrace."
            )
        stats = self._stats(
            direction,
            power,
            power,
            input_power,
            output_power,
            absorbed,
            active_absorbed,
            passive_absorbed,
        )
        if self.physics["pump_depletion"]:
            residual = input_power - output_power - float(stats["absorbed_power_W"])
            relative = abs(residual) / max(input_power, np.finfo(float).tiny)
            stats.update(
                {
                    "crystal_power_balance_residual_W": residual,
                    "crystal_power_balance_residual_relative": relative,
                    "prescribed_caustic_unresolved_redistribution_power_W": residual,
                    "depletion_closure": "coordinate_fixed_transmission_on_prescribed_caustic",
                }
            )
            if relative > 5e-6:
                warnings.append(
                    "The prescribed Pump caustic changes shape while its depletion map remains "
                    "coordinate-fixed, so the integrated power ledger does not close. Use Pump FFT, "
                    "stochastic ray tracing, or the power-conserving Super-Gaussian ABCD closure "
                    "when this residual is material."
                )
        else:
            stats.update(
                {
                    "crystal_power_balance_residual_W": None,
                    "crystal_power_balance_residual_relative": None,
                    "prescribed_caustic_unresolved_redistribution_power_W": None,
                }
            )
        return PumpPropagationResult(
            intensity,
            absorbed,
            stats,
            trace,
            warnings,
            active_absorbed,
            passive_absorbed,
        )

    def _abcd(
        self,
        population: np.ndarray,
        record_planes: bool,
        power_scale: float,
        delta_n: np.ndarray | None,
        *,
        preserve_super_gaussian: bool = False,
    ) -> PumpPropagationResult:
        indices, direction, side = self._path()
        intensity = np.zeros(self.grid.shape, dtype=float)
        absorbed = np.zeros_like(intensity)
        active_absorbed = np.zeros_like(intensity)
        passive_absorbed = np.zeros_like(intensity)
        power = self._source_power(power_scale)
        n_gain = material_refractive_index(self.material, float(self.cfg["wavelength_m"]), role="pump")
        interface = gain_interface_element(self.gain_marker, n_gain, side, self.physics["interface_model"])
        warnings: list[str] = []
        mode_name = "Super-Gaussian ABCD" if preserve_super_gaussian else "ABCD"
        if delta_n is not None and np.any(delta_n):
            warnings.append(
                f"Pump {mode_name} does not propagate arbitrary 3D thermal aberrations; use Pump FFT "
                "or hybrid_abcd_fft for thermal-optical feedback."
            )
        plane_propagator = (
            super_gaussian_abcd_to_plane
            if preserve_super_gaussian
            else gaussian_abcd_to_plane
        )
        entrance_diagnostics: dict[str, Any] = {}
        entrance_field, _, _, entrance_warnings = plane_propagator(
            self.cfg,
            power,
            [*self.pre_elements, interface],
            self.grid,
            "power_W",
            self.physics["interface_model"],
            record_planes,
            diagnostics=entrance_diagnostics,
        )
        warnings.extend(entrance_warnings)
        if self.clip_crystal:
            entrance_field *= self.crystal_mask
        input_power = float(self.grid.integrate_transverse(np.abs(entrance_field) ** 2))
        envelope_power = input_power
        envelope_side_loss = 0.0
        entrance_analytic_power = max(
            float(entrance_diagnostics.get("analytic_quantity", power)),
            np.finfo(float).tiny,
        )
        envelope_minimum_capture = min(
            max(input_power / entrance_analytic_power, 0.0),
            1.0,
        )
        path_cosine = gain_medium_path_cosine(self.gain_marker)
        step_length = self.grid.dz / path_cosine
        for path_index, iz in enumerate(indices):
            distance = (path_index + 0.5) * step_length
            elements = [*self.pre_elements, interface, {"type": "free_space", "length_m": distance}]
            local_diagnostics: dict[str, Any] = {}
            ideal_field, _, _, local_warnings = plane_propagator(
                self.cfg,
                power,
                elements,
                self.grid,
                "power_W",
                self.physics["interface_model"],
                False,
                diagnostics=local_diagnostics,
            )
            if not warnings and local_warnings:
                warnings.extend(local_warnings)
            ideal = np.abs(ideal_field) ** 2
            represented = ideal * self.crystal_mask if self.clip_crystal else ideal
            represented_power = float(
                self.grid.integrate_transverse(represented)
            )
            analytic_power = max(
                float(local_diagnostics.get("analytic_quantity", power)),
                np.finfo(float).tiny,
            )
            capture = min(max(represented_power / analytic_power, 0.0), 1.0)
            next_minimum_capture = min(envelope_minimum_capture, capture)
            if envelope_minimum_capture > np.finfo(float).tiny:
                survival = next_minimum_capture / envelope_minimum_capture
                escaped = envelope_power * max(1.0 - survival, 0.0)
                envelope_side_loss += escaped
                envelope_power *= survival
            envelope_minimum_capture = next_minimum_capture
            if represented_power > np.finfo(float).tiny:
                # Fixed-envelope closure: local depletion changes the scalar
                # transported power, then the next z plane is projected back
                # onto the normalized analytic ABCD envelope.  Keeping a
                # coordinate-fixed transmission map while w(z), the centroid,
                # or the ellipse orientation changes violates the prescribed
                # envelope ansatz and generally breaks the power ledger.
                local = max(envelope_power, 0.0) * represented / represented_power
            else:
                local = np.zeros_like(ideal)
            alpha, alpha_active, alpha_passive = self._absorption_components(
                population[:, iz], iz
            )
            _, total_source, active_source, passive_source = (
                self._decompose_absorbed_power_density(
                    local,
                    alpha,
                    alpha_active,
                    alpha_passive,
                    step_length,
                    self.grid.dz,
                )
            )
            intensity[iz] = self._cell_average_intensity(
                local, alpha, total_source, step_length, self.grid.dz
            )
            absorbed[iz] = total_source
            active_absorbed[iz] = active_source
            passive_absorbed[iz] = passive_source
            if self.physics["pump_depletion"]:
                absorbed_slice = float(
                    self.grid.integrate_transverse(total_source)
                    * self.grid.dz
                )
                envelope_power -= absorbed_slice
        exit_diagnostics: dict[str, Any] = {}
        exit_ideal_field, _, _, exit_warnings = plane_propagator(
            self.cfg,
            power,
            [
                *self.pre_elements,
                interface,
                {"type": "free_space", "length_m": len(self.grid.z) * step_length},
            ],
            self.grid,
            "power_W",
            self.physics["interface_model"],
            record_planes,
            diagnostics=exit_diagnostics,
        )
        if not warnings and exit_warnings:
            warnings.extend(exit_warnings)
        exit_ideal = np.abs(exit_ideal_field) ** 2
        exit_represented = (
            exit_ideal * self.crystal_mask if self.clip_crystal else exit_ideal
        )
        exit_represented_power = float(
            self.grid.integrate_transverse(exit_represented)
        )
        exit_analytic_power = max(
            float(exit_diagnostics.get("analytic_quantity", power)),
            np.finfo(float).tiny,
        )
        exit_capture = min(
            max(exit_represented_power / exit_analytic_power, 0.0),
            1.0,
        )
        final_minimum_capture = min(envelope_minimum_capture, exit_capture)
        if envelope_minimum_capture > np.finfo(float).tiny:
            survival = final_minimum_capture / envelope_minimum_capture
            escaped = envelope_power * max(1.0 - survival, 0.0)
            envelope_side_loss += escaped
            envelope_power *= survival
        envelope_minimum_capture = final_minimum_capture
        exit_field = np.zeros_like(exit_ideal_field)
        if exit_represented_power > np.finfo(float).tiny:
            phase = np.ones_like(exit_ideal_field, dtype=complex)
            carrier = np.abs(exit_ideal_field)
            nonzero = carrier > np.finfo(float).tiny
            phase[nonzero] = exit_ideal_field[nonzero] / carrier[nonzero]
            exit_field = (
                np.sqrt(
                    max(envelope_power, 0.0)
                    * exit_represented
                    / exit_represented_power
                ).astype(complex)
                * phase
            )
        output_power = float(self.grid.integrate_transverse(np.abs(exit_field) ** 2))
        source = source_field(self.cfg, power, self.grid)
        trace = OpticalTrace()
        if record_planes:
            trace.add("source", source, self.grid, "power_W")
            trace.add("gain_entrance", entrance_field, self.grid, "power_W")
            trace.add("gain_exit", exit_field, self.grid, "power_W")
        stats = self._stats(
            direction,
            power,
            # The analytic ABCD source is normalized on the infinite plane.
            # The sampled source field is only a display window and must not
            # be reported as transported input-power clipping.
            power,
            input_power,
            output_power,
            absorbed,
            active_absorbed,
            passive_absorbed,
        )
        envelope_label = "super_gaussian" if preserve_super_gaussian else "gaussian"
        stats.update(
            {
                "transport_representation": entrance_diagnostics.get(
                    "transport_representation",
                    "self_similar_gaussian_intensity_envelope",
                ),
                "envelope_propagation": entrance_diagnostics.get(
                    "envelope_propagation", "astigmatic_first_order_abcd_q"
                ),
                "profile_evolution": entrance_diagnostics.get(
                    "profile_evolution", "gaussian_q_parameter"
                ),
                "coherence_contract": entrance_diagnostics.get(
                    "coherence_contract", "analytic_complex_gaussian_field"
                ),
                "exact_paraxial_wave_solution": entrance_diagnostics.get(
                    "exact_paraxial_wave_solution", True
                ),
                "thermal_refraction_feedback": False,
                "depletion_closure": (
                    f"slice_power_projection_preserving_{envelope_label}_envelope"
                ),
                "crystal_escape_model": "monotone_minimum_transverse_capture_fraction",
                "gain_entrance_capture_fraction": min(
                    max(input_power / entrance_analytic_power, 0.0), 1.0
                ),
                "minimum_crystal_capture_fraction": envelope_minimum_capture,
                "crystal_side_aperture_loss_W": envelope_side_loss,
                "gain_entrance_radius_x_m": entrance_diagnostics.get("radius_x_m"),
                "gain_entrance_radius_y_m": entrance_diagnostics.get("radius_y_m"),
                "gain_exit_radius_x_m": exit_diagnostics.get("radius_x_m"),
                "gain_exit_radius_y_m": exit_diagnostics.get("radius_y_m"),
            }
        )
        if preserve_super_gaussian:
            stats.update(
                {
                    "super_gaussian_exponent_x": entrance_diagnostics.get(
                        "super_gaussian_exponent_x"
                    ),
                    "super_gaussian_exponent_y": entrance_diagnostics.get(
                        "super_gaussian_exponent_y"
                    ),
                    "radial_equal_axis_convention": entrance_diagnostics.get(
                        "radial_equal_axis_convention"
                    ),
                }
            )
        if self.physics["pump_depletion"]:
            accounted_power = (
                output_power
                + float(stats["absorbed_power_W"])
                + envelope_side_loss
            )
            balance_residual = input_power - accounted_power
            stats["crystal_transport_accounted_power_W"] = accounted_power
            stats["crystal_power_balance_residual_W"] = balance_residual
            stats["crystal_power_balance_residual_relative"] = abs(
                balance_residual
            ) / max(input_power, np.finfo(float).tiny)
            if stats["crystal_power_balance_residual_relative"] > 5e-6:
                warnings.append(
                    f"Pump {mode_name} crystal power balance did not close after "
                    "accounting for absorption and monotone side escape; refine "
                    "the longitudinal/transverse grid or use FFT/ray transport."
                )
        else:
            stats["crystal_transport_accounted_power_W"] = None
            stats["crystal_power_balance_residual_W"] = None
            stats["crystal_power_balance_residual_relative"] = None
        if envelope_minimum_capture < 0.98:
            warnings.append(
                f"{mode_name} envelope capture fell below 98% inside the "
                "crystal/window. The lost monotone envelope fraction is "
                "reported as crystal_side_aperture_loss_W; use stochastic "
                "ray tracing or FFT for trajectory-resolved side escape."
            )
        return PumpPropagationResult(
            intensity,
            absorbed,
            stats,
            trace,
            warnings,
            active_absorbed,
            passive_absorbed,
        )

    def _fft(
        self,
        population: np.ndarray,
        record_planes: bool,
        power_scale: float,
        delta_n: np.ndarray | None,
    ) -> PumpPropagationResult:
        indices, direction, side = self._path()
        power = self._source_power(power_scale)
        external = self._fft_external_field(power, record_planes)
        n_gain = material_refractive_index(self.material, float(self.cfg["wavelength_m"]), role="pump")
        field = gain_interface(
            external.field,
            external.refractive_index,
            n_gain,
            self.gain_marker,
            side,
            self.physics["interface_model"],
            float(self.cfg["wavelength_m"]),
            entering=True,
            surface_sag_m=self._endface_surface_sag_m[side],
        )
        return self._propagate_fft_inside_crystal(
            population,
            field,
            n_gain,
            indices,
            direction,
            power,
            external.source_grid_quantity,
            external.trace,
            external.warnings,
            record_planes,
            delta_n,
            {
                "external_propagation": "fft",
                "crystal_propagation": "fft",
                "source_plane_sampling_affects_transport": True,
            },
        )

    def _hybrid_abcd_fft(
        self,
        population: np.ndarray,
        record_planes: bool,
        power_scale: float,
        delta_n: np.ndarray | None,
    ) -> PumpPropagationResult:
        """Propagate analytically to the gain face, then use FFT in crystal.

        A broad source plane is never sampled for transport.  The external
        ideal-Gaussian path is reduced to its full complex boundary field just
        inside the entrance interface; diffraction, depletion and arbitrary
        three-dimensional thermal phase are then applied on the gain grid.
        """

        indices, direction, side = self._path()
        power = self._source_power(power_scale)
        external, diagnostics = self._hybrid_external_field(
            power,
            record_planes,
            side,
        )
        n_gain = material_refractive_index(
            self.material,
            float(self.cfg["wavelength_m"]),
            role="pump",
        )
        warnings = list(external.warnings)
        if not math.isclose(
            float(external.refractive_index),
            n_gain,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            warnings.append(
                "Hybrid Pump ABCD boundary did not end in the configured gain index; "
                "the crystal FFT used the material Pump refractive index."
            )
        sampled_fraction = float(diagnostics.get("sampled_fraction", 1.0))
        if power > 0.0 and sampled_fraction < 0.98:
            warnings.append(
                "Hybrid Pump gain-entrance FFT window contains only "
                f"{sampled_fraction:.3%} of the analytically transported field; "
                "enlarge the gain optical window."
            )
        sampled_before_crystal = float(
            self.grid.integrate_transverse(np.abs(external.field) ** 2)
        )
        entrance_stats = {
            "external_propagation": "abcd",
            "crystal_propagation": "fft",
            "source_plane_sampling_affects_transport": False,
            "gain_entrance_analytic_power_W": float(
                diagnostics.get("analytic_quantity", sampled_before_crystal)
            ),
            "gain_entrance_sampled_power_W": sampled_before_crystal,
            "gain_entrance_capture_fraction": sampled_fraction,
            "gain_entrance_radius_x_m": diagnostics.get("radius_x_m"),
            "gain_entrance_radius_y_m": diagnostics.get("radius_y_m"),
            "gain_entrance_curvature_x_m": diagnostics.get("curvature_x_m"),
            "gain_entrance_curvature_y_m": diagnostics.get("curvature_y_m"),
            "gain_entrance_center_x_m": diagnostics.get("center_x_m"),
            "gain_entrance_center_y_m": diagnostics.get("center_y_m"),
            "gain_entrance_angle_x_rad": diagnostics.get("angle_x_rad"),
            "gain_entrance_angle_y_rad": diagnostics.get("angle_y_rad"),
            "gain_entrance_tilt_phase_included": diagnostics.get(
                "tilt_phase_included", False
            ),
        }
        outside_index = float(
            self.gain_marker.get(
                "outside_refractive_index",
                self.cfg.get("source_refractive_index", 1.0),
            )
        )
        entrance_field = apply_endface_sag_phase(
            external.field,
            n_crystal=n_gain,
            n_outside=outside_index,
            side=side,
            wavelength_m=float(self.cfg["wavelength_m"]),
            surface_sag_m=self._endface_surface_sag_m[side],
        )
        return self._propagate_fft_inside_crystal(
            population,
            entrance_field,
            n_gain,
            indices,
            direction,
            power,
            external.source_grid_quantity,
            external.trace,
            warnings,
            record_planes,
            delta_n,
            entrance_stats,
            transport_axis_angles_rad=(
                float(diagnostics["angle_x_rad"]),
                float(diagnostics["angle_y_rad"]),
            ),
        )

    def _stochastic_raytrace(
        self,
        population: np.ndarray,
        record_planes: bool,
        power_scale: float,
        delta_n: np.ndarray | None,
    ) -> PumpPropagationResult:
        """Trace a reproducible incoherent ray bundle and deposit its tracks.

        The external geometry is linear and independent of population, so its
        unit-power ray bundle is cached.  Pump-saturation/fixed-point updates
        reuse the same rays; otherwise Monte-Carlo resampling noise would look
        like a physical change in absorption and obstruct convergence.
        """

        indices, direction, side = self._path()
        power = self._source_power(power_scale)
        n_gain = material_refractive_index(
            self.material,
            float(self.cfg["wavelength_m"]),
            role="pump",
        )
        if self._raytrace_external_cache is None or (
            record_planes
            and not self._raytrace_external_cache.plane_intensities_W_m2
        ):
            interface = gain_interface_element(
                self.gain_marker,
                n_gain,
                side,
                self.physics["interface_model"],
            )
            self._raytrace_external_cache = build_ray_entrance(
                self.cfg,
                [*self.pre_elements, interface],
                self.grid,
                self.physics["interface_model"],
                record_planes=record_planes,
            )
        entrance = self._raytrace_external_cache
        volume = propagate_ray_volume(
            entrance,
            power,
            self.grid,
            indices,
            self.crystal_mask,
            n_gain,
            self._absorption_components,
            population,
            bool(self.physics["pump_depletion"]),
            self.cfg.get("raytrace", {}),
            delta_n,
        )
        stats = self._stats(
            direction,
            power,
            power,
            float(volume.stats["grid_input_power_W"]),
            float(volume.stats["output_power_W"]),
            volume.absorbed_W_m3,
            volume.active_absorbed_W_m3,
            volume.passive_absorbed_W_m3,
        )
        stats.update(entrance.stats)
        stats.update(volume.stats)
        entrance_before_aperture = float(
            volume.stats["gain_entrance_power_before_crystal_aperture_W"]
        )
        input_power = float(volume.stats["grid_input_power_W"])
        stats.update(
            {
                "external_train_output_power_W": entrance_before_aperture,
                "external_train_loss_W": max(power - entrance_before_aperture, 0.0),
                "computed_source_to_crystal_efficiency": input_power
                / max(power, np.finfo(float).tiny),
                "configured_geometric_efficiency_is_additional": True,
                "raytrace_contract": "sequential_paraxial_geometrical_optics_with_track_length_deposition",
                "raytrace_is_nonsequential_cad": False,
                "source_definition_transport_representation": "stochastic_phase_space_rays",
            }
        )
        if bool(self.physics["pump_depletion"]):
            overall_accounted = (
                float(stats["external_train_loss_W"])
                + float(volume.stats["gain_entrance_crystal_aperture_loss_W"])
                + float(volume.stats["crystal_transport_accounted_power_W"])
            )
            stats["source_to_exit_accounted_power_W"] = overall_accounted
            stats["source_to_exit_balance_residual_W"] = power - overall_accounted
            stats["source_to_exit_balance_residual_relative"] = abs(
                power - overall_accounted
            ) / max(power, np.finfo(float).tiny)
        else:
            stats["source_to_exit_accounted_power_W"] = None
            stats["source_to_exit_balance_residual_W"] = None
            stats["source_to_exit_balance_residual_relative"] = None

        trace = OpticalTrace()
        if record_planes:
            source_intensity = entrance.source_intensity_W_m2 * power
            entrance_intensity = (
                entrance.entrance_intensity_W_m2 * power * self.crystal_mask
            )
            trace.add(
                "source",
                np.sqrt(np.maximum(source_intensity, 0.0)).astype(complex),
                self.grid,
                "power_W",
                complex_field_available=False,
            )
            for plane_name, plane_intensity in entrance.plane_intensities_W_m2:
                trace.add(
                    plane_name,
                    np.sqrt(np.maximum(plane_intensity * power, 0.0)).astype(complex),
                    self.grid,
                    "power_W",
                    complex_field_available=False,
                )
            trace.add(
                "gain_entrance",
                np.sqrt(np.maximum(entrance_intensity, 0.0)).astype(complex),
                self.grid,
                "power_W",
                complex_field_available=False,
            )
            trace.add(
                "gain_exit",
                np.sqrt(np.maximum(volume.exit_intensity_W_m2, 0.0)).astype(complex),
                self.grid,
                "power_W",
                complex_field_available=False,
            )
        return PumpPropagationResult(
            volume.intensity_W_m2,
            volume.absorbed_W_m3,
            stats,
            trace,
            volume.warnings,
            volume.active_absorbed_W_m3,
            volume.passive_absorbed_W_m3,
        )

    def _propagate_fft_inside_crystal(
        self,
        population: np.ndarray,
        entrance_field: np.ndarray,
        n_gain: float,
        indices: list[int],
        direction: str,
        configured_power: float,
        source_grid_power: float,
        trace: OpticalTrace,
        warnings: list[str],
        record_planes: bool,
        delta_n: np.ndarray | None,
        extra_stats: dict[str, Any] | None = None,
        *,
        transport_axis_angles_rad: tuple[float, float] | None = None,
    ) -> PumpPropagationResult:
        """Shared sampled-field propagation from a gain boundary onward."""

        field = np.asarray(entrance_field, dtype=complex).copy()
        sampled_before_crystal = float(
            self.grid.integrate_transverse(np.abs(field) ** 2)
        )
        if self.clip_crystal:
            field *= self.crystal_mask
        input_power = float(self.grid.integrate_transverse(np.abs(field) ** 2))
        entrance_aperture_loss = max(sampled_before_crystal - input_power, 0.0)
        side_aperture_loss = 0.0
        if record_planes:
            trace.add("gain_entrance", field, self.grid, "power_W")
        intensity = np.zeros(self.grid.shape, dtype=float)
        absorbed = np.zeros_like(intensity)
        active_absorbed = np.zeros_like(intensity)
        passive_absorbed = np.zeros_like(intensity)
        wavelength = float(self.cfg["wavelength_m"])
        m2x = float(self.cfg.get("m2_x", 1.0))
        m2y = float(self.cfg.get("m2_y", 1.0))
        thermal_delta_n = (
            np.zeros(self.grid.shape, dtype=float)
            if delta_n is None
            else np.asarray(delta_n, dtype=float)
        )
        if thermal_delta_n.shape != self.grid.shape:
            raise ValueError("Pump thermal delta_n must match the Pump grid")
        k0 = 2.0 * math.pi / wavelength
        angle_x, angle_y = gain_medium_axis_angles_rad(self.gain_marker)
        path_cosine = gain_medium_path_cosine(self.gain_marker)
        if transport_axis_angles_rad is not None:
            # The hybrid handoff omits the carrier tilt from its sampled field.
            # Its ABCD chief ray already includes source/element tilts and the
            # gain-marker axis break, so transport that complete angle once.
            angle_x, angle_y = transport_axis_angles_rad
            path_cosine = 1.0 / math.sqrt(
                1.0 + math.tan(angle_x)**2 + math.tan(angle_y)**2
            )
        step_length = self.grid.dz / path_cosine
        for iz in indices:
            field = angular_spectrum(field, self.grid, step_length / 2.0, wavelength, n_gain, m2x, m2y)
            if angle_x != 0.0 or angle_y != 0.0:
                field = shift_field(
                    field,
                    self.grid,
                    0.5 * self.grid.dz * math.tan(angle_x),
                    0.5 * self.grid.dz * math.tan(angle_y),
                )
            if self.clip_crystal:
                power_before_aperture = float(
                    self.grid.integrate_transverse(np.abs(field) ** 2)
                )
                field *= self.crystal_mask
                power_after_aperture = float(
                    self.grid.integrate_transverse(np.abs(field) ** 2)
                )
                side_aperture_loss += max(
                    power_before_aperture - power_after_aperture,
                    0.0,
                )
            field *= np.exp(1j * k0 * thermal_delta_n[iz] * step_length)
            local = np.abs(field) ** 2
            alpha, alpha_active, alpha_passive = self._absorption_components(
                population[:, iz], iz
            )
            local_transmission, total_source, active_source, passive_source = (
                self._decompose_absorbed_power_density(
                    local,
                    alpha,
                    alpha_active,
                    alpha_passive,
                    step_length,
                    self.grid.dz,
                )
            )
            intensity[iz] = self._cell_average_intensity(
                local, alpha, total_source, step_length, self.grid.dz
            )
            absorbed[iz] = total_source
            active_absorbed[iz] = active_source
            passive_absorbed[iz] = passive_source
            if self.physics["pump_depletion"]:
                field *= np.sqrt(local_transmission)
            field = angular_spectrum(field, self.grid, step_length / 2.0, wavelength, n_gain, m2x, m2y)
            if angle_x != 0.0 or angle_y != 0.0:
                field = shift_field(
                    field,
                    self.grid,
                    0.5 * self.grid.dz * math.tan(angle_x),
                    0.5 * self.grid.dz * math.tan(angle_y),
                )
            if self.clip_crystal:
                power_before_aperture = float(
                    self.grid.integrate_transverse(np.abs(field) ** 2)
                )
                field *= self.crystal_mask
                power_after_aperture = float(
                    self.grid.integrate_transverse(np.abs(field) ** 2)
                )
                side_aperture_loss += max(
                    power_before_aperture - power_after_aperture,
                    0.0,
                )
        output_power = float(self.grid.integrate_transverse(np.abs(field) ** 2))
        if record_planes:
            trace.add("gain_exit", field, self.grid, "power_W")
        stats = self._stats(
            direction,
            configured_power,
            source_grid_power,
            input_power,
            output_power,
            absorbed,
            active_absorbed,
            passive_absorbed,
        )
        if extra_stats:
            stats.update(extra_stats)
        stats["crystal_transport_step_length_m"] = step_length
        stats["gain_entrance_crystal_aperture_transmission"] = input_power / max(
            sampled_before_crystal,
            np.finfo(float).tiny,
        )
        stats["gain_entrance_crystal_aperture_loss_W"] = entrance_aperture_loss
        stats["crystal_side_aperture_loss_W"] = side_aperture_loss
        if bool(self.physics["pump_depletion"]):
            accounted = (
                float(stats["absorbed_power_W"])
                + output_power
                + side_aperture_loss
            )
            balance_residual = input_power - accounted
            stats["crystal_transport_accounted_power_W"] = accounted
            stats["crystal_power_balance_residual_W"] = balance_residual
            stats["crystal_power_balance_residual_relative"] = abs(
                balance_residual
            ) / max(input_power, np.finfo(float).tiny)
            if stats["crystal_power_balance_residual_relative"] > 5e-6:
                warnings.append(
                    "Pump FFT crystal power balance did not close after accounting for "
                    "active/passive absorption and side-aperture leakage; increase the "
                    "transverse/longitudinal sampling and inspect the optical window."
                )
        else:
            stats["crystal_transport_accounted_power_W"] = None
            stats["crystal_power_balance_residual_W"] = None
            stats["crystal_power_balance_residual_relative"] = None
        return PumpPropagationResult(
            intensity,
            absorbed,
            stats,
            trace,
            list(warnings),
            active_absorbed,
            passive_absorbed,
        )

    def _hybrid_external_field(
        self,
        power: float,
        record_planes: bool,
        side: str,
    ) -> tuple[ExternalField, dict[str, Any]]:
        """Cache the linear analytic source-to-gain boundary condition."""

        configured_power = float(self.cfg["power_W"])

        def build(quantity: float, record: bool) -> tuple[ExternalField, dict[str, Any]]:
            diagnostics: dict[str, Any] = {}
            n_gain = material_refractive_index(
                self.material,
                float(self.cfg["wavelength_m"]),
                role="pump",
            )
            interface = gain_interface_element(
                self.gain_marker,
                n_gain,
                side,
                self.physics["interface_model"],
            )
            external = build_external_field(
                self.cfg,
                quantity,
                [*self.pre_elements, interface],
                self.grid,
                "abcd",
                "power_W",
                self.physics["interface_model"],
                record,
                # The sampled FFT grid represents a slowly varying envelope,
                # not the optical carrier.  Its usual tens-of-microns pitch
                # cannot resolve a carrier phase ramp at degree-scale angles.
                # The complete ABCD chief-ray angle is handed to the crystal
                # marcher for explicit walk-off; adding this tilt phase as
                # well would alias and count that same direction twice.
                include_abcd_tilt_phase=False,
                abcd_diagnostics=diagnostics,
            )
            return external, diagnostics

        if record_planes or configured_power <= 0.0:
            return build(power, record_planes)
        if self._hybrid_external_cache is None:
            self._hybrid_external_cache = build(configured_power, False)
        cached, cached_diagnostics = self._hybrid_external_cache
        scale = max(power / configured_power, 0.0)
        diagnostics = dict(cached_diagnostics)
        for key in ("analytic_quantity", "sampled_quantity"):
            if key in diagnostics:
                diagnostics[key] = float(diagnostics[key]) * scale
        return (
            ExternalField(
                field=cached.field * math.sqrt(scale),
                refractive_index=cached.refractive_index,
                source_grid_quantity=cached.source_grid_quantity * scale,
                trace=OpticalTrace(),
                warnings=list(cached.warnings),
            ),
            diagnostics,
        )

    def _fft_external_field(self, power: float, record_planes: bool) -> ExternalField:
        """Reuse the linear source-to-crystal field during Pump time stepping."""

        configured_power = float(self.cfg["power_W"])
        if record_planes or configured_power <= 0.0:
            return build_external_field(
                self.cfg,
                power,
                self.pre_elements,
                self.grid,
                "fft",
                "power_W",
                self.physics["interface_model"],
                record_planes,
            )
        if self._fft_external_cache is None:
            self._fft_external_cache = build_external_field(
                self.cfg,
                configured_power,
                self.pre_elements,
                self.grid,
                "fft",
                "power_W",
                self.physics["interface_model"],
                False,
            )
        scale = max(power / configured_power, 0.0)
        cached = self._fft_external_cache
        return ExternalField(
            field=cached.field * math.sqrt(scale),
            refractive_index=cached.refractive_index,
            source_grid_quantity=cached.source_grid_quantity * scale,
            trace=OpticalTrace(),
            warnings=list(cached.warnings),
        )

    def _stats(
        self,
        direction: str,
        configured_power: float,
        source_grid_power: float,
        input_power: float,
        output_power: float,
        absorbed: np.ndarray,
        active_absorbed: np.ndarray | None = None,
        passive_absorbed: np.ndarray | None = None,
    ) -> dict[str, Any]:
        stats = {
            "configured_power_W": configured_power,
            "grid_source_power_W": source_grid_power,
            "grid_input_power_W": input_power,
            "output_power_W": output_power,
            "absorbed_power_W": float(self.grid.integrate_volume(absorbed)),
            "direction": direction,
            "propagation": self.physics["pump_propagation"],
            "depletion": bool(self.physics["pump_depletion"]),
            "nominal_power_before_geometric_efficiency_W": (
                configured_power
                / max(float(self.cfg.get("geometric_efficiency", 1.0)), np.finfo(float).tiny)
                if float(self.cfg.get("geometric_efficiency", 1.0)) > 0.0
                else float(self.cfg["power_W"])
            ),
            "geometric_efficiency": float(self.cfg.get("geometric_efficiency", 1.0)),
            "source_definition": dict(self.cfg.get("source_definition", {}).get("derived", {})),
        }
        if active_absorbed is not None:
            stats["active_absorbed_power_W"] = float(
                self.grid.integrate_volume(active_absorbed)
            )
        if passive_absorbed is not None:
            stats["passive_absorbed_power_W"] = float(
                self.grid.integrate_volume(passive_absorbed)
            )
        if active_absorbed is not None and passive_absorbed is not None:
            component_total = float(
                self.grid.integrate_volume(active_absorbed + passive_absorbed)
            )
            absorbed_total = float(stats["absorbed_power_W"])
            stats["absorption_component_closure_relative"] = abs(
                component_total - absorbed_total
            ) / max(
                abs(component_total),
                abs(absorbed_total),
                np.finfo(float).tiny,
            )
        return stats
