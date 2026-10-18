"""Optional intrapulse gain saturation for coherent broadband fields.

The coherent field remains a one-dimensional complex spectrum at each spatial
pixel.  For one gain step it is transformed to time, decomposed into ordered,
overlapping sine-squared windows, and each window is transformed back to the
frequency domain for wavelength-dependent amplification.  Population changes
are committed after every window, so the pulse front changes the gain seen by
its peak and tail without introducing an independent time-by-wavelength state
array.

The soft-window construction follows the numerical idea in R. Paschotta,
Optics Express 25, 19112 (2017).  It is deliberately isolated from the main
coherent propagator: callers can catch any failure and retain the original
whole-fluence model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .coherent import CoherentPulse
from .coherent_gain_kernel import (
    NUMBA_AVAILABLE,
    configure_numba_threads,
    numba_kernel_compiled_in_process,
    run_single_transition_sweep_numba,
)
from .constants import photon_energy
from .levels import LevelSystem
from .profiling import profile_operation, profiled_operation
from .signal import frantz_nodvik


def energy_protected_soft_windows(time_mass: np.ndarray, slice_count: int, energy_coverage: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Partition the full field into one left tail, N-2 central windows, one right tail.

    The central centres span the equal-tail energy interval. Cosine-squared
    crossfades preserve a partition of unity; the outer windows have constant
    plateaus all the way to the FFT boundaries, so no tail field is discarded.
    """
    from .energy_tail_binning import energy_tail_groups

    mass = np.asarray(time_mass, dtype=float)
    if isinstance(slice_count, bool) or int(slice_count) != slice_count or slice_count < 3:
        raise ValueError("Protected soft windows require at least three total windows")
    count = int(slice_count)
    plan = energy_tail_groups(np.arange(len(mass), dtype=float), mass,
                              coverage=energy_coverage, tail_points=2)
    lo, hi = plan["central_indices"]
    # A delta-like pulse still has a finite sample cell. Fractional centres
    # avoid duplicate windows when that cell has fewer samples than N-2.
    left, right = float(lo), float(hi)
    if left == right:
        left, right = left - 0.5, right + 0.5
    central_count = count - 2
    central = (np.array([(left + right) / 2]) if central_count == 1
               else np.linspace(left, right, central_count))
    spacing = (right - left) / max(central_count - 1, 1)
    centres = np.concatenate(([central[0] - spacing], central, [central[-1] + spacing]))
    x = np.arange(len(mass), dtype=float)
    windows = np.zeros((count, len(mass)))
    windows[0, x <= centres[0]] = 1.0
    windows[-1, x >= centres[-1]] = 1.0
    for i in range(count - 1):
        inside = (x >= centres[i]) & (x < centres[i + 1])
        fraction = (x[inside] - centres[i]) / (centres[i + 1] - centres[i])
        first = np.cos(0.5 * np.pi * fraction) ** 2
        windows[i, inside] = first
        windows[i + 1, inside] = 1.0 - first
    return windows, centres, {
        "enabled": True, "energy_coverage": float(energy_coverage),
        "central_indices": [int(lo), int(hi)],
        "central_energy_fraction": plan["central_energy_fraction"],
        "central_window_count": central_count, "tail_window_counts": [1, 1],
        "total_window_count": count,
        "selection": "first_gain_input_fixed_per_propagation",
    }


def soft_sine_squared_windows(sample_count: int, slice_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered compact windows whose *amplitudes* sum to one.

    Adjacent windows overlap by one half-width and obey cos²(u)+sin²(u)=1.
    Therefore summing untouched windowed complex fields reconstructs the input
    field to floating-point precision.
    """

    samples = int(sample_count)
    if samples < 1:
        raise ValueError("sample_count must be positive")
    count = min(max(int(slice_count), 1), samples)
    if count == 1:
        return np.ones((1, samples), dtype=float), np.zeros(1, dtype=float)

    centers = np.linspace(0.0, float(samples - 1), count)
    spacing = centers[1] - centers[0]
    index = np.arange(samples, dtype=float)
    windows = np.zeros((count, samples), dtype=float)
    for iw, center in enumerate(centers):
        distance = np.abs(index - center)
        inside = distance <= spacing * (1.0 + 8.0 * np.finfo(float).eps)
        windows[iw, inside] = np.cos(0.5 * np.pi * distance[inside] / spacing) ** 2
    # Remove tiny discretization residuals without changing the soft shape.
    windows /= np.maximum(np.sum(windows, axis=0, keepdims=True), np.finfo(float).tiny)
    return windows, centers


@dataclass
class _StepResult:
    field: np.ndarray
    population: np.ndarray
    override_gain: np.ndarray | None
    maximum_reservoir_depletion: float
    energy_balance_error: float
    correction_mean: float
    trace: dict[str, Any]
    passive_loss_fluence_J_m2: np.ndarray


@dataclass(frozen=True)
class _SpectralMaterialData:
    """Environment-versioned tensors reused by every gain call in one z cell."""

    sigma_abs: np.ndarray
    sigma_em: np.ndarray
    saturation_normal: np.ndarray
    saturation_projected: np.ndarray
    valid_saturation_by_frequency: np.ndarray
    valid_saturation_any: np.ndarray
    valid_saturation_all: np.ndarray
    population_update_by_frequency: np.ndarray
    has_finite_saturation: bool


class CoherentTransientGain:
    """Sequential soft-window gain operator for one coherent spectral field."""

    method = "soft_window_spectral"

    def __init__(
        self,
        pulse: CoherentPulse,
        levels: LevelSystem,
        levels_cfg: dict[str, Any],
        material: dict[str, Any],
        physics: dict[str, Any],
        config: dict[str, Any],
        cell_areas_m2: np.ndarray | None = None,
        crystal_mask: np.ndarray | None = None,
    ):
        self.pulse = pulse
        self.levels = levels
        self.levels_cfg = levels_cfg
        self.material = material
        self.physics = physics
        self.config = config
        self.cell_areas_m2 = None if cell_areas_m2 is None else np.asarray(cell_areas_m2, dtype=float)
        self.crystal_mask = None if crystal_mask is None else np.asarray(crystal_mask, dtype=float)
        self.last_passive_loss_fluence_J_m2: np.ndarray | None = None
        if self.cell_areas_m2 is not None and (
            self.cell_areas_m2.ndim != 2 or np.any(self.cell_areas_m2 <= 0.0)
        ):
            raise ValueError("cell_areas_m2 must be a positive (ny, nx) array")
        if self.crystal_mask is not None and (
            self.crystal_mask.ndim != 2
            or np.any(~np.isfinite(self.crystal_mask))
            or np.any((self.crystal_mask < 0.0) | (self.crystal_mask > 1.0))
        ):
            raise ValueError("crystal_mask must be a finite (ny, nx) array between zero and one")
        if str(config.get("method", self.method)) != self.method:
            raise ValueError(f"Unsupported coherent transient-gain method {config.get('method')!r}")
        self.requested_slices = max(1, int(config.get("slice_count", 9)))
        self.tail_protection = bool(config.get("tail_protection_enabled", False))
        self.tail_energy_coverage = float(config.get("tail_energy_coverage", 0.99))
        if not np.isfinite(self.tail_energy_coverage) or not 0 < self.tail_energy_coverage <= 1:
            raise ValueError("tail_energy_coverage must lie in (0, 1]")
        if self.tail_protection and self.requested_slices < 3:
            raise ValueError("Protected soft windows require at least three total windows")
        self._protected_time_mass: np.ndarray | None = None
        self._protected_window_plans: dict[int, dict[str, Any]] = {}
        self.adaptive = bool(config.get("adaptive_slices", True))
        self.adaptive_refinement_predictor = bool(
            config.get("adaptive_refinement_predictor", True)
        )
        self.reuse_adaptive_slice_floor = bool(
            config.get("reuse_adaptive_slice_floor", True)
        )
        self.reuse_adaptive_slice_floor_across_propagations = bool(
            config.get(
                "reuse_adaptive_slice_floor_across_propagations", False
            )
        )
        self.reuse_scratch_buffers = bool(
            config.get("reuse_scratch_buffers", True)
        )
        self.batch_window_ffts = bool(
            config.get("batch_window_ffts", False)
        )
        self.batch_window_fft_max_bytes = int(
            config.get("batch_window_fft_max_bytes", 256 * 1024**2)
        )
        if self.batch_window_fft_max_bytes < 1:
            raise ValueError("batch_window_fft_max_bytes must be positive")
        self.maximum_bypassed_output_energy_fraction = float(
            config.get(
                "maximum_bypassed_output_energy_fraction",
                0.0,
            )
        )
        if not (
            np.isfinite(self.maximum_bypassed_output_energy_fraction)
            and 0.0
            <= self.maximum_bypassed_output_energy_fraction
            < 1.0
        ):
            raise ValueError(
                "maximum_bypassed_output_energy_fraction must lie in [0, 1)"
            )
        self.inactive_window_local_guard_maximum_pixels = int(
            config.get(
                "inactive_window_local_guard_maximum_pixels",
                512,
            )
        )
        if self.inactive_window_local_guard_maximum_pixels < 1:
            raise ValueError(
                "inactive_window_local_guard_maximum_pixels must be positive"
            )
        self.fuse_single_transition_events = bool(
            config.get("fuse_single_transition_events", True)
        )
        self.spectral_coupling_requested = str(
            config.get("spectral_coupling", "ordered_sweeps")
        )
        if self.spectral_coupling_requested not in {
            "ordered_sweeps",
            "broadband_two_stage",
            "auto",
        }:
            raise ValueError(
                "spectral_coupling must be ordered_sweeps, "
                "broadband_two_stage, or auto"
            )
        self.single_transition_kernel_requested = str(
            config.get("single_transition_kernel", "auto")
        )
        if self.single_transition_kernel_requested not in {
            "auto",
            "numpy",
            "numba",
        }:
            raise ValueError(
                "single_transition_kernel must be auto, numpy, or numba"
            )
        if (
            self.single_transition_kernel_requested == "numba"
            and not NUMBA_AVAILABLE
        ):
            raise ValueError(
                "single_transition_kernel=numba requires the performance extra"
            )
        self.single_transition_numba_minimum_packet_cells = int(
            config.get(
                "single_transition_numba_minimum_packet_cells",
                100_000,
            )
        )
        if self.single_transition_numba_minimum_packet_cells < 1:
            raise ValueError(
                "single_transition_numba_minimum_packet_cells must be positive"
            )
        self.spectral_material_cache_max_bytes = int(
            config.get(
                "spectral_material_cache_max_bytes", 512 * 1024**2
            )
        )
        if self.spectral_material_cache_max_bytes < 1:
            raise ValueError(
                "spectral_material_cache_max_bytes must be positive"
            )
        self.single_transition_kernel_selected = (
            "numba"
            if self.single_transition_kernel_requested == "numba"
            else "numpy"
        )
        self.single_transition_kernel_workers = int(
            config.get("single_transition_kernel_workers", 0)
        )
        if self.single_transition_kernel_workers < 0:
            raise ValueError(
                "single_transition_kernel_workers must be non-negative"
            )
        self.single_transition_kernel_active_workers = (
            configure_numba_threads(self.single_transition_kernel_workers)
            if self.single_transition_kernel_selected == "numba"
            else None
        )
        self.maximum_slices = max(self.requested_slices, int(config.get("maximum_slice_count", 65)))
        self.depletion_target = float(config.get("maximum_depletion_per_slice", 0.05))
        if not 0.0 < self.depletion_target < 1.0:
            raise ValueError("maximum_depletion_per_slice must lie between 0 and 1")
        self.adaptive_depletion_metric = str(
            config.get(
                "adaptive_depletion_metric", "total_population_change"
            )
        )
        if self.adaptive_depletion_metric not in {
            "total_population_change",
            "legacy_upper_relative",
        }:
            raise ValueError(
                "adaptive_depletion_metric must be total_population_change "
                "or legacy_upper_relative"
            )
        self.include_relaxation = bool(config.get("include_relaxation_during_pulse", False))
        self.relaxation_energy_fraction = float(
            config.get("relaxation_energy_fraction", 0.999999)
        )
        if not 0.0 < self.relaxation_energy_fraction <= 1.0:
            raise ValueError("relaxation_energy_fraction must lie in (0, 1]")
        self._override_gain_m_inv: np.ndarray | None = None
        self._call_count = 0
        self._diagnostics: dict[str, Any] = {}
        self._scratch: dict[str, np.ndarray] = {}
        self._window_cache: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self._spectral_material_cache_key: tuple[Any, ...] | None = None
        self._spectral_material_cache: _SpectralMaterialData | None = None
        self._wavelengths = np.asarray(self.pulse.wavelength_m, dtype=float)
        self._photon_energies = photon_energy(self._wavelengths)
        self._sample_indices = np.arange(len(self.pulse.time_s), dtype=float)
        self._forward_frequency_order = tuple(range(len(self._wavelengths)))
        self._backward_frequency_order = tuple(
            range(len(self._wavelengths) - 1, -1, -1)
        )
        self._signal_upper_indices = tuple(
            sorted(
                {
                    transition.upper
                    for transition in self.levels.signal_transitions
                }
            )
        )
        self._signal_population_indices = tuple(
            sorted(
                {
                    index
                    for transition in self.levels.signal_transitions
                    for index in (transition.upper, transition.lower)
                }
            )
        )
        self._adaptive_slice_floor_by_z: dict[int, int] = {}
        self.begin_propagation(None)

    def begin_propagation(self, population_shape: tuple[int, ...] | None) -> None:
        """Reset per-pulse diagnostics and the optional user-gain reservoir."""

        if self.tail_protection:
            self._protected_time_mass = None
            self._window_cache.clear()
            self._protected_window_plans.clear()
        estimated_packet_cells = 0
        if population_shape is not None:
            if len(population_shape) != 4:
                raise ValueError("population must have shape (level, z, y, x)")
            estimated_packet_cells = int(
                2
                * len(self.pulse.frequency_Hz)
                * self.requested_slices
                * population_shape[1]
                * population_shape[2]
                * population_shape[3]
            )
        if self.single_transition_kernel_requested == "auto":
            self.single_transition_kernel_selected = (
                "numba"
                if NUMBA_AVAILABLE
                and estimated_packet_cells
                >= self.single_transition_numba_minimum_packet_cells
                else "numpy"
            )
        self.single_transition_kernel_active_workers = (
            configure_numba_threads(self.single_transition_kernel_workers)
            if self.single_transition_kernel_selected == "numba"
            else None
        )

        override = self.levels_cfg.get("gain_override_m_inv")
        if override is not None and population_shape is not None:
            self._override_gain_m_inv = np.full(population_shape[1:], float(override), dtype=float)
        else:
            self._override_gain_m_inv = None
        self._call_count = 0
        if not self.reuse_adaptive_slice_floor_across_propagations:
            self._adaptive_slice_floor_by_z = {}
        self._spectral_material_cache_key = None
        self._spectral_material_cache = None
        self._diagnostics = {
            "enabled": True,
            "status": "ready",
            "method": self.method,
            "tail_protection": {"enabled": self.tail_protection},
            "requested_slice_count": self.requested_slices,
            "maximum_slice_count": self.maximum_slices,
            "adaptive_slices": self.adaptive,
            "adaptive_refinement_predictor": (
                self.adaptive_refinement_predictor
            ),
            "reuse_adaptive_slice_floor": self.reuse_adaptive_slice_floor,
            "reuse_adaptive_slice_floor_across_propagations": (
                self.reuse_adaptive_slice_floor_across_propagations
            ),
            "persistent_adaptive_floor_entries": len(
                self._adaptive_slice_floor_by_z
            ),
            "scratch_buffer_reuse": self.reuse_scratch_buffers,
            "batch_window_ffts": self.batch_window_ffts,
            "batch_window_ffts_requested": self.batch_window_ffts,
            "batch_window_ffts_active": False,
            "batch_window_fft_max_bytes": self.batch_window_fft_max_bytes,
            "batch_window_fft_max_estimated_bytes": 0,
            "batch_window_fft_calls": 0,
            "batch_window_fft_fallback_calls": 0,
            "batch_window_fft_fallback_reason": None,
            "maximum_bypassed_output_energy_fraction": (
                self.maximum_bypassed_output_energy_fraction
            ),
            "inactive_window_local_guard_maximum_pixels": (
                self.inactive_window_local_guard_maximum_pixels
            ),
            "inactive_window_bypass_calls": 0,
            "inactive_windows_bypassed": 0,
            "active_windows_processed": 0,
            "maximum_actual_bypassed_input_energy_fraction": 0.0,
            "maximum_estimated_bypassed_output_energy_fraction": 0.0,
            "maximum_local_estimated_bypassed_output_energy_fraction": 0.0,
            "inactive_window_bypass_fallback_reason": None,
            "inactive_window_adaptive_guard_rechecks": 0,
            "inactive_window_adaptive_guard_refinements": 0,
            "scratch_buffer_count": 0,
            "scratch_buffer_bytes": 0,
            "single_transition_fused_path_configured": (
                self.fuse_single_transition_events
            ),
            "single_transition_fused_path_used": False,
            "single_transition_fused_event_calls": 0,
            "single_transition_kernel_requested": (
                self.single_transition_kernel_requested
            ),
            "single_transition_kernel_selected": (
                self.single_transition_kernel_selected
            ),
            "single_transition_kernel_workers": (
                self.single_transition_kernel_active_workers
            ),
            "single_transition_numba_minimum_packet_cells": (
                self.single_transition_numba_minimum_packet_cells
            ),
            "single_transition_estimated_packet_cells_per_propagation": (
                estimated_packet_cells
            ),
            "single_transition_numba_windows": 0,
            "single_transition_numba_cold_start_calls": 0,
            "spectral_coupling_requested": self.spectral_coupling_requested,
            "spectral_coupling_selected": "ordered_sweeps",
            "broadband_two_stage_windows": 0,
            "broadband_two_stage_event_batches": 0,
            "broadband_fallback_windows": 0,
            "broadband_fallback_reason": None,
            "spectral_material_cache_hits": 0,
            "spectral_material_cache_misses": 0,
            "spectral_material_cache_bypasses": 0,
            "spectral_material_cache_max_bytes": (
                self.spectral_material_cache_max_bytes
            ),
            "spectral_material_cache_max_estimated_bytes": 0,
            "spectral_material_cache_scope": (
                "single_current_z_and_environment_revision"
            ),
            "maximum_depletion_per_slice_target": self.depletion_target,
            "adaptive_depletion_metric": self.adaptive_depletion_metric,
            "include_relaxation_during_pulse": self.include_relaxation,
            "relaxation_energy_fraction": self.relaxation_energy_fraction,
            "maximum_physical_relaxation_window_s": 0.0,
            "spectral_coupling": "symmetric_forward_reverse_half_sweeps",
            "gain_steps_completed": 0,
            "largest_slice_count_used": 0,
            "adaptive_retries": 0,
            "adaptive_predictive_jumps": 0,
            "adaptive_floor_warm_starts": 0,
            "maximum_adaptive_start_slice_count": 0,
            "last_starting_slice_count": 0,
            "last_slice_count_used": 0,
            "maximum_reservoir_depletion_per_slice": 0.0,
            "maximum_resynthesis_energy_error": 0.0,
            "maximum_unsaturated_gain_exponent": None,
            "positive_gain_exponent_clipping_active": False,
            "depletion_target_met": True,
            "native_temporal_sample_count": int(len(self.pulse.time_s)),
            "native_temporal_step_s": float(self.pulse.dt_s),
            "representative_trace": None,
        }

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._diagnostics)

    def _scratch_array(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype[Any] | type,
    ) -> np.ndarray:
        array = self._scratch.get(name)
        expected_dtype = np.dtype(dtype)
        if (
            array is None
            or array.shape != shape
            or array.dtype != expected_dtype
        ):
            array = np.empty(shape, dtype=expected_dtype)
            self._scratch[name] = array
        return array

    def _windows(
        self, slice_count: int, temporal_energy: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._window_cache.get(int(slice_count))
        if cached is None:
            if self.tail_protection:
                if self._protected_time_mass is None:
                    if temporal_energy is None:
                        raise ValueError("Protected windows need the first gain input field")
                    weights = np.ones(temporal_energy.shape[1:])
                    if self.cell_areas_m2 is not None:
                        weights *= self.cell_areas_m2
                    if self.crystal_mask is not None:
                        weights *= self.crystal_mask
                    self._protected_time_mass = np.sum(temporal_energy * weights[None], axis=(1, 2))
                windows, centers, plan = energy_protected_soft_windows(
                    self._protected_time_mass, slice_count, self.tail_energy_coverage)
                self._protected_window_plans[int(slice_count)] = plan
            else:
                windows, centers = soft_sine_squared_windows(
                    len(self.pulse.time_s), slice_count
                )
            coverage = np.sum(windows**2, axis=0)
            windows.setflags(write=False)
            centers.setflags(write=False)
            coverage.setflags(write=False)
            cached = (windows, centers, coverage)
            self._window_cache[int(slice_count)] = cached
        if self.tail_protection:
            self._diagnostics["tail_protection"] = self._protected_window_plans[int(slice_count)]
        return cached

    def _spectral_material_data(
        self,
        *,
        population_shape: tuple[int, int],
        z_index: int,
        projected_area_cosine: float,
        use_saturation: bool,
    ) -> _SpectralMaterialData:
        """Return local spectroscopy without crossing an environment update.

        Temperature and concentration are constant during one accepted Seed
        propagation today, but may change between pulses or in future transient
        coupling.  The LevelSystem revision makes that boundary explicit.  A
        single-entry cache is sufficient because z is traversed sequentially
        and avoids retaining hundreds of megabytes of wavelength/radius data.
        """

        environment_revision = int(
            getattr(self.levels.response, "environment_revision", 0)
        )
        key = (
            int(z_index),
            tuple(int(value) for value in population_shape),
            float(projected_area_cosine),
            bool(use_saturation),
            environment_revision,
        )
        if (
            key == self._spectral_material_cache_key
            and self._spectral_material_cache is not None
        ):
            self._diagnostics["spectral_material_cache_hits"] = int(
                self._diagnostics.get("spectral_material_cache_hits", 0)
            ) + 1
            return self._spectral_material_cache

        # Release the preceding z plane before constructing the next one.  A
        # Cartesian broadband plane can be hundreds of megabytes; retaining
        # old and new tensors during construction would double the peak with no
        # possible cache hit because propagation is monotonic in z.
        self._spectral_material_cache_key = None
        self._spectral_material_cache = None

        spectral_cells = int(
            len(self._wavelengths) * np.prod(population_shape, dtype=np.int64)
        )
        # Four float tensors (absorption/emission and normal/projected
        # saturation), three boolean masks, and frequency-only masks.  The
        # estimate is deliberately conservative and is evaluated before any
        # large allocation.
        estimated_cache_bytes = int(
            spectral_cells
            * (4 * np.dtype(float).itemsize + 3 * np.dtype(bool).itemsize)
            + 3 * len(self._wavelengths) * np.dtype(bool).itemsize
        )
        self._diagnostics[
            "spectral_material_cache_max_estimated_bytes"
        ] = max(
            int(
                self._diagnostics.get(
                    "spectral_material_cache_max_estimated_bytes", 0
                )
            ),
            estimated_cache_bytes,
        )
        cache_allowed = bool(
            estimated_cache_bytes <= self.spectral_material_cache_max_bytes
        )

        sigma_abs, sigma_em = self.levels.local_cross_sections(
            "signal",
            self._wavelengths,
            population_shape,
            z_index,
        )
        sigma_sum = self.levels.signal_cross_section_sum_m2(
            self._wavelengths,
            population_shape,
            z_index,
        )
        configured_saturation = self.levels_cfg.get(
            "saturation_fluence_J_m2"
        )
        if configured_saturation is None:
            saturation_normal = np.divide(
                self._photon_energies[:, None, None],
                sigma_sum,
                out=np.full_like(sigma_sum, np.inf),
                where=sigma_sum > 0.0,
            )
        else:
            saturation_normal = np.full_like(
                sigma_sum, float(configured_saturation)
            )
        saturation_projected = (
            saturation_normal * float(projected_area_cosine)
        )
        valid_saturation_by_frequency = np.isfinite(
            saturation_projected
        ) & (saturation_projected > 0.0)
        spatial_axes = tuple(
            range(1, valid_saturation_by_frequency.ndim)
        )
        valid_saturation_any = np.any(
            valid_saturation_by_frequency, axis=spatial_axes
        )
        valid_saturation_all = np.all(
            valid_saturation_by_frequency, axis=spatial_axes
        )
        population_update_by_frequency = (
            np.any(sigma_sum > 0.0, axis=tuple(range(1, sigma_sum.ndim)))
            & valid_saturation_any
        )
        data = _SpectralMaterialData(
            sigma_abs=np.asarray(sigma_abs, dtype=float),
            sigma_em=np.asarray(sigma_em, dtype=float),
            saturation_normal=saturation_normal,
            saturation_projected=saturation_projected,
            valid_saturation_by_frequency=valid_saturation_by_frequency,
            valid_saturation_any=valid_saturation_any,
            valid_saturation_all=valid_saturation_all,
            population_update_by_frequency=population_update_by_frequency,
            has_finite_saturation=bool(
                use_saturation and np.any(valid_saturation_by_frequency)
            ),
        )
        if cache_allowed:
            self._spectral_material_cache_key = key
            self._spectral_material_cache = data
        else:
            self._diagnostics["spectral_material_cache_bypasses"] = int(
                self._diagnostics.get(
                    "spectral_material_cache_bypasses", 0
                )
            ) + 1
        self._diagnostics["spectral_material_cache_misses"] = int(
            self._diagnostics.get("spectral_material_cache_misses", 0)
        ) + 1
        return data

    @staticmethod
    def _magnitude_squared(
        values: np.ndarray,
        output: np.ndarray,
    ) -> np.ndarray:
        # NumPy's complex absolute is stable for very large/small components.
        # Squaring it in place needs no second full-size float buffer.
        np.absolute(values, out=output)
        np.square(output, out=output)
        return output

    def _run_broadband_two_stage(
        self,
        *,
        input_fluence_by_frequency: np.ndarray,
        correction: np.ndarray,
        population: np.ndarray,
        transition: Any,
        sigma_abs: np.ndarray,
        sigma_em: np.ndarray,
        saturation: np.ndarray,
        valid_saturation: np.ndarray,
        update_frequency: np.ndarray,
        material_domain: np.ndarray,
        photon_energies: np.ndarray,
        dz: float,
        cell_length_m: float,
        output_by_frequency: np.ndarray,
        output_weight: np.ndarray,
        event_applied: np.ndarray,
        event_scale: np.ndarray,
        total_population: np.ndarray,
        meaningful_threshold: np.ndarray,
    ) -> None:
        """Advance one soft window with order-independent broadband coupling.

        All frequencies in each of two equal exposure stages see the same
        population state.  Their requested photon events are then reduced and
        committed as one conservative transition update.  The second stage
        sees the population left by the first, retaining front-to-tail
        saturation while removing the arbitrary native-bin ordering from the
        population chain.  The coherent field remains on its full frequency
        grid; no spectral samples are discarded or interpolated.

        A common limiter scale is applied back to every contributing spectral
        event in a stage, so the field-energy change and committed photon
        events remain identical even at a depleted pixel.
        """

        shape = input_fluence_by_frequency.shape
        effective_input = self._scratch_array(
            "broadband_effective_input", shape, float
        )
        delta_fluence = self._scratch_array(
            "broadband_delta_fluence", shape, float
        )
        exponent_or_events = self._scratch_array(
            "broadband_exponent_or_events", shape, float
        )
        requested_events = self._scratch_array(
            "broadband_requested_events_xy",
            population.shape[-2:],
            float,
        )

        np.multiply(
            input_fluence_by_frequency,
            0.5 * correction[None, ...],
            out=effective_input,
        )
        update_mask = (
            valid_saturation
            & np.asarray(update_frequency, dtype=bool)[:, None, None]
        )
        all_saturated = bool(np.all(valid_saturation))
        any_saturated = bool(np.any(valid_saturation))
        photon_volume = (
            np.asarray(photon_energies, dtype=float)[:, None, None]
            * float(cell_length_m)
        )
        tiny = np.finfo(float).tiny
        output_by_frequency.fill(0.0)

        for _stage in range(2):
            np.multiply(
                sigma_em,
                population[transition.upper][None, ...],
                out=exponent_or_events,
            )
            np.multiply(
                sigma_abs,
                population[transition.lower][None, ...],
                out=delta_fluence,
            )
            np.subtract(
                exponent_or_events,
                delta_fluence,
                out=exponent_or_events,
            )
            exponent_or_events *= material_domain[None, ...] * float(dz)

            if all_saturated:
                np.copyto(
                    delta_fluence,
                    frantz_nodvik(
                        effective_input,
                        exponent_or_events,
                        saturation,
                    ),
                )
            else:
                np.exp(
                    np.minimum(exponent_or_events, 50.0),
                    out=delta_fluence,
                )
                delta_fluence *= effective_input
                if any_saturated:
                    safe_saturation = np.where(
                        valid_saturation,
                        saturation,
                        1.0,
                    )
                    saturated_output = frantz_nodvik(
                        effective_input,
                        exponent_or_events,
                        safe_saturation,
                    )
                    np.copyto(
                        delta_fluence,
                        saturated_output,
                        where=valid_saturation,
                    )
                unsaturated = ~valid_saturation
                if np.any(unsaturated):
                    local_maximum = float(
                        np.max(exponent_or_events[unsaturated])
                    )
                    previous_maximum = self._diagnostics.get(
                        "maximum_unsaturated_gain_exponent"
                    )
                    self._diagnostics[
                        "maximum_unsaturated_gain_exponent"
                    ] = (
                        local_maximum
                        if previous_maximum is None
                        else max(float(previous_maximum), local_maximum)
                    )
                    self._diagnostics[
                        "positive_gain_exponent_clipping_active"
                    ] = bool(
                        self._diagnostics.get(
                            "positive_gain_exponent_clipping_active", False
                        )
                        or np.any(exponent_or_events[unsaturated] > 50.0)
                    )

            np.subtract(
                delta_fluence,
                effective_input,
                out=delta_fluence,
            )
            np.divide(
                delta_fluence,
                photon_volume,
                out=exponent_or_events,
            )
            np.copyto(exponent_or_events, 0.0, where=~update_mask)
            np.sum(
                exponent_or_events,
                axis=0,
                out=requested_events,
            )
            applied = self.levels.apply_single_signal_photon_events_inplace(
                population,
                requested_events,
                applied_events_out=event_applied,
                scale_work=event_scale,
                total_population_m3=total_population,
                meaningful_event_threshold_m3=meaningful_threshold,
            )
            finite_request = np.abs(requested_events) > tiny
            np.divide(
                applied,
                requested_events,
                out=event_scale,
                where=finite_request,
            )
            event_scale[~finite_request] = 1.0
            np.clip(event_scale, 0.0, 1.0, out=event_scale)
            np.multiply(
                delta_fluence,
                event_scale[None, ...],
                out=delta_fluence,
                where=update_mask,
            )
            np.add(delta_fluence, effective_input, out=delta_fluence)
            np.maximum(delta_fluence, 0.0, out=delta_fluence)
            output_by_frequency += delta_fluence

        np.sum(output_by_frequency, axis=0, out=output_weight)
        self._diagnostics["spectral_coupling_selected"] = (
            "broadband_two_stage"
        )
        self._diagnostics["spectral_coupling"] = (
            "simultaneous_broadband_two_equal_exposure_stages"
        )
        self._diagnostics["broadband_two_stage_windows"] = int(
            self._diagnostics.get("broadband_two_stage_windows", 0)
        ) + 1
        self._diagnostics["broadband_two_stage_event_batches"] = int(
            self._diagnostics.get("broadband_two_stage_event_batches", 0)
        ) + 2

    @profiled_operation("signal.coherent_transient_gain")
    def apply(
        self,
        spectral_field: np.ndarray,
        population_plane: np.ndarray,
        iz: int,
        step_length_m: float,
        cell_length_m: float | None = None,
        relaxation_time_scale: float = 1.0,
        projected_area_cosine: float = 1.0,
    ) -> np.ndarray:
        """Apply one z-step and commit the accepted sequential population state."""

        field = np.asarray(spectral_field, dtype=complex)
        if field.ndim != 3 or field.shape[0] != len(self.pulse.frequency_Hz):
            raise ValueError("spectral_field must have shape (frequency, y, x)")
        if population_plane.ndim != 3:
            raise ValueError("population_plane must have shape (level, y, x)")
        if self.crystal_mask is not None and self.crystal_mask.shape != field.shape[-2:]:
            raise ValueError("coherent field shape does not match crystal_mask")
        dz = float(step_length_m)
        if not np.isfinite(dz) or dz <= 0.0:
            raise ValueError("step_length_m must be positive and finite")
        cell_length = dz if cell_length_m is None else float(cell_length_m)
        if not np.isfinite(cell_length) or cell_length <= 0.0:
            raise ValueError("cell_length_m must be positive and finite")
        relaxation_scale = float(relaxation_time_scale)
        if not np.isfinite(relaxation_scale) or not 0.0 <= relaxation_scale <= 1.0:
            raise ValueError("relaxation_time_scale must lie in [0, 1]")
        area_cosine = float(projected_area_cosine)
        if not np.isfinite(area_cosine) or not 0.0 < area_cosine <= 1.0:
            raise ValueError("projected_area_cosine must lie in (0, 1]")

        initial_population = population_plane.copy()
        initial_override = None
        if self._override_gain_m_inv is not None:
            initial_override = self._override_gain_m_inv[iz].copy()

        sample_limit = len(self.pulse.time_s)
        slice_limit = min(self.maximum_slices, sample_limit)
        requested_count = min(self.requested_slices, slice_limit)
        remembered_count = (
            self._adaptive_slice_floor_by_z.get(int(iz), requested_count)
            if self.adaptive and self.reuse_adaptive_slice_floor
            else requested_count
        )
        count = min(max(requested_count, int(remembered_count)), slice_limit)
        starting_count = count
        if starting_count > requested_count:
            self._diagnostics["adaptive_floor_warm_starts"] = (
                int(self._diagnostics["adaptive_floor_warm_starts"]) + 1
            )
        retries = 0
        while True:
            trial = self._apply_once(
                field,
                initial_population.copy(),
                None if initial_override is None else initial_override.copy(),
                dz,
                cell_length,
                count,
                reverse_spectral_order=bool(self._call_count % 2),
                z_index=iz,
                relaxation_time_scale=relaxation_scale,
                projected_area_cosine=area_cosine,
                allow_broadband=True,
                allow_inactive_window_bypass=True,
            )
            bypass_can_affect_adaptive_decision = bool(
                self.maximum_bypassed_output_energy_fraction > 0.0
                and not self.include_relaxation
                and (
                    initial_override is not None
                    or (
                        self.fuse_single_transition_events
                        and self.reuse_scratch_buffers
                        and len(self.levels.signal_transitions) == 1
                    )
                )
            )
            adaptive_guard_margin = min(
                self.depletion_target,
                max(
                    0.01 * self.depletion_target,
                    4.0
                    * np.sqrt(
                        self.maximum_bypassed_output_energy_fraction
                    )
                    * self.depletion_target,
                ),
            )
            if (
                bypass_can_affect_adaptive_decision
                and self.adaptive
                and bool(self.physics.get("gain_saturation", True))
                and self.depletion_target - adaptive_guard_margin
                <= trial.maximum_reservoir_depletion
                <= self.depletion_target
            ):
                bypassed_trial_depletion = (
                    trial.maximum_reservoir_depletion
                )
                trial = self._apply_once(
                    field,
                    initial_population.copy(),
                    (
                        None
                        if initial_override is None
                        else initial_override.copy()
                    ),
                    dz,
                    cell_length,
                    count,
                    reverse_spectral_order=bool(self._call_count % 2),
                    z_index=iz,
                    relaxation_time_scale=relaxation_scale,
                    projected_area_cosine=area_cosine,
                    allow_broadband=True,
                    allow_inactive_window_bypass=False,
                )
                self._diagnostics[
                    "inactive_window_adaptive_guard_rechecks"
                ] = int(
                    self._diagnostics.get(
                        "inactive_window_adaptive_guard_rechecks", 0
                    )
                ) + 1
                if (
                    bypassed_trial_depletion <= self.depletion_target
                    < trial.maximum_reservoir_depletion
                ):
                    self._diagnostics[
                        "inactive_window_adaptive_guard_refinements"
                    ] = int(
                        self._diagnostics.get(
                            "inactive_window_adaptive_guard_refinements", 0
                        )
                    ) + 1
            needs_refinement = (
                self.adaptive
                and bool(self.physics.get("gain_saturation", True))
                and trial.maximum_reservoir_depletion > self.depletion_target
                and count < slice_limit
            )
            if not needs_refinement:
                break
            geometric_refinement = max(count + 1, 2 * count - 1)
            if self.adaptive_refinement_predictor:
                predicted_refinement = max(
                    count + 1,
                    int(
                        np.ceil(
                            1.1
                            * count
                            * trial.maximum_reservoir_depletion
                            / self.depletion_target
                        )
                    ),
                )
            else:
                predicted_refinement = geometric_refinement
            refined = min(slice_limit, predicted_refinement)
            if refined == count:
                break
            if (
                self.adaptive_refinement_predictor
                and refined != min(slice_limit, geometric_refinement)
            ):
                self._diagnostics["adaptive_predictive_jumps"] = int(
                    self._diagnostics.get("adaptive_predictive_jumps", 0)
                ) + 1
            count = refined
            retries += 1

        if (
            self.spectral_coupling_requested
            in {"broadband_two_stage", "auto"}
            and trial.maximum_reservoir_depletion
            > self.depletion_target
        ):
            # The broadband closure assumes that one exposure stage changes
            # the shared reservoir only weakly.  If temporal refinement cannot
            # establish that contract, recompute this gain call with the
            # retained ordered reference instead of silently accepting an
            # uncontrolled approximation.
            self._diagnostics["broadband_fallback_reason"] = (
                "population_change_exceeds_broadband_contract"
            )
            self._diagnostics["broadband_fallback_windows"] = int(
                self._diagnostics.get("broadband_fallback_windows", 0)
            ) + count
            trial = self._apply_once(
                field,
                initial_population.copy(),
                None if initial_override is None else initial_override.copy(),
                dz,
                cell_length,
                count,
                reverse_spectral_order=bool(self._call_count % 2),
                z_index=iz,
                relaxation_time_scale=relaxation_scale,
                projected_area_cosine=area_cosine,
                allow_broadband=False,
                allow_inactive_window_bypass=True,
            )

        if self.adaptive and self.reuse_adaptive_slice_floor:
            self._adaptive_slice_floor_by_z[int(iz)] = max(
                int(self._adaptive_slice_floor_by_z.get(int(iz), requested_count)),
                count,
            )
            self._diagnostics["persistent_adaptive_floor_entries"] = len(
                self._adaptive_slice_floor_by_z
            )

        population_plane[:] = trial.population
        self.last_passive_loss_fluence_J_m2 = trial.passive_loss_fluence_J_m2.copy()
        if self._override_gain_m_inv is not None and trial.override_gain is not None:
            self._override_gain_m_inv[iz] = trial.override_gain

        self._call_count += 1
        previous_max = float(self._diagnostics["maximum_reservoir_depletion_per_slice"])
        self._diagnostics["status"] = "applied"
        self._diagnostics["gain_steps_completed"] = self._call_count
        self._diagnostics["largest_slice_count_used"] = max(
            int(self._diagnostics["largest_slice_count_used"]), count
        )
        self._diagnostics["adaptive_retries"] = int(self._diagnostics["adaptive_retries"]) + retries
        self._diagnostics["maximum_adaptive_start_slice_count"] = max(
            int(self._diagnostics["maximum_adaptive_start_slice_count"]),
            starting_count,
        )
        self._diagnostics["last_starting_slice_count"] = starting_count
        self._diagnostics["last_slice_count_used"] = count
        self._diagnostics["maximum_reservoir_depletion_per_slice"] = max(
            previous_max, trial.maximum_reservoir_depletion
        )
        self._diagnostics["maximum_resynthesis_energy_error"] = max(
            float(self._diagnostics["maximum_resynthesis_energy_error"]), trial.energy_balance_error
        )
        self._diagnostics["soft_window_energy_correction_mean"] = trial.correction_mean
        if self._diagnostics["representative_trace"] is None or trial.maximum_reservoir_depletion >= previous_max:
            self._diagnostics["representative_trace"] = trial.trace
        if trial.maximum_reservoir_depletion > self.depletion_target:
            self._diagnostics["depletion_target_met"] = False
            if not self.adaptive:
                self._diagnostics["accuracy_warning_kind"] = "adaptive_slices_disabled"
                self._diagnostics["accuracy_warning"] = (
                    "The fixed soft temporal slices exceeded the configured reservoir "
                    "depletion target. Enable adaptive_slices or increase slice_count "
                    "and verify output convergence."
                )
            elif self.maximum_slices < sample_limit:
                self._diagnostics["accuracy_warning_kind"] = "maximum_slice_count_reached"
                self._diagnostics["accuracy_warning"] = (
                    "The strongest soft temporal slice depleted the local gain reservoir by more than the "
                    "configured target. Increase maximum_slice_count up to the native temporal sample count "
                    "and verify output convergence."
                )
            else:
                self._diagnostics["accuracy_warning_kind"] = "native_time_bin_limit_reached"
                self._diagnostics["accuracy_warning"] = (
                    "The strongest native temporal bin depleted the local gain reservoir by more than the "
                    "configured target. More soft slices or a larger sample_count alone cannot reduce the "
                    "native time step; increase coherent.span_fwhm while retaining enough samples, or use a "
                    "time-transport solver, then verify output convergence."
                )
        if self.reuse_scratch_buffers:
            self._diagnostics["scratch_buffer_count"] = len(self._scratch)
            self._diagnostics["scratch_buffer_bytes"] = int(
                sum(array.nbytes for array in self._scratch.values())
            )
        return trial.field

    def _apply_once(
        self,
        spectral_field: np.ndarray,
        population: np.ndarray,
        override_gain: np.ndarray | None,
        dz: float,
        cell_length_m: float,
        slice_count: int,
        reverse_spectral_order: bool,
        z_index: int,
        relaxation_time_scale: float,
        projected_area_cosine: float,
        allow_broadband: bool,
        allow_inactive_window_bypass: bool,
    ) -> _StepResult:
        temporal_field_before_passive = self.pulse.spectrum_to_time(
            spectral_field, axis=0
        )
        material_domain = (
            np.ones(population.shape[-2:], dtype=float)
            if self.crystal_mask is None
            else self.crystal_mask
        )
        passive_loss = float(self.material.get("signal_passive_loss_m_inv", 0.0))
        passive_half_amplitude = (
            None
            if self.reuse_scratch_buffers and passive_loss == 0.0
            else np.exp(-0.25 * passive_loss * dz * material_domain)
        )
        original_input_fluence: np.ndarray | None = None
        if self.reuse_scratch_buffers:
            temporal_energy = self._scratch_array(
                "frequency_float_a",
                temporal_field_before_passive.shape,
                float,
            )
            self._magnitude_squared(
                temporal_field_before_passive,
                temporal_energy,
            )
            original_input_fluence = np.sum(temporal_energy, axis=0)
            temporal_field = temporal_field_before_passive
            if passive_half_amplitude is not None:
                temporal_field *= passive_half_amplitude[None, :, :]
                temporal_energy *= passive_half_amplitude[None, :, :] ** 2
        else:
            assert passive_half_amplitude is not None
            temporal_field = (
                temporal_field_before_passive
                * passive_half_amplitude[None, :, :]
            )
            temporal_energy = np.abs(temporal_field) ** 2
        windows, center_indices, coverage = self._windows(slice_count, temporal_energy)
        passive_half_power_loss_fraction = -np.expm1(
            -0.5 * passive_loss * dz * material_domain
        )
        relaxation_start_s, relaxation_end_s = (
            self.physical_relaxation_support_s()
        )
        physical_relaxation_window_s = max(
            relaxation_end_s - relaxation_start_s, 0.0
        )
        self._diagnostics["maximum_physical_relaxation_window_s"] = max(
            float(
                self._diagnostics.get("maximum_physical_relaxation_window_s", 0.0)
            ),
            physical_relaxation_window_s,
        )
        if self.include_relaxation:
            # Protected tail windows can place their first exposure inside the
            # physical support. Their optical plateau still owns the left
            # tail, but the elapsed dark time must not disappear with it.
            first_center_s = float(
                np.interp(
                    center_indices[0], self._sample_indices, self.pulse.time_s
                )
            )
            initial_dark_dt = max(
                float(np.clip(first_center_s, relaxation_start_s, relaxation_end_s))
                - relaxation_start_s,
                0.0,
            ) * relaxation_time_scale
            if initial_dark_dt > 0.0:
                population[:] = self.levels.advance_dark_implicit(
                    population, initial_dark_dt, z_index
                )
        total_energy_xy = np.sum(temporal_energy, axis=0)
        covered_energy_xy = np.sum(coverage[:, None, None] * temporal_energy, axis=0)
        correction = np.divide(
            total_energy_xy,
            covered_energy_xy,
            out=np.ones_like(total_energy_xy),
            where=covered_energy_xy > np.finfo(float).tiny,
        )
        batched_slice_spectra: np.ndarray | None = None
        batched_processed_spectra: np.ndarray | None = None
        # A batched call removes per-window FFT dispatch overhead but keeps
        # several slice_count-sized complex tensors alive at once.  Six is a
        # conservative allowance for the windowed input, shifted input, FFT
        # output, processed spectra, inverse-FFT workspace, and shifted output.
        batch_window_fft_estimated_bytes = int(
            6 * slice_count * temporal_field.nbytes
        )
        self._diagnostics["batch_window_fft_max_estimated_bytes"] = max(
            int(
                self._diagnostics.get(
                    "batch_window_fft_max_estimated_bytes", 0
                )
            ),
            batch_window_fft_estimated_bytes,
        )
        batch_window_ffts_active = bool(
            self.batch_window_ffts
            and batch_window_fft_estimated_bytes
            <= self.batch_window_fft_max_bytes
        )
        if batch_window_ffts_active:
            self._diagnostics["batch_window_ffts_active"] = True
            self._diagnostics["batch_window_fft_calls"] = int(
                self._diagnostics.get("batch_window_fft_calls", 0)
            ) + 1
        elif self.batch_window_ffts:
            self._diagnostics["batch_window_fft_fallback_calls"] = int(
                self._diagnostics.get("batch_window_fft_fallback_calls", 0)
            ) + 1
            self._diagnostics["batch_window_fft_fallback_reason"] = (
                "estimated_peak_bytes_exceeds_limit"
            )
        if batch_window_ffts_active:
            windowed_time = (
                windows[:, :, None, None]
                * temporal_field[None, ...]
            )
            batched_slice_spectra = self.pulse.time_to_spectrum(
                windowed_time,
                axis=1,
            )
            batched_processed_spectra = np.empty_like(
                batched_slice_spectra
            )

        wavelengths = self._wavelengths
        photon_energies = self._photon_energies
        use_saturation = bool(self.physics.get("gain_saturation", True))
        spectral_material = self._spectral_material_data(
            population_shape=population.shape[-2:],
            z_index=z_index,
            projected_area_cosine=projected_area_cosine,
            use_saturation=use_saturation,
        )
        sigma_abs = spectral_material.sigma_abs
        sigma_em = spectral_material.sigma_em
        saturation_normal = spectral_material.saturation_normal
        saturation_projected = spectral_material.saturation_projected
        valid_saturation_by_frequency = (
            spectral_material.valid_saturation_by_frequency
        )
        valid_saturation_any = spectral_material.valid_saturation_any
        valid_saturation_all = spectral_material.valid_saturation_all
        population_update_by_frequency = (
            spectral_material.population_update_by_frequency
        )
        has_finite_saturation = spectral_material.has_finite_saturation
        single_transition_fused = bool(
            self.fuse_single_transition_events
            and self.reuse_scratch_buffers
            and override_gain is None
            and len(self.levels.signal_transitions) == 1
        )
        single_transition = (
            self.levels.signal_transitions[0]
            if single_transition_fused
            else None
        )
        if self.reuse_scratch_buffers:
            result_time = self._scratch_array(
                "result_time", temporal_field.shape, complex
            )
            result_time.fill(0.0)
            slice_time_buffer = self._scratch_array(
                "slice_or_processed", temporal_field.shape, complex
            )
            # Once the temporal slice has been transformed, its buffer can
            # hold the independently processed spectrum for the same window.
            processed_buffer = slice_time_buffer
            input_by_frequency_buffer = self._scratch_array(
                "frequency_float_a", temporal_field.shape, float
            )
            output_by_frequency_buffer = self._scratch_array(
                "frequency_float_b", temporal_field.shape, float
            )
            input_weight_buffer = self._scratch_array(
                "input_weight_xy", temporal_field.shape[-2:], float
            )
            output_weight_buffer = self._scratch_array(
                "output_weight_xy", temporal_field.shape[-2:], float
            )
            event_applied_buffer = self._scratch_array(
                "event_applied_xy", temporal_field.shape[-2:], float
            )
            event_scale_buffer = self._scratch_array(
                "event_scale_xy", temporal_field.shape[-2:], float
            )
            event_threshold_buffer = self._scratch_array(
                "event_threshold_xy", temporal_field.shape[-2:], float
            )
            event_total_population_buffer = self._scratch_array(
                "event_total_population_xy",
                temporal_field.shape[-2:],
                float,
            )
            active_gain_buffer = self._scratch_array(
                "active_gain_xy", temporal_field.shape[-2:], float
            )
            if (
                single_transition_fused
                and self.single_transition_kernel_selected == "numba"
            ):
                kernel_minimum_scale_buffer = self._scratch_array(
                    "kernel_minimum_scale_xy",
                    temporal_field.shape[-2:],
                    float,
                )
                kernel_limited_count_buffer = self._scratch_array(
                    "kernel_limited_count_xy",
                    temporal_field.shape[-2:],
                    np.int64,
                )
                kernel_maximum_change_buffer = self._scratch_array(
                    "kernel_maximum_change_xy",
                    temporal_field.shape[-2:],
                    float,
                )
                kernel_maximum_unsaturated_buffer = self._scratch_array(
                    "kernel_maximum_unsaturated_xy",
                    temporal_field.shape[-2:],
                    float,
                )
                kernel_clipping_buffer = self._scratch_array(
                    "kernel_clipping_xy",
                    temporal_field.shape[-2:],
                    np.uint8,
                )
            else:
                kernel_minimum_scale_buffer = None
                kernel_limited_count_buffer = None
                kernel_maximum_change_buffer = None
                kernel_maximum_unsaturated_buffer = None
                kernel_clipping_buffer = None
            if single_transition_fused:
                np.sum(
                    population,
                    axis=0,
                    out=event_total_population_buffer,
                )
                np.copyto(
                    event_threshold_buffer,
                    event_total_population_buffer,
                )
                np.maximum(event_threshold_buffer, 1.0, out=event_threshold_buffer)
                event_threshold_buffer *= 64.0 * np.finfo(float).eps
        else:
            result_time = np.zeros_like(temporal_field)
            slice_time_buffer = None
            processed_buffer = None
            input_by_frequency_buffer = None
            output_by_frequency_buffer = None
            input_weight_buffer = None
            output_weight_buffer = None
            event_applied_buffer = None
            event_scale_buffer = None
            event_threshold_buffer = None
            event_total_population_buffer = None
            active_gain_buffer = None
            kernel_minimum_scale_buffer = None
            kernel_limited_count_buffer = None
            kernel_maximum_change_buffer = None
            kernel_maximum_unsaturated_buffer = None
            kernel_clipping_buffer = None

        active_window_mask = np.ones(slice_count, dtype=bool)
        bypassed_input_fraction = 0.0
        estimated_bypassed_output_fraction = 0.0
        maximum_local_estimated_bypassed_output_energy_fraction = 0.0
        spatial_pixel_count = int(np.prod(population.shape[-2:]))
        local_guard_supported = bool(
            spatial_pixel_count
            <= self.inactive_window_local_guard_maximum_pixels
        )
        bypass_supported = bool(
            allow_inactive_window_bypass
            and self.maximum_bypassed_output_energy_fraction > 0.0
            and not self.include_relaxation
            and local_guard_supported
            and (single_transition_fused or override_gain is not None)
        )
        if bypass_supported:
            if override_gain is not None:
                emission_exponent_xy = np.maximum(
                    override_gain * material_domain * dz,
                    0.0,
                )
            else:
                assert single_transition is not None
                emission_exponent_xy = np.maximum(
                    np.max(
                        population[single_transition.upper][None, ...]
                        * sigma_em
                        * material_domain[None, ...],
                        axis=0,
                    )
                    * dz,
                    0.0,
                )
            # Do not clip an extreme exponent: clipping would understate the
            # claimed worst-case gain.  An unrepresentable bound disables the
            # optimization below and therefore fails closed to full windows.
            with np.errstate(over="ignore", invalid="ignore"):
                conservative_gain_factor_xy = np.exp(
                    emission_exponent_xy
                )
            spatial_weight = material_domain.copy()
            if self.cell_areas_m2 is not None:
                spatial_weight *= self.cell_areas_m2
            with np.errstate(
                divide="ignore",
                invalid="ignore",
                over="ignore",
                under="ignore",
            ):
                window_input_energy_xy = np.einsum(
                    "st,tyx->syx",
                    np.square(windows),
                    temporal_energy * correction[None, ...],
                    optimize=True,
                )
            finite_spatial_energy_plan = bool(
                np.all(np.isfinite(window_input_energy_xy))
                and np.all(np.isfinite(conservative_gain_factor_xy))
            )
            if finite_spatial_energy_plan:
                total_window_input_energy_xy = np.sum(
                    window_input_energy_xy,
                    axis=0,
                )
                window_input_energy = np.sum(
                    window_input_energy_xy
                    * spatial_weight[None, ...],
                    axis=(1, 2),
                )
                total_window_input_energy = float(
                    np.sum(
                        total_window_input_energy_xy * spatial_weight
                    )
                )
                local_output_limit = (
                    self.maximum_bypassed_output_energy_fraction
                    * total_window_input_energy_xy
                )
                meaningful_local_energy = (
                    (material_domain > 0.0)
                    & (
                        total_window_input_energy_xy
                        > np.finfo(float).tiny
                    )
                )
                cumulative_input_energy_xy = np.zeros_like(
                    total_window_input_energy_xy
                )
                active_count = int(slice_count)
                for window_index in np.argsort(window_input_energy):
                    if active_count <= 1:
                        break
                    candidate_input_energy_xy = (
                        cumulative_input_energy_xy
                        + window_input_energy_xy[window_index]
                    )
                    candidate_output_energy_xy = (
                        candidate_input_energy_xy
                        * conservative_gain_factor_xy
                    )
                    if np.any(
                        candidate_output_energy_xy[
                            meaningful_local_energy
                        ]
                        > local_output_limit[meaningful_local_energy]
                        * (1.0 + 64.0 * np.finfo(float).eps)
                    ):
                        continue
                    active_window_mask[window_index] = False
                    cumulative_input_energy_xy = (
                        candidate_input_energy_xy
                    )
                    active_count -= 1
                if total_window_input_energy > np.finfo(float).tiny:
                    bypassed_input_fraction = float(
                        np.sum(
                            cumulative_input_energy_xy * spatial_weight
                        )
                        / total_window_input_energy
                    )
                    estimated_bypassed_output_fraction = float(
                        np.sum(
                            cumulative_input_energy_xy
                            * conservative_gain_factor_xy
                            * spatial_weight
                        )
                        / total_window_input_energy
                    )
                local_fraction = np.divide(
                    cumulative_input_energy_xy
                    * conservative_gain_factor_xy,
                    total_window_input_energy_xy,
                    out=np.zeros_like(total_window_input_energy_xy),
                    where=meaningful_local_energy,
                )
                maximum_local_estimated_bypassed_output_energy_fraction = (
                    float(np.max(local_fraction))
                )
            else:
                self._diagnostics[
                    "inactive_window_bypass_fallback_reason"
                ] = "nonfinite_spatial_energy_plan"
        elif (
            allow_inactive_window_bypass
            and self.maximum_bypassed_output_energy_fraction > 0.0
        ):
            if self.include_relaxation:
                bypass_fallback_reason = (
                    "disabled_when_intrapulse_relaxation_is_enabled"
                )
            elif not local_guard_supported:
                bypass_fallback_reason = (
                    "spatial_local_guard_pixel_limit_exceeded"
                )
            else:
                bypass_fallback_reason = (
                    "requires_single_fused_transition_or_gain_override"
                )
            self._diagnostics[
                "inactive_window_bypass_fallback_reason"
            ] = bypass_fallback_reason

        bypassed_window_count = int(np.count_nonzero(~active_window_mask))
        active_window_count = int(np.count_nonzero(active_window_mask))
        if bypassed_window_count:
            if batched_processed_spectra is not None:
                # The batched output was allocated with ``empty_like``.  Mark
                # bypassed slots as zero before the final batch resynthesis;
                # their unchanged input contribution is added directly below.
                batched_processed_spectra[~active_window_mask] = 0.0
            bypass_window = np.sum(
                windows[~active_window_mask],
                axis=0,
            )
            result_time += temporal_field * bypass_window[:, None, None]
            self._diagnostics["inactive_window_bypass_calls"] = int(
                self._diagnostics.get("inactive_window_bypass_calls", 0)
            ) + 1
        self._diagnostics["inactive_windows_bypassed"] = int(
            self._diagnostics.get("inactive_windows_bypassed", 0)
        ) + bypassed_window_count
        self._diagnostics["active_windows_processed"] = int(
            self._diagnostics.get("active_windows_processed", 0)
        ) + active_window_count
        self._diagnostics[
            "maximum_actual_bypassed_input_energy_fraction"
        ] = max(
            float(
                self._diagnostics.get(
                    "maximum_actual_bypassed_input_energy_fraction",
                    0.0,
                )
            ),
            bypassed_input_fraction,
        )
        self._diagnostics[
            "maximum_estimated_bypassed_output_energy_fraction"
        ] = max(
            float(
                self._diagnostics.get(
                    "maximum_estimated_bypassed_output_energy_fraction",
                    0.0,
                )
            ),
            estimated_bypassed_output_fraction,
        )
        self._diagnostics[
            "maximum_local_estimated_bypassed_output_energy_fraction"
        ] = max(
            float(
                self._diagnostics.get(
                    "maximum_local_estimated_bypassed_output_energy_fraction",
                    0.0,
                )
            ),
            maximum_local_estimated_bypassed_output_energy_fraction,
        )

        frequency_orders = (
            (self._backward_frequency_order, self._forward_frequency_order)
            if reverse_spectral_order
            else (self._forward_frequency_order, self._backward_frequency_order)
        )

        trace_time: list[float] = []
        trace_gain: list[float] = []
        trace_reservoir_before: list[float] = []
        trace_reservoir_after: list[float] = []
        trace_input: list[float] = []
        maximum_depletion = 0.0
        effective_energy_change = 0.0

        for window_index in np.flatnonzero(active_window_mask):
            window = windows[window_index]
            center_index = center_indices[window_index]
            if batched_slice_spectra is not None:
                slice_spectrum = batched_slice_spectra[window_index]
            elif self.reuse_scratch_buffers:
                assert slice_time_buffer is not None
                np.multiply(
                    temporal_field,
                    window[:, None, None],
                    out=slice_time_buffer,
                )
                slice_time = slice_time_buffer
                slice_spectrum = self.pulse.time_to_spectrum(
                    slice_time,
                    axis=0,
                )
            else:
                slice_time = temporal_field * window[:, None, None]
                slice_spectrum = self.pulse.time_to_spectrum(
                    slice_time,
                    axis=0,
                )
            if self.reuse_scratch_buffers:
                assert input_by_frequency_buffer is not None
                assert output_by_frequency_buffer is not None
                assert input_weight_buffer is not None
                assert output_weight_buffer is not None
                if batched_processed_spectra is not None:
                    processed = batched_processed_spectra[window_index]
                else:
                    assert processed_buffer is not None
                    processed = processed_buffer
                np.copyto(processed, slice_spectrum)
                self._magnitude_squared(
                    slice_spectrum,
                    input_by_frequency_buffer,
                )
                np.sum(
                    input_by_frequency_buffer,
                    axis=0,
                    out=input_weight_buffer,
                )
                np.multiply(
                    input_weight_buffer,
                    correction,
                    out=input_weight_buffer,
                )
                input_weight_xy = input_weight_buffer
                output_weight_buffer.fill(0.0)
                output_weight_xy = output_weight_buffer
                output_by_frequency_buffer.fill(0.0)
                output_by_frequency = output_by_frequency_buffer
            else:
                if batched_processed_spectra is not None:
                    processed = batched_processed_spectra[window_index]
                    np.copyto(processed, slice_spectrum)
                else:
                    processed = slice_spectrum.copy()
                input_weight_xy = (
                    np.sum(np.abs(slice_spectrum) ** 2, axis=0)
                    * correction
                )
                output_weight_xy = np.zeros_like(total_energy_xy)
                output_by_frequency = np.zeros_like(
                    np.abs(slice_spectrum) ** 2
                )
            reservoir_before = self._weighted_reservoir(population, override_gain, input_weight_xy)
            signal_upper_indices = self._signal_upper_indices
            upper_before = np.take(
                population, signal_upper_indices, axis=0
            ).copy()
            signal_population_before = np.take(
                population, self._signal_population_indices, axis=0
            ).copy()
            total_population_before = np.sum(population, axis=0)
            override_before = None if override_gain is None else override_gain.copy()

            window_frequency_orders = frequency_orders
            broadband_configured = self.spectral_coupling_requested in {
                "broadband_two_stage",
                "auto",
            }
            broadband_requested = bool(
                allow_broadband and broadband_configured
            )
            broadband_supported = bool(
                single_transition_fused
                and use_saturation
                and has_finite_saturation
            )
            if broadband_requested and broadband_supported:
                assert single_transition is not None
                assert input_by_frequency_buffer is not None
                assert output_by_frequency_buffer is not None
                assert event_applied_buffer is not None
                assert event_scale_buffer is not None
                assert event_threshold_buffer is not None
                assert event_total_population_buffer is not None
                self._run_broadband_two_stage(
                    input_fluence_by_frequency=(
                        input_by_frequency_buffer
                    ),
                    correction=correction,
                    population=population,
                    transition=single_transition,
                    sigma_abs=sigma_abs,
                    sigma_em=sigma_em,
                    saturation=saturation_projected,
                    valid_saturation=valid_saturation_by_frequency,
                    update_frequency=population_update_by_frequency,
                    material_domain=material_domain,
                    photon_energies=photon_energies,
                    dz=dz,
                    cell_length_m=cell_length_m,
                    output_by_frequency=output_by_frequency_buffer,
                    output_weight=output_weight_buffer,
                    event_applied=event_applied_buffer,
                    event_scale=event_scale_buffer,
                    total_population=event_total_population_buffer,
                    meaningful_threshold=event_threshold_buffer,
                )
                self._diagnostics[
                    "single_transition_fused_path_used"
                ] = True
                self._diagnostics[
                    "single_transition_fused_event_calls"
                ] = int(
                    self._diagnostics[
                        "single_transition_fused_event_calls"
                    ]
                ) + 2
                window_frequency_orders = ()
            elif broadband_requested:
                self._diagnostics["broadband_fallback_windows"] = int(
                    self._diagnostics.get("broadband_fallback_windows", 0)
                ) + 1
                if not single_transition_fused:
                    fallback_reason = (
                        "requires_one_physical_signal_transition_and_scratch_reuse"
                    )
                elif not use_saturation:
                    fallback_reason = "gain_saturation_disabled"
                else:
                    fallback_reason = "no_finite_signal_saturation"
                self._diagnostics[
                    "broadband_fallback_reason"
                ] = fallback_reason
            elif broadband_configured:
                self._diagnostics["spectral_coupling_selected"] = (
                    "ordered_sweeps"
                )
                self._diagnostics["spectral_coupling"] = (
                    "symmetric_forward_reverse_half_sweeps"
                )

            if (
                window_frequency_orders
                and
                single_transition_fused
                and self.single_transition_kernel_selected == "numba"
            ):
                assert single_transition is not None
                assert input_by_frequency_buffer is not None
                assert output_by_frequency_buffer is not None
                assert event_total_population_buffer is not None
                assert event_threshold_buffer is not None
                assert kernel_minimum_scale_buffer is not None
                assert kernel_limited_count_buffer is not None
                assert kernel_maximum_change_buffer is not None
                assert kernel_maximum_unsaturated_buffer is not None
                assert kernel_clipping_buffer is not None
                numba_cold_start = not numba_kernel_compiled_in_process()
                with profile_operation(
                    "signal.coherent_single_transition_sweep",
                    array=input_by_frequency_buffer,
                ):
                    run_single_transition_sweep_numba(
                        input_fluence_by_frequency=(
                            input_by_frequency_buffer
                        ),
                        correction=correction,
                        population_upper=population[
                            single_transition.upper
                        ],
                        population_lower=population[
                            single_transition.lower
                        ],
                        sigma_abs=sigma_abs,
                        sigma_em=sigma_em,
                        saturation=saturation_projected,
                        material_domain=material_domain,
                        photon_energies=photon_energies,
                        update_frequency=population_update_by_frequency,
                        total_population=event_total_population_buffer,
                        meaningful_threshold=event_threshold_buffer,
                        dz=dz,
                        cell_length_m=cell_length_m,
                        reverse_spectral_order=reverse_spectral_order,
                        use_saturation=use_saturation,
                        output_by_frequency=output_by_frequency_buffer,
                        output_weight=output_weight_buffer,
                        minimum_scale_by_pixel=(
                            kernel_minimum_scale_buffer
                        ),
                        limited_count_by_pixel=(
                            kernel_limited_count_buffer
                        ),
                        maximum_change_by_pixel=(
                            kernel_maximum_change_buffer
                        ),
                        maximum_unsaturated_exponent_by_pixel=(
                            kernel_maximum_unsaturated_buffer
                        ),
                        clipping_by_pixel=kernel_clipping_buffer,
                    )
                if numba_cold_start:
                    self._diagnostics[
                        "single_transition_numba_cold_start_calls"
                    ] = int(
                        self._diagnostics[
                            "single_transition_numba_cold_start_calls"
                        ]
                    ) + 1
                event_calls = (
                    2 * int(np.count_nonzero(population_update_by_frequency))
                    if use_saturation
                    else 0
                )
                self.levels.record_single_signal_event_summary(
                    calls=event_calls,
                    limited_cell_updates=int(
                        np.sum(kernel_limited_count_buffer)
                    ),
                    minimum_limiter_scale=float(
                        np.min(kernel_minimum_scale_buffer)
                    ),
                    maximum_population_change_fraction=float(
                        np.max(kernel_maximum_change_buffer)
                    ),
                )
                finite_unsaturated = kernel_maximum_unsaturated_buffer[
                    np.isfinite(kernel_maximum_unsaturated_buffer)
                ]
                if finite_unsaturated.size:
                    local_maximum = float(np.max(finite_unsaturated))
                    previous_maximum = self._diagnostics.get(
                        "maximum_unsaturated_gain_exponent"
                    )
                    self._diagnostics[
                        "maximum_unsaturated_gain_exponent"
                    ] = (
                        local_maximum
                        if previous_maximum is None
                        else max(float(previous_maximum), local_maximum)
                    )
                self._diagnostics[
                    "positive_gain_exponent_clipping_active"
                ] = bool(
                    self._diagnostics.get(
                        "positive_gain_exponent_clipping_active", False
                    )
                    or np.any(kernel_clipping_buffer)
                )
                self._diagnostics[
                    "single_transition_fused_path_used"
                ] = True
                self._diagnostics[
                    "single_transition_fused_event_calls"
                ] = int(
                    self._diagnostics[
                        "single_transition_fused_event_calls"
                    ]
                ) + event_calls
                self._diagnostics["single_transition_numba_windows"] = int(
                    self._diagnostics["single_transition_numba_windows"]
                ) + 1
                window_frequency_orders = ()

            for frequency_order in window_frequency_orders:
                for iw in frequency_order:
                    actual_fluence = (
                        input_by_frequency_buffer[iw]
                        if self.reuse_scratch_buffers
                        else np.abs(slice_spectrum[iw]) ** 2
                    )
                    effective_fluence = 0.5 * actual_fluence * correction
                    if single_transition_fused:
                        assert single_transition is not None
                        assert active_gain_buffer is not None
                        assert event_scale_buffer is not None
                        np.multiply(
                            population[single_transition.upper],
                            sigma_em[iw],
                            out=active_gain_buffer,
                        )
                        np.multiply(
                            population[single_transition.lower],
                            sigma_abs[iw],
                            out=event_scale_buffer,
                        )
                        np.subtract(
                            active_gain_buffer,
                            event_scale_buffer,
                            out=active_gain_buffer,
                        )
                        active_gain = active_gain_buffer
                    elif override_gain is None:
                        active_gain = self.levels.signal_gain_m_inv(
                            population,
                            wavelength_m=float(wavelengths[iw]),
                            z_index=z_index,
                        )
                    else:
                        active_gain = override_gain
                    if single_transition_fused:
                        np.multiply(
                            active_gain,
                            material_domain,
                            out=active_gain,
                        )
                        np.multiply(active_gain, dz, out=active_gain)
                        exponent = active_gain
                    else:
                        exponent = active_gain * material_domain * dz
                    valid_saturation = valid_saturation_by_frequency[iw]
                    fallback_exponent = (
                        np.where(valid_saturation, -np.inf, exponent)
                        if use_saturation and valid_saturation_any[iw]
                        else exponent
                    )
                    local_maximum = float(np.max(fallback_exponent))
                    if np.isfinite(local_maximum):
                        previous_maximum = self._diagnostics.get(
                            "maximum_unsaturated_gain_exponent"
                        )
                        self._diagnostics["maximum_unsaturated_gain_exponent"] = (
                            local_maximum
                            if previous_maximum is None
                            else max(float(previous_maximum), local_maximum)
                        )
                    self._diagnostics["positive_gain_exponent_clipping_active"] = bool(
                        self._diagnostics.get(
                            "positive_gain_exponent_clipping_active", False
                        )
                        or np.any(fallback_exponent > 50.0)
                    )
                    if use_saturation and valid_saturation_all[iw]:
                        output_effective = frantz_nodvik(
                            effective_fluence,
                            exponent,
                            saturation_projected[iw],
                        )
                    elif use_saturation and valid_saturation_any[iw]:
                        saturated = frantz_nodvik(
                            effective_fluence,
                            exponent,
                            np.where(
                                valid_saturation,
                                saturation_projected[iw],
                                1.0,
                            ),
                        )
                        output_effective = np.where(
                            valid_saturation,
                            saturated,
                            effective_fluence
                            * np.exp(np.minimum(exponent, 50.0)),
                        )
                    else:
                        output_effective = effective_fluence * np.exp(
                            np.minimum(exponent, 50.0)
                        )

                    can_update_population = (
                        override_gain is None
                        and use_saturation
                        and population_update_by_frequency[iw]
                    )
                    if can_update_population:
                        event_energy_volume = (
                            photon_energies[iw] * cell_length_m
                        )
                        if single_transition_fused:
                            assert event_applied_buffer is not None
                            assert event_scale_buffer is not None
                            assert event_threshold_buffer is not None
                            assert event_total_population_buffer is not None
                            np.subtract(
                                output_effective,
                                effective_fluence,
                                out=output_effective,
                            )
                            output_effective /= event_energy_volume
                            limited = (
                                self.levels.apply_single_signal_photon_events_inplace(
                                    population,
                                    output_effective,
                                    applied_events_out=event_applied_buffer,
                                    scale_work=event_scale_buffer,
                                    total_population_m3=(
                                        event_total_population_buffer
                                    ),
                                    meaningful_event_threshold_m3=(
                                        event_threshold_buffer
                                    ),
                                )
                            )
                            np.multiply(
                                limited,
                                event_energy_volume,
                                out=output_effective,
                            )
                            np.add(
                                output_effective,
                                effective_fluence,
                                out=output_effective,
                            )
                            self._diagnostics[
                                "single_transition_fused_path_used"
                            ] = True
                            self._diagnostics[
                                "single_transition_fused_event_calls"
                            ] = int(
                                self._diagnostics[
                                    "single_transition_fused_event_calls"
                                ]
                            ) + 1
                        else:
                            requested_events = (
                                output_effective - effective_fluence
                            ) / event_energy_volume
                            updated, event_stats = (
                                self.levels.apply_signal_photon_events(
                                    population,
                                    requested_events,
                                    float(wavelengths[iw]),
                                    z_index=z_index,
                                    input_photon_density_m3=(
                                        effective_fluence
                                        / event_energy_volume
                                    ),
                                    interaction_length_m=dz,
                                )
                            )
                            population[:] = updated
                            limited = np.asarray(
                                event_stats["applied_photon_events_m3"],
                                dtype=float,
                            )
                            output_effective = (
                                effective_fluence
                                + limited * event_energy_volume
                            )
                    elif (
                        override_gain is not None
                        and use_saturation
                        and np.any(np.isfinite(saturation_normal[iw]))
                        and np.any(saturation_normal[iw] > 0.0)
                    ):
                        override_gain -= (output_effective - effective_fluence) / (
                            saturation_normal[iw] * cell_length_m
                        )
                        if not np.all(np.isfinite(override_gain)):
                            raise FloatingPointError(
                                "coherent gain-override reservoir became non-finite"
                            )
                        configured_override = float(
                            self.levels_cfg.get("gain_override_m_inv", 0.0)
                        )
                        if configured_override >= 0.0:
                            override_gain[:] = np.maximum(override_gain, 0.0)
                        else:
                            override_gain[:] = np.minimum(override_gain, 0.0)

                    output_effective = np.maximum(output_effective, 0.0)
                    output_by_frequency[iw] += output_effective
                    output_weight_xy += output_effective

            if self.reuse_scratch_buffers:
                assert input_by_frequency_buffer is not None
                np.multiply(
                    input_by_frequency_buffer,
                    correction[None, ...],
                    out=input_by_frequency_buffer,
                )
                np.divide(
                    output_by_frequency,
                    input_by_frequency_buffer,
                    out=output_by_frequency,
                    where=input_by_frequency_buffer > np.finfo(float).tiny,
                )
                output_by_frequency[
                    input_by_frequency_buffer <= np.finfo(float).tiny
                ] = 0.0
                np.maximum(output_by_frequency, 0.0, out=output_by_frequency)
                np.sqrt(output_by_frequency, out=output_by_frequency)
                processed *= output_by_frequency
            else:
                effective_input_by_frequency = (
                    np.abs(slice_spectrum) ** 2 * correction[None, ...]
                )
                ratio = np.divide(
                    output_by_frequency,
                    effective_input_by_frequency,
                    out=np.zeros_like(output_by_frequency),
                    where=effective_input_by_frequency
                    > np.finfo(float).tiny,
                )
                processed *= np.sqrt(np.maximum(ratio, 0.0))

            if batched_processed_spectra is None:
                processed_time = self.pulse.spectrum_to_time(
                    processed,
                    axis=0,
                )
                result_time += processed_time
            effective_energy_change += self._integrate_xy(output_weight_xy - input_weight_xy)
            slice_input = self._integrate_xy(input_weight_xy)
            slice_output = self._integrate_xy(output_weight_xy)
            trace_time.append(
                float(
                    np.interp(
                        center_index,
                        self._sample_indices,
                        self.pulse.time_s,
                    )
                )
            )
            trace_input.append(slice_input)
            trace_gain.append(slice_output / max(slice_input, np.finfo(float).tiny))
            trace_reservoir_before.append(reservoir_before)
            trace_reservoir_after.append(self._weighted_reservoir(population, override_gain, input_weight_xy))

            if override_gain is None:
                if self.adaptive_depletion_metric == "legacy_upper_relative":
                    upper_after = np.take(
                        population, signal_upper_indices, axis=0
                    )
                    depleted = np.maximum(
                        upper_before - upper_after, 0.0
                    )
                    relative = np.divide(
                        depleted,
                        upper_before,
                        out=np.zeros_like(depleted),
                        where=upper_before > np.finfo(float).tiny,
                    )
                else:
                    signal_population_after = np.take(
                        population,
                        self._signal_population_indices,
                        axis=0,
                    )
                    maximum_level_change = np.max(
                        np.abs(
                            signal_population_after
                            - signal_population_before
                        ),
                        axis=0,
                    )
                    relative = np.divide(
                        maximum_level_change,
                        total_population_before,
                        out=np.zeros_like(maximum_level_change),
                        where=total_population_before
                        > np.finfo(float).tiny,
                    )
            else:
                assert override_before is not None
                reservoir_change = np.abs(override_before - override_gain)
                relative = np.divide(
                    reservoir_change,
                    np.maximum(np.abs(override_before), 1e-30),
                    out=np.zeros_like(reservoir_change),
                )
            maximum_depletion = max(maximum_depletion, float(np.max(relative)))

            if self.include_relaxation:
                next_index = len(trace_time)
                # Every window plan owns the full declared physical support,
                # including the interval after its final exposure center.
                next_time = (
                    float(
                        np.interp(
                            center_indices[next_index],
                            self._sample_indices,
                            self.pulse.time_s,
                        )
                    )
                    if next_index < len(center_indices)
                    else relaxation_end_s
                )
                current_physical_time = float(
                    np.clip(
                        trace_time[-1],
                        relaxation_start_s,
                        relaxation_end_s,
                    )
                )
                next_physical_time = float(
                    np.clip(
                        next_time,
                        relaxation_start_s,
                        relaxation_end_s,
                    )
                )
                dt = (
                    max(next_physical_time - current_physical_time, 0.0)
                    * relaxation_time_scale
                )
                if dt > 0.0:
                    population[:] = self.levels.advance_dark_implicit(
                        population,
                        dt,
                        z_index,
                    )
                    if single_transition_fused:
                        assert event_threshold_buffer is not None
                        assert event_total_population_buffer is not None
                        np.sum(
                            population,
                            axis=0,
                            out=event_total_population_buffer,
                        )
                        np.copyto(
                            event_threshold_buffer,
                            event_total_population_buffer,
                        )
                        np.maximum(
                            event_threshold_buffer,
                            1.0,
                            out=event_threshold_buffer,
                        )
                        event_threshold_buffer *= (
                            64.0 * np.finfo(float).eps
                        )

        if batched_processed_spectra is not None:
            batched_result_time = self.pulse.spectrum_to_time(
                batched_processed_spectra,
                axis=1,
            )
            # Bypassed windows were already added to ``result_time`` above.
            # The batched array contains zeros for those slots, so accumulate
            # the processed windows instead of erasing the bypass contribution.
            result_time += np.sum(batched_result_time, axis=0)

        if self.reuse_scratch_buffers:
            assert input_by_frequency_buffer is not None
            assert output_by_frequency_buffer is not None
            self._magnitude_squared(result_time, input_by_frequency_buffer)
            output_before_loss_fluence = np.sum(
                input_by_frequency_buffer, axis=0
            )
            if passive_half_amplitude is not None:
                result_time *= passive_half_amplitude[None, :, :]
            result_time_before_second_half = None
        else:
            assert passive_half_amplitude is not None
            result_time_before_second_half = result_time.copy()
            result_time *= passive_half_amplitude[None, :, :]
        output_field = self.pulse.time_to_spectrum(result_time, axis=0)
        if self.reuse_scratch_buffers:
            assert input_by_frequency_buffer is not None
            self._magnitude_squared(
                temporal_field, input_by_frequency_buffer
            )
            active_input_fluence = np.sum(
                input_by_frequency_buffer, axis=0
            )
        else:
            active_input_fluence = np.sum(
                np.abs(temporal_field) ** 2, axis=0
            )
        input_energy = self._integrate_xy(active_input_fluence)
        if result_time_before_second_half is not None:
            output_before_loss_fluence = np.sum(
                np.abs(result_time_before_second_half) ** 2, axis=0
            )
        output_before_loss = self._integrate_xy(output_before_loss_fluence)
        if original_input_fluence is None:
            original_input_fluence = np.sum(
                np.abs(temporal_field_before_passive) ** 2, axis=0
            )
        # Account directly from the two passive half-step operators.  This is
        # both more stable than subtracting nearly equal fluences and keeps
        # the optimized and allocation-heavy paths algebraically identical.
        passive_loss_fluence = (
            np.maximum(original_input_fluence, 0.0)
            + np.maximum(output_before_loss_fluence, 0.0)
        ) * passive_half_power_loss_fraction
        expected_output = max(input_energy + effective_energy_change, np.finfo(float).tiny)
        balance_error = (
            abs(output_before_loss - expected_output) / expected_output if has_finite_saturation else 0.0
        )
        center_spacing_samples = np.diff(center_indices)
        spacing_samples = (
            0.0
            if center_spacing_samples.size == 0
            else float(np.median(center_spacing_samples))
        )
        minimum_spacing_samples = (
            0.0
            if center_spacing_samples.size == 0
            else float(np.min(center_spacing_samples))
        )
        maximum_spacing_samples = (
            0.0
            if center_spacing_samples.size == 0
            else float(np.max(center_spacing_samples))
        )
        trace = {
            "slice_count": int(slice_count),
            "active_window_count": active_window_count,
            "bypassed_window_count": bypassed_window_count,
            "bypassed_input_energy_fraction": bypassed_input_fraction,
            "estimated_bypassed_output_energy_fraction": (
                estimated_bypassed_output_fraction
            ),
            "maximum_local_estimated_bypassed_output_energy_fraction": (
                maximum_local_estimated_bypassed_output_energy_fraction
            ),
            "time_s": trace_time,
            "slice_energy_gain": trace_gain,
            "reservoir_before": trace_reservoir_before,
            "reservoir_after": trace_reservoir_after,
            "slice_input_weight": trace_input,
            "reservoir_quantity": "effective_gain_fraction" if override_gain is not None else "upper_level_fraction",
            "soft_window_full_width_s": 2.0 * spacing_samples * self.pulse.dt_s,
            "soft_window_minimum_center_spacing_s": (
                minimum_spacing_samples * self.pulse.dt_s
            ),
            "soft_window_maximum_center_spacing_s": (
                maximum_spacing_samples * self.pulse.dt_s
            ),
            "spectral_resolution_estimate_Hz": 1.0
            / max(
                2.0 * maximum_spacing_samples * self.pulse.dt_s,
                np.finfo(float).tiny,
            ),
            "physical_relaxation_window_s": physical_relaxation_window_s,
            "relaxation_time_scale": relaxation_time_scale,
        }
        return _StepResult(
            output_field,
            population,
            override_gain,
            maximum_depletion,
            float(balance_error),
            float(np.mean(correction)),
            trace,
            np.maximum(passive_loss_fluence, 0.0),
        )

    def physical_relaxation_support_s(self) -> tuple[float, float]:
        """Return the common physical relaxation window for every gain cell.

        The FFT time span contains numerical zero padding whose length changes
        with sampling choices.  The input pulse energy quantiles define one
        physical interval instead.  Using that same interval at every z/x/y
        cell also gives the top-level periodic map an exact amount of level-
        kinetics time to subtract from its external Pump/dark evolution.
        """

        time_mass = np.asarray(self.pulse.temporal_energy_J, dtype=float)
        time_mass = np.maximum(time_mass, 0.0)
        total = float(np.sum(time_mass))
        if not np.isfinite(total) or total <= np.finfo(float).tiny:
            return float(self.pulse.time_s[0]), float(self.pulse.time_s[-1])
        centers = (np.cumsum(time_mass) - 0.5 * time_mass) / total
        tail = 0.5 * (1.0 - self.relaxation_energy_fraction)
        start = float(
            np.interp(
                tail,
                centers,
                self.pulse.time_s,
                left=self.pulse.time_s[0],
                right=self.pulse.time_s[-1],
            )
        )
        end = float(
            np.interp(
                1.0 - tail,
                centers,
                self.pulse.time_s,
                left=self.pulse.time_s[0],
                right=self.pulse.time_s[-1],
            )
        )
        return min(start, end), max(start, end)

    def _weighted_reservoir(
        self,
        population: np.ndarray,
        override_gain: np.ndarray | None,
        weight: np.ndarray,
    ) -> float:
        positive_weight = np.maximum(weight, 0.0)
        if self.cell_areas_m2 is not None:
            if positive_weight.shape != self.cell_areas_m2.shape:
                raise ValueError("coherent field shape does not match cell_areas_m2")
            positive_weight = positive_weight * self.cell_areas_m2
        denominator = float(np.sum(positive_weight))
        if denominator <= np.finfo(float).tiny:
            positive_weight = (
                self.cell_areas_m2.copy()
                if self.cell_areas_m2 is not None
                else np.ones_like(weight)
            )
            denominator = float(np.sum(positive_weight))
        if override_gain is not None:
            reference = float(self.levels_cfg.get("gain_override_m_inv") or 0.0)
            if abs(reference) <= np.finfo(float).tiny:
                return 0.0
            local = override_gain / reference
        else:
            total = np.sum(population, axis=0)
            upper = self.levels.aggregate_signal_upper_population(population)
            local = np.divide(
                upper,
                total,
                out=np.zeros_like(total),
                where=total > np.finfo(float).tiny,
            )
        return float(np.sum(positive_weight * local) / denominator)

    def _integrate_xy(self, values: np.ndarray) -> float:
        array = np.asarray(values, dtype=float)
        if self.cell_areas_m2 is None:
            return float(np.sum(array))
        if array.shape != self.cell_areas_m2.shape:
            raise ValueError("coherent field shape does not match cell_areas_m2")
        return float(np.sum(array * self.cell_areas_m2))
