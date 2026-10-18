"""Coherent CPA adapter and fail-safe wrapper for the spectral propagator."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .coherent import (
    build_coherent_pulse,
    intensity_fwhm,
)
from .coherent_step import (
    CoherentGainUpdatePlan,
    CoherentLocalOperatorSettings,
    CoherentLocalStepPlan,
    choose_local_substeps,
    choose_multirate_gain_updates,
    nonlinear_gain_coupling_blockers,
)
from .dispersion import (
    material_dispersion_phase,
    material_refractive_index,
    phase_from_coefficients,
)
from .errors import OPTIONAL_BACKEND_FAILURES, OPTIONAL_MODEL_FAILURES
from .profiles import DiscreteProfile
from .nonlinear import (
    apply_incoherent_strong_field_step,
    apply_strong_field_step,
    multiphoton_transmission_fraction,
)
from .caustic import caustic_positions, scan_spectral_caustic
from .optics import OpticalTrace
from .spectral_signal import SpectralSignalPropagator
from .seed_spatio_spectral import seed_config_at_wavelength
from .signal import SignalPropagationResult
from .spatial_modes import incoherent_hg_modes


class _TransientGainFallbackRequested(RuntimeError):
    """Request a full coherent-pass restart after an optional local failure."""

    def __init__(self, reason: str, diagnostics: dict[str, Any]):
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics


def kerr_self_focusing_screen(
    *,
    wavelength_m: float,
    refractive_index: float,
    n2_m2_W: float,
    peak_power_W: float | None,
) -> dict[str, Any]:
    """Return a deliberately labelled bulk-Kerr critical-power screen.

    The common Gaussian/Townes estimate
    ``Pcr = 3.77 lambda0^2 / (8 pi n0 n2)`` is useful for detecting the
    threshold-like regime in which a paraxial Kerr calculation becomes very
    sensitive to ``n2``, focus, gain, and thermal lensing.  It is not a
    collapse proof for a finite, externally focused, gain-guided crystal.
    """

    wavelength = float(wavelength_m)
    index = float(refractive_index)
    n2 = float(n2_m2_W)
    peak_power = None if peak_power_W is None else float(peak_power_W)
    base = {
        "enabled": bool(n2 > 0.0),
        "method": "bulk_gaussian_townes_critical_power_screen",
        "formula": "Pcr = 3.77*lambda0^2/(8*pi*n0*n2)",
        "wavelength_m": wavelength,
        "linear_refractive_index": index,
        "n2_m2_W": n2,
        "peak_power_W": peak_power,
        "scope": (
            "Screening estimate only; finite crystal length, imposed focusing, "
            "gain guiding, thermal lensing, dispersion, and loss determine the "
            "actual propagated field."
        ),
    }
    if n2 <= 0.0:
        return {**base, "regime": "not_self_focusing"}
    if wavelength <= 0.0 or index <= 0.0:
        return {
            **base,
            "enabled": False,
            "regime": "invalid_material_or_wavelength",
        }
    critical_power = 3.77 * wavelength**2 / (8.0 * math.pi * index * n2)
    ratio = (
        None
        if peak_power is None or not math.isfinite(peak_power) or peak_power < 0.0
        else peak_power / critical_power
    )
    return {
        **base,
        "critical_power_W": critical_power,
        "peak_to_critical_power_ratio": ratio,
        "regime": (
            "not_evaluated"
            if ratio is None
            else "above_screening_threshold"
            if ratio >= 1.0
            else "near_screening_threshold"
            if ratio >= 0.5
            else "below_screening_threshold"
        ),
    }


class CoherentSignalPropagator(SpectralSignalPropagator):
    """Propagate a uniform-frequency complex spectrum through the 3D model.

    The default gain path retains the fast whole-fluence approximation.  An
    optional, independently guarded soft-window operator resolves intrapulse
    front-to-tail depletion while retaining wavelength-dependent coherent gain.
    """

    def _defer_transient_gain_initialization(self) -> bool:
        """Allow reduced-coordinate subclasses to finish their geometry first."""

        return False

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.pulse = build_coherent_pulse(self.cfg)
        self.coherent_fidelity_mode = str(
            self.cfg.get("coherent", {}).get("fidelity_mode", "configured")
        )
        energy = float(np.sum(self.pulse.spectral_energy_J))
        if energy <= 0.0:
            raise ValueError("coherent_cpa requires positive seed pulse energy")
        weights = self.pulse.spectral_energy_J / energy
        self.spectrum = DiscreteProfile(self.pulse.wavelength_m.copy(), weights)
        temporal_weights = self.pulse.temporal_energy_J / energy
        self.temporal = DiscreteProfile(self.pulse.time_s.copy(), temporal_weights)
        self.bench_warnings.extend(self.pulse.warnings)
        self.bench_warnings.append(
            "CPA reported spectral phase uses a reference-mode projection. "
            "Temporal energy and compressor power are integrated from the full "
            "complex field unless output diagnostics are explicitly suppressed; "
            "inspect projection efficiency when interpreting a single spectral phase curve."
        )
        self.reference_modes: np.ndarray | None = None
        self._reference_modes_source: np.ndarray | None = None
        self.maximum_b_integral_rad = 0.0
        self.maximum_single_step_nonlinear_phase_rad = 0.0
        self.maximum_nonlinear_step_peak_power_W = 0.0
        self.maximum_local_nonlinear_intensity_W_m2 = 0.0
        self.nonlinear_phase_sampling_diagnostics: dict[str, Any] = {}
        self.cycle_b_integral_diagnostics: dict[str, Any] = {}
        self.per_pass_b_integral_diagnostics: list[dict[str, Any]] = []
        self._cumulative_b_integral_rad: np.ndarray | None = None
        self.nonlinear_loss_energy_J_m3 = np.zeros(self.grid.shape, dtype=float)
        self.multiphoton_loss_energy_J_m3 = np.zeros(self.grid.shape, dtype=float)
        self.free_carrier_loss_energy_J_m3 = np.zeros(self.grid.shape, dtype=float)
        self.maximum_peak_carrier_density_m3 = 0.0
        self.final_free_carrier_density_m3 = np.zeros(self.grid.shape, dtype=float)
        self._active_free_carrier_z_index: int | None = None
        self._active_free_carrier_initial_m3: np.ndarray | None = None
        self._pending_free_carrier_density_m3: np.ndarray | None = None
        self._pending_free_carrier_cells: np.ndarray | None = None
        self._initial_free_carrier_density_m3: np.ndarray | None = None
        self.maximum_strong_field_fractional_loss = 0.0
        self.local_operator_settings = CoherentLocalOperatorSettings.from_seed_config(
            self.cfg
        )
        if self.local_operator_settings.spatial_population_update == "independent_groups":
            if (self.coherent_fidelity_mode not in {"hybrid_whole_fluence", "axisymmetric_kerr"}
                    or self.cfg.get("coherent", {}).get("transient_gain", {}).get("enabled", False)
                    or self.local_operator_settings.scheme != "symmetric_adaptive"):
                raise ValueError("independent_groups requires symmetric whole-fluence gain")
        self.maximum_local_operator_substeps = 1
        self.maximum_recommended_local_operator_substeps = 1
        self.maximum_estimated_gain_exponent_per_cell = 0.0
        self.maximum_estimated_nonlinear_phase_rad_per_cell = 0.0
        self.maximum_estimated_dispersion_phase_rad_per_cell = 0.0
        self.maximum_estimated_thermal_phase_rad_per_cell = 0.0
        self.maximum_estimated_fractional_loss_per_cell = 0.0
        self.local_operator_substep_cap_active = False
        self.sequential_coupling_accuracy_warning = False
        self._current_crystal_cell_substeps = 1
        self._current_gain_update_schedule = (1,)
        self._current_gain_group_size = 1
        self._pending_gain_cell = None
        self.multirate_gain_cells_active = 0
        self.multirate_gain_optical_substeps = 0
        self.multirate_gain_updates = 0
        self.multirate_gain_maximum_group_size = 1
        self.multirate_gain_schedule_reasons: dict[str, int] = {}
        self.maximum_gain_updates_used_per_cell = 0
        self.maximum_estimated_gain_exponent_per_update = 0.0
        self._material_dispersion_factor_cache: dict[
            tuple[float, int], np.ndarray | None
        ] = {}
        self._material_dispersion_cache_hits = 0
        self._material_dispersion_cache_misses = 0
        transient_cfg = self.cfg.get("coherent", {}).get("transient_gain", {})
        self.transient_gain_cfg = transient_cfg
        self.transient_gain_failure_policy = str(transient_cfg.get("failure_policy", "error"))
        self.transient_gain: Any | None = None
        self.transient_gain_status: dict[str, Any] = {
            "enabled": bool(transient_cfg.get("enabled", False)),
            "status": "disabled",
            "method": "whole_fluence",
        }
        self._persistent_coherent_warnings: list[str] = []
        self.transient_gain_warnings: list[str] = []
        if bool(transient_cfg.get("enabled", False)) and not (
            self._defer_transient_gain_initialization()
        ):
            try:
                self.transient_gain = self._new_transient_gain()
                self.transient_gain_status = self.transient_gain.diagnostics()
            except OPTIONAL_BACKEND_FAILURES as exc:
                if self.transient_gain_failure_policy == "error":
                    raise
                reason = f"{type(exc).__name__}: {exc}"
                self.transient_gain_status = {
                    "enabled": True,
                    "status": "fallback_whole_fluence",
                    "method": "whole_fluence",
                    "reason": reason,
                    "fallback_origin": "initialization",
                    "fallback_stage": "constructor_probe",
                }
                self._add_persistent_warning(
                    f"Coherent transient gain could not start ({reason}); coherent CPA continued with whole-fluence saturation."
                )
        if self.coherent_fidelity_mode == "hybrid_whole_fluence":
            self._add_persistent_warning(
                "Coherent Kerr hybrid profile is active: the full complex "
                "E(f,y,x) field, diffraction, dispersion, interference, and "
                "local time-domain Kerr phase are retained, while intrapulse "
                "gain depletion uses a whole-fluence FN closure with guarded "
                "multirate gain updates. Compare a small case with transient "
                "soft-window gain before using "
                "front-to-tail or fine gain-shaped spectral details quantitatively."
            )

    def _new_transient_gain(self) -> Any:
        # Lazy loading makes this optional refinement independent from the rest
        # of coherent CPA and from every non-coherent model.
        from .coherent_gain import CoherentTransientGain

        return CoherentTransientGain(
            self.pulse,
            self.levels,
            self.levels_cfg,
            self.material,
            self.physics,
            self.transient_gain_cfg,
            self.grid.cell_areas,
            self.crystal_mask,
        )

    def resolved_intrapulse_population_time_s(self) -> float:
        """Return level-kinetics time advanced by one coherent pass."""

        engine = self.transient_gain
        if engine is None or not bool(getattr(engine, "include_relaxation", False)):
            return 0.0
        start_s, end_s = engine.physical_relaxation_support_s()
        return max(float(end_s) - float(start_s), 0.0)

    def resolved_intrapulse_free_carrier_time_s(self) -> float:
        """Return carrier-rate time advanced by one coherent FFT window."""

        return float(len(self.pulse.time_s) * self.pulse.dt_s)

    def _transient_gain_population_shape(
        self,
        population: np.ndarray,
    ) -> tuple[int, ...]:
        """Return the reservoir shape seen by the transient-gain operator."""

        return tuple(int(value) for value in population.shape)

    def _transport_progress_metrics(self) -> dict[str, Any]:
        """Expose stable coherent state without emitting one UI event per window."""

        metrics: dict[str, Any] = {
            "coherent_fidelity_mode": self.coherent_fidelity_mode,
            "coherent_sample_count": int(len(self.pulse.frequency_Hz)),
            "transient_gain_enabled": self.transient_gain is not None,
        }
        if self.transient_gain is not None:
            diagnostics = self.transient_gain.diagnostics()
            for source, target in (
                ("last_slice_count_used", "transient_gain_slice_count"),
                ("maximum_slice_count", "transient_gain_maximum_slice_count"),
                (
                    "maximum_reservoir_depletion_per_slice",
                    "transient_gain_maximum_depletion_fraction",
                ),
            ):
                value = diagnostics.get(source)
                if value is not None:
                    if source == "last_slice_count_used" and int(value) <= 0:
                        continue
                    metrics[target] = value
        return metrics

    def propagate(self, population: np.ndarray, delta_n: np.ndarray):
        population_before_seed = population.copy()
        self._reset_coherent_run_diagnostics()
        if (
            self.transient_gain is None
            and bool(self.transient_gain_cfg.get("enabled", False))
        ):
            try:
                self.transient_gain = self._new_transient_gain()
                self.transient_gain_status = self.transient_gain.diagnostics()
            except OPTIONAL_BACKEND_FAILURES as exc:
                if self.transient_gain_failure_policy == "error":
                    raise
                reason = f"{type(exc).__name__}: {exc}"
                self.transient_gain_status = {
                    "enabled": True,
                    "status": "fallback_whole_fluence",
                    "method": "whole_fluence",
                    "reason": reason,
                    "fallback_origin": "runtime",
                }
                self.transient_gain_warnings.append(
                    "Coherent transient gain could not start for this propagation "
                    f"({reason}); whole-fluence saturation was used."
                )
        if self.transient_gain is not None:
            self.transient_gain.begin_propagation(
                self._transient_gain_population_shape(population)
            )
        try:
            result = super().propagate(population, delta_n)
        except _TransientGainFallbackRequested as fallback:
            # A local continuation would mix the transient reservoir/population
            # already committed upstream with a freshly initialized whole-fluence
            # reservoir.  Restore the entire pre-Seed state and rerun all passes.
            population[:] = population_before_seed
            self.transient_gain_status = fallback.diagnostics
            self.transient_gain = None
            self._reset_coherent_run_diagnostics()
            self.transient_gain_warnings.append(
                "Coherent transient gain failed safely "
                f"({fallback.reason}); the entire coherent Seed propagation was "
                "rolled back and restarted with whole-fluence saturation."
            )
            result = super().propagate(population, delta_n)
        diagnostics = self._transient_diagnostics()
        if diagnostics.get("accuracy_warning"):
            self.transient_gain_warnings.append(str(diagnostics["accuracy_warning"]))
        if float(diagnostics.get("maximum_resynthesis_energy_error", 0.0)) > 0.1:
            self.transient_gain_warnings.append(
                "The coherent soft-window gain resynthesis energy mismatch exceeded 10%; increase temporal slices/coherent samples or shorten the z step."
            )
        if bool(diagnostics.get("positive_gain_exponent_clipping_active", False)):
            self.transient_gain_warnings.append(
                "A positive unsaturated coherent soft-window gain exponent exceeded 50 and "
                "was clipped before exponentiation. Enable a finite saturation model or refine "
                "Nz before interpreting this result quantitatively."
            )
        if self.maximum_strong_field_fractional_loss > 0.1:
            self.transient_gain_warnings.append(
                "A single realized strong-field substep removed more than 10% of local intensity; increase Nz or the local refinement budget and verify coherent temporal sampling."
            )
        if self.local_operator_substep_cap_active:
            self.transient_gain_warnings.append(
                "The coherent local-operator refinement reached maximum_substeps before all configured gain/dispersion/nonlinear limits were met; increase the cap or refine Nz."
            )
        if self.sequential_coupling_accuracy_warning:
            self.transient_gain_warnings.append(
                "The legacy sequential coherent operator exceeded at least one configured gain/dispersion/nonlinear coupling limit. Use symmetric_adaptive or refine Nz before quantitative use."
            )
        if self.multirate_gain_cells_active > 0:
            self.transient_gain_warnings.append(
                "Coherent multirate gain reduced transient soft-window updates while retaining fine optical substeps. This is a controlled approximation; compare output energy, compressed pulse, and B-integral with multirate_gain.enabled=false before quantitative use."
            )
        if (
            self.cfg.get("coherent", {})
            .get("local_operator", {})
            .get("accuracy_profile", "reference")
            == "cpu_fast_convergence_required"
        ):
            self.transient_gain_warnings.append(
                "The L4.5 CPU-fast split-step profile is active. It retains the "
                "native gain-frequency grid, adaptive soft windows, and "
                "front-to-tail ordering, but may group complete gain/population "
                "calls through the separately reported guarded multirate "
                "approximation. It also uses relaxed thermal/dispersion split "
                "heuristics and disables the optional output caustic. Compare "
                "energy, compressed pulse, B-integral, and population against "
                "the reference profile before quantitative use."
            )
        if (
            self.local_operator_settings.scheme == "symmetric_adaptive"
            and (
                self.maximum_single_step_nonlinear_phase_rad
                > 1.01
                * self.local_operator_settings.maximum_nonlinear_phase_rad_per_substep
                or self.maximum_strong_field_fractional_loss
                > 1.01
                * self.local_operator_settings.maximum_fractional_loss_per_substep
            )
        ):
            self.transient_gain_warnings.append(
                "A realized coherent nonlinear substep exceeded its configured phase/loss limit even after planning, which can occur with rapidly generated free carriers or stronger-than-estimated gain. Increase Nz or tighten/increase the local-operator refinement budget."
            )
        if (
            self.physics.get("nonlinear_phase_model", "off") == "kerr_time_domain"
            and self.physics.get("optical_propagation", "off") != "off"
        ):
            self.transient_gain_warnings.append(
                "The reported cumulative B-integral is accumulated on fixed transverse cells. With diffraction or walk-off it is not a Lagrangian ray/path integral; use the propagated field and z-refinement as authoritative."
            )
        result.warnings = list(dict.fromkeys([*result.warnings, *self.transient_gain_warnings]))
        if result.cpa_diagnostics is not None:
            result.cpa_diagnostics["transient_gain"] = diagnostics
            result.cpa_diagnostics.setdefault("strong_field", {}).update(
                {
                    "initial_free_carrier_density_max_m3": (
                        0.0
                        if self._initial_free_carrier_density_m3 is None
                        else float(
                            np.max(self._initial_free_carrier_density_m3)
                        )
                    ),
                    "final_free_carrier_density_max_m3": float(
                        np.max(self.final_free_carrier_density_m3)
                    ),
                    "free_carrier_state_exported": True,
                }
            )
        result.free_carrier_density_m3 = (
            self.final_free_carrier_density_m3.copy()
        )
        self._initial_free_carrier_density_m3 = None
        return result

    def set_initial_free_carrier_density(
        self, density_m3: np.ndarray | None
    ) -> None:
        if density_m3 is None:
            self._initial_free_carrier_density_m3 = None
            return
        density = np.asarray(density_m3, dtype=float)
        if density.shape != self.grid.shape:
            raise ValueError(
                "Initial free-carrier state must match the coherent crystal grid"
            )
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError(
                "Initial free-carrier state must be finite and non-negative"
            )
        self._initial_free_carrier_density_m3 = density.copy()

    def _reset_coherent_run_diagnostics(self) -> None:
        """Reset every state that belongs to one reported coherent propagation."""

        self._pending_gain_cell = None
        # The same solver object can be reused by periodic/fixed-point probes, so
        # retaining a prior attempt's B-integral or losses would corrupt reports.
        self.transient_gain_warnings = list(self._persistent_coherent_warnings)
        self.maximum_b_integral_rad = 0.0
        self.maximum_single_step_nonlinear_phase_rad = 0.0
        self.maximum_nonlinear_step_peak_power_W = 0.0
        self.maximum_local_nonlinear_intensity_W_m2 = 0.0
        self.nonlinear_phase_sampling_diagnostics = {}
        self.cycle_b_integral_diagnostics = {}
        self.per_pass_b_integral_diagnostics = []
        self._cumulative_b_integral_rad = None
        self.nonlinear_loss_energy_J_m3.fill(0.0)
        self.multiphoton_loss_energy_J_m3.fill(0.0)
        self.free_carrier_loss_energy_J_m3.fill(0.0)
        self.maximum_peak_carrier_density_m3 = 0.0
        self.final_free_carrier_density_m3.fill(0.0)
        self._active_free_carrier_z_index = None
        self._active_free_carrier_initial_m3 = None
        self._pending_free_carrier_density_m3 = (
            None
            if self._initial_free_carrier_density_m3 is None
            else self._initial_free_carrier_density_m3.copy()
        )
        self._pending_free_carrier_cells = (
            None
            if self._pending_free_carrier_density_m3 is None
            else np.ones(len(self.grid.z), dtype=bool)
        )
        self.maximum_strong_field_fractional_loss = 0.0
        self.maximum_local_operator_substeps = 1
        self.maximum_recommended_local_operator_substeps = 1
        self.maximum_estimated_gain_exponent_per_cell = 0.0
        self.maximum_estimated_nonlinear_phase_rad_per_cell = 0.0
        self.maximum_estimated_dispersion_phase_rad_per_cell = 0.0
        self.maximum_estimated_thermal_phase_rad_per_cell = 0.0
        self.maximum_estimated_fractional_loss_per_cell = 0.0
        self.local_operator_substep_cap_active = False
        self.sequential_coupling_accuracy_warning = False
        self._current_crystal_cell_substeps = 1
        self._current_gain_update_schedule = (1,)
        self._current_gain_group_size = 1
        self.multirate_gain_cells_active = 0
        self.multirate_gain_optical_substeps = 0
        self.multirate_gain_updates = 0
        self.multirate_gain_maximum_group_size = 1
        self.multirate_gain_schedule_reasons = {}
        self.maximum_gain_updates_used_per_cell = 0
        self.maximum_estimated_gain_exponent_per_update = 0.0
        # Material/environment state may change between accepted outer
        # multiphysics or transient-thermal updates.  Cache only within one
        # coherent propagation so a future coupled model cannot see stale
        # dispersion factors.
        self._material_dispersion_factor_cache.clear()
        self._material_dispersion_cache_hits = 0
        self._material_dispersion_cache_misses = 0

    def _gain_step(
        self,
        field: np.ndarray,
        population: np.ndarray,
        iz: int,
        step_length_m: float,
    ) -> np.ndarray:
        if self.transient_gain is None:
            return super()._gain_step(field, population, iz, step_length_m)
        population_backup = population[:, iz].copy()
        try:
            cell_substeps = max(self._current_crystal_cell_substeps, 1)
            gain_group_size = max(int(self._current_gain_group_size), 1)
            effective_gain_substeps = cell_substeps / gain_group_size
            path_cosine = self.grid.dz / (
                step_length_m * effective_gain_substeps
            )
            if not 0.0 < path_cosine <= 1.0 + 1e-12:
                raise ValueError(
                    "Seed optical step is inconsistent with the axial cell length"
                )
            output = self.transient_gain.apply(
                field[0],
                population[:, iz],
                iz,
                step_length_m,
                cell_length_m=self.grid.dz,
                relaxation_time_scale=gain_group_size / cell_substeps,
                projected_area_cosine=min(float(path_cosine), 1.0),
            )
            passive_loss = self.transient_gain.last_passive_loss_fluence_J_m2
            if passive_loss is not None:
                self.passive_loss_energy_J_m3[iz] += passive_loss / max(
                    self.grid.dz, np.finfo(float).tiny
                )
            result = field.copy()
            result[0] = output
            if self.clip_crystal:
                result *= self.crystal_mask[None, None, :, :]
            return result
        except OPTIONAL_MODEL_FAILURES as exc:
            population[:, iz] = population_backup
            if self.transient_gain_failure_policy == "error":
                raise
            reason = f"{type(exc).__name__}: {exc}"
            failed = self.transient_gain.diagnostics()
            failed.update(
                {
                    "enabled": True,
                    "status": "fallback_whole_fluence",
                    "method": "whole_fluence",
                    "reason": reason,
                    "rollback_scope": "entire_seed_propagation",
                    "fallback_origin": "runtime",
                }
            )
            raise _TransientGainFallbackRequested(reason, failed) from exc

    def _add_persistent_warning(self, warning: str) -> None:
        if warning not in self._persistent_coherent_warnings:
            self._persistent_coherent_warnings.append(warning)
        if warning not in self.transient_gain_warnings:
            self.transient_gain_warnings.append(warning)

    def _transient_diagnostics(self) -> dict[str, Any]:
        if self.transient_gain is not None:
            return self.transient_gain.diagnostics()
        return dict(self.transient_gain_status)

    def _coherent_spectral_modes(self, field: np.ndarray) -> np.ndarray:
        """Expose the mutually incoherent modal axis used by local CPA physics.

        The ordinary coherent solver has one member.  The explicit-HG
        specialization keeps its transport-compatible flattened channel axis
        but reshapes it here, allowing dispersion, temporal FFTs, Kerr, and
        strong-field loss to use one implementation.
        """

        values = np.asarray(field)
        if values.ndim != 4 or values.shape[0] != 1:
            raise ValueError(
                "coherent field must have shape (1, frequency, y, x)"
            )
        return values

    def _store_coherent_spectral_modes(
        self,
        field: np.ndarray,
        modes: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(modes)
        if values.shape != field.shape:
            raise ValueError("coherent modal field shape changed during propagation")
        field[...] = values
        return field

    def _initialize_field(self, spectral_field: np.ndarray, temporal_weights: np.ndarray) -> np.ndarray:
        del temporal_weights
        if self.reference_modes is None or self._reference_modes_source is not spectral_field:
            energy = self.grid.integrate_transverse(np.abs(spectral_field) ** 2)
            normalized_modes = spectral_field / np.sqrt(
                np.maximum(energy, np.finfo(float).tiny)
            )[:, None, None]
            # Fix each modal reference to a real-valued peak pixel.  This preserves
            # wavelength-dependent global phase accumulated by upstream optics in
            # the later modal projection instead of accidentally cancelling it.
            flattened = normalized_modes.reshape(normalized_modes.shape[0], -1)
            peak_indices = np.argmax(np.abs(flattened) ** 2, axis=1)
            anchor_phase = np.angle(
                flattened[np.arange(len(flattened)), peak_indices]
            )
            self.reference_modes = normalized_modes * np.exp(
                -1j * anchor_phase
            )[:, None, None]
            self._reference_modes_source = spectral_field
        phase = np.exp(1j * np.angle(self.pulse.spectral_field_sqrt_J))
        return (spectral_field * phase[:, None, None])[None, ...]

    def _after_gain_step(
        self,
        field: np.ndarray,
        step_length_m: float,
        z_index: int | None = None,
    ) -> np.ndarray:
        field = self._apply_material_dispersion(field, step_length_m)
        return self._apply_nonlinear_step(
            field,
            step_length_m,
            z_index,
            physical_slice_length_m=self.grid.dz,
        )

    def _update_nonlinear_phase_sampling_diagnostics(
        self,
        nonlinear_phase_rad: np.ndarray,
        intensity_W_m2: np.ndarray,
        *,
        axis_names: tuple[str, ...],
        peak_power_W: float | None = None,
        energy_weighted_intensity_W_m2: float | None = None,
    ) -> None:
        """Track the worst resolved nonlinear screen using scalar values only."""

        phase = np.asarray(nonlinear_phase_rad, dtype=float)
        intensity = np.asarray(intensity_W_m2, dtype=float)
        self.maximum_local_nonlinear_intensity_W_m2 = max(
            float(
                getattr(
                    self,
                    "maximum_local_nonlinear_intensity_W_m2",
                    0.0,
                )
            ),
            float(np.max(intensity)),
        )
        cell = getattr(self, "_active_longitudinal_cell", None)
        if self._longitudinal_nonlinear_cell_active():
            cell_metrics = cell.setdefault("nonlinear", {})
            cell_metrics["maximum_local_intensity_W_m2"] = max(
                float(cell_metrics.get("maximum_local_intensity_W_m2", 0.0)),
                float(np.max(intensity)),
            )
            cell_metrics["maximum_substep_B_increment_rad"] = max(
                float(
                    cell_metrics.get(
                        "maximum_substep_B_increment_rad",
                        0.0,
                    )
                ),
                float(np.max(np.abs(phase))),
            )
            if peak_power_W is not None:
                cell_metrics["maximum_peak_power_W"] = max(
                    float(cell_metrics.get("maximum_peak_power_W", 0.0)),
                    float(peak_power_W),
                )
            if energy_weighted_intensity_W_m2 is not None:
                cell_metrics["energy_weighted_intensity_W_m2"] = max(
                    float(
                        cell_metrics.get(
                            "energy_weighted_intensity_W_m2",
                            0.0,
                        )
                    ),
                    float(energy_weighted_intensity_W_m2),
                )
        if not self._diagnostics_enabled() or not bool(self.physics.get("_record_cycle_scalar_diagnostics", False)):
            return
        step_maximum = float(np.max(np.abs(phase)))
        previous = float(
            getattr(self, "nonlinear_phase_sampling_diagnostics", {}).get(
                "worst_screen_maximum_phase_rad", -1.0
            )
        )
        if step_maximum < previous:
            return
        gradients: dict[str, float] = {}
        for axis, name in enumerate(axis_names):
            if phase.shape[axis] < 2:
                gradients[name] = 0.0
                continue
            gradients[name] = float(
                np.max(np.abs(np.diff(phase, axis=axis))) / math.pi
            )
        self.nonlinear_phase_sampling_diagnostics = {
            "method": "maximum_adjacent_unwrapped_nonlinear_phase_step",
            "worst_screen_maximum_phase_rad": step_maximum,
            "maximum_adjacent_phase_step_fraction_of_pi": gradients,
            "axis_order": list(axis_names),
        }

    @staticmethod
    def _weighted_histogram_percentile(
        values: np.ndarray,
        weights: np.ndarray,
        percentile: float,
        *,
        bins: int = 256,
    ) -> float:
        maximum = float(np.max(values))
        total_weight = float(np.sum(weights))
        if maximum <= 0.0 or total_weight <= np.finfo(float).tiny:
            return 0.0
        histogram, edges = np.histogram(
            values,
            bins=int(bins),
            range=(0.0, maximum),
            weights=weights,
        )
        cumulative = np.cumsum(histogram)
        index = min(
            int(np.searchsorted(cumulative, percentile * total_weight, side="left")),
            len(edges) - 2,
        )
        return float(edges[index + 1])

    def _weighted_b_integral_diagnostics(
        self,
        temporal_field: np.ndarray,
        spatial_cell_weights_m2: np.ndarray,
        *,
        pass_number: int,
        coordinate_axes: tuple[tuple[str, np.ndarray], ...],
        coordinate_model: str,
    ) -> dict[str, Any]:
        """Summarize the current cumulative B distribution at a pass exit."""

        if (
            not self._diagnostics_enabled()
            or not bool(self.physics.get("_record_cycle_scalar_diagnostics", False))
            or self._cumulative_b_integral_rad is None
        ):
            return {}
        b_integral = np.abs(np.asarray(self._cumulative_b_integral_rad, dtype=float))
        field = np.asarray(temporal_field, dtype=complex)
        if b_integral.shape != field.shape:
            return {
                "applicable": False,
                "reason": "cumulative B and pass-exit temporal field shapes differ",
            }
        spatial_weights = np.asarray(spatial_cell_weights_m2, dtype=float)
        if spatial_weights.shape != field.shape[1:]:
            return {
                "applicable": False,
                "reason": "spatial quadrature weights do not match the reduced field",
            }
        intensity = np.abs(field) ** 2 / max(
            float(self.pulse.dt_s), np.finfo(float).tiny
        )
        weights = intensity * spatial_weights[None, ...]
        total_weight = max(float(np.sum(weights)), np.finfo(float).tiny)
        maximum = float(np.max(b_integral))
        maximum_index = tuple(
            int(value) for value in np.unravel_index(np.argmax(b_integral), b_integral.shape)
        )
        location: dict[str, Any] = {
            "indices": list(maximum_index),
        }
        if len(coordinate_axes) == len(maximum_index):
            for (name, coordinate), index in zip(
                coordinate_axes,
                maximum_index,
                strict=True,
            ):
                location[name] = float(np.asarray(coordinate)[index])
        if "x_m" in location and "y_m" in location:
            center_x = 0.5 * float(self.grid.x_edges[0] + self.grid.x_edges[-1])
            center_y = 0.5 * float(self.grid.y_edges[0] + self.grid.y_edges[-1])
            half_x = 0.5 * float(self.grid.x_edges[-1] - self.grid.x_edges[0])
            half_y = 0.5 * float(self.grid.y_edges[-1] - self.grid.y_edges[0])
            location["normalized_window_radius"] = float(
                np.hypot(
                    (float(location["x_m"]) - center_x) / half_x,
                    (float(location["y_m"]) - center_y) / half_y,
                )
            )
        peak_intensity = max(float(np.max(intensity)), np.finfo(float).tiny)
        diagnostics = {
            "applicable": True,
            "method": "pass_exit_intensity_weighted_eulerian_cumulative_B",
            "percentile_method": "256_bin_intensity_weighted_histogram_upper_edge",
            "coordinate_model": coordinate_model,
            "pass": int(pass_number + 1),
            "energy_weighted_mean_rad": float(
                np.sum(b_integral * weights) / total_weight
            ),
            "energy_weighted_p99_rad": self._weighted_histogram_percentile(
                b_integral,
                weights,
                0.99,
            ),
            "maximum_rad": maximum,
            "energy_fraction_above_half_maximum": float(
                0.0
                if maximum <= 0.0
                else np.sum(weights[b_integral >= 0.5 * maximum]) / total_weight
            ),
            "maximum_location": location,
            "intensity_at_maximum_relative_to_pass_peak": float(
                intensity[maximum_index] / peak_intensity
            ),
        }
        self.per_pass_b_integral_diagnostics.append(dict(diagnostics))
        self.cycle_b_integral_diagnostics = dict(diagnostics)
        return diagnostics

    def _gain_pass_scalar_diagnostics(
        self,
        field: np.ndarray,
        pass_number: int,
    ) -> dict[str, Any]:
        if not self._diagnostics_enabled() or not bool(self.physics.get("_record_cycle_scalar_diagnostics", False)):
            return {}
        spectral_modes = self._coherent_spectral_modes(field)
        temporal_modes = self.pulse.spectrum_to_time(
            spectral_modes,
            axis=1,
        )
        # B is a common material phase driven by the summed incoherent modal
        # intensity.  A real surrogate retains that exact intensity while
        # matching the scalar diagnostic's (time,y,x) contract.
        temporal_field = np.sqrt(
            np.sum(np.abs(temporal_modes) ** 2, axis=0)
        ).astype(complex, copy=False)
        diagnostics = self._weighted_b_integral_diagnostics(
            temporal_field,
            self.grid.cell_areas * self.crystal_mask,
            pass_number=pass_number,
            coordinate_axes=(
                ("time_s", self.pulse.time_s),
                ("y_m", self.grid.y),
                ("x_m", self.grid.x),
            ),
            coordinate_model="cartesian_time_y_x",
        )
        return {"b_integral": diagnostics} if diagnostics else {}

    def _local_material_step(
        self,
        field: np.ndarray,
        population: np.ndarray,
        z_index: int,
        step_length_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance one already-planned crystal substep."""

        if self.local_operator_settings.scheme == "sequential":
            return super()._local_material_step(
                field,
                population,
                z_index,
                step_length_m,
            )

        physical_substep = (
            float(self.grid.dz) / self._current_crystal_cell_substeps
        )
        field = self._apply_material_dispersion(field, 0.5 * step_length_m)
        field = self._apply_nonlinear_step(
            field,
            0.5 * step_length_m,
            z_index,
            physical_slice_length_m=0.5 * physical_substep,
        )
        gain_group_size = int(self._current_gain_group_size)
        if gain_group_size > 0:
            fluence_before_gain = np.sum(np.abs(field) ** 2, axis=(0, 1))
            passive_before = self.passive_loss_energy_J_m3[z_index].copy()
            field = self._gain_step(
                field,
                population,
                z_index,
                step_length_m * gain_group_size,
            )
            fluence_after_gain_and_loss = np.sum(
                np.abs(field) ** 2, axis=(0, 1)
            )
            active_transfer = (
                fluence_after_gain_and_loss - fluence_before_gain
            ) / self.grid.dz + (
                self.passive_loss_energy_J_m3[z_index] - passive_before
            )
        else:
            active_transfer = np.zeros(field.shape[-2:], dtype=float)
        field = self._apply_nonlinear_step(
            field,
            0.5 * step_length_m,
            z_index,
            physical_slice_length_m=0.5 * physical_substep,
        )
        field = self._apply_material_dispersion(field, 0.5 * step_length_m)
        return field, active_transfer

    def _begin_crystal_substep(self, index: int, count: int) -> None:
        """Select the gain group represented by this optical microstep."""

        if count != len(self._current_gain_update_schedule):
            raise RuntimeError("Coherent gain schedule does not match local substeps")
        self._current_gain_group_size = int(
            self._current_gain_update_schedule[index]
        )

    def _crystal_cell_substeps(
        self,
        field: np.ndarray,
        population: np.ndarray,
        delta_n_slice: np.ndarray,
        z_index: int,
        step_length_m: float,
    ) -> int:
        if self._pending_gain_cell is not None:
            raise RuntimeError("Previous independent gain cell was not committed")
        # A pending carrier density belongs to this physical z cell and may
        # survive between passes.  Within the current pass, however, every
        # adaptive spatial fraction/Strang half represents a different local
        # z volume.  Those fractions all start from the same pre-pulse cell
        # state and their final densities are volume-averaged below; a carrier
        # history must never travel with the optical field along z.
        self._active_free_carrier_z_index = z_index
        if (
            self._pending_free_carrier_density_m3 is not None
            and self._pending_free_carrier_cells is not None
            and bool(self._pending_free_carrier_cells[z_index])
        ):
            self._active_free_carrier_initial_m3 = (
                self._pending_free_carrier_density_m3[z_index].copy()
            )
            self._pending_free_carrier_cells[z_index] = False
        else:
            self._active_free_carrier_initial_m3 = None
        self.final_free_carrier_density_m3[z_index].fill(0.0)
        active = self.crystal_mask > 0.0
        thermal_phase = 0.0
        if np.any(active):
            shortest_wavelength = float(np.min(self.spectrum.coordinate))
            phase = (
                2.0
                * math.pi
                * np.asarray(delta_n_slice, dtype=float)
                * float(step_length_m)
                / shortest_wavelength
            )
            thermal_phase = float(np.ptp(phase[active]))
        plan = self._local_step_plan(
            field,
            population,
            z_index,
            step_length_m,
            thermal_phase_rad=thermal_phase,
        )
        used_substeps = (
            1
            if self.local_operator_settings.scheme == "sequential"
            else plan.count
        )
        self._record_local_step_plan(plan, used_substeps=used_substeps)
        if self.local_operator_settings.scheme == "sequential":
            self.sequential_coupling_accuracy_warning = (
                self.sequential_coupling_accuracy_warning
                or plan.uncapped_count > 1
            )
        self._current_crystal_cell_substeps = used_substeps
        gain_exponent = self._estimated_gain_operator_exponent(
            population,
            z_index,
            step_length_m,
        )
        gain_plan = choose_multirate_gain_updates(
            self.local_operator_settings,
            optical_substeps=used_substeps,
            gain_exponent=gain_exponent,
            transient_gain_enabled=self.transient_gain is not None,
            whole_fluence_gain_enabled=bool(
                self.transient_gain is None
                and self.coherent_fidelity_mode
                in {"hybrid_whole_fluence", "axisymmetric_kerr"}
            ),
            intrapulse_relaxation_enabled=bool(
                self.transient_gain is not None
                and getattr(self.transient_gain, "include_relaxation", False)
            ),
            coupling_blockers=self._multirate_gain_coupling_blockers(),
        )
        self._current_gain_update_schedule = gain_plan.schedule
        self.maximum_gain_updates_used_per_cell = max(
            self.maximum_gain_updates_used_per_cell, gain_plan.update_count
        )
        self.maximum_estimated_gain_exponent_per_update = max(
            self.maximum_estimated_gain_exponent_per_update,
            gain_exponent * gain_plan.maximum_group_size / used_substeps,
        )
        self._record_multirate_gain_plan(gain_plan)
        return used_substeps

    def _estimated_gain_operator_exponent(
        self,
        population: np.ndarray,
        z_index: int,
        step_length_m: float,
    ) -> float:
        """Bound active gain and passive loss without cancellation."""

        override = self.levels_cfg.get("gain_override_m_inv")
        if override is None:
            active_gain = self.levels.signal_gain_spectrum_m_inv(
                population[:, z_index],
                self.spectrum.coordinate,
                z_index=z_index,
            )
        else:
            active_gain = np.full(
                (1, *population.shape[-2:]),
                float(override),
                dtype=float,
            )
        active_exponent = float(
            np.max(
                np.abs(active_gain * self.crystal_mask[None, ...])
            )
            * float(step_length_m)
        )
        passive_exponent = (
            abs(float(self.material.get("signal_passive_loss_m_inv", 0.0)))
            * float(step_length_m)
        )
        return active_exponent + passive_exponent

    def _multirate_gain_coupling_blockers(self) -> tuple[str, ...]:
        """Allow memoryless MPA; keep carrier/unknown-operator state guards."""
        return nonlinear_gain_coupling_blockers(self.physics, self.material)

    def _record_multirate_gain_plan(
        self,
        plan: CoherentGainUpdatePlan,
    ) -> None:
        self.multirate_gain_optical_substeps += len(plan.schedule)
        self.multirate_gain_updates += plan.update_count
        self.multirate_gain_maximum_group_size = max(
            self.multirate_gain_maximum_group_size,
            plan.maximum_group_size,
        )
        if plan.active:
            self.multirate_gain_cells_active += 1
        self.multirate_gain_schedule_reasons[plan.reason] = (
            self.multirate_gain_schedule_reasons.get(plan.reason, 0) + 1
        )

    def _advance_interpass_state(self, delay_s: float) -> None:
        carrier = self.material.get("strong_field", {}).get("free_carrier", {})
        if str(carrier.get("model", "off")) != "rate_equation":
            return
        lifetime = float(carrier.get("lifetime_s", 0.0))
        if lifetime <= 0.0:
            return
        decay = math.exp(-max(float(delay_s), 0.0) / lifetime)
        self._pending_free_carrier_density_m3 = (
            self.final_free_carrier_density_m3 * decay
        )
        self._pending_free_carrier_cells = np.ones(len(self.grid.z), dtype=bool)

    def _apply_material_dispersion(
        self,
        field: np.ndarray,
        step_length_m: float,
    ) -> np.ndarray:
        factor = self._material_dispersion_factor(step_length_m)
        if factor is not None:
            modes = self._coherent_spectral_modes(field)
            material_phase = factor[None, :, None, None]
            modes *= 1.0 + self.crystal_mask[None, None, :, :] * (
                material_phase - 1.0
            )
            field = self._store_coherent_spectral_modes(field, modes)
        return field

    def _material_dispersion_factor(
        self,
        step_length_m: float,
    ) -> np.ndarray | None:
        """Return one propagation-local complex dispersion multiplier."""

        environment_revision = int(
            getattr(self.levels.response, "environment_revision", 0)
        )
        key = (float(step_length_m), environment_revision)
        if key in self._material_dispersion_factor_cache:
            self._material_dispersion_cache_hits += 1
            return self._material_dispersion_factor_cache[key]
        linear_phase = material_dispersion_phase(
            self.pulse.frequency_Hz,
            self.material.get("dispersion", {}),
            step_length_m,
        )
        factor = (
            None
            if not np.any(linear_phase)
            else np.exp(1j * linear_phase)
        )
        if factor is not None:
            factor.setflags(write=False)
        self._material_dispersion_factor_cache[key] = factor
        self._material_dispersion_cache_misses += 1
        return factor

    def _apply_nonlinear_step(
        self,
        field: np.ndarray,
        step_length_m: float,
        z_index: int | None,
        *,
        physical_slice_length_m: float,
    ) -> np.ndarray:
        use_kerr = bool(
            self.physics.get("nonlinear_phase_model", "off")
            == "kerr_time_domain"
            and float(self.material.get("n2_m2_W", 0.0)) != 0.0
        )
        use_loss = self.physics.get("nonlinear_loss_model", "off") == "strong_field_time_domain"
        if use_kerr or use_loss:
            spectral_modes = self._coherent_spectral_modes(field)
            temporal_modes = self.pulse.spectrum_to_time(
                spectral_modes,
                axis=1,
            )
            k0 = 2.0 * math.pi / float(self.cfg["wavelength_m"])
            # The sampled field is normalized per unit area of the crystal
            # x-y plane.  At oblique incidence that plane sees
            # I_projected = I_beam*cos(theta), whereas Kerr, multiphoton
            # absorption, and free-carrier generation are defined using the
            # intensity on a plane normal to the ray and the distance along
            # that ray.  Convert only while evaluating the material operator;
            # convert its field and fluence ledgers back to the crystal-plane
            # convention before returning them.
            if float(step_length_m) > np.finfo(float).tiny:
                projected_area_cosine = float(physical_slice_length_m) / float(
                    step_length_m
                )
            else:
                projected_area_cosine = 1.0
            if not 0.0 < projected_area_cosine <= 1.0 + 1e-12:
                raise ValueError(
                    "Nonlinear optical step is inconsistent with its physical axial slice"
                )
            projected_area_cosine = min(projected_area_cosine, 1.0)
            beam_field_scale = math.sqrt(projected_area_cosine)
            nonlinear_intensity_path_W_m: np.ndarray | None = None
            incident_peak_power_W: float | None = None
            # Only absorption requires a separate incident-power snapshot.
            if use_kerr and use_loss:
                incident_projected_intensity_W_m2 = (
                    np.sum(np.abs(temporal_modes) ** 2, axis=0)
                    / max(self.pulse.dt_s, np.finfo(float).tiny)
                )
                incident_peak_power_W = float(
                    np.max(
                        np.sum(
                            incident_projected_intensity_W_m2
                            * self.crystal_mask[None, :, :]
                            * self.grid.cell_areas[None, :, :],
                            axis=(1, 2),
                        )
                    )
                )
            if use_loss:
                if z_index is None:
                    raise RuntimeError("strong-field loss requires the active z index")
                strong_input = (
                    temporal_modes
                    * self.crystal_mask[None, None, :, :]
                    / beam_field_scale
                )
                strong_arguments = (
                    self.pulse.dt_s,
                    step_length_m,
                    float(self.cfg["wavelength_m"]),
                    self.material.get("strong_field", {}),
                    k0,
                )
                strong_keywords = {
                    "cell_length_m": step_length_m,
                    "initial_carrier_density_m3": (
                        self._active_free_carrier_initial_m3
                        if self._active_free_carrier_z_index == z_index
                        else None
                    ),
                }
                if temporal_modes.shape[0] == 1:
                    # Preserve the original scalar CPA call contract (and its
                    # independently tested oblique-incidence normalization).
                    strong = apply_strong_field_step(
                        strong_input[0],
                        *strong_arguments,
                        **strong_keywords,
                    )
                    strong.field = strong.field[None, ...]
                else:
                    strong = apply_incoherent_strong_field_step(
                        strong_input,
                        *strong_arguments,
                        **strong_keywords,
                    )
                self._active_free_carrier_z_index = z_index
                spatial_weight = float(physical_slice_length_m) / max(
                    float(self.grid.dz), np.finfo(float).tiny
                )
                self.final_free_carrier_density_m3[z_index] += (
                    strong.carrier_density_m3[-1] * spatial_weight
                )
                temporal_modes = (
                    temporal_modes
                    * (1.0 - self.crystal_mask[None, None, :, :])
                    + strong.field * beam_field_scale
                )
                nonlinear_intensity_path_W_m = strong.nonlinear_intensity_path_W_m
                volume_scale = 1.0 / max(self.grid.dz, np.finfo(float).tiny)
                self.nonlinear_loss_energy_J_m3[z_index] += (
                    strong.absorbed_fluence_J_m2
                    * projected_area_cosine
                    * volume_scale
                )
                self.multiphoton_loss_energy_J_m3[z_index] += (
                    strong.multiphoton_absorbed_fluence_J_m2
                    * projected_area_cosine
                    * volume_scale
                )
                self.free_carrier_loss_energy_J_m3[z_index] += (
                    strong.free_carrier_absorbed_fluence_J_m2
                    * projected_area_cosine
                    * volume_scale
                )
                self.maximum_peak_carrier_density_m3 = max(
                    self.maximum_peak_carrier_density_m3,
                    strong.peak_carrier_density_m3,
                )
                self.maximum_strong_field_fractional_loss = max(
                    self.maximum_strong_field_fractional_loss,
                    strong.maximum_fractional_loss,
                )
            if use_kerr:
                n2 = float(self.material.get("n2_m2_W", 0.0))
                projected_intensity_W_m2 = (
                    np.sum(np.abs(temporal_modes) ** 2, axis=0)
                    / max(self.pulse.dt_s, np.finfo(float).tiny)
                )
                if nonlinear_intensity_path_W_m is None:
                    beam_normal_intensity_W_m2 = (
                        projected_intensity_W_m2 / projected_area_cosine
                    )
                    nonlinear_intensity_path_W_m = (
                        beam_normal_intensity_W_m2
                        * step_length_m
                        * self.crystal_mask[None, :, :]
                    )
                peak_power_W = float(
                    np.max(
                        np.sum(
                            projected_intensity_W_m2
                            * self.crystal_mask[None, :, :]
                            * self.grid.cell_areas[None, :, :],
                            axis=(1, 2),
                        )
                    )
                )
                # A nonlinear-loss operator can only reduce the power relevant
                # to the critical-power warning.  Preserve the incident value
                # as well as the post-loss value so that enabling absorption
                # cannot make the self-focusing diagnostic less conservative.
                if incident_peak_power_W is not None:
                    peak_power_W = max(peak_power_W, incident_peak_power_W)
                self.maximum_nonlinear_step_peak_power_W = max(
                    self.maximum_nonlinear_step_peak_power_W,
                    peak_power_W,
                )
                nonlinear_phase = k0 * n2 * nonlinear_intensity_path_W_m
                energy_weighted_intensity = None
                if self._longitudinal_nonlinear_cell_active():
                    intensity_weight = (
                        self.grid.cell_areas[None, :, :]
                        * self.crystal_mask[None, :, :]
                    )
                    intensity_integral = float(
                        np.sum(projected_intensity_W_m2 * intensity_weight)
                    )
                    energy_weighted_intensity = float(
                        np.sum(projected_intensity_W_m2**2 * intensity_weight)
                        / max(intensity_integral, np.finfo(float).tiny)
                    )
                self._update_nonlinear_phase_sampling_diagnostics(
                    nonlinear_phase,
                    projected_intensity_W_m2,
                    axis_names=("time", "y", "x"),
                    peak_power_W=peak_power_W,
                    energy_weighted_intensity_W_m2=(
                        energy_weighted_intensity
                    ),
                )
                if self._cumulative_b_integral_rad is None:
                    self._cumulative_b_integral_rad = np.zeros_like(nonlinear_phase)
                self._cumulative_b_integral_rad += nonlinear_phase
                self.maximum_single_step_nonlinear_phase_rad = max(
                    self.maximum_single_step_nonlinear_phase_rad,
                    float(np.max(np.abs(nonlinear_phase))),
                )
                self.maximum_b_integral_rad = max(
                    self.maximum_b_integral_rad,
                    float(np.max(np.abs(self._cumulative_b_integral_rad))),
                )
                temporal_modes *= np.exp(1j * nonlinear_phase)[None, ...]
            spectral_modes = self.pulse.time_to_spectrum(
                temporal_modes,
                axis=1,
            )
            field = self._store_coherent_spectral_modes(
                field,
                spectral_modes,
            )
        return field

    def _local_step_plan(
        self,
        field: np.ndarray,
        population: np.ndarray,
        z_index: int,
        step_length_m: float,
        *,
        thermal_phase_rad: float = 0.0,
    ) -> CoherentLocalStepPlan:
        gain_exponent, positive_gain_exponent = self._estimated_gain_exponents(
            population,
            z_index,
            step_length_m,
        )
        use_kerr = bool(
            self.physics.get("nonlinear_phase_model", "off")
            == "kerr_time_domain"
            and float(self.material.get("n2_m2_W", 0.0)) != 0.0
        )
        use_loss = bool(
            self.physics.get("nonlinear_loss_model", "off")
            == "strong_field_time_domain"
        )
        if use_kerr or use_loss:
            spectral_modes = self._coherent_spectral_modes(field)
            temporal_modes = self.pulse.spectrum_to_time(
                spectral_modes,
                axis=1,
            )
            projected_area_cosine = float(self.grid.dz) / float(step_length_m)
            if not 0.0 < projected_area_cosine <= 1.0 + 1e-12:
                raise ValueError(
                    "Nonlinear planning step is inconsistent with the axial grid cell"
                )
            projected_area_cosine = min(projected_area_cosine, 1.0)
            peak_input_intensity = float(
                np.max(np.sum(np.abs(temporal_modes) ** 2, axis=0))
                / max(self.pulse.dt_s, np.finfo(float).tiny)
                / projected_area_cosine
            )
            peak_intensity = peak_input_intensity * math.exp(
                min(max(positive_gain_exponent, 0.0), 50.0)
            )
        else:
            peak_intensity = 0.0
        nonlinear_phase = 0.0
        if use_kerr:
            nonlinear_phase = (
                2.0
                * math.pi
                / float(self.cfg["wavelength_m"])
                * abs(float(self.material.get("n2_m2_W", 0.0)))
                * peak_intensity
                * step_length_m
            )
        dispersion_phase = 0.0
        if use_kerr or use_loss or self.transient_gain is not None:
            phase = material_dispersion_phase(
                self.pulse.frequency_Hz,
                self.material.get("dispersion", {}),
                step_length_m,
            )
            if phase.size:
                dispersion_phase = float(np.ptp(phase))
        coupled_gain_exponent = (
            gain_exponent
            if (
                use_kerr
                or use_loss
                or (self.transient_gain is not None and dispersion_phase > 0.0)
                or (
                    bool(self.physics.get("gain_saturation", False))
                    and float(self.material.get("signal_passive_loss_m_inv", 0.0))
                    > 0.0
                )
            )
            else 0.0
        )
        fractional_loss = self._estimated_strong_field_loss(
            peak_intensity,
            step_length_m,
        )
        return choose_local_substeps(
            self.local_operator_settings,
            gain_exponent=coupled_gain_exponent,
            nonlinear_phase_rad=nonlinear_phase,
            dispersion_phase_rad=dispersion_phase,
            thermal_phase_rad=thermal_phase_rad,
            fractional_loss=fractional_loss,
            loss_order=int(self.material.get("strong_field", {}).get("multiphoton_order", 2)),
        )

    def _estimated_gain_exponents(
        self,
        population: np.ndarray,
        z_index: int,
        step_length_m: float,
    ) -> tuple[float, float]:
        override = self.levels_cfg.get("gain_override_m_inv")
        if override is None:
            active_gain = self.levels.signal_gain_spectrum_m_inv(
                population[:, z_index],
                self.spectrum.coordinate,
                z_index=z_index,
            )
        else:
            reservoir = getattr(self, "_override_gain_m_inv", None)
            active_gain = (
                np.asarray(reservoir[z_index], dtype=float)[None, ...]
                if reservoir is not None
                else np.full(
                    (1, *population.shape[-2:]),
                    float(override),
                    dtype=float,
                )
            )
        # Saturable gain and passive loss are separate, noncommuting operators.
        # Their cancellation cannot make either operator safe for a large step.
        active_exponent = (
            active_gain * self.crystal_mask[None, ...] * float(step_length_m)
        )
        passive_exponent = (
            abs(float(self.material.get("signal_passive_loss_m_inv", 0.0)))
            * float(step_length_m)
        )
        return (
            float(np.max(np.abs(active_exponent))) + passive_exponent,
            max(float(np.max(active_exponent)), 0.0),
        )

    def _estimated_strong_field_loss(
        self,
        peak_intensity_W_m2: float,
        step_length_m: float,
    ) -> float:
        if self.physics.get("nonlinear_loss_model", "off") != "strong_field_time_domain":
            return 0.0
        strong_field = self.material.get("strong_field", {})
        order = int(strong_field.get("multiphoton_order", 2))
        beta = float(strong_field.get("beta_SI", 0.0))
        if beta <= 0.0 or peak_intensity_W_m2 <= 0.0:
            return 0.0
        transmission = multiphoton_transmission_fraction(
            peak_intensity_W_m2,
            beta,
            order,
            float(step_length_m),
        )
        return float(1.0 - transmission)

    def _record_local_step_plan(
        self,
        plan: CoherentLocalStepPlan,
        *,
        used_substeps: int,
    ) -> None:
        self.maximum_local_operator_substeps = max(
            self.maximum_local_operator_substeps,
            used_substeps,
        )
        self.maximum_recommended_local_operator_substeps = max(
            self.maximum_recommended_local_operator_substeps,
            plan.uncapped_count,
        )
        self.maximum_estimated_gain_exponent_per_cell = max(
            self.maximum_estimated_gain_exponent_per_cell,
            plan.gain_exponent,
        )
        self.maximum_estimated_nonlinear_phase_rad_per_cell = max(
            self.maximum_estimated_nonlinear_phase_rad_per_cell,
            plan.nonlinear_phase_rad,
        )
        self.maximum_estimated_dispersion_phase_rad_per_cell = max(
            self.maximum_estimated_dispersion_phase_rad_per_cell,
            plan.dispersion_phase_rad,
        )
        self.maximum_estimated_thermal_phase_rad_per_cell = max(
            self.maximum_estimated_thermal_phase_rad_per_cell,
            plan.thermal_phase_rad,
        )
        self.maximum_estimated_fractional_loss_per_cell = max(
            self.maximum_estimated_fractional_loss_per_cell,
            plan.fractional_loss,
        )
        self.local_operator_substep_cap_active = (
            self.local_operator_substep_cap_active or plan.capped
        )

    def _initialize_longitudinal_cell_context(
        self,
        field: np.ndarray,
        context: dict[str, Any],
    ) -> None:
        super()._initialize_longitudinal_cell_context(field, context)
        if (
            not bool(self._result_recording["longitudinal"]["nonlinear"])
            or not self._longitudinal_nonlinear_applicable()
        ):
            return
        z_index = int(context["z_index"])
        transverse_volume = self.grid.cell_areas * self.grid.dz
        context["nonlinear_loss_start_J"] = float(
            np.sum(self.nonlinear_loss_energy_J_m3[z_index] * transverse_volume)
        )
        context["multiphoton_loss_start_J"] = float(
            np.sum(self.multiphoton_loss_energy_J_m3[z_index] * transverse_volume)
        )
        context["free_carrier_loss_start_J"] = float(
            np.sum(self.free_carrier_loss_energy_J_m3[z_index] * transverse_volume)
        )

    def _longitudinal_extra_metrics(
        self,
        field: np.ndarray,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        del field
        if not self._longitudinal_nonlinear_applicable():
            return {}
        metrics = dict(context.get("nonlinear", {}))
        z_index = int(context["z_index"])
        transverse_volume = self.grid.cell_areas * self.grid.dz

        def accumulated_cell_energy(values: np.ndarray, start_key: str) -> float:
            total = float(np.sum(values[z_index] * transverse_volume))
            return max(total - float(context.get(start_key, 0.0)), 0.0)

        metrics.update(
            {
                "nonlinear_loss_energy_J": accumulated_cell_energy(
                    self.nonlinear_loss_energy_J_m3,
                    "nonlinear_loss_start_J",
                ),
                "multiphoton_loss_energy_J": accumulated_cell_energy(
                    self.multiphoton_loss_energy_J_m3,
                    "multiphoton_loss_start_J",
                ),
                "free_carrier_loss_energy_J": accumulated_cell_energy(
                    self.free_carrier_loss_energy_J_m3,
                    "free_carrier_loss_start_J",
                ),
                "final_free_carrier_density_max_m3": float(
                    np.max(self.final_free_carrier_density_m3[z_index])
                ),
                "maximum_cumulative_B_integral_rad": float(
                    self.maximum_b_integral_rad
                ),
            }
        )
        peak_power = metrics.get("maximum_peak_power_W")
        if peak_power is not None:
            screen = kerr_self_focusing_screen(
                wavelength_m=float(self.cfg["wavelength_m"]),
                refractive_index=material_refractive_index(
                    self.material,
                    float(self.cfg["wavelength_m"]),
                    role="signal",
                ),
                n2_m2_W=float(self.material.get("n2_m2_W", 0.0)),
                peak_power_W=float(peak_power),
            )
            metrics["critical_power_W"] = screen.get("critical_power_W")
            metrics["peak_to_critical_power_ratio"] = screen.get(
                "peak_to_critical_power_ratio"
            )
            metrics["critical_power_screen_regime"] = screen.get("regime")
        return metrics

    def _longitudinal_nonlinear_applicable(self) -> bool:
        return bool(
            str(self.physics.get("nonlinear_phase_model", "off")) != "off"
            or str(self.physics.get("nonlinear_loss_model", "off")) != "off"
        )

    def _longitudinal_nonlinear_cell_active(self) -> bool:
        cell = getattr(self, "_active_longitudinal_cell", None)
        return bool(
            cell is not None
            and cell.get("regular_sample", False)
            and self._result_recording["longitudinal"]["nonlinear"]
            and self._longitudinal_nonlinear_applicable()
        )

    def _longitudinal_capabilities(self) -> dict[str, bool]:
        capabilities = super()._longitudinal_capabilities()
        capabilities["nonlinear"] = self._longitudinal_nonlinear_applicable()
        capabilities["intensity_snapshots"] = True
        return capabilities

    def _intensity_snapshot_geometry(
        self, field: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del field
        return self.grid.cell_areas * self.crystal_mask, {
            "coordinate_system": "cartesian",
            "transverse_axes": ["y", "x"],
            "x_m": self.grid.x.tolist(),
            "y_m": self.grid.y.tolist(),
        }

    def _record_longitudinal_intensity(
        self, field: np.ndarray, *, pass_number: int, reverse: bool,
        cell_number: int, cell_count: int, z_index: int, step_length_m: float,
    ) -> None:
        """Retain two instantaneous native-grid slices at physical faces.

        The peak is selected by spatially integrated P(t), never by the
        brightest pixel or an incoherent spectral sum. The second slice
        follows the same retarded-time bin throughout each crystal pass.
        This observer never mutates the field or adaptive-step state.
        """
        policy = self._result_recording["longitudinal"]
        if not policy.get("intensity_snapshots", False) or not self._diagnostics_enabled():
            return
        stride = int(policy["intensity_z_stride"])
        if cell_number not in {0, cell_count} and cell_number % stride:
            return
        area, geometry = self._intensity_snapshot_geometry(field)
        cosine = min(float(self.grid.dz) / float(step_length_m), 1.0)
        if cosine <= 0.0:
            raise ValueError("Intensity snapshot requires a positive path cosine")
        temporal = self.pulse.spectrum_to_time(
            self._coherent_spectral_modes(field), axis=1,
        )
        intensity = np.sum(np.abs(temporal) ** 2, axis=0)
        intensity /= self.pulse.dt_s * cosine
        power = np.sum(intensity * area[None, :, :], axis=(1, 2)) * cosine
        peak_index = int(np.argmax(power))
        if cell_number == 0:
            self._intensity_fixed_time_index = peak_index
            if not self._intensity_metadata:
                self._intensity_metadata = {
                    "schema_version": 1,
                    "array_name": "longitudinal_intensity_W_m2",
                    "array_axes": ["plane", "time_selection", *geometry["transverse_axes"]],
                    "selection_names": ["local_power_peak", "entrance_peak_fixed_time"],
                    "intensity_units": "W/m^2",
                    "intensity_convention": "beam_normal_intensity_on_crystal_xy_coordinates",
                    "area_array_name": "longitudinal_intensity_area_m2",
                    "power_integral": "sum(I * area) * projected_area_cosine",
                    "temporal_power_array_name": "longitudinal_temporal_power_W",
                    "time_s": self.pulse.time_s.tolist(),
                    "dt_s": float(self.pulse.dt_s),
                    "time_reference": "solver_retarded_time; fixed index resets at each pass entrance",
                    "intensity_z_stride": stride,
                    "retains_complex_field": False,
                    "retains_all_time_slices": False,
                    "scope": "Final reported Seed propagation, crystal entrance after interface and transport projection, and selected cell exits before exit interface/compression. No adaptive microstep snapshots. D4sigma radii describe the whole selected instantaneous slice, not just its central lobe.",
                    "grids_by_pass": {},
                    "planes": [],
                }
                self._intensity_area_m2 = np.array(area, dtype=float, copy=True)
            self._intensity_metadata["grids_by_pass"][str(pass_number + 1)] = geometry
        selected_indices = [peak_index, self._intensity_fixed_time_index]
        selected = intensity[selected_indices].copy()
        screen = kerr_self_focusing_screen(
            wavelength_m=float(self.cfg["wavelength_m"]),
            refractive_index=material_refractive_index(
                self.material, float(self.cfg["wavelength_m"]), role="signal",
            ),
            n2_m2_W=float(self.material.get("n2_m2_W", 0.0)),
            peak_power_W=float(power[peak_index]),
        )
        selections = []
        for index, plane in zip(selected_indices, selected):
            # Reuse the native spatial quadrature (including complete m=0
            # annuli), but keep only metrics with intensity-independent units.
            spatial = self._longitudinal_spatial_sample(
                np.sqrt(plane)[None, None, :, :],
            )
            moments = self._longitudinal_beam_metrics(*spatial)
            critical = screen.get("critical_power_W")
            selections.append({
                "time_index": index,
                "time_s": float(self.pulse.time_s[index]),
                "power_W": float(power[index]),
                "peak_intensity_W_m2": float(np.max(plane[area > 0.0]))
                if np.any(area > 0.0) else 0.0,
                "P_over_Pcr": float(power[index] / critical) if critical else None,
                **{name: moments[name] for name in (
                    "centroid_x_m", "centroid_y_m", "radius_x_D4sigma_m",
                    "radius_y_D4sigma_m", "effective_area_m2",
                )},
            })
        edge_index = (
            z_index + (1 if reverse else 0) if cell_number == 0
            else z_index + (0 if reverse else 1)
        )
        self._intensity_metadata["planes"].append({
            "plane_index": len(self._intensity_planes),
            "pass": pass_number + 1,
            "direction": "backward" if reverse else "forward",
            "cell_number": cell_number,
            "z_index": z_index,
            "sample_location": "crystal_entrance" if cell_number == 0 else "physical_z_cell_exit",
            "z_m": float(self.grid.z_edges[edge_index]),
            "path_position_m": float(cell_number * step_length_m),
            "projected_area_cosine": cosine,
            "critical_power_W": screen.get("critical_power_W"),
            "critical_power_method": screen["method"],
            "selections": selections,
        })
        self._intensity_planes.append(selected)
        self._intensity_temporal_power.append(power)

    def _result_extras(
        self,
        field: np.ndarray,
        source_total_J: float,
        source_spectrum_J: np.ndarray,
        output_spectrum_J: np.ndarray,
    ) -> dict[str, Any]:
        input_equivalent = np.sqrt(np.maximum(source_spectrum_J, 0.0)) * np.exp(
            1j * np.angle(self.pulse.spectral_field_sqrt_J)
        )
        output_phase, projection_efficiency = self._equivalent_output_phase(field[0], output_spectrum_J)
        output_equivalent = np.sqrt(np.maximum(output_spectrum_J, 0.0)) * np.exp(1j * output_phase)
        compressor_phase = phase_from_coefficients(
            self.pulse.frequency_Hz,
            self.cfg.get("coherent", {}).get("compressor", {}),
        )
        compressed_equivalent = output_equivalent * np.exp(1j * compressor_phase)
        input_temporal = self.pulse.spectrum_to_time(input_equivalent)
        input_energy = np.abs(input_temporal) ** 2
        suppress_output_diagnostics = bool(
            self.physics.get("_suppress_output_diagnostics", False)
        )
        if suppress_output_diagnostics:
            output_energy = np.abs(
                self.pulse.spectrum_to_time(output_equivalent)
            ) ** 2
            compressed_energy = np.abs(
                self.pulse.spectrum_to_time(compressed_equivalent)
            ) ** 2
            temporal_energy_method = "suppressed_reference_mode_proxy"
        else:
            output_energy = self._integrated_spatiotemporal_energy(field)
            compressed_energy = self._integrated_spatiotemporal_energy(
                field,
                spectral_phase_rad=compressor_phase,
            )
            temporal_energy_method = "spatially_integrated_complex_field_ifft"
        dt = max(self.pulse.dt_s, np.finfo(float).tiny)
        boundary_fraction = float(
            self.cfg.get("coherent", {}).get("temporal_boundary_fraction", 0.05)
        )
        spectral_boundary_fraction = float(
            self.cfg.get("coherent", {})
            .get("sampling_contract", {})
            .get("boundary_sample_fraction", 0.05)
        )

        def edge_energy_fraction(
            values: np.ndarray,
            sample_fraction: float,
        ) -> float:
            energy = np.maximum(np.asarray(values, dtype=float), 0.0)
            boundary_count = max(
                1,
                min(
                    len(energy) // 2,
                    int(math.ceil(sample_fraction * len(energy))),
                ),
            )
            boundary = float(
                np.sum(energy[:boundary_count])
                + np.sum(energy[-boundary_count:])
            )
            return boundary / max(float(np.sum(energy)), np.finfo(float).tiny)

        temporal_boundaries = {
            "input": edge_energy_fraction(input_energy, boundary_fraction),
            "output": edge_energy_fraction(output_energy, boundary_fraction),
            "compressed": edge_energy_fraction(
                compressed_energy,
                boundary_fraction,
            ),
        }
        maximum_temporal_boundary = max(temporal_boundaries.values())
        spectral_boundaries = {
            "input": edge_energy_fraction(
                source_spectrum_J,
                spectral_boundary_fraction,
            ),
            "output": edge_energy_fraction(
                output_spectrum_J,
                spectral_boundary_fraction,
            ),
        }
        maximum_output_spectral_boundary = spectral_boundaries["output"]
        boundary_warning = float(
            self.cfg.get("coherent", {}).get(
                "temporal_boundary_warning_fraction", 1e-6
            )
        )
        spectral_boundary_warning = float(
            self.cfg.get("coherent", {})
            .get("sampling_contract", {})
            .get("maximum_spectral_boundary_energy_fraction", 1e-6)
        )
        if maximum_temporal_boundary > boundary_warning:
            self.transient_gain_warnings.append(
                "The coherent temporal FFT boundary contains "
                f"{maximum_temporal_boundary:.3e} of the pulse energy; enlarge "
                "the temporal span before trusting FWHM, peak power, or compressor results."
            )
        if maximum_output_spectral_boundary > spectral_boundary_warning:
            self.transient_gain_warnings.append(
                "The nonlinear output occupies "
                f"{maximum_output_spectral_boundary:.3e} of the spectral FFT "
                "boundary. Enlarge the frequency guard band before trusting "
                "SPM-broadened spectral structure or the conjugate time trace."
            )
        input_peak_power_W = float(np.max(input_energy) / dt)
        output_peak_power_W = float(np.max(output_energy) / dt)
        compressed_peak_power_W = float(np.max(compressed_energy) / dt)
        maximum_screened_peak_power_W = max(
            input_peak_power_W,
            output_peak_power_W,
            self.maximum_nonlinear_step_peak_power_W,
        )
        self_focusing = kerr_self_focusing_screen(
            wavelength_m=float(self.cfg["wavelength_m"]),
            refractive_index=material_refractive_index(
                self.material,
                float(self.cfg["wavelength_m"]),
                role="signal",
            ),
            n2_m2_W=float(self.material.get("n2_m2_W", 0.0)),
            peak_power_W=maximum_screened_peak_power_W,
        )
        self_focusing.update(
            {
                "input_peak_power_W": input_peak_power_W,
                "output_peak_power_W": output_peak_power_W,
                "maximum_in_crystal_step_peak_power_W": (
                    self.maximum_nonlinear_step_peak_power_W
                ),
            }
        )
        if self_focusing.get("regime") == "above_screening_threshold":
            self.transient_gain_warnings.append(
                "The coherent field exceeds the approximate bulk Kerr "
                "self-focusing critical-power screen "
                f"(P/Pcr={float(self_focusing['peak_to_critical_power_ratio']):.3g}). "
                "The result is therefore threshold-sensitive to n2, focus, gain, "
                "thermal lensing, transverse sampling, and omitted nonlinear loss; "
                "treat large M2/SPM changes as a regime warning rather than a "
                "sample-calibrated prediction."
            )
        diagnostics = dict(self.pulse.diagnostics)
        diagnostics.update(
            {
                "fidelity_mode": self.coherent_fidelity_mode,
                "spatial_temporal_field_resolved": True,
                "gain_temporal_closure": (
                    "soft_window_front_to_tail"
                    if self.transient_gain is not None
                    else "whole_fluence_frantz_nodvik"
                ),
                "gain_phase_contract": {
                    "model": "real_amplitude_gain_with_independent_material_dispersion",
                    "population_dependent_resonant_dispersion_included": False,
                    "scope": (
                        "Gain cross sections determine amplitude and population transfer. "
                        "The gain spectrum is not converted into a population-dependent "
                        "complex susceptibility or Kramers-Kronig phase; configured "
                        "material dispersion is applied separately."
                    ),
                },
                "hybrid_approximation": (
                    {
                        "active": True,
                        "retained": [
                            "complex_E_frequency_y_x",
                            "fft_diffraction",
                            "dispersion",
                            "interference",
                            "local_time_domain_kerr",
                        ],
                        "approximated": [
                            "intrapulse_front_to_tail_gain_depletion",
                            "periodic_population_if_cycle_average_selected",
                        ],
                    }
                    if self.coherent_fidelity_mode == "hybrid_whole_fluence"
                    else {"active": False}
                ),
                "input_grid_energy_J": source_total_J,
                "output_energy_J": float(np.sum(output_spectrum_J)),
                "input_fwhm_s": intensity_fwhm(self.pulse.time_s, input_energy),
                "output_fwhm_s": intensity_fwhm(self.pulse.time_s, output_energy),
                "compressed_fwhm_s": intensity_fwhm(self.pulse.time_s, compressed_energy),
                "input_peak_power_W": input_peak_power_W,
                "output_peak_power_W": output_peak_power_W,
                "compressed_peak_power_W": compressed_peak_power_W,
                "kerr_self_focusing_screen": self_focusing,
                "temporal_boundary_energy_fraction": temporal_boundaries,
                "maximum_temporal_boundary_energy_fraction": (
                    maximum_temporal_boundary
                ),
                "temporal_boundary_sample_fraction_each_side": boundary_fraction,
                "temporal_boundary_warning_fraction": boundary_warning,
                "spectral_boundary_energy_fraction": spectral_boundaries,
                "maximum_output_spectral_boundary_energy_fraction": (
                    maximum_output_spectral_boundary
                ),
                "spectral_boundary_warning_fraction": spectral_boundary_warning,
                "spectral_boundary_sample_fraction_each_side": (
                    spectral_boundary_fraction
                ),
                "spatial_mode_projection_efficiency": projection_efficiency,
                "temporal_energy_method": temporal_energy_method,
                # Keep the old key as a compatibility alias, but make its
                # corrected cumulative meaning explicit with a new name.
                "maximum_local_B_integral_rad": self.maximum_b_integral_rad,
                "maximum_cumulative_B_integral_rad": self.maximum_b_integral_rad,
                "maximum_eulerian_cumulative_B_integral_rad": self.maximum_b_integral_rad,
                "maximum_lagrangian_path_B_integral_rad": None,
                "maximum_single_step_nonlinear_phase_rad": self.maximum_single_step_nonlinear_phase_rad,
                "maximum_local_nonlinear_intensity_W_m2": (
                    self.maximum_local_nonlinear_intensity_W_m2
                ),
                "nonlinear_phase_sampling": dict(
                    self.nonlinear_phase_sampling_diagnostics
                ),
                "b_integral_distribution": dict(
                    self.cycle_b_integral_diagnostics
                ),
                "b_integral_per_pass": [
                    dict(row) for row in self.per_pass_b_integral_diagnostics
                ],
                "b_integral_diagnostic_scope": (
                    "eulerian_fixed_transverse_cells; exact as a path integral only when transverse transport is negligible"
                ),
                "local_operator": {
                    "spatial_population_update": self.local_operator_settings.spatial_population_update,
                    "accuracy_profile": str(
                        self.cfg.get("coherent", {})
                        .get("local_operator", {})
                        .get("accuracy_profile", "reference")
                    ),
                    "scheme": self.local_operator_settings.scheme,
                    "adaptive_substeps": self.local_operator_settings.adaptive_substeps,
                    "maximum_substeps_used": self.maximum_local_operator_substeps,
                    "maximum_substeps_recommended_by_limits": self.maximum_recommended_local_operator_substeps,
                    "maximum_substeps_configured": self.local_operator_settings.maximum_substeps,
                    "substep_cap_active": self.local_operator_substep_cap_active,
                    "maximum_estimated_gain_exponent_per_cell": self.maximum_estimated_gain_exponent_per_cell,
                    "maximum_estimated_nonlinear_phase_rad_per_cell": self.maximum_estimated_nonlinear_phase_rad_per_cell,
                    "maximum_estimated_dispersion_phase_rad_per_cell": self.maximum_estimated_dispersion_phase_rad_per_cell,
                    "maximum_estimated_thermal_phase_rad_per_cell": self.maximum_estimated_thermal_phase_rad_per_cell,
                    "maximum_estimated_fractional_loss_per_cell": self.maximum_estimated_fractional_loss_per_cell,
                    "multirate_gain": {
                        "maximum_updates_per_cell": self.local_operator_settings.maximum_gain_updates_per_cell,
                        "maximum_updates_used_per_cell": self.maximum_gain_updates_used_per_cell,
                        "maximum_estimated_gain_exponent_per_update": self.maximum_estimated_gain_exponent_per_update,
                        "gain_exponent_limit_exceeded": (
                            self.maximum_estimated_gain_exponent_per_update
                            > self.local_operator_settings.maximum_gain_exponent_per_update
                        ),
                        "configured_enabled": self.local_operator_settings.multirate_gain_enabled,
                        "active": self.multirate_gain_cells_active > 0,
                        "cells_using_multirate": self.multirate_gain_cells_active,
                        "optical_substeps": self.multirate_gain_optical_substeps,
                        "gain_updates": self.multirate_gain_updates,
                        "gain_update_reduction_fraction": (
                            0.0
                            if self.multirate_gain_optical_substeps <= 0
                            else 1.0
                            - self.multirate_gain_updates
                            / self.multirate_gain_optical_substeps
                        ),
                        "maximum_group_size": self.multirate_gain_maximum_group_size,
                        "maximum_gain_exponent_per_update": self.local_operator_settings.maximum_gain_exponent_per_update,
                        "maximum_optical_substeps_per_update": self.local_operator_settings.maximum_optical_substeps_per_gain_update,
                        "schedule_reason_counts": dict(self.multirate_gain_schedule_reasons),
                        "current_coupling_blockers": list(
                            self._multirate_gain_coupling_blockers()
                        ),
                    },
                },
                "material_dispersion_cache": {
                    "scope": "single_coherent_propagation_and_environment_revision",
                    "hits": self._material_dispersion_cache_hits,
                    "misses": self._material_dispersion_cache_misses,
                    "factor_count": len(
                        self._material_dispersion_factor_cache
                    ),
                },
                "strong_field": {
                    "enabled": self.physics.get("nonlinear_loss_model", "off") == "strong_field_time_domain",
                    "model": self.material.get("strong_field", {}).get("model", "off"),
                    "mechanism": self.material.get("strong_field", {}).get(
                        "mechanism", "unspecified"
                    ),
                    "multiphoton_order": int(
                        self.material.get("strong_field", {}).get(
                            "multiphoton_order", 2
                        )
                    ),
                    "beta_SI": float(
                        self.material.get("strong_field", {}).get("beta_SI", 0.0)
                    ),
                    "provenance": dict(
                        self.material.get("strong_field", {}).get(
                            "provenance", {}
                        )
                    ),
                    "free_carrier_model": str(
                        self.material.get("strong_field", {})
                        .get("free_carrier", {})
                        .get("model", "off")
                    ),
                    "heat_deposition_contract": (
                        "cycle_averaged_effective_fraction_no_explicit_carrier_energy_reservoir"
                    ),
                    "deposit_fraction_as_heat": float(
                        self.material.get("strong_field", {}).get(
                            "deposit_fraction_as_heat", 1.0
                        )
                    ),
                    "total_absorbed_energy_J": float(
                        self.grid.integrate_volume(self.nonlinear_loss_energy_J_m3)
                    ),
                    "multiphoton_absorbed_energy_J": float(
                        self.grid.integrate_volume(self.multiphoton_loss_energy_J_m3)
                    ),
                    "free_carrier_absorbed_energy_J": float(
                        self.grid.integrate_volume(self.free_carrier_loss_energy_J_m3)
                    ),
                    "maximum_peak_carrier_density_m3": self.maximum_peak_carrier_density_m3,
                    "maximum_fractional_loss_per_step": self.maximum_strong_field_fractional_loss,
                },
                "compressor": dict(self.cfg.get("coherent", {}).get("compressor", {})),
                "transient_gain": self._transient_diagnostics(),
            }
        )
        return {
            "time_s": self.pulse.time_s.copy(),
            "input_temporal_energy_J": input_energy,
            "output_temporal_energy_J": output_energy,
            "model_used": "coherent_cpa",
            "frequency_Hz": self.pulse.frequency_Hz.copy(),
            "input_spectral_phase_rad": np.unwrap(np.angle(input_equivalent)),
            "output_spectral_phase_rad": np.unwrap(np.angle(output_equivalent)),
            "compressed_spectral_phase_rad": np.unwrap(np.angle(compressed_equivalent)),
            "compressed_temporal_power_W": compressed_energy / dt,
            "cpa_diagnostics": diagnostics,
            "nonlinear_loss_energy_J_m3": self.nonlinear_loss_energy_J_m3.copy(),
        }

    def _integrated_spatiotemporal_energy(
        self,
        field: np.ndarray,
        *,
        spectral_phase_rad: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return exact spatially integrated temporal energy without a 4-D copy.

        A single reference-mode phase is not a valid temporal closure once
        Kerr, diffraction, or chromatic propagation makes the output
        spatio-spectral field multimode.  Transforming small y-slabs retains
        the complete ``E(f,x,y)`` interference within every spatial point and
        then integrates intensity, keeping peak memory bounded.
        """

        spectral_spatial = np.asarray(field)
        if spectral_spatial.ndim != 4:
            raise ValueError(
                "coherent spatiotemporal integration expects "
                "(polarization, frequency, y, x)"
            )
        frequency_count = spectral_spatial.shape[1]
        ny = spectral_spatial.shape[2]
        nx = spectral_spatial.shape[3]
        # The FFT backend may allocate an output-sized temporary.  Keep each
        # transformed slab near 64 MiB so large coherent grids remain usable.
        target_complex_bytes = 64 * 1024 * 1024
        rows_per_chunk = max(
            1,
            min(
                ny,
                target_complex_bytes
                // max(frequency_count * nx * np.dtype(complex).itemsize, 1),
            ),
        )
        phase_factor = None
        if spectral_phase_rad is not None:
            phase = np.asarray(spectral_phase_rad, dtype=float)
            if phase.shape != (frequency_count,):
                raise ValueError(
                    "coherent compressor phase must match the frequency grid"
                )
            phase_factor = np.exp(1j * phase)[:, None, None]
        temporal_energy = np.zeros(frequency_count, dtype=float)
        for polarization_index in range(spectral_spatial.shape[0]):
            for row_start in range(0, ny, rows_per_chunk):
                row_stop = min(row_start + rows_per_chunk, ny)
                spectral_chunk = spectral_spatial[
                    polarization_index,
                    :,
                    row_start:row_stop,
                    :,
                ]
                if phase_factor is not None:
                    spectral_chunk = spectral_chunk * phase_factor
                temporal_chunk = self.pulse.spectrum_to_time(
                    spectral_chunk,
                    axis=0,
                )
                temporal_energy += np.sum(
                    np.abs(temporal_chunk) ** 2
                    * self.grid.cell_areas[None, row_start:row_stop, :],
                    axis=(1, 2),
                )
        return temporal_energy

    def _equivalent_output_phase(
        self,
        spectral_spatial_field: np.ndarray,
        output_spectrum_J: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if self.reference_modes is None:
            raise RuntimeError("Coherent reference modes were not initialized")
        overlap = self.grid.integrate_transverse(
            spectral_spatial_field * np.conj(self.reference_modes)
        )
        projection_energy = np.abs(overlap) ** 2
        phase = np.angle(overlap)
        weak = projection_energy <= np.finfo(float).eps * np.maximum(
            output_spectrum_J,
            np.finfo(float).tiny,
        )
        for index in np.flatnonzero(weak):
            local = spectral_spatial_field[index]
            iy, ix = np.unravel_index(int(np.argmax(np.abs(local) ** 2)), local.shape)
            phase[index] = np.angle(local[iy, ix])
        efficiency = float(
            np.sum(projection_energy)
            / max(float(np.sum(output_spectrum_J)), np.finfo(float).tiny)
        )
        return phase, min(max(efficiency, 0.0), 1.0)


class IncoherentHGCoherentSignalPropagator(CoherentSignalPropagator):
    """Coherent CPA propagation for an incoherent explicit HG ensemble.

    Every HG member retains its own complex ``E(f,y,x)`` field.  Linear
    propagation is independent, while modes at one frequency consume the
    local gain reservoir simultaneously.  Kerr, multiphoton absorption, and
    free carriers see the summed instantaneous modal intensity and therefore
    apply one common local material response without coherent cross terms.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        seed_cfg = (
            args[2]
            if len(args) >= 3
            else kwargs.get("seed_cfg", {})
        )
        if bool(
            seed_cfg.get("coherent", {})
            .get("transient_gain", {})
            .get("enabled", False)
        ):
            raise ValueError(
                "coherent_cpa with incoherent_hg currently requires "
                "seed.coherent.transient_gain.enabled=false; explicit HG "
                "supports the whole-fluence shared-reservoir CPA closure"
            )
        super().__init__(*args, **kwargs)
        physical_wavelengths = self.spectrum.coordinate.copy()
        physical_weights = self.spectrum.weights.copy()
        self.incoherent_modes = incoherent_hg_modes(self.cfg)
        self._reported_wavelengths_m = physical_wavelengths.copy()
        self._physical_spectral_count = len(physical_wavelengths)
        self._modal_channels: list[dict[str, Any]] = []
        wavelengths: list[float] = []
        weights: list[float] = []
        for mode_index, mode in enumerate(self.incoherent_modes):
            for spectral_index, wavelength in enumerate(physical_wavelengths):
                self._modal_channels.append(
                    {
                        "mode_index": mode_index,
                        "spectral_index": spectral_index,
                        **mode,
                    }
                )
                wavelengths.append(float(wavelength))
                weights.append(
                    float(mode["weight"])
                    * float(physical_weights[spectral_index])
                )
        self.spectrum = DiscreteProfile(
            np.asarray(wavelengths, dtype=float),
            np.asarray(weights, dtype=float),
        )
        self.joint_time_frequency_weights = None
        self._shared_gain_channel_groups = tuple(
            np.asarray(
                [
                    channel_index
                    for channel_index, channel in enumerate(
                        self._modal_channels
                    )
                    if int(channel["spectral_index"]) == spectral_index
                ],
                dtype=int,
            )
            for spectral_index in range(self._physical_spectral_count)
        )
        # The parent constructed no source or diffraction plan yet, but make
        # the invalidation explicit because this specialization replaces its
        # spectral coordinate with mode x frequency channels.
        self._spectral_entrance_plan = None
        self._diffraction_spectral_state = None
        self._diffraction_transfer_cache.clear()
        self._diffraction_transfer_cache_bytes = 0
        self.reference_modes = None
        self._reference_modes_source = None
        self.bench_warnings.append(
            "Explicit HG CPA uses mutually incoherent spatial modes: each mode "
            "retains coherent frequency/time phase, while modal intensities "
            "share gain, Kerr, multiphoton, and free-carrier material response."
        )

    def _simultaneous_gain_channel_groups(self) -> tuple[np.ndarray, ...]:
        return self._shared_gain_channel_groups

    def _transport_progress_metrics(self) -> dict[str, Any]:
        result = super()._transport_progress_metrics()
        result.update(
            {
                "spatial_mode_model": "incoherent_hg",
                "spatial_mode_count": len(self.incoherent_modes),
                "mode_spectral_channel_count": len(self._modal_channels),
            }
        )
        return result

    def _source_config_for_channel(
        self,
        channel_index: int,
        wavelength_m: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        local, diagnostics = seed_config_at_wavelength(
            self.cfg,
            wavelength_m,
        )
        channel = self._modal_channels[channel_index]
        represented_m2_x = float(local.get("m2_x", 1.0))
        represented_m2_y = float(local.get("m2_y", 1.0))
        local["waist_x_m"] = float(local["waist_x_m"]) / math.sqrt(
            represented_m2_x
        )
        local["waist_y_m"] = float(local["waist_y_m"]) / math.sqrt(
            represented_m2_y
        )
        local["m2_x"] = 1.0
        local["m2_y"] = 1.0
        local["spatial_modes"] = {
            "model": "single_hg_mode",
            "active_order_x": int(channel["order_x"]),
            "active_order_y": int(channel["order_y"]),
        }
        diagnostics["incoherent_mode"] = {
            "mode_index": int(channel["mode_index"]),
            "order_x": int(channel["order_x"]),
            "order_y": int(channel["order_y"]),
            "mode_weight": float(channel["weight"]),
            "represented_input_m2_x": represented_m2_x,
            "represented_input_m2_y": represented_m2_y,
            "fundamental_waist_x_m": float(local["waist_x_m"]),
            "fundamental_waist_y_m": float(local["waist_y_m"]),
        }
        return local, diagnostics

    def _initialize_field(
        self,
        spectral_field: np.ndarray,
        temporal_weights: np.ndarray,
    ) -> np.ndarray:
        del temporal_weights
        expected = len(self._modal_channels)
        if spectral_field.shape[0] != expected:
            raise ValueError(
                "explicit HG CPA source channel count does not match its "
                "mode-frequency basis"
            )
        modal_source = spectral_field.reshape(
            len(self.incoherent_modes),
            self._physical_spectral_count,
            *spectral_field.shape[-2:],
        )
        fundamental_source = modal_source[0]
        if (
            self.reference_modes is None
            or self._reference_modes_source is not spectral_field
        ):
            energy = self.grid.integrate_transverse(
                np.abs(fundamental_source) ** 2
            )
            normalized = fundamental_source / np.sqrt(
                np.maximum(energy, np.finfo(float).tiny)
            )[:, None, None]
            flattened = normalized.reshape(normalized.shape[0], -1)
            peak_indices = np.argmax(np.abs(flattened) ** 2, axis=1)
            anchor_phase = np.angle(
                flattened[np.arange(len(flattened)), peak_indices]
            )
            self.reference_modes = normalized * np.exp(
                -1j * anchor_phase
            )[:, None, None]
            self._reference_modes_source = spectral_field
        phase = np.exp(1j * np.angle(self.pulse.spectral_field_sqrt_J))
        modal_field = modal_source * phase[None, :, None, None]
        return modal_field.reshape(
            1,
            expected,
            *spectral_field.shape[-2:],
        )

    def _coherent_spectral_modes(self, field: np.ndarray) -> np.ndarray:
        values = np.asarray(field)
        expected = len(self._modal_channels)
        if values.ndim != 4 or values.shape[:2] != (1, expected):
            raise ValueError(
                "explicit HG CPA field must have shape "
                "(1, mode*frequency, y, x)"
            )
        return values[0].reshape(
            len(self.incoherent_modes),
            self._physical_spectral_count,
            *values.shape[-2:],
        )

    def _store_coherent_spectral_modes(
        self,
        field: np.ndarray,
        modes: np.ndarray,
    ) -> np.ndarray:
        expected_shape = (
            len(self.incoherent_modes),
            self._physical_spectral_count,
            *field.shape[-2:],
        )
        values = np.asarray(modes)
        if values.shape != expected_shape:
            raise ValueError("explicit HG CPA modal field shape changed")
        field[0] = values.reshape(len(self._modal_channels), *field.shape[-2:])
        return field

    def _aggregate_spectral_channels(
        self,
        values: np.ndarray | None,
    ) -> np.ndarray | None:
        if values is None:
            return None
        source = np.asarray(values, dtype=float)
        if source.shape != (len(self._modal_channels),):
            raise ValueError(
                "explicit HG CPA spectrum must contain one value per "
                "mode-frequency channel"
            )
        return np.sum(
            source.reshape(
                len(self.incoherent_modes),
                self._physical_spectral_count,
            ),
            axis=0,
        )

    def _result_extras(
        self,
        field: np.ndarray,
        source_total_J: float,
        source_spectrum_J: np.ndarray,
        output_spectrum_J: np.ndarray,
    ) -> dict[str, Any]:
        source_spectrum = self._aggregate_spectral_channels(
            source_spectrum_J
        )
        output_spectrum = self._aggregate_spectral_channels(
            output_spectrum_J
        )
        if source_spectrum is None or output_spectrum is None:
            raise RuntimeError("explicit HG CPA spectra were not initialized")
        modal_field = self._coherent_spectral_modes(field)
        result = super()._result_extras(
            modal_field,
            source_total_J,
            source_spectrum,
            output_spectrum,
        )
        diagnostics = dict(result.get("cpa_diagnostics") or {})
        diagnostics.update(
            {
                "spatial_mode_model": "incoherent_hg",
                "spatial_mode_count": len(self.incoherent_modes),
                "mode_spectral_channel_count": len(self._modal_channels),
                "modal_coherence": "incoherent_between_HG_modes",
                "independent_hg_channel_energy": self._independent_hg_channel_energy(
                    source_spectrum_J, output_spectrum_J
                ),
                "modal_linear_transport": "independent_complex_E_frequency_y_x",
                "modal_gain_coupling": (
                    "same_frequency_packets_share_one_local_population_update"
                ),
                "modal_nonlinear_coupling": (
                    "summed_instantaneous_intensity_common_Kerr_MPA_FCA_response"
                ),
                "reported_spectral_phase_scope": (
                    "HG00_projection_proxy; temporal energy and compressor "
                    "power sum every HG mode incoherently"
                ),
            }
        )
        result["cpa_diagnostics"] = diagnostics
        return result

    def _independent_hg_channel_energy(
        self, source_spectrum_J: np.ndarray, output_spectrum_J: np.ndarray
    ) -> dict[str, Any]:
        """Retain channel ancestry before spectra are summed over HG members.

        The spectra already include transverse quadrature. Summing physical
        frequency bins therefore yields pulse energy, without another df or
        spatial weight. These are evolved input channels, not HG projections.
        """
        shape = (len(self.incoherent_modes), self._physical_spectral_count)
        source = np.asarray(source_spectrum_J, dtype=float).reshape(shape).sum(axis=1)
        output = np.asarray(output_spectrum_J, dtype=float).reshape(shape).sum(axis=1)
        source_total, output_total = float(source.sum()), float(output.sum())
        rate = float(self.simulation.get("repetition_rate_Hz", 0.0))
        channels = []
        for mode, input_J, output_J in zip(self.incoherent_modes, source, output):
            nx, ny = int(mode["order_x"]), int(mode["order_y"])
            channels.append({
                "input_mode_label": f"HG{nx}{ny}",
                "order_x": nx,
                "order_y": ny,
                "configured_input_weight": float(mode["weight"]),
                "input_energy_J": float(input_J),
                "output_energy_J": float(output_J),
                "input_power_W": float(input_J * rate) if rate > 0 else None,
                "output_power_W": float(output_J * rate) if rate > 0 else None,
                "input_fraction": float(input_J / source_total) if source_total > 0 else None,
                "output_fraction": float(output_J / output_total) if output_total > 0 else None,
                "energy_gain": float(output_J / input_J) if input_J > 0 else None,
            })
        return {
            "definition": "energy_of_independently_evolved_input_HG_channels_not_output_HG_basis_projection",
            "input_plane": "grid_source",
            "output_plane": "system_output_after_post_gain_optical_train_before_compressor",
            "fraction_normalization": "sum_of_all_independent_channel_energies_at_same_plane",
            "zero_total_fraction": None,
            "repetition_rate_Hz": rate,
            "input_total_energy_J": source_total,
            "output_total_energy_J": output_total,
            "channels": channels,
        }

    def propagate(
        self,
        population: np.ndarray,
        delta_n: np.ndarray,
    ) -> SignalPropagationResult:
        result = super().propagate(population, delta_n)
        result.wavelengths_m = self._reported_wavelengths_m.copy()
        result.input_spectrum_J = self._aggregate_spectral_channels(
            result.input_spectrum_J
        )
        result.output_spectrum_J = self._aggregate_spectral_channels(
            result.output_spectrum_J
        )
        if result.source_spatio_spectral is not None:
            result.source_spatio_spectral.update(
                {
                    "spatial_mode_model": "incoherent_hg",
                    "spatial_mode_count": len(self.incoherent_modes),
                    "mode_spectral_channel_count": len(self._modal_channels),
                    "reported_spectrum_aggregated_over_modes": True,
                    "coherent_frequency_evolution_per_mode": True,
                }
            )
        return result

    def _scan_output_caustic(
        self,
        field: np.ndarray,
        trace: OpticalTrace,
        quantity: str,
        wavelengths_m: np.ndarray,
    ) -> dict[str, Any] | None:
        if not self._diagnostics_enabled() or bool(self.physics.get("_suppress_output_diagnostics", False)):
            return None
        config = self.cfg.get("caustic_diagnostics", {})
        if not bool(config.get("enabled", False)):
            return {
                "enabled": False,
                "application": "diagnostic_only",
                "affects_propagation": False,
            }
        retain = bool(config.get("record_planes", True)) and bool(
            self.physics.get("record_optical_planes", False)
        )
        diagnostics, planes = scan_spectral_caustic(
            field,
            self.grid,
            caustic_positions(config),
            np.asarray(wavelengths_m, dtype=float),
            float(config.get("refractive_index", 1.0)),
            np.ones(len(wavelengths_m), dtype=float),
            np.ones(len(wavelengths_m), dtype=float),
            retain_fields=retain,
        )
        if retain:
            for index, (position, plane) in enumerate(planes):
                trace.add(
                    f"caustic_{index:03d}_z_{position:+.9g}_m",
                    plane,
                    self.grid,
                    quantity,
                    complex_field_available=False,
                )
        diagnostics.update(
            {
                "method": (
                    "coherent_cpa_incoherent_hg_channel_energy_integrated_"
                    "D4sigma_fit"
                ),
                "planes_recorded": bool(retain),
                "physical_wavelengths_m": self._reported_wavelengths_m.tolist(),
                "spatial_mode_count": len(self.incoherent_modes),
                "mode_spectral_channel_count": len(self._modal_channels),
                "per_channel_transport_M2_x": 1.0,
                "per_channel_transport_M2_y": 1.0,
                "represented_beam_quality_source": (
                    "explicit_incoherent_hg_orders_and_weights"
                ),
                "scope": (
                    "Every HG mode keeps a coherent CPA frequency field and "
                    "is propagated with transport M2=1. Caustic planes sum "
                    "mode/frequency energy incoherently, so configured M2 is "
                    "not multiplied into diffraction a second time."
                ),
            }
        )
        return diagnostics

    def _model_used(self) -> str:
        return "coherent_cpa"

    def _gain_transport_diagnostics(self) -> dict[str, Any]:
        result = super()._gain_transport_diagnostics()
        result.update(
            {
                "backend": "coherent_cpa_explicit_incoherent_hg",
                "incoherent_spatial_modes": True,
                "spatial_mode_count": len(self.incoherent_modes),
                "mode_spectral_channel_count": len(self._modal_channels),
                "spectral_bins": self._physical_spectral_count,
                "modal_coupling": (
                    "independent_complex_fields_with_shared_local_material_state"
                ),
                "modal_gain_update": (
                    "sum_same_frequency_local_fluence_then_one_population_update"
                ),
                "modal_nonlinear_update": (
                    "sum_time_resolved_intensity_then_common_Kerr_MPA_FCA_screen"
                ),
            }
        )
        return result
