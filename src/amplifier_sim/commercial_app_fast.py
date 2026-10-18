"""Commercial App-style reduced-order Seed transport.

This backend keeps the existing spectral gain and population bookkeeping, but
replaces sampled in-crystal diffraction and coherent time-domain Kerr stepping
with an astigmatic Gaussian duct.  At every half cell the sampled field is
power-preservingly projected onto its centroid, radii, tilt, and quadratic
phase; the resulting complex q parameters are then advanced analytically.

Thermal and spatial Kerr phases are retained through their equivalent
quadratic Gaussian lens.  Higher spatial moments and gain-guided non-Gaussian
structure are deliberately not part of this model.  An optional incoherent-HG
closure gives every retained mode its own q/energy channel while updating all
modes at one wavelength against the same local population reservoir.  The optional
``effective_area_spm`` closure additionally retains one coherent 1-D envelope
for material/element dispersion, SPM broadening, and compressor diagnostics,
without resolving their transverse variation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .average_signal import AveragePowerSignalPropagator
from .caustic import (
    caustic_positions,
    fit_second_moment_caustic,
    scan_spectral_caustic,
)
from .coherent import CoherentPulse, build_coherent_pulse, intensity_fwhm
from .constants import LIGHT_SPEED, photon_energy
from .directions import axis_direction_components
from .dispersion import material_dispersion_phase, phase_from_coefficients
from .field_diagnostics import (
    normalized_fit_rmse,
    stabilize_quadratic_fit_coefficients,
)
from .optical_system import (
    gain_medium_axis_angles_rad,
    gain_medium_path_cosine,
)
from .optics import gaussian_abcd_to_plane
from .nonlinear import apply_strong_field_step
from .profiles import DiscreteProfile, temporal_profile
from .pulse_definition import rebin_mass
from .seed_spatio_spectral import seed_config_at_wavelength
from .seed_gain_transport import GaussianProjection, project_to_gaussian_mode
from .signal import frantz_nodvik
from .spatial_modes import (
    axis_aligned_q_anisotropy_reasons,
    incoherent_hg_modes,
    single_hg_mode_amplitude,
)
from .spectral_signal import SpectralSignalPropagator


class CommercialAppFastGaussianDuctMixin:
    """Shared Gaussian-duct closure for pulse and cycle-average solvers."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        commercial_app_config = self.cfg.get("commercial_app_fast", {})
        self._commercial_app_temporal_model = str(
            commercial_app_config.get("temporal_model", "fixed_duration")
        )
        self._commercial_app_spectral_feedback = bool(
            commercial_app_config.get("spectral_feedback", True)
        )
        self._commercial_app_include_intrapulse_relaxation = bool(
            commercial_app_config.get("include_relaxation_during_pulse", False)
        )
        self._commercial_app_gain_temporal_bin_count = int(
            commercial_app_config.get("gain_temporal_bin_count", 129)
        )
        self._commercial_app_carrier_periodic_relative_tolerance = float(
            commercial_app_config.get("carrier_periodic_relative_tolerance", 1e-6)
        )
        self._commercial_app_carrier_periodic_maximum_iterations = int(
            commercial_app_config.get("carrier_periodic_maximum_iterations", 32)
        )
        self._commercial_app_thermal_lens_config = dict(
            commercial_app_config.get("thermal_lens", {})
        )
        self._commercial_app_thermal_lens_model = str(
            self._commercial_app_thermal_lens_config.get("model", "sampled_phase")
        )
        self._commercial_app_thermal_lens_quadratic_model = str(
            self._commercial_app_thermal_lens_config.get(
                "quadratic_model", "full_xy"
            )
        )
        self._commercial_app_axis_coupling_ratio_limit = float(
            self.simulation.get("strict_validation", {}).get(
                "maximum_commercial_app_axis_coupling_ratio", 1e-3
            )
        )
        self._commercial_app_input_anisotropy_reasons = (
            axis_aligned_q_anisotropy_reasons(self.cfg)
        )
        self._commercial_app_input_axis_aligned_q_applicable = bool(
            not self._commercial_app_input_anisotropy_reasons
            or np.isclose(
                float(self.cfg.get("rotation_deg", 0.0)) % 180.0,
                0.0,
                atol=1e-12,
            )
        )
        geometry = self.material.get("geometry", {})
        self._commercial_app_thermal_lens_center_x_m = float(
            geometry.get("center_x_m", 0.0)
            if self._commercial_app_thermal_lens_config.get("center_x_m") is None
            else self._commercial_app_thermal_lens_config["center_x_m"]
        )
        self._commercial_app_thermal_lens_center_y_m = float(
            geometry.get("center_y_m", 0.0)
            if self._commercial_app_thermal_lens_config.get("center_y_m") is None
            else self._commercial_app_thermal_lens_config["center_y_m"]
        )
        self._commercial_app_thermal_lens_radius_x_m = self._commercial_app_thermal_lens_config.get(
            "radius_x_m"
        )
        self._commercial_app_thermal_lens_radius_y_m = self._commercial_app_thermal_lens_config.get(
            "radius_y_m"
        )
        if self._commercial_app_thermal_lens_radius_x_m is not None:
            self._commercial_app_thermal_lens_radius_x_m = float(
                self._commercial_app_thermal_lens_radius_x_m
            )
        if self._commercial_app_thermal_lens_radius_y_m is not None:
            self._commercial_app_thermal_lens_radius_y_m = float(
                self._commercial_app_thermal_lens_radius_y_m
            )
        self._commercial_app_is_cw = (
            str(self.cfg.get("temporal", {}).get("type", "gaussian")) == "cw"
        )
        if self._commercial_app_is_cw:
            # CW has no coherent time envelope.  Keep a one-sample carrier only
            # so shared diagnostics have stable array shapes; importantly, do
            # not parse pulse/phase files that are physically inactive in CW.
            center_wavelength = float(self.cfg["wavelength_m"])
            carrier_amplitude = math.sqrt(
                max(float(self.cfg.get("pulse_energy_J", 0.0)), 0.0)
            )
            pulse = CoherentPulse(
                frequency_Hz=np.asarray(
                    [LIGHT_SPEED / center_wavelength], dtype=float
                ),
                wavelength_m=np.asarray([center_wavelength], dtype=float),
                spectral_field_sqrt_J=np.asarray(
                    [carrier_amplitude], dtype=complex
                ),
                time_s=np.asarray([0.0], dtype=float),
                temporal_field_sqrt_J=np.asarray(
                    [carrier_amplitude], dtype=complex
                ),
                warnings=(),
                diagnostics={
                    "input_mode": "not_applicable_cw",
                    "sample_count": 1,
                },
            )
        else:
            pulse = build_coherent_pulse(self.cfg)
        self._commercial_app_feedback_grid_contract: dict[str, Any] = {
            "mode": "native_seed_gain_grid",
            "requested_bin_count": int(
                commercial_app_config.get("spectral_feedback_bin_count", 65)
            ),
            "original_bin_count": int(len(self.spectrum.coordinate)),
            "executed_bin_count": int(len(self.spectrum.coordinate)),
        }
        if (
            self._commercial_app_temporal_model == "effective_area_spm"
            and self._commercial_app_spectral_feedback
            and not self._commercial_app_is_cw
        ):
            self._configure_commercial_app_spectral_feedback_grid(
                pulse,
                int(commercial_app_config.get("spectral_feedback_bin_count", 65)),
                mode=str(
                    commercial_app_config.get(
                        "spectral_feedback_grid_mode",
                        "full_coherent_span",
                    )
                ),
            )
        self._commercial_app_hg_enabled = (
            str(self.cfg.get("spatial_modes", {}).get("model", "effective_m2"))
            == "incoherent_hg"
        )
        self._commercial_app_incoherent_modes: list[dict[str, float | int]] = []
        self._commercial_app_modal_channels: list[dict[str, Any]] = []
        self._commercial_app_reported_gain_wavelengths_m = self.spectrum.coordinate.copy()
        self._commercial_app_reported_gain_spectral_weights = self.spectrum.weights.copy()
        if self._commercial_app_hg_enabled:
            self._configure_commercial_app_hg_channels()
        self._commercial_app_pulse = pulse
        self._commercial_app_beam_quality_model = str(
            commercial_app_config.get("beam_quality_model", "configured_m2")
        )
        self._commercial_app_input_coherent_spectrum = np.asarray(
            pulse.spectral_field_sqrt_J, dtype=complex
        ).copy()
        if (
            self._commercial_app_temporal_model == "effective_area_spm"
            and self._commercial_app_spectral_feedback
            and not self._commercial_app_is_cw
        ):
            # ``_configure_commercial_app_spectral_feedback_grid`` deliberately prunes
            # numerically negligible source channels.  Apply that exact
            # support contract to the coherent envelope as well; otherwise
            # the first synchronization sees energy in a channel whose coarse
            # source weight is zero and silently removes it.  Besides making a
            # nominally linear pass non-identical, that mismatch also makes
            # wavelength-dependent entrance filtering ambiguous at the grid
            # edges.
            source_energy = float(
                np.sum(np.abs(self._commercial_app_input_coherent_spectrum) ** 2)
            )
            source_indices = self._commercial_app_high_resolution_bin_indices()
            supported = (
                np.asarray(self.spectrum.weights, dtype=float)[source_indices]
                > np.finfo(float).tiny
            )
            self._commercial_app_input_coherent_spectrum[~supported] = 0.0
            retained_energy = float(
                np.sum(np.abs(self._commercial_app_input_coherent_spectrum) ** 2)
            )
            if retained_energy <= np.finfo(float).tiny:
                raise ValueError(
                    "Commercial App coherent feedback support removed the complete "
                    "input spectrum"
                )
            self._commercial_app_input_coherent_spectrum *= math.sqrt(
                source_energy / retained_energy
            )
        pulse_energy = float(np.sum(pulse.temporal_energy_J))
        peak_power = float(np.max(pulse.temporal_power_W))
        configured_duration = float(self.cfg.get("pulse_duration_s", 1e-12))
        self._commercial_app_effective_duration_s = (
            configured_duration
            if self._commercial_app_is_cw
            else pulse_energy / peak_power
            if pulse_energy > 0.0 and peak_power > 0.0
            else configured_duration
        )
        self._commercial_app_input_fwhm_s = pulse.diagnostics.get("input_fwhm_s")
        self._commercial_app_active_repetition_rate_Hz: float | None = None
        if self._commercial_app_is_cw:
            temporal_warning = (
                "Commercial App CW mode uses the transported average intensity directly for "
                "its equivalent Kerr lens. It does not synthesize a pulse from "
                "pulse_duration_s or report SPM broadening/compressor diagnostics."
            )
        elif self._commercial_app_include_intrapulse_relaxation:
            temporal_warning = (
                "Commercial App fast mode resolves ordered gain-time fluence bins and advances "
                "the configured level kinetics between them. All time bins share one "
                "power-preserving Gaussian-q spatial mode per wavelength; retarded "
                "time-of-flight and space-time coupling remain reduced."
            )
        elif self._commercial_app_effective_area_strong_field_enabled():
            temporal_warning = (
                "Commercial App fast mode uses a one-dimensional coherent envelope and a "
                "Gaussian effective-area moment closure for multiphoton loss, "
                "causal carrier generation, FCA, and carrier refraction. The "
                "local inter-pulse carrier fixed point is solved directly; radial "
                "carrier gradients and space-time coupling remain unresolved."
            )
        elif self._commercial_app_temporal_model == "effective_area_spm":
            temporal_warning = (
                "Commercial App fast mode uses a one-dimensional coherent envelope with a "
                "Gaussian-mode effective area for SPM and material dispersion. It "
                "does not resolve space-time coupling, radial variation of spectral "
                "broadening, self-steepening, or irreversible strong-field loss."
            )
        else:
            temporal_warning = (
                "Commercial App fast mode uses one fixed input effective duration to close "
                "peak intensity; gain narrowing, SPM spectral broadening, and "
                "compressor output are not time-resolved."
            )
        self._commercial_app_persistent_warnings = [
            *pulse.warnings,
            temporal_warning,
            (
                "Commercial App fast mode projects the field onto an axis-aligned astigmatic "
                "Gaussian after each in-crystal half step. Thermal phase, Kerr "
                "phase, and gain guiding survive only through centroid, radius, "
                "tilt, and quadratic-curvature moments."
            ),
        ]
        if self._commercial_app_hg_enabled:
            self._commercial_app_persistent_warnings.append(
                "Commercial App fast propagates the configured incoherent Hermite-Gaussian "
                "channels independently with one qx/qy state per mode. Modes at "
                "the same wavelength are gain-updated simultaneously against one "
                "shared local population reservoir and are summed incoherently."
            )
        if (
            self._commercial_app_temporal_envelope_enabled()
            and self._commercial_app_spectral_feedback
            and len(self.spectrum.coordinate) < 2
        ):
            self._commercial_app_persistent_warnings.append(
                "Commercial App spectral feedback is requested, but the coarse Seed gain "
                "spectrum has only one bin. The high-resolution SPM/dispersion "
                "output remains available, but its spectral redistribution cannot "
                "feed wavelength-dependent gain or population depletion. Use at "
                "least two (preferably several) seed.spectrum samples for feedback."
            )
        # The legacy reduced model has one whole-pulse fluence packet.  When
        # intrapulse relaxation is explicitly enabled, keep a bounded set of
        # ordered energy packets from the configured temporal profile.  They
        # share the configured Gaussian-q channel set and local population, so
        # fidelity scales with time bins, HG channels, and level kinetics rather
        # than x-y-t diffraction fields.
        if self._commercial_app_include_intrapulse_relaxation:
            source_domain = str(
                self.cfg.get("pulse_definition", {}).get(
                    "domain", "independent"
                )
            )
            temporal_source = self.temporal
            if (
                source_domain == "independent"
                and str(
                    self.cfg.get("temporal", {}).get("type", "gaussian")
                )
                != "file"
            ):
                temporal_cfg = dict(self.cfg.get("temporal", {}))
                temporal_cfg["sample_count"] = self._commercial_app_gain_temporal_bin_count
                temporal_source = temporal_profile(
                    temporal_cfg,
                    float(self.cfg["pulse_duration_s"]),
                )
            time_s, temporal_weights = rebin_mass(
                temporal_source.coordinate,
                temporal_source.weights,
                self._commercial_app_gain_temporal_bin_count,
            )
            self.temporal = DiscreteProfile(time_s, temporal_weights)
        else:
            self.temporal = DiscreteProfile(
                np.asarray([0.0], dtype=float),
                np.asarray([1.0], dtype=float),
            )
        self.joint_time_frequency_weights = None
        self._reset_commercial_app_diagnostics()

    def _configure_commercial_app_hg_channels(self) -> None:
        """Expand physical wavelengths into independently propagated HG channels."""

        original_wavelengths = self.spectrum.coordinate.copy()
        original_weights = self.spectrum.weights.copy()
        self._commercial_app_incoherent_modes = incoherent_hg_modes(self.cfg)
        wavelengths: list[float] = []
        weights: list[float] = []
        for mode_index, mode in enumerate(self._commercial_app_incoherent_modes):
            for spectral_index, wavelength in enumerate(original_wavelengths):
                self._commercial_app_modal_channels.append(
                    {
                        "mode_index": int(mode_index),
                        "spectral_index": int(spectral_index),
                        "wavelength_m": float(wavelength),
                        **mode,
                    }
                )
                wavelengths.append(float(wavelength))
                weights.append(
                    float(mode["weight"]) * float(original_weights[spectral_index])
                )
        self.spectrum = DiscreteProfile(
            np.asarray(wavelengths, dtype=float),
            np.asarray(weights, dtype=float),
        )
        self.joint_time_frequency_weights = None

    def _source_config_for_channel(
        self, channel_index: int, wavelength_m: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        local, diagnostics = seed_config_at_wavelength(self.cfg, wavelength_m)
        if not self._commercial_app_hg_enabled:
            return local, diagnostics
        channel = self._commercial_app_modal_channels[channel_index]
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

    def _commercial_app_channel_orders(self, channel_index: int) -> tuple[int, int]:
        if not self._commercial_app_hg_enabled:
            return 0, 0
        channel = self._commercial_app_modal_channels[channel_index]
        return int(channel["order_x"]), int(channel["order_y"])

    def _commercial_app_channel_m2(self, channel_index: int) -> tuple[float, float]:
        if self._commercial_app_hg_enabled:
            order_x, order_y = self._commercial_app_channel_orders(channel_index)
            return float(2 * order_x + 1), float(2 * order_y + 1)
        configs = getattr(
            self,
            "_wavelength_source_configs",
            [self.cfg] * len(self.spectrum.coordinate),
        )
        local = configs[channel_index]
        return float(local.get("m2_x", 1.0)), float(local.get("m2_y", 1.0))

    def _configure_commercial_app_spectral_feedback_grid(
        self,
        pulse: CoherentPulse,
        requested_bin_count: int,
        *,
        mode: str = "full_coherent_span",
    ) -> None:
        """Give generated SPM energy explicit coarse transport channels.

        Rebinning only the nonzero *input* support makes every newly generated
        out-of-band component fall into the nearest edge gain bin.  For the
        Commercial App effective-area closure, construct a uniform-frequency feedback
        grid over the full coherent FFT span instead.  Zero-energy source
        channels are valid and can later receive SPM energy without inventing
        an edge-bin material response.
        """

        original_count = int(len(self.spectrum.coordinate))
        count = min(
            len(pulse.frequency_Hz),
            max(3, original_count, int(requested_bin_count)),
        )
        pulse_frequency = np.asarray(pulse.frequency_Hz, dtype=float)
        pulse_energy = np.asarray(pulse.spectral_energy_J, dtype=float)
        total_pulse_energy = float(np.sum(pulse_energy))
        if total_pulse_energy <= np.finfo(float).tiny:
            raise ValueError(
                "Commercial App coherent feedback grid has zero input spectral energy"
            )
        covered_input_energy_fraction = 1.0
        if mode == "seed_energy_support":
            # The coherent FFT necessarily contains a very broad numerical
            # frequency window.  A small Commercial App-style gain grid spread over that
            # complete window can miss a narrow material band altogether (for
            # example a 40-bin, 0.5-ps Yb:YAG calculation around 1030 nm).
            # Retain all but 1e-12 of the configured input energy and add a 25%
            # guard band on each side for SPM-generated energy.  Leakage beyond
            # this declared support remains an explicit runtime diagnostic.
            order = np.argsort(pulse_frequency)
            sorted_frequency = pulse_frequency[order]
            sorted_energy = pulse_energy[order]
            cumulative = np.cumsum(sorted_energy)
            tail_energy = 5e-13 * total_pulse_energy
            lower_index = int(np.searchsorted(cumulative, tail_energy))
            upper_index = int(
                np.searchsorted(
                    cumulative,
                    total_pulse_energy - tail_energy,
                )
            )
            lower_index = min(max(lower_index, 0), len(sorted_frequency) - 1)
            upper_index = min(
                max(upper_index, lower_index), len(sorted_frequency) - 1
            )
            core_minimum = float(sorted_frequency[lower_index])
            core_maximum = float(sorted_frequency[upper_index])
            core_span = core_maximum - core_minimum
            if core_span <= 0.0:
                configured_frequency = LIGHT_SPEED / np.asarray(
                    self.spectrum.coordinate, dtype=float
                )
                core_minimum = float(np.min(configured_frequency))
                core_maximum = float(np.max(configured_frequency))
                core_span = core_maximum - core_minimum
            padding = 0.25 * max(core_span, np.finfo(float).eps)
            minimum_frequency = max(
                float(np.min(pulse_frequency)), core_minimum - padding
            )
            maximum_frequency = min(
                float(np.max(pulse_frequency)), core_maximum + padding
            )
            feedback_mode = "uniform_frequency_seed_energy_support"
        elif mode == "full_coherent_span":
            minimum_frequency = float(np.min(pulse_frequency))
            maximum_frequency = float(np.max(pulse_frequency))
            feedback_mode = "uniform_frequency_full_coherent_span"
        else:
            raise ValueError(
                "Commercial App spectral feedback grid mode must be "
                "full_coherent_span or seed_energy_support"
            )
        frequency = np.linspace(minimum_frequency, maximum_frequency, count)
        if count == 1:
            indices = np.zeros(len(pulse_frequency), dtype=int)
        else:
            edges = 0.5 * (frequency[:-1] + frequency[1:])
            indices = np.searchsorted(edges, pulse_frequency)
        represented = (
            (pulse_frequency >= minimum_frequency)
            & (pulse_frequency <= maximum_frequency)
        )
        weights_frequency = np.bincount(
            indices[represented],
            weights=pulse_energy[represented],
            minlength=count,
        ).astype(float)
        represented_energy = float(np.sum(weights_frequency))
        covered_input_energy_fraction = represented_energy / total_pulse_energy
        # Make the guard channels genuinely empty at initialization instead of
        # retaining floating-point Gaussian tails with no physical weight.
        # A 1e-10 energy fraction is far below the quantitative accuracy of the
        # reduced gain grid, while retaining it can turn harmless analytic
        # Gaussian tails outside a finite material table into a false blocking
        # range violation.
        negligible = weights_frequency < 1e-10 * total_pulse_energy
        pruned_energy = float(np.sum(weights_frequency[negligible]))
        weights_frequency[negligible] = 0.0
        total = float(np.sum(weights_frequency))
        if total <= np.finfo(float).tiny:
            raise ValueError(
                "Commercial App coherent feedback grid has zero input spectral energy"
            )
        wavelength = LIGHT_SPEED / frequency
        order = np.argsort(wavelength)
        self.spectrum = DiscreteProfile(
            wavelength[order],
            weights_frequency[order] / total,
        )
        self.joint_time_frequency_weights = None
        self._commercial_app_feedback_grid_contract = {
            "mode": feedback_mode,
            "requested_bin_count": int(requested_bin_count),
            "original_bin_count": original_count,
            "executed_bin_count": count,
            "minimum_frequency_Hz": float(np.min(frequency)),
            "maximum_frequency_Hz": float(np.max(frequency)),
            "full_coherent_minimum_frequency_Hz": float(
                np.min(pulse_frequency)
            ),
            "full_coherent_maximum_frequency_Hz": float(
                np.max(pulse_frequency)
            ),
            "covered_input_energy_fraction": covered_input_energy_fraction,
            "pruned_input_energy_fraction": pruned_energy
            / total_pulse_energy,
            "zero_energy_source_channels": int(
                np.count_nonzero(weights_frequency <= np.finfo(float).tiny)
            ),
        }

    def _reset_commercial_app_diagnostics(self) -> None:
        self._commercial_app_projection_count = 0
        self._commercial_app_projection_fallback_count = 0
        self._commercial_app_projection_fallback_reasons: list[str] = []
        self._commercial_app_minimum_projection_overlap = 1.0
        self._commercial_app_maximum_single_step_b_integral_rad = 0.0
        self._commercial_app_maximum_cumulative_b_integral_rad = 0.0
        self._commercial_app_maximum_peak_intensity_W_m2 = 0.0
        self._commercial_app_cumulative_b_integral_rad = np.zeros(
            (len(self.grid.y), len(self.grid.x)), dtype=float
        )
        self._commercial_app_runtime_warnings: list[str] = []
        self._commercial_app_coherent_spectrum = (
            self._commercial_app_input_coherent_spectrum.copy()
        )
        self._commercial_app_spm_cumulative_phase_rad = np.zeros(
            len(self._commercial_app_pulse.time_s), dtype=float
        )
        self._commercial_app_maximum_single_step_spm_phase_rad = 0.0
        self._commercial_app_maximum_cumulative_spm_phase_rad = 0.0
        self._commercial_app_maximum_spectral_support_leakage_fraction = 0.0
        self._commercial_app_maximum_spectral_sync_error_fraction = 0.0
        self._commercial_app_donor_initialized_channel_count = 0
        self._commercial_app_maximum_donor_initialized_energy_fraction = 0.0
        self._commercial_app_current_effective_duration_s = float(
            self._commercial_app_effective_duration_s
        )
        self._commercial_app_previous_coarse_energy_J: np.ndarray | None = None
        self._commercial_app_modal_output_energy_J: np.ndarray | None = None
        self._commercial_app_report_wavelengths_m: np.ndarray | None = None
        self._commercial_app_report_input_spectrum_J: np.ndarray | None = None
        self._commercial_app_report_output_spectrum_J: np.ndarray | None = None
        self._commercial_app_intrapulse_dark_step_count = 0
        self._commercial_app_intrapulse_relaxation_span_s = 0.0
        channel_count = len(self.spectrum.coordinate)
        if self._commercial_app_hg_enabled:
            modal_m2 = np.asarray(
                [self._commercial_app_channel_m2(index) for index in range(channel_count)],
                dtype=float,
            )
            self._commercial_app_configured_m2_x = modal_m2[:, 0]
            self._commercial_app_configured_m2_y = modal_m2[:, 1]
        else:
            base_m2_x = float(self.cfg.get("m2_x", 1.0))
            base_m2_y = float(self.cfg.get("m2_y", 1.0))
            self._commercial_app_configured_m2_x = np.full(channel_count, base_m2_x)
            self._commercial_app_configured_m2_y = np.full(channel_count, base_m2_y)
        self._commercial_app_proxy_m2_squared_x = self._commercial_app_configured_m2_x**2
        self._commercial_app_proxy_m2_squared_y = self._commercial_app_configured_m2_y**2
        self._commercial_app_maximum_discarded_m2_x = np.ones(channel_count)
        self._commercial_app_maximum_discarded_m2_y = np.ones(channel_count)
        self._commercial_app_last_discarded_m2_x = np.ones(channel_count)
        self._commercial_app_last_discarded_m2_y = np.ones(channel_count)
        self._commercial_app_m2_channels_initialized = False
        self._commercial_app_thermal_lens_fit_cache: dict[int, np.ndarray] = {}
        self._commercial_app_thermal_lens_fit_records: dict[int, dict[str, Any]] = {}
        self._commercial_app_thermal_lens_fit_fallback_count = 0
        self._commercial_app_axis_coupling_warning_emitted = False
        self._commercial_app_strong_field_total_absorbed_energy_J = 0.0
        self._commercial_app_strong_field_multiphoton_absorbed_energy_J = 0.0
        self._commercial_app_strong_field_free_carrier_absorbed_energy_J = 0.0
        self._commercial_app_strong_field_maximum_peak_carrier_density_m3 = 0.0
        self._commercial_app_strong_field_maximum_fractional_loss = 0.0
        self._commercial_app_strong_field_maximum_periodic_residual = 0.0
        self._commercial_app_strong_field_maximum_periodic_iterations = 0
        self._commercial_app_strong_field_periodic_solve_count = 0
        self._commercial_app_strong_field_loss_energy_J_m3 = np.zeros(
            self.grid.shape, dtype=float
        )

    def _model_used(self) -> str:
        return "commercial_app_fast"

    def _commercial_app_effective_area_strong_field_enabled(self) -> bool:
        return bool(
            self.physics.get("nonlinear_loss_model", "off")
            == "strong_field_effective_area"
            and self._commercial_app_temporal_model == "effective_area_spm"
            and not self._commercial_app_is_cw
        )

    def _commercial_app_temporal_envelope_enabled(self) -> bool:
        return bool(
            self._commercial_app_temporal_model == "effective_area_spm"
            and not self._commercial_app_is_cw
        )

    def _commercial_app_thermal_lens_diagnostics(self) -> dict[str, Any]:
        records = list(self._commercial_app_thermal_lens_fit_records.values())
        valid_records = [row for row in records if bool(row.get("valid", False))]
        focal_x: float | None = None
        focal_y: float | None = None
        curvature_x = 0.0
        curvature_y = 0.0
        curvature_xy = 0.0
        maximum_slice_axis_coupling_ratio = 0.0
        principal_curvatures: list[float] | None = None
        principal_axis_angle_deg: float | None = None
        integrated_axis_coupling_ratio: float | None = None
        if valid_records:
            z_widths = np.asarray(self.grid.z_widths, dtype=float)
            for record in valid_records:
                z_index = int(record["z_index"])
                local_x = 2.0 * float(record["quadratic_x_per_m2"])
                local_y = 2.0 * float(record["quadratic_y_per_m2"])
                local_xy = float(record["quadratic_xy_per_m2"])
                maximum_slice_axis_coupling_ratio = max(
                    maximum_slice_axis_coupling_ratio,
                    abs(local_xy)
                    / max(
                        abs(local_x),
                        abs(local_y),
                        np.finfo(float).tiny,
                    ),
                )
                curvature_x += (
                    local_x * float(z_widths[z_index])
                )
                curvature_y += (
                    local_y * float(z_widths[z_index])
                )
                curvature_xy += local_xy * float(z_widths[z_index])
            if abs(curvature_x) > np.finfo(float).eps:
                focal_x = -1.0 / curvature_x
            if abs(curvature_y) > np.finfo(float).eps:
                focal_y = -1.0 / curvature_y
            hessian = np.asarray(
                [[curvature_x, curvature_xy], [curvature_xy, curvature_y]],
                dtype=float,
            )
            eigenvalues, eigenvectors = np.linalg.eigh(hessian)
            principal_curvatures = eigenvalues.tolist()
            principal_axis_angle_deg = math.degrees(
                math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
            )
            integrated_axis_coupling_ratio = abs(curvature_xy) / max(
                abs(curvature_x),
                abs(curvature_y),
                np.finfo(float).tiny,
            )
        axis_aligned_q_applicable = bool(
            maximum_slice_axis_coupling_ratio
            <= self._commercial_app_axis_coupling_ratio_limit
        )
        return {
            "model": self._commercial_app_thermal_lens_model,
            "affects_propagation": self._commercial_app_thermal_lens_model
            == "local_parabolic",
            "fit_radius_mode": str(
                self._commercial_app_thermal_lens_config.get(
                    "fit_radius_mode", "manual"
                )
            ),
            "quadratic_model": self._commercial_app_thermal_lens_quadratic_model,
            "fit_radius_x_m": self._commercial_app_thermal_lens_radius_x_m,
            "fit_radius_y_m": self._commercial_app_thermal_lens_radius_y_m,
            "fit_center_x_m": self._commercial_app_thermal_lens_center_x_m,
            "fit_center_y_m": self._commercial_app_thermal_lens_center_y_m,
            "fitted_slice_count": len(valid_records),
            "fallback_slice_count": int(
                self._commercial_app_thermal_lens_fit_fallback_count
            ),
            "maximum_fit_normalized_rmse": (
                None
                if not valid_records
                else max(float(row["normalized_rmse"]) for row in valid_records)
            ),
            "single_pass_equivalent_focal_x_m": focal_x,
            "single_pass_equivalent_focal_y_m": focal_y,
            "single_pass_quadratic_hessian_per_m": [
                [curvature_x, curvature_xy],
                [curvature_xy, curvature_y],
            ],
            "single_pass_principal_curvatures_per_m": principal_curvatures,
            "single_pass_principal_axis_angle_deg": principal_axis_angle_deg,
            "integrated_axis_coupling_ratio": integrated_axis_coupling_ratio,
            "maximum_slice_axis_coupling_ratio": (
                maximum_slice_axis_coupling_ratio
            ),
            "maximum_axis_coupling_ratio": (
                self._commercial_app_axis_coupling_ratio_limit
            ),
            "axis_aligned_q_applicable": axis_aligned_q_applicable,
            "scope": (
                "Each sampled delta-n z slice is replaced during propagation by "
                "an area-weighted quadratic fit over the configured ellipse."
                if self._commercial_app_thermal_lens_model == "local_parabolic"
                else "The sampled delta-n phase screen is propagated without a local fit."
            ),
        }

    def _commercial_app_gain_transport_diagnostics(self) -> dict[str, Any]:
        nonlinear_enabled = (
            self.physics.get("nonlinear_phase_model", "off")
            == "kerr_parabolic_lens"
            and float(self.material.get("n2_m2_W", 0.0)) != 0.0
        )
        repetition_rate = self._commercial_app_active_repetition_rate_Hz
        coherent_envelope_resolved = self._commercial_app_temporal_envelope_enabled()
        gain_time_resolved = bool(
            self._commercial_app_include_intrapulse_relaxation
            and len(self.temporal.coordinate) > 1
        )
        temporal_resolved = coherent_envelope_resolved or gain_time_resolved
        temporal_spacing = (
            np.diff(self.temporal.coordinate)
            if gain_time_resolved
            else np.asarray([], dtype=float)
        )
        weights = (
            np.asarray(self._commercial_app_modal_output_energy_J, dtype=float)
            if self._commercial_app_hg_enabled
            and self._commercial_app_modal_output_energy_J is not None
            else np.asarray(self._commercial_app_previous_coarse_energy_J, dtype=float)
            if self._commercial_app_previous_coarse_energy_J is not None
            else np.asarray(self.spectrum.weights, dtype=float)
        )
        weights = weights / max(float(np.sum(weights)), np.finfo(float).tiny)
        proxy_m2_x = np.sqrt(np.maximum(self._commercial_app_proxy_m2_squared_x, 1.0))
        proxy_m2_y = np.sqrt(np.maximum(self._commercial_app_proxy_m2_squared_y, 1.0))
        if self._commercial_app_hg_enabled:
            # Mutually incoherent HG channels add second moments, so their
            # beam-quality factors are energy-weighted linearly.  The RMS
            # wavelength closure used by the single-mode surrogate would
            # count the modal M2 spread a second time.
            effective_m2_x = max(float(np.sum(weights * proxy_m2_x)), 1.0)
            effective_m2_y = max(float(np.sum(weights * proxy_m2_y)), 1.0)
        else:
            effective_m2_x = max(
                float(np.sqrt(np.sum(weights * proxy_m2_x**2))), 1.0
            )
            effective_m2_y = max(
                float(np.sqrt(np.sum(weights * proxy_m2_y**2))), 1.0
            )
        if self._commercial_app_hg_enabled:
            physical_count = len(self._commercial_app_reported_gain_wavelengths_m)

            def modal_average(
                values: np.ndarray, channel_weights: np.ndarray
            ) -> list[float]:
                aggregated: list[float] = []
                for spectral_index in range(physical_count):
                    indices = np.asarray(
                        [
                            index
                            for index, channel in enumerate(
                                self._commercial_app_modal_channels
                            )
                            if int(channel["spectral_index"]) == spectral_index
                        ],
                        dtype=int,
                    )
                    local_weights = channel_weights[indices]
                    local_weights = local_weights / max(
                        float(np.sum(local_weights)), np.finfo(float).tiny
                    )
                    aggregated.append(
                        float(np.sum(local_weights * np.asarray(values)[indices]))
                    )
                return aggregated

            source_weights = np.asarray(self.spectrum.weights, dtype=float)
            configured_m2_x_report = modal_average(
                self._commercial_app_configured_m2_x, source_weights
            )
            configured_m2_y_report = modal_average(
                self._commercial_app_configured_m2_y, source_weights
            )
            proxy_m2_x_report = modal_average(proxy_m2_x, weights)
            proxy_m2_y_report = modal_average(proxy_m2_y, weights)
            discarded_m2_x_report = modal_average(
                self._commercial_app_maximum_discarded_m2_x, weights
            )
            discarded_m2_y_report = modal_average(
                self._commercial_app_maximum_discarded_m2_y, weights
            )
        else:
            configured_m2_x_report = self._commercial_app_configured_m2_x.tolist()
            configured_m2_y_report = self._commercial_app_configured_m2_y.tolist()
            proxy_m2_x_report = proxy_m2_x.tolist()
            proxy_m2_y_report = proxy_m2_y.tolist()
            discarded_m2_x_report = self._commercial_app_maximum_discarded_m2_x.tolist()
            discarded_m2_y_report = self._commercial_app_maximum_discarded_m2_y.tolist()
        return {
            "backend": "commercial_app_fast_gaussian_duct",
            "status": "applied",
            "representation": (
                "time_binned_spectral_fluence_plus_shared_astigmatic_gaussian_q"
                if gain_time_resolved
                else
                "spectral_fluence_plus_astigmatic_gaussian_q_plus_1d_coherent_envelope"
                if coherent_envelope_resolved
                else "spectral_fluence_plus_astigmatic_gaussian_q"
            ),
            "input_axis_aligned_q_applicable": (
                self._commercial_app_input_axis_aligned_q_applicable
            ),
            "input_axis_aligned_q_anisotropy_reasons": list(
                self._commercial_app_input_anisotropy_reasons
            ),
            "time_resolved": temporal_resolved,
            "temporal_bins": (
                len(self.temporal.coordinate)
                if gain_time_resolved
                else len(self._commercial_app_pulse.time_s)
                if coherent_envelope_resolved
                else 1
            ),
            "gain_temporal_bins": len(self.temporal.coordinate),
            "spectral_bins": len(self.spectrum.coordinate),
            "temporal_closure": (
                "cw_average_intensity"
                if self._commercial_app_is_cw
                else
                "time_binned_intrapulse_population_relaxation"
                if gain_time_resolved
                else
                "one_dimensional_effective_area_strong_field"
                if self._commercial_app_effective_area_strong_field_enabled()
                else
                "one_dimensional_effective_area_spm"
                if coherent_envelope_resolved
                else "fixed_input_effective_duration"
            ),
            "include_relaxation_during_pulse": gain_time_resolved,
            "configured_gain_temporal_bin_count": int(
                self._commercial_app_gain_temporal_bin_count
            ),
            "minimum_gain_temporal_step_s": (
                None
                if temporal_spacing.size == 0
                else float(np.min(temporal_spacing))
            ),
            "maximum_gain_temporal_step_s": (
                None
                if temporal_spacing.size == 0
                else float(np.max(temporal_spacing))
            ),
            "gain_temporal_span_s": (
                0.0
                if not gain_time_resolved
                else float(
                    self.temporal.coordinate[-1]
                    - self.temporal.coordinate[0]
                )
            ),
            "intrapulse_dark_step_count": int(
                self._commercial_app_intrapulse_dark_step_count
            ),
            "intrapulse_accumulated_dark_time_s": float(
                self._commercial_app_intrapulse_relaxation_span_s
            ),
            "intrapulse_accumulated_dark_time_scope": (
                "sum_of_local_dark_operator_intervals_over_all_z_cells_and_passes; "
                "not_elapsed_pulse_time"
            ),
            "fixed_input_effective_duration_s": (
                None
                if self._commercial_app_is_cw
                else float(self._commercial_app_effective_duration_s)
            ),
            "input_fwhm_s": (
                None
                if self._commercial_app_is_cw or self._commercial_app_input_fwhm_s is None
                else float(self._commercial_app_input_fwhm_s)
            ),
            "input_peak_power_W": (
                None
                if self._commercial_app_is_cw
                else float(self.cfg["pulse_energy_J"])
                / max(self._commercial_app_effective_duration_s, np.finfo(float).tiny)
            ),
            "input_average_power_W": (
                None
                if repetition_rate is None
                else float(self.cfg["pulse_energy_J"])
                * max(float(repetition_rate), 0.0)
            ),
            "current_effective_duration_s": (
                None
                if self._commercial_app_is_cw
                else float(self._commercial_app_current_effective_duration_s)
            ),
            "spatial_closure": (
                "independent_hg_q_channels_simultaneous_shared_population"
                if self._commercial_app_hg_enabled
                else "power_preserving_shared_time_bin_gaussian_moment_projection"
                if gain_time_resolved
                else "power_preserving_gaussian_moment_projection"
            ),
            "incoherent_hg_modes": bool(self._commercial_app_hg_enabled),
            "spatial_mode_count": len(self._commercial_app_incoherent_modes),
            "mode_spectral_channel_count": len(self._commercial_app_modal_channels),
            "modal_population_coupling": (
                "simultaneous_local_shared_population"
                if self._commercial_app_hg_enabled
                else None
            ),
            "modal_channels": [dict(channel) for channel in self._commercial_app_modal_channels],
            "thermal_phase_closure": (
                "local_parabolic_fit_window_gaussian_lens"
                if self._commercial_app_thermal_lens_model == "local_parabolic"
                else "sampled_phase_gaussian_moment_lens"
            ),
            "material_phase_contract": (
                "Physical thermal/Kerr/endface phase-gradient moments are converted "
                "to the effective-M2 q carrier independently on x/y. Explicit HG "
                "fields, physical B, and the 1-D SPM envelope use physical phase."
            ),
            "thermal_lens": self._commercial_app_thermal_lens_diagnostics(),
            "kerr_phase_closure": (
                (
                    "B_integral_equivalent_quadratic_gaussian_lens_plus_effective_area_1d_spm"
                    if coherent_envelope_resolved
                    else "B_integral_equivalent_quadratic_gaussian_lens"
                )
                if nonlinear_enabled
                else "off"
            ),
            "gain_guiding": "moment_projected_after_each_cell",
            "coherent_spectral_phase_resolved": coherent_envelope_resolved,
            "external_optical_train_dispersion_resolved": coherent_envelope_resolved,
            "spectral_feedback_to_gain_bins": bool(
                coherent_envelope_resolved
                and self._commercial_app_spectral_feedback
                and len(self.spectrum.coordinate) > 1
            ),
            "spectral_feedback_grid": dict(
                self._commercial_app_feedback_grid_contract
            ),
            "gaussian_projection_count": int(self._commercial_app_projection_count),
            "gaussian_projection_fallback_count": int(
                self._commercial_app_projection_fallback_count
            ),
            "gaussian_projection_fallback_reasons": list(
                dict.fromkeys(self._commercial_app_projection_fallback_reasons)
            ),
            "minimum_gaussian_projection_overlap": float(
                self._commercial_app_minimum_projection_overlap
            ),
            "maximum_peak_intensity_W_m2": float(
                self._commercial_app_maximum_peak_intensity_W_m2
            ),
            "maximum_single_step_B_integral_rad": float(
                self._commercial_app_maximum_single_step_b_integral_rad
            ),
            "maximum_cumulative_B_integral_rad": float(
                self._commercial_app_maximum_cumulative_b_integral_rad
            ),
            "maximum_single_step_effective_area_SPM_phase_rad": float(
                self._commercial_app_maximum_single_step_spm_phase_rad
            ),
            "maximum_cumulative_effective_area_SPM_phase_rad": float(
                self._commercial_app_maximum_cumulative_spm_phase_rad
            ),
            "maximum_spectral_support_leakage_fraction": float(
                self._commercial_app_maximum_spectral_support_leakage_fraction
            ),
            "maximum_spectral_sync_error_fraction": float(
                self._commercial_app_maximum_spectral_sync_error_fraction
            ),
            "donor_initialized_channel_count": int(
                self._commercial_app_donor_initialized_channel_count
            ),
            "maximum_donor_initialized_energy_fraction": float(
                self._commercial_app_maximum_donor_initialized_energy_fraction
            ),
            "new_channel_spatial_mode_contract": (
                "A target coarse spectral channel with zero incident energy "
                "copies the nearest populated channel's normalized transverse "
                "field. This is a reduced-model closure, not a predicted "
                "frequency-dependent spatial mode."
            ),
            "b_integral_diagnostic_scope": (
                "CW average-intensity closure before Gaussian projection"
                if self._commercial_app_is_cw
                else "effective-area temporal peak-intensity closure before Gaussian projection"
                if coherent_envelope_resolved
                else "time-binned shared-Gaussian peak-intensity closure before projection"
                if gain_time_resolved
                else "fixed-duration peak-intensity closure before Gaussian projection"
            ),
            "repetition_rate_Hz": (
                None if repetition_rate is None else float(repetition_rate)
            ),
            "beam_quality_closure": self._commercial_app_beam_quality_model,
            "configured_M2_x_by_wavelength": configured_m2_x_report,
            "configured_M2_y_by_wavelength": configured_m2_y_report,
            "maximum_preprojection_sampled_M2_x_by_wavelength": (
                discarded_m2_x_report
            ),
            "maximum_preprojection_sampled_M2_y_by_wavelength": (
                discarded_m2_y_report
            ),
            "output_effective_M2_x": effective_m2_x,
            "output_effective_M2_y": effective_m2_y,
            "output_effective_M2_x_by_wavelength": proxy_m2_x_report,
            "output_effective_M2_y_by_wavelength": proxy_m2_y_report,
            "beam_quality_scope": (
                "Configured Gaussian M2 plus the quadrature sum of excess "
                "second-moment angular variance seen immediately before each "
                "Gaussian projection. This is an Commercial App residual-aberration proxy, "
                "not a reconstructed full-field ISO measurement."
                if self._commercial_app_beam_quality_model == "projection_residual"
                else "Configured Gaussian transport M2; discarded aberration is audited but not accumulated."
            ),
        }

    def _gain_transport_diagnostics(self) -> dict[str, Any]:
        diagnostics = super()._gain_transport_diagnostics()
        diagnostics.update(self._commercial_app_gain_transport_diagnostics())
        return diagnostics

    def resolved_intrapulse_population_time_s(self) -> float:
        """Return level-kinetics time advanced by one Commercial App Seed pass."""

        if (
            not self._commercial_app_include_intrapulse_relaxation
            or len(self.temporal.coordinate) < 2
        ):
            return 0.0
        return float(
            self.temporal.coordinate[-1] - self.temporal.coordinate[0]
        )

    def _advance_population_between_temporal_bins(
        self,
        population_slice: np.ndarray,
        previous_bin_index: int,
        current_bin_index: int,
        z_index: int,
    ) -> np.ndarray:
        if not self._commercial_app_include_intrapulse_relaxation:
            return population_slice
        dt_s = float(
            self.temporal.coordinate[current_bin_index]
            - self.temporal.coordinate[previous_bin_index]
        )
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError(
                "Commercial App gain temporal coordinates must be finite and strictly increasing"
            )
        self._commercial_app_intrapulse_dark_step_count += 1
        self._commercial_app_intrapulse_relaxation_span_s += dt_s
        return self.levels.advance_dark_implicit(
            population_slice,
            dt_s,
            z_index=z_index,
        )

    def _gain_step(
        self,
        field: np.ndarray,
        population: np.ndarray,
        iz: int,
        step_length_m: float,
    ) -> np.ndarray:
        """Apply one simultaneous shared-population gain step to HG channels."""

        if not self._commercial_app_hg_enabled or not bool(self.physics["gain_saturation"]):
            return super()._gain_step(field, population, iz, step_length_m)

        photon_energies = photon_energy(self.spectrum.coordinate)
        cross_section_sum = self.levels.signal_cross_section_sum_m2(
            self.spectrum.coordinate,
            population.shape[-2:],
            iz,
        )
        passive_loss = float(self.material.get("signal_passive_loss_m_inv", 0.0))
        override = self.levels_cfg.get("gain_override_m_inv")
        if override is None:
            override_gain = None
        else:
            reservoir = getattr(self, "_override_gain_m_inv", None)
            if reservoir is None or reservoir.shape != population.shape[1:]:
                reservoir = np.full(population.shape[1:], float(override), dtype=float)
                self._override_gain_m_inv = reservoir
            override_gain = reservoir[iz]
        cell_length_m = float(self.grid.dz)
        saturation_cfg = self.levels_cfg.get("saturation_fluence_J_m2")
        if saturation_cfg is None:
            saturation_normal = np.divide(
                photon_energies[:, None, None],
                cross_section_sum,
                out=np.full_like(cross_section_sum, np.inf),
                where=cross_section_sum > 0.0,
            )
        else:
            saturation_normal = np.full_like(
                cross_section_sum, float(saturation_cfg)
            )
        cell_substeps = max(
            int(getattr(self, "_current_crystal_cell_substeps", 1)), 1
        )
        gain_group_size = max(
            int(getattr(self, "_current_gain_group_size", 1)), 1
        )
        full_optical_cell_length = (
            float(step_length_m) * cell_substeps / gain_group_size
        )
        path_cosine = cell_length_m / full_optical_cell_length
        if not 0.0 < path_cosine <= 1.0 + 1e-12:
            raise ValueError(
                "Seed optical step is inconsistent with the axial cell length"
            )
        path_cosine = min(path_cosine, 1.0)
        saturation_projected = saturation_normal * path_cosine
        passive_half_transmission = np.exp(
            -0.5 * passive_loss * step_length_m * self.crystal_mask
        )[None, :, :]
        physical_count = len(self._commercial_app_reported_gain_wavelengths_m)
        grouped_channels = [
            np.asarray(
                [
                    index
                    for index, channel in enumerate(self._commercial_app_modal_channels)
                    if int(channel["spectral_index"]) == spectral_index
                ],
                dtype=int,
            )
            for spectral_index in range(physical_count)
        ]

        for it in range(field.shape[0]):
            if it > 0:
                population[:, iz] = self._advance_population_between_temporal_bins(
                    population[:, iz], it - 1, it, iz
                )
            fluence = np.abs(field[it]) ** 2
            active_input_fluence = fluence * passive_half_transmission
            gain_fluence = np.zeros_like(fluence)
            for spectral_order in (
                range(physical_count),
                range(physical_count - 1, -1, -1),
            ):
                for spectral_index in spectral_order:
                    channels = grouped_channels[spectral_index]
                    if channels.size == 0:
                        continue
                    representative = int(channels[0])
                    modal_input = 0.5 * active_input_fluence[channels]
                    packet_in = np.sum(modal_input, axis=0)
                    if override_gain is None:
                        active_gain = self.levels.signal_gain_m_inv(
                            population[:, iz],
                            wavelength_m=float(
                                self._commercial_app_reported_gain_wavelengths_m[
                                    spectral_index
                                ]
                            ),
                            z_index=iz,
                        )
                    else:
                        active_gain = override_gain
                    exponent = active_gain * self.crystal_mask * step_length_m
                    valid = np.isfinite(saturation_projected[representative]) & (
                        saturation_projected[representative] > 0.0
                    )
                    fallback_exponent = np.where(valid, -np.inf, exponent)
                    self._maximum_unsaturated_gain_exponent = max(
                        getattr(self, "_maximum_unsaturated_gain_exponent", -np.inf),
                        float(np.max(fallback_exponent)),
                    )
                    self._positive_gain_exponent_clipping_active = bool(
                        getattr(
                            self, "_positive_gain_exponent_clipping_active", False
                        )
                        or np.any(fallback_exponent > 50.0)
                    )
                    packet_out = packet_in * np.exp(np.minimum(exponent, 50.0))
                    if np.any(valid):
                        packet_out = np.where(
                            valid,
                            frantz_nodvik(
                                packet_in,
                                exponent,
                                np.where(
                                    valid,
                                    saturation_projected[representative],
                                    1.0,
                                ),
                            ),
                            packet_out,
                        )
                    if override_gain is None and np.any(
                        cross_section_sum[representative] > 0.0
                    ):
                        requested_events = (packet_out - packet_in) / (
                            photon_energies[representative] * cell_length_m
                        )
                        updated, event_stats = self.levels.apply_signal_photon_events(
                            population[:, iz],
                            requested_events,
                            float(
                                self._commercial_app_reported_gain_wavelengths_m[
                                    spectral_index
                                ]
                            ),
                            z_index=iz,
                            input_photon_density_m3=(
                                packet_in
                                / (
                                    photon_energies[representative]
                                    * cell_length_m
                                )
                            ),
                            interaction_length_m=step_length_m,
                        )
                        population[:, iz] = updated
                        events = np.asarray(
                            event_stats["applied_photon_events_m3"], dtype=float
                        )
                        packet_out = packet_in + (
                            events
                            * photon_energies[representative]
                            * cell_length_m
                        )
                    elif override_gain is not None and np.any(valid):
                        depletion = np.divide(
                            packet_out - packet_in,
                            saturation_normal[representative] * cell_length_m,
                            out=np.zeros_like(packet_out),
                            where=valid,
                        )
                        override_gain -= depletion
                        if float(override) >= 0.0:
                            override_gain[:] = np.maximum(override_gain, 0.0)
                        else:
                            override_gain[:] = np.minimum(override_gain, 0.0)
                    packet_scale = np.divide(
                        np.maximum(packet_out, 0.0),
                        packet_in,
                        out=np.zeros_like(packet_out),
                        where=packet_in > np.finfo(float).tiny,
                    )
                    gain_fluence[channels] += modal_input * packet_scale[None, ...]

            output_fluence = np.maximum(
                gain_fluence * passive_half_transmission, 0.0
            )
            self.passive_loss_energy_J_m3[iz] += np.sum(
                np.maximum(fluence - active_input_fluence, 0.0)
                + np.maximum(gain_fluence - output_fluence, 0.0),
                axis=0,
            ) / max(self.grid.dz, np.finfo(float).tiny)
            scale = np.sqrt(
                output_fluence / np.maximum(fluence, np.finfo(float).tiny)
            )
            field[it] *= np.where(fluence > 0.0, scale, 0.0)
            if self.clip_crystal:
                field[it] *= self.crystal_mask[None, :, :]
        return field

    def _commercial_app_coarse_energy_per_pulse(self, field: np.ndarray) -> np.ndarray:
        energy = np.sum(
            self.grid.integrate_transverse(np.abs(field) ** 2), axis=0
        )
        if self._commercial_app_active_repetition_rate_Hz is not None:
            energy = energy / max(
                self._commercial_app_active_repetition_rate_Hz,
                np.finfo(float).tiny,
            )
        return np.asarray(energy, dtype=float)

    def _commercial_app_high_resolution_bin_indices(self) -> np.ndarray:
        coarse_frequency = LIGHT_SPEED / np.asarray(
            self.spectrum.coordinate, dtype=float
        )
        if len(coarse_frequency) == 1:
            return np.zeros(len(self._commercial_app_pulse.frequency_Hz), dtype=int)
        order = np.argsort(coarse_frequency)
        sorted_frequency = coarse_frequency[order]
        edges = 0.5 * (sorted_frequency[:-1] + sorted_frequency[1:])
        sorted_indices = np.searchsorted(
            edges,
            self._commercial_app_pulse.frequency_Hz,
        )
        return order[sorted_indices]

    def _commercial_app_aggregate_coherent_spectrum(
        self, spectral_energy_J: np.ndarray
    ) -> np.ndarray:
        values = np.asarray(spectral_energy_J, dtype=float)
        indices = self._commercial_app_high_resolution_bin_indices()
        return np.bincount(
            indices,
            weights=values,
            minlength=len(self.spectrum.coordinate),
        ).astype(float)

    def _commercial_app_sync_coherent_spectrum_to_field(
        self, field: np.ndarray
    ) -> np.ndarray:
        """Smoothly reconcile the 1-D coherent envelope with gain-bin energy."""

        target = self._commercial_app_coarse_energy_per_pulse(field)
        target_total = float(np.sum(target))
        tiny = np.finfo(float).tiny
        if target_total <= tiny:
            self._commercial_app_coherent_spectrum[:] = 0.0
            self._commercial_app_previous_coarse_energy_J = target.copy()
            return target
        current_total = float(
            np.sum(np.abs(self._commercial_app_coherent_spectrum) ** 2)
        )
        if current_total <= tiny:
            self._commercial_app_coherent_spectrum = self._commercial_app_input_coherent_spectrum.copy()
            current_total = float(
                np.sum(np.abs(self._commercial_app_coherent_spectrum) ** 2)
            )
        previous = self._commercial_app_previous_coarse_energy_J
        if previous is None:
            # The first synchronization occurs after the pre-gain optical
            # train.  Match each populated coarse bin here so wavelength-
            # dependent entrance transmission/filtering is not reduced to one
            # global loss factor in the 1-D envelope.
            original_spectrum = self._commercial_app_coherent_spectrum.copy()
            configured_weights = np.maximum(
                np.asarray(self.spectrum.weights, dtype=float), 0.0
            )
            configured_weights /= max(float(np.sum(configured_weights)), tiny)
            nominal_bins = target_total * configured_weights
            ratios = np.zeros_like(target)
            populated = nominal_bins > tiny
            ratios[populated] = target[populated] / nominal_bins[populated]
            indices = self._commercial_app_high_resolution_bin_indices()
            self._commercial_app_coherent_spectrum *= np.sqrt(
                np.maximum(ratios[indices], 0.0)
            )
            synchronized_total = float(
                np.sum(np.abs(self._commercial_app_coherent_spectrum) ** 2)
            )
            if synchronized_total <= tiny:
                self._commercial_app_coherent_spectrum = original_spectrum
                synchronized_total = current_total
            self._commercial_app_coherent_spectrum *= math.sqrt(
                target_total / max(synchronized_total, tiny)
            )
            synchronized_bins = self._commercial_app_aggregate_coherent_spectrum(
                np.abs(self._commercial_app_coherent_spectrum) ** 2
            )
            sync_error = 0.5 * float(
                np.sum(np.abs(synchronized_bins - target))
            ) / max(target_total, tiny)
            self._commercial_app_maximum_spectral_sync_error_fraction = max(
                self._commercial_app_maximum_spectral_sync_error_fraction,
                sync_error,
            )
            self._commercial_app_previous_coarse_energy_J = target.copy()
            return target

        previous_total = float(np.sum(previous))
        global_ratio = target_total / max(previous_total, tiny)
        ratios = np.full(len(target), global_ratio, dtype=float)
        valid = (previous > tiny) & (target > tiny)
        ratios[valid] = target[valid] / previous[valid]
        extinguished = (previous > tiny) & (target <= tiny)
        ratios[extinguished] = tiny / previous[extinguished]
        ratios = np.clip(ratios, math.exp(-50.0), math.exp(50.0))

        coarse_frequency = LIGHT_SPEED / np.asarray(
            self.spectrum.coordinate, dtype=float
        )
        order = np.argsort(coarse_frequency)
        if len(coarse_frequency) == 1:
            logarithmic_ratio = np.full(
                len(self._commercial_app_pulse.frequency_Hz),
                math.log(float(ratios[0])),
            )
        else:
            logarithmic_ratio = np.interp(
                self._commercial_app_pulse.frequency_Hz,
                coarse_frequency[order],
                np.log(ratios[order]),
                left=float(np.log(ratios[order][0])),
                right=float(np.log(ratios[order][-1])),
            )
        self._commercial_app_coherent_spectrum *= np.exp(0.5 * logarithmic_ratio)
        synchronized_total = float(
            np.sum(np.abs(self._commercial_app_coherent_spectrum) ** 2)
        )
        self._commercial_app_coherent_spectrum *= math.sqrt(
            target_total / max(synchronized_total, tiny)
        )
        synchronized_bins = self._commercial_app_aggregate_coherent_spectrum(
            np.abs(self._commercial_app_coherent_spectrum) ** 2
        )
        sync_error = 0.5 * float(np.sum(np.abs(synchronized_bins - target))) / max(
            target_total,
            tiny,
        )
        self._commercial_app_maximum_spectral_sync_error_fraction = max(
            self._commercial_app_maximum_spectral_sync_error_fraction,
            sync_error,
        )
        self._commercial_app_previous_coarse_energy_J = target.copy()
        return target

    def _commercial_app_redistribute_field_spectrum(
        self, field: np.ndarray
    ) -> np.ndarray:
        """Feed the 1-D SPM spectrum back to the coarse gain channels."""

        current = self._commercial_app_coarse_energy_per_pulse(field)
        total = float(np.sum(current))
        tiny = np.finfo(float).tiny
        if total <= tiny:
            return field
        high_energy = np.abs(self._commercial_app_coherent_spectrum) ** 2
        coarse_frequency = LIGHT_SPEED / np.asarray(
            self.spectrum.coordinate, dtype=float
        )
        if len(coarse_frequency) > 1:
            outside = (
                (self._commercial_app_pulse.frequency_Hz < float(np.min(coarse_frequency)))
                | (
                    self._commercial_app_pulse.frequency_Hz
                    > float(np.max(coarse_frequency))
                )
            )
            leakage = float(np.sum(high_energy[outside])) / max(
                float(np.sum(high_energy)), tiny
            )
            self._commercial_app_maximum_spectral_support_leakage_fraction = max(
                self._commercial_app_maximum_spectral_support_leakage_fraction,
                leakage,
            )
        target = self._commercial_app_aggregate_coherent_spectrum(high_energy)
        target *= total / max(float(np.sum(target)), tiny)
        result = np.asarray(field, dtype=complex).copy()
        nonzero = np.flatnonzero(current > tiny)
        donor_initialized = (current <= tiny) & (target > tiny)
        donor_initialized_energy_fraction = float(
            np.sum(target[donor_initialized]) / max(float(np.sum(target)), tiny)
        )
        self._commercial_app_donor_initialized_channel_count += int(
            np.count_nonzero(donor_initialized)
        )
        self._commercial_app_maximum_donor_initialized_energy_fraction = max(
            self._commercial_app_maximum_donor_initialized_energy_fraction,
            donor_initialized_energy_fraction,
        )
        for index in range(len(current)):
            if current[index] > tiny:
                result[:, index] *= math.sqrt(
                    max(float(target[index]), 0.0) / current[index]
                )
            elif target[index] > tiny and nonzero.size:
                donor = int(
                    nonzero[
                        np.argmin(
                            np.abs(
                                coarse_frequency[nonzero]
                                - coarse_frequency[index]
                            )
                        )
                    ]
                )
                # Copy from the immutable input.  ``result[:, donor]`` may
                # already have been rescaled earlier in this loop, which made
                # the energy assigned to a newly populated bin depend on bin
                # order and could violate the spectral-energy contract.
                result[:, index] = field[:, donor] * math.sqrt(
                    float(target[index]) / current[donor]
                )
        self._commercial_app_previous_coarse_energy_J = target.copy()
        return result

    def _commercial_app_inverse_effective_area_per_m2(self, field: np.ndarray) -> float:
        fluence = np.sum(np.abs(field) ** 2, axis=(0, 1))
        if self._commercial_app_active_repetition_rate_Hz is not None:
            fluence = fluence / max(
                self._commercial_app_active_repetition_rate_Hz,
                np.finfo(float).tiny,
            )
        total = float(self.grid.integrate_transverse(fluence))
        if total <= np.finfo(float).tiny:
            return 0.0
        return float(self.grid.integrate_transverse(fluence**2)) / total**2

    def _apply_commercial_app_effective_area_strong_field(
        self,
        temporal_field: np.ndarray,
        *,
        inverse_effective_area_per_m2: float,
        distance_m: float,
    ) -> np.ndarray:
        """Apply the L3.5 Gaussian-moment 7PA/carrier/FCA closure.

        ``temporal_field`` is normalized so ``abs(field)**2`` is energy in a
        time bin.  Multiplication by ``sqrt(1/A_eff)`` therefore produces the
        fluence-density normalization required by the shared strong-field
        kernel.  For the two-dimensional Gaussian convention used here,
        ``integral(I**K dA) = 2**(K-1) * P**K /
        (K*A_eff**(K-1))``.  Scaling beta by that exact moment factor makes
        the scalar closure reproduce the weak-loss transverse integral.  The
        returned scalar attenuation/phase is then mapped back onto the
        coherent envelope.

        When a pulse train is configured, the pre-pulse carrier density is a
        local fixed point of the causal rate equation plus the dark-time
        exponential decay.  Solving that tiny scalar state here avoids adding
        an effective-area carrier residual to the expensive global population
        pulse map.
        """

        if (
            not self._commercial_app_effective_area_strong_field_enabled()
            or distance_m == 0.0
            or inverse_effective_area_per_m2 <= 0.0
        ):
            return temporal_field

        strong_field_cfg = dict(self.material.get("strong_field", {}))
        order = int(strong_field_cfg.get("multiphoton_order", 2))
        # Gaussian moment closure for
        # I=2P/(pi*wx*wy)*exp[-2(x^2/wx^2+y^2/wy^2)] and
        # A_eff=(integral I)^2/integral(I^2)=pi*wx*wy:
        # integral I(r)^K dA = 2^(K-1) P^K/(K A_eff^(K-1)).
        strong_field_cfg["beta_SI"] = float(
            strong_field_cfg.get("beta_SI", 0.0)
        ) * (2.0 ** (order - 1)) / float(order)
        area_scale = math.sqrt(float(inverse_effective_area_per_m2))
        effective_field = np.asarray(temporal_field, dtype=complex) * area_scale
        dt_s = max(float(self._commercial_app_pulse.dt_s), np.finfo(float).tiny)
        k0_rad_m = 2.0 * math.pi / float(self.cfg["wavelength_m"])

        def advance(initial_density_m3: float | None):
            initial = (
                None
                if initial_density_m3 is None
                else np.asarray(max(float(initial_density_m3), 0.0))
            )
            return apply_strong_field_step(
                effective_field,
                dt_s,
                float(distance_m),
                float(self.cfg["wavelength_m"]),
                strong_field_cfg,
                k0_rad_m,
                cell_length_m=float(distance_m),
                initial_carrier_density_m3=initial,
            )

        carrier_cfg = strong_field_cfg.get("free_carrier", {})
        carrier_enabled = str(carrier_cfg.get("model", "off")) == "rate_equation"
        repetition_rate_Hz = float(
            self._commercial_app_active_repetition_rate_Hz
            if self._commercial_app_active_repetition_rate_Hz is not None
            else self.simulation.get("repetition_rate_Hz", 0.0)
        )
        periodic_residual = 0.0
        periodic_iterations = 0
        if carrier_enabled and repetition_rate_Hz > 0.0:
            lifetime_s = float(carrier_cfg.get("lifetime_s", 1e-9))
            pulse_window_s = dt_s * len(effective_field)
            repetition_period_s = 1.0 / repetition_rate_Hz
            timing_tolerance_s = 64.0 * np.finfo(float).eps * max(
                pulse_window_s,
                repetition_period_s,
                np.finfo(float).tiny,
            )
            if pulse_window_s > repetition_period_s + timing_tolerance_s:
                raise ValueError(
                    "Commercial App effective-area carrier pulse window exceeds the "
                    "repetition period; the discrete-pulse fixed point cannot "
                    "represent temporally overlapping pulses"
                )
            dark_time_s = max(
                repetition_period_s - pulse_window_s,
                0.0,
            )
            dark_decay = math.exp(-dark_time_s / lifetime_s)
            initial_density = 0.0
            for iteration in range(
                1, self._commercial_app_carrier_periodic_maximum_iterations + 1
            ):
                strong = advance(initial_density)
                terminal_density = float(strong.carrier_density_m3[-1])
                next_initial = terminal_density * dark_decay
                periodic_residual = abs(next_initial - initial_density) / max(
                    abs(next_initial),
                    abs(initial_density),
                    1.0,
                )
                periodic_iterations = iteration
                initial_density = next_initial
                if (
                    periodic_residual
                    <= self._commercial_app_carrier_periodic_relative_tolerance
                ):
                    break
            # Re-evaluate at the converged pre-pulse state so the optical
            # output and reported carrier history use the same fixed point.
            strong = advance(initial_density)
            self._commercial_app_strong_field_periodic_solve_count += 1
            self._commercial_app_strong_field_maximum_periodic_residual = max(
                self._commercial_app_strong_field_maximum_periodic_residual,
                periodic_residual,
            )
            self._commercial_app_strong_field_maximum_periodic_iterations = max(
                self._commercial_app_strong_field_maximum_periodic_iterations,
                periodic_iterations,
            )
            if (
                periodic_residual
                > self._commercial_app_carrier_periodic_relative_tolerance
                and not any(
                    "effective-area carrier fixed point" in warning
                    for warning in self._commercial_app_runtime_warnings
                )
            ):
                self._commercial_app_runtime_warnings.append(
                    "Commercial App effective-area carrier fixed point reached its "
                    "iteration limit; inspect the reported periodic residual."
                )
        else:
            strong = advance(None)

        inverse_area = float(inverse_effective_area_per_m2)
        multiphoton_energy_J = float(
            np.sum(strong.multiphoton_absorbed_fluence_J_m2)
        ) / inverse_area
        free_carrier_energy_J = float(
            np.sum(strong.free_carrier_absorbed_fluence_J_m2)
        ) / inverse_area
        self._commercial_app_strong_field_multiphoton_absorbed_energy_J += (
            multiphoton_energy_J
        )
        self._commercial_app_strong_field_free_carrier_absorbed_energy_J += (
            free_carrier_energy_J
        )
        self._commercial_app_strong_field_total_absorbed_energy_J += (
            multiphoton_energy_J + free_carrier_energy_J
        )
        self._commercial_app_strong_field_maximum_peak_carrier_density_m3 = max(
            self._commercial_app_strong_field_maximum_peak_carrier_density_m3,
            float(strong.peak_carrier_density_m3),
        )
        self._commercial_app_strong_field_maximum_fractional_loss = max(
            self._commercial_app_strong_field_maximum_fractional_loss,
            float(strong.maximum_fractional_loss),
        )
        return strong.field / area_scale

    def _apply_commercial_app_temporal_step(
        self,
        field: np.ndarray,
        distance_m: float,
    ) -> np.ndarray:
        if not self._commercial_app_temporal_envelope_enabled():
            return field
        working = np.asarray(field, dtype=complex).copy()
        self._commercial_app_sync_coherent_spectrum_to_field(working)
        linear_phase = material_dispersion_phase(
            self._commercial_app_pulse.frequency_Hz,
            self.material.get("dispersion", {}),
            float(distance_m),
        )
        if np.any(linear_phase):
            self._commercial_app_coherent_spectrum *= np.exp(1j * linear_phase)

        temporal_field = self._commercial_app_pulse.spectrum_to_time(
            self._commercial_app_coherent_spectrum
        )
        temporal_energy = np.abs(temporal_field) ** 2
        total_energy = float(np.sum(temporal_energy))
        dt = max(self._commercial_app_pulse.dt_s, np.finfo(float).tiny)
        inverse_effective_area_projected = (
            self._commercial_app_inverse_effective_area_per_m2(working)
        )
        marker = getattr(self, "_active_gain_marker", self.gain_marker)
        projected_area_cosine = gain_medium_path_cosine(marker)
        # ``working`` is sampled per crystal-face area.  Kerr and KPA use the
        # intensity in a plane normal to the beam, whose effective area is
        # smaller by cos(theta).
        inverse_effective_area = (
            inverse_effective_area_projected / projected_area_cosine
        )
        multiphoton_energy_before_J = (
            self._commercial_app_strong_field_multiphoton_absorbed_energy_J
        )
        free_carrier_energy_before_J = (
            self._commercial_app_strong_field_free_carrier_absorbed_energy_J
        )
        temporal_field = self._apply_commercial_app_effective_area_strong_field(
            temporal_field,
            inverse_effective_area_per_m2=inverse_effective_area,
            distance_m=float(distance_m),
        )
        post_loss_energy = float(np.sum(np.abs(temporal_field) ** 2))
        local_multiphoton_energy_J = max(
            self._commercial_app_strong_field_multiphoton_absorbed_energy_J
            - multiphoton_energy_before_J,
            0.0,
        )
        local_free_carrier_energy_J = max(
            self._commercial_app_strong_field_free_carrier_absorbed_energy_J
            - free_carrier_energy_before_J,
            0.0,
        )
        local_absorbed_energy_J = (
            local_multiphoton_energy_J + local_free_carrier_energy_J
        )
        active_z_index = getattr(self, "_active_z_index", None)
        if (
            local_absorbed_energy_J > 0.0
            and active_z_index is not None
            and 0 <= int(active_z_index) < len(self.grid.z)
        ):
            spatial_fluence = np.sum(np.abs(working) ** 2, axis=(0, 1))
            if self._commercial_app_active_repetition_rate_Hz is not None:
                spatial_fluence = spatial_fluence / max(
                    self._commercial_app_active_repetition_rate_Hz,
                    np.finfo(float).tiny,
                )
            spatial_total = float(
                self.grid.integrate_transverse(spatial_fluence)
            )
            if spatial_total > np.finfo(float).tiny:
                order = int(
                    self.material.get("strong_field", {}).get(
                        "multiphoton_order", 2
                    )
                )

                def normalized_moment_density(power: int) -> np.ndarray:
                    peak = float(np.max(spatial_fluence))
                    if peak <= np.finfo(float).tiny:
                        return spatial_fluence / spatial_total
                    weight = np.power(
                        np.maximum(spatial_fluence / peak, 0.0),
                        power,
                    )
                    integral = float(
                        self.grid.integrate_transverse(weight)
                    )
                    if integral <= np.finfo(float).tiny:
                        return spatial_fluence / spatial_total
                    return weight / integral

                # The scalar closure preserves each optical-energy loss.  Its
                # reconstructed map uses the corresponding separable Gaussian
                # weak-loss moments: I^K for KPA and I^(K+1) for FCA by
                # KPA-generated carriers.  Normalizing the two maps separately
                # keeps the volume ledger exact without pretending that radial
                # carrier transport was resolved.
                multiphoton_density = normalized_moment_density(order)
                free_carrier_density = normalized_moment_density(order + 1)
                self._commercial_app_strong_field_loss_energy_J_m3[
                    int(active_z_index)
                ] += (
                    local_multiphoton_energy_J * multiphoton_density
                    + local_free_carrier_energy_J * free_carrier_density
                ) / max(
                    float(self.grid.dz), np.finfo(float).tiny
                )
        if total_energy > np.finfo(float).tiny:
            working *= math.sqrt(
                max(post_loss_energy, 0.0) / total_energy
            )
        temporal_energy = np.abs(temporal_field) ** 2
        nonlinear_phase = np.zeros_like(temporal_energy)
        if (
            self.physics.get("nonlinear_phase_model", "off")
            == "kerr_parabolic_lens"
        ):
            n2 = float(self.material.get("n2_m2_W", 0.0))
            if n2 != 0.0 and inverse_effective_area > 0.0:
                temporal_power = temporal_energy / dt
                nonlinear_phase = (
                    2.0
                    * math.pi
                    / float(self.cfg["wavelength_m"])
                    * n2
                    * float(distance_m)
                    * inverse_effective_area
                    * temporal_power
                )
                temporal_field *= np.exp(1j * nonlinear_phase)
                self._commercial_app_spm_cumulative_phase_rad += nonlinear_phase
                self._commercial_app_maximum_single_step_spm_phase_rad = max(
                    self._commercial_app_maximum_single_step_spm_phase_rad,
                    float(np.max(np.abs(nonlinear_phase))),
                )
                self._commercial_app_maximum_cumulative_spm_phase_rad = max(
                    self._commercial_app_maximum_cumulative_spm_phase_rad,
                    float(np.max(np.abs(self._commercial_app_spm_cumulative_phase_rad))),
                )
        self._commercial_app_coherent_spectrum = self._commercial_app_pulse.time_to_spectrum(
            temporal_field
        )
        output_temporal_energy = np.abs(temporal_field) ** 2
        total_energy = float(np.sum(output_temporal_energy))
        output_peak_power = float(np.max(output_temporal_energy)) / dt
        self._commercial_app_current_effective_duration_s = (
            total_energy / output_peak_power
            if total_energy > 0.0 and output_peak_power > 0.0
            else self._commercial_app_effective_duration_s
        )
        if self._commercial_app_spectral_feedback:
            working = self._commercial_app_redistribute_field_spectrum(working)
        return working

    @staticmethod
    def _commercial_app_temporal_boundary_fraction(values: np.ndarray) -> float:
        energy = np.maximum(np.asarray(values, dtype=float), 0.0)
        if energy.size == 0:
            return 0.0
        count = max(1, min(len(energy) // 2, int(math.ceil(0.05 * len(energy)))))
        return float(np.sum(energy[:count]) + np.sum(energy[-count:])) / max(
            float(np.sum(energy)), np.finfo(float).tiny
        )

    def _apply_commercial_app_element_dispersion(
        self,
        field: np.ndarray,
        elements: list[dict[str, Any]],
    ) -> None:
        """Apply one optical train's dispersion to the coherent 1-D envelope."""

        if not self._commercial_app_temporal_envelope_enabled():
            return
        self._commercial_app_sync_coherent_spectrum_to_field(field)
        self._apply_commercial_app_element_dispersion_phase(elements)

    def _apply_commercial_app_element_dispersion_phase(
        self,
        elements: list[dict[str, Any]],
    ) -> None:
        """Apply element phase once without changing envelope amplitudes."""

        if not self._commercial_app_temporal_envelope_enabled():
            return
        phase = np.zeros_like(self._commercial_app_pulse.frequency_Hz, dtype=float)
        for element in elements:
            dispersion = element.get("dispersion")
            if isinstance(dispersion, dict) and dispersion:
                phase += phase_from_coefficients(
                    self._commercial_app_pulse.frequency_Hz,
                    dispersion,
                )
        if np.any(phase):
            self._commercial_app_coherent_spectrum *= np.exp(1j * phase)

    def _result_extras(
        self,
        field: np.ndarray,
        source_total_J: float,
        source_spectrum_J: np.ndarray,
        output_spectrum_J: np.ndarray,
    ) -> dict[str, Any]:
        extras = super()._result_extras(
            field,
            source_total_J,
            source_spectrum_J,
            output_spectrum_J,
        )
        # This field feeds the thermal solve even when plotting is suppressed.
        if self._commercial_app_temporal_envelope_enabled():
            extras["nonlinear_loss_energy_J_m3"] = (
                self._commercial_app_strong_field_loss_energy_J_m3.copy()
            )
        if (
            not self._commercial_app_temporal_envelope_enabled()
            or bool(self.physics.get("_suppress_output_diagnostics", False))
        ):
            return extras
        self._commercial_app_sync_coherent_spectrum_to_field(field)
        tiny = np.finfo(float).tiny
        input_spectrum = self._commercial_app_input_coherent_spectrum.copy()
        input_spectrum *= math.sqrt(
            float(source_total_J)
            / max(float(np.sum(np.abs(input_spectrum) ** 2)), tiny)
        )
        output_equivalent = self._commercial_app_coherent_spectrum.copy()
        output_equivalent *= math.sqrt(
            float(np.sum(output_spectrum_J))
            / max(float(np.sum(np.abs(output_equivalent) ** 2)), tiny)
        )
        compressor_phase = phase_from_coefficients(
            self._commercial_app_pulse.frequency_Hz,
            self.cfg.get("coherent", {}).get("compressor", {}),
        )
        compressed_equivalent = output_equivalent * np.exp(
            1j * compressor_phase
        )
        input_temporal = self._commercial_app_pulse.spectrum_to_time(input_spectrum)
        output_temporal = self._commercial_app_pulse.spectrum_to_time(output_equivalent)
        compressed_temporal = self._commercial_app_pulse.spectrum_to_time(
            compressed_equivalent
        )
        input_temporal_energy = np.abs(input_temporal) ** 2
        output_temporal_energy = np.abs(output_temporal) ** 2
        compressed_temporal_energy = np.abs(compressed_temporal) ** 2
        input_spectral_energy = np.abs(input_spectrum) ** 2
        output_spectral_energy = np.abs(output_equivalent) ** 2
        input_bandwidth = intensity_fwhm(
            self._commercial_app_pulse.frequency_Hz, input_spectral_energy
        )
        output_bandwidth = intensity_fwhm(
            self._commercial_app_pulse.frequency_Hz, output_spectral_energy
        )
        dt = max(self._commercial_app_pulse.dt_s, tiny)
        boundary_fractions = {
            "input": self._commercial_app_temporal_boundary_fraction(
                input_temporal_energy
            ),
            "output": self._commercial_app_temporal_boundary_fraction(
                output_temporal_energy
            ),
            "compressed": self._commercial_app_temporal_boundary_fraction(
                compressed_temporal_energy
            ),
        }
        diagnostics = dict(self._commercial_app_pulse.diagnostics)
        diagnostics.update(
            {
                "model": (
                    "commercial_app_effective_area_1d_strong_field"
                    if self._commercial_app_effective_area_strong_field_enabled()
                    else "commercial_app_effective_area_1d_spm"
                ),
                "approximation": (
                    "One coherent time/frequency envelope coupled to the "
                    "Gaussian spatial mode through its instantaneous effective "
                    "area; irreversible loss uses a Gaussian transverse-moment "
                    "closure."
                    if self._commercial_app_effective_area_strong_field_enabled()
                    else "One coherent time/frequency envelope coupled to the "
                    "Gaussian spatial mode through its instantaneous effective area."
                ),
                "input_grid_energy_J": float(source_total_J),
                "output_energy_J": float(np.sum(output_spectral_energy)),
                "input_fwhm_s": intensity_fwhm(
                    self._commercial_app_pulse.time_s, input_temporal_energy
                ),
                "output_fwhm_s": intensity_fwhm(
                    self._commercial_app_pulse.time_s, output_temporal_energy
                ),
                "compressed_fwhm_s": intensity_fwhm(
                    self._commercial_app_pulse.time_s, compressed_temporal_energy
                ),
                "input_peak_power_W": float(np.max(input_temporal_energy)) / dt,
                "output_peak_power_W": float(np.max(output_temporal_energy)) / dt,
                "compressed_peak_power_W": (
                    float(np.max(compressed_temporal_energy)) / dt
                ),
                "input_spectral_fwhm_Hz": input_bandwidth,
                "output_spectral_fwhm_Hz": output_bandwidth,
                "spectral_fwhm_broadening_ratio": (
                    output_bandwidth / input_bandwidth
                    if input_bandwidth > 0.0
                    else None
                ),
                "maximum_local_B_integral_rad": float(
                    self._commercial_app_maximum_cumulative_b_integral_rad
                ),
                "maximum_cumulative_B_integral_rad": float(
                    self._commercial_app_maximum_cumulative_b_integral_rad
                ),
                "maximum_effective_area_SPM_phase_rad": float(
                    self._commercial_app_maximum_cumulative_spm_phase_rad
                ),
                "temporal_boundary_energy_fraction": boundary_fractions,
                "maximum_temporal_boundary_energy_fraction": max(
                    boundary_fractions.values()
                ),
                "maximum_spectral_support_leakage_fraction": float(
                    self._commercial_app_maximum_spectral_support_leakage_fraction
                ),
                "maximum_spectral_sync_error_fraction": float(
                    self._commercial_app_maximum_spectral_sync_error_fraction
                ),
                "spectral_feedback_grid": dict(
                    self._commercial_app_feedback_grid_contract
                ),
                "compressor": dict(
                    self.cfg.get("coherent", {}).get("compressor", {})
                ),
                "spatial_temporal_coupling_resolved": False,
                "gain_temporal_bins": 1,
                "strong_field": {
                    "enabled": self._commercial_app_effective_area_strong_field_enabled(),
                    "model": self.material.get("strong_field", {}).get(
                        "model", "off"
                    ),
                    "mechanism": self.material.get("strong_field", {}).get(
                        "mechanism", "unspecified"
                    ),
                    "multiphoton_order": int(
                        self.material.get("strong_field", {}).get(
                            "multiphoton_order", 2
                        )
                    ),
                    "beta_SI": float(
                        self.material.get("strong_field", {}).get(
                            "beta_SI", 0.0
                        )
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
                    "spatial_closure": (
                        "gaussian_effective_area_Kth_optical_energy_moment_2powKm1_beta_over_K"
                    ),
                    "carrier_density_interpretation": (
                        "single_energy_equivalent_scalar_state; radial carrier "
                        "density and its exact intensity-weighted FCA/phase "
                        "moments are not resolved"
                    ),
                    "carrier_periodic_closure": (
                        "local_scalar_fixed_point_with_dark_time_decay"
                    ),
                    "carrier_periodic_solve_count": int(
                        self._commercial_app_strong_field_periodic_solve_count
                    ),
                    "carrier_periodic_relative_tolerance": float(
                        self._commercial_app_carrier_periodic_relative_tolerance
                    ),
                    "carrier_periodic_maximum_iterations": int(
                        self._commercial_app_carrier_periodic_maximum_iterations
                    ),
                    "maximum_carrier_periodic_relative_residual": float(
                        self._commercial_app_strong_field_maximum_periodic_residual
                    ),
                    "maximum_carrier_periodic_iterations": int(
                        self._commercial_app_strong_field_maximum_periodic_iterations
                    ),
                    "carrier_periodic_converged": bool(
                        self._commercial_app_strong_field_periodic_solve_count == 0
                        or self._commercial_app_strong_field_maximum_periodic_residual
                        <= self._commercial_app_carrier_periodic_relative_tolerance
                    ),
                    "heat_deposition_contract": (
                        "scalar_KPA_and_FCA_energy_ledgers_distributed_by_"
                        "normalized_separable_IpowK_and_IpowKplus1_maps"
                    ),
                    "deposit_fraction_as_heat": float(
                        self.material.get("strong_field", {}).get(
                            "deposit_fraction_as_heat", 1.0
                        )
                    ),
                    "total_absorbed_energy_J": float(
                        self._commercial_app_strong_field_total_absorbed_energy_J
                    ),
                    "multiphoton_absorbed_energy_J": float(
                        self._commercial_app_strong_field_multiphoton_absorbed_energy_J
                    ),
                    "free_carrier_absorbed_energy_J": float(
                        self._commercial_app_strong_field_free_carrier_absorbed_energy_J
                    ),
                    "maximum_peak_carrier_density_m3": float(
                        self._commercial_app_strong_field_maximum_peak_carrier_density_m3
                    ),
                    "maximum_fractional_loss_per_step": float(
                        self._commercial_app_strong_field_maximum_fractional_loss
                    ),
                },
            }
        )
        self._commercial_app_report_wavelengths_m = self._commercial_app_pulse.wavelength_m.copy()
        self._commercial_app_report_input_spectrum_J = input_spectral_energy.copy()
        self._commercial_app_report_output_spectrum_J = output_spectral_energy.copy()
        extras.update(
            {
                "time_s": self._commercial_app_pulse.time_s.copy(),
                "input_temporal_energy_J": input_temporal_energy,
                "output_temporal_energy_J": output_temporal_energy,
                "frequency_Hz": self._commercial_app_pulse.frequency_Hz.copy(),
                "input_spectral_phase_rad": np.unwrap(np.angle(input_spectrum)),
                "output_spectral_phase_rad": np.unwrap(
                    np.angle(output_equivalent)
                ),
                "compressed_spectral_phase_rad": np.unwrap(
                    np.angle(compressed_equivalent)
                ),
                "compressed_temporal_power_W": compressed_temporal_energy / dt,
                "cpa_diagnostics": diagnostics,
            }
        )
        return extras

    def _scan_output_caustic(
        self,
        field: np.ndarray,
        trace: Any,
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
        if self._commercial_app_beam_quality_model == "projection_residual":
            m2_x = np.sqrt(np.maximum(self._commercial_app_proxy_m2_squared_x, 1.0))
            m2_y = np.sqrt(np.maximum(self._commercial_app_proxy_m2_squared_y, 1.0))
        else:
            m2_x = self._commercial_app_configured_m2_x
            m2_y = self._commercial_app_configured_m2_y
        wavelengths = np.asarray(wavelengths_m, dtype=float)
        positions = caustic_positions(config)
        refractive_index = float(config.get("refractive_index", 1.0))
        try:
            projection = project_to_gaussian_mode(
                self._commercial_app_temporally_collapsed_field(field),
                self.grid,
                derivative_method="finite_difference",
            )
            quantities = np.asarray(
                [
                    float(
                        self.grid.integrate_transverse(
                            np.abs(projection.field[index]) ** 2
                        )
                    )
                    for index in range(len(wavelengths))
                ],
                dtype=float,
            )
            total_quantity = float(np.sum(quantities))
            weights = quantities / max(
                total_quantity, np.finfo(float).tiny
            )
            qx_values: list[complex] = []
            qy_values: list[complex] = []
            angle_x_values: list[float] = []
            angle_y_values: list[float] = []
            for index, wavelength in enumerate(wavelengths):
                qx, qy, angle_x, angle_y = self._q_from_projection(
                    projection,
                    index,
                    float(wavelength),
                    refractive_index,
                    float(m2_x[index]),
                    float(m2_y[index]),
                    explicit_hg=self._commercial_app_hg_enabled,
                )
                qx_values.append(qx)
                qy_values.append(qy)
                angle_x_values.append(angle_x)
                angle_y_values.append(angle_y)
            qx_array = np.asarray(qx_values, dtype=complex)
            qy_array = np.asarray(qy_values, dtype=complex)
            angle_x_array = np.asarray(angle_x_values, dtype=float)
            angle_y_array = np.asarray(angle_y_values, dtype=float)
            center_x_0 = np.asarray(projection.center_x_m, dtype=float)
            center_y_0 = np.asarray(projection.center_y_m, dtype=float)
            samples: list[dict[str, Any]] = []
            plane_states: list[
                tuple[
                    float,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    float,
                    float,
                ]
            ] = []
            for position in positions:
                inverse_qx = 1.0 / (qx_array + float(position))
                inverse_qy = 1.0 / (qy_array + float(position))
                radius_x = np.sqrt(
                    -m2_x
                    * wavelengths
                    / (
                        math.pi
                        * refractive_index
                        * np.imag(inverse_qx)
                    )
                )
                radius_y = np.sqrt(
                    -m2_y
                    * wavelengths
                    / (
                        math.pi
                        * refractive_index
                        * np.imag(inverse_qy)
                    )
                )
                centers_x = center_x_0 + float(position) * np.tan(
                    angle_x_array
                )
                centers_y = center_y_0 + float(position) * np.tan(
                    angle_y_array
                )
                center_x = float(np.sum(weights * centers_x))
                center_y = float(np.sum(weights * centers_y))
                variance_x = float(
                    np.sum(
                        weights
                        * (radius_x**2 / 4.0 + (centers_x - center_x) ** 2)
                    )
                )
                variance_y = float(
                    np.sum(
                        weights
                        * (radius_y**2 / 4.0 + (centers_y - center_y) ** 2)
                    )
                )
                covariance_xy = float(
                    np.sum(
                        weights
                        * (centers_x - center_x)
                        * (centers_y - center_y)
                    )
                )
                covariance_matrix = np.asarray(
                    [
                        [variance_x, covariance_xy],
                        [covariance_xy, variance_y],
                    ]
                )
                eigenvalues = np.linalg.eigvalsh(covariance_matrix)
                major_variance = max(float(eigenvalues[-1]), 0.0)
                minor_variance = max(float(eigenvalues[0]), 0.0)
                major_angle = 0.5 * math.degrees(
                    math.atan2(
                        2.0 * covariance_xy,
                        variance_x - variance_y,
                    )
                )
                samples.append(
                    {
                        "position_m_relative_to_system_output": float(position),
                        "quantity": total_quantity,
                        "centroid_x_m": center_x,
                        "centroid_y_m": center_y,
                        "radius_x_D4sigma_m": 2.0
                        * math.sqrt(max(variance_x, 0.0)),
                        "radius_y_D4sigma_m": 2.0
                        * math.sqrt(max(variance_y, 0.0)),
                        "radius_major_D4sigma_m": 2.0
                        * math.sqrt(major_variance),
                        "radius_minor_D4sigma_m": 2.0
                        * math.sqrt(minor_variance),
                        "major_axis_angle_deg": major_angle,
                        "covariance_xy_m2": covariance_xy,
                        "boundary_energy_fraction": 0.0,
                        "outer_x_energy_fraction": 0.0,
                        "outer_y_energy_fraction": 0.0,
                    }
                )
                plane_states.append(
                    (
                        float(position),
                        radius_x,
                        radius_y,
                        centers_x,
                        centers_y,
                        center_x,
                        center_y,
                    )
                )

            reference_wavelength = float(np.sum(weights * wavelengths))
            fit_x = fit_second_moment_caustic(
                positions,
                np.asarray(
                    [row["radius_x_D4sigma_m"] for row in samples]
                ),
                reference_wavelength,
                refractive_index,
            )
            fit_y = fit_second_moment_caustic(
                positions,
                np.asarray(
                    [row["radius_y_D4sigma_m"] for row in samples]
                ),
                reference_wavelength,
                refractive_index,
            )
            caustic_warnings: list[str] = []
            if len(wavelengths) > 1:
                caustic_warnings.append(
                    "Broadband Commercial App caustic M2 is an energy-integrated effective "
                    "value referenced to the energy-weighted wavelength, not a "
                    "monochromatic ISO-11146 result."
                )
            if len(positions) < 10:
                caustic_warnings.append(
                    "Fewer than ten caustic planes were sampled; this is a "
                    "numerical fit, not an ISO-complete measurement layout."
                )
            if not bool(fit_x.get("iso_style_sampling_coverage", False)) or not bool(
                fit_y.get("iso_style_sampling_coverage", False)
            ):
                caustic_warnings.append(
                    "The selected analytic Commercial App scan does not include at least "
                    "five points near the fitted waist and five beyond two "
                    "Rayleigh ranges on both axes. The quadratic fit is still "
                    "reported, but the sampling is not ISO-style."
                )
            diagnostics = {
                "enabled": True,
                "application": "diagnostic_only",
                "affects_propagation": False,
                "method": "commercial_app_analytic_spectral_gaussian_q_D4sigma_fit",
                "reference_plane": "system_output",
                "broadband_effective": len(wavelengths) > 1,
                "reference_wavelength_m": reference_wavelength,
                "refractive_index": refractive_index,
                "sample_count": int(len(samples)),
                "samples": samples,
                "fit_x": fit_x,
                "fit_y": fit_y,
                "wavelengths_m": wavelengths.tolist(),
                "source_energy_fraction_by_wavelength": weights.tolist(),
                "transport_M2_x_by_wavelength": m2_x.tolist(),
                "transport_M2_y_by_wavelength": m2_y.tolist(),
                "maximum_relative_power_drift": 0.0,
                "maximum_boundary_energy_fraction": 0.0,
                "warnings": caustic_warnings,
                "scope": (
                    "Commercial App Gaussian q parameters are advanced analytically at "
                    "each wavelength, then combined as incoherent D4sigma "
                    "moments. The caustic fit is independent of the finite FFT window."
                ),
            }

            if retain:
                nx = max(int(len(self.grid.x)), 33)
                ny = max(int(len(self.grid.y)), 33)
                for (
                    position,
                    radius_x,
                    radius_y,
                    centers_x,
                    centers_y,
                    center_x,
                    center_y,
                ) in plane_states:
                    half_x = max(
                        float(
                            np.max(np.abs(centers_x - center_x) + 3.5 * radius_x)
                        ),
                        np.finfo(float).tiny,
                    )
                    half_y = max(
                        float(
                            np.max(np.abs(centers_y - center_y) + 3.5 * radius_y)
                        ),
                        np.finfo(float).tiny,
                    )
                    x_values = center_x + np.linspace(
                        -half_x + half_x / nx,
                        half_x - half_x / nx,
                        nx,
                    )
                    y_values = center_y + np.linspace(
                        -half_y + half_y / ny,
                        half_y - half_y / ny,
                        ny,
                    )
                    xx, yy = np.meshgrid(x_values, y_values, indexing="xy")
                    intensity = np.zeros((ny, nx), dtype=float)
                    for index in range(len(wavelengths)):
                        if self._commercial_app_hg_enabled:
                            order_x, order_y = self._commercial_app_channel_orders(index)
                            mode_m2_x, mode_m2_y = self._commercial_app_channel_m2(index)
                            amplitude = single_hg_mode_amplitude(
                                xx,
                                yy,
                                radius_x[index] / math.sqrt(mode_m2_x),
                                radius_y[index] / math.sqrt(mode_m2_y),
                                order_x,
                                order_y,
                                center_x_m=float(centers_x[index]),
                                center_y_m=float(centers_y[index]),
                            )
                            density = amplitude**2
                            trapezoid = getattr(np, "trapezoid", np.trapz)
                            sampled_density = float(
                                trapezoid(
                                    trapezoid(density, x_values, axis=1),
                                    y_values,
                                    axis=0,
                                )
                            )
                            density /= max(sampled_density, np.finfo(float).tiny)
                        else:
                            density = 2.0 / (
                                math.pi * radius_x[index] * radius_y[index]
                            ) * np.exp(
                                -2.0
                                * (
                                    ((xx - centers_x[index]) / radius_x[index]) ** 2
                                    + ((yy - centers_y[index]) / radius_y[index]) ** 2
                                )
                            )
                        intensity += quantities[index] * density
                    name = f"caustic_{len(trace.summaries):03d}_z_{position:+.9g}_m"
                    trace.planes[name] = intensity
                    trace.plane_coordinates[name] = (
                        x_values.copy(),
                        y_values.copy(),
                    )
                    sample = next(
                        row
                        for row in samples
                        if row["position_m_relative_to_system_output"] == position
                    )
                    trace.summaries.append(
                        {
                            "name": name,
                            quantity: total_quantity,
                            "grid_nx": nx,
                            "grid_ny": ny,
                            "grid_window_x_m": 2.0 * half_x,
                            "grid_window_y_m": 2.0 * half_y,
                            "centroid_x_m": sample["centroid_x_m"],
                            "centroid_y_m": sample["centroid_y_m"],
                            "radius_x_D4sigma_m": sample[
                                "radius_x_D4sigma_m"
                            ],
                            "radius_y_D4sigma_m": sample[
                                "radius_y_D4sigma_m"
                            ],
                            "covariance_xy_m2": sample["covariance_xy_m2"],
                            "radius_major_D4sigma_m": sample[
                                "radius_major_D4sigma_m"
                            ],
                            "radius_minor_D4sigma_m": sample[
                                "radius_minor_D4sigma_m"
                            ],
                            "major_axis_angle_deg": sample[
                                "major_axis_angle_deg"
                            ],
                            "boundary_energy_fraction": 0.0,
                            "outer_x_energy_fraction": 0.0,
                            "outer_y_energy_fraction": 0.0,
                            "beam_quality_method": "commercial_app_analytic_gaussian_q",
                            "beam_quality_valid": True,
                            "M2_x_sampled": None,
                            "M2_y_sampled": None,
                            "beam_quality_channel_count": len(wavelengths),
                            "beam_quality_sampling_warning": None,
                        }
                    )
        except (FloatingPointError, OverflowError, ValueError) as exc:
            diagnostics, planes = scan_spectral_caustic(
                field,
                self.grid,
                positions,
                wavelengths,
                refractive_index,
                m2_x,
                m2_y,
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
            diagnostics["commercial_app_analytic_fallback"] = True
            diagnostics["commercial_app_analytic_fallback_reason"] = (
                f"{type(exc).__name__}: {exc}"
            )
            self._commercial_app_projection_fallback_reasons.append(
                "analytic caustic: "
                f"{type(exc).__name__}: {exc}"
            )
        diagnostics["commercial_app_beam_quality_closure"] = (
            self._commercial_app_beam_quality_model
        )
        diagnostics["commercial_app_transport_field_remains_gaussian_q"] = True
        diagnostics["planes_recorded"] = bool(retain)
        return diagnostics

    def _finalize_commercial_app_result(self, result: Any) -> Any:
        result.model_used = "commercial_app_fast"
        if self._commercial_app_hg_enabled and result.output_spectrum_J is not None:
            self._commercial_app_modal_output_energy_J = np.asarray(
                result.output_spectrum_J, dtype=float
            ).copy()
        if self._commercial_app_report_wavelengths_m is not None:
            result.wavelengths_m = self._commercial_app_report_wavelengths_m.copy()
            result.input_spectrum_J = self._commercial_app_report_input_spectrum_J.copy()
            result.output_spectrum_J = self._commercial_app_report_output_spectrum_J.copy()
        diagnostics = dict(result.gain_transport or {})
        diagnostics.update(self._commercial_app_gain_transport_diagnostics())
        result.gain_transport = diagnostics
        warnings = [*result.warnings, *self._commercial_app_persistent_warnings]
        warnings.extend(self._commercial_app_runtime_warnings)
        if self._commercial_app_minimum_projection_overlap < 0.99:
            warnings.append(
                "Commercial App fast Gaussian projection discarded measurable non-Gaussian "
                "content; the minimum coherent overlap was "
                f"{self._commercial_app_minimum_projection_overlap:.6g}. Compare this case "
                "with coherent_cpa/FFT before trusting hot-spot or wavefront details."
            )
        if self._commercial_app_projection_fallback_count:
            fallback_reasons = "; ".join(
                dict.fromkeys(self._commercial_app_projection_fallback_reasons)
            )
            warnings.append(
                "Commercial App fast encountered an invalid Gaussian q state and used the "
                "sampled FFT fallback for at least one propagation step. "
                f"Recorded cause(s): {fallback_reasons or 'unavailable'}."
            )
        if self._commercial_app_maximum_donor_initialized_energy_fraction > 0.0:
            warnings.append(
                "Commercial App spectral feedback populated one or more previously empty "
                "coarse channels by copying the nearest populated channel's "
                "normalized transverse mode. Review gain_transport donor-channel "
                "diagnostics before interpreting wavelength-resolved beam shape."
            )
        if (
            self._commercial_app_temporal_envelope_enabled()
            and self._commercial_app_maximum_spectral_support_leakage_fraction > 0.01
        ):
            warnings.append(
                "The Commercial App 1-D SPM envelope placed more than 1% of its energy "
                "outside the coarse gain-spectrum support. The out-of-support "
                "energy is retained in the reported coherent spectrum but is "
                "assigned to the nearest edge gain bin for population feedback; "
                "widen/increase seed.spectrum sampling before using the result quantitatively."
            )
        if (
            self._commercial_app_temporal_envelope_enabled()
            and self._commercial_app_maximum_cumulative_b_integral_rad
            > float(
                self.simulation.get("strict_validation", {}).get(
                    "maximum_commercial_app_reduced_spatiotemporal_B_integral_rad",
                    1.0,
                )
            )
        ):
            warnings.append(
                "The Commercial App effective-area envelope plus Gaussian-q Kerr lens is "
                "outside its configured nonlinear-phase applicability limit. "
                "This separable closure can show an SPM trend, but it cannot "
                "certify the output spectrum when radial spectral broadening "
                "and Kerr self-focusing are strongly coupled; use the coherent "
                "full-field or coherent Kerr hybrid profile."
            )
        if self._commercial_app_beam_quality_model == "projection_residual":
            warnings.append(
                "Commercial App output M2 includes a residual-aberration second-moment proxy. "
                "The propagated field is still Gaussian-q; use coherent_cpa/FFT "
                "for a full-field ISO-style M2 when the proxy differs materially "
                "from the configured M2."
            )
        result.warnings = list(dict.fromkeys(warnings))
        return self._aggregate_commercial_app_hg_result(result)

    def _aggregate_commercial_app_hg_result(self, result: Any) -> Any:
        """Collapse internal mode×wavelength channels for public reporting."""

        if not self._commercial_app_hg_enabled:
            return result
        count = len(self._commercial_app_reported_gain_wavelengths_m)

        def aggregate(values: np.ndarray | None) -> np.ndarray | None:
            if values is None:
                return None
            source = np.asarray(values, dtype=float)
            if source.shape[0] != len(self._commercial_app_modal_channels):
                return source
            target = np.zeros((count, *source.shape[1:]), dtype=float)
            for channel_index, channel in enumerate(self._commercial_app_modal_channels):
                target[int(channel["spectral_index"])] += source[channel_index]
            return target

        result.wavelengths_m = self._commercial_app_reported_gain_wavelengths_m.copy()
        result.input_spectrum_J = aggregate(result.input_spectrum_J)
        result.output_spectrum_J = aggregate(result.output_spectrum_J)
        result.average_spectral_intensity_W_m2 = aggregate(
            result.average_spectral_intensity_W_m2
        )
        if result.source_spatio_spectral is not None:
            result.source_spatio_spectral.update(
                {
                    "spatial_mode_model": "incoherent_hg",
                    "spatial_mode_count": len(self._commercial_app_incoherent_modes),
                    "mode_spectral_channel_count": len(
                        self._commercial_app_modal_channels
                    ),
                    "reported_spectrum_aggregated_over_modes": True,
                    "modal_population_coupling": (
                        "simultaneous_local_shared_population"
                    ),
                    "modal_channels": [
                        dict(channel) for channel in self._commercial_app_modal_channels
                    ],
                }
            )
        return result

    def propagate(self, population: np.ndarray, delta_n: np.ndarray) -> Any:
        self._commercial_app_active_repetition_rate_Hz = None
        self._reset_commercial_app_diagnostics()
        self._apply_commercial_app_element_dispersion_phase(self.pre_elements)
        result = super().propagate(population, delta_n)
        return self._finalize_commercial_app_result(result)

    def propagate_average(
        self,
        population: np.ndarray,
        delta_n: np.ndarray,
        repetition_rate_Hz: float,
        record_planes: bool = False,
    ) -> Any:
        self._commercial_app_active_repetition_rate_Hz = float(repetition_rate_Hz)
        self._reset_commercial_app_diagnostics()
        self._apply_commercial_app_element_dispersion_phase(self.pre_elements)
        result = super().propagate_average(
            population,
            delta_n,
            repetition_rate_Hz,
            record_planes,
        )
        return self._finalize_commercial_app_result(result)

    def _commercial_app_local_parabolic_delta_n(
        self,
        delta_n_slice: np.ndarray,
        z_index: int,
    ) -> np.ndarray:
        cached = self._commercial_app_thermal_lens_fit_cache.get(int(z_index))
        if cached is not None:
            return cached
        delta_n = np.asarray(delta_n_slice, dtype=float)
        radius_x = self._commercial_app_thermal_lens_radius_x_m
        radius_y = self._commercial_app_thermal_lens_radius_y_m
        if radius_x is None or radius_y is None:
            raise ValueError(
                "local_parabolic Commercial App thermal lens requires explicit fit radii"
            )
        xx, yy = self.grid.transverse_mesh
        local_x = xx - self._commercial_app_thermal_lens_center_x_m
        local_y = yy - self._commercial_app_thermal_lens_center_y_m
        normalized_x = local_x / float(radius_x)
        normalized_y = local_y / float(radius_y)
        ellipse = normalized_x**2 + normalized_y**2 <= 1.0
        material = np.asarray(self.crystal_mask, dtype=bool)
        selected = material & ellipse & np.isfinite(delta_n)
        point_count = int(np.count_nonzero(selected))
        if point_count < 6:
            self._commercial_app_thermal_lens_fit_fallback_count += 1
            self._commercial_app_thermal_lens_fit_records[int(z_index)] = {
                "valid": False,
                "z_index": int(z_index),
                "point_count": point_count,
                "reason": "fewer than six material cells lie inside the fit ellipse",
            }
            self._commercial_app_thermal_lens_fit_cache[int(z_index)] = delta_n
            if self._commercial_app_thermal_lens_fit_fallback_count == 1:
                self._commercial_app_runtime_warnings.append(
                    "Commercial App local thermal-lens fitting had fewer than six cells in "
                    "at least one z slice and retained its sampled phase screen."
                )
            return delta_n
        design_columns = [
            np.ones(point_count),
            normalized_x[selected],
            normalized_y[selected],
            normalized_x[selected] ** 2,
            normalized_y[selected] ** 2,
        ]
        if self._commercial_app_thermal_lens_quadratic_model == "full_xy":
            design_columns.append(
                normalized_x[selected] * normalized_y[selected]
            )
        design = np.column_stack(design_columns)
        weights = np.sqrt(self.grid.cell_areas[selected])
        coefficients, _, rank, _ = np.linalg.lstsq(
            design * weights[:, None],
            delta_n[selected] * weights,
            rcond=None,
        )
        if rank < design.shape[1]:
            self._commercial_app_thermal_lens_fit_fallback_count += 1
            self._commercial_app_thermal_lens_fit_records[int(z_index)] = {
                "valid": False,
                "z_index": int(z_index),
                "point_count": point_count,
                "reason": "fit ellipse does not resolve the two-dimensional quadratic basis",
            }
            self._commercial_app_thermal_lens_fit_cache[int(z_index)] = delta_n
            if self._commercial_app_thermal_lens_fit_fallback_count == 1:
                self._commercial_app_runtime_warnings.append(
                    "Commercial App local thermal-lens fitting was rank deficient in at "
                    "least one z slice and retained its sampled phase screen."
                )
            return delta_n
        coefficients = stabilize_quadratic_fit_coefficients(
            coefficients,
            delta_n[selected],
        )
        fitted_selected = np.sum(design * coefficients[None, :], axis=1)
        if self._commercial_app_thermal_lens_quadratic_model == "axis_aligned":
            coefficients = np.concatenate(
                [coefficients, np.asarray([0.0], dtype=float)]
            )
        residual = delta_n[selected] - fitted_selected
        area = self.grid.cell_areas[selected]
        rmse = float(
            np.sqrt(np.sum(area * residual**2) / np.sum(area))
        )
        normalized_rmse = normalized_fit_rmse(delta_n[selected], rmse)
        fitted = (
            coefficients[0]
            + coefficients[1] * normalized_x
            + coefficients[2] * normalized_y
            + coefficients[3] * normalized_x**2
            + coefficients[4] * normalized_y**2
            + coefficients[5] * normalized_x * normalized_y
        )
        fitted = np.where(material, fitted, 0.0)
        fitted.setflags(write=False)
        self._commercial_app_thermal_lens_fit_cache[int(z_index)] = fitted
        quadratic_x = float(coefficients[3]) / float(radius_x) ** 2
        quadratic_y = float(coefficients[4]) / float(radius_y) ** 2
        quadratic_xy = float(coefficients[5]) / (
            float(radius_x) * float(radius_y)
        )
        slice_axis_coupling_ratio = abs(quadratic_xy) / max(
            abs(2.0 * quadratic_x),
            abs(2.0 * quadratic_y),
            np.finfo(float).tiny,
        )
        self._commercial_app_thermal_lens_fit_records[int(z_index)] = {
            "valid": True,
            "z_index": int(z_index),
            "point_count": point_count,
            "quadratic_x_per_m2": quadratic_x,
            "quadratic_y_per_m2": quadratic_y,
            "quadratic_xy_per_m2": quadratic_xy,
            "axis_coupling_ratio": slice_axis_coupling_ratio,
            "rmse": rmse,
            "normalized_rmse": normalized_rmse,
        }
        if (
            slice_axis_coupling_ratio > self._commercial_app_axis_coupling_ratio_limit
            and not self._commercial_app_axis_coupling_warning_emitted
        ):
            self._commercial_app_runtime_warnings.append(
                "Commercial App local-parabolic thermal lens contains a significant xy "
                "curvature term. The sampled phase is applied, but the following "
                "axis-aligned q projection cannot retain rotated astigmatism; "
                "strict validation will mark this reduced result non-quantitative."
            )
            self._commercial_app_axis_coupling_warning_emitted = True
        return fitted

    def _apply_thermal_phase(
        self,
        field: np.ndarray,
        delta_n_slice: np.ndarray,
        step_length_m: float,
        z_index: int,
    ) -> np.ndarray:
        if self._commercial_app_thermal_lens_model == "local_parabolic":
            delta_n_slice = self._commercial_app_local_parabolic_delta_n(
                delta_n_slice,
                z_index,
            )
        if self._commercial_app_needs_phase_conversion() and np.any(delta_n_slice):
            # The effective-M2 carrier stores transverse tilt/curvature divided
            # by M2, whereas delta-n is a physical optical-path increment.
            # Feeding the unscaled screen into _q_from_projection would make
            # the physical thermal lens M2 times too strong.
            phase = (
                2.0 * math.pi * float(step_length_m)
                * np.asarray(delta_n_slice)[None, :, :]
                / np.asarray(self.spectrum.coordinate)[:, None, None]
            )
            return self._apply_commercial_app_physical_phase(field, phase)
        return super()._apply_thermal_phase(
            field,
            delta_n_slice,
            step_length_m,
            z_index,
        )

    def _commercial_app_needs_phase_conversion(self) -> bool:
        return not self._commercial_app_hg_enabled and any(
            mx != 1.0 or my != 1.0
            for mx, my in (
                self._commercial_app_channel_m2(index)
                for index in range(len(self.spectrum.coordinate))
            )
        )

    def _apply_commercial_app_physical_phase(
        self,
        field: np.ndarray,
        physical_phase: np.ndarray,
        *,
        channel_indices: list[int] | None = None,
        already_applied: bool = False,
    ) -> np.ndarray:
        """Convert physical phase moments to the effective-M2 q carrier.

        For a phase increment phi the physical mean transverse momentum changes
        by <grad phi>, and its position covariance by <(x-<x>) grad phi>.
        The axis-aligned q carrier encodes those two moments divided by the
        respective M2. Leave the residual phase in place for the subsequent
        Gaussian projection/overlap audit; its weighted gradient moments vanish.
        Explicit HG channels already carry their physical phase and need no
        conversion. Neither physical B nor the 1-D SPM phase is rescaled.
        """

        phase = np.asarray(physical_phase, dtype=float)
        if not self._commercial_app_needs_phase_conversion():
            return field if already_applied else field * np.exp(1j * phase)[None, ...]
        correction = np.zeros_like(phase)
        indices = range(phase.shape[0]) if channel_indices is None else channel_indices
        xx, yy = self.grid.transverse_mesh
        for local_index, channel_index in enumerate(indices):
            if self._commercial_app_hg_enabled:
                continue
            m2_x, m2_y = self._commercial_app_channel_m2(channel_index)
            if m2_x == 1.0 and m2_y == 1.0:
                continue
            intensity = np.sum(np.abs(field[:, local_index]) ** 2, axis=0)
            power = float(self.grid.integrate_transverse(intensity))
            if power <= np.finfo(float).tiny:
                continue
            if hasattr(self.grid, "transverse_derivatives"):
                gradient_x, gradient_y = self.grid.transverse_derivatives(phase[local_index])
            else:
                gradient_x = np.gradient(phase[local_index], self.grid.x, axis=-1,
                                         edge_order=2 if len(self.grid.x) >= 3 else 1)
                gradient_y = np.gradient(phase[local_index], self.grid.y, axis=-2,
                                         edge_order=2 if len(self.grid.y) >= 3 else 1)
            for coordinate, gradient, m2 in (
                (xx, gradient_x, m2_x), (yy, gradient_y, m2_y)
            ):
                if m2 == 1.0:
                    continue
                center = float(self.grid.integrate_transverse(intensity * coordinate) / power)
                local = coordinate - center
                variance = float(self.grid.integrate_transverse(intensity * local**2) / power)
                if variance <= np.finfo(float).tiny:
                    continue
                # Polar theta differentiation uses a complex FFT; phi is real.
                gradient = np.real(gradient)
                tilt = float(self.grid.integrate_transverse(intensity * gradient) / power)
                curvature = float(self.grid.integrate_transverse(intensity * local * gradient) / power)
                correction[local_index] += (1.0 / m2 - 1.0) * (
                    tilt * local + 0.5 * curvature / variance * local**2
                )
        return field * np.exp(1j * (correction if already_applied else phase + correction))[None, ...]

    def _gain_interface(self, field, *args, channel_index=0, **kwargs):
        result = super()._gain_interface(field, *args, channel_index=channel_index, **kwargs)
        sag = kwargs.get("surface_sag_m")
        if sag is None or not self._commercial_app_needs_phase_conversion():
            return result
        n_before, n_after, _marker, side, _model, wavelength = args
        entering = kwargs["entering"]
        n_crystal = n_after if entering else n_before
        n_outside = n_before if entering else n_after
        phase = (2.0 * math.pi / float(wavelength)
                 * (-1.0 if side == "front" else 1.0)
                 * (float(n_crystal) - float(n_outside)) * np.asarray(sag))
        return self._apply_commercial_app_physical_phase(
            result[:, None], phase[None], channel_indices=[channel_index], already_applied=True
        )[:, 0]

    def _peak_intensity_from_field(self, field: np.ndarray) -> np.ndarray:
        density = np.sum(np.abs(field) ** 2, axis=(0, 1))
        if self._commercial_app_is_cw:
            return density
        if self._commercial_app_temporal_envelope_enabled():
            if self._commercial_app_active_repetition_rate_Hz is not None:
                density = density / max(
                    self._commercial_app_active_repetition_rate_Hz,
                    np.finfo(float).tiny,
                )
            temporal = self._commercial_app_pulse.spectrum_to_time(
                self._commercial_app_coherent_spectrum
            )
            temporal_energy = np.abs(temporal) ** 2
            total_energy = float(np.sum(temporal_energy))
            peak_power = float(np.max(temporal_energy)) / max(
                self._commercial_app_pulse.dt_s,
                np.finfo(float).tiny,
            )
            return density * peak_power / max(
                total_energy,
                np.finfo(float).tiny,
            )
        denominator = max(self._commercial_app_effective_duration_s, np.finfo(float).tiny)
        if self._commercial_app_active_repetition_rate_Hz is not None:
            denominator *= max(
                self._commercial_app_active_repetition_rate_Hz,
                np.finfo(float).tiny,
            )
        return density / denominator

    def _apply_commercial_app_kerr(
        self,
        field: np.ndarray,
        distance_m: float,
    ) -> np.ndarray:
        field = self._apply_commercial_app_temporal_step(field, distance_m)
        if (
            self.physics.get("nonlinear_phase_model", "off")
            != "kerr_parabolic_lens"
        ):
            return field
        n2 = float(self.material.get("n2_m2_W", 0.0))
        if n2 == 0.0 or distance_m == 0.0:
            return field
        marker = getattr(self, "_active_gain_marker", self.gain_marker)
        projected_area_cosine = gain_medium_path_cosine(marker)
        peak_intensity = (
            self._peak_intensity_from_field(field) / projected_area_cosine
        )
        self._commercial_app_maximum_peak_intensity_W_m2 = max(
            self._commercial_app_maximum_peak_intensity_W_m2,
            float(np.max(peak_intensity)),
        )
        wavelengths = np.asarray(self.spectrum.coordinate, dtype=float)
        nonlinear_phase = (
            2.0
            * math.pi
            * n2
            * float(distance_m)
            * peak_intensity[None, :, :]
            / wavelengths[:, None, None]
        )
        nonlinear_phase *= self.crystal_mask[None, :, :]
        reference_index = int(
            np.argmin(np.abs(wavelengths - float(self.cfg["wavelength_m"])))
        )
        reference_phase = nonlinear_phase[reference_index]
        self._commercial_app_cumulative_b_integral_rad += reference_phase
        self._commercial_app_maximum_single_step_b_integral_rad = max(
            self._commercial_app_maximum_single_step_b_integral_rad,
            float(np.max(np.abs(reference_phase))),
        )
        self._commercial_app_maximum_cumulative_b_integral_rad = max(
            self._commercial_app_maximum_cumulative_b_integral_rad,
            float(np.max(np.abs(self._commercial_app_cumulative_b_integral_rad))),
        )
        return self._apply_commercial_app_physical_phase(field, nonlinear_phase)

    def _commercial_app_temporally_collapsed_field(
        self, field: np.ndarray
    ) -> np.ndarray:
        """Collapse incoherent time packets onto one shared spatial carrier."""

        values = np.asarray(field, dtype=complex)
        if values.ndim != 4:
            raise ValueError(
                "Commercial App field must have shape (time, wavelength, y, x)"
            )
        if values.shape[0] == 1:
            return values[0]
        intensity = np.sum(np.abs(values) ** 2, axis=0)
        # Intensity-weighted unit phasors retain a common tilt/curvature while
        # avoiding coherent interference between separate temporal packets.
        phasor = np.sum(values * np.abs(values), axis=0)
        unit_phase = np.divide(
            phasor,
            np.abs(phasor),
            out=np.ones_like(phasor),
            where=np.abs(phasor) > np.finfo(float).tiny,
        )
        return np.sqrt(np.maximum(intensity, 0.0)) * unit_phase

    def _commercial_app_temporal_mode_coefficients(
        self,
        field: np.ndarray,
        shared_projected_field: np.ndarray,
    ) -> np.ndarray:
        """Return energy-preserving complex weights for a shared q mode."""

        values = np.asarray(field, dtype=complex)
        energy = np.asarray(
            self.grid.integrate_transverse(np.abs(values) ** 2),
            dtype=float,
        )
        total = np.sum(energy, axis=0)
        amplitudes = np.sqrt(
            np.divide(
                energy,
                total[None, :],
                out=np.zeros_like(energy),
                where=total[None, :] > np.finfo(float).tiny,
            )
        )
        overlap = self.grid.integrate_transverse(
            np.conj(shared_projected_field)[None, ...] * values
        )
        unit_phase = np.divide(
            overlap,
            np.abs(overlap),
            out=np.ones_like(overlap),
            where=np.abs(overlap) > np.finfo(float).tiny,
        )
        return amplitudes * unit_phase

    def _project_commercial_app_field(self, field: np.ndarray) -> GaussianProjection:
        collapsed = self._commercial_app_temporally_collapsed_field(field)
        projection = project_to_gaussian_mode(
            collapsed,
            self.grid,
            derivative_method="finite_difference",
            compute_sampled_m2=not bool(
                self.physics.get("_suppress_output_diagnostics", False)
            ),
        )
        overlaps = np.asarray(projection.coherent_overlap, dtype=float)
        if self._commercial_app_hg_enabled:
            matched = np.empty_like(collapsed)
            matched_overlaps = np.ones(len(self.spectrum.coordinate), dtype=float)
            xx, yy = self.grid.transverse_mesh
            tiny = np.finfo(float).tiny
            for index in range(len(self.spectrum.coordinate)):
                quantity = float(
                    self.grid.integrate_transverse(np.abs(collapsed[index]) ** 2)
                )
                if quantity <= tiny:
                    matched[index] = collapsed[index]
                    continue
                order_x, order_y = self._commercial_app_channel_orders(index)
                mode_m2_x, mode_m2_y = self._commercial_app_channel_m2(index)
                base_radius_x = float(projection.radius_x_m[index]) / math.sqrt(
                    mode_m2_x
                )
                base_radius_y = float(projection.radius_y_m[index]) / math.sqrt(
                    mode_m2_y
                )
                local_x = xx - float(projection.center_x_m[index])
                local_y = yy - float(projection.center_y_m[index])
                candidate = single_hg_mode_amplitude(
                    xx,
                    yy,
                    base_radius_x,
                    base_radius_y,
                    order_x,
                    order_y,
                    center_x_m=float(projection.center_x_m[index]),
                    center_y_m=float(projection.center_y_m[index]),
                ).astype(complex)
                candidate *= np.exp(
                    1j
                    * (
                        float(projection.mean_kx_per_m[index]) * local_x
                        + float(projection.mean_ky_per_m[index]) * local_y
                        + float(projection.quadratic_phase_x_per_m2[index])
                        * local_x**2
                        + float(projection.quadratic_phase_y_per_m2[index])
                        * local_y**2
                    )
                )
                sampled = float(
                    self.grid.integrate_transverse(np.abs(candidate) ** 2)
                )
                candidate *= math.sqrt(quantity / max(sampled, tiny))
                inner = self.grid.integrate_transverse(
                    np.conj(candidate) * collapsed[index]
                )
                if abs(inner) > tiny:
                    candidate *= np.exp(1j * np.angle(inner))
                candidate_quantity = float(
                    self.grid.integrate_transverse(np.abs(candidate) ** 2)
                )
                matched_overlaps[index] = float(
                    np.clip(
                        abs(inner) ** 2
                        / max(quantity * candidate_quantity, tiny),
                        0.0,
                        1.0,
                    )
                )
                matched[index] = candidate
            projection = GaussianProjection(
                field=matched,
                coherent_overlap=matched_overlaps,
                radius_x_m=projection.radius_x_m,
                radius_y_m=projection.radius_y_m,
                center_x_m=projection.center_x_m,
                center_y_m=projection.center_y_m,
                mean_kx_per_m=projection.mean_kx_per_m,
                mean_ky_per_m=projection.mean_ky_per_m,
                quadratic_phase_x_per_m2=projection.quadratic_phase_x_per_m2,
                quadratic_phase_y_per_m2=projection.quadratic_phase_y_per_m2,
                sampled_m2_x=projection.sampled_m2_x,
                sampled_m2_y=projection.sampled_m2_y,
            )
            overlaps = matched_overlaps
        if not self._commercial_app_m2_channels_initialized:
            if self._commercial_app_hg_enabled:
                modal_m2 = np.asarray(
                    [
                        self._commercial_app_channel_m2(index)
                        for index in range(len(self.spectrum.coordinate))
                    ],
                    dtype=float,
                )
                self._commercial_app_configured_m2_x = modal_m2[:, 0]
                self._commercial_app_configured_m2_y = modal_m2[:, 1]
            else:
                configs = getattr(
                    self,
                    "_wavelength_source_configs",
                    [self.cfg] * len(self.spectrum.coordinate),
                )
                self._commercial_app_configured_m2_x = np.asarray(
                    [float(config.get("m2_x", 1.0)) for config in configs],
                    dtype=float,
                )
                self._commercial_app_configured_m2_y = np.asarray(
                    [float(config.get("m2_y", 1.0)) for config in configs],
                    dtype=float,
                )
            self._commercial_app_proxy_m2_squared_x = self._commercial_app_configured_m2_x**2
            self._commercial_app_proxy_m2_squared_y = self._commercial_app_configured_m2_y**2
            self._commercial_app_maximum_discarded_m2_x = np.ones_like(
                self._commercial_app_configured_m2_x
            )
            self._commercial_app_maximum_discarded_m2_y = np.ones_like(
                self._commercial_app_configured_m2_y
            )
            self._commercial_app_m2_channels_initialized = True
        sampled_m2_x = np.nan_to_num(
            np.asarray(projection.sampled_m2_x, dtype=float),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )
        sampled_m2_y = np.nan_to_num(
            np.asarray(projection.sampled_m2_y, dtype=float),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )
        if self._commercial_app_hg_enabled:
            sampled_m2_x = sampled_m2_x / np.maximum(
                self._commercial_app_configured_m2_x, 1.0
            )
            sampled_m2_y = sampled_m2_y / np.maximum(
                self._commercial_app_configured_m2_y, 1.0
            )
        self._commercial_app_last_discarded_m2_x = sampled_m2_x.copy()
        self._commercial_app_last_discarded_m2_y = sampled_m2_y.copy()
        self._commercial_app_maximum_discarded_m2_x = np.maximum(
            self._commercial_app_maximum_discarded_m2_x,
            sampled_m2_x,
        )
        self._commercial_app_maximum_discarded_m2_y = np.maximum(
            self._commercial_app_maximum_discarded_m2_y,
            sampled_m2_y,
        )
        if self._commercial_app_beam_quality_model == "projection_residual":
            # Every q regeneration is diffraction limited in its coherent
            # carrier.  Excess determinant above M2=1 is therefore the angular
            # variance discarded by this projection.  Independent discarded
            # residuals are accumulated in quadrature as an explicit Commercial App
            # second-moment proxy; a small floor prevents finite-difference
            # roundoff from accumulating over hundreds of clean projections.
            significant = overlaps < (1.0 - 1e-6)
            excess_x = np.maximum(sampled_m2_x**2 - 1.001**2, 0.0)
            excess_y = np.maximum(sampled_m2_y**2 - 1.001**2, 0.0)
            self._commercial_app_proxy_m2_squared_x += np.where(
                significant, excess_x, 0.0
            )
            self._commercial_app_proxy_m2_squared_y += np.where(
                significant, excess_y, 0.0
            )
        self._commercial_app_projection_count += int(overlaps.size)
        if overlaps.size:
            self._commercial_app_minimum_projection_overlap = min(
                self._commercial_app_minimum_projection_overlap,
                float(np.min(overlaps)),
            )
        return projection

    @staticmethod
    def _q_from_projection(
        projection: GaussianProjection,
        index: int,
        wavelength_m: float,
        refractive_index: float,
        m2_x: float,
        m2_y: float,
        *,
        explicit_hg: bool = False,
    ) -> tuple[complex, complex, float, float]:
        tiny = np.finfo(float).tiny
        radius_x = float(projection.radius_x_m[index])
        radius_y = float(projection.radius_y_m[index])
        if radius_x <= tiny or radius_y <= tiny:
            raise ValueError("zero-width Gaussian projection")
        phase_scale_x = 1.0 if explicit_hg else m2_x
        phase_scale_y = 1.0 if explicit_hg else m2_y
        inverse_qx = complex(
            float(projection.quadratic_phase_x_per_m2[index])
            * wavelength_m
            * phase_scale_x
            / (math.pi * refractive_index),
            -m2_x
            * wavelength_m
            / (math.pi * refractive_index * radius_x**2),
        )
        inverse_qy = complex(
            float(projection.quadratic_phase_y_per_m2[index])
            * wavelength_m
            * phase_scale_y
            / (math.pi * refractive_index),
            -m2_y
            * wavelength_m
            / (math.pi * refractive_index * radius_y**2),
        )
        if abs(inverse_qx) <= tiny or abs(inverse_qy) <= tiny:
            raise ValueError("singular Gaussian q projection")
        qx = 1.0 / inverse_qx
        qy = 1.0 / inverse_qy
        if qx.imag <= 0.0 or qy.imag <= 0.0:
            raise ValueError("non-physical Gaussian q projection")
        k0n = 2.0 * math.pi * refractive_index / wavelength_m
        direction_x = (
            float(projection.mean_kx_per_m[index]) * phase_scale_x / k0n
        )
        direction_y = (
            float(projection.mean_ky_per_m[index]) * phase_scale_y / k0n
        )
        transverse_squared = direction_x**2 + direction_y**2
        if not np.isfinite(transverse_squared) or transverse_squared >= 1.0:
            raise ValueError("projected Gaussian carrier has no forward direction")
        direction_z = math.sqrt(max(1.0 - transverse_squared, tiny))
        angle_x = math.atan2(direction_x, direction_z)
        angle_y = math.atan2(direction_y, direction_z)
        return qx, qy, angle_x, angle_y

    def _regenerate_gaussian(
        self,
        quantity: float,
        qx: complex,
        qy: complex,
        center_x_m: float,
        center_y_m: float,
        angle_x_rad: float,
        angle_y_rad: float,
        wavelength_m: float,
        refractive_index: float,
        m2_x: float,
        m2_y: float,
        order_x: int = 0,
        order_y: int = 0,
        *,
        explicit_hg: bool = False,
    ) -> np.ndarray:
        inverse_qx = 1.0 / qx
        inverse_qy = 1.0 / qy
        if inverse_qx.imag >= 0.0 or inverse_qy.imag >= 0.0:
            raise ValueError("advanced Gaussian q has non-physical width")
        radius_x = math.sqrt(
            -m2_x
            * wavelength_m
            / (math.pi * refractive_index * inverse_qx.imag)
        )
        radius_y = math.sqrt(
            -m2_y
            * wavelength_m
            / (math.pi * refractive_index * inverse_qy.imag)
        )
        xx, yy = self.grid.transverse_mesh
        local_x = xx - center_x_m
        local_y = yy - center_y_m
        if order_x or order_y:
            amplitude = single_hg_mode_amplitude(
                xx,
                yy,
                radius_x / math.sqrt(m2_x),
                radius_y / math.sqrt(m2_y),
                order_x,
                order_y,
                center_x_m=center_x_m,
                center_y_m=center_y_m,
            )
            density = amplitude**2
        else:
            amplitude = None
            density = 2.0 / (math.pi * radius_x * radius_y) * np.exp(
                -2.0
                * ((local_x / radius_x) ** 2 + (local_y / radius_y) ** 2)
            )
        sampled_density = float(self.grid.integrate_transverse(density))
        if not np.isfinite(sampled_density) or sampled_density <= 0.0:
            raise ValueError("advanced Gaussian is outside the sampled grid")
        density /= sampled_density
        phase_scale_x = 1.0 if explicit_hg else m2_x
        phase_scale_y = 1.0 if explicit_hg else m2_y
        phase = (
            math.pi
            * refractive_index
            / wavelength_m
            * (
                inverse_qx.real * local_x**2 / phase_scale_x
                + inverse_qy.real * local_y**2 / phase_scale_y
            )
        )
        direction_x, direction_y, _ = axis_direction_components(
            angle_x_rad,
            angle_y_rad,
        )
        phase += (
            2.0
            * math.pi
            * refractive_index
            / wavelength_m
            * (
                direction_x * local_x / phase_scale_x
                + direction_y * local_y / phase_scale_y
            )
        )
        regenerated = np.sqrt(max(quantity, 0.0) * density).astype(complex)
        if amplitude is not None:
            regenerated *= np.where(amplitude < 0.0, -1.0, 1.0)
        return regenerated * np.exp(1j * phase)

    def _diffract(
        self,
        field: np.ndarray,
        distance_m: float,
        refractive_index: float | np.ndarray,
    ) -> np.ndarray:
        if distance_m == 0.0:
            return field
        working = self._apply_commercial_app_kerr(field, distance_m)
        try:
            projection = self._project_commercial_app_field(working)
            temporal_coefficients = self._commercial_app_temporal_mode_coefficients(
                working,
                projection.field,
            )
            indices = np.broadcast_to(
                np.asarray(refractive_index, dtype=float),
                self.spectrum.coordinate.shape,
            )
            configs = getattr(
                self,
                "_wavelength_source_configs",
                [self.cfg] * len(indices),
            )
            result = np.zeros_like(working)
            marker = getattr(self, "_active_gain_marker", self.gain_marker)
            gain_angle_x, gain_angle_y = gain_medium_axis_angles_rad(marker)
            axial_distance = float(distance_m) * gain_medium_path_cosine(marker)
            for index, wavelength in enumerate(self.spectrum.coordinate):
                quantity = float(
                    self.grid.integrate_transverse(
                        np.abs(projection.field[index]) ** 2
                    )
                )
                if quantity <= np.finfo(float).tiny:
                    result[:, index] = working[:, index]
                    continue
                local_cfg = configs[index]
                m2_x, m2_y = self._commercial_app_channel_m2(index)
                order_x, order_y = self._commercial_app_channel_orders(index)
                qx, qy, angle_x, angle_y = self._q_from_projection(
                    projection,
                    index,
                    float(wavelength),
                    float(indices[index]),
                    m2_x,
                    m2_y,
                    explicit_hg=self._commercial_app_hg_enabled,
                )
                qx += float(distance_m)
                qy += float(distance_m)
                center_x = (
                    float(projection.center_x_m[index])
                    + float(distance_m) * math.tan(angle_x)
                    + axial_distance * math.tan(gain_angle_x)
                )
                center_y = (
                    float(projection.center_y_m[index])
                    + float(distance_m) * math.tan(angle_y)
                    + axial_distance * math.tan(gain_angle_y)
                )
                shared_mode = self._regenerate_gaussian(
                    quantity,
                    qx,
                    qy,
                    center_x,
                    center_y,
                    angle_x,
                    angle_y,
                    float(wavelength),
                    float(indices[index]),
                    m2_x,
                    m2_y,
                    order_x,
                    order_y,
                    explicit_hg=self._commercial_app_hg_enabled,
                )
                result[:, index] = (
                    temporal_coefficients[:, index, None, None]
                    * shared_mode[None, :, :]
                )
            return result
        except (FloatingPointError, OverflowError, ValueError) as exc:
            self._commercial_app_projection_fallback_count += 1
            self._commercial_app_projection_fallback_reasons.append(
                f"in-crystal q step: {type(exc).__name__}: {exc}"
            )
            return super()._diffract(working, distance_m, refractive_index)

    def _propagate_train(
        self,
        field: np.ndarray,
        elements: list[dict[str, Any]],
        refractive_index: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not elements:
            return field, np.broadcast_to(
                np.asarray(refractive_index, dtype=float),
                self.spectrum.coordinate.shape,
            ).copy()
        try:
            projection = self._project_commercial_app_field(field)
            temporal_coefficients = self._commercial_app_temporal_mode_coefficients(
                field,
                projection.field,
            )
            input_indices = np.broadcast_to(
                np.asarray(refractive_index, dtype=float),
                self.spectrum.coordinate.shape,
            )
            configs = getattr(
                self,
                "_wavelength_source_configs",
                [self.cfg] * len(input_indices),
            )
            result = np.zeros_like(field)
            output_indices: list[float] = []
            for index, wavelength in enumerate(self.spectrum.coordinate):
                quantity = float(
                    self.grid.integrate_transverse(
                        np.abs(projection.field[index]) ** 2
                    )
                )
                if quantity <= np.finfo(float).tiny:
                    result[:, index] = field[:, index]
                    output_indices.append(float(input_indices[index]))
                    continue
                local_cfg = configs[index]
                m2_x, m2_y = self._commercial_app_channel_m2(index)
                order_x, order_y = self._commercial_app_channel_orders(index)
                qx, qy, angle_x, angle_y = self._q_from_projection(
                    projection,
                    index,
                    float(wavelength),
                    float(input_indices[index]),
                    m2_x,
                    m2_y,
                    explicit_hg=self._commercial_app_hg_enabled,
                )
                focus_x = -qx.real
                focus_y = -qy.real
                state_cfg = dict(local_cfg)
                state_m2_x = 1.0 if self._commercial_app_hg_enabled else m2_x
                state_m2_y = 1.0 if self._commercial_app_hg_enabled else m2_y
                state_cfg.update(
                    {
                        "profile": "gaussian",
                        "wavelength_m": float(wavelength),
                        "source_refractive_index": float(input_indices[index]),
                        "waist_x_m": math.sqrt(
                            qx.imag
                            * state_m2_x
                            * float(wavelength)
                            / (math.pi * float(input_indices[index]))
                        ),
                        "waist_y_m": math.sqrt(
                            qy.imag
                            * state_m2_y
                            * float(wavelength)
                            / (math.pi * float(input_indices[index]))
                        ),
                        "m2_x": state_m2_x,
                        "m2_y": state_m2_y,
                        "focus_z_m": 0.0,
                        "focus_x_m": focus_x,
                        "focus_y_m": focus_y,
                        "center_x_m": float(projection.center_x_m[index])
                        + focus_x * math.tan(angle_x),
                        "center_y_m": float(projection.center_y_m[index])
                        + focus_y * math.tan(angle_y),
                        "incidence_angle_x_deg": math.degrees(angle_x),
                        "incidence_angle_y_deg": math.degrees(angle_y),
                        "rotation_deg": 0.0,
                        "wavefront_radius_m": None,
                        "order_x": 1.0,
                        "order_y": 1.0,
                        "spatial_modes": (
                            {
                                "model": "single_hg_mode",
                                "active_order_x": order_x,
                                "active_order_y": order_y,
                            }
                            if self._commercial_app_hg_enabled
                            else {"model": "effective_m2"}
                        ),
                    }
                )
                propagated, output_n, _, warnings = gaussian_abcd_to_plane(
                    state_cfg,
                    quantity,
                    elements,
                    self.grid,
                    (
                        "power_W"
                        if self._commercial_app_active_repetition_rate_Hz is not None
                        else "energy_J"
                    ),
                    self.physics["interface_model"],
                    False,
                    include_tilt_phase=True,
                    warn_non_gaussian_approximation=False,
                )
                result[:, index] = (
                    temporal_coefficients[:, index, None, None]
                    * propagated[None, :, :]
                )
                output_indices.append(float(output_n))
                self._commercial_app_runtime_warnings.extend(warnings)
            output_indices_array = np.asarray(output_indices, dtype=float)
        except (FloatingPointError, OverflowError, ValueError) as exc:
            self._commercial_app_projection_fallback_count += 1
            self._commercial_app_projection_fallback_reasons.append(
                f"external q train: {type(exc).__name__}: {exc}"
            )
            result, output_indices_array = super()._propagate_train(
                field,
                elements,
                refractive_index,
            )
        self._apply_commercial_app_element_dispersion(result, elements)
        return result, output_indices_array


class CommercialAppFastSignalPropagator(
    CommercialAppFastGaussianDuctMixin,
    SpectralSignalPropagator,
):
    """Whole-pulse Commercial App-style spectral gain propagator."""


class CommercialAppFastAveragePowerSignalPropagator(
    CommercialAppFastGaussianDuctMixin,
    AveragePowerSignalPropagator,
):
    """Cycle-average companion using the same Gaussian/Kerr closure."""
