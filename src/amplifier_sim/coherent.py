"""Coherent pulse construction, phase retrieval, and CPA dispersion helpers.

This module is intentionally independent from pump, population, and spatial
propagation code.  It works on one energy-normalized complex spectrum, so a
failure can be reported or replaced without corrupting the amplifier state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .compute_backend import fft1, ifft1
from .constants import LIGHT_SPEED
from .dispersion import material_dispersion_phase, phase_from_coefficients
from .errors import CoherentInputError, OPTIONAL_MODEL_FAILURES
from .pulse_definition import (
    build_fourier_pulse_pair,
    intensity_fwhm as interpolated_intensity_fwhm,
    retrieve_spectral_phase,
    spectral_phase_on_grid,
)
from .profiles import super_gaussian_fwhm_density
from .spectral import exact_frequency_fwhm_from_wavelength
from .units import time_scale_to_s, wavelength_scale_to_m


@dataclass(frozen=True)
class CoherentPulse:
    frequency_Hz: np.ndarray
    wavelength_m: np.ndarray
    spectral_field_sqrt_J: np.ndarray
    time_s: np.ndarray
    temporal_field_sqrt_J: np.ndarray
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]

    @property
    def spectral_energy_J(self) -> np.ndarray:
        return np.abs(self.spectral_field_sqrt_J) ** 2

    @property
    def temporal_energy_J(self) -> np.ndarray:
        return np.abs(self.temporal_field_sqrt_J) ** 2

    @property
    def dt_s(self) -> float:
        if self.time_s.size < 2:
            return 1.0
        return float(self.time_s[1] - self.time_s[0])

    @property
    def temporal_power_W(self) -> np.ndarray:
        return self.temporal_energy_J / max(self.dt_s, np.finfo(float).tiny)

    def spectrum_to_time(self, spectral_field: np.ndarray, axis: int = 0) -> np.ndarray:
        # Physical convention: A(t) = sum A(omega) exp(-i omega t).
        # exp(+i omega * delay) therefore moves the envelope to later time,
        # consistently with positive propagation, Kerr and material phases.
        shifted = np.fft.ifftshift(spectral_field, axes=axis)
        return np.fft.fftshift(
            fft1(shifted, axis=axis, norm="ortho"), axes=axis
        )

    def time_to_spectrum(self, temporal_field: np.ndarray, axis: int = 0) -> np.ndarray:
        shifted = np.fft.ifftshift(temporal_field, axes=axis)
        return np.fft.fftshift(
            ifft1(shifted, axis=axis, norm="ortho"), axes=axis
        )

    def with_added_phase(self, coefficients: dict[str, Any]) -> "CoherentPulse":
        phase = phase_from_coefficients(self.frequency_Hz, coefficients)
        spectrum = self.spectral_field_sqrt_J * np.exp(1j * phase)
        temporal = self.spectrum_to_time(spectrum)
        return CoherentPulse(
            self.frequency_Hz.copy(),
            self.wavelength_m.copy(),
            spectrum,
            self.time_s.copy(),
            temporal,
            self.warnings,
            dict(self.diagnostics),
        )


def build_coherent_pulse(seed_cfg: dict[str, Any]) -> CoherentPulse:
    coherent_cfg = seed_cfg.get("coherent", {})
    failure_policy = str(coherent_cfg.get("failure_policy", "error"))
    warnings: list[str] = []
    try:
        return _build_coherent_pulse(seed_cfg, warnings)
    except OPTIONAL_MODEL_FAILURES as exc:
        if failure_policy == "error":
            raise
        warnings.append(
            f"Coherent input failed ({type(exc).__name__}: {exc}); coefficient-based Gaussian fallback was used."
        )
        fallback_seed = dict(seed_cfg)
        fallback_cfg = dict(coherent_cfg)
        fallback_cfg["input_mode"] = "coefficients"
        fallback_seed["coherent"] = fallback_cfg
        fallback_spectrum = dict(seed_cfg.get("spectrum", {}))
        if fallback_spectrum.get("type") in {"file", "monochromatic"}:
            fallback_spectrum["type"] = "gaussian"
        fallback_seed["spectrum"] = fallback_spectrum
        fallback_definition = dict(seed_cfg.get("pulse_definition", {}))
        fallback_definition["domain"] = "independent"
        fallback_seed["pulse_definition"] = fallback_definition
        fallback_pulse = _build_coherent_pulse(fallback_seed, warnings)
        diagnostics = dict(fallback_pulse.diagnostics)
        diagnostics["input_fallback"] = {
            "active": True,
            "requested_input_mode": str(coherent_cfg.get("input_mode", "coefficients")),
            "executed_input_mode": "coefficients",
            "reason": f"{type(exc).__name__}: {exc}",
        }
        return CoherentPulse(
            fallback_pulse.frequency_Hz,
            fallback_pulse.wavelength_m,
            fallback_pulse.spectral_field_sqrt_J,
            fallback_pulse.time_s,
            fallback_pulse.temporal_field_sqrt_J,
            fallback_pulse.warnings,
            diagnostics,
        )


def _build_coherent_pulse(seed_cfg: dict[str, Any], warnings: list[str]) -> CoherentPulse:
    coherent_cfg = seed_cfg.get("coherent", {})
    count = max(16, int(coherent_cfg.get("sample_count", 128)))
    center_wavelength = float(seed_cfg["wavelength_m"])
    center_frequency = LIGHT_SPEED / center_wavelength
    input_mode = str(coherent_cfg.get("input_mode", "coefficients"))
    spectrum_cfg = seed_cfg.get("spectrum", {})
    linked_domain = str(seed_cfg.get("pulse_definition", {}).get("domain", "independent"))
    use_linked_definition = (
        linked_domain == "dual" and input_mode != "complex_spectrum"
    ) or (
        linked_domain in {"time", "spectrum"} and input_mode == "coefficients"
    )
    if use_linked_definition:
        linked = build_fourier_pulse_pair(
            seed_cfg,
            sample_count=count,
            include_additional_phase=True,
        )
        frequency = linked.frequency_Hz
        count = len(frequency)
        wavelength = linked.wavelength_m
        time = linked.time_s
        df = linked.df_Hz
        amplitude = np.abs(linked.spectral_field)
        phase = np.unwrap(np.angle(linked.spectral_field))
        warnings.extend(linked.warnings)
        retrieval = linked.diagnostics.get("phase_retrieval")
        if linked_domain == "dual":
            input_mode = "phase_retrieval" if retrieval is not None else "measured_phase"
    else:
        bandwidth_wavelength = float(
            spectrum_cfg.get(
                "bandwidth_fwhm_m", seed_cfg.get("spectral_bandwidth_m", 0.0)
            )
        )
        bandwidth_Hz_raw = spectrum_cfg.get("bandwidth_fwhm_Hz")
        configured_bandwidth_frequency = (
            0.0 if bandwidth_Hz_raw is None else float(bandwidth_Hz_raw)
        )
        density_domain = str(
            spectrum_cfg.get("spectral_density_domain", "wavelength")
        )
        if configured_bandwidth_frequency > 0.0 and bandwidth_wavelength > 0.0:
            raise CoherentInputError(
                "Specify only one of bandwidth_fwhm_m and bandwidth_fwhm_Hz"
            )
        if configured_bandwidth_frequency > 0.0 and density_domain != "frequency":
            raise CoherentInputError(
                "bandwidth_fwhm_Hz requires spectral_density_domain=frequency"
            )
        physical_span = max(
            1.0, float(spectrum_cfg.get("span_fwhm", 3.0))
        )
        if configured_bandwidth_frequency > 0.0:
            bandwidth_frequency = configured_bandwidth_frequency
        elif bandwidth_wavelength > 0.0:
            bandwidth_frequency = exact_frequency_fwhm_from_wavelength(
                center_wavelength, bandwidth_wavelength
            )
        elif input_mode != "complex_spectrum" and str(spectrum_cfg.get("type", "gaussian")) != "file":
            duration = float(seed_cfg.get("pulse_duration_s", 1e-12))
            bandwidth_frequency = 0.441 / max(duration, np.finfo(float).tiny)
            warnings.append("No finite seed bandwidth was supplied; a Gaussian transform-limited bandwidth was inferred from pulse_duration_s.")
        else:
            bandwidth_frequency = 0.0
        tabulated_cfg: dict[str, Any] | None = None
        minimum_columns = 0
        if input_mode == "complex_spectrum":
            tabulated_cfg = coherent_cfg.get("complex_spectrum", {})
            minimum_columns = 3
        elif str(spectrum_cfg.get("type", "gaussian")) == "file":
            tabulated_cfg = spectrum_cfg
            minimum_columns = 2
        if tabulated_cfg is not None:
            rows = _rows(tabulated_cfg, minimum_columns)
            scale = wavelength_scale_to_m(
                tabulated_cfg.get("wavelength_unit", "m")
            )
            sample_frequency = LIGHT_SPEED / (rows[:, 0] * scale)
            sample_minimum = float(np.min(sample_frequency))
            sample_maximum = float(np.max(sample_frequency))
            padding_factor = float(
                coherent_cfg.get("frequency_padding_factor", 1.0)
            )
            guard = (
                0.5
                * (padding_factor - 1.0)
                * (sample_maximum - sample_minimum)
            )
            if sample_minimum - guard <= 0.0:
                raise CoherentInputError(
                    "seed.coherent.frequency_padding_factor makes the absolute "
                    "frequency grid non-positive"
                )
            frequency = np.linspace(
                sample_minimum - guard,
                sample_maximum + guard,
                count,
            )
            df = float(frequency[1] - frequency[0])
        else:
            requested_span = max(
                physical_span,
                2.0,
                float(coherent_cfg.get("span_fwhm", 6.0)),
            )
            if density_domain == "wavelength" and bandwidth_wavelength > 0.0:
                span = max(
                    physical_span,
                    min(
                        requested_span,
                        0.9 * center_wavelength / bandwidth_wavelength,
                    ),
                )
                lower_wavelength = center_wavelength - span * bandwidth_wavelength
                upper_wavelength = center_wavelength + span * bandwidth_wavelength
                if lower_wavelength <= 0.0:
                    raise CoherentInputError(
                        "Coherent wavelength-domain spectrum crosses zero wavelength"
                    )
                frequency = np.linspace(
                    LIGHT_SPEED / upper_wavelength,
                    LIGHT_SPEED / lower_wavelength,
                    count,
                )
                df = float(frequency[1] - frequency[0])
            else:
                span = max(
                    physical_span,
                    min(
                        requested_span,
                        0.9
                        * center_frequency
                        / max(bandwidth_frequency, np.finfo(float).tiny),
                    ),
                )
                frequency = np.linspace(
                    center_frequency - span * bandwidth_frequency,
                    center_frequency + span * bandwidth_frequency,
                    count,
                    endpoint=False,
                )
                df = float(frequency[1] - frequency[0])
        if np.any(frequency <= 0.0):
            raise CoherentInputError("Coherent frequency grid crosses zero; reduce bandwidth/span")
        if not np.isfinite(df) or df <= 0.0:
            raise CoherentInputError("Coherent frequency grid requires a finite nonzero bandwidth")
        wavelength = LIGHT_SPEED / frequency
        time = np.fft.fftshift(np.fft.fftfreq(count, d=df))
        if input_mode == "complex_spectrum":
            spectrum = _complex_spectrum_on_grid(coherent_cfg.get("complex_spectrum", {}), frequency)
            amplitude = np.abs(spectrum)
            phase = np.unwrap(np.angle(spectrum))
            standalone_phase = spectral_phase_on_grid(seed_cfg, frequency)
            if standalone_phase is not None:
                phase += standalone_phase
            phase += phase_from_coefficients(frequency, coherent_cfg.get("phase", {}))
            phase += phase_from_coefficients(frequency, coherent_cfg.get("stretcher", {}))
            retrieval = None
        else:
            density = _spectral_density_on_grid(
                spectrum_cfg,
                frequency,
                center_frequency,
                bandwidth_frequency,
                bandwidth_wavelength,
                physical_span,
            )
            amplitude = np.sqrt(_normalize_mass(density))
            standalone_phase = spectral_phase_on_grid(seed_cfg, frequency)
            phase = np.zeros_like(frequency) if standalone_phase is None else standalone_phase
            if input_mode == "phase_retrieval":
                phase, retrieval = _retrieve_phase(
                    amplitude,
                    time,
                    seed_cfg,
                    coherent_cfg.get("phase_retrieval", {}),
                    phase,
                )
                warnings.append(
                    "Intensity-only phase retrieval can have sign/time-reversal ambiguities; use a measured complex field when the phase must be unique."
                )
            elif input_mode == "coefficients":
                retrieval = None
            else:
                raise CoherentInputError(f"Unknown coherent input_mode {input_mode!r}")
            phase += phase_from_coefficients(frequency, coherent_cfg.get("phase", {}))
            phase += phase_from_coefficients(frequency, coherent_cfg.get("stretcher", {}))
    energy = float(seed_cfg["pulse_energy_J"])
    norm = float(np.sum(amplitude**2))
    if not np.isfinite(norm) or norm <= 0.0:
        raise CoherentInputError("Coherent spectral amplitude has zero or invalid energy")
    spectral_field = amplitude * math.sqrt(max(energy, 0.0) / norm) * np.exp(1j * phase)
    temporal_field = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(spectral_field), norm="ortho")
    )
    spectral_energy = np.abs(spectral_field) ** 2
    temporal_energy = np.abs(temporal_field) ** 2
    transform_limited_temporal_field = np.fft.fftshift(
        np.fft.fft(
            np.fft.ifftshift(np.abs(spectral_field)),
            norm="ortho",
        )
    )
    transform_limited_temporal_energy = (
        np.abs(transform_limited_temporal_field) ** 2
    )
    diagnostics: dict[str, Any] = {
        "fourier_convention": "exp_minus_i_omega_t",
        "input_mode": input_mode,
        "sample_count": count,
        "frequency_padding_factor": float(
            coherent_cfg.get("frequency_padding_factor", 1.0)
        ),
        "frequency_step_Hz": df,
        "time_step_s": float(time[1] - time[0]),
        "input_fwhm_s": intensity_fwhm(time, temporal_energy),
        "transform_limited_fwhm_s": intensity_fwhm(
            time, transform_limited_temporal_energy
        ),
    }
    diagnostics["sampling"] = _coherent_sampling_diagnostics(
        frequency,
        time,
        spectral_energy,
        temporal_energy,
        transform_limited_temporal_energy,
        coherent_cfg.get("sampling_contract", {}),
    )
    if not bool(diagnostics["sampling"]["adequate"]):
        failed = ", ".join(diagnostics["sampling"]["failed_contracts"])
        warnings.append(
            "The coherent FFT input grid is below its sampling contract "
            f"({failed}). The solve may continue for diagnostics, but increase "
            "sample_count and/or physical spectral span before interpreting "
            "fine spectral structure quantitatively."
        )
    if input_mode == "complex_spectrum":
        diagnostics["complex_spectrum_amplitude_domain"] = str(
            coherent_cfg.get("complex_spectrum", {}).get(
                "amplitude_domain", "per_frequency"
            )
        )
    if retrieval is not None:
        diagnostics["phase_retrieval"] = retrieval
    return CoherentPulse(
        frequency,
        wavelength,
        spectral_field,
        time,
        temporal_field,
        tuple(warnings),
        diagnostics,
    )


def _coherent_sampling_diagnostics(
    frequency_Hz: np.ndarray,
    time_s: np.ndarray,
    spectral_energy_J: np.ndarray,
    temporal_energy_J: np.ndarray,
    transform_limited_temporal_energy_J: np.ndarray,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Audit both sides of the Fourier-grid trade-off without changing it.

    Trimming zero-valued tails of a measured spectrum is not unconditionally
    safe: it improves frequency resolution at fixed sample count but coarsens
    the time step.  The correct contract therefore measures spectral
    resolution, temporal resolution, and both periodic FFT boundaries.
    """

    frequency = np.asarray(frequency_Hz, dtype=float)
    time = np.asarray(time_s, dtype=float)
    spectral = np.maximum(np.asarray(spectral_energy_J, dtype=float), 0.0)
    temporal = np.maximum(np.asarray(temporal_energy_J, dtype=float), 0.0)
    transform_limited = np.maximum(
        np.asarray(transform_limited_temporal_energy_J, dtype=float), 0.0
    )
    count = len(frequency)
    df = float(abs(frequency[1] - frequency[0]))
    dt = float(abs(time[1] - time[0]))
    spectral_fwhm = intensity_fwhm(frequency, spectral)
    input_fwhm = intensity_fwhm(time, temporal)
    transform_limited_fwhm = intensity_fwhm(time, transform_limited)
    spectral_bins = spectral_fwhm / max(df, np.finfo(float).tiny)
    input_time_bins = input_fwhm / max(dt, np.finfo(float).tiny)
    transform_limited_time_bins = transform_limited_fwhm / max(
        dt, np.finfo(float).tiny
    )

    boundary_fraction = float(contract.get("boundary_sample_fraction", 0.05))
    boundary_count = max(
        1,
        min(count // 2, int(math.ceil(boundary_fraction * count))),
    )

    def edge_fraction(values: np.ndarray) -> float:
        total = float(np.sum(values))
        if total <= np.finfo(float).tiny:
            return 0.0
        return float(
            np.sum(values[:boundary_count]) + np.sum(values[-boundary_count:])
        ) / total

    spectral_edge = edge_fraction(spectral)
    temporal_edge = edge_fraction(temporal)
    minimum_spectral_bins = float(
        contract.get("minimum_spectral_fwhm_bins", 12.0)
    )
    minimum_input_time_bins = float(
        contract.get("minimum_input_fwhm_bins", 6.0)
    )
    minimum_transform_limited_bins = float(
        contract.get("minimum_transform_limited_fwhm_bins", 6.0)
    )
    maximum_spectral_edge = float(
        contract.get("maximum_spectral_boundary_energy_fraction", 1e-6)
    )
    maximum_temporal_edge = float(
        contract.get("maximum_input_temporal_boundary_energy_fraction", 1e-6)
    )
    contracts = {
        "spectral_fwhm_resolution": spectral_bins >= minimum_spectral_bins,
        "input_temporal_fwhm_resolution": (
            input_time_bins >= minimum_input_time_bins
        ),
        "transform_limited_temporal_resolution": (
            transform_limited_time_bins >= minimum_transform_limited_bins
        ),
        "input_spectral_boundary": spectral_edge <= maximum_spectral_edge,
        "input_temporal_boundary": temporal_edge <= maximum_temporal_edge,
    }
    failed = [name for name, passed in contracts.items() if not passed]
    support_count = 0
    if float(np.sum(spectral)) > 0.0:
        cumulative = np.cumsum(spectral) / float(np.sum(spectral))
        lower = int(np.searchsorted(cumulative, 5e-4, side="left"))
        upper = int(np.searchsorted(cumulative, 1.0 - 5e-4, side="left"))
        support_count = max(upper - lower + 1, 1)
    # A line narrower than one FFT interval has a numerically zero FWHM.
    # It is unresolved, but the recommendation itself must remain finite.
    # One observed bin is a conservative floor for this estimate; the failed
    # contract above still preserves the fact that the actual width is unknown.
    resolved_bins_for_recommendation = (
        spectral_bins
        if math.isfinite(spectral_bins) and spectral_bins > 1.0
        else 1.0
    )
    required_count = int(
        math.ceil(
            count
            * max(
                minimum_spectral_bins / resolved_bins_for_recommendation,
                1.0,
            )
        )
    )
    return {
        "adequate": not failed,
        "failed_contracts": failed,
        "contract_results": contracts,
        "frequency_span_Hz": float(count * df),
        "time_window_s": float(count * dt),
        "spectral_fwhm_Hz": spectral_fwhm,
        "spectral_fwhm_bins": spectral_bins,
        "input_temporal_fwhm_bins": input_time_bins,
        "transform_limited_temporal_fwhm_bins": transform_limited_time_bins,
        "energy_support_99_9_percent_bins": support_count,
        "input_spectral_boundary_energy_fraction": spectral_edge,
        "input_temporal_boundary_energy_fraction": temporal_edge,
        "boundary_sample_fraction_each_side": boundary_fraction,
        "recommended_minimum_sample_count_at_current_span": required_count,
        "limits": {
            "minimum_spectral_fwhm_bins": minimum_spectral_bins,
            "minimum_input_fwhm_bins": minimum_input_time_bins,
            "minimum_transform_limited_fwhm_bins": (
                minimum_transform_limited_bins
            ),
            "maximum_spectral_boundary_energy_fraction": maximum_spectral_edge,
            "maximum_input_temporal_boundary_energy_fraction": (
                maximum_temporal_edge
            ),
        },
        "grid_policy": (
            "preserve configured physical spectral span; do not trim zero tails "
            "unless temporal resolution is re-established independently"
        ),
    }


def _spectral_density_on_grid(
    cfg: dict[str, Any],
    frequency: np.ndarray,
    center_frequency: float,
    bandwidth_frequency: float,
    bandwidth_wavelength: float,
    physical_span_fwhm: float,
) -> np.ndarray:
    kind = str(cfg.get("type", "gaussian"))
    if kind == "file":
        rows = _rows(cfg, 2)
        scale = wavelength_scale_to_m(cfg.get("wavelength_unit", "m"))
        wavelength = rows[:, 0] * scale
        sample_frequency = LIGHT_SPEED / wavelength
        order = np.argsort(sample_frequency)
        sample_frequency = sample_frequency[order]
        wavelength = wavelength[order]
        if np.any(np.diff(sample_frequency) <= 0.0):
            raise CoherentInputError(
                "Tabulated coherent spectrum coordinates must be unique"
            )
        values = _validated_nonnegative(
            rows[:, 1][order], cfg, label="Coherent spectrum file"
        )
        if str(cfg.get("value_mode", "relative")) == "energy_per_bin_J":
            density_frequency = values / np.maximum(
                _bin_widths(sample_frequency),
                np.finfo(float).tiny,
            )
        elif str(cfg.get("spectral_density_domain", "wavelength")) == "wavelength":
            # For a wavelength-density table, conserve mass under
            # nu=c/lambda: rho_nu=rho_lambda*lambda^2/c.  The native
            # wavelength unit contributes only a constant factor and is
            # removed by the subsequent normalization.
            density_frequency = values * wavelength**2 / LIGHT_SPEED
        else:
            density_frequency = values
        return np.interp(
            frequency,
            sample_frequency,
            density_frequency,
            left=0.0,
            right=0.0,
        )
    density_domain = str(cfg.get("spectral_density_domain", "wavelength"))
    if density_domain == "wavelength" and bandwidth_wavelength > 0.0:
        wavelength = LIGHT_SPEED / np.asarray(frequency, dtype=float)
        offset = (
            wavelength - LIGHT_SPEED / center_frequency
        ) / bandwidth_wavelength
        jacobian = wavelength**2 / LIGHT_SPEED
    elif density_domain == "frequency":
        offset = (frequency - center_frequency) / max(
            bandwidth_frequency, np.finfo(float).tiny
        )
        jacobian = 1.0
    elif density_domain == "wavelength":
        # A transform-limited frequency bandwidth inferred from duration has
        # no wavelength-domain FWHM definition, so it is explicitly treated
        # as a frequency density.
        offset = (frequency - center_frequency) / max(
            bandwidth_frequency, np.finfo(float).tiny
        )
        jacobian = 1.0
    else:
        raise CoherentInputError(
            "spectral_density_domain must be wavelength or frequency"
        )
    support = np.abs(offset) <= float(physical_span_fwhm)
    if kind in {"monochromatic", "gaussian"}:
        return (
            np.exp(-4.0 * np.log(2.0) * offset**2)
            * jacobian
            * support
        )
    if kind == "lorentzian":
        return jacobian * support / (1.0 + 4.0 * offset**2)
    if kind == "super_gaussian":
        order = max(1.0, float(cfg.get("order", 2.0)))
        return (
            super_gaussian_fwhm_density(offset, order)
            * jacobian
            * support
        )
    raise CoherentInputError(f"Unsupported coherent spectrum type {kind!r}")


def _complex_spectrum_on_grid(cfg: dict[str, Any], frequency: np.ndarray) -> np.ndarray:
    rows = _rows(cfg, 3)
    scale = wavelength_scale_to_m(cfg.get("wavelength_unit", "m"))
    wavelength = rows[:, 0] * scale
    sample_frequency = LIGHT_SPEED / wavelength
    order = np.argsort(sample_frequency)
    sample_frequency = sample_frequency[order]
    sample_wavelength = wavelength[order]
    if np.any(np.diff(sample_frequency) <= 0.0):
        raise CoherentInputError(
            "Tabulated coherent spectrum coordinates must be unique"
        )
    data_format = str(cfg.get("format", "amplitude_phase"))
    amplitude_domain = str(cfg.get("amplitude_domain", "per_frequency"))
    if amplitude_domain == "per_frequency":
        amplitude_scale = np.ones_like(sample_frequency)
    elif amplitude_domain == "per_wavelength":
        # |E_lambda|^2 d lambda = |E_nu|^2 d nu.
        amplitude_scale = sample_wavelength / math.sqrt(LIGHT_SPEED)
    elif amplitude_domain == "per_bin":
        # Per-bin values carry energy mass rather than a coordinate density.
        # Convert them to a frequency-density amplitude before interpolation.
        bin_width_Hz = _bin_widths(sample_frequency)
        if np.any(bin_width_Hz <= 0.0) or not np.all(np.isfinite(bin_width_Hz)):
            raise CoherentInputError(
                "complex_spectrum per_bin coordinates require positive finite bin widths"
            )
        amplitude_scale = 1.0 / np.sqrt(bin_width_Hz)
    else:
        raise CoherentInputError(
            "complex_spectrum.amplitude_domain must be per_wavelength, "
            "per_frequency, or per_bin"
        )
    if data_format == "amplitude_phase":
        amplitude = _validated_nonnegative(
            rows[:, 1][order], cfg, label="Coherent complex-spectrum amplitude"
        ) * amplitude_scale
        phase = np.unwrap(rows[:, 2][order])
        interpolated_amplitude = np.interp(frequency, sample_frequency, amplitude, left=0.0, right=0.0)
        interpolated_phase = np.interp(frequency, sample_frequency, phase)
        return interpolated_amplitude * np.exp(1j * interpolated_phase)
    if data_format == "real_imag":
        real = np.interp(
            frequency,
            sample_frequency,
            rows[:, 1][order] * amplitude_scale,
            left=0.0,
            right=0.0,
        )
        imaginary = np.interp(
            frequency,
            sample_frequency,
            rows[:, 2][order] * amplitude_scale,
            left=0.0,
            right=0.0,
        )
        return real + 1j * imaginary
    raise CoherentInputError("complex_spectrum.format must be amplitude_phase or real_imag")


def _retrieve_phase(
    spectral_amplitude: np.ndarray,
    time_s: np.ndarray,
    seed_cfg: dict[str, Any],
    cfg: dict[str, Any],
    initial_phase: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_density = _temporal_density(seed_cfg.get("temporal", {}), time_s, float(seed_cfg["pulse_duration_s"]))
    target_amplitude = np.sqrt(_normalize_mass(target_density))
    return retrieve_spectral_phase(
        spectral_amplitude,
        target_amplitude,
        initial_phase,
        int(cfg.get("max_iterations", 200)),
        float(cfg.get("tolerance", 1e-8)),
    )


def _temporal_density(cfg: dict[str, Any], time_s: np.ndarray, pulse_fwhm_s: float) -> np.ndarray:
    kind = str(cfg.get("type", "gaussian"))
    if kind == "file":
        rows = _rows(cfg, 2)
        scale = time_scale_to_s(cfg.get("time_unit", "s"))
        sample_time = rows[:, 0] * scale
        order = np.argsort(sample_time)
        sample_time = sample_time[order]
        if np.any(np.diff(sample_time) <= 0.0):
            raise CoherentInputError(
                "Coherent temporal file coordinates must be unique"
            )
        values = _validated_nonnegative(
            rows[:, 1][order], cfg, label="Coherent temporal file"
        )
        return np.interp(time_s, sample_time, values, left=0.0, right=0.0)
    offset = time_s / max(pulse_fwhm_s, np.finfo(float).tiny)
    if kind == "gaussian":
        return np.exp(-4.0 * np.log(2.0) * offset**2)
    if kind == "sech2":
        factor = 2.0 * np.arccosh(np.sqrt(2.0))
        return 1.0 / np.cosh(np.clip(factor * offset, -350.0, 350.0)) ** 2
    if kind == "rectangular":
        return (np.abs(offset) <= 0.5).astype(float)
    if kind == "super_gaussian":
        order = max(1.0, float(cfg.get("order", 2.0)))
        return super_gaussian_fwhm_density(offset, order)
    raise CoherentInputError(f"Unsupported phase-retrieval temporal profile {kind!r}")


def _rows(cfg: dict[str, Any], minimum_columns: int) -> np.ndarray:
    rows = np.asarray(cfg.get("samples", []), dtype=float)
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] < minimum_columns:
        raise CoherentInputError(f"Tabulated coherent input requires at least two rows and {minimum_columns} columns")
    if not np.all(np.isfinite(rows)):
        raise CoherentInputError("Tabulated coherent input must contain only finite values")
    return rows


def _validated_nonnegative(
    values: np.ndarray, cfg: dict[str, Any], *, label: str
) -> np.ndarray:
    samples = np.asarray(values, dtype=float)
    negative = np.flatnonzero(samples < 0.0)
    if not negative.size:
        return samples
    if str(cfg.get("invalid_sample_policy", "error")) == "clip":
        return np.maximum(samples, 0.0)
    rows = ", ".join(str(int(index) + 1) for index in negative[:8])
    raise CoherentInputError(f"{label} contains negative values at row(s) {rows}")


def _bin_widths(coordinate: np.ndarray) -> np.ndarray:
    """Return midpoint-bin widths for a strictly ordered coordinate."""

    values = np.asarray(coordinate, dtype=float)
    if values.ndim != 1 or len(values) < 2 or np.any(np.diff(values) <= 0.0):
        raise CoherentInputError("Tabulated coherent spectrum coordinates must be unique")
    edges = np.empty(len(values) + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.diff(edges)


def _normalize_mass(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(np.sum(clipped))
    if not np.isfinite(total) or total <= 0.0:
        raise CoherentInputError("Requested pulse profile has zero or invalid integral")
    return clipped / total


def intensity_fwhm(coordinate: np.ndarray, intensity: np.ndarray) -> float:
    """Return an interpolated intensity FWHM on the coherent grid.

    Keep coherent diagnostics consistent with the pulse-definition path.  In
    particular, a native 5-fs FROG grid must use the two half-maximum crossings
    rather than quantizing its reported duration to a multiple of 5 fs.
    """

    return interpolated_intensity_fwhm(coordinate, intensity)
