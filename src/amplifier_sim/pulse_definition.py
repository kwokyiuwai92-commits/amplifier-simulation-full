"""Fourier-linked seed pulse definitions.

The ordinary spectral-time solver stores phase-insensitive energy bins, while
the coherent solver stores a complex spectrum.  This module gives both paths a
single, auditable source definition: either a temporal intensity or a spectral
intensity is authoritative, zero phase is assumed at that source plane, and
the conjugate domain is derived with a unitary FFT.  The physical convention
is A(t) = sum A(omega) exp(-i omega t), so spectrum-to-time uses fft and
time-to-spectrum uses ifft.  A positive spectral phase slope is a delay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import LIGHT_SPEED
from .errors import PulseDefinitionInputError
from .dispersion import phase_from_coefficients
from .profiles import super_gaussian_fwhm_density
from .spectral import exact_frequency_fwhm_from_wavelength
from .units import time_scale_to_s, wavelength_scale_to_m


@dataclass(frozen=True)
class FourierPulsePair:
    """Energy-normalized complex pulse on conjugate uniform FFT grids."""

    source_domain: str
    time_s: np.ndarray
    temporal_field: np.ndarray
    frequency_Hz: np.ndarray
    wavelength_m: np.ndarray
    spectral_field: np.ndarray
    warnings: tuple[str, ...] = ()
    temporal_target_weights: np.ndarray | None = None
    spectral_target_weights: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def temporal_weights(self) -> np.ndarray:
        return np.abs(self.temporal_field) ** 2

    @property
    def spectral_weights(self) -> np.ndarray:
        return np.abs(self.spectral_field) ** 2

    @property
    def dt_s(self) -> float:
        return float(abs(self.time_s[1] - self.time_s[0]))

    @property
    def df_Hz(self) -> float:
        return float(abs(self.frequency_Hz[1] - self.frequency_Hz[0]))


def build_fourier_pulse_pair(
    seed_cfg: dict[str, Any],
    sample_count: int | None = None,
    *,
    include_additional_phase: bool = False,
) -> FourierPulsePair:
    """Build the linked time/frequency pulse selected by ``pulse_definition``.

    Two-column files contain intensity only.  A third column is accepted as
    phase in radians.  When both intensity domains are supplied without phase,
    alternating projections infer one admissible (non-unique) complex field.

    ``include_additional_phase`` applies the configured source/stretcher
    GD/GDD/TOD/FOD after the measured/assumed/retrieved source field.  Keeping
    it explicit prevents a coherent solver from accidentally applying those
    coefficients twice.
    """

    definition = seed_cfg.get("pulse_definition", {})
    domain = str(definition.get("domain", "independent"))
    if domain not in {"time", "spectrum", "dual"}:
        raise PulseDefinitionInputError("seed.pulse_definition.domain must be time, spectrum, or dual for Fourier linking")
    count = int(sample_count or definition.get("fft_sample_count", 256))
    count = max(32, count)
    if count % 2:
        count += 1
    center_wavelength = float(seed_cfg["wavelength_m"])
    center_frequency = LIGHT_SPEED / center_wavelength
    warnings: list[str] = []

    diagnostics: dict[str, Any] = {"source_domain": domain, "fourier_convention": "exp_minus_i_omega_t"}
    diagnostics["frequency_padding_factor"] = float(
        seed_cfg.get("coherent", {}).get("frequency_padding_factor", 1.0)
    )
    temporal_target: np.ndarray | None = None
    spectral_target: np.ndarray | None = None

    if domain == "time":
        time_s, intensity, phase = _temporal_input(seed_cfg, count, center_frequency)
        temporal_target = _normalized_intensity(intensity)
        temporal_field = _normalized_field(intensity, phase)
        dt = float(time_s[1] - time_s[0])
        offset_frequency = np.fft.fftshift(np.fft.fftfreq(count, d=dt))
        frequency = center_frequency + offset_frequency
        if np.any(frequency <= 0.0):
            raise PulseDefinitionInputError("Temporal sampling creates non-positive optical frequencies; enlarge the time step")
        spectral_field = np.fft.fftshift(
            np.fft.ifft(np.fft.ifftshift(temporal_field), norm="ortho")
        )
        if not _has_phase_column(seed_cfg.get("temporal", {})):
            warnings.append("Temporal intensity has no phase column; zero temporal phase was assumed.")
        diagnostics["phase_source"] = "temporal_file" if _has_phase_column(seed_cfg.get("temporal", {})) else "flat_temporal"
    elif domain == "spectrum":
        frequency, intensity, phase = _spectral_input(seed_cfg, count)
        spectral_target = _normalized_intensity(intensity)
        independent_phase = spectral_phase_on_grid(seed_cfg, frequency)
        if independent_phase is not None:
            phase = independent_phase
        spectral_field = _normalized_field(intensity, phase)
        df = float(frequency[1] - frequency[0])
        time_s = np.fft.fftshift(np.fft.fftfreq(count, d=df))
        temporal_field = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(spectral_field), norm="ortho")
        )
        if not _has_phase_column(seed_cfg.get("spectrum", {})):
            if independent_phase is None:
                warnings.append("Spectral intensity has no phase column; flat spectral phase was assumed.")
        diagnostics["phase_source"] = (
            "separate_spectral_phase"
            if independent_phase is not None
            else "spectral_file"
            if _has_phase_column(seed_cfg.get("spectrum", {}))
            else "flat_spectral"
        )
    else:
        frequency, spectral_intensity, embedded_spectral_phase = _spectral_input(seed_cfg, count)
        df = float(frequency[1] - frequency[0])
        time_s = np.fft.fftshift(np.fft.fftfreq(count, d=df))
        temporal_intensity, temporal_phase = _temporal_profile_on_grid(seed_cfg, time_s)
        temporal_target = _normalized_intensity(temporal_intensity)
        spectral_target = _normalized_intensity(spectral_intensity)
        target_temporal_amplitude = np.sqrt(temporal_target)
        target_spectral_amplitude = np.sqrt(spectral_target)
        separate_spectral_phase = spectral_phase_on_grid(seed_cfg, frequency)
        spectrum_has_phase = _has_phase_column(seed_cfg.get("spectrum", {}))
        time_has_phase = _has_phase_column(seed_cfg.get("temporal", {}))
        if separate_spectral_phase is not None or spectrum_has_phase:
            selected_phase = separate_spectral_phase if separate_spectral_phase is not None else embedded_spectral_phase
            spectral_field = target_spectral_amplitude * np.exp(1j * selected_phase)
            temporal_field = np.fft.fftshift(
                np.fft.fft(np.fft.ifftshift(spectral_field), norm="ortho")
            )
            diagnostics["phase_source"] = (
                "separate_spectral_phase" if separate_spectral_phase is not None else "spectral_file"
            )
            warnings.append(
                "Both intensities and a spectral phase were supplied; the measured spectral complex field is authoritative, "
                "and the temporal intensity is retained as a comparison target."
            )
        elif time_has_phase:
            temporal_field = target_temporal_amplitude * np.exp(1j * temporal_phase)
            spectral_field = np.fft.fftshift(
                np.fft.ifft(np.fft.ifftshift(temporal_field), norm="ortho")
            )
            diagnostics["phase_source"] = "temporal_file"
            warnings.append(
                "Both intensities and a temporal phase were supplied; the measured temporal complex field is authoritative, "
                "and the spectral intensity is retained as a comparison target."
            )
        else:
            retrieval_cfg = seed_cfg.get("coherent", {}).get("phase_retrieval", {})
            retrieved_phase, retrieval = retrieve_spectral_phase(
                target_spectral_amplitude,
                target_temporal_amplitude,
                np.zeros(count),
                int(retrieval_cfg.get("max_iterations", 200)),
                float(retrieval_cfg.get("tolerance", 1e-8)),
            )
            spectral_field = target_spectral_amplitude * np.exp(1j * retrieved_phase)
            temporal_field = np.fft.fftshift(
                np.fft.fft(np.fft.ifftshift(spectral_field), norm="ortho")
            )
            diagnostics["phase_source"] = "dual_intensity_retrieval"
            diagnostics["phase_retrieval"] = retrieval
            warnings.append(
                "Both intensity domains lacked phase; an admissible phase was retrieved. "
                "Intensity-only phase retrieval is not unique and can retain sign/time-reversal ambiguities."
            )

    if domain == "time":
        independent_phase = spectral_phase_on_grid(seed_cfg, frequency)
        if independent_phase is not None:
            spectral_field *= np.exp(1j * independent_phase)
            temporal_field = np.fft.fftshift(
                np.fft.fft(np.fft.ifftshift(spectral_field), norm="ortho")
            )
            diagnostics["phase_source"] = f"{diagnostics['phase_source']}+separate_spectral_phase"

    if include_additional_phase:
        coherent_cfg = seed_cfg.get("coherent", {})
        added_phase = phase_from_coefficients(frequency, coherent_cfg.get("phase", {}))
        added_phase += phase_from_coefficients(frequency, coherent_cfg.get("stretcher", {}))
        if np.any(np.abs(added_phase) > 0.0):
            spectral_field *= np.exp(1j * added_phase)
            temporal_field = np.fft.fftshift(
                np.fft.fft(np.fft.ifftshift(spectral_field), norm="ortho")
            )
            diagnostics["additional_phase_applied"] = True
        else:
            diagnostics["additional_phase_applied"] = False

    if temporal_target is not None:
        diagnostics["temporal_target_relative_amplitude_error"] = _relative_amplitude_error(
            np.abs(temporal_field), np.sqrt(temporal_target)
        )
    if spectral_target is not None:
        diagnostics["spectral_target_relative_amplitude_error"] = _relative_amplitude_error(
            np.abs(spectral_field), np.sqrt(spectral_target)
        )

    return FourierPulsePair(
        source_domain=domain,
        time_s=np.asarray(time_s, dtype=float),
        temporal_field=np.asarray(temporal_field, dtype=complex),
        frequency_Hz=np.asarray(frequency, dtype=float),
        wavelength_m=LIGHT_SPEED / np.asarray(frequency, dtype=float),
        spectral_field=np.asarray(spectral_field, dtype=complex),
        warnings=tuple(warnings),
        temporal_target_weights=temporal_target,
        spectral_target_weights=spectral_target,
        diagnostics=diagnostics,
    )


def retrieve_spectral_phase(
    spectral_amplitude: np.ndarray,
    temporal_amplitude: np.ndarray,
    initial_phase: np.ndarray,
    maximum_iterations: int = 200,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Retrieve one spectral phase compatible with two intensity constraints."""

    spectral_amplitude = np.asarray(spectral_amplitude, dtype=float)
    temporal_amplitude = np.asarray(temporal_amplitude, dtype=float)
    if spectral_amplitude.shape != temporal_amplitude.shape:
        raise PulseDefinitionInputError("Temporal and spectral phase-retrieval grids must have equal length")
    spectrum = spectral_amplitude * np.exp(1j * np.asarray(initial_phase, dtype=float))
    maximum_iterations = max(1, int(maximum_iterations))
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise PulseDefinitionInputError(
            "Phase-retrieval tolerance must be positive and finite"
        )
    projection_error = float("inf")
    completed = 0
    for completed in range(1, maximum_iterations + 1):
        temporal = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(spectrum), norm="ortho"))
        temporal = temporal_amplitude * np.exp(1j * np.angle(temporal))
        candidate = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(temporal), norm="ortho"))
        projection_error = _relative_amplitude_error(np.abs(candidate), spectral_amplitude)
        spectrum = spectral_amplitude * np.exp(1j * np.angle(candidate))
        if projection_error <= tolerance:
            break
    final_temporal = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(spectrum), norm="ortho"))
    return np.unwrap(np.angle(spectrum)), {
        "iterations": completed,
        "converged": projection_error <= tolerance,
        "relative_amplitude_error": projection_error,
        "temporal_relative_amplitude_error": _relative_amplitude_error(
            np.abs(final_temporal), temporal_amplitude
        ),
        "tolerance": tolerance,
    }


def spectral_file_pulse_energy_J(spectrum_cfg: dict[str, Any]) -> float | None:
    """Return total pulse energy encoded by an absolute spectrum file."""

    if str(spectrum_cfg.get("type", "monochromatic")) != "file":
        return None
    mode = str(spectrum_cfg.get("value_mode", "relative"))
    if mode == "relative":
        return None
    if mode == "energy_density_J_per_nm" and spectrum_cfg.get("spectral_density_domain", "wavelength") != "wavelength":
        raise PulseDefinitionInputError(
            "energy_density_J_per_nm requires spectral_density_domain=wavelength; "
            "a per-nm density cannot be interpreted as a per-Hz density"
        )
    rows = _rows(spectrum_cfg, 2)
    values = _validated_nonnegative_samples(
        rows[:, 1], spectrum_cfg, label="Spectral pulse file"
    )
    wavelength_nm = _wavelength_nm(rows[:, 0], spectrum_cfg)
    if np.any(np.diff(np.sort(wavelength_nm)) <= 0.0):
        raise PulseDefinitionInputError("Seed spectrum file coordinates must be unique")
    if mode == "energy_per_bin_J":
        return float(np.sum(values))
    if mode == "energy_density_J_per_nm":
        order = np.argsort(wavelength_nm)
        # ``numpy.trapezoid`` was added after the project's supported NumPy
        # 1.24 floor.  Keep the same integral while remaining compatible with
        # both the 1.x and 2.x API families.
        trapezoid = getattr(np, "trapezoid", np.trapz)
        return float(trapezoid(values[order], wavelength_nm[order]))
    raise PulseDefinitionInputError(f"Unsupported seed spectrum value_mode {mode!r}")


def resolve_seed_pulse_energy_J(seed_cfg: dict[str, Any]) -> float:
    """Use absolute spectral-file energy consistently across UI and solvers."""
    if seed_cfg.get("pulse_definition", {}).get("domain", "independent") != "time":
        encoded = spectral_file_pulse_energy_J(seed_cfg.get("spectrum", {}))
        if encoded is not None:
            if not np.isfinite(encoded) or encoded <= 0.0:
                raise PulseDefinitionInputError("Absolute Seed spectrum must contain positive finite energy")
            return encoded
    return float(seed_cfg["pulse_energy_J"])


def rebin_mass(coordinate: np.ndarray, weights: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a normalized discrete mass profile without changing its total."""

    coordinate = np.asarray(coordinate, dtype=float)
    weights = np.asarray(weights, dtype=float)
    count = max(1, int(count))
    order = np.argsort(coordinate)
    coordinate = coordinate[order]
    weights = np.maximum(weights[order], 0.0)
    weights /= max(float(np.sum(weights)), np.finfo(float).tiny)
    if count >= len(coordinate):
        return coordinate, weights
    edges = np.linspace(float(coordinate[0]), float(coordinate[-1]), count + 1)
    indices = np.clip(np.searchsorted(edges, coordinate, side="right") - 1, 0, count - 1)
    rebinned_weights = np.bincount(indices, weights=weights, minlength=count).astype(float)
    weighted_coordinate = np.bincount(indices, weights=weights * coordinate, minlength=count).astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    occupied = rebinned_weights > 0.0
    centers[occupied] = weighted_coordinate[occupied] / rebinned_weights[occupied]
    rebinned_weights /= float(np.sum(rebinned_weights))
    return centers, rebinned_weights


def trim_negligible_mass_support(
    coordinate: np.ndarray,
    weights: np.ndarray,
    *,
    discarded_mass_fraction: float = 1e-12,
    padding_samples: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Remove FFT zero-padding tails before a coarse physical rebin.

    Fourier pulse construction intentionally uses a wide frequency window so
    the conjugate time field is sampled correctly.  Those almost-zero bins are
    numerical padding, not physical spectral samples.  Converting the lowest
    padded frequencies to wavelength can stretch them to many micrometres and
    make a small solver grid waste nearly every bin outside the real spectrum.

    The returned weights are renormalized; ``removed_mass`` reports the actual
    discarded fraction and is bounded approximately by the requested value.
    """

    coordinate = np.asarray(coordinate, dtype=float)
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if coordinate.ndim != 1 or weights.ndim != 1 or len(coordinate) != len(weights):
        raise PulseDefinitionInputError("coordinate and weights must be equal-length one-dimensional arrays")
    if len(coordinate) == 0:
        raise PulseDefinitionInputError("cannot trim an empty mass profile")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise PulseDefinitionInputError("mass profile has zero or invalid total weight")
    order = np.argsort(coordinate)
    coordinate = coordinate[order]
    normalized = weights[order] / total
    tail = min(max(float(discarded_mass_fraction), 0.0), 0.25) / 2.0
    cumulative = np.cumsum(normalized)
    first = int(np.searchsorted(cumulative, tail, side="left"))
    last = int(np.searchsorted(cumulative, 1.0 - tail, side="left"))
    padding = max(0, int(padding_samples))
    first = max(0, first - padding)
    last = min(len(coordinate) - 1, last + padding)
    kept_coordinate = coordinate[first : last + 1]
    kept_weights = normalized[first : last + 1]
    kept_mass = float(np.sum(kept_weights))
    if kept_mass <= 0.0:
        peak = int(np.argmax(normalized))
        return coordinate[peak : peak + 1], np.ones(1), 1.0 - float(normalized[peak])
    return kept_coordinate, kept_weights / kept_mass, max(0.0, 1.0 - kept_mass)


def intensity_fwhm(coordinate: np.ndarray, intensity: np.ndarray) -> float:
    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    coordinate = np.asarray(coordinate, dtype=float)
    if values.size < 2 or float(np.max(values)) <= 0.0:
        return 0.0
    half = 0.5 * float(np.max(values))
    above = np.flatnonzero(values >= half)
    if above.size == 0:
        return 0.0
    first = int(above[0])
    last = int(above[-1])

    def crossing(left: int, right: int) -> float:
        x0, x1 = float(coordinate[left]), float(coordinate[right])
        y0, y1 = float(values[left]), float(values[right])
        if y1 == y0:
            return 0.5 * (x0 + x1)
        return x0 + (half - y0) / (y1 - y0) * (x1 - x0)

    lower = crossing(first - 1, first) if first > 0 else float(coordinate[first])
    upper = crossing(last, last + 1) if last + 1 < len(values) else float(coordinate[last])
    return float(abs(upper - lower))


def _temporal_input(
    seed_cfg: dict[str, Any], count: int, center_frequency_Hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = seed_cfg.get("temporal", {})
    kind = str(cfg.get("type", "gaussian"))
    duration = float(seed_cfg.get("pulse_duration_s", 1e-12))
    if kind == "file":
        rows = _rows(cfg, 2)
        scale = time_scale_to_s(cfg.get("time_unit", "s"))
        sample_time = rows[:, 0] * scale
        order = np.argsort(sample_time)
        sample_time = sample_time[order]
        if np.any(np.diff(sample_time) <= 0.0):
            raise PulseDefinitionInputError("Temporal pulse file coordinates must be unique")
        values = _validated_nonnegative_samples(
            rows[:, 1][order], cfg, label="Temporal pulse file"
        )
        midpoint = 0.5 * float(sample_time[0] + sample_time[-1])
        half_span = max(
            0.5 * float(sample_time[-1] - sample_time[0]),
            count / (3.6 * center_frequency_Hz),
        )
        time_s = np.linspace(midpoint - half_span, midpoint + half_span, count, endpoint=False)
        intensity = np.interp(
            time_s, sample_time, values, left=0.0, right=0.0
        )
        phase = (
            np.interp(time_s, sample_time, np.unwrap(rows[:, 2][order]))
            if rows.shape[1] >= 3
            else np.zeros(count)
        )
        return time_s, intensity, phase

    span = max(3.0, float(cfg.get("span_fwhm", 3.0)))
    half_span = max(span * duration, count / (3.6 * center_frequency_Hz))
    time_s = np.linspace(-half_span, half_span, count, endpoint=False)
    offset = time_s / max(duration, np.finfo(float).tiny)
    if kind == "gaussian":
        intensity = np.exp(-4.0 * np.log(2.0) * offset**2)
    elif kind == "sech2":
        factor = 2.0 * np.arccosh(np.sqrt(2.0))
        intensity = 1.0 / np.cosh(np.clip(factor * offset, -350.0, 350.0)) ** 2
    elif kind == "rectangular":
        intensity = (np.abs(offset) <= 0.5).astype(float)
    elif kind == "super_gaussian":
        order = max(1.0, float(cfg.get("order", 2.0)))
        intensity = super_gaussian_fwhm_density(offset, order)
    else:
        raise PulseDefinitionInputError(f"Unsupported seed temporal type {kind!r}")
    return time_s, intensity, np.zeros(count)


def _spectral_input(seed_cfg: dict[str, Any], count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = seed_cfg.get("spectrum", {})
    kind = str(cfg.get("type", "gaussian"))
    center_wavelength = float(seed_cfg["wavelength_m"])
    center_frequency = LIGHT_SPEED / center_wavelength
    if kind == "file":
        rows = _rows(cfg, 2)
        wavelength_m = _wavelength_nm(rows[:, 0], cfg) * 1e-9
        sample_frequency = LIGHT_SPEED / wavelength_m
        order = np.argsort(sample_frequency)
        sample_frequency = sample_frequency[order]
        if np.any(np.diff(sample_frequency) <= 0.0):
            raise PulseDefinitionInputError("Seed spectrum file coordinates must be unique")
        padding_factor = float(
            seed_cfg.get("coherent", {}).get("frequency_padding_factor", 1.0)
        )
        sample_span = float(sample_frequency[-1] - sample_frequency[0])
        guard = 0.5 * (padding_factor - 1.0) * sample_span
        frequency_minimum = float(sample_frequency[0]) - guard
        frequency_maximum = float(sample_frequency[-1]) + guard
        if frequency_minimum <= 0.0:
            raise PulseDefinitionInputError(
                "seed.coherent.frequency_padding_factor makes the absolute "
                "frequency grid non-positive"
            )
        frequency = np.linspace(
            frequency_minimum,
            frequency_maximum,
            count,
            endpoint=False,
        )
        values = _validated_nonnegative_samples(
            rows[:, 1][order], cfg, label="Spectral pulse file"
        )
        mode = str(cfg.get("value_mode", "relative"))
        if mode == "energy_per_bin_J":
            widths = _bin_widths(sample_frequency)
            density_frequency = values / np.maximum(widths, np.finfo(float).tiny)
        elif str(cfg.get("spectral_density_domain", "wavelength")) == "wavelength":
            density_lambda = values
            wavelength_ordered = wavelength_m[order]
            density_frequency = density_lambda * wavelength_ordered**2 / LIGHT_SPEED
        else:
            density_frequency = values
        intensity = np.interp(frequency, sample_frequency, density_frequency, left=0.0, right=0.0)
        phase = (
            np.interp(
                frequency,
                sample_frequency,
                np.unwrap(rows[:, 2][order]),
            )
            if rows.shape[1] >= 3
            else np.zeros(count)
        )
        return frequency, intensity, phase

    bandwidth_wavelength = float(cfg.get("bandwidth_fwhm_m", seed_cfg.get("spectral_bandwidth_m", 0.0)))
    bandwidth_Hz_raw = cfg.get("bandwidth_fwhm_Hz")
    configured_bandwidth_frequency = (
        0.0 if bandwidth_Hz_raw is None else float(bandwidth_Hz_raw)
    )
    density_domain = str(cfg.get("spectral_density_domain", "wavelength"))
    if configured_bandwidth_frequency > 0.0 and bandwidth_wavelength > 0.0:
        raise PulseDefinitionInputError(
            "Specify only one of bandwidth_fwhm_m and bandwidth_fwhm_Hz"
        )
    if configured_bandwidth_frequency > 0.0 and density_domain != "frequency":
        raise PulseDefinitionInputError(
            "bandwidth_fwhm_Hz requires spectral_density_domain=frequency"
        )
    physical_span = max(1.0, float(cfg.get("span_fwhm", 3.0)))
    if configured_bandwidth_frequency > 0.0:
        bandwidth_frequency = configured_bandwidth_frequency
    elif bandwidth_wavelength <= 0.0:
        duration = float(seed_cfg.get("pulse_duration_s", 1e-12))
        bandwidth_frequency = 0.441 / max(duration, np.finfo(float).tiny)
    else:
        bandwidth_frequency = exact_frequency_fwhm_from_wavelength(
            center_wavelength, bandwidth_wavelength
        )
    # Zero-pad around the physical spectrum so the conjugate temporal pulse is
    # sampled finely enough, while keeping the absolute frequency positive.
    requested_span = max(16.0, physical_span)
    if density_domain == "wavelength" and bandwidth_wavelength > 0.0:
        span = max(
            physical_span,
            min(
                requested_span,
                0.9 * center_wavelength / bandwidth_wavelength,
            ),
        )
    elif density_domain == "frequency" or bandwidth_wavelength <= 0.0:
        span = max(
            physical_span,
            min(
                requested_span,
                0.9 * center_frequency / bandwidth_frequency,
            ),
        )
    else:
        raise PulseDefinitionInputError("spectral_density_domain must be wavelength or frequency")
    if span < 1.0:
        raise PulseDefinitionInputError("Seed bandwidth is too broad for the positive-frequency envelope model")
    if density_domain == "wavelength" and bandwidth_wavelength > 0.0:
        lower_wavelength = center_wavelength - span * bandwidth_wavelength
        upper_wavelength = center_wavelength + span * bandwidth_wavelength
        frequency = np.linspace(
            LIGHT_SPEED / upper_wavelength,
            LIGHT_SPEED / lower_wavelength,
            count,
            endpoint=False,
        )
    else:
        frequency = np.linspace(
            center_frequency - span * bandwidth_frequency,
            center_frequency + span * bandwidth_frequency,
            count,
            endpoint=False,
        )
    if np.any(frequency <= 0.0):
        raise PulseDefinitionInputError("Seed spectrum crosses zero optical frequency; reduce bandwidth/span")
    if density_domain == "wavelength" and bandwidth_wavelength > 0.0:
        wavelength = LIGHT_SPEED / frequency
        offset = (wavelength - center_wavelength) / bandwidth_wavelength
        jacobian = wavelength**2 / LIGHT_SPEED
    else:
        offset = (frequency - center_frequency) / max(
            bandwidth_frequency, np.finfo(float).tiny
        )
        jacobian = 1.0
    support = np.abs(offset) <= physical_span
    if kind in {"monochromatic", "gaussian"}:
        intensity = (
            np.exp(-4.0 * np.log(2.0) * offset**2)
            * jacobian
            * support
        )
    elif kind == "lorentzian":
        intensity = jacobian * support / (1.0 + 4.0 * offset**2)
    elif kind == "super_gaussian":
        order = max(1.0, float(cfg.get("order", 2.0)))
        intensity = (
            super_gaussian_fwhm_density(offset, order)
            * jacobian
            * support
        )
    else:
        raise PulseDefinitionInputError(f"Unsupported seed spectrum type {kind!r}")
    return frequency, intensity, np.zeros(count)


def _normalized_field(intensity: np.ndarray, phase: np.ndarray) -> np.ndarray:
    normalized = _normalized_intensity(intensity)
    return np.sqrt(normalized) * np.exp(1j * np.asarray(phase, dtype=float))


def _normalized_intensity(intensity: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    total = float(np.sum(clipped))
    if not np.isfinite(total) or total <= 0.0:
        raise PulseDefinitionInputError("Seed pulse input has zero or invalid integrated intensity")
    return clipped / total


def _relative_amplitude_error(actual: np.ndarray, target: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    target = np.asarray(target, dtype=float)
    return float(
        np.linalg.norm(actual - target)
        / max(float(np.linalg.norm(target)), np.finfo(float).tiny)
    )


def _temporal_profile_on_grid(seed_cfg: dict[str, Any], time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cfg = seed_cfg.get("temporal", {})
    kind = str(cfg.get("type", "gaussian"))
    duration = float(seed_cfg.get("pulse_duration_s", 1e-12))
    if kind == "file":
        rows = _rows(cfg, 2)
        scale = time_scale_to_s(cfg.get("time_unit", "s"))
        sample_time = rows[:, 0] * scale
        order = np.argsort(sample_time)
        sample_time = sample_time[order]
        if np.any(np.diff(sample_time) <= 0.0):
            raise PulseDefinitionInputError("Temporal pulse file coordinates must be unique")
        values = _validated_nonnegative_samples(
            rows[:, 1][order], cfg, label="Temporal pulse file"
        )
        intensity = np.interp(
            time_s,
            sample_time,
            values,
            left=0.0,
            right=0.0,
        )
        phase = (
            np.interp(
                time_s,
                sample_time,
                np.unwrap(rows[:, 2][order]),
            )
            if rows.shape[1] >= 3
            else np.zeros_like(time_s)
        )
        return intensity, phase
    offset = np.asarray(time_s, dtype=float) / max(duration, np.finfo(float).tiny)
    if kind == "gaussian":
        intensity = np.exp(-4.0 * np.log(2.0) * offset**2)
    elif kind == "sech2":
        factor = 2.0 * np.arccosh(np.sqrt(2.0))
        intensity = 1.0 / np.cosh(np.clip(factor * offset, -350.0, 350.0)) ** 2
    elif kind == "rectangular":
        intensity = (np.abs(offset) <= 0.5).astype(float)
    elif kind == "super_gaussian":
        order = max(1.0, float(cfg.get("order", 2.0)))
        intensity = super_gaussian_fwhm_density(offset, order)
    else:
        raise PulseDefinitionInputError(f"Unsupported seed temporal type {kind!r}")
    return intensity, np.zeros_like(time_s)


def spectral_phase_on_grid(seed_cfg: dict[str, Any], frequency_Hz: np.ndarray) -> np.ndarray | None:
    """Interpolate an optional standalone wavelength/phase file onto a frequency grid."""

    cfg = seed_cfg.get("coherent", {}).get("spectral_phase", {})
    if not bool(cfg.get("enabled", False)):
        return None
    rows = _rows(cfg, 2)
    wavelength_m = _wavelength_nm(rows[:, 0], cfg) * 1e-9
    sample_frequency = LIGHT_SPEED / wavelength_m
    order = np.argsort(sample_frequency)
    sample_frequency = sample_frequency[order]
    if np.any(np.diff(sample_frequency) <= 0.0):
        raise PulseDefinitionInputError("Spectral phase file coordinates must be unique")
    unwrapped = np.unwrap(rows[:, 1][order])
    return np.interp(np.asarray(frequency_Hz, dtype=float), sample_frequency, unwrapped)


def _rows(cfg: dict[str, Any], minimum_columns: int) -> np.ndarray:
    rows = np.asarray(cfg.get("samples", []), dtype=float)
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] < minimum_columns:
        raise PulseDefinitionInputError(f"Tabulated pulse input requires at least two rows and {minimum_columns} columns")
    if not np.all(np.isfinite(rows)):
        raise PulseDefinitionInputError(
            "Tabulated pulse input contains a non-finite value in a supplied column"
        )
    return rows


def _validated_nonnegative_samples(
    values: np.ndarray, cfg: dict[str, Any], *, label: str
) -> np.ndarray:
    samples = np.asarray(values, dtype=float)
    negative = np.flatnonzero(samples < 0.0)
    if not negative.size:
        return samples
    if str(cfg.get("invalid_sample_policy", "error")) == "clip":
        return np.maximum(samples, 0.0)
    rows = ", ".join(str(int(index) + 1) for index in negative[:8])
    raise PulseDefinitionInputError(f"{label} contains negative values at row(s) {rows}")


def _has_phase_column(cfg: dict[str, Any]) -> bool:
    samples = np.asarray(cfg.get("samples", []))
    return bool(str(cfg.get("type", "")) == "file" and samples.ndim == 2 and samples.shape[1] >= 3)


def _wavelength_nm(values: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    return (
        np.asarray(values, dtype=float)
        * wavelength_scale_to_m(cfg.get("wavelength_unit", "m"))
        * 1e9
    )


def _bin_widths(coordinate: np.ndarray) -> np.ndarray:
    coordinate = np.asarray(coordinate, dtype=float)
    if len(coordinate) < 2:
        return np.ones_like(coordinate)
    edges = np.empty(len(coordinate) + 1)
    edges[1:-1] = 0.5 * (coordinate[:-1] + coordinate[1:])
    edges[0] = coordinate[0] - 0.5 * (coordinate[1] - coordinate[0])
    edges[-1] = coordinate[-1] + 0.5 * (coordinate[-1] - coordinate[-2])
    return np.abs(np.diff(edges))
