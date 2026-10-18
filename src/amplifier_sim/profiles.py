"""Temporal and spectral source profiles with analytic and tabulated inputs.

Profiles are represented by discrete *energy/power fractions*, not by an
interpolation object hidden inside a solver.  This keeps normalization and file
formats auditable and lets pump, signal, CLI, and UI share exactly one parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import LIGHT_SPEED
from .spectral import exact_frequency_fwhm_from_wavelength
from .units import time_scale_to_s, wavelength_scale_to_m


@dataclass(frozen=True)
class DiscreteProfile:
    coordinate: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        if self.coordinate.ndim != 1 or self.weights.ndim != 1:
            raise ValueError("Profile coordinates and weights must be one-dimensional")
        if self.coordinate.size != self.weights.size or self.coordinate.size == 0:
            raise ValueError("Profile coordinates and weights must have the same non-zero length")
        if not np.all(np.isfinite(self.coordinate)) or not np.all(np.isfinite(self.weights)):
            raise ValueError("Profile contains non-finite values")
        if np.any(self.weights < 0) or not np.isclose(float(np.sum(self.weights)), 1.0):
            raise ValueError("Profile weights must be non-negative and sum to one")


@dataclass(frozen=True)
class PumpWaveform:
    time_s: np.ndarray
    power_W: np.ndarray
    dt_s: float
    period_s: float
    normalization_period_s: float | None = None

    @property
    def average_power_W(self) -> float:
        return float(np.mean(self.power_W))

    @property
    def period_average_power_W(self) -> float:
        reference_period = (
            self.period_s
            if self.normalization_period_s is None
            else float(self.normalization_period_s)
        )
        return self.average_power_W * self.period_s / reference_period


def _tabulated_rows(cfg: dict[str, Any], minimum_columns: int = 2) -> np.ndarray:
    samples = cfg.get("samples")
    if samples:
        array = np.asarray(samples, dtype=float)
    elif cfg.get("file"):
        path = Path(cfg["file"])
        try:
            array = np.genfromtxt(path, delimiter=",", comments="#")
            if array.ndim != 2 or array.shape[1] < minimum_columns or np.all(~np.isfinite(array)):
                raise ValueError
        except (ValueError, IndexError):
            array = np.genfromtxt(path, comments="#")
    else:
        raise ValueError("Tabulated profile needs 'samples' or 'file'")
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.shape[0] < 2 or array.shape[1] < minimum_columns:
        raise ValueError(f"Tabulated profile needs at least two rows and {minimum_columns} columns")
    if not np.all(np.isfinite(array)):
        raise ValueError(
            "Tabulated profile contains a non-finite value in a supplied column"
        )
    return array


def _bin_weights(coordinate: np.ndarray, density: np.ndarray) -> np.ndarray:
    order = np.argsort(coordinate)
    coordinate[:] = coordinate[order]
    density = np.asarray(density, dtype=float)[order]
    if np.any(~np.isfinite(coordinate)) or np.any(~np.isfinite(density)):
        raise ValueError("Profile coordinates and density must be finite")
    if np.any(density < 0.0):
        raise ValueError("Profile density must be non-negative")
    if coordinate.size == 1:
        return np.ones(1)
    if np.any(np.diff(coordinate) <= 0.0):
        raise ValueError("Profile coordinates must be unique")
    edges = np.empty(coordinate.size + 1)
    edges[1:-1] = 0.5 * (coordinate[:-1] + coordinate[1:])
    edges[0] = coordinate[0] - 0.5 * (coordinate[1] - coordinate[0])
    edges[-1] = coordinate[-1] + 0.5 * (coordinate[-1] - coordinate[-2])
    mass = density * np.diff(edges)
    total = float(np.sum(mass))
    if total <= 0:
        raise ValueError("Profile integral must be positive")
    return mass / total


def _validated_nonnegative_samples(
    values: np.ndarray,
    cfg: dict[str, Any],
    *,
    label: str,
) -> np.ndarray:
    """Apply the explicit file-sample policy without silently changing data."""

    samples = np.asarray(values, dtype=float)
    negative = np.flatnonzero(samples < 0.0)
    if not negative.size:
        return samples
    policy = str(cfg.get("invalid_sample_policy", "error"))
    if policy == "clip":
        return np.maximum(samples, 0.0)
    if policy != "error":
        raise ValueError("invalid_sample_policy must be error or clip")
    rows = ", ".join(str(int(index) + 1) for index in negative[:8])
    suffix = "..." if negative.size > 8 else ""
    raise ValueError(f"{label} contains negative values at data row(s) {rows}{suffix}")


def _profile_density(
    kind: str,
    offset: np.ndarray,
    order: float,
) -> np.ndarray:
    if kind == "gaussian":
        return np.exp(-4.0 * np.log(2.0) * offset**2)
    if kind == "lorentzian":
        return 1.0 / (1.0 + 4.0 * offset**2)
    if kind == "super_gaussian":
        return super_gaussian_fwhm_density(offset, order)
    raise ValueError(f"Unknown spectral profile {kind!r}")


def super_gaussian_fwhm_density(
    normalized_offset: np.ndarray | float,
    order: float,
) -> np.ndarray:
    """Return a unit-peak super-Gaussian whose configured width is FWHM.

    ``normalized_offset`` is the coordinate divided by the requested full
    width.  Therefore the intensity must equal one half at offsets ``+/-0.5``
    for every order, not only for the Gaussian (order-one) special case.
    """

    exponent_order = max(float(order), 1.0)
    offset = np.asarray(normalized_offset, dtype=float)
    with np.errstate(over="ignore"):
        log_density = (
            -np.log(2.0)
            * np.abs(2.0 * offset) ** (2.0 * exponent_order)
        )
    return np.exp(np.maximum(log_density, -745.0))


def spectral_profile(cfg: dict[str, Any], center_wavelength_m: float, fallback_bandwidth_m: float = 0.0) -> DiscreteProfile:
    if not np.isfinite(center_wavelength_m) or center_wavelength_m <= 0.0:
        raise ValueError("Spectrum centre wavelength must be positive and finite")
    kind = str(cfg.get("type", "monochromatic"))
    if kind == "monochromatic":
        return DiscreteProfile(np.array([center_wavelength_m]), np.ones(1))
    if kind == "file":
        rows = _tabulated_rows(cfg)
        scale = wavelength_scale_to_m(cfg.get("wavelength_unit", "m"))
        wavelength = rows[:, 0].astype(float) * scale
        if np.any(wavelength <= 0.0):
            raise ValueError("Spectrum wavelengths must be positive")
        values = _validated_nonnegative_samples(
            rows[:, 1], cfg, label="Spectrum file"
        )
        if str(cfg.get("value_mode", "relative")) == "energy_per_bin_J":
            if np.any(np.diff(np.sort(wavelength)) <= 0.0):
                raise ValueError("Spectrum file wavelengths must be unique")
            weights = values
            total = float(np.sum(weights))
            if total <= 0.0:
                raise ValueError("Spectrum file contains zero total energy")
            order = np.argsort(wavelength)
            wavelength = wavelength[order]
            weights = weights[order] / total
        else:
            domain = str(cfg.get("spectral_density_domain", "wavelength"))
            if domain == "wavelength":
                weights = _bin_weights(wavelength, values)
            elif domain == "frequency":
                frequency = LIGHT_SPEED / wavelength
                frequency_order = np.argsort(frequency)
                frequency = frequency[frequency_order]
                frequency_density = values[frequency_order]
                frequency_weights = _bin_weights(frequency, frequency_density)
                wavelength = LIGHT_SPEED / frequency
                wavelength_order = np.argsort(wavelength)
                wavelength = wavelength[wavelength_order]
                weights = frequency_weights[wavelength_order]
            else:
                raise ValueError(
                    "spectral_density_domain must be wavelength or frequency"
                )
        return DiscreteProfile(wavelength, weights)

    bandwidth = float(cfg.get("bandwidth_fwhm_m", fallback_bandwidth_m))
    bandwidth_Hz_raw = cfg.get("bandwidth_fwhm_Hz")
    bandwidth_Hz = 0.0 if bandwidth_Hz_raw is None else float(bandwidth_Hz_raw)
    if (
        not np.isfinite(bandwidth)
        or bandwidth < 0.0
        or not np.isfinite(bandwidth_Hz)
        or bandwidth_Hz < 0.0
    ):
        raise ValueError("Spectrum bandwidth must be finite and non-negative")
    if bandwidth <= 0.0 and bandwidth_Hz <= 0.0:
        return DiscreteProfile(np.array([center_wavelength_m]), np.ones(1))
    count = max(3, int(cfg.get("sample_count", 21)))
    span = max(1.0, float(cfg.get("span_fwhm", 3.0)))
    domain = str(cfg.get("spectral_density_domain", "wavelength"))
    if bandwidth_Hz > 0.0 and bandwidth > 0.0:
        raise ValueError(
            "Specify only one of bandwidth_fwhm_m and bandwidth_fwhm_Hz"
        )
    if bandwidth_Hz > 0.0 and domain != "frequency":
        raise ValueError(
            "bandwidth_fwhm_Hz requires spectral_density_domain=frequency"
        )
    order_value = float(cfg.get("order", 2.0))
    if domain == "wavelength":
        if bandwidth <= 0.0:
            raise ValueError(
                "A wavelength-density spectrum requires bandwidth_fwhm_m"
            )
        wavelength = np.linspace(
            center_wavelength_m - span * bandwidth,
            center_wavelength_m + span * bandwidth,
            count,
        )
        if np.any(wavelength <= 0.0):
            raise ValueError("Spectrum bandwidth/span crosses zero wavelength")
        density = _profile_density(
            kind, (wavelength - center_wavelength_m) / bandwidth, order_value
        )
        weights = _bin_weights(wavelength, density)
    elif domain == "frequency":
        center_frequency = LIGHT_SPEED / center_wavelength_m
        bandwidth_frequency = (
            bandwidth_Hz
            if bandwidth_Hz > 0.0
            else float(
                exact_frequency_fwhm_from_wavelength(
                    center_wavelength_m,
                    bandwidth,
                )
            )
        )
        if not np.isfinite(bandwidth_frequency) or bandwidth_frequency <= 0.0:
            raise ValueError("Frequency-domain spectrum bandwidth must be positive")
        frequency = np.linspace(
            center_frequency - span * bandwidth_frequency,
            center_frequency + span * bandwidth_frequency,
            count,
        )
        if np.any(frequency <= 0.0):
            raise ValueError("Spectrum bandwidth/span crosses zero frequency")
        density = _profile_density(
            kind, (frequency - center_frequency) / bandwidth_frequency, order_value
        )
        frequency_weights = _bin_weights(frequency, density)
        wavelength = LIGHT_SPEED / frequency
        sort_order = np.argsort(wavelength)
        wavelength = wavelength[sort_order]
        weights = frequency_weights[sort_order]
    else:
        raise ValueError("spectral_density_domain must be wavelength or frequency")
    return DiscreteProfile(wavelength, weights)


def temporal_profile(cfg: dict[str, Any], pulse_fwhm_s: float) -> DiscreteProfile:
    kind = str(cfg.get("type", "gaussian"))
    # CW has no intraperiod envelope to discretize.  It is consumed only by
    # the validated cycle-average Seed closure, where one unit-weight channel
    # represents the stationary photon flux.
    if kind == "cw":
        return DiscreteProfile(np.array([0.0]), np.ones(1))
    if kind == "file":
        rows = _tabulated_rows(cfg)
        scale = time_scale_to_s(cfg.get("time_unit", "s"))
        time = rows[:, 0].astype(float) * scale
        values = _validated_nonnegative_samples(
            rows[:, 1], cfg, label="Temporal profile file"
        )
        weights = _bin_weights(time, values)
        return DiscreteProfile(time, weights)

    count = max(1, int(cfg.get("sample_count", 21)))
    if count == 1:
        return DiscreteProfile(np.array([0.0]), np.ones(1))
    if not np.isfinite(pulse_fwhm_s) or pulse_fwhm_s <= 0.0:
        raise ValueError("Pulse FWHM must be positive and finite")
    span = max(1.0, float(cfg.get("span_fwhm", 3.0)))
    time = np.linspace(-span * pulse_fwhm_s, span * pulse_fwhm_s, count)
    offset = time / pulse_fwhm_s
    if kind == "gaussian":
        density = np.exp(-4.0 * np.log(2.0) * offset**2)
    elif kind == "sech2":
        scale = 2.0 * np.arccosh(np.sqrt(2.0))
        density = 1.0 / np.cosh(scale * offset) ** 2
    elif kind == "rectangular":
        edges = np.empty(count + 1, dtype=float)
        edges[1:-1] = 0.5 * (time[:-1] + time[1:])
        edges[0] = time[0] - 0.5 * (time[1] - time[0])
        edges[-1] = time[-1] + 0.5 * (time[-1] - time[-2])
        overlap = np.maximum(
            np.minimum(edges[1:], 0.5 * pulse_fwhm_s)
            - np.maximum(edges[:-1], -0.5 * pulse_fwhm_s),
            0.0,
        )
        total = float(np.sum(overlap))
        if total <= 0.0:
            raise ValueError("Rectangular temporal profile has zero bin overlap")
        return DiscreteProfile(time, overlap / total)
    elif kind == "super_gaussian":
        order = max(1.0, float(cfg.get("order", 2.0)))
        density = super_gaussian_fwhm_density(offset, order)
    else:
        raise ValueError(f"Unknown temporal profile {kind!r}")
    weights = _bin_weights(time, density)
    return DiscreteProfile(time, weights)


def periodic_pump_waveform(
    cfg: dict[str, Any],
    configured_power_W: float,
    repetition_rate_Hz: float,
    default_steps: int,
) -> PumpWaveform:
    """Sample one period; ``power_W`` is average or peak as selected in cfg."""

    kind = str(cfg.get("type", "cw"))
    if not np.isfinite(configured_power_W) or configured_power_W < 0.0:
        raise ValueError("Configured Pump power must be finite and non-negative")
    periodic_extension = bool(repetition_rate_Hz > 0.0)
    if repetition_rate_Hz <= 0:
        period = float(cfg.get("period_s", 1.0))
    else:
        period = 1.0 / repetition_rate_Hz
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("Pump waveform period must be positive and finite")
    steps = max(1, int(cfg.get("sample_count", default_steps)))
    dt = period / steps
    time = (np.arange(steps) + 0.5) * dt
    if kind == "cw":
        shape = np.ones(steps)
    elif kind == "file":
        rows = _tabulated_rows(cfg)
        scale = time_scale_to_s(cfg.get("time_unit", "s"))
        sample_time = rows[:, 0] * scale
        order = np.argsort(sample_time)
        sample_time = sample_time[order]
        sample_power = _validated_nonnegative_samples(
            rows[:, 1][order], cfg, label="Pump file waveform"
        )
        if np.any(np.diff(sample_time) <= 0.0):
            raise ValueError(
                "Pump file waveform time coordinates must be unique after sorting"
            )
        contract = str(cfg.get("file_period_contract", "zero_outside"))
        if contract not in {"zero_outside", "cover_period"}:
            raise ValueError(
                "pump.temporal.file_period_contract must be zero_outside or cover_period"
            )
        if sample_time[-1] < 0.0 or sample_time[0] > period:
            raise ValueError(
                "Pump file waveform does not overlap the simulated period"
            )
        tolerance = 64.0 * np.finfo(float).eps * max(
            abs(period),
            abs(float(sample_time[0])),
            abs(float(sample_time[-1])),
            np.finfo(float).tiny,
        )
        if contract == "cover_period" and (
            sample_time[0] > tolerance or sample_time[-1] < period - tolerance
        ):
            raise ValueError(
                "Pump file waveform with file_period_contract=cover_period must cover [0, period]"
            )
        shape = np.interp(
            time, sample_time, sample_power, left=0.0, right=0.0
        )
        if str(cfg.get("value_mode", "relative")) == "absolute_W":
            return PumpWaveform(time, shape, dt, period)
    else:
        center = float(cfg.get("delay_s", period / 2.0))
        if not np.isfinite(center):
            raise ValueError("Pump waveform delay must be finite")
        if periodic_extension:
            center %= period
        width = float(cfg.get("pulse_fwhm_s", period / 10.0))
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("Pump waveform FWHM must be positive and finite")
        offset = time - center
        profile_coordinate = (
            (offset + period / 2.0) % period - period / 2.0
            if periodic_extension
            else offset
        )
        if kind == "rectangular":
            if periodic_extension and width >= period:
                shape = np.ones(steps, dtype=float)
            else:
                shape = np.zeros(steps, dtype=float)
                bin_starts = np.arange(steps, dtype=float) * dt
                bin_ends = bin_starts + dt
                shifts = (-period, 0.0, period) if periodic_extension else (0.0,)
                for shift in shifts:
                    interval_start = center - 0.5 * width + shift
                    interval_end = center + 0.5 * width + shift
                    shape += np.maximum(
                        np.minimum(bin_ends, interval_end)
                        - np.maximum(bin_starts, interval_start),
                        0.0,
                    ) / dt
                shape = np.minimum(shape, 1.0)
        elif kind == "gaussian":
            shape = np.exp(
                -4.0 * np.log(2.0) * (profile_coordinate / width) ** 2
            )
        elif kind == "super_gaussian":
            order = max(1.0, float(cfg.get("order", 2.0)))
            shape = super_gaussian_fwhm_density(
                profile_coordinate / width,
                order,
            )
        else:
            raise ValueError(f"Unknown pump temporal profile {kind!r}")
    shape = np.maximum(shape, 0.0)
    mode = str(cfg.get("power_mode", "average"))
    if not np.any(shape > 0.0):
        raise ValueError("Pump temporal waveform is identically zero")
    if mode == "average":
        normalizer = float(np.mean(shape))
    elif mode == "peak":
        # Analytic profiles already have unit continuous peak. In particular,
        # rectangular bins contain duty fractions, so renormalizing their
        # sampled maximum would invent energy for an unresolved short pulse.
        normalizer = float(np.max(shape)) if kind == "file" else 1.0
    else:
        raise ValueError("Pump power_mode must be average or peak")
    return PumpWaveform(time, configured_power_W * shape / normalizer, dt, period)
