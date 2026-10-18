"""Small dependency-free spectral-dispersion utilities used by core optics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .constants import LIGHT_SPEED


def sellmeier_refractive_index(
    wavelength_m: np.ndarray | float,
    sellmeier: dict[str, Any] | None,
) -> np.ndarray:
    """Evaluate ``n(lambda)`` for B/C coefficients using C in micrometre²."""

    wavelength = np.asarray(wavelength_m, dtype=float)
    if not sellmeier or not sellmeier.get("enabled", True):
        raise ValueError("Sellmeier model is not enabled")
    b = np.asarray(sellmeier.get("B", []), dtype=float)
    c = np.asarray(sellmeier.get("C_um2", sellmeier.get("C", [])), dtype=float)
    if b.shape != c.shape or b.ndim != 1 or len(b) < 1 or not np.all(np.isfinite(b)) or not np.all(np.isfinite(c)):
        raise ValueError("Sellmeier B and C arrays must have equal finite length")
    lambda_um2 = (wavelength * 1e6) ** 2
    denominator = lambda_um2[..., None] - c
    if np.any(np.isclose(denominator, 0.0, rtol=0.0, atol=1e-15)):
        raise ValueError("Requested wavelength lies on a Sellmeier pole")
    n_squared = 1.0 + np.sum(b * lambda_um2[..., None] / denominator, axis=-1)
    if np.any(n_squared <= 0.0) or not np.all(np.isfinite(n_squared)):
        raise ValueError("Sellmeier coefficients produced a non-physical refractive index")
    return np.sqrt(n_squared)


def material_refractive_index(
    material: dict[str, Any],
    wavelength_m: float,
    *,
    role: str = "signal",
) -> float:
    """Resolve an imported wavelength model while retaining scalar fallback."""

    if role == "pump" and material.get("pump_refractive_index") not in (None, 0):
        return float(material["pump_refractive_index"])
    sellmeier = material.get("dispersion", {}).get("sellmeier")
    if sellmeier and sellmeier.get("enabled", True):
        return float(sellmeier_refractive_index(float(wavelength_m), sellmeier)) + float(
            material.get("sellmeier_temperature_index_offset", 0.0)
        )
    return float(material["refractive_index"])


def material_group_index(
    material: dict[str, Any],
    wavelength_m: float,
    *,
    role: str = "signal",
) -> float:
    """Return the envelope group index ``n_g = n - lambda dn/dlambda``.

    Imported Sellmeier data are differentiated numerically.  Scalar-index
    materials are non-dispersive by construction, so their group and phase
    indices are identical.  Keeping this helper next to the phase-index
    resolver prevents a transport solver from silently using vacuum ``c``.
    """

    wavelength = float(wavelength_m)
    if wavelength <= 0.0 or not math.isfinite(wavelength):
        raise ValueError("wavelength_m must be positive and finite")
    sellmeier = material.get("dispersion", {}).get("sellmeier")
    if not sellmeier or not sellmeier.get("enabled", True):
        return material_refractive_index(material, wavelength, role=role)
    relative_step = 1e-5
    lower = wavelength * (1.0 - relative_step)
    upper = wavelength * (1.0 + relative_step)
    n_lower = material_refractive_index(material, lower, role=role)
    n_center = material_refractive_index(material, wavelength, role=role)
    n_upper = material_refractive_index(material, upper, role=role)
    derivative = (n_upper - n_lower) / (upper - lower)
    group_index = n_center - wavelength * derivative
    if not math.isfinite(group_index) or group_index <= 0.0:
        raise ValueError("material dispersion produced a non-physical group index")
    return float(group_index)


def phase_from_coefficients(
    frequency_Hz: np.ndarray | float,
    coefficients: dict[str, Any] | None,
    scale: float = 1.0,
) -> np.ndarray:
    """Return spectral phase using SI-valued GD/GDD/TOD/FOD coefficients."""

    frequency = np.asarray(frequency_Hz, dtype=float)
    if not coefficients:
        return np.zeros_like(frequency)
    reference_frequency = coefficients.get("reference_frequency_Hz")
    if reference_frequency is None:
        reference_wavelength = coefficients.get("reference_wavelength_m")
        reference_frequency = (
            LIGHT_SPEED / float(reference_wavelength)
            if reference_wavelength not in (None, 0)
            else float(np.mean(frequency))
        )
    omega_offset = 2.0 * math.pi * (frequency - float(reference_frequency))
    phase = float(coefficients.get("constant_phase_rad", 0.0))
    phase = phase + float(coefficients.get("gd_s", 0.0)) * omega_offset
    phase = phase + 0.5 * float(coefficients.get("gdd_s2", 0.0)) * omega_offset**2
    phase = phase + (1.0 / 6.0) * float(coefficients.get("tod_s3", 0.0)) * omega_offset**3
    phase = phase + (1.0 / 24.0) * float(coefficients.get("fod_s4", 0.0)) * omega_offset**4
    return np.asarray(phase, dtype=float) * float(scale)


def material_dispersion_phase(
    frequency_Hz: np.ndarray,
    dispersion_cfg: dict[str, Any] | None,
    length_m: float,
) -> np.ndarray:
    if not dispersion_cfg:
        return np.zeros_like(frequency_Hz)
    sellmeier_phase = np.zeros_like(frequency_Hz, dtype=float)
    sellmeier = dispersion_cfg.get("sellmeier")
    if sellmeier and sellmeier.get("enabled", True):
        frequency = np.asarray(frequency_Hz, dtype=float)
        omega = 2.0 * math.pi * frequency
        reference_wavelength = dispersion_cfg.get("reference_wavelength_m")
        reference_frequency = (
            LIGHT_SPEED / float(reference_wavelength)
            if reference_wavelength not in (None, 0)
            else float(np.mean(frequency))
        )
        omega_reference = 2.0 * math.pi * reference_frequency

        def beta(angular_frequency: np.ndarray | float) -> np.ndarray:
            local_omega = np.asarray(angular_frequency, dtype=float)
            wavelength = 2.0 * math.pi * LIGHT_SPEED / local_omega
            return sellmeier_refractive_index(wavelength, sellmeier) * local_omega / LIGHT_SPEED

        beta_reference = float(beta(omega_reference))
        step = max(abs(omega_reference) * 1e-6, 1.0)
        beta1 = float((beta(omega_reference + step) - beta(omega_reference - step)) / (2.0 * step))
        relative_beta = beta(omega) - beta_reference - beta1 * (omega - omega_reference)
        sellmeier_phase = np.asarray(relative_beta * float(length_m), dtype=float)
    coefficients = {
        "reference_wavelength_m": dispersion_cfg.get("reference_wavelength_m"),
        "gd_s": dispersion_cfg.get("gd_s_per_m", 0.0),
        "gdd_s2": dispersion_cfg.get("gdd_s2_per_m", 0.0),
        "tod_s3": dispersion_cfg.get("tod_s3_per_m", 0.0),
        "fod_s4": dispersion_cfg.get("fod_s4_per_m", 0.0),
    }
    return sellmeier_phase + phase_from_coefficients(frequency_Hz, coefficients, scale=length_m)
