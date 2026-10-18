"""Multi-plane caustic diagnostics for sampled Seed fields.

The propagation itself uses the same angular-spectrum operator as the main
optical train.  Beam radii are D4-sigma second-moment radii and the quadratic
fit is the usual ISO-11146-style caustic model.  The routine is diagnostic
only: none of the scanned planes is fed back into gain, heat, or population.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .beam_quality import sampled_beam_quality
from .compute_backend import fft2, ifft2
from .grid import Grid3D
from .optics import (
    angular_spectrum,
    transverse_boundary_energy_fractions,
    transverse_intensity_moments,
)
from .validation import require_exact_integer


def validate_caustic_config(config: dict[str, Any]) -> None:
    """Validate ``seed.caustic_diagnostics`` without changing its values."""

    if not isinstance(config, dict):
        raise ValueError("seed.caustic_diagnostics must be an object")
    for key in ("enabled", "record_planes"):
        if not isinstance(config.get(key, False), bool):
            raise ValueError(f"seed.caustic_diagnostics.{key} must be true or false")
    positions = config.get("positions_m", [])
    if positions:
        if len(positions) < 3:
            raise ValueError(
                "seed.caustic_diagnostics.positions_m needs at least three planes"
            )
        values = np.asarray(positions, dtype=float)
        if np.any(~np.isfinite(values)) or len(np.unique(values)) < 3:
            raise ValueError(
                "seed.caustic_diagnostics.positions_m must contain at least three distinct finite positions"
            )
    else:
        start = float(config.get("start_m", 0.0))
        stop = float(config.get("stop_m", 0.0))
        count = require_exact_integer(
            config.get("sample_count", 21),
            name="seed.caustic_diagnostics.sample_count",
            minimum=3,
        )
        if not math.isfinite(start) or not math.isfinite(stop) or start == stop:
            raise ValueError(
                "seed.caustic_diagnostics.start_m and stop_m must be finite and different"
            )
    refractive_index = float(config.get("refractive_index", 1.0))
    if not math.isfinite(refractive_index) or refractive_index <= 0.0:
        raise ValueError(
            "seed.caustic_diagnostics.refractive_index must be positive and finite"
        )


def caustic_positions(config: dict[str, Any]) -> np.ndarray:
    """Resolve explicit or evenly spaced positions relative to system output."""

    validate_caustic_config(config)
    explicit = config.get("positions_m", [])
    if explicit:
        return np.asarray(explicit, dtype=float)
    return np.linspace(
        float(config.get("start_m", 0.0)),
        float(config.get("stop_m", 0.1)),
        int(config.get("sample_count", 21)),
    )


def fit_second_moment_caustic(
    positions_m: np.ndarray,
    radii_D4sigma_m: np.ndarray,
    wavelength_m: float,
    refractive_index: float = 1.0,
) -> dict[str, Any]:
    """Fit ``w^2(z)=w0^2+theta^2*(z-z0)^2`` and return M2.

    ``w`` is the D4-sigma radius (half of the ISO second-moment diameter).
    The vacuum wavelength is used, so propagation inside a homogeneous medium
    carries the expected factor ``n`` in ``M2 = pi*n*w0*theta/lambda0``.
    """

    z = np.asarray(positions_m, dtype=float)
    radius = np.asarray(radii_D4sigma_m, dtype=float)
    valid = np.isfinite(z) & np.isfinite(radius) & (radius > 0.0)
    z = z[valid]
    radius = radius[valid]
    if len(z) < 3 or len(np.unique(z)) < 3:
        return {
            "valid": False,
            "point_count": int(len(z)),
            "reason": "at least three distinct finite caustic planes are required",
        }
    wavelength = float(wavelength_m)
    index = float(refractive_index)
    if not math.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("caustic wavelength must be positive and finite")
    if not math.isfinite(index) or index <= 0.0:
        raise ValueError("caustic refractive index must be positive and finite")

    center = float(np.mean(z))
    scale = float(np.max(np.abs(z - center)))
    if scale <= np.finfo(float).tiny:
        return {
            "valid": False,
            "point_count": int(len(z)),
            "reason": "caustic positions have zero numerical span",
        }
    coordinate = (z - center) / scale
    design = np.column_stack(
        [np.ones(len(z)), coordinate, coordinate**2]
    )
    coefficients, *_ = np.linalg.lstsq(design, radius**2, rcond=None)
    constant, linear, quadratic = map(float, coefficients)
    if quadratic <= 0.0:
        return {
            "valid": False,
            "point_count": int(len(z)),
            "reason": "fitted squared radius is not a diverging parabola",
            "scaled_coefficients_m2": coefficients.tolist(),
        }
    waist_coordinate = -linear / (2.0 * quadratic)
    waist_position = center + scale * waist_coordinate
    waist_radius_squared = constant - linear**2 / (4.0 * quadratic)
    divergence_squared = quadratic / scale**2
    if waist_radius_squared <= 0.0 or divergence_squared <= 0.0:
        return {
            "valid": False,
            "point_count": int(len(z)),
            "reason": "fitted waist radius or divergence is non-positive",
            "scaled_coefficients_m2": coefficients.tolist(),
        }
    waist_radius = math.sqrt(waist_radius_squared)
    divergence = math.sqrt(divergence_squared)
    m2 = math.pi * index * waist_radius * divergence / wavelength
    fitted_squared = (
        waist_radius_squared + divergence_squared * (z - waist_position) ** 2
    )
    residual_squared = radius**2 - fitted_squared
    rms_squared = float(np.sqrt(np.mean(residual_squared**2)))
    span_squared = float(np.ptp(radius**2))
    rayleigh_range = waist_radius / divergence
    distance = np.abs(z - waist_position)
    near_count = int(np.count_nonzero(distance <= rayleigh_range))
    far_count = int(np.count_nonzero(distance >= 2.0 * rayleigh_range))
    coverage_ok = len(z) >= 10 and near_count >= 5 and far_count >= 5
    return {
        "valid": True,
        "method": "D4sigma_quadratic_caustic_fit",
        "standard_scope": (
            "ISO-11146-style second-moment fit; full compliance also depends on "
            "experimental sampling, background subtraction, detector calibration, and uncertainty"
        ),
        "point_count": int(len(z)),
        "waist_position_m_relative_to_system_output": waist_position,
        "waist_radius_D4sigma_m": waist_radius,
        "far_field_divergence_half_angle_rad": divergence,
        "rayleigh_range_m": rayleigh_range,
        "M2": m2,
        "rms_residual_radius_squared_m2": rms_squared,
        "normalized_rms_residual_radius_squared": rms_squared
        / max(span_squared, np.finfo(float).tiny),
        "near_waist_point_count": near_count,
        "far_field_point_count": far_count,
        "iso_style_sampling_coverage": coverage_ok,
    }


def scan_sampled_caustic(
    field_value: np.ndarray,
    grid: Grid3D,
    positions_m: np.ndarray,
    wavelength_m: float,
    refractive_index: float = 1.0,
    m2_x: float = 1.0,
    m2_y: float = 1.0,
    *,
    retain_fields: bool = False,
) -> tuple[dict[str, Any], list[tuple[float, np.ndarray]]]:
    """Propagate a monochromatic sampled field to independent scan planes."""

    source = np.asarray(field_value, dtype=complex)
    if source.shape[-2:] != (len(grid.y), len(grid.x)):
        raise ValueError("caustic source field must end in the grid y/x dimensions")
    positions = np.asarray(positions_m, dtype=float)
    if positions.ndim != 1 or len(positions) < 3 or np.any(~np.isfinite(positions)):
        raise ValueError("caustic positions must be a finite one-dimensional vector")

    samples: list[dict[str, Any]] = []
    retained: list[tuple[float, np.ndarray]] = []
    source_energy: float | None = None
    maximum_energy_drift = 0.0
    for position in positions:
        propagated = angular_spectrum(
            source,
            grid,
            float(position),
            float(wavelength_m),
            float(refractive_index),
            float(m2_x),
            float(m2_y),
        )
        intensity = np.abs(propagated) ** 2
        if intensity.ndim > 2:
            intensity = np.sum(
                intensity, axis=tuple(range(intensity.ndim - 2))
            )
        moments = transverse_intensity_moments(intensity, grid)
        boundary = transverse_boundary_energy_fractions(intensity, grid)
        energy = float(moments["total"])
        if source_energy is None:
            source_energy = energy
        maximum_energy_drift = max(
            maximum_energy_drift,
            abs(energy - source_energy) / max(abs(source_energy), np.finfo(float).tiny),
        )
        samples.append(
            {
                "position_m_relative_to_system_output": float(position),
                "quantity": energy,
                "centroid_x_m": moments["centroid_x_m"],
                "centroid_y_m": moments["centroid_y_m"],
                "radius_x_D4sigma_m": moments["radius_x_D4sigma_m"],
                "radius_y_D4sigma_m": moments["radius_y_D4sigma_m"],
                "radius_major_D4sigma_m": moments["radius_major_D4sigma_m"],
                "radius_minor_D4sigma_m": moments["radius_minor_D4sigma_m"],
                "major_axis_angle_deg": moments["major_axis_angle_deg"],
                **boundary,
            }
        )
        if retain_fields:
            retained.append((float(position), propagated.copy()))

    radius_x = np.asarray(
        [row["radius_x_D4sigma_m"] for row in samples], dtype=float
    )
    radius_y = np.asarray(
        [row["radius_y_D4sigma_m"] for row in samples], dtype=float
    )
    fit_x = fit_second_moment_caustic(
        positions, radius_x, wavelength_m, refractive_index
    )
    fit_y = fit_second_moment_caustic(
        positions, radius_y, wavelength_m, refractive_index
    )
    source_quality = sampled_beam_quality(source, grid)
    warnings: list[str] = []
    if len(positions) < 10:
        warnings.append(
            "Fewer than ten caustic planes were sampled; this is a numerical fit, not an ISO-complete measurement layout."
        )
    if not bool(fit_x.get("iso_style_sampling_coverage", False)) or not bool(
        fit_y.get("iso_style_sampling_coverage", False)
    ):
        warnings.append(
            "The selected scan does not place at least five planes near the fitted waist and five beyond two Rayleigh ranges on both axes."
        )
    maximum_boundary = max(
        (float(row["boundary_energy_fraction"]) for row in samples), default=0.0
    )
    if maximum_boundary > 1e-2:
        warnings.append(
            "More than 1% of a caustic plane's energy reaches the outer 5% of the physical FFT-window width on each side; enlarge the window before interpreting M2."
        )
    return (
        {
            "enabled": True,
            "application": "diagnostic_only",
            "affects_propagation": False,
            "method": "independent_angular_spectrum_planes_with_D4sigma_fit",
            "reference_plane": "system_output",
            "wavelength_m": float(wavelength_m),
            "refractive_index": float(refractive_index),
            "sample_count": int(len(samples)),
            "samples": samples,
            "fit_x": fit_x,
            "fit_y": fit_y,
            "source_plane_phase_space_M2_x": source_quality.get("m2_x"),
            "source_plane_phase_space_M2_y": source_quality.get("m2_y"),
            "transport_M2_x": float(m2_x),
            "transport_M2_y": float(m2_y),
            "source_plane_effective_M2_x": (
                None
                if source_quality.get("m2_x") is None
                else float(source_quality["m2_x"]) * float(m2_x)
            ),
            "source_plane_effective_M2_y": (
                None
                if source_quality.get("m2_y") is None
                else float(source_quality["m2_y"]) * float(m2_y)
            ),
            "maximum_relative_power_drift": maximum_energy_drift,
            "maximum_boundary_energy_fraction": maximum_boundary,
            "warnings": warnings,
            "scope": (
                "Monochromatic paraxial caustic. Leading field channels (including orthogonal Jones "
                "components) are propagated independently and summed in intensity. Each plane starts "
                "from the system-output complex field so scan order cannot accumulate numerical error."
            ),
        },
        retained,
    )


def scan_spectral_caustic(
    field_value: np.ndarray,
    grid: Grid3D,
    positions_m: np.ndarray,
    wavelengths_m: np.ndarray,
    refractive_index: float = 1.0,
    m2_x: float | np.ndarray = 1.0,
    m2_y: float | np.ndarray = 1.0,
    *,
    retain_fields: bool = False,
) -> tuple[dict[str, Any], list[tuple[float, np.ndarray]]]:
    """Scan an energy-integrated broadband caustic wavelength by wavelength.

    Each spectral channel is propagated with its own wavelength and transport
    M2 before intensities are summed.  The fitted scalar M2 is therefore an
    explicitly labelled broadband effective value referenced to the
    energy-weighted wavelength; it is not claimed as a monochromatic
    ISO-11146 measurement.
    """

    source = np.asarray(field_value, dtype=complex)
    wavelengths = np.asarray(wavelengths_m, dtype=float)
    if source.shape[-2:] != (len(grid.y), len(grid.x)):
        raise ValueError(
            "spectral caustic field must end in the grid y/x dimensions"
        )
    if source.ndim < 3 or source.shape[-3] != len(wavelengths):
        raise ValueError(
            "spectral caustic wavelength axis must be the third axis from the end"
        )
    if (
        wavelengths.ndim != 1
        or len(wavelengths) == 0
        or np.any(~np.isfinite(wavelengths))
        or np.any(wavelengths <= 0.0)
    ):
        raise ValueError("spectral caustic wavelengths must be positive and finite")
    positions = np.asarray(positions_m, dtype=float)
    if positions.ndim != 1 or len(positions) < 3 or np.any(~np.isfinite(positions)):
        raise ValueError(
            "spectral caustic positions must be a finite one-dimensional vector"
        )
    index = float(refractive_index)
    if not math.isfinite(index) or index <= 0.0:
        raise ValueError("spectral caustic refractive index must be positive")
    transport_m2_x = np.broadcast_to(
        np.asarray(m2_x, dtype=float), wavelengths.shape
    )
    transport_m2_y = np.broadcast_to(
        np.asarray(m2_y, dtype=float), wavelengths.shape
    )
    if (
        np.any(~np.isfinite(transport_m2_x))
        or np.any(~np.isfinite(transport_m2_y))
        or np.any(transport_m2_x < 1.0)
        or np.any(transport_m2_y < 1.0)
    ):
        raise ValueError("spectral caustic transport M2 values must be finite and >= 1")

    # Move wavelength to the first axis while preserving any mutually
    # incoherent leading channels (for example a singleton pulse/Jones axis).
    wavelength_first = np.moveaxis(source, -3, 0)
    wavelength_fields = wavelength_first.reshape(
        (len(wavelengths), -1, len(grid.y), len(grid.x))
    )
    source_energy_by_wavelength = np.sum(
        grid.integrate_transverse(np.abs(wavelength_fields) ** 2),
        axis=1,
    ).astype(float)
    total_source_energy = float(np.sum(source_energy_by_wavelength))
    has_source_energy = bool(
        np.isfinite(total_source_energy)
        and total_source_energy > np.finfo(float).tiny
    )
    broadband_effective = bool(
        len(wavelengths) > 1
        and not np.allclose(
            wavelengths,
            wavelengths[0],
            rtol=1e-12,
            atol=0.0,
        )
    )
    reference_wavelength = (
        float(
            np.sum(source_energy_by_wavelength * wavelengths)
            / total_source_energy
        )
        if has_source_energy
        else float(np.mean(wavelengths))
    )

    if not grid.is_uniform_xy:
        raise ValueError("spectral caustic angular-spectrum scan requires a uniform x/y grid")

    # A coherent CPA result can contain several thousand frequency samples.
    # Repeating one forward FFT for every wavelength *and* every caustic plane
    # is needlessly expensive, while retaining every propagated complex field
    # can consume tens of gigabytes.  Transform each spectral chunk once, use
    # the exact wavelength-dependent transfer function at every requested
    # plane, and accumulate only detector-equivalent incoherent intensity.
    # The chunk size is bounded by an approximate 64 MiB scratch budget.
    leading_channel_count = int(wavelength_fields.shape[1])
    transverse_count = len(grid.y) * len(grid.x)
    scratch_bytes_per_wavelength = max(
        transverse_count * 16,
        transverse_count * 16 * (2 * leading_channel_count + 1),
    )
    spectral_chunk_size = max(
        1,
        min(
            len(wavelengths),
            64,
            int((64 * 1024**2) / scratch_bytes_per_wavelength),
        ),
    )
    plane_intensities = [
        np.zeros((len(grid.y), len(grid.x)), dtype=float)
        for _ in positions
    ]
    fx = np.fft.fftfreq(len(grid.x), d=float(grid.dx))
    fy = np.fft.fftfreq(len(grid.y), d=float(grid.dy))
    fxx, fyy = np.meshgrid(fx, fy, indexing="xy")
    fxx_squared = fxx**2
    fyy_squared = fyy**2
    for start in range(0, len(wavelengths), spectral_chunk_size):
        stop = min(start + spectral_chunk_size, len(wavelengths))
        chunk = wavelength_fields[start:stop]
        chunk_spectrum = fft2(chunk)
        spatial_frequency_term = (
            transport_m2_x[start:stop, None, None] * fxx_squared[None, ...]
            + transport_m2_y[start:stop, None, None] * fyy_squared[None, ...]
        )
        wavelength_factor = wavelengths[start:stop, None, None] / index
        for position_index, position in enumerate(positions):
            transfer = np.exp(
                -1j
                * math.pi
                * float(position)
                * wavelength_factor
                * spatial_frequency_term
            )
            propagated = ifft2(chunk_spectrum * transfer[:, None, ...])
            plane_intensities[position_index] += np.sum(
                np.abs(propagated) ** 2,
                axis=(0, 1),
            )

    samples: list[dict[str, Any]] = []
    retained: list[tuple[float, np.ndarray]] = []
    maximum_energy_drift = 0.0
    for position, intensity in zip(positions, plane_intensities, strict=True):
        moments = transverse_intensity_moments(intensity, grid)
        boundary = transverse_boundary_energy_fractions(intensity, grid)
        energy = float(moments["total"])
        maximum_energy_drift = max(
            maximum_energy_drift,
            abs(energy - total_source_energy)
            / max(total_source_energy, np.finfo(float).tiny),
        )
        samples.append(
            {
                "position_m_relative_to_system_output": float(position),
                "quantity": energy,
                "centroid_x_m": moments["centroid_x_m"],
                "centroid_y_m": moments["centroid_y_m"],
                "radius_x_D4sigma_m": moments["radius_x_D4sigma_m"],
                "radius_y_D4sigma_m": moments["radius_y_D4sigma_m"],
                "radius_major_D4sigma_m": moments["radius_major_D4sigma_m"],
                "radius_minor_D4sigma_m": moments["radius_minor_D4sigma_m"],
                "major_axis_angle_deg": moments["major_axis_angle_deg"],
                **boundary,
            }
        )
        if retain_fields:
            # OpticalTrace stores intensity after taking abs(field)**2.  A
            # real non-negative square-root therefore preserves the exact
            # detector plane without retaining an enormous wavelength stack.
            retained.append(
                (float(position), np.sqrt(np.maximum(intensity, 0.0)).astype(complex))
            )

    radius_x = np.asarray(
        [row["radius_x_D4sigma_m"] for row in samples], dtype=float
    )
    radius_y = np.asarray(
        [row["radius_y_D4sigma_m"] for row in samples], dtype=float
    )
    fit_x = fit_second_moment_caustic(
        positions, radius_x, reference_wavelength, index
    )
    fit_y = fit_second_moment_caustic(
        positions, radius_y, reference_wavelength, index
    )
    maximum_boundary = max(
        (float(row["boundary_energy_fraction"]) for row in samples),
        default=0.0,
    )
    warnings: list[str] = []
    if not has_source_energy:
        warnings.append(
            "The output field has zero sampled energy; caustic radii and M2 fits are invalid."
        )
    if broadband_effective:
        warnings.append(
            "Broadband caustic M2 is an energy-integrated effective value referenced "
            "to the energy-weighted wavelength, not a monochromatic ISO-11146 result."
        )
    if len(positions) < 10:
        warnings.append(
            "Fewer than ten caustic planes were sampled; this is a numerical fit, not an ISO-complete measurement layout."
        )
    if not bool(fit_x.get("iso_style_sampling_coverage", False)) or not bool(
        fit_y.get("iso_style_sampling_coverage", False)
    ):
        warnings.append(
            "The selected broadband scan does not place at least five planes near the fitted waist and five beyond two Rayleigh ranges on both axes."
        )
    if maximum_boundary > 1e-2:
        warnings.append(
            "More than 1% of a broadband caustic plane's energy reaches the outer 5% of the physical FFT-window width on each side; enlarge the window before interpreting M2."
        )
    return (
        {
            "enabled": True,
            "application": "diagnostic_only",
            "affects_propagation": False,
            "method": "spectrally_resolved_energy_integrated_D4sigma_fit",
            "reference_plane": "system_output",
            "broadband_effective": broadband_effective,
            "reference_wavelength_m": reference_wavelength,
            "refractive_index": index,
            "sample_count": int(len(samples)),
            "samples": samples,
            "fit_x": fit_x,
            "fit_y": fit_y,
            "wavelengths_m": wavelengths.tolist(),
            "source_energy_fraction_by_wavelength": (
                source_energy_by_wavelength
                / max(total_source_energy, np.finfo(float).tiny)
            ).tolist(),
            "transport_M2_x_by_wavelength": transport_m2_x.tolist(),
            "transport_M2_y_by_wavelength": transport_m2_y.tolist(),
            "spectral_chunk_size": int(spectral_chunk_size),
            "maximum_relative_power_drift": maximum_energy_drift,
            "maximum_boundary_energy_fraction": maximum_boundary,
            "warnings": warnings,
            "scope": (
                "Every wavelength channel is propagated independently and summed "
                "in intensity. The fitted M2 uses an energy-weighted reference "
                "wavelength and is intended as a broadband engineering diagnostic. "
                "Retained display planes contain the summed detector intensity, not "
                "the full per-wavelength complex-field stack."
            ),
        },
        retained,
    )
