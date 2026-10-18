"""Second-moment beam-quality diagnostics for sampled complex fields.

The implementation follows the paraxial ISO-style phase-space definition of
beam quality.  It uses the complex transverse field, rather than inferring an
output M2 from the configured input value.  Leading dimensions are treated as
mutually incoherent channels, which is appropriate for the time/spectral bins
stored by the amplifier propagators.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .grid import Grid3D


def _integrate_channels(grid: Grid3D, values: np.ndarray) -> float:
    return float(grid.integrate_transverse(np.sum(values, axis=0)))


def sampled_beam_quality(
    field_value: np.ndarray,
    grid: Grid3D,
    *,
    derivative_method: str = "auto",
    channel_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Return coherent second-moment M2 for x and y.

    ``M2 = 2*sqrt(<x^2><kx^2>-<x kx>^2)`` for each transverse axis.  The
    spatial-frequency operator is evaluated by differentiating the complex
    field on the actual x/y coordinates, so the diagnostic also works on a
    nonuniform physical grid.  A finite aperture is part of the measured field
    and therefore legitimately changes the reported value.
    """

    field = np.asarray(field_value, dtype=complex)
    if field.shape[-2:] != (len(grid.y), len(grid.x)):
        raise ValueError("beam-quality field must end in the grid y/x dimensions")
    channels = field.reshape((-1, len(grid.y), len(grid.x)))
    chunk_size = (
        len(channels)
        if channel_chunk_size is None
        else max(1, min(int(channel_chunk_size), len(channels)))
    )
    if chunk_size < len(channels):
        intensity = np.zeros((len(grid.y), len(grid.x)), dtype=float)
        for start in range(0, len(channels), chunk_size):
            intensity += np.sum(
                np.abs(channels[start : start + chunk_size]) ** 2,
                axis=0,
            )
    else:
        intensity = np.sum(np.abs(channels) ** 2, axis=0)
    total = float(grid.integrate_transverse(intensity))
    if not np.isfinite(total) or total <= np.finfo(float).tiny:
        return {
            "valid": False,
            "method": "complex_field_second_moment",
            "reason": "zero or non-finite sampled field energy",
        }

    xx, yy = grid.transverse_mesh
    center_x = float(grid.integrate_transverse(intensity * xx) / total)
    center_y = float(grid.integrate_transverse(intensity * yy) / total)
    sigma_x2 = float(
        grid.integrate_transverse(intensity * (xx - center_x) ** 2) / total
    )
    sigma_y2 = float(
        grid.integrate_transverse(intensity * (yy - center_y) ** 2) / total
    )

    requested_derivative = str(derivative_method)
    if requested_derivative not in {"auto", "spectral_periodic", "finite_difference"}:
        raise ValueError(
            "derivative_method must be auto, spectral_periodic, or finite_difference"
        )
    use_spectral_derivative = bool(
        grid.is_uniform_xy
        and requested_derivative in {"auto", "spectral_periodic"}
    )
    if requested_derivative == "spectral_periodic" and not grid.is_uniform_xy:
        raise ValueError(
            "spectral_periodic beam-quality derivatives require a uniform x/y grid"
        )
    if chunk_size < len(channels):
        return _sampled_beam_quality_chunked(
            channels,
            grid,
            chunk_size=chunk_size,
            use_spectral_derivative=use_spectral_derivative,
            total=total,
            center_x=center_x,
            center_y=center_y,
            sigma_x2=sigma_x2,
            sigma_y2=sigma_y2,
        )
    if use_spectral_derivative:
        kx = 2.0 * np.pi * np.fft.fftfreq(len(grid.x), d=float(grid.dx))
        ky = 2.0 * np.pi * np.fft.fftfreq(len(grid.y), d=float(grid.dy))
        derivative_x = np.fft.ifft(
            np.fft.fft(channels, axis=-1) * (1j * kx)[None, None, :],
            axis=-1,
        )
        derivative_y = np.fft.ifft(
            np.fft.fft(channels, axis=-2) * (1j * ky)[None, :, None],
            axis=-2,
        )
        derivative_method = "spectral_periodic"
    else:
        edge_order_x = 2 if len(grid.x) >= 3 else 1
        edge_order_y = 2 if len(grid.y) >= 3 else 1
        derivative_x = np.gradient(
            channels, grid.x, axis=-1, edge_order=edge_order_x
        )
        derivative_y = np.gradient(
            channels, grid.y, axis=-2, edge_order=edge_order_y
        )
        derivative_method = "nonuniform_finite_difference"
    momentum_density_x = np.imag(np.conj(channels) * derivative_x)
    momentum_density_y = np.imag(np.conj(channels) * derivative_y)
    mean_kx = _integrate_channels(grid, momentum_density_x) / total
    mean_ky = _integrate_channels(grid, momentum_density_y) / total
    covariance_xk = float(
        grid.integrate_transverse(
            np.sum(momentum_density_x, axis=0) * (xx - center_x)
        )
        / total
    )
    covariance_yk = float(
        grid.integrate_transverse(
            np.sum(momentum_density_y, axis=0) * (yy - center_y)
        )
        / total
    )
    # Evaluate the covariance determinant in a conditioned form.  Directly
    # forming ``sigma_x2*variance_kx-covariance_xk**2`` loses many digits for
    # a strongly curved but otherwise diffraction-limited Gaussian, exactly
    # the situation after a short-focus relay lens.  Projecting the best-fit
    # tilt and quadratic phase from the momentum field is algebraically
    # equivalent and leaves only the uncorrelated angular spread.
    chirp_x = covariance_xk / max(sigma_x2, np.finfo(float).tiny)
    chirp_y = covariance_yk / max(sigma_y2, np.finfo(float).tiny)
    residual_derivative_x = derivative_x - 1j * (
        mean_kx + chirp_x * (xx - center_x)
    )[None, ...] * channels
    residual_derivative_y = derivative_y - 1j * (
        mean_ky + chirp_y * (yy - center_y)
    )[None, ...] * channels
    residual_variance_kx = max(
        _integrate_channels(grid, np.abs(residual_derivative_x) ** 2) / total,
        0.0,
    )
    residual_variance_ky = max(
        _integrate_channels(grid, np.abs(residual_derivative_y) ** 2) / total,
        0.0,
    )
    variance_kx = residual_variance_kx + chirp_x**2 * sigma_x2
    variance_ky = residual_variance_ky + chirp_y**2 * sigma_y2
    determinant_x = max(sigma_x2 * residual_variance_kx, 0.0)
    determinant_y = max(sigma_y2 * residual_variance_ky, 0.0)
    m2_x = 2.0 * float(np.sqrt(determinant_x))
    m2_y = 2.0 * float(np.sqrt(determinant_y))
    physically_resolved = m2_x >= 0.95 and m2_y >= 0.95
    return {
        "valid": physically_resolved,
        "method": "complex_field_second_moment",
        "derivative_method": derivative_method,
        "m2_x": m2_x,
        "m2_y": m2_y,
        "centroid_x_m": center_x,
        "centroid_y_m": center_y,
        "sigma_x_m": float(np.sqrt(max(sigma_x2, 0.0))),
        "sigma_y_m": float(np.sqrt(max(sigma_y2, 0.0))),
        "mean_kx_rad_m": float(mean_kx),
        "mean_ky_rad_m": float(mean_ky),
        "sigma_kx_rad_m": float(np.sqrt(variance_kx)),
        "sigma_ky_rad_m": float(np.sqrt(variance_ky)),
        "residual_sigma_kx_rad_m": float(np.sqrt(residual_variance_kx)),
        "residual_sigma_ky_rad_m": float(np.sqrt(residual_variance_ky)),
        "covariance_x_kx": covariance_xk,
        "covariance_y_ky": covariance_yk,
        "conditioned_quadratic_phase_x_rad_m2": chirp_x,
        "conditioned_quadratic_phase_y_rad_m2": chirp_y,
        "channel_count": int(channels.shape[0]),
        "sampling_warning": (
            None
            if physically_resolved
            else "sampled M2 fell below the paraxial lower bound; enlarge/refine the optical window"
        ),
        "scope": (
            "Paraxial second moments of the sampled complex field. It is not "
            "the configured source M2 copied to the output."
        ),
    }


def _sampled_beam_quality_chunked(
    channels: np.ndarray,
    grid: Grid3D,
    *,
    chunk_size: int,
    use_spectral_derivative: bool,
    total: float,
    center_x: float,
    center_y: float,
    sigma_x2: float,
    sigma_y2: float,
) -> dict[str, Any]:
    """Evaluate the same conditioned M2 while bounding temporary memory."""

    xx, yy = grid.transverse_mesh
    if use_spectral_derivative:
        kx = 2.0 * np.pi * np.fft.fftfreq(len(grid.x), d=float(grid.dx))
        ky = 2.0 * np.pi * np.fft.fftfreq(len(grid.y), d=float(grid.dy))

        def derivatives(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.fft.ifft(
                    np.fft.fft(values, axis=-1)
                    * (1j * kx)[None, None, :],
                    axis=-1,
                ),
                np.fft.ifft(
                    np.fft.fft(values, axis=-2)
                    * (1j * ky)[None, :, None],
                    axis=-2,
                ),
            )

        derivative_label = "spectral_periodic"
    else:
        edge_order_x = 2 if len(grid.x) >= 3 else 1
        edge_order_y = 2 if len(grid.y) >= 3 else 1

        def derivatives(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.gradient(
                    values,
                    grid.x,
                    axis=-1,
                    edge_order=edge_order_x,
                ),
                np.gradient(
                    values,
                    grid.y,
                    axis=-2,
                    edge_order=edge_order_y,
                ),
            )

        derivative_label = "nonuniform_finite_difference"

    momentum_x = 0.0
    momentum_y = 0.0
    covariance_xk_numerator = 0.0
    covariance_yk_numerator = 0.0
    for start in range(0, len(channels), chunk_size):
        local = channels[start : start + chunk_size]
        derivative_x, derivative_y = derivatives(local)
        density_x = np.imag(np.conj(local) * derivative_x)
        density_y = np.imag(np.conj(local) * derivative_y)
        integrated_x = np.sum(density_x, axis=0)
        integrated_y = np.sum(density_y, axis=0)
        momentum_x += float(grid.integrate_transverse(integrated_x))
        momentum_y += float(grid.integrate_transverse(integrated_y))
        covariance_xk_numerator += float(
            grid.integrate_transverse(integrated_x * (xx - center_x))
        )
        covariance_yk_numerator += float(
            grid.integrate_transverse(integrated_y * (yy - center_y))
        )
    mean_kx = momentum_x / total
    mean_ky = momentum_y / total
    covariance_xk = covariance_xk_numerator / total
    covariance_yk = covariance_yk_numerator / total
    chirp_x = covariance_xk / max(sigma_x2, np.finfo(float).tiny)
    chirp_y = covariance_yk / max(sigma_y2, np.finfo(float).tiny)

    residual_x_numerator = 0.0
    residual_y_numerator = 0.0
    fitted_x = (mean_kx + chirp_x * (xx - center_x))[None, ...]
    fitted_y = (mean_ky + chirp_y * (yy - center_y))[None, ...]
    for start in range(0, len(channels), chunk_size):
        local = channels[start : start + chunk_size]
        derivative_x, derivative_y = derivatives(local)
        residual_x_numerator += _integrate_channels(
            grid,
            np.abs(derivative_x - 1j * fitted_x * local) ** 2,
        )
        residual_y_numerator += _integrate_channels(
            grid,
            np.abs(derivative_y - 1j * fitted_y * local) ** 2,
        )
    residual_variance_kx = max(residual_x_numerator / total, 0.0)
    residual_variance_ky = max(residual_y_numerator / total, 0.0)
    variance_kx = residual_variance_kx + chirp_x**2 * sigma_x2
    variance_ky = residual_variance_ky + chirp_y**2 * sigma_y2
    m2_x = 2.0 * float(np.sqrt(max(sigma_x2 * residual_variance_kx, 0.0)))
    m2_y = 2.0 * float(np.sqrt(max(sigma_y2 * residual_variance_ky, 0.0)))
    physically_resolved = m2_x >= 0.95 and m2_y >= 0.95
    return {
        "valid": physically_resolved,
        "method": "complex_field_second_moment",
        "derivative_method": derivative_label,
        "channel_evaluation": "exact_memory_bounded_chunks",
        "channel_chunk_size": int(chunk_size),
        "m2_x": m2_x,
        "m2_y": m2_y,
        "centroid_x_m": center_x,
        "centroid_y_m": center_y,
        "sigma_x_m": float(np.sqrt(max(sigma_x2, 0.0))),
        "sigma_y_m": float(np.sqrt(max(sigma_y2, 0.0))),
        "mean_kx_rad_m": float(mean_kx),
        "mean_ky_rad_m": float(mean_ky),
        "sigma_kx_rad_m": float(np.sqrt(variance_kx)),
        "sigma_ky_rad_m": float(np.sqrt(variance_ky)),
        "residual_sigma_kx_rad_m": float(np.sqrt(residual_variance_kx)),
        "residual_sigma_ky_rad_m": float(np.sqrt(residual_variance_ky)),
        "covariance_x_kx": float(covariance_xk),
        "covariance_y_ky": float(covariance_yk),
        "conditioned_quadratic_phase_x_rad_m2": float(chirp_x),
        "conditioned_quadratic_phase_y_rad_m2": float(chirp_y),
        "channel_count": int(channels.shape[0]),
        "sampling_warning": (
            None
            if physically_resolved
            else "sampled M2 fell below the paraxial lower bound; enlarge/refine the optical window"
        ),
        "scope": (
            "Paraxial second moments of the sampled complex field. It is not "
            "the configured source M2 copied to the output."
        ),
    }


def sampled_spatial_frequency_diagnostics(
    field_value: np.ndarray,
    grid: Grid3D,
    *,
    high_frequency_nyquist_fraction: float = 0.8,
    channel_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Return compact transverse high-frequency occupancy diagnostics.

    A tiny amount of power near Nyquist can dominate the angular second moment
    and therefore M2.  Report both ordinary spectral energy and the much more
    sensitive k-squared-moment contribution without retaining an FFT plane.
    """

    fraction = float(high_frequency_nyquist_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("high_frequency_nyquist_fraction must lie in (0, 1)")
    if not grid.is_uniform_xy:
        return {
            "applicable": False,
            "reason": "transverse Nyquist bands require a uniform x/y grid",
            "high_frequency_nyquist_fraction": fraction,
        }

    field = np.asarray(field_value, dtype=complex)
    if field.shape[-2:] != (len(grid.y), len(grid.x)):
        raise ValueError("spatial-frequency field must end in grid y/x dimensions")
    channels = field.reshape((-1, len(grid.y), len(grid.x)))
    chunk_size = (
        len(channels)
        if channel_chunk_size is None
        else max(1, min(int(channel_chunk_size), len(channels)))
    )

    def axis_diagnostics(axis: int, spacing_m: float) -> dict[str, float]:
        resolved_axis = axis % channels.ndim
        marginal = np.zeros(channels.shape[resolved_axis], dtype=float)
        for start in range(0, len(channels), chunk_size):
            transformed = np.fft.fft(
                channels[start : start + chunk_size],
                axis=axis,
            )
            power = np.abs(transformed) ** 2
            reduction_axes = tuple(
                index for index in range(power.ndim) if index != resolved_axis
            )
            marginal += np.sum(power, axis=reduction_axes)
        frequencies = 2.0 * np.pi * np.fft.fftfreq(
            channels.shape[resolved_axis], d=float(spacing_m)
        )
        nyquist = np.pi / float(spacing_m)
        high = np.abs(frequencies) >= fraction * nyquist
        total_energy = max(float(np.sum(marginal)), np.finfo(float).tiny)
        weighted = frequencies**2 * marginal
        total_second_moment = max(
            float(np.sum(weighted)), np.finfo(float).tiny
        )
        return {
            "nyquist_rad_m": float(nyquist),
            "high_frequency_energy_fraction": float(
                np.sum(marginal[high]) / total_energy
            ),
            "high_frequency_k2_moment_fraction": float(
                np.sum(weighted[high]) / total_second_moment
            ),
            "rms_spatial_frequency_nyquist_fraction": float(
                np.sqrt(float(np.sum(weighted)) / total_energy) / nyquist
            ),
        }

    return {
        "applicable": True,
        "method": "marginal_fft_energy_and_k2_moment",
        "channel_evaluation": "exact_memory_bounded_chunks",
        "channel_chunk_size": int(chunk_size),
        "high_frequency_nyquist_fraction": fraction,
        "x": axis_diagnostics(-1, float(grid.dx)),
        "y": axis_diagnostics(-2, float(grid.dy)),
    }
