"""Spatial Seed gain closures shared by pulse and average-power solvers.

``local_field`` is the full sampled-field operator: every transverse pixel sees
its own gain and the optical field can be reshaped by gain guiding.

``gaussian_mode_overlap`` is a reduced-order closure for Gaussian-amplifier
studies with gain guiding deliberately disabled. The transported mode still responds to
diffraction, walk-off, apertures, and thermal phase, but one scalar gain is
applied to each retained mode in every z cell.  The scalar is the
intensity-weighted overlap of the local 3-D gain with that mode.  Consequently
gain cannot reshape the transverse field.  This is an explicit modelling
choice, not a numerical shortcut silently substituted for the local-field
model.

``parabolic_gain_fit`` is the sampled-field companion used by imported Commercial App
Gaussian-amplifier projects.  It fits the local active-gain map inside the
configured transverse ellipse and applies that quadratic map to the complex
field.  Pump, populations, cross-sections, and lifetimes remain unchanged;
only the spatial representation used by the Gaussian/BPM step is
approximated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LOCAL_FIELD = "local_field"
GAUSSIAN_MODE_OVERLAP = "gaussian_mode_overlap"
PARABOLIC_GAIN_FIT = "parabolic_gain_fit"
SPATIAL_GAIN_MODELS = {
    LOCAL_FIELD,
    GAUSSIAN_MODE_OVERLAP,
    PARABOLIC_GAIN_FIT,
}


def spatial_gain_transport_model(seed_cfg: dict[str, Any]) -> str:
    """Return the configured spatial gain closure."""

    return str(
        seed_cfg.get("spatial_gain_transport", {}).get("model", LOCAL_FIELD)
    )


def exponential_cell_average_factor(exponent: np.ndarray) -> np.ndarray:
    """Return ``(exp(x)-1)/x`` with the analytic value one at ``x=0``."""

    values = np.asarray(exponent, dtype=float)
    return np.divide(
        np.expm1(values),
        values,
        out=np.ones_like(values),
        where=np.abs(values) > 1e-12,
    )


def transverse_weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    grid: Any,
) -> np.ndarray:
    """Average maps over their final y/x axes using sampled modal intensity."""

    local_values = np.asarray(values, dtype=float)
    local_weights = np.asarray(weights, dtype=float)
    numerator = np.asarray(
        grid.integrate_transverse(local_values * local_weights), dtype=float
    )
    denominator = np.asarray(grid.integrate_transverse(local_weights), dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > np.finfo(float).tiny,
    )


def parabolic_gain_fit_map(
    values_m_inv: np.ndarray,
    grid: Any,
    crystal_mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return axis-aligned quadratic gain maps fitted in one ellipse.

    Leading dimensions (for example wavelength) are fitted independently.
    The fitted gain is clipped at the configured minimum because the Commercial App
    amplifier fit represents an active-gain duct, not an invented absorber.
    """

    values = np.asarray(values_m_inv, dtype=float)
    if values.shape[-2:] != (len(grid.y), len(grid.x)):
        raise ValueError("parabolic gain values must end in the grid y/x shape")
    mask = np.asarray(crystal_mask, dtype=bool)
    if mask.shape != values.shape[-2:]:
        raise ValueError("parabolic gain crystal mask must match the grid")
    radius_x = float(config["fit_radius_x_m"])
    radius_y = float(config["fit_radius_y_m"])
    center_x = float(config.get("center_x_m", 0.0))
    center_y = float(config.get("center_y_m", 0.0))
    minimum_gain = float(config.get("minimum_gain_m_inv", 0.0))
    xx, yy = grid.transverse_mesh
    x = xx - center_x
    y = yy - center_y
    normalized_x = x / radius_x
    normalized_y = y / radius_y
    selected = (
        mask
        & (normalized_x**2 + normalized_y**2 <= 1.0)
    )
    point_count = int(np.count_nonzero(selected))
    if point_count < 5:
        raise ValueError(
            "parabolic_gain_fit needs at least five active grid cells inside "
            "the configured fit ellipse"
        )
    design = np.column_stack(
        [
            np.ones(point_count),
            normalized_x[selected],
            normalized_y[selected],
            normalized_x[selected] ** 2,
            normalized_y[selected] ** 2,
        ]
    )
    weights = np.sqrt(np.asarray(grid.cell_areas)[selected])
    weighted_design = design * weights[:, None]
    full_design = np.stack(
        [
            np.ones_like(xx),
            normalized_x,
            normalized_y,
            normalized_x**2,
            normalized_y**2,
        ],
        axis=0,
    )
    channels = values.reshape((-1, len(grid.y), len(grid.x)))
    fitted_channels = np.empty_like(channels)
    diagnostics: list[dict[str, Any]] = []
    area_weights = np.asarray(grid.cell_areas)[selected]
    for index, channel in enumerate(channels):
        weighted_values = channel[selected] * weights
        coefficients, _, rank, _ = np.linalg.lstsq(
            weighted_design, weighted_values, rcond=None
        )
        if rank < design.shape[1]:
            raise ValueError(
                "parabolic_gain_fit pupil does not resolve every quadratic coefficient; "
                "enlarge the fit ellipse or refine the transverse grid"
            )
        quadratic = np.sum(
            full_design * coefficients[:, None, None], axis=0
        )
        # The Commercial App fit radius is a local approximation window.  Extrapolating
        # a noisy quadratic across the whole crystal can turn a small positive
        # curvature into arbitrarily large edge gain.  Apply the fitted duct
        # only inside that declared window and retain the sampled physical
        # gain elsewhere.
        fitted = channel.copy()
        fitted[selected] = np.maximum(quadratic[selected], minimum_gain)
        fitted *= mask
        fitted_channels[index] = fitted
        # Elementwise reduction avoids a macOS Accelerate/NumPy matmul warning
        # observed for very small least-squares coefficients.
        fitted_selected = np.sum(
            design * coefficients[None, :], axis=1
        )
        residual = channel[selected] - fitted_selected
        rmse = float(
            np.sqrt(
                np.sum(area_weights * residual**2)
                / np.sum(area_weights)
            )
        )
        span = float(np.ptp(channel[selected]))
        diagnostics.append(
            {
                "channel_index": index,
                "point_count": point_count,
                "fit_radius_x_m": radius_x,
                "fit_radius_y_m": radius_y,
                "center_x_m": center_x,
                "center_y_m": center_y,
                "minimum_gain_m_inv": minimum_gain,
                "constant_m_inv": float(coefficients[0]),
                "linear_x_m_inv2": float(coefficients[1] / radius_x),
                "linear_y_m_inv2": float(coefficients[2] / radius_y),
                "quadratic_x_m_inv3": float(
                    coefficients[3] / radius_x**2
                ),
                "quadratic_y_m_inv3": float(
                    coefficients[4] / radius_y**2
                ),
                "rmse_m_inv": rmse,
                "normalized_rmse": rmse
                / max(span, np.finfo(float).tiny),
                "sampled_min_m_inv": float(np.min(channel[selected])),
                "sampled_max_m_inv": float(np.max(channel[selected])),
                "fitted_min_m_inv": float(np.min(fitted[selected])),
                "fitted_max_m_inv": float(np.max(fitted[selected])),
            }
        )
    return fitted_channels.reshape(values.shape), diagnostics


@dataclass(frozen=True)
class ModalGainStep:
    """One fixed-mode gain step and its energy-consistent local ledgers."""

    field: np.ndarray
    intensity_before: np.ndarray
    intensity_after: np.ndarray
    cell_average_intensity: np.ndarray
    active_transfer_per_volume: np.ndarray
    passive_loss_per_volume: np.ndarray
    effective_active_gain_m_inv: np.ndarray
    effective_passive_loss_m_inv: np.ndarray
    effective_total_gain_m_inv: np.ndarray
    raw_gain_exponent: np.ndarray
    gain_exponent: np.ndarray


@dataclass(frozen=True)
class GaussianProjection:
    """Power-preserving projection of sampled channels onto Gaussian modes."""

    field: np.ndarray
    coherent_overlap: np.ndarray
    radius_x_m: np.ndarray
    radius_y_m: np.ndarray
    center_x_m: np.ndarray
    center_y_m: np.ndarray
    mean_kx_per_m: np.ndarray
    mean_ky_per_m: np.ndarray
    quadratic_phase_x_per_m2: np.ndarray
    quadratic_phase_y_per_m2: np.ndarray
    sampled_m2_x: np.ndarray
    sampled_m2_y: np.ndarray


def project_to_gaussian_mode(
    field: np.ndarray,
    grid: Any,
    *,
    derivative_method: str = "spectral",
    compute_sampled_m2: bool = True,
) -> GaussianProjection:
    """Moment-match every leading channel to an axis-aligned complex Gaussian.

    Intensity centroid/radii and first-order wavefront tilt/curvature are retained.
    Higher spatial moments and phase aberrations are discarded.  Each channel's
    sampled power is preserved exactly before any physical aperture is reapplied.
    ``coherent_overlap`` quantifies how much information that projection removed.
    ``compute_sampled_m2=False`` skips only the residual angular-variance audit;
    the projected field and every q-defining moment remain unchanged.
    """

    original = np.asarray(field, dtype=complex)
    if original.shape[-2:] != (len(grid.y), len(grid.x)):
        raise ValueError("Gaussian projection field must end in the grid y/x shape")
    leading_shape = original.shape[:-2]
    channels = original.reshape((-1, len(grid.y), len(grid.x)))
    xx, yy = grid.transverse_mesh
    projected = np.empty_like(channels)
    overlaps: list[float] = []
    radii_x: list[float] = []
    radii_y: list[float] = []
    centers_x: list[float] = []
    centers_y: list[float] = []
    mean_kx_values: list[float] = []
    mean_ky_values: list[float] = []
    quadratic_phase_x_values: list[float] = []
    quadratic_phase_y_values: list[float] = []
    sampled_m2_x_values: list[float] = []
    sampled_m2_y_values: list[float] = []

    if derivative_method not in {"spectral", "finite_difference"}:
        raise ValueError(
            "Gaussian projection derivative_method must be spectral or "
            "finite_difference"
        )
    if hasattr(grid, "transverse_derivatives"):
        derivative_x, derivative_y = grid.transverse_derivatives(channels)
    elif grid.is_uniform_xy and derivative_method == "spectral":
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
    else:
        derivative_x = np.gradient(
            channels,
            grid.x,
            axis=-1,
            edge_order=2 if len(grid.x) >= 3 else 1,
        )
        derivative_y = np.gradient(
            channels,
            grid.y,
            axis=-2,
            edge_order=2 if len(grid.y) >= 3 else 1,
        )

    tiny = np.finfo(float).tiny
    for index, channel in enumerate(channels):
        intensity = np.abs(channel) ** 2
        total = float(grid.integrate_transverse(intensity))
        if not np.isfinite(total) or total <= tiny:
            projected[index] = channel
            overlaps.append(1.0)
            radii_x.append(0.0)
            radii_y.append(0.0)
            centers_x.append(0.0)
            centers_y.append(0.0)
            mean_kx_values.append(0.0)
            mean_ky_values.append(0.0)
            quadratic_phase_x_values.append(0.0)
            quadratic_phase_y_values.append(0.0)
            sampled_m2_x_values.append(1.0)
            sampled_m2_y_values.append(1.0)
            continue
        center_x = float(grid.integrate_transverse(intensity * xx) / total)
        center_y = float(grid.integrate_transverse(intensity * yy) / total)
        local_x = xx - center_x
        local_y = yy - center_y
        variance_x = max(
            float(grid.integrate_transverse(intensity * local_x**2) / total),
            tiny,
        )
        variance_y = max(
            float(grid.integrate_transverse(intensity * local_y**2) / total),
            tiny,
        )
        radius_x = 2.0 * np.sqrt(variance_x)
        radius_y = 2.0 * np.sqrt(variance_y)
        density = 2.0 / (np.pi * radius_x * radius_y) * np.exp(
            -2.0 * ((local_x / radius_x) ** 2 + (local_y / radius_y) ** 2)
        )
        sampled_density = float(grid.integrate_transverse(density))
        density /= max(sampled_density, tiny)

        momentum_x = np.imag(np.conj(channel) * derivative_x[index])
        momentum_y = np.imag(np.conj(channel) * derivative_y[index])
        mean_kx = float(grid.integrate_transverse(momentum_x) / total)
        mean_ky = float(grid.integrate_transverse(momentum_y) / total)
        covariance_xk = float(
            grid.integrate_transverse(momentum_x * local_x) / total
        )
        covariance_yk = float(
            grid.integrate_transverse(momentum_y * local_y) / total
        )
        chirp_x = covariance_xk / variance_x
        chirp_y = covariance_yk / variance_y
        quadratic_phase_x = 0.5 * chirp_x
        quadratic_phase_y = 0.5 * chirp_y
        if compute_sampled_m2:
            residual_derivative_x = derivative_x[index] - 1j * (
                mean_kx + chirp_x * local_x
            ) * channel
            residual_derivative_y = derivative_y[index] - 1j * (
                mean_ky + chirp_y * local_y
            ) * channel
            residual_variance_kx = max(
                float(
                    grid.integrate_transverse(np.abs(residual_derivative_x) ** 2)
                    / total
                ),
                0.0,
            )
            residual_variance_ky = max(
                float(
                    grid.integrate_transverse(np.abs(residual_derivative_y) ** 2)
                    / total
                ),
                0.0,
            )
            sampled_m2_x = 2.0 * np.sqrt(
                variance_x * residual_variance_kx
            )
            sampled_m2_y = 2.0 * np.sqrt(
                variance_y * residual_variance_ky
            )
        else:
            sampled_m2_x = 1.0
            sampled_m2_y = 1.0
        phase = (
            mean_kx * local_x
            + mean_ky * local_y
            + quadratic_phase_x * local_x**2
            + quadratic_phase_y * local_y**2
        )
        candidate = np.sqrt(total * density).astype(complex) * np.exp(1j * phase)
        coherent_inner = grid.integrate_transverse(np.conj(candidate) * channel)
        if abs(coherent_inner) > tiny:
            candidate *= np.exp(1j * np.angle(coherent_inner))
        if derivative_method == "finite_difference" and compute_sampled_m2:
            # A centred finite difference slightly underestimates the angular
            # variance of a narrow Gaussian.  Calibrate the raw determinant by
            # the moment-matched Gaussian on the identical grid so a clean
            # projected mode remains exactly M2=1 without introducing an FFT.
            candidate_derivative_x = np.gradient(
                candidate,
                grid.x,
                axis=-1,
                edge_order=2 if len(grid.x) >= 3 else 1,
            )
            candidate_derivative_y = np.gradient(
                candidate,
                grid.y,
                axis=-2,
                edge_order=2 if len(grid.y) >= 3 else 1,
            )
            if hasattr(grid, "transverse_derivatives"):
                candidate_derivative_x, candidate_derivative_y = grid.transverse_derivatives(candidate)
            candidate_intensity = np.abs(candidate) ** 2
            candidate_total = float(
                grid.integrate_transverse(candidate_intensity)
            )
            candidate_center_x = float(
                grid.integrate_transverse(candidate_intensity * xx)
                / candidate_total
            )
            candidate_center_y = float(
                grid.integrate_transverse(candidate_intensity * yy)
                / candidate_total
            )
            candidate_local_x = xx - candidate_center_x
            candidate_local_y = yy - candidate_center_y
            candidate_variance_x = max(
                float(
                    grid.integrate_transverse(
                        candidate_intensity * candidate_local_x**2
                    )
                    / candidate_total
                ),
                tiny,
            )
            candidate_variance_y = max(
                float(
                    grid.integrate_transverse(
                        candidate_intensity * candidate_local_y**2
                    )
                    / candidate_total
                ),
                tiny,
            )
            candidate_momentum_x = np.imag(
                np.conj(candidate) * candidate_derivative_x
            )
            candidate_momentum_y = np.imag(
                np.conj(candidate) * candidate_derivative_y
            )
            candidate_mean_kx = float(
                grid.integrate_transverse(candidate_momentum_x)
                / candidate_total
            )
            candidate_mean_ky = float(
                grid.integrate_transverse(candidate_momentum_y)
                / candidate_total
            )
            candidate_covariance_xk = float(
                grid.integrate_transverse(
                    candidate_momentum_x * candidate_local_x
                )
                / candidate_total
            )
            candidate_covariance_yk = float(
                grid.integrate_transverse(
                    candidate_momentum_y * candidate_local_y
                )
                / candidate_total
            )
            candidate_chirp_x = (
                candidate_covariance_xk / candidate_variance_x
            )
            candidate_chirp_y = (
                candidate_covariance_yk / candidate_variance_y
            )
            candidate_residual_x = candidate_derivative_x - 1j * (
                candidate_mean_kx
                + candidate_chirp_x * candidate_local_x
            ) * candidate
            candidate_residual_y = candidate_derivative_y - 1j * (
                candidate_mean_ky
                + candidate_chirp_y * candidate_local_y
            ) * candidate
            candidate_m2_x = 2.0 * np.sqrt(
                candidate_variance_x
                * max(
                    float(
                        grid.integrate_transverse(
                            np.abs(candidate_residual_x) ** 2
                        )
                        / candidate_total
                    ),
                    0.0,
                )
            )
            candidate_m2_y = 2.0 * np.sqrt(
                candidate_variance_y
                * max(
                    float(
                        grid.integrate_transverse(
                            np.abs(candidate_residual_y) ** 2
                        )
                        / candidate_total
                    ),
                    0.0,
                )
            )
            sampled_m2_x /= max(float(candidate_m2_x), tiny)
            sampled_m2_y /= max(float(candidate_m2_y), tiny)
        candidate_power = float(grid.integrate_transverse(np.abs(candidate) ** 2))
        overlap = abs(coherent_inner) ** 2 / max(total * candidate_power, tiny)
        projected[index] = candidate
        overlaps.append(float(np.clip(overlap, 0.0, 1.0)))
        radii_x.append(float(radius_x))
        radii_y.append(float(radius_y))
        centers_x.append(center_x)
        centers_y.append(center_y)
        mean_kx_values.append(mean_kx)
        mean_ky_values.append(mean_ky)
        quadratic_phase_x_values.append(quadratic_phase_x)
        quadratic_phase_y_values.append(quadratic_phase_y)
        sampled_m2_x_values.append(float(sampled_m2_x))
        sampled_m2_y_values.append(float(sampled_m2_y))

    return GaussianProjection(
        field=projected.reshape(original.shape),
        coherent_overlap=np.asarray(overlaps).reshape(leading_shape),
        radius_x_m=np.asarray(radii_x).reshape(leading_shape),
        radius_y_m=np.asarray(radii_y).reshape(leading_shape),
        center_x_m=np.asarray(centers_x).reshape(leading_shape),
        center_y_m=np.asarray(centers_y).reshape(leading_shape),
        mean_kx_per_m=np.asarray(mean_kx_values).reshape(leading_shape),
        mean_ky_per_m=np.asarray(mean_ky_values).reshape(leading_shape),
        quadratic_phase_x_per_m2=np.asarray(quadratic_phase_x_values).reshape(
            leading_shape
        ),
        quadratic_phase_y_per_m2=np.asarray(quadratic_phase_y_values).reshape(
            leading_shape
        ),
        sampled_m2_x=np.asarray(sampled_m2_x_values).reshape(leading_shape),
        sampled_m2_y=np.asarray(sampled_m2_y_values).reshape(leading_shape),
    )


def apply_modal_overlap_gain(
    field: np.ndarray,
    active_gain_m_inv: np.ndarray,
    passive_loss_m_inv: float | np.ndarray,
    grid: Any,
    optical_step_length_m: float,
    physical_cell_length_m: float,
    *,
    maximum_positive_exponent: float = 50.0,
) -> ModalGainStep:
    """Apply one scalar, intensity-overlap gain to each leading field channel.

    ``field``, ``active_gain_m_inv``, and the passive-loss map must be
    broadcast-compatible and end in ``(ny, nx)``.  Leading axes are independent
    channels (for example wavelength bins).  The returned local active/passive
    ledgers integrate exactly to the corresponding modal power contributions
    under the fixed-mode closure.
    """

    optical_length = float(optical_step_length_m)
    cell_length = float(physical_cell_length_m)
    if optical_length <= 0.0 or cell_length <= 0.0:
        raise ValueError("optical and physical cell lengths must be positive")

    input_field = np.asarray(field)
    intensity_before = np.abs(input_field) ** 2
    active = np.broadcast_to(
        np.asarray(active_gain_m_inv, dtype=float), intensity_before.shape
    )
    passive = np.broadcast_to(
        np.asarray(passive_loss_m_inv, dtype=float), intensity_before.shape
    )
    effective_active = transverse_weighted_mean(active, intensity_before, grid)
    effective_passive = transverse_weighted_mean(passive, intensity_before, grid)
    effective_total = effective_active - effective_passive
    raw_exponent = effective_total * optical_length
    exponent = np.minimum(raw_exponent, float(maximum_positive_exponent))

    # A positive exponent cap is a numerical safety valve.  If it activates,
    # the energy ledgers must describe the gain that was actually applied to
    # the field, rather than the uncapped gain map.  Passive loss is retained
    # and the active contribution is reduced by one channel-wide factor, which
    # is the only shape-preserving correction compatible with this closure.
    effective_total_applied = exponent / optical_length
    effective_active_applied = effective_total_applied + effective_passive
    active_scale = np.divide(
        effective_active_applied,
        effective_active,
        out=np.ones_like(effective_active_applied),
        where=np.abs(effective_active) > np.finfo(float).tiny,
    )
    active_scale = np.where(raw_exponent > maximum_positive_exponent, active_scale, 1.0)

    channel_shape = exponent.shape + (1, 1)
    intensity_scale = np.exp(exponent).reshape(channel_shape)
    output_field = input_field * np.sqrt(intensity_scale)
    intensity_after = intensity_before * intensity_scale

    average_factor = exponential_cell_average_factor(exponent).reshape(channel_shape)
    cell_average = intensity_before * average_factor * optical_length / cell_length
    active_transfer = active * active_scale.reshape(channel_shape) * cell_average
    passive_transfer = passive * cell_average
    return ModalGainStep(
        field=output_field,
        intensity_before=intensity_before,
        intensity_after=intensity_after,
        cell_average_intensity=cell_average,
        active_transfer_per_volume=active_transfer,
        passive_loss_per_volume=passive_transfer,
        effective_active_gain_m_inv=np.where(
            raw_exponent > maximum_positive_exponent,
            effective_active_applied,
            effective_active,
        ),
        effective_passive_loss_m_inv=effective_passive,
        effective_total_gain_m_inv=effective_total_applied,
        raw_gain_exponent=raw_exponent,
        gain_exponent=exponent,
    )


def validate_spatial_gain_transport(
    seed_cfg: dict[str, Any],
    *,
    resolved_signal_model: str,
    gain_saturation: bool,
) -> None:
    """Validate the reduced-order backend at the configuration boundary."""

    model = spatial_gain_transport_model(seed_cfg)
    if model not in SPATIAL_GAIN_MODELS:
        raise ValueError(
            "seed.spatial_gain_transport.model must be local_field, "
            "gaussian_mode_overlap, or parabolic_gain_fit"
        )
    if model == LOCAL_FIELD:
        return
    if model == PARABOLIC_GAIN_FIT:
        if resolved_signal_model not in {"fluence", "commercial_app_fast"}:
            raise ValueError(
                "parabolic_gain_fit requires simulation.signal_model=fluence "
                "or commercial_app_fast"
            )
        if str(seed_cfg.get("temporal", {}).get("type", "gaussian")) != "cw":
            raise ValueError(
                "parabolic_gain_fit currently requires a CW Seed so the fitted "
                "gain and stationary photon-flux equations share one closure"
            )
        fit = seed_cfg.get("spatial_gain_transport", {})
        for axis in "xy":
            value = fit.get(f"fit_radius_{axis}_m")
            if (
                value is None
                or not np.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(
                    "seed.spatial_gain_transport.fit_radius_"
                    f"{axis}_m must be positive for parabolic_gain_fit"
                )
            center = float(fit.get(f"center_{axis}_m", 0.0))
            if not np.isfinite(center):
                raise ValueError(
                    "seed.spatial_gain_transport.center_"
                    f"{axis}_m must be finite"
                )
        minimum_gain = float(fit.get("minimum_gain_m_inv", 0.0))
        if not np.isfinite(minimum_gain):
            raise ValueError(
                "seed.spatial_gain_transport.minimum_gain_m_inv must be finite"
            )
        return
    if resolved_signal_model not in {"fluence", "commercial_app_fast"}:
        raise ValueError(
            "gaussian_mode_overlap requires simulation.signal_model=fluence "
            "or commercial_app_fast; spectral_time, JOSA, and coherent CPA retain their "
            "own spatial gain operators"
        )
    if str(seed_cfg.get("spectrum", {}).get("type", "monochromatic")) != "monochromatic":
        raise ValueError(
            "gaussian_mode_overlap currently requires a monochromatic Seed spectrum; "
            "use local_field with spectral_time for wavelength-resolved gain"
        )
    if gain_saturation and resolved_signal_model != "commercial_app_fast":
        raise ValueError(
            "gaussian_mode_overlap currently requires physics.gain_saturation=false; "
            "use local_field for spatially resolved Frantz-Nodvik depletion"
        )
    spatial_modes = seed_cfg.get("spatial_modes", {})
    spatial_mode_model = str(spatial_modes.get("model", "effective_m2"))
    pure_hg00 = (
        spatial_mode_model == "hermite_gaussian"
        and int(spatial_modes.get("maximum_order_x", 0)) == 0
        and int(spatial_modes.get("maximum_order_y", 0)) == 0
        and np.isclose(float(seed_cfg.get("m2_x", 1.0)), 1.0)
        and np.isclose(float(seed_cfg.get("m2_y", 1.0)), 1.0)
    )
    commercial_app_mode_resolved = (
        resolved_signal_model == "commercial_app_fast"
        and spatial_mode_model == "incoherent_hg"
    )
    if (
        spatial_mode_model != "effective_m2"
        and not pure_hg00
        and not commercial_app_mode_resolved
    ):
        raise ValueError(
            "gaussian_mode_overlap requires effective_m2 or a pure HG00 mode; "
            "an incoherent higher-order Hermite-Gaussian ensemble needs the "
            "mode-resolved commercial_app_fast overlap model"
        )
    profile = str(seed_cfg.get("profile", "super_gaussian"))
    order_x = float(seed_cfg.get("order_x", 1.0))
    order_y = float(seed_cfg.get("order_y", 1.0))
    gaussian_profile = profile == "gaussian" or (
        profile in {"super_gaussian", "radial_super_gaussian"}
        and np.isclose(order_x, 1.0, rtol=0.0, atol=1e-12)
        and np.isclose(order_y, 1.0, rtol=0.0, atol=1e-12)
    )
    if not gaussian_profile:
        raise ValueError(
            "gaussian_mode_overlap requires a Gaussian Seed profile (order_x=order_y=1); "
            "use local_field for super-Gaussian, top-hat, annular, or custom shapes"
        )
    rotation_deg = float(seed_cfg.get("rotation_deg", 0.0))
    unequal_axes = not np.isclose(
        float(seed_cfg.get("waist_x_m", 0.0)),
        float(seed_cfg.get("waist_y_m", 0.0)),
        rtol=1e-12,
        atol=0.0,
    )
    if unequal_axes and not np.isclose(rotation_deg % 180.0, 0.0, atol=1e-12):
        raise ValueError(
            "gaussian_mode_overlap currently projects an axis-aligned Gaussian; "
            "set seed.rotation_deg=0 for an elliptical Seed, use a circular Seed, "
            "or select local_field for a rotated ellipse"
        )
