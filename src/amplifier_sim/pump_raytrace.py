"""Sequential stochastic Pump-ray transport and conservative 3-D deposition.

This module deliberately separates two ideas which are easy to conflate:

* a Pump *source parameterization* (waist/M2, waist/divergence, or an
  emitting diode radius and NA), and
* a Pump *propagation backend*.

The stochastic backend samples the configured source phase space, transports
the rays through the existing ordered optical train, and uses a track-length
estimator inside the gain medium.  It is a sequential geometrical-optics
solver, not a non-sequential CAD/stray-light package.  That boundary is kept
explicit in diagnostics and warnings.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from .grid import Grid3D
from .optics import expand_path_distances
from .spatial_modes import axis_focus_m
from .spectral import response_value


AbsorptionCallback = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray, np.ndarray]]
_MAX_DEPOSITION_STATISTICS_RECORDS = 1_000_000


@dataclass
class RayBundle:
    """A weighted local-coordinate ray bundle.

    ``slope_x`` and ``slope_y`` are dx/dz and dy/dz in the current sequential
    coordinate frame.  ``weight_W`` is optical power carried by each ray.
    """

    x_m: np.ndarray
    y_m: np.ndarray
    slope_x: np.ndarray
    slope_y: np.ndarray
    weight_W: np.ndarray
    refractive_index: float

    def scaled_copy(self, scale: float) -> "RayBundle":
        return RayBundle(
            self.x_m.copy(),
            self.y_m.copy(),
            self.slope_x.copy(),
            self.slope_y.copy(),
            self.weight_W * float(scale),
            float(self.refractive_index),
        )


@dataclass
class RayEntrance:
    bundle: RayBundle
    source_intensity_W_m2: np.ndarray
    entrance_intensity_W_m2: np.ndarray
    plane_intensities_W_m2: list[tuple[str, np.ndarray]]
    stats: dict[str, Any]
    warnings: list[str]


@dataclass
class RayVolumeResult:
    intensity_W_m2: np.ndarray
    absorbed_W_m3: np.ndarray
    active_absorbed_W_m3: np.ndarray
    passive_absorbed_W_m3: np.ndarray
    exit_intensity_W_m2: np.ndarray
    stats: dict[str, Any]
    warnings: list[str]


def _combine_ray_deposition_tracks(
    ray_cell_keys: list[np.ndarray], tracks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine all recorded contributions with the same ray/voxel key."""

    keys = np.concatenate(ray_cell_keys)
    contributions = np.concatenate(tracks)
    if not keys.size:
        return keys, contributions
    order = np.argsort(keys)
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
    ray_tracks = np.add.reduceat(contributions[order], starts)
    return sorted_keys[starts], ray_tracks


def _ray_deposition_moments(
    ray_cell_keys: list[np.ndarray], tracks: list[np.ndarray], cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Count rays and square their total voxel tracks, including correlations."""

    keys, ray_tracks = _combine_ray_deposition_tracks(ray_cell_keys, tracks)
    positive = ray_tracks > 0.0
    voxel_indices = keys[positive] % cells
    return (
        np.bincount(voxel_indices, minlength=cells),
        np.bincount(voxel_indices, weights=ray_tracks[positive]**2, minlength=cells),
    )


def _truncate_ray_weights(bundle: RayBundle, minimum_weight: float) -> float:
    """Remove sub-cutoff ray power, including chunks with no live rays."""

    truncated = (bundle.weight_W > 0.0) & (bundle.weight_W <= minimum_weight)
    lost_power = float(np.sum(bundle.weight_W[truncated]))
    bundle.weight_W[truncated] = 0.0
    return lost_power


def build_ray_entrance(
    pump_cfg: dict[str, Any],
    elements: list[dict[str, Any]],
    grid: Grid3D,
    interface_model: str,
    record_planes: bool = False,
) -> RayEntrance:
    """Sample a unit-power source and transport it to the crystal boundary."""

    ray_cfg = pump_cfg.get("raytrace", {})
    count = int(ray_cfg.get("ray_count", 100_000))
    seed = int(ray_cfg.get("random_seed", 12345))
    rng = np.random.default_rng(seed)
    bundle, launch_diagnostics = _sample_source(pump_cfg, count, rng)
    source_intensity, source_captured = _plane_intensity(bundle, grid)
    source_weights = bundle.weight_W.copy()
    ledger: list[dict[str, Any]] = []
    plane_intensities: list[tuple[str, np.ndarray]] = []
    warnings: list[str] = []
    _trace_external_elements(
        bundle,
        elements,
        pump_cfg,
        interface_model,
        ledger,
        warnings,
        grid if record_planes else None,
        plane_intensities,
    )
    interface_models = [
        str(element.get("model", interface_model))
        for element in elements
        if str(element.get("type", "")) == "interface"
    ]
    external_angles = np.arctan(
        np.sqrt(bundle.slope_x**2 + bundle.slope_y**2)
    )
    maximum_external_angle_deg = math.degrees(float(np.max(external_angles)))
    if maximum_external_angle_deg > 20.0 and any(
        element.get("type") in {"thin_lens", "thick_lens", "mirror"}
        for element in elements
    ):
        warnings.append(
            "The stochastic Pump path exceeds 20 degrees in an element modeled "
            "with paraxial lens/surface power; use a non-paraxial CAD ray tracer "
            "for quantitative high-angle imaging."
        )
    entrance_intensity, entrance_window_power = _plane_intensity(bundle, grid)
    source_power = float(np.sum(source_weights))
    external_power = float(np.sum(bundle.weight_W))
    per_launch_transmission = np.divide(
        bundle.weight_W,
        source_weights,
        out=np.zeros_like(bundle.weight_W),
        where=source_weights > 0.0,
    )
    transmission_mean = float(np.mean(per_launch_transmission))
    transmission_se = (
        float(np.std(per_launch_transmission, ddof=1) / math.sqrt(count))
        if count > 1
        else 0.0
    )
    if count < 10_000:
        warnings.append(
            "Stochastic Pump ray count is below 10,000; three-dimensional deposition noise may be large."
        )
    if source_captured < 0.98 * source_power:
        warnings.append(
            "The ray source display window contains only "
            f"{source_captured / max(source_power, np.finfo(float).tiny):.3%} of launched power. "
            "This does not clip transport, but the source-plane plot is incomplete."
        )
    stats: dict[str, Any] = {
        "ray_count": count,
        "random_seed": seed,
        "source_unit_power_W": source_power,
        "source_display_capture_fraction": source_captured
        / max(source_power, np.finfo(float).tiny),
        "power_after_external_train_fraction": external_power
        / max(source_power, np.finfo(float).tiny),
        "entrance_window_power_fraction": entrance_window_power
        / max(source_power, np.finfo(float).tiny),
        "external_transmission_monte_carlo_standard_error": transmission_se,
        "external_transmission_monte_carlo_relative_standard_error": transmission_se
        / max(transmission_mean, np.finfo(float).tiny),
        "element_power_ledger": ledger,
        "maximum_external_ray_angle_deg": maximum_external_angle_deg,
        "lens_and_curved_surface_model": "paraxial",
        "planar_interface_refraction": (
            "direct_pass"
            if interface_models and all(
                model == "direct_pass" for model in interface_models
            )
            else "vector_snell"
        ),
        "planar_interface_models": interface_models,
        **launch_diagnostics,
    }
    return RayEntrance(
        bundle,
        source_intensity,
        entrance_intensity,
        plane_intensities,
        stats,
        warnings,
    )


def _interior_index_gradient(
    values: np.ndarray,
    crystal_mask: np.ndarray,
    coordinate: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Differentiate within the material, never across its artificial zero mask.

    Use the usual nonuniform central stencil where both neighbours exist and
    a one-sided internal stencil at the material edge. Interface refraction
    is handled separately from this bulk gradient-index force.
    """

    data = np.moveaxis(values, axis, 0)
    valid = np.moveaxis(np.broadcast_to(crystal_mask, values.shape).astype(bool), axis, 0)
    gradient = np.moveaxis(np.gradient(values, coordinate, axis=axis), axis, 0)
    slope = np.diff(data, axis=0) / np.diff(coordinate).reshape(
        (-1,) + (1,) * (data.ndim - 1)
    )
    lower = np.zeros_like(valid)
    upper = np.zeros_like(valid)
    lower[1:] = valid[1:] & valid[:-1]
    upper[:-1] = lower[1:]
    gradient[:-1] = np.where(upper[:-1] & ~lower[:-1], slope, gradient[:-1])
    gradient[1:] = np.where(lower[1:] & ~upper[1:], slope, gradient[1:])
    return np.moveaxis(np.where(valid & (lower | upper), gradient, 0.0), 0, axis)


def propagate_ray_volume(
    entrance: RayEntrance,
    source_power_W: float,
    grid: Grid3D,
    path_indices: list[int],
    crystal_mask: np.ndarray,
    n_gain: float,
    absorption_components: AbsorptionCallback,
    population: np.ndarray,
    pump_depletion: bool,
    ray_cfg: dict[str, Any],
    delta_n: np.ndarray | None,
) -> RayVolumeResult:
    """Transport cached entrance rays through the crystal.

    A path-length estimator is used rather than counting ray hits.  For a ray
    segment with input power P, absorption alpha and length ds, its cell-average
    intensity contribution is

        P * (1 - exp(-alpha*ds)) / (alpha * cell_volume),

    with the analytic zero-alpha limit P*ds/cell_volume.  Consequently the
    deposited absorption and the intensity consumed by the rate equations are
    discretely consistent.
    """

    scale = float(source_power_W)
    bundle = entrance.bundle.scaled_copy(scale)
    intensity = np.zeros(grid.shape, dtype=float)
    absorbed = np.zeros(grid.shape, dtype=float)
    active_absorbed = np.zeros(grid.shape, dtype=float)
    passive_absorbed = np.zeros(grid.shape, dtype=float)
    deposition_samples = np.zeros(grid.shape, dtype=np.int64)
    deposition_contribution_sum = np.zeros(grid.shape, dtype=float)
    deposition_contribution_square_sum = np.zeros(grid.shape, dtype=float)
    warnings = list(entrance.warnings)
    minimum_relative = float(ray_cfg.get("minimum_relative_ray_power", 1e-8))
    per_ray_launch = scale / max(len(bundle.weight_W), 1)
    minimum_weight = per_ray_launch * minimum_relative
    maximum_substeps = int(ray_cfg.get("maximum_substeps_per_z_cell", 16))
    maximum_step_fraction = float(ray_cfg.get("maximum_transverse_step_fraction", 0.5))
    processing_chunk_size = int(ray_cfg.get("processing_chunk_size", 200_000))
    deposition_model = str(
        ray_cfg.get("deposition_model", "nearest_cell_track_length")
    )
    if deposition_model not in {
        "nearest_cell_track_length",
        "bilinear_track_length",
    }:
        raise ValueError(
            "Pump ray deposition_model must be nearest_cell_track_length or "
            "bilinear_track_length"
        )
    base_n = float(n_gain)
    bundle.refractive_index = base_n

    entrance_flat, entrance_inside = _inside_indices(bundle, grid, crystal_mask)
    entrance_before_aperture = float(np.sum(bundle.weight_W))
    entrance_aperture_loss = float(np.sum(bundle.weight_W[~entrance_inside]))
    bundle.weight_W[~entrance_inside] = 0.0
    input_power = float(np.sum(bundle.weight_W))
    entrance_surviving_rays = int(
        np.count_nonzero(bundle.weight_W > minimum_weight)
    )
    side_loss = 0.0
    truncation_loss = 0.0
    maximum_used_substeps = 1
    maximum_statistics_chunk_rays = 0
    maximum_positive_gain_exponent = 0.0
    maximum_absorption_optical_depth = 0.0

    thermal_refraction_enabled = bool(ray_cfg.get("thermal_refraction", True))
    thermal_delta_n = (
        np.zeros(grid.shape, dtype=float)
        if delta_n is None or not thermal_refraction_enabled
        else np.asarray(delta_n, dtype=float)
    )
    if thermal_delta_n.shape != grid.shape:
        raise ValueError("Pump ray-trace thermal delta_n must match the Pump grid")
    if np.any(thermal_delta_n):
        gradient_y = _interior_index_gradient(thermal_delta_n, crystal_mask, grid.y, 1)
        gradient_x = _interior_index_gradient(thermal_delta_n, crystal_mask, grid.x, 2)
        if float(
            entrance.stats.get("maximum_external_ray_angle_deg", 0.0)
        ) > 10.0:
            warnings.append(
                "Thermal Pump-ray bending uses the paraxial eikonal slope equation while "
                "the launch bundle exceeds 10 degrees. Check angular convergence or use a "
                "non-paraxial gradient-index ray solver for quantitative high-NA thermal feedback."
            )
    else:
        gradient_x = np.zeros_like(thermal_delta_n)
        gradient_y = np.zeros_like(thermal_delta_n)

    for iz in path_indices:
        alpha_total, alpha_active, alpha_passive = absorption_components(
            population[:, iz], iz
        )
        dz_cell = float(grid.z_widths[iz])
        alive = bundle.weight_W > minimum_weight
        if np.any(alive):
            transverse_cells = np.maximum(
                np.abs(bundle.slope_x[alive]) * dz_cell
                / max(float(np.min(grid.x_widths)), np.finfo(float).tiny),
                np.abs(bundle.slope_y[alive]) * dz_cell
                / max(float(np.min(grid.y_widths)), np.finfo(float).tiny),
            )
            requested_substeps = int(
                max(
                    1,
                    math.ceil(
                        float(np.max(transverse_cells))
                        / max(maximum_step_fraction, np.finfo(float).tiny)
                    ),
                )
            )
        else:
            requested_substeps = 1
        substeps = min(requested_substeps, maximum_substeps)
        maximum_used_substeps = max(maximum_used_substeps, substeps)
        if requested_substeps > maximum_substeps and not any(
            "maximum ray substeps" in warning for warning in warnings
        ):
            warnings.append(
                "The automatic ray marcher reached maximum ray substeps per z cell; "
                "increase the cap or refine z for high-NA transport."
            )
        dz = dz_cell / substeps
        cells = len(grid.x) * len(grid.y)
        volume_flat = grid.cell_volumes[iz].ravel()
        plane_shape = (len(grid.y), len(grid.x))
        # A ray is one independent source sample. Keep all its substeps in the
        # same chunk, so voxel contributions can be combined before squaring
        # without retaining histories for the entire launch bundle.
        stencil_size = 4 if deposition_model == "bilinear_track_length" else 1
        statistics_chunk_size = min(
            processing_chunk_size,
            max(1, _MAX_DEPOSITION_STATISTICS_RECORDS // (substeps * stencil_size)),
        )
        maximum_statistics_chunk_rays = max(
            maximum_statistics_chunk_rays,
            min(statistics_chunk_size, len(bundle.weight_W)),
        )
        for chunk_start in range(0, len(bundle.weight_W), statistics_chunk_size):
            chunk_slice = slice(chunk_start, chunk_start + statistics_chunk_size)
            chunk_bundle = RayBundle(
                bundle.x_m[chunk_slice], bundle.y_m[chunk_slice],
                bundle.slope_x[chunk_slice], bundle.slope_y[chunk_slice],
                bundle.weight_W[chunk_slice], bundle.refractive_index,
            )
            ray_cell_keys: list[np.ndarray] = []
            ray_cell_tracks: list[np.ndarray] = []
            statistics_records = 0
            for _ in range(substeps):
                _advance(chunk_bundle, 0.5 * dz)
                flat, inside = _inside_indices(chunk_bundle, grid, crystal_mask)
                escaped = (chunk_bundle.weight_W > minimum_weight) & ~inside
                if np.any(escaped):
                    side_loss += float(np.sum(chunk_bundle.weight_W[escaped]))
                    chunk_bundle.weight_W[escaped] = 0.0
                indices = np.flatnonzero(
                    inside & (chunk_bundle.weight_W > minimum_weight)
                )
                if indices.size == 0:
                    if pump_depletion and minimum_weight > 0.0:
                        truncation_loss += _truncate_ray_weights(chunk_bundle, minimum_weight)
                    _advance(chunk_bundle, 0.5 * dz)
                    continue
                ray_flat = flat[indices]
                if deposition_model == "bilinear_track_length":
                    stencil_flat, stencil_weights = _bilinear_transverse_stencil(
                        chunk_bundle.x_m[indices],
                        chunk_bundle.y_m[indices],
                        grid,
                        crystal_mask,
                    )
                else:
                    stencil_flat = ray_flat[:, None]
                    stencil_weights = np.ones((len(ray_flat), 1), dtype=float)

                def interpolate(plane: np.ndarray) -> np.ndarray:
                    values = np.asarray(plane, dtype=float).ravel()[stencil_flat]
                    return np.sum(values * stencil_weights, axis=1)

                ray_weight = chunk_bundle.weight_W[indices]
                ray_alpha = interpolate(alpha_total)
                direction_factor = np.sqrt(
                    1.0
                    + chunk_bundle.slope_x[indices] ** 2
                    + chunk_bundle.slope_y[indices] ** 2
                )
                ds = dz * direction_factor
                raw_exponent = -ray_alpha * ds
                maximum_absorption_optical_depth = max(
                    maximum_absorption_optical_depth,
                    float(np.max(np.maximum(ray_alpha, 0.0) * ds)),
                )
                maximum_positive_gain_exponent = max(
                    maximum_positive_gain_exponent,
                    float(np.max(raw_exponent)),
                )
                # Strong absorption is numerically safe and should be allowed
                # to underflow to zero.  Clip only the positive Pump-gain side.
                clipped_exponent = np.minimum(raw_exponent, 50.0)
                transmission = np.exp(clipped_exponent)
                optical_depth = ray_alpha * ds
                removed_fraction = -np.expm1(clipped_exponent)
                phi = np.ones_like(optical_depth)
                finite_depth = np.abs(optical_depth) > 1e-8
                np.divide(
                    removed_fraction,
                    optical_depth,
                    out=phi,
                    where=finite_depth,
                )
                if np.any(~finite_depth):
                    small = optical_depth[~finite_depth]
                    phi[~finite_depth] = (
                        1.0 - 0.5 * small + small * small / 6.0
                    )
                track = ray_weight * ds * phi
                deposit_flat = stencil_flat.ravel()
                deposit_weight = stencil_weights.ravel()
                contributing = deposit_weight > 0.0
                deposit_flat = deposit_flat[contributing]
                deposit_weight = deposit_weight[contributing]

                def deposit(values: np.ndarray) -> np.ndarray:
                    weighted = np.repeat(values, stencil_flat.shape[1])[
                        contributing
                    ] * deposit_weight
                    return np.bincount(
                        deposit_flat,
                        weights=weighted,
                        minlength=cells,
                    )

                deposited_track = deposit(track).reshape(plane_shape)
                local_intensity = deposited_track / volume_flat.reshape(plane_shape)
                intensity[iz] += local_intensity
                track_contributions = np.repeat(
                    track, stencil_flat.shape[1]
                )[contributing] * deposit_weight
                deposition_contribution_sum[iz] += deposited_track
                ray_cell_keys.append(
                    np.repeat(indices, stencil_flat.shape[1])[contributing]
                    * cells + deposit_flat
                )
                ray_cell_tracks.append(track_contributions)
                statistics_records += len(track_contributions)
                if statistics_records >= _MAX_DEPOSITION_STATISTICS_RECORDS:
                    # Also bound histories when a single ray has more substeps
                    # than the budget. Repeated voxel visits collapse to one
                    # entry while their cross terms remain in the final square.
                    combined_keys, combined_tracks = _combine_ray_deposition_tracks(
                        ray_cell_keys, ray_cell_tracks
                    )
                    ray_cell_keys = [combined_keys]
                    ray_cell_tracks = [combined_tracks]
                    statistics_records = len(combined_keys)
                # Gather alpha to attenuate a ray, then scatter its integrated
                # intensity with the same weights. The material cells must obey
                # Q_j = alpha_j I_j. Scattering the already interpolated ray
                # absorption instead smooths alpha twice and can deposit active
                # absorption into a cell with alpha_active=0. The adjoint gather/
                # scatter identity also preserves the ray's total power loss.
                local_active = np.asarray(alpha_active) * local_intensity
                local_passive = np.asarray(alpha_passive) * local_intensity
                active_absorbed[iz] += local_active
                passive_absorbed[iz] += local_passive
                absorbed[iz] += local_active + local_passive
                if pump_depletion:
                    chunk_bundle.weight_W[indices] *= transmission

                local_n = np.maximum(
                    base_n + interpolate(thermal_delta_n[iz]),
                    np.finfo(float).tiny,
                )
                chunk_bundle.slope_x[indices] += (
                    interpolate(gradient_x[iz]) / local_n * dz
                )
                chunk_bundle.slope_y[indices] += (
                    interpolate(gradient_y[iz]) / local_n * dz
                )
                if pump_depletion and minimum_weight > 0.0:
                    truncation_loss += _truncate_ray_weights(chunk_bundle, minimum_weight)
                _advance(chunk_bundle, 0.5 * dz)
            if ray_cell_keys:
                sample_counts, squared_contributions = _ray_deposition_moments(
                    ray_cell_keys, ray_cell_tracks, cells
                )
                deposition_samples[iz] += sample_counts.reshape(plane_shape)
                deposition_contribution_square_sum[iz] += (
                    squared_contributions.reshape(plane_shape)
                )

    _, exit_inside = _inside_indices(bundle, grid, crystal_mask)
    escaped_at_exit = (bundle.weight_W > minimum_weight) & ~exit_inside
    if np.any(escaped_at_exit):
        side_loss += float(np.sum(bundle.weight_W[escaped_at_exit]))
        bundle.weight_W[escaped_at_exit] = 0.0
    exit_intensity, output_power = _plane_intensity(bundle, grid)
    absorbed_power = float(grid.integrate_volume(absorbed))
    active_power = float(grid.integrate_volume(active_absorbed))
    passive_power = float(grid.integrate_volume(passive_absorbed))
    accounted = absorbed_power + output_power + side_loss + truncation_loss
    balance_residual = input_power - accounted
    relative_balance = abs(balance_residual) / max(input_power, np.finfo(float).tiny)
    if pump_depletion and relative_balance > 2e-10:
        warnings.append(
            "Pump ray-trace power balance did not close after absorption, side leakage, "
            "and numerical ray truncation."
        )
    if maximum_positive_gain_exponent > 50.0:
        warnings.append(
            "At least one stochastic Pump ray segment had a positive gain exponent above 50; "
            "the exponential overflow guard was active. Refine the ray/z steps and verify "
            "whether Pump stimulated emission belongs in the selected level model."
        )
    nonzero_cells = int(np.count_nonzero(intensity > 0.0))
    populated_sample_counts = deposition_samples[intensity > 0.0]
    effective_deposition_samples = np.divide(
        deposition_contribution_sum**2,
        deposition_contribution_square_sum,
        out=np.zeros_like(deposition_contribution_sum),
        where=deposition_contribution_square_sum > 0.0,
    )
    populated_effective_counts = effective_deposition_samples[intensity > 0.0]
    if populated_sample_counts.size:
        sample_count_minimum = int(np.min(populated_sample_counts))
        sample_count_median = float(np.median(populated_sample_counts))
        sample_count_tenth_percentile = float(
            np.percentile(populated_sample_counts, 10.0)
        )
        effective_sample_minimum = float(np.min(populated_effective_counts))
        effective_sample_median = float(np.median(populated_effective_counts))
        effective_sample_tenth_percentile = float(
            np.percentile(populated_effective_counts, 10.0)
        )
        low_sample_fraction = float(np.mean(populated_effective_counts < 30.0))
    else:
        sample_count_minimum = 0
        sample_count_median = 0.0
        sample_count_tenth_percentile = 0.0
        effective_sample_minimum = 0.0
        effective_sample_median = 0.0
        effective_sample_tenth_percentile = 0.0
        low_sample_fraction = 0.0
    if low_sample_fraction > 0.1:
        warnings.append(
            "More than 10% of populated Pump deposition cells have fewer than "
            "30 effective weighted ray samples; increase ray_count or "
            "reduce deposition-grid resolution before interpreting cell-scale structure."
        )
    effective_rays = _effective_sample_size(bundle.weight_W)
    stats: dict[str, Any] = {
        "gain_entrance_power_before_crystal_aperture_W": entrance_before_aperture,
        "gain_entrance_crystal_aperture_loss_W": entrance_aperture_loss,
        "grid_input_power_W": input_power,
        "output_power_W": output_power,
        "absorbed_power_W": absorbed_power,
        "active_absorbed_power_W": active_power,
        "passive_absorbed_power_W": passive_power,
        "crystal_side_aperture_loss_W": side_loss,
        "ray_power_truncation_loss_W": truncation_loss,
        "crystal_transport_accounted_power_W": accounted if pump_depletion else None,
        "crystal_power_balance_residual_W": balance_residual if pump_depletion else None,
        "crystal_power_balance_residual_relative": relative_balance if pump_depletion else None,
        "maximum_used_substeps_per_z_cell": maximum_used_substeps,
        "processing_chunk_size": processing_chunk_size,
        "maximum_deposition_statistics_chunk_rays": maximum_statistics_chunk_rays,
        "deposition_statistics_record_budget": _MAX_DEPOSITION_STATISTICS_RECORDS,
        "maximum_positive_pump_ray_gain_exponent": maximum_positive_gain_exponent,
        "maximum_pump_ray_absorption_optical_depth": (
            maximum_absorption_optical_depth
        ),
        "positive_pump_ray_exponent_clipping_active": bool(
            maximum_positive_gain_exponent > 50.0
        ),
        "deposition_estimator": deposition_model,
        "deposition_sample_unit": "independent_ray_per_voxel",
        "nonzero_deposition_cells": nonzero_cells,
        "entrance_surviving_ray_count": entrance_surviving_rays,
        "deposition_samples_minimum": sample_count_minimum,
        "deposition_samples_median": sample_count_median,
        "deposition_samples_tenth_percentile": sample_count_tenth_percentile,
        "deposition_effective_samples_minimum": effective_sample_minimum,
        "deposition_effective_samples_median": effective_sample_median,
        "deposition_effective_samples_tenth_percentile": (
            effective_sample_tenth_percentile
        ),
        "deposition_cells_below_30_samples_fraction": low_sample_fraction,
        "median_poisson_relative_noise_proxy": (
            1.0 / math.sqrt(effective_sample_median)
            if effective_sample_median > 0.0
            else None
        ),
        "exit_effective_sample_size": effective_rays,
        "thermal_ray_bending": bool(np.any(thermal_delta_n)),
        "thermal_refraction_enabled": thermal_refraction_enabled,
    }
    return RayVolumeResult(
        intensity,
        absorbed,
        active_absorbed,
        passive_absorbed,
        exit_intensity,
        stats,
        warnings,
    )


def _sample_source(
    cfg: dict[str, Any],
    count: int,
    rng: np.random.Generator,
) -> tuple[RayBundle, dict[str, Any]]:
    source = cfg.get("source_definition", {})
    mode = str(source.get("mode", "waist_m2"))
    wavelength = float(cfg["wavelength_m"])
    profile = str(cfg.get("profile", "super_gaussian"))
    if mode == "emitter_na":
        radius_x = float(source.get("emitting_radius_x_m", cfg["waist_x_m"]))
        radius_y = float(source.get("emitting_radius_y_m", cfg["waist_y_m"]))
        exponent_x = float(source.get("super_gaussian_exponent_x", 2.0))
        exponent_y = float(source.get("super_gaussian_exponent_y", 2.0))
    else:
        radius_x = float(cfg["waist_x_m"])
        radius_y = float(cfg["waist_y_m"])
        exponent_x = 2.0 * float(cfg.get("order_x", 1.0))
        exponent_y = 2.0 * float(cfg.get("order_y", 1.0))
    local_x, local_y = _sample_spatial_profile(
        rng,
        count,
        radius_x,
        radius_y,
        exponent_x,
        exponent_y,
        profile,
        cfg,
    )

    angular_model = str(cfg.get("raytrace", {}).get("angular_distribution", "auto"))
    angular_exponent_x = float(
        cfg.get("raytrace", {}).get("angular_super_gaussian_exponent_x")
        if cfg.get("raytrace", {}).get("angular_super_gaussian_exponent_x")
        is not None
        else exponent_x
    )
    angular_exponent_y = float(
        cfg.get("raytrace", {}).get("angular_super_gaussian_exponent_y")
        if cfg.get("raytrace", {}).get("angular_super_gaussian_exponent_y")
        is not None
        else exponent_y
    )
    if angular_model == "auto":
        angular_model = (
            "uniform_na"
            if mode in {"emitter_na", "waist_divergence"}
            else "generalized_gaussian"
        )
    if mode == "emitter_na":
        source_n = float(source.get("free_space_refractive_index", 1.0))
        na_x = float(source.get("numerical_aperture_x", 0.0))
        na_y = float(source.get("numerical_aperture_y", 0.0))
        sin_x = np.clip(na_x / source_n, 0.0, 1.0 - 1e-15)
        sin_y = np.clip(na_y / source_n, 0.0, 1.0 - 1e-15)
        if angular_model == "uniform_na":
            direction_x, direction_y = _sample_uniform_ellipse(
                rng, count, sin_x, sin_y
            )
            angular_measure = "uniform_direction_cosine_area_constant_radiance"
        elif angular_model == "uniform_solid_angle":
            direction_x, direction_y = _sample_uniform_solid_angle_ellipse(
                rng, count, sin_x, sin_y
            )
            angular_measure = "uniform_solid_angle_inside_na_support"
        else:
            angular_measure = "truncated_generalized_gaussian_slope"
        if angular_model in {"uniform_na", "uniform_solid_angle"}:
            direction_z = np.sqrt(
                np.maximum(1.0 - direction_x**2 - direction_y**2, 1e-15)
            )
            local_sx = direction_x / direction_z
            local_sy = direction_y / direction_z
        else:
            half_x = math.tan(math.asin(sin_x))
            half_y = math.tan(math.asin(sin_y))
            local_sx, local_sy = _sample_truncated_generalized_ellipse(
                rng,
                count,
                half_x,
                half_y,
                angular_exponent_x,
                angular_exponent_y,
            )
    else:
        source_n = float(cfg.get("source_refractive_index", 1.0))
        if mode == "waist_divergence":
            half_x = math.tan(
                math.radians(float(source.get("divergence_half_angle_x_deg", 0.0)))
            )
            half_y = math.tan(
                math.radians(float(source.get("divergence_half_angle_y_deg", 0.0)))
            )
        else:
            half_x = float(cfg.get("m2_x", 1.0)) * wavelength / (
                math.pi * source_n * radius_x
            )
            half_y = float(cfg.get("m2_y", 1.0)) * wavelength / (
                math.pi * source_n * radius_y
            )
        if angular_model == "uniform_na":
            local_sx, local_sy = _sample_uniform_ellipse(
                rng, count, half_x, half_y
            )
            angular_measure = "uniform_slope_area"
        elif angular_model == "uniform_solid_angle":
            sin_x = half_x / math.sqrt(1.0 + half_x**2)
            sin_y = half_y / math.sqrt(1.0 + half_y**2)
            direction_x, direction_y = _sample_uniform_solid_angle_ellipse(
                rng, count, sin_x, sin_y
            )
            direction_z = np.sqrt(
                np.maximum(1.0 - direction_x**2 - direction_y**2, 1e-15)
            )
            local_sx = direction_x / direction_z
            local_sy = direction_y / direction_z
            angular_measure = "uniform_solid_angle_inside_configured_support"
        else:
            local_sx, local_sy = _sample_generalized_pair(
                rng,
                count,
                half_x,
                half_y,
                angular_exponent_x,
                angular_exponent_y,
                radial=profile == "radial_super_gaussian"
                and math.isclose(
                    angular_exponent_x,
                    angular_exponent_y,
                    rel_tol=0.0,
                    abs_tol=1e-14,
                ),
            )
            angular_measure = "generalized_gaussian_slope"

    rotation = math.radians(float(cfg.get("rotation_deg", 0.0)))
    cosine, sine = math.cos(rotation), math.sin(rotation)
    rotated_x = cosine * local_x - sine * local_y
    rotated_y = sine * local_x + cosine * local_y
    rotated_sx = cosine * local_sx - sine * local_sy
    rotated_sy = sine * local_sx + cosine * local_sy
    nominal_sx = math.tan(math.radians(float(cfg.get("incidence_angle_x_deg", 0.0))))
    nominal_sy = math.tan(math.radians(float(cfg.get("incidence_angle_y_deg", 0.0))))
    slope_x = rotated_sx + nominal_sx
    slope_y = rotated_sy + nominal_sy
    center_x = float(cfg.get("center_x_m", 0.0))
    center_y = float(cfg.get("center_y_m", 0.0))
    if mode == "emitter_na":
        x = center_x + rotated_x
        y = center_y + rotated_y
    else:
        # The public parameters define a waist/caustic.  Launch the ray bundle
        # from z=0 such that its sampled phase space reaches that waist.
        # Back-project the two astigmatic waists in their principal axes
        # before rotating the phase space. Unequal focus distances do not
        # commute with a rotation. The chief-ray displacement retains the
        # same laboratory-frame centre convention as the field source.
        focus_x, focus_y = axis_focus_m(cfg, "x"), axis_focus_m(cfg, "y")
        source_local_x = local_x - focus_x * local_sx
        source_local_y = local_y - focus_y * local_sy
        x = center_x + cosine * source_local_x - sine * source_local_y - focus_x * nominal_sx
        y = center_y + sine * source_local_x + cosine * source_local_y - focus_y * nominal_sy
    weights = np.full(count, 1.0 / count, dtype=float)
    direction_norm = np.sqrt(1.0 + slope_x**2 + slope_y**2)
    direction_x = slope_x / direction_norm
    direction_y = slope_y / direction_norm

    def d4sigma_radius(values: np.ndarray) -> float:
        # D4-sigma is a diameter (4 sigma); plots conventionally show its
        # radius, 2 sigma.  Report the radius explicitly to avoid that common
        # factor-of-two ambiguity.
        centered = np.asarray(values, dtype=float) - float(np.mean(values))
        return 2.0 * math.sqrt(float(np.mean(centered**2)))

    def correlation(first: np.ndarray, second: np.ndarray) -> float:
        first_centered = first - float(np.mean(first))
        second_centered = second - float(np.mean(second))
        denominator = math.sqrt(
            float(np.mean(first_centered**2) * np.mean(second_centered**2))
        )
        if denominator <= np.finfo(float).tiny:
            return 0.0
        return float(np.mean(first_centered * second_centered) / denominator)

    source_radius_d4_x = d4sigma_radius(rotated_x)
    source_radius_d4_y = d4sigma_radius(rotated_y)
    slope_radius_d4_x = d4sigma_radius(slope_x)
    slope_radius_d4_y = d4sigma_radius(slope_y)
    direction_radius_d4_x = d4sigma_radius(direction_x)
    direction_radius_d4_y = d4sigma_radius(direction_y)
    x_slope_correlation = correlation(rotated_x, slope_x)
    y_slope_correlation = correlation(rotated_y, slope_y)
    x_direction_correlation = correlation(rotated_x, direction_x)
    y_direction_correlation = correlation(rotated_y, direction_y)

    def phase_space_m2(
        spatial_radius: float,
        angular_radius: float,
        position_angle_correlation: float,
    ) -> float:
        determinant_factor = math.sqrt(
            max(1.0 - position_angle_correlation**2, 0.0)
        )
        return (
            math.pi
            * source_n
            * spatial_radius
            * angular_radius
            * determinant_factor
            / wavelength
        )

    diagnostics = {
        "source_phase_space_mode": mode,
        "ray_angular_distribution": angular_model,
        "ray_angular_sampling_measure": angular_measure,
        "source_radius_x_m": radius_x,
        "source_radius_y_m": radius_y,
        "source_super_gaussian_exponent_x": exponent_x,
        "source_super_gaussian_exponent_y": exponent_y,
        "angular_super_gaussian_exponent_x": angular_exponent_x,
        "angular_super_gaussian_exponent_y": angular_exponent_y,
        "source_wavelength_m": wavelength,
        "source_profile_sampling": (
            "stochastic_super_gaussian_phase_space"
            if profile in {"super_gaussian", "radial_super_gaussian"}
            else f"stochastic_{profile}_phase_space"
        ),
        "configured_source_radius_semantics": (
            "profile scale in I proportional to exp[-2 (r/radius)^exponent]"
            if profile in {"super_gaussian", "radial_super_gaussian"}
            else "profile-specific support or scale"
        ),
        "sampled_source_d4sigma_radius_x_m": source_radius_d4_x,
        "sampled_source_d4sigma_radius_y_m": source_radius_d4_y,
        "sampled_launch_slope_d4sigma_radius_x": slope_radius_d4_x,
        "sampled_launch_slope_d4sigma_radius_y": slope_radius_d4_y,
        "sampled_direction_cosine_d4sigma_radius_x": direction_radius_d4_x,
        "sampled_direction_cosine_d4sigma_radius_y": direction_radius_d4_y,
        "sampled_x_slope_correlation": x_slope_correlation,
        "sampled_y_slope_correlation": y_slope_correlation,
        "sampled_x_direction_cosine_correlation": x_direction_correlation,
        "sampled_y_direction_cosine_correlation": y_direction_correlation,
        "sampled_paraxial_phase_space_m2_x": phase_space_m2(
            source_radius_d4_x, slope_radius_d4_x, x_slope_correlation
        ),
        "sampled_paraxial_phase_space_m2_y": phase_space_m2(
            source_radius_d4_y, slope_radius_d4_y, y_slope_correlation
        ),
        "sampled_direction_cosine_phase_space_m2_x": phase_space_m2(
            source_radius_d4_x,
            direction_radius_d4_x,
            x_direction_correlation,
        ),
        "sampled_direction_cosine_phase_space_m2_y": phase_space_m2(
            source_radius_d4_y,
            direction_radius_d4_y,
            y_direction_correlation,
        ),
        "sampled_phase_space_m2_semantics": (
            "ray-ensemble D4sigma second moments including position-angle correlation; geometric diagnostic, not a wavefront measurement"
        ),
        "maximum_launch_ray_angle_deg": math.degrees(
            float(np.max(np.arctan(np.sqrt(slope_x**2 + slope_y**2))))
        ),
    }
    return RayBundle(x, y, slope_x, slope_y, weights, source_n), diagnostics


def _sample_spatial_profile(
    rng: np.random.Generator,
    count: int,
    radius_x: float,
    radius_y: float,
    exponent_x: float,
    exponent_y: float,
    profile: str,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if profile == "top_hat":
        radial = np.sqrt(rng.random(count))
        angle = rng.uniform(0.0, 2.0 * math.pi, count)
        return radius_x * radial * np.cos(angle), radius_y * radial * np.sin(angle)
    if profile == "annular":
        ring_order = float(cfg.get("extras", {}).get("ring_order", 0.5))
        radial = np.sqrt(rng.gamma(ring_order + 1.0, 1.0, count) / 2.0)
        angle = rng.uniform(0.0, 2.0 * math.pi, count)
        return radius_x * radial * np.cos(angle), radius_y * radial * np.sin(angle)
    radial_profile = profile == "radial_super_gaussian" and math.isclose(
        exponent_x, exponent_y, rel_tol=0.0, abs_tol=1e-14
    )
    return _sample_generalized_pair(
        rng,
        count,
        radius_x,
        radius_y,
        exponent_x,
        exponent_y,
        radial_profile,
    )


def _sample_generalized_pair(
    rng: np.random.Generator,
    count: int,
    scale_x: float,
    scale_y: float,
    exponent_x: float,
    exponent_y: float,
    radial: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if scale_x == 0.0 and scale_y == 0.0:
        return np.zeros(count), np.zeros(count)
    if radial:
        exponent = exponent_x
        gamma_sample = rng.gamma(2.0 / exponent, 1.0, count)
        radius = (gamma_sample / 2.0) ** (1.0 / exponent)
        angle = rng.uniform(0.0, 2.0 * math.pi, count)
        return scale_x * radius * np.cos(angle), scale_y * radius * np.sin(angle)

    def axis(scale: float, exponent: float) -> np.ndarray:
        if scale == 0.0:
            return np.zeros(count)
        magnitude = (rng.gamma(1.0 / exponent, 1.0, count) / 2.0) ** (
            1.0 / exponent
        )
        sign = np.where(rng.random(count) < 0.5, -1.0, 1.0)
        return scale * sign * magnitude

    return axis(scale_x, exponent_x), axis(scale_y, exponent_y)


def _sample_uniform_ellipse(
    rng: np.random.Generator,
    count: int,
    scale_x: float,
    scale_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    radial = np.sqrt(rng.random(count))
    azimuth = rng.uniform(0.0, 2.0 * math.pi, count)
    return (
        scale_x * radial * np.cos(azimuth),
        scale_y * radial * np.sin(azimuth),
    )


def _sample_uniform_solid_angle_ellipse(
    rng: np.random.Generator,
    count: int,
    sin_half_angle_x: float,
    sin_half_angle_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample directions uniformly in solid angle inside elliptical NA support.

    Direction-cosine area obeys ``du_x du_y = cos(theta) dOmega``.  Uniform
    sampling in the direction-cosine ellipse therefore represents constant
    radiance from a plane source, whereas this rejection sampler adds the
    reciprocal-cosine Jacobian required for a distribution uniform in dOmega.
    Keeping both models explicit avoids calling two physically different
    measures simply "uniform NA".
    """

    if count < 1:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    scale_x = float(sin_half_angle_x)
    scale_y = float(sin_half_angle_y)
    if scale_x == 0.0 and scale_y == 0.0:
        return np.zeros(count), np.zeros(count)
    maximum_sine = max(abs(scale_x), abs(scale_y))
    minimum_cosine = math.sqrt(max(1.0 - maximum_sine**2, 1e-15))
    result_x = np.empty(count, dtype=float)
    result_y = np.empty(count, dtype=float)
    filled = 0
    while filled < count:
        candidate_count = max(256, 2 * (count - filled))
        candidate_x, candidate_y = _sample_uniform_ellipse(
            rng, candidate_count, scale_x, scale_y
        )
        cosine = np.sqrt(
            np.maximum(1.0 - candidate_x**2 - candidate_y**2, 1e-15)
        )
        accepted = rng.random(candidate_count) < minimum_cosine / cosine
        take = min(int(np.count_nonzero(accepted)), count - filled)
        if take:
            result_x[filled : filled + take] = candidate_x[accepted][:take]
            result_y[filled : filled + take] = candidate_y[accepted][:take]
            filled += take
    return result_x, result_y


def _sample_truncated_generalized_ellipse(
    rng: np.random.Generator,
    count: int,
    scale_x: float,
    scale_y: float,
    exponent_x: float,
    exponent_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a generalized Gaussian without exceeding configured NA support."""

    result_x = np.empty(count, dtype=float)
    result_y = np.empty(count, dtype=float)
    filled = 0
    while filled < count:
        candidate_count = max(256, 2 * (count - filled))
        candidate_x, candidate_y = _sample_generalized_pair(
            rng,
            candidate_count,
            scale_x,
            scale_y,
            exponent_x,
            exponent_y,
            radial=False,
        )
        normalized = np.zeros(candidate_count, dtype=float)
        if scale_x > 0.0:
            normalized += (candidate_x / scale_x) ** 2
        if scale_y > 0.0:
            normalized += (candidate_y / scale_y) ** 2
        accepted = normalized <= 1.0
        take = min(int(np.count_nonzero(accepted)), count - filled)
        if take:
            result_x[filled : filled + take] = candidate_x[accepted][:take]
            result_y[filled : filled + take] = candidate_y[accepted][:take]
            filled += take
    return result_x, result_y


def _trace_external_elements(
    bundle: RayBundle,
    elements: list[dict[str, Any]],
    cfg: dict[str, Any],
    interface_model: str,
    ledger: list[dict[str, Any]],
    warnings: list[str],
    record_grid: Grid3D | None,
    plane_intensities: list[tuple[str, np.ndarray]],
) -> None:
    wavelength = float(cfg["wavelength_m"])
    for index, element in enumerate(expand_path_distances(elements)):
        kind = str(element.get("type", ""))
        name = str(element.get("name", f"{index + 1}_{kind}"))
        before = float(np.sum(bundle.weight_W))
        if kind in {"observation", "plane", "dispersive"}:
            pass
        elif kind == "free_space":
            length = float(element["length_m"])
            _advance(bundle, length)
            bundle.weight_W *= math.exp(
                -float(element.get("absorption_m_inv", 0.0)) * length
            )
        elif kind == "thin_lens":
            _thin_lens(bundle, element)
            bundle.weight_W *= _element_scalar(element, "power_transmission", wavelength, 1.0)
            _apply_optional_aperture(bundle, element)
        elif kind == "thick_lens":
            lens_n = float(element["refractive_index"])
            n_after = float(element.get("n_after", bundle.refractive_index))
            _curved_refraction(bundle, element, "front", lens_n, interface_model, wavelength)
            thickness = float(element["thickness_m"])
            _advance(bundle, thickness)
            bundle.weight_W *= math.exp(
                -float(element.get("absorption_m_inv", 0.0)) * thickness
            )
            _curved_refraction(bundle, element, "back", n_after, interface_model, wavelength)
            _apply_optional_aperture(bundle, element)
        elif kind == "aperture":
            bundle.weight_W *= _aperture_transmission(bundle, element)
            bundle.weight_W *= _element_scalar(
                element, "power_transmission", wavelength, 1.0
            )
        elif kind == "interface":
            _planar_interface(
                bundle,
                float(element["n_after"]),
                str(element.get("model", interface_model)),
                _element_scalar(element, "power_transmission", wavelength, 1.0),
                str(element.get("polarization", "unpolarized")),
                float(element.get("axis_angle_x_deg", 0.0)),
                float(element.get("axis_angle_y_deg", 0.0)),
            )
        elif kind in {"tilt", "coordinate_break"}:
            bundle.x_m += float(element.get("shift_x_m", 0.0))
            bundle.y_m += float(element.get("shift_y_m", 0.0))
            bundle.slope_x += math.tan(
                math.radians(float(element.get("angle_x_deg", 0.0)))
            )
            bundle.slope_y += math.tan(
                math.radians(float(element.get("angle_y_deg", 0.0)))
            )
        elif kind == "mirror":
            radius_x = element.get("radius_x_m")
            radius_y = element.get("radius_y_m")
            if radius_x not in (None, 0):
                bundle.slope_x -= 2.0 * (
                    bundle.x_m - float(element.get("center_x_m", 0.0))
                ) / float(radius_x)
            if radius_y not in (None, 0):
                bundle.slope_y -= 2.0 * (
                    bundle.y_m - float(element.get("center_y_m", 0.0))
                ) / float(radius_y)
            bundle.weight_W *= _element_scalar(
                element, "power_reflectivity", wavelength, 1.0
            )
            if bool(element.get("flip_x", False)):
                bundle.x_m *= -1.0
                bundle.slope_x *= -1.0
            if bool(element.get("flip_y", False)):
                bundle.y_m *= -1.0
                bundle.slope_y *= -1.0
        elif kind == "phase_screen":
            wave_number = 2.0 * math.pi / wavelength * bundle.refractive_index
            bundle.slope_x += (
                2.0
                * float(element.get("quadratic_x_rad_m2") or 0.0)
                * bundle.x_m
                / wave_number
            )
            bundle.slope_y += (
                2.0
                * float(element.get("quadratic_y_rad_m2") or 0.0)
                * bundle.y_m
                / wave_number
            )
            bundle.weight_W *= _element_scalar(
                element, "power_transmission", wavelength, 1.0
            )
            warnings.append(
                f"{name}: stochastic rays used only the phase-screen gradient; constant phase is not a ray observable."
            )
        elif kind == "loss":
            bundle.weight_W *= _element_scalar(
                element, "power_transmission", wavelength, 1.0
            )
        else:
            raise ValueError(
                f"Unsupported stochastic Pump ray-trace element type {kind!r}"
            )
        after = float(np.sum(bundle.weight_W))
        ledger.append(
            {
                "name": name,
                "type": kind,
                **(
                    {"interface_model": str(element.get("model", interface_model))}
                    if kind == "interface"
                    else {}
                ),
                "input_fraction": before,
                "output_fraction": after,
                "loss_fraction": before - after,
                "surviving_rays": int(np.count_nonzero(bundle.weight_W > 0.0)),
            }
        )
        if record_grid is not None:
            plane_intensity, _ = _plane_intensity(bundle, record_grid)
            plane_intensities.append((name, plane_intensity))


def _advance(bundle: RayBundle, distance_m: float) -> None:
    bundle.x_m += float(distance_m) * bundle.slope_x
    bundle.y_m += float(distance_m) * bundle.slope_y


def _thin_lens(bundle: RayBundle, element: dict[str, Any]) -> None:
    fx = element.get("focal_length_x_m", element.get("focal_length_m"))
    fy = element.get("focal_length_y_m", element.get("focal_length_m"))
    cx = float(element.get("center_x_m", 0.0))
    cy = float(element.get("center_y_m", 0.0))
    if fx not in (None, 0):
        bundle.slope_x -= (bundle.x_m - cx) / float(fx)
    if fy not in (None, 0):
        bundle.slope_y -= (bundle.y_m - cy) / float(fy)


def _curved_refraction(
    bundle: RayBundle,
    element: dict[str, Any],
    surface: str,
    n_after: float,
    interface_model: str,
    wavelength_m: float,
) -> None:
    n_before = float(bundle.refractive_index)
    radius_x = element.get(
        f"radius_{surface}_x_m", element.get(f"radius_{surface}_m")
    )
    radius_y = element.get(
        f"radius_{surface}_y_m", element.get(f"radius_{surface}_m")
    )
    model = str(
        element.get(f"{surface}_model", element.get("model", interface_model))
    )
    specified = _element_scalar(
        element, f"{surface}_power_transmission", wavelength_m, 1.0
    )
    if model == "direct_pass":
        bundle.weight_W *= specified
        bundle.refractive_index = n_after
        return
    surface_slope_x = (
        np.zeros_like(bundle.x_m)
        if radius_x in (None, 0)
        else -(bundle.x_m - float(element.get("center_x_m", 0.0)))
        / float(radius_x)
    )
    surface_slope_y = (
        np.zeros_like(bundle.y_m)
        if radius_y in (None, 0)
        else -(bundle.y_m - float(element.get("center_y_m", 0.0)))
        / float(radius_y)
    )
    relative_angle = np.arctan(
        np.sqrt(
            (bundle.slope_x - surface_slope_x) ** 2
            + (bundle.slope_y - surface_slope_y) ** 2
        )
    )
    bundle.weight_W *= _interface_transmission(
        n_before,
        n_after,
        relative_angle,
        model,
        specified,
        str(element.get("polarization", "unpolarized")),
    )
    if radius_x not in (None, 0):
        bundle.slope_x = (
            n_before / n_after * bundle.slope_x
            + (n_before - n_after)
            / (float(radius_x) * n_after)
            * (bundle.x_m - float(element.get("center_x_m", 0.0)))
        )
    else:
        bundle.slope_x *= n_before / n_after
    if radius_y not in (None, 0):
        bundle.slope_y = (
            n_before / n_after * bundle.slope_y
            + (n_before - n_after)
            / (float(radius_y) * n_after)
            * (bundle.y_m - float(element.get("center_y_m", 0.0)))
        )
    else:
        bundle.slope_y *= n_before / n_after
    bundle.refractive_index = n_after


def _planar_interface(
    bundle: RayBundle,
    n_after: float,
    model: str,
    specified: float,
    polarization: str,
    axis_angle_x_deg: float = 0.0,
    axis_angle_y_deg: float = 0.0,
) -> None:
    n_before = float(bundle.refractive_index)
    bundle.slope_x += math.tan(math.radians(axis_angle_x_deg))
    bundle.slope_y += math.tan(math.radians(axis_angle_y_deg))
    if model == "direct_pass":
        # Commercial App's ``crystalRayTrace`` object can be configured as a geometric
        # volume boundary: rays enter the gain volume without a Snell change
        # of direction, while the material index still becomes active for
        # in-volume optical/thermal calculations.  Keep this distinct from
        # setting the material refractive index to one.
        bundle.weight_W *= float(specified)
        bundle.refractive_index = n_after
        return
    norm = np.sqrt(1.0 + bundle.slope_x**2 + bundle.slope_y**2)
    direction_x = bundle.slope_x / norm
    direction_y = bundle.slope_y / norm
    direction_z = 1.0 / norm
    incidence = np.arccos(np.clip(direction_z, 0.0, 1.0))
    bundle.weight_W *= _interface_transmission(
        n_before,
        n_after,
        incidence,
        model,
        specified,
        polarization,
    )
    transmitted_x = n_before / n_after * direction_x
    transmitted_y = n_before / n_after * direction_y
    transverse_squared = transmitted_x**2 + transmitted_y**2
    tir = transverse_squared >= 1.0
    bundle.weight_W[tir] = 0.0
    transmitted_z = np.sqrt(np.maximum(1.0 - transverse_squared, 1e-15))
    bundle.slope_x = transmitted_x / transmitted_z
    bundle.slope_y = transmitted_y / transmitted_z
    bundle.refractive_index = n_after


def _interface_transmission(
    n_before: float,
    n_after: float,
    incidence_rad: np.ndarray,
    model: str,
    specified: float,
    polarization: str,
) -> np.ndarray:
    if model == "ideal":
        return np.ones_like(incidence_rad)
    if model == "coating":
        return np.full_like(incidence_rad, float(specified))
    if model == "fresnel":
        value = 4.0 * n_before * n_after / (n_before + n_after) ** 2
        return np.full_like(incidence_rad, value)
    if model != "fresnel_angle":
        raise ValueError(f"Unknown interface model {model!r}")
    sin_transmitted = n_before / n_after * np.sin(incidence_rad)
    tir = np.abs(sin_transmitted) >= 1.0
    cos_i = np.cos(incidence_rad)
    cos_t = np.sqrt(np.maximum(1.0 - sin_transmitted**2, 0.0))
    ts = (
        4.0
        * n_before
        * n_after
        * cos_i
        * cos_t
        / np.maximum((n_before * cos_i + n_after * cos_t) ** 2, np.finfo(float).tiny)
    )
    tp = (
        4.0
        * n_before
        * n_after
        * cos_i
        * cos_t
        / np.maximum((n_after * cos_i + n_before * cos_t) ** 2, np.finfo(float).tiny)
    )
    if polarization == "s":
        transmission = ts
    elif polarization == "p":
        transmission = tp
    elif polarization == "unpolarized":
        transmission = 0.5 * (ts + tp)
    else:
        raise ValueError("Fresnel polarization must be s, p, or unpolarized")
    transmission[tir] = 0.0
    return np.clip(transmission, 0.0, 1.0)


def _apply_optional_aperture(bundle: RayBundle, element: dict[str, Any]) -> None:
    if any(
        element.get(key) is not None
        for key in (
            "aperture_radius_m",
            "aperture_radius_x_m",
            "aperture_radius_y_m",
            "aperture_width_m",
            "aperture_height_m",
        )
    ):
        bundle.weight_W *= _aperture_transmission(bundle, element)


def _aperture_transmission(bundle: RayBundle, element: dict[str, Any]) -> np.ndarray:
    cx = float(element.get("center_x_m") or 0.0)
    cy = float(element.get("center_y_m") or 0.0)
    raw_shape = element.get("shape")
    shape = "ellipse" if raw_shape is None else str(raw_shape)
    if shape in {"circle", "ellipse", "super_gaussian"}:
        common = element.get("aperture_radius_m")
        radius_x = element.get("aperture_radius_x_m")
        radius_y = element.get("aperture_radius_y_m")
        if radius_x is None:
            radius_x = common
        if radius_y is None:
            radius_y = common
        if radius_x is None:
            radius_x = element.get("radius_x_m", math.inf)
        if radius_y is None:
            radius_y = element.get("radius_y_m", math.inf)
        rx = float(radius_x)
        ry = float(radius_y)
        coordinate = ((bundle.x_m - cx) / rx) ** 2 + ((bundle.y_m - cy) / ry) ** 2
        if shape == "super_gaussian":
            order = float(element.get("order") or 10.0)
            # optics._aperture_mask is an amplitude mask.  Rays carry power,
            # so use its squared value here.
            return np.exp(-coordinate**order)
        return (coordinate <= 1.0).astype(float)
    if shape == "rectangle":
        width = float(element["aperture_width_m"])
        height = float(element.get("aperture_height_m") or width)
        return (
            (np.abs(bundle.x_m - cx) <= width / 2.0)
            & (np.abs(bundle.y_m - cy) <= height / 2.0)
        ).astype(float)
    raise ValueError(f"Unknown aperture shape {shape!r}")


def _element_scalar(
    element: dict[str, Any],
    key: str,
    wavelength_m: float,
    default: float,
) -> float:
    configured = element.get(key, default)
    if configured is None:
        configured = default
    value = float(response_value(element, key, wavelength_m, configured))
    if key.startswith("power_") and not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must lie in [0, 1] at {wavelength_m:.6g} m")
    return value


def _inside_indices(
    bundle: RayBundle,
    grid: Grid3D,
    crystal_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ix = np.searchsorted(grid.x_edges, bundle.x_m, side="right") - 1
    iy = np.searchsorted(grid.y_edges, bundle.y_m, side="right") - 1
    in_window = (
        (ix >= 0)
        & (ix < len(grid.x))
        & (iy >= 0)
        & (iy < len(grid.y))
    )
    flat = np.zeros(len(ix), dtype=np.int64)
    flat[in_window] = iy[in_window] * len(grid.x) + ix[in_window]
    inside = in_window.copy()
    inside[in_window] &= crystal_mask.ravel()[flat[in_window]] > 0.0
    return flat, inside


def _bilinear_transverse_stencil(
    x_m: np.ndarray,
    y_m: np.ndarray,
    grid: Grid3D,
    crystal_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a conservative cell-centred bilinear stencil for ray samples.

    The ray has already passed the physical cell-mask aperture check.  Near a
    curved staircase boundary, invalid neighbour weights are removed and the
    remaining active weights are renormalized.  This preserves every ray
    segment's power while avoiding deposition outside the gain material.
    """

    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    if x.shape != y.shape:
        raise ValueError("bilinear ray coordinates must have matching shapes")

    def bracket(coordinate: np.ndarray, centers: np.ndarray):
        upper = np.searchsorted(centers, coordinate, side="right")
        lower = upper - 1
        below = upper <= 0
        above = upper >= len(centers)
        lower = np.clip(lower, 0, len(centers) - 1)
        upper = np.clip(upper, 0, len(centers) - 1)
        lower[below] = 0
        upper[below] = 0
        lower[above] = len(centers) - 1
        upper[above] = len(centers) - 1
        separation = centers[upper] - centers[lower]
        fraction = np.divide(
            coordinate - centers[lower],
            separation,
            out=np.zeros_like(coordinate),
            where=np.abs(separation) > np.finfo(float).tiny,
        )
        return lower, upper, np.clip(fraction, 0.0, 1.0)

    ix0, ix1, tx = bracket(x, grid.x)
    iy0, iy1, ty = bracket(y, grid.y)
    nx = len(grid.x)
    flat = np.column_stack(
        [
            iy0 * nx + ix0,
            iy0 * nx + ix1,
            iy1 * nx + ix0,
            iy1 * nx + ix1,
        ]
    ).astype(np.int64, copy=False)
    weights = np.column_stack(
        [
            (1.0 - tx) * (1.0 - ty),
            tx * (1.0 - ty),
            (1.0 - tx) * ty,
            tx * ty,
        ]
    )
    active = np.asarray(crystal_mask, dtype=bool).ravel()[flat]
    weights *= active
    normalization = np.sum(weights, axis=1)
    if np.any(normalization <= np.finfo(float).tiny):
        raise RuntimeError(
            "bilinear Pump deposition found no active neighbour for a ray "
            "that passed the crystal aperture"
        )
    weights /= normalization[:, None]
    return flat, weights


def _plane_intensity(bundle: RayBundle, grid: Grid3D) -> tuple[np.ndarray, float]:
    histogram, _, _ = np.histogram2d(
        bundle.y_m,
        bundle.x_m,
        bins=(grid.y_edges, grid.x_edges),
        weights=bundle.weight_W,
    )
    return histogram / grid.cell_areas, float(np.sum(histogram))


def _effective_sample_size(weights: np.ndarray) -> float:
    values = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = float(np.sum(values))
    square = float(np.sum(values**2))
    return total**2 / max(square, np.finfo(float).tiny)


__all__ = [
    "RayBundle",
    "RayEntrance",
    "RayVolumeResult",
    "build_ray_entrance",
    "propagate_ray_volume",
]
