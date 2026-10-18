"""Diagnostic-only parabolic fits of sampled thermal-index and gain fields.

The fitted fields are never returned to a propagation backend.  Every signal
model continues to use the sampled three-dimensional fields, which prevents a
diagnostic fit from replacing or duplicating thermal lensing and gain guiding.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .grid import Grid3D
from .validation import require_exact_integer


FIT_RADIUS_MODES = {"automatic", "manual", "seed_entrance"}


def stabilize_quadratic_fit_coefficients(
    coefficients: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    """Remove only least-squares coefficients below floating-point resolution.

    The quadratic design uses normalized coordinates, so every coefficient has
    the same units as the fitted field.  A constant slice can otherwise acquire
    round-off-sized quadratic terms which later look like a finite focal length
    or an arbitrary x-y coupling ratio.
    """

    result = np.asarray(coefficients, dtype=float).copy()
    values = np.asarray(samples, dtype=float)
    magnitude = max(
        float(np.max(np.abs(values))) if values.size else 0.0,
        np.finfo(float).tiny,
    )
    threshold = 512.0 * np.finfo(float).eps * magnitude
    result[np.abs(result) <= threshold] = 0.0
    return result


def normalized_fit_rmse(samples: np.ndarray, rmse: float) -> float:
    """Normalize a fit error without dividing a constant field by tiny.

    Peak-to-peak variation remains the physical scale whenever it is resolved.
    If both the variation and residual are below round-off relative to the field
    magnitude, the fit is numerically exact and its normalized error is zero.
    """

    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        return 0.0
    span = float(np.ptp(values))
    magnitude = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    roundoff = 512.0 * np.finfo(float).eps * magnitude
    error = max(float(rmse), 0.0)
    if span <= roundoff and error <= roundoff:
        return 0.0
    return error / max(span, roundoff, np.finfo(float).tiny)


def _fit_center(config: dict[str, Any]) -> tuple[float, float]:
    x_value = config.get("center_x_m")
    y_value = config.get("center_y_m")
    return (
        0.0 if x_value is None else float(x_value),
        0.0 if y_value is None else float(y_value),
    )


def validate_field_diagnostics(config: dict[str, Any]) -> None:
    """Validate the optional diagnostic-fit configuration."""

    if not isinstance(config, dict):
        raise ValueError("material.field_diagnostics must be an object")
    if not isinstance(config.get("enabled", False), bool):
        raise ValueError("material.field_diagnostics.enabled must be true or false")
    mode = str(config.get("fit_radius_mode", "automatic"))
    if mode not in FIT_RADIUS_MODES:
        raise ValueError(
            "material.field_diagnostics.fit_radius_mode must be "
            "automatic, manual, or seed_entrance"
        )
    if mode == "manual":
        for axis in "xy":
            value = config.get(f"manual_radius_{axis}_m")
            if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"material.field_diagnostics.manual_radius_{axis}_m must be positive"
                )
    scale = float(config.get("beam_radius_scale", 1.0))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("material.field_diagnostics.beam_radius_scale must be positive")
    require_exact_integer(
        config.get("slice_count", 5),
        name="material.field_diagnostics.slice_count",
        minimum=1,
    )
    require_exact_integer(
        config.get("zernike_max_radial_order", 4),
        name="material.field_diagnostics.zernike_max_radial_order",
        minimum=4,
        maximum=12,
    )
    require_exact_integer(
        config.get("radial_bin_count", 48),
        name="material.field_diagnostics.radial_bin_count",
        minimum=8,
    )
    for axis in "xy":
        center = config.get(f"center_{axis}_m")
        if center is not None and not math.isfinite(float(center)):
            raise ValueError(
                f"material.field_diagnostics.center_{axis}_m must be finite when supplied"
            )


def _resolved_fit_radii(
    config: dict[str, Any],
    grid: Grid3D,
    transverse_mask: np.ndarray,
    seed_entrance_radius_m: tuple[float | None, float | None] | None,
) -> tuple[float, float, str, list[str]]:
    mode = str(config.get("fit_radius_mode", "automatic"))
    warnings: list[str] = []
    xx, yy = grid.transverse_mesh
    if not np.any(transverse_mask):
        return float(grid.x_edges[-1] - grid.x_edges[0]) / 2.0, float(
            grid.y_edges[-1] - grid.y_edges[0]
        ) / 2.0, "automatic", warnings
    center_x, center_y = _fit_center(config)
    automatic_x = float(np.max(np.abs(xx[transverse_mask] - center_x)))
    automatic_y = float(np.max(np.abs(yy[transverse_mask] - center_y)))
    if mode == "manual":
        return (
            float(config["manual_radius_x_m"]),
            float(config["manual_radius_y_m"]),
            mode,
            warnings,
        )
    if mode == "seed_entrance":
        radii = seed_entrance_radius_m or (None, None)
        if radii[0] is not None and radii[1] is not None:
            scale = float(config.get("beam_radius_scale", 1.0))
            return float(radii[0]) * scale, float(radii[1]) * scale, mode, warnings
        warnings.append(
            "Seed entrance radii were unavailable; field diagnostics used the automatic active aperture."
        )
        mode = "automatic_fallback"
    return automatic_x, automatic_y, mode, warnings


def _weighted_quadratic_fit(
    values: np.ndarray,
    grid: Grid3D,
    mask: np.ndarray,
    radius_x_m: float,
    radius_y_m: float,
    center_x_m: float,
    center_y_m: float,
) -> dict[str, Any]:
    plane = np.asarray(values, dtype=float)
    active = np.asarray(mask, dtype=bool)
    if plane.shape != grid.cell_areas.shape or active.shape != plane.shape:
        raise ValueError("quadratic-fit values and mask must match the grid transverse shape")
    xx, yy = grid.transverse_mesh
    ellipse = (
        ((xx - center_x_m) / max(radius_x_m, np.finfo(float).tiny)) ** 2
        + ((yy - center_y_m) / max(radius_y_m, np.finfo(float).tiny)) ** 2
        <= 1.0
    )
    selected = active & ellipse & np.isfinite(plane)
    point_count = int(np.count_nonzero(selected))
    if point_count < 6:
        return {
            "valid": False,
            "point_count": point_count,
            "reason": "fewer than six active cells lie inside the fit ellipse",
        }
    x = xx[selected] - center_x_m
    y = yy[selected] - center_y_m
    normalized_x = x / radius_x_m
    normalized_y = y / radius_y_m
    design = np.column_stack(
        [
            np.ones(point_count),
            normalized_x,
            normalized_y,
            normalized_x**2,
            normalized_y**2,
            normalized_x * normalized_y,
        ]
    )
    weights = np.sqrt(grid.cell_areas[selected])
    weighted_design = design * weights[:, None]
    weighted_values = plane[selected] * weights
    coefficients, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)
    if rank < design.shape[1]:
        return {
            "valid": False, "point_count": point_count, "design_rank": int(rank),
            "reason": "fit pupil does not resolve every quadratic coefficient",
        }
    coefficients = stabilize_quadratic_fit_coefficients(
        coefficients,
        plane[selected],
    )
    # Elementwise reduction avoids a macOS Accelerate/NumPy matmul warning
    # observed for very small weighted least-squares coefficients.
    fitted = np.sum(design * coefficients[None, :], axis=1)
    residual = plane[selected] - fitted
    area_weights = grid.cell_areas[selected]
    rmse = float(np.sqrt(np.sum(area_weights * residual**2) / np.sum(area_weights)))
    normalized_rmse = normalized_fit_rmse(plane[selected], rmse)
    normalized_coefficients = list(map(float, coefficients))
    c0 = normalized_coefficients[0]
    cx = normalized_coefficients[1] / radius_x_m
    cy = normalized_coefficients[2] / radius_y_m
    cxx = normalized_coefficients[3] / radius_x_m**2
    cyy = normalized_coefficients[4] / radius_y_m**2
    cxy = normalized_coefficients[5] / (radius_x_m * radius_y_m)
    hessian = np.array([[2.0 * cxx, cxy], [cxy, 2.0 * cyy]], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    principal_angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    return {
        "valid": True,
        "point_count": point_count,
        "constant": c0,
        "linear_x_per_m": cx,
        "linear_y_per_m": cy,
        "quadratic_x_per_m2": cxx,
        "quadratic_y_per_m2": cyy,
        "quadratic_xy_per_m2": cxy,
        "hessian_eigenvalues_per_m2": eigenvalues.tolist(),
        "principal_axis_angle_deg": principal_angle,
        "rmse": rmse,
        "normalized_rmse": normalized_rmse,
        "fit_value_min": float(np.min(plane[selected])),
        "fit_value_max": float(np.max(plane[selected])),
    }


def _zernike_radial(n: int, m: int, radius: np.ndarray) -> np.ndarray:
    """Return the standard radial polynomial R_n^m on the unit disk."""

    order = abs(int(m))
    if n < 0 or order > n or (n - order) % 2:
        return np.zeros_like(radius, dtype=float)
    result = np.zeros_like(radius, dtype=float)
    for k in range((n - order) // 2 + 1):
        coefficient = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + order) // 2 - k)
                * math.factorial((n - order) // 2 - k)
            )
        )
        result += coefficient * radius ** (n - 2 * k)
    return result


def _real_zernike_modes(
    radius: np.ndarray,
    angle: np.ndarray,
    maximum_radial_order: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build orthonormal real Zernike modes through the requested order."""

    columns: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for n in range(maximum_radial_order + 1):
        for m in range(0, n + 1):
            if (n - m) % 2:
                continue
            radial = _zernike_radial(n, m, radius)
            if m == 0:
                columns.append(math.sqrt(n + 1.0) * radial)
                metadata.append(
                    {"n": n, "m": 0, "orientation": "axisymmetric"}
                )
            else:
                normalization = math.sqrt(2.0 * (n + 1.0))
                columns.append(normalization * radial * np.cos(m * angle))
                metadata.append({"n": n, "m": m, "orientation": "cos"})
                columns.append(normalization * radial * np.sin(m * angle))
                metadata.append({"n": n, "m": -m, "orientation": "sin"})
    return np.column_stack(columns), metadata


def zernike_wavefront_fit(
    values_m: np.ndarray,
    grid: Grid3D,
    mask: np.ndarray,
    radius_x_m: float,
    radius_y_m: float,
    center_x_m: float = 0.0,
    center_y_m: float = 0.0,
    maximum_radial_order: int = 4,
    wavelength_m: float | None = None,
) -> dict[str, Any]:
    """Area-weighted real-Zernike decomposition of an OPD map.

    The fit ellipse is mapped to a unit disk.  Coefficients use orthonormal
    real Zernikes, so the n=4,m=0 coefficient is the primary-spherical RMS
    contribution for a continuously sampled full pupil.  Discrete residual
    and conditioning are retained because clipped pupils break exact
    orthogonality.
    """

    plane = np.asarray(values_m, dtype=float)
    active = np.asarray(mask, dtype=bool)
    if plane.shape != grid.cell_areas.shape or active.shape != plane.shape:
        raise ValueError("Zernike values and mask must match the transverse grid")
    if maximum_radial_order < 0:
        raise ValueError("maximum Zernike radial order cannot be negative")
    xx, yy = grid.transverse_mesh
    normalized_x = (xx - center_x_m) / radius_x_m
    normalized_y = (yy - center_y_m) / radius_y_m
    rho = np.sqrt(normalized_x**2 + normalized_y**2)
    selected = active & (rho <= 1.0) & np.isfinite(plane)
    point_count = int(np.count_nonzero(selected))
    mode_count = (maximum_radial_order + 1) * (maximum_radial_order + 2) // 2
    if point_count < mode_count:
        return {
            "valid": False,
            "point_count": point_count,
            "mode_count": mode_count,
            "reason": "fewer active pupil cells than requested Zernike modes",
        }
    selected_rho = rho[selected]
    selected_angle = np.arctan2(normalized_y[selected], normalized_x[selected])
    design, metadata = _real_zernike_modes(
        selected_rho, selected_angle, maximum_radial_order
    )
    weights = np.sqrt(grid.cell_areas[selected])
    weighted_design = design * weights[:, None]
    weighted_values = plane[selected] * weights
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_design, weighted_values, rcond=None
    )
    if rank < mode_count:
        return {
            "valid": False, "point_count": point_count, "mode_count": mode_count,
            "design_rank": int(rank),
            "reason": "fit pupil does not resolve every requested Zernike mode",
        }
    fitted = np.sum(design * coefficients[None, :], axis=1)
    residual = plane[selected] - fitted
    area = grid.cell_areas[selected]
    residual_rms = float(np.sqrt(np.sum(area * residual**2) / np.sum(area)))
    coefficient_rows: list[dict[str, Any]] = []
    primary_spherical: float | None = None
    for meta, coefficient in zip(metadata, coefficients, strict=True):
        row = {**meta, "coefficient_m": float(coefficient)}
        if wavelength_m is not None:
            row["coefficient_waves"] = float(coefficient) / float(wavelength_m)
        coefficient_rows.append(row)
        if meta["n"] == 4 and meta["m"] == 0:
            primary_spherical = float(coefficient)
    return {
        "valid": True,
        "method": "area_weighted_orthonormal_real_zernike_least_squares",
        "pupil_mapping": "fit ellipse mapped to unit disk",
        "point_count": point_count,
        "mode_count": len(metadata),
        "maximum_radial_order": int(maximum_radial_order),
        "coefficients": coefficient_rows,
        "primary_spherical_n4_m0_coefficient_m": primary_spherical,
        "primary_spherical_n4_m0_coefficient_waves": (
            None
            if primary_spherical is None or wavelength_m is None
            else primary_spherical / float(wavelength_m)
        ),
        "residual_rms_m": residual_rms,
        "residual_rms_waves": (
            None if wavelength_m is None else residual_rms / float(wavelength_m)
        ),
        "weighted_design_condition_number": float(np.linalg.cond(weighted_design)),
        "normalization": (
            "Z_n^0=sqrt(n+1)R_n^0; sine/cosine modes use sqrt(2(n+1))"
        ),
    }


def _quadratic_fit_map(
    fit: dict[str, Any],
    grid: Grid3D,
    center_x_m: float,
    center_y_m: float,
) -> np.ndarray | None:
    if not fit.get("valid", False):
        return None
    xx, yy = grid.transverse_mesh
    x = xx - center_x_m
    y = yy - center_y_m
    return (
        float(fit["constant"])
        + float(fit["linear_x_per_m"]) * x
        + float(fit["linear_y_per_m"]) * y
        + float(fit["quadratic_x_per_m2"]) * x**2
        + float(fit["quadratic_y_per_m2"]) * y**2
        + float(fit["quadratic_xy_per_m2"]) * x * y
    )


def _nonparabolic_profiles(
    optical_path_m: np.ndarray,
    fitted_path_m: np.ndarray,
    grid: Grid3D,
    mask: np.ndarray,
    radius_x_m: float,
    radius_y_m: float,
    center_x_m: float,
    center_y_m: float,
    radial_bin_count: int,
    wavelength_m: float | None,
) -> dict[str, Any]:
    """Return compact radial and principal-axis OPD/residual profiles."""

    xx, yy = grid.transverse_mesh
    nx = (xx - center_x_m) / radius_x_m
    ny = (yy - center_y_m) / radius_y_m
    rho = np.sqrt(nx**2 + ny**2)
    selected = np.asarray(mask, dtype=bool) & (rho <= 1.0)
    residual = np.asarray(optical_path_m, dtype=float) - fitted_path_m
    edges = np.linspace(0.0, 1.0, radial_bin_count + 1)
    radial_rows: list[dict[str, Any]] = []
    for index in range(radial_bin_count):
        local = selected & (rho >= edges[index]) & (rho < edges[index + 1])
        if not np.any(local):
            continue
        weights = grid.cell_areas[local]
        weight_sum = float(np.sum(weights))
        mean_path = float(np.sum(weights * optical_path_m[local]) / weight_sum)
        mean_fit = float(np.sum(weights * fitted_path_m[local]) / weight_sum)
        mean_residual = float(np.sum(weights * residual[local]) / weight_sum)
        row = {
            "normalized_radius": 0.5 * (edges[index] + edges[index + 1]),
            "equivalent_radius_m": 0.5
            * (edges[index] + edges[index + 1])
            * math.sqrt(radius_x_m * radius_y_m),
            "optical_path_m": mean_path,
            "parabolic_fit_m": mean_fit,
            "nonparabolic_residual_m": mean_residual,
        }
        if wavelength_m is not None:
            row["nonparabolic_residual_waves"] = mean_residual / wavelength_m
        radial_rows.append(row)

    x_index = int(np.argmin(np.abs(grid.y - center_y_m)))
    y_index = int(np.argmin(np.abs(grid.x - center_x_m)))
    x_selected = selected[x_index]
    y_selected = selected[:, y_index]

    def axis_rows(
        coordinate: np.ndarray,
        path: np.ndarray,
        fit: np.ndarray,
        remainder: np.ndarray,
        valid: np.ndarray,
    ) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for position, value, fitted, difference in zip(
            coordinate[valid], path[valid], fit[valid], remainder[valid], strict=True
        ):
            row = {
                "position_m": float(position),
                "optical_path_m": float(value),
                "parabolic_fit_m": float(fitted),
                "nonparabolic_residual_m": float(difference),
            }
            if wavelength_m is not None:
                row["nonparabolic_residual_waves"] = float(difference) / wavelength_m
            rows.append(row)
        return rows

    return {
        "radial_azimuthal_average": radial_rows,
        "x_axis_cut": axis_rows(
            grid.x - center_x_m,
            optical_path_m[x_index],
            fitted_path_m[x_index],
            residual[x_index],
            x_selected,
        ),
        "y_axis_cut": axis_rows(
            grid.y - center_y_m,
            optical_path_m[:, y_index],
            fitted_path_m[:, y_index],
            residual[:, y_index],
            y_selected,
        ),
        "quadratic_terms_removed": (
            "piston, x/y tilt, x2/y2 defocus-astigmatism, and xy astigmatism"
        ),
    }


def _selected_slice_indices(mask: np.ndarray, requested: int) -> np.ndarray:
    active = np.flatnonzero(np.any(mask, axis=(1, 2)))
    if active.size == 0:
        return np.array([], dtype=int)
    count = min(int(requested), int(active.size))
    positions = np.linspace(0, active.size - 1, count)
    return np.unique(active[np.rint(positions).astype(int)])


def parabolic_field_diagnostics(
    delta_n: np.ndarray,
    gain_m_inv: np.ndarray,
    grid: Grid3D,
    crystal_mask: np.ndarray,
    config: dict[str, Any],
    seed_entrance_radius_m: tuple[float | None, float | None] | None = None,
    wavelength_m: float | None = None,
) -> dict[str, Any]:
    """Fit thermal index and gain without feeding the fit back into propagation."""

    validate_field_diagnostics(config)
    enabled = bool(config.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "application": "diagnostic_only",
            "affects_propagation": False,
        }
    mask = np.asarray(crystal_mask, dtype=bool)
    dn = np.asarray(delta_n, dtype=float)
    gain = np.asarray(gain_m_inv, dtype=float)
    if dn.shape != grid.shape or gain.shape != grid.shape or mask.shape != grid.shape:
        raise ValueError("field-diagnostic arrays must match the physical grid")
    transverse_mask = np.any(mask, axis=0)
    center_x, center_y = _fit_center(config)
    radius_x, radius_y, resolved_mode, warnings = _resolved_fit_radii(
        config, grid, transverse_mask, seed_entrance_radius_m
    )
    slice_indices = _selected_slice_indices(mask, int(config.get("slice_count", 5)))
    thermal_slices: list[dict[str, Any]] = []
    gain_slices: list[dict[str, Any]] = []
    for index in slice_indices:
        local_mask = mask[index]
        thermal_fit = _weighted_quadratic_fit(
            dn[index], grid, local_mask, radius_x, radius_y, center_x, center_y
        )
        thermal_fit.update({"z_index": int(index), "z_m": float(grid.z[index])})
        thermal_slices.append(thermal_fit)
        gain_fit = _weighted_quadratic_fit(
            gain[index], grid, local_mask, radius_x, radius_y, center_x, center_y
        )
        gain_fit.update({"z_index": int(index), "z_m": float(grid.z[index])})
        gain_slices.append(gain_fit)

    optical_path = np.sum(dn * mask * grid.z_widths[:, None, None], axis=0)
    gain_exponent = np.sum(gain * mask * grid.z_widths[:, None, None], axis=0)
    optical_path_fit = _weighted_quadratic_fit(
        optical_path, grid, transverse_mask, radius_x, radius_y, center_x, center_y
    )
    gain_exponent_fit = _weighted_quadratic_fit(
        gain_exponent, grid, transverse_mask, radius_x, radius_y, center_x, center_y
    )
    zernike_fit = zernike_wavefront_fit(
        optical_path,
        grid,
        transverse_mask,
        radius_x,
        radius_y,
        center_x,
        center_y,
        int(config.get("zernike_max_radial_order", 4)),
        wavelength_m,
    )
    fitted_optical_path = _quadratic_fit_map(
        optical_path_fit, grid, center_x, center_y
    )
    nonparabolic_profiles = (
        None
        if fitted_optical_path is None
        else _nonparabolic_profiles(
            optical_path,
            fitted_optical_path,
            grid,
            transverse_mask,
            radius_x,
            radius_y,
            center_x,
            center_y,
            int(config.get("radial_bin_count", 48)),
            wavelength_m,
        )
    )
    focal_x: float | None = None
    focal_y: float | None = None
    principal_focal: list[float | None] = [None, None]
    if optical_path_fit.get("valid", False):
        curvature_x = 2.0 * float(optical_path_fit["quadratic_x_per_m2"])
        curvature_y = 2.0 * float(optical_path_fit["quadratic_y_per_m2"])
        if abs(curvature_x) > np.finfo(float).eps:
            focal_x = -1.0 / curvature_x
        if abs(curvature_y) > np.finfo(float).eps:
            focal_y = -1.0 / curvature_y
        principal_focal = [
            None if abs(value) <= np.finfo(float).eps else -1.0 / float(value)
            for value in optical_path_fit["hessian_eigenvalues_per_m2"]
        ]
    return {
        "enabled": True,
        "application": "diagnostic_only",
        "affects_propagation": False,
        "fit_radius_mode": str(config.get("fit_radius_mode", "automatic")),
        "resolved_fit_radius_mode": resolved_mode,
        "fit_radius_x_m": radius_x,
        "fit_radius_y_m": radius_y,
        "fit_center_x_m": center_x,
        "fit_center_y_m": center_y,
        "slice_count": int(len(slice_indices)),
        "thermal_delta_n_slices": thermal_slices,
        "gain_m_inv_slices": gain_slices,
        "integrated_optical_path_fit": optical_path_fit,
        "integrated_optical_path_zernike": zernike_fit,
        "optical_path_integration_axis": "grid_z",
        "optical_path_integrand": "delta_n_dz",
        "oblique_ray_path_correction_applied": False,
        "nonparabolic_optical_path_profiles": nonparabolic_profiles,
        "integrated_gain_exponent_fit": gain_exponent_fit,
        "estimated_thin_lens_focal_x_m": focal_x,
        "estimated_thin_lens_focal_y_m": focal_y,
        "estimated_principal_focal_lengths_m": principal_focal,
        "principal_axis_angle_deg": optical_path_fit.get("principal_axis_angle_deg"),
        "warnings": warnings,
        "scope": (
            "Weighted parabolic diagnostic of sampled delta-n and gain. "
            "Integrated optical path here means the axial grid integral "
            "sum(delta_n*dz), not a traced oblique-ray line integral. "
            "The full fields, never the fit, remain active in propagation. "
            "End-face deformation is unavailable without a structural solver."
        ),
    }
