"""Local parabolic thermal-index closure for sampled-field propagation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import crystal_transverse_mask
from .grid import Grid3D
from .field_diagnostics import (
    normalized_fit_rmse,
    stabilize_quadratic_fit_coefficients,
)


class LocalParabolicThermalLens:
    """Fit each sampled delta-n slice over one declared elliptical pupil.

    The fitted parabola replaces the sampled slice during propagation.  This
    is intentionally separate from diagnostic-only field fitting: callers can
    therefore report exactly which reduced thermal lens acted on the field.
    """

    def __init__(
        self,
        grid: Grid3D,
        material: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        self.grid = grid
        self.config = dict(config)
        self.radius_x_m = float(self.config["radius_x_m"])
        self.radius_y_m = float(self.config["radius_y_m"])
        geometry = material.get("geometry", {})
        self.center_x_m = float(
            geometry.get("center_x_m", 0.0)
            if self.config.get("center_x_m") is None
            else self.config["center_x_m"]
        )
        self.center_y_m = float(
            geometry.get("center_y_m", 0.0)
            if self.config.get("center_y_m") is None
            else self.config["center_y_m"]
        )
        self.quadratic_model = str(
            self.config.get("quadratic_model", "axis_aligned")
        )
        self.material_mask = np.asarray(
            crystal_transverse_mask(grid, material), dtype=bool
        )
        xx, yy = grid.transverse_mesh
        self.normalized_x = (xx - self.center_x_m) / self.radius_x_m
        self.normalized_y = (yy - self.center_y_m) / self.radius_y_m
        self.fit_mask = (
            self.material_mask
            & (self.normalized_x**2 + self.normalized_y**2 <= 1.0)
        )
        self._cache: dict[int, np.ndarray] = {}
        self._records: dict[int, dict[str, Any]] = {}
        self._fallback_count = 0

    def reset(self) -> None:
        self._cache.clear()
        self._records.clear()
        self._fallback_count = 0

    def fit_slice(self, delta_n_slice: np.ndarray, z_index: int) -> np.ndarray:
        cached = self._cache.get(int(z_index))
        if cached is not None:
            return cached
        delta_n = np.asarray(delta_n_slice, dtype=float)
        selected = self.fit_mask & np.isfinite(delta_n)
        point_count = int(np.count_nonzero(selected))
        if point_count < 6:
            self._fallback_count += 1
            self._records[int(z_index)] = {
                "valid": False,
                "z_index": int(z_index),
                "point_count": point_count,
                "reason": "fewer than six material cells lie inside the fit ellipse",
            }
            self._cache[int(z_index)] = delta_n
            return delta_n

        columns = [
            np.ones(point_count),
            self.normalized_x[selected],
            self.normalized_y[selected],
            self.normalized_x[selected] ** 2,
            self.normalized_y[selected] ** 2,
        ]
        if self.quadratic_model == "full_xy":
            columns.append(
                self.normalized_x[selected] * self.normalized_y[selected]
            )
        design = np.column_stack(columns)
        weights = np.sqrt(self.grid.cell_areas[selected])
        coefficients, _, rank, _ = np.linalg.lstsq(
            design * weights[:, None],
            delta_n[selected] * weights,
            rcond=None,
        )
        if rank < design.shape[1]:
            self._fallback_count += 1
            self._records[int(z_index)] = {
                "valid": False, "z_index": int(z_index),
                "point_count": point_count, "design_rank": int(rank),
                "reason": "fit pupil does not resolve every quadratic coefficient",
            }
            self._cache[int(z_index)] = delta_n
            return delta_n
        coefficients = stabilize_quadratic_fit_coefficients(
            coefficients,
            delta_n[selected],
        )
        # Avoid platform BLAS overflow warnings seen for this tiny, strongly
        # scaled design matrix; the explicit reduction is equivalent here.
        fitted_selected = np.sum(design * coefficients[None, :], axis=1)
        if self.quadratic_model == "axis_aligned":
            coefficients = np.concatenate(
                [coefficients, np.asarray([0.0], dtype=float)]
            )
        residual = delta_n[selected] - fitted_selected
        area = self.grid.cell_areas[selected]
        rmse = float(np.sqrt(np.sum(area * residual**2) / np.sum(area)))
        fitted = (
            coefficients[0]
            + coefficients[1] * self.normalized_x
            + coefficients[2] * self.normalized_y
            + coefficients[3] * self.normalized_x**2
            + coefficients[4] * self.normalized_y**2
            + coefficients[5] * self.normalized_x * self.normalized_y
        )
        fitted = np.where(self.material_mask, fitted, 0.0)
        fitted.setflags(write=False)
        self._cache[int(z_index)] = fitted
        self._records[int(z_index)] = {
            "valid": True,
            "z_index": int(z_index),
            "point_count": point_count,
            "quadratic_x_per_m2": float(coefficients[3]) / self.radius_x_m**2,
            "quadratic_y_per_m2": float(coefficients[4]) / self.radius_y_m**2,
            "quadratic_xy_per_m2": float(coefficients[5])
            / (self.radius_x_m * self.radius_y_m),
            "rmse": rmse,
            "normalized_rmse": normalized_fit_rmse(
                delta_n[selected], rmse
            ),
        }
        return fitted

    def diagnostics(self) -> dict[str, Any]:
        valid = [row for row in self._records.values() if row.get("valid")]
        curvature_x = 0.0
        curvature_y = 0.0
        curvature_xy = 0.0
        for row in valid:
            width = float(self.grid.z_widths[int(row["z_index"])])
            curvature_x += 2.0 * float(row["quadratic_x_per_m2"]) * width
            curvature_y += 2.0 * float(row["quadratic_y_per_m2"]) * width
            curvature_xy += float(row["quadratic_xy_per_m2"]) * width
        focal_x = (
            None
            if abs(curvature_x) <= np.finfo(float).eps
            else -1.0 / curvature_x
        )
        focal_y = (
            None
            if abs(curvature_y) <= np.finfo(float).eps
            else -1.0 / curvature_y
        )
        return {
            "model": "local_parabolic",
            "application": "sampled_field_propagation",
            "affects_propagation": True,
            "fit_radius_mode": "manual",
            "quadratic_model": self.quadratic_model,
            "fit_radius_x_m": self.radius_x_m,
            "fit_radius_y_m": self.radius_y_m,
            "fit_center_x_m": self.center_x_m,
            "fit_center_y_m": self.center_y_m,
            "fitted_slice_count": len(valid),
            "fallback_slice_count": int(self._fallback_count),
            "maximum_fit_normalized_rmse": (
                None
                if not valid
                else max(float(row["normalized_rmse"]) for row in valid)
            ),
            "single_pass_equivalent_focal_x_m": focal_x,
            "single_pass_equivalent_focal_y_m": focal_y,
            "single_pass_quadratic_hessian_per_m": [
                [curvature_x, curvature_xy],
                [curvature_xy, curvature_y],
            ],
            "scope": (
                "Each sampled delta-n z slice is replaced during FFT propagation "
                "by an area-weighted quadratic fit over the configured ellipse."
            ),
        }
