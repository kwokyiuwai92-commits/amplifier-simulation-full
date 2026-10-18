"""Physical gain-medium geometry on the Cartesian solver grid."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .grid import Grid3D


SUPPORTED_CRYSTAL_SHAPES = {"cuboid", "cylinder"}


def crystal_aperture_clipping_required(
    material: dict[str, Any],
    transverse_mask: np.ndarray,
) -> bool:
    """Return whether applying the sampled crystal aperture can change a field.

    A cuboid that fills the complete optical grid has an all-true mask.  In
    that common case, repeatedly multiplying large optical fields by the mask
    and integrating them before/after the multiplication is exactly a no-op.
    Keep the configured intent separate from the effective operation so finite
    and offset apertures continue to clip normally.
    """

    configured = bool(
        material.get("geometry", {}).get("clip_optical_field", True)
    )
    mask = np.asarray(transverse_mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("crystal transverse mask must be a non-empty 2D array")
    return configured and not bool(np.all(mask))


def crystal_transverse_mask(grid: Grid3D, material: dict[str, Any]) -> np.ndarray:
    """Return the physical crystal aperture as a ``(y, x)`` boolean mask."""

    cfg = material.get("geometry", {})
    shape = str(cfg.get("shape", "cuboid"))
    if shape not in SUPPORTED_CRYSTAL_SHAPES:
        raise ValueError(f"material.geometry.shape must be one of {sorted(SUPPORTED_CRYSTAL_SHAPES)}")
    center_x = float(cfg.get("center_x_m", 0.0))
    center_y = float(cfg.get("center_y_m", 0.0))
    xx, yy = grid.transverse_mesh
    grid_width = float(grid.x_edges[-1] - grid.x_edges[0])
    grid_height = float(grid.y_edges[-1] - grid.y_edges[0])

    if shape == "cuboid":
        width = grid_width if cfg.get("width_x_m") is None else float(cfg["width_x_m"])
        height = grid_height if cfg.get("height_y_m") is None else float(cfg["height_y_m"])
        mask = (np.abs(xx - center_x) <= width / 2.0) & (np.abs(yy - center_y) <= height / 2.0)
    else:
        diameter = min(grid_width, grid_height) if cfg.get("diameter_m") is None else float(cfg["diameter_m"])
        radius = diameter / 2.0
        mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2

    if not np.any(mask):
        raise ValueError("The configured crystal geometry contains no grid-cell centers")
    return mask


def crystal_volume_mask(grid: Grid3D, material: dict[str, Any]) -> np.ndarray:
    """Extrude the transverse aperture along the crystal optical axis."""

    transverse = crystal_transverse_mask(grid, material)
    return np.broadcast_to(transverse, grid.shape).copy()


def crystal_geometry_summary(grid: Grid3D, material: dict[str, Any]) -> dict[str, Any]:
    cfg = material.get("geometry", {})
    mask = crystal_volume_mask(grid, material)
    shape = str(cfg.get("shape", "cuboid"))
    represented_volume = float(grid.integrate_volume(mask.astype(float)))
    length = float(grid.z_edges[-1] - grid.z_edges[0])
    summary: dict[str, Any] = {
        "shape": shape,
        "center_x_m": float(cfg.get("center_x_m", 0.0)),
        "center_y_m": float(cfg.get("center_y_m", 0.0)),
        "active_cell_count": int(np.count_nonzero(mask)),
        "represented_volume_m3": represented_volume,
        "grid_fill_fraction": float(np.mean(mask)),
        "discretization": "cell_center_binary_voxelization",
        "cut_cell_volume_fractions_used": False,
    }
    if shape == "cuboid":
        width = float(
            grid.x_edges[-1] - grid.x_edges[0]
            if cfg.get("width_x_m") is None
            else cfg["width_x_m"]
        )
        height = float(
            grid.y_edges[-1] - grid.y_edges[0]
            if cfg.get("height_y_m") is None
            else cfg["height_y_m"]
        )
        summary["width_x_m"] = width
        summary["height_y_m"] = height
        analytic_volume = width * height * length
    else:
        diameter = float(
            min(
                grid.x_edges[-1] - grid.x_edges[0],
                grid.y_edges[-1] - grid.y_edges[0],
            )
            if cfg.get("diameter_m") is None
            else cfg["diameter_m"]
        )
        summary["diameter_m"] = diameter
        analytic_volume = math.pi * (0.5 * diameter) ** 2 * length
    summary["analytic_volume_m3"] = analytic_volume
    summary["represented_volume_relative_error"] = (
        (represented_volume - analytic_volume) / analytic_volume
    )
    return summary
