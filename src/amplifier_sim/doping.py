"""Geometry-aware three-dimensional dopant-density fields."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .geometry import crystal_volume_mask
from .grid import Grid3D


EXPONENTIAL_DIRECTIONS = {
    "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "radial_outer"
}
LINEAR_DIRECTIONS = {"x", "y", "z"}
SYMMETRIC_DIRECTIONS = {"x", "y", "z", "radial_xy"}


def validate_doping_profile(material: dict[str, Any]) -> None:
    cfg = material.get("doping_profile", {})
    if not isinstance(cfg, dict):
        raise ValueError("material.doping_profile must be an object")
    if not isinstance(cfg.get("enabled", False), bool):
        raise ValueError("material.doping_profile.enabled must be true or false")
    normalization = str(cfg.get("normalization", "volume_average"))
    if normalization not in {"volume_average", "peak", "none"}:
        raise ValueError(
            "material.doping_profile.normalization must be volume_average, peak, or none"
        )
    background = float(cfg.get("background_fraction", 1.0))
    minimum = float(cfg.get("minimum_raw_fraction", 0.0))
    if not math.isfinite(background) or background < 0.0:
        raise ValueError("material.doping_profile.background_fraction must be finite and non-negative")
    if not math.isfinite(minimum) or minimum < 0.0:
        raise ValueError("material.doping_profile.minimum_raw_fraction must be finite and non-negative")
    components = cfg.get("components", [])
    endcaps = cfg.get("endcaps", {})
    if not isinstance(endcaps, dict):
        raise ValueError("material.doping_profile.endcaps must be an object")
    if not isinstance(endcaps.get("enabled", False), bool):
        raise ValueError("material.doping_profile.endcaps.enabled must be true or false")
    for key in ("front_length_m", "back_length_m"):
        value = float(endcaps.get(key, 0.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"material.doping_profile.endcaps.{key} must be non-negative")
    endcap_density = float(endcaps.get("relative_density", 0.0))
    if not math.isfinite(endcap_density) or endcap_density < 0.0:
        raise ValueError(
            "material.doping_profile.endcaps.relative_density must be finite and non-negative"
        )
    geometry_shape = str(material.get("geometry", {}).get("shape", "cuboid"))
    if not isinstance(components, list):
        raise ValueError("material.doping_profile.components must be a list")
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            raise ValueError(f"material.doping_profile.components[{index}] must be an object")
        kind = str(component.get("type", "surface_exponential"))
        direction = str(component.get("direction", "z_min"))
        amplitude = float(component.get("amplitude", 0.0))
        if not math.isfinite(amplitude):
            raise ValueError(
                f"material.doping_profile.components[{index}].amplitude must be finite"
            )
        if kind == "surface_exponential":
            if direction not in EXPONENTIAL_DIRECTIONS:
                raise ValueError(
                    f"material.doping_profile.components[{index}].direction is unsupported"
                )
            if direction == "radial_outer" and geometry_shape != "cylinder":
                raise ValueError(
                    f"material.doping_profile.components[{index}].direction "
                    "radial_outer requires material.geometry.shape=cylinder"
                )
            decay = float(component.get("decay_length_m", 0.0))
            if not math.isfinite(decay) or decay <= 0.0:
                raise ValueError(
                    f"material.doping_profile.components[{index}].decay_length_m must be positive"
                )
        elif kind == "axis_linear":
            if direction not in LINEAR_DIRECTIONS:
                raise ValueError(
                    f"material.doping_profile.components[{index}].direction must be x, y, or z"
                )
        elif kind == "axis_symmetric":
            if direction not in SYMMETRIC_DIRECTIONS:
                raise ValueError(
                    f"material.doping_profile.components[{index}].direction must be "
                    "x, y, z, or radial_xy"
                )
            if direction == "radial_xy" and geometry_shape != "cylinder":
                raise ValueError(
                    f"material.doping_profile.components[{index}].direction "
                    "radial_xy requires material.geometry.shape=cylinder"
                )
            center_value = float(component.get("center_relative_density", 1.0))
            surface_value = float(component.get("surface_relative_density", 1.0))
            transition = float(component.get("transition_length_m", 0.0))
            if not math.isfinite(center_value) or center_value < 0.0:
                raise ValueError(
                    f"material.doping_profile.components[{index}].center_relative_density "
                    "must be finite and non-negative"
                )
            if not math.isfinite(surface_value) or surface_value < 0.0:
                raise ValueError(
                    f"material.doping_profile.components[{index}].surface_relative_density "
                    "must be finite and non-negative"
                )
            if center_value == 0.0 and surface_value == 0.0:
                raise ValueError(
                    f"material.doping_profile.components[{index}] cannot have both "
                    "center and surface relative density equal to zero"
                )
            if not math.isfinite(transition) or transition <= 0.0:
                raise ValueError(
                    f"material.doping_profile.components[{index}].transition_length_m "
                    "must be positive"
                )
        elif kind in {"axis_symmetric_linear", "axis_monotonic_linear"}:
            allowed = SYMMETRIC_DIRECTIONS if kind == "axis_symmetric_linear" else LINEAR_DIRECTIONS
            if direction not in allowed:
                readable = "x, y, z, or radial_xy" if kind == "axis_symmetric_linear" else "x, y, or z"
                raise ValueError(
                    f"material.doping_profile.components[{index}].direction must be {readable}"
                )
            if direction == "radial_xy" and geometry_shape != "cylinder":
                raise ValueError(
                    f"material.doping_profile.components[{index}].direction "
                    "radial_xy requires material.geometry.shape=cylinder"
                )
            relative_change = float(component.get("relative_change", 0.0))
            if not math.isfinite(relative_change) or relative_change < -1.0:
                raise ValueError(
                    f"material.doping_profile.components[{index}].relative_change "
                    "must be finite and no smaller than -1"
                )
        elif kind == "axis_segment_linear":
            if direction not in LINEAR_DIRECTIONS:
                raise ValueError(
                    f"material.doping_profile.components[{index}].direction "
                    "must be x, y, or z"
                )
            start = float(component.get("start_fraction", 0.0))
            end = float(component.get("end_fraction", 1.0))
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0.0
                or end > 1.0
                or start >= end
            ):
                raise ValueError(
                    f"material.doping_profile.components[{index}] requires "
                    "0 <= start_fraction < end_fraction <= 1"
                )
            for key in (
                "before_relative_density",
                "start_relative_density",
                "end_relative_density",
                "after_relative_density",
            ):
                value = float(component.get(key, 1.0))
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"material.doping_profile.components[{index}].{key} "
                        "must be finite and non-negative"
                    )
        else:
            raise ValueError(
                f"material.doping_profile.components[{index}].type must be "
                "surface_exponential, axis_linear, axis_symmetric, "
                "axis_symmetric_linear, axis_monotonic_linear, or "
                "axis_segment_linear"
            )


def _geometry_bounds(grid: Grid3D, material: dict[str, Any]) -> dict[str, float]:
    geometry = material.get("geometry", {})
    center_x = float(geometry.get("center_x_m", 0.0))
    center_y = float(geometry.get("center_y_m", 0.0))
    if str(geometry.get("shape", "cuboid")) == "cylinder":
        diameter = float(
            min(
                grid.x_edges[-1] - grid.x_edges[0],
                grid.y_edges[-1] - grid.y_edges[0],
            )
            if geometry.get("diameter_m") is None
            else geometry["diameter_m"]
        )
        half_x = half_y = diameter / 2.0
    else:
        half_x = float(
            grid.x_edges[-1] - grid.x_edges[0]
            if geometry.get("width_x_m") is None
            else geometry["width_x_m"]
        ) / 2.0
        half_y = float(
            grid.y_edges[-1] - grid.y_edges[0]
            if geometry.get("height_y_m") is None
            else geometry["height_y_m"]
        ) / 2.0
    return {
        "x_min": center_x - half_x,
        "x_max": center_x + half_x,
        "y_min": center_y - half_y,
        "y_max": center_y + half_y,
        "z_min": float(grid.z_edges[0]),
        "z_max": float(grid.z_edges[-1]),
        "center_x": center_x,
        "center_y": center_y,
        "radius": min(half_x, half_y),
    }


def _coordinates(grid: Grid3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zz, yy, xx = np.meshgrid(grid.z, grid.y, grid.x, indexing="ij")
    return xx, yy, zz


def _surface_depth(
    direction: str,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    bounds: dict[str, float],
    shape: str,
) -> np.ndarray:
    if direction == "x_min":
        return xx - bounds["x_min"]
    if direction == "x_max":
        return bounds["x_max"] - xx
    if direction == "y_min":
        return yy - bounds["y_min"]
    if direction == "y_max":
        return bounds["y_max"] - yy
    if direction == "z_min":
        return zz - bounds["z_min"]
    if direction == "z_max":
        return bounds["z_max"] - zz
    if direction == "radial_outer":
        if shape != "cylinder":
            raise ValueError(
                "doping direction radial_outer requires material.geometry.shape=cylinder"
            )
        radius = np.sqrt((xx - bounds["center_x"]) ** 2 + (yy - bounds["center_y"]) ** 2)
        return bounds["radius"] - radius
    raise ValueError(f"Unsupported doping surface direction {direction!r}")


def _normalized_axis(
    direction: str,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    bounds: dict[str, float],
) -> np.ndarray:
    if direction == "x":
        return 2.0 * (xx - 0.5 * (bounds["x_min"] + bounds["x_max"])) / (
            bounds["x_max"] - bounds["x_min"]
        )
    if direction == "y":
        return 2.0 * (yy - 0.5 * (bounds["y_min"] + bounds["y_max"])) / (
            bounds["y_max"] - bounds["y_min"]
        )
    if direction == "z":
        return 2.0 * (zz - 0.5 * (bounds["z_min"] + bounds["z_max"])) / (
            bounds["z_max"] - bounds["z_min"]
        )
    raise ValueError(f"Unsupported doping axis {direction!r}")


def _symmetric_center_to_surface_factor(
    component: dict[str, Any],
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    bounds: dict[str, float],
) -> np.ndarray:
    """Return a positive profile with exact centre and surface endpoint values.

    ``axis_symmetric`` is deliberately multiplicative.  Independent x/y/z
    entries therefore form a genuinely three-dimensional separable profile
    rather than three additive centre planes.  Exponential interpolation is
    normalized on the finite crystal half-width, so the configured endpoint
    values retain an unambiguous meaning for every transition length.
    """

    direction = str(component.get("direction", "x"))
    if direction == "x":
        distance = np.abs(xx - bounds["center_x"])
        half_extent = 0.5 * (bounds["x_max"] - bounds["x_min"])
    elif direction == "y":
        distance = np.abs(yy - bounds["center_y"])
        half_extent = 0.5 * (bounds["y_max"] - bounds["y_min"])
    elif direction == "z":
        center_z = 0.5 * (bounds["z_min"] + bounds["z_max"])
        distance = np.abs(zz - center_z)
        half_extent = 0.5 * (bounds["z_max"] - bounds["z_min"])
    elif direction == "radial_xy":
        distance = np.sqrt(
            (xx - bounds["center_x"]) ** 2 + (yy - bounds["center_y"]) ** 2
        )
        half_extent = bounds["radius"]
    else:  # Validation should make this unreachable.
        raise ValueError(f"Unsupported symmetric doping direction {direction!r}")

    transition = float(component["transition_length_m"])
    center_value = float(component.get("center_relative_density", 1.0))
    surface_value = float(component.get("surface_relative_density", 1.0))
    edge_exponential = math.exp(-half_extent / transition)
    denominator = 1.0 - edge_exponential
    if denominator <= np.finfo(float).eps:
        # For transition lengths far larger than the crystal, the normalized
        # exponential tends to a linear centre-to-surface interpolation.
        center_weight = 1.0 - np.minimum(distance / half_extent, 1.0)
    else:
        center_weight = (
            np.exp(-np.minimum(distance, half_extent) / transition)
            - edge_exponential
        ) / denominator
    return surface_value + (center_value - surface_value) * center_weight


def _simple_relative_factor(
    component: dict[str, Any],
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    bounds: dict[str, float],
) -> np.ndarray:
    """Build the simple UI profile from one signed, dimensionless change.

    Symmetric: ``relative_change`` is (centre / surface - 1).
    Monotonic: it is (positive-coordinate face / negative-coordinate face - 1).
    Global volume-average or peak normalization does not change these ratios.
    """

    kind = str(component["type"])
    direction = str(component["direction"])
    relative_change = float(component.get("relative_change", 0.0))
    if kind == "axis_symmetric_linear":
        if direction == "x":
            distance = np.abs(xx - bounds["center_x"])
            half_extent = 0.5 * (bounds["x_max"] - bounds["x_min"])
        elif direction == "y":
            distance = np.abs(yy - bounds["center_y"])
            half_extent = 0.5 * (bounds["y_max"] - bounds["y_min"])
        elif direction == "z":
            center_z = 0.5 * (bounds["z_min"] + bounds["z_max"])
            distance = np.abs(zz - center_z)
            half_extent = 0.5 * (bounds["z_max"] - bounds["z_min"])
        elif direction == "radial_xy":
            distance = np.sqrt(
                (xx - bounds["center_x"]) ** 2
                + (yy - bounds["center_y"]) ** 2
            )
            half_extent = bounds["radius"]
        else:  # Validation should make this unreachable.
            raise ValueError(f"Unsupported symmetric doping direction {direction!r}")
        center_weight = 1.0 - np.minimum(distance / half_extent, 1.0)
        return 1.0 + relative_change * center_weight

    normalized = 0.5 * (
        _normalized_axis(direction, xx, yy, zz, bounds) + 1.0
    )
    return 1.0 + relative_change * np.clip(normalized, 0.0, 1.0)


def _segment_linear_factor(
    component: dict[str, Any],
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    bounds: dict[str, float],
) -> np.ndarray:
    """Return a profile with constant regions around one linear segment.

    Fractions measure distance from the negative-coordinate material face.
    The values immediately before and at the start may differ, allowing a
    bonded interface or other intentional concentration step.
    """

    direction = str(component["direction"])
    position = np.clip(
        0.5 * (_normalized_axis(direction, xx, yy, zz, bounds) + 1.0),
        0.0,
        1.0,
    )
    start = float(component["start_fraction"])
    end = float(component["end_fraction"])
    before = float(component.get("before_relative_density", 1.0))
    start_value = float(component.get("start_relative_density", 1.0))
    end_value = float(component.get("end_relative_density", 1.0))
    after = float(component.get("after_relative_density", end_value))
    within = start_value + (end_value - start_value) * (
        (position - start) / (end - start)
    )
    return np.where(position < start, before, np.where(position <= end, within, after))


def dopant_density_field(
    grid: Grid3D,
    material: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the absolute dopant density on cell centers.

    With ``volume_average`` normalization, ``material.ion_density_m3`` remains
    the volume-weighted active-crystal average and therefore preserves total
    ion inventory when a nonuniform profile is enabled.
    """

    validate_doping_profile(material)
    base_density = float(material["ion_density_m3"])
    mask = crystal_volume_mask(grid, material)
    cfg = material.get("doping_profile", {})
    enabled = bool(cfg.get("enabled", False))
    raw = np.ones(grid.shape, dtype=float)
    clipped_cells = 0
    if enabled:
        raw.fill(float(cfg.get("background_fraction", 1.0)))
        xx, yy, zz = _coordinates(grid)
        bounds = _geometry_bounds(grid, material)
        endcaps = cfg.get("endcaps", {})
        endcap_cells: np.ndarray | None = None
        if bool(endcaps.get("enabled", False)):
            front_length = float(endcaps.get("front_length_m", 0.0))
            back_length = float(endcaps.get("back_length_m", 0.0))
            crystal_length = bounds["z_max"] - bounds["z_min"]
            if front_length + back_length >= crystal_length:
                raise ValueError(
                    "doping endcap lengths must leave a non-zero active core between them"
                )
            endcap_cells = (zz < bounds["z_min"] + front_length) | (
                zz > bounds["z_max"] - back_length
            )
        shape = str(material.get("geometry", {}).get("shape", "cuboid"))
        multiplicative_components: list[dict[str, Any]] = []
        for component in cfg.get("components", []):
            amplitude = float(component.get("amplitude", 0.0))
            kind = str(component.get("type", "surface_exponential"))
            direction = str(component.get("direction", "z_min"))
            if kind in {
                "axis_symmetric",
                "axis_symmetric_linear",
                "axis_monotonic_linear",
                "axis_segment_linear",
            }:
                multiplicative_components.append(component)
            elif amplitude == 0.0:
                continue
            elif kind == "surface_exponential":
                depth = np.maximum(
                    _surface_depth(direction, xx, yy, zz, bounds, shape), 0.0
                )
                raw += amplitude * np.exp(
                    -depth / float(component["decay_length_m"])
                )
            else:
                raw += amplitude * _normalized_axis(direction, xx, yy, zz, bounds)
        for component in multiplicative_components:
            if component.get("type") == "axis_symmetric":
                factor = _symmetric_center_to_surface_factor(
                    component, xx, yy, zz, bounds
                )
            elif component.get("type") == "axis_segment_linear":
                factor = _segment_linear_factor(component, xx, yy, zz, bounds)
            else:
                factor = _simple_relative_factor(
                    component, xx, yy, zz, bounds
                )
            raw *= factor
        minimum = float(cfg.get("minimum_raw_fraction", 0.0))
        clipped_cells = int(np.count_nonzero(mask & (raw < minimum)))
        raw = np.maximum(raw, minimum)
        if endcap_cells is not None:
            raw[endcap_cells] *= float(endcaps.get("relative_density", 0.0))

    active_volume = float(grid.integrate_volume(mask.astype(float)))
    raw_active_integral = float(grid.integrate_volume(raw * mask))
    normalization = str(cfg.get("normalization", "volume_average")) if enabled else "uniform"
    if not enabled:
        scale = base_density
    elif normalization == "volume_average":
        if raw_active_integral <= 0.0:
            raise ValueError("doping profile has zero active-volume average")
        scale = base_density * active_volume / raw_active_integral
    elif normalization == "peak":
        peak = float(np.max(raw[mask]))
        if peak <= 0.0:
            raise ValueError("doping profile has zero peak")
        scale = base_density / peak
    else:
        scale = base_density
    density = np.where(mask, raw * scale, 0.0)
    active_values = density[mask]
    mean_density = float(grid.integrate_volume(density) / active_volume)
    summary = {
        "enabled": enabled,
        "normalization": normalization,
        "reference_density_m3": base_density,
        "background_fraction": float(cfg.get("background_fraction", 1.0)),
        "minimum_raw_fraction": float(cfg.get("minimum_raw_fraction", 0.0)),
        "components": [dict(component) for component in cfg.get("components", [])],
        "endcaps": dict(cfg.get("endcaps", {})),
        "minimum_density_m3": float(np.min(active_values)),
        "maximum_density_m3": float(np.max(active_values)),
        "volume_average_density_m3": mean_density,
        "total_dopant_ions": float(grid.integrate_volume(density)),
        "active_volume_m3": active_volume,
        "raw_clipped_cell_count": clipped_cells,
    }
    return density, summary
