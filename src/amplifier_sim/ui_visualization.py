"""Plotly builders used by the Streamlit UI.

The numerical solvers do not import this module.  Keeping presentation code
here lets the UI evolve without coupling Plotly to the simulation engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from .bench_preview import BenchPreview, PreviewElement
from .geometry import crystal_volume_mask
from .path_view import (
    OpticalPathDisplayScale,
    align_previews_to_shared_gain,
    build_path_display_scale,
    gain_sample_mask,
    optical_trace_anchors,
)


SLICE_AXES: dict[str, tuple[int, str, str]] = {
    "z — 横截面 (x-y)": (0, "x", "y"),
    "y — 纵截面 (x-z)": (1, "x", "z"),
    "x — 纵截面 (y-z)": (2, "y", "z"),
}


def finite_plot_value(value: Any) -> float | None:
    """Return a finite plotting scalar, preserving JSON ``null`` as a gap."""

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if np.isfinite(numeric) else None


def signal_cross_section_preview(levels: Any, wavelength_m: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Sample a display sweep without extending a strict material data range.

    A plotting range can be wider than the valid simulated spectrum. Missing
    samples remain gaps and their errors are returned to the UI, never fed
    back into the solver or replaced by endpoint-clamped physical values.
    """
    wavelengths = np.asarray(wavelength_m, dtype=float)
    values = np.full((4, wavelengths.size), np.nan)
    errors: list[str] = []
    for index, wavelength in enumerate(wavelengths):
        try:
            base_abs, base_em = levels.cross_sections("signal", float(wavelength))
            local_abs, local_em = levels.local_cross_sections("signal", float(wavelength), ())
            # Static cross sections are scalars for a scalar wavelength;
            # local cross sections retain their leading wavelength axis.
            # Extract one value from each without creating a ragged array.
            values[:, index] = [
                np.asarray(value, dtype=float).item()
                for value in (base_abs, base_em, local_abs, local_em)
            ]
        except ValueError as exc:
            message = str(exc)
            if message not in errors:
                errors.append(message)
    return values, errors


def convergence_residual_value(row: Mapping[str, Any]) -> float | None:
    """Combine every available stopping residual without inventing zero.

    The first fixed-point record has no previous output, so its output
    residual is infinity in memory. Portable result JSON intentionally turns
    non-finite numbers into ``null``; a reloaded archive renders that point as
    a gap instead of attempting ``float(None)``.
    """

    required_values = (
        finite_plot_value(row.get("population_residual_relative")),
        finite_plot_value(row.get("output_residual_relative")),
    )
    if any(value is None for value in required_values):
        return None
    values = [value for value in required_values if value is not None]
    carrier_key = "free_carrier_convergence_residual_relative"
    if carrier_key in row:
        carrier_value = finite_plot_value(row.get(carrier_key))
        if carrier_value is None:
            # A serialized infinity is ``null``.  Once the carrier gate is
            # present it is a required residual, not an optional zero-valued
            # diagnostic, so preserve the same plot gap as the other gates.
            return None
        values.append(carrier_value)
    return max(values)


def finite_weighted_statistics(
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Return finite minimum, weighted mean, and maximum.

    Physical result grids may be centre-stretched, so an arithmetic mean over
    cells is not a material-area or material-volume mean.  The caller supplies
    the finite-volume cell areas/volumes explicitly; invalid or masked cells
    carry no statistical weight.  Empty domains return NaNs so the UI can
    render an unavailable value without inventing zero.
    """

    array = np.asarray(values, dtype=float)
    statistical_weights = np.asarray(weights, dtype=float)
    if array.shape != statistical_weights.shape:
        raise ValueError("values and weights must have identical shapes")
    valid = (
        np.isfinite(array)
        & np.isfinite(statistical_weights)
        & (statistical_weights > 0.0)
    )
    if mask is not None:
        material_mask = np.asarray(mask, dtype=bool)
        if material_mask.shape != array.shape:
            raise ValueError("mask must have the same shape as values")
        valid &= material_mask
    if not np.any(valid):
        return float("nan"), float("nan"), float("nan")
    active_values = array[valid]
    active_weights = statistical_weights[valid]
    weighted_mean = float(
        np.sum(active_values * active_weights) / np.sum(active_weights)
    )
    return (
        float(np.min(active_values)),
        weighted_mean,
        float(np.max(active_values)),
    )


def axial_weighted_statistics(
    values: np.ndarray,
    cell_areas: np.ndarray,
    material_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-z min, area-weighted mean, and max for a 3-D field."""

    array = np.asarray(values, dtype=float)
    areas = np.asarray(cell_areas, dtype=float)
    if array.ndim != 3:
        raise ValueError("values must have shape (nz, ny, nx)")
    if areas.shape != array.shape[-2:]:
        raise ValueError("cell_areas must have shape (ny, nx)")
    if material_mask is None:
        mask = np.ones(array.shape, dtype=bool)
    else:
        mask = np.asarray(material_mask, dtype=bool)
        if mask.shape != array.shape:
            raise ValueError("material_mask must have the same shape as values")

    minimum = np.full(array.shape[0], np.nan, dtype=float)
    average = np.full(array.shape[0], np.nan, dtype=float)
    maximum = np.full(array.shape[0], np.nan, dtype=float)
    for index, plane in enumerate(array):
        minimum[index], average[index], maximum[index] = (
            finite_weighted_statistics(
                plane,
                areas,
                mask[index],
            )
        )
    return minimum, average, maximum


def solver_notice_presentation(message: Any) -> tuple[str, str]:
    """Classify a solver message for UI severity without changing archives.

    Solver warnings deliberately preserve model contracts and display-window
    disclosures alongside actionable numerical warnings.  Presenting every
    one as the same yellow warning makes a valid approximation look like a
    numerical failure, so the UI separates non-blocking information here.
    """

    text = str(message)
    if text.startswith("ABCD uses an analytic infinite-plane source:"):
        return "info", "解析源面显示"
    if (
        text.startswith("Commercial App fast mode uses")
        or text.startswith("Commercial App fast mode projects")
        or text.startswith("Commercial App CW mode uses")
        or text.startswith("Commercial App output M2 includes")
        or (
            text.startswith("Higher-order spectral phase was configured")
            and "; Commercial App carries it" in text
        )
    ):
        return "info", "Commercial App 模型合同"
    if "has no phase column" in text or "phase was assumed" in text:
        return "warning", "输入相位假设"
    return "warning", "求解器"


def optical_plane_axes(result: Any, plane_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Resolve axes for a saved optical plane without assuming one shared grid.

    New result archives store explicit coordinates for every plane.  The grid
    fallback keeps older archives readable and, importantly, distinguishes a
    Pump plane from a Seed plane when those beams used different optical grids.
    """

    plane = np.asarray(result.optical_planes[plane_name])
    if plane.ndim != 2:
        raise ValueError(
            f"optical plane {plane_name!r} must be two-dimensional, got {plane.shape}"
        )
    coordinates = (result.optical_plane_coordinates or {}).get(plane_name)
    if coordinates is not None:
        x_m = np.asarray(coordinates[0], dtype=float)
        y_m = np.asarray(coordinates[1], dtype=float)
        if x_m.ndim != 1 or y_m.ndim != 1 or not len(x_m) or not len(y_m):
            raise ValueError(
                f"optical plane {plane_name!r} coordinates must be non-empty 1D arrays"
            )
        if plane.shape != (len(y_m), len(x_m)):
            raise ValueError(
                f"optical plane {plane_name!r} shape {plane.shape} does not match "
                f"its saved coordinates {(len(y_m), len(x_m))}"
            )
        return x_m, y_m

    pump_first = str(plane_name).lower().startswith("pump")
    candidate_grids = (
        (result.pump_optical_grid, result.optical_grid, result.grid)
        if pump_first
        else (result.optical_grid, result.grid, result.pump_optical_grid)
    )
    seen_shapes: set[tuple[int, int]] = set()
    for grid in candidate_grids:
        if grid is None:
            continue
        shape = (len(grid.y), len(grid.x))
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        if plane.shape == shape:
            return np.asarray(grid.x, dtype=float), np.asarray(grid.y, dtype=float)
    raise ValueError(
        f"optical plane {plane_name!r} shape {plane.shape} matches neither its "
        "saved coordinates nor any archived optical/physical grid"
    )


def volume_slice(
    values: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    axis_label: str,
    index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, float]:
    """Return one Cartesian slice and matching horizontal/vertical axes."""

    array = np.asarray(values, dtype=float)
    if array.shape != (len(z_m), len(y_m), len(x_m)):
        raise ValueError("volume must have shape (nz, ny, nx)")
    if axis_label not in SLICE_AXES:
        raise ValueError(f"unsupported slice axis {axis_label!r}")
    axis, horizontal_name, vertical_name = SLICE_AXES[axis_label]
    coordinates = (z_m, y_m, x_m)
    coordinate = coordinates[axis]
    if not 0 <= int(index) < len(coordinate):
        raise IndexError("slice index is outside the selected axis")
    if axis == 0:
        plane = array[index, :, :]
        horizontal, vertical = x_m, y_m
    elif axis == 1:
        plane = array[:, index, :]
        horizontal, vertical = x_m, z_m
    else:
        plane = array[:, :, index]
        horizontal, vertical = y_m, z_m
    return (
        plane,
        np.asarray(horizontal) * 1e3,
        np.asarray(vertical) * 1e3,
        f"{horizontal_name} (mm)",
        f"{vertical_name} (mm)",
        float(coordinate[index] * 1e3),
    )


def slice_figure(
    values: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    axis_label: str,
    index: int,
    title: str,
    unit: str,
    colorscale: str = "Viridis",
    signed: bool = False,
) -> go.Figure:
    plane, horizontal, vertical, horizontal_label, vertical_label, position_mm = volume_slice(
        values, x_m, y_m, z_m, axis_label, index
    )
    normal = axis_label[0]
    figure = go.Figure(
        go.Heatmap(
            z=plane,
            x=horizontal,
            y=vertical,
            colorscale=colorscale,
            zmid=0.0 if signed else None,
            colorbar_title=unit,
            hovertemplate=(
                f"{horizontal_label.split()[0]} %{{x:.5g}} mm<br>"
                f"{vertical_label.split()[0]} %{{y:.5g}} mm<br>"
                f"{normal} {position_mm:.5g} mm<br>%{{z:.5g}} {unit}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=f"{title} · {normal}={position_mm:.5g} mm",
        xaxis_title=horizontal_label,
        yaxis_title=vertical_label,
        height=500,
        margin=dict(t=55, l=60, r=25, b=55),
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1.0)
    return figure


def _display_crystal_mask(result: Any) -> np.ndarray:
    """Recover legacy missing/empty masks from the recorded physical geometry."""

    if result.crystal_mask is not None:
        mask = np.asarray(result.crystal_mask, dtype=bool)
        if np.any(mask):
            return mask
    geometry = dict(result.summary.get("crystal_geometry", {}))
    if not geometry:
        # Older archives without geometry use the full recorded grid domain.
        return np.ones(result.grid.shape, dtype=bool)
    try:
        return crystal_volume_mask(result.grid, {"geometry": geometry})
    except ValueError:
        # An explicit geometry outside the grid is not permission to display
        # exterior samples as material.
        return np.zeros(result.grid.shape, dtype=bool)


def volume_figure(
    values: np.ndarray,
    result: Any,
    title: str,
    unit: str,
    colorscale: str = "Viridis",
    signed: bool = False,
) -> go.Figure:
    """Render a masked 3D scalar field with all spatial axes in millimetres."""

    array = np.asarray(values, dtype=float)
    mask = _display_crystal_mask(result)
    mask_has_active_cells = bool(mask is not None and np.any(mask))
    if mask_has_active_cells:
        active = np.argwhere(mask)
        lower = np.maximum(np.min(active, axis=0) - 1, 0)
        upper = np.minimum(np.max(active, axis=0) + 2, np.asarray(array.shape))
        bounds = tuple(slice(int(lower[axis]), int(upper[axis])) for axis in range(3))
    else:
        bounds = (slice(None), slice(None), slice(None))
    cropped = array[bounds]
    stride = tuple(max(1, int(np.ceil(size / 34))) for size in cropped.shape)
    sampled = cropped[:: stride[0], :: stride[1], :: stride[2]].copy()
    if mask is not None:
        sampled_mask = mask[bounds][:: stride[0], :: stride[1], :: stride[2]]
        sampled[~sampled_mask] = np.nan
    z = np.asarray(result.grid.z)[bounds[0]][:: stride[0]] * 1e3
    y = np.asarray(result.grid.y)[bounds[1]][:: stride[1]] * 1e3
    x = np.asarray(result.grid.x)[bounds[2]][:: stride[2]] * 1e3
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    finite = sampled[np.isfinite(sampled)]
    if signed:
        maximum = max(float(np.max(np.abs(finite))) if finite.size else 0.0, np.finfo(float).tiny)
        isomin, isomax = -maximum, maximum
    else:
        positive = finite[finite > 0.0]
        isomin = float(np.percentile(positive, 35)) if positive.size else float(np.min(finite)) if finite.size else 0.0
        isomax = float(np.max(finite)) if finite.size else 1.0
        if isomax <= isomin:
            isomax = isomin + max(abs(isomin), 1.0) * 1e-12
    figure = go.Figure(
        go.Volume(
            x=xx.ravel(),
            y=yy.ravel(),
            z=zz.ravel(),
            value=sampled.ravel(),
            isomin=isomin,
            isomax=isomax,
            opacity=0.13,
            surface_count=16,
            colorscale=colorscale,
            colorbar_title=unit,
            caps=dict(x_show=False, y_show=False, z_show=False),
            hovertemplate="x %{x:.5g} mm<br>y %{y:.5g} mm<br>z %{z:.5g} mm<br>%{value:.5g} "
            + unit
            + "<extra></extra>",
        )
    )
    geometry = dict(result.summary.get("crystal_geometry", {}))
    if not geometry:
        geometry = {
            "shape": "cuboid",
            "center_x_m": float(
                0.5 * (result.grid.x_edges[0] + result.grid.x_edges[-1])
            ),
            "center_y_m": float(
                0.5 * (result.grid.y_edges[0] + result.grid.y_edges[-1])
            ),
            "width_x_m": float(np.ptp(result.grid.x_edges)),
            "height_y_m": float(np.ptp(result.grid.y_edges)),
        }
    geometry.setdefault(
        "center_x_m",
        float(0.5 * (result.grid.x_edges[0] + result.grid.x_edges[-1])),
    )
    geometry.setdefault(
        "center_y_m",
        float(0.5 * (result.grid.y_edges[0] + result.grid.y_edges[-1])),
    )
    if geometry.get("shape") == "cylinder":
        geometry.setdefault(
            "diameter_m",
            float(min(np.ptp(result.grid.x_edges), np.ptp(result.grid.y_edges))),
        )
    else:
        geometry.setdefault("width_x_m", float(np.ptp(result.grid.x_edges)))
        geometry.setdefault("height_y_m", float(np.ptp(result.grid.y_edges)))
    figure.add_traces(_crystal_outline_traces(geometry, result.grid))
    length_z_mm = float(result.grid.z_edges[-1] - result.grid.z_edges[0]) * 1e3
    if geometry.get("shape") == "cylinder":
        width_x_mm = height_y_mm = float(
            geometry.get(
                "diameter_m",
                min(np.ptp(result.grid.x_edges), np.ptp(result.grid.y_edges)),
            )
        ) * 1e3
    else:
        width_x_mm = float(
            geometry.get("width_x_m", np.ptp(result.grid.x_edges))
        ) * 1e3
        height_y_mm = float(
            geometry.get("height_y_m", np.ptp(result.grid.y_edges))
        ) * 1e3
    center_x_mm = float(geometry.get("center_x_m", 0.0)) * 1e3
    center_y_mm = float(geometry.get("center_y_m", 0.0)) * 1e3
    physical_spans = np.asarray([width_x_mm, height_y_mm, length_z_mm], dtype=float)
    physical_aspect = physical_spans / max(float(np.max(physical_spans)), np.finfo(float).tiny)
    figure.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title="x (mm)", range=[center_x_mm - 0.5 * width_x_mm, center_x_mm + 0.5 * width_x_mm]
            ),
            yaxis=dict(
                title="y (mm)", range=[center_y_mm - 0.5 * height_y_mm, center_y_mm + 0.5 * height_y_mm]
            ),
            zaxis=dict(
                title="z (mm)",
                range=[float(result.grid.z_edges[0]) * 1e3, float(result.grid.z_edges[-1]) * 1e3],
            ),
            aspectmode="manual",
            aspectratio=dict(
                x=float(physical_aspect[0]),
                y=float(physical_aspect[1]),
                z=float(physical_aspect[2]),
            ),
            camera=dict(
                eye=dict(x=1.45, y=1.35, z=1.15),
                projection=dict(type="orthographic"),
            ),
        ),
        height=690,
        margin=dict(t=55, l=0, r=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
    )
    return figure


def optical_path_3d_figure(
    previews: Sequence[BenchPreview],
    config: dict[str, Any],
    *,
    result: Any | None = None,
    field_values: np.ndarray | None = None,
    field_name: str = "",
    unit: str = "",
    colorscale: str = "Viridis",
    signed: bool = False,
    active_beam_kind: str = "seed",
    focus_gain: bool = True,
    optical_trace_summaries: Sequence[dict[str, Any]] | None = None,
    optical_gain_path_samples: Sequence[dict[str, Any]] | None = None,
) -> go.Figure:
    """Render cold q volumes and optional solver-recorded D4-sigma planes."""

    figure = go.Figure()
    palette = {"seed": "#ef4444", "pump": "#2563eb"}
    aligned_previews = align_previews_to_shared_gain(previews, float(config["grid"]["length_z_m"]))
    gain_inserted = any(
        element.kind == "gain_medium" and element.material_enabled
        for preview in aligned_previews
        for element in preview.elements
    )
    scale = build_path_display_scale(aligned_previews, focus_gain=focus_gain)
    transverse_x_extent = 0.0
    transverse_y_extent = 0.0
    ordered_pairs = sorted(
        zip(previews, aligned_previews, strict=True),
        key=lambda pair: pair[0].beam_kind == active_beam_kind,
    )
    for preview, aligned in ordered_pairs:
        color = palette[preview.beam_kind]
        active = preview.beam_kind == active_beam_kind
        physical_path_mm = np.asarray(preview.optical_path_m) * 1e3
        path_mm = np.asarray(scale.map_m(aligned.optical_path_m)) * 1e3
        center_x_mm = np.asarray(preview.centroid_x_m) * 1e3
        center_y_mm = np.asarray(preview.centroid_y_m) * 1e3
        radius_x_mm = np.asarray(preview.radius_x_D4sigma_m) * 1e3
        radius_y_mm = np.asarray(preview.radius_y_D4sigma_m) * 1e3
        _add_beam_volume(
            figure,
            preview.beam_kind,
            path_mm,
            physical_path_mm,
            center_x_mm,
            center_y_mm,
            radius_x_mm,
            radius_y_mm,
            color,
            active,
        )
        figure.add_scatter3d(
            x=path_mm,
            y=center_x_mm,
            z=center_y_mm,
            mode="lines",
            line=dict(color=color, width=5 if active else 3, dash="solid"),
            opacity=1.0 if active else 0.72,
            name=f"{preview.beam_kind} 光轴",
            showlegend=False,
            customdata=np.column_stack((physical_path_mm, radius_x_mm, radius_y_mm)),
            hovertemplate=(
                "真实光程 %{customdata[0]:.5g} mm<br>x 束心 %{y:.5g} mm<br>y 束心 %{z:.5g} mm<br>"
                "D4σ rx %{customdata[1]:.5g} mm<br>D4σ ry %{customdata[2]:.5g} mm<extra></extra>"
            ),
        )
        figure.add_scatter3d(
            x=[path_mm[0]],
            y=[center_x_mm[0]],
            z=[center_y_mm[0]],
            mode="markers+text",
            marker=dict(color=color, size=7 if active else 6, symbol="diamond", opacity=1.0 if active else 0.82),
            text=[f"{preview.beam_kind.upper()} LASER"],
            textposition="top center" if preview.beam_kind == "seed" else "bottom center",
            name=f"{preview.beam_kind} 激光源",
            showlegend=False,
            hovertemplate=f"{preview.beam_kind.upper()} 激光源<extra></extra>",
        )
        if active:
            _add_preview_elements(figure, aligned.elements, config, color, scale)
        extent_mask = gain_sample_mask(preview) if focus_gain else np.ones(len(path_mm), dtype=bool)
        if np.any(extent_mask):
            transverse_x_extent = max(
                transverse_x_extent,
                float(np.max(np.abs(center_x_mm[extent_mask]) + radius_x_mm[extent_mask])),
            )
            transverse_y_extent = max(
                transverse_y_extent,
                float(np.max(np.abs(center_y_mm[extent_mask]) + radius_y_mm[extent_mask])),
            )
        if optical_trace_summaries or optical_gain_path_samples:
            anchors = optical_trace_anchors(
                preview,
                optical_trace_summaries or (),
                optical_gain_path_samples or (),
            )
            if anchors:
                extent_anchors = (
                    [
                        anchor
                        for anchor in anchors
                        if any(
                            element.kind == "gain_medium"
                            and element.start_m - 1e-12
                            <= anchor.optical_path_m
                            <= element.end_m + 1e-12
                            for element in preview.elements
                        )
                    ]
                    if focus_gain
                    else list(anchors)
                )
                if extent_anchors:
                    transverse_x_extent = max(
                        transverse_x_extent,
                        max(
                            abs(float(anchor.centroid_x_m))
                            + float(anchor.radius_x_D4sigma_m)
                            for anchor in extent_anchors
                        )
                        * 1e3,
                    )
                    transverse_y_extent = max(
                        transverse_y_extent,
                        max(
                            abs(float(anchor.centroid_y_m))
                            + float(anchor.radius_y_D4sigma_m)
                            for anchor in extent_anchors
                        )
                        * 1e3,
                    )
                _add_result_trace_anchor_rings(
                    figure,
                    preview,
                    aligned,
                    anchors,
                    scale,
                    color,
                    active,
                )
    if result is not None and field_values is not None and previews:
        preferred = next((preview for preview in aligned_previews if preview.beam_kind == active_beam_kind), aligned_previews[0])
        gain_element = next((element for element in preferred.elements if element.kind == "gain_medium"), None)
        if gain_element is not None and gain_element.material_enabled:
            _add_crystal_field_overlay(
                figure,
                gain_element,
                result,
                np.asarray(field_values, dtype=float),
                field_name,
                unit,
                colorscale,
                signed,
                scale,
            )
    geometry = config["material"].get("geometry", {})
    if geometry.get("shape", "cuboid") == "cylinder":
        diameter_m = geometry.get("diameter_m")
        if diameter_m is None:
            diameter_m = min(
                config["grid"]["size_x_m"], config["grid"]["size_y_m"]
            )
        crystal_half_x = crystal_half_y = 0.5 * float(
            diameter_m
        ) * 1e3
    else:
        width_x_m = geometry.get("width_x_m")
        if width_x_m is None:
            width_x_m = config["grid"]["size_x_m"]
        height_y_m = geometry.get("height_y_m")
        if height_y_m is None:
            height_y_m = config["grid"]["size_y_m"]
        crystal_half_x = 0.5 * float(width_x_m) * 1e3
        crystal_half_y = 0.5 * float(height_y_m) * 1e3
    transverse_x_extent = max(1.12 * transverse_x_extent, 1.12 * (abs(float(geometry.get("center_x_m", 0.0))) * 1e3 + crystal_half_x), 1e-3)
    transverse_y_extent = max(1.12 * transverse_y_extent, 1.12 * (abs(float(geometry.get("center_y_m", 0.0))) * 1e3 + crystal_half_y), 1e-3)
    display_start_mm = scale.display_start_m * 1e3
    display_end_mm = scale.display_end_m * 1e3
    display_span_mm = max(display_end_mm - display_start_mm, 1e-6)
    path_padding = max(0.025 * display_span_mm, 0.05)
    path_range = [display_start_mm - path_padding, display_end_mm + path_padding]
    scene_spans = np.asarray(
        [path_range[1] - path_range[0], 2.0 * transverse_x_extent, 2.0 * transverse_y_extent],
        dtype=float,
    )
    normalized_spans = scene_spans / max(float(np.max(scene_spans)), np.finfo(float).tiny)
    tick_values, tick_labels = scale.axis_ticks_mm()
    x_title = "相对共享晶体入口位置 (mm)" if not scale.compressed else "晶体对齐显示（长自由空间已压缩）"
    figure.update_layout(
        title=(
            (
                f"三维光学平台 · 冷态光束体 + 运行后 D4σ 采样环 · 当前强调 {active_beam_kind.upper()}"
                if optical_trace_summaries or optical_gain_path_samples
                else f"三维光学平台 · Pump + Seed 冷态 D4σ 光束体 · 当前编辑 {active_beam_kind.upper()}"
            )
            if gain_inserted
            else f"三维空气调光 · 晶体位置仅作占位 · 当前编辑 {active_beam_kind.upper()}"
        ),
        scene=dict(
            xaxis=dict(
                title=x_title,
                range=path_range,
                tickmode="array" if scale.compressed else "auto",
                tickvals=tick_values if scale.compressed else None,
                ticktext=tick_labels if scale.compressed else None,
            ),
            yaxis=dict(title="局部 x (mm)", range=[-transverse_x_extent, transverse_x_extent]),
            zaxis=dict(title="局部 y (mm)", range=[-transverse_y_extent, transverse_y_extent]),
            aspectmode="manual",
            # In focused mode the three scene axes use one common mm scale, so
            # the crystal's longitudinal and transverse proportions remain
            # physical after the long free-space sections are compressed.
            aspectratio=(
                dict(x=float(normalized_spans[0]), y=float(normalized_spans[1]), z=float(normalized_spans[2]))
                if focus_gain
                else dict(x=3.0, y=0.9, z=0.9)
            ),
            camera=dict(
                eye=dict(x=1.55, y=1.15, z=0.72),
                projection=dict(type="orthographic"),
            ),
            dragmode="orbit",
        ),
        height=590,
        margin=dict(t=55, l=0, r=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
        uirevision=f"optical-bench-{active_beam_kind}-{focus_gain}-{gain_inserted}",
    )
    return figure


def _add_result_trace_anchor_rings(
    figure: go.Figure,
    preview: BenchPreview,
    aligned: BenchPreview,
    anchors: Sequence[Any],
    scale: OpticalPathDisplayScale,
    color: str,
    active: bool,
) -> None:
    """Draw actual solver D4-sigma cross sections over the cold q volume."""

    preview_path = np.asarray(preview.optical_path_m, dtype=float)
    aligned_path = np.asarray(aligned.optical_path_m, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, 49)
    ring_x: list[float | None] = []
    ring_y: list[float | None] = []
    ring_z: list[float | None] = []
    ring_custom: list[list[Any] | None] = []
    center_x: list[float] = []
    center_y: list[float] = []
    center_z: list[float] = []
    center_custom: list[list[Any]] = []
    for anchor in anchors:
        # Solver anchors inside a gain medium generally do not coincide with
        # the cold-q preview samples.  The aligned preview is a continuous
        # re-parameterization of the same optical path, so interpolate that
        # coordinate instead of snapping a physical D4-sigma plane to the
        # nearest display sample.
        aligned_coordinate_m = float(
            np.interp(
                float(anchor.optical_path_m),
                preview_path,
                aligned_path,
            )
        )
        display_path_mm = float(scale.map_m(aligned_coordinate_m)) * 1e3
        cx = float(anchor.centroid_x_m) * 1e3
        cy = float(anchor.centroid_y_m) * 1e3
        rx = float(anchor.radius_x_D4sigma_m) * 1e3
        ry = float(anchor.radius_y_D4sigma_m) * 1e3
        label = str(anchor.name)
        center_x.append(display_path_mm)
        center_y.append(cx)
        center_z.append(cy)
        center_custom.append([label, rx, ry])
        ring_x.extend([display_path_mm] * len(theta) + [None])
        ring_y.extend((cx + rx * np.cos(theta)).tolist() + [None])
        ring_z.extend((cy + ry * np.sin(theta)).tolist() + [None])
        ring_custom.extend([[label, rx, ry]] * len(theta) + [None])
    label = preview.beam_kind.upper()
    figure.add_scatter3d(
        x=ring_x,
        y=ring_y,
        z=ring_z,
        mode="lines",
        line=dict(color=color, width=7 if active else 5, dash="dash"),
        opacity=1.0 if active else 0.82,
        name=f"{label} 运行后 D4σ 采样面",
        legendgroup=f"{preview.beam_kind}_result",
        customdata=ring_custom,
        hovertemplate=(
            "%{customdata[0]}<br>运行后 D4σ rx %{customdata[1]:.5g} mm<br>"
            "运行后 D4σ ry %{customdata[2]:.5g} mm<extra></extra>"
        ),
    )
    figure.add_scatter3d(
        x=center_x,
        y=center_y,
        z=center_z,
        mode="lines+markers",
        line=dict(color=color, width=6 if active else 4, dash="dot"),
        marker=dict(color=color, size=5 if active else 4, symbol="circle"),
        opacity=1.0 if active else 0.82,
        name=f"{label} 运行后采样束心",
        legendgroup=f"{preview.beam_kind}_result",
        showlegend=False,
        customdata=center_custom,
        hovertemplate=(
            "%{customdata[0]}<br>运行后 D4σ rx %{customdata[1]:.5g} mm<br>"
            "运行后 D4σ ry %{customdata[2]:.5g} mm<extra></extra>"
        ),
    )


def _add_beam_volume(
    figure: go.Figure,
    beam_kind: str,
    path_mm: np.ndarray,
    physical_path_mm: np.ndarray,
    center_x_mm: np.ndarray,
    center_y_mm: np.ndarray,
    radius_x_mm: np.ndarray,
    radius_y_mm: np.ndarray,
    color: str,
    active: bool,
) -> None:
    """Add an elliptical, capped tube whose surface is the beam's D4-sigma radius."""

    if not len(path_mm):
        return
    # Keep the live view responsive even when a many-pass bench produces
    # thousands of longitudinal samples. End points are always retained.
    maximum_rings = 180
    if len(path_mm) > maximum_rings:
        indices = np.unique(np.linspace(0, len(path_mm) - 1, maximum_rings).round().astype(int))
    else:
        indices = np.arange(len(path_mm))
    path = np.asarray(path_mm)[indices]
    physical_path = np.asarray(physical_path_mm)[indices]
    center_x = np.asarray(center_x_mm)[indices]
    center_y = np.asarray(center_y_mm)[indices]
    radius_x = np.maximum(np.asarray(radius_x_mm)[indices], 0.0)
    radius_y = np.maximum(np.asarray(radius_y_mm)[indices], 0.0)

    sides = 24
    theta = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    vertex_x = np.repeat(path[:, None], sides, axis=1)
    vertex_y = center_x[:, None] + radius_x[:, None] * np.cos(theta)[None, :]
    vertex_z = center_y[:, None] + radius_y[:, None] * np.sin(theta)[None, :]
    custom = np.stack(
        (
            np.repeat(physical_path[:, None], sides, axis=1),
            np.repeat(radius_x[:, None], sides, axis=1),
            np.repeat(radius_y[:, None], sides, axis=1),
        ),
        axis=-1,
    ).reshape(-1, 3)

    x = vertex_x.ravel().tolist()
    y = vertex_y.ravel().tolist()
    z = vertex_z.ravel().tolist()
    i: list[int] = []
    j: list[int] = []
    k: list[int] = []
    for ring in range(len(path) - 1):
        first = ring * sides
        second = (ring + 1) * sides
        for side in range(sides):
            following = (side + 1) % sides
            i.extend((first + side, first + side))
            j.extend((second + side, second + following))
            k.extend((second + following, first + following))

    # Close both ends so the source and crystal-side beam cross sections read
    # as volumes instead of open wire-frame envelopes.
    start_center = len(x)
    x.append(float(path[0]))
    y.append(float(center_x[0]))
    z.append(float(center_y[0]))
    end_center = len(x)
    x.append(float(path[-1]))
    y.append(float(center_x[-1]))
    z.append(float(center_y[-1]))
    custom = np.vstack(
        (
            custom,
            [physical_path[0], radius_x[0], radius_y[0]],
            [physical_path[-1], radius_x[-1], radius_y[-1]],
        )
    )
    end_ring = (len(path) - 1) * sides
    for side in range(sides):
        following = (side + 1) % sides
        i.extend((start_center, end_center))
        j.extend((following, end_ring + side))
        k.extend((side, end_ring + following))

    figure.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=0.32 if active else 0.20,
            flatshading=False,
            lighting=dict(ambient=0.72, diffuse=0.74, specular=0.22, roughness=0.76, fresnel=0.08),
            lightposition=dict(x=-300, y=500, z=800),
            name=f"{beam_kind.upper()} D4σ 光束体",
            legendgroup=beam_kind,
            customdata=custom,
            hovertemplate=(
                f"{beam_kind.upper()} D4σ 光束体<br>"
                "真实光程 %{customdata[0]:.5g} mm<br>"
                "D4σ rx %{customdata[1]:.5g} mm<br>"
                "D4σ ry %{customdata[2]:.5g} mm<extra></extra>"
            ),
        )
    )

    # Plotly's triangular mesh has no independent edge style. Overlay only the
    # four longitudinal generators and the two end ellipses: this gives the
    # translucent volume a crisp, thin outline without turning it into a dense
    # wire frame.
    outline_x: list[float | None] = []
    outline_y: list[float | None] = []
    outline_z: list[float | None] = []
    for angle in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
        outline_x.extend(path.tolist())
        outline_y.extend((center_x + radius_x * np.cos(angle)).tolist())
        outline_z.extend((center_y + radius_y * np.sin(angle)).tolist())
        outline_x.append(None)
        outline_y.append(None)
        outline_z.append(None)
    rim_theta = np.linspace(0.0, 2.0 * np.pi, 49)
    for ring in (0, -1):
        outline_x.extend(np.full_like(rim_theta, path[ring]).tolist())
        outline_y.extend((center_x[ring] + radius_x[ring] * np.cos(rim_theta)).tolist())
        outline_z.extend((center_y[ring] + radius_y[ring] * np.sin(rim_theta)).tolist())
        outline_x.append(None)
        outline_y.append(None)
        outline_z.append(None)
    figure.add_scatter3d(
        x=outline_x,
        y=outline_y,
        z=outline_z,
        mode="lines",
        line=dict(color=color, width=1.4),
        opacity=0.82 if active else 0.58,
        name=f"{beam_kind} D4σ 描边",
        legendgroup=beam_kind,
        showlegend=False,
        hoverinfo="skip",
    )


def _add_crystal_field_overlay(
    figure: go.Figure,
    gain_element: PreviewElement,
    result: Any,
    values: np.ndarray,
    field_name: str,
    unit: str,
    colorscale: str,
    signed: bool,
    scale: OpticalPathDisplayScale,
) -> None:
    """Map a crystal `(z,y,x)` field into the first gain segment of path space."""

    stride = tuple(max(1, int(np.ceil(size / 28))) for size in values.shape)
    sampled = values[:: stride[0], :: stride[1], :: stride[2]].copy()
    mask = _display_crystal_mask(result)[:: stride[0], :: stride[1], :: stride[2]]
    sampled[~mask] = np.nan
    local_z = np.asarray(result.grid.z)[:: stride[0]]
    z_edges = np.asarray(result.grid.z_edges, dtype=float)
    crystal_span_m = float(z_edges[-1] - z_edges[0])
    if crystal_span_m > 0.0:
        # Grid z values are cell centres.  Map them relative to the physical
        # crystal faces, not relative to the first/last retained centre; the
        # latter stretched the volume by half a cell at both ends.
        fraction = (local_z - z_edges[0]) / crystal_span_m
    else:
        fraction = np.full_like(local_z, 0.5)
    if str(gain_element.direction or "forward") == "backward":
        fraction = 1.0 - fraction
    physical_path_m = gain_element.start_m + fraction * (gain_element.end_m - gain_element.start_m)
    path_mm = np.asarray(scale.map_m(physical_path_m)) * 1e3
    local_y_mm = np.asarray(result.grid.y)[:: stride[1]] * 1e3
    local_x_mm = np.asarray(result.grid.x)[:: stride[2]] * 1e3
    ss, yy, xx = np.meshgrid(path_mm, local_y_mm, local_x_mm, indexing="ij")
    true_ss, _, _ = np.meshgrid(
        physical_path_m * 1e3,
        local_y_mm,
        local_x_mm,
        indexing="ij",
    )
    # The second custom coordinate needs the same y dimension as the volume.
    local_zz = np.broadcast_to(
        (local_z * 1e3)[:, None, None], sampled.shape
    )
    hover_coordinates = np.column_stack(
        (true_ss.ravel(), local_zz.ravel())
    )
    finite = sampled[np.isfinite(sampled)]
    if signed:
        maximum = max(float(np.max(np.abs(finite))) if finite.size else 0.0, np.finfo(float).tiny)
        isomin, isomax = -maximum, maximum
    else:
        positive = finite[finite > 0.0]
        isomin = float(np.percentile(positive, 35)) if positive.size else float(np.min(finite)) if finite.size else 0.0
        isomax = float(np.max(finite)) if finite.size else 1.0
        if isomax <= isomin:
            isomax = isomin + max(abs(isomin), 1.0) * 1e-12
    figure.add_volume(
        x=ss.ravel(),
        y=xx.ravel(),
        z=yy.ravel(),
        value=sampled.ravel(),
        customdata=hover_coordinates,
        isomin=isomin,
        isomax=isomax,
        opacity=0.16,
        surface_count=13,
        colorscale=colorscale,
        colorbar=dict(title=unit, x=1.02),
        caps=dict(x_show=False, y_show=False, z_show=False),
        name=field_name or "晶体体场",
        hovertemplate=(
            "路径坐标 %{customdata[0]:.5g} mm<br>晶体轴向 z %{customdata[1]:.5g} mm<br>"
            "局部 x %{y:.5g} mm<br>局部 y %{z:.5g} mm<br>"
            "%{value:.5g} " + unit + "<extra>" + (field_name or "晶体体场") + "</extra>"
        ),
    )


def _add_preview_elements(
    figure: go.Figure,
    elements: Sequence[PreviewElement],
    config: dict[str, Any],
    beam_color: str,
    scale: OpticalPathDisplayScale,
) -> None:
    geometry = config["material"].get("geometry", {})
    center_x = float(geometry.get("center_x_m", 0.0)) * 1e3
    center_y = float(geometry.get("center_y_m", 0.0)) * 1e3
    if geometry.get("shape", "cuboid") == "cylinder":
        diameter_m = geometry.get("diameter_m")
        if diameter_m is None:
            diameter_m = min(
                config["grid"]["size_x_m"], config["grid"]["size_y_m"]
            )
        half_x = half_y = 0.5 * float(
            diameter_m
        ) * 1e3
    else:
        width_x_m = geometry.get("width_x_m")
        if width_x_m is None:
            width_x_m = config["grid"]["size_x_m"]
        height_y_m = geometry.get("height_y_m")
        if height_y_m is None:
            height_y_m = config["grid"]["size_y_m"]
        half_x = 0.5 * float(width_x_m) * 1e3
        half_y = 0.5 * float(height_y_m) * 1e3
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    display_counts: dict[str, int] = {}
    for element in elements:
        midpoint = 0.5 * (
            float(scale.map_m(element.start_m)) + float(scale.map_m(element.end_m))
        ) * 1e3
        if element.kind == "gain_medium":
            start = float(scale.map_m(element.start_m)) * 1e3
            end = float(scale.map_m(element.end_m)) * 1e3
            material_enabled = bool(element.material_enabled)
            gain_color = "#f97316" if material_enabled else "#94a3b8"
            gain_dash = "solid" if material_enabled else "dash"
            if geometry.get("shape", "cuboid") == "cylinder":
                for path_position in (start, end):
                    figure.add_scatter3d(
                        x=np.full_like(theta, path_position),
                        y=center_x + half_x * np.cos(theta),
                        z=center_y + half_y * np.sin(theta),
                        mode="lines", line=dict(color=gain_color, width=3, dash=gain_dash),
                        showlegend=False, hoverinfo="skip",
                    )
                for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
                    figure.add_scatter3d(
                        x=[start, end],
                        y=[center_x + half_x * np.cos(angle)] * 2,
                        z=[center_y + half_y * np.sin(angle)] * 2,
                        mode="lines", line=dict(color=gain_color, width=2, dash=gain_dash),
                        showlegend=False, hoverinfo="skip",
                    )
            else:
                vertices = [
                    (start, center_x - half_x, center_y - half_y),
                    (start, center_x + half_x, center_y - half_y),
                    (start, center_x + half_x, center_y + half_y),
                    (start, center_x - half_x, center_y + half_y),
                    (end, center_x - half_x, center_y - half_y),
                    (end, center_x + half_x, center_y - half_y),
                    (end, center_x + half_x, center_y + half_y),
                    (end, center_x - half_x, center_y + half_y),
                ]
                edges = (
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                )
                for first, second in edges:
                    figure.add_scatter3d(
                        x=[vertices[first][0], vertices[second][0]],
                        y=[vertices[first][1], vertices[second][1]],
                        z=[vertices[first][2], vertices[second][2]],
                        mode="lines",
                        line=dict(color=gain_color, width=3, dash=gain_dash),
                        showlegend=False,
                        hoverinfo="skip",
                    )
            figure.add_scatter3d(
                x=[midpoint], y=[center_x], z=[center_y + half_y], mode="markers+text",
                marker=dict(color=gain_color, size=5, symbol="square"),
                text=[f"G{element.pass_index}" if material_enabled else f"G{element.pass_index} 空位"],
                textposition="top center",
                name="增益晶体" if material_enabled else "晶体预留位置",
                showlegend=False,
                hovertemplate=(
                    f"{element.name}<br>晶体第{element.pass_index}程<br>{element.direction}<extra></extra>"
                    if material_enabled
                    else f"{element.name}<br>晶体已取下；该段按外部介质传播<extra></extra>"
                ),
            )
        elif element.kind in {"thin_lens", "thick_lens", "double_pass_lens_relay"}:
            display_counts["L"] = display_counts.get("L", 0) + 1
            figure.add_scatter3d(
                x=np.full_like(theta, midpoint),
                y=element.center_x_m * 1e3 + 1.15 * half_x * np.cos(theta),
                z=element.center_y_m * 1e3 + 1.15 * half_y * np.sin(theta),
                mode="lines",
                line=dict(
                    color="#06b6d4",
                    width=7
                    if element.kind in {"thick_lens", "double_pass_lens_relay"}
                    else 4,
                ),
                name="透镜",
                showlegend=False,
                hovertemplate=f"{element.name}<br>{element.kind}<br>元件尺寸为示意，偏心采用配置值<extra></extra>",
            )
            figure.add_scatter3d(
                x=[midpoint], y=[element.center_x_m * 1e3], z=[element.center_y_m * 1e3 + 1.25 * half_y], mode="text",
                text=[f"L{display_counts['L']}"], showlegend=False,
                hovertemplate=f"{element.name}<br>{element.kind}<extra></extra>",
            )
        elif element.kind == "mirror":
            display_counts["M"] = display_counts.get("M", 0) + 1
            figure.add_scatter3d(
                x=[midpoint, midpoint],
                y=[element.center_x_m * 1e3 - half_x, element.center_x_m * 1e3 + half_x],
                z=[element.center_y_m * 1e3 - half_y, element.center_y_m * 1e3 + half_y], mode="lines+text",
                line=dict(color="#a3a3a3", width=7), text=[None, f"M{display_counts['M']}"],
                name="反射镜", showlegend=False,
                hovertemplate=f"{element.name}<br>mirror<br>元件尺寸为示意，偏心采用配置值<extra></extra>",
            )
        elif element.kind in {"aperture", "interface", "coordinate_break", "tilt"}:
            prefixes = {"aperture": "A", "interface": "I", "coordinate_break": "C", "tilt": "T"}
            prefix = prefixes[element.kind]
            display_counts[prefix] = display_counts.get(prefix, 0) + 1
            figure.add_scatter3d(
                x=[midpoint], y=[element.center_x_m * 1e3], z=[element.center_y_m * 1e3], mode="markers+text",
                marker=dict(color=beam_color, size=5), text=[f"{prefix}{display_counts[prefix]}"],
                textposition="top center", showlegend=False,
                hovertemplate=f"{element.name}<br>{element.kind}<extra></extra>",
            )


def _crystal_outline_traces(geometry: dict[str, Any], grid: Any) -> list[go.Scatter3d]:
    center_x = float(geometry.get("center_x_m", 0.0)) * 1e3
    center_y = float(geometry.get("center_y_m", 0.0)) * 1e3
    if geometry.get("shape") == "cylinder":
        width = height = float(geometry["diameter_m"]) * 1e3
    else:
        width = float(geometry["width_x_m"]) * 1e3
        height = float(geometry["height_y_m"]) * 1e3
    x_min = center_x - 0.5 * width
    x_max = center_x + 0.5 * width
    y_min = center_y - 0.5 * height
    y_max = center_y + 0.5 * height
    z_min = float(grid.z_edges[0]) * 1e3
    z_max = float(grid.z_edges[-1]) * 1e3
    traces: list[go.Scatter3d] = []
    if geometry.get("shape") == "cylinder":
        radius = 0.5 * float(geometry["diameter_m"]) * 1e3
        theta = np.linspace(0.0, 2.0 * np.pi, 97)
        for z_value in (z_min, z_max):
            traces.append(
                go.Scatter3d(
                    x=center_x + radius * np.cos(theta), y=center_y + radius * np.sin(theta), z=np.full_like(theta, z_value),
                    mode="lines", line=dict(color="rgba(249,115,22,0.9)", width=4),
                    name="晶体边界", showlegend=z_value == z_min, hoverinfo="skip",
                )
            )
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            traces.append(
                go.Scatter3d(
                    x=[center_x + radius * np.cos(angle)] * 2, y=[center_y + radius * np.sin(angle)] * 2, z=[z_min, z_max],
                    mode="lines", line=dict(color="rgba(249,115,22,0.65)", width=2),
                    showlegend=False, hoverinfo="skip",
                )
            )
    else:
        vertices = [
            (x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_max, z_min), (x_min, y_max, z_min),
            (x_min, y_min, z_max), (x_max, y_min, z_max), (x_max, y_max, z_max), (x_min, y_max, z_max),
        ]
        edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
        for edge_index, (first, second) in enumerate(edges):
            traces.append(
                go.Scatter3d(
                    x=[vertices[first][0], vertices[second][0]],
                    y=[vertices[first][1], vertices[second][1]],
                    z=[vertices[first][2], vertices[second][2]],
                    mode="lines", line=dict(color="rgba(249,115,22,0.8)", width=3),
                    name="晶体边界", showlegend=edge_index == 0, hoverinfo="skip",
                )
            )
    return traces
