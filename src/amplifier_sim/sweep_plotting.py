"""Plot builders for slicing a two-parameter sweep along either axis."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go


def _axis_title(label: str, unit: str) -> str:
    return f"{label} ({unit})" if unit else label


def _legend_value(label: str, value: float, unit: str) -> str:
    suffix = f" {unit}" if unit else ""
    return f"{label} = {value:.6g}{suffix}"


def _finite_metric(record: Mapping[str, Any], metric_key: str) -> float | None:
    try:
        value = float(record[metric_key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def add_invalid_grid_markers(
    figure: go.Figure, records: Iterable[Mapping[str, Any]], *,
    x_field: str, y_field: str, x_scale: float, y_scale: float,
    metric_key: str, metric_scale: float, metric_unit: str,
) -> None:
    """Keep diagnostic heatmap values visibly distinct from accepted points."""
    points = [record for record in records if _finite_metric(record, metric_key) is not None]
    if not points:
        return
    figure.add_scatter(
        x=[float(record[x_field]) * x_scale for record in points],
        y=[float(record[y_field]) * y_scale for record in points],
        customdata=[[float(record[metric_key]) * metric_scale] for record in points],
        mode="markers", name="完成但无效（非定量）",
        marker={"symbol": "x", "size": 10, "color": "#d62728"},
        hovertemplate="完成但无效（非定量）<br>x=%{x:.6g}<br>y=%{y:.6g}<br>原始指标=%{customdata[0]:.6g} " + metric_unit + "<extra></extra>",
    )


def build_sweep_slice_figures(
    *,
    valid_records: Iterable[Mapping[str, Any]],
    invalid_records: Iterable[Mapping[str, Any]] = (),
    x_values: Sequence[float],
    y_values: Sequence[float],
    x_field: str,
    y_field: str,
    metric_key: str,
    x_label: str,
    x_unit: str,
    x_scale: float,
    y_label: str,
    y_unit: str,
    y_scale: float,
    metric_label: str,
    metric_unit: str,
    metric_scale: float,
    x_spacing: str = "linear",
    y_spacing: str = "linear",
    connect_invalid: bool = False,
) -> tuple[go.Figure, go.Figure]:
    """Return both families of thin-line slices through a 2-D sweep grid.

    The first figure scans x with one trace per fixed y.  The second scans y
    with one trace per fixed x.  Missing or invalid grid points remain gaps in
    valid lines; completed-but-invalid values are shown only as red crosses.
    """

    x_grid = np.asarray(x_values, dtype=float)
    y_grid = np.asarray(y_values, dtype=float)
    valid_lookup: dict[tuple[float, float], float] = {}
    for record in valid_records:
        metric_value = _finite_metric(record, metric_key)
        if metric_value is None:
            continue
        try:
            coordinate = (float(record[x_field]), float(record[y_field]))
        except (KeyError, TypeError, ValueError):
            continue
        valid_lookup[coordinate] = metric_value * metric_scale

    x_display = x_grid * x_scale
    y_display = y_grid * y_scale
    metric_axis_title = _axis_title(metric_label, metric_unit)
    scan_x = go.Figure()
    for fixed_y, fixed_y_display in zip(y_grid, y_display, strict=True):
        values = [valid_lookup.get((float(x), float(fixed_y))) for x in x_grid]
        if not any(value is not None for value in values):
            continue
        scan_x.add_scatter(
            x=x_display,
            y=values,
            mode="lines+markers",
            name=_legend_value(y_label, float(fixed_y_display), y_unit),
            line={"width": 1.15},
            marker={"size": 3.5},
            connectgaps=False,
            hovertemplate=(
                f"{x_label} %{{x:.6g}} {x_unit}<br>"
                f"{y_label} {fixed_y_display:.6g} {y_unit}<br>"
                f"{metric_label} %{{y:.6g}} {metric_unit}<extra></extra>"
            ),
        )

    scan_y = go.Figure()
    for fixed_x, fixed_x_display in zip(x_grid, x_display, strict=True):
        values = [valid_lookup.get((float(fixed_x), float(y))) for y in y_grid]
        if not any(value is not None for value in values):
            continue
        scan_y.add_scatter(
            x=y_display,
            y=values,
            mode="lines+markers",
            name=_legend_value(x_label, float(fixed_x_display), x_unit),
            line={"width": 1.15},
            marker={"size": 3.5},
            connectgaps=False,
            hovertemplate=(
                f"{y_label} %{{x:.6g}} {y_unit}<br>"
                f"{x_label} {fixed_x_display:.6g} {x_unit}<br>"
                f"{metric_label} %{{y:.6g}} {metric_unit}<extra></extra>"
            ),
        )

    invalid_lookup: dict[tuple[float, float], float] = {}
    invalid_x: list[float] = []
    invalid_y: list[float] = []
    invalid_metric: list[float] = []
    invalid_fixed_x: list[float] = []
    invalid_fixed_y: list[float] = []
    for record in invalid_records:
        metric_value = _finite_metric(record, metric_key)
        if metric_value is None:
            continue
        try:
            x_value = float(record[x_field]) * x_scale
            y_value = float(record[y_field]) * y_scale
        except (KeyError, TypeError, ValueError):
            continue
        invalid_x.append(x_value)
        invalid_y.append(y_value)
        invalid_metric.append(metric_value * metric_scale)
        invalid_fixed_x.append(x_value)
        invalid_fixed_y.append(y_value)
        invalid_lookup[(float(record[x_field]), float(record[y_field]))] = (
            metric_value * metric_scale
        )

    invalid_marker = {"symbol": "x", "size": 8, "color": "#d62728"}
    if invalid_metric and connect_invalid:
        for fixed_y, fixed_y_display in zip(y_grid, y_display, strict=True):
            values = [
                invalid_lookup.get((float(x), float(fixed_y))) for x in x_grid
            ]
            if not any(value is not None for value in values):
                continue
            scan_x.add_scatter(
                x=x_display,
                y=values,
                mode="lines+markers",
                name=(
                    _legend_value(y_label, float(fixed_y_display), y_unit)
                    + "（无效诊断）"
                ),
                line={"width": 1.0, "dash": "dot"},
                marker={"size": 4, "symbol": "x"},
                opacity=0.62,
                connectgaps=False,
            )
        for fixed_x, fixed_x_display in zip(x_grid, x_display, strict=True):
            values = [
                invalid_lookup.get((float(fixed_x), float(y))) for y in y_grid
            ]
            if not any(value is not None for value in values):
                continue
            scan_y.add_scatter(
                x=y_display,
                y=values,
                mode="lines+markers",
                name=(
                    _legend_value(x_label, float(fixed_x_display), x_unit)
                    + "（无效诊断）"
                ),
                line={"width": 1.0, "dash": "dot"},
                marker={"size": 4, "symbol": "x"},
                opacity=0.62,
                connectgaps=False,
            )
    elif invalid_metric:
        scan_x.add_scatter(
            x=invalid_x,
            y=invalid_metric,
            customdata=np.asarray(invalid_fixed_y)[:, None],
            mode="markers",
            name="完成但无效（不连线）",
            marker=invalid_marker,
            hovertemplate=(
                f"{x_label} %{{x:.6g}} {x_unit}<br>"
                f"{y_label} %{{customdata[0]:.6g}} {y_unit}<br>"
                f"{metric_label} %{{y:.6g}} {metric_unit}<extra></extra>"
            ),
        )
        scan_y.add_scatter(
            x=invalid_y,
            y=invalid_metric,
            customdata=np.asarray(invalid_fixed_x)[:, None],
            mode="markers",
            name="完成但无效（不连线）",
            marker=invalid_marker,
            hovertemplate=(
                f"{y_label} %{{x:.6g}} {y_unit}<br>"
                f"{x_label} %{{customdata[0]:.6g}} {x_unit}<br>"
                f"{metric_label} %{{y:.6g}} {metric_unit}<extra></extra>"
            ),
        )

    common_layout = {
        "yaxis_title": metric_axis_title,
        "height": 555,
        "margin": {"t": 70, "l": 70, "r": 20, "b": 195},
        "hovermode": "closest",
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.30,
            "xanchor": "left",
            "x": 0.0,
            "font": {"size": 10},
            "entrywidth": 245,
            "entrywidthmode": "pixels",
            "maxheight": 0.20,
        },
    }
    scan_x.update_layout(
        **common_layout,
        title={
            "text": f"固定 {y_label}：扫描 {x_label}",
            "x": 0.01,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "font": {"size": 18},
        },
        xaxis_title=_axis_title(x_label, x_unit),
        xaxis_type="log" if x_spacing == "log" else "linear",
    )
    scan_y.update_layout(
        **common_layout,
        title={
            "text": f"固定 {x_label}：扫描 {y_label}",
            "x": 0.01,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "font": {"size": 18},
        },
        xaxis_title=_axis_title(y_label, y_unit),
        xaxis_type="log" if y_spacing == "log" else "linear",
    )
    return scan_x, scan_y
