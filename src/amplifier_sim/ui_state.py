"""Small UI-state transforms kept independent from Streamlit for testing."""

from __future__ import annotations

import copy
import json
import math
from typing import Any


PATH_EDITOR_COLUMNS = (
    "type", "name", "distance_before_m", "length_m", "focal_length_m",
    "refractive_index", "group_refractive_index", "group_refractive_index_after",
    "group_delay_s", "thickness_m", "radius_front_m", "radius_back_m",
    "radius_x_m", "radius_y_m", "power_transmission", "power_reflectivity",
    "incidence_angle_x_deg", "incidence_angle_y_deg", "turn_angle_x_deg",
    "turn_angle_y_deg", "direction",
)


def path_editor_widget_prefix(
    beam_kind: str,
    config_generation: int,
    editor_version: int,
) -> str:
    """Return a path-editor key that cannot revive a prior import's state.

    Streamlit keeps widget values in the browser as well as in
    ``session_state``.  A hard JSON import clears the server-side path editor
    version, so a key based on ``editor_version`` alone can be reused and the
    browser may submit an old optical path on the next rerun.  Including the
    monotonically increasing configuration generation gives every imported
    configuration a distinct widget identity.
    """

    return f"ui_path_{beam_kind}_{int(config_generation)}_{int(editor_version)}"


def format_table_value(value: Any) -> str:
    """Return one Arrow-safe, human-readable diagnostic table cell.

    Streamlit converts dataframe columns through PyArrow.  A diagnostic
    column that contains both a scalar and a shape/list cannot be represented
    by one Arrow dtype and raises ``cannot mix list and non-list``.  Keeping
    this conversion in one tested helper also protects future result fields
    containing NumPy scalars, arrays, tuples or dictionaries.
    """

    if value is None:
        return "—"
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        value = value.tolist()
    elif hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        value = value.item()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)
    return str(value)


def stringify_table_column(rows: list[dict[str, Any]], column: str) -> list[dict[str, Any]]:
    """Copy rows while making one heterogeneous display column Arrow-safe."""

    return [
        {**row, column: format_table_value(row.get(column))}
        for row in rows
    ]


def clean_table_rows(table: Any, original_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Clean editable cells without discarding hidden optical-element data."""

    records = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
    cleaned: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
                continue
            row[str(key)] = value.item() if hasattr(value, "item") else value
        if original_rows is not None and index < len(original_rows):
            # ``column_order`` hides nested/advanced values from the editor.
            # Streamlit/Pandas may otherwise drop or coerce them during the
            # table round trip, e.g. turning a dispersion dict into a list.
            for key, value in original_rows[index].items():
                if key not in PATH_EDITOR_COLUMNS:
                    row[key] = copy.deepcopy(value)
        if row.get("type"):
            cleaned.append(row)
    return cleaned


def apply_quick_physics_config(config: dict[str, Any]) -> dict[str, Any]:
    """Apply the eight displayed physics choices without importing a solver profile."""
    result = copy.deepcopy(config)
    result["physics"].update({
        "optical_propagation": "hybrid_abcd_fft",
        "pump_propagation": "hybrid_abcd_fft",
        "population_dynamics": "transient",
        "pump_depletion": True,
        "gain_saturation": True,
        "thermal_model": "steady_3d",
        "refractive_index_model": "thermal_3d",
        "interface_model": "coating",
    })
    return result
