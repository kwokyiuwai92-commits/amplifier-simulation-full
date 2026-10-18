"""Import and calibrate external laser-material databases.

The simulator uses SI units.  This module deliberately keeps parsing and
calibration outside the propagation solvers so a malformed/unsupported
material file cannot disable manual configurations.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import re
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from .json_contract import loads_json_strict

MATERIAL_LIBRARY_SCHEMA = "amplifier-sim-material-library/v1"


_MATERIAL_CURVE_KEYS = (
    "pump_absorption_x_per_percent",
    "pump_absorption_y_per_percent",
    "emission_temperature_curve",
    "emission_reduction_curve",
    "reabsorption_temperature_curve",
    "reabsorption_reduction_curve",
)

_MATERIAL_SURFACE_KEYS = (
    "pump_absorption_surface_x_per_percent",
    "pump_absorption_surface_y_per_percent",
    "emission_surface",
    "reabsorption_surface",
)

_LIFETIME_TEMPERATURE_CURVE_KEY = "lifetime_temperature_curve"


_LIBRARY_MANAGED_PATHS = (
    "material.ion_density_m3",
    "material.refractive_index",
    "material.pump_refractive_index",
    "material.dn_dT_per_K",
    "material.n2_m2_W",
    "material.thermal.conductivity_W_mK",
    "material.thermal.density_kg_m3",
    "material.thermal.heat_capacity_J_kgK",
    "material.mechanical.young_modulus_Pa",
    "material.mechanical.thermal_expansion_per_K",
    "material.mechanical.poisson_ratio",
    "material.dispersion.sellmeier",
    "material.dispersion.reference_wavelength_m",
    "energy_levels.model",
    "energy_levels.pump_absorption_cross_section_m2",
    "energy_levels.pump_emission_cross_section_m2",
    "energy_levels.signal_absorption_cross_section_m2",
    "energy_levels.signal_emission_cross_section_m2",
    "energy_levels.upper_state_lifetime_s",
    "energy_levels.upper_state_radiative_lifetime_s",
    "energy_levels.pump_quantum_efficiency",
    "energy_levels.pump_cross_section_spectrum",
    "energy_levels.signal_cross_section_spectrum",
)


def _path_value(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


def _set_path_value(mapping: dict[str, Any], path: str, value: Any) -> None:
    components = path.split(".")
    target = mapping
    for component in components[:-1]:
        target = target.setdefault(component, {})
    target[components[-1]] = copy.deepcopy(value)


def _portable_values_equal(left: Any, right: Any) -> bool:
    try:
        return json.dumps(
            left,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) == json.dumps(
            right,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
        except (TypeError, ValueError):
            return False


def _parameter_axis_allowed(
    record: dict[str, Any], path: str, axis: str
) -> tuple[bool, str | None]:
    """Return whether a provenance-qualified anisotropic value may activate."""

    constraints = record.get("activation_constraints", {})
    parameters = (
        constraints.get("parameters", {})
        if isinstance(constraints, dict)
        else {}
    )
    entry = parameters.get(path, {}) if isinstance(parameters, dict) else {}
    if not isinstance(entry, dict) or not entry:
        return True, None
    status = str(entry.get("status", "runtime_ready"))
    allowed_axes = entry.get("allowed_axes")
    quarantined_axes = set(map(str, entry.get("quarantined_axes", [])))
    allowed = status != "quarantined" and axis not in quarantined_axes
    if allowed_axes is not None:
        allowed = allowed and axis in set(map(str, allowed_axes))
    return allowed, None if allowed else str(
        entry.get(
            "reason",
            f"{path} axis {axis} is quarantined pending independent provenance",
        )
    )


def _quarantined_rate_system_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Expose an unusable imported network without making it activatable."""

    raw = record.get("rate_system", {})
    constraints = record.get("activation_constraints", {})
    entry = (
        constraints.get("rate_system", {})
        if isinstance(constraints, dict)
        else {}
    )
    if not isinstance(entry, dict) or str(entry.get("status", "")) != "quarantined":
        return copy.deepcopy(raw)
    return {
        "usable": False,
        "status": str(entry.get("code", "quarantined")),
        "reason": str(entry.get("reason", "Imported rate system is quarantined.")),
        "action_required": str(
            entry.get(
                "action_required",
                "Independently validate the source before defining a custom runtime network.",
            )
        ),
        "quarantined_source_level_names": list(raw.get("level_names", [])),
    }


def _managed_defaults() -> dict[str, Any]:
    # Imported lazily to keep material parsing independent from configuration
    # module initialisation.
    from .config import DEFAULTS

    return {
        path: copy.deepcopy(_path_value(DEFAULTS, path))
        for path in _LIBRARY_MANAGED_PATHS
    }


def _legacy_applied_fields(library: dict[str, Any]) -> dict[str, Any]:
    snapshot = library.get("applied_parameter_snapshot", {})
    if not isinstance(snapshot, dict):
        return {}
    names = {
        "ion_density_m3": "material.ion_density_m3",
        "refractive_index": "material.refractive_index",
        "pump_refractive_index": "material.pump_refractive_index",
        "dn_dT_per_K": "material.dn_dT_per_K",
        "pump_absorption_cross_section_m2": (
            "energy_levels.pump_absorption_cross_section_m2"
        ),
        "pump_emission_cross_section_m2": (
            "energy_levels.pump_emission_cross_section_m2"
        ),
        "signal_absorption_cross_section_m2": (
            "energy_levels.signal_absorption_cross_section_m2"
        ),
        "signal_emission_cross_section_m2": (
            "energy_levels.signal_emission_cross_section_m2"
        ),
        "upper_state_lifetime_s": "energy_levels.upper_state_lifetime_s",
        "pump_quantum_efficiency": "energy_levels.pump_quantum_efficiency",
    }
    return {
        path: copy.deepcopy(snapshot[name])
        for name, path in names.items()
        if name in snapshot
    }


def _library_user_overrides(
    config: dict[str, Any], prior_library: dict[str, Any]
) -> dict[str, Any]:
    applied = prior_library.get("applied_fields")
    legacy_snapshot = not isinstance(applied, dict)
    if legacy_snapshot:
        applied = _legacy_applied_fields(prior_library)
    overrides: dict[str, Any] = {}
    for path, previous in applied.items():
        if path not in _LIBRARY_MANAGED_PATHS:
            continue
        if not legacy_snapshot:
            # New snapshots require an explicit provenance transition. The UI
            # performs it through reconcile_material_field_provenance when a
            # user edits a material-owned field.
            continue
        if legacy_snapshot and path not in {
            "energy_levels.pump_emission_cross_section_m2"
        }:
            # Older snapshots did not carry a provenance source. Preserve the
            # one legacy scalar whose established contract already treated a
            # post-snapshot difference as an explicit override; all other
            # fields require the new user_override marker.
            continue
        current = _path_value(config, path)
        if not _portable_values_equal(current, previous):
            overrides[path] = copy.deepcopy(current)
    prior_provenance = prior_library.get("field_provenance", {})
    if isinstance(prior_provenance, dict):
        for path, entry in prior_provenance.items():
            if (
                path in _LIBRARY_MANAGED_PATHS
                and isinstance(entry, dict)
                and entry.get("source") == "user_override"
            ):
                overrides[path] = copy.deepcopy(_path_value(config, path))
    return overrides


def reconcile_material_field_provenance(config: dict[str, Any]) -> None:
    """Mark post-application form edits as explicit material overrides.

    The function intentionally mutates only provenance metadata and its
    comparison snapshot. It is used by interactive editors after users change
    a material-owned field; physics values themselves are never modified.
    """

    library = config.get("material", {}).get("library", {})
    if not isinstance(library, dict):
        return
    applied = library.get("applied_fields")
    provenance = library.get("field_provenance")
    if not isinstance(applied, dict) or not isinstance(provenance, dict):
        return
    for path in _LIBRARY_MANAGED_PATHS:
        if path not in applied:
            continue
        current = copy.deepcopy(_path_value(config, path))
        entry = provenance.get(path, {})
        if (
            not _portable_values_equal(current, applied[path])
            or (
                isinstance(entry, dict)
                and entry.get("source") == "user_override"
            )
        ):
            provenance[path] = {
                "source": "user_override",
                "record_sha256": None,
            }
            applied[path] = current


def material_provenance_mismatches(config: dict[str, Any]) -> tuple[str, ...]:
    """Return library-owned paths changed without an override transition."""

    library = config.get("material", {}).get("library", {})
    if not isinstance(library, dict):
        return ()
    applied = library.get("applied_fields")
    provenance = library.get("field_provenance")
    if not isinstance(applied, dict) or not isinstance(provenance, dict):
        return ()
    mismatches: list[str] = []
    for path, entry in provenance.items():
        if (
            path in _LIBRARY_MANAGED_PATHS
            and isinstance(entry, dict)
            and entry.get("source") == "material_library_record"
            and (
                path not in applied
                or not _portable_values_equal(_path_value(config, path), applied[path])
            )
        ):
            mismatches.append(path)
    return tuple(mismatches)


def _numbers(value: str | None, *, field_name: str = "numeric field") -> list[float]:
    if not value:
        return []
    result: list[float] = []
    for item in str(value).replace(",", " ").split():
        try:
            number = float(item)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} contains non-numeric token {item!r}"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(f"{field_name} contains non-finite value {item!r}")
        result.append(number)
    return result


def _scalar(values: list[float], default: float | None = None) -> float | None:
    return float(values[0]) if values else default


def _clean_notes(value: str) -> str:
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def _field_map(element: ET.Element) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    for child in element:
        description = child.attrib.get("description", "").strip()
        fields.setdefault(description, []).append(child.attrib.get("value", ""))
    return fields


def _one(fields: dict[str, list[str]], description: str, default: str = "") -> str:
    values = fields.get(description, [])
    return values[0] if values else default


def _last(fields: dict[str, list[str]], description: str, default: str = "") -> str:
    values = fields.get(description, [])
    return values[-1] if values else default


def _curve(x: list[float], y: list[float], *, x_scale: float = 1.0, y_scale: float = 1.0) -> dict[str, list[float]]:
    if not x or not y:
        return {}
    if len(x) != len(y):
        raise ValueError(
            f"material curve coordinate/value lengths differ ({len(x)} != {len(y)})"
        )
    if len(x) < 2:
        return {}
    pairs = sorted(
        (float(x[index]) * x_scale, float(y[index]) * y_scale)
        for index in range(len(x))
    )
    # Several records in the bundled Commercial App source contain a repeated wavelength
    # at the join between two measured spectrum blocks.  The normalized JSON
    # library historically represents those joins by their arithmetic mean.
    # Consolidate only the XML parser's raw coordinate/value pairs here; strict
    # normalized-JSON validation still rejects duplicate axes so hand-authored
    # ambiguity is never hidden.
    consolidated: list[tuple[float, float]] = []
    start = 0
    while start < len(pairs):
        stop = start + 1
        while stop < len(pairs) and pairs[stop][0] == pairs[start][0]:
            stop += 1
        consolidated.append(
            (
                pairs[start][0],
                float(np.mean([value for _, value in pairs[start:stop]])),
            )
        )
        start = stop
    # A constant coordinate does not define an interpolable curve.  This is
    # how the bundled normalized library represents the corresponding legacy
    # Commercial App fields as well.
    if len(consolidated) < 2:
        return {}
    return {
        "x": [coordinate for coordinate, _ in consolidated],
        "y": [value for _, value in consolidated],
    }


def _surface(
    values: list[float],
    temperatures: list[float],
    wavelengths: list[float],
    enabled: bool,
    value_scale: float,
) -> dict[str, Any]:
    if not enabled or len(temperatures) < 1 or len(wavelengths) < 2:
        return {}
    expected = len(temperatures) * len(wavelengths)
    if len(values) != expected:
        raise ValueError(
            "material surface value count does not match temperature/wavelength axes"
        )
    temperature_order = np.argsort(np.asarray(temperatures, dtype=float))
    wavelength_order = np.argsort(np.asarray(wavelengths, dtype=float))
    sorted_temperatures = np.asarray(temperatures, dtype=float)[temperature_order]
    sorted_wavelengths = np.asarray(wavelengths, dtype=float)[wavelength_order]
    if np.any(np.diff(sorted_temperatures) <= 0.0) or np.any(
        np.diff(sorted_wavelengths) <= 0.0
    ):
        raise ValueError("material surface axes must contain unique coordinates")
    matrix = np.asarray(values, dtype=float).reshape(
        len(temperatures), len(wavelengths)
    )
    matrix = matrix[temperature_order][:, wavelength_order]
    return {
        "temperature_K": sorted_temperatures.tolist(),
        "wavelength_m": (sorted_wavelengths * 1e-9).tolist(),
        "values": (matrix * value_scale).tolist(),
    }


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"material field {name} must contain only numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"material field {name} contains non-finite values")
    return result


def _normalize_curve_record(curve: dict[str, Any], *, name: str) -> None:
    if not curve:
        return
    x = _finite_array(curve.get("x", []), name=f"{name}.x")
    y = _finite_array(curve.get("y", []), name=f"{name}.y")
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError(
            f"material curve {name} needs equal one-dimensional x/y arrays with at least two points"
        )
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if np.any(np.diff(x) <= 0.0):
        raise ValueError(f"material curve {name} coordinates must be unique")
    curve["x"] = x.tolist()
    curve["y"] = y.tolist()


def _normalize_surface_record(surface: dict[str, Any], *, name: str) -> None:
    if not surface:
        return
    temperatures = _finite_array(
        surface.get("temperature_K", []), name=f"{name}.temperature_K"
    )
    wavelengths = _finite_array(
        surface.get("wavelength_m", []), name=f"{name}.wavelength_m"
    )
    values = _finite_array(surface.get("values", []), name=f"{name}.values")
    if temperatures.ndim != 1 or wavelengths.ndim != 1:
        raise ValueError(f"material surface {name} axes must be one-dimensional")
    if len(temperatures) < 1 or len(wavelengths) < 2:
        raise ValueError(
            f"material surface {name} needs at least one temperature and two wavelengths"
        )
    if values.shape != (len(temperatures), len(wavelengths)):
        raise ValueError(
            f"material surface {name} values must have shape "
            f"({len(temperatures)}, {len(wavelengths)})"
        )
    temperature_order = np.argsort(temperatures)
    wavelength_order = np.argsort(wavelengths)
    temperatures = temperatures[temperature_order]
    wavelengths = wavelengths[wavelength_order]
    values = values[temperature_order][:, wavelength_order]
    if np.any(np.diff(temperatures) <= 0.0) or np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError(f"material surface {name} axes must contain unique coordinates")
    if np.any(temperatures <= 0.0) or np.any(wavelengths <= 0.0):
        raise ValueError(f"material surface {name} axes must be positive")
    surface["temperature_K"] = temperatures.tolist()
    surface["wavelength_m"] = wavelengths.tolist()
    surface["values"] = values.tolist()


def _validate_and_normalize_material_record(record: dict[str, Any]) -> None:
    spectroscopy = record.get("spectroscopy", {})
    if spectroscopy is not None and not isinstance(spectroscopy, dict):
        raise ValueError("material spectroscopy must be an object")
    spectroscopy = spectroscopy or {}
    for key in _MATERIAL_CURVE_KEYS:
        curve = spectroscopy.get(key, {})
        if curve is not None and not isinstance(curve, dict):
            raise ValueError(f"material curve spectroscopy.{key} must be an object")
        _normalize_curve_record(curve or {}, name=f"spectroscopy.{key}")
    for key in _MATERIAL_SURFACE_KEYS:
        surface = spectroscopy.get(key, {})
        if surface is not None and not isinstance(surface, dict):
            raise ValueError(f"material surface spectroscopy.{key} must be an object")
        _normalize_surface_record(surface or {}, name=f"spectroscopy.{key}")

    doping = record.get("doping", {})
    if doping is not None and not isinstance(doping, dict):
        raise ValueError("material doping must be an object")
    doping = doping or {}
    lifetime_temperature_curve = doping.get(_LIFETIME_TEMPERATURE_CURVE_KEY, {})
    if lifetime_temperature_curve is not None and not isinstance(
        lifetime_temperature_curve, dict
    ):
        raise ValueError(
            f"material curve doping.{_LIFETIME_TEMPERATURE_CURVE_KEY} must be an object"
        )
    _normalize_curve_record(
        lifetime_temperature_curve or {},
        name=f"doping.{_LIFETIME_TEMPERATURE_CURVE_KEY}",
    )
    reference_lifetime = doping.get("reference_lifetime_s")
    if reference_lifetime is not None and (
        not math.isfinite(float(reference_lifetime))
        or float(reference_lifetime) <= 0.0
    ):
        raise ValueError("material doping.reference_lifetime_s must be positive and finite")
    concentration = _finite_array(
        doping.get("lifetime_percent", []), name="doping.lifetime_percent"
    )
    lifetime = _finite_array(
        doping.get("lifetime_s", []), name="doping.lifetime_s"
    )
    if concentration.size or lifetime.size:
        if (
            concentration.ndim != 1
            or lifetime.ndim != 1
            or concentration.shape != lifetime.shape
            or len(concentration) < 1
        ):
            raise ValueError(
                "material lifetime concentration/value arrays must have equal one-dimensional lengths"
            )
        if np.any(concentration <= 0.0) or np.any(lifetime <= 0.0):
            raise ValueError("material lifetime concentration/value arrays must be positive")
        order = np.argsort(concentration)
        concentration = concentration[order]
        lifetime = lifetime[order]
        if len(concentration) > 1 and np.any(np.diff(concentration) <= 0.0):
            raise ValueError("material lifetime concentrations must be unique")
        doping["lifetime_percent"] = concentration.tolist()
        doping["lifetime_s"] = lifetime.tolist()


def _parse_commercial_app_material(element: ET.Element) -> dict[str, Any]:
    fields = _field_map(element)
    nums = lambda description: _numbers(
        _one(fields, description), field_name=description
    )
    last_nums = lambda description: _numbers(
        _last(fields, description), field_name=description
    )

    ion_density_per_percent = (_scalar(nums("Doping density [ions/mm^3] for 1% doping concentration"), 0.0) or 0.0) * 1e9
    lifetime_s = nums("Spontaneous decay rate [seconds]")
    lifetime_percent = nums("Doping concentration corresponding to spontaneous decay rate [%]")
    lifetime_temperature_K = nums(
        "Temperature corresponding to spontaneous decay rate [K]"
    )
    lifetime_temperature_curve = _curve(lifetime_temperature_K, lifetime_s)

    absorption_x = _curve(
        nums("Corresponding wavelength of spectrum of aborption"),
        nums("Spectrum of aborption [1/mm] for 1% doping concentration"),
        x_scale=1e-9,
        y_scale=1e3,
    )
    absorption_y = _curve(
        nums("Corresponding wavelength of spectrum of aborption for y-polarization"),
        nums("Spectrum of aborption [1/mm] for 1% doping concentration and y-polarization"),
        x_scale=1e-9,
        y_scale=1e3,
    )

    emission_values = last_nums("Stimulated emission cross section [mm^2]")
    emission_temperatures = nums("temperature corresponding to stimulated emission cross section")
    emission_temperature_curve = _curve(emission_temperatures, emission_values, y_scale=1e-6)
    emission_reduction_curve = _curve(
        nums("wavelength corresponding to reduction stimulated emission cross section"),
        nums("reduction stimulated emission cross section"),
        x_scale=1e-9,
    )
    reabsorption_temperature_curve = _curve(
        nums("temperature corresponding to reabsorption cross section"),
        nums("reabsorption cross section [mm^2]"),
        y_scale=1e-6,
    )
    reabsorption_reduction_curve = _curve(
        nums("wavelength corresponding to reduction reabsorption cross section"),
        nums("reduction reabsorption cross section"),
        x_scale=1e-9,
    )

    emission_surface = _surface(
        nums("Big stimulated emission cross section [mm^2]"),
        nums("temperature corresponding to big stimulated emission cross section"),
        nums("wavelength corresponding to big stimulated emission cross section"),
        bool(_scalar(nums("stimulated emission cross section is given as gig data"), 0.0)),
        1e-6,
    )
    reabsorption_surface = _surface(
        nums("Big reabsorption cross section [mm^2]"),
        nums("temperature corresponding to big reabsorption cross section"),
        nums("wavelength corresponding to big reabsorption cross section"),
        bool(_scalar(nums("reabsorption cross section is given as gig data"), 0.0)),
        1e-6,
    )
    absorption_surface_x = _surface(
        nums("Big absorption [1/mm]"),
        nums("temperature corresponding to big absorption"),
        nums("wavelength corresponding to big absorption"),
        bool(_scalar(nums("absorption is given as gig data"), 0.0)),
        1e3,
    )
    absorption_surface_y = _surface(
        nums("Big absorption Y-polarization [1/mm]"),
        nums("temperature corresponding to big absorption Y-polarization"),
        nums("wavelength corresponding to big absorption Y-polarization"),
        bool(_scalar(nums("absorption is given as gig data"), 0.0)),
        1e3,
    )

    b = [_scalar(nums(f"B{i} of Sellmeier equation"), 0.0) or 0.0 for i in range(1, 4)]
    c = [_scalar(nums(f"C{i} of Sellmeier equation"), 0.0) or 0.0 for i in range(1, 4)]
    conductivity = [(_scalar(nums(f"Thermal conductivity in {axis}-dir. [W/mmK]"), 0.0) or 0.0) * 1e3 for axis in "xyz"]
    young = [(_scalar(nums(f"Elastic modulus in {axis}-dir. [N/mm^2]"), 0.0) or 0.0) * 1e6 for axis in "xyz"]
    expansion = [(_scalar(nums(f"Coefficient of thermal expansion in {axis}-dir. [1/K]"), 0.0) or 0.0) for axis in "xyz"]
    refractive = [(_scalar(nums(f"Refractive Index ({axis}-polarization)"), 0.0) or 0.0) for axis in "xyz"]

    level_count = int(_scalar(nums("number of levels in system of rate equations"), 0.0) or 0)
    level_names = [_one(fields, f"name of level {index}") for index in range(level_count)]
    record: dict[str, Any] = {
        "schema": MATERIAL_LIBRARY_SCHEMA,
        "name": element.attrib.get("description", element.tag),
        "source": {
            "format": "COMMERCIAL_APP_material_xml",
            "record_id": element.tag,
            "notes": _clean_notes(_one(fields, "notes of laser")),
        },
        "doping": {
            "reference_percent": 1.0,
            "ion_density_per_percent_m3": ion_density_per_percent,
            "absorption_at_reference_per_percent_m_inv": (_scalar(nums("Absorption coefficient [1/mm] for 1% doping concentration"), 0.0) or 0.0) * 1e3,
            "lifetime_percent": lifetime_percent,
            "lifetime_s": lifetime_s if lifetime_percent else [],
            "reference_lifetime_s": (
                float(lifetime_s[0]) if lifetime_temperature_curve else None
            ),
            "lifetime_temperature_curve": lifetime_temperature_curve,
        },
        "optical": {
            "refractive_index_xyz": refractive,
            "pump_refractive_index": _scalar(nums("Refractive index pumplight"), None),
            "dn_dT_xyz_per_K": [
                _scalar(nums("Temperature dependence of refractive index dn/dT in x-dir. [1/K]"), 0.0) or 0.0,
                _scalar(nums("Temperature dependence of refractive index dn/dT in y-dir. [1/K]"), 0.0) or 0.0,
                _scalar(nums("Temperature dependence of refractive index dn/dT in z-dir. [1/K]"), 0.0) or 0.0,
            ],
            "n2_m2_W": (_scalar(nums("non linear index of Kerr effect [mm²/W]"), 0.0) or 0.0) * 1e-6,
            "sellmeier": {"enabled": bool(_scalar(nums("dispersion by Sellmeier equation"), 0.0)), "B": b, "C_um2": c},
        },
        "thermal": {
            "conductivity_xyz_W_mK": conductivity,
            "density_kg_m3": (_scalar(nums("Density [g/mm^3]."), 0.0) or 0.0) * 1e6,
            "heat_capacity_J_kgK": (_scalar(nums("Specific heat capacity ((0.59 for Nd:YAG) [J·g^-1·K^-1])."), 0.0) or 0.0) * 1e3,
            "reference_temperature_K": _scalar(nums("Reference temperature for material data [K]"), 293.15),
            "default_heat_fraction": _scalar(nums("Default value of heat efficiency of pump source"), None),
            "pump_quantum_efficiency": _scalar(nums("Quantum efficiency for pump light absorption in rate equations"), None),
        },
        "mechanical": {
            "young_modulus_xyz_Pa": young,
            "poisson_xy_yz_xz": [
                _scalar(nums("Poisson ratio in xy plane"), 0.0) or 0.0,
                _scalar(nums("Poisson ratio in yz plane"), 0.0) or 0.0,
                _scalar(nums("Poisson ratio in xz plane"), 0.0) or 0.0,
            ],
            "shear_modulus_xy_yz_xz_Pa": [
                (_scalar(nums(f"shear modulus in {plane} plane"), 0.0) or 0.0) * 1e6 for plane in ("xy", "yz", "xz")
            ],
            "thermal_expansion_xyz_per_K": expansion,
            "photoelastic_6x6": nums("photoelsatic constants"),
            "isotropic": bool(_scalar(nums("isotropic material"), 0.0)),
        },
        "spectroscopy": {
            "anisotropic_absorption": bool(_scalar(nums("Anisotropic aborption"), 0.0)),
            "pump_absorption_x_per_percent": absorption_x,
            "pump_absorption_y_per_percent": absorption_y,
            "pump_absorption_surface_x_per_percent": absorption_surface_x,
            "pump_absorption_surface_y_per_percent": absorption_surface_y,
            "emission_cross_section_m2": (_scalar(nums("Stimulated emission cross section [mm^2]"), 0.0) or 0.0) * 1e-6,
            "reabsorption_cross_section_m2": (_scalar(nums("Reabsorption cross-section at laser wavelength [mm^2]"), 0.0) or 0.0) * 1e-6,
            "emission_temperature_curve": emission_temperature_curve,
            "emission_reduction_curve": emission_reduction_curve,
            "emission_spectrum_type": _one(fields, "type of wavelength dependence of emission cross-section", "constantSpectrum"),
            "reabsorption_temperature_curve": reabsorption_temperature_curve,
            "reabsorption_reduction_curve": reabsorption_reduction_curve,
            "reabsorption_spectrum_type": _one(fields, "type of wavelength dependence of reabsorption cross-section", "constantSpectrum"),
            "emission_surface": emission_surface,
            "reabsorption_surface": reabsorption_surface,
            "quasi_three_level": bool(_scalar(nums("Material shows quasi-three level behaviour"), 0.0)),
            "stark_manifolds_enabled": bool(
                _scalar(nums("Material descriptions includes laser manifolds (energs levels)"), 0.0)
            ),
            "lower_stark_energy_cm_inv": nums(
                "List of Stark levels of lower laser manifold [cm^-1]"
            ),
            "lower_stark_degeneracy": nums(
                "Degeneracy of Stark levels of lower laser manifold"
            ),
            "upper_stark_energy_cm_inv": nums(
                "List of Stark levels of upper laser manifold [cm^-1]"
            ),
            "upper_stark_degeneracy": nums(
                "Degeneracy of Stark levels of upper laser manifold"
            ),
            "signal_lower_stark_index": int(
                _scalar(
                    nums("Stark level of lower laser manifold which is used for laser transition [cm^-1]"),
                    -1.0,
                )
                or 0
            ),
            "signal_upper_stark_index": int(
                _scalar(
                    nums("Stark level of upper laser manifold which is used for laser transition [cm^-1]"),
                    -1.0,
                )
                or 0
            ),
            "emission_temperature_dependent": bool(
                _scalar(nums("stimulated emission cross section is temperature dependent given"), 0.0)
            ),
            "emission_calculated_from_stark_levels": bool(
                _scalar(nums("stimulated emission cross section calculated by energy levels"), 0.0)
            ),
            "reabsorption_temperature_dependent": bool(
                _scalar(nums("reabsorption cross section is temperature dependent given"), 0.0)
            ),
            "reabsorption_calculated_from_stark_levels": bool(
                _scalar(nums("reabsorption cross section calculated by energy levels"), 0.0)
            ),
            "effective_fluorescence_wavelength_enabled": bool(
                _scalar(nums("is effective fluorescence wavelength set"), 0.0)
            ),
            "effective_fluorescence_wavelength_m": (_scalar(nums("effective fluorescence wavelength related to decay rate"), 0.0) or 0.0) * 1e-9,
        },
        "rate_system": {
            "level_names": level_names,
            "level_energy_cm_inv": nums("energy of level [1/cm]"),
            "level_doping_index": nums("number of doping on level"),
            "doping_density_m3": [value * 1e9 for value in nums("doping density [ions/mm³]")],
            "doping_count": int(_scalar(nums("number of dopings in material"), 0.0) or 0),
            "rate_type": nums("type rate line"),
            "rate_from": nums("I level rate line"),
            "rate_to": nums("J level rate line"),
            "rate_parameter": nums("parameter of rate line"),
            "heat_fraction": nums("fraction of heat"),
        },
    }
    return record


def load_material_library(data: bytes | str | Path, *, format_hint: str | None = None) -> list[dict[str, Any]]:
    """Load normalized JSON or Commercial App-style XML records without touching config."""

    if isinstance(data, Path):
        payload = data.read_bytes()
        format_hint = format_hint or data.suffix.lower().lstrip(".")
    elif isinstance(data, bytes):
        payload = data
    elif isinstance(data, str) and not data.lstrip().startswith(("<", "{", "[")) and Path(data).exists():
        path = Path(data)
        payload = path.read_bytes()
        format_hint = format_hint or path.suffix.lower().lstrip(".")
    else:
        payload = str(data).encode("utf-8")
    text = payload.decode("utf-8-sig")
    kind = (format_hint or ("xml" if text.lstrip().startswith("<") else "json")).lower()
    if kind == "xml":
        root = ET.fromstring(text)
        container = root.find("list_of_materials") if root.tag != "list_of_materials" else root
        if container is None:
            raise ValueError("XML does not contain list_of_materials")
        records = [_parse_commercial_app_material(element) for element in container]
    elif kind == "json":
        decoded = loads_json_strict(text)
        if isinstance(decoded, dict) and "materials" in decoded:
            records = decoded["materials"]
        elif isinstance(decoded, list):
            records = decoded
        elif isinstance(decoded, dict):
            records = [decoded]
        else:
            raise ValueError("Material JSON must contain an object, list, or materials list")
    else:
        raise ValueError("Material library must be XML or JSON")
    if not records or not all(isinstance(record, dict) and record.get("name") for record in records):
        raise ValueError("Material library contains no named records")
    for record in records:
        _validate_and_normalize_material_record(record)
    return records


def _interpolate_curve(
    curve: dict[str, Any],
    coordinate: float,
    default: float,
    *,
    out_of_range: str = "error",
    name: str = "material_curve",
    audit: list[dict[str, Any]] | None = None,
) -> float:
    x = np.asarray(curve.get("x", []), dtype=float)
    y = np.asarray(curve.get("y", []), dtype=float)
    if len(x) < 1 or len(x) != len(y):
        return float(default)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if len(x) > 1 and np.any(np.diff(x) <= 0.0):
        raise ValueError("material curve coordinates must be unique")
    if out_of_range not in {"error", "edge", "zero"}:
        raise ValueError("material curve out_of_range must be error, edge, or zero")
    requested = float(coordinate)
    outside = requested < float(x[0]) or requested > float(x[-1])
    if audit is not None:
        audit.append(
            {
                "name": name,
                "coordinate": requested,
                "minimum": float(x[0]),
                "maximum": float(x[-1]),
                "out_of_range": outside,
                "policy": out_of_range,
            }
        )
    if outside and out_of_range == "error":
        raise ValueError(
            f"{name} was requested at {requested:.9g} outside its calibrated "
            f"range [{float(x[0]):.9g}, {float(x[-1]):.9g}]."
        )
    if out_of_range == "zero":
        left = right = 0.0
    else:
        left = float(y[0])
        right = float(y[-1])
    return float(np.interp(requested, x, y, left=left, right=right))


def _surface_at_temperature(
    surface: dict[str, Any],
    temperature_K: float,
    *,
    name: str,
    out_of_range: str = "error",
    audit: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    temperatures = np.asarray(surface.get("temperature_K", []), dtype=float)
    wavelengths = np.asarray(surface.get("wavelength_m", []), dtype=float)
    values = np.asarray(surface.get("values", []), dtype=float)
    if values.shape != (len(temperatures), len(wavelengths)) or len(temperatures) < 1 or len(wavelengths) < 2:
        return None
    temperature_order = np.argsort(temperatures)
    wavelength_order = np.argsort(wavelengths)
    temperatures = temperatures[temperature_order]
    wavelengths = wavelengths[wavelength_order]
    values = values[temperature_order][:, wavelength_order]
    if np.any(np.diff(wavelengths) <= 0.0) or (
        len(temperatures) > 1 and np.any(np.diff(temperatures) <= 0.0)
    ):
        raise ValueError("material surface axes must contain unique coordinates")
    requested = float(temperature_K)
    outside = requested < float(temperatures[0]) or requested > float(temperatures[-1])
    if audit is not None:
        audit.append(
            {
                "name": f"{name}.temperature_K",
                "coordinate": requested,
                "minimum": float(temperatures[0]),
                "maximum": float(temperatures[-1]),
                "out_of_range": outside,
                "policy": out_of_range,
            }
        )
    if outside and out_of_range == "error":
        raise ValueError(
            f"{name} was requested at {requested:.9g} K outside its calibrated "
            f"temperature range [{float(temperatures[0]):.9g}, "
            f"{float(temperatures[-1]):.9g}] K."
        )
    if out_of_range not in {"error", "edge"}:
        raise ValueError("material surface temperature policy must be error or edge")
    if len(temperatures) == 1:
        return wavelengths, values[0]
    result = np.array([
        np.interp(float(temperature_K), temperatures, values[:, index], left=values[0, index], right=values[-1, index])
        for index in range(len(wavelengths))
    ])
    return wavelengths, result


def _signal_curve(
    record: dict[str, Any],
    temperature_K: float,
    *,
    snapshot_out_of_range: str = "error",
    audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    spectroscopy = record.get("spectroscopy", {})
    emission_surface = _surface_at_temperature(
        spectroscopy.get("emission_surface", {}),
        temperature_K,
        name="spectroscopy.emission_surface",
        out_of_range=snapshot_out_of_range,
        audit=audit,
    )
    absorption_surface = _surface_at_temperature(
        spectroscopy.get("reabsorption_surface", {}),
        temperature_K,
        name="spectroscopy.reabsorption_surface",
        out_of_range=snapshot_out_of_range,
        audit=audit,
    )
    emission_shape = spectroscopy.get("emission_reduction_curve", {})
    absorption_shape = spectroscopy.get("reabsorption_reduction_curve", {})
    emission_is_numerical = str(spectroscopy.get("emission_spectrum_type", "constantSpectrum")) != "constantSpectrum"
    absorption_is_numerical = str(spectroscopy.get("reabsorption_spectrum_type", "constantSpectrum")) != "constantSpectrum"
    grids = [
        item[0]
        for item in (emission_surface, absorption_surface)
        if item is not None
    ]
    if (
        emission_surface is None
        and emission_is_numerical
        and len(emission_shape.get("x", [])) >= 2
    ):
        grids.append(np.asarray(emission_shape["x"], dtype=float))
    if (
        absorption_surface is None
        and absorption_is_numerical
        and len(absorption_shape.get("x", [])) >= 2
    ):
        grids.append(np.asarray(absorption_shape["x"], dtype=float))
    if not grids:
        return {}
    wavelength = np.unique(np.concatenate(grids))
    emission_base = (
        None
        if emission_surface is not None
        else _interpolate_curve(
            spectroscopy.get("emission_temperature_curve", {}),
            temperature_K,
            float(spectroscopy.get("emission_cross_section_m2", 0.0)),
            out_of_range=snapshot_out_of_range,
            name="spectroscopy.emission_temperature_curve",
            audit=audit,
        )
    )
    absorption_base = (
        None
        if absorption_surface is not None
        else _interpolate_curve(
            spectroscopy.get("reabsorption_temperature_curve", {}),
            temperature_K,
            float(spectroscopy.get("reabsorption_cross_section_m2", 0.0)),
            out_of_range=snapshot_out_of_range,
            name="spectroscopy.reabsorption_temperature_curve",
            audit=audit,
        )
    )
    emission = np.interp(
        wavelength,
        *emission_surface,
        left=0.0,
        right=0.0,
    ) if emission_surface is not None else (
        np.array(
            [
                _interpolate_curve(
                    emission_shape,
                    value,
                    0.0,
                    out_of_range="zero",
                    name="spectroscopy.emission_reduction_curve",
                )
                for value in wavelength
            ]
        )
        * float(emission_base)
        if emission_is_numerical
        else np.full_like(wavelength, float(emission_base))
    )
    absorption = np.interp(
        wavelength,
        *absorption_surface,
        left=0.0,
        right=0.0,
    ) if absorption_surface is not None else (
        np.array(
            [
                _interpolate_curve(
                    absorption_shape,
                    value,
                    0.0,
                    out_of_range="zero",
                    name="spectroscopy.reabsorption_reduction_curve",
                )
                for value in wavelength
            ]
        )
        * float(absorption_base)
        if absorption_is_numerical
        else np.full_like(wavelength, float(absorption_base))
    )
    return {
        "wavelength_m": wavelength.tolist(),
        "sigma_abs_m2": absorption.tolist(),
        "sigma_em_m2": emission.tolist(),
        "out_of_range": "zero",
    }


def _pump_curve(
    record: dict[str, Any],
    temperature_K: float,
    polarization: str,
    ion_density_per_percent: float,
    *,
    snapshot_out_of_range: str = "error",
    audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    spectroscopy = record.get("spectroscopy", {})
    suffix = "y" if polarization == "y" and spectroscopy.get("anisotropic_absorption") else "x"
    surface = _surface_at_temperature(
        spectroscopy.get(
            f"pump_absorption_surface_{suffix}_per_percent", {}
        ),
        temperature_K,
        name=f"spectroscopy.pump_absorption_surface_{suffix}_per_percent",
        out_of_range=snapshot_out_of_range,
        audit=audit,
    )
    if surface is not None:
        wavelength, alpha = surface
    else:
        curve = spectroscopy.get(f"pump_absorption_{suffix}_per_percent", {})
        wavelength = np.asarray(curve.get("x", []), dtype=float)
        alpha = np.asarray(curve.get("y", []), dtype=float)
    if len(wavelength) < 2 or len(wavelength) != len(alpha) or ion_density_per_percent <= 0.0:
        return {}
    sigma_abs = np.maximum(alpha, 0.0) / ion_density_per_percent
    return {
        "wavelength_m": wavelength.tolist(),
        "sigma_abs_m2": sigma_abs.tolist(),
        "sigma_em_m2": np.zeros_like(sigma_abs).tolist(),
        "out_of_range": "zero",
    }


def _absolute_cross_section_response(
    surface: dict[str, Any],
    *,
    scale: float = 1.0,
) -> dict[str, Any] | None:
    """Convert a normalized material-library surface to a runtime response law."""

    temperatures = np.asarray(surface.get("temperature_K", []), dtype=float)
    wavelengths = np.asarray(surface.get("wavelength_m", []), dtype=float)
    values = np.asarray(surface.get("values", []), dtype=float)
    if (
        len(temperatures) < 2
        or len(wavelengths) < 2
        or values.shape != (len(temperatures), len(wavelengths))
        or not np.all(np.isfinite(values))
        or scale <= 0.0
    ):
        return None
    temperature_order = np.argsort(temperatures)
    wavelength_order = np.argsort(wavelengths)
    temperatures = temperatures[temperature_order]
    wavelengths = wavelengths[wavelength_order]
    values = values[temperature_order][:, wavelength_order]
    if np.any(np.diff(temperatures) <= 0.0) or np.any(
        np.diff(wavelengths) <= 0.0
    ):
        raise ValueError("material response surface axes must be unique")
    return {
        "model": "absolute_temperature_wavelength_table",
        "temperature_K": temperatures.tolist(),
        "wavelength_m": wavelengths.tolist(),
        "values_m2": np.maximum(values / scale, 0.0).tolist(),
        "source": "material_library_temperature_wavelength_surface",
    }


def _temperature_scale_response(
    curve: dict[str, Any],
    *,
    reference_temperature_K: float,
    normalization_value: float | None = None,
    source: str = "material_library_separable_temperature_curve",
) -> dict[str, Any] | None:
    """Convert a separable absolute-vs-temperature curve to a relative law."""

    temperatures = np.asarray(curve.get("x", []), dtype=float)
    values = np.asarray(curve.get("y", []), dtype=float)
    if (
        len(temperatures) < 2
        or temperatures.shape != values.shape
        or not np.all(np.isfinite(temperatures))
        or not np.all(np.isfinite(values))
        or np.any(temperatures <= 0.0)
        or np.any(values < 0.0)
    ):
        return None
    order = np.argsort(temperatures)
    temperatures = temperatures[order]
    values = values[order]
    reference_value = (
        float(normalization_value)
        if normalization_value is not None
        else float(
            np.interp(
                reference_temperature_K,
                temperatures,
                values,
                left=values[0],
                right=values[-1],
            )
        )
    )
    if reference_value <= 0.0:
        return None
    return {
        "model": "scale_table",
        "coordinate": temperatures.tolist(),
        "scale": (values / reference_value).tolist(),
        "source": source,
    }


def _stark_temperature_response(
    spectroscopy: dict[str, Any],
    *,
    manifold: str,
    reference_temperature_K: float,
) -> dict[str, Any] | None:
    """Build an exact Boltzmann occupation law for one laser manifold."""

    energies = np.asarray(
        spectroscopy.get(f"{manifold}_stark_energy_cm_inv", []), dtype=float
    )
    degeneracies = np.asarray(
        spectroscopy.get(f"{manifold}_stark_degeneracy", []), dtype=float
    )
    selected = int(spectroscopy.get(f"signal_{manifold}_stark_index", -1))
    if (
        not spectroscopy.get("stark_manifolds_enabled", False)
        or len(energies) < 1
        or energies.shape != degeneracies.shape
        or not np.all(np.isfinite(energies))
        or not np.all(np.isfinite(degeneracies))
        or np.any(degeneracies <= 0.0)
        or selected < 0
        or selected >= len(energies)
    ):
        return None
    return {
        "model": "boltzmann_stark_fraction",
        "energy_cm_inv": energies.tolist(),
        "degeneracy": degeneracies.tolist(),
        "selected_level_index": selected,
        "reference_temperature_K": float(reference_temperature_K),
        "source": "material_library_stark_boltzmann_occupation",
    }


def _material_spectroscopic_response(
    record: dict[str, Any],
    *,
    density_per_percent_m3: float,
    doping_percent: float,
    polarization: str,
    reference_temperature_K: float,
    reference_density_m3: float,
    reference_lifetime_s: float,
) -> tuple[dict[str, Any], list[str]]:
    """Build optional response ports from data actually present in a record.

    The returned laws are data, not an activation request.  ``apply_material_record``
    preserves the user's current ``mode`` so importing a material cannot silently
    turn a formerly one-way simulation into a thermal-spectroscopic fixed point.
    """

    spectroscopy = record.get("spectroscopy", {})
    doping = record.get("doping", {})
    suffix = "y" if polarization == "y" and spectroscopy.get("anisotropic_absorption") else "x"
    cross_sections: dict[str, dict[str, Any]] = {
        "pump_absorption": {"model": "off"},
        "pump_emission": {"model": "off"},
        "signal_absorption": {"model": "off"},
        "signal_emission": {"model": "off"},
    }
    imported: list[str] = []

    stark_laws = {
        "signal_absorption": (
            bool(spectroscopy.get("reabsorption_calculated_from_stark_levels", False)),
            _stark_temperature_response(
                spectroscopy,
                manifold="lower",
                reference_temperature_K=reference_temperature_K,
            ),
        ),
        "signal_emission": (
            bool(spectroscopy.get("emission_calculated_from_stark_levels", False)),
            _stark_temperature_response(
                spectroscopy,
                manifold="upper",
                reference_temperature_K=reference_temperature_K,
            ),
        ),
    }
    for runtime_key, (enabled, law) in stark_laws.items():
        if enabled and law is not None:
            cross_sections[runtime_key] = law
            imported.append(f"{runtime_key}(T) Stark-Boltzmann occupation")

    pump_law = _absolute_cross_section_response(
        spectroscopy.get(f"pump_absorption_surface_{suffix}_per_percent", {}),
        scale=density_per_percent_m3,
    )
    if pump_law is not None:
        cross_sections["pump_absorption"] = pump_law
        imported.append("pump_absorption(T, wavelength)")

    for runtime_key, source_key in (
        ("signal_absorption", "reabsorption_surface"),
        ("signal_emission", "emission_surface"),
    ):
        if cross_sections[runtime_key].get("model") != "off":
            continue
        law = _absolute_cross_section_response(spectroscopy.get(source_key, {}))
        if law is not None:
            cross_sections[runtime_key] = law
            imported.append(f"{runtime_key}(T, wavelength)")

    # Some Commercial App records store sigma(T) and a separate normalised wavelength
    # shape.  That is a real separable model, not a two-dimensional measured
    # surface.  Preserve the distinction by applying sigma(T)/sigma(Tref) as a
    # scale to the already imported wavelength curve.
    for runtime_key, curve_key in (
        ("signal_absorption", "reabsorption_temperature_curve"),
        ("signal_emission", "emission_temperature_curve"),
    ):
        if cross_sections[runtime_key].get("model") != "off":
            continue
        law = _temperature_scale_response(
            spectroscopy.get(curve_key, {}),
            reference_temperature_K=reference_temperature_K,
        )
        if law is not None:
            cross_sections[runtime_key] = law
            imported.append(f"{runtime_key}(T) separable scale")

    concentration_law: dict[str, Any] = {"model": "off"}
    concentration = np.asarray(doping.get("lifetime_percent", []), dtype=float)
    lifetimes = np.asarray(doping.get("lifetime_s", []), dtype=float)
    if (
        density_per_percent_m3 > 0.0
        and reference_lifetime_s > 0.0
        and len(concentration) >= 2
        and concentration.shape == lifetimes.shape
        and np.all(np.isfinite(concentration))
        and np.all(np.isfinite(lifetimes))
        and np.all(concentration > 0.0)
        and np.all(lifetimes > 0.0)
    ):
        order = np.argsort(concentration)
        concentration_reference_lifetime = float(
            np.interp(doping_percent, concentration[order], lifetimes[order])
        )
        concentration_law = {
            "model": "scale_table",
            "coordinate": (concentration[order] * density_per_percent_m3).tolist(),
            "scale": (lifetimes[order] / concentration_reference_lifetime).tolist(),
            "source": "material_library_lifetime_vs_doping",
        }
        imported.append("upper_state_lifetime(concentration)")

    temperature_law = _temperature_scale_response(
        doping.get(_LIFETIME_TEMPERATURE_CURVE_KEY, {}),
        reference_temperature_K=reference_temperature_K,
        source="material_library_lifetime_vs_temperature",
    )
    if temperature_law is None:
        temperature_law = {"model": "off"}
    else:
        curve_metadata = doping.get(_LIFETIME_TEMPERATURE_CURVE_KEY, {})
        response_metadata: dict[str, Any] = {}
        for key in ("citation", "doi", "measurement_wavelength_m", "data_quality"):
            if key in curve_metadata:
                response_metadata[key] = copy.deepcopy(curve_metadata[key])
        if response_metadata:
            temperature_law["extras"] = response_metadata
        imported.append("upper_state_lifetime(T)")

    return {
        "reference_temperature_K": float(reference_temperature_K),
        "reference_density_m3": max(float(reference_density_m3), np.finfo(float).tiny),
        "cross_sections": cross_sections,
        "upper_state_lifetime": {
            "temperature": temperature_law,
            "concentration": concentration_law,
        },
        "data_origin": "material_library",
        "available_data": imported,
        "reference_doping_percent": float(doping_percent),
    }, imported


def apply_material_record(
    config: dict[str, Any],
    record: dict[str, Any],
    *,
    doping_percent: float = 1.0,
    temperature_K: float | None = None,
    polarization: str = "x",
    pump_polarization: str = "x",
    thermomechanical_axis: str = "x",
    use_heat_fraction: bool = False,
    use_sellmeier: bool = True,
    use_spectroscopy: bool = True,
    snapshot_out_of_range: str = "error",
) -> tuple[dict[str, Any], list[str]]:
    """Return a calibrated config; unsupported fields are retained as metadata."""

    record = copy.deepcopy(record)
    _validate_and_normalize_material_record(record)
    # Raw measured libraries may include negative noise/background samples.
    # Keep those records inspectable, but never silently clamp them into a
    # different physical cross-section law when activating spectroscopy.
    if use_spectroscopy:
        spectroscopy_data = record.get("spectroscopy", {}) or {}
        for key in _MATERIAL_CURVE_KEYS:
            curve = spectroscopy_data.get(key, {}) or {}
            if np.any(np.asarray(curve.get("x", []), dtype=float) <= 0.0):
                raise ValueError(f"spectroscopy.{key} coordinates must be positive")
            if np.any(np.asarray(curve.get("y", []), dtype=float) < 0.0):
                raise ValueError(f"spectroscopy.{key} values must be non-negative before activation")
        for key in _MATERIAL_SURFACE_KEYS:
            surface = spectroscopy_data.get(key, {}) or {}
            if np.any(np.asarray(surface.get("values", []), dtype=float) < 0.0):
                raise ValueError(f"spectroscopy.{key} values must be non-negative before activation")
    if doping_percent <= 0.0 or not math.isfinite(doping_percent):
        raise ValueError("doping_percent must be positive")
    for name, axis_name in (
        ("polarization", polarization),
        ("pump_polarization", pump_polarization),
        ("thermomechanical_axis", thermomechanical_axis),
    ):
        if axis_name not in {"x", "y", "z"}:
            raise ValueError(f"{name} must be x, y, or z")
    if snapshot_out_of_range not in {"error", "edge"}:
        raise ValueError("snapshot_out_of_range must be error or edge")
    doping_count = int(record.get("rate_system", {}).get("doping_count", 0))
    if doping_count > 1:
        raise ValueError(
            f"Material {record.get('name', '<unnamed>')!r} contains "
            f"{doping_count} dopant species and is metadata-only. A scalar "
            "single-density materialization would be physically invalid; "
            "define an explicit multi-ion custom rate network instead."
        )
    result = copy.deepcopy(config)
    material = result["material"]
    levels = result["energy_levels"]
    doping = record.get("doping", {})
    optical = record.get("optical", {})
    thermal = record.get("thermal", {})
    mechanical = record.get("mechanical", {})
    spectroscopy = record.get("spectroscopy", {})
    warnings: list[str] = []
    imported_response_names: list[str] = []
    snapshot_range_contract: list[dict[str, Any]] = []
    record_digest = hashlib.sha256(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    prior_library = material.get("library", {})
    preserved_response_mode = str(
        material.get("spectroscopic_response", {}).get("mode", "off")
    )
    user_overrides = _library_user_overrides(result, prior_library)
    prior_record_digest = prior_library.get("record_sha256")
    same_library_record = bool(
        prior_record_digest is not None
        and str(prior_record_digest) == record_digest
    )
    prior_snapshot = prior_library.get("applied_parameter_snapshot", {})
    prior_snapshot_qe = prior_snapshot.get("pump_quantum_efficiency")
    current_qe = float(levels.get("pump_quantum_efficiency", 1.0))
    source_qe = thermal.get("pump_quantum_efficiency")
    prior_snapshot_pump_emission = prior_snapshot.get(
        "pump_emission_cross_section_m2"
    )
    current_pump_emission = float(
        levels.get("pump_emission_cross_section_m2", 0.0)
    )
    manual_qe_override = bool(
        same_library_record
        and prior_snapshot_qe is not None
        and not math.isclose(
            current_qe,
            float(prior_snapshot_qe),
            rel_tol=0.0,
            abs_tol=64.0 * np.finfo(float).eps,
        )
    )
    declared_qe_override = bool(
        same_library_record
        and source_qe is not None
        and not math.isclose(
            current_qe,
            float(source_qe),
            rel_tol=0.0,
            abs_tol=64.0 * np.finfo(float).eps,
        )
    )
    preserve_existing_qe = manual_qe_override or declared_qe_override
    previous_material_name = str(
        prior_library.get("material_name", material.get("name", ""))
    )
    different_material_record = bool(
        (prior_record_digest is not None and not same_library_record)
        or (
            prior_record_digest is None
            and previous_material_name
            and previous_material_name != str(record["name"])
        )
    )
    if different_material_record:
        if levels.get("upper_state_radiative_lifetime_s") is not None:
            levels["upper_state_radiative_lifetime_s"] = None
            warnings.append(
                "The explicit upper-state radiative lifetime belonged to the "
                "previous material and was cleared instead of being silently "
                "reused for a different library record."
            )
        if (
            material.get("thermal", {}).get("heat_fraction")
            == "lifetime_quantum_defect"
        ):
            material["thermal"]["heat_fraction"] = "quantum_defect"
            material["thermal"]["extra_heat_fraction"] = 0.0
            warnings.append(
                "Lifetime-decomposed heating was changed to quantum_defect "
                "because the new material has no independently verified "
                "radiative lifetime. Enter tau_rad and re-enable the lifetime "
                "model only when that datum is available."
            )
        if (
            use_spectroscopy
            and prior_snapshot_pump_emission is not None
            and math.isclose(
                current_pump_emission,
                float(prior_snapshot_pump_emission),
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        ):
            levels["pump_emission_cross_section_m2"] = 0.0
            warnings.append(
                "The new material record contains Pump absorption but no Pump "
                "stimulated-emission data. The value traced to the previous "
                "material was cleared to zero; enter a measured sigma_pe if "
                "Pump stimulated emission is relevant."
            )
        if prior_library:
            defaults = _managed_defaults()
            for path, default in defaults.items():
                if path not in user_overrides:
                    _set_path_value(result, path, default)
            # The response tables belong to the library record, whereas
            # activating them is a user/solver choice. Replace the former and
            # retain only the latter across a material switch.
            from .config import DEFAULTS

            material["spectroscopic_response"] = copy.deepcopy(
                DEFAULTS["material"]["spectroscopic_response"]
            )
            material["spectroscopic_response"]["mode"] = (
                preserved_response_mode
            )
            if user_overrides:
                warnings.append(
                    "Explicit user overrides detected from differences against "
                    "the previous material snapshot were preserved; all other "
                    "library-owned fields were reset before applying the new record."
                )
    temperature = float(temperature_K if temperature_K is not None else thermal.get("reference_temperature_K", 293.15))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_K must be positive and finite")
    data_reference_temperature = float(thermal.get("reference_temperature_K", temperature))
    signal_axis = {"x": 0, "y": 1, "z": 2}[polarization]
    pump_axis = {"x": 0, "y": 1, "z": 2}[pump_polarization]
    mechanical_axis = {"x": 0, "y": 1, "z": 2}[
        thermomechanical_axis
    ]

    refractive_allowed, refractive_block_reason = _parameter_axis_allowed(
        record, "optical.refractive_index_xyz", polarization
    )
    if (
        len(optical.get("refractive_index_xyz", [])) > signal_axis
        and not refractive_allowed
    ):
        raise ValueError(
            f"Material {record['name']!r} cannot activate optical.refractive_index_xyz "
            f"on axis {polarization}: {refractive_block_reason}"
        )
    young_allowed, young_block_reason = _parameter_axis_allowed(
        record, "mechanical.young_modulus_xyz_Pa", thermomechanical_axis
    )
    if (
        len(mechanical.get("young_modulus_xyz_Pa", [])) > mechanical_axis
        and not young_allowed
    ):
        raise ValueError(
            f"Material {record['name']!r} cannot activate mechanical.young_modulus_xyz_Pa "
            f"on axis {thermomechanical_axis}: {young_block_reason}"
        )

    material["name"] = str(record["name"])
    density_per_percent = float(doping.get("ion_density_per_percent_m3", 0.0))
    if density_per_percent > 0.0:
        material["ion_density_m3"] = density_per_percent * float(doping_percent)
    else:
        warnings.append("The material record has no positive ion density per 1%; the existing ion density was retained.")
    refractive = optical.get("refractive_index_xyz", [])
    dndt = optical.get("dn_dT_xyz_per_K", [])
    selected_dndt = (
        float(dndt[signal_axis])
        if len(dndt) > signal_axis
        else float(material.get("dn_dT_per_K", 0.0))
    )
    pump_dndt = float(dndt[pump_axis]) if len(dndt) > pump_axis else 0.0
    index_temperature_offset = selected_dndt * (temperature - data_reference_temperature)
    pump_index_temperature_offset = pump_dndt * (
        temperature - data_reference_temperature
    )
    if len(refractive) > signal_axis and float(refractive[signal_axis]) > 0.0:
        material["refractive_index"] = (
            float(refractive[signal_axis]) + index_temperature_offset
        )
    if len(dndt) > signal_axis:
        material["dn_dT_per_K"] = float(dndt[signal_axis])
    material["material_data_reference_temperature_K"] = data_reference_temperature
    material["reference_temperature_K"] = temperature
    material["operating_temperature_K"] = temperature
    material["sellmeier_temperature_index_offset"] = index_temperature_offset
    material["n2_m2_W"] = float(optical.get("n2_m2_W", material.get("n2_m2_W", 0.0)))
    if optical.get("pump_refractive_index") not in (None, 0):
        material["pump_refractive_index"] = (
            float(optical["pump_refractive_index"])
            + pump_index_temperature_offset
        )
    elif different_material_record:
        # ``None`` invokes the documented material-index fallback.  Retaining
        # a wavelength-specific index from a different crystal is never a
        # physically meaningful fallback.
        material["pump_refractive_index"] = None

    conductivity = thermal.get("conductivity_xyz_W_mK", [])
    if len(conductivity) == 3 and min(map(float, conductivity)) > 0.0:
        material["thermal"]["conductivity_W_mK"] = list(map(float, conductivity))
    for source, target in (("density_kg_m3", "density_kg_m3"), ("heat_capacity_J_kgK", "heat_capacity_J_kgK")):
        if float(thermal.get(source, 0.0)) > 0.0:
            material["thermal"][target] = float(thermal[source])
    if use_heat_fraction and thermal.get("default_heat_fraction") is not None:
        heat_fraction = float(thermal["default_heat_fraction"])
        if 0.0 <= heat_fraction <= 1.0:
            material["thermal"]["heat_fraction"] = heat_fraction
        else:
            warnings.append("The database heat fraction lies outside [0,1] and was not applied.")

    young = mechanical.get("young_modulus_xyz_Pa", [])
    expansion = mechanical.get("thermal_expansion_xyz_per_K", [])
    poisson = mechanical.get("poisson_xy_yz_xz", [])
    if len(young) > mechanical_axis and float(young[mechanical_axis]) > 0.0:
        material["mechanical"]["young_modulus_Pa"] = float(
            young[mechanical_axis]
        )
    if len(expansion) > mechanical_axis:
        material["mechanical"]["thermal_expansion_per_K"] = float(
            expansion[mechanical_axis]
        )
    poisson_allowed, poisson_block_reason = _parameter_axis_allowed(
        record, "mechanical.poisson_xy_yz_xz", thermomechanical_axis
    )
    if len(poisson) >= 3 and poisson_allowed:
        transverse_pairs = ((0, 2), (0, 1), (1, 2))[mechanical_axis]
        material["mechanical"]["poisson_ratio"] = float(np.mean([poisson[index] for index in transverse_pairs]))
    elif len(poisson) >= 3:
        warnings.append(
            "The imported anisotropic Poisson ratios were retained as quarantined "
            f"metadata and not materialized: {poisson_block_reason}"
        )

    lifetime_curve = _curve(doping.get("lifetime_percent", []), doping.get("lifetime_s", []))
    if lifetime_curve:
        levels["upper_state_lifetime_s"] = _interpolate_curve(
            lifetime_curve,
            doping_percent,
            levels["upper_state_lifetime_s"],
            out_of_range=snapshot_out_of_range,
            name="doping.upper_state_lifetime_vs_percent",
            audit=snapshot_range_contract,
        )
    elif float(doping.get("reference_lifetime_s", 0.0) or 0.0) > 0.0:
        levels["upper_state_lifetime_s"] = float(doping["reference_lifetime_s"])
    elif doping.get("lifetime_s"):
        levels["upper_state_lifetime_s"] = float(doping["lifetime_s"][0])

    lifetime_temperature_curve = doping.get(_LIFETIME_TEMPERATURE_CURVE_KEY, {})
    if lifetime_temperature_curve:
        temperature_lifetime = _interpolate_curve(
            lifetime_temperature_curve,
            temperature,
            levels["upper_state_lifetime_s"],
            out_of_range=snapshot_out_of_range,
            name="doping.lifetime_temperature_curve",
            audit=snapshot_range_contract,
        )
        if lifetime_curve:
            # The concentration table belongs to the material-data reference
            # temperature. Combine the two separable calibrations there, then
            # normalize each dynamic law at the selected snapshot coordinate.
            reference_temperature_lifetime = _interpolate_curve(
                lifetime_temperature_curve,
                data_reference_temperature,
                levels["upper_state_lifetime_s"],
                out_of_range=snapshot_out_of_range,
                name="doping.lifetime_temperature_curve.reference",
                audit=snapshot_range_contract,
            )
            if reference_temperature_lifetime <= 0.0:
                raise ValueError("reference-temperature upper-state lifetime must be positive")
            levels["upper_state_lifetime_s"] *= (
                temperature_lifetime / reference_temperature_lifetime
            )
        else:
            levels["upper_state_lifetime_s"] = temperature_lifetime

    pump_curve: dict[str, Any] = {}
    signal_curve: dict[str, Any] = {}
    if use_spectroscopy:
        # A library application is a replacement of the wavelength tables, not
        # a sparse merge with the previously selected crystal.  The supported
        # library contract contains only Pump absorption data.  Carry the
        # separately configured scalar sigma_pe across the new wavelength grid
        # rather than making the runtime curve silently disagree with it.
        levels["pump_cross_section_spectrum"] = {}
        levels["signal_cross_section_spectrum"] = {}
        pump_curve = _pump_curve(
            record,
            temperature,
            pump_polarization,
            density_per_percent,
            snapshot_out_of_range=snapshot_out_of_range,
            audit=snapshot_range_contract,
        )
        signal_curve = _signal_curve(
            record,
            temperature,
            snapshot_out_of_range=snapshot_out_of_range,
            audit=snapshot_range_contract,
        )
        if pump_curve:
            pump_curve["sigma_em_m2"] = [
                float(levels["pump_emission_cross_section_m2"])
            ] * len(pump_curve["wavelength_m"])
            if float(levels["pump_emission_cross_section_m2"]) > 0.0:
                warnings.append(
                    "The material library provides wavelength-dependent Pump "
                    "absorption only. The explicit scalar sigma_pe was retained "
                    "as a constant across the imported Pump spectrum."
                )
            levels["pump_cross_section_spectrum"] = pump_curve
            levels["pump_absorption_cross_section_m2"] = _interpolate_curve(
                {"x": pump_curve["wavelength_m"], "y": pump_curve["sigma_abs_m2"]},
                float(result["pump"]["wavelength_m"]),
                levels["pump_absorption_cross_section_m2"],
                out_of_range=snapshot_out_of_range,
                name="pump_cross_section_snapshot",
                audit=snapshot_range_contract,
            )
            levels["pump_emission_cross_section_m2"] = _interpolate_curve(
                {
                    "x": pump_curve["wavelength_m"],
                    "y": pump_curve["sigma_em_m2"],
                },
                float(result["pump"]["wavelength_m"]),
                0.0,
                out_of_range=snapshot_out_of_range,
                name="pump_emission_cross_section_snapshot",
                audit=snapshot_range_contract,
            )
        elif float(doping.get("absorption_at_reference_per_percent_m_inv", 0.0)) > 0.0 and density_per_percent > 0.0:
            levels["pump_absorption_cross_section_m2"] = float(doping["absorption_at_reference_per_percent_m_inv"]) / density_per_percent
        else:
            warnings.append(
                "The material record has no usable Pump absorption spectrum or "
                "reference absorption calibration; the existing explicit "
                "sigma_pa was retained."
            )
        if signal_curve:
            levels["signal_cross_section_spectrum"] = signal_curve
            levels["signal_absorption_cross_section_m2"] = _interpolate_curve(
                {"x": signal_curve["wavelength_m"], "y": signal_curve["sigma_abs_m2"]},
                float(result["seed"]["wavelength_m"]),
                levels["signal_absorption_cross_section_m2"],
                out_of_range=snapshot_out_of_range,
                name="signal_absorption_cross_section_snapshot",
                audit=snapshot_range_contract,
            )
            levels["signal_emission_cross_section_m2"] = _interpolate_curve(
                {"x": signal_curve["wavelength_m"], "y": signal_curve["sigma_em_m2"]},
                float(result["seed"]["wavelength_m"]),
                levels["signal_emission_cross_section_m2"],
                out_of_range=snapshot_out_of_range,
                name="signal_emission_cross_section_snapshot",
                audit=snapshot_range_contract,
            )
        else:
            if spectroscopy.get("emission_cross_section_m2") is None and not spectroscopy.get("emission_temperature_curve"):
                warnings.append(
                    "The material record has no Signal emission cross section; "
                    "the existing explicit sigma_se was retained."
                )
            else:
                levels["signal_emission_cross_section_m2"] = _interpolate_curve(
                    spectroscopy.get("emission_temperature_curve", {}),
                    temperature,
                    float(spectroscopy.get("emission_cross_section_m2") or 0.0),
                    out_of_range=snapshot_out_of_range,
                    name="spectroscopy.emission_temperature_curve",
                    audit=snapshot_range_contract,
                )
            if spectroscopy.get("reabsorption_cross_section_m2") is None and not spectroscopy.get("reabsorption_temperature_curve"):
                warnings.append(
                    "The material record has no Signal reabsorption cross "
                    "section; the existing explicit sigma_sa was retained."
                )
            else:
                levels["signal_absorption_cross_section_m2"] = _interpolate_curve(
                    spectroscopy.get("reabsorption_temperature_curve", {}),
                    temperature,
                    float(spectroscopy.get("reabsorption_cross_section_m2") or 0.0),
                    out_of_range=snapshot_out_of_range,
                    name="spectroscopy.reabsorption_temperature_curve",
                    audit=snapshot_range_contract,
                )

        existing_response = material.get("spectroscopic_response", {})
        imported_response, imported_response_names = _material_spectroscopic_response(
            record,
            density_per_percent_m3=density_per_percent,
            doping_percent=float(doping_percent),
            polarization=pump_polarization,
            reference_temperature_K=temperature,
            reference_density_m3=float(material.get("ion_density_m3", 0.0)),
            reference_lifetime_s=float(levels["upper_state_lifetime_s"]),
        )
        # Response data and activation are intentionally separate.  Preserve
        # the user's mode, but replace stale library-derived tables whenever a
        # different material record is applied.
        imported_response["mode"] = str(existing_response.get("mode", "off"))
        material["spectroscopic_response"] = imported_response
    else:
        from .config import DEFAULTS

        material["spectroscopic_response"] = copy.deepcopy(
            DEFAULTS["material"]["spectroscopic_response"]
        )
        material["spectroscopic_response"]["mode"] = preserved_response_mode
        levels["pump_cross_section_spectrum"] = {}
        levels["signal_cross_section_spectrum"] = {}

    sellmeier = copy.deepcopy(optical.get("sellmeier", {}))
    if use_sellmeier and sellmeier.get("enabled"):
        material["dispersion"]["sellmeier"] = sellmeier
        material["dispersion"]["reference_wavelength_m"] = float(result["seed"]["wavelength_m"])
    else:
        material["dispersion"].pop("sellmeier", None)

    quantum_efficiency = source_qe
    if quantum_efficiency is not None:
        if 0.0 <= float(quantum_efficiency) <= 1.0:
            if preserve_existing_qe:
                warnings.append(
                    "The same material record was reapplied, so the existing "
                    "explicit Pump excitation-efficiency override was preserved "
                    f"at {current_qe:.8g}; the source value "
                    f"{float(quantum_efficiency):.8g} remains provenance only."
                )
            else:
                levels["pump_quantum_efficiency"] = float(quantum_efficiency)
        else:
            warnings.append(
                "The database pump quantum efficiency is outside [0,1]; it was retained as metadata only (it may represent effective cross-relaxation yield)."
            )
    rate_system = record.get("rate_system", {})
    level_count = len(rate_system.get("level_names", []))
    rate_lines_present = any(
        rate_system.get(key)
        for key in ("rate_type", "rate_from", "rate_to", "rate_parameter")
    )
    if spectroscopy.get("quasi_three_level") and doping_count <= 1:
        levels["model"] = "two_level"
    if rate_lines_present and not spectroscopy.get("quasi_three_level"):
        warnings.append(
            f"The record contains a {level_count}-level proprietary rate-line system; scalar spectroscopic data were applied, but the rate-line network was retained as metadata and was not auto-activated."
        )
    if mechanical.get("photoelastic_6x6") and material["mechanical"].get("stress_model", "none") != "none":
        warnings.append(
            "The 6x6 strain-optic tensor was retained as metadata; the selected scalar stress proxy cannot use it as a Pa^-1 coefficient."
        )

    effective_wavelength_raw = float(
        spectroscopy.get("effective_fluorescence_wavelength_m", 0.0) or 0.0
    )
    effective_flag_present = "effective_fluorescence_wavelength_enabled" in spectroscopy
    effective_wavelength_enabled = bool(
        spectroscopy.get("effective_fluorescence_wavelength_enabled", False)
    )
    if effective_wavelength_raw > 0.0 and not effective_flag_present:
        warnings.append(
            "The material record contains an effective fluorescence wavelength but no explicit enabled flag; "
            "it was retained as metadata and disabled for heating until explicitly enabled."
        )

    library_written_paths: set[str] = set()
    if density_per_percent > 0.0:
        library_written_paths.add("material.ion_density_m3")
    if (
        len(refractive) > signal_axis
        and float(refractive[signal_axis]) > 0.0
    ):
        library_written_paths.add("material.refractive_index")
    if len(dndt) > signal_axis:
        library_written_paths.add("material.dn_dT_per_K")
    if "n2_m2_W" in optical:
        library_written_paths.add("material.n2_m2_W")
    if optical.get("pump_refractive_index") not in (None, 0):
        library_written_paths.add("material.pump_refractive_index")
    if len(conductivity) == 3 and min(map(float, conductivity)) > 0.0:
        library_written_paths.add("material.thermal.conductivity_W_mK")
    if float(thermal.get("density_kg_m3", 0.0)) > 0.0:
        library_written_paths.add("material.thermal.density_kg_m3")
    if float(thermal.get("heat_capacity_J_kgK", 0.0)) > 0.0:
        library_written_paths.add("material.thermal.heat_capacity_J_kgK")
    if (
        len(young) > mechanical_axis
        and float(young[mechanical_axis]) > 0.0
        and young_allowed
    ):
        library_written_paths.add("material.mechanical.young_modulus_Pa")
    if len(expansion) > mechanical_axis:
        library_written_paths.add(
            "material.mechanical.thermal_expansion_per_K"
        )
    if len(poisson) >= 3 and poisson_allowed:
        library_written_paths.add("material.mechanical.poisson_ratio")
    if lifetime_curve or float(
        doping.get("reference_lifetime_s", 0.0) or 0.0
    ) > 0.0 or doping.get("lifetime_s"):
        library_written_paths.add("energy_levels.upper_state_lifetime_s")
    if use_spectroscopy:
        library_written_paths.update(
            {
                "energy_levels.pump_cross_section_spectrum",
                "energy_levels.signal_cross_section_spectrum",
            }
        )
        if pump_curve or (
            float(
                doping.get(
                    "absorption_at_reference_per_percent_m_inv", 0.0
                )
            )
            > 0.0
            and density_per_percent > 0.0
        ):
            library_written_paths.add(
                "energy_levels.pump_absorption_cross_section_m2"
            )
        if signal_curve or spectroscopy.get("emission_cross_section_m2") is not None or spectroscopy.get("emission_temperature_curve"):
            library_written_paths.add(
                "energy_levels.signal_emission_cross_section_m2"
            )
        if signal_curve or spectroscopy.get("reabsorption_cross_section_m2") is not None or spectroscopy.get("reabsorption_temperature_curve"):
            library_written_paths.add(
                "energy_levels.signal_absorption_cross_section_m2"
            )
    if use_sellmeier and sellmeier.get("enabled"):
        library_written_paths.update(
            {
                "material.dispersion.sellmeier",
                "material.dispersion.reference_wavelength_m",
            }
        )
    if (
        quantum_efficiency is not None
        and 0.0 <= float(quantum_efficiency) <= 1.0
        and not preserve_existing_qe
    ):
        library_written_paths.add("energy_levels.pump_quantum_efficiency")
    if spectroscopy.get("quasi_three_level"):
        library_written_paths.add("energy_levels.model")

    # Source application never silently wins over a value demonstrably edited
    # after the previous snapshot. Restore those explicit overrides now and
    # record them as such in field-level provenance.
    for path, value in user_overrides.items():
        _set_path_value(result, path, value)
    if "energy_levels.pump_emission_cross_section_m2" in user_overrides:
        pump_spectrum = levels.get("pump_cross_section_spectrum", {})
        if pump_spectrum.get("wavelength_m"):
            pump_spectrum["sigma_em_m2"] = [
                float(levels["pump_emission_cross_section_m2"])
            ] * len(pump_spectrum["wavelength_m"])

    defaults = _managed_defaults()
    field_provenance: dict[str, dict[str, Any]] = {}
    applied_fields: dict[str, Any] = {}
    missing_library_fields: list[str] = []
    for path, default in defaults.items():
        current = copy.deepcopy(_path_value(result, path))
        applied_fields[path] = current
        if path in user_overrides:
            source = "user_override"
            source_digest = None
        elif path in library_written_paths:
            source = "material_library_record"
            source_digest = record_digest
        elif (
            not prior_library or not different_material_record
        ) and not _portable_values_equal(current, default):
            # On the first library application, an unsupported non-default
            # base-config value is the only available evidence of a deliberate
            # manual override.
            source = "user_override"
            source_digest = None
        else:
            source = "default"
            source_digest = None
            missing_library_fields.append(path)
        field_provenance[path] = {
            "source": source,
            "record_sha256": source_digest,
        }
    if prior_library and different_material_record and missing_library_fields:
        warnings.append(
            "The new record omitted some previously library-owned fields; they "
            "were reset to simulator defaults rather than inherited from the "
            "old material. See material.library.missing_library_fields and "
            "field_provenance before quantitative use."
        )

    material["library"] = {
        "schema": MATERIAL_LIBRARY_SCHEMA,
        "record_sha256": record_digest,
        "material_name": record["name"],
        "record_id": record.get("source", {}).get("record_id"),
        "source": copy.deepcopy(record.get("source", {})),
        "activation_policy": "runtime_ready",
        "activation_constraints": copy.deepcopy(
            record.get("activation_constraints", {})
        ),
        "experimental_calibration": copy.deepcopy(
            record.get("experimental_calibration", {})
        ),
        "doping_percent": float(doping_percent),
        "reference_percent": float(doping.get("reference_percent", 1.0)),
        "ion_density_per_percent_m3": density_per_percent,
        "operating_temperature_K": temperature,
        "optical_polarization_axis": polarization,
        "pump_polarization_axis": pump_polarization,
        "thermomechanical_axis": thermomechanical_axis,
        "polarization": polarization,
        "use_spectroscopy": bool(use_spectroscopy),
        "spectroscopic_response_available_data": imported_response_names,
        "use_sellmeier": bool(use_sellmeier and sellmeier.get("enabled")),
        "snapshot_range_contract": {
            "policy": snapshot_out_of_range,
            "evaluations": snapshot_range_contract,
            "all_within_calibrated_range": all(
                not bool(item["out_of_range"])
                for item in snapshot_range_contract
            ),
        },
        "default_heat_fraction": thermal.get("default_heat_fraction"),
        "pump_quantum_efficiency": quantum_efficiency,
        "effective_fluorescence_wavelength_enabled": effective_wavelength_enabled,
        "effective_fluorescence_wavelength_m": (
            effective_wavelength_raw if effective_wavelength_enabled else None
        ),
        "effective_fluorescence_wavelength_raw_m": effective_wavelength_raw,
        "pump_index_temperature_offset": pump_index_temperature_offset,
        "anisotropic": {
            "refractive_index_xyz": refractive,
            "dn_dT_xyz_per_K": dndt,
            "thermal_conductivity_xyz_W_mK": conductivity,
            "young_modulus_xyz_Pa": young,
            "thermal_expansion_xyz_per_K": expansion,
            "poisson_xy_yz_xz": poisson,
            "shear_modulus_xy_yz_xz_Pa": mechanical.get("shear_modulus_xy_yz_xz_Pa", []),
            "photoelastic_6x6": mechanical.get("photoelastic_6x6", []),
        },
        "rate_system": _quarantined_rate_system_metadata(record),
        "field_provenance": field_provenance,
        "applied_fields": applied_fields,
        "missing_library_fields": missing_library_fields,
        "applied_parameter_snapshot": {
            "ion_density_m3": float(material["ion_density_m3"]),
            "refractive_index": float(material["refractive_index"]),
            "pump_refractive_index": material.get("pump_refractive_index"),
            "dn_dT_per_K": float(material.get("dn_dT_per_K", 0.0)),
            "pump_absorption_cross_section_m2": float(levels["pump_absorption_cross_section_m2"]),
            "pump_emission_cross_section_m2": float(levels["pump_emission_cross_section_m2"]),
            "signal_absorption_cross_section_m2": float(levels["signal_absorption_cross_section_m2"]),
            "signal_emission_cross_section_m2": float(levels["signal_emission_cross_section_m2"]),
            "upper_state_lifetime_s": float(levels["upper_state_lifetime_s"]),
            "pump_quantum_efficiency": float(levels.get("pump_quantum_efficiency", 1.0)),
            "source_pump_quantum_efficiency": quantum_efficiency,
            "pump_quantum_efficiency_application": (
                "preserved_existing_override"
                if preserve_existing_qe
                else "applied_source_value"
            ),
        },
        "warnings": warnings,
    }
    outside_snapshot_ranges = [
        item
        for item in snapshot_range_contract
        if bool(item["out_of_range"])
    ]
    if outside_snapshot_ranges:
        warnings.append(
            "Explicit snapshot_out_of_range=edge used calibrated endpoint "
            "values outside "
            + ", ".join(str(item["name"]) for item in outside_snapshot_ranges)
            + "; this is a declared reduced-order extrapolation."
        )
        material["library"]["warnings"] = warnings
    return result, warnings
