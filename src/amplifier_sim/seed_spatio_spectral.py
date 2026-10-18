"""Wavelength-dependent Seed source geometry.

The temporal/spectral pulse definition controls how much complex field exists
at each optical frequency.  This module independently controls the transverse
Gaussian state assigned to each frequency before the external optical train.
Keeping those two concerns separate avoids treating M2 or a measured spectrum
as if either one uniquely determined spatial chirp or chromatic focus.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


_SCALAR_FIELDS = (
    "waist_x_m",
    "waist_y_m",
    "m2_x",
    "m2_y",
    "focus_x_m",
    "focus_y_m",
    "center_x_m",
    "center_y_m",
    "incidence_angle_x_deg",
    "incidence_angle_y_deg",
)


def _base_focus(seed_cfg: dict[str, Any], axis: str) -> float:
    value = seed_cfg.get(f"focus_{axis}_m")
    return float(seed_cfg.get("focus_z_m", 0.0) if value is None else value)


def _reference_wavelength(seed_cfg: dict[str, Any], model: dict[str, Any]) -> float:
    value = model.get("reference_wavelength_m")
    return float(seed_cfg["wavelength_m"] if value is None else value)


def _interpolate_samples(
    samples: list[dict[str, Any]],
    wavelength_m: float,
    key: str,
    fallback: float,
) -> float:
    rows = sorted(
        (
            (float(row["wavelength_m"]), row.get(key))
            for row in samples
            if row.get("wavelength_m") is not None
        ),
        key=lambda item: item[0],
    )
    if not rows:
        return fallback
    wavelengths = np.asarray([row[0] for row in rows], dtype=float)
    insertion = int(np.searchsorted(wavelengths, wavelength_m, side="left"))
    if insertion < len(rows) and wavelengths[insertion] == wavelength_m:
        value = rows[insertion][1]
        return fallback if value is None else float(value)
    if insertion == 0:
        value = rows[0][1]
        return fallback if value is None else float(value)
    if insertion == len(rows):
        value = rows[-1][1]
        return fallback if value is None else float(value)
    lower_wavelength, lower_value = rows[insertion - 1]
    upper_wavelength, upper_value = rows[insertion]
    # Missing table cells explicitly retain the parametric/base value.  Do not
    # interpolate across an omitted cell using a more distant sample: doing so
    # silently turns a blank UI entry into a measured constraint.
    if lower_value is None or upper_value is None:
        return fallback
    fraction = (wavelength_m - lower_wavelength) / (
        upper_wavelength - lower_wavelength
    )
    return float(lower_value) + fraction * (float(upper_value) - float(lower_value))


def seed_config_at_wavelength(
    seed_cfg: dict[str, Any], wavelength_m: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a shallow Seed config specialised for one wavelength.

    ``shared_source`` exactly preserves the historical source definition.
    ``chromatic_parametric`` applies smooth wavelength scaling and first-order
    spatial chirp. ``wavelength_table`` then overrides any supplied quantities
    by linearly interpolating explicit samples; missing columns retain the
    parametric/base value.
    """

    wavelength = float(wavelength_m)
    local = dict(seed_cfg)
    local["wavelength_m"] = wavelength
    model = dict(seed_cfg.get("spatio_spectral", {}))
    kind = str(model.get("model", "shared_source"))
    reference = _reference_wavelength(seed_cfg, model)
    fractional_offset = wavelength / reference - 1.0

    # Start with explicit axis focus values so each wavelength receives an
    # unambiguous astigmatic q parameter even when legacy focus_z_m was used.
    local["focus_x_m"] = _base_focus(seed_cfg, "x")
    local["focus_y_m"] = _base_focus(seed_cfg, "y")
    for axis in ("x", "y"):
        local[f"center_{axis}_m"] = float(seed_cfg.get(f"center_{axis}_m", 0.0))
        local[f"incidence_angle_{axis}_deg"] = float(
            seed_cfg.get(f"incidence_angle_{axis}_deg", 0.0)
        )

    if kind in {"chromatic_parametric", "wavelength_table"}:
        ratio = wavelength / reference
        for axis in ("x", "y"):
            waist = float(seed_cfg[f"waist_{axis}_m"])
            exponent = float(model.get(f"waist_scaling_exponent_{axis}", 0.0))
            local[f"waist_{axis}_m"] = waist * ratio**exponent

            m2 = float(seed_cfg.get(f"m2_{axis}", 1.0))
            m2_exponent = float(model.get(f"m2_scaling_exponent_{axis}", 0.0))
            local[f"m2_{axis}"] = m2 * ratio**m2_exponent

            local[f"focus_{axis}_m"] += float(
                model.get(f"focus_shift_{axis}_m_per_fraction", 0.0)
            ) * fractional_offset
            local[f"center_{axis}_m"] = float(seed_cfg.get(f"center_{axis}_m", 0.0)) + float(
                model.get(f"center_shift_{axis}_m_per_fraction", 0.0)
            ) * fractional_offset
            local[f"incidence_angle_{axis}_deg"] = float(
                seed_cfg.get(f"incidence_angle_{axis}_deg", 0.0)
            ) + float(model.get(f"angle_shift_{axis}_deg_per_fraction", 0.0)) * fractional_offset

    if kind == "wavelength_table":
        samples = list(model.get("samples", []))
        policy = str(model.get("out_of_range", "error"))
        table_wavelengths = sorted(
            float(row["wavelength_m"])
            for row in samples
            if row.get("wavelength_m") is not None
        )
        outside_table = bool(
            table_wavelengths
            and (
                wavelength < table_wavelengths[0]
                or wavelength > table_wavelengths[-1]
            )
        )
        if outside_table and policy == "error":
            raise ValueError(
                "wavelength-resolved Seed source wavelength "
                f"{wavelength:.9g} m lies outside wavelength_table support "
                f"[{table_wavelengths[0]:.9g}, {table_wavelengths[-1]:.9g}] m; "
                "set seed.spatio_spectral.out_of_range=edge to request explicit "
                "endpoint extrapolation"
            )
        for key in _SCALAR_FIELDS:
            local[key] = _interpolate_samples(samples, wavelength, key, float(local[key]))

    for axis in ("x", "y"):
        m2 = float(local[f"m2_{axis}"])
        if not math.isfinite(m2) or m2 < 1.0:
            raise ValueError(
                f"wavelength-resolved Seed m2_{axis} must be finite and at least 1 "
                "for a physical paraxial beam"
            )
        angle = float(local[f"incidence_angle_{axis}_deg"])
        if not math.isfinite(angle) or abs(angle) >= 89.0:
            raise ValueError(
                f"wavelength-resolved Seed incidence angle {axis} must be finite and have magnitude below 89 degrees"
            )

    local["focus_z_m"] = 0.5 * (float(local["focus_x_m"]) + float(local["focus_y_m"]))
    diagnostics = {
        "model": kind,
        "wavelength_m": wavelength,
        "reference_wavelength_m": reference,
        "fractional_wavelength_offset": fractional_offset,
        **{key: float(local[key]) for key in _SCALAR_FIELDS},
    }
    if kind == "wavelength_table":
        diagnostics.update(
            {
                "wavelength_table_out_of_range": outside_table,
                "wavelength_table_out_of_range_policy": policy,
                "wavelength_table_minimum_m": (
                    table_wavelengths[0] if table_wavelengths else None
                ),
                "wavelength_table_maximum_m": (
                    table_wavelengths[-1] if table_wavelengths else None
                ),
            }
        )
    return local, diagnostics


def validate_seed_spatio_spectral(seed_cfg: dict[str, Any]) -> None:
    """Validate wavelength-dependent Seed geometry without requiring it."""

    model = seed_cfg.get("spatio_spectral", {})
    kind = str(model.get("model", "shared_source"))
    allowed = {"shared_source", "chromatic_parametric", "wavelength_table"}
    if kind not in allowed:
        raise ValueError(f"seed.spatio_spectral.model must be one of {sorted(allowed)}")
    out_of_range = str(model.get("out_of_range", "error"))
    if out_of_range not in {"error", "edge"}:
        raise ValueError(
            "seed.spatio_spectral.out_of_range must be error or edge"
        )
    reference = _reference_wavelength(seed_cfg, model)
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError("seed.spatio_spectral.reference_wavelength_m must be positive")
    numeric_keys = (
        *(f"waist_scaling_exponent_{axis}" for axis in ("x", "y")),
        *(f"m2_scaling_exponent_{axis}" for axis in ("x", "y")),
        *(f"focus_shift_{axis}_m_per_fraction" for axis in ("x", "y")),
        *(f"center_shift_{axis}_m_per_fraction" for axis in ("x", "y")),
        *(f"angle_shift_{axis}_deg_per_fraction" for axis in ("x", "y")),
    )
    for key in numeric_keys:
        if key in model and not math.isfinite(float(model[key])):
            raise ValueError(f"seed.spatio_spectral.{key} must be finite")
    samples = model.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("seed.spatio_spectral.samples must be a list")
    sample_wavelengths: list[float] = []
    for index, row in enumerate(samples):
        if not isinstance(row, dict):
            raise ValueError(f"seed.spatio_spectral.samples[{index}] must be an object")
        wavelength = float(row.get("wavelength_m", 0.0))
        if not math.isfinite(wavelength) or wavelength <= 0.0:
            raise ValueError(
                f"seed.spatio_spectral.samples[{index}].wavelength_m must be positive"
            )
        sample_wavelengths.append(wavelength)
        for key in _SCALAR_FIELDS:
            if row.get(key) is not None and not math.isfinite(float(row[key])):
                raise ValueError(f"seed.spatio_spectral.samples[{index}].{key} must be finite")
        for key in ("waist_x_m", "waist_y_m", "m2_x", "m2_y"):
            if row.get(key) is not None and float(row[key]) <= 0.0:
                raise ValueError(f"seed.spatio_spectral.samples[{index}].{key} must be positive")
        for axis in ("x", "y"):
            m2_key = f"m2_{axis}"
            if row.get(m2_key) is not None:
                m2 = float(row[m2_key])
                if m2 < 1.0:
                    raise ValueError(
                        f"seed.spatio_spectral.samples[{index}].{m2_key} must be at least 1"
                    )
                if (
                    kind == "wavelength_table"
                    and seed_cfg.get("spatial_modes", {}).get("model") == "incoherent_hg"
                    and not math.isclose(m2, float(seed_cfg.get(m2_key, 1.0)),
                                         rel_tol=1e-12, abs_tol=0.0)
                ):
                    raise ValueError(
                        "incoherent_hg currently requires wavelength-independent M2 "
                        "mode weights; wavelength_table M2 must match the base Seed M2"
                    )
            key = f"incidence_angle_{axis}_deg"
            if row.get(key) is not None and abs(float(row[key])) >= 89.0:
                raise ValueError(
                    f"seed.spatio_spectral.samples[{index}].{key} must have magnitude below 89 degrees"
                )
    if len(set(sample_wavelengths)) != len(sample_wavelengths):
        raise ValueError(
            "seed.spatio_spectral.samples wavelengths must be unique"
        )


def sampled_field_moments(field: np.ndarray, grid: Any) -> dict[str, float | None]:
    """Return finite-window centroids and Gaussian-equivalent 1/e2 radii."""

    intensity = np.abs(np.asarray(field)) ** 2
    # Sum any leading temporal/mode axes while preserving the final y/x axes.
    if intensity.ndim > 2:
        intensity = np.sum(intensity, axis=tuple(range(intensity.ndim - 2)))
    total = float(grid.integrate_transverse(intensity))
    if total <= np.finfo(float).tiny:
        return {
            "sampled_quantity": total,
            "center_x_m": None,
            "center_y_m": None,
            "radius_x_m": None,
            "radius_y_m": None,
        }
    xx, yy = grid.transverse_mesh
    center_x = float(grid.integrate_transverse(intensity * xx) / total)
    center_y = float(grid.integrate_transverse(intensity * yy) / total)
    radius_x = 2.0 * math.sqrt(
        max(float(grid.integrate_transverse(intensity * (xx - center_x) ** 2) / total), 0.0)
    )
    radius_y = 2.0 * math.sqrt(
        max(float(grid.integrate_transverse(intensity * (yy - center_y) ** 2) / total), 0.0)
    )
    return {
        "sampled_quantity": total,
        "center_x_m": center_x,
        "center_y_m": center_y,
        "radius_x_m": radius_x,
        "radius_y_m": radius_y,
    }


def normalized_field_overlap(field: np.ndarray, reference: np.ndarray, grid: Any) -> float | None:
    """Return coherent spatial-mode overlap, independent of bin energy."""

    first = np.asarray(field)
    second = np.asarray(reference)
    first_energy = float(grid.integrate_transverse(np.abs(first) ** 2))
    second_energy = float(grid.integrate_transverse(np.abs(second) ** 2))
    if min(first_energy, second_energy) <= np.finfo(float).tiny:
        return None
    overlap = grid.integrate_transverse(np.conj(second) * first)
    return float(np.clip(abs(overlap) ** 2 / (first_energy * second_energy), 0.0, 1.0))
