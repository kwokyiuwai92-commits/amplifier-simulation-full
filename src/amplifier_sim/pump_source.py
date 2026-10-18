"""Canonical Pump source parameterizations and phase-space conversions.

A diode or fiber emitter can be described by its emitting radius and numerical
aperture.  The stochastic backend samples those native phase-space values;
the field backends consume the effective waist/M2 fallback generated here.
Keeping both representations in one layer prevents UI, preview, and solver
definitions from silently diverging.
"""

from __future__ import annotations

import math
from typing import Any


PUMP_SOURCE_MODES = {"waist_m2", "waist_divergence", "emitter_na"}


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _half_angle_from_na(numerical_aperture: Any, refractive_index: Any, axis: str) -> float:
    na = float(numerical_aperture)
    n = _positive(refractive_index, "pump.source_definition.free_space_refractive_index")
    if not math.isfinite(na) or na < 0.0 or na >= n:
        raise ValueError(
            f"pump.source_definition.numerical_aperture_{axis} must lie in [0, n)"
        )
    return math.asin(na / n)


def _effective_m2(
    radius_m: float,
    half_angle_rad: float,
    wavelength_m: float,
    refractive_index: float,
) -> tuple[float, float]:
    """Return an envelope M2 and the admissible field-transport surrogate.

    ``wavelength_m`` is the vacuum wavelength.  For a paraxial beam in a
    homogeneous medium, ``theta = M2 * wavelength / (pi * n * w0)``.  The
    radius and angle supplied by the public source definitions are profile
    scales/supports rather than ISO second moments, so this is deliberately
    described as an envelope conversion below.
    """

    raw = (
        math.pi
        * _positive(refractive_index, "pump source refractive index")
        * radius_m
        * math.tan(half_angle_rad)
        / wavelength_m
    )
    return raw, max(raw, 1.0)


def canonicalize_pump_source(pump: dict[str, Any]) -> dict[str, Any]:
    """Update solver-facing Pump fields from the selected public definition.

    ``waist_m2`` directly supplies a paraxial waist and M2.
    ``waist_divergence`` supplies waist/focus, divergence, and a radial
    super-Gaussian exponent. ``emitter_na`` supplies emitting radius and NA.
    The latter two keep both the native ray-source values and a second-moment
    fallback for field propagation; selecting the transport backend is a
    separate configuration decision.
    """

    source = pump.setdefault("source_definition", {})
    mode = str(source.get("mode", "waist_m2"))
    if mode not in PUMP_SOURCE_MODES:
        raise ValueError(
            "pump.source_definition.mode must be waist_m2, waist_divergence, or emitter_na"
        )
    wavelength = _positive(pump.get("wavelength_m"), "pump.wavelength_m")
    geometric_efficiency = float(pump.get("geometric_efficiency", 1.0))
    if not math.isfinite(geometric_efficiency) or not 0.0 <= geometric_efficiency <= 1.0:
        raise ValueError("pump.geometric_efficiency must lie in [0, 1]")

    diagnostics: dict[str, Any] = {
        "mode": mode,
        "geometric_efficiency": geometric_efficiency,
    }
    if mode == "waist_m2":
        diagnostics.update(
            {
                "radius_x_m": float(pump["waist_x_m"]),
                "radius_y_m": float(pump["waist_y_m"]),
                "effective_m2_x": float(pump.get("m2_x", 1.0)),
                "effective_m2_y": float(pump.get("m2_y", 1.0)),
                "solver_representation": "direct_waist_m2",
                "native_ray_parameterization": "waist_phase_space",
                "field_solver_fallback": "direct_waist_m2",
            }
        )
        source["derived"] = diagnostics
        return diagnostics

    exponent_x = _positive(
        source.get("super_gaussian_exponent_x", 2.0),
        "pump.source_definition.super_gaussian_exponent_x",
    )
    exponent_y = _positive(
        source.get("super_gaussian_exponent_y", 2.0),
        "pump.source_definition.super_gaussian_exponent_y",
    )
    # Internal ``order`` uses exp[-2 (x/w)^(2*order)], whereas the public
    # complete exponent is twice that value.  Equal x/y exponents select the
    # elliptical-radial law exp[-2 r^p]; unequal exponents select the explicit
    # separable law exp[-2 (|x/wx|^px + |y/wy|^py)].  The field and ray
    # implementations already make this distinction, so the canonical
    # profile token must not continue to call an unequal-exponent source radial.
    pump["order_x"] = exponent_x / 2.0
    pump["order_y"] = exponent_y / 2.0
    radial_profile = math.isclose(
        exponent_x, exponent_y, rel_tol=0.0, abs_tol=1e-14
    )
    pump["profile"] = (
        "radial_super_gaussian" if radial_profile else "super_gaussian"
    )

    if mode == "waist_divergence":
        source_n = _positive(
            pump.get("source_refractive_index", 1.0),
            "pump.source_refractive_index",
        )
        radius_x = _positive(
            source.get("waist_radius_x_m", pump.get("waist_x_m")),
            "pump.source_definition.waist_radius_x_m",
        )
        radius_y = _positive(
            source.get("waist_radius_y_m", pump.get("waist_y_m")),
            "pump.source_definition.waist_radius_y_m",
        )
        angle_x = math.radians(float(source.get("divergence_half_angle_x_deg", 0.0)))
        angle_y = math.radians(float(source.get("divergence_half_angle_y_deg", 0.0)))
        if angle_x < 0.0 or angle_x >= math.pi / 2.0 or angle_y < 0.0 or angle_y >= math.pi / 2.0:
            raise ValueError("Pump divergence half-angles must lie in [0, 90) deg")
        raw_x, m2_x = _effective_m2(radius_x, angle_x, wavelength, source_n)
        raw_y, m2_y = _effective_m2(radius_y, angle_y, wavelength, source_n)
        native_ray_parameterization = "waist_and_divergence_phase_space"
        field_solver_fallback = (
            "radial_super_gaussian_caustic"
            if radial_profile
            else "separable_super_gaussian_caustic"
        )
    else:
        radius_x = _positive(
            source.get("emitting_radius_x_m", pump.get("waist_x_m")),
            "pump.source_definition.emitting_radius_x_m",
        )
        radius_y = _positive(
            source.get("emitting_radius_y_m", pump.get("waist_y_m")),
            "pump.source_definition.emitting_radius_y_m",
        )
        n_free = _positive(
            source.get("free_space_refractive_index", 1.0),
            "pump.source_definition.free_space_refractive_index",
        )
        # The emitter NA and the fallback field must refer to the same launch
        # medium.  Keeping two independent indices here made the ray and field
        # backends disagree whenever n != 1.
        pump["source_refractive_index"] = n_free
        angle_x = _half_angle_from_na(
            source.get("numerical_aperture_x", 0.0), n_free, "x"
        )
        angle_y = _half_angle_from_na(
            source.get("numerical_aperture_y", 0.0), n_free, "y"
        )
        raw_x, m2_x = _effective_m2(radius_x, angle_x, wavelength, n_free)
        raw_y, m2_y = _effective_m2(radius_y, angle_y, wavelength, n_free)
        # A diode source is an emitting phase-space area, not a coherent waist.
        native_ray_parameterization = "emitting_area_and_na_phase_space"
        field_solver_fallback = "profile_envelope_field_surrogate"

    pump["waist_x_m"] = radius_x
    pump["waist_y_m"] = radius_y
    pump["m2_x"] = m2_x
    pump["m2_y"] = m2_y
    diagnostics.update(
        {
            "radius_x_m": radius_x,
            "radius_y_m": radius_y,
            "full_width_x_m": 2.0 * radius_x,
            "full_width_y_m": 2.0 * radius_y,
            "maximum_divergence_half_angle_x_deg": math.degrees(angle_x),
            "maximum_divergence_half_angle_y_deg": math.degrees(angle_y),
            "raw_geometric_m2_x": raw_x,
            "raw_geometric_m2_y": raw_y,
            # ``raw_geometric_m2_*`` is retained as a compatibility alias.
            "geometric_envelope_m2_x": raw_x,
            "geometric_envelope_m2_y": raw_y,
            "effective_m2_x": m2_x,
            "effective_m2_y": m2_y,
            "source_refractive_index": source_n if mode == "waist_divergence" else n_free,
            "configured_radius_semantics": (
                "elliptical radial scale in I proportional to exp[-2 r^p]"
                if radial_profile
                else "axis scales in separable I proportional to exp[-2 (|x/wx|^px + |y/wy|^py)]"
            ),
            "m2_conversion_semantics": (
                "profile-envelope radius and configured divergence support; not an ISO second-moment measurement"
            ),
            "radial_super_gaussian_exponent_x": exponent_x,
            "radial_super_gaussian_exponent_y": exponent_y,
            "internal_super_gaussian_order_x": exponent_x / 2.0,
            "internal_super_gaussian_order_y": exponent_y / 2.0,
            "spatial_profile_contract": (
                "elliptical_radial_super_gaussian"
                if radial_profile
                else "separable_axis_super_gaussian"
            ),
            "solver_representation": "propagation_backend_dependent",
            "native_ray_parameterization": native_ray_parameterization,
            "field_solver_fallback": field_solver_fallback,
        }
    )
    source["derived"] = diagnostics
    return diagnostics
