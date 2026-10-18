"""Shared boundaries between source models, optical trains, and a gain medium."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .beams import make_beam
from .directions import (
    axis_direction_components,
    refract_axis_angles_rad as _refract_axis_angles_rad,
)
from .grid import Grid3D
from .optics import (
    OpticalTrace,
    apply_interface,
    gaussian_abcd_to_plane,
    propagate_fft_train,
    reference_ray_samples,
    source_field,
)
from .spectral import response_value


@dataclass
class ExternalField:
    field: np.ndarray
    refractive_index: float
    source_grid_quantity: float
    trace: OpticalTrace
    warnings: list[str]


def transverse_carrier_nyquist_fractions(
    *,
    wavelength_m: float,
    angle_x_deg: float,
    angle_y_deg: float,
    refractive_index: float,
    m2_x: float,
    m2_y: float,
    dx_max_m: float,
    dy_max_m: float,
) -> tuple[float, float]:
    """Return x/y sampled-carrier frequencies as fractions of Nyquist.

    This is the single contract used by both runtime warnings and the strict
    preflight gate. It matches the effective-M2 source carrier implemented by
    ``source_field`` and uses the largest grid-cell width.
    """

    wavelength = float(wavelength_m)
    index = float(refractive_index)
    if (
        wavelength <= 0.0
        or index <= 0.0
        or float(m2_x) <= 0.0
        or float(m2_y) <= 0.0
        or float(dx_max_m) <= 0.0
        or float(dy_max_m) <= 0.0
    ):
        raise ValueError(
            "carrier Nyquist inputs must be positive"
        )
    direction_x, direction_y, _ = axis_direction_components(
        math.radians(float(angle_x_deg)),
        math.radians(float(angle_y_deg)),
    )
    return (
        2.0
        * float(dx_max_m)
        * abs(index * direction_x)
        / (float(m2_x) * wavelength),
        2.0
        * float(dy_max_m)
        * abs(index * direction_y)
        / (float(m2_y) * wavelength),
    )


def additive_source_gain_angle_warning(
    beam_cfg: dict[str, Any],
    gain_markers: list[dict[str, Any]],
    label: str,
) -> str | None:
    """Describe the two independent angle controls before they are combined.

    The source angle belongs to the launched field/ray.  A gain-medium marker
    supplies an additional local reference-ray walk-off through the crystal.
    Keeping both is useful for an intentionally tilted source relative to an
    angled crystal, but entering the same measured incidence in both places
    silently counts it twice.
    """

    source_is_angled = (
        abs(float(beam_cfg.get("incidence_angle_x_deg", 0.0)))
        + abs(float(beam_cfg.get("incidence_angle_y_deg", 0.0)))
        > 0.0
    )
    marker_is_angled = any(
        abs(float(marker.get("incidence_angle_x_deg", 0.0)))
        + abs(float(marker.get("incidence_angle_y_deg", 0.0)))
        > 0.0
        for marker in gain_markers
    )
    if not source_is_angled or not marker_is_angled:
        return None
    return (
        f"{label} source launch angle and gain_medium incidence angle are both non-zero. "
        "The sampled source phase/ray tilt and the per-pass crystal walk-off are independent "
        "and additive; if they describe the same physical beam angle, set one pair to zero "
        "to avoid double counting."
    )


def _tilt_phase_sampling_warning(
    beam_cfg: dict[str, Any],
    grid: Grid3D,
    refractive_index: float,
) -> str | None:
    """Warn when a carrier tilt is too close to the sampled-grid Nyquist limit."""

    # The sampled M2 surrogate propagates with lambda_eff=M2*lambda, so its
    # carrier spatial frequency is n*sin(theta)/(M2*lambda).  Express it as a
    # fraction of the sampled Nyquist frequency 1/(2*d).  Use the largest cell
    # width so this remains conservative on a nonuniform display grid.
    ratio_x, ratio_y = transverse_carrier_nyquist_fractions(
        wavelength_m=float(beam_cfg["wavelength_m"]),
        angle_x_deg=float(
            beam_cfg.get("incidence_angle_x_deg", 0.0)
        ),
        angle_y_deg=float(
            beam_cfg.get("incidence_angle_y_deg", 0.0)
        ),
        refractive_index=float(refractive_index),
        m2_x=float(beam_cfg.get("m2_x", 1.0)),
        m2_y=float(beam_cfg.get("m2_y", 1.0)),
        dx_max_m=float(np.max(grid.x_widths)),
        dy_max_m=float(np.max(grid.y_widths)),
    )
    maximum = max(ratio_x, ratio_y)
    if maximum < 0.8:
        return None
    status = "aliases" if maximum >= 1.0 else "is close to the Nyquist limit"
    return (
        "Configured source incidence-angle carrier phase "
        f"{status} on the optical grid (x/y Nyquist fractions "
        f"{ratio_x:.3g}/{ratio_y:.3g}). Increase transverse sampling or use "
        "ABCD/ray propagation for the external angled path; a gain_medium "
        "incidence angle is transported inside the crystal as explicit walk-off."
    )


def build_external_field(
    beam_cfg: dict[str, Any],
    total_quantity: float,
    elements: list[dict[str, Any]],
    grid: Grid3D,
    mode: str,
    quantity_name: str,
    interface_model: str,
    record_planes: bool,
    *,
    include_abcd_tilt_phase: bool = False,
    abcd_diagnostics: dict[str, Any] | None = None,
) -> ExternalField:
    """Propagate a configured source to the gain-medium marker."""
    source = source_field(beam_cfg, total_quantity, grid)
    source_total = float(grid.integrate_transverse(np.abs(source) ** 2))
    trace = OpticalTrace()
    if record_planes:
        trace.add("source", source, grid, quantity_name)
    warnings: list[str] = []
    if (
        float(beam_cfg.get("rotation_deg", 0.0)) != 0.0
        and not math.isclose(
            float(beam_cfg.get("m2_x", 1.0)),
            float(beam_cfg.get("m2_y", 1.0)),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        and mode in {"fft", "abcd"}
    ):
        warnings.append(
            "The sampled M2 propagation kernel is axis-aligned while the source has a rotated anisotropic M2 tensor; use an explicit HG modal field or verify against a coupled first/second-moment model."
        )
    raw_fraction = 1.0
    if total_quantity > 0.0:
        xx, yy = grid.transverse_mesh
        raw_fraction = float(grid.integrate_transverse(make_beam(beam_cfg).density(xx, yy, 0.0)))
        if raw_fraction > 1.02:
            warnings.append(
                f"Source beam is under-resolved: raw grid quadrature was {raw_fraction:.3%} of configured energy; "
                "the source was conservatively capped. Increase Nx/Ny for an accurate spatial profile."
            )
    source_n = float(beam_cfg.get("source_refractive_index", 1.0))
    if mode == "fft":
        tilt_warning = _tilt_phase_sampling_warning(beam_cfg, grid, source_n)
        if tilt_warning is not None:
            warnings.append(tilt_warning)
        field, current_n, propagated = propagate_fft_train(
            source,
            elements,
            grid,
            float(beam_cfg["wavelength_m"]),
            source_n,
            quantity_name,
            interface_model,
            float(beam_cfg.get("m2_x", 1.0)),
            float(beam_cfg.get("m2_y", 1.0)),
            record_planes,
        )
        trace = propagated if propagated.summaries else trace
    elif mode == "abcd":
        field, current_n, propagated, abcd_warnings = gaussian_abcd_to_plane(
            beam_cfg,
            total_quantity,
            elements,
            grid,
            quantity_name,
            interface_model,
            record_planes,
            include_tilt_phase=include_abcd_tilt_phase,
            diagnostics=abcd_diagnostics,
        )
        warnings.extend(abcd_warnings)
        if include_abcd_tilt_phase:
            tilt_warning = _tilt_phase_sampling_warning(
                beam_cfg,
                grid,
                float(abcd_diagnostics.get("refractive_index", source_n))
                if abcd_diagnostics is not None
                else source_n,
            )
            if tilt_warning is not None:
                warnings.append(tilt_warning)
        # ABCD evaluates the infinite-plane Gaussian analytically at the gain;
        # its narrow numerical window is only a source-plane display window.
        # Reporting the clipped display integral as transported source power
        # made a configured 0.7 W Seed appear to be roughly 0.32 W even though
        # the calculation correctly propagated the full 0.7 W.
        source_total = float(total_quantity)
        if total_quantity > 0.0 and raw_fraction < 0.98:
            warnings.append(
                "ABCD uses an analytic infinite-plane source: the source-plane display window "
                f"contains {raw_fraction:.3%}, but no power was clipped unless an aperture is configured."
            )
        trace.extend(propagated)
    elif mode == "off":
        field, current_n = source, source_n
        if elements:
            warnings.append("optical_propagation=off ignores source-to-gain geometric elements.")
    else:
        raise ValueError(f"Unknown external optical propagation mode {mode!r}")
    trace.reference_ray_samples = reference_ray_samples(
        beam_cfg,
        elements,
        wavelength_m=float(beam_cfg["wavelength_m"]),
        m2_x=float(beam_cfg.get("m2_x", 1.0)),
        m2_y=float(beam_cfg.get("m2_y", 1.0)),
    )
    return ExternalField(field, current_n, source_total, trace, warnings)


def gain_interface(
    field: np.ndarray,
    n_before: float,
    n_after: float,
    marker: dict[str, Any],
    side: str,
    default_model: str,
    wavelength_m: float | None = None,
    *,
    entering: bool | None = None,
    surface_sag_m: np.ndarray | None = None,
) -> np.ndarray:
    model = str(marker.get(f"{side}_model", marker.get("model", default_model)))
    key = f"{side}_power_transmission"
    transmission = gain_surface_power_transmission(
        marker,
        side,
        wavelength_m,
    )
    angle_x, angle_y = gain_medium_axis_angles_rad(marker)
    if entering is True:
        angle_x, angle_y = refract_axis_angles_rad(
            angle_x,
            angle_y,
            n_before=n_after,
            n_after=n_before,
        )
    incidence_angle_deg = math.degrees(math.acos(_axis_direction_cosine(angle_x, angle_y)))
    result = apply_interface(
        field,
        n_before,
        n_after,
        model,
        transmission,
        incidence_angle_deg,
        str(marker.get("polarization", "unpolarized")),
    )
    if surface_sag_m is None:
        return result
    if wavelength_m is None or float(wavelength_m) <= 0.0:
        raise ValueError("An endface sag phase screen requires a positive wavelength")
    if entering is None:
        raise ValueError("Endface sag phase requires an explicit entering flag")
    return apply_endface_sag_phase(
        result,
        n_crystal=(float(n_after) if entering else float(n_before)),
        n_outside=(float(n_before) if entering else float(n_after)),
        side=side,
        wavelength_m=float(wavelength_m),
        surface_sag_m=surface_sag_m,
    )


def apply_endface_sag_phase(
    field: np.ndarray,
    *,
    n_crystal: float,
    n_outside: float,
    side: str,
    wavelength_m: float,
    surface_sag_m: np.ndarray | None,
) -> np.ndarray:
    """Apply one wavelength/role-specific moving-face OPD approximation."""

    if surface_sag_m is None:
        return np.asarray(field)
    if side not in {"front", "back"}:
        raise ValueError("Endface side must be front or back")
    if not math.isfinite(float(wavelength_m)) or float(wavelength_m) <= 0.0:
        raise ValueError("An endface sag phase screen requires a positive wavelength")
    if not math.isfinite(float(n_crystal)) or not math.isfinite(float(n_outside)):
        raise ValueError("Endface sag refractive indices must be finite")
    sag = np.asarray(surface_sag_m, dtype=float)
    if sag.shape != np.asarray(field).shape[-2:]:
        raise ValueError("Endface sag must match the optical transverse grid")
    if not np.all(np.isfinite(sag)):
        raise ValueError("Endface sag must contain only finite displacements")
    # With displacements defined along +z, the two contributions sum to
    # (n_crystal-n_outside)*(h_back-h_front) for either traversal direction.
    orientation = -1.0 if side == "front" else 1.0
    phase = (
        2.0
        * math.pi
        / float(wavelength_m)
        * orientation
        * (float(n_crystal) - float(n_outside))
        * sag
    )
    return np.asarray(field) * np.exp(1j * phase)


def gain_surface_power_transmission(
    marker: dict[str, Any],
    side: str,
    wavelength_m: float | None,
) -> float:
    """Resolve a gain-face coating with side-specific then generic fallback."""

    side_key = f"{side}_power_transmission"
    configured = marker.get(side_key)
    if configured is None:
        configured = marker.get("power_transmission")
    default = 1.0 if configured is None else float(configured)
    if wavelength_m is None:
        return default
    response = marker.get("spectral_response")
    if isinstance(response, dict):
        if side_key in response:
            return response_value(marker, side_key, wavelength_m, default)
        if "power_transmission" in response:
            return response_value(
                marker,
                "power_transmission",
                wavelength_m,
                default,
            )
    return default


def gain_interface_element(marker: dict[str, Any], n_after: float, side: str, default_model: str) -> dict[str, Any]:
    internal_x, internal_y = gain_medium_axis_angles_rad(marker)
    n_before = float(marker.get("outside_refractive_index", 1.0))
    angle_x, angle_y = refract_axis_angles_rad(
        internal_x,
        internal_y,
        n_before=n_after,
        n_after=n_before,
    )
    transmission = marker.get(f"{side}_power_transmission")
    if transmission is None:
        transmission = marker.get("power_transmission")
    if transmission is None:
        transmission = 1.0
    result = {
        "type": "interface",
        "name": f"gain_{side}_surface",
        "n_after": n_after,
        "model": marker.get(f"{side}_model", marker.get("model", default_model)),
        "power_transmission": transmission,
        "incidence_angle_deg": math.degrees(
            math.acos(_axis_direction_cosine(angle_x, angle_y))
        ),
        # gain_medium incidence angles are defined inside the crystal.  This
        # interface element is on the outside, so its axis break is the Snell-
        # conjugate external angle; the ray/ABCD interface then refracts it
        # back to the configured internal direction.
        "axis_angle_x_deg": math.degrees(angle_x),
        "axis_angle_y_deg": math.degrees(angle_y),
        "polarization": marker.get("polarization", "unpolarized"),
    }
    response = marker.get("spectral_response")
    if isinstance(response, dict):
        side_key = f"{side}_power_transmission"
        source_key = (
            side_key
            if side_key in response
            else "power_transmission"
            if "power_transmission" in response
            else None
        )
        if source_key is not None:
            result["spectral_response"] = {
                "wavelength_m": list(response.get("wavelength_m", [])),
                "power_transmission": list(response.get(source_key, [])),
                "out_of_range": str(
                    response.get("out_of_range", "error")
                ),
            }
    return result


def gain_medium_axis_angles_rad(marker: dict[str, Any]) -> tuple[float, float]:
    """Return the canonical signed in-crystal gain-axis angles."""

    return (
        math.radians(float(marker.get("incidence_angle_x_deg", 0.0))),
        math.radians(float(marker.get("incidence_angle_y_deg", 0.0))),
    )


def _axis_direction_cosine(angle_x_rad: float, angle_y_rad: float) -> float:
    return axis_direction_components(angle_x_rad, angle_y_rad)[2]


def gain_medium_path_cosine(marker: dict[str, Any]) -> float:
    """Return dz/ds for the configured in-crystal propagation direction."""

    return max(_axis_direction_cosine(*gain_medium_axis_angles_rad(marker)), 1e-3)


def refract_axis_angles_rad(
    angle_x_rad: float,
    angle_y_rad: float,
    *,
    n_before: float,
    n_after: float,
) -> tuple[float, float]:
    """Apply vector Snell refraction to signed x/y slope angles."""

    return _refract_axis_angles_rad(
        angle_x_rad,
        angle_y_rad,
        n_before=n_before,
        n_after=n_after,
    )
