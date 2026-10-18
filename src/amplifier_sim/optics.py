"""Sequential physical-optics and Gaussian/ABCD optical trains.

The optical train operates on a local transverse x-y plane. Free-space tilt and
decenter therefore move the field inside that plane, while mirrors may flip the
local axes. It is not a non-sequential CAD ray tracer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from .directions import axis_direction_components, refract_axis_angles_rad
from .beams import make_beam
from .beam_quality import (
    sampled_beam_quality,
    sampled_spatial_frequency_diagnostics,
)
from .compute_backend import fft2, ifft2
from .constants import LIGHT_SPEED
from .dispersion import phase_from_coefficients
from .grid import Grid3D
from .spatial_modes import axis_focus_m, effective_hg_density, single_hg_mode_amplitude
from .spectral import response_value


@dataclass
class OpticalTrace:
    summaries: list[dict[str, Any]] = field(default_factory=list)
    planes: dict[str, np.ndarray] = field(default_factory=dict)
    plane_coordinates: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    reference_ray_samples: list[dict[str, Any]] = field(
        default_factory=list
    )

    def add(
        self, name: str, field_value: np.ndarray, grid: Grid3D, quantity: str,
        *, complex_field_available: bool = True,
    ) -> None:
        intensity = np.abs(field_value) ** 2
        if intensity.ndim > 2:
            intensity = np.sum(intensity, axis=tuple(range(intensity.ndim - 2)))
        moments = transverse_intensity_moments(intensity, grid)
        boundary = transverse_boundary_energy_fractions(intensity, grid)
        total = moments["total"]
        key = _unique_key(name, self.planes)
        self.planes[key] = intensity.copy()
        self.plane_coordinates[key] = (grid.x.copy(), grid.y.copy())
        summary = {
            "name": key,
            quantity: total,
            "grid_nx": len(grid.x),
            "grid_ny": len(grid.y),
            "grid_window_x_m": float(grid.x_edges[-1] - grid.x_edges[0]),
            "grid_window_y_m": float(grid.y_edges[-1] - grid.y_edges[0]),
            "centroid_x_m": moments["centroid_x_m"],
            "centroid_y_m": moments["centroid_y_m"],
            "radius_x_D4sigma_m": moments["radius_x_D4sigma_m"],
            "radius_y_D4sigma_m": moments["radius_y_D4sigma_m"],
            "covariance_xy_m2": moments["covariance_xy_m2"],
            "radius_major_D4sigma_m": moments["radius_major_D4sigma_m"],
            "radius_minor_D4sigma_m": moments["radius_minor_D4sigma_m"],
            "major_axis_angle_deg": moments["major_axis_angle_deg"],
            **boundary,
        }
        quality = (
            sampled_beam_quality(field_value, grid)
            if complex_field_available
            else {
                "valid": False,
                "method": "intensity_only_no_phase_space_moments",
                "sampling_warning": "incoherent_channel_fields_not_retained",
            }
        )
        summary.update(
            {
                "beam_quality_method": quality.get("method"),
                "beam_quality_valid": quality.get("valid", False),
                "M2_x_sampled": quality.get("m2_x"),
                "M2_y_sampled": quality.get("m2_y"),
                "beam_quality_channel_count": quality.get("channel_count"),
                "beam_quality_sampling_warning": quality.get("sampling_warning"),
            }
        )
        self.summaries.append(summary)

    def extend(self, other: "OpticalTrace", prefix: str = "") -> None:
        for summary in other.summaries:
            original = str(summary["name"])
            key = _unique_key(f"{prefix}{original}", self.planes)
            copied = dict(summary)
            copied["name"] = key
            self.summaries.append(copied)
            self.planes[key] = other.planes[original].copy()
            coordinates = other.plane_coordinates.get(original)
            if coordinates is not None:
                self.plane_coordinates[key] = (
                    coordinates[0].copy(),
                    coordinates[1].copy(),
                )
        self.reference_ray_samples.extend(
            {
                **sample,
                "location": f"{prefix}{sample['location']}",
            }
            for sample in other.reference_ray_samples
        )


@dataclass
class ReferenceRayState:
    """Paraxial centroid/carrier state shared by runtime and preflight."""

    center_x_m: float
    center_y_m: float
    angle_x_rad: float
    angle_y_rad: float
    refractive_index: float


def reference_ray_samples(
    beam_cfg: dict[str, Any],
    elements: list[dict[str, Any]],
    *,
    wavelength_m: float,
    m2_x: float,
    m2_y: float,
) -> list[dict[str, Any]]:
    """Trace the first-order state represented by the sampled-field operators.

    This is diagnostic ray tracing, not a second propagation model.  Each
    update is the first-order phase-gradient/centroid equivalent of the
    corresponding FFT/ABCD operator.  In particular, decentered curved
    mirrors and quadratic phase screens contribute their local linear phase.
    """

    wavelength = float(wavelength_m)
    if wavelength <= 0.0 or float(m2_x) <= 0.0 or float(m2_y) <= 0.0:
        raise ValueError("reference-ray wavelength and M2 values must be positive")
    angle_x = math.radians(
        float(beam_cfg.get("incidence_angle_x_deg", 0.0))
    )
    angle_y = math.radians(
        float(beam_cfg.get("incidence_angle_y_deg", 0.0))
    )
    state = ReferenceRayState(
        center_x_m=(
            float(beam_cfg.get("center_x_m", 0.0))
            - axis_focus_m(beam_cfg, "x") * math.tan(angle_x)
        ),
        center_y_m=(
            float(beam_cfg.get("center_y_m", 0.0))
            - axis_focus_m(beam_cfg, "y") * math.tan(angle_y)
        ),
        angle_x_rad=angle_x,
        angle_y_rad=angle_y,
        refractive_index=float(
            beam_cfg.get("source_refractive_index", 1.0)
        ),
    )

    def sample(
        location: str,
        *,
        additional_angle_x_deg: float = 0.0,
        additional_angle_y_deg: float = 0.0,
    ) -> dict[str, Any]:
        local_x = state.angle_x_rad + math.radians(
            float(additional_angle_x_deg)
        )
        local_y = state.angle_y_rad + math.radians(
            float(additional_angle_y_deg)
        )
        magnitude = math.degrees(
            math.atan(
                math.hypot(math.tan(local_x), math.tan(local_y))
            )
        )
        return {
            "location": location,
            "angle_x_deg": math.degrees(local_x),
            "angle_y_deg": math.degrees(local_y),
            "axis_angle_magnitude_deg": magnitude,
            "center_x_m": state.center_x_m,
            "center_y_m": state.center_y_m,
            "refractive_index": state.refractive_index,
        }

    samples = [sample("source")]
    for index, element in enumerate(elements):
        raw_distance_before = element.get("distance_before_m", 0.0)
        distance_before = float(
            0.0 if raw_distance_before is None else raw_distance_before
        )
        if distance_before > 0.0:
            state.center_x_m += distance_before * math.tan(
                state.angle_x_rad
            )
            state.center_y_m += distance_before * math.tan(
                state.angle_y_rad
            )
        kind = str(element.get("type", ""))
        if kind == "free_space":
            length = float(element.get("length_m", 0.0))
            state.center_x_m += length * math.tan(state.angle_x_rad)
            state.center_y_m += length * math.tan(state.angle_y_rad)
        elif kind == "thin_lens":
            for axis in ("x", "y"):
                focal = _optional_focal_length(
                    element, f"focal_length_{axis}_m"
                )
                if focal is None:
                    continue
                center = getattr(state, f"center_{axis}_m")
                lens_center = float(
                    element.get(f"center_{axis}_m", 0.0)
                )
                angle = getattr(state, f"angle_{axis}_rad")
                setattr(
                    state,
                    f"angle_{axis}_rad",
                    math.atan(
                        math.tan(angle)
                        - (center - lens_center) / focal
                    ),
                )
        elif kind == "double_pass_lens_relay":
            matrix = double_pass_lens_relay_matrix(element)
            for axis in ("x", "y"):
                ray = np.asarray(
                    [
                        getattr(state, f"center_{axis}_m"),
                        getattr(state, f"angle_{axis}_rad"),
                    ],
                    dtype=float,
                )
                center, angle = matrix @ ray
                setattr(state, f"center_{axis}_m", float(center))
                setattr(state, f"angle_{axis}_rad", float(angle))
        elif kind == "thick_lens":
            lens_index = float(
                element.get(
                    "refractive_index", state.refractive_index
                )
            )
            next_index = float(
                element.get("n_after", state.refractive_index)
            )
            thickness = float(element.get("thickness_m", 0.0))
            for axis in ("x", "y"):
                center = getattr(state, f"center_{axis}_m")
                lens_center = float(
                    element.get(f"center_{axis}_m", 0.0)
                )
                relative = center - lens_center
                angle = getattr(state, f"angle_{axis}_rad")
                front_radius = _surface_radius(element, "front", axis)
                angle = (
                    (
                        0.0
                        if front_radius is None
                        else (
                            state.refractive_index - lens_index
                        )
                        * relative
                        / (front_radius * lens_index)
                    )
                    + state.refractive_index / lens_index * angle
                )
                center += thickness * math.tan(angle)
                relative = center - lens_center
                back_radius = _surface_radius(element, "back", axis)
                angle = (
                    (
                        0.0
                        if back_radius is None
                        else (lens_index - next_index)
                        * relative
                        / (back_radius * next_index)
                    )
                    + lens_index / next_index * angle
                )
                setattr(state, f"center_{axis}_m", center)
                setattr(state, f"angle_{axis}_rad", angle)
            state.refractive_index = next_index
        elif kind == "interface":
            state.angle_x_rad += math.radians(
                float(element.get("axis_angle_x_deg", 0.0))
            )
            state.angle_y_rad += math.radians(
                float(element.get("axis_angle_y_deg", 0.0))
            )
            next_index = float(
                element.get("n_after", state.refractive_index)
            )
            state.angle_x_rad, state.angle_y_rad = refract_axis_angles_rad(
                state.angle_x_rad,
                state.angle_y_rad,
                n_before=state.refractive_index,
                n_after=next_index,
            )
            state.refractive_index = next_index
        elif kind in {"tilt", "coordinate_break"}:
            state.center_x_m += float(element.get("shift_x_m", 0.0))
            state.center_y_m += float(element.get("shift_y_m", 0.0))
            state.angle_x_rad += math.radians(
                float(element.get("angle_x_deg", 0.0))
            )
            state.angle_y_rad += math.radians(
                float(element.get("angle_y_deg", 0.0))
            )
        elif kind == "mirror":
            for axis in ("x", "y"):
                radius = element.get(f"radius_{axis}_m")
                if radius not in (None, 0, 0.0):
                    center = getattr(state, f"center_{axis}_m")
                    mirror_center = float(
                        element.get(f"center_{axis}_m", 0.0)
                    )
                    angle = getattr(state, f"angle_{axis}_rad")
                    sine = (
                        math.sin(angle)
                        - 2.0
                        * (center - mirror_center)
                        / float(radius)
                    )
                    setattr(
                        state,
                        f"angle_{axis}_rad",
                        math.asin(float(np.clip(sine, -1.0, 1.0))),
                    )
                if bool(element.get(f"flip_{axis}", False)):
                    setattr(
                        state,
                        f"center_{axis}_m",
                        -getattr(state, f"center_{axis}_m"),
                    )
                    setattr(
                        state,
                        f"angle_{axis}_rad",
                        -getattr(state, f"angle_{axis}_rad"),
                    )
        elif kind == "phase_screen":
            for axis, m2 in (("x", m2_x), ("y", m2_y)):
                quadratic = float(
                    element.get(f"quadratic_{axis}_rad_m2", 0.0)
                )
                if quadratic == 0.0:
                    continue
                center = getattr(state, f"center_{axis}_m")
                angle = getattr(state, f"angle_{axis}_rad")
                carrier_cycles_m = (
                    state.refractive_index
                    * math.sin(angle)
                    / (float(m2) * wavelength)
                )
                carrier_cycles_m += quadratic * center / math.pi
                sine = (
                    carrier_cycles_m
                    * float(m2)
                    * wavelength
                    / state.refractive_index
                )
                setattr(
                    state,
                    f"angle_{axis}_rad",
                    math.asin(float(np.clip(sine, -1.0, 1.0))),
                )
        state.angle_x_rad += math.radians(
            float(element.get("turn_angle_x_deg", 0.0))
        )
        state.angle_y_rad += math.radians(
            float(element.get("turn_angle_y_deg", 0.0))
        )
        samples.append(
            sample(
                f"{kind or 'element'}[{index}]",
                additional_angle_x_deg=(
                    float(element.get("incidence_angle_x_deg", 0.0))
                    if kind == "gain_medium"
                    else 0.0
                ),
                additional_angle_y_deg=(
                    float(element.get("incidence_angle_y_deg", 0.0))
                    if kind == "gain_medium"
                    else 0.0
                ),
            )
        )
    return samples


def transverse_intensity_moments(
    intensity: np.ndarray,
    grid: Grid3D,
) -> dict[str, Any]:
    """Return centroid and ISO-style second moments for one transverse bin.

    The historical x/y D4-sigma radii remain available.  Principal-axis
    radii and orientation are also reported so a rotated elliptical beam is
    not silently treated as axis aligned.
    """

    values = np.asarray(intensity, dtype=float)
    if values.shape != (len(grid.y), len(grid.x)):
        raise ValueError("transverse intensity must match the grid y/x shape")
    total = float(grid.integrate_transverse(values))
    xx, yy = grid.transverse_mesh
    safe_total = max(total, np.finfo(float).tiny)
    centroid_x = float(grid.integrate_transverse(values * xx) / safe_total)
    centroid_y = float(grid.integrate_transverse(values * yy) / safe_total)
    variance_x = max(
        float(
            grid.integrate_transverse(values * (xx - centroid_x) ** 2)
            / safe_total
        ),
        0.0,
    )
    variance_y = max(
        float(
            grid.integrate_transverse(values * (yy - centroid_y) ** 2)
            / safe_total
        ),
        0.0,
    )
    covariance_xy = float(
        grid.integrate_transverse(
            values * (xx - centroid_x) * (yy - centroid_y)
        )
        / safe_total
    )
    discriminant = math.sqrt(
        max((variance_x - variance_y) ** 2 + 4.0 * covariance_xy**2, 0.0)
    )
    principal_major_variance = max(
        0.5 * (variance_x + variance_y + discriminant),
        0.0,
    )
    principal_minor_variance = max(
        0.5 * (variance_x + variance_y - discriminant),
        0.0,
    )
    circular_tolerance = 64.0 * np.finfo(float).eps * max(
        variance_x + variance_y,
        np.finfo(float).tiny,
    )
    major_axis_angle_deg = (
        None
        if discriminant <= circular_tolerance
        else math.degrees(
            0.5 * math.atan2(2.0 * covariance_xy, variance_x - variance_y)
        )
    )
    radius_x = 2.0 * math.sqrt(variance_x)
    radius_y = 2.0 * math.sqrt(variance_y)
    return {
        "total": total,
        "centroid_x_m": centroid_x,
        "centroid_y_m": centroid_y,
        "radius_x_D4sigma_m": radius_x,
        "radius_y_D4sigma_m": radius_y,
        "covariance_xy_m2": covariance_xy,
        "radius_major_D4sigma_m": 2.0 * math.sqrt(principal_major_variance),
        "radius_minor_D4sigma_m": 2.0 * math.sqrt(principal_minor_variance),
        "major_axis_angle_deg": major_axis_angle_deg,
    }


def transverse_boundary_energy_fractions(
    intensity: np.ndarray,
    grid: Grid3D,
    *,
    boundary_fraction: float = 0.05,
) -> dict[str, float | str]:
    """Return energy fractions in physical outer strips of a numerical window.

    FFT propagation is periodic.  Non-negligible energy in these strips is a
    direct warning that a field can wrap from one edge to the opposite edge.
    ``boundary_fraction`` is the fraction of the *physical window width* on
    each side, not the fraction of transverse samples.  This distinction is
    essential on a centre-stretched grid: a few coarse edge cells can span a
    much larger physical interval than the same number of refined centre
    cells.  Cells cut by a strip boundary contribute in proportion to their
    physical overlap, under the same piecewise-constant cell convention used
    by :meth:`Grid3D.integrate_transverse`.

    The diagnostic does not absorb or renormalize the field.
    """

    values = np.asarray(intensity, dtype=float)
    if values.shape != (len(grid.y), len(grid.x)):
        raise ValueError("transverse intensity must match the grid y/x shape")
    fraction = float(boundary_fraction)
    if not 0.0 < fraction <= 0.5:
        raise ValueError("boundary_fraction must be in (0, 0.5]")

    def physical_boundary_cell_fractions(
        edges: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        widths = np.diff(edges)
        strip_width = fraction * float(edges[-1] - edges[0])
        left_stop = float(edges[0]) + strip_width
        right_start = float(edges[-1]) - strip_width
        left_overlap = np.clip(
            np.minimum(edges[1:], left_stop) - edges[:-1],
            0.0,
            widths,
        )
        right_overlap = np.clip(
            edges[1:] - np.maximum(edges[:-1], right_start),
            0.0,
            widths,
        )
        overlap = np.clip(left_overlap + right_overlap, 0.0, widths)
        return overlap / widths, strip_width

    x_cell_fractions, x_strip_width = physical_boundary_cell_fractions(
        grid.x_edges
    )
    y_cell_fractions, y_strip_width = physical_boundary_cell_fractions(
        grid.y_edges
    )
    weighted_energy = values * grid.cell_areas
    total = float(np.sum(weighted_energy))
    safe_total = max(total, np.finfo(float).tiny)
    x_weights = x_cell_fractions[None, :]
    y_weights = y_cell_fractions[:, None]
    x_energy = float(np.sum(weighted_energy * x_weights))
    y_energy = float(np.sum(weighted_energy * y_weights))
    # Inclusion-exclusion gives the exact union area for the separable x/y
    # boundary strips, including partially intersected corner cells.
    either_weights = x_weights + y_weights - x_weights * y_weights
    either_energy = float(np.sum(weighted_energy * either_weights))
    xx, yy = grid.transverse_mesh
    center_x = float(np.sum(weighted_energy * xx) / safe_total)
    center_y = float(np.sum(weighted_energy * yy) / safe_total)
    x_second_moment = float(np.sum(weighted_energy * (xx - center_x) ** 2))
    y_second_moment = float(np.sum(weighted_energy * (yy - center_y) ** 2))
    return {
        "boundary_region_fraction": fraction,
        "boundary_region_basis": "physical_window_width_each_side",
        "boundary_region_width_x_m": x_strip_width,
        "boundary_region_width_y_m": y_strip_width,
        "boundary_energy_fraction_x": x_energy / safe_total,
        "boundary_energy_fraction_y": y_energy / safe_total,
        "boundary_energy_fraction": either_energy / safe_total,
        "boundary_x_second_moment_fraction": float(
            np.sum(weighted_energy * (xx - center_x) ** 2 * x_weights)
            / max(x_second_moment, np.finfo(float).tiny)
        ),
        "boundary_y_second_moment_fraction": float(
            np.sum(weighted_energy * (yy - center_y) ** 2 * y_weights)
            / max(y_second_moment, np.finfo(float).tiny)
        ),
    }


def transverse_volume_boundary_diagnostics(
    intensity: np.ndarray,
    grid: Grid3D,
    *,
    boundary_fraction: float = 0.05,
) -> dict[str, Any]:
    """Summarize the worst physical-edge occupancy across a sampled volume."""

    values = np.asarray(intensity, dtype=float)
    if values.shape != grid.shape:
        raise ValueError("volume intensity must match the grid z/y/x shape")
    samples = [
        transverse_boundary_energy_fractions(
            values[z_index],
            grid,
            boundary_fraction=boundary_fraction,
        )
        for z_index in range(len(grid.z))
    ]
    maximum_index = max(
        range(len(samples)),
        key=lambda index: float(samples[index]["boundary_energy_fraction"]),
    )
    maximum = samples[maximum_index]
    return {
        "method": "outer_physical_window_energy_fraction",
        "boundary_region_fraction": float(boundary_fraction),
        "boundary_region_basis": "physical_window_width_each_side",
        "boundary_region_width_x_m": float(
            maximum["boundary_region_width_x_m"]
        ),
        "boundary_region_width_y_m": float(
            maximum["boundary_region_width_y_m"]
        ),
        "maximum_boundary_energy_fraction": float(
            maximum["boundary_energy_fraction"]
        ),
        "maximum_boundary_energy_fraction_x": float(
            max(sample["boundary_energy_fraction_x"] for sample in samples)
        ),
        "maximum_boundary_energy_fraction_y": float(
            max(sample["boundary_energy_fraction_y"] for sample in samples)
        ),
        "z_bin_index": int(maximum_index),
        "grid_nx": int(len(grid.x)),
        "grid_ny": int(len(grid.y)),
        "grid_window_x_m": float(grid.x_edges[-1] - grid.x_edges[0]),
        "grid_window_y_m": float(grid.y_edges[-1] - grid.y_edges[0]),
    }


def sampled_field_scalar_diagnostics(
    field_value: np.ndarray,
    grid: Grid3D,
    *,
    boundary_fraction: float = 0.05,
    high_frequency_nyquist_fraction: float = 0.8,
) -> dict[str, Any]:
    """Summarize one sampled field without retaining any field-sized array."""

    field = np.asarray(field_value, dtype=complex)
    if field.shape[-2:] != (len(grid.y), len(grid.x)):
        raise ValueError("sampled diagnostic field must end in grid y/x dimensions")
    intensity = np.abs(field) ** 2
    if intensity.ndim > 2:
        intensity = np.sum(
            intensity,
            axis=tuple(range(intensity.ndim - 2)),
        )
    moments = transverse_intensity_moments(intensity, grid)
    boundary = transverse_boundary_energy_fractions(
        intensity,
        grid,
        boundary_fraction=boundary_fraction,
    )
    complex_bytes_per_channel = max(
        len(grid.y) * len(grid.x) * np.dtype(np.complex128).itemsize,
        1,
    )
    channel_chunk_size = max(
        1,
        int((32 * 1024**2) // complex_bytes_per_channel),
    )
    periodic_quality = sampled_beam_quality(
        field,
        grid,
        channel_chunk_size=channel_chunk_size,
    )
    finite_difference_quality = sampled_beam_quality(
        field,
        grid,
        derivative_method="finite_difference",
        channel_chunk_size=channel_chunk_size,
    )
    return {
        "schema_version": 1,
        "retains_field_arrays": False,
        "grid": {
            "nx": int(len(grid.x)),
            "ny": int(len(grid.y)),
            "window_x_m": float(grid.x_edges[-1] - grid.x_edges[0]),
            "window_y_m": float(grid.y_edges[-1] - grid.y_edges[0]),
            "uniform_xy": bool(grid.is_uniform_xy),
        },
        "intensity_moments": {
            key: value
            for key, value in moments.items()
            if key != "total"
        },
        "beam_quality": periodic_quality,
        "beam_quality_finite_difference": finite_difference_quality,
        "boundary": boundary,
        "spatial_frequency": sampled_spatial_frequency_diagnostics(
            field,
            grid,
            high_frequency_nyquist_fraction=(
                high_frequency_nyquist_fraction
            ),
            channel_chunk_size=channel_chunk_size,
        ),
    }


def _unique_key(name: str, existing: dict[str, np.ndarray]) -> str:
    cleaned = "".join(character if character.isalnum() or character in "_-" else "_" for character in name).strip("_") or "plane"
    if cleaned not in existing:
        return cleaned
    index = 2
    while f"{cleaned}_{index}" in existing:
        index += 1
    return f"{cleaned}_{index}"


def expand_path_distances(
    elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand ``distance_before_m`` into explicit free-space operators.

    ``distance_before_m`` is part of the public sequential-train schema, so it
    must mean the same thing whether a caller supplies a complete train, a
    pre-gain slice, or a between-pass route.  Keeping the expansion here also
    prevents flight-time accounting from seeing a path that the optical field
    did not actually traverse.
    """

    expanded: list[dict[str, Any]] = []
    for index, raw in enumerate(elements):
        element = dict(raw)
        raw_distance_before = element.pop("distance_before_m", 0.0)
        distance_before = float(
            0.0 if raw_distance_before is None else raw_distance_before
        )
        if not math.isfinite(distance_before) or distance_before < 0.0:
            raise ValueError("distance_before_m must be finite and non-negative")
        if distance_before > 0.0:
            expanded.append(
                {
                    "type": "free_space",
                    "name": f"path_space_before_{index + 1}",
                    "length_m": distance_before,
                }
            )
        expanded.append(element)
    return expanded


def split_gain_path(elements: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    expanded = expand_path_distances(elements)
    positions = [index for index, element in enumerate(expanded) if element.get("type") == "gain_medium"]
    if len(positions) > 1:
        raise ValueError("One optical train currently supports one gain_medium marker; use passes for repeated traversal")
    if not positions:
        return expanded, {"type": "gain_medium", "name": "gain_medium"}, []
    position = positions[0]
    return expanded[:position], dict(expanded[position]), expanded[position + 1 :]


def angular_spectrum(
    field_value: np.ndarray,
    grid: Grid3D,
    distance_m: float,
    wavelength_m: float,
    refractive_index: float,
    m2_x: float = 1.0,
    m2_y: float = 1.0,
) -> np.ndarray:
    """Scalar paraxial angular-spectrum step on the grid's transverse plane."""
    if not grid.is_uniform_xy:
        raise ValueError("angular-spectrum FFT requires a uniform x/y grid")
    transfer = angular_spectrum_transfer(
        grid,
        distance_m,
        wavelength_m,
        refractive_index,
        m2_x,
        m2_y,
    )
    return ifft2(fft2(field_value) * transfer)


def angular_spectrum_transfer(
    grid: Grid3D,
    distance_m: float,
    wavelength_m: float,
    refractive_index: float,
    m2_x: float = 1.0,
    m2_y: float = 1.0,
) -> np.ndarray:
    """Return a cached paraxial transfer function for a uniform grid."""

    if not grid.is_uniform_xy:
        raise ValueError("angular-spectrum FFT requires a uniform x/y grid")
    return _cached_angular_spectrum_transfer(
        len(grid.y),
        len(grid.x),
        float(grid.dy),
        float(grid.dx),
        float(distance_m),
        float(wavelength_m),
        float(refractive_index),
        float(m2_x),
        float(m2_y),
    )


@lru_cache(maxsize=128)
def _cached_angular_spectrum_transfer(
    ny: int,
    nx: int,
    dy: float,
    dx: float,
    distance_m: float,
    wavelength_m: float,
    refractive_index: float,
    m2_x: float,
    m2_y: float,
) -> np.ndarray:
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    fxx, fyy = np.meshgrid(fx, fy, indexing="xy")
    transfer = np.exp(
        -1j
        * math.pi
        * wavelength_m
        * distance_m
        / refractive_index
        * (m2_x * fxx**2 + m2_y * fyy**2)
    )
    transfer.setflags(write=False)
    return transfer


def interface_power_transmission(
    n_before: float,
    n_after: float,
    model: str,
    specified_power_transmission: float | None = None,
    incidence_angle_deg: float = 0.0,
    polarization: str = "unpolarized",
) -> float:
    theta_i = math.radians(abs(float(incidence_angle_deg)))
    sin_theta_t = n_before * math.sin(theta_i) / n_after
    if abs(sin_theta_t) >= 1.0:
        if model == "fresnel_angle":
            return 0.0
        raise ValueError(
            "The configured interface is above the total-internal-reflection "
            "critical angle. A transmitted-only ideal/coating/normal-Fresnel "
            "element cannot represent this geometry; use a reflected route."
        )
    if model == "ideal":
        return 1.0
    if model == "coating":
        return 1.0 if specified_power_transmission is None else float(specified_power_transmission)
    if model == "fresnel":
        return 4.0 * n_before * n_after / (n_before + n_after) ** 2
    if model == "fresnel_angle":
        if polarization not in {"s", "p", "unpolarized"}:
            raise ValueError("Fresnel polarization must be s, p, or unpolarized")
        cos_i = math.cos(theta_i)
        cos_t = math.sqrt(max(1.0 - sin_theta_t**2, 0.0))
        transmission_s = (
            4.0
            * n_before
            * n_after
            * cos_i
            * cos_t
            / max((n_before * cos_i + n_after * cos_t) ** 2, np.finfo(float).tiny)
        )
        transmission_p = (
            4.0
            * n_before
            * n_after
            * cos_i
            * cos_t
            / max((n_after * cos_i + n_before * cos_t) ** 2, np.finfo(float).tiny)
        )
        if polarization == "s":
            return float(np.clip(transmission_s, 0.0, 1.0))
        if polarization == "p":
            return float(np.clip(transmission_p, 0.0, 1.0))
        return float(np.clip(0.5 * (transmission_s + transmission_p), 0.0, 1.0))
    raise ValueError(f"Unknown interface model {model!r}")


def apply_interface(
    field_value: np.ndarray,
    n_before: float,
    n_after: float,
    model: str,
    specified_power_transmission: float | None = None,
    incidence_angle_deg: float = 0.0,
    polarization: str = "unpolarized",
) -> np.ndarray:
    transmission = interface_power_transmission(
        n_before,
        n_after,
        model,
        specified_power_transmission,
        incidence_angle_deg,
        polarization,
    )
    if not 0.0 <= transmission <= 1.0:
        raise ValueError("Interface power transmission must be in [0, 1]")
    return field_value * math.sqrt(transmission)


def _element_response(element: dict[str, Any], key: str, wavelength_m: float, default: float = 1.0) -> float:
    value = response_value(element, key, wavelength_m, default)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1] at {wavelength_m:.6g} m")
    return value


def _element_dispersion(element: dict[str, Any]) -> dict[str, Any] | None:
    """Return valid optional dispersion data and tolerate legacy UI coercion."""

    value = element.get("dispersion")
    return value if isinstance(value, dict) and value else None


def double_pass_lens_relay_matrix(element: dict[str, Any]) -> np.ndarray:
    """Return the unfolded round-trip ABCD matrix of an ideal two-lens relay.

    The sequence is object plane -> F3 -> F4 -> flat mirror -> F4 -> F3 ->
    object plane.  Positive distances are travelled along the local optical
    direction on both legs, as in the unfolded-ray convention used by the
    sequential propagator.
    """

    object_distance = float(element["object_to_first_lens_m"])
    separation = float(element["lens_separation_m"])
    mirror_distance = float(element["second_lens_to_mirror_m"])
    first_focal = float(element["first_focal_length_m"])
    second_focal = float(element["second_focal_length_m"])

    def propagation(distance: float) -> np.ndarray:
        return np.asarray([[1.0, distance], [0.0, 1.0]])

    def lens(focal_length: float) -> np.ndarray:
        return np.asarray([[1.0, 0.0], [-1.0 / focal_length, 1.0]])

    operators = [
        propagation(object_distance),
        lens(first_focal),
        propagation(separation),
        lens(second_focal),
        propagation(mirror_distance),
        propagation(mirror_distance),
        lens(second_focal),
        propagation(separation),
        lens(first_focal),
        propagation(object_distance),
    ]
    matrix = np.eye(2)
    for operator in operators:
        matrix = operator @ matrix
    return matrix


def _apply_double_pass_lens_relay(
    field_value: np.ndarray,
    element: dict[str, Any],
    xx: np.ndarray,
    yy: np.ndarray,
    wavelength_m: float,
    refractive_index: float,
    m2_x: float,
    m2_y: float,
) -> np.ndarray:
    """Apply the exact B=0 linear-canonical operator on the existing grid."""

    matrix = double_pass_lens_relay_matrix(element)
    a, b, c, d = matrix.ravel()
    scale = max(
        abs(float(element["object_to_first_lens_m"])),
        abs(float(element["lens_separation_m"])),
        abs(float(element["second_lens_to_mirror_m"])),
        1.0,
    )
    if abs(b) > 1e-10 * scale or not math.isclose(
        abs(a), 1.0, rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError(
            "double_pass_lens_relay currently requires unit-magnification B=0 imaging; "
            f"its configured ABCD matrix is [[{a:.9g}, {b:.9g}], [{c:.9g}, {d:.9g}]]"
        )
    result = np.asarray(field_value, dtype=complex).copy()
    if a < 0.0:
        result = np.flip(result, axis=(-2, -1)).copy()
    center_x = float(element.get("center_x_m", 0.0))
    center_y = float(element.get("center_y_m", 0.0))
    k = 2.0 * math.pi * float(refractive_index) / float(wavelength_m)
    phase = 0.5 * k * c / a * (
        (xx - center_x) ** 2 / float(m2_x)
        + (yy - center_y) ** 2 / float(m2_y)
    )
    result *= np.exp(1j * phase)
    result *= math.sqrt(
        _element_response(element, "power_transmission", wavelength_m)
    )
    return result


def propagate_fft_train(
    field_value: np.ndarray,
    elements: list[dict[str, Any]],
    grid: Grid3D,
    wavelength_m: float,
    refractive_index: float,
    quantity_name: str,
    interface_model: str = "ideal",
    m2_x: float = 1.0,
    m2_y: float = 1.0,
    record_planes: bool = True,
) -> tuple[np.ndarray, float, OpticalTrace]:
    """Apply sequential element operators to a sampled complex field."""
    result = field_value.astype(complex, copy=True)
    current_n = float(refractive_index)
    trace = OpticalTrace()
    if record_planes:
        trace.add("source", result, grid, quantity_name)
    xx, yy = grid.transverse_mesh
    for index, element in enumerate(expand_path_distances(elements)):
        kind = element.get("type")
        name = str(element.get("name", f"{index + 1}_{kind}"))
        if kind in {"observation", "plane", "dispersive"}:
            pass
        elif kind == "free_space":
            length = float(element["length_m"])
            result = angular_spectrum(result, grid, length, wavelength_m, current_n, m2_x, m2_y)
            absorption = float(element.get("absorption_m_inv", 0.0))
            result *= math.exp(-0.5 * absorption * length)
        elif kind == "thin_lens":
            fx = _optional_focal_length(element, "focal_length_x_m")
            fy = _optional_focal_length(element, "focal_length_y_m")
            cx = float(element.get("center_x_m", 0.0))
            cy = float(element.get("center_y_m", 0.0))
            phase = np.zeros_like(xx)
            # The sampled M2 surrogate propagates with lambda_eff=M2*lambda.
            # Its quadratic phase must use the same effective wavelength or a
            # physical lens would become M2 times too strong at the FFT handoff.
            if fx is not None:
                phase += (xx - cx) ** 2 / (fx * m2_x)
            if fy is not None:
                phase += (yy - cy) ** 2 / (fy * m2_y)
            result *= np.exp(-1j * math.pi * current_n / wavelength_m * phase)
            result *= math.sqrt(_element_response(element, "power_transmission", wavelength_m))
            result = _apply_optional_aperture(result, element, xx, yy)
        elif kind == "double_pass_lens_relay":
            result = _apply_double_pass_lens_relay(
                result,
                element,
                xx,
                yy,
                wavelength_m,
                current_n,
                m2_x,
                m2_y,
            )
        elif kind == "thick_lens":
            lens_n = float(element["refractive_index"])
            n_after = float(element.get("n_after", current_n))
            result *= _curved_interface_phase(
                element,
                xx,
                yy,
                wavelength_m,
                current_n,
                lens_n,
                "front",
                m2_x,
                m2_y,
            )
            result = apply_interface(
                result,
                current_n,
                lens_n,
                str(element.get("front_model", element.get("model", interface_model))),
                _element_response(element, "front_power_transmission", wavelength_m),
                float(element.get("front_incidence_angle_deg", element.get("incidence_angle_deg", 0.0))),
                str(element.get("polarization", "unpolarized")),
            )
            thickness = float(element["thickness_m"])
            result = angular_spectrum(result, grid, thickness, wavelength_m, lens_n, m2_x, m2_y)
            result *= math.exp(-0.5 * float(element.get("absorption_m_inv", 0.0)) * thickness)
            result *= _curved_interface_phase(
                element,
                xx,
                yy,
                wavelength_m,
                lens_n,
                n_after,
                "back",
                m2_x,
                m2_y,
            )
            result = apply_interface(
                result,
                lens_n,
                n_after,
                str(element.get("back_model", element.get("model", interface_model))),
                _element_response(element, "back_power_transmission", wavelength_m),
                float(element.get("back_incidence_angle_deg", element.get("incidence_angle_deg", 0.0))),
                str(element.get("polarization", "unpolarized")),
            )
            current_n = n_after
            result = _apply_optional_aperture(result, element, xx, yy)
        elif kind == "aperture":
            result *= _aperture_mask(element, xx, yy)
            result *= math.sqrt(
                _element_response(element, "power_transmission", wavelength_m)
            )
        elif kind == "interface":
            next_n = float(element["n_after"])
            model = str(element.get("model", interface_model))
            result = apply_interface(
                result,
                current_n,
                next_n,
                model,
                _element_response(element, "power_transmission", wavelength_m),
                float(element.get("incidence_angle_deg", 0.0)),
                str(element.get("polarization", "unpolarized")),
            )
            current_n = next_n
        elif kind == "tilt":
            result = shift_field(
                result,
                grid,
                float(element.get("shift_x_m", 0.0)),
                float(element.get("shift_y_m", 0.0)),
            )
            angle_x = math.radians(float(element.get("angle_x_deg", 0.0)))
            angle_y = math.radians(float(element.get("angle_y_deg", 0.0)))
            direction_x, direction_y, _ = axis_direction_components(
                angle_x,
                angle_y,
            )
            k0 = 2.0 * math.pi / wavelength_m
            result *= np.exp(
                1j
                * k0
                * current_n
                * (
                    direction_x * xx / m2_x
                    + direction_y * yy / m2_y
                )
            )
        elif kind == "coordinate_break":
            result = shift_field(result, grid, float(element.get("shift_x_m", 0.0)), float(element.get("shift_y_m", 0.0)))
            angle_x = math.radians(float(element.get("angle_x_deg", 0.0)))
            angle_y = math.radians(float(element.get("angle_y_deg", 0.0)))
            direction_x, direction_y, _ = axis_direction_components(
                angle_x,
                angle_y,
            )
            k0 = 2.0 * math.pi / wavelength_m
            result *= np.exp(
                1j
                * k0
                * current_n
                * (
                    direction_x * xx / m2_x
                    + direction_y * yy / m2_y
                )
            )
        elif kind == "mirror":
            curvature_x = element.get("radius_x_m")
            curvature_y = element.get("radius_y_m")
            phase = np.zeros_like(xx)
            if curvature_x not in (None, 0):
                phase += (
                    (xx - float(element.get("center_x_m", 0.0))) ** 2
                    / (float(curvature_x) * m2_x)
                )
            if curvature_y not in (None, 0):
                phase += (
                    (yy - float(element.get("center_y_m", 0.0))) ** 2
                    / (float(curvature_y) * m2_y)
                )
            result *= np.exp(-1j * 2.0 * math.pi * current_n / wavelength_m * phase)
            result *= math.sqrt(_element_response(element, "power_reflectivity", wavelength_m))
            if element.get("flip_x", False):
                result = np.flip(result, axis=-1).copy()
            if element.get("flip_y", False):
                result = np.flip(result, axis=-2).copy()
        elif kind == "phase_screen":
            phase = float(element.get("constant_phase_rad") or 0.0)
            quadratic_x = float(element.get("quadratic_x_rad_m2") or 0.0)
            quadratic_y = float(element.get("quadratic_y_rad_m2") or 0.0)
            result *= np.exp(1j * (phase + quadratic_x * xx**2 + quadratic_y * yy**2))
            result *= math.sqrt(
                _element_response(element, "power_transmission", wavelength_m)
            )
        elif kind == "loss":
            transmission = _element_response(element, "power_transmission", wavelength_m)
            result *= math.sqrt(transmission)
        else:
            raise ValueError(f"Unsupported optical element type {kind!r}")
        dispersion = _element_dispersion(element)
        if dispersion:
            frequency = np.asarray([LIGHT_SPEED / wavelength_m])
            phase = float(phase_from_coefficients(frequency, dispersion)[0])
            result *= np.exp(1j * phase)
        if record_planes or bool(element.get("record", False)):
            trace.add(name, result, grid, quantity_name)
    return result, current_n, trace


def shift_field(field_value: np.ndarray, grid: Grid3D, shift_x_m: float, shift_y_m: float) -> np.ndarray:
    if not grid.is_uniform_xy:
        raise ValueError("Fourier field shifts require a uniform x/y grid")
    fx = np.fft.fftfreq(len(grid.x), d=grid.dx)
    fy = np.fft.fftfreq(len(grid.y), d=grid.dy)
    fxx, fyy = np.meshgrid(fx, fy, indexing="xy")
    phase = np.exp(-2j * math.pi * (fxx * shift_x_m + fyy * shift_y_m))
    return ifft2(fft2(field_value) * phase)


def gaussian_abcd_to_plane(
    beam_cfg: dict[str, Any],
    total_quantity: float,
    elements: list[dict[str, Any]],
    grid: Grid3D,
    quantity_name: str,
    interface_model: str = "ideal",
    record_plane: bool = True,
    *,
    include_tilt_phase: bool = False,
    diagnostics: dict[str, Any] | None = None,
    warn_non_gaussian_approximation: bool = True,
) -> tuple[np.ndarray, float, OpticalTrace, list[str]]:
    """Fast astigmatic Gaussian q propagation through the pre-gain train.

    ``include_tilt_phase`` is deliberately opt-in for backward compatibility.
    Pure ABCD intensity propagation only needs the centroid ray, while a field
    handed to a following FFT domain must also carry the ray's linear phase.
    The optional diagnostics mapping exposes the analytic retained quantity and
    sampled gain-plane fraction without changing the long-standing return
    signature used by the existing ABCD solvers.
    """
    wavelength = float(beam_cfg["wavelength_m"])
    # ``wavelength_m`` is the vacuum wavelength throughout the public model.
    # The Gaussian q parameter in a homogeneous source medium therefore starts
    # with z_R = pi*n*w0^2/(M2*lambda0), and subsequent planar interfaces scale
    # q by n_after/n_before.  Starting from n=1 here made ABCD disagree with the
    # FFT and ray backends whenever the configured source was immersed.
    n = float(beam_cfg.get("source_refractive_index", 1.0))
    m2x, m2y = float(beam_cfg.get("m2_x", 1.0)), float(beam_cfg.get("m2_y", 1.0))
    focus_x = axis_focus_m(beam_cfg, "x")
    focus_y = axis_focus_m(beam_cfg, "y")
    qx = complex(-focus_x, math.pi * n * float(beam_cfg["waist_x_m"]) ** 2 / (m2x * wavelength))
    qy = complex(-focus_y, math.pi * n * float(beam_cfg["waist_y_m"]) ** 2 / (m2y * wavelength))
    explicit_wavefront = beam_cfg.get("wavefront_radius_m")
    if explicit_wavefront not in (None, 0):
        radius = float(explicit_wavefront)
        if not math.isfinite(radius):
            raise ValueError("wavefront_radius_m must be finite when supplied")
        source_wx, _ = _q_radius_curvature(qx, wavelength, n, m2x)
        source_wy, _ = _q_radius_curvature(qy, wavelength, n, m2y)
        qx = 1.0 / complex(
            1.0 / radius,
            -m2x * wavelength / (math.pi * n * source_wx**2),
        )
        qy = 1.0 / complex(
            1.0 / radius,
            -m2y * wavelength / (math.pi * n * source_wy**2),
        )
    tx = math.radians(float(beam_cfg.get("incidence_angle_x_deg", 0.0)))
    ty = math.radians(float(beam_cfg.get("incidence_angle_y_deg", 0.0)))
    # Public centres are waist-plane centres.  The ABCD state starts at the
    # source plane z=0, exactly like BeamProfile/source_field and the realtime
    # preview, so an off-plane tilted waist must be back-projected first.
    cx = float(beam_cfg.get("center_x_m", 0.0)) - focus_x * math.tan(tx)
    cy = float(beam_cfg.get("center_y_m", 0.0)) - focus_y * math.tan(ty)
    retained = float(total_quantity)
    global_phase = 0.0
    warnings: list[str] = []
    source_rotation_deg = float(beam_cfg.get("rotation_deg", 0.0))
    rotated_astigmatism_warned = False
    if explicit_wavefront not in (None, 0):
        warnings.append(
            "ABCD used wavefront_radius_m as the source-plane curvature; it overrides the curvature implied by focus_z_m/focus_x_m/focus_y_m."
        )
    if warn_non_gaussian_approximation and (
        beam_cfg.get("profile", "gaussian") != "gaussian"
        or float(beam_cfg.get("order_x", 1.0)) != 1.0
        or float(beam_cfg.get("order_y", 1.0)) != 1.0
    ):
        warnings.append("ABCD mode approximates the configured non-Gaussian source by an astigmatic Gaussian.")
    for element in expand_path_distances(elements):
        kind = element.get("type")
        if kind in {"observation", "plane"}:
            continue
        if kind == "dispersive":
            pass
        elif kind == "free_space":
            length = float(element["length_m"])
            qx += length
            qy += length
            cx += length * math.tan(tx)
            cy += length * math.tan(ty)
            retained *= math.exp(-float(element.get("absorption_m_inv", 0.0)) * length)
        elif kind == "thin_lens":
            fx = _optional_focal_length(element, "focal_length_x_m")
            fy = _optional_focal_length(element, "focal_length_y_m")
            if (
                source_rotation_deg != 0.0
                and fx != fy
                and not rotated_astigmatism_warned
            ):
                warnings.append(
                    "ABCD preserves the source ellipse rotation through isotropic optics, but a rotated astigmatic beam passing axis-dependent lens power needs a coupled 4x4 ray matrix; the current qx/qy result is an uncoupled approximation."
                )
                rotated_astigmatism_warned = True
            lens_x, lens_y = float(element.get("center_x_m", 0.0)), float(element.get("center_y_m", 0.0))
            if fx is not None:
                qx = 1.0 / (1.0 / qx - 1.0 / fx)
                tx -= (cx - lens_x) / fx
            if fy is not None:
                qy = 1.0 / (1.0 / qy - 1.0 / fy)
                ty -= (cy - lens_y) / fy
            retained *= _element_response(element, "power_transmission", wavelength)
        elif kind == "double_pass_lens_relay":
            matrix = double_pass_lens_relay_matrix(element)
            a, b, c, d = matrix.ravel()
            qx = (a * qx + b) / (c * qx + d)
            qy = (a * qy + b) / (c * qy + d)
            cx, tx = a * cx + b * tx, c * cx + d * tx
            cy, ty = a * cy + b * ty, c * cy + d * ty
            retained *= _element_response(
                element, "power_transmission", wavelength
            )
        elif kind == "thick_lens":
            lens_n = float(element["refractive_index"])
            n_after = float(element.get("n_after", n))
            thickness = float(element["thickness_m"])
            front_x = _surface_radius(element, "front", "x")
            front_y = _surface_radius(element, "front", "y")
            back_x = _surface_radius(element, "back", "x")
            back_y = _surface_radius(element, "back", "y")
            if (
                source_rotation_deg != 0.0
                and (front_x != front_y or back_x != back_y)
                and not rotated_astigmatism_warned
            ):
                warnings.append(
                    "ABCD preserves the source ellipse rotation through isotropic optics, but a rotated astigmatic beam passing axis-dependent lens power needs a coupled 4x4 ray matrix; the current qx/qy result is an uncoupled approximation."
                )
                rotated_astigmatism_warned = True
            for axis in ("x", "y"):
                q = qx if axis == "x" else qy
                center = cx if axis == "x" else cy
                angle = tx if axis == "x" else ty
                lens_center = float(element.get(f"center_{axis}_m", 0.0))
                radius_front = _surface_radius(element, "front", axis)
                q, center, angle = _refract_q_ray(
                    q,
                    center,
                    angle,
                    n,
                    lens_n,
                    radius_front,
                    surface_center_m=lens_center,
                )
                q += thickness
                center += thickness * math.tan(angle)
                radius_back = _surface_radius(element, "back", axis)
                q, center, angle = _refract_q_ray(
                    q,
                    center,
                    angle,
                    lens_n,
                    n_after,
                    radius_back,
                    surface_center_m=lens_center,
                )
                if axis == "x":
                    qx, cx, tx = q, center, angle
                else:
                    qy, cy, ty = q, center, angle
            retained *= interface_power_transmission(
                n,
                lens_n,
                str(element.get("front_model", element.get("model", interface_model))),
                _element_response(element, "front_power_transmission", wavelength),
                float(element.get("front_incidence_angle_deg", element.get("incidence_angle_deg", 0.0))),
                str(element.get("polarization", "unpolarized")),
            )
            retained *= math.exp(-float(element.get("absorption_m_inv", 0.0)) * thickness)
            retained *= interface_power_transmission(
                lens_n,
                n_after,
                str(element.get("back_model", element.get("model", interface_model))),
                _element_response(element, "back_power_transmission", wavelength),
                float(element.get("back_incidence_angle_deg", element.get("incidence_angle_deg", 0.0))),
                str(element.get("polarization", "unpolarized")),
            )
            n = n_after
        elif kind == "interface":
            next_n = float(element["n_after"])
            has_axis_break = bool(
                float(element.get("axis_angle_x_deg", 0.0))
                or float(element.get("axis_angle_y_deg", 0.0))
            )
            tx += math.radians(float(element.get("axis_angle_x_deg", 0.0)))
            ty += math.radians(float(element.get("axis_angle_y_deg", 0.0)))
            incidence_angle_deg = (
                math.degrees(
                    math.atan(
                        math.hypot(math.tan(tx), math.tan(ty))
                    )
                )
                if has_axis_break
                else float(element.get("incidence_angle_deg", 0.0))
            )
            retained *= interface_power_transmission(
                n,
                next_n,
                str(element.get("model", interface_model)),
                _element_response(element, "power_transmission", wavelength),
                incidence_angle_deg,
                str(element.get("polarization", "unpolarized")),
            )
            qx *= next_n / n
            qy *= next_n / n
            tx, ty = refract_axis_angles_rad(
                tx,
                ty,
                n_before=n,
                n_after=next_n,
            )
            n = next_n
        elif kind in {"tilt", "coordinate_break"}:
            cx += float(element.get("shift_x_m", 0.0))
            cy += float(element.get("shift_y_m", 0.0))
            tx += math.radians(float(element.get("angle_x_deg", 0.0)))
            ty += math.radians(float(element.get("angle_y_deg", 0.0)))
        elif kind == "mirror":
            rx, ry = element.get("radius_x_m"), element.get("radius_y_m")
            if (
                source_rotation_deg != 0.0
                and rx != ry
                and not rotated_astigmatism_warned
            ):
                warnings.append(
                    "ABCD preserves the source ellipse rotation through isotropic optics, but a rotated astigmatic beam passing axis-dependent mirror power needs a coupled 4x4 ray matrix; the current qx/qy result is an uncoupled approximation."
                )
                rotated_astigmatism_warned = True
            if rx not in (None, 0):
                qx = 1.0 / (1.0 / qx - 2.0 / float(rx))
                tx -= 2.0 * (
                    cx - float(element.get("center_x_m", 0.0))
                ) / float(rx)
            if ry not in (None, 0):
                qy = 1.0 / (1.0 / qy - 2.0 / float(ry))
                ty -= 2.0 * (
                    cy - float(element.get("center_y_m", 0.0))
                ) / float(ry)
            retained *= _element_response(element, "power_reflectivity", wavelength)
            if bool(element.get("flip_x", False)):
                cx = -cx
                tx = -tx
                source_rotation_deg = -source_rotation_deg
            if bool(element.get("flip_y", False)):
                cy = -cy
                ty = -ty
                source_rotation_deg = -source_rotation_deg
        elif kind in {"aperture", "phase_screen"}:
            warnings.append(f"ABCD mode ignores {kind} field reshaping; only configured scalar transmission is applied.")
            retained *= _element_response(element, "power_transmission", wavelength)
        elif kind == "loss":
            retained *= _element_response(element, "power_transmission", wavelength)
        else:
            raise ValueError(f"Unsupported ABCD optical element type {kind!r}")
        raw_dispersion = element.get("dispersion")
        dispersion = _element_dispersion(element)
        if raw_dispersion and dispersion is None:
            warnings.append(
                f"{element.get('name', kind)}: ignored malformed optional dispersion data; "
                "reload the JSON to restore its coefficient mapping."
            )
        if dispersion:
            frequency = np.asarray([LIGHT_SPEED / wavelength])
            global_phase += float(phase_from_coefficients(frequency, dispersion)[0])
    wx, rx = _q_radius_curvature(qx, wavelength, n, m2x)
    wy, ry = _q_radius_curvature(qy, wavelength, n, m2y)
    xx, yy = grid.transverse_mesh
    modes = beam_cfg.get("spatial_modes", {})
    signed_mode: np.ndarray | None = None
    if modes.get("model") == "single_hg_mode":
        signed_mode = single_hg_mode_amplitude(
            xx,
            yy,
            wx,
            wy,
            int(modes.get("active_order_x", 0)),
            int(modes.get("active_order_y", 0)),
            center_x_m=cx,
            center_y_m=cy,
            rotation_deg=source_rotation_deg,
        )
        density = signed_mode**2
    elif modes.get("model") == "hermite_gaussian":
        density = effective_hg_density(
            xx,
            yy,
            wx,
            wy,
            m2x,
            m2y,
            int(modes.get("maximum_order_x", 0)),
            int(modes.get("maximum_order_y", 0)),
            center_x_m=cx,
            center_y_m=cy,
            rotation_deg=source_rotation_deg,
        )
        warnings.append(
            "ABCD propagated the truncated incoherent HG ensemble with a moment-matched scalar field; "
            "use FFT for apertures and detailed field reshaping."
        )
    else:
        rotation = math.radians(source_rotation_deg)
        local_x = (xx - cx) * math.cos(rotation) + (yy - cy) * math.sin(rotation)
        local_y = -(xx - cx) * math.sin(rotation) + (yy - cy) * math.cos(rotation)
        density = 2.0 / (math.pi * wx * wy) * np.exp(
            -2.0 * ((local_x / wx) ** 2 + (local_y / wy) ** 2)
        )
    raw_sampled_fraction = float(grid.integrate_transverse(density))
    sampled_fraction = raw_sampled_fraction
    if sampled_fraction > 1.0:
        # A continuous Gaussian is normalized on the infinite plane, but a
        # coarse midpoint grid can over-integrate a narrow spot. Never allow
        # that quadrature error to create optical energy.
        density /= sampled_fraction
        if sampled_fraction > 1.02:
            warnings.append(
                f"ABCD beam is under-resolved: raw grid quadrature was {sampled_fraction:.3%} of retained energy; "
                "it was conservatively capped. Increase Nx/Ny for an accurate spatial profile."
            )
    field_value = np.sqrt(max(retained, 0.0) * density).astype(complex)
    if signed_mode is not None:
        field_value *= np.where(signed_mode < 0.0, -1.0, 1.0)
    phase = np.zeros_like(xx)
    # q uses z_R=pi*w0^2/(M2*lambda), while the following FFT propagation
    # likewise uses M2*lambda.  Encode the boundary curvature consistently.
    rotation = math.radians(source_rotation_deg)
    phase_x = (xx - cx) * math.cos(rotation) + (yy - cy) * math.sin(rotation)
    phase_y = -(xx - cx) * math.sin(rotation) + (yy - cy) * math.cos(rotation)
    if math.isfinite(rx):
        phase += phase_x**2 / (rx * m2x)
    if math.isfinite(ry):
        phase += phase_y**2 / (ry * m2y)
    field_value *= np.exp(1j * math.pi * n / wavelength * phase)
    if include_tilt_phase and (tx != 0.0 or ty != 0.0):
        wave_number = 2.0 * math.pi * n / wavelength
        field_value *= np.exp(
            1j
            * wave_number
            * (
                math.sin(tx) * (xx - cx) / m2x
                + math.sin(ty) * (yy - cy) / m2y
            )
        )
    field_value *= np.exp(1j * global_phase)
    represented_fraction = min(max(raw_sampled_fraction, 0.0), 1.0)
    if diagnostics is not None:
        diagnostics.update(
            {
                "analytic_quantity": float(retained),
                "sampled_quantity": float(retained * represented_fraction),
                "sampled_fraction": float(represented_fraction),
                "raw_quadrature_fraction": float(raw_sampled_fraction),
                "radius_x_m": float(wx),
                "radius_y_m": float(wy),
                "curvature_x_m": float(rx),
                "curvature_y_m": float(ry),
                "center_x_m": float(cx),
                "center_y_m": float(cy),
                "angle_x_rad": float(tx),
                "angle_y_rad": float(ty),
                "rotation_deg": float(source_rotation_deg),
                "refractive_index": float(n),
                "tilt_phase_included": bool(include_tilt_phase),
            }
        )
    trace = OpticalTrace()
    if record_plane:
        trace.add("abcd_gain_entrance", field_value, grid, quantity_name)
    return field_value, n, trace, warnings


def super_gaussian_abcd_to_plane(
    beam_cfg: dict[str, Any],
    total_quantity: float,
    elements: list[dict[str, Any]],
    grid: Grid3D,
    quantity_name: str,
    interface_model: str = "ideal",
    record_plane: bool = True,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[np.ndarray, float, OpticalTrace, list[str]]:
    """Propagate a self-similar super-Gaussian intensity envelope by ABCD.

    The transverse intensity is constrained to the configured normalized
    super-Gaussian family at every plane.  Its 1/e^2 radii, centroid, chief-ray
    angle and quadratic wavefront are transported by the same astigmatic q/ray
    matrices as :func:`gaussian_abcd_to_plane`; the configured super-Gaussian
    exponents are held fixed.  In homogeneous free space this reduces to

    ``w_i(z)^2 = w_0_i^2 + (z - f_i)^2 tan(alpha_i)^2``

    when ``M2_i = pi*n*w_0_i*tan(alpha_i)/lambda0``.  This is the analytic
    self-similar engineering envelope commonly used for pump-beam transport,
    extended through first-order optical elements. For exponent greater than two it is a
    reduced-order intensity closure, not an exact shape-preserving solution of
    the paraxial wave equation.  Aperture reshaping and arbitrary phase screens
    therefore require FFT propagation instead.
    """

    profile = str(beam_cfg.get("profile", "super_gaussian"))
    if profile not in {"gaussian", "super_gaussian", "radial_super_gaussian"}:
        raise ValueError(
            "super_gaussian_abcd requires pump.profile=gaussian, "
            "super_gaussian, or radial_super_gaussian"
        )

    envelope_diagnostics: dict[str, Any] = {}
    gaussian_field, refractive_index, _, warnings = gaussian_abcd_to_plane(
        beam_cfg,
        total_quantity,
        elements,
        grid,
        quantity_name,
        interface_model,
        False,
        include_tilt_phase=True,
        diagnostics=envelope_diagnostics,
        warn_non_gaussian_approximation=False,
    )
    # The Gaussian helper's sampling warning concerns its temporary Gaussian
    # carrier, not the super-Gaussian intensity actually returned here.
    warnings = [
        warning
        for warning in warnings
        if not warning.startswith("ABCD beam is under-resolved:")
    ]

    local_cfg = dict(beam_cfg)
    local_cfg.update(
        {
            "waist_x_m": float(envelope_diagnostics["radius_x_m"]),
            "waist_y_m": float(envelope_diagnostics["radius_y_m"]),
            "focus_z_m": 0.0,
            "focus_x_m": 0.0,
            "focus_y_m": 0.0,
            "center_x_m": float(envelope_diagnostics["center_x_m"]),
            "center_y_m": float(envelope_diagnostics["center_y_m"]),
            "rotation_deg": float(envelope_diagnostics["rotation_deg"]),
            "incidence_angle_x_deg": 0.0,
            "incidence_angle_y_deg": 0.0,
            "m2_x": 1.0,
            "m2_y": 1.0,
            "source_refractive_index": float(refractive_index),
            "propagation_refractive_index": float(refractive_index),
        }
    )
    xx, yy = grid.transverse_mesh
    density = make_beam(local_cfg).density(xx, yy, 0.0)
    raw_sampled_fraction = float(grid.integrate_transverse(density))
    if raw_sampled_fraction > 1.0:
        density /= raw_sampled_fraction
        if raw_sampled_fraction > 1.02:
            warnings.append(
                "Super-Gaussian ABCD envelope is under-resolved: raw grid "
                f"quadrature was {raw_sampled_fraction:.3%} of retained energy; "
                "it was conservatively capped. Increase Nx/Ny for an accurate "
                "spatial profile."
            )

    analytic_quantity = max(
        float(envelope_diagnostics.get("analytic_quantity", total_quantity)),
        0.0,
    )
    amplitude = np.sqrt(analytic_quantity * np.maximum(density, 0.0))
    carrier_amplitude = np.abs(gaussian_field)
    carrier_phase = np.ones_like(gaussian_field, dtype=complex)
    nonzero = carrier_amplitude > np.finfo(float).tiny
    carrier_phase[nonzero] = gaussian_field[nonzero] / carrier_amplitude[nonzero]
    field_value = amplitude.astype(complex) * carrier_phase

    represented_fraction = min(max(raw_sampled_fraction, 0.0), 1.0)
    sg_x = 2.0 * float(beam_cfg.get("order_x", 1.0))
    sg_y = 2.0 * float(beam_cfg.get("order_y", 1.0))
    envelope_diagnostics.update(
        {
            "sampled_quantity": analytic_quantity * represented_fraction,
            "sampled_fraction": represented_fraction,
            "raw_quadrature_fraction": raw_sampled_fraction,
            "transport_representation": "self_similar_super_gaussian_intensity_envelope",
            "envelope_propagation": "astigmatic_first_order_abcd_q",
            "profile_evolution": "fixed_super_gaussian_exponent",
            "super_gaussian_exponent_x": sg_x,
            "super_gaussian_exponent_y": sg_y,
            "radial_equal_axis_convention": bool(
                profile == "radial_super_gaussian"
                and math.isclose(sg_x, sg_y, rel_tol=0.0, abs_tol=1e-14)
            ),
            "coherence_contract": "intensity_envelope_with_quadratic_phase_carrier",
            "exact_paraxial_wave_solution": bool(
                math.isclose(sg_x, 2.0, rel_tol=0.0, abs_tol=1e-14)
                and math.isclose(sg_y, 2.0, rel_tol=0.0, abs_tol=1e-14)
            ),
        }
    )
    if diagnostics is not None:
        diagnostics.update(envelope_diagnostics)

    trace = OpticalTrace()
    if record_plane:
        trace.add("super_gaussian_abcd_gain_entrance", field_value, grid, quantity_name)
    return field_value, refractive_index, trace, warnings


def source_field(beam_cfg: dict[str, Any], total_quantity: float, grid: Grid3D) -> np.ndarray:
    xx, yy = grid.transverse_mesh
    density = make_beam(beam_cfg).density(xx, yy, 0.0)
    sampled_fraction = float(grid.integrate_transverse(density))
    if sampled_fraction > 1.0:
        # Preserve finite-window loss when the beam is clipped, but cap the
        # opposite (non-physical) error caused by under-resolved midpoint
        # quadrature. This guarantees that a sampled source never contains
        # more energy/power than the configured source.
        density = density / sampled_fraction
    field_value = np.sqrt(np.maximum(total_quantity * density, 0.0)).astype(complex)
    spatial_modes = beam_cfg.get("spatial_modes", {})
    if spatial_modes.get("model") == "single_hg_mode":
        beam = make_beam(beam_cfg)
        wavelength_for_mode = float(beam_cfg["wavelength_m"])
        radius_x = beam._radius(  # type: ignore[attr-defined]
            float(beam_cfg["waist_x_m"]), 1.0, wavelength_for_mode, 0.0, "x"
        )
        radius_y = beam._radius(  # type: ignore[attr-defined]
            float(beam_cfg["waist_y_m"]), 1.0, wavelength_for_mode, 0.0, "y"
        )
        angle_x_mode = math.radians(
            float(beam_cfg.get("incidence_angle_x_deg", 0.0))
        )
        angle_y_mode = math.radians(
            float(beam_cfg.get("incidence_angle_y_deg", 0.0))
        )
        center_x_mode = float(beam_cfg.get("center_x_m", 0.0)) - axis_focus_m(
            beam_cfg, "x"
        ) * math.tan(angle_x_mode)
        center_y_mode = float(beam_cfg.get("center_y_m", 0.0)) - axis_focus_m(
            beam_cfg, "y"
        ) * math.tan(angle_y_mode)
        signed_mode = single_hg_mode_amplitude(
            xx,
            yy,
            radius_x,
            radius_y,
            int(spatial_modes.get("active_order_x", 0)),
            int(spatial_modes.get("active_order_y", 0)),
            center_x_m=center_x_mode,
            center_y_m=center_y_mode,
            rotation_deg=float(beam_cfg.get("rotation_deg", 0.0)),
        )
        field_value *= np.where(signed_mode < 0.0, -1.0, 1.0)
    radius = beam_cfg.get("wavefront_radius_m")
    wavelength = float(beam_cfg["wavelength_m"])
    source_n = float(beam_cfg.get("source_refractive_index", 1.0))
    wave_number = 2.0 * math.pi * source_n / wavelength
    phase = np.zeros_like(xx)
    m2_x = float(beam_cfg.get("m2_x", 1.0))
    m2_y = float(beam_cfg.get("m2_y", 1.0))
    # This scalar M2 surrogate diffracts with lambda_eff=M2*lambda.  Ideal
    # source curvature and carrier tilt must therefore use the same effective
    # wave number, just like the lens/mirror operators below.  User/thermal
    # phase screens remain physical radians and are explicitly warned about.
    angle_x = math.radians(float(beam_cfg.get("incidence_angle_x_deg", 0.0)))
    angle_y = math.radians(float(beam_cfg.get("incidence_angle_y_deg", 0.0)))
    source_center_x = float(beam_cfg.get("center_x_m", 0.0)) - axis_focus_m(
        beam_cfg, "x"
    ) * math.tan(angle_x)
    source_center_y = float(beam_cfg.get("center_y_m", 0.0)) - axis_focus_m(
        beam_cfg, "y"
    ) * math.tan(angle_y)
    # Curvature belongs to the same principal axes as the rotated intensity
    # envelope. Keep the linear carrier tilt in the laboratory frame below.
    rotation = math.radians(float(beam_cfg.get("rotation_deg", 0.0)))
    phase_x = (xx - source_center_x) * math.cos(rotation) + (yy - source_center_y) * math.sin(rotation)
    phase_y = -(xx - source_center_x) * math.sin(rotation) + (yy - source_center_y) * math.cos(rotation)
    if radius not in (None, 0):
        phase += phase_x**2 / (float(radius) * m2_x)
        phase += phase_y**2 / (float(radius) * m2_y)
    elif axis_focus_m(beam_cfg, "x") != 0.0 or axis_focus_m(beam_cfg, "y") != 0.0:
        for coordinate, waist_key, m2_key, axis in (
            (phase_x, "waist_x_m", "m2_x", "x"),
            (phase_y, "waist_y_m", "m2_y", "y"),
        ):
            distance = -axis_focus_m(beam_cfg, axis)
            if math.isclose(distance, 0.0, rel_tol=0.0, abs_tol=np.finfo(float).tiny):
                # At a waist the wavefront radius is infinite, hence this
                # axis contributes no quadratic phase.  The other axis may
                # still be away from its waist in a valid astigmatic source.
                continue
            waist = float(beam_cfg[waist_key])
            rayleigh = (
                math.pi
                * source_n
                * waist**2
                / (float(beam_cfg.get(m2_key, 1.0)) * wavelength)
            )
            curvature = distance * (1.0 + (rayleigh / distance) ** 2)
            phase += (
                coordinate**2
                / (curvature * float(beam_cfg.get(m2_key, 1.0)))
            )
    if np.any(phase):
        field_value *= np.exp(1j * wave_number * phase / 2.0)
    if angle_x or angle_y:
        direction_x, direction_y, _ = axis_direction_components(
            angle_x,
            angle_y,
        )
        field_value *= np.exp(
            1j
            * wave_number
            * (
                direction_x * xx / m2_x
                + direction_y * yy / m2_y
            )
        )
    return field_value


def _surface_radius(element: dict[str, Any], surface: str, axis: str) -> float | None:
    value = element.get(f"radius_{surface}_{axis}_m", element.get(f"radius_{surface}_m"))
    return None if value in (None, 0) else float(value)


def _curved_interface_phase(
    element: dict[str, Any],
    xx: np.ndarray,
    yy: np.ndarray,
    wavelength_m: float,
    n_before: float,
    n_after: float,
    surface: str,
    m2_x: float,
    m2_y: float,
) -> np.ndarray:
    cx = float(element.get("center_x_m", 0.0))
    cy = float(element.get("center_y_m", 0.0))
    optical_path = np.zeros_like(xx)
    radius_x = _surface_radius(element, surface, "x")
    radius_y = _surface_radius(element, surface, "y")
    if radius_x is not None:
        optical_path += (
            (n_after - n_before)
            * (xx - cx) ** 2
            / (2.0 * radius_x * m2_x)
        )
    if radius_y is not None:
        optical_path += (
            (n_after - n_before)
            * (yy - cy) ** 2
            / (2.0 * radius_y * m2_y)
        )
    return np.exp(-1j * 2.0 * math.pi / wavelength_m * optical_path)


def _refract_q_ray(
    q: complex,
    center: float,
    angle: float,
    n_before: float,
    n_after: float,
    radius_m: float | None,
    *,
    surface_center_m: float = 0.0,
) -> tuple[complex, float, float]:
    curvature = 0.0 if radius_m is None else (n_before - n_after) / (radius_m * n_after)
    d = n_before / n_after
    q_after = q / (curvature * q + d)
    angle_after = curvature * (center - surface_center_m) + d * angle
    return q_after, center, angle_after


def _q_radius_curvature(q: complex, wavelength: float, refractive_index: float, m2: float) -> tuple[float, float]:
    inverse = 1.0 / q
    radius = math.sqrt(max(-m2 * wavelength / (math.pi * refractive_index * inverse.imag), np.finfo(float).tiny))
    curvature = math.inf if abs(inverse.real) < 1e-15 else 1.0 / inverse.real
    return radius, curvature


def _optional_focal_length(element: dict[str, Any], key: str) -> float | None:
    value = element.get(key, element.get("focal_length_m"))
    if value in (None, 0):
        return None
    return float(value)


def _apply_optional_aperture(field_value: np.ndarray, element: dict[str, Any], xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    if not any(
        element.get(key) is not None
        for key in (
            "aperture_radius_m",
            "aperture_radius_x_m",
            "aperture_radius_y_m",
            "aperture_width_m",
            "aperture_height_m",
        )
    ):
        return field_value
    aperture = dict(element)
    aperture["type"] = "aperture"
    return field_value * _aperture_mask(aperture, xx, yy)


def _aperture_mask(element: dict[str, Any], xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    cx = float(element.get("center_x_m") or 0.0)
    cy = float(element.get("center_y_m") or 0.0)
    raw_shape = element.get("shape")
    shape = "ellipse" if raw_shape is None else str(raw_shape)
    if shape in {"circle", "ellipse", "super_gaussian"}:
        # The click editor historically stored standalone aperture radii as
        # radius_x_m/radius_y_m, while JSON and lens clear apertures use the
        # explicit aperture_radius_* names.  Accept both only here, after the
        # element is already known to be an aperture, so lens curvatures are
        # never mistaken for stops.
        common = element.get("aperture_radius_m")
        radius_x = element.get("aperture_radius_x_m")
        radius_y = element.get("aperture_radius_y_m")
        if radius_x is None:
            radius_x = common
        if radius_y is None:
            radius_y = common
        if radius_x is None:
            radius_x = element.get("radius_x_m", math.inf)
        if radius_y is None:
            radius_y = element.get("radius_y_m", math.inf)
        rx = float(radius_x)
        ry = float(radius_y)
        coordinate = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        if shape == "super_gaussian":
            order = float(element.get("order") or 10.0)
            return np.exp(-0.5 * coordinate**order)
        return (coordinate <= 1.0).astype(float)
    if shape == "rectangle":
        width = float(element["aperture_width_m"])
        height = float(element.get("aperture_height_m") or width)
        return ((np.abs(xx - cx) <= width / 2.0) & (np.abs(yy - cy) <= height / 2.0)).astype(float)
    raise ValueError(f"Unknown aperture shape {shape!r}")
