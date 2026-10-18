"""Fast Gaussian-moment optical-bench preview.

The full solvers propagate sampled fields through the gain medium.  This module
is intentionally lighter: it traces paraxial q parameters and centroids so a UI
can update a D4-sigma envelope on every parameter edit without running rate
equations, thermal iterations, or a 3D FFT volume.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .dispersion import material_refractive_index
from .optical_system import (
    gain_medium_axis_angles_rad,
    gain_medium_path_cosine,
    refract_axis_angles_rad,
)
from .pass_routing import resolve_pass_reverse, resolve_pump_direction
from .multipass import compile_multipass_plan
from .spatial_modes import axis_focus_m

from .optics import split_gain_path


@dataclass(frozen=True)
class PreviewElement:
    name: str
    kind: str
    start_m: float
    end_m: float
    pass_index: int | None = None
    direction: str | None = None
    material_enabled: bool = True
    center_x_m: float = 0.0
    center_y_m: float = 0.0


@dataclass(frozen=True)
class BenchPreview:
    beam_kind: str
    optical_path_m: np.ndarray
    radius_x_D4sigma_m: np.ndarray
    radius_y_D4sigma_m: np.ndarray
    centroid_x_m: np.ndarray
    centroid_y_m: np.ndarray
    elements: tuple[PreviewElement, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class BeamPathSample:
    """One interpolated beam cross section on a branch's true path coordinate."""

    beam_kind: str
    local_path_m: float
    gain_relative_path_m: float | None
    radius_x_D4sigma_m: float
    radius_y_D4sigma_m: float
    centroid_x_m: float
    centroid_y_m: float

    @property
    def x_bounds_D4sigma_m(self) -> tuple[float, float]:
        return (
            self.centroid_x_m - self.radius_x_D4sigma_m,
            self.centroid_x_m + self.radius_x_D4sigma_m,
        )

    @property
    def y_bounds_D4sigma_m(self) -> tuple[float, float]:
        return (
            self.centroid_y_m - self.radius_y_D4sigma_m,
            self.centroid_y_m + self.radius_y_D4sigma_m,
        )


@dataclass
class _AxisState:
    q_m: complex
    center_m: float
    angle_rad: float
    m2: float
    d4_scale: float


def build_bench_preview(
    config: dict[str, Any],
    beam_kind: str = "seed",
    samples_per_section: int = 24,
    *,
    include_gain_medium: bool = True,
) -> BenchPreview:
    """Build a fast D4-sigma radius/centroid trace for seed or pump optics.

    ``include_gain_medium=False`` is a non-destructive alignment mode.  Every
    gain marker keeps its physical position and traversal length, but that
    reserved segment is propagated in the surrounding medium without crystal
    refraction.  The simulation configuration itself is never modified.
    """

    if beam_kind not in {"seed", "pump"}:
        raise ValueError("beam_kind must be 'seed' or 'pump'")
    beam = config[beam_kind]
    wavelength = float(beam["wavelength_m"])
    source_n = float(beam.get("source_refractive_index", 1.0))
    focus_x = axis_focus_m(beam, "x")
    focus_y = axis_focus_m(beam, "y")
    angle_x = math.radians(float(beam.get("incidence_angle_x_deg", 0.0)))
    angle_y = math.radians(float(beam.get("incidence_angle_y_deg", 0.0)))
    order_x, order_y = _profile_orders(beam)
    d4_x, d4_y = _profile_d4_scales(beam, order_x, order_y)
    state_x = _AxisState(
        q_m=complex(
            -focus_x,
            math.pi * source_n * float(beam["waist_x_m"]) ** 2
            / (max(float(beam.get("m2_x", 1.0)), np.finfo(float).tiny) * wavelength),
        ),
        center_m=float(beam.get("center_x_m", 0.0)) - focus_x * math.tan(angle_x),
        angle_rad=angle_x,
        m2=float(beam.get("m2_x", 1.0)),
        d4_scale=d4_x,
    )
    state_y = _AxisState(
        q_m=complex(
            -focus_y,
            math.pi * source_n * float(beam["waist_y_m"]) ** 2
            / (max(float(beam.get("m2_y", 1.0)), np.finfo(float).tiny) * wavelength),
        ),
        center_m=float(beam.get("center_y_m", 0.0)) - focus_y * math.tan(angle_y),
        angle_rad=angle_y,
        m2=float(beam.get("m2_y", 1.0)),
        d4_scale=d4_y,
    )
    path = _preview_path(config, beam_kind)
    position = 0.0
    current_n = source_n
    positions: list[float] = []
    radius_x: list[float] = []
    radius_y: list[float] = []
    centroid_x: list[float] = []
    centroid_y: list[float] = []
    markers: list[PreviewElement] = []
    warnings: list[str] = []
    profile = str(beam.get("profile", "super_gaussian"))
    if profile not in {"gaussian", "super_gaussian"} or order_x != 1.0 or order_y != 1.0:
        warnings.append(
            "Realtime preview uses a second-moment-equivalent Gaussian q trace for the configured non-Gaussian profile; the full FFT result remains authoritative."
        )
    if float(beam.get("rotation_deg", 0.0)) != 0.0:
        warnings.append(
            "Realtime q preview does not rotate the full second-moment tensor; use saved FFT optical planes for a rotated astigmatic beam."
        )
    if beam_kind == "seed" and bool(
        config["simulation"].get("multipass", {}).get("enabled", False)
    ):
        warnings.append(
            "The horizontal coordinate is cumulative optical-path distance for the compiled multipass route, not a mechanical CAD x coordinate."
        )

    def record(at_m: float) -> None:
        positions.append(float(at_m))
        radius_x.append(_axis_radius(state_x, wavelength, current_n))
        radius_y.append(_axis_radius(state_y, wavelength, current_n))
        centroid_x.append(state_x.center_m)
        centroid_y.append(state_y.center_m)

    def propagate_space(
        length_m: float,
        refractive_index: float,
        *,
        centroid_axial_length_m: float | None = None,
        centroid_slope_x: float | None = None,
        centroid_slope_y: float | None = None,
    ) -> None:
        nonlocal position
        if length_m < 0.0:
            raise ValueError("Optical preview distances must be non-negative")
        if length_m == 0.0:
            record(position)
            return
        count = max(2, int(samples_per_section))
        qx0, qy0 = state_x.q_m, state_y.q_m
        cx0, cy0 = state_x.center_m, state_y.center_m
        centroid_length = (
            length_m
            if centroid_axial_length_m is None
            else float(centroid_axial_length_m)
        )
        slope_x = (
            math.tan(state_x.angle_rad)
            if centroid_slope_x is None
            else float(centroid_slope_x)
        )
        slope_y = (
            math.tan(state_y.angle_rad)
            if centroid_slope_y is None
            else float(centroid_slope_y)
        )
        for fraction in np.linspace(0.0, 1.0, count, endpoint=True)[1:]:
            distance = float(fraction * length_m)
            state_x.q_m = qx0 + distance
            state_y.q_m = qy0 + distance
            state_x.center_m = cx0 + fraction * centroid_length * slope_x
            state_y.center_m = cy0 + fraction * centroid_length * slope_y
            position_at_sample = position + distance
            positions.append(position_at_sample)
            radius_x.append(_axis_radius(state_x, wavelength, refractive_index))
            radius_y.append(_axis_radius(state_y, wavelength, refractive_index))
            centroid_x.append(state_x.center_m)
            centroid_y.append(state_y.center_m)
        position += length_m

    record(position)
    gain_count = 0
    for raw_index, raw in enumerate(path):
        element = copy.deepcopy(raw)
        raw_distance_before = element.pop("distance_before_m", 0.0)
        distance_before = float(
            0.0 if raw_distance_before is None else raw_distance_before
        )
        if distance_before:
            propagate_space(distance_before, current_n)
        kind = str(element.get("type", ""))
        name = str(element.get("name", f"{raw_index + 1}_{kind}"))
        start = position
        previous_marker_count = len(markers)
        if kind == "free_space":
            propagate_space(float(element.get("length_m", 0.0)), current_n)
        elif kind == "gain_medium":
            gain_count += 1
            incidence_x, incidence_y = gain_medium_axis_angles_rad(element)
            path_cosine = gain_medium_path_cosine(element)
            length = float(config["grid"]["length_z_m"]) / path_cosine
            if include_gain_medium:
                gain_n = material_refractive_index(config["material"], wavelength, role=beam_kind)
                _refract_plane_pair(state_x, state_y, current_n, gain_n)
                previous_n = current_n
                current_n = gain_n
                record(position)
                # The sampled-field solver treats a gain marker angle as an
                # additional in-crystal walk-off, not as a carrier tilt that
                # survives the exit face.  Source/external carrier tilt and
                # marker walk-off therefore add as transverse slopes over the
                # physical z thickness, while q diffraction advances over the
                # longer oblique path length.
                propagate_space(
                    length,
                    current_n,
                    centroid_axial_length_m=float(config["grid"]["length_z_m"]),
                    centroid_slope_x=(
                        math.tan(state_x.angle_rad) + math.tan(incidence_x)
                    ),
                    centroid_slope_y=(
                        math.tan(state_y.angle_rad) + math.tan(incidence_y)
                    ),
                )
                outside_n = float(element.get("outside_refractive_index", previous_n))
                _refract_plane_pair(state_x, state_y, current_n, outside_n)
                current_n = outside_n
                record(position)
            else:
                # Keep the same mechanical gap and all downstream element
                # coordinates while tracing the empty holder in its current
                # surrounding medium (normally air).
                record(position)
                propagate_space(length, current_n)
                record(position)
            markers.append(
                PreviewElement(
                    name=name,
                    kind=kind,
                    start_m=start,
                    end_m=position,
                    pass_index=gain_count,
                    direction=(
                        resolve_pump_direction(
                            str(beam.get("direction", "forward")),
                            str(element.get("direction", "auto")),
                        )
                        if beam_kind == "pump"
                        else str(element.get("direction", beam.get("direction", "forward")))
                    ),
                    material_enabled=bool(include_gain_medium),
                )
            )
        elif kind == "thin_lens":
            _apply_thin_lens(state_x, element, "x")
            _apply_thin_lens(state_y, element, "y")
            markers.append(PreviewElement(name, kind, position, position))
            record(position)
        elif kind == "double_pass_lens_relay":
            object_distance = float(element["object_to_first_lens_m"])
            separation = float(element["lens_separation_m"])
            mirror_distance = float(element["second_lens_to_mirror_m"])
            first_lens = {
                "focal_length_m": float(element["first_focal_length_m"])
            }
            second_lens = {
                "focal_length_m": float(element["second_focal_length_m"])
            }
            propagate_space(object_distance, current_n)
            _apply_thin_lens(state_x, first_lens, "x")
            _apply_thin_lens(state_y, first_lens, "y")
            record(position)
            propagate_space(separation, current_n)
            _apply_thin_lens(state_x, second_lens, "x")
            _apply_thin_lens(state_y, second_lens, "y")
            record(position)
            propagate_space(2.0 * mirror_distance, current_n)
            _apply_thin_lens(state_x, second_lens, "x")
            _apply_thin_lens(state_y, second_lens, "y")
            record(position)
            propagate_space(separation, current_n)
            _apply_thin_lens(state_x, first_lens, "x")
            _apply_thin_lens(state_y, first_lens, "y")
            record(position)
            propagate_space(object_distance, current_n)
            markers.append(PreviewElement(name, kind, start, position))
        elif kind == "thick_lens":
            lens_n = float(element["refractive_index"])
            thickness = float(element.get("thickness_m", 0.0))
            n_after = float(element.get("n_after", current_n))
            _refract_curved(
                state_x,
                current_n,
                lens_n,
                _surface_radius(element, "front", "x"),
                float(element.get("center_x_m", 0.0)),
            )
            _refract_curved(
                state_y,
                current_n,
                lens_n,
                _surface_radius(element, "front", "y"),
                float(element.get("center_y_m", 0.0)),
            )
            current_n = lens_n
            record(position)
            propagate_space(thickness, current_n)
            _refract_curved(
                state_x,
                current_n,
                n_after,
                _surface_radius(element, "back", "x"),
                float(element.get("center_x_m", 0.0)),
            )
            _refract_curved(
                state_y,
                current_n,
                n_after,
                _surface_radius(element, "back", "y"),
                float(element.get("center_y_m", 0.0)),
            )
            current_n = n_after
            markers.append(PreviewElement(name, kind, start, position))
            record(position)
        elif kind == "interface":
            n_after = float(element["n_after"])
            state_x.angle_rad += math.radians(
                float(element.get("axis_angle_x_deg", 0.0))
            )
            state_y.angle_rad += math.radians(
                float(element.get("axis_angle_y_deg", 0.0))
            )
            _refract_plane_pair(state_x, state_y, current_n, n_after)
            current_n = n_after
            markers.append(PreviewElement(name, kind, position, position))
            record(position)
        elif kind == "mirror":
            _apply_mirror(state_x, element, "x")
            _apply_mirror(state_y, element, "y")
            markers.append(PreviewElement(name, kind, position, position))
            record(position)
        elif kind in {"aperture", "phase_screen"}:
            warnings.append(
                f"{name}: realtime q preview does not include {kind} reshaping; inspect the full FFT optical-plane result."
            )
            markers.append(PreviewElement(name, kind, position, position))
        elif kind in {"tilt", "coordinate_break"}:
            state_x.center_m += float(element.get("shift_x_m", 0.0))
            state_y.center_m += float(element.get("shift_y_m", 0.0))
            state_x.angle_rad += math.radians(float(element.get("angle_x_deg", 0.0)))
            state_y.angle_rad += math.radians(float(element.get("angle_y_deg", 0.0)))
            markers.append(PreviewElement(name, kind, position, position))
            record(position)
        elif kind in {"loss", "observation", "plane", "dispersive"}:
            markers.append(PreviewElement(name, kind, position, position))
        else:
            warnings.append(f"{name}: unsupported realtime-preview element {kind!r}; it was left unchanged.")
        if len(markers) > previous_marker_count:
            markers[-1] = replace(
                markers[-1],
                center_x_m=float(element.get("center_x_m") or 0.0),
                center_y_m=float(element.get("center_y_m") or 0.0),
            )
        turn_x = float(element.get("turn_angle_x_deg", 0.0))
        turn_y = float(element.get("turn_angle_y_deg", 0.0))
        if turn_x or turn_y:
            state_x.angle_rad += math.radians(turn_x)
            state_y.angle_rad += math.radians(turn_y)
            record(position)

    # Crystal clipping must only be checked where the beam is inside a gain
    # segment.  A deliberately broad source beam before a focusing lens is not
    # constrained by the crystal aperture and must not trigger a false warning.
    sampled_positions = np.asarray(positions, dtype=float)
    gain_samples = np.zeros(len(sampled_positions), dtype=bool)
    tolerance = 1e-12 * max(
        float(np.max(np.abs(sampled_positions))) if len(sampled_positions) else 0.0,
        float(np.ptp(sampled_positions)) if len(sampled_positions) else 0.0,
        np.finfo(float).tiny,
    )
    for marker in markers:
        if marker.kind == "gain_medium" and marker.material_enabled:
            gain_samples |= (sampled_positions >= marker.start_m - tolerance) & (
                sampled_positions <= marker.end_m + tolerance
            )
    sampled_radius_x = np.asarray(radius_x, dtype=float)[gain_samples]
    sampled_radius_y = np.asarray(radius_y, dtype=float)[gain_samples]
    sampled_centroid_x = np.asarray(centroid_x, dtype=float)[gain_samples]
    sampled_centroid_y = np.asarray(centroid_y, dtype=float)[gain_samples]
    geometry = config["material"].get("geometry", {})
    crystal_center_x = float(geometry.get("center_x_m", 0.0))
    crystal_center_y = float(geometry.get("center_y_m", 0.0))
    relative_centroid_x = sampled_centroid_x - crystal_center_x
    relative_centroid_y = sampled_centroid_y - crystal_center_y
    if geometry.get("shape", "cuboid") == "cylinder":
        diameter_m = geometry.get("diameter_m")
        if diameter_m is None:
            diameter_m = min(
                config["grid"]["size_x_m"], config["grid"]["size_y_m"]
            )
        radius = 0.5 * float(diameter_m)
        crystal_half_x = crystal_half_y = radius
    else:
        width_x_m = geometry.get("width_x_m")
        if width_x_m is None:
            width_x_m = config["grid"]["size_x_m"]
        height_y_m = geometry.get("height_y_m")
        if height_y_m is None:
            height_y_m = config["grid"]["size_y_m"]
        crystal_half_x = 0.5 * float(width_x_m)
        crystal_half_y = 0.5 * float(height_y_m)
    if include_gain_medium and np.any(gain_samples):
        if geometry.get("shape", "cuboid") == "cylinder":
            exceeds_cylinder = any(
                _axis_aligned_ellipse_maximum_radius_squared(cx, cy, rx, ry)
                > radius * radius
                for cx, cy, rx, ry in zip(
                    relative_centroid_x,
                    relative_centroid_y,
                    sampled_radius_x,
                    sampled_radius_y,
                )
            )
            if exceeds_cylinder:
                warnings.append(
                    "The offset D4-sigma beam envelope exceeds the physical crystal radius; crystal-aperture clipping is likely."
                )
        else:
            if np.any(np.abs(relative_centroid_x) + sampled_radius_x > crystal_half_x):
                warnings.append(
                    "The offset x-direction D4-sigma envelope exceeds the physical crystal half-width; crystal-aperture clipping is likely."
                )
            if np.any(np.abs(relative_centroid_y) + sampled_radius_y > crystal_half_y):
                warnings.append(
                    "The offset y-direction D4-sigma envelope exceeds the physical crystal half-height; crystal-aperture clipping is likely."
                )
    return BenchPreview(
        beam_kind=beam_kind,
        optical_path_m=np.asarray(positions, dtype=float),
        radius_x_D4sigma_m=np.asarray(radius_x, dtype=float),
        radius_y_D4sigma_m=np.asarray(radius_y, dtype=float),
        centroid_x_m=np.asarray(centroid_x, dtype=float),
        centroid_y_m=np.asarray(centroid_y, dtype=float),
        elements=tuple(markers),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def first_gain_entrance_m(preview: BenchPreview) -> float | None:
    """Return the first gain entrance on this branch's own LASER-based path."""

    gain = next((element for element in preview.elements if element.kind == "gain_medium"), None)
    return None if gain is None else float(gain.start_m)


def sample_bench_preview(
    preview: BenchPreview,
    coordinate_m: float,
    *,
    reference: str = "laser",
) -> BeamPathSample:
    """Interpolate radii and centroid at a true branch coordinate.

    ``reference='laser'`` interprets ``coordinate_m`` from this beam's own
    source plane.  ``reference='gain'`` interprets it from the first shared
    gain entrance, then converts back to the branch-local coordinate before
    interpolation.  This conversion is what keeps unequal Pump/Seed source
    distances from being confused with each other.
    """

    if reference not in {"laser", "gain"}:
        raise ValueError("reference must be 'laser' or 'gain'")
    gain_entrance = first_gain_entrance_m(preview)
    if reference == "gain":
        if gain_entrance is None:
            raise ValueError("gain-relative coordinates need a gain_medium marker")
        local_path = float(coordinate_m) + gain_entrance
    else:
        local_path = float(coordinate_m)

    path = np.asarray(preview.optical_path_m, dtype=float)
    if path.size == 0:
        raise ValueError("the optical preview has no path samples")
    tolerance = 1e-12 * max(
        float(np.max(np.abs(path))),
        float(np.ptp(path)),
        np.finfo(float).tiny,
    )
    if local_path < float(path[0]) - tolerance or local_path > float(path[-1]) + tolerance:
        raise ValueError(
            f"coordinate is outside this {preview.beam_kind} path "
            f"[{path[0]:.8g}, {path[-1]:.8g}] m"
        )
    local_path = float(np.clip(local_path, path[0], path[-1]))

    # Ideal lenses, planar interfaces, and coordinate breaks can create
    # several states at one longitudinal coordinate.  Keep the final state at
    # each duplicate position, matching the downstream side of that element.
    if path.size == 1:
        keep = np.asarray([0], dtype=int)
    else:
        keep = np.concatenate((np.flatnonzero(np.diff(path) > 0.0), [path.size - 1]))
    interpolation_path = path[keep]

    def interpolate(values: np.ndarray) -> float:
        sampled = np.asarray(values, dtype=float)[keep]
        if sampled.size == 1:
            return float(sampled[0])
        return float(np.interp(local_path, interpolation_path, sampled))

    return BeamPathSample(
        beam_kind=preview.beam_kind,
        local_path_m=local_path,
        gain_relative_path_m=None if gain_entrance is None else local_path - gain_entrance,
        radius_x_D4sigma_m=interpolate(preview.radius_x_D4sigma_m),
        radius_y_D4sigma_m=interpolate(preview.radius_y_D4sigma_m),
        centroid_x_m=interpolate(preview.centroid_x_m),
        centroid_y_m=interpolate(preview.centroid_y_m),
    )


def _preview_path(config: dict[str, Any], beam_kind: str) -> list[dict[str, Any]]:
    if beam_kind == "pump":
        return copy.deepcopy(config.get("optical_train", {}).get("pump", []))
    pre, gain, post = split_gain_path(config.get("optical_train", {}).get("seed", []))
    multipass = config["simulation"].get("multipass", {})
    if bool(multipass.get("enabled", False)):
        plan = compile_multipass_plan(
            multipass,
            gain,
            crystal_length_m=float(config["grid"]["length_z_m"]),
            crystal_refractive_index=material_refractive_index(
                config["material"],
                float(config["seed"]["wavelength_m"]),
                role="signal",
            ),
            global_extra_delay_s=float(
                config["simulation"].get(
                    "unmodeled_between_pass_delay_s", 0.0
                )
            ),
        )
        result = copy.deepcopy(pre)
        for index, hit in enumerate(plan.hits):
            result.append(copy.deepcopy(hit.marker))
            if index < len(plan.transitions):
                result.extend(
                    copy.deepcopy(list(plan.transitions[index].route))
                )
        result.extend(copy.deepcopy(post))
        return result
    passes = int(config["simulation"].get("passes", 1))
    between = config.get("optical_train", {}).get("between_pass", [])
    result = copy.deepcopy(pre)
    for pass_index in range(passes):
        marker = copy.deepcopy(gain)
        marker["name"] = str(marker.get("name", "gain_medium")) + f"_pass_{pass_index + 1}"
        backward = resolve_pass_reverse(
            str(config.get("seed", {}).get("direction", "forward")),
            str(marker.get("direction", "auto")),
            pass_index,
            bool(config["simulation"].get("alternate_direction", True)),
            explicit_gain_hit=False,
        )
        marker["direction"] = "backward" if backward else "forward"
        result.append(marker)
        if pass_index + 1 < passes:
            result.extend(copy.deepcopy(between))
    result.extend(copy.deepcopy(post))
    return result


def _profile_orders(beam: dict[str, Any]) -> tuple[float, float]:
    profile = str(beam.get("profile", "super_gaussian"))
    order_x = float(beam.get("order_x", 1.0))
    order_y = float(beam.get("order_y", 1.0))
    if profile == "top_hat":
        order_x = order_x if order_x > 1.0 else 20.0
        order_y = order_y if order_y > 1.0 else 20.0
    if profile == "annular":
        return 1.0, 1.0
    return max(order_x, 0.05), max(order_y, 0.05)


def _profile_d4_scales(beam: dict[str, Any], order_x: float, order_y: float) -> tuple[float, float]:
    """Match the analytic source second moments before the q approximation."""

    if beam.get("spatial_modes", {}).get("model") in {"hermite_gaussian", "incoherent_hg"}:
        return 1.0, 1.0
    profile = beam.get("profile", "super_gaussian")
    if profile == "annular":
        extras = beam.get("extras", {})
        order = float(extras.get("ring_order", 2.0 * float(extras.get("inner_radius_ratio", 0.5))**2))
        scale = math.sqrt(order + 1.0)
        return scale, scale
    if profile == "radial_super_gaussian" and math.isclose(order_x, order_y, rel_tol=0.0, abs_tol=5e-15):
        # <x²>/wx² = 2^(-1/m) Gamma(2/m) / (2 Gamma(1/m)).
        scale = math.sqrt(2.0 * 2.0**(-1.0/order_x) * math.gamma(2.0/order_x) / math.gamma(1.0/order_x))
        return scale, scale
    return _super_gaussian_d4_scale(order_x), _super_gaussian_d4_scale(order_y)


def _super_gaussian_d4_scale(order: float) -> float:
    """Return D4-sigma radius divided by the configured 1/e2 radius."""

    return 2.0 * 2.0 ** (-1.0 / (2.0 * order)) * math.sqrt(
        math.gamma(3.0 / (2.0 * order)) / math.gamma(1.0 / (2.0 * order))
    )


def _axis_aligned_ellipse_maximum_radius_squared(
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
) -> float:
    """Return the exact farthest squared radius of an axis-aligned ellipse."""

    # Stationary points of
    #   (cx + rx cos(theta))**2 + (cy + ry sin(theta))**2
    # become a quartic after t=tan(theta/2).  Cardinal points are included
    # explicitly both for the t=infinity root and for degenerate circles.
    difference = radius_y * radius_y - radius_x * radius_x
    coefficients = np.asarray(
        [
            -center_y * radius_y,
            -2.0 * (center_x * radius_x + difference),
            0.0,
            2.0 * (difference - center_x * radius_x),
            center_y * radius_y,
        ],
        dtype=float,
    )
    nonzero = np.flatnonzero(np.abs(coefficients) > np.finfo(float).eps)
    angles = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi]
    if nonzero.size:
        roots = np.roots(coefficients[int(nonzero[0]) :])
        angles.extend(
            2.0 * math.atan(float(root.real))
            for root in roots
            if abs(float(root.imag)) <= 1e-10 * max(1.0, abs(float(root.real)))
        )
    return max(
        (center_x + radius_x * math.cos(angle)) ** 2
        + (center_y + radius_y * math.sin(angle)) ** 2
        for angle in angles
    )


def _axis_radius(state: _AxisState, wavelength_m: float, refractive_index: float) -> float:
    inverse_q = 1.0 / state.q_m
    denominator = math.pi * refractive_index * inverse_q.imag
    radius_squared = -state.m2 * wavelength_m / denominator
    return state.d4_scale * math.sqrt(max(radius_squared, np.finfo(float).tiny))


def _refract_plane_pair(
    state_x: _AxisState,
    state_y: _AxisState,
    n_before: float,
    n_after: float,
) -> None:
    """Apply one physical plane interface to the coupled two-axis ray."""

    state_x.q_m *= n_after / n_before
    state_y.q_m *= n_after / n_before
    state_x.angle_rad, state_y.angle_rad = refract_axis_angles_rad(
        state_x.angle_rad,
        state_y.angle_rad,
        n_before=n_before,
        n_after=n_after,
    )


def _refract_curved(
    state: _AxisState,
    n_before: float,
    n_after: float,
    radius_m: float | None,
    surface_center_m: float = 0.0,
) -> None:
    curvature = 0.0 if radius_m is None else (n_before - n_after) / (radius_m * n_after)
    d_value = n_before / n_after
    state.q_m = state.q_m / (curvature * state.q_m + d_value)
    state.angle_rad = (
        curvature * (state.center_m - float(surface_center_m))
        + d_value * state.angle_rad
    )


def _apply_thin_lens(state: _AxisState, element: dict[str, Any], axis: str) -> None:
    focal = element.get(f"focal_length_{axis}_m", element.get("focal_length_m"))
    if focal in (None, 0):
        return
    focal = float(focal)
    lens_center = float(element.get(f"center_{axis}_m", 0.0))
    state.q_m = 1.0 / (1.0 / state.q_m - 1.0 / focal)
    state.angle_rad -= (state.center_m - lens_center) / focal


def _apply_mirror(state: _AxisState, element: dict[str, Any], axis: str) -> None:
    radius = element.get(f"radius_{axis}_m")
    if radius not in (None, 0):
        radius = float(radius)
        state.q_m = 1.0 / (1.0 / state.q_m - 2.0 / radius)
        mirror_center = float(element.get(f"center_{axis}_m", 0.0))
        state.angle_rad -= 2.0 * (state.center_m - mirror_center) / radius
    if bool(element.get(f"flip_{axis}", False)):
        state.center_m *= -1.0
        state.angle_rad *= -1.0


def _surface_radius(element: dict[str, Any], surface: str, axis: str) -> float | None:
    value = element.get(f"radius_{surface}_{axis}_m", element.get(f"radius_{surface}_m"))
    return None if value in (None, 0) else float(value)
