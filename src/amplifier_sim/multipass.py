"""Compile practical multipass templates into one causal crystal-hit plan.

The public configuration intentionally does not describe a non-sequential CAD
scene.  Each template produces the boundary conditions that the time-domain
solver actually needs: successive crystal hits, the optical route between
them, and the corresponding flight time.  All templates therefore share the
same overlap and gain-competition implementation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any

from .constants import LIGHT_SPEED


@dataclass(frozen=True)
class MultipassHit:
    """One traversal of the gain crystal by the injected pulse."""

    name: str
    direction: str
    marker: dict[str, Any]
    center_x_m: float
    center_y_m: float


@dataclass(frozen=True)
class MultipassTransition:
    """External propagation from one crystal exit to the next entrance."""

    name: str
    route: tuple[dict[str, Any], ...]
    initial_refractive_index: float
    initial_group_refractive_index: float
    power_transmission: float
    extra_delay_s: float
    alignment_contract: str
    # Physical length represented only on the causal clock.  Unlike a
    # free-space route, this intentionally does not diffract or translate the
    # sampled field; it is the explicit contract of route_model=delay_only.
    unpropagated_geometric_length_m: float = 0.0


@dataclass(frozen=True)
class CausalMultipassPlan:
    """Canonical plan consumed by the global-clock multipass solver."""

    template: str
    hits: tuple[MultipassHit, ...]
    transitions: tuple[MultipassTransition, ...]
    warnings: tuple[str, ...]
    assumptions: tuple[str, ...]


def uses_causal_clock(
    simulation: dict[str, Any],
    seed: dict[str, Any],
    resolved_signal_model: str,
) -> bool:
    """Return whether the unified multipass contract uses the global CFL clock.

    ``sequential`` is the retained reduced execution layer for backends that do
    not carry a resolved phase-insensitive time packet.  It consumes the same
    compiled hits and transition routes; it is no longer a separate public
    multipass configuration model.
    """

    multipass = simulation.get("multipass", {})
    if not bool(multipass.get("enabled", False)):
        return False
    execution = str(multipass.get("execution_mode", "auto"))
    if execution not in {"auto", "causal", "sequential"}:
        raise ValueError(
            "simulation.multipass.execution_mode must be auto, causal, or sequential"
        )
    if execution == "causal":
        return True
    if execution == "sequential":
        return False
    temporal = seed.get("temporal", {})
    return bool(
        resolved_signal_model == "spectral_time"
        and str(temporal.get("type", "gaussian")) != "cw"
        and temporal.get("resolved", False)
        and str(
            seed.get("joint_time_frequency", {}).get("model", "separable")
        )
        == "separable"
        and str(seed.get("spatial_modes", {}).get("model", "effective_m2"))
        != "incoherent_hg"
    )


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _fraction(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _pass_count(config: dict[str, Any], *, fixed: int | None = None) -> int:
    raw = config.get("pass_count", 2)
    if isinstance(raw, bool):
        raise ValueError("simulation.multipass.pass_count must be an integer")
    try:
        count = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "simulation.multipass.pass_count must be an integer"
        ) from exc
    if float(raw) != float(count) or count < 2 or count > 100:
        raise ValueError(
            "simulation.multipass.pass_count must be an integer in [2, 100]"
        )
    if fixed is not None and count != fixed:
        raise ValueError(
            f"simulation.multipass.template=double_pass requires pass_count={fixed}"
        )
    return count


def _hit(
    base_marker: dict[str, Any],
    *,
    index: int,
    direction: str,
    center_x_m: float,
    center_y_m: float,
    angle_x_deg: float,
    angle_y_deg: float,
    name: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> MultipassHit:
    if direction not in {"forward", "backward"}:
        raise ValueError(
            f"multipass hit {index + 1} direction must be forward or backward"
        )
    marker = copy.deepcopy(base_marker)
    marker.update(copy.deepcopy(overrides or {}))
    resolved_name = str(name or f"pass_{index + 1}")
    marker.update(
        {
            "type": "gain_medium",
            "name": resolved_name,
            "direction": direction,
            "incidence_angle_x_deg": float(angle_x_deg),
            "incidence_angle_y_deg": float(angle_y_deg),
            "center_x_m": float(center_x_m),
            "center_y_m": float(center_y_m),
        }
    )
    return MultipassHit(
        name=resolved_name,
        direction=direction,
        marker=marker,
        center_x_m=float(center_x_m),
        center_y_m=float(center_y_m),
    )


def _aperture(
    *,
    name: str,
    radius_x_m: float,
    radius_y_m: float,
    center_x_m: float,
    center_y_m: float,
) -> list[dict[str, Any]]:
    if radius_x_m <= 0.0 and radius_y_m <= 0.0:
        return []
    if radius_x_m <= 0.0 or radius_y_m <= 0.0:
        raise ValueError(
            f"{name} requires both aperture radii, or both must be zero"
        )
    return [
        {
            "type": "aperture",
            "name": name,
            "shape": "ellipse",
            "aperture_radius_x_m": radius_x_m,
            "aperture_radius_y_m": radius_y_m,
            "center_x_m": center_x_m,
            "center_y_m": center_y_m,
        }
    ]


def _mirror_route(
    *,
    name: str,
    distance_m: float,
    radius_x_m: float,
    radius_y_m: float,
    center_x_m: float,
    center_y_m: float,
    reflectivity: float,
    aperture_radius_x_m: float,
    aperture_radius_y_m: float,
    flip_x: bool = False,
    flip_y: bool = False,
) -> tuple[dict[str, Any], ...]:
    prefix = [
        {
            "type": "free_space",
            "name": f"{name}_outbound",
            "length_m": distance_m,
        }
    ]
    aperture = _aperture(
        name=f"{name}_clear_aperture",
        radius_x_m=aperture_radius_x_m,
        radius_y_m=aperture_radius_y_m,
        center_x_m=center_x_m,
        center_y_m=center_y_m,
    )
    mirror = {
        "type": "mirror",
        "name": name,
        "power_reflectivity": reflectivity,
        "radius_x_m": radius_x_m,
        "radius_y_m": radius_y_m,
        "center_x_m": center_x_m,
        "center_y_m": center_y_m,
        "flip_x": bool(flip_x),
        "flip_y": bool(flip_y),
    }
    suffix = [
        {
            "type": "free_space",
            "name": f"{name}_return",
            "length_m": distance_m,
        }
    ]
    return tuple([*prefix, *aperture, mirror, *suffix])


def _transition(
    common: dict[str, Any],
    *,
    index: int,
    route: tuple[dict[str, Any], ...],
    power_transmission: float | None = None,
    extra_delay_s: float = 0.0,
    initial_refractive_index: float | None = None,
    initial_group_refractive_index: float | None = None,
    alignment_contract: str = "template_predicted_hit",
    name: str | None = None,
    unpropagated_geometric_length_m: float = 0.0,
) -> MultipassTransition:
    phase_n = _positive(
        common.get("external_refractive_index", 1.0)
        if initial_refractive_index is None
        else initial_refractive_index,
        f"multipass transition {index + 1} refractive index",
    )
    group_n = _positive(
        common.get("external_group_refractive_index", phase_n)
        if initial_group_refractive_index is None
        else initial_group_refractive_index,
        f"multipass transition {index + 1} group refractive index",
    )
    common_transmission = _fraction(
        common.get("per_transition_power_transmission", 1.0),
        "simulation.multipass.per_transition_power_transmission",
    )
    local_transmission = 1.0 if power_transmission is None else _fraction(
        power_transmission,
        f"multipass transition {index + 1} power_transmission",
    )
    return MultipassTransition(
        name=str(name or f"transition_{index + 1}_{index + 2}"),
        route=route,
        initial_refractive_index=phase_n,
        initial_group_refractive_index=group_n,
        power_transmission=common_transmission * local_transmission,
        extra_delay_s=(
            _nonnegative(
                common.get("per_transition_extra_delay_s", 0.0),
                "simulation.multipass.per_transition_extra_delay_s",
            )
            + _nonnegative(
                extra_delay_s,
                f"multipass transition {index + 1} extra_delay_s",
            )
        ),
        alignment_contract=alignment_contract,
        unpropagated_geometric_length_m=_nonnegative(
            unpropagated_geometric_length_m,
            f"multipass transition {index + 1} unpropagated geometric length",
        ),
    )


def _compile_double_pass(
    common: dict[str, Any],
    base_marker: dict[str, Any],
) -> CausalMultipassPlan:
    _pass_count(common, fixed=2)
    cfg = common.get("double_pass", {})
    distance = _nonnegative(
        cfg.get("mirror_distance_m", 0.0),
        "simulation.multipass.double_pass.mirror_distance_m",
    )
    reflectivity = _fraction(
        cfg.get("mirror_power_reflectivity", 1.0),
        "simulation.multipass.double_pass.mirror_power_reflectivity",
    )
    values = {
        key: _finite(cfg.get(key, 0.0), f"simulation.multipass.double_pass.{key}")
        for key in (
            "mirror_radius_x_m",
            "mirror_radius_y_m",
            "mirror_center_x_m",
            "mirror_center_y_m",
            "first_incidence_angle_x_deg",
            "first_incidence_angle_y_deg",
            "return_incidence_angle_x_deg",
            "return_incidence_angle_y_deg",
            "first_center_x_m",
            "first_center_y_m",
            "return_center_x_m",
            "return_center_y_m",
        )
    }
    aperture_x = _nonnegative(
        cfg.get("mirror_aperture_radius_x_m", 0.0),
        "simulation.multipass.double_pass.mirror_aperture_radius_x_m",
    )
    aperture_y = _nonnegative(
        cfg.get("mirror_aperture_radius_y_m", 0.0),
        "simulation.multipass.double_pass.mirror_aperture_radius_y_m",
    )
    hits = (
        _hit(
            base_marker,
            index=0,
            direction="forward",
            center_x_m=values["first_center_x_m"],
            center_y_m=values["first_center_y_m"],
            angle_x_deg=values["first_incidence_angle_x_deg"],
            angle_y_deg=values["first_incidence_angle_y_deg"],
        ),
        _hit(
            base_marker,
            index=1,
            direction="backward",
            center_x_m=values["return_center_x_m"],
            center_y_m=values["return_center_y_m"],
            angle_x_deg=values["return_incidence_angle_x_deg"],
            angle_y_deg=values["return_incidence_angle_y_deg"],
        ),
    )
    route = _mirror_route(
        name="double_pass_return_mirror",
        distance_m=distance,
        radius_x_m=values["mirror_radius_x_m"],
        radius_y_m=values["mirror_radius_y_m"],
        center_x_m=values["mirror_center_x_m"],
        center_y_m=values["mirror_center_y_m"],
        reflectivity=reflectivity,
        aperture_radius_x_m=aperture_x,
        aperture_radius_y_m=aperture_y,
    )
    return CausalMultipassPlan(
        template="double_pass",
        hits=hits,
        transitions=(_transition(common, index=0, route=route),),
        warnings=(),
        assumptions=(
            "The configured first/return crystal centres and angles are boundary conditions; the external mirror route propagates the full sampled field between them.",
        ),
    )


def _ray_return(
    center_m: float,
    angle_deg: float,
    *,
    crystal_length_m: float,
    mirror_distance_m: float,
    mirror_radius_m: float,
    mirror_center_m: float,
    crystal_refractive_index: float,
    external_refractive_index: float,
) -> tuple[float, float]:
    """Paraxial unfolded round trip from one crystal entrance to the next."""

    crystal_slope = math.tan(math.radians(angle_deg))
    exit_center = center_m + crystal_length_m * crystal_slope
    external_slope = (
        crystal_refractive_index
        / external_refractive_index
        * crystal_slope
    )
    mirror_spot = exit_center + mirror_distance_m * external_slope
    returned_slope = external_slope
    if mirror_radius_m != 0.0:
        returned_slope -= 2.0 * (mirror_spot - mirror_center_m) / mirror_radius_m
    returned_center = mirror_spot + mirror_distance_m * returned_slope
    returned_crystal_slope = (
        external_refractive_index
        / crystal_refractive_index
        * returned_slope
    )
    return returned_center, math.degrees(math.atan(returned_crystal_slope))


def _compile_two_mirror_like(
    common: dict[str, Any],
    base_marker: dict[str, Any],
    *,
    template: str,
    crystal_length_m: float,
    crystal_refractive_index: float,
) -> CausalMultipassPlan:
    count = _pass_count(common)
    source = common.get(template, {})
    if template == "herriott":
        separation = _positive(
            source.get("mirror_separation_m", 0.2),
            "simulation.multipass.herriott.mirror_separation_m",
        )
        crystal_center_from_left = _nonnegative(
            source.get("crystal_position_from_left_m", 0.1),
            "simulation.multipass.herriott.crystal_position_from_left_m",
        )
        crystal_half_length = 0.5 * crystal_length_m
        if (
            crystal_center_from_left < crystal_half_length
            or crystal_center_from_left > separation - crystal_half_length
        ):
            raise ValueError(
                "the Herriott crystal centre must lie between the mirrors with the full crystal length inside the mirror separation"
            )
        left_distance = crystal_center_from_left - crystal_half_length
        right_distance = (
            separation - crystal_center_from_left - crystal_half_length
        )
    else:
        left_distance = _nonnegative(
            source.get("left_mirror_distance_m", 0.1),
            "simulation.multipass.two_mirror.left_mirror_distance_m",
        )
        right_distance = _nonnegative(
            source.get("right_mirror_distance_m", 0.1),
            "simulation.multipass.two_mirror.right_mirror_distance_m",
        )
        separation = left_distance + right_distance

    mirror: dict[str, dict[str, float]] = {}
    for side in ("left", "right"):
        mirror[side] = {
            "distance_m": left_distance if side == "left" else right_distance,
            "radius_x_m": _finite(
                source.get(f"{side}_mirror_radius_x_m", 0.0),
                f"simulation.multipass.{template}.{side}_mirror_radius_x_m",
            ),
            "radius_y_m": _finite(
                source.get(f"{side}_mirror_radius_y_m", 0.0),
                f"simulation.multipass.{template}.{side}_mirror_radius_y_m",
            ),
            "center_x_m": _finite(
                source.get(f"{side}_mirror_center_x_m", 0.0),
                f"simulation.multipass.{template}.{side}_mirror_center_x_m",
            ),
            "center_y_m": _finite(
                source.get(f"{side}_mirror_center_y_m", 0.0),
                f"simulation.multipass.{template}.{side}_mirror_center_y_m",
            ),
            "reflectivity": _fraction(
                source.get(f"{side}_mirror_power_reflectivity", 1.0),
                f"simulation.multipass.{template}.{side}_mirror_power_reflectivity",
            ),
            "aperture_x_m": _nonnegative(
                source.get(f"{side}_mirror_aperture_radius_x_m", 0.0),
                f"simulation.multipass.{template}.{side}_mirror_aperture_radius_x_m",
            ),
            "aperture_y_m": _nonnegative(
                source.get(f"{side}_mirror_aperture_radius_y_m", 0.0),
                f"simulation.multipass.{template}.{side}_mirror_aperture_radius_y_m",
            ),
        }

    center_x = _finite(
        source.get("injection_center_x_m", 0.0),
        f"simulation.multipass.{template}.injection_center_x_m",
    )
    center_y = _finite(
        source.get("injection_center_y_m", 0.0),
        f"simulation.multipass.{template}.injection_center_y_m",
    )
    angle_x = _finite(
        source.get("injection_angle_x_deg", 0.0),
        f"simulation.multipass.{template}.injection_angle_x_deg",
    )
    angle_y = _finite(
        source.get("injection_angle_y_deg", 0.0),
        f"simulation.multipass.{template}.injection_angle_y_deg",
    )
    hits: list[MultipassHit] = []
    transitions: list[MultipassTransition] = []
    for index in range(count):
        direction = "forward" if index % 2 == 0 else "backward"
        hits.append(
            _hit(
                base_marker,
                index=index,
                direction=direction,
                center_x_m=center_x,
                center_y_m=center_y,
                angle_x_deg=angle_x,
                angle_y_deg=angle_y,
            )
        )
        if index + 1 == count:
            continue
        side = "right" if direction == "forward" else "left"
        spec = mirror[side]
        transitions.append(
            _transition(
                common,
                index=index,
                route=_mirror_route(
                    name=f"{template}_{side}_mirror_pass_{index + 1}",
                    distance_m=spec["distance_m"],
                    radius_x_m=spec["radius_x_m"],
                    radius_y_m=spec["radius_y_m"],
                    center_x_m=spec["center_x_m"],
                    center_y_m=spec["center_y_m"],
                    reflectivity=spec["reflectivity"],
                    aperture_radius_x_m=spec["aperture_x_m"],
                    aperture_radius_y_m=spec["aperture_y_m"],
                ),
            )
        )
        center_x, angle_x = _ray_return(
            center_x,
            angle_x,
            crystal_length_m=crystal_length_m,
            mirror_distance_m=spec["distance_m"],
            mirror_radius_m=spec["radius_x_m"],
            mirror_center_m=spec["center_x_m"],
            crystal_refractive_index=crystal_refractive_index,
            external_refractive_index=_positive(
                common.get("external_refractive_index", 1.0),
                "simulation.multipass.external_refractive_index",
            ),
        )
        center_y, angle_y = _ray_return(
            center_y,
            angle_y,
            crystal_length_m=crystal_length_m,
            mirror_distance_m=spec["distance_m"],
            mirror_radius_m=spec["radius_y_m"],
            mirror_center_m=spec["center_y_m"],
            crystal_refractive_index=crystal_refractive_index,
            external_refractive_index=_positive(
                common.get("external_refractive_index", 1.0),
                "simulation.multipass.external_refractive_index",
            ),
        )

    warnings: list[str] = []
    assumptions = [
        "Mirror-to-mirror spot locations use a paraxial unfolded chief-ray recurrence with first-order Snell slope scaling at the crystal end faces; the sampled complex envelope is propagated through each explicit route and aligned to the predicted crystal hit.",
    ]
    if template == "herriott":
        # Express the plane-parallel crystal in the external medium's ray
        # coordinates. Its ABCD propagation length is L*n_out/n_crystal;
        # using the geometric mirror separation gives the empty-cavity
        # stability criterion even when the compiled ray includes a slab.
        effective_separation = (
            left_distance + right_distance
            + crystal_length_m
            * _positive(common.get("external_refractive_index", 1.0),
                        "simulation.multipass.external_refractive_index")
            / crystal_refractive_index
        )
        for axis in ("x", "y"):
            left_radius = mirror["left"][f"radius_{axis}_m"]
            right_radius = mirror["right"][f"radius_{axis}_m"]
            if left_radius != 0.0 and right_radius != 0.0:
                g_product = (1.0 - effective_separation / left_radius) * (
                    1.0 - effective_separation / right_radius
                )
                if not 0.0 < g_product < 1.0:
                    warnings.append(
                        f"Herriott {axis}-axis cold cavity including the plane-parallel crystal has g1*g2={g_product:.6g}, outside the ordinary stable-resonator interval (0, 1)."
                    )
        if not bool(source.get("geometric_output_after_pass", True)):
            raise ValueError(
                "finite Herriott pass_count requires geometric_output_after_pass=true; a passive closed cell does not stop after an arbitrary pass"
            )
        assumptions.append(
            "The finite Herriott sequence assumes geometrical injection/extraction after the requested final hit; it is not a coherently leaking passive cavity model."
        )
    else:
        extraction = str(source.get("finite_pass_extraction", "ideal_switch"))
        if extraction not in {"ideal_switch", "geometric"}:
            raise ValueError(
                "simulation.multipass.two_mirror.finite_pass_extraction must be ideal_switch or geometric"
            )
        assumptions.append(
            "A finite two-mirror pass count requires the configured ideal_switch or geometric extraction; ordinary passive mirror leakage is not represented as a coherent cavity."
        )
    return CausalMultipassPlan(
        template=template,
        hits=tuple(hits),
        transitions=tuple(transitions),
        warnings=tuple(warnings),
        assumptions=tuple(assumptions),
    )


def _compile_relay_4f(
    common: dict[str, Any],
    base_marker: dict[str, Any],
) -> CausalMultipassPlan:
    count = _pass_count(common)
    cfg = common.get("relay_4f", {})
    positive_keys = (
        "object_to_first_lens_m",
        "first_focal_length_m",
        "lens_separation_m",
        "second_focal_length_m",
        "second_lens_to_mirror_m",
    )
    values = {
        key: _positive(
            cfg.get(key, 0.1), f"simulation.multipass.relay_4f.{key}"
        )
        for key in positive_keys
    }
    relay_transmission = _fraction(
        cfg.get("relay_power_transmission", 1.0),
        "simulation.multipass.relay_4f.relay_power_transmission",
    )
    scalar = {
        key: _finite(cfg.get(key, 0.0), f"simulation.multipass.relay_4f.{key}")
        for key in (
            "center_x_m",
            "center_y_m",
            "first_center_x_m",
            "first_center_y_m",
            "first_incidence_angle_x_deg",
            "first_incidence_angle_y_deg",
            "spot_step_x_m",
            "spot_step_y_m",
            "angle_step_x_deg",
            "angle_step_y_deg",
        )
    }
    hits = tuple(
        _hit(
            base_marker,
            index=index,
            direction="forward" if index % 2 == 0 else "backward",
            center_x_m=scalar["first_center_x_m"]
            + index * scalar["spot_step_x_m"],
            center_y_m=scalar["first_center_y_m"]
            + index * scalar["spot_step_y_m"],
            angle_x_deg=scalar["first_incidence_angle_x_deg"]
            + index * scalar["angle_step_x_deg"],
            angle_y_deg=scalar["first_incidence_angle_y_deg"]
            + index * scalar["angle_step_y_deg"],
        )
        for index in range(count)
    )
    route = (
        {
            "type": "double_pass_lens_relay",
            "name": "multipass_4f_relay",
            **values,
            "center_x_m": scalar["center_x_m"],
            "center_y_m": scalar["center_y_m"],
            "power_transmission": relay_transmission,
        },
    )
    from .optics import double_pass_lens_relay_matrix

    relay_matrix = double_pass_lens_relay_matrix(route[0])
    a, b, _c, _d = relay_matrix.ravel()
    scale = max(*(abs(value) for value in values.values()), 1.0)
    if abs(float(b)) > 1e-10 * scale or not math.isclose(
        abs(float(a)), 1.0, rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError(
            "simulation.multipass.relay_4f must form a unit-magnification B=0 round trip; "
            f"configured ABCD is [[{relay_matrix[0, 0]:.9g}, {relay_matrix[0, 1]:.9g}], "
            f"[{relay_matrix[1, 0]:.9g}, {relay_matrix[1, 1]:.9g}]]"
        )
    transitions = tuple(
        _transition(common, index=index, route=route)
        for index in range(count - 1)
    )
    return CausalMultipassPlan(
        template="relay_4f",
        hits=hits,
        transitions=transitions,
        warnings=(
            "The current FFT relay operator requires a unit-magnification B=0 round trip; validation reports the configured ABCD matrix if that condition is not met.",
        ),
        assumptions=(
            "Per-pass spot/angle increments are prescribed pick-off or mirror-array boundary conditions, while the 4f relay propagates the sampled beam size and wavefront.",
        ),
    )


def _compile_prescribed(
    common: dict[str, Any],
    base_marker: dict[str, Any],
) -> CausalMultipassPlan:
    cfg = common.get("prescribed", {})
    raw_hits = cfg.get("hits", [])
    raw_transitions = cfg.get("transitions", [])
    if not isinstance(raw_hits, list) or len(raw_hits) < 2:
        raise ValueError(
            "simulation.multipass.prescribed.hits must contain at least two crystal hits"
        )
    if len(raw_transitions) != len(raw_hits) - 1:
        raise ValueError(
            "simulation.multipass.prescribed.transitions must contain exactly len(hits)-1 routes"
        )
    if _pass_count(common) != len(raw_hits):
        raise ValueError(
            "simulation.multipass.pass_count must equal the number of prescribed hits"
        )
    hits: list[MultipassHit] = []
    for index, raw in enumerate(raw_hits):
        if not isinstance(raw, dict):
            raise ValueError(f"prescribed hit {index + 1} must be an object")
        hits.append(
            _hit(
                base_marker,
                index=index,
                direction=str(
                    raw.get(
                        "direction",
                        "forward" if index % 2 == 0 else "backward",
                    )
                ),
                center_x_m=_finite(
                    raw.get("center_x_m", 0.0),
                    f"prescribed hit {index + 1} center_x_m",
                ),
                center_y_m=_finite(
                    raw.get("center_y_m", 0.0),
                    f"prescribed hit {index + 1} center_y_m",
                ),
                angle_x_deg=_finite(
                    raw.get("incidence_angle_x_deg", 0.0),
                    f"prescribed hit {index + 1} incidence_angle_x_deg",
                ),
                angle_y_deg=_finite(
                    raw.get("incidence_angle_y_deg", 0.0),
                    f"prescribed hit {index + 1} incidence_angle_y_deg",
                ),
                name=str(raw.get("name") or f"pass_{index + 1}"),
                # Preserve the complete gain-hit contract when a former
                # ordered bench is migrated.  Routing-only fields are
                # overwritten by ``_hit`` above, while coating, aperture and
                # plugin extras remain attached to the physical crystal hit.
                overrides={
                    key: copy.deepcopy(value)
                    for key, value in raw.items()
                    if key
                    not in {
                        "name",
                        "direction",
                        "center_x_m",
                        "center_y_m",
                        "incidence_angle_x_deg",
                        "incidence_angle_y_deg",
                    }
                    and value is not None
                },
            )
        )
    transitions: list[MultipassTransition] = []
    for index, raw in enumerate(raw_transitions):
        if not isinstance(raw, dict):
            raise ValueError(
                f"prescribed transition {index + 1} must be an object"
            )
        route_model = str(raw.get("route_model", "delay_only"))
        length = _nonnegative(
            raw.get("geometric_length_m", 0.0),
            f"prescribed transition {index + 1} geometric_length_m",
        )
        if route_model == "delay_only":
            route: tuple[dict[str, Any], ...] = ()
        elif route_model == "optical_train":
            raw_route = raw.get("route", [])
            if not isinstance(raw_route, list) or not all(
                isinstance(element, dict) for element in raw_route
            ):
                raise ValueError(
                    f"prescribed transition {index + 1} route must be a list of optical elements"
                )
            route = tuple(copy.deepcopy(raw_route))
        elif route_model == "free_space":
            route = (
                {
                    "type": "free_space",
                    "name": f"prescribed_transition_{index + 1}",
                    "length_m": length,
                },
            )
        elif route_model == "round_trip_mirror":
            route = _mirror_route(
                name=f"prescribed_mirror_{index + 1}",
                distance_m=0.5 * length,
                radius_x_m=_finite(raw.get("mirror_radius_x_m", 0.0), "mirror_radius_x_m"),
                radius_y_m=_finite(raw.get("mirror_radius_y_m", 0.0), "mirror_radius_y_m"),
                center_x_m=_finite(raw.get("mirror_center_x_m", 0.0), "mirror_center_x_m"),
                center_y_m=_finite(raw.get("mirror_center_y_m", 0.0), "mirror_center_y_m"),
                reflectivity=_fraction(raw.get("mirror_power_reflectivity", 1.0), "mirror_power_reflectivity"),
                aperture_radius_x_m=_nonnegative(raw.get("mirror_aperture_radius_x_m", 0.0), "mirror_aperture_radius_x_m"),
                aperture_radius_y_m=_nonnegative(raw.get("mirror_aperture_radius_y_m", 0.0), "mirror_aperture_radius_y_m"),
                flip_x=bool(raw.get("flip_x", False)),
                flip_y=bool(raw.get("flip_y", False)),
            )
        else:
            raise ValueError(
                f"prescribed transition {index + 1} route_model must be delay_only, free_space, optical_train, or round_trip_mirror"
            )
        phase_n = raw.get("refractive_index")
        group_n = raw.get("group_refractive_index")
        transitions.append(
            _transition(
                common,
                index=index,
                route=route,
                power_transmission=raw.get("power_transmission", 1.0),
                extra_delay_s=raw.get("extra_delay_s", 0.0),
                initial_refractive_index=(
                    None if phase_n is None else float(phase_n)
                ),
                initial_group_refractive_index=(
                    None if group_n is None else float(group_n)
                ),
                alignment_contract="prescribed_boundary_condition",
                name=str(
                    raw.get("name")
                    or f"transition_{index + 1}_{index + 2}"
                ),
                unpropagated_geometric_length_m=(
                    length if route_model == "delay_only" else 0.0
                ),
            )
        )
    return CausalMultipassPlan(
        template="prescribed",
        hits=tuple(hits),
        transitions=tuple(transitions),
        warnings=(
            "Prescribed mode does not claim to reconstruct omitted mirrors: supplied hit centres, angles, route length and delay are the model boundary conditions.",
        ),
        assumptions=(
            "The full sampled field is propagated through every supplied route and then registered to the next supplied crystal-hit centre.",
        ),
    )


def _compile_sequential(
    common: dict[str, Any],
    base_marker: dict[str, Any],
) -> CausalMultipassPlan:
    """Compile a migrated quick-pass contract into the unified plan.

    This template is intentionally not offered for new physical layouts.  It
    keeps useful non-time-resolved backend coverage for old configurations
    while making every hit and transition flow through ``simulation.multipass``.
    """

    count = _pass_count(common)
    cfg = common.get("sequential", {})
    alternate = bool(cfg.get("alternate_direction", True))
    first_direction = str(base_marker.get("direction", "forward"))
    if first_direction == "auto":
        first_direction = "forward"
    if first_direction not in {"forward", "backward"}:
        raise ValueError(
            "simulation.multipass.sequential first direction must be forward or backward"
        )
    raw_route = cfg.get("route", [])
    if not isinstance(raw_route, list) or not all(
        isinstance(element, dict) for element in raw_route
    ):
        raise ValueError(
            "simulation.multipass.sequential.route must be a list of optical elements"
        )
    route = copy.deepcopy(raw_route)
    if bool(cfg.get("flip_x", False)):
        route.append(
            {
                "type": "mirror",
                "name": "migrated_interpass_x_flip",
                "power_reflectivity": 1.0,
                "flip_x": True,
            }
        )
    hits = []
    for index in range(count):
        reverse = alternate and index % 2 == 1
        direction = (
            "backward" if first_direction == "forward" else "forward"
        ) if reverse else first_direction
        hits.append(
            _hit(
                base_marker,
                index=index,
                direction=direction,
                center_x_m=_finite(
                    base_marker.get("center_x_m", 0.0), "gain hit center_x_m"
                ),
                center_y_m=_finite(
                    base_marker.get("center_y_m", 0.0), "gain hit center_y_m"
                ),
                angle_x_deg=_finite(
                    base_marker.get("incidence_angle_x_deg", 0.0),
                    "gain hit incidence_angle_x_deg",
                ),
                angle_y_deg=_finite(
                    base_marker.get("incidence_angle_y_deg", 0.0),
                    "gain hit incidence_angle_y_deg",
                ),
                name=f"pass_{index + 1}",
            )
        )
    transitions = tuple(
        _transition(
            common,
            index=index,
            route=tuple(copy.deepcopy(route)),
            alignment_contract="migrated_sequential_route",
        )
        for index in range(count - 1)
    )
    return CausalMultipassPlan(
        template="sequential",
        hits=tuple(hits),
        transitions=transitions,
        warnings=(
            "This migrated quick-pass case executes the unified hit/route plan sequentially because its original configuration did not define a causal physical layout.",
        ),
        assumptions=(
            "Each complete crystal pass finishes before the next begins; simultaneous occupancy is not represented in this migrated reduced contract.",
        ),
    )


def compile_multipass_plan(
    multipass_config: dict[str, Any],
    base_gain_marker: dict[str, Any],
    *,
    crystal_length_m: float,
    crystal_refractive_index: float = 1.0,
    global_extra_delay_s: float = 0.0,
) -> CausalMultipassPlan:
    """Compile and validate the selected public multipass template."""

    if not isinstance(multipass_config, dict):
        raise ValueError("simulation.multipass must be an object")
    template = str(multipass_config.get("template", "double_pass"))
    if template == "double_pass":
        plan = _compile_double_pass(multipass_config, base_gain_marker)
    elif template in {"two_mirror", "herriott"}:
        plan = _compile_two_mirror_like(
            multipass_config,
            base_gain_marker,
            template=template,
            crystal_length_m=_nonnegative(
                crystal_length_m, "crystal_length_m"
            ),
            crystal_refractive_index=_positive(
                crystal_refractive_index,
                "crystal_refractive_index",
            ),
        )
    elif template == "relay_4f":
        plan = _compile_relay_4f(multipass_config, base_gain_marker)
    elif template == "prescribed":
        plan = _compile_prescribed(multipass_config, base_gain_marker)
    elif template == "sequential":
        plan = _compile_sequential(multipass_config, base_gain_marker)
    else:
        raise ValueError(
            "simulation.multipass.template must be double_pass, two_mirror, herriott, relay_4f, prescribed, or sequential"
        )
    global_delay = _nonnegative(
        global_extra_delay_s,
        "simulation.unmodeled_between_pass_delay_s",
    )
    if global_delay == 0.0:
        return plan
    transitions = tuple(
        MultipassTransition(
            name=transition.name,
            route=transition.route,
            initial_refractive_index=transition.initial_refractive_index,
            initial_group_refractive_index=transition.initial_group_refractive_index,
            power_transmission=transition.power_transmission,
            extra_delay_s=transition.extra_delay_s + global_delay,
            alignment_contract=transition.alignment_contract,
            unpropagated_geometric_length_m=(
                transition.unpropagated_geometric_length_m
            ),
        )
        for transition in plan.transitions
    )
    return CausalMultipassPlan(
        template=plan.template,
        hits=plan.hits,
        transitions=transitions,
        warnings=plan.warnings,
        assumptions=plan.assumptions,
    )


def sequential_plan_components(
    plan: CausalMultipassPlan,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Adapt one compiled plan to backends that advance complete passes.

    Transition transmission and timing are encoded as a named zero-thickness
    loss element.  The same route elements therefore drive diffraction,
    coating losses and the population-flight-time ledger in every sequential
    backend.
    """

    markers = [copy.deepcopy(hit.marker) for hit in plan.hits]
    segments: list[list[dict[str, Any]]] = []
    for index, transition in enumerate(plan.transitions):
        # The transition defines the medium immediately after this crystal
        # exit. Preserve both indices when adapting to complete-pass solvers.
        markers[index]["outside_refractive_index"] = transition.initial_refractive_index
        markers[index]["outside_group_refractive_index"] = transition.initial_group_refractive_index
        route = [copy.deepcopy(element) for element in transition.route]
        group_delay_s = float(transition.extra_delay_s) + (
            float(transition.initial_group_refractive_index)
            * float(transition.unpropagated_geometric_length_m)
            / LIGHT_SPEED
        )
        if transition.power_transmission != 1.0 or group_delay_s != 0.0:
            route.append(
                {
                    "type": "loss",
                    "name": f"{transition.name}_contract",
                    "power_transmission": float(
                        transition.power_transmission
                    ),
                    "group_delay_s": group_delay_s,
                }
            )
        segments.append(route)
    return markers, segments
