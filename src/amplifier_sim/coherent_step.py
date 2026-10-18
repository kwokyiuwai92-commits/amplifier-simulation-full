"""Local operator policies for coherent propagation inside one crystal cell.

The legacy coherent solver applies one full gain step followed by material
dispersion and nonlinear response.  That inexpensive ordering remains useful
for weakly coupled problems.  The symmetric policy uses this module to select
enough local substeps that gain, dispersion, Kerr phase, and strong-field loss
remain small before applying a reversible Strang ordering.

This module deliberately contains no field or population mutation.  Keeping
policy selection separate makes the compatibility boundary explicit and lets
the propagator retain one energy-accounting implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


def nonlinear_gain_coupling_blockers(physics: dict[str, Any], material: dict[str, Any]) -> tuple[str, ...]:
    """Memoryless multiphoton loss may use the configured gain schedule.

    Carrier dynamics and unknown loss operators retain the full-rate guard.
    Grouped gain with pure MPA is an operator-splitting approximation; the
    existing user-selected step limits and convergence diagnostics still apply.
    """
    loss = str(physics.get("nonlinear_loss_model", "off"))
    if loss == "off":
        return ()
    if (loss == "strong_field_time_domain"
            and str(material.get("strong_field", {}).get("free_carrier", {}).get("model", "off")) == "off"):
        return ()
    return (loss,)


@dataclass(frozen=True)
class CoherentLocalOperatorSettings:
    """Configuration for the coherent crystal-local advancement."""

    scheme: str = "symmetric_adaptive"
    adaptive_substeps: bool = True
    minimum_substeps: int = 1
    maximum_substeps: int = 64
    maximum_gain_exponent_per_substep: float = 0.1
    maximum_nonlinear_phase_rad_per_substep: float = 0.05
    maximum_dispersion_phase_rad_per_substep: float = 0.1
    maximum_thermal_phase_rad_per_substep: float = 0.1
    maximum_fractional_loss_per_substep: float = 0.05
    spatial_population_update: str = "shared_cell"
    multirate_gain_enabled: bool = False
    maximum_gain_exponent_per_update: float = 0.05
    maximum_optical_substeps_per_gain_update: int = 4
    maximum_gain_updates_per_cell: int | None = None

    @classmethod
    def from_seed_config(cls, seed_config: dict[str, Any]) -> "CoherentLocalOperatorSettings":
        configured = seed_config.get("coherent", {}).get("local_operator", {})
        multirate = configured.get("multirate_gain", {})
        return cls(
            scheme=str(configured.get("scheme", cls.scheme)),
            adaptive_substeps=bool(
                configured.get("adaptive_substeps", cls.adaptive_substeps)
            ),
            minimum_substeps=int(
                configured.get("minimum_substeps", cls.minimum_substeps)
            ),
            maximum_substeps=int(
                configured.get("maximum_substeps", cls.maximum_substeps)
            ),
            maximum_gain_exponent_per_substep=float(
                configured.get(
                    "maximum_gain_exponent_per_substep",
                    cls.maximum_gain_exponent_per_substep,
                )
            ),
            maximum_nonlinear_phase_rad_per_substep=float(
                configured.get(
                    "maximum_nonlinear_phase_rad_per_substep",
                    cls.maximum_nonlinear_phase_rad_per_substep,
                )
            ),
            maximum_dispersion_phase_rad_per_substep=float(
                configured.get(
                    "maximum_dispersion_phase_rad_per_substep",
                    cls.maximum_dispersion_phase_rad_per_substep,
                )
            ),
            maximum_thermal_phase_rad_per_substep=float(
                configured.get(
                    "maximum_thermal_phase_rad_per_substep",
                    cls.maximum_thermal_phase_rad_per_substep,
                )
            ),
            maximum_fractional_loss_per_substep=float(
                configured.get(
                    "maximum_fractional_loss_per_substep",
                    cls.maximum_fractional_loss_per_substep,
                )
            ),
            spatial_population_update=str(
                configured.get("spatial_population_update", cls.spatial_population_update)
            ),
            multirate_gain_enabled=bool(
                multirate.get("enabled", cls.multirate_gain_enabled)
            ),
            maximum_gain_exponent_per_update=float(
                multirate.get(
                    "maximum_gain_exponent_per_update",
                    cls.maximum_gain_exponent_per_update,
                )
            ),
            maximum_optical_substeps_per_gain_update=int(
                multirate.get(
                    "maximum_optical_substeps_per_update",
                    cls.maximum_optical_substeps_per_gain_update,
                )
            ),
            maximum_gain_updates_per_cell=(
                None if multirate.get("maximum_updates_per_cell") is None
                else int(multirate["maximum_updates_per_cell"])
            ),
        )


@dataclass(frozen=True)
class CoherentLocalStepPlan:
    """Chosen substep count together with the estimates that caused it."""

    count: int
    uncapped_count: int
    capped: bool
    gain_exponent: float
    nonlinear_phase_rad: float
    dispersion_phase_rad: float
    thermal_phase_rad: float
    fractional_loss: float


@dataclass(frozen=True)
class CoherentGainUpdatePlan:
    """Gain-call schedule inside a finer optical local-step plan.

    A positive schedule entry is the number of adjacent optical substeps
    represented by the gain call at that index.  Zero means that dispersion,
    nonlinear response, diffraction, and thermal phase still advance at the
    fine rate while gain is deferred to a representative step near the group
    midpoint.
    """

    schedule: tuple[int, ...]
    update_count: int
    maximum_group_size: int
    reason: str

    @property
    def active(self) -> bool:
        return self.update_count < len(self.schedule)


def choose_local_substeps(
    settings: CoherentLocalOperatorSettings,
    *,
    gain_exponent: float,
    nonlinear_phase_rad: float,
    dispersion_phase_rad: float,
    thermal_phase_rad: float,
    fractional_loss: float,
    loss_order: int = 1,
) -> CoherentLocalStepPlan:
    """Return a conservative local substep plan from dimensionless changes."""

    gain = abs(float(gain_exponent))
    nonlinear = abs(float(nonlinear_phase_rad))
    dispersion = abs(float(dispersion_phase_rad))
    thermal = abs(float(thermal_phase_rad))
    loss = min(max(float(fractional_loss), 0.0), 1.0)
    required = max(int(settings.minimum_substeps), 1)
    if settings.adaptive_substeps:
        required = max(
            required,
            _positive_ceiling(gain / settings.maximum_gain_exponent_per_substep),
            _positive_ceiling(
                nonlinear / settings.maximum_nonlinear_phase_rad_per_substep
            ),
            _positive_ceiling(
                dispersion / settings.maximum_dispersion_phase_rad_per_substep
            ),
            _positive_ceiling(
                thermal / settings.maximum_thermal_phase_rad_per_substep
            ),
            _loss_substeps(loss, settings.maximum_fractional_loss_per_substep, loss_order),
        )
    count = min(required, int(settings.maximum_substeps))
    return CoherentLocalStepPlan(
        count=count,
        uncapped_count=required,
        capped=count < required,
        gain_exponent=gain,
        nonlinear_phase_rad=nonlinear,
        dispersion_phase_rad=dispersion,
        thermal_phase_rad=thermal,
        fractional_loss=loss,
    )


def choose_multirate_gain_updates(
    settings: CoherentLocalOperatorSettings,
    *,
    optical_substeps: int,
    gain_exponent: float,
    transient_gain_enabled: bool,
    whole_fluence_gain_enabled: bool,
    intrapulse_relaxation_enabled: bool,
    coupling_blockers: tuple[str, ...] = (),
) -> CoherentGainUpdatePlan:
    """Choose a guarded coarse gain schedule for fine optical substeps.

    The full-rate schedule is returned whenever the optional approximation is
    disabled, the selected gain backend does not support grouped updates, or
    its accuracy limits cannot merge even two optical substeps. Intrapulse
    relaxation and any caller-declared stateful local operator also remain
    full-rate because changing their split ordering can matter even when the
    optical gain exponent is small. The coherent hybrid, transient soft-window,
    and axisymmetric closures support grouped updates only after those guards:
    their gain and population operator is evaluated at a representative
    microstep while diffraction, frozen thermal phase, dispersion, and Kerr
    continue at the fine optical rate.
    """

    count = max(int(optical_substeps), 1)

    def full_rate(reason: str) -> CoherentGainUpdatePlan:
        return CoherentGainUpdatePlan(
            schedule=tuple(1 for _ in range(count)),
            update_count=count,
            maximum_group_size=1,
            reason=reason,
        )

    if not settings.multirate_gain_enabled:
        return full_rate("disabled")
    update_cap = settings.maximum_gain_updates_per_cell
    if update_cap is not None:
        transient_cap = transient_gain_enabled and settings.spatial_population_update == "shared_cell"
        whole_cap = (whole_fluence_gain_enabled and not transient_gain_enabled
                     and settings.spatial_population_update == "independent_groups")
        if (not (transient_cap or whole_cap) or settings.scheme != "symmetric_adaptive"
                or intrapulse_relaxation_enabled or coupling_blockers):
            raise ValueError("A gain update count cap requires independent whole-fluence or shared-cell transient gain without coupling blockers")
        if update_cap < 1:
            raise ValueError("maximum_updates_per_cell must be positive")
        return _balanced_gain_schedule(count, min(count, update_cap),
                                       "transient_count_cap" if transient_cap else "independent_count_cap")
    if settings.scheme != "symmetric_adaptive":
        return full_rate("sequential_scheme")
    if not transient_gain_enabled and not whole_fluence_gain_enabled:
        return full_rate("gain_backend_unsupported")
    if intrapulse_relaxation_enabled:
        return full_rate("intrapulse_relaxation")
    normalized_blockers = tuple(
        str(blocker).strip().replace(" ", "_")
        for blocker in coupling_blockers
        if str(blocker).strip()
    )
    if normalized_blockers:
        return full_rate(f"coupling_blocked_{normalized_blockers[0]}")
    if count <= 1:
        return full_rate("single_optical_substep")

    gain = abs(float(gain_exponent))
    if not math.isfinite(gain):
        return full_rate("nonfinite_gain_estimate")
    if gain <= 0.0:
        gain_limited_group = count
    else:
        gain_limited_group = max(
            int(
                math.floor(
                    settings.maximum_gain_exponent_per_update
                    * count
                    / gain
                    + 8.0 * math.ulp(1.0)
                )
            ),
            1,
        )
    maximum_group = min(
        count,
        gain_limited_group,
        max(int(settings.maximum_optical_substeps_per_gain_update), 1),
    )
    if maximum_group <= 1:
        return full_rate("gain_accuracy_limit")

    updates = int(math.ceil(count / maximum_group))
    return _balanced_gain_schedule(count, updates, "multirate")


def _balanced_gain_schedule(count: int, updates: int, reason: str) -> CoherentGainUpdatePlan:
    """Partition every optical step once; even groups use the earlier midpoint."""
    base, remainder = divmod(count, updates)
    group_sizes = [
        base + int(index < remainder) for index in range(updates)
    ]
    schedule = [0] * count
    start = 0
    for size in group_sizes:
        group_size = int(size)
        representative = start + (group_size - 1) // 2
        schedule[representative] = group_size
        start += group_size
    return CoherentGainUpdatePlan(
        schedule=tuple(schedule),
        update_count=updates,
        maximum_group_size=max(group_sizes),
        reason=reason,
    )


def _positive_ceiling(value: float) -> int:
    if not math.isfinite(value):
        return 2**31 - 1
    return max(int(math.ceil(value)), 1)


def _loss_substeps(total_loss: float, maximum_step_loss: float, order: int = 1) -> int:
    """Bound the first (highest-intensity) step of dI/dz = -beta I**order."""
    if total_loss <= 0.0:
        return 1
    if total_loss >= 1.0:
        return 2**31 - 1
    if order == 1:
        return _positive_ceiling(
            math.log1p(-total_loss) / math.log1p(-maximum_step_loss)
        )
    # A K-photon optical depth scales with distance; its logarithm does not.
    # D = (1-loss)**(-(K-1)) - 1, hence N >= D_total / D_allowed.
    def log_depth(loss: float) -> float:
        exponent = -(order - 1) * math.log1p(-loss)
        return exponent + math.log(-math.expm1(-exponent))

    log_required = log_depth(total_loss) - log_depth(maximum_step_loss)
    if log_required >= math.log(2**31 - 1):
        return 2**31 - 1
    return _positive_ceiling(math.exp(log_required))
