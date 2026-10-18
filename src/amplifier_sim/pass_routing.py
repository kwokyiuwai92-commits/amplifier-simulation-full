"""Shared pass-direction semantics for simple and explicit optical paths."""

from __future__ import annotations


def resolve_pump_direction(source_direction: str, marker_direction: str = "auto") -> str:
    """Resolve the single Pump gain hit, with the same override as Seed.

    An explicit gain marker describes that crystal encounter; ``auto`` uses
    the source's default direction. Both preview and transport use this rule.
    """

    if source_direction not in {"forward", "backward"}:
        raise ValueError("pump.direction must be forward or backward")
    if marker_direction not in {"auto", "forward", "backward"}:
        raise ValueError("gain_medium.direction must be auto, forward, or backward")
    return source_direction if marker_direction == "auto" else marker_direction


def resolve_pass_reverse(
    seed_direction: str,
    marker_direction: str,
    pass_number: int,
    alternate_direction: bool,
    *,
    explicit_gain_hit: bool,
) -> bool:
    """Return whether a gain traversal runs from the back face to the front.

    In the simple multipass model there is only one shared gain marker. Its
    direction describes the first traversal, so ``alternate_direction`` must
    still be applied on subsequent passes. In an explicit optical-bench path,
    every gain marker is a distinct physical hit and an explicit direction is
    therefore authoritative for that hit.
    """

    source_direction = str(seed_direction)
    if source_direction not in {"forward", "backward"}:
        raise ValueError("seed.direction must be forward or backward")

    requested = str(marker_direction)
    if requested not in {"auto", "forward", "backward"}:
        raise ValueError("gain_medium.direction must be auto, forward, or backward")

    if explicit_gain_hit and requested in {"forward", "backward"}:
        return requested == "backward"

    first_pass_backward = (
        requested == "backward"
        if requested in {"forward", "backward"}
        else source_direction == "backward"
    )
    return first_pass_backward ^ bool(alternate_direction and pass_number % 2)
