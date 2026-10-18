"""Physical flight-time accounting for sequential multipass routes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .constants import LIGHT_SPEED


@dataclass(frozen=True)
class FlightTime:
    delay_s: float
    geometric_path_length_m: float
    phase_optical_path_m: float
    geometric_optical_path_m: float
    group_optical_path_m: float
    explicit_group_delay_s: float
    final_refractive_index: float
    final_group_refractive_index: float


def interpass_dark_time_s(center_delay_s: float, intrapulse_time_s: float) -> float:
    """Subtract time already advanced within a sequential pulse window.

    For equal per-pass windows the end-to-start gap is the center-to-center
    delay minus one window, not the full route delay. Overlapping windows have
    no dark interval; callers must disclose that sequential pulse transport
    cannot resolve the overlap.
    """

    delay, window = float(center_delay_s), float(intrapulse_time_s)
    if not all(math.isfinite(value) and value >= 0.0 for value in (delay, window)):
        raise ValueError("Interpass delay and intrapulse time must be finite and non-negative")
    return max(delay - window, 0.0)


def optical_train_flight_time(
    elements: list[dict[str, Any]],
    *,
    initial_refractive_index: float = 1.0,
    initial_group_refractive_index: float | None = None,
    extra_delay_s: float = 0.0,
) -> FlightTime:
    """Return elapsed time for path lengths explicitly present in a train.

    Thin elements contribute no unknown physical thickness.  A non-negative
    ``group_delay_s`` can be attached to an element, and ``extra_delay_s`` can
    represent relay/cavity length omitted from the sequential geometry.
    """

    current_n = float(initial_refractive_index)
    if not math.isfinite(current_n) or current_n <= 0.0:
        raise ValueError("initial_refractive_index must be positive and finite")
    current_group_n = float(
        current_n
        if initial_group_refractive_index is None
        else initial_group_refractive_index
    )
    if not math.isfinite(current_group_n) or current_group_n <= 0.0:
        raise ValueError(
            "initial_group_refractive_index must be positive and finite"
        )
    geometric_path = 0.0
    phase_path = 0.0
    group_path = 0.0
    explicit_delay = float(extra_delay_s)
    if not math.isfinite(explicit_delay) or explicit_delay < 0.0:
        raise ValueError("extra_delay_s must be finite and non-negative")

    for element in elements:
        raw_distance_before = element.get("distance_before_m", 0.0)
        distance_before = float(
            0.0 if raw_distance_before is None else raw_distance_before
        )
        if distance_before < 0.0 or not math.isfinite(distance_before):
            raise ValueError("distance_before_m must be finite and non-negative")
        phase_path += current_n * distance_before
        group_path += current_group_n * distance_before
        geometric_path += distance_before

        raw_group_delay = element.get("group_delay_s", 0.0)
        group_delay = float(
            0.0 if raw_group_delay is None else raw_group_delay
        )
        if group_delay < 0.0 or not math.isfinite(group_delay):
            raise ValueError("optical element group_delay_s must be finite and non-negative")
        explicit_delay += group_delay

        kind = str(element.get("type", ""))
        if kind == "free_space":
            length = float(element.get("length_m", 0.0))
            if length < 0.0 or not math.isfinite(length):
                raise ValueError("free-space length_m must be finite and non-negative")
            phase_path += current_n * length
            group_path += current_group_n * length
            geometric_path += length
        elif kind == "thick_lens":
            lens_n = float(element["refractive_index"])
            lens_group_n = float(
                element.get("group_refractive_index", lens_n)
            )
            thickness = float(element["thickness_m"])
            if (
                lens_n <= 0.0
                or lens_group_n <= 0.0
                or not math.isfinite(lens_n)
                or not math.isfinite(lens_group_n)
                or not math.isfinite(thickness)
                or thickness < 0.0
            ):
                raise ValueError("thick-lens index/length is non-physical")
            phase_path += lens_n * thickness
            group_path += lens_group_n * thickness
            geometric_path += thickness
            surrounding_group_n = current_group_n
            current_n = float(element.get("n_after", current_n))
            current_group_n = float(
                element.get(
                    "group_refractive_index_after",
                    current_n if "n_after" in element else surrounding_group_n,
                )
            )
        elif kind == "interface":
            current_n = float(element["n_after"])
            current_group_n = float(
                element.get("group_refractive_index_after", current_n)
            )
        elif kind == "double_pass_lens_relay":
            geometric_length = 2.0 * (
                float(element["object_to_first_lens_m"])
                + float(element["lens_separation_m"])
                + float(element["second_lens_to_mirror_m"])
            )
            phase_path += current_n * geometric_length
            group_path += current_group_n * geometric_length
            geometric_path += geometric_length

        if not math.isfinite(current_n) or current_n <= 0.0:
            raise ValueError("optical element final refractive index must be positive and finite")
        if not math.isfinite(current_group_n) or current_group_n <= 0.0:
            raise ValueError(
                "optical element final group refractive index must be positive and finite"
            )

    return FlightTime(
        delay_s=group_path / LIGHT_SPEED + explicit_delay,
        geometric_path_length_m=geometric_path,
        phase_optical_path_m=phase_path,
        # Compatibility alias retained for existing result consumers.  The
        # historical field contains phase optical path, not geometric length.
        geometric_optical_path_m=phase_path,
        group_optical_path_m=group_path,
        explicit_group_delay_s=explicit_delay,
        final_refractive_index=current_n,
        final_group_refractive_index=current_group_n,
    )
