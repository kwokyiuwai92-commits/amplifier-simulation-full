"""State whose lifetime is exactly one reported simulator run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunContext:
    last_ase_result: Any | None = None
    ase_warnings: list[str] = field(default_factory=list)
    spectral_heat_stats: dict[str, Any] = field(
        default_factory=lambda: {"enabled": False}
    )
    event_energy_stats: dict[str, Any] = field(
        default_factory=lambda: {"enabled": False}
    )
    lifetime_heat_stats: dict[str, Any] = field(
        default_factory=lambda: {"enabled": False}
    )
    pump_component_applicability: dict[str, Any] = field(
        default_factory=lambda: {
            "applicable": None,
            "reason": "not_evaluated",
        }
    )
    fallback_events: list[dict[str, Any]] = field(default_factory=list)
    carrier_state: dict[str, Any] = field(default_factory=dict)
    fixed_point_history: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.last_ase_result = None
        self.ase_warnings.clear()
        self.spectral_heat_stats = {"enabled": False}
        self.event_energy_stats = {"enabled": False}
        self.lifetime_heat_stats = {"enabled": False}
        self.pump_component_applicability = {
            "applicable": None,
            "reason": "not_evaluated",
        }
        self.fallback_events.clear()
        self.carrier_state.clear()
        self.fixed_point_history.clear()
