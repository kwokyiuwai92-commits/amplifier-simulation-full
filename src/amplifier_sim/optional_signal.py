"""Failure isolation for optional signal solvers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import OPTIONAL_MODEL_FAILURES
from .signal import SignalPropagationResult


def optional_fallback_origin(result: SignalPropagationResult) -> str | None:
    """Return whether an optional fallback was selected initially or at runtime."""

    def normalized_origin(value: object) -> str | None:
        if value is None:
            return None
        origin = str(value)
        # Normalize the short-lived pre-schema spelling retained in old
        # in-memory diagnostics; portable results expose only the two-state
        # initialization/runtime contract.
        return "initialization" if origin == "initialization_probe" else origin

    cpa = result.cpa_diagnostics
    for diagnostics in (cpa, result.gain_transport):
        if isinstance(diagnostics, dict) and bool(diagnostics.get("fallback")):
            return normalized_origin(diagnostics.get("fallback_origin"))
    transient_gain = (
        cpa.get("transient_gain") if isinstance(cpa, dict) else None
    )
    if (
        isinstance(transient_gain, dict)
        and str(transient_gain.get("status", "")).startswith("fallback")
    ):
        # Transient-gain construction is retried at the start of each Seed
        # call, so a fallback present in a completed result is a per-call
        # runtime change even if the first import attempt failed earlier.
        return normalized_origin(
            transient_gain.get("fallback_origin", "runtime")
        )
    return None


class ResilientSignalPropagator:
    """Restore state and fall back for one typed optional-solver failure."""

    def __init__(
        self,
        primary: Any | None,
        fallback: Any,
        initial_failure: str | None = None,
        *,
        feature_name: str = "coherent_cpa",
        fallback_model: str = "spectral_time_fallback",
        diagnostic_target: str = "cpa",
    ):
        self.primary = primary
        self.fallback = fallback
        self.disabled_reason = initial_failure
        self.feature_name = feature_name
        self.fallback_model = fallback_model
        self.diagnostic_target = diagnostic_target

    def _active_solver(self) -> Any:
        """Return the backend that will receive the next propagation call."""

        if self.primary is not None and self.disabled_reason is None:
            return self.primary
        return self.fallback

    def resolved_intrapulse_population_time_s(self) -> float:
        """Forward the active backend's ion-population clock contract."""

        getter = getattr(
            self._active_solver(),
            "resolved_intrapulse_population_time_s",
            None,
        )
        return 0.0 if getter is None else float(getter())

    def resolved_intrapulse_free_carrier_time_s(self) -> float:
        """Forward the active backend's coherent carrier clock contract."""

        getter = getattr(
            self._active_solver(),
            "resolved_intrapulse_free_carrier_time_s",
            None,
        )
        return 0.0 if getter is None else float(getter())

    def periodic_free_carrier_state_enabled(self) -> bool:
        """Report whether the active backend can carry the exact carrier state."""

        active = self._active_solver()
        nested = getattr(active, "periodic_free_carrier_state_enabled", None)
        if nested is not None:
            return bool(nested())
        return bool(
            getattr(active, "resolved_intrapulse_free_carrier_time_s", None)
            is not None
            and getattr(active, "set_initial_free_carrier_density", None) is not None
        )

    def set_photoelastic_tensor(
        self, fields: dict[str, np.ndarray] | None
    ) -> None:
        for solver in (self.primary, self.fallback):
            setter = getattr(solver, "set_photoelastic_tensor", None)
            if setter is not None:
                setter(fields)

    def set_endface_surface_sag(
        self,
        front_surface_sag_m: np.ndarray | None,
        back_surface_sag_m: np.ndarray | None,
    ) -> None:
        for solver in (self.primary, self.fallback):
            setter = getattr(solver, "set_endface_surface_sag", None)
            if setter is not None:
                setter(front_surface_sag_m, back_surface_sag_m)

    def set_initial_free_carrier_density(
        self, density_m3: np.ndarray | None
    ) -> None:
        for solver in (self.primary, self.fallback):
            setter = getattr(solver, "set_initial_free_carrier_density", None)
            if setter is not None:
                setter(density_m3)

    def propagate(self, population: np.ndarray, delta_n: np.ndarray) -> SignalPropagationResult:
        runtime_reason: str | None = None
        if self.primary is not None and self.disabled_reason is None:
            backup = population.copy()
            try:
                return self.primary.propagate(population, delta_n)
            except OPTIONAL_MODEL_FAILURES as exc:
                # The optional solver may have depleted part of the caller-owned
                # population before failing.  Restore that same array in place so
                # wrappers and periodic-state callers cannot observe a half-applied
                # primary step.  Keep the primary available: an intermediate
                # fixed-point probe must not permanently disable the requested
                # backend for the final operating point.
                population[...] = backup
                runtime_reason = f"{type(exc).__name__}: {exc}"
        result = self.fallback.propagate(population, delta_n)
        reason = (
            runtime_reason
            or self.disabled_reason
            or "unknown optional-module failure"
        )
        result.warnings.insert(
            0,
            f"{self.feature_name} was disabled and the spectral_time FN fallback completed the run ({reason}).",
        )
        result.model_used = self.fallback_model
        diagnostics = {
            "fallback": True,
            "status": "fallback",
            "reason": reason,
            "fallback_origin": (
                "runtime" if runtime_reason is not None else "initialization"
            ),
        }
        if self.diagnostic_target == "cpa":
            result.cpa_diagnostics = diagnostics
        elif self.diagnostic_target == "gain_transport":
            result.gain_transport = {
                "backend": "frantz_nodvik",
                "requested_backend": self.feature_name,
                **diagnostics,
            }
        return result
