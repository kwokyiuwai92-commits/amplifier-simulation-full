"""Small, dependency-free helpers for nonlinear fixed-point acceleration.

The amplifier has several expensive maps whose state is much larger than the
number of useful quasi-Newton history vectors.  Anderson acceleration is a good
fit because it only needs previous map evaluations and never forms a Jacobian.
This implementation is deliberately conservative: it limits history memory,
requires a predicted residual reduction, bounds the extrapolation, and restarts
after an accelerated step makes the next measured residual substantially worse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AndersonStep:
    """Result of one proposed fixed-point update."""

    state: np.ndarray
    accepted: bool
    reason: str
    raw_residual_rms: float
    predicted_residual_ratio: float | None
    history_depth: int
    restarted: bool


class SafeguardedAnderson:
    """Low-memory type-II Anderson accelerator with Picard fallback."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.method = str(cfg.get("method", "anderson"))
        self.enabled = self.method == "anderson"
        self.requested_depth = max(1, int(cfg.get("history_depth", 4)))
        self.start_iteration = max(2, int(cfg.get("start_iteration", 3)))
        self.damping = float(cfg.get("damping", 0.8))
        self.regularization = float(cfg.get("regularization", 1e-10))
        self.maximum_predicted_ratio = float(
            cfg.get("maximum_predicted_residual_ratio", 0.99)
        )
        self.maximum_step_factor = float(cfg.get("maximum_step_factor", 5.0))
        self.restart_growth_factor = float(
            cfg.get("restart_residual_growth_factor", 2.0)
        )
        self.maximum_history_bytes = int(
            float(cfg.get("maximum_history_memory_MB", 128.0)) * 1024.0 * 1024.0
        )
        self._delta_state: list[np.ndarray] = []
        self._delta_residual: list[np.ndarray] = []
        self._previous_state: np.ndarray | None = None
        self._previous_residual: np.ndarray | None = None
        self._previous_residual_rms: float | None = None
        self._previous_step_accelerated = False
        self._stats: dict[str, Any] = {
            "method": self.method,
            "enabled": self.enabled,
            "attempted_steps": 0,
            "accepted_steps": 0,
            "rejected_steps": 0,
            "restart_count": 0,
            "memory_limited": False,
            "effective_history_depth": 0,
            "maximum_history_depth_used": 0,
            "last_reason": "not_started",
        }

    @staticmethod
    def _rms(values: np.ndarray) -> float:
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(array * array)))

    @staticmethod
    def _dot(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.vdot(left.ravel(), right.ravel()).real)

    def _clear_history(self) -> None:
        self._delta_state.clear()
        self._delta_residual.clear()

    def reset(self) -> None:
        """Discard numerical history while retaining cumulative diagnostics."""

        self._clear_history()
        self._previous_state = None
        self._previous_residual = None
        self._previous_residual_rms = None
        self._previous_step_accelerated = False

    def _effective_depth(self, state: np.ndarray) -> int:
        pair_bytes = max(2 * int(state.nbytes), 1)
        # One pair is reserved for the previous state/residual used to form the
        # next difference.  The remaining pairs are Anderson delta history.
        memory_depth = max(self.maximum_history_bytes // pair_bytes - 1, 0)
        depth = min(self.requested_depth, int(memory_depth))
        self._stats["memory_limited"] = depth < self.requested_depth
        self._stats["effective_history_depth"] = max(depth, 0)
        return max(depth, 0)

    def propose(
        self,
        iteration: int,
        state: np.ndarray,
        mapped_state: np.ndarray,
    ) -> AndersonStep:
        """Return an accelerated state or the unmodified Picard update.

        ``state`` and ``mapped_state`` should already be nondimensionalized so
        Euclidean residuals give comparable weight to all components.
        """

        current = np.asarray(state, dtype=float)
        mapped = np.asarray(mapped_state, dtype=float)
        if current.shape != mapped.shape:
            raise ValueError("Anderson state and mapped_state must have the same shape")
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(mapped)):
            self._stats["last_reason"] = "nonfinite_map"
            self._previous_step_accelerated = False
            return AndersonStep(
                mapped.copy(), False, "nonfinite_map", float("inf"), None, 0, False
            )

        residual = mapped - current
        residual_rms = self._rms(residual)
        restarted = False
        if (
            self._previous_step_accelerated
            and self._previous_residual_rms is not None
            and residual_rms
            > self.restart_growth_factor
            * max(self._previous_residual_rms, np.finfo(float).tiny)
        ):
            self._clear_history()
            self._previous_state = None
            self._previous_residual = None
            self._stats["restart_count"] += 1
            restarted = True

        if self.enabled:
            depth = self._effective_depth(current)
        else:
            depth = 0
            self._stats["effective_history_depth"] = 0
        if (
            self._previous_state is not None
            and self._previous_residual is not None
            and depth > 0
        ):
            self._delta_state.append(current - self._previous_state)
            self._delta_residual.append(residual - self._previous_residual)
            if len(self._delta_state) > depth:
                self._delta_state.pop(0)
                self._delta_residual.pop(0)

        if depth > 0:
            self._previous_state = current.copy()
            self._previous_residual = residual.copy()
        else:
            self._previous_state = None
            self._previous_residual = None
        self._previous_residual_rms = residual_rms

        reason = "disabled" if not self.enabled else "insufficient_history"
        if depth == 0 and self.enabled:
            reason = "history_memory_limit"
        if (
            not self.enabled
            or iteration < self.start_iteration
            or not self._delta_residual
            or depth == 0
            or residual_rms <= np.finfo(float).tiny
        ):
            self._previous_step_accelerated = False
            self._stats["last_reason"] = reason
            return AndersonStep(
                mapped.copy(), False, reason, residual_rms, None,
                len(self._delta_residual), restarted,
            )

        self._stats["attempted_steps"] += 1
        history_count = len(self._delta_residual)
        gram = np.empty((history_count, history_count), dtype=float)
        right = np.empty(history_count, dtype=float)
        for row in range(history_count):
            right[row] = self._dot(self._delta_residual[row], residual)
            for column in range(row, history_count):
                value = self._dot(
                    self._delta_residual[row], self._delta_residual[column]
                )
                gram[row, column] = value
                gram[column, row] = value
        gram_scale = max(
            float(np.trace(gram)) / max(history_count, 1),
            np.finfo(float).tiny,
        )
        system = gram + self.regularization * gram_scale * np.eye(history_count)
        try:
            coefficients = np.linalg.solve(system, right)
        except np.linalg.LinAlgError:
            coefficients, *_ = np.linalg.lstsq(system, right, rcond=None)
        if not np.all(np.isfinite(coefficients)):
            self._stats["rejected_steps"] += 1
            self._stats["last_reason"] = "nonfinite_coefficients"
            self._previous_step_accelerated = False
            return AndersonStep(
                mapped.copy(), False, "nonfinite_coefficients", residual_rms,
                None, history_count, restarted,
            )

        predicted = residual.copy()
        correction = np.zeros_like(mapped)
        for coefficient, delta_state, delta_residual in zip(
            coefficients,
            self._delta_state,
            self._delta_residual,
            strict=True,
        ):
            predicted -= coefficient * delta_residual
            correction += coefficient * (delta_state + delta_residual)
        predicted_ratio = self._rms(predicted) / max(
            residual_rms, np.finfo(float).tiny
        )
        correction_ratio = self._rms(correction) / max(
            residual_rms, np.finfo(float).tiny
        )
        if predicted_ratio > self.maximum_predicted_ratio:
            reason = "insufficient_predicted_improvement"
        elif correction_ratio > self.maximum_step_factor:
            reason = "extrapolation_too_large"
        else:
            candidate = mapped - self.damping * correction
            if np.all(np.isfinite(candidate)):
                self._stats["accepted_steps"] += 1
                self._stats["maximum_history_depth_used"] = max(
                    int(self._stats["maximum_history_depth_used"]), history_count
                )
                self._stats["last_reason"] = "accepted"
                self._previous_step_accelerated = True
                return AndersonStep(
                    candidate,
                    True,
                    "accepted",
                    residual_rms,
                    predicted_ratio,
                    history_count,
                    restarted,
                )
            reason = "nonfinite_candidate"

        self._stats["rejected_steps"] += 1
        self._stats["last_reason"] = reason
        self._previous_step_accelerated = False
        return AndersonStep(
            mapped.copy(), False, reason, residual_rms, predicted_ratio,
            history_count, restarted,
        )

    def reject_last_step(self, reason: str) -> None:
        """Mark an accepted proposal as rejected by an external constraint."""

        if self._previous_step_accelerated:
            self._stats["accepted_steps"] = max(
                int(self._stats["accepted_steps"]) - 1, 0
            )
            self._stats["rejected_steps"] += 1
        self._stats["last_reason"] = str(reason)
        self._previous_step_accelerated = False

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._stats)
