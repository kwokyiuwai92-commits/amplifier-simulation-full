"""JOSA B photon-transport gain updates for ordered Seed time bins.

The schemes follow Springer et al., JOSA B 36, 717-727 (2019), Eqs. (7)-
(8) and (13)-(21).  The optical field is represented by energy fluence per
time/wavelength bin.  This is the characteristic-coordinate equivalent of
the paper's photon-density stencil and lets the existing three-dimensional
transverse propagation remain independent from the local gain integrator.

This module deliberately has no knowledge of Streamlit, optical benches, or
CPA.  A failed optional transport step can therefore be rolled back and
replaced by the legacy Frantz-Nodvik step by its caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import photon_energy
from .errors import ModelApplicabilityError
from .josa_modes import JOSA_MODES, josa_mode_info
from .levels import LevelSystem


@dataclass
class JosaTransportStats:
    calls: int = 0
    temporal_bin_updates: int = 0
    limited_population_updates: int = 0
    maximum_abs_gain_step: float = 0.0
    maximum_relative_population_change_per_bin: float = 0.0
    maximum_population_conservation_error_relative: float = 0.0


class JosaTransportOperator:
    """Apply one active-medium z step to ordered time/wavelength fluences."""

    def __init__(
        self,
        levels: LevelSystem,
        levels_cfg: dict[str, Any],
        material: dict[str, Any],
        physics: dict[str, Any],
        config: dict[str, Any],
        wavelengths_m: np.ndarray,
        temporal_coordinate_s: np.ndarray,
        crystal_mask: np.ndarray,
    ):
        self.levels = levels
        self.levels_cfg = levels_cfg
        self.material = material
        self.physics = physics
        self.config = config
        self.mode = str(config.get("mode", "retarded_pc"))
        if self.mode not in JOSA_MODES:
            raise ValueError(f"Unsupported JOSA transport mode {self.mode!r}")
        if levels_cfg.get("gain_override_m_inv") is not None:
            raise ModelApplicabilityError(
                "josa_transport needs physical upper/lower populations; direct gain_override_m_inv "
                "is supported by the legacy FN solvers only"
            )
        if not bool(physics.get("gain_saturation", True)):
            raise ModelApplicabilityError(
                "josa_transport requires gain_saturation=true because photon transport and population depletion are coupled"
            )
        mode_info = josa_mode_info(self.mode)
        self.tier = str(mode_info["tier"])
        self.scheme = str(mode_info["scheme"])
        self.coordinate_mode = str(mode_info["coordinate_mode"])
        self.include_relaxation = bool(config.get("include_relaxation_during_pulse", False))
        self.recommended_gain_step = float(config.get("recommended_maximum_gain_step", 0.05))
        if not math.isfinite(self.recommended_gain_step) or self.recommended_gain_step <= 0.0:
            raise ValueError("recommended_maximum_gain_step must be positive and finite")
        self.wavelengths_m = np.asarray(wavelengths_m, dtype=float)
        self.temporal_coordinate_s = np.asarray(temporal_coordinate_s, dtype=float)
        self.crystal_mask = np.asarray(crystal_mask, dtype=float)
        self.sigma_abs, self.sigma_em = levels.cross_sections("signal", self.wavelengths_m)
        self.sigma_sum = self.sigma_abs + self.sigma_em
        self.photon_energies = np.asarray(photon_energy(self.wavelengths_m), dtype=float)
        self.stats = JosaTransportStats()
        self.warnings: list[str] = []
        self.last_passive_loss_fluence_J_m2: np.ndarray | None = None

    def begin_propagation(self) -> None:
        """Reset diagnostics that belong to one reported Seed pulse.

        The same operator is reused by thermal and periodic fixed-point probes.
        Carrying maxima and call counts across those hidden solves makes the
        final result describe solver history instead of the returned pulse.
        """

        self.stats = JosaTransportStats()
        self.warnings = []
        self.last_passive_loss_fluence_J_m2 = None

    def apply(
        self,
        field: np.ndarray,
        population: np.ndarray,
        step_length_m: float,
        z_index: int | None = None,
        cell_length_m: float | None = None,
    ) -> np.ndarray:
        """Return the amplified field and update ``population`` in place.

        ``field`` has shape ``(Nt, Nlambda, Ny, Nx)`` and stores a complex
        amplitude whose squared magnitude is bin fluence in J/m2.
        ``population`` has shape ``(Nlevels, Ny, Nx)``.
        """

        step = float(step_length_m)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("JOSA transport step length must be positive and finite")
        cell_length = step if cell_length_m is None else float(cell_length_m)
        if not math.isfinite(cell_length) or cell_length <= 0.0:
            raise ValueError("JOSA physical cell length must be positive and finite")
        if field.ndim != 4 or field.shape[1] != len(self.wavelengths_m):
            raise ValueError("JOSA field must have shape (Nt, Nlambda, Ny, Nx)")
        if population.ndim != 3 or population.shape[1:] != field.shape[2:]:
            raise ValueError("JOSA population and optical transverse shapes do not match")
        if field.shape[0] != len(self.temporal_coordinate_s):
            raise ValueError("JOSA temporal coordinates do not match the field time bins")

        result = field.copy()
        passive_loss_fluence = np.zeros(population.shape[-2:], dtype=float)
        self.sigma_abs, self.sigma_em = self.levels.local_cross_sections(
            "signal", self.wavelengths_m, population.shape[-2:], z_index
        )
        self.sigma_sum = self.sigma_abs + self.sigma_em
        self.stats.calls += 1
        passive_half = np.exp(
            -0.5
            * float(self.material.get("signal_passive_loss_m_inv", 0.0))
            * step
            * self.crystal_mask
        )
        for time_index in range(field.shape[0]):
            fluence = np.abs(result[time_index]) ** 2
            active_input = fluence * passive_half[None, ...]
            population_before = population.copy()
            total_before = np.sum(population_before, axis=0)
            gain_start = self._active_gain(population_before, z_index)
            self.stats.maximum_abs_gain_step = max(
                self.stats.maximum_abs_gain_step,
                float(np.max(np.abs(gain_start * step))),
            )

            if self.scheme == "explicit":
                proposed = active_input + step * gain_start * active_input
            else:
                predictor = np.maximum(
                    active_input + step * gain_start * active_input, 0.0
                )
                predicted_population, _ = self._population_after_fluence_change(
                    population_before,
                    active_input,
                    predictor,
                    cell_length,
                    step,
                    commit=False,
                    z_index=z_index,
                )
                gain_end = self._active_gain(predicted_population, z_index)
                proposed = active_input + 0.5 * step * (
                    gain_start * active_input + gain_end * predictor
                )

            proposed = np.maximum(proposed, 0.0)
            if bool(self.physics.get("gain_saturation", True)):
                population_after, proposed = self._population_after_fluence_change(
                    population,
                    active_input,
                    proposed,
                    cell_length,
                    step,
                    commit=True,
                    z_index=z_index,
                )
                population[:] = population_after

            output_fluence = np.maximum(
                proposed * passive_half[None, ...], 0.0
            )
            passive_loss_fluence += np.sum(
                np.maximum(fluence - active_input, 0.0)
                + np.maximum(proposed - output_fluence, 0.0),
                axis=0,
            )
            ratio = np.divide(
                output_fluence,
                fluence,
                out=np.zeros_like(output_fluence),
                where=fluence > np.finfo(float).tiny,
            )
            result[time_index] *= np.sqrt(np.maximum(ratio, 0.0))

            if self.include_relaxation and time_index + 1 < field.shape[0]:
                dt = float(
                    self.temporal_coordinate_s[time_index + 1]
                    - self.temporal_coordinate_s[time_index]
                )
                if dt > 0.0:
                    population[:] = self.levels.advance_dark_implicit(
                        population,
                        dt,
                        z_index,
                    )

            population_change = np.max(np.abs(population - population_before), axis=0)
            relative_change = np.divide(
                population_change,
                np.maximum(total_before, np.finfo(float).tiny),
            )
            self.stats.maximum_relative_population_change_per_bin = max(
                self.stats.maximum_relative_population_change_per_bin,
                float(np.max(relative_change)),
            )
            conservation_error = np.abs(np.sum(population, axis=0) - total_before)
            conservation_relative = np.divide(
                conservation_error,
                np.maximum(total_before, np.finfo(float).tiny),
            )
            self.stats.maximum_population_conservation_error_relative = max(
                self.stats.maximum_population_conservation_error_relative,
                float(np.max(conservation_relative)),
            )
            self.stats.temporal_bin_updates += 1

        if self.stats.maximum_abs_gain_step > self.recommended_gain_step:
            warning = (
                "JOSA gain step |g*dz| reached "
                f"{self.stats.maximum_abs_gain_step:.4g}, above the configured recommendation "
                f"{self.recommended_gain_step:.4g}; increase Nz and check convergence."
            )
            if warning not in self.warnings:
                self.warnings.append(warning)
        self.last_passive_loss_fluence_J_m2 = passive_loss_fluence
        return result

    def _active_gain(
        self,
        population: np.ndarray,
        z_index: int | None,
    ) -> np.ndarray:
        gain = self.levels.signal_gain_spectrum_m_inv(
            population,
            self.wavelengths_m,
            z_index=z_index,
        )
        return gain * self.crystal_mask[None, ...]

    def _population_after_fluence_change(
        self,
        population: np.ndarray,
        input_fluence: np.ndarray,
        proposed_fluence: np.ndarray,
        cell_length_m: float,
        interaction_length_m: float,
        *,
        commit: bool,
        z_index: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        events = np.divide(
            proposed_fluence - input_fluence,
            self.photon_energies[:, None, None] * cell_length_m,
        )
        if len(self.levels.signal_transitions) > 1:
            updated = population if commit else population.copy()
            applied_events = np.zeros_like(events)
            # A symmetric half-sweep avoids making a physically arbitrary
            # wavelength ordering dominate a shared, competing reservoir.
            for order in (
                range(len(self.wavelengths_m)),
                range(len(self.wavelengths_m) - 1, -1, -1),
            ):
                for wavelength_index in order:
                    next_population, stats = (
                        self.levels.apply_signal_photon_events(
                            updated,
                            0.5 * events[wavelength_index],
                            float(self.wavelengths_m[wavelength_index]),
                            z_index=z_index,
                            input_photon_density_m3=(
                                0.5
                                * input_fluence[wavelength_index]
                                / (
                                    self.photon_energies[wavelength_index]
                                    * cell_length_m
                                )
                            ),
                            interaction_length_m=interaction_length_m,
                            record_diagnostics=commit,
                        )
                    )
                    updated[:] = next_population
                    applied_events[wavelength_index] += np.asarray(
                        stats["applied_photon_events_m3"],
                        dtype=float,
                    )
                    if commit:
                        limited = np.asarray(
                            stats["limiter_scale"], dtype=float
                        ) < 1.0 - 64.0 * np.finfo(float).eps
                        self.stats.limited_population_updates += int(
                            np.count_nonzero(limited)
                        )
            limited_fluence = input_fluence + (
                applied_events
                * self.photon_energies[:, None, None]
                * cell_length_m
            )
            return updated, np.maximum(limited_fluence, 0.0)

        net_events = np.sum(events, axis=0)
        upper = population[self.levels.signal.upper]
        lower = population[self.levels.signal.lower]
        scale = np.ones_like(net_events)
        emitting = net_events > 0.0
        absorbing = net_events < 0.0
        # Gaussian benchmark tails can produce positive subnormal events.
        # Their available-population ratio is physically enormous and is
        # clipped to one below; suppress only that harmless intermediate
        # overflow so a valid low-fluence convergence run stays warning-free.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scale[emitting] = np.minimum(
                1.0,
                np.divide(
                    upper[emitting],
                    net_events[emitting],
                    out=np.ones_like(net_events[emitting]),
                    where=net_events[emitting] > 0.0,
                ),
            )
            scale[absorbing] = np.minimum(
                1.0,
                np.divide(
                    lower[absorbing],
                    -net_events[absorbing],
                    out=np.ones_like(net_events[absorbing]),
                    where=net_events[absorbing] < 0.0,
                ),
            )
        if commit and np.any(scale < 1.0):
            self.stats.limited_population_updates += int(np.count_nonzero(scale < 1.0))
        events *= scale[None, ...]
        net_events = np.sum(events, axis=0)
        updated = population if commit else population.copy()
        updated[self.levels.signal.upper] -= net_events
        updated[self.levels.signal.lower] += net_events
        updated[:] = np.maximum(updated, 0.0)
        limited_fluence = input_fluence + (
            events * self.photon_energies[:, None, None] * cell_length_m
        )
        return updated, np.maximum(limited_fluence, 0.0)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "backend": "josa_transport",
            "mode": self.mode,
            "tier": self.tier,
            "scheme": self.scheme,
            "coordinate_mode": self.coordinate_mode,
            "scheme_order_nominal": 1 if self.scheme == "explicit" else 2,
            "include_relaxation_during_pulse": self.include_relaxation,
            "calls": self.stats.calls,
            "temporal_bin_updates": self.stats.temporal_bin_updates,
            "limited_population_updates": self.stats.limited_population_updates,
            "maximum_abs_gain_step": self.stats.maximum_abs_gain_step,
            "recommended_maximum_gain_step": self.recommended_gain_step,
            "maximum_relative_population_change_per_bin": (
                self.stats.maximum_relative_population_change_per_bin
            ),
            "maximum_population_conservation_error_relative": (
                self.stats.maximum_population_conservation_error_relative
            ),
            "reference": "Springer et al., JOSA B 36, 717-727 (2019)",
        }
