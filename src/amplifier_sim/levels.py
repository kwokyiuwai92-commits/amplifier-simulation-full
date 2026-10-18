"""Preset and arbitrary N-level population models."""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .spectral_volume import SpectralIntensityVolume, spectral_intensity_blocks

from .profiling import profile_operation, profiled_operation

from .constants import LIGHT_SPEED, photon_energy
from .material_response import SpectroscopicResponse
from .spectral import CrossSectionCurve
from .small_matrix_kernel import (
    NUMBA_AVAILABLE as SMALL_MATRIX_NUMBA_AVAILABLE,
    numba_kernel_compiled_in_process as small_matrix_kernel_compiled,
    solve_batched_small_systems,
)


@dataclass(frozen=True)
class OpticalTransition:
    lower: int
    upper: int
    sigma_abs_m2: float
    sigma_em_m2: float
    name: str = ""
    field: str = "signal"
    quantum_efficiency: float = 1.0
    cross_section_spectrum: dict[str, Any] | None = None
    center_wavelength_m: float | None = None
    homogeneous_fwhm_Hz: float | None = None
    line_shape: str = "none"
    absorption_scale: float = 1.0
    emission_scale: float = 1.0
    ensemble: str | None = None


@dataclass(frozen=True)
class Decay:
    source: int
    target: int
    rate_s: float
    scales_with_upper_lifetime: bool = False


class LevelSystem:
    """Linear optical/decay network plus optional nonlinear stoichiometric steps."""

    def __init__(
        self,
        names: list[str],
        pump: OpticalTransition,
        signal: OpticalTransition,
        decays: list[Decay],
        nonlinear: list[dict[str, Any]],
        cfg: dict[str, Any],
        material: dict[str, Any] | None = None,
        *,
        pump_transitions: list[OpticalTransition] | None = None,
        signal_transitions: list[OpticalTransition] | None = None,
        ensemble_contract: dict[str, Any] | None = None,
    ):
        self.names = names
        self.pump = pump
        self.signal = signal
        self.pump_transitions = list(pump_transitions or [pump])
        self.signal_transitions = list(signal_transitions or [signal])
        self.decays = decays
        self.nonlinear = nonlinear
        self.cfg = cfg
        linear_solver = cfg.get("linear_solver", {})
        self.small_matrix_kernel_requested = str(
            linear_solver.get("small_matrix_kernel", "auto")
        )
        if self.small_matrix_kernel_requested not in {
            "auto",
            "numpy",
            "numba",
        }:
            raise ValueError(
                "linear_solver.small_matrix_kernel must be auto, numpy, or numba"
            )
        if (
            self.small_matrix_kernel_requested == "numba"
            and not SMALL_MATRIX_NUMBA_AVAILABLE
        ):
            raise ValueError(
                "linear_solver.small_matrix_kernel=numba requires the performance extra"
            )
        self.small_matrix_numba_minimum_systems = int(
            linear_solver.get("numba_minimum_systems", 4096)
        )
        self.small_matrix_maximum_numba_order = int(
            linear_solver.get("maximum_numba_order", 8)
        )
        self.reset_implicit_linear_solver_diagnostics()
        self.pump_curve = CrossSectionCurve.from_config(cfg.get("pump_cross_section_spectrum"))
        self.signal_curve = CrossSectionCurve.from_config(cfg.get("signal_cross_section_spectrum"))
        self._transition_curves = {
            id(transition): CrossSectionCurve.from_config(
                transition.cross_section_spectrum
            )
            for transition in (
                *self.pump_transitions,
                *self.signal_transitions,
            )
            if transition.cross_section_spectrum
        }
        self.pump_quantum_efficiency = float(cfg.get("pump_quantum_efficiency", 1.0))
        self.ensemble_contract = dict(ensemble_contract or {"enabled": False})
        self.response = SpectroscopicResponse.from_config(material, cfg)
        self._absolute_cross_section_keys = {
            key for key, law in self.response.cfg.get("cross_sections", {}).items()
            if self.response.enabled and law.get("model") == "absolute_temperature_wavelength_table"
        }
        self._static_cross_section_cache: OrderedDict[
            tuple[Any, ...], tuple[np.ndarray, np.ndarray, int]
        ] = OrderedDict()
        self._static_cross_section_cache_maximum_entries = 512
        self._static_cross_section_cache_maximum_bytes = 16 * 1024**2
        self._static_cross_section_cache_bytes = 0
        self.reset_static_cross_section_cache_diagnostics()
        self.initial_population_diagnostics: dict[str, Any] = {
            "configured_fraction_sum": None,
            "normalization_correction": 0.0,
            "normalization_applied": False,
            "absolute_sum_tolerance": 1e-12,
        }
        self.decay_contract_diagnostics = self._decay_contract_diagnostics()
        self.steady_state_diagnostics: dict[str, Any] = {
            "solver": "not_run",
            "conservation_component_count": len(
                self.conservation_components()
            ),
        }
        self.reset_nonlinear_diagnostics()
        self.reset_population_correction_diagnostics()
        self.reset_signal_event_diagnostics()

    def reset_implicit_linear_solver_diagnostics(self) -> None:
        self.implicit_linear_solver_diagnostics: dict[str, Any] = {
            "requested": self.small_matrix_kernel_requested,
            "numba_available": SMALL_MATRIX_NUMBA_AVAILABLE,
            "numba_minimum_systems": (
                self.small_matrix_numba_minimum_systems
            ),
            "maximum_numba_order": self.small_matrix_maximum_numba_order,
            "numpy_calls": 0,
            "numba_calls": 0,
            "numba_cold_start_calls": 0,
            "numba_failed_batches_falling_back_to_numpy": 0,
            "maximum_system_count": 0,
            "maximum_order": 0,
            "selected": "not_run",
        }

    def _solve_implicit_systems(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
    ) -> np.ndarray:
        order = int(lhs.shape[-1])
        system_count = int(np.prod(lhs.shape[:-2], dtype=np.int64))
        diagnostics = self.implicit_linear_solver_diagnostics
        diagnostics["maximum_system_count"] = max(
            int(diagnostics["maximum_system_count"]), system_count
        )
        diagnostics["maximum_order"] = max(
            int(diagnostics["maximum_order"]), order
        )
        use_numba = bool(
            self.small_matrix_kernel_requested != "numpy"
            and SMALL_MATRIX_NUMBA_AVAILABLE
            and order <= self.small_matrix_maximum_numba_order
            and (
                self.small_matrix_kernel_requested == "numba"
                or system_count >= self.small_matrix_numba_minimum_systems
            )
        )
        if use_numba:
            cold_start = not small_matrix_kernel_compiled()
            with profile_operation(
                "population.implicit_small_matrix_solve",
                array=lhs,
            ):
                advanced, failed = solve_batched_small_systems(lhs, rhs)
            if cold_start:
                diagnostics["numba_cold_start_calls"] = int(
                    diagnostics["numba_cold_start_calls"]
                ) + 1
            diagnostics["numba_calls"] = int(diagnostics["numba_calls"]) + 1
            if not failed:
                diagnostics["selected"] = "numba_parallel_small_matrix"
                return advanced
            diagnostics[
                "numba_failed_batches_falling_back_to_numpy"
            ] = int(
                diagnostics[
                    "numba_failed_batches_falling_back_to_numpy"
                ]
            ) + 1
        with profile_operation(
            "population.implicit_numpy_solve",
            array=lhs,
        ):
            advanced = np.linalg.solve(lhs, rhs[..., None])[..., 0]
        diagnostics["numpy_calls"] = int(diagnostics["numpy_calls"]) + 1
        diagnostics["selected"] = (
            "numpy_fallback_after_numba_failure" if use_numba else "numpy"
        )
        return advanced

    def reset_static_cross_section_cache_diagnostics(self) -> None:
        self._static_cross_section_cache_hits = 0
        self._static_cross_section_cache_misses = 0
        self._static_cross_section_cache_evictions = 0

    def static_cross_section_cache_diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "entries": len(self._static_cross_section_cache),
            "bytes": self._static_cross_section_cache_bytes,
            "maximum_entries": self._static_cross_section_cache_maximum_entries,
            "maximum_bytes": self._static_cross_section_cache_maximum_bytes,
            "hits": self._static_cross_section_cache_hits,
            "misses": self._static_cross_section_cache_misses,
            "evictions": self._static_cross_section_cache_evictions,
            "scope": (
                "static transition table and homogeneous line shape only; "
                "temperature/density response remains live"
            ),
        }

    def reset_nonlinear_diagnostics(self) -> None:
        self.nonlinear_diagnostics: dict[str, float | int | str | bool] = {
            "enabled": bool(self.nonlinear),
            "process_coupling": "simultaneous_competing_reactant_limiter",
            "method": str(
                self.cfg.get("nonlinear_solver", {}).get(
                    "method", "adaptive_substeps"
                )
            ),
            "calls": 0,
            "maximum_substeps_used": 0,
            "cap_limited_calls": 0,
            "maximum_estimated_fractional_depletion_step": 0.0,
            "configured_maximum_fractional_depletion_per_substep": float(
                self.cfg.get("nonlinear_solver", {}).get(
                    "maximum_fractional_depletion_per_substep",
                    0.05,
                )
            ),
        }

    def reset_population_correction_diagnostics(self) -> None:
        """Reset audits for non-negativity and ion-conservation projections."""

        self.population_correction_diagnostics: dict[
            str, float | int | str | bool
        ] = {
            "enabled": True,
            "clipping_projection_calls": 0,
            "renormalization_calls": 0,
            "calls_with_negative_clipping": 0,
            "calls_with_renormalization": 0,
            "negative_clipped_cells_total": 0,
            "renormalized_cells_total": 0,
            "maximum_negative_population_m3": 0.0,
            "maximum_negative_population_fraction": 0.0,
            "maximum_conservation_error_before_renormalization_relative": 0.0,
            "maximum_component_inventory_error_before_renormalization_relative": 0.0,
            "maximum_normalization_scale_deviation": 0.0,
            "zero_total_active_cells_total": 0,
            "worst_negative_source": "none",
            "worst_renormalization_source": "none",
        }

    def reset_signal_event_diagnostics(self) -> None:
        """Reset numerical evidence for multi-transition stimulated events."""

        self.signal_event_diagnostics: dict[str, float | int | bool] = {
            "enabled": len(self.signal_transitions) > 1,
            "calls": 0,
            "limited_cell_updates": 0,
            "minimum_limiter_scale": 1.0,
            "maximum_requested_applied_residual_relative": 0.0,
            "maximum_channel_requested_applied_residual_relative": 0.0,
            "maximum_population_change_fraction_per_update": 0.0,
            "packet_count_used": 0,
            "maximum_population_change_per_packet": 0.0,
            "limited_packet_cell_count": 0,
            "packet_population_change_target": float(
                self.cfg.get("multi_transition_solver", {}).get(
                    "maximum_population_change_fraction_per_packet", 0.05
                )
            ),
            "packet_target_met": True,
        }

    def _decay_contract_diagnostics(self) -> dict[str, Any]:
        totals: dict[str, float] = {}
        for decay in self.cfg.get("decays", []):
            fraction = decay.get(
                "branching_fraction", decay.get("branching")
            )
            if fraction is None:
                continue
            source = str(decay["from"])
            totals[source] = totals.get(source, 0.0) + float(fraction)
        return {
            "semantics": (
                "partial_rate_s/partial_lifetime_s are independent channel rates; "
                "total_lifetime_s + branching_fraction uses a shared source lifetime"
            ),
            "branching_fraction_sum_by_source": totals,
            "branching_population_closed": all(
                abs(total - 1.0) <= 1e-12 for total in totals.values()
            ),
        }

    def conservation_components(self) -> list[list[int]]:
        """Return independently conserved level-network components.

        Optical transitions, decays, and nonlinear stoichiometric processes
        move ions only inside an edge-connected component.  A nonlinear
        process is a hyperedge: every level on either side of the reaction
        belongs to the same conserved inventory.  Spectral diffusion joins
        matching levels across ensemble groups only when its configured rate
        is non-zero.
        """

        adjacency = [set() for _ in self.names]

        def connect(first: int, second: int) -> None:
            adjacency[first].add(second)
            adjacency[second].add(first)

        for transition in (*self.pump_transitions, *self.signal_transitions):
            connect(transition.lower, transition.upper)
        for decay in self.decays:
            connect(decay.source, decay.target)
        level_index = {
            name: index for index, name in enumerate(self.names)
        }
        for process in self.nonlinear:
            participants = sorted(
                level_index[name]
                for name in (
                    set(process.get("reactants", {}))
                    | set(process.get("products", {}))
                )
            )
            if participants:
                anchor = participants[0]
                for participant in participants[1:]:
                    connect(anchor, participant)
        if (
            self.ensemble_contract.get("enabled", False)
            and float(
                self.ensemble_contract.get(
                    "spectral_diffusion_rate_s", 0.0
                )
            )
            > 0.0
        ):
            groups = list(self.ensemble_contract["groups"])
            base_count = int(self.ensemble_contract["base_level_count"])
            for level in range(base_count):
                indices = [
                    int(group["level_offset"]) + level for group in groups
                ]
                for first, second in zip(indices[:-1], indices[1:]):
                    connect(first, second)

        components: list[list[int]] = []
        unseen = set(range(len(self.names)))
        while unseen:
            seed = min(unseen)
            stack = [seed]
            component: list[int] = []
            unseen.remove(seed)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in sorted(adjacency[current]):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
            components.append(sorted(component))
        return components

    def conservation_component_fractions(self) -> list[float]:
        """Configured ion inventory assigned to each conserved component."""

        fractions = self.cfg.get("initial_fractions")
        if fractions is None:
            fractions = [1.0] + [0.0] * (len(self.names) - 1)
        values = np.asarray(fractions, dtype=float)
        return [
            float(np.sum(values[component]))
            for component in self.conservation_components()
        ]

    def project_component_inventories(
        self,
        population: np.ndarray,
        target_total: np.ndarray,
    ) -> np.ndarray:
        """Project non-negative populations onto every conserved inventory."""

        values = np.asarray(population, dtype=float).copy()
        target = np.asarray(target_total, dtype=float)
        tiny = np.finfo(float).tiny
        for component, fraction in zip(
            self.conservation_components(),
            self.conservation_component_fractions(),
        ):
            component_target = fraction * target
            component_total = np.sum(values[component], axis=0)
            active = component_target > tiny
            erased = active & (component_total <= tiny)
            if np.any(erased):
                raise ValueError(
                    "Population projection erased every level in an independently "
                    "conserved level-network component in an active cell"
                )
            scale = np.divide(
                component_target,
                component_total,
                out=np.zeros_like(component_target),
                where=component_total > tiny,
            )
            values[component] *= scale[None, ...]
        return values

    def ensemble_inventory_diagnostics(
        self,
        population: np.ndarray,
        density_m3: np.ndarray | float,
    ) -> dict[str, Any]:
        """Audit per-group ion inventory for static (zero-diffusion) ensembles."""

        if not self.ensemble_contract.get("enabled", False):
            return {"enabled": False, "applicable": False}
        diffusion = float(
            self.ensemble_contract.get("spectral_diffusion_rate_s", 0.0)
        )
        applicable = diffusion == 0.0
        density = np.broadcast_to(
            np.asarray(density_m3, dtype=float),
            np.asarray(population).shape[1:],
        )
        tiny = np.finfo(float).tiny
        groups: list[dict[str, Any]] = []
        maximum = 0.0
        base_count = int(self.ensemble_contract["base_level_count"])
        for group in self.ensemble_contract["groups"]:
            offset = int(group["level_offset"])
            actual = np.sum(
                np.asarray(population, dtype=float)[
                    offset : offset + base_count
                ],
                axis=0,
            )
            expected = float(group["fraction"]) * density
            relative = np.divide(
                np.abs(actual - expected),
                np.maximum(expected, tiny),
            )
            local_maximum = float(np.max(relative)) if relative.size else 0.0
            maximum = max(maximum, local_maximum)
            groups.append(
                {
                    "name": str(group["name"]),
                    "configured_fraction": float(group["fraction"]),
                    "maximum_local_inventory_error_relative": local_maximum,
                }
            )
        return {
            "enabled": True,
            "applicable": applicable,
            "spectral_diffusion_rate_s": diffusion,
            "groups": groups,
            "maximum_local_inventory_error_relative": maximum,
            "invariant": (
                "each static ensemble group retains its configured ion inventory"
                if applicable
                else "spectral diffusion permits time-dependent exchange between groups"
            ),
        }

    def spectral_diffusion_energy_exchange_diagnostics(
        self,
        level_energies_J: np.ndarray,
    ) -> dict[str, Any]:
        """Report whether spectral diffusion can exchange energy with a bath.

        The diffusion operator conserves the population of each base level
        across ensemble groups.  It also conserves modeled stored energy only
        when every cloned copy of that base level has the same reference
        energy.  Event-energy accounting currently has no explicit diffusion
        bath, so this diagnostic defines its strict applicability boundary.
        """

        rate = float(
            self.ensemble_contract.get("spectral_diffusion_rate_s", 0.0)
        )
        enabled = bool(
            self.ensemble_contract.get("enabled", False) and rate > 0.0
        )
        energies = np.asarray(level_energies_J, dtype=float)
        if energies.shape != (len(self.names),):
            raise ValueError(
                "spectral-diffusion energy diagnostics require one energy "
                "per expanded level"
            )
        if not enabled:
            return {
                "enabled": False,
                "spectral_diffusion_rate_s": rate,
                "stored_energy_exchange_possible": False,
                "maximum_matching_level_energy_spread_J": 0.0,
            }
        groups = list(self.ensemble_contract["groups"])
        base_count = int(self.ensemble_contract["base_level_count"])
        spreads: dict[str, float] = {}
        for base_level in range(base_count):
            values = np.asarray(
                [
                    energies[int(group["level_offset"]) + base_level]
                    for group in groups
                ],
                dtype=float,
            )
            spreads[str(self.ensemble_contract["base_level_names"][base_level])] = (
                float(np.max(values) - np.min(values))
            )
        maximum = max(spreads.values(), default=0.0)
        scale = max(float(np.max(np.abs(energies))), np.finfo(float).tiny)
        tolerance = 64.0 * np.finfo(float).eps * scale
        return {
            "enabled": True,
            "spectral_diffusion_rate_s": rate,
            "stored_energy_exchange_possible": maximum > tolerance,
            "maximum_matching_level_energy_spread_J": maximum,
            "energy_equality_absolute_tolerance_J": tolerance,
            "matching_level_energy_spread_J": spreads,
        }

    def _clip_nonnegative_and_audit(
        self,
        population: np.ndarray,
        target_total: np.ndarray,
        source: str,
    ) -> np.ndarray:
        """Clip a level-first population array while recording the correction."""

        raw = np.asarray(population, dtype=float)
        target = np.asarray(target_total, dtype=float)
        negative = np.maximum(-raw, 0.0)
        negative_per_cell = np.sum(negative, axis=0)
        active = target > np.finfo(float).tiny
        relative = np.zeros_like(target, dtype=float)
        np.divide(
            negative_per_cell,
            target,
            out=relative,
            where=active,
        )
        maximum_negative = float(np.max(negative)) if negative.size else 0.0
        maximum_fraction = float(np.max(relative)) if relative.size else 0.0
        correction_tolerance = 64.0 * np.finfo(float).eps
        corrected_cells = int(np.count_nonzero(relative > correction_tolerance))

        diagnostics = self.population_correction_diagnostics
        diagnostics["clipping_projection_calls"] = int(
            diagnostics["clipping_projection_calls"]
        ) + 1
        diagnostics["negative_clipped_cells_total"] = int(
            diagnostics["negative_clipped_cells_total"]
        ) + corrected_cells
        diagnostics["calls_with_negative_clipping"] = int(
            diagnostics["calls_with_negative_clipping"]
        ) + int(corrected_cells > 0)
        diagnostics["maximum_negative_population_m3"] = max(
            float(diagnostics["maximum_negative_population_m3"]),
            maximum_negative,
        )
        if maximum_fraction > float(
            diagnostics["maximum_negative_population_fraction"]
        ):
            diagnostics["maximum_negative_population_fraction"] = maximum_fraction
            diagnostics["worst_negative_source"] = source
        return np.maximum(raw, 0.0)

    def _renormalize_and_audit(
        self,
        population: np.ndarray,
        target_total: np.ndarray,
        source: str,
    ) -> np.ndarray:
        """Apply the existing ion-conservation normalization and audit it."""

        values = np.asarray(population, dtype=float)
        target = np.asarray(target_total, dtype=float)
        active = target > np.finfo(float).tiny
        total = np.sum(values, axis=0)
        relative_error = np.zeros_like(target, dtype=float)
        np.divide(
            np.abs(total - target),
            target,
            out=relative_error,
            where=active,
        )
        scale_deviation = np.zeros_like(target, dtype=float)
        component_error = np.zeros_like(target, dtype=float)
        for component, fraction in zip(
            self.conservation_components(),
            self.conservation_component_fractions(),
        ):
            component_total = np.sum(values[component], axis=0)
            component_target = fraction * target
            component_active = component_target > np.finfo(float).tiny
            erased = component_active & (
                component_total <= np.finfo(float).tiny
            )
            zero_count = int(np.count_nonzero(erased))
            if zero_count:
                self.population_correction_diagnostics[
                    "zero_total_active_cells_total"
                ] = int(
                    self.population_correction_diagnostics[
                        "zero_total_active_cells_total"
                    ]
                ) + zero_count
                raise ValueError(
                    "Population projection erased every level in an independently "
                    "conserved level-network component in an active cell"
                )
            local_error = np.divide(
                np.abs(component_total - component_target),
                np.maximum(component_target, np.finfo(float).tiny),
            )
            component_error = np.maximum(component_error, local_error)
            component_scale = np.divide(
                component_target,
                component_total,
                out=np.zeros_like(component_target),
                where=component_total > np.finfo(float).tiny,
            )
            scale_deviation = np.maximum(
                scale_deviation,
                np.where(component_active, np.abs(component_scale - 1.0), 0.0),
            )
            values[component] *= component_scale[None, ...]
        maximum_error = (
            float(np.max(relative_error)) if relative_error.size else 0.0
        )
        maximum_scale_deviation = (
            float(np.max(scale_deviation)) if scale_deviation.size else 0.0
        )
        maximum_component_error = (
            float(np.max(component_error)) if component_error.size else 0.0
        )
        combined_error = np.maximum(relative_error, component_error)
        maximum_combined_error = max(maximum_error, maximum_component_error)
        correction_tolerance = 64.0 * np.finfo(float).eps
        corrected_cells = int(
            np.count_nonzero(combined_error > correction_tolerance)
        )

        diagnostics = self.population_correction_diagnostics
        previous_combined_error = max(
            float(
                diagnostics[
                    "maximum_conservation_error_before_renormalization_relative"
                ]
            ),
            float(
                diagnostics[
                    "maximum_component_inventory_error_before_renormalization_relative"
                ]
            ),
        )
        diagnostics["renormalization_calls"] = int(
            diagnostics["renormalization_calls"]
        ) + 1
        diagnostics["renormalized_cells_total"] = int(
            diagnostics["renormalized_cells_total"]
        ) + corrected_cells
        diagnostics["calls_with_renormalization"] = int(
            diagnostics["calls_with_renormalization"]
        ) + int(corrected_cells > 0)
        if maximum_error > float(
            diagnostics[
                "maximum_conservation_error_before_renormalization_relative"
            ]
        ):
            diagnostics[
                "maximum_conservation_error_before_renormalization_relative"
            ] = maximum_error
        diagnostics["maximum_normalization_scale_deviation"] = max(
            float(diagnostics["maximum_normalization_scale_deviation"]),
            maximum_scale_deviation,
        )
        diagnostics[
            "maximum_component_inventory_error_before_renormalization_relative"
        ] = max(
            float(
                diagnostics[
                    "maximum_component_inventory_error_before_renormalization_relative"
                ]
            ),
            maximum_component_error,
        )
        if maximum_combined_error > previous_combined_error:
            diagnostics["worst_renormalization_source"] = source
        return values

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        material: dict[str, Any] | None = None,
    ) -> "LevelSystem":
        model = cfg["model"]
        pa = float(cfg["pump_absorption_cross_section_m2"])
        pe = float(cfg["pump_emission_cross_section_m2"])
        sa = float(cfg["signal_absorption_cross_section_m2"])
        se = float(cfg["signal_emission_cross_section_m2"])
        tau_u = float(cfg["upper_state_lifetime_s"])
        tau_p = float(cfg["pump_level_lifetime_s"])
        tau_l = float(cfg["lower_level_lifetime_s"])
        if model == "two_level":
            names = ["ground_lower", "upper"]
            pump = OpticalTransition(
                0,
                1,
                pa,
                pe,
                name="pump",
                field="pump",
                quantum_efficiency=float(
                    cfg.get("pump_quantum_efficiency", 1.0)
                ),
            )
            signal = OpticalTransition(
                0, 1, sa, se, name="signal", field="signal"
            )
            decays = [Decay(1, 0, 1.0 / tau_u, True)]
        elif model == "three_level":
            names = ["ground_lower", "upper", "pump"]
            pump = OpticalTransition(
                0,
                2,
                pa,
                pe,
                name="pump",
                field="pump",
                quantum_efficiency=float(
                    cfg.get("pump_quantum_efficiency", 1.0)
                ),
            )
            signal = OpticalTransition(
                0, 1, sa, se, name="signal", field="signal"
            )
            decays = [
                Decay(2, 1, 1.0 / tau_p),
                Decay(1, 0, 1.0 / tau_u, True),
            ]
        elif model == "four_level":
            names = ["ground", "lower", "upper", "pump"]
            pump = OpticalTransition(
                0,
                3,
                pa,
                pe,
                name="pump",
                field="pump",
                quantum_efficiency=float(
                    cfg.get("pump_quantum_efficiency", 1.0)
                ),
            )
            signal = OpticalTransition(
                1, 2, sa, se, name="signal", field="signal"
            )
            decays = [
                Decay(3, 2, 1.0 / tau_p),
                Decay(2, 1, 1.0 / tau_u, True),
                Decay(1, 0, 1.0 / tau_l),
            ]
        else:
            names = list(cfg["levels"])
            index = {name: i for i, name in enumerate(names)}

            def transition(
                data: dict[str, Any],
                *,
                field: str,
                fallback_name: str,
            ) -> OpticalTransition:
                return OpticalTransition(
                    index[data["lower"]],
                    index[data["upper"]],
                    float(data.get("sigma_abs_m2", 0.0)),
                    float(data.get("sigma_em_m2", 0.0)),
                    name=str(data.get("name", fallback_name)),
                    field=field,
                    quantum_efficiency=float(
                        data.get(
                            "quantum_efficiency",
                            cfg.get("pump_quantum_efficiency", 1.0)
                            if field == "pump"
                            else 1.0,
                        )
                    ),
                    cross_section_spectrum=copy.deepcopy(
                        data.get("cross_section_spectrum") or None
                    ),
                    center_wavelength_m=(
                        None
                        if data.get("center_wavelength_m") is None
                        else float(data["center_wavelength_m"])
                    ),
                    homogeneous_fwhm_Hz=(
                        None
                        if data.get("homogeneous_fwhm_Hz") is None
                        else float(data["homogeneous_fwhm_Hz"])
                    ),
                    line_shape=str(data.get("line_shape", "none")),
                    absorption_scale=float(data.get("absorption_scale", 1.0)),
                    emission_scale=float(data.get("emission_scale", 1.0)),
                    ensemble=(
                        None
                        if data.get("ensemble") is None
                        else str(data["ensemble"])
                    ),
                )

            configured_transitions = list(cfg.get("optical_transitions", []))
            if configured_transitions:
                pump_transitions = [
                    transition(
                        data,
                        field="pump",
                        fallback_name=f"pump_{position}",
                    )
                    for position, data in enumerate(configured_transitions)
                    if str(data.get("field", "")) == "pump"
                ]
                signal_transitions = [
                    transition(
                        data,
                        field="signal",
                        fallback_name=f"signal_{position}",
                    )
                    for position, data in enumerate(configured_transitions)
                    if str(data.get("field", "")) == "signal"
                ]
                pump = pump_transitions[0]
                signal = signal_transitions[0]
            else:
                pump = transition(
                    cfg["pump_transition"],
                    field="pump",
                    fallback_name="pump",
                )
                signal = transition(
                    cfg["signal_transition"],
                    field="signal",
                    fallback_name="signal",
                )
                pump_transitions = [pump]
                signal_transitions = [signal]
            decays = []
            for decay in cfg.get("decays", []):
                shared_upper_lifetime_contract = False
                if decay.get("partial_rate_s") is not None:
                    rate = float(decay["partial_rate_s"])
                    lifetime = None
                elif decay.get("partial_lifetime_s") is not None:
                    lifetime = decay["partial_lifetime_s"]
                    rate = 1.0 / float(lifetime)
                elif decay.get("rate_s") not in (None, 0, 0.0):
                    rate = float(decay["rate_s"])
                    lifetime = None
                elif decay.get("lifetime_s") is not None:
                    lifetime = decay["lifetime_s"]
                    rate = float(decay.get("branching", 1.0)) / float(
                        lifetime
                    )
                    shared_upper_lifetime_contract = "branching" in decay
                else:
                    lifetime = decay["total_lifetime_s"]
                    rate = (
                        float(decay["branching_fraction"]) / float(lifetime)
                    )
                    shared_upper_lifetime_contract = True
                source = index[decay["from"]]
                scales_with_upper = bool(
                    decay.get(
                        "scale_with_upper_state_lifetime",
                        source == signal.upper
                        and shared_upper_lifetime_contract,
                    )
                )
                decays.append(
                    Decay(
                        source,
                        index[decay["to"]],
                        rate,
                        scales_with_upper,
                    )
                )
        if model != "custom":
            pump_transitions = [pump]
            signal_transitions = [signal]
        (
            names,
            pump_transitions,
            signal_transitions,
            decays,
            system_cfg,
            ensemble_contract,
        ) = cls._expand_inhomogeneous_ensembles(
            names,
            pump_transitions,
            signal_transitions,
            decays,
            cfg,
        )
        return cls(
            names,
            pump_transitions[0],
            signal_transitions[0],
            decays,
            list(system_cfg.get("nonlinear_processes", [])),
            system_cfg,
            material,
            pump_transitions=pump_transitions,
            signal_transitions=signal_transitions,
            ensemble_contract=ensemble_contract,
        )

    @staticmethod
    def _expand_inhomogeneous_ensembles(
        names: list[str],
        pump_transitions: list[OpticalTransition],
        signal_transitions: list[OpticalTransition],
        decays: list[Decay],
        cfg: dict[str, Any],
    ) -> tuple[
        list[str],
        list[OpticalTransition],
        list[OpticalTransition],
        list[Decay],
        dict[str, Any],
        dict[str, Any],
    ]:
        """Clone a level network into independently saturating spectral groups."""

        ensemble_cfg = cfg.get("inhomogeneous_ensembles", {}) or {}
        if not bool(ensemble_cfg.get("enabled", False)):
            return (
                names,
                pump_transitions,
                signal_transitions,
                decays,
                cfg,
                {"enabled": False},
            )
        groups = list(ensemble_cfg.get("groups", []))
        base_count = len(names)
        expanded_names: list[str] = []
        expanded_pump: list[OpticalTransition] = []
        expanded_signal: list[OpticalTransition] = []
        expanded_decays: list[Decay] = []
        group_contracts: list[dict[str, Any]] = []
        for group_index, group in enumerate(groups):
            group_name = str(group["name"])
            offset = group_index * base_count
            expanded_names.extend(f"{group_name}:{name}" for name in names)
            signal_center = group.get("signal_center_wavelength_m")
            pump_center = group.get("pump_center_wavelength_m")
            linewidth = group.get(
                "homogeneous_fwhm_Hz",
                ensemble_cfg.get("homogeneous_fwhm_Hz"),
            )
            line_shape = str(
                group.get(
                    "line_shape",
                    ensemble_cfg.get("line_shape", "gaussian"),
                )
            )
            for transition_item in pump_transitions:
                expanded_pump.append(
                    replace(
                        transition_item,
                        lower=transition_item.lower + offset,
                        upper=transition_item.upper + offset,
                        name=f"{group_name}:{transition_item.name}",
                        # Resolve the primary channel's legacy spectrum before
                        # cloning: object identity selects only the first clone
                        # in the unexpanded legacy fallback.
                        cross_section_spectrum=copy.deepcopy(
                            transition_item.cross_section_spectrum
                            or (
                                cfg.get("pump_cross_section_spectrum")
                                if transition_item is pump_transitions[0]
                                else None
                            )
                        ),
                        center_wavelength_m=(
                            transition_item.center_wavelength_m
                            if pump_center is None
                            else float(pump_center)
                        ),
                        homogeneous_fwhm_Hz=(
                            transition_item.homogeneous_fwhm_Hz
                            if linewidth is None
                            else float(linewidth)
                        ),
                        line_shape=line_shape,
                        absorption_scale=transition_item.absorption_scale
                        * float(group.get("pump_absorption_scale", 1.0)),
                        emission_scale=transition_item.emission_scale
                        * float(group.get("pump_emission_scale", 1.0)),
                        ensemble=group_name,
                    )
                )
            for transition_item in signal_transitions:
                expanded_signal.append(
                    replace(
                        transition_item,
                        lower=transition_item.lower + offset,
                        upper=transition_item.upper + offset,
                        name=f"{group_name}:{transition_item.name}",
                        cross_section_spectrum=copy.deepcopy(
                            transition_item.cross_section_spectrum
                            or (
                                cfg.get("signal_cross_section_spectrum")
                                if transition_item is signal_transitions[0]
                                else None
                            )
                        ),
                        center_wavelength_m=(
                            transition_item.center_wavelength_m
                            if signal_center is None
                            else float(signal_center)
                        ),
                        homogeneous_fwhm_Hz=(
                            transition_item.homogeneous_fwhm_Hz
                            if linewidth is None
                            else float(linewidth)
                        ),
                        line_shape=line_shape,
                        absorption_scale=transition_item.absorption_scale
                        * float(group.get("signal_absorption_scale", 1.0)),
                        emission_scale=transition_item.emission_scale
                        * float(group.get("signal_emission_scale", 1.0)),
                        ensemble=group_name,
                    )
                )
            expanded_decays.extend(
                replace(
                    decay,
                    source=decay.source + offset,
                    target=decay.target + offset,
                )
                for decay in decays
            )
            group_contracts.append(
                {
                    "name": group_name,
                    "fraction": float(group["fraction"]),
                    "level_offset": offset,
                    "signal_center_wavelength_m": (
                        None
                        if signal_center is None
                        else float(signal_center)
                    ),
                    "pump_center_wavelength_m": (
                        None if pump_center is None else float(pump_center)
                    ),
                }
            )
        system_cfg = copy.deepcopy(cfg)
        base_fractions = cfg.get("initial_fractions")
        if base_fractions is None:
            base_fractions = [1.0] + [0.0] * (base_count - 1)
        system_cfg["initial_fractions"] = [
            float(group["fraction"]) * float(fraction)
            for group in groups
            for fraction in base_fractions
        ]
        level_energies = list(cfg.get("level_energies_J", []))
        if level_energies:
            system_cfg["level_energies_J"] = level_energies * len(groups)
        return (
            expanded_names,
            expanded_pump,
            expanded_signal,
            expanded_decays,
            system_cfg,
            {
                "enabled": True,
                "base_level_names": list(names),
                "base_level_count": base_count,
                "groups": group_contracts,
                "spectral_diffusion_rate_s": float(
                    ensemble_cfg.get("spectral_diffusion_rate_s", 0.0)
                ),
                "model": (
                    "independent_static_subensembles_with_optional_"
                    "spectral_diffusion"
                ),
            },
        )

    def set_environment(
        self,
        temperature_K: np.ndarray | float,
        density_m3: np.ndarray | float,
    ) -> None:
        """Set the environment used by optional local spectroscopy feedback."""

        self.response.set_environment(temperature_K, density_m3)

    def environment_copy(self) -> tuple[np.ndarray | float, np.ndarray | float]:
        return self.response.environment_copy()

    def reference_level_energies_J(
        self, pump_wavelength_m: float, signal_wavelength_m: float
    ) -> tuple[np.ndarray, str]:
        """Return a documented reference-energy assignment for stored-energy audits.

        Explicit custom energies are authoritative.  Built-in ladder models can
        be inferred from their named pump and signal transitions; the four-level
        lower-state energy is referenced to ground because its configured fast
        decay does not provide an independent spectroscopic gap.
        """

        configured = np.asarray(self.cfg.get("level_energies_J", []), dtype=float)
        if configured.size:
            if configured.shape != (len(self.names),):
                raise ValueError(
                    "energy_levels.level_energies_J must match the active level count"
                )
            return configured.copy(), "explicit_user_level_energies"
        model = str(self.cfg.get("model", "custom"))
        groups = (
            list(self.ensemble_contract.get("groups", []))
            if self.ensemble_contract.get("enabled", False)
            else [
                {
                    "signal_center_wavelength_m": None,
                    "pump_center_wavelength_m": None,
                }
            ]
        )
        inferred: list[float] = []
        for group in groups:
            local_pump = float(
                group.get("pump_center_wavelength_m")
                or pump_wavelength_m
            )
            local_signal = float(
                group.get("signal_center_wavelength_m")
                or signal_wavelength_m
            )
            pump_energy = float(photon_energy(local_pump))
            signal_energy = float(photon_energy(local_signal))
            if model == "two_level":
                inferred.extend([0.0, signal_energy])
            elif model == "three_level":
                inferred.extend([0.0, signal_energy, pump_energy])
            elif model == "four_level":
                inferred.extend([0.0, 0.0, signal_energy, pump_energy])
            else:
                raise ValueError(
                    "custom level networks require "
                    "energy_levels.level_energies_J for a stored-energy ledger"
                )
        energies = np.asarray(inferred, dtype=float)
        if energies.shape != (len(self.names),):
            raise ValueError(
                "inferred event-energy level vector does not match the "
                "expanded level network"
            )
        if model == "two_level":
            return energies, (
                (
                    "inferred_per_ensemble_signal_gap"
                    if self.ensemble_contract.get("enabled", False)
                    else "inferred_signal_gap"
                )
                + "; Pump quantum defect is not represented as stored energy"
            )
        if model == "three_level":
            return energies, (
                "inferred_per_ensemble_builtin_three_level_reference"
                if self.ensemble_contract.get("enabled", False)
                else "inferred_builtin_three_level_reference"
            )
        if model == "four_level":
            return energies, (
                (
                    "inferred_per_ensemble_builtin_four_level_reference"
                    if self.ensemble_contract.get("enabled", False)
                    else "inferred_builtin_four_level_reference"
                )
                + "; lower "
                "state referenced to ground"
            )
        raise ValueError(
            "custom level networks require energy_levels.level_energies_J for a stored-energy ledger"
        )

    def initial_population(self, density_m3: float | np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        fractions = self.cfg.get("initial_fractions")
        if fractions is None:
            fractions = [1.0] + [0.0] * (len(self.names) - 1)
        fractions_array = np.asarray(fractions, dtype=float)
        if fractions_array.shape != (len(self.names),):
            raise ValueError(f"initial_fractions must contain {len(self.names)} values")
        fraction_sum = float(np.sum(fractions_array))
        tolerance = 1e-12
        if (
            np.any(~np.isfinite(fractions_array))
            or np.any(fractions_array < 0)
            or abs(fraction_sum - 1.0) > tolerance
        ):
            raise ValueError("initial_fractions must be non-negative and sum to one")
        correction = abs(fraction_sum - 1.0)
        fractions_array = fractions_array / fraction_sum
        self.initial_population_diagnostics = {
            "configured_fraction_sum": fraction_sum,
            "normalization_correction": correction,
            "normalization_applied": correction > 0.0,
            "absolute_sum_tolerance": tolerance,
        }
        density = np.broadcast_to(np.asarray(density_m3, dtype=float), shape)
        if np.any(density < 0.0) or not np.all(np.isfinite(density)):
            raise ValueError("dopant density must be finite and non-negative")
        return fractions_array[:, None, None, None] * density[None, ...]

    @staticmethod
    def _add_transition(matrix: np.ndarray, transition: OpticalTransition, flux: np.ndarray) -> None:
        wa = transition.sigma_abs_m2 * flux
        we = transition.sigma_em_m2 * flux
        lo, up = transition.lower, transition.upper
        matrix[..., up, lo] += wa
        matrix[..., lo, lo] -= wa
        matrix[..., lo, up] += we
        matrix[..., up, up] -= we

    @staticmethod
    def _add_transition_rates(
        matrix: np.ndarray,
        transition: OpticalTransition,
        absorption_rate_s: np.ndarray,
        emission_rate_s: np.ndarray,
    ) -> None:
        lo, up = transition.lower, transition.upper
        matrix[..., up, lo] += absorption_rate_s
        matrix[..., lo, lo] -= absorption_rate_s
        matrix[..., lo, up] += emission_rate_s
        matrix[..., up, up] -= emission_rate_s

    def _add_spectral_diffusion(self, matrix: np.ndarray) -> None:
        """Relax matching levels between inhomogeneous groups conservatively."""

        if not self.ensemble_contract.get("enabled", False):
            return
        rate = float(
            self.ensemble_contract.get("spectral_diffusion_rate_s", 0.0)
        )
        if rate <= 0.0:
            return
        groups = list(self.ensemble_contract["groups"])
        base_count = int(self.ensemble_contract["base_level_count"])
        for source_group, source in enumerate(groups):
            source_offset = int(source["level_offset"])
            for target_group, target in enumerate(groups):
                if source_group == target_group:
                    continue
                target_offset = int(target["level_offset"])
                transfer_rate = rate * float(target["fraction"])
                for level in range(base_count):
                    source_index = source_offset + level
                    target_index = target_offset + level
                    matrix[..., target_index, source_index] += transfer_rate
                    matrix[..., source_index, source_index] -= transfer_rate

    def cross_sections(
        self,
        transition_name: str,
        wavelength_m: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if transition_name == "pump":
            transition, curve = self.pump, self.pump_curve
        elif transition_name == "signal":
            transition, curve = self.signal, self.signal_curve
        else:
            raise ValueError("transition_name must be pump or signal")
        wavelengths = np.asarray(wavelength_m, dtype=float)
        if curve is None:
            return (
                np.full_like(wavelengths, transition.sigma_abs_m2, dtype=float),
                np.full_like(wavelengths, transition.sigma_em_m2, dtype=float),
            )
        return curve.values(
            wavelengths,
            guard=self.response.range_guard,
            name=f"energy_levels.{transition_name}_cross_section_spectrum",
        )

    def _transition_cross_sections(
        self,
        transition: OpticalTransition,
        wavelength_m: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return static cross sections for one named optical channel."""

        wavelengths = np.asarray(wavelength_m, dtype=float)
        curve = self._transition_curves.get(id(transition))
        if curve is None:
            legacy_curve = (
                self.pump_curve
                if transition.field == "pump"
                else self.signal_curve
            )
            use_legacy_curve = (
                transition is self.pump or transition is self.signal
            )
            curve = legacy_curve if use_legacy_curve else None
        contiguous_wavelengths = np.ascontiguousarray(wavelengths)
        cache_key = (
            id(transition),
            contiguous_wavelengths.dtype.str,
            contiguous_wavelengths.shape,
            contiguous_wavelengths.tobytes(),
        )
        cached = self._static_cross_section_cache.get(cache_key)
        curve_name = (
            "energy_levels.optical_transitions."
            f"{transition.name}.cross_section_spectrum"
        )
        if cached is not None:
            self._static_cross_section_cache.move_to_end(cache_key)
            self._static_cross_section_cache_hits += 1
            if curve is not None:
                self.response.range_guard.record(
                    curve_name,
                    wavelengths,
                    float(curve.wavelength_m[0]),
                    float(curve.wavelength_m[-1]),
                )
            return cached[0], cached[1]

        self._static_cross_section_cache_misses += 1
        if curve is None:
            absorption = np.full_like(
                wavelengths, transition.sigma_abs_m2, dtype=float
            )
            emission = np.full_like(
                wavelengths, transition.sigma_em_m2, dtype=float
            )
        else:
            absorption, emission = curve.values(
                wavelengths,
                guard=self.response.range_guard,
                name=curve_name,
            )
        line_multiplier = self._transition_line_shape(
            transition, wavelengths
        )
        result = (
            np.asarray(absorption, dtype=float)
            * transition.absorption_scale
            * line_multiplier,
            np.asarray(emission, dtype=float)
            * transition.emission_scale
            * line_multiplier,
        )
        for values in result:
            values.setflags(write=False)
        entry_bytes = int(
            sum(values.nbytes for values in result)
            + len(cache_key[3])
        )
        self._static_cross_section_cache[cache_key] = (
            result[0],
            result[1],
            entry_bytes,
        )
        self._static_cross_section_cache_bytes += entry_bytes
        while self._static_cross_section_cache and (
            len(self._static_cross_section_cache)
            > self._static_cross_section_cache_maximum_entries
            or self._static_cross_section_cache_bytes
            > self._static_cross_section_cache_maximum_bytes
        ):
            _evicted_key, evicted = self._static_cross_section_cache.popitem(
                last=False
            )
            self._static_cross_section_cache_bytes -= int(evicted[2])
            self._static_cross_section_cache_evictions += 1
        return result

    @staticmethod
    def _transition_line_shape(
        transition: OpticalTransition,
        wavelengths_m: np.ndarray,
    ) -> np.ndarray:
        if (
            transition.line_shape == "none"
            or transition.center_wavelength_m is None
            or transition.homogeneous_fwhm_Hz is None
        ):
            return np.ones_like(wavelengths_m, dtype=float)
        center_frequency = LIGHT_SPEED / float(
            transition.center_wavelength_m
        )
        frequencies = LIGHT_SPEED / np.asarray(wavelengths_m, dtype=float)
        normalized = (frequencies - center_frequency) / float(
            transition.homogeneous_fwhm_Hz
        )
        if transition.line_shape == "gaussian":
            return np.exp(-4.0 * np.log(2.0) * normalized**2)
        if transition.line_shape == "lorentzian":
            return 1.0 / (1.0 + 4.0 * normalized**2)
        raise ValueError(
            f"Unknown optical transition line shape {transition.line_shape!r}"
        )

    def local_transition_cross_sections(
        self,
        transition: OpticalTransition,
        wavelength_m: np.ndarray | float,
        spatial_shape: tuple[int, ...],
        z_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        wavelengths = np.atleast_1d(np.asarray(wavelength_m, dtype=float))
        base_abs, base_em = self._transition_cross_sections(
            transition, wavelengths
        )
        prefix = "pump" if transition.field == "pump" else "signal"
        absorption = self.response.cross_sections(
            f"{prefix}_absorption",
            wavelengths,
            np.asarray(base_abs, dtype=float),
            spatial_shape,
            z_index,
        )
        emission = self.response.cross_sections(
            f"{prefix}_emission",
            wavelengths,
            np.asarray(base_em, dtype=float),
            spatial_shape,
            z_index,
        )
        # An absolute table replaces the reference cross section, but not an
        # explicitly configured channel strength (including a disabled channel).
        if f"{prefix}_absorption" in self._absolute_cross_section_keys:
            absorption = absorption * transition.absorption_scale
        if f"{prefix}_emission" in self._absolute_cross_section_keys:
            emission = emission * transition.emission_scale
        return absorption, emission

    def record_spectral_support(
        self,
        transition_name: str,
        wavelengths_m: np.ndarray,
        power_weights: np.ndarray,
    ) -> None:
        """Audit weighted source support against a static cross-section table."""

        if transition_name == "pump":
            transitions = self.pump_transitions
        elif transition_name == "signal":
            transitions = self.signal_transitions
        else:
            raise ValueError("transition_name must be pump or signal")
        for transition in transitions:
            curve = self._transition_curves.get(id(transition))
            if curve is None:
                curve = (
                    self.pump_curve
                    if transition is self.pump
                    else self.signal_curve
                    if transition is self.signal
                    else None
                )
            if curve is None:
                continue
            self.response.range_guard.record(
                (
                    "energy_levels.optical_transitions."
                    f"{transition.name}.cross_section_spectrum"
                ),
                np.asarray(wavelengths_m, dtype=float),
                float(curve.wavelength_m[0]),
                float(curve.wavelength_m[-1]),
                weights=np.asarray(power_weights, dtype=float),
                zeroed_out_of_range=curve.out_of_range == "zero",
            )

    def local_cross_sections(
        self,
        transition_name: str,
        wavelength_m: np.ndarray | float,
        spatial_shape: tuple[int, ...],
        z_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cross sections with leading wavelength and trailing space axes."""

        transition = self.pump if transition_name == "pump" else self.signal
        if transition_name not in {"pump", "signal"}:
            raise ValueError("transition_name must be pump or signal")
        return self.local_transition_cross_sections(
            transition,
            wavelength_m,
            spatial_shape,
            z_index,
        )

    def upper_state_lifetime_s(
        self,
        spatial_shape: tuple[int, ...],
        z_index: int | None = None,
    ) -> np.ndarray:
        base = float(self.cfg["upper_state_lifetime_s"])
        return np.maximum(
            base * self.response.lifetime_scale(spatial_shape, z_index),
            np.finfo(float).tiny,
        )

    def signal_upper_total_lifetime_s(
        self,
        spatial_shape: tuple[int, ...],
        z_index: int | None = None,
    ) -> np.ndarray:
        """Return the actual linear-network lifetime of the Signal upper.

        Built-in models use the configured upper-state lifetime directly.
        Custom networks instead sum every outgoing linear decay from the
        unique Signal upper reservoir, including any environment-dependent
        lifetime scaling that is also used by the population propagator.
        """

        upper_indices = sorted(
            {transition.upper for transition in self.signal_transitions}
        )
        if len(upper_indices) != 1:
            raise ValueError(
                "A single total Signal-upper lifetime requires exactly one "
                "independent Signal upper reservoir."
            )
        if str(self.cfg.get("model", "four_level")) != "custom":
            return self.upper_state_lifetime_s(spatial_shape, z_index)
        return self.linear_decay_lifetime_s(
            upper_indices[0], spatial_shape, z_index
        )

    def linear_decay_lifetime_s(
        self,
        source: int,
        spatial_shape: tuple[int, ...],
        z_index: int | None = None,
    ) -> np.ndarray:
        """Use the same outgoing linear rates as the population matrix."""

        environment_lifetime = self.upper_state_lifetime_s(
            spatial_shape,
            z_index,
        )
        total_rate = np.zeros(spatial_shape, dtype=float)
        for decay in self.decays:
            if decay.source != source:
                continue
            rate: np.ndarray | float = decay.rate_s
            if decay.scales_with_upper_lifetime:
                rate = (
                    decay.rate_s
                    * float(self.cfg["upper_state_lifetime_s"])
                    / environment_lifetime
                )
            total_rate += rate
        if np.any(total_rate <= 0.0) or not np.all(np.isfinite(total_rate)):
            raise ValueError(
                "The source reservoir must have a positive finite "
                "total outgoing linear decay rate."
            )
        return 1.0 / total_rate

    def _linear_matrix(
        self,
        spatial_shape: tuple[int, ...],
        pump_intensity_W_m2: np.ndarray,
        pump_wavelength_m: float,
        signal_intensity_W_m2: np.ndarray | None = None,
        signal_wavelength_m: float | None = None,
        pump_wavelengths_m: np.ndarray | None = None,
        pump_power_weights: np.ndarray | None = None,
        signal_spectral_intensity_W_m2: np.ndarray | SpectralIntensityVolume | None = None,
        signal_wavelengths_m: np.ndarray | None = None,
        pump_spectral_intensity_W_m2: np.ndarray | None = None,
    ) -> np.ndarray:
        matrix = np.zeros((*spatial_shape, len(self.names), len(self.names)), dtype=float)
        if pump_spectral_intensity_W_m2 is not None:
            if pump_wavelengths_m is None:
                raise ValueError(
                    "pump_spectral_intensity_W_m2 requires pump_wavelengths_m"
                )
            wavelengths = np.asarray(pump_wavelengths_m, dtype=float)
            intensities = np.asarray(pump_spectral_intensity_W_m2, dtype=float)
            if intensities.shape != (len(wavelengths), *spatial_shape):
                raise ValueError(
                    "pump_spectral_intensity_W_m2 must have shape "
                    f"(Nlambda, {', '.join(str(item) for item in spatial_shape)})"
                )
            if np.any(intensities < 0.0) or not np.all(np.isfinite(intensities)):
                raise ValueError(
                    "pump_spectral_intensity_W_m2 must be finite and non-negative"
                )
            spectral_shape = (len(wavelengths),) + (1,) * len(spatial_shape)
            photon_flux = intensities / photon_energy(wavelengths).reshape(
                spectral_shape
            )
            for transition in self.pump_transitions:
                sigma_abs, sigma_em = self.local_transition_cross_sections(
                    transition, wavelengths, spatial_shape
                )
                self._add_transition_rates(
                    matrix,
                    transition,
                    transition.quantum_efficiency
                    * np.sum(sigma_abs * photon_flux, axis=0),
                    np.sum(sigma_em * photon_flux, axis=0),
                )
        elif pump_wavelengths_m is None or pump_power_weights is None:
            flux = pump_intensity_W_m2 / photon_energy(pump_wavelength_m)
            for transition in self.pump_transitions:
                sigma_abs, sigma_em = self.local_transition_cross_sections(
                    transition, pump_wavelength_m, spatial_shape
                )
                self._add_transition_rates(
                    matrix,
                    transition,
                    transition.quantum_efficiency * sigma_abs[0] * flux,
                    sigma_em[0] * flux,
                )
        else:
            wavelengths = np.asarray(pump_wavelengths_m, dtype=float)
            weights = np.asarray(pump_power_weights, dtype=float)
            spectral_shape = (len(wavelengths),) + (1,) * len(spatial_shape)
            for transition in self.pump_transitions:
                sigma_abs, sigma_em = self.local_transition_cross_sections(
                    transition, wavelengths, spatial_shape
                )
                absorption_per_watt = (
                    transition.quantum_efficiency
                    * np.sum(
                        weights.reshape(spectral_shape)
                        * sigma_abs
                        / photon_energy(wavelengths).reshape(spectral_shape),
                        axis=0,
                    )
                )
                emission_per_watt = np.sum(
                    weights.reshape(spectral_shape)
                    * sigma_em
                    / photon_energy(wavelengths).reshape(spectral_shape),
                    axis=0,
                )
                self._add_transition_rates(
                    matrix,
                    transition,
                    pump_intensity_W_m2 * absorption_per_watt,
                    pump_intensity_W_m2 * emission_per_watt,
                )
        if signal_spectral_intensity_W_m2 is not None or signal_wavelengths_m is not None:
            if signal_spectral_intensity_W_m2 is None or signal_wavelengths_m is None:
                raise ValueError(
                    "signal_spectral_intensity_W_m2 and signal_wavelengths_m must be provided together"
                )
            if signal_intensity_W_m2 is not None or signal_wavelength_m is not None:
                raise ValueError("provide either scalar or spectral signal intensity, not both")
            wavelengths = np.asarray(signal_wavelengths_m, dtype=float)
            for block_wavelengths, intensities in spectral_intensity_blocks(
                signal_spectral_intensity_W_m2, wavelengths, spatial_shape
            ):
                spectral_shape = (len(block_wavelengths),) + (1,) * len(spatial_shape)
                photon_flux = intensities / photon_energy(block_wavelengths).reshape(spectral_shape)
                for transition in self.signal_transitions:
                    sigma_abs, sigma_em = self.local_transition_cross_sections(
                        transition, block_wavelengths, spatial_shape
                    )
                    self._add_transition_rates(
                        matrix, transition,
                        np.sum(sigma_abs * photon_flux, axis=0),
                        np.sum(sigma_em * photon_flux, axis=0),
                    )
        elif signal_intensity_W_m2 is not None and signal_wavelength_m is not None:
            flux = signal_intensity_W_m2 / photon_energy(signal_wavelength_m)
            for transition in self.signal_transitions:
                sigma_abs, sigma_em = self.local_transition_cross_sections(
                    transition, signal_wavelength_m, spatial_shape
                )
                self._add_transition_rates(
                    matrix,
                    transition,
                    sigma_abs[0] * flux,
                    sigma_em[0] * flux,
                )
        upper_lifetime = self.upper_state_lifetime_s(spatial_shape)
        for decay in self.decays:
            rate: np.ndarray | float = decay.rate_s
            if decay.scales_with_upper_lifetime:
                rate = decay.rate_s * float(self.cfg["upper_state_lifetime_s"]) / upper_lifetime
            matrix[..., decay.target, decay.source] += rate
            matrix[..., decay.source, decay.source] -= rate
        self._add_spectral_diffusion(matrix)
        return matrix

    def advance_implicit(
        self,
        population: np.ndarray,
        dt: float,
        pump_intensity_W_m2: np.ndarray,
        pump_wavelength_m: float,
        signal_intensity_W_m2: np.ndarray | None = None,
        signal_wavelength_m: float | None = None,
        pump_wavelengths_m: np.ndarray | None = None,
        pump_power_weights: np.ndarray | None = None,
        signal_spectral_intensity_W_m2: np.ndarray | SpectralIntensityVolume | None = None,
        signal_wavelengths_m: np.ndarray | None = None,
        pump_spectral_intensity_W_m2: np.ndarray | None = None,
        *,
        step_events: dict[str, Any] | None = None,
    ) -> np.ndarray:
        spatial_shape = population.shape[1:]
        total_before = np.sum(population, axis=0)
        matrix = self._linear_matrix(
            spatial_shape,
            pump_intensity_W_m2,
            pump_wavelength_m,
            signal_intensity_W_m2,
            signal_wavelength_m,
            pump_wavelengths_m,
            pump_power_weights,
            signal_spectral_intensity_W_m2,
            signal_wavelengths_m,
            pump_spectral_intensity_W_m2,
        )
        lhs = np.eye(len(self.names)) - dt * matrix
        rhs = np.moveaxis(population, 0, -1)
        advanced = self._solve_implicit_systems(lhs, rhs)
        advanced = np.moveaxis(advanced, -1, 0)
        advanced = self._clip_nonnegative_and_audit(
            advanced,
            total_before,
            "advance_implicit_linear_solve",
        )
        nonlinear_counts = None
        if step_events is not None:
            step_events["linear_population_m3"] = advanced.copy()
            nonlinear_counts = [np.zeros(spatial_shape) for _ in self.nonlinear]
            step_events["nonlinear_event_counts_m3"] = nonlinear_counts
        advanced = self._advance_nonlinear(advanced, dt, nonlinear_counts)
        return self._renormalize_and_audit(
            advanced,
            total_before,
            "advance_implicit_post_nonlinear",
        )

    def advance_dark_implicit(
        self,
        population: np.ndarray,
        dt: float,
        z_index: int | None = None,
    ) -> np.ndarray:
        """Advance decay and nonlinear kinetics without evaluating optical data.

        A zero optical intensity is mathematically sufficient, but evaluating a
        strict pump cross-section curve at an unrelated Seed wavelength can
        still raise before it is multiplied by zero.  Pulse-transport backends
        use this explicit dark-step API between time bins.
        """

        spatial_shape = population.shape[1:]
        total_before = np.sum(population, axis=0)
        matrix = np.zeros(
            (*spatial_shape, len(self.names), len(self.names)),
            dtype=float,
        )
        upper_lifetime = self.upper_state_lifetime_s(spatial_shape, z_index)
        for decay in self.decays:
            rate: np.ndarray | float = decay.rate_s
            if decay.scales_with_upper_lifetime:
                rate = decay.rate_s * float(self.cfg["upper_state_lifetime_s"]) / upper_lifetime
            matrix[..., decay.target, decay.source] += rate
            matrix[..., decay.source, decay.source] -= rate
        self._add_spectral_diffusion(matrix)
        lhs = np.eye(len(self.names)) - float(dt) * matrix
        rhs = np.moveaxis(population, 0, -1)
        advanced = self._solve_implicit_systems(lhs, rhs)
        advanced = np.moveaxis(advanced, -1, 0)
        advanced = self._clip_nonnegative_and_audit(
            advanced,
            total_before,
            "advance_dark_implicit_linear_solve",
        )
        advanced = self._advance_nonlinear(advanced, float(dt))
        return self._renormalize_and_audit(
            advanced,
            total_before,
            "advance_dark_implicit_post_nonlinear",
        )

    def maximum_dark_linear_outflow_rate_s(
        self,
        spatial_shape: tuple[int, ...],
        z_index: int | None = None,
    ) -> float:
        """Return the fastest local total linear dark outflow rate.

        This is used only to choose a safe block size for operator splitting;
        the actual dark update remains the implicit, spatially resolved solve.
        Summing rates by source level matters when several decay branches leave
        one reservoir, and environment-scaled lifetimes must use the shortest
        local lifetime rather than the cold configured scalar.
        """

        constant_by_source = np.zeros(len(self.names), dtype=float)
        scaled_coefficient_by_source = np.zeros(len(self.names), dtype=float)
        base_upper_lifetime_s = float(self.cfg["upper_state_lifetime_s"])
        for decay in self.decays:
            if decay.scales_with_upper_lifetime:
                scaled_coefficient_by_source[decay.source] += (
                    float(decay.rate_s) * base_upper_lifetime_s
                )
            else:
                constant_by_source[decay.source] += float(decay.rate_s)

        minimum_upper_lifetime_s = float(
            np.min(self.upper_state_lifetime_s(spatial_shape, z_index))
        )
        total_by_source = constant_by_source + (
            scaled_coefficient_by_source / minimum_upper_lifetime_s
        )

        if self.ensemble_contract.get("enabled", False):
            diffusion_rate_s = float(
                self.ensemble_contract.get("spectral_diffusion_rate_s", 0.0)
            )
            if diffusion_rate_s > 0.0:
                groups = list(self.ensemble_contract["groups"])
                base_count = int(self.ensemble_contract["base_level_count"])
                for group in groups:
                    group_offset = int(group["level_offset"])
                    outward_fraction = 1.0 - float(group["fraction"])
                    total_by_source[
                        group_offset : group_offset + base_count
                    ] += diffusion_rate_s * outward_fraction
        return max(float(np.max(total_by_source)), 0.0)

    def steady_state(
        self,
        density_m3: float | np.ndarray,
        pump_intensity_W_m2: np.ndarray,
        pump_wavelength_m: float,
        signal_intensity_W_m2: np.ndarray | None = None,
        signal_wavelength_m: float | None = None,
        pump_wavelengths_m: np.ndarray | None = None,
        pump_power_weights: np.ndarray | None = None,
        signal_spectral_intensity_W_m2: np.ndarray | SpectralIntensityVolume | None = None,
        signal_wavelengths_m: np.ndarray | None = None,
        pump_spectral_intensity_W_m2: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve the local linear rate equations with ion conservation."""
        if self.nonlinear:
            raise ValueError(
                "population_dynamics=steady_state currently supports linear transitions only; "
                "use transient for upconversion or cross-relaxation processes"
            )
        if (
            self.cfg.get("model") in {"two_level", "three_level", "four_level"}
            and len(self.pump_transitions) == 1
            and len(self.signal_transitions) == 1
            and not self.ensemble_contract.get("enabled", False)
        ):
            self.steady_state_diagnostics = {
                "solver": "analytic_builtin_cascade",
                "conservation_component_count": 1,
                "constrained_matrix_rank_minimum": len(self.names),
                "minimum_singular_value_s_inv": None,
                "maximum_condition_number": None,
            }
            return self._cascade_steady_state(
                density_m3,
                pump_intensity_W_m2,
                pump_wavelength_m,
                signal_intensity_W_m2,
                signal_wavelength_m,
                pump_wavelengths_m,
                pump_power_weights,
                signal_spectral_intensity_W_m2,
                signal_wavelengths_m,
                pump_spectral_intensity_W_m2,
            )
        spatial_shape = pump_intensity_W_m2.shape
        matrix = self._linear_matrix(
            spatial_shape,
            pump_intensity_W_m2,
            pump_wavelength_m,
            signal_intensity_W_m2,
            signal_wavelength_m,
            pump_wavelengths_m,
            pump_power_weights,
            signal_spectral_intensity_W_m2,
            signal_wavelengths_m,
            pump_spectral_intensity_W_m2,
        ).copy()
        rhs = np.zeros((*spatial_shape, len(self.names)), dtype=float)
        density = np.broadcast_to(np.asarray(density_m3, dtype=float), spatial_shape)
        components = self.conservation_components()
        fractions = self.conservation_component_fractions()
        constraint_scale_s = np.maximum(
            np.max(np.abs(matrix), axis=(-2, -1)),
            1.0,
        )
        for component, fraction in zip(components, fractions):
            constraint_row = component[-1]
            matrix[..., constraint_row, :] = 0.0
            matrix[..., constraint_row, component] = (
                constraint_scale_s[..., None]
            )
            rhs[..., constraint_row] = (
                constraint_scale_s * fraction * density
            )

        flattened = matrix.reshape((-1, len(self.names), len(self.names)))
        sample_indices = np.unique(
            np.linspace(
                0,
                max(len(flattened) - 1, 0),
                min(len(flattened), 9),
                dtype=int,
            )
        )
        ranks: list[int] = []
        minimum_singular = float("inf")
        maximum_condition = 0.0
        for sample_index in sample_indices:
            sample = flattened[int(sample_index)]
            singular = np.linalg.svd(sample, compute_uv=False)
            rank = int(np.linalg.matrix_rank(sample))
            ranks.append(rank)
            minimum_singular = min(
                minimum_singular,
                float(np.min(singular)),
            )
            maximum_condition = max(
                maximum_condition,
                float(np.linalg.cond(sample)),
            )
        self.steady_state_diagnostics = {
            "solver": "component_constrained_dense_linear_solve",
            "conservation_component_count": len(components),
            "conservation_components": [
                [self.names[index] for index in component]
                for component in components
            ],
            "configured_component_inventory_fractions": fractions,
            "diagnostic_spatial_sample_count": int(len(sample_indices)),
            "constrained_matrix_rank_minimum": min(ranks),
            "matrix_dimension": len(self.names),
            "minimum_singular_value_s_inv": minimum_singular,
            "maximum_condition_number": maximum_condition,
        }
        if min(ranks) < len(self.names):
            raise ValueError(
                "steady-state level matrix remains rank deficient after "
                f"applying {len(components)} independent ion-inventory "
                "constraints; add physical transitions/decays or use transient "
                "population dynamics"
            )
        try:
            solved = np.linalg.solve(matrix, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "steady-state level matrix is singular in at least one spatial "
                "cell after applying all independent ion-inventory constraints; "
                "add physical transitions/decays or use transient population "
                "dynamics"
            ) from exc
        solved = np.moveaxis(solved, -1, 0)
        solved = self._clip_nonnegative_and_audit(
            solved,
            density,
            "steady_state_linear_solve",
        )
        return self._renormalize_and_audit(
            solved,
            density,
            "steady_state_post_clip",
        )

    def _cascade_steady_state(
        self,
        density_m3: float | np.ndarray,
        pump_intensity_W_m2: np.ndarray,
        pump_wavelength_m: float,
        signal_intensity_W_m2: np.ndarray | None,
        signal_wavelength_m: float | None,
        pump_wavelengths_m: np.ndarray | None,
        pump_power_weights: np.ndarray | None,
        signal_spectral_intensity_W_m2: np.ndarray | SpectralIntensityVolume | None,
        signal_wavelengths_m: np.ndarray | None,
        pump_spectral_intensity_W_m2: np.ndarray | None,
    ) -> np.ndarray:
        """Exact closed-form stationary solution for the built-in cascades.

        The generic custom network still uses a local dense solve.  The built-in
        topologies are simple chains, so ratios to the ground population remove
        the temporary ``(..., Nlevel, Nlevel)`` matrix and are substantially
        faster on large 3D grids without changing the equations.
        """

        pump_abs, pump_em, signal_abs, signal_em = self._optical_rates(
            pump_intensity_W_m2,
            pump_wavelength_m,
            signal_intensity_W_m2,
            signal_wavelength_m,
            pump_wavelengths_m,
            pump_power_weights,
            signal_spectral_intensity_W_m2,
            signal_wavelengths_m,
            pump_spectral_intensity_W_m2,
        )
        model = str(self.cfg["model"])
        upper_decay = 1.0 / self.upper_state_lifetime_s(
            pump_intensity_W_m2.shape
        )
        tiny = np.finfo(float).tiny
        density = np.broadcast_to(
            np.asarray(density_m3, dtype=float), pump_intensity_W_m2.shape
        )
        if model == "two_level":
            ratio_upper = (pump_abs + signal_abs) / np.maximum(
                pump_em + signal_em + upper_decay,
                tiny,
            )
            ground = density / (1.0 + ratio_upper)
            return np.stack([ground, ratio_upper * ground], axis=0)

        pump_decay = 1.0 / float(self.cfg["pump_level_lifetime_s"])
        ratio_pump = pump_abs / np.maximum(pump_em + pump_decay, tiny)
        if model == "three_level":
            ratio_upper = (pump_decay * ratio_pump + signal_abs) / np.maximum(
                signal_em + upper_decay,
                tiny,
            )
            ground = density / (1.0 + ratio_upper + ratio_pump)
            return np.stack(
                [ground, ratio_upper * ground, ratio_pump * ground],
                axis=0,
            )

        lower_decay = 1.0 / float(self.cfg["lower_level_lifetime_s"])
        ratio_lower = pump_decay * ratio_pump / lower_decay
        ratio_upper = ratio_lower * (signal_abs + lower_decay) / np.maximum(
            signal_em + upper_decay,
            tiny,
        )
        ground = density / (
            1.0 + ratio_lower + ratio_upper + ratio_pump
        )
        return np.stack(
            [
                ground,
                ratio_lower * ground,
                ratio_upper * ground,
                ratio_pump * ground,
            ],
            axis=0,
        )

    def _optical_rates(
        self,
        pump_intensity_W_m2: np.ndarray,
        pump_wavelength_m: float,
        signal_intensity_W_m2: np.ndarray | None,
        signal_wavelength_m: float | None,
        pump_wavelengths_m: np.ndarray | None,
        pump_power_weights: np.ndarray | None,
        signal_spectral_intensity_W_m2: np.ndarray | SpectralIntensityVolume | None,
        signal_wavelengths_m: np.ndarray | None,
        pump_spectral_intensity_W_m2: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        intensity = np.asarray(pump_intensity_W_m2, dtype=float)
        if pump_spectral_intensity_W_m2 is not None:
            if pump_wavelengths_m is None:
                raise ValueError(
                    "pump_spectral_intensity_W_m2 requires pump_wavelengths_m"
                )
            wavelengths = np.asarray(pump_wavelengths_m, dtype=float)
            spectral_intensity = np.asarray(
                pump_spectral_intensity_W_m2, dtype=float
            )
            if spectral_intensity.shape != (len(wavelengths), *intensity.shape):
                raise ValueError(
                    "pump_spectral_intensity_W_m2 shape does not match Pump wavelengths and grid"
                )
            if np.any(spectral_intensity < 0.0) or not np.all(
                np.isfinite(spectral_intensity)
            ):
                raise ValueError(
                    "pump_spectral_intensity_W_m2 must be finite and non-negative"
                )
            sigma_abs, sigma_em = self.local_cross_sections(
                "pump", wavelengths, intensity.shape
            )
            spectral_shape = (len(wavelengths),) + (1,) * intensity.ndim
            photon_flux = spectral_intensity / photon_energy(wavelengths).reshape(
                spectral_shape
            )
            pump_abs = self.pump_quantum_efficiency * np.sum(
                sigma_abs * photon_flux, axis=0
            )
            pump_em = np.sum(sigma_em * photon_flux, axis=0)
        elif pump_wavelengths_m is None or pump_power_weights is None:
            sigma_abs, sigma_em = self.local_cross_sections(
                "pump", pump_wavelength_m, intensity.shape
            )
            sigma_abs = sigma_abs[0]
            sigma_em = sigma_em[0]
            pump_flux = intensity / photon_energy(pump_wavelength_m)
            pump_abs = self.pump_quantum_efficiency * sigma_abs * pump_flux
            pump_em = sigma_em * pump_flux
        else:
            wavelengths = np.asarray(pump_wavelengths_m, dtype=float)
            weights = np.asarray(pump_power_weights, dtype=float)
            sigma_abs, sigma_em = self.local_cross_sections(
                "pump", wavelengths, intensity.shape
            )
            spectral_shape = (len(wavelengths),) + (1,) * intensity.ndim
            pump_abs = intensity * self.pump_quantum_efficiency * np.sum(
                weights.reshape(spectral_shape)
                * sigma_abs
                / photon_energy(wavelengths).reshape(spectral_shape),
                axis=0,
            )
            pump_em = intensity * np.sum(
                weights.reshape(spectral_shape)
                * sigma_em
                / photon_energy(wavelengths).reshape(spectral_shape),
                axis=0,
            )

        signal_abs = np.zeros_like(intensity)
        signal_em = np.zeros_like(intensity)
        if signal_spectral_intensity_W_m2 is not None or signal_wavelengths_m is not None:
            if signal_spectral_intensity_W_m2 is None or signal_wavelengths_m is None:
                raise ValueError(
                    "signal_spectral_intensity_W_m2 and signal_wavelengths_m must be provided together"
                )
            if signal_intensity_W_m2 is not None or signal_wavelength_m is not None:
                raise ValueError("provide either scalar or spectral signal intensity, not both")
            wavelengths = np.asarray(signal_wavelengths_m, dtype=float)
            for block_wavelengths, spectral_intensity in spectral_intensity_blocks(
                signal_spectral_intensity_W_m2, wavelengths, intensity.shape
            ):
                sigma_abs, sigma_em = self.local_cross_sections(
                    "signal", block_wavelengths, intensity.shape
                )
                spectral_shape = (len(block_wavelengths),) + (1,) * intensity.ndim
                photon_flux = spectral_intensity / photon_energy(block_wavelengths).reshape(spectral_shape)
                signal_abs += np.sum(sigma_abs * photon_flux, axis=0)
                signal_em += np.sum(sigma_em * photon_flux, axis=0)
        elif signal_intensity_W_m2 is not None and signal_wavelength_m is not None:
            sigma_abs, sigma_em = self.local_cross_sections(
                "signal", signal_wavelength_m, intensity.shape
            )
            sigma_abs = sigma_abs[0]
            sigma_em = sigma_em[0]
            signal_flux = np.asarray(signal_intensity_W_m2, dtype=float) / photon_energy(
                signal_wavelength_m
            )
            signal_abs = sigma_abs * signal_flux
            signal_em = sigma_em * signal_flux
        return pump_abs, pump_em, signal_abs, signal_em

    def _advance_nonlinear(
        self, population: np.ndarray, dt: float,
        event_counts: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        if not self.nonlinear:
            return population
        index = {name: i for i, name in enumerate(self.names)}
        result = population.copy()
        solver = self.cfg.get("nonlinear_solver", {})
        method = str(solver.get("method", "adaptive_substeps"))
        maximum_steps = int(solver.get("maximum_substeps", 256))
        target = float(
            solver.get("maximum_fractional_depletion_per_substep", 0.05)
        )
        remaining = float(dt)
        if not np.isfinite(remaining) or remaining < 0.0:
            raise ValueError("nonlinear population timestep must be finite and non-negative")
        steps = 0
        cap_limited = False
        maximum_fractional_step = 0.0
        # Scale the termination tolerance to the requested interval itself.
        # A one-second floor made every nonlinear step shorter than machine
        # epsilon seconds disappear entirely, which is reachable in finely
        # sampled ultrafast-pulse propagation.
        time_tolerance = abs(remaining) * np.finfo(float).eps
        while remaining > time_tolerance:
            step = remaining
            maximum_rate = self._maximum_nonlinear_fractional_rate(result, index)
            if method == "adaptive_substeps" and steps + 1 < maximum_steps:
                if maximum_rate > 0.0:
                    requested = target / maximum_rate
                    # If the requested step would need more than the remaining
                    # budget, distribute the unresolved interval over all
                    # remaining slots rather than taking one destructive jump.
                    available = maximum_steps - steps
                    step = min(remaining, max(requested, remaining / available))
            maximum_fractional_step = max(
                maximum_fractional_step,
                maximum_rate * step,
            )
            if (
                method == "adaptive_substeps"
                and maximum_rate * step > target * (1.0 + 64.0 * np.finfo(float).eps)
            ):
                # A budget-forced large step can occur before the final slot;
                # later depletion must not erase evidence of that violation.
                cap_limited = True
            self._apply_nonlinear_step(result, step, index, event_counts)
            remaining = max(0.0, remaining - step)
            steps += 1
            if method == "single_step" or steps >= maximum_steps:
                if remaining > 0.0:
                    remaining_rate = self._maximum_nonlinear_fractional_rate(
                        result,
                        index,
                    )
                    maximum_fractional_step = max(
                        maximum_fractional_step,
                        remaining_rate * remaining,
                    )
                    if (
                        method == "adaptive_substeps"
                        and remaining_rate * remaining > target
                    ):
                        cap_limited = True
                    self._apply_nonlinear_step(result, remaining, index, event_counts)
                break
        diagnostics = self.nonlinear_diagnostics
        diagnostics["calls"] = int(diagnostics["calls"]) + 1
        diagnostics["maximum_substeps_used"] = max(
            int(diagnostics["maximum_substeps_used"]),
            steps,
        )
        diagnostics["cap_limited_calls"] = int(
            diagnostics["cap_limited_calls"]
        ) + int(cap_limited)
        diagnostics["maximum_estimated_fractional_depletion_step"] = max(
            float(diagnostics["maximum_estimated_fractional_depletion_step"]),
            maximum_fractional_step,
        )
        return np.maximum(result, 0.0)

    def _maximum_nonlinear_fractional_rate(
        self,
        population: np.ndarray,
        index: dict[str, int],
    ) -> float:
        """Return the largest instantaneous reactant depletion rate in 1/s."""

        depletion_rate = np.zeros_like(population, dtype=float)
        for process in self.nonlinear:
            reactants = {index[k]: float(v) for k, v in process["reactants"].items()}
            rate = np.full(population.shape[1:], float(process["rate_coefficient"]))
            for level, order in reactants.items():
                rate *= population[level] ** order
            for level, coefficient in reactants.items():
                depletion_rate[level] += coefficient * rate
        fractional = np.zeros_like(population, dtype=float)
        np.divide(
            depletion_rate,
            population,
            out=fractional,
            where=population > np.finfo(float).tiny,
        )
        return 0.0 if fractional.size == 0 else float(np.max(fractional))

    def _apply_nonlinear_step(
        self,
        population: np.ndarray,
        dt: float,
        index: dict[str, int],
        event_counts: list[np.ndarray] | None = None,
    ) -> None:
        """Advance competing processes simultaneously from one frozen state.

        Every raw extent is evaluated from the same population.  A shared
        reactant budget then scales all processes that compete for a level,
        making a finite substep independent of configuration ordering while
        preserving non-negativity.
        """

        raw_extents: list[np.ndarray] = []
        reactants_by_process: list[dict[int, float]] = []
        products_by_process: list[dict[int, float]] = []
        requested_depletion = np.zeros_like(population, dtype=float)
        for process in self.nonlinear:
            reactants = {index[k]: float(v) for k, v in process["reactants"].items()}
            products = {index[k]: float(v) for k, v in process["products"].items()}
            rate = np.full(population.shape[1:], float(process["rate_coefficient"]))
            for level, order in reactants.items():
                rate *= population[level] ** order
            extent = rate * float(dt)
            for level, coefficient in reactants.items():
                requested_depletion[level] += coefficient * extent
            raw_extents.append(extent)
            reactants_by_process.append(reactants)
            products_by_process.append(products)

        reactant_scale = np.ones_like(population, dtype=float)
        np.divide(
            population,
            requested_depletion,
            out=reactant_scale,
            where=requested_depletion > population,
        )
        np.clip(reactant_scale, 0.0, 1.0, out=reactant_scale)
        delta = np.zeros_like(population, dtype=float)
        for process_index, (extent, reactants, products) in enumerate(zip(
            raw_extents,
            reactants_by_process,
            products_by_process,
            strict=True,
        )):
            if reactants:
                process_scale = np.minimum.reduce(
                    [reactant_scale[level] for level in reactants]
                )
                extent = extent * process_scale
            if event_counts is not None:
                event_counts[process_index] += extent
            for level, coefficient in reactants.items():
                delta[level] -= coefficient * extent
            for level, coefficient in products.items():
                delta[level] += coefficient * extent
        population += delta

    def pump_absorption_m_inv(
        self,
        population: np.ndarray,
        passive_m_inv: float,
        z_index: int | None = None,
        wavelength_m: float | None = None,
    ) -> np.ndarray:
        result = np.zeros_like(population[0], dtype=float)
        for transition in self.pump_transitions:
            lo = population[transition.lower]
            up = population[transition.upper]
            if wavelength_m is None:
                result += (
                    transition.sigma_abs_m2
                    * transition.absorption_scale
                    * lo
                    - transition.sigma_em_m2
                    * transition.emission_scale
                    * up
                )
            else:
                sigma_abs, sigma_em = self.local_transition_cross_sections(
                    transition,
                    float(wavelength_m),
                    lo.shape,
                    z_index,
                )
                result += sigma_abs[0] * lo - sigma_em[0] * up
        return result + passive_m_inv

    def pump_absorption_spectral_m_inv(
        self,
        population: np.ndarray,
        passive_m_inv: float,
        wavelengths_m: np.ndarray,
        power_weights: np.ndarray,
        z_index: int | None = None,
    ) -> np.ndarray:
        wavelengths = np.asarray(wavelengths_m, dtype=float)
        weights = np.asarray(power_weights, dtype=float)
        result = np.zeros_like(population[0], dtype=float)
        spectral_shape = (len(wavelengths),) + (1,) * result.ndim
        for transition in self.pump_transitions:
            lo = population[transition.lower]
            up = population[transition.upper]
            sigma_abs, sigma_em = self.local_transition_cross_sections(
                transition, wavelengths, lo.shape, z_index
            )
            effective_abs = np.sum(
                weights.reshape(spectral_shape) * sigma_abs, axis=0
            )
            effective_em = np.sum(
                weights.reshape(spectral_shape) * sigma_em, axis=0
            )
            result += effective_abs * lo - effective_em * up
        return result + passive_m_inv

    def signal_gain_components_m_inv(
        self,
        population: np.ndarray,
        wavelength_m: float,
        z_index: int | None = None,
    ) -> np.ndarray:
        """Return the signed gain contribution from every Signal transition."""

        components = []
        spatial_shape = population.shape[1:]
        for transition in self.signal_transitions:
            sigma_abs, sigma_em = self.local_transition_cross_sections(
                transition,
                np.asarray([wavelength_m]),
                spatial_shape,
                z_index,
            )
            components.append(
                sigma_em[0] * population[transition.upper]
                - sigma_abs[0] * population[transition.lower]
            )
        return np.stack(components, axis=0)

    def signal_gain_spectral_components_m_inv(
        self,
        population: np.ndarray,
        wavelengths_m: np.ndarray,
        z_index: int | None = None,
    ) -> np.ndarray:
        """Return ``(transition, wavelength, ...space)`` signed gain."""

        wavelengths = np.asarray(wavelengths_m, dtype=float)
        components = []
        spatial_shape = population.shape[1:]
        for transition in self.signal_transitions:
            sigma_abs, sigma_em = self.local_transition_cross_sections(
                transition, wavelengths, spatial_shape, z_index
            )
            components.append(
                sigma_em * population[transition.upper][None, ...]
                - sigma_abs * population[transition.lower][None, ...]
            )
        return np.stack(components, axis=0)

    def signal_gain_m_inv(
        self,
        population: np.ndarray,
        passive_m_inv: float = 0.0,
        wavelength_m: float | None = None,
        z_index: int | None = None,
    ) -> np.ndarray:
        if wavelength_m is None:
            result = np.zeros_like(population[0], dtype=float)
            for transition in self.signal_transitions:
                result += (
                    transition.sigma_em_m2 * transition.emission_scale
                    * population[transition.upper]
                    - transition.sigma_abs_m2 * transition.absorption_scale
                    * population[transition.lower]
                )
        else:
            result = np.sum(
                self.signal_gain_components_m_inv(
                    population, wavelength_m, z_index
                ),
                axis=0,
            )
        return result - passive_m_inv

    def signal_gain_spectrum_m_inv(
        self,
        population: np.ndarray,
        wavelengths_m: np.ndarray,
        passive_m_inv: float = 0.0,
        z_index: int | None = None,
    ) -> np.ndarray:
        return (
            np.sum(
                self.signal_gain_spectral_components_m_inv(
                    population, wavelengths_m, z_index
                ),
                axis=0,
            )
            - passive_m_inv
        )

    def signal_cross_section_sum_m2(
        self,
        wavelengths_m: np.ndarray | float,
        spatial_shape: tuple[int, ...],
        z_index: int | None = None,
    ) -> np.ndarray:
        """Sum absorption and emission cross sections over all Seed channels."""

        wavelengths = np.atleast_1d(np.asarray(wavelengths_m, dtype=float))
        result = np.zeros((len(wavelengths), *spatial_shape), dtype=float)
        for transition in self.signal_transitions:
            sigma_abs, sigma_em = self.local_transition_cross_sections(
                transition,
                wavelengths,
                spatial_shape,
                z_index,
            )
            result += sigma_abs + sigma_em
        return result

    @staticmethod
    def _limit_single_transition_events_reference(
        population: np.ndarray,
        events: np.ndarray,
        upper: int,
        lower: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Allocation-heavy reference limiter retained for numerical A/B tests."""

        channel_events = events[None, ...].copy()
        depletion = np.zeros_like(population)
        depletion[upper] += events
        depletion[lower] -= events
        positive_depletion = np.maximum(depletion, 0.0)
        scale = np.ones_like(events)
        for level in range(population.shape[0]):
            required = positive_depletion[level]
            local = np.ones_like(events)
            np.divide(
                population[level],
                required,
                out=local,
                where=required > np.finfo(float).tiny,
            )
            scale = np.minimum(scale, local)
        scale = np.clip(scale, 0.0, 1.0)
        channel_events *= scale[None, ...]
        updated = population.copy()
        updated[upper] -= channel_events[0]
        updated[lower] += channel_events[0]
        updated = np.maximum(updated, 0.0)
        return updated, channel_events[0], scale

    @staticmethod
    def _limit_single_transition_events_fast(
        population: np.ndarray,
        events: np.ndarray,
        upper: int,
        lower: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Equivalent limiter without level-sized temporary volumes."""

        required = np.abs(events)
        available = np.where(
            events > 0.0,
            population[upper],
            population[lower],
        )
        scale = np.ones_like(events)
        np.divide(
            available,
            required,
            out=scale,
            where=required > np.finfo(float).tiny,
        )
        np.clip(scale, 0.0, 1.0, out=scale)
        applied = events * scale
        updated = population.copy()
        updated[upper] -= applied
        updated[lower] += applied
        np.maximum(updated, 0.0, out=updated)
        return updated, applied, scale

    @profiled_operation("population.signal_photon_events")
    def apply_single_signal_photon_events_inplace(
        self,
        population: np.ndarray,
        photon_events_m3: np.ndarray,
        *,
        applied_events_out: np.ndarray,
        scale_work: np.ndarray,
        total_population_m3: np.ndarray,
        meaningful_event_threshold_m3: np.ndarray,
    ) -> np.ndarray:
        """Commit one-transition photon events without copying the level volume.

        This is the exact single-channel specialization of
        :meth:`apply_signal_photon_events`.  It is intentionally unavailable
        to multi-transition and inhomogeneous-ensemble models, whose channel
        allocation at transparency must continue through the general path.
        The caller owns both work arrays, so a coherent frequency sweep can
        reuse them for every packet.
        """

        if len(self.signal_transitions) != 1:
            raise ValueError(
                "the in-place photon-event path requires exactly one Signal "
                "transition"
            )
        events = np.asarray(photon_events_m3, dtype=float)
        applied = np.asarray(applied_events_out, dtype=float)
        scale = np.asarray(scale_work, dtype=float)
        total_population = np.asarray(total_population_m3, dtype=float)
        threshold = np.asarray(meaningful_event_threshold_m3, dtype=float)
        if (
            events.shape != population.shape[1:]
            or applied.shape != events.shape
            or scale.shape != events.shape
            or total_population.shape != events.shape
            or threshold.shape != events.shape
        ):
            raise ValueError(
                "single-transition event arrays must match the population "
                "spatial shape"
            )
        if np.shares_memory(events, applied) or np.shares_memory(events, scale):
            raise ValueError(
                "photon_events_m3 must not alias the in-place event work arrays"
            )

        transition = self.signal_transitions[0]
        upper = population[transition.upper]
        lower = population[transition.lower]
        tiny = np.finfo(float).tiny

        # applied temporarily holds |requested|; scale temporarily holds the
        # population available on the emitting/absorbing source level.
        np.abs(events, out=applied)
        np.copyto(scale, lower)
        np.copyto(scale, upper, where=events > 0.0)
        finite_requirement = applied > tiny
        # A vanishingly small but nonzero requested event can make the
        # unconstrained ratio overflow even though it is immediately clipped
        # to one.  Suppress that benign intermediate warning without changing
        # the limiter result or hiding non-finite inputs elsewhere.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            np.divide(scale, applied, out=scale, where=finite_requirement)
        scale[~finite_requirement] = 1.0
        np.clip(scale, 0.0, 1.0, out=scale)

        meaningful = applied > threshold
        limiter_roundoff = 64.0 * np.finfo(float).eps
        limited = meaningful & (scale < 1.0 - limiter_roundoff)
        minimum_scale = 1.0
        if np.any(meaningful):
            minimum_scale = float(np.min(scale[meaningful]))

        np.multiply(events, scale, out=applied)
        upper -= applied
        lower += applied
        # The limiter makes these clips numerical no-ops for finite,
        # non-negative input, while retaining the public limiter's safeguard.
        np.maximum(upper, 0.0, out=upper)
        np.maximum(lower, 0.0, out=lower)

        # Reuse scale after the limiter evidence has been reduced.
        np.abs(applied, out=scale)
        np.divide(
            scale,
            total_population,
            out=scale,
            where=total_population > tiny,
        )
        scale[total_population <= tiny] = 0.0
        self.record_single_signal_event_summary(
            calls=1,
            limited_cell_updates=int(np.count_nonzero(limited)),
            minimum_limiter_scale=minimum_scale,
            maximum_population_change_fraction=float(np.max(scale)),
        )
        return applied

    def record_single_signal_event_summary(
        self,
        *,
        calls: int,
        limited_cell_updates: int,
        minimum_limiter_scale: float,
        maximum_population_change_fraction: float,
    ) -> None:
        """Merge reduced evidence from an exact one-channel event kernel."""

        diagnostics = self.signal_event_diagnostics
        diagnostics["calls"] = int(diagnostics["calls"]) + int(calls)
        diagnostics["limited_cell_updates"] = int(
            diagnostics["limited_cell_updates"]
        ) + int(limited_cell_updates)
        minimum_scale = float(minimum_limiter_scale)
        diagnostics["minimum_limiter_scale"] = min(
            float(diagnostics["minimum_limiter_scale"]),
            minimum_scale,
        )
        # For one channel applied=requested*scale, so both the net and channel
        # requested/applied residuals are exactly 1-scale.
        maximum_residual = max(0.0, 1.0 - minimum_scale)
        diagnostics["maximum_requested_applied_residual_relative"] = max(
            float(
                diagnostics[
                    "maximum_requested_applied_residual_relative"
                ]
            ),
            maximum_residual,
        )
        diagnostics[
            "maximum_channel_requested_applied_residual_relative"
        ] = max(
            float(
                diagnostics[
                    "maximum_channel_requested_applied_residual_relative"
                ]
            ),
            maximum_residual,
        )
        diagnostics["maximum_population_change_fraction_per_update"] = max(
            float(
                diagnostics[
                    "maximum_population_change_fraction_per_update"
                ]
            ),
            float(maximum_population_change_fraction),
        )

    @profiled_operation("population.signal_photon_events")
    def apply_signal_photon_events(
        self,
        population: np.ndarray,
        photon_events_m3: np.ndarray,
        wavelength_m: float,
        *,
        z_index: int | None = None,
        input_photon_density_m3: np.ndarray | None = None,
        interaction_length_m: float | None = None,
        record_diagnostics: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply net stimulated photon events across competing transitions.

        Positive events are net stimulated emission; negative events are net
        absorption. For multiple transitions, channel events are formed from
        the integrated incident photon density using
        ``g_i L * expm1(g_total L)/(g_total L)``. This remains finite at net
        transparency, where emission and absorption channels can exchange
        population although the net optical energy is unchanged. A single
        limiter preserves non-negativity and photon/ion event closure.
        """

        events = np.asarray(photon_events_m3, dtype=float)
        if len(self.signal_transitions) == 1:
            allocation_method = "single_transition_exact_net_event"
            transport_scale = np.ones_like(events)
            transparency_limit = np.zeros_like(events, dtype=bool)
            # Diagnostics only read requested events, so a view avoids two
            # full-size copies before the independently allocated applied
            # channel is available.
            requested_channel_events = events[None, ...]
        else:
            components = self.signal_gain_components_m_inv(
                population, wavelength_m, z_index
            )
            total_gain = np.sum(components, axis=0)
            if input_photon_density_m3 is None or interaction_length_m is None:
                raise ValueError(
                    "multi-transition photon-event updates require "
                    "input_photon_density_m3 and interaction_length_m so "
                    "transparent counter-propagating channel events are not lost"
                )
            incident_density = np.asarray(
                input_photon_density_m3, dtype=float
            )
            if incident_density.shape != events.shape:
                incident_density = np.broadcast_to(
                    incident_density, events.shape
                )
            length = float(interaction_length_m)
            if (
                not np.isfinite(length)
                or length <= 0.0
                or np.any(~np.isfinite(incident_density))
                or np.any(incident_density < 0.0)
            ):
                raise ValueError(
                    "multi-transition incident photon density must be finite "
                    "and non-negative and interaction_length_m must be positive"
                )
            exponent = total_gain * length
            small = np.abs(exponent) < 1e-7
            safely_nonzero = ~small
            physically_consistent = safely_nonzero & (
                events * exponent >= 0.0
            )
            channel_events = np.zeros(
                (len(self.signal_transitions), *events.shape),
                dtype=float,
            )
            # Away from transparency every raw channel contains the same
            # expm1(gL)/(gL) factor.  Cancel it analytically before allocating
            # the requested saturated net event.  This is algebraically
            # identical to raw_channel * requested/raw_total, but remains
            # valid when exp(gL) would overflow or when a historical exponent
            # cap would understate a very-high-gain, very-low-input channel.
            np.divide(
                components * events[None, ...],
                total_gain[None, ...],
                out=channel_events,
                where=physically_consistent,
            )

            # Preserve the finite opposing transition activity at the exact
            # transparency limit, where components/total_gain is undefined.
            phi_small = np.zeros_like(exponent)
            small_exponent = exponent[small]
            phi_small[small] = (
                1.0
                + 0.5 * small_exponent
                + small_exponent**2 / 6.0
                + small_exponent**3 / 24.0
            )
            raw_channel_events = np.zeros_like(channel_events)
            np.multiply(
                incident_density[None, ...]
                * components
                * length,
                phi_small[None, ...],
                out=raw_channel_events,
                where=small[None, ...],
            )
            absolute_channel_scale = np.sum(
                np.abs(raw_channel_events), axis=0
            )
            # At exact or near transparency the ratio requested/raw_total is
            # undefined. Preserve the finite opposing channel events and put
            # only their tiny net optical residual into a normalized,
            # non-negative magnitude weighting.
            magnitude_weights = np.divide(
                np.abs(raw_channel_events),
                absolute_channel_scale[None, ...],
                out=np.full_like(
                    raw_channel_events,
                    1.0 / len(self.signal_transitions),
                ),
                where=absolute_channel_scale[None, ...]
                > np.finfo(float).tiny,
            )
            transparent_channels = raw_channel_events + magnitude_weights * (
                events - np.sum(raw_channel_events, axis=0)
            )[None, ...]
            np.copyto(
                channel_events,
                transparent_channels,
                where=(~safely_nonzero)[None, ...],
            )

            # Retain a stable diagnostic of the saturated/unsaturated
            # transport ratio without ever materializing exp(gL).  This value
            # does not participate in the event allocation above.
            log_raw_magnitude = np.full_like(events, -np.inf)
            positive_incident = incident_density > 0.0
            positive_exponent = exponent > 50.0
            log_abs_expm1 = np.full_like(exponent, -np.inf)
            log_abs_expm1[positive_exponent] = exponent[
                positive_exponent
            ] + np.log1p(-np.exp(-exponent[positive_exponent]))
            remaining = ~positive_exponent
            with np.errstate(divide="ignore", invalid="ignore"):
                log_abs_expm1[remaining] = np.log(
                    np.abs(np.expm1(exponent[remaining]))
                )
                log_raw_magnitude[positive_incident] = np.log(
                    incident_density[positive_incident]
                ) + log_abs_expm1[positive_incident]
            saturation_scale = np.zeros_like(events)
            meaningful_requested = np.abs(events) > 0.0
            with np.errstate(over="ignore", under="ignore"):
                saturation_scale[physically_consistent & meaningful_requested] = np.exp(
                    np.minimum(
                        np.log(
                            np.abs(
                                events[
                                    physically_consistent
                                    & meaningful_requested
                                ]
                            )
                        )
                        - log_raw_magnitude[
                            physically_consistent & meaningful_requested
                        ],
                        0.0,
                    )
                )
            saturation_scale[~safely_nonzero] = 1.0
            transport_scale = saturation_scale
            transparency_limit = ~safely_nonzero
            allocation_method = (
                "algebraically_cancelled_integrated_channel_gain_with_transparency_limit"
            )
            requested_channel_events = channel_events.copy()
        if len(self.signal_transitions) == 1:
            transition = self.signal_transitions[0]
            updated, applied, scale = (
                self._limit_single_transition_events_fast(
                    population,
                    events,
                    transition.upper,
                    transition.lower,
                )
            )
            channel_events = applied[None, ...]
        else:
            depletion = np.zeros_like(population)
            for index, transition in enumerate(self.signal_transitions):
                event = channel_events[index]
                depletion[transition.upper] += event
                depletion[transition.lower] -= event
            positive_depletion = np.maximum(depletion, 0.0)
            scale = np.ones_like(events)
            for level in range(len(self.names)):
                required = positive_depletion[level]
                available = population[level]
                local = np.ones_like(events)
                np.divide(
                    available,
                    required,
                    out=local,
                    where=required > np.finfo(float).tiny,
                )
                scale = np.minimum(scale, local)
            scale = np.clip(scale, 0.0, 1.0)
            channel_events *= scale[None, ...]
            updated = population.copy()
            for index, transition in enumerate(self.signal_transitions):
                event = channel_events[index]
                updated[transition.upper] -= event
                updated[transition.lower] += event
            updated = np.maximum(updated, 0.0)
            applied = np.sum(channel_events, axis=0)
        requested_minus_applied = events - applied
        if record_diagnostics:
            total_population = np.sum(population, axis=0)
            meaningful_net = np.maximum(
                np.abs(events),
                np.abs(applied),
            ) > (
                64.0
                * np.finfo(float).eps
                * np.maximum(total_population, 1.0)
            )
            residual_relative = np.divide(
                np.abs(requested_minus_applied),
                np.maximum(np.abs(events), np.abs(applied)),
                out=np.zeros_like(events),
                where=meaningful_net,
            )
            requested_channel_activity = np.sum(
                np.abs(requested_channel_events), axis=0
            )
            applied_channel_activity = np.sum(np.abs(channel_events), axis=0)
            meaningful_channel = np.maximum(
                requested_channel_activity,
                applied_channel_activity,
            ) > (
                64.0
                * np.finfo(float).eps
                * np.maximum(total_population, 1.0)
            )
            channel_residual_relative = np.divide(
                np.sum(
                    np.abs(requested_channel_events - channel_events),
                    axis=0,
                ),
                np.maximum(
                    requested_channel_activity,
                    applied_channel_activity,
                ),
                out=np.zeros_like(events),
                where=meaningful_channel,
            )
            change_fraction = np.divide(
                np.max(np.abs(updated - population), axis=0),
                total_population,
                out=np.zeros_like(total_population),
                where=total_population > np.finfo(float).tiny,
            )
            diagnostics = self.signal_event_diagnostics
            meaningful = meaningful_net | meaningful_channel
            limited = meaningful_channel & (
                scale < 1.0 - 64.0 * np.finfo(float).eps
            )
            diagnostics["calls"] = int(diagnostics["calls"]) + 1
            diagnostics["limited_cell_updates"] = int(
                diagnostics["limited_cell_updates"]
            ) + int(np.count_nonzero(limited))
            if np.any(meaningful):
                diagnostics["minimum_limiter_scale"] = min(
                    float(diagnostics["minimum_limiter_scale"]),
                    float(np.min(scale[meaningful])),
                )
            diagnostics[
                "maximum_requested_applied_residual_relative"
            ] = max(
                float(
                    diagnostics[
                        "maximum_requested_applied_residual_relative"
                    ]
                ),
                float(np.max(residual_relative)),
                float(np.max(channel_residual_relative)),
            )
            diagnostics[
                "maximum_channel_requested_applied_residual_relative"
            ] = max(
                float(
                    diagnostics[
                        "maximum_channel_requested_applied_residual_relative"
                    ]
                ),
                float(np.max(channel_residual_relative)),
            )
            diagnostics["maximum_population_change_fraction_per_update"] = max(
                float(
                    diagnostics[
                        "maximum_population_change_fraction_per_update"
                    ]
                ),
                float(np.max(change_fraction)),
            )
        return updated, {
            "requested_photon_events_m3": events,
            "applied_photon_events_m3": applied,
            "limiter_scale": scale,
            "requested_channel_events_m3": requested_channel_events,
            "channel_events_m3": channel_events,
            "channel_allocation_method": allocation_method,
            "channel_transport_scale": transport_scale,
            "transparency_limit_active": transparency_limit,
            "requested_minus_applied_photon_events_m3": requested_minus_applied,
            "transition_names": [
                transition.name for transition in self.signal_transitions
            ],
        }

    def aggregate_signal_upper_population(
        self, population: np.ndarray
    ) -> np.ndarray:
        indices = sorted(
            {transition.upper for transition in self.signal_transitions}
        )
        return np.sum(population[indices], axis=0)
