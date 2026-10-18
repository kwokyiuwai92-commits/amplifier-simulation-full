"""Explicit event-rate energy and heat ledger for level kinetics.

The ledger never assigns an unspecified transition energy to heat.  Every
decay and nonlinear event is matched to a configured energy partition whose
fractions sum to one.  Pump absorption/stimulated-emission events are closed
directly against the configured level-energy gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import photon_energy
from .grid import Grid3D
from .levels import LevelSystem


_DESTINATIONS = (
    "heat",
    "fluorescence_escape",
    "ase_source",
    "unresolved",
)


@dataclass
class EventEnergyResult:
    heat_W_m3: np.ndarray
    destination_W_m3: dict[str, np.ndarray]
    event_rate_m3_s: dict[str, np.ndarray]
    stats: dict[str, Any]


def _base_level_name(name: str) -> str:
    return name.split(":", 1)[-1]


def ase_source_interface_diagnostics(
    event_energy_stats: dict[str, Any],
    ase_stats: dict[str, Any],
) -> dict[str, Any]:
    """Compare the event-declared ASE source with the transported source.

    Event partitions describe the full-space radiative destination.  ASE
    transport can intentionally retain only a configured solid-angle
    fraction, so both sources are put on that modeled angular basis before
    their relative mismatch is evaluated.
    """

    event_enabled = bool(event_energy_stats.get("enabled", False))
    ase_enabled = bool(ase_stats.get("enabled", False))
    if not event_enabled or not ase_enabled:
        return {
            "enabled": False,
            "applicable": False,
            "reason": (
                "event_energy_disabled"
                if not event_enabled
                else "ase_transport_disabled"
            ),
        }
    destination_power = event_energy_stats.get("destination_power_W") or {}
    ase_energy = ase_stats.get("energy_ledger_W") or {}
    declared = destination_power.get("ase_source")
    transported = ase_energy.get("spontaneous_source")
    if declared is None or transported is None:
        return {
            "enabled": True,
            "applicable": False,
            "reason": "source_power_missing",
            "event_declared_full_space_source_W": declared,
            "transport_spontaneous_source_W": transported,
        }
    solid_angle_fraction = float(
        ase_stats.get("modeled_solid_angle_fraction", 1.0)
    )
    declared_modeled = float(declared) * solid_angle_fraction
    transported_value = float(transported)
    mismatch = declared_modeled - transported_value
    scale = max(
        abs(declared_modeled),
        abs(transported_value),
        np.finfo(float).tiny,
    )
    return {
        "enabled": True,
        "applicable": True,
        "event_declared_full_space_source_W": float(declared),
        "modeled_solid_angle_fraction": solid_angle_fraction,
        "event_declared_modeled_solid_angle_source_W": declared_modeled,
        "transport_spontaneous_source_W": transported_value,
        "signed_mismatch_W": mismatch,
        "relative_mismatch": abs(mismatch) / scale,
        "identity": (
            "event full-space ase_source * modeled solid-angle fraction "
            "= ASE transport spontaneous_source"
        ),
    }


class EventEnergyLedger:
    """Evaluate optical, decay, and nonlinear energy rates at one population."""

    def __init__(
        self,
        levels: LevelSystem,
        cfg: dict[str, Any],
        pump_wavelength_m: float,
        signal_wavelength_m: float,
    ):
        self.levels = levels
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", False))
        self.require_complete = bool(
            cfg.get("require_complete_partitions", True)
        )
        if self.enabled:
            self.level_energies_J, self.level_energy_source = (
                levels.reference_level_energies_J(
                    pump_wavelength_m,
                    signal_wavelength_m,
                )
            )
        else:
            self.level_energies_J = np.asarray([], dtype=float)
            self.level_energy_source = "disabled"
        self._decay_partitions = list(cfg.get("decay_partitions", []))
        self._nonlinear_partitions = list(
            cfg.get("nonlinear_partitions", [])
        )

    @staticmethod
    def _fractions(item: dict[str, Any]) -> dict[str, float]:
        configured = item.get("fractions", item)
        return {
            destination: float(configured.get(destination, 0.0))
            for destination in _DESTINATIONS
        }

    def _decay_partition(
        self,
        source_name: str,
        target_name: str,
        spatial_shape: tuple[int, ...],
        *,
        scales_with_upper_lifetime: bool,
    ) -> dict[str, float | np.ndarray] | None:
        base_source = _base_level_name(source_name)
        base_target = _base_level_name(target_name)
        matches = [
            item
            for item in self._decay_partitions
            if (
                str(item.get("from")) in {source_name, base_source}
                and str(item.get("to")) in {target_name, base_target}
            )
        ]
        if len(matches) > 1:
            raise ValueError(
                f"multiple event-energy partitions match decay "
                f"{source_name!r}->{target_name!r}"
            )
        if not matches:
            return None
        selected = matches[0]
        if str(selected.get("model", "fixed")) != "lifetime_branching":
            return self._fractions(selected)
        if not scales_with_upper_lifetime:
            raise ValueError(
                "event-energy lifetime_branching can only be applied to a "
                "decay that scales with the upper-state lifetime"
            )
        radiative_lifetime_s = float(
            self.levels.cfg["upper_state_radiative_lifetime_s"]
        )
        total_lifetime_s = np.asarray(
            self.levels.linear_decay_lifetime_s(
                self.levels.names.index(source_name), spatial_shape
            ),
            dtype=float,
        )
        if (
            not np.isfinite(radiative_lifetime_s)
            or radiative_lifetime_s <= 0.0
            or not np.all(np.isfinite(total_lifetime_s))
            or np.any(total_lifetime_s <= 0.0)
        ):
            raise ValueError("event-energy lifetime_branching requires positive finite lifetimes")
        if np.any(total_lifetime_s > radiative_lifetime_s * (1.0 + 1e-12)):
            raise ValueError(
                "event-energy lifetime_branching: the local upper-state total "
                "lifetime cannot exceed the radiative lifetime"
            )
        radiative_fraction = np.clip(
            total_lifetime_s / radiative_lifetime_s,
            0.0,
            1.0,
        )
        radiative_destination = str(
            selected.get(
                "radiative_destination",
                "fluorescence_escape",
            )
        )
        fractions: dict[str, float | np.ndarray] = {
            destination: 0.0 for destination in _DESTINATIONS
        }
        fractions["heat"] = 1.0 - radiative_fraction
        fractions[radiative_destination] = radiative_fraction
        return fractions

    def _nonlinear_partition(
        self,
        process: dict[str, Any],
        index: int,
    ) -> dict[str, float] | None:
        name = str(process.get("name", f"process_{index}"))
        matches = [
            item
            for item in self._nonlinear_partitions
            if (
                str(item.get("process", "")) == name
                or item.get("process_index") == index
            )
        ]
        if len(matches) > 1:
            raise ValueError(
                f"multiple event-energy partitions match nonlinear process {name!r}"
            )
        return None if not matches else self._fractions(matches[0])

    @staticmethod
    def _accumulate(
        destination: dict[str, np.ndarray],
        fractions: dict[str, float | np.ndarray],
        released_power: np.ndarray,
    ) -> None:
        for name in _DESTINATIONS:
            destination[name] += fractions[name] * released_power

    def evaluate(
        self,
        population_m3: np.ndarray,
        *,
        pump_wavelengths_m: np.ndarray,
        pump_spectral_intensity_W_m2: np.ndarray,
        grid: Grid3D | None = None,
        nonlinear_event_counts_m3: list[np.ndarray] | None = None,
        step_duration_s: float | None = None,
    ) -> EventEnergyResult:
        population = np.asarray(population_m3, dtype=float)
        spatial_shape = population.shape[1:]
        if not self.enabled:
            zeros = np.zeros(spatial_shape, dtype=float)
            return EventEnergyResult(
                zeros,
                {name: zeros.copy() for name in _DESTINATIONS},
                {},
                {"enabled": False},
            )
        if population.shape[0] != len(self.levels.names):
            raise ValueError("event-energy population level count is inconsistent")
        if self.level_energies_J.shape != (len(self.levels.names),):
            raise ValueError(
                "event-energy level-energy vector must contain exactly one "
                "value per expanded level"
            )

        wavelengths = np.asarray(pump_wavelengths_m, dtype=float)
        intensity = np.asarray(pump_spectral_intensity_W_m2, dtype=float)
        if intensity.shape != (len(wavelengths), *spatial_shape):
            raise ValueError(
                "event-energy Pump spectral intensity must have shape "
                "(Nwavelength, Nz, Ny, Nx)"
            )
        destination = {
            name: np.zeros(spatial_shape, dtype=float)
            for name in _DESTINATIONS
        }
        rates: dict[str, np.ndarray] = {}
        energy_rate_by_event: dict[str, np.ndarray] = {}
        optical_input = np.zeros(spatial_shape, dtype=float)
        optical_output = np.zeros(spatial_shape, dtype=float)
        stored_rate = np.zeros(spatial_shape, dtype=float)
        spectral_shape = (len(wavelengths),) + (1,) * len(spatial_shape)
        photon_energies = photon_energy(wavelengths).reshape(spectral_shape)
        photon_flux = intensity / photon_energies

        for transition in self.levels.pump_transitions:
            sigma_abs, sigma_em = (
                self.levels.local_transition_cross_sections(
                    transition,
                    wavelengths,
                    spatial_shape,
                )
            )
            raw_absorption = (
                sigma_abs
                * photon_flux
                * population[transition.lower][None, ...]
            )
            stimulated_emission = (
                sigma_em
                * photon_flux
                * population[transition.upper][None, ...]
            )
            successful_absorption = (
                transition.quantum_efficiency * raw_absorption
            )
            gap = float(
                self.level_energies_J[transition.upper]
                - self.level_energies_J[transition.lower]
            )
            if gap < -64.0 * np.finfo(float).eps * max(
                float(np.max(np.abs(self.level_energies_J))),
                np.finfo(float).tiny,
            ):
                raise ValueError(
                    f"Pump transition {transition.name!r} has a negative configured energy gap"
                )
            absorbed_power = np.sum(raw_absorption * photon_energies, axis=0)
            emitted_power = np.sum(
                stimulated_emission * photon_energies,
                axis=0,
            )
            transition_storage = gap * np.sum(
                successful_absorption - stimulated_emission,
                axis=0,
            )
            transition_heat = (
                absorbed_power - emitted_power - transition_storage
            )
            destination["heat"] += transition_heat
            optical_input += absorbed_power
            optical_output += emitted_power
            stored_rate += transition_storage
            absorption_name = f"pump:{transition.name}:absorption"
            emission_name = f"pump:{transition.name}:stimulated_emission"
            rates[absorption_name] = np.sum(
                raw_absorption,
                axis=0,
            )
            rates[emission_name] = np.sum(
                stimulated_emission,
                axis=0,
            )
            energy_rate_by_event[absorption_name] = absorbed_power
            energy_rate_by_event[emission_name] = emitted_power

        pump_heat = destination["heat"].copy()
        kinetic_release = np.zeros(spatial_shape, dtype=float)
        missing_decay: list[str] = []
        for decay_index, decay in enumerate(self.levels.decays):
            source_name = self.levels.names[decay.source]
            target_name = self.levels.names[decay.target]
            partition = self._decay_partition(
                source_name,
                target_name,
                spatial_shape,
                scales_with_upper_lifetime=decay.scales_with_upper_lifetime,
            )
            if partition is None:
                missing_decay.append(f"{source_name}->{target_name}")
                partition = {
                    "heat": 0.0,
                    "fluorescence_escape": 0.0,
                    "ase_source": 0.0,
                    "unresolved": 1.0,
                }
            local_rate: np.ndarray | float = decay.rate_s
            if decay.scales_with_upper_lifetime:
                local_rate = (
                    decay.rate_s
                    * float(self.levels.cfg["upper_state_lifetime_s"])
                    / self.levels.upper_state_lifetime_s(spatial_shape)
                )
            event_rate = population[decay.source] * local_rate
            gap = float(
                self.level_energies_J[decay.source]
                - self.level_energies_J[decay.target]
            )
            if gap < -64.0 * np.finfo(float).eps * max(
                float(np.max(np.abs(self.level_energies_J))),
                np.finfo(float).tiny,
            ):
                raise ValueError(
                    f"decay {source_name!r}->{target_name!r} has a negative energy release"
                )
            released = max(gap, 0.0) * event_rate
            kinetic_release += released
            self._accumulate(destination, partition, released)
            event_name = (
                f"decay:{decay_index}:{source_name}->{target_name}"
            )
            rates[event_name] = event_rate
            energy_rate_by_event[event_name] = released

        missing_nonlinear: list[str] = []
        level_index = {
            name: index for index, name in enumerate(self.levels.names)
        }
        for process_index, process in enumerate(self.levels.nonlinear):
            name = str(process.get("name", f"process_{process_index}"))
            partition = self._nonlinear_partition(process, process_index)
            if partition is None:
                missing_nonlinear.append(name)
                partition = {
                    "heat": 0.0,
                    "fluorescence_escape": 0.0,
                    "ase_source": 0.0,
                    "unresolved": 1.0,
                }
            if nonlinear_event_counts_m3 is None:
                event_rate = np.full(
                    spatial_shape, float(process["rate_coefficient"]), dtype=float
                )
                for level_name, order in process["reactants"].items():
                    event_rate *= population[level_index[level_name]] ** float(order)
            else:
                if step_duration_s is None or step_duration_s <= 0.0:
                    raise ValueError("Integrated nonlinear events require a positive step duration")
                event_rate = nonlinear_event_counts_m3[process_index] / step_duration_s
            reactant_energy = sum(
                float(coefficient)
                * float(self.level_energies_J[level_index[level_name]])
                for level_name, coefficient in process["reactants"].items()
            )
            product_energy = sum(
                float(coefficient)
                * float(self.level_energies_J[level_index[level_name]])
                for level_name, coefficient in process["products"].items()
            )
            release_per_event = reactant_energy - product_energy
            if release_per_event < -64.0 * np.finfo(float).eps * max(
                abs(reactant_energy),
                abs(product_energy),
                np.finfo(float).tiny,
            ):
                raise ValueError(
                    f"nonlinear process {name!r} is endothermic; an external "
                    "energy source is required and cannot be silently assigned"
                )
            released = max(release_per_event, 0.0) * event_rate
            kinetic_release += released
            self._accumulate(destination, partition, released)
            event_name = f"nonlinear:{process_index}:{name}"
            rates[event_name] = event_rate
            energy_rate_by_event[event_name] = released

        if self.require_complete and (missing_decay or missing_nonlinear):
            details = [
                *[f"decay {name}" for name in missing_decay],
                *[f"nonlinear {name}" for name in missing_nonlinear],
            ]
            raise ValueError(
                "event_energy_ledger requires explicit partitions for "
                + ", ".join(details)
            )

        heat = destination["heat"]
        pointwise_optical_residual = (
            optical_input - optical_output - stored_rate - pump_heat
        )
        total_storage_derivative = stored_rate - kinetic_release
        total_destination = sum(
            destination.values(),
            np.zeros(spatial_shape, dtype=float),
        )
        full_identity_residual = (
            optical_input
            - optical_output
            - total_destination
            - total_storage_derivative
        )
        stats: dict[str, Any] = {
            "enabled": True,
            "level_energy_source": self.level_energy_source,
            "level_names": list(self.levels.names),
            "level_energies_J": self.level_energies_J.tolist(),
            "level_energy_shape_valid": True,
            "temporal_applicability": "instantaneous_rate_evaluation",
            "included_event_types": [
                "Pump absorption",
                "Pump stimulated emission",
                "linear decay",
                "nonlinear stoichiometric process",
            ],
            "missing_decay_partitions": missing_decay,
            "missing_nonlinear_partitions": missing_nonlinear,
            "complete_partitions": not missing_decay and not missing_nonlinear,
            "require_complete_partitions": self.require_complete,
            "pump_optical_identity": (
                "absorbed optical - stimulated optical = stored excitation rate + immediate lattice heat"
            ),
            "maximum_pointwise_pump_identity_residual_W_m3": float(
                np.max(np.abs(pointwise_optical_residual))
            ),
            "maximum_pointwise_full_identity_residual_W_m3": float(
                np.max(np.abs(full_identity_residual))
            ),
            "full_identity": (
                "net Pump optical input = all declared event destinations "
                "+ modeled level-storage derivative"
            ),
            "spectral_diffusion_energy_exchange": (
                self.levels.spectral_diffusion_energy_exchange_diagnostics(
                    self.level_energies_J
                )
            ),
        }
        if grid is not None:
            destination_power = {
                name: float(grid.integrate_volume(field))
                for name, field in destination.items()
            }
            stats.update(
                {
                    "destination_power_W": destination_power,
                    "event_count_rate_s": {
                        name: float(grid.integrate_volume(field))
                        for name, field in rates.items()
                    },
                    "event_energy_rate_W": {
                        name: float(grid.integrate_volume(field))
                        for name, field in energy_rate_by_event.items()
                    },
                    "heat_power_W": destination_power["heat"],
                    "pump_absorbed_optical_power_W": float(
                        grid.integrate_volume(optical_input)
                    ),
                    "pump_stimulated_optical_power_W": float(
                        grid.integrate_volume(optical_output)
                    ),
                    "pump_stored_energy_rate_W": float(
                        grid.integrate_volume(stored_rate)
                    ),
                    "pump_identity_residual_W": float(
                        grid.integrate_volume(pointwise_optical_residual)
                    ),
                    "modeled_level_storage_derivative_W": float(
                        grid.integrate_volume(total_storage_derivative)
                    ),
                    "full_identity_residual_W": float(
                        grid.integrate_volume(full_identity_residual)
                    ),
                }
            )
            full_scale = max(
                abs(float(stats["pump_absorbed_optical_power_W"])),
                abs(float(stats["pump_stimulated_optical_power_W"])),
                abs(float(stats["modeled_level_storage_derivative_W"])),
                sum(abs(value) for value in destination_power.values()),
                np.finfo(float).tiny,
            )
            stats["full_identity_residual_relative"] = (
                abs(float(stats["full_identity_residual_W"])) / full_scale
            )
            destination_scale = max(
                sum(abs(value) for value in destination_power.values()),
                np.finfo(float).tiny,
            )
            stats["unresolved_destination_fraction"] = (
                abs(float(destination_power["unresolved"]))
                / destination_scale
            )
        return EventEnergyResult(heat, destination, rates, stats)
