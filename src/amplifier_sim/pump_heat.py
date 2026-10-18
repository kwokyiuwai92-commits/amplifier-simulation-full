"""Pump-spectrum-resolved active heat fractions.

The already transported active absorbed-power density is split
locally according to the configured Pump spectrum and the local net active
absorption at each wavelength.  It therefore captures wavelength-dependent
cross sections and populations while keeping one shared spatial Pump field.
It is not a flashlamp ray tracer and is deliberately labelled as such.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import photon_energy
from .levels import LevelSystem


SPECTRAL_HEAT_MODELS = {
    "spectral_quantum_defect",
    "spectral_scaled_quantum_defect",
}


def lifetime_nonradiative_heat(
    levels: LevelSystem,
    population_m3: np.ndarray,
    radiative_lifetime_s: float,
    transition_wavelength_m: float,
    crystal_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return heat from the non-radiative part of upper-state decay.

    The population equations use the measured total upper-state lifetime,
    ``tau_total``.  An independently configured radiative lifetime,
    ``tau_rad``, then separates the total rate into

    ``1/tau_total = 1/tau_rad + 1/tau_nonradiative``.

    Only the non-radiative rate is deposited as heat here.  This is deliberately
    evaluated from the actual upper-state population rather than multiplying
    absorbed Pump power by a fluorescence-yield surrogate: stimulated
    extraction therefore reduces this heat automatically.
    """

    population = np.asarray(population_m3, dtype=float)
    if population.ndim != 4:
        raise ValueError(
            "lifetime non-radiative heat requires population shape "
            "(level,z,y,x)"
        )
    if population.shape[0] != len(levels.names):
        raise ValueError(
            "lifetime non-radiative heat population level count is inconsistent"
        )
    radiative = float(radiative_lifetime_s)
    wavelength = float(transition_wavelength_m)
    if not np.isfinite(radiative) or radiative <= 0.0:
        raise ValueError("upper-state radiative lifetime must be positive")
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError(
            "lifetime non-radiative heat transition wavelength must be positive"
        )

    if len(levels.signal_transitions) != 1:
        raise ValueError(
            "lifetime non-radiative heat uses one explicit transition energy "
            "and therefore requires exactly one Signal transition"
        )
    upper_indices = sorted(
        {transition.upper for transition in levels.signal_transitions}
    )
    if len(upper_indices) != 1:
        raise ValueError(
            "lifetime non-radiative heat requires exactly one shared Signal "
            "upper-state reservoir"
        )
    total = np.asarray(
        levels.signal_upper_total_lifetime_s(population.shape[1:]),
        dtype=float,
    )
    if total.shape != population.shape[1:]:
        raise ValueError(
            "local total upper-state lifetime must match the population grid"
        )
    if np.any(~np.isfinite(total)) or np.any(total <= 0.0):
        raise ValueError("local total upper-state lifetime must be positive")
    lifetime_tolerance = 64.0 * np.finfo(float).eps * max(
        radiative,
        float(np.max(total)),
        np.finfo(float).tiny,
    )
    if float(np.max(total)) > radiative + lifetime_tolerance:
        raise ValueError(
            "the total upper-state lifetime cannot exceed the configured "
            "radiative lifetime in lifetime-based non-radiative heating"
        )

    nonradiative_rate_s = np.maximum(
        1.0 / total - 1.0 / radiative,
        0.0,
    )
    upper_population = np.maximum(population[upper_indices[0]], 0.0)
    heat = upper_population * nonradiative_rate_s * photon_energy(wavelength)
    if crystal_mask is not None:
        mask = np.asarray(crystal_mask, dtype=bool)
        if mask.shape != heat.shape:
            raise ValueError(
                "lifetime non-radiative heat mask must match the population grid"
            )
        heat = np.where(mask, heat, 0.0)
        active_heat = heat[mask]
        active_total = total[mask]
        active_rate = nonradiative_rate_s[mask]
    else:
        active_heat = heat.ravel()
        active_total = total.ravel()
        active_rate = nonradiative_rate_s.ravel()

    quantum_yield = np.clip(total / radiative, 0.0, 1.0)
    active_yield = (
        quantum_yield[np.asarray(crystal_mask, dtype=bool)]
        if crystal_mask is not None
        else quantum_yield.ravel()
    )
    return heat, {
        "enabled": True,
        "model": "lifetime_rate_decomposition",
        "radiative_lifetime_s": radiative,
        "total_lifetime_s_min": (
            float(np.min(active_total)) if active_total.size else None
        ),
        "total_lifetime_s_max": (
            float(np.max(active_total)) if active_total.size else None
        ),
        "fluorescence_quantum_yield_min": (
            float(np.min(active_yield)) if active_yield.size else None
        ),
        "fluorescence_quantum_yield_max": (
            float(np.max(active_yield)) if active_yield.size else None
        ),
        "nonradiative_decay_rate_s_min": (
            float(np.min(active_rate)) if active_rate.size else None
        ),
        "nonradiative_decay_rate_s_max": (
            float(np.max(active_rate)) if active_rate.size else None
        ),
        "transition_wavelength_m": wavelength,
        "heat_density_W_m3_min": (
            float(np.min(active_heat)) if active_heat.size else 0.0
        ),
        "heat_density_W_m3_max": (
            float(np.max(active_heat)) if active_heat.size else 0.0
        ),
        "scope": (
            "N_upper * h*nu_signal * "
            "(1/tau_total(T, concentration) - 1/tau_radiative)"
        ),
    }


def pump_component_ledger_applicability(
    levels: LevelSystem,
    wavelengths_m: np.ndarray,
    spatial_shape: tuple[int, ...],
    minimum_active_absorption_m_inv: float | None,
    *,
    negative_absorption_tolerance_m_inv: float = 1e-12,
    level_gap_relative_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Audit the reduced Pump component-heat ledger's physical boundary.

    That ledger multiplies *net* active absorption by one heat fraction.  It
    is therefore valid only when Pump stimulated emission is absent and the
    runtime active coefficient is not negative.  Event-energy accounting is
    the supported alternative when either condition fails.
    """

    wavelengths = np.asarray(wavelengths_m, dtype=float)
    maximum_emission = 0.0
    emitting_transitions: list[str] = []
    for transition in levels.pump_transitions:
        _, sigma_em = levels.local_transition_cross_sections(
            transition,
            wavelengths,
            spatial_shape,
        )
        local_maximum = float(np.max(np.asarray(sigma_em, dtype=float)))
        maximum_emission = max(maximum_emission, local_maximum)
        if local_maximum > 0.0:
            emitting_transitions.append(transition.name)
    tolerance = float(negative_absorption_tolerance_m_inv)
    minimum_active = (
        None
        if minimum_active_absorption_m_inv is None
        else float(minimum_active_absorption_m_inv)
    )
    emission_absent = maximum_emission == 0.0
    active_absorption_nonnegative = (
        minimum_active is None or minimum_active >= -tolerance
    )
    configured_energies = np.asarray(
        levels.cfg.get("level_energies_J", []), dtype=float
    )
    level_gap_checked = bool(
        str(levels.cfg.get("model", "")) == "custom"
        and configured_energies.shape == (len(levels.names),)
    )
    level_gap_mismatches: list[dict[str, Any]] = []
    maximum_level_gap_relative_mismatch = 0.0
    if level_gap_checked:
        for transition in levels.pump_transitions:
            transition_wavelengths = (
                np.asarray([transition.center_wavelength_m], dtype=float)
                if transition.center_wavelength_m is not None
                else np.asarray([float(np.median(wavelengths))], dtype=float)
            )
            configured_gap = float(
                configured_energies[transition.upper]
                - configured_energies[transition.lower]
            )
            reference_energies = np.asarray(
                photon_energy(transition_wavelengths), dtype=float
            )
            relative = np.abs(reference_energies - configured_gap) / np.maximum(
                np.abs(reference_energies), np.finfo(float).tiny
            )
            local_maximum = float(np.max(relative))
            maximum_level_gap_relative_mismatch = max(
                maximum_level_gap_relative_mismatch, local_maximum
            )
            if (
                configured_gap <= 0.0
                or local_maximum > float(level_gap_relative_tolerance)
            ):
                level_gap_mismatches.append(
                    {
                        "transition": transition.name,
                        "configured_level_gap_J": configured_gap,
                        "reference_pump_photon_energy_min_J": float(
                            np.min(reference_energies)
                        ),
                        "reference_pump_photon_energy_max_J": float(
                            np.max(reference_energies)
                        ),
                        "maximum_relative_mismatch": local_maximum,
                    }
                )
    level_gap_consistent = not level_gap_mismatches
    applicable = (
        emission_absent
        and active_absorption_nonnegative
        and level_gap_consistent
    )
    return {
        "applicable": applicable,
        "pump_stimulated_emission_absent": emission_absent,
        "maximum_pump_emission_cross_section_m2": maximum_emission,
        "emitting_transition_names": emitting_transitions,
        "minimum_active_absorption_m_inv": minimum_active,
        "negative_active_absorption_tolerance_m_inv": tolerance,
        "active_absorption_nonnegative": active_absorption_nonnegative,
        "custom_level_gap_checked": level_gap_checked,
        "custom_level_gap_consistent_with_pump_photon_energy": (
            level_gap_consistent
        ),
        "custom_level_gap_relative_tolerance": float(
            level_gap_relative_tolerance
        ),
        "maximum_custom_level_gap_relative_mismatch": (
            maximum_level_gap_relative_mismatch
        ),
        "custom_level_gap_mismatches": level_gap_mismatches,
        "recommended_alternative": (
            None if applicable else "event_energy_ledger"
        ),
    }


def spectral_active_heat_fraction(
    levels: LevelSystem,
    population_m3: np.ndarray,
    wavelengths_m: np.ndarray,
    power_weights: np.ndarray,
    reference_wavelength_m: float,
    model: str,
    extra_heat_fraction: float = 0.0,
    stokes_shift_multiplier: float = 1.0,
    crystal_mask: np.ndarray | None = None,
    *,
    spectral_active_absorbed_W_m3: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a local active-absorption heat fraction on ``(z,y,x)``.

    Spectral weights are multiplied by the positive local net absorption
    ``sigma_abs*N_lower - sigma_em*N_upper``.  Where every wavelength has zero
    positive active absorption, the configured source power weights are used
    only as a finite diagnostic fallback; those cells normally multiply zero
    active absorbed power in the outer energy ledger.
    """

    if model not in SPECTRAL_HEAT_MODELS:
        raise ValueError(f"Unknown spectral Pump heat model {model!r}")
    population = np.asarray(population_m3, dtype=float)
    if population.ndim != 4:
        raise ValueError("spectral Pump heat requires population shape (level,z,y,x)")
    wavelengths = np.asarray(wavelengths_m, dtype=float)
    weights = np.asarray(power_weights, dtype=float)
    if wavelengths.ndim != 1 or weights.shape != wavelengths.shape or len(wavelengths) < 1:
        raise ValueError("Pump heat wavelengths and power weights must be matching non-empty vectors")
    if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("Pump heat wavelengths must be finite and positive")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("Pump heat power weights must be finite, non-negative, and non-zero")
    weights = weights / np.sum(weights)
    transported_absorption = (
        None if spectral_active_absorbed_W_m3 is None
        else np.asarray(spectral_active_absorbed_W_m3, dtype=float)
    )
    if transported_absorption is not None and (
        transported_absorption.shape != (len(wavelengths), *population.shape[1:])
        or not np.all(np.isfinite(transported_absorption))
    ):
        raise ValueError("Transported spectral absorption must match wavelengths and population grid")
    reference = float(reference_wavelength_m)
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("Pump heat reference wavelength must be positive")
    extra = float(extra_heat_fraction)
    multiplier = float(stokes_shift_multiplier)
    if not np.isfinite(extra) or not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("Pump spectral heat fractions and multiplier must be finite")

    group_references = {
        str(group["name"]): float(
            group.get("signal_center_wavelength_m") or reference
        )
        for group in levels.ensemble_contract.get("groups", [])
    }
    per_transition = []
    for transition in levels.pump_transitions:
        transition_reference = group_references.get(
            str(transition.ensemble), reference
        )
        if model == "spectral_quantum_defect":
            fractions = np.maximum(
                1.0
                - float(transition.quantum_efficiency)
                * wavelengths
                / transition_reference,
                0.0,
            )
        else:
            fractions = multiplier * np.maximum(
                1.0 - wavelengths / transition_reference,
                0.0,
            )
        per_transition.append(fractions + extra)
    per_transition_fraction = np.asarray(per_transition, dtype=float)
    if np.any(per_transition_fraction > 1.0 + 1e-12) or np.any(
        per_transition_fraction < -1e-12
    ):
        raise ValueError("Every spectrally resolved active heat fraction must lie in [0, 1]")
    per_transition_fraction = np.clip(
        per_transition_fraction, 0.0, 1.0
    )

    result = np.zeros(population.shape[1:], dtype=float)
    fallback = float(
        np.mean(np.sum(per_transition_fraction * weights[None, :], axis=1))
    )
    fallback_cells = 0
    for iz in range(population.shape[1]):
        denominator = np.zeros((len(wavelengths), *population.shape[-2:]), dtype=float)
        numerator = np.zeros_like(denominator)
        for transition_index, transition in enumerate(
            levels.pump_transitions
        ):
            lower = population[transition.lower, iz]
            upper = population[transition.upper, iz]
            sigma_abs, sigma_em = levels.local_transition_cross_sections(
                transition, wavelengths, lower.shape, iz
            )
            net_absorption = (
                sigma_abs * lower[None, ...]
                - sigma_em * upper[None, ...]
            )
            contributions = np.maximum(net_absorption, 0.0)
            denominator += contributions
            numerator += (
                contributions * per_transition_fraction[transition_index, :, None, None]
            )
        if transported_absorption is None:
            numerator = np.sum(numerator * weights[:, None, None], axis=0)
            denominator = np.sum(denominator * weights[:, None, None], axis=0)
        else:
            # Allocate each transported wavelength among its local channels,
            # then weight by the actual absorbed spectrum, not the source.
            wavelength_fractions = np.divide(
                numerator, denominator,
                out=np.broadcast_to(
                    np.mean(per_transition_fraction, axis=0)[:, None, None],
                    denominator.shape,
                ).copy(),
                where=denominator > np.finfo(float).tiny,
            )
            absorbed = np.maximum(transported_absorption[:, iz], 0.0)
            numerator = np.sum(absorbed * wavelength_fractions, axis=0)
            denominator = np.sum(absorbed, axis=0)
        valid = denominator > np.finfo(float).tiny
        result[iz] = np.divide(
            numerator,
            denominator,
            out=np.full_like(denominator, fallback),
            where=valid,
        )
        fallback_cells += int(np.count_nonzero(~valid))
    if crystal_mask is not None:
        mask = np.asarray(crystal_mask, dtype=bool)
        if mask.shape != result.shape:
            raise ValueError("Pump spectral heat mask must match the population grid")
        result = np.where(mask, result, 0.0)
        active_values = result[mask]
    else:
        active_values = result.ravel()
    return result, {
        "enabled": True,
        "model": model,
        "method": (
            "transported_per_wavelength_multi_transition_absorption_ledger"
            if transported_absorption is not None
            else "local_net_absorption_spectral_weighting"
        ),
        "spatial_spectrum_resolved": transported_absorption is not None,
        "wavelength_count": int(len(wavelengths)),
        "wavelength_min_m": float(np.min(wavelengths)),
        "wavelength_max_m": float(np.max(wavelengths)),
        "transition_names": [
            transition.name for transition in levels.pump_transitions
        ],
        "transition_quantum_efficiencies": [
            float(transition.quantum_efficiency)
            for transition in levels.pump_transitions
        ],
        "per_transition_per_wavelength_heat_fraction": (
            per_transition_fraction.tolist()
        ),
        "per_wavelength_heat_fraction": (
            per_transition_fraction[0].tolist()
            if len(per_transition_fraction) == 1
            else np.mean(per_transition_fraction, axis=0).tolist()
        ),
        "power_weights": weights.tolist(),
        "fallback_cell_count": fallback_cells,
        "fraction_min": float(np.min(active_values)) if active_values.size else 0.0,
        "fraction_max": float(np.max(active_values)) if active_values.size else 0.0,
        "fraction_mean": float(np.mean(active_values)) if active_values.size else 0.0,
        "scope": (
            "The shared transported Pump field is spectrally partitioned by local net active absorption. "
            "This is a broadband end-pump heat model, not non-sequential flashlamp ray tracing."
        ),
    }
