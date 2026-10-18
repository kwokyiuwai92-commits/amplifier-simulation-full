"""Model-independent diagnostic quantities for amplifier simulations.

This module deliberately contains no UI code.  Propagators can report energy
accounting in a common form, while front ends remain free to choose slices,
volumes, or integral plots.
"""

from __future__ import annotations

import numpy as np


def transition_inversion_m3(
    population_m3: np.ndarray,
    upper_level: int,
    lower_level: int,
) -> np.ndarray:
    """Return the population-difference inversion ``N_upper - N_lower``."""

    return np.asarray(population_m3[upper_level] - population_m3[lower_level], dtype=float)


def active_signal_energy_transfer_J_m3(
    input_fluence_J_m2: np.ndarray,
    output_fluence_after_loss_J_m2: np.ndarray,
    passive_loss_m_inv: float | np.ndarray,
    optical_step_length_m: float,
    cell_length_m: float,
) -> np.ndarray:
    """Recover signed active-medium energy transfer from a propagation step.

    ``output_fluence_after_loss_J_m2`` is the field fluence after the symmetric
    passive-half / active / passive-half operator used by the signal solvers.
    The active input is therefore the original fluence after the first passive
    half-step, and the active output is the reported fluence with only the
    second half-step removed.  This distinction is essential for saturated
    gain, where the active map does not commute with passive attenuation.  The
    result is mapped to the physical z cell, so integrating it with the grid
    cell volume returns the net signal-energy change caused by the active
    transition.  Positive values mean extraction; negative values mean signal
    absorption by the active ions.
    """

    optical_length = float(optical_step_length_m)
    cell_length = float(cell_length_m)
    if optical_length <= 0.0 or cell_length <= 0.0:
        raise ValueError("Optical step and cell length must be positive")
    passive_loss = np.asarray(passive_loss_m_inv, dtype=float)
    half_transmission = np.exp(-0.5 * passive_loss * optical_length)
    half_transmission = np.maximum(
        half_transmission, np.finfo(float).tiny
    )
    active_output = (
        np.asarray(output_fluence_after_loss_J_m2, dtype=float)
        / half_transmission
    )
    active_input = (
        np.asarray(input_fluence_J_m2, dtype=float)
        * half_transmission
    )
    return (active_output - active_input) / cell_length
