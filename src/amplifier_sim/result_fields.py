"""Canonical NumPy field mapping shared by every result writer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .grid import Grid3D

if TYPE_CHECKING:
    from .solver import SimulationResult


REQUIRED_RESULT_ARRAYS = (
    "population_m3",
    "population_after_pump_m3",
    "inversion_after_pump_m3",
    "pump_intensity_W_m2",
    "pump_absorbed_W_m3",
    "signal_fluence_J_m2",
    "gain_m_inv",
    "initial_active_gain_m_inv",
    "remaining_active_gain_m_inv",
    "passive_loss_m_inv",
    "net_gain_m_inv",
    "heat_W_m3",
    "extracted_energy_J_m3",
    "temperature_K",
    "delta_n",
)


OPTIONAL_RESULT_ARRAYS = (
    "temporal_bin_edges_s",
    "spectral_bin_edges_Hz",
    "wavelengths_m",
    "input_spectrum_J",
    "output_spectrum_J",
    "time_s",
    "input_temporal_energy_J",
    "output_temporal_energy_J",
    "frequency_Hz",
    "input_spectral_phase_rad",
    "output_spectral_phase_rad",
    "compressed_spectral_phase_rad",
    "compressed_temporal_power_W",
    "crystal_mask",
    "pump_active_absorbed_W_m3",
    "pump_passive_absorbed_W_m3",
    "signal_fluence_last_pass_J_m2",
    "signal_fluence_cumulative_J_m2",
    "signal_fluence_per_pass_J_m2",
    "dopant_density_m3",
    "nonlinear_loss_energy_J_m3",
    "ase_intensity_W_m2",
    "signal_passive_loss_energy_J_m3",
    "output_jones_field",
    "pump_wavelengths_m",
    "pump_spectral_intensity_W_m2",
    "pump_spectral_active_absorbed_W_m3",
    "pump_spectral_passive_absorbed_W_m3",
    "longitudinal_intensity_W_m2",
    "longitudinal_intensity_area_m2",
    "longitudinal_temporal_power_W",
)


def grid_coordinate_arrays(grid: Grid3D, prefix: str = "") -> dict[str, np.ndarray]:
    """Return centres and edges for one complete three-dimensional grid."""

    return {
        f"{prefix}x_m": np.asarray(grid.x),
        f"{prefix}y_m": np.asarray(grid.y),
        f"{prefix}z_m": np.asarray(grid.z),
        f"{prefix}x_edges_m": np.asarray(grid.x_edges),
        f"{prefix}y_edges_m": np.asarray(grid.y_edges),
        f"{prefix}z_edges_m": np.asarray(grid.z_edges),
    }


def standard_result_arrays(result: "SimulationResult") -> dict[str, np.ndarray]:
    """Build the lossless standard field payload for directory, UI, and ZIP.

    Stress fields and optical planes have separate name/metadata contracts, but
    every solver-owned coordinate and field uses this single mapping.  This
    prevents one output format from silently omitting per-pass or cumulative
    Seed volumes that another format preserves.
    """

    arrays = grid_coordinate_arrays(result.grid)
    arrays.update(
        {
            name: np.asarray(getattr(result, name))
            for name in REQUIRED_RESULT_ARRAYS
        }
    )
    arrays.update(
        {
            name: np.asarray(value)
            for name in OPTIONAL_RESULT_ARRAYS
            if (value := getattr(result, name)) is not None
        }
    )
    if result.crystal_mask is None:
        arrays["crystal_mask"] = np.ones(result.grid.shape, dtype=np.uint8)
    else:
        arrays["crystal_mask"] = np.asarray(result.crystal_mask, dtype=np.uint8)
    if result.dopant_density_m3 is None:
        arrays["dopant_density_m3"] = np.sum(
            np.asarray(result.population_m3), axis=0
        )
    else:
        arrays["dopant_density_m3"] = np.asarray(result.dopant_density_m3)
    if result.optical_grid is not None:
        arrays.update(grid_coordinate_arrays(result.optical_grid, "optical_"))
    if result.pump_optical_grid is not None:
        arrays.update(
            grid_coordinate_arrays(result.pump_optical_grid, "pump_optical_")
        )
    return arrays
