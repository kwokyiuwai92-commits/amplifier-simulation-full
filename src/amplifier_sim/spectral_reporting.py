"""Coordinate-safe reporting helpers for discrete optical spectra.

Solver spectra are stored as energy *masses* per numerical bin.  A line plot
of those masses is useful for debugging the native grid, but it is not a
spectral density and changes when the same field is resampled.  This module
performs the explicit bin-width/Jacobian conversion needed for comparison
with spectrometers and papers that plot a density versus wavelength.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import LIGHT_SPEED


def spectral_bin_coordinate_for_result(
    model_used: str,
    seed_config: dict[str, Any],
) -> str:
    """Return the coordinate whose finite bins own the reported energy masses."""

    if str(model_used) in {"coherent_cpa", "commercial_app_fast"}:
        # Both backends originate from the uniformly sampled coherent
        # frequency grid, even though their public plotting coordinate is λ.
        return "frequency"
    pulse_domain = str(
        seed_config.get("pulse_definition", {}).get("domain", "independent")
    )
    if pulse_domain in {"time", "spectrum", "dual"}:
        # Phase-insensitive transports conservatively rebin the linked pulse
        # masses on their wavelength centres before propagation.
        return "wavelength"
    configured = str(
        seed_config.get("spectrum", {}).get(
            "spectral_density_domain", "wavelength"
        )
    )
    return "frequency" if configured == "frequency" else "wavelength"


def coordinate_bin_edges(coordinate: np.ndarray) -> np.ndarray:
    """Return midpoint-cell edges for a finite, strictly increasing grid."""

    values = np.asarray(coordinate, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("spectral coordinate needs at least two samples")
    if np.any(~np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        raise ValueError("spectral coordinate must be finite and strictly increasing")
    differences = np.diff(values)
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * differences[0]
    edges[-1] = values[-1] + 0.5 * differences[-1]
    return edges


def spectral_density_from_bin_energy(
    wavelengths_m: np.ndarray,
    bin_energy_J: np.ndarray,
    *,
    bin_coordinate: str = "frequency",
    bin_edges: np.ndarray | None = None,
) -> dict[str, Any]:
    """Convert per-bin energy to wavelength- and frequency-domain densities.

    Returned wavelength arrays are ascending in wavelength; returned
    frequency arrays are ascending in frequency. ``bin_coordinate`` declares
    the coordinate on which the stored energy masses were discretized; the
    other coordinate uses the exact transformed finite-bin edges.  Each
    density therefore integrates back to the original energy.
    ``bin_edges``, when supplied, is ascending in ``bin_coordinate`` and owns
    the energy bins even when their centres are energy-weighted representatives.
    """

    wavelength = np.asarray(wavelengths_m, dtype=float)
    energy = np.asarray(bin_energy_J, dtype=float)
    if wavelength.ndim != 1 or energy.ndim != 1 or wavelength.shape != energy.shape:
        raise ValueError("wavelengths_m and bin_energy_J must be matching 1-D arrays")
    if wavelength.size == 0:
        raise ValueError("spectrum must contain at least one bin")
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("spectral wavelengths must be finite and positive")
    if np.any(~np.isfinite(energy)) or np.any(energy < 0.0):
        raise ValueError("spectral bin energies must be finite and non-negative")
    if bin_coordinate not in {"frequency", "wavelength"}:
        raise ValueError("bin_coordinate must be frequency or wavelength")
    supplied_edges = None if bin_edges is None else np.asarray(bin_edges, dtype=float)
    if supplied_edges is not None and (
        supplied_edges.shape != (len(wavelength) + 1,)
        or not np.all(np.isfinite(supplied_edges))
        or np.any(supplied_edges <= 0.0)
        or np.any(np.diff(supplied_edges) <= 0.0)
    ):
        raise ValueError("Explicit spectral bin edges must be positive and strictly increasing")

    frequency = LIGHT_SPEED / wavelength
    if wavelength.size == 1 and supplied_edges is None:
        # A single line has no inferable density bandwidth.  Preserve the
        # historical unit-width diagnostic while labelling it explicitly.
        frequency_sorted = frequency.copy()
        frequency_energy = energy.copy()
        wavelength_sorted = wavelength.copy()
        wavelength_energy = energy.copy()
        frequency_width_Hz = np.ones(1, dtype=float)
        wavelength_width_m = np.ones(1, dtype=float)
        width_policy = "unit_width_single_bin"
    elif bin_coordinate == "frequency":
        frequency_order = np.argsort(frequency)
        frequency_sorted = frequency[frequency_order]
        frequency_energy = energy[frequency_order]
        frequency_edges_Hz = (
            coordinate_bin_edges(frequency_sorted) if supplied_edges is None else supplied_edges
        )
        if frequency_edges_Hz[0] <= 0.0:
            raise ValueError(
                "inferred lower frequency-bin edge is non-positive"
            )
        frequency_width_Hz = np.diff(frequency_edges_Hz)
        # Each stored mass belongs to a frequency bin.  Transform the actual
        # frequency-bin edges, rather than constructing unrelated midpoint
        # cells in wavelength.  This is the finite-bin form of
        # dE/dlambda = dE/dnu * c/lambda^2 and remains accurate on coarse FFTs.
        wavelength_edges_by_frequency_m = LIGHT_SPEED / frequency_edges_Hz
        wavelength_width_m = (
            wavelength_edges_by_frequency_m[:-1]
            - wavelength_edges_by_frequency_m[1:]
        )[::-1]
        wavelength_sorted = (LIGHT_SPEED / frequency_sorted)[::-1]
        wavelength_energy = frequency_energy[::-1]
        width_policy = "frequency_midpoint_edges_transformed_exactly"
        if supplied_edges is not None:
            width_policy = "explicit_frequency_edges_transformed_exactly"
    else:
        wavelength_order = np.argsort(wavelength)
        wavelength_sorted = wavelength[wavelength_order]
        wavelength_energy = energy[wavelength_order]
        wavelength_edges_m = (
            coordinate_bin_edges(wavelength_sorted) if supplied_edges is None else supplied_edges
        )
        if wavelength_edges_m[0] <= 0.0:
            raise ValueError(
                "inferred lower wavelength-bin edge is non-positive"
            )
        wavelength_width_m = np.diff(wavelength_edges_m)
        frequency_edges_by_wavelength_Hz = LIGHT_SPEED / wavelength_edges_m
        frequency_width_Hz = (
            frequency_edges_by_wavelength_Hz[:-1]
            - frequency_edges_by_wavelength_Hz[1:]
        )[::-1]
        frequency_sorted = (LIGHT_SPEED / wavelength_sorted)[::-1]
        frequency_energy = wavelength_energy[::-1]
        width_policy = "wavelength_midpoint_edges_transformed_exactly"
        if supplied_edges is not None:
            width_policy = "explicit_wavelength_edges_transformed_exactly"

    density_J_per_Hz = frequency_energy / frequency_width_Hz
    density_J_per_m = wavelength_energy / wavelength_width_m

    total = float(np.sum(energy))
    wavelength_integral = float(np.sum(density_J_per_m * wavelength_width_m))
    frequency_integral = float(np.sum(density_J_per_Hz * frequency_width_Hz))
    scale = max(abs(total), np.finfo(float).tiny)

    def normalized(values: np.ndarray) -> np.ndarray:
        peak = float(np.max(values))
        return values / peak if peak > 0.0 else np.zeros_like(values)

    return {
        "wavelength_m": wavelength_sorted,
        "wavelength_bin_width_m": wavelength_width_m,
        "energy_by_wavelength_bin_J": wavelength_energy,
        "density_J_per_m": density_J_per_m,
        "density_J_per_nm": density_J_per_m * 1e-9,
        "normalized_wavelength_density": normalized(density_J_per_m),
        "frequency_Hz": frequency_sorted,
        "frequency_bin_width_Hz": frequency_width_Hz,
        "energy_by_frequency_bin_J": frequency_energy,
        "density_J_per_Hz": density_J_per_Hz,
        "density_J_per_THz": density_J_per_Hz * 1e12,
        "normalized_frequency_density": normalized(density_J_per_Hz),
        "total_energy_J": total,
        "wavelength_closure_relative": abs(wavelength_integral - total) / scale,
        "frequency_closure_relative": abs(frequency_integral - total) / scale,
        "bin_width_policy": width_policy,
    }
