"""Optional fused CPU kernel for one-transition coherent gain sweeps.

The generic LevelSystem path remains authoritative for multiple optical
transitions and inhomogeneous ensembles.  This module specializes only the
independent-pixel, single-transition case: frequency packets remain strictly
ordered within each pixel, while Numba may run different pixels in parallel.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import get_num_threads, njit, prange, set_num_threads
except ImportError:  # core-only installation
    get_num_threads = None
    njit = None
    prange = range
    set_num_threads = None


NUMBA_AVAILABLE = njit is not None
_DEFAULT_NUMBA_THREADS = (
    int(get_num_threads())
    if NUMBA_AVAILABLE and get_num_threads is not None
    else None
)


def configure_numba_threads(requested: int) -> int | None:
    """Apply an explicit process-local thread limit and return the active count."""

    if not NUMBA_AVAILABLE:
        return None
    count = int(requested)
    if count < 0:
        raise ValueError("Numba thread count must be non-negative")
    target = (
        int(_DEFAULT_NUMBA_THREADS)
        if count == 0 and _DEFAULT_NUMBA_THREADS is not None
        else count
    )
    assert set_num_threads is not None
    assert _DEFAULT_NUMBA_THREADS is not None
    set_num_threads(min(target, int(_DEFAULT_NUMBA_THREADS)))
    assert get_num_threads is not None
    return int(get_num_threads())


def numba_kernel_compiled_in_process() -> bool:
    """Report whether this process has initialized the fused signature."""

    if not NUMBA_AVAILABLE:
        return False
    kernel = globals().get("_single_transition_sweep_numba")
    return bool(getattr(kernel, "signatures", ()))


if NUMBA_AVAILABLE:

    @njit(cache=True, inline="always")
    def _frantz_nodvik_scalar(
        fluence: float,
        exponent: float,
        saturation: float,
    ) -> float:
        scaled = max(fluence, 0.0) / saturation
        if scaled > 50.0:
            log_expm1 = scaled
        elif scaled > 0.0:
            log_expm1 = np.log(np.expm1(scaled))
        else:
            log_expm1 = -np.inf
        value = exponent + log_expm1
        if value > 0.0:
            logarithm = value + np.log1p(np.exp(-value))
        else:
            logarithm = np.log1p(np.exp(value))
        return saturation * logarithm


    @njit(cache=True, parallel=True)
    def _single_transition_sweep_numba(
        input_fluence_by_frequency: np.ndarray,
        correction: np.ndarray,
        population_upper: np.ndarray,
        population_lower: np.ndarray,
        sigma_abs: np.ndarray,
        sigma_em: np.ndarray,
        saturation: np.ndarray,
        material_domain: np.ndarray,
        photon_energies: np.ndarray,
        update_frequency: np.ndarray,
        total_population: np.ndarray,
        meaningful_threshold: np.ndarray,
        dz: float,
        cell_length_m: float,
        reverse_spectral_order: bool,
        use_saturation: bool,
        output_by_frequency: np.ndarray,
        output_weight: np.ndarray,
        minimum_scale_by_pixel: np.ndarray,
        limited_count_by_pixel: np.ndarray,
        maximum_change_by_pixel: np.ndarray,
        maximum_unsaturated_exponent_by_pixel: np.ndarray,
        clipping_by_pixel: np.ndarray,
    ) -> None:
        frequency_count = input_fluence_by_frequency.shape[0]
        ny = input_fluence_by_frequency.shape[1]
        nx = input_fluence_by_frequency.shape[2]
        tiny = np.finfo(np.float64).tiny
        limiter_roundoff = 64.0 * np.finfo(np.float64).eps

        for flat_index in prange(ny * nx):
            iy = flat_index // nx
            ix = flat_index - iy * nx
            upper = population_upper[iy, ix]
            lower = population_lower[iy, ix]
            local_minimum_scale = 1.0
            local_limited_count = 0
            local_maximum_change = 0.0
            local_maximum_unsaturated = -np.inf
            local_clipping = 0
            local_output_weight = 0.0
            correction_value = correction[iy, ix]
            domain_value = material_domain[iy, ix]
            total_value = total_population[iy, ix]
            threshold_value = meaningful_threshold[iy, ix]

            for sweep in range(2):
                descending = (
                    reverse_spectral_order if sweep == 0
                    else not reverse_spectral_order
                )
                for position in range(frequency_count):
                    iw = frequency_count - 1 - position if descending else position
                    effective_input = (
                        0.5
                        * input_fluence_by_frequency[iw, iy, ix]
                        * correction_value
                    )
                    exponent = (
                        (
                            sigma_em[iw, iy, ix] * upper
                            - sigma_abs[iw, iy, ix] * lower
                        )
                        * domain_value
                        * dz
                    )
                    saturation_value = saturation[iw, iy, ix]
                    valid_saturation = (
                        np.isfinite(saturation_value)
                        and saturation_value > 0.0
                    )
                    if not (use_saturation and valid_saturation):
                        if exponent > local_maximum_unsaturated:
                            local_maximum_unsaturated = exponent
                        if exponent > 50.0:
                            local_clipping = 1

                    if use_saturation and valid_saturation:
                        output = _frantz_nodvik_scalar(
                            effective_input,
                            exponent,
                            saturation_value,
                        )
                    else:
                        output = effective_input * np.exp(min(exponent, 50.0))

                    if use_saturation and update_frequency[iw]:
                        event_energy_volume = photon_energies[iw] * cell_length_m
                        requested = (output - effective_input) / event_energy_volume
                        required = abs(requested)
                        scale = 1.0
                        if required > tiny:
                            available = upper if requested > 0.0 else lower
                            scale = min(max(available / required, 0.0), 1.0)
                        meaningful = required > threshold_value
                        if meaningful:
                            local_minimum_scale = min(
                                local_minimum_scale, scale
                            )
                            if scale < 1.0 - limiter_roundoff:
                                local_limited_count += 1
                        applied = requested * scale
                        upper = max(upper - applied, 0.0)
                        lower = max(lower + applied, 0.0)
                        output = effective_input + applied * event_energy_volume
                        if total_value > tiny:
                            local_maximum_change = max(
                                local_maximum_change,
                                abs(applied) / total_value,
                            )

                    output = max(output, 0.0)
                    output_by_frequency[iw, iy, ix] += output
                    local_output_weight += output

            population_upper[iy, ix] = upper
            population_lower[iy, ix] = lower
            output_weight[iy, ix] += local_output_weight
            minimum_scale_by_pixel[iy, ix] = local_minimum_scale
            limited_count_by_pixel[iy, ix] = local_limited_count
            maximum_change_by_pixel[iy, ix] = local_maximum_change
            maximum_unsaturated_exponent_by_pixel[iy, ix] = (
                local_maximum_unsaturated
            )
            clipping_by_pixel[iy, ix] = local_clipping


def run_single_transition_sweep_numba(
    *,
    input_fluence_by_frequency: np.ndarray,
    correction: np.ndarray,
    population_upper: np.ndarray,
    population_lower: np.ndarray,
    sigma_abs: np.ndarray,
    sigma_em: np.ndarray,
    saturation: np.ndarray,
    material_domain: np.ndarray,
    photon_energies: np.ndarray,
    update_frequency: np.ndarray,
    total_population: np.ndarray,
    meaningful_threshold: np.ndarray,
    dz: float,
    cell_length_m: float,
    reverse_spectral_order: bool,
    use_saturation: bool,
    output_by_frequency: np.ndarray,
    output_weight: np.ndarray,
    minimum_scale_by_pixel: np.ndarray,
    limited_count_by_pixel: np.ndarray,
    maximum_change_by_pixel: np.ndarray,
    maximum_unsaturated_exponent_by_pixel: np.ndarray,
    clipping_by_pixel: np.ndarray,
) -> None:
    """Execute the compiled kernel or fail explicitly when it is unavailable."""

    if not NUMBA_AVAILABLE:
        raise RuntimeError(
            "Numba is not installed; install the performance extra or use "
            "single_transition_kernel=numpy"
        )
    kernel: Any = globals()["_single_transition_sweep_numba"]
    kernel(
        input_fluence_by_frequency,
        correction,
        population_upper,
        population_lower,
        sigma_abs,
        sigma_em,
        saturation,
        material_domain,
        photon_energies,
        update_frequency,
        total_population,
        meaningful_threshold,
        float(dz),
        float(cell_length_m),
        bool(reverse_spectral_order),
        bool(use_saturation),
        output_by_frequency,
        output_weight,
        minimum_scale_by_pixel,
        limited_count_by_pixel,
        maximum_change_by_pixel,
        maximum_unsaturated_exponent_by_pixel,
        clipping_by_pixel,
    )
