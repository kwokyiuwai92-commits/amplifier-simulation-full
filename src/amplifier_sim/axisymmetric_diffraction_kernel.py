"""Optional parallel Crank–Nicolson kernel for axisymmetric diffraction.

Each coherent frequency owns an independent radial tridiagonal system.  The
radial recurrence remains ordered, while Numba distributes the thousands of
frequency systems across CPU cores.  The kernel is deliberately independent
of gain, thermal, Kerr, and population models; those operators retain their
existing coupling order around this pure diffraction step.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # core-only installation
    njit = None
    prange = range


NUMBA_AVAILABLE = njit is not None


def numba_kernel_compiled_in_process() -> bool:
    if not NUMBA_AVAILABLE:
        return False
    kernel = globals().get("_radial_cn_numba")
    return bool(getattr(kernel, "signatures", ()))


if NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True)
    def _radial_cn_numba(
        values: np.ndarray,
        alpha: np.ndarray,
        lower: np.ndarray,
        cprime: np.ndarray,
        inverse_denominator: np.ndarray,
        laplacian_lower: np.ndarray,
        laplacian_diagonal: np.ndarray,
        laplacian_upper: np.ndarray,
        radial_area: np.ndarray,
        forward: np.ndarray,
        output: np.ndarray,
        input_energy: np.ndarray,
        norm_correction: np.ndarray,
    ) -> None:
        polarization_count = values.shape[0]
        frequency_count = values.shape[1]
        radial_count = values.shape[2]
        tiny = np.finfo(np.float64).tiny

        for system_index in prange(polarization_count * frequency_count):
            polarization = system_index // frequency_count
            frequency = system_index - polarization * frequency_count
            before = 0.0

            for radial in range(radial_count):
                value = values[polarization, frequency, radial]
                magnitude = abs(value)
                before += magnitude * magnitude * radial_area[radial]
                laplacian = laplacian_diagonal[radial] * value
                if radial > 0:
                    laplacian += (
                        laplacian_lower[radial]
                        * values[polarization, frequency, radial - 1]
                    )
                if radial + 1 < radial_count:
                    laplacian += (
                        laplacian_upper[radial]
                        * values[polarization, frequency, radial + 1]
                    )
                rhs = value + alpha[frequency] * laplacian
                if radial == 0:
                    forward[polarization, frequency, radial] = (
                        rhs * inverse_denominator[frequency, radial]
                    )
                else:
                    forward[polarization, frequency, radial] = (
                        rhs
                        - lower[frequency, radial - 1]
                        * forward[polarization, frequency, radial - 1]
                    ) * inverse_denominator[frequency, radial]

            output[polarization, frequency, radial_count - 1] = forward[
                polarization, frequency, radial_count - 1
            ]
            for radial in range(radial_count - 2, -1, -1):
                output[polarization, frequency, radial] = (
                    forward[polarization, frequency, radial]
                    - cprime[frequency, radial]
                    * output[polarization, frequency, radial + 1]
                )

            after = 0.0
            for radial in range(radial_count):
                magnitude = abs(output[polarization, frequency, radial])
                after += magnitude * magnitude * radial_area[radial]
            denominator = after
            if denominator < tiny:
                denominator = tiny
            scale = np.sqrt(before / denominator)
            input_energy[polarization, frequency] = before
            norm_correction[polarization, frequency] = abs(scale - 1.0)
            for radial in range(radial_count):
                output[polarization, frequency, radial] *= scale


def run_radial_cn_numba(
    *,
    values: np.ndarray,
    alpha: np.ndarray,
    lower: np.ndarray,
    cprime: np.ndarray,
    inverse_denominator: np.ndarray,
    laplacian_lower: np.ndarray,
    laplacian_diagonal: np.ndarray,
    laplacian_upper: np.ndarray,
    radial_area: np.ndarray,
    forward: np.ndarray,
    output: np.ndarray,
    input_energy: np.ndarray,
    norm_correction: np.ndarray,
) -> None:
    """Execute the compiled complex128/float64 radial solver."""

    if not NUMBA_AVAILABLE:
        raise RuntimeError(
            "The axisymmetric Numba diffraction kernel requires the "
            "performance extra"
        )
    kernel: Any = globals()["_radial_cn_numba"]
    kernel(
        values,
        alpha,
        lower,
        cprime,
        inverse_denominator,
        laplacian_lower,
        laplacian_diagonal,
        laplacian_upper,
        radial_area,
        forward,
        output,
        input_energy,
        norm_correction,
    )
