"""Optional parallel solver for large batches of independent small systems."""

from __future__ import annotations

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
    kernel = globals().get("_solve_small_systems_numba")
    return bool(getattr(kernel, "signatures", ()))


if NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True)
    def _solve_small_systems_numba(
        matrices: np.ndarray,
        right_hand_sides: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        system_count = matrices.shape[0]
        order = matrices.shape[1]
        output = np.empty_like(right_hand_sides)
        failed = np.zeros(system_count, dtype=np.uint8)
        tiny = np.finfo(np.float64).tiny

        for system_index in prange(system_count):
            matrix = matrices[system_index].copy()
            rhs = right_hand_sides[system_index].copy()
            valid = True
            for pivot_index in range(order):
                pivot_row = pivot_index
                pivot_magnitude = abs(matrix[pivot_index, pivot_index])
                for row in range(pivot_index + 1, order):
                    candidate = abs(matrix[row, pivot_index])
                    if candidate > pivot_magnitude:
                        pivot_magnitude = candidate
                        pivot_row = row
                if not np.isfinite(pivot_magnitude) or pivot_magnitude <= tiny:
                    valid = False
                    break
                if pivot_row != pivot_index:
                    for column in range(order):
                        temporary = matrix[pivot_index, column]
                        matrix[pivot_index, column] = matrix[pivot_row, column]
                        matrix[pivot_row, column] = temporary
                    temporary = rhs[pivot_index]
                    rhs[pivot_index] = rhs[pivot_row]
                    rhs[pivot_row] = temporary
                for row in range(pivot_index + 1, order):
                    factor = matrix[row, pivot_index] / matrix[
                        pivot_index, pivot_index
                    ]
                    for column in range(pivot_index + 1, order):
                        matrix[row, column] -= (
                            factor * matrix[pivot_index, column]
                        )
                    rhs[row] -= factor * rhs[pivot_index]

            if not valid:
                failed[system_index] = 1
                for row in range(order):
                    output[system_index, row] = np.nan
                continue
            for row in range(order - 1, -1, -1):
                value = rhs[row]
                for column in range(row + 1, order):
                    value -= matrix[row, column] * output[
                        system_index, column
                    ]
                diagonal = matrix[row, row]
                if not np.isfinite(diagonal) or abs(diagonal) <= tiny:
                    valid = False
                    break
                output[system_index, row] = value / diagonal
            if not valid:
                failed[system_index] = 1
                for row in range(order):
                    output[system_index, row] = np.nan
        return output, failed


def solve_batched_small_systems(
    matrices: np.ndarray,
    right_hand_sides: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Solve ``(..., n, n) @ x = (..., n)`` and flag any failed system."""

    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is unavailable")
    matrix = np.asarray(matrices, dtype=float)
    rhs = np.asarray(right_hand_sides, dtype=float)
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrices must have shape (..., n, n)")
    if rhs.shape != matrix.shape[:-1]:
        raise ValueError("right_hand_sides must have shape (..., n)")
    order = matrix.shape[-1]
    flattened_matrix = np.ascontiguousarray(matrix.reshape((-1, order, order)))
    flattened_rhs = np.ascontiguousarray(rhs.reshape((-1, order)))
    kernel = globals()["_solve_small_systems_numba"]
    output, failed = kernel(flattened_matrix, flattened_rhs)
    return output.reshape(rhs.shape), bool(np.any(failed))
