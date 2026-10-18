"""Small, optional acceleration layer for FFT-heavy propagation.

The physics code remains NumPy-native.  SciPy's threaded pocketfft is used when
available and profitable; core-only installations continue to use NumPy.  This
module deliberately does not move individual z steps to a GPU: the gain and
population operators are NumPy arrays, so per-step CPU/GPU transfers generally
cost more than a 2D FFT at the grids used by the application.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
from contextvars import ContextVar
from functools import lru_cache
from typing import Any

import numpy as np

from .profiling import operation_profiler_enabled, profile_operation


_backend_config: ContextVar[tuple[str, int, int]] = ContextVar(
    "amplifier_sim_fft_backend_config",
    default=("auto", -1, 4096),
)
_scipy_fft: Any | None = None
_scipy_checked = False


def configure_compute_backend(config: dict[str, Any] | None = None) -> None:
    """Configure the current execution context's FFT executor."""

    supplied = config or {}
    requested_backend = str(supplied.get("fft_backend", "auto"))
    configured_workers = int(supplied.get("fft_workers", 0))
    workers = -1 if configured_workers == 0 else configured_workers
    minimum_elements = int(supplied.get("fft_min_elements", 4096))
    _backend_config.set((requested_backend, workers, minimum_elements))


def _load_scipy_fft() -> Any | None:
    global _scipy_fft, _scipy_checked
    if _scipy_checked:
        return _scipy_fft
    _scipy_checked = True
    try:
        from scipy import fft as scipy_fft
    except ImportError:
        _scipy_fft = None
    else:
        _scipy_fft = scipy_fft
    return _scipy_fft


def selected_fft_backend() -> str:
    requested_backend, _, _ = _backend_config.get()
    if requested_backend == "numpy":
        return "numpy"
    scipy_fft = _load_scipy_fft()
    if scipy_fft is not None:
        return "scipy_pocketfft"
    return "numpy_fallback" if requested_backend == "scipy" else "numpy"


def _execute_fft2(array: np.ndarray, *, inverse: bool) -> np.ndarray:
    requested_backend, workers, minimum_elements = _backend_config.get()
    scipy_fft = None if requested_backend == "numpy" else _load_scipy_fft()
    if scipy_fft is not None and array.size >= minimum_elements:
        transform = scipy_fft.ifft2 if inverse else scipy_fft.fft2
        return transform(array, axes=(-2, -1), workers=workers)
    transform = np.fft.ifft2 if inverse else np.fft.fft2
    return transform(array, axes=(-2, -1))


def fft2(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if not operation_profiler_enabled():
        return _execute_fft2(array, inverse=False)
    with profile_operation("fft.spatial.forward", array=array):
        return _execute_fft2(array, inverse=False)


def ifft2(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if not operation_profiler_enabled():
        return _execute_fft2(array, inverse=True)
    with profile_operation("fft.spatial.inverse", array=array):
        return _execute_fft2(array, inverse=True)


def _execute_fft1(
    array: np.ndarray,
    *,
    axis: int,
    inverse: bool,
    norm: str | None,
) -> np.ndarray:
    requested_backend, workers, minimum_elements = _backend_config.get()
    scipy_fft = None if requested_backend == "numpy" else _load_scipy_fft()
    if scipy_fft is not None and array.size >= minimum_elements:
        transform = scipy_fft.ifft if inverse else scipy_fft.fft
        return transform(array, axis=axis, norm=norm, workers=workers)
    transform = np.fft.ifft if inverse else np.fft.fft
    return transform(array, axis=axis, norm=norm)


def fft1(
    values: np.ndarray,
    *,
    axis: int = -1,
    norm: str | None = None,
) -> np.ndarray:
    """One-dimensional FFT using the same context-local backend as spatial FFTs."""

    array = np.asarray(values)
    if not operation_profiler_enabled():
        return _execute_fft1(array, axis=axis, inverse=False, norm=norm)
    with profile_operation("fft.temporal.forward", array=array):
        return _execute_fft1(array, axis=axis, inverse=False, norm=norm)


def ifft1(
    values: np.ndarray,
    *,
    axis: int = -1,
    norm: str | None = None,
) -> np.ndarray:
    """One-dimensional inverse FFT using the configured context-local backend."""

    array = np.asarray(values)
    if not operation_profiler_enabled():
        return _execute_fft1(array, axis=axis, inverse=True, norm=norm)
    with profile_operation("fft.temporal.inverse", array=array):
        return _execute_fft1(array, axis=axis, inverse=True, norm=norm)


@lru_cache(maxsize=1)
def _host_is_apple_silicon() -> bool:
    if sys.platform != "darwin":
        return False
    if platform.machine().lower() in {"arm64", "aarch64"}:
        return True
    # Under Rosetta, platform.machine()/uname report x86_64 but the Darwin
    # kernel build string still identifies the native ARM64 kernel.  This also
    # works in restricted environments where reading sysctl is denied.
    if "ARM64" in platform.uname().version.upper():
        return True
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.optional.arm64"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and completed.stdout.strip() == "1"


def _mlx_available() -> bool:
    try:
        return importlib.util.find_spec("mlx.core") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def compute_backend_info() -> dict[str, Any]:
    requested_backend, workers, minimum_elements = _backend_config.get()
    process_architecture = platform.machine().lower()
    host_apple_silicon = _host_is_apple_silicon()
    native_apple_silicon = host_apple_silicon and process_architecture in {"arm64", "aarch64"}
    mlx_available = _mlx_available()
    if not host_apple_silicon:
        gpu_reason = "host_not_detected_as_apple_silicon"
        gpu_note = "Apple Metal is unavailable on this host; CPU FFT is used."
    elif not native_apple_silicon:
        gpu_reason = "python_process_is_x86_64_under_rosetta"
        gpu_note = "MLX requires a native arm64 Python process; CPU FFT is used in this environment."
    elif not mlx_available:
        gpu_reason = "mlx_not_installed"
        gpu_note = "MLX is not installed; threaded SciPy FFT remains the optional accelerator."
    else:
        gpu_reason = "not_selected_precision_and_transfer_cost"
        gpu_note = (
            "The solver keeps complex128 fields and updates NumPy populations at every z step; "
            "the tested MLX path would reduce precision or add CPU/GPU transfers, so threaded SciPy FFT is used."
        )
    return {
        "fft_backend_requested": requested_backend,
        "fft_backend_selected": selected_fft_backend(),
        "fft_workers": "all_available" if workers == -1 else workers,
        "fft_minimum_array_elements": minimum_elements,
        "process_architecture": process_architecture,
        "host_apple_silicon": host_apple_silicon,
        "native_apple_silicon_python": native_apple_silicon,
        "running_under_rosetta": host_apple_silicon and not native_apple_silicon,
        "mlx_available": mlx_available,
        "apple_gpu_used": False,
        "apple_gpu_reason": gpu_reason,
        "apple_gpu_note": gpu_note,
        "cpu_count": os.cpu_count(),
    }


def compute_backend_warnings() -> list[str]:
    info = compute_backend_info()
    requested_backend, _, _ = _backend_config.get()
    warnings: list[str] = []
    if info["running_under_rosetta"]:
        warnings.append(
            "Python is x86_64 on an Apple-silicon Mac, so NumPy/SciPy run through Rosetta and MLX/Metal is unavailable; "
            "use a native arm64 Python environment to avoid Rosetta and to make MLX available, then benchmark that environment."
        )
    if requested_backend == "scipy" and info["fft_backend_selected"] == "numpy_fallback":
        warnings.append("SciPy FFT was requested but SciPy is not installed; propagation continued with NumPy FFT.")
    return warnings
