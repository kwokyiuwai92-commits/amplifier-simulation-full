"""Strict unit scales shared by every tabulated-input backend."""

from __future__ import annotations


WAVELENGTH_UNITS_TO_M = {
    "m": 1.0,
    "um": 1e-6,
    "nm": 1e-9,
}

TIME_UNITS_TO_S = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
    "ps": 1e-12,
    "fs": 1e-15,
}


def wavelength_scale_to_m(unit: object) -> float:
    """Return the wavelength scale or fail instead of silently assuming metres."""

    name = str(unit)
    try:
        return WAVELENGTH_UNITS_TO_M[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported wavelength unit {name!r}; use one of {sorted(WAVELENGTH_UNITS_TO_M)}"
        ) from exc


def time_scale_to_s(unit: object) -> float:
    """Return the time scale or fail instead of silently assuming seconds."""

    name = str(unit)
    try:
        return TIME_UNITS_TO_S[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported time unit {name!r}; use one of {sorted(TIME_UNITS_TO_S)}"
        ) from exc
