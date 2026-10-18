"""Wavelength-dependent material and optical response helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import LIGHT_SPEED
from .material_range_guard import MaterialRangeGuard
from .units import wavelength_scale_to_m


def exact_frequency_fwhm_from_wavelength(
    center_wavelength_m: float,
    bandwidth_wavelength_m: float,
) -> float:
    """Map a wavelength-domain FWHM to its exact frequency-domain width.

    The conversion preserves the declared peak wavelength and the total
    wavelength-domain FWHM after mapping a frequency-symmetric profile through
    ``wavelength = c / frequency``.  The wavelength half-maximum points are
    therefore allowed to be asymmetric about the peak.
    """

    center = float(center_wavelength_m)
    bandwidth = float(bandwidth_wavelength_m)
    if bandwidth <= 0.0:
        return 0.0
    ratio = bandwidth / center
    return (
        2.0
        * LIGHT_SPEED
        * bandwidth
        / (center**2 * (np.sqrt(1.0 + ratio**2) + 1.0))
    )


@dataclass(frozen=True)
class CrossSectionCurve:
    wavelength_m: np.ndarray
    sigma_abs_m2: np.ndarray
    sigma_em_m2: np.ndarray
    out_of_range: str = "zero"

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "CrossSectionCurve | None":
        if not cfg:
            return None
        if cfg.get("samples"):
            rows = np.asarray(cfg["samples"], dtype=float)
            if rows.ndim != 2 or rows.shape[1] < 3:
                raise ValueError("Cross-section samples need wavelength, sigma_abs, sigma_em")
            wavelength, sigma_abs, sigma_em = rows[:, 0], rows[:, 1], rows[:, 2]
            wavelength = wavelength * wavelength_scale_to_m(
                cfg.get("wavelength_unit", "m")
            )
        else:
            wavelength = np.asarray(cfg.get("wavelength_m", []), dtype=float)
            sigma_abs = np.asarray(cfg.get("sigma_abs_m2", []), dtype=float)
            sigma_em = np.asarray(cfg.get("sigma_em_m2", []), dtype=float)
        if wavelength.size < 2 or not (wavelength.size == sigma_abs.size == sigma_em.size):
            raise ValueError("Cross-section curve arrays must have equal length >= 2")
        if (
            np.any(~np.isfinite(wavelength))
            or np.any(wavelength <= 0.0)
            or np.any(~np.isfinite(sigma_abs))
            or np.any(~np.isfinite(sigma_em))
            or np.any(sigma_abs < 0.0)
            or np.any(sigma_em < 0.0)
        ):
            raise ValueError(
                "Cross-section wavelengths must be positive and cross sections finite and non-negative"
            )
        order = np.argsort(wavelength)
        if np.any(np.diff(wavelength[order]) <= 0.0):
            raise ValueError("Cross-section wavelengths must be unique")
        out_of_range = str(cfg.get("out_of_range", "zero"))
        if out_of_range not in {"zero", "edge", "error"}:
            raise ValueError(
                "Cross-section out_of_range must be 'zero', 'edge', or 'error'"
            )
        return cls(
            wavelength[order],
            sigma_abs[order],
            sigma_em[order],
            out_of_range,
        )

    def values(
        self,
        wavelength_m: np.ndarray | float,
        *,
        guard: MaterialRangeGuard | None = None,
        name: str = "cross_section_spectrum",
    ) -> tuple[np.ndarray, np.ndarray]:
        wavelengths = np.asarray(wavelength_m, dtype=float)
        if guard is not None:
            guard.record(
                name,
                wavelengths,
                float(self.wavelength_m[0]),
                float(self.wavelength_m[-1]),
            )
        outside = (wavelengths < float(self.wavelength_m[0])) | (
            wavelengths > float(self.wavelength_m[-1])
        )
        if self.out_of_range == "error" and np.any(outside):
            requested_minimum = float(np.min(wavelengths[outside]))
            requested_maximum = float(np.max(wavelengths[outside]))
            raise ValueError(
                "Cross-section spectrum was evaluated outside its calibrated "
                f"range [{float(self.wavelength_m[0]):.9g}, "
                f"{float(self.wavelength_m[-1]):.9g}] m; outside requests span "
                f"[{requested_minimum:.9g}, {requested_maximum:.9g}] m"
            )
        if self.out_of_range == "edge":
            left_abs, right_abs = float(self.sigma_abs_m2[0]), float(self.sigma_abs_m2[-1])
            left_em, right_em = float(self.sigma_em_m2[0]), float(self.sigma_em_m2[-1])
        elif self.out_of_range == "zero":
            left_abs = right_abs = left_em = right_em = 0.0
        elif self.out_of_range == "error":
            # All requests were checked above; endpoint values only satisfy
            # np.interp's scalar API and are never used out of range.
            left_abs, right_abs = (
                float(self.sigma_abs_m2[0]),
                float(self.sigma_abs_m2[-1]),
            )
            left_em, right_em = (
                float(self.sigma_em_m2[0]),
                float(self.sigma_em_m2[-1]),
            )
        else:
            raise ValueError(
                "Cross-section out_of_range must be 'zero', 'edge', or 'error'"
            )
        absorption = np.interp(wavelengths, self.wavelength_m, self.sigma_abs_m2, left=left_abs, right=right_abs)
        emission = np.interp(wavelengths, self.wavelength_m, self.sigma_em_m2, left=left_em, right=right_em)
        return np.asarray(absorption, dtype=float), np.asarray(emission, dtype=float)


def response_value(element: dict[str, Any], key: str, wavelength_m: float, default: float) -> float:
    """Interpolate a wavelength-dependent scalar response on an optical element."""

    response = element.get("spectral_response")
    base_value = element.get(key, default)
    if base_value is None:
        base_value = default
    if not response:
        return float(base_value)
    wavelengths = np.asarray(response.get("wavelength_m", []), dtype=float)
    values = np.asarray(response.get(key, []), dtype=float)
    if wavelengths.size < 2 or wavelengths.size != values.size:
        return float(base_value)
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    values = values[order]
    policy = str(response.get("out_of_range", "error"))
    outside = (
        float(wavelength_m) < float(wavelengths[0])
        or float(wavelength_m) > float(wavelengths[-1])
    )
    if outside and policy == "error":
        raise ValueError(
            f"Optical spectral response {key} was requested at "
            f"{float(wavelength_m):.9g} m outside "
            f"[{float(wavelengths[0]):.9g}, {float(wavelengths[-1]):.9g}] m."
        )
    if policy == "edge":
        left = float(values[0])
        right = float(values[-1])
    elif policy == "zero":
        left = right = 0.0
    elif policy == "error":
        left = float(values[0])
        right = float(values[-1])
    else:
        raise ValueError(
            "Optical spectral-response out_of_range must be error, edge, or zero"
        )
    return float(
        np.interp(
            wavelength_m,
            wavelengths,
            values,
            left=left,
            right=right,
        )
    )
