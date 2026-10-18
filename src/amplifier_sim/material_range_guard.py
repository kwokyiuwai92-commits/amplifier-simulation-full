"""Auditable range handling for tabulated material response data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MaterialRangeGuard:
    policy: str = "record"
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def reset(self) -> None:
        self.records.clear()

    def record(
        self,
        name: str,
        requested: np.ndarray | float,
        lower: float,
        upper: float,
        *,
        weights: np.ndarray | None = None,
        zeroed_out_of_range: bool = False,
    ) -> None:
        values = np.asarray(requested, dtype=float)
        if values.size == 0:
            return
        low_count = int(np.count_nonzero(values < lower))
        high_count = int(np.count_nonzero(values > upper))
        outside = (values < lower) | (values > upper)
        weighted_mass = 0.0
        weighted_outside = 0.0
        if weights is not None:
            weight_array = np.asarray(weights, dtype=float)
            if weight_array.shape != values.shape:
                raise ValueError(
                    f"material range weights for {name} must match requested coordinates"
                )
            if np.any(weight_array < 0.0) or not np.all(np.isfinite(weight_array)):
                raise ValueError(
                    f"material range weights for {name} must be finite and non-negative"
                )
            weighted_mass = float(np.sum(weight_array))
            weighted_outside = float(np.sum(weight_array[outside]))
        entry = self.records.setdefault(
            name,
            {
                "calls": 0,
                "samples": 0,
                "clamped_low": 0,
                "clamped_high": 0,
                "table_min": float(lower),
                "table_max": float(upper),
                "requested_min": float(np.min(values)),
                "requested_max": float(np.max(values)),
                "weighted_calls": 0,
                "weighted_requested_mass": 0.0,
                "weighted_out_of_range_mass": 0.0,
                "zeroed_spectral_mass": 0.0,
                "maximum_power_weighted_out_of_range_fraction": 0.0,
                "maximum_zeroed_spectral_power_fraction": 0.0,
            },
        )
        entry["calls"] += 1
        entry["samples"] += int(values.size)
        entry["clamped_low"] += low_count
        entry["clamped_high"] += high_count
        entry["requested_min"] = min(
            float(entry["requested_min"]), float(np.min(values))
        )
        entry["requested_max"] = max(
            float(entry["requested_max"]), float(np.max(values))
        )
        if weights is not None:
            entry["weighted_calls"] += 1
            entry["weighted_requested_mass"] += weighted_mass
            entry["weighted_out_of_range_mass"] += weighted_outside
            if zeroed_out_of_range:
                entry["zeroed_spectral_mass"] += weighted_outside
            call_denominator = max(weighted_mass, np.finfo(float).tiny)
            entry["maximum_power_weighted_out_of_range_fraction"] = max(
                float(entry["maximum_power_weighted_out_of_range_fraction"]),
                weighted_outside / call_denominator,
            )
            if zeroed_out_of_range:
                entry["maximum_zeroed_spectral_power_fraction"] = max(
                    float(entry["maximum_zeroed_spectral_power_fraction"]),
                    weighted_outside / call_denominator,
                )
            entry["power_weighted_out_of_range_fraction"] = (
                float(entry["weighted_out_of_range_mass"])
                / max(float(entry["weighted_requested_mass"]), np.finfo(float).tiny)
            )
            entry["zeroed_spectral_power_fraction"] = (
                float(entry["zeroed_spectral_mass"])
                / max(float(entry["weighted_requested_mass"]), np.finfo(float).tiny)
            )
        if self.policy == "error" and (low_count or high_count):
            raise ValueError(
                f"material table {name} was requested outside [{lower:.9g}, {upper:.9g}]"
            )

    def summary(self) -> dict[str, Any]:
        total_low = sum(int(item["clamped_low"]) for item in self.records.values())
        total_high = sum(int(item["clamped_high"]) for item in self.records.values())
        weighted_total = sum(
            float(item.get("weighted_requested_mass", 0.0))
            for item in self.records.values()
        )
        weighted_outside = sum(
            float(item.get("weighted_out_of_range_mass", 0.0))
            for item in self.records.values()
        )
        zeroed_mass = sum(
            float(item.get("zeroed_spectral_mass", 0.0))
            for item in self.records.values()
        )
        maximum_weighted_fraction = max(
            (
                float(
                    item.get(
                        "maximum_power_weighted_out_of_range_fraction", 0.0
                    )
                )
                for item in self.records.values()
            ),
            default=0.0,
        )
        maximum_zeroed_fraction = max(
            (
                float(
                    item.get("maximum_zeroed_spectral_power_fraction", 0.0)
                )
                for item in self.records.values()
            ),
            default=0.0,
        )
        unweighted_only_out_of_range = sum(
            int(item["clamped_low"]) + int(item["clamped_high"])
            for item in self.records.values()
            if int(item.get("weighted_calls", 0)) == 0
        )
        return {
            "policy": self.policy,
            "clamped_low_total": total_low,
            "clamped_high_total": total_high,
            "out_of_range_total": total_low + total_high,
            "weighted_requested_mass": weighted_total,
            "weighted_out_of_range_mass": weighted_outside,
            "aggregate_power_weighted_out_of_range_fraction": (
                0.0
                if weighted_total <= 0.0
                else weighted_outside / weighted_total
            ),
            # Strict validation uses the worst source/table support rather
            # than an iteration-count-weighted aggregate that could dilute a
            # narrow invalid spectrum with many calls to another transition.
            "power_weighted_out_of_range_fraction": maximum_weighted_fraction,
            "aggregate_zeroed_spectral_power_fraction": (
                0.0 if weighted_total <= 0.0 else zeroed_mass / weighted_total
            ),
            "zeroed_spectral_power_fraction": maximum_zeroed_fraction,
            "unweighted_only_out_of_range_total": unweighted_only_out_of_range,
            "tables": {name: dict(value) for name, value in self.records.items()},
        }


def interpolate_clamped(
    requested: np.ndarray | float,
    coordinate: np.ndarray,
    values: np.ndarray,
    *,
    guard: MaterialRangeGuard | None = None,
    name: str = "table",
) -> np.ndarray:
    xp = np.asarray(coordinate, dtype=float)
    fp = np.asarray(values, dtype=float)
    if xp.ndim != 1 or fp.ndim != 1 or len(xp) != len(fp) or len(xp) < 1:
        raise ValueError(f"material table {name} needs equal non-empty 1D arrays")
    if not np.all(np.isfinite(xp)) or not np.all(np.isfinite(fp)):
        raise ValueError(f"material table {name} contains non-finite data")
    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]
    if len(xp) > 1 and np.any(np.diff(xp) <= 0.0):
        raise ValueError(f"material table {name} coordinates must be unique")
    target = np.asarray(requested, dtype=float)
    if not np.all(np.isfinite(target)):
        raise ValueError(f"material table {name} received a non-finite coordinate")
    if guard is not None:
        guard.record(name, target, float(xp[0]), float(xp[-1]))
    return np.asarray(
        np.interp(target, xp, fp, left=fp[0], right=fp[-1]),
        dtype=float,
    )
