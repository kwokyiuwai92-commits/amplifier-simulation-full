"""Moment-matched and independently propagated Hermite--Gaussian modes.

The retained reduced path converts a truncated incoherent ensemble into one
sampled scalar intensity with effective-M2 diffraction.  The high-fidelity
path exposes signed normalized HG fields so each mode can propagate as an
independent incoherent channel while sharing the amplifier population.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def axis_focus_m(beam_cfg: dict[str, Any], axis: str) -> float:
    """Return an axis-specific waist position with legacy fallback."""

    value = beam_cfg.get(f"focus_{axis}_m")
    return float(beam_cfg.get("focus_z_m", 0.0) if value is None else value)


def rayleigh_range_m(waist_m: float, wavelength_m: float, m2: float, refractive_index: float = 1.0) -> float:
    return math.pi * float(refractive_index) * float(waist_m) ** 2 / (float(m2) * float(wavelength_m))


def divergence_half_angle_rad(
    waist_m: float,
    wavelength_m: float,
    m2: float,
    refractive_index: float = 1.0,
) -> float:
    """Paraxial far-field half angle for a 1/e2 second-moment radius."""

    return float(m2) * float(wavelength_m) / (
        math.pi * float(refractive_index) * float(waist_m)
    )


def maximum_entropy_mode_weights(m2: float, maximum_order: int) -> np.ndarray:
    """Return non-negative HG weights matching ``sum p_n(2n+1)=M2``.

    The maximum-entropy distribution avoids an arbitrary preference for two
    neighbouring modes.  A finite order N can represent 1 <= M2 <= 2N+1.
    """

    quality = float(m2)
    order = int(maximum_order)
    if order < 0:
        raise ValueError("maximum HG order must be non-negative")
    target = 0.5 * (quality - 1.0)
    tolerance = 1e-12
    if target < -tolerance or target > order + tolerance:
        raise ValueError(
            f"M2={quality:g} requires maximum HG order at least {math.ceil(max(target, 0.0))}; "
            f"configured order is {order}"
        )
    if order == 0 or target <= tolerance:
        result = np.zeros(order + 1, dtype=float)
        result[0] = 1.0
        return result
    if target >= order - tolerance:
        result = np.zeros(order + 1, dtype=float)
        result[-1] = 1.0
        return result

    indices = np.arange(order + 1, dtype=float)

    def weights(beta: float) -> np.ndarray:
        exponent = beta * indices
        exponent -= float(np.max(exponent))
        values = np.exp(exponent)
        return values / float(np.sum(values))

    low, high = -64.0, 64.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        candidate = weights(middle)
        if float(np.dot(candidate, indices)) < target:
            low = middle
        else:
            high = middle
    return weights(0.5 * (low + high))


def normalized_hg_fields(
    coordinate: np.ndarray, radius_m: float, maximum_order: int
) -> np.ndarray:
    """Sample signed, normalized 1-D HG fields from order zero through N."""

    radius = max(float(radius_m), np.finfo(float).tiny)
    argument = math.sqrt(2.0) * np.asarray(coordinate, dtype=float) / radius
    gaussian = np.exp(np.maximum(-argument**2, -745.0))
    fields = np.empty((int(maximum_order) + 1, *argument.shape), dtype=float)
    fields[0] = (2.0 / math.pi) ** 0.25 / math.sqrt(radius) * np.sqrt(gaussian)
    if maximum_order >= 1:
        fields[1] = math.sqrt(2.0) * argument * fields[0]
    for order in range(1, int(maximum_order)):
        fields[order + 1] = (
            math.sqrt(2.0 / (order + 1.0)) * argument * fields[order]
            - math.sqrt(order / (order + 1.0)) * fields[order - 1]
        )
    return fields


def _normalized_hg_intensities(
    coordinate: np.ndarray, radius_m: float, maximum_order: int
) -> np.ndarray:
    return normalized_hg_fields(coordinate, radius_m, maximum_order) ** 2


def single_hg_mode_amplitude(
    x: np.ndarray,
    y: np.ndarray,
    radius_x_m: float,
    radius_y_m: float,
    order_x: int,
    order_y: int,
    *,
    center_x_m: float = 0.0,
    center_y_m: float = 0.0,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """Return a signed, unit-integral separable HG mode amplitude."""

    theta = math.radians(float(rotation_deg))
    shifted_x = np.asarray(x, dtype=float) - float(center_x_m)
    shifted_y = np.asarray(y, dtype=float) - float(center_y_m)
    xr = shifted_x * math.cos(theta) + shifted_y * math.sin(theta)
    yr = -shifted_x * math.sin(theta) + shifted_y * math.cos(theta)
    field_x = normalized_hg_fields(xr, radius_x_m, int(order_x))[int(order_x)]
    field_y = normalized_hg_fields(yr, radius_y_m, int(order_y))[int(order_y)]
    return field_x * field_y


def incoherent_hg_modes(beam_cfg: dict[str, Any]) -> list[dict[str, float | int]]:
    """Return nonzero HG mode weights whose ensemble matches configured M2."""

    modes = beam_cfg.get("spatial_modes", {})
    maximum_x = int(modes.get("maximum_order_x", 0))
    maximum_y = int(modes.get("maximum_order_y", 0))
    maximum_total = modes.get("maximum_total_order")
    if maximum_total is not None:
        maximum_total = int(maximum_total)
        if maximum_total < 0:
            raise ValueError("maximum total HG order must be non-negative")
        pairs = [
            (order_x, order_y)
            for order_x in range(maximum_x + 1)
            for order_y in range(maximum_y + 1)
            if order_x + order_y <= maximum_total
        ]
        target = np.asarray(
            [
                0.5 * (float(beam_cfg.get("m2_x", 1.0)) - 1.0),
                0.5 * (float(beam_cfg.get("m2_y", 1.0)) - 1.0),
            ],
            dtype=float,
        )
        if np.any(target < -1e-12) or float(np.sum(target)) > maximum_total + 1e-12:
            raise ValueError(
                "configured M2 cannot be represented by the total-order HG basis"
            )
        if target[0] <= 1e-12:
            pairs = [pair for pair in pairs if pair[0] == 0]
            target[0] = 0.0
        if target[1] <= 1e-12:
            pairs = [pair for pair in pairs if pair[1] == 0]
            target[1] = 0.0
        coordinates = np.asarray(pairs, dtype=float)
        beta = np.zeros(2, dtype=float)
        for _ in range(100):
            logits = coordinates @ beta
            logits -= float(np.max(logits))
            weights = np.exp(logits)
            weights /= float(np.sum(weights))
            mean = weights @ coordinates
            residual = mean - target
            if float(np.max(np.abs(residual))) < 1e-13:
                break
            centered = coordinates - mean[None, :]
            covariance = (centered * weights[:, None]).T @ centered
            step = np.linalg.lstsq(
                covariance + 1e-14 * np.eye(2),
                residual,
                rcond=None,
            )[0]
            beta -= np.clip(step, -8.0, 8.0)
        else:
            raise ValueError(
                "maximum-entropy total-order HG weights did not converge"
            )
        modal_weights = {
            pair: float(weight) for pair, weight in zip(pairs, weights, strict=True)
        }
    else:
        weights_x = maximum_entropy_mode_weights(
            float(beam_cfg.get("m2_x", 1.0)), maximum_x
        )
        weights_y = maximum_entropy_mode_weights(
            float(beam_cfg.get("m2_y", 1.0)), maximum_y
        )
        modal_weights = {
            (order_x, order_y): float(weight_x * weight_y)
            for order_x, weight_x in enumerate(weights_x)
            for order_y, weight_y in enumerate(weights_y)
        }
    result: list[dict[str, float | int]] = []
    for (order_x, order_y), weight in modal_weights.items():
        if weight > 32.0 * np.finfo(float).eps:
            result.append(
                {
                    "order_x": order_x,
                    "order_y": order_y,
                    "weight": weight,
                }
            )
    total = sum(float(item["weight"]) for item in result)
    for item in result:
        item["weight"] = float(item["weight"]) / total
    return result


def axis_aligned_q_anisotropy_reasons(
    beam_cfg: dict[str, Any],
) -> tuple[str, ...]:
    """Return Seed features that require coupled/rotated transverse q optics.

    Commercial App's reduced spatial state stores independent lab-frame ``qx`` and
    ``qy`` values.  A non-zero source rotation is therefore harmless only when
    every represented transverse channel is rotationally symmetric.  Looking
    solely at the two waist radii misses astigmatic focus, anisotropic M2,
    wavelength-dependent axis scaling, and anisotropic HG channels.
    """

    reasons: list[str] = []

    def differs(left: float, right: float) -> bool:
        return not np.isclose(
            float(left),
            float(right),
            rtol=1e-12,
            atol=1e-15,
        )

    if differs(beam_cfg.get("waist_x_m", 0.0), beam_cfg.get("waist_y_m", 0.0)):
        reasons.append("waist_x_m != waist_y_m")
    if differs(beam_cfg.get("m2_x", 1.0), beam_cfg.get("m2_y", 1.0)):
        reasons.append("m2_x != m2_y")
    if differs(axis_focus_m(beam_cfg, "x"), axis_focus_m(beam_cfg, "y")):
        reasons.append("focus_x_m != focus_y_m")

    spatio_spectral = beam_cfg.get("spatio_spectral", {})
    for stem in (
        "waist_scaling_exponent",
        "m2_scaling_exponent",
        "focus_shift_m_per_fraction",
    ):
        if stem == "focus_shift_m_per_fraction":
            left = spatio_spectral.get("focus_shift_x_m_per_fraction", 0.0)
            right = spatio_spectral.get("focus_shift_y_m_per_fraction", 0.0)
            label = "wavelength-dependent focus shift differs by axis"
        else:
            left = spatio_spectral.get(f"{stem}_x", 0.0)
            right = spatio_spectral.get(f"{stem}_y", 0.0)
            label = f"wavelength-dependent {stem} differs by axis"
        if differs(left, right):
            reasons.append(label)

    if (
        str(beam_cfg.get("spatial_modes", {}).get("model", "effective_m2"))
        == "incoherent_hg"
    ):
        modes = incoherent_hg_modes(beam_cfg)
        if any(
            int(mode["order_x"]) != int(mode["order_y"])
            for mode in modes
        ):
            reasons.append("at least one HG channel has order_x != order_y")

    return tuple(reasons)


def effective_hg_density(
    x: np.ndarray,
    y: np.ndarray,
    radius_x_m: float,
    radius_y_m: float,
    m2_x: float,
    m2_y: float,
    maximum_order_x: int,
    maximum_order_y: int,
    *,
    center_x_m: float = 0.0,
    center_y_m: float = 0.0,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """Return the incoherent HG ensemble intensity on a 2D mesh.

    The fundamental HG scale is divided by sqrt(M2), so the mixture has the
    same x/y second moments as a Gaussian with the configured radii.
    """

    theta = math.radians(float(rotation_deg))
    shifted_x = np.asarray(x, dtype=float) - float(center_x_m)
    shifted_y = np.asarray(y, dtype=float) - float(center_y_m)
    xr = shifted_x * math.cos(theta) + shifted_y * math.sin(theta)
    yr = -shifted_x * math.sin(theta) + shifted_y * math.cos(theta)
    wx = float(radius_x_m) / math.sqrt(float(m2_x))
    wy = float(radius_y_m) / math.sqrt(float(m2_y))
    weights_x = maximum_entropy_mode_weights(float(m2_x), int(maximum_order_x))
    weights_y = maximum_entropy_mode_weights(float(m2_y), int(maximum_order_y))
    intensity_x = np.tensordot(weights_x, _normalized_hg_intensities(xr, wx, int(maximum_order_x)), axes=(0, 0))
    intensity_y = np.tensordot(weights_y, _normalized_hg_intensities(yr, wy, int(maximum_order_y)), axes=(0, 0))
    return intensity_x * intensity_y


def spatial_mode_summary(beam_cfg: dict[str, Any]) -> dict[str, Any]:
    modes = beam_cfg.get("spatial_modes", {})
    max_x = int(modes.get("maximum_order_x", 0))
    max_y = int(modes.get("maximum_order_y", 0))
    m2x = float(beam_cfg.get("m2_x", 1.0))
    m2y = float(beam_cfg.get("m2_y", 1.0))
    model = str(modes.get("model", "effective_m2"))
    propagated_modes = incoherent_hg_modes(beam_cfg) if model == "incoherent_hg" else []
    if propagated_modes:
        weights_x = np.zeros(max_x + 1, dtype=float)
        weights_y = np.zeros(max_y + 1, dtype=float)
        for mode in propagated_modes:
            weights_x[int(mode["order_x"])] += float(mode["weight"])
            weights_y[int(mode["order_y"])] += float(mode["weight"])
    elif model == "hermite_gaussian":
        weights_x = maximum_entropy_mode_weights(m2x, max_x)
        weights_y = maximum_entropy_mode_weights(m2y, max_y)
    else:
        # effective_m2 has no HG basis; dormant truncation settings must not
        # reject a valid beam while the solver is assembling its result.
        weights_x = weights_y = np.empty(0, dtype=float)
    return {
        "model": model,
        "maximum_order_x": max_x,
        "maximum_order_y": max_y,
        "maximum_total_order": modes.get("maximum_total_order"),
        "weights_x": weights_x.tolist() if model in {"hermite_gaussian", "incoherent_hg"} else [],
        "weights_y": weights_y.tolist() if model in {"hermite_gaussian", "incoherent_hg"} else [],
        "independently_propagated_modes": propagated_modes,
        "represented_m2_x": m2x,
        "represented_m2_y": m2y,
    }
