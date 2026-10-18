"""Canonical conversion between two-axis slope angles and ray directions."""

from __future__ import annotations

import math

import numpy as np


def axis_direction_components(
    angle_x_rad: float,
    angle_y_rad: float,
) -> tuple[float, float, float]:
    """Return the unit direction encoded by x/z and y/z slope angles.

    The public angle pair defines ``tan(angle_x)=s_x/s_z`` and
    ``tan(angle_y)=s_y/s_z``.  Converting through this one helper prevents the
    x and y carrier components from being treated as two independent
    one-dimensional sines when both angles are non-zero.
    """

    slope_x = math.tan(float(angle_x_rad))
    slope_y = math.tan(float(angle_y_rad))
    norm = math.sqrt(1.0 + slope_x * slope_x + slope_y * slope_y)
    return slope_x / norm, slope_y / norm, 1.0 / norm


def refract_axis_angles_rad(
    angle_x_rad: float,
    angle_y_rad: float,
    *,
    n_before: float,
    n_after: float,
) -> tuple[float, float]:
    """Apply vector Snell refraction to a canonical two-axis direction."""

    if n_before <= 0.0 or n_after <= 0.0:
        raise ValueError("Interface refractive indices must be positive")
    direction_x, direction_y, _ = axis_direction_components(
        angle_x_rad,
        angle_y_rad,
    )
    scale = float(n_before) / float(n_after)
    transmitted_x = scale * direction_x
    transmitted_y = scale * direction_y
    transverse_squared = transmitted_x * transmitted_x + transmitted_y * transmitted_y
    if transverse_squared >= 1.0:
        raise ValueError(
            "Configured interface direction has no propagating transmitted ray "
            "(total internal reflection)."
        )
    transmitted_z = math.sqrt(
        max(1.0 - transverse_squared, np.finfo(float).tiny)
    )
    return (
        math.atan2(transmitted_x, transmitted_z),
        math.atan2(transmitted_y, transmitted_z),
    )
