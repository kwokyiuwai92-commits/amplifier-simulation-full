"""Extensible pump and seed beam profiles."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .spatial_modes import (
    axis_focus_m,
    effective_hg_density,
    single_hg_mode_amplitude,
)


BeamFactory = Callable[[dict[str, Any]], "BeamProfile"]
BEAM_REGISTRY: dict[str, BeamFactory] = {}


def register_beam(name: str, factory: BeamFactory) -> None:
    """Register a user beam factory before constructing the simulator."""
    BEAM_REGISTRY[name] = factory


class BeamProfile:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    def density(self, x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
        """Return a transverse distribution normalized to integrate to one."""
        raise NotImplementedError


class SuperGaussianBeam(BeamProfile):
    """Astigmatic, rotated, separable super-Gaussian with an M2 caustic.

    ``order=1`` is Gaussian; increasing order approaches a top-hat profile.
    Waist is the 1/e^2 intensity radius. M2 controls the far-field radius by
    w(z)=w0*sqrt(1+(M2*lambda*(z-z0)/(pi*w0^2))^2).
    """

    def _radius(self, waist: float, m2: float, wavelength: float, z: float, axis: str = "x") -> float:
        focus = axis_focus_m(self.cfg, axis)
        refractive_index = float(
            self.cfg.get(
                "propagation_refractive_index",
                self.cfg.get("source_refractive_index", 1.0),
            )
        )
        if not math.isfinite(refractive_index) or refractive_index <= 0.0:
            raise ValueError("beam propagation refractive index must be positive and finite")
        return waist * math.sqrt(
            1.0
            + (
                m2
                * wavelength
                * (z - focus)
                / (math.pi * refractive_index * waist**2)
            )
            ** 2
        )

    @staticmethod
    def _integral_1d(radius: float, order: float) -> float:
        return radius / order * 2.0 ** (-1.0 / (2.0 * order)) * math.gamma(1.0 / (2.0 * order))

    def density(self, x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
        wavelength = float(self.cfg["wavelength_m"])
        wx = self._radius(float(self.cfg["waist_x_m"]), float(self.cfg.get("m2_x", 1.0)), wavelength, z, "x")
        wy = self._radius(float(self.cfg["waist_y_m"]), float(self.cfg.get("m2_y", 1.0)), wavelength, z, "y")
        angle_x = math.radians(float(self.cfg.get("incidence_angle_x_deg", 0.0)))
        angle_y = math.radians(float(self.cfg.get("incidence_angle_y_deg", 0.0)))
        cx = float(self.cfg.get("center_x_m", 0.0)) + (z - axis_focus_m(self.cfg, "x")) * math.tan(angle_x)
        cy = float(self.cfg.get("center_y_m", 0.0)) + (z - axis_focus_m(self.cfg, "y")) * math.tan(angle_y)
        theta = math.radians(float(self.cfg.get("rotation_deg", 0.0)))
        xr = (x - cx) * math.cos(theta) + (y - cy) * math.sin(theta)
        yr = -(x - cx) * math.sin(theta) + (y - cy) * math.cos(theta)
        mx = float(self.cfg.get("order_x", 1.0))
        my = float(self.cfg.get("order_y", 1.0))
        exponent = -2.0 * (np.abs(xr / wx) ** (2.0 * mx) + np.abs(yr / wy) ** (2.0 * my))
        norm = self._integral_1d(wx, mx) * self._integral_1d(wy, my)
        return np.exp(np.maximum(exponent, -745.0)) / norm


class RadialSuperGaussianBeam(SuperGaussianBeam):
    """Elliptic radial super-Gaussian ``exp(-2*r**p)``.

    ``p=2`` is Gaussian.  :class:`SuperGaussianBeam` uses a separable x/y
    definition; this class supplies the rotationally symmetric radial
    convention used by many measured diode and homogenized pump profiles.
    """

    def density(self, x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
        wavelength = float(self.cfg["wavelength_m"])
        wx = self._radius(float(self.cfg["waist_x_m"]), float(self.cfg.get("m2_x", 1.0)), wavelength, z, "x")
        wy = self._radius(float(self.cfg["waist_y_m"]), float(self.cfg.get("m2_y", 1.0)), wavelength, z, "y")
        angle_x = math.radians(float(self.cfg.get("incidence_angle_x_deg", 0.0)))
        angle_y = math.radians(float(self.cfg.get("incidence_angle_y_deg", 0.0)))
        cx = float(self.cfg.get("center_x_m", 0.0)) + (z - axis_focus_m(self.cfg, "x")) * math.tan(angle_x)
        cy = float(self.cfg.get("center_y_m", 0.0)) + (z - axis_focus_m(self.cfg, "y")) * math.tan(angle_y)
        theta = math.radians(float(self.cfg.get("rotation_deg", 0.0)))
        xr = (x - cx) * math.cos(theta) + (y - cy) * math.sin(theta)
        yr = -(x - cx) * math.sin(theta) + (y - cy) * math.cos(theta)
        # Equal x/y orders use elliptical radial contours. Unequal orders
        # fall back to the separable profile below; they are not averaged.
        sg_x = 2.0 * float(self.cfg.get("order_x", 1.0))
        sg_y = 2.0 * float(self.cfg.get("order_y", 1.0))
        if math.isclose(sg_x, sg_y, rel_tol=0.0, abs_tol=1e-14):
            sg = sg_x
            radial = np.sqrt((xr / wx) ** 2 + (yr / wy) ** 2)
            exponent = -2.0 * radial**sg
            norm = (
                2.0
                * math.pi
                * wx
                * wy
                / sg
                * 2.0 ** (-2.0 / sg)
                * math.gamma(2.0 / sg)
            )
            return np.exp(np.maximum(exponent, -745.0)) / norm
        # Independent x/y exponents use the separable generalization when the
        # profile is not rotationally symmetric.
        exponent = -2.0 * (
            np.abs(xr / wx) ** sg_x + np.abs(yr / wy) ** sg_y
        )
        norm = self._integral_1d(wx, sg_x / 2.0) * self._integral_1d(wy, sg_y / 2.0)
        return np.exp(np.maximum(exponent, -745.0)) / norm


class TopHatBeam(SuperGaussianBeam):
    def density(self, x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
        cfg = dict(self.cfg)
        cfg["order_x"] = cfg.get("order_x", 20.0) if float(cfg.get("order_x", 1.0)) > 1 else 20.0
        cfg["order_y"] = cfg.get("order_y", 20.0) if float(cfg.get("order_y", 1.0)) > 1 else 20.0
        return SuperGaussianBeam(cfg).density(x, y, z)


class AnnularBeam(SuperGaussianBeam):
    def density(self, x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
        wavelength = float(self.cfg["wavelength_m"])
        wx = self._radius(float(self.cfg["waist_x_m"]), float(self.cfg.get("m2_x", 1.0)), wavelength, z, "x")
        wy = self._radius(float(self.cfg["waist_y_m"]), float(self.cfg.get("m2_y", 1.0)), wavelength, z, "y")
        cx = float(self.cfg.get("center_x_m", 0.0)) + (z - axis_focus_m(self.cfg, "x")) * math.tan(math.radians(float(self.cfg.get("incidence_angle_x_deg", 0.0))))
        cy = float(self.cfg.get("center_y_m", 0.0)) + (z - axis_focus_m(self.cfg, "y")) * math.tan(math.radians(float(self.cfg.get("incidence_angle_y_deg", 0.0))))
        theta = math.radians(float(self.cfg.get("rotation_deg", 0.0)))
        xr = (x - cx) * math.cos(theta) + (y - cy) * math.sin(theta)
        yr = -(x - cx) * math.sin(theta) + (y - cy) * math.cos(theta)
        radial_squared = (xr / wx) ** 2 + (yr / wy) ** 2
        extras = self.cfg.get("extras", {})
        peak_ratio = float(extras.get("inner_radius_ratio", 0.5))
        ring_order = float(extras.get("ring_order", 2.0 * peak_ratio**2))
        norm = math.pi * wx * wy * math.gamma(ring_order + 1.0) / 2.0 ** (ring_order + 1.0)
        return radial_squared**ring_order * np.exp(np.maximum(-2.0 * radial_squared, -745.0)) / norm


class HermiteGaussianEnsembleBeam(SuperGaussianBeam):
    """Effective scalar intensity of a truncated incoherent HG ensemble."""

    def density(self, x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
        wavelength = float(self.cfg["wavelength_m"])
        m2x = float(self.cfg.get("m2_x", 1.0))
        m2y = float(self.cfg.get("m2_y", 1.0))
        wx = self._radius(float(self.cfg["waist_x_m"]), m2x, wavelength, z, "x")
        wy = self._radius(float(self.cfg["waist_y_m"]), m2y, wavelength, z, "y")
        angle_x = math.radians(float(self.cfg.get("incidence_angle_x_deg", 0.0)))
        angle_y = math.radians(float(self.cfg.get("incidence_angle_y_deg", 0.0)))
        cx = float(self.cfg.get("center_x_m", 0.0)) + (z - axis_focus_m(self.cfg, "x")) * math.tan(angle_x)
        cy = float(self.cfg.get("center_y_m", 0.0)) + (z - axis_focus_m(self.cfg, "y")) * math.tan(angle_y)
        modes = self.cfg.get("spatial_modes", {})
        return effective_hg_density(
            x,
            y,
            wx,
            wy,
            m2x,
            m2y,
            int(modes.get("maximum_order_x", 0)),
            int(modes.get("maximum_order_y", 0)),
            center_x_m=cx,
            center_y_m=cy,
            rotation_deg=float(self.cfg.get("rotation_deg", 0.0)),
        )


class HermiteGaussianSingleModeBeam(SuperGaussianBeam):
    """One signed-HG channel; intensity is exposed through BeamProfile."""

    def density(self, x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
        wavelength = float(self.cfg["wavelength_m"])
        wx = self._radius(
            float(self.cfg["waist_x_m"]), 1.0, wavelength, z, "x"
        )
        wy = self._radius(
            float(self.cfg["waist_y_m"]), 1.0, wavelength, z, "y"
        )
        angle_x = math.radians(float(self.cfg.get("incidence_angle_x_deg", 0.0)))
        angle_y = math.radians(float(self.cfg.get("incidence_angle_y_deg", 0.0)))
        cx = float(self.cfg.get("center_x_m", 0.0)) + (
            z - axis_focus_m(self.cfg, "x")
        ) * math.tan(angle_x)
        cy = float(self.cfg.get("center_y_m", 0.0)) + (
            z - axis_focus_m(self.cfg, "y")
        ) * math.tan(angle_y)
        modes = self.cfg.get("spatial_modes", {})
        amplitude = single_hg_mode_amplitude(
            x,
            y,
            wx,
            wy,
            int(modes.get("active_order_x", 0)),
            int(modes.get("active_order_y", 0)),
            center_x_m=cx,
            center_y_m=cy,
            rotation_deg=float(self.cfg.get("rotation_deg", 0.0)),
        )
        return amplitude**2


BEAM_REGISTRY.update(
    {
        "gaussian": SuperGaussianBeam,
        "super_gaussian": SuperGaussianBeam,
        "radial_super_gaussian": RadialSuperGaussianBeam,
        "top_hat": TopHatBeam,
        "annular": AnnularBeam,
    }
)


def make_beam(cfg: dict[str, Any]) -> BeamProfile:
    if cfg.get("spatial_modes", {}).get("model") == "single_hg_mode":
        return HermiteGaussianSingleModeBeam(cfg)
    if cfg.get("spatial_modes", {}).get("model") == "hermite_gaussian":
        return HermiteGaussianEnsembleBeam(cfg)
    name = cfg.get("profile", "super_gaussian")
    try:
        return BEAM_REGISTRY[name](cfg)
    except KeyError as exc:
        raise ValueError(f"Unknown beam profile {name!r}; registered: {sorted(BEAM_REGISTRY)}") from exc
