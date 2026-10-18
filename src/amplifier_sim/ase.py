"""Reduced-order and arbitrary-angle 3D ASE transport.

This is discrete-ordinates radiative transfer, not a coherent resonator-mode
solver.  The retained six-ray backend (+/- x, y, z) is the inexpensive screen;
the angular backend carries spontaneous photons on an arbitrary-direction
quadrature and estimates the reflected boundary operator's dominant mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .constants import photon_energy
from .errors import ModelInputError
from .ase_angular import AngularDiscreteOrdinates
from .grid import Grid3D
from .levels import LevelSystem


@dataclass
class ASEResult:
    wavelengths_m: np.ndarray
    spectral_intensity_W_m2: np.ndarray
    passive_heat_W_m3: np.ndarray
    escaped_power_W: float
    stats: dict[str, Any]
    warnings: list[str]


class ASETransport:
    """Solve stationary six-direction ASE transport for a fixed population."""

    def __init__(
        self,
        grid: Grid3D,
        levels: LevelSystem,
        material: dict[str, Any],
        cfg: dict[str, Any],
        default_wavelength_m: float,
        crystal_mask: np.ndarray,
    ):
        self.grid = grid
        self.levels = levels
        self.material = material
        self.cfg = cfg
        self.mask = np.asarray(crystal_mask, dtype=float)
        configured_wavelengths = cfg.get("wavelengths_m", [])
        self.wavelengths = np.asarray(
            configured_wavelengths or [default_wavelength_m], dtype=float
        )
        weights = np.asarray(cfg.get("spectral_weights", []), dtype=float)
        if weights.size == 0:
            weights = np.ones(len(self.wavelengths), dtype=float)
        if weights.shape != self.wavelengths.shape or np.any(weights < 0.0) or np.sum(weights) <= 0.0:
            raise ValueError("ase.spectral_weights must be non-negative and match wavelengths_m")
        self.weights = weights / np.sum(weights)
        self.face_reflectivity = {
            face: float(cfg.get("face_reflectivity", {}).get(face, 0.0))
            for face in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        }

    @property
    def enabled(self) -> bool:
        return str(self.cfg.get("model", "off")) in {
            "six_ray",
            "angular_discrete_ordinates",
        }

    def solve(self, population: np.ndarray) -> ASEResult:
        if not self.enabled:
            zeros = np.zeros((len(self.wavelengths), *self.grid.shape), dtype=float)
            return ASEResult(
                self.wavelengths.copy(),
                zeros,
                np.zeros(self.grid.shape, dtype=float),
                0.0,
                {"enabled": False, "model": "off"},
                [],
            )
        self.levels.record_spectral_support(
            "signal", self.wavelengths, self.weights
        )
        shape = self.grid.shape
        upper = np.asarray(
            self.levels.aggregate_signal_upper_population(population),
            dtype=float,
        )
        total_upper_lifetime = self.levels.signal_upper_total_lifetime_s(shape)
        configured_lifetime = self.cfg.get("radiative_lifetime_s")
        lifetime_config_path = None
        if configured_lifetime is not None:
            lifetime_config_path = "ase.radiative_lifetime_s"
        elif self.levels.cfg.get("upper_state_radiative_lifetime_s") is not None:
            configured_lifetime = self.levels.cfg[
                "upper_state_radiative_lifetime_s"
            ]
            lifetime_config_path = (
                "energy_levels.upper_state_radiative_lifetime_s"
            )
        if configured_lifetime is None:
            if not bool(self.cfg.get("use_upper_state_lifetime_as_radiative", True)):
                raise ModelInputError(
                    "ase.radiative_lifetime_s is required unless "
                    "use_upper_state_lifetime_as_radiative=true"
                )
            radiative_lifetime = total_upper_lifetime.copy()
            lifetime_source = "actual_signal_upper_total_lifetime_assumed_radiative"
        else:
            radiative_lifetime = np.full(shape, float(configured_lifetime), dtype=float)
            lifetime_source = "explicit_radiative_lifetime"
        # The measured/modelled upper-state lifetime contains every decay
        # channel: 1/tau_total = 1/tau_rad + 1/tau_nonrad + ....  Therefore an
        # independently supplied radiative lifetime cannot be shorter than
        # the local total lifetime.  Without this guard the ASE source could
        # remove more ions per second than the population model decays.
        lifetime_tolerance = 32.0 * np.finfo(float).eps
        inconsistent = radiative_lifetime < total_upper_lifetime * (
            1.0 - lifetime_tolerance
        )
        if np.any(inconsistent):
            raise ModelInputError(
                "the explicit radiative lifetime must be greater than or equal "
                "to the local total upper-state lifetime; got radiative lifetime "
                f"{float(np.min(radiative_lifetime)):.6g} s and maximum local "
                f"total lifetime {float(np.max(total_upper_lifetime)):.6g} s"
            )
        radiative_branching_fraction = np.clip(
            total_upper_lifetime / radiative_lifetime,
            0.0,
            1.0,
        )
        solid_angle_fraction = float(self.cfg.get("modeled_solid_angle_fraction", 1.0))
        angular_geometry = self._angular_geometry_diagnostics(solid_angle_fraction)
        active_gain = self.levels.signal_gain_spectrum_m_inv(
            population,
            self.wavelengths,
            0.0,
        ) * self.mask[None, ...]
        gain = self.levels.signal_gain_spectrum_m_inv(
            population,
            self.wavelengths,
            float(self.material.get("signal_passive_loss_m_inv", 0.0)),
        ) * self.mask[None, ...]
        emission = (
            upper[None, ...]
            / np.maximum(radiative_lifetime[None, ...], np.finfo(float).tiny)
            * photon_energy(self.wavelengths)[:, None, None, None]
            * self.weights[:, None, None, None]
            * solid_angle_fraction
            * self.mask[None, ...]
        )
        if str(self.cfg.get("model", "off")) == "angular_discrete_ordinates":
            return self._attach_energy_ledger(self._solve_angular(
                gain,
                emission,
                radiative_lifetime,
                total_upper_lifetime,
                radiative_branching_fraction,
                lifetime_source,
                lifetime_config_path,
                solid_angle_fraction,
                angular_geometry,
            ), emission, active_gain)
        spectral_intensity = np.zeros_like(emission)
        escaped_power = 0.0
        direction_iterations = 0
        direction_converged = True
        maximum_intensity = float(self.cfg.get("maximum_intensity_W_m2", 1e18))
        intensity_clamp_active = False
        positive_exponent_clipping_active = False
        maximum_cell_gain_exponent = -np.inf
        maximum_boundary_residual_relative = 0.0
        roundtrip: dict[str, float] = {}
        warnings: list[str] = []
        for iw in range(len(self.wavelengths)):
            for axis, (minus_face, plus_face) in enumerate(
                (("z_min", "z_max"), ("y_min", "y_max"), ("x_min", "x_max"))
            ):
                (
                    center,
                    escaped,
                    iterations,
                    converged,
                    clamped,
                    exponent_clipped,
                    cell_gain_exponent,
                    boundary_residual_relative,
                ) = self._axis_pair(
                    gain[iw],
                    emission[iw] / 6.0,
                    axis,
                    self.face_reflectivity[minus_face],
                    self.face_reflectivity[plus_face],
                    maximum_intensity,
                )
                spectral_intensity[iw] += center
                escaped_power += escaped
                direction_iterations = max(direction_iterations, iterations)
                direction_converged &= converged
                intensity_clamp_active |= clamped
                positive_exponent_clipping_active |= exponent_clipped
                maximum_cell_gain_exponent = max(
                    maximum_cell_gain_exponent,
                    cell_gain_exponent,
                )
                maximum_boundary_residual_relative = max(
                    maximum_boundary_residual_relative,
                    boundary_residual_relative,
                )
                line_gain = np.sum(
                    np.moveaxis(gain[iw], axis, 0)
                    * self._axis_widths(axis).reshape((-1,) + (1,) * (gain[iw].ndim - 1)),
                    axis=0,
                )
                loop = (
                    self.face_reflectivity[minus_face]
                    * self.face_reflectivity[plus_face]
                    * np.exp(np.minimum(2.0 * line_gain, 100.0))
                )
                roundtrip[f"{minus_face}<->{plus_face}"] = max(
                    roundtrip.get(f"{minus_face}<->{plus_face}", 0.0),
                    float(np.max(loop)),
                )
        if not direction_converged:
            warnings.append(
                "ASE boundary iteration did not converge; a reflected ray channel may be above its "
                "parasitic-lasing threshold or maximum_intensity_W_m2 may be active."
            )
        if intensity_clamp_active:
            warnings.append(
                "ASE intensity reached maximum_intensity_W_m2; the six-ray result is numerically "
                "clamped and must not be interpreted as an unconstrained stationary ASE solution."
            )
        if positive_exponent_clipping_active:
            warnings.append(
                "At least one positive ASE single-cell gain exponent exceeded 50 and was clipped "
                "before intensity limiting. Refine Nz/Nx/Ny or reduce the cell gain before using "
                "the spatial ASE distribution quantitatively."
            )
        above_threshold = [name for name, value in roundtrip.items() if value >= 1.0]
        if above_threshold:
            warnings.append(
                "Axis-aligned parasitic round-trip gain is >= 1 for "
                + ", ".join(above_threshold)
                + "; treat the six-ray result as a threshold/clamping approximation, not a resolved laser mode."
            )
        if angular_geometry.get("high_aspect_ratio_cylinder", False):
            warnings.append(
                "ASE uses equal angular weight for the three axis pairs in a high-aspect-ratio "
                "cylinder. The reported direct axial-cone fractions do not include sidewall TIR, "
                "surface scattering, cladding, or arbitrary oblique rays; treat "
                "modeled_solid_angle_fraction as an effective sensitivity parameter unless those "
                "boundary data are known."
            )
        passive_heat = (
            float(self.material.get("signal_passive_loss_m_inv", 0.0))
            * np.sum(spectral_intensity, axis=0)
            * self.mask
        )
        return self._attach_energy_ledger(ASEResult(
            wavelengths_m=self.wavelengths.copy(),
            spectral_intensity_W_m2=spectral_intensity,
            passive_heat_W_m3=passive_heat,
            escaped_power_W=float(escaped_power),
            stats={
                "enabled": True,
                "model": "six_ray_discrete_ordinates",
                "wavelengths_m": self.wavelengths.tolist(),
                "spectral_weights": self.weights.tolist(),
                "modeled_solid_angle_fraction": solid_angle_fraction,
                "angular_geometry": angular_geometry,
                "radiative_lifetime_source": lifetime_source,
                "radiative_lifetime_config_path": lifetime_config_path,
                "radiative_lifetime_s_min": float(np.min(radiative_lifetime)),
                "radiative_lifetime_s_max": float(np.max(radiative_lifetime)),
                "total_upper_state_lifetime_s_min": float(
                    np.min(total_upper_lifetime)
                ),
                "total_upper_state_lifetime_s_max": float(
                    np.max(total_upper_lifetime)
                ),
                "implied_radiative_branching_fraction_min": float(
                    np.min(radiative_branching_fraction)
                ),
                "implied_radiative_branching_fraction_max": float(
                    np.max(radiative_branching_fraction)
                ),
                "escaped_power_W": float(escaped_power),
                "boundary_iterations": direction_iterations,
                "boundary_converged": direction_converged,
                "boundary_residual_relative": (
                    maximum_boundary_residual_relative
                ),
                "intensity_clamp_active": intensity_clamp_active,
                "positive_cell_exponent_clipping_active": positive_exponent_clipping_active,
                "maximum_cell_gain_exponent": float(maximum_cell_gain_exponent),
                "configured_maximum_intensity_W_m2": maximum_intensity,
                "maximum_total_intensity_W_m2": float(np.max(np.sum(spectral_intensity, axis=0))),
                "parasitic_roundtrip_gain": roundtrip,
                "parasitic_threshold_exceeded": bool(above_threshold),
                "scope": (
                    "stationary six-direction radiative transfer with optional effective face "
                    "reflectivity; it is not a transverse eigenmode or wave-optics resonator solve"
                ),
            },
            warnings=warnings,
        ), emission, active_gain)

    def _attach_energy_ledger(
        self,
        result: ASEResult,
        emission_W_m3: np.ndarray,
        active_gain_m_inv: np.ndarray,
    ) -> ASEResult:
        """Close the stationary ASE source/gain/loss/escape power identity."""

        source = float(
            self.grid.integrate_volume(np.sum(emission_W_m3, axis=0))
        )
        stimulated = float(
            self.grid.integrate_volume(
                np.sum(
                    active_gain_m_inv * result.spectral_intensity_W_m2,
                    axis=0,
                )
            )
        )
        passive = float(self.grid.integrate_volume(result.passive_heat_W_m3))
        residual = source + stimulated - passive - float(result.escaped_power_W)
        scale = max(source, abs(stimulated), passive, abs(result.escaped_power_W), 1e-300)
        result.stats["energy_ledger_W"] = {
            "spontaneous_source": source,
            "stimulated_population_transfer": stimulated,
            "passive_absorption": passive,
            "escaped": float(result.escaped_power_W),
            "signed_residual": residual,
            "relative_residual": abs(residual) / scale,
            "identity": "spontaneous + stimulated = passive + escaped + residual",
        }
        return result

    def _solve_angular(
        self,
        gain: np.ndarray,
        emission: np.ndarray,
        radiative_lifetime: np.ndarray,
        total_upper_lifetime: np.ndarray,
        radiative_branching_fraction: np.ndarray,
        lifetime_source: str,
        lifetime_config_path: str | None,
        solid_angle_fraction: float,
        angular_geometry: dict[str, Any],
    ) -> ASEResult:
        angular_solver = AngularDiscreteOrdinates(
            self.grid,
            self.cfg,
            self.face_reflectivity,
        )
        spectral_intensity = np.zeros_like(emission)
        escaped_power = 0.0
        wavelength_stats: list[dict[str, Any]] = []
        warnings: list[str] = []
        for index in range(len(self.wavelengths)):
            result = angular_solver.solve(gain[index], emission[index])
            spectral_intensity[index] = result.intensity_W_m2
            escaped_power += result.escaped_power_W
            wavelength_stats.append(
                {
                    "wavelength_m": float(self.wavelengths[index]),
                    **result.stats,
                }
            )
            warnings.extend(result.warnings)
        boundary_converged = all(
            bool(item["boundary_converged"]) for item in wavelength_stats
        )
        clamp_active = any(
            bool(item["intensity_clamp_active"]) for item in wavelength_stats
        )
        exponent_clipped = any(
            bool(item["positive_cell_exponent_clipping_active"])
            for item in wavelength_stats
        )
        threshold_exceeded = any(
            bool(item["parasitic_threshold_exceeded"])
            for item in wavelength_stats
        )
        eigenmode_converged = all(
            bool(item["parasitic_eigenmode_converged"])
            for item in wavelength_stats
        )
        geometry_kind = str(
            self.material.get("geometry", {}).get("shape", "cuboid")
        )
        embedded_boundary_resolved = geometry_kind == "cuboid"
        if not embedded_boundary_resolved:
            warnings.append(
                "Angular discrete-ordinates ASE resolves arbitrary directions and Cartesian "
                "external-face reflections, but the selected non-cuboid crystal uses a voxel "
                "mask without body-fitted specular/TIR reflection. Use this result as an open-"
                "boundary angular estimate unless a body-fitted backend is supplied."
            )
        passive_heat = (
            float(self.material.get("signal_passive_loss_m_inv", 0.0))
            * np.sum(spectral_intensity, axis=0)
            * self.mask
        )
        maximum_eigenvalue = max(
            (
                float(item["dominant_parasitic_boundary_eigenvalue"])
                for item in wavelength_stats
            ),
            default=0.0,
        )
        return ASEResult(
            wavelengths_m=self.wavelengths.copy(),
            spectral_intensity_W_m2=spectral_intensity,
            passive_heat_W_m3=passive_heat,
            escaped_power_W=float(escaped_power),
            stats={
                "enabled": True,
                "model": "angular_discrete_ordinates",
                "wavelengths_m": self.wavelengths.tolist(),
                "spectral_weights": self.weights.tolist(),
                "modeled_solid_angle_fraction": solid_angle_fraction,
                "angular_geometry": angular_geometry,
                "direction_count": len(angular_solver.directions),
                "angular_quadrature": "paired_equal_weight_fibonacci_sphere",
                "wavelength_transport": wavelength_stats,
                "radiative_lifetime_source": lifetime_source,
                "radiative_lifetime_config_path": lifetime_config_path,
                "radiative_lifetime_s_min": float(np.min(radiative_lifetime)),
                "radiative_lifetime_s_max": float(np.max(radiative_lifetime)),
                "total_upper_state_lifetime_s_min": float(
                    np.min(total_upper_lifetime)
                ),
                "total_upper_state_lifetime_s_max": float(
                    np.max(total_upper_lifetime)
                ),
                "implied_radiative_branching_fraction_min": float(
                    np.min(radiative_branching_fraction)
                ),
                "implied_radiative_branching_fraction_max": float(
                    np.max(radiative_branching_fraction)
                ),
                "escaped_power_W": float(escaped_power),
                "boundary_iterations": max(
                    (int(item["boundary_iterations"]) for item in wavelength_stats),
                    default=0,
                ),
                "boundary_converged": boundary_converged,
                "intensity_clamp_active": clamp_active,
                "positive_cell_exponent_clipping_active": exponent_clipped,
                "maximum_cell_gain_exponent": max(
                    (
                        float(item["maximum_cell_gain_exponent"])
                        for item in wavelength_stats
                    ),
                    default=0.0,
                ),
                "configured_maximum_intensity_W_m2": float(
                    self.cfg.get("maximum_intensity_W_m2", 1e18)
                ),
                "maximum_total_intensity_W_m2": float(
                    np.max(np.sum(spectral_intensity, axis=0))
                ),
                "dominant_parasitic_boundary_eigenvalue": maximum_eigenvalue,
                "parasitic_threshold_exceeded": threshold_exceeded,
                "parasitic_eigenmode_converged": eigenmode_converged,
                "absolute_boundary_residual_W_m2": max(
                    (
                        float(item.get("absolute_boundary_residual_W_m2", 0.0))
                        for item in wavelength_stats
                    ),
                    default=0.0,
                ),
                "relative_boundary_residual": max(
                    (
                        float(item.get("relative_boundary_residual", 0.0))
                        for item in wavelength_stats
                    ),
                    default=0.0,
                ),
                "parasitic_eigenpair_residual": max(
                    (
                        float(item.get("parasitic_eigenpair_residual", 0.0))
                        for item in wavelength_stats
                    ),
                    default=0.0,
                ),
                "parasitic_eigenvector_change": max(
                    (
                        float(item.get("parasitic_eigenvector_change", 0.0))
                        for item in wavelength_stats
                    ),
                    default=0.0,
                ),
                "parasitic_spectral_gap_estimate": min(
                    (
                        float(item["parasitic_spectral_gap_estimate"])
                        for item in wavelength_stats
                        if item.get("parasitic_spectral_gap_estimate") is not None
                    ),
                    default=None,
                ),
                "reflected_boundary_flux_relative_residual": max(
                    (
                        float(
                            item.get(
                                "reflected_boundary_flux_relative_residual", 0.0
                            )
                        )
                        for item in wavelength_stats
                    ),
                    default=0.0,
                ),
                "embedded_boundary_resolved": embedded_boundary_resolved,
                "boundary_geometry": "Cartesian external faces",
                "scope": (
                    "stationary arbitrary-angle Cartesian discrete ordinates with "
                    "matrix-free reflected-boundary eigenvalue estimation"
                ),
            },
            warnings=list(dict.fromkeys(warnings)),
        )

    def _angular_geometry_diagnostics(
        self,
        modeled_solid_angle_fraction: float,
    ) -> dict[str, Any]:
        """Expose how equal-weight six-ray quadrature samples a long rod.

        This is a diagnostic, not an angular correction.  For a cylindrical
        rod, ``1-cos(theta)`` is the fraction of an isotropic sphere occupied
        by the two opposing cones of half-angle ``theta``.  The far-end and
        centre-to-end values bracket two simple direct-view path lengths.  TIR
        and scattering can make the effective axial population very different,
        so no geometry-derived factor is applied silently.
        """

        length = float(self.grid.z_edges[-1] - self.grid.z_edges[0])
        geometry = self.material.get("geometry", {})
        if str(geometry.get("shape", "cuboid")) != "cylinder":
            return {
                "available": False,
                "reason": "direct axial-cone diagnostic is defined for cylinder geometry",
                "six_ray_axis_pair_source_fraction": float(
                    modeled_solid_angle_fraction / 3.0
                ),
            }
        diameter = geometry.get("diameter_m")
        if diameter is None:
            diameter = min(
                float(self.grid.x_edges[-1] - self.grid.x_edges[0]),
                float(self.grid.y_edges[-1] - self.grid.y_edges[0]),
            )
        diameter = float(diameter)
        radius = 0.5 * diameter
        if length <= 0.0 or radius <= 0.0:
            return {
                "available": False,
                "reason": "cylinder length and diameter must be positive",
                "six_ray_axis_pair_source_fraction": float(
                    modeled_solid_angle_fraction / 3.0
                ),
            }

        def opposing_cones_fraction(path_length: float) -> float:
            half_angle = math.atan(radius / path_length)
            return 1.0 - math.cos(half_angle)

        aspect_ratio = length / diameter
        return {
            "available": True,
            "geometry": "cylinder",
            "length_m": length,
            "diameter_m": diameter,
            "length_to_diameter_ratio": aspect_ratio,
            "high_aspect_ratio_cylinder": aspect_ratio >= 10.0,
            "direct_two_cone_fraction_far_end_path": opposing_cones_fraction(
                length
            ),
            "direct_two_cone_fraction_center_to_end_path": opposing_cones_fraction(
                0.5 * length
            ),
            "six_ray_axis_pair_source_fraction": float(
                modeled_solid_angle_fraction / 3.0
            ),
            "scope": (
                "direct line-of-sight solid-angle diagnostic only; sidewall TIR, "
                "surface scattering, cladding, and oblique transport are excluded"
            ),
        }

    def _axis_widths(self, axis: int) -> np.ndarray:
        return (
            np.asarray(self.grid.z_widths, dtype=float)
            if axis == 0
            else np.asarray(self.grid.y_widths, dtype=float)
            if axis == 1
            else np.asarray(self.grid.x_widths, dtype=float)
        )

    def _face_areas(self, axis: int) -> np.ndarray:
        if axis == 0:
            return self.grid.y_widths[:, None] * self.grid.x_widths[None, :]
        if axis == 1:
            return self.grid.z_widths[:, None] * self.grid.x_widths[None, :]
        return self.grid.z_widths[:, None] * self.grid.y_widths[None, :]

    def _axis_pair(
        self,
        gain: np.ndarray,
        source: np.ndarray,
        axis: int,
        reflect_minus: float,
        reflect_plus: float,
        maximum_intensity: float,
    ) -> tuple[np.ndarray, float, int, bool, bool, bool, float, float]:
        moved_gain = np.moveaxis(gain, axis, 0)
        moved_source = np.moveaxis(source, axis, 0)
        widths = self._axis_widths(axis)
        boundary_shape = moved_gain.shape[1:]
        incoming_plus = np.zeros(boundary_shape, dtype=float)
        incoming_minus = np.zeros(boundary_shape, dtype=float)
        tolerance = float(self.cfg.get("relative_tolerance", 1e-5))
        maximum_iterations = int(self.cfg.get("maximum_boundary_iterations", 40))
        relaxation = float(self.cfg.get("boundary_relaxation", 0.5))
        converged = False
        plus_center = np.zeros_like(moved_gain)
        minus_center = np.zeros_like(moved_gain)
        plus_out = incoming_plus
        minus_out = incoming_minus
        clamped = False
        exponent_clipped = False
        residual = math.inf
        maximum_cell_gain_exponent = float(
            np.max(
                moved_gain
                * widths.reshape((-1,) + (1,) * (moved_gain.ndim - 1))
            )
        )
        for iteration in range(1, maximum_iterations + 1):
            plus_center, plus_out, plus_clamped, plus_exponent_clipped = self._sweep(
                moved_gain, moved_source, widths, incoming_plus, False, maximum_intensity
            )
            minus_center, minus_out, minus_clamped, minus_exponent_clipped = self._sweep(
                moved_gain, moved_source, widths, incoming_minus, True, maximum_intensity
            )
            clamped |= plus_clamped or minus_clamped
            exponent_clipped |= plus_exponent_clipped or minus_exponent_clipped
            target_plus = reflect_minus * minus_out
            target_minus = reflect_plus * plus_out
            scale = max(
                float(np.max(np.abs(target_plus))),
                float(np.max(np.abs(target_minus))),
                float(np.max(np.abs(incoming_plus))),
                float(np.max(np.abs(incoming_minus))),
                np.finfo(float).tiny,
            )
            residual = max(
                float(np.max(np.abs(target_plus - incoming_plus))),
                float(np.max(np.abs(target_minus - incoming_minus))),
            ) / scale
            if residual <= tolerance:
                converged = True
                break
            relaxed_plus = (1.0 - relaxation) * incoming_plus + relaxation * target_plus
            relaxed_minus = (1.0 - relaxation) * incoming_minus + relaxation * target_minus
            clamped |= bool(
                np.any(relaxed_plus > maximum_intensity)
                or np.any(relaxed_minus > maximum_intensity)
            )
            incoming_plus = np.minimum(relaxed_plus, maximum_intensity)
            incoming_minus = np.minimum(relaxed_minus, maximum_intensity)
        escaped = float(
            np.sum((1.0 - reflect_plus) * plus_out * self._face_areas(axis))
            + np.sum((1.0 - reflect_minus) * minus_out * self._face_areas(axis))
        )
        return (
            np.moveaxis(plus_center + minus_center, 0, axis),
            escaped,
            iteration,
            converged,
            clamped,
            exponent_clipped,
            maximum_cell_gain_exponent,
            float(residual),
        )

    @staticmethod
    def _sweep(
        gain: np.ndarray,
        source: np.ndarray,
        widths: np.ndarray,
        incoming_boundary: np.ndarray,
        reverse: bool,
        maximum_intensity: float,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        center = np.zeros_like(gain)
        incoming = np.asarray(incoming_boundary, dtype=float).copy()
        clamped = False
        exponent_clipped = False
        indices = range(len(widths) - 1, -1, -1) if reverse else range(len(widths))
        for index in indices:
            raw_exponent = gain[index] * widths[index]
            # exp/expm1 are intrinsically stable for large negative values.
            # Clipping those values to -50 corrupts the cell-average intensity
            # (and therefore passive heating) in optically thick absorbers.
            # Only the overflowing positive branch needs a numerical guard.
            exponent_clipped |= bool(np.any(raw_exponent > 50.0))
            exponent = np.minimum(raw_exponent, 50.0)
            transmission = np.exp(exponent)
            small = np.abs(exponent) < 1e-5
            phi1 = 1.0 + exponent / 2.0 + exponent * exponent / 6.0
            np.divide(
                np.expm1(exponent),
                exponent,
                out=phi1,
                where=~small,
            )
            source_step = source[index] * widths[index] * phi1
            unbounded_outgoing = np.maximum(
                incoming * transmission + source_step,
                0.0,
            )
            clamped |= bool(np.any(unbounded_outgoing > maximum_intensity))
            outgoing = np.minimum(unbounded_outgoing, maximum_intensity)
            # Population coupling and passive heating require the cell-average
            # intensity, not the trapezoidal average of the two faces. For
            # dI/ds=gI+j, the two exponential phi functions below integrate I
            # analytically over the cell and close the ASE energy ledger even
            # when |g*ds| is not small.
            phi2 = 0.5 + exponent / 6.0 + exponent * exponent / 24.0
            np.divide(
                np.expm1(exponent) - exponent,
                exponent * exponent,
                out=phi2,
                where=~small,
            )
            unbounded_center = np.maximum(
                incoming * phi1
                + source[index] * widths[index] * phi2,
                0.0,
            )
            clamped |= bool(np.any(unbounded_center > maximum_intensity))
            center[index] = np.minimum(unbounded_center, maximum_intensity)
            incoming = outgoing
        return center, incoming, clamped, exponent_clipped
