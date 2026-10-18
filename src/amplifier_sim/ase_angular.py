"""Arbitrary-angle stationary ASE by 3-D discrete ordinates.

The legacy six-ray backend remains the inexpensive axis-aligned screen.  This
module supplies a configurable angular quadrature, first-order positive
step-characteristics transport, specular face reflection, and a matrix-free
power iteration of the reflected boundary operator.  It deliberately reports
its Cartesian-face boundary contract: curved embedded interfaces require an
angular Monte-Carlo or body-fitted extension rather than a silent TIR guess.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .grid import Grid3D


_FACES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


@dataclass
class AngularTransportResult:
    intensity_W_m2: np.ndarray
    escaped_power_W: float
    stats: dict[str, Any]
    warnings: list[str]


def paired_fibonacci_directions(count: int, azimuth_offset_rad: float = 0.0) -> np.ndarray:
    """Return antipodally paired, equal-weight directions on the sphere."""

    direction_count = int(count)
    if direction_count < 8 or direction_count % 2:
        raise ValueError("angular ASE direction_count must be an even integer >= 8")
    half = direction_count // 2
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    positive: list[tuple[float, float, float]] = []
    for index in range(half):
        z = (index + 0.5) / half
        radius = math.sqrt(max(1.0 - z * z, 0.0))
        phi = index * golden_angle + float(azimuth_offset_rad)
        positive.append((radius * math.cos(phi), radius * math.sin(phi), z))
    directions = np.asarray([*positive, *[tuple(-v for v in row) for row in positive]])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return directions


class AngularDiscreteOrdinates:
    """Solve one wavelength at a time on a Cartesian finite-volume grid."""

    def __init__(
        self,
        grid: Grid3D,
        cfg: dict[str, Any],
        face_reflectivity: dict[str, float],
    ):
        self.grid = grid
        angular = cfg.get("angular", {})
        self.directions = paired_fibonacci_directions(
            int(angular.get("direction_count", 24)),
            math.radians(float(angular.get("azimuth_offset_deg", 0.0))),
        )
        self.weights = np.full(
            len(self.directions), 1.0 / len(self.directions), dtype=float
        )
        self.reflectivity = {
            face: float(face_reflectivity.get(face, 0.0)) for face in _FACES
        }
        self.tolerance = float(cfg.get("relative_tolerance", 1e-5))
        self.maximum_iterations = int(cfg.get("maximum_boundary_iterations", 40))
        self.relaxation = float(cfg.get("boundary_relaxation", 0.5))
        self.maximum_intensity = float(cfg.get("maximum_intensity_W_m2", 1e18))
        self.eigen_iterations = int(angular.get("eigenmode_iterations", 30))
        self.eigen_tolerance = float(angular.get("eigenmode_tolerance", 1e-5))
        self.absolute_boundary_tolerance = float(
            angular.get("absolute_boundary_tolerance_W_m2", 1e-12)
        )
        self._reflection_targets = self._build_reflection_targets()

    def solve(self, gain_m_inv: np.ndarray, source_W_m3: np.ndarray) -> AngularTransportResult:
        gain = np.asarray(gain_m_inv, dtype=float)
        source = np.asarray(source_W_m3, dtype=float)
        if gain.shape != self.grid.shape or source.shape != self.grid.shape:
            raise ValueError("angular ASE gain/source must match the physical grid")
        if not np.all(np.isfinite(gain)) or not np.all(np.isfinite(source)):
            raise ValueError("angular ASE gain/source must be finite")
        if np.any(source < 0.0):
            raise ValueError("angular ASE source must be non-negative")

        incoming = self._empty_boundary()
        centers = np.zeros((len(self.directions), *self.grid.shape), dtype=float)
        outgoing: list[dict[str, np.ndarray]] = []
        converged = False
        clamped = False
        exponent_clipped = False
        maximum_exponent = -np.inf
        residual = math.inf
        absolute_residual = math.inf
        maximum_reflection_flux_residual = 0.0
        for iteration in range(1, self.maximum_iterations + 1):
            outgoing = []
            for direction_index, direction in enumerate(self.directions):
                center, boundary, local_clamped, local_clipped, local_maximum = (
                    self._sweep(
                        gain,
                        source,
                        direction,
                        incoming[direction_index],
                        self.maximum_intensity,
                    )
                )
                centers[direction_index] = center
                outgoing.append(boundary)
                clamped |= local_clamped
                exponent_clipped |= local_clipped
                maximum_exponent = max(maximum_exponent, local_maximum)
            target = self._reflect(outgoing)
            maximum_reflection_flux_residual = max(
                maximum_reflection_flux_residual,
                self._reflection_flux_residual(outgoing, target),
            )
            absolute_residual = self._absolute_boundary_difference(target, incoming)
            residual = self._relative_boundary_difference(target, incoming)
            if (
                absolute_residual <= self.absolute_boundary_tolerance
                or residual <= self.tolerance
            ):
                converged = True
                break
            incoming = self._blend_and_clamp(incoming, target)

        escaped = self._escaped_power(outgoing)
        (
            eigenvalue,
            eigen_converged,
            eigen_steps,
            eigenpair_residual,
            eigenvector_change,
            spectral_gap_estimate,
        ) = self._dominant_boundary_eigenvalue(gain)
        warnings: list[str] = []
        if not converged:
            warnings.append(
                "Angular ASE reflected-boundary iteration did not converge; the stationary field is invalid."
            )
        if clamped:
            warnings.append(
                "Angular ASE reached maximum_intensity_W_m2; the stationary solution is clamped."
            )
        if exponent_clipped:
            warnings.append(
                "Angular ASE clipped a positive effective-cell gain exponent above 50."
            )
        if eigenvalue >= 1.0:
            warnings.append(
                "The arbitrary-angle reflected transport operator has dominant eigenvalue >= 1; "
                "a parasitic mode is above the stationary threshold."
            )
        if not eigen_converged and any(value > 0.0 for value in self.reflectivity.values()):
            warnings.append(
                "Angular ASE parasitic boundary-eigenvalue power iteration did not converge."
            )
        return AngularTransportResult(
            intensity_W_m2=np.tensordot(self.weights, centers, axes=(0, 0)),
            escaped_power_W=escaped,
            stats={
                "direction_count": len(self.directions),
                "angular_quadrature": "paired_equal_weight_fibonacci_sphere",
                "angular_weight_sum": float(np.sum(self.weights)),
                "directions_xyz": self.directions.tolist(),
                "boundary_iterations": iteration,
                "absolute_boundary_residual_W_m2": float(absolute_residual),
                "relative_boundary_residual": float(residual),
                "boundary_residual_relative": float(residual),
                "absolute_boundary_tolerance_W_m2": self.absolute_boundary_tolerance,
                "boundary_converged": converged,
                "reflected_boundary_flux_relative_residual": float(
                    maximum_reflection_flux_residual
                ),
                "intensity_clamp_active": clamped,
                "positive_cell_exponent_clipping_active": exponent_clipped,
                "maximum_cell_gain_exponent": float(maximum_exponent),
                "dominant_parasitic_boundary_eigenvalue": float(eigenvalue),
                "parasitic_eigenmode_iterations": eigen_steps,
                "parasitic_eigenmode_converged": eigen_converged,
                "parasitic_eigenpair_residual": float(eigenpair_residual),
                "parasitic_eigenvector_change": float(eigenvector_change),
                "parasitic_spectral_gap_estimate": spectral_gap_estimate,
                "parasitic_threshold_exceeded": bool(eigenvalue >= 1.0),
                "boundary_geometry": "Cartesian external faces",
            },
            warnings=warnings,
        )

    def _face_shape(self, face: str) -> tuple[int, int]:
        nz, ny, nx = self.grid.shape
        if face.startswith("x_"):
            return nz, ny
        if face.startswith("y_"):
            return nz, nx
        return ny, nx

    def _face_area(self, face: str) -> np.ndarray:
        if face.startswith("x_"):
            return self.grid.z_widths[:, None] * self.grid.y_widths[None, :]
        if face.startswith("y_"):
            return self.grid.z_widths[:, None] * self.grid.x_widths[None, :]
        return self.grid.y_widths[:, None] * self.grid.x_widths[None, :]

    def _empty_boundary(self) -> list[dict[str, np.ndarray]]:
        return [
            {
                face: np.zeros(self._face_shape(face), dtype=float)
                for face in _FACES
            }
            for _ in self.directions
        ]

    def _build_reflection_targets(
        self,
    ) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
        """Return a local conservative redistribution for each mirror ray.

        Fibonacci ordinates are not closed under Cartesian reflection.  A
        reflected direction is therefore shared by the three closest incoming
        ordinates.  The normal-flux Jacobian is applied later in `_reflect`.
        """
        normals = {
            "x_min": np.asarray([-1.0, 0.0, 0.0]),
            "x_max": np.asarray([1.0, 0.0, 0.0]),
            "y_min": np.asarray([0.0, -1.0, 0.0]),
            "y_max": np.asarray([0.0, 1.0, 0.0]),
            "z_min": np.asarray([0.0, 0.0, -1.0]),
            "z_max": np.asarray([0.0, 0.0, 1.0]),
        }
        targets: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
        for index, direction in enumerate(self.directions):
            for face, normal in normals.items():
                reflected = direction - 2.0 * float(np.dot(direction, normal)) * normal
                incoming = np.flatnonzero(self.directions @ normal < -1e-14)
                similarity = self.directions[incoming] @ reflected
                nearest_count = min(3, len(incoming))
                selected_local = np.argpartition(
                    similarity, -nearest_count
                )[-nearest_count:]
                selected = incoming[selected_local]
                distance = np.maximum(1.0 - similarity[selected_local], 1e-12)
                fractions = 1.0 / distance
                fractions /= float(np.sum(fractions))
                targets[(index, face)] = (
                    selected.astype(np.int64), fractions.astype(float)
                )
        return targets

    @staticmethod
    def _phi_functions(exponent: float) -> tuple[float, float, float]:
        clipped = min(float(exponent), 50.0)
        transmission = math.exp(clipped)
        if abs(clipped) < 1e-5:
            phi1 = 1.0 + clipped / 2.0 + clipped * clipped / 6.0
            phi2 = 0.5 + clipped / 6.0 + clipped * clipped / 24.0
        else:
            phi1 = math.expm1(clipped) / clipped
            phi2 = (math.expm1(clipped) - clipped) / (clipped * clipped)
        return transmission, phi1, phi2

    def _sweep(
        self,
        gain: np.ndarray,
        source: np.ndarray,
        direction: np.ndarray,
        incoming_boundary: dict[str, np.ndarray],
        maximum_intensity: float,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], bool, bool, float]:
        dx, dy, dz = map(float, direction)
        nz, ny, nx = self.grid.shape
        z_indices = range(nz) if dz >= 0.0 else range(nz - 1, -1, -1)
        y_indices = range(ny) if dy >= 0.0 else range(ny - 1, -1, -1)
        x_indices = range(nx) if dx >= 0.0 else range(nx - 1, -1, -1)
        outgoing_volume = np.zeros(self.grid.shape, dtype=float)
        center = np.zeros(self.grid.shape, dtype=float)
        clamped = False
        clipped = False
        maximum_exponent = -np.inf
        for iz in z_indices:
            cz = abs(dz) / float(self.grid.z_widths[iz])
            for iy in y_indices:
                cy = abs(dy) / float(self.grid.y_widths[iy])
                for ix in x_indices:
                    cx = abs(dx) / float(self.grid.x_widths[ix])
                    coefficient = cx + cy + cz
                    if coefficient <= np.finfo(float).tiny:
                        continue
                    incoming_sum = 0.0
                    if cx > 0.0:
                        if dx >= 0.0:
                            value = (
                                outgoing_volume[iz, iy, ix - 1]
                                if ix > 0
                                else incoming_boundary["x_min"][iz, iy]
                            )
                        else:
                            value = (
                                outgoing_volume[iz, iy, ix + 1]
                                if ix + 1 < nx
                                else incoming_boundary["x_max"][iz, iy]
                            )
                        incoming_sum += cx * float(value)
                    if cy > 0.0:
                        if dy >= 0.0:
                            value = (
                                outgoing_volume[iz, iy - 1, ix]
                                if iy > 0
                                else incoming_boundary["y_min"][iz, ix]
                            )
                        else:
                            value = (
                                outgoing_volume[iz, iy + 1, ix]
                                if iy + 1 < ny
                                else incoming_boundary["y_max"][iz, ix]
                            )
                        incoming_sum += cy * float(value)
                    if cz > 0.0:
                        if dz >= 0.0:
                            value = (
                                outgoing_volume[iz - 1, iy, ix]
                                if iz > 0
                                else incoming_boundary["z_min"][iy, ix]
                            )
                        else:
                            value = (
                                outgoing_volume[iz + 1, iy, ix]
                                if iz + 1 < nz
                                else incoming_boundary["z_max"][iy, ix]
                            )
                        incoming_sum += cz * float(value)
                    incoming = incoming_sum / coefficient
                    path = 1.0 / coefficient
                    exponent = float(gain[iz, iy, ix]) * path
                    maximum_exponent = max(maximum_exponent, exponent)
                    clipped |= exponent > 50.0
                    transmission, phi1, phi2 = self._phi_functions(exponent)
                    local_source = float(source[iz, iy, ix])
                    unbounded_out = max(
                        incoming * transmission + local_source * path * phi1,
                        0.0,
                    )
                    unbounded_center = max(
                        incoming * phi1 + local_source * path * phi2,
                        0.0,
                    )
                    clamped |= (
                        unbounded_out > maximum_intensity
                        or unbounded_center > maximum_intensity
                    )
                    outgoing_volume[iz, iy, ix] = min(
                        unbounded_out, maximum_intensity
                    )
                    center[iz, iy, ix] = min(
                        unbounded_center, maximum_intensity
                    )
        boundary: dict[str, np.ndarray] = {}
        if dx >= 0.0:
            boundary["x_max"] = outgoing_volume[:, :, -1].copy()
        else:
            boundary["x_min"] = outgoing_volume[:, :, 0].copy()
        if dy >= 0.0:
            boundary["y_max"] = outgoing_volume[:, -1, :].copy()
        else:
            boundary["y_min"] = outgoing_volume[:, 0, :].copy()
        if dz >= 0.0:
            boundary["z_max"] = outgoing_volume[-1, :, :].copy()
        else:
            boundary["z_min"] = outgoing_volume[0, :, :].copy()
        return center, boundary, clamped, clipped, maximum_exponent

    def _reflect(
        self, outgoing: list[dict[str, np.ndarray]]
    ) -> list[dict[str, np.ndarray]]:
        target = self._empty_boundary()
        component = {
            "x_min": 0,
            "x_max": 0,
            "y_min": 1,
            "y_max": 1,
            "z_min": 2,
            "z_max": 2,
        }
        for direction_index, faces in enumerate(outgoing):
            for face, values in faces.items():
                reflectivity = self.reflectivity[face]
                if reflectivity <= 0.0:
                    continue
                reflected_indices, fractions = self._reflection_targets[
                    (direction_index, face)
                ]
                axis = component[face]
                source_flux_factor = (
                    self.weights[direction_index]
                    * abs(float(self.directions[direction_index, axis]))
                )
                for reflected_index, fraction in zip(
                    reflected_indices, fractions, strict=True
                ):
                    reflected_index = int(reflected_index)
                    target_flux_factor = (
                        self.weights[reflected_index]
                        * abs(float(self.directions[reflected_index, axis]))
                    )
                    target[reflected_index][face] += (
                        reflectivity
                        * float(fraction)
                        * source_flux_factor
                        / max(target_flux_factor, np.finfo(float).tiny)
                        * values
                    )
        return target

    def _reflection_flux_residual(
        self,
        outgoing: list[dict[str, np.ndarray]],
        incoming: list[dict[str, np.ndarray]],
    ) -> float:
        component = {
            "x_min": 0,
            "x_max": 0,
            "y_min": 1,
            "y_max": 1,
            "z_min": 2,
            "z_max": 2,
        }
        expected = 0.0
        actual = 0.0
        for index, faces in enumerate(outgoing):
            for face, values in faces.items():
                expected += (
                    self.reflectivity[face]
                    * self.weights[index]
                    * abs(float(self.directions[index, component[face]]))
                    * float(np.sum(values * self._face_area(face)))
                )
        for index, faces in enumerate(incoming):
            for face, values in faces.items():
                actual += (
                    self.weights[index]
                    * abs(float(self.directions[index, component[face]]))
                    * float(np.sum(values * self._face_area(face)))
                )
        return abs(actual - expected) / max(abs(expected), abs(actual), 1e-300)

    @staticmethod
    def _boundary_max(boundary: list[dict[str, np.ndarray]]) -> float:
        return max(
            (float(np.max(values)) for item in boundary for values in item.values()),
            default=0.0,
        )

    def _relative_boundary_difference(
        self,
        first: list[dict[str, np.ndarray]],
        second: list[dict[str, np.ndarray]],
    ) -> float:
        difference = self._absolute_boundary_difference(first, second)
        return difference / max(
            self._boundary_max(first),
            self._boundary_max(second),
            np.finfo(float).tiny,
        )

    @staticmethod
    def _absolute_boundary_difference(
        first: list[dict[str, np.ndarray]],
        second: list[dict[str, np.ndarray]],
    ) -> float:
        return max(
            (
                float(np.max(np.abs(first[index][face] - second[index][face])))
                for index in range(len(first))
                for face in _FACES
            ),
            default=0.0,
        )

    def _blend_and_clamp(
        self,
        old: list[dict[str, np.ndarray]],
        target: list[dict[str, np.ndarray]],
    ) -> list[dict[str, np.ndarray]]:
        return [
            {
                face: np.minimum(
                    (1.0 - self.relaxation) * old[index][face]
                    + self.relaxation * target[index][face],
                    self.maximum_intensity,
                )
                for face in _FACES
            }
            for index in range(len(old))
        ]

    def _escaped_power(self, outgoing: list[dict[str, np.ndarray]]) -> float:
        component = {
            "x_min": 0,
            "x_max": 0,
            "y_min": 1,
            "y_max": 1,
            "z_min": 2,
            "z_max": 2,
        }
        total = 0.0
        for index, faces in enumerate(outgoing):
            direction = self.directions[index]
            for face, values in faces.items():
                total += (
                    (1.0 - self.reflectivity[face])
                    * self.weights[index]
                    * abs(float(direction[component[face]]))
                    * float(np.sum(values * self._face_area(face)))
                )
        return total

    def _dominant_boundary_eigenvalue(
        self, gain: np.ndarray
    ) -> tuple[float, bool, int, float, float, float | None]:
        if not any(value > 0.0 for value in self.reflectivity.values()):
            return 0.0, True, 0, 0.0, 0.0, None
        state = self._empty_boundary()
        for index, direction in enumerate(self.directions):
            dx, dy, dz = direction
            for face, active in (
                ("x_min", dx > 0.0),
                ("x_max", dx < 0.0),
                ("y_min", dy > 0.0),
                ("y_max", dy < 0.0),
                ("z_min", dz > 0.0),
                ("z_max", dz < 0.0),
            ):
                if active:
                    state[index][face].fill(1.0)
        initial_norm = self._boundary_flux_norm(state)
        if initial_norm > np.finfo(float).tiny:
            for item in state:
                for face in _FACES:
                    item[face] /= initial_norm
        previous = 0.0
        eigenvalue = 0.0
        converged = False
        eigenpair_residual = math.inf
        eigenvector_change = math.inf
        previous_vector_change: float | None = None
        spectral_gap_estimate: float | None = None
        zero_source = np.zeros(self.grid.shape, dtype=float)
        for iteration in range(1, self.eigen_iterations + 1):
            outgoing: list[dict[str, np.ndarray]] = []
            for index, direction in enumerate(self.directions):
                _, boundary, _, _, _ = self._sweep(
                    gain,
                    zero_source,
                    direction,
                    state[index],
                    np.finfo(float).max,
                )
                outgoing.append(boundary)
            mapped = self._reflect(outgoing)
            eigenvalue = self._boundary_flux_norm(mapped)
            if eigenvalue <= np.finfo(float).tiny:
                return 0.0, True, iteration, 0.0, 0.0, None
            normalized = [
                {face: values / eigenvalue for face, values in item.items()}
                for item in mapped
            ]
            residual_boundary = [
                {
                    face: mapped[index][face] - eigenvalue * state[index][face]
                    for face in _FACES
                }
                for index in range(len(state))
            ]
            eigenpair_residual = self._boundary_flux_l1(residual_boundary) / max(
                self._boundary_flux_l1(mapped), np.finfo(float).tiny
            )
            change_boundary = [
                {
                    face: normalized[index][face] - state[index][face]
                    for face in _FACES
                }
                for index in range(len(state))
            ]
            eigenvector_change = self._boundary_flux_l1(change_boundary) / max(
                self._boundary_flux_l1(normalized),
                self._boundary_flux_l1(state),
                np.finfo(float).tiny,
            )
            if previous_vector_change is not None and previous_vector_change > 0.0:
                ratio = eigenvector_change / previous_vector_change
                if math.isfinite(ratio):
                    spectral_gap_estimate = float(np.clip(1.0 - ratio, 0.0, 1.0))
            eigenvalue_change = abs(eigenvalue - previous) / max(
                abs(eigenvalue), abs(previous), np.finfo(float).tiny
            )
            if (
                eigenpair_residual <= self.eigen_tolerance
                and eigenvector_change <= self.eigen_tolerance
                and eigenvalue_change <= self.eigen_tolerance
            ):
                converged = True
                state = normalized
                break
            previous_vector_change = eigenvector_change
            previous = eigenvalue
            state = normalized
        return (
            float(eigenvalue),
            converged,
            iteration,
            float(eigenpair_residual),
            float(eigenvector_change),
            spectral_gap_estimate,
        )

    def _boundary_flux_l1(
        self, boundary: list[dict[str, np.ndarray]]
    ) -> float:
        absolute = [
            {face: np.abs(values) for face, values in item.items()}
            for item in boundary
        ]
        return self._boundary_flux_norm(absolute)

    def _boundary_flux_norm(
        self, boundary: list[dict[str, np.ndarray]]
    ) -> float:
        component = {
            "x_min": 0,
            "x_max": 0,
            "y_min": 1,
            "y_max": 1,
            "z_min": 2,
            "z_max": 2,
        }
        return float(
            sum(
                self.weights[index]
                * abs(float(self.directions[index, component[face]]))
                * float(np.sum(values * self._face_area(face)))
                for index, item in enumerate(boundary)
                for face, values in item.items()
            )
        )
