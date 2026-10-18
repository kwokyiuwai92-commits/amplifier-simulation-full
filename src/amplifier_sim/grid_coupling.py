"""Coupling between a full crystal grid and a local optical grid.

Two different numerical contracts are required here:

* optical/physical ledgers are finite-volume quantities and use conservative
  cell-overlap remapping;
* temperature, refractive index and level populations are intensive fields
  sampled by a coherent wave solver and require a continuous reconstruction.

Using the finite-volume, piecewise-constant reconstruction for the second
group preserves integrals but inserts phase and gain steps at every physical
cell boundary.  Those artificial high spatial frequencies strongly bias M2.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .spectral_volume import SpectralIntensityVolume

from .grid import Grid3D
from .optics import transverse_volume_boundary_diagnostics
from .pump import PumpPropagationResult
from .signal import SignalPropagationResult


def _shared_levels(propagator: Any) -> Any | None:
    levels = getattr(propagator, "levels", None)
    if levels is not None:
        return levels
    for name in ("primary", "fallback", "inner"):
        candidate = getattr(propagator, name, None)
        levels = getattr(candidate, "levels", None)
        if levels is not None:
            return levels
    return None


def _map_level_environment_to_optical(levels: Any, coupler: "GridCoupler") -> tuple[Any, Any] | None:
    if levels is None or not getattr(levels.response, "enabled", False):
        return None
    temperature, density = levels.environment_copy()

    def mapped(value: Any) -> Any:
        array = np.asarray(value)
        return (
            coupler.intensive_to_optical(array)
            if array.shape == coupler.physical.shape
            else value
        )

    levels.set_environment(mapped(temperature), mapped(density))
    return temperature, density


def _restore_level_environment(levels: Any, snapshot: tuple[Any, Any] | None) -> None:
    if levels is not None and snapshot is not None:
        levels.set_environment(*snapshot)


def _signal_optical_boundary_diagnostics(
    result: SignalPropagationResult,
    grid: Grid3D,
) -> dict[str, Any] | None:
    """Capture native optical-grid occupancy before conservative remapping."""

    per_pass = getattr(result, "fluence_per_pass_J_m2", None)
    candidates: list[tuple[int, np.ndarray]] = []
    if per_pass is not None:
        stacked = np.asarray(per_pass, dtype=float)
        if stacked.ndim == 4 and stacked.shape[1:] == grid.shape:
            candidates.extend(
                (index + 1, stacked[index]) for index in range(len(stacked))
            )
    if not candidates:
        last_pass = getattr(result, "fluence_last_pass_J_m2", None)
        values = (
            np.asarray(last_pass, dtype=float)
            if last_pass is not None
            else np.asarray(result.fluence_J_m2, dtype=float)
        )
        if values.shape != grid.shape:
            return None
        pass_count = max(len(getattr(result, "passes", []) or []), 1)
        candidates.append((pass_count, values))

    summaries: list[dict[str, Any]] = []
    for pass_index, values in candidates:
        summary = transverse_volume_boundary_diagnostics(values, grid)
        summary.update(
            {
                "pass_index": int(pass_index),
                "sample_location": "native_optical_z_bin_exit_after_split_step",
                "diagnostic_grid": "native_optical",
            }
        )
        summaries.append(summary)
    return max(
        summaries,
        key=lambda item: float(item["maximum_boundary_energy_fraction"]),
    )


def _cell_average_matrix(target_edges: np.ndarray, source_edges: np.ndarray) -> np.ndarray:
    """Return target-cell averages of piecewise-constant source cell data."""

    target_left = target_edges[:-1, None]
    target_right = target_edges[1:, None]
    source_left = source_edges[None, :-1]
    source_right = source_edges[None, 1:]
    overlap = np.maximum(
        np.minimum(target_right, source_right) - np.maximum(target_left, source_left),
        0.0,
    )
    return overlap / np.diff(target_edges)[:, None]


def _cell_average_operator(target_edges: np.ndarray, source_edges: np.ndarray) -> Any:
    matrix = _cell_average_matrix(target_edges, source_edges)
    try:
        from scipy.sparse import csr_matrix
    except ImportError:
        return matrix
    return csr_matrix(matrix)


def _linear_center_matrix(
    target_centers: np.ndarray,
    source_centers: np.ndarray,
) -> np.ndarray:
    """Return bounded linear interpolation weights between cell centres.

    Target centres outside the outer source centres use constant extension.
    The optical window is validated to remain inside the physical window, so
    this affects only the half-cell strip between a boundary and its first
    physical centre.  All weights are non-negative and rows sum exactly to
    one, preventing overshoot of temperature, index or population values.
    """

    target = np.asarray(target_centers, dtype=float)
    source = np.asarray(source_centers, dtype=float)
    if target.ndim != 1 or source.ndim != 1 or len(source) == 0:
        raise ValueError("grid interpolation coordinates must be non-empty one-dimensional arrays")
    if len(source) == 1:
        return np.ones((len(target), 1), dtype=float)
    if np.any(np.diff(source) <= 0.0):
        raise ValueError("source grid centres must be strictly increasing")

    matrix = np.zeros((len(target), len(source)), dtype=float)
    lower = np.searchsorted(source, target, side="right") - 1
    lower = np.clip(lower, 0, len(source) - 2)
    upper = lower + 1
    span = source[upper] - source[lower]
    upper_weight = np.clip((target - source[lower]) / span, 0.0, 1.0)
    rows = np.arange(len(target))
    matrix[rows, lower] = 1.0 - upper_weight
    matrix[rows, upper] += upper_weight
    return matrix


def _linear_center_operator(
    target_centers: np.ndarray,
    source_centers: np.ndarray,
) -> Any:
    matrix = _linear_center_matrix(target_centers, source_centers)
    try:
        from scipy.sparse import csr_matrix
    except ImportError:
        return matrix
    return csr_matrix(matrix)


@dataclass(frozen=True)
class GridCoupler:
    """Finite-volume ledger transfer plus continuous intensive-field sampling."""

    physical: Grid3D
    optical: Grid3D
    physical_to_optical_y: Any
    physical_to_optical_x: Any
    optical_to_physical_y: Any
    optical_to_physical_x: Any
    physical_to_optical_intensive_y: Any
    physical_to_optical_intensive_x: Any
    identity_mapping: bool
    remap_backend: str

    @classmethod
    def build(cls, physical: Grid3D, optical: Grid3D) -> "GridCoupler":
        if len(physical.z) != len(optical.z) or not np.allclose(
            physical.z_edges, optical.z_edges, rtol=1e-12, atol=1e-15
        ):
            raise ValueError("dual-grid coupling currently requires identical physical and optical z cells")
        if not optical.is_uniform_xy:
            raise ValueError("the optical grid must be uniform in x/y")
        physical_to_optical_y = _cell_average_operator(optical.y_edges, physical.y_edges)
        physical_to_optical_x = _cell_average_operator(optical.x_edges, physical.x_edges)
        optical_to_physical_y = _cell_average_operator(physical.y_edges, optical.y_edges)
        optical_to_physical_x = _cell_average_operator(physical.x_edges, optical.x_edges)
        physical_to_optical_intensive_y = _linear_center_operator(
            optical.y, physical.y
        )
        physical_to_optical_intensive_x = _linear_center_operator(
            optical.x, physical.x
        )
        return cls(
            physical=physical,
            optical=optical,
            physical_to_optical_y=physical_to_optical_y,
            physical_to_optical_x=physical_to_optical_x,
            optical_to_physical_y=optical_to_physical_y,
            optical_to_physical_x=optical_to_physical_x,
            physical_to_optical_intensive_y=physical_to_optical_intensive_y,
            physical_to_optical_intensive_x=physical_to_optical_intensive_x,
            identity_mapping=bool(
                np.array_equal(physical.x_edges, optical.x_edges)
                and np.array_equal(physical.y_edges, optical.y_edges)
            ),
            remap_backend="scipy_csr" if hasattr(physical_to_optical_x, "tocsr") else "numpy_dense",
        )

    @staticmethod
    def _map(
        values: np.ndarray,
        matrix_y: Any,
        matrix_x: Any,
    ) -> np.ndarray:
        array = np.asarray(values)
        if array.shape[-2:] != (matrix_y.shape[1], matrix_x.shape[1]):
            raise ValueError("coupled field transverse shape does not match its source grid")
        if hasattr(matrix_x, "tocsr"):
            leading = array.shape[:-2]
            source_y, source_x = array.shape[-2:]
            target_y, target_x = matrix_y.shape[0], matrix_x.shape[0]
            mapped_x = matrix_x.dot(array.reshape(-1, source_x).T).T.reshape(
                *leading, source_y, target_x
            )
            y_last = np.moveaxis(mapped_x, -2, -1)
            mapped_y_last = matrix_y.dot(y_last.reshape(-1, source_y).T).T.reshape(
                *leading, target_x, target_y
            )
            return np.moveaxis(mapped_y_last, -1, -2)
        # Some BLAS builds raise spurious overflow/invalid warnings for the
        # very large but finite population densities used here, even though
        # both interpolation matrices are bounded stochastic weights.  A
        # power-of-two normalization is exact in binary arithmetic and keeps
        # the dense fallback in a numerically ordinary magnitude range.
        if array.dtype.kind in {"f", "c"} and array.size:
            maximum = float(np.max(np.abs(array)))
            if math.isfinite(maximum) and maximum > 2.0**32:
                exponent = math.frexp(maximum)[1]
                scale = math.ldexp(1.0, min(exponent - 1, 1023))
                normalized = array / scale
                mapped_y = np.einsum(
                    "yi,...ij->...yj",
                    matrix_y,
                    normalized,
                    optimize=False,
                )
                return (
                    np.einsum(
                        "...yj,xj->...yx",
                        mapped_y,
                        matrix_x,
                        optimize=False,
                    )
                    * scale
                )
        # Keep the two contractions separate.  Besides avoiding the affected
        # BLAS path, this changes the work from one four-index contraction to
        # the same two-stage complexity as the sparse/matmul implementation.
        mapped_y = np.einsum(
            "yi,...ij->...yj",
            matrix_y,
            array,
            optimize=False,
        )
        return np.einsum(
            "...yj,xj->...yx",
            mapped_y,
            matrix_x,
            optimize=False,
        )

    def to_optical(self, values: np.ndarray) -> np.ndarray:
        """Finite-volume reconstruction for conservative ledger quantities."""

        if self.identity_mapping:
            return np.asarray(values).copy()
        return self._map(values, self.physical_to_optical_y, self.physical_to_optical_x)

    def intensive_to_optical(self, values: np.ndarray) -> np.ndarray:
        """Continuously sample a physical intensive field on the optical grid."""

        if self.identity_mapping:
            return np.asarray(values).copy()
        return self._map(
            values,
            self.physical_to_optical_intensive_y,
            self.physical_to_optical_intensive_x,
        )

    def to_physical(self, values: np.ndarray) -> np.ndarray:
        if self.identity_mapping:
            return np.asarray(values).copy()
        return self._map(values, self.optical_to_physical_y, self.optical_to_physical_x)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "method": "finite_volume_cell_overlap",
            "ledger_remap_method": "finite_volume_cell_overlap",
            "physical_to_optical_intensive_method": (
                "bounded_bilinear_cell_center_interpolation"
            ),
            "remap_backend": self.remap_backend,
            "physical_shape_nz_ny_nx": list(self.physical.shape),
            "optical_shape_nz_ny_nx": list(self.optical.shape),
            "optical_window_x_m": float(self.optical.x_edges[-1] - self.optical.x_edges[0]),
            "optical_window_y_m": float(self.optical.y_edges[-1] - self.optical.y_edges[0]),
            "optical_dx_m": float(self.optical.dx),
            "optical_dy_m": float(self.optical.dy),
        }


class CoupledPumpPropagator:
    """Run Pump optics on the optical grid and return physical-grid source terms."""

    def __init__(self, inner: Any, coupler: GridCoupler):
        self.inner = inner
        self.coupler = coupler
        self.physics = inner.physics
        self.spectrum = inner.spectrum

    def set_endface_surface_sag(
        self,
        front_surface_sag_m: np.ndarray | None,
        back_surface_sag_m: np.ndarray | None,
    ) -> None:
        setter = getattr(self.inner, "set_endface_surface_sag", None)
        if setter is None:
            return
        setter(
            None
            if front_surface_sag_m is None
            else self.coupler.intensive_to_optical(front_surface_sag_m),
            None
            if back_surface_sag_m is None
            else self.coupler.intensive_to_optical(back_surface_sag_m),
        )

    def propagate(
        self,
        population: np.ndarray,
        record_planes: bool = False,
        power_scale: float = 1.0,
        delta_n: np.ndarray | None = None,
    ) -> PumpPropagationResult:
        optical_population = np.maximum(
            self.coupler.intensive_to_optical(population), 0.0
        )
        optical_delta_n = (
            None
            if delta_n is None
            else self.coupler.intensive_to_optical(delta_n)
        )
        levels = _shared_levels(self.inner)
        environment = _map_level_environment_to_optical(levels, self.coupler)
        try:
            result = self.inner.propagate(
                optical_population,
                record_planes,
                power_scale=power_scale,
                delta_n=optical_delta_n,
            )
        finally:
            _restore_level_environment(levels, environment)
        optical_boundary = transverse_volume_boundary_diagnostics(
            result.intensity_W_m2,
            self.coupler.optical,
        )
        optical_boundary.update(
            {
                "pass_index": 1,
                "sample_location": "native_optical_z_bin_cell_average",
                "diagnostic_grid": "native_optical",
            }
        )
        optical_absorbed = float(self.coupler.optical.integrate_volume(result.absorbed_W_m3))
        result.intensity_W_m2 = self.coupler.to_physical(result.intensity_W_m2)
        result.absorbed_W_m3 = self.coupler.to_physical(result.absorbed_W_m3)
        if result.active_absorbed_W_m3 is not None:
            result.active_absorbed_W_m3 = self.coupler.to_physical(
                result.active_absorbed_W_m3
            )
        if result.passive_absorbed_W_m3 is not None:
            result.passive_absorbed_W_m3 = self.coupler.to_physical(
                result.passive_absorbed_W_m3
            )
        for name in (
            "spectral_intensity_W_m2",
            "spectral_active_absorbed_W_m3",
            "spectral_passive_absorbed_W_m3",
        ):
            values = getattr(result, name, None)
            if values is not None:
                setattr(result, name, self.coupler.to_physical(values))
        physical_absorbed = float(self.coupler.physical.integrate_volume(result.absorbed_W_m3))
        relative_error = abs(physical_absorbed - optical_absorbed) / max(
            abs(optical_absorbed), np.finfo(float).tiny
        )
        result.stats.update(
            {
                "absorbed_power_W": physical_absorbed,
                "optical_grid_absorbed_power_W": optical_absorbed,
                "grid_coupling_relative_power_error": relative_error,
                "optical_grid_boundary_diagnostics": optical_boundary,
            }
        )
        if result.active_absorbed_W_m3 is not None:
            result.stats["active_absorbed_power_W"] = float(
                self.coupler.physical.integrate_volume(result.active_absorbed_W_m3)
            )
        if result.passive_absorbed_W_m3 is not None:
            result.stats["passive_absorbed_power_W"] = float(
                self.coupler.physical.integrate_volume(result.passive_absorbed_W_m3)
            )
        if relative_error > 1e-10:
            result.warnings.append(
                f"Pump optical-to-physical remap power error is {relative_error:.3e}."
            )
        return result


class CoupledSignalPropagator:
    """Run Seed optics on the optical grid and commit its population change physically."""

    def __init__(self, inner: Any, coupler: GridCoupler, physical_mask: np.ndarray):
        self.inner = inner
        self.coupler = coupler
        fallback = getattr(inner, "fallback", None)
        self.physics = getattr(inner, "physics", getattr(fallback, "physics", None))
        if self.physics is None:
            raise ValueError("coupled signal propagator requires a shared physics configuration")
        self.physical_mask = np.asarray(physical_mask, dtype=bool)
        self._initial_free_carrier_density_m3: np.ndarray | None = None
        self._initial_free_carrier_density_optical_m3: np.ndarray | None = None
        self._initial_free_carrier_remap_correction_relative = 0.0

    def _optical_carrier_support(self) -> np.ndarray:
        """Return the active optical-cell support used by the inner backend."""

        pending = [self.inner]
        visited: set[int] = set()
        while pending:
            candidate = pending.pop(0)
            if candidate is None or id(candidate) in visited:
                continue
            visited.add(id(candidate))
            mask = getattr(candidate, "crystal_mask", None)
            if mask is not None:
                array = np.asarray(mask, dtype=float)
                if array.shape == self.coupler.optical.shape[-2:]:
                    return np.broadcast_to(
                        array[None, ...], self.coupler.optical.shape
                    ).copy()
            pending.extend(
                getattr(candidate, name, None)
                for name in ("primary", "fallback", "inner")
            )
        # Lightweight/custom backends need not expose a crystal mask.  The
        # conservative overlap of the physical mask is then the best available
        # cell-volume support and is exact for an all-active rectangular grid.
        return np.clip(
            self.coupler.to_optical(self.physical_mask.astype(float)),
            0.0,
            1.0,
        )

    def _carrier_state_to_optical(
        self, density_m3: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Sample a representable carrier state and conserve every z-slice count.

        A smaller optical window cannot carry an independently populated part
        of the physical crystal.  Concentrating such out-of-window carriers
        into the optical window would preserve particle count while creating
        fictitious local FCA/refraction.  States produced by this wrapper are
        in the conservative backprojection support and satisfy this contract.
        """

        density = np.asarray(density_m3, dtype=float)
        scale = max(float(np.max(np.abs(density))), 1.0)
        if float(np.max(np.abs(density[~self.physical_mask]), initial=0.0)) > (
            64.0 * np.finfo(float).eps * scale
        ):
            raise ValueError(
                "initial free-carrier state must be zero outside the physical crystal"
            )
        physical = np.where(self.physical_mask, density, 0.0)
        optical_support = self._optical_carrier_support()
        physical_reachable = np.clip(
            self.coupler.to_physical(optical_support), 0.0, 1.0
        )
        unreachable = self.physical_mask & (
            physical_reachable <= 64.0 * np.finfo(float).eps
        )
        if float(np.max(np.abs(physical[unreachable]), initial=0.0)) > (
            64.0 * np.finfo(float).eps * scale
        ):
            raise ValueError(
                "initial free-carrier state extends outside the optical "
                "carrier-state support; enlarge the optical grid"
            )
        optical = self.coupler.intensive_to_optical(physical)
        optical *= optical_support
        physical_count = np.asarray(
            self.coupler.physical.integrate_transverse(physical), dtype=float
        )
        optical_count = np.asarray(
            self.coupler.optical.integrate_transverse(optical), dtype=float
        )
        positive = physical_count > np.finfo(float).tiny
        if np.any(positive & (optical_count <= np.finfo(float).tiny)):
            raise ValueError(
                "the optical grid does not sample a carrier-bearing physical slice"
            )
        factors = np.ones_like(physical_count)
        np.divide(
            physical_count,
            optical_count,
            out=factors,
            where=positive,
        )
        factors[~positive] = 0.0
        optical *= factors[:, None, None]
        correction = (
            float(np.max(np.abs(factors[positive] - 1.0)))
            if np.any(positive)
            else 0.0
        )
        return optical, correction

    def resolved_intrapulse_population_time_s(self) -> float:
        """Preserve the inner solver's population-clock contract."""

        getter = getattr(
            self.inner,
            "resolved_intrapulse_population_time_s",
            None,
        )
        return 0.0 if getter is None else float(getter())

    def resolved_intrapulse_free_carrier_time_s(self) -> float:
        """Preserve the inner solver's free-carrier clock contract."""

        getter = getattr(
            self.inner,
            "resolved_intrapulse_free_carrier_time_s",
            None,
        )
        return 0.0 if getter is None else float(getter())

    def periodic_free_carrier_state_enabled(self) -> bool:
        """Preserve the inner solver's exact carrier-state capability."""

        nested = getattr(
            self.inner,
            "periodic_free_carrier_state_enabled",
            None,
        )
        if nested is not None:
            return bool(nested())
        return bool(
            getattr(
                self.inner,
                "resolved_intrapulse_free_carrier_time_s",
                None,
            )
            is not None
            and getattr(self.inner, "set_initial_free_carrier_density", None)
            is not None
        )

    def set_photoelastic_tensor(
        self, fields: dict[str, np.ndarray] | None
    ) -> None:
        setter = getattr(self.inner, "set_photoelastic_tensor", None)
        if setter is None:
            return
        if fields is None:
            setter(None)
            return
        setter(
            {
                key: self.coupler.intensive_to_optical(np.asarray(values, dtype=float))
                for key, values in fields.items()
            }
        )

    def set_endface_surface_sag(
        self,
        front_surface_sag_m: np.ndarray | None,
        back_surface_sag_m: np.ndarray | None,
    ) -> None:
        setter = getattr(self.inner, "set_endface_surface_sag", None)
        if setter is None:
            return
        setter(
            None
            if front_surface_sag_m is None
            else self.coupler.intensive_to_optical(front_surface_sag_m),
            None
            if back_surface_sag_m is None
            else self.coupler.intensive_to_optical(back_surface_sag_m),
        )

    def set_initial_free_carrier_density(
        self, density_m3: np.ndarray | None
    ) -> None:
        setter = getattr(self.inner, "set_initial_free_carrier_density", None)
        if setter is None:
            return
        if density_m3 is None:
            self._initial_free_carrier_density_m3 = None
            self._initial_free_carrier_density_optical_m3 = None
            self._initial_free_carrier_remap_correction_relative = 0.0
            setter(None)
            return
        density = np.asarray(density_m3, dtype=float)
        if density.shape != self.coupler.physical.shape:
            raise ValueError(
                "initial free-carrier state must match the physical grid"
            )
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError(
                "initial free-carrier state must be finite and non-negative"
            )
        optical_density, correction = self._carrier_state_to_optical(density)
        self._initial_free_carrier_density_m3 = np.where(
            self.physical_mask, density, 0.0
        )
        self._initial_free_carrier_density_optical_m3 = optical_density.copy()
        self._initial_free_carrier_remap_correction_relative = correction
        setter(optical_density)

    def propagate(self, population: np.ndarray, delta_n: np.ndarray) -> SignalPropagationResult:
        optical_initial = np.maximum(
            self.coupler.intensive_to_optical(population), 0.0
        )
        physical_carrier_initial = (
            None
            if self._initial_free_carrier_density_m3 is None
            else self._initial_free_carrier_density_m3.copy()
        )
        optical_carrier_initial = (
            None
            if self._initial_free_carrier_density_optical_m3 is None
            else self._initial_free_carrier_density_optical_m3.copy()
        )
        initial_carrier_remap_correction = (
            self._initial_free_carrier_remap_correction_relative
        )
        levels = _shared_levels(self.inner)
        environment = _map_level_environment_to_optical(levels, self.coupler)
        try:
            result = self.inner.propagate(
                optical_initial.copy(), self.coupler.intensive_to_optical(delta_n)
            )
        finally:
            _restore_level_environment(levels, environment)
            # The coherent backend treats the entrance carrier field as a
            # one-shot boundary state.  Mirror that ownership here so a later
            # top-level run cannot inherit a stale physical-grid snapshot.
            self._initial_free_carrier_density_m3 = None
            self._initial_free_carrier_density_optical_m3 = None
            self._initial_free_carrier_remap_correction_relative = 0.0
        optical_boundary = _signal_optical_boundary_diagnostics(
            result,
            self.coupler.optical,
        )
        physical_change = self.coupler.to_physical(result.population_m3 - optical_initial)
        mapped_population = np.asarray(population, dtype=float) + physical_change
        totals = np.sum(population, axis=0)
        before_projection = mapped_population.copy()
        mapped_population = np.maximum(mapped_population, 0.0)
        if levels is not None and hasattr(
            levels, "project_component_inventories"
        ):
            mapped_population = levels.project_component_inventories(
                mapped_population, totals
            )
        else:
            sums = np.sum(mapped_population, axis=0)
            erased_active = (totals > np.finfo(float).tiny) & (
                sums <= np.finfo(float).tiny
            )
            if np.any(erased_active):
                raise ValueError(
                    "Dual-grid population projection erased every level in an active "
                    "physical cell; ion conservation cannot be restored"
                )
            scale = np.divide(
                totals,
                sums,
                out=np.zeros_like(totals),
                where=sums > 0.0,
            )
            mapped_population *= scale[None, ...]
        mapped_population *= self.physical_mask[None, ...]
        absolute_projection_correction = np.max(
            np.abs(mapped_population - before_projection), axis=0
        )
        local_projection_correction = np.divide(
            absolute_projection_correction,
            totals,
            out=np.zeros_like(totals),
            where=totals > np.finfo(float).tiny,
        )
        projection_correction = float(np.max(local_projection_correction))
        global_density_scale = max(float(np.max(totals)), np.finfo(float).tiny)
        global_projection_correction = float(
            np.max(absolute_projection_correction) / global_density_scale
        )
        result.population_m3 = mapped_population
        result.fluence_J_m2 = self.coupler.to_physical(result.fluence_J_m2)
        for name in ("fluence_last_pass_J_m2", "fluence_cumulative_J_m2"):
            values = getattr(result, name)
            if values is not None:
                setattr(result, name, self.coupler.to_physical(values))
        if result.fluence_per_pass_J_m2 is not None:
            result.fluence_per_pass_J_m2 = self.coupler.to_physical(
                result.fluence_per_pass_J_m2
            )
        if result.extracted_energy_J_m3 is not None:
            optical_extracted = float(
                self.coupler.optical.integrate_volume(result.extracted_energy_J_m3)
            )
            optical_extracted_norm = float(
                self.coupler.optical.integrate_volume(np.abs(result.extracted_energy_J_m3))
            )
            result.extracted_energy_J_m3 = self.coupler.to_physical(result.extracted_energy_J_m3)
            physical_extracted = float(
                self.coupler.physical.integrate_volume(result.extracted_energy_J_m3)
            )
            relative_error = abs(physical_extracted - optical_extracted) / max(
                optical_extracted_norm, np.finfo(float).tiny
            )
        else:
            optical_extracted = 0.0
            optical_extracted_norm = 0.0
            physical_extracted = 0.0
            relative_error = 0.0
        if result.nonlinear_loss_energy_J_m3 is not None:
            result.nonlinear_loss_energy_J_m3 = self.coupler.to_physical(
                result.nonlinear_loss_energy_J_m3
            )
        if result.passive_loss_energy_J_m3 is not None:
            result.passive_loss_energy_J_m3 = self.coupler.to_physical(
                result.passive_loss_energy_J_m3
            )
        carrier_projection_correction_m3 = 0.0
        carrier_change_relative_error = 0.0
        if result.free_carrier_density_m3 is not None:
            optical_carrier_final = np.asarray(
                result.free_carrier_density_m3, dtype=float
            )
            if optical_carrier_initial is None:
                optical_carrier_initial = np.zeros_like(
                    optical_carrier_final
                )
                physical_carrier_initial = np.zeros(
                    self.coupler.physical.shape, dtype=float
                )
            optical_carrier_change = (
                optical_carrier_final - optical_carrier_initial
            )
            physical_carrier_change = self.coupler.to_physical(
                optical_carrier_change
            )
            unprojected_carrier = (
                physical_carrier_initial + physical_carrier_change
            )
            mapped_carrier = np.where(
                self.physical_mask,
                np.maximum(unprojected_carrier, 0.0),
                0.0,
            )
            # A conservative delta can contain local undershoots after the
            # smooth forward interpolation and piecewise-constant reverse
            # projection.  Clipping those undershoots alone creates carriers.
            # Restore the exact optical net count independently in every
            # physical z slice while retaining non-negativity.
            initial_count_by_z = np.asarray(
                self.coupler.physical.integrate_transverse(
                    physical_carrier_initial
                ),
                dtype=float,
            )
            optical_change_count_by_z = np.asarray(
                self.coupler.optical.integrate_transverse(
                    optical_carrier_change
                ),
                dtype=float,
            )
            target_count_by_z = (
                initial_count_by_z + optical_change_count_by_z
            )
            count_scale = np.maximum(
                np.maximum(
                    np.abs(initial_count_by_z),
                    np.abs(optical_change_count_by_z),
                ),
                1.0,
            )
            if np.any(
                target_count_by_z
                < -64.0 * np.finfo(float).eps * count_scale
            ):
                raise RuntimeError(
                    "dual-grid carrier update produced a negative slice inventory"
                )
            target_count_by_z = np.maximum(target_count_by_z, 0.0)
            projected_count_by_z = np.asarray(
                self.coupler.physical.integrate_transverse(mapped_carrier),
                dtype=float,
            )
            for z_index, target_count in enumerate(target_count_by_z):
                if target_count <= np.finfo(float).tiny:
                    mapped_carrier[z_index].fill(0.0)
                    continue
                projected_count = float(projected_count_by_z[z_index])
                if projected_count <= np.finfo(float).tiny:
                    raise RuntimeError(
                        "dual-grid carrier projection lost a positive slice inventory"
                    )
                mapped_carrier[z_index] *= target_count / projected_count
            carrier_projection_correction_m3 = float(
                np.max(np.abs(mapped_carrier - unprojected_carrier))
            )
            optical_change_count = float(
                self.coupler.optical.integrate_volume(
                    optical_carrier_change
                )
            )
            physical_change_count = float(
                self.coupler.physical.integrate_volume(
                    mapped_carrier - physical_carrier_initial
                )
            )
            carrier_change_relative_error = abs(
                physical_change_count - optical_change_count
            ) / max(
                float(
                    self.coupler.optical.integrate_volume(
                        np.abs(optical_carrier_change)
                    )
                ),
                np.finfo(float).tiny,
            )
            result.free_carrier_density_m3 = mapped_carrier
        local_population_error = np.abs(np.sum(mapped_population, axis=0) - totals)
        active_density = totals > 0.0
        population_total_error = (
            float(np.max(local_population_error[active_density] / totals[active_density]))
            if np.any(active_density)
            else 0.0
        )
        result.grid_coupling = {
            **self.coupler.summary(),
            "population_projection_correction_relative": projection_correction,
            "population_projection_correction_relative_global_density_scale": (
                global_projection_correction
            ),
            "population_projection_correction_normalization": (
                "local_dopant_density"
            ),
            "population_conservation_max_relative": population_total_error,
            "ensemble_inventory": (
                levels.ensemble_inventory_diagnostics(
                    mapped_population, totals
                )
                if levels is not None
                and hasattr(levels, "ensemble_inventory_diagnostics")
                else {"enabled": False, "applicable": False}
            ),
            "optical_extracted_energy_J": optical_extracted,
            "physical_extracted_energy_J": physical_extracted,
            "optical_absolute_extracted_energy_J": optical_extracted_norm,
            "extraction_energy_relative_error": relative_error,
            "free_carrier_state_remap": (
                "conservative_delta_backprojection"
                if result.free_carrier_density_m3 is not None
                else "not_applicable"
            ),
            "free_carrier_projection_correction_m3": (
                carrier_projection_correction_m3
            ),
            "free_carrier_initial_remap_correction_relative": (
                initial_carrier_remap_correction
            ),
            "free_carrier_change_relative_error": (
                carrier_change_relative_error
            ),
        }
        if optical_boundary is not None:
            result.grid_coupling[
                "optical_grid_boundary_diagnostics"
            ] = optical_boundary
        if projection_correction > 1e-8:
            result.warnings.append(
                "Dual-grid population projection applied a relative correction of "
                f"{projection_correction:.3e}; refine the physical grid or enlarge the optical window."
            )
        if relative_error > 1e-10:
            result.warnings.append(
                f"Seed optical-to-physical remap extraction-energy error is {relative_error:.3e}."
            )
        return result


class CoupledAveragePowerSignalPropagator:
    """Map the cycle-averaged optical flux probe onto the physical grid."""

    def __init__(self, inner: Any, coupler: GridCoupler, physical_mask: np.ndarray):
        self.inner = inner
        self.coupler = coupler
        self.physics = inner.physics
        self.physical_mask = np.asarray(physical_mask, dtype=bool)

    def set_photoelastic_tensor(
        self, fields: dict[str, np.ndarray] | None
    ) -> None:
        setter = getattr(self.inner, "set_photoelastic_tensor", None)
        if setter is None:
            return
        if fields is None:
            setter(None)
            return
        setter(
            {
                key: self.coupler.intensive_to_optical(np.asarray(values, dtype=float))
                for key, values in fields.items()
            }
        )

    def set_endface_surface_sag(
        self,
        front_surface_sag_m: np.ndarray | None,
        back_surface_sag_m: np.ndarray | None,
    ) -> None:
        setter = getattr(self.inner, "set_endface_surface_sag", None)
        if setter is None:
            return
        setter(
            None
            if front_surface_sag_m is None
            else self.coupler.intensive_to_optical(front_surface_sag_m),
            None
            if back_surface_sag_m is None
            else self.coupler.intensive_to_optical(back_surface_sag_m),
        )

    def propagate_average(
        self,
        population: np.ndarray,
        delta_n: np.ndarray,
        repetition_rate_Hz: float,
        record_planes: bool = False,
    ) -> SignalPropagationResult:
        optical_population = np.maximum(
            self.coupler.intensive_to_optical(population), 0.0
        )
        levels = _shared_levels(self.inner)
        environment = _map_level_environment_to_optical(levels, self.coupler)
        try:
            result = self.inner.propagate_average(
                optical_population,
                self.coupler.intensive_to_optical(delta_n),
                repetition_rate_Hz,
                record_planes,
            )
        finally:
            _restore_level_environment(levels, environment)
        optical_boundary = _signal_optical_boundary_diagnostics(
            result,
            self.coupler.optical,
        )
        result.population_m3 = np.asarray(population, dtype=float).copy()
        result.fluence_J_m2 = self.coupler.to_physical(result.fluence_J_m2)
        for name in ("fluence_last_pass_J_m2", "fluence_cumulative_J_m2"):
            values = getattr(result, name)
            if values is not None:
                setattr(result, name, self.coupler.to_physical(values))
        if result.fluence_per_pass_J_m2 is not None:
            result.fluence_per_pass_J_m2 = self.coupler.to_physical(
                result.fluence_per_pass_J_m2
            )
        if result.average_spectral_intensity_W_m2 is not None:
            intensity = result.average_spectral_intensity_W_m2
            result.average_spectral_intensity_W_m2 = (
                intensity.mapped(self.coupler.to_physical, self.coupler.physical.shape)
                if isinstance(intensity, SpectralIntensityVolume)
                else self.coupler.to_physical(intensity)
            )
        if result.extracted_energy_J_m3 is not None:
            optical_extracted = float(
                self.coupler.optical.integrate_volume(result.extracted_energy_J_m3)
            )
            result.extracted_energy_J_m3 = self.coupler.to_physical(
                result.extracted_energy_J_m3
            )
            physical_extracted = float(
                self.coupler.physical.integrate_volume(result.extracted_energy_J_m3)
            )
            normalization = max(abs(optical_extracted), np.finfo(float).tiny)
            relative_error = abs(physical_extracted - optical_extracted) / normalization
        else:
            optical_extracted = 0.0
            physical_extracted = 0.0
            relative_error = 0.0
        if result.nonlinear_loss_energy_J_m3 is not None:
            result.nonlinear_loss_energy_J_m3 = self.coupler.to_physical(
                result.nonlinear_loss_energy_J_m3
            )
        if result.passive_loss_energy_J_m3 is not None:
            result.passive_loss_energy_J_m3 = self.coupler.to_physical(
                result.passive_loss_energy_J_m3
            )
        result.grid_coupling = {
            **self.coupler.summary(),
            "optical_extracted_energy_J": optical_extracted,
            "physical_extracted_energy_J": physical_extracted,
            "extraction_energy_relative_error": relative_error,
            "population_committed": False,
        }
        if optical_boundary is not None:
            result.grid_coupling[
                "optical_grid_boundary_diagnostics"
            ] = optical_boundary
        if relative_error > 1e-10:
            result.warnings.append(
                "Cycle-averaged Seed optical-to-physical extraction remap error is "
                f"{relative_error:.3e}."
            )
        return result
