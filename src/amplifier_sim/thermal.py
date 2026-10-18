"""Three-dimensional steady-state heat conduction with configurable cooling."""

from __future__ import annotations

import inspect
import math
import time
from typing import Any, Callable

import numpy as np

from .grid import Grid3D
from .profiling import profiled_operation


FACE_DIRECTIONS = (
    ("x_min", 2, -1),
    ("x_max", 2, 1),
    ("y_min", 1, -1),
    ("y_max", 1, 1),
    ("z_min", 0, -1),
    ("z_max", 0, 1),
)


def _report_thermal_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    event: dict[str, Any],
) -> None:
    """Keep thermal iteration reporting observational and failure-isolated."""

    if callback is None:
        return
    try:
        callback(dict(event))
    except Exception:
        # Rendering or logging must never alter the thermal solution.
        return


def _canonical_boundary_type(kind: str) -> str:
    aliases = {
        "dirichlet": "fixed_temperature",
        "fixed": "fixed_temperature",
        "fixed_temperature": "fixed_temperature",
        "convective": "water",
        "convection": "water",
        "water": "water",
        "adiabatic": "adiabatic",
    }
    try:
        return aliases[str(kind)]
    except KeyError as exc:
        raise ValueError(f"Unknown thermal boundary type {kind!r}") from exc


def _shift_neighbor(values: np.ndarray, axis: int, offset: int) -> np.ndarray:
    """Shift a cell array so each output cell contains its requested neighbor."""

    result = np.zeros_like(values)
    source = [slice(None)] * values.ndim
    target = [slice(None)] * values.ndim
    if offset < 0:
        source[axis] = slice(None, -1)
        target[axis] = slice(1, None)
    else:
        source[axis] = slice(1, None)
        target[axis] = slice(None, -1)
    result[tuple(target)] = values[tuple(source)]
    return result


class ThermalSolver:
    """Cell-centered SOR solution of ``-div(k grad(T)) = q``.

    ``water`` is a Robin/Newton cooling boundary parameterized by coolant
    temperature and heat-transfer coefficient. It is deliberately a solid-side
    heat-transfer model, not a coolant CFD solver.
    """

    def __init__(
        self,
        grid: Grid3D,
        cfg: dict[str, Any],
        material_mask: np.ndarray | None = None,
        material_geometry: dict[str, Any] | None = None,
    ):
        self.grid = grid
        self.cfg = cfg
        self.material_geometry = dict(material_geometry or {})
        self.crystal_mask = (
            np.ones(grid.shape, dtype=bool)
            if material_mask is None
            else np.asarray(material_mask, dtype=bool)
        )
        if self.crystal_mask.shape != grid.shape:
            raise ValueError("thermal material_mask must match grid.shape")
        if not np.any(self.crystal_mask):
            raise ValueError("thermal material_mask contains no crystal cells")
        self._configure_entities()
        self._boundary_face_scale, self._boundary_area_stats = (
            self._embedded_boundary_area_scales()
        )
        # Geometry, conductivity, contacts, and boundary conditions are fixed
        # for the lifetime of a solver.  Cache their finite-volume operator;
        # coupled thermal iterations then update only the heat-source RHS.
        self._steady_stencil_cache: dict[str, Any] | None = None
        self._sparse_system_cache: dict[str, Any] | None = None
        self._last_temperature: np.ndarray | None = None

    def reset_initial_guess(self) -> None:
        """Discard the warm start while retaining geometry/operator caches."""

        self._last_temperature = None

    def set_initial_guess(self, temperature_K: np.ndarray) -> None:
        """Install a finite positive temperature field as the next solve guess."""

        temperature = np.asarray(temperature_K, dtype=float)
        if temperature.shape != self.grid.shape:
            raise ValueError("thermal initial guess must match grid.shape")
        if not np.all(np.isfinite(temperature)) or float(
            np.min(temperature)
        ) <= 0.0:
            raise ValueError("thermal initial guess must be finite and positive")
        self._last_temperature = temperature.copy()

    @profiled_operation("thermal.solve")
    def solve(
        self,
        heat_W_m3: np.ndarray,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        heat = np.asarray(heat_W_m3, dtype=float)
        if heat.shape != self.grid.shape:
            raise ValueError("heat_W_m3 must match grid.shape")
        if not np.all(np.isfinite(heat)):
            raise ValueError("heat_W_m3 must be finite")
        if np.any((np.abs(heat) > 0.0) & ~self.mask):
            raise ValueError(
                "thermal entity union does not cover every nonzero heat-source cell"
            )
        if str(self.cfg.get("heat_source_model", "")) == "event_energy_ledger":
            heat = np.where(self.mask, heat, 0.0)
        else:
            scale = max(float(np.max(np.abs(heat))), 1.0)
            negative_tolerance = 64.0 * np.finfo(float).eps * scale
            if float(np.min(heat)) < -negative_tolerance:
                raise ValueError(
                    "non-event thermal heat sources must be non-negative"
                )
            heat = np.where(self.mask, np.maximum(heat, 0.0), 0.0)
        boundary = self.cfg.get("boundary", {})
        boundary_type = _canonical_boundary_type(boundary.get("type", "fixed_temperature"))
        if boundary_type == "water":
            reference_temperature = boundary.get(
                "coolant_temperature_K",
                boundary.get("ambient_temperature_K", 293.15),
            )
        elif boundary_type == "fixed_temperature":
            reference_temperature = boundary.get(
                "temperature_K",
                boundary.get("ambient_temperature_K", 293.15),
            )
        else:
            reference_temperature = boundary.get(
                "ambient_temperature_K",
                boundary.get("coolant_temperature_K", 293.15),
            )
        ambient = float(reference_temperature)
        if not self.cfg.get("enabled", False):
            return np.full(self.grid.shape, ambient), {
                "enabled": False,
                "iterations": 0,
                "residual_K": 0.0,
                "active_cell_count": int(np.count_nonzero(self.mask)),
                "heat_input_W": float(self.grid.integrate_volume(heat)),
                "energy_balance_available": False,
            }

        face_cfg = {name: self._face_boundary(name) for name, _, _ in FACE_DIRECTIONS}
        axis_properties = {
            2: (self.conductivity_xyz[0], self.grid.x_widths),
            1: (self.conductivity_xyz[1], self.grid.y_widths),
            0: (self.conductivity_xyz[2], self.grid.z_widths),
        }
        if self.cfg.get("transient", {}).get("enabled", False):
            return self._solve_transient(
                heat,
                ambient,
                face_cfg,
                axis_properties,
                progress_callback=progress_callback,
            )
        if np.any(np.abs(heat) > 0.0) and all(item["type"] == "adiabatic" for item in face_cfg.values()):
            raise ValueError("A nonzero steady heat load cannot use fully adiabatic boundaries")

        linear_solver = str(self.cfg.get("linear_solver", "auto"))
        if linear_solver in {"auto", "cg"}:
            cg_result = self._solve_steady_cg(
                heat,
                ambient,
                face_cfg,
                axis_properties,
                progress_callback=progress_callback,
            )
            if cg_result is not None:
                temperature, stats = cg_result
                if stats["converged"] or linear_solver == "cg":
                    if stats["converged"]:
                        self._last_temperature = temperature.copy()
                    return temperature, stats

        stencil_started = time.perf_counter()
        stencil, stencil_reused = self._steady_stencil(
            face_cfg,
            axis_properties,
            reference_temperature_K=ambient,
        )
        stencil_elapsed = time.perf_counter() - stencil_started
        if (
            self._last_temperature is not None
            and self._last_temperature.shape == self.grid.shape
            and np.all(np.isfinite(self._last_temperature))
        ):
            temperature = self._last_temperature.copy()
            temperature[~self.mask] = ambient
            warm_start_used = True
        else:
            temperature = np.full(self.grid.shape, ambient, dtype=float)
            warm_start_used = False
        relaxation = float(self.cfg.get("relaxation", 1.8))
        if not 0.0 < relaxation < 2.0:
            raise ValueError("thermal.relaxation must be between 0 and 2 for SOR")
        tolerance = float(self.cfg.get("tolerance_K", 1e-4))
        algebraic_tolerance = float(
            self.cfg.get("algebraic_relative_tolerance", 1e-8)
        )
        energy_tolerance = float(
            self.cfg.get("energy_balance_relative_tolerance", 1e-10)
        )
        parity = stencil["parity"]
        update_residual = float("inf")
        equation_stats = {
            "residual_K": float("inf"),
            "algebraic_residual_relative": float("inf"),
        }
        energy_stats: dict[str, Any] = {}
        converged = False
        iterations = 0
        solve_started = time.perf_counter()

        maximum_iterations = int(self.cfg.get("max_iterations", 3000))
        progress_stride = max(maximum_iterations // 100, 1)
        _report_thermal_progress(
            progress_callback,
            {
                "current": 0,
                "total": maximum_iterations,
                "fraction": 0.0,
                "solver": "red_black_sor",
                "phase": "steady",
            },
        )
        for iterations in range(1, maximum_iterations + 1):
            update_residual = 0.0
            for color in (0, 1):
                numerator = heat + stencil["boundary_rhs"]
                for _face_name, axis, offset in FACE_DIRECTIONS:
                    neighbor_coefficient = stencil["neighbor_coefficients"][(axis, offset)]
                    neighbor_mask = stencil["neighbor_masks"][(axis, offset)]
                    neighbor_temperature = _shift_neighbor(temperature, axis, offset)
                    numerator += neighbor_coefficient * neighbor_temperature * neighbor_mask

                selected = (
                    self.mask
                    & (parity == color)
                    & (stencil["diagonal"] > 0.0)
                )
                if not np.any(selected):
                    continue
                candidate = numerator[selected] / stencil["diagonal"][selected]
                old = temperature[selected].copy()
                temperature[selected] = (1.0 - relaxation) * old + relaxation * candidate
                update_residual = max(
                    update_residual,
                    float(np.max(np.abs(temperature[selected] - old))),
                )
            if update_residual < tolerance:
                equation_stats = self._steady_equation_residual_stats(
                    heat, temperature, stencil
                )
                energy_stats = self._energy_balance_stats(
                    heat, temperature, face_cfg, axis_properties
                )
                converged = bool(
                    equation_stats["residual_K"] < tolerance
                    and equation_stats["algebraic_residual_relative"]
                    < algebraic_tolerance
                    and energy_stats["energy_balance_relative_residual"]
                    < energy_tolerance
                )
            if (
                iterations <= 3
                or iterations == maximum_iterations
                or iterations % progress_stride == 0
                or converged
            ):
                _report_thermal_progress(
                    progress_callback,
                    {
                        "current": iterations,
                        "total": maximum_iterations,
                        "fraction": iterations / max(maximum_iterations, 1),
                        "solver": "red_black_sor",
                        "phase": "steady",
                        "update_residual_K": update_residual,
                        "equation_residual_K": equation_stats["residual_K"],
                        "peak_temperature_K": float(np.max(temperature)),
                        "converged": converged,
                    },
                )
            if converged:
                break

        if not converged:
            # Report the residual of the returned iterate.  A non-monotone SOR
            # history can briefly cross the update threshold and then rise
            # again before exhausting its budget; the earlier residual would
            # otherwise misdescribe the final temperature field.
            equation_stats = self._steady_equation_residual_stats(
                heat, temperature, stencil
            )
            energy_stats = self._energy_balance_stats(
                heat, temperature, face_cfg, axis_properties
            )

        temperature[~self.mask] = ambient
        # A bounded non-converged iterate is still a useful initial guess for
        # the next outer fixed-point call.  ``AmplifierSimulator.run`` resets
        # this state, so repeated top-level runs remain deterministic.
        self._last_temperature = temperature.copy()
        stats = {
            "enabled": True,
            "iterations": iterations,
            "residual_K": equation_stats["residual_K"],
            "iteration_update_residual_K": update_residual,
            "algebraic_residual_relative": equation_stats[
                "algebraic_residual_relative"
            ],
            "algebraic_relative_tolerance": algebraic_tolerance,
            "energy_balance_relative_tolerance": energy_tolerance,
            "converged": converged,
            "active_cell_count": int(np.count_nonzero(self.mask)),
            "active_volume_m3": float(self.grid.integrate_volume(self.mask.astype(float))),
            "boundary_types": {name: item["type"] for name, item in face_cfg.items()},
            "cooling_model": "solid_conduction_with_fixed_temperature_or_robin_boundaries",
            "linear_solver": "red_black_sor",
            "operator_assembly_reused": stencil_reused,
            "warm_start_used": warm_start_used,
            "operator_assembly_time_s": stencil_elapsed,
            "linear_solve_time_s": time.perf_counter() - solve_started,
            **self._boundary_area_stats,
            **self._entity_summary(),
        }
        stats.update(energy_stats)
        stats.update(self._contact_diagnostics(temperature))
        return temperature, stats

    def _steady_stencil(
        self,
        face_cfg: dict[str, dict[str, Any]],
        axis_properties: dict[int, tuple[float | np.ndarray, np.ndarray]],
        *,
        reference_temperature_K: float,
    ) -> tuple[dict[str, Any], bool]:
        """Build the heat-independent finite-volume stencil once."""

        if self._steady_stencil_cache is not None:
            return self._steady_stencil_cache, True

        neighbor_masks: dict[tuple[int, int], np.ndarray] = {}
        neighbor_coefficients: dict[tuple[int, int], np.ndarray] = {}
        diagonal = np.zeros(self.grid.shape, dtype=float)
        boundary_rhs = np.zeros(self.grid.shape, dtype=float)
        boundary_offset_rhs = np.zeros(self.grid.shape, dtype=float)

        for face_name, axis, offset in FACE_DIRECTIONS:
            conductivity_axis, widths = axis_properties[axis]
            spacing = self._broadcast_axis(widths, axis)
            neighbor_mask = _shift_neighbor(self.mask, axis, offset)
            missing = self.mask & ~neighbor_mask
            coefficient = np.broadcast_to(
                self._neighbor_coefficient(
                    conductivity_axis, widths, axis, offset
                ),
                self.grid.shape,
            ).copy()
            boundary_coefficient, boundary_temperature = self._boundary_coefficient(
                face_cfg[face_name], conductivity_axis, spacing
            )
            boundary_array = np.broadcast_to(
                boundary_coefficient * self._boundary_face_scale[face_name],
                self.grid.shape,
            ).copy()

            neighbor_masks[(axis, offset)] = neighbor_mask
            neighbor_coefficients[(axis, offset)] = coefficient
            diagonal += coefficient * neighbor_mask
            diagonal += boundary_array * missing
            boundary_rhs += boundary_array * float(boundary_temperature) * missing
            boundary_offset_rhs += (
                boundary_array
                * (float(boundary_temperature) - reference_temperature_K)
                * missing
            )

        self._steady_stencil_cache = {
            "neighbor_masks": neighbor_masks,
            "neighbor_coefficients": neighbor_coefficients,
            "diagonal": diagonal,
            "boundary_rhs": boundary_rhs,
            "boundary_offset_rhs": boundary_offset_rhs,
            "reference_temperature_K": float(reference_temperature_K),
            "parity": np.indices(self.grid.shape).sum(axis=0) % 2,
        }
        return self._steady_stencil_cache, False

    def _steady_equation_residual_stats(
        self,
        heat: np.ndarray,
        temperature: np.ndarray,
        stencil: dict[str, Any],
        *,
        storage: np.ndarray | float = 0.0,
        previous_temperature: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Evaluate ``A T - b`` using the exact cached SOR/CG stencil."""

        storage_array = np.asarray(storage, dtype=float)
        lhs = (stencil["diagonal"] + storage_array) * temperature
        for _face_name, axis, offset in FACE_DIRECTIONS:
            lhs -= (
                stencil["neighbor_coefficients"][(axis, offset)]
                * _shift_neighbor(temperature, axis, offset)
                * stencil["neighbor_masks"][(axis, offset)]
            )
        rhs = heat + stencil["boundary_rhs"]
        if previous_temperature is not None:
            rhs = rhs + storage_array * previous_temperature
        residual = np.where(self.mask, lhs - rhs, 0.0)
        safe_diagonal = np.maximum(
            stencil["diagonal"] + storage_array, np.finfo(float).tiny
        )
        residual_K = float(
            np.max(np.abs(residual[self.mask]) / safe_diagonal[self.mask])
        )
        volumes = self.grid.cell_volumes
        weighted_residual = (residual * volumes)[self.mask]
        weighted_rhs = (rhs * volumes)[self.mask]
        relative = float(
            np.linalg.norm(weighted_residual)
            / max(np.linalg.norm(weighted_rhs), np.finfo(float).tiny)
        )
        return {
            "residual_K": residual_K,
            "algebraic_residual_relative": relative,
        }

    def _solve_steady_cg(
        self,
        heat: np.ndarray,
        ambient: float,
        face_cfg: dict[str, dict[str, Any]],
        axis_properties: dict[int, tuple[float | np.ndarray, np.ndarray]],
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Solve the volume-weighted symmetric finite-volume system with CG."""

        try:
            from scipy.sparse import coo_matrix, diags
            from scipy.sparse.linalg import cg
        except ImportError:
            return None

        assembly_started = time.perf_counter()
        stencil, stencil_reused = self._steady_stencil(
            face_cfg,
            axis_properties,
            reference_temperature_K=ambient,
        )
        sparse_reused = self._sparse_system_cache is not None
        if self._sparse_system_cache is None:
            active_count = int(np.count_nonzero(self.mask))
            active_ids = np.full(self.grid.shape, -1, dtype=np.int64)
            active_ids[self.mask] = np.arange(active_count, dtype=np.int64)
            volumes = self.grid.cell_volumes
            diagonal = (
                stencil["diagonal"] * volumes
            )[self.mask].astype(float, copy=True)
            row_parts: list[np.ndarray] = []
            column_parts: list[np.ndarray] = []
            value_parts: list[np.ndarray] = []

            for _face_name, axis, offset in FACE_DIRECTIONS:
                connected = self.mask & stencil["neighbor_masks"][(axis, offset)]
                if not np.any(connected):
                    continue
                rows = active_ids[connected]
                neighbor_ids = _shift_neighbor(active_ids, axis, offset)[connected]
                weighted_coefficient = (
                    stencil["neighbor_coefficients"][(axis, offset)][connected]
                    * volumes[connected]
                )
                row_parts.append(rows)
                column_parts.append(neighbor_ids)
                value_parts.append(-weighted_coefficient)

            rows = np.arange(active_count, dtype=np.int64)
            row_parts.append(rows)
            column_parts.append(rows)
            value_parts.append(diagonal.copy())
            matrix = coo_matrix(
                (
                    np.concatenate(value_parts),
                    (np.concatenate(row_parts), np.concatenate(column_parts)),
                ),
                shape=(active_count, active_count),
            ).tocsr()
            safe_diagonal = np.where(
                diagonal > np.finfo(float).tiny, diagonal, 1.0
            )
            self._sparse_system_cache = {
                "matrix": matrix,
                "safe_diagonal": safe_diagonal,
                "preconditioner": diags(1.0 / safe_diagonal),
                "active_count": active_count,
            }
        sparse_system = self._sparse_system_cache
        matrix = sparse_system["matrix"]
        safe_diagonal = sparse_system["safe_diagonal"]
        preconditioner = sparse_system["preconditioner"]
        active_count = int(sparse_system["active_count"])
        volumes = self.grid.cell_volumes
        # Solve for the temperature rise above a uniform reference instead of
        # the ~293 K absolute temperature.  This is algebraically identical,
        # including for mixed/nonuniform boundary temperatures, but prevents
        # the physically small heat-load residual from being obtained by
        # subtracting two much larger ambient-temperature terms.  The former
        # formulation could reach a machine-precision CG residual and still
        # miss the heat-flux gate by roundoff, triggering a much slower SOR
        # fallback.
        reference_state = np.full(active_count, ambient, dtype=float)
        incremental_rhs = (
            (heat + stencil["boundary_offset_rhs"]) * volumes
        )[self.mask].astype(float, copy=True)
        assembly_elapsed = time.perf_counter() - assembly_started
        iterations = 0
        maximum_iterations = int(self.cfg.get("max_iterations", 3000))
        progress_stride = max(maximum_iterations // 100, 1)
        _report_thermal_progress(
            progress_callback,
            {
                "current": 0,
                "total": maximum_iterations,
                "fraction": 0.0,
                "solver": "scipy_conjugate_gradient",
                "phase": "steady",
            },
        )

        def count_iteration(iterate: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1
            if (
                iterations <= 3
                or iterations == maximum_iterations
                or iterations % progress_stride == 0
            ):
                _report_thermal_progress(
                    progress_callback,
                    {
                        "current": iterations,
                        "total": maximum_iterations,
                        "fraction": iterations / max(maximum_iterations, 1),
                        "solver": "scipy_conjugate_gradient",
                        "phase": "steady",
                        "peak_temperature_K": float(
                            ambient + np.max(iterate)
                        ),
                    },
                )

        requested_relative_tolerance = float(
            self.cfg.get("cg_relative_tolerance", 1e-8)
        )
        algebraic_tolerance = float(
            self.cfg.get("algebraic_relative_tolerance", 1e-8)
        )
        energy_tolerance = float(
            self.cfg.get("energy_balance_relative_tolerance", 1e-10)
        )
        # A normwise CG residual can be much smaller than the signed global
        # boundary-flux mismatch, especially when hot and cold faces nearly
        # cancel.  Start two decades below the physical energy-balance
        # tolerance, then evaluate the actual equation and boundary-flux
        # residuals.  Only difficult cases continue to the float64-level target.
        # This avoids oversolving ordinary thermal grids without weakening any
        # of the acceptance criteria below.
        precision_floor = 64.0 * np.finfo(float).eps
        final_relative_tolerance = max(
            precision_floor,
            min(
                requested_relative_tolerance,
                algebraic_tolerance,
                energy_tolerance * 1e-4,
            ),
        )
        initial_relative_tolerance = max(
            precision_floor,
            min(
                requested_relative_tolerance,
                algebraic_tolerance,
                energy_tolerance * 1e-2,
            ),
        )
        uses_rtol = "rtol" in inspect.signature(cg).parameters

        def tolerance_keywords(relative_tolerance: float) -> dict[str, float]:
            return (
                {"rtol": relative_tolerance, "atol": 0.0}
                if uses_rtol
                else {"tol": relative_tolerance, "atol": 0.0}
            )
        if (
            self._last_temperature is not None
            and self._last_temperature.shape == self.grid.shape
            and np.all(np.isfinite(self._last_temperature))
        ):
            initial = self._last_temperature[self.mask].copy() - ambient
            warm_start_used = True
        else:
            initial = np.zeros(active_count, dtype=float)
            warm_start_used = False
        solve_started = time.perf_counter()
        relative_tolerance = initial_relative_tolerance
        refinement_stages = 1
        solution, info = cg(
            matrix,
            incremental_rhs,
            x0=initial,
            maxiter=maximum_iterations,
            M=preconditioner,
            callback=count_iteration,
            **tolerance_keywords(relative_tolerance),
        )
        tolerance = float(self.cfg.get("tolerance_K", 1e-4))

        def evaluate_solution(
            candidate: np.ndarray, solver_info: int
        ) -> tuple[np.ndarray, float, float, dict[str, Any], bool]:
            absolute_candidate = candidate + reference_state
            algebraic_residual = matrix @ candidate - incremental_rhs
            residual_K = float(
                np.max(np.abs(algebraic_residual) / safe_diagonal)
            )
            algebraic_relative = float(
                np.linalg.norm(algebraic_residual)
                / max(np.linalg.norm(incremental_rhs), np.finfo(float).tiny)
            )
            temperature = np.full(self.grid.shape, ambient, dtype=float)
            temperature[self.mask] = absolute_candidate
            temperature_rise = np.zeros(self.grid.shape, dtype=float)
            temperature_rise[self.mask] = candidate
            energy_stats = self._energy_balance_stats(
                heat,
                temperature,
                face_cfg,
                axis_properties,
                temperature_offset_K=temperature_rise,
                offset_reference_temperature_K=ambient,
            )
            converged = bool(
                solver_info == 0
                and residual_K
                <= max(tolerance, 10.0 * np.finfo(float).eps * abs(ambient))
                and algebraic_relative < algebraic_tolerance
                and energy_stats["energy_balance_relative_residual"]
                < energy_tolerance
            )
            return (
                temperature,
                residual_K,
                algebraic_relative,
                energy_stats,
                converged,
            )

        (
            temperature,
            residual_K,
            algebraic_relative,
            energy_stats,
            converged,
        ) = evaluate_solution(solution, int(info))
        remaining_iterations = maximum_iterations - iterations
        if (
            not converged
            and info >= 0
            and remaining_iterations > 0
            and initial_relative_tolerance > final_relative_tolerance
        ):
            relative_tolerance = final_relative_tolerance
            refinement_stages = 2
            solution, info = cg(
                matrix,
                incremental_rhs,
                x0=solution,
                maxiter=remaining_iterations,
                M=preconditioner,
                callback=count_iteration,
                **tolerance_keywords(relative_tolerance),
            )
            (
                temperature,
                residual_K,
                algebraic_relative,
                energy_stats,
                converged,
            ) = evaluate_solution(solution, int(info))
        solve_elapsed = time.perf_counter() - solve_started
        _report_thermal_progress(
            progress_callback,
            {
                "current": iterations,
                "total": maximum_iterations,
                "fraction": iterations / max(maximum_iterations, 1),
                "solver": "scipy_conjugate_gradient",
                "phase": "steady",
                "equation_residual_K": residual_K,
                "peak_temperature_K": float(np.max(temperature)),
                "converged": converged,
            },
        )
        stats = {
            "enabled": True,
            "iterations": iterations,
            "residual_K": residual_K,
            "iteration_update_residual_K": None,
            "algebraic_residual_relative": algebraic_relative,
            "algebraic_relative_tolerance": algebraic_tolerance,
            "energy_balance_relative_tolerance": energy_tolerance,
            "converged": converged,
            "linear_solver": "scipy_conjugate_gradient",
            "linear_solver_info": int(info),
            "cg_relative_tolerance_requested": requested_relative_tolerance,
            "cg_initial_relative_tolerance": initial_relative_tolerance,
            "cg_relative_tolerance_effective": relative_tolerance,
            "cg_refinement_stages": refinement_stages,
            "linear_system_unknowns": active_count,
            "active_cell_count": active_count,
            "active_volume_m3": float(self.grid.integrate_volume(self.mask.astype(float))),
            "boundary_types": {name: item["type"] for name, item in face_cfg.items()},
            "cooling_model": "solid_conduction_with_fixed_temperature_or_robin_boundaries",
            "operator_assembly_reused": bool(stencil_reused and sparse_reused),
            "warm_start_used": warm_start_used,
            "operator_assembly_time_s": assembly_elapsed,
            "linear_solve_time_s": solve_elapsed,
            **self._boundary_area_stats,
            **self._entity_summary(),
        }
        stats.update(energy_stats)
        stats.update(self._contact_diagnostics(temperature))
        return temperature, stats

    def _energy_balance_stats(
        self,
        heat: np.ndarray,
        temperature: np.ndarray,
        face_cfg: dict[str, dict[str, Any]],
        axis_properties: dict[int, tuple[float | np.ndarray, np.ndarray]],
        *,
        storage_rate_W: float = 0.0,
        temperature_offset_K: np.ndarray | None = None,
        offset_reference_temperature_K: float = 0.0,
    ) -> dict[str, Any]:
        """Integrate the same finite-volume boundary fluxes used by the solve."""

        volumes = self.grid.cell_volumes
        temperature_offset = (
            None
            if temperature_offset_K is None
            else np.asarray(temperature_offset_K, dtype=float)
        )
        if (
            temperature_offset is not None
            and temperature_offset.shape != self.grid.shape
        ):
            raise ValueError("thermal temperature offset must match the solver grid")
        by_face: dict[str, float] = {}
        boundary_reference_power = 0.0
        for face_name, axis, offset in FACE_DIRECTIONS:
            conductivity_axis, widths = axis_properties[axis]
            spacing = self._broadcast_axis(widths, axis)
            neighbor = _shift_neighbor(self.mask, axis, offset)
            missing = self.mask & ~neighbor
            coefficient, boundary_temperature = self._boundary_coefficient(
                face_cfg[face_name], conductivity_axis, spacing
            )
            coefficient = coefficient * self._boundary_face_scale[face_name]
            coefficient_array = np.broadcast_to(coefficient, self.grid.shape)
            boundary_temperature_difference = (
                temperature - boundary_temperature
                if temperature_offset is None
                else temperature_offset
                - (boundary_temperature - float(offset_reference_temperature_K))
            )
            outward_power = np.where(
                missing,
                coefficient_array
                * boundary_temperature_difference
                * volumes,
                0.0,
            )
            by_face[face_name] = float(np.sum(outward_power))
            boundary_reference_power += float(
                np.sum(
                    np.where(
                        missing,
                        np.abs(
                            coefficient_array
                            * boundary_temperature
                            * volumes
                        ),
                        0.0,
                    )
                )
            )

        heat_input = float(self.grid.integrate_volume(heat))
        boundary_removal = float(sum(by_face.values()))
        boundary_throughput = float(
            sum(abs(value) for value in by_face.values())
        )
        residual = heat_input - boundary_removal - float(storage_rate_W)
        scale = max(
            abs(heat_input),
            abs(boundary_removal),
            boundary_throughput,
            abs(float(storage_rate_W)),
            np.finfo(float).tiny,
        )
        relative_unfloored = abs(residual) / scale
        absolute_source_power = float(
            self.grid.integrate_volume(np.abs(heat))
        )
        roundoff_limit = (
            512.0
            * np.finfo(float).eps
            * max(
                boundary_reference_power,
                absolute_source_power,
                abs(float(storage_rate_W)),
                np.finfo(float).tiny,
            )
        )
        zero_load_roundoff = bool(
            max(
                abs(heat_input),
                abs(boundary_removal),
                boundary_throughput,
                abs(float(storage_rate_W)),
            )
            <= roundoff_limit
            and abs(residual) <= roundoff_limit
        )
        return {
            "energy_balance_available": True,
            "heat_input_W": heat_input,
            "boundary_heat_removal_W": boundary_removal,
            "boundary_absolute_heat_throughput_W": boundary_throughput,
            "boundary_heat_removal_by_face_W": by_face,
            "thermal_storage_rate_W": float(storage_rate_W),
            "energy_balance_residual_W": residual,
            "energy_balance_relative_residual": (
                0.0 if zero_load_roundoff else relative_unfloored
            ),
            "energy_balance_relative_residual_unfloored": relative_unfloored,
            "energy_balance_roundoff_limit_W": roundoff_limit,
            "energy_balance_zero_load_roundoff": zero_load_roundoff,
        }

    def _solve_transient(
        self,
        heat: np.ndarray,
        ambient: float,
        face_cfg: dict[str, dict[str, Any]],
        axis_properties: dict[int, tuple[float | np.ndarray, np.ndarray]],
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Implicit-Euler transient conduction with the same face boundaries."""

        transient = self.cfg.get("transient", {})
        duration = float(transient.get("duration_s", 1.0))
        time_steps = int(transient.get("time_steps", 20))
        dt = duration / time_steps
        volumetric_heat_capacity = self.density_kg_m3 * self.heat_capacity_J_kgK
        storage = volumetric_heat_capacity / dt
        initial = transient.get("initial_temperature_K")
        temperature = np.full(self.grid.shape, ambient if initial is None else float(initial), dtype=float)
        temperature[~self.mask] = ambient
        relaxation = float(transient.get("relaxation", min(float(self.cfg.get("relaxation", 1.8)), 1.5)))
        if not 0.0 < relaxation < 2.0:
            raise ValueError("thermal.transient.relaxation must be between 0 and 2")
        tolerance = float(self.cfg.get("tolerance_K", 1e-4))
        algebraic_tolerance = float(
            self.cfg.get("algebraic_relative_tolerance", 1e-8)
        )
        energy_tolerance = float(
            self.cfg.get("energy_balance_relative_tolerance", 1e-10)
        )
        max_iterations = int(transient.get("max_iterations_per_step", 500))
        assembly_started = time.perf_counter()
        stencil, stencil_reused = self._steady_stencil(
            face_cfg,
            axis_properties,
            reference_temperature_K=ambient,
        )
        assembly_elapsed = time.perf_counter() - assembly_started
        parity = stencil["parity"]
        transient_diagonal = stencil["diagonal"] + storage

        total_iterations = 0
        maximum_update_residual = 0.0
        maximum_equation_residual = 0.0
        maximum_algebraic_residual = 0.0
        maximum_energy_residual = 0.0
        all_converged = True
        last_update_residual = 0.0
        last_equation_stats = {
            "residual_K": 0.0,
            "algebraic_residual_relative": 0.0,
        }
        last_energy_stats: dict[str, Any] = {}
        solve_started = time.perf_counter()
        _report_thermal_progress(
            progress_callback,
            {
                "current": 0,
                "total": time_steps,
                "fraction": 0.0,
                "solver": "red_black_sor",
                "phase": "transient_time_step",
                "thermal_time_s": 0.0,
            },
        )
        for time_step_index in range(1, time_steps + 1):
            previous = temperature.copy()
            step_converged = False
            step_equation_stats: dict[str, float] | None = None
            step_energy_stats: dict[str, Any] | None = None
            for _iteration in range(1, max_iterations + 1):
                total_iterations += 1
                update_residual = 0.0
                for color in (0, 1):
                    numerator = (
                        heat
                        + storage * previous
                        + stencil["boundary_rhs"]
                    )
                    for _face_name, axis, offset in FACE_DIRECTIONS:
                        neighbor_coefficient = stencil["neighbor_coefficients"][
                            (axis, offset)
                        ]
                        neighbor_mask = stencil["neighbor_masks"][(axis, offset)]
                        neighbor_temperature = _shift_neighbor(temperature, axis, offset)
                        numerator += neighbor_coefficient * neighbor_temperature * neighbor_mask
                    selected = (
                        self.mask
                        & (parity == color)
                        & (transient_diagonal > 0.0)
                    )
                    if not np.any(selected):
                        continue
                    candidate = numerator[selected] / transient_diagonal[selected]
                    old = temperature[selected].copy()
                    temperature[selected] = (1.0 - relaxation) * old + relaxation * candidate
                    update_residual = max(
                        update_residual,
                        float(np.max(np.abs(temperature[selected] - old))),
                    )
                last_update_residual = update_residual
                maximum_update_residual = max(
                    maximum_update_residual, update_residual
                )
                if update_residual < tolerance:
                    step_equation_stats = self._steady_equation_residual_stats(
                        heat,
                        temperature,
                        stencil,
                        storage=storage,
                        previous_temperature=previous,
                    )
                    storage_power = float(
                        self.grid.integrate_volume(
                            volumetric_heat_capacity
                            * (temperature - previous)
                            / dt
                        )
                    )
                    step_energy_stats = self._energy_balance_stats(
                        heat,
                        temperature,
                        face_cfg,
                        axis_properties,
                        storage_rate_W=storage_power,
                    )
                    step_converged = bool(
                        step_equation_stats["residual_K"] < tolerance
                        and step_equation_stats[
                            "algebraic_residual_relative"
                        ]
                        < algebraic_tolerance
                        and step_energy_stats[
                            "energy_balance_relative_residual"
                        ]
                        < energy_tolerance
                    )
                    if step_converged:
                        break
            if not step_converged:
                step_equation_stats = self._steady_equation_residual_stats(
                    heat,
                    temperature,
                    stencil,
                    storage=storage,
                    previous_temperature=previous,
                )
                storage_power = float(
                    self.grid.integrate_volume(
                        volumetric_heat_capacity
                        * (temperature - previous)
                        / dt
                    )
                )
                step_energy_stats = self._energy_balance_stats(
                    heat,
                    temperature,
                    face_cfg,
                    axis_properties,
                    storage_rate_W=storage_power,
                )
            last_equation_stats = step_equation_stats
            last_energy_stats = step_energy_stats
            maximum_equation_residual = max(
                maximum_equation_residual,
                float(step_equation_stats["residual_K"]),
            )
            maximum_algebraic_residual = max(
                maximum_algebraic_residual,
                float(step_equation_stats["algebraic_residual_relative"]),
            )
            maximum_energy_residual = max(
                maximum_energy_residual,
                float(step_energy_stats["energy_balance_relative_residual"]),
            )
            all_converged &= step_converged
            temperature[~self.mask] = ambient
            _report_thermal_progress(
                progress_callback,
                {
                    "current": time_step_index,
                    "total": time_steps,
                    "fraction": time_step_index / max(time_steps, 1),
                    "solver": "red_black_sor",
                    "phase": "transient_time_step",
                    "thermal_time_s": time_step_index * dt,
                    "inner_iterations": _iteration,
                    "update_residual_K": last_update_residual,
                    "equation_residual_K": float(
                        step_equation_stats["residual_K"]
                    ),
                    "peak_temperature_K": float(np.max(temperature)),
                    "converged": step_converged,
                },
            )

        stats = {
            "enabled": True,
            "transient": True,
            "duration_s": duration,
            "time_steps": time_steps,
            "time_step_s": dt,
            "iterations": total_iterations,
            "residual_K": last_equation_stats["residual_K"],
            "iteration_update_residual_K": last_update_residual,
            "maximum_iteration_update_residual_K": maximum_update_residual,
            "maximum_equation_residual_K": maximum_equation_residual,
            "algebraic_residual_relative": last_equation_stats[
                "algebraic_residual_relative"
            ],
            "maximum_algebraic_residual_relative": maximum_algebraic_residual,
            "algebraic_relative_tolerance": algebraic_tolerance,
            "energy_balance_relative_tolerance": energy_tolerance,
            "maximum_energy_balance_relative_residual": maximum_energy_residual,
            "converged": all_converged,
            "active_cell_count": int(np.count_nonzero(self.mask)),
            "active_volume_m3": float(self.grid.integrate_volume(self.mask.astype(float))),
            "boundary_types": {name: item["type"] for name, item in face_cfg.items()},
            "cooling_model": "implicit_transient_solid_conduction_with_fixed_temperature_or_robin_boundaries",
            "linear_solver": "red_black_sor",
            "operator_assembly_reused": stencil_reused,
            "warm_start_used": False,
            "operator_assembly_time_s": assembly_elapsed,
            "linear_solve_time_s": time.perf_counter() - solve_started,
            **self._boundary_area_stats,
            **self._entity_summary(),
        }
        stats.update(last_energy_stats)
        stats.update(self._contact_diagnostics(temperature))
        return temperature, stats

    @staticmethod
    def _conductivity_components(
        value: float | list[float] | tuple[float, ...],
        *,
        name: str,
    ) -> tuple[float, float, float]:
        if isinstance(value, (int, float)):
            components = (float(value),) * 3
        else:
            components = tuple(float(item) for item in value)
        if (
            len(components) != 3
            or any(not math.isfinite(item) or item <= 0.0 for item in components)
        ):
            raise ValueError(
                f"{name} must be a positive scalar or positive x/y/z list"
            )
        return components

    def _entity_region_mask(self, region: dict[str, Any]) -> np.ndarray:
        shape = str(region.get("shape", "material_mask"))
        if shape == "material_mask":
            selected = self.crystal_mask.copy()
        else:
            z, y, x = np.meshgrid(
                self.grid.z,
                self.grid.y,
                self.grid.x,
                indexing="ij",
            )
            if shape == "grid":
                selected = np.ones(self.grid.shape, dtype=bool)
            elif shape == "cuboid":
                selected = (
                    (x >= float(region.get("x_min_m", self.grid.x_edges[0])))
                    & (x <= float(region.get("x_max_m", self.grid.x_edges[-1])))
                    & (y >= float(region.get("y_min_m", self.grid.y_edges[0])))
                    & (y <= float(region.get("y_max_m", self.grid.y_edges[-1])))
                    & (z >= float(region.get("z_min_m", self.grid.z_edges[0])))
                    & (z <= float(region.get("z_max_m", self.grid.z_edges[-1])))
                )
            elif shape == "cylinder_z":
                center_x = float(region.get("center_x_m", 0.0))
                center_y = float(region.get("center_y_m", 0.0))
                diameter = float(region["diameter_m"])
                selected = (
                    (x - center_x) ** 2 + (y - center_y) ** 2
                    <= (0.5 * diameter) ** 2
                ) & (
                    z >= float(region.get("z_min_m", self.grid.z_edges[0]))
                ) & (
                    z <= float(region.get("z_max_m", self.grid.z_edges[-1]))
                )
            else:
                raise ValueError(
                    f"Unknown thermal entity region shape {shape!r}"
                )
        if bool(region.get("exclude_material_mask", False)):
            selected &= ~self.crystal_mask
        return selected

    def _configure_entities(self) -> None:
        entities = list(self.cfg.get("entities", []))
        base_conductivity = self._conductivity_components(
            self.cfg["conductivity_W_mK"],
            name="material.thermal.conductivity_W_mK",
        )
        base_density = float(self.cfg["density_kg_m3"])
        base_capacity = float(self.cfg["heat_capacity_J_kgK"])
        if not entities:
            entities = [
                {
                    "name": "crystal",
                    "region": {"shape": "material_mask"},
                    "conductivity_W_mK": list(base_conductivity),
                    "density_kg_m3": base_density,
                    "heat_capacity_J_kgK": base_capacity,
                }
            ]

        entity_id = np.full(self.grid.shape, -1, dtype=np.int32)
        conductivity = np.zeros((3, *self.grid.shape), dtype=float)
        density = np.zeros(self.grid.shape, dtype=float)
        capacity = np.zeros(self.grid.shape, dtype=float)
        names: list[str] = []
        entity_cells: list[int] = []
        for index, entity in enumerate(entities):
            name = str(entity.get("name", "")).strip()
            if not name or name in names:
                raise ValueError(
                    "thermal entity names must be non-empty and unique"
                )
            region = entity.get("region", {"shape": "material_mask"})
            if not isinstance(region, dict):
                raise ValueError(f"thermal entity {name!r} region must be an object")
            selected = self._entity_region_mask(region)
            if not np.any(selected):
                raise ValueError(f"thermal entity {name!r} contains no grid cells")
            if np.any(selected & (entity_id >= 0)):
                raise ValueError(
                    f"thermal entity {name!r} overlaps an earlier entity; "
                    "entity regions must be disjoint"
                )
            components = self._conductivity_components(
                entity.get("conductivity_W_mK", base_conductivity),
                name=f"material.thermal.entities[{index}].conductivity_W_mK",
            )
            local_density = float(entity.get("density_kg_m3", base_density))
            local_capacity = float(
                entity.get("heat_capacity_J_kgK", base_capacity)
            )
            if (
                not math.isfinite(local_density)
                or local_density <= 0.0
                or not math.isfinite(local_capacity)
                or local_capacity <= 0.0
            ):
                raise ValueError(
                    f"thermal entity {name!r} density and heat capacity must be positive"
                )
            entity_id[selected] = index
            for component_index, component in enumerate(components):
                conductivity[component_index, selected] = component
            density[selected] = local_density
            capacity[selected] = local_capacity
            names.append(name)
            entity_cells.append(int(np.count_nonzero(selected)))

        mask = entity_id >= 0
        if not np.all(mask[self.crystal_mask]):
            raise ValueError(
                "thermal entities must cover the complete crystal material mask"
            )
        contacts: dict[tuple[int, int], float] = {}
        name_to_index = {name: index for index, name in enumerate(names)}
        for contact_index, contact in enumerate(self.cfg.get("contacts", [])):
            first_name = str(contact.get("entity_a", ""))
            second_name = str(contact.get("entity_b", ""))
            if (
                first_name not in name_to_index
                or second_name not in name_to_index
                or first_name == second_name
            ):
                raise ValueError(
                    f"thermal contact {contact_index} must name two distinct entities"
                )
            resistance = float(
                contact.get("thermal_contact_resistance_m2K_W", 0.0)
            )
            if not math.isfinite(resistance) or resistance < 0.0:
                raise ValueError(
                    "thermal_contact_resistance_m2K_W must be finite and non-negative"
                )
            pair = tuple(
                sorted(
                    (
                        name_to_index[first_name],
                        name_to_index[second_name],
                    )
                )
            )
            if pair in contacts:
                raise ValueError(
                    f"thermal contact pair {first_name!r}/{second_name!r} is duplicated"
                )
            contacts[pair] = resistance

        for first, second in contacts:
            adjacent = False
            for axis in (0, 1, 2):
                shifted = _shift_neighbor(entity_id, axis, 1)
                connected = mask & _shift_neighbor(mask, axis, 1)
                adjacent |= bool(
                    np.any(
                        (
                            ((entity_id == first) & (shifted == second))
                            | ((entity_id == second) & (shifted == first))
                        )
                        & connected
                    )
                )
            if not adjacent:
                raise ValueError(
                    "configured thermal contact entities "
                    f"{names[first]!r} and {names[second]!r} share no grid face"
                )

        self.mask = mask
        self.entity_id = entity_id
        self.entity_names = names
        self.entity_cell_counts = entity_cells
        self.conductivity_xyz = conductivity
        self.density_kg_m3 = density
        self.heat_capacity_J_kgK = capacity
        self.contact_resistance_by_pair = contacts

    def _entity_summary(self) -> dict[str, Any]:
        contacts = []
        for (first, second), resistance in sorted(
            self.contact_resistance_by_pair.items()
        ):
            contacts.append(
                {
                    "entity_a": self.entity_names[first],
                    "entity_b": self.entity_names[second],
                    "thermal_contact_resistance_m2K_W": resistance,
                }
            )
        return {
            "thermal_entity_count": len(self.entity_names),
            "thermal_entities": [
                {
                    "name": name,
                    "cell_count": self.entity_cell_counts[index],
                }
                for index, name in enumerate(self.entity_names)
            ],
            "thermal_contacts": contacts,
            "multi_entity_conduction": len(self.entity_names) > 1,
        }

    def _contact_resistance_field(
        self,
        axis: int,
        offset: int,
    ) -> np.ndarray:
        neighbor_ids = _shift_neighbor(self.entity_id, axis, offset)
        connected = self.mask & _shift_neighbor(self.mask, axis, offset)
        result = np.zeros(self.grid.shape, dtype=float)
        for (first, second), resistance in self.contact_resistance_by_pair.items():
            interface = (
                ((self.entity_id == first) & (neighbor_ids == second))
                | ((self.entity_id == second) & (neighbor_ids == first))
            ) & connected
            result[interface] = resistance
        return result

    def _contact_diagnostics(self, temperature: np.ndarray) -> dict[str, Any]:
        by_pair: dict[tuple[int, int], dict[str, float]] = {}
        volumes = self.grid.cell_volumes
        # Count every interface once, on its positive-axis orientation.
        for axis in (0, 1, 2):
            neighbor_ids = _shift_neighbor(self.entity_id, axis, 1)
            connected = self.mask & _shift_neighbor(self.mask, axis, 1)
            neighbor_temperature = _shift_neighbor(temperature, axis, 1)
            widths = (self.grid.z_widths, self.grid.y_widths, self.grid.x_widths)[axis]
            local_width = self._broadcast_axis(widths, axis)
            area = volumes / local_width
            for pair, resistance in self.contact_resistance_by_pair.items():
                if resistance <= 0.0:
                    continue
                first, second = pair
                interface = (
                    ((self.entity_id == first) & (neighbor_ids == second))
                    | ((self.entity_id == second) & (neighbor_ids == first))
                ) & connected
                if not np.any(interface):
                    continue
                local_k = self.conductivity_xyz[2 - axis]
                neighbor_k = _shift_neighbor(local_k, axis, 1)
                neighbor_width = _shift_neighbor(
                    np.broadcast_to(local_width, self.grid.shape),
                    axis,
                    1,
                )
                first_half = np.divide(
                    local_width,
                    2.0 * local_k,
                    out=np.zeros(self.grid.shape, dtype=float),
                    where=local_k > 0.0,
                )
                second_half = np.divide(
                    neighbor_width,
                    2.0 * neighbor_k,
                    out=np.zeros(self.grid.shape, dtype=float),
                    where=neighbor_k > 0.0,
                )
                total_resistance = first_half + second_half + resistance
                flux = np.divide(
                    temperature - neighbor_temperature,
                    total_resistance,
                    out=np.zeros_like(temperature),
                    where=interface,
                )
                interface_power = np.abs(flux) * area
                item = by_pair.setdefault(
                    pair,
                    {
                        "interface_area_m2": 0.0,
                        "absolute_heat_transfer_W": 0.0,
                        "area_weighted_contact_jump_K": 0.0,
                    },
                )
                local_area = float(np.sum(area[interface]))
                item["interface_area_m2"] += local_area
                item["absolute_heat_transfer_W"] += float(
                    np.sum(interface_power[interface])
                )
                item["area_weighted_contact_jump_K"] += float(
                    np.sum(np.abs(flux[interface]) * resistance * area[interface])
                )
        rows = []
        for pair, item in sorted(by_pair.items()):
            area = item["interface_area_m2"]
            rows.append(
                {
                    "entity_a": self.entity_names[pair[0]],
                    "entity_b": self.entity_names[pair[1]],
                    "interface_area_m2": area,
                    "absolute_heat_transfer_W": item[
                        "absolute_heat_transfer_W"
                    ],
                    "mean_contact_temperature_jump_K": (
                        item["area_weighted_contact_jump_K"] / area
                        if area > 0.0
                        else 0.0
                    ),
                }
            )
        return {"thermal_contact_diagnostics": rows}

    def _embedded_boundary_area_scales(
        self,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Return conservative area factors for staircase embedded boundaries.

        A cell-centred circular mask has a Manhattan staircase perimeter.  If
        every exposed Cartesian face is assigned its full area, the integrated
        Robin conductance converges to ``8 R L`` rather than the physical
        cylinder area ``2 pi R L``.  The analytic option retains the existing
        stencil but normalises its integrated lateral face area to the exact
        cylinder area.  It is explicit and opt-in; cuboids keep the original
        Cartesian-face model.
        """

        scales = {name: 1.0 for name, _, _ in FACE_DIRECTIONS}
        model = str(self.cfg.get("embedded_boundary_area_model", "cartesian"))
        if model == "cartesian":
            return scales, {
                "embedded_boundary_area_model": model,
                "lateral_boundary_area_scale": 1.0,
            }
        if model != "analytic_cylinder":
            raise ValueError(f"Unknown embedded boundary area model {model!r}")
        if str(self.material_geometry.get("shape", "")) != "cylinder":
            raise ValueError(
                "analytic_cylinder boundary area requires cylinder material geometry"
            )
        diameter = self.material_geometry.get("diameter_m")
        if diameter is None:
            diameter = min(
                float(self.grid.x_edges[-1] - self.grid.x_edges[0]),
                float(self.grid.y_edges[-1] - self.grid.y_edges[0]),
            )
        diameter = float(diameter)
        center_x = float(self.material_geometry.get("center_x_m", 0.0))
        center_y = float(self.material_geometry.get("center_y_m", 0.0))
        radius = 0.5 * diameter
        coordinate_scale = max(
            abs(float(self.grid.x_edges[0])),
            abs(float(self.grid.x_edges[-1])),
            abs(float(self.grid.y_edges[0])),
            abs(float(self.grid.y_edges[-1])),
            abs(center_x),
            abs(center_y),
            diameter,
            np.finfo(float).tiny,
        )
        tolerance = 32.0 * np.finfo(float).eps * coordinate_scale
        if (
            center_x - radius < float(self.grid.x_edges[0]) - tolerance
            or center_x + radius > float(self.grid.x_edges[-1]) + tolerance
            or center_y - radius < float(self.grid.y_edges[0]) - tolerance
            or center_y + radius > float(self.grid.y_edges[-1]) + tolerance
        ):
            raise ValueError(
                "analytic_cylinder boundary area requires the full cylinder to fit inside the thermal grid"
            )

        volumes = self.grid.cell_volumes
        raw_area = 0.0
        for face_name, axis, offset in FACE_DIRECTIONS[:4]:
            neighbor = _shift_neighbor(self.mask, axis, offset)
            missing = self.mask & ~neighbor
            widths = (self.grid.z_widths, self.grid.y_widths, self.grid.x_widths)[axis]
            spacing = self._broadcast_axis(widths, axis)
            raw_area += float(np.sum(np.where(missing, volumes / spacing, 0.0)))
        length = float(self.grid.z_edges[-1] - self.grid.z_edges[0])
        exact_area = math.pi * diameter * length
        if raw_area <= np.finfo(float).tiny:
            raise ValueError("analytic_cylinder found no exposed lateral mask faces")
        scale = exact_area / raw_area
        for face_name, _, _ in FACE_DIRECTIONS[:4]:
            scales[face_name] = scale
        return scales, {
            "embedded_boundary_area_model": model,
            "raw_lateral_boundary_area_m2": raw_area,
            "analytic_lateral_boundary_area_m2": exact_area,
            "lateral_boundary_area_scale": scale,
        }

    def _face_boundary(self, face_name: str) -> dict[str, Any]:
        global_cfg = dict(self.cfg.get("boundary", {}))
        faces = global_cfg.pop("faces", {}) or {}
        override = faces.get(face_name, {})
        if isinstance(override, str):
            override = {"type": override}
        merged = {**global_cfg, **dict(override)}
        merged["type"] = _canonical_boundary_type(merged.get("type", "fixed_temperature"))
        return merged

    @staticmethod
    def _boundary_coefficient(
        boundary: dict[str, Any],
        conductivity: float | np.ndarray,
        spacing: float | np.ndarray,
    ) -> tuple[float | np.ndarray, float]:
        kind = boundary["type"]
        if kind == "adiabatic":
            return 0.0, 0.0
        if kind == "fixed_temperature":
            temperature = float(boundary.get("temperature_K", boundary.get("ambient_temperature_K", 293.15)))
            return 2.0 * conductivity / spacing**2, temperature
        coolant = float(boundary.get("coolant_temperature_K", boundary.get("ambient_temperature_K", 293.15)))
        h = float(boundary.get("heat_transfer_W_m2K", 0.0))
        if h <= 0.0:
            raise ValueError("water cooling requires heat_transfer_W_m2K > 0")
        conduction_resistance = np.divide(
            spacing,
            2.0 * np.asarray(conductivity, dtype=float),
            out=np.full(
                np.broadcast_shapes(
                    np.shape(spacing),
                    np.shape(conductivity),
                ),
                np.inf,
                dtype=float,
            ),
            where=np.asarray(conductivity, dtype=float) > 0.0,
        )
        effective_h = 1.0 / (1.0 / h + conduction_resistance)
        return effective_h / spacing, coolant

    def _broadcast_axis(self, values: np.ndarray, axis: int) -> np.ndarray:
        shape = [1, 1, 1]
        shape[axis] = len(values)
        return np.asarray(values, dtype=float).reshape(shape)

    def _neighbor_coefficient(
        self,
        conductivity: float | np.ndarray,
        widths: np.ndarray,
        axis: int,
        offset: int,
    ) -> np.ndarray:
        local_width = np.broadcast_to(
            self._broadcast_axis(widths, axis),
            self.grid.shape,
        )
        local_k = np.broadcast_to(
            np.asarray(conductivity, dtype=float),
            self.grid.shape,
        )
        neighbor_width = _shift_neighbor(local_width, axis, offset)
        neighbor_k = _shift_neighbor(local_k, axis, offset)
        contact_resistance = self._contact_resistance_field(axis, offset)
        total_resistance = (
            local_width / np.maximum(2.0 * local_k, np.finfo(float).tiny)
            + np.divide(
                neighbor_width,
                np.maximum(2.0 * neighbor_k, np.finfo(float).tiny),
                out=np.zeros_like(local_width),
                where=neighbor_k > 0.0,
            )
            + contact_resistance
        )
        return np.divide(
            1.0,
            local_width * total_resistance,
            out=np.zeros_like(local_width),
            where=total_resistance > np.finfo(float).tiny,
        )
