"""Replaceable thermo-mechanical solvers."""

from __future__ import annotations

from collections.abc import Callable
import inspect
import math
import time
from typing import Any

import numpy as np

from .profiling import profiled_operation


def _engineering_strain_to_tensor(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    return np.asarray(
        [
            [values[0], 0.5 * values[3], 0.5 * values[5]],
            [0.5 * values[3], values[1], 0.5 * values[4]],
            [0.5 * values[5], 0.5 * values[4], values[2]],
        ]
    )


def _tensor_to_engineering_strain(tensor: np.ndarray) -> np.ndarray:
    values = np.asarray(tensor, dtype=float)
    return np.asarray(
        [
            values[0, 0],
            values[1, 1],
            values[2, 2],
            2.0 * values[0, 1],
            2.0 * values[1, 2],
            2.0 * values[0, 2],
        ]
    )


def _stress_voigt_to_tensor(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    return np.asarray(
        [
            [values[0], values[3], values[5]],
            [values[3], values[1], values[4]],
            [values[5], values[4], values[2]],
        ]
    )


def _tensor_to_stress_voigt(tensor: np.ndarray) -> np.ndarray:
    values = np.asarray(tensor, dtype=float)
    return np.asarray(
        [
            values[0, 0],
            values[1, 1],
            values[2, 2],
            values[0, 1],
            values[1, 2],
            values[0, 2],
        ]
    )


def _rotation_from_config(config: dict[str, Any]) -> np.ndarray:
    supplied = np.asarray(
        config.get("orientation_matrix_material_to_lab", []), dtype=float
    )
    if supplied.size:
        return supplied
    rx, ry, rz = np.radians(
        np.asarray(
            config.get("orientation_euler_xyz_deg", [0.0, 0.0, 0.0]),
            dtype=float,
        )
    )
    rotation_x = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, math.cos(rx), -math.sin(rx)], [0.0, math.sin(rx), math.cos(rx)]]
    )
    rotation_y = np.asarray(
        [[math.cos(ry), 0.0, math.sin(ry)], [0.0, 1.0, 0.0], [-math.sin(ry), 0.0, math.cos(ry)]]
    )
    rotation_z = np.asarray(
        [[math.cos(rz), -math.sin(rz), 0.0], [math.sin(rz), math.cos(rz), 0.0], [0.0, 0.0, 1.0]]
    )
    return rotation_z @ rotation_y @ rotation_x


def rotate_stiffness_to_lab(
    stiffness_material: np.ndarray, rotation_material_to_lab: np.ndarray
) -> np.ndarray:
    """Rotate a 6x6 engineering-strain stiffness operator to lab axes."""

    stiffness = np.asarray(stiffness_material, dtype=float)
    rotation = np.asarray(rotation_material_to_lab, dtype=float)
    result = np.zeros((6, 6), dtype=float)
    for column in range(6):
        lab_strain_vector = np.zeros(6, dtype=float)
        lab_strain_vector[column] = 1.0
        lab_strain = _engineering_strain_to_tensor(lab_strain_vector)
        material_strain = rotation.T @ lab_strain @ rotation
        material_stress_vector = stiffness @ _tensor_to_engineering_strain(
            material_strain
        )
        material_stress = _stress_voigt_to_tensor(material_stress_vector)
        lab_stress = rotation @ material_stress @ rotation.T
        result[:, column] = _tensor_to_stress_voigt(lab_stress)
    return 0.5 * (result + result.T)


def rotate_expansion_to_lab(
    expansion_material: np.ndarray, rotation_material_to_lab: np.ndarray
) -> np.ndarray:
    tensor = _engineering_strain_to_tensor(expansion_material)
    rotation = np.asarray(rotation_material_to_lab, dtype=float)
    return _tensor_to_engineering_strain(rotation @ tensor @ rotation.T)


def rotate_photoelastic_to_lab(
    photoelastic_material: np.ndarray, rotation_material_to_lab: np.ndarray
) -> np.ndarray:
    """Rotate a matrix mapping engineering strain to impermeability tensor."""

    operator = np.asarray(photoelastic_material, dtype=float)
    rotation = np.asarray(rotation_material_to_lab, dtype=float)
    result = np.zeros((6, 6), dtype=float)
    for column in range(6):
        lab_strain_vector = np.zeros(6, dtype=float)
        lab_strain_vector[column] = 1.0
        lab_strain = _engineering_strain_to_tensor(lab_strain_vector)
        material_strain = rotation.T @ lab_strain @ rotation
        material_impermeability = _stress_voigt_to_tensor(
            operator @ _tensor_to_engineering_strain(material_strain)
        )
        lab_impermeability = rotation @ material_impermeability @ rotation.T
        result[:, column] = _tensor_to_stress_voigt(lab_impermeability)
    return result


StressFactory = Callable[[dict], "StressSolver"]
STRESS_REGISTRY: dict[str, StressFactory] = {}


def register_stress_solver(name: str, factory: StressFactory) -> None:
    STRESS_REGISTRY[name] = factory


class StressSolver:
    approximation = "unknown"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.grid: Any | None = None
        self.geometry: dict[str, Any] = {}

    def solve(self, temperature_K: np.ndarray, reference_K: float) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def solve_on_material(
        self,
        temperature_K: np.ndarray,
        reference_K: float,
        material_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Solve on a material mask while retaining old plugin compatibility.

        Third-party stress plugins written for the original two-argument
        ``solve`` API continue to work. Built-in solvers that need to
        distinguish crystal from the surrounding computational box override
        this method.
        """

        del material_mask
        return self.solve(temperature_K, reference_K)


class NullStressSolver(StressSolver):
    approximation = "disabled"

    def solve(self, temperature_K: np.ndarray, reference_K: float) -> dict[str, np.ndarray]:
        zeros = np.zeros_like(temperature_K)
        return {"von_mises_Pa": zeros, "delta_n_stress": zeros}


class IsotropicProxyStressSolver(StressSolver):
    """Fast stress-risk proxy, not a mechanical equilibrium/FEM solution.

    The proxy removes each z-plane's mean temperature and evaluates the local
    constrained thermal stress scale E*alpha*dT/(1-nu). It is useful for finding
    hot-gradient regions, but not for crystal fracture qualification.
    """

    approximation = "local isotropic plane-strain risk proxy; replace with FEM for design limits"

    def solve(self, temperature_K: np.ndarray, reference_K: float) -> dict[str, np.ndarray]:
        return self.solve_on_material(
            temperature_K,
            reference_K,
            np.ones_like(temperature_K, dtype=bool),
        )

    def solve_on_material(
        self,
        temperature_K: np.ndarray,
        reference_K: float,
        material_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        del reference_K
        temperature = np.asarray(temperature_K, dtype=float)
        mask = np.asarray(material_mask, dtype=bool)
        if mask.shape != temperature.shape:
            raise ValueError("stress material_mask must match the temperature grid")
        young = float(self.cfg["young_modulus_Pa"])
        poisson = float(self.cfg["poisson_ratio"])
        expansion = float(self.cfg["thermal_expansion_per_K"])
        cell_areas = (
            np.ones(temperature.shape[-2:], dtype=float)
            if self.grid is None else np.asarray(self.grid.cell_areas, dtype=float)
        )
        active_area = np.sum(mask * cell_areas, axis=(1, 2), keepdims=True)
        plane_mean = np.divide(
            np.sum(np.where(mask, temperature, 0.0) * cell_areas, axis=(1, 2), keepdims=True),
            active_area,
            out=np.zeros_like(active_area),
            where=active_area > 0.0,
        )
        temperature_departure = np.where(mask, temperature - plane_mean, 0.0)
        stress = np.abs(
            young * expansion * temperature_departure / (1.0 - poisson)
        )
        delta_n = float(self.cfg.get("photoelastic_coefficient_Pa_inv", 0.0)) * stress
        return {"von_mises_Pa": stress, "delta_n_stress": delta_n}


class ThermoelasticFEMStressSolver(StressSolver):
    """Small-strain 3D isotropic thermoelastic hexahedral FEM.

    Every active thermal cell is one trilinear eight-node element.  Exterior
    surfaces are traction-free unless explicitly listed in ``fixed_faces``.
    With no fixed face, six rigid-body modes are removed by the standard
    three-point minimal constraint; this does not prevent uniform free thermal
    expansion.  The implementation deliberately fails when the requested mesh
    exceeds its explicit element budget instead of silently coarsening stress
    hot spots.
    """

    approximation = (
        "3D small-strain isotropic linear thermoelastic hexahedral FEM with "
        "traction-free natural boundaries and optional fixed faces"
    )

    _FACE_NAMES = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
    _NODE_SIGNS = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._fem_operator_cache: dict[str, Any] | None = None
        self._last_reduced_displacement: np.ndarray | None = None

    def reset_initial_guess(self) -> None:
        """Reset CG warm-start state without discarding FEM factorizations."""

        self._last_reduced_displacement = None

    def _fem_operator(self, mask: np.ndarray) -> tuple[dict[str, Any], bool]:
        cached = self._fem_operator_cache
        if cached is not None and np.array_equal(cached["mask"], mask):
            return cached, True

        active_cells = np.argwhere(mask)
        fem_cfg = self.cfg.get("thermoelastic_fem", {})
        maximum_elements = int(fem_cfg.get("maximum_active_elements", 8000))
        if len(active_cells) > maximum_elements:
            raise ValueError(
                "thermoelastic_fem active-element count "
                f"{len(active_cells)} exceeds maximum_active_elements={maximum_elements}; "
                "use a coarser physical grid (the optical grid may remain fine) or raise the "
                "explicit mechanics budget after a convergence check"
            )

        try:
            from scipy.ndimage import label
            from scipy.sparse import coo_matrix
            from scipy.sparse.linalg import LinearOperator, factorized
        except ImportError as exc:  # pragma: no cover - optional dependency contract
            raise RuntimeError("thermoelastic_fem requires SciPy sparse solvers") from exc

        face_connected = np.zeros((3, 3, 3), dtype=int)
        face_connected[1, 1, 1] = 1
        face_connected[0, 1, 1] = face_connected[2, 1, 1] = 1
        face_connected[1, 0, 1] = face_connected[1, 2, 1] = 1
        face_connected[1, 1, 0] = face_connected[1, 1, 2] = 1
        _, component_count = label(mask, structure=face_connected)
        if int(component_count) != 1:
            raise ValueError(
                "thermoelastic_fem material mask must contain exactly one "
                f"face-connected mechanical component; found {int(component_count)}"
            )

        assembly_started = time.perf_counter()
        constitutive, expansion = self._material_tensors_lab()
        connectivity, node_coordinates = self._active_connectivity(active_cells)
        node_count = len(node_coordinates)
        dof_count = 3 * node_count
        entries_per_element = 24 * 24
        total_entries = len(active_cells) * entries_per_element
        rows = np.empty(total_entries, dtype=np.int64)
        columns = np.empty(total_entries, dtype=np.int64)
        values = np.empty(total_entries, dtype=float)
        load_rows = np.empty(len(active_cells) * 24, dtype=np.int64)
        load_columns = np.repeat(
            np.arange(len(active_cells), dtype=np.int64), 24
        )
        load_values = np.empty(len(active_cells) * 24, dtype=float)
        element_cache: dict[
            tuple[float, float, float], tuple[np.ndarray, np.ndarray]
        ] = {}

        for element_index, ((iz, iy, ix), nodes) in enumerate(
            zip(active_cells, connectivity, strict=True)
        ):
            dimensions = (
                float(self.grid.x_widths[ix]),
                float(self.grid.y_widths[iy]),
                float(self.grid.z_widths[iz]),
            )
            local = element_cache.get(dimensions)
            if local is None:
                local = self._element_matrices(
                    dimensions,
                    constitutive,
                    expansion,
                )
                element_cache[dimensions] = local
            stiffness, unit_thermal_load = local
            dofs = self._element_dofs(nodes)
            start = element_index * entries_per_element
            stop = start + entries_per_element
            rows[start:stop] = np.repeat(dofs, len(dofs))
            columns[start:stop] = np.tile(dofs, len(dofs))
            values[start:stop] = stiffness.ravel()
            load_start = element_index * 24
            load_stop = load_start + 24
            load_rows[load_start:load_stop] = dofs
            load_values[load_start:load_stop] = unit_thermal_load

        stiffness_global = coo_matrix(
            (values, (rows, columns)),
            shape=(dof_count, dof_count),
        ).tocsr()
        thermal_load_operator = coo_matrix(
            (load_values, (load_rows, load_columns)),
            shape=(dof_count, len(active_cells)),
        ).tocsr()
        constrained = self._constrained_dofs(node_coordinates)
        free = np.ones(dof_count, dtype=bool)
        free[constrained] = False
        if not np.any(free):
            raise ValueError(
                "thermoelastic FEM boundary conditions constrain every degree of freedom"
            )
        reduced = stiffness_global[free][:, free].tocsr()
        solver_name = str(fem_cfg.get("linear_solver", "auto"))
        if solver_name == "auto":
            solver_name = "direct" if reduced.shape[0] <= 12000 else "cg"
        if solver_name not in {"direct", "cg"}:
            raise ValueError(
                "thermoelastic_fem.linear_solver must be auto, direct, or cg"
            )
        assembly_elapsed = time.perf_counter() - assembly_started

        factorization_started = time.perf_counter()
        direct_factor = None
        preconditioner = None
        if solver_name == "direct":
            direct_factor = factorized(reduced.tocsc())
        else:
            diagonal = np.asarray(reduced.diagonal(), dtype=float)
            inverse_diagonal = np.divide(
                1.0,
                diagonal,
                out=np.zeros_like(diagonal),
                where=np.abs(diagonal) > np.finfo(float).tiny,
            )
            preconditioner = LinearOperator(
                reduced.shape,
                matvec=lambda vector: inverse_diagonal * vector,
            )
        factorization_elapsed = time.perf_counter() - factorization_started
        cached = {
            "mask": mask.copy(),
            "active_cells": active_cells,
            "connectivity": connectivity,
            "node_coordinates": node_coordinates,
            "node_count": node_count,
            "dof_count": dof_count,
            "constitutive": constitutive,
            "expansion": expansion,
            "thermal_load_operator": thermal_load_operator,
            "constrained": constrained,
            "free": free,
            "reduced": reduced,
            "solver_name": solver_name,
            "direct_factor": direct_factor,
            "preconditioner": preconditioner,
            "assembly_time_s": assembly_elapsed,
            "factorization_time_s": factorization_elapsed,
        }
        self._fem_operator_cache = cached
        self._last_reduced_displacement = None
        return cached, False

    def solve(self, temperature_K: np.ndarray, reference_K: float) -> dict[str, np.ndarray]:
        return self.solve_on_material(
            temperature_K,
            reference_K,
            np.ones_like(temperature_K, dtype=bool),
        )

    @profiled_operation("stress.thermoelastic_fem")
    def solve_on_material(
        self,
        temperature_K: np.ndarray,
        reference_K: float,
        material_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if self.grid is None:
            raise RuntimeError("thermoelastic_fem requires the physical Grid3D")
        temperature = np.asarray(temperature_K, dtype=float)
        mask = np.asarray(material_mask, dtype=bool)
        if temperature.shape != self.grid.shape or mask.shape != self.grid.shape:
            raise ValueError("thermoelastic FEM temperature and mask must match Grid3D")
        if not np.all(np.isfinite(temperature[mask])):
            raise ValueError("thermoelastic FEM temperature must be finite in the material")
        active_cells = np.argwhere(mask)
        if len(active_cells) == 0:
            return self._zero_result(temperature.shape)
        fem_cfg = self.cfg.get("thermoelastic_fem", {})
        operator, operator_reused = self._fem_operator(mask)
        active_cells = operator["active_cells"]
        connectivity = operator["connectivity"]
        node_coordinates = operator["node_coordinates"]
        node_count = int(operator["node_count"])
        dof_count = int(operator["dof_count"])
        constitutive = operator["constitutive"]
        expansion = operator["expansion"]
        free = operator["free"]
        constrained = operator["constrained"]
        reduced = operator["reduced"]
        solver_name = str(operator["solver_name"])
        delta_temperature = temperature[tuple(active_cells.T)] - reference_K
        load = np.asarray(
            operator["thermal_load_operator"] @ delta_temperature,
            dtype=float,
        ).ravel()
        reduced_load = load[free]
        iterations = 0
        solve_started = time.perf_counter()
        if solver_name == "direct":
            reduced_displacement = np.asarray(
                operator["direct_factor"](reduced_load), dtype=float
            )
            solver_info = 0
        elif solver_name == "cg":
            try:
                from scipy.sparse.linalg import cg
            except ImportError as exc:  # pragma: no cover - optional dependency contract
                raise RuntimeError(
                    "thermoelastic_fem requires SciPy sparse solvers"
                ) from exc

            def count_iteration(_: np.ndarray) -> None:
                nonlocal iterations
                iterations += 1

            relative_tolerance = float(
                fem_cfg.get("relative_tolerance", 1e-8)
            )
            tolerance_keyword = (
                {"rtol": relative_tolerance, "atol": 0.0}
                if "rtol" in inspect.signature(cg).parameters
                else {"tol": relative_tolerance, "atol": 0.0}
            )
            reduced_displacement, solver_info = cg(
                reduced,
                reduced_load,
                x0=self._last_reduced_displacement,
                maxiter=int(fem_cfg.get("maximum_iterations", 2000)),
                M=operator["preconditioner"],
                callback=count_iteration,
                **tolerance_keyword,
            )
        solve_elapsed = time.perf_counter() - solve_started
        if solver_info != 0 or not np.all(np.isfinite(reduced_displacement)):
            raise RuntimeError(
                "thermoelastic FEM linear solve did not converge "
                f"(solver={solver_name}, info={solver_info}, iterations={iterations})"
            )
        self._last_reduced_displacement = np.asarray(
            reduced_displacement, dtype=float
        ).copy()
        displacement = np.zeros(dof_count, dtype=float)
        displacement[free] = reduced_displacement
        residual = reduced @ reduced_displacement - reduced_load
        equilibrium_residual = float(
            np.linalg.norm(residual)
            / max(np.linalg.norm(reduced_load), np.finfo(float).tiny)
        )
        result = self._recover_cell_fields(
            temperature,
            reference_K,
            mask,
            active_cells,
            connectivity,
            displacement,
            constitutive,
            expansion,
        )
        result.update(
            self._endface_displacement_fields(node_coordinates, displacement)
        )
        result.update(
            {
                "fem_equilibrium_residual_relative": np.asarray(equilibrium_residual),
                "fem_active_elements": np.asarray(len(active_cells)),
                "fem_active_nodes": np.asarray(node_count),
                "fem_free_dofs": np.asarray(int(np.sum(free))),
                "fem_constrained_dofs": np.asarray(len(constrained)),
                "fem_linear_iterations": np.asarray(iterations),
                "fem_converged": np.asarray(True),
                "fem_operator_assembly_reused": np.asarray(operator_reused),
                "fem_factorization_reused": np.asarray(
                    operator_reused and solver_name == "direct"
                ),
                "fem_preconditioner_reused": np.asarray(
                    operator_reused and solver_name == "cg"
                ),
                "fem_operator_assembly_time_s": np.asarray(
                    0.0 if operator_reused else operator["assembly_time_s"]
                ),
                "fem_factorization_time_s": np.asarray(
                    0.0 if operator_reused else operator["factorization_time_s"]
                ),
                "fem_linear_solve_time_s": np.asarray(solve_elapsed),
            }
        )
        return result

    def _material_tensors_lab(self) -> tuple[np.ndarray, np.ndarray]:
        young = float(self.cfg["young_modulus_Pa"])
        poisson = float(self.cfg["poisson_ratio"])
        expansion = float(self.cfg["thermal_expansion_per_K"])
        return self._constitutive_matrix(young, poisson), np.asarray(
            [expansion, expansion, expansion, 0.0, 0.0, 0.0], dtype=float
        )

    def _active_connectivity(
        self,
        active_cells: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nz, ny, nx = self.grid.shape
        global_connectivity = np.empty((len(active_cells), 8), dtype=np.int64)
        offsets = np.asarray(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
                [0, 1, 0],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 1],
                [1, 1, 0],
            ],
            dtype=np.int64,
        )
        for index, cell in enumerate(active_cells):
            nodes = cell[None, :] + offsets
            global_connectivity[index] = np.ravel_multi_index(
                (nodes[:, 0], nodes[:, 1], nodes[:, 2]),
                (nz + 1, ny + 1, nx + 1),
            )
        global_nodes = np.unique(global_connectivity)
        compact = np.full((nz + 1) * (ny + 1) * (nx + 1), -1, dtype=np.int64)
        compact[global_nodes] = np.arange(len(global_nodes), dtype=np.int64)
        connectivity = compact[global_connectivity]
        iz, iy, ix = np.unravel_index(global_nodes, (nz + 1, ny + 1, nx + 1))
        coordinates = np.column_stack(
            (
                self.grid.x_edges[ix],
                self.grid.y_edges[iy],
                self.grid.z_edges[iz],
            )
        )
        return connectivity, coordinates

    @staticmethod
    def _constitutive_matrix(young: float, poisson: float) -> np.ndarray:
        lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        shear = young / (2.0 * (1.0 + poisson))
        matrix = np.zeros((6, 6), dtype=float)
        matrix[:3, :3] = lame
        np.fill_diagonal(matrix[:3, :3], lame + 2.0 * shear)
        matrix[3:, 3:] = np.eye(3) * shear
        return matrix

    @classmethod
    def _strain_matrix(
        cls,
        dimensions: tuple[float, float, float],
        xi: float,
        eta: float,
        zeta: float,
    ) -> np.ndarray:
        hx, hy, hz = dimensions
        signs = cls._NODE_SIGNS
        derivatives_natural = np.column_stack(
            (
                signs[:, 0] * (1.0 + signs[:, 1] * eta) * (1.0 + signs[:, 2] * zeta) / 8.0,
                signs[:, 1] * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 2] * zeta) / 8.0,
                signs[:, 2] * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 1] * eta) / 8.0,
            )
        )
        derivatives = derivatives_natural * np.asarray(
            [2.0 / hx, 2.0 / hy, 2.0 / hz]
        )
        matrix = np.zeros((6, 24), dtype=float)
        for node, (dx, dy, dz) in enumerate(derivatives):
            base = 3 * node
            matrix[0, base] = dx
            matrix[1, base + 1] = dy
            matrix[2, base + 2] = dz
            matrix[3, base] = dy
            matrix[3, base + 1] = dx
            matrix[4, base + 1] = dz
            matrix[4, base + 2] = dy
            matrix[5, base] = dz
            matrix[5, base + 2] = dx
        return matrix

    @classmethod
    def _element_matrices(
        cls,
        dimensions: tuple[float, float, float],
        constitutive: np.ndarray,
        expansion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        stiffness = np.zeros((24, 24), dtype=float)
        thermal_load = np.zeros(24, dtype=float)
        determinant = math.prod(dimensions) / 8.0
        thermal_strain_per_K = np.asarray(expansion, dtype=float)
        point = 1.0 / math.sqrt(3.0)
        for xi in (-point, point):
            for eta in (-point, point):
                for zeta in (-point, point):
                    strain_matrix = cls._strain_matrix(dimensions, xi, eta, zeta)
                    stiffness += (
                        np.linalg.multi_dot(
                            (strain_matrix.T, constitutive, strain_matrix)
                        )
                        * determinant
                    )
                    thermal_load += (
                        strain_matrix.T
                        @ constitutive
                        @ thermal_strain_per_K
                        * determinant
                    )
        return stiffness, thermal_load

    @staticmethod
    def _element_dofs(nodes: np.ndarray) -> np.ndarray:
        return (3 * nodes[:, None] + np.arange(3, dtype=np.int64)).ravel()

    def _constrained_dofs(self, coordinates: np.ndarray) -> np.ndarray:
        fem_cfg = self.cfg.get("thermoelastic_fem", {})
        fixed_faces = list(fem_cfg.get("fixed_faces", []))
        constraints: set[int] = set()
        if fixed_faces:
            limits = {
                "x_min": (0, float(np.min(coordinates[:, 0]))),
                "x_max": (0, float(np.max(coordinates[:, 0]))),
                "y_min": (1, float(np.min(coordinates[:, 1]))),
                "y_max": (1, float(np.max(coordinates[:, 1]))),
                "z_min": (2, float(np.min(coordinates[:, 2]))),
                "z_max": (2, float(np.max(coordinates[:, 2]))),
            }
            for face in fixed_faces:
                axis, value = limits[face]
                selected = np.flatnonzero(
                    np.isclose(coordinates[:, axis], value, rtol=0.0, atol=1e-14)
                )
                for node in selected:
                    constraints.update((3 * int(node), 3 * int(node) + 1, 3 * int(node) + 2))
            if not constraints:
                raise ValueError("thermoelastic_fem.fixed_faces selected no material nodes")
            return np.asarray(sorted(constraints), dtype=np.int64)

        # Minimal rigid-body stabilization.  Choose A near the transverse
        # centre of the lowest-z surface, B along +x through A, and C along +y
        # through A so a uniform free expansion satisfies every constraint.
        z_min = float(np.min(coordinates[:, 2]))
        bottom = np.flatnonzero(np.isclose(coordinates[:, 2], z_min, atol=1e-14, rtol=0.0))
        center_x = 0.5 * (float(np.min(coordinates[bottom, 0])) + float(np.max(coordinates[bottom, 0])))
        center_y = 0.5 * (float(np.min(coordinates[bottom, 1])) + float(np.max(coordinates[bottom, 1])))
        a = int(bottom[np.argmin((coordinates[bottom, 0] - center_x) ** 2 + (coordinates[bottom, 1] - center_y) ** 2)])
        same_y = bottom[np.isclose(coordinates[bottom, 1], coordinates[a, 1], atol=1e-14, rtol=0.0)]
        same_x = bottom[np.isclose(coordinates[bottom, 0], coordinates[a, 0], atol=1e-14, rtol=0.0)]
        if len(same_y) < 2 or len(same_x) < 2:
            raise ValueError(
                "thermoelastic FEM could not construct non-collinear traction-free rigid-body constraints; "
                "specify at least one fixed face or use a connected Cartesian material mask"
            )
        b = int(same_y[np.argmax(np.abs(coordinates[same_y, 0] - coordinates[a, 0]))])
        c = int(same_x[np.argmax(np.abs(coordinates[same_x, 1] - coordinates[a, 1]))])
        return np.asarray(
            sorted({3 * a, 3 * a + 1, 3 * a + 2, 3 * b + 1, 3 * b + 2, 3 * c + 2}),
            dtype=np.int64,
        )

    def _recover_cell_fields(
        self,
        temperature: np.ndarray,
        reference_K: float,
        mask: np.ndarray,
        active_cells: np.ndarray,
        connectivity: np.ndarray,
        displacement: np.ndarray,
        constitutive: np.ndarray,
        expansion: np.ndarray,
    ) -> dict[str, np.ndarray]:
        shape = temperature.shape
        names = (
            "stress_xx_Pa",
            "stress_yy_Pa",
            "stress_zz_Pa",
            "stress_xy_Pa",
            "stress_yz_Pa",
            "stress_xz_Pa",
            "strain_xx",
            "strain_yy",
            "strain_zz",
            "strain_xy_engineering",
            "strain_yz_engineering",
            "strain_xz_engineering",
            "displacement_x_m",
            "displacement_y_m",
            "displacement_z_m",
        )
        elastic_strain_names = (
            "elastic_strain_xx",
            "elastic_strain_yy",
            "elastic_strain_zz",
            "elastic_strain_xy_engineering",
            "elastic_strain_yz_engineering",
            "elastic_strain_xz_engineering",
        )
        fields = {
            name: np.zeros(shape, dtype=float)
            for name in (*names, *elastic_strain_names)
        }
        von_mises = np.zeros(shape, dtype=float)
        for (iz, iy, ix), nodes in zip(active_cells, connectivity, strict=True):
            dimensions = (
                float(self.grid.x_widths[ix]),
                float(self.grid.y_widths[iy]),
                float(self.grid.z_widths[iz]),
            )
            dofs = self._element_dofs(nodes)
            local_displacement = displacement[dofs]
            strain = self._strain_matrix(dimensions, 0.0, 0.0, 0.0) @ local_displacement
            thermal_strain = np.asarray(expansion, dtype=float) * (
                temperature[iz, iy, ix] - reference_K
            )
            stress = constitutive @ (strain - thermal_strain)
            cell = (int(iz), int(iy), int(ix))
            for field_name, value in zip(names[:6], stress, strict=True):
                fields[field_name][cell] = value
            for field_name, value in zip(names[6:12], strain, strict=True):
                fields[field_name][cell] = value
            for field_name, value in zip(
                elastic_strain_names, strain - thermal_strain, strict=True
            ):
                fields[field_name][cell] = value
            nodal_displacement = local_displacement.reshape(8, 3)
            for axis, field_name in enumerate(names[12:]):
                fields[field_name][cell] = float(np.mean(nodal_displacement[:, axis]))
            sxx, syy, szz, txy, tyz, txz = stress
            von_mises[cell] = math.sqrt(
                max(
                    0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                    + 3.0 * (txy**2 + tyz**2 + txz**2),
                    0.0,
                )
            )
        coupling = str(
            self.cfg.get("thermoelastic_fem", {}).get(
                "photoelastic_coupling", "off"
            )
        )
        coefficient = (
            float(self.cfg.get("photoelastic_coefficient_Pa_inv", 0.0))
            if coupling == "scalar_von_mises_proxy"
            else 0.0
        )
        fields["von_mises_Pa"] = von_mises
        fields["delta_n_stress"] = coefficient * von_mises
        if coupling == "tensor_photoelastic":
            fields.update(self._tensor_photoelastic_fields(fields, mask))
        axial_displacement = fields["displacement_z_m"]
        active_axial = axial_displacement[mask]
        fields["axial_displacement_peak_to_valley_m"] = np.asarray(
            float(np.ptp(active_axial)) if active_axial.size else 0.0
        )
        return fields

    def _tensor_photoelastic_fields(
        self,
        fields: dict[str, np.ndarray],
        mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        raise ValueError(
            "tensor_photoelastic requires anisotropic_thermoelastic_fem"
        )

    def _endface_displacement_fields(
        self,
        node_coordinates: np.ndarray,
        displacement: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Sample FEM nodal z displacement at front/back cell footprints."""

        coordinates = np.asarray(node_coordinates, dtype=float)
        nodal = np.asarray(displacement, dtype=float).reshape((-1, 3))
        maps: dict[str, np.ndarray] = {}
        for name, surface_z in (
            ("front_surface_displacement_z_m", float(self.grid.z_edges[0])),
            ("back_surface_displacement_z_m", float(self.grid.z_edges[-1])),
        ):
            selected = np.flatnonzero(
                np.isclose(coordinates[:, 2], surface_z, rtol=0.0, atol=1e-14)
            )
            node_values = {
                (round(float(coordinates[index, 0]), 15), round(float(coordinates[index, 1]), 15)): float(nodal[index, 2])
                for index in selected
            }
            surface = np.full((len(self.grid.y), len(self.grid.x)), np.nan)
            for iy in range(len(self.grid.y)):
                for ix in range(len(self.grid.x)):
                    corners = [
                        node_values.get(
                            (
                                round(float(self.grid.x_edges[jx]), 15),
                                round(float(self.grid.y_edges[jy]), 15),
                            )
                        )
                        for jy, jx in (
                            (iy, ix),
                            (iy, ix + 1),
                            (iy + 1, ix),
                            (iy + 1, ix + 1),
                        )
                    ]
                    available = [value for value in corners if value is not None]
                    if available:
                        surface[iy, ix] = float(np.mean(available))
            maps[name] = surface
        difference = maps["back_surface_displacement_z_m"] - maps[
            "front_surface_displacement_z_m"
        ]
        maps["endface_sag_difference_m"] = difference
        finite = difference[np.isfinite(difference)]
        maps["endface_sag_peak_to_valley_m"] = np.asarray(
            float(np.ptp(finite)) if finite.size else 0.0
        )
        return maps

    @staticmethod
    def _zero_result(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
        zeros = np.zeros(shape, dtype=float)
        return {
            "von_mises_Pa": zeros.copy(),
            "delta_n_stress": zeros.copy(),
            "fem_equilibrium_residual_relative": np.asarray(0.0),
            "fem_active_elements": np.asarray(0),
            "fem_active_nodes": np.asarray(0),
            "fem_free_dofs": np.asarray(0),
            "fem_constrained_dofs": np.asarray(0),
            "fem_linear_iterations": np.asarray(0),
            "fem_converged": np.asarray(True),
        }


class AnisotropicThermoelasticFEMStressSolver(ThermoelasticFEMStressSolver):
    """3-D thermoelastic FEM with rotated stiffness/expansion/photoelastic tensors."""

    approximation = (
        "3D small-strain anisotropic linear thermoelastic hexahedral FEM with "
        "user-supplied crystal tensors and orientation"
    )

    def _material_tensors_lab(self) -> tuple[np.ndarray, np.ndarray]:
        config = self.cfg.get("anisotropic", {})
        rotation = _rotation_from_config(config)
        stiffness = rotate_stiffness_to_lab(
            np.asarray(config["elastic_stiffness_Pa"], dtype=float), rotation
        )
        expansion = rotate_expansion_to_lab(
            np.asarray(config["thermal_expansion_tensor_per_K"], dtype=float),
            rotation,
        )
        return stiffness, expansion

    def _tensor_photoelastic_fields(
        self,
        fields: dict[str, np.ndarray],
        mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        config = self.cfg.get("anisotropic", {})
        rotation = _rotation_from_config(config)
        photoelastic = rotate_photoelastic_to_lab(
            np.asarray(config["photoelastic_matrix"], dtype=float), rotation
        )
        strain_definition = str(
            config.get("photoelastic_strain_definition", "elastic_strain")
        )
        prefix = "elastic_strain" if strain_definition == "elastic_strain" else "strain"
        strain = np.stack(
            [
                fields[f"{prefix}_xx"],
                fields[f"{prefix}_yy"],
                fields[f"{prefix}_zz"],
                fields[f"{prefix}_xy_engineering"],
                fields[f"{prefix}_yz_engineering"],
                fields[f"{prefix}_xz_engineering"],
            ],
            axis=0,
        )
        impermeability = np.einsum("ab,b...->a...", photoelastic, strain)
        refractive_index = float(config["reference_refractive_index"])
        scale = -0.5 * refractive_index**3
        delta_xx = scale * impermeability[0]
        delta_yy = scale * impermeability[1]
        delta_zz = scale * impermeability[2]
        delta_xy = scale * impermeability[3]
        delta_yz = scale * impermeability[4]
        delta_xz = scale * impermeability[5]
        common = 0.5 * (delta_xx + delta_yy)
        birefringence = 2.0 * np.sqrt(
            (0.5 * (delta_xx - delta_yy)) ** 2 + delta_xy**2
        )
        result = {
            "delta_n_stress": np.where(mask, common, 0.0),
            "delta_n_photoelastic_xx": np.where(mask, delta_xx, 0.0),
            "delta_n_photoelastic_yy": np.where(mask, delta_yy, 0.0),
            "delta_n_photoelastic_zz": np.where(mask, delta_zz, 0.0),
            "delta_n_photoelastic_xy": np.where(mask, delta_xy, 0.0),
            "delta_n_photoelastic_yz": np.where(mask, delta_yz, 0.0),
            "delta_n_photoelastic_xz": np.where(mask, delta_xz, 0.0),
            "photoelastic_birefringence_magnitude": np.where(
                mask, birefringence, 0.0
            ),
            "photoelastic_orientation_matrix_material_to_lab": rotation,
            "photoelastic_strain_definition": np.asarray(strain_definition),
        }
        return result


STRESS_REGISTRY.update(
    {
        "none": NullStressSolver,
        "isotropic_proxy": IsotropicProxyStressSolver,
        "thermoelastic_fem": ThermoelasticFEMStressSolver,
        "anisotropic_thermoelastic_fem": (
            AnisotropicThermoelasticFEMStressSolver
        ),
    }
)


def make_stress_solver(
    cfg: dict,
    grid: Any | None = None,
    geometry: dict[str, Any] | None = None,
) -> StressSolver:
    name = cfg.get("stress_model", "none")
    try:
        solver = STRESS_REGISTRY[name](cfg)
    except KeyError as exc:
        raise ValueError(f"Unknown stress model {name!r}; registered: {sorted(STRESS_REGISTRY)}") from exc
    solver.grid = grid
    solver.geometry = {} if geometry is None else dict(geometry)
    return solver
