"""Cartesian crystal grid."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .validation import require_exact_integer


@dataclass(frozen=True)
class Grid3D:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    dx: float
    dy: float
    dz: float
    x_edges: np.ndarray
    y_edges: np.ndarray
    z_edges: np.ndarray

    def __post_init__(self) -> None:
        """Enforce the coordinate contract at every construction boundary.

        Result archives and tests may construct ``Grid3D`` directly rather
        than through :meth:`from_config`.  Keeping the invariants here avoids
        accepting a grid whose centres, edges, or nominal spacings describe
        different physical cells.
        """

        for name in ("x", "y", "z", "x_edges", "y_edges", "z_edges"):
            values = np.asarray(getattr(self, name), dtype=float)
            object.__setattr__(self, name, values)
            if values.ndim != 1:
                raise ValueError(f"Grid3D.{name} must be one-dimensional")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Grid3D.{name} must contain only finite values")

        for axis, spacing_name in (("x", "dx"), ("y", "dy"), ("z", "dz")):
            centers = getattr(self, axis)
            edges = getattr(self, f"{axis}_edges")
            if len(centers) < 1 or len(edges) != len(centers) + 1:
                raise ValueError(
                    f"Grid3D.{axis}_edges must contain exactly one more value "
                    f"than Grid3D.{axis}"
                )
            widths = np.diff(edges)
            if np.any(widths <= 0.0):
                raise ValueError(f"Grid3D.{axis}_edges must be strictly increasing")
            expected_centers = 0.5 * (edges[:-1] + edges[1:])
            # Scale the absolute tolerance to the coordinates themselves.  A
            # one-metre floor would silently accept picometre-scale midpoint
            # or spacing corruption in the micron/nanometre grids this class
            # is also expected to represent.
            scale = max(
                float(np.max(np.abs(edges))),
                float(np.ptp(edges)),
                float(np.max(widths)),
                np.finfo(float).tiny,
            )
            if not np.allclose(
                centers,
                expected_centers,
                rtol=64.0 * np.finfo(float).eps,
                atol=64.0 * np.finfo(float).eps * scale,
            ):
                raise ValueError(
                    f"Grid3D.{axis} must contain the adjacent-edge midpoints"
                )
            spacing = float(getattr(self, spacing_name))
            mean_width = float(np.mean(widths))
            if (
                not np.isfinite(spacing)
                or spacing <= 0.0
                or not np.isclose(
                    spacing,
                    mean_width,
                    rtol=1e-12,
                    atol=(
                        64.0
                        * np.finfo(float).eps
                        * max(mean_width, np.finfo(float).tiny)
                    ),
                )
            ):
                raise ValueError(
                    f"Grid3D.{spacing_name} must equal the mean {axis}-cell width"
                )

    @classmethod
    def from_config(cls, cfg: dict) -> "Grid3D":
        nx = require_exact_integer(cfg["nx"], name="grid.nx", minimum=1)
        ny = require_exact_integer(cfg["ny"], name="grid.ny", minimum=1)
        nz = require_exact_integer(cfg["nz"], name="grid.nz", minimum=1)
        size_x = float(cfg["size_x_m"])
        size_y = float(cfg["size_y_m"])
        length_z = float(cfg["length_z_m"])
        center_x = float(cfg.get("center_x_m", 0.0))
        center_y = float(cfg.get("center_y_m", 0.0))
        for name, value in (
            ("grid.size_x_m", size_x),
            ("grid.size_y_m", size_y),
            ("grid.length_z_m", length_z),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(center_x) or not math.isfinite(center_y):
            raise ValueError("grid transverse centres must be finite")
        refinement_x_raw = cfg.get("refinement_center_x_m")
        refinement_y_raw = cfg.get("refinement_center_y_m")
        refinement_x = (
            center_x if refinement_x_raw is None else float(refinement_x_raw)
        )
        refinement_y = (
            center_y if refinement_y_raw is None else float(refinement_y_raw)
        )
        if not math.isfinite(refinement_x) or not math.isfinite(refinement_y):
            raise ValueError("grid refinement centres must be finite or null")
        for axis, refinement, center, size in (
            ("x", refinement_x, center_x, size_x),
            ("y", refinement_y, center_y, size_y),
        ):
            lower = center - 0.5 * size
            upper = center + 0.5 * size
            if not lower < refinement < upper:
                raise ValueError(
                    f"grid.refinement_center_{axis}_m must lie strictly inside "
                    f"the physical grid {axis} extent"
                )
        spacing = str(cfg.get("spacing_xy", "uniform"))
        if spacing == "center_stretched":
            x_edges = _focused_stretched_edges(
                size_x,
                nx,
                float(cfg.get("stretch_x", 4.0)),
                center_x,
                refinement_x,
            )
            y_edges = _focused_stretched_edges(
                size_y,
                ny,
                float(cfg.get("stretch_y", 4.0)),
                center_y,
                refinement_y,
            )
        elif spacing == "refined_region":
            region = cfg.get("refined_region", {})
            if not isinstance(region, dict):
                raise ValueError("grid.refined_region must be an object")
            x_edges = _refined_region_edges(
                size_x, nx, center_x, refinement_x,
                region.get("size_x_m", 200e-6), region.get("nx", 24),
            )
            y_edges = _refined_region_edges(
                size_y, ny, center_y, refinement_y,
                region.get("size_y_m", 200e-6), region.get("ny", 24),
            )
        elif spacing == "uniform":
            x_edges = np.linspace(center_x - size_x / 2.0, center_x + size_x / 2.0, nx + 1)
            y_edges = np.linspace(center_y - size_y / 2.0, center_y + size_y / 2.0, ny + 1)
        else:
            raise ValueError("grid.spacing_xy must be uniform, center_stretched or refined_region")
        z_edges = np.linspace(0.0, length_z, nz + 1)
        x = 0.5 * (x_edges[:-1] + x_edges[1:])
        y = 0.5 * (y_edges[:-1] + y_edges[1:])
        z = 0.5 * (z_edges[:-1] + z_edges[1:])
        return cls(
            x=x,
            y=y,
            z=z,
            dx=size_x / nx,
            dy=size_y / ny,
            dz=length_z / nz,
            x_edges=x_edges,
            y_edges=y_edges,
            z_edges=z_edges,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.z), len(self.y), len(self.x)

    @cached_property
    def transverse_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y, indexing="xy")

    @property
    def cell_volume(self) -> float:
        """Mean cell volume; use ``cell_volumes`` for nonuniform integration."""

        return self.dx * self.dy * self.dz

    @cached_property
    def x_widths(self) -> np.ndarray:
        return np.diff(self.x_edges)

    @cached_property
    def y_widths(self) -> np.ndarray:
        return np.diff(self.y_edges)

    @cached_property
    def z_widths(self) -> np.ndarray:
        return np.diff(self.z_edges)

    @cached_property
    def cell_areas(self) -> np.ndarray:
        return self.y_widths[:, None] * self.x_widths[None, :]

    @cached_property
    def cell_volumes(self) -> np.ndarray:
        return self.z_widths[:, None, None] * self.cell_areas[None, :, :]

    @cached_property
    def is_uniform_xy(self) -> bool:
        return bool(
            np.allclose(self.x_widths, self.x_widths[0], rtol=1e-12, atol=0.0)
            and np.allclose(self.y_widths, self.y_widths[0], rtol=1e-12, atol=0.0)
        )

    def integrate_transverse(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.shape[-2:] != (len(self.y), len(self.x)):
            raise ValueError("transverse values must end in (ny, nx)")
        return np.sum(array * self.cell_areas, axis=(-2, -1))

    def integrate_volume(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.shape[-3:] != self.shape:
            raise ValueError("volume values must end in (nz, ny, nx)")
        return np.sum(array * self.cell_volumes, axis=(-3, -2, -1))


def _refined_region_edges(
    size_m: float, count: int, domain_center_m: float,
    refinement_center_m: float, region_size_m: float, region_count: int,
) -> np.ndarray:
    """Uniform core with geometrically graded outer cells and fixed boundaries.

    Allocate remaining cells to minimize the worst adjacent-cell growth ratio,
    including the core/outer interfaces. This avoids assigning too few cells
    to the short side of an off-axis core merely in proportion to its length.
    Each outer side receives at least two cells. No outer cell is finer than
    the core; incompatible widths/counts are rejected instead of clipped.
    """
    core_count = require_exact_integer(
        region_count, name="grid.refined_region cell count", minimum=2,
    )
    width = float(region_size_m)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("grid.refined_region size must be finite and positive")
    lower, upper = domain_center_m - size_m / 2, domain_center_m + size_m / 2
    core_lower = refinement_center_m - width / 2
    core_upper = refinement_center_m + width / 2
    if not lower < core_lower < core_upper < upper:
        raise ValueError("grid.refined_region must lie strictly inside the physical axis")
    remaining = count - core_count
    if remaining < 4:
        raise ValueError("grid.refined_region must leave at least four outer cells")
    step = width / core_count
    left_length, right_length = core_lower - lower, upper - core_upper

    def growth_log(length: float, n: int) -> float | None:
        target = length / step
        if target < n * (1 - 1e-12):
            return None
        if target <= n * (1 + 1e-12):
            return 0.0
        # Solve sum(exp(j*a), j=1..n) = length/core_step in log space.
        def log_expm1(value: float) -> float:
            return (value + math.log1p(-math.exp(-value))
                    if value > 50 else math.log(math.expm1(value)))
        lo, hi = 0.0, math.log(target)
        for _ in range(60):
            mid = (lo + hi) / 2
            value = mid + log_expm1(n * mid) - log_expm1(mid)
            if value < math.log(target):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    candidates = []
    for left_count in range(2, remaining - 1):
        right_count = remaining - left_count
        left_growth = growth_log(left_length, left_count)
        right_growth = growth_log(right_length, right_count)
        if left_growth is not None and right_growth is not None:
            candidates.append((max(left_growth, right_growth),
                               abs(left_growth - right_growth), left_count,
                               left_growth, right_growth))
    if not candidates:
        raise ValueError(
            "grid.refined_region cannot fit coarser outer cells; increase core "
            "cell count, reduce core width, or reduce total axis cell count"
        )
    _, _, left_count, left_growth, right_growth = min(candidates)
    right_count = remaining - left_count

    def outer_widths(length: float, n: int, growth: float) -> np.ndarray:
        widths = step * np.exp(growth * np.arange(1, n + 1))
        return widths * (length / float(np.sum(widths)))

    left_widths = outer_widths(left_length, left_count, left_growth)
    right_widths = outer_widths(right_length, right_count, right_growth)
    edges = np.concatenate((
        core_lower - np.cumsum(left_widths)[::-1],
        np.linspace(core_lower, core_upper, core_count + 1),
        core_upper + np.cumsum(right_widths),
    ))
    edges[0], edges[-1] = lower, upper
    if len(edges) != count + 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("grid.refined_region produced invalid cell edges")
    return edges


def _center_stretched_edges(size_m: float, count: int, edge_to_center_ratio: float) -> np.ndarray:
    """Symmetric sinh mapping with an edge/centre local-derivative ratio.

    The finite edge/centre cell-width ratio approaches this parameter as the
    cell count increases, but is not exactly equal to it on a finite grid.
    """

    ratio = float(edge_to_center_ratio)
    if not math.isfinite(ratio) or ratio < 1.0:
        raise ValueError("grid stretch ratios must be at least 1")
    computational = np.linspace(-1.0, 1.0, count + 1)
    if ratio <= 1.0 + 1e-12:
        mapped = computational
    else:
        strength = float(np.arccosh(ratio))
        mapped = np.sinh(strength * computational) / np.sinh(strength)
    return 0.5 * size_m * mapped


def _focused_stretched_edges(
    size_m: float,
    count: int,
    edge_to_focus_ratio: float,
    domain_center_m: float,
    refinement_center_m: float,
) -> np.ndarray:
    """Return fixed-domain sinh edges focused at an arbitrary interior point.

    The computational coordinate assigned to the refinement point is its
    fractional position in the physical interval.  That choice makes the
    mapping continuously differentiable at the focus: the limiting cell scale
    is the same from the left and right even when the focus is off-centre.  The
    outer-to-focus derivative ratio remains ``edge_to_focus_ratio`` on both
    sides.

    A missing refinement centre is resolved by :meth:`Grid3D.from_config`
    before this helper is called.  When it equals the domain centre, delegate
    to the historical symmetric implementation so legacy configurations retain
    exactly the same floating-point edge construction.
    """

    size = float(size_m)
    center = float(domain_center_m)
    focus = float(refinement_center_m)
    lower = center - 0.5 * size
    upper = center + 0.5 * size
    if not lower < focus < upper:
        raise ValueError("grid refinement centre must lie strictly inside its domain")
    if focus == center or float(edge_to_focus_ratio) <= 1.0 + 1e-12:
        return center + _center_stretched_edges(
            size, count, edge_to_focus_ratio
        )

    ratio = float(edge_to_focus_ratio)
    if not math.isfinite(ratio) or ratio < 1.0:
        raise ValueError("grid stretch ratios must be at least 1")
    strength = float(np.arccosh(ratio))
    denominator = float(np.sinh(strength))
    computational = np.linspace(0.0, 1.0, count + 1)
    focus_fraction = (focus - lower) / size
    left = computational <= focus_fraction
    edges = np.empty_like(computational)
    edges[left] = focus - (focus - lower) * np.sinh(
        strength
        * (focus_fraction - computational[left])
        / focus_fraction
    ) / denominator
    edges[~left] = focus + (upper - focus) * np.sinh(
        strength
        * (computational[~left] - focus_fraction)
        / (1.0 - focus_fraction)
    ) / denominator
    # Fix the analytic endpoints exactly so downstream containment checks and
    # result archives retain the configured physical domain without roundoff.
    edges[0] = lower
    edges[-1] = upper
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("focused grid mapping produced non-monotonic edges")
    return edges


def optical_grid_from_config(
    physical_grid: Grid3D,
    cfg: dict,
    *,
    section: str = "optical",
) -> Grid3D:
    """Build one optional uniform optical grid sharing the physical z cells."""

    optical = cfg.get(section, {})
    if not bool(optical.get("enabled", False)):
        return physical_grid
    return Grid3D.from_config(
        {
            "size_x_m": float(optical["size_x_m"]),
            "size_y_m": float(optical["size_y_m"]),
            "length_z_m": float(physical_grid.z_edges[-1] - physical_grid.z_edges[0]),
            "nx": int(optical["nx"]),
            "ny": int(optical["ny"]),
            "nz": len(physical_grid.z),
            "center_x_m": float(optical.get("center_x_m", 0.0)),
            "center_y_m": float(optical.get("center_y_m", 0.0)),
            "spacing_xy": "uniform",
        }
    )


def pump_optical_grid_from_config(
    physical_grid: Grid3D,
    cfg: dict,
    shared_optical_grid: Grid3D,
) -> Grid3D:
    """Resolve the Pump grid without silently tying it to the Seed/BPM window.

    ``shared`` preserves the historical single-optical-grid behaviour.  The
    explicit ``physical`` and ``custom`` modes are useful when a narrow,
    finely sampled Seed/BPM window would otherwise crop a broader Pump.
    """

    pump_optical = cfg.get("pump_optical", {})
    mode = str(pump_optical.get("mode", "shared"))
    if mode == "shared":
        return shared_optical_grid
    if mode == "physical":
        return physical_grid
    if mode == "custom":
        return optical_grid_from_config(
            physical_grid,
            {"pump_optical": {**pump_optical, "enabled": True}},
            section="pump_optical",
        )
    raise ValueError("grid.pump_optical.mode must be shared, physical, or custom")
