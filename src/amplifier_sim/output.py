"""Simulation result writers: JSON, compressed NumPy, VTK, and quick-look PNG."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .json_contract import dumps_json_portable
from .result_fields import standard_result_arrays

if TYPE_CHECKING:
    from .solver import SimulationResult


_OWNED_RESULT_ARTIFACTS = (
    "summary.json",
    "fields.npz",
    "optical_planes.npz",
    "optical_plane_coordinates.npz",
    "fields.vti",
    "fields.vtr",
    "slices.png",
)


def save_result(result: "SimulationResult", output_dir: Path) -> None:
    """Write one self-consistent result generation without stale sidecars.

    All files are produced in a sibling staging directory first.  Only after
    every writer succeeds are the artifacts swapped into the destination; a
    failed swap restores the previous generation.  Unrelated user files in an
    existing output directory are deliberately preserved.
    """

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        _write_result_generation(result, staging)
        _commit_result_generation(staging, output_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _write_result_generation(result: "SimulationResult", output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(
        dumps_json_portable(result.summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    arrays = standard_result_arrays(result)
    arrays.update(
        {name: np.asarray(values) for name, values in result.stress.items()}
    )
    np.savez_compressed(output_dir / "fields.npz", **arrays)
    if result.optical_planes:
        np.savez_compressed(output_dir / "optical_planes.npz", **result.optical_planes)
        coordinates = result.optical_plane_coordinates or {}
        if coordinates:
            coordinate_arrays = {
                f"{name}__{axis}_m": values[index]
                for name, values in coordinates.items()
                for index, axis in enumerate(("x", "y"))
            }
            np.savez_compressed(
                output_dir / "optical_plane_coordinates.npz",
                **coordinate_arrays,
            )
    if result.summary.get("save_vtk", True):
        if result.grid.is_uniform_xy:
            _write_vti(result, output_dir / "fields.vti")
        else:
            _write_vtr(result, output_dir / "fields.vtr")
    _write_quicklook(result, output_dir / "slices.png")


def _commit_result_generation(staging: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    backup = staging / ".previous-generation"
    backup.mkdir()
    moved_old: list[str] = []
    moved_new: list[str] = []
    try:
        for name in _OWNED_RESULT_ARTIFACTS:
            current = output_dir / name
            if not current.exists() and not current.is_symlink():
                continue
            if current.is_dir() and not current.is_symlink():
                raise IsADirectoryError(
                    f"Cannot replace result artifact directory {current}"
                )
            os.replace(current, backup / name)
            moved_old.append(name)
        for name in _OWNED_RESULT_ARTIFACTS:
            candidate = staging / name
            if candidate.exists():
                os.replace(candidate, output_dir / name)
                moved_new.append(name)
    except Exception:
        for name in reversed(moved_new):
            current = output_dir / name
            if current.exists() or current.is_symlink():
                current.unlink()
        for name in reversed(moved_old):
            os.replace(backup / name, output_dir / name)
        raise


def _write_vti(result: "SimulationResult", path: Path) -> None:
    nx, ny, nz = len(result.grid.x), len(result.grid.y), len(result.grid.z)
    upper = result.population_m3[result.level_names.index(result.summary["population"]["upper_level"])]
    density = (
        np.sum(result.population_m3, axis=0)
        if result.dopant_density_m3 is None
        else np.asarray(result.dopant_density_m3, dtype=float)
    )
    upper_fraction = np.divide(
        upper, density, out=np.zeros_like(upper), where=density > 0.0
    )
    fields = {
        "pump_intensity_W_m2": result.pump_intensity_W_m2,
        "signal_fluence_J_m2": result.signal_fluence_J_m2,
        "gain_m_inv": result.gain_m_inv,
        "initial_active_gain_m_inv": result.initial_active_gain_m_inv,
        "remaining_active_gain_m_inv": result.remaining_active_gain_m_inv,
        "passive_loss_m_inv": result.passive_loss_m_inv,
        "net_gain_m_inv": result.net_gain_m_inv,
        "heat_W_m3": result.heat_W_m3,
        "inversion_after_pump_m3": result.inversion_after_pump_m3,
        "extracted_energy_J_m3": result.extracted_energy_J_m3,
        "temperature_K": result.temperature_K,
        "delta_n": result.delta_n,
        "upper_fraction": upper_fraction,
        "dopant_density_m3": density,
        "von_mises_Pa": result.stress["von_mises_Pa"],
        "crystal_mask": np.ones(result.grid.shape) if result.crystal_mask is None else result.crystal_mask.astype(float),
    }
    optional_fields = {
        "nonlinear_loss_energy_J_m3": result.nonlinear_loss_energy_J_m3,
        "signal_passive_loss_energy_J_m3": result.signal_passive_loss_energy_J_m3,
        "ase_intensity_W_m2": result.ase_intensity_W_m2,
    }
    fields.update({name: value for name, value in optional_fields.items() if value is not None})
    fields.update(
        {
            name: np.asarray(value)
            for name, value in result.stress.items()
            if name not in fields and np.asarray(value).shape == result.grid.shape
        }
    )
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">',
        f'  <ImageData WholeExtent="0 {nx} 0 {ny} 0 {nz}" Origin="{result.grid.x_edges[0]} {result.grid.y_edges[0]} {result.grid.z_edges[0]}" Spacing="{result.grid.dx} {result.grid.dy} {result.grid.dz}">',
        f'    <Piece Extent="0 {nx} 0 {ny} 0 {nz}">',
        '      <PointData/>',
        '      <CellData Scalars="gain_m_inv">',
    ]
    for name, values in fields.items():
        payload = " ".join(f"{value:.9g}" for value in np.asarray(values).ravel(order="C"))
        lines.append(f'        <DataArray type="Float64" Name="{name}" format="ascii">{payload}</DataArray>')
    lines.extend(["      </CellData>", "    </Piece>", "  </ImageData>", "</VTKFile>"])
    path.write_text("\n".join(lines), encoding="ascii")


def _write_vtr(result: "SimulationResult", path: Path) -> None:
    """Write finite-volume cell fields on their true rectilinear-grid edges."""

    nx, ny, nz = len(result.grid.x), len(result.grid.y), len(result.grid.z)
    upper = result.population_m3[result.level_names.index(result.summary["population"]["upper_level"])]
    density = (
        np.sum(result.population_m3, axis=0)
        if result.dopant_density_m3 is None
        else np.asarray(result.dopant_density_m3, dtype=float)
    )
    upper_fraction = np.divide(
        upper, density, out=np.zeros_like(upper), where=density > 0.0
    )
    fields = {
        "pump_intensity_W_m2": result.pump_intensity_W_m2,
        "signal_fluence_J_m2": result.signal_fluence_J_m2,
        "gain_m_inv": result.gain_m_inv,
        "initial_active_gain_m_inv": result.initial_active_gain_m_inv,
        "remaining_active_gain_m_inv": result.remaining_active_gain_m_inv,
        "passive_loss_m_inv": result.passive_loss_m_inv,
        "net_gain_m_inv": result.net_gain_m_inv,
        "heat_W_m3": result.heat_W_m3,
        "inversion_after_pump_m3": result.inversion_after_pump_m3,
        "extracted_energy_J_m3": result.extracted_energy_J_m3,
        "temperature_K": result.temperature_K,
        "delta_n": result.delta_n,
        "upper_fraction": upper_fraction,
        "dopant_density_m3": density,
        "von_mises_Pa": result.stress["von_mises_Pa"],
        "crystal_mask": np.ones(result.grid.shape) if result.crystal_mask is None else result.crystal_mask.astype(float),
    }
    optional_fields = {
        "nonlinear_loss_energy_J_m3": result.nonlinear_loss_energy_J_m3,
        "signal_passive_loss_energy_J_m3": result.signal_passive_loss_energy_J_m3,
        "ase_intensity_W_m2": result.ase_intensity_W_m2,
    }
    fields.update({name: value for name, value in optional_fields.items() if value is not None})
    fields.update(
        {
            name: np.asarray(value)
            for name, value in result.stress.items()
            if name not in fields and np.asarray(value).shape == result.grid.shape
        }
    )
    extent = f"0 {nx} 0 {ny} 0 {nz}"
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">',
        f'  <RectilinearGrid WholeExtent="{extent}">',
        f'    <Piece Extent="{extent}">',
        '      <PointData/>',
        '      <CellData Scalars="gain_m_inv">',
    ]
    for name, values in fields.items():
        payload = " ".join(f"{value:.9g}" for value in np.asarray(values).ravel(order="C"))
        lines.append(f'        <DataArray type="Float64" Name="{name}" format="ascii">{payload}</DataArray>')
    lines.extend(['      </CellData>', '      <Coordinates>'])
    for name, values in (
        ("X", result.grid.x_edges),
        ("Y", result.grid.y_edges),
        ("Z", result.grid.z_edges),
    ):
        payload = " ".join(f"{value:.9g}" for value in values)
        lines.append(f'        <DataArray type="Float64" Name="{name}" format="ascii">{payload}</DataArray>')
    lines.extend(['      </Coordinates>', '    </Piece>', '  </RectilinearGrid>', '</VTKFile>'])
    path.write_text("\n".join(lines), encoding="ascii")


def _write_quicklook(result: "SimulationResult", path: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    iz = len(result.grid.z) // 2
    panels = [
        (np.max(result.pump_intensity_W_m2, axis=0), "max pump intensity [W/m2]"),
        (np.max(result.signal_fluence_J_m2, axis=0), "max last-pass signal fluence [J/m2]"),
        (result.gain_m_inv[iz], "mid-plane gain [1/m]"),
        (
            result.temperature_K[iz] - float(result.summary["thermal"]["reference_temperature_K"]),
            "mid-plane T - reference [K]",
        ),
    ]
    width, height = 420, 330
    canvas = Image.new("RGB", (width * 2, height * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (array, title) in enumerate(panels):
        rgb, low, high = _heatmap(array)
        image = Image.fromarray(rgb, mode="RGB").resize((width - 24, height - 64), Image.Resampling.NEAREST)
        ox, oy = (index % 2) * width, (index // 2) * height
        canvas.paste(image, (ox + 12, oy + 34))
        draw.text((ox + 12, oy + 10), title, fill="black")
        draw.text((ox + 12, oy + height - 24), f"min={low:.4g}  max={high:.4g}", fill="black")
    canvas.save(path)


def _heatmap(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if not np.any(finite):
        low = high = math.nan
        scaled = np.zeros_like(array)
    else:
        low = float(np.min(array[finite]))
        high = float(np.max(array[finite]))
        scaled = (
            np.zeros_like(array)
            if high <= low
            else np.where(
                finite,
                np.clip((array - low) / (high - low), 0.0, 1.0),
                0.0,
            )
        )
    anchors = np.array([[8, 8, 35], [30, 70, 190], [20, 190, 210], [245, 220, 70], [255, 250, 240]], dtype=float)
    position = scaled * (len(anchors) - 1)
    index = np.minimum(position.astype(int), len(anchors) - 2)
    fraction = position - index
    rgb = anchors[index] * (1.0 - fraction[..., None]) + anchors[index + 1] * fraction[..., None]
    return rgb.astype(np.uint8), low, high
