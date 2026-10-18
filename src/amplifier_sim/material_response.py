"""Data-driven local spectroscopy feedback.

The base material record remains the source of the wavelength-dependent cross
sections and lifetimes.  This module only applies explicitly configured
temperature/concentration corrections.  Every correction defaults to ``off``;
there are no hidden Cr:ZnS constants in the implementation.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import pickle
from typing import Any

import numpy as np

from .material_range_guard import MaterialRangeGuard, interpolate_clamped


_CROSS_SECTION_KEYS = {
    "pump_absorption",
    "pump_emission",
    "signal_absorption",
    "signal_emission",
}

# k_B/(h c), expressed in inverse centimetres per kelvin.  Stark energies in
# laser-material databases are conventionally tabulated in cm^-1.
_BOLTZMANN_WAVENUMBER_CM_INV_K = 0.6950348004861274


def _scale_law(
    law: dict[str, Any] | None,
    coordinate: np.ndarray,
    reference: float,
    *,
    guard: MaterialRangeGuard | None = None,
    name: str = "scale_table",
) -> np.ndarray:
    cfg = law or {}
    model = str(cfg.get("model", "off"))
    values = np.asarray(coordinate, dtype=float)
    if model == "off":
        return np.ones_like(values)
    if model == "linear_scale":
        scale = 1.0 + float(cfg.get("slope_per_unit", 0.0)) * (values - reference)
    elif model == "scale_table":
        points = np.asarray(cfg.get("coordinate", []), dtype=float)
        scales = np.asarray(cfg.get("scale", []), dtype=float)
        if len(points) < 2 or points.shape != scales.shape:
            raise ValueError("scale_table requires equal coordinate/scale arrays with at least two points")
        scale = interpolate_clamped(
            values,
            points,
            scales,
            guard=guard,
            name=name,
        )
    elif model == "boltzmann_stark_fraction":
        energies = np.asarray(cfg.get("energy_cm_inv", []), dtype=float)
        degeneracies = np.asarray(cfg.get("degeneracy", []), dtype=float)
        selected = int(cfg.get("selected_level_index", -1))
        if (
            len(energies) < 1
            or energies.shape != degeneracies.shape
            or selected < 0
            or selected >= len(energies)
            or np.any(degeneracies <= 0.0)
        ):
            raise ValueError(
                "boltzmann_stark_fraction requires equal energy/degeneracy arrays "
                "and a valid selected_level_index"
            )

        def occupation(temperature_K: np.ndarray) -> np.ndarray:
            temperature = np.maximum(
                np.asarray(temperature_K, dtype=float), np.finfo(float).tiny
            )
            relative_energy = energies - float(np.min(energies))
            exponent = -relative_energy.reshape(
                (len(relative_energy),) + (1,) * temperature.ndim
            ) / (_BOLTZMANN_WAVENUMBER_CM_INV_K * temperature)
            weights = degeneracies.reshape(
                (len(degeneracies),) + (1,) * temperature.ndim
            ) * np.exp(exponent)
            return weights[selected] / np.sum(weights, axis=0)

        reference_temperature = float(cfg.get("reference_temperature_K", reference))
        reference_occupation = float(occupation(np.asarray(reference_temperature)))
        if reference_occupation <= 0.0:
            raise ValueError("boltzmann_stark_fraction has zero reference occupation")
        scale = occupation(values) / reference_occupation
    elif model == "stern_volmer":
        quenching = float(cfg.get("quenching_coordinate", 0.0))
        order = float(cfg.get("order", 1.0))
        if quenching <= 0.0 or order <= 0.0:
            raise ValueError("stern_volmer requires positive quenching_coordinate and order")
        # The configured base lifetime belongs to the reference material
        # record, so normalise the law to unity at that reference density.
        scale = (1.0 + (reference / quenching) ** order) / (
            1.0 + (np.maximum(values, 0.0) / quenching) ** order
        )
    else:
        raise ValueError(f"unsupported response law {model!r}")
    minimum = float(cfg.get("minimum_scale", 0.0))
    maximum = float(cfg.get("maximum_scale", np.inf))
    return np.clip(scale, minimum, maximum)


@dataclass
class SpectroscopicResponse:
    """Evaluate optional voxel-local spectroscopy corrections."""

    cfg: dict[str, Any]
    reference_temperature_K: float
    reference_density_m3: float
    temperature_K: np.ndarray | float
    density_m3: np.ndarray | float
    range_guard: MaterialRangeGuard = field(default_factory=MaterialRangeGuard)
    # Consumers may cache wavelength-by-space response tensors, but only while
    # the accepted thermal/concentration environment is unchanged.  A monotonic
    # revision is safer than inferring that state from array identity and keeps
    # future transient-thermal updates from reusing stale spectroscopy.
    environment_revision: int = field(default=0, init=False, repr=False)

    _scale_cache: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    _scale_cache_bytes: int = field(default=0, init=False, repr=False)
    _scale_cache_max_bytes: int = field(default=32 * 1024**2, init=False, repr=False)

    @classmethod
    def from_config(
        cls,
        material: dict[str, Any] | None,
        levels: dict[str, Any],
    ) -> "SpectroscopicResponse":
        material_cfg = material or {}
        cfg = dict(material_cfg.get("spectroscopic_response", {}))
        reference_temperature = float(
            cfg.get(
                "reference_temperature_K",
                material_cfg.get("material_data_reference_temperature_K", 293.15),
            )
        )
        reference_density = float(
            cfg.get("reference_density_m3", material_cfg.get("ion_density_m3", 1.0))
        )
        operating_temperature = float(
            material_cfg.get("operating_temperature_K", reference_temperature)
        )
        return cls(
            cfg=cfg,
            reference_temperature_K=reference_temperature,
            reference_density_m3=max(reference_density, np.finfo(float).tiny),
            temperature_K=operating_temperature,
            density_m3=reference_density,
            range_guard=MaterialRangeGuard(
                policy=str(cfg.get("range_policy", "record"))
            ),
        )

    @property
    def enabled(self) -> bool:
        return str(self.cfg.get("mode", "off")) != "off"

    @property
    def mode(self) -> str:
        return str(self.cfg.get("mode", "off"))

    def set_environment(
        self,
        temperature_K: np.ndarray | float,
        density_m3: np.ndarray | float,
    ) -> None:
        temperature = np.asarray(temperature_K, dtype=float)
        density = np.asarray(density_m3, dtype=float)
        if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise ValueError("spectroscopic-response temperature must be positive and finite")
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("spectroscopic-response density must be finite and non-negative")
        self.temperature_K = temperature.copy() if temperature.ndim else float(temperature)
        self.density_m3 = density.copy() if density.ndim else float(density)
        # Environment updates, including optical-grid mapping/restoration, are
        # the invalidation boundary. Never keep scales from a previous field.
        if isinstance(self.temperature_K, np.ndarray):
            self.temperature_K.setflags(write=False)
        if isinstance(self.density_m3, np.ndarray):
            self.density_m3.setflags(write=False)
        self.environment_revision += 1
        self._scale_cache.clear()
        self._scale_cache_bytes = 0

    def environment_copy(self) -> tuple[np.ndarray | float, np.ndarray | float]:
        def copied(value: np.ndarray | float) -> np.ndarray | float:
            array = np.asarray(value)
            return array.copy() if array.ndim else float(array)

        return copied(self.temperature_K), copied(self.density_m3)

    def _environment(
        self,
        spatial_shape: tuple[int, ...],
        z_index: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        def select(value: np.ndarray | float) -> np.ndarray:
            array = np.asarray(value, dtype=float)
            if z_index is not None and array.ndim == len(spatial_shape) + 1:
                array = array[z_index]
            return np.broadcast_to(array, spatial_shape)

        return select(self.temperature_K), select(self.density_m3)

    def _environment_scale(
        self, law: dict[str, Any] | None, coordinate: np.ndarray,
        reference: float, *, name: str, z_index: int | None,
    ) -> np.ndarray:
        """Reuse a local material scale, never a population-dependent gain.

        Bound retained arrays; config values are part of the key, so changing
        a law in place cannot retain its old result. Range evidence is replayed
        on hits, including after a guard reset or a change to error policy.
        """
        cfg = law or {}
        key = (self.environment_revision, name, z_index, coordinate.shape,
               float(reference), pickle.dumps(cfg, protocol=5))
        cached = self._scale_cache.get(key)
        if cached is not None:
            self._scale_cache.move_to_end(key)
            if str(cfg.get("model", "off")) == "scale_table":
                points = np.asarray(cfg["coordinate"], dtype=float)
                self.range_guard.record(name, coordinate, float(np.min(points)), float(np.max(points)))
            return cached
        result = _scale_law(cfg, coordinate, reference, guard=self.range_guard, name=name)
        result.setflags(write=False)
        if result.nbytes <= self._scale_cache_max_bytes:
            self._scale_cache[key] = result
            self._scale_cache_bytes += result.nbytes
            while (len(self._scale_cache) > 512
                   or self._scale_cache_bytes > self._scale_cache_max_bytes):
                _, evicted = self._scale_cache.popitem(last=False)
                self._scale_cache_bytes -= evicted.nbytes
        return result

    def lifetime_scale(
        self,
        spatial_shape: tuple[int, ...],
        z_index: int | None = None,
    ) -> np.ndarray:
        if not self.enabled:
            return np.ones(spatial_shape, dtype=float)
        temperature, density = self._environment(spatial_shape, z_index)
        lifetime = self.cfg.get("upper_state_lifetime", {})
        return self._environment_scale(
            lifetime.get("temperature"),
            temperature,
            self.reference_temperature_K,
            z_index=z_index,
            name="upper_state_lifetime.temperature",
        ) * self._environment_scale(
            lifetime.get("concentration"),
            density,
            self.reference_density_m3,
            z_index=z_index,
            name="upper_state_lifetime.concentration",
        )

    def cross_sections(
        self,
        key: str,
        wavelengths_m: np.ndarray,
        base_values_m2: np.ndarray,
        spatial_shape: tuple[int, ...],
        z_index: int | None = None,
    ) -> np.ndarray:
        if key not in _CROSS_SECTION_KEYS:
            raise ValueError(f"unsupported cross-section response key {key!r}")
        wavelengths = np.atleast_1d(np.asarray(wavelengths_m, dtype=float))
        base = np.atleast_1d(np.asarray(base_values_m2, dtype=float))
        if wavelengths.shape != base.shape:
            raise ValueError("base cross sections must match wavelength samples")
        spectral_shape = (len(wavelengths),) + (1,) * len(spatial_shape)
        if not self.enabled:
            return np.broadcast_to(base.reshape(spectral_shape), (len(base), *spatial_shape))
        temperature, density = self._environment(spatial_shape, z_index)
        law = self.cfg.get("cross_sections", {}).get(key, {})
        model = str(law.get("model", "off"))
        concentration_law = law.get("concentration") or {}
        if model == "off" and str(concentration_law.get("model", "off")) == "off":
            return np.broadcast_to(
                np.maximum(base, 0.0).reshape(spectral_shape), (len(base), *spatial_shape)
            )
        if model == "absolute_temperature_wavelength_table":
            table_wavelengths = np.asarray(law.get("wavelength_m", []), dtype=float)
            table_temperatures = np.asarray(law.get("temperature_K", []), dtype=float)
            values = np.asarray(law.get("values_m2", []), dtype=float)
            if (
                len(table_wavelengths) < 2
                or len(table_temperatures) < 2
                or values.shape != (len(table_temperatures), len(table_wavelengths))
            ):
                raise ValueError(
                    "absolute_temperature_wavelength_table requires values_m2 with shape "
                    "(Ntemperature, Nwavelength)"
                )
            wavelength_interpolated = np.stack(
                [
                    interpolate_clamped(
                        wavelengths,
                        table_wavelengths,
                        row,
                        guard=(self.range_guard if index == 0 else None),
                        name=f"cross_sections.{key}.wavelength",
                    )
                    for index, row in enumerate(values)
                ],
                axis=0,
            )
            flattened_temperature = temperature.reshape(-1)
            result = np.empty((len(wavelengths), flattened_temperature.size), dtype=float)
            for index in range(len(wavelengths)):
                result[index] = interpolate_clamped(
                    flattened_temperature,
                    table_temperatures,
                    wavelength_interpolated[:, index],
                    guard=(self.range_guard if index == 0 else None),
                    name=f"cross_sections.{key}.temperature",
                )
            result = result.reshape(len(wavelengths), *spatial_shape)
            concentration_law = law.get("concentration")
            if concentration_law:
                result *= self._environment_scale(
                    concentration_law,
                    density,
                    self.reference_density_m3,
                    z_index=z_index,
                    name=f"cross_sections.{key}.concentration",
                )[None, ...]
            return np.maximum(result, 0.0)
        scale = self._environment_scale(
            law,
            temperature,
            self.reference_temperature_K,
            z_index=z_index,
            name=f"cross_sections.{key}.temperature",
        )
        concentration_law = law.get("concentration")
        if concentration_law:
            scale = scale * self._environment_scale(
                concentration_law,
                density,
                self.reference_density_m3,
                z_index=z_index,
                name=f"cross_sections.{key}.concentration",
            )
        return np.maximum(base.reshape(spectral_shape) * scale[None, ...], 0.0)

    def summary(self) -> dict[str, Any]:
        temperature = np.asarray(self.temperature_K, dtype=float)
        density = np.asarray(self.density_m3, dtype=float)
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "reference_temperature_K": self.reference_temperature_K,
            "reference_density_m3": self.reference_density_m3,
            "temperature_min_K": float(np.min(temperature)),
            "temperature_max_K": float(np.max(temperature)),
            "density_min_m3": float(np.min(density)),
            "density_max_m3": float(np.max(density)),
            "cross_section_models": {
                key: str(value.get("model", "off"))
                for key, value in self.cfg.get("cross_sections", {}).items()
            },
            "upper_state_lifetime": dict(self.cfg.get("upper_state_lifetime", {})),
            "material_range": self.range_guard.summary(),
        }
