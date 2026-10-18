"""Cycle-average linear gain on the selected radial/azimuthal spatial basis."""
from __future__ import annotations

import numpy as np
try:
    from scipy import sparse
except ImportError:  # Preserve the NumPy-only simulator installation.
    sparse = None

from .average_signal import AveragePowerSignalPropagator
from .axisymmetric_coherent import AxisymmetricCoherentSignalPropagator
from .seed_gain_transport import spatial_gain_transport_model
from .spectral_volume import SpectralIntensityVolume


class AxisymmetricAveragePowerSignalPropagator(AveragePowerSignalPropagator):
    """Reuse the pulse solver's spatial operators, without its pulse physics.

    External optics remain Cartesian, as in the coherent solver. Inside the
    crystal, fields, gain and spectral intensity use (theta, radius). Spectral
    volumes are expanded only in bounded blocks for the material rate equations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if spatial_gain_transport_model(self.cfg) != "local_field":
            raise ValueError("axisymmetric cycle-average transport requires local_field spatial gain")
        self.radial = self._make_radial_helper(*args, **kwargs)
        # Average power uses the original spectral intensity samples, not the
        # coherent FFT frequency grid. No spectral resampling is introduced.
        self.radial.spectrum = self.spectrum
        # Average/Spectral Time channels already use their own energy profiles.
        # Never apply the coherent FFT-frequency selection to those channels.
        self.radial._tail_frequency_plan = None
        self.radial._diffraction_spectral_state = None
        self._average_environment = None
        self._scalar_expansion = self._build_scalar_expansion()
        self._thermal_projection_loss_W = 0.0
        self._thermal_projection_input_W = 0.0

    def _make_radial_helper(self, *args, **kwargs):
        return AxisymmetricCoherentSignalPropagator(*args, **kwargs)

    def _build_scalar_expansion(self):
        """Positive, linear, area-conservative polar-to-Cartesian mapping.

        Normalize each polar cell's interpolation weights. Empty angular cells
        near the origin share their radial ring; no negative intensity is added
        to enforce conservation.
        """
        radial = self.radial
        pixels = np.flatnonzero(radial._axisym_valid_flat)
        rings = radial._axisym_bin_flat[radial._axisym_valid_flat]
        lower = radial._cartesian_angular_lower
        upper = radial._cartesian_angular_upper
        weight = radial._cartesian_angular_upper_weight
        rows = np.concatenate([pixels, pixels])
        columns = np.concatenate([lower * radial.radial_bins + rings, upper * radial.radial_bins + rings])
        data = np.concatenate([1.0 - weight, weight])
        shape = (self.crystal_mask.size, radial.angular_samples * radial.radial_bins)
        areas = self.grid.cell_areas.ravel()
        actual = np.bincount(columns, weights=data * areas[rows], minlength=shape[1])
        empty = np.flatnonzero(actual == 0)
        if len(empty):
            extra_rows, extra_columns = [], []
            for column in empty:
                ring_pixels = pixels[rings == column % radial.radial_bins]
                extra_rows.extend(ring_pixels)
                extra_columns.extend([column] * len(ring_pixels))
            rows = np.concatenate([rows, np.asarray(extra_rows, dtype=int)])
            columns = np.concatenate([columns, np.asarray(extra_columns, dtype=int)])
            data = np.concatenate([data, np.ones(len(extra_rows))])
            actual = np.bincount(columns, weights=data * areas[rows], minlength=shape[1])
        if np.any(actual <= 0):
            raise ValueError("axisymmetric cycle-average radial rings must overlap Cartesian cells")
        target = np.broadcast_to(radial.angular_cell_area_m2, (radial.angular_samples, radial.radial_bins)).ravel()
        data *= (target / actual)[columns]
        self._scalar_rows, self._scalar_columns, self._scalar_weights = rows, columns, data
        return sparse.csr_matrix((data, (rows, columns)), shape=shape) if sparse is not None else None

    def _average_volume_shape(self):
        return (len(self.grid.z), self.radial.angular_samples, self.radial.radial_bins)

    def _average_transport_mask(self):
        return self.radial._axisym_mask

    def _restore_average_environment(self):
        if self._average_environment is not None:
            self.levels.set_environment(*self._average_environment)
            self._average_environment = None

    def propagate_average(self, *args, **kwargs):
        radial = self.radial
        radial._reset_axisymmetric_diagnostics()
        radial._reset_thermal_phase_runtime()
        self._thermal_projection_loss_W = 0.0
        self._thermal_projection_input_W = 0.0
        try:
            result = super().propagate_average(*args, **kwargs)
            if self._thermal_projection_loss_W:
                rate = float(result.gain_transport["repetition_rate_Hz"])
                result.external_loss_ledger.append({
                    "name": "cycle_average_thermal_azimuthal_projection",
                    "category": "numerical_projection",
                    "input_energy_J": self._thermal_projection_input_W / rate,
                    "output_energy_J": (self._thermal_projection_input_W - self._thermal_projection_loss_W) / rate,
                    "input_average_power_W": self._thermal_projection_input_W,
                    "output_average_power_W": self._thermal_projection_input_W - self._thermal_projection_loss_W,
                    "signed_loss_energy_J": self._thermal_projection_loss_W / rate,
                    "included_in_seed_balance": True,
                })
            checks = (
                (1.0 - radial.axisym_total_projection_energy_fraction, "maximum_projection_loss_fraction", 1e-3, "entrance field projection"),
                (radial.axisym_maximum_dynamic_field_omitted_energy_fraction, "maximum_projection_loss_fraction", 1e-3, "dynamic field projection"),
                (radial.axisym_maximum_energy_weighted_boundary_fraction, "maximum_boundary_energy_fraction", 1e-4, "radial boundary energy"),
                (max(radial.axisym_maximum_population_azimuthal_residual, radial.axisym_maximum_delta_n_azimuthal_residual), "maximum_azimuthal_residual", 0.05, "population/thermal azimuthal residual"),
                (radial.axisym_maximum_population_realizability_correction_fraction, "maximum_population_realizability_correction_fraction", 0.05, "population realizability correction"),
            )
            for value, key, default, label in checks:
                limit = float(radial.axisymmetric_cfg.get(key, default))
                if value > limit:
                    result.warnings.append(f"Cycle-average {label} ({value:.6g}) exceeds the configured axisymmetric limit ({limit:.6g}).")
            return result
        finally:
            self._restore_average_environment()
            self.radial._axisym_delta_n = None

    def _enter_average_transport(self, field, population, delta_n):
        radial = self.radial
        radial._wavelength_source_configs = self._wavelength_source_configs
        projected = radial._project_cartesian_scalar_to_angular(population)
        radial.axisym_maximum_population_azimuthal_residual = max(
            radial.axisym_maximum_population_azimuthal_residual,
            radial._azimuthal_residual(population, projected))
        if radial.maximum_azimuthal_order:
            projected = radial._make_angular_population_realizable(projected)
        thermal = delta_n
        if self._local_parabolic_thermal_lens is not None:
            thermal = np.stack([self._local_parabolic_thermal_lens.fit_slice(delta_n[iz], iz) for iz in range(len(self.grid.z))])
        radial._axisym_delta_n = radial._radial_average(thermal) if not radial.maximum_azimuthal_order else radial._project_cartesian_scalar_to_angular(thermal)
        radial.axisym_maximum_delta_n_azimuthal_residual = max(
            radial.axisym_maximum_delta_n_azimuthal_residual,
            radial._azimuthal_residual(thermal, radial._axisym_delta_n))
        self._average_environment = self.levels.environment_copy()
        def project_environment(value):
            array = np.asarray(value)
            if array.ndim == 0:
                return float(array)
            if array.shape[-2:] != self.crystal_mask.shape:
                raise ValueError("cycle-average spectroscopy environment must match the optical grid")
            return radial._project_cartesian_scalar_to_angular(array)
        if self.levels.response.enabled:
            self.levels.set_environment(*(project_environment(value) for value in self._average_environment))
        output = np.empty((*field.shape[:-2], radial.angular_samples, radial.radial_bins), dtype=field.dtype)
        previous_projection = radial.axisym_total_projection_energy_fraction
        previous_boundary = radial.axisym_maximum_energy_weighted_boundary_fraction
        powers, fractions = [], []
        for index in range(field.shape[1]):
            channel = field[:, index:index + 1]
            powers.append(super()._integral(channel))
            radial.axisym_total_projection_energy_fraction = 1.0
            output[:, index:index + 1] = radial._compress_cartesian_field(channel)
            fractions.append(radial.axisym_total_projection_energy_fraction)
        radial.axisym_total_projection_energy_fraction = min(
            previous_projection,
            float(np.dot(powers, fractions)) / max(sum(powers), np.finfo(float).tiny))
        # Per-channel compression must not turn an insignificant spectral tail
        # into the total-energy boundary diagnostic.
        # Discard per-channel ratios from this entrance, retaining every
        # earlier pass's energy-weighted boundary evidence.
        radial.axisym_maximum_energy_weighted_boundary_fraction = previous_boundary
        outer_count = max(1, int(np.ceil(0.05 * radial.radial_bins)))
        radial._update_axisymmetric_boundary_diagnostics(
            np.sum(np.abs(output)**2 * radial.angular_cell_area_m2, axis=(-2, -1)),
            np.sum(np.abs(output[..., -outer_count:])**2 * radial.angular_cell_area_m2[..., -outer_count:], axis=(-2, -1)),
        )
        radial.axisym_passes_reduced += 1
        return output, projected

    def _exit_average_transport(self, field):
        self._restore_average_environment()
        output = np.empty((*field.shape[:-2], *self.crystal_mask.shape), dtype=field.dtype)
        for index in range(field.shape[1]):
            output[:, index:index + 1] = self.radial._expand_radial_field(field[:, index:index + 1])
        return output

    def _integral(self, field):
        return self.radial._integral(field) if self.radial._is_reduced_field(field) else super()._integral(field)

    def _apply_crystal_aperture(self, field):
        if self.radial._is_reduced_field(field):
            return field
        return super()._apply_crystal_aperture(field)

    def _diffract(self, field, distance_m, refractive_index):
        if self.radial._is_reduced_field(field):
            return self.radial._diffract(field, distance_m, refractive_index)
        return super()._diffract(field, distance_m, refractive_index)

    def _apply_thermal_phase(self, field, delta_n_slice, step_length_m, z_index):
        before = self._integral(field)
        field = self.radial._apply_thermal_phase(field, delta_n_slice, step_length_m, z_index)
        self._thermal_projection_input_W += before
        self._thermal_projection_loss_W += max(0.0, before - self._integral(field))
        self._thermal_phase_stats = self.radial._thermal_phase_stats
        return field

    def _project_average_gain_field(self, field):
        if not self.radial.maximum_azimuthal_order:
            return field, None
        before = self._integral(field)
        field = self.radial._truncate_angular_samples(field, track_field_omission=True)
        return field, (before, self._integral(field))

    def _expand_scalar(self, values):
        shape = values.shape[:-2]
        rows = values.reshape(-1, self.radial.angular_samples * self.radial.radial_bins)
        if self._scalar_expansion is not None:
            result = (self._scalar_expansion @ rows.T).T
        else:
            result = np.empty((len(rows), self.crystal_mask.size), dtype=values.dtype)
            for index, row in enumerate(rows):
                result[index] = np.bincount(
                    self._scalar_rows,
                    weights=self._scalar_weights * row[self._scalar_columns],
                    minlength=self.crystal_mask.size,
                )
        return result.reshape(*shape, *self.crystal_mask.shape)

    def _commit_axisymmetric_override_depletion(self, initial, final):
        """Commit the synthetic reservoir without creating gain or absorbers.

        Conservative polar interpolation can concentrate a nearly complete
        depletion onto too few Cartesian pixels. Redistribute that excess
        over the remaining reservoir in the same z slice and record the
        spatial correction so a coarse remap remains visible.
        """
        sign = 1.0 if float(self.levels_cfg["gain_override_m_inv"]) >= 0 else -1.0
        requested = sign * self._expand_scalar(
            initial - final)
        available = sign * self._override_gain_m_inv
        removed = np.minimum(requested, available)
        integrate = self.grid.integrate_transverse
        excess = integrate(requested - removed)
        remaining = (available - removed) * self.radial._axisym_valid_xy
        capacity = integrate(remaining)
        tolerance = 128 * np.finfo(float).eps * np.maximum(integrate(available), np.finfo(float).tiny)
        if np.any(excess > capacity + tolerance):
            raise ValueError("Axisymmetric gain depletion exceeds the Cartesian reservoir; refine the radial/optical grid")
        fraction = np.divide(excess, capacity, out=np.zeros_like(excess), where=capacity > 0)
        removed += remaining * np.minimum(fraction, 1.0)[:, None, None]
        correction = integrate(np.abs(removed - requested))
        total = integrate(requested)
        self._override_gain_m_inv[:] = sign * np.maximum(available - removed, 0.0)
        return float(np.max(np.divide(correction, total, out=np.zeros_like(total), where=total > 0)))

    def _finish_average_volume(self, values, *, spectral=False):
        if spectral:
            return SpectralIntensityVolume.reduced(values, self.grid.shape, self._expand_scalar)
        return self._expand_scalar(values)

    def _average_transport_diagnostics(self):
        radial = self.radial
        shape = (len(self.spectrum.coordinate), *self._average_volume_shape())
        return {
            "spatial_backend": "axisymmetric",
            "azimuthal_fidelity": radial.azimuthal_fidelity,
            "radial_bins": radial.radial_bins,
            "angular_samples": radial.angular_samples,
            "spectral_storage_shape": list(shape),
            "spectral_storage_bytes": int(np.prod(shape)) * 8,
            "spectral_material_coupling": "bounded_blocks",
            "entrance_projection_retained_energy_fraction": radial.axisym_total_projection_energy_fraction,
            "maximum_energy_weighted_boundary_fraction": radial.axisym_maximum_energy_weighted_boundary_fraction,
            "maximum_population_realizability_correction_fraction": radial.axisym_maximum_population_realizability_correction_fraction,
            "maximum_population_azimuthal_residual": radial.axisym_maximum_population_azimuthal_residual,
            "maximum_delta_n_azimuthal_residual": radial.axisym_maximum_delta_n_azimuthal_residual,
            "maximum_dynamic_field_omitted_energy_fraction": radial.axisym_maximum_dynamic_field_omitted_energy_fraction,
            "thermal_projection_loss_W": self._thermal_projection_loss_W,
        }
