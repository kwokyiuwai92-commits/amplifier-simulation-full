"""Optional polar quadrature for Commercial App's Gaussian-q pulse and average transport.

The grid shares radial/angle counts with seed.coherent.axisymmetric, but Commercial App
retains all angular collocation samples: a Gaussian ellipse must not lose its
m=2 content merely because a separate coherent calculation retained m=0,+/-1.
No coherent diffraction or coherent time-domain gain is used by this backend.
External optics and material/thermal results retain their Cartesian contract.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager

import numpy as np

from .commercial_app_fast import CommercialAppFastGaussianDuctMixin
from .axisymmetric_average import AxisymmetricAveragePowerSignalPropagator
from .axisymmetric_coherent import AxisymmetricCoherentSignalPropagator
from .spectral_signal import SpectralSignalPropagator
from .spectral_volume import SpectralIntensityVolume


class _PolarGeometry(AxisymmetricCoherentSignalPropagator):
    """Reuse geometry, conservative mapping and boundary audits, never CN."""

    def _validate_axisymmetric_contract(self):
        if not self.grid.is_uniform_xy or not np.isclose(
            np.ptp(self.grid.x_edges), np.ptp(self.grid.y_edges), rtol=1e-10, atol=1e-15
        ):
            raise ValueError("Commercial App axisymmetric requires a uniform square optical window")
        if not 8 <= self.radial_bins <= min(len(self.grid.x), len(self.grid.y)):
            raise ValueError("Commercial App radial_bins must be between 8 and the optical grid dimension")
        if np.any(self.crystal_mask < 1 - 1e-12):
            raise ValueError("Commercial App axisymmetric requires the crystal to cover the optical window")
        if self.cfg.get("polarization", {}).get("model", "scalar") != "scalar":
            raise ValueError("Commercial App axisymmetric requires scalar polarization")

    def _truncate_angular_samples(self, values, *, real_output=False, **kwargs):
        # Collocation, not the coherent model's finite m-mode truncation.
        return np.real(values) if real_output else np.asarray(values)

    def _diffract(self, *args, **kwargs):
        raise RuntimeError("Commercial App polar Gaussian-q projection failed; coherent diffraction fallback is disabled")


class _PolarQuadrature:
    def __init__(self, radial):
        self.x = radial.radial_centers_m
        self.y = radial.angular_coordinate_rad
        self.z, self.dz = radial.grid.z, radial.grid.dz
        self.shape = (len(self.z), len(self.y), len(self.x))
        self.is_uniform_xy = False
        self.cell_areas = np.broadcast_to(radial.angular_cell_area_m2, self.shape[-2:])
        cx = .5 * (radial.grid.x_edges[0] + radial.grid.x_edges[-1])
        cy = .5 * (radial.grid.y_edges[0] + radial.grid.y_edges[-1])
        self.transverse_mesh = (cx + np.cos(self.y[:, None]) * self.x,
                                cy + np.sin(self.y[:, None]) * self.x)

    def integrate_transverse(self, value):
        return np.sum(value * self.cell_areas, axis=(-2, -1))

    def transverse_derivatives(self, value):
        # Chain rule in polar coordinates; periodic theta derivative, no seam.
        dr = np.gradient(value, self.x, axis=-1, edge_order=2)
        orders = np.fft.fftfreq(len(self.y), d=1 / len(self.y))
        dt = np.fft.ifft(np.fft.fft(value, axis=-2) * (1j * orders[:, None]), axis=-2)
        c, s = np.cos(self.y[:, None]), np.sin(self.y[:, None])
        return c * dr - s * dt / self.x, s * dr + c * dt / self.x


class AxisymmetricCommercialAppFastSignalPropagator(
    CommercialAppFastGaussianDuctMixin, AxisymmetricAveragePowerSignalPropagator
):
    """Commercial App operators on polar samples, with positive material bookkeeping."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radial.spectrum = self.spectrum
        self.polar_grid = _PolarQuadrature(self.radial)
        self._polar_population = None
        self._polar_population_initial = None
        self._polar_passive = None
        self._polar_override = None
        self._polar_override_initial = None
        self._polar_mapping_error = 0.0
        self._polar_gain_redistribution = 0.0
        self._reset_commercial_app_diagnostics()

    def _reset_commercial_app_diagnostics(self):
        super()._reset_commercial_app_diagnostics()
        if hasattr(self, "polar_grid"):
            self._commercial_app_cumulative_b_integral_rad = np.zeros(self.polar_grid.shape[-2:])
            self._commercial_app_strong_field_loss_energy_J_m3 = np.zeros(self.polar_grid.shape)

    def _result_extras(self, *args, **kwargs):
        loss = self._commercial_app_strong_field_loss_energy_J_m3
        self._commercial_app_strong_field_loss_energy_J_m3 = self._expand_scalar(loss)
        try:
            return super()._result_extras(*args, **kwargs)
        finally:
            self._commercial_app_strong_field_loss_energy_J_m3 = loss

    def _finish_average_volume(self, values, *, spectral=False):
        if spectral and self._commercial_app_hg_enabled:
            combined = np.zeros((len(self._commercial_app_reported_gain_wavelengths_m), *values.shape[1:]))
            for index, channel in enumerate(self._commercial_app_modal_channels):
                combined[int(channel["spectral_index"])] += values[index]
            values = combined
        return super()._finish_average_volume(values, spectral=spectral)

    def _aggregate_commercial_app_hg_result(self, result):
        volume = result.average_spectral_intensity_W_m2
        if self._commercial_app_hg_enabled and isinstance(volume, SpectralIntensityVolume):
            # Aggregated on the reduced grid above, before bounded expansion.
            result.average_spectral_intensity_W_m2 = None
            try:
                return super()._aggregate_commercial_app_hg_result(result)
            finally:
                result.average_spectral_intensity_W_m2 = volume
        return super()._aggregate_commercial_app_hg_result(result)

    def _make_radial_helper(self, *args, **kwargs):
        values = list(args)
        seed = copy.deepcopy(values[2])
        coherent = seed.setdefault("coherent", {})
        coherent["fidelity_mode"] = "axisymmetric_kerr"
        coherent.setdefault("transient_gain", {})["enabled"] = False
        if seed.get("temporal", {}).get("type") == "cw":
            # Geometry does not require a physical coherent pulse for CW.
            seed["temporal"]["type"] = "gaussian"
            coherent["input_mode"] = "coefficients"
        # The helper's mode labels are not the Commercial App physical approximation.
        coherent.setdefault("axisymmetric", {})["azimuthal_fidelity"] = "m0_m1_m2"
        values[2] = seed
        radial = _PolarGeometry(*values, **kwargs)
        radial._retained_angular_indices = np.arange(radial.angular_samples)
        radial._retained_angular_fft_mask[:] = True
        return radial

    @contextmanager
    def _polar_operators(self):
        grid, mask = self.grid, self.crystal_mask
        self.grid, self.crystal_mask = self.polar_grid, self.radial._axisym_mask
        try:
            yield
        finally:
            self.grid, self.crystal_mask = grid, mask

    def _diffract(self, field, distance_m, refractive_index):
        if not self.radial._is_reduced_field(field):
            return CommercialAppFastGaussianDuctMixin._diffract(self, field, distance_m, refractive_index)
        with self._polar_operators():
            result = CommercialAppFastGaussianDuctMixin._diffract(self, field, distance_m, refractive_index)
        outer = max(1, int(np.ceil(.05 * self.radial.radial_bins)))
        areas = self.polar_grid.cell_areas
        self.radial._update_axisymmetric_boundary_diagnostics(
            np.sum(abs(result)**2 * areas, axis=(-2, -1)),
            np.sum(abs(result[..., -outer:])**2 * areas[..., -outer:], axis=(-2, -1)))
        return result

    def _apply_thermal_phase(self, field, delta_n_slice, step_length_m, z_index):
        if self.radial._is_reduced_field(field):
            delta_n_slice = self.radial._axisym_delta_n[z_index]
            with self._polar_operators():
                if self._commercial_app_thermal_lens_model == "local_parabolic":
                    delta_n_slice = self._commercial_app_local_parabolic_delta_n(delta_n_slice, z_index)
                if self._commercial_app_needs_phase_conversion() and np.any(delta_n_slice):
                    phase = (2.0 * np.pi * float(step_length_m)
                             * np.asarray(delta_n_slice)[None, :, :]
                             / np.asarray(self.spectrum.coordinate)[:, None, None])
                    return self._apply_commercial_app_physical_phase(field, phase)
                return SpectralSignalPropagator._apply_thermal_phase(self, field, delta_n_slice, step_length_m, z_index)
        return CommercialAppFastGaussianDuctMixin._apply_thermal_phase(self, field, delta_n_slice, step_length_m, z_index)

    def _project_average_gain_field(self, field):
        return field, None

    def _enter_gain_transport(self, field, population, delta_n, pass_number):
        field, projected = self._enter_average_transport(field, population, delta_n)
        self._polar_population = projected.copy()
        self._polar_population_initial = projected.copy()
        self._polar_passive = np.zeros(projected.shape[1:])
        self._polar_override = (None if self._override_gain_m_inv is None else
                                self.radial._project_cartesian_scalar_to_angular(self._override_gain_m_inv))
        self._polar_override_initial = (
            None if self._polar_override is None else self._polar_override.copy()
        )
        return field

    def _local_material_step(self, field, population, z_index, step_length_m):
        passive, override = self.passive_loss_energy_J_m3, self._override_gain_m_inv
        self.passive_loss_energy_J_m3, self._override_gain_m_inv = self._polar_passive, self._polar_override
        try:
            with self._polar_operators():
                field, transfer = SpectralSignalPropagator._local_material_step(
                    self, field, self._polar_population, z_index, step_length_m)
            self._polar_override = self._override_gain_m_inv
        finally:
            self.passive_loss_energy_J_m3, self._override_gain_m_inv = passive, override
        return field, self._expand_scalar(transfer)

    def _exit_gain_transport(self, field, population, delta_n, pass_number):
        delta, error = self.radial._map_angular_population_delta(
            population, self._polar_population - self._polar_population_initial)
        population += delta
        self._polar_mapping_error = max(self._polar_mapping_error, error)
        self.passive_loss_energy_J_m3 += self._expand_scalar(self._polar_passive)
        if self._polar_override is not None:
            # Only stimulated depletion changes the Cartesian reservoir.
            # Expanding the absolute polar state would reshape untouched gain
            # and erase it outside the inscribed disk on every pass.
            self._commit_polar_override_depletion()
        return self._exit_average_transport(field)

    def _commit_polar_override_depletion(self):
        self._polar_gain_redistribution = max(
            self._polar_gain_redistribution,
            self._commit_axisymmetric_override_depletion(
                self._polar_override_initial, self._polar_override))

    def _signal_volume_density(self, field):
        return self._expand_scalar(np.sum(np.abs(field)**2, axis=(0, 1)))

    def _longitudinal_spatial_sample(self, field):
        if self.radial._is_reduced_field(field):
            with self._polar_operators():
                return SpectralSignalPropagator._longitudinal_spatial_sample(self, field)
        return super()._longitudinal_spatial_sample(field)

    def _longitudinal_cartesian_field(self, field):
        if self.radial._is_reduced_field(field):
            return self.radial._expand_radial_field(field)
        return field

    def propagate(self, *args, **kwargs):
        self.radial._reset_axisymmetric_diagnostics()
        self._polar_mapping_error = 0.0
        self._polar_gain_redistribution = 0.0
        try:
            result = super().propagate(*args, **kwargs)
            if self._polar_mapping_error > 1e-8:
                result.warnings.append(f"Commercial App polar population mapping residual: {self._polar_mapping_error:.6g}")
            result.gain_transport["override_gain_mapping_redistribution_fraction"] = self._polar_gain_redistribution
            if self._polar_gain_redistribution > float(self.radial.axisymmetric_cfg.get("maximum_population_positivity_redistribution_fraction", .05)):
                result.warnings.append(
                    f"Commercial App polar gain-reservoir remap redistributed {self._polar_gain_redistribution:.6g} of the requested depletion; refine the radial/optical grid.")
            return result
        finally:
            self._restore_average_environment()
            self._polar_population = self._polar_population_initial = None
            self._polar_passive = self._polar_override = None
            self._polar_override_initial = None
            self.radial._axisym_delta_n = None

    def _average_transport_diagnostics(self):
        result = super()._average_transport_diagnostics()
        result.update(azimuthal_fidelity="full_angular_collocation", diffraction_model="gaussian_q")
        return result

    def _transport_progress_metrics(self):
        result = super()._transport_progress_metrics()
        result.update(commercial_app_spatial_backend="axisymmetric", commercial_app_radial_bins=self.radial.radial_bins,
                      commercial_app_angular_samples=self.radial.angular_samples, commercial_app_diffraction_model="gaussian_q")
        return result

    def _commercial_app_gain_transport_diagnostics(self):
        result = super()._commercial_app_gain_transport_diagnostics()
        result.update(spatial_backend="axisymmetric", radial_bins=self.radial.radial_bins,
                      angular_samples=self.radial.angular_samples, angular_representation="full_angular_collocation",
                      population_mapping_relative_error=self._polar_mapping_error,
                      spectral_material_coupling="bounded_blocks")
        result.update(
            entrance_projection_retained_energy_fraction=self.radial.axisym_total_projection_energy_fraction,
            maximum_energy_weighted_boundary_fraction=self.radial.axisym_maximum_energy_weighted_boundary_fraction,
        )
        return result
