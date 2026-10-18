"""Phase-insensitive spectral/time packets on the shared radial spatial basis.

Frequency channels retain their own spatial wavefront, but never interfere
across frequency or exchange photons through Kerr/SPM. The existing radial
Crank-Nicolson operator is shared with cycle-average and coherent transport.
"""
from __future__ import annotations

import copy
import numpy as np

from .axisymmetric_average import AxisymmetricAveragePowerSignalPropagator
from .axisymmetric_coherent import AxisymmetricCoherentSignalPropagator
from .spectral_signal import SpectralSignalPropagator


class AxisymmetricSpectralSignalPropagator(AxisymmetricAveragePowerSignalPropagator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reduced_population = self._reduced_population_initial = None
        self._reduced_passive = self._reduced_override = None
        self._reduced_override_initial = None
        self._population_mapping_error = 0.0
        self._gain_mapping_redistribution = 0.0
        self._gain_projection_input_J = self._gain_projection_loss_J = 0.0

    def _make_radial_helper(self, *args, **kwargs):
        values = list(args)
        seed = copy.deepcopy(values[2])
        seed["coherent"]["fidelity_mode"] = "axisymmetric_kerr"
        seed["coherent"]["transient_gain"]["enabled"] = False
        if seed.get("temporal", {}).get("type") == "cw":
            seed["temporal"]["type"] = "gaussian"
            seed["coherent"]["input_mode"] = "coefficients"
        values[2] = seed
        return AxisymmetricCoherentSignalPropagator(*values, **kwargs)

    def _enter_gain_transport(self, field, population, delta_n, pass_number):
        field, projected = self._enter_average_transport(field, population, delta_n)
        self._reduced_population = projected.copy()
        self._reduced_population_initial = projected.copy()
        self._reduced_passive = np.zeros(projected.shape[1:])
        self._reduced_override = (
            None if self._override_gain_m_inv is None else
            self.radial._project_cartesian_scalar_to_angular(self._override_gain_m_inv)
        )
        self._reduced_override_initial = (
            None if self._reduced_override is None else self._reduced_override.copy()
        )
        return field

    def _local_material_step(self, field, population, z_index, step_length_m):
        mask = self.crystal_mask
        passive, override = self.passive_loss_energy_J_m3, self._override_gain_m_inv
        self.crystal_mask = self.radial._axisym_mask
        self.passive_loss_energy_J_m3 = self._reduced_passive
        self._override_gain_m_inv = self._reduced_override
        try:
            field, transfer = SpectralSignalPropagator._local_material_step(
                self, field, self._reduced_population, z_index, step_length_m)
            self._reduced_override = self._override_gain_m_inv
        finally:
            self.crystal_mask = mask
            self.passive_loss_energy_J_m3, self._override_gain_m_inv = passive, override
        # Projection loss is numerical, not stimulated absorption or heat.
        before = self._integral(field)
        if self.radial.maximum_azimuthal_order:
            field = self.radial._truncate_angular_samples(field, track_field_omission=True)
        self._gain_projection_input_J += before
        self._gain_projection_loss_J += max(0., before - self._integral(field))
        return field, self._expand_scalar(transfer)

    def _exit_gain_transport(self, field, population, delta_n, pass_number):
        delta, error = self.radial._map_angular_population_delta(
            population, self._reduced_population - self._reduced_population_initial)
        population += delta
        self._population_mapping_error = max(self._population_mapping_error, error)
        self.passive_loss_energy_J_m3 += self._expand_scalar(self._reduced_passive)
        if self._reduced_override is not None:
            self._gain_mapping_redistribution = max(
                self._gain_mapping_redistribution,
                self._commit_axisymmetric_override_depletion(
                    self._reduced_override_initial, self._reduced_override))
        return self._exit_average_transport(field)

    def _signal_volume_density(self, field):
        return self._expand_scalar(np.sum(abs(field)**2, axis=(0, 1)))

    def _longitudinal_spatial_sample(self, field):
        if not self.radial._is_reduced_field(field):
            return super()._longitudinal_spatial_sample(field)
        radial = self.radial
        theta = radial.angular_coordinate_rad[:, None]
        r = radial.radial_centers_m
        cx = .5 * (self.grid.x_edges[0] + self.grid.x_edges[-1])
        cy = .5 * (self.grid.y_edges[0] + self.grid.y_edges[-1])
        fluence = np.sum(abs(field)**2, axis=tuple(range(field.ndim - 2)))
        area = np.broadcast_to(radial.angular_cell_area_m2, field.shape[-2:])
        if len(theta) == 1:
            # One propagated m=0 sample represents a whole annulus. Four
            # diagnostic angles integrate its first and second moments exactly.
            theta = (np.arange(4)[:, None] + 0.5) * (np.pi / 2.0)
            fluence = np.broadcast_to(fluence, (4, len(r)))
            area = np.broadcast_to(area / 4.0, fluence.shape)
        return (fluence, area,
                cx + np.cos(theta) * r, cy + np.sin(theta) * r)

    def _longitudinal_cartesian_field(self, field):
        return self.radial._expand_radial_field(field) if self.radial._is_reduced_field(field) else field

    def _spatial_diagnostics(self):
        r = self.radial
        return dict(spatial_backend="axisymmetric", azimuthal_fidelity=r.azimuthal_fidelity,
                    radial_bins=r.radial_bins, angular_samples=r.angular_samples,
                    temporal_bins=len(self.temporal.coordinate), spectral_bins=len(self.spectrum.coordinate),
                    diffraction_model="radial_crank_nicolson", kerr_enabled=False,
                    population_mapping_relative_error=self._population_mapping_error,
                    override_gain_mapping_redistribution_fraction=self._gain_mapping_redistribution,
                    entrance_projection_retained_energy_fraction=r.axisym_total_projection_energy_fraction,
                    maximum_energy_weighted_boundary_fraction=r.axisym_maximum_energy_weighted_boundary_fraction,
                    maximum_dynamic_field_omitted_energy_fraction=r.axisym_maximum_dynamic_field_omitted_energy_fraction,
                    maximum_population_azimuthal_residual=r.axisym_maximum_population_azimuthal_residual,
                    maximum_delta_n_azimuthal_residual=r.axisym_maximum_delta_n_azimuthal_residual,
                    maximum_population_realizability_correction_fraction=r.axisym_maximum_population_realizability_correction_fraction,
                    input_bin_reduction=self.input_pulse_projection.get("bin_reduction", {}))

    def _gain_transport_diagnostics(self):
        return {**super()._gain_transport_diagnostics(), **self._spatial_diagnostics()}

    def _average_transport_diagnostics(self):
        return {**super()._average_transport_diagnostics(), **self._spatial_diagnostics(), "temporal_bins": 1}

    def _transport_progress_metrics(self):
        return {**super()._transport_progress_metrics(), **self._spatial_diagnostics()}

    def propagate(self, *args, **kwargs):
        self.radial._reset_axisymmetric_diagnostics()
        self.radial._reset_thermal_phase_runtime()
        self._population_mapping_error = 0.0
        self._gain_mapping_redistribution = 0.0
        self._gain_projection_input_J = self._gain_projection_loss_J = 0.0
        self._thermal_projection_input_W = self._thermal_projection_loss_W = 0.0
        try:
            result = super().propagate(*args, **kwargs)
            # The shared average helper accumulates power in average runs and
            # energy in pulse runs. Here both accumulators therefore contain J.
            for name, total, loss in (
                ("gain", self._gain_projection_input_J, self._gain_projection_loss_J),
                ("thermal", self._thermal_projection_input_W, self._thermal_projection_loss_W),
            ):
                if loss:
                    result.external_loss_ledger.append(dict(
                        name="spectral_time_" + name + "_azimuthal_projection",
                        category="numerical_projection", input_energy_J=total,
                        output_energy_J=total - loss, signed_loss_energy_J=loss,
                        included_in_seed_balance=True))
            if self._gain_mapping_redistribution > float(self.radial.axisymmetric_cfg.get(
                    "maximum_population_positivity_redistribution_fraction", .05)):
                result.warnings.append(
                    f"Spectral Time polar gain-reservoir remap redistributed {self._gain_mapping_redistribution:.6g} "
                    "of the requested depletion; refine the radial/optical grid.")
            return result
        finally:
            self._restore_average_environment()
            self._reduced_population = self._reduced_population_initial = None
            self._reduced_passive = self._reduced_override = self._reduced_override_initial = None
            self.radial._axisym_delta_n = None
