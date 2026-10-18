"""Independent incoherent HG channels on a finite-azimuthal radial grid.

Each member carries its own complex spectrum. Angular Fourier orders describe
each member, not mutual coherence between the members. M2 is carried by modal
weights; every member uses unit-M2 diffraction.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .axisymmetric_coherent import AxisymmetricCoherentSignalPropagator
from .coherent_signal import IncoherentHGCoherentSignalPropagator
from .spectral_signal import SpectralSignalPropagator
from .spectral_signal import IncoherentHGSignalPropagator
from .axisymmetric_average import AxisymmetricAveragePowerSignalPropagator


class AxisymmetricHGCoherentSignalPropagator(
    AxisymmetricCoherentSignalPropagator, IncoherentHGCoherentSignalPropagator
):
    """Whole-fluence HG00/HG10/HG01 with shared gain and mean-intensity Kerr."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # The radial gain quadrature is defined on physical frequencies.
        # Combine all incoherent members into each frequency packet before
        # consuming population, then apply that packet's gain to every member.
        count = self._physical_spectral_count
        modes = len(self.incoherent_modes)
        self._axisym_gain_groups = tuple(
            np.concatenate([indices + mode * count for mode in range(modes)])
            for indices in self._axisym_gain_groups
        )
        for name in (
            "_axisym_gain_interp_lower",
            "_axisym_gain_interp_upper",
            "_axisym_gain_interp_upper_weight",
        ):
            setattr(self, name, np.tile(getattr(self, name), modes))

    def _validate_axisymmetric_contract(self) -> None:
        if self.coherent_fidelity_mode != "axisymmetric_kerr":
            raise ValueError("radial incoherent HG requires axisymmetric_kerr whole-fluence gain")
        if self.maximum_azimuthal_order != 1:
            raise ValueError("radial incoherent HG requires azimuthal_fidelity=m0_m1")
        if any(int(mode["order_x"]) + int(mode["order_y"]) > 1 for mode in self.incoherent_modes):
            raise ValueError("radial incoherent HG currently retains only HG00, HG10, HG01")
        if self.physics.get("nonlinear_loss_model", "off") != "off":
            raise ValueError("radial incoherent HG currently requires nonlinear loss off")
        if not np.isclose(float(self.cfg.get("m2_x", 1)), float(self.cfg.get("m2_y", 1))):
            raise ValueError("radial incoherent HG currently requires equal x/y M2")
        # Validate the geometry using the actual coherent members' M2, not the
        # ensemble moment. The source builder separately scales their waists.
        ensemble_cfg = self.cfg
        self.cfg = copy.deepcopy(ensemble_cfg)
        self.cfg.update(m2_x=1.0, m2_y=1.0)
        try:
            super()._validate_axisymmetric_contract()
        finally:
            self.cfg = ensemble_cfg

    def _simultaneous_gain_channel_groups(self) -> tuple[np.ndarray, ...]:
        if self.spectrum is getattr(self, "_axisym_gain_profile", None):
            return SpectralSignalPropagator._simultaneous_gain_channel_groups(self)
        return super()._simultaneous_gain_channel_groups()

    def _result_extras(self, field, source_total_J, source_spectrum_J, output_spectrum_J):
        extras = super()._result_extras(field, source_total_J, source_spectrum_J, output_spectrum_J)
        extras["cpa_diagnostics"].update(
            modal_linear_transport="independent_complex_E_frequency_theta_radius",
            modal_nonlinear_coupling="summed_instantaneous_intensity_common_Kerr_response",
        )
        return extras

    def propagate(self, population, delta_n):
        result = super().propagate(population, delta_n)
        if result.gain_transport is not None:
            result.gain_transport["backend"] = "coherent_cpa_radial_incoherent_hg"
        return result

    def _apply_material_dispersion(self, field: np.ndarray, step_length_m: float) -> np.ndarray:
        if not self._is_reduced_field(field):
            return super()._apply_material_dispersion(field, step_length_m)
        factor = self._material_dispersion_factor(step_length_m)
        if factor is not None:
            modes = self._coherent_spectral_modes(field)
            modes *= factor[None, :, None, None]
            self._store_coherent_spectral_modes(field, modes)
        return field

    def _diffract(self, field: np.ndarray, distance_m: float, refractive_index):
        plan = self._tail_frequency_plan
        if plan is None or not self._is_reduced_field(field):
            return super()._diffract(field, distance_m, refractive_index)
        # The energy-tail selection is a physical-frequency interval and must
        # be applied separately to every flattened HG member.
        lo, hi = plan["central_indices"]
        output = field.copy()
        for mode in range(len(self.incoherent_modes)):
            offset = mode * self._physical_spectral_count
            selected = slice(offset + lo, offset + hi + 1)
            output[:, selected] = self._diffract_selected(
                field[:, selected], distance_m, refractive_index, frequency_slice=selected
            )
        area = self.angular_cell_area_m2
        outer = max(1, int(np.ceil(.05 * self.radial_bins)))
        self._update_axisymmetric_boundary_diagnostics(
            np.sum(abs(output)**2 * area, axis=(-2, -1)),
            np.sum(abs(output[..., -outer:])**2 * area[..., -outer:], axis=(-2, -1)),
        )
        return output


class AxisymmetricHGAveragePowerSignalPropagator(
    AxisymmetricAveragePowerSignalPropagator, IncoherentHGSignalPropagator
):
    """Cycle-average initializer with the same independent input HG ensemble."""

    def _make_radial_helper(self, *args: Any, **kwargs: Any):
        return AxisymmetricHGCoherentSignalPropagator(*args, **kwargs)
