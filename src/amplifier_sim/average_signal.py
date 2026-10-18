"""Cycle-averaged Seed propagation for fast high-repetition steady states.

The ordinary signal solvers transport one pulse and therefore store fluence.
For a high-repetition amplifier whose population changes only weakly during a
single pulse, the pulse train can instead be represented by its average photon
flux.  This module propagates average spectral power through the same optical
train and crystal grid without committing a per-pulse population jump.  The
resulting local spectral intensity is consumed by ``LevelSystem.steady_state``
as the stimulated-transition term of the algebraic rate equations.

This is deliberately an approximation and does not replace the exact periodic
pulse-map solver when one pulse appreciably depletes the stored population.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .dispersion import material_refractive_index
from .optical_system import (
    additive_source_gain_angle_warning,
    build_external_field,
    gain_medium_axis_angles_rad,
    gain_medium_path_cosine,
)
from .optics import OpticalTrace
from .pass_routing import resolve_pass_reverse
from .polarization import (
    apply_thermal_birefringence,
    initialize_jones_field,
    jones_enabled,
    polarization_diagnostics,
)
from .signal import (
    SignalPropagationResult,
    append_gain_entrance_loss_terms,
    gain_entrance_energy_partition,
    reference_ray_warnings,
    report_signal_transport_progress,
)
from .spectral_signal import SpectralSignalPropagator
from .seed_spatio_spectral import (
    normalized_field_overlap,
    sampled_field_moments,
    seed_config_at_wavelength,
)
from .seed_gain_transport import (
    GAUSSIAN_MODE_OVERLAP,
    PARABOLIC_GAIN_FIT,
    apply_modal_overlap_gain,
    parabolic_gain_fit_map,
    project_to_gaussian_mode,
    spatial_gain_transport_model,
)


class AveragePowerSignalPropagator(SpectralSignalPropagator):
    """Propagate average Seed power and report its local spectral intensity."""

    def _average_volume_shape(self):
        return self.grid.shape

    def _average_transport_mask(self):
        return self.crystal_mask

    def _enter_average_transport(self, field, population, delta_n):
        return field, population

    def _exit_average_transport(self, field):
        return field

    def _project_average_gain_field(self, field):
        return field, None

    def _finish_average_volume(self, values, *, spectral=False):
        return values

    def _average_transport_diagnostics(self):
        return {}

    @staticmethod
    def _cell_average_exponential_intensity(
        entrance_intensity: np.ndarray,
        gain_exponent: np.ndarray,
    ) -> np.ndarray:
        """Return the exact cell average for ``I(s)=I(0) exp(g s)``."""

        exponent = np.asarray(gain_exponent, dtype=float)
        phi_one = np.divide(
            np.expm1(exponent),
            exponent,
            out=np.ones_like(exponent),
            where=np.abs(exponent) > 1e-12,
        )
        return np.asarray(entrance_intensity, dtype=float) * phi_one

    @classmethod
    def _cell_volume_average_exponential_intensity(
        cls,
        entrance_intensity: np.ndarray,
        gain_exponent: np.ndarray,
        optical_length_m: float,
        cell_length_m: float,
    ) -> np.ndarray:
        """Return ``integral(I ds) / dz`` for a physical axial voxel."""

        optical_length = float(optical_length_m)
        cell_length = float(cell_length_m)
        if optical_length <= 0.0 or cell_length <= 0.0:
            raise ValueError("optical and physical cell lengths must be positive")
        return (
            cls._cell_average_exponential_intensity(
                entrance_intensity,
                gain_exponent,
            )
            * optical_length
            / cell_length
        )

    def propagate_average(
        self,
        population: np.ndarray,
        delta_n: np.ndarray,
        repetition_rate_Hz: float,
        record_planes: bool = False,
    ) -> SignalPropagationResult:
        record_planes = bool(record_planes) and self._diagnostics_enabled()
        self._reset_thermal_phase_runtime()
        self.levels.record_spectral_support(
            "signal", self.spectrum.coordinate, self.spectrum.weights
        )
        repetition_rate = float(repetition_rate_Hz)
        if repetition_rate <= 0.0:
            raise ValueError("cycle-averaged Seed propagation requires repetition_rate_Hz > 0")
        if self.levels_cfg.get("gain_override_m_inv") is not None:
            raise ValueError(
                "cycle-averaged population steady state requires physical upper/lower levels; "
                "remove energy_levels.gain_override_m_inv"
            )
        # The inherited aperture ledger integrates |field|^2.  In this
        # average-power propagator that quantity is watts and is converted to
        # per-pulse joules only when constructing the public result.
        self.crystal_aperture_loss_energy_J = 0.0

        wavelengths = np.asarray(self.spectrum.coordinate, dtype=float)
        configured_power = float(self.cfg["pulse_energy_J"]) * repetition_rate
        spectral_fields: list[np.ndarray] = []
        outside_indices: list[float] = []
        source_powers: list[float] = []
        source_spatio_samples: list[dict[str, Any]] = []
        wavelength_source_configs: list[dict[str, Any]] = []
        warnings = [
            *self.bench_warnings,
            *self.source_definition_warnings,
            *reference_ray_warnings(
                self.gain_marker,
                self.gain_markers,
                self.post_elements,
                self.between_pass_elements,
                self.between_pass_segments,
            ),
        ]
        angle_warning = additive_source_gain_angle_warning(
            self.cfg,
            self.gain_markers if self.gain_markers else [self.gain_marker],
            "Seed",
        )
        if angle_warning is not None:
            warnings.append(angle_warning)
        maximum_cell_gain_exponent = -np.inf
        positive_exponent_clipping_active = False
        propagation_mode = str(self.physics["optical_propagation"])
        spatial_gain_model = spatial_gain_transport_model(self.cfg)
        modal_axial_samples: list[dict[str, Any]] = []
        modal_projection_overlaps: list[float] = []
        parabolic_axial_fits: list[dict[str, Any]] = []
        external_mode = (
            "abcd" if propagation_mode in {"abcd", "hybrid_abcd_fft"} else propagation_mode
        )
        source_channel_total = len(wavelengths)
        report_signal_transport_progress(
            self,
            current=0,
            total=source_channel_total,
            pass_index=0,
            pass_total=0,
            z_index=None,
            phase="spectral_source_setup",
            fraction_start=0.0,
            fraction_stop=0.12,
            metrics={
                **self._average_transport_diagnostics(),
                "spectral_channels": source_channel_total,
                "optical_grid_nx": len(self.grid.x),
                "optical_grid_ny": len(self.grid.y),
                "working_field_estimate_bytes": (
                    source_channel_total
                    * len(self.grid.x)
                    * len(self.grid.y)
                    * np.dtype(np.complex128).itemsize
                ),
            },
        )
        for channel_index, (wavelength, weight) in enumerate(
            zip(wavelengths, self.spectrum.weights, strict=True)
        ):
            local_cfg, sample_diagnostics = self._source_config_for_channel(
                channel_index, float(wavelength)
            )
            entrance_diagnostics: dict[str, Any] = {}
            external = build_external_field(
                local_cfg,
                configured_power * float(weight),
                self.pre_elements,
                self.grid,
                external_mode,
                "power_W",
                self.physics["interface_model"],
                False,
                include_abcd_tilt_phase=propagation_mode in {"abcd", "hybrid_abcd_fft"},
                abcd_diagnostics=entrance_diagnostics,
            )
            if not entrance_diagnostics:
                entrance_diagnostics.update(sampled_field_moments(external.field, self.grid))
                entrance_diagnostics["tilt_phase_included"] = bool(
                    local_cfg.get("incidence_angle_x_deg", 0.0)
                    or local_cfg.get("incidence_angle_y_deg", 0.0)
                )
            sample_diagnostics["gain_entrance"] = entrance_diagnostics
            source_spatio_samples.append(sample_diagnostics)
            wavelength_source_configs.append(local_cfg)
            spectral_fields.append(external.field)
            outside_indices.append(external.refractive_index)
            source_powers.append(external.source_grid_quantity)
            warnings.extend(external.warnings)
            report_signal_transport_progress(
                self,
                current=channel_index + 1,
                total=source_channel_total,
                pass_index=0,
                pass_total=0,
                z_index=None,
                phase="spectral_source_setup",
                fraction_start=0.0,
                fraction_stop=0.12,
                metrics={
                    "source_channel_wavelength_m": float(wavelength),
                    "prepared_channel_power_W": float(
                        external.source_grid_quantity
                    ),
                    "spectral_channels": source_channel_total,
                },
            )
        # Axis zero is a singleton channel in the scalar case and the two
        # orthogonal Jones components when vector propagation is requested.
        scalar_spectral_field = np.stack(spectral_fields, axis=0)
        self._jones = jones_enabled(self.cfg)
        field = (
            initialize_jones_field(scalar_spectral_field, self.cfg)
            if self._jones
            else scalar_spectral_field[None, ...]
        )
        external_loss_ledger: list[dict[str, Any]] = []

        def record_external_loss(
            name: str, before_W: float, after_W: float, category: str
        ) -> None:
            before_J = float(before_W) / repetition_rate
            after_J = float(after_W) / repetition_rate
            external_loss_ledger.append(
                {
                    "name": name,
                    "category": category,
                    "input_energy_J": before_J,
                    "output_energy_J": after_J,
                    "signed_loss_energy_J": before_J - after_J,
                    "input_average_power_W": float(before_W),
                    "output_average_power_W": float(after_W),
                    "included_in_seed_balance": True,
                }
            )

        gain_entrance_partition = gain_entrance_energy_partition(
            configured_source_energy_J=configured_power / repetition_rate,
            sampled_gain_entrance_quantity=self._integral(field),
            wavelength_samples=source_spatio_samples,
            quantity_to_energy=1.0 / repetition_rate,
        )
        if gain_entrance_partition is None:
            record_external_loss(
                "source_to_gain_optical_train",
                float(np.sum(source_powers)),
                self._integral(field),
                "pre_gain_routing",
            )
        else:
            append_gain_entrance_loss_terms(
                external_loss_ledger, gain_entrance_partition
            )
        accumulated_retardance = np.zeros(self.grid.shape[-2:], dtype=float)
        reference_index = int(np.argmin(np.abs(wavelengths - float(self.cfg["wavelength_m"]))))
        for index, sample in enumerate(source_spatio_samples):
            sample["gain_entrance"]["mode_overlap_with_reference"] = normalized_field_overlap(
                spectral_fields[index], spectral_fields[reference_index], self.grid
            )
        spectral_fields.clear()
        del scalar_spectral_field
        reference_entrance = source_spatio_samples[reference_index]["gain_entrance"]
        self._wavelength_source_configs = wavelength_source_configs
        current_outside_n = np.asarray(outside_indices, dtype=float)
        n_gain = np.asarray(
            [
                material_refractive_index(
                    self.material,
                    float(wavelength),
                    role="signal",
                )
                for wavelength in wavelengths
            ],
            dtype=float,
        )
        trace = OpticalTrace()
        if record_planes:
            trace.add("average_seed_source_to_gain", field, self.grid, "power_W")

        volume_shape = self._average_volume_shape()
        transport_mask = self._average_transport_mask()
        spectral_intensity = np.zeros((len(wavelengths), *volume_shape), dtype=float)
        signal_intensity = np.zeros(volume_shape, dtype=float)
        last_pass_intensity = np.zeros(volume_shape, dtype=float)
        per_pass_intensities: list[np.ndarray] = []
        volume_recording = str(
            self.simulation.get("signal_volume_recording", "last_and_cumulative")
        )
        extracted_power = np.zeros(volume_shape, dtype=float)
        passive_loss_power = np.zeros(volume_shape, dtype=float)
        pass_stats: list[dict[str, Any]] = []
        passes = len(self.gain_markers) if self.gain_markers else int(self.simulation["passes"])
        passive_loss = float(self.material.get("signal_passive_loss_m_inv", 0.0))
        transport_total = passes * len(self.grid.z)
        report_signal_transport_progress(
            self,
            current=0,
            total=transport_total,
            pass_index=1,
            pass_total=passes,
            z_index=None,
            fraction_start=0.12,
            fraction_stop=0.96,
        )

        for pass_number in range(passes):
            pass_aperture_start = self.crystal_aperture_loss_energy_J
            pass_intensity = np.zeros(volume_shape, dtype=float)
            marker = self.gain_markers[pass_number] if self.gain_markers else self.gain_marker
            reverse = resolve_pass_reverse(
                str(self.cfg.get("direction", "forward")),
                str(marker.get("direction", "auto")),
                pass_number,
                bool(self.simulation.get("alternate_direction", True)),
                explicit_gain_hit=bool(self.gain_markers),
            )
            self._active_gain_marker = marker
            self._active_gain_reverse = reverse
            entrance_side = "back" if reverse else "front"
            exit_side = "front" if reverse else "back"
            interface_input = self._integral(field)
            for il, wavelength in enumerate(wavelengths):
                field[:, il] = self._gain_interface(
                    field[:, il],
                    float(current_outside_n[il]),
                    float(n_gain[il]),
                    marker,
                    entrance_side,
                    self.physics["interface_model"],
                    float(wavelength),
                    entering=True,
                    surface_sag_m=self._endface_surface_sag_m[entrance_side],
                    channel_index=il,
                )
            record_external_loss(
                f"pass_{pass_number + 1}_{entrance_side}_gain_interface",
                interface_input,
                self._integral(field),
                "gain_interface",
            )
            if self.clip_crystal:
                field = self._apply_crystal_aperture(field)
            if record_planes:
                trace.add(
                    f"average_pass_{pass_number + 1}_gain_entrance",
                    field,
                    self.grid,
                    "power_W",
                )
            field, transport_population = self._enter_average_transport(field, population, delta_n)
            power_in = self._integral(field)
            angle_x, angle_y = gain_medium_axis_angles_rad(marker)
            path_cosine = gain_medium_path_cosine(marker)
            step_length = self.grid.dz / path_cosine
            indices = list(reversed(range(len(self.grid.z)))) if reverse else list(range(len(self.grid.z)))

            for cell_number, iz in enumerate(indices, start=1):
                if self.physics["optical_propagation"] in {"abcd", "fft", "hybrid_abcd_fft"}:
                    field = self._diffract(field, step_length / 2.0, n_gain)
                    if self.clip_crystal:
                        field = self._apply_crystal_aperture(field)
                if self._jones:
                    field, local_retardance = apply_thermal_birefringence(
                        field,
                        delta_n[iz],
                        step_length,
                        wavelengths,
                        n_gain,
                        self.grid,
                        self.material,
                        self.cfg,
                        None
                        if self.photoelastic_tensor_fields is None
                        else {
                            key: values[iz]
                            for key, values in self.photoelastic_tensor_fields.items()
                        },
                    )
                    accumulated_retardance += local_retardance
                else:
                    field = self._apply_thermal_phase(
                        field,
                        delta_n[iz],
                        step_length,
                        iz,
                    )

                intensity_before = np.sum(np.abs(field) ** 2, axis=0)
                active_gain = self.levels.signal_gain_spectrum_m_inv(
                    transport_population[:, iz],
                    wavelengths,
                    passive_m_inv=0.0,
                    z_index=iz,
                )
                active_gain *= transport_mask[None, :, :]
                if spatial_gain_model == PARABOLIC_GAIN_FIT:
                    active_gain, fit_rows = parabolic_gain_fit_map(
                        active_gain,
                        self.grid,
                        transport_mask,
                        self.cfg.get("spatial_gain_transport", {}),
                    )
                    for row in fit_rows:
                        parabolic_axial_fits.append(
                            {
                                "pass": pass_number + 1,
                                "direction": (
                                    "backward" if reverse else "forward"
                                ),
                                "z_index": int(iz),
                                "z_m": float(self.grid.z[iz]),
                                **row,
                            }
                        )
                total_gain = active_gain - passive_loss * transport_mask[None, :, :]
                if spatial_gain_model == GAUSSIAN_MODE_OVERLAP:
                    projection = project_to_gaussian_mode(field[0], self.grid)
                    field[0] = projection.field
                    modal_projection_overlaps.extend(
                        float(value)
                        for value in np.ravel(projection.coherent_overlap)
                    )
                    if self.clip_crystal:
                        field = self._apply_crystal_aperture(field)
                    intensity_before = np.abs(field[0]) ** 2
                    modal_step = apply_modal_overlap_gain(
                        field[0],
                        active_gain,
                        passive_loss * transport_mask[None, :, :],
                        self.grid,
                        step_length,
                        self.grid.dz,
                    )
                    field[0] = modal_step.field
                    if self.clip_crystal:
                        field *= transport_mask[None, None, :, :]
                    intensity_after = modal_step.intensity_after
                    local_spectral = modal_step.cell_average_intensity
                    raw_gain_exponent = modal_step.raw_gain_exponent
                    extracted_power[iz] += np.sum(
                        modal_step.active_transfer_per_volume, axis=0
                    )
                    passive_loss_power[iz] += np.sum(
                        modal_step.passive_loss_per_volume, axis=0
                    )
                    if bool(
                        self.cfg.get("spatial_gain_transport", {}).get(
                            "record_axial_diagnostics", True
                        )
                    ):
                        combined_field = np.sqrt(
                            np.sum(np.abs(field[0]) ** 2, axis=0)
                        ).astype(complex)
                        moments = sampled_field_moments(combined_field, self.grid)
                        modal_axial_samples.append(
                            {
                                "pass": pass_number + 1,
                                "direction": "backward" if reverse else "forward",
                                "z_index": int(iz),
                                "z_m": float(self.grid.z[iz]),
                                "input_average_power_W": float(
                                    np.sum(
                                        self.grid.integrate_transverse(
                                            modal_step.intensity_before
                                        )
                                    )
                                ),
                                "output_average_power_W": float(
                                    np.sum(
                                        self.grid.integrate_transverse(
                                            modal_step.intensity_after
                                        )
                                    )
                                ),
                                "effective_active_gain_m_inv": [
                                    float(value)
                                    for value in np.atleast_1d(
                                        modal_step.effective_active_gain_m_inv
                                    )
                                ],
                                "effective_passive_loss_m_inv": [
                                    float(value)
                                    for value in np.atleast_1d(
                                        modal_step.effective_passive_loss_m_inv
                                    )
                                ],
                                "effective_total_gain_m_inv": [
                                    float(value)
                                    for value in np.atleast_1d(
                                        modal_step.effective_total_gain_m_inv
                                    )
                                ],
                                "gain_exponent": [
                                    float(value)
                                    for value in np.atleast_1d(
                                        modal_step.gain_exponent
                                    )
                                ],
                                "raw_gain_exponent": [
                                    float(value)
                                    for value in np.atleast_1d(
                                        modal_step.raw_gain_exponent
                                    )
                                ],
                                "gaussian_projection_overlap": float(
                                    np.min(projection.coherent_overlap)
                                ),
                                "radius_x_m": moments.get("radius_x_m"),
                                "radius_y_m": moments.get("radius_y_m"),
                                "center_x_m": moments.get("center_x_m"),
                                "center_y_m": moments.get("center_y_m"),
                            }
                        )
                else:
                    raw_gain_exponent = total_gain * step_length
                    # Large negative exponents are safe in exp/expm1 and must
                    # not be clipped: the exact cell average scales as 1/|g ds|.
                    gain_exponent = np.minimum(raw_gain_exponent, 50.0)
                    intensity_after = intensity_before * np.exp(gain_exponent)
                    scale = np.sqrt(
                        np.divide(
                            intensity_after,
                            np.maximum(intensity_before, np.finfo(float).tiny),
                        )
                    )
                    field *= np.where(
                        intensity_before > 0.0, scale, 0.0
                    )[None, ...]
                    if self.clip_crystal:
                        field *= transport_mask[None, None, :, :]
                    local_spectral = self._cell_volume_average_exponential_intensity(
                        intensity_before,
                        gain_exponent,
                        step_length,
                        self.grid.dz,
                    )
                    # For linear cycle-averaged transport, the exact active
                    # source is g_active * integral(I ds) / dz.  If the
                    # positive exponent safety cap is active, the field no
                    # longer experienced ``active_gain`` itself: use the
                    # active part of the *applied* net exponent so that the
                    # active-minus-passive ledger still equals the actual
                    # optical power change across this cell.
                    applied_active_gain = (
                        gain_exponent / step_length
                        + passive_loss * transport_mask[None, :, :]
                    )
                    extracted_power[iz] += np.sum(
                        applied_active_gain * local_spectral,
                        axis=0,
                    )
                    # ``local_spectral`` is integral(I ds) / dz, so alpha*I is
                    # already expressed per physical axial voxel.  The modal
                    # branch records the same quantity in its dedicated
                    # passive-loss ledger above.
                    passive_loss_power[iz] += (
                        passive_loss
                        * transport_mask
                        * np.sum(local_spectral, axis=0)
                    )
                field, projection_loss = self._project_average_gain_field(field)
                if projection_loss is not None:
                    record_external_loss(
                        f"pass_{pass_number + 1}_cell_{iz}_azimuthal_projection",
                        *projection_loss, "numerical_projection",
                    )
                maximum_cell_gain_exponent = max(
                    maximum_cell_gain_exponent,
                    float(np.max(raw_gain_exponent)),
                )
                positive_exponent_clipping_active |= bool(
                    np.any(raw_gain_exponent > 50.0)
                )
                # The exponential average is first taken along the slanted
                # optical path.  Rate equations, however, consume photon flux
                # per physical axial voxel: integral(I ds) / dz.  This factor
                # is unity at normal incidence and closes stimulated power
                # against the optical energy change at oblique incidence.
                spectral_intensity[:, iz] += local_spectral
                signal_intensity[iz] += np.sum(local_spectral, axis=0)
                pass_intensity[iz] = np.sum(local_spectral, axis=0)
                if self.physics["optical_propagation"] in {"abcd", "fft", "hybrid_abcd_fft"}:
                    field = self._diffract(field, step_length / 2.0, n_gain)
                    if self.clip_crystal:
                        field = self._apply_crystal_aperture(field)
                if spatial_gain_model == GAUSSIAN_MODE_OVERLAP:
                    projection = project_to_gaussian_mode(field[0], self.grid)
                    field[0] = projection.field
                    modal_projection_overlaps.extend(
                        float(value)
                        for value in np.ravel(projection.coherent_overlap)
                    )
                    if self.clip_crystal:
                        field = self._apply_crystal_aperture(field)
                completed_cells = pass_number * len(indices) + cell_number
                report_signal_transport_progress(
                    self,
                    current=completed_cells,
                    total=transport_total,
                    pass_index=pass_number + 1,
                    pass_total=passes,
                    z_index=iz,
                    fraction_start=0.12,
                    fraction_stop=0.96,
                    metrics={"transport_power_W": self._integral(field)},
                )

            gain_output_power = self._integral(field)
            field = self._exit_average_transport(field)
            if record_planes:
                trace.add(
                    f"average_pass_{pass_number + 1}_gain_exit",
                    field,
                    self.grid,
                    "power_W",
                )
            last_pass_intensity = pass_intensity
            if volume_recording == "per_pass":
                per_pass_intensities.append(pass_intensity.copy())
            configured_outside_n = marker.get("outside_refractive_index")
            outside_n = (
                current_outside_n.copy()
                if configured_outside_n is None
                else np.full_like(current_outside_n, float(configured_outside_n))
            )
            interface_input = self._integral(field)
            for il, wavelength in enumerate(wavelengths):
                field[:, il] = self._gain_interface(
                    field[:, il],
                    float(n_gain[il]),
                    float(outside_n[il]),
                    marker,
                    exit_side,
                    self.physics["interface_model"],
                    float(wavelength),
                    entering=False,
                    surface_sag_m=self._endface_surface_sag_m[exit_side],
                    channel_index=il,
                )
            record_external_loss(
                f"pass_{pass_number + 1}_{exit_side}_gain_interface",
                interface_input,
                self._integral(field),
                "gain_interface",
            )
            current_outside_n = outside_n
            crystal_output_power = self._integral(field)
            pass_stats.append(
                {
                    "pass": pass_number + 1,
                    "direction": "backward" if reverse else "forward",
                    "incidence_angle_x_deg": math.degrees(angle_x),
                    "incidence_angle_y_deg": math.degrees(angle_y),
                    "effective_path_length_m": len(self.grid.z) * self.grid.dz / path_cosine,
                    "gain_input_J": power_in / repetition_rate,
                    "gain_output_J": gain_output_power / repetition_rate,
                    "crystal_exit_J": crystal_output_power / repetition_rate,
                    "gain_input_average_power_W": power_in,
                    "gain_output_average_power_W": gain_output_power,
                    "crystal_exit_average_power_W": crystal_output_power,
                    "crystal_aperture_loss_J": (
                        self.crystal_aperture_loss_energy_J
                        - pass_aperture_start
                    )
                    / repetition_rate,
                    "crystal_aperture_loss_average_power_W": (
                        self.crystal_aperture_loss_energy_J
                        - pass_aperture_start
                    ),
                }
            )
            if pass_number + 1 < passes:
                routing_input = self._integral(field)
                field *= math.sqrt(float(self.simulation["between_pass_transmission"]))
                record_external_loss(
                    f"between_pass_{pass_number + 1}_{pass_number + 2}_configured_transmission",
                    routing_input,
                    self._integral(field),
                    "between_pass_transmission",
                )
                elements = (
                    self.between_pass_segments[pass_number]
                    if self.gain_markers and pass_number < len(self.between_pass_segments)
                    else self.between_pass_elements
                )
                train_input = self._integral(field)
                field, current_outside_n = self._propagate_train(
                    field,
                    elements,
                    current_outside_n,
                )
                record_external_loss(
                    f"between_pass_{pass_number + 1}_{pass_number + 2}_optical_train",
                    train_input,
                    self._integral(field),
                    "between_pass_elements",
                )
                if record_planes:
                    trace.add(
                        f"average_between_pass_{pass_number + 1}_{pass_number + 2}",
                        field,
                        self.grid,
                        "power_W",
                    )
                if self.simulation.get("flip_x_between_passes", False):
                    field = np.flip(field, axis=-1).copy()

        report_signal_transport_progress(
            self,
            current=0,
            total=1,
            pass_index=0,
            pass_total=0,
            z_index=None,
            phase="output_postprocess",
            fraction_start=0.96,
            fraction_stop=0.995,
        )
        post_input = self._integral(field)
        field, current_outside_n = self._propagate_train(
            field,
            self.post_elements,
            current_outside_n,
        )
        record_external_loss(
            "post_gain_optical_train",
            post_input,
            self._integral(field),
            "post_gain_routing",
        )
        if record_planes:
            trace.add("average_seed_system_output", field, self.grid, "power_W")
        output_power = self._integral(field)
        if pass_stats:
            pass_stats[-1]["system_output_J"] = output_power / repetition_rate
            pass_stats[-1]["system_output_average_power_W"] = output_power
        output_spectrum_power = self._spectral_energy(field)
        output_polarization = (
            polarization_diagnostics(
                field,
                self.grid,
                self.cfg,
                accumulated_retardance,
            )
            if self._jones
            else None
        )
        source_spectrum_power = np.asarray(source_powers, dtype=float)
        source_power = float(np.sum(source_spectrum_power))
        caustic_diagnostics = self._scan_output_caustic(
            field,
            trace,
            "power_W",
            wavelengths,
        )
        result_extras = self._result_extras(
            field,
            source_power / repetition_rate,
            source_spectrum_power / repetition_rate,
            output_spectrum_power / repetition_rate,
        )
        report_signal_transport_progress(
            self,
            current=1,
            total=1,
            pass_index=0,
            pass_total=0,
            z_index=None,
            phase="output_postprocess",
            fraction_start=0.96,
            fraction_stop=0.995,
            metrics={"output_power_W": output_power},
        )
        if positive_exponent_clipping_active:
            warnings.append(
                "At least one positive cycle-average Seed single-cell gain exponent exceeded "
                "50 and was clipped before exponentiation. Refine Nz or use a saturated pulse "
                "transport model before interpreting this operating point quantitatively."
            )
        if (
            spatial_gain_model == GAUSSIAN_MODE_OVERLAP
            and modal_projection_overlaps
            and min(modal_projection_overlaps) < 0.99
        ):
            warnings.append(
                "Gaussian-mode overlap discarded non-Gaussian spatial content; the minimum "
                f"coherent projection overlap was {min(modal_projection_overlaps):.6g}. "
                "Compare with local_field before treating this reduced-order result as physical."
            )

        spectral_intensity = self._finish_average_volume(spectral_intensity, spectral=True)
        signal_intensity = self._finish_average_volume(signal_intensity)
        last_pass_intensity = self._finish_average_volume(last_pass_intensity)
        per_pass_intensities = [self._finish_average_volume(value) for value in per_pass_intensities]
        extracted_power = self._finish_average_volume(extracted_power)
        passive_loss_power = self._finish_average_volume(passive_loss_power)
        result = SignalPropagationResult(
            fluence_J_m2=signal_intensity / repetition_rate,
            population_m3=population.copy(),
            passes=pass_stats,
            source_grid_energy_J=source_power / repetition_rate,
            system_output_energy_J=output_power / repetition_rate,
            trace=trace,
            warnings=list(dict.fromkeys(warnings)),
            wavelengths_m=wavelengths.copy(),
            input_spectrum_J=source_spectrum_power / repetition_rate,
            output_spectrum_J=output_spectrum_power / repetition_rate,
            model_used="cycle_averaged_steady_probe",
            extracted_energy_J_m3=extracted_power / repetition_rate,
            gain_transport={
                "backend": (
                    GAUSSIAN_MODE_OVERLAP
                    if spatial_gain_model == GAUSSIAN_MODE_OVERLAP
                    else PARABOLIC_GAIN_FIT
                    if spatial_gain_model == PARABOLIC_GAIN_FIT
                    else "cycle_averaged_linear_transport"
                ),
                "status": "applied",
                "temporal_closure": "cycle_averaged_linear_transport",
                "spatial_closure": (
                    "fixed_transported_mode"
                    if spatial_gain_model == GAUSSIAN_MODE_OVERLAP
                    else "axis_aligned_local_parabolic_gain_fit"
                    if spatial_gain_model == PARABOLIC_GAIN_FIT
                    else "local_field"
                ),
                "gain_guiding": spatial_gain_model != GAUSSIAN_MODE_OVERLAP,
                "gaussian_projection": (
                    "power_preserving_moment_match_each_z_cell"
                    if spatial_gain_model == GAUSSIAN_MODE_OVERLAP
                    else None
                ),
                "minimum_gaussian_projection_overlap": (
                    min(modal_projection_overlaps)
                    if modal_projection_overlaps
                    else None
                ),
                "downstream_spatial_reshaping": any(
                    element.get("type") in {"aperture", "phase_screen"}
                    for element in (
                        self.post_elements
                        + self.between_pass_elements
                        + [
                            item
                            for segment in self.between_pass_segments
                            for item in segment
                        ]
                    )
                ),
                **self._average_transport_diagnostics(),
                "spectral_bins": len(wavelengths),
                "repetition_rate_Hz": repetition_rate,
                "source_average_power_W": source_power,
                "output_average_power_W": output_power,
                "population_committed": False,
                "maximum_cell_gain_exponent": float(maximum_cell_gain_exponent),
                "positive_cell_exponent_clipping_active": positive_exponent_clipping_active,
                "axial_samples": modal_axial_samples,
                "parabolic_gain_fits": parabolic_axial_fits,
                **(
                    {
                        "thermal_lens": (
                            self._local_parabolic_thermal_lens.diagnostics()
                        )
                    }
                    if self._local_parabolic_thermal_lens is not None
                    else {}
                ),
            },
            average_spectral_intensity_W_m2=spectral_intensity,
            fluence_last_pass_J_m2=last_pass_intensity / repetition_rate,
            fluence_cumulative_J_m2=(
                signal_intensity / repetition_rate
                if volume_recording != "last_only"
                else None
            ),
            fluence_per_pass_J_m2=(
                np.stack(per_pass_intensities, axis=0) / repetition_rate
                if per_pass_intensities
                else None
            ),
            source_spatio_spectral={
                "model": str(self.cfg.get("spatio_spectral", {}).get("model", "shared_source")),
                "reference_wavelength_m": float(
                    self.cfg.get("spatio_spectral", {}).get("reference_wavelength_m")
                    or self.cfg["wavelength_m"]
                ),
                "wavelength_samples": source_spatio_samples,
                **(gain_entrance_partition or {}),
            },
            crystal_aperture_loss_energy_J=(
                self.crystal_aperture_loss_energy_J / repetition_rate
            ),
            external_loss_ledger=external_loss_ledger,
            passive_loss_energy_J_m3=passive_loss_power / repetition_rate,
            # Hidden fixed-point probes set ``_suppress_output_diagnostics``;
            # the final reported CW pass therefore performs this scan once.
            caustic_diagnostics=caustic_diagnostics,
            polarization_diagnostics=output_polarization,
            output_jones_field=field.copy() if self._jones else None,
            **result_extras,
        )
        return result
