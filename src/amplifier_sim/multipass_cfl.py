"""Shared CFL-grid and local gain-competition operators for causal multipass."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .constants import photon_energy
from .optical_system import build_external_field
from .profiles import DiscreteProfile
from .seed_spatio_spectral import normalized_field_overlap, sampled_field_moments
from .signal import frantz_nodvik
from .spectral_signal import SpectralSignalPropagator, _SpectralEntrancePlan


@dataclass(frozen=True)
class _CflTemporalGrid:
    profile: DiscreteProfile
    edges_s: np.ndarray
    dt_s: float
    diagnostics: dict[str, Any]


def _finite_volume_edges(
    profile: DiscreteProfile,
    fallback_span_s: float,
) -> np.ndarray:
    coordinate = np.asarray(profile.coordinate, dtype=float)
    if len(coordinate) == 1:
        half = 0.5 * max(float(fallback_span_s), np.finfo(float).tiny)
        return np.array([coordinate[0] - half, coordinate[0] + half])
    edges = np.empty(len(coordinate) + 1, dtype=float)
    edges[1:-1] = 0.5 * (coordinate[:-1] + coordinate[1:])
    edges[0] = coordinate[0] - 0.5 * (coordinate[1] - coordinate[0])
    edges[-1] = coordinate[-1] + 0.5 * (coordinate[-1] - coordinate[-2])
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("Seed temporal coordinates must be strictly increasing")
    return edges


def _resample_profile_to_cfl(
    profile: DiscreteProfile,
    dt_s: float,
    fallback_span_s: float,
    maximum_bins: int,
) -> _CflTemporalGrid:
    """Conservatively rebin temporal energy onto the crystal CFL interval."""

    old_edges = _finite_volume_edges(profile, fallback_span_s)
    old_widths = np.diff(old_edges)
    old_span = float(old_edges[-1] - old_edges[0])
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("Multipass CFL time step must be positive")
    count = max(1, int(math.ceil(old_span / dt_s)))
    if count > int(maximum_bins):
        raise ValueError(
            "Multipass source grid needs "
            f"{count} CFL bins, above maximum_time_bins={maximum_bins}; "
            "reduce Nz, shorten the temporal support, or raise the limit and "
            "verify time/z convergence"
        )
    padded_span = count * float(dt_s)
    center = 0.5 * float(old_edges[0] + old_edges[-1])
    new_edges = center - 0.5 * padded_span + np.arange(count + 1) * dt_s
    new_weights = np.zeros(count, dtype=float)
    source = 0
    for target in range(count):
        left = float(new_edges[target])
        right = float(new_edges[target + 1])
        while source < len(profile.weights) and float(old_edges[source + 1]) <= left:
            source += 1
        candidate = source
        while candidate < len(profile.weights) and float(old_edges[candidate]) < right:
            overlap = max(
                0.0,
                min(right, float(old_edges[candidate + 1]))
                - max(left, float(old_edges[candidate])),
            )
            if overlap > 0.0:
                new_weights[target] += (
                    float(profile.weights[candidate])
                    * overlap
                    / float(old_widths[candidate])
                )
            candidate += 1
    new_weights = np.maximum(new_weights, 0.0)
    mass = float(np.sum(new_weights))
    if mass <= 0.0:
        raise ValueError("CFL temporal rebinning lost the entire Seed pulse")
    new_weights /= mass
    centers = 0.5 * (new_edges[:-1] + new_edges[1:])
    return _CflTemporalGrid(
        profile=DiscreteProfile(centers, new_weights),
        edges_s=new_edges,
        dt_s=float(dt_s),
        diagnostics={
            "input_temporal_bins": int(len(profile.coordinate)),
            "transport_source_bins": int(count),
            "input_temporal_support_s": old_span,
            "transport_source_support_s": padded_span,
            "cfl_time_step_s": float(dt_s),
            "finite_volume_mass_residual": float(np.sum(new_weights) - 1.0),
        },
    )



class MultipassCFLKernel(SpectralSignalPropagator):
    """Reusable time-grid and shared-reservoir operators for causal multipass."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._resolved_population_clock_s = 0.0

    def _prepare_entrance_plan(
        self,
        configured_energy_J: float,
        external_mode: str,
    ) -> tuple[_SpectralEntrancePlan, bool]:
        plan = self._spectral_entrance_plan
        if plan is not None:
            self._spectral_entrance_cache_hits += 1
            return plan, True

        spectral_fields: list[np.ndarray] = []
        outside_indices: list[float] = []
        source_energies: list[float] = []
        source_spatio_samples: list[dict[str, Any]] = []
        wavelength_source_configs: list[dict[str, Any]] = []
        source_warnings: list[str] = []
        for channel_index, (wavelength, weight) in enumerate(
            zip(
                self.spectrum.coordinate,
                self.spectrum.weights,
                strict=True,
            )
        ):
            local_cfg, sample_diagnostics = self._source_config_for_channel(
                channel_index, float(wavelength)
            )
            entrance_diagnostics: dict[str, Any] = {}
            external = build_external_field(
                local_cfg,
                configured_energy_J * float(weight),
                self.pre_elements,
                self.grid,
                external_mode,
                "energy_J",
                self.physics["interface_model"],
                False,
                include_abcd_tilt_phase=str(
                    self.physics["optical_propagation"]
                )
                in {"abcd", "hybrid_abcd_fft"},
                abcd_diagnostics=entrance_diagnostics,
            )
            if not entrance_diagnostics:
                entrance_diagnostics.update(
                    sampled_field_moments(external.field, self.grid)
                )
            sample_diagnostics["gain_entrance"] = entrance_diagnostics
            spectral_fields.append(external.field)
            outside_indices.append(float(external.refractive_index))
            source_energies.append(float(external.source_grid_quantity))
            source_spatio_samples.append(sample_diagnostics)
            wavelength_source_configs.append(local_cfg)
            source_warnings.extend(external.warnings)

        spectral_field = np.stack(spectral_fields, axis=0)
        reference_index = int(
            np.argmin(
                np.abs(
                    self.spectrum.coordinate
                    - float(self.cfg["wavelength_m"])
                )
            )
        )
        for index, sample in enumerate(source_spatio_samples):
            sample["gain_entrance"]["mode_overlap_with_reference"] = (
                normalized_field_overlap(
                    spectral_field[index],
                    spectral_field[reference_index],
                    self.grid,
                )
            )
        for array in (spectral_field,):
            array.setflags(write=False)
        plan = _SpectralEntrancePlan(
            spectral_field=spectral_field,
            outside_indices=np.asarray(outside_indices, dtype=float),
            source_energies=np.asarray(source_energies, dtype=float),
            source_spatio_samples=tuple(source_spatio_samples),
            wavelength_source_configs=tuple(wavelength_source_configs),
            warnings=tuple(source_warnings),
        )
        self._spectral_entrance_plan = plan
        self._spectral_entrance_cache_misses += 1
        return plan, False


    def _apply_all_cell_half_step(
        self,
        field: np.ndarray,
        marker: dict[str, Any],
        reverse: bool,
        distance_m: float,
        refractive_indices: np.ndarray,
    ) -> np.ndarray:
        if not np.any(field):
            return field
        if self.physics["optical_propagation"] not in {
            "abcd",
            "fft",
            "hybrid_abcd_fft",
        }:
            return field
        self._active_gain_marker = marker
        self._active_gain_reverse = reverse
        z_first = np.moveaxis(field, 1, 0)
        z_first = self._diffract(
            z_first,
            float(distance_m),
            refractive_indices,
        )
        result = np.moveaxis(z_first, 0, 1)
        if self.clip_crystal:
            before = float(
                np.sum(
                    self.grid.integrate_transverse(np.abs(result) ** 2)
                )
            )
            result *= self.crystal_mask[None, None, :, :]
            after = float(
                np.sum(
                    self.grid.integrate_transverse(np.abs(result) ** 2)
                )
            )
            loss = max(before - after, 0.0)
            self.crystal_aperture_loss_energy_J += loss
            direction = "backward" if reverse else "forward"
            ledger = getattr(
                self,
                "_counter_aperture_loss_by_direction",
                {"forward": 0.0, "backward": 0.0},
            )
            ledger[direction] += loss
            self._counter_aperture_loss_by_direction = ledger
        return result


    def _apply_all_cell_thermal_phase(
        self,
        field: np.ndarray,
        delta_n: np.ndarray,
        step_length_m: float,
    ) -> np.ndarray:
        if not np.any(field) or not np.any(delta_n):
            return field
        wavelengths = np.asarray(self.spectrum.coordinate, dtype=float)
        coefficient = 1j * 2.0 * math.pi * float(step_length_m)
        # One wavelength at a time bounds scratch memory at one 3-D crystal
        # volume while retaining the exact static phase screen.
        for wavelength_index, wavelength in enumerate(wavelengths):
            field[wavelength_index] *= np.exp(
                coefficient * delta_n / float(wavelength)
            )
        return field


    def _simultaneous_multipass_gain_step(
        self,
        fields: np.ndarray,
        population: np.ndarray,
        step_length_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply one shared local gain event to every active pass channel."""

        fields = np.asarray(fields, dtype=complex)
        if fields.ndim != 5 or fields.shape[0] < 1:
            raise ValueError(
                "simultaneous multipass fields must have shape "
                "[pass,wavelength,z,y,x]"
            )
        fluence = np.abs(fields) ** 2
        passive_loss = float(
            self.material.get("signal_passive_loss_m_inv", 0.0)
        )
        mask_3d = self.crystal_mask[None, :, :]
        passive_half = np.exp(
            -0.5 * passive_loss * float(step_length_m) * mask_3d
        )[None, None, :, :, :]
        active_input = fluence * passive_half
        output_active = np.zeros_like(fluence)
        wavelengths = np.asarray(self.spectrum.coordinate, dtype=float)
        photon_energies = photon_energy(wavelengths)
        spatial_shape = population.shape[1:]
        cross_section_sum = self.levels.signal_cross_section_sum_m2(
            wavelengths,
            spatial_shape,
        )
        saturation_cfg = self.levels_cfg.get("saturation_fluence_J_m2")
        if saturation_cfg is None:
            saturation_normal = np.divide(
                photon_energies.reshape((-1, 1, 1, 1)),
                cross_section_sum,
                out=np.full_like(cross_section_sum, np.inf),
                where=cross_section_sum > 0.0,
            )
        else:
            saturation_normal = np.full_like(
                cross_section_sum, float(saturation_cfg)
            )
        path_cosine = float(self.grid.dz) / float(step_length_m)
        saturation_projected = saturation_normal * path_cosine
        override = self.levels_cfg.get("gain_override_m_inv")
        if override is None:
            override_gain = None
        else:
            reservoir = getattr(self, "_override_gain_m_inv", None)
            if reservoir is None or reservoir.shape != spatial_shape:
                reservoir = np.full(spatial_shape, float(override), dtype=float)
                self._override_gain_m_inv = reservoir
            override_gain = reservoir

        if not bool(self.physics["gain_saturation"]):
            if override_gain is None:
                gain = self.levels.signal_gain_spectrum_m_inv(
                    population, wavelengths
                )
            else:
                gain = np.broadcast_to(
                    override_gain[None, ...],
                    active_input.shape[1:],
                )
            exponent = gain * mask_3d[None, ...] * float(step_length_m)
            self._maximum_unsaturated_gain_exponent = max(
                self._maximum_unsaturated_gain_exponent,
                float(np.max(exponent)),
            )
            self._positive_gain_exponent_clipping_active |= bool(
                np.any(exponent > 50.0)
            )
            output_active = active_input * np.exp(
                np.minimum(exponent[None, ...], 50.0)
            )
        else:
            channel_groups = self._simultaneous_gain_channel_groups()
            for spectral_order in (
                range(len(channel_groups)),
                range(len(channel_groups) - 1, -1, -1),
            ):
                for group_index in spectral_order:
                    channels = channel_groups[group_index]
                    representative = int(channels[0])
                    contributor_input = 0.5 * active_input[:, channels]
                    packet_in = np.sum(
                        contributor_input, axis=(0, 1)
                    )
                    if override_gain is None:
                        active_gain = self.levels.signal_gain_m_inv(
                            population,
                            wavelength_m=float(wavelengths[representative]),
                        )
                    else:
                        active_gain = override_gain
                    exponent = (
                        active_gain
                        * mask_3d
                        * float(step_length_m)
                    )
                    valid = (
                        np.isfinite(saturation_projected[representative])
                        & (saturation_projected[representative] > 0.0)
                    )
                    fallback = np.where(valid, -np.inf, exponent)
                    self._maximum_unsaturated_gain_exponent = max(
                        self._maximum_unsaturated_gain_exponent,
                        float(np.max(fallback)),
                    )
                    self._positive_gain_exponent_clipping_active |= bool(
                        np.any(fallback > 50.0)
                    )
                    packet_out = packet_in * np.exp(
                        np.minimum(exponent, 50.0)
                    )
                    if np.any(valid):
                        packet_out = np.where(
                            valid,
                            frantz_nodvik(
                                packet_in,
                                exponent,
                                np.where(
                                    valid,
                                    saturation_projected[representative],
                                    1.0,
                                ),
                            ),
                            packet_out,
                        )
                    if override_gain is None and np.any(
                        cross_section_sum[representative] > 0.0
                    ):
                        requested_events = (packet_out - packet_in) / (
                            photon_energies[representative]
                            * float(self.grid.dz)
                        )
                        population[:], event_stats = (
                            self.levels.apply_signal_photon_events(
                                population,
                                requested_events,
                                float(wavelengths[representative]),
                                input_photon_density_m3=(
                                    packet_in
                                    / (
                                        photon_energies[representative]
                                        * float(self.grid.dz)
                                    )
                                ),
                                interaction_length_m=float(step_length_m),
                            )
                        )
                        applied = np.asarray(
                            event_stats["applied_photon_events_m3"],
                            dtype=float,
                        )
                        packet_out = packet_in + applied * (
                            photon_energies[representative]
                            * float(self.grid.dz)
                        )
                    elif override_gain is not None and np.any(valid):
                        depletion = np.divide(
                            packet_out - packet_in,
                            saturation_normal[representative]
                            * float(self.grid.dz),
                            out=np.zeros_like(packet_out),
                            where=valid,
                        )
                        override_gain -= depletion
                        if float(override) >= 0.0:
                            np.maximum(override_gain, 0.0, out=override_gain)
                        else:
                            np.minimum(override_gain, 0.0, out=override_gain)
                    packet_scale = np.divide(
                        np.maximum(packet_out, 0.0),
                        packet_in,
                        out=np.zeros_like(packet_out),
                        where=packet_in > np.finfo(float).tiny,
                    )
                    output_active[:, channels] += (
                        contributor_input * packet_scale[None, None, ...]
                    )

        output_fluence = np.maximum(output_active * passive_half, 0.0)
        passive_density = np.sum(
            np.maximum(fluence - active_input, 0.0)
            + np.maximum(output_active - output_fluence, 0.0),
            axis=(0, 1),
        ) / max(float(self.grid.dz), np.finfo(float).tiny)
        self.passive_loss_energy_J_m3 += passive_density
        active_transfer = (
            np.sum(output_fluence - fluence, axis=(0, 1))
            / float(self.grid.dz)
            + passive_density
        )
        scale = np.sqrt(
            output_fluence
            / np.maximum(fluence, np.finfo(float).tiny)
        )
        fields *= np.where(fluence > 0.0, scale, 0.0)
        if self.clip_crystal:
            fields *= self.crystal_mask[None, None, None, :, :]
        return fields, active_transfer


    def _dark_relaxation_block_steps(
        self,
        total_duration_s: float,
        dt_s: float,
    ) -> int:
        configured = float(
            self.multipass_cfg.get("maximum_dark_step_s", 0.0)
        )
        if configured > 0.0:
            maximum_step = configured
        elif self.levels.nonlinear:
            # Nonlinear dark kinetics do not expose one state-independent
            # lifetime from which a safe block size can be inferred.  Resolve
            # them on every optical CFL slab unless the user supplies and
            # converges an explicit maximum_dark_step_s.
            maximum_step = dt_s
        else:
            maximum_outflow_rate_s = (
                self.levels.maximum_dark_linear_outflow_rate_s(
                    self.grid.shape
                )
            )
            fastest_lifetime = (
                math.inf
                if maximum_outflow_rate_s <= 0.0
                else 1.0 / maximum_outflow_rate_s
            )
            maximum_step = min(
                total_duration_s,
                0.01 * fastest_lifetime,
            )
        return max(1, int(math.floor(maximum_step / dt_s)))


    @staticmethod
    def _empty_trace():
        from .optics import OpticalTrace

        return OpticalTrace()

