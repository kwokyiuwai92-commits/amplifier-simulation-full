"""Global-clock, spatially local transport for a finite multipass sequence."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from .constants import LIGHT_SPEED
from .multipass_cfl import (
    MultipassCFLKernel,
    _resample_profile_to_cfl,
)
from .dispersion import material_group_index, material_refractive_index
from .flight_time import FlightTime, optical_train_flight_time
from .multipass import CausalMultipassPlan, compile_multipass_plan
from .optical_system import (
    additive_source_gain_angle_warning,
    gain_interface,
    gain_medium_axis_angles_rad,
    gain_medium_path_cosine,
)
from .signal import (
    SignalPropagationResult,
    append_gain_entrance_loss_terms,
    gain_entrance_energy_partition,
    reference_ray_warnings,
    report_signal_transport_progress,
)
from .seed_spatio_spectral import sampled_field_moments
from .spectral_signal import SpectralSignalPropagator


class CausalMultipassSpectralSignalPropagator(
    MultipassCFLKernel
):
    """Advance every hit of one injected pulse on one absolute-time grid."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.multipass_cfg = self.simulation.get("multipass", {})
        # Shared helpers inherited from the original two-channel solver read
        # this alias for dark-step and time-grid controls.
        self.counter_cfg = self.multipass_cfg
        self._multipass_diagnostics: dict[str, Any] = {}
        self._compiled_multipass_plan: CausalMultipassPlan | None = None

    def _model_used(self) -> str:
        return "spectral_time_causal_multipass"

    def _compile_plan(self) -> CausalMultipassPlan:
        if self._compiled_multipass_plan is None:
            self._compiled_multipass_plan = compile_multipass_plan(
                self.multipass_cfg,
                self.gain_marker,
                crystal_length_m=float(len(self.grid.z) * self.grid.dz),
                crystal_refractive_index=material_refractive_index(
                    self.material,
                    float(self.cfg["wavelength_m"]),
                    role="signal",
                ),
                global_extra_delay_s=float(
                    self.simulation.get(
                        "unmodeled_between_pass_delay_s", 0.0
                    )
                ),
            )
        return self._compiled_multipass_plan

    def _transition_flight(
        self, transition_index: int, plan: CausalMultipassPlan
    ) -> FlightTime:
        transition = plan.transitions[transition_index]
        flight = optical_train_flight_time(
            list(transition.route),
            initial_refractive_index=transition.initial_refractive_index,
            initial_group_refractive_index=(
                transition.initial_group_refractive_index
            ),
            extra_delay_s=transition.extra_delay_s,
        )
        timing_length_m = float(transition.unpropagated_geometric_length_m)
        if timing_length_m == 0.0:
            return flight
        phase_path_increment = (
            transition.initial_refractive_index * timing_length_m
        )
        group_path_increment = (
            transition.initial_group_refractive_index * timing_length_m
        )
        return FlightTime(
            delay_s=flight.delay_s + group_path_increment / LIGHT_SPEED,
            geometric_path_length_m=(
                flight.geometric_path_length_m + timing_length_m
            ),
            phase_optical_path_m=(
                flight.phase_optical_path_m + phase_path_increment
            ),
            geometric_optical_path_m=(
                flight.phase_optical_path_m + phase_path_increment
            ),
            group_optical_path_m=(
                flight.group_optical_path_m + group_path_increment
            ),
            explicit_group_delay_s=flight.explicit_group_delay_s,
            final_refractive_index=flight.final_refractive_index,
            final_group_refractive_index=flight.final_group_refractive_index,
        )

    def _timing_contract(
        self,
        plan: CausalMultipassPlan,
    ) -> tuple[float, float, list[FlightTime], list[int], Any]:
        path_cosines = np.asarray(
            [gain_medium_path_cosine(hit.marker) for hit in plan.hits],
            dtype=float,
        )
        mean_path_cosine = float(np.mean(path_cosines))
        step_length_m = float(self.grid.dz) / mean_path_cosine
        group_index = material_group_index(
            self.material,
            float(self.cfg["wavelength_m"]),
            role="signal",
        )
        dt_s = group_index * step_length_m / LIGHT_SPEED
        fallback_span = (
            2.0
            * max(
                1.0,
                float(self.cfg.get("temporal", {}).get("span_fwhm", 3.0)),
            )
            * float(self.cfg["pulse_duration_s"])
        )
        maximum_bins = int(
            self.multipass_cfg.get("maximum_time_bins", 20000)
        )
        temporal_grid = _resample_profile_to_cfl(
            self.temporal,
            dt_s,
            fallback_span,
            maximum_bins,
        )
        flights = [
            self._transition_flight(index, plan)
            for index in range(len(plan.transitions))
        ]
        delay_steps = [
            max(0, int(round(flight.delay_s / dt_s)))
            for flight in flights
        ]
        return step_length_m, dt_s, flights, delay_steps, temporal_grid

    def resolved_seed_population_time_s(self) -> float:
        if not bool(
            self.multipass_cfg.get("include_dark_relaxation", True)
        ) or self.levels_cfg.get("gain_override_m_inv") is not None:
            return 0.0
        if self._resolved_population_clock_s > 0.0:
            return float(self._resolved_population_clock_s)
        plan = self._compile_plan()
        _step_length, dt_s, _flights, delays, temporal_grid = (
            self._timing_contract(plan)
        )
        total_bins = (
            len(temporal_grid.profile.coordinate)
            + len(plan.hits) * len(self.grid.z)
            + sum(delays)
        )
        maximum_bins = int(
            self.multipass_cfg.get("maximum_time_bins", 20000)
        )
        if total_bins > maximum_bins:
            raise ValueError(
                "Causal multipass sequence exceeds "
                "simulation.multipass.maximum_time_bins"
            )
        return float((total_bins - 1) * dt_s)

    def _gain_transport_diagnostics(self) -> dict[str, Any]:
        diagnostics = SpectralSignalPropagator._gain_transport_diagnostics(
            self
        )
        diagnostics.update(
            {
                "backend": "causal_multipass_cfl_frantz_nodvik",
                "spectral_coupling": (
                    "shared_reservoir_with_simultaneous_pass_channels"
                ),
                "include_relaxation_during_pulse": bool(
                    self._multipass_diagnostics.get(
                        "dark_relaxation_included", False
                    )
                ),
                "multipass": copy.deepcopy(self._multipass_diagnostics),
            }
        )
        return diagnostics

    def _align_packet_center(
        self,
        packet: np.ndarray,
        target_x_m: float,
        target_y_m: float,
    ) -> tuple[np.ndarray, float, float]:
        intensity = np.sum(np.abs(packet) ** 2, axis=0)
        mass = float(np.sum(intensity))
        if mass <= np.finfo(float).tiny:
            return packet, 0.0, 0.0
        center_x = float(
            np.sum(intensity * self.grid.x[None, :]) / mass
        )
        center_y = float(
            np.sum(intensity * self.grid.y[:, None]) / mass
        )
        shift_x = float(target_x_m) - center_x
        shift_y = float(target_y_m) - center_y
        if shift_x == 0.0 and shift_y == 0.0:
            return packet, 0.0, 0.0
        window_x = len(self.grid.x) * float(self.grid.dx)
        window_y = len(self.grid.y) * float(self.grid.dy)
        if abs(shift_x) >= 2.0 * window_x or abs(shift_y) >= 2.0 * window_y:
            return np.zeros_like(packet), shift_x, shift_y
        # A direct Fourier shift on the simulation window is periodic and can
        # wrap a Herriott spot from one edge to the opposite edge.  Pad by at
        # least one full window plus the requested displacement, shift on that
        # larger plane, then crop.  Energy that truly leaves the sampled plane
        # is therefore lost to the named route/window ledger instead of being
        # teleported across the crystal.
        pad_x = len(self.grid.x) + int(
            math.ceil(abs(shift_x) / float(self.grid.dx))
        )
        pad_y = len(self.grid.y) + int(
            math.ceil(abs(shift_y) / float(self.grid.dy))
        )
        padded = np.pad(
            packet,
            ((0, 0), (pad_y, pad_y), (pad_x, pad_x)),
            mode="constant",
        )
        fx = np.fft.fftfreq(padded.shape[-1], d=float(self.grid.dx))
        fy = np.fft.fftfreq(padded.shape[-2], d=float(self.grid.dy))
        fxx, fyy = np.meshgrid(fx, fy, indexing="xy")
        translation = np.exp(
            -2j * math.pi * (fxx * shift_x + fyy * shift_y)
        )
        shifted = np.fft.ifft2(
            np.fft.fft2(padded, axes=(-2, -1))
            * translation[None, ...],
            axes=(-2, -1),
        )
        aligned = shifted[
            :,
            pad_y : pad_y + len(self.grid.y),
            pad_x : pad_x + len(self.grid.x),
        ].copy()
        return aligned, shift_x, shift_y

    @staticmethod
    def _entry_side(direction: str) -> str:
        return "front" if direction == "forward" else "back"

    @staticmethod
    def _exit_side(direction: str) -> str:
        return "back" if direction == "forward" else "front"

    @staticmethod
    def _entry_z(direction: str) -> int:
        return 0 if direction == "forward" else -1

    @staticmethod
    def _exit_z(direction: str) -> int:
        return -1 if direction == "forward" else 0

    def _apply_gain_interface_train(
        self,
        packet: np.ndarray,
        indices_before: np.ndarray,
        indices_after: np.ndarray,
        marker: dict[str, Any],
        side: str,
        *,
        entering: bool,
    ) -> np.ndarray:
        result = packet.copy()
        for wavelength_index, wavelength in enumerate(
            self.spectrum.coordinate
        ):
            result[wavelength_index] = gain_interface(
                result[wavelength_index],
                float(indices_before[wavelength_index]),
                float(indices_after[wavelength_index]),
                marker,
                side,
                self.physics["interface_model"],
                float(wavelength),
                entering=entering,
                surface_sag_m=self._endface_surface_sag_m[side],
            )
        return result

    def _clip_entry(
        self,
        packet: np.ndarray,
        pass_index: int,
        aperture_loss_by_pass: np.ndarray,
    ) -> np.ndarray:
        if not self.clip_crystal:
            return packet
        before = self._integral(packet[None, ...])
        result = packet * self.crystal_mask[None, :, :]
        after = self._integral(result[None, ...])
        loss = max(before - after, 0.0)
        self.crystal_aperture_loss_energy_J += loss
        aperture_loss_by_pass[pass_index] += loss
        return result

    def _apply_pass_half_step(
        self,
        field: np.ndarray,
        hit: Any,
        pass_index: int,
        distance_m: float,
        refractive_indices: np.ndarray,
        aperture_loss_by_pass: np.ndarray,
    ) -> np.ndarray:
        before = float(self.crystal_aperture_loss_energy_J)
        result = self._apply_all_cell_half_step(
            field,
            hit.marker,
            hit.direction == "backward",
            distance_m,
            refractive_indices,
        )
        aperture_loss_by_pass[pass_index] += max(
            float(self.crystal_aperture_loss_energy_J) - before, 0.0
        )
        return result

    def propagate(
        self,
        population: np.ndarray,
        delta_n: np.ndarray,
    ) -> SignalPropagationResult:
        self._reset_thermal_phase_runtime()
        self._diffraction_transfer_cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "uncached_oversize_screens": 0,
        }
        self.levels.record_spectral_support(
            "signal", self.spectrum.coordinate, self.spectrum.weights
        )
        self.passive_loss_energy_J_m3 = np.zeros(
            self.grid.shape, dtype=float
        )
        self.crystal_aperture_loss_energy_J = 0.0
        self._counter_aperture_loss_by_direction = {
            "forward": 0.0,
            "backward": 0.0,
        }
        self._positive_gain_exponent_clipping_active = False
        self._maximum_unsaturated_gain_exponent = -np.inf
        # A propagate() call owns one injected pulse and all of its passes.
        # Share depletion within that sequence, but do not carry a manual
        # reservoir into an independent solve or an outer fixed-point probe.
        override = self.levels_cfg.get("gain_override_m_inv")
        self._override_gain_m_inv = (
            None
            if override is None
            else np.full(self.grid.shape, float(override), dtype=float)
        )
        self._resolved_population_clock_s = 0.0
        self._multipass_diagnostics = {}

        plan = self._compile_plan()
        pass_count = len(plan.hits)
        configured_energy = float(self.cfg["pulse_energy_J"])
        wavelengths = np.asarray(self.spectrum.coordinate, dtype=float)
        propagation_mode = str(self.physics["optical_propagation"])
        external_mode = (
            "abcd"
            if propagation_mode in {"abcd", "hybrid_abcd_fft"}
            else propagation_mode
        )
        entrance_plan, entrance_plan_reused = self._prepare_entrance_plan(
            configured_energy, external_mode
        )
        self._wavelength_source_configs = (
            entrance_plan.wavelength_source_configs
        )
        source_samples = copy.deepcopy(
            list(entrance_plan.source_spatio_samples)
        )
        warnings = [
            *self.bench_warnings,
            *self.source_definition_warnings,
            *entrance_plan.warnings,
            *plan.warnings,
            *reference_ray_warnings(
                self.gain_marker,
                self.gain_markers,
                self.post_elements,
                self.between_pass_elements,
                self.between_pass_segments,
            ),
        ]
        angle_warning = additive_source_gain_angle_warning(
            self.cfg, [hit.marker for hit in plan.hits], "Seed"
        )
        if angle_warning is not None:
            warnings.append(angle_warning)

        step_length, dt_s, flights, delay_steps, temporal_grid = (
            self._timing_contract(plan)
        )
        source_weights = temporal_grid.profile.weights
        path_cosines = np.asarray(
            [gain_medium_path_cosine(hit.marker) for hit in plan.hits],
            dtype=float,
        )
        mean_path_cosine = float(np.mean(path_cosines))
        relative_path_span = float(
            np.ptp(path_cosines)
            / max(abs(mean_path_cosine), np.finfo(float).tiny)
        )
        group_indices = np.asarray(
            [
                material_group_index(
                    self.material, float(wavelength), role="signal"
                )
                for wavelength in wavelengths
            ],
            dtype=float,
        )
        reference_index = int(
            np.argmin(
                np.abs(wavelengths - float(self.cfg["wavelength_m"]))
            )
        )
        group_index = float(group_indices[reference_index])
        relative_group_span = float(
            np.ptp(group_indices)
            / max(abs(float(np.mean(group_indices))), np.finfo(float).tiny)
        )
        maximum_group_span = float(
            self.multipass_cfg.get(
                "maximum_relative_group_index_span", 1e-3
            )
        )
        if relative_group_span > maximum_group_span:
            raise ValueError(
                "Causal multipass uses one shared group-velocity CFL grid, "
                f"but the spectral group-index span is {relative_group_span:.6g}; "
                "narrow the spectrum or explicitly converge a larger tolerance"
            )

        total_bins = (
            len(source_weights)
            + pass_count * len(self.grid.z)
            + sum(delay_steps)
        )
        maximum_bins = int(
            self.multipass_cfg.get("maximum_time_bins", 20000)
        )
        if total_bins > maximum_bins:
            raise ValueError(
                "Causal multipass sequence needs "
                f"{total_bins} absolute-time bins, above "
                f"maximum_time_bins={maximum_bins}"
            )
        total_steps = total_bins - 1
        global_edges = temporal_grid.edges_s[0] + np.arange(
            total_bins + 1
        ) * dt_s
        global_time = 0.5 * (global_edges[:-1] + global_edges[1:])
        source_total_J = float(np.sum(entrance_plan.source_energies))
        input_temporal = np.zeros(total_bins, dtype=float)
        input_temporal[: len(source_weights)] = (
            source_total_J * source_weights
        )
        output_temporal = np.zeros(total_bins, dtype=float)
        output_spectrum = np.zeros(len(wavelengths), dtype=float)
        output_intensity = np.zeros(
            (len(wavelengths), len(self.grid.y), len(self.grid.x)),
            dtype=float,
        )

        n_gain = np.asarray(
            [
                material_refractive_index(
                    self.material, float(wavelength), role="signal"
                )
                for wavelength in wavelengths
            ],
            dtype=float,
        )
        first_boundary, first_shift_x, first_shift_y = (
            self._align_packet_center(
                entrance_plan.spectral_field.copy(),
                plan.hits[0].center_x_m,
                plan.hits[0].center_y_m,
            )
        )
        source_gain_entrance_J = self._integral(
            first_boundary[None, ...]
        )
        outside_source = entrance_plan.outside_indices.copy()
        first_boundary = self._apply_gain_interface_train(
            first_boundary,
            outside_source,
            n_gain,
            plan.hits[0].marker,
            self._entry_side(plan.hits[0].direction),
            entering=True,
        )
        first_entry_interface_output_J = self._integral(
            first_boundary[None, ...]
        )

        aperture_loss_by_pass = np.zeros(pass_count, dtype=float)
        first_boundary = self._clip_entry(
            first_boundary, 0, aperture_loss_by_pass
        )
        gain_input_J = np.zeros(pass_count, dtype=float)
        gain_output_J = np.zeros(pass_count, dtype=float)
        crystal_exit_J = np.zeros(pass_count, dtype=float)
        entry_interface_input_J = np.zeros(pass_count, dtype=float)
        entry_interface_output_J = np.zeros(pass_count, dtype=float)
        exit_interface_input_J = np.zeros(pass_count, dtype=float)
        transition_launch_J = np.zeros(pass_count - 1, dtype=float)
        transition_route_output_J = np.zeros(pass_count - 1, dtype=float)
        transition_alignment_shift_x = np.zeros(pass_count - 1, dtype=float)
        transition_alignment_shift_y = np.zeros(pass_count - 1, dtype=float)
        entry_interface_input_J[0] = source_gain_entrance_J
        entry_interface_output_J[0] = first_entry_interface_output_J
        gain_input_J[0] = self._integral(first_boundary[None, ...])

        field_shape = (
            pass_count,
            len(wavelengths),
            len(self.grid.z),
            len(self.grid.y),
            len(self.grid.x),
        )
        predicted_working_bytes = int(
            8 * np.prod(field_shape, dtype=np.int64) * np.dtype(complex).itemsize
            + pass_count * np.prod(self.grid.shape) * np.dtype(float).itemsize
            + 2
            * pass_count
            * len(wavelengths)
            * len(self.grid.y)
            * len(self.grid.x)
            * np.dtype(float).itemsize
        )
        maximum_memory_bytes = int(
            float(
                self.multipass_cfg.get(
                    "maximum_field_memory_MB", 1024.0
                )
            )
            * 1024**2
        )
        if predicted_working_bytes > maximum_memory_bytes:
            raise ValueError(
                "Causal multipass active-field estimate is "
                f"{predicted_working_bytes / 1024**2:.3g} MiB, above "
                "simulation.multipass.maximum_field_memory_MB; reduce passes, "
                "spectral/grid samples, or raise the explicit memory budget"
            )

        fields = np.zeros(field_shape, dtype=complex)
        scheduled_entries: list[dict[int, np.ndarray]] = [
            {} for _ in range(pass_count)
        ]
        pass_volumes = np.zeros(
            (pass_count, *self.grid.shape), dtype=float
        )
        entry_intensity_by_pass = np.zeros(
            (
                pass_count,
                len(wavelengths),
                len(self.grid.y),
                len(self.grid.x),
            ),
            dtype=float,
        )
        exit_intensity_by_pass = np.zeros_like(entry_intensity_by_pass)
        extracted_energy = np.zeros(self.grid.shape, dtype=float)
        time_energy_by_pass_z = np.zeros(
            (pass_count, len(self.grid.z)), dtype=float
        )
        pair_common_energy_by_z = np.zeros(
            (pass_count, pass_count, len(self.grid.z)), dtype=float
        )
        overlap_bins_by_z = np.zeros(len(self.grid.z), dtype=int)
        maximum_simultaneous_passes = 0
        maximum_pair_common_energy_J = 0.0
        post_input_J = 0.0
        post_output_J = 0.0

        include_dark = bool(
            self.multipass_cfg.get("include_dark_relaxation", True)
        ) and self.levels_cfg.get("gain_override_m_inv") is None
        duration_s = total_steps * dt_s
        dark_block_steps = self._dark_relaxation_block_steps(
            duration_s, dt_s
        )
        dark_updates = 0
        current_dark_half_s = 0.0

        report_signal_transport_progress(
            self,
            current=0,
            total=total_steps,
            pass_index=1,
            pass_total=pass_count,
            z_index=None,
            phase="causal_multipass_transport",
            fraction_start=0.12,
            fraction_stop=0.94,
            metrics={
                "cfl_time_step_s": dt_s,
                "absolute_time_bins": total_bins,
                "compiled_passes": pass_count,
            },
        )

        # The new contract has one unambiguous common loss field under
        # simulation.multipass.  The legacy simulation-level value remains for
        # sequential solvers and is intentionally not multiplied a second time.
        common_configured_transmission = 1.0
        for step in range(total_steps):
            block_start = step % dark_block_steps == 0
            block_stop = min(
                total_steps,
                step - step % dark_block_steps + dark_block_steps,
            )
            if include_dark and block_start:
                current_dark_half_s = 0.5 * (block_stop - step) * dt_s
                population[:] = self.levels.advance_dark_implicit(
                    population, current_dark_half_s
                )
                dark_updates += 1

            if step < len(source_weights):
                entry_z = self._entry_z(plan.hits[0].direction)
                source_packet = (
                    math.sqrt(float(source_weights[step]))
                    * first_boundary
                )
                fields[0, :, entry_z] += source_packet
                entry_intensity_by_pass[0] += np.abs(source_packet) ** 2
            for pass_index in range(1, pass_count):
                packet = scheduled_entries[pass_index].pop(step, None)
                if packet is None:
                    continue
                entry_z = self._entry_z(plan.hits[pass_index].direction)
                fields[pass_index, :, entry_z] += packet
                entry_intensity_by_pass[pass_index] += np.abs(packet) ** 2

            local_fluence = np.sum(np.abs(fields) ** 2, axis=1)
            active_by_pass_z = np.sum(local_fluence, axis=(-2, -1))
            active_count_by_z = np.count_nonzero(
                active_by_pass_z > np.finfo(float).tiny, axis=0
            )
            overlap_bins_by_z += (active_count_by_z >= 2).astype(int)
            maximum_simultaneous_passes = max(
                maximum_simultaneous_passes,
                int(np.max(active_count_by_z)),
            )
            for pass_index in range(pass_count):
                pass_energy_z = self.grid.integrate_transverse(
                    local_fluence[pass_index]
                )
                time_energy_by_pass_z[pass_index] += pass_energy_z
                for other_index in range(pass_index + 1, pass_count):
                    common_energy_z = self.grid.integrate_transverse(
                        np.minimum(
                            local_fluence[pass_index],
                            local_fluence[other_index],
                        )
                    )
                    pair_common_energy_by_z[
                        pass_index, other_index
                    ] += common_energy_z
                    pair_common_energy_by_z[
                        other_index, pass_index
                    ] += common_energy_z
                    maximum_pair_common_energy_J = max(
                        maximum_pair_common_energy_J,
                        float(np.max(common_energy_z)),
                    )

            for pass_index, hit in enumerate(plan.hits):
                fields[pass_index] = self._apply_pass_half_step(
                    fields[pass_index],
                    hit,
                    pass_index,
                    step_length / 2.0,
                    n_gain,
                    aperture_loss_by_pass,
                )
                fields[pass_index] = self._apply_all_cell_thermal_phase(
                    fields[pass_index], delta_n, step_length
                )
            fields, local_transfer = (
                self._simultaneous_multipass_gain_step(
                    fields, population, step_length
                )
            )
            extracted_energy += local_transfer
            for pass_index, hit in enumerate(plan.hits):
                fields[pass_index] = self._apply_pass_half_step(
                    fields[pass_index],
                    hit,
                    pass_index,
                    step_length / 2.0,
                    n_gain,
                    aperture_loss_by_pass,
                )
                pass_volumes[pass_index] += np.sum(
                    np.abs(fields[pass_index]) ** 2, axis=0
                )

            for pass_index, hit in enumerate(plan.hits):
                exit_z = self._exit_z(hit.direction)
                packet = fields[pass_index, :, exit_z].copy()
                packet_gain_output = self._integral(packet[None, ...])
                gain_output_J[pass_index] += packet_gain_output
                exit_interface_input_J[pass_index] += packet_gain_output
                if pass_index + 1 < pass_count:
                    transition = plan.transitions[pass_index]
                    outside_after = np.full_like(
                        wavelengths,
                        transition.initial_refractive_index,
                    )
                else:
                    configured_outside = hit.marker.get(
                        "outside_refractive_index"
                    )
                    outside_after = (
                        entrance_plan.outside_indices.copy()
                        if configured_outside is None
                        else np.full_like(
                            wavelengths, float(configured_outside)
                        )
                    )
                packet = self._apply_gain_interface_train(
                    packet,
                    n_gain,
                    outside_after,
                    hit.marker,
                    self._exit_side(hit.direction),
                    entering=False,
                )
                packet_crystal_exit = self._integral(packet[None, ...])
                crystal_exit_J[pass_index] += packet_crystal_exit
                exit_intensity_by_pass[pass_index] += np.abs(packet) ** 2
                if not np.any(packet):
                    continue

                if pass_index + 1 < pass_count:
                    transition = plan.transitions[pass_index]
                    launch = packet * math.sqrt(
                        common_configured_transmission
                        * transition.power_transmission
                    )
                    transition_launch_J[pass_index] += self._integral(
                        launch[None, ...]
                    )
                    routed, routed_indices = self._propagate_train(
                        launch[None, ...],
                        list(transition.route),
                        outside_after,
                    )
                    routed_packet = routed[0]
                    next_hit = plan.hits[pass_index + 1]
                    (
                        routed_packet,
                        alignment_x,
                        alignment_y,
                    ) = self._align_packet_center(
                        routed_packet,
                        next_hit.center_x_m,
                        next_hit.center_y_m,
                    )
                    transition_alignment_shift_x[pass_index] = alignment_x
                    transition_alignment_shift_y[pass_index] = alignment_y
                    routed_energy = self._integral(
                        routed_packet[None, ...]
                    )
                    transition_route_output_J[pass_index] += routed_energy
                    entry_interface_input_J[pass_index + 1] += routed_energy
                    routed_packet = self._apply_gain_interface_train(
                        routed_packet,
                        routed_indices,
                        n_gain,
                        next_hit.marker,
                        self._entry_side(next_hit.direction),
                        entering=True,
                    )
                    entry_interface_output_J[
                        pass_index + 1
                    ] += self._integral(routed_packet[None, ...])
                    routed_packet = self._clip_entry(
                        routed_packet,
                        pass_index + 1,
                        aperture_loss_by_pass,
                    )
                    gain_input_J[pass_index + 1] += self._integral(
                        routed_packet[None, ...]
                    )
                    arrival_step = (
                        step + 1 + delay_steps[pass_index]
                    )
                    if arrival_step < total_bins:
                        if (
                            arrival_step
                            in scheduled_entries[pass_index + 1]
                        ):
                            raise RuntimeError(
                                "Causal multipass delay quantization mapped two "
                                "coherent source packets into one entry slot; "
                                "refine the z/time grid"
                            )
                        scheduled_entries[pass_index + 1][
                            arrival_step
                        ] = routed_packet
                else:
                    post_input_J += packet_crystal_exit
                    post_field, _post_indices = self._propagate_train(
                        packet[None, ...],
                        self.post_elements,
                        outside_after,
                    )
                    output_packet = post_field[0]
                    packet_spectrum = self.grid.integrate_transverse(
                        np.abs(output_packet) ** 2
                    )
                    output_spectrum += packet_spectrum
                    output_index = step + 1
                    output_temporal[output_index] += float(
                        np.sum(packet_spectrum)
                    )
                    output_intensity += np.abs(output_packet) ** 2
                    post_output_J += float(np.sum(packet_spectrum))

            next_fields = np.zeros_like(fields)
            for pass_index, hit in enumerate(plan.hits):
                if hit.direction == "forward":
                    next_fields[pass_index, :, 1:] = fields[
                        pass_index, :, :-1
                    ]
                else:
                    next_fields[pass_index, :, :-1] = fields[
                        pass_index, :, 1:
                    ]
            fields = next_fields

            if include_dark and step + 1 == block_stop:
                population[:] = self.levels.advance_dark_implicit(
                    population, current_dark_half_s
                )
                dark_updates += 1

            active_passes = np.flatnonzero(
                np.sum(active_by_pass_z, axis=1) > np.finfo(float).tiny
            )
            report_signal_transport_progress(
                self,
                current=step + 1,
                total=total_steps,
                pass_index=(
                    int(active_passes[-1] + 1)
                    if len(active_passes)
                    else 1
                ),
                pass_total=pass_count,
                z_index=None,
                phase="causal_multipass_transport",
                fraction_start=0.12,
                fraction_stop=0.94,
                metrics={
                    "absolute_time_s": float(global_edges[step + 1]),
                    "simultaneous_pass_channels": int(
                        np.max(active_count_by_z)
                    ),
                    "overlap_z_cells": int(
                        np.count_nonzero(active_count_by_z >= 2)
                    ),
                },
            )

        remaining_packets = sum(
            len(schedule) for schedule in scheduled_entries
        )
        if remaining_packets:
            raise RuntimeError(
                "Causal multipass time window ended before all scheduled "
                "packets re-entered the crystal"
            )
        self._resolved_population_clock_s = (
            duration_s if include_dark else 0.0
        )

        gain_partition = gain_entrance_energy_partition(
            configured_source_energy_J=configured_energy,
            sampled_gain_entrance_quantity=source_gain_entrance_J,
            wavelength_samples=source_samples,
        )
        external_loss_ledger: list[dict[str, Any]] = []
        if gain_partition is None:
            external_loss_ledger.append(
                {
                    "name": "source_to_gain_optical_train",
                    "category": "pre_gain_routing",
                    "input_energy_J": source_total_J,
                    "output_energy_J": source_gain_entrance_J,
                    "signed_loss_energy_J": source_total_J
                    - source_gain_entrance_J,
                    "included_in_seed_balance": True,
                }
            )
        else:
            append_gain_entrance_loss_terms(
                external_loss_ledger, gain_partition
            )
        for pass_index in range(pass_count):
            entry_side = self._entry_side(plan.hits[pass_index].direction)
            exit_side = self._exit_side(plan.hits[pass_index].direction)
            external_loss_ledger.append(
                {
                    "name": f"pass_{pass_index + 1}_{entry_side}_gain_interface",
                    "category": "gain_interface",
                    "input_energy_J": float(
                        entry_interface_input_J[pass_index]
                    ),
                    "output_energy_J": float(
                        entry_interface_output_J[pass_index]
                    ),
                    "signed_loss_energy_J": float(
                        entry_interface_input_J[pass_index]
                        - entry_interface_output_J[pass_index]
                    ),
                    "included_in_seed_balance": True,
                }
            )
            external_loss_ledger.append(
                {
                    "name": f"pass_{pass_index + 1}_{exit_side}_gain_interface",
                    "category": "gain_interface",
                    "input_energy_J": float(
                        exit_interface_input_J[pass_index]
                    ),
                    "output_energy_J": float(crystal_exit_J[pass_index]),
                    "signed_loss_energy_J": float(
                        exit_interface_input_J[pass_index]
                        - crystal_exit_J[pass_index]
                    ),
                    "included_in_seed_balance": True,
                }
            )
            if pass_index + 1 < pass_count:
                external_loss_ledger.extend(
                    [
                        {
                            "name": f"between_pass_{pass_index + 1}_{pass_index + 2}_configured_transmission",
                            "category": "between_pass_transmission",
                            "input_energy_J": float(
                                crystal_exit_J[pass_index]
                            ),
                            "output_energy_J": float(
                                transition_launch_J[pass_index]
                            ),
                            "signed_loss_energy_J": float(
                                crystal_exit_J[pass_index]
                                - transition_launch_J[pass_index]
                            ),
                            "included_in_seed_balance": True,
                        },
                        {
                            "name": f"between_pass_{pass_index + 1}_{pass_index + 2}_optical_route",
                            "category": "between_pass_elements",
                            "input_energy_J": float(
                                transition_launch_J[pass_index]
                            ),
                            "output_energy_J": float(
                                transition_route_output_J[pass_index]
                            ),
                            "signed_loss_energy_J": float(
                                transition_launch_J[pass_index]
                                - transition_route_output_J[pass_index]
                            ),
                            "included_in_seed_balance": True,
                        },
                    ]
                )
        external_loss_ledger.append(
            {
                "name": "post_gain_optical_train",
                "category": "post_gain_routing",
                "input_energy_J": post_input_J,
                "output_energy_J": post_output_J,
                "signed_loss_energy_J": post_input_J - post_output_J,
                "included_in_seed_balance": True,
            }
        )

        entry_beam_moments = [
            sampled_field_moments(
                np.sqrt(
                    np.maximum(
                        np.sum(entry_intensity_by_pass[index], axis=0),
                        0.0,
                    )
                ),
                self.grid,
            )
            for index in range(pass_count)
        ]
        exit_beam_moments = [
            sampled_field_moments(
                np.sqrt(
                    np.maximum(
                        np.sum(exit_intensity_by_pass[index], axis=0),
                        0.0,
                    )
                ),
                self.grid,
            )
            for index in range(pass_count)
        ]
        pass_stats: list[dict[str, Any]] = []
        effective_length = len(self.grid.z) * step_length
        for pass_index, hit in enumerate(plan.hits):
            angle_x, angle_y = gain_medium_axis_angles_rad(hit.marker)
            row: dict[str, Any] = {
                "pass": pass_index + 1,
                "direction": hit.direction,
                "incidence_angle_x_deg": math.degrees(angle_x),
                "incidence_angle_y_deg": math.degrees(angle_y),
                "entry_center_x_m": hit.center_x_m,
                "entry_center_y_m": hit.center_y_m,
                "sampled_gain_entry_center_x_m": entry_beam_moments[
                    pass_index
                ].get("center_x_m"),
                "sampled_gain_entry_center_y_m": entry_beam_moments[
                    pass_index
                ].get("center_y_m"),
                "sampled_gain_entry_radius_x_m": entry_beam_moments[
                    pass_index
                ].get("radius_x_m"),
                "sampled_gain_entry_radius_y_m": entry_beam_moments[
                    pass_index
                ].get("radius_y_m"),
                "sampled_crystal_exit_center_x_m": exit_beam_moments[
                    pass_index
                ].get("center_x_m"),
                "sampled_crystal_exit_center_y_m": exit_beam_moments[
                    pass_index
                ].get("center_y_m"),
                "sampled_crystal_exit_radius_x_m": exit_beam_moments[
                    pass_index
                ].get("radius_x_m"),
                "sampled_crystal_exit_radius_y_m": exit_beam_moments[
                    pass_index
                ].get("radius_y_m"),
                "effective_path_length_m": effective_length,
                "gain_input_J": float(gain_input_J[pass_index]),
                "gain_output_J": float(gain_output_J[pass_index]),
                "crystal_exit_J": float(crystal_exit_J[pass_index]),
                "crystal_aperture_loss_J": float(
                    aperture_loss_by_pass[pass_index]
                ),
                "crystal_group_transit_time_s": len(self.grid.z)
                * dt_s,
                "population_delay_contract": (
                    "resolved on one causal global multipass clock"
                ),
            }
            if pass_index + 1 < pass_count:
                flight = flights[pass_index]
                row.update(
                    {
                        "external_flight_time_to_next_pass_s": flight.delay_s,
                        "flight_time_to_next_pass_s": flight.delay_s
                        + len(self.grid.z) * dt_s,
                        "between_pass_geometric_path_length_m": flight.geometric_path_length_m,
                        "between_pass_phase_optical_path_m": flight.phase_optical_path_m,
                        "between_pass_group_optical_path_m": flight.group_optical_path_m,
                        "between_pass_population_relaxation": include_dark,
                        "next_hit_alignment_shift_x_m": float(
                            transition_alignment_shift_x[pass_index]
                        ),
                        "next_hit_alignment_shift_y_m": float(
                            transition_alignment_shift_y[pass_index]
                        ),
                    }
                )
            else:
                row["system_output_J"] = post_output_J
            pass_stats.append(row)

        pairwise_fraction = np.zeros_like(pair_common_energy_by_z)
        for pass_index in range(pass_count):
            for other_index in range(pass_count):
                if pass_index == other_index:
                    continue
                denominator = np.minimum(
                    time_energy_by_pass_z[pass_index],
                    time_energy_by_pass_z[other_index],
                )
                pairwise_fraction[pass_index, other_index] = np.divide(
                    pair_common_energy_by_z[pass_index, other_index],
                    denominator,
                    out=np.zeros_like(denominator),
                    where=denominator > np.finfo(float).tiny,
                )
        overlap_time_by_z = overlap_bins_by_z.astype(float) * dt_s
        maximum_pairwise_overlap_fraction = float(
            np.max(pairwise_fraction)
        )
        diagnostic_overlap_floor = 1e-12
        transition_diagnostics = []
        for index, (transition, flight, represented_steps) in enumerate(
            zip(plan.transitions, flights, delay_steps, strict=True)
        ):
            represented_delay = represented_steps * dt_s
            transition_diagnostics.append(
                {
                    "from_pass": index + 1,
                    "to_pass": index + 2,
                    "name": transition.name,
                    "alignment_contract": transition.alignment_contract,
                    "exact_external_delay_s": flight.delay_s,
                    "represented_external_delay_s": represented_delay,
                    "delay_quantization_error_s": represented_delay
                    - flight.delay_s,
                    "delay_steps": int(represented_steps),
                    "geometric_path_length_m": flight.geometric_path_length_m,
                    "phase_optical_path_m": flight.phase_optical_path_m,
                    "group_optical_path_m": flight.group_optical_path_m,
                    "power_transmission_before_route": (
                        common_configured_transmission
                        * transition.power_transmission
                    ),
                }
            )
            if abs(represented_delay - flight.delay_s) > 0.1 * dt_s:
                warnings.append(
                    f"Multipass transition {index + 1}->{index + 2} delay was quantized to the nearest crystal CFL bin; refine Nz and compare overlap/output convergence."
                )

        compiled_hits = [
            {
                "pass": index + 1,
                "name": hit.name,
                "direction": hit.direction,
                "center_x_m": hit.center_x_m,
                "center_y_m": hit.center_y_m,
                "incidence_angle_x_deg": math.degrees(
                    gain_medium_axis_angles_rad(hit.marker)[0]
                ),
                "incidence_angle_y_deg": math.degrees(
                    gain_medium_axis_angles_rad(hit.marker)[1]
                ),
                "path_cosine": float(path_cosines[index]),
            }
            for index, hit in enumerate(plan.hits)
        ]
        self._multipass_diagnostics = {
            "enabled": True,
            "template": plan.template,
            "pass_count": pass_count,
            "scheme": "first_order_multi_channel_upwind_cfl",
            "gain_competition": (
                "all_simultaneous_pass_fluences_share_one_local_population_event"
            ),
            "overlap_control": "causal_geometry_and_timing_only_no_user_toggle",
            "compiled_hits": compiled_hits,
            "per_pass_beam_moments": [
                {
                    "pass": index + 1,
                    "gain_entry": entry_beam_moments[index],
                    "crystal_exit": exit_beam_moments[index],
                }
                for index in range(pass_count)
            ],
            "transitions": transition_diagnostics,
            "template_assumptions": list(plan.assumptions),
            "crystal_group_index": group_index,
            "relative_group_index_span": relative_group_span,
            "mean_path_cosine": mean_path_cosine,
            "relative_path_cosine_span": relative_path_span,
            "maximum_axial_registration_relative_error": float(
                np.max(
                    np.abs(path_cosines - mean_path_cosine)
                    / mean_path_cosine
                )
            ),
            "cfl_time_step_s": dt_s,
            "crystal_cell_group_delay_s": dt_s,
            "source_temporal_bins": int(len(source_weights)),
            "absolute_time_bins": int(total_bins),
            "resolved_absolute_duration_s": duration_s,
            "temporal_cooccupancy_detected": bool(
                maximum_simultaneous_passes >= 2
            ),
            "overlap_detected": bool(
                maximum_pairwise_overlap_fraction
                > diagnostic_overlap_floor
            ),
            "overlap_detection_fraction_floor": diagnostic_overlap_floor,
            "overlap_time_by_z_s": overlap_time_by_z.tolist(),
            "maximum_overlap_time_s": float(np.max(overlap_time_by_z)),
            "maximum_simultaneous_pass_channels": int(
                maximum_simultaneous_passes
            ),
            "maximum_pairwise_common_local_energy_J": (
                maximum_pair_common_energy_J
            ),
            "maximum_pairwise_spatiotemporal_overlap_fraction": (
                maximum_pairwise_overlap_fraction
            ),
            "pairwise_spatiotemporal_overlap_fraction_by_z": (
                pairwise_fraction.tolist()
            ),
            "pairwise_overlap_definition": (
                "sum_t integral_xy(min(F_i,F_j)) / "
                "min(sum_t E_i,sum_t E_j), evaluated independently at each z"
            ),
            "dark_relaxation_included": include_dark,
            "dark_relaxation_block_steps": int(dark_block_steps),
            "dark_relaxation_operator_calls": int(dark_updates),
            "predicted_working_field_bytes": predicted_working_bytes,
            "first_hit_alignment_shift_x_m": first_shift_x,
            "first_hit_alignment_shift_y_m": first_shift_y,
            "successive_injected_pulse_coexistence_modeled": False,
            "timing_convergence_control": (
                "refine Nz and maximum_time_bins; every route delay is rounded to the nearest shared CFL bin"
            ),
            "temporal_rebin": temporal_grid.diagnostics,
        }
        if not include_dark:
            warnings.append(
                "Causal multipass resolved optical timing but did not advance dark level kinetics; enable simulation.multipass.include_dark_relaxation for a closed population clock."
            )
        if self.cfg.get("caustic_diagnostics", {}).get("enabled", False):
            warnings.append(
                "Output caustic diagnostics are disabled for streaming causal multipass transport because incoherent time slices are not retained as one complex field."
            )
        if self._positive_gain_exponent_clipping_active:
            warnings.append(
                "At least one positive unsaturated Seed single-cell gain exponent exceeded 50 and was clipped before exponentiation."
            )

        trace = self._empty_trace()
        if bool(self.physics["record_optical_planes"]):
            trace.add(
                "causal_multipass_first_gain_entrance",
                first_boundary[None, ...],
                self.grid,
                "energy_J",
            )
            trace.add(
                "system_output_time_incoherent_energy_collapse",
                np.sqrt(np.maximum(output_intensity, 0.0))[None, ...],
                self.grid,
                "energy_J",
            )
            # This plane retains intensity moments only. Its artificial zero
            # phase cannot recover angular moments of the discarded channels.
            trace.summaries[-1].update({
                "beam_quality_method": "unavailable_incoherent_channels_not_retained",
                "beam_quality_valid": False,
                "M2_x_sampled": None,
                "M2_y_sampled": None,
                "beam_quality_channel_count": None,
                "beam_quality_sampling_warning": "output_incoherent_channels_not_retained",
            })

        source_spectrum = entrance_plan.source_energies.copy()
        cumulative_volume = np.sum(pass_volumes, axis=0)
        per_pass = (
            pass_volumes.copy()
            if str(
                self.simulation.get(
                    "signal_volume_recording", "last_and_cumulative"
                )
            )
            == "per_pass"
            else None
        )
        return SignalPropagationResult(
            fluence_J_m2=pass_volumes[-1].copy(),
            population_m3=population,
            passes=pass_stats,
            source_grid_energy_J=source_total_J,
            system_output_energy_J=post_output_J,
            trace=trace,
            warnings=list(dict.fromkeys(warnings)),
            wavelengths_m=wavelengths.copy(),
            input_spectrum_J=source_spectrum,
            output_spectrum_J=output_spectrum,
            time_s=global_time,
            input_temporal_energy_J=input_temporal,
            output_temporal_energy_J=output_temporal,
            model_used=self._model_used(),
            extracted_energy_J_m3=extracted_energy,
            gain_transport=self._gain_transport_diagnostics(),
            fluence_last_pass_J_m2=pass_volumes[-1].copy(),
            fluence_cumulative_J_m2=(
                cumulative_volume.copy()
                if str(
                    self.simulation.get(
                        "signal_volume_recording", "last_and_cumulative"
                    )
                )
                != "last_only"
                else None
            ),
            fluence_per_pass_J_m2=per_pass,
            source_spatio_spectral={
                "model": str(
                    self.cfg.get("spatio_spectral", {}).get(
                        "model", "shared_source"
                    )
                ),
                "reference_wavelength_m": float(
                    self.cfg.get("spatio_spectral", {}).get(
                        "reference_wavelength_m"
                    )
                    or self.cfg["wavelength_m"]
                ),
                "wavelength_samples": source_samples,
                "entrance_plan_cache_reused": entrance_plan_reused,
                "entrance_plan_cached_bytes": int(
                    entrance_plan.spectral_field.nbytes
                ),
                **(gain_partition or {}),
            },
            passive_loss_energy_J_m3=self.passive_loss_energy_J_m3.copy(),
            crystal_aperture_loss_energy_J=(
                self.crystal_aperture_loss_energy_J
            ),
            external_loss_ledger=external_loss_ledger,
            caustic_diagnostics={
                "enabled": False,
                "reason": "streaming_causal_multipass_time_slices",
                "application": "diagnostic_only",
                "affects_propagation": False,
            },
        )
