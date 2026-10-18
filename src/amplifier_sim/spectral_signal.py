"""Time- and wavelength-resolved seed amplification.

The field stores energy in discrete temporal and spectral bins.  Spectral bins
propagate with their own wavelength and coating response.  Temporal bins cross
the gain medium in order and therefore share/deplete the same local population.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np

from .caustic import (
    caustic_positions,
    scan_sampled_caustic,
    scan_spectral_caustic,
)
from .constants import LIGHT_SPEED, photon_energy
from .dispersion import material_group_index, material_refractive_index
from .flight_time import interpass_dark_time_s, optical_train_flight_time
from .grid import Grid3D
from .geometry import (
    crystal_aperture_clipping_required,
    crystal_transverse_mask,
)
from .levels import LevelSystem
from .local_thermal_lens import LocalParabolicThermalLens
from .optical_system import (
    additive_source_gain_angle_warning,
    build_external_field,
    gain_interface,
    gain_medium_axis_angles_rad,
    gain_medium_path_cosine,
)
from .compute_backend import fft2, ifft2
from .optics import (
    OpticalTrace,
    angular_spectrum_transfer,
    propagate_fft_train,
    sampled_field_scalar_diagnostics,
)
from .profiles import DiscreteProfile, spectral_profile, temporal_profile
from .profiling import operation_profiler_enabled
from .pulse_definition import (
    build_fourier_pulse_pair,
    rebin_mass,
    trim_negligible_mass_support,
)
from .pass_routing import resolve_pass_reverse
from .result_recording import resolve_result_recording
from .signal import (
    SignalPropagationResult,
    append_gain_entrance_loss_terms,
    frantz_nodvik,
    gain_entrance_energy_partition,
    reference_ray_warnings,
    report_signal_transport_progress,
)
from .seed_spatio_spectral import (
    normalized_field_overlap,
    sampled_field_moments,
    seed_config_at_wavelength,
)
from .spatial_modes import incoherent_hg_modes


@dataclass(frozen=True)
class _SpectralEntrancePlan:
    """Immutable source-to-gain entrance data shared by repeated Seed pulses."""

    spectral_field: np.ndarray
    outside_indices: np.ndarray
    source_energies: np.ndarray
    source_spatio_samples: tuple[dict[str, Any], ...]
    wavelength_source_configs: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]


class SpectralSignalPropagator:
    """Split-step spatial propagation with sequential time-bin saturation."""

    _THERMAL_PHASE_CACHE_SCREEN_BYTES = 32 * 1024**2
    _THERMAL_PHASE_CACHE_TOTAL_BYTES = 128 * 1024**2
    _THERMAL_PHASE_CHUNK_BYTES = 16 * 1024**2
    _DIFFRACTION_TRANSFER_CACHE_TOTAL_BYTES = 1024 * 1024**2

    def __init__(
        self,
        grid: Grid3D,
        material: dict[str, Any],
        seed_cfg: dict[str, Any],
        levels_cfg: dict[str, Any],
        levels: LevelSystem,
        simulation_cfg: dict[str, Any],
        physics: dict[str, Any],
        pre_elements: list[dict[str, Any]],
        gain_marker: dict[str, Any],
        post_elements: list[dict[str, Any]],
        between_pass_elements: list[dict[str, Any]],
        gain_markers: list[dict[str, Any]] | None = None,
        between_pass_segments: list[list[dict[str, Any]]] | None = None,
        bench_warnings: list[str] | None = None,
    ):
        self.grid = grid
        self.material = material
        self.cfg = seed_cfg
        self.levels_cfg = levels_cfg
        self.levels = levels
        self.simulation = simulation_cfg
        self._result_recording = resolve_result_recording(simulation_cfg)
        self._longitudinal_profile: list[dict[str, Any]] = []
        self._active_longitudinal_cell: dict[str, Any] | None = None
        self._intensity_planes: list[np.ndarray] = []
        self._intensity_temporal_power: list[np.ndarray] = []
        self._intensity_metadata: dict[str, Any] = {}
        self._intensity_area_m2: np.ndarray | None = None
        self.physics = physics
        self.pre_elements = pre_elements
        self.gain_marker = gain_marker
        self.post_elements = post_elements
        self.between_pass_elements = between_pass_elements
        self.gain_markers = list(gain_markers or [])
        self.between_pass_segments = list(between_pass_segments or [])
        self.bench_warnings = list(bench_warnings or [])
        self.crystal_mask = crystal_transverse_mask(grid, material)
        self.clip_crystal_configured = bool(
            material.get("geometry", {}).get("clip_optical_field", True)
        )
        self.clip_crystal = crystal_aperture_clipping_required(
            material, self.crystal_mask
        )
        self._diffraction_transfer_cache: OrderedDict[
            tuple[float, ...], np.ndarray
        ] = OrderedDict()
        self._diffraction_transfer_cache_bytes = 0
        self._diffraction_transfer_cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "uncached_oversize_screens": 0,
        }
        self._diffraction_spectral_state: dict[str, Any] | None = None
        self._diffraction_spectral_generation = 0
        self._fallback_wavelength_source_configs: tuple[
            dict[str, Any], ...
        ] | None = None
        self._spectral_entrance_plan: _SpectralEntrancePlan | None = None
        self._spectral_entrance_cache_hits = 0
        self._spectral_entrance_cache_misses = 0
        self._thermal_phase_cache: OrderedDict[
            tuple[int, float], np.ndarray
        ] = OrderedDict()
        self._thermal_phase_cache_bytes = 0
        self._thermal_phase_zero_slices: set[int] = set()
        self._thermal_phase_seen_keys: set[tuple[int, float]] = set()
        self._thermal_phase_stats: dict[str, Any] = {}
        sampled_thermal_lens = dict(seed_cfg.get("thermal_lens", {}))
        self._local_parabolic_thermal_lens = (
            LocalParabolicThermalLens(grid, material, sampled_thermal_lens)
            if str(sampled_thermal_lens.get("model", "sampled_phase"))
            == "local_parabolic"
            else None
        )
        pulse_definition = seed_cfg.get("pulse_definition", {})
        source_domain = str(pulse_definition.get("domain", "independent"))
        self.source_definition_warnings: list[str] = []
        self.input_pulse_projection: dict[str, Any] = {
            "source_domain": source_domain,
            "fourier_linked": False,
            "phase_source": "none",
            "source_and_stretcher_phase_applied": False,
        }
        self._positive_gain_exponent_clipping_active = False
        self._maximum_unsaturated_gain_exponent = -np.inf
        self.photoelastic_tensor_fields: dict[str, np.ndarray] | None = None
        self._endface_surface_sag_m: dict[str, np.ndarray | None] = {
            "front": None,
            "back": None,
        }

        from .config import resolve_signal_model
        tail_settings = seed_cfg.get("tail_binning", {})
        self.temporal_bin_edges_s = None
        self.spectral_bin_edges_Hz = None
        is_spectral_time = resolve_signal_model({"simulation": simulation_cfg, "seed": seed_cfg, "physics": physics}) == "spectral_time"
        if source_domain in {"time", "spectrum", "dual"}:
            linked = build_fourier_pulse_pair(seed_cfg, include_additional_phase=True)
            self.input_pulse_projection = {
                "source_domain": source_domain,
                "fourier_linked": True,
                "phase_source": str(linked.diagnostics.get("phase_source", "unknown")),
                "source_and_stretcher_phase_applied": bool(
                    linked.diagnostics.get("additional_phase_applied", False)
                ),
                "projection": "complex_input_to_time_and_wavelength_energy_marginals",
            }
            legacy_samples = int(pulse_definition.get("solver_sample_count", 9))
            time_samples = int(pulse_definition.get("solver_time_sample_count", legacy_samples))
            spectral_samples = int(
                pulse_definition.get("solver_spectral_sample_count", legacy_samples)
            )
            time_s, time_weights = rebin_mass(
                linked.time_s,
                linked.temporal_weights,
                time_samples,
            )
            active_wavelength_m, active_spectral_weights, _ = (
                trim_negligible_mass_support(
                    linked.wavelength_m,
                    linked.spectral_weights,
                )
            )
            wavelength_m, spectral_weights = rebin_mass(
                active_wavelength_m,
                active_spectral_weights,
                spectral_samples,
            )
            # Tail-only compaction starts with the source samples, before any
            # uniform legacy cap can merge the energetic centre.
            if is_spectral_time and tail_settings.get("temporal_enabled", False):
                time_s, time_weights = linked.time_s, linked.temporal_weights
            if is_spectral_time and tail_settings.get("spectral_enabled", False):
                wavelength_m, spectral_weights = linked.wavelength_m, linked.spectral_weights
            self.temporal = DiscreteProfile(time_s, time_weights)
            self.spectrum = DiscreteProfile(wavelength_m, spectral_weights)
            self.source_definition_warnings.extend(linked.warnings)
        else:
            self.spectrum = spectral_profile(
                seed_cfg.get("spectrum", {}),
                float(seed_cfg["wavelength_m"]),
                float(seed_cfg.get("spectral_bandwidth_m", 0.0)),
            )
            self.temporal = temporal_profile(seed_cfg.get("temporal", {}), float(seed_cfg["pulse_duration_s"]))
        self.joint_time_frequency_weights: np.ndarray | None = None
        from .config import resolve_signal_model
        if resolve_signal_model({"simulation": simulation_cfg, "seed": seed_cfg, "physics": physics}) == "spectral_time":
            reduction = seed_cfg.get("spectral_time", {})
            self.input_pulse_projection["bin_reduction"] = {}
            for key, attribute in (("temporal_bins", "temporal"), ("spectral_bins", "spectrum")):
                requested = reduction.get(key)
                profile = getattr(self, attribute)
                original_count = len(profile.coordinate)
                prefix = "temporal" if attribute == "temporal" else "spectral"
                if tail_settings.get(prefix + "_enabled", False):
                    from .energy_tail_binning import energy_tail_groups
                    # Frequency quadrature is selected in frequency, not in
                    # wavelength; sort back to the profile's wavelength order.
                    coordinates = (299792458. / profile.coordinate if attribute == "spectrum" else profile.coordinate)
                    order = np.argsort(coordinates)
                    from .spectral_reporting import coordinate_bin_edges, spectral_bin_coordinate_for_result
                    native_edges = None
                    if original_count > 1:
                        if attribute == "spectrum" and spectral_bin_coordinate_for_result(
                            "spectral_time", seed_cfg
                        ) == "wavelength":
                            native_edges = 299792458. / coordinate_bin_edges(
                                np.sort(profile.coordinate)
                            )[::-1]
                        else:
                            native_edges = coordinate_bin_edges(coordinates[order])
                    plan = energy_tail_groups(coordinates[order], profile.weights[order],
                        coverage=tail_settings.get("energy_coverage", .999999),
                        tail_points=tail_settings.get(prefix + "_tail_points", 3),
                        bin_edges=native_edges)
                    if attribute == "spectrum":
                        self.spectral_bin_edges_Hz = plan["bin_edges"]
                    else:
                        self.temporal_bin_edges_s = plan["bin_edges"]
                    coordinates = (299792458. / plan["coordinate"] if attribute == "spectrum" else plan["coordinate"])
                    order = np.argsort(coordinates)
                    setattr(self, attribute, DiscreteProfile(coordinates[order], plan["mass"][order]))
                    self.input_pulse_projection["bin_reduction"][key] = {
                        "source_count": original_count, "actual_count": plan["total_groups"],
                        "central_count": plan["central_count"], "tail_counts": list(plan["tail_counts"]),
                        "initial_energy_coverage": plan["central_energy_fraction"],
                        "method": "initial_energy_native_centre_merged_tails",
                    }
                    continue
                if requested is not None and int(requested) < original_count:
                    coordinate, weights = profile.coordinate, profile.weights
                    if attribute == "spectrum":
                        coordinate, weights, _ = trim_negligible_mass_support(coordinate, weights)
                    coordinate, weights = rebin_mass(coordinate, weights, min(int(requested), len(coordinate)))
                    setattr(self, attribute, DiscreteProfile(coordinate, weights))
                self.input_pulse_projection["bin_reduction"][key] = {
                    "source_count": original_count, "requested": requested,
                    "actual_count": len(getattr(self, attribute).coordinate),
                    "method": "energy_conserving_rebin_mass",
                }
        joint = seed_cfg.get("joint_time_frequency", {})
        if str(joint.get("model", "separable")) == "matrix":
            weights = np.asarray(joint.get("weights", []), dtype=float)
            expected = (len(self.temporal.coordinate), len(self.spectrum.coordinate))
            if weights.shape != expected:
                raise ValueError(
                    "seed.joint_time_frequency.weights must have shape "
                    f"(temporal bins, spectral bins)={expected}; got {weights.shape}"
                )
            if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
                raise ValueError(
                    "seed.joint_time_frequency.weights must be finite and non-negative"
                )
            total = float(np.sum(weights))
            if total <= 0.0:
                raise ValueError(
                    "seed.joint_time_frequency.weights must contain positive mass"
                )
            weights = weights / total
            self.joint_time_frequency_weights = weights
            self.temporal = DiscreteProfile(
                self.temporal.coordinate.copy(), np.sum(weights, axis=1)
            )
            self.spectrum = DiscreteProfile(
                self.spectrum.coordinate.copy(), np.sum(weights, axis=0)
            )

    def set_photoelastic_tensor(
        self, fields: dict[str, np.ndarray] | None
    ) -> None:
        """Attach transverse FEM index-tensor fields for Jones propagation."""

        self.photoelastic_tensor_fields = fields

    def set_endface_surface_sag(
        self,
        front_surface_sag_m: np.ndarray | None,
        back_surface_sag_m: np.ndarray | None,
    ) -> None:
        self._endface_surface_sag_m = {
            "front": (
                None
                if front_surface_sag_m is None
                else np.asarray(front_surface_sag_m, dtype=float).copy()
            ),
            "back": (
                None
                if back_surface_sag_m is None
                else np.asarray(back_surface_sag_m, dtype=float).copy()
            ),
        }

    def _source_config_for_channel(
        self, channel_index: int, wavelength_m: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del channel_index
        return seed_config_at_wavelength(self.cfg, wavelength_m)

    def _diagnostics_enabled(self) -> bool:
        from .result_recording import diagnostic_observers_enabled
        return diagnostic_observers_enabled(self)

    def _scan_output_caustic(
        self,
        field: np.ndarray,
        trace: OpticalTrace,
        quantity: str,
        wavelengths_m: np.ndarray,
    ) -> dict[str, Any] | None:
        """Run the optional output caustic without a centre-wavelength shortcut."""

        if not self._diagnostics_enabled() or bool(self.physics.get("_suppress_output_diagnostics", False)):
            return None
        config = self.cfg.get("caustic_diagnostics", {})
        if not bool(config.get("enabled", False)):
            return {
                "enabled": False,
                "application": "diagnostic_only",
                "affects_propagation": False,
            }
        wavelengths = np.asarray(wavelengths_m, dtype=float)
        retain = bool(config.get("record_planes", True)) and bool(
            self.physics.get("record_optical_planes", False)
        )
        positions = caustic_positions(config)
        if len(wavelengths) == 1:
            diagnostics, planes = scan_sampled_caustic(
                field,
                self.grid,
                positions,
                float(wavelengths[0]),
                float(config.get("refractive_index", 1.0)),
                float(self.cfg.get("m2_x", 1.0)),
                float(self.cfg.get("m2_y", 1.0)),
                retain_fields=retain,
            )
        else:
            diagnostics, planes = scan_spectral_caustic(
                field,
                self.grid,
                positions,
                wavelengths,
                float(config.get("refractive_index", 1.0)),
                float(self.cfg.get("m2_x", 1.0)),
                float(self.cfg.get("m2_y", 1.0)),
                retain_fields=retain,
            )
        if retain:
            for index, (position, plane) in enumerate(planes):
                trace.add(
                    f"caustic_{index:03d}_z_{position:+.9g}_m",
                    plane,
                    self.grid,
                    quantity,
                    complex_field_available=(len(wavelengths) == 1),
                )
        diagnostics["planes_recorded"] = bool(retain)
        return diagnostics

    def _reset_thermal_phase_runtime(self) -> None:
        # Thermal screens are valid only for one accepted temperature field.
        # Keep them across passes/substeps within this propagation, never
        # across an outer thermal fixed-point update.
        self._thermal_phase_cache.clear()
        self._thermal_phase_cache_bytes = 0
        self._thermal_phase_zero_slices.clear()
        self._thermal_phase_seen_keys.clear()
        self._thermal_phase_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_evictions": 0,
            "deferred_cache_fills": 0,
            "zero_operator_skips": 0,
            "chunked_applications": 0,
            "maximum_scratch_bytes": 0,
        }
        if self._local_parabolic_thermal_lens is not None:
            self._local_parabolic_thermal_lens.reset()

    def propagate(self, population: np.ndarray, delta_n: np.ndarray) -> SignalPropagationResult:
        self._reset_thermal_phase_runtime()
        self._longitudinal_profile = []
        self._active_longitudinal_cell = None
        self._intensity_planes = []
        self._intensity_temporal_power = []
        self._intensity_metadata = {}
        self._intensity_area_m2 = None
        self._diffraction_transfer_cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "uncached_oversize_screens": 0,
        }
        self.levels.record_spectral_support(
            "signal", self.spectrum.coordinate, self.spectrum.weights
        )
        self.passive_loss_energy_J_m3 = np.zeros(self.grid.shape, dtype=float)
        self.crystal_aperture_loss_energy_J = 0.0
        self._positive_gain_exponent_clipping_active = False
        self._maximum_unsaturated_gain_exponent = -np.inf
        override = self.levels_cfg.get("gain_override_m_inv")
        self._override_gain_m_inv = (
            None
            if override is None
            else np.full(self.grid.shape, float(override), dtype=float)
        )
        record_planes = bool(self.physics["record_optical_planes"]) and self._diagnostics_enabled()
        record_cycle_scalar_diagnostics = bool(
            self.physics.get("_record_cycle_scalar_diagnostics", False)
        ) and self._diagnostics_enabled()
        wavelengths = self.spectrum.coordinate
        temporal_weights = self.temporal.weights
        configured_energy = float(self.cfg["pulse_energy_J"])
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
        propagation_mode = str(self.physics["optical_propagation"])
        external_mode = (
            "abcd" if propagation_mode in {"abcd", "hybrid_abcd_fft"} else propagation_mode
        )
        source_channel_total = len(wavelengths)
        entrance_plan = self._spectral_entrance_plan
        entrance_plan_reused = entrance_plan is not None
        if entrance_plan is None:
            spectral_fields: list[np.ndarray] = []
            outside_indices: list[float] = []
            source_energies: list[float] = []
            source_spatio_samples: list[dict[str, Any]] = []
            wavelength_source_configs: list[dict[str, Any]] = []
            source_warnings: list[str] = []
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
                    "spectral_channels": source_channel_total,
                    "optical_grid_nx": len(self.grid.x),
                    "optical_grid_ny": len(self.grid.y),
                    "working_field_estimate_bytes": (
                        source_channel_total
                        * len(self.grid.x)
                        * len(self.grid.y)
                        * np.dtype(np.complex128).itemsize
                    ),
                    "source_entrance_cache_reused": False,
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
                    configured_energy * float(weight),
                    self.pre_elements,
                    self.grid,
                    external_mode,
                    "energy_J",
                    self.physics["interface_model"],
                    False,
                    include_abcd_tilt_phase=propagation_mode
                    in {"abcd", "hybrid_abcd_fft"},
                    abcd_diagnostics=entrance_diagnostics,
                )
                if not entrance_diagnostics:
                    entrance_diagnostics.update(
                        sampled_field_moments(external.field, self.grid)
                    )
                    entrance_diagnostics["tilt_phase_included"] = bool(
                        local_cfg.get("incidence_angle_x_deg", 0.0)
                        or local_cfg.get("incidence_angle_y_deg", 0.0)
                    )
                sample_diagnostics["gain_entrance"] = entrance_diagnostics
                source_spatio_samples.append(sample_diagnostics)
                wavelength_source_configs.append(local_cfg)
                spectral_fields.append(external.field)
                outside_indices.append(external.refractive_index)
                source_energies.append(external.source_grid_quantity)
                source_warnings.extend(external.warnings)
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
                        "prepared_channel_energy_J": float(
                            external.source_grid_quantity
                        ),
                        "spectral_channels": source_channel_total,
                        "source_entrance_cache_reused": False,
                    },
                )
            spectral_field = np.stack(spectral_fields, axis=0)
            reference_index = int(
                np.argmin(
                    np.abs(wavelengths - float(self.cfg["wavelength_m"]))
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
            outside_index_array = np.asarray(outside_indices, dtype=float)
            source_energy_array = np.asarray(source_energies, dtype=float)
            for array in (
                spectral_field,
                outside_index_array,
                source_energy_array,
            ):
                array.setflags(write=False)
            entrance_plan = _SpectralEntrancePlan(
                spectral_field=spectral_field,
                outside_indices=outside_index_array,
                source_energies=source_energy_array,
                source_spatio_samples=tuple(source_spatio_samples),
                wavelength_source_configs=tuple(wavelength_source_configs),
                warnings=tuple(source_warnings),
            )
            self._spectral_entrance_plan = entrance_plan
            self._spectral_entrance_cache_misses += 1
            # The stacked immutable plan owns the source field now.  Releasing
            # the per-channel arrays before in-crystal propagation avoids
            # retaining a second full broadband Cartesian field for the rest of
            # the pulse.
            del spectral_fields, external
        else:
            self._spectral_entrance_cache_hits += 1
            report_signal_transport_progress(
                self,
                current=source_channel_total,
                total=source_channel_total,
                pass_index=0,
                pass_total=0,
                z_index=None,
                phase="spectral_source_setup",
                fraction_start=0.0,
                fraction_stop=0.12,
                metrics={
                    "spectral_channels": source_channel_total,
                    "source_entrance_cache_reused": True,
                    "source_entrance_cached_bytes": int(
                        entrance_plan.spectral_field.nbytes
                    ),
                },
            )
        spectral_field = entrance_plan.spectral_field
        source_spatio_samples = copy.deepcopy(
            list(entrance_plan.source_spatio_samples)
        )
        source_energies = entrance_plan.source_energies
        warnings.extend(entrance_plan.warnings)
        # Used by every later in-crystal/between-pass FFT so a configured
        # wavelength-dependent M2 remains consistent after the gain entrance.
        self._wavelength_source_configs = entrance_plan.wavelength_source_configs
        field = self._initialize_field(spectral_field, temporal_weights)
        external_loss_ledger: list[dict[str, Any]] = []

        def record_external_loss(
            name: str, before_J: float, after_J: float, category: str
        ) -> None:
            external_loss_ledger.append(
                {
                    "name": name,
                    "category": category,
                    "input_energy_J": float(before_J),
                    "output_energy_J": float(after_J),
                    "signed_loss_energy_J": float(before_J - after_J),
                    "included_in_seed_balance": True,
                }
            )

        gain_entrance_partition = gain_entrance_energy_partition(
            configured_source_energy_J=configured_energy,
            sampled_gain_entrance_quantity=self._integral(field),
            wavelength_samples=source_spatio_samples,
        )
        if gain_entrance_partition is None:
            record_external_loss(
                "source_to_gain_optical_train",
                float(np.sum(source_energies)),
                self._integral(field),
                "pre_gain_routing",
            )
        else:
            append_gain_entrance_loss_terms(
                external_loss_ledger, gain_entrance_partition
            )
        current_outside_n = entrance_plan.outside_indices.copy()
        n_gain = np.asarray(
            [
                material_refractive_index(self.material, float(wavelength), role="signal")
                for wavelength in wavelengths
            ],
            dtype=float,
        )
        trace = OpticalTrace()
        if record_planes:
            trace.add("spectral_source_to_gain", field, self.grid, "energy_J")
        signal_volume = np.zeros(self.grid.shape, dtype=float)
        cumulative_signal_volume = np.zeros(self.grid.shape, dtype=float)
        per_pass_signal_volumes: list[np.ndarray] = []
        volume_recording = str(
            self.simulation.get("signal_volume_recording", "last_and_cumulative")
        )
        extracted_energy = np.zeros(self.grid.shape, dtype=float)
        pass_stats: list[dict[str, Any]] = []
        passes = len(self.gain_markers) if self.gain_markers else int(self.simulation["passes"])
        transport_total = passes * len(self.grid.z)
        report_signal_transport_progress(
            self,
            current=0,
            total=transport_total,
            pass_index=1,
            pass_total=passes,
            z_index=None,
            fraction_start=0.12,
            fraction_stop=0.94,
            metrics=self._transport_progress_metrics(),
        )
        for pass_number in range(passes):
            pass_started_s = (
                time.perf_counter() if operation_profiler_enabled() else None
            )
            z_operator_timing: list[dict[str, Any]] = []
            pass_aperture_start = self.crystal_aperture_loss_energy_J
            pass_signal_volume = np.zeros(self.grid.shape, dtype=float)
            marker = self.gain_markers[pass_number] if self.gain_markers else self.gain_marker
            reverse = resolve_pass_reverse(
                str(self.cfg.get("direction", "forward")),
                str(marker.get("direction", "auto")),
                pass_number,
                bool(self.simulation.get("alternate_direction", True)),
                explicit_gain_hit=bool(self.gain_markers),
            )
            # Optional high-fidelity subclasses use the active marker while
            # applying each in-crystal half step.  The base implementation is
            # unchanged and therefore preserves every legacy result.
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
                trace.add(f"pass_{pass_number + 1}_gain_entrance", field, self.grid, "energy_J")
            energy_in = self._integral(field)
            field = self._enter_gain_transport(
                field,
                population,
                delta_n,
                pass_number,
            )
            angle_x, angle_y = gain_medium_axis_angles_rad(marker)
            path_cosine = gain_medium_path_cosine(marker)
            step_length = self.grid.dz / path_cosine
            indices = list(reversed(range(len(self.grid.z)))) if reverse else list(range(len(self.grid.z)))
            self._record_longitudinal_intensity(
                field, pass_number=pass_number, reverse=reverse,
                cell_number=0, cell_count=len(indices), z_index=indices[0],
                step_length_m=step_length,
            )
            for cell_number, iz in enumerate(indices, start=1):
                # Optional reduced/high-fidelity nonlinear closures need the
                # physical crystal slice while reusing the shared diffraction
                # hook.  Ordinary backends ignore this marker.
                self._active_z_index = int(iz)
                self._begin_longitudinal_cell(
                    field,
                    pass_number=pass_number,
                    reverse=reverse,
                    cell_number=cell_number,
                    cell_count=len(indices),
                    z_index=iz,
                    step_length_m=step_length,
                )
                cell_input_energy_J = (
                    self._integral(field)
                    if self._active_longitudinal_cell is not None
                    else 0.0
                )
                cell_active_transfer_J = 0.0
                completed_before = pass_number * len(indices) + cell_number - 1
                cell_started_s = (
                    time.perf_counter()
                    if pass_started_s is not None
                    or getattr(self, "_transport_progress_callback", None) is not None
                    else None
                )
                report_signal_transport_progress(
                    self,
                    current=completed_before + 1,
                    total=transport_total,
                    pass_index=pass_number + 1,
                    pass_total=passes,
                    z_index=iz,
                    fraction_start=0.12,
                    fraction_stop=0.94,
                    fraction_override=0.12
                    + 0.82 * completed_before / max(transport_total, 1),
                    metrics={
                        **self._transport_progress_metrics(),
                        "crystal_cell_target": cell_number,
                        "crystal_cell_state": "starting",
                    },
                )
                cell_substeps = self._crystal_cell_substeps(
                    field,
                    population,
                    delta_n[iz],
                    iz,
                    step_length,
                )
                local_step_length = step_length / cell_substeps
                for substep_index in range(cell_substeps):
                    self._begin_crystal_substep(substep_index, cell_substeps)
                    if self.physics["optical_propagation"] in {"abcd", "fft", "hybrid_abcd_fft"}:
                        field = self._diffract(field, local_step_length / 2.0, n_gain)
                        if self.clip_crystal:
                            field = self._apply_crystal_aperture(field)
                    field = self._apply_thermal_phase(
                        field,
                        delta_n[iz],
                        local_step_length,
                        iz,
                    )
                    field, local_active_transfer = self._local_material_step(
                        field,
                        population,
                        iz,
                        local_step_length,
                    )
                    extracted_energy[iz] += local_active_transfer
                    if (
                        self._active_longitudinal_cell is not None
                        and bool(
                            self._active_longitudinal_cell["regular_sample"]
                        )
                        and bool(
                            self._result_recording["longitudinal"]["gain"]
                        )
                    ):
                        cell_active_transfer_J += float(
                            np.sum(
                                np.asarray(local_active_transfer, dtype=float)
                                * self.grid.cell_areas
                            )
                            * self.grid.dz
                        )
                    if self.physics["optical_propagation"] in {"abcd", "fft", "hybrid_abcd_fft"}:
                        field = self._diffract(field, local_step_length / 2.0, n_gain)
                        if self.clip_crystal:
                            field = self._apply_crystal_aperture(field)
                    substep_fraction = (
                        completed_before
                        + (substep_index + 1) / max(cell_substeps, 1)
                    ) / max(transport_total, 1)
                    report_signal_transport_progress(
                        self,
                        current=completed_before + 1,
                        total=transport_total,
                        pass_index=pass_number + 1,
                        pass_total=passes,
                        z_index=iz,
                        fraction_start=0.12,
                        fraction_stop=0.94,
                        fraction_override=0.12 + 0.82 * substep_fraction,
                        metrics={
                            **self._transport_progress_metrics(),
                            "crystal_cell_target": cell_number,
                            "crystal_cell_state": "substep_complete",
                            "substep_index": substep_index + 1,
                            "substep_total": cell_substeps,
                        },
                    )
                self._finish_crystal_cell(population, iz)
                pass_signal_volume[iz] = self._signal_volume_density(field)
                self._record_longitudinal_intensity(
                    field, pass_number=pass_number, reverse=reverse,
                    cell_number=cell_number, cell_count=len(indices), z_index=iz,
                    step_length_m=step_length,
                )
                self._finish_longitudinal_cell(
                    field,
                    input_energy_J=cell_input_energy_J,
                    active_transfer_J=cell_active_transfer_J,
                    adaptive_substeps=cell_substeps,
                )
                cell_wall_time_s = (
                    None
                    if cell_started_s is None
                    else time.perf_counter() - cell_started_s
                )
                if cell_started_s is not None:
                    z_operator_timing.append(
                        {
                            "z_index": int(iz),
                            "cell_number": int(cell_number),
                            "adaptive_substeps": int(cell_substeps),
                            "wall_time_s": cell_wall_time_s,
                        }
                    )
                completed_cells = pass_number * len(indices) + cell_number
                report_signal_transport_progress(
                    self,
                    current=completed_cells,
                    total=transport_total,
                    pass_index=pass_number + 1,
                    pass_total=passes,
                    z_index=iz,
                    fraction_start=0.12,
                    fraction_stop=0.94,
                    metrics={
                        **self._transport_progress_metrics(),
                        "transport_energy_J": self._integral(field),
                        "adaptive_substeps": cell_substeps,
                        "cell_wall_time_s": cell_wall_time_s,
                        "crystal_cell_target": cell_number,
                        "crystal_cell_state": "complete",
                    },
                )
            signal_volume = pass_signal_volume
            cumulative_signal_volume += pass_signal_volume
            if volume_recording == "per_pass":
                per_pass_signal_volumes.append(pass_signal_volume.copy())
            gain_output = self._integral(field)
            gain_pass_scalar_diagnostics = self._gain_pass_scalar_diagnostics(
                field,
                pass_number,
            )
            field = self._exit_gain_transport(
                field,
                population,
                delta_n,
                pass_number,
            )
            if record_planes:
                trace.add(f"pass_{pass_number + 1}_gain_exit", field, self.grid, "energy_J")
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
            crystal_output = self._integral(field)
            pass_record: dict[str, Any] = {
                    "pass": pass_number + 1,
                    "direction": "backward" if reverse else "forward",
                    "incidence_angle_x_deg": math.degrees(angle_x),
                    "incidence_angle_y_deg": math.degrees(angle_y),
                    "effective_path_length_m": float(len(self.grid.z) * self.grid.dz / path_cosine),
                    "gain_input_J": energy_in,
                    "gain_output_J": gain_output,
                    "crystal_exit_J": crystal_output,
                    "crystal_aperture_loss_J": (
                        self.crystal_aperture_loss_energy_J
                        - pass_aperture_start
                    ),
                }
            if gain_pass_scalar_diagnostics:
                pass_record["scalar_diagnostics"] = gain_pass_scalar_diagnostics
            if pass_started_s is not None:
                pass_record["operator_timing"] = {
                    "crystal_pass_wall_time_s": (
                        time.perf_counter() - pass_started_s
                    ),
                    "z_cells": z_operator_timing,
                    "scope": (
                        "inclusive per-cell wall time; operator-profile entries "
                        "inside each cell overlap this total"
                    ),
                }
            pass_stats.append(pass_record)
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
                reference_index = int(
                    np.argmin(
                        np.abs(wavelengths - float(self.cfg["wavelength_m"]))
                    )
                )
                flight_initial_n = float(current_outside_n[reference_index])
                train_input = self._integral(field)
                field, current_outside_n = self._propagate_train(field, elements, current_outside_n)
                record_external_loss(
                    f"between_pass_{pass_number + 1}_{pass_number + 2}_optical_train",
                    train_input,
                    self._integral(field),
                    "between_pass_elements",
                )
                flight = optical_train_flight_time(
                    elements,
                    initial_refractive_index=flight_initial_n,
                    initial_group_refractive_index=marker.get("outside_group_refractive_index"),
                    extra_delay_s=float(
                        self.simulation.get("unmodeled_between_pass_delay_s", 0.0)
                    ),
                )
                reference_wavelength = float(wavelengths[reference_index])
                crystal_group_delay = (
                    material_group_index(
                        self.material,
                        reference_wavelength,
                        role="signal",
                    )
                    * float(pass_stats[-1]["effective_path_length_m"])
                    / LIGHT_SPEED
                )
                state_delay = flight.delay_s + crystal_group_delay
                pass_stats[-1]["external_flight_time_to_next_pass_s"] = (
                    flight.delay_s
                )
                pass_stats[-1]["crystal_group_transit_time_s"] = (
                    crystal_group_delay
                )
                pass_stats[-1]["flight_time_to_next_pass_s"] = state_delay
                pass_stats[-1]["population_delay_contract"] = (
                    "external route plus one crystal group transit; exact reversed-pass "
                    "same-cell delays vary around this mean"
                )
                pass_stats[-1]["between_pass_geometric_optical_path_m"] = (
                    flight.geometric_optical_path_m
                )
                pass_stats[-1]["between_pass_geometric_path_length_m"] = (
                    flight.geometric_path_length_m
                )
                pass_stats[-1]["between_pass_phase_optical_path_m"] = (
                    flight.phase_optical_path_m
                )
                pass_stats[-1]["between_pass_group_optical_path_m"] = (
                    flight.group_optical_path_m
                )
                population_window = float(getattr(
                    self, "resolved_intrapulse_population_time_s", lambda: 0.0
                )())
                carrier_window = float(getattr(
                    self, "resolved_intrapulse_free_carrier_time_s", lambda: 0.0
                )()) if (
                    self.physics.get("nonlinear_loss_model") == "strong_field_time_domain"
                    and self.material.get("strong_field", {}).get("free_carrier", {}).get("model") == "rate_equation"
                ) else 0.0
                population_gap = interpass_dark_time_s(state_delay, population_window)
                carrier_gap = interpass_dark_time_s(state_delay, carrier_window)
                pass_stats[-1]["between_pass_population_dark_time_s"] = population_gap
                pass_stats[-1]["between_pass_free_carrier_dark_time_s"] = carrier_gap
                window_overlap = max(population_window, carrier_window) > state_delay
                pass_stats[-1]["sequential_state_windows_overlap"] = window_overlap
                if window_overlap:
                    warnings.append(
                        "Resolved intrapulse state windows overlap the next pass. "
                        "Sequential transport uses zero dark time in the overlap; "
                        "it cannot resolve simultaneous pulses. Check the FFT window "
                        "and use causal multipass transport for physical pulse overlap."
                    )
                if (
                    population_gap > 0.0
                    and self.levels_cfg.get("gain_override_m_inv") is None
                ):
                    population[:] = self.levels.advance_dark_implicit(
                        population,
                        population_gap,
                    )
                    pass_stats[-1]["between_pass_population_relaxation"] = True
                else:
                    pass_stats[-1]["between_pass_population_relaxation"] = False
                self._advance_interpass_state(carrier_gap)
                if record_planes:
                    trace.add(f"between_pass_{pass_number + 1}_{pass_number + 2}", field, self.grid, "energy_J")
                if self.simulation.get("flip_x_between_passes", False):
                    field = np.flip(field, axis=-1).copy()
        report_signal_transport_progress(
            self,
            current=0,
            total=3,
            pass_index=0,
            pass_total=0,
            z_index=None,
            phase="output_postprocess",
            fraction_start=0.94,
            fraction_stop=0.995,
        )
        post_input = self._integral(field)
        field, current_outside_n = self._propagate_train(field, self.post_elements, current_outside_n)
        record_external_loss(
            "post_gain_optical_train",
            post_input,
            self._integral(field),
            "post_gain_routing",
        )
        if record_planes:
            trace.add("system_output", field, self.grid, "energy_J")
        output_energy = self._integral(field)
        output_field_diagnostics = (
            sampled_field_scalar_diagnostics(field, self.grid)
            if record_cycle_scalar_diagnostics
            else None
        )
        if pass_stats:
            pass_stats[-1]["system_output_J"] = output_energy
        output_spectrum = self._spectral_energy(field)
        output_temporal = self._temporal_energy(field)
        report_signal_transport_progress(
            self,
            current=1,
            total=3,
            pass_index=0,
            pass_total=0,
            z_index=None,
            phase="output_postprocess",
            fraction_start=0.94,
            fraction_stop=0.995,
            metrics={"output_energy_J": output_energy},
        )
        caustic_diagnostics = self._scan_output_caustic(
            field, trace, "energy_J", wavelengths
        )
        report_signal_transport_progress(
            self,
            current=2,
            total=3,
            pass_index=0,
            pass_total=0,
            z_index=None,
            phase="output_postprocess",
            fraction_start=0.94,
            fraction_stop=0.995,
            metrics={"output_energy_J": output_energy},
        )
        source_spectrum = np.asarray(source_energies, dtype=float).copy()
        source_total = float(np.sum(source_spectrum))
        extras = self._result_extras(field, source_total, source_spectrum, output_spectrum)
        report_signal_transport_progress(
            self,
            current=3,
            total=3,
            pass_index=0,
            pass_total=0,
            z_index=None,
            phase="output_postprocess",
            fraction_start=0.94,
            fraction_stop=0.995,
            metrics={"output_energy_J": output_energy},
        )
        optional_result: dict[str, Any] = {
            "time_s": self.temporal.coordinate.copy(),
            "input_temporal_energy_J": source_total * temporal_weights.copy(),
            "output_temporal_energy_J": output_temporal,
            "model_used": self._model_used(),
        }
        optional_result.update(extras)
        if self._positive_gain_exponent_clipping_active:
            warnings.append(
                "At least one positive unsaturated Seed single-cell gain exponent exceeded 50 "
                "and was clipped before exponentiation. Enable saturation or refine Nz before "
                "using this result quantitatively."
            )
        return SignalPropagationResult(
            fluence_J_m2=signal_volume,
            population_m3=population,
            passes=pass_stats,
            source_grid_energy_J=source_total,
            system_output_energy_J=output_energy,
            trace=trace,
            warnings=list(dict.fromkeys(warnings)),
            wavelengths_m=wavelengths.copy(),
            temporal_bin_edges_s=self.temporal_bin_edges_s,
            spectral_bin_edges_Hz=self.spectral_bin_edges_Hz,
            input_spectrum_J=source_spectrum,
            output_spectrum_J=output_spectrum,
            extracted_energy_J_m3=extracted_energy,
            gain_transport=self._gain_transport_diagnostics(),
            fluence_last_pass_J_m2=signal_volume.copy(),
            fluence_cumulative_J_m2=(
                cumulative_signal_volume.copy()
                if volume_recording != "last_only"
                else None
            ),
            fluence_per_pass_J_m2=(
                np.stack(per_pass_signal_volumes, axis=0)
                if per_pass_signal_volumes
                else None
            ),
            source_spatio_spectral={
                "model": str(self.cfg.get("spatio_spectral", {}).get("model", "shared_source")),
                "reference_wavelength_m": float(
                    self.cfg.get("spatio_spectral", {}).get("reference_wavelength_m")
                    or self.cfg["wavelength_m"]
                ),
                "wavelength_samples": source_spatio_samples,
                "entrance_plan_cache_reused": bool(entrance_plan_reused),
                "entrance_plan_cached_bytes": int(
                    entrance_plan.spectral_field.nbytes
                ),
                **(gain_entrance_partition or {}),
            },
            passive_loss_energy_J_m3=self.passive_loss_energy_J_m3.copy(),
            crystal_aperture_loss_energy_J=self.crystal_aperture_loss_energy_J,
            external_loss_ledger=external_loss_ledger,
            caustic_diagnostics=caustic_diagnostics,
            output_field_diagnostics=output_field_diagnostics,
            longitudinal_intensity_W_m2=(
                np.stack(self._intensity_planes) if self._intensity_planes else None
            ),
            longitudinal_intensity_area_m2=self._intensity_area_m2,
            longitudinal_temporal_power_W=(
                np.stack(self._intensity_temporal_power)
                if self._intensity_temporal_power else None
            ),
            **optional_result,
        )

    def _transport_progress_metrics(self) -> dict[str, Any]:
        """Return backend-specific metrics for the fixed progress surface."""

        return {}

    def _begin_longitudinal_cell(
        self,
        field: np.ndarray,
        *,
        pass_number: int,
        reverse: bool,
        cell_number: int,
        cell_count: int,
        z_index: int,
        step_length_m: float,
    ) -> None:
        """Open one scalar record for a physical z cell, never a microstep."""

        longitudinal = self._result_recording["longitudinal"]
        if not bool(longitudinal["enabled"]) or not self._diagnostics_enabled():
            self._active_longitudinal_cell = None
            return
        z_stride = max(int(longitudinal["z_stride"]), 1)
        m2_stride = max(int(longitudinal["m2_z_stride"]), 1)
        endpoint = cell_number in {1, cell_count}
        regular_sample = endpoint or (cell_number - 1) % z_stride == 0
        m2_sample = bool(longitudinal["m2_checkpoints"]) and (
            endpoint or (cell_number - 1) % m2_stride == 0
        )
        if not regular_sample and not m2_sample:
            self._active_longitudinal_cell = None
            return
        context: dict[str, Any] = {
            "pass": int(pass_number + 1),
            "direction": "backward" if reverse else "forward",
            "cell_number": int(cell_number),
            "cell_count": int(cell_count),
            "z_index": int(z_index),
            # Metrics are evaluated after the complete cell operator.  Locate
            # them at the exit face, including the reversed-pass orientation.
            "z_m": float(self.grid.z_edges[z_index if reverse else z_index + 1]),
            "path_position_m": float(cell_number * step_length_m),
            "regular_sample": bool(regular_sample),
            "m2_sample": bool(m2_sample),
            "nonlinear": {},
        }
        self._active_longitudinal_cell = context
        self._initialize_longitudinal_cell_context(field, context)

    def _record_longitudinal_intensity(self, field: np.ndarray, **location: Any) -> None:
        """Only coherent backends have a physical instantaneous intensity."""

        del field, location

    def _initialize_longitudinal_cell_context(
        self,
        field: np.ndarray,
        context: dict[str, Any],
    ) -> None:
        """Backend hook for scalar snapshots needed at the cell entrance."""

        del field, context

    def _longitudinal_spatial_sample(
        self,
        field: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return fluence, area weights, and coordinates on the native grid."""

        fluence = np.abs(np.asarray(field)) ** 2
        if fluence.ndim > 2:
            fluence = np.sum(
                fluence,
                axis=tuple(range(fluence.ndim - 2)),
            )
        xx, yy = self.grid.transverse_mesh
        return (
            np.asarray(fluence, dtype=float),
            np.asarray(self.grid.cell_areas * self.crystal_mask, dtype=float),
            np.asarray(xx, dtype=float),
            np.asarray(yy, dtype=float),
        )

    def _longitudinal_cartesian_field(self, field: np.ndarray) -> np.ndarray:
        """Return a Cartesian field only for sparse M2 checkpoints."""

        return np.asarray(field)

    def _longitudinal_mode_metrics(self, field: np.ndarray) -> dict[str, Any]:
        del field
        return {}

    def _longitudinal_capabilities(self) -> dict[str, bool]:
        return {
            "beam": True,
            "gain": True,
            "nonlinear": False,
            "modes": False,
            "m2_checkpoints": True,
            "intensity_snapshots": False,
        }

    def _longitudinal_extra_metrics(
        self,
        field: np.ndarray,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        del field, context
        return {}

    @staticmethod
    def _longitudinal_beam_metrics(
        fluence_J_m2: np.ndarray,
        area_m2: np.ndarray,
        x_m: np.ndarray,
        y_m: np.ndarray,
    ) -> dict[str, float]:
        fluence = np.asarray(fluence_J_m2, dtype=float)
        weights = np.asarray(area_m2, dtype=float)
        active = weights > 0.0
        weighted_energy = fluence * weights
        energy = float(np.sum(weighted_energy))
        safe_energy = max(energy, np.finfo(float).tiny)
        centroid_x = float(np.sum(weighted_energy * x_m) / safe_energy)
        centroid_y = float(np.sum(weighted_energy * y_m) / safe_energy)
        variance_x = max(
            float(np.sum(weighted_energy * (x_m - centroid_x) ** 2) / safe_energy),
            0.0,
        )
        variance_y = max(
            float(np.sum(weighted_energy * (y_m - centroid_y) ** 2) / safe_energy),
            0.0,
        )
        fluence_squared_integral = float(np.sum(fluence**2 * weights))
        effective_area = energy**2 / max(
            fluence_squared_integral,
            np.finfo(float).tiny,
        )
        return {
            "fluence_integrated_energy_J": energy,
            "peak_fluence_J_m2": float(np.max(fluence[active]))
            if np.any(active)
            else 0.0,
            "energy_weighted_fluence_J_m2": (
                fluence_squared_integral / safe_energy
            ),
            "effective_area_m2": effective_area,
            "centroid_x_m": centroid_x,
            "centroid_y_m": centroid_y,
            "radius_x_D4sigma_m": 2.0 * math.sqrt(variance_x),
            "radius_y_D4sigma_m": 2.0 * math.sqrt(variance_y),
        }

    def _finish_longitudinal_cell(
        self,
        field: np.ndarray,
        *,
        input_energy_J: float,
        active_transfer_J: float,
        adaptive_substeps: int,
    ) -> None:
        context = self._active_longitudinal_cell
        self._active_longitudinal_cell = None
        if context is None:
            return
        longitudinal = self._result_recording["longitudinal"]
        row = {
            key: context[key]
            for key in (
                "pass",
                "direction",
                "cell_number",
                "cell_count",
                "z_index",
                "z_m",
                "path_position_m",
            )
        }
        row.update(
            {
                "sample_location": "physical_z_cell_exit",
                "adaptive_substeps": int(adaptive_substeps),
                "input_energy_J": float(input_energy_J),
            }
        )
        exit_energy_J = self._integral(field)
        row["exit_energy_J"] = float(exit_energy_J)
        row["net_energy_change_J"] = float(exit_energy_J - input_energy_J)
        if bool(context["regular_sample"]) and bool(longitudinal["beam"]):
            row.update(
                self._longitudinal_beam_metrics(
                    *self._longitudinal_spatial_sample(field)
                )
            )
        if bool(context["regular_sample"]) and bool(longitudinal["gain"]):
            row["active_medium_energy_transfer_J"] = float(active_transfer_J)
            row["non_active_net_energy_change_J"] = float(
                exit_energy_J - input_energy_J - active_transfer_J
            )
        if bool(context["regular_sample"]) and bool(longitudinal["nonlinear"]):
            row.update(self._longitudinal_extra_metrics(field, context))
        if bool(context["regular_sample"]) and bool(longitudinal["modes"]):
            row.update(self._longitudinal_mode_metrics(field))
        if bool(context["m2_sample"]):
            try:
                checkpoint = sampled_field_scalar_diagnostics(
                    self._longitudinal_cartesian_field(field),
                    self.grid,
                )
            except (ValueError, FloatingPointError) as exc:
                row["m2_checkpoint_status"] = "unavailable"
                row["m2_checkpoint_reason"] = str(exc)
            else:
                quality = checkpoint.get("beam_quality", {})
                finite = checkpoint.get("beam_quality_finite_difference", {})
                boundary = checkpoint.get("boundary", {})
                row.update(
                    {
                        "m2_checkpoint_status": "computed",
                        "M2_x": quality.get("m2_x"),
                        "M2_y": quality.get("m2_y"),
                        "M2_x_finite_difference": finite.get("m2_x"),
                        "M2_y_finite_difference": finite.get("m2_y"),
                        "boundary_energy_fraction": boundary.get(
                            "boundary_energy_fraction"
                        ),
                    }
                )
        self._longitudinal_profile.append(row)

    def _model_used(self) -> str:
        return "spectral_time"

    def _gain_transport_diagnostics(self) -> dict[str, Any]:
        maximum_exponent = (
            None
            if not np.isfinite(self._maximum_unsaturated_gain_exponent)
            else float(self._maximum_unsaturated_gain_exponent)
        )
        result = {
            "backend": "frantz_nodvik",
            "status": "applied",
            "time_resolved": len(self.temporal.coordinate) > 1,
            "temporal_bins": len(self.temporal.coordinate),
            "spectral_bins": len(self.spectrum.coordinate),
            "spectral_coupling": "symmetric_sequential_shared_reservoir",
            "spectral_order_sweeps": 2,
            "joint_time_frequency_model": (
                "matrix"
                if self.joint_time_frequency_weights is not None
                else "separable_marginals"
            ),
            "override_gain_reservoir_shared_across_passes": True,
            "maximum_unsaturated_gain_exponent": maximum_exponent,
            "positive_gain_exponent_clipping_active": (
                self._positive_gain_exponent_clipping_active
            ),
            "thermal_phase_screen": {
                **self._thermal_phase_stats,
                "cached_screen_count": len(self._thermal_phase_cache),
                "cached_bytes": self._thermal_phase_cache_bytes,
                "cache_screen_budget_bytes": self._THERMAL_PHASE_CACHE_SCREEN_BYTES,
                "cache_total_budget_bytes": self._THERMAL_PHASE_CACHE_TOTAL_BYTES,
                "chunk_budget_bytes": self._THERMAL_PHASE_CHUNK_BYTES,
            },
            "diffraction_transfer_cache": {
                **self._diffraction_transfer_cache_stats,
                "entries": len(self._diffraction_transfer_cache),
                "cached_bytes": self._diffraction_transfer_cache_bytes,
                "cache_total_budget_bytes": (
                    self._DIFFRACTION_TRANSFER_CACHE_TOTAL_BYTES
                ),
            },
            "source_entrance_cache": {
                "hits": int(self._spectral_entrance_cache_hits),
                "misses": int(self._spectral_entrance_cache_misses),
                "cached": self._spectral_entrance_plan is not None,
                "cached_bytes": (
                    0
                    if self._spectral_entrance_plan is None
                    else int(self._spectral_entrance_plan.spectral_field.nbytes)
                ),
            },
            "result_recording": copy.deepcopy(self._result_recording),
        }
        result["result_recording"]["longitudinal"]["row_count"] = len(
            self._longitudinal_profile
        )
        capabilities = self._longitudinal_capabilities()
        result["result_recording"]["longitudinal"]["capabilities"] = (
            capabilities
        )
        if self._intensity_metadata:
            result["intensity_snapshots"] = copy.deepcopy(self._intensity_metadata)
        intensity_enabled = bool(
            self._result_recording["longitudinal"].get("intensity_snapshots", False)
        )
        result["result_recording"]["longitudinal"].update(
            intensity_plane_count=len(self._intensity_planes),
            intensity_status=(
                "recorded" if self._intensity_planes else
                "unavailable_for_backend" if intensity_enabled else "disabled"
            ),
            retains_field_arrays=bool(self._intensity_planes),
        )
        result["result_recording"]["longitudinal"][
            "effective_categories"
        ] = {
            key: bool(
                result["result_recording"]["longitudinal"].get(key, False)
                and available
            )
            for key, available in capabilities.items()
        }
        if self._longitudinal_profile:
            result["longitudinal_profile"] = [
                dict(row) for row in self._longitudinal_profile
            ]
        critical_power_rows = [
            row
            for row in self._longitudinal_profile
            if row.get("peak_to_critical_power_ratio") is not None
        ]
        if critical_power_rows:
            maximum_row = max(
                critical_power_rows,
                key=lambda row: float(row["peak_to_critical_power_ratio"]),
            )
            first_crossing = next(
                (
                    row
                    for row in critical_power_rows
                    if float(row["peak_to_critical_power_ratio"]) >= 1.0
                ),
                None,
            )

            def critical_power_location(row: dict[str, Any]) -> dict[str, Any]:
                return {
                    key: row.get(key)
                    for key in (
                        "pass",
                        "direction",
                        "cell_number",
                        "z_index",
                        "z_m",
                        "path_position_m",
                        "peak_to_critical_power_ratio",
                        "maximum_peak_power_W",
                        "critical_power_W",
                    )
                }

            result["critical_power_longitudinal"] = {
                "evaluated": True,
                "first_P_over_Pcr_ge_1": (
                    None
                    if first_crossing is None
                    else critical_power_location(first_crossing)
                ),
                "maximum_P_over_Pcr": critical_power_location(maximum_row),
                "scope": (
                    "physical-z cell records in optical path order; Pcr is a "
                    "bulk screening estimate, not a finite-crystal collapse proof"
                ),
            }
        if self._local_parabolic_thermal_lens is not None:
            result["thermal_lens"] = (
                self._local_parabolic_thermal_lens.diagnostics()
            )
        return result

    def _initialize_field(self, spectral_field: np.ndarray, temporal_weights: np.ndarray) -> np.ndarray:
        if self.joint_time_frequency_weights is not None:
            spectral_marginal = np.sum(
                self.joint_time_frequency_weights, axis=0
            )
            conditional = np.divide(
                self.joint_time_frequency_weights,
                spectral_marginal[None, :],
                out=np.zeros_like(self.joint_time_frequency_weights),
                where=spectral_marginal[None, :] > 0.0,
            )
            return (
                np.sqrt(conditional)[:, :, None, None]
                * spectral_field[None, ...]
            )
        return np.sqrt(temporal_weights[:, None, None, None]) * spectral_field[None, ...]

    def _after_gain_step(
        self,
        field: np.ndarray,
        step_length_m: float,
        z_index: int | None = None,
    ) -> np.ndarray:
        del z_index
        return field


    def _local_material_step(
        self,
        field: np.ndarray,
        population: np.ndarray,
        z_index: int,
        step_length_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance gain and optional post-gain material physics in one z cell.

        Subclasses may split the local operators into smaller pieces, but must
        return active-transition energy density separately from passive and
        nonlinear losses.  Keeping that ledger at this boundary prevents a
        coupled operator from being mistaken for stimulated extraction.
        """

        fluence_before_gain = np.sum(np.abs(field) ** 2, axis=(0, 1))
        passive_before = self.passive_loss_energy_J_m3[z_index].copy()
        field = self._gain_step(field, population, z_index, step_length_m)
        fluence_after_gain_and_loss = np.sum(np.abs(field) ** 2, axis=(0, 1))
        active_transfer = (
            fluence_after_gain_and_loss - fluence_before_gain
        ) / self.grid.dz + (
            self.passive_loss_energy_J_m3[z_index] - passive_before
        )
        return (
            self._after_gain_step(field, step_length_m, z_index),
            active_transfer,
        )

    def _crystal_cell_substeps(
        self,
        field: np.ndarray,
        population: np.ndarray,
        delta_n_slice: np.ndarray,
        z_index: int,
        step_length_m: float,
    ) -> int:
        """Return spatial split steps for one physical crystal z cell."""

        del field, population, delta_n_slice, z_index, step_length_m
        return 1

    def _begin_crystal_substep(self, index: int, count: int) -> None:
        """Notify optional high-fidelity subclasses of the active local step."""

        del index, count

    def _finish_crystal_cell(self, population: np.ndarray, z_index: int) -> None:
        """Commit independent gain volumes before leaving the physical cell.

        Reduced-coordinate solvers pass their own population array to gain;
        the transaction therefore retains that array, not the outer argument.
        Passive/extracted energy ledgers already use the parent-cell volume.
        """
        del population
        state = getattr(self, "_pending_gain_cell", None)
        if state is None:
            return
        if state["z_index"] != z_index or not math.isclose(
            state["fraction"], 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError("Independent gain volumes do not cover the physical cell")
        state["population"][:, z_index] = state["final_population"]
        if state["override"] is not None:
            state["override"][z_index] = state["final_override"]
        self._pending_gain_cell = None

    def _result_extras(
        self,
        field: np.ndarray,
        source_total_J: float,
        source_spectrum_J: np.ndarray,
        output_spectrum_J: np.ndarray,
    ) -> dict[str, Any]:
        return {}

    def _advance_interpass_state(self, delay_s: float) -> None:
        """Advance optional subclass state between crystal hits."""

        del delay_s

    def _advance_population_between_temporal_bins(
        self,
        population_slice: np.ndarray,
        previous_bin_index: int,
        current_bin_index: int,
        z_index: int,
    ) -> np.ndarray:
        """Advance optional population kinetics between ordered Seed packets.

        The ordinary spectral/FN backend treats its time bins as sequential
        fluence packets with no intervening dark evolution.  Reduced backends
        that explicitly opt into intrapulse relaxation override this hook;
        keeping it here avoids duplicating the shared-reservoir FN operator.
        """

        del previous_bin_index, current_bin_index, z_index
        return population_slice

    def _enter_gain_transport(
        self,
        field: np.ndarray,
        population: np.ndarray,
        delta_n: np.ndarray,
        pass_number: int,
    ) -> np.ndarray:
        """Convert a full entrance field into an optional reduced basis.

        The base backend keeps the Cartesian field unchanged.  Reduced-order
        subclasses may compress only inside the gain medium; interfaces,
        external routing, recorded planes, and final diagnostics continue to
        receive the ordinary sampled x/y field.
        """

        del population, delta_n, pass_number
        return field

    def _exit_gain_transport(
        self,
        field: np.ndarray,
        population: np.ndarray,
        delta_n: np.ndarray,
        pass_number: int,
    ) -> np.ndarray:
        """Restore an optional in-crystal reduced field to the x/y grid."""

        del population, delta_n, pass_number
        return field

    def _gain_pass_scalar_diagnostics(
        self,
        field: np.ndarray,
        pass_number: int,
    ) -> dict[str, Any]:
        """Return optional compact diagnostics before a reduced field is restored."""

        del field, pass_number
        return {}

    def _signal_volume_density(self, field: np.ndarray) -> np.ndarray:
        """Return the transverse fluence density stored for one z cell."""

        return np.sum(np.abs(field) ** 2, axis=(0, 1))

    def _diffract(
        self,
        field: np.ndarray,
        distance_m: float,
        refractive_index: float | np.ndarray,
    ) -> np.ndarray:
        self._ensure_diffraction_cache_state()
        state_token, indices, m2_x, m2_y = (
            self._diffraction_spectral_parameters(refractive_index)
        )
        key = (float(state_token), float(distance_m))
        transfers = self._diffraction_transfer_cache.get(key)
        if transfers is None:
            self._diffraction_transfer_cache_stats["misses"] += 1
            predicted_bytes = int(
                len(self.spectrum.coordinate)
                * len(self.grid.y)
                * len(self.grid.x)
                * np.dtype(np.complex128).itemsize
            )
            cacheable = bool(
                predicted_bytes
                <= self._DIFFRACTION_TRANSFER_CACHE_TOTAL_BYTES
            )
            # Evict before allocating a new screen so adaptive step lengths do
            # not create an old+new cache peak.  Oversize screens execute
            # correctly without persistence and first release stale entries.
            while self._diffraction_transfer_cache and (
                not cacheable
                or self._diffraction_transfer_cache_bytes + predicted_bytes
                > self._DIFFRACTION_TRANSFER_CACHE_TOTAL_BYTES
            ):
                _evicted_key, evicted = (
                    self._diffraction_transfer_cache.popitem(last=False)
                )
                self._diffraction_transfer_cache_bytes -= int(evicted.nbytes)
                self._diffraction_transfer_cache_stats["evictions"] += 1
            # Fill the final tensor directly.  ``np.stack([...])`` retained one
            # complete list of per-wavelength screens while allocating the
            # stacked result, nearly doubling the peak for future broadband
            # Cartesian coherent fields.
            transfers = np.empty(
                (
                    len(self.spectrum.coordinate),
                    len(self.grid.y),
                    len(self.grid.x),
                ),
                dtype=np.complex128,
            )
            for index, wavelength in enumerate(self.spectrum.coordinate):
                transfers[index] = angular_spectrum_transfer(
                    self.grid,
                    distance_m,
                    float(wavelength),
                    float(indices[index]),
                    float(m2_x[index]),
                    float(m2_y[index]),
                )
            if cacheable:
                transfers.setflags(write=False)
                self._diffraction_transfer_cache[key] = transfers
                self._diffraction_transfer_cache_bytes += int(
                    transfers.nbytes
                )
            else:
                self._diffraction_transfer_cache_stats[
                    "uncached_oversize_screens"
                ] += 1
        else:
            self._diffraction_transfer_cache.move_to_end(key)
            self._diffraction_transfer_cache_stats["hits"] += 1
        marker = getattr(self, "_active_gain_marker", self.gain_marker)
        angle_x, angle_y = gain_medium_axis_angles_rad(marker)
        fourier_field = np.asarray(
            fft2(field),
            dtype=np.result_type(field.dtype, transfers.dtype),
        )
        fourier_field *= transfers[None, ...]
        if angle_x == 0.0 and angle_y == 0.0:
            return ifft2(fourier_field)
        axial_distance = float(distance_m) * gain_medium_path_cosine(marker)
        shift_x = axial_distance * math.tan(angle_x)
        shift_y = axial_distance * math.tan(angle_y)
        fx = np.fft.fftfreq(len(self.grid.x), d=self.grid.dx)
        fy = np.fft.fftfreq(len(self.grid.y), d=self.grid.dy)
        fxx, fyy = np.meshgrid(fx, fy, indexing="xy")
        walkoff = np.exp(-2j * math.pi * (fxx * shift_x + fyy * shift_y))
        # Diffraction and a Fourier shift are adjacent diagonal operators in
        # the same transverse-frequency basis.  Their transfer functions can
        # therefore be multiplied before one shared FFT pair.
        fourier_field *= walkoff[None, None, ...]
        return ifft2(fourier_field)

    def _ensure_diffraction_cache_state(self) -> None:
        """Lazily initialize cache bookkeeping for wrappers and old tests.

        Production propagators initialize these fields in ``__init__``.  Some
        lightweight adapters and operator-level tests deliberately construct a
        propagator without running the full constructor, so the pure
        diffraction operator must still degrade to a valid empty cache.
        """

        cache = getattr(self, "_diffraction_transfer_cache", None)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict((cache or {}).items())
            self._diffraction_transfer_cache = cache
        if not hasattr(self, "_diffraction_transfer_cache_bytes"):
            self._diffraction_transfer_cache_bytes = int(
                sum(int(getattr(value, "nbytes", 0)) for value in cache.values())
            )
        stats = getattr(self, "_diffraction_transfer_cache_stats", None)
        if not isinstance(stats, dict):
            stats = {}
            self._diffraction_transfer_cache_stats = stats
        for name in (
            "hits",
            "misses",
            "evictions",
            "uncached_oversize_screens",
        ):
            stats.setdefault(name, 0)
        if not hasattr(self, "_diffraction_spectral_state"):
            self._diffraction_spectral_state = None
        if not hasattr(self, "_diffraction_spectral_generation"):
            self._diffraction_spectral_generation = 0
        if not hasattr(self, "_fallback_wavelength_source_configs"):
            self._fallback_wavelength_source_configs = None

    def _diffraction_spectral_parameters(
        self,
        refractive_index: float | np.ndarray,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Return immutable wavelength-dependent diffraction parameters.

        The previous cache key expanded every refractive-index and M² sample
        into a Python tuple on every half-step.  At broadband coherent sizes
        that meant hashing more than ten thousand Python floats per call.  The
        accepted spectral source is immutable during one propagation, so a
        compact generation token is sufficient; value comparisons retain the
        cache safely across later propagations with equivalent source data.
        """

        self._ensure_diffraction_cache_state()

        coordinate = np.asarray(self.spectrum.coordinate, dtype=float)
        indices = np.broadcast_to(
            np.asarray(refractive_index, dtype=float), coordinate.shape
        )
        configs = getattr(self, "_wavelength_source_configs", None)
        if configs is None:
            if (
                self._fallback_wavelength_source_configs is None
                or len(self._fallback_wavelength_source_configs)
                != len(coordinate)
            ):
                self._fallback_wavelength_source_configs = tuple(
                    self.cfg for _ in range(len(coordinate))
                )
            configs = self._fallback_wavelength_source_configs

        state = self._diffraction_spectral_state
        if (
            state is not None
            and state["configs"] is configs
            and state["coordinate_object"] is self.spectrum.coordinate
            and np.array_equal(indices, state["indices"])
        ):
            return (
                int(state["generation"]),
                state["indices"],
                state["m2_x"],
                state["m2_y"],
            )

        m2_x = np.asarray(
            [float(config.get("m2_x", 1.0)) for config in configs],
            dtype=float,
        )
        m2_y = np.asarray(
            [float(config.get("m2_y", 1.0)) for config in configs],
            dtype=float,
        )
        equivalent = bool(
            state is not None
            and np.array_equal(coordinate, state["coordinate_values"])
            and np.array_equal(indices, state["indices"])
            and np.array_equal(m2_x, state["m2_x"])
            and np.array_equal(m2_y, state["m2_y"])
        )
        if equivalent:
            generation = int(state["generation"])
        else:
            self._diffraction_spectral_generation += 1
            generation = self._diffraction_spectral_generation
            self._diffraction_transfer_cache.clear()
            self._diffraction_transfer_cache_bytes = 0
        coordinate_values = np.array(coordinate, dtype=float, copy=True)
        index_values = np.array(indices, dtype=float, copy=True)
        for values in (coordinate_values, index_values, m2_x, m2_y):
            values.setflags(write=False)
        self._diffraction_spectral_state = {
            "generation": generation,
            "configs": configs,
            "coordinate_object": self.spectrum.coordinate,
            "coordinate_values": coordinate_values,
            "indices": index_values,
            "m2_x": m2_x,
            "m2_y": m2_y,
        }
        return generation, index_values, m2_x, m2_y

    def _gain_interface(self, field, *args, channel_index=0, **kwargs):
        """Apply a gain face; reduced field representations may convert its phase."""
        return gain_interface(field, *args, **kwargs)

    def _apply_thermal_phase(
        self,
        field: np.ndarray,
        delta_n_slice: np.ndarray,
        step_length_m: float,
        z_index: int,
    ) -> np.ndarray:
        """Apply a bounded-memory, propagation-local thermal phase screen."""

        delta_n = np.asarray(delta_n_slice, dtype=float)
        if self._local_parabolic_thermal_lens is not None:
            delta_n = self._local_parabolic_thermal_lens.fit_slice(
                delta_n,
                int(z_index),
            )
        if int(z_index) in self._thermal_phase_zero_slices:
            self._thermal_phase_stats["zero_operator_skips"] += 1
            return field
        if not np.any(delta_n):
            self._thermal_phase_zero_slices.add(int(z_index))
            self._thermal_phase_stats["zero_operator_skips"] += 1
            return field

        wavelengths = np.asarray(self.spectrum.coordinate, dtype=float)
        key = (int(z_index), float(step_length_m))
        cached = self._thermal_phase_cache.get(key)
        if cached is not None:
            self._thermal_phase_cache.move_to_end(key)
            field *= cached[None, ...]
            self._thermal_phase_stats["cache_hits"] += 1
            return field

        bytes_per_channel = delta_n.size * np.dtype(np.complex128).itemsize
        predicted_bytes = len(wavelengths) * bytes_per_channel
        can_cache = bool(
            predicted_bytes <= self._THERMAL_PHASE_CACHE_SCREEN_BYTES
            and predicted_bytes <= self._THERMAL_PHASE_CACHE_TOTAL_BYTES
        )
        coefficient = 1j * 2.0 * math.pi * float(step_length_m)
        repeated_key = key in self._thermal_phase_seen_keys
        self._thermal_phase_seen_keys.add(key)
        if can_cache and repeated_key:
            phase = np.exp(
                coefficient
                * delta_n[None, :, :]
                / wavelengths[:, None, None]
            )
            phase.setflags(write=False)
            while (
                self._thermal_phase_cache
                and self._thermal_phase_cache_bytes + phase.nbytes
                > self._THERMAL_PHASE_CACHE_TOTAL_BYTES
            ):
                _evicted_key, evicted = self._thermal_phase_cache.popitem(
                    last=False
                )
                self._thermal_phase_cache_bytes -= int(evicted.nbytes)
                self._thermal_phase_stats["cache_evictions"] = int(
                    self._thermal_phase_stats.get("cache_evictions", 0)
                ) + 1
            self._thermal_phase_cache[key] = phase
            self._thermal_phase_cache_bytes += phase.nbytes
            self._thermal_phase_stats["cache_misses"] += 1
            self._thermal_phase_stats["maximum_scratch_bytes"] = max(
                int(self._thermal_phase_stats["maximum_scratch_bytes"]),
                int(phase.nbytes),
            )
            field *= phase[None, ...]
            return field

        if can_cache:
            # A one-pass/one-substep screen has no consumer after this call.
            # Delay persistent caching until the key is observed again; this
            # preserves the acceleration for repeated passes/substeps without
            # retaining up to the full cache budget for one-use screens.
            phase = np.exp(
                coefficient
                * delta_n[None, :, :]
                / wavelengths[:, None, None]
            )
            field *= phase[None, ...]
            self._thermal_phase_stats["cache_misses"] += 1
            self._thermal_phase_stats["deferred_cache_fills"] += 1
            self._thermal_phase_stats["maximum_scratch_bytes"] = max(
                int(self._thermal_phase_stats["maximum_scratch_bytes"]),
                int(phase.nbytes),
            )
            return field

        channel_count = max(
            1,
            int(self._THERMAL_PHASE_CHUNK_BYTES // max(bytes_per_channel, 1)),
        )
        for start in range(0, len(wavelengths), channel_count):
            stop = min(start + channel_count, len(wavelengths))
            phase = np.exp(
                coefficient
                * delta_n[None, :, :]
                / wavelengths[start:stop, None, None]
            )
            field[:, start:stop] *= phase[None, ...]
            self._thermal_phase_stats["maximum_scratch_bytes"] = max(
                int(self._thermal_phase_stats["maximum_scratch_bytes"]),
                int(phase.nbytes),
            )
        self._thermal_phase_stats["cache_misses"] += 1
        self._thermal_phase_stats["chunked_applications"] += 1
        return field

    def _propagate_train(
        self,
        field: np.ndarray,
        elements: list[dict[str, Any]],
        refractive_index: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not elements:
            return field, np.broadcast_to(
                np.asarray(refractive_index, dtype=float),
                self.spectrum.coordinate.shape,
            ).copy()
        result = np.empty_like(field)
        indices: list[float] = []
        input_indices = np.broadcast_to(
            np.asarray(refractive_index, dtype=float),
            self.spectrum.coordinate.shape,
        )
        for il, wavelength in enumerate(self.spectrum.coordinate):
            local_cfg = getattr(self, "_wavelength_source_configs", [self.cfg] * len(self.spectrum.coordinate))[il]
            result[:, il], current_n, _ = propagate_fft_train(
                field[:, il],
                elements,
                self.grid,
                float(wavelength),
                float(input_indices[il]),
                "energy_J",
                self.physics["interface_model"],
                float(local_cfg.get("m2_x", 1.0)),
                float(local_cfg.get("m2_y", 1.0)),
                False,
            )
            indices.append(current_n)
        return result, np.asarray(indices, dtype=float)

    def _simultaneous_gain_channel_groups(self) -> tuple[np.ndarray, ...]:
        """Return channels that must consume one gain packet simultaneously.

        Ordinary spectral bins are distinct photon packets and therefore form
        singleton groups.  Spatial-mode specializations may override this to
        group mutually incoherent fields at one physical wavelength, so the
        shared population is updated once from their summed local fluence.
        """

        return tuple(
            np.asarray([index], dtype=int)
            for index in range(len(self.spectrum.coordinate))
        )

    def _gain_step(self, field: np.ndarray, population: np.ndarray, iz: int, step_length_m: float) -> np.ndarray:
        policy = getattr(
            getattr(self, "local_operator_settings", None),
            "spatial_population_update", "shared_cell",
        )
        if policy != "independent_groups":
            return self._gain_step_local(field, population, iz, step_length_m)
        # One local gain call represents an entire group of optical substeps.
        # Groups occupy distinct z volumes, with a common pre-pulse density.
        # Each call still resolves spectral competition within its own volume.
        count = max(int(self._current_crystal_cell_substeps), 1)
        group = int(self._current_gain_group_size)
        if not 1 <= group <= count:
            raise RuntimeError("Invalid independent gain group size")
        fraction = group / count
        override_value = self.levels_cfg.get("gain_override_m_inv")
        if override_value is not None and getattr(self, "_override_gain_m_inv", None) is None:
            self._override_gain_m_inv = np.full(population.shape[1:], float(override_value))
        override = getattr(self, "_override_gain_m_inv", None) if override_value is not None else None
        state = getattr(self, "_pending_gain_cell", None)
        if state is None:
            state = {
                "z_index": iz, "population": population, "override": override,
                "initial_population": population[:, iz].copy(),
                "final_population": np.zeros_like(population[:, iz]),
                "initial_override": None if override is None else override[iz].copy(),
                "final_override": None if override is None else np.zeros_like(override[iz]),
                "fraction": 0.0,
            }
            self._pending_gain_cell = state
        if (state["z_index"] != iz or state["population"] is not population
                or state["override"] is not override or state["fraction"] + fraction > 1.0 + 1e-12):
            raise RuntimeError("Independent gain state crossed a cell or transport boundary")
        original = population[:, iz].copy()
        original_override = None if override is None else override[iz].copy()
        passive_before = self.passive_loss_energy_J_m3[iz].copy()
        population[:, iz] = state["initial_population"]
        if override is not None:
            override[iz] = state["initial_override"]
        try:
            output = self._gain_step_local(
                field, population, iz, step_length_m,
                population_cell_length_m=float(self.grid.dz) * fraction,
            )
            state["final_population"] += fraction * population[:, iz]
            if override is not None:
                state["final_override"] += fraction * override[iz]
            state["fraction"] += fraction
            return output
        except BaseException:
            self.passive_loss_energy_J_m3[iz] = passive_before
            self._pending_gain_cell = None
            raise
        finally:
            population[:, iz] = original
            if override is not None:
                override[iz] = original_override

    def _gain_step_local(
        self, field: np.ndarray, population: np.ndarray, iz: int, step_length_m: float,
        *, population_cell_length_m: float | None = None,
    ) -> np.ndarray:
        photon_energies = photon_energy(self.spectrum.coordinate)
        cross_section_sum = self.levels.signal_cross_section_sum_m2(
            self.spectrum.coordinate,
            population.shape[-2:],
            iz,
        )
        passive_loss = float(self.material.get("signal_passive_loss_m_inv", 0.0))
        override = self.levels_cfg.get("gain_override_m_inv")
        if override is None:
            override_gain = None
        else:
            reservoir = getattr(self, "_override_gain_m_inv", None)
            if reservoir is None or reservoir.shape != population.shape[1:]:
                reservoir = np.full(population.shape[1:], float(override), dtype=float)
                self._override_gain_m_inv = reservoir
            override_gain = reservoir[iz]
        cell_length_m = (float(self.grid.dz) if population_cell_length_m is None
                         else float(population_cell_length_m))
        saturation_cfg = self.levels_cfg.get("saturation_fluence_J_m2")
        if saturation_cfg is None:
            saturation_normal = np.divide(
                photon_energies[:, None, None],
                cross_section_sum,
                out=np.full_like(cross_section_sum, np.inf),
                where=cross_section_sum > 0.0,
            )
        else:
            saturation_normal = np.full_like(cross_section_sum, float(saturation_cfg))
        # A coherent local cell may be split into several z substeps.  Recover
        # the full slanted cell length before forming the projected-area
        # cosine; ordinary spectral/FN propagation simply uses one substep.
        cell_substeps = max(
            int(getattr(self, "_current_crystal_cell_substeps", 1)),
            1,
        )
        gain_group_size = max(
            int(getattr(self, "_current_gain_group_size", 1)),
            1,
        )
        effective_gain_substeps = cell_substeps / gain_group_size
        full_optical_cell_length = (
            float(step_length_m) * effective_gain_substeps
        )
        # Projection geometry belongs to the full parent cell even when the
        # population reservoir represents only one independent spatial group.
        path_cosine = float(self.grid.dz) / full_optical_cell_length
        if not 0.0 < path_cosine <= 1.0 + 1e-12:
            raise ValueError("Seed optical step is inconsistent with the axial cell length")
        path_cosine = min(path_cosine, 1.0)
        saturation_projected = saturation_normal * path_cosine
        use_saturation = bool(self.physics["gain_saturation"])
        passive_half_transmission = np.exp(
            -0.5 * passive_loss * step_length_m * self.crystal_mask
        )[None, :, :]
        channel_groups: tuple[np.ndarray, ...] = ()
        if use_saturation:
            channel_groups = self._simultaneous_gain_channel_groups()
            covered_channels = np.concatenate(channel_groups)
            if (
                len(covered_channels) != field.shape[1]
                or not np.array_equal(
                    np.sort(covered_channels), np.arange(field.shape[1])
                )
            ):
                raise RuntimeError(
                    "simultaneous gain channel groups must partition every channel exactly once"
                )

        for it in range(field.shape[0]):
            if it > 0:
                population[:, iz] = self._advance_population_between_temporal_bins(
                    population[:, iz],
                    it - 1,
                    it,
                    iz,
                )
            fluence = np.abs(field[it]) ** 2
            active_input_fluence = fluence * passive_half_transmission
            if not use_saturation:
                if override_gain is None:
                    active_gain = self.levels.signal_gain_spectrum_m_inv(
                        population[:, iz],
                        self.spectrum.coordinate,
                        z_index=iz,
                    )
                else:
                    active_gain = np.broadcast_to(override_gain, fluence.shape)
                exponent = active_gain * self.crystal_mask[None, :, :] * step_length_m
                self._maximum_unsaturated_gain_exponent = max(
                    getattr(self, "_maximum_unsaturated_gain_exponent", -np.inf),
                    float(np.max(exponent)),
                )
                self._positive_gain_exponent_clipping_active = bool(
                    getattr(self, "_positive_gain_exponent_clipping_active", False)
                    or np.any(exponent > 50.0)
                )
                gain_fluence = active_input_fluence * np.exp(
                    np.minimum(exponent, 50.0)
                )
            else:
                # All wavelength bins in one time bin share one gain reservoir.
                # A forward half-sweep followed by a reverse half-sweep is a
                # symmetric packet splitting. Spatial channels that represent
                # the same physical wavelength are first summed into one local
                # packet, update population once, and then receive the common
                # packet gain in proportion to their local input fluence.
                gain_fluence = np.zeros_like(fluence)
                forward = range(len(channel_groups))
                backward = range(len(channel_groups) - 1, -1, -1)
                for spectral_order in (forward, backward):
                    for group_index in spectral_order:
                        channels = channel_groups[group_index]
                        representative = int(channels[0])
                        channel_input = 0.5 * active_input_fluence[channels]
                        packet_in = np.sum(channel_input, axis=0)
                        if override_gain is None:
                            active_gain = self.levels.signal_gain_m_inv(
                                population[:, iz],
                                wavelength_m=float(
                                    self.spectrum.coordinate[representative]
                                ),
                                z_index=iz,
                            )
                        else:
                            active_gain = override_gain
                        exponent = active_gain * self.crystal_mask * step_length_m
                        valid = np.isfinite(saturation_projected[representative]) & (
                            saturation_projected[representative] > 0.0
                        )
                        fallback_exponent = np.where(valid, -np.inf, exponent)
                        self._maximum_unsaturated_gain_exponent = max(
                            getattr(self, "_maximum_unsaturated_gain_exponent", -np.inf),
                            float(np.max(fallback_exponent)),
                        )
                        self._positive_gain_exponent_clipping_active = bool(
                            getattr(self, "_positive_gain_exponent_clipping_active", False)
                            or np.any(fallback_exponent > 50.0)
                        )
                        packet_out = packet_in * np.exp(
                            np.minimum(exponent, 50.0)
                        )
                        if np.any(valid):
                            saturated = frantz_nodvik(
                                packet_in,
                                exponent,
                                np.where(
                                    valid,
                                    saturation_projected[representative],
                                    1.0,
                                ),
                            )
                            packet_out = np.where(valid, saturated, packet_out)

                        if override_gain is None and np.any(
                            cross_section_sum[representative] > 0.0
                        ):
                            requested_events = (packet_out - packet_in) / (
                                photon_energies[representative] * cell_length_m
                            )
                            updated, event_stats = (
                                self.levels.apply_signal_photon_events(
                                    population[:, iz],
                                    requested_events,
                                    float(
                                        self.spectrum.coordinate[representative]
                                    ),
                                    z_index=iz,
                                    input_photon_density_m3=(
                                        packet_in
                                        / (
                                            photon_energies[representative]
                                            * cell_length_m
                                        )
                                    ),
                                    interaction_length_m=step_length_m,
                                )
                            )
                            population[:, iz] = updated
                            events = np.asarray(
                                event_stats["applied_photon_events_m3"],
                                dtype=float,
                            )
                            packet_out = (
                                packet_in
                                + events
                                * photon_energies[representative]
                                * cell_length_m
                            )
                        elif override_gain is not None and np.any(valid):
                            depletion = np.divide(
                                packet_out - packet_in,
                                saturation_normal[representative] * cell_length_m,
                                out=np.zeros_like(packet_out),
                                where=valid,
                            )
                            override_gain -= depletion
                            if float(override) >= 0.0:
                                override_gain[:] = np.maximum(override_gain, 0.0)
                            else:
                                override_gain[:] = np.minimum(override_gain, 0.0)
                        packet_scale = np.divide(
                            np.maximum(packet_out, 0.0),
                            packet_in,
                            out=np.zeros_like(packet_out),
                            where=packet_in > np.finfo(float).tiny,
                        )
                        gain_fluence[channels] += (
                            channel_input * packet_scale[None, ...]
                        )
            output_fluence = np.maximum(
                gain_fluence * passive_half_transmission, 0.0
            )
            self.passive_loss_energy_J_m3[iz] += np.sum(
                np.maximum(fluence - active_input_fluence, 0.0)
                + np.maximum(gain_fluence - output_fluence, 0.0),
                axis=0,
            ) / max(self.grid.dz, np.finfo(float).tiny)
            scale = np.sqrt(
                output_fluence / np.maximum(fluence, np.finfo(float).tiny)
            )
            field[it] *= np.where(fluence > 0.0, scale, 0.0)
            if self.clip_crystal:
                field[it] *= self.crystal_mask[None, :, :]
        return field

    def _integral(self, field: np.ndarray) -> float:
        return float(np.sum(self.grid.integrate_transverse(np.abs(field) ** 2)))

    def _apply_crystal_aperture(self, field: np.ndarray) -> np.ndarray:
        before = self._integral(field)
        clipped = field * self.crystal_mask
        after = self._integral(clipped)
        self.crystal_aperture_loss_energy_J += max(before - after, 0.0)
        return clipped

    def _spectral_energy(self, field: np.ndarray) -> np.ndarray:
        return np.sum(self.grid.integrate_transverse(np.abs(field) ** 2), axis=0)

    def _temporal_energy(self, field: np.ndarray) -> np.ndarray:
        return np.sum(self.grid.integrate_transverse(np.abs(field) ** 2), axis=1)


class IncoherentHGSignalPropagator(SpectralSignalPropagator):
    """Propagate mutually incoherent HG channels with one shared population.

    Modes and wavelengths form a Cartesian channel basis.  The parent solver
    advances all channels together at every time/z cell, so gain saturation,
    aperture loss and thermal phase are mode-specific without introducing an
    arbitrary whole-pulse modal ordering.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        original_wavelengths = self.spectrum.coordinate.copy()
        original_weights = self.spectrum.weights.copy()
        original_joint = (
            None
            if self.joint_time_frequency_weights is None
            else self.joint_time_frequency_weights.copy()
        )
        self.incoherent_modes = incoherent_hg_modes(self.cfg)
        self._reported_wavelengths_m = original_wavelengths.copy()
        self._modal_channels: list[dict[str, Any]] = []
        wavelengths: list[float] = []
        weights: list[float] = []
        joint_columns: list[np.ndarray] = []
        for mode_index, mode in enumerate(self.incoherent_modes):
            for spectral_index, wavelength in enumerate(original_wavelengths):
                self._modal_channels.append(
                    {
                        "mode_index": mode_index,
                        "spectral_index": spectral_index,
                        **mode,
                    }
                )
                wavelengths.append(float(wavelength))
                weights.append(
                    float(mode["weight"]) * float(original_weights[spectral_index])
                )
                if original_joint is not None:
                    joint_columns.append(
                        original_joint[:, spectral_index]
                        * float(mode["weight"])
                    )
        self.spectrum = DiscreteProfile(
            np.asarray(wavelengths, dtype=float), np.asarray(weights, dtype=float)
        )
        if joint_columns:
            self.joint_time_frequency_weights = np.stack(joint_columns, axis=1)
        self._shared_gain_channel_groups = tuple(
            np.asarray(
                [
                    channel_index
                    for channel_index, channel in enumerate(self._modal_channels)
                    if int(channel["spectral_index"]) == spectral_index
                ],
                dtype=int,
            )
            for spectral_index in range(len(self._reported_wavelengths_m))
        )

    def _simultaneous_gain_channel_groups(self) -> tuple[np.ndarray, ...]:
        return self._shared_gain_channel_groups

    def propagate(
        self, population: np.ndarray, delta_n: np.ndarray
    ) -> SignalPropagationResult:
        result = super().propagate(population, delta_n)

        def aggregate(values: np.ndarray | None) -> np.ndarray | None:
            if values is None:
                return None
            source = np.asarray(values, dtype=float)
            aggregated = np.zeros(len(self._reported_wavelengths_m), dtype=float)
            for channel_index, channel in enumerate(self._modal_channels):
                aggregated[int(channel["spectral_index"])] += source[channel_index]
            return aggregated

        result.wavelengths_m = self._reported_wavelengths_m.copy()
        result.input_spectrum_J = aggregate(result.input_spectrum_J)
        result.output_spectrum_J = aggregate(result.output_spectrum_J)
        if result.source_spatio_spectral is not None:
            result.source_spatio_spectral.update(
                {
                    "spatial_mode_model": "incoherent_hg",
                    "spatial_mode_count": len(self.incoherent_modes),
                    "mode_spectral_channel_count": len(self._modal_channels),
                    "reported_spectrum_aggregated_over_modes": True,
                }
            )
        return result

    def _source_config_for_channel(
        self, channel_index: int, wavelength_m: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        local, diagnostics = seed_config_at_wavelength(self.cfg, wavelength_m)
        channel = self._modal_channels[channel_index]
        represented_m2_x = float(local.get("m2_x", 1.0))
        represented_m2_y = float(local.get("m2_y", 1.0))
        local["waist_x_m"] = float(local["waist_x_m"]) / math.sqrt(
            represented_m2_x
        )
        local["waist_y_m"] = float(local["waist_y_m"]) / math.sqrt(
            represented_m2_y
        )
        local["m2_x"] = 1.0
        local["m2_y"] = 1.0
        local["spatial_modes"] = {
            "model": "single_hg_mode",
            "active_order_x": int(channel["order_x"]),
            "active_order_y": int(channel["order_y"]),
        }
        diagnostics["incoherent_mode"] = {
            "mode_index": int(channel["mode_index"]),
            "order_x": int(channel["order_x"]),
            "order_y": int(channel["order_y"]),
            "mode_weight": float(channel["weight"]),
            "represented_input_m2_x": represented_m2_x,
            "represented_input_m2_y": represented_m2_y,
            "fundamental_waist_x_m": float(local["waist_x_m"]),
            "fundamental_waist_y_m": float(local["waist_y_m"]),
        }
        return local, diagnostics

    def _scan_output_caustic(
        self,
        field: np.ndarray,
        trace: OpticalTrace,
        quantity: str,
        wavelengths_m: np.ndarray,
    ) -> dict[str, Any] | None:
        """Scan explicit HG fields without applying represented M2 twice."""

        if not self._diagnostics_enabled() or bool(self.physics.get("_suppress_output_diagnostics", False)):
            return None
        config = self.cfg.get("caustic_diagnostics", {})
        if not bool(config.get("enabled", False)):
            return {
                "enabled": False,
                "application": "diagnostic_only",
                "affects_propagation": False,
            }
        retain = bool(config.get("record_planes", True)) and bool(
            self.physics.get("record_optical_planes", False)
        )
        diagnostics, planes = scan_spectral_caustic(
            field,
            self.grid,
            caustic_positions(config),
            np.asarray(wavelengths_m, dtype=float),
            float(config.get("refractive_index", 1.0)),
            np.ones(len(wavelengths_m), dtype=float),
            np.ones(len(wavelengths_m), dtype=float),
            retain_fields=retain,
        )
        if retain:
            for index, (position, plane) in enumerate(planes):
                trace.add(
                    f"caustic_{index:03d}_z_{position:+.9g}_m",
                    plane,
                    self.grid,
                    quantity,
                    complex_field_available=False,
                )
        diagnostics.update(
            {
                "method": (
                    "incoherent_hg_channel_energy_integrated_D4sigma_fit"
                ),
                "planes_recorded": bool(retain),
                "physical_wavelengths_m": self._reported_wavelengths_m.tolist(),
                "spatial_mode_count": len(self.incoherent_modes),
                "mode_spectral_channel_count": len(self._modal_channels),
                "per_channel_transport_M2_x": 1.0,
                "per_channel_transport_M2_y": 1.0,
                "represented_beam_quality_source": (
                    "explicit_incoherent_hg_orders_and_weights"
                ),
                "scope": (
                    "Every explicit HG/wavelength channel is propagated with "
                    "transport M2=1 and summed in intensity at each plane. "
                    "Higher-order divergence is already carried by the sampled "
                    "HG fields and is not multiplied by the configured ensemble M2."
                ),
            }
        )
        return diagnostics

    def _model_used(self) -> str:
        return "incoherent_hg_shared_reservoir"

    def _gain_transport_diagnostics(self) -> dict[str, Any]:
        result = super()._gain_transport_diagnostics()
        result.update(
            {
                "incoherent_spatial_modes": True,
                "spatial_mode_count": len(self.incoherent_modes),
                "mode_spectral_channel_count": len(self._modal_channels),
                "spectral_bins": len(self._reported_wavelengths_m),
                "modal_coupling": "simultaneous_local_shared_population",
                "modal_gain_update": (
                    "sum_same_wavelength_local_fluence_then_one_population_update"
                ),
            }
        )
        return result
