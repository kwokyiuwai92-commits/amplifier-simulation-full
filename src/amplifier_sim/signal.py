"""Seed optical train, gain-medium traversal, saturation, and multipass routing."""

from __future__ import annotations

from .spectral_volume import SpectralIntensityVolume

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .profiling import operation_profiler_enabled, profile_operation

from .caustic import caustic_positions, scan_sampled_caustic
from .constants import LIGHT_SPEED, photon_energy
from .dispersion import material_group_index, material_refractive_index
from .flight_time import optical_train_flight_time
from .grid import Grid3D
from .geometry import (
    crystal_aperture_clipping_required,
    crystal_transverse_mask,
)
from .levels import LevelSystem
from .optical_system import (
    additive_source_gain_angle_warning,
    build_external_field,
    gain_interface,
    gain_medium_axis_angles_rad,
    gain_medium_path_cosine,
)
from .optics import OpticalTrace, angular_spectrum, propagate_fft_train, shift_field
from .pass_routing import resolve_pass_reverse
from .polarization import (
    apply_thermal_birefringence,
    initialize_jones_field,
    jones_enabled,
    polarization_diagnostics,
    total_transverse_density,
)
from .seed_spatio_spectral import sampled_field_moments, seed_config_at_wavelength
from .seed_gain_transport import (
    GAUSSIAN_MODE_OVERLAP,
    apply_modal_overlap_gain,
    project_to_gaussian_mode,
    spatial_gain_transport_model,
)


@dataclass
class SignalPropagationResult:
    fluence_J_m2: np.ndarray
    population_m3: np.ndarray
    passes: list[dict[str, Any]]
    source_grid_energy_J: float
    system_output_energy_J: float
    trace: OpticalTrace
    warnings: list[str]
    wavelengths_m: np.ndarray | None = None
    input_spectrum_J: np.ndarray | None = None
    output_spectrum_J: np.ndarray | None = None
    time_s: np.ndarray | None = None
    input_temporal_energy_J: np.ndarray | None = None
    output_temporal_energy_J: np.ndarray | None = None
    model_used: str | None = None
    frequency_Hz: np.ndarray | None = None
    input_spectral_phase_rad: np.ndarray | None = None
    output_spectral_phase_rad: np.ndarray | None = None
    compressed_spectral_phase_rad: np.ndarray | None = None
    compressed_temporal_power_W: np.ndarray | None = None
    cpa_diagnostics: dict[str, Any] | None = None
    extracted_energy_J_m3: np.ndarray | None = None
    grid_coupling: dict[str, Any] | None = None
    gain_transport: dict[str, Any] | None = None
    average_spectral_intensity_W_m2: np.ndarray | SpectralIntensityVolume | None = None
    fluence_last_pass_J_m2: np.ndarray | None = None
    fluence_cumulative_J_m2: np.ndarray | None = None
    fluence_per_pass_J_m2: np.ndarray | None = None
    source_spatio_spectral: dict[str, Any] | None = None
    nonlinear_loss_energy_J_m3: np.ndarray | None = None
    passive_loss_energy_J_m3: np.ndarray | None = None
    crystal_aperture_loss_energy_J: float = 0.0
    external_loss_ledger: list[dict[str, Any]] | None = None
    caustic_diagnostics: dict[str, Any] | None = None
    polarization_diagnostics: dict[str, Any] | None = None
    output_jones_field: np.ndarray | None = None
    initial_active_gain_m_inv: np.ndarray | None = None
    remaining_active_gain_m_inv: np.ndarray | None = None
    passive_loss_m_inv: np.ndarray | None = None
    net_gain_m_inv: np.ndarray | None = None
    free_carrier_density_m3: np.ndarray | None = None
    output_field_diagnostics: dict[str, Any] | None = None
    longitudinal_intensity_W_m2: np.ndarray | None = None
    longitudinal_intensity_area_m2: np.ndarray | None = None
    longitudinal_temporal_power_W: np.ndarray | None = None
    temporal_bin_edges_s: np.ndarray | None = None
    spectral_bin_edges_Hz: np.ndarray | None = None


def gain_entrance_energy_partition(
    *,
    configured_source_energy_J: float,
    sampled_gain_entrance_quantity: float,
    wavelength_samples: list[dict[str, Any]],
    quantity_to_energy: float = 1.0,
) -> dict[str, Any] | None:
    """Separate physical pre-gain loss from finite-window truncation.

    ABCD transport represents an infinite-plane field and samples it only at
    the gain entrance.  That numerical omission must not be mislabeled as an
    optical routing loss.  Quantities may be energies or powers; the latter
    are converted with ``quantity_to_energy=1/repetition_rate``.
    """

    entrances = [
        sample.get("gain_entrance", {})
        for sample in wavelength_samples
        if isinstance(sample, dict)
    ]
    analytic_quantities = [
        entrance.get("analytic_quantity") for entrance in entrances
    ]
    if not entrances or any(value is None for value in analytic_quantities):
        return None

    scale = float(quantity_to_energy)
    analytic_energy = scale * sum(float(value) for value in analytic_quantities)
    sampled_energy = scale * float(sampled_gain_entrance_quantity)
    diagnostic_sampled_energy = scale * sum(
        float(
            entrance.get(
                "sampled_quantity",
                float(entrance["analytic_quantity"])
                * float(entrance.get("sampled_fraction", 1.0)),
            )
        )
        for entrance in entrances
    )
    fractions = [
        float(entrance["sampled_fraction"])
        for entrance in entrances
        if entrance.get("sampled_fraction") is not None
    ]
    if analytic_energy > 0.0:
        capture_fraction = sampled_energy / analytic_energy
    elif fractions:
        capture_fraction = min(fractions)
    else:
        capture_fraction = None

    return {
        "configured_source_energy_J": float(configured_source_energy_J),
        "analytic_gain_entrance_energy_J": float(analytic_energy),
        "sampled_gain_entrance_energy_J": float(sampled_energy),
        "gain_entrance_capture_fraction": (
            None if capture_fraction is None else float(capture_fraction)
        ),
        "minimum_wavelength_gain_entrance_capture_fraction": (
            min(fractions) if fractions else capture_fraction
        ),
        "numerical_window_truncation_J": float(
            analytic_energy - sampled_energy
        ),
        "physical_external_loss_J": float(
            configured_source_energy_J - analytic_energy
        ),
        "diagnostic_sampled_gain_entrance_energy_J": float(
            diagnostic_sampled_energy
        ),
        "sampled_energy_diagnostic_residual_J": float(
            sampled_energy - diagnostic_sampled_energy
        ),
        "loss_partition_identity": (
            "configured_source_energy_J = physical_external_loss_J + "
            "analytic_gain_entrance_energy_J; analytic_gain_entrance_energy_J = "
            "sampled_gain_entrance_energy_J + numerical_window_truncation_J"
        ),
    }


def append_gain_entrance_loss_terms(
    ledger: list[dict[str, Any]], partition: dict[str, Any]
) -> None:
    """Append separately named physical and numerical pre-gain losses."""

    configured = float(partition["configured_source_energy_J"])
    analytic = float(partition["analytic_gain_entrance_energy_J"])
    sampled = float(partition["sampled_gain_entrance_energy_J"])
    ledger.extend(
        [
            {
                "name": "source_to_gain_physical_optical_train",
                "category": "physical_external_loss",
                "input_energy_J": configured,
                "output_energy_J": analytic,
                "signed_loss_energy_J": configured - analytic,
                "included_in_seed_balance": True,
            },
            {
                "name": "gain_entrance_numerical_window_truncation",
                "category": "numerical_window_truncation",
                "input_energy_J": analytic,
                "output_energy_J": sampled,
                "signed_loss_energy_J": analytic - sampled,
                "included_in_seed_balance": True,
            },
        ]
    )


def report_signal_transport_progress(
    owner: Any,
    *,
    current: int,
    total: int,
    pass_index: int,
    pass_total: int,
    z_index: int | None,
    metrics: dict[str, Any] | None = None,
    phase: str = "crystal_transport",
    fraction_start: float = 0.0,
    fraction_stop: float = 1.0,
    fraction_override: float | None = None,
) -> None:
    """Publish bounded setup/axial/output updates for an observed Seed solve."""

    callback = getattr(owner, "_transport_progress_callback", None)
    if callback is None:
        return
    stride = max(int(math.ceil(max(int(total), 1) / 100.0)), 1)
    if current not in {0, total} and (current - 1) % stride:
        return
    local_fraction = float(current) / max(int(total), 1)
    mapped_fraction = (
        float(fraction_override)
        if fraction_override is not None
        else float(fraction_start)
        + (float(fraction_stop) - float(fraction_start)) * local_fraction
    )
    try:
        callback(
            {
                "current": int(current),
                "total": int(total),
                "fraction": mapped_fraction,
                "pass_index": int(pass_index),
                "pass_total": int(pass_total),
                "z_index": None if z_index is None else int(z_index),
                "phase": str(phase),
                "backend": type(owner).__name__,
                "metrics": dict(metrics or {}),
            }
        )
    except Exception:
        # Progress is observational and must not change a physical result.
        owner._transport_progress_callback = None


def _log_expm1(value: np.ndarray) -> np.ndarray:
    result = np.empty_like(value)
    large = value > 50.0
    result[large] = value[large]
    positive = (~large) & (value > 0.0)
    result[positive] = np.log(np.expm1(value[positive]))
    result[~large & ~positive] = -np.inf
    return result


def frantz_nodvik(fluence: np.ndarray, gain_exponent: np.ndarray, saturation: float) -> np.ndarray:
    if not operation_profiler_enabled():
        scaled = np.maximum(fluence, 0.0) / saturation
        logarithm = np.logaddexp(0.0, gain_exponent + _log_expm1(scaled))
        return saturation * logarithm
    with profile_operation("signal.frantz_nodvik", array=fluence):
        scaled = np.maximum(fluence, 0.0) / saturation
        logarithm = np.logaddexp(0.0, gain_exponent + _log_expm1(scaled))
        return saturation * logarithm


def reference_ray_warnings(
    gain_marker: dict[str, Any],
    gain_markers: list[dict[str, Any]],
    post_elements: list[dict[str, Any]],
    between_pass_elements: list[dict[str, Any]],
    between_pass_segments: list[list[dict[str, Any]]],
) -> list[str]:
    """Describe the current boundary of explicit gain-marker walk-off."""

    markers = gain_markers if gain_markers else [gain_marker]
    angled = any(
        abs(float(marker.get("incidence_angle_x_deg", 0.0)))
        + abs(float(marker.get("incidence_angle_y_deg", 0.0)))
        > 0.0
        for marker in markers
    )
    if not angled:
        return []
    warnings: list[str] = []
    maximum_angle = max(
        max(
            abs(float(marker.get("incidence_angle_x_deg", 0.0))),
            abs(float(marker.get("incidence_angle_y_deg", 0.0))),
        )
        for marker in markers
    )
    if maximum_angle > 10.0:
        warnings.append(
            "A gain_medium incidence angle exceeds 10 degrees. The z-marching FFT model uses "
            "paraxial diffraction plus explicit walk-off, not a rotated-plane/nonparaxial field "
            "solver; verify angle convergence or use ray/ABCD propagation where appropriate."
        )
    directional_kinds = {
        "free_space",
        "thin_lens",
        "double_pass_lens_relay",
        "thick_lens",
        "interface",
        "tilt",
        "coordinate_break",
        "mirror",
        "phase_screen",
        "dispersive",
    }
    routed_elements = [*post_elements, *between_pass_elements]
    for segment in between_pass_segments:
        routed_elements.extend(segment)
    if any(element.get("type") in directional_kinds for element in routed_elements):
        warnings.append(
            "gain_medium incidence angles are transported as alias-free walk-off inside each crystal pass, "
            "but the current sampled-field backend does not carry that marker's reference-ray angle through "
            "downstream/between-pass optical elements. Use explicit coordinate breaks/beam centering for those "
            "external segments and verify their plane centroids; this is a known paraxial routing limitation."
        )
    return warnings


class SignalPropagator:
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
        self._positive_gain_exponent_clipping_active = False
        self._maximum_unsaturated_gain_exponent = -np.inf
        self.photoelastic_tensor_fields: dict[str, np.ndarray] | None = None
        self._endface_surface_sag_m: dict[str, np.ndarray | None] = {
            "front": None,
            "back": None,
        }

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

    def _reference_ray_warnings(self) -> list[str]:
        warnings = reference_ray_warnings(
            self.gain_marker,
            self.gain_markers,
            self.post_elements,
            self.between_pass_elements,
            self.between_pass_segments,
        )
        angle_warning = additive_source_gain_angle_warning(
            self.cfg,
            self.gain_markers if self.gain_markers else [self.gain_marker],
            "Seed",
        )
        if angle_warning is not None:
            warnings.append(angle_warning)
        return warnings

    def _diagnostics_enabled(self) -> bool:
        from .result_recording import diagnostic_observers_enabled
        return diagnostic_observers_enabled(self)

    def _scan_output_caustic(
        self,
        field: np.ndarray,
        trace: OpticalTrace,
        quantity: str,
        wavelengths_m: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        """Run the optional monochromatic output-caustic diagnostic."""

        if not self._diagnostics_enabled() or bool(self.physics.get("_suppress_output_diagnostics", False)):
            return None
        config = self.cfg.get("caustic_diagnostics", {})
        if not bool(config.get("enabled", False)):
            return {
                "enabled": False,
                "application": "diagnostic_only",
                "affects_propagation": False,
            }
        wavelengths = (
            np.asarray([self.cfg["wavelength_m"]], dtype=float)
            if wavelengths_m is None
            else np.asarray(wavelengths_m, dtype=float)
        )
        if len(wavelengths) != 1:
            return {
                "enabled": False,
                "requested": True,
                "application": "diagnostic_only",
                "affects_propagation": False,
                "reason": (
                    "output caustic diagnostics currently require one sampled Seed wavelength; "
                    "a broadband intensity caustic needs wavelength-resolved propagation and an explicit chromatic M2 definition"
                ),
                "warnings": [
                    "Caustic scan was not approximated with the centre wavelength because that would hide chromatic beam-size and focus differences."
                ],
            }
        retain = bool(config.get("record_planes", True)) and bool(
            self.physics.get("record_optical_planes", False)
        )
        diagnostics, planes = scan_sampled_caustic(
            field,
            self.grid,
            caustic_positions(config),
            float(wavelengths[0]),
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
                )
        diagnostics["planes_recorded"] = bool(retain)
        return diagnostics

    def propagate(self, population: np.ndarray, delta_n: np.ndarray) -> SignalPropagationResult:
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
        mode = self.physics["optical_propagation"]
        spatial_gain_model = spatial_gain_transport_model(self.cfg)
        modal_axial_samples: list[dict[str, Any]] = []
        modal_projection_overlaps: list[float] = []
        external_mode = "abcd" if mode in {"abcd", "hybrid_abcd_fft"} else mode
        record_planes = bool(self.physics["record_optical_planes"]) and self._diagnostics_enabled()
        local_cfg, source_spatio = seed_config_at_wavelength(
            self.cfg, float(self.cfg["wavelength_m"])
        )
        entrance_diagnostics: dict[str, Any] = {}
        external = build_external_field(
            local_cfg,
            float(self.cfg["pulse_energy_J"]),
            self.pre_elements,
            self.grid,
            external_mode,
            "energy_J",
            self.physics["interface_model"],
            record_planes,
            include_abcd_tilt_phase=mode in {"abcd", "hybrid_abcd_fft"},
            abcd_diagnostics=entrance_diagnostics,
        )
        if not entrance_diagnostics:
            entrance_diagnostics.update(sampled_field_moments(external.field, self.grid))
            entrance_diagnostics["tilt_phase_included"] = bool(
                local_cfg.get("incidence_angle_x_deg", 0.0)
                or local_cfg.get("incidence_angle_y_deg", 0.0)
            )
        source_spatio["gain_entrance"] = entrance_diagnostics
        field = external.field
        self._jones = jones_enabled(self.cfg)
        if self._jones:
            field = initialize_jones_field(field, self.cfg)
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
            configured_source_energy_J=float(self.cfg["pulse_energy_J"]),
            sampled_gain_entrance_quantity=self._integral(field),
            wavelength_samples=[source_spatio],
        )
        if gain_entrance_partition is None:
            record_external_loss(
                "source_to_gain_optical_train",
                float(external.source_grid_quantity),
                self._integral(field),
                "pre_gain_routing",
            )
        else:
            append_gain_entrance_loss_terms(
                external_loss_ledger, gain_entrance_partition
            )
        accumulated_retardance = np.zeros(self.grid.shape[-2:], dtype=float)
        current_outside_n = external.refractive_index
        n_gain = material_refractive_index(self.material, float(self.cfg["wavelength_m"]), role="signal")
        wavelength = float(self.cfg["wavelength_m"])
        trace = external.trace
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
        )
        for pass_number in range(passes):
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
            # Keep the currently traversed crystal marker available to
            # specialised gain backends.  Most signal solvers do not need
            # this state, but the full JOSA branch uses each pass' own signed
            # incidence angles for transverse walk-off.  Setting it here
            # prevents a multi-pass optical bench from silently reusing the
            # first/default marker on every pass.
            self._active_gain_marker = marker
            self._active_gain_reverse = reverse
            entrance_side = "back" if reverse else "front"
            exit_side = "front" if reverse else "back"
            interface_input = self._integral(field)
            field = gain_interface(
                field,
                current_outside_n,
                n_gain,
                marker,
                entrance_side,
                self.physics["interface_model"],
                wavelength,
                entering=True,
                surface_sag_m=self._endface_surface_sag_m[entrance_side],
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
            angle_x, angle_y = gain_medium_axis_angles_rad(marker)
            path_cosine = gain_medium_path_cosine(marker)
            step_length = self.grid.dz / path_cosine
            indices = list(reversed(range(len(self.grid.z)))) if reverse else list(range(len(self.grid.z)))
            for cell_number, iz in enumerate(indices, start=1):
                if mode in {"abcd", "fft", "hybrid_abcd_fft"}:
                    field = angular_spectrum(
                        field,
                        self.grid,
                        step_length / 2.0,
                        wavelength,
                        n_gain,
                        float(self.cfg.get("m2_x", 1.0)),
                        float(self.cfg.get("m2_y", 1.0)),
                    )
                    if angle_x != 0.0 or angle_y != 0.0:
                        field = shift_field(
                            field,
                            self.grid,
                            0.5 * self.grid.dz * math.tan(angle_x),
                            0.5 * self.grid.dz * math.tan(angle_y),
                        )
                    if self.clip_crystal:
                        field = self._apply_crystal_aperture(field)
                if self._jones:
                    field, local_retardance = apply_thermal_birefringence(
                        field,
                        delta_n[iz],
                        step_length,
                        wavelength,
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
                    field *= np.exp(
                        1j
                        * (2.0 * math.pi / wavelength)
                        * delta_n[iz]
                        * step_length
                    )
                fluence_in = total_transverse_density(
                    field, is_jones=self._jones
                )
                if spatial_gain_model == GAUSSIAN_MODE_OVERLAP:
                    projection = project_to_gaussian_mode(field, self.grid)
                    field = projection.field
                    modal_projection_overlaps.extend(
                        float(value)
                        for value in np.ravel(projection.coherent_overlap)
                    )
                    if self.clip_crystal:
                        field = self._apply_crystal_aperture(field)
                    fluence_in = total_transverse_density(
                        field, is_jones=self._jones
                    )
                    active_gain = self._active_gain_map(population, iz, fluence_in.shape)
                    modal_step = apply_modal_overlap_gain(
                        field,
                        active_gain,
                        float(self.material.get("signal_passive_loss_m_inv", 0.0))
                        * self.crystal_mask,
                        self.grid,
                        step_length,
                        self.grid.dz,
                    )
                    field = modal_step.field
                    if self.clip_crystal:
                        field *= self.crystal_mask
                    extracted_energy[iz] += modal_step.active_transfer_per_volume
                    self.passive_loss_energy_J_m3[iz] += modal_step.passive_loss_per_volume
                    raw_exponent = float(np.max(modal_step.raw_gain_exponent))
                    self._maximum_unsaturated_gain_exponent = max(
                        self._maximum_unsaturated_gain_exponent,
                        raw_exponent,
                    )
                    self._positive_gain_exponent_clipping_active |= raw_exponent > 50.0
                    if bool(
                        self.cfg.get("spatial_gain_transport", {}).get(
                            "record_axial_diagnostics", True
                        )
                    ):
                        moments = sampled_field_moments(field, self.grid)
                        modal_axial_samples.append(
                            {
                                "pass": pass_number + 1,
                                "direction": "backward" if reverse else "forward",
                                "z_index": int(iz),
                                "z_m": float(self.grid.z[iz]),
                                "input_energy_J": float(
                                    self.grid.integrate_transverse(
                                        modal_step.intensity_before
                                    )
                                ),
                                "output_energy_J": float(
                                    self.grid.integrate_transverse(
                                        modal_step.intensity_after
                                    )
                                ),
                                "effective_active_gain_m_inv": float(
                                    modal_step.effective_active_gain_m_inv
                                ),
                                "effective_passive_loss_m_inv": float(
                                    modal_step.effective_passive_loss_m_inv
                                ),
                                "effective_total_gain_m_inv": float(
                                    modal_step.effective_total_gain_m_inv
                                ),
                                "gain_exponent": float(modal_step.gain_exponent),
                                "raw_gain_exponent": float(
                                    modal_step.raw_gain_exponent
                                ),
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
                    passive_before = self.passive_loss_energy_J_m3[iz].copy()
                    field = self._gain_step(field, fluence_in, population, iz, step_length)
                    fluence_after_gain_and_loss = total_transverse_density(
                        field, is_jones=self._jones
                    )
                    # The gain operator uses a passive half-step on each side
                    # of the saturated active map.  Recover the active
                    # transfer from the exact discrete balance rather than by
                    # pretending that all passive loss occurred downstream.
                    extracted_energy[iz] += (
                        fluence_after_gain_and_loss - fluence_in
                    ) / self.grid.dz + (
                        self.passive_loss_energy_J_m3[iz] - passive_before
                    )
                if mode in {"abcd", "fft", "hybrid_abcd_fft"}:
                    field = angular_spectrum(
                        field,
                        self.grid,
                        step_length / 2.0,
                        wavelength,
                        n_gain,
                        float(self.cfg.get("m2_x", 1.0)),
                        float(self.cfg.get("m2_y", 1.0)),
                    )
                    if angle_x != 0.0 or angle_y != 0.0:
                        field = shift_field(
                            field,
                            self.grid,
                            0.5 * self.grid.dz * math.tan(angle_x),
                            0.5 * self.grid.dz * math.tan(angle_y),
                        )
                    if self.clip_crystal:
                        field = self._apply_crystal_aperture(field)
                if spatial_gain_model == GAUSSIAN_MODE_OVERLAP:
                    projection = project_to_gaussian_mode(field, self.grid)
                    field = projection.field
                    modal_projection_overlaps.extend(
                        float(value)
                        for value in np.ravel(projection.coherent_overlap)
                    )
                    if self.clip_crystal:
                        field = self._apply_crystal_aperture(field)
                pass_signal_volume[iz] = total_transverse_density(
                    field, is_jones=self._jones
                )
                completed_cells = pass_number * len(indices) + cell_number
                report_signal_transport_progress(
                    self,
                    current=completed_cells,
                    total=transport_total,
                    pass_index=pass_number + 1,
                    pass_total=passes,
                    z_index=iz,
                    metrics={"transport_energy_J": self._integral(field)},
                )
            signal_volume = pass_signal_volume
            cumulative_signal_volume += pass_signal_volume
            if volume_recording == "per_pass":
                per_pass_signal_volumes.append(pass_signal_volume.copy())
            gain_output = self._integral(field)
            if record_planes:
                trace.add(f"pass_{pass_number + 1}_gain_exit", field, self.grid, "energy_J")
            outside_n = float(marker.get("outside_refractive_index", current_outside_n))
            interface_input = self._integral(field)
            field = gain_interface(
                field,
                n_gain,
                outside_n,
                marker,
                exit_side,
                self.physics["interface_model"],
                wavelength,
                entering=False,
                surface_sag_m=self._endface_surface_sag_m[exit_side],
            )
            record_external_loss(
                f"pass_{pass_number + 1}_{exit_side}_gain_interface",
                interface_input,
                self._integral(field),
                "gain_interface",
            )
            current_outside_n = outside_n
            crystal_output = self._integral(field)
            pass_stats.append(
                {
                    "pass": pass_number + 1,
                    "direction": "backward" if reverse else "forward",
                    "incidence_angle_x_deg": math.degrees(angle_x),
                    "incidence_angle_y_deg": math.degrees(angle_y),
                    "effective_path_length_m": float(self.grid.z[-1] - self.grid.z[0] + self.grid.dz) / path_cosine,
                    "gain_input_J": energy_in,
                    "gain_output_J": gain_output,
                    "crystal_exit_J": crystal_output,
                    "crystal_aperture_loss_J": (
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
                between_elements = (
                    self.between_pass_segments[pass_number]
                    if self.gain_markers and pass_number < len(self.between_pass_segments)
                    else self.between_pass_elements
                )
                flight_initial_n = current_outside_n
                if between_elements:
                    train_input = self._integral(field)
                    field, current_outside_n, local_trace = propagate_fft_train(
                        field,
                        between_elements,
                        self.grid,
                        wavelength,
                        current_outside_n,
                        "energy_J",
                        self.physics["interface_model"],
                        float(self.cfg.get("m2_x", 1.0)),
                        float(self.cfg.get("m2_y", 1.0)),
                        record_planes,
                    )
                    trace.extend(local_trace, f"between_{pass_number + 1}_{pass_number + 2}_")
                    record_external_loss(
                        f"between_pass_{pass_number + 1}_{pass_number + 2}_optical_train",
                        train_input,
                        self._integral(field),
                        "between_pass_elements",
                    )
                flight = optical_train_flight_time(
                    between_elements,
                    initial_refractive_index=flight_initial_n,
                    initial_group_refractive_index=marker.get("outside_group_refractive_index"),
                    extra_delay_s=float(
                        self.simulation.get("unmodeled_between_pass_delay_s", 0.0)
                    ),
                )
                crystal_group_delay = (
                    material_group_index(
                        self.material,
                        wavelength,
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
                if (
                    state_delay > 0.0
                    and self.levels_cfg.get("gain_override_m_inv") is None
                ):
                    population[:] = self.levels.advance_dark_implicit(
                        population,
                        state_delay,
                    )
                    pass_stats[-1]["between_pass_population_relaxation"] = True
                else:
                    pass_stats[-1]["between_pass_population_relaxation"] = False
                if self.simulation.get("flip_x_between_passes", False):
                    field = np.flip(field, axis=-1).copy()
        if self.post_elements:
            post_input = self._integral(field)
            field, current_outside_n, post_trace = propagate_fft_train(
                field,
                self.post_elements,
                self.grid,
                wavelength,
                current_outside_n,
                "energy_J",
                self.physics["interface_model"],
                float(self.cfg.get("m2_x", 1.0)),
                float(self.cfg.get("m2_y", 1.0)),
                bool(self.physics["record_optical_planes"]),
            )
            trace.extend(post_trace, "post_")
            record_external_loss(
                "post_gain_optical_train",
                post_input,
                self._integral(field),
                "post_gain_routing",
            )
        if record_planes:
            trace.add("system_output", field, self.grid, "energy_J")
        output_energy = self._integral(field)
        if pass_stats:
            pass_stats[-1]["system_output_J"] = output_energy
        caustic_diagnostics = self._scan_output_caustic(
            field, trace, "energy_J"
        )
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
        gain_warnings: list[str] = []
        if self._positive_gain_exponent_clipping_active:
            gain_warnings.append(
                "At least one positive unsaturated Seed single-cell gain exponent exceeded 50 "
                "and was clipped before exponentiation. Enable saturation or refine Nz before "
                "using this result quantitatively."
            )
        if (
            spatial_gain_model == GAUSSIAN_MODE_OVERLAP
            and modal_projection_overlaps
            and min(modal_projection_overlaps) < 0.99
        ):
            gain_warnings.append(
                "Gaussian-mode overlap discarded non-Gaussian spatial content; the minimum "
                f"coherent projection overlap was {min(modal_projection_overlaps):.6g}. "
                "Compare with local_field before treating this reduced-order result as physical."
            )
        return SignalPropagationResult(
            fluence_J_m2=signal_volume,
            population_m3=population,
            passes=pass_stats,
            source_grid_energy_J=external.source_grid_quantity,
            system_output_energy_J=output_energy,
            trace=trace,
            warnings=[
                *external.warnings,
                *self.bench_warnings,
                *self._reference_ray_warnings(),
                *gain_warnings,
            ],
            extracted_energy_J_m3=extracted_energy,
            gain_transport=(
                {
                    "backend": GAUSSIAN_MODE_OVERLAP,
                    "status": "applied",
                    "spatial_closure": "fixed_transported_mode",
                    "gain_guiding": False,
                    "thermal_phase_propagated": bool(np.any(delta_n != 0.0)),
                    "gaussian_projection": "power_preserving_moment_match_each_z_cell",
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
                    "axial_samples": modal_axial_samples,
                    "maximum_cell_gain_exponent": float(
                        self._maximum_unsaturated_gain_exponent
                    ),
                    "positive_cell_exponent_clipping_active": bool(
                        self._positive_gain_exponent_clipping_active
                    ),
                }
                if spatial_gain_model == GAUSSIAN_MODE_OVERLAP
                else None
            ),
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
                "model": source_spatio["model"],
                "reference_wavelength_m": source_spatio["reference_wavelength_m"],
                "wavelength_samples": [source_spatio],
                **(gain_entrance_partition or {}),
            },
            passive_loss_energy_J_m3=self.passive_loss_energy_J_m3.copy(),
            crystal_aperture_loss_energy_J=self.crystal_aperture_loss_energy_J,
            external_loss_ledger=external_loss_ledger,
            caustic_diagnostics=caustic_diagnostics,
            polarization_diagnostics=output_polarization,
            output_jones_field=field.copy() if self._jones else None,
        )

    def _active_gain_map(
        self,
        population: np.ndarray,
        iz: int,
        transverse_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Return the unsaturated active gain map for one monochromatic z cell."""

        override = self.levels_cfg.get("gain_override_m_inv")
        if override is not None:
            reservoir = getattr(self, "_override_gain_m_inv", None)
            if reservoir is None or reservoir.shape != population.shape[1:]:
                reservoir = np.full(population.shape[1:], float(override), dtype=float)
                self._override_gain_m_inv = reservoir
            return np.broadcast_to(reservoir[iz] * self.crystal_mask, transverse_shape)
        active_gain = self.levels.signal_gain_m_inv(
            population[:, iz],
            wavelength_m=float(self.cfg["wavelength_m"]),
            z_index=iz,
        )
        return active_gain * self.crystal_mask

    def _apply_crystal_aperture(self, field: np.ndarray) -> np.ndarray:
        before = self._integral(field)
        clipped = field * self.crystal_mask
        after = self._integral(clipped)
        self.crystal_aperture_loss_energy_J += max(before - after, 0.0)
        return clipped

    def _gain_step(
        self,
        field: np.ndarray,
        fluence: np.ndarray,
        population: np.ndarray,
        iz: int,
        step_length_m: float | None = None,
    ) -> np.ndarray:
        step_length = self.grid.dz if step_length_m is None else float(step_length_m)
        passive_half_transmission = np.exp(
            -0.5
            * float(self.material.get("signal_passive_loss_m_inv", 0.0))
            * step_length
            * self.crystal_mask
        )
        active_input_fluence = fluence * passive_half_transmission
        override = self.levels_cfg.get("gain_override_m_inv")
        multi_transition = (
            override is None and len(self.levels.signal_transitions) > 1
        )
        sigma_abs, sigma_em = self.levels.local_cross_sections(
            "signal",
            np.asarray([float(self.cfg["wavelength_m"])]),
            fluence.shape,
            iz,
        )
        cross_section_sum = sigma_abs[0] + sigma_em[0]
        if multi_transition:
            cross_section_sum = np.zeros_like(fluence, dtype=float)
            for transition in self.levels.signal_transitions:
                transition_abs, transition_em = (
                    self.levels.local_transition_cross_sections(
                        transition,
                        float(self.cfg["wavelength_m"]),
                        fluence.shape,
                        iz,
                    )
                )
                cross_section_sum += transition_abs[0] + transition_em[0]
        if override is None:
            active_gain = self.levels.signal_gain_m_inv(
                population[:, iz],
                wavelength_m=float(self.cfg["wavelength_m"]),
                z_index=iz,
            )
        else:
            reservoir = getattr(self, "_override_gain_m_inv", None)
            if reservoir is None or reservoir.shape != population.shape[1:]:
                reservoir = np.full(population.shape[1:], float(override), dtype=float)
                self._override_gain_m_inv = reservoir
            active_gain = reservoir[iz]
        active_gain *= self.crystal_mask
        exponent = active_gain * step_length
        saturation = self.levels_cfg.get("saturation_fluence_J_m2")
        if saturation is None:
            saturation_normal = np.divide(
                photon_energy(float(self.cfg["wavelength_m"])),
                cross_section_sum,
                out=np.full_like(cross_section_sum, np.inf),
                where=cross_section_sum > 0.0,
            )
        else:
            saturation_normal = np.full_like(cross_section_sum, float(saturation))
        # ``fluence`` is sampled per fixed x/y projected area.  Frantz--Nodvik
        # is written per area normal to the propagation direction, for which
        # F_perp = F_xy / cos(theta).  Keeping the field on the x/y grid is
        # therefore equivalent to using Fsat_xy = cos(theta) * Fsat.  The
        # population/override depletion below deliberately retains the
        # beam-normal saturation because it is converted with the physical
        # axial cell volume.
        path_cosine = float(self.grid.dz) / step_length
        if not 0.0 < path_cosine <= 1.0 + 1e-12:
            raise ValueError("Seed optical step is inconsistent with the axial cell length")
        path_cosine = min(path_cosine, 1.0)
        saturation_projected = saturation_normal * path_cosine
        use_saturation = bool(self.physics["gain_saturation"]) and np.any(
            np.isfinite(saturation_projected) & (saturation_projected > 0.0)
        )
        population_already_updated = False
        if use_saturation and multi_transition:
            packet_count = int(
                self.levels_cfg.get("multi_transition_solver", {}).get(
                    "photon_packet_count", 32
                )
            )
            packet_input = active_input_fluence / packet_count
            gain_fluence = np.zeros_like(active_input_fluence)
            cell_length = float(self.grid.dz)
            packet_limiter_before = int(
                self.levels.signal_event_diagnostics[
                    "limited_cell_updates"
                ]
            )
            for _ in range(packet_count):
                active_gain = self.levels.signal_gain_m_inv(
                    population[:, iz],
                    wavelength_m=float(self.cfg["wavelength_m"]),
                    z_index=iz,
                ) * self.crystal_mask
                packet_exponent = active_gain * step_length
                packet_output = packet_input * np.exp(
                    np.minimum(packet_exponent, 50.0)
                )
                requested_events = (
                    packet_output - packet_input
                ) / (
                    photon_energy(float(self.cfg["wavelength_m"]))
                    * cell_length
                )
                updated, event_stats = self.levels.apply_signal_photon_events(
                    population[:, iz],
                    requested_events,
                    float(self.cfg["wavelength_m"]),
                    z_index=iz,
                    input_photon_density_m3=(
                        packet_input
                        / (
                            photon_energy(float(self.cfg["wavelength_m"]))
                            * cell_length
                        )
                    ),
                    interaction_length_m=step_length,
                )
                population[:, iz] = updated
                applied_events = np.asarray(
                    event_stats["applied_photon_events_m3"], dtype=float
                )
                packet_output = (
                    packet_input
                    + applied_events
                    * photon_energy(float(self.cfg["wavelength_m"]))
                    * cell_length
                )
                gain_fluence += np.maximum(packet_output, 0.0)
                self._maximum_unsaturated_gain_exponent = max(
                    getattr(
                        self,
                        "_maximum_unsaturated_gain_exponent",
                        -np.inf,
                    ),
                    float(np.max(packet_exponent)),
                )
                self._positive_gain_exponent_clipping_active = bool(
                    getattr(
                        self,
                        "_positive_gain_exponent_clipping_active",
                        False,
                    )
                    or np.any(packet_exponent > 50.0)
                )
            packet_diagnostics = self.levels.signal_event_diagnostics
            packet_diagnostics["packet_count_used"] = max(
                int(packet_diagnostics["packet_count_used"]),
                packet_count,
            )
            packet_diagnostics["maximum_population_change_per_packet"] = max(
                float(
                    packet_diagnostics[
                        "maximum_population_change_per_packet"
                    ]
                ),
                float(
                    packet_diagnostics[
                        "maximum_population_change_fraction_per_update"
                    ]
                ),
            )
            packet_diagnostics["limited_packet_cell_count"] = int(
                packet_diagnostics["limited_packet_cell_count"]
            ) + (
                int(packet_diagnostics["limited_cell_updates"])
                - packet_limiter_before
            )
            packet_diagnostics["packet_target_met"] = bool(
                float(
                    packet_diagnostics[
                        "maximum_population_change_per_packet"
                    ]
                )
                <= float(
                    packet_diagnostics[
                        "packet_population_change_target"
                    ]
                )
            )
            population_already_updated = True
        elif use_saturation:
            valid_saturation = np.isfinite(saturation_projected) & (saturation_projected > 0.0)
            fallback_exponent = np.where(valid_saturation, -np.inf, exponent)
            self._maximum_unsaturated_gain_exponent = max(
                getattr(self, "_maximum_unsaturated_gain_exponent", -np.inf),
                float(np.max(fallback_exponent)),
            )
            self._positive_gain_exponent_clipping_active = bool(
                getattr(self, "_positive_gain_exponent_clipping_active", False)
                or np.any(fallback_exponent > 50.0)
            )
            saturated = frantz_nodvik(
                active_input_fluence,
                exponent,
                np.where(valid_saturation, saturation_projected, 1.0),
            )
            gain_fluence = np.where(
                valid_saturation,
                saturated,
                active_input_fluence * np.exp(np.minimum(exponent, 50.0)),
            )
        else:
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
        if (
            use_saturation
            and np.any(cross_section_sum > 0.0)
            and override is None
            and not population_already_updated
        ):
            # ``fluence`` is normalised on the fixed transverse x/y grid, so
            # its change is energy per projected cell area.  Oblique incidence
            # increases the optical interaction length, but it does not
            # increase the physical voxel volume.  Convert photons to an ion
            # density with the axial cell depth, not the slanted ray length.
            cell_length = float(self.grid.dz)
            event_density = (gain_fluence - active_input_fluence) / (
                photon_energy(float(self.cfg["wavelength_m"])) * cell_length
            )
            upper = self.levels.signal.upper
            lower = self.levels.signal.lower
            event_density = np.minimum(event_density, population[upper, iz])
            event_density = np.maximum(event_density, -population[lower, iz])
            population[upper, iz] -= event_density
            population[lower, iz] += event_density
            population[upper, iz] = np.maximum(population[upper, iz], 0.0)
            population[lower, iz] = np.maximum(population[lower, iz], 0.0)
            gain_fluence = (
                active_input_fluence
                + event_density
                * photon_energy(float(self.cfg["wavelength_m"]))
                * cell_length
            )
        elif use_saturation and override is not None:
            valid_saturation = np.isfinite(saturation_normal) & (saturation_normal > 0.0)
            depletion = np.divide(
                gain_fluence - active_input_fluence,
                saturation_normal * float(self.grid.dz),
                out=np.zeros_like(gain_fluence),
                where=valid_saturation,
            )
            self._override_gain_m_inv[iz] -= depletion
            if float(override) >= 0.0:
                self._override_gain_m_inv[iz] = np.maximum(
                    self._override_gain_m_inv[iz], 0.0
                )
            else:
                self._override_gain_m_inv[iz] = np.minimum(
                    self._override_gain_m_inv[iz], 0.0
                )
        output_fluence = np.maximum(
            gain_fluence * passive_half_transmission, 0.0
        )
        passive_loss_fluence = np.maximum(
            fluence - active_input_fluence, 0.0
        ) + np.maximum(gain_fluence - output_fluence, 0.0)
        self.passive_loss_energy_J_m3[iz] += passive_loss_fluence / max(
            self.grid.dz, np.finfo(float).tiny
        )
        scale = np.sqrt(output_fluence / np.maximum(fluence, np.finfo(float).tiny))
        output = field * np.where(fluence > 0.0, scale, 0.0)
        return output * self.crystal_mask if self.clip_crystal else output

    def _integral(self, field: np.ndarray) -> float:
        return float(np.sum(self.grid.integrate_transverse(np.abs(field) ** 2)))
