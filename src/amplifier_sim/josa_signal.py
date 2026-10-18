"""Optional JOSA B transport adapter for the spectral/time Seed solver."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .constants import LIGHT_SPEED
from .dispersion import material_group_index
from .errors import ModelApplicabilityError
from .josa_modes import JOSA_FULL_3D_MODE, josa_full_3d_physics_overrides, josa_mode_info
from .josa_transport import JosaTransportOperator
from .optical_system import gain_medium_axis_angles_rad, gain_medium_path_cosine
from .profiles import DiscreteProfile
from .spectral_signal import SpectralSignalPropagator


def _profile_edges(coordinate: np.ndarray, fallback_span_s: float) -> np.ndarray:
    coordinate = np.asarray(coordinate, dtype=float)
    if len(coordinate) == 1:
        half = 0.5 * max(float(fallback_span_s), np.finfo(float).tiny)
        return np.array([coordinate[0] - half, coordinate[0] + half])
    edges = np.empty(len(coordinate) + 1, dtype=float)
    edges[1:-1] = 0.5 * (coordinate[:-1] + coordinate[1:])
    edges[0] = coordinate[0] - 0.5 * (coordinate[1] - coordinate[0])
    edges[-1] = coordinate[-1] + 0.5 * (coordinate[-1] - coordinate[-2])
    return edges


def _resample_profile_for_cfl(
    profile: DiscreteProfile,
    target_dt_s: float,
    fallback_span_s: float,
    maximum_bins: int,
) -> tuple[DiscreteProfile, dict[str, float | int]]:
    """Conservatively resample a temporal mass profile near ``dt=dz/vg``."""

    edges = _profile_edges(profile.coordinate, fallback_span_s)
    original_span = float(edges[-1] - edges[0])
    if original_span <= 0.0:
        raise ValueError("JOSA temporal support must be positive")
    count = max(1, int(math.ceil(original_span / target_dt_s)))
    if count > int(maximum_bins):
        raise ModelApplicabilityError(
            f"paper-CFL temporal grid needs {count} bins, above maximum_time_bins={maximum_bins}; "
            "use retarded_pc, or raise maximum_time_bins only after estimating memory and checking convergence"
        )
    # Keep the paper Courant number exactly one.  The source support is padded
    # symmetrically by less than one bin when it is not an integer multiple of
    # dz/vg; extending the empty wings is preferable to silently changing dt.
    resampled_span = count * float(target_dt_s)
    center = 0.5 * float(edges[0] + edges[-1])
    new_edges = np.linspace(
        center - 0.5 * resampled_span,
        center + 0.5 * resampled_span,
        count + 1,
    )
    new_centers = 0.5 * (new_edges[:-1] + new_edges[1:])
    # Treat every source weight as mass uniformly distributed inside its bin
    # and integrate exact old/new bin overlaps.  Interpolating density at the
    # new centres and renormalising preserves only total energy: it can move a
    # sharp pulse front, alter the peak, and (for one input bin) spread energy
    # into the deliberately empty padding used to enforce Courant number one.
    old_widths = np.diff(edges)
    new_weights = np.zeros(count, dtype=float)
    source = 0
    for target in range(count):
        left = float(new_edges[target])
        right = float(new_edges[target + 1])
        while source < len(profile.weights) and float(edges[source + 1]) <= left:
            source += 1
        candidate = source
        while candidate < len(profile.weights) and float(edges[candidate]) < right:
            overlap = max(
                0.0,
                min(right, float(edges[candidate + 1]))
                - max(left, float(edges[candidate])),
            )
            if overlap > 0.0:
                new_weights[target] += (
                    float(profile.weights[candidate])
                    * overlap
                    / float(old_widths[candidate])
                )
            candidate += 1
    new_weights = np.maximum(new_weights, 0.0)
    if float(np.sum(new_weights)) <= 0.0:
        nearest = int(np.argmin(np.abs(new_centers - float(np.mean(profile.coordinate)))))
        new_weights[nearest] = 1.0
    new_weights /= float(np.sum(new_weights))
    actual_dt = resampled_span / count
    return DiscreteProfile(new_centers, new_weights), {
        "input_temporal_bins": len(profile.coordinate),
        "transport_temporal_bins": count,
        "input_temporal_support_s": original_span,
        "temporal_support_s": resampled_span,
        "target_dt_s": float(target_dt_s),
        "actual_dt_s": actual_dt,
    }


def conservative_rebin_joint_time_matrix(
    original_profile: DiscreteProfile,
    target_profile: DiscreteProfile,
    joint_mass: np.ndarray,
    fallback_span_s: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Rebin every spectral column of a joint time-frequency mass matrix.

    Each old temporal-bin mass is treated as uniform over its finite-volume
    support and integrated against the target CFL bins.  Spectral-column mass
    and total pulse energy are therefore preserved independently.
    """

    matrix = np.asarray(joint_mass, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(original_profile.coordinate):
        raise ValueError(
            "joint time-frequency matrix must match the original temporal profile"
        )
    old_edges = _profile_edges(original_profile.coordinate, fallback_span_s)
    target_span = (
        fallback_span_s
        if len(target_profile.coordinate) == 1
        else float(
            target_profile.coordinate[-1]
            - target_profile.coordinate[0]
            + np.median(np.diff(target_profile.coordinate))
        )
    )
    new_edges = _profile_edges(target_profile.coordinate, target_span)
    rebinned = np.zeros((len(target_profile.coordinate), matrix.shape[1]))
    old_widths = np.diff(old_edges)
    for old_index in range(matrix.shape[0]):
        left = float(old_edges[old_index])
        right = float(old_edges[old_index + 1])
        first = max(0, int(np.searchsorted(new_edges, left, side="right") - 1))
        target = first
        while target < len(target_profile.coordinate) and new_edges[target] < right:
            overlap = max(
                0.0,
                min(right, float(new_edges[target + 1]))
                - max(left, float(new_edges[target])),
            )
            if overlap > 0.0:
                rebinned[target] += matrix[old_index] * (
                    overlap / float(old_widths[old_index])
                )
            target += 1
    old_columns = np.sum(matrix, axis=0)
    new_columns = np.sum(rebinned, axis=0)
    # The CFL grid pads, rather than truncates, the old support.  Any residual
    # here is roundoff; rescale each non-empty column to close it exactly.
    factors = np.divide(
        old_columns,
        new_columns,
        out=np.ones_like(old_columns),
        where=new_columns > np.finfo(float).tiny,
    )
    rebinned *= factors[None, :]
    final_columns = np.sum(rebinned, axis=0)
    column_error = np.max(
        np.abs(final_columns - old_columns)
        / np.maximum(np.abs(old_columns), np.finfo(float).tiny)
    )
    return rebinned, {
        "joint_matrix_input_mass": float(np.sum(matrix)),
        "joint_matrix_output_mass": float(np.sum(rebinned)),
        "joint_matrix_maximum_spectral_column_mass_error_relative": float(
            column_error
        ),
    }


def conservative_time_shift_links(
    edges_s: np.ndarray,
    delay_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a sparse conservative remap for incoherent temporal-bin energy.

    Input bin ``i`` is translated by ``delay_s`` and its uniform-in-bin energy
    density is distributed over every intersecting fixed output bin.  Energy
    outside the configured time window is deliberately not wrapped around.
    """

    edges = np.asarray(edges_s, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("Temporal edges must be a strictly increasing one-dimensional array")
    output_indices: list[int] = []
    input_indices: list[int] = []
    fractions: list[float] = []
    count = len(edges) - 1
    for source in range(count):
        source_left = float(edges[source] + delay_s)
        source_right = float(edges[source + 1] + delay_s)
        source_width = float(edges[source + 1] - edges[source])
        target = max(0, int(np.searchsorted(edges, source_left, side="right") - 1))
        while target < count and edges[target] < source_right:
            overlap = max(
                0.0,
                min(source_right, float(edges[target + 1]))
                - max(source_left, float(edges[target])),
            )
            if overlap > 0.0:
                output_indices.append(target)
                input_indices.append(source)
                fractions.append(overlap / source_width)
            target += 1
    return (
        np.asarray(output_indices, dtype=np.int64),
        np.asarray(input_indices, dtype=np.int64),
        np.asarray(fractions, dtype=float),
    )


def shift_incoherent_temporal_field(
    field: np.ndarray,
    links: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[np.ndarray, float]:
    """Apply the legacy energy-conservative local phase-closure remap.

    The returned complex field preserves energy in each target time/pixel and
    uses a weighted circular phase moment.  When multiple mutually incoherent
    source bins enter one target bin, collapsing them to one complex field is
    an approximation for any later phase-sensitive diffraction.  ``diagnostics``
    exposes when that closure is active; it does not claim to make it exact.
    """

    value = np.asarray(field, dtype=complex)
    if value.ndim != 3:
        raise ValueError("A wavelength-resolved temporal field must have shape (time, y, x)")
    output_index, input_index, fraction = links
    energy = np.abs(value) ** 2
    shifted_energy = np.zeros_like(energy)
    phase_moment = np.zeros_like(value)
    contributing_sources = (
        np.zeros_like(energy, dtype=np.int32) if diagnostics is not None else None
    )
    if len(fraction):
        contribution = fraction[:, None, None] * energy[input_index]
        np.add.at(shifted_energy, output_index, contribution)
        unit_phase = np.divide(
            value,
            np.abs(value),
            out=np.ones_like(value),
            where=np.abs(value) > np.finfo(float).tiny,
        )
        np.add.at(phase_moment, output_index, contribution * unit_phase[input_index])
        if contributing_sources is not None:
            np.add.at(
                contributing_sources,
                output_index,
                contribution > np.finfo(float).tiny,
            )
    result = np.sqrt(np.maximum(shifted_energy, 0.0)) * np.exp(1j * np.angle(phase_moment))
    input_total = float(np.sum(energy))
    shifted_total = float(np.sum(shifted_energy))
    retained = shifted_total / max(input_total, np.finfo(float).tiny)
    if diagnostics is not None:
        assert contributing_sources is not None
        mixed_mask = contributing_sources > 1
        mixed_time = np.any(mixed_mask, axis=(1, 2))
        resultant = np.divide(
            np.abs(phase_moment),
            shifted_energy,
            out=np.ones_like(shifted_energy),
            where=shifted_energy > np.finfo(float).tiny,
        )
        mixed_energy = float(np.sum(shifted_energy[mixed_mask]))
        diagnostics.update(
            {
                "multi_source_target_bins": int(np.count_nonzero(mixed_time)),
                "maximum_source_bins_per_target": int(
                    np.max(contributing_sources, initial=0)
                ),
                "multi_source_mixed_energy_fraction": (
                    mixed_energy / max(shifted_total, np.finfo(float).tiny)
                ),
                "minimum_phase_resultant": (
                    float(np.min(resultant[mixed_mask]))
                    if np.any(mixed_mask)
                    else 1.0
                ),
                "maximum_phase_dispersion": (
                    float(np.max(1.0 - resultant[mixed_mask]))
                    if np.any(mixed_mask)
                    else 0.0
                ),
                "strict_incoherent_diffraction_closure": not bool(
                    np.any(mixed_mask)
                ),
            }
        )
    return result, min(max(retained, 0.0), 1.0)


class JosaTransportSignalPropagator(SpectralSignalPropagator):
    """Three-dimensional, phase-insensitive JOSA transport Seed branch.

    Transverse diffraction and optical elements remain in the established
    spectral solver.  Only the local gain/depletion update is replaced by the
    paper-derived Explicit or prediction-correction operator.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.josa_cfg = self.cfg.get("josa", {})
        self.josa_mode = str(self.josa_cfg.get("mode", "retarded_pc"))
        self.mode_info = josa_mode_info(self.josa_mode)
        self.full_3d = self.josa_mode == JOSA_FULL_3D_MODE
        self.group_index = material_group_index(
            self.material,
            float(self.cfg["wavelength_m"]),
            role="signal",
        )
        self.spectral_group_indices = np.asarray(
            [
                material_group_index(self.material, float(wavelength), role="signal")
                for wavelength in self.spectrum.coordinate
            ],
            dtype=float,
        )
        self.relative_group_index_span = float(
            np.ptp(self.spectral_group_indices)
            / max(float(np.mean(self.spectral_group_indices)), np.finfo(float).tiny)
        )
        if self.relative_group_index_span > 1e-3 and not self.full_3d:
            self.source_definition_warnings.append(
                "JOSA uses the central-wavelength group velocity for one shared time stencil; "
                f"the configured spectral bins span {100.0 * self.relative_group_index_span:.3g}% in group index. "
                "Phase-insensitive group-delay walk-off is not resolved by this backend."
            )
        self.group_velocity_m_s = LIGHT_SPEED / self.group_index
        transport_markers = self.gain_markers if self.gain_markers else [self.gain_marker]
        self.source_launch_angle_x_deg = float(self.cfg.get("incidence_angle_x_deg", 0.0))
        self.source_launch_angle_y_deg = float(self.cfg.get("incidence_angle_y_deg", 0.0))
        self.marker_incidence_is_additional = bool(
            self.full_3d
            and (
                abs(self.source_launch_angle_x_deg) + abs(self.source_launch_angle_y_deg) > 0.0
            )
            and any(
                abs(float(marker.get("incidence_angle_x_deg", 0.0)))
                + abs(float(marker.get("incidence_angle_y_deg", 0.0)))
                > 0.0
                for marker in transport_markers
            )
        )
        if self.marker_incidence_is_additional:
            self.source_definition_warnings.append(
                "Seed source launch angle and gain_medium incidence angle are both non-zero. "
                "The sampled source phase tilt and the per-pass crystal walk-off are independent and additive; "
                "if they describe the same physical beam angle, set one pair to zero to avoid double counting."
            )
        path_cosines = np.asarray(
            [
                gain_medium_path_cosine(marker)
                for marker in transport_markers
            ],
            dtype=float,
        )
        self.transport_step_length_m = self.grid.dz / float(path_cosines[0])
        if self.josa_mode.endswith("_cfl") and not np.allclose(
            path_cosines,
            path_cosines[0],
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ModelApplicabilityError(
                "paper-CFL JOSA modes require the same effective crystal step length on every pass; "
                "use retarded_pc when gain-medium incidence angles differ"
            )
        self.paper_cfl_dt_s = self.transport_step_length_m / self.group_velocity_m_s
        fallback_span = (
            2.0
            * max(1.0, float(self.cfg.get("temporal", {}).get("span_fwhm", 3.0)))
            * float(self.cfg.get("pulse_duration_s", 1e-12))
        )
        self.cfl_resampling: dict[str, Any] = {
            "input_temporal_bins": len(self.temporal.coordinate),
            "transport_temporal_bins": len(self.temporal.coordinate),
            "target_dt_s": self.paper_cfl_dt_s,
        }
        if self.josa_mode.endswith("_cfl"):
            original_temporal = self.temporal
            original_joint = (
                None
                if self.joint_time_frequency_weights is None
                else self.joint_time_frequency_weights.copy()
            )
            self.temporal, self.cfl_resampling = _resample_profile_for_cfl(
                self.temporal,
                self.paper_cfl_dt_s,
                fallback_span,
                int(self.josa_cfg.get("maximum_time_bins", 4096)),
            )
            if original_joint is not None:
                (
                    self.joint_time_frequency_weights,
                    joint_diagnostics,
                ) = conservative_rebin_joint_time_matrix(
                    original_temporal,
                    self.temporal,
                    original_joint,
                    fallback_span,
                )
                self.temporal = DiscreteProfile(
                    self.temporal.coordinate.copy(),
                    np.sum(self.joint_time_frequency_weights, axis=1),
                )
                self.spectrum = DiscreteProfile(
                    self.spectrum.coordinate.copy(),
                    np.sum(self.joint_time_frequency_weights, axis=0),
                )
                self.cfl_resampling.update(joint_diagnostics)
                self.cfl_resampling[
                    "joint_matrix_rebin"
                ] = "finite_volume_overlap_per_spectral_column"
            if len(self.temporal.coordinate) < 5:
                self.source_definition_warnings.append(
                    "Paper-CFL JOSA grid has fewer than five temporal bins across the configured pulse window; "
                    "increase Nz or use retarded_pc before interpreting the pulse shape."
                )
        if self.full_3d and len(self.temporal.coordinate) < 3:
            raise ValueError(
                "full_3d_pc needs at least three ordered temporal bins; increase the Seed solver time samples"
            )
        if self.full_3d:
            required = josa_full_3d_physics_overrides(self.physics)
            mismatches = {
                key: (self.physics.get(key), expected)
                for key, expected in required.items()
                if self.physics.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(f"full_3d_pc solver contract was not resolved: {mismatches}")
        self._time_edges = _profile_edges(self.temporal.coordinate, fallback_span)
        self._group_delay_link_cache: dict[
            tuple[int, float], tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self._maximum_group_delay_loss_fraction = 0.0
        self._group_delay_steps = 0
        self._phase_closure_stats = self._empty_phase_closure_stats()
        self._maximum_runtime_half_step_walkoff_m = 0.0
        self.operator = JosaTransportOperator(
            self.levels,
            self.levels_cfg,
            self.material,
            self.physics,
            self.josa_cfg,
            self.spectrum.coordinate,
            self.temporal.coordinate,
            self.crystal_mask,
        )

    def _gain_step(
        self,
        field: np.ndarray,
        population: np.ndarray,
        iz: int,
        step_length_m: float,
    ) -> np.ndarray:
        output = self.operator.apply(
            field,
            population[:, iz],
            step_length_m,
            iz,
            cell_length_m=self.grid.dz,
        )
        passive_loss = self.operator.last_passive_loss_fluence_J_m2
        if passive_loss is not None:
            self.passive_loss_energy_J_m3[iz] += passive_loss / max(
                self.grid.dz, np.finfo(float).tiny
            )
        if self.clip_crystal:
            output *= self.crystal_mask[None, None, :, :]
        return output

    def _after_gain_step(
        self,
        field: np.ndarray,
        step_length_m: float,
        z_index: int | None = None,
    ) -> np.ndarray:
        """Resolve wavelength-dependent arrival time in the full 3D mode."""

        result = super()._after_gain_step(field, step_length_m, z_index)
        if not self.full_3d or len(self.spectrum.coordinate) < 2:
            return result
        shifted = result.copy()
        for wavelength_index, group_index in enumerate(self.spectral_group_indices):
            delay = (float(group_index) - self.group_index) * float(step_length_m) / LIGHT_SPEED
            if abs(delay) <= np.finfo(float).tiny:
                continue
            key = (wavelength_index, float(step_length_m))
            links = self._group_delay_link_cache.get(key)
            if links is None:
                links = conservative_time_shift_links(self._time_edges, delay)
                self._group_delay_link_cache[key] = links
            remap_diagnostics: dict[str, Any] = {}
            shifted[:, wavelength_index], retained = shift_incoherent_temporal_field(
                result[:, wavelength_index],
                links,
                diagnostics=remap_diagnostics,
            )
            self._record_phase_closure(remap_diagnostics)
            self._maximum_group_delay_loss_fraction = max(
                self._maximum_group_delay_loss_fraction,
                max(0.0, 1.0 - retained),
            )
            self._group_delay_steps += 1
        return shifted

    def _diffract(self, field: np.ndarray, distance_m: float, refractive_index: float) -> np.ndarray:
        """Track the shared in-crystal walk-off applied by the base FFT step."""

        result = super()._diffract(field, distance_m, refractive_index)
        if not self.full_3d:
            return result
        marker = getattr(self, "_active_gain_marker", self.gain_marker)
        angle_x, angle_y = gain_medium_axis_angles_rad(marker)
        path_cosine = gain_medium_path_cosine(marker)
        axial_distance = float(distance_m) * path_cosine
        shift_x = axial_distance * math.tan(angle_x)
        shift_y = axial_distance * math.tan(angle_y)
        self._maximum_runtime_half_step_walkoff_m = max(
            self._maximum_runtime_half_step_walkoff_m,
            math.hypot(shift_x, shift_y),
        )
        if shift_x == 0.0 and shift_y == 0.0:
            return result
        return result

    def _model_used(self) -> str:
        return "josa_transport"

    def resolved_intrapulse_population_time_s(self) -> float:
        """Return level-kinetics time advanced by one JOSA Seed pass."""

        if not self.operator.include_relaxation or len(self.temporal.coordinate) < 2:
            return 0.0
        return float(
            self.temporal.coordinate[-1] - self.temporal.coordinate[0]
        )

    def _gain_transport_diagnostics(self) -> dict[str, Any]:
        diagnostics = self.operator.diagnostics()
        coordinate = np.asarray(self.temporal.coordinate, dtype=float)
        if len(coordinate) > 1:
            median_dt = float(np.median(np.diff(coordinate)))
        else:
            median_dt = float(self.cfl_resampling.get("actual_dt_s", self.paper_cfl_dt_s))
        diagnostics.update(
            {
                "group_index": self.group_index,
                "spectral_group_index_min": float(np.min(self.spectral_group_indices)),
                "spectral_group_index_max": float(np.max(self.spectral_group_indices)),
                "spectral_group_index_relative_span": self.relative_group_index_span,
                "group_velocity_m_s": self.group_velocity_m_s,
                "dz_m": self.grid.dz,
                "transport_step_length_m": self.transport_step_length_m,
                "paper_cfl_dt_s": self.paper_cfl_dt_s,
                "median_temporal_bin_spacing_s": median_dt,
                "courant_number_from_reported_bins": (
                    self.group_velocity_m_s * median_dt / self.transport_step_length_m
                ),
                "cfl_resampling": dict(self.cfl_resampling),
                "fidelity": str(self.mode_info["tier"]),
                "resolved_dimensions": ["x", "y", "z", "time", "wavelength"],
                "full_3d_rate_contract": self._full_3d_contract(),
                "group_delay_walkoff": self._group_delay_diagnostics(),
                "incidence_walkoff": self._incidence_walkoff_diagnostics(),
                "phase_contract": {
                    **self.input_pulse_projection,
                    "input_phase_retained_after_projection": False,
                    "coherent_frequency_phase_propagated": False,
                    "material_or_train_gdd_tod_fod_propagated": False,
                    "compressor_phase_propagated": False,
                    "kerr_or_spm_propagated": False,
                    "scope": (
                        "configured source/stretcher phase may shape the input temporal "
                        "energy bins once; JOSA then transports mutually incoherent "
                        "time/wavelength fluence bins"
                    ),
                    "coherent_alternative": "coherent_cpa",
                },
                "status": "applied",
            }
        )
        return diagnostics

    def _full_3d_contract(self) -> dict[str, Any]:
        required = josa_full_3d_physics_overrides(self.physics)
        checks = {key: self.physics.get(key) == expected for key, expected in required.items()}
        solver_stack_satisfied = bool(self.full_3d and all(checks.values()))
        strict_phase_closure = bool(
            self._phase_closure_stats[
                "strict_incoherent_diffraction_closure"
            ]
        )
        return {
            "requested": self.full_3d,
            # Backward-compatible stack contract: this says the requested
            # modules ran, not that the group-delay phase closure is exact.
            "satisfied": solver_stack_satisfied,
            "solver_stack_satisfied": solver_stack_satisfied,
            "strict_fidelity_satisfied": bool(
                solver_stack_satisfied and strict_phase_closure
            ),
            "strict_incoherent_diffraction_closure": strict_phase_closure,
            "checks": checks,
            "effective_physics": {key: self.physics.get(key) for key in required},
            "scope": (
                "three-dimensional phase-insensitive rate-equation/SVEA transport with "
                "energy-conservative group-delay phase-closure approximation"
            ),
            "coherent_phase_included": False,
            "coherent_alternative": "coherent_cpa",
        }

    def _group_delay_diagnostics(self) -> dict[str, Any]:
        markers = self.gain_markers if self.gain_markers else [self.gain_marker]
        maximum_path = 0.0
        for marker in markers:
            path_cosine = gain_medium_path_cosine(marker)
            maximum_path = max(
                maximum_path,
                len(self.grid.z) * self.grid.dz / path_cosine,
            )
        relative_delays = (
            (self.spectral_group_indices - self.group_index) * maximum_path / LIGHT_SPEED
        )
        return {
            "implemented": self.full_3d,
            "active": bool(self.full_3d and len(self.spectrum.coordinate) > 1),
            "maximum_relative_delay_per_pass_s": float(np.max(np.abs(relative_delays))),
            "remap_steps": self._group_delay_steps,
            "maximum_energy_fraction_lost_at_time_window": (
                self._maximum_group_delay_loss_fraction
            ),
            "boundary": "zero outside configured temporal support; no periodic wrap",
            "phase_closure": {
                **self._phase_closure_stats,
                "method": "energy_conservative_local_phase_moment",
                "approximation_active": bool(
                    self._phase_closure_stats["remaps_with_multi_source_mixing"]
                    > 0
                ),
                "limitation": (
                    "Mutually incoherent source-bin contributions are collapsed to one complex transverse field; "
                    "subsequent FFT diffraction is not a strict incoherent-channel propagation."
                ),
            },
        }

    @staticmethod
    def _empty_phase_closure_stats() -> dict[str, Any]:
        return {
            "remaps_with_multi_source_mixing": 0,
            "maximum_source_bins_per_target": 1,
            "maximum_multi_source_mixed_energy_fraction": 0.0,
            "minimum_phase_resultant": 1.0,
            "maximum_phase_dispersion": 0.0,
            "strict_incoherent_diffraction_closure": True,
        }

    def _record_phase_closure(self, diagnostics: dict[str, Any]) -> None:
        mixed = int(diagnostics.get("multi_source_target_bins", 0)) > 0
        if mixed:
            self._phase_closure_stats["remaps_with_multi_source_mixing"] += 1
            self._phase_closure_stats[
                "strict_incoherent_diffraction_closure"
            ] = False
        self._phase_closure_stats["maximum_source_bins_per_target"] = max(
            int(self._phase_closure_stats["maximum_source_bins_per_target"]),
            int(diagnostics.get("maximum_source_bins_per_target", 1)),
        )
        self._phase_closure_stats[
            "maximum_multi_source_mixed_energy_fraction"
        ] = max(
            float(
                self._phase_closure_stats[
                    "maximum_multi_source_mixed_energy_fraction"
                ]
            ),
            float(diagnostics.get("multi_source_mixed_energy_fraction", 0.0)),
        )
        self._phase_closure_stats["minimum_phase_resultant"] = min(
            float(self._phase_closure_stats["minimum_phase_resultant"]),
            float(diagnostics.get("minimum_phase_resultant", 1.0)),
        )
        self._phase_closure_stats["maximum_phase_dispersion"] = max(
            float(self._phase_closure_stats["maximum_phase_dispersion"]),
            float(diagnostics.get("maximum_phase_dispersion", 0.0)),
        )

    def _incidence_walkoff_diagnostics(self) -> dict[str, Any]:
        markers = self.gain_markers if self.gain_markers else [self.gain_marker]
        per_pass: list[dict[str, float]] = []
        length_z = len(self.grid.z) * self.grid.dz
        for marker in markers:
            angle_x, angle_y = gain_medium_axis_angles_rad(marker)
            per_pass.append(
                {
                    "angle_x_deg": math.degrees(angle_x),
                    "angle_y_deg": math.degrees(angle_y),
                    "shift_x_m": length_z * math.tan(angle_x),
                    "shift_y_m": length_z * math.tan(angle_y),
                }
            )
        return {
            "implemented": self.full_3d,
            "active": bool(
                self.full_3d
                and any(abs(item["shift_x_m"]) + abs(item["shift_y_m"]) > 0.0 for item in per_pass)
            ),
            "per_pass": per_pass,
            "maximum_runtime_half_step_shift_m": self._maximum_runtime_half_step_walkoff_m,
            "angle_definition": "signed local beam direction at each gain_medium marker",
            "source_launch_angle_x_deg": self.source_launch_angle_x_deg,
            "source_launch_angle_y_deg": self.source_launch_angle_y_deg,
            "marker_walkoff_is_additional_to_source_phase_tilt": self.marker_incidence_is_additional,
        }

    def propagate(self, population: np.ndarray, delta_n: np.ndarray):
        self.operator.begin_propagation()
        self._maximum_group_delay_loss_fraction = 0.0
        self._group_delay_steps = 0
        self._phase_closure_stats = self._empty_phase_closure_stats()
        self._maximum_runtime_half_step_walkoff_m = 0.0
        result = super().propagate(population, delta_n)
        if self.full_3d and self._maximum_group_delay_loss_fraction > 1e-3:
            self.operator.warnings.append(
                "Full-3D group-delay remapping lost more than 0.1% in one step at the configured temporal boundary; "
                "increase the Seed temporal span/sample count and check convergence."
            )
        if (
            self.full_3d
            and self._phase_closure_stats["remaps_with_multi_source_mixing"] > 0
        ):
            self.operator.warnings.append(
                "JOSA full_3d_pc group-delay remapping used an energy-conservative local phase-closure approximation: "
                "multiple incoherent source time bins were collapsed before later FFT diffraction. "
                "Treat this mode as phase-closure approximate, not strict incoherent 3-D propagation."
            )
        result.warnings = list(dict.fromkeys([*result.warnings, *self.operator.warnings]))
        result.gain_transport = self._gain_transport_diagnostics()
        return result
