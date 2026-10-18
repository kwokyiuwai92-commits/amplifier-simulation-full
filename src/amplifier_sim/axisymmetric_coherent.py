"""Low-order azimuthal coherent Kerr propagation inside the gain medium.

This is a deliberately explicit intermediate model between the Gaussian-duct
Commercial App closure and the Cartesian ``E(f, y, x)`` coherent solver.  External
optics, interfaces, source sampling, population/thermal grids, and output
diagnostics remain Cartesian.  At each gain entrance an azimuthally symmetric
field is projected once onto ``E(f, r)``; inside the crystal a conservative
radial paraxial step retains radial diffraction/self-focusing, material
dispersion, local time-domain Kerr phase, wavelength-dependent FN gain, and
radial population depletion.  The field is reconstructed on x/y only at the
gain exit.

The default approximation removes all non-zero azimuthal orders.  Optional
finite Fourier closures retain m=1 (dipole/decenter/tilt) and m=2
(quadrupole/astigmatism) fields on a small angular collocation grid.  Runtime
then scales with that small angular grid rather than the Cartesian x-y sample
count.  Every retained optical order is propagated by its own radial paraxial
operator; material, gain, Kerr, and thermal operators are evaluated in angle
space and projected back to the declared order after each local step.
"""

from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any

import numpy as np

from .axisymmetric_diffraction_kernel import (
    NUMBA_AVAILABLE as RADIAL_NUMBA_AVAILABLE,
    numba_kernel_compiled_in_process as radial_numba_kernel_compiled,
    run_radial_cn_numba,
)
from .coherent_signal import CoherentSignalPropagator
from .nonlinear import apply_strong_field_step
from .optical_system import gain_medium_axis_angles_rad
from .profiles import DiscreteProfile
from .profiling import profiled_operation
from .spectral_signal import SpectralSignalPropagator
from .spatial_modes import axis_focus_m


def _radial_linear_interpolation_indices(
    radius_m: np.ndarray,
    radial_step_m: float,
    radial_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return clamped cell-centre interpolation indices for a radial mesh.

    Radial unknowns live at ``dr/2, 3dr/2, ...``.  Points between the axis and
    the first centre must therefore use the first value (the regularity/zero-
    slope boundary), not interpolate across a fictitious negative-radius node.
    """

    fractional = np.asarray(radius_m, dtype=float) / radial_step_m - 0.5
    raw_lower = np.floor(fractional).astype(int)
    upper_weight = fractional - raw_lower
    lower = np.clip(raw_lower, 0, radial_bins - 1)
    upper = np.clip(raw_lower + 1, 0, radial_bins - 1)
    clamped = (raw_lower < 0) | (raw_lower >= radial_bins - 1)
    upper_weight = np.where(clamped, 0.0, upper_weight)
    return lower, upper, np.clip(upper_weight, 0.0, 1.0)


class AxisymmetricCoherentSignalPropagator(CoherentSignalPropagator):
    """Propagate coherent ``E(f,r,m)`` with a finite azimuthal closure."""

    _AZIMUTHAL_FIDELITY_ORDERS = {
        "m0": 0,
        "m0_m1": 1,
        "m0_m1_m2_quadratic": 2,
        "m0_m1_m2": 2,
    }

    _SUPPORTED_FIDELITY_MODES = {
        "axisymmetric_kerr",
        "axisymmetric_transient_gain",
    }

    def _defer_transient_gain_initialization(self) -> bool:
        # ``CoherentSignalPropagator.__init__`` runs before the radial annulus
        # areas and mask exist.  The common propagate() entry point constructs
        # the operator lazily after this subclass has finished its geometry.
        return True

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if self.coherent_fidelity_mode not in self._SUPPORTED_FIDELITY_MODES:
            raise ValueError(
                "AxisymmetricCoherentSignalPropagator requires "
                "seed.coherent.fidelity_mode=axisymmetric_kerr or "
                "axisymmetric_transient_gain"
            )
        axis_cfg = dict(self.cfg.get("coherent", {}).get("axisymmetric", {}))
        self.axisymmetric_cfg = axis_cfg
        self.azimuthal_fidelity = str(
            axis_cfg.get("azimuthal_fidelity", "m0")
        )
        if self.azimuthal_fidelity not in self._AZIMUTHAL_FIDELITY_ORDERS:
            raise ValueError(
                "seed.coherent.axisymmetric.azimuthal_fidelity must be m0, "
                "m0_m1, m0_m1_m2_quadratic, or m0_m1_m2"
            )
        self.maximum_azimuthal_order = self._AZIMUTHAL_FIDELITY_ORDERS[
            self.azimuthal_fidelity
        ]
        self.angular_samples = (
            1
            if self.maximum_azimuthal_order == 0
            else int(axis_cfg.get("angular_samples", 12))
        )
        minimum_angular_samples = 4 * self.maximum_azimuthal_order + 1
        if (
            self.maximum_azimuthal_order > 0
            and self.angular_samples < minimum_angular_samples
        ):
            raise ValueError(
                "seed.coherent.axisymmetric.angular_samples must be at least "
                f"{minimum_angular_samples} for the selected azimuthal fidelity"
            )
        if self.angular_samples > 128:
            raise ValueError(
                "seed.coherent.axisymmetric.angular_samples must not exceed 128"
            )
        self.radial_bins = int(axis_cfg.get("radial_bins", 24))
        self.radial_diffraction_kernel_requested = str(
            axis_cfg.get("radial_diffraction_kernel", "auto")
        )
        if self.radial_diffraction_kernel_requested not in {
            "auto",
            "numpy",
            "numba",
        }:
            raise ValueError(
                "seed.coherent.axisymmetric.radial_diffraction_kernel must "
                "be auto, numpy, or numba"
            )
        self.radial_numba_minimum_systems = int(
            axis_cfg.get("radial_numba_minimum_systems", 512)
        )
        if self.radial_numba_minimum_systems < 1:
            raise ValueError(
                "seed.coherent.axisymmetric.radial_numba_minimum_systems "
                "must be positive"
            )
        self.radial_coefficient_cache_max_bytes = int(
            axis_cfg.get(
                "radial_coefficient_cache_max_bytes", 128 * 1024**2
            )
        )
        if self.radial_coefficient_cache_max_bytes < 1:
            raise ValueError(
                "seed.coherent.axisymmetric."
                "radial_coefficient_cache_max_bytes must be positive"
            )
        if (
            self.radial_diffraction_kernel_requested == "numba"
            and not RADIAL_NUMBA_AVAILABLE
        ):
            raise RuntimeError(
                "axisymmetric radial_diffraction_kernel=numba requires the "
                "performance extra"
            )
        self.radial_diffraction_kernel_selected = (
            "numba"
            if RADIAL_NUMBA_AVAILABLE
            and self.radial_diffraction_kernel_requested != "numpy"
            and len(self.pulse.frequency_Hz)
            >= self.radial_numba_minimum_systems
            else "numpy"
        )
        self.radial_diffraction_kernel_calls = 0
        self.radial_diffraction_numba_cold_starts = 0
        self.configured_gain_spectral_bins = min(
            int(axis_cfg.get("gain_spectral_bins", 65)),
            len(self.pulse.frequency_Hz),
        )
        self.gain_spectral_bins = self.configured_gain_spectral_bins
        self._validate_axisymmetric_contract()

        fine_frequency = np.asarray(self.pulse.frequency_Hz, dtype=float)
        native_gain_grid = self.configured_gain_spectral_bins == len(
            fine_frequency
        )
        if native_gain_grid:
            # A requested bin count equal to the coherent frequency count is
            # a strict no-reduction contract.  In particular, do not collapse
            # the low-energy spectral tails into two aggregate guard groups.
            central_indices = np.arange(len(fine_frequency), dtype=int)
        else:
            source_energy = np.maximum(
                np.asarray(self.pulse.spectral_energy_J, dtype=float),
                0.0,
            )
            normalized_source = source_energy / max(
                float(np.sum(source_energy)), np.finfo(float).tiny
            )
            cumulative_source = np.cumsum(normalized_source)
            support_first = max(
                int(
                    np.searchsorted(
                        cumulative_source, 0.5e-12, side="left"
                    )
                )
                - 1,
                0,
            )
            support_last = min(
                int(
                    np.searchsorted(
                        cumulative_source,
                        1.0 - 0.5e-12,
                        side="left",
                    )
                )
                + 1,
                len(fine_frequency) - 1,
            )
            relevant_low = float(fine_frequency[support_first])
            relevant_high = float(fine_frequency[support_last])
            signal_curves = [self.levels.signal_curve]
            signal_curves.extend(
                self.levels._transition_curves.get(id(transition))
                for transition in self.levels.signal_transitions
            )
            calibrated_wavelengths = [
                np.asarray(curve.wavelength_m, dtype=float)
                for curve in signal_curves
                if curve is not None
            ]
            if calibrated_wavelengths:
                calibration = np.concatenate(calibrated_wavelengths)
                relevant_low = min(
                    relevant_low,
                    299_792_458.0 / float(np.max(calibration)),
                )
                relevant_high = max(
                    relevant_high,
                    299_792_458.0 / float(np.min(calibration)),
                )
            central_indices = np.flatnonzero(
                (fine_frequency >= relevant_low)
                & (fine_frequency <= relevant_high)
            )
            if not len(central_indices):
                central_indices = np.arange(len(fine_frequency), dtype=int)
        lower_guard = np.arange(0, int(central_indices[0]), dtype=int)
        upper_guard = np.arange(
            int(central_indices[-1]) + 1,
            len(fine_frequency),
            dtype=int,
        )
        guard_count = int(bool(len(lower_guard))) + int(bool(len(upper_guard)))
        central_budget = max(
            self.configured_gain_spectral_bins - guard_count,
            1,
        )
        central_groups = [
            np.asarray(indices, dtype=int)
            for indices in np.array_split(
                central_indices,
                min(central_budget, len(central_indices)),
            )
            if len(indices)
        ]
        gain_groups: list[np.ndarray] = []
        gain_group_roles: list[str] = []
        if len(lower_guard):
            gain_groups.append(lower_guard)
            gain_group_roles.append("lower_guard")
        central_group_start = len(gain_groups)
        gain_groups.extend(central_groups)
        gain_group_roles.extend("resolved" for _ in central_groups)
        central_group_stop = len(gain_groups)
        if len(upper_guard):
            gain_groups.append(upper_guard)
            gain_group_roles.append("upper_guard")
        self._axisym_gain_groups = tuple(gain_groups)
        self._axisym_gain_group_roles = tuple(gain_group_roles)
        self._axisym_gain_central_group_positions = np.arange(
            central_group_start,
            central_group_stop,
            dtype=int,
        )
        self._axisym_gain_resolved_frequency_range_Hz = (
            float(fine_frequency[central_indices[0]]),
            float(fine_frequency[central_indices[-1]]),
        )
        self.gain_spectral_bins = len(self._axisym_gain_groups)
        representative_wavelengths: list[float] = []
        representative_frequencies: list[float] = []
        maximum_relative_span = 0.0
        for indices, role in zip(
            self._axisym_gain_groups,
            self._axisym_gain_group_roles,
            strict=True,
        ):
            frequency = self.pulse.frequency_Hz[indices]
            weights = self.pulse.spectral_energy_J[indices]
            if float(np.sum(weights)) > np.finfo(float).tiny:
                representative_frequency = float(
                    np.sum(frequency * weights) / np.sum(weights)
                )
            else:
                representative_frequency = float(np.mean(frequency))
            representative_wavelengths.append(
                299_792_458.0 / representative_frequency
            )
            representative_frequencies.append(representative_frequency)
            if role == "resolved":
                maximum_relative_span = max(
                    maximum_relative_span,
                    float(np.ptp(frequency))
                    / max(abs(representative_frequency), np.finfo(float).tiny),
                )
        self.axisym_maximum_gain_bin_relative_frequency_span = (
            maximum_relative_span
        )
        self._axisym_gain_representative_frequency_Hz = np.asarray(
            representative_frequencies,
            dtype=float,
        )
        lower_gain = np.zeros(len(fine_frequency), dtype=int)
        upper_gain = np.zeros(len(fine_frequency), dtype=int)
        gain_weight = np.zeros(len(fine_frequency), dtype=float)
        central_positions = self._axisym_gain_central_group_positions
        central_representatives = self._axisym_gain_representative_frequency_Hz[
            central_positions
        ]
        central_mask = (
            (fine_frequency >= self._axisym_gain_resolved_frequency_range_Hz[0])
            & (fine_frequency <= self._axisym_gain_resolved_frequency_range_Hz[1])
        )
        if len(central_positions) == 1:
            lower_gain[central_mask] = int(central_positions[0])
            upper_gain[central_mask] = int(central_positions[0])
        else:
            active_frequency = fine_frequency[central_mask]
            active_upper = np.searchsorted(
                central_representatives,
                active_frequency,
                side="right",
            )
            active_upper = np.clip(active_upper, 1, len(central_positions) - 1)
            active_lower = active_upper - 1
            below = active_frequency <= central_representatives[0]
            above = active_frequency >= central_representatives[-1]
            active_lower[below] = 0
            active_upper[below] = 0
            active_lower[above] = len(central_positions) - 1
            active_upper[above] = active_lower[above]
            lower_frequency = central_representatives[active_lower]
            upper_frequency = central_representatives[active_upper]
            active_weight = np.divide(
                active_frequency - lower_frequency,
                upper_frequency - lower_frequency,
                out=np.zeros_like(active_frequency),
                where=upper_frequency > lower_frequency,
            )
            lower_gain[central_mask] = central_positions[active_lower]
            upper_gain[central_mask] = central_positions[active_upper]
            gain_weight[central_mask] = np.clip(active_weight, 0.0, 1.0)
        if len(lower_guard):
            lower_gain[lower_guard] = 0
            upper_gain[lower_guard] = 0
        if len(upper_guard):
            upper_guard_position = len(self._axisym_gain_groups) - 1
            lower_gain[upper_guard] = upper_guard_position
            upper_gain[upper_guard] = upper_guard_position
        self._axisym_gain_interp_lower = lower_gain
        self._axisym_gain_interp_upper = upper_gain
        self._axisym_gain_interp_upper_weight = np.clip(gain_weight, 0.0, 1.0)
        self._axisym_gain_profile = DiscreteProfile(
            np.asarray(representative_wavelengths, dtype=float),
            np.full(
                len(representative_wavelengths),
                1.0 / len(representative_wavelengths),
                dtype=float,
            ),
        )

        self._tail_frequency_plan = None
        from .config import resolve_signal_model
        tail = self.cfg.get("tail_binning", {})
        if tail.get("spectral_enabled", False) and resolve_signal_model({
            "seed": self.cfg, "simulation": self.simulation, "physics": self.physics,
        }) == "coherent_cpa":
            self._configure_input_tail_bins(tail)

        x_center = 0.5 * float(self.grid.x_edges[0] + self.grid.x_edges[-1])
        y_center = 0.5 * float(self.grid.y_edges[0] + self.grid.y_edges[-1])
        self._axisym_center_x_m = x_center
        self._axisym_center_y_m = y_center
        self.radial_extent_m = 0.5 * min(
            float(np.ptp(self.grid.x_edges)),
            float(np.ptp(self.grid.y_edges)),
        )
        self.radial_edges_m = np.linspace(
            0.0, self.radial_extent_m, self.radial_bins + 1
        )
        self.radial_centers_m = 0.5 * (
            self.radial_edges_m[:-1] + self.radial_edges_m[1:]
        )
        self.radial_step_m = float(self.radial_edges_m[1])
        xx, yy = np.meshgrid(
            self.grid.x - x_center,
            self.grid.y - y_center,
            indexing="xy",
        )
        radius = np.hypot(xx, yy)
        raw_bins = np.floor(radius / self.radial_step_m).astype(int)
        self._axisym_valid_xy = raw_bins < self.radial_bins
        self._axisym_bin_xy = np.clip(raw_bins, 0, self.radial_bins - 1)
        self._axisym_valid_flat = self._axisym_valid_xy.ravel()
        self._axisym_bin_flat = self._axisym_bin_xy.ravel()
        self._axisym_flat_indices_by_bin = tuple(
            np.flatnonzero(
                self._axisym_valid_flat
                & (self._axisym_bin_flat == radial_index)
            )
            for radial_index in range(self.radial_bins)
        )
        cell_area_flat = np.asarray(self.grid.cell_areas, dtype=float).ravel()
        self._axisym_cell_area_flat = cell_area_flat
        self.cartesian_annulus_area_m2 = np.bincount(
            self._axisym_bin_flat[self._axisym_valid_flat],
            weights=cell_area_flat[self._axisym_valid_flat],
            minlength=self.radial_bins,
        ).astype(float)
        if np.any(self.cartesian_annulus_area_m2 <= 0.0):
            raise ValueError(
                "axisymmetric radial_bins is too large for the Cartesian optical grid; "
                "at least one radial annulus contains no x/y cell centres"
            )
        self.radial_area_m2 = math.pi * (
            self.radial_edges_m[1:] ** 2 - self.radial_edges_m[:-1] ** 2
        )
        self._axisym_conservative_bin_scale = (
            self.radial_area_m2 / self.cartesian_annulus_area_m2
        )

        radial_position = radius.ravel()[self._axisym_valid_flat]
        lower, upper, upper_weight = _radial_linear_interpolation_indices(
            radial_position,
            self.radial_step_m,
            self.radial_bins,
        )
        self._axisym_interp_lower = lower
        self._axisym_interp_upper = upper
        self._axisym_interp_upper_weight = upper_weight

        # Weighted least-squares projector for the piecewise-linear radial
        # representation used at the Cartesian/radial interface.  A hard
        # annulus average loses the rapidly varying quadratic wavefront near a
        # focus even when the field is perfectly axisymmetric; the projector
        # below measures only content that the actual radial representation
        # cannot carry.  Its normal matrix is tridiagonal because every x/y
        # sample touches at most two neighbouring radial nodes.
        valid_areas = cell_area_flat[self._axisym_valid_flat]
        lower_weight = 1.0 - upper_weight
        projection_diagonal = np.zeros(self.radial_bins, dtype=float)
        projection_off_diagonal = np.zeros(self.radial_bins - 1, dtype=float)
        np.add.at(
            projection_diagonal,
            lower,
            valid_areas * lower_weight**2,
        )
        np.add.at(
            projection_diagonal,
            upper,
            valid_areas * upper_weight**2,
        )
        adjacent = upper == lower + 1
        np.add.at(
            projection_off_diagonal,
            lower[adjacent],
            valid_areas[adjacent]
            * lower_weight[adjacent]
            * upper_weight[adjacent],
        )
        cholesky_diagonal = np.empty(self.radial_bins, dtype=float)
        cholesky_subdiagonal = np.empty(self.radial_bins - 1, dtype=float)
        cholesky_diagonal[0] = math.sqrt(projection_diagonal[0])
        for radial_index in range(1, self.radial_bins):
            cholesky_subdiagonal[radial_index - 1] = (
                projection_off_diagonal[radial_index - 1]
                / cholesky_diagonal[radial_index - 1]
            )
            remainder = (
                projection_diagonal[radial_index]
                - cholesky_subdiagonal[radial_index - 1] ** 2
            )
            if remainder <= np.finfo(float).tiny:
                raise ValueError(
                    "axisymmetric Cartesian/radial projection is singular; "
                    "reduce seed.coherent.axisymmetric.radial_bins"
                )
            cholesky_diagonal[radial_index] = math.sqrt(remainder)
        self._axisym_projection_lower_weight = lower_weight
        self._axisym_projection_cholesky_diagonal = cholesky_diagonal
        self._axisym_projection_cholesky_subdiagonal = cholesky_subdiagonal

        # Finite-volume radial Laplacian at annular cell centres.  The outer
        # face uses zero normal derivative; applicability is guarded by the
        # measured outer-annulus energy, so that boundary must remain dark.
        j = np.arange(self.radial_bins, dtype=float)
        denominator = np.maximum(j + 0.5, np.finfo(float).tiny)
        lower_l = j / denominator / self.radial_step_m**2
        upper_l = (j + 1.0) / denominator / self.radial_step_m**2
        diagonal_l = -(lower_l + upper_l)
        lower_l[0] = 0.0
        upper_l[-1] = 0.0
        diagonal_l[-1] = -lower_l[-1]
        self._radial_laplacian_lower = lower_l
        self._radial_laplacian_diagonal = diagonal_l
        self._radial_laplacian_upper = upper_l
        self._radial_laplacian_diagonal_by_order = {
            0: self._radial_laplacian_diagonal
        }

        self.angular_coordinate_rad = np.linspace(
            0.0,
            2.0 * math.pi,
            self.angular_samples,
            endpoint=False,
        )
        self.angular_cell_area_m2 = (
            self.radial_area_m2[None, :] / self.angular_samples
        )
        angular_orders = np.rint(
            np.fft.fftfreq(self.angular_samples, d=1.0 / self.angular_samples)
        ).astype(int)
        self._angular_fft_orders = angular_orders
        self._retained_angular_fft_mask = (
            np.abs(angular_orders) <= self.maximum_azimuthal_order
        )
        self._retained_angular_indices = np.flatnonzero(
            self._retained_angular_fft_mask
        )

        polar_x = x_center + np.cos(self.angular_coordinate_rad)[:, None] * (
            self.radial_centers_m[None, :]
        )
        polar_y = y_center + np.sin(self.angular_coordinate_rad)[:, None] * (
            self.radial_centers_m[None, :]
        )
        x_fraction = (polar_x - float(self.grid.x[0])) / float(self.grid.dx)
        y_fraction = (polar_y - float(self.grid.y[0])) / float(self.grid.dy)
        x_lower = np.clip(
            np.floor(x_fraction).astype(int), 0, len(self.grid.x) - 2
        )
        y_lower = np.clip(
            np.floor(y_fraction).astype(int), 0, len(self.grid.y) - 2
        )
        self._polar_x_lower = x_lower
        self._polar_y_lower = y_lower
        self._polar_x_weight = np.clip(x_fraction - x_lower, 0.0, 1.0)
        self._polar_y_weight = np.clip(y_fraction - y_lower, 0.0, 1.0)

        cartesian_theta = np.mod(
            np.arctan2(yy, xx), 2.0 * math.pi
        ).ravel()[self._axisym_valid_flat]
        angular_fraction = (
            cartesian_theta * self.angular_samples / (2.0 * math.pi)
        )
        self._cartesian_angular_lower = (
            np.floor(angular_fraction).astype(int) % self.angular_samples
        )
        self._cartesian_angular_upper = (
            self._cartesian_angular_lower + 1
        ) % self.angular_samples
        self._cartesian_angular_upper_weight = (
            angular_fraction - np.floor(angular_fraction)
        )
        self._cartesian_theta_valid = cartesian_theta
        self._radial_cn_cache: OrderedDict[
            tuple[float, ...], tuple[np.ndarray, ...]
        ] = OrderedDict()
        self._radial_cn_cache_bytes = 0
        self._radial_cn_cache_hits = 0
        self._radial_cn_cache_misses = 0
        self._radial_cn_cache_evictions = 0
        self._radial_cn_cache_oversize_bypasses = 0
        self._radial_diffraction_scratch: dict[str, np.ndarray] = {}
        self._axisym_mask = np.ones(
            (self.angular_samples, self.radial_bins), dtype=float
        )
        self._axisym_population: np.ndarray | None = None
        self._axisym_population_initial: np.ndarray | None = None
        self._axisym_passive_loss: np.ndarray | None = None
        self._axisym_passive_loss_initial: np.ndarray | None = None
        self._axisym_override_gain: np.ndarray | None = None
        self._axisym_override_gain_initial: np.ndarray | None = None
        self._axisym_delta_n: np.ndarray | None = None
        self._axisym_active_free_carrier_initial_m3: np.ndarray | None = None
        self._axisym_free_carrier_initial_cartesian_m3: np.ndarray | None = None
        self._axisym_free_carrier_initial_radial_m3: np.ndarray | None = None
        self._axisym_free_carrier_final_radial_m3: np.ndarray | None = None
        self._axisym_environment_original: tuple[
            np.ndarray | float, np.ndarray | float
        ] | None = None
        self._reset_axisymmetric_diagnostics()
        temporal_closure = (
            "front-to-tail soft-window transient gain"
            if self.coherent_fidelity_mode == "axisymmetric_transient_gain"
            else "whole-fluence Frantz-Nodvik gain"
        )
        carrier_closure = (
            ", and radial time-ordered free-carrier generation/response"
            if self.physics.get("nonlinear_loss_model", "off")
            == "strong_field_time_domain"
            and self.material.get("strong_field", {})
            .get("free_carrier", {})
            .get("model", "off")
            == "rate_equation"
            else ""
        )
        retained_order_text = (
            "only m=0"
            if self.maximum_azimuthal_order == 0
            else f"m=-{self.maximum_azimuthal_order}…+{self.maximum_azimuthal_order}"
        )
        quadratic_text = (
            " Material m=2 is constrained to a quadratic r² closure."
            if self.azimuthal_fidelity == "m0_m1_m2_quadratic"
            else ""
        )
        self._add_persistent_warning(
            "Low-order azimuthal coherent Kerr mode is active: the crystal "
            f"transports E(f,r,θ) with {retained_order_text}, including "
            "order-dependent radial diffraction/self-focusing, spectral "
            f"variation, dispersion, local time-domain Kerr, {temporal_closure}"
            f"{carrier_closure}.{quadratic_text} This is not a Cartesian "
            "full-field calculation and is valid only while the reported "
            "projection, omitted-order, and radial-boundary gates pass."
        )

    def _new_transient_gain(self) -> Any:
        from .coherent_gain import CoherentTransientGain

        return CoherentTransientGain(
            self.pulse,
            self.levels,
            self.levels_cfg,
            self.material,
            self.physics,
            self.transient_gain_cfg,
            np.broadcast_to(
                self.angular_cell_area_m2,
                (self.angular_samples, self.radial_bins),
            ),
            self._axisym_mask,
        )

    def _transient_gain_population_shape(
        self,
        population: np.ndarray,
    ) -> tuple[int, ...]:
        return (
            int(population.shape[0]),
            int(population.shape[1]),
            int(self.angular_samples),
            int(self.radial_bins),
        )

    def _transport_progress_metrics(self) -> dict[str, Any]:
        metrics = super()._transport_progress_metrics()
        metrics.update(
            {
                "spatial_coordinate_model": (
                    "axisymmetric_radius"
                    if self.maximum_azimuthal_order == 0
                    else "finite_azimuthal_fourier_radius"
                ),
                "axisymmetric_radial_bins": int(self.radial_bins),
                "azimuthal_fidelity": self.azimuthal_fidelity,
                "maximum_azimuthal_order": int(
                    self.maximum_azimuthal_order
                ),
                "azimuthal_angular_samples": int(self.angular_samples),
            }
        )
        return metrics

    def _validate_axisymmetric_contract(self) -> None:
        if not self.grid.is_uniform_xy:
            raise ValueError("axisymmetric coherent Kerr requires a uniform optical grid")
        size_x = float(np.ptp(self.grid.x_edges))
        size_y = float(np.ptp(self.grid.y_edges))
        if not math.isclose(size_x, size_y, rel_tol=1e-10, abs_tol=1e-15):
            raise ValueError("axisymmetric coherent Kerr requires a square optical window")
        if self.radial_bins < 8 or self.radial_bins > min(len(self.grid.x), len(self.grid.y)):
            raise ValueError(
                "seed.coherent.axisymmetric.radial_bins must lie between 8 and "
                "the smaller optical-grid dimension"
            )
        if self.gain_spectral_bins < 3:
            raise ValueError(
                "seed.coherent.axisymmetric.gain_spectral_bins must be at least 3"
            )
        if int(self.simulation.get("passes", 1)) != 1 or len(self.gain_markers) > 1:
            raise ValueError(
                "axisymmetric coherent Kerr currently requires one gain-medium pass"
            )
        transient_enabled = bool(
            self.cfg.get("coherent", {})
            .get("transient_gain", {})
            .get("enabled", False)
        )
        if self.coherent_fidelity_mode == "axisymmetric_kerr" and transient_enabled:
            raise ValueError(
                "axisymmetric_kerr requires whole-fluence FN gain; use "
                "axisymmetric_transient_gain for soft-window front-to-tail gain"
            )
        if (
            self.coherent_fidelity_mode == "axisymmetric_transient_gain"
            and not transient_enabled
        ):
            raise ValueError(
                "axisymmetric_transient_gain requires soft-window transient gain"
            )
        if str(self.cfg.get("polarization", {}).get("model", "scalar")) != "scalar":
            raise ValueError("axisymmetric coherent Kerr currently requires scalar polarization")
        if self.maximum_azimuthal_order < 2:
            for first, second, label in (
                (
                    self.cfg.get("waist_x_m"),
                    self.cfg.get("waist_y_m"),
                    "Seed waist",
                ),
                (
                    self.cfg.get("order_x", 1.0),
                    self.cfg.get("order_y", 1.0),
                    "Seed super-Gaussian order",
                ),
                (
                    axis_focus_m(self.cfg, "x"),
                    axis_focus_m(self.cfg, "y"),
                    "Seed focus",
                ),
            ):
                if first is not None and second is not None and not math.isclose(
                    float(first), float(second), rel_tol=1e-8, abs_tol=1e-15
                ):
                    raise ValueError(
                        "the selected axisymmetric azimuthal fidelity requires "
                        f"equal x/y {label}; choose an m=2 option"
                    )
        for key in ("m2_x", "m2_y"):
            if not math.isclose(
                float(self.cfg.get(key, 1.0)),
                1.0,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "axisymmetric coherent Kerr propagates an explicit complex "
                    "field and therefore requires seed.m2_x=seed.m2_y=1. "
                    "An empirical M2 multiplier in the radial wave equation "
                    "would double-count beam quality; represent non-Gaussian "
                    "content with a resolved field/backend instead."
                )
        window_scale = max(size_x, size_y)
        centre_tolerance = max(1e-15, 1e-12 * window_scale)
        if self.maximum_azimuthal_order < 1:
            for configured, edges in (
                (self.cfg.get("center_x_m", 0.0), self.grid.x_edges),
                (self.cfg.get("center_y_m", 0.0), self.grid.y_edges),
            ):
                grid_centre = 0.5 * float(edges[0] + edges[-1])
                if not math.isclose(
                    float(configured),
                    grid_centre,
                    rel_tol=0.0,
                    abs_tol=centre_tolerance,
                ):
                    raise ValueError(
                        "m=0 axisymmetric coherent Kerr requires the Seed centre "
                        "to coincide with the optical-grid centre; choose m=0+1"
                    )
            if any(
                abs(float(self.cfg.get(key, 0.0) or 0.0)) > 1e-15
                for key in (
                    "incidence_angle_x_deg",
                    "incidence_angle_y_deg",
                )
            ):
                raise ValueError(
                    "m=0 axisymmetric coherent Kerr requires a centred, "
                    "untilted Seed; choose m=0+1"
                )
        for marker in self.gain_markers or [self.gain_marker]:
            angle_x, angle_y = gain_medium_axis_angles_rad(marker)
            if abs(angle_x) + abs(angle_y) > 1e-15:
                raise ValueError(
                    "axisymmetric coherent Kerr requires normal incidence through the gain medium"
                )
        # The reduced disk must be fully inside the material aperture.  This
        # still permits a larger rectangular crystal, as in the Cr:ZnS case.
        yy, xx = np.indices(self.crystal_mask.shape)
        del yy, xx
        if np.any(self.crystal_mask < 1.0 - 1e-12):
            raise ValueError(
                "axisymmetric coherent Kerr currently requires the crystal aperture to cover the complete optical window"
            )

    def _configure_input_tail_bins(self, settings: dict[str, Any]) -> None:
        from .energy_tail_binning import energy_tail_groups
        f = self.pulse.frequency_Hz
        plan = energy_tail_groups(f, self.pulse.spectral_energy_J,
            coverage=settings.get("energy_coverage", .999999),
            tail_points=settings.get("spectral_tail_points", 3))
        self._tail_frequency_plan = plan
        groups = plan["groups"]
        left, right = plan["tail_counts"]
        self._axisym_gain_groups = groups
        self._axisym_gain_group_roles = tuple(
            ["lower_guard"]*left + ["resolved"]*plan["central_count"] + ["upper_guard"]*right)
        self._axisym_gain_central_group_positions = np.arange(left, left+plan["central_count"])
        lo, hi = plan["central_indices"]
        self._axisym_gain_resolved_frequency_range_Hz = (float(f[lo]), float(f[hi]))
        reps = plan["coordinate"]
        self._axisym_gain_representative_frequency_Hz = reps
        lower = np.clip(np.searchsorted(reps, f, side="right")-1, 0, len(reps)-1)
        upper = np.minimum(lower+1, len(reps)-1)
        upper[f <= reps[0]] = 0
        weight = np.divide(f-reps[lower], reps[upper]-reps[lower],
            out=np.zeros_like(f), where=upper != lower)
        self._axisym_gain_interp_lower = lower
        self._axisym_gain_interp_upper = upper
        self._axisym_gain_interp_upper_weight = np.clip(weight, 0., 1.)
        self._axisym_gain_profile = DiscreteProfile(299792458./reps, np.full(len(reps), 1./len(reps)))
        self.gain_spectral_bins = len(groups)
        self.axisym_maximum_gain_bin_relative_frequency_span = 0.

    def _reset_axisymmetric_diagnostics(self) -> None:
        self.radial_diffraction_kernel_calls = 0
        self.radial_diffraction_numba_cold_starts = 0
        self._radial_cn_cache_hits = 0
        self._radial_cn_cache_misses = 0
        self._radial_cn_cache_evictions = 0
        self._radial_cn_cache_oversize_bypasses = 0
        self.axisym_minimum_projection_energy_fraction = 1.0
        self.axisym_total_projection_energy_fraction = 1.0
        # Keep the legacy worst-frequency value for forensic diagnostics, but
        # do not use it as the physical boundary-validity gate.  A nearly
        # empty FFT tail can have an arbitrary spatial ratio while carrying no
        # meaningful fraction of the pulse energy.
        self.axisym_maximum_boundary_energy_fraction = 0.0
        self.axisym_maximum_energy_weighted_boundary_fraction = 0.0
        self.axisym_worst_boundary_component_energy_fraction = 0.0
        self.axisym_worst_boundary_component_outer_energy_fraction = 0.0
        self.axisym_worst_boundary_component_wavelength_m: float | None = None
        self.axisym_maximum_population_azimuthal_residual = 0.0
        self.axisym_maximum_delta_n_azimuthal_residual = 0.0
        self.axisym_maximum_free_carrier_azimuthal_residual = 0.0
        self.axisym_maximum_cheap_m2_material_fit_residual = 0.0
        self.axisym_maximum_dynamic_field_omitted_energy_fraction = 0.0
        self.axisym_maximum_population_realizability_correction_fraction = 0.0
        self.axisym_maximum_population_positivity_redistribution_fraction = 0.0
        self.axisym_maximum_diffraction_norm_correction = 0.0
        self.axisym_population_mapping_relative_error = 0.0
        self.axisym_free_carrier_mapping_relative_error = 0.0
        self.axisym_maximum_gain_reconstruction_normalization_correction = 0.0
        self.axisym_passes_reduced = 0
        self._axisym_active_free_carrier_initial_m3 = None
        self._axisym_free_carrier_initial_cartesian_m3 = np.zeros(
            self.grid.shape,
            dtype=float,
        )
        carrier_reduced_shape = (
            (len(self.grid.z), self.radial_bins)
            if self.maximum_azimuthal_order == 0
            else (
                len(self.grid.z),
                self.angular_samples,
                self.radial_bins,
            )
        )
        self._axisym_free_carrier_initial_radial_m3 = np.zeros(
            carrier_reduced_shape, dtype=float
        )
        self._axisym_free_carrier_final_radial_m3 = np.zeros(
            carrier_reduced_shape, dtype=float
        )
        self._axisym_free_carrier_generated_m3 = np.zeros(
            carrier_reduced_shape, dtype=float
        )
        self._axisym_free_carrier_survival = np.zeros(len(self.grid.z))

    def _update_axisymmetric_boundary_diagnostics(
        self,
        radial_energy: np.ndarray,
        outer_annulus_energy: np.ndarray,
    ) -> None:
        """Track both pulse-weighted and worst-frequency boundary fractions.

        The propagated coherent field is represented on a uniform FFT
        frequency grid, so summing the per-frequency radial energies gives the
        correct pulse-energy ratio (the common frequency-bin width cancels).
        The weighted ratio is the validity metric.  The worst-frequency ratio
        remains available together with that frequency's share of total energy
        so a harmless spectral tail can be distinguished from a real aperture
        failure.
        """

        energy = np.asarray(radial_energy, dtype=float)
        outer = np.asarray(outer_annulus_energy, dtype=float)
        if energy.shape != outer.shape or energy.size == 0:
            raise ValueError(
                "axisymmetric boundary diagnostics require matching nonempty energy arrays"
            )
        total_energy = float(np.sum(energy))
        if not np.isfinite(total_energy) or total_energy <= np.finfo(float).tiny:
            return
        weighted_fraction = float(np.sum(outer)) / total_energy
        if np.isfinite(weighted_fraction):
            self.axisym_maximum_energy_weighted_boundary_fraction = max(
                self.axisym_maximum_energy_weighted_boundary_fraction,
                weighted_fraction,
            )

        frequency_count = len(self.pulse.frequency_Hz)
        if energy.ndim >= 1 and energy.shape[-1] == frequency_count:
            reduction_axes = tuple(range(energy.ndim - 1))
            frequency_energy = (
                np.sum(energy, axis=reduction_axes)
                if reduction_axes
                else energy
            )
            frequency_outer = (
                np.sum(outer, axis=reduction_axes)
                if reduction_axes
                else outer
            )
            wavelengths_m: np.ndarray | None = np.asarray(
                self.pulse.wavelength_m,
                dtype=float,
            )
        else:
            # This fallback keeps the helper valid for a future representation
            # whose leading axes cannot be identified as frequency.  It still
            # records the energy significance, but not a misleading wavelength.
            frequency_energy = energy.reshape(-1)
            frequency_outer = outer.reshape(-1)
            wavelengths_m = None

        peak_energy = float(np.max(frequency_energy))
        active = frequency_energy > np.finfo(float).eps * peak_energy
        if not np.any(active):
            return
        per_frequency_fraction = np.divide(
            frequency_outer,
            frequency_energy,
            out=np.zeros_like(frequency_outer),
            where=frequency_energy > 0.0,
        )
        active_indices = np.flatnonzero(active)
        worst_index = int(
            active_indices[
                np.argmax(per_frequency_fraction[active_indices])
            ]
        )
        worst_fraction = float(per_frequency_fraction[worst_index])
        if worst_fraction <= self.axisym_maximum_boundary_energy_fraction:
            return
        component_energy_fraction = (
            float(frequency_energy[worst_index]) / total_energy
        )
        self.axisym_maximum_boundary_energy_fraction = worst_fraction
        self.axisym_worst_boundary_component_energy_fraction = (
            component_energy_fraction
        )
        self.axisym_worst_boundary_component_outer_energy_fraction = (
            float(frequency_outer[worst_index]) / total_energy
        )
        self.axisym_worst_boundary_component_wavelength_m = (
            None
            if wavelengths_m is None
            else float(wavelengths_m[worst_index])
        )

    def propagate(self, population: np.ndarray, delta_n: np.ndarray):
        self._reset_axisymmetric_diagnostics()
        try:
            return super().propagate(population, delta_n)
        finally:
            if self._axisym_environment_original is not None:
                self.levels.set_environment(*self._axisym_environment_original)
                self._axisym_environment_original = None

    def _is_reduced_field(self, field: np.ndarray) -> bool:
        # ``angular_samples`` was introduced with the finite-azimuthal
        # extension.  Keep the low-level m=0 hooks compatible with lightweight
        # test doubles (and downstream subclasses) that construct the solver
        # without running ``__init__``.
        angular_samples = int(getattr(self, "angular_samples", 1))
        return tuple(np.asarray(field).shape[-2:]) == (
            angular_samples,
            self.radial_bins,
        )

    def _sample_cartesian_on_polar_grid(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """Bilinearly sample Cartesian trailing y/x axes on (theta, r)."""

        array = np.asarray(values)
        if array.shape[-2:] != self.crystal_mask.shape:
            raise ValueError(
                "azimuthal projection requires the Cartesian optical-grid shape"
            )
        y0 = self._polar_y_lower
        x0 = self._polar_x_lower
        wx = self._polar_x_weight
        wy = self._polar_y_weight
        return (
            array[..., y0, x0] * (1.0 - wx) * (1.0 - wy)
            + array[..., y0, x0 + 1] * wx * (1.0 - wy)
            + array[..., y0 + 1, x0] * (1.0 - wx) * wy
            + array[..., y0 + 1, x0 + 1] * wx * wy
        )

    def _truncate_angular_samples(
        self,
        values: np.ndarray,
        *,
        real_output: bool = False,
        quadratic_material_m2: bool = False,
        track_field_omission: bool = False,
    ) -> np.ndarray:
        """Project angle samples onto the configured finite Fourier basis."""

        array = np.asarray(values)
        if array.shape[-2:] != (self.angular_samples, self.radial_bins):
            raise ValueError("reduced azimuthal array has the wrong shape")
        if self.maximum_azimuthal_order == 0:
            averaged = np.mean(array, axis=-2, keepdims=True)
            return np.real(averaged) if real_output else averaged
        coefficients = np.fft.fft(array, axis=-2) / self.angular_samples
        if track_field_omission:
            modal_energy = np.abs(coefficients) ** 2 * self.radial_area_m2
            full_modal_energy = float(np.sum(modal_energy))
            retained_modal_energy = float(
                np.sum(
                    modal_energy[..., self._retained_angular_fft_mask, :]
                )
            )
            if full_modal_energy > np.finfo(float).tiny:
                omitted_fraction = max(
                    0.0,
                    1.0 - retained_modal_energy / full_modal_energy,
                )
                self.axisym_maximum_dynamic_field_omitted_energy_fraction = max(
                    self.axisym_maximum_dynamic_field_omitted_energy_fraction,
                    omitted_fraction,
                )
        coefficients[..., ~self._retained_angular_fft_mask, :] = 0.0
        if (
            quadratic_material_m2
            and self.azimuthal_fidelity == "m0_m1_m2_quadratic"
        ):
            coefficients = self._fit_quadratic_material_m2(coefficients)
        projected = np.fft.ifft(
            coefficients * self.angular_samples,
            axis=-2,
        )
        return np.real(projected) if real_output else projected

    def _fit_quadratic_material_m2(
        self,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        """Replace a sampled material m=2 radial function by q*r²."""

        output = np.array(coefficients, copy=True)
        positive = np.flatnonzero(self._angular_fft_orders == 2)
        negative = np.flatnonzero(self._angular_fft_orders == -2)
        if not len(positive) or not len(negative):
            return output
        positive_index = int(positive[0])
        negative_index = int(negative[0])
        radius_squared = self.radial_centers_m**2
        weights = self.radial_area_m2
        denominator = float(np.sum(weights * radius_squared**2))
        original = output[..., positive_index, :]
        original_negative = output[..., negative_index, :]
        coefficient = np.sum(
            original * (weights * radius_squared), axis=-1
        ) / max(denominator, np.finfo(float).tiny)
        fitted = coefficient[..., None] * radius_squared
        fitted_negative = np.conjugate(fitted)
        residual_energy = float(
            np.sum(
                (
                    np.abs(original - fitted) ** 2
                    + np.abs(original_negative - fitted_negative) ** 2
                )
                * weights
            )
        )
        material_energy = float(
            np.sum(np.abs(output) ** 2 * weights)
        )
        if material_energy > np.finfo(float).tiny:
            residual = math.sqrt(residual_energy / material_energy)
            self.axisym_maximum_cheap_m2_material_fit_residual = max(
                self.axisym_maximum_cheap_m2_material_fit_residual,
                residual,
            )
        output[..., positive_index, :] = fitted
        output[..., negative_index, :] = fitted_negative
        return output

    def _enforce_cheap_material_m2_state(self) -> None:
        if self.azimuthal_fidelity != "m0_m1_m2_quadratic":
            return
        if self._axisym_population is not None:
            fitted_population = self._truncate_angular_samples(
                self._axisym_population,
                real_output=True,
                quadratic_material_m2=True,
            )
            self._axisym_population[:] = (
                self._make_angular_population_realizable(
                    fitted_population,
                    optimization_iterations=0,
                )
            )
        if self._axisym_override_gain is not None:
            self._axisym_override_gain[:] = self._truncate_angular_samples(
                self._axisym_override_gain,
                real_output=True,
                quadratic_material_m2=True,
            )

    def _project_cartesian_scalar_to_angular(
        self,
        values: np.ndarray,
        *,
        quadratic_material_m2: bool = True,
    ) -> np.ndarray:
        """Project real Cartesian material fields to retained angle samples."""

        if self.maximum_azimuthal_order == 0:
            return self._radial_average(values)[..., None, :]
        polar = self._sample_cartesian_on_polar_grid(values)
        return self._truncate_angular_samples(
            polar,
            real_output=True,
            quadratic_material_m2=quadratic_material_m2,
        )

    def _make_angular_population_realizable(
        self,
        values: np.ndarray,
        *,
        optimization_iterations: int = 64,
    ) -> np.ndarray:
        """Return a non-negative, conservative finite-mode population state.

        Truncating the Fourier series of each level density independently can
        make a sparsely populated level negative even though the Cartesian
        state is physical.  The gain-event limiter cannot consume such a
        signed reservoir: its final roundoff clip would create ions.  Project
        the angular state onto the intersection of the retained Fourier
        subspace and the non-negative population cone instead.

        The affine projection preserves every level's annular mean and the
        local sum over levels.  Dykstra iterations retain substantially more
        of the requested low-order angular pattern than uniformly damping all
        modes.  A final convex blend toward a feasible proportional state
        guarantees non-negativity even when the fixed iteration budget has
        not fully converged.
        """

        population = np.asarray(values, dtype=float)
        if population.ndim != 4:
            raise ValueError(
                "finite-azimuthal population state must be four-dimensional"
            )
        expected_shape = (
            population.shape[0],
            len(self.grid.z),
            self.angular_samples,
            self.radial_bins,
        )
        if population.shape != expected_shape:
            raise ValueError(
                "finite-azimuthal population state has an incompatible shape"
            )
        if np.all(population >= 0.0):
            return population

        level_count = population.shape[0]
        tiny = np.finfo(float).tiny
        annular_scale = np.maximum(
            np.max(np.abs(population), axis=(0, 2), keepdims=True),
            1.0,
        )
        level_means = np.mean(population, axis=-2, keepdims=True)
        if np.any(level_means < -1e-10 * annular_scale):
            raise ValueError(
                "finite-azimuthal population projection has a negative "
                "annular level inventory; increase retained order or use "
                "full x-y"
            )
        level_means = np.maximum(level_means, 0.0)

        local_total = np.sum(population, axis=0)
        total_scale = np.maximum(
            np.max(np.abs(local_total), axis=-2, keepdims=True),
            1.0,
        )
        if np.any(local_total < -1e-10 * total_scale):
            raise ValueError(
                "finite-azimuthal population projection has a negative "
                "local dopant inventory"
            )
        local_total = np.maximum(local_total, 0.0)
        total_mean = np.mean(local_total, axis=-2, keepdims=True)
        mean_sum = np.sum(level_means, axis=0)
        level_means *= np.divide(
            total_mean,
            mean_sum,
            out=np.ones_like(total_mean),
            where=mean_sum > tiny,
        )[None, ...]
        local_total *= np.divide(
            total_mean,
            np.mean(local_total, axis=-2, keepdims=True),
            out=np.ones_like(total_mean),
            where=total_mean > tiny,
        )

        # This proportional allocation is always feasible, has the requested
        # level means and local total, and remains in the retained Fourier
        # subspace because local_total is itself a retained projection.
        feasible = level_means * np.divide(
            local_total,
            total_mean,
            out=np.zeros_like(local_total),
            where=total_mean > tiny,
        )[None, ...]
        target_level_mean = level_means[..., 0, :]
        target_total_coefficients = (
            np.fft.fft(local_total, axis=-2) / self.angular_samples
        )
        zero_mode = int(
            np.flatnonzero(self._angular_fft_orders == 0)[0]
        )
        retained_nonzero = np.flatnonzero(
            self._retained_angular_fft_mask
            & (self._angular_fft_orders != 0)
        )

        def project_affine(array: np.ndarray) -> np.ndarray:
            coefficients = (
                np.fft.fft(array, axis=-2) / self.angular_samples
            )
            coefficients[..., ~self._retained_angular_fft_mask, :] = 0.0
            coefficients[..., zero_mode, :] = target_level_mean
            if len(retained_nonzero):
                discrepancy = (
                    np.sum(
                        coefficients[..., retained_nonzero, :],
                        axis=0,
                    )
                    - target_total_coefficients[
                        ..., retained_nonzero, :
                    ]
                )
                coefficients[..., retained_nonzero, :] -= (
                    discrepancy[None, ...] / level_count
                )
            return np.real(
                np.fft.ifft(
                    coefficients * self.angular_samples,
                    axis=-2,
                )
            )

        candidate = population.copy()
        correction_affine = np.zeros_like(candidate)
        correction_positive = np.zeros_like(candidate)
        for _ in range(max(int(optimization_iterations), 0)):
            affine_input = candidate + correction_affine
            affine = project_affine(affine_input)
            correction_affine = affine_input - affine
            positive_input = affine + correction_positive
            candidate = np.maximum(positive_input, 0.0)
            correction_positive = positive_input - candidate
        if optimization_iterations:
            candidate = project_affine(candidate + correction_affine)

        # The finite Dykstra budget can leave a very small negative residual.
        # A convex blend with ``feasible`` stays inside every affine
        # conservation constraint and reaches the non-negative cone exactly.
        direction = candidate - feasible
        limits = np.full_like(candidate, np.inf)
        np.divide(
            feasible,
            -direction,
            out=limits,
            where=direction < 0.0,
        )
        blend = np.clip(
            np.min(limits, axis=(0, 2)),
            0.0,
            1.0,
        )
        blend = np.where(
            blend < 1.0,
            blend * (1.0 - 1e-12),
            blend,
        )
        corrected = feasible + blend[None, :, None, :] * direction
        corrected = np.maximum(corrected, 0.0)
        corrected_total = np.sum(corrected, axis=0)
        corrected *= np.divide(
            local_total,
            corrected_total,
            out=np.ones_like(local_total),
            where=corrected_total > tiny,
        )[None, ...]

        correction_weights = (
            self.grid.z_widths[None, :, None, None]
            * self.angular_cell_area_m2[None, None, :, :]
        )
        inventory = float(
            np.sum(local_total * correction_weights[0])
        )
        redistributed = 0.5 * float(
            np.sum(np.abs(corrected - population) * correction_weights)
        )
        if inventory > tiny:
            self.axisym_maximum_population_realizability_correction_fraction = max(
                self.axisym_maximum_population_realizability_correction_fraction,
                redistributed / inventory,
            )
        negative_tolerance = 1e-10 * max(
            float(np.max(np.abs(corrected))),
            1.0,
        )
        if float(np.min(corrected)) < -negative_tolerance:
            raise ValueError(
                "finite-azimuthal population realizability projection did "
                "not reach a non-negative state"
            )
        return corrected

    def _expand_angular_nearest(
        self,
        values: np.ndarray,
        *,
        conservative: bool = False,
        non_negative: bool = False,
    ) -> np.ndarray:
        """Map angle/radius scalar samples back to Cartesian cell centres."""

        angular = np.asarray(values)
        if non_negative and (not np.all(np.isfinite(angular)) or np.any(angular < 0.0)):
            raise ValueError("non-negative angular density must be finite and non-negative")
        if angular.shape[-2:] != (self.angular_samples, self.radial_bins):
            raise ValueError("reduced azimuthal scalar array has the wrong shape")
        output = np.zeros(
            (*angular.shape[:-2], len(self.grid.y) * len(self.grid.x)),
            dtype=angular.dtype,
        )
        radial = self._axisym_bin_flat[self._axisym_valid_flat]
        angular_lower = self._cartesian_angular_lower
        angular_upper = self._cartesian_angular_upper
        upper_weight = self._cartesian_angular_upper_weight
        output[..., self._axisym_valid_flat] = (
            angular[..., angular_lower, radial] * (1.0 - upper_weight)
            + angular[..., angular_upper, radial] * upper_weight
        )
        if conservative:
            valid_areas = self._axisym_cell_area_flat[self._axisym_valid_flat]
            for radial_index, flat_indices in enumerate(
                self._axisym_flat_indices_by_bin
            ):
                selected = self._axisym_bin_flat[
                    self._axisym_valid_flat
                ] == radial_index
                actual = np.sum(
                    output[..., flat_indices]
                    * self._axisym_cell_area_flat[flat_indices],
                    axis=-1,
                )
                target = (
                    np.mean(angular[..., :, radial_index], axis=-1)
                    * self.radial_area_m2[radial_index]
                )
                if non_negative:
                    # Intensity and absorbed energy cannot use an additive
                    # signed correction, which can make dark cells negative.
                    area = float(np.sum(valid_areas[selected]))
                    factor = np.divide(
                        target, actual, out=np.zeros_like(target),
                        where=actual > np.finfo(float).tiny,
                    )
                    output[..., flat_indices] = np.where(
                        (actual > np.finfo(float).tiny)[..., None],
                        output[..., flat_indices] * factor[..., None],
                        (target / area)[..., None],
                    )
                    continue
                correction = (target - actual) / max(
                    float(np.sum(valid_areas[selected])),
                    np.finfo(float).tiny,
                )
                output[..., flat_indices] += correction[..., None]
        return output.reshape(
            (*angular.shape[:-2], len(self.grid.y), len(self.grid.x))
        )

    def _expand_azimuthal_field(self, field: np.ndarray) -> np.ndarray:
        """Evaluate retained radial Fourier modes on Cartesian x/y samples."""

        angular = np.asarray(field)
        if not self._is_reduced_field(angular):
            raise ValueError("reduced coherent field has the wrong shape")
        coefficients = np.fft.fft(angular, axis=-2) / self.angular_samples
        output = np.zeros(
            (*angular.shape[:-2], len(self.grid.y) * len(self.grid.x)),
            dtype=angular.dtype,
        )
        lower = self._axisym_interp_lower
        upper = self._axisym_interp_upper
        radial_weight = self._axisym_interp_upper_weight
        reconstructed = np.zeros(
            (*angular.shape[:-2], int(np.count_nonzero(self._axisym_valid_flat))),
            dtype=angular.dtype,
        )
        for angular_index in self._retained_angular_indices:
            order = int(self._angular_fft_orders[angular_index])
            radial_coefficient = coefficients[..., angular_index, :]
            interpolated = (
                radial_coefficient[..., lower] * (1.0 - radial_weight)
                + radial_coefficient[..., upper] * radial_weight
            )
            reconstructed += interpolated * np.exp(
                1j * order * self._cartesian_theta_valid
            )
        output[..., self._axisym_valid_flat] = reconstructed
        output = output.reshape(
            (*angular.shape[:-2], len(self.grid.y), len(self.grid.x))
        )
        reduced_energy = np.sum(
            np.abs(angular) ** 2
            * np.broadcast_to(
                self.angular_cell_area_m2,
                (self.angular_samples, self.radial_bins),
            ),
            axis=(-2, -1),
        )
        cartesian_energy = np.sum(
            np.abs(output) ** 2 * self.grid.cell_areas,
            axis=(-2, -1),
        )
        scale = np.sqrt(
            reduced_energy
            / np.maximum(cartesian_energy, np.finfo(float).tiny)
        )
        return output * scale[..., None, None]

    def _compress_azimuthal_field(self, field: np.ndarray) -> np.ndarray:
        polar = self._sample_cartesian_on_polar_grid(field)
        reduced = self._truncate_angular_samples(polar)
        full_energy = np.sum(
            np.abs(field) ** 2 * self.grid.cell_areas,
            axis=(-2, -1),
        )
        disk_energy = np.sum(
            np.abs(field) ** 2
            * self.grid.cell_areas
            * self._axisym_valid_xy,
            axis=(-2, -1),
        )
        polar_energy = np.sum(
            np.abs(polar) ** 2 * self.angular_cell_area_m2,
            axis=(-2, -1),
        )
        retained_polar_energy = np.sum(
            np.abs(reduced) ** 2 * self.angular_cell_area_m2,
            axis=(-2, -1),
        )
        # Keep the two independent losses separate before combining them:
        # ``disk/full`` measures Cartesian energy that lies outside the
        # inscribed radial domain, while ``retained/polar`` is an exact
        # Parseval measure of the discarded angular Fourier orders on the
        # collocation grid.  Using ``retained/full`` directly would fold the
        # polar interpolation/quadrature error into the physical validity
        # gate (about one percent even for a well-resolved centred Gaussian on
        # a coarse grid); using only ``retained/polar`` would miss energy in
        # the square-window corners entirely.
        disk_fraction = disk_energy / np.maximum(
            full_energy, np.finfo(float).tiny
        )
        angular_fraction = retained_polar_energy / np.maximum(
            polar_energy, np.finfo(float).tiny
        )
        projection_fraction = np.clip(disk_fraction, 0.0, 1.0) * np.clip(
            angular_fraction, 0.0, 1.0
        )
        active = full_energy > np.finfo(float).eps * float(
            np.max(full_energy)
        )
        if np.any(active):
            self.axisym_minimum_projection_energy_fraction = min(
                self.axisym_minimum_projection_energy_fraction,
                float(np.min(projection_fraction[active])),
            )
        self.axisym_total_projection_energy_fraction = min(
            self.axisym_total_projection_energy_fraction,
            float(
                np.sum(
                    full_energy * np.clip(projection_fraction, 0.0, 1.0)
                )
            )
            / max(float(np.sum(full_energy)), np.finfo(float).tiny),
        )
        reduced *= np.sqrt(
            full_energy
            / np.maximum(retained_polar_energy, np.finfo(float).tiny)
        )[..., None, None]
        radial_total = np.sum(
            np.abs(reduced) ** 2
            * np.broadcast_to(
                self.angular_cell_area_m2,
                (self.angular_samples, self.radial_bins),
            ),
            axis=(-2, -1),
        )
        outer_count = max(1, int(math.ceil(0.05 * self.radial_bins)))
        outer = np.sum(
            np.abs(reduced[..., -outer_count:]) ** 2
            * self.angular_cell_area_m2[..., -outer_count:],
            axis=(-2, -1),
        )
        self._update_axisymmetric_boundary_diagnostics(radial_total, outer)
        return reduced

    def _retained_azimuthal_residual(self, values: np.ndarray) -> float:
        projected = self._project_cartesian_scalar_to_angular(values)
        reconstructed = self._expand_angular_nearest(projected)
        residual = np.asarray(values) - reconstructed
        weights = self.grid.cell_areas * self._axisym_valid_xy
        numerator = float(np.sum(np.abs(residual) ** 2 * weights))
        denominator = float(np.sum(np.abs(values) ** 2 * weights))
        return math.sqrt(
            numerator / max(denominator, np.finfo(float).tiny)
        )

    def _radial_average(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.shape[-2:] != self.crystal_mask.shape:
            raise ValueError("axisymmetric projection requires the Cartesian optical-grid shape")
        flattened = array.reshape((-1, array.shape[-2] * array.shape[-1]))
        valid_values = flattened[:, self._axisym_valid_flat]
        valid_bins = self._axisym_bin_flat[self._axisym_valid_flat]
        valid_areas = self._axisym_cell_area_flat[self._axisym_valid_flat]
        output = np.empty((flattened.shape[0], self.radial_bins), dtype=array.dtype)
        for radial_index in range(self.radial_bins):
            selected = valid_bins == radial_index
            output[:, radial_index] = np.sum(
                valid_values[:, selected] * valid_areas[selected], axis=1
            ) / self.cartesian_annulus_area_m2[radial_index]
        return output.reshape((*array.shape[:-2], self.radial_bins))

    def _expand_radial_nearest(
        self,
        values: np.ndarray,
        *,
        conservative: bool = False,
    ) -> np.ndarray:
        radial = np.asarray(values)
        if radial.shape[-2:] == (1, self.radial_bins):
            radial = radial[..., 0, :]
        if radial.shape[-1] != self.radial_bins:
            raise ValueError("axisymmetric radial array has the wrong bin count")
        output = np.zeros(
            (*radial.shape[:-1], len(self.grid.y) * len(self.grid.x)),
            dtype=radial.dtype,
        )
        valid_bins = self._axisym_bin_flat[self._axisym_valid_flat]
        expanded = radial[..., valid_bins]
        if conservative:
            expanded = expanded * self._axisym_conservative_bin_scale[valid_bins]
        output[..., self._axisym_valid_flat] = expanded
        return output.reshape((*radial.shape[:-1], len(self.grid.y), len(self.grid.x)))

    def _expand_radial_field(self, field: np.ndarray) -> np.ndarray:
        if int(getattr(self, "maximum_azimuthal_order", 0)) > 0:
            return self._expand_azimuthal_field(field)
        radial = np.asarray(field)
        if radial.shape[-2:] != (1, self.radial_bins):
            raise ValueError("axisymmetric field has the wrong radial shape")
        radial = radial[..., 0, :]
        output = np.zeros(
            (*radial.shape[:-1], len(self.grid.y) * len(self.grid.x)),
            dtype=radial.dtype,
        )
        lower = radial[..., self._axisym_interp_lower]
        upper = radial[..., self._axisym_interp_upper]
        weight = self._axisym_interp_upper_weight
        output[..., self._axisym_valid_flat] = lower * (1.0 - weight) + upper * weight
        output = output.reshape((*radial.shape[:-1], len(self.grid.y), len(self.grid.x)))
        radial_energy = np.sum(np.abs(radial) ** 2 * self.radial_area_m2, axis=-1)
        cart_energy = np.sum(
            np.abs(output) ** 2 * self.grid.cell_areas,
            axis=(-2, -1),
        )
        scale = np.sqrt(
            radial_energy / np.maximum(cart_energy, np.finfo(float).tiny)
        )
        return output * scale[..., None, None]

    def _azimuthal_residual(
        self,
        values: np.ndarray,
        radial_average: np.ndarray,
    ) -> float:
        if self.maximum_azimuthal_order > 0:
            del radial_average
            return self._retained_azimuthal_residual(values)
        reconstructed = self._expand_radial_nearest(radial_average)
        residual = np.asarray(values) - reconstructed
        weights = self.grid.cell_areas * self._axisym_valid_xy
        numerator = float(np.sum(np.abs(residual) ** 2 * weights))
        denominator = float(np.sum(np.abs(values) ** 2 * weights))
        return math.sqrt(numerator / max(denominator, np.finfo(float).tiny))

    def _compress_cartesian_field(self, field: np.ndarray) -> np.ndarray:
        if self.maximum_azimuthal_order > 0:
            return self._compress_azimuthal_field(field)
        full_energy = np.sum(
            np.abs(field) ** 2 * self.grid.cell_areas,
            axis=(-2, -1),
        )
        original_shape = field.shape[:-2]
        flattened = np.asarray(field).reshape((-1, field.shape[-2] * field.shape[-1]))
        valid_values = flattened[:, self._axisym_valid_flat]
        valid_areas = self._axisym_cell_area_flat[self._axisym_valid_flat]
        lower = self._axisym_interp_lower
        upper = self._axisym_interp_upper
        upper_weight = self._axisym_interp_upper_weight
        lower_weight = self._axisym_projection_lower_weight
        rhs = np.zeros((flattened.shape[0], self.radial_bins), dtype=field.dtype)
        for radial_index in range(self.radial_bins):
            selected_lower = lower == radial_index
            if np.any(selected_lower):
                rhs[:, radial_index] += np.sum(
                    valid_values[:, selected_lower]
                    * (valid_areas[selected_lower] * lower_weight[selected_lower]),
                    axis=1,
                )
            selected_upper = (upper == radial_index) & (upper_weight > 0.0)
            if np.any(selected_upper):
                rhs[:, radial_index] += np.sum(
                    valid_values[:, selected_upper]
                    * (valid_areas[selected_upper] * upper_weight[selected_upper]),
                    axis=1,
                )
        diagonal = self._axisym_projection_cholesky_diagonal
        subdiagonal = self._axisym_projection_cholesky_subdiagonal
        forward = np.empty_like(rhs)
        forward[:, 0] = rhs[:, 0] / diagonal[0]
        for radial_index in range(1, self.radial_bins):
            forward[:, radial_index] = (
                rhs[:, radial_index]
                - subdiagonal[radial_index - 1] * forward[:, radial_index - 1]
            ) / diagonal[radial_index]
        radial_flat = np.empty_like(rhs)
        radial_flat[:, -1] = forward[:, -1] / diagonal[-1]
        for radial_index in range(self.radial_bins - 2, -1, -1):
            radial_flat[:, radial_index] = (
                forward[:, radial_index]
                - subdiagonal[radial_index] * radial_flat[:, radial_index + 1]
            ) / diagonal[radial_index]
        projected_values = (
            radial_flat[:, lower] * lower_weight
            + radial_flat[:, upper] * upper_weight
        )
        projected_energy = np.sum(
            np.abs(projected_values) ** 2 * valid_areas,
            axis=-1,
        ).reshape(original_shape)
        radial = radial_flat.reshape((*original_shape, self.radial_bins))
        projection_fraction = projected_energy / np.maximum(
            full_energy, np.finfo(float).tiny
        )
        active = full_energy > np.finfo(float).eps * float(
            np.max(full_energy)
        )
        if np.any(active):
            self.axisym_minimum_projection_energy_fraction = min(
                self.axisym_minimum_projection_energy_fraction,
                float(np.min(projection_fraction[active])),
            )
        total_projection_fraction = float(np.sum(projected_energy)) / max(
            float(np.sum(full_energy)), np.finfo(float).tiny
        )
        self.axisym_total_projection_energy_fraction = min(
            self.axisym_total_projection_energy_fraction,
            total_projection_fraction,
        )
        # The least-squares projection error is measured on the Cartesian
        # samples above, but the reduced propagator integrates energy with the
        # analytic annulus areas.  Those two discrete norms are not identical
        # on a finite grid.  Normalizing with projected_energy here injects or
        # removes energy as soon as the radial solver takes over (several
        # percent on coarse grids, even with zero gain/loss).  Match the norm
        # actually used by every radial operator; _expand_radial_field applies
        # the reciprocal Cartesian reconstruction normalization at the exit.
        radial_energy = np.sum(
            np.abs(radial) ** 2 * self.radial_area_m2,
            axis=-1,
        )
        scale = np.sqrt(
            full_energy / np.maximum(radial_energy, np.finfo(float).tiny)
        )
        radial *= scale[..., None]
        radial_total = np.sum(np.abs(radial) ** 2 * self.radial_area_m2, axis=-1)
        outer_count = max(1, int(math.ceil(0.05 * self.radial_bins)))
        outer = np.sum(
            np.abs(radial[..., -outer_count:]) ** 2
            * self.radial_area_m2[-outer_count:],
            axis=-1,
        )
        self._update_axisymmetric_boundary_diagnostics(radial_total, outer)
        return radial[..., None, :]

    def _enter_gain_transport(
        self,
        field: np.ndarray,
        population: np.ndarray,
        delta_n: np.ndarray,
        pass_number: int,
    ) -> np.ndarray:
        del pass_number
        projected_delta_n_source = np.asarray(delta_n, dtype=float)
        if self._local_parabolic_thermal_lens is not None:
            projected_delta_n_source = np.stack(
                [
                    self._local_parabolic_thermal_lens.fit_slice(
                        projected_delta_n_source[z_index], z_index
                    )
                    for z_index in range(len(self.grid.z))
                ],
                axis=0,
            )
        radial_population = self._project_cartesian_scalar_to_angular(
            population
        )
        radial_delta_n = (
            self._radial_average(projected_delta_n_source)
            if self.maximum_azimuthal_order == 0
            else self._project_cartesian_scalar_to_angular(
                projected_delta_n_source
            )
        )
        self.axisym_maximum_population_azimuthal_residual = max(
            self.axisym_maximum_population_azimuthal_residual,
            self._azimuthal_residual(population, radial_population),
        )
        self.axisym_maximum_delta_n_azimuthal_residual = max(
            self.axisym_maximum_delta_n_azimuthal_residual,
            self._azimuthal_residual(
                projected_delta_n_source, radial_delta_n
            ),
        )
        if self.maximum_azimuthal_order > 0:
            radial_population = self._make_angular_population_realizable(
                radial_population
            )
        self._axisym_population = radial_population.copy()
        self._axisym_population_initial = self._axisym_population.copy()
        radial_passive = self._project_cartesian_scalar_to_angular(
            self.passive_loss_energy_J_m3
        )
        self._axisym_passive_loss = radial_passive.copy()
        self._axisym_passive_loss_initial = self._axisym_passive_loss.copy()
        self._axisym_delta_n = radial_delta_n
        environment = self.levels.environment_copy()
        self._axisym_environment_original = environment

        def radial_environment(value: np.ndarray | float) -> np.ndarray | float:
            array = np.asarray(value, dtype=float)
            if array.ndim == 0:
                return float(array)
            if array.shape[-2:] != self.crystal_mask.shape:
                raise ValueError(
                    "axisymmetric spectroscopy environment does not match the optical grid"
                )
            return self._project_cartesian_scalar_to_angular(array)

        self.levels.set_environment(
            radial_environment(environment[0]),
            radial_environment(environment[1]),
        )
        if self._override_gain_m_inv is None:
            self._axisym_override_gain = None
            self._axisym_override_gain_initial = None
        else:
            radial_override = self._project_cartesian_scalar_to_angular(
                self._override_gain_m_inv
            )
            self._axisym_override_gain = radial_override.copy()
            self._axisym_override_gain_initial = self._axisym_override_gain.copy()
        self.axisym_passes_reduced += 1
        return self._compress_cartesian_field(field)

    def _map_angular_population_delta(
        self,
        population: np.ndarray,
        reduced_delta: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Map angular level transfers with a positivity-preserving closure.

        A truncated Fourier reconstruction can ring below zero where a donor
        level is locally scarce even though the reduced gain update is
        physically conservative.  First limit every Cartesian cell's
        donor/receiver vector without changing its level sum.  Then restore
        the requested level-integrated transfer inside each radial annulus by
        moving population between donor and receiver levels in cells that
        still contain donor inventory.  This preserves non-negativity, every
        cell's dopant inventory, the radial depletion profile, and the
        annulus-integrated level update while perturbing the retained angular
        pattern only as much as positivity requires.
        """

        current = np.asarray(population, dtype=float)
        delta = np.asarray(reduced_delta, dtype=float)
        level_count, nz, ny, nx = current.shape
        expected_shape = (
            level_count,
            nz,
            self.angular_samples,
            self.radial_bins,
        )
        if delta.shape != expected_shape:
            raise ValueError(
                "finite-azimuthal population delta has an incompatible shape"
            )
        expanded_delta = self._expand_angular_nearest(
            delta,
            conservative=True,
        )
        current_flat = current.reshape((level_count, nz, ny * nx))
        desired_flat = expanded_delta.reshape((level_count, nz, ny * nx))
        mapped_flat = np.zeros_like(current_flat)
        area_flat = self._axisym_cell_area_flat
        maximum_error = 0.0
        tiny = np.finfo(float).tiny

        for z_index in range(nz):
            for radial_index, flat_indices in enumerate(
                self._axisym_flat_indices_by_bin
            ):
                if not len(flat_indices):
                    continue
                cell_area = area_flat[flat_indices]
                local_current = current_flat[:, z_index, flat_indices]
                local_desired = desired_flat[
                    :, z_index, flat_indices
                ].copy()
                target = (
                    np.sum(delta[:, z_index, :, radial_index], axis=-1)
                    * self.radial_area_m2[radial_index]
                    / self.angular_samples
                ).astype(float, copy=False)
                inventory = float(
                    np.sum(local_current * cell_area[None, :])
                )
                target_transfer = float(np.sum(np.abs(target)))
                target_scale = max(target_transfer, inventory, tiny)
                target_balance = float(np.sum(target))
                target_tolerance = max(
                    1e-10 * target_transfer,
                    256.0 * np.finfo(float).eps * inventory,
                )
                if abs(target_balance) > target_tolerance:
                    raise ValueError(
                        "finite-azimuthal gain update does not conserve the "
                        "annular dopant inventory"
                    )
                if target_transfer > tiny:
                    target[int(np.argmax(np.abs(target)))] -= target_balance

                local_balance = np.sum(local_desired, axis=0)
                local_transfer = np.sum(np.abs(local_desired), axis=0)
                local_inventory = np.sum(local_current, axis=0)
                local_tolerance = np.maximum(
                    1e-10 * local_transfer,
                    256.0 * np.finfo(float).eps * local_inventory,
                )
                if np.any(np.abs(local_balance) > local_tolerance):
                    raise ValueError(
                        "finite-azimuthal Cartesian reconstruction does not "
                        "conserve local dopant inventory"
                    )
                dominant = np.argmax(np.abs(local_desired), axis=0)
                local_desired[
                    dominant,
                    np.arange(len(flat_indices)),
                ] -= local_balance

                requested_removal = np.maximum(-local_desired, 0.0)
                donor_ratio = np.full_like(requested_removal, np.inf)
                np.divide(
                    local_current,
                    requested_removal,
                    out=donor_ratio,
                    where=requested_removal > 0.0,
                )
                limiter = np.clip(np.min(donor_ratio, axis=0), 0.0, 1.0)
                local_mapped = local_desired * limiter[None, :]
                local_candidate = local_current + local_mapped
                actual = np.sum(
                    local_mapped * cell_area[None, :],
                    axis=1,
                )
                residual = target - actual
                residual_balance = float(np.sum(residual))
                residual_scale = max(
                    float(np.sum(np.abs(residual))),
                    target_scale,
                    tiny,
                )
                if abs(residual_balance) > 1e-10 * residual_scale:
                    raise ValueError(
                        "finite-azimuthal positivity limiter produced an "
                        "unbalanced level transfer"
                    )
                if np.any(residual):
                    residual[int(np.argmax(np.abs(residual)))] -= (
                        residual_balance
                    )
                residual_tolerance = max(
                    256.0 * np.finfo(float).eps * inventory,
                    1e-12 * target_transfer,
                    tiny,
                )

                while True:
                    residual_activity = float(np.sum(np.abs(residual)))
                    if residual_activity <= residual_tolerance:
                        break
                    # Pair by sign rather than applying the numerical floor to
                    # every level independently.  One donor residual can be
                    # balanced by several receiver residuals that are each
                    # below that floor; thresholding the individual entries
                    # falsely reported an unmatched donor/receiver pair.
                    donor_levels = np.flatnonzero(residual < 0.0)
                    receiver_levels = np.flatnonzero(residual > 0.0)
                    if not len(donor_levels) or not len(receiver_levels):
                        if residual_activity <= 1e-10 * max(
                            inventory,
                            target_transfer,
                            tiny,
                        ):
                            break
                        raise ValueError(
                            "finite-azimuthal positivity correction has a "
                            "materially unbalanced donor/receiver transfer"
                        )
                    donor = int(
                        donor_levels[np.argmax(-residual[donor_levels])]
                    )
                    receiver = int(
                        receiver_levels[
                            np.argmax(residual[receiver_levels])
                        ]
                    )
                    transfer = min(-residual[donor], residual[receiver])
                    remaining = transfer
                    intended_donor = local_desired[donor] < 0.0
                    for eligible in (
                        intended_donor,
                        np.ones(len(flat_indices), dtype=bool),
                    ):
                        if remaining <= residual_tolerance:
                            break
                        available_density = np.where(
                            eligible,
                            np.maximum(local_candidate[donor], 0.0),
                            0.0,
                        )
                        available = float(
                            np.sum(available_density * cell_area)
                        )
                        if available <= tiny:
                            continue
                        applied = min(remaining, available)
                        removal = available_density * (applied / available)
                        local_mapped[donor] -= removal
                        local_mapped[receiver] += removal
                        local_candidate[donor] -= removal
                        local_candidate[receiver] += removal
                        remaining -= applied
                    if remaining > max(
                        1e-10 * residual_scale,
                        256.0 * np.finfo(float).eps * inventory,
                    ):
                        raise ValueError(
                            "finite-azimuthal population transfer exceeds the "
                            "available Cartesian donor inventory in a radial "
                            "annulus; increase radial_bins or use full x-y"
                        )
                    residual[donor] += transfer
                    residual[receiver] -= transfer

                mapped_flat[:, z_index, flat_indices] = local_mapped
                final_actual = np.sum(
                    local_mapped * cell_area[None, :],
                    axis=1,
                )
                # This gate asks whether the annulus gained or lost ions, so
                # normalize local closure against its ion inventory.  A
                # transfer-relative metric is ill-conditioned when the
                # requested event is the subtraction residue of two O(1e24)
                # states; the global level-transfer check below remains
                # transfer-relative and catches a coherent mapping bias.
                closure_scale = max(
                    float(np.max(np.abs(target))),
                    inventory,
                    tiny,
                )
                maximum_error = max(
                    maximum_error,
                    float(np.max(np.abs(final_actual - target)))
                    / closure_scale,
                )

        mapped = mapped_flat.reshape(current.shape)
        candidate = current + mapped
        population_scale = max(float(np.max(np.abs(current))), 1.0)
        if float(np.min(candidate)) < -1e-10 * population_scale:
            raise ValueError(
                "finite-azimuthal positivity reconstruction produced an "
                "invalid level density"
            )
        candidate = np.maximum(candidate, 0.0)
        original_total = np.sum(current, axis=0)
        candidate_total = np.sum(candidate, axis=0)
        candidate *= np.divide(
            original_total,
            candidate_total,
            out=np.ones_like(original_total),
            where=candidate_total > tiny,
        )[None, ...]
        mapped = candidate - current

        # Recheck every annulus after the final roundoff clip/renormalization,
        # since that is the delta committed to the public Cartesian state.
        mapped_flat = mapped.reshape((level_count, nz, ny * nx))
        for z_index in range(nz):
            for radial_index, flat_indices in enumerate(
                self._axisym_flat_indices_by_bin
            ):
                if not len(flat_indices):
                    continue
                cell_area = area_flat[flat_indices]
                target = (
                    np.sum(
                        delta[:, z_index, :, radial_index],
                        axis=-1,
                    )
                    * self.radial_area_m2[radial_index]
                    / self.angular_samples
                )
                actual_annular = np.sum(
                    mapped_flat[:, z_index, flat_indices]
                    * cell_area[None, :],
                    axis=1,
                )
                inventory = float(
                    np.sum(
                        current_flat[:, z_index, flat_indices]
                        * cell_area[None, :]
                    )
                )
                maximum_error = max(
                    maximum_error,
                    float(np.max(np.abs(actual_annular - target)))
                    / max(
                        float(np.max(np.abs(target))),
                        inventory,
                        tiny,
                    ),
                )

        volume_weights = (
            self.grid.z_widths[:, None, None]
            * self.grid.cell_areas[None, :, :]
        )
        desired_transfer = 0.5 * float(
            np.sum(np.abs(expanded_delta) * volume_weights[None, ...])
        )
        redistributed_transfer = 0.5 * float(
            np.sum(
                np.abs(mapped - expanded_delta)
                * volume_weights[None, ...]
            )
        )
        if desired_transfer > tiny:
            self.axisym_maximum_population_positivity_redistribution_fraction = max(
                self.axisym_maximum_population_positivity_redistribution_fraction,
                redistributed_transfer / desired_transfer,
            )
        requested = np.sum(
            delta
            * self.angular_cell_area_m2[None, None, :, :]
            * self.grid.z_widths[None, :, None, None],
            axis=(1, 2, 3),
        )
        actual = np.sum(
            mapped
            * self.grid.cell_areas[None, None, :, :]
            * self.grid.z_widths[None, :, None, None],
            axis=(1, 2, 3),
        )
        error = float(np.max(np.abs(actual - requested))) / max(
            float(np.max(np.abs(requested))),
            float(np.max(np.abs(actual))),
            np.finfo(float).tiny,
        )
        return mapped, max(error, maximum_error)

    def _exit_azimuthal_gain_transport(
        self,
        field: np.ndarray,
        population: np.ndarray,
    ) -> np.ndarray:
        if self._axisym_population is None or self._axisym_population_initial is None:
            raise RuntimeError("azimuthal population state was not initialized")
        population_delta = self._axisym_population - self._axisym_population_initial
        expanded_population_delta, mapping_error = (
            self._map_angular_population_delta(population, population_delta)
        )
        population += expanded_population_delta
        self.axisym_population_mapping_relative_error = max(
            self.axisym_population_mapping_relative_error,
            mapping_error,
        )

        if (
            self._axisym_passive_loss is not None
            and self._axisym_passive_loss_initial is not None
        ):
            self.passive_loss_energy_J_m3 += self._expand_angular_nearest(
                self._axisym_passive_loss - self._axisym_passive_loss_initial,
                conservative=True,
                non_negative=True,
            )
        if (
            self._override_gain_m_inv is not None
            and self._axisym_override_gain is not None
            and self._axisym_override_gain_initial is not None
        ):
            self._override_gain_m_inv += self._expand_angular_nearest(
                self._axisym_override_gain
                - self._axisym_override_gain_initial
            )

        transient_override = getattr(
            self.transient_gain,
            "_override_gain_m_inv",
            None,
        )
        if transient_override is not None:
            transient_override = np.asarray(transient_override, dtype=float)
            if transient_override.shape[-2:] != (
                self.angular_samples,
                self.radial_bins,
            ):
                raise ValueError(
                    "azimuthal transient-gain override reservoir has the wrong shape"
                )
            self.transient_gain._override_gain_m_inv = (
                self._expand_angular_nearest(transient_override)
            )

        carrier_model = str(
            self.material.get("strong_field", {})
            .get("free_carrier", {})
            .get("model", "off")
        )
        if (
            self.physics.get("nonlinear_loss_model", "off")
            == "strong_field_time_domain"
            and carrier_model == "rate_equation"
        ):
            if (
                self._axisym_free_carrier_initial_cartesian_m3 is None
                or self._axisym_free_carrier_initial_radial_m3 is None
                or self._axisym_free_carrier_final_radial_m3 is None
            ):
                raise RuntimeError("azimuthal free-carrier state was not initialized")
            mapped_carrier, carrier_mapping_error = self._map_angular_carrier_sources(
                self._axisym_free_carrier_initial_cartesian_m3,
                self._axisym_free_carrier_generated_m3,
                self._axisym_free_carrier_survival,
            )
            self.final_free_carrier_density_m3[:] = mapped_carrier
            self.axisym_free_carrier_mapping_relative_error = max(
                self.axisym_free_carrier_mapping_relative_error,
                carrier_mapping_error,
            )

        restored = self._expand_azimuthal_field(field)
        if self._axisym_environment_original is not None:
            self.levels.set_environment(*self._axisym_environment_original)
            self._axisym_environment_original = None
        self._axisym_population = None
        self._axisym_population_initial = None
        self._axisym_passive_loss = None
        self._axisym_passive_loss_initial = None
        self._axisym_override_gain = None
        self._axisym_override_gain_initial = None
        self._axisym_delta_n = None
        self._axisym_active_free_carrier_initial_m3 = None
        self._axisym_free_carrier_initial_cartesian_m3 = None
        self._axisym_free_carrier_initial_radial_m3 = None
        self._axisym_free_carrier_final_radial_m3 = None
        return restored

    def _exit_gain_transport(
        self,
        field: np.ndarray,
        population: np.ndarray,
        delta_n: np.ndarray,
        pass_number: int,
    ) -> np.ndarray:
        del delta_n, pass_number
        if self.maximum_azimuthal_order > 0:
            return self._exit_azimuthal_gain_transport(field, population)
        if self._axisym_population is None or self._axisym_population_initial is None:
            raise RuntimeError("axisymmetric population state was not initialized")
        population_delta = self._axisym_population - self._axisym_population_initial
        expanded_population_delta, mapping_error = (
            self._map_radial_population_delta(population, population_delta)
        )
        self.axisym_population_mapping_relative_error = max(
            self.axisym_population_mapping_relative_error,
            mapping_error,
        )
        candidate = population + expanded_population_delta
        negative_tolerance = 1e-10 * max(float(np.max(np.abs(population))), 1.0)
        if float(np.min(candidate)) < -negative_tolerance:
            raise ValueError(
                "axisymmetric population reconstruction produced a negative level density"
            )
        population[:] = np.maximum(candidate, 0.0)

        if (
            self._axisym_passive_loss is not None
            and self._axisym_passive_loss_initial is not None
        ):
            passive_delta = (
                self._axisym_passive_loss - self._axisym_passive_loss_initial
            )
            self.passive_loss_energy_J_m3 += self._expand_radial_nearest(
                passive_delta,
                conservative=True,
            )
        if (
            self._override_gain_m_inv is not None
            and self._axisym_override_gain is not None
            and self._axisym_override_gain_initial is not None
        ):
            override_delta = (
                self._axisym_override_gain - self._axisym_override_gain_initial
            )
            self._override_gain_m_inv += self._expand_radial_nearest(override_delta)

        # The soft-window operator owns the optional synthetic gain reservoir.
        # Convert it back to the public Cartesian contract before the solver's
        # energy ledger reads it; otherwise an otherwise valid axisymmetric run
        # would expose a private (z, 1, r) array where (z, y, x) is required.
        transient_override = getattr(
            self.transient_gain,
            "_override_gain_m_inv",
            None,
        )
        if transient_override is not None:
            transient_override = np.asarray(transient_override, dtype=float)
            if transient_override.shape[-2:] != (1, self.radial_bins):
                raise ValueError(
                    "axisymmetric transient-gain override reservoir has the wrong shape"
                )
            self.transient_gain._override_gain_m_inv = (
                self._expand_radial_nearest(transient_override)
            )

        carrier_model = str(
            self.material.get("strong_field", {})
            .get("free_carrier", {})
            .get("model", "off")
        )
        if (
            self.physics.get("nonlinear_loss_model", "off")
            == "strong_field_time_domain"
            and carrier_model == "rate_equation"
        ):
            if (
                self._axisym_free_carrier_initial_cartesian_m3 is None
                or self._axisym_free_carrier_initial_radial_m3 is None
                or self._axisym_free_carrier_final_radial_m3 is None
            ):
                raise RuntimeError(
                    "axisymmetric free-carrier state was not initialized"
                )
            mapped_carrier, carrier_mapping_error = (
                self._map_radial_carrier_sources(
                    self._axisym_free_carrier_initial_cartesian_m3,
                    self._axisym_free_carrier_generated_m3,
                    self._axisym_free_carrier_survival,
                )
            )
            self.final_free_carrier_density_m3[:] = mapped_carrier
            self.axisym_free_carrier_mapping_relative_error = max(
                self.axisym_free_carrier_mapping_relative_error,
                carrier_mapping_error,
            )

        restored = self._expand_radial_field(field)
        if self._axisym_environment_original is not None:
            self.levels.set_environment(*self._axisym_environment_original)
            self._axisym_environment_original = None
        self._axisym_population = None
        self._axisym_population_initial = None
        self._axisym_passive_loss = None
        self._axisym_passive_loss_initial = None
        self._axisym_override_gain = None
        self._axisym_override_gain_initial = None
        self._axisym_delta_n = None
        self._axisym_active_free_carrier_initial_m3 = None
        self._axisym_free_carrier_initial_cartesian_m3 = None
        self._axisym_free_carrier_initial_radial_m3 = None
        self._axisym_free_carrier_final_radial_m3 = None
        return restored

    def _finish_crystal_cell(self, population: np.ndarray, z_index: int) -> None:
        independent = getattr(self, "_pending_gain_cell", None) is not None
        super()._finish_crystal_cell(population, z_index)
        if independent:
            self._enforce_cheap_material_m2_state()

    def _crystal_cell_substeps(
        self,
        field: np.ndarray,
        population: np.ndarray,
        delta_n_slice: np.ndarray,
        z_index: int,
        step_length_m: float,
    ) -> int:
        """Prepare one radial carrier boundary for every physical z cell.

        The parent coherent transport owns the public Cartesian carrier state
        and the volume-weighted substep accounting.  The axisymmetric solver
        projects only the pre-pulse boundary into radius.  Every Strang half
        and adaptive spatial fraction in this cell starts from that same
        boundary because each fraction represents a different local z volume,
        not a later time integration of the same carriers.
        """

        count = super()._crystal_cell_substeps(
            field,
            population,
            delta_n_slice,
            z_index,
            step_length_m,
        )
        initial = self._active_free_carrier_initial_m3
        if (
            self._axisym_free_carrier_initial_cartesian_m3 is None
            or self._axisym_free_carrier_initial_radial_m3 is None
            or self._axisym_free_carrier_final_radial_m3 is None
        ):
            raise RuntimeError("axisymmetric carrier work arrays were not initialized")
        if initial is None:
            cartesian_initial = np.zeros(self.crystal_mask.shape, dtype=float)
            radial_initial = np.zeros(
                self.radial_bins
                if self.maximum_azimuthal_order == 0
                else (self.angular_samples, self.radial_bins),
                dtype=float,
            )
            self._axisym_active_free_carrier_initial_m3 = None
        else:
            cartesian_initial = np.asarray(initial, dtype=float)
            radial_initial = (
                self._radial_average(initial)
                if self.maximum_azimuthal_order == 0
                # Density is a positive scalar sampled on the angular grid,
                # not a field amplitude in the truncated optical Fourier basis.
                else self._sample_cartesian_on_polar_grid(initial)
            )
            self.axisym_maximum_free_carrier_azimuthal_residual = max(
                self.axisym_maximum_free_carrier_azimuthal_residual,
                self._azimuthal_residual(initial, radial_initial),
            )
            self._axisym_active_free_carrier_initial_m3 = radial_initial
        self._axisym_free_carrier_initial_cartesian_m3[z_index] = (
            cartesian_initial
        )
        self._axisym_free_carrier_initial_radial_m3[z_index] = radial_initial
        self._axisym_free_carrier_final_radial_m3[z_index].fill(0.0)
        self._axisym_free_carrier_generated_m3[z_index].fill(0.0)
        self._axisym_free_carrier_survival[z_index] = 0.0
        return count

    def _map_angular_carrier_sources(
        self,
        initial_cartesian_m3: np.ndarray,
        generated_angular_m3: np.ndarray,
        survival: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Reconstruct N_end = exp(-T/tau) N_start + surviving births.

        This is the exact source/survival decomposition of dN/dt = G - N/tau
        for the configured constant lifetime. The optical rate equation samples
        N_start by positive bilinear interpolation; that sampling is not an
        inventory transfer. Only the newly generated carriers are transferred
        back, with their count conserved in every radial annulus. Recombination
        acts on the original Cartesian inventory, preserving its spatial shape.
        No signed delta correction or density clipping is involved. Different
        z/Strang fractions contribute volume-weighted sources and survival,
        never successive applications of the same temporal decay.
        """
        initial = np.asarray(initial_cartesian_m3, dtype=float)
        generated = np.asarray(generated_angular_m3, dtype=float)
        survival = np.asarray(survival, dtype=float)
        if initial.shape != self.grid.shape or survival.shape != (len(self.grid.z),):
            raise ValueError("azimuthal carrier source reconstruction has wrong shape")
        if (np.any(~np.isfinite(initial)) or np.any(initial < 0.0)
                or np.any(~np.isfinite(survival)) or np.any(survival < 0.0)
                or np.any(survival > 1.0 + 64.0 * np.finfo(float).eps)):
            raise ValueError("azimuthal carrier source reconstruction has invalid state")
        births = self._expand_angular_nearest(
            generated, conservative=True, non_negative=True,
        )
        mapped = initial * survival[:, None, None] + births
        requested = np.sum(generated, axis=-2) * self.angular_cell_area_m2
        flat_births = births.reshape((len(self.grid.z), -1))
        actual = np.stack([
            np.sum(flat_births[:, indices] * self._axisym_cell_area_flat[indices], axis=-1)
            for indices in self._axisym_flat_indices_by_bin
        ], axis=-1)
        error = float(np.max(np.abs(actual - requested) / np.maximum(
            np.maximum(np.abs(actual), np.abs(requested)), np.finfo(float).tiny,
        )))
        if np.any(~np.isfinite(mapped)) or np.any(mapped < 0.0):
            raise ValueError("azimuthal carrier source reconstruction produced invalid density")
        return mapped, error

    def _map_radial_carrier_sources(
        self,
        initial_cartesian_m3: np.ndarray,
        generated_radial_m3: np.ndarray,
        survival_fraction: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Preserve old-carrier survival and conservatively map radial births.

        The sampled Cartesian inventory and analytic radial annulus area are
        different discrete measures. Recombination acts on the original local
        density with its physical survival probability; only newly generated
        carriers require conversion from the radial finite-volume measure.
        """

        initial = np.asarray(initial_cartesian_m3, dtype=float)
        generated = np.asarray(generated_radial_m3, dtype=float)
        survival = np.asarray(survival_fraction, dtype=float)
        if initial.shape != self.grid.shape:
            raise ValueError("axisymmetric Cartesian carrier state has the wrong shape")
        expected_radial_shape = (len(self.grid.z), self.radial_bins)
        if (
            generated.shape != expected_radial_shape
            or survival.shape != (len(self.grid.z),)
        ):
            raise ValueError("axisymmetric radial carrier source has the wrong shape")
        if (
            np.any(~np.isfinite(initial)) or np.any(initial < 0.0)
            or np.any(~np.isfinite(generated)) or np.any(generated < 0.0)
            or np.any(~np.isfinite(survival)) or np.any(survival < 0.0)
            or np.any(survival > 1.0 + 64.0 * np.finfo(float).eps)
        ):
            raise ValueError("axisymmetric radial carrier source has invalid state")
        births = self._expand_radial_nearest(generated, conservative=True)
        mapped = initial * survival[:, None, None] + births
        requested = generated * self.radial_area_m2[None, :]
        flat_births = births.reshape((len(self.grid.z), -1))
        actual = np.stack([
            np.sum(
                flat_births[:, indices] * self._axisym_cell_area_flat[indices],
                axis=-1,
            )
            for indices in self._axisym_flat_indices_by_bin
        ], axis=-1)
        error = float(np.max(np.abs(actual - requested) / np.maximum(
            np.maximum(np.abs(actual), np.abs(requested)), np.finfo(float).tiny,
        )))
        if np.any(~np.isfinite(mapped)) or np.any(mapped < 0.0):
            raise ValueError(
                "axisymmetric carrier reconstruction produced an invalid density"
            )
        return mapped, error

    def _map_radial_population_delta(
        self,
        population: np.ndarray,
        radial_delta: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Map radial level transfers without erasing azimuthal populations.

        A direct additive annulus reconstruction can make a locally scarce
        donor level negative even when the annulus-integrated update is
        physical.  Here every decreasing level is removed in proportion to
        its existing local inventory; the locally removed population is then
        distributed among increasing levels in the radial solver's receiver
        ratio.  This preserves non-negativity, every cell's dopant inventory,
        azimuthal population variation, and each annulus-integrated level
        transfer.
        """

        current = np.asarray(population, dtype=float)
        delta = np.asarray(radial_delta, dtype=float)
        if delta.shape[-2:] != (1, self.radial_bins):
            raise ValueError("axisymmetric radial population delta has the wrong shape")
        level_count, nz, ny, nx = current.shape
        if delta.shape[:2] != (level_count, nz):
            raise ValueError("axisymmetric radial population delta has incompatible levels/z")
        current_flat = current.reshape((level_count, nz, ny * nx))
        mapped_flat = np.zeros_like(current_flat)
        area_flat = self._axisym_cell_area_flat
        maximum_error = 0.0
        for z_index in range(nz):
            for radial_index, flat_indices in enumerate(
                self._axisym_flat_indices_by_bin
            ):
                desired = (
                    delta[:, z_index, 0, radial_index]
                    * self.radial_area_m2[radial_index]
                ).astype(float, copy=True)
                scale = max(
                    float(np.sum(np.abs(desired))),
                    float(
                        np.sum(
                            current_flat[:, z_index, flat_indices]
                            * area_flat[flat_indices][None, :]
                        )
                    ),
                    np.finfo(float).tiny,
                )
                # The radial level integrator is conservative to floating-point
                # precision. Remove its tiny roundoff balance before deciding
                # donor/receiver signs.
                balance = float(np.sum(desired))
                if abs(balance) > 1e-10 * scale:
                    raise ValueError(
                        "axisymmetric radial gain update does not conserve local "
                        "dopant inventory"
                    )
                desired[int(np.argmax(np.abs(desired)))] -= balance
                threshold = 64.0 * np.finfo(float).eps * scale
                if float(np.max(np.abs(desired))) <= threshold:
                    continue
                # One resolved donor can feed several receivers individually
                # below the numerical floor; keep every sign-balanced transfer.
                donor_levels = np.flatnonzero(desired < 0.0)
                receiver_levels = np.flatnonzero(desired > 0.0)
                if not len(donor_levels) or not len(receiver_levels):
                    raise ValueError(
                        "axisymmetric radial population update has unmatched "
                        "donor/receiver levels"
                    )
                cell_area = area_flat[flat_indices]
                removed_density = np.zeros(len(flat_indices), dtype=float)
                for level_index in donor_levels:
                    local_population = current_flat[
                        level_index,
                        z_index,
                        flat_indices,
                    ]
                    available = float(np.sum(local_population * cell_area))
                    requested = float(-desired[level_index])
                    fraction = requested / max(available, np.finfo(float).tiny)
                    if fraction > 1.0 + 1e-10:
                        raise ValueError(
                            "axisymmetric radial population transfer exceeds the "
                            "available Cartesian donor inventory; increase radial_bins"
                        )
                    fraction = min(max(fraction, 0.0), 1.0)
                    local_removal = local_population * fraction
                    mapped_flat[
                        level_index,
                        z_index,
                        flat_indices,
                    ] -= local_removal
                    removed_density += local_removal
                receiver_total = float(np.sum(desired[receiver_levels]))
                removed_total = float(np.sum(removed_density * cell_area))
                if removed_total <= np.finfo(float).tiny:
                    raise ValueError(
                        "axisymmetric radial population transfer has zero donor inventory"
                    )
                for level_index in receiver_levels:
                    mapped_flat[
                        level_index,
                        z_index,
                        flat_indices,
                    ] += removed_density * (
                        float(desired[level_index]) / receiver_total
                    )
                actual = np.sum(
                    mapped_flat[:, z_index, flat_indices]
                    * cell_area[None, :],
                    axis=1,
                )
                maximum_error = max(
                    maximum_error,
                    float(np.max(np.abs(actual - desired)))
                    / max(
                        float(np.max(np.abs(desired))),
                        np.finfo(float).tiny,
                    ),
                )
        return mapped_flat.reshape(current.shape), maximum_error

    def _integral(self, field: np.ndarray) -> float:
        if self._is_reduced_field(field):
            return float(
                np.sum(
                    np.abs(field) ** 2
                    * self.angular_cell_area_m2
                )
            )
        return super()._integral(field)

    def _signal_volume_density(self, field: np.ndarray) -> np.ndarray:
        if not self._is_reduced_field(field):
            return super()._signal_volume_density(field)
        reduced = np.sum(np.abs(field) ** 2, axis=(0, 1))
        if self.maximum_azimuthal_order == 0:
            return self._expand_radial_nearest(reduced, conservative=True)
        return self._expand_angular_nearest(reduced, conservative=True, non_negative=True)

    def _apply_crystal_aperture(self, field: np.ndarray) -> np.ndarray:
        if self._is_reduced_field(field):
            return field
        return super()._apply_crystal_aperture(field)

    def _radial_cn_coefficients(
        self,
        distance_m: float,
        refractive_index: np.ndarray,
        azimuthal_order: int = 0,
    ) -> tuple[np.ndarray, ...]:
        state_token, indices, m2_x, m2_y = (
            self._diffraction_spectral_parameters(refractive_index)
        )
        m2 = 0.5 * (m2_x + m2_y)
        order = abs(int(azimuthal_order))
        key = (float(state_token), float(distance_m), float(order))
        cached = self._radial_cn_cache.get(key)
        if cached is not None:
            self._radial_cn_cache.move_to_end(key)
            self._radial_cn_cache_hits += 1
            return cached
        self._radial_cn_cache_misses += 1
        complex_bytes = np.dtype(np.complex128).itemsize
        estimated_bytes = int(
            len(indices)
            * (1 + 2 * (self.radial_bins - 1) + self.radial_bins)
            * complex_bytes
        )
        cacheable = bool(
            estimated_bytes <= self.radial_coefficient_cache_max_bytes
        )
        while self._radial_cn_cache and (
            not cacheable
            or self._radial_cn_cache_bytes + estimated_bytes
            > self.radial_coefficient_cache_max_bytes
        ):
            _evicted_key, evicted = self._radial_cn_cache.popitem(
                last=False
            )
            self._radial_cn_cache_bytes -= int(
                sum(array.nbytes for array in evicted)
            )
            self._radial_cn_cache_evictions += 1
        k = 2.0 * math.pi * indices / (self.spectrum.coordinate * m2)
        alpha = 1j * float(distance_m) / (4.0 * k)
        laplacian_diagonal = self._radial_laplacian_diagonal_by_order.get(
            order
        )
        if laplacian_diagonal is None:
            laplacian_diagonal = (
                self._radial_laplacian_diagonal
                - float(order**2)
                / np.maximum(
                    self.radial_centers_m**2,
                    np.finfo(float).tiny,
                )
            )
            self._radial_laplacian_diagonal_by_order[order] = (
                laplacian_diagonal
            )
        lower = -alpha[:, None] * self._radial_laplacian_lower[None, 1:]
        diagonal = 1.0 - alpha[:, None] * laplacian_diagonal[None, :]
        upper = -alpha[:, None] * self._radial_laplacian_upper[None, :-1]
        cprime = np.empty_like(upper)
        inverse_denominator = np.empty_like(diagonal)
        inverse_denominator[:, 0] = 1.0 / diagonal[:, 0]
        cprime[:, 0] = upper[:, 0] * inverse_denominator[:, 0]
        for radial_index in range(1, self.radial_bins):
            denominator = diagonal[:, radial_index] - lower[:, radial_index - 1] * cprime[:, radial_index - 1]
            inverse_denominator[:, radial_index] = 1.0 / denominator
            if radial_index < self.radial_bins - 1:
                cprime[:, radial_index] = upper[:, radial_index] * inverse_denominator[:, radial_index]
        cached = (alpha, lower, cprime, inverse_denominator)
        if cacheable:
            self._radial_cn_cache[key] = cached
            self._radial_cn_cache_bytes += int(
                sum(array.nbytes for array in cached)
            )
        else:
            self._radial_cn_cache_oversize_bypasses += 1
        return cached

    def _radial_diffraction_scratch_array(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype[Any] | type,
    ) -> np.ndarray:
        array = self._radial_diffraction_scratch.get(name)
        expected_dtype = np.dtype(dtype)
        if (
            array is None
            or array.shape != shape
            or array.dtype != expected_dtype
        ):
            array = np.empty(shape, dtype=expected_dtype)
            self._radial_diffraction_scratch[name] = array
        return array

    def _diffract_radial_mode_values(
        self,
        values: np.ndarray,
        distance_m: float,
        refractive_index: float | np.ndarray,
        azimuthal_order: int,
        frequency_slice: slice | None = None,
    ) -> np.ndarray:
        """Advance one Fourier order with its radial Bessel Laplacian."""

        order = abs(int(azimuthal_order))
        alpha, lower, cprime, inverse_denominator = (
            self._radial_cn_coefficients(
                distance_m,
                np.broadcast_to(
                    np.asarray(refractive_index, dtype=float),
                    self.spectrum.coordinate.shape,
                ),
                order,
            )
        )
        if frequency_slice is not None:
            alpha, lower, cprime, inverse_denominator = (
                a[frequency_slice] for a in (alpha, lower, cprime, inverse_denominator)
            )
        original_shape = values.shape
        if values.ndim != 3 or values.shape[1] != len(alpha):
            raise ValueError(
                "azimuthal coherent mode must have shape "
                "(polarization, frequency, radius)"
            )
        laplacian_diagonal = self._radial_laplacian_diagonal_by_order[order]
        work_dtype = np.result_type(values.dtype, alpha.dtype)
        forward = self._radial_diffraction_scratch_array(
            "forward", original_shape, work_dtype
        )
        propagated = np.empty(original_shape, dtype=work_dtype)
        self.radial_diffraction_kernel_calls += 1
        if self.radial_diffraction_kernel_selected == "numba":
            before = self._radial_diffraction_scratch_array(
                "numba_input_energy", original_shape[:2], float
            )
            correction = self._radial_diffraction_scratch_array(
                "numba_norm_correction", original_shape[:2], float
            )
            cold_start = not radial_numba_kernel_compiled()
            run_radial_cn_numba(
                values=values,
                alpha=alpha,
                lower=lower,
                cprime=cprime,
                inverse_denominator=inverse_denominator,
                laplacian_lower=self._radial_laplacian_lower,
                laplacian_diagonal=laplacian_diagonal,
                laplacian_upper=self._radial_laplacian_upper,
                radial_area=self.radial_area_m2,
                forward=forward,
                output=propagated,
                input_energy=before,
                norm_correction=correction,
            )
            if cold_start:
                self.radial_diffraction_numba_cold_starts += 1
        else:
            laplacian = self._radial_diffraction_scratch_array(
                "laplacian_or_work", original_shape, work_dtype
            )
            rhs = self._radial_diffraction_scratch_array(
                "rhs", original_shape, work_dtype
            )
            np.multiply(
                values,
                laplacian_diagonal[None, None, :],
                out=laplacian,
            )
            np.multiply(
                self._radial_laplacian_lower[None, None, 1:],
                values[:, :, :-1],
                out=forward[:, :, 1:],
            )
            laplacian[:, :, 1:] += forward[:, :, 1:]
            np.multiply(
                self._radial_laplacian_upper[None, None, :-1],
                values[:, :, 1:],
                out=forward[:, :, :-1],
            )
            laplacian[:, :, :-1] += forward[:, :, :-1]
            np.multiply(alpha[None, :, None], laplacian, out=rhs)
            np.add(values, rhs, out=rhs)
            np.multiply(
                rhs[:, :, 0],
                inverse_denominator[None, :, 0],
                out=forward[:, :, 0],
            )
            for radial_index in range(1, self.radial_bins):
                np.multiply(
                    lower[None, :, radial_index - 1],
                    forward[:, :, radial_index - 1],
                    out=laplacian[:, :, radial_index],
                )
                np.subtract(
                    rhs[:, :, radial_index],
                    laplacian[:, :, radial_index],
                    out=forward[:, :, radial_index],
                )
                forward[:, :, radial_index] *= inverse_denominator[
                    None, :, radial_index
                ]
            propagated[:, :, -1] = forward[:, :, -1]
            for radial_index in range(self.radial_bins - 2, -1, -1):
                np.multiply(
                    cprime[None, :, radial_index],
                    propagated[:, :, radial_index + 1],
                    out=laplacian[:, :, radial_index],
                )
                np.subtract(
                    forward[:, :, radial_index],
                    laplacian[:, :, radial_index],
                    out=propagated[:, :, radial_index],
                )
            energy_work = self._radial_diffraction_scratch_array(
                "energy", original_shape, float
            )
            np.absolute(values, out=energy_work)
            np.square(energy_work, out=energy_work)
            energy_work *= self.radial_area_m2[None, None, :]
            before = np.sum(energy_work, axis=-1)
            np.absolute(propagated, out=energy_work)
            np.square(energy_work, out=energy_work)
            energy_work *= self.radial_area_m2[None, None, :]
            after = np.sum(energy_work, axis=-1)
            scale = np.sqrt(
                before / np.maximum(after, np.finfo(float).tiny)
            )
            correction = np.abs(scale - 1.0)
            propagated *= scale[:, :, None]
        active = before > np.finfo(float).eps * max(
            float(np.max(before)), np.finfo(float).tiny
        )
        if np.any(active):
            self.axisym_maximum_diffraction_norm_correction = max(
                self.axisym_maximum_diffraction_norm_correction,
                float(np.max(correction[active])),
            )
        return propagated

    @profiled_operation("signal.axisymmetric_diffraction")
    def _diffract(
        self,
        field: np.ndarray,
        distance_m: float,
        refractive_index: float | np.ndarray,
    ) -> np.ndarray:
        plan = self._tail_frequency_plan
        if plan is None or not self._is_reduced_field(field):
            return self._diffract_selected(field, distance_m, refractive_index)
        lo, hi = plan["central_indices"]
        if lo == 0 and hi == field.shape[1]-1:
            return self._diffract_selected(field, distance_m, refractive_index)
        selected = slice(lo, hi+1)
        output = field.copy()
        output[:, selected] = self._diffract_selected(
            field[:, selected], distance_m, refractive_index, frequency_slice=selected)
        # Diagnose the complete field, including weak channels held unchanged.
        area = self.angular_cell_area_m2
        n = max(1, int(math.ceil(.05*self.radial_bins)))
        self._update_axisymmetric_boundary_diagnostics(
            np.sum(abs(output)**2*area, axis=(-2,-1)),
            np.sum(abs(output[..., -n:])**2*area[..., -n:], axis=(-2,-1)))
        return output

    def _diffract_selected(
        self, field: np.ndarray, distance_m: float,
        refractive_index: float | np.ndarray, *, frequency_slice: slice | None = None,
    ) -> np.ndarray:
        if self.maximum_azimuthal_order > 0 and self._is_reduced_field(field):
            coefficients = np.fft.fft(field, axis=-2) / self.angular_samples
            propagated = np.zeros_like(
                coefficients,
                dtype=np.result_type(coefficients.dtype, np.complex128),
            )
            for angular_index in self._retained_angular_indices:
                order = int(self._angular_fft_orders[angular_index])
                propagated[..., angular_index, :] = (
                    self._diffract_radial_mode_values(
                        coefficients[..., angular_index, :],
                        distance_m,
                        refractive_index,
                        order,
                        frequency_slice=frequency_slice,
                    )
                )
            output = np.fft.ifft(
                propagated * self.angular_samples,
                axis=-2,
            )
            radial_total = np.sum(
                np.abs(output) ** 2 * self.angular_cell_area_m2,
                axis=(-2, -1),
            )
            outer_count = max(1, int(math.ceil(0.05 * self.radial_bins)))
            outer = np.sum(
                np.abs(output[..., -outer_count:]) ** 2
                * self.angular_cell_area_m2[..., -outer_count:],
                axis=(-2, -1),
            )
            if frequency_slice is None:
                self._update_axisymmetric_boundary_diagnostics(radial_total, outer)
            return output
        if field.shape[-2:] != (1, self.radial_bins):
            return super()._diffract(field, distance_m, refractive_index)
        alpha, lower, cprime, inverse_denominator = self._radial_cn_coefficients(
            distance_m,
            np.broadcast_to(
                np.asarray(refractive_index, dtype=float),
                self.spectrum.coordinate.shape,
            ),
        )
        if frequency_slice is not None:
            alpha, lower, cprime, inverse_denominator = (
                a[frequency_slice] for a in (alpha, lower, cprime, inverse_denominator)
            )
        values = field[..., 0, :]
        original_shape = values.shape
        if values.ndim != 3 or values.shape[1] != len(alpha):
            raise ValueError(
                "axisymmetric coherent field must have shape "
                "(polarization, frequency, radius)"
            )
        # Match the legacy NumPy expression promotion. Normal coherent fields
        # are complex128, but a future complex64 caller must not silently lose
        # precision when combined with complex128 propagation coefficients.
        work_dtype = np.result_type(values.dtype, alpha.dtype)
        forward = self._radial_diffraction_scratch_array(
            "forward", original_shape, work_dtype
        )
        propagated = np.empty(original_shape, dtype=work_dtype)
        self.radial_diffraction_kernel_calls += 1
        if self.radial_diffraction_kernel_selected == "numba":
            before = self._radial_diffraction_scratch_array(
                "numba_input_energy", original_shape[:2], float
            )
            correction = self._radial_diffraction_scratch_array(
                "numba_norm_correction", original_shape[:2], float
            )
            cold_start = not radial_numba_kernel_compiled()
            run_radial_cn_numba(
                values=values,
                alpha=alpha,
                lower=lower,
                cprime=cprime,
                inverse_denominator=inverse_denominator,
                laplacian_lower=self._radial_laplacian_lower,
                laplacian_diagonal=self._radial_laplacian_diagonal,
                laplacian_upper=self._radial_laplacian_upper,
                radial_area=self.radial_area_m2,
                forward=forward,
                output=propagated,
                input_energy=before,
                norm_correction=correction,
            )
            if cold_start:
                self.radial_diffraction_numba_cold_starts += 1
        else:
            laplacian = self._radial_diffraction_scratch_array(
                "laplacian_or_work", original_shape, work_dtype
            )
            rhs = self._radial_diffraction_scratch_array(
                "rhs", original_shape, work_dtype
            )
            np.multiply(
                values,
                self._radial_laplacian_diagonal[None, None, :],
                out=laplacian,
            )
            np.multiply(
                self._radial_laplacian_lower[None, None, 1:],
                values[:, :, :-1],
                out=forward[:, :, 1:],
            )
            laplacian[:, :, 1:] += forward[:, :, 1:]
            np.multiply(
                self._radial_laplacian_upper[None, None, :-1],
                values[:, :, 1:],
                out=forward[:, :, :-1],
            )
            laplacian[:, :, :-1] += forward[:, :, :-1]
            np.multiply(alpha[None, :, None], laplacian, out=rhs)
            np.add(values, rhs, out=rhs)
            np.multiply(
                rhs[:, :, 0],
                inverse_denominator[None, :, 0],
                out=forward[:, :, 0],
            )
            for radial_index in range(1, self.radial_bins):
                np.multiply(
                    lower[None, :, radial_index - 1],
                    forward[:, :, radial_index - 1],
                    out=laplacian[:, :, radial_index],
                )
                np.subtract(
                    rhs[:, :, radial_index],
                    laplacian[:, :, radial_index],
                    out=forward[:, :, radial_index],
                )
                forward[:, :, radial_index] *= inverse_denominator[
                    None, :, radial_index
                ]
            propagated[:, :, -1] = forward[:, :, -1]
            for radial_index in range(self.radial_bins - 2, -1, -1):
                np.multiply(
                    cprime[None, :, radial_index],
                    propagated[:, :, radial_index + 1],
                    out=laplacian[:, :, radial_index],
                )
                np.subtract(
                    forward[:, :, radial_index],
                    laplacian[:, :, radial_index],
                    out=propagated[:, :, radial_index],
                )
            energy_work = self._radial_diffraction_scratch_array(
                "energy", original_shape, float
            )
            np.absolute(values, out=energy_work)
            np.square(energy_work, out=energy_work)
            energy_work *= self.radial_area_m2[None, None, :]
            before = np.sum(energy_work, axis=-1)
            np.absolute(propagated, out=energy_work)
            np.square(energy_work, out=energy_work)
            energy_work *= self.radial_area_m2[None, None, :]
            after = np.sum(energy_work, axis=-1)
            scale = np.sqrt(
                before / np.maximum(after, np.finfo(float).tiny)
            )
            correction = np.abs(scale - 1.0)
            propagated *= scale[:, :, None]
        active = before > np.finfo(float).eps * float(np.max(before))
        if np.any(active):
            self.axisym_maximum_diffraction_norm_correction = max(
                self.axisym_maximum_diffraction_norm_correction,
                float(np.max(correction[active])),
            )
        output = propagated.reshape(original_shape)[..., None, :]
        radial_total = np.sum(np.abs(output[..., 0, :]) ** 2 * self.radial_area_m2, axis=-1)
        outer_count = max(1, int(math.ceil(0.05 * self.radial_bins)))
        outer = np.sum(
            np.abs(output[..., 0, -outer_count:]) ** 2
            * self.radial_area_m2[-outer_count:],
            axis=-1,
        )
        if frequency_slice is None:
            self._update_axisymmetric_boundary_diagnostics(radial_total, outer)
        return output

    def _apply_thermal_phase(
        self,
        field: np.ndarray,
        delta_n_slice: np.ndarray,
        step_length_m: float,
        z_index: int,
    ) -> np.ndarray:
        if not self._is_reduced_field(field):
            return super()._apply_thermal_phase(
                field, delta_n_slice, step_length_m, z_index
            )
        if self._axisym_delta_n is None:
            raise RuntimeError("axisymmetric thermal state was not initialized")
        radial_delta_n = self._axisym_delta_n[z_index]
        if not np.any(radial_delta_n):
            self._thermal_phase_stats["zero_operator_skips"] = int(
                self._thermal_phase_stats.get("zero_operator_skips", 0)
            ) + 1
            return field
        reduced_delta_n = (
            radial_delta_n[None, :]
            if self.maximum_azimuthal_order == 0
            else radial_delta_n
        )
        wavelengths = np.asarray(self.spectrum.coordinate, dtype=float)
        field *= np.exp(
            1j
            * 2.0
            * math.pi
            * float(step_length_m)
            * reduced_delta_n[None, None, :, :]
            / wavelengths[None, :, None, None]
        )
        if self.maximum_azimuthal_order > 0:
            field = self._truncate_angular_samples(
                field,
                track_field_omission=True,
            )
        self._thermal_phase_stats["reduced_azimuthal_applications"] = int(
            self._thermal_phase_stats.get(
                "reduced_azimuthal_applications", 0
            )
        ) + 1
        return field

    def _apply_material_dispersion(
        self,
        field: np.ndarray,
        step_length_m: float,
    ) -> np.ndarray:
        if not self._is_reduced_field(field):
            return super()._apply_material_dispersion(field, step_length_m)
        factor = self._material_dispersion_factor(step_length_m)
        if factor is not None:
            field[0] *= factor[:, None, None]
        return field

    def _apply_azimuthal_nonlinear_step(
        self,
        field: np.ndarray,
        step_length_m: float,
        z_index: int | None,
        *,
        physical_slice_length_m: float,
    ) -> np.ndarray:
        use_kerr = bool(
            self.physics.get("nonlinear_phase_model", "off")
            == "kerr_time_domain"
            and float(self.material.get("n2_m2_W", 0.0)) != 0.0
        )
        use_loss = bool(
            self.physics.get("nonlinear_loss_model", "off")
            == "strong_field_time_domain"
        )
        if not use_kerr and not use_loss:
            return field
        spectral_modes = self._coherent_spectral_modes(field)
        multiple_modes = spectral_modes.shape[0] > 1
        if multiple_modes and use_loss:
            raise ValueError("finite-azimuthal incoherent HG currently supports Kerr without strong-field loss")
        temporal_modes = self.pulse.spectrum_to_time(spectral_modes, axis=1)
        temporal_field = temporal_modes if multiple_modes else temporal_modes[0]
        k0 = 2.0 * math.pi / float(self.cfg["wavelength_m"])
        nonlinear_intensity_path_W_m: np.ndarray | None = None
        incident_peak_power_W: float | None = None
        # Without nonlinear absorption the later intensity is the same field.
        if use_kerr and use_loss:
            incident_intensity_W_m2 = np.abs(temporal_field) ** 2 / max(
                self.pulse.dt_s, np.finfo(float).tiny
            )
            incident_peak_power_W = float(
                np.max(
                    np.sum(
                        incident_intensity_W_m2
                        * self.angular_cell_area_m2[None, :, :],
                        axis=(1, 2),
                    )
                )
            )
        if use_loss:
            if z_index is None:
                raise RuntimeError("strong-field loss requires the active z index")
            strong = apply_strong_field_step(
                temporal_field,
                self.pulse.dt_s,
                step_length_m,
                float(self.cfg["wavelength_m"]),
                self.material.get("strong_field", {}),
                k0,
                cell_length_m=physical_slice_length_m,
                initial_carrier_density_m3=(
                    self._axisym_active_free_carrier_initial_m3
                    if self._active_free_carrier_z_index == z_index
                    else None
                ),
            )
            spatial_weight = float(physical_slice_length_m) / max(
                float(self.grid.dz), np.finfo(float).tiny
            )
            if self._axisym_free_carrier_final_radial_m3 is None:
                raise RuntimeError(
                    "azimuthal final free-carrier state was not initialized"
                )
            final_carrier = strong.carrier_density_m3[-1]
            # I**K generates angular harmonics outside the optical field basis.
            # Keep the positive carrier collocation samples, including in m2 mode.
            self._axisym_free_carrier_final_radial_m3[z_index] += (
                final_carrier * spatial_weight
            )
            self._axisym_free_carrier_generated_m3[z_index] += (
                strong.generated_carrier_density_m3 * spatial_weight
            )
            self._axisym_free_carrier_survival[z_index] += (
                strong.initial_carrier_survival_fraction * spatial_weight
            )
            temporal_field = strong.field
            nonlinear_intensity_path_W_m = strong.nonlinear_intensity_path_W_m
            volume_scale = 1.0 / max(
                float(self.grid.dz), np.finfo(float).tiny
            )
            for target, absorbed in (
                (self.nonlinear_loss_energy_J_m3, strong.absorbed_fluence_J_m2),
                (
                    self.multiphoton_loss_energy_J_m3,
                    strong.multiphoton_absorbed_fluence_J_m2,
                ),
                (
                    self.free_carrier_loss_energy_J_m3,
                    strong.free_carrier_absorbed_fluence_J_m2,
                ),
            ):
                target[z_index] += self._expand_angular_nearest(
                    absorbed * volume_scale,
                    conservative=True,
                    non_negative=True,
                )
            self.maximum_peak_carrier_density_m3 = max(
                self.maximum_peak_carrier_density_m3,
                strong.peak_carrier_density_m3,
            )
            self.maximum_strong_field_fractional_loss = max(
                self.maximum_strong_field_fractional_loss,
                strong.maximum_fractional_loss,
            )
        if use_kerr:
            intensity_W_m2 = np.abs(temporal_field) ** 2 / max(
                self.pulse.dt_s, np.finfo(float).tiny
            )
            if multiple_modes:
                intensity_W_m2 = np.sum(intensity_W_m2, axis=0)
            peak_power_W = float(
                np.max(
                    np.sum(
                        intensity_W_m2
                        * self.angular_cell_area_m2[None, :, :],
                        axis=(1, 2),
                    )
                )
            )
            if incident_peak_power_W is not None:
                peak_power_W = max(peak_power_W, incident_peak_power_W)
            self.maximum_nonlinear_step_peak_power_W = max(
                self.maximum_nonlinear_step_peak_power_W,
                peak_power_W,
            )
            if nonlinear_intensity_path_W_m is None:
                nonlinear_intensity_path_W_m = (
                    intensity_W_m2 * float(step_length_m)
                )
            nonlinear_phase = (
                k0
                * float(self.material.get("n2_m2_W", 0.0))
                * nonlinear_intensity_path_W_m
            )
            energy_weighted_intensity = None
            if self._longitudinal_nonlinear_cell_active():
                intensity_integral = float(
                    np.sum(
                        intensity_W_m2
                        * self.angular_cell_area_m2[None, :, :]
                    )
                )
                energy_weighted_intensity = float(
                    np.sum(
                        intensity_W_m2**2
                        * self.angular_cell_area_m2[None, :, :]
                    )
                    / max(intensity_integral, np.finfo(float).tiny)
                )
            self._update_nonlinear_phase_sampling_diagnostics(
                nonlinear_phase,
                intensity_W_m2,
                axis_names=("time", "azimuth", "radius"),
                peak_power_W=peak_power_W,
                energy_weighted_intensity_W_m2=energy_weighted_intensity,
            )
            if self._cumulative_b_integral_rad is None:
                self._cumulative_b_integral_rad = np.zeros_like(nonlinear_phase)
            self._cumulative_b_integral_rad += nonlinear_phase
            self.maximum_single_step_nonlinear_phase_rad = max(
                self.maximum_single_step_nonlinear_phase_rad,
                float(np.max(np.abs(nonlinear_phase))),
            )
            self.maximum_b_integral_rad = max(
                self.maximum_b_integral_rad,
                float(np.max(np.abs(self._cumulative_b_integral_rad))),
            )
            temporal_field *= np.exp(1j * nonlinear_phase)
        transformed = self.pulse.time_to_spectrum(
            temporal_field if multiple_modes else temporal_field[None, ...], axis=1
        )
        field = self._store_coherent_spectral_modes(field, transformed)
        return self._truncate_angular_samples(
            field,
            track_field_omission=True,
        )

    def _apply_nonlinear_step(
        self,
        field: np.ndarray,
        step_length_m: float,
        z_index: int | None,
        *,
        physical_slice_length_m: float,
    ) -> np.ndarray:
        if not self._is_reduced_field(field):
            return super()._apply_nonlinear_step(
                field,
                step_length_m,
                z_index,
                physical_slice_length_m=physical_slice_length_m,
            )
        if int(getattr(self, "maximum_azimuthal_order", 0)) > 0:
            return self._apply_azimuthal_nonlinear_step(
                field,
                step_length_m,
                z_index,
                physical_slice_length_m=physical_slice_length_m,
            )
        use_kerr = bool(
            self.physics.get("nonlinear_phase_model", "off") == "kerr_time_domain"
            and float(self.material.get("n2_m2_W", 0.0)) != 0.0
        )
        use_loss = bool(
            self.physics.get("nonlinear_loss_model", "off")
            == "strong_field_time_domain"
        )
        if not use_kerr and not use_loss:
            return field
        temporal_field = self.pulse.spectrum_to_time(field[0, :, 0, :], axis=0)
        k0 = 2.0 * math.pi / float(self.cfg["wavelength_m"])
        nonlinear_intensity_path_W_m: np.ndarray | None = None
        incident_peak_power_W: float | None = None
        # Without nonlinear absorption the later intensity is the same field.
        if use_kerr and use_loss:
            incident_intensity_W_m2 = np.abs(temporal_field) ** 2 / max(
                self.pulse.dt_s, np.finfo(float).tiny
            )
            incident_peak_power_W = float(
                np.max(
                    np.sum(
                        incident_intensity_W_m2
                        * self.radial_area_m2[None, :],
                        axis=1,
                    )
                )
            )
        if use_loss:
            if z_index is None:
                raise RuntimeError("strong-field loss requires the active z index")
            strong = apply_strong_field_step(
                temporal_field,
                self.pulse.dt_s,
                step_length_m,
                float(self.cfg["wavelength_m"]),
                self.material.get("strong_field", {}),
                k0,
                cell_length_m=physical_slice_length_m,
                initial_carrier_density_m3=(
                    self._axisym_active_free_carrier_initial_m3
                    if self._active_free_carrier_z_index == z_index
                    else None
                ),
            )
            spatial_weight = float(physical_slice_length_m) / max(
                float(self.grid.dz), np.finfo(float).tiny
            )
            if self._axisym_free_carrier_final_radial_m3 is None:
                raise RuntimeError(
                    "axisymmetric final free-carrier state was not initialized"
                )
            self._axisym_free_carrier_final_radial_m3[z_index] += (
                strong.carrier_density_m3[-1] * spatial_weight
            )
            self._axisym_free_carrier_generated_m3[z_index] += (
                strong.generated_carrier_density_m3 * spatial_weight
            )
            self._axisym_free_carrier_survival[z_index] += (
                strong.initial_carrier_survival_fraction * spatial_weight
            )
            temporal_field = strong.field
            nonlinear_intensity_path_W_m = strong.nonlinear_intensity_path_W_m
            volume_scale = 1.0 / max(float(self.grid.dz), np.finfo(float).tiny)
            self.nonlinear_loss_energy_J_m3[z_index] += self._expand_radial_nearest(
                strong.absorbed_fluence_J_m2 * volume_scale,
                conservative=True,
            )
            self.multiphoton_loss_energy_J_m3[z_index] += self._expand_radial_nearest(
                strong.multiphoton_absorbed_fluence_J_m2 * volume_scale,
                conservative=True,
            )
            self.free_carrier_loss_energy_J_m3[z_index] += self._expand_radial_nearest(
                strong.free_carrier_absorbed_fluence_J_m2 * volume_scale,
                conservative=True,
            )
            self.maximum_peak_carrier_density_m3 = max(
                self.maximum_peak_carrier_density_m3,
                strong.peak_carrier_density_m3,
            )
            self.maximum_strong_field_fractional_loss = max(
                self.maximum_strong_field_fractional_loss,
                strong.maximum_fractional_loss,
            )
        if use_kerr:
            intensity_W_m2 = np.abs(temporal_field) ** 2 / max(
                self.pulse.dt_s, np.finfo(float).tiny
            )
            peak_power_W = float(
                np.max(
                    np.sum(
                        intensity_W_m2 * self.radial_area_m2[None, :],
                        axis=1,
                    )
                )
            )
            if incident_peak_power_W is not None:
                peak_power_W = max(peak_power_W, incident_peak_power_W)
            self.maximum_nonlinear_step_peak_power_W = max(
                self.maximum_nonlinear_step_peak_power_W,
                peak_power_W,
            )
            if nonlinear_intensity_path_W_m is None:
                nonlinear_intensity_path_W_m = (
                    intensity_W_m2 * float(step_length_m)
                )
            nonlinear_phase = (
                k0
                * float(self.material.get("n2_m2_W", 0.0))
                * nonlinear_intensity_path_W_m
            )
            energy_weighted_intensity = None
            if self._longitudinal_nonlinear_cell_active():
                intensity_integral = float(
                    np.sum(intensity_W_m2 * self.radial_area_m2[None, :])
                )
                energy_weighted_intensity = float(
                    np.sum(intensity_W_m2**2 * self.radial_area_m2[None, :])
                    / max(intensity_integral, np.finfo(float).tiny)
                )
            self._update_nonlinear_phase_sampling_diagnostics(
                nonlinear_phase,
                intensity_W_m2,
                axis_names=("time", "radius"),
                peak_power_W=peak_power_W,
                energy_weighted_intensity_W_m2=energy_weighted_intensity,
            )
            if self._cumulative_b_integral_rad is None:
                self._cumulative_b_integral_rad = np.zeros_like(nonlinear_phase)
            self._cumulative_b_integral_rad += nonlinear_phase
            self.maximum_single_step_nonlinear_phase_rad = max(
                self.maximum_single_step_nonlinear_phase_rad,
                float(np.max(np.abs(nonlinear_phase))),
            )
            self.maximum_b_integral_rad = max(
                self.maximum_b_integral_rad,
                float(np.max(np.abs(self._cumulative_b_integral_rad))),
            )
            temporal_field *= np.exp(1j * nonlinear_phase)
        field[0, :, 0, :] = self.pulse.time_to_spectrum(temporal_field, axis=0)
        return field

    def _gain_pass_scalar_diagnostics(
        self,
        field: np.ndarray,
        pass_number: int,
    ) -> dict[str, Any]:
        if not self._diagnostics_enabled() or not bool(self.physics.get("_record_cycle_scalar_diagnostics", False)):
            return {}
        if not self._is_reduced_field(field):
            return super()._gain_pass_scalar_diagnostics(field, pass_number)
        temporal_modes = self.pulse.spectrum_to_time(
            self._coherent_spectral_modes(field), axis=1
        )
        # Preserve the physical time axis of each incoherent member. B is a
        # common phase, weighted by summed modal intensity, without modal
        # interference or an FFT across the flattened mode/frequency axis.
        temporal_field = np.sqrt(
            np.sum(np.abs(temporal_modes) ** 2, axis=0)
        ).astype(complex, copy=False)
        if self.maximum_azimuthal_order > 0:
            diagnostics = self._weighted_b_integral_diagnostics(
                temporal_field,
                np.broadcast_to(
                    self.angular_cell_area_m2,
                    (self.angular_samples, self.radial_bins),
                ),
                pass_number=pass_number,
                coordinate_axes=(
                    ("time_s", self.pulse.time_s),
                    ("azimuth_rad", self.angular_coordinate_rad),
                    ("radius_m", self.radial_centers_m),
                ),
                coordinate_model="finite_azimuthal_time_angle_radius",
            )
        else:
            diagnostics = self._weighted_b_integral_diagnostics(
                temporal_field[:, 0, :],
                self.radial_area_m2,
                pass_number=pass_number,
                coordinate_axes=(
                    ("time_s", self.pulse.time_s),
                    ("radius_m", self.radial_centers_m),
                ),
                coordinate_model="axisymmetric_time_radius",
            )
        return {"b_integral": diagnostics} if diagnostics else {}

    def _longitudinal_spatial_sample(
        self,
        field: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self._is_reduced_field(field):
            return super()._longitudinal_spatial_sample(field)
        fluence = np.abs(np.asarray(field)) ** 2
        if fluence.ndim > 2:
            fluence = np.sum(
                fluence,
                axis=tuple(range(fluence.ndim - 2)),
            )
        if self.maximum_azimuthal_order == 0:
            # One m=0 sample represents a complete annulus, not a ray along
            # +x.  A small exact angular quadrature recovers centroid and
            # second moments without expanding onto the Cartesian grid.
            moment_angles = 8
            theta_coordinate = np.linspace(
                0.0,
                2.0 * math.pi,
                moment_angles,
                endpoint=False,
            )
            fluence = np.broadcast_to(
                fluence,
                (moment_angles, self.radial_bins),
            )
            area = np.broadcast_to(
                self.radial_area_m2[None, :] / moment_angles,
                (moment_angles, self.radial_bins),
            )
        else:
            theta_coordinate = self.angular_coordinate_rad
            area = np.broadcast_to(
                self.angular_cell_area_m2,
                (self.angular_samples, self.radial_bins),
            )
        theta = theta_coordinate[:, None]
        radius = self.radial_centers_m[None, :]
        x_m = self._axisym_center_x_m + np.cos(theta) * radius
        y_m = self._axisym_center_y_m + np.sin(theta) * radius
        return (
            np.asarray(fluence, dtype=float),
            area,
            np.asarray(x_m, dtype=float),
            np.asarray(y_m, dtype=float),
        )

    def _longitudinal_cartesian_field(self, field: np.ndarray) -> np.ndarray:
        if self._is_reduced_field(field):
            return self._expand_radial_field(field)
        return super()._longitudinal_cartesian_field(field)

    def _intensity_snapshot_geometry(
        self, field: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self._is_reduced_field(field):
            return super()._intensity_snapshot_geometry(field)
        return np.broadcast_to(
            self.angular_cell_area_m2, (self.angular_samples, self.radial_bins),
        ), {
            "coordinate_system": "polar",
            "transverse_axes": ["theta", "radius"],
            "radius_m": self.radial_centers_m.tolist(),
            "theta_rad": self.angular_coordinate_rad.tolist(),
            "center_x_m": float(self._axisym_center_x_m),
            "center_y_m": float(self._axisym_center_y_m),
            "m0_represents_complete_annuli": self.maximum_azimuthal_order == 0,
            "azimuthal_fidelity": self.azimuthal_fidelity,
        }

    def _longitudinal_mode_metrics(self, field: np.ndarray) -> dict[str, Any]:
        if not self._is_reduced_field(field):
            return super()._longitudinal_mode_metrics(field)
        coefficients = np.fft.fft(field, axis=-2) / self.angular_samples
        modal_energy = np.sum(
            np.abs(coefficients) ** 2 * self.radial_area_m2,
            axis=tuple(index for index in range(coefficients.ndim) if index != coefficients.ndim - 2),
        )
        total = max(float(np.sum(modal_energy)), np.finfo(float).tiny)
        metrics: dict[str, Any] = {
            "azimuthal_fidelity": self.azimuthal_fidelity,
            "maximum_azimuthal_order": int(self.maximum_azimuthal_order),
        }
        for order in range(0, self.maximum_azimuthal_order + 1):
            selected = np.abs(self._angular_fft_orders) == order
            metrics[f"m{order}_energy_fraction"] = float(
                np.sum(modal_energy[selected]) / total
            )
        return metrics

    def _longitudinal_extra_metrics(
        self,
        field: np.ndarray,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = super()._longitudinal_extra_metrics(field, context)
        radial_carrier = getattr(
            self,
            "_axisym_free_carrier_final_radial_m3",
            None,
        )
        if (
            self._longitudinal_nonlinear_applicable()
            and self._is_reduced_field(field)
            and radial_carrier is not None
        ):
            metrics["final_free_carrier_density_max_m3"] = float(
                np.max(radial_carrier[int(context["z_index"])])
            )
        return metrics

    def _longitudinal_capabilities(self) -> dict[str, bool]:
        capabilities = super()._longitudinal_capabilities()
        capabilities["modes"] = True
        return capabilities

    def _gain_step(
        self,
        field: np.ndarray,
        population: np.ndarray,
        iz: int,
        step_length_m: float,
    ) -> np.ndarray:
        if not self._is_reduced_field(field):
            return super()._gain_step(field, population, iz, step_length_m)
        if self._axisym_population is None or self._axisym_passive_loss is None:
            raise RuntimeError("axisymmetric gain state was not initialized")
        full_mask = self.crystal_mask
        full_passive = self.passive_loss_energy_J_m3
        full_override = self._override_gain_m_inv
        self.crystal_mask = self._axisym_mask
        self.passive_loss_energy_J_m3 = self._axisym_passive_loss
        self._override_gain_m_inv = self._axisym_override_gain
        try:
            if self.transient_gain is not None:
                output = CoherentSignalPropagator._gain_step(
                    self,
                    field,
                    self._axisym_population,
                    iz,
                    step_length_m,
                )
                self._axisym_passive_loss = self.passive_loss_energy_J_m3
                self._enforce_cheap_material_m2_state()
                return (
                    self._truncate_angular_samples(
                        output,
                        track_field_omission=True,
                    )
                    if self.maximum_azimuthal_order > 0
                    else output
                )
            coarse_fields: list[np.ndarray] = []
            for indices in self._axisym_gain_groups:
                coarse_fields.append(
                    np.sqrt(
                        np.sum(np.abs(field[:, indices]) ** 2, axis=1)
                    ).astype(complex)
                )
            coarse_field = np.stack(coarse_fields, axis=1)
            coarse_input_fluence = np.abs(coarse_field) ** 2
            fine_spectrum = self.spectrum
            self.spectrum = self._axisym_gain_profile
            try:
                coarse_output = SpectralSignalPropagator._gain_step(
                    self,
                    coarse_field,
                    self._axisym_population,
                    iz,
                    step_length_m,
                )
            finally:
                self.spectrum = fine_spectrum
            coarse_output_fluence = np.abs(coarse_output) ** 2
            coarse_log_amplitude_gain = 0.5 * np.log(
                np.maximum(coarse_output_fluence, np.finfo(float).tiny)
                / np.maximum(coarse_input_fluence, np.finfo(float).tiny)
            )
            lower_gain = self._axisym_gain_interp_lower
            upper_gain = self._axisym_gain_interp_upper
            gain_weight = self._axisym_gain_interp_upper_weight
            fine_log_amplitude_gain = (
                coarse_log_amplitude_gain[:, lower_gain]
                * (1.0 - gain_weight[None, :, None, None])
                + coarse_log_amplitude_gain[:, upper_gain]
                * gain_weight[None, :, None, None]
            )
            # Interpolating the coarse FN gain in log-amplitude avoids the
            # artificial spectral steps (and long temporal sinc tails) created
            # by a boxcar gain per quadrature group.  One scalar correction per
            # polarization/radius restores the exact total output fluence of
            # the coarse shared-reservoir FN update, so population depletion
            # and field-energy extraction remain closed locally.
            reconstructed_scale = np.exp(fine_log_amplitude_gain)
            reconstructed_output_fluence = np.sum(
                np.abs(field * reconstructed_scale) ** 2,
                axis=1,
            )
            target_output_fluence = np.sum(coarse_output_fluence, axis=1)
            closure_scale = np.sqrt(
                target_output_fluence
                / np.maximum(
                    reconstructed_output_fluence,
                    np.finfo(float).tiny,
                )
            )
            active = target_output_fluence > np.finfo(float).eps * float(
                np.max(target_output_fluence)
            )
            if np.any(active):
                self.axisym_maximum_gain_reconstruction_normalization_correction = max(
                    self.axisym_maximum_gain_reconstruction_normalization_correction,
                    float(np.max(np.abs(closure_scale[active] - 1.0))),
                )
            output = (
                field
                * reconstructed_scale
                * closure_scale[:, None, ...]
            )
            self._axisym_passive_loss = self.passive_loss_energy_J_m3
            self._axisym_override_gain = self._override_gain_m_inv
            self._enforce_cheap_material_m2_state()
            return (
                self._truncate_angular_samples(
                    output,
                    track_field_omission=True,
                )
                if self.maximum_azimuthal_order > 0
                else output
            )
        finally:
            self.crystal_mask = full_mask
            self.passive_loss_energy_J_m3 = full_passive
            self._override_gain_m_inv = full_override

    def _estimated_gain_exponents(
        self,
        population: np.ndarray,
        z_index: int,
        step_length_m: float,
    ) -> tuple[float, float]:
        if self._axisym_population is None:
            return super()._estimated_gain_exponents(
                population, z_index, step_length_m
            )
        full_mask = self.crystal_mask
        full_override = self._override_gain_m_inv
        self.crystal_mask = self._axisym_mask
        self._override_gain_m_inv = self._axisym_override_gain
        try:
            return CoherentSignalPropagator._estimated_gain_exponents(
                self,
                self._axisym_population,
                z_index,
                step_length_m,
            )
        finally:
            self.crystal_mask = full_mask
            self._override_gain_m_inv = full_override

    def _estimated_gain_operator_exponent(
        self,
        population: np.ndarray,
        z_index: int,
        step_length_m: float,
    ) -> float:
        if self._axisym_population is None:
            return super()._estimated_gain_operator_exponent(
                population, z_index, step_length_m
            )
        full_mask = self.crystal_mask
        self.crystal_mask = self._axisym_mask
        try:
            return CoherentSignalPropagator._estimated_gain_operator_exponent(
                self,
                self._axisym_population,
                z_index,
                step_length_m,
            )
        finally:
            self.crystal_mask = full_mask

    def _local_material_step(
        self,
        field: np.ndarray,
        population: np.ndarray,
        z_index: int,
        step_length_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_reduced_field(field):
            return super()._local_material_step(
                field, population, z_index, step_length_m
            )
        if self.local_operator_settings.scheme == "sequential":
            raise ValueError(
                "axisymmetric coherent Kerr requires symmetric_adaptive local operators"
            )
        physical_substep = float(self.grid.dz) / self._current_crystal_cell_substeps
        field = self._apply_material_dispersion(field, 0.5 * step_length_m)
        field = self._apply_nonlinear_step(
            field,
            0.5 * step_length_m,
            z_index,
            physical_slice_length_m=0.5 * physical_substep,
        )
        gain_group_size = int(self._current_gain_group_size)
        if gain_group_size > 0:
            if self._axisym_passive_loss is None:
                raise RuntimeError("axisymmetric passive-loss state was not initialized")
            fluence_before = np.sum(np.abs(field) ** 2, axis=(0, 1))
            passive_before = self._axisym_passive_loss[z_index].copy()
            field = self._gain_step(
                field,
                population,
                z_index,
                step_length_m * gain_group_size,
            )
            fluence_after = np.sum(np.abs(field) ** 2, axis=(0, 1))
            radial_active_transfer = (
                (fluence_after - fluence_before) / self.grid.dz
                + self._axisym_passive_loss[z_index]
                - passive_before
            )
            active_transfer = (
                self._expand_radial_nearest(
                    radial_active_transfer,
                    conservative=True,
                )
                if self.maximum_azimuthal_order == 0
                else self._expand_angular_nearest(
                    radial_active_transfer,
                    conservative=True,
                )
            )
        else:
            active_transfer = np.zeros(self.crystal_mask.shape, dtype=float)
        field = self._apply_nonlinear_step(
            field,
            0.5 * step_length_m,
            z_index,
            physical_slice_length_m=0.5 * physical_substep,
        )
        field = self._apply_material_dispersion(field, 0.5 * step_length_m)
        return field, active_transfer

    def _gain_transport_diagnostics(self) -> dict[str, Any]:
        diagnostics = super()._gain_transport_diagnostics()
        diagnostics["axisymmetric_reduction"] = self._axisymmetric_diagnostics()
        return diagnostics

    def _axisymmetric_diagnostics(self) -> dict[str, Any]:
        transient_active = self.transient_gain is not None
        retained = [
            "coherent_frequency_time_phase",
            "radial_diffraction_and_self_focusing",
            "radial_spectral_variation",
            "local_time_domain_kerr",
            "radial_population_depletion",
        ]
        omitted: list[str] = []
        if self.maximum_azimuthal_order >= 1:
            retained.extend(
                [
                    "m1_dipole_decenter_tilt_and_thermal_prism",
                    "m1_order_dependent_radial_diffraction",
                ]
            )
        else:
            omitted.extend(
                [
                    "m1_and_higher_azimuthal_orders",
                    "beam_decenter_or_tilt",
                ]
            )
        if self.maximum_azimuthal_order >= 2:
            retained.extend(
                [
                    "m2_quadrupole_astigmatism",
                    "m2_order_dependent_radial_diffraction",
                ]
            )
            if self.azimuthal_fidelity == "m0_m1_m2_quadratic":
                retained.append("quadratic_r2_material_m2_closure")
                omitted.append("nonquadratic_radial_material_m2_structure")
            else:
                retained.append("full_radial_material_m2_structure")
            omitted.append("azimuthal_orders_above_m2")
        elif self.maximum_azimuthal_order == 1:
            omitted.extend(
                [
                    "m2_and_higher_azimuthal_orders",
                    "astigmatism_and_asymmetric_mode_breakup",
                ]
            )
        if transient_active:
            retained.append("intrapulse_front_to_tail_gain_depletion")
        else:
            omitted.append("intrapulse_front_to_tail_gain_depletion")
        if (
            self.physics.get("nonlinear_loss_model", "off")
            == "strong_field_time_domain"
        ):
            retained.append("local_time_domain_multiphoton_loss")
            carrier_model = str(
                self.material.get("strong_field", {})
                .get("free_carrier", {})
                .get("model", "off")
            )
            if carrier_model == "rate_equation":
                retained.append(
                    "radial_time_ordered_free_carrier_generation_absorption_and_refraction"
                )
        return {
            "active": True,
            "field_coordinates": (
                "complex_E_frequency_radius"
                if self.maximum_azimuthal_order == 0
                else "complex_E_frequency_azimuth_radius"
            ),
            "azimuthal_fidelity": self.azimuthal_fidelity,
            "maximum_retained_azimuthal_order": int(
                self.maximum_azimuthal_order
            ),
            "retained_azimuthal_orders": list(
                range(
                    -self.maximum_azimuthal_order,
                    self.maximum_azimuthal_order + 1,
                )
            ),
            "angular_samples": int(self.angular_samples),
            "material_m2_radial_model": (
                "quadratic_r_squared"
                if self.azimuthal_fidelity == "m0_m1_m2_quadratic"
                else (
                    "sampled_radial"
                    if self.maximum_azimuthal_order >= 2
                    else "omitted"
                )
            ),
            "radial_bins": self.radial_bins,
            "radial_diffraction_kernel_requested": (
                self.radial_diffraction_kernel_requested
            ),
            "radial_diffraction_kernel_selected": (
                self.radial_diffraction_kernel_selected
            ),
            "radial_diffraction_kernel_calls": (
                self.radial_diffraction_kernel_calls
            ),
            "radial_diffraction_numba_cold_starts": (
                self.radial_diffraction_numba_cold_starts
            ),
            "radial_numba_minimum_systems": (
                self.radial_numba_minimum_systems
            ),
            "radial_coefficient_cache": {
                "hits": self._radial_cn_cache_hits,
                "misses": self._radial_cn_cache_misses,
                "evictions": self._radial_cn_cache_evictions,
                "oversize_bypasses": (
                    self._radial_cn_cache_oversize_bypasses
                ),
                "entries": len(self._radial_cn_cache),
                "cached_bytes": self._radial_cn_cache_bytes,
                "maximum_bytes": (
                    self.radial_coefficient_cache_max_bytes
                ),
            },
            "configured_gain_spectral_bins": self.configured_gain_spectral_bins,
            "initial_frequency_tail_binning": (
                None if self._tail_frequency_plan is None else {
                    key: self._tail_frequency_plan[key] for key in (
                        "central_count", "tail_counts", "total_groups", "source_count",
                        "central_indices", "central_energy_fraction")
                }
            ),
            "gain_spectral_bins": (
                len(self.pulse.frequency_Hz)
                if transient_active
                else self.gain_spectral_bins
            ),
            "gain_spectral_reduction_active": (
                not transient_active
                and self.gain_spectral_bins < len(self.pulse.frequency_Hz)
            ),
            "gain_guard_groups": int(
                0
                if transient_active
                else sum(
                    role != "resolved" for role in self._axisym_gain_group_roles
                )
            ),
            "gain_resolved_frequency_range_Hz": list(
                self._axisym_gain_resolved_frequency_range_Hz
            ),
            "coherent_spectral_bins": len(self.pulse.frequency_Hz),
            "maximum_gain_bin_relative_frequency_span": (
                0.0
                if transient_active
                else self.axisym_maximum_gain_bin_relative_frequency_span
            ),
            "radial_extent_m": self.radial_extent_m,
            "passes_reduced": self.axisym_passes_reduced,
            "minimum_input_projection_energy_fraction": (
                self.axisym_minimum_projection_energy_fraction
            ),
            "total_input_projection_energy_fraction": (
                self.axisym_total_projection_energy_fraction
            ),
            "maximum_outer_annulus_energy_fraction": (
                self.axisym_maximum_boundary_energy_fraction
            ),
            "maximum_energy_weighted_outer_annulus_fraction": (
                self.axisym_maximum_energy_weighted_boundary_fraction
            ),
            "worst_outer_annulus_frequency_energy_fraction": (
                self.axisym_worst_boundary_component_energy_fraction
            ),
            "worst_outer_annulus_frequency_contribution_to_total_energy": (
                self.axisym_worst_boundary_component_outer_energy_fraction
            ),
            "worst_outer_annulus_frequency_wavelength_m": (
                self.axisym_worst_boundary_component_wavelength_m
            ),
            "maximum_population_azimuthal_residual": (
                self.axisym_maximum_population_azimuthal_residual
            ),
            "maximum_delta_n_azimuthal_residual": (
                self.axisym_maximum_delta_n_azimuthal_residual
            ),
            "free_carrier_representation": (
                "radial" if self.maximum_azimuthal_order == 0
                else "positive_angular_collocation_exact_survival_conservative_births"
            ),
            "maximum_free_carrier_azimuthal_residual": (
                self.axisym_maximum_free_carrier_azimuthal_residual
            ),
            "maximum_cheap_m2_material_fit_residual": (
                self.axisym_maximum_cheap_m2_material_fit_residual
            ),
            "maximum_dynamic_field_omitted_energy_fraction": (
                self.axisym_maximum_dynamic_field_omitted_energy_fraction
            ),
            "maximum_population_realizability_correction_fraction": (
                self.axisym_maximum_population_realizability_correction_fraction
            ),
            "maximum_population_positivity_redistribution_fraction": (
                self.axisym_maximum_population_positivity_redistribution_fraction
            ),
            "maximum_diffraction_norm_correction": (
                self.axisym_maximum_diffraction_norm_correction
            ),
            "population_mapping_relative_error": (
                self.axisym_population_mapping_relative_error
            ),
            "free_carrier_mapping_relative_error": (
                self.axisym_free_carrier_mapping_relative_error
            ),
            "maximum_gain_reconstruction_normalization_correction": (
                self.axisym_maximum_gain_reconstruction_normalization_correction
            ),
            "cartesian_to_radial_projection": (
                "cell_area_weighted_piecewise_linear_least_squares"
                if self.maximum_azimuthal_order == 0
                else "bilinear_polar_sampling_then_truncated_angular_fft"
            ),
            "fine_spectral_gain_reconstruction": (
                "native_coherent_frequency_grid_soft_window_gain"
                if transient_active
                else "linear_log_amplitude_interpolation_with_local_energy_closure"
            ),
            "gain_temporal_closure": (
                "soft_window_front_to_tail"
                if transient_active
                else "whole_fluence_frantz_nodvik"
            ),
            "retained": retained,
            "omitted": omitted,
        }

    def _result_extras(
        self,
        field: np.ndarray,
        source_total_J: float,
        source_spectrum_J: np.ndarray,
        output_spectrum_J: np.ndarray,
    ) -> dict[str, Any]:
        extras = super()._result_extras(
            field,
            source_total_J,
            source_spectrum_J,
            output_spectrum_J,
        )
        diagnostics = extras.get("cpa_diagnostics", {})
        diagnostics["fidelity_mode"] = self.coherent_fidelity_mode
        diagnostics["spatial_temporal_field_resolved"] = True
        diagnostics["spatial_coordinate_model"] = (
            "axisymmetric_radius"
            if self.maximum_azimuthal_order == 0
            else "finite_azimuthal_fourier_radius"
        )
        diagnostics["axisymmetric_reduction"] = self._axisymmetric_diagnostics()
        diagnostics["hybrid_approximation"] = {
            "active": True,
            "retained": self._axisymmetric_diagnostics()["retained"],
            "approximated": self._axisymmetric_diagnostics()["omitted"],
        }
        extras["cpa_diagnostics"] = diagnostics

        projection_limit = float(
            self.axisymmetric_cfg.get("maximum_projection_loss_fraction", 1e-3)
        )
        boundary_limit = float(
            self.axisymmetric_cfg.get("maximum_boundary_energy_fraction", 1e-4)
        )
        asymmetry_limit = float(
            self.axisymmetric_cfg.get("maximum_azimuthal_residual", 0.05)
        )
        realizability_limit = float(
            self.axisymmetric_cfg.get(
                "maximum_population_realizability_correction_fraction",
                0.05,
            )
        )
        positivity_redistribution_limit = float(
            self.axisymmetric_cfg.get(
                "maximum_population_positivity_redistribution_fraction",
                0.05,
            )
        )
        gain_span_limit = float(
            self.axisymmetric_cfg.get(
                "maximum_gain_bin_relative_frequency_span",
                0.05,
            )
        )
        gain_reconstruction_limit = float(
            self.axisymmetric_cfg.get(
                "maximum_gain_reconstruction_normalization_correction",
                0.02,
            )
        )
        if 1.0 - self.axisym_total_projection_energy_fraction > projection_limit:
            self.transient_gain_warnings.append(
                "The finite-azimuthal entrance projection discarded more energy "
                "than its configured applicability limit."
            )
        if (
            self.axisym_maximum_dynamic_field_omitted_energy_fraction
            > projection_limit
        ):
            self.transient_gain_warnings.append(
                "A local gain, thermal, or nonlinear operator generated more "
                "optical energy above the configured retained azimuthal order "
                "than the applicability limit."
            )
        if (
            self.axisym_maximum_energy_weighted_boundary_fraction
            > boundary_limit
        ):
            self.transient_gain_warnings.append(
                "The total-energy-weighted axisymmetric radial outer-annulus "
                "fraction exceeds its boundary limit; enlarge the optical window."
            )
        if max(
            self.axisym_maximum_population_azimuthal_residual,
            self.axisym_maximum_delta_n_azimuthal_residual,
            self.axisym_maximum_free_carrier_azimuthal_residual,
        ) > asymmetry_limit:
            self.transient_gain_warnings.append(
                "The population, thermal index, or pre-pulse free-carrier field "
                "contains too much content above the configured retained "
                "azimuthal order."
            )
        if (
            self.axisym_maximum_population_realizability_correction_fraction
            > realizability_limit
        ):
            self.transient_gain_warnings.append(
                "Making the retained angular population non-negative changed "
                "more ion inventory than the configured applicability limit; "
                "increase retained azimuthal order or use full x-y."
            )
        if (
            self.axisym_maximum_population_positivity_redistribution_fraction
            > positivity_redistribution_limit
        ):
            self.transient_gain_warnings.append(
                "Positivity preservation redistributed more of the angular "
                "population transfer than the configured applicability limit; "
                "increase retained azimuthal order or use full x-y."
            )
        if self.axisym_maximum_gain_bin_relative_frequency_span > gain_span_limit:
            self.transient_gain_warnings.append(
                "Axisymmetric gain-frequency groups are too coarse across the calibrated/occupied spectrum."
            )
        if (
            self.axisym_maximum_gain_reconstruction_normalization_correction
            > gain_reconstruction_limit
        ):
            self.transient_gain_warnings.append(
                "Axisymmetric fine-spectrum gain reconstruction required too large a local energy correction."
            )
        return extras
