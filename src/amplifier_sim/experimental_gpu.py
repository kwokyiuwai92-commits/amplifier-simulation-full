"""Experimental resident-MLX kernel for one axisymmetric coherent pulse.

This module is deliberately not wired into the validated solver or the public
configuration schema.  Apple GPUs execute the field in ``complex64`` and the
population fractions in ``float32``; the production reference path remains
NumPy/SciPy ``complex128``/``float64``.  The prototype exists to measure a
genuinely resident implementation: after one upload, no array is converted
back to NumPy inside the z/substep loop.

Only the L4.5 single-signal-transition closure is represented.  Pumping,
periodic convergence, Cartesian/radial projection, and final diagnostics stay
on the CPU and bracket this one-pulse crystal transport experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np

from .constants import photon_energy
from .dispersion import material_dispersion_phase, material_refractive_index


@dataclass
class GPUResidentPulseResult:
    """Host copy and timing evidence from one resident GPU propagation."""

    field: np.ndarray
    population: np.ndarray
    initialization_time_s: float
    propagation_time_s: float
    download_time_s: float
    diagnostics: dict[str, Any]


_RADIAL_KERNELS: dict[int, Any] = {}
_GAIN_KERNEL: Any | None = None


def _load_mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "The experimental resident GPU path requires MLX on native "
            "Apple-silicon Python."
        ) from exc
    return mx


def _radial_kernel(mx: Any, radial_bins: int) -> Any:
    cached = _RADIAL_KERNELS.get(int(radial_bins))
    if cached is not None:
        return cached
    count = int(radial_bins)
    source = f"""
        uint frequency = thread_position_in_grid.x;
        uint frequency_count = dimensions[0];
        uint radial_count = dimensions[1];
        if (frequency >= frequency_count || radial_count != {count}) {{
            return;
        }}

        complex64_t forward[{count}];
        complex64_t propagated[{count}];
        uint base = frequency * radial_count;
        uint off_base = frequency * (radial_count - 1);
        complex64_t alpha_value = alpha[frequency];

        for (uint radial = 0; radial < radial_count; ++radial) {{
            uint index = base + radial;
            complex64_t laplacian = field[index] * complex64_t(lap_diagonal[radial]);
            if (radial > 0) {{
                laplacian += field[index - 1] * complex64_t(lap_lower[radial]);
            }}
            if (radial + 1 < radial_count) {{
                laplacian += field[index + 1] * complex64_t(lap_upper[radial]);
            }}
            complex64_t rhs = field[index] + alpha_value * laplacian;
            if (radial == 0) {{
                forward[radial] = rhs * inverse_denominator[index];
            }} else {{
                forward[radial] = (
                    rhs - lower[off_base + radial - 1] * forward[radial - 1]
                ) * inverse_denominator[index];
            }}
        }}

        propagated[radial_count - 1] = forward[radial_count - 1];
        for (int radial = int(radial_count) - 2; radial >= 0; --radial) {{
            propagated[radial] = forward[radial] - (
                cprime[off_base + uint(radial)] * propagated[radial + 1]
            );
        }}

        float before = 0.0f;
        float after = 0.0f;
        for (uint radial = 0; radial < radial_count; ++radial) {{
            complex64_t old_value = field[base + radial];
            complex64_t new_value = propagated[radial];
            before += (
                old_value.real * old_value.real
                + old_value.imag * old_value.imag
            ) * radial_area[radial];
            after += (
                new_value.real * new_value.real
                + new_value.imag * new_value.imag
            ) * radial_area[radial];
        }}
        float scale = metal::sqrt(before / metal::max(after, 1.0e-30f));
        for (uint radial = 0; radial < radial_count; ++radial) {{
            propagated_field[base + radial] = (
                propagated[radial] * complex64_t(scale)
            );
        }}
    """
    kernel = mx.fast.metal_kernel(
        name=f"axisymmetric_cn_r{count}",
        input_names=[
            "field",
            "alpha",
            "lower",
            "cprime",
            "inverse_denominator",
            "lap_lower",
            "lap_diagonal",
            "lap_upper",
            "radial_area",
            "dimensions",
        ],
        output_names=["propagated_field"],
        source=source,
        compile_options={"math_mode": "safe"},
    )
    _RADIAL_KERNELS[count] = kernel
    return kernel


def _gain_kernel(mx: Any) -> Any:
    global _GAIN_KERNEL
    if _GAIN_KERNEL is not None:
        return _GAIN_KERNEL
    source = r"""
        uint radial = thread_position_in_grid.x;
        uint frequency_count = dimensions[0];
        uint radial_count = dimensions[1];
        if (radial >= radial_count) {
            return;
        }

        float lower_value = lower_fraction[radial];
        float upper_value = upper_fraction[radial];
        float signal_pair_value = lower_value + upper_value;
        float density_value = total_density[radial];
        float correction_value = correction[radial];
        float dz = parameters[0];
        float cell_length = parameters[1];
        bool reverse_first = flags[0] != 0;

        for (uint sweep = 0; sweep < 2; ++sweep) {
            bool descending = sweep == 0 ? reverse_first : !reverse_first;
            for (uint position = 0; position < frequency_count; ++position) {
                uint frequency = descending
                    ? frequency_count - 1 - position
                    : position;
                uint index = frequency * radial_count + radial;
                complex64_t value = field[index];
                float actual = value.real * value.real + value.imag * value.imag;
                float input_value = 0.5f * actual * correction_value;
                float exponent = (
                    sigma_em[index] * upper_value
                    - sigma_abs[index] * lower_value
                ) * density_value * dz;
                float saturation_value = saturation[index];
                float output_value;

                if (metal::isfinite(saturation_value) && saturation_value > 0.0f) {
                    float scaled = metal::max(input_value, 0.0f) / saturation_value;
                    float log_expm1;
                    if (scaled > 50.0f) {
                        log_expm1 = scaled;
                    } else if (scaled > 0.0f) {
                        log_expm1 = metal::log(stable_expm1(scaled));
                    } else {
                        log_expm1 = -INFINITY;
                    }
                    float combined = exponent + log_expm1;
                    float logarithm = combined > 0.0f
                        ? combined + log1p(metal::exp(-combined))
                        : log1p(metal::exp(combined));
                    output_value = saturation_value * logarithm;

                    float event_denominator = (
                        photon_energy_J[frequency] * cell_length * density_value
                    );
                    float requested_fraction = event_denominator > 0.0f
                        ? (output_value - input_value) / event_denominator
                        : 0.0f;
                    float required = metal::abs(requested_fraction);
                    float available = requested_fraction > 0.0f
                        ? upper_value
                        : lower_value;
                    float limiter = required > 0.0f
                        ? metal::clamp(available / required, 0.0f, 1.0f)
                        : 1.0f;
                    float applied = requested_fraction * limiter;
                    upper_value = metal::clamp(
                        upper_value - applied, 0.0f, signal_pair_value
                    );
                    lower_value = signal_pair_value - upper_value;
                    output_value = input_value + applied * event_denominator;
                } else {
                    output_value = input_value * metal::exp(metal::min(exponent, 50.0f));
                }
                output_value = metal::max(output_value, 0.0f);
                if (sweep == 0) {
                    output_fluence[index] = output_value;
                } else {
                    output_fluence[index] += output_value;
                }
            }
        }

        for (uint frequency = 0; frequency < frequency_count; ++frequency) {
            uint index = frequency * radial_count + radial;
            complex64_t value = field[index];
            float actual = value.real * value.real + value.imag * value.imag;
            float denominator = actual * correction_value;
            float ratio = denominator > 1.0e-30f
                ? output_fluence[index] / denominator
                : 0.0f;
            processed_field[index] = value * complex64_t(
                metal::sqrt(metal::max(ratio, 0.0f))
            );
        }
        lower_out[radial] = lower_value;
        upper_out[radial] = upper_value;
    """
    _GAIN_KERNEL = mx.fast.metal_kernel(
        name="single_transition_soft_window_sweep",
        input_names=[
            "field",
            "correction",
            "lower_fraction",
            "upper_fraction",
            "total_density",
            "sigma_abs",
            "sigma_em",
            "saturation",
            "photon_energy_J",
            "parameters",
            "flags",
            "dimensions",
        ],
        output_names=[
            "processed_field",
            "lower_out",
            "upper_out",
            "output_fluence",
        ],
        source=source,
        header=r"""
            inline float stable_expm1(float value) {
                float magnitude = metal::abs(value);
                if (magnitude < 1.0e-3f) {
                    float square = value * value;
                    return value + 0.5f * square
                        + value * square / 6.0f
                        + square * square / 24.0f;
                }
                return metal::exp(value) - 1.0f;
            }
        """,
        compile_options={"math_mode": "safe"},
    )
    return _GAIN_KERNEL


def _fft_shift(mx: Any, values: Any, *, inverse: bool) -> Any:
    count = int(values.shape[0])
    shift = -(count // 2) if inverse else count // 2
    return mx.roll(values, shift=shift, axis=0)


def _spectrum_to_time(mx: Any, field: Any) -> Any:
    # Match CoherentPulse: physical time dependence is exp(-i omega t).
    shifted = _fft_shift(mx, field, inverse=True)
    return _fft_shift(
        mx,
        mx.fft.fft(shifted, axis=0, norm="ortho"),
        inverse=False,
    )


def _time_to_spectrum(mx: Any, field: Any) -> Any:
    shifted = _fft_shift(mx, field, inverse=True)
    return _fft_shift(
        mx,
        mx.fft.ifft(shifted, axis=0, norm="ortho"),
        inverse=False,
    )


class ResidentAxisymmetricMLXStepper:
    """Keep one L4.5 crystal field and its signal populations on Metal."""

    def __init__(
        self,
        propagator: Any,
        population: np.ndarray,
        radial_delta_n: np.ndarray,
        *,
        z_cells: int,
        substeps_per_cell: int,
        slice_count: int,
        maximum_gain_group_size: int = 1,
    ):
        initialized_at = time.perf_counter()
        self.mx = _load_mlx()
        self.propagator = propagator
        transient = dict(getattr(propagator, "transient_gain_cfg", {}) or {})
        unsupported: list[str] = []
        if not bool(transient.get("enabled", False)):
            unsupported.append("transient gain must be enabled")
        if str(transient.get("method", "")) != "soft_window_spectral":
            unsupported.append("only soft_window_spectral gain is implemented")
        if bool(transient.get("include_relaxation_during_pulse", False)):
            unsupported.append("intrapulse population relaxation is not implemented")
        if (
            abs(
                float(
                    propagator.material.get("signal_passive_loss_m_inv", 0.0)
                )
            )
            > 0.0
        ):
            unsupported.append("signal passive loss is not implemented")
        if str(propagator.physics.get("nonlinear_loss_model", "off")) != "off":
            unsupported.append("strong-field nonlinear loss is not implemented")
        if getattr(propagator, "_override_gain_m_inv", None) is not None:
            unsupported.append("gain overrides are not implemented")
        if unsupported:
            raise ValueError(
                "Unsupported resident GPU prototype configuration: "
                + "; ".join(unsupported)
            )
        self.frequency_count = len(propagator.pulse.frequency_Hz)
        self.radial_bins = int(propagator.radial_bins)
        self.z_cells = min(max(int(z_cells), 1), int(population.shape[1]))
        self.substeps = max(int(substeps_per_cell), 1)
        self.maximum_gain_group_size = min(
            max(int(maximum_gain_group_size), 1),
            self.substeps,
        )
        gain_update_count = int(
            math.ceil(self.substeps / self.maximum_gain_group_size)
        )
        base_group, group_remainder = divmod(
            self.substeps,
            gain_update_count,
        )
        group_sizes = [
            base_group + int(index < group_remainder)
            for index in range(gain_update_count)
        ]
        gain_schedule = [0] * self.substeps
        group_start = 0
        for group_size in group_sizes:
            representative = group_start + (group_size - 1) // 2
            gain_schedule[representative] = group_size
            group_start += group_size
        self.gain_schedule = tuple(gain_schedule)
        self.slice_count = min(
            max(int(slice_count), 2), self.frequency_count
        )
        if len(propagator.levels.signal_transitions) != 1:
            raise ValueError(
                "The resident GPU prototype requires exactly one Signal transition"
            )
        transition = propagator.levels.signal_transitions[0]
        self.lower_level = int(transition.lower)
        self.upper_level = int(transition.upper)
        initial = np.asarray(population, dtype=float)[
            :, : self.z_cells, 0, :
        ]
        if initial.shape[-1] != self.radial_bins:
            raise ValueError("population does not match the radial GPU grid")
        total = np.sum(initial, axis=0)
        if np.any(total <= 0.0):
            raise ValueError("resident GPU population density must be positive")
        self._initial_population = initial.copy()
        self._total_density_np = total
        lower_fraction = initial[self.lower_level] / total
        upper_fraction = initial[self.upper_level] / total

        radial_dn = np.asarray(radial_delta_n, dtype=float)
        if radial_dn.ndim == 3 and radial_dn.shape[-2] == 1:
            radial_dn = radial_dn[:, 0, :]
        radial_dn = radial_dn[: self.z_cells]
        if radial_dn.shape != (self.z_cells, self.radial_bins):
            raise ValueError("radial_delta_n must have shape (z, 1, r) or (z, r)")

        wavelengths = np.asarray(propagator.pulse.wavelength_m, dtype=float)
        frequencies = np.asarray(propagator.pulse.frequency_Hz, dtype=float)
        sigma_abs = np.empty(
            (self.z_cells, self.frequency_count, self.radial_bins),
            dtype=np.float32,
        )
        sigma_em = np.empty_like(sigma_abs)
        saturation = np.empty_like(sigma_abs)
        for z_index in range(self.z_cells):
            local_abs, local_em = propagator.levels.local_cross_sections(
                "signal",
                wavelengths,
                (1, self.radial_bins),
                z_index,
            )
            local_abs = np.asarray(local_abs[:, 0, :], dtype=float)
            local_em = np.asarray(local_em[:, 0, :], dtype=float)
            sigma_abs[z_index] = local_abs.astype(np.float32)
            sigma_em[z_index] = local_em.astype(np.float32)
            sigma_sum = local_abs + local_em
            saturation[z_index] = np.divide(
                photon_energy(wavelengths)[:, None],
                sigma_sum,
                out=np.full_like(sigma_sum, np.inf),
                where=sigma_sum > 0.0,
            ).astype(np.float32)

        local_length = float(propagator.grid.dz) / self.substeps
        half_local = 0.5 * local_length
        dispersion = material_dispersion_phase(
            frequencies,
            propagator.material.get("dispersion", {}),
            half_local,
        )
        dispersion_factor = np.exp(1j * dispersion).astype(np.complex64)
        refractive_index = np.asarray(
            [
                material_refractive_index(
                    propagator.material,
                    float(wavelength),
                    role="signal",
                )
                for wavelength in wavelengths
            ],
            dtype=float,
        )
        alpha, lower, cprime, inverse = propagator._radial_cn_coefficients(
            0.5 * local_length,
            refractive_index,
        )
        windows, _ = self._soft_windows(
            self.frequency_count, self.slice_count
        )
        coverage = np.sum(windows**2, axis=0).astype(np.float32)

        mx = self.mx
        self._lower_initial = mx.array(lower_fraction.astype(np.float32))
        self._upper_initial = mx.array(upper_fraction.astype(np.float32))
        self._total_density = mx.array(total.astype(np.float32))
        self._delta_n = mx.array(radial_dn.astype(np.float32))
        self._sigma_abs = mx.array(sigma_abs)
        self._sigma_em = mx.array(sigma_em)
        self._saturation = mx.array(saturation)
        self._photon_energy = mx.array(
            photon_energy(wavelengths).astype(np.float32)
        )
        self._wavelength = mx.array(wavelengths.astype(np.float32))
        self._dispersion_factor = mx.array(dispersion_factor)[:, None]
        self._windows = mx.array(windows.astype(np.float32))
        self._coverage = mx.array(coverage)
        self._radial_coefficients = (
            mx.array(np.asarray(alpha, dtype=np.complex64)),
            mx.array(np.asarray(lower, dtype=np.complex64)),
            mx.array(np.asarray(cprime, dtype=np.complex64)),
            mx.array(np.asarray(inverse, dtype=np.complex64)),
        )
        self._laplacian = (
            mx.array(
                np.asarray(propagator._radial_laplacian_lower, dtype=np.float32)
            ),
            mx.array(
                np.asarray(
                    propagator._radial_laplacian_diagonal, dtype=np.float32
                )
            ),
            mx.array(
                np.asarray(propagator._radial_laplacian_upper, dtype=np.float32)
            ),
        )
        self._radial_area = mx.array(
            np.asarray(propagator.radial_area_m2, dtype=np.float32)
        )
        self._dimensions = mx.array(
            np.asarray(
                [self.frequency_count, self.radial_bins], dtype=np.uint32
            )
        )
        self._gain_parameters = {
            group_size: mx.array(
                np.asarray(
                    [local_length * group_size, propagator.grid.dz],
                    dtype=np.float32,
                )
            )
            for group_size in set(self.gain_schedule)
            if group_size > 0
        }
        self._radial_kernel = _radial_kernel(mx, self.radial_bins)
        self._gain_kernel = _gain_kernel(mx)
        self._local_length = local_length
        self._half_local = half_local
        self._kerr_coefficient = (
            2.0
            * math.pi
            / float(propagator.cfg["wavelength_m"])
            * float(propagator.material.get("n2_m2_W", 0.0))
            * half_local
            / max(float(propagator.pulse.dt_s), np.finfo(float).tiny)
        )
        mx.eval(
            self._lower_initial,
            self._upper_initial,
            self._total_density,
            self._sigma_abs,
            self._sigma_em,
            self._saturation,
            self._dispersion_factor,
            self._windows,
            *self._radial_coefficients,
        )
        self.initialization_time_s = time.perf_counter() - initialized_at

    @staticmethod
    def _soft_windows(
        sample_count: int,
        slice_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Local import keeps the optional MLX module isolated from the validated
        # CPU path while reusing the exact production window definition.
        from .coherent_gain import soft_sine_squared_windows

        return soft_sine_squared_windows(sample_count, slice_count)

    def _diffract(self, field: Any) -> Any:
        mx = self.mx
        alpha, lower, cprime, inverse = self._radial_coefficients
        lap_lower, lap_diagonal, lap_upper = self._laplacian
        return self._radial_kernel(
            inputs=[
                field,
                alpha,
                lower,
                cprime,
                inverse,
                lap_lower,
                lap_diagonal,
                lap_upper,
                self._radial_area,
                self._dimensions,
            ],
            grid=(self.frequency_count, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[field.shape],
            output_dtypes=[mx.complex64],
        )[0]

    def _kerr(self, field: Any) -> Any:
        if self._kerr_coefficient == 0.0:
            return field
        mx = self.mx
        temporal = _spectrum_to_time(mx, field)
        intensity_like = mx.abs(temporal) ** 2
        phase = intensity_like * self._kerr_coefficient
        temporal = temporal * mx.exp(mx.array(1j, dtype=mx.complex64) * phase)
        return _time_to_spectrum(mx, temporal)

    def _transient_gain(
        self,
        field: Any,
        lower_fraction: Any,
        upper_fraction: Any,
        z_index: int,
        reverse_spectral_order: bool,
        gain_group_size: int,
    ) -> tuple[Any, Any, Any]:
        mx = self.mx
        temporal = _spectrum_to_time(mx, field)
        temporal_energy = mx.abs(temporal) ** 2
        total = mx.sum(temporal_energy, axis=0)
        covered = mx.sum(
            self._coverage[:, None] * temporal_energy,
            axis=0,
        )
        correction = mx.where(covered > 1.0e-30, total / covered, 1.0)
        result_time = mx.zeros_like(temporal)
        flags = mx.array(
            np.asarray([int(reverse_spectral_order)], dtype=np.uint32)
        )
        for window_index in range(self.slice_count):
            slice_time = temporal * self._windows[window_index, :, None]
            slice_spectrum = _time_to_spectrum(mx, slice_time)
            outputs = self._gain_kernel(
                inputs=[
                    slice_spectrum,
                    correction,
                    lower_fraction,
                    upper_fraction,
                    self._total_density[z_index],
                    self._sigma_abs[z_index],
                    self._sigma_em[z_index],
                    self._saturation[z_index],
                    self._photon_energy,
                    self._gain_parameters[int(gain_group_size)],
                    flags,
                    self._dimensions,
                ],
                grid=(self.radial_bins, 1, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[
                    slice_spectrum.shape,
                    lower_fraction.shape,
                    upper_fraction.shape,
                    slice_spectrum.shape,
                ],
                output_dtypes=[
                    mx.complex64,
                    mx.float32,
                    mx.float32,
                    mx.float32,
                ],
            )
            processed, lower_fraction, upper_fraction = outputs[:3]
            result_time = result_time + _spectrum_to_time(mx, processed)
        return (
            _time_to_spectrum(mx, result_time),
            lower_fraction,
            upper_fraction,
        )

    def propagate(self, radial_field: np.ndarray) -> GPUResidentPulseResult:
        """Run one field through the configured z range with no host loop transfers."""

        mx = self.mx
        field_np = np.asarray(radial_field)
        if field_np.ndim == 4:
            if field_np.shape[0] != 1 or field_np.shape[-2] != 1:
                raise ValueError("radial_field must contain one polarization")
            field_np = field_np[0, :, 0, :]
        if field_np.shape != (self.frequency_count, self.radial_bins):
            raise ValueError("radial_field must have shape (frequency, radius)")

        upload_started = time.perf_counter()
        field = mx.array(field_np.astype(np.complex64))
        lower_by_z = [self._lower_initial[index] for index in range(self.z_cells)]
        upper_by_z = [self._upper_initial[index] for index in range(self.z_cells)]
        mx.eval(field, *lower_by_z, *upper_by_z)
        upload_time = time.perf_counter() - upload_started

        mx.reset_peak_memory()
        gain_call = 0
        propagation_started = time.perf_counter()
        for z_index in range(self.z_cells):
            lower_fraction = lower_by_z[z_index]
            upper_fraction = upper_by_z[z_index]
            for substep_index in range(self.substeps):
                field = self._diffract(field)
                thermal_phase = (
                    2.0
                    * math.pi
                    * self._local_length
                    * self._delta_n[z_index][None, :]
                    / self._wavelength[:, None]
                )
                field = field * mx.exp(
                    mx.array(1j, dtype=mx.complex64) * thermal_phase
                )
                field = field * self._dispersion_factor
                field = self._kerr(field)
                gain_group_size = self.gain_schedule[substep_index]
                if gain_group_size > 0:
                    field, lower_fraction, upper_fraction = self._transient_gain(
                        field,
                        lower_fraction,
                        upper_fraction,
                        z_index,
                        reverse_spectral_order=bool(gain_call % 2),
                        gain_group_size=gain_group_size,
                    )
                    gain_call += 1
                field = self._kerr(field)
                field = field * self._dispersion_factor
                field = self._diffract(field)
            lower_by_z[z_index] = lower_fraction
            upper_by_z[z_index] = upper_fraction
            # Evaluate the graph without materializing a host array.  This keeps
            # memory bounded while the accepted field/population stay on Metal.
            mx.eval(field, lower_fraction, upper_fraction)
            # Completed FFT/window temporaries are cacheable but never reused
            # across z because the accepted field and population have changed.
            # Retain those accepted arrays on the GPU while releasing cached
            # command buffers and temporaries before constructing the next cell.
            mx.clear_cache()
        propagation_time = time.perf_counter() - propagation_started

        download_started = time.perf_counter()
        field_out = np.asarray(field).astype(np.complex64, copy=True)
        lower_out = np.asarray(mx.stack(lower_by_z, axis=0)).astype(float)
        upper_out = np.asarray(mx.stack(upper_by_z, axis=0)).astype(float)
        download_time = time.perf_counter() - download_started

        population_out = self._initial_population.copy()
        population_out[self.lower_level] = lower_out * self._total_density_np
        population_out[self.upper_level] = upper_out * self._total_density_np
        diagnostics = {
            "experimental": True,
            "device": str(mx.default_device()),
            "field_dtype": "complex64",
            "population_fraction_dtype": "float32",
            "z_cells": self.z_cells,
            "substeps_per_cell": self.substeps,
            "maximum_gain_group_size": self.maximum_gain_group_size,
            "gain_update_schedule": list(self.gain_schedule),
            "soft_window_count": self.slice_count,
            "gain_frequency_bins": self.frequency_count,
            "gain_frequency_reduction": False,
            "adaptive_substeps": False,
            "adaptive_soft_windows": False,
            "gain_calls": gain_call,
            "host_array_materializations_inside_z_loop": 0,
            "gpu_peak_memory_bytes": int(mx.get_peak_memory()),
            "gpu_active_memory_bytes_after_run": int(mx.get_active_memory()),
            "gpu_cache_memory_bytes_after_run": int(mx.get_cache_memory()),
            "upload_time_s": upload_time,
            "precision_contract": (
                "experimental_float32_complex64_not_validated_reference"
            ),
        }
        return GPUResidentPulseResult(
            field=field_out[None, :, None, :],
            population=population_out[..., None, :],
            initialization_time_s=self.initialization_time_s,
            propagation_time_s=propagation_time,
            download_time_s=download_time,
            diagnostics=diagnostics,
        )
