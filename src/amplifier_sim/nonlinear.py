"""Strong-field loss and free-carrier response for coherent pulses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import photon_energy


@dataclass
class StrongFieldStep:
    field: np.ndarray
    absorbed_fluence_J_m2: np.ndarray
    multiphoton_absorbed_fluence_J_m2: np.ndarray
    free_carrier_absorbed_fluence_J_m2: np.ndarray
    nonlinear_intensity_path_W_m: np.ndarray
    peak_carrier_density_m3: float
    carrier_density_m3: np.ndarray
    carrier_density_bin_average_m3: np.ndarray
    maximum_fractional_loss: float
    # Surviving carriers born in this spatial volume during this time window.
    # Kept separately to avoid subtracting a large decaying initial inventory
    # when remapping the non-negative source between numerical grids.
    generated_carrier_density_m3: np.ndarray | None = None
    initial_carrier_survival_fraction: float = 1.0


def multiphoton_transmission_fraction(
    intensity_W_m2: np.ndarray | float,
    beta_SI: float,
    order: int,
    length_m: float,
) -> np.ndarray:
    """Return the exact K-photon intensity transmission without powers overflowing.

    The analytic solution is

    ``I(L)/I(0) = (1 + (K-1)*beta*I(0)**(K-1)*L)**(-1/(K-1))``.

    Forming ``I**(K-1)`` first can overflow even when the final transmitted
    intensity is representable.  Evaluating the nonlinear optical depth in
    log space keeps the public arbitrary-order model finite over the full
    floating-point input range.
    """

    intensity = np.asarray(intensity_W_m2, dtype=float)
    if np.any(~np.isfinite(intensity)) or np.any(intensity < 0.0):
        raise ValueError("multiphoton intensity must be finite and non-negative")
    if order < 2:
        raise ValueError("multiphoton order must be at least two")
    if not np.isfinite(beta_SI) or beta_SI < 0.0:
        raise ValueError("multiphoton beta must be finite and non-negative")
    if not np.isfinite(length_m) or length_m < 0.0:
        raise ValueError("multiphoton length must be finite and non-negative")
    transmission = np.ones_like(intensity)
    positive = intensity > 0.0
    if beta_SI == 0.0 or length_m == 0.0 or not np.any(positive):
        return transmission
    log_depth = (
        np.log(order - 1.0)
        + np.log(float(beta_SI))
        + np.log(float(length_m))
        + (order - 1.0) * np.log(intensity[positive])
    )
    log_one_plus_depth = np.logaddexp(0.0, log_depth)
    transmission[positive] = np.exp(
        -log_one_plus_depth / (order - 1.0)
    )
    return transmission


def _multiphoton_intensity_path(
    intensity_W_m2: np.ndarray,
    beta_SI: float,
    order: int,
    length_m: float,
) -> np.ndarray:
    """Return ``integral I(z) dz`` for ``dI/dz = -beta*I**order``.

    The normalized forms remain well conditioned when nonlinear optical depth
    tends to zero. They are exact for the analytic attenuation law used by
    :func:`apply_strong_field_step`.
    """

    intensity = np.maximum(np.asarray(intensity_W_m2, dtype=float), 0.0)
    if beta_SI == 0.0 or length_m == 0.0:
        return intensity * length_m
    path = np.zeros_like(intensity)
    positive = intensity > 0.0
    if not np.any(positive):
        return path
    log_depth = (
        np.log(order - 1.0)
        + np.log(float(beta_SI))
        + np.log(float(length_m))
        + (order - 1.0) * np.log(intensity[positive])
    )
    log_ratio = np.empty_like(log_depth)
    small = log_depth < -40.0
    large = log_depth > 40.0
    moderate = ~(small | large)
    # The ratio tends to one as nonlinear optical depth tends to zero.
    log_ratio[small] = 0.0
    if np.any(moderate):
        depth = np.exp(log_depth[moderate])
        if order == 2:
            log_ratio[moderate] = (
                np.log(np.log1p(depth)) - log_depth[moderate]
            )
        else:
            exponent = (order - 2.0) / (order - 1.0)
            numerator = np.expm1(exponent * np.log1p(depth))
            log_ratio[moderate] = (
                np.log((order - 1.0) / (order - 2.0))
                + np.log(numerator)
                - log_depth[moderate]
            )
    if np.any(large):
        if order == 2:
            # log1p(exp(x)) / exp(x) ~ x / exp(x).
            log_ratio[large] = np.log(log_depth[large]) - log_depth[large]
        else:
            # ((1+D)^p - 1)/D ~ D^(-1/(K-1)).
            log_ratio[large] = (
                np.log((order - 1.0) / (order - 2.0))
                - log_depth[large] / (order - 1.0)
            )
    log_path = (
        np.log(intensity[positive])
        + np.log(float(length_m))
        + log_ratio
    )
    if np.any(log_path > np.log(np.finfo(float).max)):
        raise FloatingPointError(
            "multiphoton intensity path exceeds the floating-point range"
        )
    path[positive] = np.exp(log_path)
    return path


def apply_strong_field_step(
    temporal_field: np.ndarray,
    dt_s: float,
    step_length_m: float,
    wavelength_m: float,
    cfg: dict[str, Any],
    k0_rad_m: float,
    cell_length_m: float | None = None,
    initial_carrier_density_m3: np.ndarray | None = None,
) -> StrongFieldStep:
    """Apply analytic K-photon attenuation and optional carrier dynamics.

    ``abs(field)**2 / dt`` is instantaneous intensity.  The K-photon loss is
    integrated analytically over ``step_length_m`` so the result stays positive
    even for a coarse z step.  Carrier generation is then advanced in ordered
    time bins and can contribute absorption and refraction.
    """

    field = np.asarray(temporal_field, dtype=complex).copy()
    if not np.isfinite(dt_s) or float(dt_s) <= 0.0:
        raise ValueError("strong-field time-bin width must be finite and positive")
    if step_length_m < 0.0 or not np.isfinite(step_length_m):
        raise ValueError("strong-field step length must be finite and non-negative")
    if not np.isfinite(wavelength_m) or float(wavelength_m) <= 0.0:
        raise ValueError("strong-field wavelength must be finite and positive")
    if not np.isfinite(k0_rad_m):
        raise ValueError("strong-field vacuum wavenumber must be finite")
    cell_length = (
        float(step_length_m)
        if cell_length_m is None
        else float(cell_length_m)
    )
    if cell_length < 0.0 or not np.isfinite(cell_length):
        raise ValueError("strong-field physical cell length must be finite and non-negative")
    intensity_before = np.abs(field) ** 2 / max(float(dt_s), np.finfo(float).tiny)
    if not np.all(np.isfinite(intensity_before)):
        raise ValueError("strong-field input intensity must be finite")
    intensity = intensity_before.copy()
    order = int(cfg.get("multiphoton_order", 2))
    beta = float(cfg.get("beta_SI", 0.0))
    if order < 2:
        raise ValueError("strong-field multiphoton_order must be at least two")
    if beta < 0.0 or not np.isfinite(beta):
        raise ValueError("strong-field beta_SI must be finite and non-negative")
    intensity_after_mpa = intensity * multiphoton_transmission_fraction(
        intensity,
        beta,
        order,
        float(step_length_m),
    )
    nonlinear_intensity_path = _multiphoton_intensity_path(
        intensity_before,
        beta,
        order,
        float(step_length_m),
    )
    carrier_cfg = cfg.get("free_carrier", {})
    carrier_model = str(carrier_cfg.get("model", "off"))
    intensity_after_carrier = intensity_after_mpa.copy()
    carrier_density = np.zeros(intensity.shape[1:], dtype=float)
    carrier_history = np.zeros_like(intensity, dtype=float)
    carrier_average_history = np.zeros_like(intensity, dtype=float)
    carrier_optical_depth = np.zeros_like(intensity)
    peak_carrier = 0.0
    incremental_density = np.zeros_like(carrier_density)
    initial_survival = 1.0
    if carrier_model == "rate_equation":
        lifetime = float(carrier_cfg.get("lifetime_s", 1e-9))
        absorption_cross_section = float(
            carrier_cfg.get("absorption_cross_section_m2", 0.0)
        )
        index_per_density = float(carrier_cfg.get("refractive_index_m3", 0.0))
        generation_efficiency = float(carrier_cfg.get("generation_efficiency", 1.0))
        if not np.isfinite(lifetime) or lifetime <= 0.0:
            raise ValueError("free-carrier lifetime must be finite and positive")
        if not np.isfinite(absorption_cross_section) or absorption_cross_section < 0.0:
            raise ValueError("free-carrier absorption must be finite and non-negative")
        if not np.isfinite(index_per_density):
            raise ValueError("free-carrier refractive-index coefficient must be finite")
        if (
            not np.isfinite(generation_efficiency)
            or not 0.0 <= generation_efficiency <= 1.0
        ):
            raise ValueError("free-carrier generation efficiency must lie in [0, 1]")
        decay = np.exp(-float(dt_s) / lifetime)
        # Exact constant-source update of dN/dt = G - N/tau.  Using G*dt
        # overestimates the carriers that survive a time bin when dt is not
        # negligible compared with the lifetime.
        source_interval = -lifetime * np.expm1(-float(dt_s) / lifetime)
        mean_survival = source_interval / max(
            float(dt_s), np.finfo(float).tiny
        )
        # tau*(1-mean_survival) tends to dt/2. Direct subtraction loses
        # that limit for femtosecond bins and long-lived carriers.
        relative_interval = float(dt_s) / lifetime
        mean_source_interval = (
            float(dt_s) * (
                0.5 - relative_interval / 6.0
                + relative_interval**2 / 24.0 - relative_interval**3 / 120.0
            )
            if relative_interval < 1e-3
            else lifetime * (1.0 - mean_survival)
        )
        quantum_energy = order * float(photon_energy(wavelength_m))
        supplied_carrier = (
            None
            if initial_carrier_density_m3 is None
            else np.asarray(initial_carrier_density_m3, dtype=float)
        )
        if supplied_carrier is not None and (
            np.any(~np.isfinite(supplied_carrier))
            or np.any(supplied_carrier < 0.0)
        ):
            raise ValueError(
                "initial_carrier_density_m3 must be finite and non-negative"
            )
        if supplied_carrier is not None and supplied_carrier.shape not in {
            intensity.shape,
            intensity.shape[1:],
        }:
            raise ValueError(
                "initial_carrier_density_m3 must match (time,y,x) or (y,x)"
            )
        # A full time history has already received its lifetime decay over this
        # window in an earlier spatial/Strang fraction.  Add only the causal
        # history generated by this fraction.  A transverse-only array is the
        # pre-window state and is advanced through time in the usual way.
        existing_history = (
            supplied_carrier
            if supplied_carrier is not None
            and supplied_carrier.shape == intensity.shape
            else None
        )
        if supplied_carrier is not None and supplied_carrier.shape == intensity.shape[1:]:
            carrier_density = supplied_carrier.copy()
        initial_survival = float(np.exp(-len(field) * float(dt_s) / lifetime))
        for time_index in range(len(field)):
            absorbed_power_density = (
                intensity[time_index] - intensity_after_mpa[time_index]
            ) / max(cell_length, np.finfo(float).tiny)
            generation_rate = (
                generation_efficiency
                * np.maximum(absorbed_power_density, 0.0)
                / max(quantum_energy, np.finfo(float).tiny)
            )
            incremental_start = incremental_density
            incremental_density = (
                incremental_start * decay + generation_rate * source_interval
            )
            if existing_history is None:
                carrier_start = carrier_density
                carrier_density = (
                    carrier_start * decay
                    + generation_rate * source_interval
                )
                carrier_average = (
                    carrier_start * mean_survival
                    + generation_rate * mean_source_interval
                )
            else:
                carrier_density = existing_history[time_index] + incremental_density
                carrier_average = (
                    existing_history[time_index]
                    + incremental_start * mean_survival
                    + generation_rate * mean_source_interval
                )
            carrier_history[time_index] = carrier_density
            carrier_average_history[time_index] = carrier_average
            carrier_transmission = np.exp(
                -absorption_cross_section * carrier_average * step_length_m
            )
            carrier_optical_depth[time_index] = (
                absorption_cross_section * carrier_average * step_length_m
            )
            intensity_after_carrier[time_index] *= carrier_transmission
            if index_per_density != 0.0:
                field[time_index] *= np.exp(
                    1j
                    * k0_rad_m
                    * index_per_density
                    * carrier_average
                    * step_length_m
                )
            peak_carrier = max(peak_carrier, float(np.max(carrier_density)))
    elif carrier_model != "off":
        raise ValueError("free_carrier.model must be off or rate_equation")
    if carrier_model == "rate_equation":
        # The carrier model assumes a time-bin-local, z-uniform carrier
        # density. Its exact linear-absorption mean-intensity factor is
        # multiplied into the exact multiphoton path integral.
        carrier_path_factor = np.ones_like(carrier_optical_depth)
        positive_depth = carrier_optical_depth > 0.0
        np.divide(
            -np.expm1(-carrier_optical_depth),
            carrier_optical_depth,
            out=carrier_path_factor,
            where=positive_depth,
        )
        nonlinear_intensity_path *= carrier_path_factor
    scale = np.sqrt(
        np.divide(
            intensity_after_carrier,
            np.maximum(intensity_before, np.finfo(float).tiny),
            out=np.zeros_like(intensity_before),
            where=intensity_before > 0.0,
        )
    )
    field *= scale
    mpa_fluence = np.sum(
        np.maximum(intensity_before - intensity_after_mpa, 0.0), axis=0
    ) * float(dt_s)
    carrier_fluence = np.sum(
        np.maximum(intensity_after_mpa - intensity_after_carrier, 0.0), axis=0
    ) * float(dt_s)
    total_fluence = mpa_fluence + carrier_fluence
    fractional = np.divide(
        intensity_before - intensity_after_carrier,
        np.maximum(intensity_before, np.finfo(float).tiny),
        out=np.zeros_like(intensity_before),
        where=intensity_before > 0.0,
    )
    return StrongFieldStep(
        field=field,
        absorbed_fluence_J_m2=total_fluence,
        multiphoton_absorbed_fluence_J_m2=mpa_fluence,
        free_carrier_absorbed_fluence_J_m2=carrier_fluence,
        nonlinear_intensity_path_W_m=nonlinear_intensity_path,
        peak_carrier_density_m3=peak_carrier,
        carrier_density_m3=carrier_history,
        carrier_density_bin_average_m3=carrier_average_history,
        maximum_fractional_loss=float(np.max(fractional)),
        generated_carrier_density_m3=incremental_density,
        # A supplied time history has already received its lifetime decay.
        initial_carrier_survival_fraction=(
            1.0 if carrier_model == "rate_equation" and existing_history is not None
            else initial_survival
        ),
    )


def apply_incoherent_strong_field_step(
    modal_temporal_field: np.ndarray,
    dt_s: float,
    step_length_m: float,
    wavelength_m: float,
    cfg: dict[str, Any],
    k0_rad_m: float,
    cell_length_m: float | None = None,
    initial_carrier_density_m3: np.ndarray | None = None,
) -> StrongFieldStep:
    """Apply one shared strong-field response to incoherent modal envelopes.

    The leading axis contains mutually incoherent spatial modes.  Multiphoton
    absorption, carrier generation/FCA, and carrier refraction are evaluated
    from their summed instantaneous intensity.  The resulting common complex
    attenuation is then applied to every mode, so no arbitrary modal ordering
    is introduced and the summed local energy ledger remains identical to the
    scalar strong-field operator.
    """

    modes = np.asarray(modal_temporal_field, dtype=complex)
    if modes.ndim < 2 or modes.shape[0] < 1:
        raise ValueError(
            "incoherent strong-field input must have a non-empty modal axis"
        )
    summed_energy_per_bin = np.sum(np.abs(modes) ** 2, axis=0)
    surrogate = np.sqrt(np.maximum(summed_energy_per_bin, 0.0)).astype(
        complex,
        copy=False,
    )
    result = apply_strong_field_step(
        surrogate,
        dt_s,
        step_length_m,
        wavelength_m,
        cfg,
        k0_rad_m,
        cell_length_m=cell_length_m,
        initial_carrier_density_m3=initial_carrier_density_m3,
    )
    common_factor = np.divide(
        result.field,
        surrogate,
        out=np.ones_like(result.field),
        where=np.abs(surrogate) > np.finfo(float).tiny,
    )
    result.field = modes * common_factor[None, ...]
    return result
