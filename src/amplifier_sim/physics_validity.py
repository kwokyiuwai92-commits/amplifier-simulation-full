"""Model-applicability diagnostics that do not alter the selected physics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .grid import Grid3D, optical_grid_from_config, pump_optical_grid_from_config
from .optical_system import transverse_carrier_nyquist_fractions
from .optics import reference_ray_samples, split_gain_path
from .profiles import spectral_profile
from .pulse_definition import intensity_fwhm
from .units import time_scale_to_s


SHORT_PULSE_LIFETIME_RATIO_WARNING = 1e-2


def _element_label(element: dict[str, Any], index: int) -> str:
    return str(element.get("name") or f"{element.get('type', 'element')}[{index}]")


def _has_explicit_clear_aperture(element: dict[str, Any]) -> bool:
    return any(
        element.get(key) is not None
        for key in (
            "aperture_radius_m",
            "aperture_radius_x_m",
            "aperture_radius_y_m",
            "aperture_width_m",
            "aperture_height_m",
        )
    )


def _abcd_reduced_order_elements(
    elements: list[dict[str, Any]],
) -> list[str]:
    affected: list[str] = []
    for index, element in enumerate(elements):
        kind = str(element.get("type", ""))
        if kind in {"aperture", "phase_screen"} or (
            kind in {"thin_lens", "thick_lens"}
            and _has_explicit_clear_aperture(element)
        ):
            affected.append(_element_label(element, index))
    return affected


def propagation_applicability(
    config: dict[str, Any],
    *,
    resolved_physics: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Report paraxial-angle and FFT carrier-sampling applicability."""

    physics = resolved_physics or config.get("physics", {})
    limits = config.get("physics", {}).get("applicability", {})
    angle_limit = float(limits.get("maximum_paraxial_angle_deg", 10.0))
    nyquist_limit = float(
        limits.get("maximum_fft_transverse_nyquist_fraction", 0.8)
    )
    routes = config.get("optical_train", {})
    physical_grid = Grid3D.from_config(config["grid"])
    seed_grid = optical_grid_from_config(
        physical_grid, config["grid"]
    )
    pump_grid = pump_optical_grid_from_config(
        physical_grid,
        config["grid"],
        seed_grid,
    )
    seed_elements = list(routes.get("seed", []))
    seed_pre_elements, _, _ = split_gain_path(seed_elements)
    pump_elements = list(routes.get("pump", []))
    pump_pre_elements, _, pump_post_elements = split_gain_path(pump_elements)
    beam_contracts: dict[str, Any] = {}
    review_reasons: list[str] = []
    for name, backend, elements in (
        (
            "seed",
            str(physics.get("optical_propagation", "off")),
            seed_elements,
        ),
        (
            "pump",
            str(physics.get("pump_propagation", "prescribed_caustic")),
            pump_elements,
        ),
    ):
        spectrum = spectral_profile(
            config[name].get("spectrum", {}),
            float(config[name]["wavelength_m"]),
            float(config[name].get("spectral_bandwidth_m", 0.0)),
        )
        wavelength = float(np.min(spectrum.coordinate))
        m2_x = float(config[name].get("m2_x", 1.0))
        m2_y = float(config[name].get("m2_y", 1.0))
        samples = reference_ray_samples(
            config[name],
            elements,
            wavelength_m=wavelength,
            m2_x=m2_x,
            m2_y=m2_y,
        )
        maximum_angle = max(
            (float(item["axis_angle_magnitude_deg"]) for item in samples),
            default=0.0,
        )
        paraxial_backend = backend in {
            "abcd",
            "super_gaussian_abcd",
            "prescribed_caustic",
            "fft",
            "hybrid_abcd_fft",
        }
        angle_ok = not paraxial_backend or maximum_angle <= angle_limit
        if not angle_ok:
            review_reasons.append(
                f"{name} {backend} axis angle {maximum_angle:.6g} deg "
                f"exceeds the configured paraxial limit {angle_limit:.6g} deg"
            )

        pre_elements = seed_pre_elements if name == "seed" else pump_pre_elements
        reduced_order_elements: list[str] = []
        ignored_elements: list[str] = []
        if backend in {"abcd", "super_gaussian_abcd", "hybrid_abcd_fft"}:
            reduced_order_elements = _abcd_reduced_order_elements(pre_elements)
            if reduced_order_elements:
                review_reasons.append(
                    f"{name} {backend} source-to-gain ABCD transport does not "
                    "apply configured aperture/phase-screen spatial reshaping for "
                    + ", ".join(reduced_order_elements)
                )
        if backend == "off" and pre_elements:
            ignored_elements = [
                _element_label(element, index)
                for index, element in enumerate(pre_elements)
            ]
            review_reasons.append(
                "seed optical_propagation=off ignores source-to-gain elements "
                + ", ".join(ignored_elements)
            )
        if name == "pump" and backend == "prescribed_caustic" and pre_elements:
            ignored_elements = [
                _element_label(element, index)
                for index, element in enumerate(pre_elements)
            ]
            review_reasons.append(
                "pump prescribed_caustic ignores source-to-gain optical elements "
                + ", ".join(ignored_elements)
            )
        if name == "pump" and pump_post_elements:
            post_labels = [
                _element_label(element, index)
                for index, element in enumerate(pump_post_elements)
            ]
            review_reasons.append(
                "pump elements after gain_medium are not executed: "
                + ", ".join(post_labels)
            )

        fft_backend = backend in {"fft", "hybrid_abcd_fft"}
        nyquist_fraction = None
        if fft_backend:
            sampled_grid = pump_grid if name == "pump" else seed_grid
            nyquist_fraction = max(
                max(
                    transverse_carrier_nyquist_fractions(
                        wavelength_m=wavelength,
                        angle_x_deg=float(item["angle_x_deg"]),
                        angle_y_deg=float(item["angle_y_deg"]),
                        refractive_index=float(
                            item["refractive_index"]
                        ),
                        m2_x=m2_x,
                        m2_y=m2_y,
                        dx_max_m=float(np.max(sampled_grid.x_widths)),
                        dy_max_m=float(np.max(sampled_grid.y_widths)),
                    )
                )
                for item in samples
            )
            if nyquist_fraction > nyquist_limit:
                review_reasons.append(
                    f"{name} FFT transverse carrier uses "
                    f"{nyquist_fraction:.6g} of Nyquist, above "
                    f"{nyquist_limit:.6g}"
                )
        beam_contracts[name] = {
            "backend": backend,
            "paraxial_contract_applies": paraxial_backend,
            "maximum_axis_angle_deg": maximum_angle,
            "maximum_paraxial_angle_deg": angle_limit,
            "angle_within_contract": angle_ok,
            "fft_transverse_nyquist_fraction": nyquist_fraction,
            "fft_shortest_sampled_wavelength_m": (
                wavelength if fft_backend else None
            ),
            "fft_grid_maximum_cell_width_x_m": (
                float(np.max(sampled_grid.x_widths))
                if fft_backend
                else None
            ),
            "fft_grid_maximum_cell_width_y_m": (
                float(np.max(sampled_grid.y_widths))
                if fft_backend
                else None
            ),
            "maximum_fft_transverse_nyquist_fraction": (
                nyquist_limit if fft_backend else None
            ),
            "direction_samples": samples,
            "reduced_order_spatial_elements": reduced_order_elements,
            "ignored_source_to_gain_elements": ignored_elements,
        }
    status = (
        "review_required_outside_propagation_contract"
        if review_reasons
        else "within_propagation_contract"
    )
    warnings = [
        "Propagation applicability: " + reason + "."
        for reason in review_reasons
    ]
    return {
        "status": status,
        "beams": beam_contracts,
        "review_reasons": review_reasons,
    }, warnings


def _relevant_relaxation_times_s(levels: dict[str, Any]) -> dict[str, float]:
    model = str(levels.get("model", "four_level"))
    times = (
        {}
        if model == "custom"
        else {"upper_state_lifetime_s": float(levels["upper_state_lifetime_s"])}
    )
    if model in {"three_level", "four_level"}:
        times["pump_level_lifetime_s"] = float(
            levels["pump_level_lifetime_s"]
        )
    if model == "four_level":
        times["lower_level_lifetime_s"] = float(
            levels["lower_level_lifetime_s"]
        )
    if model == "custom":
        rates_by_source: dict[str, float] = {}
        for index, decay in enumerate(levels.get("decays", [])):
            if decay.get("partial_rate_s") is not None:
                rate = float(decay["partial_rate_s"])
            elif decay.get("partial_lifetime_s") is not None:
                rate = 1.0 / float(decay["partial_lifetime_s"])
            elif decay.get("rate_s") not in (None, 0, 0.0):
                rate = float(decay["rate_s"])
            elif decay.get("lifetime_s") is not None:
                rate = float(decay.get("branching", 1.0)) / float(
                    decay["lifetime_s"]
                )
            else:
                rate = float(decay.get("branching_fraction", 0.0)) / float(
                    decay.get("total_lifetime_s", math.inf)
                )
            value = 1.0 / rate if rate > 0.0 else math.inf
            if math.isfinite(value) and value > 0.0:
                times[f"custom_decay_{index}_lifetime_s"] = value
                source = str(decay.get("from", f"index_{index}"))
                rates_by_source[source] = rates_by_source.get(source, 0.0) + 1.0 / value
        for source, total_rate in rates_by_source.items():
            if total_rate > 0.0:
                times[f"custom_{source}_total_lifetime_s"] = 1.0 / total_rate
    return {
        name: value
        for name, value in times.items()
        if math.isfinite(value) and value > 0.0
    }


def _intrapulse_relaxation_is_resolved(
    seed: dict[str, Any],
    model_used: str,
    cpa_diagnostics: dict[str, Any] | None,
    simulation: dict[str, Any] | None = None,
) -> bool:
    if model_used == "spectral_time_causal_multipass":
        return bool(
            (simulation or {}).get("multipass", {}).get(
                "include_dark_relaxation", True
            )
        )
    if model_used == "josa_transport":
        return bool(
            seed.get("josa", {}).get("include_relaxation_during_pulse", False)
        )
    if model_used == "commercial_app_fast":
        return bool(
            seed.get("commercial_app_fast", {}).get(
                "include_relaxation_during_pulse", False
            )
        )
    if model_used == "coherent_cpa":
        transient = (cpa_diagnostics or {}).get("transient_gain", {}) or {}
        return bool(
            transient.get("status") == "applied"
            and transient.get("include_relaxation_during_pulse", False)
        )
    return False


def pulse_relaxation_validity(
    config: dict[str, Any],
    model_used: str,
    cpa_diagnostics: dict[str, Any] | None = None,
    resolved_physics: dict[str, Any] | None = None,
    actual_fastest_dark_linear_relaxation_s: float | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Audit the short-pulse/no-relaxation assumption of a signal backend.

    The configured pulse duration is an intensity FWHM, so this is an
    applicability indicator rather than a substitute for a time-step
    convergence study.  It deliberately reports rather than changes the
    user's selected signal model.
    """

    seed = config["seed"]
    pulse_duration_s = float(seed["pulse_duration_s"])
    temporal_type = str(seed.get("temporal", {}).get("type", "gaussian"))
    pulse_duration_source = "seed.pulse_duration_s"
    if temporal_type == "file":
        rows = np.asarray(seed.get("temporal", {}).get("samples", []), dtype=float)
        if rows.ndim == 2 and rows.shape[0] >= 2 and rows.shape[1] >= 2:
            finite = np.all(np.isfinite(rows[:, :2]), axis=1)
            rows = rows[finite]
            if len(rows) >= 2:
                order = np.argsort(rows[:, 0])
                coordinate = (
                    rows[:, 0][order]
                    * time_scale_to_s(
                        seed.get("temporal", {}).get("time_unit", "s")
                    )
                )
                measured = intensity_fwhm(
                    coordinate,
                    np.maximum(rows[:, 1][order], 0.0),
                )
                if measured > 0.0:
                    pulse_duration_s = measured
                    pulse_duration_source = "seed.temporal.samples_intensity_fwhm"
    if str(model_used) == "coherent_cpa":
        coherent_fwhm = (cpa_diagnostics or {}).get("input_fwhm_s")
        if coherent_fwhm is not None:
            coherent_fwhm = float(coherent_fwhm)
            if math.isfinite(coherent_fwhm) and coherent_fwhm > 0.0:
                pulse_duration_s = coherent_fwhm
                pulse_duration_source = "coherent_cpa.actual_input_fwhm_s"
    signal_energy_J = float(seed.get("pulse_energy_J", 0.0))
    actual_model = str(model_used)
    if actual_model in {"josa_transport_fallback", "spectral_time_fallback"}:
        actual_model = "spectral_time"

    lifetimes = _relevant_relaxation_times_s(config["energy_levels"])
    if actual_fastest_dark_linear_relaxation_s is not None:
        actual_relaxation = float(actual_fastest_dark_linear_relaxation_s)
        if math.isfinite(actual_relaxation) and actual_relaxation > 0.0:
            lifetimes["actual_local_fastest_dark_linear_relaxation_s"] = (
                actual_relaxation
            )
    shortest = min(lifetimes.items(), key=lambda item: item[1]) if lifetimes else None
    shortest_name = None if shortest is None else shortest[0]
    shortest_lifetime_s = None if shortest is None else shortest[1]
    ratios = {
        f"pulse_to_{name.removesuffix('_s')}_ratio": pulse_duration_s / lifetime
        for name, lifetime in lifetimes.items()
    }
    maximum_ratio = max(ratios.values(), default=0.0)
    relaxation_resolved = _intrapulse_relaxation_is_resolved(
        seed,
        actual_model,
        cpa_diagnostics,
        config.get("simulation", {}),
    )

    if temporal_type == "cw":
        status = "not_applicable_cw"
    elif signal_energy_J <= 0.0:
        status = "not_applicable_zero_seed_energy"
    elif not lifetimes:
        status = "no_modeled_linear_relaxation"
    elif maximum_ratio <= SHORT_PULSE_LIFETIME_RATIO_WARNING:
        status = "within_short_pulse_assumption"
    elif relaxation_resolved:
        status = "intrapulse_relaxation_resolved"
    else:
        status = "review_required_unresolved_intrapulse_relaxation"

    pulse_status = status
    propagation, propagation_warnings = propagation_applicability(
        config,
        resolved_physics=resolved_physics,
    )
    if propagation["status"].startswith("review_required"):
        status = "review_required_propagation_applicability"

    diagnostics: dict[str, Any] = {
        "model_used": actual_model,
        "pulse_duration_definition": "intensity_fwhm",
        "pulse_duration_s": pulse_duration_s,
        "pulse_duration_source": pulse_duration_source,
        "relevant_relaxation_times_s": lifetimes,
        "pulse_to_relaxation_ratios": ratios,
        "maximum_pulse_to_relaxation_ratio": maximum_ratio,
        "shortest_relaxation_name": shortest_name,
        "shortest_relaxation_time_s": shortest_lifetime_s,
        "warning_threshold": SHORT_PULSE_LIFETIME_RATIO_WARNING,
        "intrapulse_relaxation_resolved": relaxation_resolved,
        "pulse_relaxation_status": pulse_status,
        "propagation": propagation,
        "status": status,
    }
    warnings: list[str] = []
    if pulse_status == "review_required_unresolved_intrapulse_relaxation":
        warnings.append(
            f"The Seed intensity-FWHM duration is {maximum_ratio:.3e} of the "
            f"shortest modeled relaxation time ({shortest_name}), above the "
            f"{SHORT_PULSE_LIFETIME_RATIO_WARNING:.0e} short-pulse diagnostic "
            f"threshold, while {actual_model} does not resolve intrapulse "
            "relaxation. Use JOSA or coherent transient gain with relaxation, "
            "or demonstrate duration/time-bin convergence before quantitative use."
        )
    warnings.extend(propagation_warnings)
    return diagnostics, warnings
