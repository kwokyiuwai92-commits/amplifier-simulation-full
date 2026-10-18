"""One result-validity gate shared by CLI, UI, and parameter sweeps."""

from __future__ import annotations

import math
from typing import Any


VALIDATION_MODES = {"exploratory", "strict"}
NON_QUANTITATIVE_BANNER = "NON-QUANTITATIVE — VALIDATION FAILED"
STRICT_DIAGNOSTIC_CONTRACT_SCHEMA_VERSION = 1

# These registries are deliberately explicit.  Adding a solver choice without
# registering its strict evidence contract must turn a strict result invalid;
# otherwise an unfamiliar backend could inherit only the generic fields and
# silently pass because its backend-specific diagnostics were never declared.
_REGISTERED_STRICT_BACKENDS: dict[str, frozenset[str]] = {
    "pump": frozenset(
        {
            "prescribed_caustic",
            "abcd",
            "super_gaussian_abcd",
            "fft",
            "hybrid_abcd_fft",
            "stochastic_raytrace",
        }
    ),
    "signal": frozenset(
        {
            "fluence",
            "spectral_time",
            "spectral_time_causal_multipass",
            "coherent_cpa",
            "commercial_app_fast",
            "josa_transport",
            "incoherent_hg_shared_reservoir",
            "spectral_time_fallback",
            "coherent_cpa_fallback",
            "josa_transport_fallback",
        }
    ),
    "population": frozenset(
        {
            "frozen",
            "steady_state",
            "transient",
            "transient_waveform",
            "transient_pulse_train",
            "cycle_averaged_steady_state",
            "periodic_steady_state",
            "reduced_periodic_steady_state",
        }
    ),
    "thermal": frozenset(
        {"red_black_sor", "scipy_conjugate_gradient"}
    ),
    "stress": frozenset(
        {
            "none",
            "isotropic_proxy",
            "thermoelastic_fem",
            "anisotropic_thermoelastic_fem",
        }
    ),
    "ase": frozenset(
        {
            "off",
            "six_ray",
            # The six-ray transport reports this more specific executed
            # backend name.  Retain the public configuration spelling above
            # for disabled and pre-execution summaries.
            "six_ray_discrete_ordinates",
            "angular_discrete_ordinates",
        }
    ),
    "grid_coupling": frozenset({"numpy_dense", "scipy_csr"}),
    "thermal_heat_source": frozenset(
        {"pump_component_ledger", "event_energy_ledger"}
    ),
}


def build_required_diagnostic_contract(
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Declare evidence required from the backends that actually executed.

    The value checks below intentionally remain separate from this manifest.
    This first layer prevents a new backend from passing strict validation by
    simply omitting a diagnostic key and thereby turning a check into
    ``not_applicable``.
    """

    pump = summary.get("pump", {}) or {}
    signal = summary.get("signal", {}) or {}
    population = summary.get("population", {}) or {}
    thermal = summary.get("thermal", {}) or {}
    stress = summary.get("stress", {}) or {}
    ase = summary.get("ase", {}) or {}
    grid_coupling = summary.get("grid_coupling", {}) or {}
    event_energy = thermal.get("event_energy_ledger", {}) or {}

    domains: dict[str, dict[str, Any]] = {
        "pump": {
            "backend": str(pump.get("propagation", "unknown")),
            "required": [
                "pump.propagation",
                "pump.depletion",
                "pump.grid_input_power_W",
                "pump.output_power_W",
                "pump.absorbed_power_W",
                "pump.maximum_pump_absorption_optical_depth",
            ],
            "required_any": [
                [
                    "pump.crystal_power_balance_residual_relative",
                    "pump.source_to_exit_balance_residual_relative",
                ],
                [
                    "pump.positive_pump_exponent_clipping_active",
                    "pump.positive_pump_ray_exponent_clipping_active",
                ],
            ],
        },
        "signal": {
            "backend": str(signal.get("executed_model", "unknown")),
            "required": [
                "signal.executed_model",
                "signal.propagation",
                "signal.phase_insensitive_spectral_phase_requested",
                "energy_ledger.seed_pulse_energy_J.input",
                "energy_ledger.seed_pulse_energy_J.output",
                "energy_ledger.seed_pulse_energy_J.active_transition_transfer",
                "energy_ledger.seed_pulse_energy_J.unresolved_external_interface_and_routing_term",
                "energy_ledger.level_storage_J.available",
                "energy_ledger.level_storage_J.contract",
            ],
            "required_any": [],
        },
        "population": {
            "backend": str(population.get("dynamics", "unknown")),
            "required": [
                "population.dynamics",
                "population.conservation_max_relative",
                "population.numerical_population_correction.maximum_negative_population_fraction",
                "population.numerical_population_correction.maximum_conservation_error_before_renormalization_relative",
                "population.numerical_population_correction.maximum_component_inventory_error_before_renormalization_relative",
            ],
            "required_any": [],
        },
        "applicability_and_boundaries": {
            "backend": "shared_solver_evidence",
            "required": [
                "physics.pump_depletion",
                "physics_validity.status",
                "thermal.enabled",
                "stress.model",
                "grid_coupling.enabled",
                "ase.requested",
                "ase.executed",
                "spectroscopic_response.material_range.out_of_range_total",
                "spectroscopic_response.material_range.unweighted_only_out_of_range_total",
                "spectroscopic_response.material_range.power_weighted_out_of_range_fraction",
                "optical_train.boundary_window_diagnostics.beams",
            ],
            "required_any": [],
        },
    }
    if str(population.get("dynamics", "")) in {
        "steady_state",
        "cycle_averaged_steady_state",
        "periodic_steady_state",
    } or population.get("converged") is not None:
        domains["population"]["required"].append("population.converged")

    cycle_dynamics = str(population.get("dynamics", ""))
    cycle_timing = population.get("cycle_timing", {}) or {}
    if cycle_dynamics in {
        "transient_pulse_train",
        "periodic_steady_state",
        "reduced_periodic_steady_state",
    }:
        domains["population"]["required"].extend(
            [
                "population.cycle_timing.repetition_period_s",
                "population.cycle_timing.population_dark_duration_s",
                "population.cycle_timing.population_seed_resolved_elapsed_time_s",
                "population.cycle_timing.population_time_closure_residual_s",
            ]
        )
        if cycle_timing.get("free_carrier_seed_resolved_elapsed_time_s") is not None:
            domains["population"]["required"].extend(
                [
                    "population.cycle_timing.free_carrier_dark_duration_s",
                    "population.cycle_timing.free_carrier_seed_resolved_elapsed_time_s",
                    "population.cycle_timing.free_carrier_time_closure_residual_s",
                ]
            )

    if str(pump.get("propagation", "")) == "hybrid_abcd_fft":
        domains["pump"]["required"].append(
            "pump.gain_entrance_capture_fraction"
        )

    if str(signal.get("propagation", "")) in {"abcd", "hybrid_abcd_fft"}:
        domains["signal"]["required"].append(
            "signal.gain_entrance_capture_fraction"
        )

    if bool(thermal.get("enabled", False)):
        domains["thermal"] = {
            "backend": str(thermal.get("linear_solver", "unknown")),
            "required": [
                "thermal.model",
                "thermal.linear_solver",
                "thermal.heat_source_model",
                "thermal.converged",
                "thermal.residual_K",
                "thermal.algebraic_residual_relative",
                "thermal.energy_balance_relative_residual",
                "thermal.heat_input_W",
                "thermal.boundary_heat_removal_W",
            ],
            "required_any": [],
        }

    executed_model = str(signal.get("executed_model", ""))
    if executed_model in {
        "spectral_time",
        "spectral_time_causal_multipass",
        "coherent_cpa",
        "josa_transport",
        "commercial_app_fast",
        "incoherent_hg_shared_reservoir",
        "spectral_time_fallback",
        "coherent_cpa_fallback",
        "josa_transport_fallback",
    }:
        domains["signal"]["required"].extend(
            ["signal.gain_transport", "signal.gain_transport.backend"]
        )
        domains["signal"]["required_any"].append(
            [
                "signal.gain_transport.positive_cell_exponent_clipping_active",
                "signal.gain_transport.positive_gain_exponent_clipping_active",
            ]
        )
    if executed_model == "spectral_time_causal_multipass":
        domains["signal"]["required"].extend(
            [
                "signal.gain_transport.multipass.enabled",
                "signal.gain_transport.multipass.template",
                "signal.gain_transport.multipass.scheme",
                "signal.gain_transport.multipass.cfl_time_step_s",
                "signal.gain_transport.multipass.compiled_hits",
                "signal.gain_transport.multipass.transitions",
                "signal.gain_transport.multipass.overlap_detected",
                "signal.gain_transport.multipass.overlap_time_by_z_s",
                "signal.gain_transport.multipass.pairwise_spatiotemporal_overlap_fraction_by_z",
            ]
        )
    if executed_model == "coherent_cpa":
        domains["coherent_cpa"] = {
            "backend": executed_model,
            "required": [
                "signal.cpa.maximum_temporal_boundary_energy_fraction",
                "signal.cpa.maximum_output_spectral_boundary_energy_fraction",
                "signal.cpa.maximum_cumulative_B_integral_rad",
                "signal.cpa.local_operator.substep_cap_active",
                "signal.cpa.local_operator.maximum_substeps_used",
                "signal.cpa.transient_gain.enabled",
                "signal.cpa.strong_field.enabled",
                "signal.cpa.sampling.adequate",
                "signal.cpa.fidelity_mode",
            ],
            "required_any": [],
        }
        transient_gain = signal.get("cpa", {}).get("transient_gain", {}) or {}
        if bool(transient_gain.get("enabled", False)):
            domains["coherent_cpa"]["required"].extend(
                [
                    "signal.cpa.transient_gain.maximum_resynthesis_energy_error",
                    "signal.cpa.transient_gain.depletion_target_met",
                    "signal.cpa.transient_gain.maximum_reservoir_depletion_per_slice",
                    "signal.cpa.transient_gain.maximum_depletion_per_slice_target",
                ]
            )
            if float(
                transient_gain.get(
                    "maximum_bypassed_output_energy_fraction", 0.0
                )
                or 0.0
            ) > 0.0:
                domains["coherent_cpa"]["required"].extend(
                    [
                        "signal.cpa.transient_gain.inactive_windows_bypassed",
                        "signal.cpa.transient_gain.maximum_estimated_bypassed_output_energy_fraction",
                        "signal.cpa.transient_gain.maximum_local_estimated_bypassed_output_energy_fraction",
                    ]
                )
        strong_field = signal.get("cpa", {}).get("strong_field", {}) or {}
        if bool(strong_field.get("enabled", False)):
            domains["coherent_cpa"]["required"].extend(
                [
                    "signal.cpa.strong_field.maximum_fractional_loss_per_step",
                    "signal.cpa.strong_field.mechanism",
                    "signal.cpa.strong_field.multiphoton_order",
                    "signal.cpa.strong_field.beta_SI",
                    "signal.cpa.strong_field.provenance",
                    "signal.cpa.strong_field.free_carrier_model",
                    "signal.cpa.strong_field.heat_deposition_contract",
                    "signal.cpa.strong_field.deposit_fraction_as_heat",
                ]
            )
        if str(signal.get("cpa", {}).get("input_mode", "")) == "phase_retrieval":
            domains["coherent_cpa"]["required"].append(
                "signal.cpa.phase_retrieval.converged"
            )
        if str(signal.get("cpa", {}).get("fidelity_mode", "")) in {
            "axisymmetric_kerr",
            "axisymmetric_transient_gain",
        }:
            domains["coherent_cpa"]["required"].extend(
                [
                    "signal.cpa.axisymmetric_reduction.active",
                    "signal.cpa.axisymmetric_reduction.minimum_input_projection_energy_fraction",
                    "signal.cpa.axisymmetric_reduction.total_input_projection_energy_fraction",
                    "signal.cpa.axisymmetric_reduction.maximum_energy_weighted_outer_annulus_fraction",
                    "signal.cpa.axisymmetric_reduction.maximum_population_azimuthal_residual",
                    "signal.cpa.axisymmetric_reduction.maximum_delta_n_azimuthal_residual",
                    "signal.cpa.axisymmetric_reduction.maximum_diffraction_norm_correction",
                    "signal.cpa.axisymmetric_reduction.maximum_gain_bin_relative_frequency_span",
                    "signal.cpa.axisymmetric_reduction.maximum_gain_reconstruction_normalization_correction",
                    "signal.cpa.axisymmetric_reduction.population_mapping_relative_error",
                    "signal.cpa.axisymmetric_reduction.free_carrier_mapping_relative_error",
                ]
            )
    cpa_model = str((signal.get("cpa") or {}).get("model", ""))
    if (
        executed_model == "commercial_app_fast"
        and cpa_model
        in {
            "commercial_app_effective_area_1d_spm",
            "commercial_app_effective_area_1d_strong_field",
        }
    ):
        domains["commercial_app_effective_area_spm"] = {
            "backend": cpa_model,
            "required": [
                "signal.cpa.sampling.adequate",
                "signal.cpa.maximum_spectral_support_leakage_fraction",
                "signal.cpa.maximum_spectral_sync_error_fraction",
                "signal.cpa.maximum_cumulative_B_integral_rad",
                "signal.cpa.spatial_temporal_coupling_resolved",
                "signal.cpa.spectral_feedback_grid.mode",
            ],
            "required_any": [],
        }
        strong_field = signal.get("cpa", {}).get("strong_field", {}) or {}
        if bool(strong_field.get("enabled", False)):
            domains["commercial_app_effective_area_spm"]["required"].extend(
                [
                    "signal.cpa.strong_field.maximum_fractional_loss_per_step",
                    "signal.cpa.strong_field.mechanism",
                    "signal.cpa.strong_field.multiphoton_order",
                    "signal.cpa.strong_field.beta_SI",
                    "signal.cpa.strong_field.provenance",
                    "signal.cpa.strong_field.free_carrier_model",
                    "signal.cpa.strong_field.spatial_closure",
                    "signal.cpa.strong_field.carrier_periodic_closure",
                    "signal.cpa.strong_field.carrier_periodic_solve_count",
                    "signal.cpa.strong_field.carrier_periodic_relative_tolerance",
                    "signal.cpa.strong_field.maximum_carrier_periodic_relative_residual",
                    "signal.cpa.strong_field.carrier_periodic_converged",
                    "signal.cpa.strong_field.heat_deposition_contract",
                    "signal.cpa.strong_field.deposit_fraction_as_heat",
                ]
            )

    gain_transport = signal.get("gain_transport", {}) or {}
    if str(gain_transport.get("backend", "")) == "josa_transport":
        domains["josa"] = {
            "backend": "josa_transport",
            "required": [
                "signal.gain_transport.limited_population_updates",
                "signal.gain_transport.maximum_abs_gain_step",
                "signal.gain_transport.recommended_maximum_gain_step",
            ],
            "required_any": [],
        }

    if bool(grid_coupling.get("enabled", False)):
        domains["grid_coupling"] = {
            "backend": str(grid_coupling.get("remap_backend", "unknown")),
            "required": [
                "grid_coupling.remap_backend",
                "grid_coupling.ledger_remap_method",
                "grid_coupling.seed",
                "grid_coupling.pump",
                "signal.grid_coupling.population_projection_correction_relative",
            ],
            "required_any": [],
        }
        signal_grid = signal.get("grid_coupling", {}) or {}
        if (
            str(signal_grid.get("free_carrier_state_remap", "not_applicable"))
            != "not_applicable"
        ):
            domains["grid_coupling"]["required"].extend(
                [
                    "signal.grid_coupling.free_carrier_state_remap",
                    "signal.grid_coupling.free_carrier_change_relative_error",
                ]
            )

    if bool(ase.get("requested", ase.get("enabled", False))):
        domains["ase"] = {
            "backend": str(ase.get("model", "unknown")),
            "required": ["ase.requested", "ase.executed"],
            "required_any": [],
        }
        if bool(ase.get("enabled", False)):
            domains["ase"]["required"].extend(
                [
                    "ase.boundary_converged",
                    "ase.energy_ledger_W.relative_residual",
                    "ase.radiative_lifetime_source",
                    "ase.intensity_clamp_active",
                    "ase.positive_cell_exponent_clipping_active",
                ]
            )
            if str(ase.get("model", "")) == "angular_discrete_ordinates":
                domains["ase"]["required"].extend(
                    [
                        "ase.dominant_parasitic_boundary_eigenvalue",
                        "ase.parasitic_eigenmode_converged",
                        "ase.parasitic_eigenpair_residual",
                        "ase.reflected_boundary_flux_relative_residual",
                        "ase.embedded_boundary_resolved",
                    ]
                )

    if str(thermal.get("heat_source_model", "")) == "event_energy_ledger":
        domains["event_energy"] = {
            "backend": "event_energy_ledger",
            "required": [
                "thermal.event_energy_ledger.enabled",
                "thermal.event_energy_ledger.complete_partitions",
                "thermal.event_energy_ledger.full_identity_residual_relative",
                "thermal.event_energy_ledger.unresolved_destination_fraction",
                "thermal.event_energy_ledger.temporal_applicability",
                "thermal.event_energy_ledger.level_energy_shape_valid",
                "thermal.event_energy_ledger.spectral_diffusion_energy_exchange",
            ],
            "required_any": [],
        }

    if str(thermal.get("heat_source_model", "")) == "pump_component_ledger":
        domains.setdefault(
            "thermal_heat_source",
            {
                "backend": "pump_component_ledger",
                "required": [],
                "required_any": [],
            },
        )["required"].append(
            "thermal.pump_component_ledger_applicability.applicable"
        )

    nonlinear_solver = population.get("nonlinear_solver", {}) or {}
    if bool(nonlinear_solver.get("enabled", False)):
        domains["population"]["required"].extend(
            [
                "population.nonlinear_solver.cap_limited_calls",
                "population.nonlinear_solver.maximum_estimated_fractional_depletion_step",
                "population.nonlinear_solver.configured_maximum_fractional_depletion_per_substep",
            ]
        )
    signal_event = population.get("signal_event_limiter", {}) or {}
    if bool(signal_event.get("enabled", False)):
        domains["population"]["required"].extend(
            [
                "population.signal_event_limiter.maximum_requested_applied_residual_relative",
                "population.signal_event_limiter.maximum_channel_requested_applied_residual_relative",
                "population.signal_event_limiter.minimum_limiter_scale",
                "population.signal_event_limiter.limited_cell_updates",
                "population.signal_event_limiter.maximum_population_change_fraction_per_update",
                "population.signal_event_limiter.packet_count_used",
            ]
        )

    if str(stress.get("model", "none")) in {
        "thermoelastic_fem",
        "anisotropic_thermoelastic_fem",
    }:
        domains["stress_fem"] = {
            "backend": str(stress.get("model")),
            "required": [
                "stress.equilibrium_residual_relative",
                "stress.active_elements",
                "stress.active_nodes",
                "stress.linear_iterations",
            ],
            "required_any": [],
        }

    registrations: list[tuple[str, str]] = [
        ("pump", str(pump.get("propagation", "unknown"))),
        ("signal", executed_model or "unknown"),
        ("population", str(population.get("dynamics", "unknown"))),
        ("stress", str(stress.get("model", "unknown"))),
        ("ase", str(ase.get("model", "unknown"))),
    ]
    if bool(thermal.get("enabled", False)):
        registrations.extend(
            [
                ("thermal", str(thermal.get("linear_solver", "unknown"))),
                (
                    "thermal_heat_source",
                    str(thermal.get("heat_source_model", "unknown")),
                ),
            ]
        )
    if bool(grid_coupling.get("enabled", False)):
        registrations.append(
            (
                "grid_coupling",
                str(grid_coupling.get("remap_backend", "unknown")),
            )
        )
    unregistered_backends = [
        {"domain": domain, "backend": backend}
        for domain, backend in registrations
        if backend not in _REGISTERED_STRICT_BACKENDS[domain]
    ]

    return {
        "schema_version": STRICT_DIAGNOSTIC_CONTRACT_SCHEMA_VERSION,
        "domains": domains,
        "unregistered_backends": unregistered_backends,
    }


def build_validation_report(
    summary: dict[str, Any],
    *,
    mode: str,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = settings or {}
    if mode not in VALIDATION_MODES:
        raise ValueError(f"validation mode must be one of {sorted(VALIDATION_MODES)}")
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    checks: dict[str, dict[str, Any]] = {}

    def check(
        name: str,
        passed: bool | None,
        *,
        value: Any = None,
        limit: Any = None,
        message: str,
        severity: str = "blocking",
    ) -> None:
        if severity not in {"blocking", "warning"}:
            raise ValueError("validation check severity must be blocking or warning")
        status = "not_applicable" if passed is None else ("passed" if passed else "failed")
        checks[name] = {
            "status": status,
            "severity": severity,
            "value": value,
            "limit": limit,
        }
        if passed is False:
            issue = {
                "check": name,
                "severity": severity,
                "message": message,
                "value": value,
                "limit": limit,
            }
            (failures if severity == "blocking" else warnings).append(issue)

    def diagnostic_path_present(path: str) -> bool:
        current: Any = summary
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        return current is not None

    diagnostic_contract = summary.get("strict_diagnostic_contract")
    contract_required = bool(
        mode == "strict"
        and (
            diagnostic_contract is not None
            or "configuration_provenance" in summary
        )
    )
    if contract_required:
        domains = (
            diagnostic_contract.get("domains")
            if isinstance(diagnostic_contract, dict)
            else None
        )
        schema_valid = bool(
            isinstance(diagnostic_contract, dict)
            and diagnostic_contract.get("schema_version")
            == STRICT_DIAGNOSTIC_CONTRACT_SCHEMA_VERSION
            and isinstance(domains, dict)
            and isinstance(
                diagnostic_contract.get("unregistered_backends"), list
            )
            and all(
                isinstance(item, dict)
                and isinstance(item.get("domain"), str)
                and isinstance(item.get("backend"), str)
                for item in diagnostic_contract.get(
                    "unregistered_backends", []
                )
            )
            and all(
                isinstance(domain, str)
                and isinstance(contract, dict)
                and isinstance(contract.get("required", []), list)
                and all(
                    isinstance(path, str)
                    for path in contract.get("required", [])
                )
                and isinstance(contract.get("required_any", []), list)
                and all(
                    isinstance(group, list)
                    and all(isinstance(path, str) for path in group)
                    for group in contract.get("required_any", [])
                )
                for domain, contract in domains.items()
            )
        )
        check(
            "required_diagnostics_contract_schema",
            schema_valid,
            value=(
                diagnostic_contract.get("schema_version")
                if isinstance(diagnostic_contract, dict)
                else None
            ),
            limit=STRICT_DIAGNOSTIC_CONTRACT_SCHEMA_VERSION,
            message="The strict backend diagnostic contract is missing or has an unsupported schema.",
        )
        if schema_valid:
            unregistered_backends = list(
                diagnostic_contract.get("unregistered_backends", [])
            )
            check(
                "required_diagnostics_registered_backends",
                not unregistered_backends,
                value=unregistered_backends,
                limit=[],
                message=(
                    "At least one executed backend has no registered strict "
                    "diagnostic contract."
                ),
            )
            for domain, contract in domains.items():
                required = list(contract.get("required", []))
                required_any = list(contract.get("required_any", []))
                missing = [
                    path for path in required if not diagnostic_path_present(path)
                ]
                missing_alternatives = [
                    list(group)
                    for group in required_any
                    if not any(diagnostic_path_present(path) for path in group)
                ]
                check(
                    f"required_diagnostics_{domain}",
                    not missing and not missing_alternatives,
                    value={
                        "backend": contract.get("backend"),
                        "missing": missing,
                        "missing_any_of": missing_alternatives,
                    },
                    limit={"missing": [], "missing_any_of": []},
                    message=(
                        f"The executed {domain} backend did not emit all evidence "
                        "required for strict validation."
                    ),
                )

    pump = summary.get("pump", {})
    pump_tolerance = float(cfg.get("pump_power_balance_relative_tolerance", 1e-8))
    pump_residuals = [
        float(value)
        for value in (
            pump.get("crystal_power_balance_residual_relative"),
            pump.get("source_to_exit_balance_residual_relative"),
        )
        if value is not None
    ]
    maximum_pump_residual = max(pump_residuals, default=None)
    check(
        "pump_power_balance",
        None if maximum_pump_residual is None else maximum_pump_residual <= pump_tolerance,
        value=maximum_pump_residual,
        limit=pump_tolerance,
        message="Pump power ledger residual exceeds the strict tolerance.",
    )
    pump_clipping = bool(
        pump.get("positive_pump_exponent_clipping_active", False)
        or pump.get("positive_pump_ray_exponent_clipping_active", False)
    )
    check(
        "pump_exponent_clipping",
        not pump_clipping,
        value=pump_clipping,
        limit=False,
        message="Pump positive-gain exponent clipping was active.",
    )
    maximum_absorption_step = pump.get("maximum_pump_absorption_optical_depth")
    check(
        "pump_absorption_step_resolution",
        (
            None
            if maximum_absorption_step is None
            else float(maximum_absorption_step)
            <= float(cfg.get("maximum_pump_absorption_optical_depth", 1.0))
        ),
        value=maximum_absorption_step,
        limit=float(cfg.get("maximum_pump_absorption_optical_depth", 1.0)),
        message="Pump alpha*ds exceeds the configured resolution contract.",
    )
    hybrid_entrance_capture = pump.get("gain_entrance_capture_fraction")
    minimum_hybrid_entrance_capture = float(
        cfg.get(
            "minimum_hybrid_pump_gain_entrance_capture_fraction", 0.98
        )
    )
    hybrid_pump_active = str(pump.get("propagation", "")) == "hybrid_abcd_fft"
    check(
        "hybrid_pump_gain_entrance_window_capture",
        (
            None
            if not hybrid_pump_active
            else hybrid_entrance_capture is not None
            and float(hybrid_entrance_capture)
            >= minimum_hybrid_entrance_capture
        ),
        value=hybrid_entrance_capture,
        limit=f">= {minimum_hybrid_entrance_capture}",
        message=(
            "The analytic-to-FFT Pump boundary does not fit inside the gain "
            "optical window; omitted Pump power changes the modeled operating point."
        ),
    )

    seed_ledger = summary.get("energy_ledger", {}).get("seed_pulse_energy_J")
    seed_unresolved = (
        None
        if not seed_ledger
        else seed_ledger.get("unresolved_external_interface_and_routing_term")
    )
    seed_scale = (
        None
        if not seed_ledger
        else max(
            abs(float(seed_ledger.get("input", 0.0))),
            abs(float(seed_ledger.get("output", 0.0))),
            abs(float(seed_ledger.get("active_transition_transfer", 0.0))),
            1e-300,
        )
    )
    seed_negative_unresolved_relative = (
        None
        if seed_unresolved is None or seed_scale is None
        else max(-float(seed_unresolved), 0.0) / seed_scale
    )
    seed_tolerance = float(
        cfg.get("seed_energy_balance_relative_tolerance", 1e-8)
    )
    require_named_external_losses = bool(
        cfg.get("require_named_external_losses", True)
    )
    seed_contract_residual_relative = (
        None
        if seed_unresolved is None or seed_scale is None
        else (
            abs(float(seed_unresolved)) / seed_scale
            if require_named_external_losses
            else seed_negative_unresolved_relative
        )
    )
    check(
        "seed_energy_balance",
        (
            None
            if seed_contract_residual_relative is None
            else seed_contract_residual_relative <= seed_tolerance
        ),
        value=seed_contract_residual_relative,
        limit=seed_tolerance,
        message=(
            "Seed energy is not closed by active transfer, named passive/strong-field/aperture "
            "losses, and named external routing terms."
        ),
    )
    check(
        "no_unaccounted_energy_creation",
        (
            None
            if seed_negative_unresolved_relative is None
            else seed_negative_unresolved_relative <= seed_tolerance
        ),
        value=seed_negative_unresolved_relative,
        limit=seed_tolerance,
        message="Seed ledger indicates unaccounted energy creation.",
    )
    check(
        "closed_seed_energy_ledger",
        (
            None
            if not require_named_external_losses
            or seed_unresolved is None
            or seed_scale is None
            else abs(float(seed_unresolved)) / seed_scale <= seed_tolerance
        ),
        value=(
            None
            if seed_unresolved is None or seed_scale is None
            else abs(float(seed_unresolved)) / seed_scale
        ),
        limit=seed_tolerance,
        message=(
            "Seed first-law ledger is not closed in both directions by named terms."
        ),
    )

    population = summary.get("population", {})
    population_tolerance = float(
        cfg.get("population_conservation_relative_tolerance", 1e-10)
    )
    conservation = float(population.get("conservation_max_relative", 0.0))
    check(
        "population_conservation",
        conservation <= population_tolerance,
        value=conservation,
        limit=population_tolerance,
        message="Local ion-number conservation exceeds the strict tolerance.",
    )
    ensemble_inventory = population.get("ensemble_inventory") or {}
    ensemble_inventory_error = ensemble_inventory.get(
        "maximum_local_inventory_error_relative"
    )
    ensemble_inventory_limit = float(
        cfg.get("ensemble_inventory_relative_tolerance", 1e-10)
    )
    check(
        "ensemble_group_inventory",
        (
            None
            if not bool(ensemble_inventory.get("applicable", False))
            else ensemble_inventory_error is not None
            and float(ensemble_inventory_error) <= ensemble_inventory_limit
        ),
        value=ensemble_inventory_error,
        limit=ensemble_inventory_limit,
        message=(
            "A zero-diffusion ensemble group does not retain its configured "
            "local ion inventory."
        ),
    )
    correction = population.get("numerical_population_correction", {})
    maximum_correction = max(
        float(correction.get("maximum_negative_population_fraction", 0.0)),
        float(
            correction.get(
                "maximum_conservation_error_before_renormalization_relative",
                0.0,
            )
        ),
        float(
            correction.get(
                "maximum_component_inventory_error_before_renormalization_relative",
                0.0,
            )
        ),
    )
    correction_tolerance = float(
        cfg.get("population_correction_relative_tolerance", 1e-8)
    )
    check(
        "population_projection_correction",
        maximum_correction <= correction_tolerance,
        value=maximum_correction,
        limit=correction_tolerance,
        message="Population clipping/renormalization correction is too large.",
    )

    cycle_dynamics = str(population.get("dynamics", ""))
    cycle_timing = population.get("cycle_timing", {}) or {}
    cycle_timing_active = bool(cycle_timing) and cycle_dynamics in {
        "transient_pulse_train",
        "periodic_steady_state",
        "reduced_periodic_steady_state",
    }
    cycle_time_tolerance = float(
        cfg.get("cycle_time_closure_relative_tolerance", 1e-12)
    )

    def cycle_clock_evidence(
        dark_key: str,
        resolved_key: str,
        residual_key: str,
    ) -> tuple[bool, dict[str, Any]]:
        raw_period = cycle_timing.get("repetition_period_s")
        raw_dark = cycle_timing.get(dark_key)
        raw_resolved = cycle_timing.get(resolved_key)
        raw_reported = cycle_timing.get(residual_key)
        try:
            period = float(raw_period)
            dark = float(raw_dark)
            resolved = float(raw_resolved)
            reported = float(raw_reported)
        except (TypeError, ValueError):
            return False, {
                "period_s": raw_period,
                "dark_duration_s": raw_dark,
                "seed_resolved_elapsed_time_s": raw_resolved,
                "reported_residual_s": raw_reported,
                "recomputed_residual_s": None,
                "maximum_relative_residual": None,
            }
        finite_components = all(
            math.isfinite(value) for value in (period, dark, resolved, reported)
        )
        physical_components = bool(period > 0.0 and dark >= 0.0 and resolved >= 0.0)
        recomputed = period - dark - resolved
        scale = max(abs(period), 1e-300)
        maximum_relative = max(
            abs(recomputed),
            abs(reported),
            abs(recomputed - reported),
        ) / scale
        evidence = {
            "period_s": period,
            "dark_duration_s": dark,
            "seed_resolved_elapsed_time_s": resolved,
            "reported_residual_s": reported,
            "recomputed_residual_s": recomputed,
            "maximum_relative_residual": maximum_relative,
        }
        return bool(
            finite_components
            and physical_components
            and maximum_relative <= cycle_time_tolerance
        ), evidence

    population_clock_passed: bool | None = None
    population_clock_evidence: dict[str, Any] | None = None
    if cycle_timing_active:
        population_clock_passed, population_clock_evidence = cycle_clock_evidence(
            "population_dark_duration_s",
            "population_seed_resolved_elapsed_time_s",
            "population_time_closure_residual_s",
        )
    check(
        "population_cycle_time_closure",
        population_clock_passed,
        value=population_clock_evidence,
        limit=cycle_time_tolerance,
        message=(
            "Ion-population time in one pulse period is not closed by the "
            "reported dark and Seed-resolved intervals."
        ),
    )

    free_carrier_clock_active = bool(
        cycle_timing_active
        and cycle_timing.get("free_carrier_seed_resolved_elapsed_time_s") is not None
    )
    free_carrier_clock_passed: bool | None = None
    free_carrier_clock_evidence: dict[str, Any] | None = None
    if free_carrier_clock_active:
        free_carrier_clock_passed, free_carrier_clock_evidence = (
            cycle_clock_evidence(
                "free_carrier_dark_duration_s",
                "free_carrier_seed_resolved_elapsed_time_s",
                "free_carrier_time_closure_residual_s",
            )
        )
    check(
        "free_carrier_cycle_time_closure",
        free_carrier_clock_passed,
        value=free_carrier_clock_evidence,
        limit=cycle_time_tolerance,
        message=(
            "Free-carrier time in one coherent pulse period is not closed by "
            "the reported dark and Seed-resolved intervals."
        ),
    )

    signal = summary.get("signal", {})
    requested_signal_model = str(signal.get("requested_model", ""))
    executed_signal_model = str(
        signal.get("executed_model", signal.get("model", ""))
    )
    backend_specialization = signal.get("backend_specialization")
    if "fallback_active" in signal:
        # The solver owns this classification.  Some intentional closures,
        # notably the CW cycle-averaged steady probe, execute under a more
        # specific model label than the requested public backend without
        # falling back.  Only infer from names for legacy summaries that do
        # not carry the explicit flag.
        signal_fallback_active = bool(signal["fallback_active"])
    else:
        signal_fallback_active = bool(
            executed_signal_model.endswith("_fallback")
            or (
                backend_specialization is None
                and requested_signal_model
                and executed_signal_model
                and requested_signal_model != executed_signal_model
            )
        )
    check(
        "signal_backend_fallback",
        not signal_fallback_active,
        value={
            "requested": requested_signal_model,
            "executed": executed_signal_model,
            "fallback_active": signal_fallback_active,
        },
        limit={"fallback_active": False},
        message=(
            "The requested Seed backend did not execute. A fallback result is "
            "retained for diagnosis but must not be labeled quantitative."
        ),
    )
    gain_transport = signal.get("gain_transport", {}) or {}
    commercial_app_thermal_lens = gain_transport.get("thermal_lens", {}) or {}
    spectral_polar = executed_signal_model == "spectral_time" and gain_transport.get("spatial_backend") == "axisymmetric"
    for label, key, limit, complement in (
        ("input_projection", "entrance_projection_retained_energy_fraction", float(cfg.get("maximum_axisymmetric_projection_loss_fraction", 1e-3)), True),
        ("radial_boundary", "maximum_energy_weighted_boundary_fraction", float(cfg.get("maximum_axisymmetric_boundary_energy_fraction", 1e-4)), False),
        ("dynamic_projection", "maximum_dynamic_field_omitted_energy_fraction", float(cfg.get("maximum_axisymmetric_projection_loss_fraction", 1e-3)), False),
        ("population_mapping", "population_mapping_relative_error", 1e-8, False),
        ("population_azimuthal", "maximum_population_azimuthal_residual", float(cfg.get("maximum_axisymmetric_azimuthal_residual", .05)), False),
        ("thermal_azimuthal", "maximum_delta_n_azimuthal_residual", float(cfg.get("maximum_axisymmetric_azimuthal_residual", .05)), False),
    ):
        value = gain_transport.get(key)
        residual = None if value is None else (1.0 - float(value) if complement else float(value))
        check("spectral_time_axisymmetric_" + label,
              None if not spectral_polar else residual is not None and residual <= limit,
              value=residual, limit=limit,
              message="Spectral Time radial transport did not meet its projection/boundary/material tolerance.")
    commercial_app_polar = executed_signal_model == "commercial_app_fast" and gain_transport.get("spatial_backend") == "axisymmetric"
    for label, key, limit, complement in (
        ("input_projection", "entrance_projection_retained_energy_fraction", float(cfg.get("maximum_axisymmetric_projection_loss_fraction", 1e-3)), True),
        ("radial_boundary", "maximum_energy_weighted_boundary_fraction", float(cfg.get("maximum_axisymmetric_boundary_energy_fraction", 1e-4)), False),
        ("population_mapping", "population_mapping_relative_error", 1e-8, False),
    ):
        value = gain_transport.get(key)
        residual = None if value is None else (1.0 - float(value) if complement else float(value))
        check("commercial_app_axisymmetric_" + label,
              None if not commercial_app_polar else residual is not None and residual <= limit,
              value=residual, limit=limit,
              message="Commercial App polar quadrature did not meet its projection/boundary/material mapping tolerance.")
    gain_redistribution = gain_transport.get("override_gain_mapping_redistribution_fraction")
    gain_redistribution_limit = float(cfg.get("maximum_axisymmetric_population_positivity_redistribution_fraction", .05))
    check("commercial_app_axisymmetric_gain_redistribution",
          None if not commercial_app_polar or gain_redistribution is None else float(gain_redistribution) <= gain_redistribution_limit,
          value=gain_redistribution, limit=gain_redistribution_limit,
          message="Commercial App polar gain-reservoir remapping required excessive spatial redistribution; refine the radial/optical grid.")
    check("spectral_time_axisymmetric_gain_redistribution",
          None if not spectral_polar or gain_redistribution is None else float(gain_redistribution) <= gain_redistribution_limit,
          value=gain_redistribution, limit=gain_redistribution_limit,
          message="Spectral Time polar gain-reservoir remapping required excessive spatial redistribution; refine the radial/optical grid.")
    commercial_app_local_parabolic = bool(
        executed_signal_model == "commercial_app_fast"
        and str(commercial_app_thermal_lens.get("model", "")) == "local_parabolic"
    )
    commercial_app_axis_aligned_q_applicable = commercial_app_thermal_lens.get(
        "axis_aligned_q_applicable"
    )
    check(
        "commercial_app_axis_aligned_q_thermal_lens",
        (
            None
            if not commercial_app_local_parabolic
            else bool(commercial_app_axis_aligned_q_applicable)
        ),
        value={
            "applicable": commercial_app_axis_aligned_q_applicable,
            "maximum_slice_axis_coupling_ratio": commercial_app_thermal_lens.get(
                "maximum_slice_axis_coupling_ratio"
            ),
            "integrated_axis_coupling_ratio": commercial_app_thermal_lens.get(
                "integrated_axis_coupling_ratio"
            ),
        },
        limit={
            "maximum_axis_coupling_ratio": commercial_app_thermal_lens.get(
                "maximum_axis_coupling_ratio"
            )
        },
        message=(
            "Commercial App stores independent axis-aligned qx/qy states, but the fitted "
            "thermal lens contains significant xy curvature. Use a sampled-field "
            "backend for quantitative rotated-astigmatism transport."
        ),
    )
    seed_entrance_capture = signal.get("gain_entrance_capture_fraction")
    minimum_seed_entrance_capture = float(
        cfg.get(
            "minimum_hybrid_seed_gain_entrance_capture_fraction", 0.98
        )
    )
    # ``abcd`` is the backwards-compatible Seed spelling for the same
    # analytic external boundary followed by sampled in-crystal propagation.
    hybrid_seed_active = str(signal.get("propagation", "")) in {
        "abcd",
        "hybrid_abcd_fft",
    }
    check(
        "hybrid_seed_gain_entrance_window_capture",
        (
            None
            if not hybrid_seed_active
            else seed_entrance_capture is not None
            and float(seed_entrance_capture) >= minimum_seed_entrance_capture
        ),
        value=seed_entrance_capture,
        limit=f">= {minimum_seed_entrance_capture}",
        message=(
            "The analytic-to-FFT Seed boundary does not fit inside the gain "
            "optical window; omitted Seed energy changes gain saturation, "
            "extraction, nonlinear phase, and deposited heat."
        ),
    )
    signal_grid = signal.get("grid_coupling") or {}
    root_grid = summary.get("grid_coupling", {}) or {}
    projection_candidates = [
        item.get("population_projection_correction_relative")
        for item in (
            signal_grid,
            root_grid,
            root_grid.get("seed", {}) or {},
            root_grid.get("pump", {}) or {},
        )
        if isinstance(item, dict)
    ]
    projection_values = [
        float(value) for value in projection_candidates if value is not None
    ]
    projection = max(projection_values, default=None)
    projection_tolerance = float(
        cfg.get("grid_projection_relative_tolerance", 1e-8)
    )
    check(
        "dual_grid_population_projection",
        None if projection is None else float(projection) <= projection_tolerance,
        value=projection,
        limit=projection_tolerance,
        message="Dual-grid population projection is too large relative to local dopant density.",
    )
    carrier_remap = str(
        signal_grid.get("free_carrier_state_remap", "not_applicable")
    )
    carrier_change_error = signal_grid.get(
        "free_carrier_change_relative_error"
    )
    check(
        "dual_grid_free_carrier_inventory",
        (
            None
            if carrier_remap == "not_applicable"
            else carrier_change_error is not None
            and float(carrier_change_error) <= 1e-10
        ),
        value=carrier_change_error,
        limit=1e-10,
        message=(
            "Dual-grid free-carrier projection did not preserve the optical "
            "net carrier-count change."
        ),
    )
    gain_transport = signal.get("gain_transport") or {}
    seed_clipping = bool(
        gain_transport.get("positive_cell_exponent_clipping_active", False)
        or gain_transport.get("positive_gain_exponent_clipping_active", False)
    )
    check(
        "seed_exponent_clipping",
        not seed_clipping,
        value=seed_clipping,
        limit=False,
        message="Seed positive-gain exponent clipping was active.",
    )
    physics_validity = summary.get("physics_validity", {}) or {}
    validity_status = str(physics_validity.get("status", ""))
    check(
        "signal_model_applicability",
        not validity_status.startswith("review_required"),
        value=validity_status,
        limit="not review_required",
        message="The selected Seed model does not resolve a material pulse/lifetime process.",
    )
    phase_insensitive_phase = bool(
        signal.get("phase_insensitive_spectral_phase_requested", False)
    )
    check(
        "phase_insensitive_dispersion_applicability",
        not phase_insensitive_phase,
        value=phase_insensitive_phase,
        limit=False,
        message=(
            "A phase-insensitive Seed backend cannot predict coherent GDD/TOD pulse reshaping."
        ),
    )
    if gain_transport.get("backend") == "josa_transport":
        group_delay = gain_transport.get("group_delay_walkoff", {}) or {}
        phase_closure = group_delay.get("phase_closure", {}) or {}
        if gain_transport.get("mode") == "full_3d_pc":
            strict_phase_closure = phase_closure.get(
                "strict_incoherent_diffraction_closure"
            )
            check(
                "josa_group_delay_phase_closure",
                strict_phase_closure is True,
                value=strict_phase_closure,
                limit=True,
                message=(
                    "JOSA group-delay remapping mixed incoherent source time bins into one complex field before FFT diffraction; "
                    "the result is energy-conservative but not strict phase closure."
                ),
            )
        limited_updates_raw = gain_transport.get("limited_population_updates")
        limited_updates = (
            None if limited_updates_raw is None else int(limited_updates_raw)
        )
        check(
            "josa_population_limiter",
            limited_updates is not None and limited_updates == 0,
            value=limited_updates,
            limit=0,
            message=(
                "JOSA did not report a valid population-limiter diagnostic, or "
                "required population limiting; refine time/z resolution."
            ),
        )
        maximum_gain_step_raw = gain_transport.get("maximum_abs_gain_step")
        maximum_gain_step = (
            None
            if maximum_gain_step_raw is None
            else float(maximum_gain_step_raw)
        )
        recommended_gain_step = float(
            gain_transport.get("recommended_maximum_gain_step", 0.05)
        )
        check(
            "josa_gain_step_resolution",
            maximum_gain_step is not None
            and maximum_gain_step <= recommended_gain_step,
            value=maximum_gain_step,
            limit=recommended_gain_step,
            message=(
                "JOSA did not report |g dz|, or it exceeds its configured "
                "resolution contract."
            ),
        )

    multiphysics = summary.get("multiphysics", {})
    convergence_items = {
        "population_convergence": population.get("converged"),
        "thermal_convergence": (
            summary.get("thermal", {}).get("converged")
            if summary.get("thermal", {}).get("enabled", False)
            else None
        ),
        "multiphysics_convergence": (
            multiphysics.get("converged")
            if multiphysics.get("converged") is not None
            else None
        ),
    }
    for name, converged in convergence_items.items():
        check(
            name,
            None if converged is None else bool(converged),
            value=converged,
            limit=True,
            message=f"{name.replace('_', ' ')} did not converge.",
        )

    thermal = summary.get("thermal", {})
    thermal_balance = (
        thermal.get("energy_balance_relative_residual")
        if thermal.get("enabled", False)
        else None
    )
    thermal_balance_tolerance = float(
        cfg.get("thermal_energy_balance_relative_tolerance", 1e-5)
    )
    check(
        "thermal_energy_balance",
        (
            None
            if thermal_balance is None
            else abs(float(thermal_balance)) <= thermal_balance_tolerance
        ),
        value=thermal_balance,
        limit=thermal_balance_tolerance,
        message="Thermal energy balance exceeds the strict tolerance.",
    )
    event_energy = thermal.get("event_energy_ledger") or {}
    event_enabled = bool(event_energy.get("enabled", False))
    # The event ledger is still useful as a diagnostic product when the
    # thermal solve is disabled.  In that case an incomplete ledger must not
    # invalidate an otherwise quantitative optical result.  Treat summaries
    # that predate the explicit ``thermal.enabled`` marker conservatively.
    thermal_diagnostic_only = thermal.get("enabled") is False
    event_energy_severity = (
        "warning" if thermal_diagnostic_only else "blocking"
    )
    check(
        "event_energy_complete_partitions",
        (
            None
            if not event_enabled
            else bool(event_energy.get("complete_partitions", False))
        ),
        value=(
            None
            if not event_enabled
            else event_energy.get("complete_partitions")
        ),
        limit=True,
        message=(
            "The event-energy ledger contains unresolved decay or nonlinear "
            "event partitions."
        ),
        severity=event_energy_severity,
    )
    event_identity = event_energy.get("full_identity_residual_relative")
    event_identity_limit = float(
        cfg.get("event_energy_identity_relative_tolerance", 1e-10)
    )
    check(
        "event_energy_identity_residual",
        (
            None
            if not event_enabled
            else event_identity is not None
            and float(event_identity) <= event_identity_limit
        ),
        value=event_identity,
        limit=event_identity_limit,
        message="The event-energy algebraic identity exceeds tolerance.",
        severity=event_energy_severity,
    )
    unresolved_fraction = event_energy.get(
        "unresolved_destination_fraction"
    )
    unresolved_limit = float(
        cfg.get("event_energy_unresolved_fraction_tolerance", 0.0)
    )
    check(
        "event_energy_unresolved_fraction",
        (
            None
            if not event_enabled
            else unresolved_fraction is not None
            and float(unresolved_fraction) <= unresolved_limit
        ),
        value=unresolved_fraction,
        limit=unresolved_limit,
        message=(
            "The event-energy ledger assigns more energy to unresolved than "
            "strict mode permits."
        ),
        severity=event_energy_severity,
    )
    temporal_contract = str(
        event_energy.get("temporal_applicability", "")
    )
    time_resolved_population = (
        str(population.get("dynamics", ""))
        in {
            "transient",
            "transient_pulse_train",
            "transient_waveform",
            "periodic_steady_state",
        }
        and int(population.get("iterations", 0)) > 0
    )
    temporal_valid = (
        temporal_contract.startswith("time_bin_")
        if time_resolved_population
        else temporal_contract
        in {
            "stationary_operating_point_snapshot",
            "time_bin_trapezoidal_period_average",
            "time_bin_backward_euler_event_counts_period_average",
        }
    )
    check(
        "event_energy_temporal_applicability",
        None if not event_enabled else temporal_valid,
        value=temporal_contract or None,
        limit=(
            "time-bin integrated"
            if time_resolved_population
            else "stationary or time-bin integrated"
        ),
        message=(
            "The event-energy heat source is not integrated over the resolved "
            "population/Pump time evolution."
        ),
        severity=event_energy_severity,
    )
    diffusion_exchange = event_energy.get(
        "spectral_diffusion_energy_exchange"
    )
    diffusion_exchange_possible = (
        None
        if not isinstance(diffusion_exchange, dict)
        else bool(
            diffusion_exchange.get(
                "stored_energy_exchange_possible", False
            )
        )
    )
    check(
        "event_energy_spectral_diffusion_coverage",
        (
            None
            if not event_enabled
            else diffusion_exchange_possible is not None
            and not diffusion_exchange_possible
        ),
        value=diffusion_exchange,
        limit="no unmodeled stored-energy exchange",
        message=(
            "Event-energy accounting has no spectral-diffusion bath, but the "
            "configured ensemble diffusion can change modeled stored energy."
        ),
        severity=event_energy_severity,
    )
    seed_relaxation = {
        "intrapulse": bool(
            signal.get("intrapulse_relaxation_enabled", False)
        ),
        "between_pass": bool(
            signal.get("between_pass_population_relaxation", False)
        ),
    }
    check(
        "event_energy_seed_relaxation_coverage",
        (
            None
            if not event_enabled
            else not any(seed_relaxation.values())
        ),
        value=seed_relaxation,
        limit={"intrapulse": False, "between_pass": False},
        message=(
            "Seed-side dark population relaxation changed the state without "
            "being accumulated into the event-energy destination ledger."
        ),
        severity=event_energy_severity,
    )
    level_shape_valid = event_energy.get("level_energy_shape_valid")
    check(
        "event_energy_level_energy_shape",
        (
            None
            if not event_enabled
            else bool(level_shape_valid)
        ),
        value=level_shape_valid,
        limit=True,
        message=(
            "The event-energy level vector does not match the expanded level "
            "network."
        ),
        severity=event_energy_severity,
    )

    ase = summary.get("ase", {})
    ase_requested = bool(ase.get("requested", ase.get("enabled", False)))
    ase_executed = bool(ase.get("executed", ase.get("enabled", False)))
    check(
        "ase_requested_execution",
        None if not ase_requested else ase_executed,
        value={
            "requested": ase_requested,
            "executed": ase_executed,
            "disabled_reason": ase.get("disabled_reason"),
        },
        limit={"requested_implies_executed": True},
        message=(
            "ASE was requested but did not execute. A reduced run without ASE "
            "cannot satisfy the requested operating-point contract."
        ),
    )
    ase_interface = event_energy.get("ase_source_interface") or {}
    ase_interface_limit = float(
        cfg.get("event_ase_interface_relative_tolerance", 0.05)
    )
    ase_interface_required = bool(
        event_enabled and ase.get("enabled", False)
    )
    ase_interface_residual = ase_interface.get("relative_mismatch")
    check(
        "event_energy_ase_source_interface",
        (
            None
            if not ase_interface_required
            else bool(ase_interface.get("applicable", False))
            and ase_interface_residual is not None
            and float(ase_interface_residual) <= ase_interface_limit
        ),
        value=ase_interface_residual,
        limit=ase_interface_limit,
        message=(
            "The event-energy ase_source destination does not match the ASE "
            "transport spontaneous source on the modeled solid-angle basis."
        ),
        severity=event_energy_severity,
    )
    ase_clamped = bool(ase.get("intensity_clamp_active", False))
    check(
        "ase_intensity_clamp",
        not ase_clamped,
        value=ase_clamped,
        limit=False,
        message="ASE intensity clamping was active.",
    )
    ase_exponent_clipped = bool(
        ase.get("positive_cell_exponent_clipping_active", False)
    )
    check(
        "ase_exponent_clipping",
        not ase_exponent_clipped,
        value=ase_exponent_clipped,
        limit=False,
        message="ASE positive cell-gain exponent clipping was active.",
    )
    ase_enabled = bool(ase.get("enabled", False))
    angular_diagnostic_keys = {
        "dominant_parasitic_boundary_eigenvalue",
        "parasitic_eigenmode_converged",
        "parasitic_eigenpair_residual",
        "reflected_boundary_flux_relative_residual",
        "embedded_boundary_resolved",
    }
    angular_ase_enabled = bool(
        ase_enabled
        and (
            ase.get("model") == "angular_discrete_ordinates"
            or any(key in ase for key in angular_diagnostic_keys)
        )
    )
    ase_converged = ase.get("boundary_converged") if ase_enabled else None
    check(
        "ase_boundary_convergence",
        None if not ase_enabled else bool(ase_converged),
        value=ase_converged,
        limit=True,
        message="ASE boundary iteration did not report convergence or did not converge.",
    )
    parasitic_eigenvalue = ase.get("dominant_parasitic_boundary_eigenvalue")
    check(
        "ase_parasitic_threshold",
        (
            None
            if not angular_ase_enabled
            else parasitic_eigenvalue is not None
            and float(parasitic_eigenvalue) < 1.0
        ),
        value=parasitic_eigenvalue,
        limit="< 1",
        message=(
            "The reflected ASE boundary operator is at or above the parasitic "
            "oscillation threshold."
        ),
    )
    eigenmode_converged = ase.get("parasitic_eigenmode_converged")
    check(
        "ase_parasitic_eigenmode_convergence",
        None if not angular_ase_enabled else bool(eigenmode_converged),
        value=eigenmode_converged,
        limit=True,
        message=(
            "ASE parasitic boundary-eigenmode iteration did not report "
            "convergence or did not converge."
        ),
    )
    eigenpair_residual = ase.get("parasitic_eigenpair_residual")
    eigenpair_limit = float(cfg.get("ase_eigenpair_relative_tolerance", 1e-5))
    check(
        "ase_parasitic_eigenpair_residual",
        (
            None
            if not angular_ase_enabled
            else eigenpair_residual is not None
            and float(eigenpair_residual) <= eigenpair_limit
        ),
        value=eigenpair_residual,
        limit=eigenpair_limit,
        message="ASE parasitic eigenpair residual exceeds tolerance.",
    )
    reflection_flux_residual = ase.get(
        "reflected_boundary_flux_relative_residual"
    )
    check(
        "ase_reflected_boundary_flux_conservation",
        (
            None
            if not angular_ase_enabled
            else reflection_flux_residual is not None
            and float(reflection_flux_residual) <= 1e-12
        ),
        value=reflection_flux_residual,
        limit=1e-12,
        message="ASE angular reflection does not conserve discrete boundary flux.",
    )
    angular_geometry_resolved = ase.get("embedded_boundary_resolved")
    check(
        "ase_angular_boundary_geometry",
        (
            None
            if not angular_ase_enabled
            else bool(angular_geometry_resolved)
        ),
        value=angular_geometry_resolved,
        limit=True,
        message=(
            "Angular ASE was requested for a non-Cartesian material boundary, but "
            "body-fitted reflection/TIR is not resolved by this backend."
        ),
    )
    ase_energy = ase.get("energy_ledger_W") or {}
    radiative_lifetime_source = ase.get("radiative_lifetime_source")
    check(
        "ase_explicit_radiative_lifetime",
        (
            None
            if not ase.get("enabled", False)
            else radiative_lifetime_source == "explicit_radiative_lifetime"
        ),
        value=radiative_lifetime_source,
        limit="explicit_radiative_lifetime",
        message=(
            "Strict quantitative ASE requires ase.radiative_lifetime_s or "
            "energy_levels.upper_state_radiative_lifetime_s; the total "
            "upper-state lifetime may include non-radiative quenching."
        ),
    )
    ase_energy_residual = ase_energy.get("relative_residual")
    ase_energy_limit = float(cfg.get("ase_energy_balance_relative_tolerance", 0.05))
    check(
        "ase_energy_balance",
        (
            None
            if not ase.get("enabled", False) or ase_energy_residual is None
            else float(ase_energy_residual) <= ase_energy_limit
        ),
        value=ase_energy_residual,
        limit=ase_energy_limit,
        message="ASE spontaneous/gain/loss/escape ledger exceeds its tolerance.",
    )

    component_heat = thermal.get(
        "pump_component_ledger_applicability"
    ) or {}
    component_heat_required = (
        str(thermal.get("heat_source_model", ""))
        == "pump_component_ledger"
    )
    component_heat_severity = (
        "warning"
        if component_heat_required and thermal_diagnostic_only
        else "blocking"
    )
    check(
        "pump_component_heat_ledger_applicability",
        (
            None
            if not component_heat_required
            else bool(component_heat.get("applicable", False))
        ),
        value=component_heat,
        limit={
            "pump_stimulated_emission_absent": True,
            "active_absorption_nonnegative": True,
            "custom_level_gap_consistent_with_pump_photon_energy": True,
        },
        message=(
            "The reduced Pump component heat ledger cannot combine Pump "
            "absorption/stimulated emission or a custom level gap inconsistent "
            "with the Pump photon energy; use event_energy_ledger."
        ),
        severity=component_heat_severity,
    )

    storage = summary.get("energy_ledger", {}).get("level_storage_J", {})
    storage_residual = storage.get("relative_closure_residual")
    storage_limit = float(
        cfg.get("level_storage_relative_tolerance", 1e-6)
    )
    check(
        "seed_level_storage_closure",
        (
            None
            if storage_residual is None
            or not bool(storage.get("simple_snapshot_closure_applicable", False))
            else float(storage_residual) <= storage_limit
        ),
        value=storage_residual,
        limit=storage_limit,
        message="Seed active optical transfer does not close against level stored-energy change.",
    )

    boundary = (
        summary.get("optical_train", {})
        .get("boundary_window_diagnostics", {})
        .get("beams", {})
    )
    risky_beams = sorted(
        name
        for name, value in boundary.items()
        if value.get("at_risk", False)
        and bool(value.get("periodic_fft_applies", True))
    )
    recorded_boundary = (
        summary.get("optical_train", {})
        .get("recorded_plane_boundary_diagnostics", {})
        .get("beams", {})
    )
    risky_recorded_planes = sorted(
        f"{name}:{value.get('plane', 'unknown')}"
        for name, value in recorded_boundary.items()
        if value.get("at_risk", False)
    )
    check(
        "fft_spatial_boundary",
        not risky_beams,
        value=risky_beams,
        limit=[],
        message=(
            "A field propagated with a periodic spatial FFT exceeds its "
            "in-medium boundary-occupancy contract."
        ),
    )
    check(
        "recorded_plane_window_occupancy",
        not risky_recorded_planes,
        value=risky_recorded_planes,
        limit=[],
        message=(
            "A recorded diagnostic plane does not contain enough of the "
            "beam for quantitative profile or sampled-M2 interpretation; "
            "this does not by itself invalidate power propagation."
        ),
        severity="warning",
    )
    multiphysics = summary.get("multiphysics", {}) or {}
    thermal_enabled = bool((summary.get("thermal") or {}).get("enabled", False))
    strong_field_diagnostics = (
        (signal.get("cpa") or {}).get("strong_field", {}) or {}
    )
    strong_field_absorbed_energy_J = float(
        strong_field_diagnostics.get(
            "total_absorbed_energy_J",
            signal.get("nonlinear_loss_energy_J", 0.0),
        )
        or 0.0
    )
    strong_field_heat_fraction = float(
        strong_field_diagnostics.get("deposit_fraction_as_heat", 1.0) or 0.0
    )
    strong_field_thermalized_energy_J = (
        strong_field_absorbed_energy_J * strong_field_heat_fraction
    )
    seed_heat_feedback_required = bool(
        thermal_enabled
        and bool(strong_field_diagnostics.get("enabled", False))
        and strong_field_thermalized_energy_J > 0.0
    )
    seed_to_thermal_feedback = bool(
        multiphysics.get("seed_to_thermal_feedback", False)
    )
    check(
        "strong_field_seed_heat_feedback",
        (
            None
            if not seed_heat_feedback_required
            else seed_to_thermal_feedback
        ),
        value={
            "thermal_enabled": thermal_enabled,
            "strong_field_absorbed_energy_J": strong_field_absorbed_energy_J,
            "deposit_fraction_as_heat": strong_field_heat_fraction,
            "strong_field_thermalized_energy_J": strong_field_thermalized_energy_J,
            "seed_to_thermal_feedback": seed_to_thermal_feedback,
        },
        limit={"seed_to_thermal_feedback": True},
        message=(
            "The optical solution removed Seed energy through strong-field loss, "
            "but that irreversible loss was excluded from the active thermal field."
        ),
        severity="blocking" if mode == "strict" else "warning",
    )
    seed_heat_one_way = bool(
        multiphysics.get("seed_heat_one_way_postprocess", False)
    )
    check(
        "seed_heat_one_way_postprocess_scope",
        None if not seed_heat_one_way else False,
        value={
            "active": seed_heat_one_way,
            "closure_iterations": multiphysics.get(
                "seed_heat_closure_iterations"
            ),
            "self_consistent": multiphysics.get("seed_heat_self_consistent"),
        },
        limit={"self_consistent": True},
        message=(
            "Seed irreversible heat was applied through one Picard update only; "
            "the reported Seed and its deposited heat are not a converged "
            "thermo-optic fixed point."
        ),
        severity="blocking" if mode == "strict" else "warning",
    )
    strong_field_configured = bool(
        bool(strong_field_diagnostics.get("enabled", False))
        and float(strong_field_diagnostics.get("beta_SI", 0.0) or 0.0) > 0.0
    )
    strong_field_mechanism = str(
        strong_field_diagnostics.get("mechanism", "unspecified")
    )
    strong_field_provenance = (
        strong_field_diagnostics.get("provenance", {}) or {}
    )
    provenance_status = str(
        strong_field_provenance.get("status", "unverified")
    )
    provenance_reference = str(
        strong_field_provenance.get("reference", "") or ""
    ).strip()
    provenance_sample = str(
        strong_field_provenance.get("sample_identifier", "") or ""
    ).strip()
    if strong_field_mechanism == "measured_sample_effective":
        strong_field_provenance_ready = bool(
            provenance_status == "sample_measured"
            and provenance_reference
            and provenance_sample
        )
    else:
        strong_field_provenance_ready = bool(
            strong_field_mechanism
            not in {"unspecified", "phenomenological_sensitivity"}
            and provenance_status in {"literature", "sample_measured"}
            and provenance_reference
        )
    check(
        "strong_field_physical_provenance",
        (
            None
            if not strong_field_configured
            else strong_field_provenance_ready
        ),
        value={
            "mechanism": strong_field_mechanism,
            "status": provenance_status,
            "reference_present": bool(provenance_reference),
            "sample_identifier_present": bool(provenance_sample),
        },
        limit={
            "mechanism": "physically identified",
            "status": "literature or sample_measured",
            "reference_present": True,
        },
        message=(
            "The enabled strong-field coefficient is unverified or is explicitly "
            "a phenomenological sensitivity parameter; it cannot support a "
            "quantitative mechanism claim."
        ),
        severity="blocking" if mode == "strict" else "warning",
    )
    cpa = signal.get("cpa") or {}
    cpa_model = str(cpa.get("model", ""))
    coherent_sampling_required = bool(
        executed_signal_model == "coherent_cpa"
        or cpa_model
        in {
            "commercial_app_effective_area_1d_spm",
            "commercial_app_effective_area_1d_strong_field",
        }
    )
    sampling = cpa.get("sampling") or {}
    sampling_adequate = sampling.get("adequate")
    check(
        "coherent_fft_input_sampling",
        (
            None
            if not coherent_sampling_required
            else sampling_adequate is not None and bool(sampling_adequate)
        ),
        value=sampling,
        limit={"adequate": True},
        message=(
            "The coherent input does not meet the joint spectral-resolution, "
            "temporal-resolution, and FFT-boundary sampling contract."
        ),
    )
    input_fallback = cpa.get("input_fallback") or {}
    input_fallback_active = bool(input_fallback.get("active", False))
    check(
        "coherent_input_fallback",
        None if not cpa else not input_fallback_active,
        value=input_fallback,
        limit={"active": False},
        message=(
            "The requested coherent input failed and a synthetic Gaussian "
            "input was substituted. The result is diagnostic only."
        ),
    )
    fidelity_mode = str(cpa.get("fidelity_mode", "configured"))
    reduced_coherent_mode = fidelity_mode in {
        "hybrid_whole_fluence",
        "axisymmetric_kerr",
        "axisymmetric_transient_gain",
    }
    check(
        "coherent_hybrid_approximation_scope",
        not reduced_coherent_mode,
        value=fidelity_mode,
        limit="configured/reference_transient_gain, or passed reduced-model gates",
        message=(
            "The selected coherent Kerr acceleration uses an explicit reduced "
            "closure. Hybrid whole-fluence and axisymmetric_kerr omit intrapulse "
            "front-to-tail depletion. The azimuthal solver retains only its "
            "configured finite m range; axisymmetric_transient_gain retains the "
            "ordered soft-window gain closure but is still a reduced spatial "
            "model. Use the reported projection/omitted-order gates and a smaller "
            "Cartesian reference before quantitative interpretation."
        ),
        severity="warning",
    )
    multirate_gain = (
        (cpa.get("local_operator") or {}).get("multirate_gain") or {}
    )
    multirate_active = bool(multirate_gain.get("active", False))
    independent_gain = (cpa.get("local_operator") or {}).get("spatial_population_update") == "independent_groups"
    check(
        "independent_spatial_gain_closure",
        None if not cpa else not independent_gain,
        value={"spatial_population_update": (cpa.get("local_operator") or {}).get("spatial_population_update", "shared_cell")},
        limit={"independent_groups": False},
        message=("Independent spatial gain groups start from one parent-cell pre-pulse density, "
                 "resolve local saturation, and commit a volume-weighted final density at the "
                 "cell boundary. Subcell gradients are not retained between pulses; verify spatial convergence."),
        severity="warning",
    )
    capped_gain = multirate_gain.get("maximum_updates_per_cell") is not None
    check(
        "coherent_gain_update_count_cap",
        None if not cpa else not (capped_gain and multirate_gain.get("gain_exponent_limit_exceeded", False)),
        value={"maximum_updates_per_cell": multirate_gain.get("maximum_updates_per_cell"),
               "maximum_estimated_gain_exponent_per_update": multirate_gain.get("maximum_estimated_gain_exponent_per_update")},
        limit={"maximum_gain_exponent_per_update": multirate_gain.get("maximum_gain_exponent_per_update")},
        message="An explicit gain-call count cap takes priority over automatic gain-exponent grouping limits; a count cap is not an accuracy guarantee.",
        severity="warning",
    )
    check(
        "coherent_multirate_gain_approximation_scope",
        None if not cpa else not multirate_active,
        value={
            "active": multirate_active,
            "gain_update_reduction_fraction": multirate_gain.get(
                "gain_update_reduction_fraction"
            ),
            "maximum_group_size": multirate_gain.get(
                "maximum_group_size"
            ),
            "coupling_blockers": multirate_gain.get(
                "current_coupling_blockers", []
            ),
        },
        limit={"active": False},
        message=(
            "Coherent multirate gain grouped population/gain updates across "
            "finer diffraction, frozen thermal-phase, dispersion, Kerr, and memoryless multiphoton-loss "
            "substeps. This is a controlled approximation: compare output "
            "energy, fluence, population, spectrum, pulse duration, and "
            "B-integral against multirate_gain.enabled=false before "
            "quantitative use."
        ),
        severity="warning",
    )
    transient_acceleration = cpa.get("transient_gain") or {}
    inactive_windows_bypassed = int(
        transient_acceleration.get("inactive_windows_bypassed", 0) or 0
    )
    inactive_window_bypass_active = inactive_windows_bypassed > 0
    check(
        "coherent_inactive_window_bypass_approximation_scope",
        None if not cpa else not inactive_window_bypass_active,
        value={
            "active": inactive_window_bypass_active,
            "bypassed_windows": inactive_windows_bypassed,
            "active_windows": transient_acceleration.get(
                "active_windows_processed"
            ),
            "configured_output_energy_budget_per_gain_step": (
                transient_acceleration.get(
                    "maximum_bypassed_output_energy_fraction"
                )
            ),
            "maximum_estimated_output_energy_fraction": (
                transient_acceleration.get(
                    "maximum_estimated_bypassed_output_energy_fraction"
                )
            ),
            "maximum_local_estimated_output_energy_fraction": (
                transient_acceleration.get(
                    "maximum_local_estimated_bypassed_output_energy_fraction"
                )
            ),
            "coupling_fallback_reason": transient_acceleration.get(
                "inactive_window_bypass_fallback_reason"
            ),
        },
        limit={"active": False},
        message=(
            "The coherent gain solver bypassed soft windows under a "
            "conservative per-gain-step output-energy budget. Active windows "
            "retain the native frequency grid and chronological population "
            "updates, but quantitative use still requires comparison with "
            "maximum_bypassed_output_energy_fraction=0."
        ),
        severity="warning",
    )
    axisymmetric = cpa.get("axisymmetric_reduction") or {}
    axisymmetric_active = fidelity_mode in {
        "axisymmetric_kerr",
        "axisymmetric_transient_gain",
    }
    projection_fraction = axisymmetric.get(
        "total_input_projection_energy_fraction"
    )
    projection_loss_limit = float(
        cfg.get("maximum_axisymmetric_projection_loss_fraction", 1e-3)
    )
    check(
        "axisymmetric_input_projection",
        (
            None
            if not axisymmetric_active
            else projection_fraction is not None
            and 1.0 - float(projection_fraction) <= projection_loss_limit
        ),
        value=projection_fraction,
        limit={"minimum_retained_fraction": 1.0 - projection_loss_limit},
        message=(
            "The gain-entrance field contains too much content outside the "
            "configured retained azimuthal orders."
        ),
    )
    dynamic_omitted_fraction = float(
        axisymmetric.get(
            "maximum_dynamic_field_omitted_energy_fraction",
            0.0,
        )
    )
    check(
        "axisymmetric_dynamic_azimuthal_closure",
        (
            None
            if not axisymmetric_active
            else dynamic_omitted_fraction <= projection_loss_limit
        ),
        value=dynamic_omitted_fraction,
        limit=projection_loss_limit,
        message=(
            "Local gain, thermal, or nonlinear propagation generated too "
            "much optical energy above the retained azimuthal order."
        ),
    )
    axisymmetric_boundary = axisymmetric.get(
        "maximum_energy_weighted_outer_annulus_fraction",
        axisymmetric.get("maximum_outer_annulus_energy_fraction"),
    )
    axisymmetric_boundary_limit = float(
        cfg.get("maximum_axisymmetric_boundary_energy_fraction", 1e-4)
    )
    check(
        "axisymmetric_radial_boundary",
        (
            None
            if not axisymmetric_active
            else axisymmetric_boundary is not None
            and float(axisymmetric_boundary) <= axisymmetric_boundary_limit
        ),
        value=axisymmetric_boundary,
        limit=axisymmetric_boundary_limit,
        message=(
            "The total-energy-weighted axisymmetric field reaches the finite "
            "radial boundary."
        ),
    )
    axisymmetric_residual = max(
        float(axisymmetric.get("maximum_population_azimuthal_residual", 0.0)),
        float(axisymmetric.get("maximum_delta_n_azimuthal_residual", 0.0)),
        float(
            axisymmetric.get(
                "maximum_free_carrier_azimuthal_residual",
                0.0,
            )
        ),
        float(
            axisymmetric.get(
                "maximum_cheap_m2_material_fit_residual",
                0.0,
            )
        ),
    )
    axisymmetric_residual_limit = float(
        cfg.get("maximum_axisymmetric_azimuthal_residual", 5e-2)
    )
    check(
        "axisymmetric_material_symmetry",
        (
            None
            if not axisymmetric_active
            else axisymmetric_residual <= axisymmetric_residual_limit
        ),
        value=axisymmetric_residual,
        limit=axisymmetric_residual_limit,
        message=(
            "The population, thermo-optic, pre-pulse free-carrier, or cheap "
            "quadratic m=2 material field is not represented accurately "
            "enough by the configured azimuthal reduction."
        ),
    )
    realizability_correction = float(
        axisymmetric.get(
            "maximum_population_realizability_correction_fraction",
            0.0,
        )
    )
    realizability_correction_limit = float(
        cfg.get(
            "maximum_axisymmetric_population_realizability_correction_fraction",
            5e-2,
        )
    )
    check(
        "axisymmetric_population_realizability",
        (
            None
            if not axisymmetric_active
            else realizability_correction <= realizability_correction_limit
        ),
        value=realizability_correction,
        limit=realizability_correction_limit,
        message=(
            "Making the retained angular population state non-negative "
            "changed too much ion inventory."
        ),
    )
    positivity_redistribution = float(
        axisymmetric.get(
            "maximum_population_positivity_redistribution_fraction",
            0.0,
        )
    )
    positivity_redistribution_limit = float(
        cfg.get(
            "maximum_axisymmetric_population_positivity_redistribution_fraction",
            5e-2,
        )
    )
    check(
        "axisymmetric_population_positivity_redistribution",
        (
            None
            if not axisymmetric_active
            else positivity_redistribution
            <= positivity_redistribution_limit
        ),
        value=positivity_redistribution,
        limit=positivity_redistribution_limit,
        message=(
            "The non-negative Cartesian population closure had to "
            "redistribute too much of the retained angular transfer."
        ),
    )
    axisymmetric_norm = axisymmetric.get("maximum_diffraction_norm_correction")
    axisymmetric_norm_limit = float(
        cfg.get("maximum_axisymmetric_diffraction_norm_correction", 1e-3)
    )
    check(
        "axisymmetric_diffraction_norm",
        (
            None
            if not axisymmetric_active
            else axisymmetric_norm is not None
            and float(axisymmetric_norm) <= axisymmetric_norm_limit
        ),
        value=axisymmetric_norm,
        limit=axisymmetric_norm_limit,
        message="The radial finite-volume diffraction step required too much norm correction.",
    )
    axisymmetric_gain_span = axisymmetric.get(
        "maximum_gain_bin_relative_frequency_span"
    )
    axisymmetric_gain_span_limit = float(
        cfg.get(
            "maximum_axisymmetric_gain_bin_relative_frequency_span",
            5e-2,
        )
    )
    check(
        "axisymmetric_gain_spectral_resolution",
        (
            None
            if not axisymmetric_active
            else axisymmetric_gain_span is not None
            and float(axisymmetric_gain_span) <= axisymmetric_gain_span_limit
        ),
        value=axisymmetric_gain_span,
        limit=axisymmetric_gain_span_limit,
        message=(
            "The reduced gain-frequency groups are too broad across the "
            "calibrated or occupied spectrum."
        ),
    )
    axisymmetric_gain_correction = axisymmetric.get(
        "maximum_gain_reconstruction_normalization_correction"
    )
    axisymmetric_gain_correction_limit = float(
        cfg.get(
            "maximum_axisymmetric_gain_reconstruction_normalization_correction",
            2e-2,
        )
    )
    check(
        "axisymmetric_gain_reconstruction",
        (
            None
            if not axisymmetric_active
            else axisymmetric_gain_correction is not None
            and float(axisymmetric_gain_correction)
            <= axisymmetric_gain_correction_limit
        ),
        value=axisymmetric_gain_correction,
        limit=axisymmetric_gain_correction_limit,
        message=(
            "Fine-spectrum gain reconstruction required too large a local "
            "energy normalization."
        ),
    )
    axisymmetric_mapping = axisymmetric.get("population_mapping_relative_error")
    axisymmetric_mapping_limit = float(
        cfg.get("axisymmetric_population_mapping_relative_tolerance", 1e-10)
    )
    check(
        "axisymmetric_population_mapping",
        (
            None
            if not axisymmetric_active
            else axisymmetric_mapping is not None
            and float(axisymmetric_mapping) <= axisymmetric_mapping_limit
        ),
        value=axisymmetric_mapping,
        limit=axisymmetric_mapping_limit,
        message="The radial-to-Cartesian population update is not conservative.",
    )
    temporal_boundary = cpa.get("maximum_temporal_boundary_energy_fraction")
    carrier_mapping = axisymmetric.get("free_carrier_mapping_relative_error")
    carrier_mapping_active = bool(
        axisymmetric_active
        and strong_field_diagnostics.get("enabled", False)
        and strong_field_diagnostics.get("free_carrier_model") == "rate_equation"
    )
    check(
        "axisymmetric_free_carrier_mapping",
        None if not carrier_mapping_active else (
            carrier_mapping is not None and float(carrier_mapping) <= 1e-10
        ),
        value=carrier_mapping,
        limit=1e-10,
        message="The radial/azimuthal-to-Cartesian carrier update is not conservative.",
    )
    temporal_limit = min(
        float(cpa.get("temporal_boundary_warning_fraction", 1.0)),
        float(cfg.get("temporal_boundary_energy_fraction", 1e-6)),
    )
    check(
        "fft_temporal_boundary",
        (
            None
            if temporal_boundary is None
            else float(temporal_boundary) <= float(temporal_limit)
        ),
        value=temporal_boundary,
        limit=temporal_limit,
        message="The coherent temporal FFT boundary occupancy is too large.",
    )
    output_spectral_boundary = cpa.get(
        "maximum_output_spectral_boundary_energy_fraction"
    )
    output_spectral_limit = min(
        float(cpa.get("spectral_boundary_warning_fraction", 1.0)),
        float(cfg.get("output_spectral_boundary_energy_fraction", 1e-6)),
    )
    check(
        "fft_output_spectral_boundary",
        (
            None
            if output_spectral_boundary is None
            else float(output_spectral_boundary) <= output_spectral_limit
        ),
        value=output_spectral_boundary,
        limit=output_spectral_limit,
        message=(
            "The nonlinear coherent output reaches the spectral FFT boundary; "
            "SPM broadening may be wrapped or truncated."
        ),
    )
    transient_gain = cpa.get("transient_gain") or {}
    resynthesis_error = transient_gain.get("maximum_resynthesis_energy_error")
    resynthesis_limit = float(
        cfg.get("cpa_resynthesis_energy_relative_tolerance", 1e-3)
    )
    transient_gain_enabled = bool(transient_gain.get("enabled", False))
    depletion_met = transient_gain.get("depletion_target_met")
    depletion_actual = transient_gain.get("maximum_reservoir_depletion_per_slice")
    depletion_target = transient_gain.get("maximum_depletion_per_slice_target")
    check(
        "cpa_transient_gain_depletion_resolution",
        None if not transient_gain_enabled or depletion_met is None else (
            bool(depletion_met)
            and depletion_actual is not None
            and depletion_target is not None
            and float(depletion_actual) <= float(depletion_target)
        ),
        value=depletion_actual,
        limit=depletion_target,
        message="Coherent transient gain exceeded its per-slice reservoir-depletion target.",
    )
    transient_gain_fallback = bool(
        transient_gain_enabled
        and str(transient_gain.get("status", "")).startswith("fallback")
    )
    check(
        "cpa_transient_gain_fallback",
        None if not transient_gain_enabled else not transient_gain_fallback,
        value={
            "status": transient_gain.get("status"),
            "reason": transient_gain.get("reason"),
        },
        limit={"fallback": False},
        message=(
            "Transient coherent gain was requested but the solve restarted "
            "with whole-fluence gain. The run completed, but not under the "
            "requested scientific contract."
        ),
    )
    check(
        "cpa_soft_window_resynthesis",
        (
            None
            if not transient_gain_enabled
            else resynthesis_error is not None
            and float(resynthesis_error) <= resynthesis_limit
        ),
        value=resynthesis_error,
        limit=resynthesis_limit,
        message="CPA soft-window resynthesis energy error exceeds tolerance.",
    )
    bypass_budget = float(
        transient_gain.get(
            "maximum_bypassed_output_energy_fraction", 0.0
        )
        or 0.0
    )
    bypass_global_bound = float(
        transient_gain.get(
            "maximum_estimated_bypassed_output_energy_fraction", 0.0
        )
        or 0.0
    )
    bypass_local_bound = float(
        transient_gain.get(
            "maximum_local_estimated_bypassed_output_energy_fraction",
            0.0,
        )
        or 0.0
    )
    bypass_bound = max(bypass_global_bound, bypass_local_bound)
    bypass_active = int(
        transient_gain.get("inactive_windows_bypassed", 0) or 0
    ) > 0
    check(
        "cpa_inactive_window_bypass_budget",
        (
            None
            if not transient_gain_enabled or not bypass_active
            else bypass_bound
            <= bypass_budget * (1.0 + 1e-12)
        ),
        value=bypass_bound if bypass_active else None,
        limit=bypass_budget,
        message=(
            "The inactive soft-window bypass exceeded its configured "
            "conservative output-energy budget."
        ),
    )
    local_operator = cpa.get("local_operator") or {}
    local_cap = local_operator.get("substep_cap_active")
    coherent_backend = str(signal.get("model", "")) == "coherent_cpa"
    local_operator_required = bool(coherent_backend or local_cap is not None)
    check(
        "cpa_local_operator_substep_cap",
        (
            None
            if not local_operator_required
            else local_cap is not None and not bool(local_cap)
        ),
        value=local_cap,
        limit=False,
        message="CPA local-operator diagnostics are missing or its substep cap was reached.",
    )
    strong_field = cpa.get("strong_field") or {}
    strong_loss = strong_field.get("maximum_fractional_loss_per_step")
    strong_loss_limit = float(
        cfg.get("strong_field_fractional_loss_per_step", 0.05)
    )
    check(
        "strong_field_step_resolution",
        (
            None
            if not strong_field.get("enabled", False)
            else strong_loss is not None
            and float(strong_loss) <= strong_loss_limit
        ),
        value=strong_loss,
        limit=strong_loss_limit,
        message="A strong-field substep removed too much optical energy.",
    )
    commercial_app_carrier_periodic_solve_count = int(
        strong_field.get("carrier_periodic_solve_count", 0) or 0
    )
    commercial_app_carrier_periodic_residual = strong_field.get(
        "maximum_carrier_periodic_relative_residual"
    )
    commercial_app_carrier_periodic_limit = strong_field.get(
        "carrier_periodic_relative_tolerance"
    )
    check(
        "commercial_app_effective_area_carrier_periodic_state",
        (
            None
            if commercial_app_carrier_periodic_solve_count <= 0
            else commercial_app_carrier_periodic_residual is not None
            and commercial_app_carrier_periodic_limit is not None
            and bool(strong_field.get("carrier_periodic_converged", False))
            and float(commercial_app_carrier_periodic_residual)
            <= float(commercial_app_carrier_periodic_limit)
        ),
        value=commercial_app_carrier_periodic_residual,
        limit=commercial_app_carrier_periodic_limit,
        message=(
            "The Commercial App effective-area carrier state did not close across the "
            "pulse and inter-pulse dark interval."
        ),
    )
    commercial_app_effective_area = cpa_model in {
        "commercial_app_effective_area_1d_spm",
        "commercial_app_effective_area_1d_strong_field",
    }
    commercial_app_support_leakage = cpa.get(
        "maximum_spectral_support_leakage_fraction"
    )
    commercial_app_support_limit = float(
        cfg.get("maximum_commercial_app_spectral_support_leakage_fraction", 0.01)
    )
    check(
        "commercial_app_spectral_feedback_support",
        (
            None
            if not commercial_app_effective_area
            else commercial_app_support_leakage is not None
            and float(commercial_app_support_leakage) <= commercial_app_support_limit
        ),
        value=commercial_app_support_leakage,
        limit=commercial_app_support_limit,
        message=(
            "The Commercial App 1-D SPM spectrum extends beyond the coarse gain bins, "
            "where feedback is forced into edge channels."
        ),
    )
    commercial_app_sync_error = cpa.get("maximum_spectral_sync_error_fraction")
    commercial_app_sync_limit = float(
        cfg.get("maximum_commercial_app_spectral_sync_error_fraction", 0.01)
    )
    check(
        "commercial_app_spectral_feedback_sync",
        (
            None
            if not commercial_app_effective_area
            else commercial_app_sync_error is not None
            and float(commercial_app_sync_error) <= commercial_app_sync_limit
        ),
        value=commercial_app_sync_error,
        limit=commercial_app_sync_limit,
        message=(
            "The Commercial App coherent envelope and coarse gain-bin energies could not "
            "be synchronized within tolerance."
        ),
    )
    commercial_app_b_integral = cpa.get("maximum_cumulative_B_integral_rad")
    commercial_app_b_limit = float(
        cfg.get("maximum_commercial_app_reduced_spatiotemporal_B_integral_rad", 1.0)
    )
    check(
        "commercial_app_reduced_spatiotemporal_kerr_applicability",
        (
            None
            if not commercial_app_effective_area
            else commercial_app_b_integral is not None
            and float(commercial_app_b_integral) <= commercial_app_b_limit
        ),
        value=commercial_app_b_integral,
        limit=commercial_app_b_limit,
        message=(
            "The separable Commercial App Gaussian-q plus effective-area envelope is being "
            "used beyond the configured nonlinear-phase applicability limit; "
            "space-time Kerr coupling needs the coherent full-field backend."
        ),
    )
    retrieval = cpa.get("phase_retrieval")
    retrieval_required = bool(
        retrieval is not None
        or (
            coherent_backend
            and str(cpa.get("input_mode", "")) == "phase_retrieval"
        )
    )
    check(
        "phase_retrieval_convergence",
        (
            None
            if not retrieval_required
            else retrieval is not None and bool(retrieval.get("converged", False))
        ),
        value=(
            None if retrieval is None else retrieval.get("relative_amplitude_error")
        ),
        limit=None if retrieval is None else retrieval.get("tolerance"),
        message="Coherent phase-retrieval diagnostics are missing or did not converge.",
    )

    material_range = (
        summary.get("spectroscopic_response", {}).get("material_range", {})
    )
    out_of_range = int(material_range.get("out_of_range_total", 0))
    weighted_out_of_range = float(
        material_range.get("power_weighted_out_of_range_fraction", 0.0)
    )
    weighted_range_limit = float(
        cfg.get("static_material_power_out_of_range_tolerance", 0.0)
    )
    unweighted_only_out_of_range = int(
        material_range.get("unweighted_only_out_of_range_total", out_of_range)
    )
    check(
        "material_table_range",
        unweighted_only_out_of_range == 0
        and weighted_out_of_range <= weighted_range_limit,
        value={
            "sample_count": out_of_range,
            "unweighted_only_sample_count": unweighted_only_out_of_range,
            "power_weighted_fraction": weighted_out_of_range,
        },
        limit={
            "unweighted_only_sample_count": 0,
            "power_weighted_fraction": weighted_range_limit,
        },
        message="A material table was evaluated outside its calibrated range.",
    )

    material_calibration = (
        (summary.get("material_library") or {}).get(
            "experimental_calibration", {}
        )
        or {}
    )
    require_material_calibration = bool(
        cfg.get("require_experimentally_calibrated_material", False)
    )
    calibration_status = str(material_calibration.get("status", "unspecified"))
    required_parameters = {
        str(value)
        for value in material_calibration.get("required_parameters", [])
    }
    calibrated_parameters = {
        str(value)
        for value in material_calibration.get("calibrated_parameters", [])
    }
    missing_calibrations = sorted(required_parameters - calibrated_parameters)
    sample_identifier = str(
        material_calibration.get("sample_identifier", "") or ""
    ).strip()
    dataset_sha256 = str(
        material_calibration.get("dataset_sha256", "") or ""
    ).lower()
    evidence_hash_valid = len(dataset_sha256) == 64 and all(
        character in "0123456789abcdef" for character in dataset_sha256
    )
    calibration_ready = (
        calibration_status == "experimentally_calibrated"
        and bool(sample_identifier)
        and evidence_hash_valid
        and not missing_calibrations
    )
    check(
        "experimental_material_calibration",
        None if not require_material_calibration else calibration_ready,
        value={
            "required": require_material_calibration,
            "status": calibration_status,
            "sample_identifier_present": bool(sample_identifier),
            "dataset_sha256_valid": evidence_hash_valid,
            "missing_parameters": missing_calibrations,
        },
        limit={
            "status": "experimentally_calibrated",
            "sample_identifier_present": True,
            "dataset_sha256_valid": True,
            "missing_parameters": [],
        },
        message=(
            "The setup requires sample-specific experimental material calibration, "
            "but its parameter list and evidence fingerprint are incomplete."
        ),
    )

    stress = summary.get("stress", {}) or {}
    fem_residual = stress.get("equilibrium_residual_relative")
    fem_limit = float(cfg.get("fem_equilibrium_relative_tolerance", 1e-6))
    fem_enabled = str(stress.get("model", "none")) in {
        "thermoelastic_fem",
        "anisotropic_thermoelastic_fem",
    }
    check(
        "fem_equilibrium_residual",
        (
            None
            if not fem_enabled
            else fem_residual is not None and float(fem_residual) <= fem_limit
        ),
        value=fem_residual,
        limit=fem_limit,
        message="FEM force-equilibrium residual exceeds tolerance.",
    )
    nonlinear_solver = population.get("nonlinear_solver", {}) or {}
    nonlinear_cap_calls = nonlinear_solver.get("cap_limited_calls")
    nonlinear_enabled = bool(nonlinear_solver.get("enabled", False))
    check(
        "nonlinear_level_substep_cap",
        (
            None
            if not nonlinear_enabled
            else nonlinear_cap_calls is not None
            and int(nonlinear_cap_calls) == 0
        ),
        value=nonlinear_cap_calls,
        limit=0,
        message="Nonlinear level kinetics reached the configured substep cap.",
    )
    nonlinear_fractional_step = nonlinear_solver.get(
        "maximum_estimated_fractional_depletion_step"
    )
    nonlinear_fractional_limit = min(
        float(
            nonlinear_solver.get(
                "configured_maximum_fractional_depletion_per_substep",
                0.05,
            )
        ),
        float(
            cfg.get(
                "nonlinear_fractional_depletion_step_tolerance",
                0.05,
            )
        ),
    )
    check(
        "nonlinear_level_fractional_depletion_step",
        (
            None
            if not nonlinear_enabled
            else nonlinear_fractional_step is not None
            and float(nonlinear_fractional_step)
            <= nonlinear_fractional_limit
        ),
        value=nonlinear_fractional_step,
        limit=nonlinear_fractional_limit,
        message=(
            "A nonlinear population substep is too large for the configured "
            "reaction-order accuracy contract."
        ),
    )
    signal_event = population.get("signal_event_limiter", {}) or {}
    signal_event_enabled = bool(signal_event.get("enabled", False))
    event_residual = signal_event.get(
        "maximum_requested_applied_residual_relative"
    )
    event_residual_limit = float(
        cfg.get(
            "signal_event_requested_applied_relative_tolerance",
            1e-3,
        )
    )
    check(
        "multi_transition_signal_event_limiter",
        (
            None
            if not signal_event_enabled
            else event_residual is not None
            and float(event_residual) <= event_residual_limit
        ),
        value={
            "maximum_requested_applied_residual_relative": event_residual,
            "maximum_channel_requested_applied_residual_relative": (
                signal_event.get(
                    "maximum_channel_requested_applied_residual_relative"
                )
            ),
            "minimum_limiter_scale": signal_event.get(
                "minimum_limiter_scale"
            ),
            "limited_cell_updates": signal_event.get(
                "limited_cell_updates"
            ),
        },
        limit=event_residual_limit,
        message=(
            "Multi-transition stimulated events required a large population "
            "limiter correction; refine photon packets, time bins, or Nz."
        ),
    )
    event_population_change = signal_event.get(
        "maximum_population_change_fraction_per_update"
    )
    event_population_change_limit = float(
        cfg.get(
            "signal_event_maximum_population_change_fraction",
            0.05,
        )
    )
    check(
        "multi_transition_signal_event_step_resolution",
        (
            None
            if not signal_event_enabled
            else event_population_change is not None
            and float(event_population_change)
            <= event_population_change_limit
        ),
        value=event_population_change,
        limit=event_population_change_limit,
        message=(
            "One multi-transition photon packet/time-bin update changed too "
            "much of a local population reservoir."
        ),
    )
    packet_count = int(signal_event.get("packet_count_used", 0) or 0)
    check(
        "multi_transition_photon_packet_resolution",
        (
            None
            if not signal_event_enabled or packet_count == 0
            else bool(signal_event.get("packet_target_met", False))
        ),
        value={
            "packet_count_used": packet_count,
            "maximum_population_change_per_packet": signal_event.get(
                "maximum_population_change_per_packet"
            ),
            "limited_packet_cell_count": signal_event.get(
                "limited_packet_cell_count"
            ),
        },
        limit=signal_event.get("packet_population_change_target"),
        message=(
            "The configured multi-transition photon packet count is too "
            "coarse for its per-packet population-change target; increase "
            "energy_levels.multi_transition_solver.photon_packet_count."
        ),
    )
    dynamics = str(population.get("dynamics", ""))
    cycle_jump = population.get("final_pulse_population_jump_relative")
    cycle_jump_limit = float(
        cfg.get("cycle_average_population_jump_relative_tolerance", 1e-2)
    )
    check(
        "cycle_average_population_jump",
        (
            None
            if dynamics != "cycle_averaged_steady_state"
            else cycle_jump is not None and float(cycle_jump) <= cycle_jump_limit
        ),
        value=cycle_jump,
        limit=cycle_jump_limit,
        message="One pulse changes too much population for cycle-average validity.",
    )
    probe_difference = population.get(
        "reported_vs_average_probe_relative_difference"
    )
    probe_limit = float(
        cfg.get("cycle_average_probe_difference_relative_tolerance", 5e-2)
    )
    check(
        "cycle_average_probe_consistency",
        (
            None
            if dynamics != "cycle_averaged_steady_state"
            else probe_difference is not None
            and float(probe_difference) <= probe_limit
        ),
        value=probe_difference,
        limit=probe_limit,
        message="Cycle-average probe and reported pulse output disagree.",
    )
    reduced_population_difference = population.get(
        "reported_vs_reduced_proxy_population_relative_difference"
    )
    reduced_population_limit = float(
        cfg.get(
            "reduced_pulse_map_population_difference_relative_tolerance",
            1e-2,
        )
    )
    check(
        "reduced_pulse_map_population_consistency",
        (
            None
            if dynamics != "reduced_periodic_steady_state"
            else reduced_population_difference is not None
            and float(reduced_population_difference)
            <= reduced_population_limit
        ),
        value=reduced_population_difference,
        limit=reduced_population_limit,
        message=(
            "The reduced discrete pulse map and reported configured pulse "
            "produce inconsistent post-pulse populations."
        ),
    )
    reduced_output_difference = population.get(
        "reported_vs_reduced_proxy_output_relative_difference"
    )
    reduced_output_limit = float(
        cfg.get(
            "reduced_pulse_map_output_difference_relative_tolerance",
            5e-2,
        )
    )
    check(
        "reduced_pulse_map_output_consistency",
        (
            None
            if dynamics != "reduced_periodic_steady_state"
            else reduced_output_difference is not None
            and float(reduced_output_difference) <= reduced_output_limit
        ),
        value=reduced_output_difference,
        limit=reduced_output_limit,
        message=(
            "The reduced discrete pulse map and reported configured pulse "
            "disagree in output energy."
        ),
    )
    carrier_field_residual = population.get(
        "final_free_carrier_residual_relative"
    )
    carrier_convergence_residual = population.get(
        "final_free_carrier_convergence_residual_relative",
        carrier_field_residual,
    )
    carrier_convergence_limit = float(
        population.get(
            "free_carrier_convergence_relative_tolerance",
            population.get("relative_tolerance", 1e-5),
        )
    )
    check(
        "free_carrier_periodic_state",
        (
            None
            if not population.get("free_carrier_periodic_state_enabled", False)
            else carrier_convergence_residual is not None
            and float(carrier_convergence_residual)
            <= carrier_convergence_limit
        ),
        value={
            "metric": population.get(
                "free_carrier_convergence_metric", "field"
            ),
            "residual": carrier_convergence_residual,
            "used_as_stopping_criterion": population.get("free_carrier_convergence_gate_enabled", True),
        },
        limit=carrier_convergence_limit,
        message=(
            "The configured free-carrier periodic residual remains above the "
            "fixed-point tolerance. Population, output energy, and the "
            "optically active carrier state must all converge."
        ),
    )
    carrier_field_limit = float(
        cfg.get("free_carrier_periodic_residual_relative_tolerance", 1e-5)
    )
    check(
        "free_carrier_periodic_field_diagnostic",
        (
            None
            if (
                not population.get(
                    "free_carrier_periodic_state_enabled", False
                )
                or str(
                    population.get(
                        "free_carrier_convergence_metric", "field"
                    )
                )
                == "field"
            )
            else carrier_field_residual is not None
            and float(carrier_field_residual) <= carrier_field_limit
        ),
        value=carrier_field_residual,
        limit=carrier_field_limit,
        severity="warning",
        message=(
            "The carrier density field still changes above the diagnostic "
            "max-norm tolerance, although its configured optical effect has "
            "converged."
        ),
    )
    physics = summary.get("physics", {}) or {}
    if not bool(physics.get("pump_depletion", True)):
        pump_input = abs(float(pump.get("grid_input_power_W", 0.0) or 0.0))
        absorbed_fraction = abs(
            float(pump.get("absorbed_power_W", 0.0) or 0.0)
        ) / max(pump_input, 1e-300)
        undepleted_limit = float(
            cfg.get("undepleted_pump_maximum_absorbed_fraction", 1e-3)
        )
        check(
            "undepleted_pump_applicability",
            absorbed_fraction <= undepleted_limit,
            value=absorbed_fraction,
            limit=undepleted_limit,
            message="Undepleted Pump approximation absorbs a non-negligible input fraction.",
        )
    else:
        check(
            "undepleted_pump_applicability",
            None,
            value=None,
            limit=float(
                cfg.get("undepleted_pump_maximum_absorbed_fraction", 1e-3)
            ),
            message="Undepleted Pump applicability is not relevant when depletion is enabled.",
        )

    status = (
        "invalid"
        if failures
        else ("valid_with_warnings" if warnings else "valid")
    )
    return {
        "mode": mode,
        "enforced": mode == "strict",
        "valid": not failures,
        "quantitative": not failures,
        "quantitative_scope": (
            "configured_numerical_and_physical_contracts_only; experimental "
            "agreement additionally requires an enabled and passing material "
            "calibration contract"
        ),
        "banner": NON_QUANTITATIVE_BANNER if failures else None,
        "status": status,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "issue_count": len(failures) + len(warnings),
        "failures": failures,
        "warnings": warnings,
        "issues": [*failures, *warnings],
        "checks": checks,
    }
