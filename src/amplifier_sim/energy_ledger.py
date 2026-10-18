"""Cross-module optical-energy ledgers with explicit unresolved terms."""

from __future__ import annotations

from typing import Any


def build_energy_ledger(summary: dict[str, Any]) -> dict[str, Any]:
    pump = summary.get("pump", {})
    pump_input = float(pump.get("grid_input_power_W", 0.0) or 0.0)
    pump_output = float(pump.get("output_power_W", 0.0) or 0.0)
    pump_absorbed = float(pump.get("absorbed_power_W", 0.0) or 0.0)
    pump_side = float(pump.get("crystal_side_aperture_loss_W", 0.0) or 0.0)
    pump_truncation = float(pump.get("ray_power_truncation_loss_W", 0.0) or 0.0)
    pump_balance_applicable = bool(pump.get("depletion", True))
    pump_residual = (
        pump_input
        - (pump_output + pump_absorbed + pump_side + pump_truncation)
        if pump_balance_applicable
        else None
    )

    signal = summary.get("signal", {})
    extraction = summary.get("extraction", {})
    seed_input = float(signal.get("grid_source_energy_J", 0.0) or 0.0)
    seed_output = float(signal.get("system_output_energy_J", 0.0) or 0.0)
    active_transfer = float(extraction.get("net_J", 0.0) or 0.0)
    passive = float(signal.get("passive_loss_energy_J", 0.0) or 0.0)
    nonlinear = float(signal.get("nonlinear_loss_energy_J", 0.0) or 0.0)
    aperture = float(signal.get("crystal_aperture_loss_energy_J", 0.0) or 0.0)
    external_terms = list(signal.get("external_loss_ledger", []) or [])
    numerical_window = float(
        sum(
            float(item.get("signed_loss_energy_J", 0.0) or 0.0)
            for item in external_terms
            if bool(item.get("included_in_seed_balance", True))
            and str(item.get("category", "")) == "numerical_window_truncation"
        )
    )
    named_external = float(
        sum(
            float(item.get("signed_loss_energy_J", 0.0) or 0.0)
            for item in external_terms
            if bool(item.get("included_in_seed_balance", True))
            and str(item.get("category", "")) != "numerical_window_truncation"
        )
    )
    seed_unresolved = (
        seed_input
        + active_transfer
        - seed_output
        - passive
        - nonlinear
        - aperture
        - named_external
        - numerical_window
    )

    repetition_rate = signal.get("system_output_average_power_W")
    configured_output = signal.get("system_output_energy_J")
    inferred_rate = None
    if repetition_rate is not None and configured_output not in (None, 0):
        inferred_rate = float(repetition_rate) / float(configured_output)

    stored = summary.get("stored_level_energy", {})
    stored_change = (
        float(stored["seed_step_change_J"])
        if stored.get("available", False)
        else None
    )
    stored_closure = (
        None if stored_change is None else active_transfer + stored_change
    )
    stored_scale = max(abs(active_transfer), abs(stored_change or 0.0), 1e-300)
    stored_closure_relative = (
        None if stored_closure is None else abs(stored_closure) / stored_scale
    )
    temporal_type = str(signal.get("temporal_type", "unknown"))
    gain_source = str(signal.get("gain_source", "physical_level_population"))
    intrapulse_relaxation = bool(
        signal.get("intrapulse_relaxation_enabled", False)
    )
    interpass_relaxation = bool(
        signal.get("between_pass_population_relaxation", False)
    )
    spectral_samples = int(signal.get("spectral_samples", 0) or 0)
    simple_storage_applicable = bool(
        stored_change is not None
        and temporal_type != "cw"
        and gain_source == "physical_level_population"
        and not intrapulse_relaxation
        and not interpass_relaxation
        and spectral_samples <= 1
    )
    if stored_change is None:
        storage_reason = "level energies are unavailable"
    elif temporal_type == "cw":
        storage_reason = (
            "CW is a stationary-flow snapshot, not a committed pulse state change"
        )
    elif gain_source != "physical_level_population":
        storage_reason = (
            "gain override is not represented by the physical level-energy snapshot"
        )
    elif intrapulse_relaxation or interpass_relaxation:
        storage_reason = "modeled relaxation exchanges energy with a transition bath"
    elif spectral_samples > 1:
        storage_reason = "broadband transitions require wavelength-resolved bath exchange"
    else:
        storage_reason = "simple transient storage closure applies"
    inferred_bath = None if stored_closure is None else -stored_closure
    closure_with_bath = (
        None
        if stored_closure is None or inferred_bath is None
        else stored_closure + inferred_bath
    )
    measured_terms_closed = bool(
        simple_storage_applicable
        and stored_closure_relative is not None
        and stored_closure_relative <= 1e-6
    )
    event_rates = (
        summary.get("thermal", {}).get("event_energy_ledger", {})
        or {"enabled": False}
    )
    strong_field = ((signal.get("cpa") or {}).get("strong_field") or {})
    strong_field_enabled = bool(strong_field.get("enabled", False))
    strong_field_absorbed = float(
        strong_field.get("total_absorbed_energy_J", nonlinear) or 0.0
    )
    strong_field_heat_fraction = float(
        strong_field.get("deposit_fraction_as_heat", 1.0) or 0.0
    )
    free_carrier_enabled = bool(
        strong_field_enabled
        and str(strong_field.get("free_carrier_model", "off")) != "off"
    )
    strong_field_unassigned = max(
        strong_field_absorbed * (1.0 - strong_field_heat_fraction), 0.0
    )
    if measured_terms_closed:
        system_status = "measured_terms_closed"
    elif simple_storage_applicable:
        system_status = "measured_terms_not_closed"
    elif stored_change is not None:
        system_status = "partial_ledger_with_inferred_bath"
    else:
        system_status = "partial_cross_module_ledger"

    return {
        "pump_power_W": {
            "input": pump_input,
            "output": pump_output,
            "absorbed": pump_absorbed,
            "side_escape": pump_side,
            "numerical_ray_truncation": pump_truncation,
            "signed_residual": pump_residual,
            "balance_applicable": pump_balance_applicable,
            "balance_scope": (
                "depleted_transport_power_identity"
                if pump_balance_applicable
                else "not_applicable_to_undepleted_intensity_approximation"
            ),
        },
        "seed_pulse_energy_J": {
            "input": seed_input,
            "active_transition_transfer": active_transfer,
            "output": seed_output,
            "passive_absorption": passive,
            "strong_field_absorption": nonlinear,
            "crystal_aperture_loss": aperture,
            "named_external_element_and_routing_loss": named_external,
            "numerical_window_truncation": numerical_window,
            "external_loss_terms": external_terms,
            "unresolved_external_interface_and_routing_term": seed_unresolved,
            "identity": (
                "input + active_transfer = output + passive + strong_field + "
                "aperture + named_external_element_and_routing_loss + "
                "numerical_window_truncation + unresolved_term"
            ),
        },
        "level_storage_J": {
            "available": stored_change is not None,
            "after_pump": stored.get("after_pump_J"),
            "after_seed": stored.get("after_seed_J"),
            "seed_step_change": stored_change,
            "active_optical_transfer_plus_storage_change": stored_closure,
            "relative_closure_residual": stored_closure_relative,
            "simple_snapshot_closure_applicable": simple_storage_applicable,
            "simple_snapshot_closure_reason": storage_reason,
            "contract": (
                "transient_monochromatic_no_relaxation"
                if simple_storage_applicable
                else "model_aware_bath_or_flow_ledger"
            ),
            "inferred_transition_bath_exchange": inferred_bath,
            "active_transfer_plus_storage_plus_inferred_bath": closure_with_bath,
            "relative_closure_with_inferred_bath": (
                None
                if closure_with_bath is None
                else abs(closure_with_bath) / stored_scale
            ),
            "bath_exchange_required": bool(
                stored_change is not None and not simple_storage_applicable
            ),
            "identity": (
                "active optical transfer + stored-energy change + inferred transition-bath exchange = 0"
            ),
            "energy_assignment": stored.get("source"),
            "level_energies_J": stored.get("level_energies_J"),
        },
        "event_energy_rates_W": {
            "enabled": bool(event_rates.get("enabled", False)),
            "complete_partitions": event_rates.get("complete_partitions"),
            "destination_power_W": event_rates.get("destination_power_W"),
            "event_count_rate_s": event_rates.get("event_count_rate_s"),
            "event_energy_rate_W": event_rates.get("event_energy_rate_W"),
            "modeled_level_storage_derivative_W": event_rates.get(
                "modeled_level_storage_derivative_W"
            ),
            "full_identity_residual_W": event_rates.get(
                "full_identity_residual_W"
            ),
            "finite_step_level_storage_derivative_W": event_rates.get(
                "finite_step_level_storage_derivative_W"
            ),
            "finite_step_kinetic_storage_residual_W": event_rates.get(
                "finite_step_kinetic_storage_residual_W"
            ),
            "pump_transport_kinetic_mismatch_W": event_rates.get(
                "pump_transport_kinetic_mismatch_W"
            ),
            "identity": event_rates.get("full_identity"),
            "level_energy_source": event_rates.get("level_energy_source"),
        },
        "strong_field_energy_scope": {
            "enabled": strong_field_enabled,
            "absorbed_optical_energy_J": strong_field_absorbed,
            "effective_heat_fraction": strong_field_heat_fraction,
            "effective_thermalized_energy_J": (
                strong_field_absorbed * strong_field_heat_fraction
            ),
            "unassigned_nonthermal_energy_J": strong_field_unassigned,
            "free_carrier_state_enabled": free_carrier_enabled,
            "explicit_carrier_internal_energy_reservoir": False,
            "contract": strong_field.get(
                "heat_deposition_contract",
                "cycle_averaged_effective_fraction_no_explicit_carrier_energy_reservoir",
            ),
        },
        "system_scope": {
            "status": system_status,
            "measured_terms_closed": measured_terms_closed,
            "closed_after_assigned_unresolved_bath": bool(
                stored_change is not None and not simple_storage_applicable
            ),
            "inferred_bath_is_independent_evidence": False,
            "full_first_law_closed": False,
            "repetition_rate_Hz_inferred": inferred_rate,
            "included": [
                "Pump output/absorption/side escape",
                "Seed active transfer/passive loss/strong-field loss/aperture loss",
                "Named Seed gain-interface and physical external routing losses",
                "Numerical Seed gain-entrance window truncation as a separate term",
                *(
                    ["Explicit Pump/decay/nonlinear event-energy destinations"]
                    if bool(event_rates.get("enabled", False))
                    else []
                ),
            ],
            "not_yet_closed": [
                *(
                    []
                    if stored_change is not None
                    else ["level-resolved stored-energy change (provide explicit custom level energies)"]
                ),
                "fluorescence and ASE optical escape on a common angular boundary",
                "Unconfigured fluorescence/ASE coupling to the Seed route",
                *(
                    [
                        "free-carrier internal energy, recombination radiation, and carrier transport"
                    ]
                    if free_carrier_enabled
                    else []
                ),
                *(
                    [
                        "destination of the non-thermalized fraction of strong-field optical loss"
                    ]
                    if strong_field_enabled and strong_field_unassigned > 0.0
                    else []
                ),
            ],
        },
    }
