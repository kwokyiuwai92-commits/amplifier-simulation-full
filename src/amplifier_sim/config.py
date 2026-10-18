"""Configuration loading, defaults, schema policy, and physical sanity checks."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np

from .caustic import validate_caustic_config
from .coherent_step import nonlinear_gain_coupling_blockers
from .josa_modes import (
    JOSA_FIDELITY_INFO,
    JOSA_FULL_3D_MODE,
    JOSA_MODES,
    josa_full_3d_physics_overrides,
)
from .json_contract import load_json_strict
from .material_response import SpectroscopicResponse
from .optical_system import gain_medium_path_cosine
from .seed_spatio_spectral import validate_seed_spatio_spectral
from .doping import validate_doping_profile
from .dispersion import material_refractive_index
from .field_diagnostics import validate_field_diagnostics
from .pump_source import canonicalize_pump_source
from .pass_routing import resolve_pump_direction
from .polarization import validate_polarization_config
from .seed_gain_transport import validate_spatial_gain_transport
from .spatial_modes import axis_aligned_q_anisotropy_reasons, axis_focus_m
from .result_recording import (
    RESULT_RECORDING_DEFAULTS,
    validate_result_recording,
)
from .units import TIME_UNITS_TO_S, WAVELENGTH_UNITS_TO_M
from .validation import require_exact_integer


DEFAULTS: dict[str, Any] = {
    "schema_policy": "strict_core_with_plugin_extras",
    "setup_provenance": {},
    "performance": {
        "fft_backend": "auto",
        "fft_workers": 0,
        "fft_min_elements": 4096,
        "operator_profiling": False,
    },
    "multiphysics": {
        "coupling_model": "one_way",
        "iteration_scheme": "nested",
        "joint_cycles_per_iteration": 1,
        "joint_consecutive_iterations": None,
        "frozen_probe_cycles": 3,
        "minimum_iterations": 2,
        "maximum_iterations": 20,
        "relative_tolerance": 1e-4,
        "relaxation": 0.5,
        "seed_heat_coupling": "off",
        "acceleration": {
            "method": "anderson",
            "history_depth": 4,
            "start_iteration": 3,
            "damping": 0.8,
            "regularization": 1e-10,
            "maximum_predicted_residual_ratio": 0.99,
            "maximum_step_factor": 5.0,
            "restart_residual_growth_factor": 2.0,
            "maximum_history_memory_MB": 128.0,
        },
    },
    "ase": {
        "model": "off",
        "wavelengths_m": [],
        "spectral_weights": [],
        "radiative_lifetime_s": None,
        "use_upper_state_lifetime_as_radiative": True,
        "modeled_solid_angle_fraction": 1.0,
        "face_reflectivity": {
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
            "z_min": 0.0,
            "z_max": 0.0,
        },
        "maximum_population_iterations": 12,
        "population_relaxation": 0.5,
        "maximum_boundary_iterations": 40,
        "boundary_relaxation": 0.5,
        "relative_tolerance": 1e-5,
        "maximum_intensity_W_m2": 1e18,
        "angular": {
            "direction_count": 24,
            "azimuth_offset_deg": 0.0,
            "eigenmode_iterations": 30,
            "eigenmode_tolerance": 1e-5,
            "absolute_boundary_tolerance_W_m2": 1e-12,
        },
        "failure_policy": "error",
    },
    "simulation": {
        "name": "amplifier",
        "pump_duration_s": 20e-6,
        "pump_time_steps": 40,
        "repetition_rate_Hz": 0.0,
        "pulse_train_cycles": 1,
        "pulse_train_initialization": "configured_initial_population",
        # ``None`` preserves the historical finite-train convention: the
        # first Seed pulse follows one complete inter-pulse dark window.  A
        # positive value explicitly places the first pulse that many seconds
        # after the configured initial population at t=0.
        "first_pulse_delay_s": None,
        "pump_time_steps_per_cycle": 40,
        "periodic_steady_state": {
            "enabled": False,
            "method": "pulse_map",
            "minimum_cycles": 3,
            "maximum_cycles": 1000,
            "relative_tolerance": 1e-5,
            "consecutive_cycles": 3,
            "average_minimum_iterations": 2,
            "average_maximum_iterations": 40,
            "average_relaxation": 0.5,
            "pulse_map_initialization": "auto",
            "cycle_average_initializer_minimum_lifetime_periods": 20.0,
            "maximum_population_fraction_violation": 0.05,
            "maximum_unmodeled_free_carrier_residual_fraction": 1e-4,
            "free_carrier_convergence": {
                "enabled": True,
                # ``optical_effect`` measures the cycle-to-cycle change that
                # carriers can actually impose on the pulse (FCA optical
                # depth, carrier phase, and absorbed energy).  The legacy
                # density-field max norm remains available as a diagnostic
                # and can still be selected explicitly for reference runs.
                "metric": "optical_effect",
            },
            "reduced_pulse_map": {
                "model": "phase_insensitive",
                "temporal_bins": 1,
                "spectral_bins": 33,
                "coherent_sample_count": 1024,
                "axisymmetric_radial_bins": 24,
                "axisymmetric_gain_spectral_bins": 33,
                "relative_tolerance": 1e-3,
                "consecutive_cycles": 3,
            },
            "acceleration": {
                "method": "anderson",
                "history_depth": 4,
                "start_iteration": 3,
                "damping": 0.8,
                "regularization": 1e-10,
                "maximum_predicted_residual_ratio": 0.99,
                "maximum_step_factor": 5.0,
                "restart_residual_growth_factor": 2.0,
                "maximum_history_memory_MB": 128.0,
            },
        },
        "signal_model": "auto",
        "pass_mode": "simple",
        "passes": 1,
        "between_pass_transmission": 1.0,
        "unmodeled_between_pass_delay_s": 0.0,
        "multipass": {
            "enabled": False,
            "execution_mode": "auto",
            "template": "double_pass",
            "pass_count": 2,
            "maximum_time_bins": 20000,
            "maximum_field_memory_MB": 1024.0,
            "include_dark_relaxation": True,
            "maximum_dark_step_s": 0.0,
            "maximum_relative_group_index_span": 1e-3,
            "maximum_relative_path_cosine_span": 1e-4,
            "external_refractive_index": 1.0,
            "external_group_refractive_index": 1.0,
            "per_transition_power_transmission": 1.0,
            "per_transition_extra_delay_s": 0.0,
            "double_pass": {
                "mirror_distance_m": 0.0,
                "mirror_power_reflectivity": 1.0,
                "mirror_radius_x_m": 0.0,
                "mirror_radius_y_m": 0.0,
                "mirror_center_x_m": 0.0,
                "mirror_center_y_m": 0.0,
                "mirror_aperture_radius_x_m": 0.0,
                "mirror_aperture_radius_y_m": 0.0,
                "first_incidence_angle_x_deg": 0.0,
                "first_incidence_angle_y_deg": 0.0,
                "return_incidence_angle_x_deg": 0.0,
                "return_incidence_angle_y_deg": 0.0,
                "first_center_x_m": 0.0,
                "first_center_y_m": 0.0,
                "return_center_x_m": 0.0,
                "return_center_y_m": 0.0,
            },
            "two_mirror": {
                "left_mirror_distance_m": 0.1,
                "right_mirror_distance_m": 0.1,
                "left_mirror_radius_x_m": 0.0,
                "left_mirror_radius_y_m": 0.0,
                "right_mirror_radius_x_m": 0.0,
                "right_mirror_radius_y_m": 0.0,
                "left_mirror_center_x_m": 0.0,
                "left_mirror_center_y_m": 0.0,
                "right_mirror_center_x_m": 0.0,
                "right_mirror_center_y_m": 0.0,
                "left_mirror_power_reflectivity": 1.0,
                "right_mirror_power_reflectivity": 1.0,
                "left_mirror_aperture_radius_x_m": 0.0,
                "left_mirror_aperture_radius_y_m": 0.0,
                "right_mirror_aperture_radius_x_m": 0.0,
                "right_mirror_aperture_radius_y_m": 0.0,
                "injection_center_x_m": 0.0,
                "injection_center_y_m": 0.0,
                "injection_angle_x_deg": 0.0,
                "injection_angle_y_deg": 0.0,
                "finite_pass_extraction": "ideal_switch",
            },
            "herriott": {
                "mirror_separation_m": 0.2,
                "crystal_position_from_left_m": 0.1,
                "left_mirror_radius_x_m": 0.3,
                "left_mirror_radius_y_m": 0.3,
                "right_mirror_radius_x_m": 0.3,
                "right_mirror_radius_y_m": 0.3,
                "left_mirror_center_x_m": 0.0,
                "left_mirror_center_y_m": 0.0,
                "right_mirror_center_x_m": 0.0,
                "right_mirror_center_y_m": 0.0,
                "left_mirror_power_reflectivity": 0.999,
                "right_mirror_power_reflectivity": 0.999,
                "left_mirror_aperture_radius_x_m": 0.0125,
                "left_mirror_aperture_radius_y_m": 0.0125,
                "right_mirror_aperture_radius_x_m": 0.0125,
                "right_mirror_aperture_radius_y_m": 0.0125,
                "injection_center_x_m": 1e-3,
                "injection_center_y_m": 0.0,
                "injection_angle_x_deg": 0.0,
                "injection_angle_y_deg": 0.5,
                "geometric_output_after_pass": True,
            },
            "relay_4f": {
                "object_to_first_lens_m": 0.1,
                "first_focal_length_m": 0.1,
                "lens_separation_m": 0.2,
                "second_focal_length_m": 0.1,
                "second_lens_to_mirror_m": 0.1,
                "relay_power_transmission": 1.0,
                "center_x_m": 0.0,
                "center_y_m": 0.0,
                "first_center_x_m": 0.0,
                "first_center_y_m": 0.0,
                "first_incidence_angle_x_deg": 0.0,
                "first_incidence_angle_y_deg": 0.0,
                "spot_step_x_m": 0.0,
                "spot_step_y_m": 0.0,
                "angle_step_x_deg": 0.0,
                "angle_step_y_deg": 0.0,
            },
            "prescribed": {
                "hits": [],
                "transitions": [],
            },
            # Internal landing zone for old ``passes`` configurations.  New
            # cases select one of the physical templates above; migrated
            # cases keep their useful reduced backend coverage without
            # reviving a second public multipass system.
            "sequential": {
                "alternate_direction": True,
                "flip_x": False,
                "route": [],
            },
        },
        "alternate_direction": True,
        "flip_x_between_passes": False,
        "signal_propagation": "split_step",
        "signal_volume_recording": "last_and_cumulative",
        "result_recording": copy.deepcopy(RESULT_RECORDING_DEFAULTS),
        "validation_mode": "exploratory",
        "strict_validation": {
            "pump_power_balance_relative_tolerance": 1e-8,
            "minimum_hybrid_pump_gain_entrance_capture_fraction": 0.98,
            "minimum_hybrid_seed_gain_entrance_capture_fraction": 0.98,
            "require_experimentally_calibrated_material": False,
            "seed_energy_balance_relative_tolerance": 1e-8,
            "require_named_external_losses": True,
            "maximum_pump_absorption_optical_depth": 1.0,
            "population_conservation_relative_tolerance": 1e-10,
            "ensemble_inventory_relative_tolerance": 1e-10,
            "population_correction_relative_tolerance": 1e-8,
            "grid_projection_relative_tolerance": 1e-8,
            "thermal_energy_balance_relative_tolerance": 1e-5,
            "temporal_boundary_energy_fraction": 1e-6,
            "output_spectral_boundary_energy_fraction": 1e-6,
            "ase_energy_balance_relative_tolerance": 0.05,
            "level_storage_relative_tolerance": 1e-6,
            "ase_eigenpair_relative_tolerance": 1e-5,
            "cpa_resynthesis_energy_relative_tolerance": 1e-3,
            "strong_field_fractional_loss_per_step": 0.05,
            "fem_equilibrium_relative_tolerance": 1e-6,
            "cycle_average_population_jump_relative_tolerance": 1e-2,
            "cycle_average_probe_difference_relative_tolerance": 5e-2,
            "reduced_pulse_map_population_difference_relative_tolerance": 1e-2,
            "reduced_pulse_map_output_difference_relative_tolerance": 5e-2,
            "free_carrier_periodic_residual_relative_tolerance": 1e-5,
            "cycle_time_closure_relative_tolerance": 1e-12,
            "static_material_power_out_of_range_tolerance": 0.0,
            "undepleted_pump_maximum_absorbed_fraction": 1e-3,
            "event_energy_identity_relative_tolerance": 1e-10,
            "event_energy_unresolved_fraction_tolerance": 0.0,
            "event_ase_interface_relative_tolerance": 0.05,
            "pump_component_negative_active_absorption_tolerance_m_inv": 1e-12,
            "pump_component_level_gap_relative_tolerance": 1e-3,
            "signal_event_requested_applied_relative_tolerance": 1e-3,
            "signal_event_maximum_population_change_fraction": 0.05,
            "nonlinear_fractional_depletion_step_tolerance": 0.05,
            "maximum_commercial_app_spectral_support_leakage_fraction": 0.01,
            "maximum_commercial_app_spectral_sync_error_fraction": 0.01,
            "maximum_commercial_app_reduced_spatiotemporal_B_integral_rad": 1.0,
            "maximum_commercial_app_axis_coupling_ratio": 1e-3,
            "maximum_axisymmetric_projection_loss_fraction": 1e-3,
            "maximum_axisymmetric_boundary_energy_fraction": 1e-4,
            "maximum_axisymmetric_azimuthal_residual": 5e-2,
            "maximum_axisymmetric_population_realizability_correction_fraction": (
                5e-2
            ),
            "maximum_axisymmetric_population_positivity_redistribution_fraction": 5e-2,
            "maximum_axisymmetric_diffraction_norm_correction": 1e-3,
            "maximum_axisymmetric_gain_bin_relative_frequency_span": 5e-2,
            "maximum_axisymmetric_gain_reconstruction_normalization_correction": 2e-2,
            "axisymmetric_population_mapping_relative_tolerance": 1e-10,
        },
        "save_vtk": True,
    },
    "physics": {
        "preset": "balanced",
        "optical_propagation": "auto",
        "pump_propagation": "auto",
        "population_dynamics": "auto",
        "pump_depletion": "auto",
        "gain_saturation": "auto",
        "thermal_model": "auto",
        "refractive_index_model": "auto",
        "interface_model": "auto",
        "nonlinear_phase_model": "auto",
        "nonlinear_loss_model": "auto",
        "steady_state_iterations": 6,
        "record_optical_planes": True,
        "applicability": {
            "maximum_paraxial_angle_deg": 10.0,
            "maximum_fft_transverse_nyquist_fraction": 0.8,
        },
    },
    "optical_train": {
        "seed": [{"type": "gain_medium", "name": "gain_medium"}],
        "pump": [{"type": "gain_medium", "name": "gain_medium"}],
        "between_pass": [],
    },
    "grid": {
        "size_x_m": 1e-3,
        "size_y_m": 1e-3,
        "length_z_m": 5e-3,
        "center_x_m": 0.0,
        "center_y_m": 0.0,
        "refinement_center_x_m": None,
        "refinement_center_y_m": None,
        "nx": 48,
        "ny": 48,
        "nz": 48,
        "spacing_xy": "uniform",
        "stretch_x": 4.0,
        "stretch_y": 4.0,
        "refined_region": {
            "size_x_m": 200e-6,
            "size_y_m": 200e-6,
            "nx": 24,
            "ny": 24,
        },
        "optical": {
            "enabled": False,
            "size_x_m": 1e-3,
            "size_y_m": 1e-3,
            "center_x_m": 0.0,
            "center_y_m": 0.0,
            "nx": 128,
            "ny": 128,
        },
        "pump_optical": {
            "mode": "shared",
            "size_x_m": 1e-3,
            "size_y_m": 1e-3,
            "center_x_m": 0.0,
            "center_y_m": 0.0,
            "nx": 128,
            "ny": 128,
        },
    },
    "material": {
        "name": "user_material",
        "ion_density_m3": 1e24,
        "doping_profile": {
            "enabled": False,
            "normalization": "volume_average",
            "background_fraction": 1.0,
            "minimum_raw_fraction": 0.0,
            "components": [],
            "endcaps": {
                "enabled": False,
                "front_length_m": 0.0,
                "back_length_m": 0.0,
                "relative_density": 0.0,
            },
        },
        "refractive_index": 1.8,
        "pump_refractive_index": None,
        "pump_passive_absorption_m_inv": 0.0,
        "signal_passive_loss_m_inv": 0.0,
        "dn_dT_per_K": 0.0,
        "dn_dT_definition": "measured_free_expansion",
        "reference_temperature_K": 293.15,
        "material_data_reference_temperature_K": 293.15,
        "operating_temperature_K": 293.15,
        "sellmeier_temperature_index_offset": 0.0,
        "n2_m2_W": 0.0,
        "strong_field": {
            "model": "off",
            "mechanism": "unspecified",
            "multiphoton_order": 2,
            "beta_SI": 0.0,
            "deposit_fraction_as_heat": 1.0,
            "provenance": {
                "status": "unverified",
                "reference": None,
                "sample_identifier": None,
                "reference_wavelength_m": None,
                "reference_pulse_duration_s": None,
                "bandgap_eV": None,
                "relative_uncertainty": None,
            },
            "free_carrier": {
                "model": "off",
                "lifetime_s": 1e-9,
                "generation_efficiency": 1.0,
                "absorption_cross_section_m2": 0.0,
                "refractive_index_m3": 0.0,
            },
        },
        "spectroscopic_response": {
            "mode": "off",
            "range_policy": "record",
            "data_origin": None,
            "available_data": [],
            "reference_temperature_K": 293.15,
            "reference_density_m3": 1e24,
            "reference_doping_percent": None,
            "cross_sections": {
                "pump_absorption": {"model": "off", "extras": {}},
                "pump_emission": {"model": "off", "extras": {}},
                "signal_absorption": {"model": "off", "extras": {}},
                "signal_emission": {"model": "off", "extras": {}},
            },
            "upper_state_lifetime": {
                "temperature": {"model": "off"},
                "concentration": {"model": "off"},
            },
        },
        "geometry": {
            "shape": "cuboid",
            "width_x_m": None,
            "height_y_m": None,
            "diameter_m": None,
            "center_x_m": 0.0,
            "center_y_m": 0.0,
            "clip_optical_field": True,
        },
        "field_diagnostics": {
            "enabled": False,
            "fit_radius_mode": "automatic",
            "manual_radius_x_m": None,
            "manual_radius_y_m": None,
            "beam_radius_scale": 1.0,
            "center_x_m": None,
            "center_y_m": None,
            "slice_count": 5,
            "zernike_max_radial_order": 4,
            "radial_bin_count": 48,
        },
        "dispersion": {
            "reference_wavelength_m": None,
            "gd_s_per_m": 0.0,
            "gdd_s2_per_m": 0.0,
            "tod_s3_per_m": 0.0,
            "fod_s4_per_m": 0.0,
            "sellmeier": {},
        },
        "thermal": {
            "enabled": False,
            "linear_solver": "auto",
            "cg_relative_tolerance": 1e-8,
            "algebraic_relative_tolerance": 1e-8,
            "energy_balance_relative_tolerance": 1e-10,
            "embedded_boundary_area_model": "cartesian",
            "conductivity_W_mK": [10.0, 10.0, 10.0],
            "density_kg_m3": 4000.0,
            "heat_capacity_J_kgK": 500.0,
            "entities": [],
            "contacts": [],
            "heat_source_model": "pump_component_ledger",
            "heat_fraction": "quantum_defect",
            "stokes_shift_multiplier": 1.0,
            "extra_heat_fraction": 0.0,
            "include_seed_passive_absorption_heat": False,
            "transient": {
                "enabled": False,
                "duration_s": 1.0,
                "time_steps": 20,
                "initial_temperature_K": None,
                "max_iterations_per_step": 500,
            },
            "boundary": {
                "type": "dirichlet",
                "ambient_temperature_K": 293.15,
                "coolant_temperature_K": 293.15,
                "temperature_K": 293.15,
                "heat_transfer_W_m2K": 1000.0,
                "faces": {},
            },
            "max_iterations": 3000,
            "tolerance_K": 1e-4,
            "relaxation": 1.8,
        },
        "mechanical": {
            "stress_model": "none",
            "young_modulus_Pa": 1e11,
            "poisson_ratio": 0.25,
            "thermal_expansion_per_K": 8e-6,
            "photoelastic_coefficient_Pa_inv": 0.0,
            "thermoelastic_fem": {
                "maximum_active_elements": 8000,
                "linear_solver": "auto",
                "relative_tolerance": 1e-8,
                "maximum_iterations": 2000,
                "fixed_faces": [],
                "photoelastic_coupling": "off",
            },
            "anisotropic": {
                "voigt_order": ["xx", "yy", "zz", "xy", "yz", "xz"],
                "voigt_shear_convention": "engineering_strain_tensor_stress",
                "elastic_stiffness_Pa": [],
                "thermal_expansion_tensor_per_K": [],
                "photoelastic_matrix": [],
                "photoelastic_strain_definition": "elastic_strain",
                "orientation_matrix_material_to_lab": [],
                "orientation_euler_xyz_deg": [0.0, 0.0, 0.0],
                "reference_refractive_index": None,
            },
            "endface_deformation": {
                "model": "off",
                "outside_refractive_index": 1.0,
            },
        },
        "library": {},
        "extras": {},
    },
    "pump": {
        "profile": "super_gaussian",
        "power_W": 10.0,
        "wavelength_m": 1.55e-6,
        "waist_x_m": 100e-6,
        "waist_y_m": 100e-6,
        "focus_z_m": 2.5e-3,
        "focus_x_m": None,
        "focus_y_m": None,
        "m2_x": 1.0,
        "m2_y": 1.0,
        "geometric_efficiency": 1.0,
        "source_refractive_index": 1.0,
        "source_definition": {
            "mode": "waist_m2",
            "waist_radius_x_m": 100e-6,
            "waist_radius_y_m": 100e-6,
            "divergence_half_angle_x_deg": 0.0,
            "divergence_half_angle_y_deg": 0.0,
            "emitting_radius_x_m": 100e-6,
            "emitting_radius_y_m": 100e-6,
            "numerical_aperture_x": 0.0,
            "numerical_aperture_y": 0.0,
            "free_space_refractive_index": 1.0,
            "super_gaussian_exponent_x": 2.0,
            "super_gaussian_exponent_y": 2.0,
            "derived": {},
        },
        "absorption_transport": {
            "model": "population_dependent",
            "fixed_active_coefficient_m_inv": None,
        },
        "raytrace": {
            "ray_count": 100000,
            "random_seed": 12345,
            "angular_distribution": "auto",
            "angular_super_gaussian_exponent_x": None,
            "angular_super_gaussian_exponent_y": None,
            "deposition_model": "nearest_cell_track_length",
            "thermal_refraction": True,
            "minimum_relative_ray_power": 1e-8,
            "maximum_transverse_step_fraction": 0.5,
            "maximum_substeps_per_z_cell": 16,
            "processing_chunk_size": 200000,
        },
        "order_x": 1.0,
        "order_y": 1.0,
        "center_x_m": 0.0,
        "center_y_m": 0.0,
        "rotation_deg": 0.0,
        "incidence_angle_x_deg": 0.0,
        "incidence_angle_y_deg": 0.0,
        "direction": "forward",
        "temporal": {
            "type": "cw",
            "power_mode": "average",
            "invalid_sample_policy": "error",
            "value_mode": "relative",
            "file_period_contract": "zero_outside",
            "pulse_fwhm_s": 1e-6,
            "delay_s": 0.0,
            "sample_count": 40,
            "order": 2.0,
            "samples": [],
            "time_unit": "s",
        },
        "spectrum": {
            "type": "monochromatic",
            "spectral_density_domain": "wavelength",
            "invalid_sample_policy": "error",
            "bandwidth_fwhm_m": 0.0,
            "bandwidth_fwhm_Hz": None,
            "sample_count": 1,
            "span_fwhm": 3.0,
            "order": 2.0,
            "samples": [],
            "spatial_transport": "shared",
            "wavelength_unit": "m",
        },
        "extras": {},
    },
    "seed": {
        "profile": "super_gaussian",
        "pulse_energy_J": 1e-9,
        "wavelength_m": 1.0e-6,
        "pulse_duration_s": 1e-12,
        "source_refractive_index": 1.0,
        "pulse_definition": {
            "domain": "independent",
            "fft_sample_count": 256,
            "solver_sample_count": 9,
        },
        "waist_x_m": 80e-6,
        "waist_y_m": 80e-6,
        "focus_z_m": 0.0,
        "focus_x_m": None,
        "focus_y_m": None,
        "m2_x": 1.0,
        "m2_y": 1.0,
        "spatial_modes": {
            "model": "effective_m2",
            "maximum_order_x": 4,
            "maximum_order_y": 4,
            "maximum_total_order": None,
            "maximum_propagated_modes": 64,
        },
        "spatial_gain_transport": {
            "model": "local_field",
            "record_axial_diagnostics": True,
            "fit_radius_x_m": None,
            "fit_radius_y_m": None,
            "center_x_m": 0.0,
            "center_y_m": 0.0,
            "minimum_gain_m_inv": 0.0,
        },
        "thermal_lens": {
            "model": "sampled_phase",
            "quadratic_model": "axis_aligned",
            "fit_radius_mode": "manual",
            "radius_x_m": None,
            "radius_y_m": None,
            "center_x_m": None,
            "center_y_m": None,
        },
        "tail_binning": {
            "temporal_enabled": False,
            "spectral_enabled": False,
            "energy_coverage": 0.999999,
            "temporal_tail_points": 3,
            "spectral_tail_points": 3,
        },
        "spectral_time": {
            "spatial_backend": "cartesian",
            "temporal_bins": None,
            "spectral_bins": None,
        },
        "commercial_app_fast": {
            "spatial_backend": "cartesian",
            "temporal_model": "fixed_duration",
            "include_relaxation_during_pulse": False,
            "gain_temporal_bin_count": 129,
            "carrier_periodic_relative_tolerance": 1e-6,
            "carrier_periodic_maximum_iterations": 32,
            "beam_quality_model": "configured_m2",
            "spectral_feedback": True,
            "spectral_feedback_bin_count": 65,
            "spectral_feedback_grid_mode": "full_coherent_span",
            "thermal_lens": {
                "model": "sampled_phase",
                "quadratic_model": "full_xy",
                "fit_radius_mode": "manual",
                "radius_x_m": None,
                "radius_y_m": None,
                "center_x_m": None,
                "center_y_m": None,
            },
        },
        "polarization": {
            "model": "scalar",
            "input_state": "linear",
            "linear_angle_deg": 0.0,
            "relative_phase_y_minus_x_rad": 0.0,
            "birefringence_model": "off",
            "radial_photoelastic_C": 0.0,
            "tangential_photoelastic_C": 0.0,
            "thermal_expansion_per_K": None,
            "center_x_m": 0.0,
            "center_y_m": 0.0,
        },
        "caustic_diagnostics": {
            "enabled": False,
            "start_m": 0.0,
            "stop_m": 0.1,
            "sample_count": 21,
            "positions_m": [],
            "refractive_index": 1.0,
            "record_planes": True,
        },
        "spatio_spectral": {
            "model": "shared_source",
            "out_of_range": "error",
            "reference_wavelength_m": None,
            "waist_scaling_exponent_x": 0.0,
            "waist_scaling_exponent_y": 0.0,
            "m2_scaling_exponent_x": 0.0,
            "m2_scaling_exponent_y": 0.0,
            "focus_shift_x_m_per_fraction": 0.0,
            "focus_shift_y_m_per_fraction": 0.0,
            "center_shift_x_m_per_fraction": 0.0,
            "center_shift_y_m_per_fraction": 0.0,
            "angle_shift_x_deg_per_fraction": 0.0,
            "angle_shift_y_deg_per_fraction": 0.0,
            "samples": [],
        },
        "order_x": 1.0,
        "order_y": 1.0,
        "center_x_m": 0.0,
        "center_y_m": 0.0,
        "rotation_deg": 0.0,
        "incidence_angle_x_deg": 0.0,
        "incidence_angle_y_deg": 0.0,
        "wavefront_radius_m": None,
        "temporal": {
            "type": "gaussian",
            "invalid_sample_policy": "error",
            "resolved": False,
            "sample_count": 21,
            "span_fwhm": 3.0,
            "order": 2.0,
            "samples": [],
            "time_unit": "s",
        },
        "joint_time_frequency": {
            "model": "separable",
            "weights": [],
        },
        "spectrum": {
            "type": "monochromatic",
            "spectral_density_domain": "wavelength",
            "invalid_sample_policy": "error",
            "value_mode": "relative",
            "bandwidth_fwhm_m": 0.0,
            "bandwidth_fwhm_Hz": None,
            "sample_count": 1,
            "span_fwhm": 3.0,
            "order": 2.0,
            "samples": [],
            "wavelength_unit": "m",
        },
        "coherent": {
            "failure_policy": "error",
            "input_mode": "coefficients",
            "fidelity_mode": "configured",
            "sample_count": 128,
            "span_fwhm": 6.0,
            "frequency_padding_factor": 1.0,
            "temporal_boundary_fraction": 0.05,
            "temporal_boundary_warning_fraction": 1e-6,
            "sampling_contract": {
                "minimum_spectral_fwhm_bins": 12.0,
                "minimum_input_fwhm_bins": 6.0,
                "minimum_transform_limited_fwhm_bins": 6.0,
                "maximum_spectral_boundary_energy_fraction": 1e-6,
                "maximum_input_temporal_boundary_energy_fraction": 1e-6,
                "boundary_sample_fraction": 0.05,
            },
            "axisymmetric": {
                "azimuthal_fidelity": "m0",
                "angular_samples": 12,
                # Every annulus must contain Cartesian samples; 32 leaves
                # an empty inner annulus on the default 48 x 48 grid.
                "radial_bins": 24,
                "gain_spectral_bins": 65,
                "radial_diffraction_kernel": "auto",
                "radial_numba_minimum_systems": 512,
                "radial_coefficient_cache_max_bytes": 134217728,
                "maximum_projection_loss_fraction": 1e-3,
                "maximum_boundary_energy_fraction": 1e-4,
                "maximum_azimuthal_residual": 5e-2,
                "maximum_population_realizability_correction_fraction": (
                    5e-2
                ),
                "maximum_population_positivity_redistribution_fraction": 5e-2,
                "maximum_gain_bin_relative_frequency_span": 5e-2,
                "maximum_gain_reconstruction_normalization_correction": 2e-2,
            },
            "local_operator": {
                "accuracy_profile": "reference",
                "scheme": "symmetric_adaptive",
                "adaptive_substeps": True,
                "minimum_substeps": 1,
                "maximum_substeps": 64,
                "maximum_gain_exponent_per_substep": 0.1,
                "maximum_nonlinear_phase_rad_per_substep": 0.05,
                "maximum_dispersion_phase_rad_per_substep": 0.1,
                "maximum_thermal_phase_rad_per_substep": 0.1,
                "maximum_fractional_loss_per_substep": 0.05,
                "spatial_population_update": "shared_cell",
                "multirate_gain": {
                    "enabled": False,
                    "maximum_gain_exponent_per_update": 0.05,
                    "maximum_optical_substeps_per_update": 4,
                    "maximum_updates_per_cell": None,
                },
            },
            "phase": {
                "reference_wavelength_m": None,
                "constant_phase_rad": 0.0,
                "gd_s": 0.0,
                "gdd_s2": 0.0,
                "tod_s3": 0.0,
                "fod_s4": 0.0,
            },
            "stretcher": {
                "reference_wavelength_m": None,
                "constant_phase_rad": 0.0,
                "gd_s": 0.0,
                "gdd_s2": 0.0,
                "tod_s3": 0.0,
                "fod_s4": 0.0,
            },
            "compressor": {
                "reference_wavelength_m": None,
                "constant_phase_rad": 0.0,
                "gd_s": 0.0,
                "gdd_s2": 0.0,
                "tod_s3": 0.0,
                "fod_s4": 0.0,
            },
            "complex_spectrum": {
                "format": "amplitude_phase",
                "amplitude_domain": "per_frequency",
                "wavelength_unit": "nm",
                "invalid_sample_policy": "error",
                "samples": [],
            },
            "spectral_phase": {
                "enabled": False,
                "wavelength_unit": "nm",
                "samples": [],
            },
            "phase_retrieval": {
                "max_iterations": 200,
                "tolerance": 1e-8,
            },
            "transient_gain": {
                "enabled": False,
                "method": "soft_window_spectral",
                "tail_protection_enabled": False,
                "tail_energy_coverage": 0.99,
                "slice_count": 9,
                "adaptive_slices": True,
                "adaptive_refinement_predictor": True,
                "reuse_adaptive_slice_floor": True,
                "reuse_adaptive_slice_floor_across_propagations": False,
                "reuse_scratch_buffers": True,
                "batch_window_ffts": False,
                "batch_window_fft_max_bytes": 268435456,
                "maximum_bypassed_output_energy_fraction": 0.0,
                "inactive_window_local_guard_maximum_pixels": 512,
                "fuse_single_transition_events": True,
                "spectral_coupling": "ordered_sweeps",
                "single_transition_kernel": "auto",
                "single_transition_kernel_workers": 0,
                "single_transition_numba_minimum_packet_cells": 100000,
                "spectral_material_cache_max_bytes": 536870912,
                "maximum_slice_count": 65,
                "maximum_depletion_per_slice": 0.05,
                "adaptive_depletion_metric": "total_population_change",
                "include_relaxation_during_pulse": False,
                "relaxation_energy_fraction": 0.999999,
                "failure_policy": "error",
            },
        },
        "josa": {
            "mode": "retarded_pc",
            "failure_policy": "error",
            "include_relaxation_during_pulse": False,
            "maximum_time_bins": 4096,
            "recommended_maximum_gain_step": 0.05,
        },
        "extras": {},
    },
    "energy_levels": {
        "model": "four_level",
        "initial_fractions": None,
        "pump_absorption_cross_section_m2": 1e-24,
        "pump_emission_cross_section_m2": 0.0,
        "pump_quantum_efficiency": 1.0,
        "signal_emission_cross_section_m2": 1e-23,
        "signal_absorption_cross_section_m2": 0.0,
        "upper_state_lifetime_s": 10e-6,
        "upper_state_radiative_lifetime_s": None,
        "pump_level_lifetime_s": 1e-9,
        "lower_level_lifetime_s": 1e-9,
        "gain_override_m_inv": None,
        "saturation_fluence_J_m2": None,
        "level_energies_J": [],
        "levels": [],
        "pump_transition": {},
        "signal_transition": {},
        "optical_transitions": [],
        "inhomogeneous_ensembles": {
            "enabled": False,
            "line_shape": "gaussian",
            "homogeneous_fwhm_Hz": None,
            "spectral_diffusion_rate_s": 0.0,
            "groups": [],
        },
        "pump_cross_section_spectrum": {},
        "signal_cross_section_spectrum": {},
        "decays": [],
        "nonlinear_processes": [],
        "nonlinear_solver": {
            "method": "adaptive_substeps",
            "maximum_fractional_depletion_per_substep": 0.05,
            "maximum_substeps": 256,
        },
        "linear_solver": {
            "small_matrix_kernel": "auto",
            "numba_minimum_systems": 4096,
            "maximum_numba_order": 8,
        },
        "multi_transition_solver": {
            "photon_packet_count": 32,
            "maximum_population_change_fraction_per_packet": 0.05,
        },
        "event_energy_ledger": {
            "enabled": False,
            "require_complete_partitions": True,
            "decay_partitions": [],
            "nonlinear_partitions": [],
        },
        "extras": {},
    },
    "plugins": {},
}


PHYSICS_PRESETS: dict[str, dict[str, Any]] = {
    "commercial_app_fast": {
        "optical_propagation": "abcd",
        "pump_propagation": "abcd",
        "population_dynamics": "steady_state",
        "pump_depletion": True,
        "gain_saturation": True,
        "thermal_model": "configured",
        "refractive_index_model": "thermal_3d",
        "interface_model": "coating",
        "nonlinear_phase_model": "kerr_parabolic_lens",
        "nonlinear_loss_model": "off",
    },
    "fast": {
        "optical_propagation": "abcd",
        "pump_propagation": "abcd",
        "population_dynamics": "steady_state",
        "pump_depletion": False,
        "gain_saturation": False,
        "thermal_model": "off",
        "refractive_index_model": "constant",
        "interface_model": "ideal",
        "nonlinear_phase_model": "off",
        "nonlinear_loss_model": "off",
    },
    "balanced": {
        "optical_propagation": "fft",
        # The balanced preset enables Pump depletion, so its Pump backend must
        # also close the depleted-power ledger and transport the configured
        # source-to-gain optical train.  ``prescribed_caustic`` does neither
        # when the analytic envelope changes along z; ABCD uses the
        # power-conserving slice-projection closure while retaining the fast
        # paraxial operating point expected from this preset.
        "pump_propagation": "abcd",
        "population_dynamics": "transient",
        "pump_depletion": True,
        "gain_saturation": True,
        "thermal_model": "configured",
        "refractive_index_model": "thermal_3d",
        "interface_model": "coating",
        "nonlinear_phase_model": "off",
        "nonlinear_loss_model": "off",
    },
    "detailed": {
        "optical_propagation": "fft",
        "pump_propagation": "fft",
        "population_dynamics": "transient",
        "pump_depletion": True,
        "gain_saturation": True,
        "thermal_model": "steady_3d",
        "refractive_index_model": "thermal_3d",
        "interface_model": "fresnel",
        "nonlinear_phase_model": "off",
        "nonlinear_loss_model": "off",
    },
}


_ENERGY_DESTINATION_SCHEMA = {
    "heat": None,
    "fluorescence_escape": None,
    "ase_source": None,
    "unresolved": None,
}

_OPTICAL_ELEMENT_ITEM_SCHEMA: dict[str, Any] = {
    "type": None,
    "name": None,
    "distance_before_m": None,
    "record": False,
    "length_m": None,
    "focal_length_m": None,
    "focal_length_x_m": None,
    "focal_length_y_m": None,
    "refractive_index": None,
    "group_refractive_index": None,
    "group_refractive_index_after": None,
    "thickness_m": None,
    "n_after": None,
    "outside_refractive_index": None,
    "radius_front_m": None,
    "radius_back_m": None,
    "radius_front_x_m": None,
    "radius_front_y_m": None,
    "radius_back_x_m": None,
    "radius_back_y_m": None,
    "radius_x_m": None,
    "radius_y_m": None,
    "center_x_m": None,
    "center_y_m": None,
    "shift_x_m": None,
    "shift_y_m": None,
    "angle_x_deg": None,
    "angle_y_deg": None,
    "axis_angle_x_deg": None,
    "axis_angle_y_deg": None,
    "turn_angle_x_deg": None,
    "turn_angle_y_deg": None,
    "incidence_angle_deg": None,
    "incidence_angle_x_deg": None,
    "incidence_angle_y_deg": None,
    "front_incidence_angle_deg": None,
    "back_incidence_angle_deg": None,
    "direction": None,
    "model": None,
    "front_model": None,
    "back_model": None,
    "polarization": None,
    "power_transmission": None,
    "power_reflectivity": None,
    "front_power_transmission": None,
    "back_power_transmission": None,
    "absorption_m_inv": None,
    "shape": None,
    "aperture_radius_m": None,
    "aperture_radius_x_m": None,
    "aperture_radius_y_m": None,
    "aperture_width_m": None,
    "aperture_height_m": None,
    "order": None,
    "constant_phase_rad": None,
    "quadratic_x_rad_m2": None,
    "quadratic_y_rad_m2": None,
    "group_delay_s": None,
    "flip_x": False,
    "flip_y": False,
    "object_to_first_lens_m": None,
    "first_focal_length_m": None,
    "lens_separation_m": None,
    "second_focal_length_m": None,
    "second_lens_to_mirror_m": None,
    "dispersion": {
        "constant_phase_rad": None,
        "reference_frequency_Hz": None,
        "reference_wavelength_m": None,
        "gd_s": None,
        "gdd_s2": None,
        "tod_s3": None,
        "fod_s4": None,
        "extras": {},
    },
    "spectral_response": {
        "wavelength_m": None,
        "power_transmission": None,
        "power_reflectivity": None,
        "front_power_transmission": None,
        "back_power_transmission": None,
        "out_of_range": None,
        "extras": {},
    },
    "extras": {},
}

_LIST_ITEM_SCHEMAS: dict[str, dict[str, Any]] = {
    "optical_train.seed": _OPTICAL_ELEMENT_ITEM_SCHEMA,
    "optical_train.pump": _OPTICAL_ELEMENT_ITEM_SCHEMA,
    "optical_train.between_pass": _OPTICAL_ELEMENT_ITEM_SCHEMA,
    "simulation.multipass.prescribed.hits": _OPTICAL_ELEMENT_ITEM_SCHEMA,
    "simulation.multipass.prescribed.transitions": {
        "name": None,
        "route_model": None,
        "geometric_length_m": None,
        "refractive_index": None,
        "group_refractive_index": None,
        "power_transmission": None,
        "extra_delay_s": None,
        "mirror_power_reflectivity": None,
        "mirror_radius_x_m": None,
        "mirror_radius_y_m": None,
        "mirror_center_x_m": None,
        "mirror_center_y_m": None,
        "mirror_aperture_radius_x_m": None,
        "mirror_aperture_radius_y_m": None,
        "flip_x": False,
        "flip_y": False,
        "route": [],
        "extras": {},
    },
    "simulation.multipass.prescribed.transitions.route": _OPTICAL_ELEMENT_ITEM_SCHEMA,
    "simulation.multipass.sequential.route": _OPTICAL_ELEMENT_ITEM_SCHEMA,
    "material.thermal.entities": {
        "name": None,
        "region": {
            "shape": None,
            "x_min_m": None,
            "x_max_m": None,
            "y_min_m": None,
            "y_max_m": None,
            "z_min_m": None,
            "z_max_m": None,
            "center_x_m": None,
            "center_y_m": None,
            "diameter_m": None,
            "exclude_material_mask": False,
            "extras": {},
        },
        "conductivity_W_mK": None,
        "density_kg_m3": None,
        "heat_capacity_J_kgK": None,
        "extras": {},
    },
    "material.thermal.contacts": {
        "entity_a": None,
        "entity_b": None,
        "thermal_contact_resistance_m2K_W": None,
        "extras": {},
    },
    "energy_levels.optical_transitions": {
        "name": None,
        "field": None,
        "lower": None,
        "upper": None,
        "sigma_abs_m2": None,
        "sigma_em_m2": None,
        "quantum_efficiency": None,
        "cross_section_spectrum": {},
        "center_wavelength_m": None,
        "homogeneous_fwhm_Hz": None,
        "line_shape": None,
        "absorption_scale": None,
        "emission_scale": None,
        "ensemble": None,
        "extras": {},
    },
    "energy_levels.inhomogeneous_ensembles.groups": {
        "name": None,
        "fraction": None,
        "signal_center_wavelength_m": None,
        "pump_center_wavelength_m": None,
        "homogeneous_fwhm_Hz": None,
        "line_shape": None,
        "pump_absorption_scale": None,
        "pump_emission_scale": None,
        "signal_absorption_scale": None,
        "signal_emission_scale": None,
        "extras": {},
    },
    "energy_levels.decays": {
        "from": None,
        "to": None,
        "partial_rate_s": None,
        "partial_lifetime_s": None,
        "total_lifetime_s": None,
        "branching_fraction": None,
        "scale_with_upper_state_lifetime": False,
        "extras": {},
    },
    "energy_levels.nonlinear_processes": {
        "name": None,
        "rate_coefficient": None,
        "reactants": {},
        "products": {},
        "extras": {},
    },
    "energy_levels.event_energy_ledger.decay_partitions": {
        "from": None,
        "to": None,
        "model": None,
        "radiative_destination": None,
        "fractions": _ENERGY_DESTINATION_SCHEMA,
        **_ENERGY_DESTINATION_SCHEMA,
        "extras": {},
    },
    "energy_levels.event_energy_ledger.nonlinear_partitions": {
        "process": None,
        "process_index": None,
        "fractions": _ENERGY_DESTINATION_SCHEMA,
        **_ENERGY_DESTINATION_SCHEMA,
        "extras": {},
    },
}


def _normalize_null_optical_element_fields(config: dict[str, Any]) -> None:
    """Treat schema-declared optional ``null`` fields as being omitted.

    Optical elements are typed list records rather than dictionaries merged
    with per-element defaults.  UI table editors and JSON generators commonly
    retain blank cells as explicit ``null`` values, while the propagation
    backends use an absent key to select their documented default.  Removing
    only known schema fields keeps those two representations equivalent without
    touching plugin metadata, where ``null`` may have application-specific
    meaning.  ``type`` remains present so a null/missing element type is still
    rejected by validation rather than silently normalized.
    """

    element_lists: list[Any] = []
    optical_train = config.get("optical_train")
    if isinstance(optical_train, dict):
        element_lists.extend(
            optical_train.get(name, [])
            for name in ("seed", "pump", "between_pass")
        )
    simulation = config.get("simulation", {})
    multipass = simulation.get("multipass", {}) if isinstance(simulation, dict) else {}
    if isinstance(multipass, dict):
        sequential = multipass.get("sequential", {})
        if isinstance(sequential, dict):
            element_lists.append(sequential.get("route", []))
        prescribed = multipass.get("prescribed", {})
        transitions = prescribed.get("transitions", []) if isinstance(prescribed, dict) else []
        if isinstance(transitions, list):
            element_lists.extend(
                transition.get("route", [])
                for transition in transitions
                if isinstance(transition, dict)
            )
    optional_keys = set(_OPTICAL_ELEMENT_ITEM_SCHEMA) - {"type"}
    for elements in element_lists:
        if not isinstance(elements, list):
            continue
        for element in elements:
            if not isinstance(element, dict):
                continue
            for key in optional_keys.intersection(element):
                if element[key] is None:
                    element.pop(key)

_OPTIONAL_FINITE_NUMBER_PATHS = {
    "ase.radiative_lifetime_s",
    "simulation.first_pulse_delay_s",
    "grid.refinement_center_x_m",
    "grid.refinement_center_y_m",
    "material.pump_refractive_index",
    "material.spectroscopic_response.reference_doping_percent",
    "material.geometry.width_x_m",
    "material.geometry.height_y_m",
    "material.geometry.diameter_m",
    "material.field_diagnostics.manual_radius_x_m",
    "material.field_diagnostics.manual_radius_y_m",
    "material.field_diagnostics.center_x_m",
    "material.field_diagnostics.center_y_m",
    "material.dispersion.reference_wavelength_m",
    "material.thermal.transient.initial_temperature_K",
    "material.mechanical.anisotropic.reference_refractive_index",
    "pump.focus_x_m",
    "pump.focus_y_m",
    "seed.focus_x_m",
    "seed.focus_y_m",
    "seed.polarization.thermal_expansion_per_K",
    "seed.spatio_spectral.reference_wavelength_m",
    "seed.wavefront_radius_m",
    "seed.coherent.phase.reference_wavelength_m",
    "seed.coherent.stretcher.reference_wavelength_m",
    "seed.coherent.compressor.reference_wavelength_m",
    "energy_levels.gain_override_m_inv",
    "energy_levels.saturation_fluence_J_m2",
    "energy_levels.inhomogeneous_ensembles.homogeneous_fwhm_Hz",
}

_OPTIONAL_STRING_PATHS = {
    "material.spectroscopic_response.data_origin",
}

_OPTIONAL_LIST_PATHS = {
    "energy_levels.initial_fractions",
}

_AUTO_NONSTRING_PATHS = {
    "physics.pump_depletion",
    "physics.gain_saturation",
}

_STRING_OR_FINITE_NUMBER_PATHS = {
    "material.thermal.heat_fraction",
}


def resolve_physics(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve a named accuracy preset and retain every explicit override."""
    supplied = config.get("physics", {})
    preset = str(supplied.get("preset", "balanced"))
    if preset not in PHYSICS_PRESETS:
        raise ValueError(f"physics.preset must be one of {sorted(PHYSICS_PRESETS)}")
    resolved = copy.deepcopy(PHYSICS_PRESETS[preset])
    for key, value in supplied.items():
        if key != "preset" and value != "auto":
            resolved[key] = copy.deepcopy(value)
    if (
        str(config.get("simulation", {}).get("signal_model", "auto")) == "josa_transport"
        and str(config.get("seed", {}).get("josa", {}).get("mode", "retarded_pc"))
        == JOSA_FULL_3D_MODE
    ):
        # This is an explicit solver contract rather than another vague UI
        # quality label.  JSON and Streamlit runs therefore resolve identically.
        resolved.update(josa_full_3d_physics_overrides(supplied))
    resolved["preset"] = preset
    resolved.setdefault("steady_state_iterations", 6)
    resolved.setdefault("record_optical_planes", True)
    return resolved


def resolve_thermal_model(config: dict[str, Any]) -> str:
    """Resolve the material-owned thermal selector to the model that will run."""
    model = str(resolve_physics(config)["thermal_model"])
    if model != "configured":
        return model
    thermal = config.get("material", {}).get("thermal", {})
    if not bool(thermal.get("enabled", False)):
        return "off"
    return "transient_3d" if bool(thermal.get("transient", {}).get("enabled", False)) else "steady_3d"


def resolve_population_dynamics(config: dict[str, Any]) -> str:
    """Describe the population path dispatched by AmplifierSimulator.

    ``resolve_physics`` supplies the local rate-equation mode. The simulation
    scheduler can wrap it in a periodic fixed point or a finite pulse train.
    Keep that distinction visible before running as well as in result stats.
    """

    simulation = config["simulation"]
    repetition_rate = float(simulation.get("repetition_rate_Hz", 0.0))
    periodic = simulation.get("periodic_steady_state", {})
    if repetition_rate > 0.0 and bool(periodic.get("enabled", False)):
        return (
            "cycle_averaged_steady_state"
            if periodic.get("method", "pulse_map") == "cycle_average"
            else "periodic_steady_state"
        )
    local_mode = str(resolve_physics(config)["population_dynamics"])
    if local_mode == "transient":
        if repetition_rate > 0.0:
            return "transient_pulse_train"
        if (
            config["pump"].get("temporal", {}).get("type", "cw") != "cw"
            and float(simulation.get("pump_duration_s", 0.0)) > 0.0
        ):
            return "transient_waveform"
    return local_mode


def resolve_signal_model(config: dict[str, Any]) -> str:
    """Resolve ``simulation.signal_model=auto`` in one shared location.

    The Streamlit preview and the runtime must make the same decision.  Keeping
    the rule here avoids a UI that says ``fluence`` while the solver silently
    selects ``spectral_time`` after a waveform or spectrum file is supplied.
    """

    requested = str(config.get("simulation", {}).get("signal_model", "auto"))
    if requested != "auto":
        return requested
    if str(config.get("physics", {}).get("preset", "balanced")) == "commercial_app_fast":
        return "commercial_app_fast"
    seed = config.get("seed", {})
    if (
        str(seed.get("spatial_modes", {}).get("model", "effective_m2"))
        == "incoherent_hg"
    ):
        # The independently propagated modal channels use the spectral/time
        # field representation even for a monochromatic unresolved pulse.
        # Never let ``auto`` label this path as the whole-fluence backend.
        return "spectral_time"
    spectrum_type = str(seed.get("spectrum", {}).get("type", "monochromatic"))
    temporal = seed.get("temporal", {})
    time_resolved = bool(temporal.get("resolved", False)) or str(
        temporal.get("type", "gaussian")
    ) == "file"
    linked_domain = str(seed.get("pulse_definition", {}).get("domain", "independent"))
    return (
        "spectral_time"
        if spectrum_type != "monochromatic"
        or time_resolved
        or linked_domain in {"time", "spectrum", "dual"}
        else "fluence"
    )


def apply_commercial_app_fast_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy configured for the explicit Commercial App-style fast contract.

    The helper changes solver/coupling choices only.  It deliberately leaves
    the user's material, grid, beam, spectrum, and thermal boundary data
    untouched so that a fast/high-fidelity comparison uses the same device.
    """

    result = copy.deepcopy(config)
    simulation = result.setdefault("simulation", {})
    simulation["signal_model"] = "commercial_app_fast"
    simulation["signal_propagation"] = "split_step"
    periodic = simulation.setdefault("periodic_steady_state", {})
    repetition_rate = float(simulation.get("repetition_rate_Hz", 0.0))
    periodic["enabled"] = repetition_rate > 0.0
    periodic["method"] = "cycle_average"

    physics = result.setdefault("physics", {})
    physics["preset"] = "commercial_app_fast"
    for key in PHYSICS_PRESETS["commercial_app_fast"]:
        physics[key] = "auto"
    physics["record_optical_planes"] = False

    multiphysics = result.setdefault("multiphysics", {})
    multiphysics["coupling_model"] = "one_way"
    multiphysics["seed_heat_coupling"] = "off"
    seed = result.setdefault("seed", {})
    seed.setdefault("spatial_gain_transport", {})["model"] = "local_field"
    seed.setdefault("commercial_app_fast", {}).update(
        {
            "temporal_model": "fixed_duration",
            "beam_quality_model": "configured_m2",
            "spectral_feedback": True,
        }
    )
    seed.setdefault("caustic_diagnostics", {})["enabled"] = False
    return result


def apply_commercial_app_seed_only_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Return an apples-to-apples Commercial App Seed reduction of ``config``.

    Unlike :func:`apply_commercial_app_fast_profile`, this helper deliberately preserves
    the Pump solver, population dynamics, periodic pulse map, thermal model,
    thermal-optical coupling, Seed heat coupling, grids, and optical-plane
    recording.  Only the Seed field representation is changed: transverse
    diffraction becomes a Gaussian-q duct, Kerr lensing uses the parabolic
    closure, and a separate one-dimensional coherent envelope carries
    effective-area SPM and output pulse diagnostics.
    """

    result = copy.deepcopy(config)
    simulation = result.setdefault("simulation", {})
    simulation["signal_model"] = "commercial_app_fast"
    simulation["signal_propagation"] = "split_step"

    physics = result.setdefault("physics", {})
    physics["optical_propagation"] = "abcd"
    physics["nonlinear_phase_model"] = "kerr_parabolic_lens"
    physics["nonlinear_loss_model"] = "off"

    seed = result.setdefault("seed", {})
    seed.setdefault("spatial_gain_transport", {})["model"] = "local_field"
    commercial_app = seed.setdefault("commercial_app_fast", {})
    commercial_app["temporal_model"] = (
        "fixed_duration"
        if str(seed.get("spatial_modes", {}).get("model", "effective_m2"))
        == "incoherent_hg"
        else "effective_area_spm"
    )
    commercial_app["beam_quality_model"] = "projection_residual"
    commercial_app["spectral_feedback"] = True
    commercial_app["spectral_feedback_bin_count"] = max(
        int(commercial_app.get("spectral_feedback_bin_count", 65)),
        65,
    )
    caustic = seed.setdefault("caustic_diagnostics", {})
    caustic["enabled"] = True
    caustic["record_planes"] = True
    return result


def apply_commercial_app_effective_area_strong_field_profile(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return a fast Commercial App effective-area 7PA/free-carrier screening profile.

    This is an explicit reduced-order ``L3.5`` contract.  It keeps the Commercial App
    Gaussian-q transverse transport and one coherent temporal envelope, then
    closes multiphoton absorption and the causal carrier rate equation with a
    Gaussian effective-area moment.  It deliberately does not change the
    meaning of the existing L2/L3 profiles, whose irreversible nonlinear loss
    remains disabled.

    The fast profile uses a cycle-averaged gain population while solving the
    local inter-pulse carrier fixed point analytically/iteratively inside the
    one-dimensional strong-field operator.  It is intended for trend scans;
    full x-y-t/r-t validation still belongs to coherent L4/L4.5.
    """

    result = apply_commercial_app_fast_profile(config)
    simulation = result.setdefault("simulation", {})
    periodic = simulation.setdefault("periodic_steady_state", {})
    periodic["method"] = "cycle_average"

    physics = result.setdefault("physics", {})
    physics["optical_propagation"] = "abcd"
    physics["nonlinear_phase_model"] = "kerr_parabolic_lens"
    physics["nonlinear_loss_model"] = "strong_field_effective_area"

    seed = result.setdefault("seed", {})
    seed.setdefault("spatial_gain_transport", {})["model"] = "local_field"
    commercial_app = seed.setdefault("commercial_app_fast", {})
    commercial_app.update(
        {
            "temporal_model": "effective_area_spm",
            "include_relaxation_during_pulse": False,
            "beam_quality_model": "projection_residual",
            "spectral_feedback": True,
        }
    )
    commercial_app["spectral_feedback_bin_count"] = max(
        int(commercial_app.get("spectral_feedback_bin_count", 65)), 65
    )

    strong_field = result.setdefault("material", {}).setdefault(
        "strong_field", {}
    )
    strong_field["model"] = "multiphoton"
    return result


def apply_josa_transport_profile(
    config: dict[str, Any],
    mode: str = "retarded_pc",
    *,
    periodic_method: str | None = None,
) -> dict[str, Any]:
    """Return a copy with a physically closed JOSA gain-transport contract.

    This helper intentionally does not pretend that wavelength bins are a
    mutually coherent spectrum.  Source/stretcher phase may still be projected
    once into the *input temporal intensity* by the shared Fourier pulse
    builder, but JOSA itself transports photon fluence and populations.  Kerr,
    SPM, material/train phase accumulation, and compressor prediction are
    therefore disabled rather than silently ignored.

    Pump, material, beam, grid, optical train, and (except for the explicit
    full-3D mode) thermal choices are otherwise preserved.  The returned
    profile uses real level populations, commits stimulated depletion, and
    never silently falls back to Frantz--Nodvik.
    """

    selected_mode = str(mode)
    if selected_mode not in JOSA_MODES:
        raise ValueError(f"Unsupported JOSA transport mode {selected_mode!r}")
    if periodic_method not in {None, "cycle_average", "pulse_map"}:
        raise ValueError(
            "JOSA periodic_method must be cycle_average, pulse_map, or None"
        )

    result = copy.deepcopy(config)
    simulation = result.setdefault("simulation", {})
    simulation["signal_model"] = "josa_transport"
    simulation["signal_propagation"] = "split_step"
    periodic = simulation.setdefault("periodic_steady_state", {})
    repetition_rate = float(simulation.get("repetition_rate_Hz", 0.0))
    periodic["enabled"] = repetition_rate > 0.0
    selected_periodic = str(
        periodic_method or periodic.get("method", "pulse_map")
    )
    # reduced_pulse_map is a coherent-CPA proxy contract.  JOSA already is the
    # ordered pulse operator, so its discrete fixed point must call the real
    # JOSA pulse rather than a second reduced proxy.  Store the compatible
    # method even when repetition_rate_Hz=0 so enabling a train later cannot
    # resurrect the coherent-only proxy.
    if selected_periodic == "reduced_pulse_map":
        selected_periodic = "pulse_map"
    periodic["method"] = selected_periodic
    if str(periodic.get("pulse_map_initialization", "auto")) == (
        "reduced_pulse_map"
    ):
        periodic["pulse_map_initialization"] = "cycle_average"

    physics = result.setdefault("physics", {})
    physics["gain_saturation"] = True
    physics["nonlinear_phase_model"] = "off"
    physics["nonlinear_loss_model"] = "off"
    if periodic.get("enabled", False):
        physics["population_dynamics"] = (
            "transient"
            if str(periodic.get("method", "pulse_map")) == "pulse_map"
            else "steady_state"
        )

    seed = result.setdefault("seed", {})
    seed.setdefault("spatial_gain_transport", {})["model"] = "local_field"
    if (
        str(
            seed.setdefault("spatial_modes", {}).get(
                "model", "effective_m2"
            )
        )
        == "incoherent_hg"
    ):
        seed["spatial_modes"]["model"] = "effective_m2"
    seed.setdefault("josa", {}).update(
        {
            "mode": selected_mode,
            "failure_policy": "error",
        }
    )
    result.setdefault("energy_levels", {})["gain_override_m_inv"] = None

    if selected_mode == JOSA_FULL_3D_MODE:
        physics.update(josa_full_3d_physics_overrides(physics))
    return result


def apply_josa_fidelity_profile(
    config: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    """Apply one shared J1--J4 JOSA fidelity profile used by JSON and UI."""

    profile_name = str(profile)
    try:
        info = JOSA_FIDELITY_INFO[profile_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported JOSA fidelity profile {profile_name!r}") from exc

    result = apply_josa_transport_profile(
        config,
        str(info["mode"]),
        periodic_method=str(info["periodic_method"]),
    )
    physics = result["physics"]
    stack = str(info["stack"])
    if stack == "transport_only":
        physics.update(
            {
                "optical_propagation": "off",
                "pump_propagation": "abcd",
                "population_dynamics": "steady_state",
                "pump_depletion": True,
                "gain_saturation": True,
                "thermal_model": "off",
                "refractive_index_model": "constant",
                "nonlinear_phase_model": "off",
                "nonlinear_loss_model": "off",
                "record_optical_planes": False,
            }
        )
        result.setdefault("multiphysics", {})["coupling_model"] = "one_way"
        result["multiphysics"]["seed_heat_coupling"] = "off"
        result["seed"].setdefault("caustic_diagnostics", {})["enabled"] = False
    elif stack == "spatial_operating_point":
        thermal_enabled = bool(
            result.get("material", {}).get("thermal", {}).get("enabled", False)
        )
        physics.update(
            {
                "optical_propagation": "hybrid_abcd_fft",
                "pump_propagation": "abcd",
                "population_dynamics": "transient",
                "pump_depletion": True,
                "gain_saturation": True,
                "thermal_model": "configured" if thermal_enabled else "off",
                "refractive_index_model": (
                    "thermal_3d" if thermal_enabled else "constant"
                ),
                "nonlinear_phase_model": "off",
                "nonlinear_loss_model": "off",
                "record_optical_planes": True,
            }
        )
        result.setdefault("multiphysics", {})["coupling_model"] = "one_way"
        result["multiphysics"]["seed_heat_coupling"] = "off"
    elif stack == "full_3d":
        physics.update(josa_full_3d_physics_overrides(physics))
    else:  # pragma: no cover - guarded by module-owned metadata
        raise RuntimeError(f"Unknown JOSA profile stack {stack!r}")
    return result


def apply_coherent_kerr_hybrid_profile(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return a faster full-field Kerr profile with reduced slow/gain closure.

    This is intentionally a profile of ``coherent_cpa``, not a new optical
    backend.  The propagated object remains the complex E(f, y, x) field, so
    diffraction, dispersion, spatially varying Kerr phase, and interference
    are retained.  Runtime is reduced by using a low-dimensional
    discrete-pulse map to initialize the exact configured pulse-map fixed
    point and by using a gain-exponent-bounded multirate schedule for
    whole-fluence FN updates instead of sequential soft temporal windows at
    every fine Kerr/dispersion step.

    Material, beam, grid, optical-train, thermal, and interface settings are
    left untouched.  The result records the approximation explicitly through
    ``seed.coherent.fidelity_mode``.
    """

    result = copy.deepcopy(config)
    simulation = result.setdefault("simulation", {})
    simulation["signal_model"] = "coherent_cpa"
    simulation["signal_propagation"] = "split_step"
    periodic = simulation.setdefault("periodic_steady_state", {})
    repetition_rate = float(simulation.get("repetition_rate_Hz", 0.0))
    periodic["enabled"] = repetition_rate > 0.0
    if repetition_rate > 0.0:
        periodic["method"] = "pulse_map"
        periodic["pulse_map_initialization"] = "reduced_pulse_map"
        reduced_map = periodic.setdefault("reduced_pulse_map", {})
        # A profile must fully define its approximation instead of inheriting
        # an incompatible proxy selected by the previous form state.  In
        # particular, retaining axisymmetric_coherent_kerr while switching the
        # reported solver to hybrid_whole_fluence violates the validation
        # contract and makes the preset button produce an unloadable config.
        reduced_map["model"] = "phase_insensitive"
        reduced_map["temporal_bins"] = 1
        reduced_map["spectral_bins"] = max(
            int(reduced_map.get("spectral_bins", 33)),
            33,
        )
        # A full-field defect check follows the proxy solve.  Driving the
        # proxy orders of magnitude below that independent gate only wastes
        # Pump-period evaluations.
        reduced_map["relative_tolerance"] = min(
            max(float(reduced_map.get("relative_tolerance", 1e-3)), 1e-6),
            1e-3,
        )
        reduced_map["consecutive_cycles"] = max(
            int(reduced_map.get("consecutive_cycles", 3)),
            2,
        )

    physics = result.setdefault("physics", {})
    # Use analytic ABCD transport only up to the sampled gain entrance.  A
    # source-plane-only FFT grid can severely truncate a large collimated beam
    # even when the focused beam fits comfortably inside the crystal window.
    # The field is sampled once at the gain entrance and remains a full local
    # complex field throughout the nonlinear crystal; there is no repeated
    # Gaussian projection inside the gain medium.
    physics["optical_propagation"] = "hybrid_abcd_fft"
    physics["nonlinear_phase_model"] = "kerr_time_domain"

    seed = result.setdefault("seed", {})
    seed.setdefault("spatial_gain_transport", {})["model"] = "local_field"
    coherent = seed.setdefault("coherent", {})
    coherent["failure_policy"] = "error"
    coherent["fidelity_mode"] = "hybrid_whole_fluence"
    local_operator = coherent.setdefault("local_operator", {})
    local_operator["scheme"] = "symmetric_adaptive"
    multirate = local_operator.setdefault("multirate_gain", {})
    multirate["enabled"] = True
    multirate["maximum_optical_substeps_per_update"] = max(
        int(multirate.get("maximum_optical_substeps_per_update", 4)),
        8,
    )
    coherent.setdefault("transient_gain", {})["enabled"] = False
    coherent["transient_gain"]["failure_policy"] = "error"
    return result


def apply_axisymmetric_transient_gain_profile(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return the L4.5 ``E(f,r)`` Kerr + front-to-tail gain profile.

    The material, source, grids, optical train, thermal boundary conditions,
    and coherent sample count are preserved.  Only the solver closure is made
    explicit: the crystal field uses the guarded axisymmetric reduction while
    every reported/periodic Seed pulse uses the native-frequency soft-window
    gain operator in front-to-peak-to-tail order.
    """

    result = copy.deepcopy(config)
    simulation = result.setdefault("simulation", {})
    simulation["signal_model"] = "coherent_cpa"
    simulation["signal_propagation"] = "split_step"
    periodic = simulation.setdefault("periodic_steady_state", {})
    repetition_rate = float(simulation.get("repetition_rate_Hz", 0.0))
    periodic["enabled"] = repetition_rate > 0.0
    if repetition_rate > 0.0:
        periodic["method"] = "pulse_map"

    physics = result.setdefault("physics", {})
    physics["optical_propagation"] = "hybrid_abcd_fft"
    physics["nonlinear_phase_model"] = "kerr_time_domain"
    if repetition_rate > 0.0:
        physics["population_dynamics"] = "transient"

    seed = result.setdefault("seed", {})
    seed.setdefault("spatial_gain_transport", {})["model"] = "local_field"
    coherent = seed.setdefault("coherent", {})
    coherent["failure_policy"] = "error"
    coherent["fidelity_mode"] = "axisymmetric_transient_gain"
    local_operator = coherent.setdefault("local_operator", {})
    local_operator["accuracy_profile"] = "reference"
    local_operator["scheme"] = "symmetric_adaptive"
    local_operator["spatial_population_update"] = "shared_cell"
    local_operator.setdefault("multirate_gain", {})["maximum_updates_per_cell"] = None
    local_operator.setdefault("multirate_gain", {})["enabled"] = False
    transient = coherent.setdefault("transient_gain", {})
    transient.update(
        {
            "enabled": True,
            "method": "soft_window_spectral",
            "batch_window_ffts": True,
            "batch_window_fft_max_bytes": 268435456,
            "single_transition_kernel": "auto",
            "spectral_coupling": "ordered_sweeps",
            "single_transition_numba_minimum_packet_cells": 100000,
            "failure_policy": "error",
        }
    )
    return result


def apply_axisymmetric_transient_gain_cpu_fast_profile(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return the measured L4.5 CPU-fast profile without spectral grouping.

    The native coherent frequency grid, soft-window count, and front-to-tail
    population order are unchanged. The profile combines guarded multirate
    gain with relaxed exact-linear phase split heuristics at the tested
    four-substep scale and disables the optional output caustic diagnostic.
    It remains a numerical approximation that must be checked against the
    reference profile for materially different beams or thermal lenses.
    """

    result = apply_axisymmetric_transient_gain_profile(config)
    coherent = result["seed"]["coherent"]
    local_operator = coherent["local_operator"]
    local_operator["accuracy_profile"] = "cpu_fast_convergence_required"
    local_operator["maximum_dispersion_phase_rad_per_substep"] = max(
        float(
            local_operator.get(
                "maximum_dispersion_phase_rad_per_substep", 0.1
            )
        ),
        8.0,
    )
    local_operator["maximum_thermal_phase_rad_per_substep"] = max(
        float(
            local_operator.get(
                "maximum_thermal_phase_rad_per_substep", 0.1
            )
        ),
        0.2,
    )
    multirate = local_operator.setdefault("multirate_gain", {})
    multirate["enabled"] = True
    multirate["maximum_gain_exponent_per_update"] = min(
        float(multirate.get("maximum_gain_exponent_per_update", 0.03)),
        0.03,
    )
    # This preset is validated at a maximum group of eight.  Do not inherit a
    # looser user value (for example 32 or 64) and silently claim the measured
    # CPU-fast accuracy envelope for an untested approximation.
    multirate["maximum_optical_substeps_per_update"] = 8
    coherent["transient_gain"]["spectral_coupling"] = "ordered_sweeps"
    coherent["transient_gain"][
        "maximum_bypassed_output_energy_fraction"
    ] = 1e-6
    coherent["transient_gain"][
        "inactive_window_local_guard_maximum_pixels"
    ] = 512
    coherent["transient_gain"][
        "reuse_adaptive_slice_floor_across_propagations"
    ] = True
    result["seed"].setdefault("caustic_diagnostics", {})["enabled"] = False
    result["seed"].setdefault("extras", {})["cpu_fast_profile"] = {
        "name": "axisymmetric_transient_gain_cpu_fast",
        "reference_required": True,
        "native_gain_frequency_grid_retained": True,
        "soft_window_count_retained": True,
        "transient_gain_multirate_enabled": True,
        "broadband_population_coupling_enabled": False,
        "broadband_population_coupling_available_experimental": True,
        "inactive_window_output_energy_budget_per_gain_step": 1e-6,
        "inactive_window_local_energy_guard": True,
        "inactive_window_local_guard_maximum_pixels": 512,
        "inactive_window_bypass_has_coupling_fallbacks": True,
        "multirate_gain_reference_required": True,
        "output_caustic_disabled": True,
    }
    return result


def _merge(default: Any, supplied: Any) -> Any:
    if isinstance(default, dict) and isinstance(supplied, dict):
        result = copy.deepcopy(default)
        for key, value in supplied.items():
            result[key] = _merge(result[key], value) if key in result else copy.deepcopy(value)
        return result
    return copy.deepcopy(supplied)


def _validate_core_schema(
    supplied: Any,
    schema: Any,
    *,
    path: str = "",
) -> None:
    """Reject misspelled keys in default-backed core namespaces.

    Empty dictionaries remain open for tabulated/plugin data. Core typed list
    records use explicit per-item schemas; extension metadata belongs under
    ``extras`` or ``plugins``.
    """

    if isinstance(schema, bool):
        if not isinstance(supplied, bool):
            raise ValueError(
                f"{path} must be a JSON boolean"
            )
        return
    if isinstance(schema, int) and not isinstance(schema, bool):
        require_exact_integer(supplied, name=path)
        return
    if isinstance(schema, float):
        if isinstance(supplied, bool):
            raise ValueError(
                f"{path} must be a finite number, "
                "not a boolean"
            )
        try:
            numeric = float(supplied)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path} must be a finite number"
            ) from exc
        if not math.isfinite(numeric):
            if path.endswith((".size_x_m", ".size_y_m", ".length_z_m")):
                raise ValueError(f"{path} must be positive and finite")
            raise ValueError(
                f"{path} must be finite"
            )
        return
    if isinstance(schema, str):
        if path in _STRING_OR_FINITE_NUMBER_PATHS:
            if isinstance(supplied, str):
                return
            if isinstance(supplied, bool):
                raise ValueError(f"{path} must be a named model or finite number")
            try:
                numeric = float(supplied)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path} must be a named model or finite number"
                ) from exc
            if not math.isfinite(numeric):
                raise ValueError(f"{path} must be a named model or finite number")
            return
        # Physics presets use the string sentinel ``auto`` but intentionally
        # accept concrete booleans for these two switches. Other ``auto``
        # defaults are still string-valued enums.
        if schema == "auto" and path in _AUTO_NONSTRING_PATHS:
            if supplied != "auto" and not isinstance(supplied, bool):
                raise ValueError(f"{path} must be auto or a JSON boolean")
            return
        if not isinstance(supplied, str):
            raise ValueError(f"{path} must be a string")
        return
    if schema is None and path in _OPTIONAL_FINITE_NUMBER_PATHS:
        if supplied is None:
            return
        if isinstance(supplied, bool):
            raise ValueError(f"{path} must be numeric, not a boolean")
        try:
            numeric = float(supplied)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} must be a finite number") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{path} must be finite")
        return
    if schema is None and path in _OPTIONAL_STRING_PATHS:
        if supplied is not None and not isinstance(supplied, str):
            raise ValueError(f"{path} must be a string or null")
        return
    if schema is None and path in _OPTIONAL_LIST_PATHS:
        if supplied is not None and not isinstance(supplied, list):
            raise ValueError(f"{path} must be a list or null")
        return
    if not isinstance(schema, dict):
        return
    if not isinstance(supplied, dict):
        raise ValueError(f"{path} must be an object")
    if not schema or path.endswith(".extras") or path == "plugins":
        return
    optional_core_keys = {
        "seed.pulse_definition.solver_time_sample_count",
        "seed.pulse_definition.solver_spectral_sample_count",
    }
    response_law_keys = {
        "model",
        "extras",
        "slope_per_unit",
        "coordinate",
        "scale",
        "energy_cm_inv",
        "degeneracy",
        "selected_level_index",
        "reference_temperature_K",
        "minimum_scale",
        "maximum_scale",
        "wavelength_m",
        "temperature_K",
        "values_m2",
        "concentration",
        "quenching_coordinate",
        "order",
        "source",
    }
    response_law_path = (
        path.startswith(
            "material.spectroscopic_response.cross_sections."
        )
        or path
        in {
            "material.spectroscopic_response.upper_state_lifetime.temperature",
            "material.spectroscopic_response.upper_state_lifetime.concentration",
        }
    )
    for key, value in supplied.items():
        child_path = f"{path}.{key}" if path else str(key)
        if str(key).startswith("_"):
            continue
        if key not in schema:
            if child_path == "material.crystal_approximation":
                continue
            if child_path in optional_core_keys:
                continue
            if response_law_path and key in response_law_keys:
                continue
            raise ValueError(
                f"Unknown core configuration key {child_path!r}; "
                "put extension metadata under extras or plugins, or set "
                "schema_policy='permissive' for a legacy plugin configuration"
            )
        schema_value = schema[key]
        if isinstance(schema_value, list):
            # Isotropic conductivity has a scalar public contract as well as
            # the three principal components used by the default material.
            if child_path == "material.thermal.conductivity_W_mK" and isinstance(value, (int, float)):
                continue
            if not isinstance(value, list):
                raise ValueError(
                    f"Core configuration key {child_path!r} must be a list"
                )
            if child_path == (
                "material.mechanical.anisotropic.orientation_euler_xyz_deg"
            ):
                if len(value) != 3:
                    raise ValueError(
                        f"Core configuration key {child_path!r} must contain "
                        "three angles"
                    )
                for index, angle in enumerate(value):
                    if isinstance(angle, bool):
                        raise ValueError(
                            f"{child_path}[{index}] must be a finite number"
                        )
                    try:
                        finite_angle = math.isfinite(float(angle))
                    except (TypeError, ValueError):
                        finite_angle = False
                    if not finite_angle:
                        raise ValueError(
                            f"{child_path}[{index}] must be a finite number"
                        )
                continue
            item_schema = _LIST_ITEM_SCHEMAS.get(child_path)
            if item_schema is None:
                continue
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"{child_path}[{index}] must be an object"
                    )
                _validate_core_schema(
                    item,
                    item_schema,
                    path=f"{child_path}[{index}]",
                )
            continue
        _validate_core_schema(value, schema_value, path=child_path)


def _canonicalize_custom_decay_aliases(config: dict[str, Any]) -> None:
    """Migrate legacy custom-decay spellings into an unambiguous contract."""

    levels = config.get("energy_levels", {})
    for index, decay in enumerate(levels.get("decays", [])):
        if not isinstance(decay, dict):
            continue
        legacy_rate = decay.get("rate_s")
        legacy_lifetime = decay.get("lifetime_s")
        legacy_branching = decay.get("branching")
        has_new = any(
            key in decay
            for key in (
                "partial_rate_s",
                "partial_lifetime_s",
                "total_lifetime_s",
                "branching_fraction",
            )
        )
        has_legacy = (
            legacy_rate not in (None, 0, 0.0)
            or legacy_lifetime is not None
            or legacy_branching is not None
        )
        if has_new and has_legacy:
            raise ValueError(
                f"energy_levels.decays[{index}] mixes legacy rate_s/lifetime_s/"
                "branching fields with the new decay contract"
            )
        if legacy_rate not in (None, 0, 0.0) and (
            legacy_lifetime is not None or legacy_branching is not None
        ):
            raise ValueError(
                f"energy_levels.decays[{index}] cannot combine rate_s with "
                "lifetime_s or branching"
            )
        if not has_legacy:
            continue
        if legacy_rate not in (None, 0, 0.0):
            decay["partial_rate_s"] = decay.pop("rate_s")
            decay.pop("lifetime_s", None)
            decay.pop("branching", None)
        elif legacy_branching is not None:
            if legacy_lifetime is None:
                raise ValueError(
                    f"energy_levels.decays[{index}].branching requires lifetime_s"
                )
            decay["total_lifetime_s"] = decay.pop("lifetime_s")
            decay["branching_fraction"] = decay.pop("branching")
            decay.pop("rate_s", None)
        elif legacy_lifetime is not None:
            decay["partial_lifetime_s"] = decay.pop("lifetime_s")
            decay.pop("rate_s", None)


def _canonicalize_config_aliases(config: dict[str, Any]) -> None:
    """Migrate supported legacy spellings before defaults can hide them."""

    def legacy_finite_number(value: Any, *, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number, not a boolean")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must be finite")
        return numeric

    _canonicalize_custom_decay_aliases(config)
    simulation = config.get("simulation", {})
    if isinstance(simulation, dict):
        periodic = simulation.get("periodic_steady_state", {})
        if (
            isinstance(periodic, dict)
            and str(periodic.get("method", "pulse_map"))
            == "reduced_pulse_map"
        ):
            # Historical files exposed the reduced proxy as a final periodic
            # method.  It is now only an initializer for the configured exact
            # pulse map, so old files acquire the corrected semantics without
            # losing their reduced-map resolution settings.
            periodic["method"] = "pulse_map"
            periodic["pulse_map_initialization"] = "reduced_pulse_map"
    seed = config.get("seed", {})
    if isinstance(seed, dict):
        legacy_temporal_shape = seed.pop("temporal_shape", None)
        if legacy_temporal_shape is not None:
            temporal = seed.setdefault("temporal", {})
            if not isinstance(temporal, dict):
                raise ValueError("seed.temporal must be an object")
            canonical_shape = temporal.get("type")
            if (
                canonical_shape is not None
                and str(canonical_shape) != str(legacy_temporal_shape)
            ):
                raise ValueError(
                    "seed.temporal_shape conflicts with seed.temporal.type; "
                    "keep only the canonical seed.temporal.type field"
                )
            temporal["type"] = legacy_temporal_shape

        legacy_bandwidth = seed.pop("spectral_bandwidth_m", None)
        if legacy_bandwidth is not None:
            spectrum = seed.setdefault("spectrum", {})
            if not isinstance(spectrum, dict):
                raise ValueError("seed.spectrum must be an object")
            legacy_numeric = legacy_finite_number(
                legacy_bandwidth,
                name="seed.spectral_bandwidth_m",
            )
            canonical_bandwidth = spectrum.get("bandwidth_fwhm_m")
            if canonical_bandwidth is None or float(canonical_bandwidth) == 0.0:
                spectrum["bandwidth_fwhm_m"] = legacy_numeric
            elif legacy_numeric != 0.0 and not math.isclose(
                float(canonical_bandwidth),
                legacy_numeric,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError(
                    "seed.spectral_bandwidth_m conflicts with "
                    "seed.spectrum.bandwidth_fwhm_m; keep only the canonical field"
                )

        legacy_chirp = seed.pop("chirp_s2", None)
        if legacy_chirp is not None:
            coherent = seed.setdefault("coherent", {})
            if not isinstance(coherent, dict):
                raise ValueError("seed.coherent must be an object")
            phase = coherent.setdefault("phase", {})
            if not isinstance(phase, dict):
                raise ValueError("seed.coherent.phase must be an object")
            legacy_numeric = legacy_finite_number(
                legacy_chirp,
                name="seed.chirp_s2",
            )
            canonical_chirp = phase.get("gdd_s2")
            if canonical_chirp is None or float(canonical_chirp) == 0.0:
                phase["gdd_s2"] = legacy_numeric
            elif legacy_numeric != 0.0 and not math.isclose(
                float(canonical_chirp),
                legacy_numeric,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError(
                    "seed.chirp_s2 conflicts with seed.coherent.phase.gdd_s2; "
                    "keep only the canonical field"
                )

    thermal = config.get("material", {}).get("thermal", {})
    legacy_key = "specific_heat_J_kgK"
    canonical_key = "heat_capacity_J_kgK"
    if legacy_key not in thermal:
        return
    legacy_value = legacy_finite_number(
        thermal.pop(legacy_key),
        name="material.thermal.specific_heat_J_kgK",
    )
    if canonical_key in thermal and not math.isclose(
        float(thermal[canonical_key]),
        legacy_value,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            "material.thermal.specific_heat_J_kgK conflicts with "
            "material.thermal.heat_capacity_J_kgK; keep only the canonical field"
        )
    thermal[canonical_key] = legacy_value


def _split_legacy_bench_path(
    path: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    """Split a retired ordered bench solely for one-time JSON migration."""

    if not isinstance(path, list):
        raise ValueError("legacy optical_bench.seed_path must be a list")
    segments: list[list[dict[str, Any]]] = [[]]
    markers: list[dict[str, Any]] = []
    for index, raw in enumerate(path):
        if not isinstance(raw, dict):
            raise ValueError("legacy optical_bench.seed_path entries must be objects")
        element = copy.deepcopy(raw)
        distance = float(element.pop("distance_before_m", 0.0) or 0.0)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError(
                "legacy optical_bench distance_before_m must be finite and non-negative"
            )
        if distance > 0.0:
            segments[-1].append(
                {
                    "type": "free_space",
                    "name": f"migrated_bench_space_{index + 1}",
                    "length_m": distance,
                }
            )
        kind = str(element.get("type", ""))
        turn_x = float(element.pop("turn_angle_x_deg", 0.0) or 0.0)
        turn_y = float(element.pop("turn_angle_y_deg", 0.0) or 0.0)
        if kind == "gain_medium":
            markers.append(element)
            segments.append([])
        else:
            segments[-1].append(element)
        if turn_x or turn_y:
            segments[-1].append(
                {
                    "type": "coordinate_break",
                    "name": f"{element.get('name', kind)}_path_turn",
                    "angle_x_deg": turn_x,
                    "angle_y_deg": turn_y,
                }
            )
    if not markers:
        raise ValueError(
            "legacy optical_bench.seed_path needs at least one gain_medium hit"
        )
    return segments[0], markers, segments[1:-1], segments[-1]


def _migrate_legacy_multipass_contract(config: dict[str, Any]) -> None:
    """Fold the three retired multipass contracts into ``simulation.multipass``.

    The migration runs before strict schema validation.  Old JSON files remain
    usable, but the prepared/exported runtime has one hit/transition model and
    one solver dispatch point.  Counter-propagating double pass becomes the
    physical causal template; old quick passes and ordered benches become the
    sequential execution layer of the same compiled plan.
    """

    simulation = config.get("simulation")
    if not isinstance(simulation, dict):
        return
    multipass = simulation.get("multipass")
    if multipass is not None and not isinstance(multipass, dict):
        return
    multipass_active = bool(
        isinstance(multipass, dict) and multipass.get("enabled", False)
    )
    counter = simulation.pop("counterpropagation", None)
    counter_active = bool(
        isinstance(counter, dict) and counter.get("enabled", False)
    )
    legacy_pass_mode = str(simulation.get("pass_mode", "simple"))
    legacy_passes = int(simulation.get("passes", 1))
    legacy_bench = config.pop("optical_bench", None)

    if counter_active and multipass_active:
        raise ValueError(
            "Legacy simulation.counterpropagation conflicts with simulation.multipass; keep only the new multipass contract"
        )

    optical_train = config.setdefault("optical_train", {})
    seed_path = optical_train.setdefault("seed", [])
    between_path = optical_train.setdefault("between_pass", [])
    base_marker = next(
        (
            copy.deepcopy(element)
            for element in seed_path
            if isinstance(element, dict)
            and str(element.get("type", "")) == "gain_medium"
        ),
        {"type": "gain_medium", "name": "multipass_gain"},
    )

    if counter_active:
        migrated = copy.deepcopy(DEFAULTS["simulation"]["multipass"])
        migrated.update(
            {
                "enabled": True,
                "execution_mode": "causal",
                "pass_count": 2,
                "maximum_time_bins": counter.get(
                    "maximum_time_bins", migrated["maximum_time_bins"]
                ),
                "include_dark_relaxation": counter.get(
                    "include_dark_relaxation",
                    migrated["include_dark_relaxation"],
                ),
                "maximum_dark_step_s": counter.get(
                    "maximum_dark_step_s", migrated["maximum_dark_step_s"]
                ),
                "maximum_relative_group_index_span": counter.get(
                    "maximum_relative_group_index_span",
                    migrated["maximum_relative_group_index_span"],
                ),
                "external_refractive_index": counter.get(
                    "external_refractive_index",
                    migrated["external_refractive_index"],
                ),
                "external_group_refractive_index": counter.get(
                    "external_group_refractive_index",
                    migrated["external_group_refractive_index"],
                ),
                "per_transition_extra_delay_s": float(
                    simulation.get("unmodeled_between_pass_delay_s", 0.0)
                ),
            }
        )
        if str(counter.get("return_path_model", "explicit_mirror")) == "optical_train":
            migrated["template"] = "prescribed"
            forward = copy.deepcopy(base_marker)
            forward.update({"name": "pass_1", "direction": "forward"})
            backward = copy.deepcopy(base_marker)
            backward.update({"name": "pass_2", "direction": "backward"})
            migrated["prescribed"] = {
                "hits": [forward, backward],
                "transitions": [
                    {
                        "name": "return_route",
                        "route_model": "optical_train",
                        "route": copy.deepcopy(between_path),
                        "power_transmission": float(
                            simulation.get("between_pass_transmission", 1.0)
                        ),
                    }
                ],
            }
        else:
            migrated["template"] = "double_pass"
            migrated["double_pass"].update(
                {
                    "mirror_distance_m": counter.get("mirror_distance_m", 0.0),
                    "mirror_power_reflectivity": counter.get(
                        "mirror_power_reflectivity", 1.0
                    ),
                    "mirror_radius_x_m": counter.get("mirror_radius_x_m", 0.0),
                    "mirror_radius_y_m": counter.get("mirror_radius_y_m", 0.0),
                    "mirror_center_x_m": counter.get("mirror_center_x_m", 0.0),
                    "mirror_center_y_m": counter.get("mirror_center_y_m", 0.0),
                }
            )
        simulation["multipass"] = migrated
        simulation["passes"] = 2
        simulation["pass_mode"] = "simple"
        simulation["between_pass_transmission"] = 1.0
        simulation["unmodeled_between_pass_delay_s"] = 0.0
        simulation["flip_x_between_passes"] = False
        optical_train["between_pass"] = []
        return

    if multipass_active:
        # ``passes`` is still consumed by several complete-pass backends and
        # result schemas, but it is no longer a second user-facing source of
        # truth.  Derive it from the unified multipass contract before defaults
        # are merged and before validation compiles the plan.
        simulation["passes"] = int(multipass.get("pass_count", 2))
        simulation["pass_mode"] = "simple"
        legacy_delay = float(
            simulation.get("unmodeled_between_pass_delay_s", 0.0)
        )
        if legacy_delay:
            multipass["per_transition_extra_delay_s"] = float(
                multipass.get("per_transition_extra_delay_s", 0.0)
            ) + legacy_delay
            simulation["unmodeled_between_pass_delay_s"] = 0.0
        return

    if legacy_pass_mode == "optical_bench" and isinstance(legacy_bench, dict):
        pre_elements, gain_markers, between_segments, post_elements = (
            _split_legacy_bench_path(legacy_bench.get("seed_path", []))
        )
        optical_train["seed"] = [
            *copy.deepcopy(pre_elements),
            copy.deepcopy(gain_markers[0]),
            *copy.deepcopy(post_elements),
        ]
        if len(gain_markers) > 1:
            migrated = copy.deepcopy(DEFAULTS["simulation"]["multipass"])
            migrated.update(
                {
                    "enabled": True,
                    "execution_mode": "sequential",
                    "template": "prescribed",
                    "pass_count": len(gain_markers),
                    "per_transition_power_transmission": 1.0,
                }
            )
            hits = []
            for index, marker in enumerate(gain_markers):
                hit = copy.deepcopy(marker)
                hit.setdefault("name", f"pass_{index + 1}")
                if str(hit.get("direction", "")) == "auto":
                    hit.pop("direction", None)
                hits.append(hit)
            migrated["prescribed"] = {
                "hits": hits,
                "transitions": [
                    {
                        "name": f"transition_{index + 1}_{index + 2}",
                        "route_model": "optical_train",
                        "route": copy.deepcopy(segment),
                        "power_transmission": float(
                            simulation.get("between_pass_transmission", 1.0)
                        ),
                        "extra_delay_s": float(
                            simulation.get(
                                "unmodeled_between_pass_delay_s", 0.0
                            )
                        ),
                    }
                    for index, segment in enumerate(between_segments)
                ],
            }
            simulation["multipass"] = migrated
            simulation["passes"] = len(gain_markers)
        simulation["pass_mode"] = "simple"
        simulation["between_pass_transmission"] = 1.0
        simulation["unmodeled_between_pass_delay_s"] = 0.0
        simulation["flip_x_between_passes"] = False
        optical_train["between_pass"] = []
        return

    if legacy_passes > 1:
        migrated = copy.deepcopy(DEFAULTS["simulation"]["multipass"])
        migrated.update(
            {
                "enabled": True,
                "execution_mode": "sequential",
                "template": "sequential",
                "pass_count": legacy_passes,
                "per_transition_power_transmission": float(
                    simulation.get("between_pass_transmission", 1.0)
                ),
                "per_transition_extra_delay_s": float(
                    simulation.get("unmodeled_between_pass_delay_s", 0.0)
                ),
            }
        )
        migrated["sequential"] = {
            "alternate_direction": bool(
                simulation.get("alternate_direction", True)
            ),
            "flip_x": bool(
                simulation.get("flip_x_between_passes", False)
            ),
            "route": copy.deepcopy(between_path),
        }
        simulation["multipass"] = migrated
        simulation["pass_mode"] = "simple"
        simulation["between_pass_transmission"] = 1.0
        simulation["unmodeled_between_pass_delay_s"] = 0.0
        simulation["flip_x_between_passes"] = False
        optical_train["between_pass"] = []


def prepare_config(supplied: dict[str, Any]) -> dict[str, Any]:
    """Normalize, merge, and validate a configuration mapping.

    Keeping this sequence in one public entry point is important: aliases must
    be migrated before defaults are merged, otherwise a default canonical value
    can appear to conflict with a legitimate legacy-only upload.
    """

    if not isinstance(supplied, dict):
        raise ValueError("Configuration JSON must contain an object")
    normalized = copy.deepcopy(supplied)
    _canonicalize_config_aliases(normalized)
    _migrate_legacy_multipass_contract(normalized)
    _normalize_null_optical_element_fields(normalized)
    policy = str(normalized.get("schema_policy", DEFAULTS["schema_policy"]))
    if policy == "strict_core_with_plugin_extras":
        _validate_core_schema(normalized, DEFAULTS)
    elif policy != "permissive":
        raise ValueError(
            "schema_policy must be strict_core_with_plugin_extras or permissive"
        )
    config = _merge(DEFAULTS, normalized)
    # Editing a loaded JSON configuration is an explicit user action. Convert
    # differences from its material snapshot into user_override provenance
    # before validation, just as the interactive form does.
    from .material_library import reconcile_material_field_provenance

    reconcile_material_field_provenance(config)
    validate_config(config)
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        supplied = load_json_strict(handle)
    return prepare_config(supplied)


def _positive(config: dict[str, Any], dotted: str, allow_zero: bool = False) -> None:
    value: Any = config
    for part in dotted.split("."):
        value = value[part]
    numeric = float(value)
    valid = math.isfinite(numeric) and (numeric >= 0 if allow_zero else numeric > 0)
    if not valid:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{dotted} must be {relation}; got {value!r}")


def _validate_scale_response_law(
    law: dict[str, Any],
    *,
    name: str,
    allowed: set[str],
) -> None:
    model = str(law.get("model", "off"))
    if model not in allowed:
        raise ValueError(f"{name}.model must be one of {sorted(allowed)}")
    if model == "linear_scale" and not math.isfinite(float(law.get("slope_per_unit", 0.0))):
        raise ValueError(f"{name}.slope_per_unit must be finite")
    if model == "scale_table":
        coordinate = [float(value) for value in law.get("coordinate", [])]
        scale = [float(value) for value in law.get("scale", [])]
        if len(coordinate) < 2 or len(coordinate) != len(scale):
            raise ValueError(f"{name} scale_table needs equal arrays with at least two points")
        if not all(math.isfinite(value) for value in [*coordinate, *scale]):
            raise ValueError(f"{name} scale_table values must be finite")
        if any(value < 0.0 for value in scale):
            raise ValueError(f"{name} scale_table scales must be non-negative")
        if len(set(coordinate)) != len(coordinate):
            raise ValueError(f"{name} scale_table coordinates must be unique")
    if model == "boltzmann_stark_fraction":
        energies = [float(value) for value in law.get("energy_cm_inv", [])]
        degeneracies = [float(value) for value in law.get("degeneracy", [])]
        selected = require_exact_integer(
            law.get("selected_level_index", -1),
            name=f"{name}.selected_level_index",
        )
        reference_temperature = float(law.get("reference_temperature_K", 0.0))
        if len(energies) < 1 or len(energies) != len(degeneracies):
            raise ValueError(
                f"{name} boltzmann_stark_fraction needs equal non-empty energy/degeneracy arrays"
            )
        if not all(math.isfinite(value) for value in [*energies, *degeneracies]):
            raise ValueError(f"{name} Stark energies and degeneracies must be finite")
        if any(value <= 0.0 for value in degeneracies):
            raise ValueError(f"{name} Stark degeneracies must be positive")
        if selected < 0 or selected >= len(energies):
            raise ValueError(f"{name}.selected_level_index is outside the Stark manifold")
        if not math.isfinite(reference_temperature) or reference_temperature <= 0.0:
            raise ValueError(f"{name}.reference_temperature_K must be positive and finite")
    if model == "stern_volmer":
        if (
            not math.isfinite(float(law.get("quenching_coordinate", 0.0)))
            or float(law.get("quenching_coordinate", 0.0)) <= 0.0
        ):
            raise ValueError(f"{name}.quenching_coordinate must be positive")
        if (
            not math.isfinite(float(law.get("order", 1.0)))
            or float(law.get("order", 1.0)) <= 0.0
        ):
            raise ValueError(f"{name}.order must be positive")
    minimum = float(law.get("minimum_scale", 0.0))
    maximum = float(law.get("maximum_scale", math.inf))
    if (
        not math.isfinite(minimum)
        or math.isnan(maximum)
        or minimum < 0.0
        or maximum < minimum
    ):
        raise ValueError(f"{name} scale limits must satisfy 0 <= minimum <= maximum")


def _validate_cross_section_response_law(law: dict[str, Any], *, name: str) -> None:
    model = str(law.get("model", "off"))
    allowed = {
        "off",
        "linear_scale",
        "scale_table",
        "boltzmann_stark_fraction",
        "absolute_temperature_wavelength_table",
    }
    if model != "absolute_temperature_wavelength_table":
        _validate_scale_response_law(law, name=name, allowed=allowed - {"absolute_temperature_wavelength_table"})
    else:
        temperatures = [float(value) for value in law.get("temperature_K", [])]
        wavelengths = [float(value) for value in law.get("wavelength_m", [])]
        values = law.get("values_m2", [])
        if len(temperatures) < 2 or len(wavelengths) < 2:
            raise ValueError(f"{name} absolute table needs at least two temperatures and wavelengths")
        if len(values) != len(temperatures) or any(
            not isinstance(row, list) or len(row) != len(wavelengths) for row in values
        ):
            raise ValueError(f"{name}.values_m2 must have shape (Ntemperature, Nwavelength)")
        flattened = [float(value) for row in values for value in row]
        if not all(math.isfinite(value) for value in [*temperatures, *wavelengths, *flattened]):
            raise ValueError(f"{name} absolute table values must be finite")
        if any(value <= 0.0 for value in temperatures) or any(value <= 0.0 for value in wavelengths):
            raise ValueError(f"{name} absolute table coordinates must be positive")
        if any(value < 0.0 for value in flattened):
            raise ValueError(f"{name} cross sections must be non-negative")
        if len(set(temperatures)) != len(temperatures) or len(set(wavelengths)) != len(wavelengths):
            raise ValueError(f"{name} absolute table coordinates must be unique")
    concentration_law = law.get("concentration")
    if concentration_law:
        _validate_scale_response_law(
            concentration_law,
            name=f"{name}.concentration",
            allowed={"off", "linear_scale", "scale_table", "stern_volmer"},
        )


def _signal_upper_reservoir_names(
    levels_cfg: dict[str, Any],
) -> list[str]:
    """Return the independently populated upper reservoirs seen by ASE."""

    if str(levels_cfg.get("model", "four_level")) == "custom":
        transitions = list(levels_cfg.get("optical_transitions", []))
        if transitions:
            base_uppers = sorted(
                {
                    str(transition["upper"])
                    for transition in transitions
                    if str(transition.get("field", "")) == "signal"
                }
            )
        else:
            base_uppers = [
                str(levels_cfg.get("signal_transition", {})["upper"])
            ]
    else:
        base_uppers = ["upper"]
    ensemble = levels_cfg.get("inhomogeneous_ensembles", {}) or {}
    if not bool(ensemble.get("enabled", False)):
        return base_uppers
    return [
        f"{group['name']}:{upper}"
        for group in ensemble.get("groups", [])
        for upper in base_uppers
    ]


def _signal_reservoir_pairs(
    levels_cfg: dict[str, Any],
) -> set[tuple[str, str]]:
    """Return independent lower/upper reservoirs used by Signal transitions."""

    if str(levels_cfg.get("model", "four_level")) != "custom":
        base_pairs = {("lower", "upper")}
    else:
        configured = list(levels_cfg.get("optical_transitions", []))
        transitions = (
            [
                item
                for item in configured
                if str(item.get("field", "")) == "signal"
            ]
            if configured
            else [levels_cfg.get("signal_transition", {})]
        )
        base_pairs = {
            (str(item.get("lower")), str(item.get("upper")))
            for item in transitions
        }
    ensemble = levels_cfg.get("inhomogeneous_ensembles", {}) or {}
    if not bool(ensemble.get("enabled", False)):
        return base_pairs
    return {
        (f"{group['name']}:{lower}", f"{group['name']}:{upper}")
        for group in ensemble.get("groups", [])
        for lower, upper in base_pairs
    }


def _validate_energy_level_network(levels_cfg: dict[str, Any]) -> None:
    """Reject non-physical level-network inputs before any coupled solve."""

    model = str(levels_cfg.get("model", "four_level"))
    for key in (
        "pump_absorption_cross_section_m2",
        "pump_emission_cross_section_m2",
        "signal_absorption_cross_section_m2",
        "signal_emission_cross_section_m2",
    ):
        value = float(levels_cfg.get(key, 0.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"energy_levels.{key} must be finite and non-negative")
    for key in (
        "upper_state_lifetime_s",
        "pump_level_lifetime_s",
        "lower_level_lifetime_s",
    ):
        value = float(levels_cfg.get(key, 0.0))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"energy_levels.{key} must be positive and finite")
    saturation = levels_cfg.get("saturation_fluence_J_m2")
    if saturation is not None and (
        not math.isfinite(float(saturation)) or float(saturation) <= 0.0
    ):
        raise ValueError(
            "energy_levels.saturation_fluence_J_m2 must be positive and finite when supplied"
        )
    override = levels_cfg.get("gain_override_m_inv")
    if override is not None and not math.isfinite(float(override)):
        raise ValueError("energy_levels.gain_override_m_inv must be finite when supplied")

    builtin_names = {
        "two_level": ["ground_lower", "upper"],
        "three_level": ["ground_lower", "upper", "pump"],
        "four_level": ["ground", "lower", "upper", "pump"],
    }
    if model == "custom":
        names = list(levels_cfg.get("levels", []))
        if (
            len(names) < 2
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "energy_levels.levels must contain at least two unique non-empty names"
            )
        name_set = set(names)
        configured_optical_transitions = list(
            levels_cfg.get("optical_transitions", [])
        )
        transitions_to_validate: list[tuple[str, dict[str, Any]]] = []
        if configured_optical_transitions:
            field_counts = {"pump": 0, "signal": 0}
            transition_names: list[str] = []
            for index, transition in enumerate(configured_optical_transitions):
                if not isinstance(transition, dict):
                    raise ValueError(
                        f"energy_levels.optical_transitions[{index}] must be an object"
                    )
                field = str(transition.get("field", ""))
                if field not in {"pump", "signal"}:
                    raise ValueError(
                        f"energy_levels.optical_transitions[{index}].field "
                        "must be pump or signal; ASE uses the configured signal "
                        "transitions and is not a separate transition family"
                    )
                field_counts[field] += 1
                transition_name = str(
                    transition.get("name", f"{field}_{index}")
                )
                transition_names.append(transition_name)
                transitions_to_validate.append(
                    (
                        f"optical_transitions[{index}]",
                        transition,
                    )
                )
            if field_counts["pump"] < 1 or field_counts["signal"] < 1:
                raise ValueError(
                    "energy_levels.optical_transitions requires at least one "
                    "pump and one signal transition"
                )
            if len(transition_names) != len(set(transition_names)):
                raise ValueError(
                    "energy_levels.optical_transitions names must be unique"
                )
        else:
            transitions_to_validate = [
                ("pump_transition", levels_cfg.get("pump_transition", {})),
                ("signal_transition", levels_cfg.get("signal_transition", {})),
            ]
        for transition_name, transition in transitions_to_validate:
            lower = transition.get("lower")
            upper = transition.get("upper")
            if lower not in name_set or upper not in name_set or lower == upper:
                raise ValueError(
                    f"energy_levels.{transition_name} must connect two distinct configured levels"
                )
            for key in (
                "sigma_abs_m2",
                "sigma_em_m2",
                "quantum_efficiency",
                "center_wavelength_m",
                "homogeneous_fwhm_Hz",
                "absorption_scale",
                "emission_scale",
            ):
                if isinstance(transition.get(key), bool):
                    raise ValueError(
                        f"energy_levels.{transition_name}.{key} must be numeric, "
                        "not a boolean"
                    )
            for key in ("sigma_abs_m2", "sigma_em_m2"):
                value = float(transition.get(key, 0.0))
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"energy_levels.{transition_name}.{key} must be finite and non-negative"
                    )
            efficiency = float(transition.get("quantum_efficiency", 1.0))
            if (
                not math.isfinite(efficiency)
                or not 0.0 <= efficiency <= 1.0
            ):
                raise ValueError(
                    f"energy_levels.{transition_name}.quantum_efficiency "
                    "must lie in [0, 1]"
                )
            line_shape = str(transition.get("line_shape", "none"))
            if line_shape not in {"none", "gaussian", "lorentzian"}:
                raise ValueError(
                    f"energy_levels.{transition_name}.line_shape must be "
                    "none, gaussian, or lorentzian"
                )
            center = transition.get("center_wavelength_m")
            linewidth = transition.get("homogeneous_fwhm_Hz")
            if (center is None) != (linewidth is None):
                raise ValueError(
                    f"energy_levels.{transition_name} requires both "
                    "center_wavelength_m and homogeneous_fwhm_Hz"
                )
            for key, value in (
                ("center_wavelength_m", center),
                ("homogeneous_fwhm_Hz", linewidth),
            ):
                if value is not None and (
                    not math.isfinite(float(value)) or float(value) <= 0.0
                ):
                    raise ValueError(
                        f"energy_levels.{transition_name}.{key} must be "
                        "positive and finite"
                    )
            for key in ("absorption_scale", "emission_scale"):
                value = float(transition.get(key, 1.0))
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"energy_levels.{transition_name}.{key} must be "
                        "finite and non-negative"
                    )
        branching_by_source: dict[str, float] = {}
        total_lifetime_by_source: dict[str, float] = {}
        decay_semantics_by_source: dict[str, str] = {}
        for index, decay in enumerate(levels_cfg.get("decays", [])):
            source = decay.get("from")
            target = decay.get("to")
            if source not in name_set or target not in name_set or source == target:
                raise ValueError(
                    f"energy_levels.decays[{index}] must connect two distinct configured levels"
                )
            partial_rate = decay.get("partial_rate_s")
            partial_lifetime = decay.get("partial_lifetime_s")
            total_lifetime = decay.get("total_lifetime_s")
            branching = decay.get("branching_fraction")
            for key in (
                "partial_rate_s",
                "partial_lifetime_s",
                "total_lifetime_s",
                "branching_fraction",
            ):
                if isinstance(decay.get(key), bool):
                    raise ValueError(
                        f"energy_levels.decays[{index}].{key} must be numeric, "
                        "not a boolean"
                    )
            direct_definitions = sum(
                value is not None
                for value in (partial_rate, partial_lifetime)
            )
            branching_definition = total_lifetime is not None or branching is not None
            if direct_definitions + int(branching_definition) != 1:
                raise ValueError(
                    f"energy_levels.decays[{index}] must define exactly one of "
                    "partial_rate_s, partial_lifetime_s, or "
                    "total_lifetime_s + branching_fraction"
                )
            if branching_definition and (
                total_lifetime is None or branching is None
            ):
                raise ValueError(
                    f"energy_levels.decays[{index}] requires both "
                    "total_lifetime_s and branching_fraction"
                )
            channel_semantics = (
                "shared_total_lifetime_branching"
                if branching_definition
                else "independent_partial_channel"
            )
            previous_semantics = decay_semantics_by_source.get(str(source))
            if (
                previous_semantics is not None
                and previous_semantics != channel_semantics
            ):
                raise ValueError(
                    f"energy_levels.decays from {source!r} cannot mix "
                    "independent partial channels with shared "
                    "total_lifetime_s branching"
                )
            decay_semantics_by_source[str(source)] = channel_semantics
            for key, value in (
                ("partial_rate_s", partial_rate),
                ("partial_lifetime_s", partial_lifetime),
                ("total_lifetime_s", total_lifetime),
            ):
                if value is not None and (
                    not math.isfinite(float(value)) or float(value) <= 0.0
                ):
                    raise ValueError(
                        f"energy_levels.decays[{index}].{key} must be "
                        "positive and finite"
                    )
            if branching is not None:
                fraction = float(branching)
                if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                    raise ValueError(
                        f"energy_levels.decays[{index}].branching_fraction "
                        "must lie in [0, 1]"
                    )
                previous_lifetime = total_lifetime_by_source.get(str(source))
                if previous_lifetime is not None and not math.isclose(
                    previous_lifetime,
                    float(total_lifetime),
                    rel_tol=1e-12,
                    abs_tol=0.0,
                ):
                    raise ValueError(
                        f"energy_levels.decays from {source!r} must use one "
                        "consistent total_lifetime_s"
                    )
                total_lifetime_by_source[str(source)] = float(total_lifetime)
                branching_by_source[str(source)] = (
                    branching_by_source.get(str(source), 0.0) + fraction
                )
        for source, total in branching_by_source.items():
            if abs(total - 1.0) > 1e-12:
                comparison = (
                    "sum exceeds 1"
                    if total > 1.0
                    else "sum is below 1 and leaves population without a destination"
                )
                raise ValueError(
                    f"energy_levels.decays branching_fraction {comparison} for source "
                    f"{source!r} must equal 1 within absolute tolerance 1e-12 "
                    f"when total_lifetime_s is used (got {total:.16g}); use "
                    "partial_rate_s/partial_lifetime_s for independent channels"
                )
    else:
        name_set = set(builtin_names.get(model, []))

    ensemble_cfg = levels_cfg.get("inhomogeneous_ensembles", {}) or {}
    if bool(ensemble_cfg.get("enabled", False)):
        groups = list(ensemble_cfg.get("groups", []))
        if len(groups) < 2:
            raise ValueError(
                "energy_levels.inhomogeneous_ensembles requires at least two groups"
            )
        if levels_cfg.get("nonlinear_processes"):
            raise ValueError(
                "inhomogeneous ensembles with nonlinear_processes require "
                "explicit cross-group stoichiometry and are not combined "
                "implicitly"
            )
        names_seen: list[str] = []
        fractions_seen: list[float] = []
        default_line_shape = str(
            ensemble_cfg.get("line_shape", "gaussian")
        )
        if default_line_shape not in {"gaussian", "lorentzian"}:
            raise ValueError(
                "energy_levels.inhomogeneous_ensembles.line_shape must be "
                "gaussian or lorentzian"
            )
        default_linewidth = ensemble_cfg.get("homogeneous_fwhm_Hz")
        if default_linewidth is not None and (
            not math.isfinite(float(default_linewidth))
            or float(default_linewidth) <= 0.0
        ):
            raise ValueError(
                "energy_levels.inhomogeneous_ensembles."
                "homogeneous_fwhm_Hz must be positive and finite"
            )
        for index, group in enumerate(groups):
            if not isinstance(group, dict):
                raise ValueError(
                    "energy_levels.inhomogeneous_ensembles.groups entries "
                    "must be objects"
                )
            group_name = str(group.get("name", "")).strip()
            if not group_name:
                raise ValueError(
                    f"energy_levels.inhomogeneous_ensembles.groups[{index}]."
                    "name must be non-empty"
                )
            names_seen.append(group_name)
            for key in (
                "fraction",
                "homogeneous_fwhm_Hz",
                "signal_center_wavelength_m",
                "pump_center_wavelength_m",
                "pump_absorption_scale",
                "pump_emission_scale",
                "signal_absorption_scale",
                "signal_emission_scale",
            ):
                if isinstance(group.get(key), bool):
                    raise ValueError(
                        "energy_levels.inhomogeneous_ensembles.groups"
                        f"[{index}].{key} must be numeric, not a boolean"
                    )
            fraction = float(group.get("fraction", math.nan))
            if not math.isfinite(fraction) or fraction <= 0.0:
                raise ValueError(
                    f"energy_levels.inhomogeneous_ensembles.groups[{index}]."
                    "fraction must be positive and finite"
                )
            fractions_seen.append(fraction)
            linewidth = group.get(
                "homogeneous_fwhm_Hz", default_linewidth
            )
            if linewidth is None:
                raise ValueError(
                    f"energy_levels.inhomogeneous_ensembles.groups[{index}] "
                    "requires homogeneous_fwhm_Hz"
                )
            if (
                not math.isfinite(float(linewidth))
                or float(linewidth) <= 0.0
            ):
                raise ValueError(
                    f"energy_levels.inhomogeneous_ensembles.groups[{index}]."
                    "homogeneous_fwhm_Hz must be positive and finite"
                )
            if (
                group.get("signal_center_wavelength_m") is None
                and group.get("pump_center_wavelength_m") is None
            ):
                raise ValueError(
                    f"energy_levels.inhomogeneous_ensembles.groups[{index}] "
                    "requires a signal_center_wavelength_m or "
                    "pump_center_wavelength_m"
                )
            for key in (
                "signal_center_wavelength_m",
                "pump_center_wavelength_m",
            ):
                value = group.get(key)
                if value is not None and (
                    not math.isfinite(float(value)) or float(value) <= 0.0
                ):
                    raise ValueError(
                        f"energy_levels.inhomogeneous_ensembles.groups[{index}]."
                        f"{key} must be positive and finite"
                    )
            for key in (
                "pump_absorption_scale",
                "pump_emission_scale",
                "signal_absorption_scale",
                "signal_emission_scale",
            ):
                value = float(group.get(key, 1.0))
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"energy_levels.inhomogeneous_ensembles.groups[{index}]."
                        f"{key} must be finite and non-negative"
                    )
        if len(names_seen) != len(set(names_seen)):
            raise ValueError(
                "energy_levels.inhomogeneous_ensembles group names must be unique"
            )
        if abs(sum(fractions_seen) - 1.0) > 1e-12:
            raise ValueError(
                "energy_levels.inhomogeneous_ensembles group fractions must "
                "sum to one within absolute tolerance 1e-12"
            )
        spectral_diffusion = float(
            ensemble_cfg.get("spectral_diffusion_rate_s", 0.0)
        )
        if not math.isfinite(spectral_diffusion) or spectral_diffusion < 0.0:
            raise ValueError(
                "energy_levels.inhomogeneous_ensembles."
                "spectral_diffusion_rate_s must be finite and non-negative"
            )

    fractions = levels_cfg.get("initial_fractions")
    if fractions is not None:
        values = np.asarray(fractions, dtype=float)
        expected_count = len(name_set)
        if values.shape != (expected_count,):
            raise ValueError(
                f"energy_levels.initial_fractions must contain {expected_count} values"
            )
        total = float(np.sum(values))
        if (
            np.any(~np.isfinite(values))
            or np.any(values < 0.0)
            or abs(total - 1.0) > 1e-12
        ):
            raise ValueError(
                "energy_levels.initial_fractions must be non-negative, finite, "
                "and sum to one within absolute tolerance 1e-12"
            )

    for index, process in enumerate(levels_cfg.get("nonlinear_processes", [])):
        if isinstance(process.get("rate_coefficient"), bool):
            raise ValueError(
                f"energy_levels.nonlinear_processes[{index}].rate_coefficient "
                "must be numeric, not a boolean"
            )
        coefficient = float(process.get("rate_coefficient", float("nan")))
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError(
                f"energy_levels.nonlinear_processes[{index}].rate_coefficient must be finite and non-negative"
            )
        reactants = process.get("reactants", {})
        products = process.get("products", {})
        if not isinstance(reactants, dict) or not reactants or not isinstance(products, dict) or not products:
            raise ValueError(
                f"energy_levels.nonlinear_processes[{index}] needs non-empty reactants and products"
            )
        for side_name, entries in (("reactants", reactants), ("products", products)):
            for name, stoichiometry in entries.items():
                if isinstance(stoichiometry, bool):
                    raise ValueError(
                        f"energy_levels.nonlinear_processes[{index}].{side_name} "
                        "stoichiometry must be numeric, not a boolean"
                    )
                value = float(stoichiometry)
                if name not in name_set or not math.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"energy_levels.nonlinear_processes[{index}].{side_name} "
                        "must use configured levels with positive finite stoichiometry"
                    )
        reactant_total = sum(float(value) for value in reactants.values())
        product_total = sum(float(value) for value in products.values())
        if not math.isclose(reactant_total, product_total, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"energy_levels.nonlinear_processes[{index}] must conserve the modeled ion count"
            )

    event_ledger = levels_cfg.get("event_energy_ledger", {}) or {}
    if not isinstance(event_ledger.get("enabled", False), bool):
        raise ValueError("energy_levels.event_energy_ledger.enabled must be boolean")
    if not isinstance(
        event_ledger.get("require_complete_partitions", True), bool
    ):
        raise ValueError(
            "energy_levels.event_energy_ledger."
            "require_complete_partitions must be boolean"
        )
    if bool(event_ledger.get("enabled", False)) and model == "custom":
        energies = np.asarray(levels_cfg.get("level_energies_J", []), dtype=float)
        if energies.shape != (len(name_set),):
            raise ValueError(
                "custom event_energy_ledger requires one explicit "
                "energy_levels.level_energies_J value per level"
            )

    destinations = (
        "heat",
        "fluorescence_escape",
        "ase_source",
        "unresolved",
    )

    def validate_partition(
        item: Any,
        *,
        path: str,
        allow_lifetime_branching: bool = False,
    ) -> None:
        if not isinstance(item, dict):
            raise ValueError(f"{path} must be an object")
        partition_model = str(item.get("model", "fixed"))
        if partition_model == "lifetime_branching":
            if not allow_lifetime_branching:
                raise ValueError(
                    f"{path}.model=lifetime_branching is only valid for a decay partition"
                )
            radiative_destination = str(
                item.get("radiative_destination", "fluorescence_escape")
            )
            if radiative_destination not in {
                "fluorescence_escape",
                "ase_source",
            }:
                raise ValueError(
                    f"{path}.radiative_destination must be fluorescence_escape or ase_source"
                )
            radiative_lifetime = levels_cfg.get(
                "upper_state_radiative_lifetime_s"
            )
            if radiative_lifetime is None or (
                not math.isfinite(float(radiative_lifetime))
                or float(radiative_lifetime) <= 0.0
            ):
                raise ValueError(
                    f"{path}.model=lifetime_branching requires a positive "
                    "energy_levels.upper_state_radiative_lifetime_s"
                )
            return
        if partition_model != "fixed":
            raise ValueError(
                f"{path}.model must be fixed or lifetime_branching"
            )
        fractions = item.get("fractions", item)
        if not isinstance(fractions, dict):
            raise ValueError(f"{path}.fractions must be an object")
        for destination in destinations:
            if isinstance(fractions.get(destination, 0.0), bool):
                raise ValueError(
                    f"{path}.{destination} must be numeric, not a boolean"
                )
        values = [float(fractions.get(name, 0.0)) for name in destinations]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                f"{path} energy fractions must be finite and non-negative"
            )
        if abs(sum(values) - 1.0) > 1e-12:
            raise ValueError(
                f"{path} energy fractions must sum to one within "
                "absolute tolerance 1e-12"
            )

    decay_partition_keys: set[tuple[str, str]] = set()
    for index, item in enumerate(event_ledger.get("decay_partitions", [])):
        path = f"energy_levels.event_energy_ledger.decay_partitions[{index}]"
        validate_partition(item, path=path, allow_lifetime_branching=True)
        source = str(item.get("from", ""))
        target = str(item.get("to", ""))
        if source not in name_set or target not in name_set or source == target:
            raise ValueError(
                f"{path} must name two distinct configured levels"
            )
        pair = (source, target)
        if pair in decay_partition_keys:
            raise ValueError(f"{path} duplicates a decay partition")
        decay_partition_keys.add(pair)

    process_names = [
        str(process.get("name", f"process_{index}"))
        for index, process in enumerate(
            levels_cfg.get("nonlinear_processes", [])
        )
    ]
    if len(process_names) != len(set(process_names)):
        raise ValueError(
            "energy_levels.nonlinear_processes names must be unique"
        )
    nonlinear_partition_keys: set[tuple[str, Any]] = set()
    for index, item in enumerate(
        event_ledger.get("nonlinear_partitions", [])
    ):
        path = (
            f"energy_levels.event_energy_ledger."
            f"nonlinear_partitions[{index}]"
        )
        validate_partition(item, path=path)
        has_name = item.get("process") is not None
        has_index = item.get("process_index") is not None
        if has_name == has_index:
            raise ValueError(
                f"{path} must define exactly one of process or process_index"
            )
        if has_name:
            name = str(item["process"])
            if name not in process_names:
                raise ValueError(f"{path}.process does not match a process")
            key: tuple[str, Any] = ("name", name)
        else:
            process_index = require_exact_integer(
                item["process_index"],
                name=f"{path}.process_index",
                minimum=0,
            )
            if process_index >= len(process_names):
                raise ValueError(f"{path}.process_index is out of range")
            key = ("name", process_names[process_index])
        if key in nonlinear_partition_keys:
            raise ValueError(f"{path} duplicates a nonlinear partition")
        nonlinear_partition_keys.add(key)

    if (
        bool(event_ledger.get("enabled", False))
        and bool(event_ledger.get("require_complete_partitions", True))
    ):
        if model == "two_level":
            required_decay_pairs = {("upper", "ground_lower")}
        elif model == "three_level":
            required_decay_pairs = {
                ("pump", "upper"),
                ("upper", "ground_lower"),
            }
        elif model == "four_level":
            required_decay_pairs = {
                ("pump", "upper"),
                ("upper", "lower"),
                ("lower", "ground"),
            }
        else:
            required_decay_pairs = {
                (str(decay.get("from")), str(decay.get("to")))
                for decay in levels_cfg.get("decays", [])
            }
        missing_decay_pairs = sorted(
            required_decay_pairs - decay_partition_keys
        )
        if missing_decay_pairs:
            formatted = ", ".join(
                f"{source}->{target}"
                for source, target in missing_decay_pairs
            )
            raise ValueError(
                "energy_levels.event_energy_ledger is missing explicit "
                f"decay partitions for {formatted}"
            )
        covered_processes = {
            (
                process_names[int(value)]
                if kind == "index"
                else str(value)
            )
            for kind, value in nonlinear_partition_keys
        }
        missing_processes = sorted(set(process_names) - covered_processes)
        if missing_processes:
            raise ValueError(
                "energy_levels.event_energy_ledger is missing explicit "
                "nonlinear partitions for "
                + ", ".join(missing_processes)
            )


def _validate_optical_element_response(element: dict[str, Any], *, name: str) -> None:
    """Validate scalar and tabulated power responses before propagation.

    ``spectral.response_value`` intentionally retains a scalar fallback for
    old files that have no spectral table.  A *present but malformed* table,
    however, must not silently fall back: that can change Pump/Seed energy at
    an interface while leaving a perfectly plausible-looking trace.
    """

    response_keys = (
        "power_transmission",
        "power_reflectivity",
        "front_power_transmission",
        "back_power_transmission",
    )
    for key in response_keys:
        value = element.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}.{key} must be a numeric power fraction")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{name}.{key} must be finite and lie in [0, 1]")

    response = element.get("spectral_response")
    if response is None:
        return
    if not isinstance(response, dict):
        raise ValueError(f"{name}.spectral_response must be an object")
    out_of_range = str(response.get("out_of_range", "error"))
    if out_of_range not in {"error", "edge", "zero"}:
        raise ValueError(
            f"{name}.spectral_response.out_of_range must be error, edge, or zero"
        )
    raw_wavelengths = response.get("wavelength_m")
    if not isinstance(raw_wavelengths, (list, tuple)):
        raise ValueError(
            f"{name}.spectral_response.wavelength_m must be a numeric array"
        )
    if any(isinstance(value, bool) for value in raw_wavelengths):
        raise ValueError(
            f"{name}.spectral_response.wavelength_m values must be numeric, "
            "not boolean"
        )
    try:
        wavelengths = [float(value) for value in raw_wavelengths]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name}.spectral_response.wavelength_m must be a numeric array"
        ) from exc
    if len(wavelengths) < 2:
        raise ValueError(
            f"{name}.spectral_response.wavelength_m needs at least two samples"
        )
    if (
        any(not math.isfinite(value) or value <= 0.0 for value in wavelengths)
        or len(set(wavelengths)) != len(wavelengths)
    ):
        raise ValueError(
            f"{name}.spectral_response wavelengths must be positive, finite, and unique"
        )
    populated = False
    for key in response_keys:
        if key not in response:
            continue
        raw_values = response[key]
        if not isinstance(raw_values, (list, tuple)):
            raise ValueError(
                f"{name}.spectral_response.{key} must be a numeric array"
            )
        populated = True
        if any(isinstance(value, bool) for value in raw_values):
            raise ValueError(
                f"{name}.spectral_response.{key} values must be numeric, "
                "not boolean"
            )
        try:
            values = [float(value) for value in raw_values]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name}.spectral_response.{key} must be a numeric array"
            ) from exc
        if len(values) != len(wavelengths):
            raise ValueError(
                f"{name}.spectral_response.{key} must match wavelength_m"
            )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(
                f"{name}.spectral_response.{key} values must be finite and lie in [0, 1]"
            )
    if not populated:
        raise ValueError(
            f"{name}.spectral_response needs at least one supported power-response array"
        )


def _validate_optical_element_geometry(element: dict[str, Any], *, name: str) -> None:
    """Reject incomplete/no-op optical placeholders before a solver is chosen.

    Direct low-level optics functions retain a few compatibility fallbacks, but
    a public JSON configuration should never turn a half-filled table row into
    a plausible no-op element.  Validate only core built-in fields here so
    plugin metadata and unknown extension keys remain available.
    """

    kind = str(element["type"])

    def finite(key: str, default: float = 0.0) -> float:
        value = element.get(key, default)
        if isinstance(value, bool):
            raise ValueError(f"{name}.{key} must be numeric")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}.{key} must be numeric") from exc
        if not math.isfinite(result):
            if key.endswith("_deg"):
                raise ValueError(
                    f"{name}.{key} must be finite and have magnitude below 89 degrees"
                )
            raise ValueError(f"{name}.{key} must be finite")
        return result

    def positive_aperture_value(value: Any, *, key: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name}.{key} must be numeric")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}.{key} must be numeric") from exc
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name}.{key} must be positive and finite")
        return numeric

    def validate_aperture(*, allow_legacy_radii: bool) -> None:
        raw_shape = element.get("shape")
        shape = "ellipse" if raw_shape is None else str(raw_shape)
        if shape not in {"circle", "ellipse", "super_gaussian", "rectangle"}:
            raise ValueError(f"{name}.shape is unsupported")
        if shape == "rectangle":
            width = element.get("aperture_width_m")
            height = element.get("aperture_height_m")
            if width is None:
                raise ValueError(
                    f"{name} rectangular aperture needs aperture_width_m"
                )
            positive_aperture_value(width, key="aperture_width_m")
            positive_aperture_value(
                width if height is None else height,
                key="aperture_height_m",
            )
            return

        common = element.get("aperture_radius_m")
        radius_x = element.get("aperture_radius_x_m")
        radius_y = element.get("aperture_radius_y_m")
        if radius_x is None:
            radius_x = common
        if radius_y is None:
            radius_y = common
        if allow_legacy_radii:
            if radius_x is None:
                radius_x = element.get("radius_x_m")
            if radius_y is None:
                radius_y = element.get("radius_y_m")
        if radius_x is None or radius_y is None:
            raise ValueError(f"{name} aperture needs positive x/y radii")
        positive_aperture_value(radius_x, key="aperture_radius_x_m")
        positive_aperture_value(radius_y, key="aperture_radius_y_m")
        if shape == "super_gaussian":
            order = element.get("order")
            positive_aperture_value(10.0 if order is None else order, key="order")

    numeric_element_keys = {
        "distance_before_m",
        "length_m",
        "focal_length_m",
        "focal_length_x_m",
        "focal_length_y_m",
        "refractive_index",
        "group_refractive_index",
        "group_refractive_index_after",
        "thickness_m",
        "n_after",
        "outside_refractive_index",
        "radius_front_m",
        "radius_back_m",
        "radius_front_x_m",
        "radius_front_y_m",
        "radius_back_x_m",
        "radius_back_y_m",
        "radius_x_m",
        "radius_y_m",
        "center_x_m",
        "center_y_m",
        "shift_x_m",
        "shift_y_m",
        "angle_x_deg",
        "angle_y_deg",
        "axis_angle_x_deg",
        "axis_angle_y_deg",
        "turn_angle_x_deg",
        "turn_angle_y_deg",
        "incidence_angle_deg",
        "incidence_angle_x_deg",
        "incidence_angle_y_deg",
        "front_incidence_angle_deg",
        "back_incidence_angle_deg",
        "absorption_m_inv",
        "aperture_radius_m",
        "aperture_radius_x_m",
        "aperture_radius_y_m",
        "aperture_width_m",
        "aperture_height_m",
        "order",
        "constant_phase_rad",
        "quadratic_x_rad_m2",
        "quadratic_y_rad_m2",
        "group_delay_s",
        "object_to_first_lens_m",
        "first_focal_length_m",
        "lens_separation_m",
        "second_focal_length_m",
        "second_lens_to_mirror_m",
    }
    for key in numeric_element_keys:
        if key in element and element[key] is not None:
            finite(key)

    for key in (
        "center_x_m",
        "center_y_m",
        "shift_x_m",
        "shift_y_m",
        "constant_phase_rad",
        "quadratic_x_rad_m2",
        "quadratic_y_rad_m2",
        "group_delay_s",
        "group_refractive_index",
        "group_refractive_index_after",
        "axis_angle_x_deg",
        "axis_angle_y_deg",
        "turn_angle_x_deg",
        "turn_angle_y_deg",
        "incidence_angle_x_deg",
        "incidence_angle_y_deg",
    ):
        if key in element and element[key] is not None:
            value = finite(key)
            if key == "group_delay_s" and value < 0.0:
                raise ValueError(f"{name}.group_delay_s must be non-negative")
            if key.startswith("group_refractive_index") and value <= 0.0:
                raise ValueError(f"{name}.{key} must be positive")
            if key.endswith("_deg") and abs(value) >= 89.0:
                raise ValueError(
                    f"{name}.{key} must have magnitude below 89 degrees"
                )

    if kind == "free_space":
        if "length_m" not in element:
            raise ValueError(f"{name}.length_m is required for free_space")
        if finite("length_m") < 0.0:
            raise ValueError(f"{name}.length_m must be non-negative")
    elif kind == "thin_lens":
        focal_keys = (
            "focal_length_m",
            "focal_length_x_m",
            "focal_length_y_m",
        )
        populated = []
        for key in focal_keys:
            if key in element and element[key] is not None:
                value = finite(key)
                if value != 0.0:
                    populated.append(value)
        if not populated:
            raise ValueError(
                f"{name} thin_lens needs a nonzero focal_length_m or axis-specific focal length"
            )
    elif kind == "thick_lens":
        for key in ("refractive_index", "thickness_m"):
            if key not in element:
                raise ValueError(f"{name}.{key} is required for thick_lens")
        if finite("refractive_index") <= 0.0:
            raise ValueError(f"{name}.refractive_index must be positive")
        if finite("thickness_m") <= 0.0:
            raise ValueError(f"{name}.thickness_m must be positive")
        if "n_after" in element and finite("n_after") <= 0.0:
            raise ValueError(f"{name}.n_after must be positive")
        for surface in ("front", "back"):
            for axis in ("", "_x", "_y"):
                key = f"radius_{surface}{axis}_m"
                if key in element and element[key] is not None:
                    finite(key)
    elif kind == "interface":
        if "n_after" not in element or finite("n_after") <= 0.0:
            raise ValueError(f"{name}.n_after must be positive for interface")
    elif kind in {"tilt", "coordinate_break"}:
        for axis in ("x", "y"):
            angle = finite(f"angle_{axis}_deg")
            if abs(angle) >= 89.0:
                raise ValueError(
                    f"{name}.angle_{axis}_deg must have magnitude below 89 degrees"
                )
    elif kind == "mirror":
        for key in ("radius_x_m", "radius_y_m"):
            if key in element and element[key] is not None:
                finite(key)
    elif kind == "aperture":
        validate_aperture(allow_legacy_radii=True)
    elif kind == "double_pass_lens_relay":
        for key in (
            "object_to_first_lens_m",
            "first_focal_length_m",
            "lens_separation_m",
            "second_focal_length_m",
            "second_lens_to_mirror_m",
        ):
            if key not in element or element[key] is None or finite(key) <= 0.0:
                raise ValueError(f"{name}.{key} must be positive and finite")
        from .optics import double_pass_lens_relay_matrix

        matrix = double_pass_lens_relay_matrix(element)
        a, b, _, _ = matrix.ravel()
        scale = max(
            abs(float(element["object_to_first_lens_m"])),
            abs(float(element["lens_separation_m"])),
            abs(float(element["second_lens_to_mirror_m"])),
            1.0,
        )
        if abs(b) > 1e-10 * scale or not math.isclose(
            abs(a), 1.0, rel_tol=0.0, abs_tol=1e-10
        ):
            raise ValueError(
                f"{name} must define unit-magnification B=0 imaging; "
                f"configured A={a:.9g}, B={b:.9g} m"
            )
    elif kind == "loss":
        response = element.get("spectral_response", {})
        if element.get("power_transmission") is None and not (
            isinstance(response, dict)
            and response.get("power_transmission") is not None
        ):
            raise ValueError(
                f"{name} loss needs power_transmission or a spectral power_transmission table"
            )

    if kind in {"thin_lens", "thick_lens"} and any(
        element.get(key) is not None
        for key in (
            "aperture_radius_m",
            "aperture_radius_x_m",
            "aperture_radius_y_m",
            "aperture_width_m",
            "aperture_height_m",
        )
    ):
        # Lens surface radii are optical powers, never aliases for a clear
        # aperture.  Requiring explicit aperture_* fields prevents a lens from
        # being silently clipped differently by the FFT and ray backends.
        validate_aperture(allow_legacy_radii=False)

    if "absorption_m_inv" in element and finite("absorption_m_inv") < 0.0:
        raise ValueError(f"{name}.absorption_m_inv must be non-negative")
    for key in ("incidence_angle_deg", "front_incidence_angle_deg", "back_incidence_angle_deg"):
        if key in element and abs(finite(key)) >= 90.0:
            raise ValueError(f"{name}.{key} must have magnitude below 90 degrees")
    for key in ("model", "front_model", "back_model"):
        if key in element and str(element[key]) not in {
            "ideal",
            "coating",
            "fresnel",
            "fresnel_angle",
            "direct_pass",
        }:
            raise ValueError(f"{name}.{key} is an unsupported interface model")
    if "polarization" in element and str(element["polarization"]) not in {
        "s",
        "p",
        "unpolarized",
    }:
        raise ValueError(f"{name}.polarization must be s, p, or unpolarized")

    raw_dispersion = element.get("dispersion")
    if raw_dispersion is not None:
        if not isinstance(raw_dispersion, dict):
            raise ValueError(f"{name}.dispersion must be an object")
        for key in (
            "constant_phase_rad",
            "reference_frequency_Hz",
            "reference_wavelength_m",
            "gd_s",
            "gdd_s2",
            "tod_s3",
            "fod_s4",
        ):
            if key in raw_dispersion and raw_dispersion[key] is not None:
                if isinstance(raw_dispersion[key], bool):
                    raise ValueError(
                        f"{name}.dispersion.{key} must be numeric, not a boolean"
                    )
                try:
                    value = float(raw_dispersion[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{name}.dispersion.{key} must be numeric"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(f"{name}.dispersion.{key} must be finite")
                if key in {"reference_frequency_Hz", "reference_wavelength_m"} and value <= 0.0:
                    raise ValueError(f"{name}.dispersion.{key} must be positive")
        has_frequency_dependent_phase = any(
            float(raw_dispersion.get(key) or 0.0) != 0.0
            for key in ("gd_s", "gdd_s2", "tod_s3", "fod_s4")
        )
        has_reference = raw_dispersion.get("reference_frequency_Hz") not in (
            None,
            0,
        ) or raw_dispersion.get("reference_wavelength_m") not in (None, 0)
        if has_frequency_dependent_phase and not has_reference:
            raise ValueError(
                f"{name}.dispersion needs reference_frequency_Hz or "
                "reference_wavelength_m when GD/GDD/TOD/FOD is nonzero"
            )
    if kind == "dispersive" and not raw_dispersion:
        raise ValueError(f"{name} dispersive element needs a non-empty dispersion object")


def _validate_incidence_angles(config: dict[str, Any], *, name: str) -> None:
    """Reject directions that cannot be represented by a z-marching solver."""

    for axis in ("x", "y"):
        key = f"incidence_angle_{axis}_deg"
        raw_angle = config.get(key)
        angle = float(0.0 if raw_angle is None else raw_angle)
        if not math.isfinite(angle) or abs(angle) >= 89.0:
            raise ValueError(
                f"{name}.{key} must be finite and have magnitude below 89 degrees"
            )


def _validate_fixed_point_acceleration(config: dict[str, Any], path: str) -> None:
    method = str(config.get("method", "anderson"))
    if method not in {"off", "anderson"}:
        raise ValueError(f"{path}.method must be off or anderson")
    require_exact_integer(
        config.get("history_depth", 4),
        name=f"{path}.history_depth",
        minimum=1,
    )
    require_exact_integer(
        config.get("start_iteration", 3),
        name=f"{path}.start_iteration",
        minimum=2,
    )
    damping = float(config.get("damping", 0.8))
    if not math.isfinite(damping) or not 0.0 < damping <= 1.0:
        raise ValueError(f"{path}.damping must lie in (0, 1]")
    regularization = float(config.get("regularization", 1e-10))
    if not math.isfinite(regularization) or regularization < 0.0:
        raise ValueError(f"{path}.regularization must be non-negative")
    predicted_ratio = float(
        config.get("maximum_predicted_residual_ratio", 0.99)
    )
    if not math.isfinite(predicted_ratio) or not 0.0 < predicted_ratio <= 1.0:
        raise ValueError(
            f"{path}.maximum_predicted_residual_ratio must lie in (0, 1]"
        )
    maximum_step_factor = float(config.get("maximum_step_factor", 5.0))
    if not math.isfinite(maximum_step_factor) or maximum_step_factor <= 0.0:
        raise ValueError(f"{path}.maximum_step_factor must be positive")
    restart_growth = float(config.get("restart_residual_growth_factor", 2.0))
    if not math.isfinite(restart_growth) or restart_growth <= 1.0:
        raise ValueError(
            f"{path}.restart_residual_growth_factor must be greater than 1"
        )
    maximum_memory = float(config.get("maximum_history_memory_MB", 128.0))
    if not math.isfinite(maximum_memory) or maximum_memory <= 0.0:
        raise ValueError(f"{path}.maximum_history_memory_MB must be positive")


def validate_config(config: dict[str, Any]) -> None:
    _canonicalize_config_aliases(config)
    policy = str(config.get("schema_policy", DEFAULTS["schema_policy"]))
    if policy not in {"strict_core_with_plugin_extras", "permissive"}:
        raise ValueError(
            "schema_policy must be strict_core_with_plugin_extras or permissive"
        )
    if policy == "strict_core_with_plugin_extras":
        _validate_core_schema(config, DEFAULTS)
    # Canonicalize public Pump definitions before any solver-facing checks.
    # This also makes direct ``AmplifierSimulator(dict)`` use consistent with
    # JSON loading and with the Streamlit form.
    canonicalize_pump_source(config["pump"])
    # The form derives energy from J/bin or J/nm uploads. The same JSON must
    # have the same energy when run directly or in a background worker.
    from .pulse_definition import resolve_seed_pulse_energy_J

    config["seed"]["pulse_energy_J"] = resolve_seed_pulse_energy_J(config["seed"])
    performance = config.get("performance", {})
    if performance.get("fft_backend", "auto") not in {"auto", "numpy", "scipy"}:
        raise ValueError("performance.fft_backend must be auto, numpy, or scipy")
    require_exact_integer(
        performance.get("fft_workers", 0),
        name="performance.fft_workers",
        minimum=0,
    )
    require_exact_integer(
        performance.get("fft_min_elements", 4096),
        name="performance.fft_min_elements",
        minimum=1,
    )
    if not isinstance(performance.get("operator_profiling", False), bool):
        raise ValueError("performance.operator_profiling must be boolean")
    multiphysics = config.get("multiphysics", {})
    coupling_model = str(multiphysics.get("coupling_model", "one_way"))
    if coupling_model not in {"one_way", "thermal_optical_fixed_point"}:
        raise ValueError(
            "multiphysics.coupling_model must be one_way or thermal_optical_fixed_point"
        )
    iteration_scheme = multiphysics.get("iteration_scheme", "nested")
    if iteration_scheme not in {"nested", "joint", "joint_frozen"}:
        raise ValueError("multiphysics.iteration_scheme must be nested, joint or joint_frozen")
    if iteration_scheme in {"joint", "joint_frozen"} and coupling_model == "thermal_optical_fixed_point":
        periodic_joint = config.get("simulation", {}).get("periodic_steady_state", {})
        if not periodic_joint.get("enabled", False) or periodic_joint.get("method") != "pulse_map":
            raise ValueError("Joint thermal-optical iteration requires enabled periodic pulse_map")
    for key in ("joint_cycles_per_iteration", "joint_consecutive_iterations", "frozen_probe_cycles"):
        if key in multiphysics and not (key == "joint_consecutive_iterations" and multiphysics[key] is None):
            require_exact_integer(multiphysics[key], name=f"multiphysics.{key}", minimum=1)
    minimum_coupling_iterations = require_exact_integer(
        multiphysics.get("minimum_iterations", 2),
        name="multiphysics.minimum_iterations",
        minimum=2,
    )
    maximum_coupling_iterations = require_exact_integer(
        multiphysics.get("maximum_iterations", 20),
        name="multiphysics.maximum_iterations",
        minimum=1,
    )
    if maximum_coupling_iterations < minimum_coupling_iterations:
        raise ValueError(
            "multiphysics.maximum_iterations must be >= minimum_iterations"
        )
    if (
        not math.isfinite(float(multiphysics.get("relative_tolerance", 1e-4)))
        or float(multiphysics.get("relative_tolerance", 1e-4)) <= 0.0
    ):
        raise ValueError("multiphysics.relative_tolerance must be positive")
    coupling_relaxation = float(multiphysics.get("relaxation", 0.5))
    if not math.isfinite(coupling_relaxation) or not 0.0 < coupling_relaxation <= 1.0:
        raise ValueError("multiphysics.relaxation must lie in (0, 1]")
    _validate_fixed_point_acceleration(
        multiphysics.get("acceleration", {}), "multiphysics.acceleration"
    )
    if multiphysics.get("seed_heat_coupling", "off") not in {
        "off",
        "one_way_postprocess",
        "fixed_point",
    }:
        raise ValueError(
            "multiphysics.seed_heat_coupling must be off, one_way_postprocess, or fixed_point"
        )
    if (
        multiphysics.get("seed_heat_coupling", "off") == "fixed_point"
        and coupling_model != "thermal_optical_fixed_point"
    ):
        raise ValueError(
            "seed_heat_coupling=fixed_point requires coupling_model=thermal_optical_fixed_point"
        )
    ase = config.get("ase", {})
    if ase.get("model", "off") not in {
        "off",
        "six_ray",
        "angular_discrete_ordinates",
    }:
        raise ValueError(
            "ase.model must be off, six_ray, or angular_discrete_ordinates"
        )
    if ase.get("failure_policy", "error") not in {"error", "disable_with_warning"}:
        raise ValueError("ase.failure_policy must be error or disable_with_warning")
    wavelengths = [float(value) for value in ase.get("wavelengths_m", [])]
    if any(value <= 0.0 or not math.isfinite(value) for value in wavelengths):
        raise ValueError("ase.wavelengths_m must contain positive finite values")
    weights = [float(value) for value in ase.get("spectral_weights", [])]
    if weights and (
        len(weights) != len(wavelengths)
        or any(value < 0.0 or not math.isfinite(value) for value in weights)
        or sum(weights) <= 0.0
    ):
        raise ValueError("ase.spectral_weights must be non-negative and match wavelengths_m")
    radiative_lifetime = ase.get("radiative_lifetime_s")
    if radiative_lifetime is not None and (
        not math.isfinite(float(radiative_lifetime))
        or float(radiative_lifetime) <= 0.0
    ):
        raise ValueError("ase.radiative_lifetime_s must be positive")
    solid_angle = float(ase.get("modeled_solid_angle_fraction", 1.0))
    if not math.isfinite(solid_angle) or not 0.0 <= solid_angle <= 1.0:
        raise ValueError("ase.modeled_solid_angle_fraction must lie in [0, 1]")
    for face, reflectivity in ase.get("face_reflectivity", {}).items():
        if face not in {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}:
            raise ValueError(f"unsupported ASE face {face!r}")
        if (
            not math.isfinite(float(reflectivity))
            or not 0.0 <= float(reflectivity) < 1.0
        ):
            raise ValueError("ASE effective face reflectivity must lie in [0, 1)")
    for key in ("population_relaxation", "boundary_relaxation"):
        value = float(ase.get(key, 0.5))
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"ase.{key} must lie in (0, 1]")
    require_exact_integer(
        ase.get("maximum_boundary_iterations", 40),
        name="ase.maximum_boundary_iterations",
        minimum=1,
    )
    require_exact_integer(
        ase.get("maximum_population_iterations", 12),
        name="ase.maximum_population_iterations",
        minimum=1,
    )
    angular_ase = ase.get("angular", {})
    direction_count = require_exact_integer(
        angular_ase.get("direction_count", 24),
        name="ase.angular.direction_count",
        minimum=8,
    )
    if direction_count < 8 or direction_count % 2:
        raise ValueError("ase.angular.direction_count must be an even integer >= 8")
    require_exact_integer(
        angular_ase.get("eigenmode_iterations", 30),
        name="ase.angular.eigenmode_iterations",
        minimum=1,
    )
    eigen_tolerance = float(angular_ase.get("eigenmode_tolerance", 1e-5))
    if not math.isfinite(eigen_tolerance) or eigen_tolerance <= 0.0:
        raise ValueError("ase.angular.eigenmode_tolerance must be positive")
    absolute_boundary_tolerance = float(
        angular_ase.get("absolute_boundary_tolerance_W_m2", 1e-12)
    )
    if (
        not math.isfinite(absolute_boundary_tolerance)
        or absolute_boundary_tolerance < 0.0
    ):
        raise ValueError(
            "ase.angular.absolute_boundary_tolerance_W_m2 must be finite and non-negative"
        )
    if not math.isfinite(float(angular_ase.get("azimuth_offset_deg", 0.0))):
        raise ValueError("ase.angular.azimuth_offset_deg must be finite")
    if (
        not math.isfinite(float(ase.get("relative_tolerance", 1e-5)))
        or float(ase.get("relative_tolerance", 1e-5)) <= 0.0
    ):
        raise ValueError("ase.relative_tolerance must be positive")
    if (
        not math.isfinite(float(ase.get("maximum_intensity_W_m2", 1e18)))
        or float(ase.get("maximum_intensity_W_m2", 1e18)) <= 0.0
    ):
        raise ValueError("ase.maximum_intensity_W_m2 must be positive")
    for key in ("size_x_m", "size_y_m", "length_z_m"):
        _positive(config, f"grid.{key}")
    for key in ("center_x_m", "center_y_m"):
        if not math.isfinite(float(config["grid"].get(key, 0.0))):
            raise ValueError(f"grid.{key} must be finite")
    for key in ("nx", "ny", "nz"):
        require_exact_integer(
            config["grid"][key],
            name=f"grid.{key}",
            minimum=3,
        )
    if config["grid"].get("spacing_xy", "uniform") not in {"uniform", "center_stretched", "refined_region"}:
        raise ValueError("grid.spacing_xy must be uniform, center_stretched or refined_region")
    for axis in ("x", "y"):
        raw_refinement = config["grid"].get(f"refinement_center_{axis}_m")
        if raw_refinement is None:
            continue
        refinement = float(raw_refinement)
        center = float(config["grid"].get(f"center_{axis}_m", 0.0))
        size = float(config["grid"][f"size_{axis}_m"])
        if not math.isfinite(refinement):
            raise ValueError(
                f"grid.refinement_center_{axis}_m must be finite or null"
            )
        if not center - 0.5 * size < refinement < center + 0.5 * size:
            raise ValueError(
                f"grid.refinement_center_{axis}_m must lie strictly inside "
                f"the physical grid {axis} extent"
            )
    for key in ("stretch_x", "stretch_y"):
        value = float(config["grid"].get(key, 1.0))
        if not math.isfinite(value) or value < 1.0:
            raise ValueError(f"grid.{key} must be at least 1")
    if config["grid"].get("spacing_xy") == "refined_region":
        from .grid import Grid3D
        Grid3D.from_config(config["grid"])
    optical_grid = config["grid"].get("optical", {})
    optical_grid_enabled = bool(optical_grid.get("enabled", False))
    pump_optical_grid = config["grid"].get("pump_optical", {})
    pump_optical_mode = str(pump_optical_grid.get("mode", "shared"))
    if pump_optical_mode not in {"shared", "physical", "custom"}:
        raise ValueError(
            "grid.pump_optical.mode must be shared, physical, or custom"
        )

    def validate_local_optical_grid(local: dict, path: str) -> None:
        for key in ("size_x_m", "size_y_m"):
            value = float(local.get(key, 0.0))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{path}.{key} must be positive")
        for key in ("center_x_m", "center_y_m"):
            if not math.isfinite(float(local.get(key, 0.0))):
                raise ValueError(f"{path}.{key} must be finite")
        for key in ("nx", "ny"):
            require_exact_integer(
                local.get(key, 0),
                name=f"{path}.{key}",
                minimum=8,
                maximum=1024,
            )
        physical_x_min = float(config["grid"].get("center_x_m", 0.0)) - float(config["grid"]["size_x_m"]) / 2.0
        physical_x_max = float(config["grid"].get("center_x_m", 0.0)) + float(config["grid"]["size_x_m"]) / 2.0
        physical_y_min = float(config["grid"].get("center_y_m", 0.0)) - float(config["grid"]["size_y_m"]) / 2.0
        physical_y_max = float(config["grid"].get("center_y_m", 0.0)) + float(config["grid"]["size_y_m"]) / 2.0
        optical_x_min = float(local.get("center_x_m", 0.0)) - float(local["size_x_m"]) / 2.0
        optical_x_max = float(local.get("center_x_m", 0.0)) + float(local["size_x_m"]) / 2.0
        optical_y_min = float(local.get("center_y_m", 0.0)) - float(local["size_y_m"]) / 2.0
        optical_y_max = float(local.get("center_y_m", 0.0)) + float(local["size_y_m"]) / 2.0
        tolerance = 1e-12 * max(float(config["grid"]["size_x_m"]), float(config["grid"]["size_y_m"]))
        if optical_x_min < physical_x_min - tolerance or optical_x_max > physical_x_max + tolerance:
            raise ValueError(f"{path} x window must lie inside the physical grid")
        if optical_y_min < physical_y_min - tolerance or optical_y_max > physical_y_max + tolerance:
            raise ValueError(f"{path} y window must lie inside the physical grid")

    if optical_grid_enabled:
        validate_local_optical_grid(optical_grid, "grid.optical")
    if pump_optical_mode == "custom":
        validate_local_optical_grid(pump_optical_grid, "grid.pump_optical")
    # Resolve this lazily so third-party beam registrations remain valid and
    # configuration module import does not acquire a beam/config cycle.
    from .beams import BEAM_REGISTRY

    for prefix in ("pump", "seed"):
        profile = str(config[prefix].get("profile", "super_gaussian"))
        if profile not in BEAM_REGISTRY:
            raise ValueError(f"{prefix}.profile is unsupported")
        _positive(config, f"{prefix}.wavelength_m")
        _positive(config, f"{prefix}.waist_x_m")
        _positive(config, f"{prefix}.waist_y_m")
        _positive(config, f"{prefix}.m2_x")
        _positive(config, f"{prefix}.m2_y")
        for axis in ("x", "y"):
            m2 = float(config[prefix][f"m2_{axis}"])
            if m2 < 1.0:
                raise ValueError(
                    f"{prefix}.m2_{axis} must be at least 1 for a physical paraxial beam"
                )
        _positive(config, f"{prefix}.order_x")
        _positive(config, f"{prefix}.order_y")
        for key in (
            "focus_z_m",
            "focus_x_m",
            "focus_y_m",
            "center_x_m",
            "center_y_m",
            "rotation_deg",
            "wavefront_radius_m",
        ):
            value = config[prefix].get(key)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{prefix}.{key} must be finite when supplied")
        source_index = float(config[prefix].get("source_refractive_index", 1.0))
        if not math.isfinite(source_index) or source_index <= 0.0:
            raise ValueError(
                f"{prefix}.source_refractive_index must be positive and finite"
            )
        _validate_incidence_angles(config[prefix], name=prefix)
        if profile == "annular":
            extras = config[prefix].get("extras", {})
            if not isinstance(extras, dict):
                raise ValueError(f"{prefix}.extras must be an object")
            peak_ratio = float(extras.get("inner_radius_ratio", 0.5))
            if not math.isfinite(peak_ratio) or peak_ratio < 0.0:
                raise ValueError(
                    f"{prefix}.extras.inner_radius_ratio must be finite and non-negative"
                )
            ring_order = float(
                extras.get("ring_order", 2.0 * peak_ratio**2)
            )
            if not math.isfinite(ring_order) or ring_order < 0.0:
                raise ValueError(
                    f"{prefix}.extras.ring_order must be finite and non-negative"
                )
    spatial_modes = config["seed"].get("spatial_modes", {})
    if spatial_modes.get("model", "effective_m2") not in {
        "effective_m2",
        "hermite_gaussian",
        "incoherent_hg",
    }:
        raise ValueError(
            "seed.spatial_modes.model must be effective_m2, hermite_gaussian, or incoherent_hg"
        )
    for axis in ("x", "y"):
        maximum_order = require_exact_integer(
            spatial_modes.get(f"maximum_order_{axis}", 0),
            name=f"seed.spatial_modes.maximum_order_{axis}",
            minimum=0,
        )
        if spatial_modes.get("model", "effective_m2") in {
            "hermite_gaussian",
            "incoherent_hg",
        }:
            required = math.ceil(max((float(config["seed"][f"m2_{axis}"]) - 1.0) / 2.0, 0.0))
            if maximum_order < required:
                raise ValueError(
                    f"seed M2 {axis}={config['seed'][f'm2_{axis}']!r} requires maximum HG order "
                    f"at least {required}; got {maximum_order}"
                )
    if spatial_modes.get("model", "effective_m2") == "incoherent_hg":
        maximum_modes = require_exact_integer(
            spatial_modes.get("maximum_propagated_modes", 64),
            name="seed.spatial_modes.maximum_propagated_modes",
            minimum=1,
        )
        maximum_x = require_exact_integer(
            spatial_modes.get("maximum_order_x", 0),
            name="seed.spatial_modes.maximum_order_x",
            minimum=0,
        )
        maximum_y = require_exact_integer(
            spatial_modes.get("maximum_order_y", 0),
            name="seed.spatial_modes.maximum_order_y",
            minimum=0,
        )
        maximum_total_raw = spatial_modes.get("maximum_total_order")
        if maximum_total_raw is None:
            requested_modes = (maximum_x + 1) * (maximum_y + 1)
        else:
            maximum_total = require_exact_integer(
                maximum_total_raw,
                name="seed.spatial_modes.maximum_total_order",
                minimum=0,
            )
            requested_modes = sum(
                1
                for order_x in range(maximum_x + 1)
                for order_y in range(maximum_y + 1)
                if order_x + order_y <= maximum_total
            )
            required_total_mean = 0.5 * (
                float(config["seed"]["m2_x"])
                + float(config["seed"]["m2_y"])
                - 2.0
            )
            if required_total_mean > maximum_total + 1e-12:
                raise ValueError(
                    "seed.spatial_modes.maximum_total_order cannot represent "
                    "the configured x/y M2 pair"
                )
        if maximum_modes < 1 or requested_modes > maximum_modes:
            raise ValueError(
                "incoherent_hg requested modal basis exceeds "
                "seed.spatial_modes.maximum_propagated_modes"
            )
        resolved_model = resolve_signal_model(config)
        if resolved_model == "josa_transport":
            raise ValueError(
                "incoherent_hg currently uses the phase-insensitive shared-reservoir "
                "transport; select signal_model=fluence, spectral_time, "
                "commercial_app_fast, or coherent_cpa"
            )
        if resolved_model == "coherent_cpa":
            coherent_hg = config["seed"].get("coherent", {})
            if str(
                coherent_hg.get("fidelity_mode", "configured")
            ) in {
                "axisymmetric_transient_gain",
                "reference_transient_gain",
            }:
                raise ValueError(
                    "coherent_cpa with incoherent_hg requires a Cartesian "
                    "whole-fluence fidelity mode (configured or "
                    "hybrid_whole_fluence); explicit HG modes are not "
                    "axisymmetric"
                )
            if bool(
                coherent_hg.get("transient_gain", {}).get(
                    "enabled", False
                )
            ):
                raise ValueError(
                    "coherent_cpa with incoherent_hg currently requires "
                    "seed.coherent.transient_gain.enabled=false"
                )
        if (
            str(config["seed"].get("temporal", {}).get("type", "gaussian")) == "cw"
            and resolved_model != "commercial_app_fast"
        ):
            raise ValueError(
                "incoherent_hg pulsed transport does not yet replace the CW average solver"
            )
        required_optical = "abcd" if resolved_model == "commercial_app_fast" else "fft"
        radial_hg_hybrid = (
            resolved_model == "coherent_cpa"
            and config["seed"].get("coherent", {}).get("fidelity_mode") == "axisymmetric_kerr"
            and resolve_physics(config).get("optical_propagation") == "hybrid_abcd_fft"
        )
        if not radial_hg_hybrid and resolve_physics(config).get("optical_propagation") != required_optical:
            raise ValueError(
                "incoherent_hg requires physics.optical_propagation="
                f"{required_optical} for the selected Seed backend"
            )
        spatio_spectral = config["seed"].get("spatio_spectral", {})
        if any(
            float(spatio_spectral.get(f"m2_scaling_exponent_{axis}", 0.0))
            != 0.0
            for axis in ("x", "y")
        ):
            raise ValueError(
                "incoherent_hg currently requires wavelength-independent M2 mode weights"
            )
    validate_seed_spatio_spectral(config["seed"])
    joint_time_frequency = config["seed"].get("joint_time_frequency", {})
    joint_model = str(joint_time_frequency.get("model", "separable"))
    if joint_model not in {"separable", "matrix"}:
        raise ValueError(
            "seed.joint_time_frequency.model must be separable or matrix"
        )
    if joint_model == "matrix":
        weights = np.asarray(joint_time_frequency.get("weights", []), dtype=float)
        if weights.ndim != 2 or min(weights.shape, default=0) < 1:
            raise ValueError(
                "seed.joint_time_frequency.weights must be a non-empty 2D matrix"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError(
                "seed.joint_time_frequency.weights must be finite and non-negative"
            )
        if float(np.sum(weights)) <= 0.0:
            raise ValueError(
                "seed.joint_time_frequency.weights must contain positive mass"
            )
        if resolve_signal_model(config) in {"commercial_app_fast", "coherent_cpa"}:
            raise ValueError(
                "commercial_app_fast/coherent_cpa derive their temporal closure from one complex input pulse; "
                "seed.joint_time_frequency.model=matrix is for phase-insensitive spectral_time transport"
            )
    _positive(config, "pump.power_W", allow_zero=True)
    if config["pump"].get("direction", "forward") not in {
        "forward",
        "backward",
    }:
        raise ValueError("pump.direction must be forward or backward")
    raytrace = config["pump"].get("raytrace", {})
    absorption_transport = config["pump"].get(
        "absorption_transport", {}
    )
    absorption_transport_model = str(
        absorption_transport.get("model", "population_dependent")
    )
    if absorption_transport_model not in {
        "population_dependent",
        "fixed_active_coefficient",
    }:
        raise ValueError(
            "pump.absorption_transport.model must be population_dependent "
            "or fixed_active_coefficient"
        )
    fixed_active_coefficient = absorption_transport.get(
        "fixed_active_coefficient_m_inv"
    )
    if absorption_transport_model == "fixed_active_coefficient":
        if (
            fixed_active_coefficient is None
            or not math.isfinite(float(fixed_active_coefficient))
            or float(fixed_active_coefficient) < 0.0
        ):
            raise ValueError(
                "pump.absorption_transport.fixed_active_coefficient_m_inv "
                "must be finite and non-negative for fixed_active_coefficient"
            )
    elif fixed_active_coefficient is not None and (
        not math.isfinite(float(fixed_active_coefficient))
        or float(fixed_active_coefficient) < 0.0
    ):
        raise ValueError(
            "pump.absorption_transport.fixed_active_coefficient_m_inv must "
            "be finite and non-negative when supplied"
        )
    require_exact_integer(
        raytrace.get("ray_count", 100000),
        name="pump.raytrace.ray_count",
        minimum=100,
    )
    require_exact_integer(
        raytrace.get("random_seed", 12345),
        name="pump.raytrace.random_seed",
        minimum=0,
    )
    if raytrace.get("angular_distribution", "auto") not in {
        "auto",
        "uniform_na",
        "uniform_solid_angle",
        "generalized_gaussian",
    }:
        raise ValueError(
            "pump.raytrace.angular_distribution must be auto, uniform_na, "
            "uniform_solid_angle, or generalized_gaussian"
        )
    for axis in ("x", "y"):
        angular_exponent = raytrace.get(
            f"angular_super_gaussian_exponent_{axis}"
        )
        if angular_exponent is not None and (
            not math.isfinite(float(angular_exponent))
            or float(angular_exponent) <= 0.0
        ):
            raise ValueError(
                "pump.raytrace.angular_super_gaussian_exponent_"
                f"{axis} must be positive and finite when supplied"
            )
    if raytrace.get("deposition_model", "nearest_cell_track_length") not in {
        "nearest_cell_track_length",
        "bilinear_track_length",
    }:
        raise ValueError(
            "pump.raytrace.deposition_model must be nearest_cell_track_length "
            "or bilinear_track_length"
        )
    if not isinstance(raytrace.get("thermal_refraction", True), bool):
        raise ValueError("pump.raytrace.thermal_refraction must be true or false")
    minimum_ray_power = float(
        raytrace.get("minimum_relative_ray_power", 1e-8)
    )
    if not math.isfinite(minimum_ray_power) or not 0.0 <= minimum_ray_power < 1.0:
        raise ValueError(
            "pump.raytrace.minimum_relative_ray_power must lie in [0, 1)"
        )
    maximum_step_fraction = float(
        raytrace.get("maximum_transverse_step_fraction", 0.5)
    )
    if not math.isfinite(maximum_step_fraction) or maximum_step_fraction <= 0.0:
        raise ValueError(
            "pump.raytrace.maximum_transverse_step_fraction must be positive"
        )
    require_exact_integer(
        raytrace.get("maximum_substeps_per_z_cell", 16),
        name="pump.raytrace.maximum_substeps_per_z_cell",
        minimum=1,
    )
    require_exact_integer(
        raytrace.get("processing_chunk_size", 200000),
        name="pump.raytrace.processing_chunk_size",
        minimum=1000,
    )
    _positive(config, "seed.pulse_energy_J", allow_zero=True)
    _positive(config, "seed.pulse_duration_s")
    _positive(config, "material.ion_density_m3")
    validate_doping_profile(config["material"])
    if "crystal_approximation" in config["material"]:
        raise ValueError(
            "material.crystal_approximation was removed because fitted fields must "
            "not replace propagation; use diagnostic-only material.field_diagnostics"
        )
    validate_field_diagnostics(
        config["material"].get("field_diagnostics", {})
    )
    validate_caustic_config(config["seed"].get("caustic_diagnostics", {}))
    _positive(config, "material.refractive_index")
    material_library = config["material"].get("library", {})
    if material_library:
        from .material_library import material_provenance_mismatches

        activation_policy = str(
            material_library.get("activation_policy", "runtime_ready")
        )
        if activation_policy != "runtime_ready":
            raise ValueError(
                "material.library.activation_policy is not runtime_ready; "
                "metadata-only material records cannot enter the solver"
            )
        record_digest = material_library.get("record_sha256")
        experimental_calibration = material_library.get(
            "experimental_calibration", {}
        )
        if not isinstance(experimental_calibration, dict):
            raise ValueError(
                "material.library.experimental_calibration must be an object"
            )
        if experimental_calibration:
            status = str(
                experimental_calibration.get("status", "unspecified")
            )
            if status not in {
                "unspecified",
                "not_required",
                "requires_sample_specific_calibration",
                "experimentally_calibrated",
            }:
                raise ValueError(
                    "material.library.experimental_calibration.status is unsupported"
                )
            for key in ("required_parameters", "calibrated_parameters"):
                values = experimental_calibration.get(key, [])
                if not isinstance(values, list) or any(
                    not isinstance(value, str) or not value.strip()
                    for value in values
                ):
                    raise ValueError(
                        f"material.library.experimental_calibration.{key} "
                        "must be a list of non-empty strings"
                    )
                if len(set(values)) != len(values):
                    raise ValueError(
                        f"material.library.experimental_calibration.{key} "
                        "must not contain duplicates"
                    )
            for key in ("sample_identifier", "dataset_sha256"):
                value = experimental_calibration.get(key)
                if value is not None and not isinstance(value, str):
                    raise ValueError(
                        f"material.library.experimental_calibration.{key} "
                        "must be a string or null"
                    )
        field_provenance = material_library.get("field_provenance", {})
        if not isinstance(field_provenance, dict):
            raise ValueError("material.library.field_provenance must be an object")
        mismatches = material_provenance_mismatches(config)
        if mismatches:
            raise ValueError(
                "material library-owned values differ from their applied "
                "snapshot without user_override provenance: "
                + ", ".join(mismatches)
            )
        for path, entry in field_provenance.items():
            if not isinstance(entry, dict):
                raise ValueError(
                    f"material.library.field_provenance[{path!r}] must be an object"
                )
            source = str(entry.get("source", ""))
            if source not in {
                "material_library_record",
                "user_override",
                "default",
                "derived",
            }:
                raise ValueError(
                    f"material.library.field_provenance[{path!r}] has unsupported source {source!r}"
                )
            if source == "material_library_record" and (
                record_digest is None
                or str(entry.get("record_sha256", ""))
                != str(record_digest)
            ):
                raise ValueError(
                    "material library field provenance mixes record digests; "
                    f"{path} does not belong to the active material record"
                )
    pump_index = config["material"].get("pump_refractive_index")
    if pump_index is not None and (
        not math.isfinite(float(pump_index)) or float(pump_index) <= 0.0
    ):
        raise ValueError("material.pump_refractive_index must be positive when supplied")
    for key in (
        "reference_temperature_K",
        "material_data_reference_temperature_K",
        "operating_temperature_K",
    ):
        value = float(config["material"].get(key, 293.15))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"material.{key} must be positive and finite")
    if not math.isfinite(float(config["material"].get("dn_dT_per_K", 0.0))):
        raise ValueError("material.dn_dT_per_K must be finite")
    dn_dT_definition = str(
        config["material"].get(
            "dn_dT_definition", "measured_free_expansion"
        )
    )
    if dn_dT_definition not in {
        "measured_free_expansion",
        "intrinsic_fixed_strain",
    }:
        raise ValueError(
            "material.dn_dT_definition must be measured_free_expansion or intrinsic_fixed_strain"
        )
    geometry = config["material"].get("geometry", {})
    if geometry.get("shape", "cuboid") not in {"cuboid", "cylinder"}:
        raise ValueError("material.geometry.shape must be cuboid or cylinder")
    for key in ("center_x_m", "center_y_m"):
        if not math.isfinite(float(geometry.get(key, 0.0))):
            raise ValueError(f"material.geometry.{key} must be finite")
    for key in ("width_x_m", "height_y_m", "diameter_m"):
        if geometry.get(key) is not None and (
            not math.isfinite(float(geometry[key])) or float(geometry[key]) <= 0.0
        ):
            raise ValueError(f"material.geometry.{key} must be positive when supplied")
    grid_center_x = float(config["grid"].get("center_x_m", 0.0))
    grid_center_y = float(config["grid"].get("center_y_m", 0.0))
    grid_half_x = 0.5 * float(config["grid"]["size_x_m"])
    grid_half_y = 0.5 * float(config["grid"]["size_y_m"])
    crystal_center_x = float(geometry.get("center_x_m", 0.0))
    crystal_center_y = float(geometry.get("center_y_m", 0.0))
    if geometry.get("shape", "cuboid") == "cuboid":
        width_x = geometry.get("width_x_m")
        height_y = geometry.get("height_y_m")
        crystal_half_x = 0.5 * float(
            config["grid"]["size_x_m"] if width_x is None else width_x
        )
        crystal_half_y = 0.5 * float(
            config["grid"]["size_y_m"] if height_y is None else height_y
        )
    else:
        diameter = geometry.get("diameter_m")
        crystal_radius = 0.5 * float(
            min(config["grid"]["size_x_m"], config["grid"]["size_y_m"])
            if diameter is None
            else diameter
        )
        crystal_half_x = crystal_radius
        crystal_half_y = crystal_radius
    geometry_tolerance = (
        1e-12
        * max(float(config["grid"]["size_x_m"]), float(config["grid"]["size_y_m"]))
    )
    if (
        abs(crystal_center_x - grid_center_x) + crystal_half_x
        > grid_half_x + geometry_tolerance
    ):
        raise ValueError(
            "material.geometry x extent must lie inside the physical grid; "
            "otherwise the modeled crystal is silently truncated"
        )
    if (
        abs(crystal_center_y - grid_center_y) + crystal_half_y
        > grid_half_y + geometry_tolerance
    ):
        raise ValueError(
            "material.geometry y extent must lie inside the physical grid; "
            "otherwise the modeled crystal is silently truncated"
        )
    if not isinstance(geometry.get("clip_optical_field", True), bool):
        raise ValueError("material.geometry.clip_optical_field must be true or false")
    boundary = config["material"].get("thermal", {}).get("boundary", {})
    thermal = config["material"].get("thermal", {})
    if thermal.get("linear_solver", "auto") not in {"auto", "cg", "sor"}:
        raise ValueError("material.thermal.linear_solver must be auto, cg, or sor")
    if (
        not math.isfinite(float(thermal.get("cg_relative_tolerance", 1e-8)))
        or float(thermal.get("cg_relative_tolerance", 1e-8)) <= 0.0
    ):
        raise ValueError("material.thermal.cg_relative_tolerance must be positive")
    for tolerance_name, default in (
        ("algebraic_relative_tolerance", 1e-8),
        ("energy_balance_relative_tolerance", 1e-10),
    ):
        tolerance_value = float(thermal.get(tolerance_name, default))
        if not math.isfinite(tolerance_value) or tolerance_value <= 0.0:
            raise ValueError(
                f"material.thermal.{tolerance_name} must be positive and finite"
            )
    embedded_area_model = str(
        thermal.get("embedded_boundary_area_model", "cartesian")
    )
    if embedded_area_model not in {"cartesian", "analytic_cylinder"}:
        raise ValueError(
            "material.thermal.embedded_boundary_area_model must be cartesian or analytic_cylinder"
        )
    if embedded_area_model == "analytic_cylinder" and geometry.get("shape") != "cylinder":
        raise ValueError(
            "analytic_cylinder thermal boundary area requires material.geometry.shape=cylinder"
        )
    configured_heat_fraction = thermal.get("heat_fraction", "quantum_defect")
    if isinstance(configured_heat_fraction, str):
        if configured_heat_fraction not in {
            "quantum_defect",
            "lifetime_quantum_defect",
            "scaled_quantum_defect",
            "spectral_quantum_defect",
            "spectral_scaled_quantum_defect",
        }:
            raise ValueError(
                "material.thermal.heat_fraction must be quantum_defect, "
                "lifetime_quantum_defect, "
                "scaled_quantum_defect, spectral_quantum_defect, "
                "spectral_scaled_quantum_defect, or a numeric fraction"
            )
    elif (
        not math.isfinite(float(configured_heat_fraction))
        or not 0.0 <= float(configured_heat_fraction) <= 1.0
    ):
        raise ValueError("numeric material.thermal.heat_fraction must lie in [0, 1]")
    stokes_multiplier = float(thermal.get("stokes_shift_multiplier", 1.0))
    if not math.isfinite(stokes_multiplier) or stokes_multiplier < 0.0:
        raise ValueError("material.thermal.stokes_shift_multiplier must be finite and non-negative")
    conductivity = thermal.get("conductivity_W_mK", 0.0)
    if isinstance(conductivity, bool):
        raise ValueError(
            "material.thermal.conductivity_W_mK must be numeric, not a boolean"
        )
    conductivity_values = [conductivity] * 3 if isinstance(conductivity, (int, float)) else conductivity
    if (
        not isinstance(conductivity_values, (list, tuple))
        or len(conductivity_values) != 3
        or any(
            isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0
            for value in conductivity_values
        )
    ):
        raise ValueError("material.thermal.conductivity_W_mK must be positive scalar or x/y/z list")

    thermal_entities = list(thermal.get("entities", []))
    entity_names: list[str] = []
    for index, entity in enumerate(thermal_entities):
        if not isinstance(entity, dict):
            raise ValueError(
                f"material.thermal.entities[{index}] must be an object"
            )
        name = str(entity.get("name", "")).strip()
        if not name:
            raise ValueError(
                f"material.thermal.entities[{index}].name must be non-empty"
            )
        entity_names.append(name)
        entity_conductivity = entity.get(
            "conductivity_W_mK", conductivity
        )
        if isinstance(entity_conductivity, bool):
            raise ValueError(
                f"material.thermal.entities[{index}].conductivity_W_mK "
                "must be numeric, not a boolean"
            )
        entity_conductivity_values = (
            [entity_conductivity] * 3
            if isinstance(entity_conductivity, (int, float))
            else entity_conductivity
        )
        if (
            not isinstance(entity_conductivity_values, (list, tuple))
            or len(entity_conductivity_values) != 3
            or any(
                isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0
                for value in entity_conductivity_values
            )
        ):
            raise ValueError(
                f"material.thermal.entities[{index}].conductivity_W_mK "
                "must be a positive scalar or x/y/z list"
            )
        for key, fallback in (
            ("density_kg_m3", thermal.get("density_kg_m3")),
            ("heat_capacity_J_kgK", thermal.get("heat_capacity_J_kgK")),
        ):
            if isinstance(entity.get(key, fallback), bool):
                raise ValueError(
                    f"material.thermal.entities[{index}].{key} must be numeric, "
                    "not a boolean"
                )
            value = float(entity.get(key, fallback))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"material.thermal.entities[{index}].{key} must be positive"
                )
        region = entity.get("region", {"shape": "material_mask"})
        if not isinstance(region, dict):
            raise ValueError(
                f"material.thermal.entities[{index}].region must be an object"
            )
        region_shape = str(region.get("shape", "material_mask"))
        if not isinstance(region.get("exclude_material_mask", False), bool):
            raise ValueError(
                f"material.thermal.entities[{index}].region."
                "exclude_material_mask must be boolean"
            )
        if region_shape not in {
            "material_mask",
            "grid",
            "cuboid",
            "cylinder_z",
        }:
            raise ValueError(
                f"material.thermal.entities[{index}].region.shape must be "
                "material_mask, grid, cuboid, or cylinder_z"
            )
        if region_shape == "cuboid":
            for minimum_key, maximum_key in (
                ("x_min_m", "x_max_m"),
                ("y_min_m", "y_max_m"),
                ("z_min_m", "z_max_m"),
            ):
                minimum = region.get(minimum_key)
                maximum = region.get(maximum_key)
                if isinstance(minimum, bool) or isinstance(maximum, bool):
                    raise ValueError(
                        f"material.thermal.entities[{index}].region bounds "
                        "must be numeric, not boolean"
                    )
                if minimum is not None and not math.isfinite(float(minimum)):
                    raise ValueError(
                        f"material.thermal.entities[{index}].region."
                        f"{minimum_key} must be finite"
                    )
                if maximum is not None and not math.isfinite(float(maximum)):
                    raise ValueError(
                        f"material.thermal.entities[{index}].region."
                        f"{maximum_key} must be finite"
                    )
                if (
                    minimum is not None
                    and maximum is not None
                    and float(minimum) >= float(maximum)
                ):
                    raise ValueError(
                        f"material.thermal.entities[{index}].region "
                        f"requires {minimum_key} < {maximum_key}"
                    )
        if region_shape == "cylinder_z":
            diameter = region.get("diameter_m")
            if isinstance(diameter, bool):
                raise ValueError(
                    f"material.thermal.entities[{index}].region."
                    "diameter_m must be numeric, not a boolean"
                )
            if (
                diameter is None
                or not math.isfinite(float(diameter))
                or float(diameter) <= 0.0
            ):
                raise ValueError(
                    f"material.thermal.entities[{index}].region."
                    "diameter_m must be positive"
                )
    if len(entity_names) != len(set(entity_names)):
        raise ValueError("material.thermal entity names must be unique")
    if thermal_entities and embedded_area_model != "cartesian":
        raise ValueError(
            "multi-entity thermal conduction currently requires "
            "embedded_boundary_area_model=cartesian"
        )
    contact_pairs: set[tuple[str, str]] = set()
    for index, contact in enumerate(thermal.get("contacts", [])):
        if not isinstance(contact, dict):
            raise ValueError(
                f"material.thermal.contacts[{index}] must be an object"
            )
        first = str(contact.get("entity_a", ""))
        second = str(contact.get("entity_b", ""))
        if (
            first not in entity_names
            or second not in entity_names
            or first == second
        ):
            raise ValueError(
                f"material.thermal.contacts[{index}] must reference two "
                "distinct configured entities"
            )
        pair = tuple(sorted((first, second)))
        if pair in contact_pairs:
            raise ValueError(
                f"material.thermal.contacts[{index}] duplicates entity pair"
            )
        contact_pairs.add(pair)
        resistance = float(
            contact.get("thermal_contact_resistance_m2K_W", 0.0)
        )
        if isinstance(
            contact.get("thermal_contact_resistance_m2K_W", 0.0), bool
        ):
            raise ValueError(
                f"material.thermal.contacts[{index}]."
                "thermal_contact_resistance_m2K_W must be numeric, not a boolean"
            )
        if not math.isfinite(resistance) or resistance < 0.0:
            raise ValueError(
                f"material.thermal.contacts[{index}]."
                "thermal_contact_resistance_m2K_W must be finite and non-negative"
            )
    for key in ("density_kg_m3", "heat_capacity_J_kgK"):
        value = float(thermal.get(key, 0.0))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"material.thermal.{key} must be positive and finite")
    require_exact_integer(
        thermal.get("max_iterations", 0),
        name="material.thermal.max_iterations",
        minimum=1,
    )
    tolerance_K = float(thermal.get("tolerance_K", 0.0))
    if not math.isfinite(tolerance_K) or tolerance_K <= 0.0:
        raise ValueError("material.thermal.tolerance_K must be positive and finite")
    thermal_relaxation = float(thermal.get("relaxation", 1.8))
    if not math.isfinite(thermal_relaxation) or not 0.0 < thermal_relaxation < 2.0:
        raise ValueError("material.thermal.relaxation must lie in (0, 2)")
    transient_thermal = thermal.get("transient", {})
    if (
        not math.isfinite(float(transient_thermal.get("duration_s", 0.0)))
        or float(transient_thermal.get("duration_s", 0.0)) <= 0.0
    ):
        raise ValueError("material.thermal.transient.duration_s must be positive")
    require_exact_integer(
        transient_thermal.get("time_steps", 0),
        name="material.thermal.transient.time_steps",
        minimum=1,
    )
    require_exact_integer(
        transient_thermal.get("max_iterations_per_step", 0),
        name="material.thermal.transient.max_iterations_per_step",
        minimum=1,
    )
    initial_temperature = transient_thermal.get("initial_temperature_K")
    if initial_temperature is not None and (
        not math.isfinite(float(initial_temperature))
        or float(initial_temperature) <= 0.0
    ):
        raise ValueError("material.thermal.transient.initial_temperature_K must be positive when supplied")
    transient_relaxation = transient_thermal.get("relaxation")
    if transient_relaxation is not None and (
        not math.isfinite(float(transient_relaxation))
        or not 0.0 < float(transient_relaxation) < 2.0
    ):
        raise ValueError("material.thermal.transient.relaxation must lie in (0, 2)")
    boundary_aliases = {
        "dirichlet", "fixed", "fixed_temperature", "convective", "convection", "water", "adiabatic"
    }
    if str(boundary.get("type", "dirichlet")) not in boundary_aliases:
        raise ValueError("material.thermal.boundary.type is unsupported")
    faces = boundary.get("faces", {}) or {}
    if not isinstance(faces, dict) or not set(faces).issubset(
        {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
    ):
        raise ValueError("thermal boundary faces must use x_min/x_max/y_min/y_max/z_min/z_max")
    for face_name, face in faces.items():
        face_type = face if isinstance(face, str) else face.get("type", boundary.get("type", "dirichlet"))
        if str(face_type) not in boundary_aliases:
            raise ValueError(f"Unsupported thermal boundary type on {face_name}")
    boundary_sets = [("boundary", boundary)]
    boundary_sets.extend(
        (
            f"boundary.faces.{face_name}",
            {**boundary, **({"type": face} if isinstance(face, str) else face)},
        )
        for face_name, face in faces.items()
    )
    for boundary_name, boundary_cfg in boundary_sets:
        boundary_type = str(boundary_cfg.get("type", "dirichlet"))
        if boundary_type in {"dirichlet", "fixed", "fixed_temperature"}:
            temperature = float(
                boundary_cfg.get(
                    "temperature_K",
                    boundary_cfg.get("ambient_temperature_K", 293.15),
                )
            )
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise ValueError(
                    f"material.thermal.{boundary_name} fixed temperature must be positive and finite"
                )
        elif boundary_type in {"convective", "convection", "water"}:
            coolant = float(
                boundary_cfg.get(
                    "coolant_temperature_K",
                    boundary_cfg.get("ambient_temperature_K", 293.15),
                )
            )
            heat_transfer = float(boundary_cfg.get("heat_transfer_W_m2K", 0.0))
            if not math.isfinite(coolant) or coolant <= 0.0:
                raise ValueError(
                    f"material.thermal.{boundary_name} coolant temperature must be positive and finite"
                )
            if not math.isfinite(heat_transfer) or heat_transfer <= 0.0:
                raise ValueError(
                    f"material.thermal.{boundary_name} heat_transfer_W_m2K must be positive and finite"
                )
    mechanical = config["material"].get("mechanical", {})
    stress_model = str(mechanical.get("stress_model", "none"))
    if stress_model not in {
        "none",
        "isotropic_proxy",
        "thermoelastic_fem",
        "anisotropic_thermoelastic_fem",
    }:
        raise ValueError(
            "material.mechanical.stress_model must be none, isotropic_proxy, "
            "thermoelastic_fem, or anisotropic_thermoelastic_fem"
        )
    if stress_model in {"isotropic_proxy", "thermoelastic_fem"}:
        young = float(mechanical.get("young_modulus_Pa", 0.0))
        poisson = float(mechanical.get("poisson_ratio", 0.25))
        expansion = float(mechanical.get("thermal_expansion_per_K", 0.0))
        photoelastic = float(
            mechanical.get("photoelastic_coefficient_Pa_inv", 0.0)
        )
        if not math.isfinite(young) or young <= 0.0:
            raise ValueError(
                "material.mechanical.young_modulus_Pa must be positive and finite"
            )
        if not math.isfinite(poisson) or not -1.0 < poisson < 0.5:
            raise ValueError(
                "material.mechanical.poisson_ratio must lie in (-1, 0.5)"
            )
        if not math.isfinite(expansion):
            raise ValueError(
                "material.mechanical.thermal_expansion_per_K must be finite"
            )
        if not math.isfinite(photoelastic):
            raise ValueError(
                "material.mechanical.photoelastic_coefficient_Pa_inv must be finite"
            )
    if stress_model in {"thermoelastic_fem", "anisotropic_thermoelastic_fem"}:
        fem = mechanical.get("thermoelastic_fem", {})
        require_exact_integer(
            fem.get("maximum_active_elements", 8000),
            name=(
                "material.mechanical.thermoelastic_fem."
                "maximum_active_elements"
            ),
            minimum=1,
        )
        if str(fem.get("linear_solver", "auto")) not in {"auto", "direct", "cg"}:
            raise ValueError(
                "material.mechanical.thermoelastic_fem.linear_solver must be auto, direct, or cg"
            )
        if str(fem.get("photoelastic_coupling", "off")) not in {
            "off",
            "scalar_von_mises_proxy",
            "tensor_photoelastic",
        }:
            raise ValueError(
                "material.mechanical.thermoelastic_fem.photoelastic_coupling must be "
                "off, scalar_von_mises_proxy, or tensor_photoelastic"
            )
        if (
            str(fem.get("photoelastic_coupling", "off"))
            == "tensor_photoelastic"
            and stress_model != "anisotropic_thermoelastic_fem"
        ):
            raise ValueError(
                "tensor_photoelastic requires stress_model=anisotropic_thermoelastic_fem"
            )
        fem_tolerance = float(fem.get("relative_tolerance", 1e-8))
        if not math.isfinite(fem_tolerance) or fem_tolerance <= 0.0:
            raise ValueError(
                "material.mechanical.thermoelastic_fem.relative_tolerance must be positive and finite"
            )
        require_exact_integer(
            fem.get("maximum_iterations", 2000),
            name="material.mechanical.thermoelastic_fem.maximum_iterations",
            minimum=1,
        )
        fixed_faces = fem.get("fixed_faces", [])
        allowed_mechanical_faces = {
            "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"
        }
        if (
            not isinstance(fixed_faces, list)
            or len(set(fixed_faces)) != len(fixed_faces)
            or any(face not in allowed_mechanical_faces for face in fixed_faces)
        ):
            raise ValueError(
                "material.mechanical.thermoelastic_fem.fixed_faces must be a unique list of crystal faces"
            )
    if stress_model == "anisotropic_thermoelastic_fem":
        anisotropic = mechanical.get("anisotropic", {})
        if anisotropic.get(
            "voigt_order", ["xx", "yy", "zz", "xy", "yz", "xz"]
        ) != ["xx", "yy", "zz", "xy", "yz", "xz"]:
            raise ValueError(
                "material.mechanical.anisotropic.voigt_order must be "
                "[xx, yy, zz, xy, yz, xz]; reorder external tensors before use"
            )
        if (
            anisotropic.get(
                "voigt_shear_convention",
                "engineering_strain_tensor_stress",
            )
            != "engineering_strain_tensor_stress"
        ):
            raise ValueError(
                "material.mechanical.anisotropic.voigt_shear_convention must be "
                "engineering_strain_tensor_stress; convert tensor-shear source "
                "data before use"
            )
        strain_definition = str(
            anisotropic.get(
                "photoelastic_strain_definition", "elastic_strain"
            )
        )
        if strain_definition not in {"elastic_strain", "total_strain"}:
            raise ValueError(
                "anisotropic.photoelastic_strain_definition must be elastic_strain or total_strain"
            )
        stiffness = np.asarray(
            anisotropic.get("elastic_stiffness_Pa", []), dtype=float
        )
        expansion = np.asarray(
            anisotropic.get("thermal_expansion_tensor_per_K", []), dtype=float
        )
        if stiffness.shape != (6, 6) or not np.all(np.isfinite(stiffness)):
            raise ValueError(
                "material.mechanical.anisotropic.elastic_stiffness_Pa must be a finite 6x6 matrix"
            )
        if not np.allclose(stiffness, stiffness.T, rtol=1e-10, atol=1e-3):
            raise ValueError("anisotropic elastic stiffness must be symmetric")
        if float(np.min(np.linalg.eigvalsh(0.5 * (stiffness + stiffness.T)))) <= 0.0:
            raise ValueError("anisotropic elastic stiffness must be positive definite")
        if expansion.shape != (6,) or not np.all(np.isfinite(expansion)):
            raise ValueError(
                "material.mechanical.anisotropic.thermal_expansion_tensor_per_K must contain six finite Voigt components"
            )
        orientation = np.asarray(
            anisotropic.get("orientation_matrix_material_to_lab", []),
            dtype=float,
        )
        if orientation.size:
            if orientation.shape != (3, 3) or not np.all(np.isfinite(orientation)):
                raise ValueError(
                    "orientation_matrix_material_to_lab must be a finite 3x3 matrix"
                )
            if not np.allclose(
                orientation.T @ orientation, np.eye(3), rtol=1e-10, atol=1e-10
            ) or not np.isclose(np.linalg.det(orientation), 1.0, atol=1e-10):
                raise ValueError(
                    "orientation_matrix_material_to_lab must be a proper orthonormal rotation"
                )
        else:
            euler = np.asarray(
                anisotropic.get("orientation_euler_xyz_deg", [0.0, 0.0, 0.0]),
                dtype=float,
            )
            if euler.shape != (3,) or not np.all(np.isfinite(euler)):
                raise ValueError(
                    "orientation_euler_xyz_deg must contain three finite angles"
                )
        if (
            mechanical.get("thermoelastic_fem", {}).get(
                "photoelastic_coupling", "off"
            )
            == "tensor_photoelastic"
        ):
            photoelastic = np.asarray(
                anisotropic.get("photoelastic_matrix", []), dtype=float
            )
            reference_index = anisotropic.get("reference_refractive_index")
            if photoelastic.shape != (6, 6) or not np.all(
                np.isfinite(photoelastic)
            ):
                raise ValueError(
                    "tensor_photoelastic requires a finite anisotropic.photoelastic_matrix 6x6"
                )
            if reference_index is None or not math.isfinite(
                float(reference_index)
            ) or float(reference_index) <= 0.0:
                raise ValueError(
                    "tensor_photoelastic requires positive anisotropic.reference_refractive_index"
                )
            if (
                strain_definition == "total_strain"
                and dn_dT_definition == "measured_free_expansion"
                and float(config["material"].get("dn_dT_per_K", 0.0)) != 0.0
            ):
                raise ValueError(
                    "tensor_photoelastic with total_strain would double count free thermal expansion "
                    "when material.dn_dT_definition=measured_free_expansion; use elastic_strain or "
                    "provide an intrinsic_fixed_strain dn/dT"
                )
            if (
                strain_definition == "elastic_strain"
                and dn_dT_definition == "intrinsic_fixed_strain"
                and np.any(photoelastic @ expansion != 0.0)
            ):
                raise ValueError(
                    "tensor_photoelastic with elastic_strain and an "
                    "intrinsic_fixed_strain dn/dT omits free thermal expansion "
                    "from the refractive-index response; use total_strain or "
                    "provide a measured_free_expansion dn/dT"
                )
    endface = mechanical.get("endface_deformation", {})
    if endface.get("model", "off") not in {"off", "distributed_opd"}:
        raise ValueError(
            "material.mechanical.endface_deformation.model must be off or distributed_opd"
        )
    outside_index = float(endface.get("outside_refractive_index", 1.0))
    if not math.isfinite(outside_index) or outside_index <= 0.0:
        raise ValueError(
            "material.mechanical.endface_deformation.outside_refractive_index must be positive"
        )
    if endface.get("model", "off") != "off" and stress_model not in {
        "thermoelastic_fem",
        "anisotropic_thermoelastic_fem",
    }:
        raise ValueError(
            "endface deformation feedback requires a thermoelastic FEM stress model"
        )
    _positive(config, "simulation.pump_duration_s", allow_zero=True)
    require_exact_integer(
        config["simulation"]["pump_time_steps"],
        name="simulation.pump_time_steps",
        minimum=1,
    )
    _positive(config, "simulation.repetition_rate_Hz", allow_zero=True)
    require_exact_integer(
        config["simulation"]["pulse_train_cycles"],
        name="simulation.pulse_train_cycles",
        minimum=1,
    )
    pulse_train_initialization = str(
        config["simulation"].get(
            "pulse_train_initialization",
            "configured_initial_population",
        )
    )
    if pulse_train_initialization not in {
        "configured_initial_population",
        "cycle_average",
        "auto",
    }:
        raise ValueError(
            "simulation.pulse_train_initialization must be "
            "configured_initial_population, cycle_average, or auto"
        )
    first_pulse_delay = config["simulation"].get("first_pulse_delay_s")
    if first_pulse_delay is not None:
        delay_s = float(first_pulse_delay)
        repetition_rate = float(config["simulation"]["repetition_rate_Hz"])
        if not math.isfinite(delay_s) or delay_s <= 0.0:
            raise ValueError(
                "simulation.first_pulse_delay_s must be positive and finite "
                "when supplied; use null to retain the legacy one-period "
                "first-pulse timing"
            )
        if repetition_rate <= 0.0:
            raise ValueError(
                "simulation.first_pulse_delay_s requires "
                "simulation.repetition_rate_Hz > 0"
            )
        if delay_s > 1.0 / repetition_rate:
            raise ValueError(
                "simulation.first_pulse_delay_s must not exceed one Seed "
                "repetition period"
            )
        if bool(
            config["simulation"].get("periodic_steady_state", {}).get(
                "enabled", False
            )
        ):
            raise ValueError(
                "simulation.first_pulse_delay_s applies only to a finite "
                "pulse train, not periodic steady state"
            )
        if pulse_train_initialization != "configured_initial_population":
            raise ValueError(
                "simulation.first_pulse_delay_s requires "
                "pulse_train_initialization=configured_initial_population "
                "so t=0 has an unambiguous physical state"
            )
    require_exact_integer(
        config["simulation"]["pump_time_steps_per_cycle"],
        name="simulation.pump_time_steps_per_cycle",
        minimum=1,
    )
    periodic = config["simulation"].get("periodic_steady_state", {})
    periodic_method = str(periodic.get("method", "pulse_map"))
    if periodic_method not in {
        "pulse_map",
        "reduced_pulse_map",
        "cycle_average",
    }:
        raise ValueError(
            "periodic steady-state method must be pulse_map or cycle_average "
            "(legacy reduced_pulse_map is accepted as an alias for pulse_map "
            "with a reduced initializer)"
        )
    minimum_cycles = require_exact_integer(
        periodic.get("minimum_cycles", 1),
        name="simulation.periodic_steady_state.minimum_cycles",
        minimum=1,
    )
    maximum_cycles = require_exact_integer(
        periodic.get("maximum_cycles", 1),
        name="simulation.periodic_steady_state.maximum_cycles",
        minimum=1,
    )
    if maximum_cycles < minimum_cycles:
        raise ValueError("periodic steady-state maximum_cycles must be >= minimum_cycles")
    if (
        not math.isfinite(float(periodic.get("relative_tolerance", 0.0)))
        or float(periodic.get("relative_tolerance", 0.0)) <= 0.0
    ):
        raise ValueError("periodic steady-state relative_tolerance must be positive")
    consecutive_cycles = require_exact_integer(
        periodic.get("consecutive_cycles", 1),
        name="simulation.periodic_steady_state.consecutive_cycles",
        minimum=1,
    )
    minimum_viable_maximum = minimum_cycles + consecutive_cycles - 1
    if maximum_cycles < minimum_viable_maximum:
        raise ValueError(
            "periodic steady-state maximum_cycles must be at least "
            "minimum_cycles + consecutive_cycles - 1 so convergence is reachable"
        )
    average_minimum = require_exact_integer(
        periodic.get("average_minimum_iterations", 1),
        name=(
            "simulation.periodic_steady_state.average_minimum_iterations"
        ),
        minimum=1,
    )
    average_maximum = require_exact_integer(
        periodic.get("average_maximum_iterations", 1),
        name=(
            "simulation.periodic_steady_state.average_maximum_iterations"
        ),
        minimum=1,
    )
    if average_maximum < average_minimum:
        raise ValueError("cycle-average maximum iterations must be >= minimum iterations")
    relaxation = float(periodic.get("average_relaxation", 0.5))
    if not math.isfinite(relaxation) or not 0.0 < relaxation <= 1.0:
        raise ValueError("cycle-average relaxation must lie in (0, 1]")
    initialization = str(periodic.get("pulse_map_initialization", "auto"))
    if initialization not in {
        "initial_population",
        "cycle_average",
        "reduced_pulse_map",
        "auto",
    }:
        raise ValueError(
            "simulation.periodic_steady_state.pulse_map_initialization must be "
            "initial_population, cycle_average, reduced_pulse_map, or auto"
        )
    reduced_initializer = bool(
        periodic_method == "reduced_pulse_map"
        or (
            periodic_method == "pulse_map"
            and initialization == "reduced_pulse_map"
        )
    )
    initializer_threshold = float(
        periodic.get("cycle_average_initializer_minimum_lifetime_periods", 20.0)
    )
    if not math.isfinite(initializer_threshold) or initializer_threshold < 0.0:
        raise ValueError(
            "cycle-average initializer minimum lifetime periods must be non-negative"
        )
    population_violation = float(
        periodic.get("maximum_population_fraction_violation", 0.05)
    )
    if not math.isfinite(population_violation) or not 0.0 <= population_violation <= 1.0:
        raise ValueError(
            "maximum_population_fraction_violation must lie in [0, 1]"
        )
    carrier_residual_limit = float(
        periodic.get("maximum_unmodeled_free_carrier_residual_fraction", 1e-4)
    )
    if (
        not math.isfinite(carrier_residual_limit)
        or not 0.0 <= carrier_residual_limit < 1.0
    ):
        raise ValueError(
            "maximum_unmodeled_free_carrier_residual_fraction must lie in [0, 1)"
        )
    carrier_convergence = periodic.get("free_carrier_convergence", {})
    if not isinstance(carrier_convergence, dict):
        raise ValueError(
            "simulation.periodic_steady_state.free_carrier_convergence must be an object"
        )
    if not isinstance(carrier_convergence.get("enabled", True), bool):
        raise ValueError("free_carrier_convergence.enabled must be a boolean")
    carrier_convergence_metric = str(
        carrier_convergence.get("metric", "optical_effect")
    )
    if carrier_convergence_metric not in {"optical_effect", "field"}:
        raise ValueError(
            "simulation.periodic_steady_state.free_carrier_convergence.metric "
            "must be optical_effect or field"
        )
    reduced_map = periodic.get("reduced_pulse_map", {})
    reduced_model = str(reduced_map.get("model", "phase_insensitive"))
    if reduced_model not in {"phase_insensitive", "axisymmetric_coherent_kerr"}:
        raise ValueError(
            "simulation.periodic_steady_state.reduced_pulse_map.model must be "
            "phase_insensitive or axisymmetric_coherent_kerr"
        )
    require_exact_integer(
        reduced_map.get("temporal_bins", 1),
        name="simulation.periodic_steady_state.reduced_pulse_map.temporal_bins",
        minimum=1,
        maximum=256,
    )
    require_exact_integer(
        reduced_map.get("spectral_bins", 33),
        name="simulation.periodic_steady_state.reduced_pulse_map.spectral_bins",
        minimum=3,
        maximum=4096,
    )
    require_exact_integer(
        reduced_map.get("coherent_sample_count", 1024),
        name=(
            "simulation.periodic_steady_state.reduced_pulse_map."
            "coherent_sample_count"
        ),
        minimum=64,
        maximum=32768,
    )
    require_exact_integer(
        reduced_map.get("axisymmetric_radial_bins", 24),
        name=(
            "simulation.periodic_steady_state.reduced_pulse_map."
            "axisymmetric_radial_bins"
        ),
        minimum=8,
        maximum=4096,
    )
    require_exact_integer(
        reduced_map.get("axisymmetric_gain_spectral_bins", 33),
        name=(
            "simulation.periodic_steady_state.reduced_pulse_map."
            "axisymmetric_gain_spectral_bins"
        ),
        minimum=3,
        maximum=4096,
    )
    reduced_tolerance = float(reduced_map.get("relative_tolerance", 1e-3))
    if not math.isfinite(reduced_tolerance) or reduced_tolerance <= 0.0:
        raise ValueError(
            "simulation.periodic_steady_state.reduced_pulse_map.relative_tolerance "
            "must be finite and positive"
        )
    require_exact_integer(
        reduced_map.get("consecutive_cycles", 3),
        name=(
            "simulation.periodic_steady_state.reduced_pulse_map."
            "consecutive_cycles"
        ),
        minimum=1,
        maximum=1000,
    )
    _validate_fixed_point_acceleration(
        periodic.get("acceleration", {}),
        "simulation.periodic_steady_state.acceleration",
    )
    if bool(periodic.get("enabled", False)) and float(config["simulation"]["repetition_rate_Hz"]) <= 0.0:
        raise ValueError("periodic steady-state requires simulation.repetition_rate_Hz > 0")
    if (
        bool(periodic.get("enabled", False))
        and periodic.get("method", "pulse_map") == "cycle_average"
        and config["energy_levels"].get("gain_override_m_inv") is not None
    ):
        raise ValueError(
            "cycle-average steady state requires physical upper/lower populations; "
            "remove energy_levels.gain_override_m_inv"
        )
    if (
        bool(periodic.get("enabled", False))
        and periodic_method in {"pulse_map", "reduced_pulse_map"}
        and config["energy_levels"].get("gain_override_m_inv") is not None
    ):
        raise ValueError(
            "discrete pulse-map periodic steady state requires a physical population reservoir; "
            "energy_levels.gain_override_m_inv is re-created for each propagated pulse "
            "and therefore cannot be part of a periodic fixed point"
        )
    require_exact_integer(
        config["simulation"]["passes"],
        name="simulation.passes",
        minimum=1,
    )
    if config["simulation"].get("validation_mode", "exploratory") not in {
        "exploratory",
        "strict",
    }:
        raise ValueError(
            "simulation.validation_mode must be exploratory or strict"
        )
    strict_validation = config["simulation"].get("strict_validation", {})
    if not isinstance(
        strict_validation.get("require_named_external_losses", True), bool
    ):
        raise ValueError(
            "simulation.strict_validation.require_named_external_losses must be boolean"
        )
    if not isinstance(
        strict_validation.get(
            "require_experimentally_calibrated_material", False
        ),
        bool,
    ):
        raise ValueError(
            "simulation.strict_validation.require_experimentally_calibrated_material "
            "must be boolean"
        )
    for key in (
        "pump_power_balance_relative_tolerance",
        "minimum_hybrid_pump_gain_entrance_capture_fraction",
        "minimum_hybrid_seed_gain_entrance_capture_fraction",
        "seed_energy_balance_relative_tolerance",
        "maximum_pump_absorption_optical_depth",
        "population_conservation_relative_tolerance",
        "ensemble_inventory_relative_tolerance",
        "population_correction_relative_tolerance",
        "grid_projection_relative_tolerance",
        "thermal_energy_balance_relative_tolerance",
        "temporal_boundary_energy_fraction",
        "output_spectral_boundary_energy_fraction",
        "ase_energy_balance_relative_tolerance",
        "level_storage_relative_tolerance",
        "ase_eigenpair_relative_tolerance",
        "cpa_resynthesis_energy_relative_tolerance",
        "strong_field_fractional_loss_per_step",
        "fem_equilibrium_relative_tolerance",
        "cycle_average_population_jump_relative_tolerance",
        "cycle_average_probe_difference_relative_tolerance",
        "reduced_pulse_map_population_difference_relative_tolerance",
        "reduced_pulse_map_output_difference_relative_tolerance",
        "free_carrier_periodic_residual_relative_tolerance",
        "cycle_time_closure_relative_tolerance",
        "static_material_power_out_of_range_tolerance",
        "undepleted_pump_maximum_absorbed_fraction",
        "event_energy_identity_relative_tolerance",
        "event_energy_unresolved_fraction_tolerance",
        "event_ase_interface_relative_tolerance",
        "pump_component_negative_active_absorption_tolerance_m_inv",
        "pump_component_level_gap_relative_tolerance",
        "signal_event_requested_applied_relative_tolerance",
        "signal_event_maximum_population_change_fraction",
        "nonlinear_fractional_depletion_step_tolerance",
        "maximum_commercial_app_spectral_support_leakage_fraction",
        "maximum_commercial_app_spectral_sync_error_fraction",
        "maximum_commercial_app_reduced_spatiotemporal_B_integral_rad",
        "maximum_commercial_app_axis_coupling_ratio",
        "maximum_axisymmetric_projection_loss_fraction",
        "maximum_axisymmetric_boundary_energy_fraction",
        "maximum_axisymmetric_azimuthal_residual",
        "maximum_axisymmetric_population_realizability_correction_fraction",
        "maximum_axisymmetric_population_positivity_redistribution_fraction",
        "maximum_axisymmetric_diffraction_norm_correction",
        "maximum_axisymmetric_gain_bin_relative_frequency_span",
        "maximum_axisymmetric_gain_reconstruction_normalization_correction",
        "axisymmetric_population_mapping_relative_tolerance",
    ):
        value = float(strict_validation.get(key, 0.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"simulation.strict_validation.{key} must be finite and non-negative"
            )
    hybrid_pump_capture = float(
        strict_validation.get(
            "minimum_hybrid_pump_gain_entrance_capture_fraction", 0.98
        )
    )
    if hybrid_pump_capture > 1.0:
        raise ValueError(
            "simulation.strict_validation.minimum_hybrid_pump_gain_entrance_capture_fraction "
            "must not exceed one"
        )
    hybrid_seed_capture = float(
        strict_validation.get(
            "minimum_hybrid_seed_gain_entrance_capture_fraction", 0.98
        )
    )
    if hybrid_seed_capture > 1.0:
        raise ValueError(
            "simulation.strict_validation.minimum_hybrid_seed_gain_entrance_capture_fraction "
            "must not exceed one"
        )
    for key in (
        "temporal_boundary_energy_fraction",
        "output_spectral_boundary_energy_fraction",
        "strong_field_fractional_loss_per_step",
        "static_material_power_out_of_range_tolerance",
        "undepleted_pump_maximum_absorbed_fraction",
        "event_energy_unresolved_fraction_tolerance",
        "signal_event_maximum_population_change_fraction",
        "nonlinear_fractional_depletion_step_tolerance",
        "maximum_commercial_app_spectral_support_leakage_fraction",
        "maximum_commercial_app_spectral_sync_error_fraction",
        "maximum_axisymmetric_projection_loss_fraction",
        "maximum_axisymmetric_boundary_energy_fraction",
        "maximum_axisymmetric_population_realizability_correction_fraction",
        "maximum_axisymmetric_population_positivity_redistribution_fraction",
    ):
        if float(strict_validation.get(key, 0.0)) > 1.0:
            raise ValueError(
                f"simulation.strict_validation.{key} must not exceed one"
            )
    transmission = float(config["simulation"]["between_pass_transmission"])
    if not math.isfinite(transmission) or not 0 <= transmission <= 1:
        raise ValueError("simulation.between_pass_transmission must be in [0, 1]")
    unmodeled_delay = float(
        config["simulation"].get("unmodeled_between_pass_delay_s", 0.0)
    )
    if not math.isfinite(unmodeled_delay) or unmodeled_delay < 0.0:
        raise ValueError(
            "simulation.unmodeled_between_pass_delay_s must be finite and non-negative"
        )
    multipass = config["simulation"].get("multipass", {})
    if not isinstance(multipass, dict):
        raise ValueError("simulation.multipass must be an object")
    if not isinstance(multipass.get("enabled", False), bool):
        raise ValueError("simulation.multipass.enabled must be boolean")
    if bool(multipass.get("enabled", False)):
        from .multipass import uses_causal_clock

        resolved_signal_model = resolve_signal_model(config)
        causal_clock = uses_causal_clock(
            config["simulation"], config["seed"], resolved_signal_model
        )
        if (
            str(multipass.get("execution_mode", "auto")) == "causal"
            and resolved_signal_model != "spectral_time"
        ):
            raise ValueError(
                "simulation.multipass.execution_mode=causal requires signal_model=spectral_time"
            )
        if config["simulation"].get("pass_mode", "simple") != "simple":
            raise ValueError(
                "simulation.multipass is the only multi-pass organization; simulation.pass_mode must remain the internal value simple"
            )
        if config.get("optical_train", {}).get("between_pass", []):
            raise ValueError(
                "simulation.multipass templates own every inter-hit route; move old optical_train.between_pass elements into the selected multipass template"
            )
        if not math.isclose(
            float(config["simulation"].get("between_pass_transmission", 1.0)),
            1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                "simulation.multipass uses simulation.multipass.per_transition_power_transmission; legacy simulation.between_pass_transmission must be 1"
            )
        if bool(config["simulation"].get("flip_x_between_passes", False)):
            raise ValueError(
                "simulation.multipass does not use the old global flip_x_between_passes switch; describe the physical imaging "
                "in the selected template or a prescribed mirror transition"
            )
        if causal_clock and str(config["seed"].get("direction", "forward")) != "forward":
            raise ValueError(
                "causal multipass templates currently require seed.direction=forward"
            )
        temporal = config["seed"].get("temporal", {})
        if causal_clock and str(temporal.get("type", "gaussian")) == "cw":
            raise ValueError(
                "causal multipass pulse transport requires a finite Seed "
                "pulse; seed.temporal.type=cw must use the sequential "
                "multi-pass solver"
            )
        if causal_clock and not bool(temporal.get("resolved", False)):
            raise ValueError(
                "causal multipass pulse transport requires seed.temporal.resolved=true"
            )
        if causal_clock and (
            str(temporal.get("type", "gaussian")) != "file"
            and int(temporal.get("sample_count", 21)) < 3
        ):
            raise ValueError(
                "causal multipass pulse transport requires at least three input temporal samples"
            )
        if causal_clock and str(
            config["seed"].get("joint_time_frequency", {}).get(
                "model", "separable"
            )
        ) != "separable":
            raise ValueError(
                "causal multipass pulse transport currently requires a separable time-frequency input"
            )
        if causal_clock and str(
            config["seed"].get("spatial_modes", {}).get(
                "model", "effective_m2"
            )
        ) == "incoherent_hg":
            raise ValueError(
                "causal multipass pulse transport does not yet support the incoherent_hg modal expansion"
            )
        for key in (
            "maximum_dark_step_s",
            "maximum_relative_group_index_span",
            "maximum_relative_path_cosine_span",
            "per_transition_extra_delay_s",
        ):
            value = float(multipass.get(key, 0.0))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"simulation.multipass.{key} must be finite and non-negative"
                )
        for key in (
            "external_refractive_index",
            "external_group_refractive_index",
            "maximum_field_memory_MB",
        ):
            value = float(multipass.get(key, 1.0))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"simulation.multipass.{key} must be positive and finite"
                )
        transition_transmission = float(
            multipass.get("per_transition_power_transmission", 1.0)
        )
        if (
            not math.isfinite(transition_transmission)
            or not 0.0 <= transition_transmission <= 1.0
        ):
            raise ValueError(
                "simulation.multipass.per_transition_power_transmission must lie in [0, 1]"
            )
        require_exact_integer(
            multipass.get("maximum_time_bins", 20000),
            name="simulation.multipass.maximum_time_bins",
            minimum=3,
            maximum=10_000_000,
        )
        if not isinstance(multipass.get("include_dark_relaxation", True), bool):
            raise ValueError(
                "simulation.multipass.include_dark_relaxation must be boolean"
            )
        from .multipass import compile_multipass_plan

        configured_seed_route = list(
            config.get("optical_train", {}).get("seed", [])
        )
        base_marker = next(
            (
                element
                for element in configured_seed_route
                if isinstance(element, dict)
                and str(element.get("type", "")) == "gain_medium"
            ),
            {"type": "gain_medium", "name": "multipass_gain"},
        )
        plan = compile_multipass_plan(
            multipass,
            base_marker,
            crystal_length_m=float(config["grid"]["length_z_m"]),
            crystal_refractive_index=material_refractive_index(
                config["material"],
                float(config["seed"]["wavelength_m"]),
                role="signal",
            ),
            global_extra_delay_s=float(
                config["simulation"].get(
                    "unmodeled_between_pass_delay_s", 0.0
                )
            ),
        )
        if int(config["simulation"].get("passes", 1)) != len(plan.hits):
            raise ValueError(
                "simulation.passes is an internal derived assertion for multipass: "
                "it must equal the compiled template hit count"
            )
        if causal_clock and plan.hits[0].direction != "forward":
            raise ValueError(
                "the first causal multipass hit must be forward for seed.direction=forward"
            )
        path_cosines: list[float] = []
        for index, hit in enumerate(plan.hits):
            _validate_incidence_angles(
                hit.marker,
                name=f"compiled multipass hit {index + 1}",
            )
            path_cosines.append(gain_medium_path_cosine(hit.marker))
        mean_path_cosine = float(np.mean(path_cosines))
        relative_path_span = float(
            np.ptp(path_cosines)
            / max(abs(mean_path_cosine), np.finfo(float).tiny)
        )
        maximum_path_span = float(
            multipass.get("maximum_relative_path_cosine_span", 1e-4)
        )
        if causal_clock and relative_path_span > maximum_path_span:
            raise ValueError(
                "compiled multipass hit angles do not share one sufficiently "
                "accurate crystal CFL clock: relative path-cosine span is "
                f"{relative_path_span:.6g}, above "
                "simulation.multipass.maximum_relative_path_cosine_span="
                f"{maximum_path_span:.6g}"
            )
        multipass_periodic = config["simulation"].get(
            "periodic_steady_state", {}
        )
        if causal_clock and (
            bool(multipass_periodic.get("enabled", False))
            and str(multipass_periodic.get("method", "pulse_map"))
            in {"pulse_map", "reduced_pulse_map"}
            and not bool(multipass.get("include_dark_relaxation", True))
        ):
            raise ValueError(
                "causal multipass pulse-map periodic steady state requires "
                "simulation.multipass.include_dark_relaxation=true so the "
                "absolute-time population clock closes"
            )
    model = config["energy_levels"]["model"]
    if model not in {"two_level", "three_level", "four_level", "custom"}:
        raise ValueError("energy_levels.model must be two_level, three_level, four_level, or custom")
    _validate_energy_level_network(config["energy_levels"])
    response_cfg = config["material"].get("spectroscopic_response", {})
    if str(response_cfg.get("mode", "off")) != "off":
        absolute_roles = {
            key.split("_", 1)[0]
            for key, law in response_cfg.get("cross_sections", {}).items()
            if law.get("model") == "absolute_temperature_wavelength_table"
        }
        if absolute_roles and (
            (
                config["energy_levels"].get("inhomogeneous_ensembles", {}).get("enabled", False)
                and any(
                    group.get(f"{role}_center_wavelength_m") is not None
                    for role in absolute_roles
                    for group in config["energy_levels"]["inhomogeneous_ensembles"].get("groups", [])
                )
            )
            or any(
                transition.get("field") in absolute_roles
                and transition.get("line_shape", "none") != "none"
                for transition in config["energy_levels"].get("optical_transitions", [])
            )
        ):
            raise ValueError(
                "A global absolute cross-section table cannot be combined with "
                "inhomogeneous ensembles or a channel line shape: the table "
                "does not define channel-resolved spectra. Use relative response "
                "laws with explicit channel spectra instead."
            )
    if str(config.get("ase", {}).get("model", "off")) != "off":
        ase_upper_reservoirs = _signal_upper_reservoir_names(
            config["energy_levels"]
        )
        if len(ase_upper_reservoirs) != 1:
            raise ValueError(
                "enabled ASE transport requires exactly one independently "
                "populated Signal upper reservoir because its spontaneous "
                "source uses one radiative lifetime and one normalized "
                "emission spectrum; configured reservoirs are "
                f"{ase_upper_reservoirs}"
            )
    configured_level_energies = np.asarray(
        config["energy_levels"].get("level_energies_J", []), dtype=float
    )
    if configured_level_energies.size:
        expected_level_count = (
            {"two_level": 2, "three_level": 3, "four_level": 4}.get(model)
            or len(config["energy_levels"].get("levels", []))
        )
        if configured_level_energies.shape != (expected_level_count,) or not np.all(
            np.isfinite(configured_level_energies)
        ):
            raise ValueError(
                "energy_levels.level_energies_J must contain one finite energy per level"
            )
    nonlinear_solver = config["energy_levels"].get("nonlinear_solver", {})
    if nonlinear_solver.get("method", "adaptive_substeps") not in {
        "single_step",
        "adaptive_substeps",
    }:
        raise ValueError(
            "energy_levels.nonlinear_solver.method must be single_step or adaptive_substeps"
        )
    nonlinear_target = float(
        nonlinear_solver.get("maximum_fractional_depletion_per_substep", 0.05)
    )
    if not math.isfinite(nonlinear_target) or not 0.0 < nonlinear_target <= 1.0:
        raise ValueError(
            "energy_levels.nonlinear_solver.maximum_fractional_depletion_per_substep "
            "must lie in (0, 1]"
        )
    require_exact_integer(
        nonlinear_solver.get("maximum_substeps", 256),
        name="energy_levels.nonlinear_solver.maximum_substeps",
        minimum=1,
    )
    linear_solver = config["energy_levels"].get("linear_solver", {})
    if linear_solver.get("small_matrix_kernel", "auto") not in {
        "auto",
        "numpy",
        "numba",
    }:
        raise ValueError(
            "energy_levels.linear_solver.small_matrix_kernel must be auto, "
            "numpy, or numba"
        )
    require_exact_integer(
        linear_solver.get("numba_minimum_systems", 4096),
        name="energy_levels.linear_solver.numba_minimum_systems",
        minimum=1,
    )
    require_exact_integer(
        linear_solver.get("maximum_numba_order", 8),
        name="energy_levels.linear_solver.maximum_numba_order",
        minimum=2,
        maximum=32,
    )
    require_exact_integer(
        config["energy_levels"].get("multi_transition_solver", {}).get(
            "photon_packet_count", 32
        ),
        name="energy_levels.multi_transition_solver.photon_packet_count",
        minimum=2,
        maximum=4096,
    )
    packet_change_target = float(
        config["energy_levels"].get("multi_transition_solver", {}).get(
            "maximum_population_change_fraction_per_packet", 0.05
        )
    )
    if (
        not math.isfinite(packet_change_target)
        or not 0.0 < packet_change_target <= 1.0
    ):
        raise ValueError(
            "energy_levels.multi_transition_solver."
            "maximum_population_change_fraction_per_packet must lie in (0, 1]"
        )
    pump_quantum_efficiency = float(config["energy_levels"].get("pump_quantum_efficiency", 1.0))
    if (
        not math.isfinite(pump_quantum_efficiency)
        or not 0.0 <= pump_quantum_efficiency <= 1.0
    ):
        raise ValueError("energy_levels.pump_quantum_efficiency must lie in [0, 1]")
    radiative_lifetime = config["energy_levels"].get(
        "upper_state_radiative_lifetime_s"
    )
    if radiative_lifetime is not None and (
        not math.isfinite(float(radiative_lifetime))
        or float(radiative_lifetime) <= 0.0
    ):
        raise ValueError(
            "energy_levels.upper_state_radiative_lifetime_s must be positive"
        )
    heat_source_model = str(
        config["material"].get("thermal", {}).get(
            "heat_source_model", "pump_component_ledger"
        )
    )
    if heat_source_model not in {
        "pump_component_ledger",
        "event_energy_ledger",
    }:
        raise ValueError(
            "material.thermal.heat_source_model must be "
            "pump_component_ledger or event_energy_ledger"
        )
    if (
        heat_source_model == "event_energy_ledger"
        and not bool(
            config["energy_levels"]
            .get("event_energy_ledger", {})
            .get("enabled", False)
        )
    ):
        raise ValueError(
            "material.thermal.heat_source_model=event_energy_ledger requires "
            "energy_levels.event_energy_ledger.enabled=true"
        )
    event_lifetime_branching = bool(
        config["energy_levels"].get("event_energy_ledger", {}).get("enabled", False)
    ) and any(
        partition.get("model", "fixed") == "lifetime_branching"
        for partition in config["energy_levels"].get("event_energy_ledger", {}).get("decay_partitions", [])
    )
    if event_lifetime_branching or (
        config["material"]["thermal"].get("heat_fraction")
        == "lifetime_quantum_defect"
        and heat_source_model == "pump_component_ledger"
    ):
        if radiative_lifetime is None:
            raise ValueError(
                "lifetime_quantum_defect requires "
                "energy_levels.upper_state_radiative_lifetime_s"
            )
        if (
            float(config["energy_levels"]["upper_state_lifetime_s"])
            > float(radiative_lifetime)
            * (1.0 + 1e-12)
        ):
            raise ValueError(
                "energy_levels.upper_state_lifetime_s cannot exceed "
                "energy_levels.upper_state_radiative_lifetime_s"
            )
        response_for_lifetime = SpectroscopicResponse.from_config(
            config["material"],
            config["energy_levels"],
        )
        response_for_lifetime.set_environment(
            float(
                config["material"].get(
                    "operating_temperature_K",
                    response_for_lifetime.reference_temperature_K,
                )
            ),
            float(config["material"]["ion_density_m3"]),
        )
        operating_total_lifetime = float(
            config["energy_levels"]["upper_state_lifetime_s"]
        ) * float(response_for_lifetime.lifetime_scale(()))
        if (
            operating_total_lifetime
            > float(radiative_lifetime) * (1.0 + 1e-12)
        ):
            raise ValueError(
                "the temperature/concentration-adjusted upper-state total "
                f"lifetime ({operating_total_lifetime:.8g} s) exceeds "
                "energy_levels.upper_state_radiative_lifetime_s at the "
                "configured operating point"
            )
    if (
        config["material"]["thermal"].get("heat_fraction")
        == "lifetime_quantum_defect"
        and heat_source_model == "pump_component_ledger"
    ):
        if str(config["energy_levels"].get("model", "four_level")) == "custom":
            signal_transitions = [
                transition
                for transition in config["energy_levels"].get(
                    "optical_transitions", []
                )
                if str(
                    transition.get(
                        "field",
                        transition.get("role", ""),
                    )
                )
                == "signal"
            ]
            if len(signal_transitions) != 1:
                raise ValueError(
                    "lifetime_quantum_defect uses one explicit transition "
                    "energy and therefore requires exactly one custom Signal "
                    "optical transition"
                )
        if float(
            config["material"]["thermal"].get("extra_heat_fraction", 0.0)
        ) != 0.0:
            raise ValueError(
                "lifetime_quantum_defect derives non-radiative heat from "
                "lifetimes; material.thermal.extra_heat_fraction must be zero "
                "to avoid double counting"
            )
    if (
        heat_source_model == "event_energy_ledger"
        and model in {"two_level", "three_level", "four_level"}
        and not config["energy_levels"].get("level_energies_J")
        and float(config["pump"]["wavelength_m"])
        > float(config["seed"]["wavelength_m"])
    ):
        raise ValueError(
            "the built-in event-energy level assignment requires Pump photon "
            "energy >= Seed transition energy (pump wavelength <= seed "
            "wavelength); otherwise provide a physically ordered custom "
            "level network with explicit level_energies_J"
        )
    if config["simulation"]["signal_propagation"] not in {"gain_only", "split_step"}:
        raise ValueError("simulation.signal_propagation must be gain_only or split_step")
    if config["simulation"].get("signal_model", "auto") not in {
        "auto", "commercial_app_fast", "fluence", "spectral_time", "josa_transport", "coherent_cpa"
    }:
        raise ValueError(
            "simulation.signal_model must be auto, commercial_app_fast, fluence, spectral_time, josa_transport, or coherent_cpa"
        )
    tail_binning = config["seed"].get("tail_binning", {})
    coverage = tail_binning.get("energy_coverage", .999999)
    if isinstance(coverage, bool) or not np.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError("seed.tail_binning.energy_coverage must be in (0, 1]")
    for key in ("temporal_enabled", "spectral_enabled"):
        if not isinstance(tail_binning.get(key, False), bool):
            raise ValueError("seed.tail_binning." + key + " must be boolean")
    for key in ("temporal_tail_points", "spectral_tail_points"):
        require_exact_integer(tail_binning.get(key, 3), name="seed.tail_binning." + key, minimum=2, maximum=64)
    if resolve_signal_model(config) == "coherent_cpa" and tail_binning.get("spectral_enabled", False):
        if config["seed"]["coherent"].get("fidelity_mode") != "axisymmetric_kerr":
            raise ValueError("Coherent frequency tail binning requires axisymmetric_kerr")
        if config["seed"]["coherent"].get("transient_gain", {}).get("enabled", False):
            raise ValueError("Coherent frequency tail binning requires whole-fluence gain")
    spectral_time = config["seed"].get("spectral_time", {})
    if spectral_time.get("spatial_backend", "cartesian") not in {"cartesian", "axisymmetric"}:
        raise ValueError("seed.spectral_time.spatial_backend must be cartesian or axisymmetric")
    for key in ("temporal_bins", "spectral_bins"):
        if spectral_time.get(key) is not None:
            require_exact_integer(spectral_time[key], name="seed.spectral_time." + key, minimum=1, maximum=65536)
    if resolve_signal_model(config) == "spectral_time":
        if (any(spectral_time.get(key) is not None for key in ("temporal_bins", "spectral_bins")) or any(tail_binning.get(key, False) for key in ("temporal_enabled", "spectral_enabled"))) and config["seed"].get("joint_time_frequency", {}).get("model", "separable") == "matrix":
            raise ValueError("Spectral Time bin reduction currently requires separable time/frequency weights")
        if spectral_time.get("spatial_backend") == "axisymmetric":
            if config["seed"].get("spatial_modes", {}).get("model") == "incoherent_hg":
                raise ValueError("Spectral Time axisymmetric currently requires effective_m2 spatial modes")
            if config["seed"].get("spatial_gain_transport", {}).get("model", "local_field") != "local_field":
                raise ValueError("Spectral Time axisymmetric requires local_field gain")
            if int(config["simulation"].get("passes", 1)) != 1:
                raise ValueError("Spectral Time axisymmetric currently supports one gain pass")
    if (
        str(config["simulation"].get("signal_model", "auto")) == "fluence"
        and str(
            config["seed"].get("spatial_modes", {}).get(
                "model", "effective_m2"
            )
        )
        == "incoherent_hg"
    ):
        raise ValueError(
            "incoherent_hg uses independently propagated wavelength/mode "
            "channels and requires simulation.signal_model=spectral_time (or "
            "auto); the whole-fluence backend has no mode-resolved contract"
        )
    independent_signal_reservoirs = _signal_reservoir_pairs(
        config["energy_levels"]
    )
    if (
        bool(resolve_physics(config).get("gain_saturation", False))
        and len(independent_signal_reservoirs) > 1
        and resolve_signal_model(config) not in {"fluence", "josa_transport"}
    ):
        raise ValueError(
            "Strongly saturated independent Signal reservoirs currently "
            "require simulation.signal_model=fluence or josa_transport, whose "
            "photon-packet/time-cell event integrators update every reservoir. "
            "The merged-Fsat spectral/coherent paths are accepted only when "
            "all Signal transitions share one lower/upper reservoir."
        )
    if config["simulation"].get("pass_mode", "simple") != "simple":
        raise ValueError(
            "simulation.pass_mode is retired; load old optical_bench JSON through prepare_config so it can be migrated to simulation.multipass"
        )
    if config["simulation"].get("signal_volume_recording", "last_and_cumulative") not in {
        "last_only",
        "last_and_cumulative",
        "per_pass",
    }:
        raise ValueError(
            "simulation.signal_volume_recording must be last_only, last_and_cumulative, or per_pass"
        )
    validate_result_recording(config["simulation"].get("result_recording", {}))
    temporal_choices = {"cw", "gaussian", "sech2", "rectangular", "super_gaussian", "file"}
    spectrum_choices = {"monochromatic", "gaussian", "lorentzian", "super_gaussian", "file"}
    if config["pump"]["temporal"].get("type", "cw") not in temporal_choices - {"sech2"}:
        raise ValueError("pump.temporal.type is unsupported")
    if config["seed"]["temporal"].get("type", "gaussian") not in temporal_choices:
        raise ValueError("seed.temporal.type is unsupported")
    if config["seed"]["temporal"].get("type") == "cw":
        periodic = config["simulation"].get("periodic_steady_state", {})
        if not (
            bool(periodic.get("enabled", False))
            and periodic.get("method", "pulse_map") == "cycle_average"
        ):
            raise ValueError(
                "seed.temporal.type=cw requires periodic_steady_state.enabled=true "
                "and method=cycle_average so the Seed is coupled through average photon flux; "
                "pulse_map methods are reserved for discrete Seed pulses"
            )
    for prefix in ("pump", "seed"):
        if config[prefix]["spectrum"].get("type", "monochromatic") not in spectrum_choices:
            raise ValueError(f"{prefix}.spectrum.type is unsupported")
        spectrum_cfg = config[prefix].get("spectrum", {})
        if str(spectrum_cfg.get("spectral_density_domain", "wavelength")) not in {
            "wavelength",
            "frequency",
        }:
            raise ValueError(
                f"{prefix}.spectrum.spectral_density_domain must be wavelength or frequency"
            )
        for section_name, section in (
            ("spectrum", spectrum_cfg),
            ("temporal", config[prefix].get("temporal", {})),
        ):
            if str(section.get("invalid_sample_policy", "error")) not in {
                "error",
                "clip",
            }:
                raise ValueError(
                    f"{prefix}.{section_name}.invalid_sample_policy must be error or clip"
                )
        bandwidth = float(
            spectrum_cfg.get(
                "bandwidth_fwhm_m",
                config[prefix].get("spectral_bandwidth_m", 0.0),
            )
        )
        if not math.isfinite(bandwidth) or bandwidth < 0.0:
            raise ValueError(
                f"{prefix}.spectrum.bandwidth_fwhm_m must be finite and non-negative"
            )
        bandwidth_Hz_raw = spectrum_cfg.get("bandwidth_fwhm_Hz")
        bandwidth_Hz = (
            0.0 if bandwidth_Hz_raw is None else float(bandwidth_Hz_raw)
        )
        if not math.isfinite(bandwidth_Hz) or bandwidth_Hz < 0.0:
            raise ValueError(
                f"{prefix}.spectrum.bandwidth_fwhm_Hz must be null or finite and non-negative"
            )
        if bandwidth_Hz > 0.0:
            if str(spectrum_cfg.get("spectral_density_domain", "wavelength")) != "frequency":
                raise ValueError(
                    f"{prefix}.spectrum.bandwidth_fwhm_Hz requires spectral_density_domain=frequency"
                )
            if bandwidth > 0.0:
                raise ValueError(
                    f"{prefix}.spectrum must specify only one of bandwidth_fwhm_m and bandwidth_fwhm_Hz"
                )
        require_exact_integer(
            spectrum_cfg.get("sample_count", 1),
            name=f"{prefix}.spectrum.sample_count",
            minimum=1,
        )
        if prefix == "pump" and spectrum_cfg.get(
            "spatial_transport", "shared"
        ) not in {"shared", "per_wavelength"}:
            raise ValueError(
                "pump.spectrum.spatial_transport must be shared or per_wavelength"
            )
        if (
            not math.isfinite(float(spectrum_cfg.get("span_fwhm", 1.0)))
            or float(spectrum_cfg.get("span_fwhm", 1.0)) <= 0.0
        ):
            raise ValueError(f"{prefix}.spectrum.span_fwhm must be positive")
        if (
            not math.isfinite(float(spectrum_cfg.get("order", 1.0)))
            or float(spectrum_cfg.get("order", 1.0)) <= 0.0
        ):
            raise ValueError(f"{prefix}.spectrum.order must be positive")
        temporal_cfg = config[prefix].get("temporal", {})
        require_exact_integer(
            temporal_cfg.get("sample_count", 1),
            name=f"{prefix}.temporal.sample_count",
            minimum=1,
        )
        if (
            not math.isfinite(float(temporal_cfg.get("order", 1.0)))
            or float(temporal_cfg.get("order", 1.0)) <= 0.0
        ):
            raise ValueError(f"{prefix}.temporal.order must be positive")
        if str(spectrum_cfg.get("wavelength_unit", "m")) not in WAVELENGTH_UNITS_TO_M:
            raise ValueError(
                f"{prefix}.spectrum.wavelength_unit must be one of "
                f"{sorted(WAVELENGTH_UNITS_TO_M)}"
            )
        if str(temporal_cfg.get("time_unit", "s")) not in TIME_UNITS_TO_S:
            raise ValueError(
                f"{prefix}.temporal.time_unit must be one of {sorted(TIME_UNITS_TO_S)}"
            )
    pump_pulse_width = float(
        config["pump"].get("temporal", {}).get("pulse_fwhm_s", 1.0)
    )
    if not math.isfinite(pump_pulse_width) or pump_pulse_width <= 0.0:
        raise ValueError("pump.temporal.pulse_fwhm_s must be positive and finite")
    if config["pump"]["temporal"].get("power_mode", "average") not in {"average", "peak"}:
        raise ValueError("pump.temporal.power_mode must be average or peak")
    pump_temporal = config["pump"].get("temporal", {})
    pump_temporal_type = str(pump_temporal.get("type", "cw"))
    try:
        pump_temporal_delay = float(pump_temporal.get("delay_s", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("pump.temporal.delay_s must be finite") from exc
    if not math.isfinite(pump_temporal_delay):
        raise ValueError("pump.temporal.delay_s must be finite")
    if str(pump_temporal.get("value_mode", "relative")) not in {
        "relative",
        "absolute_W",
    }:
        raise ValueError("pump.temporal.value_mode must be relative or absolute_W")
    if str(
        pump_temporal.get("file_period_contract", "zero_outside")
    ) not in {"zero_outside", "cover_period"}:
        raise ValueError(
            "pump.temporal.file_period_contract must be zero_outside or cover_period"
        )
    repetition_rate = float(config["simulation"].get("repetition_rate_Hz", 0.0))
    periodic_enabled = bool(
        config["simulation"].get("periodic_steady_state", {}).get("enabled", False)
    )
    physics = resolve_physics(config)
    applicability = config.get("physics", {}).get("applicability", {})
    maximum_paraxial_angle = float(
        applicability.get("maximum_paraxial_angle_deg", 10.0)
    )
    if (
        not math.isfinite(maximum_paraxial_angle)
        or not 0.0 < maximum_paraxial_angle < 89.0
    ):
        raise ValueError(
            "physics.applicability.maximum_paraxial_angle_deg must lie in (0, 89)"
        )
    maximum_nyquist_fraction = float(
        applicability.get("maximum_fft_transverse_nyquist_fraction", 0.8)
    )
    if (
        not math.isfinite(maximum_nyquist_fraction)
        or not 0.0 < maximum_nyquist_fraction <= 1.0
    ):
        raise ValueError(
            "physics.applicability.maximum_fft_transverse_nyquist_fraction "
            "must lie in (0, 1]"
        )
    population_dynamics = str(physics["population_dynamics"])
    lifetime_heat_enabled = bool(
        heat_source_model == "pump_component_ledger"
        and configured_heat_fraction == "lifetime_quantum_defect"
    )
    lifetime_heat_has_stationary_population = bool(
        population_dynamics == "steady_state"
        or (
            periodic_enabled
            and str(
                config["simulation"]
                .get("periodic_steady_state", {})
                .get("method", "pulse_map")
            )
            == "cycle_average"
        )
    )
    if lifetime_heat_enabled and not lifetime_heat_has_stationary_population:
        raise ValueError(
            "lifetime_quantum_defect requires a stationary upper-state "
            "population: use physics.population_dynamics=steady_state or "
            "periodic_steady_state.method=cycle_average. An exact pulse_map "
            "returns a cycle-boundary population, not the cycle-average "
            "population required for quantitative lifetime-decay heat; use "
            "the event-energy ledger when exact event-resolved heat is needed."
        )
    pump_duration = float(config["simulation"].get("pump_duration_s", 0.0))
    waveform_is_sampled = bool(
        (repetition_rate > 0.0 and periodic_enabled)
        or (
            population_dynamics == "transient"
            and (repetition_rate > 0.0 or pump_duration > 0.0)
        )
    )
    if pump_temporal_type != "cw" and not waveform_is_sampled:
        if str(pump_temporal.get("power_mode", "average")) == "peak":
            raise ValueError(
                "A peak Pump waveform requires transient population dynamics or an enabled "
                "periodic steady-state calculation; otherwise pump.power_W would be treated as CW."
            )
        if (
            pump_temporal_type == "file"
            and str(pump_temporal.get("value_mode", "relative")) == "absolute_W"
        ):
            raise ValueError(
                "An absolute-W Pump waveform requires transient population dynamics or an enabled "
                "periodic steady-state calculation."
            )
    pulse_definition = config["seed"].get("pulse_definition", {})
    if pulse_definition.get("domain", "independent") not in {"independent", "time", "spectrum", "dual"}:
        raise ValueError("seed.pulse_definition.domain must be independent, time, spectrum, or dual")
    require_exact_integer(
        pulse_definition.get("fft_sample_count", 256),
        name="seed.pulse_definition.fft_sample_count",
        minimum=32,
    )
    require_exact_integer(
        pulse_definition.get("solver_sample_count", 9),
        name="seed.pulse_definition.solver_sample_count",
        minimum=1,
    )
    for key in ("solver_time_sample_count", "solver_spectral_sample_count"):
        if key in pulse_definition:
            require_exact_integer(
                pulse_definition[key],
                name=f"seed.pulse_definition.{key}",
                minimum=1,
            )
    if config["seed"].get("spectrum", {}).get("value_mode", "relative") not in {
        "relative", "energy_per_bin_J", "energy_density_J_per_nm"
    }:
        raise ValueError("seed.spectrum.value_mode is unsupported")
    local_operator = config["seed"].get("coherent", {}).get("local_operator", {})
    local_scheme = str(local_operator.get("scheme", "symmetric_adaptive"))
    if local_scheme not in {"sequential", "symmetric_adaptive"}:
        raise ValueError(
            "seed.coherent.local_operator.scheme must be sequential or symmetric_adaptive"
        )
    local_operator_counts = {
        key: require_exact_integer(
            local_operator.get(key, 1 if key == "minimum_substeps" else 64),
            name=f"seed.coherent.local_operator.{key}",
            minimum=1,
        )
        for key in ("minimum_substeps", "maximum_substeps")
    }
    if (
        local_operator_counts["minimum_substeps"]
        > local_operator_counts["maximum_substeps"]
    ):
        raise ValueError(
            "seed.coherent.local_operator.minimum_substeps cannot exceed maximum_substeps"
        )
    for key, default in (
        ("maximum_gain_exponent_per_substep", 0.1),
        ("maximum_nonlinear_phase_rad_per_substep", 0.05),
        ("maximum_dispersion_phase_rad_per_substep", 0.1),
        ("maximum_thermal_phase_rad_per_substep", 0.1),
        ("maximum_fractional_loss_per_substep", 0.05),
    ):
        value = float(local_operator.get(key, default))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"seed.coherent.local_operator.{key} must be positive and finite"
            )
    if local_operator.get("accuracy_profile", "reference") not in {
        "reference",
        "cpu_fast_convergence_required",
    }:
        raise ValueError(
            "seed.coherent.local_operator.accuracy_profile must be reference "
            "or cpu_fast_convergence_required"
        )
    maximum_step_loss = float(
        local_operator.get("maximum_fractional_loss_per_substep", 0.05)
    )
    if maximum_step_loss >= 1.0:
        raise ValueError(
            "seed.coherent.local_operator.maximum_fractional_loss_per_substep must be below one"
        )
    multirate_gain = local_operator.get("multirate_gain", {})
    if not isinstance(multirate_gain, dict):
        raise ValueError(
            "seed.coherent.local_operator.multirate_gain must be an object"
        )
    if not isinstance(multirate_gain.get("enabled", False), bool):
        raise ValueError(
            "seed.coherent.local_operator.multirate_gain.enabled must be true or false"
        )
    maximum_gain_update = float(
        multirate_gain.get("maximum_gain_exponent_per_update", 0.05)
    )
    if not math.isfinite(maximum_gain_update) or maximum_gain_update <= 0.0:
        raise ValueError(
            "seed.coherent.local_operator.multirate_gain.maximum_gain_exponent_per_update must be positive and finite"
        )
    require_exact_integer(
        multirate_gain.get("maximum_optical_substeps_per_update", 4),
        name=(
            "seed.coherent.local_operator.multirate_gain."
            "maximum_optical_substeps_per_update"
        ),
        minimum=1,
    )
    population_update = local_operator.get("spatial_population_update", "shared_cell")
    if population_update not in {"shared_cell", "independent_groups"}:
        raise ValueError("seed.coherent.local_operator.spatial_population_update must be shared_cell or independent_groups")
    if population_update == "independent_groups":
        coherent_cfg = config["seed"].get("coherent", {})
        if (resolve_signal_model(config) != "coherent_cpa"
                or coherent_cfg.get("fidelity_mode") not in {"hybrid_whole_fluence", "axisymmetric_kerr"}
                or coherent_cfg.get("transient_gain", {}).get("enabled", False)
                or local_scheme != "symmetric_adaptive"):
            raise ValueError("independent_groups requires symmetric coherent whole-fluence gain (hybrid_whole_fluence or axisymmetric_kerr), without transient gain")
    gain_update_cap = multirate_gain.get("maximum_updates_per_cell")
    if gain_update_cap is not None:
        require_exact_integer(gain_update_cap, name="seed.coherent.local_operator.multirate_gain.maximum_updates_per_cell", minimum=1)
        transient_cfg = config["seed"]["coherent"].get("transient_gain", {})
        transient_cap = (resolve_signal_model(config) == "coherent_cpa"
                         and bool(transient_cfg.get("enabled", False))
                         and population_update == "shared_cell")
        if ((population_update != "independent_groups" and not transient_cap)
                or local_scheme != "symmetric_adaptive"
                or not multirate_gain.get("enabled", False)
                or transient_cfg.get("include_relaxation_during_pulse", False)
                or nonlinear_gain_coupling_blockers(physics, config["material"])):
            raise ValueError("maximum_updates_per_cell requires enabled independent_groups whole-fluence or shared_cell transient gain without intrapulse relaxation or stateful nonlinear-loss/carrier coupling")
    validate_spatial_gain_transport(
        config["seed"],
        resolved_signal_model=resolve_signal_model(config),
        gain_saturation=bool(physics["gain_saturation"]),
    )
    validate_polarization_config(
        config,
        resolved_signal_model=resolve_signal_model(config),
        resolved_physics=physics,
    )
    if coupling_model == "thermal_optical_fixed_point":
        if physics["thermal_model"] != "steady_3d":
            raise ValueError(
                "thermal_optical_fixed_point requires thermal_model=steady_3d; a transient heat solve is not a stationary fixed point"
            )
        if physics["refractive_index_model"] != "thermal_3d":
            raise ValueError(
                "thermal_optical_fixed_point requires refractive_index_model=thermal_3d"
            )
        if physics["pump_propagation"] not in {
            "fft",
            "hybrid_abcd_fft",
            "stochastic_raytrace",
        }:
            raise ValueError(
                "thermal_optical_fixed_point requires pump_propagation=fft, "
                "hybrid_abcd_fft, or stochastic_raytrace so the Pump can respond "
                "to the 3D thermal refractive-index field"
            )
        if (
            config["material"]
            .get("mechanical", {})
            .get("endface_deformation", {})
            .get("model", "off")
            == "distributed_opd"
            and physics["pump_propagation"] == "stochastic_raytrace"
        ):
            raise ValueError(
                "endface distributed_opd with thermal_optical_fixed_point requires "
                "Pump fft or hybrid_abcd_fft; the stochastic-ray backend does not yet "
                "advance rays through displaced surface normals"
            )
    if (
        bool(periodic.get("enabled", False))
        and str(periodic.get("method", "pulse_map"))
        in {"pulse_map", "reduced_pulse_map"}
        and str(config["seed"].get("temporal", {}).get("type", "gaussian"))
        == "cw"
    ):
        raise ValueError(
            "periodic pulse-map methods represent discrete Seed pulses; "
            "use cycle_average for seed.temporal.type=cw"
        )
    if (
        bool(periodic.get("enabled", False))
        and str(periodic.get("method", "pulse_map"))
        in {"pulse_map", "reduced_pulse_map"}
        and physics["population_dynamics"] != "transient"
    ):
        raise ValueError(
            "discrete pulse-map periodic steady state requires population_dynamics=transient"
        )
    if (
        bool(periodic.get("enabled", False))
        and str(periodic.get("method", "pulse_map"))
        in {"pulse_map", "reduced_pulse_map"}
        and not bool(physics["gain_saturation"])
    ):
        raise ValueError(
            "discrete pulse-map periodic steady state requires gain_saturation=true so "
            "every amplified Seed pulse depletes the stored population"
        )
    if (
        bool(periodic.get("enabled", False))
        and reduced_initializer
        and resolve_signal_model(config) != "coherent_cpa"
    ):
        raise ValueError(
            "reduced_pulse_map initialization is reserved for coherent_cpa: "
            "it accelerates the initial population before the configured "
            "exact pulse map closes the periodic fixed point"
        )
    if (
        bool(periodic.get("enabled", False))
        and reduced_initializer
        and str(
            periodic.get("reduced_pulse_map", {}).get(
                "model", "phase_insensitive"
            )
        )
        == "axisymmetric_coherent_kerr"
        and str(
            config["seed"].get("coherent", {}).get("fidelity_mode", "configured")
        )
        != "axisymmetric_kerr"
    ):
        raise ValueError(
            "reduced_pulse_map initializer model=axisymmetric_coherent_kerr requires "
            "seed.coherent.fidelity_mode=axisymmetric_kerr"
        )
    if config["grid"].get("spacing_xy", "uniform") in {"center_stretched", "refined_region"}:
        analytic_pump_modes = {
            "prescribed_caustic",
            "super_gaussian_abcd",
            "stochastic_raytrace",
        }
        if physics["optical_propagation"] != "off" and not optical_grid_enabled:
            raise ValueError(
                "center_stretched/refined_region x/y with Seed propagation requires "
                "grid.optical.enabled=true"
            )
        pump_has_uniform_grid = (
            pump_optical_mode == "custom"
            or (pump_optical_mode == "shared" and optical_grid_enabled)
        )
        if (
            not pump_has_uniform_grid
            and physics["pump_propagation"] not in analytic_pump_modes
        ):
            raise ValueError(
                "center_stretched/refined_region x/y with sampled Pump propagation requires "
                "grid.pump_optical.mode=custom or a shared enabled optical grid"
            )
    choices = {
        "optical_propagation": {"off", "abcd", "fft", "hybrid_abcd_fft"},
        "pump_propagation": {
            "prescribed_caustic",
            "abcd",
            "super_gaussian_abcd",
            "fft",
            "hybrid_abcd_fft",
            "stochastic_raytrace",
        },
        "population_dynamics": {"frozen", "steady_state", "transient"},
        "thermal_model": {"off", "configured", "steady_3d", "transient_3d"},
        "refractive_index_model": {"constant", "thermal_3d"},
        "interface_model": {"ideal", "coating", "fresnel", "fresnel_angle"},
        "nonlinear_phase_model": {
            "off",
            "kerr_parabolic_lens",
            "kerr_time_domain",
        },
        "nonlinear_loss_model": {
            "off",
            "strong_field_time_domain",
            "strong_field_effective_area",
        },
    }
    for key, allowed in choices.items():
        if physics[key] not in allowed:
            raise ValueError(f"physics.{key} must be one of {sorted(allowed)}; got {physics[key]!r}")
    if (
        str(config["material"].get("mechanical", {}).get("stress_model", "none"))
        in {"thermoelastic_fem", "anisotropic_thermoelastic_fem"}
        and not (
            physics["thermal_model"] in {"steady_3d", "transient_3d"}
            or (
                physics["thermal_model"] == "configured"
                and bool(config["material"].get("thermal", {}).get("enabled", False))
            )
        )
    ):
        raise ValueError(
            "thermoelastic FEM stress requires a steady_3d or transient_3d thermal model"
        )
    if physics["pump_propagation"] == "super_gaussian_abcd":
        pump_profile = str(config["pump"].get("profile", "super_gaussian"))
        if pump_profile not in {
            "gaussian",
            "super_gaussian",
            "radial_super_gaussian",
        }:
            raise ValueError(
                "pump_propagation=super_gaussian_abcd requires pump.profile="
                "gaussian, super_gaussian, or radial_super_gaussian"
            )
    if physics["nonlinear_loss_model"] == "strong_field_time_domain":
        if resolve_signal_model(config) != "coherent_cpa":
            raise ValueError(
                "strong_field_time_domain requires simulation.signal_model=coherent_cpa"
            )
        if config["material"].get("strong_field", {}).get("model", "off") == "off":
            raise ValueError(
                "strong_field_time_domain requires material.strong_field.model=multiphoton"
            )
    if physics["nonlinear_loss_model"] == "strong_field_effective_area":
        if resolve_signal_model(config) != "commercial_app_fast":
            raise ValueError(
                "strong_field_effective_area requires simulation.signal_model=commercial_app_fast"
            )
        if config["material"].get("strong_field", {}).get("model", "off") == "off":
            raise ValueError(
                "strong_field_effective_area requires material.strong_field.model=multiphoton"
            )
    if (
        physics["nonlinear_phase_model"] == "kerr_time_domain"
        and resolve_signal_model(config) != "coherent_cpa"
    ):
        raise ValueError(
            "kerr_time_domain requires simulation.signal_model=coherent_cpa"
        )
    if (
        physics["nonlinear_phase_model"] == "kerr_parabolic_lens"
        and resolve_signal_model(config) != "commercial_app_fast"
    ):
        raise ValueError(
            "kerr_parabolic_lens requires simulation.signal_model=commercial_app_fast"
        )
    sampled_thermal_lens = config["seed"].get("thermal_lens", {})
    if not isinstance(sampled_thermal_lens, dict):
        raise ValueError("seed.thermal_lens must be an object")
    sampled_thermal_lens_model = str(
        sampled_thermal_lens.get("model", "sampled_phase")
    )
    if sampled_thermal_lens_model not in {
        "sampled_phase",
        "local_parabolic",
    }:
        raise ValueError(
            "seed.thermal_lens.model must be sampled_phase or local_parabolic"
        )
    sampled_quadratic_model = str(
        sampled_thermal_lens.get("quadratic_model", "axis_aligned")
    )
    if sampled_quadratic_model not in {"full_xy", "axis_aligned"}:
        raise ValueError(
            "seed.thermal_lens.quadratic_model must be full_xy or axis_aligned"
        )
    if str(sampled_thermal_lens.get("fit_radius_mode", "manual")) != "manual":
        raise ValueError(
            "seed.thermal_lens.fit_radius_mode must currently be manual"
        )
    if sampled_thermal_lens_model == "local_parabolic":
        if resolve_signal_model(config) == "commercial_app_fast":
            raise ValueError(
                "sampled-field seed.thermal_lens.local_parabolic is not used by "
                "commercial_app_fast; configure seed.commercial_app_fast.thermal_lens instead"
            )
        if physics["optical_propagation"] == "off":
            raise ValueError(
                "seed.thermal_lens.local_parabolic requires optical propagation"
            )
        for axis in "xy":
            value = sampled_thermal_lens.get(f"radius_{axis}_m")
            if (
                value is None
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(
                    f"seed.thermal_lens.radius_{axis}_m must be positive "
                    "for local_parabolic"
                )
    for axis in "xy":
        center = sampled_thermal_lens.get(f"center_{axis}_m")
        if center is not None and not math.isfinite(float(center)):
            raise ValueError(
                f"seed.thermal_lens.center_{axis}_m must be finite when supplied"
            )
    if resolve_signal_model(config) == "commercial_app_fast":
        if physics["optical_propagation"] != "abcd":
            raise ValueError(
                "commercial_app_fast requires physics.optical_propagation=abcd"
            )
        if physics["nonlinear_loss_model"] not in {
            "off",
            "strong_field_effective_area",
        }:
            raise ValueError(
                "commercial_app_fast only supports physics.nonlinear_loss_model=off or "
                "strong_field_effective_area"
            )
        if str(config["seed"].get("joint_time_frequency", {}).get("model", "separable")) != "separable":
            raise ValueError(
                "commercial_app_fast requires seed.joint_time_frequency.model=separable"
            )
        commercial_app_config = config["seed"].get("commercial_app_fast", {})
        spatial_backend = commercial_app_config.get("spatial_backend", "cartesian")
        if spatial_backend not in {"cartesian", "axisymmetric"}:
            raise ValueError("seed.commercial_app_fast.spatial_backend must be cartesian or axisymmetric")
        if spatial_backend == "axisymmetric":
            if config["seed"].get("spatial_gain_transport", {}).get("model", "local_field") != "local_field":
                raise ValueError("Commercial App axisymmetric spatial backend requires local_field gain")
            optical = config["grid"]["optical"] if optical_grid_enabled else config["grid"]
            if (not optical_grid_enabled and optical.get("spacing_xy", "uniform") != "uniform") or not math.isclose(float(optical["size_x_m"]), float(optical["size_y_m"]), rel_tol=1e-10, abs_tol=1e-15):
                raise ValueError("Commercial App axisymmetric requires a uniform square optical window")
            radial = config["seed"]["coherent"]["axisymmetric"]
            require_exact_integer(radial.get("angular_samples", 12), name="Commercial App axisymmetric angular_samples", minimum=9, maximum=128)
            require_exact_integer(radial.get("radial_bins", 24), name="Commercial App axisymmetric radial_bins", minimum=8, maximum=min(int(optical["nx"]), int(optical["ny"])))
        temporal_model = str(
            commercial_app_config.get("temporal_model", "fixed_duration")
        )
        if temporal_model not in {"fixed_duration", "effective_area_spm"}:
            raise ValueError(
                "seed.commercial_app_fast.temporal_model must be fixed_duration or "
                "effective_area_spm"
            )
        if (
            physics["nonlinear_loss_model"] == "strong_field_effective_area"
            and temporal_model != "effective_area_spm"
        ):
            raise ValueError(
                "strong_field_effective_area requires "
                "seed.commercial_app_fast.temporal_model=effective_area_spm"
            )
        if (
            str(
                config["seed"].get("spatial_modes", {}).get(
                    "model", "effective_m2"
                )
            )
            == "incoherent_hg"
            and temporal_model != "fixed_duration"
        ):
            raise ValueError(
                "Commercial App incoherent HG shared-population transport currently requires "
                "seed.commercial_app_fast.temporal_model=fixed_duration"
            )
        include_relaxation = commercial_app_config.get(
            "include_relaxation_during_pulse", False
        )
        if not isinstance(include_relaxation, bool):
            raise ValueError(
                "seed.commercial_app_fast.include_relaxation_during_pulse must be true or false"
            )
        require_exact_integer(
            commercial_app_config.get("gain_temporal_bin_count", 129),
            name="seed.commercial_app_fast.gain_temporal_bin_count",
            minimum=3,
            maximum=4096,
        )
        carrier_periodic_tolerance = float(
            commercial_app_config.get("carrier_periodic_relative_tolerance", 1e-6)
        )
        if (
            not np.isfinite(carrier_periodic_tolerance)
            or carrier_periodic_tolerance <= 0.0
        ):
            raise ValueError(
                "seed.commercial_app_fast.carrier_periodic_relative_tolerance must be "
                "finite and positive"
            )
        require_exact_integer(
            commercial_app_config.get("carrier_periodic_maximum_iterations", 32),
            name="seed.commercial_app_fast.carrier_periodic_maximum_iterations",
            minimum=1,
            maximum=256,
        )
        if include_relaxation:
            if str(config["seed"].get("temporal", {}).get("type", "gaussian")) == "cw":
                raise ValueError(
                    "seed.commercial_app_fast.include_relaxation_during_pulse is not applicable to CW"
                )
            if temporal_model != "fixed_duration":
                raise ValueError(
                    "Commercial App intrapulse population relaxation currently requires "
                    "seed.commercial_app_fast.temporal_model=fixed_duration"
                )
            if config["energy_levels"].get("gain_override_m_inv") is not None:
                raise ValueError(
                    "Commercial App intrapulse population relaxation requires physical energy levels; "
                    "remove energy_levels.gain_override_m_inv"
                )
            periodic = config["simulation"].get("periodic_steady_state", {})
            if bool(periodic.get("enabled", False)) and str(
                periodic.get("method", "pulse_map")
            ) == "cycle_average":
                raise ValueError(
                    "Commercial App intrapulse population relaxation requires pulse_map or a "
                    "finite pulse train; cycle_average removes the individual pulse envelope"
                )
        beam_quality_model = str(
            commercial_app_config.get("beam_quality_model", "configured_m2")
        )
        if beam_quality_model not in {
            "configured_m2",
            "projection_residual",
        }:
            raise ValueError(
                "seed.commercial_app_fast.beam_quality_model must be configured_m2 or "
                "projection_residual"
            )
        if not isinstance(commercial_app_config.get("spectral_feedback", True), bool):
            raise ValueError(
                "seed.commercial_app_fast.spectral_feedback must be true or false"
            )
        require_exact_integer(
            commercial_app_config.get("spectral_feedback_bin_count", 65),
            name="seed.commercial_app_fast.spectral_feedback_bin_count",
            minimum=3,
            maximum=4096,
        )
        spectral_feedback_grid_mode = str(
            commercial_app_config.get(
                "spectral_feedback_grid_mode", "full_coherent_span"
            )
        )
        if spectral_feedback_grid_mode not in {
            "full_coherent_span",
            "seed_energy_support",
        }:
            raise ValueError(
                "seed.commercial_app_fast.spectral_feedback_grid_mode must be "
                "full_coherent_span or seed_energy_support"
            )
        thermal_lens = commercial_app_config.get("thermal_lens", {})
        if not isinstance(thermal_lens, dict):
            raise ValueError("seed.commercial_app_fast.thermal_lens must be an object")
        thermal_lens_model = str(thermal_lens.get("model", "sampled_phase"))
        if thermal_lens_model not in {"sampled_phase", "local_parabolic"}:
            raise ValueError(
                "seed.commercial_app_fast.thermal_lens.model must be sampled_phase or "
                "local_parabolic"
            )
        quadratic_model = str(
            thermal_lens.get("quadratic_model", "full_xy")
        )
        if quadratic_model not in {"full_xy", "axis_aligned"}:
            raise ValueError(
                "seed.commercial_app_fast.thermal_lens.quadratic_model must be "
                "full_xy or axis_aligned"
            )
        fit_radius_mode = str(thermal_lens.get("fit_radius_mode", "manual"))
        if fit_radius_mode != "manual":
            raise ValueError(
                "seed.commercial_app_fast.thermal_lens.fit_radius_mode must currently be manual"
            )
        if thermal_lens_model == "local_parabolic":
            for axis in "xy":
                value = thermal_lens.get(f"radius_{axis}_m")
                if (
                    value is None
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                ):
                    raise ValueError(
                        "seed.commercial_app_fast.thermal_lens."
                        f"radius_{axis}_m must be positive for local_parabolic"
                    )
        for axis in "xy":
            center = thermal_lens.get(f"center_{axis}_m")
            if center is not None and not math.isfinite(float(center)):
                raise ValueError(
                    "seed.commercial_app_fast.thermal_lens."
                    f"center_{axis}_m must be finite when supplied"
                )
        rotation = float(config["seed"].get("rotation_deg", 0.0))
        anisotropy_reasons = axis_aligned_q_anisotropy_reasons(config["seed"])
        if anisotropy_reasons and not np.isclose(
            rotation % 180.0, 0.0, atol=1e-12
        ):
            raise ValueError(
                "commercial_app_fast uses uncoupled axis-aligned q parameters; set "
                "seed.rotation_deg=0, remove all transverse anisotropy, or "
                "select a sampled-field backend. Coupled features: "
                + "; ".join(anisotropy_reasons)
            )
    for key in ("pump_depletion", "gain_saturation", "record_optical_planes"):
        if not isinstance(physics[key], bool):
            raise ValueError(f"physics.{key} must resolve to true or false")
    require_exact_integer(
        physics["steady_state_iterations"],
        name="physics.steady_state_iterations",
        minimum=1,
    )
    supported_elements = {
        "gain_medium",
        "free_space",
        "thin_lens",
        "double_pass_lens_relay",
        "thick_lens",
        "aperture",
        "interface",
        "tilt",
        "coordinate_break",
        "mirror",
        "phase_screen",
        "dispersive",
        "loss",
        "observation",
        "plane",
    }
    optical_routes = [
        (
            f"optical_train.{train_name}",
            train_name,
            config.get("optical_train", {}).get(train_name, []),
        )
        for train_name in ("seed", "pump", "between_pass")
    ]
    if bool(config["simulation"].get("multipass", {}).get("enabled", False)):
        # Compiled return routes execute the same Seed optics as ordinary
        # between-pass paths. Validate them with that complete contract before
        # transport can turn an incomplete element into a no-op or KeyError.
        optical_routes.extend(
            (
                f"simulation.multipass.transitions[{transition_index}].route",
                "between_pass",
                list(transition.route),
            )
            for transition_index, transition in enumerate(plan.transitions)
        )
    for route_name, train_name, elements in optical_routes:
        if not isinstance(elements, list):
            raise ValueError(f"{route_name} must be a list")
        gain_count = 0
        for element_index, element in enumerate(elements):
            if not isinstance(element, dict) or "type" not in element:
                raise ValueError(f"Every {route_name} element needs a type")
            if element["type"] not in supported_elements:
                raise ValueError(f"Unsupported {route_name} element {element['type']!r}")
            element_name = f"{route_name}[{element_index}]"
            _validate_optical_element_response(element, name=element_name)
            _validate_optical_element_geometry(element, name=element_name)
            if any(
                abs(float(element.get(key) or 0.0)) > 0.0
                for key in ("turn_angle_x_deg", "turn_angle_y_deg")
            ):
                raise ValueError(
                    f"{element_name}.turn_angle_* is a legacy migration-only "
                    "shorthand and is not executed in a canonical optical route; "
                    "use an explicit tilt or coordinate_break element instead"
                )
            route_backend = (
                str(physics.get("pump_propagation", "off"))
                if train_name == "pump"
                else str(physics.get("optical_propagation", "off"))
            )
            direct_pass_configured = any(
                str(element.get(key, "")) == "direct_pass"
                for key in ("model", "front_model", "back_model")
            )
            if direct_pass_configured and not (
                train_name == "pump"
                and route_backend == "stochastic_raytrace"
            ):
                raise ValueError(
                    f"{element_name} uses direct_pass, which is reserved for "
                    "stochastic Pump-ray volume boundaries"
                )
            if (
                element["type"] == "interface"
                and route_backend in {"fft", "hybrid_abcd_fft"}
                and (
                    abs(float(element.get("axis_angle_x_deg") or 0.0)) > 0.0
                    or abs(float(element.get("axis_angle_y_deg") or 0.0)) > 0.0
                )
            ):
                raise ValueError(
                    f"{element_name} uses an inclined generic interface, but "
                    f"{route_backend} does not implement tilted-plane "
                    "resampling. Use a supported tilt/coordinate_break element."
                )
            if element["type"] == "gain_medium":
                _validate_incidence_angles(element, name=element_name)
                if train_name == "pump":
                    resolve_pump_direction(
                        str(config["pump"].get("direction", "forward")),
                        str(element.get("direction", "auto")),
                    )
            distance_before = float(element.get("distance_before_m") or 0.0)
            if not math.isfinite(distance_before) or distance_before < 0.0:
                raise ValueError(f"{route_name} distance_before_m must be non-negative")
            if element["type"] == "double_pass_lens_relay":
                if train_name == "pump":
                    raise ValueError(
                        "double_pass_lens_relay is a Seed return-path element and cannot be placed in optical_train.pump"
                    )
            gain_count += element["type"] == "gain_medium"
        if train_name == "between_pass" and gain_count:
            raise ValueError(
                f"{route_name} cannot contain gain_medium; "
                "put repeated gain-medium encounters in simulation.multipass"
            )
        if train_name != "between_pass" and gain_count > 1:
            raise ValueError(f"{route_name} can contain at most one gain_medium marker")
    routed_seed_path = [
        *list(config.get("optical_train", {}).get("seed", [])),
        *list(config.get("optical_train", {}).get("between_pass", [])),
    ]
    if bool(config["simulation"].get("multipass", {}).get("enabled", False)):
        routed_seed_path.extend(
            element
            for transition in plan.transitions
            for element in transition.route
        )
    direction_sensitive = {
        "free_space",
        "thin_lens",
        "thick_lens",
        "aperture",
    }
    local_gain_angle_pending = False
    for index, element in enumerate(routed_seed_path):
        kind = str(element.get("type", ""))
        if local_gain_angle_pending and float(
            element.get("distance_before_m") or 0.0
        ) > 0.0:
            raise ValueError(
                "Propagation cannot apply a nonzero local gain_medium incidence "
                "angle to the outgoing reference ray before downstream free "
                f"propagation at Seed route element {index}. Add an explicit "
                "coordinate_break/tilt before that distance."
            )
        if kind == "gain_medium":
            local_gain_angle_pending = bool(
                abs(float(element.get("incidence_angle_x_deg") or 0.0)) > 0.0
                or abs(float(element.get("incidence_angle_y_deg") or 0.0)) > 0.0
            )
            continue
        if not local_gain_angle_pending:
            continue
        if kind in {"coordinate_break", "tilt"}:
            local_gain_angle_pending = False
            continue
        if kind in direction_sensitive:
            raise ValueError(
                "Propagation cannot apply a nonzero local gain_medium incidence "
                "angle to the outgoing reference ray before downstream free "
                f"propagation at Seed route element {index}. Add an explicit "
                "coordinate_break/tilt that defines the outgoing reference ray."
            )

    n2 = float(config["material"].get("n2_m2_W", 0.0))
    if not math.isfinite(n2):
        raise ValueError("material.n2_m2_W must be finite")
    strong_field = config["material"].get("strong_field", {})
    if strong_field.get("model", "off") not in {"off", "multiphoton"}:
        raise ValueError("material.strong_field.model must be off or multiphoton")
    mechanism = str(strong_field.get("mechanism", "unspecified"))
    allowed_mechanisms = {
        "unspecified",
        "host_interband",
        "measured_sample_effective",
        "impurity_sequential",
        "defect_assisted",
        "phenomenological_sensitivity",
    }
    if mechanism not in allowed_mechanisms:
        raise ValueError(
            "material.strong_field.mechanism must be one of "
            f"{sorted(allowed_mechanisms)}"
        )
    require_exact_integer(
        strong_field.get("multiphoton_order", 2),
        name="material.strong_field.multiphoton_order",
        minimum=2,
    )
    multiphoton_order = int(strong_field.get("multiphoton_order", 2))
    beta = float(strong_field.get("beta_SI", 0.0))
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("material.strong_field.beta_SI must be finite and non-negative")
    provenance = strong_field.get("provenance", {}) or {}
    if not isinstance(provenance, dict):
        raise ValueError("material.strong_field.provenance must be an object")
    provenance_status = str(provenance.get("status", "unverified"))
    allowed_provenance_statuses = {
        "unverified",
        "provisional_sensitivity",
        "literature",
        "sample_measured",
    }
    if provenance_status not in allowed_provenance_statuses:
        raise ValueError(
            "material.strong_field.provenance.status must be one of "
            f"{sorted(allowed_provenance_statuses)}"
        )
    for key in ("reference", "sample_identifier"):
        value = provenance.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"material.strong_field.provenance.{key} must be a string or null"
            )
    for key in (
        "reference_wavelength_m",
        "reference_pulse_duration_s",
        "bandgap_eV",
    ):
        value = provenance.get(key)
        if value is not None and (
            not math.isfinite(float(value)) or float(value) <= 0.0
        ):
            raise ValueError(
                f"material.strong_field.provenance.{key} must be finite and positive"
            )
    relative_uncertainty = provenance.get("relative_uncertainty")
    if relative_uncertainty is not None and (
        not math.isfinite(float(relative_uncertainty))
        or float(relative_uncertainty) < 0.0
    ):
        raise ValueError(
            "material.strong_field.provenance.relative_uncertainty must be "
            "finite and non-negative"
        )
    if mechanism == "host_interband" and beta > 0.0:
        bandgap_eV = provenance.get("bandgap_eV")
        reference_wavelength = provenance.get("reference_wavelength_m")
        if bandgap_eV is None or reference_wavelength is None:
            raise ValueError(
                "host_interband strong-field loss requires provenance.bandgap_eV "
                "and provenance.reference_wavelength_m"
            )
        photon_energy_eV = 1.2398419843320026e-6 / float(
            reference_wavelength
        )
        minimum_order = int(math.ceil(float(bandgap_eV) / photon_energy_eV))
        if multiphoton_order < minimum_order:
            raise ValueError(
                "host_interband multiphoton_order is below the band-gap threshold "
                f"at the provenance reference wavelength: configured K="
                f"{multiphoton_order}, minimum K={minimum_order}"
            )
    heat_fraction = float(strong_field.get("deposit_fraction_as_heat", 1.0))
    if not math.isfinite(heat_fraction) or not 0.0 <= heat_fraction <= 1.0:
        raise ValueError("strong-field deposit_fraction_as_heat must lie in [0, 1]")
    carrier = strong_field.get("free_carrier", {})
    if carrier.get("model", "off") not in {"off", "rate_equation"}:
        raise ValueError("strong-field free_carrier.model must be off or rate_equation")
    if (
        not math.isfinite(float(carrier.get("lifetime_s", 1e-9)))
        or float(carrier.get("lifetime_s", 1e-9)) <= 0.0
    ):
        raise ValueError("strong-field free-carrier lifetime must be positive")
    generation_efficiency = float(carrier.get("generation_efficiency", 1.0))
    if (
        not math.isfinite(generation_efficiency)
        or not 0.0 <= generation_efficiency <= 1.0
    ):
        raise ValueError("free-carrier generation_efficiency must lie in [0, 1]")
    carrier_absorption = float(carrier.get("absorption_cross_section_m2", 0.0))
    if not math.isfinite(carrier_absorption) or carrier_absorption < 0.0:
        raise ValueError("free-carrier absorption cross section must be non-negative")
    carrier_index = float(carrier.get("refractive_index_m3", 0.0))
    if not math.isfinite(carrier_index):
        raise ValueError("free-carrier refractive_index_m3 must be finite")
    if (
        carrier.get("model", "off") == "rate_equation"
        and float(config["simulation"]["repetition_rate_Hz"]) > 0.0
        and (
            (
                bool(periodic.get("enabled", False))
                and periodic.get("method", "pulse_map")
                == "cycle_average"
            )
            or (
                not bool(periodic.get("enabled", False))
                and str(
                    config["simulation"].get(
                        "pulse_train_initialization",
                        "configured_initial_population",
                    )
                )
                == "cycle_average"
            )
        )
    ):
        residual_fraction = math.exp(
            -1.0
            / (
                float(config["simulation"]["repetition_rate_Hz"])
                * float(carrier.get("lifetime_s", 1e-9))
            )
        )
        limit = float(
            periodic.get(
                "maximum_unmodeled_free_carrier_residual_fraction", 1e-4
            )
        )
        if residual_fraction > limit:
            raise ValueError(
                "cycle_average cannot discard a material free-carrier state with "
                f"inter-pulse residual fraction {residual_fraction:.3e} > {limit:.3e}; "
                "use an exact pulse map that carries free-carrier memory"
            )
    if (
        carrier.get("model", "off") == "rate_equation"
        and bool(periodic.get("enabled", False))
        and reduced_initializer
    ):
        raise ValueError(
            "reduced_pulse_map initialization does not carry a time-domain "
            "free-carrier state through its proxy; initialize the exact "
            "pulse_map without the reduced stage"
        )
    response = config["material"].get("spectroscopic_response", {})
    if response.get("range_policy", "record") not in {"record", "error"}:
        raise ValueError(
            "material.spectroscopic_response.range_policy must be record or error"
        )
    if response.get("mode", "off") not in {"off", "operating_point", "local_3d"}:
        raise ValueError(
            "material.spectroscopic_response.mode must be off, operating_point, or local_3d"
        )
    if response.get("mode", "off") == "local_3d" and coupling_model != "thermal_optical_fixed_point":
        raise ValueError(
            "material.spectroscopic_response.mode=local_3d requires "
            "multiphysics.coupling_model=thermal_optical_fixed_point"
        )
    if (
        not math.isfinite(float(response.get("reference_temperature_K", 293.15)))
        or float(response.get("reference_temperature_K", 293.15)) <= 0.0
    ):
        raise ValueError("spectroscopic-response reference temperature must be positive")
    if (
        not math.isfinite(float(response.get("reference_density_m3", 1.0)))
        or float(response.get("reference_density_m3", 1.0)) <= 0.0
    ):
        raise ValueError("spectroscopic-response reference density must be positive")
    lifetime_response = response.get("upper_state_lifetime", {})
    supported_response_keys = {
        "pump_absorption", "pump_emission", "signal_absorption", "signal_emission"
    }
    unknown_response_keys = set(response.get("cross_sections", {})) - supported_response_keys
    if unknown_response_keys:
        raise ValueError(
            "unsupported spectroscopic cross-section response keys: "
            + ", ".join(sorted(unknown_response_keys))
        )
    for key, law in response.get("cross_sections", {}).items():
        _validate_cross_section_response_law(
            law,
            name=f"material.spectroscopic_response.cross_sections.{key}",
        )
    _validate_scale_response_law(
        lifetime_response.get("temperature", {}),
        name="material.spectroscopic_response.upper_state_lifetime.temperature",
        allowed={"off", "linear_scale", "scale_table"},
    )
    _validate_scale_response_law(
        lifetime_response.get("concentration", {}),
        name="material.spectroscopic_response.upper_state_lifetime.concentration",
        allowed={"off", "linear_scale", "scale_table", "stern_volmer"},
    )
    dispersion = config["material"].get("dispersion", {})
    for key in ("gd_s_per_m", "gdd_s2_per_m", "tod_s3_per_m", "fod_s4_per_m"):
        if not math.isfinite(float(dispersion.get(key, 0.0))):
            raise ValueError(f"material.dispersion.{key} must be finite")
    sellmeier = dispersion.get("sellmeier")
    if sellmeier and sellmeier.get("enabled", True):
        b = sellmeier.get("B", [])
        c = sellmeier.get("C_um2", sellmeier.get("C", []))
        if not isinstance(b, list) or not isinstance(c, list) or len(b) != len(c) or not b:
            raise ValueError("material.dispersion.sellmeier needs equal non-empty B and C_um2 lists")
        if not all(math.isfinite(float(value)) for value in [*b, *c]):
            raise ValueError("material.dispersion.sellmeier coefficients must be finite")

    coherent = config["seed"].get("coherent", {})
    if str(
        coherent.get("complex_spectrum", {}).get(
            "invalid_sample_policy", "error"
        )
    ) not in {"error", "clip"}:
        raise ValueError(
            "seed.coherent.complex_spectrum.invalid_sample_policy must be error or clip"
        )
    frequency_padding_factor = float(
        coherent.get("frequency_padding_factor", 1.0)
    )
    if (
        not math.isfinite(frequency_padding_factor)
        or not 1.0 <= frequency_padding_factor <= 4.0
    ):
        raise ValueError(
            "seed.coherent.frequency_padding_factor must lie in [1, 4]"
        )
    boundary_fraction = float(coherent.get("temporal_boundary_fraction", 0.05))
    if not math.isfinite(boundary_fraction) or not 0.0 < boundary_fraction < 0.5:
        raise ValueError(
            "seed.coherent.temporal_boundary_fraction must lie in (0, 0.5)"
        )
    boundary_warning = float(
        coherent.get("temporal_boundary_warning_fraction", 1e-6)
    )
    if (
        not math.isfinite(boundary_warning)
        or not 0.0 <= boundary_warning <= 1.0
    ):
        raise ValueError(
            "seed.coherent.temporal_boundary_warning_fraction must lie in [0, 1]"
        )
    if coherent.get("failure_policy", "error") not in {"fallback", "error"}:
        raise ValueError("seed.coherent.failure_policy must be fallback or error")
    if coherent.get("input_mode", "coefficients") not in {"coefficients", "complex_spectrum", "phase_retrieval"}:
        raise ValueError("seed.coherent.input_mode is unsupported")
    fidelity_mode = str(coherent.get("fidelity_mode", "configured"))
    if fidelity_mode not in {
        "configured",
        "hybrid_whole_fluence",
        "axisymmetric_kerr",
        "axisymmetric_transient_gain",
        "reference_transient_gain",
    }:
        raise ValueError("seed.coherent.fidelity_mode is unsupported")
    coherent_active = resolve_signal_model(config) == "coherent_cpa"
    axisymmetric_fidelity = coherent_active and fidelity_mode in {
        "axisymmetric_kerr",
        "axisymmetric_transient_gain",
    }
    axisymmetric_transport = axisymmetric_fidelity or (
        resolve_signal_model(config) == "spectral_time"
        and spectral_time.get("spatial_backend") == "axisymmetric"
    )
    configured_azimuthal_fidelity = str(
        coherent.get("axisymmetric", {}).get(
            "azimuthal_fidelity", "m0"
        )
    )
    allowed_azimuthal_fidelities = {
        "m0",
        "m0_m1",
        "m0_m1_m2_quadratic",
        "m0_m1_m2",
    }
    if configured_azimuthal_fidelity not in allowed_azimuthal_fidelities:
        raise ValueError(
            "seed.coherent.axisymmetric.azimuthal_fidelity must be m0, "
            "m0_m1, m0_m1_m2_quadratic, or m0_m1_m2"
        )
    configured_azimuthal_order = {
        "m0": 0,
        "m0_m1": 1,
        "m0_m1_m2_quadratic": 2,
        "m0_m1_m2": 2,
    }.get(configured_azimuthal_fidelity, 0)
    if axisymmetric_fidelity and local_scheme != "symmetric_adaptive":
        raise ValueError(
            "axisymmetric coherent fidelity requires "
            "seed.coherent.local_operator.scheme=symmetric_adaptive"
        )
    radial_hg = config["seed"].get("spatial_modes", {}).get("model") == "incoherent_hg"
    if axisymmetric_transport and radial_hg:
        from .spatial_modes import incoherent_hg_modes

        modes = incoherent_hg_modes(config["seed"])
        if configured_azimuthal_order != 1 or any(
            int(mode["order_x"]) + int(mode["order_y"]) > 1 for mode in modes
        ):
            raise ValueError("radial incoherent HG requires m0_m1 and only HG00/HG10/HG01")
        if resolve_physics(config).get("nonlinear_loss_model", "off") != "off":
            raise ValueError("radial incoherent HG currently requires nonlinear loss off")
        if not math.isclose(float(config["seed"]["m2_x"]), float(config["seed"]["m2_y"])):
            raise ValueError("radial incoherent HG currently requires equal x/y M2")
    if axisymmetric_transport and not radial_hg and any(
        not math.isclose(
            float(config["seed"].get(key, 1.0)),
            1.0,
            rel_tol=1e-10,
            abs_tol=1e-12,
        )
        for key in ("m2_x", "m2_y")
    ):
        raise ValueError(
            "axisymmetric coherent fidelity requires "
            "seed.m2_x=seed.m2_y=1 because it propagates an explicit coherent "
            "field; an empirical M2 diffraction multiplier is not a physical "
            "replacement for unresolved spatial modes"
        )
    if axisymmetric_transport:
        seed = config["seed"]
        if configured_azimuthal_order < 2:
            for first_key, second_key, label in (
                ("waist_x_m", "waist_y_m", "Seed waist"),
                ("order_x", "order_y", "Seed super-Gaussian order"),
            ):
                if not math.isclose(
                    float(seed[first_key]),
                    float(seed[second_key]),
                    rel_tol=1e-8,
                    abs_tol=1e-15,
                ):
                    raise ValueError(
                        "the selected axisymmetric azimuthal fidelity requires "
                        f"equal x/y {label}; choose an m=2 option for an "
                        "elliptical entrance field"
                    )
            if not math.isclose(
                axis_focus_m(seed, "x"),
                axis_focus_m(seed, "y"),
                rel_tol=1e-8,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "the selected axisymmetric azimuthal fidelity requires "
                    "equal x/y Seed focus; choose an m=2 option for astigmatism"
                )
        active_grid = (
            config["grid"].get("optical", {})
            if bool(config["grid"].get("optical", {}).get("enabled", False))
            else config["grid"]
        )
        window_scale = max(
            float(active_grid["size_x_m"]),
            float(active_grid["size_y_m"]),
        )
        centre_tolerance = max(1e-15, 1e-12 * window_scale)
        if configured_azimuthal_order < 1:
            for axis in ("x", "y"):
                seed_centre = float(seed.get(f"center_{axis}_m", 0.0))
                grid_centre = float(active_grid.get(f"center_{axis}_m", 0.0))
                if not math.isclose(
                    seed_centre,
                    grid_centre,
                    rel_tol=0.0,
                    abs_tol=centre_tolerance,
                ):
                    raise ValueError(
                        "m=0 axisymmetric coherent fidelity requires the Seed "
                        "centre to coincide with the active optical-grid centre; "
                        "choose m=0+1 to retain decenter"
                    )
            if any(
                abs(float(seed.get(key, 0.0) or 0.0)) > 1e-15
                for key in (
                    "incidence_angle_x_deg",
                    "incidence_angle_y_deg",
                )
            ):
                raise ValueError(
                    "m=0 axisymmetric coherent fidelity requires a centred, "
                    "untilted Seed; choose m=0+1 to retain tilt"
                )
    if (
        axisymmetric_transport
        and str(
            config["seed"].get("spatio_spectral", {}).get(
                "model", "shared_source"
            )
        )
        != "shared_source"
    ):
        raise ValueError(
            "axisymmetric coherent fidelity currently requires "
            "seed.spatio_spectral.model=shared_source; wavelength-dependent "
            "waist, focus, center, tilt, or M2 contracts have not yet been "
            "proved for the radial reduction"
        )
    require_exact_integer(
        coherent.get("sample_count", 128),
        name="seed.coherent.sample_count",
        minimum=16,
    )
    sampling_contract = coherent.get("sampling_contract", {})
    for key in (
        "minimum_spectral_fwhm_bins",
        "minimum_input_fwhm_bins",
        "minimum_transform_limited_fwhm_bins",
    ):
        value = float(sampling_contract.get(key, 1.0))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"seed.coherent.sampling_contract.{key} must be positive"
            )
    for key in (
        "maximum_spectral_boundary_energy_fraction",
        "maximum_input_temporal_boundary_energy_fraction",
    ):
        value = float(sampling_contract.get(key, 0.0))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"seed.coherent.sampling_contract.{key} must lie in [0, 1]"
            )
    sampling_boundary = float(
        sampling_contract.get("boundary_sample_fraction", 0.05)
    )
    if not math.isfinite(sampling_boundary) or not 0.0 < sampling_boundary < 0.5:
        raise ValueError(
            "seed.coherent.sampling_contract.boundary_sample_fraction must lie in (0, 0.5)"
        )
    axisymmetric = coherent.get("axisymmetric", {})
    azimuthal_fidelity = str(
        axisymmetric.get("azimuthal_fidelity", "m0")
    )
    if azimuthal_fidelity not in allowed_azimuthal_fidelities:
        raise ValueError(
            "seed.coherent.axisymmetric.azimuthal_fidelity must be m0, "
            "m0_m1, m0_m1_m2_quadratic, or m0_m1_m2"
        )
    angular_samples = require_exact_integer(
        axisymmetric.get("angular_samples", 12),
        name="seed.coherent.axisymmetric.angular_samples",
        minimum=1,
        maximum=128,
    )
    retained_azimuthal_order = {
        "m0": 0,
        "m0_m1": 1,
        "m0_m1_m2_quadratic": 2,
        "m0_m1_m2": 2,
    }[azimuthal_fidelity]
    if retained_azimuthal_order > 0 and angular_samples < (
        4 * retained_azimuthal_order + 1
    ):
        raise ValueError(
            "seed.coherent.axisymmetric.angular_samples must be at least "
            f"{4 * retained_azimuthal_order + 1} for the selected azimuthal "
            "fidelity so cubic Kerr products do not alias into retained modes"
        )
    radial_bins = require_exact_integer(
        axisymmetric.get("radial_bins", 24),
        name="seed.coherent.axisymmetric.radial_bins",
        minimum=8,
        maximum=4096,
    )
    polar_commercial_app = (
        resolve_signal_model(config) == "commercial_app_fast"
        and config["seed"]["commercial_app_fast"].get("spatial_backend") == "axisymmetric"
    )
    if axisymmetric_transport or polar_commercial_app:
        from .grid import Grid3D, optical_grid_from_config

        optical = optical_grid_from_config(
            Grid3D.from_config(config["grid"]), config["grid"]
        )
        x = optical.x - 0.5 * (optical.x_edges[0] + optical.x_edges[-1])
        y = optical.y - 0.5 * (optical.y_edges[0] + optical.y_edges[-1])
        radius = 0.5 * min(np.ptp(optical.x_edges), np.ptp(optical.y_edges))
        annuli = np.floor(
            np.hypot(x[None, :], y[:, None]) / (radius / radial_bins)
        ).astype(int)
        if np.unique(annuli[annuli < radial_bins]).size != radial_bins:
            raise ValueError(
                "seed.coherent.axisymmetric.radial_bins leaves empty radial "
                "annuli on the Cartesian optical grid; "
                "reduce radial_bins or increase optical Nx/Ny"
            )
    require_exact_integer(
        axisymmetric.get("gain_spectral_bins", 65),
        name="seed.coherent.axisymmetric.gain_spectral_bins",
        minimum=3,
        maximum=65536,
    )
    if axisymmetric.get("radial_diffraction_kernel", "auto") not in {
        "auto",
        "numpy",
        "numba",
    }:
        raise ValueError(
            "seed.coherent.axisymmetric.radial_diffraction_kernel must be "
            "auto, numpy, or numba"
        )
    require_exact_integer(
        axisymmetric.get("radial_numba_minimum_systems", 512),
        name="seed.coherent.axisymmetric.radial_numba_minimum_systems",
        minimum=1,
    )
    require_exact_integer(
        axisymmetric.get(
            "radial_coefficient_cache_max_bytes", 134217728
        ),
        name=(
            "seed.coherent.axisymmetric."
            "radial_coefficient_cache_max_bytes"
        ),
        minimum=1,
    )
    for key, default in (
        ("maximum_projection_loss_fraction", 1e-3),
        ("maximum_boundary_energy_fraction", 1e-4),
        ("maximum_azimuthal_residual", 5e-2),
        ("maximum_population_realizability_correction_fraction", 5e-2),
        ("maximum_population_positivity_redistribution_fraction", 5e-2),
        ("maximum_gain_bin_relative_frequency_span", 5e-2),
        ("maximum_gain_reconstruction_normalization_correction", 2e-2),
    ):
        value = float(axisymmetric.get(key, default))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"seed.coherent.axisymmetric.{key} must lie in [0, 1]"
            )
    transient_gain = coherent.get("transient_gain", {})
    if transient_gain.get("method", "soft_window_spectral") != "soft_window_spectral":
        raise ValueError("seed.coherent.transient_gain.method must be soft_window_spectral")
    if transient_gain.get("failure_policy", "error") not in {"fallback", "error"}:
        raise ValueError("seed.coherent.transient_gain.failure_policy must be fallback or error")
    slice_count = require_exact_integer(
        transient_gain.get("slice_count", 9),
        name="seed.coherent.transient_gain.slice_count",
        minimum=2,
    )
    maximum_slice_count = require_exact_integer(
        transient_gain.get("maximum_slice_count", 65),
        name="seed.coherent.transient_gain.maximum_slice_count",
        minimum=2,
    )
    if maximum_slice_count < slice_count:
        raise ValueError("seed.coherent.transient_gain.maximum_slice_count must not be smaller than slice_count")
    if not isinstance(transient_gain.get("tail_protection_enabled", False), bool):
        raise ValueError("seed.coherent.transient_gain.tail_protection_enabled must be boolean")
    tail_coverage = transient_gain.get("tail_energy_coverage", 0.99)
    if isinstance(tail_coverage, bool) or not np.isfinite(tail_coverage) or not 0 < tail_coverage <= 1:
        raise ValueError("seed.coherent.transient_gain.tail_energy_coverage must lie in (0, 1]")
    if transient_gain.get("tail_protection_enabled", False) and slice_count < 3:
        raise ValueError("Protected soft windows require at least three total windows")
    depletion = float(transient_gain.get("maximum_depletion_per_slice", 0.05))
    if not math.isfinite(depletion) or not 0.0 < depletion < 1.0:
        raise ValueError("seed.coherent.transient_gain.maximum_depletion_per_slice must lie between 0 and 1")
    if transient_gain.get(
        "adaptive_depletion_metric", "total_population_change"
    ) not in {"total_population_change", "legacy_upper_relative"}:
        raise ValueError(
            "seed.coherent.transient_gain.adaptive_depletion_metric must be "
            "total_population_change or legacy_upper_relative"
        )
    for key in (
        "enabled",
        "adaptive_slices",
        "adaptive_refinement_predictor",
        "reuse_adaptive_slice_floor",
        "reuse_adaptive_slice_floor_across_propagations",
        "reuse_scratch_buffers",
        "fuse_single_transition_events",
        "include_relaxation_during_pulse",
    ):
        if not isinstance(transient_gain.get(key, False), bool):
            raise ValueError(f"seed.coherent.transient_gain.{key} must be true or false")
    if (
        coherent_active
        and fidelity_mode in {"hybrid_whole_fluence", "axisymmetric_kerr"}
        and bool(transient_gain.get("enabled", False))
    ):
        raise ValueError(
            "seed.coherent.fidelity_mode=hybrid_whole_fluence or "
            "axisymmetric_kerr requires "
            "seed.coherent.transient_gain.enabled=false"
        )
    if (
        coherent_active
        and fidelity_mode in {
            "reference_transient_gain",
            "axisymmetric_transient_gain",
        }
        and not bool(transient_gain.get("enabled", False))
    ):
        raise ValueError(
            "reference_transient_gain and axisymmetric_transient_gain require "
            "seed.coherent.transient_gain.enabled=true"
        )
    if (
        coherent_active
        and fidelity_mode == "axisymmetric_transient_gain"
        and bool(periodic.get("enabled", False))
        and str(periodic.get("method", "pulse_map")) != "pulse_map"
    ):
        raise ValueError(
            "axisymmetric_transient_gain requires periodic_steady_state.method="
            "pulse_map when periodic steady state is enabled"
        )
    if transient_gain.get("single_transition_kernel", "auto") not in {
        "auto",
        "numpy",
        "numba",
    }:
        raise ValueError(
            "seed.coherent.transient_gain.single_transition_kernel must be "
            "auto, numpy, or numba"
        )
    if transient_gain.get("spectral_coupling", "ordered_sweeps") not in {
        "ordered_sweeps",
        "broadband_two_stage",
        "auto",
    }:
        raise ValueError(
            "seed.coherent.transient_gain.spectral_coupling must be "
            "ordered_sweeps, broadband_two_stage, or auto"
        )
    if not isinstance(transient_gain.get("batch_window_ffts", False), bool):
        raise ValueError(
            "seed.coherent.transient_gain.batch_window_ffts must be true or false"
        )
    require_exact_integer(
        transient_gain.get("batch_window_fft_max_bytes", 268435456),
        name=(
            "seed.coherent.transient_gain.batch_window_fft_max_bytes"
        ),
        minimum=1,
    )
    bypassed_output_fraction = float(
        transient_gain.get(
            "maximum_bypassed_output_energy_fraction",
            0.0,
        )
    )
    if (
        not math.isfinite(bypassed_output_fraction)
        or not 0.0 <= bypassed_output_fraction < 1.0
    ):
        raise ValueError(
            "seed.coherent.transient_gain."
            "maximum_bypassed_output_energy_fraction must lie in [0, 1)"
        )
    require_exact_integer(
        transient_gain.get(
            "inactive_window_local_guard_maximum_pixels", 512
        ),
        name=(
            "seed.coherent.transient_gain."
            "inactive_window_local_guard_maximum_pixels"
        ),
        minimum=1,
    )
    require_exact_integer(
        transient_gain.get(
            "spectral_material_cache_max_bytes", 536870912
        ),
        name=(
            "seed.coherent.transient_gain."
            "spectral_material_cache_max_bytes"
        ),
        minimum=1,
    )
    require_exact_integer(
        transient_gain.get("single_transition_kernel_workers", 0),
        name=(
            "seed.coherent.transient_gain.single_transition_kernel_workers"
        ),
        minimum=0,
    )
    require_exact_integer(
        transient_gain.get(
            "single_transition_numba_minimum_packet_cells", 100000
        ),
        name=(
            "seed.coherent.transient_gain."
            "single_transition_numba_minimum_packet_cells"
        ),
        minimum=1,
    )
    relaxation_energy_fraction = float(
        transient_gain.get("relaxation_energy_fraction", 0.999999)
    )
    if (
        not math.isfinite(relaxation_energy_fraction)
        or not 0.0 < relaxation_energy_fraction <= 1.0
    ):
        raise ValueError(
            "seed.coherent.transient_gain.relaxation_energy_fraction must lie in (0, 1]"
        )
    retrieval = coherent.get("phase_retrieval", {})
    require_exact_integer(
        retrieval.get("max_iterations", 200),
        name="seed.coherent.phase_retrieval.max_iterations",
        minimum=1,
    )
    retrieval_tolerance = float(retrieval.get("tolerance", 1e-8))
    if not math.isfinite(retrieval_tolerance) or retrieval_tolerance <= 0.0:
        raise ValueError(
            "seed.coherent.phase_retrieval.tolerance must be positive and finite"
        )
    spectral_phase = coherent.get("spectral_phase", {})
    if not isinstance(spectral_phase.get("enabled", False), bool):
        raise ValueError("seed.coherent.spectral_phase.enabled must be boolean")
    if spectral_phase.get("wavelength_unit", "nm") not in WAVELENGTH_UNITS_TO_M:
        raise ValueError("seed.coherent.spectral_phase.wavelength_unit is unsupported")
    complex_spectrum = coherent.get("complex_spectrum", {})
    if complex_spectrum.get("wavelength_unit", "nm") not in WAVELENGTH_UNITS_TO_M:
        raise ValueError("seed.coherent.complex_spectrum.wavelength_unit is unsupported")
    if complex_spectrum.get("format", "amplitude_phase") not in {
        "amplitude_phase",
        "real_imag",
    }:
        raise ValueError(
            "seed.coherent.complex_spectrum.format must be amplitude_phase or real_imag"
        )
    if complex_spectrum.get("amplitude_domain", "per_frequency") not in {
        "per_wavelength",
        "per_frequency",
        "per_bin",
    }:
        raise ValueError(
            "seed.coherent.complex_spectrum.amplitude_domain must be "
            "per_wavelength, per_frequency, or per_bin"
        )
    for transition_name in (
        "pump_cross_section_spectrum",
        "signal_cross_section_spectrum",
    ):
        cross_section = config["energy_levels"].get(transition_name, {})
        if (
            cross_section
            and cross_section.get("wavelength_unit", "m")
            not in WAVELENGTH_UNITS_TO_M
        ):
            raise ValueError(
                f"energy_levels.{transition_name}.wavelength_unit is unsupported"
            )

    josa = config["seed"].get("josa", {})
    if josa.get("mode", "retarded_pc") not in JOSA_MODES:
        raise ValueError(f"seed.josa.mode must be one of {sorted(JOSA_MODES)}")
    if josa.get("failure_policy", "error") not in {"fallback", "error"}:
        raise ValueError("seed.josa.failure_policy must be fallback or error")
    if not isinstance(josa.get("include_relaxation_during_pulse", False), bool):
        raise ValueError("seed.josa.include_relaxation_during_pulse must be boolean")
    require_exact_integer(
        josa.get("maximum_time_bins", 4096),
        name="seed.josa.maximum_time_bins",
        minimum=1,
    )
    maximum_gain_step = float(josa.get("recommended_maximum_gain_step", 0.05))
    if not math.isfinite(maximum_gain_step) or maximum_gain_step <= 0.0:
        raise ValueError("seed.josa.recommended_maximum_gain_step must be positive and finite")
    if (
        config["simulation"].get("signal_model", "auto") == "josa_transport"
        and josa.get("mode", "retarded_pc") == JOSA_FULL_3D_MODE
        and config["energy_levels"].get("gain_override_m_inv") is not None
    ):
        raise ValueError(
            "full_3d_pc requires physical upper/lower populations; remove energy_levels.gain_override_m_inv"
        )
    for section in ("phase", "stretcher", "compressor"):
        coefficients = coherent.get(section, {})
        for key in ("constant_phase_rad", "gd_s", "gdd_s2", "tod_s3", "fod_s4"):
            if not math.isfinite(float(coefficients.get(key, 0.0))):
                raise ValueError(f"seed.coherent.{section}.{key} must be finite")
