"""Three-dimensional solid-state laser amplifier simulation."""

from .config import (
    apply_commercial_app_effective_area_strong_field_profile,
    apply_commercial_app_fast_profile,
    apply_commercial_app_seed_only_profile,
    apply_coherent_kerr_hybrid_profile,
    load_config,
    prepare_config,
    validate_config,
)
from .material_library import apply_material_record, load_material_library
from .solver import AmplifierSimulator, SimulationResult

__all__ = [
    "AmplifierSimulator",
    "SimulationResult",
    "apply_commercial_app_effective_area_strong_field_profile",
    "apply_commercial_app_fast_profile",
    "apply_commercial_app_seed_only_profile",
    "apply_coherent_kerr_hybrid_profile",
    "apply_material_record",
    "load_config",
    "load_material_library",
    "prepare_config",
    "validate_config",
]
__version__ = "0.1.0"
