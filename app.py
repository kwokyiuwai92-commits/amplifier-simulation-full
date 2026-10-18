"""Streamlit front end for the modular 3D amplifier engine."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Callable

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner_utils.script_run_context import (
    get_script_run_ctx,
)


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
UI_AUTOSAVE_DIRECTORY = Path(
    os.environ.get(
        "AMPLIFIER_SIM_UI_AUTOSAVE_DIRECTORY",
        str(PROJECT_DIR / "exports"),
    )
).expanduser()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from amplifier_sim.bench_preview import (  # noqa: E402
    BenchPreview,
    build_bench_preview,
    first_gain_entrance_m,
    sample_bench_preview,
)
from amplifier_sim.config import (  # noqa: E402
    DEFAULTS,
    prepare_config,
    resolve_physics,
    resolve_population_dynamics,
    resolve_signal_model,
    resolve_thermal_model,
    validate_config,
)
from amplifier_sim.constants import LIGHT_SPEED  # noqa: E402
from amplifier_sim.multipass import uses_causal_clock  # noqa: E402
from amplifier_sim.dispersion import (  # noqa: E402
    material_group_index,
    phase_from_coefficients,
)
from amplifier_sim.doping import dopant_density_field  # noqa: E402
from amplifier_sim.grid import Grid3D  # noqa: E402
from amplifier_sim.geometry import crystal_volume_mask  # noqa: E402
from amplifier_sim.levels import LevelSystem  # noqa: E402
from amplifier_sim.josa_modes import (  # noqa: E402
    JOSA_FULL_3D_MODE,
    JOSA_MODE_INFO,
    josa_mode_info,
)
from amplifier_sim.json_contract import (  # noqa: E402
    dumps_json_portable,
    loads_json_strict,
)
from amplifier_sim.i18n import (  # noqa: E402
    EN,
    ZH_CN,
    localize_structure,
    localize_source_message,
    normalize_locale,
    restore_localized_structure,
    translate_text,
)
from amplifier_sim.material_library import (  # noqa: E402
    apply_material_record,
    load_material_library,
    reconcile_material_field_provenance,
)
from amplifier_sim.model_explanation import (  # noqa: E402
    build_choice_help,
    build_field_help,
    build_toggle_help,
    describe_configuration,
)
from amplifier_sim.optical_system import gain_medium_path_cosine  # noqa: E402
from amplifier_sim.path_view import (  # noqa: E402
    align_previews_to_shared_gain,
    build_path_display_scale,
    gain_sample_mask,
    optical_trace_anchors,
)
from amplifier_sim.pulse_definition import (  # noqa: E402
    build_fourier_pulse_pair,
    intensity_fwhm as pulse_intensity_fwhm,
    spectral_file_pulse_energy_J,
    trim_negligible_mass_support,
)
from amplifier_sim.pump_source import canonicalize_pump_source  # noqa: E402
from amplifier_sim.result_archive import (  # noqa: E402
    ArchiveResourceBudget,
    ArchivePoint,
    ResultArchive,
    ResultArchiveError,
    diagnostic_record_for_result,
    load_result_archive_index,
    load_result_archive_point,
    require_completed_sweep_results,
    write_result_archive,
)
from amplifier_sim.result_fields import standard_result_arrays  # noqa: E402
from amplifier_sim.solver import SimulationResult  # noqa: E402
from amplifier_sim.strict_validation import NON_QUANTITATIVE_BANNER  # noqa: E402
from amplifier_sim.streaming_download import (  # noqa: E402
    register_streaming_download,
)
from amplifier_sim.seed_spatio_spectral import seed_config_at_wavelength  # noqa: E402
from amplifier_sim.spectral_reporting import (  # noqa: E402
    spectral_bin_coordinate_for_result,
    spectral_density_from_bin_energy,
)
from amplifier_sim.spatial_modes import (  # noqa: E402
    divergence_half_angle_rad,
    incoherent_hg_modes,
    maximum_entropy_mode_weights,
    rayleigh_range_m,
)
from amplifier_sim.sweep import (  # noqa: E402
    LEGACY_SWEEP_METRIC_KEYS,
    SWEEP_METRICS,
    SWEEP_PARAMETERS,
    SweepParameter,
    SweepGridRun,
    SweepRun,
    get_sweep_parameter_value,
    sweep_parameter_inactive_reason,
    supports_sweep_continuation,
    make_sweep_values,
)
from amplifier_sim.sweep_plotting import build_sweep_slice_figures, add_invalid_grid_markers  # noqa: E402
from amplifier_sim.ui_visualization import (  # noqa: E402
    SLICE_AXES,
    axial_weighted_statistics,
    convergence_residual_value,
    finite_weighted_statistics,
    finite_plot_value,
    signal_cross_section_preview,
    optical_plane_axes,
    optical_path_3d_figure,
    solver_notice_presentation,
    slice_figure,
    volume_figure,
)
from amplifier_sim.ui_state import (  # noqa: E402
    apply_quick_physics_config,
    PATH_EDITOR_COLUMNS,
    clean_table_rows,
    path_editor_widget_prefix,
    stringify_table_column,
)
from amplifier_sim.ui_solve_jobs import (  # noqa: E402
    ACTIVE_STATUSES,
    SolveJobView,
    _compact_live_progress_events,
    cancel_solve_job,
    get_solve_job,
    pause_solve_job,
    release_solve_job,
    resume_solve_job,
    start_solve_job,
)
from amplifier_sim.ui_sweep_jobs import (  # noqa: E402
    SweepJobView,
    cancel_sweep_job,
    get_sweep_job,
    pause_sweep_job,
    reconcile_sweep_session,
    release_sweep_job,
    resume_sweep_job,
    start_sweep_job,
    write_sweep_job_archive,
)
from amplifier_sim.ui_compat import (  # noqa: E402
    ase_unavailability_reason,
    dependency_contracts,
    reconcile_ui_dependencies,
    release_ui_dependency_ownership,
)

# Streamlit reruns this script in the same interpreter while retaining the
# imported ``streamlit`` module and ``DeltaGenerator`` class.  Keep the real
# widget callables on those persistent objects so a rerun never captures our
# previous wrapper as its own "native" function (which otherwise recurses).
from amplifier_sim.ui_native_api import (  # noqa: E402
    ST_NATIVE_INPUTS as _st_native_inputs,
    DG_NATIVE_INPUTS as _dg_native_inputs,
)
_native_selectbox = _st_native_inputs["selectbox"]
_native_radio = _st_native_inputs["radio"]
_native_checkbox = _st_native_inputs["checkbox"]
_native_toggle = _st_native_inputs["toggle"]
_native_multiselect = _st_native_inputs["multiselect"]
_native_slider = _st_native_inputs["slider"]
_native_number_input = _st_native_inputs["number_input"]
_native_text_input = _st_native_inputs["text_input"]
_native_text_area = _st_native_inputs["text_area"]
_native_file_uploader = _st_native_inputs["file_uploader"]
_native_button = _st_native_inputs["button"]
_native_download_button = _st_native_inputs["download_button"]
_native_markdown = _st_native_inputs["markdown"]
_native_caption = _st_native_inputs["caption"]
_native_title = _st_native_inputs["title"]
_native_header = _st_native_inputs["header"]
_native_subheader = _st_native_inputs["subheader"]
_native_info = _st_native_inputs["info"]
_native_warning = _st_native_inputs["warning"]
_native_error = _st_native_inputs["error"]
_native_success = _st_native_inputs["success"]
_native_metric = _st_native_inputs["metric"]
_native_tabs = _st_native_inputs["tabs"]
_native_expander = _st_native_inputs["expander"]
_native_popover = _st_native_inputs["popover"]
_native_spinner = _st_native_inputs["spinner"]
_native_dataframe = _st_native_inputs["dataframe"]
_native_data_editor = _st_native_inputs["data_editor"]
_native_plotly_chart = _st_native_inputs["plotly_chart"]
_native_write = _st_native_inputs["write"]

_native_dg_selectbox = _dg_native_inputs["selectbox"]
_native_dg_radio = _dg_native_inputs["radio"]
_native_dg_checkbox = _dg_native_inputs["checkbox"]
_native_dg_toggle = _dg_native_inputs["toggle"]
_native_dg_multiselect = _dg_native_inputs["multiselect"]
_native_dg_slider = _dg_native_inputs["slider"]
_native_dg_number_input = _dg_native_inputs["number_input"]
_native_dg_text_input = _dg_native_inputs["text_input"]
_native_dg_text_area = _dg_native_inputs["text_area"]
_native_dg_file_uploader = _dg_native_inputs["file_uploader"]
_native_dg_button = _dg_native_inputs["button"]
_native_dg_download_button = _dg_native_inputs["download_button"]
_native_dg_markdown = _dg_native_inputs["markdown"]
_native_dg_caption = _dg_native_inputs["caption"]
_native_dg_title = _dg_native_inputs["title"]
_native_dg_header = _dg_native_inputs["header"]
_native_dg_subheader = _dg_native_inputs["subheader"]
_native_dg_info = _dg_native_inputs["info"]
_native_dg_warning = _dg_native_inputs["warning"]
_native_dg_error = _dg_native_inputs["error"]
_native_dg_success = _dg_native_inputs["success"]
_native_dg_metric = _dg_native_inputs["metric"]
_native_dg_tabs = _dg_native_inputs["tabs"]
_native_dg_expander = _dg_native_inputs["expander"]
_native_dg_popover = _dg_native_inputs["popover"]
_native_dg_spinner = _dg_native_inputs["spinner"]
_native_dg_dataframe = _dg_native_inputs["dataframe"]
_native_dg_data_editor = _dg_native_inputs["data_editor"]
_native_dg_plotly_chart = _dg_native_inputs["plotly_chart"]
_native_dg_write = _dg_native_inputs["write"]


def active_locale() -> str:
    """Read this session's locale without sharing state across user threads."""

    return normalize_locale(st.session_state.get("ui_locale", ZH_CN))


def lt(value: Any) -> Any:
    """Localize a display scalar while preserving non-string values."""

    return translate_text(value, active_locale()) if isinstance(value, str) else value


def _localized_choice_format(format_func: Any = str) -> Any:
    source_format = format_func if callable(format_func) else str
    locale = active_locale()

    def render(value: Any) -> str:
        return translate_text(str(source_format(value)), locale)

    return render


def _choice_values_equal(left: Any, right: Any) -> bool:
    try:
        comparison = left == right
        return bool(comparison) if isinstance(comparison, (bool, np.bool_)) else False
    except Exception:
        return False


def _localized_choice_state(
    kwargs: dict[str, Any],
    *,
    default_argument: str,
) -> tuple[dict[str, Any], tuple[Any, str] | None]:
    """Give English choices a fresh browser identity without changing values.

    Streamlit's browser can retain the old formatted label when a keyed choice
    control changes only its ``format_func``.  A locale-specific presentation
    key forces the displayed options and selected marker to be rebuilt.  The
    shadow value is app-owned (not a widget key), so it safely carries the raw
    selection across Chinese → English → Chinese transitions.
    """

    localized = dict(kwargs)
    original_key = localized.get("key")
    if original_key is None:
        return localized, None

    shadow_key = f"{original_key}__i18n_choice_value"
    display_key = (
        f"{original_key}__locale_en"
        if active_locale() == EN
        else original_key
    )
    shadow_exists = shadow_key in st.session_state
    canonical_exists = original_key in st.session_state
    candidate: Any = None
    has_candidate = False

    # A direct assignment by application code (for example, returning from
    # the JSON editor to the form) takes precedence over the saved shadow.
    if canonical_exists and (
        not shadow_exists
        or not _choice_values_equal(
            st.session_state[original_key], st.session_state[shadow_key]
        )
    ):
        candidate = copy.deepcopy(st.session_state[original_key])
        has_candidate = True
        st.session_state[shadow_key] = copy.deepcopy(candidate)
    elif shadow_exists:
        candidate = copy.deepcopy(st.session_state[shadow_key])
        has_candidate = True
    elif canonical_exists:
        candidate = copy.deepcopy(st.session_state[original_key])
        has_candidate = True

    if has_candidate and (
        display_key not in st.session_state
        or (
            active_locale() == EN
            and canonical_exists
            and not _choice_values_equal(
                st.session_state[original_key], st.session_state[display_key]
            )
        )
    ):
        st.session_state[display_key] = copy.deepcopy(candidate)
    localized["key"] = display_key
    if display_key in st.session_state:
        localized.pop(default_argument, None)
    return localized, (original_key, shadow_key)


def _remember_choice_value(
    value: Any,
    state_keys: tuple[Any, str] | None,
) -> Any:
    if state_keys is not None:
        _, shadow_key = state_keys
        st.session_state[shadow_key] = copy.deepcopy(value)
    return value


def _localize_widget_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    localized = dict(kwargs)
    for name in ("help", "placeholder"):
        if isinstance(localized.get(name), str):
            localized[name] = lt(localized[name])
    captions = localized.get("captions")
    if captions is not None:
        localized["captions"] = [lt(str(caption)) for caption in captions]
    return localized


def choice_selectbox(
    label: str,
    options: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Render a selectbox whose question mark explains every option."""

    option_values = list(options)
    kwargs, state_keys = _localized_choice_state(
        kwargs, default_argument="index"
    )
    kwargs["help"] = build_choice_help(
        label,
        option_values,
        kwargs.get("help"),
    )
    kwargs["format_func"] = _localized_choice_format(
        kwargs.get("format_func", str)
    )
    kwargs = _localize_widget_kwargs(kwargs)
    return _remember_choice_value(
        _native_selectbox(lt(label), option_values, *args, **kwargs),
        state_keys,
    )


def choice_radio(
    label: str,
    options: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Render a radio group with the same complete option documentation."""

    option_values = list(options)
    kwargs, state_keys = _localized_choice_state(
        kwargs, default_argument="index"
    )
    kwargs["help"] = build_choice_help(
        label,
        option_values,
        kwargs.get("help"),
    )
    kwargs["format_func"] = _localized_choice_format(
        kwargs.get("format_func", str)
    )
    kwargs = _localize_widget_kwargs(kwargs)
    return _remember_choice_value(
        _native_radio(lt(label), option_values, *args, **kwargs),
        state_keys,
    )


def choice_multiselect(
    label: str,
    options: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    option_values = list(options)
    kwargs, state_keys = _localized_choice_state(
        kwargs, default_argument="default"
    )
    kwargs["help"] = build_choice_help(
        label,
        option_values,
        kwargs.get("help"),
    )
    kwargs["format_func"] = _localized_choice_format(
        kwargs.get("format_func", str)
    )
    kwargs = _localize_widget_kwargs(kwargs)
    return _remember_choice_value(
        _native_multiselect(lt(label), option_values, *args, **kwargs),
        state_keys,
    )


def choice_checkbox(label: str, *args: Any, **kwargs: Any) -> bool:
    key = kwargs.get("key")
    if key is not None and key in st.session_state:
        kwargs.pop("value", None)
        if args:
            args = args[1:]
    kwargs["help"] = build_toggle_help(label, kwargs.get("help"))
    kwargs = _localize_widget_kwargs(kwargs)
    return bool(_native_checkbox(lt(label), *args, **kwargs))


def choice_toggle(label: str, *args: Any, **kwargs: Any) -> bool:
    key = kwargs.get("key")
    if key is not None and key in st.session_state:
        kwargs.pop("value", None)
        if args:
            args = args[1:]
    kwargs["help"] = build_toggle_help(label, kwargs.get("help"))
    kwargs = _localize_widget_kwargs(kwargs)
    return bool(_native_toggle(lt(label), *args, **kwargs))


def _number_input_bounds_including_current(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Keep a valid imported numeric value representable by its form widget.

    Streamlit rejects a ``value`` outside an explicitly supplied min/max.
    Several controls intentionally present a recommended interactive range
    that is narrower than the runtime schema (notably the physical grid).
    Expand only the presentation boundary needed to contain the current
    configuration; this neither clamps nor rewrites the scientific value.
    """

    positional = list(args)

    def read(name: str, index: int) -> Any:
        if name in kwargs:
            return kwargs[name]
        return positional[index] if len(positional) > index else None

    def write(name: str, index: int, value: Any) -> None:
        if name in kwargs:
            kwargs[name] = value
        elif len(positional) > index:
            positional[index] = value

    current = read("value", 2)
    key = kwargs.get("key")
    if key is not None and key in st.session_state:
        session_value = st.session_state.get(key)
        if isinstance(session_value, (int, float, np.integer, np.floating)):
            current_values = (current, session_value)
        else:
            current_values = (current,)
    else:
        current_values = (current,)
    finite_values = [
        value
        for value in current_values
        if isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(value)
    ]
    if not finite_values:
        return tuple(positional), kwargs
    minimum = read("min_value", 0)
    maximum = read("max_value", 1)
    if minimum is not None:
        smallest = min(finite_values)
        if smallest < minimum:
            write("min_value", 0, type(minimum)(smallest))
    if maximum is not None:
        largest = max(finite_values)
        if largest > maximum:
            write("max_value", 1, type(maximum)(largest))
    return tuple(positional), kwargs


def documented_number_input(label: str, *args: Any, **kwargs: Any) -> Any:
    kwargs["help"] = build_field_help(
        label,
        kwargs.get("help"),
        control_kind="number_input",
    )
    kwargs = _localize_widget_kwargs(kwargs)
    args, kwargs = _number_input_bounds_including_current(args, kwargs)
    return _native_number_input(lt(label), *args, **kwargs)


def documented_text_input(label: str, *args: Any, **kwargs: Any) -> Any:
    kwargs["help"] = build_field_help(
        label,
        kwargs.get("help"),
        control_kind="text_input",
    )
    kwargs = _localize_widget_kwargs(kwargs)
    return _native_text_input(lt(label), *args, **kwargs)


def documented_text_area(label: str, *args: Any, **kwargs: Any) -> Any:
    kwargs["help"] = build_field_help(
        label,
        kwargs.get("help"),
        control_kind="text_input",
    )
    kwargs = _localize_widget_kwargs(kwargs)
    return _native_text_area(lt(label), *args, **kwargs)


def documented_file_uploader(label: str, *args: Any, **kwargs: Any) -> Any:
    kwargs["help"] = build_field_help(
        label,
        kwargs.get("help"),
        control_kind="file_uploader",
    )
    kwargs = _localize_widget_kwargs(kwargs)
    return _native_file_uploader(lt(label), *args, **kwargs)


def documented_button(label: str, *args: Any, **kwargs: Any) -> Any:
    kwargs["help"] = build_field_help(
        label,
        kwargs.get("help"),
        control_kind="button",
    )
    kwargs = _localize_widget_kwargs(kwargs)
    return _native_button(lt(label), *args, **kwargs)


def documented_download_button(label: str, *args: Any, **kwargs: Any) -> Any:
    kwargs["help"] = build_field_help(
        label,
        kwargs.get("help"),
        control_kind="download_button",
    )
    kwargs = _localize_widget_kwargs(kwargs)
    return _native_download_button(lt(label), *args, **kwargs)


def _choice_dg_selectbox(
    self: DeltaGenerator,
    label: str,
    options: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    option_values = list(options)
    kwargs, state_keys = _localized_choice_state(
        kwargs, default_argument="index"
    )
    kwargs["help"] = build_choice_help(label, option_values, kwargs.get("help"))
    kwargs["format_func"] = _localized_choice_format(kwargs.get("format_func", str))
    kwargs = _localize_widget_kwargs(kwargs)
    return _remember_choice_value(
        _native_dg_selectbox(self, lt(label), option_values, *args, **kwargs),
        state_keys,
    )


def _choice_dg_radio(
    self: DeltaGenerator,
    label: str,
    options: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    option_values = list(options)
    kwargs, state_keys = _localized_choice_state(
        kwargs, default_argument="index"
    )
    kwargs["help"] = build_choice_help(label, option_values, kwargs.get("help"))
    kwargs["format_func"] = _localized_choice_format(kwargs.get("format_func", str))
    kwargs = _localize_widget_kwargs(kwargs)
    return _remember_choice_value(
        _native_dg_radio(self, lt(label), option_values, *args, **kwargs),
        state_keys,
    )


def _choice_dg_multiselect(
    self: DeltaGenerator,
    label: str,
    options: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    option_values = list(options)
    kwargs, state_keys = _localized_choice_state(
        kwargs, default_argument="default"
    )
    kwargs["help"] = build_choice_help(label, option_values, kwargs.get("help"))
    kwargs["format_func"] = _localized_choice_format(kwargs.get("format_func", str))
    kwargs = _localize_widget_kwargs(kwargs)
    return _remember_choice_value(
        _native_dg_multiselect(self, lt(label), option_values, *args, **kwargs),
        state_keys,
    )


def _choice_dg_checkbox(
    self: DeltaGenerator,
    label: str,
    *args: Any,
    **kwargs: Any,
) -> bool:
    key = kwargs.get("key")
    if key is not None and key in st.session_state:
        kwargs.pop("value", None)
        if args:
            args = args[1:]
    kwargs["help"] = build_toggle_help(label, kwargs.get("help"))
    kwargs = _localize_widget_kwargs(kwargs)
    return bool(_native_dg_checkbox(self, lt(label), *args, **kwargs))


def _choice_dg_toggle(
    self: DeltaGenerator,
    label: str,
    *args: Any,
    **kwargs: Any,
) -> bool:
    key = kwargs.get("key")
    if key is not None and key in st.session_state:
        kwargs.pop("value", None)
        if args:
            args = args[1:]
    kwargs["help"] = build_toggle_help(label, kwargs.get("help"))
    kwargs = _localize_widget_kwargs(kwargs)
    return bool(_native_dg_toggle(self, lt(label), *args, **kwargs))


def _documented_dg_number_input(
    self: DeltaGenerator,
    label: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    kwargs["help"] = build_field_help(
        label,
        kwargs.get("help"),
        control_kind="number_input",
    )
    kwargs = _localize_widget_kwargs(kwargs)
    args, kwargs = _number_input_bounds_including_current(args, kwargs)
    return _native_dg_number_input(self, lt(label), *args, **kwargs)


def _documented_dg_text_input(
    self: DeltaGenerator,
    label: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    kwargs["help"] = build_field_help(
        label,
        kwargs.get("help"),
        control_kind="text_input",
    )
    kwargs = _localize_widget_kwargs(kwargs)
    return _native_dg_text_input(self, lt(label), *args, **kwargs)


def _documented_dg_text_area(
    self: DeltaGenerator,
    label: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    kwargs["help"] = build_field_help(
        label,
        kwargs.get("help"),
        control_kind="text_input",
    )
    kwargs = _localize_widget_kwargs(kwargs)
    return _native_dg_text_area(self, lt(label), *args, **kwargs)


def _documented_dg_file_uploader(
    self: DeltaGenerator,
    label: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    kwargs["help"] = build_field_help(
        label,
        kwargs.get("help"),
        control_kind="file_uploader",
    )
    kwargs = _localize_widget_kwargs(kwargs)
    return _native_dg_file_uploader(self, lt(label), *args, **kwargs)


def _documented_dg_button(
    self: DeltaGenerator,
    label: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    kwargs["help"] = build_field_help(
        label,
        kwargs.get("help"),
        control_kind="button",
    )
    kwargs = _localize_widget_kwargs(kwargs)
    return _native_dg_button(self, lt(label), *args, **kwargs)


def _documented_dg_download_button(
    self: DeltaGenerator,
    label: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    kwargs["help"] = build_field_help(
        label,
        kwargs.get("help"),
        control_kind="download_button",
    )
    kwargs = _localize_widget_kwargs(kwargs)
    return _native_dg_download_button(self, lt(label), *args, **kwargs)


def localized_slider(label: str, *args: Any, **kwargs: Any) -> Any:
    kwargs = _localize_widget_kwargs(kwargs)
    return _native_slider(lt(label), *args, **kwargs)


def _localized_dg_slider(
    self: DeltaGenerator,
    label: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    kwargs = _localize_widget_kwargs(kwargs)
    return _native_dg_slider(self, lt(label), *args, **kwargs)


def _localized_message_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    localized = dict(kwargs)
    for name in ("help", "title"):
        if isinstance(localized.get(name), str):
            localized[name] = lt(localized[name])
    return localized


def _module_text_wrapper(native: Any) -> Any:
    def render(body: Any, *args: Any, **kwargs: Any) -> Any:
        localized_body = lt(body) if isinstance(body, str) else body
        return native(localized_body, *args, **_localized_message_kwargs(kwargs))

    return render


def _dg_text_wrapper(native: Any) -> Any:
    def render(
        self: DeltaGenerator,
        body: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        localized_body = lt(body) if isinstance(body, str) else body
        return native(
            self,
            localized_body,
            *args,
            **_localized_message_kwargs(kwargs),
        )

    return render


def localized_metric(
    label: str,
    value: Any,
    delta: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    localized_value = lt(value) if isinstance(value, str) else value
    localized_delta = lt(delta) if isinstance(delta, str) else delta
    kwargs = _localized_message_kwargs(kwargs)
    if isinstance(kwargs.get("delta_description"), str):
        kwargs["delta_description"] = lt(kwargs["delta_description"])
    return _native_metric(
        lt(label), localized_value, localized_delta, *args, **kwargs
    )


def _localized_dg_metric(
    self: DeltaGenerator,
    label: str,
    value: Any,
    delta: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    localized_value = lt(value) if isinstance(value, str) else value
    localized_delta = lt(delta) if isinstance(delta, str) else delta
    kwargs = _localized_message_kwargs(kwargs)
    if isinstance(kwargs.get("delta_description"), str):
        kwargs["delta_description"] = lt(kwargs["delta_description"])
    return _native_dg_metric(
        self, lt(label), localized_value, localized_delta, *args, **kwargs
    )


def localized_tabs(labels: Any, *args: Any, **kwargs: Any) -> Any:
    localized_labels = [lt(str(label)) for label in labels]
    if isinstance(kwargs.get("default"), str):
        kwargs["default"] = lt(kwargs["default"])
    return _native_tabs(localized_labels, *args, **kwargs)


def _localized_dg_tabs(
    self: DeltaGenerator,
    labels: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    localized_labels = [lt(str(label)) for label in labels]
    if isinstance(kwargs.get("default"), str):
        kwargs["default"] = lt(kwargs["default"])
    return _native_dg_tabs(self, localized_labels, *args, **kwargs)


def localized_expander(label: str, *args: Any, **kwargs: Any) -> Any:
    return _native_expander(lt(label), *args, **kwargs)


def _localized_dg_expander(
    self: DeltaGenerator,
    label: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    return _native_dg_expander(self, lt(label), *args, **kwargs)


def localized_popover(label: str, *args: Any, **kwargs: Any) -> Any:
    if isinstance(kwargs.get("help"), str):
        kwargs["help"] = lt(kwargs["help"])
    return _native_popover(lt(label), *args, **kwargs)


def _localized_dg_popover(
    self: DeltaGenerator,
    label: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if isinstance(kwargs.get("help"), str):
        kwargs["help"] = lt(kwargs["help"])
    return _native_dg_popover(self, lt(label), *args, **kwargs)


def localized_spinner(text: str = "In progress...", *args: Any, **kwargs: Any) -> Any:
    return _native_spinner(lt(text), *args, **kwargs)


def _localized_dg_spinner(
    self: DeltaGenerator,
    text: str = "In progress...",
    *args: Any,
    **kwargs: Any,
) -> Any:
    return _native_dg_spinner(self, lt(text), *args, **kwargs)


EDITABLE_TABLE_ENUM_COLUMNS = frozenset({"分布形式", "作用方向"})


def _translated_editor_data(data: Any, *, restore: bool = False) -> Any:
    """Translate editor headings and known enums, preserving user text."""

    transform = restore_localized_structure if restore else localize_structure

    def column_value(column: Any, value: Any) -> Any:
        return transform(
            value,
            EN,
            translate_keys=False,
            translate_values=column in EDITABLE_TABLE_ENUM_COLUMNS,
        )

    def column_name(column: Any) -> Any:
        return transform(column, EN) if isinstance(column, str) else column

    if hasattr(data, "columns") and hasattr(data, "copy"):
        translated = data.copy()
        original_columns = list(translated.columns)
        for column in original_columns:
            canonical_column = column_name(column) if restore else column
            translated[column] = translated[column].map(
                lambda value, key=canonical_column: column_value(key, value)
            )
        translated.columns = [column_name(column) for column in original_columns]
        return translated
    if isinstance(data, dict):
        return {
            column_name(column): column_value(
                column_name(column) if restore else column, value
            )
            for column, value in data.items()
        }
    if isinstance(data, list):
        return [_translated_editor_data(row, restore=restore) for row in data]
    if isinstance(data, tuple):
        return tuple(_translated_editor_data(row, restore=restore) for row in data)
    return transform(data, EN, translate_keys=False, translate_values=False)


def _localized_table_data(data: Any, *, editable: bool = False) -> Any:
    """Localize a table copy while leaving solver/archive objects untouched."""

    if active_locale() != EN or data is None:
        return data
    if editable:
        return _translated_editor_data(data)
    if hasattr(data, "columns") and hasattr(data, "copy"):
        try:
            localized = data.copy()
            localized.columns = [
                lt(str(column)) if isinstance(column, str) else column
                for column in localized.columns
            ]
            for column in localized.columns:
                localized[column] = localized[column].map(
                    lambda item: localize_structure(item, EN)
                )
            return localized
        except Exception:
            return data
    return localize_structure(copy.deepcopy(data), EN)


def _restore_table_data(data: Any) -> Any:
    if active_locale() != EN or data is None:
        return data
    return _translated_editor_data(data, restore=True)


def _localized_table_kwargs(
    kwargs: dict[str, Any],
    *,
    isolate_editor_state: bool = False,
) -> dict[str, Any]:
    if active_locale() != EN:
        return kwargs
    localized = dict(kwargs)
    if localized.get("column_order") is not None:
        localized["column_order"] = [
            lt(str(column)) if isinstance(column, str) else column
            for column in localized["column_order"]
        ]
    if isinstance(localized.get("column_config"), dict):
        localized["column_config"] = localize_structure(
            copy.deepcopy(localized["column_config"]), EN
        )
    disabled = localized.get("disabled")
    if disabled is not None and not isinstance(disabled, (bool, str)):
        localized["disabled"] = [
            lt(str(column)) if isinstance(column, str) else column
            for column in disabled
        ]
    if isinstance(localized.get("placeholder"), str):
        localized["placeholder"] = lt(localized["placeholder"])
    if isolate_editor_state and localized.get("key") is not None:
        localized["key"] = f"{localized['key']}__locale_en"
    return localized


def localized_dataframe(data: Any = None, *args: Any, **kwargs: Any) -> Any:
    return _native_dataframe(
        _localized_table_data(data),
        *args,
        **_localized_table_kwargs(kwargs),
    )


def _localized_dg_dataframe(
    self: DeltaGenerator,
    data: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    return _native_dg_dataframe(
        self,
        _localized_table_data(data),
        *args,
        **_localized_table_kwargs(kwargs),
    )


def localized_data_editor(data: Any, *args: Any, **kwargs: Any) -> Any:
    edited = _native_data_editor(
        _localized_table_data(data, editable=True),
        *args,
        **_localized_table_kwargs(kwargs, isolate_editor_state=True),
    )
    return _restore_table_data(edited)


def _localized_dg_data_editor(
    self: DeltaGenerator,
    data: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    edited = _native_dg_data_editor(
        self,
        _localized_table_data(data, editable=True),
        *args,
        **_localized_table_kwargs(kwargs, isolate_editor_state=True),
    )
    return _restore_table_data(edited)


def _localized_plotly_data(figure_or_data: Any) -> Any:
    if active_locale() != EN:
        return figure_or_data
    if isinstance(figure_or_data, go.Figure):
        translated = localize_structure(
            figure_or_data.to_plotly_json(), EN, translate_keys=False
        )
        return go.Figure(translated)
    return localize_structure(
        copy.deepcopy(figure_or_data), EN, translate_keys=False
    )


def localized_plotly_chart(
    figure_or_data: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    return _native_plotly_chart(
        _localized_plotly_data(figure_or_data), *args, **kwargs
    )


def _localized_dg_plotly_chart(
    self: DeltaGenerator,
    figure_or_data: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    return _native_dg_plotly_chart(
        self, _localized_plotly_data(figure_or_data), *args, **kwargs
    )


def localized_write(*args: Any, **kwargs: Any) -> Any:
    localized_args = tuple(
        lt(item)
        if isinstance(item, str)
        else _localized_table_data(item)
        for item in args
    )
    return _native_write(*localized_args, **kwargs)


def _localized_dg_write(
    self: DeltaGenerator,
    *args: Any,
    **kwargs: Any,
) -> Any:
    localized_args = tuple(
        lt(item)
        if isinstance(item, str)
        else _localized_table_data(item)
        for item in args
    )
    return _native_dg_write(self, *localized_args, **kwargs)


# Streamlit exposes the same inputs both from ``st`` and from columns/forms.
# Install one documentation layer for both surfaces so newly added fields do
# not silently lose their question-mark explanation.
st.selectbox = choice_selectbox
st.radio = choice_radio
st.checkbox = choice_checkbox
st.toggle = choice_toggle
st.multiselect = choice_multiselect
st.slider = localized_slider
st.number_input = documented_number_input
st.text_input = documented_text_input
st.text_area = documented_text_area
st.file_uploader = documented_file_uploader
st.button = documented_button
st.download_button = documented_download_button
st.markdown = _module_text_wrapper(_native_markdown)
st.caption = _module_text_wrapper(_native_caption)
st.title = _module_text_wrapper(_native_title)
st.header = _module_text_wrapper(_native_header)
st.subheader = _module_text_wrapper(_native_subheader)
st.info = _module_text_wrapper(_native_info)
st.warning = _module_text_wrapper(_native_warning)
st.error = _module_text_wrapper(_native_error)
st.success = _module_text_wrapper(_native_success)
st.metric = localized_metric
st.tabs = localized_tabs
st.expander = localized_expander
st.popover = localized_popover
st.spinner = localized_spinner
st.dataframe = localized_dataframe
st.data_editor = localized_data_editor
st.plotly_chart = localized_plotly_chart
st.write = localized_write
DeltaGenerator.selectbox = _choice_dg_selectbox
DeltaGenerator.radio = _choice_dg_radio
DeltaGenerator.checkbox = _choice_dg_checkbox
DeltaGenerator.toggle = _choice_dg_toggle
DeltaGenerator.multiselect = _choice_dg_multiselect
DeltaGenerator.slider = _localized_dg_slider
DeltaGenerator.number_input = _documented_dg_number_input
DeltaGenerator.text_input = _documented_dg_text_input
DeltaGenerator.text_area = _documented_dg_text_area
DeltaGenerator.file_uploader = _documented_dg_file_uploader
DeltaGenerator.button = _documented_dg_button
DeltaGenerator.download_button = _documented_dg_download_button
DeltaGenerator.markdown = _dg_text_wrapper(_native_dg_markdown)
DeltaGenerator.caption = _dg_text_wrapper(_native_dg_caption)
DeltaGenerator.title = _dg_text_wrapper(_native_dg_title)
DeltaGenerator.header = _dg_text_wrapper(_native_dg_header)
DeltaGenerator.subheader = _dg_text_wrapper(_native_dg_subheader)
DeltaGenerator.info = _dg_text_wrapper(_native_dg_info)
DeltaGenerator.warning = _dg_text_wrapper(_native_dg_warning)
DeltaGenerator.error = _dg_text_wrapper(_native_dg_error)
DeltaGenerator.success = _dg_text_wrapper(_native_dg_success)
DeltaGenerator.metric = _localized_dg_metric
DeltaGenerator.tabs = _localized_dg_tabs
DeltaGenerator.expander = _localized_dg_expander
DeltaGenerator.popover = _localized_dg_popover
DeltaGenerator.spinner = _localized_dg_spinner
DeltaGenerator.dataframe = _localized_dg_dataframe
DeltaGenerator.data_editor = _localized_dg_data_editor
DeltaGenerator.plotly_chart = _localized_dg_plotly_chart
DeltaGenerator.write = _localized_dg_write


st.set_page_config(
    page_title=translate_text("三维激光放大器", active_locale()),
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1780px;
        padding-top: 1.35rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        min-width: min(480px, 100vw);
        max-width: min(480px, 100vw);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    [data-testid="stSidebarContent"] {
        padding-top: 0.65rem;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.16);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 0.72rem;
        margin-bottom: 0.45rem;
    }
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.34);
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 0.8rem;
        padding: 0.8rem 0.95rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.45rem;
        line-height: 1.25;
    }
    [data-testid="stSidebar"] [data-baseweb="tab-list"] {
        gap: 0.25rem;
        position: sticky;
        top: 0;
        z-index: 5;
        padding: 0.35rem 0 0.45rem 0;
        background: color-mix(in srgb, var(--background-color) 94%, transparent);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(148, 163, 184, 0.16);
    }
    [data-testid="stSidebar"] [data-baseweb="tab"] {
        min-height: 2.7rem;
        padding-left: 0.7rem;
        padding-right: 0.7rem;
        border-radius: 0.55rem 0.55rem 0 0;
        white-space: normal;
        line-height: 1.15;
    }
    [data-testid="stSidebar"] [data-baseweb="tab-list"] {
        flex-wrap: wrap;
    }
    [data-testid="stSidebar"] [data-baseweb="tab-highlight"] {
        display: none;
    }
    [data-testid="stSidebar"] [role="tab"][aria-selected="true"] {
        box-shadow: inset 0 -2px var(--primary-color, #ff4b4b);
    }
    [data-testid="stSidebar"] details {
        border-radius: 0.65rem;
    }
    .stPlotlyChart {
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 0.75rem;
        overflow: hidden;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
        min-height: 3rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    [data-testid="stSidebar"] hr {
        margin: 0.75rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    _language_before_render = active_locale()
    _language_widget_key = f"ui_locale_selector__{_language_before_render}"
    if _language_widget_key not in st.session_state:
        st.session_state[_language_widget_key] = _language_before_render
    _selected_language = _native_radio(
        lt("语言"),
        [ZH_CN, EN],
        key=_language_widget_key,
        horizontal=True,
        format_func=_localized_choice_format(
            lambda locale: "中文" if locale == ZH_CN else "English"
        ),
        help=lt(
            "切换整个界面的表单、说明、图表注释和结果显示；不会改变配置值或重新拼接旧结果。"
        ),
    )
    if _selected_language != _language_before_render:
        st.session_state["ui_locale"] = _selected_language
        st.rerun()


def initial_form_config() -> dict[str, Any]:
    """Keep the previous UI starting point while allowing JSON to replace it."""

    cfg = copy.deepcopy(DEFAULTS)
    # Keep the built-in demonstration self-consistent with its shared FFT
    # window.  A 150 mm source-to-waist distance with a 32 µm mid-IR waist
    # places most of the source field outside a 0.5 mm crystal window and made
    # the very first run fail its spatial-boundary contract.
    source_to_gain_m = 2e-3
    cfg["grid"].update({"size_x_m": 0.5e-3, "size_y_m": 0.5e-3, "length_z_m": 5e-3, "nx": 64, "ny": 64, "nz": 32})
    cfg["material"].update({"ion_density_m3": 7e24, "refractive_index": 2.2})
    cfg["material"]["geometry"].update({"shape": "cuboid", "width_x_m": 0.5e-3, "height_y_m": 0.5e-3})
    cfg["energy_levels"].update(
        {
            "pump_absorption_cross_section_m2": 1e-24,
            "pump_emission_cross_section_m2": 0.0,
            "signal_emission_cross_section_m2": 1e-22,
            "signal_absorption_cross_section_m2": 0.0,
            "upper_state_lifetime_s": 5e-6,
        }
    )
    cfg["pump"].update(
        {
            "power_W": 11.0,
            "wavelength_m": 1.55e-6,
            "waist_x_m": 60.5e-6,
            "waist_y_m": 28e-6,
            # In the source-propagation modes, focus_z_m is measured from the
            # PUMP LASER plane.  This places the waist at the crystal centre.
            "focus_z_m": source_to_gain_m + 2.5e-3,
            "m2_x": 1.5,
            "m2_y": 1.5,
            "order_x": 2.0,
            "order_y": 2.0,
        }
    )
    cfg["seed"].update(
        {
            "pulse_energy_J": 0.66 / 50e6,
            "wavelength_m": 2.4e-6,
            "pulse_duration_s": 35e-15,
            "waist_x_m": 32e-6,
            "waist_y_m": 32e-6,
            # Put the seed waist at the crystal entrance by default.
            "focus_z_m": source_to_gain_m,
        }
    )
    cfg["seed"]["pulse_definition"]["domain"] = "time"
    cfg["simulation"].update({"repetition_rate_Hz": 50e6, "passes": 1, "save_vtk": False})
    # The default page contains a real source-to-crystal pump path.  ABCD makes
    # that distance (and any lenses inserted in the pump editor) part of the
    # solver rather than retaining the prescribed-caustic shortcut.
    cfg["physics"]["pump_propagation"] = "abcd"
    cfg["optical_train"]["pump"] = [
        {"type": "gain_medium", "name": "pump_gain", "distance_before_m": source_to_gain_m}
    ]
    cfg["optical_train"]["seed"] = [
        {"type": "gain_medium", "name": "seed_gain", "distance_before_m": source_to_gain_m}
    ]
    return cfg


def clear_form_widget_state(*, hard_reset: bool = False) -> None:
    """Remove stale widget values before installing an imported config.

    Streamlit's browser-side widget state can repopulate a deleted ``ui_*``
    key when the widget keeps the same identity across a rerun.  A JSON import
    therefore uses a hard reset of all widget keys, while retaining an already
    parsed material library.  Material-library application only needs the
    lighter ``ui_*`` reset because its uploader and selection must stay live.
    """

    if hard_reset:
        retained = {
            key: value
            for key, value in st.session_state.items()
            if key in {
                "material_library_records",
                "material_library_digest",
                "ui_locale",
            }
            or key.startswith("ui_locale_selector__")
        }
        st.session_state.clear()
        for key, value in retained.items():
            st.session_state[key] = value
        return
    for key in list(st.session_state):
        if key.startswith("ui_") and not (
            key == "ui_locale" or key.startswith("ui_locale_selector__")
        ):
            del st.session_state[key]


def form_widget_key(key: str) -> str:
    """Version a form widget so an import cannot revive its old browser value."""

    if "__config_" in key:
        return key
    generation = int(st.session_state.get("form_widget_generation", 0))
    return f"{key}__config_{generation}"


def amplifier_view_widget_key(key: str) -> str:
    """Re-key transient controls whenever the simulation page is re-entered."""

    generation = int(
        st.session_state.get("amplifier_page_visit_generation", 0)
    )
    return key if generation <= 0 else f"{key}__view_{generation}"


def seed_material_form_widget_state(config: dict[str, Any]) -> None:
    """Force imported material values into the lower material/level widgets."""

    material = config["material"]
    levels = config["energy_levels"]
    thermal = material["thermal"]
    mechanical = material["mechanical"]
    conductivity = thermal.get("conductivity_W_mK", [10.0, 10.0, 10.0])
    conductivity_xyz = (
        [float(conductivity)] * 3
        if isinstance(conductivity, (int, float))
        else list(map(float, conductivity))
    )
    if len(conductivity_xyz) != 3:
        conductivity_xyz = [conductivity_xyz[0]] * 3
    widget_values: dict[str, Any] = {
        "ui_material_name": str(material.get("name", "user_material")),
        "ui_density": float(material["ion_density_m3"]),
        "ui_refractive_index": float(material["refractive_index"]),
        "ui_pump_passive": float(material.get("pump_passive_absorption_m_inv", 0.0)),
        "ui_signal_passive": float(material.get("signal_passive_loss_m_inv", 0.0)),
        "ui_sigma_pa": float(levels["pump_absorption_cross_section_m2"]),
        "ui_sigma_pe": float(levels["pump_emission_cross_section_m2"]),
        "ui_sigma_se": float(levels["signal_emission_cross_section_m2"]),
        "ui_sigma_sa": float(levels["signal_absorption_cross_section_m2"]),
        "ui_tau_upper": float(levels["upper_state_lifetime_s"]),
        "ui_tau_pump": float(levels["pump_level_lifetime_s"]),
        "ui_tau_lower": float(levels["lower_level_lifetime_s"]),
        "ui_pump_quantum_efficiency": float(levels.get("pump_quantum_efficiency", 1.0)),
        "ui_dndt": float(material.get("dn_dT_per_K", 0.0)),
        "ui_conductivity_x": conductivity_xyz[0],
        "ui_conductivity_y": conductivity_xyz[1],
        "ui_conductivity_z": conductivity_xyz[2],
        "ui_mass_density": float(thermal.get("density_kg_m3", 4000.0)),
        "ui_heat_capacity": float(thermal.get("heat_capacity_J_kgK", 500.0)),
        "ui_young_modulus": float(mechanical.get("young_modulus_Pa", 1e11)),
        "ui_poisson_ratio": float(mechanical.get("poisson_ratio", 0.25)),
        "ui_thermal_expansion": float(mechanical.get("thermal_expansion_per_K", 8e-6)),
        "ui_photoelastic_scalar": float(mechanical.get("photoelastic_coefficient_Pa_inv", 0.0)),
    }
    if material.get("pump_refractive_index") is not None:
        widget_values["ui_pump_refractive_index"] = float(material["pump_refractive_index"])
    for key, value in widget_values.items():
        st.session_state[form_widget_key(key)] = value


def install_form_config(
    supplied: dict[str, Any],
    *,
    update_json_editor: bool = True,
    hard_reset_widgets: bool = False,
) -> dict[str, Any]:
    merged = prepare_config(supplied)
    next_generation = int(st.session_state.get("form_widget_generation", 0)) + 1
    clear_form_widget_state(hard_reset=hard_reset_widgets)
    st.session_state["form_widget_generation"] = next_generation
    st.session_state["form_base_config"] = merged
    if update_json_editor:
        st.session_state["config_json"] = json.dumps(merged, ensure_ascii=False, indent=2)
    seed_mode = "seed_simple"
    seed_rows = merged.get("optical_train", {}).get("seed", [])
    st.session_state["path_rows_seed"] = copy.deepcopy(seed_rows)
    st.session_state["path_rows_pump"] = copy.deepcopy(merged.get("optical_train", {}).get("pump", []))
    st.session_state["path_mode_seed"] = seed_mode
    st.session_state["path_mode_pump"] = "pump"
    st.session_state["path_editor_version_seed"] = int(st.session_state.get("path_editor_version_seed", 0)) + 1
    st.session_state["path_editor_version_pump"] = int(st.session_state.get("path_editor_version_pump", 0)) + 1
    seed_material_form_widget_state(merged)
    return merged


def sci(label: str, value: float, *, key: str, help_text: str | None = None) -> float:
    resolved_key = form_widget_key(key)
    default = {} if resolved_key in st.session_state else {"value": float(value)}
    return float(st.number_input(label, format="%.6e", key=resolved_key, help=help_text, **default))


def widget_default(key: str, value: Any) -> dict[str, Any]:
    """Avoid Streamlit's value/session-state conflict during forced refill."""

    return {} if form_widget_key(key) in st.session_state else {"value": value}


def option_index(options: list[str], value: str, default: int = 0) -> int:
    """Find an exact option or an option whose machine-value prefix matches."""

    if value in options:
        return options.index(value)
    for index, option in enumerate(options):
        if option.split(" ")[0] == value:
            return index
    return default


def auto_boolean(
    label: str,
    value: Any,
    *,
    key: str,
    disabled: bool = False,
    help: str | None = None,
) -> bool | str:
    options = ["auto — 由配置默认值决定", "on — 开", "off — 关"]
    selected = "auto" if value == "auto" else "on" if bool(value) else "off"
    choice = choice_selectbox(
        label,
        options,
        key=key,
        disabled=disabled,
        help=help,
        **(
            {}
            if key in st.session_state
            else {"index": option_index(options, selected)}
        ),
    )
    machine = choice.split(" ")[0]
    return "auto" if machine == "auto" else machine == "on"


def phase_coefficients_ui(
    label: str,
    key_prefix: str,
    reference_wavelength_m: float,
    coefficients: dict[str, Any] | None = None,
) -> dict[str, float | None]:
    """Edit spectral phase in convenient fs units and return SI coefficients."""
    coefficients = coefficients or {}
    configured_reference_wavelength = coefficients.get(
        "reference_wavelength_m", reference_wavelength_m
    )
    st.markdown(f"**{label}**")
    first, second = st.columns(2)
    with first:
        gd_fs = st.number_input("GD (fs)", value=float(coefficients.get("gd_s", 0.0)) * 1e15, format="%.6g", key=form_widget_key(f"ui_{key_prefix}_gd"))
        gdd_fs2 = st.number_input("GDD (fs²)", value=float(coefficients.get("gdd_s2", 0.0)) * 1e30, format="%.6g", key=form_widget_key(f"ui_{key_prefix}_gdd"))
    with second:
        tod_fs3 = st.number_input("TOD (fs³)", value=float(coefficients.get("tod_s3", 0.0)) * 1e45, format="%.6g", key=form_widget_key(f"ui_{key_prefix}_tod"))
        fod_fs4 = st.number_input("FOD (fs⁴)", value=float(coefficients.get("fod_s4", 0.0)) * 1e60, format="%.6g", key=form_widget_key(f"ui_{key_prefix}_fod"))
    return {
        # ``None`` intentionally means "follow the Seed centre wavelength".
        # Opening the form must not freeze that dynamic reference, nor replace
        # an explicitly configured reference with the current centre value.
        "reference_wavelength_m": (
            None
            if configured_reference_wavelength is None
            else float(configured_reference_wavelength)
        ),
        "constant_phase_rad": float(coefficients.get("constant_phase_rad", 0.0)),
        "gd_s": float(gd_fs) * 1e-15,
        "gdd_s2": float(gdd_fs2) * 1e-30,
        "tod_s3": float(tod_fs3) * 1e-45,
        "fod_s4": float(fod_fs4) * 1e-60,
    }


def read_rows(uploaded: Any, minimum_columns: int = 2) -> list[list[float]]:
    if uploaded is None:
        return []
    payload = io.BytesIO(uploaded.getvalue())
    array = np.genfromtxt(payload, delimiter=",", comments="#")
    if array.ndim != 2 or array.shape[1] < minimum_columns or np.all(~np.isfinite(array)):
        payload.seek(0)
        array = np.genfromtxt(payload, comments="#")
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] < minimum_columns:
        raise ValueError(f"文件至少需要 {minimum_columns} 列数值")
    array = array[np.all(np.isfinite(array[:, :minimum_columns]), axis=1)]
    if len(array) < 2:
        raise ValueError("文件至少需要两行有效数值")
    return array.tolist()


def _render_seed_time_intensity_input(seed: dict[str, Any]) -> None:
    time_options = [
        "gaussian",
        "sech2",
        "rectangular",
        "super_gaussian",
        "file",
        "cw",
    ]
    time_type = choice_selectbox(
        "时间强度来源",
        time_options,
        index=option_index(time_options, str(seed["temporal"].get("type", "gaussian"))),
        key=form_widget_key("ui_seed_time_type"),
    )
    seed["temporal"]["type"] = time_type
    if time_type == "cw":
        seed["temporal"]["resolved"] = False
        return
    seed["pulse_duration_s"] = sci(
        "时间强度 FWHM (s)",
        seed["pulse_duration_s"],
        key=form_widget_key("ui_seed_duration"),
        help_text=(
            "这是强度包络的全宽半高。选择 super_gaussian 时，改变阶数只改变"
            "肩部和边缘形状；求解器会重新定标包络，使这里输入的 FWHM 对每个阶数都严格保持不变。"
        ),
    )
    if time_type == "super_gaussian":
        seed["temporal"]["order"] = st.number_input(
            "时间超高斯阶数",
            min_value=1.0,
            value=float(seed["temporal"].get("order", 2.0)),
            key=form_widget_key("ui_seed_time_order"),
            help=(
                "强度使用 exp[-ln(2)·|2t/FWHM|^(2·order)]；order=1 为 Gaussian，"
                "任意 order 下半高点都固定在 ±FWHM/2。"
            ),
        )
    if time_type == "file":
        uploaded = st.file_uploader(
            "时间强度 CSV：t(fs), intensity [, phase(rad)]",
            type=["csv", "txt"],
            key=form_widget_key("seed_wave"),
            help="intensity 会自动归一化到单脉冲总能量；可选第三列是时间相位（rad）。",
        )
        if uploaded is not None:
            seed["temporal"]["samples"] = read_rows(uploaded)
            seed["temporal"]["time_unit"] = "fs"
    seed["temporal"]["resolved"] = True


def _render_seed_spectrum_intensity_input(seed: dict[str, Any], *, allow_monochromatic: bool = False) -> None:
    spectrum_options = ["gaussian", "lorentzian", "super_gaussian", "file"]
    if allow_monochromatic:
        spectrum_options.insert(0, "monochromatic")
    configured_spectrum_type = str(seed["spectrum"].get("type", "gaussian"))
    spectrum_type = choice_selectbox(
        "频率/光谱强度来源",
        spectrum_options,
        index=option_index(spectrum_options, configured_spectrum_type),
        key=form_widget_key("ui_seed_spectrum_type"),
    )
    seed["spectrum"]["type"] = spectrum_type
    if spectrum_type == "monochromatic":
        return
    density_domains = ["wavelength", "frequency"]
    seed["spectrum"]["spectral_density_domain"] = choice_selectbox(
        "光谱密度自变量",
        density_domains,
        index=option_index(
            density_domains,
            str(seed["spectrum"].get("spectral_density_domain", "wavelength")),
        ),
        key=form_widget_key("ui_seed_spectral_density_domain"),
        help="选择纵轴是每单位波长还是每单位频率的光谱密度；坐标变换时会保持积分能量。",
    )
    if seed["spectrum"]["spectral_density_domain"] == "wavelength":
        seed["spectrum"]["bandwidth_fwhm_Hz"] = None
    if spectrum_type == "file":
        value_modes = [
            "relative — 自动归一化",
            "energy_per_bin_J — 每个波长分箱能量 (J/bin)",
            "energy_density_J_per_nm — 光谱能量密度 (J/nm)",
        ]
        value_mode_label = choice_selectbox(
            "光谱文件纵轴",
            value_modes,
            index=option_index(value_modes, str(seed["spectrum"].get("value_mode", "relative"))),
            key=form_widget_key("ui_seed_spectrum_value_mode"),
        )
        seed["spectrum"]["value_mode"] = value_mode_label.split(" ")[0]
        uploaded = st.file_uploader(
            "光谱强度 CSV：λ(nm), intensity/energy [, phase(rad)]",
            type=["csv", "txt"],
            key=form_widget_key("seed_spectrum"),
            help="relative 模式自动归一化；可选第三列是光谱相位（rad）。",
        )
        if uploaded is not None:
            seed["spectrum"]["samples"] = read_rows(uploaded)
            seed["spectrum"]["wavelength_unit"] = "nm"
    else:
        seed["spectrum"]["value_mode"] = "relative"
        default_spectrum_sample_count = (
            21
            if configured_spectrum_type == "monochromatic"
            else max(3, int(seed["spectrum"].get("sample_count", 21)))
        )
        sampling_columns = st.columns(2)
        with sampling_columns[0]:
            seed["spectrum"]["sample_count"] = int(
                st.number_input(
                    "解析 Seed 源光谱采样数",
                    min_value=3,
                    max_value=4096,
                    value=default_spectrum_sample_count,
                    key=form_widget_key("ui_seed_source_spectral_samples"),
                    help=(
                        "用于离散解析 Gaussian/Lorentzian/super-Gaussian 源光谱；"
                        "它与后面的输运求解重分箱、Commercial App 回灌分箱和 coherent CPA "
                        "FFT 采样分别控制不同阶段。"
                    ),
                )
            )
        with sampling_columns[1]:
            seed["spectrum"]["span_fwhm"] = st.number_input(
                "解析 Seed 源光谱半跨度 (×FWHM)",
                min_value=1.0,
                value=float(seed["spectrum"].get("span_fwhm", 3.0)),
                key=form_widget_key("ui_seed_source_spectral_span"),
                help=(
                    "在中心频率或中心波长两侧各覆盖多少个 FWHM；增加跨度时也要"
                    "同步增加采样数，避免固定分箱被摊薄。"
                ),
            )
        if seed["spectrum"]["spectral_density_domain"] == "frequency":
            bandwidth_units = ["wavelength_equivalent_m", "frequency_Hz"]
            current_unit = (
                "frequency_Hz"
                if float(seed["spectrum"].get("bandwidth_fwhm_Hz") or 0.0) > 0.0
                else "wavelength_equivalent_m"
            )
            bandwidth_unit = choice_selectbox(
                "频率密度 FWHM 输入",
                bandwidth_units,
                index=option_index(bandwidth_units, current_unit),
                key=form_widget_key("ui_seed_bw_unit"),
                help=(
                    "Hz 输入最无歧义。m 输入保持峰值波长，并保持变换回波长坐标后的总 FWHM；"
                    "两个半高点不会强制关于峰值波长对称。"
                ),
            )
        else:
            bandwidth_unit = "wavelength_equivalent_m"
        if bandwidth_unit == "frequency_Hz":
            seed["spectrum"]["bandwidth_fwhm_m"] = 0.0
            seed["spectrum"]["bandwidth_fwhm_Hz"] = sci(
                "光谱强度 FWHM (Hz)",
                seed["spectrum"].get("bandwidth_fwhm_Hz") or 0.0,
                key=form_widget_key("ui_seed_bw_Hz"),
            )
        else:
            seed["spectrum"]["bandwidth_fwhm_Hz"] = None
            seed["spectrum"]["bandwidth_fwhm_m"] = sci(
                "光谱强度 FWHM (m)",
                seed["spectrum"].get("bandwidth_fwhm_m", 0.0),
                key=form_widget_key("ui_seed_bw"),
            )
        if spectrum_type == "super_gaussian":
            seed["spectrum"]["order"] = st.number_input(
                "光谱超高斯阶数",
                min_value=1.0,
                value=float(seed["spectrum"].get("order", 2.0)),
                key=form_widget_key("ui_seed_spectrum_order"),
                help=(
                    "所选光谱密度自变量上的强度使用严格 FWHM 定标；改变 order 不会改变"
                    "上方输入的半高全宽，只会改变谱顶和谱翼形状。"
                ),
            )


def render_seed_pulse_definition(cfg: dict[str, Any]) -> None:
    """Edit normalized Seed intensity constraints, phase, and Fourier linkage."""

    seed = cfg["seed"]
    definition = seed.setdefault(
        "pulse_definition",
        {"domain": "time", "fft_sample_count": 256, "solver_sample_count": 9},
    )
    coherent = seed.setdefault("coherent", {})
    st.markdown("#### Seed 脉冲列、强度文件与相位")
    domain_options = [
        "time — 只给时间强度，傅里叶变换生成光谱",
        "spectrum — 只给光谱强度，傅里叶变换生成时间",
        "dual — 同时给时间与光谱强度，自动恢复相位",
    ]
    configured_domain = str(definition.get("domain", "time"))
    if configured_domain == "independent":
        # Keep the compatibility backend visible only when it is already in
        # the imported configuration.  It is valid for models that consume
        # independent temporal and spectral marginals, but unlike ``dual`` it
        # does not claim that the two curves form one Fourier-consistent field.
        domain_options.append(
            "independent — 兼容：时间与光谱参数分别定义（不强制傅里叶闭合）"
        )
    domain_label = choice_radio(
        "强度输入方式",
        domain_options,
        index=option_index(domain_options, configured_domain),
        horizontal=False,
        key=form_widget_key("ui_seed_definition_domain"),
        help="所有强度曲线都会归一化到同一个单脉冲能量。单域缺相位时采用零相位；双域缺相位时迭代恢复一个可行解。",
    )
    definition["domain"] = domain_label.split(" ")[0]
    cfg["simulation"]["repetition_rate_Hz"] = sci(
        "Seed 重复频率 (Hz；0=单次)",
        cfg["simulation"]["repetition_rate_Hz"],
        key=form_widget_key("ui_seed_rep_rate"),
        help_text="这是 Pump/Seed 周期求解共享的系统时钟；每个周期包含一个 Seed 脉冲。",
    )
    seed["wavelength_m"] = sci("中心波长 (m)", seed["wavelength_m"], key=form_widget_key("ui_seed_wavelength"))
    st.caption("相位约定：E(t) 使用 exp(−iωt)，正谱相位斜率表示延迟；导入相位应采用同一约定。")

    if definition["domain"] == "independent":
        st.info(
            "兼容模式分别保存时间强度和光谱强度；它们不是同一个复场的两条傅里叶约束。"
            "相干 CPA 的 coefficients 输入以光谱和相位为准，并报告实际反变换脉宽；"
            "非相干时间分箱仍使用时间强度。需要单一闭合复场时请选择 time、spectrum 或 dual。"
        )
        time_column, spectrum_column = st.columns(2)
        with time_column:
            st.markdown("**独立时间强度**")
            _render_seed_time_intensity_input(seed)
        with spectrum_column:
            st.markdown("**独立光谱强度**")
            _render_seed_spectrum_intensity_input(seed, allow_monochromatic=True)
    elif definition["domain"] == "dual":
        st.info("双强度模式不会假设光谱是Gaussian；若没有相位列，程序会寻找一个同时逼近两条归一化强度曲线的复场。解不唯一。")
        time_column, spectrum_column = st.columns(2)
        with time_column:
            st.markdown("**时间强度约束**")
            _render_seed_time_intensity_input(seed)
        with spectrum_column:
            st.markdown("**光谱强度约束**")
            _render_seed_spectrum_intensity_input(seed)
    elif definition["domain"] == "time":
        _render_seed_time_intensity_input(seed)
        st.caption("光谱由时间复场作傅里叶变换生成；两列时间文件采用零时间相位，第三列可直接给 phase(rad)。")
    else:
        _render_seed_spectrum_intensity_input(seed)
        st.caption("时间波形由复光谱作傅里叶变换生成；两列光谱文件采用平坦光谱相位，第三列可直接给 phase(rad)。")

    absolute_file_energy = (
        spectral_file_pulse_energy_J(seed["spectrum"])
        if definition["domain"] in {"independent", "spectrum", "dual"}
        and seed["spectrum"].get("samples")
        else None
    )
    if absolute_file_energy is None:
        seed["pulse_energy_J"] = sci("单脉冲总能量 (J)", seed["pulse_energy_J"], key=form_widget_key("ui_seed_energy"))
    else:
        seed["pulse_energy_J"] = absolute_file_energy
        st.session_state[form_widget_key("ui_seed_energy")] = absolute_file_energy
        energy_value, energy_unit, _ = scaled_energy(absolute_file_energy)
        st.metric("文件积分得到的单脉冲能量", f"{energy_value:.5g} {energy_unit}")

    repetition_rate = float(cfg["simulation"]["repetition_rate_Hz"])
    input_average_power = float(seed["pulse_energy_J"]) * repetition_rate if repetition_rate > 0.0 else None
    input_metrics = st.columns(2)
    energy_value, energy_unit, _ = scaled_energy(float(seed["pulse_energy_J"]))
    input_metrics[0].metric("Seed 单脉冲能量", f"{energy_value:.5g} {energy_unit}")
    input_metrics[1].metric("Seed 输入平均功率", format_power(input_average_power))

    with st.expander("相位、GDD/TOD与双强度恢复", expanded=definition["domain"] == "dual"):
        st.caption(
            "强度文件没有相位时：单域采用零相位，双域自动恢复。下面的源相位与展宽器相位会在重建后相加；"
            "若双域曲线已经测于同一平面，通常先把附加GDD/TOD设为0。"
        )
        phase_file_cfg = coherent.setdefault(
            "spectral_phase", {"enabled": False, "wavelength_unit": "nm", "samples": []}
        )
        enable_phase_file = choice_checkbox(
            "使用独立光谱相位文件",
            value=bool(phase_file_cfg.get("enabled", False)),
            key=form_widget_key("ui_seed_spectral_phase_enabled"),
            help="也可以直接把phase(rad)放在时间/光谱强度文件第三列。独立相位文件用于只有两列强度文件的情况。",
        )
        if enable_phase_file:
            uploaded_phase = st.file_uploader(
                "光谱相位 CSV：λ(nm), phase(rad)",
                type=["csv", "txt"],
                key=form_widget_key("seed_spectral_phase"),
            )
            if uploaded_phase is not None:
                phase_file_cfg["samples"] = read_rows(uploaded_phase, 2)
                phase_file_cfg["wavelength_unit"] = "nm"
            if not phase_file_cfg.get("samples"):
                st.warning("已打开独立相位文件，但尚未上传有效数据；本次仍按无独立相位处理。")
        phase_file_cfg["enabled"] = bool(enable_phase_file and phase_file_cfg.get("samples"))

        source_tab, stretcher_tab = st.tabs(["输入场附加相位", "ZnSe / 展宽器相位"])
        with source_tab:
            coherent["phase"] = phase_coefficients_ui(
                "源光谱相位系数", "source_phase", seed["wavelength_m"], coherent.get("phase")
            )
        with stretcher_tab:
            coherent["stretcher"] = phase_coefficients_ui(
                "展宽器相位系数", "stretcher", seed["wavelength_m"], coherent.get("stretcher")
            )
        if definition["domain"] == "dual":
            retrieval = coherent.setdefault("phase_retrieval", {})
            retrieval["max_iterations"] = int(
                st.number_input(
                    "双强度相位恢复最大迭代",
                    min_value=1,
                    max_value=5000,
                    value=int(retrieval.get("max_iterations", 200)),
                    key=form_widget_key("ui_phase_retrieval_iterations"),
                )
            )
            retrieval["tolerance"] = sci(
                "双强度相位恢复容差",
                retrieval.get("tolerance", 1e-8),
                key=form_widget_key("ui_phase_retrieval_tolerance"),
            )

    with st.expander("Fourier 采样与求解分箱", expanded=False):
        definition["fft_sample_count"] = int(
            st.number_input(
                "实时 FFT 采样",
                min_value=32,
                max_value=4096,
                value=int(definition.get("fft_sample_count", 256)),
                step=32,
                key=form_widget_key("ui_seed_fft_samples"),
            )
        )
        legacy_solver_samples = int(definition.get("solver_sample_count", 9))
        had_solver_time_samples = "solver_time_sample_count" in definition
        had_solver_spectral_samples = "solver_spectral_sample_count" in definition
        solver_columns = st.columns(2)
        with solver_columns[0]:
            solver_time_sample_count = int(
                st.number_input(
                    "时间求解分箱",
                    min_value=1,
                    max_value=4096,
                    value=int(definition.get("solver_time_sample_count", legacy_solver_samples)),
                    key=form_widget_key("ui_seed_solver_time_samples"),
                    help="front → peak → tail 的有序能量分箱；JOSA retarded_pc 使用这里的时间分辨率。",
                )
            )
        with solver_columns[1]:
            solver_spectral_sample_count = int(
                st.number_input(
                    "光谱求解分箱",
                    min_value=1,
                    max_value=201,
                    value=int(definition.get("solver_spectral_sample_count", legacy_solver_samples)),
                    key=form_widget_key("ui_seed_solver_spectral_samples"),
                    help="波长相关截面和增益窄化分箱；相干 CPA 使用自己的频率采样。",
                )
            )
        if had_solver_time_samples or solver_time_sample_count != legacy_solver_samples:
            definition["solver_time_sample_count"] = solver_time_sample_count
        else:
            definition.pop("solver_time_sample_count", None)
        if (
            had_solver_spectral_samples
            or solver_spectral_sample_count != legacy_solver_samples
        ):
            definition["solver_spectral_sample_count"] = solver_spectral_sample_count
        else:
            definition.pop("solver_spectral_sample_count", None)
        st.caption(
            "多色/输运计算量近似与 时间分箱 × 光谱分箱 × Nx × Ny × Nz 成正比；"
            "旧 JSON 的 solver_sample_count 会继续同时作为两个分箱的回退值。"
        )

    if definition["domain"] == "independent":
        st.caption(
            "独立时间/光谱边缘分布没有唯一的 Fourier 闭合场，因此这里不绘制伪造的"
            "时频互推预览；相干求解结果会报告由实际复光谱得到的脉宽和采样合同。"
        )
        return

    try:
        pair = build_fourier_pulse_pair(seed, include_additional_phase=True)
    except Exception as exc:
        st.warning(f"Fourier 实时推导暂不可用：{type(exc).__name__}: {exc}")
        return

    pulse_energy_J = float(seed["pulse_energy_J"])
    temporal_power = pair.temporal_weights * pulse_energy_J / pair.dt_s
    coherent_cfg = seed.get("coherent", {})
    added_dispersion_phase = phase_from_coefficients(
        pair.frequency_Hz,
        coherent_cfg.get("phase", {}),
    )
    added_dispersion_phase += phase_from_coefficients(
        pair.frequency_Hz,
        coherent_cfg.get("stretcher", {}),
    )
    before_dispersion_spectrum = pair.spectral_field * np.exp(
        -1j * added_dispersion_phase
    )
    before_dispersion_time = np.fft.fftshift(
        np.fft.fft(
            np.fft.ifftshift(before_dispersion_spectrum),
            norm="ortho",
        )
    )
    before_dispersion_power = (
        np.abs(before_dispersion_time) ** 2 * pulse_energy_J / pair.dt_s
    )
    before_dispersion_display = before_dispersion_power / max(
        float(np.max(before_dispersion_power)),
        np.finfo(float).tiny,
    )
    temporal_display = temporal_power / max(
        float(np.max(temporal_power)),
        np.finfo(float).tiny,
    )
    spectrum_report = spectral_density_from_bin_energy(
        pair.wavelength_m,
        pair.spectral_weights * pulse_energy_J,
        bin_coordinate="frequency",
    )
    wavelength_order = np.argsort(pair.wavelength_m)
    wavelength_nm = spectrum_report["wavelength_m"] * 1e9
    spectral_density_J_per_nm = spectrum_report["density_J_per_nm"]
    active_wavelength_m, _, _ = trim_negligible_mass_support(
        pair.wavelength_m, pair.spectral_weights
    )
    active_wavelength_range_nm = [
        float(np.min(active_wavelength_m)) * 1e9,
        float(np.max(active_wavelength_m)) * 1e9,
    ]
    before_dispersion_fwhm = pulse_intensity_fwhm(
        pair.time_s,
        before_dispersion_power,
    )
    time_fwhm = pulse_intensity_fwhm(pair.time_s, temporal_power)
    spectrum_fwhm = pulse_intensity_fwhm(
        wavelength_nm,
        spectral_density_J_per_nm,
    )
    preview_metrics = st.columns(2)
    preview_metrics[0].metric(
        "附加色散前 时间 FWHM",
        f"{before_dispersion_fwhm * 1e15:.5g} fs",
    )
    preview_metrics[1].metric(
        "GDD/TOD/FOD 后 时间 FWHM",
        f"{time_fwhm * 1e15:.5g} fs",
    )
    if before_dispersion_fwhm > 0.0:
        st.caption(
            f"当前强度展宽倍率 {time_fwhm / before_dispersion_fwhm:.5g}×; "
            f"频谱强度 FWHM {spectrum_fwhm:.5g} nm（纯相位色散前后不变）。"
        )
    st.caption(
        "时域图中的两条强度曲线分别按各自峰值归一化，用于直接比较波形和展宽；"
        "归一化不改变上面的 FWHM。"
    )
    retrieval_diagnostics = pair.diagnostics.get("phase_retrieval")
    if retrieval_diagnostics is not None:
        status = "已达到容差" if retrieval_diagnostics.get("converged") else "达到最大迭代"
        st.caption(
            f"双强度相位恢复：{status}；迭代 {retrieval_diagnostics.get('iterations')}；"
            f"时间振幅相对误差 {retrieval_diagnostics.get('temporal_relative_amplitude_error', float('nan')):.3g}。"
        )

    time_figure = go.Figure()
    time_figure.add_scatter(
        x=pair.time_s * 1e15,
        y=before_dispersion_display,
        mode="lines",
        line={"dash": "dash"},
        name="附加色散前",
    )
    time_figure.add_scatter(
        x=pair.time_s * 1e15,
        y=temporal_display,
        mode="lines",
        name="GDD/TOD/FOD 后",
    )
    if pair.temporal_target_weights is not None:
        target_temporal_power = (
            pair.temporal_target_weights * pulse_energy_J / pair.dt_s
        )
        if not np.allclose(
            target_temporal_power,
            temporal_power,
            rtol=1e-5,
            atol=0.0,
        ) and not np.allclose(
            target_temporal_power,
            before_dispersion_power,
            rtol=1e-5,
            atol=0.0,
        ):
            target_temporal_display = target_temporal_power / max(
                float(np.max(target_temporal_power)),
                np.finfo(float).tiny,
            )
            time_figure.add_scatter(
                x=pair.time_s * 1e15,
                y=target_temporal_display,
                mode="lines",
                line={"dash": "dash"},
                name="输入强度目标",
            )
    time_figure.update_layout(
        title="时间域归一化强度（附加色散前/后实时对照）",
        xaxis_title="时间 (fs)",
        yaxis_title="归一化强度（各自峰值 = 1）",
        yaxis_range=[0.0, 1.05],
        height=245,
        margin=dict(t=42, l=52, r=12, b=46),
        showlegend=True,
    )
    st.plotly_chart(time_figure, width="stretch", key=f"seed_time_preview_{definition['domain']}")

    spectrum_figure = go.Figure()
    spectrum_figure.add_scatter(
        x=wavelength_nm,
        y=spectral_density_J_per_nm,
        mode="lines",
        name="波长谱密度（色散前后相同）",
    )
    if pair.spectral_target_weights is not None:
        target_spectrum_report = spectral_density_from_bin_energy(
            pair.wavelength_m,
            pair.spectral_target_weights * pulse_energy_J,
            bin_coordinate="frequency",
        )
        target_spectral_density = target_spectrum_report["density_J_per_nm"]
        if not np.allclose(
            target_spectral_density,
            spectral_density_J_per_nm,
            rtol=1e-5,
            atol=0.0,
        ):
            spectrum_figure.add_scatter(
                x=wavelength_nm,
                y=target_spectral_density,
                mode="lines",
                line={"dash": "dash"},
                name="输入波长谱密度目标",
            )
    spectrum_figure.update_layout(
        title="波长谱密度（纯 GDD/TOD/FOD 不改变谱能量密度）",
        xaxis_title="波长 (nm)",
        yaxis_title="波长谱能量密度 (J/nm)",
        height=245,
        margin=dict(t=42, l=52, r=12, b=46),
        showlegend=True,
        xaxis_range=active_wavelength_range_nm,
    )
    st.plotly_chart(spectrum_figure, width="stretch", key=f"seed_spectrum_preview_{definition['domain']}")
    with st.expander("重建/输入光谱相位预览与下载", expanded=False):
        unwrapped_phase = np.unwrap(np.angle(pair.spectral_field))[wavelength_order]
        phase_figure = go.Figure()
        phase_figure.add_scatter(x=wavelength_nm, y=unwrapped_phase, mode="lines", name="phase")
        phase_figure.update_layout(
            xaxis_title="波长 (nm)",
            yaxis_title="光谱相位 (rad)",
            height=245,
            margin=dict(t=20, l=52, r=12, b=46),
            showlegend=False,
            xaxis_range=active_wavelength_range_nm,
        )
        st.plotly_chart(phase_figure, width="stretch", key=f"seed_phase_preview_{definition['domain']}")
        phase_rows = "wavelength_nm,phase_rad\n" + "\n".join(
            f"{wavelength:.12g},{phase:.12g}"
            for wavelength, phase in zip(wavelength_nm, unwrapped_phase, strict=True)
        )
        st.download_button(
            "下载当前光谱相位 CSV",
            data=phase_rows.encode("utf-8"),
            file_name="seed_spectral_phase.csv",
            mime="text/csv",
            key=form_widget_key("download_seed_spectral_phase"),
        )
    for warning in pair.warnings:
        st.caption(localize_source_message(warning, active_locale()))


def render_seed_transport_controls(cfg: dict[str, Any]) -> None:
    """Expose temporal and spatial gain closures in one explicit module."""

    resolved_model = resolve_signal_model(cfg)
    with st.expander(
        "B. Seed 增益输运（时间模型 + 空间闭合）",
        expanded=resolved_model == "josa_transport"
        or str(
            cfg["seed"].get("spatial_gain_transport", {}).get(
                "model", "local_field"
            )
        )
        != "local_field",
    ):
        spatial_transport = cfg["seed"].setdefault(
            "spatial_gain_transport",
            {"model": "local_field", "record_axial_diagnostics": True},
        )
        spatial_options = [
            "local_field — 局域复场增益（默认，可产生增益导引）",
            "gaussian_mode_overlap — 高斯模态与 3D 反转重叠（无增益导引）",
            "parabolic_gain_fit — Commercial App 式局域抛物线增益拟合（复场 BPM）",
        ]
        spatial_label = choice_selectbox(
            "Seed 空间增益闭合",
            spatial_options,
            index=option_index(
                spatial_options,
                str(spatial_transport.get("model", "local_field")),
            ),
            key=form_widget_key("ui_seed_spatial_gain_transport"),
            help=(
                "local_field 对每个 x/y 单元施加自己的增益，会让光束被增益分布重塑；"
                "gaussian_mode_overlap 每个 z 单元只用强度加权的标量增益放大整束，"
                "用于明确关闭 gain guiding 的受约束 Gaussian 模态比较；"
                "parabolic_gain_fit 在指定椭圆内逐层拟合二次增益导管，复现 Commercial App Gauss-BPM 的空间闭合。"
            ),
            disabled=resolved_model not in {"fluence", "commercial_app_fast"},
        )
        spatial_transport["model"] = spatial_label.split(" ")[0]
        if spatial_transport["model"] == "gaussian_mode_overlap":
            spatial_transport["record_axial_diagnostics"] = choice_checkbox(
                "保存每个 z 单元的模态重叠增益与功率",
                value=bool(spatial_transport.get("record_axial_diagnostics", True)),
                key=form_widget_key("ui_seed_modal_gain_axial_diagnostics"),
            )
            if resolved_model == "commercial_app_fast":
                st.info(
                    "Commercial App 中每个 Gaussian-q/HG 通道按三维反转重叠取得标量增益，"
                    "并可通过共享局部布居提交受激耗尽；增益不会把单个通道重塑成任意横向场。"
                )
            else:
                st.info(
                    "该模式仍传播衍射、走离、孔径和完整三维热相位，但增益只改变总功率，"
                    "不改变横向模态。fluence 路径仅支持单色且必须关闭空间 FN 饱和。"
                )
            if (
                resolved_model != "commercial_app_fast"
                and bool(resolve_physics(cfg).get("gain_saturation", False))
            ):
                st.error("高斯模态重叠与空间 FN 耗尽尚未定义；请关闭增益饱和或改回 local_field。")
        elif spatial_transport["model"] == "parabolic_gain_fit":
            default_fit_radius_x = (
                spatial_transport.get("fit_radius_x_m")
                or cfg["seed"].get("radius_x_m")
                or cfg["seed"].get("waist_x_m")
                or 1.0e-4
            )
            default_fit_radius_y = (
                spatial_transport.get("fit_radius_y_m")
                or cfg["seed"].get("radius_y_m")
                or cfg["seed"].get("waist_y_m")
                or 1.0e-4
            )
            fit_columns = st.columns(2)
            with fit_columns[0]:
                spatial_transport["fit_radius_x_m"] = sci(
                    "增益拟合半径 x (m)",
                    default_fit_radius_x,
                    key=form_widget_key("ui_seed_gain_fit_radius_x"),
                )
            with fit_columns[1]:
                spatial_transport["fit_radius_y_m"] = sci(
                    "增益拟合半径 y (m)",
                    default_fit_radius_y,
                    key=form_widget_key("ui_seed_gain_fit_radius_y"),
                )
            center_columns = st.columns(2)
            with center_columns[0]:
                spatial_transport["center_x_m"] = sci(
                    "增益拟合中心 x (m)",
                    spatial_transport.get("center_x_m", 0.0),
                    key=form_widget_key("ui_seed_gain_fit_center_x"),
                )
            with center_columns[1]:
                spatial_transport["center_y_m"] = sci(
                    "增益拟合中心 y (m)",
                    spatial_transport.get("center_y_m", 0.0),
                    key=form_widget_key("ui_seed_gain_fit_center_y"),
                )
            spatial_transport["minimum_gain_m_inv"] = sci(
                "拟合增益下限 (m⁻¹)",
                spatial_transport.get("minimum_gain_m_inv", 0.0),
                key=form_widget_key("ui_seed_gain_fit_minimum"),
                help_text=(
                    "只在拟合椭圆内钳制二次增益；椭圆外仍使用原始局部增益。"
                    "Nd:YAG 四能级主动增益通常取 0；存在真实再吸收时可保留负值。"
                ),
            )
            spatial_transport["record_axial_diagnostics"] = choice_checkbox(
                "保存每个 z 单元的抛物线增益拟合诊断",
                value=bool(spatial_transport.get("record_axial_diagnostics", True)),
                key=form_widget_key("ui_seed_parabolic_gain_axial_diagnostics"),
                help="只保存逐层二次拟合系数和残差诊断，不改变传播场或任何物理输入。",
            )
            st.info(
                "拟合只改变 BPM 使用的空间增益表示；泵浦、能级布居、截面、寿命和输入光束参数保持原值。"
            )
        elif resolved_model not in {"fluence", "commercial_app_fast"}:
            st.caption("多色/FN、JOSA 与 coherent CPA 使用各自的局域空间增益算子。")

        if resolved_model in {"spectral_time", "josa_transport", "coherent_cpa"}:
            st.divider()
            st.markdown("##### Seed 热相位空间闭合")
            sampled_thermal_lens = cfg["seed"].setdefault(
                "thermal_lens",
                copy.deepcopy(DEFAULTS["seed"]["thermal_lens"]),
            )
            thermal_lens_options = ["sampled_phase", "local_parabolic"]
            current_thermal_lens = str(
                sampled_thermal_lens.get("model", "sampled_phase")
            )
            sampled_thermal_lens["model"] = choice_selectbox(
                "sampled-field 热相位闭合",
                thermal_lens_options,
                index=option_index(
                    thermal_lens_options, current_thermal_lens
                ),
                key=form_widget_key("ui_seed_sampled_thermal_lens_model"),
                disabled=str(resolve_physics(cfg)["optical_propagation"]) == "off",
                format_func=lambda value: {
                    "sampled_phase": "sampled_phase — 保留采样热相位（默认）",
                    "local_parabolic": "local_parabolic — 局部抛物线拟合窗口",
                }[value],
                help=(
                    "sampled_phase 直接使用网格热相位；local_parabolic 在每个 z 层的"
                    "指定椭圆邻域拟合二次热透镜，再把拟合后的相位用于 sampled-field "
                    "Seed 传播。Commercial App Fast 使用后端专属热透镜设置。"
                ),
            )
            sampled_thermal_lens["fit_radius_mode"] = "manual"
            if sampled_thermal_lens["model"] == "local_parabolic":
                quadratic_options = ["axis_aligned", "full_xy"]
                sampled_thermal_lens["quadratic_model"] = choice_selectbox(
                    "局部热透镜二次模型",
                    quadratic_options,
                    index=option_index(
                        quadratic_options,
                        str(
                            sampled_thermal_lens.get(
                                "quadratic_model", "axis_aligned"
                            )
                        ),
                    ),
                    key=form_widget_key(
                        "ui_seed_sampled_thermal_lens_quadratic_model"
                    ),
                    format_func=lambda value: {
                        "axis_aligned": "axis_aligned — x/y 像散二次项",
                        "full_xy": "full_xy — 包含 xy 交叉二次项",
                    }[value],
                    help=(
                        "axis_aligned 舍弃旋转像散的 xy 交叉曲率；full_xy 保留完整"
                        "二维二次型。若交叉项不可忽略，不应使用轴对齐降阶。"
                    ),
                )
                radius_columns = st.columns(2)
                with radius_columns[0]:
                    sampled_thermal_lens["radius_x_m"] = sci(
                        "sampled-field 热拟合半径 x (m)",
                        sampled_thermal_lens.get("radius_x_m")
                        or 2.0 * float(cfg["seed"]["waist_x_m"]),
                        key=form_widget_key(
                            "ui_seed_sampled_thermal_lens_radius_x"
                        ),
                    )
                with radius_columns[1]:
                    sampled_thermal_lens["radius_y_m"] = sci(
                        "sampled-field 热拟合半径 y (m)",
                        sampled_thermal_lens.get("radius_y_m")
                        or 2.0 * float(cfg["seed"]["waist_y_m"]),
                        key=form_widget_key(
                            "ui_seed_sampled_thermal_lens_radius_y"
                        ),
                    )
                manual_center = choice_checkbox(
                    "手动指定 sampled-field 热拟合中心",
                    value=(
                        sampled_thermal_lens.get("center_x_m") is not None
                        or sampled_thermal_lens.get("center_y_m") is not None
                    ),
                    key=form_widget_key(
                        "ui_seed_sampled_thermal_lens_manual_center"
                    ),
                    help="关闭时逐层使用当前光束中心；开启时使用固定物理坐标。",
                )
                if manual_center:
                    center_columns = st.columns(2)
                    with center_columns[0]:
                        sampled_thermal_lens["center_x_m"] = sci(
                            "sampled-field 热拟合中心 x (m)",
                            sampled_thermal_lens.get("center_x_m") or 0.0,
                            key=form_widget_key(
                                "ui_seed_sampled_thermal_lens_center_x"
                            ),
                        )
                    with center_columns[1]:
                        sampled_thermal_lens["center_y_m"] = sci(
                            "sampled-field 热拟合中心 y (m)",
                            sampled_thermal_lens.get("center_y_m") or 0.0,
                            key=form_widget_key(
                                "ui_seed_sampled_thermal_lens_center_y"
                            ),
                        )
                else:
                    sampled_thermal_lens["center_x_m"] = None
                    sampled_thermal_lens["center_y_m"] = None

        if resolved_model == "spectral_time":
            joint = cfg["seed"].setdefault(
                "joint_time_frequency", {"model": "separable", "weights": []}
            )
            joint["model"] = choice_selectbox(
                "时间–频率联合能量",
                ["separable", "matrix"],
                index=option_index(
                    ["separable", "matrix"], str(joint.get("model", "separable"))
                ),
                key=form_widget_key("ui_seed_joint_time_frequency_model"),
                help="separable 使用时间/光谱两个边缘分布的乘积；matrix 可表示蓝前红后等非可分离强度相关。",
            )
            if joint["model"] == "matrix":
                joint["weights"] = parse_json(
                    st.text_area(
                        "联合权重矩阵 JSON（行=时间箱，列=波长箱）",
                        value=json.dumps(joint.get("weights", []), ensure_ascii=False),
                        height=110,
                        key=form_widget_key("ui_seed_joint_time_frequency_weights"),
                    ),
                    "时间–频率联合权重",
                )
                st.caption("权重必须有限、非负且总和大于零；运行时按实际求解分箱严格核对矩阵形状。该矩阵不包含复相干相位。")

        st.divider()
        if resolved_model == "fluence":
            if str(cfg["seed"].get("temporal", {}).get("type", "gaussian")) == "cw":
                st.info(
                    "当前 CW Seed 使用周期平均功率传播：每个 z 单元按局部线性指数增益推进，"
                    "并把平均光子流反馈给 Pump + Seed 布居固定点；不会调用单脉冲 FN 耗尽。"
                )
            elif bool(cfg["physics"].get("gain_saturation", True)):
                st.info(
                    "当前为快速单色整脉冲通量模型；局部饱和增益使用 FN，"
                    "没有独立的脉冲内部时间步。"
                )
            else:
                st.info(
                    "当前为快速单色通量模型；单发 FN 饱和已关闭，局部增益按"
                    "稳态布居给出的线性指数增益传播。"
                )
            return
        if resolved_model == "spectral_time":
            st.info("当前为多色×时间分箱模型；每个时间/波长分箱使用 FN。")
            return
        if resolved_model == "commercial_app_fast":
            commercial_app_config = cfg["seed"].get("commercial_app_fast", {})
            temporal_model = str(
                commercial_app_config.get("temporal_model", "fixed_duration")
            )
            if str(cfg["seed"].get("temporal", {}).get("type", "gaussian")) == "cw":
                st.info("当前 Commercial App CW 直接传播平均强度；Kerr 透镜使用平均强度，不构造脉冲或报告 SPM 展宽。")
            elif str(
                cfg["seed"].get("spatial_modes", {}).get(
                    "model", "effective_m2"
                )
            ) == "incoherent_hg":
                st.info(
                    "当前为 Commercial App 式独立 HG Gaussian-q 通道：同一波长的各模态先汇总局部"
                    "通量、同步竞争共享布居，再分别推进束宽和曲率；模态之间最终按非相干"
                    "强度求和，不解析完整 x-y 复场衍射。"
                )
            elif bool(
                commercial_app_config.get("include_relaxation_during_pulse", False)
            ):
                st.info(
                    "当前为 Commercial App 式共享 Gaussian-q + 脉内增益时间分箱：前沿、峰值和尾沿"
                    "依次耗尽布居，并在时间箱之间推进能级快速弛豫；不解析延迟时间输运。"
                )
            elif temporal_model == "effective_area_spm":
                st.info(
                    "当前为 Commercial App 式 Gaussian duct + 一维相干包络：光谱增益与布居交换仍逐层计算，"
                    "一维 FFT 保留输入相位、GDD/TOD、材料色散、effective-area SPM 与压缩诊断；"
                    "热/Kerr 透镜/增益导引的横向场仍投影为 Gaussian 二阶矩。"
                )
            else:
                st.info(
                    "当前为 Commercial App 式 Gaussian duct：光谱增益与布居交换仍逐层计算，"
                    "输入脉宽只用于固定峰值强度闭合；热/Kerr/增益导引投影为二阶矩。"
                )
            return
        if resolved_model == "coherent_cpa":
            st.info("相干 CPA 保留自己的总通量/软时间窗 FN 增益；JOSA 后端不会改写相干场分支。")
            return
        josa = cfg["seed"].setdefault(
            "josa",
            {
                "mode": "retarded_pc",
                "failure_policy": "error",
                "include_relaxation_during_pulse": False,
                "maximum_time_bins": 4096,
                "recommended_maximum_gain_step": 0.05,
            },
        )
        mode_order = ["retarded_pc", JOSA_FULL_3D_MODE, "pc_cfl", "explicit_cfl"]
        mode_labels = {
            str(JOSA_MODE_INFO[mode]["label_zh"]): mode for mode in mode_order
        }
        current_mode = str(josa.get("mode", "retarded_pc"))
        selected_label = choice_selectbox(
            "计算层级 / JOSA 输运模式",
            list(mode_labels),
            index=list(mode_labels.values()).index(current_mode) if current_mode in mode_labels.values() else 0,
            key=form_widget_key("ui_josa_mode"),
        )
        josa["mode"] = mode_labels[selected_label]
        mode_info = josa_mode_info(josa["mode"])
        st.caption(str(mode_info["description_zh"]))
        st.info(
            "JOSA 会先用源/展宽器相位构造一次输入时间强度，再把相位丢弃；随后只传播"
            "有序的时间×波长通量和真实能级布居。材料/光路 GDD、压缩器、Kerr 与 SPM "
            "不会在 JOSA 内生效；需要这些输出相位效应时请选择 coherent_cpa。"
        )
        if josa["mode"] == JOSA_FULL_3D_MODE:
            # The runtime also enforces this.  Writing the effective failure
            # policy into the form keeps downloaded JSON self-explanatory.
            josa["failure_policy"] = "error"
            effective = resolve_physics(cfg)
            st.success(
                "完整 3D 速率方程合同已启用：不会静默回退 FN；Seed/Pump FFT、真实瞬态布居、"
                "泵浦耗尽、三维热光、宽带群延迟走离和斜入射横向走离同时进入计算。"
            )
            with st.expander("查看完整 3D 实际锁定项与边界", expanded=False):
                locked_labels = {
                    "optical_propagation": "Seed 晶体内/外传播",
                    "pump_propagation": "Pump 传播",
                    "population_dynamics": "布居动力学",
                    "pump_depletion": "泵浦耗尽",
                    "gain_saturation": "受激布居耗尽",
                    "thermal_model": "热场",
                    "refractive_index_model": "热致折射率",
                    "record_optical_planes": "光学面记录",
                }
                st.dataframe(
                    [
                        {"模块": label, "实际设置": str(effective.get(key))}
                        for key, label in locked_labels.items()
                    ],
                    hide_index=True,
                    width="stretch",
                )
                st.warning(
                    "这里的“完整”限定在 JOSA 的强度/光子数密度 + 速率方程范围。"
                    "跨频率相干相位、GDD/TOD 压缩和 Kerr/SPM 请使用 coherent_cpa；"
                    "两者不能用一个强度方程假装成同一自由度。"
                )
                st.caption(
                    "角度来源：Seed 源入射角写入源面的横向相位；每个 gain_medium 的晶体入射角"
                    "写入该程的有效光程与额外横向走离。若两者表示同一个实验角度，只填一处；"
                    "两处同时非零时程序会按两个独立倾斜相加并给出警告。"
                )
        elif josa["mode"] == "retarded_pc":
            st.success("时间分箱与 Nz 独立；用 T=t-z/vg 的移动坐标保留 front → peak → tail。")
        else:
            st.warning("论文 CFL 模式会按 Δt=Δz/vg 自动重采样时间分箱；时间分辨率不足时必须增加 Nz。")

        with st.expander("四种层级怎么选", expanded=False):
            st.dataframe(
                [
                    {
                        "层级": JOSA_MODE_INFO[mode]["label_zh"],
                        "积分": JOSA_MODE_INFO[mode]["scheme"],
                        "时间坐标": JOSA_MODE_INFO[mode]["coordinate_mode"],
                        "频率相干相位": "不传播",
                        "群延迟走离": "近似" if mode == JOSA_FULL_3D_MODE else "不解析",
                        "用途": JOSA_MODE_INFO[mode]["description_zh"],
                    }
                    for mode in mode_order
                ],
                hide_index=True,
                width="stretch",
            )

        josa["include_relaxation_during_pulse"] = choice_checkbox(
            "脉冲内部推进自发弛豫/自定义非线性过程",
            value=bool(josa.get("include_relaxation_during_pulse", False)),
            key=form_widget_key("ui_josa_relaxation"),
            help="200 fs 相对 µs 寿命时通常关闭；脉宽接近某个能级或上转换时间尺度时再开启。",
        )
        advanced = choice_toggle(
            "显示 JOSA 数值保护设置",
            value=False,
            key=form_widget_key("ui_josa_advanced"),
        )
        if advanced:
            josa["failure_policy"] = choice_selectbox(
                "JOSA 子模块失败时",
                ["fallback", "error"],
                index=0 if josa.get("failure_policy", "error") == "fallback" else 1,
                key=form_widget_key("ui_josa_failure_policy"),
                help=(
                    "fallback 会先回滚本次 Seed 对布居的修改，再用 spectral_time/FN 重算。"
                    "孤立脉冲可返回该降阶结果；若有状态的脉冲串、周期图或热光闭合已经用 JOSA "
                    "推进，运行中途改换后端不会被计为收敛，并会在无法保持同一映射时终止。"
                ),
                disabled=josa["mode"] == JOSA_FULL_3D_MODE,
            )
            josa["recommended_maximum_gain_step"] = st.number_input(
                "建议每个 z 单元最大 |g·Δz|",
                min_value=0.001,
                max_value=1.0,
                value=float(josa.get("recommended_maximum_gain_step", 0.05)),
                step=0.005,
                format="%.4f",
                key=form_widget_key("ui_josa_max_gain_step"),
            )
            josa["maximum_time_bins"] = int(
                st.number_input(
                    "论文 CFL 自动时间分箱上限",
                    min_value=1,
                    max_value=100000,
                    value=int(josa.get("maximum_time_bins", 4096)),
                    step=128,
                    key=form_widget_key("ui_josa_max_time_bins"),
                )
            )

        try:
            group_index = material_group_index(
                cfg["material"],
                float(cfg["seed"]["wavelength_m"]),
                role="signal",
            )
            dz = float(cfg["grid"]["length_z_m"]) / int(cfg["grid"]["nz"])
            seed_path = cfg.get("optical_train", {}).get("seed", [])
            gain_markers = [element for element in seed_path if element.get("type") == "gain_medium"]
            if not gain_markers:
                gain_markers = [{}]
            path_cosines = [
                gain_medium_path_cosine(marker)
                for marker in gain_markers
            ]
            transport_step = dz / path_cosines[0]
            cfl_dt = group_index * transport_step / LIGHT_SPEED
            metric_columns = st.columns(3)
            metric_columns[0].metric("群折射率 ng", f"{group_index:.6g}")
            metric_columns[1].metric("当前有效 Δs", f"{transport_step * 1e6:.5g} µm")
            metric_columns[2].metric("论文 CFL Δt", f"{cfl_dt * 1e15:.5g} fs")
            if josa["mode"].endswith("_cfl") and not np.allclose(
                path_cosines,
                path_cosines[0],
                rtol=1e-12,
                atol=1e-15,
            ):
                st.error("不同晶体程次的入射角给出不同有效步长；严格 CFL 模式请改用延迟时间 PC。")
        except Exception as exc:
            st.warning(f"暂时无法计算 JOSA CFL 预览：{type(exc).__name__}: {exc}")
        if cfg["energy_levels"].get("gain_override_m_inv") is not None:
            st.error("JOSA 需要实际上下能级布居；请把增益来源切换为 Pump + 速率方程。")
        if not bool(resolve_physics(cfg).get("gain_saturation", True)):
            st.error("JOSA 需要“增益饱和/布居耗尽 = on”；表单会在运行前自动联动并显示变更。")
        st.caption(
            "模块边界：JOSA 替换晶体内局部 FN 增益更新；光路、泵浦、热和结果输出仍由独立模块提供。"
            "full_3d_pc 会锁定并核验这些模块的高保真组合，其他三种模式允许按需要退化。"
        )


def render_seed_polarization_controls(cfg: dict[str, Any]) -> None:
    """Edit the Jones input state and thermal photoelastic retarder."""

    polarization = cfg["seed"].setdefault(
        "polarization", copy.deepcopy(DEFAULTS["seed"]["polarization"])
    )
    with st.expander(
        "C. Jones 偏振与径向/切向光弹双折射",
        expanded=str(polarization.get("model", "scalar")) == "jones",
    ):
        model_options = ["scalar", "jones"]
        polarization["model"] = choice_selectbox(
            "Seed 偏振传播",
            model_options,
            index=option_index(model_options, str(polarization.get("model", "scalar"))),
            key=form_widget_key("ui_seed_polarization_model"),
            help="Jones 模式传播 Ex/Ey 两个复分量；增益饱和由两分量总强度共同决定。",
        )
        if polarization["model"] != "jones":
            polarization["birefringence_model"] = "off"
            st.caption("标量模式保持原有计算路径，不创建经验性退偏损耗。")
            return
        if resolve_signal_model(cfg) != "fluence":
            st.error("当前 Jones 后端要求单色 fluence；请在运行方式中选择 fluence。")
        state_options = ["linear", "elliptical", "right_circular", "left_circular"]
        polarization["input_state"] = choice_selectbox(
            "输入偏振态",
            state_options,
            index=option_index(
                state_options, str(polarization.get("input_state", "linear"))
            ),
            key=form_widget_key("ui_seed_polarization_state"),
        )
        if polarization["input_state"] in {"linear", "elliptical"}:
            polarization["linear_angle_deg"] = st.number_input(
                "x→y 偏振方位角 (deg)",
                value=float(polarization.get("linear_angle_deg", 0.0)),
                key=form_widget_key("ui_seed_polarization_angle"),
            )
        if polarization["input_state"] == "elliptical":
            polarization["relative_phase_y_minus_x_rad"] = st.number_input(
                "Ey−Ex 相对相位 (rad)",
                value=float(
                    polarization.get("relative_phase_y_minus_x_rad", 0.0)
                ),
                key=form_widget_key("ui_seed_polarization_phase"),
            )
        birefringence_options = [
            "off",
            "radial_tangential_plane_stress",
            "tensor_photoelastic",
        ]
        polarization["birefringence_model"] = choice_selectbox(
            "晶体内双折射",
            birefringence_options,
            index=option_index(
                birefringence_options,
                str(polarization.get("birefringence_model", "off")),
            ),
            key=form_widget_key("ui_seed_birefringence_model"),
        )
        if polarization["birefringence_model"] == "radial_tangential_plane_stress":
            columns = st.columns(2)
            with columns[0]:
                polarization["radial_photoelastic_C"] = st.number_input(
                    "径向 Cᵣ",
                    value=float(polarization.get("radial_photoelastic_C", 0.0)),
                    format="%.6g",
                    key=form_widget_key("ui_seed_photoelastic_cr"),
                )
            with columns[1]:
                polarization["tangential_photoelastic_C"] = st.number_input(
                    "切向 Cθ",
                    value=float(
                        polarization.get("tangential_photoelastic_C", 0.0)
                    ),
                    format="%.6g",
                    key=form_widget_key("ui_seed_photoelastic_ct"),
                )
            st.caption(
                "使用 Δnᵣ,θ = 2 n₀³ αT Cᵣ,θ [T(r,z)−T(0,z)]；同时保留公共 dn/dT 相位。"
                "这是轴对称端泵降阶式，不代表方形边界、夹持和晶向的完整光弹张量。"
                "检偏器正交功率只在结果中报告，不自动从总输出扣除。"
            )
            effective = resolve_physics(cfg)
            if effective.get("refractive_index_model") != "thermal_3d":
                st.error("该模型需要 thermal_3d 折射率场。")
            if cfg["material"].get("mechanical", {}).get("stress_model", "none") != "none":
                st.error("为避免把标量应力 Δn 重复计入，当前该模型要求 stress_model=none。")
        elif polarization["birefringence_model"] == "tensor_photoelastic":
            st.caption(
                "使用各向异性 FEM 的旋转光弹张量 ΔB=p:ε 生成局部 2×2 Jones 延迟；"
                "公共张量折射率和热光相位只计一次。"
            )
            mechanical = cfg["material"].get("mechanical", {})
            if mechanical.get("stress_model") != "anisotropic_thermoelastic_fem":
                st.error("张量 Jones 耦合需要 anisotropic_thermoelastic_fem。")
            if mechanical.get("thermoelastic_fem", {}).get("photoelastic_coupling") != "tensor_photoelastic":
                st.error("还需把 FEM photoelastic_coupling 设为 tensor_photoelastic。")


def render_seed_spatial_controls(cfg: dict[str, Any]) -> None:
    """Render Seed transverse-source controls as one isolated UI section."""

    profile_options = ["super_gaussian", "radial_super_gaussian", "gaussian", "top_hat", "annular"]
    cfg["seed"]["profile"] = choice_selectbox(
        "空间光斑",
        profile_options,
        index=option_index(profile_options, str(cfg["seed"]["profile"])),
        key=form_widget_key("ui_seed_profile"),
    )
    waist_columns = st.columns(2)
    with waist_columns[0]:
        cfg["seed"]["waist_x_m"] = sci(
            "1/e² 半径 wx (m)",
            cfg["seed"]["waist_x_m"],
            key=form_widget_key("ui_seed_wx"),
        )
    with waist_columns[1]:
        cfg["seed"]["waist_y_m"] = sci(
            "1/e² 半径 wy (m)",
            cfg["seed"]["waist_y_m"],
            key=form_widget_key("ui_seed_wy"),
        )
    if cfg["seed"]["profile"] != "annular":
        top_hat_profile = cfg["seed"]["profile"] == "top_hat"
        order_label_x = (
            "Seed Top-hat 软边 order x（>1）"
            if top_hat_profile
            else "Seed 横向 order x（1=Gaussian）"
        )
        order_label_y = (
            "Seed Top-hat 软边 order y（>1）"
            if top_hat_profile
            else "Seed 横向 order y（1=Gaussian）"
        )
        order_help = (
            "Top-hat 在数值上使用高阶超高斯软边；order≤1 的旧配置会按 20 执行，表单在这里显式显示为 20。"
            if top_hat_profile
            else "径向超高斯在两个 order 相同时使用 exp[-2((x/wx)²+(y/wy)²)^order]；两个 order 不同时退回 x/y 可分离超高斯。"
            if cfg["seed"]["profile"] == "radial_super_gaussian"
            else (
                "实际强度指数为 exp[-2|x/wx|^(2·order)]；profile=gaussian "
                "是兼容别名，真实形状仍由这里的 order 决定。"
            )
        )
        order_columns = st.columns(2)
        with order_columns[0]:
            cfg["seed"]["order_x"] = st.number_input(
                order_label_x,
                min_value=1.000001 if top_hat_profile else 0.05,
                value=(
                    float(cfg["seed"].get("order_x", 20.0))
                    if float(cfg["seed"].get("order_x", 1.0)) > 1.0
                    else 20.0
                )
                if top_hat_profile
                else float(cfg["seed"].get("order_x", 1.0)),
                key=form_widget_key("ui_seed_order_x"),
                help=order_help,
            )
        with order_columns[1]:
            cfg["seed"]["order_y"] = st.number_input(
                order_label_y,
                min_value=1.000001 if top_hat_profile else 0.05,
                value=(
                    float(cfg["seed"].get("order_y", 20.0))
                    if float(cfg["seed"].get("order_y", 1.0)) > 1.0
                    else 20.0
                )
                if top_hat_profile
                else float(cfg["seed"].get("order_y", 1.0)),
                key=form_widget_key("ui_seed_order_y"),
                help=order_help,
            )
    else:
        annular = cfg["seed"].setdefault("extras", {})
        explicit_ring_order = choice_toggle(
            "直接指定环形幂指数",
            value=annular.get("ring_order") is not None,
            key=form_widget_key("ui_seed_annular_explicit_ring_order"),
            help=(
                "强度为 (r²)^a exp(-2r²)。直接指定 a 时，峰值椭圆半径为 sqrt(a/2)；"
                "关闭时用峰值半径比例自动计算 a。"
            ),
        )
        if explicit_ring_order:
            annular["ring_order"] = st.number_input(
                "环形幂指数 a",
                min_value=0.0,
                value=float(
                    annular.get(
                        "ring_order",
                        2.0
                        * float(annular.get("inner_radius_ratio", 0.5)) ** 2,
                    )
                ),
                key=form_widget_key("ui_seed_annular_ring_order"),
            )
            annular["inner_radius_ratio"] = math.sqrt(
                max(float(annular["ring_order"]), 0.0) / 2.0
            )
            st.caption(
                lt("当前环峰位于归一化椭圆半径 r/w = {value}。").format(
                    value=f"{annular['inner_radius_ratio']:.6g}"
                )
            )
        else:
            annular.pop("ring_order", None)
            annular["inner_radius_ratio"] = st.number_input(
                "环峰半径 / 1/e² 半径",
                min_value=0.0,
                value=float(annular.get("inner_radius_ratio", 0.5)),
                key=form_widget_key("ui_seed_annular_peak_ratio"),
                help="求解器使用 a=2·(环峰半径比例)²，0 会退化为中心峰。",
            )

    seed_focus_is_crystal_relative = resolve_physics(cfg)["optical_propagation"] == "off"
    seed_focus_label = (
        "腰斑相对晶体入口 z (m)"
        if seed_focus_is_crystal_relative
        else "腰斑相对 SEED LASER 源面 z (m)"
    )
    separate_seed_focus = choice_checkbox(
        "x / y 使用不同腰斑位置（像散光束）",
        value=(
            cfg["seed"].get("focus_x_m") is not None
            or cfg["seed"].get("focus_y_m") is not None
        ),
        key=form_widget_key("ui_seed_separate_focus"),
    )
    if separate_seed_focus:
        focus_columns = st.columns(2)
        with focus_columns[0]:
            cfg["seed"]["focus_x_m"] = sci(
                f"x {seed_focus_label}",
                cfg["seed"].get("focus_x_m")
                if cfg["seed"].get("focus_x_m") is not None
                else cfg["seed"]["focus_z_m"],
                key=form_widget_key("ui_seed_focus_x"),
            )
        with focus_columns[1]:
            cfg["seed"]["focus_y_m"] = sci(
                f"y {seed_focus_label}",
                cfg["seed"].get("focus_y_m")
                if cfg["seed"].get("focus_y_m") is not None
                else cfg["seed"]["focus_z_m"],
                key=form_widget_key("ui_seed_focus_y"),
            )
        cfg["seed"]["focus_z_m"] = 0.5 * (
            float(cfg["seed"]["focus_x_m"]) + float(cfg["seed"]["focus_y_m"])
        )
    else:
        cfg["seed"]["focus_z_m"] = sci(
            seed_focus_label,
            cfg["seed"]["focus_z_m"],
            key=form_widget_key("ui_seed_focus"),
            help_text="ABCD/FFT 会从 LASER 源面开始传播；off 只保留晶体内规定光束。",
        )
        cfg["seed"]["focus_x_m"] = None
        cfg["seed"]["focus_y_m"] = None

    configured_wavefront_radius = cfg["seed"].get("wavefront_radius_m")
    explicit_wavefront_radius = choice_toggle(
        "显式覆盖源面波前曲率",
        value=configured_wavefront_radius not in (None, 0),
        key=form_widget_key("ui_seed_explicit_wavefront_radius"),
        help=(
            "开启后，这个曲率半径优先于上面的焦点位置来定义源面二次相位；"
            "正负号区分发散/会聚波前。"
        ),
    )
    if explicit_wavefront_radius:
        cfg["seed"]["wavefront_radius_m"] = sci(
            "Seed 源面波前曲率半径 (m)",
            float(
                configured_wavefront_radius
                if configured_wavefront_radius not in (None, 0)
                else 1.0
            ),
            key=form_widget_key("ui_seed_wavefront_radius_m"),
            help_text="非零有限值；它会覆盖由 waist focus 推出的源面曲率。",
        )
        st.caption("当前焦点位置仍决定源面光斑尺度，但源面二次相位由显式曲率半径决定。")
    else:
        cfg["seed"]["wavefront_radius_m"] = None

    pose_columns = st.columns(3)
    with pose_columns[0]:
        cfg["seed"]["center_x_m"] = sci(
            "Seed 腰斑中心 x (m)",
            cfg["seed"].get("center_x_m", 0.0),
            key=form_widget_key("ui_seed_center_x"),
        )
    with pose_columns[1]:
        cfg["seed"]["center_y_m"] = sci(
            "Seed 腰斑中心 y (m)",
            cfg["seed"].get("center_y_m", 0.0),
            key=form_widget_key("ui_seed_center_y"),
        )
    with pose_columns[2]:
        cfg["seed"]["rotation_deg"] = st.number_input(
            "Seed 横向椭圆旋转 (deg)",
            value=float(cfg["seed"].get("rotation_deg", 0.0)),
            key=form_widget_key("ui_seed_rotation_deg"),
            help="绕光轴旋转横向椭圆/超高斯主轴；不会改变光轴传播方向。",
        )

    quality_columns = st.columns(2)
    with quality_columns[0]:
        cfg["seed"]["m2_x"] = st.number_input(
            "M² x",
            min_value=1.0,
            value=float(cfg["seed"]["m2_x"]),
            step=0.05,
            format="%.3f",
            key=form_widget_key("ui_seed_m2x"),
        )
    with quality_columns[1]:
        cfg["seed"]["m2_y"] = st.number_input(
            "M² y",
            min_value=1.0,
            value=float(cfg["seed"]["m2_y"]),
            step=0.05,
            format="%.3f",
            key=form_widget_key("ui_seed_m2y"),
        )
    mode_cfg = cfg["seed"].setdefault(
        "spatial_modes",
        {"model": "effective_m2", "maximum_order_x": 4, "maximum_order_y": 4},
    )
    mode_labels = {
        "M² 有效高斯（快速）": "effective_m2",
        "HG 强度集合（快速降阶）": "hermite_gaussian",
        "HG 模态逐个传播（共享布居）": "incoherent_hg",
    }
    current_mode = str(mode_cfg.get("model", "effective_m2"))
    mode_keys = list(mode_labels.values())
    selected_mode_label = choice_selectbox(
        "横向模态模型",
        list(mode_labels),
        index=mode_keys.index(current_mode) if current_mode in mode_keys else 0,
        key=form_widget_key("ui_seed_spatial_mode_model"),
    )
    mode_cfg["model"] = mode_labels[selected_mode_label]
    radial_hg = False
    if mode_cfg["model"] == "incoherent_hg" and resolve_signal_model(cfg) == "coherent_cpa":
        radial_hg = choice_checkbox(
            "径向三模式：HG00、HG10、HG01（仅 m=0、±1）",
            value=(cfg["seed"]["coherent"].get("fidelity_mode") == "axisymmetric_kerr"
                   and mode_cfg.get("maximum_total_order") == 1),
            key=form_widget_key("ui_seed_hg_radial_three_modes"),
            help="三个非相干光场共享布居和总强度 Kerr；Cycle Average 使用相同模式。要求两轴 M² 相等、1≤M²≤2，关闭逐时间窗增益和强场损耗。",
        )
        if radial_hg:
            mode_cfg.update(maximum_order_x=1, maximum_order_y=1,
                            maximum_total_order=1, maximum_propagated_modes=3)
            cfg["seed"]["coherent"]["fidelity_mode"] = "axisymmetric_kerr"
            cfg["seed"]["coherent"]["axisymmetric"]["azimuthal_fidelity"] = "m0_m1"
            cfg["seed"]["coherent"]["transient_gain"]["enabled"] = False
            for key, value in (("ui_seed_hg_max_x", 1), ("ui_seed_hg_max_y", 1),
                               ("ui_seed_hg_maximum_propagated_modes", 3),
                               ("ui_seed_hg_maximum_total_order", 1),
                               ("ui_axisymmetric_azimuthal_fidelity", "m0_m1"),
                               ("ui_transient_gain_enabled", False)):
                st.session_state[form_widget_key(key)] = value
        elif cfg["seed"]["coherent"].get("fidelity_mode") == "axisymmetric_kerr":
            cfg["seed"]["coherent"]["fidelity_mode"] = "hybrid_whole_fluence"
    if mode_cfg["model"] in {"hermite_gaussian", "incoherent_hg"}:
        required_x = math.ceil(max((float(cfg["seed"]["m2_x"]) - 1.0) / 2.0, 0.0))
        required_y = math.ceil(max((float(cfg["seed"]["m2_y"]) - 1.0) / 2.0, 0.0))
        order_x_key = form_widget_key("ui_seed_hg_max_x")
        order_y_key = form_widget_key("ui_seed_hg_max_y")
        order_x_value = max(
            int(
                st.session_state.get(
                    order_x_key, mode_cfg.get("maximum_order_x", 4)
                )
            ),
            0 if radial_hg else required_x,
        )
        order_y_value = max(
            int(
                st.session_state.get(
                    order_y_key, mode_cfg.get("maximum_order_y", 4)
                )
            ),
            0 if radial_hg else required_y,
        )
        st.session_state[order_x_key] = order_x_value
        st.session_state[order_y_key] = order_y_value
        order_columns = st.columns(2)
        with order_columns[0]:
            mode_cfg["maximum_order_x"] = int(
                st.number_input(
                    "x 最大 HG 模阶",
                    min_value=0 if radial_hg else required_x,
                    max_value=max(50, order_x_value),
                    disabled=radial_hg,
                    key=order_x_key,
                )
            )
        with order_columns[1]:
            mode_cfg["maximum_order_y"] = int(
                st.number_input(
                    "y 最大 HG 模阶",
                    min_value=0 if radial_hg else required_y,
                    max_value=max(50, order_y_value),
                    disabled=radial_hg,
                    key=order_y_key,
                )
            )
        if mode_cfg["model"] == "hermite_gaussian":
            weights_x = maximum_entropy_mode_weights(
                cfg["seed"]["m2_x"], mode_cfg["maximum_order_x"]
            )
            weights_y = maximum_entropy_mode_weights(
                cfg["seed"]["m2_y"], mode_cfg["maximum_order_y"]
            )
            active_x = ", ".join(
                f"HG{n}: {weight:.1%}"
                for n, weight in enumerate(weights_x)
                if weight >= 0.005
            )
            active_y = ", ".join(
                f"HG{n}: {weight:.1%}"
                for n, weight in enumerate(weights_y)
                if weight >= 0.005
            )
            st.caption(
                "模态权重由 M² 约束并采用最大熵分布；最大模阶会改变横向强度形状。"
                f" x [{active_x}]；y [{active_y}]。"
            )
        if mode_cfg["model"] == "incoherent_hg":
            resolved_hg_backend = resolve_signal_model(cfg)
            commercial_app_hg = resolved_hg_backend == "commercial_app_fast"
            coherent_hg = resolved_hg_backend == "coherent_cpa"
            limit_total_order = commercial_app_hg or radial_hg or choice_checkbox(
                "限制 HG 总阶数 m+n",
                value=mode_cfg.get("maximum_total_order") is not None,
                key=form_widget_key("ui_seed_hg_limit_total_order"),
                help="总阶数1保留HG00、HG10、HG01；总阶数2另含HG20、HG11、HG02。",
            )
            if limit_total_order:
                minimum_total = math.ceil(max((float(cfg["seed"]["m2_x"]) + float(cfg["seed"]["m2_y"]) - 2.0) / 2.0 - 1e-12, 0.0))
                configured_total_order = mode_cfg.get("maximum_total_order")
                maximum_total_value = max(
                    int(configured_total_order) if configured_total_order is not None
                    else max(int(mode_cfg["maximum_order_x"]), int(mode_cfg["maximum_order_y"])),
                    0 if radial_hg else minimum_total,
                )
                mode_cfg["maximum_total_order"] = int(st.number_input(
                    "Commercial App 最大总 HG 阶数 m+n" if commercial_app_hg else "最大总 HG 阶数 m+n",
                    min_value=0 if radial_hg else minimum_total,
                    max_value=max(50, maximum_total_value), value=maximum_total_value,
                    disabled=radial_hg,
                    key=form_widget_key("ui_seed_hg_maximum_total_order"),
                ))
            else:
                mode_cfg["maximum_total_order"] = None
            maximum_total = mode_cfg.get("maximum_total_order")
            if maximum_total is None:
                requested_mode_capacity = (
                    int(mode_cfg["maximum_order_x"]) + 1
                ) * (int(mode_cfg["maximum_order_y"]) + 1)
            else:
                requested_mode_capacity = sum(
                    1
                    for order_x in range(int(mode_cfg["maximum_order_x"]) + 1)
                    for order_y in range(int(mode_cfg["maximum_order_y"]) + 1)
                    if order_x + order_y <= int(maximum_total)
                )
            mode_capacity_key = form_widget_key(
                "ui_seed_hg_maximum_propagated_modes"
            )
            mode_capacity_value = max(
                int(
                    st.session_state.get(
                        mode_capacity_key,
                        mode_cfg.get("maximum_propagated_modes", 64),
                    )
                ),
                requested_mode_capacity,
            )
            st.session_state[mode_capacity_key] = mode_capacity_value
            mode_cfg["maximum_propagated_modes"] = int(
                st.number_input(
                    "最多独立传播 HG 模态数",
                    min_value=requested_mode_capacity,
                    max_value=max(400, mode_capacity_value),
                    key=mode_capacity_key,
                    disabled=radial_hg,
                )
            )
            if commercial_app_hg:
                modes = incoherent_hg_modes(cfg["seed"])
                st.info(
                    "Commercial App Fast 为每个 HG 模态独立推进 Gaussian q 和能量；同一波长的"
                    "所有模态先合计局部通量、只更新一次共享布居，再按局部输入比例分配"
                    "输出，最后按非相干强度求和。当前共传播 "
                    f"{len(modes)} 个模态，不使用 x-y FFT。"
                )
            elif coherent_hg:
                if radial_hg:
                    st.info(
                        "径向三模式：各 HG 模态独立传播 E(f,r,θ)，仅保留角向 m=0、±1；"
                        "同频总通量更新一次共享布居，Kerr 使用所有模态的总瞬时强度。"
                        "Cycle Average 使用同样模式；每个模式以 M²=1 衍射，集合 M² 由权重体现。"
                    )
                    st.caption("空气腰半径不会因修改 M² 自动锁定。当前半径和焦点输入定义源面 q；比较不同 M² 时，请同时检查腰半径和焦点设置。")
                else:
                    st.info(
                        "CPA 显式 HG：每个模式保留完整 E(f,y,x)；模式间非相干，"
                        "同频共享布居，Kerr、多光子吸收和 FCA 读取总瞬时强度。"
                        "当前使用完整 x-y FFT 和整脉冲增益闭合。"
                    )
            else:
                st.info(
                    "每个 HG 模态独立经历衍射、孔径、局域增益和热相位，所有模态在同一 z/t 步共享布居，"
                    "最终按非相干强度求和。该路径使用 FFT；有效 M² 快速模型仍保留。"
                )

            try:
                actual_modes = incoherent_hg_modes(cfg["seed"])
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.caption(f"实际传播 {len(actual_modes)} 个非相干 HG 模式；容量 {mode_cfg['maximum_propagated_modes']}。")
                st.dataframe([
                    {"模式": f"HG{int(mode['order_x'])}{int(mode['order_y'])}",
                     "能量占比 (%)": 100.0 * float(mode["weight"])}
                    for mode in actual_modes
                ], hide_index=True, width="stretch")

    diagnostic_columns = st.columns(2)
    for column, axis in zip(diagnostic_columns, ("x", "y"), strict=False):
        waist = float(cfg["seed"][f"waist_{axis}_m"])
        quality = float(cfg["seed"][f"m2_{axis}"])
        theta = divergence_half_angle_rad(
            waist,
            float(cfg["seed"]["wavelength_m"]),
            quality,
        )
        rayleigh = rayleigh_range_m(
            waist,
            float(cfg["seed"]["wavelength_m"]),
            quality,
        )
        with column:
            st.metric(f"{axis} 发散半角", f"{math.degrees(theta):.4g}°")
            st.caption(f"{theta:.4g} rad; zR = {rayleigh * 1e3:.4g} mm")

    angle_columns = st.columns(2)
    with angle_columns[0]:
        cfg["seed"]["incidence_angle_x_deg"] = st.number_input(
            "Seed 光轴倾斜 x (deg)",
            value=float(cfg["seed"].get("incidence_angle_x_deg", 0.0)),
            key=form_widget_key("ui_seed_ax"),
            help="这是源光轴方向，不是发散角。晶体表面的入射角在光路 gain_medium 元件中设置。",
        )
    with angle_columns[1]:
        cfg["seed"]["incidence_angle_y_deg"] = st.number_input(
            "Seed 光轴倾斜 y (deg)",
            value=float(cfg["seed"].get("incidence_angle_y_deg", 0.0)),
            key=form_widget_key("ui_seed_ay"),
        )

    with st.expander("空间—光谱耦合（不同波长的腰斑/焦点/中心/角度）", expanded=False):
        spatio = cfg["seed"].setdefault(
            "spatio_spectral", copy.deepcopy(DEFAULTS["seed"]["spatio_spectral"])
        )
        model_labels = {
            "共享同一个源面参数（兼容/默认）": "shared_source",
            "参数化色差与空间啁啾": "chromatic_parametric",
            "逐波长参数表（缺项自动沿用参数化值）": "wavelength_table",
        }
        current_model = str(spatio.get("model", "shared_source"))
        selected_label = choice_selectbox(
            "不同波长的横向源模型",
            list(model_labels),
            index=list(model_labels.values()).index(current_model)
            if current_model in model_labels.values()
            else 0,
            key=form_widget_key("ui_seed_spatio_spectral_model"),
        )
        spatio["model"] = model_labels[selected_label]
        st.caption(
            "这里控制 E(ω,x,y) 的横向状态；GDD/TOD 和光谱相位仍在 Seed Fourier/相干区。"
            "M²=1 只说明各波长可为基模，并不自动保证它们的腰斑、焦点或光轴相同。"
        )
        if spatio["model"] != "shared_source":
            spatio["reference_wavelength_m"] = sci(
                "空间参数参考波长 (m)",
                spatio.get("reference_wavelength_m") or cfg["seed"]["wavelength_m"],
                key=form_widget_key("ui_seed_spatio_reference_wavelength"),
            )
            exponent_columns = st.columns(2)
            for column, axis in zip(exponent_columns, ("x", "y"), strict=True):
                with column:
                    spatio[f"waist_scaling_exponent_{axis}"] = st.number_input(
                        f"w{axis}(λ) 幂指数",
                        value=float(spatio.get(f"waist_scaling_exponent_{axis}", 0.0)),
                        step=0.1,
                        key=form_widget_key(f"ui_seed_waist_lambda_exponent_{axis}"),
                        help="w(λ)=w_ref·(λ/λ_ref)^p；p=0 表示源面腰斑不随波长变。",
                    )
                    spatio[f"m2_scaling_exponent_{axis}"] = st.number_input(
                        f"M²{axis}(λ) 幂指数",
                        value=float(spatio.get(f"m2_scaling_exponent_{axis}", 0.0)),
                        step=0.1,
                        key=form_widget_key(f"ui_seed_m2_lambda_exponent_{axis}"),
                    )
            for title, prefix, unit in (
                ("焦点随相对波长的位移", "focus_shift", "m / Δλ/λ"),
                ("横向中心随相对波长的位移", "center_shift", "m / Δλ/λ"),
                ("光轴角随相对波长的位移", "angle_shift", "deg / Δλ/λ"),
            ):
                columns = st.columns(2)
                for column, axis in zip(columns, ("x", "y"), strict=True):
                    key = f"{prefix}_{axis}_{'deg' if prefix == 'angle_shift' else 'm'}_per_fraction"
                    with column:
                        spatio[key] = sci(
                            f"{title} {axis} ({unit})",
                            float(spatio.get(key, 0.0)),
                            key=form_widget_key(f"ui_seed_{key}"),
                        )

        if spatio["model"] == "wavelength_table":
            columns = [
                "wavelength_nm", "waist_x_um", "waist_y_um", "m2_x", "m2_y",
                "focus_x_mm", "focus_y_mm", "center_x_um", "center_y_um",
                "angle_x_deg", "angle_y_deg",
            ]
            display_rows = []
            for row in spatio.get("samples", []):
                display_rows.append(
                    {
                        "wavelength_nm": float(row["wavelength_m"]) * 1e9,
                        "waist_x_um": None if row.get("waist_x_m") is None else float(row["waist_x_m"]) * 1e6,
                        "waist_y_um": None if row.get("waist_y_m") is None else float(row["waist_y_m"]) * 1e6,
                        "m2_x": row.get("m2_x"), "m2_y": row.get("m2_y"),
                        "focus_x_mm": None if row.get("focus_x_m") is None else float(row["focus_x_m"]) * 1e3,
                        "focus_y_mm": None if row.get("focus_y_m") is None else float(row["focus_y_m"]) * 1e3,
                        "center_x_um": None if row.get("center_x_m") is None else float(row["center_x_m"]) * 1e6,
                        "center_y_um": None if row.get("center_y_m") is None else float(row["center_y_m"]) * 1e6,
                        "angle_x_deg": row.get("incidence_angle_x_deg"),
                        "angle_y_deg": row.get("incidence_angle_y_deg"),
                    }
                )
            default_table_row = {column: None for column in columns}
            default_table_row["wavelength_nm"] = float(
                spatio.get("reference_wavelength_m") or cfg["seed"]["wavelength_m"]
            ) * 1e9
            edited = st.data_editor(
                display_rows or [default_table_row],
                num_rows="dynamic",
                width="stretch",
                key=form_widget_key("ui_seed_spatio_spectral_table"),
                column_order=columns,
                hide_index=True,
            )
            rows = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)
            samples = []
            conversions = {
                "waist_x_um": ("waist_x_m", 1e-6), "waist_y_um": ("waist_y_m", 1e-6),
                "m2_x": ("m2_x", 1.0), "m2_y": ("m2_y", 1.0),
                "focus_x_mm": ("focus_x_m", 1e-3), "focus_y_mm": ("focus_y_m", 1e-3),
                "center_x_um": ("center_x_m", 1e-6), "center_y_um": ("center_y_m", 1e-6),
                "angle_x_deg": ("incidence_angle_x_deg", 1.0),
                "angle_y_deg": ("incidence_angle_y_deg", 1.0),
            }
            for row in rows:
                wavelength_nm = row.get("wavelength_nm")
                if wavelength_nm is None or not np.isfinite(float(wavelength_nm)):
                    continue
                sample: dict[str, float] = {"wavelength_m": float(wavelength_nm) * 1e-9}
                for display_key, (config_key, scale) in conversions.items():
                    value = row.get(display_key)
                    if value is not None and np.isfinite(float(value)):
                        sample[config_key] = float(value) * scale
                samples.append(sample)
            spatio["samples"] = samples

        if spatio["model"] != "shared_source":
            bandwidth = float(cfg["seed"].get("spectrum", {}).get("bandwidth_fwhm_m", 0.0))
            reference = float(spatio.get("reference_wavelength_m") or cfg["seed"]["wavelength_m"])
            preview_wavelengths = [reference]
            if bandwidth > 0.0:
                preview_wavelengths = [reference - bandwidth / 2.0, reference, reference + bandwidth / 2.0]
            preview_rows = []
            for wavelength in preview_wavelengths:
                if wavelength <= 0.0:
                    continue
                local, _ = seed_config_at_wavelength(cfg["seed"], wavelength)
                preview_rows.append({
                    "λ (nm)": wavelength * 1e9,
                    "wx/wy (µm)": f"{local['waist_x_m'] * 1e6:.5g} / {local['waist_y_m'] * 1e6:.5g}",
                    "focus x/y (mm)": f"{local['focus_x_m'] * 1e3:.5g} / {local['focus_y_m'] * 1e3:.5g}",
                    "center x/y (µm)": f"{local['center_x_m'] * 1e6:.5g} / {local['center_y_m'] * 1e6:.5g}",
                    "angle x/y (deg)": f"{local['incidence_angle_x_deg']:.5g} / {local['incidence_angle_y_deg']:.5g}",
                })
            st.dataframe(preview_rows, width="stretch", hide_index=True)


def _render_axisymmetric_grid_controls(cfg: dict[str, Any], *, spectral_time: bool = False) -> None:
    """Edit the shared radial geometry at the active backend's own entry."""
    coherent = cfg["seed"]["coherent"]
    axisymmetric = coherent.setdefault("axisymmetric", {})
    azimuthal_options = [
        "m0",
        "m0_m1",
        "m0_m1_m2_quadratic",
        "m0_m1_m2",
    ]
    axisymmetric["azimuthal_fidelity"] = choice_selectbox(
        "轴对称快速模型的角向保真度",
        azimuthal_options,
        index=option_index(
            azimuthal_options,
            str(axisymmetric.get("azimuthal_fidelity", "m0")),
        ),
        key=form_widget_key("ui_axisymmetric_azimuthal_fidelity"),
        disabled=(not spectral_time and cfg["seed"].get("spatial_modes", {}).get("model") == "incoherent_hg"
                  and cfg["seed"]["spatial_modes"].get("maximum_total_order") == 1),
        format_func=lambda value: {
            "m0": "基础轴对称 — 仅 m=0",
            "m0_m1": "m=0+1 — 保留偏心、倾角与热棱镜",
            "m0_m1_m2_quadratic": (
                "m=0+1+廉价 m=2 — 材料二阶项采用 r² 二次型"
            ),
            "m0_m1_m2": (
                "m=0+1+完整 m=2 — 保留二阶项的完整径向分布"
            ),
        }[value],
        help=(
            "这是晶体内部的有限角向 Fourier 模型。m=1 是偶极项，可描述"
            "质心/角度漂移和热棱镜；m=2 是四极项，可描述椭圆化、像散及"
            "旋转二次型。Pump 仍使用所选 Pump 后端；它产生的布居和温度场"
            "会在进入 Seed 快速求解器时投影到这里选择的阶数。"
        ),
    )
    selected_azimuthal_order = {
        "m0": 0,
        "m0_m1": 1,
        "m0_m1_m2_quadratic": 2,
        "m0_m1_m2": 2,
    }[axisymmetric["azimuthal_fidelity"]]
    if not spectral_time and axisymmetric["azimuthal_fidelity"] == "m0_m1_m2":
        axisymmetric["gain_spectral_bins"] = int(
            coherent.get("sample_count", 256)
        )
    axisymmetric["radial_bins"] = int(st.number_input(
        "径向格点数", min_value=8,
        value=int(axisymmetric.get("radial_bins", 24)), step=1,
        key=form_widget_key("ui_axisymmetric_radial_bins"),
        help="晶体内半径方向的离散数；与角向阶数独立。需结合光学窗口和横向网格做径向收敛检查。",
    ))
    if spectral_time:
        st.caption("各时间箱和波长箱分别传播空间场并共享布居耗尽；这里的 m 阶数只控制横向结构，不引入频率间相干、Kerr 或压缩相位。")
    elif selected_azimuthal_order == 0:
        st.caption(
            "仅传播 E(f,r)；所有非零角向阶数都会被投影掉。适合严格居中且"
            "热场/布居近似圆对称的基准。"
        )
    elif selected_azimuthal_order == 1:
        st.info(
            "晶体内传播 m=-1,0,+1 的复场及对应材料分量；可保留偏心和"
            "热棱镜，但不会保留 m=2 椭圆化。"
        )
    elif axisymmetric["azimuthal_fidelity"] == "m0_m1_m2_quadratic":
        st.info(
            "传播 m=-2…+2。材料场的 m=2(r) 会先最小二乘约化为 q·r²，"
            "光场自身产生的 m=2 仍按径向波动方程传播；这是受控的廉价像散闭合。"
        )
    else:
        st.info(
            "传播 m=-2…+2，并保留温度、布居和增益的完整 m=2(r) 径向函数；"
            "整脉冲增益默认使用全部相干频率点；开启尾部分箱时按初始能量合并尾部。仍会舍弃 "
            "|m|>2，需检查结果里的 omitted-order residual。"
        )
    with st.expander("CPU 径向/角向衍射加速", expanded=False):
        if selected_azimuthal_order > 0:
            minimum_angular_samples = 4 * selected_azimuthal_order + 1
            configured_angular_samples = max(
                minimum_angular_samples,
                int(axisymmetric.get("angular_samples", 12)),
            )
            axisymmetric["angular_samples"] = int(
                st.number_input(
                    "角向配点数",
                    min_value=minimum_angular_samples,
                    max_value=128,
                    value=configured_angular_samples,
                    step=1,
                    key=form_widget_key(
                        "ui_axisymmetric_angular_samples"
                    ),
                    help=(
                        "局域热相位和增益在这些 θ 配点上计算，再投影回保留阶数；此模型不计算 Kerr。提高配点数改善角向伪谱乘法，不会增加保留的 m 阶数。"
                        if spectral_time else
                        "局域热相位、增益和 Kerr 在这些 θ 配点上计算，再投影回"
                        "保留阶数。下限按三次 Kerr 去混叠设置；提高它只改善角向"
                        "伪谱乘法，不会增加保留的 m 阶数。"
                    ),
                )
            )
        radial_kernel_options = ["auto", "numba", "numpy"]
        axisymmetric["radial_diffraction_kernel"] = choice_selectbox(
            "径向 Crank–Nicolson 内核",
            radial_kernel_options,
            index=option_index(
                radial_kernel_options,
                str(
                    axisymmetric.get(
                        "radial_diffraction_kernel", "auto"
                    )
                ),
            ),
            key=form_widget_key(
                "ui_axisymmetric_radial_diffraction_kernel"
            ),
            help=(
                "auto 在工作量足够且安装 performance extra 时，用多核 Numba "
                "并行不同频率的独立径向三对角系统；径向递推、complex128 精度和"
                "算符耦合顺序保持不变。"
            ),
        )


def render_crystal_substep_controls(cfg: dict[str, Any]) -> None:
    """Edit local crystal propagation budgets beside the shared physical Nz."""

    coherent = cfg["seed"]["coherent"]
    local_operator = coherent.setdefault("local_operator", {})
    with st.expander("光学子步与增益子步（每个 Nz 单元）", expanded=False):
        st.caption("Nz 是晶体共享的物理轴向分片数；下面的子步设置控制每片内部的光学传播与增益更新，不增加布居或热场的物理网格点数。")
        local_operator["adaptive_substeps"] = choice_checkbox(
            "按局部物理变化自动增加子步",
            value=bool(local_operator.get("adaptive_substeps", True)),
            key=form_widget_key("ui_coherent_local_operator_adaptive"),
        )
        local_operator["maximum_substeps"] = int(
            st.number_input(
                "每个物理 z 单元最大局部子步",
                min_value=1,
                max_value=512,
                value=int(local_operator.get("maximum_substeps", 64)),
                step=1,
                key=form_widget_key("ui_coherent_local_operator_maximum_substeps"),
            )
        )
        local_operator["minimum_substeps"] = min(
            int(local_operator.get("minimum_substeps", 1)),
            int(local_operator["maximum_substeps"]),
        )
        limit_columns = st.columns(2)
        local_operator["maximum_gain_exponent_per_substep"] = limit_columns[0].number_input(
            "单子步最大 |g·ds|",
            min_value=0.001,
            max_value=2.0,
            value=float(local_operator.get("maximum_gain_exponent_per_substep", 0.1)),
            step=0.01,
            format="%.3f",
            key=form_widget_key("ui_coherent_local_operator_gain_limit"),
        )
        local_operator["maximum_nonlinear_phase_rad_per_substep"] = limit_columns[1].number_input(
            "单子步最大 Kerr 相位 (rad)",
            min_value=0.001,
            max_value=1.0,
            value=float(local_operator.get("maximum_nonlinear_phase_rad_per_substep", 0.05)),
            step=0.01,
            format="%.3f",
            key=form_widget_key("ui_coherent_local_operator_kerr_limit"),
        )
        limit_columns = st.columns(3)
        local_operator["maximum_dispersion_phase_rad_per_substep"] = limit_columns[0].number_input(
            "单子步最大色散相位跨度 (rad)",
            min_value=0.001,
            max_value=20.0,
            value=float(local_operator.get("maximum_dispersion_phase_rad_per_substep", 0.1)),
            step=0.01,
            format="%.3f",
            key=form_widget_key("ui_coherent_local_operator_dispersion_limit"),
        )
        local_operator["maximum_thermal_phase_rad_per_substep"] = limit_columns[1].number_input(
            "单子步最大热相位 PTV (rad)",
            min_value=0.001,
            max_value=2.0,
            value=float(local_operator.get("maximum_thermal_phase_rad_per_substep", 0.1)),
            step=0.01,
            format="%.3f",
            key=form_widget_key("ui_coherent_local_operator_thermal_limit"),
        )
        local_operator["maximum_fractional_loss_per_substep"] = limit_columns[2].number_input(
            "单子步最大强场损耗比例",
            min_value=0.001,
            max_value=0.5,
            value=float(local_operator.get("maximum_fractional_loss_per_substep", 0.05)),
            step=0.01,
            format="%.3f",
            key=form_widget_key("ui_coherent_local_operator_loss_limit"),
        )
        multirate_gain = local_operator.setdefault("multirate_gain", {})
        transient_active = bool(coherent.get("transient_gain", {}).get("enabled", False))
        population_policies = ["shared_cell"] if transient_active else ["shared_cell", "independent_groups"]
        if transient_active:
            local_operator["spatial_population_update"] = "shared_cell"
            st.session_state[form_widget_key("ui_coherent_spatial_population_update")] = "shared_cell"
            st.caption("Front-to-tail：每次增益调用内部依次处理全部时间窗。光学与增益上限均为 1 时，每个 Nz 单元各执行一次；时间窗数单独设置。")
        local_operator["spatial_population_update"] = choice_selectbox(
            "空间增益小片的粒子数更新",
            population_policies,
            index=population_policies.index(local_operator.get("spatial_population_update", "shared_cell")),
            format_func=lambda value: {
                "shared_cell": "共享片：各子步更新同一布居",
                "independent_groups": "独立小片：局部消耗，整片结束后汇总",
            }[value],
            key=form_widget_key("ui_coherent_spatial_population_update"),
            help="独立小片从同一父片的脉冲前密度出发，分别计算饱和与频率竞争，最后按体积平均。适用于整脉冲增益模式。",
        )
        multirate_gain["enabled"] = choice_checkbox(
            "增益多速率更新（整脉冲 / 瞬态）",
            value=bool(multirate_gain.get("enabled", False)),
            key=form_widget_key("ui_coherent_multirate_gain_enabled"),
            help=(
                "色散、Kerr、热相位和衍射仍按光学细子步推进；整脉冲或软窗增益可按增益阈值合并调用。"
                "指定更新次数时按该上限分组：整脉冲使用独立小片，瞬态使用共享布居，不再使用下方自动阈值。"
                "脉内弛豫和自由载流子不支持次数上限；纯多光子吸收可以使用。需对增益次数和光学子步分别做收敛比较。"
            ),
        )
        if multirate_gain["enabled"]:
            capped_updates = choice_checkbox(
                "限制每个物理片的增益更新次数",
                value=multirate_gain.get("maximum_updates_per_cell") is not None,
                key=form_widget_key("ui_coherent_gain_update_count_cap"),
                help="光学 N 步时，增益计算 min(N, 上限) 次；每次瞬态增益包含全部时间窗。整脉冲使用独立小片，瞬态使用共享片；脉内弛豫和自由载流子需关闭。",
            )
            multirate_gain["maximum_updates_per_cell"] = (
                int(st.number_input(
                    "每个物理片最多增益更新次数",
                    min_value=1, max_value=512,
                    value=int(multirate_gain.get("maximum_updates_per_cell") or 2),
                    step=1, key=form_widget_key("ui_coherent_gain_maximum_updates_per_cell"),
                )) if capped_updates else None
            )
            multirate_columns = st.columns(2)
            multirate_gain[
                "maximum_gain_exponent_per_update"
            ] = multirate_columns[0].number_input(
                "每次增益更新最大 |g·ds|",
                min_value=0.001,
                max_value=0.5,
                value=float(
                    multirate_gain.get(
                        "maximum_gain_exponent_per_update",
                        0.05,
                    )
                ),
                step=0.005,
                format="%.3f",
                key=form_widget_key(
                    "ui_coherent_multirate_gain_exponent_limit"
                ),
                disabled=capped_updates,
            )
            multirate_gain[
                "maximum_optical_substeps_per_update"
            ] = int(
                multirate_columns[1].number_input(
                    "每次增益更新最多合并光学子步",
                    min_value=1,
                    max_value=32,
                    value=int(
                        multirate_gain.get(
                            "maximum_optical_substeps_per_update",
                            4,
                        )
                    ),
                    step=1,
                    key=form_widget_key(
                        "ui_coherent_multirate_gain_maximum_group"
                    ),
                    disabled=capped_updates,
                )
            )
        else:
            multirate_gain["maximum_updates_per_cell"] = None


def render_seed_coherent_controls(cfg: dict[str, Any]) -> None:
    """Render complex-envelope controls shared by CPA and Commercial App effective-area."""

    st.caption(
        "相干 CPA 与 Commercial App effective-area 模式使用复光谱场；源相位、展宽器和压缩器相位相加。"
        "强度-only 相位恢复不是唯一解。"
    )
    coherent = cfg["seed"]["coherent"]
    coherent_sample_value = int(coherent.get("sample_count", 128))
    # The solver contract has no 4096-point ceiling.  Keep that as the normal
    # form range, but never reject a larger configuration that already passed
    # JSON validation (for example the 5120-point experimental pulse grid).
    coherent["sample_count"] = int(
        st.number_input(
            "相干频率采样",
            16,
            max(4096, coherent_sample_value),
            coherent_sample_value,
            step=16,
            key=form_widget_key("ui_coherent_samples"),
            help=(
                "对固定范围的输入光谱，增大此数值会保持时间步长并扩展周期时间窗。"
                "若结果出现 fft_temporal_boundary，请增加这里，而不是“时间求解分箱”；"
                "Commercial App effective-area 只增加一维 FFT，coherent CPA 还会增加三维复场内存。"
            ),
        )
    )
    coherent["span_fwhm"] = st.number_input(
        "相干频率窗口（×光谱 FWHM）",
        2.0,
        20.0,
        float(coherent.get("span_fwhm", 6.0)),
        step=0.5,
        key=form_widget_key("ui_coherent_span"),
    )
    coherent["failure_policy"] = choice_selectbox(
        "相干模块失败时",
        ["fallback", "error"],
        index=0 if coherent.get("failure_policy", "error") == "fallback" else 1,
        key=form_widget_key("ui_coherent_failure_policy"),
        help=(
            "fallback 会回滚本次 Seed 对布居的修改，并以多色×时间强度模型重算。"
            "初始化时就选定的 fallback 可在整次运行中保持一致；孤立脉冲也可返回运行期降阶结果。"
            "但脉冲串、周期图或热光固定点进行到一半才更换后端会改变状态映射，"
            "因此不得计入收敛，必要时会直接终止该有状态闭合。error 用于严格调试。"
        ),
    )
    if resolve_signal_model(cfg) == "coherent_cpa" and str(coherent.get("fidelity_mode", "configured")) in {
        "axisymmetric_kerr", "axisymmetric_transient_gain",
    }:
        _render_axisymmetric_grid_controls(cfg)
    transient_gain = coherent["transient_gain"]
    transient_gain["enabled"] = choice_checkbox(
        "逐时间窗瞬态增益（front → peak → tail）",
        value=bool(transient_gain.get("enabled", False)),
        key=form_widget_key("ui_transient_gain_enabled"),
        disabled=(cfg["seed"].get("spatial_modes", {}).get("model") == "incoherent_hg"),
        help="保持复场相位；按时间顺序处理重叠软窗，每个窗在频域使用波长相关截面并更新能级布居。",
    )
    fidelity_mode = str(coherent.get("fidelity_mode", "configured"))
    if transient_gain["enabled"] and fidelity_mode == "hybrid_whole_fluence":
        coherent["fidelity_mode"] = "reference_transient_gain"
    elif transient_gain["enabled"] and fidelity_mode == "axisymmetric_kerr":
        coherent["fidelity_mode"] = "axisymmetric_transient_gain"
        coherent.setdefault("local_operator", {})["spatial_population_update"] = "shared_cell"
        periodic = cfg["simulation"].setdefault("periodic_steady_state", {})
        if bool(periodic.get("enabled", False)):
            periodic["method"] = "pulse_map"
            cfg["physics"]["population_dynamics"] = "transient"
    elif (
        not transient_gain["enabled"]
        and fidelity_mode == "reference_transient_gain"
    ):
        coherent["fidelity_mode"] = "configured"
    elif (
        not transient_gain["enabled"]
        and fidelity_mode == "axisymmetric_transient_gain"
    ):
        coherent["fidelity_mode"] = "axisymmetric_kerr"
    if coherent.get("fidelity_mode") == "hybrid_whole_fluence":
        st.info(
            "全场 Kerr 中间档已启用：当前关闭逐软窗增益，整脉冲 FN 使用受控"
            "多速率更新，但仍传播完整 E(f,y,x)；这不是 Commercial App 高斯投影。"
        )
    elif coherent.get("fidelity_mode") in {
        "axisymmetric_kerr",
        "axisymmetric_transient_gain",
    }:
        configured_azimuthal = str(
            coherent.get("axisymmetric", {}).get(
                "azimuthal_fidelity", "m0"
            )
        )
        transient_axisymmetric = (
            coherent.get("fidelity_mode") == "axisymmetric_transient_gain"
        )
        if configured_azimuthal == "m0" and transient_axisymmetric:
            st.info(
                "L4.5 基础轴对称瞬态 Kerr 已启用：晶体内传播 E(f,r)，保留径向衍射、"
                "色散、局域时间域 Kerr，以及 front → peak → tail 软窗增益；不保留非零"
                "方位角模式，因此必须检查结果中的投影、边界和材料场残差门限。"
            )
        elif configured_azimuthal == "m0":
            st.info(
                "L4 基础轴对称整脉冲 Kerr 已启用：晶体内传播 E(f,r)，保留径向衍射、"
                "色散和局域时间域 Kerr；增益使用整脉冲 FN 闭合，非零方位角模式被省略。"
            )
        elif transient_axisymmetric:
            st.info(
                "L4.5 有限角向瞬态 Kerr 已启用：晶体内传播 E(f,r,θ)，径向衍射会按"
                "每个保留的 m 阶分别计算；色散、Kerr 和 front → peak → tail 软窗增益"
                "也作用于这些角向自由度。它仍不是完整 x-y 场，必须检查 omitted-order "
                "residual、入口投影和径向边界门限。"
            )
        else:
            st.info(
                "L4 有限角向整脉冲 Kerr 已启用：晶体内传播 E(f,r,θ)，径向衍射会按"
                "每个保留的 m 阶分别计算，色散和 Kerr 也作用于这些角向自由度；增益"
                "使用整脉冲 FN 闭合。它仍不是完整 x-y 场，必须检查所有降阶门限。"
            )
    if transient_gain["enabled"]:
        transient_gain["method"] = "soft_window_spectral"
        coherent_sample_count = int(coherent["sample_count"])
        transient_gain["tail_protection_enabled"] = choice_checkbox(
            "时间软窗尾部保护（两侧各 1 窗）",
            value=bool(transient_gain.get("tail_protection_enabled", False)),
            key=form_widget_key("ui_transient_tail_protection"),
            help="按首次增益输入的累计时间能量选中心区；中心细分，两侧尾部各保留一个完整软窗。不是按峰值高度截断。",
        )
        if transient_gain["tail_protection_enabled"]:
            transient_gain["tail_energy_coverage"] = float(st.number_input(
                "中心累计能量覆盖率 (%)", min_value=0.001, max_value=100.0,
                value=100.0 * float(transient_gain.get("tail_energy_coverage", .99)),
                step=0.1, format="%.3f", key=form_widget_key("ui_transient_tail_coverage"),
            )) / 100.0
            total_windows = int(st.number_input(
                "软时间窗总数上限（含两侧尾窗）", min_value=3, max_value=coherent_sample_count,
                value=min(coherent_sample_count, max(3, int(transient_gain.get("slice_count", 9)))),
                step=1, key=form_widget_key("ui_transient_protected_window_count"),
            ))
            transient_gain["slice_count"] = total_windows
            transient_gain["maximum_slice_count"] = total_windows
            transient_gain["adaptive_slices"] = False
            st.caption(f"左尾 1 窗 + 中心 {total_windows - 2} 窗 + 右尾 1 窗，总计 {total_windows} 窗；固定总数，不自动加密。尾部不丢弃，完整 FFT 保留；每次脉冲传播内固定区间。")
        else:
            transient_slice_key = form_widget_key("ui_transient_slice_count")
            transient_slice_value = min(
                max(
                    int(
                        st.session_state.get(
                            transient_slice_key,
                            transient_gain.get("slice_count", 9),
                        )
                    ),
                    2,
                ),
                coherent_sample_count,
            )
            st.session_state[transient_slice_key] = transient_slice_value
            transient_gain["slice_count"] = int(
                st.number_input(
                    "初始软时间窗数",
                    min_value=2,
                    max_value=coherent_sample_count,
                    step=1,
                    key=transient_slice_key,
                )
            )
            transient_gain["adaptive_slices"] = choice_checkbox(
                "按单窗布居耗尽自动加密",
                value=bool(transient_gain.get("adaptive_slices", True)),
                key=form_widget_key("ui_transient_adaptive"),
            )
            if transient_gain["adaptive_slices"]:
                transient_gain["reuse_adaptive_slice_floor"] = choice_checkbox(
                    "复用同一 z 层已收敛的软窗下限",
                    value=bool(
                        transient_gain.get("reuse_adaptive_slice_floor", True)
                    ),
                    key=form_widget_key("ui_transient_reuse_adaptive_floor"),
                    help=(
                        "后续增益步从本次传播中已通过/达到上限的窗数开始，仍会重新检查耗尽并继续加密；"
                        "只省去已知失败的低窗数重试。"
                    ),
                )
            transient_maximum_key = form_widget_key("ui_transient_max_slices")
            transient_maximum_value = min(
                max(
                    int(
                        st.session_state.get(
                            transient_maximum_key,
                            transient_gain.get("maximum_slice_count", 65),
                        )
                    ),
                    int(transient_gain["slice_count"]),
                ),
                coherent_sample_count,
            )
            st.session_state[transient_maximum_key] = transient_maximum_value
            transient_gain["maximum_slice_count"] = int(
                st.number_input(
                    "最大软时间窗数",
                    min_value=int(transient_gain["slice_count"]),
                    max_value=coherent_sample_count,
                    step=1,
                    key=transient_maximum_key,
                )
            )
        transient_gain["maximum_depletion_per_slice"] = st.number_input(
            "目标：单窗最大增益库耗尽比例",
            min_value=0.001,
            max_value=0.5,
            value=float(transient_gain.get("maximum_depletion_per_slice", 0.05)),
            step=0.005,
            format="%.3f",
            key=form_widget_key("ui_transient_max_depletion"),
            help="开启自动加密时按此阈值增加窗数；固定窗数时仍记录是否达标，但不突破上限。",
        )
        transient_gain["include_relaxation_during_pulse"] = choice_checkbox(
            "脉冲内部同时推进能级弛豫/上转换",
            value=bool(transient_gain.get("include_relaxation_during_pulse", False)),
            key=form_widget_key("ui_transient_relaxation"),
            help="超短脉冲通常可关闭；当脉宽接近能级寿命或自定义非线性过程时间尺度时再打开。",
        )
        if transient_gain["include_relaxation_during_pulse"]:
            transient_gain["relaxation_energy_fraction"] = st.number_input(
                "弛豫采用的脉冲能量支持比例",
                min_value=0.000001,
                max_value=1.0,
                value=float(
                    transient_gain.get("relaxation_energy_fraction", 0.999999)
                ),
                step=0.000001,
                format="%.6f",
                key=form_widget_key("ui_transient_relaxation_energy_fraction"),
                help=(
                    "用实际时间域能量的对称分位区间定义物理脉冲时长，避免把 FFT "
                    "两侧零填充误算成额外暗弛豫。通常保持默认值。"
                ),
            )
        transient_gain["failure_policy"] = choice_selectbox(
            "瞬态增益子模块失败时",
            ["fallback", "error"],
            index=0 if transient_gain.get("failure_policy", "error") == "fallback" else 1,
            key=form_widget_key("ui_transient_failure_policy"),
            help="fallback 会回滚整次 Seed 传播并用总通量增益重新运行，仍保留相干场、CPA、色散与 Kerr。",
        )
        with st.expander("CPU 瞬态增益加速", expanded=False):
            transient_gain[
                "maximum_bypassed_output_energy_fraction"
            ] = st.number_input(
                "低能空窗旁路：每个增益步最大输出能量预算",
                min_value=0.0,
                max_value=1e-3,
                value=float(
                    transient_gain.get(
                        "maximum_bypassed_output_energy_fraction",
                        0.0,
                    )
                ),
                step=1e-7,
                format="%.1e",
                key=form_widget_key(
                    "ui_transient_inactive_window_output_budget"
                ),
                help=(
                    "0 为完全关闭。大于 0 时，只有预计放大后的全局能量和每个材料像素局部能量都低于该预算，"
                    "数值空尾窗才会旁路；活跃窗仍使用完整频率网格和原人口顺序。空间规模超过保护上限、"
                    "脉内弛豫或不受支持的多跃迁耦合都会自动恢复完整计算。"
                ),
            )
            transient_gain["batch_window_ffts"] = choice_checkbox(
                "批量执行软窗 FFT",
                value=bool(transient_gain.get("batch_window_ffts", False)),
                key=form_widget_key("ui_transient_batch_window_ffts"),
                help=(
                    "保持窗口和人口更新顺序不变；只合并 FFT 调度。估算峰值内存超出预算时自动退回逐窗 FFT。"
                ),
            )
            if transient_gain["batch_window_ffts"]:
                transient_gain["batch_window_fft_max_bytes"] = int(
                    st.number_input(
                        "批量 FFT 临时内存预算 (MiB)",
                        min_value=32,
                        max_value=16384,
                        value=max(
                            32,
                            int(
                                transient_gain.get(
                                    "batch_window_fft_max_bytes",
                                    268435456,
                                )
                                // (1024**2)
                            ),
                        ),
                        step=32,
                        key=form_widget_key(
                            "ui_transient_batch_window_fft_max_mib"
                        ),
                    )
                    * 1024**2
                )
            kernel_options = ["auto", "numba", "numpy"]
            transient_gain["single_transition_kernel"] = choice_selectbox(
                "单跃迁人口内核",
                kernel_options,
                index=option_index(
                    kernel_options,
                    str(
                        transient_gain.get(
                            "single_transition_kernel", "auto"
                        )
                    ),
                ),
                key=form_widget_key("ui_transient_single_transition_kernel"),
                help="auto 按工作量选择；Numba 未安装时自动保留 NumPy 路径。",
            )
        st.caption(
            "这是速率方程/SVEA 软窗算法，不另存独立 Nt×Nλ 强度表；"
            "它包含波长相关饱和，但不是 Maxwell–Bloch 材料相干模型。"
        )
        st.caption("软窗耗尽阈值和能量守恒不保证波形收敛；定量脉宽、峰值、光谱或压缩结果还须独立增加软窗数检查。")

    linked_domain = str(cfg["seed"].get("pulse_definition", {}).get("domain", "time"))
    override_with_complex_spectrum = choice_checkbox(
        "高级：用复光谱文件覆盖上面的统一强度定义",
        value=str(coherent.get("input_mode", "coefficients")) == "complex_spectrum",
        key=form_widget_key("ui_override_complex_spectrum"),
        help="通常不需要。上面的强度文件第三列或独立相位文件已可输入相位；这里保留 real/imag 复场兼容入口。",
    )
    coherent["input_mode"] = (
        "complex_spectrum"
        if override_with_complex_spectrum
        else "phase_retrieval"
        if linked_domain == "dual"
        else "coefficients"
    )
    st.caption("源 GDD/TOD、展宽器相位和独立光谱相位位于上面的统一 Seed 定义区。")
    coherent["compressor"] = phase_coefficients_ui(
        "压缩器",
        "compressor",
        cfg["seed"]["wavelength_m"],
        coherent.get("compressor"),
    )
    if coherent["input_mode"] == "complex_spectrum":
        complex_formats = ["amplitude_phase", "real_imag"]
        complex_format = choice_selectbox(
            "复光谱文件格式",
            complex_formats,
            index=option_index(
                complex_formats,
                str(coherent["complex_spectrum"].get("format", "amplitude_phase")),
            ),
            key=form_widget_key("ui_complex_spectrum_format"),
        )
        coherent["complex_spectrum"]["format"] = complex_format
        amplitude_domains = ["per_frequency", "per_wavelength", "per_bin"]
        coherent["complex_spectrum"]["amplitude_domain"] = choice_selectbox(
            "复振幅密度定义",
            amplitude_domains,
            index=option_index(
                amplitude_domains,
                str(
                    coherent["complex_spectrum"].get(
                        "amplitude_domain", "per_frequency"
                    )
                ),
            ),
            key=form_widget_key("ui_complex_spectrum_amplitude_domain"),
            help=(
                "per_frequency 表示每 Hz 的复振幅；per_wavelength 表示每单位波长，"
                "程序会乘 λ/√c 转成频率域；per_bin 表示每个原始采样格的复振幅，"
                "程序会按该格的频率宽度换算。此选择影响宽带谱形与时间波形。"
            ),
        )
        complex_file = st.file_uploader(
            "复光谱 CSV：λ(nm), amplitude, phase(rad) 或 real, imag",
            type=["csv", "txt"],
            key=form_widget_key("complex_seed_spectrum"),
        )
        # The uploader is an optional replacement for samples already embedded
        # in the JSON configuration.  Clearing the table on every rerender made
        # valid FROG configurations fail unless the user uploaded the same data
        # a second time.
        if complex_file is not None:
            coherent["complex_spectrum"]["samples"] = read_rows(complex_file, 3)
            coherent["complex_spectrum"]["wavelength_unit"] = "nm"
    if (
        cfg["seed"].get("pulse_definition", {}).get("domain") == "spectrum"
        and cfg["seed"]["spectrum"].get("type") != "file"
        and cfg["seed"]["spectrum"].get("bandwidth_fwhm_m", 0.0) <= 0.0
    ):
        st.warning("未给有限带宽时，会根据 pulse_duration_s 推断高斯变换极限带宽；定量 CPA 建议输入实测带宽或光谱文件。")


def render_seed_path_controls(cfg: dict[str, Any]) -> None:
    """Render the Seed-only optical bench and radius probe."""

    st.caption("只影响 Seed 的入射与出射光路；多次晶体命中和程间元件统一在下方“多程”中配置。")
    seed_path_rows = render_optical_path_editor(cfg, "seed")
    cfg["optical_train"]["seed"] = seed_path_rows
    render_beam_radius_probe(cfg, "seed")


def _multipass_number(
    section: dict[str, Any],
    field: str,
    label: str,
    *,
    key: str,
    default: float = 0.0,
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
    scientific: bool = True,
    help_text: str | None = None,
) -> float:
    kwargs: dict[str, Any] = {
        "value": float(section.get(field, default)),
        "key": form_widget_key(key),
        "help": help_text,
    }
    if minimum is not None:
        kwargs["min_value"] = float(minimum)
    if maximum is not None:
        kwargs["max_value"] = float(maximum)
    if step is not None:
        kwargs["step"] = float(step)
    if scientific:
        kwargs["format"] = "%.6e"
    value = float(st.number_input(label, **kwargs))
    section[field] = value
    return value


def _render_multipass_mirror(
    section: dict[str, Any],
    *,
    prefix: str,
    label: str,
    key_prefix: str,
    distance_field: str | None = None,
) -> None:
    st.markdown(f"##### {label}")
    first_row = st.columns(3)
    if distance_field is not None:
        with first_row[0]:
            _multipass_number(
                section,
                distance_field,
                "晶体端面 → 镜面距离 (m)",
                key=f"{key_prefix}_distance",
                minimum=0.0,
            )
    with first_row[1 if distance_field is not None else 0]:
        _multipass_number(
            section,
            f"{prefix}_mirror_power_reflectivity",
            "功率反射率",
            key=f"{key_prefix}_reflectivity",
            default=1.0,
            minimum=0.0,
            maximum=1.0,
            step=0.001,
            scientific=False,
        )
    curvature_columns = st.columns(2)
    with curvature_columns[0]:
        _multipass_number(
            section,
            f"{prefix}_mirror_radius_x_m",
            "x 曲率半径 Rx (m；0=平面)",
            key=f"{key_prefix}_radius_x",
        )
    with curvature_columns[1]:
        _multipass_number(
            section,
            f"{prefix}_mirror_radius_y_m",
            "y 曲率半径 Ry (m；0=平面)",
            key=f"{key_prefix}_radius_y",
        )
    center_columns = st.columns(2)
    with center_columns[0]:
        _multipass_number(
            section,
            f"{prefix}_mirror_center_x_m",
            "镜面中心 x (m)",
            key=f"{key_prefix}_center_x",
        )
    with center_columns[1]:
        _multipass_number(
            section,
            f"{prefix}_mirror_center_y_m",
            "镜面中心 y (m)",
            key=f"{key_prefix}_center_y",
        )
    aperture_columns = st.columns(2)
    with aperture_columns[0]:
        _multipass_number(
            section,
            f"{prefix}_mirror_aperture_radius_x_m",
            "有效口径半径 ax (m；0=不限)",
            key=f"{key_prefix}_aperture_x",
            minimum=0.0,
        )
    with aperture_columns[1]:
        _multipass_number(
            section,
            f"{prefix}_mirror_aperture_radius_y_m",
            "有效口径半径 ay (m；0=不限)",
            key=f"{key_prefix}_aperture_y",
            minimum=0.0,
        )


def render_causal_multipass_controls(cfg: dict[str, Any]) -> None:
    """Render only the selected finite multipass template and its inputs."""

    simulation = cfg["simulation"]
    multipass = simulation.setdefault(
        "multipass", copy.deepcopy(DEFAULTS["simulation"]["multipass"])
    )
    st.success(
        "重叠没有开关：程序由脉冲包络、晶体群速度、每段外部光程和每程二维命中位置"
        "自动判断。时间重叠但光斑不相交时，也不会产生假的局部增益竞争。"
    )
    template_labels = {
        "双程返回镜 — 晶体后单镜折返": "double_pass",
        "双侧镜有限多程 — 理想开关或几何出射": "two_mirror",
        "Herriott 两曲面镜 — 旁轴光斑轨道": "herriott",
        "4f / 中继成像 — 控制每程束腰演化": "relay_4f",
        "论文逐程边界条件 — 直接给命中/延时": "prescribed",
    }
    current_template = str(multipass.get("template", "double_pass"))
    if current_template == "sequential":
        template_labels[
            "已迁移的快速顺序多程 — 非时域后端降阶执行"
        ] = "sequential"
    selected_label = choice_selectbox(
        "多程模板",
        list(template_labels),
        index=(
            list(template_labels.values()).index(current_template)
            if current_template in template_labels.values()
            else 0
        ),
        key=form_widget_key("ui_multipass_template"),
        help=(
            "模板只生成晶体命中序列和程间算子，不要求在三维画布中摆放镜架。"
            "论文模式用于文献没有给完整镜路、只给逐程条件的情况。"
        ),
    )
    template = template_labels[selected_label]
    multipass["template"] = template

    if template == "double_pass":
        multipass["pass_count"] = 2
        st.caption("双程模板固定为 2 次晶体命中；D=0 只是镜面贴端面的极限。")
    else:
        multipass["pass_count"] = int(
            st.number_input(
                "晶体命中次数",
                min_value=2,
                max_value=100,
                value=int(multipass.get("pass_count", 4)),
                step=1,
                key=form_widget_key("ui_multipass_pass_count"),
            )
        )
    simulation["passes"] = int(multipass["pass_count"])
    if str(multipass.get("execution_mode", "auto")) != "sequential":
        simulation["signal_model"] = "spectral_time"
        cfg["seed"].setdefault("temporal", {})["resolved"] = True

    if template == "double_pass":
        section = multipass.setdefault(
            "double_pass",
            copy.deepcopy(DEFAULTS["simulation"]["multipass"]["double_pass"]),
        )
        mirror_columns = st.columns(3)
        with mirror_columns[0]:
            _multipass_number(
                section,
                "mirror_distance_m",
                "晶体后端面 → 返回镜 D (m)",
                key="ui_mp_double_distance",
                minimum=0.0,
            )
        with mirror_columns[1]:
            _multipass_number(
                section,
                "mirror_power_reflectivity",
                "返回镜功率反射率",
                key="ui_mp_double_reflectivity",
                default=1.0,
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                scientific=False,
            )
        curvature_columns = st.columns(2)
        with curvature_columns[0]:
            _multipass_number(
                section,
                "mirror_radius_x_m",
                "返回镜 Rx (m；0=平面)",
                key="ui_mp_double_rx",
            )
        with curvature_columns[1]:
            _multipass_number(
                section,
                "mirror_radius_y_m",
                "返回镜 Ry (m；0=平面)",
                key="ui_mp_double_ry",
            )
        center_columns = st.columns(2)
        with center_columns[0]:
            _multipass_number(
                section,
                "mirror_center_x_m",
                "返回镜中心 x (m)",
                key="ui_mp_double_mirror_cx",
            )
        with center_columns[1]:
            _multipass_number(
                section,
                "mirror_center_y_m",
                "返回镜中心 y (m)",
                key="ui_mp_double_mirror_cy",
            )
        aperture_columns = st.columns(2)
        with aperture_columns[0]:
            _multipass_number(
                section,
                "mirror_aperture_radius_x_m",
                "返回镜有效半口径 ax (m；0=不限)",
                key="ui_mp_double_ax",
                minimum=0.0,
            )
        with aperture_columns[1]:
            _multipass_number(
                section,
                "mirror_aperture_radius_y_m",
                "返回镜有效半口径 ay (m；0=不限)",
                key="ui_mp_double_ay",
                minimum=0.0,
            )
        st.markdown("##### 两次晶体命中边界")
        hit_columns = st.columns(4)
        for column, field, label, key in (
            (hit_columns[0], "first_center_x_m", "首程中心 x (m)", "ui_mp_double_first_cx"),
            (hit_columns[1], "first_center_y_m", "首程中心 y (m)", "ui_mp_double_first_cy"),
            (hit_columns[2], "return_center_x_m", "回程中心 x (m)", "ui_mp_double_return_cx"),
            (hit_columns[3], "return_center_y_m", "回程中心 y (m)", "ui_mp_double_return_cy"),
        ):
            with column:
                _multipass_number(section, field, label, key=key)
        angle_columns = st.columns(4)
        for column, field, label, key in (
            (angle_columns[0], "first_incidence_angle_x_deg", "首程 θx (°)", "ui_mp_double_first_ax"),
            (angle_columns[1], "first_incidence_angle_y_deg", "首程 θy (°)", "ui_mp_double_first_ay"),
            (angle_columns[2], "return_incidence_angle_x_deg", "回程局部 θx (°)", "ui_mp_double_return_ax"),
            (angle_columns[3], "return_incidence_angle_y_deg", "回程局部 θy (°)", "ui_mp_double_return_ay"),
        ):
            with column:
                _multipass_number(
                    section, field, label, key=key, scientific=False, step=0.01
                )

    elif template in {"two_mirror", "herriott"}:
        defaults = DEFAULTS["simulation"]["multipass"][template]
        section = multipass.setdefault(template, copy.deepcopy(defaults))
        if template == "herriott":
            geometry_columns = st.columns(2)
            with geometry_columns[0]:
                _multipass_number(
                    section,
                    "mirror_separation_m",
                    "两镜顶点间距 L (m)",
                    key="ui_mp_herriott_separation",
                    default=0.2,
                    minimum=1e-12,
                )
            with geometry_columns[1]:
                _multipass_number(
                    section,
                    "crystal_position_from_left_m",
                    "晶体中心距左镜 (m)",
                    key="ui_mp_herriott_position",
                    default=0.1,
                    minimum=0.0,
                )
            section["geometric_output_after_pass"] = True
            st.caption(
                "有限程数按几何注入/出射孔解释；程序不会把一个无开关的被动闭腔"
                "假装成恰好 N 程停止。"
            )
            left_distance_field = None
            right_distance_field = None
        else:
            extraction_options = {
                "理想开关（Pockels/偏振门等效）": "ideal_switch",
                "几何分离出射（最后一程进入取出孔）": "geometric",
            }
            current_extraction = str(
                section.get("finite_pass_extraction", "ideal_switch")
            )
            extraction_label = choice_selectbox(
                "有限程数如何结束",
                list(extraction_options),
                index=(
                    list(extraction_options.values()).index(current_extraction)
                    if current_extraction in extraction_options.values()
                    else 0
                ),
                key=form_widget_key("ui_mp_two_mirror_extraction"),
                help=(
                    "两面被动镜本身不会在指定的第 N 程自动停止；这里必须声明实验"
                    "使用主动开关，还是依靠光斑/角度分离进入出射孔。"
                ),
            )
            section["finite_pass_extraction"] = extraction_options[
                extraction_label
            ]
            left_distance_field = "left_mirror_distance_m"
            right_distance_field = "right_mirror_distance_m"
        mirror_tabs = st.tabs(["左侧镜", "右侧镜"])
        with mirror_tabs[0]:
            _render_multipass_mirror(
                section,
                prefix="left",
                label="左侧镜参数",
                key_prefix=f"ui_mp_{template}_left",
                distance_field=left_distance_field,
            )
        with mirror_tabs[1]:
            _render_multipass_mirror(
                section,
                prefix="right",
                label="右侧镜参数",
                key_prefix=f"ui_mp_{template}_right",
                distance_field=right_distance_field,
            )
        st.markdown("##### 注入主光线（模板据此生成每程光斑轨道）")
        injection_columns = st.columns(4)
        for column, field, label, key in (
            (injection_columns[0], "injection_center_x_m", "注入中心 x (m)", f"ui_mp_{template}_inject_cx"),
            (injection_columns[1], "injection_center_y_m", "注入中心 y (m)", f"ui_mp_{template}_inject_cy"),
            (injection_columns[2], "injection_angle_x_deg", "注入 θx (°)", f"ui_mp_{template}_inject_ax"),
            (injection_columns[3], "injection_angle_y_deg", "注入 θy (°)", f"ui_mp_{template}_inject_ay"),
        ):
            with column:
                _multipass_number(
                    section,
                    field,
                    label,
                    key=key,
                    scientific=not field.endswith("_deg"),
                    step=0.01 if field.endswith("_deg") else None,
                )

    elif template == "relay_4f":
        section = multipass.setdefault(
            "relay_4f",
            copy.deepcopy(DEFAULTS["simulation"]["multipass"]["relay_4f"]),
        )
        st.caption(
            "使用对象面→F1→F2→镜面→F2→F1→对象面的往返中继算子。当前要求"
            "往返 ABCD 满足单位放大且 B=0；不满足时配置会明确报出矩阵。"
        )
        distance_columns = st.columns(5)
        for column, field, label, key in (
            (distance_columns[0], "object_to_first_lens_m", "对象面→F1 (m)", "ui_mp_4f_object"),
            (distance_columns[1], "first_focal_length_m", "F1 焦距 (m)", "ui_mp_4f_f1"),
            (distance_columns[2], "lens_separation_m", "F1→F2 (m)", "ui_mp_4f_sep"),
            (distance_columns[3], "second_focal_length_m", "F2 焦距 (m)", "ui_mp_4f_f2"),
            (distance_columns[4], "second_lens_to_mirror_m", "F2→镜面 (m)", "ui_mp_4f_mirror"),
        ):
            with column:
                _multipass_number(
                    section, field, label, key=key, default=0.1, minimum=1e-12
                )
        relay_columns = st.columns(3)
        with relay_columns[0]:
            _multipass_number(
                section,
                "relay_power_transmission",
                "每次完整往返中继功率透过率",
                key="ui_mp_4f_transmission",
                default=1.0,
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                scientific=False,
            )
        with relay_columns[1]:
            _multipass_number(
                section,
                "center_x_m",
                "中继光轴中心 x (m)",
                key="ui_mp_4f_center_x",
            )
        with relay_columns[2]:
            _multipass_number(
                section,
                "center_y_m",
                "中继光轴中心 y (m)",
                key="ui_mp_4f_center_y",
            )
        st.markdown("##### 每程取样位置/角度程序")
        program_columns = st.columns(4)
        for column, field, label, key in (
            (program_columns[0], "first_center_x_m", "首程中心 x (m)", "ui_mp_4f_first_cx"),
            (program_columns[1], "first_center_y_m", "首程中心 y (m)", "ui_mp_4f_first_cy"),
            (program_columns[2], "spot_step_x_m", "每程中心增量 Δx (m)", "ui_mp_4f_step_x"),
            (program_columns[3], "spot_step_y_m", "每程中心增量 Δy (m)", "ui_mp_4f_step_y"),
        ):
            with column:
                _multipass_number(section, field, label, key=key)
        angle_columns = st.columns(4)
        for column, field, label, key in (
            (angle_columns[0], "first_incidence_angle_x_deg", "首程 θx (°)", "ui_mp_4f_first_ax"),
            (angle_columns[1], "first_incidence_angle_y_deg", "首程 θy (°)", "ui_mp_4f_first_ay"),
            (angle_columns[2], "angle_step_x_deg", "每程 Δθx (°)", "ui_mp_4f_angle_step_x"),
            (angle_columns[3], "angle_step_y_deg", "每程 Δθy (°)", "ui_mp_4f_angle_step_y"),
        ):
            with column:
                _multipass_number(
                    section, field, label, key=key, scientific=False, step=0.01
                )

    elif template == "sequential":
        sequential = multipass.setdefault(
            "sequential",
            copy.deepcopy(DEFAULTS["simulation"]["multipass"]["sequential"]),
        )
        multipass["execution_mode"] = "sequential"
        st.info(
            "这个配置由旧快速逐程自动迁移而来。它已使用统一的多程命中/程间路由，"
            "但原始文件没有足够镜路信息，程序不会假造因果重叠；可继续用于 Commercial App、"
            "fluence 或 CPA 的完整逐程计算。新建物理多程请切换到上面的镜路模板。"
        )
        sequential["alternate_direction"] = choice_checkbox(
            "相邻程反向通过晶体",
            value=bool(sequential.get("alternate_direction", True)),
            key=form_widget_key("ui_mp_sequential_alternate"),
        )
        route_json = st.text_area(
            "程间光学元件 route JSON",
            value=json.dumps(
                sequential.get("route", []), ensure_ascii=False, indent=2
            ),
            key=form_widget_key("ui_mp_sequential_route"),
            height=220,
        )
        sequential["route"] = parse_json(route_json, "迁移多程程间光路")

    else:
        prescribed = multipass.setdefault("prescribed", {"hits": [], "transitions": []})
        st.warning(
            "这里不是反推论文没写出的镜面。hits 是每次晶体入口的方向/中心/角度；"
            "transitions 是上一程出晶体到下一程入晶体的光程、折返镜或额外延时。"
        )
        hits_json = st.text_area(
            "晶体命中 hits JSON",
            value=json.dumps(prescribed.get("hits", []), ensure_ascii=False, indent=2),
            key=form_widget_key("ui_mp_prescribed_hits"),
            height=220,
            help=(
                "每项可含 name、direction、center_x_m、center_y_m、"
                "incidence_angle_x_deg、incidence_angle_y_deg 和界面参数。"
            ),
        )
        transitions_json = st.text_area(
            "程间 transitions JSON",
            value=json.dumps(
                prescribed.get("transitions", []), ensure_ascii=False, indent=2
            ),
            key=form_widget_key("ui_mp_prescribed_transitions"),
            height=260,
            help=(
                "每项 route_model 可为 delay_only、free_space、round_trip_mirror 或 optical_train；"
                "再给 geometric_length_m、折射率、透过率、extra_delay_s，以及相应镜面参数或 route 元件数组。"
                "delay_only 会把几何长度计入因果时钟，但不会让采样场衍射或平移。"
            ),
        )
        prescribed["hits"] = parse_json(hits_json, "多程晶体命中")
        prescribed["transitions"] = parse_json(
            transitions_json, "多程程间路由"
        )
        if isinstance(prescribed["hits"], list) and len(prescribed["hits"]) >= 2:
            multipass["pass_count"] = len(prescribed["hits"])
            simulation["passes"] = len(prescribed["hits"])

    with st.expander("所有物理多程模板共用的时间、介质与数值设置", expanded=False):
        common_columns = st.columns(3)
        with common_columns[0]:
            _multipass_number(
                multipass,
                "per_transition_power_transmission",
                "每段程间附加功率透过率",
                key="ui_mp_common_transmission",
                default=1.0,
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                scientific=False,
                help_text="镜面/中继自身损耗之外的公共附加损耗。",
            )
        with common_columns[1]:
            _multipass_number(
                multipass,
                "external_refractive_index",
                "外部路径相位折射率",
                key="ui_mp_external_n",
                default=1.0,
                minimum=1e-6,
                step=0.01,
                scientific=False,
            )
        with common_columns[2]:
            _multipass_number(
                multipass,
                "external_group_refractive_index",
                "外部路径群折射率",
                key="ui_mp_external_ng",
                default=1.0,
                minimum=1e-6,
                step=0.01,
                scientific=False,
            )
        multipass["per_transition_extra_delay_s"] = float(
            st.number_input(
                "每段程间未被模板覆盖的附加延时 (s)",
                min_value=0.0,
                value=float(
                    multipass.get("per_transition_extra_delay_s", 0.0)
                ),
                format="%.6e",
                key=form_widget_key("ui_mp_extra_delay_s"),
                help="只填模板几何光程之外的电子/光纤等已知延时。",
            )
        )
        numeric_columns = st.columns(4)
        with numeric_columns[0]:
            multipass["maximum_time_bins"] = int(
                st.number_input(
                    "绝对时间最大分箱数",
                    min_value=3,
                    max_value=10_000_000,
                    value=int(multipass.get("maximum_time_bins", 20000)),
                    step=100,
                    key=form_widget_key("ui_mp_max_time_bins"),
                )
            )
        with numeric_columns[1]:
            _multipass_number(
                multipass,
                "maximum_field_memory_MB",
                "活动场内存上限 (MiB)",
                key="ui_mp_max_memory",
                default=1024.0,
                minimum=1.0,
                scientific=False,
            )
        with numeric_columns[2]:
            _multipass_number(
                multipass,
                "maximum_relative_group_index_span",
                "允许相对群折射率跨度",
                key="ui_mp_group_span",
                default=1e-3,
                minimum=0.0,
            )
        with numeric_columns[3]:
            _multipass_number(
                multipass,
                "maximum_relative_path_cosine_span",
                "允许相对 cosθ 跨度",
                key="ui_mp_cosine_span",
                default=1e-4,
                minimum=0.0,
                help_text=(
                    "不同程共用一个 z-CFL 时钟的显式误差界；超过即拒绝计算，"
                    "不会静默把大角度差当成相同速度。"
                ),
            )
        multipass["include_dark_relaxation"] = choice_checkbox(
            "在同一绝对时间轴上推进能级暗弛豫",
            value=bool(multipass.get("include_dark_relaxation", True)),
            key=form_widget_key("ui_mp_dark_relaxation"),
            help=(
                "开启时，所有程的飞行时间与能级寿命/暗跃迁共用同一时间账本；"
                "只有已证明完整多程历时远短于全部动力学时间尺度时才应关闭。"
            ),
        )
        _multipass_number(
            multipass,
            "maximum_dark_step_s",
            "暗弛豫最大时间步 (s；0=自动)",
            key="ui_mp_max_dark_step",
            minimum=0.0,
        )


def render_doping_profile_controls(cfg: dict[str, Any]) -> None:
    """Render a geometry-aware 3D dopant profile editor."""

    profile = cfg["material"].setdefault(
        "doping_profile", copy.deepcopy(DEFAULTS["material"]["doping_profile"])
    )
    with st.expander("三维掺杂浓度分布（可选）", expanded=bool(profile.get("enabled", False))):
        profile["enabled"] = choice_checkbox(
            "启用非均匀掺杂",
            key=form_widget_key("ui_doping_profile_enabled"),
            **widget_default("ui_doping_profile_enabled", bool(profile.get("enabled", False))),
        )
        normalization_labels = {
            "volume_average": "体积平均值 — 上面的离子密度保持为平均浓度（推荐）",
            "peak": "峰值 — 上面的离子密度作为最高浓度",
            "none": "不归一化 — 上面的离子密度乘原始相对分布",
        }
        current_normalization = str(profile.get("normalization", "volume_average"))
        normalization = choice_selectbox(
            "标定方式",
            list(normalization_labels),
            index=option_index(list(normalization_labels), current_normalization),
            format_func=lambda value: normalization_labels[value],
            key=form_widget_key("ui_doping_normalization"),
            disabled=not profile["enabled"],
        )
        profile["normalization"] = normalization
        if not profile["enabled"]:
            st.caption("关闭时严格恢复均匀离子密度，旧 JSON 和旧结果保持兼容。")
            return

        geometry_shape = str(cfg["material"].get("geometry", {}).get("shape", "cuboid"))
        st.caption(
            "只需选择变化方式、方向和带正负号的百分比。多个方向相乘，"
            "体积平均标定仍保持上方填写的总平均掺杂浓度。"
        )
        type_labels = {
            "axis_symmetric_linear": "两侧/外缘 ↔ 中心（正值中心高）",
            "axis_monotonic_linear": "单向：负坐标端 → 正坐标端",
        }
        direction_labels = {
            "x": "x 方向",
            "y": "y 方向",
            "z": "z 方向",
            "radial_xy": "径向（圆柱外圆周 ↔ 中心轴）",
        }
        type_from_label = {label: token for token, label in type_labels.items()}
        direction_from_label = {label: token for token, label in direction_labels.items()}
        simple_types = set(type_labels)
        legacy_components = [
            copy.deepcopy(component)
            for component in profile.get("components", [])
            if str(component.get("type", "surface_exponential")) not in simple_types
        ]
        display_rows = []
        for component in profile.get("components", []):
            kind = str(component.get("type", "axis_symmetric_linear"))
            if kind not in simple_types:
                continue
            direction = str(component.get("direction", "y"))
            display_rows.append(
                {
                    "分布形式": type_labels.get(kind, kind),
                    "作用方向": direction_labels.get(direction, direction),
                    "变化 (%)": 100.0 * float(component.get("relative_change", 0.0)),
                }
            )
        edited = st.data_editor(
            display_rows
            or [
                {
                    "分布形式": type_labels["axis_symmetric_linear"],
                    "作用方向": direction_labels["y"],
                    "变化 (%)": 0.0,
                }
            ],
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key=form_widget_key("ui_doping_components"),
            column_config={
                "分布形式": st.column_config.SelectboxColumn(
                    options=list(type_labels.values()), required=True
                ),
                "作用方向": st.column_config.SelectboxColumn(
                    options=list(direction_labels.values()),
                    required=True,
                ),
                "变化 (%)": st.column_config.NumberColumn(
                    min_value=-100.0, format="%.6g", required=True
                ),
            },
        )
        rows = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)
        components: list[dict[str, Any]] = list(legacy_components)
        for row in rows:
            kind_label = str(
                row.get("分布形式") or type_labels["axis_symmetric_linear"]
            )
            kind = type_from_label.get(kind_label, kind_label)
            direction_label = str(row.get("作用方向") or direction_labels["y"])
            direction = direction_from_label.get(direction_label, direction_label)
            if kind == "axis_monotonic_linear" and direction == "radial_xy":
                st.error("单向掺杂变化只支持 x、y 或 z；径向变化请选择两侧/外缘到中心。当前输入保留，修正前不能运行。")
            change_value = row.get("变化 (%)")
            change_percent = (
                0.0
                if change_value is None or not np.isfinite(float(change_value))
                else float(change_value)
            )
            components.append(
                {
                    "type": kind,
                    "direction": direction,
                    "relative_change": change_percent / 100.0,
                }
            )
        profile["components"] = components
        st.caption(
            "两侧/向心：+50% 表示中心比表面高 50%，−50% 表示中心比表面低 50%。"
            "单向：+50% 表示正坐标端比负坐标端高 50%，负值表示相反方向。"
        )
        if geometry_shape != "cylinder" and any(
            component.get("direction") == "radial_xy" for component in components
        ):
            st.error("径向/向心变化只适用于圆柱晶体；长方体请选择 x、y 或 z。")
        endcaps = profile.setdefault(
            "endcaps",
            copy.deepcopy(DEFAULTS["material"]["doping_profile"]["endcaps"]),
        )
        endcaps["enabled"] = choice_checkbox(
            "启用未掺杂/低掺杂端帽",
            value=bool(endcaps.get("enabled", False)),
            key=form_widget_key("ui_doping_endcaps_enabled"),
            help=(
                "端帽仍属于同一块基质和热传导域，只改变活性离子密度；"
                "适用于未掺杂端帽或分段掺杂晶体。"
            ),
        )
        if endcaps["enabled"]:
            endcap_columns = st.columns(3)
            with endcap_columns[0]:
                endcaps["front_length_m"] = sci(
                    "入口端帽长度 (m)",
                    endcaps.get("front_length_m", 0.0),
                    key=form_widget_key("ui_doping_front_endcap"),
                )
            with endcap_columns[1]:
                endcaps["back_length_m"] = sci(
                    "出口端帽长度 (m)",
                    endcaps.get("back_length_m", 0.0),
                    key=form_widget_key("ui_doping_back_endcap"),
                )
            with endcap_columns[2]:
                endcap_percent = st.number_input(
                    "端帽/有源芯相对浓度 (%)",
                    min_value=0.0,
                    value=100.0 * float(endcaps.get("relative_density", 0.0)),
                    step=1.0,
                    key=form_widget_key("ui_doping_endcap_density"),
                )
                endcaps["relative_density"] = float(endcap_percent) / 100.0
            st.caption(
                "体积平均标定会提高有源芯浓度以保持全晶体平均值；"
                "若上方离子密度代表有源芯浓度，请选择“峰值”或“不归一化”。"
            )
        if legacy_components:
            st.info(
                f"导入配置中有 {len(legacy_components)} 个旧式复杂轮廓；它们继续参与计算，"
                "但不显示在这个简化表单中。新建配置不会再产生这些参数。"
            )

        try:
            preview_grid = Grid3D.from_config(cfg["grid"])
            preview_density, preview_summary = dopant_density_field(
                preview_grid, cfg["material"]
            )
            reference_density = max(
                float(cfg["material"]["ion_density_m3"]), np.finfo(float).tiny
            )
            normalized_density = np.where(
                crystal_volume_mask(preview_grid, cfg["material"]),
                preview_density / reference_density,
                np.nan,
            )
            st.caption(
                "当前三维场预览："
                f"最小 {preview_summary['minimum_density_m3']:.4g}、"
                f"体积平均 {preview_summary['volume_average_density_m3']:.4g}、"
                f"最大 {preview_summary['maximum_density_m3']:.4g} m⁻³。"
            )
            plane = choice_selectbox(
                "掺杂分布中心切片",
                ["xy（z 中心）", "xz（y 中心）", "yz（x 中心）"],
                key=form_widget_key("ui_doping_preview_plane"),
            )
            crystal_geometry = cfg["material"].get("geometry", {})
            crystal_center_x = float(crystal_geometry.get("center_x_m", 0.0))
            crystal_center_y = float(crystal_geometry.get("center_y_m", 0.0))
            if plane.startswith("xy"):
                image_values = normalized_density[len(preview_grid.z) // 2]
                horizontal = preview_grid.x * 1e3
                vertical = preview_grid.y * 1e3
                horizontal_label, vertical_label = "x (mm)", "y (mm)"
            elif plane.startswith("xz"):
                y_index = int(
                    np.argmin(np.abs(preview_grid.y - crystal_center_y))
                )
                image_values = normalized_density[:, y_index, :]
                horizontal = preview_grid.x * 1e3
                vertical = preview_grid.z * 1e3
                horizontal_label, vertical_label = "x (mm)", "z (mm)"
            else:
                x_index = int(
                    np.argmin(np.abs(preview_grid.x - crystal_center_x))
                )
                image_values = normalized_density[:, :, x_index]
                horizontal = preview_grid.y * 1e3
                vertical = preview_grid.z * 1e3
                horizontal_label, vertical_label = "y (mm)", "z (mm)"
            preview_figure = go.Figure(
                go.Heatmap(
                    z=image_values,
                    x=horizontal,
                    y=vertical,
                    colorscale="Viridis",
                    colorbar={"title": "N/Nref"},
                    hovertemplate=(
                        f"{horizontal_label}=%{{x:.4g}}<br>"
                        f"{vertical_label}=%{{y:.4g}}<br>N/Nref=%{{z:.5g}}<extra></extra>"
                    ),
                )
            )
            preview_figure.update_layout(
                height=300,
                margin={"l": 45, "r": 20, "t": 15, "b": 45},
                xaxis_title=horizontal_label,
                yaxis_title=vertical_label,
            )
            preview_figure.update_yaxes(scaleanchor="x", scaleratio=1.0)
            st.plotly_chart(
                preview_figure,
                width="stretch",
                key=form_widget_key("ui_doping_preview_chart"),
            )
        except (TypeError, ValueError) as exc:
            st.error(f"当前掺杂轮廓无法生成：{exc}")


def parse_json(text: str, label: str) -> Any:
    try:
        return loads_json_strict(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} JSON 格式错误：{exc}") from exc


MATERIAL_APPLICATION_FIELDS = (
    ("材料名称", "material.name"),
    ("离子密度 (m⁻³)", "material.ion_density_m3"),
    ("三维掺杂分布", "material.doping_profile"),
    ("Seed 折射率", "material.refractive_index"),
    ("Pump 折射率", "material.pump_refractive_index"),
    ("dn/dT (K⁻¹)", "material.dn_dT_per_K"),
    ("泵浦吸收截面 (m²)", "energy_levels.pump_absorption_cross_section_m2"),
    ("泵浦发射截面 (m²)", "energy_levels.pump_emission_cross_section_m2"),
    ("信号发射截面 (m²)", "energy_levels.signal_emission_cross_section_m2"),
    ("信号吸收/再吸收截面 (m²)", "energy_levels.signal_absorption_cross_section_m2"),
    ("上能级寿命 (s)", "energy_levels.upper_state_lifetime_s"),
    ("泵浦量子效率", "energy_levels.pump_quantum_efficiency"),
    ("三轴导热率 (W/m/K)", "material.thermal.conductivity_W_mK"),
)


def _nested_value(data: dict[str, Any], dotted_path: str) -> Any:
    value: Any = data
    for part in dotted_path.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _display_parameter(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def material_application_rows(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "参数": label,
            "应用前": _display_parameter(_nested_value(before, path)),
            "应用后": _display_parameter(_nested_value(after, path)),
        }
        for label, path in MATERIAL_APPLICATION_FIELDS
    ]


def _configured_path(cfg: dict[str, Any], beam_kind: str) -> tuple[list[dict[str, Any]], str, bool]:
    if beam_kind == "pump":
        return copy.deepcopy(cfg["optical_train"].get("pump", [])), "pump", False
    if bool(
        cfg["simulation"].get("multipass", {}).get("enabled", False)
    ):
        multipass = cfg["simulation"]["multipass"]
        template_names = {
            "double_pass": "双程返回镜",
            "two_mirror": "双侧镜",
            "herriott": "Herriott",
            "relay_4f": "4f 中继",
            "prescribed": "逐程边界",
            "sequential": "迁移顺序计划",
        }
        template = str(multipass.get("template", "double_pass"))
        pass_label = (
            f"{'因果' if uses_causal_clock(cfg['simulation'], cfg['seed'], resolve_signal_model(cfg)) else '顺序'} "
            f"{template_names.get(template, template)} · "
            f"{int(multipass.get('pass_count', 2))} 程"
        )
        return (
            copy.deepcopy(cfg["optical_train"].get("seed", [])),
            f"seed_multipass:{pass_label}",
            False,
        )
    return copy.deepcopy(cfg["optical_train"].get("seed", [])), "seed_simple", False


def _new_path_element(kind: str, name: str, distance_before_m: float) -> dict[str, Any]:
    element: dict[str, Any] = {"type": kind, "name": name, "distance_before_m": float(distance_before_m)}
    if kind == "thin_lens":
        element["focal_length_m"] = 0.10
    elif kind == "double_pass_lens_relay":
        element.update(
            {
                "object_to_first_lens_m": 0.07,
                "first_focal_length_m": 0.07,
                "lens_separation_m": 0.155,
                "second_focal_length_m": 0.2,
                "second_lens_to_mirror_m": 0.2,
                "power_transmission": 1.0,
            }
        )
    elif kind == "thick_lens":
        element.update(
            {
                "refractive_index": 1.5,
                "group_refractive_index": 1.5,
                "group_refractive_index_after": 1.0,
                "thickness_m": 5e-3,
                "radius_front_m": 0.05,
                "radius_back_m": -0.05,
                "n_after": 1.0,
            }
        )
    elif kind == "free_space":
        element["length_m"] = 0.10
    elif kind == "mirror":
        element.update({"power_reflectivity": 0.995, "radius_x_m": 0.20, "radius_y_m": 0.20})
    elif kind == "aperture":
        element.update({"radius_x_m": 0.5e-3, "radius_y_m": 0.5e-3})
    elif kind == "gain_medium":
        element["direction"] = "auto"
    return element


def render_optical_path_editor(cfg: dict[str, Any], beam_kind: str) -> list[dict[str, Any]]:
    """Independent click-based editor for the seed or pump optical path."""

    configured, mode_token, allow_multiple_gain = _configured_path(cfg, beam_kind)
    rows_key = f"path_rows_{beam_kind}"
    token_key = f"path_mode_{beam_kind}"
    version_key = f"path_editor_version_{beam_kind}"
    if rows_key not in st.session_state or st.session_state.get(token_key) != mode_token:
        # An empty imported train starts at the gain entrance. The editor
        # needs a gain row, but must not invent a propagation distance.
        fallback = [{"type": "gain_medium", "name": f"{beam_kind}_gain", "distance_before_m": 0.0}]
        st.session_state[rows_key] = copy.deepcopy(configured or fallback)
        st.session_state[token_key] = mode_token
        st.session_state[version_key] = int(st.session_state.get(version_key, 0)) + 1
    rows = copy.deepcopy(st.session_state[rows_key])
    if not rows:
        rows = [{"type": "gain_medium", "name": f"{beam_kind}_gain", "distance_before_m": 0.0}]
    version = int(st.session_state.get(version_key, 0))
    prefix = path_editor_widget_prefix(
        beam_kind,
        int(st.session_state.get("form_widget_generation", 0)),
        version,
    )
    source_distance = float(rows[0].get("distance_before_m", 0.0))
    entered_source_distance = st.number_input(
        f"{beam_kind.upper()} LASER 到第一个元件的距离 (m)",
        min_value=0.0,
        value=source_distance,
        format="%.6e",
        key=f"{prefix}_source_distance",
    )
    if not math.isclose(float(entered_source_distance), source_distance, rel_tol=0.0, abs_tol=1e-15):
        rows[0]["distance_before_m"] = float(entered_source_distance)
        st.session_state[rows_key] = rows
        st.session_state[version_key] = version + 1
        st.session_state["realtime_active_beam"] = beam_kind
        st.rerun()
    st.caption(
        f"{beam_kind.upper()} LASER → {source_distance * 1e3:.4g} mm → "
        f"{rows[0].get('name', rows[0].get('type', '第一个元件'))} → … → GAIN"
    )
    st.caption(
        "距离按行定义：第一行是 LASER→首元件，之后每一行是上一元件→当前元件；"
        "GAIN 行因此就是上一元件（常见为透镜）→晶体入射面。Pump / Seed 两套数值彼此独立。"
    )
    table = st.data_editor(
        rows,
        num_rows="fixed",
        width="stretch",
        disabled=["type"],
        column_order=PATH_EDITOR_COLUMNS,
        column_config={
            "distance_before_m": st.column_config.NumberColumn(
                "与前一元件距离 (m)",
                min_value=0.0,
                format="%.6e",
                help=(
                    "第一行从本束 LASER 源面起算；后续行从上一行元件起算。"
                    "修改表格后会立即回写并刷新右侧光路。"
                ),
            ),
        },
        key=f"{prefix}_table",
    )
    edited_rows = clean_table_rows(table, rows)
    if edited_rows != rows:
        st.session_state[rows_key] = copy.deepcopy(edited_rows)
        st.session_state[version_key] = version + 1
        st.session_state["realtime_active_beam"] = beam_kind
        st.rerun()
    rows = edited_rows
    st.session_state[rows_key] = copy.deepcopy(rows)

    position_options = ["最前面"] + [
        f"在 {index + 1}. {row.get('name', row.get('type', '元件'))} 后" for index, row in enumerate(rows)
    ]
    element_types = {
        "薄透镜": "thin_lens",
        "厚透镜": "thick_lens",
        "自由空间": "free_space",
        "反射镜": "mirror",
        "光阑": "aperture",
        "观察面": "observation",
    }
    if beam_kind == "seed":
        element_types["双程双透镜回程成像"] = "double_pass_lens_relay"
    if allow_multiple_gain or not any(row.get("type") == "gain_medium" for row in rows):
        element_types["晶体通过"] = "gain_medium"
    insert_position = choice_selectbox("插入位置", position_options, key=f"{prefix}_insert_position")
    insert_label = choice_selectbox("插入元件", list(element_types), key=f"{prefix}_insert_type")
    insert_distance = st.number_input(
        "与前一元件距离 (m)", min_value=0.0, value=0.05, format="%.6e", key=f"{prefix}_insert_distance"
    )
    if st.button(f"插入{insert_label}", type="primary", width="stretch", key=f"{prefix}_insert"):
        insert_at = 0 if insert_position == "最前面" else position_options.index(insert_position)
        updated = copy.deepcopy(rows)
        updated.insert(insert_at, _new_path_element(element_types[insert_label], f"{beam_kind}_{insert_label}_{len(rows) + 1}", insert_distance))
        st.session_state[rows_key] = updated
        st.session_state[version_key] = version + 1
        st.session_state["realtime_active_beam"] = beam_kind
        st.rerun()

    if rows:
        selection_options = [f"{index + 1}. {row.get('name', row.get('type', '元件'))}" for index, row in enumerate(rows)]
        selected_label = choice_selectbox("选中并编辑元件", selection_options, key=f"{prefix}_selected")
        selected_index = selection_options.index(selected_label)
        # The version changes after structural edits; the row distinguishes
        # selection changes within that version, including same-kind elements.
        edit_prefix = f"{prefix}_row_{selected_index}"
        selected = copy.deepcopy(rows[selected_index])
        selected_kind = str(selected.get("type"))
        selected["name"] = st.text_input("名称", str(selected.get("name", selected.get("type", "element"))), key=f"{edit_prefix}_edit_name")
        selected["distance_before_m"] = st.number_input(
            "上一元件→GAIN 入射面距离 (m)" if selected_kind == "gain_medium" else "到前一元件距离 (m)",
            min_value=0.0,
            value=float(selected.get("distance_before_m", 0.0)),
            format="%.6e",
            key=f"{edit_prefix}_edit_distance",
            help=(
                "这是当前 Pump 或 Seed 光路自己的距离；另一束光不会被修改。"
                if selected_kind == "gain_medium"
                else None
            ),
        )
        selected["group_delay_s"] = st.number_input(
            "元件附加群延迟 (s)",
            min_value=0.0,
            value=float(selected.get("group_delay_s", 0.0)),
            format="%.6e",
            key=f"{edit_prefix}_edit_group_delay",
            help="用于数据表已知但几何厚度未显式建模的群延迟；不会改变相位传播。",
        )
        kind = selected_kind
        if kind == "thin_lens":
            selected["focal_length_m"] = st.number_input("焦距 (m)", value=float(selected.get("focal_length_m", 0.1)), format="%.6e", key=f"{edit_prefix}_edit_focal")
        elif kind == "double_pass_lens_relay":
            selected["object_to_first_lens_m"] = st.number_input(
                "物面/晶体→第一透镜 (m)", min_value=0.0,
                value=float(selected.get("object_to_first_lens_m", 0.07)), format="%.6e",
                key=f"{edit_prefix}_edit_relay_object",
            )
            selected["first_focal_length_m"] = st.number_input(
                "第一透镜焦距 (m)", min_value=1e-12,
                value=float(selected.get("first_focal_length_m", 0.07)), format="%.6e",
                key=f"{edit_prefix}_edit_relay_f1",
            )
            selected["lens_separation_m"] = st.number_input(
                "两透镜间距 (m)", min_value=0.0,
                value=float(selected.get("lens_separation_m", 0.155)), format="%.6e",
                key=f"{edit_prefix}_edit_relay_gap",
            )
            selected["second_focal_length_m"] = st.number_input(
                "第二透镜焦距 (m)", min_value=1e-12,
                value=float(selected.get("second_focal_length_m", 0.2)), format="%.6e",
                key=f"{edit_prefix}_edit_relay_f2",
            )
            selected["second_lens_to_mirror_m"] = st.number_input(
                "第二透镜→平面回程镜 (m)", min_value=0.0,
                value=float(selected.get("second_lens_to_mirror_m", 0.2)), format="%.6e",
                key=f"{edit_prefix}_edit_relay_mirror",
            )
        elif kind == "thick_lens":
            selected["refractive_index"] = st.number_input("透镜折射率", min_value=1.0, value=float(selected.get("refractive_index", 1.5)), key=f"{edit_prefix}_edit_n")
            selected["group_refractive_index"] = st.number_input(
                "透镜群折射率",
                min_value=0.01,
                value=float(selected.get("group_refractive_index", selected.get("refractive_index", 1.5))),
                key=f"{edit_prefix}_edit_group_n",
                help="程间布居弛豫时间使用 ng·L/c；未提供时回退相位折射率。",
            )
            selected["group_refractive_index_after"] = st.number_input(
                "透镜后介质群折射率",
                min_value=0.01,
                value=float(selected.get("group_refractive_index_after", selected.get("n_after", 1.0))),
                key=f"{edit_prefix}_edit_group_n_after",
            )
            selected["thickness_m"] = st.number_input("厚度 (m)", min_value=0.0, value=float(selected.get("thickness_m", 5e-3)), format="%.6e", key=f"{edit_prefix}_edit_thickness")
            selected["radius_front_m"] = st.number_input("前表面曲率半径 (m)", value=float(selected.get("radius_front_m", 0.05)), format="%.6e", key=f"{edit_prefix}_edit_r1")
            selected["radius_back_m"] = st.number_input("后表面曲率半径 (m)", value=float(selected.get("radius_back_m", -0.05)), format="%.6e", key=f"{edit_prefix}_edit_r2")
        elif kind == "free_space":
            selected["length_m"] = st.number_input("自由空间长度 (m)", min_value=0.0, value=float(selected.get("length_m", 0.1)), format="%.6e", key=f"{edit_prefix}_edit_length")
        elif kind == "mirror":
            selected["power_reflectivity"] = st.number_input("功率反射率", 0.0, 1.0, float(selected.get("power_reflectivity", 0.995)), step=0.001, key=f"{edit_prefix}_edit_reflectivity")
            mirror_columns = st.columns(2)
            with mirror_columns[0]:
                selected["radius_x_m"] = st.number_input(
                    "x 曲率半径 (m；0=平面镜)", value=float(selected.get("radius_x_m", 0.0)),
                    format="%.6e", key=f"{edit_prefix}_edit_mirror_rx"
                )
            with mirror_columns[1]:
                selected["radius_y_m"] = st.number_input(
                    "y 曲率半径 (m；0=平面镜)", value=float(selected.get("radius_y_m", 0.0)),
                    format="%.6e", key=f"{edit_prefix}_edit_mirror_ry"
                )
        elif kind == "aperture":
            selected["radius_x_m"] = st.number_input("x 半径 (m)", min_value=0.0, value=float(selected.get("radius_x_m", 0.5e-3)), format="%.6e", key=f"{edit_prefix}_edit_aperture_x")
            selected["radius_y_m"] = st.number_input("y 半径 (m)", min_value=0.0, value=float(selected.get("radius_y_m", 0.5e-3)), format="%.6e", key=f"{edit_prefix}_edit_aperture_y")
        elif kind == "gain_medium":
            directions = ["auto", "forward", "backward"]
            selected["direction"] = choice_selectbox(
                "通过方向", directions,
                index=option_index(directions, str(selected.get("direction", "auto"))),
                key=f"{edit_prefix}_edit_direction",
                help="auto 使用光源方向；forward / backward 覆盖本次晶体通过方向。",
            )
            selected["incidence_angle_x_deg"] = st.number_input("晶体入射角 x (deg)", value=float(selected.get("incidence_angle_x_deg", 0.0)), key=f"{edit_prefix}_edit_gain_ax")
            selected["incidence_angle_y_deg"] = st.number_input("晶体入射角 y (deg)", value=float(selected.get("incidence_angle_y_deg", 0.0)), key=f"{edit_prefix}_edit_gain_ay")
        apply_col, delete_col = st.columns(2)
        if apply_col.button("应用下方所选参数", width="stretch", key=f"{prefix}_apply"):
            updated = copy.deepcopy(rows)
            updated[selected_index] = selected
            st.session_state[rows_key] = updated
            st.session_state[version_key] = version + 1
            st.session_state["realtime_active_beam"] = beam_kind
            st.rerun()
        if delete_col.button("删除所选", width="stretch", key=f"{prefix}_delete"):
            if kind == "gain_medium" and sum(row.get("type") == "gain_medium" for row in rows) <= 1:
                st.warning("每条光路至少需要一个 gain_medium，最后一个晶体不能删除。")
            else:
                updated = copy.deepcopy(rows)
                del updated[selected_index]
                st.session_state[rows_key] = updated
                st.session_state[version_key] = version + 1
                st.session_state["realtime_active_beam"] = beam_kind
                st.rerun()
    if st.button(f"右侧详细显示 {beam_kind.upper()}", width="stretch", key=f"{prefix}_activate"):
        st.session_state["realtime_active_beam"] = beam_kind
        st.rerun()
    return rows


def uploaded_file_sha256(uploaded: Any) -> str:
    """Hash an uploaded file without copying it or changing its read position."""

    digest = hashlib.sha256()
    position = uploaded.tell()
    try:
        uploaded.seek(0)
        while True:
            chunk = uploaded.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        uploaded.seek(position)
    return digest.hexdigest()


def render_beam_radius_probe(cfg: dict[str, Any], beam_kind: str) -> None:
    """Compact live radius probe embedded in each independent beam panel."""

    title = "Pump" if beam_kind == "pump" else "Seed"
    with st.popover(f"📏 {title} 坐标半径探针", width="stretch"):
        include_gain = bool(st.session_state.get("realtime_gain_inserted", True))
        try:
            preview = build_bench_preview(
                cfg,
                beam_kind,
                samples_per_section=256,
                include_gain_medium=include_gain,
            )
        except Exception as exc:
            st.warning(f"当前 {title} 光路无法取样：{type(exc).__name__}: {exc}")
            return

        gain_entrance = first_gain_entrance_m(preview)
        reference_labels = ["本束 LASER 源面 = 0"]
        if gain_entrance is not None:
            reference_labels.append("共享晶体入口 = 0")
        selected_reference = choice_radio(
            "输入坐标基准",
            reference_labels,
            horizontal=False,
            key=f"radius_probe_reference_{beam_kind}",
        )
        reference = "gain" if selected_reference.startswith("共享") else "laser"
        if reference == "gain" and gain_entrance is not None:
            allowed_start_mm = (float(preview.optical_path_m[0]) - gain_entrance) * 1e3
            allowed_end_mm = (float(preview.optical_path_m[-1]) - gain_entrance) * 1e3
            default_mm = 0.0
            coordinate_label = "相对共享晶体入口的累计光程 (mm)"
        else:
            allowed_start_mm = float(preview.optical_path_m[0]) * 1e3
            allowed_end_mm = float(preview.optical_path_m[-1]) * 1e3
            default_mm = (gain_entrance if gain_entrance is not None else float(preview.optical_path_m[0])) * 1e3
            coordinate_label = f"距 {title.upper()} LASER 源面的累计光程 (mm)"
        coordinate_mm = float(
            st.number_input(
                coordinate_label,
                value=float(default_mm),
                format="%.6f",
                key=f"radius_probe_coordinate_{beam_kind}_{reference}",
            )
        )
        st.caption(f"可取样范围：{allowed_start_mm:.6g} … {allowed_end_mm:.6g} mm")
        try:
            sample = sample_bench_preview(
                preview,
                coordinate_mm * 1e-3,
                reference=reference,
            )
        except ValueError as exc:
            st.warning(str(exc))
            return

        radius_columns = st.columns(2)
        radius_columns[0].metric("D4σ x 半径", f"{sample.radius_x_D4sigma_m * 1e6:.6g} µm")
        radius_columns[1].metric("D4σ y 半径", f"{sample.radius_y_D4sigma_m * 1e6:.6g} µm")
        center_columns = st.columns(2)
        center_columns[0].metric("x 束心 offset", f"{sample.centroid_x_m * 1e6:.6g} µm")
        center_columns[1].metric("y 束心 offset", f"{sample.centroid_y_m * 1e6:.6g} µm")
        x_lower, x_upper = sample.x_bounds_D4sigma_m
        y_lower, y_upper = sample.y_bounds_D4sigma_m
        gain_coordinate = (
            "—"
            if sample.gain_relative_path_m is None
            else f"{sample.gain_relative_path_m * 1e3:.6g} mm"
        )
        st.caption(
            f"本束真实光程 s = {sample.local_path_m * 1e3:.6g} mm；"
            f"相对共享晶体入口 = {gain_coordinate}。"
        )
        st.caption(
            f"D4σ 边界：x [{x_lower * 1e6:.6g}, {x_upper * 1e6:.6g}] µm；"
            f"y [{y_lower * 1e6:.6g}, {y_upper * 1e6:.6g}] µm。"
        )
        state = "真实晶体已插入" if include_gain else "晶体已取下，预留段按外部介质传播"
        st.caption(f"当前取样状态：{state}。半径始终是相对当前束心定义的。")


def combined_bench_preview_figure(
    previews: list[BenchPreview],
    cfg: dict[str, Any],
    *,
    focus_gain: bool = True,
    active_beam_kind: str = "seed",
    optical_trace_summaries: list[dict[str, Any]] | None = None,
    optical_gain_path_samples: list[dict[str, Any]] | None = None,
) -> go.Figure:
    """Render cold q envelopes and optional solver-recorded D4-sigma anchors."""

    figure = go.Figure()
    aligned_previews = align_previews_to_shared_gain(previews, float(cfg["grid"]["length_z_m"]))
    scale = build_path_display_scale(aligned_previews, focus_gain=focus_gain)
    colors = {
        "seed": ("#ef4444", "rgba(239,68,68,0.10)"),
        "pump": ("#2563eb", "rgba(37,99,235,0.10)"),
    }
    full_extent = 0.0
    gain_extent = 0.0
    ordered_pairs = sorted(
        zip(previews, aligned_previews, strict=True),
        key=lambda pair: pair[0].beam_kind == active_beam_kind,
    )
    for preview, aligned in ordered_pairs:
        physical_path_mm = np.asarray(preview.optical_path_m) * 1e3
        display_path_mm = np.asarray(scale.map_m(aligned.optical_path_m)) * 1e3
        cx = np.asarray(preview.centroid_x_m) * 1e3
        cy = np.asarray(preview.centroid_y_m) * 1e3
        rx = np.asarray(preview.radius_x_D4sigma_m) * 1e3
        ry = np.asarray(preview.radius_y_D4sigma_m) * 1e3
        lower = np.minimum(cx - rx, cy - ry)
        upper = np.maximum(cx + rx, cy + ry)
        color, fill = colors[preview.beam_kind]
        custom = np.column_stack((physical_path_mm, rx, ry, cx, cy))
        figure.add_scatter(
            x=display_path_mm,
            y=lower,
            mode="lines",
            line=dict(color=color, width=2),
            legendgroup=preview.beam_kind,
            showlegend=False,
            customdata=custom,
            hovertemplate=(
                f"{preview.beam_kind.upper()}<br>真实光程 %{{customdata[0]:.5g}} mm<br>"
                "D4σ rx %{customdata[1]:.5g} mm<br>D4σ ry %{customdata[2]:.5g} mm<extra></extra>"
            ),
        )
        figure.add_scatter(
            x=display_path_mm,
            y=upper,
            mode="lines",
            line=dict(color=color, width=2),
            fill="tonexty",
            fillcolor=fill,
            legendgroup=preview.beam_kind,
            name=f"{preview.beam_kind.upper()} D4σ 包络",
            customdata=custom,
            hovertemplate=(
                f"{preview.beam_kind.upper()}<br>真实光程 %{{customdata[0]:.5g}} mm<br>"
                "D4σ rx %{customdata[1]:.5g} mm<br>D4σ ry %{customdata[2]:.5g} mm<extra></extra>"
            ),
        )
        full_extent = max(full_extent, float(np.max(np.abs(lower))), float(np.max(np.abs(upper))))
        mask = gain_sample_mask(preview)
        if np.any(mask):
            gain_extent = max(gain_extent, float(np.max(np.abs(lower[mask]))), float(np.max(np.abs(upper[mask]))))
        if optical_trace_summaries or optical_gain_path_samples:
            anchors = optical_trace_anchors(
                preview,
                optical_trace_summaries or (),
                optical_gain_path_samples or (),
            )
            if anchors:
                preview_path = np.asarray(preview.optical_path_m, dtype=float)
                aligned_path = np.asarray(aligned.optical_path_m, dtype=float)
                anchor_display_path_mm = np.asarray(
                    [
                        float(
                            scale.map_m(
                                np.interp(
                                    float(anchor.optical_path_m),
                                    preview_path,
                                    aligned_path,
                                )
                            )
                        )
                        * 1e3
                        for anchor in anchors
                    ],
                    dtype=float,
                )
                anchor_rx = np.asarray(
                    [anchor.radius_x_D4sigma_m for anchor in anchors], dtype=float
                ) * 1e3
                anchor_ry = np.asarray(
                    [anchor.radius_y_D4sigma_m for anchor in anchors], dtype=float
                ) * 1e3
                anchor_cx = np.asarray(
                    [anchor.centroid_x_m for anchor in anchors], dtype=float
                ) * 1e3
                anchor_cy = np.asarray(
                    [anchor.centroid_y_m for anchor in anchors], dtype=float
                ) * 1e3
                anchor_lower = np.minimum(anchor_cx - anchor_rx, anchor_cy - anchor_ry)
                anchor_upper = np.maximum(anchor_cx + anchor_rx, anchor_cy + anchor_ry)
                anchor_custom = np.asarray(
                    [
                        [anchor.name, anchor.radius_x_D4sigma_m * 1e3, anchor.radius_y_D4sigma_m * 1e3]
                        for anchor in anchors
                    ],
                    dtype=object,
                )
                figure.add_scatter(
                    x=anchor_display_path_mm,
                    y=anchor_lower,
                    mode="lines+markers",
                    line=dict(color=color, width=3, dash="dash"),
                    marker=dict(color=color, size=7, symbol="circle-open"),
                    legendgroup=f"{preview.beam_kind}_result",
                    showlegend=False,
                    customdata=anchor_custom,
                    hovertemplate=(
                        "%{customdata[0]}<br>运行后 D4σ rx %{customdata[1]:.5g} mm<br>"
                        "运行后 D4σ ry %{customdata[2]:.5g} mm<extra></extra>"
                    ),
                )
                figure.add_scatter(
                    x=anchor_display_path_mm,
                    y=anchor_upper,
                    mode="lines+markers",
                    line=dict(color=color, width=3, dash="dash"),
                    marker=dict(color=color, size=7, symbol="circle-open"),
                    legendgroup=f"{preview.beam_kind}_result",
                    name=f"{preview.beam_kind.upper()} 运行后 D4σ 采样",
                    customdata=anchor_custom,
                    hovertemplate=(
                        "%{customdata[0]}<br>运行后 D4σ rx %{customdata[1]:.5g} mm<br>"
                        "运行后 D4σ ry %{customdata[2]:.5g} mm<extra></extra>"
                    ),
                )
                full_extent = max(
                    full_extent,
                    float(np.max(np.abs(anchor_lower))),
                    float(np.max(np.abs(anchor_upper))),
                )
                for anchor, lower_value, upper_value in zip(
                    anchors, anchor_lower, anchor_upper, strict=True
                ):
                    if any(
                        element.kind == "gain_medium"
                        and element.start_m - 1e-12
                        <= anchor.optical_path_m
                        <= element.end_m + 1e-12
                        for element in preview.elements
                    ):
                        gain_extent = max(
                            gain_extent,
                            abs(float(lower_value)),
                            abs(float(upper_value)),
                        )

    geometry = cfg["material"].get("geometry", {})
    if geometry.get("shape", "cuboid") == "cylinder":
        diameter_m = geometry.get("diameter_m")
        if diameter_m is None:
            diameter_m = min(
                cfg["grid"]["size_x_m"], cfg["grid"]["size_y_m"]
            )
        crystal_half_x_mm = crystal_half_y_mm = 0.5 * float(diameter_m) * 1e3
    else:
        width_x_m = geometry.get("width_x_m")
        if width_x_m is None:
            width_x_m = cfg["grid"]["size_x_m"]
        height_y_m = geometry.get("height_y_m")
        if height_y_m is None:
            height_y_m = cfg["grid"]["size_y_m"]
        crystal_half_x_mm = 0.5 * float(width_x_m) * 1e3
        crystal_half_y_mm = 0.5 * float(height_y_m) * 1e3
    crystal_center_x_mm = float(geometry.get("center_x_m", 0.0)) * 1e3
    crystal_center_y_mm = float(geometry.get("center_y_m", 0.0)) * 1e3
    crystal_lower_mm = min(crystal_center_x_mm - crystal_half_x_mm,
                           crystal_center_y_mm - crystal_half_y_mm)
    crystal_upper_mm = max(crystal_center_x_mm + crystal_half_x_mm,
                           crystal_center_y_mm + crystal_half_y_mm)
    crystal_extent_mm = max(abs(crystal_lower_mm), abs(crystal_upper_mm))
    drawn_gain: set[tuple[float, float, int | None]] = set()
    for preview in aligned_previews:
        for element in preview.elements:
            if element.kind != "gain_medium":
                continue
            key = (element.start_m, element.end_m, element.pass_index)
            if key in drawn_gain:
                continue
            drawn_gain.add(key)
            x0 = float(scale.map_m(element.start_m)) * 1e3
            x1 = float(scale.map_m(element.end_m)) * 1e3
            material_enabled = bool(element.material_enabled)
            figure.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=crystal_lower_mm,
                y1=crystal_upper_mm,
                fillcolor="rgba(249,115,22,0.18)" if material_enabled else "rgba(148,163,184,0.035)",
                line=dict(
                    color="rgba(249,115,22,0.9)" if material_enabled else "rgba(148,163,184,0.95)",
                    width=1.5,
                    dash="solid" if material_enabled else "dash",
                ),
                layer="below",
            )
            figure.add_annotation(
                x=0.5 * (x0 + x1),
                y=crystal_upper_mm,
                text=f"G{element.pass_index}" if material_enabled else f"G{element.pass_index} 空位",
                showarrow=False,
                yshift=10,
                font=dict(size=10),
            )
    extent = max(gain_extent, crystal_extent_mm) if focus_gain else max(full_extent, crystal_extent_mm)
    extent = max(1.18 * extent, 1e-3)
    display_start_mm = scale.display_start_m * 1e3
    display_end_mm = scale.display_end_m * 1e3
    display_span_mm = max(display_end_mm - display_start_mm, 1e-6)
    padding = max(display_span_mm * 0.025, 0.05)
    tick_values, tick_labels = scale.axis_ticks_mm()
    x_title = "相对共享晶体入口位置 (mm)" if not scale.compressed else "晶体对齐显示（长自由空间已压缩）"
    gain_inserted = any(
        element.kind == "gain_medium" and element.material_enabled
        for preview in previews
        for element in preview.elements
    )
    figure.update_layout(
        title=(
            (
                "计算前冷态 q 包络 + 运行后实际 D4σ 采样面"
                if optical_trace_summaries or optical_gain_path_samples
                else "Pump / Seed 计算前冷态 q 包络"
            )
            if gain_inserted
            else "Pump / Seed 空气调光 · 晶体位置仅作占位"
        ),
        xaxis=dict(
            title=x_title,
            range=[display_start_mm - padding, display_end_mm + padding],
            tickmode="array" if scale.compressed else "auto",
            tickvals=tick_values if scale.compressed else None,
            ticktext=tick_labels if scale.compressed else None,
            showgrid=False,
        ),
        yaxis=dict(
            title="保守 D4σ 横向包络 (mm)",
            range=[-extent, extent],
            zeroline=True,
            zerolinecolor="rgba(148,163,184,0.55)",
        ),
        hovermode="closest",
        height=470,
        margin=dict(t=58, l=64, r=20, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )
    return figure


def render_configuration_overview(cfg: dict[str, Any]) -> None:
    """Show the active calculation contract without opening the sidebar."""

    effective = resolve_physics(cfg)
    signal_model = resolve_signal_model(cfg)
    model_names = {
        "fluence": "单色通量",
        "commercial_app_fast": "Commercial App 式 Gaussian 快速",
        "spectral_time": "多色 × 时间",
        "josa_transport": "JOSA 输运",
        "coherent_cpa": "相干 CPA",
    }
    causal_multipass = cfg["simulation"].get("multipass", {})
    if bool(causal_multipass.get("enabled", False)):
        template_names = {
            "double_pass": "双程返回镜",
            "two_mirror": "双侧镜",
            "herriott": "Herriott",
            "relay_4f": "4f 中继",
            "prescribed": "逐程边界",
            "sequential": "迁移顺序计划",
        }
        template = str(causal_multipass.get("template", "double_pass"))
        pass_label = (
            f"{'因果' if uses_causal_clock(cfg['simulation'], cfg['seed'], signal_model) else '顺序'} "
            f"{template_names.get(template, template)} · "
            f"{int(causal_multipass.get('pass_count', 2))} 程"
        )
    else:
        pass_label = "单程"
    optical_grid = cfg.get("grid", {}).get("optical", {})
    pump_grid = cfg.get("grid", {}).get("pump_optical", {})
    pump_grid_mode = str(pump_grid.get("mode", "shared"))
    pump_grid_label = (
        f" · Pump {pump_grid_mode}" if pump_grid_mode != "shared" else ""
    )
    grid_label = (
        f"Seed 双网格 · {int(optical_grid.get('nx', 0))}×{int(optical_grid.get('ny', 0))}{pump_grid_label}"
        if optical_grid.get("enabled", False)
        else (
            f"Pump 独立网格 · custom"
            if pump_grid_mode == "custom"
            else f"{int(cfg['grid']['nx'])}×{int(cfg['grid']['ny'])}×{int(cfg['grid']['nz'])}"
        )
    )
    seed_propagation = effective["optical_propagation"]
    radial_wave = (
        signal_model == "spectral_time"
        and cfg["seed"]["spectral_time"].get("spatial_backend") == "axisymmetric"
    ) or (
        signal_model == "coherent_cpa"
        and cfg["seed"]["coherent"].get("fidelity_mode")
        in {"axisymmetric_kerr", "axisymmetric_transient_gain"}
    )
    polar_commercial_app = (
        signal_model == "commercial_app_fast"
        and cfg["seed"]["commercial_app_fast"].get("spatial_backend") == "axisymmetric"
    )
    if radial_wave or polar_commercial_app:
        seed_propagation = "axisymmetric CN" if radial_wave else "Commercial App Gaussian-q / polar"
        grid_label += f" · Nr={cfg['seed']['coherent']['axisymmetric'].get('radial_bins', 24)}"
    overview_cards = (
        ("信号模型", model_names.get(signal_model, signal_model)),
        (
            "空间传播",
            f"Seed `{seed_propagation}` · "
            f"Pump `{effective['pump_propagation']}`",
        ),
        ("平台与网格", f"{pass_label} · {grid_label}"),
    )
    for column, (label, value) in zip(
        st.columns(3), overview_cards, strict=True
    ):
        with column:
            with st.container(border=True):
                st.caption(label)
                # Metric values are forced onto one line and Streamlit replaces
                # overflow with an ellipsis.  Markdown wraps long backend/grid
                # contracts, so every active token remains readable on laptop
                # widths and in the narrow main pane beside the form sidebar.
                st.markdown(f"**{value}**")
    st.caption(
        f"布居：{resolve_population_dynamics(cfg)}　·　"
        f"热场：{_resolved_thermal_label(cfg, effective)}"
    )


def render_realtime_bench(cfg: dict[str, Any]) -> None:
    st.subheader("光路预览")
    st.caption(
        "运行前冷态 q 参数用于快速调光；运行后的虚线 D4σ 和热场结果会显示在结果区。"
    )
    with st.container(border=True):
        st.markdown("**实时预览控制台**")
        controls, medium_control, focus_control = st.columns(
            [1.25, 1.25, 1.15]
        )
        with controls:
            if "realtime_active_beam" not in st.session_state:
                st.session_state["realtime_active_beam"] = "seed"
            active_beam_options = ["seed", "pump"]
            active_beam_widget_key = amplifier_view_widget_key(
                "realtime_active_beam"
            )
            beam_kind = choice_radio(
                "当前查看",
                active_beam_options,
                horizontal=True,
                index=option_index(
                    active_beam_options,
                    str(
                        st.session_state.get(
                            "realtime_active_beam", "seed"
                        )
                    ),
                ),
                key=active_beam_widget_key,
                format_func=str.upper,
            )
            if active_beam_widget_key != "realtime_active_beam":
                st.session_state["realtime_active_beam"] = beam_kind
        with medium_control:
            gain_inserted_widget_key = amplifier_view_widget_key(
                "realtime_gain_inserted"
            )
            gain_inserted = choice_toggle(
                "预览中插入增益晶体",
                value=bool(
                    st.session_state.get("realtime_gain_inserted", True)
                ),
                key=gain_inserted_widget_key,
                help=(
                    "关闭后只在实时光路中取下晶体：保留同一位置和长度，"
                    "但不做晶体界面折射。不会修改正式求解配置。"
                ),
            )
            if gain_inserted_widget_key != "realtime_gain_inserted":
                st.session_state["realtime_gain_inserted"] = gain_inserted
        with focus_control:
            focus_gain_widget_key = amplifier_view_widget_key(
                "realtime_focus_gain"
            )
            focus_gain = choice_toggle(
                "聚焦到晶体区域",
                value=bool(
                    st.session_state.get("realtime_focus_gain", True)
                ),
                key=focus_gain_widget_key,
                help=(
                    "开启后只压缩预览中的长自由空间，让晶体附近的腰斑和元件更容易比较；"
                    "真实光程、光束计算和正式求解配置都不会改变。"
                ),
            )
            if focus_gain_widget_key != "realtime_focus_gain":
                st.session_state["realtime_focus_gain"] = focus_gain
        st.caption(
            "上面两个开关只改变冷态预览，不会开关正式求解中的晶体、热场、增益或传播模块。"
        )
    if not gain_inserted:
        st.info(
            "空气调光模式：晶体参数仍完整保留，虚线框是晶体预留位；"
            "当前光束在该长度内按外部介质传播。打开上方开关即恢复真实折射。"
        )
    previews: list[BenchPreview] = []
    for candidate in ("seed", "pump"):
        try:
            previews.append(
                build_bench_preview(
                    cfg,
                    candidate,
                    include_gain_medium=gain_inserted,
                )
            )
        except Exception as exc:
            st.warning(f"{candidate} 实时光路暂不可用：{type(exc).__name__}: {exc}")
    if not previews:
        return
    if not any(preview.beam_kind == beam_kind for preview in previews):
        beam_kind = previews[0].beam_kind
    preview = next(preview for preview in previews if preview.beam_kind == beam_kind)
    gain_mask = gain_sample_mask(preview)
    if np.any(gain_mask):
        gain_rx = float(np.min(preview.radius_x_D4sigma_m[gain_mask])) * 1e6
        gain_ry = float(np.min(preview.radius_y_D4sigma_m[gain_mask])) * 1e6
        gain_region_label = "晶体内" if gain_inserted else "晶体预留段（外部介质）"
        st.caption(
            f"当前 {beam_kind.upper()} · 真实总光程 {preview.optical_path_m[-1] * 1e3:.5g} mm · "
            f"{gain_region_label}"
            f"最小 D4σ 半径：x {gain_rx:.5g} µm，y {gain_ry:.5g} µm"
        )
    preview_2d, preview_3d = st.tabs(["二维 D4σ 包络", "三维光路"])
    with preview_2d:
        st.plotly_chart(
            combined_bench_preview_figure(
                previews,
                cfg,
                focus_gain=focus_gain,
                active_beam_kind=beam_kind,
            ),
            width="stretch",
            key=f"bench_preview_combined_{beam_kind}_{focus_gain}_{gain_inserted}",
            config={"scrollZoom": True, "displaylogo": False},
        )
        st.caption(
            "实线是运行前冷态 q 包络；每束只有上、下两条线，包络同时包含 x/y 两方向。"
            + ("橙色矩形是共享的同一块 gain。" if gain_inserted else "灰色虚线框是晶体预留位，当前没有材料折射。")
        )
    with preview_3d:
        st.plotly_chart(
            optical_path_3d_figure(
                previews,
                cfg,
                active_beam_kind=beam_kind,
                focus_gain=focus_gain,
            ),
            width="stretch",
            key=f"bench_preview_3d_{beam_kind}_{focus_gain}_{gain_inserted}",
            config={"scrollZoom": True, "displaylogo": False},
        )
        st.caption("两束光以椭圆 D4σ 光束体显示；透镜/镜面尺寸及光阑标记仅作示意，偏心采用配置值。孔径截断请检查实际运行光学面。滚轮缩放，拖动旋转。")
    if preview.elements:
        with st.expander(f"{beam_kind.upper()} 元件与真实光程"):
            st.dataframe(
                [
                    {
                        "顺序": index + 1,
                        "元件": element.name,
                        "类型": element.kind,
                        "起点光程 (mm)": element.start_m * 1e3,
                        "终点光程 (mm)": element.end_m * 1e3,
                        "晶体程次": element.pass_index,
                        "方向": element.direction,
                        "晶体状态": (
                            "已插入" if element.material_enabled else "空位（外部介质）"
                        ) if element.kind == "gain_medium" else "—",
                    }
                    for index, element in enumerate(preview.elements)
                ],
                width="stretch",
                hide_index=True,
            )
    clipping = [warning for warning in preview.warnings if "clipping" in warning]
    for warning in clipping:
        st.warning(localize_source_message(warning, active_locale()))
    with st.expander("预览定义、坐标与边界", expanded=False):
        st.write(
            "这是运行前冷态 q 参数预览，不包含尚未求出的热场 Δn、增益导引或 FFT 孔径重塑。"
            "Pump 与 Seed 使用各自的 LASER 起点，但在同一块 gain 处对齐；聚焦视图只压缩长自由空间，"
            "不改变晶体比例。"
        )
        if preview.warnings:
            for warning in preview.warnings:
                st.write(
                    f"- {localize_source_message(warning, active_locale())}"
                )
        else:
            st.caption("当前冷态预览没有触发窗口或近似警告。")


def _sync_josa_mode_from_widget(cfg: dict[str, Any]) -> str:
    """Apply the later Seed/JOSA widget state before solver locks are drawn.

    Streamlit evaluates the sidebar from top to bottom.  The JOSA mode editor
    lives with the Seed inputs, but its choice changes the solver contract near
    the top of the page.  Reading the already committed widget value here
    removes the former one-rerun delay without duplicating that editor.
    """

    josa = cfg["seed"].setdefault("josa", {})
    configured = str(josa.get("mode", "retarded_pc"))
    widget_value = st.session_state.get(form_widget_key("ui_josa_mode"))
    labels = {
        str(info["label_zh"]): mode for mode, info in JOSA_MODE_INFO.items()
    }
    if widget_value in labels:
        configured = labels[str(widget_value)]
    elif str(widget_value) in JOSA_MODE_INFO:
        configured = str(widget_value)
    josa["mode"] = configured
    return configured


def _full_josa_selected(cfg: dict[str, Any]) -> bool:
    return (
        str(cfg["simulation"].get("signal_model", "auto")) == "josa_transport"
        and _sync_josa_mode_from_widget(cfg) == JOSA_FULL_3D_MODE
    )


def _josa_selected(cfg: dict[str, Any]) -> bool:
    """Return whether the active Seed representation is any JOSA mode."""

    return resolve_signal_model(cfg) == "josa_transport"


def _render_periodic_solver_controls(cfg: dict[str, Any], *, full_josa: bool) -> None:
    repetition_rate = float(
        st.session_state.get(
            form_widget_key("ui_seed_rep_rate"),
            cfg["simulation"]["repetition_rate_Hz"],
        )
    )
    cfg["simulation"]["repetition_rate_Hz"] = repetition_rate
    st.caption(
        f"当前 Seed 重复频率：{repetition_rate:.6g} Hz。唯一输入框在“② 光束 → Seed”；"
        "Pump 波形和周期稳态共享这个系统时钟。"
    )
    periodic_cfg = cfg["simulation"].setdefault("periodic_steady_state", {})
    periodic_enabled = repetition_rate > 0.0
    periodic_cfg["enabled"] = periodic_enabled
    joint_mode = (
        cfg.get("multiphysics", {}).get("coupling_model") == "thermal_optical_fixed_point"
        and cfg["multiphysics"].get("iteration_scheme") in {"joint", "joint_frozen"}
    )
    if joint_mode:
        periodic_cfg["method"] = "pulse_map"
    st.caption(
        "重复运行固定求 Pump + Seed 周期稳态；每轮更新次数由上方迭代方式控制。"
        if periodic_enabled else
        "重复频率为 0 Hz，当前是单次激发；设置正重复频率后自动求周期稳态。"
    )
    if periodic_enabled:
        if str(periodic_cfg.get("method", "pulse_map")) == "reduced_pulse_map":
            periodic_cfg["method"] = "pulse_map"
            periodic_cfg["pulse_map_initialization"] = "reduced_pulse_map"
        steady_method_labels = {
            "周期平均代数稳态 — 快速，推荐高重复频率": "cycle_average",
            "逐脉冲周期固定点 — 严格，强单脉冲耗尽": "pulse_map",
        }
        current_method = str(periodic_cfg.get("method", "pulse_map"))
        if joint_mode:
            st.session_state[form_widget_key("ui_periodic_method_joint")] = list(steady_method_labels)[1]
        selected_method_label = choice_radio(
            "内层稳态求法",
            list(steady_method_labels),
            index=(
                list(steady_method_labels.values()).index(current_method)
                if current_method in steady_method_labels.values()
                else 1
            ),
            key=form_widget_key("ui_periodic_method_joint" if joint_mode else "ui_periodic_method"),
            disabled=joint_mode,
        )
        periodic_cfg["method"] = steady_method_labels[selected_method_label]
        acceleration_cfg = periodic_cfg.setdefault("acceleration", {})
        acceleration_enabled = choice_toggle(
            "自动固定点加速",
            value=str(acceleration_cfg.get("method", "anderson")) == "anderson",
            key=form_widget_key("ui_periodic_fixed_point_acceleration"),
            help=(
                "使用带残差、内存和布居守恒保护的 Anderson 加速；"
                "若候选步不可靠会自动退回原来的稳态迭代。"
            ),
        )
        acceleration_cfg["method"] = (
            "anderson" if acceleration_enabled else "off"
        )
        if periodic_cfg["method"] == "cycle_average":
            # full_3d_pc still uses transient populations for the final probe;
            # its algebraic Pump+Seed equilibrium is handled by the dedicated
            # average solver rather than by physics.population_dynamics.
            st.success(
                "快速平均稳态会把 Eseed × 重复频率转换成 Seed 平均光子流，并与 Pump、"
                "自发衰减、受激发射/再吸收一起求三维代数稳态；迭代次数不是经过的脉冲数。"
            )
            steady_columns = st.columns(2)
            with steady_columns[0]:
                periodic_cfg["average_minimum_iterations"] = int(st.number_input(
                    "最少自洽迭代", 1, 10000,
                    int(periodic_cfg.get("average_minimum_iterations", 2)),
                    key=form_widget_key("ui_average_min_iterations"),
                ))
                periodic_cfg["average_relaxation"] = st.number_input(
                    "稳态松弛系数", 0.01, 1.0,
                    float(periodic_cfg.get("average_relaxation", 0.5)), step=0.05,
                    key=form_widget_key("ui_average_relaxation"),
                    help="较小值更稳定但迭代更多；不是时间步长。",
                )
            with steady_columns[1]:
                average_maximum_key = form_widget_key(
                    "ui_average_max_iterations"
                )
                st.session_state[average_maximum_key] = max(
                    int(
                        st.session_state.get(
                            average_maximum_key,
                            periodic_cfg.get("average_maximum_iterations", 40),
                        )
                    ),
                    int(periodic_cfg["average_minimum_iterations"]),
                )
                periodic_cfg["average_maximum_iterations"] = int(
                    st.number_input(
                        "最多自洽迭代",
                        min_value=int(
                            periodic_cfg["average_minimum_iterations"]
                        ),
                        max_value=100000,
                        key=average_maximum_key,
                    )
                )
                if not joint_mode:
                    periodic_cfg["relative_tolerance"] = sci(
                        "内层相对容差（0.003 = 0.3%）", periodic_cfg.get("relative_tolerance", 1e-5),
                        key=form_widget_key("ui_periodic_tolerance"),
                    )
            st.caption(
                "适合 25 MHz 这类高重复频率且单脉冲只改变少量布居的放大器。"
                "最终仍会用当前 FN/JOSA/CPA 脉冲求解器打一发校验；"
                "若单脉冲布居跳变 >1%，结果会提示改用严格模式。"
            )
        else:
            if cfg.get("multiphysics", {}).get("iteration_scheme") != "joint_frozen" or not joint_mode:
                steady_columns = st.columns(2)
                with steady_columns[0]:
                    periodic_cfg["minimum_cycles"] = int(st.number_input(
                        "最少周期数", 1, 100000,
                        int(periodic_cfg.get("minimum_cycles", 3)),
                        key=form_widget_key("ui_periodic_min_cycles"),
                    ))
                    periodic_cfg["consecutive_cycles"] = int(st.number_input(
                        "连续满足次数", 1, 1000,
                        int(periodic_cfg.get("consecutive_cycles", 3)),
                        key=form_widget_key("ui_periodic_consecutive"),
                    ))
                with steady_columns[1]:
                    minimum_viable_cycles = (
                        int(periodic_cfg["minimum_cycles"])
                        + int(periodic_cfg["consecutive_cycles"])
                        - 1
                    )
                    periodic_maximum_key = form_widget_key(
                        "ui_periodic_max_cycles"
                    )
                    st.session_state[periodic_maximum_key] = max(
                        int(
                            st.session_state.get(
                                periodic_maximum_key,
                                periodic_cfg.get("maximum_cycles", 1000),
                            )
                        ),
                        minimum_viable_cycles,
                    )
                    periodic_cfg["maximum_cycles"] = int(
                        st.number_input(
                            "最多周期数",
                            min_value=minimum_viable_cycles,
                            max_value=1000000,
                            key=periodic_maximum_key,
                        )
                    )
                    if not joint_mode:
                        periodic_cfg["relative_tolerance"] = sci(
                            "内层相对容差（0.003 = 0.3%）", periodic_cfg.get("relative_tolerance", 1e-5),
                            key=form_widget_key("ui_periodic_tolerance"),
                        )
            initialization_labels = {
                "自动 — 长寿命/高重复频率时先求平均稳态": "auto",
                "直接从设定初始布居开始": "initial_population",
                "总是先求周期平均初值": "cycle_average",
                "先用低维脉冲图靠近固定点，再由完整主脉冲收敛": "reduced_pulse_map",
            }
            current_initialization = str(
                periodic_cfg.get("pulse_map_initialization", "auto")
            )
            initialization_label = choice_selectbox(
                "离散周期映射初值",
                list(initialization_labels),
                index=(
                    list(initialization_labels.values()).index(
                        current_initialization
                    )
                    if current_initialization in initialization_labels.values()
                    else 0
                ),
                key=form_widget_key("ui_pulse_map_initialization"),
                help=(
                    "平均稳态只用作数值初值。严格 pulse_map 用当前完整 Seed 求解器闭合；"
                    "低维脉冲图也只用作初值，随后仍由完整主脉冲逐周期闭合。"
                    "含上转换/交叉弛豫时会安全跳过平均初值。"
                ),
            )
            periodic_cfg["pulse_map_initialization"] = initialization_labels[
                initialization_label
            ]
            carrier_convergence = periodic_cfg.setdefault(
                "free_carrier_convergence", {}
            )
            carrier_gate_applicable = bool(
                resolve_signal_model(cfg) == "coherent_cpa"
                and str(resolve_physics(cfg)["nonlinear_loss_model"])
                == "strong_field_time_domain"
                and str(
                    cfg.get("material", {})
                    .get("strong_field", {})
                    .get("free_carrier", {})
                    .get("model", "off")
                )
                == "rate_equation"
            )
            if carrier_gate_applicable:
                carrier_convergence["enabled"] = st.checkbox(
                    "将 FCA 载流子残差用于收敛判断",
                    value=bool(carrier_convergence.get("enabled", True)),
                    key=form_widget_key("ui_free_carrier_convergence_enabled"),
                    help="关闭后仍计算 FCA 吸收、载流子演化和残差；仅取消其停止条件，保留验证告警。",
                )
                carrier_metric_labels = {
                    "按载流子光学效应（FCA 光深、相位和吸收能量；推荐）": (
                        "optical_effect"
                    ),
                    "按完整载流子密度场最大相对变化（更严格）": "field",
                }
                configured_carrier_metric = str(
                    carrier_convergence.get("metric", "optical_effect")
                )
                selected_carrier_metric = choice_selectbox(
                    "跨脉冲载流子收敛判据",
                    list(carrier_metric_labels),
                    index=(
                        list(carrier_metric_labels.values()).index(
                            configured_carrier_metric
                        )
                        if configured_carrier_metric
                        in carrier_metric_labels.values()
                        else 0
                    ),
                    key=form_widget_key(
                        "ui_free_carrier_convergence_metric"
                    ),
                    help=(
                        "这是周期固定点的第三个停止条件。光学效应判据检查"
                        "载流子对当前 Seed 真正能造成的 FCA、相位和吸收能量"
                        "变化，密度场判据则要求所有网格点的载流子密度本身"
                        "都收敛。"
                    ),
                )
                carrier_convergence["metric"] = carrier_metric_labels[
                    selected_carrier_metric
                ]
            if periodic_cfg["pulse_map_initialization"] == "reduced_pulse_map":
                reduced_map = periodic_cfg.setdefault("reduced_pulse_map", {})
                if str(reduced_map.get("model", "phase_insensitive")) == (
                    "axisymmetric_coherent_kerr"
                ):
                    st.success(
                        "初始化顺序：周期平均 → 轴对称相干 E(f,r) Kerr 低维脉冲图"
                        " → 当前完整主脉冲 pulse map。reduced 只负责加速靠近固定点；"
                        "最终布居和输出必须由完整主脉冲按上面的容差收敛。"
                    )
                else:
                    reduced_map["temporal_bins"] = 1
                    reduced_map["spectral_bins"] = max(
                        int(reduced_map.get("spectral_bins", 33)),
                        33,
                    )
                    st.success(
                        "初始化顺序：周期平均 → 低维整脉冲 FN/布居图 → 当前完整"
                        "主脉冲 pulse map。reduced 结果不会作为最终结果。"
                    )
            st.caption(
                "最终页面报告收敛固定点上的 Seed 输出；开启加速后横轴是周期映射评估次数，"
                "不是系统真实经历的连续脉冲数。选择低维初值时，最终仍必须由当前"
                "完整主脉冲满足相同的周期收敛条件。"
            )
    if repetition_rate <= 0.0:
        cfg["simulation"]["pump_duration_s"] = sci(
            "单次 Seed 前的 Pump 预备时间 (s)",
            cfg["simulation"]["pump_duration_s"],
            key=form_widget_key("ui_pump_duration"),
        )
        cfg["simulation"]["pump_time_steps"] = int(st.number_input(
            "预备 Pump 时间步", 1, 100000,
            int(cfg["simulation"]["pump_time_steps"]),
            key=form_widget_key("ui_pump_time_steps"),
        ))


def _render_solver_override_controls(cfg: dict[str, Any], *, full_josa: bool) -> None:
    """Render preset overrides once, next to the preset that they override."""

    physics = cfg["physics"]
    periodic = cfg["simulation"].get("periodic_steady_state", {})
    periodic_active = bool(periodic.get("enabled", False))
    periodic_method = str(periodic.get("method", "pulse_map"))
    pulse_map_active = periodic_active and periodic_method in {
        "pulse_map",
        "reduced_pulse_map",
    }
    strict_coupling = (
        str(cfg.get("multiphysics", {}).get("coupling_model", "one_way"))
        == "thermal_optical_fixed_point"
    )
    st.caption(
        "保持 auto 表示使用配置默认值；也可逐项明确指定。"
        "full_3d_pc 和周期稳态的硬约束会在下方锁定并注明来源。"
    )
    if st.button(
        "清除逐项设置，恢复配置默认值",
        width="stretch",
        key="reset_solver_overrides",
        help="不会改动晶体、Pump、Seed、光路、网格或性能参数。",
    ):
        for key in (
            "optical_propagation",
            "pump_propagation",
            "population_dynamics",
            "pump_depletion",
            "gain_saturation",
            "thermal_model",
            "refractive_index_model",
            "interface_model",
            "nonlinear_phase_model",
            "nonlinear_loss_model",
        ):
            physics[key] = "auto"
        install_form_config(cfg, update_json_editor=False)
        st.rerun()

    if full_josa:
        st.info(
            "full_3d_pc 已锁定：Seed FFT、Pump FFT、瞬态最终脉冲布居、泵浦耗尽、"
            "增益饱和和光学面记录。退出该 JOSA 层级后可重新编辑这些项。"
        )
    elif strict_coupling:
        st.info(
            "热光固定点要求 Pump 能响应三维折射率场。波动光学推荐 "
            "hybrid_abcd_fft；非相干高发散泵浦可选 stochastic_raytrace。"
            "热场和折射率模型在“③ 介质”页锁定为 steady_3d / thermal_3d。"
        )
    if not full_josa:
        physics["pump_depletion"] = auto_boolean(
            "泵浦耗尽", physics.get("pump_depletion", "auto"),
            key=form_widget_key("ui_pump_depletion"),
        )
        configured_gain_saturation = copy.deepcopy(
            physics.get("gain_saturation", "auto")
        )
        gain_saturation_key = form_widget_key(
            "ui_gain_saturation_periodic_lock"
            if pulse_map_active
            else "ui_gain_saturation"
        )
        if pulse_map_active:
            st.session_state[gain_saturation_key] = "on — 开"
        displayed_gain_saturation = auto_boolean(
            "增益饱和",
            True if pulse_map_active else configured_gain_saturation,
            key=gain_saturation_key,
            disabled=pulse_map_active,
            help=(
                "离散脉冲周期图必须把每次 Seed 受激提取反向提交到布居，"
                "因此 gain_saturation=true 由周期合同锁定。"
                if pulse_map_active
                else None
            ),
        )
        if not pulse_map_active:
            physics["gain_saturation"] = displayed_gain_saturation

    interface_options = ["auto", "coating", "fresnel", "fresnel_angle", "ideal"]
    physics["interface_model"] = choice_selectbox(
        "界面模型", interface_options,
        index=option_index(interface_options, str(physics.get("interface_model", "auto"))),
        key=form_widget_key("ui_interface_model"),
    )

    if full_josa:
        forced_population_mode = "transient"
        steady_state_label = "steady_state — 仅 Pump 代数稳态（不含 Seed）"
        transient_label = (
            "transient — full_3d_pc 最终脉冲；平均稳态由运行页接管"
            if periodic_active and periodic_method == "cycle_average"
            else "transient — full_3d_pc 锁定"
        )
        population_help = "full_3d_pc 的脉冲输运要求瞬态布居。"
    elif periodic_active and periodic_method == "cycle_average":
        forced_population_mode = "steady_state"
        steady_state_label = "steady_state — Pump + Seed 平均稳态（由运行页接管）"
        transient_label = "transient — 时序/周期布居"
        population_help = "周期平均稳态已在运行页接管布居。"
    elif periodic_active:
        forced_population_mode = "transient"
        steady_state_label = "steady_state — 仅 Pump 代数稳态（不含 Seed）"
        transient_label = "transient — 时序/周期布居"
        population_help = "离散脉冲周期图要求瞬态布居。"
    else:
        forced_population_mode = None
        steady_state_label = "steady_state — 仅 Pump 代数稳态（不含 Seed）"
        transient_label = "transient — 时序/周期布居"
        population_help = "仅 Pump 稳态不包含重复 Seed 提取；重复放大器请打开上方周期稳态。"
    population_options = [
        "auto — 按配置默认值",
        transient_label,
        steady_state_label,
        "frozen — 跳过 Pump 预备与暗弛豫（Seed 仍可耗尽）",
    ]
    configured_population_mode = copy.deepcopy(
        physics.get("population_dynamics", "auto")
    )
    population_key = form_widget_key(
        "ui_population_dynamics_contract_lock"
        if forced_population_mode is not None
        else "ui_population_dynamics"
    )
    stored_population = st.session_state.get(population_key)
    desired_token = forced_population_mode or str(
        stored_population or configured_population_mode
    ).split(" ")[0]
    if desired_token not in {"auto", "transient", "steady_state", "frozen"}:
        desired_token = "auto"
    st.session_state[population_key] = next(
        option for option in population_options if option.split(" ")[0] == desired_token
    )
    population_label = choice_selectbox(
        "布居动力学", population_options,
        key=population_key,
        disabled=forced_population_mode is not None,
        help=population_help,
    )
    if forced_population_mode is None:
        physics["population_dynamics"] = population_label.split(" ")[0]

    if not full_josa:
        optical_options = ["auto", "hybrid_abcd_fft", "fft", "abcd", "off"]
        physics["optical_propagation"] = choice_selectbox(
            "种子空间传播", optical_options,
            index=option_index(optical_options, str(physics.get("optical_propagation", "auto"))),
            key=form_widget_key("ui_optical_propagation"),
            format_func=lambda value: {
                "auto": "auto — 按配置默认值",
                "hybrid_abcd_fft": "hybrid_abcd_fft — 晶体外逐波长 ABCD，晶体内 FFT",
                "fft": "fft — 从源面开始全程 FFT",
                "abcd": "abcd — 快速 Gaussian ABCD（晶体内采样传播）",
                "off": "off — 规定光斑，不传播",
            }[value],
        )
        pump_options = (
            ["hybrid_abcd_fft", "fft", "stochastic_raytrace"]
            if strict_coupling
            else [
                "auto",
                "prescribed_caustic",
                "super_gaussian_abcd",
                "stochastic_raytrace",
                "hybrid_abcd_fft",
                "fft",
                "abcd",
            ]
        )
        configured_pump_propagation = str(
            physics.get("pump_propagation", "auto")
        )
        pump_key = form_widget_key(
            "ui_pump_propagation_coupling_lock"
            if strict_coupling
            else "ui_pump_propagation"
        )
        if strict_coupling:
            requested_pump = str(st.session_state.get(pump_key, configured_pump_propagation))
            if requested_pump not in {
                "fft",
                "hybrid_abcd_fft",
                "stochastic_raytrace",
            }:
                requested_pump = "hybrid_abcd_fft"
            st.session_state[pump_key] = requested_pump
        displayed_pump_propagation = choice_selectbox(
            "泵浦空间传播", pump_options,
            key=pump_key,
            disabled=False,
            help=(
                "hybrid_abcd_fft 用解析 Gaussian 保存晶体外功率和入口波前，"
                "再在晶体内用 FFT 响应三维热折射率；fft 从光源平面开始采样，"
                "必须保证整个外部光路的窗口都足够大。stochastic_raytrace "
                "逐条传播非相干几何光线，并按晶体内实际路径沉积吸收功率；"
                "super_gaussian_abcd 保持 SG 强度形状，用一阶光学传播束宽。"
            ),
            format_func=lambda value: {
                "auto": "auto — 按配置默认值",
                "prescribed_caustic": "prescribed_caustic — 直接规定晶体内焦散",
                "super_gaussian_abcd": "super_gaussian_abcd — 自相似 SG 强度包络 + ABCD",
                "stochastic_raytrace": "stochastic_raytrace — 随机光线 + 三维路径沉积",
                "hybrid_abcd_fft": "hybrid_abcd_fft — 晶体外 ABCD，晶体内 FFT",
                "fft": "fft — 全程采样波动光学",
                "abcd": "abcd — 快速二阶矩 Gaussian 传播",
            }[value],
            **({} if pump_key in st.session_state else {
                "index": option_index(
                    pump_options,
                    configured_pump_propagation,
                )
            }),
        )
        if (
            not strict_coupling
            or configured_pump_propagation
            in {"fft", "hybrid_abcd_fft", "stochastic_raytrace"}
        ):
            physics["pump_propagation"] = displayed_pump_propagation
    with st.expander("传播模型适用范围", expanded=False):
        applicability = physics.setdefault("applicability", {})
        applicability["maximum_paraxial_angle_deg"] = st.number_input(
            "旁轴后端最大合成倾角（°）",
            min_value=0.01,
            max_value=88.99,
            value=float(
                applicability.get("maximum_paraxial_angle_deg", 10.0)
            ),
            step=0.5,
            key=form_widget_key("ui_maximum_paraxial_angle_deg"),
            help=(
                "用于 ABCD、规定焦散、FFT 和 hybrid 的适用性门；"
                "随机光线后端不套用这个旁轴限制。"
            ),
        )
        applicability[
            "maximum_fft_transverse_nyquist_fraction"
        ] = st.number_input(
            "FFT 倾斜载波最大 Nyquist 占比",
            min_value=0.01,
            max_value=1.0,
            value=float(
                applicability.get(
                    "maximum_fft_transverse_nyquist_fraction", 0.8
                )
            ),
            step=0.05,
            key=form_widget_key(
                "ui_maximum_fft_transverse_nyquist_fraction"
            ),
            help=(
                "FFT/hybrid 的横向倾斜载波超过该比例时标记为超出采样适用范围。"
            ),
        )
    st.divider()
    st.markdown("**计算性能**")
    performance_cfg = cfg.setdefault("performance", {})
    performance_columns = st.columns(2)
    with performance_columns[0]:
        fft_backend_options = ["auto", "scipy", "numpy"]
        performance_cfg["fft_backend"] = choice_selectbox(
            "FFT 计算后端", fft_backend_options,
            index=option_index(fft_backend_options, str(performance_cfg.get("fft_backend", "auto"))),
            key=form_widget_key("ui_fft_backend"),
            help="auto 优先使用可选的 SciPy 多线程 FFT；未安装时安全回退 NumPy。",
        )
    with performance_columns[1]:
        performance_cfg["fft_workers"] = int(st.number_input(
            "FFT 线程数（0=全部可用核）", min_value=0, max_value=256,
            value=int(performance_cfg.get("fft_workers", 0)), step=1,
            key=form_widget_key("ui_fft_workers"),
        ))
    performance_cfg["fft_min_elements"] = int(st.number_input(
        "启用多线程 FFT 的最小数组元素数", min_value=1, max_value=100000000,
        value=int(performance_cfg.get("fft_min_elements", 4096)), step=1024,
        key=form_widget_key("ui_fft_min_elements"),
    ))


def _render_result_recording_controls(cfg: dict[str, Any]) -> None:
    """Render density presets and feature-coupled diagnostic categories."""

    simulation = cfg["simulation"]
    recording = simulation.setdefault(
        "result_recording",
        copy.deepcopy(DEFAULTS["simulation"]["result_recording"]),
    )
    longitudinal = recording.setdefault(
        "longitudinal",
        copy.deepcopy(
            DEFAULTS["simulation"]["result_recording"]["longitudinal"]
        ),
    )
    timing_options = ["all_propagations", "final_output"]
    recording["diagnostics_timing"] = choice_selectbox(
        "M²、沿程与焦散诊断何时计算",
        timing_options,
        index=option_index(timing_options, recording.get("diagnostics_timing", "all_propagations")),
        format_func=lambda value: {
            "all_propagations": "每次传播（沿用原行为）",
            "final_output": "仅每个工况的最终输出",
        }[value],
        key=form_widget_key("ui_result_diagnostics_timing"),
        help="仅最终输出：跳过 Cycle Average、脉冲内迭代和热迭代中的沿程诊断、M² 与焦散；最终结果按下方各开关和采样间隔完整计算。保留功率、收敛残差和必要的物理验证。")
    if recording["diagnostics_timing"] == "final_output":
        st.caption("下方开关决定最终保存哪些诊断。M² 的 Z 间隔设为 1，即最终沿每个物理片计算；出口焦散仍扫描全部指定平面。")
    level_options = [
        "simple — 简单记录（保持现有结果）",
        "medium — 中等记录（增加每轮标量）",
        "detailed — 详细记录（增加物理 Z 层标量）",
    ]
    level_label = choice_selectbox(
        "记录层级",
        level_options,
        index=option_index(
            level_options,
            str(recording.get("level", "simple")),
        ),
        key=form_widget_key("ui_result_recording_level"),
        help="记录层级只改变诊断数据密度，不改变传播、增益或热学方程。",
    )
    recording["level"] = level_label.split(" ")[0]
    medium_active = recording["level"] in {"medium", "detailed"}
    detailed_active = recording["level"] == "detailed"
    if recording["level"] == "simple":
        st.caption("简单档保持原有最终结果、收敛摘要和完整末次脉冲输出。")
    elif recording["level"] == "medium":
        st.caption("中等档额外保留外层热光与内层周期/布居迭代的标量历史。")
    else:
        st.caption("详细档再按物理 Z 单元聚合光斑、能量、非线性和稀疏 M² 标量。")

    recording["iteration_history"] = choice_checkbox(
        "记录迭代标量历史",
        value=bool(recording.get("iteration_history", True)),
        key=form_widget_key("ui_result_iteration_history"),
        disabled=not medium_active,
        help="保留外层迭代编号、内层循环编号、残差、输出能量、峰值、B 积分和可用的 M² 标量。",
    )

    effective = resolve_physics(cfg)
    signal_model = resolve_signal_model(cfg)
    sampled_backend = signal_model in {"spectral_time", "coherent_cpa"}
    coherent_backend = signal_model == "coherent_cpa"
    nonlinear_available = coherent_backend and (
        str(effective.get("nonlinear_phase_model", "off")) != "off"
        or str(effective.get("nonlinear_loss_model", "off")) != "off"
    )
    fidelity = str(
        cfg.get("seed", {}).get("coherent", {}).get(
            "fidelity_mode", "configured"
        )
    )
    modal_available = coherent_backend and fidelity in {
        "axisymmetric_kerr",
        "axisymmetric_transient_gain",
    }

    st.markdown("**详细档：晶体内纵向标量**")
    longitudinal_available = detailed_active and sampled_backend
    longitudinal["enabled"] = choice_checkbox(
        "启用物理 Z 层记录",
        value=bool(longitudinal.get("enabled", True)),
        key=form_widget_key("ui_result_longitudinal_enabled"),
        disabled=not longitudinal_available,
        help="每个被选中的物理 Z 单元只保存一行；自适应子步会聚合，并记录实际子步数。",
    )
    active_longitudinal = longitudinal_available and bool(
        longitudinal["enabled"]
    )
    if detailed_active and not sampled_backend:
        st.caption("当前求解家族不提供逐 Z 的采样场；纵向选项会保留，但本次不生效。")

    longitudinal["z_stride"] = int(
        st.number_input(
            "每隔多少个物理 Z 单元记录一行",
            min_value=1,
            max_value=100000,
            value=max(int(longitudinal.get("z_stride", 1)), 1),
            step=1,
            key=form_widget_key("ui_result_z_stride"),
            disabled=not active_longitudinal,
        )
    )
    category_columns = st.columns(2)
    with category_columns[0]:
        longitudinal["beam"] = choice_checkbox(
            "光斑与能量",
            value=bool(longitudinal.get("beam", True)),
            key=form_widget_key("ui_result_record_beam"),
            disabled=not active_longitudinal,
            help="记录能量、峰值/能量加权通量、有效面积、质心和 D4σ 半径。",
        )
        longitudinal["gain"] = choice_checkbox(
            "增益与提取",
            value=bool(longitudinal.get("gain", True)),
            key=form_widget_key("ui_result_record_gain"),
            disabled=not active_longitudinal,
            help="记录主动介质能量交换与该单元的净能量变化。",
        )
        longitudinal["nonlinear"] = choice_checkbox(
            "非线性、PCR 与损耗",
            value=bool(longitudinal.get("nonlinear", True)),
            key=form_widget_key("ui_result_record_nonlinear"),
            disabled=not (active_longitudinal and nonlinear_available),
            help="记录峰值与能量加权强度、峰值功率、P/Pcr、B 增量、非线性损耗和载流子。",
        )
    with category_columns[1]:
        longitudinal["modes"] = choice_checkbox(
            "方位模态占比",
            value=bool(longitudinal.get("modes", True)),
            key=form_widget_key("ui_result_record_modes"),
            disabled=not (active_longitudinal and modal_available),
            help="轴对称相干后端记录 m=0、|m|=1、|m|=2 中实际保留阶数的能量占比。",
        )
        longitudinal["m2_checkpoints"] = choice_checkbox(
            "稀疏 M² 检查点",
            value=bool(longitudinal.get("m2_checkpoints", True)),
            key=form_widget_key("ui_result_record_m2"),
            disabled=not active_longitudinal,
            help="M² 需要额外的空间频率矩计算，因此使用独立的较稀 Z 间隔。",
        )
        longitudinal["m2_z_stride"] = int(
            st.number_input(
                "每隔多少个物理 Z 单元计算一次 M²",
                min_value=1,
                max_value=100000,
                value=max(int(longitudinal.get("m2_z_stride", 5)), 1),
                step=1,
                key=form_widget_key("ui_result_m2_z_stride"),
                disabled=not (
                    active_longitudinal
                    and bool(longitudinal["m2_checkpoints"])
                ),
            )
        )
    if active_longitudinal and not nonlinear_available:
        st.caption("非线性标量只在相干 Kerr 或强场损耗启用时生效。")
    if active_longitudinal and not modal_available:
        st.caption("方位模态占比只适用于轴对称相干后端；其他记录不受影响。")

    longitudinal["intensity_snapshots"] = choice_checkbox(
        "记录晶体内瞬时强度截面",
        value=bool(longitudinal.get("intensity_snapshots", False)),
        key=form_widget_key("ui_result_intensity_snapshots"),
        disabled=not (active_longitudinal and coherent_backend),
        help="保存当地峰值功率时刻，以及固定入口峰值时刻的强度。入口始终记录；轴对称后端保留原始径向／方位角网格。",
    )
    longitudinal["intensity_z_stride"] = int(st.number_input(
        "每隔多少个物理 Z 单元保存强度截面",
        min_value=1, max_value=100000,
        value=max(int(longitudinal.get("intensity_z_stride", 1)), 1), step=1,
        key=form_widget_key("ui_result_intensity_z_stride"),
        disabled=not (active_longitudinal and coherent_backend and longitudinal["intensity_snapshots"]),
        help="1 表示保存入口和每个单元出口；最后出口始终记录。不保存自适应子步的完整时空场。",
    ))

    st.divider()
    st.markdown("**现有结果与文件**")
    full_josa = _full_josa_selected(cfg)
    configured_record_planes = bool(
        cfg["physics"].get("record_optical_planes", True)
    )
    record_planes_key = form_widget_key(
        "ui_record_planes_full_josa_lock"
        if full_josa
        else "ui_record_planes"
    )
    if full_josa:
        st.session_state[record_planes_key] = True
    displayed_record_planes = choice_checkbox(
        "保存各光学面",
        key=record_planes_key,
        disabled=full_josa,
        help="full_3d_pc 固定开启；其他模式可关闭以节省内存。",
        **(
            {}
            if record_planes_key in st.session_state
            else {"value": configured_record_planes}
        ),
    )
    if not full_josa:
        cfg["physics"]["record_optical_planes"] = displayed_record_planes
    volume_options = [
        "last_and_cumulative — 末程 + 累计（推荐）",
        "last_only — 仅末程（最省内存）",
        "per_pass — 每一程 + 累计（内存较大）",
    ]
    volume_label = choice_selectbox(
        "晶体内 Seed 体数据",
        volume_options,
        index=option_index(
            volume_options,
            str(
                simulation.get(
                    "signal_volume_recording", "last_and_cumulative"
                )
            ),
        ),
        key=form_widget_key("ui_signal_volume_recording"),
        help="不改变传播物理，只决定保存末程、累计或逐程三维诊断场。",
    )
    simulation["signal_volume_recording"] = volume_label.split(" ")[0]
    simulation["save_vtk"] = choice_checkbox(
        "保存 VTK",
        bool(simulation.get("save_vtk", False)),
        key=form_widget_key("ui_save_vtk"),
    )


def _solver_value_text(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


def _resolved_thermal_label(
    cfg: dict[str, Any],
    effective: dict[str, Any],
) -> str:
    """Translate the internal ``configured`` sentinel into a user model."""

    model = str(effective.get("thermal_model", "off"))
    if model != "configured":
        return model
    thermal = cfg.get("material", {}).get("thermal", {})
    if not bool(thermal.get("enabled", False)):
        return "off"
    return (
        "transient_3d"
        if bool(thermal.get("transient", {}).get("enabled", False))
        else "steady_3d"
    )


def _render_effective_solver_summary(cfg: dict[str, Any], *, full_josa: bool) -> None:
    physics = cfg["physics"]
    effective = resolve_physics(cfg)
    periodic = cfg["simulation"].get("periodic_steady_state", {})
    periodic_active = bool(periodic.get("enabled", False))
    periodic_method = str(periodic.get("method", "pulse_map"))
    thermal_fixed_point = (
        str(cfg.get("multiphysics", {}).get("coupling_model", "one_way"))
        == "thermal_optical_fixed_point"
    )
    locked_by_full = {
        "optical_propagation", "pump_propagation", "population_dynamics",
        "pump_depletion", "gain_saturation", "thermal_model",
        "refractive_index_model", "nonlinear_phase_model", "record_optical_planes",
        "nonlinear_loss_model",
    }

    def source_for(key: str) -> str:
        if full_josa and key in locked_by_full:
            if key == "population_dynamics" and periodic_active and periodic_method == "cycle_average":
                return "full_3d_pc 最终脉冲 + 周期平均稳态"
            return "full_3d_pc 锁定"
        if key == "population_dynamics" and periodic_active:
            return "周期稳态锁定"
        if thermal_fixed_point and key in {
            "thermal_model",
            "refractive_index_model",
        }:
            return "热光固定点锁定"
        if thermal_fixed_point and key == "pump_propagation":
            return "热光固定点限定为可响应三维反馈的后端"
        return "配置默认值" if physics.get(key, "auto") == "auto" else "逐项设置"

    rows = [
        {
            "模块": "多物理耦合",
            "配置值": str(cfg.get("multiphysics", {}).get("coupling_model", "one_way")),
            "实际生效": str(cfg.get("multiphysics", {}).get("coupling_model", "one_way")),
            "来源": "明确模块",
        },
        {
            "模块": "Seed 增益输运",
            "配置值": str(cfg["simulation"].get("signal_model", "auto")),
            "实际生效": resolve_signal_model(cfg),
            "来源": "auto 解析" if cfg["simulation"].get("signal_model", "auto") == "auto" else "明确选择",
        },
        {
            "模块": "Seed 空间增益闭合",
            "配置值": str(
                cfg["seed"].get("spatial_gain_transport", {}).get(
                    "model", "local_field"
                )
            ),
            "实际生效": str(
                cfg["seed"].get("spatial_gain_transport", {}).get(
                    "model", "local_field"
                )
            ),
            "来源": "明确独立模块",
        },
        {
            "模块": "ASE / 寄生激光",
            "配置值": str(cfg.get("ase", {}).get("model", "off")),
            "实际生效": str(cfg.get("ase", {}).get("model", "off")),
            "来源": "明确模块",
        },
        {
            "模块": "局部光谱反馈",
            "配置值": str(cfg.get("material", {}).get("spectroscopic_response", {}).get("mode", "off")),
            "实际生效": str(cfg.get("material", {}).get("spectroscopic_response", {}).get("mode", "off")),
            "来源": "明确模块",
        },
    ]
    labels = {
        "optical_propagation": "Seed 空间传播",
        "pump_propagation": "Pump 空间传播",
        "population_dynamics": "布居动力学",
        "pump_depletion": "泵浦耗尽",
        "gain_saturation": "增益饱和",
        "thermal_model": "热场",
        "refractive_index_model": "折射率",
        "interface_model": "界面",
        "nonlinear_phase_model": "非线性相位",
        "nonlinear_loss_model": "强场非线性损耗",
        "record_optical_planes": "光学面记录",
    }
    rows.extend(
        {
            "模块": label,
            "配置值": _solver_value_text(physics.get(key, "auto")),
            "实际生效": (
                _resolved_thermal_label(cfg, effective)
                if key == "thermal_model"
                else resolve_population_dynamics(cfg)
                if key == "population_dynamics"
                else _solver_value_text(effective[key])
            ),
            "来源": source_for(key),
        }
        for key, label in labels.items()
    )
    for row in rows:
        st.markdown(f"**{row['模块']}**　`{row['实际生效']}`")
        st.caption(f"配置：{row['配置值']}　·　来源：{row['来源']}")
    st.caption("这里是本次表单真正传给求解器的组合；运行结果仍会记录实际后端和安全回退诊断。")
    if st.button(
        "生成 / 刷新完整模型说明",
        width="stretch",
        key="generate_model_description",
        help=(
            "根据当前表单重新解析实际求解栈，并在主页面生成从 Pump/Seed 发光面、"
            "晶体内部、热光反馈到出光面验证的完整说明与功能清单。"
        ),
    ):
        st.session_state["show_model_description"] = True
    if st.session_state.get("show_model_description", False):
        st.success("完整物理说明已在主页面光路预览下方生成，并会随当前表单自动刷新。")


def render_generated_model_description(cfg: dict[str, Any]) -> None:
    """Render the requested, configuration-specific end-to-end explanation."""

    with st.container(border=True):
        if not bool(st.session_state.get("show_model_description", False)):
            launch_text, launch_button = st.columns([5, 2])
            with launch_text:
                st.subheader("当前配置的完整物理链路")
                st.caption(
                    "生成从 Pump/Seed 发光面、实际光路、能级布居、局部增益、"
                    "热与应力反馈直到出光面验证的物理说明。"
                )
            with launch_button:
                if st.button(
                    "生成完整物理说明",
                    width="stretch",
                    key="generate_model_description_main",
                    help="与侧栏按钮等价；只生成说明，不会启动数值求解。",
                ):
                    st.session_state["show_model_description"] = True
                    st.rerun()
            return

        description = describe_configuration(cfg)
        validation_error: str | None = None
        try:
            validate_config(cfg)
        except Exception as exc:  # The description must remain visible for conflicts.
            validation_error = str(exc)

        heading, close_column = st.columns([8, 1])
        with heading:
            st.subheader("当前配置的完整物理链路")
            st.caption(
                "以下内容由当前表单实时生成；“实际生效”已解析 auto、配置默认值、"
                "JOSA/周期稳态/热光固定点锁定，不是配置字段的简单抄录。"
            )
        with close_column:
            if st.button(
                "收起",
                key="hide_model_description",
                help="只收起说明，不改变任何仿真配置。",
            ):
                st.session_state["show_model_description"] = False
                st.rerun()
        if validation_error is not None:
            st.error(
                "当前组合尚不能运行，说明仍按你正在编辑的配置生成。"
                f"首个配置冲突：{validation_error}"
            )
        st.info(
            "Pump / Seed 输入面　→　实际光学电路　→　三维 Pump 吸收与能级布居　→　"
            "Seed 局部增益和多程提取　→　能量去向与热源　→　三维热/应力/折射率反馈　→　"
            "网格守恒映射　→　出光面与定量有效性合同"
        )
        if description.notices:
            st.markdown("**锁定、冲突与适用性提示**")
            for notice in description.notices:
                if notice.startswith("冲突："):
                    st.error(notice)
                else:
                    st.info(notice)
        for index, stage in enumerate(description.stages, start=1):
            with st.container(border=True):
                st.markdown(f"#### {index}. {stage.title}")
                st.markdown(stage.explanation)
                if stage.key_relations:
                    st.caption("关键物理关系：" + "　｜　".join(stage.key_relations))
        with st.expander(
            "当前模型功能清单（含关闭项与锁定项）",
            expanded=False,
        ):
            st.dataframe(
                [
                    {
                        "计算环节": item.stage,
                        "功能": item.feature,
                        "表单/配置值": item.configured,
                        "实际生效": item.effective,
                        "来源或锁定": item.source,
                        "物理与算法说明": item.explanation,
                    }
                    for item in description.features
                ],
                hide_index=True,
                width="stretch",
            )
        st.download_button(
            "下载完整物理说明 Markdown",
            description.as_markdown().encode("utf-8"),
            file_name="amplifier_model_description.md",
            mime="text/markdown",
            key="download_model_description",
            help="下载的是当前表单对应的说明，不包含尚未运行的数值结果。",
        )


def _render_model_solver_controls(
    cfg: dict[str, Any],
    *,
    show_advanced: bool,
) -> tuple[str, Any | None]:
    cfg["simulation"]["name"] = st.text_input(
        "算例名称", value=str(cfg["simulation"].get("name", "amplifier")),
        key=form_widget_key("ui_simulation_name"),
    )
    if st.button(
        "快速配置", key="apply_quick_physics_config", width="stretch",
        help="只设置下述八项物理模型；保留种子求解模型、径向设置、网格、采样、Kerr、损耗、界面透过率与迭代参数。",
    ):
        install_form_config(apply_quick_physics_config(cfg), update_json_editor=False)
        st.rerun()
    st.caption(
        "快速配置：Seed / Pump 混合 ABCD/FFT；瞬态布居；开启泵浦耗尽和增益饱和；"
        "三维稳态热、热折射率、镀膜界面。应用后可逐项修改。"
    )
    signal_options = [
        "auto — 自动", "fluence — 快速单色通量（可选 FN）",
        "commercial_app_fast — Gaussian duct / 等效热-Kerr透镜",
        "spectral_time — 多色×时间/FN", "josa_transport — JOSA 时空输运",
        "coherent_cpa — 相干 CPA",
    ]
    signal_label = choice_selectbox(
        "种子求解模型", signal_options,
        index=option_index(signal_options, str(cfg["simulation"].get("signal_model", "auto"))),
        key=form_widget_key("ui_signal_model"),
        format_func=lambda value: {
            "auto — 自动": "自动选择（按输入光谱与时间结构）",
            "fluence — 快速单色通量（可选 FN）": "只看能量与饱和 · 单色通量 / FN",
            "commercial_app_fast — Gaussian duct / 等效热-Kerr透镜": "Commercial App 风格快速 · Gaussian-q / 热-Kerr 透镜",
            "spectral_time — 多色×时间/FN": "多波长 + 脉内耗尽 · 不保留相干相位",
            "josa_transport — JOSA 时空输运": "JOSA 时空速率方程 · 不保留相干相位",
            "coherent_cpa — 相干 CPA": "完整相干脉冲 · 衍射 / 色散 / Kerr",
        }[str(value)],
        help=(
            "按你需要的输出选择：只比较能量时用 fluence；需要 Commercial App 风格热透镜快速扫描时用 Commercial App Fast；"
            "需要脉内人口竞争但不追踪相干相位时用 spectral_time/JOSA；需要压缩、色散或 SPM 相位时用 coherent CPA。"
        ),
    )
    cfg["simulation"]["signal_model"] = signal_label.split(" ")[0]
    with st.container(border=True):
        st.markdown("#### Pump · Seed · 热场收敛")
        coupling_cfg = cfg.setdefault("multiphysics", {})
        iteration_options = [
            "one_way — 单向热场；Pump + Seed 求稳态",
            "nested — 每轮热场内先完成脉冲收敛",
            "joint — Joint 交替更新；最终完整闭合检查",
            "joint_frozen — Joint 交替更新；固定热场验证后输出",
        ]
        configured_scheme = (
            "one_way" if coupling_cfg.get("coupling_model", "one_way") == "one_way"
            else coupling_cfg.get("iteration_scheme", "nested")
        )
        iteration_label = choice_selectbox(
            "迭代方式", iteration_options,
            index=option_index(iteration_options, configured_scheme),
            key=form_widget_key("ui_multiphysics_iteration_scheme"),
            help="Joint 每轮推进指定数量的 Pump + Seed 周期，再更新热场。两个 Joint 选项分别采用完整热闭合检查或固定热场的有限次脉冲验证。",
        )
        selected_scheme = iteration_label.split(" ")[0]
        coupling_cfg["coupling_model"] = "one_way" if selected_scheme == "one_way" else "thermal_optical_fixed_point"
        if selected_scheme != "one_way":
            coupling_cfg["iteration_scheme"] = selected_scheme
        joint_mode = selected_scheme in {"joint", "joint_frozen"}
        if joint_mode:
            periodic_cfg = cfg["simulation"]["periodic_steady_state"]
            periodic_cfg["method"] = "pulse_map"
            st.caption("内层检查布居、Seed 输出及已启用载流子的变化；外层检查热光反馈变化。两者同时低于各自阈值才累计达标轮数。")
            joint_columns = st.columns(2)
            with joint_columns[0]:
                periodic_cfg["relative_tolerance"] = sci(
                    "内层相对容差（0.003 = 0.3%）", periodic_cfg.get("relative_tolerance", 1e-5),
                    key=form_widget_key("ui_periodic_tolerance"),
                )
                coupling_cfg["joint_cycles_per_iteration"] = int(st.number_input(
                    "每轮热更新前的脉冲周期数", min_value=1, max_value=10000,
                    value=int(coupling_cfg.get("joint_cycles_per_iteration", 1)),
                    key=form_widget_key("ui_joint_cycles_per_iteration"),
                ))
            with joint_columns[1]:
                coupling_cfg["relative_tolerance"] = sci(
                    "外层相对容差（0.005 = 0.5%）",
                    coupling_cfg.get("relative_tolerance", 1e-4),
                    key=form_widget_key("ui_multiphysics_relative_tolerance"),
                )
                coupling_cfg["joint_consecutive_iterations"] = int(st.number_input(
                    "内外同时达标的连续轮数", min_value=1, max_value=1000,
                    value=int((coupling_cfg.get("joint_consecutive_iterations") or periodic_cfg.get("consecutive_cycles", 3))),
                    key=form_widget_key("ui_joint_consecutive_iterations"),
                ))
                if selected_scheme == "joint_frozen":
                    coupling_cfg["frozen_probe_cycles"] = int(st.number_input(
                        "固定热场验证周期数（须全部达标）", min_value=1, max_value=10000,
                        value=int(coupling_cfg.get("frozen_probe_cycles", 3)),
                        key=form_widget_key("ui_frozen_probe_cycles"),
                    ))
            if selected_scheme == "joint_frozen":
                st.info("同时连续达标 → 固定热场 → 验证指定次数的完整脉冲。每次内层残差均达标才输出；任一次未达标就返回交替迭代。此流程不再重算最终热场闭合。")
            else:
                st.caption("同时连续达标后，固定热场求完整内层收敛，再重算热场验证闭合；未通过则继续交替迭代。")
        if coupling_cfg["coupling_model"] == "one_way":
            st.caption(
                "单向热场：先在无热畸变下求 Pump + Seed 稳态，再求一次热响应，把 Δn 作用到最终 Seed；热场不反馈到前面的周期求解。"
            )
        else:
            st.info(
                "自洽模式反复传播 Pump、求布居、热场和 Δn，直到 Pump 吸收、布居与热相位共同收敛；"
                "局部光谱反馈、ASE/TPL 和 Seed 强场产热是否进入闭环由下面各自的开关决定。"
            )
            coupling_columns = st.columns(2)
            with coupling_columns[0]:
                coupling_cfg["minimum_iterations"] = int(
                    st.number_input(
                        "最少耦合迭代",
                        min_value=2,
                        max_value=100,
                        value=int(coupling_cfg.get("minimum_iterations", 2)),
                        key=form_widget_key("ui_multiphysics_minimum_iterations"),
                    )
                )
                if not joint_mode:
                    coupling_cfg["relative_tolerance"] = sci(
                        "外层相对容差（0.005 = 0.5%）",
                        coupling_cfg.get("relative_tolerance", 1e-4),
                        key=form_widget_key("ui_multiphysics_relative_tolerance"),
                    )
            with coupling_columns[1]:
                coupling_maximum_key = form_widget_key(
                    "ui_multiphysics_maximum_iterations"
                )
                coupling_maximum_value = max(
                    int(
                        st.session_state.get(
                            coupling_maximum_key,
                            coupling_cfg.get("maximum_iterations", 20),
                        )
                    ),
                    int(coupling_cfg["minimum_iterations"]),
                )
                st.session_state[coupling_maximum_key] = coupling_maximum_value
                coupling_cfg["maximum_iterations"] = int(
                    st.number_input(
                        "最多耦合迭代",
                        min_value=int(coupling_cfg["minimum_iterations"]),
                        max_value=10000,
                        key=coupling_maximum_key,
                    )
                )
                coupling_cfg["relaxation"] = float(
                    st.number_input(
                        "Δn 松弛系数",
                        min_value=0.01,
                        max_value=1.0,
                        value=float(coupling_cfg.get("relaxation", 0.5)),
                        step=0.05,
                        key=form_widget_key("ui_multiphysics_relaxation"),
                    )
                )
            st.caption("最多迭代是计算预算；达标才能输出收敛结果，达到上限本身不算收敛。热更新阶段仍检查已启用的热光反馈项。")
            thermal_acceleration = coupling_cfg.setdefault("acceleration", {})
            thermal_acceleration_enabled = choice_toggle(
                "自动加速热光固定点",
                value=str(thermal_acceleration.get("method", "anderson"))
                == "anderson",
                key=form_widget_key("ui_multiphysics_fixed_point_acceleration"),
                help=(
                    "对 Δn 和局部光谱温度状态使用带保护的 Anderson 加速；"
                    "内部周期稳态同时复用上一轮布居作为初值。"
                ),
            )
            thermal_acceleration["method"] = (
                "anderson" if thermal_acceleration_enabled else "off"
            )
        seed_heat_options = [
            "off — 不把 Seed 不可逆损耗送入热场",
            "one_way_postprocess — Seed 不可逆损耗→热场→再传播一次 Seed",
        ]
        if coupling_cfg["coupling_model"] == "thermal_optical_fixed_point":
            seed_heat_options.append(
                "fixed_point — Pump/布居/热场/Seed 不可逆产热共同迭代"
            )
        configured_seed_heat = str(coupling_cfg.get("seed_heat_coupling", "off"))
        if configured_seed_heat == "fixed_point" and coupling_cfg["coupling_model"] != "thermal_optical_fixed_point":
            configured_seed_heat = "off"
        seed_heat_label = choice_selectbox(
            "Seed 不可逆损耗产热耦合",
            seed_heat_options,
            index=option_index(seed_heat_options, configured_seed_heat),
            key=form_widget_key("ui_seed_heat_coupling"),
            help=(
                "接收多光子吸收和自由载流子吸收；若“把 Seed 被动吸收作为热源”也开启，"
                "还接收材料被动吸收。Kerr 是纯相位，不会被重复算成热。"
            ),
        )
        coupling_cfg["seed_heat_coupling"] = seed_heat_label.split(" ")[0]
        _render_periodic_solver_controls(cfg, full_josa=_full_josa_selected(cfg))
    if resolve_signal_model(cfg) in {"spectral_time", "coherent_cpa"}:
        st.markdown("#### 时间与频率 · 尾部分箱")
        tail = cfg["seed"].setdefault("tail_binning", copy.deepcopy(DEFAULTS["seed"]["tail_binning"]))
        coverage_label = f"{100*float(tail.get('energy_coverage', .999999)):.6f}".rstrip("0").rstrip(".")
        spectral_mode = resolve_signal_model(cfg) == "spectral_time"
        temporal_column, frequency_column = st.columns(2)
        with temporal_column:
            tail["temporal_enabled"] = choice_checkbox(
                lt("时间：{coverage}% 外，两侧合计 {points} 点").format(coverage=coverage_label, points=int(tail.get("temporal_tail_points", 3))),
                value=bool(tail.get("temporal_enabled", False)),
                key=form_widget_key("ui_tail_binning_temporal"),
                disabled=not spectral_mode,
                help=lt("按初始时间能量分布选中心，中心保留原采样；两侧尾部合计 {points} 点。仅用于 Spectral Time。").format(points=int(tail.get('temporal_tail_points', 3))))
        with frequency_column:
            tail["spectral_enabled"] = choice_checkbox(
                lt("频率：{coverage}% 外，两侧合计 {points} 点").format(coverage=coverage_label, points=int(tail.get("spectral_tail_points", 3))),
                value=bool(tail.get("spectral_enabled", False)),
                key=form_widget_key("ui_tail_binning_spectral"),
                disabled=not (spectral_mode or (
                    cfg["seed"]["coherent"].get("fidelity_mode") == "axisymmetric_kerr"
                    and not cfg["seed"]["coherent"].get("transient_gain", {}).get("enabled", False))),
                help=lt("按初始频率能量分布选中心，中心一频率一点；两侧尾部合计 {points} 点。当前径向 Coherent 的增益采用分箱，衍射只更新中心频率；完整 FFT 保留。").format(points=int(tail.get('spectral_tail_points', 3))))
        st.caption("只根据初始输入分箱；两侧分别合并，合计点数不是每侧点数。传播途中不重新选区间。")
        if not spectral_mode:
            if cfg["seed"]["coherent"].get("transient_gain", {}).get("enabled", False):
                st.caption("Front-to-tail 的时间保护在 Seed 的“时间软窗尾部保护”中设置；这里的时间开关只用于 Spectral Time，频谱尾部合箱暂不用于瞬态增益。完整 FFT 保留。")
            else:
                st.caption("Coherent 使用频率开关；时间开关保留供 Spectral Time 使用，FFT 与 Kerr 的时间网格保持完整。")
    if resolve_signal_model(cfg) == "spectral_time":
        st.markdown("#### Spectral Time · 计算网格")
        spectral_cfg = cfg["seed"].setdefault("spectral_time", copy.deepcopy(DEFAULTS["seed"]["spectral_time"]))
        use_radial = choice_checkbox(
            "Spectral Time 使用 Axisymmetric 径向 / 角向网格",
            value=spectral_cfg.get("spatial_backend", "cartesian") == "axisymmetric",
            key=form_widget_key("ui_spectral_time_axisymmetric"),
            help="同时用于 Cycle Average 和后续脉冲；保留所选有限角向模式，使用径向衍射。热耦合设置保持独立。")
        spectral_cfg["spatial_backend"] = "axisymmetric" if use_radial else "cartesian"
        with st.expander("时间 / 光谱分辨率（可选上限）", expanded=False):
            st.caption("启用上面的尾部分箱后，对应的手动上限不参与计算。关闭尾部分箱后才使用这里的旧设置。")
            cap_columns = st.columns(2)
            for column, field, label, default in zip(cap_columns, ("temporal_bins", "spectral_bins"), ("时间箱", "波长箱"), (9, 65)):
                with column:
                    enabled = choice_checkbox("限制 Spectral Time " + label + "数",
                        value=spectral_cfg.get(field) is not None, key=form_widget_key("ui_spectral_time_cap_" + field))
                    spectral_cfg[field] = int(st.number_input("Spectral Time " + label + "上限", min_value=1, max_value=65536,
                        value=int(spectral_cfg.get(field) or default), step=1,
                        key=form_widget_key("ui_spectral_time_" + field))) if enabled else None
        if use_radial:
            with st.expander("径向网格与角向保真度", expanded=True):
                _render_axisymmetric_grid_controls(cfg, spectral_time=True)
        st.caption("径向设置同时用于平均稳态与最终脉冲。需要多程或独立 HG 模态时使用 Cartesian；此模型不计算 Kerr/SPM 或相干压缩。")
    if resolve_signal_model(cfg) == "commercial_app_fast":
        st.markdown("#### Commercial App Fast · 当前后端专属设置")
        st.caption(
            "这里的开关只由 Commercial App Fast 读取；切换到其他 Seed 后端后会保留数值，但不会参与计算。"
        )
        commercial_app_spatial_cfg = cfg["seed"].setdefault("spatial_modes", {})
        commercial_app_spatial_model = str(
            commercial_app_spatial_cfg.get("model", "effective_m2")
        )
        if commercial_app_spatial_model == "incoherent_hg":
            st.info(
                "横向闭合：独立 HG Gaussian-q 通道共享局部布居。模阶、容量和 M² 在“② 光束 → Seed → 空间”中编辑。"
            )
        else:
            st.info(
                "横向闭合：单一 effective-M² Gaussian-q。若要启用 HG 0–2 独立传播，请在“② 光束 → Seed → 空间”选择 HG 模态逐个传播。"
            )
        commercial_app_cfg = cfg["seed"].setdefault(
            "commercial_app_fast",
            copy.deepcopy(DEFAULTS["seed"]["commercial_app_fast"]),
        )
        commercial_app_polar = choice_checkbox(
            "Commercial App Fast 使用 Axisymmetric 径向 / 角向网格",
            value=commercial_app_cfg.get("spatial_backend", "cartesian") == "axisymmetric",
            key=form_widget_key("ui_commercial_app_axisymmetric"),
            help="同时用于 Cycle Average 和后续脉冲。保持 Gaussian-q 传播和逐片三维热耦合；关闭后使用原 Cartesian 网格。",
        )
        commercial_app_cfg["spatial_backend"] = "axisymmetric" if commercial_app_polar else "cartesian"
        if commercial_app_polar:
            axis_cfg = cfg["seed"]["coherent"]["axisymmetric"]
            axis_cfg["radial_bins"] = int(st.number_input(
                "Commercial App 径向格点数", min_value=8, value=int(axis_cfg.get("radial_bins", 24)),
                step=1, key=form_widget_key("ui_commercial_app_radial_bins")))
            axis_cfg["angular_samples"] = int(st.number_input(
                "Commercial App 角向采样数", min_value=9, max_value=128,
                value=int(axis_cfg.get("angular_samples", 12)), step=1,
                key=form_widget_key("ui_commercial_app_angular_samples")))
            st.caption("网格尺寸与 Axisymmetric 设置共用。Commercial App 保留全部角向采样，不使用相干模型的 m 阶截断；晶体外光路和热场仍用 Cartesian 网格。需要均匀正方形光学窗口和 local_field 增益。")
        st.markdown("##### 1. 时间与模态")
        temporal_labels = {
            "固定等效脉宽（纯极速，无 SPM 展宽）": "fixed_duration",
            "一维 effective-area SPM（快速近似）": "effective_area_spm",
        }
        commercial_app_is_cw = (
            str(cfg["seed"].get("temporal", {}).get("type", "gaussian"))
            == "cw"
        )
        selected_temporal_label = choice_selectbox(
            "Commercial App 时间 / Kerr 展宽闭合",
            list(temporal_labels),
            index=option_index(
                list(temporal_labels),
                next(
                    label
                    for label, value in temporal_labels.items()
                    if value
                    == str(commercial_app_cfg.get("temporal_model", "fixed_duration"))
                ),
            ),
            key=form_widget_key("ui_commercial_app_temporal_model"),
            disabled=commercial_app_is_cw,
            help=(
                "固定等效脉宽只计算 Kerr 透镜；一维 effective-area 模型另沿时间轴"
                "计算相干 SPM/色散，可输出光谱展宽和压缩波形；Commercial App effective-area "
                "多光子损耗、载流子、FCA 与载流子折射也使用这条时间包络。"
            ),
        )
        commercial_app_cfg["temporal_model"] = temporal_labels[selected_temporal_label]
        commercial_app_cfg["include_relaxation_during_pulse"] = choice_checkbox(
            "Commercial App Fast 在脉冲内推进能级快速弛豫",
            value=bool(
                commercial_app_cfg.get("include_relaxation_during_pulse", False)
            ),
            key=form_widget_key("ui_commercial_app_intrapulse_relaxation"),
            disabled=(
                commercial_app_is_cw
                or commercial_app_cfg["temporal_model"] != "fixed_duration"
            ),
            help=(
                "把输入脉冲分成有序通量箱，前沿、峰值和尾沿依次耗尽同一布居，"
                "并在相邻时间箱之间推进下能级/上能级暗弛豫。所有横向通道共享同一"
                "局部布居库，因此比 JOSA/full-field 快，但不含延迟时间输运。"
            ),
        )
        if (
            not commercial_app_is_cw
            and commercial_app_cfg["temporal_model"] == "fixed_duration"
            and commercial_app_cfg["include_relaxation_during_pulse"]
        ):
            commercial_app_cfg["gain_temporal_bin_count"] = int(
                st.number_input(
                    "Commercial App Fast 增益时间分箱",
                    min_value=3,
                    max_value=4096,
                    value=int(commercial_app_cfg.get("gain_temporal_bin_count", 129)),
                    step=2,
                    key=form_widget_key("ui_commercial_app_gain_temporal_bins"),
                    help=(
                        "只控制增益/能级动力学分箱，不增加 x-y 衍射维度。应做 65/129/257 "
                        "分箱收敛检查；时间步最好显著小于最短下能级寿命。"
                    ),
                )
            )
        quality_labels = {
            "只报告设定 M²（纯高斯）": "configured_m2",
            "累计投影前残余像差的 M² 代理": "projection_residual",
        }
        selected_quality_label = choice_selectbox(
            "Commercial App 输出 M² 合同",
            list(quality_labels),
            index=option_index(
                list(quality_labels),
                next(
                    label
                    for label, value in quality_labels.items()
                    if value
                    == str(
                        commercial_app_cfg.get(
                            "beam_quality_model", "configured_m2"
                        )
                    )
                ),
            ),
            key=form_widget_key("ui_commercial_app_beam_quality_model"),
            help=(
                "纯高斯只报告输入 M²；残余像差代理会审计每次 Gaussian 投影前的"
                "超额角向二阶矩并按独立残差累计，但不等价于完整复场 ISO 测量。"
            ),
        )
        commercial_app_cfg["beam_quality_model"] = quality_labels[selected_quality_label]
        st.markdown("##### 2. 热相位与光谱反馈")
        thermal_lens_cfg = commercial_app_cfg.setdefault(
            "thermal_lens",
            copy.deepcopy(DEFAULTS["seed"]["commercial_app_fast"]["thermal_lens"]),
        )
        thermal_lens_labels = {
            "采样相位 / 二阶矩等效（通用）": "sampled_phase",
            "Commercial App 局部抛物线拟合窗口": "local_parabolic",
        }
        current_thermal_lens_model = str(
            thermal_lens_cfg.get("model", "sampled_phase")
        )
        selected_thermal_lens_label = choice_selectbox(
            "Commercial App 热相位闭合",
            list(thermal_lens_labels),
            index=list(thermal_lens_labels.values()).index(
                current_thermal_lens_model
            )
            if current_thermal_lens_model in thermal_lens_labels.values()
            else 0,
            key=form_widget_key("ui_commercial_app_thermal_lens_model"),
            help=(
                "sampled_phase 从传播场采样热相位并做二阶矩闭合；local_parabolic 只在指定的"
                "光束邻域拟合局部抛物线，再把曲率作为 Gaussian-q 热透镜。后者用于严格对齐"
                "Commercial App 风格的局部热透镜传播。"
            ),
        )
        thermal_lens_cfg["model"] = thermal_lens_labels[
            selected_thermal_lens_label
        ]
        thermal_lens_cfg["fit_radius_mode"] = "manual"
        if thermal_lens_cfg["model"] == "local_parabolic":
            fit_columns = st.columns(2)
            with fit_columns[0]:
                thermal_lens_cfg["radius_x_m"] = sci(
                    "热相位拟合半径 x (m)",
                    thermal_lens_cfg.get("radius_x_m")
                    or 2.0 * float(cfg["seed"]["waist_x_m"]),
                    key=form_widget_key("ui_commercial_app_thermal_lens_radius_x"),
                    help_text="拟合窗口应覆盖主光束，但不要把远离光束的非抛物热相位硬塞入同一个焦距。",
                )
            with fit_columns[1]:
                thermal_lens_cfg["radius_y_m"] = sci(
                    "热相位拟合半径 y (m)",
                    thermal_lens_cfg.get("radius_y_m")
                    or 2.0 * float(cfg["seed"]["waist_y_m"]),
                    key=form_widget_key("ui_commercial_app_thermal_lens_radius_y"),
                    help_text="x/y 可不同，用于像散热透镜；两者都必须为正。",
                )
            manual_fit_center = choice_checkbox(
                "手动指定热相位拟合中心",
                value=(
                    thermal_lens_cfg.get("center_x_m") is not None
                    or thermal_lens_cfg.get("center_y_m") is not None
                ),
                key=form_widget_key("ui_commercial_app_thermal_lens_manual_center"),
                help="关闭时每层跟随当前 Gaussian-q 光束中心；开启时使用下面的固定物理坐标。",
            )
            if manual_fit_center:
                center_columns = st.columns(2)
                with center_columns[0]:
                    thermal_lens_cfg["center_x_m"] = sci(
                        "拟合中心 x (m)",
                        thermal_lens_cfg.get("center_x_m") or 0.0,
                        key=form_widget_key("ui_commercial_app_thermal_lens_center_x"),
                    )
                with center_columns[1]:
                    thermal_lens_cfg["center_y_m"] = sci(
                        "拟合中心 y (m)",
                        thermal_lens_cfg.get("center_y_m") or 0.0,
                        key=form_widget_key("ui_commercial_app_thermal_lens_center_y"),
                    )
            else:
                thermal_lens_cfg["center_x_m"] = None
                thermal_lens_cfg["center_y_m"] = None
        else:
            st.caption(
                "当前不使用手动拟合半径；JSON 中保留的局部窗口参数不会参与 sampled_phase 计算。"
            )
        commercial_app_cfg["spectral_feedback"] = choice_checkbox(
            "把一维 SPM 展宽反馈到后续 z 层的粗光谱增益分箱",
            value=bool(commercial_app_cfg.get("spectral_feedback", True)),
            key=form_widget_key("ui_commercial_app_spectral_feedback"),
            disabled=(
                commercial_app_is_cw
                or commercial_app_cfg["temporal_model"] != "effective_area_spm"
            ),
            help=(
                "开启后，SPM 改变的频谱能量会映射回粗波长通道，从而影响后续层的"
                "波长相关增益与布居；关闭时只生成输出时频诊断。"
            ),
        )
        if (
            not commercial_app_is_cw
            and commercial_app_cfg["temporal_model"] == "effective_area_spm"
            and commercial_app_cfg["spectral_feedback"]
        ):
            feedback_grid_options = [
                "full_coherent_span",
                "seed_energy_support",
            ]
            commercial_app_cfg["spectral_feedback_grid_mode"] = choice_selectbox(
                "Commercial App 频谱回灌网格范围",
                feedback_grid_options,
                index=option_index(
                    feedback_grid_options,
                    str(
                        commercial_app_cfg.get(
                            "spectral_feedback_grid_mode",
                            "full_coherent_span",
                        )
                    ),
                ),
                key=form_widget_key("ui_commercial_app_spectral_feedback_grid_mode"),
                format_func=lambda value: {
                    "full_coherent_span": "完整相干频率范围（推荐，保留新边带）",
                    "seed_energy_support": "仅输入 Seed 能量支撑（更快，须检查泄漏）",
                }[value],
                help=(
                    "完整范围可接住 Kerr 新产生的边带；输入能量支撑只覆盖源频谱"
                    "的有效范围，分箱更集中但可能漏掉新边带，必须核对频谱泄漏诊断。"
                ),
            )
            commercial_app_cfg["spectral_feedback_bin_count"] = int(
                st.number_input(
                    "Commercial App SPM→增益回灌频率分箱",
                    min_value=3,
                    max_value=1025,
                    value=int(commercial_app_cfg.get("spectral_feedback_bin_count", 65)),
                    step=2,
                    key=form_widget_key("ui_commercial_app_spectral_feedback_bins"),
                    help=(
                        "分箱覆盖上面选定的频率范围。默认 65 仍远少于完整 x-y-t "
                        "coherent CPA；请对分箱数做收敛检查。"
                    ),
                )
            )
        carrier_cfg = (
            cfg["material"].get("strong_field", {}).get("free_carrier", {})
        )
        if (
            str(resolve_physics(cfg)["nonlinear_loss_model"])
            == "strong_field_effective_area"
            and str(carrier_cfg.get("model", "off")) == "rate_equation"
        ):
            st.markdown("##### 3. 跨脉冲载流子固定点")
            carrier_fixed_point_columns = st.columns(2)
            with carrier_fixed_point_columns[0]:
                commercial_app_cfg["carrier_periodic_relative_tolerance"] = sci(
                    "Commercial App 载流子周期定态相对容差",
                    commercial_app_cfg.get("carrier_periodic_relative_tolerance", 1.0e-6),
                    key=form_widget_key(
                        "ui_commercial_app_carrier_periodic_relative_tolerance"
                    ),
                    help_text=(
                        "控制每个局部 effective-area 载流子标量在相邻脉冲之间的"
                        "周期固定点收敛；它不同于 coherent CPA 的全局三维载流子收敛。"
                    ),
                )
            with carrier_fixed_point_columns[1]:
                commercial_app_cfg["carrier_periodic_maximum_iterations"] = int(
                    st.number_input(
                        "Commercial App 载流子周期定态最大迭代",
                        min_value=1,
                        max_value=256,
                        value=int(
                            commercial_app_cfg.get(
                                "carrier_periodic_maximum_iterations", 32
                            )
                        ),
                        key=form_widget_key(
                            "ui_commercial_app_carrier_periodic_maximum_iterations"
                        ),
                        help=(
                            "每次迭代都包含一个脉冲的生成/FCA 更新与脉冲间暗时间衰减；"
                            "达到上限仍不收敛时会按运行合同报错。"
                        ),
                    )
                )
        if commercial_app_is_cw:
            st.caption(
                "CW 不构造任意等效脉冲：Kerr 透镜直接使用平均强度，"
                "因此不报告 SPM 展宽、脉宽或压缩器诊断。"
            )
        elif commercial_app_cfg["temporal_model"] == "effective_area_spm":
            if (
                str(resolve_physics(cfg)["nonlinear_loss_model"])
                == "strong_field_effective_area"
            ):
                st.caption(
                    "每个 z 层仍计算三维局域增益/布居交换；横向传播保留 Gaussian q，"
                    "时间上另做一条一维 FFT。多光子损耗总量使用二维 Gaussian 的 K 阶"
                    "强度矩，载流子/FCA 是能量等效标量闭合；它不解析径向载流子输运或"
                    "不同半径处的光谱展宽，定量结论需由 coherent_cpa 复核。"
                )
            else:
                st.caption(
                    "每个 z 层仍计算三维局域增益/布居交换；横向传播保留 Gaussian q，"
                    "时间上另做一条一维 FFT，因而可显示 SPM 光谱展宽与压缩后波形，"
                    "但不含径向不同位置的展宽差异。"
                )
        else:
            if commercial_app_cfg["include_relaxation_during_pulse"]:
                st.caption(
                    "增益按有序时间箱耗尽，并在箱间推进能级暗弛豫；所有时间箱和横向"
                    "Gaussian-q 通道共享局部布居。可输出脉冲能量分箱、峰值和 FWHM，"
                    "但不保留频率间相干相位或延迟时间输运。"
                )
            else:
                st.caption(
                    "每个 z 层仍计算增益/布居交换；Kerr 与热效应只保留等效二次透镜，"
                    "当前时间闭合不会产生 SPM 光谱展宽。"
                )
    full_josa = _full_josa_selected(cfg)
    josa_backend = _josa_selected(cfg)
    gain_options = ["由泵浦和速率方程求解", "直接指定小信号增益"]
    gain_source = choice_radio(
        "增益来源", gain_options,
        index=1 if cfg["energy_levels"].get("gain_override_m_inv") is not None else 0,
        key=form_widget_key("ui_gain_source"),
        disabled=josa_backend,
        help=(
            "JOSA 必须更新真实上下能级布居，因此选择 JOSA 时固定使用 Pump + 速率方程；"
            "退出 JOSA 后，表单会恢复先前由用户设置的直接增益值。"
            if josa_backend
            else None
        ),
    )
    multipass_cfg = cfg["simulation"].setdefault(
        "multipass", copy.deepcopy(DEFAULTS["simulation"]["multipass"])
    )
    multipass_enabled = choice_checkbox(
        "启用多程放大",
        value=bool(multipass_cfg.get("enabled", False)),
        key=form_widget_key("ui_multipass_enabled"),
        help=(
            "关闭时固定为单程。开启后才显示多程模板与参数；推荐的物理模板会由"
            "真实传播时间和三维光斑位置自动产生重叠/竞争，不提供人为重叠开关。"
        ),
    )
    if not multipass_enabled:
        multipass_cfg["enabled"] = False
        cfg["simulation"]["pass_mode"] = "simple"
        cfg["simulation"]["passes"] = 1
    else:
        multipass_cfg["enabled"] = True
        cfg["simulation"]["pass_mode"] = "simple"
        cfg["simulation"]["between_pass_transmission"] = 1.0
        cfg["simulation"]["flip_x_between_passes"] = False
        cfg["optical_train"]["between_pass"] = []
        if str(multipass_cfg.get("execution_mode", "auto")) != "sequential":
            cfg["simulation"]["signal_model"] = "spectral_time"
            cfg["seed"].setdefault("temporal", {})["resolved"] = True
        cfg["simulation"]["passes"] = int(
            multipass_cfg.get("pass_count", 2)
        )
    if show_advanced:
        with st.container(border=True):
            st.markdown("#### 高级：物理模型与性能")
            _render_solver_override_controls(cfg, full_josa=full_josa)
    else:
        with st.expander(
            "高级：物理模型与性能",
            expanded=False,
        ):
            _render_solver_override_controls(cfg, full_josa=full_josa)
    summary_placeholder = None
    return gain_source, summary_placeholder


def _active_module_display(cfg: dict[str, Any]) -> dict[str, str]:
    """Return concise, user-facing names for the currently effective chain."""

    effective = resolve_physics(cfg)
    signal = resolve_signal_model(cfg)
    signal_labels = {
        "fluence": "单色能量通量",
        "spectral_time": "多色时间分箱",
        "josa_transport": "JOSA 时空输运",
        "commercial_app_fast": "Commercial App Gaussian-q 快速",
        "coherent_cpa": "相干全场 CPA",
    }
    if signal == "coherent_cpa" and str(cfg["seed"].get("coherent", {}).get("fidelity_mode", "")).startswith("axisymmetric_"):
        signal_labels["coherent_cpa"] = "轴对称相干 CPA"
    pump_labels = {
        "off": "规定泵浦分布",
        "prescribed_caustic": "规定泵浦焦散",
        "abcd": "Pump ABCD",
        "super_gaussian_abcd": "Pump 超高斯 ABCD",
        "fft": "Pump FFT",
        "hybrid_abcd_fft": "Pump 混合 ABCD/FFT",
        "stochastic_raytrace": "Pump 随机光线追迹",
    }
    population_labels = {
        "frozen": "跳过 Pump 预备与暗弛豫",
        "steady_state": "稳态布居",
        "transient": "瞬态布居",
        "periodic_steady_state": "逐脉冲周期固定点",
        "cycle_averaged_steady_state": "周期平均代数稳态",
        "transient_pulse_train": "有限脉冲列布居",
        "transient_waveform": "波形泵浦瞬态布居",
    }
    thermal_labels = {
        "off": "热场关闭",
        "steady_3d": "三维稳态热场",
        "transient_3d": "三维瞬态热场",
    }
    resolved_thermal = _resolved_thermal_label(cfg, effective)
    return {
        "signal": signal_labels.get(signal, signal),
        "pump": pump_labels.get(str(effective["pump_propagation"]), str(effective["pump_propagation"])),
        "population": population_labels.get(
            resolve_population_dynamics(cfg),
            resolve_population_dynamics(cfg),
        ),
        "thermal": thermal_labels.get(
            str(effective["thermal_model"]),
            thermal_labels.get(str(resolved_thermal), str(resolved_thermal)),
        ),
    }


def _render_active_module_card(cfg: dict[str, Any]) -> None:
    """Render the effective chain before any editing tools or long forms."""

    labels = _active_module_display(cfg)
    st.markdown(f"**{lt('当前计算链')}**")
    st.caption(
        f"{lt(labels['pump'])} → {lt(labels['population'])} → "
        f"{lt(labels['signal'])} → {lt(labels['thermal'])}"
    )
    if resolve_signal_model(cfg) == "commercial_app_fast":
        spatial = str(
            cfg.get("seed", {})
            .get("spatial_modes", {})
            .get("model", "effective_m2")
        )
        spatial_label = (
            "独立 HG 模态共享布居"
            if spatial == "incoherent_hg"
            else "单一等效 M² Gaussian-q"
        )
        temporal = str(
            cfg.get("seed", {})
            .get("commercial_app_fast", {})
            .get("temporal_model", "fixed_duration")
        )
        temporal_label = (
            "CW 平均强度"
            if str(cfg["seed"].get("temporal", {}).get("type", "gaussian")) == "cw"
            else "有序时间分箱与脉内能级弛豫"
            if cfg["seed"].get("commercial_app_fast", {}).get("include_relaxation_during_pulse", False)
            else "一维 effective-area SPM + 强场损耗/载流子"
            if (
                temporal == "effective_area_spm"
                and str(resolve_physics(cfg)["nonlinear_loss_model"])
                == "strong_field_effective_area"
            )
            else "一维 effective-area SPM"
            if temporal == "effective_area_spm"
            else "固定输入等效脉宽"
        )
        st.info(
            f"{lt('Commercial App 专属闭合')}：{lt(spatial_label)} · {lt(temporal_label)}"
        )


def _render_workspace_intro(title: str, description: str, ownership: str) -> None:
    """Give every form workspace a task, boundary, and ownership statement."""

    st.markdown(f"#### {lt(title)}")
    st.caption(lt(description))
    st.caption(f"{lt('本页负责')}: {lt(ownership)}")


def make_config() -> tuple[dict[str, Any], bool, list[str]]:
    if "form_base_config" not in st.session_state:
        st.session_state["form_base_config"] = initial_form_config()
    cfg = copy.deepcopy(st.session_state["form_base_config"])
    input_errors: list[str] = []
    with st.sidebar:
        st.title("仿真控制台")
        st.caption("先确认计算链，再按物理模块修改；未打开的高级参数仍保留在配置中。")
        active_model_placeholder = st.empty()
        run_placeholder = st.empty()
        run = False
        tool_columns = st.columns(2)
        with tool_columns[0].popover("配置文件", width="stretch"):
            uploaded = st.file_uploader(
                "上传配置 JSON（上传后自动回填表单）",
                type=["json"],
                key="config_upload",
            )
            if uploaded is not None:
                payload = uploaded.getvalue()
                digest = hashlib.sha256(payload).hexdigest()
                if digest != st.session_state.get("installed_config_digest"):
                    supplied = parse_json(payload.decode("utf-8"), "上传配置")
                    install_form_config(supplied, hard_reset_widgets=True)
                    st.session_state["installed_config_digest"] = digest
                    st.session_state["config_source_mode"] = "表单"
                    st.rerun()
            st.caption("支持只写部分字段；缺省字段会由程序默认值补齐。")
        with tool_columns[1].popover("材料库", width="stretch"):
            last_application = st.session_state.get("material_library_last_application")
            if last_application:
                st.success(
                    f"已回填材料：{last_application['material']}；"
                    f"{last_application['doping_percent']:g}% @ {last_application['temperature_K']:g} K。"
                )
                st.dataframe(last_application["rows"], hide_index=True, width="stretch")
            material_file = st.file_uploader(
                "外部材料列表",
                type=["xml", "json"],
                key="material_library_upload",
                help="支持 Commercial App 风格 Materialfile XML，以及 amplifier-sim-material-library/v1 JSON。文件只在当前会话解析。",
            )
            if material_file is not None:
                payload = material_file.getvalue()
                digest = hashlib.sha256(payload).hexdigest()
                if digest != st.session_state.get("material_library_digest"):
                    try:
                        records = load_material_library(payload, format_hint=Path(material_file.name).suffix.lstrip("."))
                    except Exception as exc:
                        st.error(f"材料库读取失败：{type(exc).__name__}: {exc}")
                    else:
                        st.session_state["material_library_records"] = records
                        st.session_state["material_library_digest"] = digest
                        st.session_state.pop(form_widget_key("ui_material_library_choice"), None)
            records = st.session_state.get("material_library_records", [])
            if records:
                names = [str(record["name"]) for record in records]
                current_name = str(cfg["material"].get("library", {}).get("material_name", cfg["material"].get("name", "")))
                selected_name = choice_selectbox(
                    f"材料（已读取 {len(records)} 项）",
                    names,
                    index=option_index(names, current_name),
                    key=form_widget_key("ui_material_library_choice"),
                    help=(
                        "每个选项是一条已解析并通过唯一性检查的材料记录。"
                        "这里只选择候选记录；只有点击下方“应用材料并回填全部表单”后，"
                        "才会按掺杂浓度、工作温度和偏振生成材料快照并覆盖相应材料字段，"
                        "光路、网格和求解器设置不会被材料记录覆盖。"
                    ),
                )
                record = records[names.index(selected_name)]
                prior_library = cfg["material"].get("library", {})
                doping_percent = st.number_input(
                    "实际掺杂浓度 (%)",
                    min_value=1e-9,
                    value=float(prior_library.get("doping_percent", 1.0)),
                    format="%.6g",
                    key=form_widget_key("ui_material_doping_percent"),
                    help="例如 0.6% 输入 0.6，而不是 0.006。离子密度和每 1% 吸收标定会自动缩放。",
                )
                reference_temperature = float(record.get("thermal", {}).get("reference_temperature_K", 293.15))
                operating_temperature = st.number_input(
                    "材料参数工作温度 (K)",
                    min_value=1.0,
                    value=float(prior_library.get("operating_temperature_K", reference_temperature)),
                    key=form_widget_key("ui_material_operating_temperature"),
                )
                polarization = choice_selectbox(
                    "Signal 光学偏振轴",
                    ["x", "y", "z"],
                    index=option_index(
                        ["x", "y", "z"],
                        str(
                            prior_library.get(
                                "optical_polarization_axis",
                                prior_library.get("polarization", "x"),
                            )
                        ),
                    ),
                    key=form_widget_key("ui_material_polarization"),
                    help="只选择 Seed/Signal 在各向异性光学数据中使用的折射率轴，不再决定热膨胀或弹性参数。",
                )
                pump_polarization = choice_selectbox(
                    "Pump 光学偏振轴",
                    ["x", "y", "z"],
                    index=option_index(
                        ["x", "y", "z"],
                        str(prior_library.get("pump_polarization_axis", "x")),
                    ),
                    key=form_widget_key("ui_material_pump_polarization"),
                    help="选择 Pump 吸收谱以及 Pump 折射率温漂所用的材料轴；它可以与 Signal 偏振不同。",
                )
                thermomechanical_axis = choice_selectbox(
                    "热机械标量代理晶轴",
                    ["x", "y", "z"],
                    index=option_index(
                        ["x", "y", "z"],
                        str(prior_library.get("thermomechanical_axis", "x")),
                    ),
                    key=form_widget_key("ui_material_thermomechanical_axis"),
                    help=(
                        "仅在标量应力代理中选择 Young 模量、热膨胀和 Poisson 比的晶轴。"
                        "完整各向异性 FEM 仍使用材料到实验坐标的 orientation matrix。"
                    ),
                )
                use_spectroscopy = choice_checkbox(
                    "应用吸收/发射/再吸收光谱",
                    value=bool(prior_library.get("use_spectroscopy", True)),
                    key=form_widget_key("ui_material_use_spectroscopy"),
                )
                use_sellmeier = choice_checkbox(
                    "应用 Sellmeier 折射率与材料色散",
                    value=bool(prior_library.get("use_sellmeier", True)),
                    key=form_widget_key("ui_material_use_sellmeier"),
                )
                use_heat = choice_checkbox(
                    "使用库中的默认产热比例",
                    value=False,
                    key=form_widget_key("ui_material_use_heat"),
                    help="关闭时保留现有量子亏损/手动热比例；建议先核对材料库对 heat efficiency 的定义。",
                )
                snapshot_range_policy = choice_selectbox(
                    "材料快照越界策略",
                    ["error", "edge"],
                    index=(
                        1
                        if str(
                            prior_library.get(
                                "snapshot_range_contract", {}
                            ).get("policy", "error")
                        )
                        == "edge"
                        else 0
                    ),
                    key=form_widget_key("ui_material_snapshot_range_policy"),
                    help=(
                        "error（推荐）会拒绝超出浓度、温度或波长标定范围的快照；"
                        "edge 仅用于显式保留旧数据的端点降阶外推，并会生成警告。"
                    ),
                )
                try:
                    calibrated, library_warnings = apply_material_record(
                        cfg,
                        record,
                        doping_percent=float(doping_percent),
                        temperature_K=float(operating_temperature),
                        polarization=polarization,
                        pump_polarization=pump_polarization,
                        thermomechanical_axis=thermomechanical_axis,
                        use_heat_fraction=use_heat,
                        use_sellmeier=use_sellmeier,
                        use_spectroscopy=use_spectroscopy,
                        snapshot_out_of_range=snapshot_range_policy,
                    )
                except Exception as exc:
                    calibrated = None
                    library_warnings = []
                    st.error(f"材料标定失败：{type(exc).__name__}: {exc}")
                if calibrated is not None:
                    preview_rows = material_application_rows(cfg, calibrated)
                    with st.expander("应用前后关键参数预览", expanded=True):
                        st.dataframe(preview_rows, hide_index=True, width="stretch")
                        st.info("这里显示的是待应用预览；只有点击下方蓝色按钮后，才会写入“③ 介质 → 晶体、材料与能级”的输入框。")
                        st.caption(
                            f"泵浦光谱点：{len(calibrated['energy_levels'].get('pump_cross_section_spectrum', {}).get('wavelength_m', []))}；"
                            f"信号光谱点：{len(calibrated['energy_levels'].get('signal_cross_section_spectrum', {}).get('wavelength_m', []))}。"
                        )
                if calibrated is not None and st.button(
                    "应用材料并回填全部表单", type="primary", width="stretch", key="apply_material_library"
                ):
                    st.session_state["config_source_mode"] = "表单"
                    install_form_config(calibrated)
                    st.session_state["material_library_last_warnings"] = library_warnings
                    st.session_state["material_library_last_application"] = {
                        "material": selected_name,
                        "doping_percent": float(doping_percent),
                        "temperature_K": float(operating_temperature),
                        "rows": preview_rows,
                    }
                    st.rerun()
            for warning in st.session_state.get("material_library_last_warnings", []):
                st.warning(warning)
        source_mode = choice_radio(
            "配置入口",
            ["表单", "完整 JSON"],
            horizontal=True,
            key="config_source_mode",
        )
        previous_source_mode = st.session_state.get("previous_config_source_mode")
        if source_mode == "完整 JSON" and previous_source_mode != "完整 JSON":
            st.session_state["config_json"] = json.dumps(cfg, ensure_ascii=False, indent=2)
        st.session_state["previous_config_source_mode"] = source_mode
        if source_mode == "完整 JSON":
            if "config_json" not in st.session_state:
                st.session_state["config_json"] = json.dumps(cfg, ensure_ascii=False, indent=2)
            text = st.text_area(
                "配置",
                height=560,
                key="config_json",
            )
            supplied = parse_json(text, "配置")
            merged = prepare_config(supplied)
            apply_column, run_column = st.columns(2)
            apply_to_form = apply_column.button("应用到表单", width="stretch", key="apply_json_to_form")
            run = run_column.button("运行 JSON", type="primary", width="stretch", key="run_json")
            if apply_to_form:
                install_form_config(supplied, update_json_editor=False, hard_reset_widgets=True)
                st.session_state["config_source_mode"] = "表单"
                st.rerun()
            return merged, run, input_errors

        sidebar_view = choice_radio(
            "界面级别",
            ["清爽设置", "完整设置"],
            horizontal=True,
            key="sidebar_view_mode",
            format_func=lambda value: {
                "清爽设置": "引导模式",
                "完整设置": "专家模式",
            }[str(value)],
            help=(
                "引导模式保留完整配置，但把后端覆盖和长参数区收起；专家模式展开逐项求解器覆盖。"
                "两种模式使用同一份物理配置，不会偷偷启用或关闭模块。"
            ),
        )
        show_advanced = sidebar_view == "完整设置"
        expand_long_sections = False
        if show_advanced:
            st.caption(
                "完整设置已启用：全部参数入口可用；长表单保持折叠，当前模块摘要始终可见。"
            )
        operating_tab, beams_tab, physics_tab, numerical_tab, recording_tab = st.tabs(
            ["① 模型", "② 光束", "③ 介质", "④ 数值", "⑤ 结果记录"]
        )

        with operating_tab:
            _render_workspace_intro(
                "模型与运行",
                "先选择需要保留的物理自由度，再决定多程、稳态和热光闭环。",
                "求解家族、增益来源、周期策略与模块联动",
            )
        with operating_tab, st.expander("运行方式与求解", expanded=True):
            gain_source, solver_summary_placeholder = _render_model_solver_controls(
                cfg,
                show_advanced=show_advanced,
            )

        with physics_tab:
            _render_workspace_intro(
                "增益介质与热结构",
                "把晶体、光谱能级、产热、冷却和应力放在同一条介质链中。",
                "材料参数、能级储能、热边界与力学反馈",
            )
        with physics_tab, st.expander("晶体、材料与能级", expanded=False):
            cfg["grid"]["length_z_m"] = sci("晶体长度 (m)", cfg["grid"]["length_z_m"], key=form_widget_key("ui_crystal_length"))
            shape_options = ["cuboid — 六面体/长方体", "cylinder — 圆柱体"]
            crystal_shape = choice_selectbox(
                "晶体形状", shape_options,
                index=option_index(shape_options, str(cfg["material"]["geometry"].get("shape", "cuboid"))),
                key=form_widget_key("ui_crystal_shape"),
            )
            cfg["material"]["geometry"]["shape"] = crystal_shape.split(" ")[0]
            if cfg["material"]["geometry"]["shape"] == "cuboid":
                width_x_m = cfg["material"]["geometry"].get("width_x_m")
                if width_x_m is None:
                    width_x_m = cfg["grid"]["size_x_m"]
                height_y_m = cfg["material"]["geometry"].get("height_y_m")
                if height_y_m is None:
                    height_y_m = cfg["grid"]["size_y_m"]
                cfg["material"]["geometry"]["width_x_m"] = sci(
                    "晶体宽度 x (m)",
                    width_x_m,
                    key=form_widget_key("ui_crystal_width_x"),
                )
                cfg["material"]["geometry"]["height_y_m"] = sci(
                    "晶体高度 y (m)",
                    height_y_m,
                    key=form_widget_key("ui_crystal_height_y"),
                )
                cfg["material"]["geometry"]["diameter_m"] = None
            else:
                diameter_m = cfg["material"]["geometry"].get("diameter_m")
                if diameter_m is None:
                    diameter_m = min(
                        cfg["grid"]["size_x_m"], cfg["grid"]["size_y_m"]
                    )
                cfg["material"]["geometry"]["diameter_m"] = sci(
                    "晶体直径 (m)",
                    diameter_m,
                    key=form_widget_key("ui_crystal_diameter"),
                )
                cfg["material"]["geometry"]["width_x_m"] = None
                cfg["material"]["geometry"]["height_y_m"] = None
            cfg["material"]["geometry"]["clip_optical_field"] = choice_checkbox(
                "晶体外区域作为硬孔径", bool(cfg["material"]["geometry"].get("clip_optical_field", True)), key=form_widget_key("ui_clip_crystal")
            )
            cfg["material"]["name"] = st.text_input(
                "材料名称", key=form_widget_key("ui_material_name"),
                **widget_default("ui_material_name", str(cfg["material"].get("name", "user_material"))),
            )
            if cfg["material"].get("library"):
                library_info = cfg["material"]["library"]
                st.caption(
                    f"材料库标定：{float(library_info.get('doping_percent', 0.0)):g}% 掺杂，"
                    f"{float(library_info.get('operating_temperature_K', 293.15)):g} K，"
                    f"Signal={library_info.get('optical_polarization_axis', library_info.get('polarization', 'x'))} 轴，"
                    f"Pump={library_info.get('pump_polarization_axis', 'x')} 轴，"
                    f"热机械={library_info.get('thermomechanical_axis', 'x')} 轴。"
                    "修改这些条件请在“导入材料库”中重新应用。"
                )
                if library_info.get("record_sha256"):
                    st.caption(
                        "材料记录 SHA-256："
                        f"{library_info['record_sha256']}（摘要同时保存应用后的关键参数快照）"
                    )
                raw_fluorescence = float(
                    library_info.get("effective_fluorescence_wavelength_raw_m", 0.0)
                    or 0.0
                )
                if raw_fluorescence > 0.0:
                    fluorescence_enabled = choice_checkbox(
                        f"产热量子亏损使用有效荧光波长 {raw_fluorescence * 1e9:.6g} nm",
                        key=form_widget_key("ui_effective_fluorescence_enabled"),
                        help=(
                            "只有材料记录明确启用或你在这里勾选后才参与产热；"
                            "不勾选时沿用以 Seed 中心波长作为参考的兼容近似。"
                        ),
                        **widget_default(
                            "ui_effective_fluorescence_enabled",
                            bool(
                                library_info.get(
                                    "effective_fluorescence_wavelength_enabled", False
                                )
                            ),
                        ),
                    )
                    library_info["effective_fluorescence_wavelength_enabled"] = fluorescence_enabled
                    library_info["effective_fluorescence_wavelength_m"] = (
                        raw_fluorescence if fluorescence_enabled else None
                    )
                    fluorescence_warning = (
                        "The material record contains an effective fluorescence wavelength but no explicit enabled flag; "
                        "it was retained as metadata and disabled for heating until explicitly enabled."
                    )
                    library_warnings = [
                        warning
                        for warning in library_info.get("warnings", [])
                        if "disabled for heating until explicitly enabled" not in warning
                    ]
                    if not fluorescence_enabled:
                        library_warnings.append(fluorescence_warning)
                    library_info["warnings"] = library_warnings
                if library_info.get("rate_system"):
                    rate_system = library_info["rate_system"]
                    if not rate_system.get("usable", True):
                        st.error(
                            "材料库 rate-line 元数据已隔离，不能作为当前材料能级来源："
                            + str(rate_system.get("reason", "物种/能级来源未通过验证"))
                        )
                    else:
                        st.warning(
                            "材料库的专有 rate-line 拓扑只作为可追溯元数据保存；"
                            "当前运行使用下方内置级联或 custom 能级网络，不会静默激活数据库行号。"
                        )
            cfg["material"]["ion_density_m3"] = sci("掺杂离子密度 (m⁻³)", cfg["material"]["ion_density_m3"], key=form_widget_key("ui_density"))
            render_doping_profile_controls(cfg)
            cfg["material"]["refractive_index"] = st.number_input(
                "晶体折射率", min_value=1.0, step=0.01, key=form_widget_key("ui_refractive_index"),
                **widget_default("ui_refractive_index", float(cfg["material"]["refractive_index"])),
            )
            if cfg["material"].get("pump_refractive_index") is not None:
                cfg["material"]["pump_refractive_index"] = st.number_input(
                    "泵浦波长折射率",
                    min_value=1.0,
                    step=0.01,
                    key=form_widget_key("ui_pump_refractive_index"),
                    **widget_default("ui_pump_refractive_index", float(cfg["material"]["pump_refractive_index"])),
                )
            cfg["material"]["pump_passive_absorption_m_inv"] = sci(
                "泵浦被动吸收 (m⁻¹)", cfg["material"]["pump_passive_absorption_m_inv"], key=form_widget_key("ui_pump_passive")
            )
            cfg["material"]["signal_passive_loss_m_inv"] = sci(
                "信号被动损耗 (m⁻¹)", cfg["material"]["signal_passive_loss_m_inv"], key=form_widget_key("ui_signal_passive")
            )
            level_options = [
                "four_level — 内置四能级级联（4 个布居）",
                "three_level — 内置三能级级联（3 个布居）",
                "two_level — 内置二能级（2 个布居）",
                "custom — 自定义 N 能级/跃迁",
            ]
            level_label = choice_selectbox(
                "能级模型", level_options,
                index=option_index(level_options, str(cfg["energy_levels"].get("model", "four_level"))),
                key=form_widget_key("ui_level_model"),
            )
            cfg["energy_levels"]["model"] = level_label.split(" ")[0]
            st.caption(
                "内置模型实时求解每个列出的布居与固定级联寿命，并非把四能级折成二能级；"
                "分支跃迁、上转换/交叉弛豫和任意网络请用 custom。"
            )
            cfg["energy_levels"]["pump_absorption_cross_section_m2"] = sci("泵浦吸收截面 σpa (m²)", cfg["energy_levels"]["pump_absorption_cross_section_m2"], key=form_widget_key("ui_sigma_pa"))
            cfg["energy_levels"]["pump_emission_cross_section_m2"] = sci("泵浦发射截面 σpe (m²)", cfg["energy_levels"]["pump_emission_cross_section_m2"], key=form_widget_key("ui_sigma_pe"))
            cfg["energy_levels"]["pump_quantum_efficiency"] = st.number_input(
                "泵浦激发量子效率",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key=form_widget_key("ui_pump_quantum_efficiency"),
                help="乘到泵浦激发布居速率；光学吸收仍由 Nσ 决定。库中大于 1 的有效量子产率不会自动套入单离子模型。",
                **widget_default("ui_pump_quantum_efficiency", float(cfg["energy_levels"].get("pump_quantum_efficiency", 1.0))),
            )
            cfg["energy_levels"]["signal_emission_cross_section_m2"] = sci("信号发射截面 σe (m²)", cfg["energy_levels"]["signal_emission_cross_section_m2"], key=form_widget_key("ui_sigma_se"))
            cfg["energy_levels"]["signal_absorption_cross_section_m2"] = sci("信号吸收截面 σa (m²)", cfg["energy_levels"]["signal_absorption_cross_section_m2"], key=form_widget_key("ui_sigma_sa"))
            cfg["energy_levels"]["upper_state_lifetime_s"] = sci("上能级寿命 (s)", cfg["energy_levels"]["upper_state_lifetime_s"], key=form_widget_key("ui_tau_upper"))
            cfg["energy_levels"]["pump_level_lifetime_s"] = sci("泵浦能级寿命 (s)", cfg["energy_levels"]["pump_level_lifetime_s"], key=form_widget_key("ui_tau_pump"))
            cfg["energy_levels"]["lower_level_lifetime_s"] = sci("下能级寿命 (s)", cfg["energy_levels"]["lower_level_lifetime_s"], key=form_widget_key("ui_tau_lower"))
            if gain_source.startswith("直接"):
                configured_gain_override = cfg["energy_levels"].get("gain_override_m_inv")
                cfg["energy_levels"]["gain_override_m_inv"] = sci(
                    "小信号增益系数 g₀ (m⁻¹)",
                    300.0 if configured_gain_override is None else configured_gain_override,
                    key=form_widget_key("ui_gain_override"),
                )
                cfg["energy_levels"]["saturation_fluence_J_m2"] = sci(
                    "饱和通量 (J/m²)", cfg["energy_levels"].get("saturation_fluence_J_m2") or 1.0, key=form_widget_key("ui_sat_fluence")
                )
            else:
                cfg["energy_levels"]["gain_override_m_inv"] = None
            if cfg["energy_levels"]["model"] == "custom":
                custom_seed = {
                    key: copy.deepcopy(cfg["energy_levels"].get(key))
                    for key in (
                        "levels",
                        "pump_transition",
                        "signal_transition",
                        "decays",
                        "nonlinear_processes",
                        "nonlinear_solver",
                    )
                }
                custom = st.text_area(
                    "自定义能级附加 JSON",
                    value=json.dumps(custom_seed, ensure_ascii=False, indent=2),
                    height=230,
                    key=form_widget_key("ui_custom_levels_json"),
                )
                st.caption(
                    "衰减通道三选一：partial_rate_s、partial_lifetime_s，或共享"
                    " total_lifetime_s + branching_fraction；同一源能级的分支比例和"
                    "不得超过 1，剩余比例会明确记为未解析衰减浴。"
                )
                cfg["energy_levels"].update(parse_json(custom, "自定义能级"))
                if cfg["energy_levels"].get("nonlinear_processes"):
                    nonlinear_solver = cfg["energy_levels"].setdefault(
                        "nonlinear_solver",
                        copy.deepcopy(DEFAULTS["energy_levels"]["nonlinear_solver"]),
                    )
                    st.caption("上转换／交叉弛豫按布居耗尽自动细分时间步；变化小时只需一步。")
                    nonlinear_columns = st.columns(2)
                    with nonlinear_columns[0]:
                        nonlinear_solver[
                            "maximum_fractional_depletion_per_substep"
                        ] = st.number_input(
                            "单子步最大目标耗尽比例",
                            min_value=1e-4,
                            max_value=1.0,
                            value=float(
                                nonlinear_solver.get(
                                    "maximum_fractional_depletion_per_substep", 0.05
                                )
                            ),
                            format="%.6g",
                            key=form_widget_key("ui_nonlinear_depletion_target"),
                        )
                    with nonlinear_columns[1]:
                        nonlinear_solver["maximum_substeps"] = int(
                            st.number_input(
                                "每个时间步最多子步",
                                min_value=1,
                                max_value=4096,
                                value=int(nonlinear_solver.get("maximum_substeps", 256)),
                                key=form_widget_key("ui_nonlinear_max_substeps"),
                            )
                        )
            pump_curve = st.file_uploader("泵浦截面响应 CSV：λ(nm), σabs, σem", type=["csv", "txt"], key=form_widget_key("pump_xs"))
            signal_curve = st.file_uploader("信号截面响应 CSV：λ(nm), σabs, σem", type=["csv", "txt"], key=form_widget_key("signal_xs"))
            if pump_curve is not None:
                cfg["energy_levels"]["pump_cross_section_spectrum"] = {"samples": read_rows(pump_curve, 3), "wavelength_unit": "nm"}
            if signal_curve is not None:
                cfg["energy_levels"]["signal_cross_section_spectrum"] = {"samples": read_rows(signal_curve, 3), "wavelength_unit": "nm"}
            for role, title in (("pump", "Pump"), ("signal", "Seed")):
                for channel, symbol in (("absorption", "σabs"), ("emission", "σem")):
                    parameter = SweepParameter(title, f"energy_levels.{role}_{channel}_cross_section_m2", "m²")
                    reason = sweep_parameter_inactive_reason(cfg, parameter)
                    if reason is not None:
                        st.info(f"{title} {symbol} 当前由光谱表或自定义跃迁提供；上方标量只作回退，修改它不会改变当前截面。实际来源：{reason}")

        with beams_tab:
            _render_workspace_intro(
                "光束、脉冲与光路",
                "Pump 决定储能来源，Seed 决定被放大的时空场；两者的传播路径彼此独立。",
                "Pump/Seed 源、脉冲定义、空间模式、偏振与多程平台",
            )
            st.caption(
                "Pump 与 Seed 分开定义；"
                f"Pump {float(cfg['pump']['power_W']):.5g} W @ {float(cfg['pump']['wavelength_m']) * 1e9:.5g} nm · "
                f"Seed {float(cfg['seed']['pulse_energy_J']) * 1e9:.5g} nJ @ {float(cfg['seed']['wavelength_m']) * 1e9:.5g} nm。"
            )
        with beams_tab, st.expander(
            "Pump · 功率、空间与光谱",
            expanded=expand_long_sections,
        ):
            st.markdown("#### A. 功率与主动吸收")
            cfg["pump"]["power_W"] = sci("泵浦功率 (W)", cfg["pump"]["power_W"], key=form_widget_key("ui_pump_power"), help_text="由泵浦求增益时，这是入射泵浦功率。")
            cfg["pump"]["wavelength_m"] = sci("中心波长 (m)", cfg["pump"]["wavelength_m"], key=form_widget_key("ui_pump_wavelength"))
            absorption_transport = cfg["pump"].setdefault(
                "absorption_transport",
                {
                    "model": "population_dependent",
                    "fixed_active_coefficient_m_inv": None,
                },
            )
            absorption_models = [
                "population_dependent",
                "fixed_active_coefficient",
            ]
            absorption_transport["model"] = choice_selectbox(
                "Pump 主动吸收输运",
                absorption_models,
                index=option_index(
                    absorption_models,
                    str(absorption_transport.get("model", "population_dependent")),
                ),
                key=form_widget_key("ui_pump_absorption_transport"),
                format_func=lambda value: {
                    "population_dependent": "population_dependent — 由局部能级布居与截面求净吸收",
                    "fixed_active_coefficient": "fixed_active_coefficient — 固定主动吸收系数（Commercial App 非饱和 Pump）",
                }[value],
                help=(
                    "population_dependent 会让 Pump 耗尽随局部基态/激发态布居变化；"
                    "fixed_active_coefficient 只用给定标量推进 Pump 衰减，适合复现关闭 Pump saturation 的 Commercial App 算例。"
                ),
            )
            if absorption_transport["model"] == "fixed_active_coefficient":
                fixed_absorption = absorption_transport.get(
                    "fixed_active_coefficient_m_inv"
                )
                if fixed_absorption is None:
                    fixed_absorption = 0.0
                absorption_transport["fixed_active_coefficient_m_inv"] = sci(
                    "固定 Pump 主动吸收系数 (m⁻¹)",
                    fixed_absorption,
                    key=form_widget_key("ui_pump_fixed_absorption_coefficient"),
                    help_text=(
                        "它决定 Pump 传播中的主动耗尽和吸收功率账本；能级激发仍按 Pump 光子流、"
                        "跃迁截面与量子效率求解，不会把固定系数冒充受激跃迁截面。"
                    ),
                )
            st.divider()
            st.markdown("#### B. 空间光源与传播")
            source_definition = cfg["pump"].setdefault("source_definition", {})
            source_modes = ["waist_m2", "waist_divergence", "emitter_na"]
            source_definition["mode"] = choice_selectbox(
                "Pump 光源定义",
                source_modes,
                index=option_index(source_modes, str(source_definition.get("mode", "waist_m2"))),
                key=form_widget_key("ui_pump_source_mode"),
                format_func=lambda value: {
                    "waist_m2": "直接束腰 + M²（常用 Gaussian 输入）",
                    "waist_divergence": "二维超高斯：束腰半径 + 发散角",
                    "emitter_na": "非相干发光面：半径 + NA",
                }[value],
                help=(
                    "这里仅定义光源相空间；传播算法在“求解器”中独立选择。emitter + "
                    "stochastic_raytrace 会直接采样发光面和 NA；super_gaussian_abcd "
                    "用等二阶矩 M² 传播束宽但保留 SG 形状；其他场模式使用 M² 代理。"
                    "x/y 完整指数相等时采用椭圆径向 SG，不等时采用明确的 x/y 可分离 SG。"
                ),
            )
            source_mode = str(source_definition["mode"])
            if source_mode == "waist_m2":
                profile_options = ["super_gaussian", "radial_super_gaussian", "gaussian", "top_hat", "annular"]
                cfg["pump"]["profile"] = choice_selectbox(
                    "空间光斑", profile_options, index=option_index(profile_options, str(cfg["pump"]["profile"])), key=form_widget_key("ui_pump_profile")
                )
                cfg["pump"]["waist_x_m"] = sci("1/e² 半径 wx (m)", cfg["pump"]["waist_x_m"], key=form_widget_key("ui_pump_wx"))
                cfg["pump"]["waist_y_m"] = sci("1/e² 半径 wy (m)", cfg["pump"]["waist_y_m"], key=form_widget_key("ui_pump_wy"))
                cfg["pump"]["m2_x"] = st.number_input("M² x", min_value=1.0, value=float(cfg["pump"]["m2_x"]), step=0.1, key=form_widget_key("ui_pump_m2x"))
                cfg["pump"]["m2_y"] = st.number_input("M² y", min_value=1.0, value=float(cfg["pump"]["m2_y"]), step=0.1, key=form_widget_key("ui_pump_m2y"))
                cfg["pump"]["order_x"] = st.number_input("内部超高斯 order x（高斯=1）", min_value=0.05, value=float(cfg["pump"]["order_x"]), key=form_widget_key("ui_pump_orderx"))
                cfg["pump"]["order_y"] = st.number_input("内部超高斯 order y（高斯=1）", min_value=0.05, value=float(cfg["pump"]["order_y"]), key=form_widget_key("ui_pump_ordery"))
            elif source_mode == "waist_divergence":
                source_definition["waist_radius_x_m"] = sci(
                    "焦点 1/e² 半径 x (m)",
                    source_definition.get("waist_radius_x_m", cfg["pump"]["waist_x_m"]),
                    key=form_widget_key("ui_pump_waist_divergence_x"),
                    help_text="输入 1/e² 半径；界面下方同时显示全直径。",
                )
                source_definition["waist_radius_y_m"] = sci(
                    "焦点 1/e² 半径 y (m)",
                    source_definition.get("waist_radius_y_m", cfg["pump"]["waist_y_m"]),
                    key=form_widget_key("ui_pump_waist_divergence_y"),
                )
                source_definition["divergence_half_angle_x_deg"] = st.number_input(
                    "最大发散半角 x (°)", min_value=0.0, max_value=89.0,
                    value=float(source_definition.get("divergence_half_angle_x_deg", 0.0)),
                    format="%.8g", key=form_widget_key("ui_pump_divergence_x"),
                )
                source_definition["divergence_half_angle_y_deg"] = st.number_input(
                    "最大发散半角 y (°)", min_value=0.0, max_value=89.0,
                    value=float(source_definition.get("divergence_half_angle_y_deg", 0.0)),
                    format="%.8g", key=form_widget_key("ui_pump_divergence_y"),
                )
            else:
                source_definition["emitting_radius_x_m"] = sci(
                    "diode 发光面半径 x (m)",
                    source_definition.get("emitting_radius_x_m", cfg["pump"]["waist_x_m"]),
                    key=form_widget_key("ui_pump_emitter_radius_x"),
                    help_text="输入发光面的 1/e²/轮廓尺度半径；程序同时报告全宽 2r。",
                )
                source_definition["emitting_radius_y_m"] = sci(
                    "diode 发光面半径 y (m)",
                    source_definition.get("emitting_radius_y_m", cfg["pump"]["waist_y_m"]),
                    key=form_widget_key("ui_pump_emitter_radius_y"),
                )
                source_definition["numerical_aperture_x"] = st.number_input(
                    "NA x", min_value=0.0, value=float(source_definition.get("numerical_aperture_x", 0.0)),
                    format="%.8g", key=form_widget_key("ui_pump_emitter_na_x"),
                )
                source_definition["numerical_aperture_y"] = st.number_input(
                    "NA y", min_value=0.0, value=float(source_definition.get("numerical_aperture_y", 0.0)),
                    format="%.8g", key=form_widget_key("ui_pump_emitter_na_y"),
                )
                source_definition["free_space_refractive_index"] = st.number_input(
                    "光源侧介质折射率", min_value=1.0,
                    value=float(source_definition.get("free_space_refractive_index", 1.0)),
                    format="%.8g", key=form_widget_key("ui_pump_emitter_index"),
                )
            if source_mode != "waist_m2":
                source_definition["super_gaussian_exponent_x"] = st.number_input(
                    "径向超高斯完整指数 x（高斯=2）", min_value=0.1,
                    value=float(source_definition.get("super_gaussian_exponent_x", 2.0)),
                    key=form_widget_key("ui_pump_radial_sg_x"),
                )
                source_definition["super_gaussian_exponent_y"] = st.number_input(
                    "径向超高斯完整指数 y（高斯=2）", min_value=0.1,
                    value=float(source_definition.get("super_gaussian_exponent_y", 2.0)),
                    key=form_widget_key("ui_pump_radial_sg_y"),
                )
            cfg["pump"]["geometric_efficiency"] = st.number_input(
                "几何泵浦效率", min_value=0.0, max_value=1.0,
                value=float(cfg["pump"].get("geometric_efficiency", 1.0)), step=0.001,
                key=form_widget_key("ui_pump_geometric_efficiency"),
                help=(
                    "这是光源进入已建模光路之前的实测/未建模损耗。随机光线会另外计算"
                    "已建模透镜、界面、光阑和晶体侧漏，二者不会相互覆盖。默认 1。"
                ),
            )
            effective_pump_transport = resolve_physics(cfg)["pump_propagation"]
            if effective_pump_transport == "stochastic_raytrace":
                st.markdown("#### 随机光线求解")
                st.caption(
                    "固定随机种子会让热光/布居迭代重复使用同一束光线，避免 Monte Carlo "
                    "噪声被误判为不收敛。射线数控制统计误差，晶体网格控制沉积空间分辨率。"
                )
                st.info(
                    "右侧未运行的实时光路仍显示确定性的 D4σ/M² 包络，便于连续拖动元件；"
                    "它不是随机光线结果。运行后保存的 Pump 三维场、入口/出口面和功率账本"
                    "才来自 stochastic_raytrace。"
                )
                raytrace_cfg = cfg["pump"].setdefault("raytrace", {})
                raytrace_columns = st.columns(2)
                with raytrace_columns[0]:
                    raytrace_cfg["ray_count"] = int(st.number_input(
                        "光线数",
                        min_value=100,
                        max_value=5000000,
                        value=int(raytrace_cfg.get("ray_count", 100000)),
                        step=1000,
                        key=form_widget_key("ui_pump_ray_count"),
                    ))
                    raytrace_cfg["random_seed"] = int(st.number_input(
                        "随机种子",
                        min_value=0,
                        max_value=2147483647,
                        value=int(raytrace_cfg.get("random_seed", 12345)),
                        key=form_widget_key("ui_pump_ray_seed"),
                    ))
                    angular_options = [
                        "auto",
                        "uniform_na",
                        "uniform_solid_angle",
                        "generalized_gaussian",
                    ]
                    raytrace_cfg["angular_distribution"] = choice_selectbox(
                        "角分布",
                        angular_options,
                        index=option_index(
                            angular_options,
                            str(raytrace_cfg.get("angular_distribution", "auto")),
                        ),
                        key=form_widget_key("ui_pump_ray_angular_distribution"),
                        format_func=lambda value: {
                            "auto": "auto — diode 用恒定辐亮度 NA；束腰输入用广义 Gaussian",
                            "uniform_na": "uniform_na — 方向余弦椭圆均匀（平面恒定辐亮度）",
                            "uniform_solid_angle": "uniform_solid_angle — NA 支撑内均匀立体角",
                            "generalized_gaussian": "generalized_gaussian — 广义 Gaussian 角分布",
                        }[value],
                    )
                    if raytrace_cfg["angular_distribution"] == "generalized_gaussian":
                        angular_exponent_columns = st.columns(2)
                        with angular_exponent_columns[0]:
                            raytrace_cfg[
                                "angular_super_gaussian_exponent_x"
                            ] = st.number_input(
                                "角分布超高斯完整指数 x（高斯=2）",
                                min_value=0.1,
                                value=float(
                                    raytrace_cfg.get(
                                        "angular_super_gaussian_exponent_x"
                                    )
                                    or source_definition.get(
                                        "super_gaussian_exponent_x", 2.0
                                    )
                                ),
                                key=form_widget_key(
                                    "ui_pump_ray_angular_exponent_x"
                                ),
                            )
                        with angular_exponent_columns[1]:
                            raytrace_cfg[
                                "angular_super_gaussian_exponent_y"
                            ] = st.number_input(
                                "角分布超高斯完整指数 y（高斯=2）",
                                min_value=0.1,
                                value=float(
                                    raytrace_cfg.get(
                                        "angular_super_gaussian_exponent_y"
                                    )
                                    or source_definition.get(
                                        "super_gaussian_exponent_y", 2.0
                                    )
                                ),
                                key=form_widget_key(
                                    "ui_pump_ray_angular_exponent_y"
                                ),
                            )
                    deposition_options = [
                        "nearest_cell_track_length",
                        "bilinear_track_length",
                    ]
                    raytrace_cfg["deposition_model"] = choice_selectbox(
                        "射线到晶体网格的沉积",
                        deposition_options,
                        index=option_index(
                            deposition_options,
                            str(
                                raytrace_cfg.get(
                                    "deposition_model",
                                    "nearest_cell_track_length",
                                )
                            ),
                        ),
                        key=form_widget_key("ui_pump_ray_deposition_model"),
                        format_func=lambda value: {
                            "nearest_cell_track_length": "nearest — 最近单元路径长度（旧结果兼容）",
                            "bilinear_track_length": "bilinear — 双线性守恒路径沉积（低噪声对照）",
                        }[value],
                        help=(
                            "两者都严格关闭功率账本。bilinear 同时插值局部吸收和热折射率梯度，"
                            "并把每段能量守恒地分配到相邻有效晶体单元。"
                        ),
                    )
                with raytrace_columns[1]:
                    raytrace_cfg["maximum_transverse_step_fraction"] = st.number_input(
                        "每子步最大横向跨格比例",
                        min_value=0.05,
                        max_value=4.0,
                        value=float(raytrace_cfg.get("maximum_transverse_step_fraction", 0.5)),
                        step=0.05,
                        key=form_widget_key("ui_pump_ray_step_fraction"),
                        help="越小越能解析高 NA 斜射线穿越网格，但运行更慢。",
                    )
                    raytrace_cfg["maximum_substeps_per_z_cell"] = int(st.number_input(
                        "每个 z 单元最多子步",
                        min_value=1,
                        max_value=256,
                        value=int(raytrace_cfg.get("maximum_substeps_per_z_cell", 16)),
                        key=form_widget_key("ui_pump_ray_max_substeps"),
                    ))
                    raytrace_cfg["minimum_relative_ray_power"] = sci(
                        "停止追踪的相对光线功率",
                        raytrace_cfg.get("minimum_relative_ray_power", 1e-8),
                        key=form_widget_key("ui_pump_ray_min_power"),
                    )
            pump_source_summary = canonicalize_pump_source(cfg["pump"])
            if source_mode != "waist_m2":
                transport_representation = (
                    "随机相空间光线（直接使用源面尺寸与角分布）"
                    if effective_pump_transport == "stochastic_raytrace"
                    else (
                        "自相似超高斯强度包络（SG 固定，束宽按 ABCD 传播）"
                        if effective_pump_transport == "super_gaussian_abcd"
                        else "二阶矩 M² 场代理"
                    )
                )
                st.dataframe(
                    [
                        {
                            "量": "全宽 x / y",
                            "数值": f"{pump_source_summary['full_width_x_m'] * 1e3:.6g} / {pump_source_summary['full_width_y_m'] * 1e3:.6g} mm",
                        },
                        {
                            "量": "最大发散半角 x / y",
                            "数值": f"{pump_source_summary['maximum_divergence_half_angle_x_deg']:.6g} / {pump_source_summary['maximum_divergence_half_angle_y_deg']:.6g} °",
                        },
                        {
                            "量": "推导的有效 M² x / y",
                            "数值": f"{pump_source_summary['effective_m2_x']:.8g} / {pump_source_summary['effective_m2_y']:.8g}",
                        },
                        {
                            "量": "当前传播表示",
                            "数值": transport_representation,
                        },
                    ],
                    hide_index=True,
                    width="stretch",
                )
            pump_focus_is_crystal_relative = effective_pump_transport == "prescribed_caustic"
            pump_focus_label = (
                "焦点相对晶体入口 z (m)"
                if pump_focus_is_crystal_relative
                else "束腰相对 PUMP LASER 源面 z (m)"
            )
            pump_focus_is_raytrace_emitter = bool(
                source_mode == "emitter_na"
                and effective_pump_transport == "stochastic_raytrace"
            )
            if pump_focus_is_raytrace_emitter:
                st.caption(
                    "当前 emitter_na + stochastic_raytrace 直接在 z=0 发光面采样位置和 NA；"
                    "focus_z_m / focus_x_m / focus_y_m 不参与光线生成。导入值会保留，"
                    "但不显示成一个实际生效的自由度。"
                )
            else:
                separate_pump_focus = choice_checkbox(
                    "Pump x / y 使用不同焦点位置（像散光束）",
                    value=(
                        cfg["pump"].get("focus_x_m") is not None
                        or cfg["pump"].get("focus_y_m") is not None
                    ),
                    key=form_widget_key("ui_pump_separate_focus"),
                )
                if separate_pump_focus:
                    pump_focus_columns = st.columns(2)
                    with pump_focus_columns[0]:
                        cfg["pump"]["focus_x_m"] = sci(
                            f"x {pump_focus_label}",
                            cfg["pump"].get("focus_x_m")
                            if cfg["pump"].get("focus_x_m") is not None
                            else cfg["pump"]["focus_z_m"],
                            key=form_widget_key("ui_pump_focus_x"),
                        )
                    with pump_focus_columns[1]:
                        cfg["pump"]["focus_y_m"] = sci(
                            f"y {pump_focus_label}",
                            cfg["pump"].get("focus_y_m")
                            if cfg["pump"].get("focus_y_m") is not None
                            else cfg["pump"]["focus_z_m"],
                            key=form_widget_key("ui_pump_focus_y"),
                        )
                    cfg["pump"]["focus_z_m"] = 0.5 * (
                        float(cfg["pump"]["focus_x_m"])
                        + float(cfg["pump"]["focus_y_m"])
                    )
                else:
                    cfg["pump"]["focus_z_m"] = sci(
                        pump_focus_label,
                        cfg["pump"]["focus_z_m"],
                        key=form_widget_key("ui_pump_focus"),
                        help_text=(
                            "prescribed_caustic 直接规定晶体内焦散，因此从晶体入口计；"
                            "SG-ABCD、普通 ABCD 和 FFT 从 PUMP LASER 源面传播，"
                            "因此从源面计。"
                        ),
                    )
                    cfg["pump"]["focus_x_m"] = None
                    cfg["pump"]["focus_y_m"] = None
            pump_pose_columns = st.columns(3)
            pump_center_semantics = (
                "发光面中心"
                if source_mode == "emitter_na"
                and effective_pump_transport == "stochastic_raytrace"
                else "腰斑/焦散中心"
            )
            with pump_pose_columns[0]:
                cfg["pump"]["center_x_m"] = sci(
                    f"Pump {pump_center_semantics} x (m)",
                    cfg["pump"].get("center_x_m", 0.0),
                    key=form_widget_key("ui_pump_center_x"),
                    help_text=(
                        "随机光线的 emitter_na 模式把它解释为 z=0 发光面的中心；"
                        "束腰/发散源和场传播代理把它解释为焦散或等效束腰处的光轴中心。"
                    ),
                )
            with pump_pose_columns[1]:
                cfg["pump"]["center_y_m"] = sci(
                    f"Pump {pump_center_semantics} y (m)",
                    cfg["pump"].get("center_y_m", 0.0),
                    key=form_widget_key("ui_pump_center_y"),
                )
            with pump_pose_columns[2]:
                cfg["pump"]["rotation_deg"] = st.number_input(
                    "Pump 横向椭圆旋转 (deg)",
                    value=float(cfg["pump"].get("rotation_deg", 0.0)),
                    key=form_widget_key("ui_pump_rotation_deg"),
                )
            pump_axis_columns = st.columns(2)
            with pump_axis_columns[0]:
                cfg["pump"]["incidence_angle_x_deg"] = st.number_input(
                    "Pump 光轴倾斜 x (deg)",
                    value=float(cfg["pump"].get("incidence_angle_x_deg", 0.0)),
                    key=form_widget_key("ui_pump_axis_x_deg"),
                    help=(
                        "这是源光轴方向，不是发散角；gain_medium 元件中的入射角"
                        "另外决定晶体内斜程。"
                    ),
                )
            with pump_axis_columns[1]:
                cfg["pump"]["incidence_angle_y_deg"] = st.number_input(
                    "Pump 光轴倾斜 y (deg)",
                    value=float(cfg["pump"].get("incidence_angle_y_deg", 0.0)),
                    key=form_widget_key("ui_pump_axis_y_deg"),
                )
            direction_options = ["forward", "backward"]
            cfg["pump"]["direction"] = choice_selectbox(
                "泵浦方向", direction_options, index=option_index(direction_options, str(cfg["pump"].get("direction", "forward"))), key=form_widget_key("ui_pump_direction")
            )
            st.divider()
            st.markdown("#### C. 时间波形与光谱")
            temporal_options = ["cw", "gaussian", "rectangular", "super_gaussian", "file"]
            pump_temporal_type = choice_selectbox(
                "泵浦波形", temporal_options, index=option_index(temporal_options, str(cfg["pump"]["temporal"].get("type", "cw"))), key=form_widget_key("ui_pump_temporal_type")
            )
            cfg["pump"]["temporal"]["type"] = pump_temporal_type
            power_mode_options = ["average", "peak"]
            if pump_temporal_type == "cw":
                # Peak and average power are identical for a constant waveform;
                # hiding an irrelevant selector prevents a false extra degree of freedom.
                cfg["pump"]["temporal"]["power_mode"] = "average"
            elif pump_temporal_type != "file":
                cfg["pump"]["temporal"]["power_mode"] = choice_selectbox(
                    "功率定义", power_mode_options, index=option_index(power_mode_options, str(cfg["pump"]["temporal"].get("power_mode", "average"))), key=form_widget_key("ui_pump_power_mode")
                )
            if pump_temporal_type in {
                "gaussian",
                "rectangular",
                "super_gaussian",
            }:
                temporal_columns = st.columns(2)
                with temporal_columns[0]:
                    cfg["pump"]["temporal"]["pulse_fwhm_s"] = sci(
                        "泵浦脉宽 FWHM (s)",
                        cfg["pump"]["temporal"]["pulse_fwhm_s"],
                        key=form_widget_key("ui_pump_fwhm"),
                        help_text=(
                            "这是功率包络的全宽半高；super_gaussian 改变 order 时会重新定标，"
                            "使这里输入的 FWHM 对每个阶数都严格保持不变。"
                        ),
                    )
                with temporal_columns[1]:
                    cfg["pump"]["temporal"]["delay_s"] = sci(
                        "泵浦脉冲中心延迟 (s)",
                        cfg["pump"]["temporal"].get("delay_s", 0.0),
                        key=form_widget_key("ui_pump_delay"),
                        help_text=(
                            "从当前 Pump/布居时间窗口起点量到脉冲中心。重复运行时按周期回卷；"
                            "重复频率为 0 的单次瞬态不会把窗口末端回卷到开头。"
                        ),
                    )
                if pump_temporal_type == "super_gaussian":
                    cfg["pump"]["temporal"]["order"] = st.number_input(
                        "Pump 超高斯时间 order",
                        min_value=1.0,
                        value=float(cfg["pump"]["temporal"].get("order", 2.0)),
                        key=form_widget_key("ui_pump_temporal_order"),
                        help=(
                            "功率使用 exp[-ln(2)·|2(t−t0)/FWHM|^(2·order)]；"
                            "任意 order 下半高点都固定在 t0±FWHM/2。"
                        ),
                    )
            cfg["pump"]["temporal"]["sample_count"] = int(st.number_input(
                "Pump/布居每周期时间采样数",
                1,
                None,
                int(cfg["pump"]["temporal"]["sample_count"]),
                key=form_widget_key("ui_pump_nt"),
                help=(
                    "这是泵浦波形和布居速率方程的周期内采样，不是热传导 FEA 时间步。"
                    "CW 下增加它只细分同一周期的常功率积分，不会增加泵浦脉冲数。"
                ),
            ))
            if pump_temporal_type == "file":
                uploaded = st.file_uploader("泵浦波形 CSV：t(s), 相对功率或 W", type=["csv", "txt"], key=form_widget_key("pump_wave"))
                if uploaded is not None:
                    cfg["pump"]["temporal"]["samples"] = read_rows(uploaded)
                value_mode_options = ["relative", "absolute_W"]
                cfg["pump"]["temporal"]["value_mode"] = choice_selectbox(
                    "文件纵轴", value_mode_options,
                    index=option_index(value_mode_options, str(cfg["pump"]["temporal"].get("value_mode", "relative"))), key=form_widget_key("ui_pump_value_mode")
                )
                if cfg["pump"]["temporal"]["value_mode"] == "relative":
                    cfg["pump"]["temporal"]["power_mode"] = choice_selectbox(
                        "功率定义",
                        power_mode_options,
                        index=option_index(
                            power_mode_options,
                            str(cfg["pump"]["temporal"].get("power_mode", "average")),
                        ),
                        key=form_widget_key("ui_pump_power_mode"),
                    )
                else:
                    # Absolute-W samples already define the instantaneous power;
                    # periodic_pump_waveform deliberately ignores power_mode here.
                    cfg["pump"]["temporal"]["power_mode"] = "average"
                file_contract_options = ["zero_outside", "cover_period"]
                cfg["pump"]["temporal"]["file_period_contract"] = choice_selectbox(
                    "文件周期覆盖合同",
                    file_contract_options,
                    index=option_index(
                        file_contract_options,
                        str(
                            cfg["pump"]["temporal"].get(
                                "file_period_contract", "zero_outside"
                            )
                        ),
                    ),
                    key=form_widget_key("ui_pump_file_period_contract"),
                    help=(
                        "zero_outside 在文件时间范围外取 0；cover_period 要求文件显式覆盖完整 [0, 周期]。"
                    ),
                )
            spectrum_options = ["monochromatic", "gaussian", "lorentzian", "super_gaussian", "file"]
            pump_spectrum_type = choice_selectbox(
                "泵浦光谱", spectrum_options,
                index=option_index(spectrum_options, str(cfg["pump"]["spectrum"].get("type", "monochromatic"))), key=form_widget_key("ui_pump_spectrum_type")
            )
            cfg["pump"]["spectrum"]["type"] = pump_spectrum_type
            pump_density_domains = ["wavelength", "frequency"]
            cfg["pump"]["spectrum"]["spectral_density_domain"] = choice_selectbox(
                "泵浦光谱密度自变量",
                pump_density_domains,
                index=option_index(
                    pump_density_domains,
                    str(cfg["pump"]["spectrum"].get("spectral_density_domain", "wavelength")),
                ),
                key=form_widget_key("ui_pump_spectral_density_domain"),
                help="选择泵浦谱纵轴是每单位波长还是每单位频率的密度。",
            )
            if cfg["pump"]["spectrum"]["spectral_density_domain"] == "wavelength":
                cfg["pump"]["spectrum"]["bandwidth_fwhm_Hz"] = None
            if cfg["pump"]["spectrum"]["spectral_density_domain"] == "frequency":
                pump_bandwidth_units = ["wavelength_equivalent_m", "frequency_Hz"]
                pump_bandwidth_unit = choice_selectbox(
                    "泵浦频率密度 FWHM 输入",
                    pump_bandwidth_units,
                    index=option_index(
                        pump_bandwidth_units,
                        "frequency_Hz"
                        if float(cfg["pump"]["spectrum"].get("bandwidth_fwhm_Hz") or 0.0) > 0.0
                        else "wavelength_equivalent_m",
                    ),
                    key=form_widget_key("ui_pump_bw_unit"),
                    help="Hz 输入最无歧义；m 输入保持峰值波长和映回波长坐标后的总 FWHM。",
                )
            else:
                pump_bandwidth_unit = "wavelength_equivalent_m"
            if pump_bandwidth_unit == "frequency_Hz":
                cfg["pump"]["spectrum"]["bandwidth_fwhm_m"] = 0.0
                cfg["pump"]["spectrum"]["bandwidth_fwhm_Hz"] = sci(
                    "泵浦光谱 FWHM (Hz)",
                    cfg["pump"]["spectrum"].get("bandwidth_fwhm_Hz") or 0.0,
                    key=form_widget_key("ui_pump_bw_Hz"),
                    help_text=(
                        "这是所选光谱密度自变量上的强度谱 FWHM。super_gaussian 会按 order "
                        "重新定标，因此改变 order 不改变这里输入的半高全宽。"
                    ),
                )
            else:
                cfg["pump"]["spectrum"]["bandwidth_fwhm_Hz"] = None
                cfg["pump"]["spectrum"]["bandwidth_fwhm_m"] = sci(
                    "泵浦光谱 FWHM (m)",
                    cfg["pump"]["spectrum"]["bandwidth_fwhm_m"],
                    key=form_widget_key("ui_pump_bw"),
                    help_text=(
                        "这是所选光谱密度自变量上的强度谱 FWHM。super_gaussian 会按 order "
                        "重新定标，因此改变 order 不改变这里输入的半高全宽。"
                    ),
                )
            if pump_spectrum_type == "super_gaussian":
                cfg["pump"]["spectrum"]["order"] = st.number_input(
                    "Pump 光谱超高斯阶数",
                    min_value=1.0,
                    value=float(cfg["pump"]["spectrum"].get("order", 2.0)),
                    key=form_widget_key("ui_pump_spectrum_order"),
                    help=(
                        "所选光谱密度自变量上的强度使用严格 FWHM 定标；改变 order 不会改变"
                        "上方输入的半高全宽，只会改变谱顶和谱翼形状。"
                    ),
                )
            cfg["pump"]["spectrum"]["sample_count"] = int(st.number_input(
                "泵浦光谱采样", 1, 101, int(cfg["pump"]["spectrum"]["sample_count"]), key=form_widget_key("ui_pump_nl")
            ))
            pump_spatial_options = [
                "shared — 共享空间场（快速降阶）",
                "per_wavelength — 各波长独立空间输运",
            ]
            pump_spatial_label = choice_selectbox(
                "宽带 Pump 空间输运",
                pump_spatial_options,
                index=option_index(
                    pump_spatial_options,
                    str(cfg["pump"]["spectrum"].get("spatial_transport", "shared")),
                ),
                key=form_widget_key("ui_pump_spectral_spatial_transport"),
                disabled=pump_spectrum_type == "monochromatic",
            )
            cfg["pump"]["spectrum"]["spatial_transport"] = pump_spatial_label.split(" ")[0]
            if pump_spectrum_type == "file":
                uploaded = st.file_uploader("泵浦光谱 CSV：λ(nm), 相对功率谱", type=["csv", "txt"], key=form_widget_key("pump_spectrum"))
                if uploaded is not None:
                    cfg["pump"]["spectrum"]["samples"] = read_rows(uploaded)
                    cfg["pump"]["spectrum"]["wavelength_unit"] = "nm"
            st.divider()
            st.markdown("#### D. Pump 独立光路与半径探针")
            st.caption("只影响 pump；这里插入的透镜、距离和光阑不会进入 seed 光路。")
            cfg["optical_train"]["pump"] = render_optical_path_editor(cfg, "pump")
            render_beam_radius_probe(cfg, "pump")
            if resolve_physics(cfg)["pump_propagation"] == "prescribed_caustic" and any(
                element.get("type") != "gain_medium" or float(element.get("distance_before_m", 0.0)) > 0.0
                for element in cfg["optical_train"]["pump"]
            ):
                st.warning(
                    "当前泵浦传播是 prescribed_caustic：右侧会预览完整光路，但完整三维求解会忽略"
                    "晶体前的距离和元件。切换到 ABCD、FFT 或随机光线才会让它们进入求解。"
                )
                if st.button("让 Pump 光路进入求解（切换为 ABCD）", width="stretch", key="enable_pump_abcd"):
                    cfg["physics"]["pump_propagation"] = "abcd"
                    install_form_config(cfg, update_json_editor=False)
                    st.rerun()

        with beams_tab, st.expander(
            "Seed · 脉冲、空间与相干场",
            expanded=expand_long_sections,
        ):
            with st.expander("A. 脉冲、重复频率与 Fourier 输入", expanded=True):
                render_seed_pulse_definition(cfg)
            render_seed_transport_controls(cfg)
            render_seed_polarization_controls(cfg)
            with st.expander("D. 空间光束、M² 与横模", expanded=False):
                render_seed_spatial_controls(cfg)
                st.divider()
                caustic = cfg["seed"].setdefault(
                    "caustic_diagnostics",
                    copy.deepcopy(DEFAULTS["seed"]["caustic_diagnostics"]),
                )
                configured_scan = (
                    caustic.get("start_m", 0.0), caustic.get("stop_m", 0.1),
                    caustic.get("sample_count", 21),
                )
                caustic["enabled"] = choice_checkbox(
                    "扫描系统输出后的焦散并拟合 D4σ / M²",
                    value=bool(caustic.get("enabled", False)),
                    key=form_widget_key("ui_seed_caustic_enabled"),
                    help=(
                        "每个平面都从系统输出复场独立做角谱传播；只生成诊断，"
                        "不会反馈到晶体内增益、热源或布居。"
                    ),
                )
                if cfg["simulation"].get("result_recording", {}).get("diagnostics_timing") == "final_output":
                    st.caption("当前选择：焦散扫描仅在每个工况的最终输出执行。计算时机可在“结果记录”中修改。")
                caustic_columns = st.columns(3)
                with caustic_columns[0]:
                    caustic["start_m"] = sci(
                        "扫描起点 (m)",
                        caustic.get("start_m", 0.0),
                        key=form_widget_key("ui_seed_caustic_start"),
                    )
                with caustic_columns[1]:
                    caustic["stop_m"] = sci(
                        "扫描终点 (m)",
                        caustic.get("stop_m", 0.1),
                        key=form_widget_key("ui_seed_caustic_stop"),
                    )
                with caustic_columns[2]:
                    caustic["sample_count"] = int(
                        st.number_input(
                            "扫描平面数",
                            min_value=3,
                            max_value=201,
                            value=int(caustic.get("sample_count", 21)),
                            key=form_widget_key("ui_seed_caustic_count"),
                            disabled=not caustic["enabled"],
                        )
                    )
                caustic["refractive_index"] = st.number_input(
                    "焦散传播介质折射率",
                    min_value=0.01,
                    value=float(caustic.get("refractive_index", 1.0)),
                    step=0.01,
                    key=form_widget_key("ui_seed_caustic_index"),
                    disabled=not caustic["enabled"],
                    help="空气中填 1；若扫描面在均匀介质中，填该介质相位折射率。",
                )
                caustic["record_planes"] = choice_checkbox(
                    "保存每个焦散平面的二维光斑",
                    value=bool(caustic.get("record_planes", True)),
                    key=form_widget_key("ui_seed_caustic_record_planes"),
                    disabled=not caustic["enabled"],
                )
                # Only an edit of the uniform-scan controls replaces imported
                # positions; opening the form must preserve a nonuniform scan.
                if configured_scan != (caustic["start_m"], caustic["stop_m"], caustic["sample_count"]):
                    caustic["positions_m"] = []
                if caustic.get("positions_m"):
                    st.caption(lt("当前使用导入的显式焦散位置；修改扫描起点、终点或平面数后切换为均匀扫描。"))
            if cfg["simulation"]["signal_model"] in {"commercial_app_fast", "coherent_cpa"}:
                with st.expander("E. 输入复光谱、展宽与 CPA", expanded=False):
                    render_seed_coherent_controls(cfg)
            with st.expander("F. Seed 独立光路与半径探针", expanded=False):
                render_seed_path_controls(cfg)

        with beams_tab, st.expander(
            "多程：模板、命中与程间光路",
            expanded=bool(
                cfg["simulation"].get("multipass", {}).get("enabled", False)
            ),
        ):
            if bool(
                cfg["simulation"].get("multipass", {}).get("enabled", False)
            ):
                render_causal_multipass_controls(cfg)
            else:
                st.caption(
                    "当前为单程：只进行一次晶体命中，因此没有程间镜面、延时或重叠参数。"
                    "如需多程，请在上方勾选“启用多程放大”。"
                )

        with physics_tab:
            physics_summary = resolve_physics(cfg)
            st.caption(
                "材料链摘要："
                f"热 {_resolved_thermal_label(cfg, physics_summary)} · "
                f"折射率 {physics_summary['refractive_index_model']} · "
                f"应力 {cfg['material'].get('mechanical', {}).get('stress_model', 'none')} · "
                f"ASE {cfg.get('ase', {}).get('model', 'off')}。"
            )
        with physics_tab, st.expander(
            "热、应力、色散与非线性",
            expanded=expand_long_sections,
        ):
            full_josa_selected = _full_josa_selected(cfg)
            josa_selected = _josa_selected(cfg)
            strict_coupling = (
                str(cfg.get("multiphysics", {}).get("coupling_model", "one_way"))
                == "thermal_optical_fixed_point"
            )
            if strict_coupling:
                coupling_sources: list[str] = []
                if str(
                    cfg["material"]
                    .get("spectroscopic_response", {})
                    .get("mode", "off")
                ) == "local_3d":
                    coupling_sources.append("local_3d 光谱反馈")
                if str(
                    cfg.get("multiphysics", {}).get(
                        "seed_heat_coupling", "off"
                    )
                ) == "fixed_point":
                    coupling_sources.append("Seed 强场产热固定点")
                source_text = "、".join(coupling_sources) or "热光固定点"
                st.info(
                    f"依赖锁定来源：{source_text}。热场固定为 steady_3d，"
                    "折射率固定为 thermal_3d；关闭最后一个上游条件后，"
                    "自动托管值会恢复到联动前设置。"
                )
            elif full_josa_selected:
                st.info(
                    "full_3d_pc 的实际求解合同固定使用三维热场（auto 默认 steady_3d，"
                    "也可预先指定 transient_3d）；这里仍保留退出该层级后的用户配置。"
                )
            if josa_selected:
                st.caption(
                    "JOSA 中 Sellmeier 的 n(λ)/ng(λ) 仍用于各波长衍射与 full_3d_pc 群延迟；"
                    "下面额外的材料 GDD/TOD/FOD 相位和 n₂ 不进入 JOSA。切换回相干/Commercial App "
                    "后这些材料值会保留。"
                )
            configured_thermal_mode = str(cfg["physics"].get("thermal_model", "auto"))
            thermal_options = (
                ["steady_3d — 三维稳态"]
                if strict_coupling
                else [
                    "auto — 跟随配置默认值/材料开关",
                    "configured — 按材料热开关",
                    "off — 关闭",
                    "steady_3d — 三维稳态",
                    "transient_3d — 三维瞬态",
                ]
            )
            thermal_widget_key = form_widget_key(
                "ui_thermal_model_coupling_lock"
                if strict_coupling
                else "ui_thermal_model"
            )
            if strict_coupling:
                st.session_state[thermal_widget_key] = thermal_options[0]
            thermal_label = choice_selectbox(
                "热场模型",
                thermal_options,
                key=thermal_widget_key,
                disabled=strict_coupling,
                **({} if thermal_widget_key in st.session_state else {
                    "index": option_index(thermal_options, configured_thermal_mode)
                }),
            )
            thermal_mode = thermal_label.split(" ")[0]
            if not strict_coupling:
                cfg["physics"]["thermal_model"] = thermal_mode
            if not strict_coupling and thermal_mode not in {"auto", "configured"}:
                cfg["material"]["thermal"]["enabled"] = thermal_mode != "off"
                cfg["material"]["thermal"]["transient"]["enabled"] = thermal_mode == "transient_3d"
            active_thermal_mode = resolve_thermal_model(cfg)
            if thermal_mode == "auto":
                st.caption(f"当前预设实际解析为：{active_thermal_mode}；选择 auto 不会改写材料热开关。")
            heat_source_options = [
                "pump_component_ledger — 量子亏损快速账本",
                "event_energy_ledger — 逐跃迁/衰变/非线性事件账本",
            ]
            configured_heat_source = str(
                cfg["material"]["thermal"].get(
                    "heat_source_model", "pump_component_ledger"
                )
            )
            heat_source_label = choice_selectbox(
                "热源账本",
                heat_source_options,
                index=option_index(
                    heat_source_options,
                    configured_heat_source,
                ),
                key=form_widget_key("ui_heat_source_model"),
            )
            selected_heat_source = heat_source_label.split(" ")[0]
            cfg["material"]["thermal"][
                "heat_source_model"
            ] = selected_heat_source
            event_heat_selected = selected_heat_source == "event_energy_ledger"
            if event_heat_selected:
                event_cfg = cfg["energy_levels"].setdefault(
                    "event_energy_ledger", {}
                )
                event_cfg["enabled"] = True
                st.caption(
                    "逐事件账本会保留导入配置中的 decay_partitions 和 "
                    "nonlinear_partitions；每条通道必须明确分配到热、荧光、"
                    "ASE 或未解析项。高级分配请在 JSON 中编辑。"
                )
                st.write(
                    {
                        "衰变分配数": len(
                            event_cfg.get("decay_partitions", [])
                        ),
                        "非线性分配数": len(
                            event_cfg.get("nonlinear_partitions", [])
                        ),
                    }
                )
            else:
                st.caption(
                    "Pump 分量账本：离子主动吸收按所选量子亏损产热，"
                    "材料被动吸收全部转热；ASE 与 Seed 不可逆损耗分别记账。"
                )
            configured_heat_fraction = cfg["material"]["thermal"].get(
                "heat_fraction", "quantum_defect"
            )
            heat_fraction_token = (
                str(configured_heat_fraction)
                if isinstance(configured_heat_fraction, str)
                else "explicit_fraction"
            )
            active_heat_options = [
                "lifetime_quantum_defect — 寿命自动分解：量子亏损 + 非辐射衰变（推荐）",
                "quantum_defect — 吸收能量守恒：1−ηq·λp/λref",
                "scaled_quantum_defect — 可调倍率×(1−λp/λref)",
                "spectral_quantum_defect — 按 Pump 光谱与局部截面计算量子亏损",
                "spectral_scaled_quantum_defect — 按 Pump 光谱与局部截面计算可调量子亏损",
                "explicit_fraction — 直接输入实测主动吸收产热比例",
            ]
            active_heat_label = choice_selectbox(
                "主动 Pump 吸收产热模型",
                active_heat_options,
                index=option_index(active_heat_options, heat_fraction_token),
                key=form_widget_key("ui_active_heat_model"),
                disabled=event_heat_selected,
                help=(
                    "泵浦激发量子效率始终只由能级窗口控制布居速率。"
                    "lifetime_quantum_defect 用总寿命与辐射寿命之差，"
                    "按实际上能级布居计算非辐射热，不把荧光效率重复乘到激发率；"
                    "quantum_defect 还把未形成有效激发的吸收能量计入热；"
                    "scaled_quantum_defect 不乘该量子效率，而使用独立倍率。"
                    "spectral 两项会按每个 Pump 波长的功率权重和局部净吸收截面计算空间变化的热比例。"
                ),
            )
            active_heat_model = active_heat_label.split(" ")[0]
            if not event_heat_selected:
                if active_heat_model == "explicit_fraction":
                    explicit_fraction = (
                        float(configured_heat_fraction)
                        if not isinstance(configured_heat_fraction, str)
                        else float(cfg["material"].get("library", {}).get("default_heat_fraction") or 0.0)
                    )
                    explicit_percent = st.number_input(
                        "主动吸收实测产热比例 (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=100.0 * explicit_fraction,
                        step=0.1,
                        key=form_widget_key("ui_explicit_heat_fraction"),
                    )
                    cfg["material"]["thermal"]["heat_fraction"] = float(explicit_percent) / 100.0
                else:
                    cfg["material"]["thermal"]["heat_fraction"] = active_heat_model
                if active_heat_model == "lifetime_quantum_defect":
                    configured_radiative_lifetime = cfg["energy_levels"].get(
                        "upper_state_radiative_lifetime_s"
                    )
                    radiative_lifetime = sci(
                        "上能级辐射寿命 τrad (s)",
                        (
                            float(configured_radiative_lifetime)
                            if configured_radiative_lifetime is not None
                            else float(cfg["energy_levels"]["upper_state_lifetime_s"])
                        ),
                        key=form_widget_key("ui_tau_upper_radiative"),
                    )
                    cfg["energy_levels"][
                        "upper_state_radiative_lifetime_s"
                    ] = radiative_lifetime
                    cfg["material"]["thermal"]["extra_heat_fraction"] = 0.0
                    st.caption(
                        "非辐射热自动按 Nupper·hν·(1/τtotal−1/τrad) 计算；"
                        "τtotal 使用上方固定寿命或已启用的温度/掺杂寿命响应。"
                        "该模式禁用手工“额外非辐射比例”，避免重复记账。"
                    )
                if active_heat_model in {"scaled_quantum_defect", "spectral_scaled_quantum_defect"}:
                    cfg["material"]["thermal"]["stokes_shift_multiplier"] = st.number_input(
                        "量子亏损产热倍率",
                        min_value=0.0,
                        value=float(cfg["material"]["thermal"].get("stokes_shift_multiplier", 1.0)),
                        step=0.01,
                        key=form_widget_key("ui_stokes_heat_multiplier"),
                    )
                if active_heat_model != "lifetime_quantum_defect":
                    extra_heat_percent = st.number_input(
                        "额外非辐射产热比例（百分点）",
                        min_value=0.0,
                        max_value=100.0,
                        value=100.0 * float(cfg["material"]["thermal"].get("extra_heat_fraction", 0.0)),
                        step=0.1,
                        key=form_widget_key("ui_extra_heat_fraction"),
                        help="最终主动产热比例为所选模型结果再加此项；总和不能超过 100%。",
                    )
                    cfg["material"]["thermal"]["extra_heat_fraction"] = float(extra_heat_percent) / 100.0
            cfg["material"]["thermal"]["include_seed_passive_absorption_heat"] = choice_checkbox(
                "把 Seed 被动吸收作为热源",
                value=bool(cfg["material"]["thermal"].get("include_seed_passive_absorption_heat", False)),
                key=form_widget_key("ui_seed_passive_heat"),
                help="需要在 A 中选择 Seed 不可逆产热耦合；主动增益提取不算热，被动损耗才按 100% 转热。",
            )
            if active_thermal_mode == "steady_3d":
                thermal_solver_options = ["auto — 自动（优先稀疏 CG）", "cg — 稀疏共轭梯度", "sor — 红黑 SOR"]
                thermal_solver_label = choice_selectbox(
                    "稳态热线性求解器",
                    thermal_solver_options,
                    index=option_index(
                        thermal_solver_options,
                        str(cfg["material"]["thermal"].get("linear_solver", "auto")),
                    ),
                    key=form_widget_key("ui_thermal_linear_solver"),
                    help="auto 在 SciPy 可用时使用守恒有限体积稀疏 CG，否则回退原有 SOR。",
                )
                cfg["material"]["thermal"]["linear_solver"] = thermal_solver_label.split(" ")[0]
            if cfg["material"].get("geometry", {}).get("shape") == "cylinder":
                boundary_area_options = [
                    "analytic_cylinder — 圆柱解析侧面积守恒（推荐）",
                    "cartesian — 保留阶梯网格表面积",
                ]
                boundary_area_label = choice_selectbox(
                    "圆柱热边界面积离散",
                    boundary_area_options,
                    index=option_index(
                        boundary_area_options,
                        str(
                            cfg["material"]["thermal"].get(
                                "embedded_boundary_area_model", "cartesian"
                            )
                        ),
                    ),
                    key=form_widget_key("ui_embedded_boundary_area_model"),
                    help=(
                        "圆柱的 cell-mask 阶梯周长会系统性放大 Robin 散热。解析模式按当前网格"
                        "计算面积归一系数；不是 FEA，也不是针对某个输出功率的拟合。"
                    ),
                )
                cfg["material"]["thermal"]["embedded_boundary_area_model"] = (
                    boundary_area_label.split(" ")[0]
                )
            else:
                cfg["material"]["thermal"]["embedded_boundary_area_model"] = "cartesian"
            cfg["material"]["dn_dT_per_K"] = sci("dn/dT (K⁻¹)", cfg["material"]["dn_dT_per_K"], key=form_widget_key("ui_dndt"))
            dn_definition_options = [
                "measured_free_expansion — 自由膨胀实测热光系数",
                "intrinsic_fixed_strain — 固定应变本征热光系数",
            ]
            dn_definition_label = choice_selectbox(
                "dn/dT 数据定义",
                dn_definition_options,
                index=option_index(
                    dn_definition_options,
                    str(cfg["material"].get("dn_dT_definition", "measured_free_expansion")),
                ),
                key=form_widget_key("ui_dndt_definition"),
                help="与张量光弹的应变定义共同决定是否会重复计入自由热膨胀。",
            )
            cfg["material"]["dn_dT_definition"] = dn_definition_label.split(" ")[0]
            refraction_options = ["auto", "constant", "thermal_3d"]
            configured_refraction_model = str(
                cfg["physics"].get("refractive_index_model", "auto")
            )
            refraction_locked = full_josa_selected or strict_coupling
            refraction_key = form_widget_key(
                "ui_refraction_model_contract_lock"
                if refraction_locked
                else "ui_refraction_model"
            )
            if refraction_locked:
                st.session_state[refraction_key] = "thermal_3d"
            displayed_refraction_model = choice_selectbox(
                "折射率模型", refraction_options,
                key=refraction_key,
                disabled=refraction_locked,
                help="full_3d_pc 或热光固定点固定为 thermal_3d；其他层级可保持 auto 跟随配置默认值。",
                **({} if refraction_key in st.session_state else {
                    "index": option_index(
                        refraction_options,
                        configured_refraction_model,
                    )
                }),
            )
            if not refraction_locked:
                cfg["physics"]["refractive_index_model"] = (
                    displayed_refraction_model
                )
            field_diagnostics = cfg["material"].setdefault(
                "field_diagnostics",
                copy.deepcopy(DEFAULTS["material"]["field_diagnostics"]),
            )
            with st.expander("三维场抛物拟合诊断", expanded=False):
                field_diagnostics["enabled"] = choice_checkbox(
                    "诊断热折射率与增益的抛物近似质量",
                    value=bool(field_diagnostics.get("enabled", False)),
                    key=form_widget_key("ui_field_diagnostics_enabled"),
                )
                st.caption(
                    "该拟合只报告焦距、主轴和残差，不会替换、叠加或校准求解场；"
                    "实际热相位传播由所选 Seed 后端及热透镜闭合决定。"
                )
                fit_mode_labels = {
                    "automatic": "automatic — 使用有效晶体横截面",
                    "manual": "manual — 手动指定椭圆拟合半径",
                    "seed_entrance": "seed_entrance — 使用 Seed 晶体入口 D4σ 半径",
                }
                fit_mode = choice_selectbox(
                    "拟合半径来源",
                    list(fit_mode_labels),
                    index=option_index(
                        list(fit_mode_labels),
                        str(field_diagnostics.get("fit_radius_mode", "automatic")),
                    ),
                    format_func=lambda value: fit_mode_labels[value],
                    key=form_widget_key("ui_crystal_fit_radius_mode"),
                    disabled=not field_diagnostics["enabled"],
                )
                field_diagnostics["fit_radius_mode"] = fit_mode
                if fit_mode == "manual":
                    fit_columns = st.columns(2)
                    with fit_columns[0]:
                        field_diagnostics["manual_radius_x_m"] = sci(
                            "拟合半径 x (m)",
                            field_diagnostics.get("manual_radius_x_m") or cfg["seed"]["waist_x_m"],
                            key=form_widget_key("ui_crystal_fit_radius_x"),
                        )
                    with fit_columns[1]:
                        field_diagnostics["manual_radius_y_m"] = sci(
                            "拟合半径 y (m)",
                            field_diagnostics.get("manual_radius_y_m") or cfg["seed"]["waist_y_m"],
                            key=form_widget_key("ui_crystal_fit_radius_y"),
                        )
                elif fit_mode == "seed_entrance":
                    field_diagnostics["beam_radius_scale"] = st.number_input(
                        "Seed 入口半径倍率",
                        min_value=0.05,
                        value=float(field_diagnostics.get("beam_radius_scale", 1.0)),
                        step=0.05,
                        key=form_widget_key("ui_crystal_fit_beam_scale"),
                    )
                field_slice_key = form_widget_key("ui_crystal_fit_slices")
                field_slice_value = min(
                    max(
                        int(
                            st.session_state.get(
                                field_slice_key,
                                field_diagnostics.get("slice_count", 5),
                            )
                        ),
                        1,
                    ),
                    int(cfg["grid"]["nz"]),
                )
                st.session_state[field_slice_key] = field_slice_value
                displayed_field_slice_count = int(
                    st.number_input(
                        "沿 z 检查的拟合截面数",
                        min_value=1,
                        max_value=int(cfg["grid"]["nz"]),
                        key=field_slice_key,
                        disabled=not field_diagnostics["enabled"],
                    )
                )
                if field_diagnostics["enabled"]:
                    field_diagnostics["slice_count"] = displayed_field_slice_count
                fit_detail_columns = st.columns(2)
                with fit_detail_columns[0]:
                    field_diagnostics["zernike_max_radial_order"] = int(
                        st.number_input(
                            "Zernike 最大径向阶数",
                            min_value=4,
                            max_value=12,
                            value=int(
                                field_diagnostics.get(
                                    "zernike_max_radial_order", 4
                                )
                            ),
                            key=form_widget_key("ui_zernike_max_order"),
                            disabled=not field_diagnostics["enabled"],
                        )
                    )
                with fit_detail_columns[1]:
                    field_diagnostics["radial_bin_count"] = int(
                        st.number_input(
                            "OPD 径向平均分箱数",
                            min_value=8,
                            max_value=256,
                            value=int(field_diagnostics.get("radial_bin_count", 48)),
                            key=form_widget_key("ui_opd_radial_bins"),
                            disabled=not field_diagnostics["enabled"],
                        )
                    )
            conductivity_value = cfg["material"]["thermal"].get("conductivity_W_mK", 10.0)
            if isinstance(conductivity_value, (int, float)):
                conductivity_values = [float(conductivity_value)] * 3
            else:
                conductivity_values = list(map(float, conductivity_value))
                if len(conductivity_values) != 3:
                    conductivity_values = [conductivity_values[0]] * 3
            k_columns = st.columns(3)
            edited_conductivity: list[float] = []
            for index, (axis, column) in enumerate(zip("xyz", k_columns, strict=True)):
                with column:
                    edited_conductivity.append(
                        sci(f"导热率 k{axis} (W/m/K)", conductivity_values[index], key=form_widget_key(f"ui_conductivity_{axis}"))
                    )
            conductivity_values = edited_conductivity
            cfg["material"]["thermal"]["conductivity_W_mK"] = conductivity_values
            thermal_entities = list(
                cfg["material"]["thermal"].get("entities", [])
            )
            thermal_contacts = list(
                cfg["material"]["thermal"].get("contacts", [])
            )
            if thermal_entities:
                st.caption(
                    f"已保留导入的 {len(thermal_entities)} 个热学实体和 "
                    f"{len(thermal_contacts)} 个接触界面；实体区域与接触热阻"
                    "属于高级 JSON 配置。"
                )
            transient_columns = st.columns(2)
            with transient_columns[0]:
                cfg["material"]["thermal"]["density_kg_m3"] = sci(
                    "材料密度 (kg/m³)", cfg["material"]["thermal"].get("density_kg_m3", 4000.0), key=form_widget_key("ui_mass_density")
                )
            with transient_columns[1]:
                cfg["material"]["thermal"]["heat_capacity_J_kgK"] = sci(
                    "比热 (J/kg/K)", cfg["material"]["thermal"].get("heat_capacity_J_kgK", 500.0), key=form_widget_key("ui_heat_capacity")
                )
            if active_thermal_mode == "transient_3d":
                transient_cfg = cfg["material"]["thermal"]["transient"]
                transient_cfg["duration_s"] = sci(
                    "瞬态热演化时间 (s)", transient_cfg.get("duration_s", 1.0), key=form_widget_key("ui_thermal_duration")
                )
                transient_cfg["time_steps"] = int(st.number_input(
                    "瞬态热时间步数（FVM）", min_value=1, max_value=10000,
                    value=int(transient_cfg.get("time_steps", 20)), key=form_widget_key("ui_thermal_time_steps")
                ))
                use_custom_initial_temperature = choice_checkbox(
                    "指定初始晶体温度", value=transient_cfg.get("initial_temperature_K") is not None,
                    key=form_widget_key("ui_thermal_custom_initial")
                )
                transient_cfg["initial_temperature_K"] = (
                    st.number_input(
                        "初始晶体温度 (K)", min_value=1.0,
                        value=float(transient_cfg.get("initial_temperature_K") or cfg["material"].get("reference_temperature_K", 293.15)),
                        key=form_widget_key("ui_thermal_initial_temperature"),
                    )
                    if use_custom_initial_temperature else None
                )
                st.caption(
                    "瞬态模型使用 ρ·cp·∂T/∂t 和隐式 Euler；允许全绝热边界，此时吸收热会累积为温升。"
                    "这里的 Δt_thermal=总热时间/热步数，与 Pump/布居时间步 Δt_population 独立；"
                    "通常热时间步可以更粗。"
                )
            else:
                st.caption("稳态模型只使用导热率和边界；密度与比热在瞬态模型中决定热惯性。")
            cooling_options = ["water — 水冷/对流", "fixed_temperature — 边界固定温度", "adiabatic — 绝热"]
            configured_cooling_kind = str(
                cfg["material"]["thermal"]["boundary"].get(
                    "type", "fixed_temperature"
                )
            )
            canonical_cooling_kind = {
                "dirichlet": "fixed_temperature",
                "fixed": "fixed_temperature",
                "fixed_temperature": "fixed_temperature",
                "convective": "water",
                "convection": "water",
                "water": "water",
                "adiabatic": "adiabatic",
            }.get(configured_cooling_kind, configured_cooling_kind)
            cooling_label = choice_selectbox(
                "冷却边界", cooling_options,
                index=option_index(cooling_options, canonical_cooling_kind), key=form_widget_key("ui_cooling_type")
            )
            cooling_kind = cooling_label.split(" ")[0]
            # Preserve a physically equivalent imported alias until the user
            # actually selects a different boundary.  In particular,
            # ``dirichlet`` must never fall through to the first option
            # (water/Robin cooling) merely because the form was opened.
            cfg["material"]["thermal"]["boundary"]["type"] = (
                configured_cooling_kind
                if cooling_kind == canonical_cooling_kind
                else cooling_kind
            )
            if cooling_kind == "water":
                cfg["material"]["thermal"]["boundary"]["coolant_temperature_K"] = st.number_input(
                    "冷却水温度 (K)", value=float(cfg["material"]["thermal"]["boundary"].get("coolant_temperature_K", 293.15)), key=form_widget_key("ui_coolant_temperature")
                )
                cfg["material"]["thermal"]["boundary"]["heat_transfer_W_m2K"] = sci(
                    "水冷换热系数 h (W/m²/K)", cfg["material"]["thermal"]["boundary"].get("heat_transfer_W_m2K", 10000.0), key=form_widget_key("ui_water_h"),
                    help_text="这是固体侧 Robin 边界，不是流体 CFD；可由流量/通道关联式或实测热阻给定。",
                )
            elif cooling_kind == "fixed_temperature":
                cfg["material"]["thermal"]["boundary"]["ambient_temperature_K"] = st.number_input(
                    "固定边界温度 (K)", value=float(cfg["material"]["thermal"]["boundary"].get("ambient_temperature_K", 293.15)), key=form_widget_key("ui_fixed_temperature")
                )
            cfg["material"]["thermal"]["boundary"]["faces"] = parse_json(
                st.text_area(
                    "各表面冷却覆盖（可选 JSON）",
                    value=json.dumps(cfg["material"]["thermal"]["boundary"].get("faces", {}), ensure_ascii=False, indent=2),
                    height=90,
                    key=form_widget_key("ui_thermal_faces_json"),
                    help="可分别覆盖 x_min/x_max/y_min/y_max/z_min/z_max，例如只让 y_min/y_max 水冷，其余绝热。",
                ),
                "各表面冷却边界",
            )
            stress_options = [
                "none — 关闭",
                "isotropic_proxy — 快速局部风险代理",
                "thermoelastic_fem — 三维热弹性力平衡",
                "anisotropic_thermoelastic_fem — 各向异性张量 FEM",
            ]
            stress_label = choice_selectbox(
                "应力模型", stress_options,
                index=option_index(stress_options, str(cfg["material"]["mechanical"].get("stress_model", "none"))), key=form_widget_key("ui_stress_model")
            )
            cfg["material"]["mechanical"]["stress_model"] = stress_label.split(" ")[0]
            if cfg["material"]["mechanical"]["stress_model"] != "none":
                mechanical_columns = st.columns(2)
                with mechanical_columns[0]:
                    cfg["material"]["mechanical"]["young_modulus_Pa"] = sci(
                        "杨氏模量 (Pa)", cfg["material"]["mechanical"].get("young_modulus_Pa", 1e11), key=form_widget_key("ui_young_modulus")
                    )
                    cfg["material"]["mechanical"]["thermal_expansion_per_K"] = sci(
                        "热膨胀系数 (K⁻¹)", cfg["material"]["mechanical"].get("thermal_expansion_per_K", 8e-6), key=form_widget_key("ui_thermal_expansion")
                    )
                with mechanical_columns[1]:
                    cfg["material"]["mechanical"]["poisson_ratio"] = st.number_input(
                        "泊松比", min_value=0.0, max_value=0.499, key=form_widget_key("ui_poisson_ratio"),
                        **widget_default("ui_poisson_ratio", float(cfg["material"]["mechanical"].get("poisson_ratio", 0.25))),
                    )
                    cfg["material"]["mechanical"]["photoelastic_coefficient_Pa_inv"] = sci(
                        "有效应力光学系数 (Pa⁻¹)", cfg["material"]["mechanical"].get("photoelastic_coefficient_Pa_inv", 0.0), key=form_widget_key("ui_photoelastic_scalar")
                    )
                if cfg["material"]["mechanical"]["stress_model"] in {
                    "thermoelastic_fem",
                    "anisotropic_thermoelastic_fem",
                }:
                    fem_cfg = cfg["material"]["mechanical"].setdefault(
                        "thermoelastic_fem", {}
                    )
                    st.caption(
                        "求解 ∇·σ=0 的小应变三维热弹性方程；外表面默认无牵引。"
                        "未固定表面时只用三个点消除刚体自由度，不阻止均匀自由膨胀。"
                    )
                    fem_columns = st.columns(2)
                    with fem_columns[0]:
                        fem_cfg["maximum_active_elements"] = st.number_input(
                            "力学最大活动单元数",
                            min_value=1,
                            step=100,
                            value=int(fem_cfg.get("maximum_active_elements", 8000)),
                            key=form_widget_key("ui_fem_maximum_elements"),
                            help="超过预算会明确报错，不会静默粗化应力热点。可使用较粗物理网格和独立精细光学网格。",
                        )
                        fem_cfg["linear_solver"] = choice_selectbox(
                            "力学线性求解器",
                            ["auto", "direct", "cg"],
                            index=option_index(
                                ["auto", "direct", "cg"],
                                str(fem_cfg.get("linear_solver", "auto")),
                            ),
                            key=form_widget_key("ui_fem_linear_solver"),
                        )
                    with fem_columns[1]:
                        photoelastic_options = (
                            ["off", "tensor_photoelastic"]
                            if cfg["material"]["mechanical"]["stress_model"] == "anisotropic_thermoelastic_fem"
                            else ["off", "scalar_von_mises_proxy"]
                        )
                        fem_cfg["photoelastic_coupling"] = choice_selectbox(
                            "FEM 应力/应变到折射率",
                            photoelastic_options,
                            index=option_index(
                                photoelastic_options,
                                str(fem_cfg.get("photoelastic_coupling", "off")),
                            ),
                            key=form_widget_key("ui_fem_photoelastic_coupling"),
                            help="off 只输出力学场；scalar_von_mises_proxy 才使用上方单个有效系数。任意晶向双折射需要完整光弹张量。",
                        )
                        fem_cfg["relative_tolerance"] = sci(
                            "力平衡相对容差",
                            fem_cfg.get("relative_tolerance", 1e-8),
                            key=form_widget_key("ui_fem_relative_tolerance"),
                        )
                        fem_cfg["maximum_iterations"] = st.number_input(
                            "CG 最大迭代",
                            min_value=1,
                            step=100,
                            value=int(fem_cfg.get("maximum_iterations", 2000)),
                            key=form_widget_key("ui_fem_maximum_iterations"),
                        )
                    mechanical_faces = [
                        "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"
                    ]
                    fem_cfg["fixed_faces"] = choice_multiselect(
                        "完全固定的晶体表面（留空=全部无牵引）",
                        mechanical_faces,
                        default=list(fem_cfg.get("fixed_faces", [])),
                        key=form_widget_key("ui_fem_fixed_faces"),
                        help="被选表面的三个位移分量均固定为零；只在真实装夹近似完全刚固时使用。",
                    )
                    st.caption(
                        "输出 ux/uy/uz、六个应力与应变分量、von Mises、轴向位移范围和力平衡残差。"
                        "单个有效应力光学系数仍是标量近似；任意晶向张量光弹需要独立材料张量数据。"
                    )
                    if cfg["material"]["mechanical"]["stress_model"] == "anisotropic_thermoelastic_fem":
                        anisotropic = cfg["material"]["mechanical"].setdefault("anisotropic", {})
                        anisotropic["elastic_stiffness_Pa"] = parse_json(
                            st.text_area(
                                "弹性刚度 C（6×6，Pa）",
                                value=json.dumps(anisotropic.get("elastic_stiffness_Pa", []), ensure_ascii=False),
                                height=120,
                                key=form_widget_key("ui_anisotropic_stiffness"),
                            ),
                            "各向异性弹性刚度",
                        )
                        anisotropic["thermal_expansion_tensor_per_K"] = parse_json(
                            st.text_area(
                                "热膨胀 Voigt 张量（6 分量，K⁻¹）",
                                value=json.dumps(anisotropic.get("thermal_expansion_tensor_per_K", []), ensure_ascii=False),
                                height=70,
                                key=form_widget_key("ui_anisotropic_expansion"),
                            ),
                            "各向异性热膨胀张量",
                        )
                        orientation_mode = choice_selectbox(
                            "晶体取向输入",
                            ["euler_xyz_deg", "rotation_matrix"],
                            index=(
                                1
                                if anisotropic.get("orientation_matrix_material_to_lab", [])
                                else 0
                            ),
                            key=form_widget_key("ui_anisotropic_orientation_mode"),
                        )
                        if orientation_mode == "euler_xyz_deg":
                            anisotropic["orientation_matrix_material_to_lab"] = []
                            anisotropic["orientation_euler_xyz_deg"] = parse_json(
                                st.text_input(
                                    "材料→实验室 Euler xyz (deg)",
                                    value=json.dumps(anisotropic.get("orientation_euler_xyz_deg", [0, 0, 0])),
                                    key=form_widget_key("ui_anisotropic_orientation"),
                                ),
                                "晶体取向",
                            )
                        else:
                            anisotropic["orientation_matrix_material_to_lab"] = parse_json(
                                st.text_area(
                                    "材料→实验室旋转矩阵（3×3）",
                                    value=json.dumps(anisotropic.get("orientation_matrix_material_to_lab", np.eye(3).tolist())),
                                    height=90,
                                    key=form_widget_key("ui_anisotropic_orientation_matrix"),
                                ),
                                "晶体旋转矩阵",
                            )
                        if fem_cfg.get("photoelastic_coupling") == "tensor_photoelastic":
                            strain_options = [
                                "elastic_strain — 仅弹性应变（推荐配合实测自由膨胀 dn/dT）",
                                "total_strain — 总应变（需本征 fixed-strain dn/dT）",
                            ]
                            strain_label = choice_selectbox(
                                "光弹矩阵作用的应变",
                                strain_options,
                                index=option_index(
                                    strain_options,
                                    str(anisotropic.get("photoelastic_strain_definition", "elastic_strain")),
                                ),
                                key=form_widget_key("ui_photoelastic_strain_definition"),
                            )
                            anisotropic["photoelastic_strain_definition"] = strain_label.split(" ")[0]
                            anisotropic["photoelastic_matrix"] = parse_json(
                                st.text_area(
                                    "光弹矩阵 p（6×6）",
                                    value=json.dumps(anisotropic.get("photoelastic_matrix", []), ensure_ascii=False),
                                    height=120,
                                    key=form_widget_key("ui_anisotropic_photoelastic"),
                                ),
                                "光弹张量",
                            )
                            anisotropic["reference_refractive_index"] = st.number_input(
                                "光弹参考折射率",
                                min_value=0.01,
                                value=float(anisotropic.get("reference_refractive_index") or cfg["material"]["refractive_index"]),
                                key=form_widget_key("ui_anisotropic_photoelastic_index"),
                            )
                    endface = cfg["material"]["mechanical"].setdefault("endface_deformation", {})
                    endface["model"] = choice_selectbox(
                        "FEM 端面位移光学反馈",
                        ["off", "distributed_opd"],
                        index=option_index(["off", "distributed_opd"], str(endface.get("model", "off"))),
                        key=form_widget_key("ui_endface_feedback"),
                        help="distributed_opd 把前后端面轴向 sag 差换算为等效分布 OPD；不会伪装成局部法向光线追迹。",
                    )
            st.divider()
            st.markdown("**局部温度 / 掺杂光谱反馈**")
            response_cfg = cfg["material"].setdefault("spectroscopic_response", {})
            imported_response_data = response_cfg.get("available_data", [])
            if imported_response_data:
                st.info(
                    "当前材料库已提供可用响应数据："
                    + "、".join(map(str, imported_response_data))
                    + "。导入数据与启用反馈是两件事；off 不会改变旧结果。"
                )
            response_options = [
                "off — 使用材料导入/表单给定的固定参数",
                "operating_point — 在统一工作温度和参考掺杂下修正",
                "local_3d — 热光固定点内逐体素重新计算",
            ]
            response_label = choice_selectbox(
                "光谱参数反馈模式",
                response_options,
                index=option_index(response_options, str(response_cfg.get("mode", "off"))),
                key=form_widget_key("ui_spectroscopic_response_mode"),
            )
            response_cfg["mode"] = response_label.split(" ")[0]
            range_policy_options = [
                "record — 记录端点夹持并交给结果有效性门判断",
                "error — 一旦超出材料标定范围立即报错",
            ]
            range_policy_label = choice_selectbox(
                "材料表越界策略",
                range_policy_options,
                index=option_index(
                    range_policy_options,
                    str(response_cfg.get("range_policy", "record")),
                ),
                key=form_widget_key("ui_spectroscopy_range_policy"),
            )
            response_cfg["range_policy"] = range_policy_label.split(" ")[0]
            response_cfg["reference_temperature_K"] = st.number_input(
                "材料数据参考温度 (K)",
                min_value=1.0,
                value=float(response_cfg.get("reference_temperature_K", cfg["material"].get("material_data_reference_temperature_K", 293.15))),
                key=form_widget_key("ui_spectroscopy_reference_temperature"),
            )
            response_cfg["reference_density_m3"] = sci(
                "材料数据参考离子密度 (m⁻³)",
                response_cfg.get("reference_density_m3", cfg["material"].get("ion_density_m3", 1e24)),
                key=form_widget_key("ui_spectroscopy_reference_density"),
            )
            lifetime_response = response_cfg.setdefault("upper_state_lifetime", {})
            temperature_law = lifetime_response.setdefault("temperature", {"model": "off"})
            concentration_law = lifetime_response.setdefault("concentration", {"model": "off"})
            response_columns = st.columns(2)
            with response_columns[0]:
                lifetime_temperature_options = ["off", "linear_scale", "scale_table"]
                temperature_law["model"] = choice_selectbox(
                    "上能级寿命随温度",
                    lifetime_temperature_options,
                    index=option_index(lifetime_temperature_options, str(temperature_law.get("model", "off"))),
                    key=form_widget_key("ui_lifetime_temperature_model"),
                )
                if temperature_law["model"] == "linear_scale":
                    temperature_law["slope_per_unit"] = sci(
                        "寿命相对斜率 (K⁻¹)",
                        temperature_law.get("slope_per_unit", 0.0),
                        key=form_widget_key("ui_lifetime_temperature_slope"),
                    )
                elif temperature_law["model"] == "scale_table":
                    lifetime_response["temperature"] = parse_json(
                        st.text_area(
                            "寿命温度缩放表 JSON",
                            value=json.dumps(temperature_law, ensure_ascii=False, indent=2),
                            height=110,
                            key=form_widget_key("ui_lifetime_temperature_table"),
                            help='格式：{"model":"scale_table","coordinate":[293,350],"scale":[1,0.7]}',
                        ),
                        "寿命温度缩放表",
                    )
            with response_columns[1]:
                concentration_options = ["off", "stern_volmer", "scale_table"]
                concentration_law["model"] = choice_selectbox(
                    "上能级寿命随掺杂浓度",
                    concentration_options,
                    index=option_index(concentration_options, str(concentration_law.get("model", "off"))),
                    key=form_widget_key("ui_lifetime_concentration_model"),
                )
                if concentration_law["model"] == "stern_volmer":
                    concentration_law["quenching_coordinate"] = sci(
                        "猝灭特征密度 Nq (m⁻³)",
                        concentration_law.get("quenching_coordinate", 1e25),
                        key=form_widget_key("ui_quenching_density"),
                    )
                    concentration_law["order"] = st.number_input(
                        "猝灭阶数",
                        min_value=0.1,
                        value=float(concentration_law.get("order", 1.0)),
                        key=form_widget_key("ui_quenching_order"),
                    )
                elif concentration_law["model"] == "scale_table":
                    lifetime_response["concentration"] = parse_json(
                        st.text_area(
                            "寿命浓度缩放表 JSON",
                            value=json.dumps(concentration_law, ensure_ascii=False, indent=2),
                            height=110,
                            key=form_widget_key("ui_lifetime_concentration_table"),
                        ),
                        "寿命浓度缩放表",
                    )
            response_cfg["cross_sections"] = parse_json(
                st.text_area(
                    "四组截面响应 JSON（高级数据端口）",
                    value=json.dumps(response_cfg.get("cross_sections", {}), ensure_ascii=False, indent=2),
                    height=150,
                    key=form_widget_key("ui_cross_section_response_json"),
                    help=(
                        "分别配置 pump_absorption/pump_emission/signal_absorption/signal_emission。"
                        "支持 off、linear_scale、scale_table、boltzmann_stark_fraction，"
                        "或 absolute_temperature_wavelength_table 的 λ×T 绝对截面表。"
                    ),
                ),
                "截面温度响应",
            )
            if response_cfg["mode"] == "local_3d" and not strict_coupling:
                st.error(
                    "local_3d 需要“① 模型 → Pump · Seed · 热场收敛”中的 "
                    "thermal_optical_fixed_point；当前组合不会降阶运行，"
                    "以免把统一工作温度误当成局部三维反馈。"
                )

            st.divider()
            st.markdown("**ASE / 横向寄生激光（TPL）**")
            ase_cfg = cfg.setdefault("ase", {})
            ase_options = [
                "off — 关闭",
                "six_ray — 六方向快速筛选",
                "angular_discrete_ordinates — 任意角离散方向 + 寄生本征值",
            ]
            ase_blocker = ase_unavailability_reason(cfg)
            ase_key = form_widget_key("ui_ase_model_locked" if ase_blocker else "ui_ase_model")
            if ase_blocker:
                st.session_state[ase_key] = ase_options[0]
            ase_label = choice_selectbox(
                "ASE 模型",
                ase_options,
                index=0 if ase_blocker else option_index(ase_options, str(ase_cfg.get("model", "off"))),
                key=ase_key,
                disabled=ase_blocker is not None,
                help=ase_blocker,
            )
            if ase_blocker:
                st.info(ase_blocker)
            else:
                ase_cfg["model"] = ase_label.split(" ")[0]
            if not ase_blocker and ase_cfg["model"] in {"six_ray", "angular_discrete_ordinates"}:
                configured_ase_wavelengths = ase_cfg.get("wavelengths_m", []) or [cfg["seed"]["wavelength_m"]]
                wavelength_text = st.text_input(
                    "ASE 波长样本 (nm，逗号分隔)",
                    value=", ".join(f"{float(value) * 1e9:.8g}" for value in configured_ase_wavelengths),
                    key=form_widget_key("ui_ase_wavelengths"),
                )
                try:
                    parsed_wavelengths = [float(value.strip()) * 1e-9 for value in wavelength_text.split(",") if value.strip()]
                    if any(value <= 0.0 or not math.isfinite(value) for value in parsed_wavelengths):
                        raise ValueError("invalid ASE wavelength")
                except ValueError:
                    input_errors.append("ASE 波长列表必须是逗号分隔的数字。")
                    st.error(input_errors[-1])
                else:
                    ase_cfg["wavelengths_m"] = parsed_wavelengths
                configured_ase_weights = ase_cfg.get("spectral_weights", [])
                weight_text = st.text_input(
                    "ASE 光谱权重（留空=等权，逗号分隔）",
                    value=", ".join(f"{float(value):.8g}" for value in configured_ase_weights),
                    key=form_widget_key("ui_ase_spectral_weights"),
                )
                try:
                    parsed_weights = [float(value.strip()) for value in weight_text.split(",") if value.strip()]
                    if parsed_weights and len(parsed_weights) != len(ase_cfg.get("wavelengths_m", [])):
                        input_errors.append("ASE 光谱权重数量必须与波长样本数量相同；留空时自动等权。")
                        st.error(input_errors[-1])
                    elif parsed_weights and (
                        any(value < 0.0 or not math.isfinite(value) for value in parsed_weights)
                        or sum(parsed_weights) <= 0.0
                    ):
                        raise ValueError("invalid ASE spectral weight")
                    else:
                        ase_cfg["spectral_weights"] = parsed_weights
                except ValueError:
                    input_errors.append("ASE 光谱权重必须是逗号分隔的非负数字。")
                    st.error(input_errors[-1])
                global_radiative_lifetime = cfg["energy_levels"].get(
                    "upper_state_radiative_lifetime_s"
                )
                if global_radiative_lifetime is not None:
                    ase_cfg["radiative_lifetime_s"] = None
                    ase_cfg["use_upper_state_lifetime_as_radiative"] = False
                    st.caption(
                        "ASE 自发辐射源复用能级区的显式辐射寿命 "
                        f"{float(global_radiative_lifetime):.6g} s；"
                        "总寿命仍负责全部辐射与非辐射布居衰减。"
                    )
                else:
                    use_upper_lifetime = choice_checkbox(
                        "暂用上能级总寿命作为辐射寿命",
                        value=bool(ase_cfg.get("use_upper_state_lifetime_as_radiative", True)),
                        key=form_widget_key("ui_ase_use_upper_lifetime"),
                    )
                    ase_cfg["use_upper_state_lifetime_as_radiative"] = use_upper_lifetime
                    ase_cfg["radiative_lifetime_s"] = (
                        None
                        if use_upper_lifetime
                        else sci(
                            "辐射寿命 (s)",
                            ase_cfg.get("radiative_lifetime_s") or cfg["energy_levels"]["upper_state_lifetime_s"],
                            key=form_widget_key("ui_ase_radiative_lifetime"),
                        )
                    )
                ase_cfg["modeled_solid_angle_fraction"] = st.number_input(
                    "六方向代表的自发辐射比例",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(ase_cfg.get("modeled_solid_angle_fraction", 1.0)),
                    key=form_widget_key("ui_ase_solid_angle_fraction"),
                )
                reflection_columns = st.columns(3)
                reflectivity = ase_cfg.setdefault("face_reflectivity", {})
                independent_faces = choice_checkbox(
                    "ASE 各端面独立反射率",
                    value=any(reflectivity.get(f"{axis}_min", 0.0) != reflectivity.get(f"{axis}_max", 0.0) for axis in "xyz"),
                    key=form_widget_key("ui_ase_independent_reflectivity"),
                    help="开启后分别设置六个端面；关闭后每轴两端使用相同反射率。",
                )
                for axis, column in zip("xyz", reflection_columns, strict=True):
                    with column:
                        if independent_faces:
                            for side in ("min", "max"):
                                face = f"{axis}_{side}"
                                reflectivity[face] = st.number_input(
                                    lt("{face} 有效反射率").format(face=face),
                                    min_value=0.0, max_value=0.999999,
                                    value=float(reflectivity.get(face, 0.0)),
                                    key=form_widget_key(f"ui_ase_reflectivity_{face}"),
                                )
                            continue
                        pair_default = max(
                            float(reflectivity.get(f"{axis}_min", 0.0)),
                            float(reflectivity.get(f"{axis}_max", 0.0)),
                        )
                        pair = st.number_input(
                            f"{axis} 两端有效反射率",
                            min_value=0.0,
                            max_value=0.999999,
                            value=pair_default,
                            key=form_widget_key(f"ui_ase_reflectivity_{axis}"),
                        )
                        reflectivity[f"{axis}_min"] = pair
                        reflectivity[f"{axis}_max"] = pair
                ase_cfg["maximum_population_iterations"] = int(st.number_input(
                    "ASE–布居最多迭代",
                    min_value=1,
                    max_value=200,
                    value=int(ase_cfg.get("maximum_population_iterations", 12)),
                    key=form_widget_key("ui_ase_population_iterations"),
                ))
                if ase_cfg["model"] == "angular_discrete_ordinates":
                    angular = ase_cfg.setdefault("angular", {})
                    direction_count = int(
                        st.number_input(
                            "ASE 角方向数（偶数）",
                            min_value=8,
                            max_value=512,
                            step=2,
                            value=int(angular.get("direction_count", 24)),
                            key=form_widget_key("ui_ase_angular_direction_count"),
                        )
                    )
                    angular["direction_count"] = direction_count + direction_count % 2
                    st.caption("任意方向输运与反射边界幂迭代会报告主寄生本征值；曲面体素边界仍不解析逐射线 TIR。")
                else:
                    st.caption("六方向模型保留为快速风险筛选；非零反射率形成有效轴向往返并报告 TPL 阈值。")

            st.divider()
            st.markdown("**晶体材料色散与强场效应**")
            sellmeier = cfg["material"]["dispersion"].get("sellmeier")
            if sellmeier:
                sellmeier["enabled"] = choice_checkbox(
                    "使用导入的 Sellmeier 模型",
                    value=bool(sellmeier.get("enabled", True)),
                    key=form_widget_key("ui_sellmeier_enabled"),
                    help="启用时相干传播直接使用 n(λ) 计算材料相位；下面手填的 GDD/TOD/FOD 仍可作为额外修正。",
                )
                st.caption(f"B={sellmeier.get('B', [])}；C(µm²)={sellmeier.get('C_um2', sellmeier.get('C', []))}")
            st.caption(lt("色散系数参考波长：{value} nm。").format(
                value=f"{float(cfg['material']['dispersion'].get('reference_wavelength_m') or cfg['seed']['wavelength_m']) * 1e9:.6g}"
            ))
            cfg["material"]["dispersion"]["gdd_s2_per_m"] = st.number_input(
                "GDD (fs²/m)", value=float(cfg["material"]["dispersion"].get("gdd_s2_per_m", 0.0)) * 1e30, format="%.6g", key=form_widget_key("ui_material_gdd"), disabled=josa_selected
            ) * 1e-30
            cfg["material"]["dispersion"]["tod_s3_per_m"] = st.number_input(
                "TOD (fs³/m)", value=float(cfg["material"]["dispersion"].get("tod_s3_per_m", 0.0)) * 1e45, format="%.6g", key=form_widget_key("ui_material_tod"), disabled=josa_selected
            ) * 1e-45
            cfg["material"]["dispersion"]["fod_s4_per_m"] = st.number_input(
                "FOD (fs⁴/m)", value=float(cfg["material"]["dispersion"].get("fod_s4_per_m", 0.0)) * 1e60, format="%.6g", key=form_widget_key("ui_material_fod"), disabled=josa_selected
            ) * 1e-60
            cfg["material"]["n2_m2_W"] = sci("非线性折射率 n₂ (m²/W)", cfg["material"]["n2_m2_W"], key=form_widget_key("ui_material_n2"))
            nonlinear_options = [
                "auto",
                "off",
                "kerr_parabolic_lens",
                "kerr_time_domain",
            ]
            configured_nonlinear_phase = str(
                cfg["physics"].get("nonlinear_phase_model", "auto")
            )
            nonlinear_key = form_widget_key(
                "ui_nonlinear_model_josa_lock"
                if josa_selected
                else "ui_nonlinear_model"
            )
            if josa_selected:
                st.session_state[nonlinear_key] = "off"
            displayed_nonlinear_phase = choice_selectbox(
                "非线性相位模型", nonlinear_options,
                key=nonlinear_key,
                help=(
                    "kerr_time_domain 只用于 coherent_cpa；kerr_parabolic_lens "
                    "只用于 commercial_app_fast。所有 JOSA 模式都传播强度/光子通量而不是 Kerr 相位。"
                ),
                disabled=josa_selected,
                **({} if nonlinear_key in st.session_state else {
                    "index": option_index(
                        nonlinear_options,
                        configured_nonlinear_phase,
                    )
                }),
            )
            if not josa_selected:
                cfg["physics"]["nonlinear_phase_model"] = (
                    displayed_nonlinear_phase
                )
            coherent_selected = resolve_signal_model(cfg) == "coherent_cpa"
            commercial_app_selected = resolve_signal_model(cfg) == "commercial_app_fast"
            nonlinear_loss_options = (
                ["off", "strong_field_time_domain"]
                if coherent_selected
                else ["off", "strong_field_effective_area"]
                if commercial_app_selected
                else ["off"]
            )
            configured_nonlinear_loss = str(
                cfg["physics"].get("nonlinear_loss_model", "off")
            )
            nonlinear_loss_key = form_widget_key(
                "ui_nonlinear_loss_model_backend_lock"
                if not (coherent_selected or commercial_app_selected)
                else "ui_nonlinear_loss_model"
            )
            if not (coherent_selected or commercial_app_selected):
                st.session_state[nonlinear_loss_key] = "off"
            displayed_nonlinear_loss = choice_selectbox(
                "不可逆强场损耗模型",
                nonlinear_loss_options,
                key=nonlinear_loss_key,
                disabled=not (coherent_selected or commercial_app_selected),
                help=(
                    "coherent_cpa 使用完整采样场；Commercial App 使用 L3.5 Gaussian effective-area "
                    "矩闭合。两者都与纯相位 Kerr 分账。"
                ),
                **({} if nonlinear_loss_key in st.session_state else {
                    "index": option_index(
                        nonlinear_loss_options,
                        configured_nonlinear_loss,
                    )
                }),
            )
            if coherent_selected or commercial_app_selected:
                cfg["physics"]["nonlinear_loss_model"] = (
                    displayed_nonlinear_loss
                )
            strong_field = cfg["material"].setdefault("strong_field", {})
            if displayed_nonlinear_loss in {
                "strong_field_time_domain",
                "strong_field_effective_area",
            }:
                strong_field["model"] = "multiphoton"
                strong_columns = st.columns(2)
                with strong_columns[0]:
                    strong_field["multiphoton_order"] = int(st.number_input(
                        "多光子吸收阶数 K",
                        min_value=2,
                        max_value=12,
                        value=int(strong_field.get("multiphoton_order", 2)),
                        key=form_widget_key("ui_multiphoton_order"),
                    ))
                    strong_field["beta_SI"] = sci(
                        "K 光子吸收系数 β（SI）",
                        strong_field.get("beta_SI", 0.0),
                        key=form_widget_key("ui_multiphoton_beta"),
                        help_text="沿光束真实路径满足 dI/ds=-β I^K；I 是垂直于光束的强度。单位随 K 改变，请直接使用论文给出的 SI 值，程序不会猜 Cr:ZnS 系数。",
                    )
                with strong_columns[1]:
                    strong_field["deposit_fraction_as_heat"] = st.number_input(
                        "不可逆损耗转热比例",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(strong_field.get("deposit_fraction_as_heat", 1.0)),
                        key=form_widget_key("ui_strong_field_heat_fraction"),
                    )
                    carrier_cfg = strong_field.setdefault("free_carrier", {})
                    carrier_options = ["off", "rate_equation"]
                    carrier_cfg["model"] = choice_selectbox(
                        "自由载流子模型",
                        carrier_options,
                        index=option_index(carrier_options, str(carrier_cfg.get("model", "off"))),
                        key=form_widget_key("ui_free_carrier_model"),
                    )
                carrier_cfg = strong_field.setdefault("free_carrier", {})
                if carrier_cfg.get("model", "off") == "rate_equation":
                    carrier_columns = st.columns(4)
                    with carrier_columns[0]:
                        carrier_cfg["lifetime_s"] = sci(
                            "载流子寿命 (s)", carrier_cfg.get("lifetime_s", 1e-9), key=form_widget_key("ui_carrier_lifetime")
                        )
                    with carrier_columns[1]:
                        carrier_cfg["generation_efficiency"] = st.number_input(
                            "载流子生成效率",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(
                                carrier_cfg.get("generation_efficiency", 1.0)
                            ),
                            key=form_widget_key(
                                "ui_carrier_generation_efficiency"
                            ),
                            help=(
                                "每份 K 光子吸收能量产生可存活电子-空穴对的量子效率；"
                                "它直接缩放速率方程的生成项，不等同于不可逆损耗转热比例。"
                            ),
                        )
                    with carrier_columns[2]:
                        carrier_cfg["absorption_cross_section_m2"] = sci(
                            "FCA 截面 (m²)", carrier_cfg.get("absorption_cross_section_m2", 0.0), key=form_widget_key("ui_carrier_absorption")
                        )
                    with carrier_columns[3]:
                        carrier_cfg["refractive_index_m3"] = sci(
                            "载流子 Δn/N (m³)", carrier_cfg.get("refractive_index_m3", 0.0), key=form_widget_key("ui_carrier_refraction")
                        )
                mechanism_options = [
                    "unspecified",
                    "host_interband",
                    "measured_sample_effective",
                    "impurity_sequential",
                    "defect_assisted",
                    "phenomenological_sensitivity",
                ]
                strong_field["mechanism"] = choice_selectbox(
                    "强场损耗物理解读",
                    mechanism_options,
                    index=option_index(
                        mechanism_options,
                        str(strong_field.get("mechanism", "unspecified")),
                    ),
                    key=form_widget_key("ui_strong_field_mechanism"),
                    help=(
                        "这里描述 β 的物理含义，不改变沿真实光束路径的 dI/ds=-βI^K 数值方程。"
                        "phenomenological_sensitivity 只能用于敏感性扫描，不能作为 TPA/MPA 机制证据。"
                    ),
                )
                provenance = strong_field.setdefault("provenance", {})
                provenance_columns = st.columns(2)
                with provenance_columns[0]:
                    provenance_statuses = [
                        "unverified",
                        "provisional_sensitivity",
                        "literature",
                        "sample_measured",
                    ]
                    provenance["status"] = choice_selectbox(
                        "β 证据状态",
                        provenance_statuses,
                        index=option_index(
                            provenance_statuses,
                            str(provenance.get("status", "unverified")),
                        ),
                        key=form_widget_key("ui_strong_field_provenance_status"),
                        help=(
                            "unverified/provisional_sensitivity 不支持定量机制结论；"
                            "literature 必须填写可追溯原始文献，sample_measured 还必须填写样品编号。"
                        ),
                    )
                with provenance_columns[1]:
                    provenance["reference"] = st.text_input(
                        "β 文献/数据引用",
                        value=str(provenance.get("reference") or ""),
                        key=form_widget_key("ui_strong_field_reference"),
                    )
                if strong_field["mechanism"] == "measured_sample_effective":
                    provenance["sample_identifier"] = st.text_input(
                        "β 测量样品编号",
                        value=str(provenance.get("sample_identifier") or ""),
                        key=form_widget_key("ui_strong_field_sample_identifier"),
                    )
                host_interband = strong_field["mechanism"] == "host_interband"
                if host_interband:
                    provenance["bandgap_eV"] = sci(
                        "宿主带隙 (eV)",
                        provenance.get("bandgap_eV") or 1.0,
                        key=form_widget_key("ui_strong_field_bandgap_eV"),
                    )
                record_conditions_key = form_widget_key(
                    "ui_strong_field_record_conditions"
                )
                if host_interband:
                    st.session_state[record_conditions_key] = True
                record_beta_conditions = choice_checkbox(
                    "记录 β 测量条件与相对不确定度",
                    value=(
                        host_interband
                        or provenance.get("reference_wavelength_m") is not None
                        or provenance.get("reference_pulse_duration_s") is not None
                        or provenance.get("relative_uncertainty") is not None
                    ),
                    key=record_conditions_key,
                    disabled=host_interband,
                    help=(
                        "定量 β 应同时记录对应波长、脉宽和相对不确定度。"
                        "host_interband 还必须用参考波长核对最低多光子阶数，因此不能关闭。"
                    ),
                )
                if record_beta_conditions:
                    condition_columns = st.columns(3)
                    with condition_columns[0]:
                        provenance["reference_wavelength_m"] = sci(
                            "β 参考波长 (m)",
                            provenance.get("reference_wavelength_m")
                            or cfg["seed"]["wavelength_m"],
                            key=form_widget_key(
                                "ui_strong_field_reference_wavelength"
                            ),
                        )
                    with condition_columns[1]:
                        provenance["reference_pulse_duration_s"] = sci(
                            "β 参考脉宽 (s)",
                            provenance.get("reference_pulse_duration_s")
                            or cfg["seed"]["pulse_duration_s"],
                            key=form_widget_key(
                                "ui_strong_field_reference_pulse_duration"
                            ),
                        )
                    with condition_columns[2]:
                        provenance["relative_uncertainty"] = st.number_input(
                            "β 相对不确定度",
                            min_value=0.0,
                            value=float(
                                provenance.get("relative_uncertainty") or 0.0
                            ),
                            key=form_widget_key(
                                "ui_strong_field_relative_uncertainty"
                            ),
                            help="例如 0.15 表示相对标准不确定度 15%。",
                        )
                else:
                    provenance["reference_wavelength_m"] = None
                    provenance["reference_pulse_duration_s"] = None
                    provenance["relative_uncertainty"] = None
                st.warning(
                    "β、载流子寿命和 FCA 截面必须来自同一材料/样品、波长和脉宽条件的测量或可追溯文献。"
                    "宿主带隙只能判断带间最低阶数，不能排除杂质、缺陷或顺序吸收；"
                    "未验证的 phenomenological_sensitivity 只可用于敏感性扫描。"
                )

        with numerical_tab:
            _render_workspace_intro(
                "离散、预算与结果可信度",
                "这里调整共享网格和算法预算；专用径向与角向参数在对应模型面板中设置。",
                "物理/光学双网格、采样、内存上限与有效性门控",
            )
            st.caption(
                "网格摘要："
                f"{int(cfg['grid']['nx'])}×{int(cfg['grid']['ny'])}×{int(cfg['grid']['nz'])} · "
                f"x/y {cfg['grid'].get('spacing_xy', 'uniform')} · "
                f"独立光学网格 {'on' if cfg['grid'].get('optical', {}).get('enabled', False) else 'off'}。"
            )
        with numerical_tab, st.expander(
            "网格、采样与内存",
            expanded=expand_long_sections,
        ):
            cfg["grid"]["size_x_m"] = sci("x 窗口 (m)", cfg["grid"]["size_x_m"], key=form_widget_key("ui_grid_x"))
            cfg["grid"]["size_y_m"] = sci("y 窗口 (m)", cfg["grid"]["size_y_m"], key=form_widget_key("ui_grid_y"))
            physical_center_columns = st.columns(2)
            with physical_center_columns[0]:
                cfg["grid"]["center_x_m"] = sci(
                    "物理网格中心 x (m)",
                    cfg["grid"].get("center_x_m", 0.0),
                    key=form_widget_key("ui_grid_center_x"),
                    help_text="定义固定实验室坐标中的物理计算域中心；不会自动移动晶体、Pump 或 Seed。",
                )
            with physical_center_columns[1]:
                cfg["grid"]["center_y_m"] = sci(
                    "物理网格中心 y (m)",
                    cfg["grid"].get("center_y_m", 0.0),
                    key=form_widget_key("ui_grid_center_y"),
                    help_text="定义固定实验室坐标中的物理计算域中心；不会自动移动晶体、Pump 或 Seed。",
                )
            cfg["grid"]["nx"] = int(st.number_input("Nx", 3, None, int(cfg["grid"]["nx"]), step=8, key=form_widget_key("ui_grid_nx")))
            cfg["grid"]["ny"] = int(st.number_input("Ny", 3, None, int(cfg["grid"]["ny"]), step=8, key=form_widget_key("ui_grid_ny")))
            cfg["grid"]["nz"] = int(st.number_input("Nz", 3, None, int(cfg["grid"]["nz"]), step=4, key=form_widget_key("ui_grid_nz")))
            if cfg["simulation"]["signal_model"] in {"commercial_app_fast", "coherent_cpa"}:
                render_crystal_substep_controls(cfg)
            optical_grid_cfg = cfg["grid"].setdefault("optical", {})
            optical_grid_was_enabled = bool(optical_grid_cfg.get("enabled", False))
            optical_grid_cfg["enabled"] = choice_toggle(
                "独立均匀光学网格（双网格耦合）",
                value=optical_grid_was_enabled,
                key=form_widget_key("ui_optical_grid_enabled"),
                help="Pump/Seed 的 ABCD/FFT 在局部均匀网格运行；布居、热和应力保留在完整晶体网格。",
            )
            if optical_grid_cfg["enabled"] and not optical_grid_was_enabled:
                # Fidelity presets may shrink the physical grid while the
                # dormant local-grid defaults retain their original 1 mm
                # window.  Enabling the feature must start from an operational
                # child grid, not immediately lock the entire form.
                for axis in ("x", "y"):
                    physical_size = float(cfg["grid"][f"size_{axis}_m"])
                    configured_size = float(
                        optical_grid_cfg.get(f"size_{axis}_m", physical_size)
                    )
                    local_size = min(configured_size, physical_size)
                    margin = max(0.5 * (physical_size - local_size), 0.0)
                    host_center = float(
                        cfg["grid"].get(f"center_{axis}_m", 0.0)
                    )
                    local_center = float(
                        np.clip(
                            float(optical_grid_cfg.get(f"center_{axis}_m", 0.0)),
                            host_center - margin,
                            host_center + margin,
                        )
                    )
                    optical_grid_cfg[f"size_{axis}_m"] = local_size
                    optical_grid_cfg[f"center_{axis}_m"] = local_center
                    st.session_state[
                        form_widget_key(f"ui_optical_grid_{axis}")
                    ] = local_size
                    st.session_state[
                        form_widget_key(f"ui_optical_grid_center_{axis}")
                    ] = local_center
            if optical_grid_cfg["enabled"]:
                optical_window_columns = st.columns(2)
                with optical_window_columns[0]:
                    optical_grid_cfg["size_x_m"] = sci(
                        "光学 x 窗口 (m)",
                        optical_grid_cfg.get("size_x_m", cfg["grid"]["size_x_m"]),
                        key=form_widget_key("ui_optical_grid_x"),
                    )
                    optical_grid_cfg["center_x_m"] = sci(
                        "光学窗口中心 x (m)",
                        optical_grid_cfg.get("center_x_m", 0.0),
                        key=form_widget_key("ui_optical_grid_center_x"),
                    )
                    optical_grid_cfg["nx"] = int(st.number_input(
                        "光学 Nx", 16, 1024, int(optical_grid_cfg.get("nx", 128)), step=16,
                        key=form_widget_key("ui_optical_grid_nx"),
                    ))
                with optical_window_columns[1]:
                    optical_grid_cfg["size_y_m"] = sci(
                        "光学 y 窗口 (m)",
                        optical_grid_cfg.get("size_y_m", cfg["grid"]["size_y_m"]),
                        key=form_widget_key("ui_optical_grid_y"),
                    )
                    optical_grid_cfg["center_y_m"] = sci(
                        "光学窗口中心 y (m)",
                        optical_grid_cfg.get("center_y_m", 0.0),
                        key=form_widget_key("ui_optical_grid_center_y"),
                    )
                    optical_grid_cfg["ny"] = int(st.number_input(
                        "光学 Ny", 16, 1024, int(optical_grid_cfg.get("ny", 128)), step=16,
                        key=form_widget_key("ui_optical_grid_ny"),
                    ))
                optical_dx = float(optical_grid_cfg["size_x_m"]) / int(optical_grid_cfg["nx"])
                optical_dy = float(optical_grid_cfg["size_y_m"]) / int(optical_grid_cfg["ny"])
                st.caption(
                    f"光学采样：Δx={optical_dx * 1e6:.4g} µm，Δy={optical_dy * 1e6:.4g} µm；"
                    "光学窗口必须完整位于晶体物理网格内。"
                )
            pump_optical_cfg = cfg["grid"].setdefault("pump_optical", {})
            pump_grid_modes = {
                "与 Seed/BPM 共用": "shared",
                "使用物理网格": "physical",
                "独立均匀网格": "custom",
            }
            current_pump_grid_mode = str(
                pump_optical_cfg.get("mode", "shared")
            )
            if current_pump_grid_mode not in pump_grid_modes.values():
                current_pump_grid_mode = "shared"
            pump_grid_label = choice_selectbox(
                "Pump 横向计算网格",
                list(pump_grid_modes),
                index=list(pump_grid_modes.values()).index(current_pump_grid_mode),
                key=form_widget_key("ui_pump_optical_grid_mode"),
                help=(
                    "宽 Pump 和窄 Seed/BPM 不应被迫使用同一个窗口。旧配置默认共用；"
                    "独立网格通过有限体积映射与晶体网格耦合。"
                ),
            )
            pump_optical_cfg["mode"] = pump_grid_modes[pump_grid_label]
            if (
                pump_optical_cfg["mode"] == "custom"
                and current_pump_grid_mode != "custom"
            ):
                for axis in ("x", "y"):
                    physical_size = float(cfg["grid"][f"size_{axis}_m"])
                    configured_size = float(
                        pump_optical_cfg.get(f"size_{axis}_m", physical_size)
                    )
                    local_size = min(configured_size, physical_size)
                    margin = max(0.5 * (physical_size - local_size), 0.0)
                    host_center = float(
                        cfg["grid"].get(f"center_{axis}_m", 0.0)
                    )
                    local_center = float(
                        np.clip(
                            float(pump_optical_cfg.get(f"center_{axis}_m", 0.0)),
                            host_center - margin,
                            host_center + margin,
                        )
                    )
                    pump_optical_cfg[f"size_{axis}_m"] = local_size
                    pump_optical_cfg[f"center_{axis}_m"] = local_center
                    st.session_state[
                        form_widget_key(f"ui_pump_optical_grid_{axis}")
                    ] = local_size
                    st.session_state[
                        form_widget_key(f"ui_pump_optical_grid_center_{axis}")
                    ] = local_center
            if pump_optical_cfg["mode"] == "custom":
                pump_grid_columns = st.columns(2)
                with pump_grid_columns[0]:
                    pump_optical_cfg["size_x_m"] = sci(
                        "Pump 网格 x 窗口 (m)",
                        pump_optical_cfg.get("size_x_m", cfg["grid"]["size_x_m"]),
                        key=form_widget_key("ui_pump_optical_grid_x"),
                    )
                    pump_optical_cfg["center_x_m"] = sci(
                        "Pump 网格中心 x (m)",
                        pump_optical_cfg.get("center_x_m", 0.0),
                        key=form_widget_key("ui_pump_optical_grid_center_x"),
                    )
                    pump_optical_cfg["nx"] = int(st.number_input(
                        "Pump Nx", 8, 1024,
                        int(pump_optical_cfg.get("nx", 128)), step=8,
                        key=form_widget_key("ui_pump_optical_grid_nx"),
                    ))
                with pump_grid_columns[1]:
                    pump_optical_cfg["size_y_m"] = sci(
                        "Pump 网格 y 窗口 (m)",
                        pump_optical_cfg.get("size_y_m", cfg["grid"]["size_y_m"]),
                        key=form_widget_key("ui_pump_optical_grid_y"),
                    )
                    pump_optical_cfg["center_y_m"] = sci(
                        "Pump 网格中心 y (m)",
                        pump_optical_cfg.get("center_y_m", 0.0),
                        key=form_widget_key("ui_pump_optical_grid_center_y"),
                    )
                    pump_optical_cfg["ny"] = int(st.number_input(
                        "Pump Ny", 8, 1024,
                        int(pump_optical_cfg.get("ny", 128)), step=8,
                        key=form_widget_key("ui_pump_optical_grid_ny"),
                    ))
            spacing_labels = ["均匀网格", "Gaussian 中心加密", "指定区域加密"]
            spacing_modes = ["uniform", "center_stretched", "refined_region"]
            spacing_label = choice_selectbox(
                "x/y 网格分布", spacing_labels,
                index=spacing_modes.index(cfg["grid"].get("spacing_xy", "uniform")),
                key=form_widget_key("ui_grid_spacing_xy"),
            )
            cfg["grid"]["spacing_xy"] = spacing_modes[spacing_labels.index(spacing_label)]
            if cfg["grid"]["spacing_xy"] in {"center_stretched", "refined_region"}:
                st.caption(
                    "加密中心逐轴设置；未独立指定的轴跟随对应物理网格中心。只改变单元分布，"
                    "不会移动物理边界、晶体、Pump、Seed 或独立光学窗口。"
                )
                refinement_center_columns = st.columns(2)
                with refinement_center_columns[0]:
                    independent_refinement_x = choice_toggle(
                        "独立设置 x 加密中心",
                        value=(
                            cfg["grid"].get("refinement_center_x_m") is not None
                        ),
                        key=form_widget_key(
                            "ui_grid_independent_refinement_center_x"
                        ),
                    )
                    if independent_refinement_x:
                        cfg["grid"]["refinement_center_x_m"] = sci(
                            "物理网格加密中心 x (m)",
                            cfg["grid"].get("refinement_center_x_m")
                            if cfg["grid"].get("refinement_center_x_m")
                            is not None
                            else cfg["grid"].get("center_x_m", 0.0),
                            key=form_widget_key("ui_grid_refinement_center_x"),
                            help_text="必须严格位于物理 x 窗口内部；最小单元会集中在该坐标附近。",
                        )
                    else:
                        cfg["grid"]["refinement_center_x_m"] = None
                with refinement_center_columns[1]:
                    independent_refinement_y = choice_toggle(
                        "独立设置 y 加密中心",
                        value=(
                            cfg["grid"].get("refinement_center_y_m") is not None
                        ),
                        key=form_widget_key(
                            "ui_grid_independent_refinement_center_y"
                        ),
                    )
                    if independent_refinement_y:
                        cfg["grid"]["refinement_center_y_m"] = sci(
                            "物理网格加密中心 y (m)",
                            cfg["grid"].get("refinement_center_y_m")
                            if cfg["grid"].get("refinement_center_y_m")
                            is not None
                            else cfg["grid"].get("center_y_m", 0.0),
                            key=form_widget_key("ui_grid_refinement_center_y"),
                            help_text="必须严格位于物理 y 窗口内部；最小单元会集中在该坐标附近。",
                        )
                    else:
                        cfg["grid"]["refinement_center_y_m"] = None
                if cfg["grid"]["spacing_xy"] == "refined_region":
                    region = cfg["grid"].setdefault("refined_region", {})
                    st.caption(
                        "覆盖宽度是完整宽度，以加密中心为中点；区域内等间距，区域外逐渐变粗。"
                        "区域内格子计入总 Nx/Ny，每轴至少保留四个外部格子。原来的加密比例在此模式下不生效。"
                    )
                    region_columns = st.columns(2)
                    for axis, column in zip(("x", "y"), region_columns):
                        with column:
                            region[f"size_{axis}_m"] = sci(
                                f"{axis} 细网格覆盖宽度 (m)",
                                region.get(f"size_{axis}_m", 200e-6),
                                key=form_widget_key(f"ui_grid_region_size_{axis}"),
                                help_text="完整细网格区域必须位于物理边界内。",
                            )
                            region[f"n{axis}"] = int(st.number_input(
                                f"{axis} 细网格区域格子数", min_value=2,
                                value=int(region.get(f"n{axis}", 24)), step=2,
                                key=form_widget_key(f"ui_grid_region_n{axis}"),
                            ))
                    try:
                        region_preview = Grid3D.from_config(cfg["grid"])
                        st.caption(
                            "区域内间距 Δx / Δy (μm): "
                            f"{1e6 * region['size_x_m'] / region['nx']:.4g} / "
                            f"{1e6 * region['size_y_m'] / region['ny']:.4g}"
                        )
                        growth = max(
                            max(float(np.max(w[1:] / w[:-1])),
                                float(np.max(w[:-1] / w[1:])))
                            for w in (region_preview.x_widths, region_preview.y_widths)
                        )
                        st.caption("相邻格子最大尺寸比: " + f"{growth:.3f}")
                    except ValueError as exc:
                        st.error("细网格区域设置无效：" + str(exc))
                else:
                    stretch_columns = st.columns(2)
                    with stretch_columns[0]:
                        cfg["grid"]["stretch_x"] = st.number_input(
                            "x sinh 映射边缘/加密点局部尺度比", 1.0, 1000.0,
                            float(cfg["grid"].get("stretch_x", 4.0)), step=0.5,
                            help="这是连续 sinh 映射的边缘/加密点局部导数比；有限 Nx 下实际单元宽度比会略小。",
                            key=form_widget_key("ui_grid_stretch_x")
                        )
                    with stretch_columns[1]:
                        cfg["grid"]["stretch_y"] = st.number_input(
                            "y sinh 映射边缘/加密点局部尺度比", 1.0, 1000.0,
                            float(cfg["grid"].get("stretch_y", 4.0)), step=0.5,
                            help="这是连续 sinh 映射的边缘/加密点局部导数比；有限 Ny 下实际单元宽度比会略小。",
                            key=form_widget_key("ui_grid_stretch_y")
                        )
                pump_has_uniform_grid = (
                    pump_optical_cfg["mode"] == "custom"
                    or (
                        pump_optical_cfg["mode"] == "shared"
                        and optical_grid_cfg["enabled"]
                    )
                )
                if optical_grid_cfg["enabled"] or pump_has_uniform_grid:
                    st.success(
                        "局部加密用于布居/热/应力物理网格；独立均匀光学网格用于相应的 Pump 或 Seed/BPM 传播，并通过有限体积守恒映射双向耦合。"
                    )
                else:
                    st.info(
                        "局部加密会把更多 x/y 单元放到指定加密中心。未启用独立光学网格时，"
                        "FFT/BPM 要求均匀网格；Seed 必须为 off，Pump 可用 "
                        "prescribed_caustic、super_gaussian_abcd 或 stochastic_raytrace。"
                    )
                resolved_for_grid = resolve_physics(cfg)
                if (
                    (
                        resolved_for_grid["optical_propagation"] != "off"
                        and not optical_grid_cfg["enabled"]
                    )
                    or (
                        resolved_for_grid["pump_propagation"]
                        not in {
                            "prescribed_caustic",
                            "super_gaussian_abcd",
                            "stochastic_raytrace",
                        }
                        and not pump_has_uniform_grid
                    )
                ):
                    full_josa_selected = _full_josa_selected(cfg)
                    if full_josa_selected:
                        st.error("full_3d_pc 固定使用 FFT；局部加密物理网格必须配一个独立均匀光学网格。")
                        if st.button("为完整 3D 启用相容双网格", width="stretch", key="enable_full_josa_dual_grid"):
                            optical_grid_cfg.update(
                                {
                                    "enabled": True,
                                    "size_x_m": float(cfg["grid"]["size_x_m"]),
                                    "size_y_m": float(cfg["grid"]["size_y_m"]),
                                    "center_x_m": float(cfg["grid"].get("center_x_m", 0.0)),
                                    "center_y_m": float(cfg["grid"].get("center_y_m", 0.0)),
                                    "nx": max(16, int(cfg["grid"]["nx"])),
                                    "ny": max(16, int(cfg["grid"]["ny"])),
                                }
                            )
                            pump_optical_cfg["mode"] = "shared"
                            install_form_config(cfg, update_json_editor=False)
                            st.rerun()
                    elif st.button("切换到局部加密相容传播模式", width="stretch", key="enable_stretched_grid_modes"):
                        cfg["physics"]["optical_propagation"] = "off"
                        cfg["physics"]["pump_propagation"] = "prescribed_caustic"
                        install_form_config(cfg, update_json_editor=False)
                        st.rerun()

        if solver_summary_placeholder is not None:
            with solver_summary_placeholder.container():
                _render_effective_solver_summary(
                    cfg,
                    full_josa=_full_josa_selected(cfg),
                )
        with numerical_tab:
            validation_options = [
                "exploratory — 保留结果并醒目标出无效点与诊断警告",
                "strict — 无效结果禁止正式导出，遍历仍保留诊断记录",
            ]
            validation_label = choice_selectbox(
                "结果有效性模式",
                validation_options,
                index=option_index(
                    validation_options,
                    str(cfg["simulation"].get("validation_mode", "exploratory")),
                ),
                key=form_widget_key("ui_validation_mode"),
                help=(
                    "阻断性检查覆盖 Pump/Seed 账本、混合 Pump/Seed 晶体入口窗口捕获、布居与投影修正、收敛、ASE clamp、"
                    "真实周期 FFT 边界和材料表范围；仅影响记录平面/M²解释的质量问题会"
                    "单列为警告，不会冒充物理失效。探索模式保留并显示无效诊断点，"
                    "严格模式还会阻止其正式导出。"
                ),
            )
            cfg["simulation"]["validation_mode"] = validation_label.split(" ")[0]
            strict_settings = cfg["simulation"].setdefault(
                "strict_validation",
                copy.deepcopy(DEFAULTS["simulation"]["strict_validation"]),
            )
            strict_settings["require_named_external_losses"] = choice_checkbox(
                "严格模式要求所有外部 Seed 损耗具名闭合",
                value=bool(strict_settings.get("require_named_external_losses", False)),
                key=form_widget_key("ui_require_named_external_losses"),
                help=(
                    "开启后，界面、程间透过率和前后光路若留下任何未命名能量差，"
                    "严格验证会判定无效；关闭时仅阻止无法解释的额外生能。"
                ),
            )
            strict_settings["require_experimentally_calibrated_material"] = (
                choice_checkbox(
                    "正式结果要求样品级材料实验标定证据",
                    value=bool(
                        strict_settings.get(
                            "require_experimentally_calibrated_material", False
                        )
                    ),
                    key=form_widget_key(
                        "ui_require_experimentally_calibrated_material"
                    ),
                    help=(
                        "开启后，材料快照必须标为 experimentally_calibrated，填写样品标识、"
                        "64 位数据集 SHA-256，并覆盖 required_parameters；否则结果固定标为非定量。"
                    ),
                )
            )
            st.caption(
                "Cartesian 输运场的内存主要随 Nt × Nλ × Nx × Ny（Spectral Time）或 "
                "Nλ × Nx × Ny（Coherent）增长。径向场以 Nr × Nθ 替代横向 Nx × Ny；"
                "外部光路和三维物理场仍占内存。相干软窗增益增加运行时间，但不常驻独立的 Nt × Nλ 状态表。"
            )
        with recording_tab:
            _render_workspace_intro(
                "结果记录",
                "控制历史与纵向诊断的数据密度，不改变传播物理。",
                "分级记录、功能联动与文件输出",
            )
        with recording_tab, st.expander("记录层级与内容", expanded=True):
            _render_result_recording_controls(cfg)
        prior_dependency_ownership = copy.deepcopy(
            list(st.session_state.get("compatibility_ownership", []))
        )
        dependency_input_cfg, released_dependencies = (
            release_ui_dependency_ownership(
                cfg,
                prior_dependency_ownership,
            )
        )
        reconciled_cfg, compatibility_adjustments = reconcile_ui_dependencies(
            dependency_input_cfg
        )
        rows = [item.as_row() for item in compatibility_adjustments]
        ownership = [
            item.as_ownership() for item in compatibility_adjustments
        ]
        config_changed = json.dumps(
            cfg, sort_keys=True, default=str
        ) != json.dumps(reconciled_cfg, sort_keys=True, default=str)
        cfg = reconciled_cfg
        if config_changed:
            try:
                install_form_config(cfg, update_json_editor=False)
            except Exception:
                # Keep the deterministic fixes in the returned config so the
                # main-pane validation can localize any remaining numeric or
                # schema problem. No session/widget state has changed because
                # install_form_config validates before mutating it.
                pass
            else:
                generation = int(
                    st.session_state.get("form_widget_generation", 0)
                )
                st.session_state["compatibility_ownership"] = ownership
                st.session_state["compatibility_adjustments"] = {
                    "generation": generation,
                    "rows": rows,
                }
                if released_dependencies and not compatibility_adjustments:
                    st.session_state["compatibility_release_event"] = {
                        "generation": generation,
                        "rows": [
                            item.as_row() for item in released_dependencies
                        ],
                    }
                elif compatibility_adjustments:
                    st.session_state.pop(
                        "compatibility_release_event", None
                    )
                st.rerun()
        generation = int(st.session_state.get("form_widget_generation", 0))
        st.session_state["compatibility_ownership"] = ownership
        st.session_state["compatibility_adjustments"] = {
            "generation": generation,
            "rows": rows,
        }
        st.session_state.pop("run_after_compatibility", None)
        run_readiness_error: Exception | None = (
            ValueError(" ".join(input_errors)) if input_errors else None
        )
        try:
            reconcile_material_field_provenance(cfg)
            validate_config(cfg)
        except Exception as exc:
            run_readiness_error = exc
        with run_placeholder.container():
            solve_job_active = bool(
                st.session_state.get("solve_job_active", False)
            )
            sweep_job_active = bool(
                st.session_state.get("sweep_job_active", False)
            )
            run = st.button(
                "运行当前配置",
                type="primary",
                width="stretch",
                key=amplifier_view_widget_key("run_form"),
                disabled=(
                    run_readiness_error is not None
                    or solve_job_active
                    or sweep_job_active
                ),
                help=(
                    "已有后台单点求解；请在主页面明确暂停、继续或取消。当前表单仍可编辑，修改只用于下一次运行。"
                    if solve_job_active
                    else "已有后台参数遍历；请先继续完成或取消遍历。当前表单修改只用于下一次运行。"
                    if sweep_job_active
                    else
                    "执行当前配置，并在右侧显示真实模块、固定点迭代、残差、速度和粗略剩余时间。"
                    if run_readiness_error is None
                    else "当前配置未通过运行前体检；右侧会显示问题和具体界面位置。"
                ),
            )
            if run_readiness_error is not None:
                st.error(
                    "运行已锁定："
                    f"{type(run_readiness_error).__name__}: {run_readiness_error}"
                )
        with active_model_placeholder.container():
            _render_active_module_card(cfg)
    reconcile_material_field_provenance(cfg)
    st.session_state["form_base_config"] = copy.deepcopy(cfg)
    return cfg, run and not input_errors, input_errors


def sweep_sidebar_controls(
    cfg: dict[str, Any],
    *,
    blocked_reason: str | None = None,
) -> dict[str, Any] | None:
    with st.sidebar.expander("参数遍历（可选工具）"):
        labels = [parameter.label for parameter in SWEEP_PARAMETERS]
        selected_label = choice_selectbox("遍历参数", labels, key="sweep_parameter")
        parameter = SWEEP_PARAMETERS[labels.index(selected_label)]
        current = get_sweep_parameter_value(cfg, parameter)
        if (
            parameter.path == "pump.wavelength_m"
            and cfg["pump"].get("spectrum", {}).get("type") == "file"
        ):
            st.warning(
                "当前 Pump 使用绝对波长文件；遍历中心波长不会平移文件中的采样点。"
                "若要做中心波长扫描，请使用 monochromatic/Gaussian/Lorentzian/Super-Gaussian，"
                "或为每个工况准备相应的绝对波长文件。"
            )
        if (
            parameter.path == "seed.wavelength_m"
            and cfg["seed"].get("spectrum", {}).get("type") == "file"
        ):
            st.warning(
                "当前 Seed 使用绝对波长文件；遍历中心波长不会平移文件中的采样点。"
                "若要扫描整条 Seed 光谱的位置，请使用解析光谱，或为每个工况准备相应的绝对波长文件。"
            )
        current_value = float(current) if current is not None else 0.0
        if current_value > 0.0:
            default_start, default_stop = 0.5 * current_value, 1.5 * current_value
        else:
            default_start = parameter.fallback_start if parameter.fallback_start is not None else 0.0
            default_stop = parameter.fallback_stop if parameter.fallback_stop is not None else 1.0
        if parameter.integer:
            default_start = parameter.fallback_start if parameter.fallback_start is not None else max(1.0, default_start)
            default_stop = parameter.fallback_stop if parameter.fallback_stop is not None else max(default_start, default_stop)
        st.caption(f"当前值：{current_value * parameter.display_scale:.6g} {parameter.unit}")
        start = float(
            st.number_input(
                f"起点 ({parameter.unit or 'SI'})",
                value=float(default_start * parameter.display_scale),
                format="%.6e",
                key=f"sweep_start_{parameter.path}",
            )
        ) / parameter.display_scale
        stop = float(
            st.number_input(
                f"终点 ({parameter.unit or 'SI'})",
                value=float(default_stop * parameter.display_scale),
                format="%.6e",
                key=f"sweep_stop_{parameter.path}",
            )
        ) / parameter.display_scale
        first, second = st.columns(2)
        with first:
            count = int(st.number_input("点数", 2, 31, 7, key="sweep_count"))
        with second:
            spacing_label = choice_selectbox("间隔", ["linear", "log"], key="sweep_spacing")
        two_dimensional = choice_checkbox(
            "二维参数网格（两个输入同时遍历）",
            value=False,
            key="sweep_two_dimensional",
            help="生成两个参数的笛卡尔网格；每个格点仍是一次独立完整模拟。",
        )
        secondary_parameter: SweepParameter | None = None
        secondary_start = secondary_stop = 0.0
        secondary_count = 1
        secondary_spacing = "linear"
        if two_dimensional:
            secondary_options = [item for item in SWEEP_PARAMETERS if item.path != parameter.path]
            secondary_labels = [item.label for item in secondary_options]
            secondary_label = choice_selectbox(
                "第二遍历参数",
                secondary_labels,
                key="sweep_secondary_parameter",
            )
            secondary_parameter = secondary_options[
                secondary_labels.index(secondary_label)
            ]
            secondary_current = float(
                get_sweep_parameter_value(cfg, secondary_parameter)
            )
            if secondary_current > 0.0:
                secondary_default_start = 0.5 * secondary_current
                secondary_default_stop = 1.5 * secondary_current
            else:
                secondary_default_start = (
                    secondary_parameter.fallback_start
                    if secondary_parameter.fallback_start is not None
                    else 0.0
                )
                secondary_default_stop = (
                    secondary_parameter.fallback_stop
                    if secondary_parameter.fallback_stop is not None
                    else 1.0
                )
            st.caption(
                f"第二参数当前值：{secondary_current * secondary_parameter.display_scale:.6g} "
                f"{secondary_parameter.unit}"
            )
            secondary_start = float(
                st.number_input(
                    f"第二参数起点 ({secondary_parameter.unit or 'SI'})",
                    value=float(
                        secondary_default_start * secondary_parameter.display_scale
                    ),
                    format="%.6e",
                    key=f"sweep_secondary_start_{secondary_parameter.path}",
                )
            ) / secondary_parameter.display_scale
            secondary_stop = float(
                st.number_input(
                    f"第二参数终点 ({secondary_parameter.unit or 'SI'})",
                    value=float(
                        secondary_default_stop * secondary_parameter.display_scale
                    ),
                    format="%.6e",
                    key=f"sweep_secondary_stop_{secondary_parameter.path}",
                )
            ) / secondary_parameter.display_scale
            secondary_columns = st.columns(2)
            with secondary_columns[0]:
                secondary_count = int(
                    st.number_input(
                        "第二参数点数",
                        2,
                        21,
                        5,
                        key="sweep_secondary_count",
                    )
                )
            with secondary_columns[1]:
                secondary_spacing = choice_selectbox(
                    "第二参数间隔",
                    ["linear", "log"],
                    key="sweep_secondary_spacing",
                )
        metric_labels = {
            definition.label: key
            for key, definition in SWEEP_METRICS.items()
            if key not in LEGACY_SWEEP_METRIC_KEYS
        }
        metric_options = list(metric_labels)
        legacy_defaults = [
            "输出种子能量",
            "总能量增益",
            "吸收泵浦功率",
            "泵后主信号上能级占比（平均）",
            "总热负载",
            "种子净提取能量",
            "晶体最大温升",
        ]
        migration_key = "sweep_metrics_default_all_v1"
        if not st.session_state.get(migration_key, False):
            existing_metrics = st.session_state.get("sweep_metrics")
            if (
                existing_metrics is not None
                and list(existing_metrics) == legacy_defaults
            ):
                st.session_state["sweep_metrics"] = metric_options
            st.session_state[migration_key] = True
        selected_metrics = choice_multiselect(
            "输出量（可多选）",
            metric_options,
            default=metric_options,
            key="sweep_metrics",
        )
        if parameter.path == "simulation.multipass.pass_count" and not bool(
            cfg["simulation"].get("multipass", {}).get("enabled", False)
        ):
            st.info("当前是单程；请先启用多程，再遍历晶体命中次数。")
        for candidate in (parameter, secondary_parameter):
            if candidate is not None:
                reason = sweep_parameter_inactive_reason(cfg, candidate)
                if reason is not None:
                    blocked_reason = blocked_reason or f"{candidate.label}：{reason}"
        expected_runs = count * secondary_count if two_dimensional else count
        pump_continuation_serial = supports_sweep_continuation(
            *([parameter, secondary_parameter] if two_dimensional else [parameter])
        )
        point_workers_key = "sweep_point_workers"
        point_workers_forced_key = "sweep_point_workers_forced_by_pump"
        point_workers_previous_key = "sweep_point_workers_before_pump"
        if pump_continuation_serial:
            if not bool(st.session_state.get(point_workers_forced_key, False)):
                st.session_state[point_workers_previous_key] = int(
                    st.session_state.get(point_workers_key, 3)
                )
            st.session_state[point_workers_key] = 1
            st.session_state[point_workers_forced_key] = True
        elif bool(st.session_state.pop(point_workers_forced_key, False)):
            st.session_state[point_workers_key] = int(
                st.session_state.pop(point_workers_previous_key, 3)
            )
        elif point_workers_key not in st.session_state:
            st.session_state[point_workers_key] = 3
        with st.container(border=True):
            resource_columns = st.columns([1, 1.45])
            with resource_columns[0]:
                point_workers = int(
                    st.number_input(
                        "同时计算点数",
                        min_value=1,
                        max_value=3,
                        step=1,
                        key=point_workers_key,
                        disabled=pump_continuation_serial,
                        help=(
                            "1 个点保留当前完整实时阶段图；2–3 个点使用独立进程，"
                            "实时区只显示每个点的阶段与百分比。"
                        ),
                    )
                )
            with resource_columns[1]:
                if pump_continuation_serial:
                    st.info(
                        "当前遍历使用 Warm Start，自动串行继承前一点的布居、温度和 Δn；"
                        "首点使用配置的初始化。每一点仍按原容差重新求解。"
                    )
                elif point_workers == 1:
                    st.caption(
                        "串行模式：显示当前点的完整实时诊断图和阶段明细。"
                    )
                else:
                    st.caption(
                        f"并行模式：最多同时运行 {point_workers} 个独立点；"
                        "每点保持当前配置的内部并行，不额外限制线程。实时区只保留状态和百分比，"
                        "完整三维结果仍逐点落盘。"
                    )
        if two_dimensional and pump_continuation_serial:
            st.info("二维 Warm Start 按蛇形顺序逐行计算，换行继承相邻点；图表仍按原坐标排列。失败点不作为新的继承源。")
        elif two_dimensional:
            st.caption("当前二维组合含不支持状态继承的参数，各点使用配置的初始化。")
        st.caption(
            f"预计执行 {expected_runs} 次完整模拟，分为约 "
            f"{math.ceil(expected_runs / max(point_workers, 1))} 组；"
            "每个失败点会被单独记录，不会使整组遍历丢失。"
        )
        requested = st.button(
            "运行参数遍历",
            width="stretch",
            key="run_sweep",
            disabled=blocked_reason is not None,
            help=(
                "每个遍历点使用与单点运行相同的配置校验和物理有效性门。"
                if blocked_reason is None
                else f"运行已锁定：{blocked_reason}"
            ),
        )
        if blocked_reason is not None:
            st.error(f"参数遍历已锁定：{blocked_reason}")
    if not requested or blocked_reason is not None:
        return None
    if not selected_metrics:
        raise ValueError("参数遍历至少选择一个输出量")
    values = make_sweep_values(start, stop, count, spacing_label, parameter.integer)
    request = {
        "parameter": parameter,
        "values": values,
        "metric_keys": [metric_labels[label] for label in selected_metrics],
        "spacing": spacing_label,
        "dimension": 2 if two_dimensional else 1,
        "point_workers": point_workers,
    }
    if two_dimensional and secondary_parameter is not None:
        request.update(
            {
                "secondary_parameter": secondary_parameter,
                "secondary_values": make_sweep_values(
                    secondary_start,
                    secondary_stop,
                    secondary_count,
                    secondary_spacing,
                    secondary_parameter.integer,
                ),
                "secondary_spacing": secondary_spacing,
            }
        )
    return request


def warm_start_sidebar_controls(
    *,
    blocked_reason: str | None = None,
) -> dict[str, Any] | None:
    """Select one full archived operating point as a numerical initial guess."""

    with st.sidebar.expander("本地结果点热启动（可选）"):
        archive_path = st.text_input(
            "完整 .ampres.zip 路径",
            value=str(st.session_state.get("single_warm_start_archive_path", "")),
            key="single_warm_start_archive_path",
            help=(
                "留空即关闭。直接读取本机完整结果包，不会打开浏览器文件选择器；"
                "只复用布居、温度、热致折射率和端面形变作为数值初值，"
                "当前配置的全部方程仍会重新迭代到收敛。"
            ),
        ).strip()
        point_text = st.text_input(
            "结果点序号（从 1 开始）",
            value=str(
                st.session_state.get("single_warm_start_point_number_text", "1")
            ),
            key="single_warm_start_point_number_text",
            help="填写结果包中的一基序号，例如第 6 点填写 6。",
        )
        try:
            point_number = int(point_text)
        except ValueError:
            st.warning("结果点序号必须是正整数。")
            return None
        if point_number < 1:
            st.warning("结果点序号必须从 1 开始。")
            return None
        st.caption(
            "warm start 不复用旧输出；新开启的 FCA 载流子状态会从当前模型重新闭合。"
        )
        if blocked_reason is not None:
            st.info(blocked_reason)
        if not archive_path:
            st.warning("请输入完整结果包路径。")
            return None
        return {
            "archive_path": archive_path,
            "point_index": point_number - 1,
        }


def compact_validation_issue(issue: dict[str, Any]) -> str:
    """Return one bounded, human-readable validation issue for sweep tables."""

    def compact(value: Any) -> str:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        return rendered if len(rendered) <= 120 else rendered[:117] + "..."

    check_name = str(issue.get("check", "validation_issue"))
    parts = [check_name]
    if "value" in issue:
        parts.append(f"value={compact(issue.get('value'))}")
    if "limit" in issue:
        parts.append(f"limit={compact(issue.get('limit'))}")
    message = str(issue.get("message", "")).strip()
    return ", ".join(parts) + (f" — {message}" if message else "")


def validation_issue_display_rows(
    issues: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Return Arrow-safe validation rows without mutating result diagnostics.

    Validation ``value`` and ``limit`` fields intentionally carry structured
    diagnostics.  Different checks can therefore place a list and a scalar in
    the same logical column, which PyArrow cannot infer as one dataframe dtype.
    Rendering every cell as portable JSON/text keeps the full information while
    leaving the original result summary unchanged for archives and analysis.
    """

    displayed: list[dict[str, str]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            displayed.append({"message": str(issue)})
            continue
        row: dict[str, str] = {}
        for key, value in issue.items():
            row[str(key)] = (
                value
                if isinstance(value, str)
                else dumps_json_portable(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )
        displayed.append(row)
    return displayed


def sweep_metric_is_valid(record: dict[str, Any], key: str) -> bool:
    """Accept old cached sweep records while honoring new metric contracts."""

    status = record.get("metric_validity", {}).get(key)
    if status is not None:
        return bool(status.get("valid", False))
    try:
        return bool(np.isfinite(float(record[key])))
    except (KeyError, TypeError, ValueError):
        return False


def archive_points_from_records(
    records: tuple[dict[str, Any], ...],
    *,
    fallback_config: dict[str, Any] | None = None,
) -> tuple[ArchivePoint, ...]:
    """Expose the complete private sweep payload through the archive contract."""

    points: list[ArchivePoint] = []
    for record in records:
        point_config = record.get("_config", fallback_config)
        if point_config is None:
            point_config = {}
        points.append(
            ArchivePoint(
                record=record,
                config=point_config,
                result=record.get("_result"),
            )
        )
    return tuple(points)


def sweep_metric_metadata(metric_keys: list[str]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "label": SWEEP_METRICS[key].label,
            "unit": SWEEP_METRICS[key].unit,
            "display_scale": SWEEP_METRICS[key].display_scale,
        }
        for key in metric_keys
    }


def render_streaming_archive_link(
    label: str,
    *,
    writer: Callable[[BinaryIO], None],
    file_name: str,
    cache_key: str,
    button_type: str = "secondary",
    help_text: str,
    lazy_generation: bool = True,
) -> None:
    """Render a loopback-only link that writes the archive on demand."""

    try:
        url = register_streaming_download(
            writer,
            cache_key=cache_key,
            file_name=file_name,
        )
    except OSError as exc:
        st.error(f"无法启动本机流式下载通道：{exc}")
        return
    st.link_button(
        lt(label),
        url,
        type=button_type,
        width="stretch",
        help=lt(help_text),
    )
    if lazy_generation:
        st.caption(
            lt(
                "只有点击后才会逐点生成下载流；不点击不创建结果 ZIP。下载完成、失败或取消后，临时压缩文件会自动清理。"
            )
        )
    else:
        st.caption(
            lt(
                "下载按钮只复制磁盘上已自动保存的归档，不会重新计算或重新压缩。"
            )
        )


def stream_existing_archive(
    destination: BinaryIO,
    archive_path: Path,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    """Copy one already-persisted archive to a browser response."""

    with archive_path.open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                return
            destination.write(chunk)


def render_sweep_archive_download(
    run: SweepRun | SweepGridRun,
    *,
    run_kind: str,
    parameters: list[dict[str, Any]],
    metric_keys: list[str],
    spacing: str | None = None,
) -> None:
    """Render one deferred full-fidelity download for every sweep point."""

    base_config = st.session_state.get("sweep_base_config")
    archive_job_id = st.session_state.get("sweep_archive_job_id")
    managed_view = get_sweep_job(_solve_session_id()) if archive_job_id else None
    managed_archive = bool(
        managed_view is not None
        and managed_view.job_id == archive_job_id
        and len(managed_view.completed_records) == len(run.records)
    )
    points = (
        ()
        if managed_archive
        else archive_points_from_records(
            run.records,
            fallback_config=base_config,
        )
    )
    metadata: dict[str, Any] = {
        "parameters": parameters,
        "metric_keys": list(metric_keys),
        "metrics": sweep_metric_metadata(metric_keys),
        "ui_compatibility_adjustments": copy.deepcopy(
            next(
                (
                    record.get("ui_compatibility_adjustments", [])
                    for record in run.records
                    if record.get("ui_compatibility_adjustments")
                ),
                [],
            )
        ),
    }
    if spacing is not None:
        metadata["spacing"] = spacing
    if not managed_archive:
        try:
            points = require_completed_sweep_results(points)
        except ResultArchiveError as exc:
            st.error(
                "完整遍历结果与后台作业的关联已丢失；已禁止生成只有摘要的伪完整结果包。"
            )
            st.caption(
                f"{exc}。请保留当前页面和后台作业现场；恢复完整结果关联后才会重新提供下载。"
            )
            summary_metadata = {
                **metadata,
                "summary_only": True,
                "full_field_data_included": False,
                "summary_record_count": len(run.records),
                "missing_full_result_points": list(
                    range(1, len(run.records) + 1)
                ),
            }
            summary_file_name = (
                "amplifier_sweep_2d_summary_only.ampres.zip"
                if run_kind == "sweep_2d"
                else "amplifier_sweep_summary_only.ampres.zip"
            )
            render_streaming_archive_link(
                "下载摘要遍历结果包（不含完整场）",
                writer=lambda destination: write_result_archive(
                    destination,
                    points,
                    run_kind=run_kind,
                    base_config=base_config,
                    run_metadata=summary_metadata,
                ),
                file_name=summary_file_name,
                cache_key=f"summary-only-sweep:{run_kind}:{id(run)}",
                button_type="secondary",
                help_text=(
                    "明确的摘要包：保留每个遍历点的页面指标、有效性状态、"
                    "warning/validation 诊断和扫描配置，但不包含完整三维场、"
                    "光学面或可重新绘制的逐点 SimulationResult。"
                ),
            )
            st.warning(
                f"这是 {len(run.records)} 点摘要包，不是完整结果包；"
                "文件名带 summary_only，载入时不会被误认为包含完整场。"
            )
            return
    completed = (
        sum(
            record.get("status") != "execution_failed"
            for record in run.records
        )
        if managed_archive
        else sum(point.result is not None for point in points)
    )
    file_name = (
        "amplifier_sweep_2d_results.ampres.zip"
        if run_kind == "sweep_2d"
        else "amplifier_sweep_results.ampres.zip"
    )
    if managed_archive:
        session_id = _solve_session_id()
        writer = lambda destination: write_sweep_job_archive(
            session_id,
            destination,
        )
        cache_key = f"managed-sweep:{session_id}:{archive_job_id}"
    else:
        writer = lambda destination: write_result_archive(
            destination,
            points,
            run_kind=run_kind,
            base_config=base_config,
            run_metadata=metadata,
        )
        cache_key = f"memory-sweep:{run_kind}:{id(run)}"
    render_streaming_archive_link(
        "下载全部遍历结果包",
        writer=writer,
        file_name=file_name,
        cache_key=cache_key,
        button_type="primary",
        help_text=(
            "一个文件包含每个点各自的配置、完整三维场、光学面、摘要、"
            "全部 warning/validation 诊断，以及失败点的异常与堆栈；可在结果分析页重新载入。"
        ),
    )
    if completed < len(run.records):
        st.caption(
            f"{completed}/{len(run.records)} 个点包含完整可视化场；其余失败点仍会在结果包中保留配置、错误和堆栈。"
        )
    else:
        st.caption(
            f"结果包包含 {len(run.records)} 个遍历点的完整独立结果，可在“结果分析”页逐点切换。"
        )


def render_sweep(run: SweepRun, parameter: SweepParameter, metric_keys: list[str], spacing: str) -> None:
    st.subheader(f"参数遍历：{parameter.label}")
    successful = list(run.successful_records)
    warning_records = list(run.warning_records)
    invalid = list(run.invalid_records)
    exploratory_invalid = [
        record
        for record in invalid
        if not bool(record.get("validation_enforced", False))
    ]
    show_invalid = bool(
        invalid
        and choice_checkbox(
            "在图中显示验证无效点（红色 ×，不连入有效趋势线）",
            value=bool(exploratory_invalid),
            key=f"show_invalid_sweep_{parameter.path}",
            help=(
                "探索模式默认显示这些诊断点；严格模式默认隐藏。无效点的原始指标"
                "始终保留，但不会冒充有效趋势。"
            ),
        )
    )
    if not successful:
        st.error("没有通过阻断性物理有效性门的遍历点；请查看下方逐点诊断表。")
    if successful or (show_invalid and invalid):
        columns = st.columns(2)
        for index, key in enumerate(metric_keys):
            definition = SWEEP_METRICS[key]
            metric_successful = [
                record for record in successful if sweep_metric_is_valid(record, key)
            ]
            metric_warnings = [
                record
                for record in warning_records
                if sweep_metric_is_valid(record, key)
            ]
            metric_invalid_runs = [
                record for record in invalid if sweep_metric_is_valid(record, key)
            ]
            metric_available_count = len(metric_successful)
            figure = go.Figure()
            model_segments: list[tuple[str, list[dict[str, Any]]]] = []
            segment_open = False
            for record in run.records:
                physically_valid = record.get("status") in {
                    "valid",
                    "valid_with_warnings",
                }
                if not physically_valid or not sweep_metric_is_valid(record, key):
                    segment_open = False
                    continue
                executed_model = str(
                    record.get("executed_signal_model", "unknown")
                )
                if (
                    not segment_open
                    or not model_segments
                    or model_segments[-1][0] != executed_model
                ):
                    model_segments.append((executed_model, [record]))
                    segment_open = True
                else:
                    model_segments[-1][1].append(record)
            legend_models: set[str] = set()
            for executed_model, model_records in model_segments:
                fallback = any(
                    bool(record.get("signal_fallback_active", False))
                    for record in model_records
                )
                show_legend = executed_model not in legend_models
                legend_models.add(executed_model)
                figure.add_trace(
                    go.Scatter(
                        x=np.asarray(
                            [record["parameter_value"] for record in model_records],
                            dtype=float,
                        )
                        * parameter.display_scale,
                        y=np.asarray(
                            [record[key] for record in model_records], dtype=float
                        )
                        * definition.display_scale,
                        mode="lines+markers",
                        name=(
                            f"{executed_model}（降阶）"
                            if fallback
                            else executed_model
                        ),
                        showlegend=show_legend,
                        legendgroup=executed_model,
                        line=dict(width=2.4, dash="dash" if fallback else "solid"),
                        marker=dict(size=8),
                        hovertemplate=(
                            f"实际模型 {executed_model}<br>{parameter.label} "
                            f"%{{x:.6g}} {parameter.unit}<br>{definition.label} "
                            f"%{{y:.6g}} {definition.unit}<extra></extra>"
                        ),
                    )
                )
            if metric_warnings:
                figure.add_trace(
                    go.Scatter(
                        x=np.asarray(
                            [record["parameter_value"] for record in metric_warnings],
                            dtype=float,
                        )
                        * parameter.display_scale,
                        y=np.asarray(
                            [record[key] for record in metric_warnings],
                            dtype=float,
                        )
                        * definition.display_scale,
                        mode="markers",
                        name="有效但有诊断警告",
                        marker=dict(
                            size=11,
                            symbol="circle-open",
                            color="#ffbf00",
                            line=dict(width=2),
                        ),
                        hovertemplate=(
                            f"有效，但诊断产品有警告<br>{parameter.label} "
                            f"%{{x:.6g}} {parameter.unit}<br>{definition.label} "
                            f"%{{y:.6g}} {definition.unit}<extra></extra>"
                        ),
                    )
                )
            if show_invalid and metric_invalid_runs:
                figure.add_trace(
                    go.Scatter(
                        x=np.asarray(
                            [record["parameter_value"] for record in metric_invalid_runs],
                            dtype=float,
                        )
                        * parameter.display_scale,
                        y=np.asarray(
                            [record[key] for record in metric_invalid_runs], dtype=float
                        )
                        * definition.display_scale,
                        mode="markers",
                        name="invalid（诊断）",
                        marker=dict(size=10, symbol="x", color="#d62728"),
                        hovertemplate=(
                            f"无效诊断点<br>{parameter.label} %{{x:.6g}} {parameter.unit}<br>"
                            f"{definition.label} %{{y:.6g}} {definition.unit}<extra></extra>"
                        ),
                    )
                )
            figure.update_layout(
                title=definition.label,
                xaxis_title=f"{parameter.label} ({parameter.unit})" if parameter.unit else parameter.label,
                yaxis_title=f"{definition.label} ({definition.unit})" if definition.unit else definition.label,
                xaxis_type="log" if spacing == "log" else "linear",
                height=330,
                margin=dict(t=55, l=60, r=20, b=55),
            )
            columns[index % 2].plotly_chart(figure, width="stretch", key=f"sweep_{parameter.path}_{key}")
            unavailable_count = len(successful) - metric_available_count
            if unavailable_count:
                columns[index % 2].caption(
                    f"{unavailable_count} 个物理有效点的“{definition.label}”诊断本身无效，"
                    "已只从此图中剔除；其它指标仍保留。"
                )
    failed = [record for record in run.records if record.get("error")]
    if invalid:
        st.warning(
            f"{len(invalid)} / {len(run.records)} 个点运行完成但未通过有效性门；"
            "这些点不会连入有效趋势线。探索模式下默认以红色 × 显示，"
            "严格模式下默认隐藏但仍保留诊断。"
        )
    if warning_records:
        st.info(
            f"{len(warning_records)} / {len(run.records)} 个点通过全部阻断性合同，"
            "但存在只影响记录平面、采样光斑或类似诊断产品的质量警告。"
        )
    if failed:
        st.warning(f"{len(failed)} / {len(run.records)} 个遍历点失败；其余点仍已保留。")
    with st.expander(
        "遍历逐点原始数据与诊断",
        expanded=bool(failed or invalid or warning_records),
    ):
        display_records: list[dict[str, Any]] = []
        for record in run.records:
            converted: dict[str, Any] = {
                f"{parameter.label} ({parameter.unit})": float(record["parameter_value"]) * parameter.display_scale,
                "status": record.get("status"),
                "error": record.get("error"),
                "requested_signal_model": record.get("requested_signal_model"),
                "executed_signal_model": record.get("executed_signal_model"),
                "signal_fallback_active": record.get("signal_fallback_active"),
                "invalid_metrics": "; ".join(record.get("invalid_metrics", [])),
                "validation_failures": "; ".join(
                    str(item.get("check", item.get("message", "invalid")))
                    for item in record.get("validation_failures", [])
                ),
                "validation_warnings": "; ".join(
                    str(item.get("check", item.get("message", "warning")))
                    for item in record.get("validation_warnings", [])
                ),
                "failure_details": "; ".join(
                    compact_validation_issue(item)
                    for item in record.get("validation_failures", [])
                ),
                "warning_details": "; ".join(
                    compact_validation_issue(item)
                    for item in record.get("validation_warnings", [])
                ),
            }
            for key in metric_keys:
                if key in record:
                    definition = SWEEP_METRICS[key]
                    converted[f"{definition.label} ({definition.unit})"] = float(record[key]) * definition.display_scale
            display_records.append(converted)
        st.dataframe(display_records, width="stretch")
    render_sweep_archive_download(
        run,
        run_kind="sweep_1d",
        parameters=[
            {
                "axis": "x",
                "path": parameter.path,
                "label": parameter.label,
                "unit": parameter.unit,
                "display_scale": parameter.display_scale,
                "value_field": "parameter_value",
            }
        ],
        metric_keys=metric_keys,
        spacing=spacing,
    )


def render_sweep_grid(
    run: SweepGridRun,
    parameter_x: SweepParameter,
    parameter_y: SweepParameter,
    metric_keys: list[str],
    spacing_x: str = "linear",
    spacing_y: str = "linear",
) -> None:
    st.subheader(f"二维参数遍历：{parameter_x.label} × {parameter_y.label}")
    successful = list(run.successful_records)
    warning_records = list(run.warning_records)
    invalid = list(run.invalid_records)
    exploratory_invalid = [
        record
        for record in invalid
        if not bool(record.get("validation_enforced", False))
    ]
    show_invalid = bool(
        invalid
        and choice_checkbox(
            "在二维图中包含验证无效格点的原始指标",
            value=bool(exploratory_invalid),
            key=f"show_invalid_sweep_grid_{parameter_x.path}_{parameter_y.path}",
            help=(
                "探索模式默认显示这些已完成工况的原始指标；热图和切片用红叉标记无效点，"
                "有效曲线在这些位置断开。"
            ),
        )
    )
    if not successful:
        st.error("二维遍历没有通过阻断性物理有效性门的格点；请查看逐点诊断表。")
    if successful or (show_invalid and invalid):
        x_values = np.asarray(run.parameter_x_values, dtype=float)
        y_values = np.asarray(run.parameter_y_values, dtype=float)
        x_lookup = {float(value): index for index, value in enumerate(x_values)}
        y_lookup = {float(value): index for index, value in enumerate(y_values)}
        columns = st.columns(2)
        slice_sources: dict[
            str, tuple[list[dict[str, Any]], list[dict[str, Any]]]
        ] = {}
        for metric_index, key in enumerate(metric_keys):
            definition = SWEEP_METRICS[key]
            metric_successful = [
                record for record in successful if sweep_metric_is_valid(record, key)
            ]
            metric_warnings = [
                record
                for record in warning_records
                if sweep_metric_is_valid(record, key)
            ]
            metric_invalid_runs = [
                record for record in invalid if sweep_metric_is_valid(record, key)
            ]
            metric_available_count = len(metric_successful)
            executed_models = list(
                dict.fromkeys(
                    str(record.get("executed_signal_model", "unknown"))
                    for record in metric_successful + (metric_invalid_runs if show_invalid else [])
                )
            )
            if len(executed_models) > 1:
                selected_model = columns[metric_index % 2].selectbox(
                    f"{definition.label}：实际 Seed 后端",
                    executed_models,
                    key=(
                        "sweep_grid_backend_"
                        f"{parameter_x.path}_{parameter_y.path}_{key}"
                    ),
                    help=(
                        "不同实际后端的结果不会混入同一热图；切换此项分别查看。"
                    ),
                )
                metric_successful = [
                    record
                    for record in metric_successful
                    if str(record.get("executed_signal_model", "unknown"))
                    == selected_model
                ]
                metric_warnings = [
                    record
                    for record in metric_warnings
                    if str(record.get("executed_signal_model", "unknown"))
                    == selected_model
                ]
                metric_invalid_runs = [
                    record
                    for record in metric_invalid_runs
                    if str(record.get("executed_signal_model", "unknown"))
                    == selected_model
                ]
            slice_sources[key] = (
                metric_successful,
                metric_invalid_runs if show_invalid else [],
            )
            values = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
            metric_display_records = metric_successful + (
                metric_invalid_runs if show_invalid else []
            )
            for record in metric_display_records:
                ix = x_lookup.get(float(record["parameter_x_value"]))
                iy = y_lookup.get(float(record["parameter_y_value"]))
                if ix is not None and iy is not None:
                    values[iy, ix] = float(record[key]) * definition.display_scale
            figure = go.Figure()
            if metric_display_records:
                figure.add_trace(go.Heatmap(
                    x=x_values * parameter_x.display_scale,
                    y=y_values * parameter_y.display_scale,
                    z=values,
                    colorscale="Viridis",
                    colorbar={"title": definition.unit},
                    hovertemplate=(
                        f"{parameter_x.label} %{{x:.6g}} {parameter_x.unit}<br>"
                        f"{parameter_y.label} %{{y:.6g}} {parameter_y.unit}<br>"
                        f"{definition.label} %{{z:.6g}} {definition.unit}<extra></extra>"
                    ),
                ))
            if show_invalid:
                add_invalid_grid_markers(
                    figure, metric_invalid_runs, x_field="parameter_x_value", y_field="parameter_y_value",
                    x_scale=parameter_x.display_scale, y_scale=parameter_y.display_scale,
                    metric_key=key, metric_scale=definition.display_scale, metric_unit=definition.unit,
                )
            if metric_warnings:
                figure.add_trace(
                    go.Scatter(
                        x=np.asarray(
                            [record["parameter_x_value"] for record in metric_warnings],
                            dtype=float,
                        )
                        * parameter_x.display_scale,
                        y=np.asarray(
                            [record["parameter_y_value"] for record in metric_warnings],
                            dtype=float,
                        )
                        * parameter_y.display_scale,
                        mode="markers",
                        name="有效但有诊断警告",
                        marker=dict(
                            size=11,
                            symbol="circle-open",
                            color="#ffbf00",
                            line=dict(width=2),
                        ),
                        hovertemplate=(
                            f"有效，但诊断产品有警告<br>{parameter_x.label} "
                            f"%{{x:.6g}} {parameter_x.unit}<br>{parameter_y.label} "
                            f"%{{y:.6g}} {parameter_y.unit}<extra></extra>"
                        ),
                    )
                )
            figure.update_layout(
                title=definition.label,
                xaxis_title=f"{parameter_x.label} ({parameter_x.unit})",
                yaxis_title=f"{parameter_y.label} ({parameter_y.unit})",
                xaxis_type="log" if spacing_x == "log" else "linear",
                yaxis_type="log" if spacing_y == "log" else "linear",
                height=390,
                margin=dict(t=55, l=70, r=20, b=60),
            )
            columns[metric_index % 2].plotly_chart(
                figure,
                width="stretch",
                key=f"sweep_grid_{parameter_x.path}_{parameter_y.path}_{key}",
            )
            unavailable_count = len(successful) - metric_available_count
            if unavailable_count:
                columns[metric_index % 2].caption(
                    f"{unavailable_count} 个物理有效格点的“{definition.label}”诊断无效，"
                    "该格只在本热图中留空。"
                )
        slice_label_to_key = {
            SWEEP_METRICS[key].label: key
            for key in metric_keys
            if key in slice_sources
        }
        if slice_label_to_key:
            st.markdown("#### 二维遍历切片曲线")
            selected_slice_label = choice_selectbox(
                "切片曲线指标",
                list(slice_label_to_key),
                key=f"sweep_grid_slice_metric_{parameter_x.path}_{parameter_y.path}",
                help=(
                    "左图固定第二参数并扫描第一参数；右图固定第一参数并扫描第二参数。"
                    "每条细线代表一个固定值。"
                ),
            )
            selected_slice_key = slice_label_to_key[selected_slice_label]
            definition = SWEEP_METRICS[selected_slice_key]
            slice_valid, _slice_invalid = slice_sources[selected_slice_key]
            scan_x, scan_y = build_sweep_slice_figures(
                valid_records=slice_valid,
                invalid_records=_slice_invalid,
                x_values=x_values,
                y_values=y_values,
                x_field="parameter_x_value",
                y_field="parameter_y_value",
                metric_key=selected_slice_key,
                x_label=parameter_x.label,
                x_unit=parameter_x.unit,
                x_scale=parameter_x.display_scale,
                y_label=parameter_y.label,
                y_unit=parameter_y.unit,
                y_scale=parameter_y.display_scale,
                metric_label=definition.label,
                metric_unit=definition.unit,
                metric_scale=definition.display_scale,
                x_spacing=spacing_x,
                y_spacing=spacing_y,
            )
            slice_columns = st.columns(2)
            slice_columns[0].plotly_chart(
                scan_x,
                width="stretch",
                key=(
                    "sweep_grid_slice_x_"
                    f"{parameter_x.path}_{parameter_y.path}_{selected_slice_key}"
                ),
            )
            slice_columns[1].plotly_chart(
                scan_y,
                width="stretch",
                key=(
                    "sweep_grid_slice_y_"
                    f"{parameter_x.path}_{parameter_y.path}_{selected_slice_key}"
                ),
            )
            if show_invalid and invalid:
                st.caption(
                    "曲线包含已完成但未通过有效性门的原始指标；"
                    "无效点以红叉显示，有效曲线在这些位置断开；它们不属于定量结果。"
                )
    failed = [record for record in run.records if record.get("error")]
    if invalid:
        st.warning(
            f"{len(invalid)} / {len(run.records)} 个二维格点运行完成但未通过有效性门；"
            "探索模式下可将其原始指标纳入热图和切片曲线，"
            "但它们仍不是通过有效性合同的定量结果。"
        )
    if warning_records:
        st.info(
            f"{len(warning_records)} / {len(run.records)} 个格点通过全部阻断性合同，"
            "但存在只影响记录平面、采样光斑或类似诊断产品的质量警告。"
        )
    if failed:
        st.warning(f"{len(failed)} / {len(run.records)} 个二维格点失败；其它格点仍已保留。")
    with st.expander(
        "二维遍历逐点原始数据与诊断",
        expanded=bool(failed or invalid or warning_records),
    ):
        display_records: list[dict[str, Any]] = []
        for record in run.records:
            converted: dict[str, Any] = {
                f"{parameter_x.label} ({parameter_x.unit})": (
                    float(record["parameter_x_value"]) * parameter_x.display_scale
                ),
                f"{parameter_y.label} ({parameter_y.unit})": (
                    float(record["parameter_y_value"]) * parameter_y.display_scale
                ),
                "status": record.get("status"),
                "error": record.get("error"),
                "requested_signal_model": record.get("requested_signal_model"),
                "executed_signal_model": record.get("executed_signal_model"),
                "signal_fallback_active": record.get("signal_fallback_active"),
                "invalid_metrics": "; ".join(record.get("invalid_metrics", [])),
                "validation_failures": "; ".join(
                    str(item.get("check", item.get("message", "invalid")))
                    for item in record.get("validation_failures", [])
                ),
                "validation_warnings": "; ".join(
                    str(item.get("check", item.get("message", "warning")))
                    for item in record.get("validation_warnings", [])
                ),
                "failure_details": "; ".join(
                    compact_validation_issue(item)
                    for item in record.get("validation_failures", [])
                ),
                "warning_details": "; ".join(
                    compact_validation_issue(item)
                    for item in record.get("validation_warnings", [])
                ),
            }
            for key in metric_keys:
                if key in record:
                    definition = SWEEP_METRICS[key]
                    converted[f"{definition.label} ({definition.unit})"] = (
                        float(record[key]) * definition.display_scale
                    )
            display_records.append(converted)
        st.dataframe(display_records, width="stretch")
    render_sweep_archive_download(
        run,
        run_kind="sweep_2d",
        parameters=[
            {
                "axis": "x",
                "path": parameter_x.path,
                "label": parameter_x.label,
                "unit": parameter_x.unit,
                "display_scale": parameter_x.display_scale,
                "value_field": "parameter_x_value",
                "spacing": spacing_x,
            },
            {
                "axis": "y",
                "path": parameter_y.path,
                "label": parameter_y.label,
                "unit": parameter_y.unit,
                "display_scale": parameter_y.display_scale,
                "value_field": "parameter_y_value",
                "spacing": spacing_y,
            },
        ],
        metric_keys=metric_keys,
    )


def heatmap(
    values: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    title: str,
    unit: str,
    colorscale: str = "Viridis",
    zmid: float | None = None,
) -> go.Figure:
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=x_m * 1e6,
            y=y_m * 1e6,
            colorbar_title=unit,
            colorscale=colorscale,
            zmid=zmid,
        )
    )
    figure.update_layout(title=title, xaxis_title="x (µm)", yaxis_title="y (µm)", height=430, margin=dict(t=55, l=55, r=20, b=45))
    return figure


def axial_field_plot(
    values: np.ndarray,
    z_m: np.ndarray,
    title: str,
    unit: str,
    material_mask: np.ndarray | None = None,
    cell_areas: np.ndarray | None = None,
) -> go.Figure:
    values = np.asarray(values, dtype=float)
    if values.ndim != 3:
        raise ValueError("axial field values must have shape (nz, ny, nx)")
    if cell_areas is None:
        cell_areas = np.ones(values.shape[-2:], dtype=float)
    minimum, average, maximum = axial_weighted_statistics(
        values,
        np.asarray(cell_areas, dtype=float),
        material_mask,
    )
    figure = go.Figure()
    figure.add_scatter(x=z_m * 1e3, y=maximum, mode="lines", line=dict(width=0), showlegend=False)
    figure.add_scatter(
        x=z_m * 1e3,
        y=minimum,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(56,189,248,0.16)",
        name="截面 min–max",
    )
    figure.add_scatter(x=z_m * 1e3, y=average, mode="lines+markers", name="面积加权截面平均", line=dict(width=2.5))
    figure.update_layout(
        title=title,
        xaxis_title="z (mm)",
        yaxis_title=unit,
        height=430,
        margin=dict(t=55, l=70, r=20, b=50),
    )
    return figure


def bin_widths(coordinate: np.ndarray) -> np.ndarray:
    if len(coordinate) < 2:
        raise ValueError("At least two coordinates are required to infer bin widths")
    edges = np.empty(len(coordinate) + 1)
    edges[1:-1] = 0.5 * (coordinate[:-1] + coordinate[1:])
    edges[0] = coordinate[0] - 0.5 * (coordinate[1] - coordinate[0])
    edges[-1] = coordinate[-1] + 0.5 * (coordinate[-1] - coordinate[-2])
    return np.diff(edges)


def temporal_result_figure(result: SimulationResult) -> go.Figure:
    """Display singleton bin energy without inventing a time duration."""

    singleton = len(result.time_s) == 1
    edges = getattr(result, "temporal_bin_edges_s", None)
    widths = None if singleton else (
        np.diff(edges) if edges is not None else bin_widths(result.time_s)
    )
    figure = go.Figure()
    for name, energy in (
        ("输入", result.input_temporal_energy_J),
        ("输出", result.output_temporal_energy_J),
    ):
        figure.add_scatter(
            x=result.time_s * 1e15,
            y=energy if singleton else energy / widths,
            mode="markers" if singleton else "lines",
            name=name,
        )
    if not singleton and result.compressed_temporal_power_W is not None:
        figure.add_scatter(
            x=result.time_s * 1e15,
            y=result.compressed_temporal_power_W,
            name="压缩后",
        )
    figure.update_layout(
        title="时间分箱能量" if singleton else "时间波形",
        xaxis_title="时间 (fs)",
        yaxis_title="每个时间分箱能量 (J)" if singleton else "功率 (W)",
    )
    return figure


def scaled_energy(joules: float) -> tuple[float, str, float]:
    """Return a readable value, unit, and J-to-display multiplier."""

    magnitude = abs(float(joules))
    for threshold, unit, scale in (
        (1.0, "J", 1.0),
        (1e-3, "mJ", 1e3),
        (1e-6, "µJ", 1e6),
        (1e-9, "nJ", 1e9),
        (1e-12, "pJ", 1e12),
    ):
        if magnitude >= threshold:
            return float(joules) * scale, unit, scale
    return float(joules) * 1e15, "fJ", 1e15


def format_power(watts: float | None) -> str:
    if watts is None:
        return "—"
    magnitude = abs(float(watts))
    if magnitude >= 1e3:
        return f"{float(watts) / 1e3:.5g} kW"
    if magnitude >= 1.0:
        return f"{float(watts):.5g} W"
    if magnitude >= 1e-3:
        return f"{float(watts) * 1e3:.5g} mW"
    if magnitude >= 1e-6:
        return f"{float(watts) * 1e6:.5g} µW"
    return f"{float(watts) * 1e9:.5g} nW"


def _render_intensity_snapshots(result: SimulationResult, result_key) -> None:
    values = result.longitudinal_intensity_W_m2
    if values is None:
        return
    metadata = result.summary.get("signal", {}).get("gain_transport", {}).get(
        "intensity_snapshots", {},
    )
    planes = metadata.get("planes", [])
    if not planes:
        return
    with st.expander("晶体内瞬时强度与收缩", expanded=False):
        st.caption("每幅图是一个时刻的强度，单位 W/m²。固定时刻取每程入口的功率峰值时间；当地峰值时刻可以随传播移动。D4σ 半径包含外围光翼，中心是否收缩需结合强度图判断。")
        pass_number = choice_selectbox(
            "强度截面的晶体程次", sorted({p["pass"] for p in planes}),
            key=result_key("intensity_pass"),
            help="选择一次晶体通过，查看该程内保存的强度截面。",
        )
        selection_label = choice_selectbox(
            "强度截面时间选择", ["当地峰值功率时刻", "固定入口峰值时刻"],
            key=result_key("intensity_time_selection"),
            help="当地峰值在每个截面重新寻找功率峰值；固定入口峰值沿用该程入口的同一时间坐标。",
        )
        selection_index = 0 if selection_label == "当地峰值功率时刻" else 1
        pass_planes = [p for p in planes if p["pass"] == pass_number]
        figure = go.Figure()
        positions = [p["path_position_m"] * 1e3 for p in pass_planes]
        for axis in ("x", "y"):
            figure.add_scatter(
                x=positions,
                y=[p["selections"][selection_index][f"radius_{axis}_D4sigma_m"] * 1e6 for p in pass_planes],
                name=f"D4σ radius {axis}", mode="lines+markers",
            )
        figure.update_layout(xaxis_title="晶体内传播距离 (mm)", yaxis_title="瞬时 D4σ 半径 (µm)")
        st.plotly_chart(figure, width="stretch")
        power_figure = go.Figure(go.Scatter(
            x=positions,
            y=[p["selections"][selection_index]["P_over_Pcr"] for p in pass_planes],
            mode="lines+markers", name="P/Pcr",
        ))
        power_figure.add_hline(y=1.0, line_dash="dash")
        power_figure.update_layout(xaxis_title="晶体内传播距离 (mm)", yaxis_title="P/Pcr")
        st.plotly_chart(power_figure, width="stretch")
        plane_index = choice_selectbox(
            "查看强度截面", [p["plane_index"] for p in pass_planes],
            format_func=lambda i: f"z={planes[i]['z_m'] * 1e3:.4f} mm / cell={planes[i]['cell_number']}",
            key=result_key("intensity_plane"),
            help="选择已保存的纵向采样平面；z 为晶体坐标，cell 为网格单元编号。",
        )
        plane = planes[plane_index]
        geometry = metadata["grids_by_pass"][str(pass_number)]
        intensity = values[plane_index, selection_index]
        spatial_figure = go.Figure()
        if geometry["coordinate_system"] == "polar":
            radius = np.asarray(geometry["radius_m"]) * 1e6
            if intensity.shape[0] == 1:
                spatial_figure.add_scatter(x=radius, y=intensity[0], mode="lines+markers")
                spatial_figure.update_layout(xaxis_title="半径 (µm)", yaxis_title="I (W/m²)")
            else:
                spatial_figure.add_heatmap(x=radius, y=np.rad2deg(geometry["theta_rad"]), z=intensity, colorbar_title="W/m²")
                spatial_figure.update_layout(xaxis_title="半径 (µm)", yaxis_title="方位角 (°)")
        else:
            spatial_figure.add_heatmap(
                x=np.asarray(geometry["x_m"]) * 1e6,
                y=np.asarray(geometry["y_m"]) * 1e6, z=intensity, colorbar_title="W/m²",
            )
            spatial_figure.update_layout(xaxis_title="x (µm)", yaxis_title="y (µm)")
            spatial_figure.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(spatial_figure, width="stretch")
        st.dataframe([{
            "z_m": p["z_m"], "path_position_m": p["path_position_m"],
            **p["selections"][selection_index],
        } for p in pass_planes], hide_index=True, width="stretch")
        st.caption("P/Pcr 为同一截面、同一时刻的体材料参考比值，不是自聚焦判据的精确开关。截面保存在结果包 fields.npz，坐标与时刻保存在 summary.json。")


def render_result(
    result: SimulationResult,
    cfg: dict[str, Any],
    *,
    archive_record: dict[str, Any] | None = None,
    archive_key: str = "single_result",
) -> None:
    def result_key(component: str) -> str:
        """Keep interactive result state local to one run/archive point."""

        return f"{component}__{archive_key}"

    summary = result.summary
    point_record = archive_record or diagnostic_record_for_result(result)
    validation = summary.get("validation", {})
    validation_failures = list(validation.get("failures", []))
    validation_warnings = list(validation.get("warnings", []))
    invalid_result = not validation.get("valid", True)
    strict_invalid = bool(validation.get("enforced", False) and invalid_result)
    configured_input_energy = summary["signal"]["configured_source_energy_J"]
    grid_input_energy = summary["signal"]["grid_source_energy_J"]
    output_energy = summary["signal"]["system_output_energy_J"]
    output_power = summary["signal"].get("system_output_average_power_W")
    population_summary = summary["population"]
    dynamics = str(population_summary.get("dynamics", "unknown"))
    converged = population_summary.get("converged")
    cycles = int(population_summary.get("cycles_completed", population_summary.get("pulse_train_cycles", 0)) or 0)
    output_energy_value, output_energy_unit, _ = scaled_energy(output_energy)
    # A result is an immutable execution snapshot.  Never re-resolve its raw
    # ``auto`` configuration through today's preset table: a later preset
    # update could otherwise relabel an archived backend that never ran.
    # Prefer diagnostics emitted by the executed modules, with the frozen
    # effective-physics record as the legacy fallback.
    recorded_physics = summary.get("physics") or {}
    signal_used = str(
        summary.get("signal", {}).get("model")
        or recorded_physics.get("signal_model")
        or "unknown"
    )
    pump_used = str(
        summary.get("pump", {}).get("propagation")
        or recorded_physics.get("pump_propagation")
        or "unknown"
    )
    population_used = str(
        population_summary.get("dynamics")
        or recorded_physics.get("population_dynamics")
        or "unknown"
    )
    thermal_used = str(
        summary.get("thermal", {}).get("model")
        or recorded_physics.get("thermal_model")
        or "unknown"
    )
    nz, ny, nx = result.grid.shape
    st.subheader("结果摘要")
    st.caption(
        f"结果快照：Seed {signal_used} · "
        f"Pump {pump_used} · "
        f"布居 {population_used} · "
        f"热 {thermal_used} · "
        f"网格 {nx}×{ny}×{nz}。"
    )
    headline = st.columns(3)
    headline[0].metric(
        "最终输出平均功率",
        format_power(None if output_power is None else float(output_power)),
        help="最后一个已报告 Seed 脉冲能量乘系统重复频率。单次模式没有平均功率。",
    )
    headline[1].metric("最终脉冲能量", f"{output_energy_value:.5g} {output_energy_unit}")
    if dynamics == "cycle_averaged_steady_state":
        state_text = "平均稳态已收敛" if converged else "平均稳态未收敛"
        state_detail = f"{int(population_summary.get('iterations', 0))} 次自洽迭代"
    elif dynamics in {
        "periodic_steady_state",
        "reduced_periodic_steady_state",
    }:
        if dynamics == "reduced_periodic_steady_state":
            state_text = "代理周期图已收敛" if converged else "代理周期图未收敛"
        else:
            state_text = "已收敛" if converged else "未收敛"
        state_detail = f"{cycles} 次周期映射评估"
    elif dynamics == "steady_state":
        state_text = "仅 CW 泵稳态" if converged else "CW 泵未收敛"
        state_detail = "不含重复 Seed 耗尽"
    elif dynamics == "frozen":
        state_text = "跳过 Pump 预备与暗弛豫"
        state_detail = "Seed 耗尽仍按增益饱和开关执行"
    else:
        state_text = "瞬态结果"
        if cycles and active_locale() == EN:
            state_detail = f"{cycles} {'Cycle' if cycles == 1 else 'Cycles'}"
        else:
            state_detail = f"{cycles} 周期" if cycles else dynamics
    headline[2].metric("运行状态", state_text, delta=state_detail, delta_color="off")

    if strict_invalid:
        st.error(
            f"{NON_QUANTITATIVE_BANNER}\n\n"
            "严格验证未通过：该结果只可用于诊断，不能作为正式输出或进入参数扫描。"
        )
        st.dataframe(
            validation_issue_display_rows(validation_failures),
            hide_index=True,
            width="stretch",
        )
    elif not validation.get("valid", True):
        st.error(
            f"{NON_QUANTITATIVE_BANNER}\n\n"
            "探索模式发现阻断性结果合同失败；计算已完成并保留诊断，但该点不能作为"
            "可信的定量结论。"
        )
        st.dataframe(
            validation_issue_display_rows(validation_failures),
            hide_index=True,
            width="stretch",
        )

    if validation_warnings:
        if validation.get("valid", True):
            st.info(
                "阻断性合同已通过，但有诊断产品质量警告；功率/能量结果仍可使用，"
                "受影响的光斑、记录平面或采样 M² 需按下表处理。"
            )
        else:
            st.info(
                "除上面的阻断性失败外，这次计算还包含诊断产品质量警告；"
                "它们不是结果无效的原因，但仍需在解释光斑、记录平面或采样 M² 时处理。"
            )
        st.dataframe(
            validation_issue_display_rows(validation_warnings),
            hide_index=True,
            width="stretch",
        )

    python_warnings = list(point_record.get("python_warnings", []))
    if python_warnings:
        with st.expander(
            f"运行时 Python warning（{len(python_warnings)} 条）",
            expanded=False,
        ):
            st.dataframe(python_warnings, hide_index=True, width="stretch")

    single_archive_points = (
        ArchivePoint(record=point_record, config=cfg, result=result),
    )
    single_archive_metadata = {
        "source": "single_run",
        "ui_compatibility_adjustments": copy.deepcopy(
            point_record.get("ui_compatibility_adjustments", [])
        ),
    }
    autosaved_archive_path_value = point_record.get("autosaved_archive_path")
    autosaved_archive_path = (
        None
        if autosaved_archive_path_value is None
        else Path(str(autosaved_archive_path_value))
    )
    autosaved_archive_exists = bool(
        autosaved_archive_path is not None and autosaved_archive_path.is_file()
    )
    if autosaved_archive_exists and autosaved_archive_path is not None:
        st.success(
            lt("完整结果包已自动保存到磁盘；刷新或关闭网页不会删除它。")
        )
        st.code(str(autosaved_archive_path), language="text")

        def archive_writer(destination: BinaryIO) -> None:
            stream_existing_archive(destination, autosaved_archive_path)

    else:
        if point_record.get("autosave_archive_error"):
            st.error(
                lt("计算已完成，但自动保存结果包失败：")
                + str(point_record["autosave_archive_error"])
                + lt("。请在当前网页会话仍然有效时使用下面的临时下载。")
            )

        def archive_writer(destination: BinaryIO) -> None:
            write_result_archive(
                destination,
                single_archive_points,
                run_kind="single",
                base_config=cfg,
                run_metadata=single_archive_metadata,
            )

    render_streaming_archive_link(
        (
            "下载严格无效诊断包（非正式结果）"
            if strict_invalid
            else "下载无效诊断包（非定量结果）"
            if invalid_result
            else (
                "下载已自动保存的完整结果包"
                if autosaved_archive_exists
                else "下载当前单点完整结果包（可在结果分析页重新载入）"
            )
        ),
        writer=archive_writer,
        file_name=(
            autosaved_archive_path.name
            if autosaved_archive_exists and autosaved_archive_path is not None
            else (
                "amplifier_invalid_diagnostic.ampres.zip"
                if invalid_result
                else "amplifier_result.ampres.zip"
            )
        ),
        cache_key=(
            f"saved-single-result:{autosaved_archive_path}"
            if autosaved_archive_exists
            else f"single-result:{archive_key}"
        ),
        button_type="secondary" if invalid_result else "primary",
        help_text=(
            "仅用于复现和定位验证失败；不是正式结果导出。包内保留配置、"
            "完整三维场、光学面、摘要及全部 warning/validation 诊断，重新载入时"
            "仍会标记为非定量。"
            if invalid_result
            else (
                "复制已经自动保存的完整结果包；不会重新计算或重新压缩。"
                if autosaved_archive_exists
                else "包含配置、完整三维场、光学面、摘要及全部 warning/validation 诊断。"
                "这里只包含当前单点；二维参数网格请使用遍历结果区的“下载全部遍历结果包”。"
            )
        ),
        lazy_generation=not autosaved_archive_exists,
    )
    st.caption(
        (
            "该文件只用于复现验证失败并定位问题，不是推荐的正式结果归档。"
            if invalid_result
            else "这是推荐归档格式；一个文件即可恢复本页的完整可视化和诊断。"
        )
        + "包内 SHA-256 只校验内容完整性，不是作者身份或来源签名。"
    )

    st.subheader("交互式三维结果工作台")
    st.caption(
        "默认先打开可拖动旋转的三维光路；“三维场与可拖动切片”可选择任意已保存体场、"
        "切换 x/y/z 方向并拖动切片位置。这里显示的是本次结果快照，不会重新求解。"
    )
    path_tab, spatial_tab, spectral_tab, diagnostics_tab = st.tabs(
        [
            "三维光路工作台",
            "三维场与可拖动切片",
            "光谱与脉冲",
            "诊断与导出",
        ]
    )
    # Per-pass energy and recorded optical planes share the optical workspace.
    optics_tab = path_tab

    with st.expander("查看其他结果指标"):
        other_metric_rows = [
                {"指标": "设定种子能量", "值": configured_input_energy * 1e9, "单位": "nJ"},
                {"指标": "网格源面种子能量", "值": grid_input_energy * 1e9, "单位": "nJ"},
                {
                    "指标": "Seed 晶体入口解析能量",
                    "值": summary["signal"].get("analytic_gain_entrance_energy_J"),
                    "单位": "J",
                },
                {
                    "指标": "Seed 晶体入口采样能量",
                    "值": summary["signal"].get("sampled_gain_entrance_energy_J"),
                    "单位": "J",
                },
                {
                    "指标": "Seed 晶体入口窗口捕获率",
                    "值": summary["signal"].get("gain_entrance_capture_fraction"),
                    "单位": "",
                },
                {
                    "指标": "Seed 数值窗口截断",
                    "值": summary["signal"].get("numerical_window_truncation_J"),
                    "单位": "J",
                },
                {
                    "指标": "Seed 晶体前物理外部损失",
                    "值": summary["signal"].get("physical_external_loss_J"),
                    "单位": "J",
                },
                {
                    "指标": "设定 Seed 输入平均功率",
                    "值": summary["signal"].get("configured_source_average_power_W"),
                    "单位": "W",
                },
                {
                    "指标": "网格源面 Seed 平均功率",
                    "值": summary["signal"].get("grid_source_average_power_W"),
                    "单位": "W",
                },
                {
                    "指标": "网格源面→系统输出增益",
                    "值": output_energy / max(grid_input_energy, np.finfo(float).tiny),
                    "单位": "×",
                },
                {"指标": "平均泵浦吸收", "值": summary["pump"]["absorbed_power_W"], "单位": "W"},
                {"指标": "种子净提取", "值": summary["extraction"]["net_J"] * 1e6, "单位": "µJ"},
                {"指标": "最高温度", "值": summary["thermal"]["temperature_max_K"], "单位": "K"},
        ]
        output_quality = summary["signal"].get("output_beam_quality") or {}
        if output_quality.get("valid", False):
            m2_quantity_kind = str(
                output_quality.get("reported_quantity_kind", "")
            )
            if m2_quantity_kind == "sampled_complex_field_phase_space_m2":
                m2_label_x = "输出 Seed M² x（采样相空间）"
                m2_label_y = "输出 Seed M² y（采样相空间）"
            elif m2_quantity_kind == "sampled_incoherent_second_moment_m2":
                m2_label_x = "输出 Seed M² x（非相干二阶矩）"
                m2_label_y = "输出 Seed M² y（非相干二阶矩）"
            elif m2_quantity_kind in {
                "configured_gaussian_modal_m2_contract",
                "commercial_app_configured_transport_m2_contract",
            }:
                m2_label_x = "输出 Seed M² x（模态合同）"
                m2_label_y = "输出 Seed M² y（模态合同）"
            else:
                m2_label_x = "输出有效 Seed M² x（代理）"
                m2_label_y = "输出有效 Seed M² y（代理）"
            other_metric_rows.extend(
                [
                    {"指标": m2_label_x, "值": output_quality.get("M2_x"), "单位": ""},
                    {"指标": m2_label_y, "值": output_quality.get("M2_y"), "单位": ""},
                ]
            )
        doping_summary = summary.get("doping_profile", {})
        if doping_summary:
            other_metric_rows.extend(
                [
                    {
                        "指标": "掺杂浓度体积平均值",
                        "值": doping_summary.get("volume_average_density_m3"),
                        "单位": "m⁻³",
                    },
                    {
                        "指标": "掺杂浓度最小值",
                        "值": doping_summary.get("minimum_density_m3"),
                        "单位": "m⁻³",
                    },
                    {
                        "指标": "掺杂浓度最大值",
                        "值": doping_summary.get("maximum_density_m3"),
                        "单位": "m⁻³",
                    },
                    {
                        "指标": "网格表示的总掺杂离子数",
                        "值": doping_summary.get("total_dopant_ions"),
                        "单位": "ions",
                    },
                ]
            )
        if dynamics == "cycle_averaged_steady_state":
            other_metric_rows.extend(
                [
                    {
                        "指标": "平均光子流自洽探针输出",
                        "值": population_summary.get("average_probe_output_power_W"),
                        "单位": "W",
                    },
                    {
                        "指标": "最终单脉冲局部最大布居跳变",
                        "值": 100.0
                        * float(population_summary.get("final_pulse_population_jump_relative", 0.0)),
                        "单位": "%",
                    },
                    {
                        "指标": "最终脉冲求解器与平均探针输出差",
                        "值": 100.0
                        * float(
                            population_summary.get(
                                "reported_vs_average_probe_relative_difference", 0.0
                            )
                        ),
                        "单位": "%",
                    },
                    {
                        "指标": "平均稳态自洽迭代数",
                        "值": int(population_summary.get("iterations", 0)),
                        "单位": "次（不是周期）",
                    },
                ]
            )
        elif dynamics == "reduced_periodic_steady_state":
            proxy_settings = population_summary.get("seed_map_settings", {})
            other_metric_rows.extend(
                [
                    {
                        "指标": "周期代理模型",
                        "值": proxy_settings.get("field_model"),
                        "单位": "",
                    },
                    {
                        "指标": "周期代理时间采样 / 通量包",
                        "值": proxy_settings.get("temporal_bins"),
                        "单位": "个",
                    },
                    {
                        "指标": "周期代理频谱采样",
                        "值": proxy_settings.get("spectral_bins"),
                        "单位": "个",
                    },
                    {
                        "指标": "代理图与全场脉后布居差",
                        "值": 100.0
                        * float(
                            population_summary.get(
                                "reported_vs_reduced_proxy_population_relative_difference",
                                0.0,
                            )
                        ),
                        "单位": "%",
                    },
                    {
                        "指标": "代理图与全场输出能量差",
                        "值": 100.0
                        * float(
                            population_summary.get(
                                "reported_vs_reduced_proxy_output_relative_difference",
                                0.0,
                            )
                        ),
                        "单位": "%",
                    },
                ]
            )
        st.dataframe(
            other_metric_rows,
            hide_index=True,
            width="stretch",
        )
        if dynamics == "cycle_averaged_steady_state":
            st.caption(
                "页面顶部的最终输出平均功率 = 在已收敛平均布居上，用当前 FN/JOSA/CPA "
                "求解器得到的一发脉冲能量 × 重复频率；“平均光子流自洽探针”是稳态迭代内部的连续平均场结果。"
            )
        elif dynamics == "reduced_periodic_steady_state":
            st.caption(
                "周期固定点由所选的降阶脉冲代理图收敛；页面输出来自同一"
                "周期边界布居上的当前相干传播。上面的两项代理—最终传播"
                "差异是该加速闭合能否定量使用的强制门。"
            )

    validity = summary.get("physics_validity", {})
    population_correction = population_summary.get(
        "numerical_population_correction", {}
    )
    if validity or population_correction:
        with st.expander("查看模型适用性与布居数值修正", expanded=False):
            propagation_validity = validity.get("propagation", {}) or {}
            propagation_beams = propagation_validity.get("beams", {}) or {}
            seed_propagation = propagation_beams.get("seed", {}) or {}
            pump_propagation = propagation_beams.get("pump", {}) or {}
            st.dataframe(
                stringify_table_column([
                    {
                        "诊断": "脉冲/寿命适用性",
                        "数值": validity.get("status", "—"),
                    },
                    {
                        "诊断": "传播适用性",
                        "数值": propagation_validity.get("status", "—"),
                    },
                    {
                        "诊断": "Seed 最大合成倾角",
                        "数值": seed_propagation.get(
                            "maximum_axis_angle_deg"
                        ),
                    },
                    {
                        "诊断": "Seed FFT Nyquist 占比",
                        "数值": seed_propagation.get(
                            "fft_transverse_nyquist_fraction"
                        ),
                    },
                    {
                        "诊断": "Pump 最大合成倾角",
                        "数值": pump_propagation.get(
                            "maximum_axis_angle_deg"
                        ),
                    },
                    {
                        "诊断": "Pump FFT Nyquist 占比",
                        "数值": pump_propagation.get(
                            "fft_transverse_nyquist_fraction"
                        ),
                    },
                    {
                        "诊断": "最大脉冲 FWHM / 弛豫时间",
                        "数值": validity.get(
                            "maximum_pulse_to_relaxation_ratio"
                        ),
                    },
                    {
                        "诊断": "脉冲内弛豫已解析",
                        "数值": validity.get("intrapulse_relaxation_resolved"),
                    },
                    {
                        "诊断": "最大单格 Pump 吸收光学厚度 α·Δs",
                        "数值": summary["pump"].get(
                            "nominal_maximum_pump_absorption_optical_depth"
                        ),
                    },
                    {
                        "诊断": "Pump 晶体功率账本相对残差",
                        "数值": summary["pump"].get(
                            "crystal_power_balance_residual_relative"
                        ),
                    },
                    {
                        "诊断": "最大负布居截断比例",
                        "数值": population_correction.get(
                            "maximum_negative_population_fraction"
                        ),
                    },
                    {
                        "诊断": "归一化前最大粒子数相对误差",
                        "数值": population_correction.get(
                            "maximum_conservation_error_before_renormalization_relative"
                        ),
                    },
                    {
                        "诊断": "发生负布居截断的调用次数",
                        "数值": population_correction.get(
                            "calls_with_negative_clipping"
                        ),
                    },
                ], "数值"),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "这些值只审计当前模型，不会静默切换后端。明显的布居修正通常意味着时间步过大、"
                "非线性速率过强或自定义能级网络需要重新检查。"
            )

    source_spatio = summary["signal"].get("source_spatio_spectral") or {}
    if source_spatio.get("wavelength_samples"):
        with st.expander("查看 Seed 各波长在增益入口的空间状态", expanded=False):
            st.caption(
                "半径为采样场的二阶矩等效 1/e² 半径；ABCD 混合模式同时报告解析 q 参数结果。"
                "空间模重叠为 1 表示该波长与参考波长的横向复场完全相同（忽略总能量）。"
            )
            rows = []
            for sample in source_spatio["wavelength_samples"]:
                entrance = sample.get("gain_entrance") or {}
                overlap = entrance.get("mode_overlap_with_reference")
                rows.append(
                    {
                        "λ (nm)": float(sample["wavelength_m"]) * 1e9,
                        "源 wx (µm)": float(sample["waist_x_m"]) * 1e6,
                        "源 wy (µm)": float(sample["waist_y_m"]) * 1e6,
                        "入口 wx (µm)": None if entrance.get("radius_x_m") is None else float(entrance["radius_x_m"]) * 1e6,
                        "入口 wy (µm)": None if entrance.get("radius_y_m") is None else float(entrance["radius_y_m"]) * 1e6,
                        "入口中心 x (µm)": None if entrance.get("center_x_m") is None else float(entrance["center_x_m"]) * 1e6,
                        "入口中心 y (µm)": None if entrance.get("center_y_m") is None else float(entrance["center_y_m"]) * 1e6,
                        "与参考空间模重叠": None if overlap is None else float(overlap),
                        "入口采样比例": entrance.get("sampled_fraction"),
                    }
                )
            st.dataframe(rows, hide_index=True, width="stretch")

    gain_transport = summary["signal"].get("gain_transport") or {}
    if gain_transport:
        backend = str(gain_transport.get("backend", "unknown"))
        with st.expander(
            "查看实际增益输运后端",
            expanded=(
                backend
                in {
                    "josa_transport",
                    "causal_multipass_cfl_frantz_nodvik",
                }
                or bool(gain_transport.get("fallback"))
            ),
        ):
            if gain_transport.get("fallback"):
                st.warning(
                    "JOSA 子模块已回滚，本次结果实际使用 spectral_time / Frantz–Nodvik："
                    f"{gain_transport.get('reason', 'unknown reason')}"
                )
            elif backend == "causal_multipass_cfl_frantz_nodvik":
                multipass_transport = gain_transport.get("multipass", {})
                transport_columns = st.columns(4)
                transport_columns[0].metric(
                    "模板", str(multipass_transport.get("template", "unknown"))
                )
                transport_columns[1].metric(
                    "晶体命中",
                    int(multipass_transport.get("pass_count", 0)),
                )
                transport_columns[2].metric(
                    "自动检测重叠",
                    "有" if multipass_transport.get("overlap_detected") else "无",
                )
                transport_columns[3].metric(
                    "最多同时活动程",
                    int(
                        multipass_transport.get(
                            "maximum_simultaneous_pass_channels", 0
                        )
                    ),
                )
                st.caption(
                    "重叠来自模板光程、脉冲包络与逐程二维命中；表中的对齐位移是"
                    "模板主光线/论文边界条件与传播后复包络之间的注册量，不改变能量。"
                )
                compiled_hits = multipass_transport.get("compiled_hits", [])
                if compiled_hits:
                    moment_rows = {
                        int(row.get("pass", 0)): row
                        for row in multipass_transport.get(
                            "per_pass_beam_moments", []
                        )
                    }
                    st.dataframe(
                        [
                            {
                                "程": row.get("pass"),
                                "名称": row.get("name"),
                                "方向": row.get("direction"),
                                "中心 x (mm)": float(row.get("center_x_m", 0.0))
                                * 1e3,
                                "中心 y (mm)": float(row.get("center_y_m", 0.0))
                                * 1e3,
                                "θx (°)": row.get("incidence_angle_x_deg"),
                                "θy (°)": row.get("incidence_angle_y_deg"),
                                "cosθ": row.get("path_cosine"),
                                "入口 wx (mm)": (
                                    None
                                    if moment_rows.get(int(row.get("pass", 0)), {})
                                    .get("gain_entry", {})
                                    .get("radius_x_m")
                                    is None
                                    else float(
                                        moment_rows[int(row.get("pass", 0))][
                                            "gain_entry"
                                        ]["radius_x_m"]
                                    )
                                    * 1e3
                                ),
                                "入口 wy (mm)": (
                                    None
                                    if moment_rows.get(int(row.get("pass", 0)), {})
                                    .get("gain_entry", {})
                                    .get("radius_y_m")
                                    is None
                                    else float(
                                        moment_rows[int(row.get("pass", 0))][
                                            "gain_entry"
                                        ]["radius_y_m"]
                                    )
                                    * 1e3
                                ),
                            }
                            for row in compiled_hits
                        ],
                        hide_index=True,
                        width="stretch",
                    )
                transitions = multipass_transport.get("transitions", [])
                if transitions:
                    st.dataframe(
                        [
                            {
                                "程间": f"{row.get('from_pass')}→{row.get('to_pass')}",
                                "几何路长 (m)": row.get(
                                    "geometric_path_length_m"
                                ),
                                "精确延时 (ns)": float(
                                    row.get("exact_external_delay_s", 0.0)
                                )
                                * 1e9,
                                "量化误差 (ps)": float(
                                    row.get("delay_quantization_error_s", 0.0)
                                )
                                * 1e12,
                                "路由前透过率": row.get(
                                    "power_transmission_before_route"
                                ),
                                "命中合同": row.get("alignment_contract"),
                            }
                            for row in transitions
                        ],
                        hide_index=True,
                        width="stretch",
                    )
            elif backend == "gaussian_mode_overlap":
                axial_samples = gain_transport.get("axial_samples", [])
                modal_columns = st.columns(4)
                modal_columns[0].metric("空间闭合", "高斯模态重叠")
                modal_columns[1].metric(
                    "增益导引",
                    "关闭" if not gain_transport.get("gain_guiding", False) else "开启",
                )
                modal_columns[2].metric("z 样本", len(axial_samples))
                modal_columns[3].metric(
                    "最大 g·Δz",
                    f"{float(gain_transport.get('maximum_cell_gain_exponent', 0.0)):.5g}",
                )
                st.caption(
                    "光束仍按当前衍射/热相位模型传播；每个 z 单元把 3D 局域增益与当前高斯模态"
                    "做强度重叠，只用一个标量更新整束功率，因此不会由增益本身重塑光斑。"
                )
                if axial_samples:
                    rows = []
                    for sample in axial_samples:
                        active_gain = sample.get("effective_active_gain_m_inv", 0.0)
                        total_gain = sample.get("effective_total_gain_m_inv", 0.0)
                        if isinstance(active_gain, list):
                            active_gain = active_gain[0] if active_gain else 0.0
                        if isinstance(total_gain, list):
                            total_gain = total_gain[0] if total_gain else 0.0
                        rows.append(
                            {
                                "程": int(sample.get("pass", 0)),
                                "z (mm)": float(sample.get("z_m", 0.0)) * 1e3,
                                "P in (W)": sample.get("input_average_power_W"),
                                "P out (W)": sample.get("output_average_power_W"),
                                "E in (J)": sample.get("input_energy_J"),
                                "E out (J)": sample.get("output_energy_J"),
                                "g active (1/m)": float(active_gain),
                                "g net (1/m)": float(total_gain),
                                "wx (µm)": None
                                if sample.get("radius_x_m") is None
                                else float(sample["radius_x_m"]) * 1e6,
                                "wy (µm)": None
                                if sample.get("radius_y_m") is None
                                else float(sample["radius_y_m"]) * 1e6,
                            }
                        )
                    st.dataframe(rows, hide_index=True, width="stretch")
            elif backend == "josa_transport":
                mode = str(gain_transport.get("mode", "unknown"))
                mode_names = {
                    "explicit_cfl": "论文 Explicit / CFL",
                    "pc_cfl": "论文 PC / CFL",
                    "retarded_pc": "延迟时间 PC",
                    JOSA_FULL_3D_MODE: "3D PC（群延迟相位闭合近似）",
                }
                transport_columns = st.columns(4)
                transport_columns[0].metric("实际模式", mode_names.get(mode, mode))
                transport_columns[1].metric(
                    "时间分箱",
                    int(
                        gain_transport.get("cfl_resampling", {}).get(
                            "transport_temporal_bins",
                            summary["signal"].get("temporal_samples", 0),
                        )
                    ),
                )
                transport_columns[2].metric(
                    "最大 |g·Δz|",
                    f"{float(gain_transport.get('maximum_abs_gain_step', 0.0)):.4g}",
                )
                transport_columns[3].metric(
                    "布居守恒相对误差",
                    f"{float(gain_transport.get('maximum_population_conservation_error_relative', 0.0)):.3g}",
                )
                time_columns = st.columns(3)
                time_columns[0].metric(
                    "论文 CFL Δt",
                    f"{float(gain_transport.get('paper_cfl_dt_s', 0.0)) * 1e15:.5g} fs",
                )
                time_columns[1].metric(
                    "实际时间箱间隔",
                    f"{float(gain_transport.get('median_temporal_bin_spacing_s', 0.0)) * 1e15:.5g} fs",
                )
                time_columns[2].metric(
                    "报告 Courant 数",
                    f"{float(gain_transport.get('courant_number_from_reported_bins', 0.0)):.5g}",
                )
                coordinate_note = (
                    "严格论文网格：Δt=Δz/vg，进入晶体后按特征线推进。"
                    if mode.endswith("_cfl")
                    else "延迟时间坐标 T=t−z/vg：时间分辨率与 Nz 独立，仍按 front → peak → tail 耗尽布居。"
                )
                st.caption(coordinate_note)
                phase_contract = gain_transport.get("phase_contract", {})
                if phase_contract:
                    phase_columns = st.columns(4)
                    phase_columns[0].metric(
                        "输入相位→时间强度",
                        "已应用"
                        if phase_contract.get(
                            "source_and_stretcher_phase_applied", False
                        )
                        else "未应用",
                    )
                    phase_columns[1].metric(
                        "传播中的相干谱相位",
                        "保留"
                        if phase_contract.get(
                            "coherent_frequency_phase_propagated", False
                        )
                        else "不保留",
                    )
                    phase_columns[2].metric(
                        "材料/光路 GDD·TOD·FOD",
                        "传播"
                        if phase_contract.get(
                            "material_or_train_gdd_tod_fod_propagated", False
                        )
                        else "不传播",
                    )
                    phase_columns[3].metric(
                        "Kerr / SPM",
                        "传播"
                        if phase_contract.get("kerr_or_spm_propagated", False)
                        else "不传播",
                    )
                    st.caption(
                        "JOSA 只在入口把可用的源/展宽器相位投影成时间强度；"
                        "进入输运后，相位不会随时间箱和波长箱继续演化。"
                    )
                contract = gain_transport.get("full_3d_rate_contract", {})
                if mode == JOSA_FULL_3D_MODE:
                    if contract.get("satisfied"):
                        st.success("3D 速率方程模块合同已通过；本次没有退回 FN。")
                    else:
                        st.error("完整 3D 合同未满足；该结果不应作为完整模式解释。")
                    phase_closure = gain_transport.get(
                        "group_delay_walkoff", {}
                    ).get("phase_closure", {})
                    if phase_closure.get("approximation_active"):
                        st.warning(
                            "群延迟重映射发生多源时间箱混合：能量守恒，但在后续 FFT 衍射前使用了局部相位矩闭合；"
                            "这不是严格的逐非相干通道传播。"
                        )
                    with st.expander("查看完整 3D 耦合审计", expanded=True):
                        check_names = {
                            "optical_propagation": "Seed FFT",
                            "pump_propagation": "Pump FFT",
                            "population_dynamics": "瞬态布居",
                            "pump_depletion": "泵浦耗尽",
                            "gain_saturation": "受激布居耗尽",
                            "thermal_model": "三维热场",
                            "refractive_index_model": "热致折射率",
                            "nonlinear_phase_model": "相位模型边界",
                            "record_optical_planes": "光学面记录",
                        }
                        st.dataframe(
                            stringify_table_column([
                                {
                                    "耦合项": check_names.get(key, key),
                                    "通过": bool(passed),
                                    "实际设置": contract.get("effective_physics", {}).get(key),
                                }
                                for key, passed in contract.get("checks", {}).items()
                            ], "实际设置"),
                            hide_index=True,
                            width="stretch",
                        )
                        group_delay = gain_transport.get("group_delay_walkoff", {})
                        incidence = gain_transport.get("incidence_walkoff", {})
                        walkoff_columns = st.columns(3)
                        walkoff_columns[0].metric(
                            "最大群延迟差/程",
                            f"{float(group_delay.get('maximum_relative_delay_per_pass_s', 0.0)) * 1e15:.5g} fs",
                        )
                        walkoff_columns[1].metric(
                            "时间窗边界最大损失",
                            f"{100.0 * float(group_delay.get('maximum_energy_fraction_lost_at_time_window', 0.0)):.4g}%",
                        )
                        largest_shift = max(
                            [
                                math.hypot(float(item.get("shift_x_m", 0.0)), float(item.get("shift_y_m", 0.0)))
                                for item in incidence.get("per_pass", [])
                            ]
                            or [0.0]
                        )
                        walkoff_columns[2].metric("最大斜入射横移/程", f"{largest_shift * 1e6:.5g} µm")
                        st.caption(
                            "完整 3D 的范围是强度/光子输运 + 真实能级速率方程；"
                            "相干光谱相位、GDD/TOD 压缩与 Kerr/SPM 需要 coherent_cpa。"
                        )
            else:
                st.caption(
                    f"实际增益输运：{backend}；"
                    f"时间分箱 {int(gain_transport.get('temporal_bins', 0))}，"
                    f"光谱分箱 {int(gain_transport.get('spectral_bins', 0))}。"
                )

    convergence_history = population_summary.get("convergence_history", [])
    if dynamics in {
        "periodic_steady_state",
        "reduced_periodic_steady_state",
    } and convergence_history:
        reduced_periodic = dynamics == "reduced_periodic_steady_state"
        with st.expander(
            "查看低维代理周期稳态收敛"
            if reduced_periodic
            else "查看周期稳态收敛"
        ):
            cycle_values = [int(row["cycle"]) for row in convergence_history]
            raw_energy_values = [
                finite_plot_value(row.get("output_energy_J"))
                for row in convergence_history
            ]
            maximum_history_energy = max(
                (abs(value) for value in raw_energy_values if value is not None),
                default=0.0,
            )
            _, history_energy_unit, history_energy_scale = scaled_energy(maximum_history_energy)
            energy_values = [
                None if value is None else value * history_energy_scale
                for value in raw_energy_values
            ]
            residual_plot = [
                value
                if value is not None and value > 0.0
                else None
                for value in map(convergence_residual_value, convergence_history)
            ]
            convergence_figure = go.Figure()
            convergence_figure.add_trace(go.Scatter(
                x=cycle_values, y=energy_values, mode="lines+markers",
                name="Seed 输出能量", yaxis="y",
            ))
            convergence_figure.add_trace(go.Scatter(
                x=cycle_values, y=residual_plot, mode="lines+markers",
                name="最大相对残差", yaxis="y2",
            ))
            convergence_figure.update_layout(
                xaxis_title=(
                    "低维 Pump + Seed 周期代理图评估次数"
                    if reduced_periodic
                    else "完整 Pump + Seed 周期映射评估次数"
                ),
                yaxis={"title": f"输出脉冲能量 ({history_energy_unit})"},
                yaxis2={
                    "title": "相对残差", "type": "log", "overlaying": "y", "side": "right",
                    "showgrid": False,
                },
                legend={"orientation": "h", "y": 1.12},
                margin={"t": 55},
            )
            st.plotly_chart(convergence_figure, width="stretch")
            if any(value is None for value in residual_plot):
                st.caption(
                    "残差曲线的断点表示该轮没有可定义的比较基准（通常是首轮尚无上一轮输出），"
                    "不是残差为 0。"
                )
            carrier_gate_text = (
                "、所选载流子影响残差"
                if bool(
                    population_summary.get(
                        "free_carrier_periodic_state_enabled", False
                    )
                )
                else ""
            )
            st.caption(
                f"收敛判据：布居、输出能量{carrier_gate_text}的残差都 ≤ "
                f"{float(population_summary.get('relative_tolerance', 0.0)):.3g}，"
                f"并连续满足 {int(population_summary.get('consecutive_cycles_required', 1))} 次映射评估。"
            )
            acceleration = population_summary.get("acceleration", {})
            initializer = population_summary.get("initializer", {})
            if initializer.get("used", False):
                if initializer.get("requested") == "reduced_pulse_map":
                    reduced_stats = initializer.get("reduced_map", {})
                    st.caption(
                        "初值：低维离散脉冲图，"
                        f"{int(reduced_stats.get('cycles_completed', 0))} 次映射评估；"
                        "随后已由当前完整主脉冲 pulse map 继续收敛，"
                        "reduced 结果没有作为最终结果。"
                    )
                else:
                    st.caption(
                        "初值：周期平均代数稳态，"
                        f"{int(initializer.get('iterations', 0))} 次自洽迭代；"
                        "它只提供初值，最终解仍满足精确周期映射。"
                    )
            elif initializer.get("failure"):
                st.warning(
                    "周期平均初值求解失败，已安全退回设定初始布居；"
                    + (
                        "低维离散脉冲代理图"
                        if reduced_periodic
                        else "精确周期映射"
                    )
                    + f"仍继续运行。原因：{initializer['failure']}"
                )
            elif population_summary.get("initialization_source") == "thermal_outer_warm_start":
                st.caption("初值：复用上一轮热光固定点的已收敛周期边界布居。")
            if acceleration.get("enabled", False):
                st.caption(
                    "Anderson 加速："
                    f"接受 {int(acceleration.get('accepted_steps', 0))} 步，"
                    f"拒绝 {int(acceleration.get('rejected_steps', 0))} 步，"
                    f"自动重启 {int(acceleration.get('restart_count', 0))} 次；"
                    "被拒绝的候选会退回原始周期映射。"
                )
    elif dynamics == "cycle_averaged_steady_state" and convergence_history:
        with st.expander("查看平均光子流稳态收敛"):
            iteration_values = [int(row["iteration"]) for row in convergence_history]
            power_values = [
                finite_plot_value(row.get("average_output_power_W"))
                for row in convergence_history
            ]
            residual_plot = [
                value
                if value is not None and value > 0.0
                else None
                for value in map(convergence_residual_value, convergence_history)
            ]
            convergence_figure = go.Figure()
            convergence_figure.add_trace(go.Scatter(
                x=iteration_values,
                y=power_values,
                mode="lines+markers",
                name="平均 Seed 输出功率",
                yaxis="y",
            ))
            convergence_figure.add_trace(go.Scatter(
                x=iteration_values,
                y=residual_plot,
                mode="lines+markers",
                name="最大相对残差",
                yaxis="y2",
            ))
            convergence_figure.update_layout(
                xaxis_title="Pump + Seed 自洽迭代（不是物理周期）",
                yaxis={"title": "平均输出功率 (W)"},
                yaxis2={
                    "title": "相对残差",
                    "type": "log",
                    "overlaying": "y",
                    "side": "right",
                    "showgrid": False,
                },
                legend={"orientation": "h", "y": 1.12},
                margin={"t": 55},
            )
            st.plotly_chart(convergence_figure, width="stretch")
            if any(value is None for value in residual_plot):
                st.caption(
                    "残差曲线的断点表示该轮没有可定义的比较基准（通常是首轮尚无上一轮输出），"
                    "不是残差为 0。"
                )
            st.caption(
                "每一步都重新传播平均 Pump/Seed 光子流并求局部代数布居；"
                f"停止容差为 {float(population_summary.get('relative_tolerance', 0.0)):.3g}。"
            )
            acceleration = population_summary.get("acceleration", {})
            if acceleration.get("enabled", False):
                st.caption(
                    "Anderson 加速："
                    f"接受 {int(acceleration.get('accepted_steps', 0))} 步，"
                    f"拒绝 {int(acceleration.get('rejected_steps', 0))} 步；"
                    "每一步仍由完整平均光子流闭合残差验证。"
                )

    with path_tab:
        st.caption(
            "冷态实线/半透明光束体来自运行前 q 参数；运行后虚线和采样环来自求解器实际复场的二阶矩，"
            "因此包含本次计算采用的材料折射率、热场 Δn、增益、孔径裁剪和 FFT 传播。"
            "采样面之间仅连线辅助阅读，不宣称是连续热态 q 解。"
        )
        active_result_beam = choice_radio(
            "当前详细标签（Pump 与 Seed 光束体始终同时显示）",
            ["seed", "pump"],
            horizontal=True,
            key=result_key("result_3d_active_beam"),
        )
        result_focus_gain = choice_toggle(
            "晶体聚焦视图（压缩长自由空间）",
            value=True,
            key=result_key("result_3d_focus_gain"),
        )
        path_overlay_fields = {
            "泵后反转密度": (result.inversion_after_pump_m3, "m⁻³", "RdBu_r", True),
            "泵浦强度": (result.pump_intensity_W_m2, "W/m²", "Turbo", False),
            "热源功率密度": (result.heat_W_m3, "W/m³", "Hot", False),
            "温升": (
                result.temperature_K
                - float(
                    summary.get("thermal", {}).get(
                        "reference_temperature_K",
                        cfg.get("material", {}).get(
                            "operating_temperature_K", 293.15
                        ),
                    )
                ),
                "K",
                "Inferno",
                False,
            ),
            "末程 seed 通量": (result.signal_fluence_J_m2, "J/m²", "Viridis", False),
            "seed 净提取能量密度": (result.extracted_energy_J_m3, "J/m³", "RdBu_r", True),
        }
        overlay_name = choice_selectbox(
            "晶体内叠加场（映射到第一程晶体）",
            list(path_overlay_fields),
            key=result_key("result_3d_overlay_field"),
        )
        overlay_values, overlay_unit, overlay_colorscale, overlay_signed = path_overlay_fields[overlay_name]
        result_plane_summaries = list(
            summary.get("optical_train", {}).get("planes", [])
        )
        result_gain_path_samples = list(
            summary.get("optical_train", {}).get("gain_path_samples", [])
        )
        boundary_window = summary.get("optical_train", {}).get(
            "boundary_window_diagnostics", {}
        )
        boundary_rows = []
        for beam_kind, diagnostic in boundary_window.get("beams", {}).items():
            boundary_rows.append(
                {
                    "光束": beam_kind,
                    "外缘能量占比": diagnostic.get(
                        "maximum_boundary_energy_fraction"
                    ),
                    "周期 FFT 合同适用": diagnostic.get(
                        "periodic_fft_applies"
                    ),
                    "外缘占比超阈值": diagnostic.get("at_risk"),
                    "程次": diagnostic.get("pass_index"),
                    "z 分箱": diagnostic.get("z_bin_index"),
                    "采样语义": diagnostic.get("sample_location"),
                }
            )
        if boundary_rows:
            st.dataframe(boundary_rows, hide_index=True, width="stretch")
            st.caption(
                "外缘能量占比取晶体路径上最坏截面；超过 0.1% 时提示扩大光学窗口，"
                "只有所选后端在该路径上实际使用周期 FFT 时才会判为绕回风险；"
                "非周期后端保留占比供诊断，但不会误判为 FFT 失效。"
            )
        previews: list[BenchPreview] = []
        for beam_kind in ("seed", "pump"):
            try:
                previews.append(build_bench_preview(cfg, beam_kind))
            except Exception as exc:
                st.warning(f"{beam_kind} 三维光路暂不可用：{type(exc).__name__}: {exc}")
        if previews:
            result_path_3d, result_path_2d = st.tabs(
                ["三维光路（拖动旋转）", "二维冷态/运行后 D4σ"]
            )
            with result_path_3d:
                st.plotly_chart(
                    optical_path_3d_figure(
                        previews,
                        cfg,
                        result=result,
                        field_values=overlay_values,
                        field_name=overlay_name,
                        unit=overlay_unit,
                        colorscale=overlay_colorscale,
                        signed=overlay_signed,
                        active_beam_kind=active_result_beam,
                        focus_gain=result_focus_gain,
                        optical_trace_summaries=result_plane_summaries,
                        optical_gain_path_samples=result_gain_path_samples,
                    ),
                    width="stretch",
                    key=result_key("result_optical_path_3d"),
                    config={"scrollZoom": True, "displaylogo": False},
                )
                st.caption(
                    "半透明连续光束体是冷态 q 预览；虚线椭圆环是运行后实际 D4σ 截面，"
                    "晶体内部按光学 z 分箱自动采样：Pump 为单元平均并放在中心，"
                    "Seed 为每个 split-step 单元出口。"
                    "当前选择的 Seed/Pump 最后绘制，不再被另一束遮住。"
                )
            with result_path_2d:
                st.plotly_chart(
                    combined_bench_preview_figure(
                        previews,
                        cfg,
                        focus_gain=result_focus_gain,
                        active_beam_kind=active_result_beam,
                        optical_trace_summaries=result_plane_summaries,
                        optical_gain_path_samples=result_gain_path_samples,
                    ),
                    width="stretch",
                    key=result_key("result_optical_path_2d"),
                    config={"scrollZoom": True, "displaylogo": False},
                )
                if result_plane_summaries or result_gain_path_samples:
                    st.caption(
                        "细实线为计算前冷态 q 包络；粗虚线和空心圆为运行后实际复场 D4σ。"
                        "晶体内部按光学 z 分箱自动采样：Pump 为单元平均并放在中心，"
                        "Seed 为每个 split-step 单元出口。"
                    )
                else:
                    st.info(
                        "本次结果没有可用的实际 D4σ 采样，因此只能显示冷态 q 包络。"
                    )
        else:
            st.info("seed 和 pump 光路都暂时不可用。")
    with spatial_tab:
        population_summary = summary.get("population", {})
        configured_density = float(
            cfg.get("material", {}).get("ion_density_m3", 0.0)
        )
        density = (
            np.full(result.grid.shape, configured_density, dtype=float)
            if result.dopant_density_m3 is None
            else np.asarray(result.dopant_density_m3, dtype=float)
        )
        upper_level_name = population_summary.get("upper_level")
        if upper_level_name in result.level_names:
            upper_level = result.level_names.index(str(upper_level_name))
        else:
            upper_level = min(1, len(result.level_names) - 1)
            st.warning(
                "结果摘要没有可匹配的上能级名称；上能级占比图临时使用保存布居数组的"
                f"第 {upper_level + 1} 层，其它三维场不受影响。"
            )
        reference_temperature = float(
            summary.get("thermal", {}).get(
                "reference_temperature_K",
                cfg.get("material", {}).get("operating_temperature_K", 293.15),
            )
        )
        volume_fields: dict[str, dict[str, Any]] = {
            "泵后反转密度 Nu−Nl": {
                "values": result.inversion_after_pump_m3,
                "unit": "m⁻³",
                "colorscale": "RdBu_r",
                "signed": True,
            },
            "泵后上能级占比": {
                "values": np.divide(
                    result.population_after_pump_m3[upper_level],
                    density,
                    out=np.zeros_like(density),
                    where=density > 0.0,
                ),
                "unit": "fraction",
                "colorscale": "Turbo",
                "signed": False,
            },
            "掺杂离子密度": {
                "values": density,
                "unit": "m⁻³",
                "colorscale": "Viridis",
                "signed": False,
            },
            "泵浦强度": {
                "values": result.pump_intensity_W_m2,
                "unit": "W/m²",
                "colorscale": "Turbo",
                "signed": False,
            },
            "泵浦吸收功率密度": {
                "values": result.pump_absorbed_W_m3,
                "unit": "W/m³",
                "colorscale": "Inferno",
                "signed": False,
            },
            "热源功率密度": {
                "values": result.heat_W_m3,
                "unit": "W/m³",
                "colorscale": "Hot",
                "signed": False,
            },
            "种子净提取能量密度": {
                "values": result.extracted_energy_J_m3,
                "unit": "J/m³",
                "colorscale": "RdBu_r",
                "signed": True,
            },
            "种子正提取能量密度": {
                "values": np.maximum(result.extracted_energy_J_m3, 0.0),
                "unit": "J/m³",
                "colorscale": "Plasma",
                "signed": False,
            },
            "末程种子通量": {
                "values": result.signal_fluence_J_m2,
                "unit": "J/m²",
                "colorscale": "Viridis",
                "signed": False,
            },
            "末程种子等效强度 F/τ": {
                "values": result.signal_fluence_J_m2 / max(float(cfg["seed"]["pulse_duration_s"]), np.finfo(float).tiny),
                "unit": "W/m²",
                "colorscale": "Turbo",
                "signed": False,
            },
            "种子后局部增益": {
                "values": result.gain_m_inv,
                "unit": "m⁻¹",
                "colorscale": "RdBu_r",
                "signed": True,
            },
            "温升": {
                "values": result.temperature_K - reference_temperature,
                "unit": "K",
                "colorscale": "Inferno",
                "signed": False,
            },
            "绝对温度": {
                "values": result.temperature_K,
                "unit": "K",
                "colorscale": "Inferno",
                "signed": False,
            },
            "热/应力折射率变化 Δn": {
                "values": result.delta_n,
                "unit": "",
                "colorscale": "RdBu_r",
                "signed": True,
            },
        }
        if result.pump_active_absorbed_W_m3 is not None:
            volume_fields["Pump 主动吸收功率密度"] = {
                "values": result.pump_active_absorbed_W_m3,
                "unit": "W/m³",
                "colorscale": "Inferno",
                "signed": True,
            }
        if result.pump_passive_absorbed_W_m3 is not None:
            volume_fields["Pump 被动吸收功率密度"] = {
                "values": result.pump_passive_absorbed_W_m3,
                "unit": "W/m³",
                "colorscale": "Hot",
                "signed": False,
            }
        if result.signal_fluence_cumulative_J_m2 is not None:
            volume_fields["各程累计种子通量"] = {
                "values": result.signal_fluence_cumulative_J_m2,
                "unit": "J/m²",
                "colorscale": "Viridis",
                "signed": False,
            }
        if result.signal_fluence_per_pass_J_m2 is not None:
            for pass_index, pass_volume in enumerate(result.signal_fluence_per_pass_J_m2):
                volume_fields[f"第 {pass_index + 1} 程种子通量"] = {
                    "values": pass_volume,
                    "unit": "J/m²",
                    "colorscale": "Viridis",
                    "signed": False,
                }
        if result.ase_intensity_W_m2 is not None:
            volume_fields["ASE 总强度"] = {
                "values": result.ase_intensity_W_m2,
                "unit": "W/m²",
                "colorscale": "Inferno",
                "signed": False,
            }
        if result.nonlinear_loss_energy_J_m3 is not None and np.any(result.nonlinear_loss_energy_J_m3):
            volume_fields["Seed 强场不可逆损耗"] = {
                "values": result.nonlinear_loss_energy_J_m3,
                "unit": "J/m³/pulse",
                "colorscale": "Hot",
                "signed": False,
            }
        if result.signal_passive_loss_energy_J_m3 is not None and np.any(result.signal_passive_loss_energy_J_m3):
            volume_fields["Seed 被动吸收"] = {
                "values": result.signal_passive_loss_energy_J_m3,
                "unit": "J/m³/pulse",
                "colorscale": "Oranges",
                "signed": False,
            }
        for level_index, level_name in enumerate(result.level_names):
            volume_fields[f"泵后布居 N[{level_name}]"] = {
                "values": result.population_after_pump_m3[level_index],
                "unit": "m⁻³",
                "colorscale": "Turbo",
                "signed": False,
            }
            volume_fields[f"seed 后布居 N[{level_name}]"] = {
                "values": result.population_m3[level_index],
                "unit": "m⁻³",
                "colorscale": "Turbo",
                "signed": False,
            }
        for stress_name, stress_values in result.stress.items():
            if np.asarray(stress_values).shape == result.grid.shape:
                stress_unit = (
                    "Pa"
                    if stress_name.endswith("_Pa")
                    else "m"
                    if stress_name.startswith("displacement_")
                    else ""
                )
                volume_fields[f"应力/热光：{stress_name}"] = {
                    "values": stress_values,
                    "unit": stress_unit,
                    "colorscale": "RdBu_r" if np.any(np.asarray(stress_values) < 0.0) else "Magma",
                    "signed": bool(np.any(np.asarray(stress_values) < 0.0)),
                }
        selector, axis_control, volume_control = st.columns([2, 1, 1])
        with selector:
            field_name = choice_selectbox(
                "三维诊断场",
                list(volume_fields),
                key=result_key("diagnostic_volume_field"),
            )
        with axis_control:
            slice_axis = choice_selectbox(
                "切片方向",
                list(SLICE_AXES),
                key=result_key("diagnostic_slice_axis"),
            )
        with volume_control:
            show_volume = choice_checkbox(
                "显示三维体渲染",
                True,
                key=result_key("diagnostic_show_volume"),
            )
        selected_axis = SLICE_AXES[slice_axis][0]
        axis_coordinates = (result.grid.z, result.grid.y, result.grid.x)[selected_axis]
        slice_index = st.slider(
            f"{slice_axis[0]} 切片位置",
            0,
            len(axis_coordinates) - 1,
            len(axis_coordinates) // 2,
            key=result_key(f"diagnostic_slice_{slice_axis[0]}"),
            help=(
                "拖动后只改变当前已保存三维场的显示切面；不会重新传播 Pump/Seed，"
                "也不会修改结果包。滑块索引会随所选 x/y/z 方向使用相应网格坐标。"
            ),
        )
        selected = volume_fields[field_name]
        selected_values = np.asarray(selected["values"], dtype=float)
        material_mask = (
            np.asarray(result.crystal_mask, dtype=bool)
            if result.crystal_mask is not None
            else np.ones(result.grid.shape, dtype=bool)
        )
        statistics_mask = material_mask & np.isfinite(selected_values)
        if not np.any(statistics_mask):
            statistics_mask = np.isfinite(selected_values)
            if np.any(statistics_mask):
                st.warning(
                    "保存的晶体 mask 没有覆盖任何有限体素；统计与切片暂时回退到全部有限体素。"
                )
            else:
                st.error(
                    "这个已保存体场没有任何有限数值；仍保留切片/体渲染用于定位损坏或数值发散。"
                )
        stats_columns = st.columns(3)
        if np.any(statistics_mask):
            statistics = finite_weighted_statistics(
                selected_values,
                result.grid.cell_volumes,
                statistics_mask,
            )
            for column, label, value in zip(
                stats_columns,
                ("材料域最小值", "材料域体积加权平均值", "材料域最大值"),
                statistics,
                strict=True,
            ):
                column.metric(label, f"{value:.5g} {selected['unit']}")
        else:
            for column, label in zip(
                stats_columns,
                ("材料域最小值", "材料域体积加权平均值", "材料域最大值"),
                strict=True,
            ):
                column.metric(label, "—")
        if show_volume:
            geometry_summary = summary.get("crystal_geometry", {})
            if not geometry_summary:
                geometry_summary = {
                    "shape": "cuboid",
                    "width_x_m": float(np.ptp(result.grid.x_edges)),
                    "height_y_m": float(np.ptp(result.grid.y_edges)),
                }
            if geometry_summary.get("shape") == "cylinder":
                diameter_mm = float(
                    geometry_summary.get(
                        "diameter_m",
                        min(np.ptp(result.grid.x_edges), np.ptp(result.grid.y_edges)),
                    )
                ) * 1e3
                transverse_text = f"{diameter_mm:.5g} × {diameter_mm:.5g}"
            else:
                transverse_text = (
                    f"{float(geometry_summary.get('width_x_m', np.ptp(result.grid.x_edges))) * 1e3:.5g} × "
                    f"{float(geometry_summary.get('height_y_m', np.ptp(result.grid.y_edges))) * 1e3:.5g}"
                )
            crystal_length_mm = float(result.grid.z_edges[-1] - result.grid.z_edges[0]) * 1e3
            st.caption(
                f"体渲染使用晶体真实等比尺寸：x × y × z = {transverse_text} × {crystal_length_mm:.5g} mm；"
                "使用正交相机，不用透视缩放截面。"
            )
            st.plotly_chart(
                volume_figure(
                    selected_values,
                    result,
                    field_name,
                    str(selected["unit"]),
                    str(selected["colorscale"]),
                    bool(selected["signed"]),
                ),
                width="stretch",
                key=result_key(f"volume_{field_name}"),
            )
        left, right = st.columns(2)
        left.plotly_chart(
            slice_figure(
                selected_values,
                result.grid.x,
                result.grid.y,
                result.grid.z,
                slice_axis,
                slice_index,
                field_name,
                str(selected["unit"]),
                str(selected["colorscale"]),
                bool(selected["signed"]),
            ),
            width="stretch",
            key=result_key(
                f"slice_{field_name}_{slice_axis[0]}_{slice_index}"
            ),
        )
        right.plotly_chart(
            axial_field_plot(
                selected_values,
                result.grid.z,
                f"{field_name} · 轴向统计",
                str(selected["unit"]),
                material_mask if np.any(material_mask) else None,
                result.grid.cell_areas,
            ),
            width="stretch",
            key=result_key(f"axial_{field_name}"),
        )
        st.caption(
            "反转密度定义为信号跃迁的 Nu−Nl。由布居求解的增益按 σeNu−σaNl 计算；"
            "直接指定增益时使用独立的可饱和增益储库，因此不能从反转密度图推断该设定增益。"
            "种子净提取能量密度为有符号的主动跃迁能量交换，已剔除设定的被动损耗，并在所有晶体程次上累计。"
            "热源按本次结果配置中的 Pump 主动/被动分量账本或事件能量账本计算；"
            "ASE/TPL、Seed 强场产热和普通 Seed 被动吸收热只在对应开关启用时计入；荧光逃逸仍未做独立输运闭环。"
        )
        with st.expander("固定对比截面"):
            middle = len(result.grid.z) // 2
            left, right = st.columns(2)
            left.plotly_chart(
                heatmap(result.inversion_after_pump_m3[middle], result.grid.x, result.grid.y, "泵后中面反转密度", "m⁻³", "RdBu_r", 0.0),
                width="stretch",
            )
            right.plotly_chart(
                heatmap(result.heat_W_m3[middle], result.grid.x, result.grid.y, "晶体中面热源", "W/m³", "Hot"),
                width="stretch",
            )
            left, right = st.columns(2)
            left.plotly_chart(
                heatmap(result.extracted_energy_J_m3[middle], result.grid.x, result.grid.y, "晶体中面种子净提取", "J/m³", "RdBu_r", 0.0),
                width="stretch",
            )
            right.plotly_chart(
                heatmap(np.max(result.pump_intensity_W_m2, axis=0), result.grid.x, result.grid.y, "沿 z 最大泵浦强度", "W/m²", "Turbo"),
                width="stretch",
            )

    with spectral_tab:
        if result.cpa_diagnostics:
            cpa = result.cpa_diagnostics
            if cpa.get("fallback"):
                st.warning(f"相干 CPA 已安全降级：{cpa.get('reason', 'unknown')}")
            else:
                if cpa.get("model") in {
                    "commercial_app_effective_area_1d_spm",
                    "commercial_app_effective_area_1d_strong_field",
                }:
                    st.info(
                        (
                            "这是 Commercial App L3.5 的一维 effective-area 强场结果：频谱/时间包络"
                            "是相干的，多光子损耗总量使用 Gaussian K 阶矩，载流子/FCA "
                            "使用能量等效标量状态；横向场仍由 Gaussian q 表示。它不等价于"
                            "解析径向载流子与时空耦合的完整 x-y-t coherent CPA。"
                            if cpa.get("model")
                            == "commercial_app_effective_area_1d_strong_field"
                            else "这是 Commercial App 的一维 effective-area SPM 结果：频谱/时间包络是相干的，"
                            "横向场仍由 Gaussian q 表示。它能给出快速展宽趋势，不等价于完整 x-y-t coherent CPA。"
                        )
                    )
                cpa_columns = st.columns(4)
                cpa_columns[0].metric("输入 FWHM", f"{float(cpa.get('input_fwhm_s', 0.0)) * 1e15:.4g} fs")
                cpa_columns[1].metric("放大后 FWHM", f"{float(cpa.get('output_fwhm_s', 0.0)) * 1e15:.4g} fs")
                cpa_columns[2].metric("压缩后 FWHM", f"{float(cpa.get('compressed_fwhm_s', 0.0)) * 1e15:.4g} fs")
                cpa_columns[3].metric(
                    "固定网格累计 B",
                    f"{float(cpa.get('maximum_cumulative_B_integral_rad', cpa.get('maximum_local_B_integral_rad', 0.0))):.4g} rad",
                    help="固定横向网格单元上的 Eulerian 累计值；发生衍射或走离时不等同于某条真实光线的路径积分。",
                )
                local_operator = cpa.get("local_operator", {})
                if local_operator:
                    st.caption(
                        "晶体局部推进："
                        f"{local_operator.get('scheme', 'unknown')}；"
                        f"最多使用 {int(local_operator.get('maximum_substeps_used', 1))} 个子步/物理 z 单元；"
                        f"触及上限：{'是' if local_operator.get('substep_cap_active') else '否'}。"
                    )
                    st.caption(
                        "粒子数更新：" + {
                            "shared_cell": "片内共享、逐次更新",
                            "independent_groups": "空间小片独立消耗，整片结束后按体积汇总",
                        }.get(local_operator.get("spatial_population_update", "shared_cell"), "未知")
                    )
                sampling = cpa.get("sampling", {}) or {}
                if sampling:
                    sampling_columns = st.columns(4)
                    sampling_columns[0].metric(
                        "频谱 FWHM 分箱",
                        f"{float(sampling.get('spectral_fwhm_bins', 0.0)):.3g}",
                    )
                    sampling_columns[1].metric(
                        "TL 脉冲 FWHM 分箱",
                        f"{float(sampling.get('transform_limited_temporal_fwhm_bins', 0.0)):.3g}",
                    )
                    sampling_columns[2].metric(
                        "时间窗",
                        f"{float(sampling.get('time_window_s', 0.0)) * 1e12:.4g} ps",
                    )
                    sampling_columns[3].metric(
                        "建议最少 FFT 点",
                        int(
                            sampling.get(
                                "recommended_minimum_sample_count_at_current_span",
                                cpa.get("sample_count", 0),
                            )
                        ),
                    )
                    if not bool(sampling.get("adequate", False)):
                        st.warning(
                            "相干输入采样合同未通过："
                            + "、".join(sampling.get("failed_contracts", []))
                            + "。细峰、旁瓣和 Kerr 展宽暂不应作定量结论。"
                        )
                if cpa.get("fidelity_mode") == "hybrid_whole_fluence":
                    st.info(
                        "当前是全场 Kerr 中间档：x-y-t 复场、衍射、干涉、色散和 Kerr "
                        "仍被求解；省略的是脉内前沿→尾沿的逐软窗增益耗尽，周期布居若"
                        "选择 cycle-average 也属于平均闭合。"
                    )
                elif cpa.get("fidelity_mode") in {
                    "axisymmetric_kerr",
                    "axisymmetric_transient_gain",
                }:
                    axisymmetric = cpa.get("axisymmetric_reduction", {}) or {}
                    transient_axisymmetric = (
                        cpa.get("fidelity_mode")
                        == "axisymmetric_transient_gain"
                    )
                    retained_order = int(
                        axisymmetric.get(
                            "maximum_retained_azimuthal_order", 0
                        )
                    )
                    if retained_order == 0 and transient_axisymmetric:
                        st.info(
                            "当前是 L4.5 基础轴对称瞬态 Kerr：求解 E(f,r) 与完整相干"
                            "时间波形，并保留 front → peak → tail 软窗增益；所有非零"
                            "方位角空间模式仍被省略。"
                        )
                    elif retained_order == 0:
                        st.info(
                            "当前是 L4 基础轴对称整脉冲 Kerr：求解 E(f,r) 与完整相干"
                            "时间波形，增益采用整脉冲 FN 闭合；所有非零方位角空间模式"
                            "仍被省略。"
                        )
                    elif transient_axisymmetric:
                        st.info(
                            f"当前是 L4.5 有限角向瞬态 Kerr：求解 E(f,r,θ)，保留 "
                            f"m=-{retained_order}…+{retained_order} 与完整相干时间波形，"
                            "front → peak → tail 软窗增益也被保留；更高角向阶数仍被省略。"
                        )
                    else:
                        st.info(
                            f"当前是 L4 有限角向整脉冲 Kerr：求解 E(f,r,θ)，保留 "
                            f"m=-{retained_order}…+{retained_order} 与完整相干时间波形，"
                            "增益采用整脉冲 FN 闭合；更高角向阶数仍被省略。"
                        )
                    axis_columns = st.columns(5)
                    axis_columns[0].metric(
                        "径向分箱",
                        int(axisymmetric.get("radial_bins", 0)),
                    )
                    axis_columns[1].metric(
                        "最大保留 |m|",
                        retained_order,
                    )
                    axis_columns[2].metric(
                        "角向配点",
                        int(axisymmetric.get("angular_samples", 1)),
                    )
                    axis_columns[3].metric(
                        "相干频率/时间点",
                        int(axisymmetric.get("coherent_spectral_bins", 0)),
                    )
                    axis_columns[4].metric(
                        "增益频率点",
                        int(axisymmetric.get("gain_spectral_bins", 0)),
                        help=(
                            "瞬态软窗增益使用原生相干频率网格；整脉冲 FN 档使用"
                            "配置的受控频率分组并在细网格重建增益。"
                        ),
                    )
                    closure_columns = st.columns(6)
                    retained_projection = float(
                        axisymmetric.get(
                            "total_input_projection_energy_fraction",
                            1.0,
                        )
                    )
                    material_residual = max(
                        float(
                            axisymmetric.get(
                                "maximum_population_azimuthal_residual",
                                0.0,
                            )
                        ),
                        float(
                            axisymmetric.get(
                                "maximum_delta_n_azimuthal_residual",
                                0.0,
                            )
                        ),
                        float(
                            axisymmetric.get(
                                "maximum_free_carrier_azimuthal_residual",
                                0.0,
                            )
                        ),
                    )
                    closure_columns[0].metric(
                        "入口投影损失",
                        f"{100.0 * max(0.0, 1.0 - retained_projection):.4g}%",
                    )
                    closure_columns[1].metric(
                        "传播中新生高阶",
                        f"{100.0 * float(axisymmetric.get('maximum_dynamic_field_omitted_energy_fraction', 0.0)):.4g}%",
                    )
                    closure_columns[2].metric(
                        "材料场高阶残差",
                        f"{100.0 * material_residual:.4g}%",
                    )
                    closure_columns[3].metric(
                        "布居可实现修正",
                        f"{100.0 * float(axisymmetric.get('maximum_population_realizability_correction_fraction', 0.0)):.4g}%",
                    )
                    closure_columns[4].metric(
                        "布居非负重分配",
                        f"{100.0 * float(axisymmetric.get('maximum_population_positivity_redistribution_fraction', 0.0)):.4g}%",
                    )
                    closure_columns[5].metric(
                        "廉价 m=2 拟合残差",
                        (
                            f"{100.0 * float(axisymmetric.get('maximum_cheap_m2_material_fit_residual', 0.0)):.4g}%"
                            if axisymmetric.get("material_m2_radial_model")
                            == "quadratic_r_squared"
                            else "—"
                        ),
                    )
                    st.caption(
                        "这六项越接近零，所选角向闭合越可信；可实现修正表示把有限阶"
                        "布居变为非负状态所移动的离子比例，非负重分配表示回写 x-y "
                        "网格时重新安排的跃迁比例。若任一项越过验证门限，应升级到"
                        "下一档，必要时使用完整 x-y。"
                    )
                    boundary_columns = st.columns(4)
                    weighted_boundary = axisymmetric.get(
                        "maximum_energy_weighted_outer_annulus_fraction"
                    )
                    worst_frequency_boundary = axisymmetric.get(
                        "maximum_outer_annulus_energy_fraction"
                    )
                    worst_frequency_weight = axisymmetric.get(
                        "worst_outer_annulus_frequency_energy_fraction"
                    )
                    worst_frequency_wavelength = axisymmetric.get(
                        "worst_outer_annulus_frequency_wavelength_m"
                    )
                    boundary_columns[0].metric(
                        "能量加权边界占比",
                        (
                            "—"
                            if weighted_boundary is None
                            else f"{100.0 * float(weighted_boundary):.4g}%"
                        ),
                    )
                    boundary_columns[1].metric(
                        "最坏单频边界占比",
                        (
                            "—"
                            if worst_frequency_boundary is None
                            else f"{100.0 * float(worst_frequency_boundary):.4g}%"
                        ),
                    )
                    boundary_columns[2].metric(
                        "该频点总能量权重",
                        (
                            "—"
                            if worst_frequency_weight is None
                            else f"{100.0 * float(worst_frequency_weight):.4g}%"
                        ),
                    )
                    boundary_columns[3].metric(
                        "最坏频点波长",
                        (
                            "—"
                            if worst_frequency_wavelength is None
                            else f"{float(worst_frequency_wavelength) * 1e9:.4g} nm"
                        ),
                    )
                    st.caption(
                        "轴对称径向边界阻断只使用总能量加权占比；最坏单频值仅用于定位。"
                        "能量权重可忽略的频谱尾部即使局部比例很高，也不会单独判废整个结果。"
                    )
                transient = cpa.get("transient_gain", {})
                if transient.get("enabled"):
                    if transient.get("status") == "applied":
                        gain_columns = st.columns(3)
                        gain_columns[0].metric("瞬态增益软窗", int(transient.get("largest_slice_count_used", 0)))
                        protection = transient.get("tail_protection", {})
                        if protection.get("enabled", False):
                            st.caption(f"时间尾部保护：中心累计能量 {100 * float(protection.get('energy_coverage', .99)):.3f}%；左尾 1 窗 + 中心 {int(protection.get('central_window_count', 0))} 窗 + 右尾 1 窗。")
                        gain_columns[1].metric(
                            "最大单窗耗尽",
                            f"{100.0 * float(transient.get('maximum_reservoir_depletion_per_slice', 0.0)):.3g}%",
                        )
                        gain_columns[2].metric(
                            "软窗重组能量误差",
                            f"{100.0 * float(transient.get('maximum_resynthesis_energy_error', 0.0)):.3g}%",
                        )
                        trace = transient.get("representative_trace") or {}
                        if trace.get("time_s"):
                            transient_figure = go.Figure()
                            transient_figure.add_scatter(
                                x=np.asarray(trace["time_s"]) * 1e15,
                                y=trace["slice_energy_gain"],
                                mode="lines+markers",
                                name="单窗能量增益",
                            )
                            transient_figure.add_scatter(
                                x=np.asarray(trace["time_s"]) * 1e15,
                                y=trace["reservoir_after"],
                                mode="lines+markers",
                                name="窗后增益库",
                                yaxis="y2",
                            )
                            transient_figure.update_layout(
                                title="代表性晶体步：front → peak → tail 瞬态增益",
                                xaxis_title="软窗中心时间 (fs)",
                                yaxis_title="单窗能量增益 (×)",
                                yaxis2=dict(title="上能级/有效增益库比例", overlaying="y", side="right"),
                            )
                            st.plotly_chart(transient_figure, width="stretch")
                            st.caption(
                                "代表软窗全宽约 "
                                f"{float(trace.get('soft_window_full_width_s', 0.0)) * 1e15:.4g} fs；"
                                "对应光谱分辨率量级约 "
                                f"{float(trace.get('spectral_resolution_estimate_Hz', 0.0)) * 1e-12:.4g} THz。"
                                "若它不足以解析你的截面响应，应增加相干时间窗口或改用更完整的场耦合模型。"
                            )
                    else:
                        st.warning(
                            f"逐时间窗增益已退回总通量近似：{transient.get('reason', transient.get('status', 'unknown'))}"
                        )
        if (
            result.wavelengths_m is not None
            and result.input_spectrum_J is not None
            and result.output_spectrum_J is not None
        ):
            spectral_bin_coordinate = spectral_bin_coordinate_for_result(
                signal_used,
                cfg["seed"],
            )
            spectral_edges = result.spectral_bin_edges_Hz
            if spectral_edges is not None:
                spectral_bin_coordinate = "frequency"
            try:
                input_density = spectral_density_from_bin_energy(
                    result.wavelengths_m,
                    result.input_spectrum_J,
                    bin_coordinate=spectral_bin_coordinate,
                    bin_edges=spectral_edges,
                )
                output_density = spectral_density_from_bin_energy(
                    result.wavelengths_m,
                    result.output_spectrum_J,
                    bin_coordinate=spectral_bin_coordinate,
                    bin_edges=spectral_edges,
                )
            except ValueError as exc:
                st.warning(f"无法构造坐标守恒的光谱密度：{exc}")
            else:
                log_display_floor = 1e-6

                def normalized_spectrum_figure(*, logarithmic: bool) -> go.Figure:
                    figure = go.Figure()
                    for density, name, color in (
                        (input_density, "输入", "#d1d5db"),
                        (output_density, "输出", "#ef4444"),
                    ):
                        values = np.asarray(
                            density["normalized_wavelength_density"],
                            dtype=float,
                        )
                        if logarithmic:
                            # Exact zeros remain absent on a logarithmic axis.
                            # This display floor never changes result/archive data.
                            values = np.where(
                                np.isfinite(values)
                                & (values >= log_display_floor),
                                values,
                                np.nan,
                            )
                        figure.add_scatter(
                            x=density["wavelength_m"] * 1e9,
                            y=values,
                            mode="lines",
                            name=name,
                            line={"color": color, "width": 2.4},
                            connectgaps=False,
                        )
                    figure.update_layout(
                        title=(
                            "输入/输出归一化波长光谱密度（对数尺度）"
                            if logarithmic
                            else "输入/输出归一化波长光谱密度（论文对比视图）"
                        ),
                        xaxis_title="波长 (nm)",
                        yaxis_title="归一化 dE/dλ",
                    )
                    if logarithmic:
                        figure.update_yaxes(
                            type="log",
                            range=[float(np.log10(log_display_floor)), 0.05],
                            dtick=1,
                            exponentformat="power",
                            showexponent="all",
                        )
                    return figure

                linear_scale_tab, log_scale_tab = st.tabs(
                    ["线性尺度", "对数尺度"]
                )
                with linear_scale_tab:
                    st.plotly_chart(
                        normalized_spectrum_figure(logarithmic=False),
                        width="stretch",
                        key=result_key("normalized_spectrum_linear"),
                    )
                with log_scale_tab:
                    st.plotly_chart(
                        normalized_spectrum_figure(logarithmic=True),
                        width="stretch",
                        key=result_key("normalized_spectrum_log"),
                    )
                    st.caption(
                        "对数谱默认显示到相对峰值 10⁻⁶；更低的点只在图中隐藏，"
                        "不会修改计算结果或归档数据。"
                    )
                st.caption(
                    "这里画的是波长密度 dE/dλ，并分别按峰值归一化；不是每个数值"
                    "分箱的能量。J/bin 在非均匀波长坐标上会产生纯绘图 Jacobian 偏差。"
                )
                st.caption(
                    "能量质量的原生分箱坐标：频率。"
                    if spectral_bin_coordinate == "frequency"
                    else "能量质量的原生分箱坐标：波长。"
                )
                with st.expander("绝对光谱密度与原生数值分箱", expanded=False):
                    absolute = go.Figure()
                    absolute.add_scatter(
                        x=input_density["wavelength_m"] * 1e9,
                        y=input_density["density_J_per_nm"],
                        mode="lines",
                        name="输入 dE/dλ",
                    )
                    absolute.add_scatter(
                        x=output_density["wavelength_m"] * 1e9,
                        y=output_density["density_J_per_nm"],
                        mode="lines",
                        name="输出 dE/dλ",
                    )
                    absolute.update_layout(
                        title="绝对波长光谱密度",
                        xaxis_title="波长 (nm)",
                        yaxis_title="dE/dλ (J/nm)",
                    )
                    st.plotly_chart(absolute, width="stretch")
                    native = go.Figure()
                    native.add_scatter(
                        x=result.wavelengths_m * 1e9,
                        y=result.input_spectrum_J,
                        mode="lines+markers",
                        name="输入 J/bin",
                    )
                    native.add_scatter(
                        x=result.wavelengths_m * 1e9,
                        y=result.output_spectrum_J,
                        mode="lines+markers",
                        name="输出 J/bin",
                    )
                    native.update_layout(
                        title="原生离散光谱能量（数值诊断）",
                        xaxis_title="波长 (nm)",
                        yaxis_title="每个数值分箱能量 (J)",
                    )
                    st.plotly_chart(native, width="stretch")
                    st.caption(
                        "密度积分闭合误差：输入 λ域 "
                        f"{float(input_density['wavelength_closure_relative']):.3e}，"
                        "输入 f域 "
                        f"{float(input_density['frequency_closure_relative']):.3e}；"
                        "输出 λ域 "
                        f"{float(output_density['wavelength_closure_relative']):.3e}，"
                        "输出 f域 "
                        f"{float(output_density['frequency_closure_relative']):.3e}。"
                    )
            if all(
                phase is not None
                for phase in (
                    result.input_spectral_phase_rad,
                    result.output_spectral_phase_rad,
                    result.compressed_spectral_phase_rad,
                )
            ):
                order = np.argsort(result.wavelengths_m)
                phase_figure = go.Figure()
                phase_figure.add_scatter(
                    x=result.wavelengths_m[order] * 1e9,
                    y=result.input_spectral_phase_rad[order],
                    mode="lines",
                    name="输入（含展宽器）",
                )
                phase_figure.add_scatter(
                    x=result.wavelengths_m[order] * 1e9,
                    y=result.output_spectral_phase_rad[order],
                    mode="lines",
                    name="放大后",
                )
                phase_figure.add_scatter(
                    x=result.wavelengths_m[order] * 1e9,
                    y=result.compressed_spectral_phase_rad[order],
                    mode="lines",
                    name="压缩后",
                )
                phase_figure.update_layout(title="相干光谱相位", xaxis_title="波长 (nm)", yaxis_title="展开相位 (rad)")
                st.plotly_chart(phase_figure, width="stretch")
        else:
            st.info("当前使用单色快速通量模型；选择多色光谱或 spectral_time 可显示增益窄化。")
        if (
            result.time_s is not None
            and result.input_temporal_energy_J is not None
            and result.output_temporal_energy_J is not None
        ):
            st.plotly_chart(temporal_result_figure(result), width="stretch")
            if len(result.time_s) == 1:
                st.caption("仅有一个时间分箱，缺少时间边界；显示分箱能量，不能据此推断功率。")
        levels = LevelSystem.from_config(cfg["energy_levels"], cfg["material"])
        levels.set_environment(
            float(cfg["material"].get("operating_temperature_K", 293.15)),
            float(cfg["material"].get("ion_density_m3", 0.0)),
        )
        center = float(cfg["seed"]["wavelength_m"])
        bandwidth = max(float(cfg["seed"]["spectrum"].get("bandwidth_fwhm_m", 0.0)), center * 0.08)
        wavelength = np.linspace(max(center - bandwidth, np.finfo(float).tiny), center + bandwidth, 301)
        preview_values, preview_errors = signal_cross_section_preview(levels, wavelength)
        base_sigma_abs, base_sigma_em, sigma_abs, sigma_em = preview_values
        if preview_errors:
            st.warning("截面预览中的不可用采样已留空：" + preview_errors[0])
        figure = go.Figure()
        figure.add_scatter(x=wavelength * 1e9, y=sigma_em, name="求解器 σem")
        figure.add_scatter(x=wavelength * 1e9, y=sigma_abs, name="求解器 σabs")
        if levels.response.enabled:
            figure.add_scatter(x=wavelength * 1e9, y=base_sigma_em, name="基础 σem", line={"dash": "dot"})
            figure.add_scatter(x=wavelength * 1e9, y=base_sigma_abs, name="基础 σabs", line={"dash": "dot"})
        figure.update_layout(title="信号跃迁响应曲线（配置工作温度）", xaxis_title="波长 (nm)", yaxis_title="截面 (m²)")
        st.caption("截面曲线使用配置的统一工作温度；局部三维温度反馈请以已保存的体场与诊断为准。")
        st.plotly_chart(figure, width="stretch")

    with optics_tab:
        st.divider()
        st.subheader("逐程与记录光学面")
        polarization = summary["signal"].get("polarization", {})
        if polarization.get("model") == "jones":
            st.markdown("#### Jones 偏振输出")
            parallel_quality = polarization.get(
                "parallel_analyzer_beam_quality", {}
            )
            orthogonal_quality = polarization.get(
                "orthogonal_analyzer_beam_quality", {}
            )
            st.dataframe(
                [
                    {
                        "正交检偏功率比例": polarization.get(
                            "orthogonal_analyzer_fraction"
                        ),
                        "积分偏振度": polarization.get(
                            "degree_of_polarization_from_integrated_stokes"
                        ),
                        "S0": polarization.get("stokes_S0"),
                        "S1": polarization.get("stokes_S1"),
                        "S2": polarization.get("stokes_S2"),
                        "S3": polarization.get("stokes_S3"),
                        "主检偏 M²x": parallel_quality.get("m2_x"),
                        "主检偏 M²y": parallel_quality.get("m2_y"),
                        "正交检偏 M²x": orthogonal_quality.get("m2_x"),
                        "正交检偏 M²y": orthogonal_quality.get("m2_y"),
                        "累积延迟峰谷 (rad)": polarization.get(
                            "coordinate_accumulated_retardance_peak_to_valley_rad"
                        ),
                    }
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption(str(polarization.get("analyzer_scope", "")))
        passes = summary["signal"]["passes"]
        if passes:
            figure = go.Figure(go.Scatter(x=[item["pass"] for item in passes], y=[item.get("crystal_exit_J", 0.0) * 1e9 for item in passes], mode="lines+markers"))
            figure.update_layout(title="逐程晶体出口能量", xaxis_title="晶体命中次数", yaxis_title="能量 (nJ)")
            st.plotly_chart(figure, width="stretch")
            st.dataframe(passes, width="stretch")
        caustic = summary["signal"].get("caustic_diagnostics") or {}
        if caustic.get("enabled", False):
            st.subheader("输出焦散与多平面 M²")
            if caustic.get("broadband_effective", False):
                st.info(
                    "当前为逐波长传播后按强度求和的宽带焦散；表中 M² 是以能量加权参考波长"
                    "定义的工程有效值，不是单色 ISO-11146 数值。"
                )
            samples = caustic.get("samples", [])
            if samples:
                caustic_figure = go.Figure()
                positions_mm = [
                    float(row["position_m_relative_to_system_output"]) * 1e3
                    for row in samples
                ]
                caustic_figure.add_scatter(
                    x=positions_mm,
                    y=[float(row["radius_x_D4sigma_m"]) * 1e6 for row in samples],
                    mode="lines+markers",
                    name="D4σ radius x",
                )
                caustic_figure.add_scatter(
                    x=positions_mm,
                    y=[float(row["radius_y_D4sigma_m"]) * 1e6 for row in samples],
                    mode="lines+markers",
                    name="D4σ radius y",
                )
                caustic_figure.update_layout(
                    xaxis_title="距系统输出面 (mm)",
                    yaxis_title="D4σ 半径 (µm)",
                )
                st.plotly_chart(caustic_figure, width="stretch")
            fit_x = caustic.get("fit_x", {})
            fit_y = caustic.get("fit_y", {})
            st.dataframe(
                [
                    {
                        "轴": "x",
                        "拟合有效": fit_x.get("valid"),
                        "M²": fit_x.get("M2"),
                        "束腰位置 (m)": fit_x.get(
                            "waist_position_m_relative_to_system_output"
                        ),
                        "D4σ 束腰半径 (m)": fit_x.get("waist_radius_D4sigma_m"),
                        "ISO式采样覆盖": fit_x.get("iso_style_sampling_coverage"),
                    },
                    {
                        "轴": "y",
                        "拟合有效": fit_y.get("valid"),
                        "M²": fit_y.get("M2"),
                        "束腰位置 (m)": fit_y.get(
                            "waist_position_m_relative_to_system_output"
                        ),
                        "D4σ 束腰半径 (m)": fit_y.get("waist_radius_D4sigma_m"),
                        "ISO式采样覆盖": fit_y.get("iso_style_sampling_coverage"),
                    },
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption(caustic.get("scope", ""))
        _render_intensity_snapshots(result, result_key)
        if result.optical_planes:
            plane_summaries = summary.get("optical_train", {}).get("planes", [])
            if plane_summaries:
                st.caption(
                    "表中 M² 是各采样复场的原始二阶矩；顶部结果会明确标注其为采样二阶矩、"
                    "配置模态合同或包含 effective-M² 传输因子的组合代理量。"
                )
                st.dataframe(
                    stringify_table_column(plane_summaries, "M2_x_sampled"),
                    hide_index=True,
                    width="stretch",
                )
            plane_name = choice_selectbox(
                "查看光学面",
                list(result.optical_planes),
                key=result_key("result_optical_plane"),
            )
            plane_x, plane_y = optical_plane_axes(result, plane_name)
            selected_plane_summary = next(
                (row for row in plane_summaries if row.get("name") == plane_name), {}
            )
            plane_unit = (
                "J/m²" if "energy_J" in selected_plane_summary
                else "W/m²" if "power_W" in selected_plane_summary
                else "保存值（原生单位）"
            )
            st.plotly_chart(
                heatmap(
                    result.optical_planes[plane_name],
                    plane_x,
                    plane_y,
                    plane_name,
                    plane_unit,
                ),
                width="stretch",
                key=result_key(f"result_optical_plane_map_{plane_name}"),
            )
            selected_plane = np.asarray(result.optical_planes[plane_name], dtype=float)
            peak_y, peak_x = np.unravel_index(
                int(np.argmax(selected_plane)), selected_plane.shape
            )
            normalization = max(float(np.max(selected_plane)), np.finfo(float).tiny)
            section_figure = go.Figure()
            section_figure.add_scatter(
                x=np.asarray(plane_x) * 1e6,
                y=selected_plane[peak_y] / normalization,
                mode="lines",
                name=f"x 截面（y={float(plane_y[peak_y]) * 1e6:.3g} µm）",
            )
            section_figure.add_scatter(
                x=np.asarray(plane_y) * 1e6,
                y=selected_plane[:, peak_x] / normalization,
                mode="lines",
                name=f"y 截面（x={float(plane_x[peak_x]) * 1e6:.3g} µm）",
            )
            section_figure.update_layout(
                title=f"{plane_name} 峰值处 x / y 光斑截面",
                xaxis_title="横向坐标 (µm)",
                yaxis_title="相对强度",
            )
            st.plotly_chart(
                section_figure,
                width="stretch",
                key=result_key(f"result_optical_plane_section_{plane_name}"),
            )
        else:
            st.info("未保存光学面；在“⑤ 结果记录 → 记录层级与内容”中打开“保存各光学面”。")

    with diagnostics_tab:
        if summary["warnings"]:
            actionable_warnings = []
            informational_notices = []
            for warning in summary["warnings"]:
                level, source = solver_notice_presentation(warning)
                target = (
                    informational_notices
                    if level == "info"
                    else actionable_warnings
                )
                target.append({"来源": source, "信息": str(warning)})
            for warning in actionable_warnings:
                st.warning(warning["信息"])
            if informational_notices:
                with st.expander(
                    f"模型合同与非阻断说明（{len(informational_notices)} 条）",
                    expanded=False,
                ):
                    st.dataframe(
                        informational_notices,
                        hide_index=True,
                        width="stretch",
                    )
            if not actionable_warnings:
                st.success("没有触发需要调整的求解器数值警告。")
        else:
            st.success("没有触发网格捕获、采样或物理开关诊断。")
        performance = summary.get("performance", {})
        population_linear_solver = (
            summary.get("population", {}).get("implicit_linear_solver", {})
            or {}
        )
        transient_backend = (
            (summary.get("signal", {}).get("cpa") or {}).get(
                "transient_gain", {}
            )
            or {}
        )
        st.subheader("运行时与加速后端")
        st.dataframe(
            stringify_table_column([
                {"项目": "FFT 后端", "数值": performance.get("fft_backend_selected")},
                {"项目": "FFT 线程", "数值": performance.get("fft_workers")},
                {"项目": "人口小矩阵后端", "数值": population_linear_solver.get("selected")},
                {"项目": "瞬态人口内核", "数值": transient_backend.get("single_transition_kernel_selected")},
                {"项目": "软窗批量 FFT", "数值": transient_backend.get("batch_window_ffts_active")},
                {"项目": "Python 进程架构", "数值": performance.get("process_architecture")},
                {"项目": "原生 Apple Silicon Python", "数值": performance.get("native_apple_silicon_python")},
                {"项目": "Rosetta 转译", "数值": performance.get("running_under_rosetta")},
                {"项目": "MLX 可用", "数值": performance.get("mlx_available")},
                {"项目": "Apple GPU 状态", "数值": performance.get("apple_gpu_reason")},
                {"项目": "Apple GPU 取舍", "数值": performance.get("apple_gpu_note")},
            ], "数值"),
            hide_index=True,
            width="stretch",
        )
        st.subheader("Rollover 相关物理模块")
        ase_summary = summary.get("ase", {})
        pump_summary = summary.get("pump", {})
        stress_summary = summary.get("stress", {})
        gain_transport_summary = summary.get("signal", {}).get("gain_transport") or {}
        spectroscopy_summary = summary.get("spectroscopic_response", {})
        strong_field_summary = (summary.get("signal", {}).get("cpa") or {}).get("strong_field", {})
        st.dataframe(
            stringify_table_column(
                [
                    {"项目": "局部光谱反馈", "数值": spectroscopy_summary.get("mode", "off")},
                    {"项目": "局部光谱温度范围 (K)", "数值": [spectroscopy_summary.get("temperature_min_K"), spectroscopy_summary.get("temperature_max_K")]},
                    {"项目": "Pump 光谱空间输运", "数值": pump_summary.get("spectral_spatial_transport", "shared")},
                    {"项目": "独立 HG 模态数", "数值": gain_transport_summary.get("spatial_mode_count")},
                    {"项目": "ASE 模型", "数值": ase_summary.get("model", "off")},
                    {"项目": "ASE 逃逸功率 (W)", "数值": ase_summary.get("escaped_power_W")},
                    {"项目": "ASE 角方向数", "数值": ase_summary.get("direction_count")},
                    {"项目": "寄生边界主本征值", "数值": ase_summary.get("dominant_parasitic_boundary_eigenvalue")},
                    {"项目": "TPL 阈值超出", "数值": ase_summary.get("parasitic_threshold_exceeded")},
                    {"项目": "TPL 各轴往返增益", "数值": ase_summary.get("parasitic_roundtrip_gain")},
                    {"项目": "力学模型", "数值": stress_summary.get("model")},
                    {"项目": "张量光弹最大双折射", "数值": stress_summary.get("photoelastic_birefringence_max")},
                    {"项目": "端面 sag P-V (m)", "数值": stress_summary.get("endface_sag_peak_to_valley_m")},
                    {"项目": "强场损耗", "数值": strong_field_summary.get("model", "off")},
                    {"项目": "多光子损耗/脉冲 (J)", "数值": strong_field_summary.get("multiphoton_absorbed_energy_J")},
                    {"项目": "自由载流子损耗/脉冲 (J)", "数值": strong_field_summary.get("free_carrier_absorbed_energy_J")},
                    {"项目": "峰值载流子密度 (m⁻³)", "数值": strong_field_summary.get("maximum_peak_carrier_density_m3")},
                    {"项目": "Seed 被动吸收/脉冲 (J)", "数值": summary.get("signal", {}).get("passive_loss_energy_J")},
                    {
                        "项目": "Seed 不可逆产热耦合",
                        "数值": summary.get("thermal", {}).get(
                            "seed_irreversible_heat_coupling",
                            summary.get("thermal", {}).get(
                                "seed_nonlinear_heat_coupling"
                            ),
                        ),
                    },
                ],
                "数值",
            ),
            hide_index=True,
            width="stretch",
        )
        provenance = summary.get("configuration_provenance", {})
        material_calibration = (
            (summary.get("material_library") or {}).get(
                "experimental_calibration", {}
            )
            or {}
        )
        st.subheader("配置与材料可追溯性")
        st.dataframe(
            stringify_table_column(
                [
                    {"项目": "运行配置 SHA-256", "数值": provenance.get("config_sha256")},
                    {"项目": "材料名称", "数值": provenance.get("material_name")},
                    {"项目": "材料记录可追溯", "数值": provenance.get("material_record_traceable")},
                    {"项目": "材料记录 SHA-256", "数值": provenance.get("material_record_sha256")},
                    {"项目": "材料快照与运行值一致", "数值": provenance.get("material_snapshot_matches_runtime")},
                    {"项目": "Setup/材料哈希一致", "数值": provenance.get("setup_material_hash_matches")},
                    {"项目": "Setup 源文件验证", "数值": provenance.get("setup_source_verification")},
                    {"项目": "Setup 源文件", "数值": (provenance.get("setup_provenance") or {}).get("source_file")},
                    {"项目": "Setup 源文件运行时 SHA-256", "数值": provenance.get("setup_source_runtime_sha256")},
                    {"项目": "材料记录 ID", "数值": provenance.get("material_record_id")},
                    {"项目": "掺杂百分比", "数值": provenance.get("doping_percent")},
                    {"项目": "材料评估温度 (K)", "数值": provenance.get("operating_temperature_K")},
                    {"项目": "样品级实验标定状态", "数值": material_calibration.get("status", "unspecified")},
                    {"项目": "样品标识", "数值": material_calibration.get("sample_identifier")},
                    {"项目": "标定数据 SHA-256", "数值": material_calibration.get("dataset_sha256")},
                    {"项目": "已标定参数", "数值": material_calibration.get("calibrated_parameters", [])},
                    {"项目": "要求标定参数", "数值": material_calibration.get("required_parameters", [])},
                ],
                "数值",
            ),
            hide_index=True,
            width="stretch",
        )
        grid_coupling = summary.get("grid_coupling", {"enabled": False})
        st.subheader("物理/光学双网格耦合")
        if grid_coupling.get("enabled", False):
            seed_grid_coupling = grid_coupling.get("seed", grid_coupling)
            pump_grid_coupling = grid_coupling.get("pump", grid_coupling)
            signal_coupling = summary["signal"].get("grid_coupling") or {}
            st.dataframe(
                stringify_table_column([
                    {"项目": "物理网格 (Nz, Ny, Nx)", "数值": grid_coupling.get("physical_shape_nz_ny_nx")},
                    {"项目": "Seed/BPM 网格 (Nz, Ny, Nx)", "数值": seed_grid_coupling.get("optical_shape_nz_ny_nx") if seed_grid_coupling.get("enabled") else "物理网格"},
                    {"项目": "Pump 网格 (Nz, Ny, Nx)", "数值": pump_grid_coupling.get("optical_shape_nz_ny_nx") if pump_grid_coupling.get("enabled") else "物理网格"},
                    {"项目": "Seed Δx / Δy (m)", "数值": [seed_grid_coupling.get("optical_dx_m"), seed_grid_coupling.get("optical_dy_m")] if seed_grid_coupling.get("enabled") else None},
                    {"项目": "Pump Δx / Δy (m)", "数值": [pump_grid_coupling.get("optical_dx_m"), pump_grid_coupling.get("optical_dy_m")] if pump_grid_coupling.get("enabled") else None},
                    {"项目": "Pump/Seed 共用光学网格", "数值": grid_coupling.get("shared_optical_grid")},
                    {"项目": "守恒映射后端", "数值": grid_coupling.get("remap_backend")},
                    {"项目": "Pump 映射功率相对误差", "数值": summary["pump"].get("grid_coupling_relative_power_error")},
                    {"项目": "Seed 提取能量映射相对误差", "数值": signal_coupling.get("extraction_energy_relative_error")},
                    {"项目": "布居投影最大相对修正", "数值": signal_coupling.get("population_projection_correction_relative")},
                ], "数值"),
                hide_index=True,
                width="stretch",
            )
            st.caption("Pump 与 Seed/BPM 可使用不同的均匀光学网格；吸收、布居变化和 Δn 通过有限体积单元重叠映射与完整晶体网格耦合。")
        else:
            st.caption("当前使用单一网格；打开“独立均匀光学网格”可让局部加密晶体网格与 ABCD/FFT 同时使用。")
        pump_diagnostics = summary.get("pump", {})
        st.subheader("Pump 外部光路与晶体入口")
        st.dataframe(
            stringify_table_column(
                [
                    {"项目": "Pump 总传播模式", "数值": pump_diagnostics.get("propagation")},
                    {"项目": "晶体外传播", "数值": pump_diagnostics.get("external_propagation")},
                    {"项目": "晶体内传播", "数值": pump_diagnostics.get("crystal_propagation")},
                    {"项目": "配置源功率 (W)", "数值": pump_diagnostics.get("configured_power_W")},
                    {"项目": "源平面数值功率 (W)", "数值": pump_diagnostics.get("grid_source_power_W")},
                    {"项目": "源平面窗口会否截断实际传播", "数值": pump_diagnostics.get("source_plane_sampling_affects_transport")},
                    {"项目": "晶体入口解析功率 (W)", "数值": pump_diagnostics.get("gain_entrance_analytic_power_W")},
                    {"项目": "晶体入口采样功率 (W)", "数值": pump_diagnostics.get("gain_entrance_sampled_power_W")},
                    {"项目": "晶体入口窗口捕获率", "数值": pump_diagnostics.get("gain_entrance_capture_fraction")},
                    {"项目": "晶体口径透过率", "数值": pump_diagnostics.get("gain_entrance_crystal_aperture_transmission")},
                    {"项目": "入口半径 x/y (m)", "数值": [pump_diagnostics.get("gain_entrance_radius_x_m"), pump_diagnostics.get("gain_entrance_radius_y_m")]},
                    {"项目": "入口波前曲率 x/y (m)", "数值": [pump_diagnostics.get("gain_entrance_curvature_x_m"), pump_diagnostics.get("gain_entrance_curvature_y_m")]},
                    {"项目": "入口中心 x/y (m)", "数值": [pump_diagnostics.get("gain_entrance_center_x_m"), pump_diagnostics.get("gain_entrance_center_y_m")]},
                    {"项目": "入口角度 x/y (rad)", "数值": [pump_diagnostics.get("gain_entrance_angle_x_rad"), pump_diagnostics.get("gain_entrance_angle_y_rad")]},
                    {"项目": "入口倾角相位已写入", "数值": pump_diagnostics.get("gain_entrance_tilt_phase_included")},
                ],
                "数值",
            ),
            hide_index=True,
            width="stretch",
        )
        if pump_diagnostics.get("propagation") == "hybrid_abcd_fft":
            st.caption(
                "混合模式不把宽源平面裁剪带入实际功率；只有晶体入口捕获率和真实晶体口径会截断进入晶体的场。"
            )
        thermo_optic = summary["thermal"]
        st.subheader("折射率与热透镜实际耦合")
        focal_x = thermo_optic.get("estimated_thin_lens_focal_x_m")
        focal_y = thermo_optic.get("estimated_thin_lens_focal_y_m")
        st.dataframe(
            stringify_table_column([
                {"项目": "Seed 材料折射率 n(λ)", "数值": thermo_optic.get("signal_refractive_index"), "作用": "晶体内衍射、界面折射/透射"},
                {"项目": "Pump 材料折射率 n(λ)", "数值": thermo_optic.get("pump_refractive_index"), "作用": "泵浦 ABCD/FFT 与界面"},
                {"项目": "热/应力 Δn 峰谷值", "数值": thermo_optic.get("delta_n_peak_to_valley"), "作用": "逐 z 步写入 Seed 复场相位"},
                {"项目": "热光相位峰谷值 (rad)", "数值": thermo_optic.get("thermo_optic_phase_peak_to_valley_rad"), "作用": "决定热透镜/像差强度"},
                {"项目": "等效薄透镜焦距 x (m)", "数值": focal_x, "作用": "由积分 Δn 二次拟合"},
                {"项目": "等效薄透镜焦距 y (m)", "数值": focal_y, "作用": "由积分 Δn 二次拟合"},
                {"项目": "产热来源模型", "数值": thermo_optic.get("heat_source_model"), "作用": "决定主动/被动 Pump 吸收如何变成热"},
                {"项目": "主动吸收产热公式", "数值": thermo_optic.get("active_heat_fraction_model"), "作用": "能量守恒、可调量子亏损或实测比例"},
                {"项目": "主动吸收实际产热比例", "数值": thermo_optic.get("resolved_active_heat_fraction"), "作用": "显式热比例，或含量子效率的量子亏损结果"},
                {"项目": "Pump 光谱产热模型", "数值": (thermo_optic.get("pump_spectral_heat") or {}).get("model"), "作用": (thermo_optic.get("pump_spectral_heat") or {}).get("method")},
                {"项目": "局部光谱产热比例 min/max", "数值": [(thermo_optic.get("pump_spectral_heat") or {}).get("fraction_min"), (thermo_optic.get("pump_spectral_heat") or {}).get("fraction_max")], "作用": "仅 spectral_* 模式"},
                {"项目": "产热参考波长 (m)", "数值": thermo_optic.get("heat_reference_wavelength_m"), "作用": thermo_optic.get("heat_reference_source")},
                {"项目": "Pump 激发量子效率", "数值": thermo_optic.get("pump_quantum_efficiency"), "作用": "布居激发率；寿命模式不把荧光产率重复填在这里"},
                {"项目": "寿命非辐射热 (W)", "数值": (thermo_optic.get("lifetime_nonradiative_heat") or {}).get("heat_power_W"), "作用": (thermo_optic.get("lifetime_nonradiative_heat") or {}).get("scope")},
                {"项目": "总寿命 min/max (s)", "数值": [(thermo_optic.get("lifetime_nonradiative_heat") or {}).get("total_lifetime_s_min"), (thermo_optic.get("lifetime_nonradiative_heat") or {}).get("total_lifetime_s_max")], "作用": "固定寿命或温度/掺杂响应后的局部寿命"},
                {"项目": "寿命推导荧光产率 min/max", "数值": [(thermo_optic.get("lifetime_nonradiative_heat") or {}).get("fluorescence_quantum_yield_min"), (thermo_optic.get("lifetime_nonradiative_heat") or {}).get("fluorescence_quantum_yield_max")], "作用": "τtotal/τrad，仅用于分解辐射与非辐射衰变"},
                {"项目": "Pump 主动吸收 (W)", "数值": thermo_optic.get("pump_active_absorbed_power_W"), "作用": "分量账本"},
                {"项目": "Pump 被动吸收 (W)", "数值": thermo_optic.get("pump_passive_absorbed_power_W"), "作用": "分量账本中全部转热"},
            ], "数值"),
            hide_index=True,
            width="stretch",
        )
        field_diagnostics = thermo_optic.get("field_diagnostics", {})
        if field_diagnostics.get("enabled", False):
            with st.expander("三维场抛物拟合质量", expanded=False):
                optical_path_fit = field_diagnostics.get(
                    "integrated_optical_path_fit", {}
                )
                gain_fit = field_diagnostics.get(
                    "integrated_gain_exponent_fit", {}
                )
                zernike_fit = field_diagnostics.get(
                    "integrated_optical_path_zernike", {}
                )
                st.dataframe(
                    stringify_table_column(
                        [
                            {
                                "项目": "实际传播是否使用拟合场",
                                "数值": field_diagnostics.get("affects_propagation"),
                            },
                            {
                                "项目": "拟合半径来源",
                                "数值": field_diagnostics.get("resolved_fit_radius_mode"),
                            },
                            {
                                "项目": "拟合半径 x/y (m)",
                                "数值": [
                                    field_diagnostics.get("fit_radius_x_m"),
                                    field_diagnostics.get("fit_radius_y_m"),
                                ],
                            },
                            {
                                "项目": "积分热光程拟合归一化 RMSE",
                                "数值": optical_path_fit.get("normalized_rmse"),
                            },
                            {
                                "项目": "积分增益指数拟合归一化 RMSE",
                                "数值": gain_fit.get("normalized_rmse"),
                            },
                            {
                                "项目": "热透镜主轴焦距 (m)",
                                "数值": field_diagnostics.get(
                                    "estimated_principal_focal_lengths_m"
                                ),
                            },
                            {
                                "项目": "热透镜主轴角度 (deg)",
                                "数值": field_diagnostics.get("principal_axis_angle_deg"),
                            },
                            {
                                "项目": "主球差 Z(4,0) 系数 (m)",
                                "数值": zernike_fit.get(
                                    "primary_spherical_n4_m0_coefficient_m"
                                ),
                            },
                            {
                                "项目": "主球差 Z(4,0) 系数 (waves)",
                                "数值": zernike_fit.get(
                                    "primary_spherical_n4_m0_coefficient_waves"
                                ),
                            },
                            {
                                "项目": "Zernike 拟合残差 RMS (waves)",
                                "数值": zernike_fit.get("residual_rms_waves"),
                            },
                        ],
                        "数值",
                    ),
                    hide_index=True,
                    width="stretch",
                )
                thermal_slices = field_diagnostics.get(
                    "thermal_delta_n_slices", []
                )
                gain_slices = field_diagnostics.get("gain_m_inv_slices", [])
                slice_rows = []
                for thermal_slice, gain_slice in zip(
                    thermal_slices, gain_slices, strict=False
                ):
                    slice_rows.append(
                        {
                            "z (mm)": float(thermal_slice.get("z_m", 0.0)) * 1e3,
                            "Δn 拟合有效": bool(thermal_slice.get("valid", False)),
                            "Δn 归一化 RMSE": thermal_slice.get("normalized_rmse"),
                            "gain 拟合有效": bool(gain_slice.get("valid", False)),
                            "gain 归一化 RMSE": gain_slice.get("normalized_rmse"),
                        }
                    )
                if slice_rows:
                    st.dataframe(slice_rows, hide_index=True, width="stretch")
                profiles = field_diagnostics.get(
                    "nonparabolic_optical_path_profiles"
                ) or {}
                radial_profile = profiles.get("radial_azimuthal_average", [])
                if radial_profile:
                    opd_figure = go.Figure()
                    radial_um = [
                        row["equivalent_radius_m"] * 1e6 for row in radial_profile
                    ]
                    wavelength_m = float(cfg["seed"]["wavelength_m"])
                    opd_figure.add_scatter(
                        x=radial_um,
                        y=[row["optical_path_m"] / wavelength_m for row in radial_profile],
                        mode="lines+markers",
                        name="积分 OPD",
                    )
                    opd_figure.add_scatter(
                        x=radial_um,
                        y=[row["parabolic_fit_m"] / wavelength_m for row in radial_profile],
                        mode="lines",
                        name="二次拟合",
                    )
                    opd_figure.update_layout(
                        xaxis_title="等效半径 (µm)",
                        yaxis_title="OPD (waves)",
                    )
                    st.plotly_chart(opd_figure, width="stretch")
                    profile_figure = go.Figure()
                    profile_figure.add_scatter(
                        x=radial_um,
                        y=[
                            row.get("nonparabolic_residual_waves", 0.0)
                            for row in radial_profile
                        ],
                        mode="lines+markers",
                        name="去二次项 OPD",
                    )
                    profile_figure.update_layout(
                        xaxis_title="等效半径 (µm)",
                        yaxis_title="非抛物 OPD (waves)",
                    )
                    st.plotly_chart(profile_figure, width="stretch")
                st.caption(field_diagnostics.get("scope", ""))
        if not thermo_optic.get("phase_screen_applied_to_seed", False):
            st.info("当前折射率模型不是 thermal_3d，因此温度场不会生成 Δn 相位屏。")
        elif not thermo_optic.get("focusing_observable_in_current_model", False):
            st.warning("Δn 相位屏已计算，但光学传播为 off；打开 ABCD/FFT 才能把相位曲率转化为可见的聚焦变化。")
        else:
            st.caption("热透镜不是展示参数：求解器把 dn/dT·ΔT 和应力光弹 Δn 逐层乘到 Seed 复场，再继续衍射传播。")
        st.subheader("模型假设")
        for assumption in summary["assumptions"]:
            st.write(f"- {assumption}")
        st.download_button("下载本次配置 JSON", json.dumps(cfg, ensure_ascii=False, indent=2), "amplifier_config.json", "application/json")
        st.download_button(
            "下载无效诊断摘要 JSON" if invalid_result else "下载摘要 JSON",
            dumps_json_portable(summary, ensure_ascii=False, indent=2),
            "invalid_diagnostic_summary.json" if invalid_result else "summary.json",
            "application/json",
        )
        buffer = io.BytesIO()
        arrays = standard_result_arrays(result)
        arrays.update(
            {name: np.asarray(values) for name, values in result.stress.items()}
        )
        np.savez_compressed(buffer, **arrays)
        st.download_button(
            "下载三维场 NPZ",
            buffer.getvalue(),
            "amplifier_fields.npz",
            "application/octet-stream",
            disabled=invalid_result,
            help=(
                "验证失败时禁用正式三维场导出；完整诊断包仍可下载。"
                if invalid_result
                else None
            ),
        )


ARCHIVE_STATUS_LABELS = {
    "valid": "有效",
    "valid_with_warnings": "有效（有诊断警告）",
    "execution_completed_but_invalid": "完成但无效",
    "execution_failed": "执行失败",
}


def archive_status_label(status: Any) -> str:
    return ARCHIVE_STATUS_LABELS.get(str(status), str(status or "未知"))


def archived_point_label(archive: ResultArchive, index: int) -> str:
    record = archive.points[index].record
    parameters = archive.manifest.get("run_metadata", {}).get("parameters", [])
    values: list[str] = []
    for parameter in parameters:
        field = str(parameter.get("value_field", "parameter_value"))
        raw_value = record.get(field)
        if raw_value is None:
            continue
        display_value = float(raw_value) * float(parameter.get("display_scale", 1.0))
        unit = str(parameter.get("unit", ""))
        values.append(
            f"{parameter.get('label', parameter.get('path', field))}={display_value:.6g}"
            + (f" {unit}" if unit else "")
        )
    condition = " · ".join(values) if values else lt("单次运行")
    status_label = lt(archive_status_label(record.get("status")))
    return (
        f"{lt('工况')} {index + 1:02d} · {condition} · {status_label}"
    )


def archive_notice_rows(archive: ResultArchive) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(archive.points):
        record = point.record
        point_name = archived_point_label(archive, index)
        if record.get("error"):
            rows.append(
                {
                    "点": point_name,
                    "级别": "error",
                    "来源": "执行异常",
                    "信息": record.get("error"),
                }
            )
        for issue in record.get("validation_failures", []):
            rows.append(
                {
                    "点": point_name,
                    "级别": "error",
                    "来源": "validation failure",
                    "信息": compact_validation_issue(issue),
                }
            )
        for issue in record.get("validation_warnings", []):
            rows.append(
                {
                    "点": point_name,
                    "级别": "warning",
                    "来源": "validation warning",
                    "信息": compact_validation_issue(issue),
                }
            )
        for message in record.get("solver_warnings", []):
            level, source = solver_notice_presentation(message)
            rows.append(
                {
                    "点": point_name,
                    "级别": level,
                    "来源": source,
                    "信息": str(message),
                }
            )
        for warning in record.get("python_warnings", []):
            location = ""
            if warning.get("filename"):
                location = f" ({warning['filename']}:{warning.get('line', '?')})"
            rows.append(
                {
                    "点": point_name,
                    "级别": "warning",
                    "来源": f"Python/{warning.get('category', 'Warning')}",
                    "信息": f"{warning.get('message', '')}{location}",
                }
            )
    return rows


def archive_notice_display_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Localize archive notices without changing their severity codes."""

    severity_labels = {
        "error": "错误",
        "warning": "警告",
        "info": "说明",
    }
    displayed: list[dict[str, Any]] = []
    for row in rows:
        item = copy.deepcopy(row)
        level = str(item.get("级别", ""))
        item["级别"] = lt(severity_labels.get(level, level))
        item["信息"] = localize_source_message(
            item.get("信息", ""), active_locale()
        )
        displayed.append(item)
    return displayed


def archive_point_table(archive: ResultArchive) -> list[dict[str, Any]]:
    metadata = archive.manifest.get("run_metadata", {})
    parameters = metadata.get("parameters", [])
    metric_keys = metadata.get("metric_keys", [])
    metrics = metadata.get("metrics", {})
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(archive.points):
        record = point.record
        solver_messages = list(record.get("solver_warnings", []))
        solver_warning_count = sum(
            solver_notice_presentation(message)[0] == "warning"
            for message in solver_messages
        )
        solver_info_count = len(solver_messages) - solver_warning_count
        row: dict[str, Any] = {
            "点": index + 1,
            "状态": archive_status_label(record.get("status")),
            "完整场": bool(
                record.get("result_available", point.result is not None)
            ),
            "求解器警告": solver_warning_count,
            "模型/显示说明": solver_info_count,
            "验证失败": len(record.get("validation_failures", [])),
            "验证警告": len(record.get("validation_warnings", [])),
            "执行错误": record.get("error"),
        }
        for parameter in parameters:
            field = str(parameter.get("value_field", "parameter_value"))
            if record.get(field) is None:
                continue
            unit = str(parameter.get("unit", ""))
            label = str(parameter.get("label", parameter.get("path", field)))
            row[f"{label} ({unit})" if unit else label] = (
                float(record[field]) * float(parameter.get("display_scale", 1.0))
            )
        for key in metric_keys:
            if record.get(key) is None:
                continue
            definition = metrics.get(key, {})
            unit = str(definition.get("unit", ""))
            label = str(definition.get("label", key))
            row[f"{label} ({unit})" if unit else label] = (
                float(record[key]) * float(definition.get("display_scale", 1.0))
            )
        rows.append(row)
    return rows


def render_archive_trend(archive: ResultArchive, archive_key: str) -> None:
    metadata = archive.manifest.get("run_metadata", {})
    metric_keys = [
        key
        for key in metadata.get("metric_keys", [])
        if any(point.record.get(key) is not None for point in archive.points)
    ]
    parameters = metadata.get("parameters", [])
    if not metric_keys or not parameters:
        return
    metrics = metadata.get("metrics", {})
    label_to_key = {
        str(metrics.get(key, {}).get("label", key)): key for key in metric_keys
    }
    selected_label = choice_selectbox(
        "总览趋势指标",
        list(label_to_key),
        key=f"archive_metric_{archive_key}",
        help="选择一个已保存的遍历输出量，在整组有效/无效工况上比较其变化趋势。",
    )
    metric_key = label_to_key[selected_label]
    metric = metrics.get(metric_key, {})
    metric_scale = float(metric.get("display_scale", 1.0))
    metric_unit = str(metric.get("unit", ""))
    valid_statuses = {"valid", "valid_with_warnings"}

    if len(parameters) == 1:
        parameter = parameters[0]
        value_field = str(parameter.get("value_field", "parameter_value"))
        parameter_scale = float(parameter.get("display_scale", 1.0))
        valid_records = [
            point.record
            for point in archive.points
            if point.record.get("status") in valid_statuses
            and point.record.get(value_field) is not None
            and sweep_metric_is_valid(point.record, metric_key)
        ]
        invalid_records = [
            point.record
            for point in archive.points
            if point.record.get("status") == "execution_completed_but_invalid"
            and point.record.get(value_field) is not None
            and sweep_metric_is_valid(point.record, metric_key)
        ]
        figure = go.Figure()
        # Keep execution failures and unavailable metrics as gaps; a backend
        # fallback must also start a new segment instead of implying one model.
        segments: list[tuple[str, list[dict[str, Any]]]] = []
        segment_open = False
        valid_ids = {id(record) for record in valid_records}
        for point in archive.points:
            record = point.record
            if id(record) not in valid_ids:
                segment_open = False
                continue
            model = str(record.get("executed_signal_model", "unknown"))
            if not segment_open or segments[-1][0] != model:
                segments.append((model, [record]))
                segment_open = True
            else:
                segments[-1][1].append(record)
        shown_models: set[str] = set()
        for model, records in segments:
            figure.add_scatter(
                x=[float(record[value_field]) * parameter_scale for record in records],
                y=[float(record[metric_key]) * metric_scale for record in records],
                mode="lines+markers",
                name=model,
                legendgroup=model,
                showlegend=model not in shown_models,
                line={"width": 2.5},
                marker={"size": 9},
            )
            shown_models.add(model)
        if invalid_records:
            figure.add_scatter(
                x=[float(record[value_field]) * parameter_scale for record in invalid_records],
                y=[float(record[metric_key]) * metric_scale for record in invalid_records],
                mode="markers",
                name="完成但无效",
                marker={"symbol": "x", "size": 11, "color": "#d62728"},
            )
        parameter_unit = str(parameter.get("unit", ""))
        parameter_label = str(parameter.get("label", parameter.get("path", "参数")))
        figure.update_layout(
            title=f"{selected_label} 随 {parameter_label} 的变化",
            xaxis_title=(
                f"{parameter_label} ({parameter_unit})"
                if parameter_unit
                else parameter_label
            ),
            yaxis_title=(
                f"{selected_label} ({metric_unit})" if metric_unit else selected_label
            ),
            xaxis_type=("log" if parameter.get("spacing", metadata.get("spacing", "linear")) == "log" else "linear"),
            height=410,
            margin={"t": 60, "l": 70, "r": 25, "b": 60},
        )
        st.plotly_chart(figure, width="stretch", key=f"archive_trend_{archive_key}")
        return

    parameter_x, parameter_y = parameters[:2]
    x_field = str(parameter_x.get("value_field", "parameter_x_value"))
    y_field = str(parameter_y.get("value_field", "parameter_y_value"))
    valid_records = [
        point.record
        for point in archive.points
        if point.record.get("status") in valid_statuses
        and point.record.get(x_field) is not None
        and point.record.get(y_field) is not None
        and sweep_metric_is_valid(point.record, metric_key)
    ]
    invalid_records = [
        point.record
        for point in archive.points
        if point.record.get("status") == "execution_completed_but_invalid"
        and point.record.get(x_field) is not None
        and point.record.get(y_field) is not None
        and sweep_metric_is_valid(point.record, metric_key)
    ]
    executed_models = list(dict.fromkeys(
        str(record.get("executed_signal_model", "unknown"))
        for record in valid_records + invalid_records
    ))
    if len(executed_models) > 1:
        selected_model = choice_selectbox(
            f"{selected_label}：实际 Seed 后端",
            executed_models,
            key=f"archive_grid_backend_{archive_key}_{metric_key}",
            help="不同实际后端的结果不会混入同一热图；切换此项分别查看。",
        )
        valid_records = [record for record in valid_records
                         if str(record.get("executed_signal_model", "unknown")) == selected_model]
        invalid_records = [record for record in invalid_records
                           if str(record.get("executed_signal_model", "unknown")) == selected_model]
    # Retain the complete saved grid so missing/invalid metrics cannot bridge
    # neighboring points in either slice family.
    coordinates = [point.record for point in archive.points
                   if point.record.get(x_field) is not None and point.record.get(y_field) is not None]
    x_values = sorted({float(record[x_field]) for record in coordinates})
    y_values = sorted({float(record[y_field]) for record in coordinates})
    values = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    x_lookup = {value: index for index, value in enumerate(x_values)}
    y_lookup = {value: index for index, value in enumerate(y_values)}
    display_records = valid_records + invalid_records
    for record in display_records:
        values[y_lookup[float(record[y_field])], x_lookup[float(record[x_field])]] = (
            float(record[metric_key]) * metric_scale
        )
    x_scale = float(parameter_x.get("display_scale", 1.0))
    y_scale = float(parameter_y.get("display_scale", 1.0))
    figure = go.Figure(
        go.Heatmap(
            x=np.asarray(x_values) * x_scale,
            y=np.asarray(y_values) * y_scale,
            z=values,
            colorscale="Viridis",
            colorbar={"title": metric_unit},
            hovertemplate=(
                f"{parameter_x.get('label', 'x')} %{{x:.6g}}<br>"
                f"{parameter_y.get('label', 'y')} %{{y:.6g}}<br>"
                f"{selected_label} %{{z:.6g}} {metric_unit}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=selected_label,
        xaxis_title=f"{parameter_x.get('label', 'x')} ({parameter_x.get('unit', '')})",
        yaxis_title=f"{parameter_y.get('label', 'y')} ({parameter_y.get('unit', '')})",
        xaxis_type="log" if parameter_x.get("spacing") == "log" else "linear",
        yaxis_type="log" if parameter_y.get("spacing") == "log" else "linear",
        height=450,
        margin={"t": 60, "l": 70, "r": 25, "b": 60},
    )
    add_invalid_grid_markers(
        figure, invalid_records, x_field=x_field, y_field=y_field,
        x_scale=x_scale, y_scale=y_scale, metric_key=metric_key,
        metric_scale=metric_scale, metric_unit=metric_unit,
    )
    st.plotly_chart(figure, width="stretch", key=f"archive_grid_{archive_key}")
    scan_x, scan_y = build_sweep_slice_figures(
        valid_records=valid_records,
        invalid_records=invalid_records,
        x_values=x_values,
        y_values=y_values,
        x_field=x_field,
        y_field=y_field,
        metric_key=metric_key,
        x_label=str(parameter_x.get("label", parameter_x.get("path", "x"))),
        x_unit=str(parameter_x.get("unit", "")),
        x_scale=x_scale,
        y_label=str(parameter_y.get("label", parameter_y.get("path", "y"))),
        y_unit=str(parameter_y.get("unit", "")),
        y_scale=y_scale,
        metric_label=selected_label,
        metric_unit=metric_unit,
        metric_scale=metric_scale,
        x_spacing=str(parameter_x.get("spacing", "linear")),
        y_spacing=str(parameter_y.get("spacing", "linear")),
    )
    st.markdown("#### 二维遍历切片曲线")
    slice_columns = st.columns(2)
    slice_columns[0].plotly_chart(
        scan_x,
        width="stretch",
        key=f"archive_grid_slice_x_{archive_key}_{metric_key}",
    )
    slice_columns[1].plotly_chart(
        scan_y,
        width="stretch",
        key=f"archive_grid_slice_y_{archive_key}_{metric_key}",
    )
    if invalid_records:
        st.caption(
            "曲线包含已完成但未通过有效性门的原始指标；"
            "无效点以红叉显示，有效曲线在这些位置断开；它们不属于定量结果。"
        )


def render_result_analysis_page() -> None:
    st.title("结果分析工作台")
    st.caption(
        "载入一个完整结果包，先检查整组工况的有效性与告警，再逐点恢复三维场、光路、光谱和诊断。"
    )
    with st.container(border=True):
        st.markdown("### 载入与复核流程")
        workflow = st.columns(3)
        workflow[0].markdown("**1 · 载入**  完整结果包与配置快照")
        workflow[1].markdown("**2 · 体检**  整组有效性、告警与失败点")
        workflow[2].markdown("**3 · 深入**  逐点三维场、光路和光谱")
        uploaded = st.file_uploader(
            "载入结果包（.ampres.zip）",
            type=["zip"],
            key="result_archive_upload",
            help=(
                "支持本网站下载的单次结果包、一维遍历结果包和二维遍历结果包；"
                "大型遍历包只先读取标量索引，切换工况时才加载该点完整场。"
            ),
        )
        st.caption(
            "载入不会重跑求解器，也不会覆盖当前表单。大型包不会一次解压全部三维场。"
        )
    if uploaded is not None:
        digest = uploaded_file_sha256(uploaded)
        if st.session_state.get("loaded_result_archive_digest") != digest:
            try:
                with st.spinner("正在校验结果包并载入遍历索引……"):
                    loaded = load_result_archive_index(
                        uploaded,
                        resource_budget=ArchiveResourceBudget(
                            maximum_archive_bytes=3 * 1024**3,
                        ),
                    )
            except Exception as exc:
                st.session_state.pop("loaded_result_archive", None)
                st.session_state.pop("loaded_result_archive_digest", None)
                st.session_state.pop("loaded_result_archive_source", None)
                st.error(f"结果包无法载入：{exc}")
                return
            st.session_state["loaded_result_archive"] = loaded
            st.session_state["loaded_result_archive_digest"] = digest
            st.session_state.pop("loaded_result_archive_point", None)
            st.session_state.pop("loaded_result_archive_point_key", None)
        # The uploader owns this seekable object across Streamlit reruns.  Keep
        # the current handle so one selected point can be restored on demand.
        st.session_state["loaded_result_archive_source"] = uploaded

    archive = st.session_state.get("loaded_result_archive")
    archive_key = str(st.session_state.get("loaded_result_archive_digest", "none"))[:12]
    if archive is None:
        st.info(
            "请先在“放大器仿真”页运行单个工况或参数遍历，并下载完整结果包；"
            "之后可在这里离线重新载入，不需要再次求解。"
        )
        st.caption(
            "结果包会同时保留配置、三维场、记录光学面、validation 结论、"
            "运行 warning 和自动依赖托管记录。"
        )
        return

    status_counts = archive.manifest.get("status_counts", {})
    notice_counts = archive.manifest.get("notice_counts", {})
    overview = st.columns(4)
    overview[0].metric("工况点", int(archive.manifest.get("point_count", 0)))
    overview[1].metric(
        "含完整场",
        int(archive.manifest.get("completed_result_count", 0)),
    )
    overview[2].metric(
        "有效点",
        int(status_counts.get("valid", 0))
        + int(status_counts.get("valid_with_warnings", 0)),
    )
    overview[3].metric(
        "失败/无效",
        int(status_counts.get("execution_failed", 0))
        + int(status_counts.get("execution_completed_but_invalid", 0)),
    )
    notices = archive_notice_rows(archive)
    actionable_notice_count = sum(
        str(item.get("级别")) in {"error", "warning"}
        for item in notices
    )
    informational_notice_count = len(notices) - actionable_notice_count
    if int(notice_counts.get("execution_errors", 0)) or int(
        notice_counts.get("validation_failures", 0)
    ):
        st.error(
            f"{NON_QUANTITATIVE_BANNER}\n\n"
            "这个结果包包含执行失败或物理有效性失败。下方总览保留了其它成功点，"
            "有效趋势在无效点处断开；无效点以红叉保留原始指标，不能作为可信定量结果。"
        )
    elif actionable_notice_count:
        if active_locale() == EN:
            st.warning(
                f"The result package contains {actionable_notice_count} "
                "warnings that require review. Check Issues and Notes before "
                "interpreting affected diagnostics."
            )
        else:
            st.warning(
                f"结果包包含 {actionable_notice_count} 条需要审阅的警告；"
                "请先查看问题与说明中心，再解释受影响的诊断。"
            )
    elif informational_notice_count:
        st.info(
            f"结果包只有 {informational_notice_count} 条模型合同或显示窗口说明，"
            "没有需要调整的数值警告。"
        )
    else:
        st.success("结果包中的全部工况均未记录错误或 warning。")

    render_archive_trend(archive, archive_key)
    if archive.base_config is not None:
        with st.expander("查看运行/遍历开始前的原始基准配置", expanded=False):
            st.json(archive.base_config)
    with st.expander("全部工况数据表", expanded=len(archive.points) > 1):
        st.dataframe(archive_point_table(archive), hide_index=True, width="stretch")
    with st.expander(
        f"{lt('问题与说明中心')} ({len(notices)})",
        expanded=bool(actionable_notice_count),
    ):
        if notices:
            st.dataframe(
                archive_notice_display_rows(notices),
                hide_index=True,
                width="stretch",
            )
        else:
            st.success("没有记录到执行错误、validation 问题或运行 warning。")

    selected_index = choice_selectbox(
        "切换工况点",
        list(range(len(archive.points))),
        format_func=lambda index: archived_point_label(archive, int(index)),
        key=f"archive_case_{archive_key}",
        help="切换后使用该点保存的实际配置和完整结果恢复光路、场、光谱及诊断。",
    )
    selected_index = int(selected_index)
    point_key = f"{archive_key}:{selected_index}"
    point = st.session_state.get("loaded_result_archive_point")
    if (
        point is None
        or st.session_state.get("loaded_result_archive_point_key") != point_key
    ):
        source = st.session_state.get("loaded_result_archive_source")
        if source is None:
            st.error("大型结果包的数据源已不可用；请重新选择该结果包。")
            return
        try:
            with st.spinner(f"正在按需恢复工况点 {selected_index + 1} 的完整场……"):
                point = load_result_archive_point(
                    source,
                    selected_index,
                    resource_budget=ArchiveResourceBudget(
                        maximum_archive_bytes=3 * 1024**3,
                    ),
                )
        except Exception as exc:
            st.error(f"工况点 {selected_index + 1} 无法载入：{exc}")
            return
        st.session_state["loaded_result_archive_point"] = point
        st.session_state["loaded_result_archive_point_key"] = point_key
    record = point.record
    st.divider()
    st.subheader(archived_point_label(archive, selected_index))
    status = str(record.get("status", "unknown"))
    if status == "execution_failed":
        st.error(record.get("error", "这个工况在执行阶段失败。"))
        if record.get("traceback"):
            with st.expander("查看完整异常堆栈", expanded=True):
                st.code(str(record["traceback"]), language="text")
    elif status == "execution_completed_but_invalid":
        st.error(
            f"{NON_QUANTITATIVE_BANNER}\n\n"
            "计算已完成，但没有通过物理有效性合同；下面的场只用于定位问题。"
        )
    elif status == "valid_with_warnings":
        st.warning("这个点通过阻断性合同，但包含需要审阅的诊断质量警告。")
    else:
        st.success("这个点通过结果有效性检查。")
    with st.expander("查看这个点的完整配置", expanded=False):
        st.json(point.config)
    if point.result is None:
        st.info("该点没有可绘制的场结果；配置、warning、错误与异常堆栈仍已完整保留。")
        return
    restored_grid = point.result.grid
    if active_locale() == EN:
        restored_message = (
            "Interactive results restored from the archive: "
            f"3D field {len(restored_grid.x)}×{len(restored_grid.y)}×{len(restored_grid.z)}; "
            f"{len(point.result.optical_planes)} recorded optical planes. "
            "The Interactive 3D Results Workbench opens with the draggable "
            "3D optical path; its second tab provides draggable x/y/z slices."
        )
    else:
        restored_message = (
            "已从结果包恢复交互式结果："
            f"三维体场 {len(restored_grid.x)}×{len(restored_grid.y)}×{len(restored_grid.z)}，"
            f"记录光学面 {len(point.result.optical_planes)} 个。"
            "下方“交互式三维结果工作台”默认显示可拖动旋转的三维光路；"
            "第二个页签提供 x/y/z 可拖动切片。"
        )
    st.info(restored_message)
    render_result(
        point.result,
        point.config,
        archive_record=record,
        archive_key=f"loaded_{archive_key}_{selected_index}",
    )


_SOLVER_STAGE_LABELS = {
    "initialization": "初始化网格、材料和后端",
    "operating_point": "Pump 与布居工作点",
    "population": "能级布居",
    "population_steady_state": "稳态布居固定点",
    "pump_propagation": "Pump 光路传播",
    "pump_waveform": "Pump 时间波形",
    "cycle_average": "周期平均 Pump + Seed 固定点",
    "pulse_map_initialization": "逐脉冲固定点初始化",
    "reduced_pulse_map": "低维离散脉冲初值加速",
    "pulse_map": "逐脉冲周期固定点",
    "pulse_train": "瞬态逐脉冲序列",
    "thermal_optical_fixed_point": "Pump–布居–热光固定点",
    "thermal_optical_closure": "热光最终闭合检查",
    "thermal_response": "热源与热光响应",
    "thermal_solver": "三维热场",
    "stress_solver": "应力、光弹与端面形变",
    "seed_propagation": "Seed 放大与传播",
    "seed_heat_postprocess": "Seed 不可逆损耗产热回算",
    "diagnostics": "能量账本与结果有效性检查",
    "complete": "全部求解完成",
}


def _progress_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "—"
    value = max(float(seconds), 0.0)
    if value < 60.0:
        return f"{value:.1f} s"
    minutes, remainder = divmod(int(round(value)), 60)
    if minutes < 60:
        return f"{minutes:d} min {remainder:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d} h {minutes:02d} min"


_PROGRESS_METRIC_LABELS = {
    "residual": "当前残差",
    "population_residual": "布居残差",
    "output_residual": "输出残差",
    "delta_n_residual": "Δn 残差",
    "free_carrier_residual": "载流子影响残差",
    "free_carrier_field_residual": "载流子场残差",
    "update_residual_K": "温度更新残差",
    "equation_residual_K": "热方程残差",
    "target_residual": "目标容差",
    "absorbed_pump_power_W": "吸收 Pump",
    "pump_output_power_W": "Pump 输出",
    "instantaneous_power_W": "瞬时 Pump",
    "pump_average_power_W": "平均 Pump",
    "seed_input_average_power_W": "Seed 输入平均功率",
    "output_power_W": "输出功率",
    "output_average_power_W": "输出平均功率",
    "transport_power_W": "当前传播功率",
    "output_energy_J": "输出脉冲能量",
    "transport_energy_J": "当前传播能量",
    "output_peak_power_W": "输出峰值功率",
    "compressed_peak_power_W": "压缩后峰值功率",
    "output_fwhm_s": "输出脉宽",
    "compressed_fwhm_s": "压缩后脉宽",
    "b_integral_rad": "累计 B 积分",
    "maximum_nonlinear_substep_rad": "最大非线性子步相位",
    "temporal_boundary_fraction": "时间窗边界能量",
    "spectral_peak_wavelength_m": "光谱峰值波长",
    "upper_population_fraction_mean": "上能级平均占比",
    "upper_population_fraction_max": "上能级最高占比",
    "excited_population_fraction_mean": "激发态平均占比",
    "excited_population_fraction_max": "激发态最高占比",
    "peak_temperature_K": "峰值温度",
    "thermal_time_s": "热场时间",
    "pump_time_s": "Pump 波形时间",
    "thermal_converged": "热场收敛",
    "converged": "阶段收敛",
    "validation_valid": "结果有效",
    "warning_count": "警告数",
    "pulse_index": "当前脉冲",
    "pulse_total": "脉冲总数",
    "repetition_rate_Hz": "重复频率",
    "seed_pulse_applied": "本周期 Seed 已入射",
    "awaiting_reported_seed_pulse": "等待最终报告脉冲",
    "stable_cycles": "连续稳定周期",
    "consecutive_cycles_required": "所需稳定周期",
    "pass": "当前程次",
    "crystal_cell": "程内晶体网格",
    "crystal_cells": "每程晶体网格",
    "z_position_m": "晶体 z 位置",
    "z_index": "z 网格索引",
    "inner_iterations": "热场内迭代",
    "backend": "Seed 后端",
    "initialization_source": "初值来源",
    "initialization_requested": "请求的初始化方式",
    "iteration_kind": "当前迭代类型",
    "initializer_used": "已用周期平均初值",
    "initializer_converged": "周期平均初值收敛",
    "initializer_iterations": "周期平均初值迭代数",
    "seed_substage": "Seed 子阶段",
    "source_channel_index": "当前光谱通道",
    "source_channel_total": "光谱通道总数",
    "source_channel_wavelength_m": "当前通道波长",
    "prepared_channel_energy_J": "当前通道输入能量",
    "prepared_channel_power_W": "当前通道输入功率",
    "spectral_channels": "光谱通道数",
    "optical_grid_nx": "光学网格 Nx",
    "optical_grid_ny": "光学网格 Ny",
    "working_field_estimate_bytes": "基础复场内存估计",
    "fixed_point_iteration": "固定点迭代",
    "fixed_point_total": "固定点迭代上限",
    "periodic_map_cycle": "逐脉冲周期",
    "periodic_map_model": "周期映射模型",
    "temporal_bins": "代理时间采样 / 通量包",
    "spectral_bins": "代理频谱采样",
    "adaptive_substeps": "当前晶体单元子步",
    "substep_index": "当前光学子步",
    "substep_total": "光学子步总数",
    "crystal_cell_state": "晶体单元状态",
    "cell_wall_time_s": "上一晶体单元耗时",
    "coherent_fidelity_mode": "相干精度档",
    "coherent_sample_count": "相干频率/时间点",
    "spatial_coordinate_model": "空间坐标模型",
    "axisymmetric_radial_bins": "轴对称径向分箱",
    "azimuthal_fidelity": "角向保真度",
    "maximum_azimuthal_order": "最大保留 |m|",
    "azimuthal_angular_samples": "角向配点",
    "transient_gain_enabled": "瞬态软窗增益",
    "transient_gain_slice_count": "当前软窗数",
    "transient_gain_maximum_slice_count": "最大软窗数",
    "transient_gain_maximum_depletion_fraction": "最大单窗耗尽",
}


_PROGRESS_VALUE_LABELS = {
    "configured_initial_population": "配置的初始布居",
    "initial_population": "初始布居",
    "cycle_average_initializer": "周期平均固定点初值",
    "pulse_map_cycle_average_initializer": "逐脉冲前的周期平均初值",
    "thermal_outer_warm_start": "热光外层热启动",
    "pulse_map_initializer": "逐脉冲前的周期平均固定点",
    "cycle_average_fixed_point": "周期平均固定点",
    "cycle_average": "周期平均",
    "reduced_pulse_map": "低维离散脉冲初值图",
    "reduced_pulse_map_initializer": "低维脉冲图初值",
    "auto": "自动选择",
    "spectral_source_setup": "宽带 Seed 光谱通道构造",
    "crystal_transport": "Seed 晶体内传播",
    "output_postprocess": "Seed 输出与脉冲诊断",
    "cycle_average_pump_probe": "周期平均 Pump 探针",
    "cycle_average_closure_pump": "周期平均最终 Pump 闭合探针",
    "starting": "正在进入晶体单元",
    "substep_complete": "光学子步已完成",
    "complete": "晶体单元已完成",
    "axisymmetric_radius": "轴对称半径 r",
    "finite_azimuthal_fourier_radius": "有限角向 Fourier × 半径 r",
    "m0": "基础轴对称 m=0",
    "m0_m1": "m=0+1",
    "m0_m1_m2_quadratic": "m=0+1+廉价 m=2",
    "m0_m1_m2": "m=0+1+完整 m=2",
    "axisymmetric_kerr": "L4 轴对称整脉冲 Kerr",
    "axisymmetric_transient_gain": "L4.5 轴对称瞬态 Kerr",
    "hybrid_whole_fluence": "L5 全场整脉冲 Kerr",
    "reference_transient_gain": "L6 全场瞬态 Kerr",
}


def _progress_scaled_value(value: float, units: tuple[tuple[float, str], ...]) -> str:
    magnitude = abs(value)
    for threshold, unit in units:
        if magnitude >= threshold:
            return f"{value / threshold:.4g} {unit}"
    threshold, unit = units[-1]
    return f"{value / threshold:.4g} {unit}"


def _progress_render_metric(key: str, value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "是" if bool(value) else "否"
    if not isinstance(value, (int, np.integer, float, np.floating)):
        return _PROGRESS_VALUE_LABELS.get(str(value), str(value))
    numeric = float(value)
    if not math.isfinite(numeric):
        return "尚不可判定"
    if key.endswith("_fraction"):
        return f"{100.0 * numeric:.3g}%"
    if key.endswith("_energy_J") or key == "output_energy_J":
        return _progress_scaled_value(
            numeric,
            ((1.0, "J"), (1e-3, "mJ"), (1e-6, "µJ"), (1e-9, "nJ"), (1e-12, "pJ")),
        )
    if key.endswith("_power_W") or key in {
        "instantaneous_power_W",
        "transport_power_W",
    }:
        return _progress_scaled_value(
            numeric,
            ((1e6, "MW"), (1e3, "kW"), (1.0, "W"), (1e-3, "mW"), (1e-6, "µW")),
        )
    if key in {"output_fwhm_s", "compressed_fwhm_s"}:
        return _progress_scaled_value(
            numeric,
            ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs"), (1e-9, "ns"), (1e-12, "ps"), (1e-15, "fs")),
        )
    if key in {"thermal_time_s", "pump_time_s"}:
        return _progress_scaled_value(
            numeric,
            ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs"), (1e-9, "ns")),
        )
    if key in {"spectral_peak_wavelength_m", "source_channel_wavelength_m"}:
        return f"{numeric * 1e9:.2f} nm"
    if key == "working_field_estimate_bytes":
        return _progress_scaled_value(
            numeric,
            ((1024.0**3, "GiB"), (1024.0**2, "MiB"), (1024.0, "KiB"), (1.0, "B")),
        )
    if key == "cell_wall_time_s":
        return _progress_duration(numeric)
    if key == "z_position_m":
        return f"{numeric * 1e3:.4g} mm"
    if key.endswith("_K"):
        return f"{numeric:.5g} K"
    if key.endswith("_rad"):
        return f"{numeric:.4g} rad"
    if "residual" in key or key == "target_residual":
        return f"{numeric:.3e}"
    if key == "repetition_rate_Hz":
        return _progress_scaled_value(
            numeric,
            ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"), (1.0, "Hz")),
        )
    return str(int(value)) if isinstance(value, (int, np.integer)) else f"{numeric:.4g}"


def _progress_context(event: dict[str, Any]) -> str:
    metrics = dict(event.get("metrics", {}))
    outer_context: list[str] = []
    if metrics.get("pulse_index") is not None:
        total = metrics.get("pulse_total") or event.get("total")
        outer_context.append(
            f"脉冲 {int(metrics['pulse_index'])}/{int(total)}"
            if total
            else f"脉冲 {int(metrics['pulse_index'])}"
        )
    elif metrics.get("fixed_point_iteration") is not None:
        total = metrics.get("fixed_point_total")
        iteration = int(metrics["fixed_point_iteration"])
        if "converged" in metrics:
            status = "已收敛" if bool(metrics["converged"]) else "未收敛"
            fixed_point_text = f"固定点{status}，共 {iteration} 轮"
        elif iteration > 0:
            fixed_point_text = f"固定点第 {iteration} 轮"
        else:
            fixed_point_text = "固定点准备中"
        if total:
            fixed_point_text += f"（上限 {int(total)}）"
        outer_context.append(fixed_point_text)
    substage = str(metrics.get("seed_substage", ""))
    if substage == "spectral_source_setup":
        channel = metrics.get("source_channel_index", event.get("current"))
        total = metrics.get("source_channel_total", event.get("total"))
        channel_text = f"光谱通道 {int(channel or 0)}/{int(total or 0)}"
        wavelength = metrics.get("source_channel_wavelength_m")
        if wavelength is not None:
            channel_text += (
                " · λ="
                + _progress_render_metric(
                    "source_channel_wavelength_m", wavelength
                )
            )
        return " · ".join([*outer_context, channel_text])
    if substage == "output_postprocess":
        output_text = f"输出诊断 {int(event.get('current') or 0)}/{int(event.get('total') or 0)}"
        return " · ".join([*outer_context, output_text])
    if metrics.get("pass_index") is not None:
        pass_total = metrics.get("pass_total")
        cell = metrics.get("crystal_cell")
        cells = metrics.get("crystal_cells")
        pass_text = f"程次 {int(metrics['pass_index'])}/{int(pass_total)}" if pass_total else f"程次 {int(metrics['pass_index'])}"
        if cell is not None and cells is not None:
            pass_text += f" · 晶体单元 {int(cell)}/{int(cells)}"
        substep = metrics.get("substep_index")
        substep_total = metrics.get("substep_total")
        if substep is not None and substep_total is not None:
            pass_text += f" · 子步 {int(substep)}/{int(substep_total)}"
        radial_bins = metrics.get("axisymmetric_radial_bins")
        if radial_bins is not None:
            pass_text += f" · 径向 {int(radial_bins)}"
        soft_windows = metrics.get("transient_gain_slice_count")
        maximum_windows = metrics.get("transient_gain_maximum_slice_count")
        if soft_windows is not None:
            pass_text += f" · 软窗 {int(soft_windows)}"
            if maximum_windows is not None:
                pass_text += f"/{int(maximum_windows)}"
        return " · ".join([*outer_context, pass_text])
    if metrics.get("pump_time_s") is not None:
        return " · ".join(
            [
                *outer_context,
                f"Pump t={_progress_render_metric('pump_time_s', metrics['pump_time_s'])}",
            ]
        )
    if metrics.get("thermal_time_s") is not None:
        return f"热场 t={_progress_render_metric('thermal_time_s', metrics['thermal_time_s'])}"
    if outer_context:
        return " · ".join(outer_context)
    current, total = event.get("current"), event.get("total")
    if current is not None and total is not None:
        return f"迭代 {int(current)}/{int(total)}"
    return "阶段切换"


def _progress_stage_count(event: dict[str, Any]) -> str:
    """Render an iteration counter without treating a convergence cap as a total."""

    metrics = dict(event.get("metrics", {}))
    if str(event.get("stage", "")) == "cycle_average":
        iteration = metrics.get("fixed_point_iteration", event.get("current"))
        limit = metrics.get("fixed_point_total", event.get("total"))
        if "converged" in metrics:
            status = "已收敛" if bool(metrics["converged"]) else "未收敛"
            text = f"{status} · 共 {int(iteration or 0)} 轮"
        elif iteration is not None and int(iteration) > 0:
            text = f"第 {int(iteration)} 轮"
        else:
            text = "准备中"
        if limit is not None:
            text += f"（上限 {int(limit)}）"
        return text

    current, total = event.get("current"), event.get("total")
    if current is not None and total is not None:
        return f"{int(current)}/{int(total)}"
    if current is not None:
        return str(int(current))
    return "—"


def _progress_residual_result(event: dict[str, Any]) -> str:
    metrics = dict(event.get("metrics", {}))
    values = []
    for key in (
        "residual",
        "population_residual",
        "output_residual",
        "delta_n_residual",
        "free_carrier_residual",
        "update_residual_K",
        "equation_residual_K",
    ):
        if metrics.get(key) is not None:
            values.append(
                f"{_PROGRESS_METRIC_LABELS[key]} {_progress_render_metric(key, metrics[key])}"
            )
        if len(values) >= 3:
            break
    return "；".join(values) or "—"


def _progress_latest_metric(
    events: list[dict[str, Any]],
    *keys: str,
) -> tuple[str, Any] | None:
    for event in reversed(events):
        metrics = dict(event.get("metrics", {}))
        for key in keys:
            value = metrics.get(key)
            if value is not None:
                return key, value
    return None


def _progress_placeholder_figure(title: str, message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": "#9ca3af"},
    )
    figure.update_layout(
        height=255,
        margin=dict(l=12, r=12, t=42, b=26),
        title=title,
        xaxis={"visible": False, "fixedrange": True},
        yaxis={"visible": False, "fixedrange": True},
        template="plotly_dark",
    )
    return figure


def _progress_pulse_history_figure(events: list[dict[str, Any]]) -> go.Figure:
    pulse_records: list[
        tuple[tuple[str, int | None], int, float, float | None]
    ] = []
    fallback: tuple[float, float | None] | None = None
    for event in events:
        metrics = dict(event.get("metrics", {}))
        energy = metrics.get("output_energy_J")
        if not isinstance(energy, (int, float, np.integer, np.floating)):
            continue
        numeric = float(energy)
        if not math.isfinite(numeric) or numeric < 0.0:
            continue
        average_power = metrics.get("output_average_power_W")
        power = (
            float(average_power)
            if isinstance(average_power, (int, float, np.integer, np.floating))
            and math.isfinite(float(average_power))
            else None
        )
        fallback = (numeric, power)
        stage = str(event.get("stage", "unknown"))
        if stage not in {"pulse_train", "pulse_map", "reduced_pulse_map"}:
            continue
        index = metrics.get("pulse_index", metrics.get("periodic_map_cycle"))
        if index is None:
            continue
        outer_iteration = metrics.get("thermal_optical_iteration")
        try:
            outer_iteration = (
                None if outer_iteration is None else int(outer_iteration)
            )
        except (TypeError, ValueError):
            outer_iteration = None
        pulse_records.append(
            ((stage, outer_iteration), int(index), numeric, power)
        )
    if pulse_records:
        current_traversal = pulse_records[-1][0]
        values = [
            (pulse_index, energy, power)
            for traversal, pulse_index, energy, power in pulse_records
            if traversal == current_traversal
        ]
    else:
        values = []
    if not values and fallback is not None:
        values = [(1, fallback[0], fallback[1])]
    records = values
    if not records:
        return _progress_placeholder_figure(
            "Seed 输出历史",
            "完成第一个 Seed 传播后显示输出能量；不会逐事件新增表格。",
        )
    figure = go.Figure()
    figure.add_scatter(
        x=[item[0] for item in records],
        y=[item[1] * 1e9 for item in records],
        mode="lines+markers",
        name="输出能量",
        line={"color": "#38bdf8"},
    )
    power_records = [item for item in records if item[2] is not None]
    if power_records:
        figure.add_scatter(
            x=[item[0] for item in power_records],
            y=[float(item[2]) for item in power_records],
            mode="lines",
            name="等效平均功率",
            yaxis="y2",
            line={"color": "#f59e0b", "dash": "dot"},
        )
    figure.update_layout(
        height=255,
        margin=dict(l=12, r=12, t=42, b=34),
        title="Seed 输出历史",
        xaxis_title="周期图评估 / 已报告脉冲",
        yaxis_title="能量 (nJ)",
        yaxis2={
            "title": "平均功率 (W)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        legend={"orientation": "h", "y": 1.13, "x": 0.0},
        template="plotly_dark",
        uirevision="stable-pulse-history",
    )
    return figure


def _progress_spectrum_preview_figure(
    events: list[dict[str, Any]],
) -> go.Figure:
    for event in reversed(events):
        metrics = dict(event.get("metrics", {}))
        wavelength = np.asarray(
            metrics.get("spectrum_preview_wavelength_m", []), dtype=float
        )
        relative = np.asarray(
            metrics.get("spectrum_preview_relative", []), dtype=float
        )
        if (
            wavelength.ndim == 1
            and relative.ndim == 1
            and len(wavelength) >= 2
            and wavelength.shape == relative.shape
            and np.all(np.isfinite(wavelength))
            and np.all(np.isfinite(relative))
        ):
            figure = go.Figure()
            figure.add_scatter(
                x=wavelength * 1e9,
                y=np.clip(relative, 0.0, None),
                mode="lines",
                name="输出 dE/dλ",
                line={"color": "#ef4444", "width": 2.2},
            )
            figure.update_layout(
                height=255,
                margin=dict(l=12, r=12, t=42, b=34),
                title="最近一次输出光谱预览",
                xaxis_title="波长 (nm)",
                yaxis_title="归一化 dE/dλ",
                yaxis={"range": [0.0, 1.05]},
                template="plotly_dark",
                uirevision="stable-spectrum-preview",
            )
            return figure
    return _progress_placeholder_figure(
        "最近一次输出光谱预览",
        "光谱只在首个、每 5 个及最终脉冲更新，避免刷新拖慢求解。",
    )


class LiveSolveProgressPanel:
    """Render one fixed-height, restart-safe view of background progress."""

    def __init__(self, placeholder: Any, *, key_prefix: str = "solve-progress"):
        self.events: list[dict[str, Any]] = []
        self._last_fraction = 0.0
        self._key_prefix = key_prefix
        self._chart_render_sequence = 0
        with placeholder.container(border=True):
            self.title_slot = st.empty()
            self.title_slot.markdown("### 阶段求解进度")
            self.progress_bar = st.progress(0.0)
            self.stage_slot = st.empty()
            primary_columns = st.columns(5)
            self.total_slot = primary_columns[0].empty()
            self.position_slot = primary_columns[1].empty()
            self.output_slot = primary_columns[2].empty()
            self.residual_slot = primary_columns[3].empty()
            self.time_slot = primary_columns[4].empty()
            physics_columns = st.columns(6)
            self.pump_slot = physics_columns[0].empty()
            self.population_slot = physics_columns[1].empty()
            self.temperature_slot = physics_columns[2].empty()
            self.b_integral_slot = physics_columns[3].empty()
            self.pulse_width_slot = physics_columns[4].empty()
            self.spectral_peak_slot = physics_columns[5].empty()
            self.detail_slot = st.empty()
            chart_columns = st.columns(2)
            self.pulse_chart_slot = chart_columns[0].empty()
            self.spectrum_chart_slot = chart_columns[1].empty()
            self.history_slot = st.empty()
            st.caption(
                "当前热—光遍历的全部已完成脉冲都画在同一张图中；脉宽和光谱只保留"
                "最新快照。轴向单元、光学子步、Pump 采样和热线性迭代不发送到网页；"
                "页面只接收完整脉冲和固定点阶段结果，这些显示数据不参与物理解算。"
            )

    def _latest_text(self, label: str, *keys: str) -> tuple[str, str]:
        latest = _progress_latest_metric(self.events, *keys)
        if latest is None:
            return label, "—"
        key, value = latest
        return label, _progress_render_metric(key, value)

    def _completed_stage_text(self) -> str:
        completed: list[str] = []
        stages = [str(event.get("stage", "unknown")) for event in self.events]
        for index, stage in enumerate(stages[:-1]):
            if stages[index + 1] == stage:
                continue
            label = _SOLVER_STAGE_LABELS.get(stage, stage)
            if label not in completed:
                completed.append(label)
        if not completed:
            return "已经过：正在启动第一个求解阶段"
        return "已经过：" + " → ".join(completed[-5:])

    def _stage_rate(self, latest: dict[str, Any]) -> float | None:
        stage = str(latest.get("stage", "unknown"))
        current = latest.get("current")
        if current is None:
            return None
        stage_events = [
            event
            for event in self.events
            if str(event.get("stage", "unknown")) == stage
            and event.get("current") is not None
        ]
        if len(stage_events) < 2:
            return None
        first = stage_events[0]
        elapsed = float(latest.get("elapsed_s", 0.0) or 0.0) - float(
            first.get("elapsed_s", 0.0) or 0.0
        )
        completed = int(current) - int(first.get("current") or 0)
        if elapsed <= 0.0 or completed <= 0:
            return None
        return completed / elapsed

    def _render_empty(self) -> None:
        self.progress_bar.progress(0.0)
        self.stage_slot.markdown("**当前阶段：后台进程启动**　·　阶段进度 `—`")
        self.total_slot.metric("总进度", "0.00%")
        self.position_slot.metric("当前求解位置", "等待首个事件")
        self.output_slot.metric("最近输出", "—")
        self.residual_slot.metric("当前残差", "—")
        self.time_slot.metric("已用 / 粗略剩余", "0.0 s / —")
        self.pump_slot.metric("吸收 Pump", "—")
        self.population_slot.metric("平均激发态", "—")
        self.temperature_slot.metric("峰值温度", "—")
        self.b_integral_slot.metric("累计 B 积分", "—")
        self.pulse_width_slot.metric("输出脉宽", "—")
        self.spectral_peak_slot.metric("输出光谱峰值", "—")
        self.detail_slot.caption("后台工作进程已创建，正在载入不可变配置快照。")
        self._render_charts()
        self.history_slot.caption("已经过：正在启动第一个求解阶段")

    def _render_charts(self) -> None:
        chart_suffix = f"-{self._chart_render_sequence}"
        self._chart_render_sequence += 1
        self.pulse_chart_slot.plotly_chart(
            _progress_pulse_history_figure(self.events),
            width="stretch",
            config={"displayModeBar": False},
            key=f"{self._key_prefix}-pulse-history{chart_suffix}",
        )
        self.spectrum_chart_slot.plotly_chart(
            _progress_spectrum_preview_figure(self.events),
            width="stretch",
            config={"displayModeBar": False},
            key=f"{self._key_prefix}-spectrum-preview{chart_suffix}",
        )

    def _render(self) -> None:
        if not self.events:
            self._render_empty()
            return
        latest = self.events[-1]
        fractions = [
            min(max(float(event["overall_fraction"]), 0.0), 1.0)
            for event in self.events
            if event.get("overall_fraction") is not None
            and math.isfinite(float(event["overall_fraction"]))
        ]
        self._last_fraction = max(self._last_fraction, *fractions, 0.0)
        self.progress_bar.progress(self._last_fraction)

        stage = str(latest.get("stage", "unknown"))
        label = _SOLVER_STAGE_LABELS.get(stage, stage)
        iteration_text = _progress_stage_count(latest)
        self.stage_slot.markdown(
            f"**当前阶段：{label}**　·　阶段进度 `{iteration_text}`"
        )

        elapsed = float(latest.get("elapsed_s", 0.0) or 0.0)
        eta: float | None = None
        if 0.0 < self._last_fraction < 1.0:
            eta = elapsed * (1.0 - self._last_fraction) / self._last_fraction
        output = _progress_latest_metric(
            self.events,
            "output_energy_J",
            "output_average_power_W",
            "output_power_W",
        )
        output_text = (
            "—"
            if output is None
            else _progress_render_metric(output[0], output[1])
        )
        residual_text = _progress_residual_result(latest)
        if residual_text == "—":
            for event in reversed(self.events[:-1]):
                residual_text = _progress_residual_result(event)
                if residual_text != "—":
                    break
        rate = self._stage_rate(latest)
        position = _progress_context(latest)
        if rate is not None:
            position += f" · {rate:.3g} 步/s"
        self.total_slot.metric("总进度", f"{100.0 * self._last_fraction:.2f}%")
        self.position_slot.metric("当前求解位置", position)
        self.output_slot.metric("最近输出", output_text)
        self.residual_slot.metric("最近收敛残差", residual_text)
        self.time_slot.metric(
            "已用 / 粗略剩余",
            f"{_progress_duration(elapsed)} / {_progress_duration(eta)}",
        )

        for slot, (metric_label, metric_value) in (
            (self.pump_slot, self._latest_text("吸收 Pump", "absorbed_pump_power_W")),
            (
                self.population_slot,
                self._latest_text(
                    "平均激发态",
                    "upper_population_fraction_mean",
                    "excited_population_fraction_mean",
                ),
            ),
            (self.temperature_slot, self._latest_text("峰值温度", "peak_temperature_K")),
            (self.b_integral_slot, self._latest_text("累计 B 积分", "b_integral_rad")),
            (
                self.pulse_width_slot,
                self._latest_text("输出脉宽", "output_fwhm_s"),
            ),
            (
                self.spectral_peak_slot,
                self._latest_text("输出光谱峰值", "spectral_peak_wavelength_m"),
            ),
        ):
            slot.metric(metric_label, metric_value)
        message = str(latest.get("message", "")).strip()
        self.detail_slot.caption(message or "当前阶段正在建立下一份物理量快照。")
        self._render_charts()
        self.history_slot.caption(self._completed_stage_text())
        if str(latest.get("state", "running")) != "running" and stage == "complete":
            self.title_slot.success(
                f"求解完成，总用时 {_progress_duration(elapsed)}"
            )

    def restore(self, events: list[dict[str, Any]]) -> None:
        """Render the newest immutable worker snapshot into fixed slots."""

        self.events = [copy.deepcopy(event) for event in events]
        self._render()

    def reset(self) -> None:
        """Clear one completed solve before reusing the fixed panel."""

        self.events = []
        self._last_fraction = 0.0
        self.title_slot.markdown("### 阶段求解进度")
        self._render_empty()

    def update(self, event: dict[str, Any]) -> None:
        """Append one synchronous event while retaining the fixed layout."""

        _compact_live_progress_events(self.events, event)
        self._render()

    def fail(self, exc: Exception) -> None:
        self.title_slot.error(f"求解停止：{type(exc).__name__}: {exc}")
        self.detail_slot.caption("面板保留停止前最后一份成功的物理量快照。")


class LiveSweepProgressPanel:
    """Combine sweep-level progress with the current point's solver panel."""

    def __init__(self, request: dict[str, Any]) -> None:
        self.metric_keys = tuple(request["metric_keys"])
        if request.get("dimension", 1) == 2:
            self.parameter_fields = (
                (request["parameter"], "parameter_x_value"),
                (request["secondary_parameter"], "parameter_y_value"),
            )
        else:
            self.parameter_fields = (
                (request["parameter"], "parameter_value"),
            )
        self.current_point: int | None = None
        self.events: list[dict[str, Any]] = []
        self.last_render_monotonic_s = 0.0
        self.completed_rows: list[dict[str, str]] = []

        with st.container(border=True):
            st.markdown("### 参数遍历阶段进度")
            overview_columns = st.columns(3)
            self.point_slot = overview_columns[0].empty()
            self.overall_slot = overview_columns[1].empty()
            self.status_slot = overview_columns[2].empty()
            self.progress_bar = st.progress(0.0)
            self.progress_text = st.empty()
            self.solve_panel = LiveSolveProgressPanel(
                st.empty(),
                key_prefix=f"sweep-live-{uuid.uuid4().hex}",
            )
            st.markdown("#### 本次遍历已完成的全部点")
            self.completed_points_slot = st.empty()
            st.caption(
                "当前点只接收完整脉冲和固定点阶段事件；界面最多每 5 秒读取一次。"
                "这只降低网页观测频率，不会改变求解器内部的物理分辨率。"
            )

    def point_label(self, record: dict[str, Any]) -> str:
        parts: list[str] = []
        for definition, field in self.parameter_fields:
            value = record.get(field)
            if value is None:
                continue
            displayed = float(value) * float(definition.display_scale)
            suffix = f" {definition.unit}" if definition.unit else ""
            parts.append(f"{definition.label}={displayed:.6g}{suffix}")
        return " · ".join(parts) or "正在建立点配置"

    def _completed_row(
        self,
        completed: int,
        total: int,
        record: dict[str, Any],
    ) -> tuple[str, dict[str, str]]:
        state = {
            "valid": "有效",
            "valid_with_warnings": "有效（有诊断警告）",
            "execution_completed_but_invalid": "完成但无效",
            "execution_failed": "执行失败",
        }.get(str(record.get("status")), "处理中")
        point_number = int(record.get("point_index", completed - 1)) + 1
        row: dict[str, str] = {
            "点": f"{point_number}/{total}",
            "参数": self.point_label(record),
            "状态": state,
        }
        for key in self.metric_keys:
            definition = SWEEP_METRICS[key]
            value = record.get(key)
            rendered = "—"
            if value is not None:
                try:
                    numeric = float(value) * float(definition.display_scale)
                except (TypeError, ValueError):
                    pass
                else:
                    if math.isfinite(numeric):
                        rendered = f"{numeric:.6g}"
                        if definition.unit:
                            rendered += f" {definition.unit}"
            row[definition.label] = rendered
        return state, row

    def update_point(
        self,
        point: int,
        total: int,
        record: dict[str, Any],
        event: dict[str, Any],
    ) -> None:
        """Render a throttled copy of one in-flight solver event."""

        if self.current_point != point:
            if self.current_point is not None:
                self.solve_panel.reset()
            self.current_point = point
            self.events = []
            self.last_render_monotonic_s = 0.0

        _compact_live_progress_events(self.events, event)

        local_fraction = event.get("overall_fraction")
        try:
            local_fraction = float(local_fraction)
        except (TypeError, ValueError):
            local_fraction = 0.0
        if not math.isfinite(local_fraction):
            local_fraction = 0.0
        local_fraction = min(max(local_fraction, 0.0), 1.0)
        overall_fraction = min(
            max(((point - 1) + local_fraction) / max(total, 1), 0.0),
            1.0,
        )

        now = time.monotonic()
        final_event = (
            str(event.get("state", "running")) != "running"
            or str(event.get("stage", "")) == "complete"
        )
        if (
            self.last_render_monotonic_s > 0.0
            and now - self.last_render_monotonic_s < 1.0
            and not final_event
        ):
            return
        self.last_render_monotonic_s = now

        self.progress_bar.progress(overall_fraction)
        self.point_slot.metric("当前遍历点", f"{point}/{total}")
        self.overall_slot.metric(
            "遍历总进度",
            f"{100.0 * overall_fraction:.2f}%",
        )
        self.status_slot.metric(
            "当前点进度",
            f"{100.0 * local_fraction:.2f}%",
        )
        self.progress_text.caption(
            f"第 {point}/{total} 点 · {self.point_label(record)} · "
            f"{_SOLVER_STAGE_LABELS.get(str(event.get('stage')), str(event.get('stage')))}"
        )
        self.solve_panel.restore(self.events)

    def complete_point(
        self,
        completed: int,
        total: int,
        record: dict[str, Any],
    ) -> None:
        """Update the stable, scalar-only table after one point finishes."""

        self.progress_bar.progress(completed / max(total, 1))
        state, row = self._completed_row(completed, total, record)
        self.point_slot.metric("当前遍历点", f"{completed}/{total}")
        self.overall_slot.metric(
            "遍历总进度",
            f"{100.0 * completed / max(total, 1):.2f}%",
        )
        self.status_slot.metric("最近完成点", state)
        self.progress_text.caption(
            f"参数遍历 {completed}/{total}：{state} · {self.point_label(record)}"
        )

        self.completed_rows.append(row)
        self.completed_points_slot.dataframe(
            self.completed_rows,
            hide_index=True,
            width="stretch",
            height=420,
        )

    def restore(
        self,
        *,
        completed_records: tuple[dict[str, Any], ...],
        current_point: int | None,
        total: int,
        current_record: dict[str, Any] | None,
        events: tuple[dict[str, Any], ...],
        status: str,
    ) -> None:
        """Render one bounded manager snapshot without retaining old 3D fields."""

        self.completed_rows = [
            self._completed_row(index, total, record)[1]
            for index, record in enumerate(completed_records, start=1)
        ]
        if self.completed_rows:
            self.completed_points_slot.dataframe(
                self.completed_rows,
                hide_index=True,
                width="stretch",
                height=420,
            )
        else:
            self.completed_points_slot.caption("尚无完整遍历点可下载。")

        completed = len(completed_records)
        local_fraction = 0.0
        latest = events[-1] if events else None
        if latest is not None:
            try:
                local_fraction = float(latest.get("overall_fraction") or 0.0)
            except (TypeError, ValueError):
                local_fraction = 0.0
            if not math.isfinite(local_fraction):
                local_fraction = 0.0
            local_fraction = min(max(local_fraction, 0.0), 1.0)
        in_flight = current_point is not None and current_point > completed
        overall = (completed + (local_fraction if in_flight else 0.0)) / max(
            total, 1
        )
        self.progress_bar.progress(min(max(overall, 0.0), 1.0))
        shown_point = current_point or min(completed + 1, total)
        self.point_slot.metric("当前遍历点", f"{shown_point}/{total}")
        self.overall_slot.metric("遍历总进度", f"{100.0 * overall:.2f}%")
        self.status_slot.metric(
            "任务状态",
            {
                "running": "正在计算",
                "paused": "已暂停",
                "completed": "已完成",
                "cancelled": "已取消",
                "failed": "已失败",
            }.get(status, status),
        )
        if latest is not None:
            label = self.point_label(current_record or {})
            stage = _SOLVER_STAGE_LABELS.get(
                str(latest.get("stage")), str(latest.get("stage"))
            )
            self.progress_text.caption(
                f"第 {shown_point}/{total} 点 · {label} · {stage}"
            )
            self.solve_panel.restore(list(events))
        elif completed:
            state, _row = self._completed_row(
                completed, total, completed_records[-1]
            )
            self.progress_text.caption(
                f"已持久化 {completed}/{total} 点 · 最近完成点：{state}"
            )
        else:
            self.progress_text.caption("后台进程正在建立第一个遍历点。")

    def complete(self) -> None:
        self.progress_text.success("参数遍历完成")

    def fail(self, exc: Exception) -> None:
        self.solve_panel.fail(exc)
        self.progress_text.error(f"参数遍历停止：{type(exc).__name__}: {exc}")


class ParallelSweepProgressPanel:
    """Render scalar-only status cards for concurrent full-result points."""

    def __init__(self, request: dict[str, Any]) -> None:
        self.metric_keys = tuple(request["metric_keys"])
        self.point_workers = max(2, min(int(request.get("point_workers", 2)), 3))
        if request.get("dimension", 1) == 2:
            self.parameter_fields = (
                (request["parameter"], "parameter_x_value"),
                (request["secondary_parameter"], "parameter_y_value"),
            )
        else:
            self.parameter_fields = (
                (request["parameter"], "parameter_value"),
            )
        with st.container(border=True):
            st.markdown("### 并行参数遍历进度")
            overview = st.columns(3)
            self.active_slot = overview[0].empty()
            self.overall_slot = overview[1].empty()
            self.completed_slot = overview[2].empty()
            self.progress_bar = st.progress(0.0)
            st.markdown("#### 正在运行的点")
            self.point_slots = st.columns(self.point_workers)
            st.markdown("#### 本次遍历已完成的全部点")
            self.completed_points_slot = st.empty()
            st.caption(
                "并行模式只传输每个点的阶段、状态和百分比，不生成实时场图；"
                "完整场与全部诊断仍在后台按点保存。"
            )

    def point_label(self, record: dict[str, Any]) -> str:
        parts: list[str] = []
        for definition, field in self.parameter_fields:
            value = record.get(field)
            if value is None:
                continue
            displayed = float(value) * float(definition.display_scale)
            suffix = f" {definition.unit}" if definition.unit else ""
            parts.append(f"{definition.label}={displayed:.6g}{suffix}")
        return " · ".join(parts) or "正在建立点配置"

    def _completed_row(
        self, total: int, record: dict[str, Any]
    ) -> dict[str, str]:
        state = {
            "valid": "有效",
            "valid_with_warnings": "有效（有诊断警告）",
            "execution_completed_but_invalid": "完成但无效",
            "execution_failed": "执行失败",
        }.get(str(record.get("status")), "处理中")
        point_number = int(record.get("point_index", 0)) + 1
        row: dict[str, str] = {
            "点": f"{point_number}/{total}",
            "参数": self.point_label(record),
            "状态": state,
        }
        for key in self.metric_keys:
            definition = SWEEP_METRICS[key]
            value = record.get(key)
            rendered = "—"
            if value is not None:
                try:
                    numeric = float(value) * float(definition.display_scale)
                except (TypeError, ValueError):
                    pass
                else:
                    if math.isfinite(numeric):
                        rendered = f"{numeric:.6g}"
                        if definition.unit:
                            rendered += f" {definition.unit}"
            row[definition.label] = rendered
        return row

    @staticmethod
    def _fraction(events: tuple[dict[str, Any], ...]) -> float:
        if not events:
            return 0.0
        try:
            fraction = float(events[-1].get("overall_fraction") or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(fraction):
            return 0.0
        return min(max(fraction, 0.0), 1.0)

    def restore(
        self,
        *,
        completed_records: tuple[dict[str, Any], ...],
        active_points: tuple[dict[str, Any], ...],
        total: int,
        status: str,
    ) -> None:
        rows = [self._completed_row(total, record) for record in completed_records]
        if rows:
            self.completed_points_slot.dataframe(
                rows,
                hide_index=True,
                width="stretch",
                height=420,
            )
        else:
            self.completed_points_slot.caption("尚无完整遍历点可下载。")

        fractions = [
            self._fraction(tuple(item.get("events", ())))
            for item in active_points
        ]
        overall = (
            len(completed_records) + sum(fractions)
        ) / max(total, 1)
        overall = min(max(overall, 0.0), 1.0)
        self.progress_bar.progress(overall)
        self.active_slot.metric(
            "并行活动点",
            f"{len(active_points)}/{self.point_workers}",
        )
        self.overall_slot.metric("遍历总进度", f"{100.0 * overall:.2f}%")
        self.completed_slot.metric(
            "已完成点", f"{len(completed_records)}/{total}"
        )

        for slot_index, slot in enumerate(self.point_slots):
            with slot.container(border=True):
                if slot_index >= len(active_points):
                    st.metric(f"并行槽 {slot_index + 1}", "空闲")
                    st.caption(
                        "等待下一个点"
                        if status in ACTIVE_STATUSES
                        else "本次遍历已停止"
                    )
                    continue
                item = active_points[slot_index]
                point = int(item.get("point", 0))
                record = dict(item.get("record", {}))
                events = tuple(item.get("events", ()))
                fraction = self._fraction(events)
                latest = events[-1] if events else {}
                stage_key = str(latest.get("stage", "initialization"))
                stage = _SOLVER_STAGE_LABELS.get(stage_key, stage_key)
                st.metric(
                    f"第 {point}/{total} 点",
                    f"{100.0 * fraction:.2f}%",
                )
                st.caption(self.point_label(record))
                st.caption(f"当前状态：{stage}")


def _solve_session_id() -> str:
    context = get_script_run_ctx(suppress_warning=True)
    if context is not None:
        return str(context.session_id)
    if "solve_session_fallback_id" not in st.session_state:
        st.session_state["solve_session_fallback_id"] = uuid.uuid4().hex
    return str(st.session_state["solve_session_fallback_id"])


def _persist_terminal_solve_job(view: SolveJobView) -> None:
    """Move one immutable worker outcome into this browser's result state."""

    events = [copy.deepcopy(event) for event in view.events]
    st.session_state["last_solver_progress_events"] = events
    st.session_state["last_solver_job_id"] = view.job_id
    st.session_state["last_solver_snapshot_digest"] = view.config_digest
    if view.status == "completed" and view.result is not None:
        completed_result = view.result
        st.session_state["advanced_result"] = completed_result
        st.session_state["advanced_config"] = copy.deepcopy(view.config_snapshot)
        advanced_record = diagnostic_record_for_result(
            completed_result,
            python_warnings=list(view.python_warnings),
            ui_compatibility_adjustments=copy.deepcopy(
                view.compatibility_adjustments
            ),
            solve_job_id=view.job_id,
            solve_config_sha256=view.config_digest,
        )
        if view.archive_path is not None:
            advanced_record["autosaved_archive_path"] = view.archive_path
        if view.archive_error is not None:
            advanced_record["autosave_archive_error"] = view.archive_error
        st.session_state["advanced_record"] = advanced_record
        st.session_state.pop("last_single_failure", None)
        st.session_state.pop("last_single_cancellation", None)
        return
    if view.status == "failed":
        failure = copy.deepcopy(view.failure or {})
        failure_record = {
            "status": "execution_failed",
            "error": failure.get("error", "Background solve failed"),
            "error_type": failure.get("error_type", "WorkerProcessError"),
            "error_message": failure.get(
                "error_message", "Background solve failed"
            ),
            "traceback": failure.get("traceback", ""),
            "solver_warnings": [],
            "python_warnings": list(failure.get("python_warnings", [])),
            "validation_failures": [],
            "validation_warnings": [],
            "ui_compatibility_adjustments": copy.deepcopy(
                view.compatibility_adjustments
            ),
            "solve_job_id": view.job_id,
            "solve_config_sha256": view.config_digest,
        }
        st.session_state["last_single_failure"] = {
            "record": failure_record,
            "config": copy.deepcopy(view.config_snapshot),
        }
        st.session_state.pop("last_single_cancellation", None)
        return
    if view.status in {"cancelled", "browser_disconnected"}:
        st.session_state["last_single_cancellation"] = {
            "status": view.status,
            "reason": view.cancellation_reason,
            "config": copy.deepcopy(view.config_snapshot),
            "config_digest": view.config_digest,
            "job_id": view.job_id,
        }
        st.session_state.pop("last_single_failure", None)


def _synchronize_solve_job(session_id: str) -> SolveJobView | None:
    view = get_solve_job(session_id)
    if view is None or view.status in ACTIVE_STATUSES:
        return view
    _persist_terminal_solve_job(view)
    release_solve_job(session_id)
    return None


def _render_solve_controls(view: SolveJobView, session_id: str) -> None:
    controls = st.columns([1, 1, 1.35])
    if view.status == "running":
        if controls[0].button(
            "暂停求解",
            key=f"pause_solve_{view.job_id}",
            width="stretch",
            help="冻结独立工作进程；配置快照、进度和内存状态都会保留。",
        ):
            pause_solve_job(session_id)
            st.rerun()
    else:
        if controls[0].button(
            "继续求解",
            key=f"resume_solve_{view.job_id}",
            type="primary",
            width="stretch",
            help="从原来的数值状态继续，不会重新初始化或更换配置。",
        ):
            resume_solve_job(session_id)
            st.rerun()
    controls[1].metric(
        "任务状态",
        "已暂停" if view.status == "paused" else "正在计算",
    )
    if controls[2].button(
        "停止并取消本次求解",
        key=f"cancel_solve_{view.job_id}",
        width="stretch",
        help="这是唯一会因页面按钮操作而终止当前单点求解的控制。未完成结果不会归档。",
    ):
        cancel_solve_job(session_id)
        st.rerun()


@st.fragment(run_every=5.0)
def render_active_solve_job_fragment(
    session_id: str,
    *,
    compact: bool = False,
) -> None:
    view = get_solve_job(session_id)
    if view is None:
        return
    if view.status not in ACTIVE_STATUSES:
        _persist_terminal_solve_job(view)
        release_solve_job(session_id)
        st.rerun()
    with st.container(border=True):
        if view.status == "paused":
            st.warning("求解已明确暂停；数值进程仍保留，只有继续或取消按钮会改变它。")
        else:
            st.info(
                "求解运行在独立进程中；修改表单、切换页面或生成完整配置不会停止、"
                "重启或改写这次任务。"
            )
        st.caption(
            f"运行配置快照 SHA-256：`{view.config_digest}` · "
            f"后台 PID：`{view.pid}`。当前表单修改只供下一次运行使用；结果包固定使用这个快照。"
        )
        if not compact:
            panel = LiveSolveProgressPanel(
                st.empty(), key_prefix=f"active-{view.job_id}"
            )
            panel.restore(list(view.events))
        elif view.events:
            latest = view.events[-1]
            st.progress(float(latest.get("overall_fraction") or 0.0))
            st.caption(
                f"后台求解：{_SOLVER_STAGE_LABELS.get(str(latest.get('stage')), str(latest.get('stage')))}"
                f" · {_progress_context(latest)}"
            )
        _render_solve_controls(view, session_id)


def _persist_terminal_sweep_job(view: SweepJobView) -> None:
    """Expose a completed lightweight sweep while full fields remain on disk."""

    already_persisted = (
        st.session_state.get("last_sweep_job_id") == view.job_id
    )
    # Refresh the full-result association on every rerun.  Session-state keys
    # can disappear independently of the still-live job registry; returning
    # before restoring this key used to expose the summary-only fallback link.
    st.session_state["last_sweep_job_id"] = view.job_id
    st.session_state["sweep_archive_job_id"] = view.job_id
    st.session_state["sweep_base_config"] = copy.deepcopy(view.config_snapshot)
    if already_persisted:
        return
    request = view.request_snapshot
    st.session_state["sweep_metric_keys"] = list(request["metric_keys"])
    if view.status == "completed" and view.summary is not None:
        if int(request.get("dimension", 1)) == 2:
            st.session_state["sweep_grid_run"] = view.summary
            st.session_state["sweep_grid_parameter_x"] = request["parameter"]
            st.session_state["sweep_grid_parameter_y"] = request[
                "secondary_parameter"
            ]
            st.session_state["sweep_grid_spacing_x"] = request["spacing"]
            st.session_state["sweep_grid_spacing_y"] = request[
                "secondary_spacing"
            ]
            st.session_state.pop("sweep_run", None)
        else:
            st.session_state["sweep_run"] = view.summary
            st.session_state["sweep_parameter_definition"] = request[
                "parameter"
            ]
            st.session_state["sweep_result_spacing"] = request["spacing"]
            st.session_state.pop("sweep_grid_run", None)
    st.session_state["last_sweep_terminal"] = {
        "job_id": view.job_id,
        "status": view.status,
        "failure": copy.deepcopy(view.failure),
        "cancellation_reason": view.cancellation_reason,
        "completed_points": len(view.completed_records),
        "requested_points": view.total_points,
    }


def _render_sweep_controls(view: SweepJobView, session_id: str) -> None:
    controls = st.columns([1, 1, 1.35])
    point_workers = int(view.request_snapshot.get("point_workers", 1))
    if view.status == "running":
        if controls[0].button(
            "暂停遍历",
            key=f"pause_sweep_{view.job_id}",
            width="stretch",
            help=(
                "冻结所有正在运行的点及整个遍历工作进程；已完成点仍可立即下载。"
                if point_workers > 1
                else "冻结当前点及整个遍历工作进程；已完成点仍可立即下载。"
            ),
        ):
            pause_sweep_job(session_id)
            st.rerun()
    elif view.status == "paused":
        if controls[0].button(
            "继续遍历",
            key=f"resume_sweep_{view.job_id}",
            type="primary",
            width="stretch",
            help="从当前点原有的数值状态继续，不重复已经完成的点。",
        ):
            resume_sweep_job(session_id)
            st.rerun()
    controls[1].metric(
        "已完成点",
        f"{len(view.completed_records)}/{view.total_points}",
    )
    if view.status in ACTIVE_STATUSES and controls[2].button(
        "停止并取消本次遍历",
        key=f"cancel_sweep_{view.job_id}",
        width="stretch",
        help="终止当前未完成点；已经落盘的完成点会继续保留并可下载。",
    ):
        cancel_sweep_job(session_id)
        st.rerun()
    elif view.status not in ACTIVE_STATUSES and controls[2].button(
        "丢弃本次遍历",
        key=f"discard_sweep_{view.job_id}",
        width="stretch",
        help="删除本次遍历的临时结果并释放新遍历；此操作不能恢复。",
    ):
        release_sweep_job(session_id)
        for key in (
            "last_sweep_job_id",
            "sweep_archive_job_id",
            "sweep_base_config",
            "sweep_run",
            "sweep_grid_run",
            "last_sweep_terminal",
        ):
            st.session_state.pop(key, None)
        st.session_state["sweep_job_active"] = False
        st.rerun()


@st.fragment(run_every=5.0)
def render_sweep_job_fragment(session_id: str, *, compact: bool = False) -> None:
    view = get_sweep_job(session_id)
    if view is None:
        return
    if view.status not in ACTIVE_STATUSES:
        newly_terminal = (
            st.session_state.get("last_sweep_job_id") != view.job_id
        )
        _persist_terminal_sweep_job(view)
        if newly_terminal:
            st.rerun()
    with st.container(border=True):
        if view.status == "paused":
            st.warning(
                "参数遍历已暂停：当前点被冻结；下方下载只包含已经完整结束并落盘的点。"
            )
        elif view.status == "running":
            st.info(
                "参数遍历在独立后台进程中运行；关闭或刷新网页不会取消任务。"
            )
        elif view.status == "completed":
            st.success("参数遍历已完成，全部点均已持久化。")
        elif view.status == "cancelled":
            st.warning("参数遍历已取消；已完成点仍可下载。")
        else:
            failure = view.failure or {}
            st.error(
                "参数遍历异常停止："
                + str(failure.get("error", "后台遍历进程失败"))
            )
        if not compact:
            point_workers = int(view.request_snapshot.get("point_workers", 1))
            if point_workers > 1:
                parallel_panel = ParallelSweepProgressPanel(
                    view.request_snapshot
                )
                parallel_panel.restore(
                    completed_records=view.completed_records,
                    active_points=view.active_points,
                    total=view.total_points,
                    status=view.status,
                )
            else:
                panel = LiveSweepProgressPanel(view.request_snapshot)
                panel.restore(
                    completed_records=view.completed_records,
                    current_point=view.current_point,
                    total=view.total_points,
                    current_record=view.current_record,
                    events=view.events,
                    status=view.status,
                )
        else:
            fraction = len(view.completed_records) / max(view.total_points, 1)
            st.progress(fraction)
            st.caption(
                f"参数遍历：已完整持久化 {len(view.completed_records)}/{view.total_points} 点"
            )
        _render_sweep_controls(view, session_id)
        if view.completed_records:
            partial = (
                view.status != "completed"
                or len(view.completed_records) < view.total_points
            )
            file_name = (
                "amplifier_sweep_partial.ampres.zip"
                if partial
                else "amplifier_sweep_2d_results.ampres.zip"
                if int(view.request_snapshot.get("dimension", 1)) == 2
                else "amplifier_sweep_results.ampres.zip"
            )
            render_streaming_archive_link(
                "下载已完成的部分" if partial else "下载全部遍历结果包",
                writer=lambda destination: write_sweep_job_archive(
                    session_id,
                    destination,
                ),
                file_name=file_name,
                cache_key=f"live-sweep:{session_id}:{view.job_id}",
                button_type=(
                    "primary" if view.status == "paused" else "secondary"
                ),
                help_text=(
                    "只打包已经完成的点；当前暂停或计算中的点不会写入。"
                    "完整三维结果从后端逐点读取并直接流向浏览器。"
                ),
            )
        else:
            st.caption("第一个点尚未完成，因此目前还没有可下载的部分。")


def _validation_location(message: str) -> str:
    lowered = str(message).lower()
    routes = (
        (("radial_bins",), "当前求解器的径向格点数，或④ 数值 → 光学 Nx/Ny"),
        (("multipass", "between_pass"), "① 模型 → 启用多程 / ② 光束 → 多程模板"),
        (("polarization", "birefringence", "jones"), "② 光束 → Seed → Jones 偏振与双折射"),
        (("spatial_gain_transport", "gaussian_mode_overlap", "parabolic_gain_fit"), "② 光束 → Seed 增益输运"),
        (("periodic", "pulse-map", "pulse_map", "reduced_pulse_map", "cycle_average"), "① 模型 → Pump · Seed · 热场收敛"),
        (("thermal_optical", "multiphysics", "seed_heat_coupling"), "① 模型 → Pump · Seed · 热场收敛"),
        (("commercial_app", "signal_model", "nonlinear_phase", "nonlinear_loss"), "① 模型 → 运行方式与求解 / 高级求解器"),
        (("grid", "nx", "ny", "nz", "spacing_xy"), "④ 数值"),
        (("material.thermal", "temperature", "conductivity", "heat_"), "③ 介质 → 热源与边界"),
        (("material.mechanical", "stress", "photoelastic", "endface"), "③ 介质 → 应力与光弹"),
        (("energy_levels", "lifetime", "cross section", "population"), "③ 介质 → 晶体、材料与能级"),
        (("pump.", "pump_"), "② 光束 → Pump"),
        (("seed.", "coherent", "josa", "spectral"), "② 光束 → Seed"),
        (("ase",), "③ 介质 → ASE / 寄生激光"),
    )
    for needles, location in routes:
        if any(needle in lowered for needle in needles):
            return location
    return "侧栏“专家模式”或完整 JSON；错误信息中的字段路径可直接搜索"


def _current_compatibility_adjustment_rows() -> list[dict[str, Any]]:
    """Return only the reducer changes belonging to the active form generation."""

    generation = int(st.session_state.get("form_widget_generation", 0))
    stored = st.session_state.get("compatibility_adjustments", {})
    if int(stored.get("generation", -1)) != generation:
        return []
    return copy.deepcopy(list(stored.get("rows", [])))


def _current_compatibility_ownership() -> list[dict[str, Any]]:
    return copy.deepcopy(
        list(st.session_state.get("compatibility_ownership", []))
    )


def _current_compatibility_release_rows() -> list[dict[str, Any]]:
    generation = int(st.session_state.get("form_widget_generation", 0))
    stored = st.session_state.get("compatibility_release_event", {})
    if int(stored.get("generation", -1)) != generation:
        return []
    return copy.deepcopy(list(stored.get("rows", [])))


def render_config_health(
    config: dict[str, Any],
    validation_error: Exception | None,
) -> None:
    """Show compatibility actions and localize blockers before a run."""

    rows = _current_compatibility_adjustment_rows()
    ownership = _current_compatibility_ownership()
    released_rows = _current_compatibility_release_rows()
    with st.container(border=True):
        st.markdown("### 配置体检与模块联动")
        if validation_error is not None:
            message = str(validation_error)
            st.error("当前配置尚不能运行，求解器没有启动。")
            st.markdown(f"**问题：** `{type(validation_error).__name__}: {message}`")
            if "empty radial annuli" in message:
                st.caption("有径向环没有对应的横向采样点。请减小当前求解器的径向格点数，或增加光学网格 Nx/Ny。")
            st.info(f"请到：**{_validation_location(message)}**")
        elif rows:
            st.success(
                f"当前有 {len(rows)} 项从属设置由依赖合同自动托管；配置已通过运行前校验。"
            )
        elif released_rows:
            st.success(
                f"最近已释放 {len(released_rows)} 项不再需要的从属设置并恢复原值；当前配置可运行。"
            )
        else:
            labels = _active_module_display(config)
            st.success(
                "运行前相容性检查通过："
                f"{labels['pump']} → {labels['population']} → "
                f"{labels['signal']} → {labels['thermal']}"
            )
        if rows:
            with st.expander("查看自动托管明细（原值 → 生效值）", expanded=False):
                st.dataframe(rows, hide_index=True, width="stretch")
                st.caption(
                    "上级条件关闭后，这些值会自动恢复联动前设置；材料常数、光束尺寸、"
                    "功率和能量不会被相容性联动猜测。"
                )
                if st.button(
                    "保留当前联动值，转为手动设置",
                    key="take_over_compatibility_values",
                    help=(
                        "清除这些值的自动托管标记；以后关闭上级功能时，"
                        "只要配置仍然相容，当前值就不会被恢复。"
                    ),
                ):
                    st.session_state["compatibility_ownership"] = []
                    st.session_state["compatibility_adjustments"] = {
                        "generation": int(
                            st.session_state.get(
                                "form_widget_generation", 0
                            )
                        ),
                        "rows": [],
                    }
                    st.session_state.pop(
                        "compatibility_release_event", None
                    )
                    st.rerun()
        contracts = dependency_contracts(config, ownership)
        with st.expander(
            f"查看 {len(contracts)} 条核心模块合同", expanded=False
        ):
            contract_columns = st.columns(2)
            for index, contract in enumerate(contracts):
                with contract_columns[index % 2]:
                    with st.container(border=True):
                        st.markdown(
                            f"**{lt(contract.module)}** · "
                            f"`{lt(contract.status)}`"
                        )
                        st.caption(
                            f"{lt('用户选择')} · `{contract.requested}`"
                        )
                        st.caption(
                            f"{lt('实际生效')} · `{contract.effective}`"
                        )
                        st.caption(
                            f"{lt('编辑位置')} · {lt(contract.location)}"
                        )
            st.caption(
                "“用户选择”保留入口意图，“实际生效”来自预设解析和全部依赖闭包；"
                "自动托管表示该行至少有一个从属值会随上级条件释放。"
            )
        if st.button(
            "生成 / 刷新当前模型完整说明",
            key="generate_model_description_health",
            help="在光路预览下方展开当前实际生效的 Pump、布居、热、Seed、非线性和验证链路。",
        ):
            st.session_state["show_model_description"] = True


solve_session_id = _solve_session_id()
st.session_state["solve_session_id"] = solve_session_id
solve_job_view = _synchronize_solve_job(solve_session_id)
st.session_state["solve_job_active"] = bool(
    solve_job_view is not None and solve_job_view.status in ACTIVE_STATUSES
)
sweep_job_view = reconcile_sweep_session(solve_session_id)
if sweep_job_view is not None and sweep_job_view.status not in ACTIVE_STATUSES:
    _persist_terminal_sweep_job(sweep_job_view)
st.session_state["sweep_job_active"] = bool(
    sweep_job_view is not None and sweep_job_view.status in ACTIVE_STATUSES
)

app_page = choice_radio(
    "页面",
    ["放大器仿真", "结果分析"],
    key="app_page",
    help="放大器仿真用于设置并运行；结果分析用于载入结果包、比较遍历并逐点查看。",
)
_previous_app_page = st.session_state.get("last_rendered_app_page")
if _previous_app_page is None:
    st.session_state["last_rendered_app_page"] = app_page
elif app_page != _previous_app_page:
    st.session_state["last_rendered_app_page"] = app_page
    if app_page == "放大器仿真":
        # Streamlit's browser may replay zero/first-option values for keyed
        # widgets that disappeared while the analysis page was open.  Refill a
        # fresh widget generation from the last committed configuration before
        # rendering any form control, so stale browser state cannot mutate the
        # physical model on page return.
        st.session_state["amplifier_page_visit_generation"] = (
            int(st.session_state.get("amplifier_page_visit_generation", 0)) + 1
        )
        install_form_config(
            copy.deepcopy(
                st.session_state.get("form_base_config", DEFAULTS)
            ),
            update_json_editor=False,
        )
if app_page == "结果分析":
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            min-width: 300px;
            max-width: 300px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if solve_job_view is not None and solve_job_view.status in ACTIVE_STATUSES:
        render_active_solve_job_fragment(solve_session_id, compact=True)
    if sweep_job_view is not None and (
        sweep_job_view.status in ACTIVE_STATUSES
        or bool(sweep_job_view.completed_records)
    ):
        render_sweep_job_fragment(solve_session_id, compact=True)
    render_result_analysis_page()
    st.stop()


st.title("固体激光放大器工作台")
st.caption("从冷态光路到 Pump、布居、增益、热场和相干脉冲的同一套可追溯模型")

config = copy.deepcopy(
    st.session_state.get("form_base_config", DEFAULTS)
)
configuration_error: Exception | None = None
form_input_errors: list[str] = []
try:
    config, run_requested, form_input_errors = make_config()
    validate_config(config)
    from amplifier_sim.ui_compat import validate_available_backends
    validate_available_backends(config)
except Exception as exc:
    configuration_error = exc
    run_requested = False

execution_error = configuration_error or (
    ValueError(" ".join(form_input_errors)) if form_input_errors else None
)
if execution_error is not None:
    run_requested = False

sweep_request: dict[str, Any] | None = None
try:
    sweep_request = sweep_sidebar_controls(
        config,
        blocked_reason=(
            f"{type(execution_error).__name__}: {execution_error}"
            if execution_error is not None
            else "已有后台单点求解正在运行"
            if st.session_state.get("solve_job_active", False)
            else "已有后台参数遍历正在运行"
            if st.session_state.get("sweep_job_active", False)
            else None
        ),
    )
except Exception as exc:
    st.sidebar.error(str(exc))

warm_start_request = warm_start_sidebar_controls(
    blocked_reason=(
        "已有后台单点或参数遍历正在运行；该设置将在下一次运行时生效。"
        if st.session_state.get("solve_job_active", False)
        or st.session_state.get("sweep_job_active", False)
        else None
    ),
)

render_configuration_overview(config)
render_config_health(config, execution_error)
if run_requested:
    try:
        solve_job_view = start_solve_job(
            solve_session_id,
            config,
            project_dir=PROJECT_DIR,
            archive_output_directory=UI_AUTOSAVE_DIRECTORY,
            compatibility_adjustments=(
                _current_compatibility_adjustment_rows()
            ),
            initial_continuation_archive=warm_start_request,
        )
    except Exception as exc:
        st.exception(exc)
    else:
        st.session_state["solve_job_active"] = True
        st.session_state.pop("last_single_cancellation", None)
        st.rerun()
if solve_job_view is not None and solve_job_view.status in ACTIVE_STATUSES:
    render_active_solve_job_fragment(solve_session_id)
elif (
    st.session_state.get("last_solver_progress_events")
    and st.session_state.get("advanced_record", {}).get("solve_job_id")
    == st.session_state.get("last_solver_job_id")
):
    with st.expander("最近一次完整求解进度", expanded=False):
        completed_panel = LiveSolveProgressPanel(
            st.empty(),
            key_prefix=f"completed-{st.session_state['last_solver_job_id']}",
        )
        completed_panel.restore(
            list(st.session_state["last_solver_progress_events"])
        )
if sweep_job_view is not None and (
    sweep_job_view.status in ACTIVE_STATUSES
    or bool(sweep_job_view.completed_records)
):
    render_sweep_job_fragment(solve_session_id)
if configuration_error is None:
    render_realtime_bench(config)
else:
    st.subheader("光路预览")
    st.info(
        "当前配置未通过校验，实时光路预览已暂停。修正上方指出的参数后，"
        "预览会自动恢复。"
    )
render_generated_model_description(config)

if sweep_request is not None:
    try:
        sweep_job_view = start_sweep_job(
            solve_session_id,
            config,
            sweep_request,
            project_dir=PROJECT_DIR,
            compatibility_adjustments=(
                _current_compatibility_adjustment_rows()
            ),
            initial_continuation_archive=warm_start_request,
        )
    except Exception as exc:
        st.exception(exc)
    else:
        st.session_state["sweep_job_active"] = True
        st.session_state.pop("sweep_run", None)
        st.session_state.pop("sweep_grid_run", None)
        st.session_state.pop("last_sweep_terminal", None)
        st.session_state.pop("sweep_archive_job_id", None)
        st.rerun()

if "sweep_grid_run" in st.session_state:
    render_sweep_grid(
        st.session_state["sweep_grid_run"],
        st.session_state["sweep_grid_parameter_x"],
        st.session_state["sweep_grid_parameter_y"],
        st.session_state["sweep_metric_keys"],
        st.session_state.get("sweep_grid_spacing_x", "linear"),
        st.session_state.get("sweep_grid_spacing_y", "linear"),
    )
elif "sweep_run" in st.session_state:
    render_sweep(
        st.session_state["sweep_run"],
        st.session_state["sweep_parameter_definition"],
        st.session_state["sweep_metric_keys"],
        st.session_state.get(
            "sweep_result_spacing",
            st.session_state.get("sweep_spacing", "linear"),
        ),
    )

if "last_single_cancellation" in st.session_state:
    cancelled_run = st.session_state["last_single_cancellation"]
    st.info(
        "最近一次单点求解已按明确操作取消；未完成状态没有覆盖更早的成功结果。"
        f"配置快照：`{cancelled_run.get('config_digest', 'unknown')}`"
    )

if "last_single_failure" in st.session_state:
    failed_run = st.session_state["last_single_failure"]
    failed_record = failed_run["record"]
    failed_config = failed_run["config"]
    st.subheader("最近一次单点运行失败")
    st.error(failed_record.get("error", "运行失败"))
    with st.expander("查看完整异常堆栈", expanded=False):
        st.code(str(failed_record.get("traceback", "")), language="text")
    failed_archive_points = (
        ArchivePoint(
            record=failed_record,
            config=failed_config,
            result=None,
        ),
    )
    failed_archive_metadata = {
        "source": "failed_single_run",
        "ui_compatibility_adjustments": failed_record.get(
            "ui_compatibility_adjustments", []
        ),
    }
    render_streaming_archive_link(
        "下载失败诊断结果包",
        writer=lambda destination: write_result_archive(
            destination,
            failed_archive_points,
            run_kind="single",
            base_config=failed_config,
            run_metadata=failed_archive_metadata,
        ),
        file_name="amplifier_failed_run.ampres.zip",
        cache_key=f"failed-single:{failed_record.get('solve_config_sha256', id(failed_record))}",
        button_type="primary",
        help_text="保留失败时的完整配置、Python warning、异常类型、消息和堆栈，可在结果分析页载入。",
    )

if "advanced_result" in st.session_state:
    st.divider()
    st.header("本次运行结果")
    st.caption(
        "以下内容来自一次完整、不可拼接的配置快照；表单后续修改只更新上方冷态预览，"
        "直到再次运行才会替换这里的布居、热场、增益和输出结果。"
    )
    result_config = st.session_state["advanced_config"]
    if "last_single_failure" in st.session_state:
        st.warning(
            "下面保留的是更早一次成功运行的结果，不是刚才失败工况的部分结果；"
            "失败工况只保存在上方诊断结果包中。"
        )
    if json.dumps(config, sort_keys=True, default=str) != json.dumps(
        result_config, sort_keys=True, default=str
    ):
        changed_sections = [
            key
            for key in sorted(set(config) | set(result_config))
            if json.dumps(config.get(key), sort_keys=True, default=str)
            != json.dumps(result_config.get(key), sort_keys=True, default=str)
        ]
        section_labels = {
            "simulation": "工况/运行方式",
            "grid": "网格",
            "material": "材料/晶体",
            "pump": "Pump",
            "seed": "Seed",
            "energy_levels": "能级",
            "optical_train": "简化光路",
            "physics": "物理后端",
            "multiphysics": "多物理耦合",
        }
        changed_text = "、".join(
            section_labels.get(key, key) for key in changed_sections
        )
        st.warning(
            f"当前表单的 {changed_text} 已修改；下面仍是上一次完整配置快照的结果。"
            "冷态光路预览会实时更新，但布居、热透镜、增益和运行后 D4σ 不会局部拼接更新。"
            "请重新点击“运行当前配置”。"
        )
    render_result(
        st.session_state["advanced_result"],
        st.session_state["advanced_config"],
        archive_record=st.session_state.get("advanced_record"),
    )
else:
    st.info(
        "从左侧五个工作区完成设置后点击“运行当前配置”。① 模型选择求解方法，② 光束设置输入与光路，"
        "③ 介质设置材料与热场，④ 数值设置分辨率，⑤ 结果记录选择需要保存的诊断。正式计算前请做收敛检查。"
    )
