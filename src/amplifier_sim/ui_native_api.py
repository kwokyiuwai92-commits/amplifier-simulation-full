"""Capture original UI APIs once under Python's module import lock.

Import this module before installing localization wrappers. Concurrent sessions
share immutable API references, never a previous session's wrapper or state.
"""
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

_LOCALIZED_UI_FUNCTIONS = (
    "selectbox",
    "radio",
    "checkbox",
    "toggle",
    "multiselect",
    "slider",
    "number_input",
    "text_input",
    "text_area",
    "file_uploader",
    "button",
    "download_button",
    "markdown",
    "caption",
    "title",
    "header",
    "subheader",
    "info",
    "warning",
    "error",
    "success",
    "metric",
    "tabs",
    "expander",
    "popover",
    "spinner",
    "dataframe",
    "data_editor",
    "plotly_chart",
    "write",
)

ST_NATIVE_INPUTS = {name: getattr(st, name) for name in _LOCALIZED_UI_FUNCTIONS}
DG_NATIVE_INPUTS = {name: getattr(DeltaGenerator, name) for name in _LOCALIZED_UI_FUNCTIONS}
