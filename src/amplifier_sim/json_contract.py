"""Strict JSON input and portable JSON output helpers."""

from __future__ import annotations

import json
import math
import numbers
from typing import IO, Any


def _reject_nonstandard_constant(token: str) -> None:
    raise ValueError(
        f"Non-standard JSON numeric constant {token!r} is not allowed; "
        "use a finite number or null"
    )


def load_json_strict(handle: IO[str]) -> Any:
    """Load standards-compliant JSON and reject NaN/Infinity spellings."""

    return json.load(handle, parse_constant=_reject_nonstandard_constant)


def loads_json_strict(text: str) -> Any:
    """Decode standards-compliant JSON and reject NaN/Infinity spellings."""

    return json.loads(text, parse_constant=_reject_nonstandard_constant)


def json_portable(value: Any) -> Any:
    """Replace non-finite diagnostic floats with JSON ``null`` recursively."""

    if isinstance(value, dict):
        return {key: json_portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_portable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def dumps_json_portable(value: Any, **kwargs: Any) -> str:
    """Serialize without emitting implementation-specific NaN/Infinity tokens."""

    return json.dumps(json_portable(value), allow_nan=False, **kwargs)
