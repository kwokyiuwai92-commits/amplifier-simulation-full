"""Shared validators for public configuration values."""

from __future__ import annotations

import math
from typing import Any


def require_exact_integer(
    value: Any,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return an integer only when the supplied value is exactly integral.

    JSON booleans are integers in Python and ``int(48.9)`` silently truncates.
    Neither behavior is acceptable for a public discretization or iteration
    contract because the executed model would differ from the requested one.
    """

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an exact finite integer; got {value!r}")
    result = int(numeric)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result
