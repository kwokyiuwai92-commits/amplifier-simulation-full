"""Typed failures that optional physics backends may safely fall back from."""

from __future__ import annotations


class ModelApplicabilityError(ValueError):
    """The requested backend is valid, but not applicable to this input/grid."""


class ModelConvergenceError(RuntimeError):
    """A numerically valid optional backend did not meet its convergence contract."""


class ModelInputError(ValueError):
    """A user-supplied optional-model input violates a declared contract."""


class CoherentInputError(ModelInputError):
    """A user-supplied coherent field or profile violates its input contract."""


class PulseDefinitionInputError(ModelInputError):
    """A Fourier-linked temporal/spectral pulse definition is invalid."""


class PhaseRetrievalError(ModelConvergenceError):
    """A phase-retrieval model failed its declared convergence contract."""


OPTIONAL_MODEL_FAILURES = (
    ModelApplicabilityError,
    ModelConvergenceError,
    ModelInputError,
)
OPTIONAL_BACKEND_FAILURES = (ImportError, *OPTIONAL_MODEL_FAILURES)
