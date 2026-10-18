"""Monte-Carlo uncertainty propagation and experiment-facing calibration.

The routines in this module deliberately operate on the public simulator and
summary contracts.  They do not introduce a second, simplified amplifier
model: every accepted sample is a complete configured run, while failed or
strictly-invalid samples remain visible and are excluded from confidence
intervals and calibration objectives.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Any

import numpy as np

from .sweep import (
    SWEEP_METRICS,
    _coherent_kernel_workers_for_processes,
    extract_standard_metrics,
    get_config_value,
    set_config_value,
    standard_metric_validity,
)
from .validation import require_exact_integer


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    """Vectorised standard-normal CDF without a mandatory SciPy dependency."""

    try:
        from scipy.special import ndtr

        return np.asarray(ndtr(values), dtype=float)
    except ImportError:
        vectorized = np.vectorize(
            lambda value: 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))),
            otypes=[float],
        )
        return vectorized(values)


def _validate_parameters(parameters: list[dict[str, Any]]) -> None:
    if not parameters:
        raise ValueError("uncertainty study requires at least one parameter")
    paths = [str(parameter.get("path", "")) for parameter in parameters]
    if any(not path for path in paths) or len(paths) != len(set(paths)):
        raise ValueError("uncertainty parameter paths must be non-empty and unique")
    for parameter in parameters:
        distribution = str(parameter.get("distribution", "normal"))
        if distribution not in {"normal", "lognormal", "uniform", "triangular"}:
            raise ValueError(
                "uncertainty distributions must be normal, lognormal, uniform, or triangular"
            )
        if distribution in {"normal", "lognormal"}:
            if not math.isfinite(float(parameter.get("mean", 0.0))):
                raise ValueError("normal/lognormal parameter mean must be finite")
            if not math.isfinite(float(parameter.get("standard_deviation", 0.0))) or float(
                parameter.get("standard_deviation", 0.0)
            ) <= 0.0:
                raise ValueError(
                    "normal/lognormal parameter standard_deviation must be positive"
                )
            if distribution == "lognormal" and float(parameter["mean"]) <= 0.0:
                raise ValueError("lognormal physical-space mean must be positive")
        else:
            lower = float(parameter.get("lower", math.nan))
            upper = float(parameter.get("upper", math.nan))
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError("uniform/triangular parameter bounds must be finite and ordered")
            if distribution == "triangular":
                mode = float(parameter.get("mode", 0.5 * (lower + upper)))
                if not lower <= mode <= upper:
                    raise ValueError("triangular parameter mode must lie inside its bounds")
        lower_clip = parameter.get("clip_lower")
        upper_clip = parameter.get("clip_upper")
        for label, value in (
            ("clip_lower", lower_clip),
            ("clip_upper", upper_clip),
        ):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"uncertainty parameter {label} must be finite when supplied"
                )
        if (
            lower_clip is not None
            and upper_clip is not None
            and float(lower_clip) > float(upper_clip)
        ):
            raise ValueError(
                "uncertainty parameter clip_lower cannot exceed clip_upper"
            )


def _correlation_factor(
    study: dict[str, Any], parameters: list[dict[str, Any]]
) -> np.ndarray:
    parameter_count = len(parameters)
    if "correlation_matrix" in study and "covariance_matrix" in study:
        raise ValueError(
            "specify either correlation_matrix or covariance_matrix, not both"
        )
    covariance_supplied = study.get("covariance_matrix")
    if covariance_supplied is not None:
        if any(
            str(parameter.get("distribution", "normal")) != "normal"
            for parameter in parameters
        ):
            raise ValueError(
                "physical covariance_matrix currently requires normal parameters; "
                "use correlation_matrix for a non-Gaussian copula"
            )
        covariance = np.asarray(covariance_supplied, dtype=float)
        if covariance.shape != (parameter_count, parameter_count):
            raise ValueError(
                "covariance_matrix must be square with one row per parameter"
            )
        standard_deviations = np.asarray(
            [float(parameter["standard_deviation"]) for parameter in parameters]
        )
        expected_diagonal = standard_deviations**2
        if not np.allclose(
            np.diag(covariance), expected_diagonal, rtol=1e-8, atol=0.0
        ):
            raise ValueError(
                "covariance_matrix diagonal must match each parameter standard_deviation squared"
            )
        correlation = covariance / np.outer(
            standard_deviations, standard_deviations
        )
    else:
        correlation = np.asarray(
            study.get("correlation_matrix", np.eye(parameter_count)), dtype=float
        )
    if correlation.shape != (parameter_count, parameter_count):
        raise ValueError("correlation_matrix must be square with one row per parameter")
    if not np.all(np.isfinite(correlation)) or not np.allclose(
        correlation, correlation.T, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("correlation_matrix must be finite and symmetric")
    if not np.allclose(np.diag(correlation), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("correlation_matrix diagonal entries must equal one")
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    if float(np.min(eigenvalues)) < -1e-10:
        raise ValueError("correlation_matrix must be positive semidefinite")
    # The eigen-factor supports exactly singular correlations, unlike Cholesky.
    return eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))


def _serializable_empirical_correlation(values: np.ndarray) -> list[list[float | None]]:
    """Return a finite JSON-safe Pearson matrix, or ``None`` when undefined."""

    samples = np.asarray(values, dtype=float)
    parameter_count = samples.shape[1]
    if samples.shape[0] < 2:
        return [[None for _ in range(parameter_count)] for _ in range(parameter_count)]
    standard_deviation = np.std(samples, axis=0)
    valid = standard_deviation > np.finfo(float).tiny
    result: list[list[float | None]] = []
    correlation = np.corrcoef(samples, rowvar=False)
    correlation = np.atleast_2d(correlation)
    for row in range(parameter_count):
        result.append(
            [
                (
                    float(correlation[row, column])
                    if valid[row]
                    and valid[column]
                    and np.isfinite(correlation[row, column])
                    else None
                )
                for column in range(parameter_count)
            ]
        )
    return result


def _draw_parameter_samples_with_diagnostics(
    study: dict[str, Any],
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """Draw samples and retain clipping/correlation distortion diagnostics."""

    parameters = list(study.get("parameters", []))
    _validate_parameters(parameters)
    sample_count = require_exact_integer(
        study.get("sample_count", 100),
        name="uncertainty.sample_count",
        minimum=1,
    )
    random_seed = require_exact_integer(
        study.get("random_seed", 0),
        name="uncertainty.random_seed",
        minimum=0,
    )
    rng = np.random.default_rng(random_seed)
    independent = rng.standard_normal((sample_count, len(parameters)))
    factor = _correlation_factor(study, parameters)
    latent = np.einsum("si,ji->sj", independent, factor, optimize=True)
    uniform = np.clip(_normal_cdf(latent), 0.0, 1.0)
    samples = np.zeros_like(latent)
    unclipped_samples = np.zeros_like(latent)
    clipping: list[dict[str, Any]] = []
    for index, parameter in enumerate(parameters):
        distribution = str(parameter.get("distribution", "normal"))
        mean = float(parameter.get("mean", 0.0))
        standard_deviation = float(parameter.get("standard_deviation", 1.0))
        if distribution == "normal":
            samples[:, index] = mean + standard_deviation * latent[:, index]
        elif distribution == "lognormal":
            variance = standard_deviation**2
            sigma_log = math.sqrt(math.log1p(variance / mean**2))
            mu_log = math.log(mean) - 0.5 * sigma_log**2
            samples[:, index] = np.exp(mu_log + sigma_log * latent[:, index])
        elif distribution == "uniform":
            lower = float(parameter["lower"])
            upper = float(parameter["upper"])
            samples[:, index] = lower + (upper - lower) * uniform[:, index]
        else:
            lower = float(parameter["lower"])
            upper = float(parameter["upper"])
            mode = float(parameter.get("mode", 0.5 * (lower + upper)))
            fraction = (mode - lower) / (upper - lower)
            probability = uniform[:, index]
            samples[:, index] = np.where(
                probability < fraction,
                lower + np.sqrt(probability * (upper - lower) * (mode - lower)),
                upper
                - np.sqrt((1.0 - probability) * (upper - lower) * (upper - mode)),
            )
        unclipped_samples[:, index] = samples[:, index]
        lower_clip = parameter.get("clip_lower")
        upper_clip = parameter.get("clip_upper")
        lower_count = (
            0
            if lower_clip is None
            else int(np.count_nonzero(samples[:, index] < float(lower_clip)))
        )
        upper_count = (
            0
            if upper_clip is None
            else int(np.count_nonzero(samples[:, index] > float(upper_clip)))
        )
        if lower_clip is not None or upper_clip is not None:
            samples[:, index] = np.clip(
                samples[:, index],
                -np.inf if lower_clip is None else float(lower_clip),
                np.inf if upper_clip is None else float(upper_clip),
            )
        clipped_count = lower_count + upper_count
        clipping.append(
            {
                "path": str(parameter["path"]),
                "clip_lower": None if lower_clip is None else float(lower_clip),
                "clip_upper": None if upper_clip is None else float(upper_clip),
                "lower_clipped_count": lower_count,
                "upper_clipped_count": upper_count,
                "clipped_count": clipped_count,
                "clipped_fraction": clipped_count / sample_count,
                "unclipped_mean": float(np.mean(unclipped_samples[:, index])),
                "unclipped_standard_deviation": float(
                    np.std(unclipped_samples[:, index], ddof=1)
                    if sample_count > 1
                    else 0.0
                ),
                "final_mean": float(np.mean(samples[:, index])),
                "final_standard_deviation": float(
                    np.std(samples[:, index], ddof=1)
                    if sample_count > 1
                    else 0.0
                ),
            }
        )
    paths = [str(parameter["path"]) for parameter in parameters]
    diagnostics = {
        "parameter_clipping": clipping,
        "any_clipping": any(item["clipped_count"] > 0 for item in clipping),
        "empirical_latent_correlation_matrix": (
            _serializable_empirical_correlation(latent)
        ),
        "empirical_unclipped_physical_correlation_matrix": (
            _serializable_empirical_correlation(unclipped_samples)
        ),
        "empirical_final_physical_correlation_matrix": (
            _serializable_empirical_correlation(samples)
        ),
        "correlation_scope": (
            "Finite-sample Pearson estimates. Non-Gaussian transforms change "
            "physical-space correlation, and clipping can additionally change "
            "marginals and correlations; undefined entries are null."
        ),
    }
    return paths, samples, diagnostics


def draw_parameter_samples(study: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    """Draw correlated physical samples; study reports retain diagnostics."""

    paths, samples, _ = _draw_parameter_samples_with_diagnostics(study)
    return paths, samples


def _simulator_factory_default(config: dict[str, Any]) -> Any:
    from .solver import AmplifierSimulator

    return AmplifierSimulator(config)


def _summarize_metric(values: list[float], quantiles: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "standard_deviation": None, "quantiles": {}}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "standard_deviation": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "quantiles": {
            f"{probability:.6g}": float(np.quantile(finite, probability))
            for probability in quantiles
        },
    }


def _evaluate_uncertainty_sample(
    config: dict[str, Any],
    sample_index: int,
    assigned: dict[str, float],
    metrics_requested: list[str],
    simulator_factory: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    """Run one sample and return only compact scalar/report diagnostics."""

    record: dict[str, Any] = {
        "sample_index": int(sample_index),
        "parameters": dict(assigned),
        "accepted": False,
        "error": None,
    }
    try:
        result = simulator_factory(config).run()
        validation = result.summary.get("validation", {})
        signal = result.summary.get("signal", {})
        requested_model = str(
            signal.get("requested_model", signal.get("model", "unknown"))
        )
        executed_model = str(
            signal.get("executed_model", signal.get("model", "unknown"))
        )
        record["requested_signal_model"] = requested_model
        record["executed_signal_model"] = executed_model
        record["signal_fallback_active"] = bool(
            signal.get("fallback_active", executed_model != requested_model)
        )
        record["validation"] = validation
        record["accepted"] = bool(validation.get("valid", True))
        metrics = extract_standard_metrics(result)
        metric_validity = standard_metric_validity(result, metrics)
        selected = (
            metrics
            if not metrics_requested
            else {key: metrics[key] for key in metrics_requested}
        )
        selected = {key: float(value) for key, value in selected.items()}
        record["metrics"] = selected
        record["metric_validity"] = {
            key: dict(metric_validity[key]) for key in selected
        }
    except Exception as exc:  # Per-sample isolation is the public contract.
        record["accepted"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


_worker_base_config: dict[str, Any] | None = None
_worker_metrics_requested: list[str] = []
_worker_fft_workers = 1


def _initialize_uncertainty_worker(
    base_config: dict[str, Any],
    metrics_requested: list[str],
    fft_workers: int,
    process_workers: int,
) -> None:
    """Install immutable study inputs once in each spawned worker."""

    global _worker_base_config, _worker_metrics_requested, _worker_fft_workers
    _worker_base_config = copy.deepcopy(base_config)
    _worker_metrics_requested = list(metrics_requested)
    _worker_fft_workers = int(fft_workers)
    transient_gain = (
        _worker_base_config.setdefault("seed", {})
        .setdefault("coherent", {})
        .setdefault("transient_gain", {})
    )
    configured_kernel_workers = _coherent_kernel_workers_for_processes(
        _worker_base_config, process_workers
    )
    if configured_kernel_workers > 0:
        transient_gain["single_transition_kernel_workers"] = (
            configured_kernel_workers
        )


def _run_default_uncertainty_sample(
    payload: tuple[int, dict[str, float]],
) -> dict[str, Any]:
    """Spawn-safe worker that never transfers a full SimulationResult."""

    sample_index, assigned = payload
    if _worker_base_config is None:
        raise RuntimeError("uncertainty worker was not initialized")
    config = copy.deepcopy(_worker_base_config)
    for path, value in assigned.items():
        set_config_value(config, path, value)
    performance = config.setdefault("performance", {})
    performance["fft_workers"] = _worker_fft_workers
    # Per-worker timing and process RSS are not meaningfully mergeable into a
    # parent-run operator profile. Benchmarks should profile representative
    # points separately.
    performance["operator_profiling"] = False
    return _evaluate_uncertainty_sample(
        config,
        sample_index,
        assigned,
        _worker_metrics_requested,
        _simulator_factory_default,
    )


def run_uncertainty_study(
    base_config: dict[str, Any],
    study: dict[str, Any],
    *,
    simulator_factory: Callable[[dict[str, Any]], Any] = _simulator_factory_default,
) -> dict[str, Any]:
    """Run a correlated Monte-Carlo study with invalid-point isolation."""

    method = str(study.get("method", "monte_carlo"))
    if method != "monte_carlo":
        raise ValueError("uncertainty method currently must be monte_carlo")
    paths, samples, sampling_diagnostics = (
        _draw_parameter_samples_with_diagnostics(study)
    )
    worker_count = require_exact_integer(
        study.get("workers", 1),
        name="uncertainty.workers",
        minimum=1,
    )
    fft_workers_per_process = require_exact_integer(
        study.get("fft_workers_per_process", 1),
        name="uncertainty.fft_workers_per_process",
        minimum=1,
    )
    if worker_count > 1 and simulator_factory is not _simulator_factory_default:
        raise ValueError(
            "uncertainty.workers > 1 requires the default simulator factory; "
            "custom factories may not be process-safe or pickleable"
        )
    worker_count = min(worker_count, max(len(samples), 1))
    raw_metrics = study.get("metrics", [])
    if not isinstance(raw_metrics, (list, tuple)):
        raise ValueError("uncertainty.metrics must be a list")
    metrics_requested = [str(value) for value in raw_metrics]
    if any(not metric for metric in metrics_requested):
        raise ValueError("uncertainty.metrics must contain non-empty names")
    unknown_metrics = sorted(set(metrics_requested) - set(SWEEP_METRICS))
    if unknown_metrics and simulator_factory is _simulator_factory_default:
        raise ValueError(
            f"unknown uncertainty metrics: {unknown_metrics}; "
            f"choose from {sorted(SWEEP_METRICS)}"
        )
    if len(metrics_requested) != len(set(metrics_requested)):
        raise ValueError("uncertainty.metrics must not contain duplicates")
    quantiles = [float(value) for value in study.get("quantiles", [0.025, 0.5, 0.975])]
    if any(not 0.0 <= value <= 1.0 for value in quantiles):
        raise ValueError("uncertainty quantiles must lie in [0, 1]")
    sample_inputs: list[tuple[int, dict[str, float]]] = []
    for sample_index, row in enumerate(samples):
        assigned = {
            path: float(value) for path, value in zip(paths, row, strict=True)
        }
        sample_inputs.append((sample_index, assigned))
    if worker_count == 1:
        records = []
        for sample_index, assigned in sample_inputs:
            config = copy.deepcopy(base_config)
            for path, value in assigned.items():
                set_config_value(config, path, value)
            performance = config.setdefault("performance", {})
            performance["fft_workers"] = fft_workers_per_process
            performance["operator_profiling"] = False
            records.append(
                _evaluate_uncertainty_sample(
                    config,
                    sample_index,
                    assigned,
                    metrics_requested,
                    simulator_factory,
                )
            )
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_uncertainty_worker,
            initargs=(
                base_config,
                metrics_requested,
                fft_workers_per_process,
                worker_count,
            ),
        ) as executor:
            # map preserves deterministic sample ordering even when workers
            # complete out of order.
            records = list(
                executor.map(_run_default_uncertainty_sample, sample_inputs)
            )
    accepted_metrics: dict[str, list[float]] = {}
    accepted_metrics_by_model: dict[str, dict[str, list[float]]] = {}
    for record in records:
        if not bool(record["accepted"]):
            continue
        executed_model = str(record.get("executed_signal_model", "unknown"))
        for key, value in record.get("metrics", {}).items():
            if bool(record.get("metric_validity", {}).get(key, {}).get("valid", False)):
                accepted_metrics.setdefault(key, []).append(float(value))
                accepted_metrics_by_model.setdefault(executed_model, {}).setdefault(
                    key, []
                ).append(float(value))
    accepted = sum(bool(record["accepted"]) for record in records)
    executed_models = sorted(accepted_metrics_by_model)
    model_homogeneous = len(executed_models) <= 1
    return {
        "method": method,
        "random_seed": require_exact_integer(
            study.get("random_seed", 0),
            name="uncertainty.random_seed",
            minimum=0,
        ),
        "sample_count": len(records),
        "execution": {
            "process_workers": worker_count,
            "fft_workers_per_process": (
                fft_workers_per_process
            ),
            "coherent_kernel_workers_per_process": (
                _coherent_kernel_workers_for_processes(
                    base_config, worker_count
                )
            ),
            "isolation": (
                "spawned_process_per_worker"
                if worker_count > 1
                else "sequential_current_process"
            ),
            "worker_result_payload": "compact_metrics_validation_only",
            "base_config_transfer": (
                "initialized_once_per_worker"
                if worker_count > 1
                else "copied_per_sample"
            ),
        },
        "accepted_sample_count": accepted,
        "invalid_or_failed_sample_count": len(records) - accepted,
        "acceptance_fraction": accepted / max(len(records), 1),
        "parameter_paths": paths,
        "sampling_diagnostics": sampling_diagnostics,
        "configured_correlation_matrix": (
            None
            if "covariance_matrix" in study
            else study.get("correlation_matrix", np.eye(len(paths)).tolist())
        ),
        "configured_covariance_matrix": study.get("covariance_matrix"),
        "metric_statistics": (
            {
                key: _summarize_metric(values, quantiles)
                for key, values in accepted_metrics.items()
            }
            if model_homogeneous
            else {}
        ),
        "metric_statistics_by_executed_signal_model": {
            model: {
                key: _summarize_metric(values, quantiles)
                for key, values in per_metric.items()
            }
            for model, per_metric in accepted_metrics_by_model.items()
        },
        "executed_signal_models": executed_models,
        "model_homogeneous_statistics": model_homogeneous,
        "records": records,
        "validity_contract": (
            "Only runs whose shared validation report is valid contribute to "
            "the accepted-sample count. Each confidence interval independently "
            "uses only samples whose requested metric is valid; an unavailable "
            "diagnostic such as sampled M² does not invalidate power metrics or "
            "the underlying run. Statistics from different executed Signal "
            "backends are never merged into metric_statistics; use the "
            "per-model table when a fallback changed backend. Every rejected "
            "sample remains in records."
        ),
    }


def _calibration_residuals(
    vector: np.ndarray,
    base_config: dict[str, Any],
    parameters: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    simulator_factory: Callable[[dict[str, Any]], Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    config = copy.deepcopy(base_config)
    for parameter, value in zip(parameters, vector, strict=True):
        set_config_value(config, str(parameter["path"]), float(value))
    result = simulator_factory(config).run()
    validation = result.summary.get("validation", {})
    if not bool(validation.get("valid", True)):
        raise ValueError("calibration point failed the shared physical validity gate")
    metrics = extract_standard_metrics(result)
    metric_validity = standard_metric_validity(result, metrics)
    requested_metrics = [str(observation["metric"]) for observation in observations]
    nonfinite_metrics = sorted(
        metric
        for metric in requested_metrics
        if (
            metric not in metrics
            or metric not in metric_validity
            or not bool(metric_validity[metric]["valid"])
        )
    )
    if nonfinite_metrics:
        raise ValueError(
            f"calibration metrics are missing or non-finite: {nonfinite_metrics}"
        )
    residuals: list[float] = []
    for observation in observations:
        metric = str(observation["metric"])
        measurement_sigma = float(observation["standard_deviation"])
        model_sigma = float(observation.get("model_error_standard_deviation", 0.0))
        sigma = math.sqrt(measurement_sigma**2 + model_sigma**2)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("calibration observation uncertainty must be positive")
        residuals.append((float(metrics[metric]) - float(observation["value"])) / sigma)
    for parameter, value in zip(parameters, vector, strict=True):
        if parameter.get("prior_mean") is not None:
            prior_sigma = float(parameter.get("prior_standard_deviation", 0.0))
            if prior_sigma <= 0.0 or not math.isfinite(prior_sigma):
                raise ValueError("calibration prior standard deviation must be positive")
            residuals.append((float(value) - float(parameter["prior_mean"])) / prior_sigma)
    return np.asarray(residuals, dtype=float), {
        "metrics": metrics,
        "validation": validation,
        "executed_signal_model": str((result.summary.get("signal") or {}).get(
            "executed_model", (result.summary.get("signal") or {}).get("model", "unknown")
        )),
    }


def calibrate_to_experiment(
    base_config: dict[str, Any],
    study: dict[str, Any],
    *,
    simulator_factory: Callable[[dict[str, Any]], Any] = _simulator_factory_default,
) -> dict[str, Any]:
    """Weighted nonlinear least-squares calibration with optional priors."""

    parameters = list(study.get("parameters", []))
    observations = list(study.get("observations", []))
    if not parameters or not observations:
        raise ValueError("calibration requires parameters and observations")
    maximum_evaluations = require_exact_integer(
        study.get("maximum_evaluations", 100),
        name="calibration.maximum_evaluations",
        minimum=1,
    )
    residual_variance_mode = str(
        study.get("residual_variance_mode", "known_absolute")
    )
    if residual_variance_mode not in {
        "known_absolute",
        "estimated_from_residuals",
    }:
        raise ValueError(
            "calibration residual_variance_mode must be known_absolute or estimated_from_residuals"
        )
    paths = [str(parameter.get("path", "")) for parameter in parameters]
    if any(not path for path in paths) or len(paths) != len(set(paths)):
        raise ValueError("calibration parameter paths must be non-empty and unique")
    try:
        initial = np.asarray(
            [
                float(parameter.get("initial", get_config_value(base_config, path)))
                for parameter, path in zip(parameters, paths, strict=True)
            ],
            dtype=float,
        )
        lower = np.asarray(
            [float(parameter["lower"]) for parameter in parameters], dtype=float
        )
        upper = np.asarray(
            [float(parameter["upper"]) for parameter in parameters], dtype=float
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "calibration parameters need numeric initial values and finite bounds"
        ) from exc
    if (
        np.any(~np.isfinite(initial))
        or np.any(~np.isfinite(lower))
        or np.any(~np.isfinite(upper))
        or np.any(lower >= upper)
        or np.any((initial < lower) | (initial > upper))
    ):
        raise ValueError("calibration initial values and finite bounds are inconsistent")
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise ValueError(f"calibration observation {index} must be an object")
        metric = str(observation.get("metric", ""))
        if not metric:
            raise ValueError(
                f"calibration observation {index} needs a non-empty metric name"
            )
        if (
            metric not in SWEEP_METRICS
            and simulator_factory is _simulator_factory_default
        ):
            raise ValueError(
                f"calibration observation {index} has unknown metric {metric!r}"
            )
        try:
            observed_value = float(observation["value"])
            measurement_sigma = float(observation["standard_deviation"])
            model_sigma = float(
                observation.get("model_error_standard_deviation", 0.0)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"calibration observation {index} needs numeric value and standard_deviation"
            ) from exc
        if not math.isfinite(observed_value):
            raise ValueError(
                f"calibration observation {index} value must be finite"
            )
        if not math.isfinite(measurement_sigma) or measurement_sigma <= 0.0:
            raise ValueError(
                f"calibration observation {index} standard_deviation must be positive"
            )
        if not math.isfinite(model_sigma) or model_sigma < 0.0:
            raise ValueError(
                f"calibration observation {index} model-error standard deviation "
                "must be finite and non-negative"
            )
    for index, parameter in enumerate(parameters):
        prior_mean = parameter.get("prior_mean")
        if prior_mean is None:
            continue
        prior_sigma = parameter.get("prior_standard_deviation")
        try:
            valid_prior = math.isfinite(float(prior_mean)) and math.isfinite(
                float(prior_sigma)
            ) and float(prior_sigma) > 0.0
        except (TypeError, ValueError):
            valid_prior = False
        if not valid_prior:
            raise ValueError(
                f"calibration parameter {index} prior mean must be finite and "
                "prior_standard_deviation must be positive"
            )
    evaluations: list[dict[str, Any]] = []
    executed_model: str | None = None

    class CalibrationModelChange(ValueError):
        pass

    def require_same_model(diagnostics: dict[str, Any]) -> None:
        nonlocal executed_model
        current = diagnostics["executed_signal_model"]
        if executed_model is None:
            executed_model = current
        elif current != executed_model:
            raise CalibrationModelChange(
                f"calibration executed Signal model changed from {executed_model!r} to {current!r}; "
                "use one consistent backend across the parameter bounds"
            )

    def objective(vector: np.ndarray) -> np.ndarray:
        try:
            residuals, diagnostics = _calibration_residuals(
                vector, base_config, parameters, observations, simulator_factory
            )
            require_same_model(diagnostics)
            evaluations.append(
                {
                    "parameters": {
                        path: float(value)
                        for path, value in zip(paths, vector, strict=True)
                    },
                    "objective": float(np.sum(residuals**2)),
                    "metrics": diagnostics["metrics"],
                    "valid": True,
                }
            )
            return residuals
        except CalibrationModelChange:
            raise
        except Exception as exc:
            evaluations.append(
                {
                    "parameters": {
                        path: float(value)
                        for path, value in zip(paths, vector, strict=True)
                    },
                    "objective": None,
                    "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            # Keep invalid physics out of the optimum while retaining a stable
            # residual dimension for scipy.optimize.
            return np.full(
                len(observations)
                + sum(parameter.get("prior_mean") is not None for parameter in parameters),
                1e6,
            )

    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise RuntimeError("calibration requires the scipy optional dependency") from exc
    span = upper - lower
    initial_unit = (initial - lower) / span

    tolerances = {
        "parameter_tolerance": float(study.get("parameter_tolerance", 1e-8)),
        "objective_tolerance": float(study.get("objective_tolerance", 1e-8)),
        "gradient_tolerance": float(study.get("gradient_tolerance", 1e-8)),
    }
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in tolerances.values()
    ):
        raise ValueError("calibration optimizer tolerances must be finite and positive")

    def normalized_objective(unit_vector: np.ndarray) -> np.ndarray:
        return objective(lower + np.asarray(unit_vector, dtype=float) * span)

    fit = least_squares(
        normalized_objective,
        initial_unit,
        bounds=(np.zeros_like(initial_unit), np.ones_like(initial_unit)),
        x_scale="jac",
        xtol=tolerances["parameter_tolerance"],
        ftol=tolerances["objective_tolerance"],
        gtol=tolerances["gradient_tolerance"],
        max_nfev=maximum_evaluations,
    )
    best = lower + fit.x * span
    final_residuals, diagnostics = _calibration_residuals(
        best, base_config, parameters, observations, simulator_factory
    )
    require_same_model(diagnostics)
    covariance = None
    covariance_scale = None
    covariance_status = "rank_deficient"
    jacobian_rank = 0
    data_jacobian_rank = 0
    residual_degrees = None
    if fit.jac.size:
        # Squaring J worsens conditioning, and pinv(J.T J) gives zero
        # variance in unconstrained directions. Those directions are unknown,
        # not known exactly: report unavailable covariance instead.
        _, singular, right = np.linalg.svd(fit.jac, full_matrices=False)
        tolerance = np.finfo(float).eps * max(fit.jac.shape) * singular[0]
        jacobian_rank = int(np.count_nonzero(singular > tolerance))
        if jacobian_rank == len(parameters):
            data_jacobian = fit.jac[:len(observations)]
            _, data_singular, data_right = np.linalg.svd(
                data_jacobian,
                full_matrices=data_jacobian.shape[0] < data_jacobian.shape[1],
            )
            data_tolerance = (
                np.finfo(float).eps * max(data_jacobian.shape)
                * (data_singular[0] if data_singular.size else 0.0)
            )
            data_jacobian_rank = int(np.count_nonzero(data_singular > data_tolerance))
            residual_degrees = len(observations) - data_jacobian_rank
            if residual_variance_mode == "estimated_from_residuals" and residual_degrees <= 0:
                covariance_status = "insufficient_residual_degrees_of_freedom"
            else:
                covariance_scale = (
                    float(np.sum(final_residuals[:len(observations)]**2)) / residual_degrees
                    if residual_variance_mode == "estimated_from_residuals"
                    else 1.0
                )
                prior_jacobian = fit.jac[len(observations):]
                if covariance_scale == 0.0:
                    # The zero-noise limit fixes data-supported directions;
                    # independent directions retain their fixed prior variance.
                    null_space = data_right[data_jacobian_rank:].T
                    if null_space.shape[1]:
                        _, prior_singular, prior_right = np.linalg.svd(
                            prior_jacobian @ null_space, full_matrices=False
                        )
                        null_covariance = (prior_right.T / prior_singular**2) @ prior_right
                        covariance_unit = null_space @ null_covariance @ null_space.T
                    else:
                        covariance_unit = np.zeros((len(parameters), len(parameters)))
                elif covariance_scale == 1.0 or not len(prior_jacobian):
                    covariance_unit = (right.T / singular**2) @ right * covariance_scale
                else:
                    # Only observation uncertainty is estimated. A declared
                    # prior is an absolute precision and is never rescaled.
                    posterior_jacobian = np.vstack((
                        data_jacobian / math.sqrt(covariance_scale), prior_jacobian
                    ))
                    _, posterior_singular, posterior_right = np.linalg.svd(
                        posterior_jacobian, full_matrices=False
                    )
                    covariance_unit = (posterior_right.T / posterior_singular**2) @ posterior_right
                covariance = span[:, None] * covariance_unit * span[None, :]
                covariance_status = "local_linear_approximation"
    return {
        "method": "weighted_nonlinear_least_squares",
        "executed_signal_model": executed_model,
        "success": bool(fit.success),
        "message": str(fit.message),
        "parameter_paths": paths,
        "best_parameters": {
            path: float(value) for path, value in zip(paths, best, strict=True)
        },
        "parameter_covariance": None if covariance is None else covariance.tolist(),
        "parameter_covariance_status": covariance_status,
        "parameter_jacobian_rank": jacobian_rank,
        "parameter_data_jacobian_rank": data_jacobian_rank,
        "parameter_residual_degrees_of_freedom": residual_degrees,
        "parameter_covariance_variance_mode": residual_variance_mode,
        "parameter_covariance_scale": covariance_scale,
        "standardized_residuals": final_residuals.tolist(),
        "chi_square": float(np.sum(final_residuals**2)),
        "best_metrics": diagnostics["metrics"],
        "function_evaluations": int(fit.nfev),
        "evaluations": evaluations,
        "observation_contract": (
            "Measurement and explicit model-error variances are added before weighting; "
            "invalid simulator points receive no likelihood support. Parameter covariance "
            + (
                "uses those known absolute variances without residual rescaling."
                if residual_variance_mode == "known_absolute"
                else "estimates a data-only residual variance scale while keeping prior precision fixed."
            )
        ),
    }
