"""Small, source-traceable validation cases from published amplifier papers.

These evaluators are intentionally isolated from the full 3-D solver.  A paper
benchmark should first reproduce the exact reduced equation used by its authors;
only then should it be promoted to a full pump/thermal/propagation example.
"""

from __future__ import annotations

import math
from importlib.resources import files
from typing import Any, Callable

from .json_contract import loads_json_strict

BenchmarkEvaluator = Callable[[dict[str, Any]], dict[str, float]]


def _load_json(name: str) -> dict[str, Any]:
    path = files("amplifier_sim").joinpath("data", f"{name}.json")
    if not path.is_file():
        raise KeyError(f"unknown published benchmark: {name}")
    return loads_json_strict(path.read_text(encoding="utf-8"))


def _zigzag_frantz_nodvik_eq6(case: dict[str, Any]) -> dict[str, float]:
    """Evaluate Eq. (6) of Sridharan, Saraf & Byer, Applied Optics 46.

    The paper maps the Gaussian 1/e^2 radius to a top-hat-equivalent radius,
    then corrects the active area by the zig-zag path fill factor f(2-f).
    """

    p = case["parameters"]
    waist = float(p["gaussian_1e2_radius_m"])
    thickness = float(p["crystal_thickness_m"])
    angle_rad = math.radians(float(p["first_pass_bounce_angle_deg"]))
    cos_angle = math.cos(angle_rad)
    if waist <= 0.0 or thickness <= 0.0 or cos_angle <= 0.0:
        raise ValueError("waist, slab thickness, and cos(bounce angle) must be positive")

    top_hat_radius = waist / math.sqrt(2.0)
    active_area = 2.0 * top_hat_radius * thickness
    overlap_fraction = waist / (math.sqrt(2.0) * thickness * cos_angle)
    if not 0.0 < overlap_fraction < 2.0:
        raise ValueError("the published zig-zag overlap factor must satisfy 0 < f < 2")
    path_fill = overlap_fraction * (2.0 - overlap_fraction)
    effective_saturation_area = active_area * cos_angle * path_fill

    saturation_energy = float(p["saturation_fluence_J_m2"]) * effective_saturation_area
    input_energy = float(p["input_energy_J"])
    logarithmic_gain = float(p["logarithmic_small_signal_gain"])
    transmission = float(p.get("single_pass_transmission", 1.0))
    if saturation_energy <= 0.0 or input_energy < 0.0 or transmission < 0.0:
        raise ValueError("saturation energy and transmission must be positive")

    output_energy = transmission * saturation_energy * math.log1p(
        math.exp(logarithmic_gain) * math.expm1(input_energy / saturation_energy)
    )
    return {
        "top_hat_equivalent_radius_m": top_hat_radius,
        "active_area_m2": active_area,
        "overlap_fraction_f": overlap_fraction,
        "zigzag_path_fill_f_times_2_minus_f": path_fill,
        "effective_saturation_area_m2": effective_saturation_area,
        "effective_saturation_energy_J": saturation_energy,
        "calculated_single_pass_output_energy_J": output_energy,
    }


EVALUATORS: dict[str, BenchmarkEvaluator] = {
    "zigzag_frantz_nodvik_eq6": _zigzag_frantz_nodvik_eq6,
}


def available_benchmarks() -> list[str]:
    return ["sridharan_2007_ybyag"]


def run_published_benchmark(name: str) -> dict[str, Any]:
    benchmark = _load_json(name)
    model = str(benchmark["model"])
    try:
        evaluator = EVALUATORS[model]
    except KeyError as exc:
        raise KeyError(f"unsupported published benchmark model: {model}") from exc

    results: list[dict[str, Any]] = []
    all_passed = True
    for case in benchmark["cases"]:
        calculated = evaluator(case)
        reference = float(case["published"]["single_pass_output_energy_J"])
        output = calculated["calculated_single_pass_output_energy_J"]
        relative_error = abs(output - reference) / reference
        passed = relative_error <= float(case["acceptance_relative_error"])
        all_passed = all_passed and passed
        results.append(
            {
                "case_id": case["id"],
                "claim_type": benchmark["claim_type"],
                "published_single_pass_output_energy_J": reference,
                **calculated,
                "relative_error": relative_error,
                "acceptance_relative_error": float(case["acceptance_relative_error"]),
                "passed": passed,
            }
        )

    return {
        "benchmark_id": benchmark["id"],
        "title": benchmark["title"],
        "doi": benchmark["doi"],
        "source_url": benchmark["source_url"],
        "author_pdf_url": benchmark["author_pdf_url"],
        "claim_type": benchmark["claim_type"],
        "evidence_note": benchmark["evidence_note"],
        "all_passed": all_passed,
        "cases": results,
    }
