"""Command-line interface."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .beams import BEAM_REGISTRY
from .config import PHYSICS_PRESETS, load_config, validate_config
from .convergence_suite import (
    CONVERGENCE_AXES,
    DEFAULT_CONVERGENCE_AXES,
    DEFAULT_CONVERGENCE_METRICS,
    run_convergence_suite,
)
from .josa_modes import JOSA_MODES
from .json_contract import dumps_json_portable, loads_json_strict
from .material_library import apply_material_record, load_material_library
from .published_benchmarks import available_benchmarks, run_published_benchmark
from .solver import AmplifierSimulator
from .stress import STRESS_REGISTRY
from .strict_validation import NON_QUANTITATIVE_BANNER
from .uncertainty import calibrate_to_experiment, run_uncertainty_study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amplifier-sim", description="3D solid-state amplifier simulation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a JSON configuration")
    run.add_argument("config", type=Path)
    run.add_argument("-o", "--output", type=Path, default=Path("output/latest"))
    validate = subparsers.add_parser("validate", help="validate and print the merged configuration")
    validate.add_argument("config", type=Path)
    convergence = subparsers.add_parser(
        "convergence", help="run one-axis-at-a-time numerical refinement studies"
    )
    convergence.add_argument("config", type=Path)
    convergence.add_argument("-o", "--output", type=Path, default=Path("convergence.json"))
    convergence.add_argument(
        "--axes",
        nargs="+",
        choices=sorted(CONVERGENCE_AXES),
        default=list(DEFAULT_CONVERGENCE_AXES),
    )
    convergence.add_argument("--factors", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    convergence.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_CONVERGENCE_METRICS),
    )
    convergence.add_argument("--relative-tolerance", type=float, default=0.01)
    uncertainty = subparsers.add_parser(
        "uncertainty", help="propagate correlated parameter uncertainty by Monte Carlo"
    )
    uncertainty.add_argument("config", type=Path)
    uncertainty.add_argument("study", type=Path)
    uncertainty.add_argument("-o", "--output", type=Path, default=Path("uncertainty.json"))
    calibration = subparsers.add_parser(
        "calibrate", help="fit configured parameters to weighted experimental observations"
    )
    calibration.add_argument("config", type=Path)
    calibration.add_argument("study", type=Path)
    calibration.add_argument("-o", "--output", type=Path, default=Path("calibration.json"))
    benchmark = subparsers.add_parser("benchmark", help="run a source-traceable published benchmark")
    benchmark.add_argument("name", nargs="?", choices=available_benchmarks())
    subparsers.add_parser("models", help="list built-in model and plugin names")
    materialize = subparsers.add_parser(
        "materialize",
        help="embed one material-library record and its evaluated parameters into a setup JSON",
    )
    materialize.add_argument("config", type=Path)
    materialize.add_argument("library", type=Path)
    materialize.add_argument("--material", required=True)
    materialize.add_argument("--doping-percent", type=float, required=True)
    materialize.add_argument("--temperature-K", type=float, required=True)
    materialize.add_argument(
        "--polarization",
        choices=("x", "y", "z"),
        default="x",
        help="Signal optical-polarization axis",
    )
    materialize.add_argument(
        "--pump-polarization",
        choices=("x", "y", "z"),
        default="x",
        help="Pump spectroscopy and thermo-optic axis",
    )
    materialize.add_argument(
        "--thermomechanical-axis",
        choices=("x", "y", "z"),
        default="x",
        help="crystal axis used by scalar thermomechanical proxies",
    )
    materialize.add_argument(
        "--snapshot-out-of-range",
        choices=("error", "edge"),
        default="error",
        help=(
            "material snapshot range policy; error is quantitative default, "
            "edge is an explicit reduced-order endpoint extrapolation"
        ),
    )
    materialize.add_argument("-o", "--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "models":
        print(dumps_json_portable({
            "energy_levels": ["two_level", "three_level", "four_level", "custom"],
            "physics_presets": sorted(PHYSICS_PRESETS),
            "signal_models": ["auto", "commercial_app_fast", "fluence", "spectral_time", "josa_transport", "coherent_cpa"],
            "nonlinear_phase_models": ["off", "kerr_parabolic_lens", "kerr_time_domain"],
            "pump_propagation_models": ["prescribed_caustic", "abcd", "super_gaussian_abcd", "hybrid_abcd_fft", "fft", "stochastic_raytrace"],
            "pump_spectral_spatial_transport": ["shared", "per_wavelength"],
            "pump_ray_angular_distributions": ["auto", "uniform_na", "uniform_solid_angle", "generalized_gaussian"],
            "pump_ray_deposition_models": ["nearest_cell_track_length", "bilinear_track_length"],
            "seed_thermal_field": "full_3d_delta_n",
            "seed_spatial_gain_models": ["local_field", "gaussian_mode_overlap", "parabolic_gain_fit"],
            "seed_spatial_mode_models": ["effective_m2", "hermite_gaussian", "incoherent_hg"],
            "ase_models": ["off", "six_ray", "angular_discrete_ordinates"],
            "josa_transport_modes": sorted(JOSA_MODES),
            "multipass_templates": ["double_pass", "two_mirror", "herriott", "relay_4f", "prescribed"],
            "temporal_profiles": ["cw", "gaussian", "sech2", "rectangular", "super_gaussian", "file"],
            "spectral_profiles": ["monochromatic", "gaussian", "lorentzian", "super_gaussian", "file"],
            "optical_elements": ["free_space", "thin_lens", "thick_lens", "aperture", "interface", "tilt", "coordinate_break", "mirror", "phase_screen", "dispersive", "loss", "observation", "gain_medium"],
            "beams": sorted(BEAM_REGISTRY),
            "stress": sorted(STRESS_REGISTRY),
        }, indent=2))
        return 0
    if args.command == "benchmark":
        if args.name is None:
            print(
                dumps_json_portable(
                    {"published_benchmarks": available_benchmarks()},
                    indent=2,
                )
            )
            return 0
        result = run_published_benchmark(args.name)
        print(dumps_json_portable(result, indent=2, ensure_ascii=False))
        return 0 if result["all_passed"] else 1
    if args.command == "materialize":
        config = load_config(args.config)
        records = load_material_library(args.library)
        matches = [record for record in records if record.get("name") == args.material]
        if len(matches) != 1:
            available = sorted(str(record.get("name")) for record in records)
            raise ValueError(
                f"material {args.material!r} matched {len(matches)} records; available: {available}"
            )
        materialized, _ = apply_material_record(
            config,
            matches[0],
            doping_percent=args.doping_percent,
            temperature_K=args.temperature_K,
            polarization=args.polarization,
            pump_polarization=args.pump_polarization,
            thermomechanical_axis=args.thermomechanical_axis,
            snapshot_out_of_range=args.snapshot_out_of_range,
        )
        materialized["setup_provenance"] = {
            "base_config_file": str(args.config),
            "base_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "material_library_file": str(args.library),
            "material_library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
            "material_name": args.material,
            "material_record_sha256": materialized["material"]["library"].get(
                "record_sha256"
            ),
            "doping_percent": args.doping_percent,
            "temperature_K": args.temperature_K,
            "polarization": args.polarization,
            "pump_polarization": args.pump_polarization,
            "thermomechanical_axis": args.thermomechanical_axis,
            "snapshot_out_of_range": args.snapshot_out_of_range,
            "generator": "amplifier-sim materialize",
        }
        # Materialization preserves the user's selected heat model and must
        # satisfy the same public schema/physics contracts as a hand-written
        # setup before an output file is created.
        validate_config(materialized)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            dumps_json_portable(materialized, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            dumps_json_portable(
                materialized["setup_provenance"],
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"\nSaved: {args.output.resolve()}")
        return 0
    config = load_config(args.config)
    if args.command == "validate":
        print(dumps_json_portable(config, indent=2, ensure_ascii=False))
        return 0
    if args.command == "convergence":
        report = run_convergence_suite(
            config,
            axes=args.axes,
            factors=args.factors,
            metric_keys=args.metrics,
            relative_tolerance=args.relative_tolerance,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            dumps_json_portable(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(dumps_json_portable(report, indent=2, ensure_ascii=False))
        print(f"\nSaved: {args.output.resolve()}")
        return 0 if report["all_assessed_axes_converged"] else 3
    if args.command in {"uncertainty", "calibrate"}:
        study = loads_json_strict(args.study.read_text(encoding="utf-8"))
        report = (
            run_uncertainty_study(config, study)
            if args.command == "uncertainty"
            else calibrate_to_experiment(config, study)
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            dumps_json_portable(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(dumps_json_portable(report, indent=2, ensure_ascii=False))
        print(f"\nSaved: {args.output.resolve()}")
        if args.command == "uncertainty":
            return 0 if report["accepted_sample_count"] > 0 else 4
        return 0 if report["success"] else 5
    result = AmplifierSimulator(config).run()
    print(dumps_json_portable(result.summary, indent=2, ensure_ascii=False))
    validation = result.summary.get("validation", {})
    if not validation.get("valid", True):
        print(f"\n{NON_QUANTITATIVE_BANNER}")
    if validation.get("enforced", False) and not validation.get("valid", True):
        args.output.mkdir(parents=True, exist_ok=True)
        diagnostic = args.output / "summary.invalid.json"
        diagnostic.write_text(
            dumps_json_portable(result.summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"\nSaved invalid diagnostic summary only: {diagnostic.resolve()}"
        )
        return 2
    result.save(args.output)
    print(f"\nSaved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
