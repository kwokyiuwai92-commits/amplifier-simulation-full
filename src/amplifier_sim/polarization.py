"""Jones-vector Seed propagation and radial/tangential thermal birefringence.

The scalar diffraction and gain operators in the package act identically on
orthogonal polarization components.  This module supplies the local operator
that does not: a reciprocal linear retarder whose eigenvectors are radial and
tangential in the crystal transverse plane.

For the axisymmetric plane-stress closure derived by Chénais et al. and used
by Veselis et al. (2021), the additional thermo-photoelastic index shifts are

    delta n_(r,theta) = 2 n0**3 alpha_T C_(r,theta) [T(r,z) - T(0,z)].

The ordinary thermo-optic ``dn/dT`` contribution is retained as a common-mode
phase.  Referencing the photoelastic term to the local axis temperature is
essential: a spatially uniform temperature rise cannot create stress
birefringence in a freely expanding crystal.  No scalar loss is invented for
depolarization; analyzer loss is only a diagnostic computed from the
propagated Jones field.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .beam_quality import sampled_beam_quality


SCALAR = "scalar"
JONES = "jones"
BIREFRINGENCE_OFF = "off"
RADIAL_TANGENTIAL = "radial_tangential_plane_stress"
TENSOR_PHOTOELASTIC = "tensor_photoelastic"


def polarization_model(seed_cfg: dict[str, Any]) -> str:
    return str(seed_cfg.get("polarization", {}).get("model", SCALAR))


def jones_enabled(seed_cfg: dict[str, Any]) -> bool:
    return polarization_model(seed_cfg) == JONES


def input_jones_vector(seed_cfg: dict[str, Any]) -> np.ndarray:
    """Return the normalized configured input Jones vector in lab x/y axes."""

    config = seed_cfg.get("polarization", {})
    state = str(config.get("input_state", "linear"))
    if state == "right_circular":
        vector = np.asarray([1.0, -1j], dtype=complex) / math.sqrt(2.0)
    elif state == "left_circular":
        vector = np.asarray([1.0, 1j], dtype=complex) / math.sqrt(2.0)
    else:
        angle = math.radians(float(config.get("linear_angle_deg", 0.0)))
        phase = 0.0 if state == "linear" else float(
            config.get("relative_phase_y_minus_x_rad", 0.0)
        )
        vector = np.asarray(
            [math.cos(angle), math.sin(angle) * np.exp(1j * phase)],
            dtype=complex,
        )
    norm = float(np.sqrt(np.sum(np.abs(vector) ** 2)))
    if norm <= np.finfo(float).tiny:
        raise ValueError("seed.polarization input Jones vector has zero norm")
    return vector / norm


def initialize_jones_field(
    scalar_field: np.ndarray,
    seed_cfg: dict[str, Any],
) -> np.ndarray:
    """Split a scalar field into normalized lab-frame Jones components."""

    field = np.asarray(scalar_field, dtype=complex)
    vector = input_jones_vector(seed_cfg)
    return vector.reshape((2,) + (1,) * field.ndim) * field[None, ...]


def total_transverse_density(field: np.ndarray, *, is_jones: bool) -> np.ndarray:
    """Return total fluence/intensity while preserving non-polarization axes."""

    values = np.abs(np.asarray(field)) ** 2
    return np.sum(values, axis=0) if is_jones else values


def _wavelength_phase_shape(
    wavelengths_m: float | np.ndarray,
    target_ndim: int,
) -> np.ndarray:
    wavelengths = np.asarray(wavelengths_m, dtype=float)
    if wavelengths.ndim == 0:
        return wavelengths
    if wavelengths.ndim != 1:
        raise ValueError("Jones wavelengths must be scalar or one-dimensional")
    # Jones fields use (polarization, wavelength, ..., y, x).  The local
    # phase has no polarization axis and therefore begins with wavelength.
    return wavelengths.reshape((len(wavelengths),) + (1,) * (target_ndim - 1))


def _sample_transverse_at(
    values: np.ndarray,
    grid: Any,
    x_m: float,
    y_m: float,
) -> float:
    """Bilinearly sample a scalar transverse map on the grid cell centres."""

    plane = np.asarray(values, dtype=float)
    if plane.shape != (len(grid.y), len(grid.x)):
        raise ValueError("transverse sample map must match the grid y/x dimensions")
    if not (
        float(grid.x_edges[0]) <= x_m <= float(grid.x_edges[-1])
        and float(grid.y_edges[0]) <= y_m <= float(grid.y_edges[-1])
    ):
        raise ValueError(
            "radial/tangential birefringence centre must lie inside the optical grid"
        )
    sampled_x = np.asarray(
        [np.interp(x_m, grid.x, row) for row in plane], dtype=float
    )
    return float(np.interp(y_m, grid.y, sampled_x))


def apply_thermal_birefringence(
    field: np.ndarray,
    delta_n_slice: np.ndarray,
    step_length_m: float,
    wavelengths_m: float | np.ndarray,
    refractive_index: float | np.ndarray,
    grid: Any,
    material: dict[str, Any],
    seed_cfg: dict[str, Any],
    photoelastic_tensor_slice: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply common thermal phase and optional radial/tangential retardance.

    ``field`` must have Jones polarization as axis zero.  Any axes between
    polarization and y/x (currently wavelength) are broadcast independently.
    The returned map is the signed local retardance in radians for diagnostics.
    """

    values = np.asarray(field, dtype=complex)
    if values.shape[0] != 2 or values.shape[-2:] != np.asarray(delta_n_slice).shape:
        raise ValueError(
            "Jones field must have shape (2, ..., ny, nx) matching delta_n_slice"
        )
    local_delta_n = np.asarray(delta_n_slice, dtype=float)
    wavelengths = _wavelength_phase_shape(wavelengths_m, values.ndim - 1)
    k_step = 2.0 * math.pi * float(step_length_m) / wavelengths
    config = seed_cfg.get("polarization", {})
    birefringence = str(config.get("birefringence_model", BIREFRINGENCE_OFF))
    if birefringence == BIREFRINGENCE_OFF:
        phase = k_step * local_delta_n
        return values * np.exp(1j * phase)[None, ...], np.zeros_like(local_delta_n)
    if birefringence == TENSOR_PHOTOELASTIC:
        if photoelastic_tensor_slice is None:
            raise RuntimeError(
                "tensor photoelastic Jones propagation requires FEM tensor fields"
            )
        delta_xx = np.asarray(photoelastic_tensor_slice["xx"], dtype=float)
        delta_yy = np.asarray(photoelastic_tensor_slice["yy"], dtype=float)
        delta_xy = np.asarray(photoelastic_tensor_slice["xy"], dtype=float)
        if any(
            component.shape != local_delta_n.shape
            for component in (delta_xx, delta_yy, delta_xy)
        ):
            raise ValueError(
                "photoelastic tensor slices must match the optical y/x grid"
            )
        # ``local_delta_n`` already contains the transverse tensor mean from
        # the mechanical solve.  Apply only the traceless transverse part here
        # so common phase is not counted twice.
        anisotropy_x = 0.5 * (delta_xx - delta_yy)
        anisotropy_xy = delta_xy
        magnitude = np.sqrt(anisotropy_x**2 + anisotropy_xy**2)
        anisotropic_phase = k_step * magnitude
        cosine = np.cos(anisotropic_phase)
        sine_over_magnitude = np.divide(
            np.sin(anisotropic_phase),
            magnitude,
            out=np.asarray(k_step * np.ones_like(magnitude)),
            where=magnitude > np.finfo(float).tiny,
        )
        ex, ey = values[0], values[1]
        output = np.empty_like(values)
        common_phase = np.exp(1j * k_step * local_delta_n)
        output[0] = common_phase * (
            cosine * ex
            + 1j * sine_over_magnitude * (anisotropy_x * ex + anisotropy_xy * ey)
        )
        output[1] = common_phase * (
            cosine * ey
            + 1j * sine_over_magnitude * (anisotropy_xy * ex - anisotropy_x * ey)
        )
        retardance = np.asarray(2.0 * anisotropic_phase)
        if retardance.ndim > 2:
            retardance = retardance[len(retardance) // 2]
        return output, retardance
    if birefringence != RADIAL_TANGENTIAL:
        raise ValueError(f"unsupported Seed birefringence model {birefringence!r}")

    dn_dT = float(material.get("dn_dT_per_K", 0.0))
    delta_temperature = local_delta_n / dn_dT
    n0 = np.asarray(refractive_index, dtype=float)
    if n0.ndim > 0:
        n0 = n0.reshape((len(n0),) + (1,) * (values.ndim - 2))
    alpha = float(
        config.get("thermal_expansion_per_K")
        if config.get("thermal_expansion_per_K") is not None
        else material.get("mechanical", {}).get("thermal_expansion_per_K", 0.0)
    )
    coefficient_r = float(config.get("radial_photoelastic_C", 0.0))
    coefficient_t = float(config.get("tangential_photoelastic_C", 0.0))
    slope_r = 2.0 * n0**3 * alpha * coefficient_r
    slope_t = 2.0 * n0**3 * alpha * coefficient_t
    center_x = float(config.get("center_x_m", 0.0))
    center_y = float(config.get("center_y_m", 0.0))
    center_delta_temperature = _sample_transverse_at(
        delta_temperature,
        grid,
        center_x,
        center_y,
    )
    # The reduced plane-stress formula is a spatial-curvature result, not a
    # constitutive law that can be applied to absolute T-T_cool.  Removing the
    # per-slice axial temperature makes uniform heating a pure common phase and
    # reproduces the analytic parabolic solution of Chénais et al.
    photoelastic_temperature = delta_temperature - center_delta_temperature
    index_r = local_delta_n + slope_r * photoelastic_temperature
    index_t = local_delta_n + slope_t * photoelastic_temperature
    phase_r = k_step * index_r
    phase_t = k_step * index_t

    xx, yy = grid.transverse_mesh
    theta = np.arctan2(yy - center_y, xx - center_x)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    ex, ey = values[0], values[1]
    radial = cosine * ex + sine * ey
    tangential = -sine * ex + cosine * ey
    radial *= np.exp(1j * phase_r)
    tangential *= np.exp(1j * phase_t)
    output = np.empty_like(values)
    output[0] = cosine * radial - sine * tangential
    output[1] = sine * radial + cosine * tangential

    # For a broadband field the public diagnostic records the centre-bin map;
    # each wavelength is nevertheless propagated with its own k and n0.
    retardance = np.asarray(phase_r - phase_t)
    if retardance.ndim > 2:
        retardance = retardance[len(retardance) // 2]
    return output, retardance


def polarization_diagnostics(
    field: np.ndarray,
    grid: Any,
    seed_cfg: dict[str, Any],
    accumulated_retardance_rad: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return integrated Stokes and input-analyzer diagnostics."""

    values = np.asarray(field, dtype=complex)
    if values.shape[0] != 2:
        raise ValueError("polarization diagnostics require Jones axis zero of length two")
    ex = values[0].reshape((-1, *values.shape[-2:]))
    ey = values[1].reshape((-1, *values.shape[-2:]))

    def integrate(density: np.ndarray) -> float:
        return float(np.sum(grid.integrate_transverse(density)))

    power_x = integrate(np.abs(ex) ** 2)
    power_y = integrate(np.abs(ey) ** 2)
    cross = np.sum(grid.integrate_transverse(ex * np.conj(ey)))
    s0 = power_x + power_y
    s1 = power_x - power_y
    s2 = 2.0 * float(np.real(cross))
    # Convention: S3 > 0 for the configured left-circular state [1,+i]/sqrt(2).
    s3 = -2.0 * float(np.imag(cross))

    input_vector = input_jones_vector(seed_cfg)
    orthogonal = np.asarray(
        [-np.conj(input_vector[1]), np.conj(input_vector[0])], dtype=complex
    )
    parallel_field = np.conj(input_vector[0]) * ex + np.conj(input_vector[1]) * ey
    orthogonal_field = np.conj(orthogonal[0]) * ex + np.conj(orthogonal[1]) * ey
    parallel_power = integrate(np.abs(parallel_field) ** 2)
    orthogonal_power = integrate(np.abs(orthogonal_field) ** 2)
    denominator = max(parallel_power + orthogonal_power, np.finfo(float).tiny)
    degree = math.sqrt(s1**2 + s2**2 + s3**2) / max(
        s0, np.finfo(float).tiny
    )
    result: dict[str, Any] = {
        "model": JONES,
        "basis": "fixed_lab_x_y",
        "input_state": str(
            seed_cfg.get("polarization", {}).get("input_state", "linear")
        ),
        "input_jones_real": [float(value.real) for value in input_vector],
        "input_jones_imag": [float(value.imag) for value in input_vector],
        "output_x_quantity": power_x,
        "output_y_quantity": power_y,
        "stokes_S0": s0,
        "stokes_S1": s1,
        "stokes_S2": s2,
        "stokes_S3": s3,
        "degree_of_polarization_from_integrated_stokes": degree,
        "parallel_analyzer_quantity": parallel_power,
        "orthogonal_analyzer_quantity": orthogonal_power,
        "orthogonal_analyzer_fraction": orthogonal_power / denominator,
        "parallel_analyzer_beam_quality": sampled_beam_quality(
            parallel_field,
            grid,
        ),
        "orthogonal_analyzer_beam_quality": sampled_beam_quality(
            orthogonal_field,
            grid,
        ),
        "analyzer_scope": (
            "Ideal analyzer referenced to the configured input Jones state; this is "
            "a diagnostic, not an automatically applied optical loss. Analyzer-resolved "
            "beam quality is reported separately from the polarization-insensitive total field."
        ),
    }
    if accumulated_retardance_rad is not None:
        retardance = np.asarray(accumulated_retardance_rad, dtype=float)
        result.update(
            {
                "coordinate_accumulated_retardance_min_rad": float(np.min(retardance)),
                "coordinate_accumulated_retardance_max_rad": float(np.max(retardance)),
                "coordinate_accumulated_retardance_peak_to_valley_rad": float(
                    np.ptp(retardance)
                ),
                "retardance_map_scope": (
                    "Signed local retardance summed on the fixed sampled coordinates. "
                    "The Jones field, not this audit map, carries diffraction and relay transport."
                ),
            }
        )
    return result


def validate_polarization_config(
    config: dict[str, Any],
    *,
    resolved_signal_model: str,
    resolved_physics: dict[str, Any],
) -> None:
    seed = config["seed"]
    polarization = seed.get("polarization", {})
    model = str(polarization.get("model", SCALAR))
    if model not in {SCALAR, JONES}:
        raise ValueError("seed.polarization.model must be scalar or jones")
    state = str(polarization.get("input_state", "linear"))
    if state not in {"linear", "elliptical", "right_circular", "left_circular"}:
        raise ValueError(
            "seed.polarization.input_state must be linear, elliptical, right_circular, or left_circular"
        )
    for key in (
        "linear_angle_deg",
        "relative_phase_y_minus_x_rad",
        "center_x_m",
        "center_y_m",
        "radial_photoelastic_C",
        "tangential_photoelastic_C",
    ):
        if not math.isfinite(float(polarization.get(key, 0.0))):
            raise ValueError(f"seed.polarization.{key} must be finite")
    birefringence = str(
        polarization.get("birefringence_model", BIREFRINGENCE_OFF)
    )
    if birefringence not in {
        BIREFRINGENCE_OFF,
        RADIAL_TANGENTIAL,
        TENSOR_PHOTOELASTIC,
    }:
        raise ValueError(
            "seed.polarization.birefringence_model must be off, "
            "radial_tangential_plane_stress, or tensor_photoelastic"
        )
    if model == SCALAR:
        if birefringence != BIREFRINGENCE_OFF:
            raise ValueError(
                "photoelastic birefringence requires seed.polarization.model=jones"
            )
        return
    if resolved_signal_model != "fluence":
        raise ValueError(
            "Jones Seed propagation currently requires simulation.signal_model=fluence"
        )
    if str(seed.get("spatial_gain_transport", {}).get("model", "local_field")) != "local_field":
        raise ValueError(
            "Jones Seed propagation requires seed.spatial_gain_transport.model=local_field"
        )
    spectrum = seed.get("spectrum", {})
    if str(spectrum.get("type", "monochromatic")) != "monochromatic" or int(
        spectrum.get("sample_count", 1)
    ) != 1:
        raise ValueError(
            "Jones Seed propagation currently requires a one-bin monochromatic spectrum"
        )
    if birefringence == RADIAL_TANGENTIAL:
        material = config["material"]
        if resolved_physics.get("refractive_index_model") != "thermal_3d":
            raise ValueError(
                "radial/tangential thermal birefringence requires refractive_index_model=thermal_3d"
            )
        if float(material.get("dn_dT_per_K", 0.0)) == 0.0:
            raise ValueError(
                "radial/tangential thermal birefringence requires nonzero material.dn_dT_per_K"
            )
        if str(material.get("mechanical", {}).get("stress_model", "none")) != "none":
            raise ValueError(
                "radial/tangential plane-stress birefringence currently requires material.mechanical.stress_model=none "
                "so temperature can be recovered unambiguously from scalar thermal delta_n"
            )
        alpha = polarization.get("thermal_expansion_per_K")
        if alpha is None:
            alpha = material.get("mechanical", {}).get(
                "thermal_expansion_per_K", 0.0
            )
        if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
            raise ValueError(
                "radial/tangential thermal birefringence requires positive thermal expansion"
            )
    if birefringence == TENSOR_PHOTOELASTIC:
        material = config["material"]
        mechanical = material.get("mechanical", {})
        if resolved_physics.get("refractive_index_model") != "thermal_3d":
            raise ValueError(
                "tensor photoelastic Jones propagation requires refractive_index_model=thermal_3d"
            )
        if mechanical.get("stress_model") != "anisotropic_thermoelastic_fem":
            raise ValueError(
                "tensor photoelastic Jones propagation requires "
                "stress_model=anisotropic_thermoelastic_fem"
            )
        if (
            mechanical.get("thermoelastic_fem", {}).get(
                "photoelastic_coupling", "off"
            )
            != TENSOR_PHOTOELASTIC
        ):
            raise ValueError(
                "tensor photoelastic Jones propagation requires "
                "thermoelastic_fem.photoelastic_coupling=tensor_photoelastic"
            )
