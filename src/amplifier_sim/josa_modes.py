"""Shared definitions for selectable JOSA transport fidelity levels.

The transport integrator and the UI both consume these definitions.  Keeping
them in a dependency-free module prevents a label in Streamlit from drifting
away from the numerical backend that is actually selected.
"""

from __future__ import annotations

from typing import Any


JOSA_FULL_3D_MODE = "full_3d_pc"

# These are fidelity *profiles*, not additional numerical kernels.  Keeping
# the profile metadata next to the four actual JOSA modes lets the generator
# and the form present the same progression without inventing a second set of
# backend names.
JOSA_FIDELITY_ORDER = (
    "j1_explicit_cfl",
    "j2_pc_cfl",
    "j3_retarded_pc",
    "j4_full_3d_pc",
)

JOSA_FIDELITY_INFO: dict[str, dict[str, Any]] = {
    "j1_explicit_cfl": {
        "label_zh": "J1 · Explicit CFL + 周期平均（输运基准）",
        "mode": "explicit_cfl",
        "periodic_method": "cycle_average",
        "stack": "transport_only",
        "description_zh": (
            "冻结晶体内衍射与热相位，用论文 CFL 绑定的一阶显式格式核对局部光子输运；"
            "适合排查符号、守恒与网格趋势，不作为高增益最终值。"
        ),
    },
    "j2_pc_cfl": {
        "label_zh": "J2 · PC CFL + 周期平均（论文格式核对）",
        "mode": "pc_cfl",
        "periodic_method": "cycle_average",
        "stack": "transport_only",
        "description_zh": (
            "保持 J1 的隔离输运工况，改用 prediction-correction；Δt=Δs/vg，"
            "因此改变 Nz 也会改变时间分箱。"
        ),
    },
    "j3_retarded_pc": {
        "label_zh": "J3 · 延迟时间 PC + 完整 pulse map（日常定量）",
        "mode": "retarded_pc",
        "periodic_method": "pulse_map",
        "stack": "spatial_operating_point",
        "description_zh": (
            "时间分箱与 Nz 解耦，恢复晶体内横向衍射、热折射率和逐脉冲周期工作点；"
            "仍是相位不敏感的时间×波长强度输运。"
        ),
    },
    "j4_full_3d_pc": {
        "label_zh": "J4 · 完整 3D PC + pulse map（JOSA 上限）",
        "mode": JOSA_FULL_3D_MODE,
        "periodic_method": "pulse_map",
        "stack": "full_3d",
        "description_zh": (
            "再锁定 Seed/Pump FFT、三维热光、泵浦耗尽、群延迟与斜入射走离；"
            "群延迟混箱仍是能量守恒相位闭合近似，不等于 coherent CPA。"
        ),
    },
}

JOSA_MODE_INFO: dict[str, dict[str, Any]] = {
    "explicit_cfl": {
        "tier": "reduced_paper",
        "scheme": "explicit",
        "coordinate_mode": "paper_cfl",
        "label_zh": "论文 Explicit（退化/快速验证，CFL 绑定）",
        "description_zh": "一阶论文格式；用于基准和网格收敛，不作为最终高增益结果首选。",
    },
    "pc_cfl": {
        "tier": "paper_reference",
        "scheme": "pc",
        "coordinate_mode": "paper_cfl",
        "label_zh": "论文 PC（严格 CFL，论文复现）",
        "description_zh": "Springer 等的 prediction-correction 格式，时间步由 Δt=Δs/vg 决定。",
    },
    "retarded_pc": {
        "tier": "configurable_3d",
        "scheme": "pc",
        "coordinate_mode": "retarded_time",
        "label_zh": "可配置 3D PC（延迟时间，日常推荐）",
        "description_zh": "x/y/z 与时间分箱独立；是否使用 FFT、热场和泵浦耗尽由全局开关决定。",
    },
    JOSA_FULL_3D_MODE: {
        "tier": "full_3d_rate_phase_closure_approximation",
        "scheme": "pc",
        "coordinate_mode": "retarded_time",
        "label_zh": "3D PC（群延迟相位闭合近似）",
        "description_zh": (
            "锁定 Seed/Pump FFT、三维瞬态布居、泵浦耗尽、三维稳态/瞬态热光、"
            "群延迟走离和晶体内斜入射横向走离；群延迟混箱采用能量守恒相位闭合近似；"
            "失败不静默退回 FN。"
        ),
    },
}

JOSA_MODES = frozenset(JOSA_MODE_INFO)


def josa_mode_info(mode: str) -> dict[str, Any]:
    """Return immutable-by-convention metadata for one configured mode."""

    try:
        return dict(JOSA_MODE_INFO[str(mode)])
    except KeyError as exc:
        raise ValueError(f"Unsupported JOSA transport mode {mode!r}") from exc


def josa_full_3d_physics_overrides(
    supplied_physics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the non-negotiable solver stack for full 3D rate transport.

    A user-selected transient thermal solve remains transient.  Otherwise the
    full profile selects the steady three-dimensional heat equation.  Surface
    coating/Fresnel choice remains user controlled because an ideal AR coating
    can legitimately be more physical than an uncoated Fresnel boundary.
    """

    supplied = supplied_physics or {}
    requested_thermal = str(supplied.get("thermal_model", "auto"))
    thermal_mode = "transient_3d" if requested_thermal == "transient_3d" else "steady_3d"
    return {
        "optical_propagation": "fft",
        "pump_propagation": "fft",
        "population_dynamics": "transient",
        "pump_depletion": True,
        "gain_saturation": True,
        "thermal_model": thermal_mode,
        "refractive_index_model": "thermal_3d",
        # JOSA transports photon/fluence bins, not a mutually coherent
        # frequency field.  Enabling a Kerr phase here would be a false claim.
        "nonlinear_phase_model": "off",
        "record_optical_planes": True,
    }
