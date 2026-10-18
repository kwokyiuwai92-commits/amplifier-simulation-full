"""Safe cross-module dependency reconciliation for the interactive UI.

The JSON/runtime validator remains strict and never mutates scientific input.
The Streamlit form, however, should not let a user select a parent solver and
then hunt through several tabs for stale child switches.  This module applies
only deterministic compatibility closures and returns an explicit audit row
for every change.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import (
    _signal_upper_reservoir_names,
    resolve_physics,
    resolve_signal_model,
    resolve_population_dynamics,
    resolve_thermal_model,
)
from .josa_modes import JOSA_FULL_3D_MODE
from .multipass import uses_causal_clock
from .coherent_step import nonlinear_gain_coupling_blockers
from .spatial_modes import axis_aligned_q_anisotropy_reasons, axis_focus_m


@dataclass(frozen=True)
class CompatibilityAdjustment:
    path: str
    previous: Any
    current: Any
    reason: str
    location: str

    def as_row(self) -> dict[str, str]:
        return {
            "设置": self.path,
            "原值": str(self.previous),
            "自动联动后": str(self.current),
            "原因": self.reason,
            "界面位置": self.location,
        }

    def as_ownership(self) -> dict[str, Any]:
        """Return the reversible UI-only ownership record for this change."""

        return {
            "path": self.path,
            "previous": copy.deepcopy(self.previous),
            "current": copy.deepcopy(self.current),
            "reason": self.reason,
            "location": self.location,
        }


@dataclass(frozen=True)
class DependencyContract:
    """One compact requested-versus-effective module-chain summary."""

    module: str
    requested: str
    effective: str
    status: str
    location: str

    def as_row(self) -> dict[str, str]:
        return {
            "模块链": self.module,
            "用户选择": self.requested,
            "实际生效": self.effective,
            "状态": self.status,
            "编辑位置": self.location,
        }


_LOCATIONS = {
    "simulation": "① 模型 → 运行方式与求解",
    "physics": "① 模型 → 高级求解器（专家模式）",
    "multiphysics": "① 模型 → Pump · Seed · 热场收敛",
    "seed": "② 光束 → Seed",
    "pump": "② 光束 → Pump",
    "material": "③ 介质",
    "energy_levels": "③ 介质 → 晶体、材料与能级",
    "grid": "④ 数值",
    "ase": "③ 介质 → ASE",
}


def _nested_get(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for token in path.split("."):
        value = value[token]
    return value


def _nested_set(config: dict[str, Any], path: str, value: Any) -> None:
    tokens = path.split(".")
    target: dict[str, Any] = config
    for token in tokens[:-1]:
        target = target.setdefault(token, {})
    target[tokens[-1]] = copy.deepcopy(value)


def _values_equal(left: Any, right: Any) -> bool:
    try:
        same = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(same, np.ndarray):
        return bool(np.all(same))
    try:
        return bool(same)
    except (TypeError, ValueError):
        return False


def release_ui_dependency_ownership(
    config: dict[str, Any],
    ownership: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], tuple[CompatibilityAdjustment, ...]]:
    """Release auto-managed values before recomputing the dependency closure.

    The UI records the exact previous/current pair for every deterministic
    compatibility edit.  On a later rerun, a value is restored only when it
    still equals the value installed by the reducer.  A user-selected child
    override therefore transfers ownership back to the user, while switching
    off the last parent feature cleanly releases stale thermal, periodic, grid,
    or solver modules.
    """

    result = copy.deepcopy(config)
    released: list[CompatibilityAdjustment] = []
    for record in reversed(list(ownership)):
        if not isinstance(record, dict):
            continue
        path = str(record.get("path", ""))
        if not path or "previous" not in record or "current" not in record:
            continue
        try:
            actual = _nested_get(result, path)
        except (KeyError, TypeError):
            continue
        if not _values_equal(actual, record["current"]):
            continue
        restored = copy.deepcopy(record["previous"])
        _nested_set(result, path, restored)
        released.append(
            CompatibilityAdjustment(
                path=path,
                previous=copy.deepcopy(actual),
                current=restored,
                reason=(
                    "上级依赖已不再要求该值，已释放自动托管并恢复联动前设置。"
                ),
                location=str(
                    record.get(
                        "location",
                        _LOCATIONS.get(path.split(".", 1)[0], "完整 JSON"),
                    )
                ),
            )
        )
    released.reverse()
    return result, tuple(released)


def ase_unavailability_reason(config: dict[str, Any]) -> str | None:
    """Share ASE availability between the form and dependency reconciliation."""

    periodic = config.get("simulation", {}).get("periodic_steady_state", {})
    periodic_enabled = bool(periodic.get("enabled", False))
    temporal = config.get("pump", {}).get("temporal", {})
    waveform_requires_transient = (
        temporal.get("power_mode", "average") == "peak"
        or (temporal.get("type", "cw") == "file"
            and temporal.get("value_mode", "relative") == "absolute_W")
    )
    if waveform_requires_transient and not periodic_enabled:
        return "当前 ASE 闭合是稳态辐射输运，不能与非周期的瞬态 Pump 波形共同推进。"
    try:
        reservoir_count = len(_signal_upper_reservoir_names(config["energy_levels"]))
    except (KeyError, TypeError, ValueError):
        reservoir_count = 1
    if reservoir_count != 1:
        return "当前 ASE 自发源只支持一个独立 Signal 上能级储库；所选能级/非均匀展宽网络包含多个储库，已关闭 ASE 而不合并真实布居。"
    return None


def reconcile_ui_dependencies(
    config: dict[str, Any],
) -> tuple[dict[str, Any], tuple[CompatibilityAdjustment, ...]]:
    """Return an idempotently reconciled form config and its visible audit."""

    result = copy.deepcopy(config)
    changes: list[CompatibilityAdjustment] = []

    def assign(path: str, value: Any, reason: str) -> None:
        previous = _nested_get(result, path)
        if _values_equal(previous, value):
            return
        _nested_set(result, path, value)
        changes.append(
            CompatibilityAdjustment(
                path=path,
                previous=copy.deepcopy(previous),
                current=copy.deepcopy(value),
                reason=reason,
                location=_LOCATIONS.get(path.split(".", 1)[0], "完整 JSON"),
            )
        )

    coupling_settings = result.setdefault("multiphysics", {})
    minimum_coupling_iterations = int(
        coupling_settings.get("minimum_iterations", 2)
    )
    if int(coupling_settings.get("maximum_iterations", 20)) < minimum_coupling_iterations:
        assign(
            "multiphysics.maximum_iterations",
            minimum_coupling_iterations,
            "最多耦合迭代不能小于最少迭代；已同步扩大上限，避免固定点在达到允许停止条件前被截断。",
        )
    coherent_settings = result.setdefault("seed", {}).setdefault(
        "coherent", {}
    )
    local_operator_settings = coherent_settings.setdefault(
        "local_operator", {}
    )
    # Retire form-only algorithm choices without changing archived configs or
    # the core reader used for historical result reproduction.
    if local_operator_settings.get("scheme") == "sequential":
        assign(
            "seed.coherent.local_operator.scheme",
            "symmetric_adaptive",
            "表单已移除顺序分裂；导入的历史设置已切换为对称自适应推进，保留原子步预算。",
        )
    nonlinear_solver_settings = result.get("energy_levels", {}).get("nonlinear_solver", {})
    if nonlinear_solver_settings.get("method") == "single_step":
        assign(
            "energy_levels.nonlinear_solver.method",
            "adaptive_substeps",
            "表单已移除单步显式更新；导入的历史设置已切换为按布居耗尽自适应细分，保留原阈值和子步预算。",
        )
    minimum_local_substeps = int(
        local_operator_settings.get("minimum_substeps", 1)
    )
    if int(
        local_operator_settings.get("maximum_substeps", 64)
    ) < minimum_local_substeps:
        assign(
            "seed.coherent.local_operator.maximum_substeps",
            minimum_local_substeps,
            "相干局域算符的最多子步不能小于最少子步；已同步扩大上限。",
        )
    transient_gain_settings = coherent_settings.setdefault(
        "transient_gain", {}
    )
    minimum_transient_slices = int(
        transient_gain_settings.get("slice_count", 9)
    )
    if int(
        transient_gain_settings.get("maximum_slice_count", 65)
    ) < minimum_transient_slices:
        assign(
            "seed.coherent.transient_gain.maximum_slice_count",
            minimum_transient_slices,
            "瞬态增益的最大软窗数不能小于初始软窗数；已同步扩大自适应上限。",
        )

    simulation = result.setdefault("simulation", {})
    seed = result.setdefault("seed", {})
    temporal = seed.setdefault("temporal", {})
    multipass = simulation.setdefault("multipass", {})
    multipass_enabled = bool(multipass.get("enabled", False))
    finite_seed_pulse = str(temporal.get("type", "gaussian")) != "cw"
    execution_mode = str(multipass.get("execution_mode", "auto"))
    if multipass_enabled and not finite_seed_pulse and execution_mode != "sequential":
        assign(
            "simulation.multipass.execution_mode",
            "sequential",
            "CW 没有有限脉冲包络，统一多程计划保留命中与光路并改用完整逐程执行。",
        )
        execution_mode = "sequential"

    physical_pulse_route = multipass_enabled and execution_mode != "sequential"
    if physical_pulse_route:
        assign(
            "simulation.signal_model",
            "spectral_time",
            "因果多程需要在一条绝对时间轴上传播各个脉冲包，因此使用 spectral_time。",
        )
        assign(
            "simulation.pass_mode",
            "simple",
            "所有晶体命中与程间光路已经统一由 simulation.multipass 编译。",
        )
        if str(seed.get("direction", "forward")) != "forward":
            assign(
                "seed.direction",
                "forward",
                "物理多程模板从晶体正向入口注入，后续方向由模板/返回路径产生。",
            )
        assign(
            "seed.temporal.resolved",
            True,
            "物理多程必须解析 Seed 脉冲包络及程间飞行时间。",
        )
        if (
            str(temporal.get("type", "gaussian")) != "file"
            and int(temporal.get("sample_count", 21)) < 3
        ):
            assign(
                "seed.temporal.sample_count",
                3,
                "物理多程的时间插值至少需要三个输入时间样本。",
            )
        assign(
            "seed.joint_time_frequency.model",
            "separable",
            "物理多程目前只接受可分离时间–频率输入；非可分离联合矩阵没有对应的绝对时间场。",
        )
        assign(
            "seed.spatial_modes.model",
            "effective_m2",
            "物理多程当前不传播互不相干的 HG 模态组；保留一个等效 M² 场。",
        )
        assign(
            "seed.spatial_gain_transport.model",
            "local_field",
            "物理多程必须在每个命中位置提交局域增益与共享布居竞争。",
        )
        if str(seed.get("polarization", {}).get("model", "scalar")) == "jones":
            assign(
                "seed.polarization.model",
                "scalar",
                "当前 Jones 后端是单色 whole-fluence，而物理多程需要 spectral_time；保留多程求解器并退回标量偏振。",
            )
            assign(
                "seed.polarization.birefringence_model",
                "off",
                "标量偏振不能继续持有 Jones 双折射算符。",
            )
        periodic_for_route = simulation.setdefault(
            "periodic_steady_state", {}
        )
        discrete_periodic_route = bool(
            periodic_for_route.get("enabled", False)
        ) and str(periodic_for_route.get("method", "pulse_map")) in {
            "pulse_map",
            "reduced_pulse_map",
        }
        if multipass_enabled:
            assign(
                "optical_train.between_pass",
                [],
                "因果 multipass 模板拥有每一段程间光路，不能再叠加旧 between_pass 元件。",
            )
            assign(
                "simulation.between_pass_transmission",
                1.0,
                "因果 multipass 使用自己的 per-transition 透过率；旧全局程间透过率必须为 1，避免重复扣损耗。",
            )
            assign(
                "simulation.flip_x_between_passes",
                False,
                "因果 multipass 的成像/坐标变换由模板显式给出，不能再叠加旧全局翻转。",
            )
            template = str(multipass.get("template", "double_pass"))
            if template == "double_pass":
                hit_count = 2
            elif template == "prescribed" and len(
                multipass.get("prescribed", {}).get("hits", [])
            ) >= 2:
                hit_count = len(
                    multipass.get("prescribed", {}).get("hits", [])
                )
            else:
                hit_count = max(2, int(multipass.get("pass_count", 2)))
            assign(
                "simulation.multipass.pass_count",
                hit_count,
                "多程模板的晶体命中数是唯一程次数来源。",
            )
            assign(
                "simulation.passes",
                hit_count,
                "simulation.passes 是因果模板编译后命中数的派生断言。",
            )
            if discrete_periodic_route:
                assign(
                    "simulation.multipass.include_dark_relaxation",
                    True,
                    "多程 pulse-map 必须把程间暗弛豫放在同一绝对时间账本中，才能闭合周期边界。",
                )
    if multipass_enabled and not physical_pulse_route:
        template = str(multipass.get("template", "double_pass"))
        if template == "double_pass":
            hit_count = 2
        elif template == "prescribed" and len(
            multipass.get("prescribed", {}).get("hits", [])
        ) >= 2:
            hit_count = len(multipass["prescribed"]["hits"])
        else:
            hit_count = max(2, int(multipass.get("pass_count", 2)))
        assign(
            "simulation.multipass.pass_count",
            hit_count,
            "统一多程模板的晶体命中数是唯一程次数来源。",
        )
        assign(
            "simulation.passes",
            hit_count,
            "内部逐程后端从统一多程计划派生命中次数。",
        )
        assign(
            "simulation.pass_mode",
            "simple",
            "多程定义已统一到 simulation.multipass。",
        )
        assign(
            "simulation.between_pass_transmission",
            1.0,
            "程间损耗由统一多程 transition 记录，关闭旧全局重复扣损耗。",
        )
        assign(
            "simulation.flip_x_between_passes",
            False,
            "成像翻转由统一多程 transition 的光学元件记录。",
        )
        assign(
            "optical_train.between_pass",
            [],
            "程间元件已归入统一多程 transition。",
        )

    polarization = seed.setdefault("polarization", {})
    polarization_model = str(polarization.get("model", "scalar"))
    birefringence_model = str(
        polarization.get("birefringence_model", "off")
    )
    if polarization_model != "jones" and birefringence_model != "off":
        assign(
            "seed.polarization.birefringence_model",
            "off",
            "双折射算符作用于 Jones 两分量；标量偏振下必须关闭。",
        )
        birefringence_model = "off"
    if polarization_model == "jones":
        assign(
            "simulation.signal_model",
            "fluence",
            "当前 Jones 偏振传播使用单色 whole-fluence 两分量场。",
        )
        assign(
            "seed.spatial_gain_transport.model",
            "local_field",
            "Jones Ex/Ey 必须共享每个横向单元的局域增益，不能使用标量模态重叠或抛物线拟合闭合。",
        )
        assign(
            "seed.spectrum.type",
            "monochromatic",
            "当前 Jones 后端只支持单色一频点传播。",
        )
        assign(
            "seed.spectrum.sample_count",
            1,
            "当前 Jones 后端只支持一个单色频点。",
        )
        assign(
            "seed.spatial_modes.model",
            "effective_m2",
            "Jones 两分量与互不相干 HG 模态组尚无联合传播合同；保留等效 M² 场。",
        )
        if birefringence_model == "radial_tangential_plane_stress":
            assign(
                "physics.thermal_model",
                "steady_3d",
                "径向/切向热双折射必须读取三维温度场。",
            )
            assign(
                "physics.refractive_index_model",
                "thermal_3d",
                "径向/切向热双折射从三维热折射率场恢复温度差。",
            )
            assign(
                "material.mechanical.stress_model",
                "none",
                "径向平面应力降阶式与独立 FEM 应力 Δn 不能同时叠加。",
            )
            assign(
                "material.mechanical.thermoelastic_fem.photoelastic_coupling",
                "off",
                "径向平面应力 Jones 模型自己生成双折射，关闭独立张量光弹耦合以避免重复计数。",
            )
            if str(
                result["material"]
                .get("mechanical", {})
                .get("endface_deformation", {})
                .get("model", "off")
            ) == "distributed_opd":
                assign(
                    "material.mechanical.endface_deformation.model",
                    "off",
                    "distributed_opd 依赖 FEM 应力/位移场，而径向平面应力 Jones 降阶式要求 stress_model=none；已关闭冲突的端面形变。",
                )
        elif birefringence_model == "tensor_photoelastic":
            assign(
                "physics.thermal_model",
                "steady_3d",
                "张量光弹 Jones 传播必须先求三维温度场。",
            )
            assign(
                "physics.refractive_index_model",
                "thermal_3d",
                "张量光弹 Jones 传播需要三维热/应变折射率场。",
            )
            assign(
                "material.mechanical.stress_model",
                "anisotropic_thermoelastic_fem",
                "张量光弹 Jones 传播必须读取各向异性热弹性 FEM 应变张量。",
            )
            assign(
                "material.mechanical.thermoelastic_fem.photoelastic_coupling",
                "tensor_photoelastic",
                "张量 Jones 延迟器需要 FEM 的 tensor_photoelastic 耦合输出。",
            )

    requested_signal_model = str(
        result.get("simulation", {}).get("signal_model", "auto")
    )
    spatial_mode_model = str(
        result.get("seed", {})
        .get("spatial_modes", {})
        .get("model", "effective_m2")
    )
    if requested_signal_model == "fluence" and spatial_mode_model == "incoherent_hg":
        assign(
            "simulation.signal_model",
            "spectral_time",
            "非相干 HG 需要独立传播模态/波长通道；whole-fluence 后端没有该合同，已切换到 spectral_time。",
        )

    signal_model = resolve_signal_model(result)
    physics = resolve_physics(result)
    nonlinear_phase = str(physics["nonlinear_phase_model"])
    nonlinear_loss = str(physics["nonlinear_loss_model"])

    if signal_model == "commercial_app_fast":
        assign(
            "physics.optical_propagation",
            "abcd",
            "Commercial App 横向场由一个或多个 Gaussian-q 通道表示，必须使用 ABCD 空间后端。",
        )
        if nonlinear_phase == "kerr_time_domain":
            assign(
                "physics.nonlinear_phase_model",
                "kerr_parabolic_lens",
                "Commercial App 不保存完整 x–y–t 复场，Kerr 时间域算符自动换成等效抛物透镜。",
            )
        if nonlinear_loss not in {"off", "strong_field_effective_area"}:
            assign(
                "physics.nonlinear_loss_model",
                "off",
                "Commercial App 只接受专用的 effective-area 强场损耗闭合；完整 x-y-t 强场算符属于 coherent_cpa。",
            )
        assign(
            "seed.joint_time_frequency.model",
            "separable",
            "Commercial App 的一维 coherent envelope 使用可分离时间–频率输入。",
        )
        anisotropy_reasons = axis_aligned_q_anisotropy_reasons(result["seed"])
        if anisotropy_reasons and not np.isclose(
            float(result["seed"].get("rotation_deg", 0.0)) % 180.0,
            0.0,
            atol=1e-12,
        ):
            assign(
                "seed.rotation_deg",
                0.0,
                "Commercial App 使用互不耦合的 x/y q 参数；存在横向各向异性时必须与计算轴对齐（"
                + "；".join(anisotropy_reasons)
                + "）。",
            )
        commercial_app = result["seed"].setdefault("commercial_app_fast", {})
        spectrum = result["seed"].setdefault("spectrum", {})
        if nonlinear_loss == "strong_field_effective_area":
            assign(
                "seed.commercial_app_fast.temporal_model",
                "effective_area_spm",
                "Commercial App effective-area 强场损耗需要一维相干时间包络。",
            )
            if str(
                result["material"].get("strong_field", {}).get(
                    "model", "off"
                )
            ) == "off":
                assign(
                    "material.strong_field.model",
                    "multiphoton",
                    "已选择 Commercial App effective-area 强场损耗，因此启用材料多光子模型。",
                )
        commercial_app_relaxation = bool(
            commercial_app.get("include_relaxation_during_pulse", False)
        )
        if commercial_app_relaxation and (
            str(commercial_app.get("temporal_model", "fixed_duration"))
            != "fixed_duration"
            or str(
                result["seed"].get("temporal", {}).get("type", "gaussian")
            )
            == "cw"
        ):
            assign(
                "seed.commercial_app_fast.include_relaxation_during_pulse",
                False,
                "Commercial App 脉内能级弛豫需要有限脉冲的 fixed_duration 时间箱合同。",
            )
            commercial_app_relaxation = False
        if commercial_app_relaxation:
            if result["energy_levels"].get("gain_override_m_inv") is not None:
                assign(
                    "energy_levels.gain_override_m_inv",
                    None,
                    "Commercial App 脉内弛豫必须更新真实上下能级布居；外加小信号增益没有可推进的储能库。",
                )
            periodic_settings = result["simulation"].setdefault(
                "periodic_steady_state", {}
            )
            if bool(periodic_settings.get("enabled", False)) and str(
                periodic_settings.get("method", "pulse_map")
            ) == "cycle_average":
                assign(
                    "simulation.periodic_steady_state.method",
                    "pulse_map",
                    "Commercial App 脉内能级弛豫必须保留逐个脉冲包络；周期平均已切换为 pulse_map。",
                )
        if (
            str(commercial_app.get("temporal_model", "fixed_duration"))
            == "effective_area_spm"
            and bool(commercial_app.get("spectral_feedback", True))
            and (
                str(spectrum.get("type", "monochromatic")) == "monochromatic"
                or int(spectrum.get("sample_count", 1)) < 2
            )
        ):
            assign(
                "seed.commercial_app_fast.spectral_feedback",
                False,
                "单色/单点粗增益谱无法分辨 SPM 的谱内重分布；保留高分辨率输出诊断，但关闭无效的增益分箱反馈。",
            )
    elif signal_model == "coherent_cpa":
        if nonlinear_loss == "strong_field_effective_area":
            assign(
                "physics.nonlinear_loss_model",
                "strong_field_time_domain",
                "coherent_cpa 使用局域时间域强场损耗；Commercial App 的 effective-area 闭合已切换为对应的完整场算符。",
            )
            nonlinear_loss = "strong_field_time_domain"
        if nonlinear_phase == "kerr_parabolic_lens":
            assign(
                "physics.nonlinear_phase_model",
                "kerr_time_domain"
                if abs(float(result["material"].get("n2_m2_W", 0.0))) > 0.0
                else "off",
                "coherent_cpa 直接传播时间域 Kerr 相位，不使用 Commercial App 抛物透镜闭合。",
            )
        if nonlinear_loss == "strong_field_time_domain" and str(
            result["material"].get("strong_field", {}).get("model", "off")
        ) == "off":
            assign(
                "material.strong_field.model",
                "multiphoton",
                "已选择时间域强场损耗，因此启用其材料多光子损耗模型。",
            )
        assign(
            "seed.joint_time_frequency.model",
            "separable",
            "coherent_cpa 从一个复脉冲场推导时间–频率关系，不能同时使用独立联合权重矩阵。",
        )
        coherent = result["seed"].setdefault("coherent", {})
        fidelity_mode = str(coherent.get("fidelity_mode", "configured"))
        if fidelity_mode in {
            "axisymmetric_kerr",
            "axisymmetric_transient_gain",
        }:
            axisymmetric_reasons: list[str] = []
            azimuthal_fidelity = str(
                coherent.get("axisymmetric", {}).get(
                    "azimuthal_fidelity", "m0"
                )
            )
            retained_azimuthal_order = {
                "m0": 0,
                "m0_m1": 1,
                "m0_m1_m2_quadratic": 2,
                "m0_m1_m2": 2,
            }.get(azimuthal_fidelity, 0)
            radial_hg_modes = result["seed"].get("spatial_modes", {})
            radial_hg_supported = (
                radial_hg_modes.get("model") == "incoherent_hg"
                and radial_hg_modes.get("maximum_total_order") == 1
                and retained_azimuthal_order == 1
                and fidelity_mode == "axisymmetric_kerr"
                and nonlinear_loss == "off"
                and np.isclose(result["seed"].get("m2_x", 1), result["seed"].get("m2_y", 1))
            )
            if not radial_hg_supported and any(
                not np.isclose(
                    float(result["seed"].get(key, 1.0)),
                    1.0,
                    rtol=1e-10,
                    atol=1e-12,
                )
                for key in ("m2_x", "m2_y")
            ):
                axisymmetric_reasons.append("M²x/M²y 不是 1")
            if retained_azimuthal_order < 2:
                for first_key, second_key, label in (
                    ("waist_x_m", "waist_y_m", "束腰"),
                    ("order_x", "order_y", "横向阶数"),
                ):
                    if not np.isclose(
                        float(result["seed"].get(first_key, 1.0)),
                        float(result["seed"].get(second_key, 1.0)),
                        rtol=1e-8,
                        atol=1e-15,
                    ):
                        axisymmetric_reasons.append(
                            f"当前阶数不保留 x/y {label}差异"
                        )
                if not np.isclose(
                    axis_focus_m(result["seed"], "x"),
                    axis_focus_m(result["seed"], "y"),
                    rtol=1e-8,
                    atol=1e-15,
                ):
                    axisymmetric_reasons.append(
                        "当前阶数不保留 x/y 焦点差异"
                    )
            active_grid = (
                result["grid"].get("optical", {})
                if bool(
                    result["grid"].get("optical", {}).get(
                        "enabled", False
                    )
                )
                else result["grid"]
            )
            window_scale = max(
                float(active_grid.get("size_x_m", 1.0)),
                float(active_grid.get("size_y_m", 1.0)),
            )
            centre_tolerance = max(1e-15, 1e-12 * window_scale)
            if retained_azimuthal_order < 1:
                if any(
                    not np.isclose(
                        float(result["seed"].get(f"center_{axis}_m", 0.0)),
                        float(active_grid.get(f"center_{axis}_m", 0.0)),
                        rtol=0.0,
                        atol=centre_tolerance,
                    )
                    for axis in ("x", "y")
                ):
                    axisymmetric_reasons.append(
                        "m=0 不保留 Seed 偏心（请选择 m=0+1）"
                    )
                if any(
                    abs(float(result["seed"].get(key, 0.0) or 0.0)) > 1e-15
                    for key in (
                        "incidence_angle_x_deg",
                        "incidence_angle_y_deg",
                    )
                ):
                    axisymmetric_reasons.append(
                        "m=0 不保留 Seed 入射倾角（请选择 m=0+1）"
                    )
            if str(
                result["seed"].get("spatio_spectral", {}).get(
                    "model", "shared_source"
                )
            ) != "shared_source":
                axisymmetric_reasons.append("存在波长相关空间源")
            if axisymmetric_reasons:
                cartesian_fallback = (
                    "reference_transient_gain"
                    if fidelity_mode == "axisymmetric_transient_gain"
                    else "hybrid_whole_fluence"
                )
                assign(
                    "seed.coherent.fidelity_mode",
                    cartesian_fallback,
                    "轴对称降阶不适用于当前物理光束（"
                    + "；".join(axisymmetric_reasons)
                    + "）；不改写束腰、M²、中心或角度，改用对应的完整笛卡尔相干求解器。",
                )
                fidelity_mode = cartesian_fallback
        transient_gain_required = fidelity_mode in {
            "reference_transient_gain",
            "axisymmetric_transient_gain",
        }
        transient_gain_forbidden = fidelity_mode in {
            "hybrid_whole_fluence",
            "axisymmetric_kerr",
        }
        if transient_gain_required:
            assign(
                "seed.coherent.transient_gain.enabled",
                True,
                "所选相干精度档包含 front→peak→tail 软窗增益，必须启用瞬态增益算符。",
            )
        elif transient_gain_forbidden:
            assign(
                "seed.coherent.transient_gain.enabled",
                False,
                "所选相干精度档使用整脉冲增益闭合，不能同时叠加软窗瞬态增益。",
            )
        if fidelity_mode in {
            "axisymmetric_kerr",
            "axisymmetric_transient_gain",
        }:
            assign(
                "seed.coherent.local_operator.scheme",
                "symmetric_adaptive",
                "轴对称相干传播依赖对称自适应局域算符分裂。",
            )
        if fidelity_mode == "axisymmetric_transient_gain":
            periodic_for_fidelity = result["simulation"].setdefault(
                "periodic_steady_state", {}
            )
            if bool(periodic_for_fidelity.get("enabled", False)):
                assign(
                    "simulation.periodic_steady_state.method",
                    "pulse_map",
                    "L4.5 轴对称瞬态增益必须由完整逐脉冲映射闭合周期布居。",
                )
    else:
        if nonlinear_phase in {"kerr_time_domain", "kerr_parabolic_lens"}:
            assign(
                "physics.nonlinear_phase_model",
                "off",
                "当前 Seed 后端不保留该 Kerr 相位表示；切换到 coherent_cpa 或 Commercial App 才能启用。",
            )
        if nonlinear_loss != "off":
            assign(
                "physics.nonlinear_loss_model",
                "off",
                "强场损耗只在 coherent_cpa 完整时间域或 Commercial App effective-area 专用闭合中定义。",
            )

    # Release stale children through the same visible, reversible ownership
    # mechanism used by the older solver-family switches.
    current_coherent = result["seed"].get("coherent", {})
    current_local = current_coherent.get("local_operator", {})
    whole_pulse_groups = (
        signal_model == "coherent_cpa"
        and current_coherent.get("fidelity_mode") in {"hybrid_whole_fluence", "axisymmetric_kerr"}
        and not current_coherent.get("transient_gain", {}).get("enabled", False)
        and current_local.get("scheme", "symmetric_adaptive") == "symmetric_adaptive"
    )
    if current_local.get("spatial_population_update") == "independent_groups" and not whole_pulse_groups:
        assign("seed.coherent.local_operator.spatial_population_update", "shared_cell",
               "当前后端不使用独立整脉冲增益小片；改用共享布居更新，返回兼容模型时可恢复原设置。")
    current_multirate = current_local.get("multirate_gain", {})
    transient_count_cap = (
        signal_model == "coherent_cpa"
        and current_coherent.get("transient_gain", {}).get("enabled", False)
        and current_local.get("spatial_population_update") == "shared_cell"
        and current_local.get("scheme", "symmetric_adaptive") == "symmetric_adaptive"
    )
    if current_multirate.get("maximum_updates_per_cell") is not None and (
        not (transient_count_cap or (whole_pulse_groups and current_local.get("spatial_population_update") == "independent_groups"))
        or not current_multirate.get("enabled", False)
        or current_coherent.get("transient_gain", {}).get("include_relaxation_during_pulse", False)
        or nonlinear_gain_coupling_blockers(resolve_physics(result), result["material"])
    ):
        assign("seed.coherent.local_operator.multirate_gain.maximum_updates_per_cell", None,
               "增益次数上限支持独立整脉冲小片或共享片瞬态增益，需关闭脉内弛豫和载流子耦合；当前组合改用原有增益更新判据。")
    if (signal_model == "coherent_cpa"
            and result["seed"].get("tail_binning", {}).get("spectral_enabled", False)
            and (current_coherent.get("fidelity_mode") != "axisymmetric_kerr"
                 or current_coherent.get("transient_gain", {}).get("enabled", False))):
        assign("seed.tail_binning.spectral_enabled", False,
               "相干尾部频率分箱只用于轴对称整脉冲增益；当前相干模型恢复完整频率网格。")

    effective_nonlinear_loss = str(
        resolve_physics(result)["nonlinear_loss_model"]
    )
    configured_carrier = (
        result.get("material", {})
        .get("strong_field", {})
        .get("free_carrier", {})
    )
    if (
        effective_nonlinear_loss
        not in {"strong_field_time_domain", "strong_field_effective_area"}
        and str(configured_carrier.get("model", "off")) != "off"
    ):
        assign(
            "material.strong_field.free_carrier.model",
            "off",
            "自由载流子由强场电离/吸收算符产生；强场损耗关闭时该子状态必须停用，重新开启父模块后会恢复原设置。",
        )

    if signal_model == "josa_transport":
        assign(
            "physics.gain_saturation",
            True,
            "JOSA 光子输运与能级布居在每个有序时间箱内共同更新，因此必须提交受激耗尽。",
        )
        if result["energy_levels"].get("gain_override_m_inv") is not None:
            assign(
                "energy_levels.gain_override_m_inv",
                None,
                "JOSA 必须读取真实上下能级布居；外加小信号增益没有可守恒更新的布居储能库。",
            )

    spatial_modes = result["seed"].setdefault("spatial_modes", {})
    if str(spatial_modes.get("model", "effective_m2")) == "incoherent_hg":
        if signal_model not in {
            "fluence",
            "spectral_time",
            "commercial_app_fast",
            "coherent_cpa",
        } or (
            signal_model != "commercial_app_fast"
            and str(result["seed"].get("temporal", {}).get("type", "gaussian"))
            == "cw"
        ):
            assign(
                "seed.spatial_modes.model",
                "effective_m2",
                "当前 Seed 后端/CW 合同不能传播互不相干 HG 通道；保留所选主后端并恢复等效 M²。",
            )
        else:
            preserve_radial_hg_hybrid = (
                signal_model == "coherent_cpa"
                and result["seed"].get("coherent", {}).get("fidelity_mode") == "axisymmetric_kerr"
                and result["physics"].get("optical_propagation") == "hybrid_abcd_fft"
            )
            assign(
                "physics.optical_propagation",
                "hybrid_abcd_fft" if preserve_radial_hg_hybrid else ("abcd" if signal_model == "commercial_app_fast" else "fft"),
                (
                    "Commercial App Fast 为每个 HG 通道独立推进 Gaussian q；其他显式 HG 后端使用 FFT。"
                ),
            )
            if signal_model == "coherent_cpa":
                coherent_hg = result["seed"].setdefault("coherent", {})
                radial_hg_supported = (
                    spatial_modes.get("maximum_total_order") == 1
                    and coherent_hg.get("fidelity_mode") == "axisymmetric_kerr"
                    and coherent_hg.get("axisymmetric", {}).get("azimuthal_fidelity") == "m0_m1"
                    and result["physics"].get("nonlinear_loss_model", "off") == "off"
                )
                if not radial_hg_supported and str(
                    coherent_hg.get("fidelity_mode", "configured")
                ) in {
                    "axisymmetric_kerr",
                    "axisymmetric_transient_gain",
                    "reference_transient_gain",
                }:
                    assign(
                        "seed.coherent.fidelity_mode",
                        "hybrid_whole_fluence",
                        "显式 HG10/HG01 等模式不是轴对称场，并且当前 HG-CPA 后端使用整脉冲共享增益；改用完整 x-y FFT 相干传播。",
                    )
                if bool(
                    coherent_hg.get("transient_gain", {}).get(
                        "enabled", False
                    )
                ):
                    assign(
                        "seed.coherent.transient_gain.enabled",
                        False,
                        "显式 HG-CPA 当前按同频模态总通量更新一次共享布居；软窗 front-to-tail 瞬态增益尚未扩展到多 HG 模态。",
                    )
            if signal_model == "commercial_app_fast":
                if str(
                    resolve_physics(result)["nonlinear_loss_model"]
                ) == "strong_field_effective_area":
                    assign(
                        "physics.nonlinear_loss_model",
                        "off",
                        "Commercial App 非相干 HG 通道需要 fixed_duration 的逐模态共享布居合同；"
                        "当前单 Gaussian effective-area 强场矩不能同时保持各 HG 模态的局部 K 阶强度矩，"
                        "因此保留显式 HG 并关闭该强场损耗近似。",
                    )
                    if str(
                        result.get("material", {})
                        .get("strong_field", {})
                        .get("free_carrier", {})
                        .get("model", "off")
                    ) != "off":
                        assign(
                            "material.strong_field.free_carrier.model",
                            "off",
                            "Commercial App HG 关闭 effective-area 强场父算符后，自由载流子子状态也必须关闭。",
                        )
                assign(
                    "seed.commercial_app_fast.temporal_model",
                    "fixed_duration",
                    "Commercial App Fast 的 HG 共享布居路径当前使用固定脉宽/有序增益时间箱。",
                )
            for axis in ("x", "y"):
                required_order = int(
                    np.ceil(
                        max(
                            (float(result["seed"].get(f"m2_{axis}", 1.0)) - 1.0)
                            / 2.0,
                            0.0,
                        )
                    )
                )
                if int(spatial_modes.get(f"maximum_order_{axis}", 0)) < required_order:
                    assign(
                        f"seed.spatial_modes.maximum_order_{axis}",
                        required_order,
                        f"当前 {axis} 轴 M² 至少需要该 HG 阶数才能表示。",
                    )
                assign(
                    f"seed.spatio_spectral.m2_scaling_exponent_{axis}",
                    0.0,
                    "incoherent_hg 的模态权重目前必须在各波长间保持一致。",
                )
            maximum_total = spatial_modes.get("maximum_total_order")
            if maximum_total is None:
                requested_mode_count = (
                    int(spatial_modes.get("maximum_order_x", 0)) + 1
                ) * (int(spatial_modes.get("maximum_order_y", 0)) + 1)
            else:
                requested_mode_count = sum(
                    1
                    for order_x in range(
                        int(spatial_modes.get("maximum_order_x", 0)) + 1
                    )
                    for order_y in range(
                        int(spatial_modes.get("maximum_order_y", 0)) + 1
                    )
                    if order_x + order_y <= int(maximum_total)
                )
            if int(
                spatial_modes.get("maximum_propagated_modes", 64)
            ) < requested_mode_count:
                assign(
                    "seed.spatial_modes.maximum_propagated_modes",
                    requested_mode_count,
                    "所选 x/y HG 基底包含更多独立模态；已同步扩大传播容量，避免只显示高阶设置却在求解前被容量上限阻断。",
                )

    response_mode = str(
        result["material"].get("spectroscopic_response", {}).get("mode", "off")
    )
    seed_heat_mode = str(
        result.get("multiphysics", {}).get("seed_heat_coupling", "off")
    )
    coupling_model = str(
        result.get("multiphysics", {}).get("coupling_model", "one_way")
    )
    if response_mode == "local_3d" and coupling_model != "thermal_optical_fixed_point":
        assign(
            "multiphysics.coupling_model",
            "thermal_optical_fixed_point",
            "局部 3D 温度/浓度光谱反馈必须在热光固定点中反复更新。",
        )
        coupling_model = "thermal_optical_fixed_point"
    if seed_heat_mode == "fixed_point" and coupling_model != "thermal_optical_fixed_point":
        assign(
            "multiphysics.coupling_model",
            "thermal_optical_fixed_point",
            "Seed fixed-point 产热要求 Pump、布居、热场和 Seed 共同闭环。",
        )
        coupling_model = "thermal_optical_fixed_point"

    if coupling_model == "thermal_optical_fixed_point":
        assign(
            "physics.thermal_model",
            "steady_3d",
            "静态热光固定点需要稳态三维导热解。",
        )
        assign(
            "physics.refractive_index_model",
            "thermal_3d",
            "热光固定点必须把三维温度/应力折射率反馈到光学传播。",
        )
        pump_backend = str(resolve_physics(result)["pump_propagation"])
        endface_model = str(
            result["material"]
            .get("mechanical", {})
            .get("endface_deformation", {})
            .get("model", "off")
        )
        allowed = {"fft", "hybrid_abcd_fft", "stochastic_raytrace"}
        if pump_backend not in allowed or (
            pump_backend == "stochastic_raytrace"
            and endface_model == "distributed_opd"
        ):
            assign(
                "physics.pump_propagation",
                "hybrid_abcd_fft",
                "热光固定点的 Pump 必须响应三维 Δn；端面分布 OPD 也要求波动光学后端。",
            )

    periodic = result["simulation"].setdefault("periodic_steady_state", {})
    minimum_cycles = int(periodic.get("minimum_cycles", 3))
    consecutive_cycles = int(periodic.get("consecutive_cycles", 3))
    minimum_viable_maximum = minimum_cycles + consecutive_cycles - 1
    if int(periodic.get("maximum_cycles", 1000)) < minimum_viable_maximum:
        assign(
            "simulation.periodic_steady_state.maximum_cycles",
            minimum_viable_maximum,
            "逐脉冲固定点必须在最少周期之后仍留出连续满足容差的周期；已扩大最多周期数，使所选停止条件在数学上可达。",
        )
    average_minimum = int(periodic.get("average_minimum_iterations", 2))
    if int(periodic.get("average_maximum_iterations", 40)) < average_minimum:
        assign(
            "simulation.periodic_steady_state.average_maximum_iterations",
            average_minimum,
            "周期平均的最多自洽迭代不能小于最少迭代；已同步扩大上限。",
        )
    seed_temporal_type = str(result["seed"].get("temporal", {}).get("type", "gaussian"))
    if seed_temporal_type == "cw":
        if signal_model == "commercial_app_fast":
            assign(
                "seed.commercial_app_fast.temporal_model",
                "fixed_duration",
                "CW 没有单脉冲时间包络；Commercial App 直接使用平均强度，因此关闭 effective-area SPM 时间闭合。",
            )
        assign(
            "simulation.periodic_steady_state.enabled",
            True,
            "CW Seed 通过平均光子流耦合，必须启用周期平均稳态。",
        )
        assign(
            "simulation.periodic_steady_state.method",
            "cycle_average",
            "pulse_map 只表示离散 Seed 脉冲；CW 自动使用 cycle-average。",
        )
    periodic_enabled = bool(periodic.get("enabled", False))
    periodic_method = str(periodic.get("method", "pulse_map"))
    if periodic_method == "reduced_pulse_map":
        assign(
            "simulation.periodic_steady_state.method",
            "pulse_map",
            "低维脉冲图现在只作为完整 pulse_map 的初值加速器；最终固定点统一由主脉冲闭合。",
        )
        assign(
            "simulation.periodic_steady_state.pulse_map_initialization",
            "reduced_pulse_map",
            "已把旧 reduced_pulse_map 方法迁移到离散周期映射初值层。",
        )
        periodic_method = "pulse_map"
    if (
        periodic_enabled
        and str(periodic.get("pulse_map_initialization", "auto"))
        == "reduced_pulse_map"
        and signal_model != "coherent_cpa"
    ):
        assign(
            "simulation.periodic_steady_state.pulse_map_initialization",
            "cycle_average",
            "低维脉冲初值只为 coherent_cpa 提供加速；当前后端直接使用周期平均初值后进入完整 pulse_map。",
        )
    carrier = (
        result.get("material", {})
        .get("strong_field", {})
        .get("free_carrier", {})
    )
    repetition_rate = float(result["simulation"].get("repetition_rate_Hz", 0.0))
    if (
        periodic_enabled
        and periodic_method == "cycle_average"
        and seed_temporal_type != "cw"
        and str(carrier.get("model", "off")) == "rate_equation"
        and repetition_rate > 0.0
        and float(carrier.get("lifetime_s", 1e-9)) > 0.0
    ):
        residual_fraction = math.exp(
            -1.0 / (repetition_rate * float(carrier.get("lifetime_s", 1e-9)))
        )
        residual_limit = float(
            periodic.get(
                "maximum_unmodeled_free_carrier_residual_fraction", 1e-4
            )
        )
        if residual_fraction > residual_limit:
            assign(
                "simulation.periodic_steady_state.method",
                "pulse_map",
                "自由载流子在相邻脉冲间不能充分衰减，cycle-average 会丢失真实状态；已切换为逐脉冲周期固定点。",
            )
            periodic_method = "pulse_map"
    if (
        periodic_enabled
        and str(periodic.get("pulse_map_initialization", "auto"))
        == "reduced_pulse_map"
        and str(carrier.get("model", "off")) == "rate_equation"
    ):
        assign(
            "simulation.periodic_steady_state.pulse_map_initialization",
            "initial_population",
            "低维脉冲初值不传播时域自由载流子状态；完整 pulse_map 将直接从已有布居与载流子边界开始。",
        )
    if (
        periodic_enabled
        and str(periodic.get("pulse_map_initialization", "auto"))
        == "reduced_pulse_map"
        and str(
            periodic.get("reduced_pulse_map", {}).get(
                "model", "phase_insensitive"
            )
        )
        == "axisymmetric_coherent_kerr"
        and str(
            result["seed"]
            .get("coherent", {})
            .get("fidelity_mode", "configured")
        )
        != "axisymmetric_kerr"
    ):
        assign(
            "simulation.periodic_steady_state.reduced_pulse_map.model",
            "phase_insensitive",
            "轴对称 Kerr 初值代理只与 axisymmetric_kerr 主求解器共享同一降阶自由度；当前主求解器改用相位不敏感初值。",
        )
    full_josa = bool(
        str(result["simulation"].get("signal_model", "auto")) == "josa_transport"
        and str(result["seed"].get("josa", {}).get("mode", "retarded_pc"))
        == JOSA_FULL_3D_MODE
    )
    if periodic_enabled and periodic_method in {
        "pulse_map",
        "reduced_pulse_map",
    }:
        assign(
            "physics.population_dynamics",
            "transient",
            "逐脉冲周期映射必须提交每次 Pump/Seed 事件后的瞬态布居。",
        )
        assign(
            "physics.gain_saturation",
            True,
            "pulse-map 必须让每个放大脉冲耗尽局部反转。",
        )
    elif periodic_enabled and periodic_method == "cycle_average":
        assign(
            "physics.population_dynamics",
            "transient" if full_josa else "steady_state",
            "周期平均稳态由平均光子流固定点接管布居；full_3d_pc 仍保留最终瞬态探针。",
        )

    if periodic_enabled and result["energy_levels"].get("gain_override_m_inv") is not None:
        assign(
            "energy_levels.gain_override_m_inv",
            None,
            "周期固定点必须保存并更新真实能级布居，不能每个周期重新建立外加 g₀。",
        )

    spatial_gain = str(
        result["seed"].get("spatial_gain_transport", {}).get("model", "local_field")
    )
    if spatial_gain == "parabolic_gain_fit":
        if signal_model not in {"fluence", "commercial_app_fast"}:
            assign(
                "seed.spatial_gain_transport.model",
                "local_field",
                "局域抛物线增益拟合只属于 fluence/Commercial App 的降阶空间闭合；当前后端保留自己的局域增益算符。",
            )
        elif str(result["seed"].get("temporal", {}).get("type", "gaussian")) != "cw":
            assign(
                "seed.spatial_gain_transport.model",
                "local_field",
                "抛物线增益拟合与当前 stationary photon-flux 闭合只支持 CW；有限脉冲保留局域增益算符。",
            )
    elif spatial_gain == "gaussian_mode_overlap":
        spatial_modes_for_gain = result["seed"].get("spatial_modes", {})
        spatial_mode_for_gain = str(
            spatial_modes_for_gain.get("model", "effective_m2")
        )
        pure_hg00 = bool(
            spatial_mode_for_gain == "hermite_gaussian"
            and int(spatial_modes_for_gain.get("maximum_order_x", 0)) == 0
            and int(spatial_modes_for_gain.get("maximum_order_y", 0)) == 0
            and np.isclose(float(result["seed"].get("m2_x", 1.0)), 1.0)
            and np.isclose(float(result["seed"].get("m2_y", 1.0)), 1.0)
        )
        commercial_app_mode_resolved = bool(
            signal_model == "commercial_app_fast"
            and spatial_mode_for_gain == "incoherent_hg"
        )
        profile = str(result["seed"].get("profile", "super_gaussian"))
        gaussian_profile = bool(
            profile == "gaussian"
            or (
                profile in {"super_gaussian", "radial_super_gaussian"}
                and np.isclose(
                    float(result["seed"].get("order_x", 1.0)),
                    1.0,
                    rtol=0.0,
                    atol=1e-12,
                )
                and np.isclose(
                    float(result["seed"].get("order_y", 1.0)),
                    1.0,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        )
        unequal_axes = not np.isclose(
            float(result["seed"].get("waist_x_m", 0.0)),
            float(result["seed"].get("waist_y_m", 0.0)),
            rtol=1e-12,
            atol=0.0,
        )
        rotated_ellipse = bool(
            unequal_axes
            and not np.isclose(
                float(result["seed"].get("rotation_deg", 0.0)) % 180.0,
                0.0,
                atol=1e-12,
            )
        )
        fallback_reason: str | None = None
        if signal_model not in {"fluence", "commercial_app_fast"}:
            fallback_reason = (
                "Gaussian 模态重叠闭合只支持 fluence 或 Commercial App；多色、JOSA 和 coherent CPA 保留各自的局域增益算符。"
            )
        elif str(
            result["seed"].get("spectrum", {}).get("type", "monochromatic")
        ) != "monochromatic":
            fallback_reason = (
                "Gaussian 模态重叠当前只定义单色标量增益；宽带 Seed 改回逐波长局域增益。"
            )
        elif (
            spatial_mode_for_gain != "effective_m2"
            and not pure_hg00
            and not commercial_app_mode_resolved
        ):
            fallback_reason = (
                "Gaussian 模态重叠只接受 effective_m2、纯 HG00，或 Commercial App 的逐模态 HG 共享布居合同。"
            )
        elif not gaussian_profile:
            fallback_reason = (
                "Gaussian 模态重叠不能代表超高斯、top-hat、环形或自定义横向形状；保留局域增益。"
            )
        elif rotated_ellipse:
            fallback_reason = (
                "当前模态重叠投影是轴对齐椭圆；旋转椭圆必须使用局域场增益。"
            )
        if fallback_reason is not None:
            assign(
                "seed.spatial_gain_transport.model",
                "local_field",
                fallback_reason,
            )
        elif periodic_enabled and periodic_method in {
            "pulse_map",
            "reduced_pulse_map",
        }:
            assign(
                "seed.spatial_gain_transport.model",
                "local_field",
                "pulse-map 必须保留局部 Frantz–Nodvik 耗尽；Gaussian 标量重叠闭合不能代替该饱和算符。",
            )
        elif (
            signal_model != "commercial_app_fast"
            and bool(resolve_physics(result)["gain_saturation"])
        ):
            assign(
                "physics.gain_saturation",
                False,
                "Gaussian 标量模态重叠尚未定义空间 FN 耗尽，需关闭增益饱和。",
            )

    thermal_for_heat = result["material"].setdefault("thermal", {})
    if (
        str(thermal_for_heat.get("heat_source_model", "pump_component_ledger"))
        == "pump_component_ledger"
        and str(thermal_for_heat.get("heat_fraction", "quantum_defect"))
        == "lifetime_quantum_defect"
    ):
        population_for_heat = str(
            resolve_physics(result)["population_dynamics"]
        )
        stationary_lifetime_heat = bool(
            population_for_heat == "steady_state"
            or (
                periodic_enabled
                and periodic_method == "cycle_average"
            )
        )
        if not stationary_lifetime_heat:
            assign(
                "material.thermal.heat_fraction",
                "quantum_defect",
                "寿命量子亏损需要周期平均/稳态上能级布居；当前瞬态或 pulse-map 只返回周期边界布居，已退回不重复计数的量子亏损账本。若要逐事件热量，请使用 event_energy_ledger。",
            )

    if str(resolve_physics(result)["pump_propagation"]) == "super_gaussian_abcd":
        if str(result["pump"].get("profile", "gaussian")) not in {
            "gaussian",
            "super_gaussian",
            "radial_super_gaussian",
        }:
            assign(
                "pump.profile",
                "radial_super_gaussian",
                "super_gaussian_abcd 需要 Gaussian 或超高斯源面轮廓。",
            )

    mechanical = result["material"].setdefault("mechanical", {})
    fem = mechanical.setdefault("thermoelastic_fem", {})
    if str(fem.get("photoelastic_coupling", "off")) == "tensor_photoelastic":
        assign(
            "material.mechanical.stress_model",
            "anisotropic_thermoelastic_fem",
            "张量光弹耦合需要各向异性热弹性 FEM 应力张量。",
        )
    endface = mechanical.setdefault("endface_deformation", {})
    if str(endface.get("model", "off")) == "distributed_opd" and str(
        mechanical.get("stress_model", "none")
    ) not in {"thermoelastic_fem", "anisotropic_thermoelastic_fem"}:
        assign(
            "material.mechanical.stress_model",
            "thermoelastic_fem",
            "分布式端面形变必须由热弹性 FEM 位移场提供。",
        )
    stress_model = str(mechanical.get("stress_model", "none"))
    if stress_model in {"thermoelastic_fem", "anisotropic_thermoelastic_fem"}:
        if str(resolve_physics(result)["thermal_model"]) not in {
            "steady_3d",
            "transient_3d",
        }:
            assign(
                "physics.thermal_model",
                "steady_3d",
                "热弹性 FEM 必须读取一个三维温度场。",
            )

    thermal = result["material"].setdefault("thermal", {})
    if str(thermal.get("embedded_boundary_area_model", "cartesian")) == "analytic_cylinder" and str(
        result["material"].get("geometry", {}).get("shape", "cuboid")
    ) != "cylinder":
        assign(
            "material.thermal.embedded_boundary_area_model",
            "cartesian",
            "解析圆柱边界面积只适用于圆柱晶体；当前几何保留并改用通用笛卡尔面积。",
        )
    if thermal.get("entities") and str(
        thermal.get("embedded_boundary_area_model", "cartesian")
    ) != "cartesian":
        assign(
            "material.thermal.embedded_boundary_area_model",
            "cartesian",
            "多热实体导热需要通用笛卡尔嵌入边界面积。",
        )

    if str(thermal.get("heat_source_model", "pump_component_ledger")) == "event_energy_ledger":
        event_ledger = result["energy_levels"].setdefault("event_energy_ledger", {})
        has_partition_contract = bool(
            event_ledger.get("decay_partitions")
            or event_ledger.get("nonlinear_partitions")
        )
        if not bool(event_ledger.get("enabled", False)) and not has_partition_contract:
            assign(
                "material.thermal.heat_source_model",
                "pump_component_ledger",
                "事件能量热源尚无衰减/非线性分区定义；保留可运行的 Pump 分量账本，避免凭空猜测能级热分配。",
            )
        else:
            assign(
                "energy_levels.event_energy_ledger.enabled",
                True,
                "已有事件能量分区，因此启用能级事件记账以供所选热源读取。",
            )

    pump_temporal = result["pump"].setdefault("temporal", {})
    waveform_requires_transient = (
        str(pump_temporal.get("power_mode", "average")) == "peak"
        or (
            str(pump_temporal.get("type", "cw")) == "file"
            and str(pump_temporal.get("value_mode", "relative")) == "absolute_W"
        )
    )
    if waveform_requires_transient and not periodic_enabled:
        assign(
            "physics.population_dynamics",
            "transient",
            "峰值/绝对功率 Pump 波形需要瞬态布居积分；否则会被错误地当作 CW。",
        )

    if str(result.get("ase", {}).get("model", "off")) != "off":
        ase_blocker = ase_unavailability_reason(result)
        if ase_blocker is not None:
            assign("ase.model", "off", ase_blocker)
        elif (
            str(resolve_physics(result)["population_dynamics"])
            != "steady_state"
            and not periodic_enabled
        ):
            assign(
                "physics.population_dynamics",
                "steady_state",
                "启用 ASE 时必须让 ASE 与稳态布居共同迭代。",
            )

    if str(result["grid"].get("spacing_xy", "uniform")) in {"center_stretched", "refined_region"}:
        resolved = resolve_physics(result)
        optical = result["grid"].setdefault("optical", {})
        if resolved["optical_propagation"] != "off" and not bool(
            optical.get("enabled", False)
        ):
            assign(
                "grid.optical.enabled",
                True,
                "局部加密物理网格不能直接做 FFT；Seed 自动启用独立均匀光学网格。",
            )
        analytic_pump = {
            "prescribed_caustic",
            "super_gaussian_abcd",
            "stochastic_raytrace",
        }
        if str(resolve_physics(result)["pump_propagation"]) not in analytic_pump:
            pump_grid = result["grid"].setdefault("pump_optical", {})
            mode = str(pump_grid.get("mode", "shared"))
            if mode not in {"shared", "custom"}:
                assign(
                    "grid.pump_optical.mode",
                    "shared",
                    "采样 Pump 在局部加密物理网格上需要复用均匀光学网格。",
                )
                mode = "shared"
            if mode == "shared" and not bool(optical.get("enabled", False)):
                assign(
                    "grid.optical.enabled",
                    True,
                    "采样 Pump 已选择共用光学网格；局部加密物理网格不能直接执行 FFT，因此自动启用均匀光学网格。",
                )

    return result, tuple(changes)


def dependency_contracts(
    config: dict[str, Any],
    ownership: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> tuple[DependencyContract, ...]:
    """Summarize the major requested/effective module chains for the UI."""

    effective = resolve_physics(config)
    owned_paths = {
        str(record.get("path", ""))
        for record in ownership
        if isinstance(record, dict)
    }

    def requested_value(path: str, current: Any) -> Any:
        """Recover the user's pre-contract value for an auto-managed path."""

        requested = current
        for record in reversed(list(ownership)):
            if not isinstance(record, dict) or record.get("path") != path:
                continue
            if _values_equal(requested, record.get("current")):
                requested = copy.deepcopy(record.get("previous"))
        return requested

    def status(active: bool, prefixes: tuple[str, ...]) -> str:
        if not active:
            return "关闭"
        if any(
            path == prefix or path.startswith(prefix + ".")
            for path in owned_paths
            for prefix in prefixes
        ):
            return "自动托管"
        return "已就绪"

    signal_model = resolve_signal_model(config)
    configured_signal = config.get("simulation", {}).get(
        "signal_model", "auto"
    )
    signal_requested = str(
        requested_value("simulation.signal_model", configured_signal)
    )
    periodic = config.get("simulation", {}).get("periodic_steady_state", {})
    periodic_active = bool(periodic.get("enabled", False))
    periodic_requested_active = bool(
        requested_value(
            "simulation.periodic_steady_state.enabled", periodic_active
        )
    )
    periodic_requested_method = str(
        requested_value(
            "simulation.periodic_steady_state.method",
            periodic.get("method", "pulse_map"),
        )
    )
    coupling = config.get("multiphysics", {})
    coupling_model = str(coupling.get("coupling_model", "one_way"))
    requested_coupling_model = str(
        requested_value("multiphysics.coupling_model", coupling_model)
    )
    seed_heat = str(coupling.get("seed_heat_coupling", "off"))
    requested_seed_heat = str(
        requested_value("multiphysics.seed_heat_coupling", seed_heat)
    )
    response_mode = str(
        config.get("material", {})
        .get("spectroscopic_response", {})
        .get("mode", "off")
    )
    thermal_model = resolve_thermal_model(config)
    thermal_active = thermal_model != "off"
    mechanics = config.get("material", {}).get("mechanical", {})
    stress_model = str(mechanics.get("stress_model", "none"))
    requested_stress_model = str(
        requested_value("material.mechanical.stress_model", stress_model)
    )
    photoelastic = str(
        mechanics.get("thermoelastic_fem", {}).get(
            "photoelastic_coupling", "off"
        )
    )
    endface = str(
        mechanics.get("endface_deformation", {}).get("model", "off")
    )
    requested_photoelastic = str(
        requested_value(
            "material.mechanical.thermoelastic_fem.photoelastic_coupling",
            photoelastic,
        )
    )
    requested_endface = str(
        requested_value(
            "material.mechanical.endface_deformation.model", endface
        )
    )
    mechanics_active = (
        stress_model != "none"
        or photoelastic != "off"
        or endface != "off"
    )
    grid = config.get("grid", {})
    optical_grid = grid.get("optical", {})
    pump_grid = grid.get("pump_optical", {})
    ase_model = str(config.get("ase", {}).get("model", "off"))
    requested_ase_model = str(requested_value("ase.model", ase_model))
    configured_phase = str(
        config.get("physics", {}).get("nonlinear_phase_model", "auto")
    )
    configured_loss = str(
        config.get("physics", {}).get("nonlinear_loss_model", "auto")
    )
    requested_phase = str(
        requested_value("physics.nonlinear_phase_model", configured_phase)
    )
    requested_loss = str(
        requested_value("physics.nonlinear_loss_model", configured_loss)
    )
    nonlinear_phase = str(effective.get("nonlinear_phase_model", "off"))
    nonlinear_loss = str(effective.get("nonlinear_loss_model", "off"))
    pump = config.get("pump", {})
    pump_absorption = str(
        pump.get("absorption_transport", {}).get(
            "model", "population_dependent"
        )
    )
    requested_pump_absorption = str(
        requested_value(
            "pump.absorption_transport.model", pump_absorption
        )
    )
    spatial_gain = str(
        config.get("seed", {})
        .get("spatial_gain_transport", {})
        .get("model", "local_field")
    )
    requested_spatial_gain = str(
        requested_value("seed.spatial_gain_transport.model", spatial_gain)
    )
    spatial_modes = str(
        config.get("seed", {})
        .get("spatial_modes", {})
        .get("model", "effective_m2")
    )
    requested_spatial_modes = str(
        requested_value("seed.spatial_modes.model", spatial_modes)
    )
    polarization = config.get("seed", {}).get("polarization", {})
    polarization_model = str(polarization.get("model", "scalar"))
    birefringence_model = str(
        polarization.get("birefringence_model", "off")
    )
    requested_polarization = str(
        requested_value("seed.polarization.model", polarization_model)
    )
    requested_birefringence = str(
        requested_value(
            "seed.polarization.birefringence_model",
            birefringence_model,
        )
    )
    multipass = config.get("simulation", {}).get("multipass", {})
    multipass_active = bool(multipass.get("enabled", False))
    passes = (
        int(multipass.get("pass_count", 2)) if multipass_active else 1
    )
    execution = (
        "off" if not multipass_active
        else "causal" if uses_causal_clock(config["simulation"], config["seed"], signal_model)
        else "sequential"
    )
    requested_multipass = bool(
        requested_value("simulation.multipass.enabled", multipass_active)
    )
    thermal_settings = config.get("material", {}).get("thermal", {})
    heat_source = str(
        thermal_settings.get("heat_source_model", "pump_component_ledger")
    )
    heat_fraction = thermal_settings.get("heat_fraction", "quantum_defect")
    requested_heat_source = str(
        requested_value("material.thermal.heat_source_model", heat_source)
    )
    requested_heat_fraction = requested_value(
        "material.thermal.heat_fraction", heat_fraction
    )
    strong_field = config.get("material", {}).get("strong_field", {})
    carrier_model = str(
        strong_field.get("free_carrier", {}).get("model", "off")
    )
    passive_seed_heat = bool(
        config.get("material", {})
        .get("thermal", {})
        .get("include_seed_passive_absorption_heat", False)
    )
    strong_active = (
        ase_model != "off"
        or nonlinear_phase != "off"
        or nonlinear_loss != "off"
    )

    return (
        DependencyContract(
            module="Pump 传播与主动吸收",
            requested=(
                f"propagation={requested_value('physics.pump_propagation', config.get('physics', {}).get('pump_propagation', 'auto'))} · "
                f"absorption={requested_pump_absorption}"
            ),
            effective=(
                f"propagation={effective['pump_propagation']} · "
                f"absorption={pump_absorption} · "
                f"depletion={effective['pump_depletion']}"
            ),
            status=status(
                True,
                (
                    "physics.pump_propagation",
                    "physics.pump_depletion",
                    "pump.absorption_transport",
                    "pump.profile",
                    "grid.pump_optical",
                ),
            ),
            location="① 模型 / ② 光束 → Pump",
        ),
        DependencyContract(
            module="Seed 场与传播",
            requested=signal_requested,
            effective=(
                f"{signal_model} · optical={effective['optical_propagation']} · "
                f"phase={nonlinear_phase} · loss={nonlinear_loss}"
            ),
            status=status(
                True,
                (
                    "simulation.signal_model",
                    "physics.optical_propagation",
                    "physics.nonlinear_phase_model",
                    "physics.nonlinear_loss_model",
                    "seed.joint_time_frequency",
                    "seed.spatial_modes",
                ),
            ),
            location="① 模型 / ② 光束",
        ),
        DependencyContract(
            module="Seed 空间增益与横模",
            requested=(
                f"gain={requested_spatial_gain} · modes={requested_spatial_modes}"
            ),
            effective=(
                f"gain={spatial_gain} · modes={spatial_modes} · "
                f"optical={effective['optical_propagation']}"
            ),
            status=status(
                spatial_gain != "local_field"
                or spatial_modes != "effective_m2",
                (
                    "seed.spatial_gain_transport",
                    "seed.spatial_modes",
                    "seed.spatio_spectral",
                    "physics.optical_propagation",
                    "physics.gain_saturation",
                ),
            ),
            location="② 光束 → Seed 增益输运 / 空间光束",
        ),
        DependencyContract(
            module="Seed 偏振与双折射",
            requested=(
                f"polarization={requested_polarization} · "
                f"birefringence={requested_birefringence}"
            ),
            effective=(
                f"polarization={polarization_model} · "
                f"birefringence={birefringence_model} · signal={signal_model}"
            ),
            status=status(
                polarization_model == "jones",
                (
                    "seed.polarization",
                    "simulation.signal_model",
                    "seed.spectrum",
                    "seed.spatial_gain_transport",
                    "material.mechanical",
                    "physics.thermal_model",
                    "physics.refractive_index_model",
                ),
            ),
            location="② 光束 → Seed → Jones 偏振",
        ),
        DependencyContract(
            module="多程命中与程间光路",
            requested=f"multipass={'on' if requested_multipass else 'off'}",
            effective=(
                f"multipass={'on' if multipass_active else 'off'} · "
                f"execution={execution} · "
                f"template={multipass.get('template', 'double_pass')} · hits={passes}"
            ),
            status=status(
                multipass_active,
                (
                    "simulation.multipass",
                    "seed.temporal",
                ),
            ),
            location="① 模型 → 启用多程 / ② 光束 → 多程模板",
        ),
        DependencyContract(
            module="周期稳态与布居",
            requested=(
                periodic_requested_method
                if periodic_requested_active
                else "off"
            ),
            effective=(
                f"periodic={periodic.get('method', 'pulse_map') if periodic_active else 'off'} · "
                f"population={resolve_population_dynamics(config)} · "
                f"gain_saturation={effective['gain_saturation']}"
            ),
            status=status(
                periodic_active,
                (
                    "simulation.periodic_steady_state",
                    "physics.population_dynamics",
                    "physics.gain_saturation",
                    "energy_levels.gain_override_m_inv",
                ),
            ),
            location="① 模型 → Pump + Seed 周期稳态",
        ),
        DependencyContract(
            module="热光闭环",
            requested=(
                f"{requested_coupling_model} · spectroscopy={response_mode} · "
                f"seed_heat={requested_seed_heat} · "
                f"heat={requested_heat_source}/{requested_heat_fraction}"
            ),
            effective=(
                f"coupling={coupling_model} · thermal={thermal_model} · "
                f"index={effective['refractive_index_model']} · "
                f"Pump={effective['pump_propagation']} · seed_heat={seed_heat} · "
                f"heat={heat_source}/{heat_fraction} · "
                f"passive_seed_heat={'on' if passive_seed_heat else 'off'}"
            ),
            status=status(
                thermal_active or coupling_model != "one_way",
                (
                    "multiphysics",
                    "physics.thermal_model",
                    "physics.refractive_index_model",
                    "physics.pump_propagation",
                    "material.thermal",
                    "energy_levels.event_energy_ledger",
                ),
            ),
            location="① 模型 → Pump · Seed · 热场收敛 / ③ 介质",
        ),
        DependencyContract(
            module="应力、光弹与端面",
            requested=(
                f"stress={requested_stress_model} · "
                f"photoelastic={requested_photoelastic} · "
                f"endface={requested_endface}"
            ),
            effective=(
                f"thermal={thermal_model} · stress={stress_model} · "
                f"photoelastic={photoelastic} · endface={endface}"
            ),
            status=status(
                mechanics_active,
                ("material.mechanical", "physics.thermal_model"),
            ),
            location="③ 介质 → 应力与光弹",
        ),
        DependencyContract(
            module="物理网格与光学网格",
            requested=(
                f"spacing={requested_value('grid.spacing_xy', grid.get('spacing_xy', 'uniform'))} · "
                "Seed optical="
                f"{'on' if requested_value('grid.optical.enabled', optical_grid.get('enabled', False)) else 'off'} · "
                "Pump grid="
                f"{requested_value('grid.pump_optical.mode', pump_grid.get('mode', 'shared'))}"
            ),
            effective=(
                f"{grid.get('nx')}×{grid.get('ny')}×{grid.get('nz')} · "
                f"Seed={effective['optical_propagation']} · Pump={effective['pump_propagation']}"
            ),
            status=status(True, ("grid",)),
            location="④ 数值",
        ),
        DependencyContract(
            module="ASE 与强场",
            requested=(
                f"ASE={requested_ase_model} · phase={requested_phase} · "
                f"loss={requested_loss}"
            ),
            effective=(
                f"ASE={ase_model} · phase={nonlinear_phase} · "
                f"loss={nonlinear_loss}"
                + (
                    f" · mechanism={strong_field.get('mechanism', 'unspecified')}"
                    f" · K={strong_field.get('multiphoton_order', 2)}"
                    f" · carrier={carrier_model}"
                    if nonlinear_loss != "off"
                    else ""
                )
            ),
            status=status(
                strong_active,
                (
                    "ase",
                    "material.strong_field",
                    "physics.nonlinear_phase_model",
                    "physics.nonlinear_loss_model",
                    "physics.population_dynamics",
                ),
            ),
            location="③ 介质 → ASE 与强场",
        ),
    )


def validate_available_backends(config: dict[str, Any]) -> None:
    """Reject explicitly unavailable compute backends before starting a UI job.

    Auto/NumPy remain untouched. This checks installation capability, including
    stored settings for optional branches, without silently changing the request.
    """
    import importlib
    paths = {
        'energy_levels.linear_solver.small_matrix_kernel': 'small_matrix_kernel',
        'seed.coherent.axisymmetric.radial_diffraction_kernel': 'axisymmetric_diffraction_kernel',
        'seed.coherent.transient_gain.single_transition_kernel': 'coherent_gain_kernel',
    }
    unavailable = []
    for path, module in paths.items():
        value: Any = config
        for key in path.split('.'):
            value = value.get(key, {}) if isinstance(value, dict) else {}
        if value == 'numba' and not importlib.import_module('.' + module, __package__).NUMBA_AVAILABLE:
            unavailable.append(path)
    if unavailable:
        raise ValueError('Numba is unavailable in this installation. Choose auto or numpy for: '
                         + ', '.join(unavailable))
