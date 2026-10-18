"""Presentation-only localization for the Streamlit application.

The simulation configuration, enum values, archive schema, and numerical
results deliberately stay language-neutral.  Only copies sent to the UI are
translated.  This separation is what lets a running session change language
without changing a model choice or stitching results from different states.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


ZH_CN = "zh-CN"
EN = "en"
SUPPORTED_LOCALES = (ZH_CN, EN)

_CATALOG_PATH = Path(__file__).with_name("data") / "ui_en.json"
_HAN_RE = re.compile(r"[\u3400-\u9fff]")

_ZH_SOURCE_MESSAGES: dict[str, str] = {
    "ABCD mode approximates the configured non-Gaussian source by an astigmatic Gaussian.": (
        "ABCD 模式会用像散高斯光束近似当前配置的非高斯光源。"
    ),
    "Temporal intensity has no phase column; zero temporal phase was assumed.": (
        "时间强度没有相位列；已采用零时间相位。"
    ),
    "Spectral intensity has no phase column; flat spectral phase was assumed.": (
        "光谱强度没有相位列；已采用平坦光谱相位。"
    ),
    (
        "Both intensities and a spectral phase were supplied; the measured "
        "spectral complex field is authoritative, and the temporal intensity "
        "is retained as a comparison target."
    ): (
        "同时提供了两域强度和光谱相位；以实测光谱复场为准，时间强度仅保留为比较目标。"
    ),
    (
        "Both intensities and a temporal phase were supplied; the measured "
        "temporal complex field is authoritative, and the spectral intensity "
        "is retained as a comparison target."
    ): (
        "同时提供了两域强度和时间相位；以实测时间复场为准，光谱强度仅保留为比较目标。"
    ),
    (
        "Both intensity domains lacked phase; an admissible phase was "
        "retrieved. Intensity-only phase retrieval is not unique and can "
        "retain sign/time-reversal ambiguities."
    ): (
        "时间和光谱强度都没有相位；已恢复一个可行相位。仅凭强度恢复相位并不唯一，"
        "仍可能存在符号或时间反演歧义。"
    ),
}


# Hand-curated, high-visibility copy and domain terminology.  The larger JSON
# catalog covers every source literal; these entries keep the primary workflow
# and Plotly annotations idiomatic and technically precise.
_EN_OVERRIDES: dict[str, str] = {
    "轴对称快速模型的角向保真度": (
        "Azimuthal Fidelity of the Axisymmetric Fast Model"
    ),
    "基础轴对称 — 仅 m=0": "Basic Axisymmetry — m=0 Only",
    "m=0+1 — 保留偏心、倾角与热棱镜": (
        "m=0+1 — Retain Decenter, Tilt, and Thermal Prism"
    ),
    "m=0+1+廉价 m=2 — 材料二阶项采用 r² 二次型": (
        "m=0+1+Cheap m=2 — Quadratic r² Material Closure"
    ),
    "m=0+1+完整 m=2 — 保留二阶项的完整径向分布": (
        "m=0+1+Full m=2 — Retain the Full Radial Quadrupole"
    ),
    "这是晶体内部的有限角向 Fourier 模型。m=1 是偶极项，可描述质心/角度漂移和热棱镜；m=2 是四极项，可描述椭圆化、像散及旋转二次型。Pump 仍使用所选 Pump 后端；它产生的布居和温度场会在进入 Seed 快速求解器时投影到这里选择的阶数。": (
        "This is a finite azimuthal Fourier model inside the crystal. m=1 is "
        "the dipole term for centroid/angle drift and thermal-prism effects; "
        "m=2 is the quadrupole term for ellipticity, astigmatism, and rotated "
        "quadratic forms. Pump still uses the selected Pump backend; its "
        "population and temperature fields are projected to the orders chosen "
        "here when they enter the fast Seed solver."
    ),
    "仅传播 E(f,r)；所有非零角向阶数都会被投影掉。适合严格居中且热场/布居近似圆对称的基准。": (
        "Propagate only E(f,r); every nonzero azimuthal order is projected out. "
        "Suitable as a baseline for a strictly centred beam with nearly radial "
        "thermal and population fields."
    ),
    "晶体内传播 m=-1,0,+1 的复场及对应材料分量；可保留偏心和热棱镜，但不会保留 m=2 椭圆化。": (
        "Propagate the m=-1,0,+1 complex fields and matching material "
        "components inside the crystal. Decenter and thermal-prism effects are "
        "retained, but m=2 ellipticity is not."
    ),
    "传播 m=-2…+2。材料场的 m=2(r) 会先最小二乘约化为 q·r²，光场自身产生的 m=2 仍按径向波动方程传播；这是受控的廉价像散闭合。": (
        "Propagate m=-2…+2. The material m=2(r) field is first reduced by "
        "least squares to q·r², while optically generated m=2 still follows the "
        "radial wave equation. This is a controlled cheap astigmatic closure."
    ),
    "传播 m=-2…+2，并保留温度、布居和增益的完整 m=2(r) 径向函数；整脉冲增益自动使用全部相干频率点，不进行频率分组；仍会舍弃 |m|>2，需检查结果里的 omitted-order residual。": (
        "Propagate m=-2…+2 and retain the full radial m=2(r) functions of "
        "temperature, population, and gain. Whole-pulse gain automatically "
        "uses every coherent frequency point without spectral grouping. "
        "Orders |m|>2 are still discarded; check the reported omitted-order "
        "residual."
    ),
    "CPU 径向/角向衍射加速": "CPU Radial/Azimuthal Diffraction Acceleration",
    "角向配点数": "Azimuthal Collocation Samples",
    "角向保真度": "Azimuthal Fidelity",
    "有限角向 Fourier × 半径 r": "Finite Azimuthal Fourier × Radius r",
    "基础轴对称 m=0": "Basic Axisymmetric m=0",
    "m=0+1+廉价 m=2": "m=0+1+Cheap m=2",
    "m=0+1+完整 m=2": "m=0+1+Full m=2",
    "局域热相位、增益和 Kerr 在这些 θ 配点上计算，再投影回保留阶数。下限按三次 Kerr 去混叠设置；提高它只改善角向伪谱乘法，不会增加保留的 m 阶数。": (
        "Evaluate local thermal phase, gain, and Kerr on these θ samples and "
        "project back to the retained orders. The lower bound de-aliases cubic "
        "Kerr products; increasing it improves only the angular pseudospectral "
        "multiplication and does not retain additional m orders."
    ),
    "晶体内按表单选择使用 E(f,r) 或有限阶 E(f,r,θ)，保留各阶径向衍射/自聚焦、完整相干频率网格、色散和局域时间域 Kerr；增益仍按 front → peak → tail 自适应软窗逐次耗尽，周期工作点使用完整 pulse map。未保留的更高方位角阶数会被投影掉，定量使用前必须通过对应门限。": (
        "Inside the crystal, use E(f,r) or finite-order E(f,r,θ) as selected in "
        "the form, retaining order-dependent radial diffraction/self-focusing, "
        "the complete coherent frequency grid, dispersion, and local time-domain "
        "Kerr. Gain is still depleted sequentially with adaptive front-to-peak-to-"
        "tail soft windows, and the periodic operating point uses the complete "
        "pulse map. Unretained higher azimuthal orders are projected out; their "
        "applicability gates must pass before quantitative use."
    ),
    "当前阶数不保留 x/y 焦点差异": (
        "The current order does not retain an x/y focus difference"
    ),
    "m=0 不保留 Seed 偏心（请选择 m=0+1）": (
        "m=0 does not retain Seed decenter (choose m=0+1)"
    ),
    "m=0 不保留 Seed 入射倾角（请选择 m=0+1）": (
        "m=0 does not retain Seed incidence tilt (choose m=0+1)"
    ),
    "当前阶数不保留 x/y ": (
        "The current order does not retain the x/y "
    ),
    "差异": " difference",
    "L4.5 基础轴对称瞬态 Kerr 已启用：晶体内传播 E(f,r)，保留径向衍射、色散、局域时间域 Kerr，以及 front → peak → tail 软窗增益；不保留非零方位角模式，因此必须检查结果中的投影、边界和材料场残差门限。": (
        "L4.5 basic axisymmetric transient Kerr is active. The crystal "
        "propagates E(f,r), retaining radial diffraction, dispersion, local "
        "time-domain Kerr, and front-to-peak-to-tail soft-window gain. Nonzero "
        "azimuthal modes are omitted, so the projection, boundary, and material-"
        "field residual gates must pass."
    ),
    "L4.5 有限角向瞬态 Kerr 已启用：晶体内传播 E(f,r,θ)，径向衍射会按每个保留的 m 阶分别计算；色散、Kerr 和 front → peak → tail 软窗增益也作用于这些角向自由度。它仍不是完整 x-y 场，必须检查 omitted-order residual、入口投影和径向边界门限。": (
        "L4.5 finite-azimuthal transient Kerr is active. The crystal propagates "
        "E(f,r,θ), evaluating radial diffraction separately for each retained m "
        "order; dispersion, Kerr, and front-to-peak-to-tail soft-window gain act "
        "on those angular degrees of freedom. This is still not a full x-y field, "
        "so the omitted-order residual, entrance-projection, and radial-boundary "
        "gates must pass."
    ),
    "最大保留 |m|": "Maximum Retained |m|",
    "角向配点": "Azimuthal Samples",
    "入口投影损失": "Entrance Projection Loss",
    "传播中新生高阶": "Higher Orders Generated During Propagation",
    "材料场高阶残差": "Material Higher-Order Residual",
    "廉价 m=2 拟合残差": "Cheap m=2 Fit Residual",
    "布居非负重分配": "Non-Negative Population Redistribution",
    "布居可实现修正": "Population Realizability Correction",
    "这六项越接近零，所选角向闭合越可信；可实现修正表示把有限阶布居变为非负状态所移动的离子比例，非负重分配表示回写 x-y 网格时重新安排的跃迁比例。若任一项越过验证门限，应升级到下一档，必要时使用完整 x-y。": (
        "All six values should approach zero for a trustworthy angular closure. "
        "Realizability correction is the ion fraction moved to make the finite-order "
        "population non-negative; non-negative redistribution is the transition "
        "fraction rearranged when mapping back to x-y. Upgrade the retained order, "
        "or use full x-y, when any value exceeds its validation gate."
    ),
    "这四项越接近零，所选角向闭合越可信；若入口、传播中新生高阶或材料残差越过验证门限，应升级到下一档，必要时使用完整 x-y。": (
        "The closer these four values are to zero, the more reliable the "
        "selected azimuthal closure. If the entrance, dynamically generated "
        "higher-order, or material residual exceeds its validation gate, "
        "move to the next level or use full x-y when necessary."
    ),
    "这五项越接近零，所选角向闭合越可信；非负重分配表示 Fourier 重建后为保持布居物理性而重新安排的跃迁比例。若任一项越过验证门限，应升级到下一档，必要时使用完整 x-y。": (
        "The closer these five values are to zero, the more reliable the "
        "selected azimuthal closure. Non-negative redistribution is the "
        "fraction of transitions rearranged after Fourier reconstruction to "
        "keep populations physical. If any value exceeds its validation "
        "gate, move to the next level or use full x-y."
    ),
    "当前是 L4.5 基础轴对称瞬态 Kerr：求解 E(f,r) 与完整相干时间波形，并保留 front → peak → tail 软窗增益；所有非零方位角空间模式仍被省略。": (
        "This is L4.5 basic axisymmetric transient Kerr: it solves E(f,r) and "
        "the complete coherent temporal waveform while retaining front-to-peak-"
        "to-tail soft-window gain; every nonzero azimuthal spatial mode is still "
        "omitted."
    ),
    "当前是 L4.5 有限角向瞬态 Kerr：求解 E(f,r,θ)，保留 m=-": (
        "This is L4.5 finite-azimuthal transient Kerr: it solves E(f,r,θ), "
        "retaining m=-"
    ),
    " 与完整相干时间波形，front → peak → tail 软窗增益也被保留；更高角向阶数仍被省略。": (
        " together with the complete coherent temporal waveform and front-to-"
        "peak-to-tail soft-window gain; higher azimuthal orders are still omitted."
    ),
    "L4 基础轴对称整脉冲 Kerr 已启用：晶体内传播 E(f,r)，保留径向衍射、色散和局域时间域 Kerr；增益使用整脉冲 FN 闭合，非零方位角模式被省略。": (
        "L4 basic axisymmetric whole-pulse Kerr is active. The crystal "
        "propagates E(f,r), retaining radial diffraction, dispersion, and local "
        "time-domain Kerr; gain uses a whole-pulse FN closure and nonzero "
        "azimuthal modes are omitted."
    ),
    "L4 有限角向整脉冲 Kerr 已启用：晶体内传播 E(f,r,θ)，径向衍射会按每个保留的 m 阶分别计算，色散和 Kerr 也作用于这些角向自由度；增益使用整脉冲 FN 闭合。它仍不是完整 x-y 场，必须检查所有降阶门限。": (
        "L4 finite-azimuthal whole-pulse Kerr is active. The crystal propagates "
        "E(f,r,θ), evaluating radial diffraction separately for every retained m "
        "order; dispersion and Kerr also act on those angular degrees of freedom, "
        "while gain uses a whole-pulse FN closure. This is still not a full x-y "
        "field, so every reduction gate must pass."
    ),
    "瞬态软窗增益使用原生相干频率网格；整脉冲 FN 档使用配置的受控频率分组并在细网格重建增益。": (
        "Transient soft-window gain uses the native coherent frequency grid; the "
        "whole-pulse FN level uses the configured controlled frequency groups and "
        "reconstructs gain on the fine grid."
    ),
    "当前是 L4 基础轴对称整脉冲 Kerr：求解 E(f,r) 与完整相干时间波形，增益采用整脉冲 FN 闭合；所有非零方位角空间模式仍被省略。": (
        "This is L4 basic axisymmetric whole-pulse Kerr: it solves E(f,r) and the "
        "complete coherent temporal waveform with a whole-pulse FN gain closure; "
        "every nonzero azimuthal spatial mode is still omitted."
    ),
    "当前是 L4 有限角向整脉冲 Kerr：求解 E(f,r,θ)，保留 m=-": (
        "This is L4 finite-azimuthal whole-pulse Kerr: it solves E(f,r,θ), "
        "retaining m=-"
    ),
    " 与完整相干时间波形，增益采用整脉冲 FN 闭合；更高角向阶数仍被省略。": (
        " together with the complete coherent temporal waveform and a whole-pulse "
        "FN gain closure; higher azimuthal orders are still omitted."
    ),
    "##### Seed 热相位空间闭合": "##### Seed Thermal-Phase Spatial Closure",
    "sampled_phase — 保留采样热相位（默认）": (
        "sampled_phase — Retain the Sampled Thermal Phase (Default)"
    ),
    "local_parabolic — 局部抛物线拟合窗口": (
        "local_parabolic — Local Parabolic Fit Window"
    ),
    "sampled-field 热相位闭合": "Sampled-Field Thermal-Phase Closure",
    "直接使用每个网格层的三维热折射率相位，保留采样到的非抛物像差结构。": (
        "Use each grid slice's three-dimensional thermo-refractive phase "
        "directly, retaining the sampled non-parabolic aberration structure."
    ),
    "只在指定椭圆邻域内拟合局部二次热透镜，再用拟合相位传播；适合与 Gaussian/Commercial App 式局部透镜近似对齐。": (
        "Fit a local quadratic thermal lens only inside the configured "
        "elliptical neighbourhood, then propagate with the fitted phase. This "
        "is suitable for alignment with Gaussian/Commercial App-style local-lens "
        "approximations."
    ),
    "sampled_phase 直接使用网格热相位；local_parabolic 在每个 z 层的指定椭圆邻域拟合二次热透镜，再把拟合后的相位用于 sampled-field Seed 传播。Commercial App Fast 使用后端专属热透镜设置。": (
        "sampled_phase uses the gridded thermal phase directly; "
        "local_parabolic fits a quadratic thermal lens inside the configured "
        "elliptical neighbourhood in each z slice, then uses the fitted phase "
        "for sampled-field Seed propagation. Commercial App Fast has its own backend-"
        "specific thermal-lens settings."
    ),
    "axis_aligned — x/y 像散二次项": (
        "axis_aligned — Astigmatic x/y Quadratic Terms"
    ),
    "full_xy — 包含 xy 交叉二次项": "full_xy — Include the xy Cross Term",
    "局部热透镜二次模型": "Local Thermal-Lens Quadratic Model",
    "axis_aligned 舍弃旋转像散的 xy 交叉曲率；full_xy 保留完整二维二次型。若交叉项不可忽略，不应使用轴对齐降阶。": (
        "axis_aligned discards the xy cross-curvature of rotated astigmatism; "
        "full_xy retains the complete two-dimensional quadratic form. Do not "
        "use the axis-aligned reduction when the cross term is significant."
    ),
    "只保留 x² 与 y² 曲率，明确舍弃 xy 交叉项；适合主轴已与网格对齐的像散热透镜。": (
        "Retain only the x² and y² curvatures and explicitly discard the xy "
        "cross term; suitable for an astigmatic thermal lens whose principal "
        "axes align with the grid."
    ),
    "保留 x²、xy 与 y² 完整二维二次型，可表示相对于网格旋转的局部像散主轴。": (
        "Retain the complete x², xy, and y² quadratic form, representing local "
        "astigmatic principal axes rotated relative to the grid."
    ),
    "把 Commercial App 粗增益回灌箱铺满相干 FFT 频率范围，优先保留由 Kerr 新产生的边带。": (
        "Distribute Commercial App's coarse gain-feedback bins across the coherent FFT "
        "frequency span, prioritizing retention of Kerr-generated sidebands."
    ),
    "只在输入 Seed 的有效能量支撑内布置回灌箱，分辨率较集中，但必须检查新生边带的支撑外泄漏。": (
        "Place feedback bins only within the input Seed's effective energy "
        "support for concentrated resolution, while checking new sidebands "
        "for out-of-support leakage."
    ),
    "sampled-field 热拟合半径 x (m)": "Sampled-Field Thermal-Fit Radius x (m)",
    "sampled-field 热拟合半径 y (m)": "Sampled-Field Thermal-Fit Radius y (m)",
    "手动指定 sampled-field 热拟合中心": (
        "Set the Sampled-Field Thermal-Fit Centre Manually"
    ),
    "关闭时逐层使用当前光束中心；开启时使用固定物理坐标。": (
        "When disabled, follow the current beam centre in each slice; when "
        "enabled, use fixed physical coordinates."
    ),
    "sampled-field 热拟合中心 x (m)": "Sampled-Field Thermal-Fit Centre x (m)",
    "sampled-field 热拟合中心 y (m)": "Sampled-Field Thermal-Fit Centre y (m)",
    "##### 1. 时间与模态": "##### 1. Temporal and Modal Closure",
    "##### 2. 热相位与光谱反馈": "##### 2. Thermal Phase and Spectral Feedback",
    "完整相干频率范围（推荐，保留新边带）": (
        "Full Coherent Frequency Span (Recommended; Retains New Sidebands)"
    ),
    "仅输入 Seed 能量支撑（更快，须检查泄漏）": (
        "Input Seed Energy Support Only (Faster; Check Leakage)"
    ),
    "Commercial App 频谱回灌网格范围": "Commercial App Spectral-Feedback Grid Span",
    "完整范围可接住 Kerr 新产生的边带；输入能量支撑只覆盖源频谱的有效范围，分箱更集中但可能漏掉新边带，必须核对频谱泄漏诊断。": (
        "The full span retains Kerr-generated sidebands. Input energy support "
        "covers only the source spectrum's effective span, concentrating the "
        "bins but potentially missing new sidebands; inspect the spectral-"
        "leakage diagnostics."
    ),
    "分箱覆盖上面选定的频率范围。默认 65 仍远少于完整 x-y-t coherent CPA；请对分箱数做收敛检查。": (
        "The bins cover the frequency span selected above. The default 65 "
        "bins remain far fewer than full x-y-t coherent CPA; perform a bin-"
        "count convergence check."
    ),
    "##### 3. 跨脉冲载流子固定点": (
        "##### 3. Inter-Pulse Free-Carrier Fixed Point"
    ),
    "Commercial App 载流子周期定态相对容差": (
        "Commercial App Free-Carrier Periodic-Steady-State Relative Tolerance"
    ),
    "控制每个局部 effective-area 载流子标量在相邻脉冲之间的周期固定点收敛；它不同于 coherent CPA 的全局三维载流子收敛。": (
        "Controls periodic fixed-point convergence of each local effective-"
        "area carrier scalar between pulses. This differs from coherent CPA's "
        "global three-dimensional carrier convergence."
    ),
    "Commercial App 载流子周期定态最大迭代": (
        "Commercial App Free-Carrier Periodic-Steady-State Maximum Iterations"
    ),
    "每次迭代都包含一个脉冲的生成/FCA 更新与脉冲间暗时间衰减；达到上限仍不收敛时会按运行合同报错。": (
        "Each iteration includes one pulse's generation/FCA update and dark-"
        "time decay between pulses. The run contract reports an error if it "
        "still has not converged at the limit."
    ),
    "#### A. 功率与主动吸收": "#### A. Power and Active Absorption",
    "#### B. 空间光源与传播": "#### B. Spatial Source and Propagation",
    "#### C. 时间波形与光谱": "#### C. Temporal Waveform and Spectrum",
    "#### D. Pump 独立光路与半径探针": (
        "#### D. Independent Pump Path and Radius Probe"
    ),
    "角分布超高斯完整指数 x（高斯=2）": (
        "Angular Super-Gaussian Full Exponent x (Gaussian = 2)"
    ),
    "角分布超高斯完整指数 y（高斯=2）": (
        "Angular Super-Gaussian Full Exponent y (Gaussian = 2)"
    ),
    "E. 输入复光谱、展宽与 CPA": "E. Input Complex Spectrum, Stretching, and CPA",
    "F. Seed 独立光路与半径探针": "F. Independent Seed Path and Radius Probe",
    "记录 β 测量条件与相对不确定度": (
        "Record β Measurement Conditions and Relative Uncertainty"
    ),
    "定量 β 应同时记录对应波长、脉宽和相对不确定度。host_interband 还必须用参考波长核对最低多光子阶数，因此不能关闭。": (
        "A quantitative β should record its wavelength, pulse duration, and "
        "relative uncertainty. host_interband also needs the reference "
        "wavelength to verify the minimum multiphoton order, so this cannot "
        "be disabled."
    ),
    "β 参考波长 (m)": "β Reference Wavelength (m)",
    "β 参考脉宽 (s)": "β Reference Pulse Duration (s)",
    "β 相对不确定度": "β Relative Uncertainty",
    "例如 0.15 表示相对标准不确定度 15%。": (
        "For example, 0.15 means a 15% relative standard uncertainty."
    ),
    "解析 Seed 源光谱采样数": "Analytic Seed Source-Spectrum Sample Count",
    "用于离散解析 Gaussian/Lorentzian/super-Gaussian 源光谱；它与后面的输运求解重分箱、Commercial App 回灌分箱和 coherent CPA FFT 采样分别控制不同阶段。": (
        "Discretizes an analytic Gaussian, Lorentzian, or super-Gaussian "
        "source spectrum. The later transport rebinning, Commercial App feedback bins, "
        "and coherent CPA FFT samples control separate stages."
    ),
    "解析 Seed 源光谱半跨度 (×FWHM)": (
        "Analytic Seed Source-Spectrum Half-Span (× FWHM)"
    ),
    "在中心频率或中心波长两侧各覆盖多少个 FWHM；增加跨度时也要同步增加采样数，避免固定分箱被摊薄。": (
        "Sets how many FWHM are covered on each side of the centre frequency "
        "or wavelength. Increase the sample count with the span to avoid "
        "diluting a fixed number of bins."
    ),
    "parabolic_gain_fit — Commercial App 式局域抛物线增益拟合（复场 BPM）": (
        "parabolic_gain_fit — Commercial App-Style Local Parabolic Gain Fit "
        "(Complex-Field BPM)"
    ),
    "在指定横向椭圆内把真实三维增益逐层拟合为轴向二次分布，再作用于复场；它复现 Commercial App Gauss-BPM 的增益导管近似，不改动泵浦、布居或光谱物理参数。": (
        "Fit the physical 3-D gain in each axial slice to an axis-aligned "
        "quadratic inside the configured transverse ellipse, then apply it "
        "to the complex field. This reproduces Commercial App Gauss-BPM's gain-duct "
        "closure without changing Pump, population, or spectroscopic inputs."
    ),
    "local_field 对每个 x/y 单元施加自己的增益，会让光束被增益分布重塑；gaussian_mode_overlap 每个 z 单元只用强度加权的标量增益放大整束，用于明确关闭 gain guiding 的受约束 Gaussian 模态比较；parabolic_gain_fit 在指定椭圆内逐层拟合二次增益导管，复现 Commercial App Gauss-BPM 的空间闭合。": (
        "local_field applies each x/y cell's gain to the field and permits "
        "gain-guided reshaping; gaussian_mode_overlap applies one "
        "intensity-weighted scalar gain per z cell for a constrained Gaussian "
        "comparison with gain guiding disabled; parabolic_gain_fit fits a "
        "quadratic gain duct in each z slice inside the configured ellipse to "
        "reproduce Commercial App Gauss-BPM's spatial closure."
    ),
    "增益拟合半径 x (m)": "Gain-Fit Radius x (m)",
    "增益拟合半径 y (m)": "Gain-Fit Radius y (m)",
    "增益拟合中心 x (m)": "Gain-Fit Centre x (m)",
    "增益拟合中心 y (m)": "Gain-Fit Centre y (m)",
    "拟合增益下限 (m⁻¹)": "Fitted-Gain Lower Bound (m⁻¹)",
    "只在拟合椭圆内钳制二次增益；椭圆外仍使用原始局部增益。Nd:YAG 四能级主动增益通常取 0；存在真实再吸收时可保留负值。": (
        "Clamp the quadratic gain only inside the fit ellipse; retain the "
        "sampled local gain outside it. Zero is typical for four-level Nd:YAG "
        "active gain, while a negative value may retain physical reabsorption."
    ),
    "保存每个 z 单元的抛物线增益拟合诊断": (
        "Save Parabolic Gain-Fit Diagnostics for Every z Cell"
    ),
    "只保存逐层二次拟合系数和残差诊断，不改变传播场或任何物理输入。": (
        "Save only the per-slice quadratic coefficients and residual "
        "diagnostics; this does not change the propagated field or any "
        "physical input."
    ),
    "拟合只改变 BPM 使用的空间增益表示；泵浦、能级布居、截面、寿命和输入光束参数保持原值。": (
        "The fit changes only the spatial gain representation used by BPM; "
        "Pump, level populations, cross-sections, lifetimes, and input-beam "
        "parameters retain their original values."
    ),
    "当前 CW Seed 使用周期平均功率传播：每个 z 单元按局部线性指数增益推进，并把平均光子流反馈给 Pump + Seed 布居固定点；不会调用单脉冲 FN 耗尽。": (
        "The active CW Seed uses cycle-averaged power transport: each z cell "
        "advances with local linear exponential gain and feeds the average "
        "photon flux back into the Pump + Seed population fixed point. It does "
        "not invoke single-pulse Frantz–Nodvik depletion."
    ),
    "Pump 主动吸收输运": "Active Pump-Absorption Transport",
    "population_dependent — 由局部能级布居与截面求净吸收": (
        "population_dependent — Derive Net Absorption from Local Populations "
        "and Cross-Sections"
    ),
    "fixed_active_coefficient — 固定主动吸收系数（Commercial App 非饱和 Pump）": (
        "fixed_active_coefficient — Fixed Active-Absorption Coefficient "
        "(Commercial App Unsaturated Pump)"
    ),
    "diode 发光面半径 x (m)": "Diode Emitting-Surface Radius x (m)",
    "diode 发光面半径 y (m)": "Diode Emitting-Surface Radius y (m)",
    "光源侧介质折射率": "Source-Side Medium Refractive Index",
    "径向超高斯完整指数 x（高斯=2）": (
        "Radial Super-Gaussian Full Exponent x (Gaussian = 2)"
    ),
    "径向超高斯完整指数 y（高斯=2）": (
        "Radial Super-Gaussian Full Exponent y (Gaussian = 2)"
    ),
    "#### 随机光线求解": "#### Stochastic Ray Solver",
    "随机光线求解": "Stochastic Ray Solver",
    "右侧未运行的实时光路仍显示确定性的 D4σ/M² 包络，便于连续拖动元件；它不是随机光线结果。运行后保存的 Pump 三维场、入口/出口面和功率账本才来自 stochastic_raytrace。": (
        "The live, pre-run optical-path view on the right still shows a deterministic "
        "D4σ/M² envelope so elements can be adjusted continuously; it is not a "
        "stochastic-ray result. The saved 3-D Pump field, entrance/exit planes, "
        "and power ledger come from stochastic_raytrace only after a run."
    ),
    "光线数": "Ray Count",
    "随机种子": "Random Seed",
    "角分布": "Angular Distribution",
    "auto — diode 用恒定辐亮度 NA；束腰输入用广义 Gaussian": (
        "auto — Constant-radiance NA for a diode; generalized Gaussian for a waist input"
    ),
    "uniform_na — 方向余弦椭圆均匀（平面恒定辐亮度）": (
        "uniform_na — Uniform direction-cosine ellipse (constant planar radiance)"
    ),
    "uniform_solid_angle — NA 支撑内均匀立体角": (
        "uniform_solid_angle — Uniform solid angle within the NA support"
    ),
    "generalized_gaussian — 广义 Gaussian 角分布": (
        "generalized_gaussian — Generalized-Gaussian angular distribution"
    ),
    "射线到晶体网格的沉积": "Ray-to-Crystal-Grid Deposition",
    "nearest — 最近单元路径长度（旧结果兼容）": (
        "nearest — Nearest-cell track length (legacy-result compatible)"
    ),
    "bilinear — 双线性守恒路径沉积（低噪声对照）": (
        "bilinear — Conservative bilinear track deposition (lower-noise comparison)"
    ),
    "两者都严格关闭功率账本。bilinear 同时插值局部吸收和热折射率梯度，并把每段能量守恒地分配到相邻有效晶体单元。": (
        "Both choices close the power ledger exactly. Bilinear deposition also "
        "interpolates local absorption and the thermo-refractive-index gradient, "
        "then distributes each segment's energy conservatively among adjacent "
        "active crystal cells."
    ),
    "每子步最大横向跨格比例": (
        "Maximum Transverse Cell-Crossing Fraction per Substep"
    ),
    "越小越能解析高 NA 斜射线穿越网格，但运行更慢。": (
        "Smaller values better resolve high-NA oblique rays crossing the grid, "
        "but run more slowly."
    ),
    "每个 z 单元最多子步": "Maximum Substeps per z Cell",
    "停止追踪的相对光线功率": "Relative Ray-Power Cutoff",
    "Pump 发光面中心 x (m)": "Pump Emitting-Surface Center x (m)",
    "Pump 发光面中心 y (m)": "Pump Emitting-Surface Center y (m)",
    "Pump 腰斑/焦散中心 x (m)": "Pump Waist/Caustic Center x (m)",
    "Pump 腰斑/焦散中心 y (m)": "Pump Waist/Caustic Center y (m)",
    "全宽 x / y": "Full Width x / y",
    "最大发散半角 x / y": "Maximum Divergence Half-Angle x / y",
    "推导的有效 M² x / y": "Derived Effective M² x / y",
    "当前传播表示": "Current Propagation Representation",
    "随机相空间光线（直接使用源面尺寸与角分布）": (
        "Stochastic Phase-Space Rays (Direct Source-Plane Size and Angular Distribution)"
    ),
    "自相似超高斯强度包络（SG 固定，束宽按 ABCD 传播）": (
        "Self-Similar Super-Gaussian Intensity Envelope (Fixed SG Shape; ABCD Beam Width)"
    ),
    "二阶矩 M² 场代理": "Second-Moment M² Field Surrogate",
    "#### Pump 独立光路布置": "#### Independent Pump Optical-Path Layout",
    "只影响 pump；这里插入的透镜、距离和光阑不会进入 seed 光路。": (
        "Affects Pump only; lenses, distances, and apertures inserted here do not "
        "enter the Seed optical path."
    ),
    "距离按行定义：第一行是 LASER→首元件，之后每一行是上一元件→当前元件；GAIN 行因此就是上一元件（常见为透镜）→晶体入射面。Pump / Seed 两套数值彼此独立。": (
        "Distances are row-based: the first row is LASER→first element, and each "
        "later row is previous element→current element. The GAIN row is therefore "
        "the distance from the previous element (often a lens) to the crystal "
        "entrance face. Pump and Seed use independent values."
    ),
    "📏 Pump 坐标半径探针": "📏 Pump Coordinate/Radius Probe",
    " 三维光路暂不可用：": " 3-D optical path unavailable: ",
    " 已修改；下面仍是上一次完整配置快照的结果。冷态光路预览会实时更新，但布居、热透镜、增益和运行后 D4σ 不会局部拼接更新。请重新点击“运行当前配置”。": (
        " The configuration has changed; the results below still belong to the "
        "previous complete configuration snapshot. The cold-state optical-path "
        "preview updates live, but population, thermal lens, gain, and post-run "
        "D4σ results are never spliced from mixed configurations. Click Run Current "
        "Configuration again."
    ),
    "D4σ 半径 (µm)": "D4σ Radius (µm)",
    "D4σ 边界：x [": "D4σ Boundary: x [",
    "Pump 发光面": "Pump Emitting Surface",
    "Pump 晶体功率账本相对残差": (
        "Pump Crystal Power-Ledger Relative Residual"
    ),
    "hybrid_abcd_fft 用解析 Gaussian 保存晶体外功率和入口波前，再在晶体内用 FFT 响应三维热折射率；fft 从光源平面开始采样，必须保证整个外部光路的窗口都足够大。stochastic_raytrace 逐条传播非相干几何光线，并按晶体内实际路径沉积吸收功率；super_gaussian_abcd 保持 SG 强度形状，用一阶光学传播束宽。": (
        "hybrid_abcd_fft preserves the external-crystal power and entrance wavefront "
        "with an analytic Gaussian, then uses FFT propagation inside the crystal to "
        "respond to the 3-D thermo-refractive index. fft samples from the source plane, "
        "so its numerical window must contain the entire external optical path. "
        "stochastic_raytrace propagates incoherent geometric rays individually and "
        "deposits absorbed power along their physical paths in the crystal. "
        "super_gaussian_abcd preserves the SG intensity shape while first-order optics "
        "propagates its beam widths."
    ),
    "例如自由空间、薄/厚透镜、镜面、光阑、coordinate_break、loss。": (
        "For example: free space, thin/thick lenses, mirrors, apertures, "
        "coordinate_break, and loss."
    ),
    "保存的晶体 mask 没有覆盖任何有限体素；统计与切片暂时回退到全部有限体素。": (
        "The saved crystal mask contains no finite voxel; statistics and slices "
        "temporarily fall back to all finite voxels."
    ),
    "决定每条 Pump 光线的吸收能量如何守恒分配到三维体素。": (
        "Controls how each Pump ray's absorbed energy is distributed conservatively "
        "among 3-D voxels."
    ),
    "把射线段能量归入最近体素；速度快但会增加沉积噪声和网格依赖。": (
        "Assign each ray segment's energy to the nearest voxel. This is fast, but "
        "increases deposition noise and grid dependence."
    ),
    "把每条光线在体素内的真实路径长度作为沉积权重，守恒分配吸收能量。": (
        "Use each ray's physical track length inside a voxel as the deposition "
        "weight and distribute absorbed energy conservatively."
    ),
    "把输入离子密度解释为掺杂轮廓的峰值；其他体素按相对轮廓降低，不再守恒全体积平均值。": (
        "Interpret the entered ion density as the peak of the doping profile; other "
        "voxels scale with the relative profile, without preserving a whole-volume average."
    ),
    "#### Seed 脉冲列、强度文件与相位": (
        "#### Seed Pulse, Intensity Files, and Phase"
    ),
    "光谱由时间复场FFT生成；两列时间文件采用零时间相位，第三列可直接给phase(rad)。": (
        "The spectrum is generated by FFT of the complex temporal field. A two-column "
        "temporal file assumes zero temporal phase; an optional third column may provide "
        "phase (rad) directly."
    ),
    "时间波形由复光谱IFFT生成；两列光谱文件采用平坦光谱相位，第三列可直接给phase(rad)。": (
        "The temporal waveform is generated by IFFT of the complex spectrum. A two-column "
        "spectral file assumes flat spectral phase; an optional third column may provide "
        "phase (rad) directly."
    ),
    "intensity 会自动归一化到单脉冲总能量；可选第三列是时间相位（rad）。": (
        "Intensity is normalized automatically to the total single-pulse energy; "
        "the optional third column is temporal phase (rad)."
    ),
    "relative 模式自动归一化；可选第三列是光谱相位（rad）。": (
        "Relative mode is normalized automatically; the optional third column is "
        "spectral phase (rad)."
    ),
    "也可以直接把phase(rad)放在时间/光谱强度文件第三列。独立相位文件用于只有两列强度文件的情况。": (
        "phase (rad) may also be placed directly in the third column of a temporal "
        "or spectral intensity file. Use the separate phase file when the intensity "
        "file has only two columns."
    ),
    "当前强度展宽倍率 ": "Current Intensity-Broadening Factor ",
    "×；频谱强度 FWHM ": "×; Spectral-Intensity FWHM ",
    "×; 频谱强度 FWHM ": "×; Spectral-Intensity FWHM ",
    "频谱强度 FWHM ": "Spectral-Intensity FWHM ",
    " nm（纯相位色散前后不变）。": (
        " nm (unchanged by phase-only dispersion)."
    ),
    "时域图中的两条强度曲线分别按各自峰值归一化，用于直接比较波形和展宽；归一化不改变上面的 FWHM。": (
        "The two temporal-intensity curves are normalized to their own peaks for a "
        "direct comparison of waveform and broadening; this display normalization "
        "does not change the FWHM values above."
    ),
    "通常不需要。上面的强度文件第三列或独立相位文件已可输入相位；这里保留 real/imag 复场兼容入口。": (
        "Usually unnecessary: the third column of the intensity file or the separate "
        "phase file already accepts phase. This real/imag complex-field input remains "
        "only as a compatibility path."
    ),
    "多色/FN、JOSA 与 coherent CPA 使用各自的局域空间增益算子。": (
        "Spectral-time/FN, JOSA, and coherent CPA each use their own local "
        "spatial-gain operator."
    ),
    "当前为多色×时间分箱模型；每个时间/波长分箱使用 FN。": (
        "The active model uses spectral × temporal energy bins; each time/wavelength "
        "bin uses a Frantz–Nodvik gain update."
    ),
    "标量模式保持原有计算路径，不创建经验性退偏损耗。": (
        "Scalar mode retains the original propagation path and does not invent an "
        "empirical depolarization loss."
    ),
    "只影响 seed；pump 的透镜和光阑不会作用到 seed。自建多程时允许重复插入 gain_medium。": (
        "Affects Seed only; Pump lenses and apertures do not act on the Seed path. "
        "Repeated gain_medium elements are allowed when constructing a custom multipass path."
    ),
    "📏 Seed 坐标半径探针": "📏 Seed Coordinate/Radius Probe",
    "快速单向模型：只求一次 Pump/布居和热场，再把 Δn 作用到最终 Seed。": (
        "Fast one-way model: solve Pump/population and the thermal field once, "
        "then apply Δn to the final Seed propagation."
    ),
    "自洽模式反复传播 Pump、求布居、热场和 Δn，直到 Pump 吸收、布居与热相位共同收敛；局部光谱反馈、ASE/TPL 和 Seed 强场产热是否进入闭环由下面各自的开关决定。": (
        "The self-consistent mode repeatedly propagates Pump and solves population, "
        "thermal field, and Δn until Pump absorption, population, and thermal phase "
        "converge together. The switches below determine whether local spectroscopy, "
        "ASE/TPL, and Seed strong-field heating enter the closed loop."
    ),
    "最少耦合迭代": "Minimum Coupling Iterations",
    "耦合相对容差": "Coupling Relative Tolerance",
    "最多耦合迭代": "Maximum Coupling Iterations",
    "Δn 松弛系数": "Δn Relaxation Factor",
    "自动加速热光固定点": "Accelerate Thermo-Optic Fixed Point Automatically",
    "对 Δn 和局部光谱温度状态使用带保护的 Anderson 加速；内部周期稳态同时复用上一轮布居作为初值。": (
        "Use safeguarded Anderson acceleration for Δn and local-spectroscopy "
        "temperature states; an inner periodic solve also reuses the previous "
        "population as its initial state."
    ),
    "最多迭代只是上限，达到容差会提前停止。Cr:ZnS 热光算例通常需要约 15–20 次；最终 converged 同时要求内层布居与热求解均收敛，并让 Δn、Pump 吸收、布居、温度以及已启用的局部光谱、Seed 产热/输出、张量光弹和端面 sag 残差全部达标。若内层使用周期脉冲图，其载流子状态也必须先按该图的合同闭合。": (
        "The maximum iteration count is only a cap; the solve stops early after "
        "meeting tolerance. Cr:ZnS thermo-optic cases often need about 15–20 "
        "iterations. Final converged requires both inner population and thermal "
        "solves to converge, plus all active residuals for Δn, Pump absorption, "
        "population, temperature, local spectroscopy, Seed heat/output, tensor "
        "photoelasticity, and end-face sag to pass. If the inner solve uses a "
        "periodic pulse map, its carrier state must also close under that map's contract."
    ),
    "当前 Seed 重复频率：": "Current Seed Repetition Rate: ",
    "Pump + Seed 稳态方法": "Pump + Seed Steady-State Method",
    "周期平均代数稳态 — 快速，推荐高重复频率": (
        "Cycle-Averaged Algebraic Steady State — Fast; Recommended at High Repetition Rate"
    ),
    "逐脉冲周期固定点 — 严格，强单脉冲耗尽": (
        "Pulse-by-Pulse Periodic Fixed Point — Strict; Resolves Strong Single-Pulse Depletion"
    ),
    "自动固定点加速": "Accelerate Fixed Point Automatically",
    "最少周期数": "Minimum Number of Cycles",
    "连续满足次数": "Required Consecutive Converged Cycles",
    "最多周期数": "Maximum Number of Cycles",
    "相对收敛容差": "Relative Convergence Tolerance",
    "最少自洽迭代": "Minimum Self-Consistent Iterations",
    "最多自洽迭代": "Maximum Self-Consistent Iterations",
    "稳态松弛系数": "Steady-State Relaxation Factor",
    "内置模型实时求解每个列出的布居与固定级联寿命，并非把四能级折成二能级；分支跃迁、上转换/交叉弛豫和任意网络请用 custom。": (
        "The built-in model evolves every listed population with its explicit "
        "cascade lifetime; it does not collapse a four-level system into two levels. "
        "Use custom for branching transitions, upconversion/cross-relaxation, or an "
        "arbitrary level network."
    ),
    "材料名称": "Material Name",
    "Pump 分量账本：离子主动吸收按所选量子亏损产热，材料被动吸收全部转热；ASE 与 Seed 不可逆损耗分别记账。": (
        "Pump component ledger: active ionic absorption generates heat according to "
        "the selected quantum-defect model; passive material absorption becomes heat "
        "in full. Irreversible ASE and Seed losses are accounted for separately."
    ),
    "稳态热线性求解器": "Steady-State Thermal Linear Solver",
    "auto — 自动（优先稀疏 CG）": "auto — Automatic (Prefer Sparse CG)",
    "cg — 稀疏共轭梯度": "cg — Sparse Conjugate Gradient",
    "sor — 红黑 SOR": "sor — Red-Black SOR",
    "auto 在 SciPy 可用时使用守恒有限体积稀疏 CG，否则回退原有 SOR。": (
        "When SciPy is available, auto uses the conservative finite-volume sparse-CG "
        "solver; otherwise it falls back to the existing SOR solver."
    ),
    "稳态模型只使用导热率和边界；密度与比热在瞬态模型中决定热惯性。": (
        "The steady-state model uses only thermal conductivity and boundary conditions; "
        "density and specific heat determine thermal inertia in the transient model."
    ),
    "各表面冷却覆盖（可选 JSON）": "Per-Surface Cooling Overrides (Optional JSON)",
    "**局部温度 / 掺杂光谱反馈**": (
        "**Local Temperature / Dopant Spectroscopic-Parameter Feedback**"
    ),
    "四组截面响应 JSON（高级数据端口）": (
        "Four Cross-Section Response Models JSON (Advanced Data Interface)"
    ),
    "**ASE / 横向寄生激光（TPL）**": (
        "**ASE / Transverse Parasitic Lasing (TPL)**"
    ),
    "**晶体材料色散与强场效应**": (
        "**Crystal Material Dispersion and Strong-Field Effects**"
    ),
    "任意方向输运与反射边界幂迭代会报告主寄生本征值；曲面体素边界仍不解析逐射线 TIR。": (
        "Arbitrary-direction transport with reflective-boundary power iteration reports "
        "the dominant parasitic eigenvalue; curved voxel boundaries still do not "
        "resolve total internal reflection ray by ray."
    ),
    "关闭时严格恢复均匀离子密度，旧 JSON 和旧结果保持兼容。": (
        "When disabled, the uniform ion density is restored exactly for compatibility "
        "with legacy JSON configurations and results."
    ),
    "只需选择变化方式、方向和带正负号的百分比。多个方向相乘，体积平均标定仍保持上方填写的总平均掺杂浓度。": (
        "Select the variation form, direction, and a signed percentage. Variations "
        "along multiple directions multiply; volume-average calibration still preserves "
        "the whole-crystal average dopant concentration entered above."
    ),
    "启用未掺杂/低掺杂端帽": "Enable Undoped / Low-Doped End Caps",
    "端帽仍属于同一块基质和热传导域，只改变活性离子密度；适用于未掺杂端帽或分段掺杂晶体。": (
        "End caps remain part of the same host and thermal-conduction domain; only "
        "active-ion density changes. Use this for undoped end caps or segmented doping."
    ),
    "当前三维场预览：": "Current 3-D Field Preview: ",
    "当前三维场预览：最小 ": "Current 3-D Field Preview: Minimum ",
    "、体积平均 ": ", Volume Average ",
    "、最大 ": ", Maximum ",
    "最小 ": "Minimum ",
    "体积平均 ": "Volume Average ",
    "最大 ": "Maximum ",
    "掺杂分布中心切片": "Central Slice of the Dopant Distribution",
    "xy（z 中心）": "xy (z Centre)",
    "xz（y 中心）": "xz (y Centre)",
    "yz（x 中心）": "yz (x Centre)",
    "光学采样：Δx=": "Optical Sampling: Δx = ",
    " µm，Δy=": " µm, Δy = ",
    " µm；光学窗口必须完整位于晶体物理网格内。": (
        " µm. The optical window must lie entirely within the physical crystal grid."
    ),
    "多色×时间内存约与 Nt × Nλ × Nx × Ny 成正比；相干 CPA 内存主要约为 Nλ × Nx × Ny。逐时间窗增益会增加运行时间，但不会常驻独立的 Nt×Nλ 状态表。": (
        "Spectral × temporal memory scales approximately as Nt × Nλ × Nx × Ny; "
        "coherent-CPA memory is dominated by Nλ × Nx × Ny. Per-window gain updates "
        "increase run time but do not retain a separate Nt × Nλ state table."
    ),
    "population_dependent 会让 Pump 耗尽随局部基态/激发态布居变化；fixed_active_coefficient 只用给定标量推进 Pump 衰减，适合复现关闭 Pump saturation 的 Commercial App 算例。": (
        "population_dependent makes Pump depletion respond to the local ground- "
        "and excited-state populations. fixed_active_coefficient advances Pump "
        "attenuation with one supplied scalar to reproduce Commercial App cases with Pump "
        "saturation disabled."
    ),
    "固定 Pump 主动吸收系数 (m⁻¹)": (
        "Fixed Active Pump-Absorption Coefficient (m⁻¹)"
    ),
    "它决定 Pump 传播中的主动耗尽和吸收功率账本；能级激发仍按 Pump 光子流、跃迁截面与量子效率求解，不会把固定系数冒充受激跃迁截面。": (
        "This controls active depletion and the absorbed-power ledger during "
        "Pump transport. Level excitation is still solved from Pump photon "
        "flux, transition cross-sections, and quantum efficiency; the fixed "
        "coefficient is not substituted for a stimulated-transition cross-section."
    ),
    "Seed 不可逆产热耦合": "Irreversible Seed-Heat Coupling",
    "需要在 A 中选择 Seed 不可逆产热耦合；主动增益提取不算热，被动损耗才按 100% 转热。": (
        "Select irreversible Seed-heat coupling in section A. Stimulated gain "
        "extraction is not heat; enabled passive absorption is deposited as heat."
    ),
    "波长谱密度（色散前后相同）": (
        "Wavelength Spectral Density (Unchanged by Dispersion)"
    ),
    "波长谱能量密度 (J/nm)": "Wavelength Spectral Energy Density (J/nm)",
    "波长谱密度（纯 GDD/TOD/FOD 不改变谱能量密度）": (
        "Wavelength Spectral Density (Pure GDD/TOD/FOD Does Not Change "
        "Spectral Energy Density)"
    ),
    "输入波长谱密度目标": "Input Wavelength-Density Target",
    "off — 不把 Seed 不可逆损耗送入热场": (
        "off — Do Not Send Irreversible Seed Losses to the Thermal Field"
    ),
    "one_way_postprocess — Seed 不可逆损耗→热场→再传播一次 Seed": (
        "one_way_postprocess — Irreversible Seed Loss → Thermal Field → "
        "One Additional Seed Propagation"
    ),
    "fixed_point — Pump/布居/热场/Seed 不可逆产热共同迭代": (
        "fixed_point — Jointly Iterate Pump, Population, Thermal Field, and "
        "Irreversible Seed Heating"
    ),
    "不把 Seed 不可逆损耗送入热场": (
        "Do Not Send Irreversible Seed Losses to the Thermal Field"
    ),
    "Seed 不可逆损耗→热场→再传播一次 Seed": (
        "Irreversible Seed Loss → Thermal Field → One Additional Seed Propagation"
    ),
    "Pump/布居/热场/Seed 不可逆产热共同迭代": (
        "Jointly Iterate Pump, Population, Thermal Field, and Irreversible Seed Heating"
    ),
    "接收多光子吸收和自由载流子吸收；若“把 Seed 被动吸收作为热源”也开启，还接收材料被动吸收。Kerr 是纯相位，不会被重复算成热。": (
        "Includes multiphoton and free-carrier absorption. If Include Passive "
        "Seed Absorption as a Heat Source is enabled, it also includes material "
        "passive absorption. Kerr is phase-only and is not counted again as heat."
    ),
    "面积加权截面平均": "Area-Weighted Cross-Section Mean",
    "材料域体积加权平均值": "Material-Domain Volume-Weighted Mean",
    "simulation.multipass 编译模板": "simulation.multipass Compiled Template",
    "不可逆损耗机制": "Irreversible-Loss Mechanism",
    "材料强场配置": "Material Strong-Field Configuration",
    "机制标签决定参数的物理解读与严格验证要求；它不由 K 或 β 自动猜测。": (
        "The mechanism label determines the physical interpretation and strict "
        "validation requirements; it is not inferred automatically from K or beta."
    ),
    "多光子阶数": "Multiphoton Order",
    "沿光束真实路径按 dI/ds=-βK·I^K 推进；βK 的量纲随 K 改变，不能跨阶直接复用。": (
        "Advances dI/ds=-beta_K*I^K along the physical beam path. The dimensions of beta_K depend on K, "
        "so a coefficient cannot be reused across orders."
    ),
    "自由载流子状态": "Free-Carrier State",
    "决定是否推进跨脉冲载流子产生、复合、吸收与折射率响应。": (
        "Controls pulse-to-pulse carrier generation, recombination, absorption, "
        "and refractive-index response."
    ),
    "强场损耗直接热化比例": "Direct Strong-Field-Loss Thermalization Fraction",
    "材料强场经验闭合": "Empirical Material Strong-Field Closure",
    "该比例把记录到的多光子与自由载流子光学损耗直接送入热源；未送入热源的部分没有独立的载流子内部能量状态，因此这是降阶热闭合，不是完整的载流子能量输运模型。": (
        "This fraction sends recorded multiphoton and free-carrier optical loss "
        "directly to the heat source. The remainder has no separate carrier "
        "internal-energy state, so this is a reduced thermal closure rather than "
        "a complete carrier-energy transport model."
    ),
    "慢变量折射率反馈 Δnslow = (dn/dT)ΔT + Δnstress（只包含已开启且相容的项）": (
        "Slow Refractive-Index Feedback delta_n_slow = (dn/dT) delta_T + "
        "delta_n_stress (Enabled Compatible Terms Only)"
    ),
    "瞬时 Kerr 由独立传播算符沿真实光束路径累积 φK=k0∫n2 I ds；它不属于热场保存的 Δnslow": (
        "Instantaneous Kerr phase phi_K=k0 integral(n2 I ds) is accumulated along the physical beam path by "
        "a Separate Propagation Operator; It Is Not Part of Thermal delta_n_slow"
    ),
    "因果多程模板 ": "Causal Multipass Template ",
    " 编译为 ": " Compiles to ",
    " 次晶体命中；模板转移段拥有程间光程、损耗、方向和边界条件": (
        " Crystal Hits; Template Transfer Segments Own Inter-Pass Optical Lengths, "
        "Losses, Directions, and Boundary Conditions"
    ),
    "低维离散脉冲初值图": "Reduced Discrete-Pulse Initializer",
    "低维脉冲图初值": "Reduced Pulse-Map Initialization",
    "无法启动本机流式下载通道：": (
        "Could not start the local streaming-download channel: "
    ),
    "初值：低维离散脉冲图，": "Initializer: reduced discrete-pulse map, ",
    (
        " 次映射评估；随后已由当前完整主脉冲 pulse map 继续收敛，"
        "reduced 结果没有作为最终结果。"
    ): (
        " map evaluations; the configured complete main-pulse map then continued "
        "to convergence, and the reduced result was not used as the final result."
    ),
    (
        "先用周期平均作初值，再用所选的低维整脉冲 FN 或轴对称相干 Kerr 图靠近周期固定点；"
        "它只负责初始化，最终仍由当前完整主脉冲 pulse map 按配置容差收敛。"
    ): (
        "Start from the cycle-average solution, then use the selected reduced whole-pulse "
        "FN or axisymmetric coherent-Kerr map to approach the periodic fixed point. This "
        "stage only initializes the solve; the configured complete main-pulse map still "
        "converges to the requested tolerance."
    ),
    "直接从配置或上一个遍历点提供的周期边界布居开始完整主脉冲固定点。": (
        "Start the complete main-pulse fixed point directly from the configured periodic "
        "boundary population or the previous sweep point."
    ),
    "先求周期平均工作点作为数值初值，随后仍由完整主脉冲固定点收敛。": (
        "Use the cycle-average operating point as the numerical initializer, then converge "
        "the complete main-pulse fixed point."
    ),
    "根据寿命与重复周期判断是否使用周期平均初值；最终仍由完整主脉冲固定点收敛。": (
        "Choose whether to use a cycle-average initializer from the lifetime and repetition "
        "period; the complete main-pulse fixed point still supplies the final convergence."
    ),
    "用户选择的初始化层": "User-Selected Initialization Layer",
    "只建立完整主脉冲固定点的数值初值。": (
        "Build only the numerical initial state for the complete main-pulse fixed point."
    ),
    "低维脉冲图现在只作为完整 pulse_map 的初值加速器；最终固定点统一由主脉冲闭合。": (
        "The reduced pulse map now only accelerates initialization of the complete pulse_map; "
        "the main pulse closes the final fixed point."
    ),
    "已把旧 reduced_pulse_map 方法迁移到离散周期映射初值层。": (
        "Moved the legacy reduced_pulse_map method into the discrete periodic-map "
        "initialization layer."
    ),
    "低维脉冲初值只为 coherent_cpa 提供加速；当前后端直接使用周期平均初值后进入完整 pulse_map。": (
        "Reduced-pulse initialization only accelerates coherent_cpa. This backend uses the "
        "cycle-average initializer and then enters the complete pulse_map directly."
    ),
    "低维脉冲初值不传播时域自由载流子状态；完整 pulse_map 将直接从已有布居与载流子边界开始。": (
        "The reduced-pulse initializer does not propagate the time-resolved free-carrier "
        "state; the complete pulse_map starts directly from the existing population and "
        "carrier boundaries."
    ),
    "下载全部遍历结果包": "Download Complete Sweep Archive",
    "下载失败诊断结果包": "Download Failure Diagnostic Archive",
    (
        "只有点击后才会逐点生成下载流；不点击不创建结果 ZIP。"
        "下载完成、失败或取消后，临时压缩文件会自动清理。"
    ): (
        "The archive stream is generated point by point only after you click. "
        "No result ZIP is created otherwise, and temporary compression files are "
        "removed after completion, failure, or cancellation."
    ),
    (
        "只打包已经完成的点；当前暂停或计算中的点不会写入。"
        "完整三维结果从后端逐点读取并直接流向浏览器。"
    ): (
        "Package completed points only; exclude the paused or in-flight point. "
        "Complete 3D results are read one at a time and streamed directly to the browser."
    ),
    "下载严格无效诊断包（非正式结果）": (
        "Download Strict-Invalid Diagnostic Package (Not an Official Result)"
    ),
    "包内 SHA-256 只校验内容完整性，不是作者身份或来源签名。": (
        "The package SHA-256 verifies content integrity only; it is not an "
        "author or provenance signature."
    ),
    (
        "仅用于复现和定位严格验证失败；不是正式结果导出。包内保留配置、"
        "完整三维场、光学面、摘要及全部 warning/validation 诊断，重新载入时"
        "仍会标记为非定量。"
    ): (
        "Use this package only to reproduce and diagnose a strict-validation "
        "failure; it is not an official result export. It retains the configuration, "
        "complete 3D fields, optical planes, summary, and all warning/validation "
        "diagnostics, and remains marked non-quantitative when reloaded."
    ),
    (
        "包含配置、完整三维场、光学面、摘要及全部 warning/validation 诊断。"
        "这里只包含当前单点；二维参数网格请使用遍历结果区的“下载全部遍历结果包”。"
    ): (
        "Contains the configuration, complete 3D fields, optical planes, summary, "
        "and all warning/validation diagnostics. This package contains only the "
        "current point; for a 2D parameter grid, use ‘Download All Sweep Results’ "
        "in the sweep-results section."
    ),
    "该文件只用于复现严格验证失败并定位问题，不是推荐的正式结果归档。": (
        "This file is only for reproducing and diagnosing a strict-validation "
        "failure; it is not the recommended official result archive."
    ),
    "这是推荐归档格式；一个文件即可恢复本页的完整可视化和诊断。": (
        "This is the recommended archive format; one file restores the complete "
        "visualization and diagnostics on this page."
    ),
    "语言": "Language",
    "中文": "Chinese",
    "英文": "English",
    "三维激光放大器": "3D Laser Amplifier",
    "固体激光放大器工作台": "Solid-State Laser Amplifier Workbench",
    "从冷态光路到 Pump、布居、增益、热场和相干脉冲的同一套可追溯模型": (
        "One traceable model from the cold-state optical path through pump, "
        "population, gain, thermal fields, and coherent pulses"
    ),
    "页面": "Page",
    "结果": "Result",
    "放大器仿真": "Amplifier Simulation",
    "结果分析": "Result Analysis",
    "结果分析工作台": "Result Analysis Workbench",
    "运行当前配置": "Run Current Configuration",
    "仿真控制台": "Simulation Console",
    "先确认计算链，再按物理模块修改；未打开的高级参数仍保留在配置中。": (
        "Confirm the calculation chain first, then edit by physics module. "
        "Advanced parameters remain in the configuration when their panels are closed."
    ),
    "配置文件": "Configuration File",
    "材料库": "Material Library",
    "界面级别": "Interface Level",
    "引导模式": "Guided",
    "专家模式": "Expert",
    (
        "引导模式保留完整配置，但把后端覆盖和长参数区收起；专家模式展开逐项求解器覆盖。"
        "两种模式使用同一份物理配置，不会偷偷启用或关闭模块。"
    ): (
        "Guided mode preserves the complete configuration while collapsing backend "
        "overrides and long parameter sections. Expert mode expands per-feature solver "
        "overrides. Both use the same physics configuration and never silently enable "
        "or disable modules."
    ),
    "① 模型": "① Model",
    "② 光束": "② Beams",
    "③ 介质": "③ Medium",
    "④ 数值": "④ Numerics",
    "模型与运行": "Model and Run",
    "先选择需要保留的物理自由度，再决定多程、稳态和热光闭环。": (
        "Choose the physical degrees of freedom first, then configure multipass routing, "
        "steady state, and the thermo-optic loop."
    ),
    "求解家族、增益来源、周期策略与模块联动": (
        "solver family, gain source, periodic strategy, and module dependencies"
    ),
    "光束、脉冲与光路": "Beams, Pulses, and Optical Paths",
    "Pump 决定储能来源，Seed 决定被放大的时空场；两者的传播路径彼此独立。": (
        "Pump defines the stored-energy source, while Seed defines the amplified "
        "spatiotemporal field; their propagation paths are independent."
    ),
    "Pump/Seed 源、脉冲定义、空间模式、偏振与多程平台": (
        "Pump/Seed sources, pulse definition, spatial modes, polarization, and multipass bench"
    ),
    "增益介质与热结构": "Gain Medium and Thermal Structure",
    "把晶体、光谱能级、产热、冷却和应力放在同一条介质链中。": (
        "Keep the crystal, spectroscopic levels, heat generation, cooling, and stress "
        "in one medium chain."
    ),
    "材料参数、能级储能、热边界与力学反馈": (
        "material parameters, level storage, thermal boundaries, and mechanical feedback"
    ),
    "离散、预算与结果可信度": "Discretization, Budgets, and Result Credibility",
    "只在这里调整网格和算法预算；物理模型应先在前三页确定。": (
        "Adjust grids and algorithm budgets only here; define the physics model in the "
        "first three workspaces."
    ),
    "物理/光学双网格、采样、内存上限与有效性门控": (
        "physical/optical dual grids, sampling, memory limits, and validity gates"
    ),
    "本页负责": "Owns",
    "当前计算链": "Current Calculation Chain",
    "单色能量通量": "Monochromatic Fluence",
    "多色时间分箱": "Spectral-Time Bins",
    "JOSA 时空输运": "JOSA Spatiotemporal Transport",
    "Commercial App Gaussian-q 快速": "Fast Commercial App Gaussian-q",
    "相干全场 CPA": "Coherent Full-Field CPA",
    "规定泵浦分布": "Prescribed Pump Distribution",
    "规定泵浦焦散": "Prescribed Pump Caustic",
    "Pump ABCD": "Pump ABCD",
    "Pump 超高斯 ABCD": "Pump Super-Gaussian ABCD",
    "Pump FFT": "Pump FFT",
    "Pump 混合 ABCD/FFT": "Pump Hybrid ABCD/FFT",
    "Pump 随机光线追迹": "Pump Stochastic Ray Tracing",
    "二维超高斯：束腰半径 + 发散角": (
        "2D Super-Gaussian: Waist Radii + Divergence Angles"
    ),
    "这里仅定义光源相空间；传播算法在“求解器”中独立选择。emitter + stochastic_raytrace 会直接采样发光面和 NA；super_gaussian_abcd 用等二阶矩 M² 传播束宽但保留 SG 形状；其他场模式使用 M² 代理。x/y 完整指数相等时采用椭圆径向 SG，不等时采用明确的 x/y 可分离 SG。": (
        "This section defines only the source phase space; choose the propagation "
        "algorithm independently under Solver. emitter + stochastic_raytrace samples "
        "the emitting surface and NA directly. super_gaussian_abcd propagates width "
        "with an equivalent-second-moment M² while retaining the SG envelope; other "
        "field modes use an M² proxy. Equal complete x/y exponents select an elliptical "
        "radial SG; unequal exponents select an explicit separable x/y SG."
    ),
    "稳态布居": "Steady-State Population",
    "瞬态布居": "Transient Population",
    "周期布居": "Periodic Population",
    "热场关闭": "Thermal Field Off",
    "三维稳态热场": "3D Steady-State Thermal Field",
    "三维瞬态热场": "3D Transient Thermal Field",
    "Commercial App 专属闭合": "Commercial App-Specific Closures",
    "独立 HG 模态共享布居": "Independent HG Modes with Shared Population",
    "单一等效 M² Gaussian-q": "Single Effective-M² Gaussian-q",
    "一维 effective-area SPM": "1D Effective-Area SPM",
    "一维 effective-area SPM + 强场损耗/载流子": (
        "1D Effective-Area SPM + Strong-Field Loss/Carriers"
    ),
    "固定脉宽时间分箱": "Fixed-Duration Temporal Bins",
    "侧栏“专家模式”或完整 JSON；错误信息中的字段路径可直接搜索": (
        "Sidebar Expert Mode or Full JSON; you can search directly for the field path from the error"
    ),
    " Hz。唯一输入框在“② 光束 → Seed”；Pump 波形和周期稳态共享这个系统时钟。": (
        " Hz. The only input is under ② Beams → Seed; the Pump waveform and periodic "
        "steady state share this system clock."
    ),
    "① 模型 → 运行方式与求解": "① Model → Run Mode and Solver",
    "① 模型 → 高级求解器（专家模式）": "① Model → Advanced Solver (Expert Mode)",
    "① 模型 → 多物理耦合": "① Model → Multiphysics Coupling",
    "① 模型 → Pump + Seed 周期稳态": "① Model → Pump + Seed Periodic Steady State",
    "① 模型 → 运行方式与求解 / 高级求解器": (
        "① Model → Run Mode and Solver / Advanced Solver"
    ),
    "① 模型 / ② 光束": "① Model / ② Beams",
    "① 模型 → 多物理耦合 / ③ 介质": (
        "① Model → Multiphysics Coupling / ③ Medium"
    ),
    "② 光束 → Seed": "② Beams → Seed",
    "② 光束 → Pump": "② Beams → Pump",
    "③ 介质 → 晶体、材料与能级": "③ Medium → Crystal, Material, and Energy Levels",
    "③ 介质 → 热源与边界": "③ Medium → Heat Source and Boundaries",
    "③ 介质 → ASE": "③ Medium → ASE",
    "③ 介质 → ASE 与强场": "③ Medium → ASE and Strong-Field Physics",
    "③ 介质 → ASE / 寄生激光": "③ Medium → ASE / Parasitic Lasing",
    "运行前相容性检查通过：": "Pre-Run Compatibility Check Passed: ",
    "Commercial App 横向场由一个或多个 Gaussian-q 通道表示，必须使用 ABCD 空间后端。": (
        "Commercial App represents the transverse field with one or more Gaussian-q channels and "
        "therefore requires the ABCD spatial backend."
    ),
    "非相干 HG 需要独立传播模态/波长通道；whole-fluence 后端没有该合同，已切换到 spectral_time。": (
        "Incoherent HG requires independently propagated mode/wavelength channels. "
        "The whole-fluence backend does not define that contract, so the solver "
        "was switched to spectral_time."
    ),
    "Commercial App 使用互不耦合的 x/y q 参数；存在横向各向异性时必须与计算轴对齐（": (
        "Commercial App uses uncoupled x/y q parameters; transverse anisotropy must be aligned "
        "with the computational axes ("
    ),
    "这里显示的是待应用预览；只有点击下方蓝色按钮后，才会写入“③ 介质 → 晶体、材料与能级”的输入框。": (
        "This is a preview of the pending application. Values are written to ③ Medium → "
        "Crystal, Material, and Energy Levels only after you click the blue button below."
    ),
    "Seed 的每次 gain_medium 命中和中间透镜/反射镜，直接在“② 光束 → Seed”中布置。": (
        "Place every Seed gain_medium encounter and intermediate lens/mirror directly under "
        "② Beams → Seed."
    ),
    "local_3d 需要“① 模型 → 多物理耦合方式”中的 thermal_optical_fixed_point；当前组合不会降阶运行，以免把统一工作温度误当成局部三维反馈。": (
        "local_3d requires thermal_optical_fixed_point under ① Model → Multiphysics Coupling. "
        "The current combination will not run in a reduced mode, preventing a uniform operating "
        "temperature from being mistaken for local 3D feedback."
    ),
    "未保存光学面；在“① 模型 → 运行方式与求解”的高级区打开“保存各光学面”。": (
        "No optical planes were saved. Enable Save Optical Planes in the advanced section under "
        "① Model → Run Mode and Solver."
    ),
    "热光固定点要求 Pump 能响应三维折射率场。波动光学推荐 hybrid_abcd_fft；非相干高发散泵浦可选 stochastic_raytrace。热场和折射率模型在“③ 介质”页锁定为 steady_3d / thermal_3d。": (
        "The thermo-optic fixed point requires Pump propagation to respond to the 3D refractive-index "
        "field. hybrid_abcd_fft is recommended for wave optics; stochastic_raytrace is available for "
        "incoherent high-divergence pumps. Thermal and refractive-index models are locked to "
        "steady_3d / thermal_3d under ③ Medium."
    ),
    "反转密度定义为信号跃迁的 Nu−Nl；实际局部增益仍按 σeNu−σaNl 计算。种子净提取能量密度为有符号的主动跃迁能量交换，已剔除设定的被动损耗，并在所有晶体程次上累计。热源按“③ 介质”页选择的简化总吸收模型或 Pump 主动/被动分量账本计算；ASE/TPL、Seed 强场产热和普通 Seed 被动吸收热只在对应开关启用时计入；荧光逃逸仍未做独立输运闭环。": (
        "Inversion density is defined as Nu−Nl for the signal transition, while local gain is still "
        "computed as σeNu−σaNl. Net Seed extraction density is the signed active-transition energy "
        "exchange with configured passive losses removed and is accumulated across all crystal passes. "
        "The heat source uses either the simplified total-absorption model selected under ③ Medium or "
        "the Pump active/passive component ledger. ASE/TPL, Seed strong-field heating, and ordinary "
        "Seed passive-absorption heating are included only when their switches are enabled; fluorescence "
        "escape does not yet have an independent transport closure."
    ),
    "采样相位 / 二阶矩等效（通用）": "Sampled Phase / Second-Moment Equivalent (General)",
    "输出 Seed M² x（采样相空间）": "Output Seed M² x (Sampled Phase Space)",
    "输出 Seed M² y（采样相空间）": "Output Seed M² y (Sampled Phase Space)",
    "输出 Seed M² x（非相干二阶矩）": "Output Seed M² x (Incoherent Second Moment)",
    "输出 Seed M² y（非相干二阶矩）": "Output Seed M² y (Incoherent Second Moment)",
    "输出 Seed M² x（模态合同）": "Output Seed M² x (Modal Contract)",
    "输出 Seed M² y（模态合同）": "Output Seed M² y (Modal Contract)",
    "输出有效 Seed M² x（代理）": "Output Effective Seed M² x (Proxy)",
    "输出有效 Seed M² y（代理）": "Output Effective Seed M² y (Proxy)",
    "报告的 Seed M² x（见每点方法）": "Reported Seed M² x (See Per-Point Method)",
    "报告的 Seed M² y（见每点方法）": "Reported Seed M² y (See Per-Point Method)",
    "Commercial App 局部抛物线拟合窗口": "Commercial App Local Parabolic Fit Window",
    "Commercial App 热相位闭合": "Commercial App Thermal-Phase Closure",
    (
        "sampled_phase 从传播场采样热相位并做二阶矩闭合；local_parabolic 只在指定的"
        "光束邻域拟合局部抛物线，再把曲率作为 Gaussian-q 热透镜。后者用于严格对齐"
        "Commercial App 风格的局部热透镜传播。"
    ): (
        "sampled_phase samples the propagated thermal phase and applies a second-moment "
        "closure. local_parabolic fits a local parabola only within the specified beam "
        "neighborhood and applies its curvature as a Gaussian-q thermal lens. Use the latter "
        "to match Commercial App-style local thermal-lens propagation."
    ),
    "热相位拟合半径 x (m)": "Thermal-Phase Fit Radius x (m)",
    "热相位拟合半径 y (m)": "Thermal-Phase Fit Radius y (m)",
    "拟合窗口应覆盖主光束，但不要把远离光束的非抛物热相位硬塞入同一个焦距。": (
        "The fit window should cover the main beam without forcing distant nonparabolic "
        "thermal phase into one focal length."
    ),
    "x/y 可不同，用于像散热透镜；两者都必须为正。": (
        "x and y may differ for an astigmatic thermal lens; both must be positive."
    ),
    "手动指定热相位拟合中心": "Specify Thermal-Phase Fit Center Manually",
    "关闭时每层跟随当前 Gaussian-q 光束中心；开启时使用下面的固定物理坐标。": (
        "When off, each layer follows the current Gaussian-q beam center. When on, the fixed "
        "physical coordinates below are used."
    ),
    "拟合中心 x (m)": "Fit Center x (m)",
    "拟合中心 y (m)": "Fit Center y (m)",
    "当前不使用手动拟合半径；JSON 中保留的局部窗口参数不会参与 sampled_phase 计算。": (
        "Manual fit radii are currently inactive. Local-window parameters retained in JSON "
        "do not participate in sampled_phase calculations."
    ),
    "自动选择（推荐；跟随精度预设）": "Automatic (Recommended; Follows Fidelity Preset)",
    "只看能量与饱和 · 单色通量 / FN": "Energy and Saturation Only · Monochromatic Fluence / FN",
    "Commercial App 风格快速 · Gaussian-q / 热-Kerr 透镜": "Fast Commercial App Style · Gaussian-q / Thermal-Kerr Lens",
    "多波长 + 脉内耗尽 · 不保留相干相位": "Multiwavelength + Intrapulse Depletion · No Coherent Phase",
    "JOSA 时空速率方程 · 不保留相干相位": "JOSA Spatiotemporal Rate Equations · No Coherent Phase",
    "完整相干脉冲 · 衍射 / 色散 / Kerr": "Full Coherent Pulse · Diffraction / Dispersion / Kerr",
    (
        "按你需要的输出选择：只比较能量时用 fluence；需要 Commercial App 风格热透镜快速扫描时用 Commercial App Fast；"
        "需要脉内人口竞争但不追踪相干相位时用 spectral_time/JOSA；需要压缩、色散或 SPM 相位时用 coherent CPA。"
    ): (
        "Choose by the outputs you need: use fluence for energy-only comparisons; Commercial App Fast "
        "for rapid Commercial App-style thermal-lens scans; spectral_time/JOSA for intrapulse population "
        "competition without coherent phase; and coherent CPA for compression, dispersion, "
        "or SPM phase."
    ),
    "③ 介质 → 应力与光弹": "③ Medium → Stress and Photoelasticity",
    "唯一输入框在“② 光束 → Seed”": "the only input is under ② Beams → Seed",
    "Pump 波形和周期稳态共享这项系统时钟": (
        "the Pump waveform and periodic steady state share this system clock"
    ),
    "Pump 波形和周期稳态共享这个系统时钟。": (
        "the Pump waveform and periodic steady state share this system clock."
    ),
    "从左侧四个工作区完成设置后点击“运行当前配置”。建议先用 balanced 和较小网格确认工作流，再在“③ 介质”和“④ 数值”页打开详细模型并做收敛检查。": (
        "Complete the four workspaces on the left, then click Run Current Configuration. "
        "Start with the balanced preset and a smaller grid, then enable detailed models "
        "and convergence checks under ③ Medium and ④ Numerics."
    ),
    (
        "这里的 Commercial App Fast 是本程序实现的 Commercial App 风格降阶求解器，不是调用或复刻"
        " Commercial App 可执行程序。它仍读取同一套材料截面、Pump、真实能级布居、光路和"
        "热场；主要把 Seed 横向复场降为一个或多个 Gaussian-q 通道，从而用较低成本"
        "保留放大、热透镜、Kerr 透镜和增益导引的一阶影响。"
    ): (
        "Commercial App Fast is this application's Commercial App-style reduced-order solver; it neither invokes "
        "nor reproduces the Commercial App executable. It still uses the same material cross sections, "
        "Pump, physical level populations, optical path, and thermal field. Its main reduction "
        "represents the transverse Seed field with one or more Gaussian-q channels while "
        "retaining first-order amplification, thermal-lens, Kerr-lens, and gain-guiding effects."
    ),
    "#### Commercial App Fast · 当前后端专属设置": "#### Commercial App Fast · Active-Backend Settings",
    "Commercial App Fast · 当前后端专属设置": "Commercial App Fast · Active-Backend Settings",
    "这里的开关只由 Commercial App Fast 读取；切换到其他 Seed 后端后会保留数值，但不会参与计算。": (
        "Only Commercial App Fast reads these controls. Their values are preserved after switching "
        "to another Seed backend, but they do not participate in that calculation."
    ),
    "横向闭合：独立 HG Gaussian-q 通道共享局部布居。模阶、容量和 M² 在“② 光束 → Seed → 空间”中编辑。": (
        "Transverse closure: independent HG Gaussian-q channels share the local population. "
        "Edit mode order, capacity, and M² under ② Beams → Seed → Spatial."
    ),
    "横向闭合：单一 effective-M² Gaussian-q。若要启用 HG 0–2 独立传播，请在“② 光束 → Seed → 空间”选择 HG 模态逐个传播。": (
        "Transverse closure: one effective-M² Gaussian-q channel. To enable independent HG "
        "0–2 propagation, select per-mode HG propagation under ② Beams → Seed → Spatial."
    ),
    (
        "把输入脉冲分成有序通量箱，前沿、峰值和尾沿依次耗尽同一布居，"
        "并在相邻时间箱之间推进下能级/上能级暗弛豫。所有横向通道共享同一"
        "局部布居库，因此比 JOSA/full-field 快，但不含延迟时间输运。"
    ): (
        "Split the input pulse into ordered fluence bins so the leading edge, peak, and "
        "trailing edge deplete one population in sequence, with lower/upper-level dark "
        "relaxation between bins. All transverse channels share the same local population "
        "reservoir, making this faster than JOSA/full-field without retarded-time transport."
    ),
    (
        "增益按有序时间箱耗尽，并在箱间推进能级暗弛豫；所有时间箱和横向"
        "Gaussian-q 通道共享局部布居。可输出脉冲能量分箱、峰值和 FWHM，"
        "但不保留频率间相干相位或延迟时间输运。"
    ): (
        "Gain is depleted in ordered temporal bins with dark level relaxation between bins; "
        "all temporal bins and transverse Gaussian-q channels share the local population. "
        "Pulse-energy bins, peak power, and FWHM are available, but inter-frequency coherent "
        "phase and retarded-time transport are not retained."
    ),
    (
        "把脉冲能量分成有序增益时间箱，在相邻箱之间推进完整能级暗弛豫；"
        "所有时间箱和 Gaussian-q 横向通道共享局部布居，不解析延迟时间输运。"
    ): (
        "Split pulse energy into ordered gain-time bins and advance complete dark level "
        "kinetics between bins. All temporal bins and Gaussian-q transverse channels share "
        "the local population; retarded-time transport is not resolved."
    ),
    "当前配置的完整物理链路": "Complete Physics Chain for the Current Configuration",
    "当前模型功能清单（含关闭项与锁定项）": (
        "Current Model Features (Including Disabled and Locked Items)"
    ),
    "当前模型功能清单（含关闭与锁定项）": (
        "Current Model Features (Including Disabled and Locked Items)"
    ),
    "清爽设置": "Essentials",
    "完整设置": "Full Settings",
    "运行": "Run",
    "材料 / 热": "Materials / Thermal",
    "网格 / 验证": "Grid / Validation",
    "Pump / Seed": "Pump / Seed",
    "运行方式与求解": "Run Mode and Solver",
    "晶体、材料与能级": "Crystal, Material, and Energy Levels",
    "Pump · 功率、空间与光谱": "Pump · Power, Spatial, and Spectral",
    "Seed · 脉冲、空间与相干场": "Seed · Pulse, Spatial, and Coherent Field",
    "A. 多物理耦合方式": "A. Multiphysics Coupling",
    "B. Pump + Seed 周期稳态": "B. Pump + Seed Periodic Steady State",
    "C. 高级：逐项覆盖预设与性能": "C. Advanced: Per-Feature Overrides and Performance",
    "#### C. 高级：逐项覆盖预设与性能": (
        "#### C. Advanced: Per-Feature Overrides and Performance"
    ),
    "网格、采样与内存": "Grid, Sampling, and Memory",
    "三维光路工作台": "3D Optical-Path Workbench",
    "三维光路（拖动旋转）": "3D Optical Path (Drag to Rotate)",
    "三维场与可拖动切片": "3D Fields and Draggable Slices",
    "光谱与脉冲": "Spectrum and Pulse",
    "诊断与导出": "Diagnostics and Export",
    "逐程与光学面": "Passes and Optical Planes",
    "z — 横截面 (x-y)": "z — Cross-Section (x-y)",
    "y — 纵截面 (x-z)": "y — Longitudinal Section (x-z)",
    "x — 纵截面 (y-z)": "x — Longitudinal Section (y-z)",
    "解析源面显示": "Analytic Source-Plane Display",
    "Commercial App 模型合同": "Commercial App Model Contract",
    "输入相位假设": "Input Phase Assumption",
    "求解器": "Solver",
    "实时求解进度": "Live Solver Progress",
    "阶段求解进度": "Solver Milestones",
    "### 阶段求解进度": "### Solver Milestones",
    "参数遍历阶段进度": "Sweep Milestones",
    "### 参数遍历阶段进度": "### Sweep Milestones",
    "初始化网格、材料和后端": "Initialize Grid, Material, and Backends",
    "Pump 与布居工作点": "Pump and Population Operating Point",
    "能级布居": "Level Population",
    "稳态布居固定点": "Steady-State Population Fixed Point",
    "Pump 光路传播": "Pump Optical Propagation",
    "Pump 时间波形": "Pump Temporal Waveform",
    "周期平均 Pump + Seed 固定点": "Cycle-Averaged Pump + Seed Fixed Point",
    "逐脉冲固定点初始化": "Pulse-Map Fixed-Point Initialization",
    "逐脉冲周期固定点": "Pulse-by-Pulse Periodic Fixed Point",
    "瞬态逐脉冲序列": "Transient Pulse-by-Pulse Train",
    "Pump–布居–热光固定点": "Pump–Population–Thermo-Optic Fixed Point",
    "热光最终闭合检查": "Final Thermo-Optic Closure Check",
    "热源与热光响应": "Heat Source and Thermo-Optic Response",
    "三维热场": "3D Thermal Field",
    "应力、光弹与端面形变": "Stress, Photoelasticity, and End-Face Deformation",
    "Seed 放大与传播": "Seed Amplification and Propagation",
    "Seed 不可逆损耗产热回算": "Seed Irreversible-Loss Heat Recalculation",
    "能量账本与结果有效性检查": "Energy Ledger and Result Validation",
    "全部求解完成": "Solve Complete",
    "当前物理状态": "Current Physical State",
    "**当前物理状态**　": "**Current Physical State**  ",
    "当前求解位置": "Current Solve Position",
    "最近输出": "Latest Output",
    "最近收敛残差": "Latest Convergence Residual",
    "平均激发态": "Mean Excited Fraction",
    "输出光谱峰值": "Output Spectral Peak",
    "阶段速度": "Stage Rate",
    "已用 / 粗略剩余": "Elapsed / Approx. Remaining",
    "阶段": "Stage",
    "求解位置": "Solve Position",
    "当前结果": "Current Result",
    "收敛状态": "Convergence",
    "累计用时": "Elapsed",
    "阶段切换": "Stage Transition",
    "脉冲 ": "Pulse ",
    "程次 ": "Pass ",
    " · 晶体单元 ": " · Crystal Cell ",
    "迭代 ": "Iteration ",
    "热场 t=": "Thermal t=",
    "当前阶段实时趋势 · ": "Live Stage Trend · ",
    "当前迭代 / 脉冲 / 传播步": "Iteration / Pulse / Propagation Step",
    "相对残差": "Relative Residual",
    "输出脉冲能量 (nJ)": "Output Pulse Energy (nJ)",
    "输出功率 (W)": "Output Power (W)",
    "传播中的 Seed": "Seed in Propagation",
    "Pump 功率 (W)": "Pump Power (W)",
    "峰值温度 (K)": "Peak Temperature (K)",
    "激发/上能级平均占比 (%)": "Mean Excited/Upper-Level Fraction (%)",
    "传播中的脉冲能量 (nJ)": "Pulse Energy in Propagation (nJ)",
    "传播中的平均功率 (W)": "Average Power in Propagation (W)",
    "当前残差": "Current Residual",
    "布居残差": "Population Residual",
    "输出残差": "Output Residual",
    "Δn 残差": "Δn Residual",
    "载流子残差": "Free-Carrier Residual",
    "载流子影响残差（诊断）": "Free-Carrier Effect Residual (Diagnostic)",
    "载流子影响残差": "Free-Carrier Effect Residual",
    "温度更新残差": "Temperature-Update Residual",
    "热方程残差": "Thermal-Equation Residual",
    "目标容差": "Target Tolerance",
    "吸收 Pump": "Absorbed Pump",
    "Pump 输出": "Pump Output",
    "瞬时 Pump": "Instantaneous Pump",
    "平均 Pump": "Average Pump",
    "Seed 输入平均功率": "Seed Input Average Power",
    "输出功率": "Output Power",
    "输出脉冲能量": "Output Pulse Energy",
    "当前传播功率": "Current Transport Power",
    "当前传播能量": "Current Transport Energy",
    "输出峰值功率": "Output Peak Power",
    "压缩后峰值功率": "Compressed Peak Power",
    "输出脉宽": "Output Pulse Width",
    "压缩后脉宽": "Compressed Pulse Width",
    "累计 B 积分": "Accumulated B Integral",
    "最大非线性子步相位": "Maximum Nonlinear Substep Phase",
    "时间窗边界能量": "Temporal-Window Boundary Energy",
    "光谱峰值波长": "Spectral Peak Wavelength",
    "上能级平均占比": "Mean Upper-Level Fraction",
    "上能级最高占比": "Maximum Upper-Level Fraction",
    "激发态平均占比": "Mean Excited-State Fraction",
    "激发态最高占比": "Maximum Excited-State Fraction",
    "峰值温度": "Peak Temperature",
    "热场时间": "Thermal Time",
    "Pump 波形时间": "Pump Waveform Time",
    "热场收敛": "Thermal Converged",
    "阶段收敛": "Stage Converged",
    "结果有效": "Result Valid",
    "警告数": "Warning Count",
    "当前脉冲": "Current Pulse",
    "脉冲总数": "Total Pulses",
    "本周期 Seed 已入射": "Seed Applied This Period",
    "等待最终报告脉冲": "Awaiting Final Reported Pulse",
    "连续稳定周期": "Consecutive Stable Cycles",
    "所需稳定周期": "Required Stable Cycles",
    "当前程次": "Current Pass",
    "程内晶体网格": "In-Pass Crystal Cell",
    "每程晶体网格": "Crystal Cells per Pass",
    "晶体 z 位置": "In-Crystal z Position",
    "z 网格索引": "z Grid Index",
    "热场内迭代": "Thermal Inner Iterations",
    "Seed 后端": "Seed Backend",
    "初值来源": "Initialization Source",
    "请求的初始化方式": "Requested Initialization Method",
    "当前迭代类型": "Current Iteration Type",
    "配置的初始布居": "Configured Initial Population",
    "初始布居": "Initial Population",
    "周期平均固定点初值": "Cycle-Average Fixed-Point Initializer",
    "逐脉冲前的周期平均初值": "Pre-Pulse-Map Cycle-Average Initializer",
    "热光外层热启动": "Thermo-Optic Outer Warm Start",
    "逐脉冲前的周期平均固定点": "Pre-Pulse-Map Cycle-Average Fixed Point",
    "周期平均固定点": "Cycle-Average Fixed Point",
    "周期平均": "Cycle Average",
    "自动选择": "Automatic Selection",
    "已用周期平均初值": "Cycle-Average Initializer Used",
    "周期平均初值收敛": "Cycle-Average Initializer Converged",
    "周期平均初值迭代数": "Cycle-Average Initializer Iterations",
    "Seed 子阶段": "Seed Substage",
    "当前光谱通道": "Current Spectral Channel",
    "光谱通道总数": "Total Spectral Channels",
    "当前通道波长": "Current Channel Wavelength",
    "当前通道输入能量": "Current Channel Input Energy",
    "当前通道输入功率": "Current Channel Input Power",
    "光谱通道数": "Spectral Channel Count",
    "光学网格 Nx": "Optical Grid Nx",
    "光学网格 Ny": "Optical Grid Ny",
    "基础复场内存估计": "Base Complex-Field Memory Estimate",
    "固定点迭代": "Fixed-Point Iteration",
    "固定点迭代上限": "Fixed-Point Iteration Limit",
    "逐脉冲周期": "Pulse-Map Cycle",
    "当前晶体单元子步": "Current Crystal-Cell Substeps",
    "当前光学子步": "Current Optical Substep",
    "光学子步总数": "Total Optical Substeps",
    "晶体单元状态": "Crystal-Cell State",
    "上一晶体单元耗时": "Previous Crystal-Cell Time",
    "相干精度档": "Coherent Fidelity Level",
    "相干频率/时间点": "Coherent Frequency/Time Samples",
    "空间坐标模型": "Spatial Coordinate Model",
    "轴对称径向分箱": "Axisymmetric Radial Bins",
    "瞬态软窗增益": "Transient Soft-Window Gain",
    "当前软窗数": "Current Soft-Window Count",
    "最大软窗数": "Maximum Soft-Window Count",
    "轴对称半径 r": "Axisymmetric Radius r",
    "L4 轴对称整脉冲 Kerr": "L4 Axisymmetric Whole-Pulse Kerr",
    "L4.5 轴对称瞬态 Kerr": "L4.5 Axisymmetric Transient Kerr",
    "L5 全场整脉冲 Kerr": "L5 Full-Field Whole-Pulse Kerr",
    "L6 全场瞬态 Kerr": "L6 Full-Field Transient Kerr",
    "上面的精度预设管理 Pump、热、界面等基础栈；Seed 求解模型管理场的自由度。需要整套一致配置时，使用下面按家族分开的快捷档。": (
        "The precision preset above controls the base Pump, thermal, and interface "
        "stack; the Seed solver model controls the field degrees of freedom. Use the "
        "family-specific shortcuts below when you need a complete consistent setup."
    ),
    "求解家族快捷配置（L / JOSA / Commercial App）": "Solver-Family Shortcuts (L / JOSA / Commercial App)",
    "相干 L 系列": "Coherent L Series",
    "JOSA J 系列": "JOSA J Series",
    "Commercial App 快速": "Commercial App Fast",
    "JOSA 整套快捷档": "Complete JOSA Shortcut Profile",
    "应用所选 JOSA 整套档": "Apply Selected Complete JOSA Profile",
    "J1/J2 隔离局部输运并用周期平均；J3 恢复空间传播和完整逐脉冲工作点；J4 再锁定完整三维热光与 Pump/Seed FFT。四档都不传播相干频率相位。": (
        "J1/J2 isolate local transport and use a cycle average; J3 restores spatial "
        "propagation and the full pulse-map operating point; J4 additionally locks the "
        "complete 3D thermo-optic stack and Pump/Seed FFT. None propagates coherent "
        "inter-frequency phase."
    ),
    "保留材料常数、Pump/Seed 源、网格尺寸、光路与 20 mm ZnSe 输入定义；按所选 J 档同步增益来源、饱和、周期方法、传播和热光合同。": (
        "Preserves material constants, Pump/Seed sources, grid dimensions, optical "
        "train, and the 20 mm ZnSe input definition; synchronizes gain origin, "
        "saturation, periodic method, propagation, and thermo-optic contracts for the "
        "selected J profile."
    ),
    "JOSA 的“时间×波长分箱”不是复光谱：源/展宽器相位只用于生成输入时间强度；输出 GDD/TOD、压缩、Kerr/SPM 必须回到相干 L 分支核对。": (
        "JOSA time-by-wavelength bins are not a complex spectrum: source/stretcher "
        "phase is used only to create the input temporal intensity. Output GDD/TOD, "
        "compression, and Kerr/SPM must be checked with the coherent L branch."
    ),
    "JOSA 会先用源/展宽器相位构造一次输入时间强度，再把相位丢弃；随后只传播有序的时间×波长通量和真实能级布居。材料/光路 GDD、压缩器、Kerr 与 SPM 不会在 JOSA 内生效；需要这些输出相位效应时请选择 coherent_cpa。": (
        "JOSA first uses source/stretcher phase once to construct the input temporal "
        "intensity, then discards phase and transports only ordered time-by-wavelength "
        "fluence and physical level populations. Material/train GDD, compressors, Kerr, "
        "and SPM do not act inside JOSA; select coherent_cpa for those output phase effects."
    ),
    "JOSA 必须更新真实上下能级布居，因此选择 JOSA 时固定使用 Pump + 速率方程；退出 JOSA 后，表单会恢复先前由用户设置的直接增益值。": (
        "JOSA must update physical upper/lower populations, so it locks gain to Pump "
        "plus rate equations. The form restores the user's previous direct-gain value "
        "after leaving JOSA."
    ),
    "JOSA 中 Sellmeier 的 n(λ)/ng(λ) 仍用于各波长衍射与 full_3d_pc 群延迟；下面额外的材料 GDD/TOD/FOD 相位和 n₂ 不进入 JOSA。切换回相干/Commercial App 后这些材料值会保留。": (
        "In JOSA, Sellmeier n(lambda)/ng(lambda) still controls wavelength-resolved "
        "diffraction and full_3d_pc group delay. The additional material GDD/TOD/FOD "
        "phase and n2 below do not enter JOSA; their values are retained when switching "
        "back to coherent or Commercial App modes."
    ),
    "kerr_time_domain 只用于 coherent_cpa；kerr_parabolic_lens 只用于 commercial_app_fast。所有 JOSA 模式都传播强度/光子通量而不是 Kerr 相位。": (
        "kerr_time_domain is exclusive to coherent_cpa; kerr_parabolic_lens is exclusive "
        "to commercial_app_fast. Every JOSA mode transports intensity/photon fluence rather than Kerr phase."
    ),
    "频率相干相位": "Coherent Spectral Phase",
    "群延迟走离": "Group-Delay Walk-Off",
    "不传播": "Not Propagated",
    "输入相位→时间强度": "Input Phase → Temporal Intensity",
    "已应用": "Applied",
    "未应用": "Not Applied",
    "传播中的相干谱相位": "Coherent Spectral Phase During Transport",
    "保留": "Retained",
    "不保留": "Not Retained",
    "材料/光路 GDD·TOD·FOD": "Material/Train GDD·TOD·FOD",
    "传播": "Propagated",
    "Kerr / SPM": "Kerr / SPM",
    "JOSA 只在入口把可用的源/展宽器相位投影成时间强度；进入输运后，相位不会随时间箱和波长箱继续演化。": (
        "JOSA only projects available source/stretcher phase into temporal intensity "
        "at the input. Phase does not continue evolving across time and wavelength "
        "bins during transport."
    ),
    "近似": "Approximate",
    "不解析": "Not Resolved",
    "JOSA 需要“增益饱和/布居耗尽 = on”；表单会在运行前自动联动并显示变更。": (
        "JOSA requires gain saturation/population depletion to be on; the form "
        "automatically reconciles it before the run and displays the change."
    ),
    "J1 · Explicit CFL + 周期平均（输运基准）": "J1 · Explicit CFL + Cycle Average (Transport Baseline)",
    "冻结晶体内衍射与热相位，用论文 CFL 绑定的一阶显式格式核对局部光子输运；适合排查符号、守恒与网格趋势，不作为高增益最终值。": (
        "Freezes in-crystal diffraction and thermal phase, then checks local photon "
        "transport with the first-order paper-CFL scheme. Use it for signs, conservation, "
        "and grid trends, not as a final high-gain value."
    ),
    "J2 · PC CFL + 周期平均（论文格式核对）": "J2 · PC CFL + Cycle Average (Paper-Format Check)",
    "保持 J1 的隔离输运工况，改用 prediction-correction；Δt=Δs/vg，因此改变 Nz 也会改变时间分箱。": (
        "Keeps J1's isolated transport setup but uses prediction-correction. Because "
        "dt=ds/vg, changing Nz also changes the temporal bins."
    ),
    "J3 · 延迟时间 PC + 完整 pulse map（日常定量）": "J3 · Retarded-Time PC + Full Pulse Map (Routine Quantitative)",
    "时间分箱与 Nz 解耦，恢复晶体内横向衍射、热折射率和逐脉冲周期工作点；仍是相位不敏感的时间×波长强度输运。": (
        "Decouples time bins from Nz and restores transverse diffraction, thermal index, "
        "and the pulse-by-pulse periodic operating point; it remains phase-insensitive "
        "time-by-wavelength intensity transport."
    ),
    "J4 · 完整 3D PC + pulse map（JOSA 上限）": "J4 · Full 3D PC + Pulse Map (JOSA Upper Tier)",
    "再锁定 Seed/Pump FFT、三维热光、泵浦耗尽、群延迟与斜入射走离；群延迟混箱仍是能量守恒相位闭合近似，不等于 coherent CPA。": (
        "Additionally locks Seed/Pump FFT, 3D thermo-optics, Pump depletion, group delay, "
        "and oblique-incidence walk-off. Group-delay bin mixing remains an "
        "energy-conserving phase-closure approximation, not coherent CPA."
    ),
    "按时间×波长×横向空间输运光子通量并依次更新真实布居，适合强脉内耗尽；它不保留频率间相干相位，四个模式改变积分/时间坐标，full_3d_pc 还锁定多物理栈和近似群延迟走离。": (
        "Transports photon fluence over time, wavelength, and transverse space while "
        "updating physical populations in order. It handles strong intrapulse depletion "
        "but not coherent inter-frequency phase; the four modes change integration/time "
        "coordinates, and full_3d_pc also locks the multiphysics stack and approximate group-delay walk-off."
    ),
    "四种模式都使用时间×波长×横向空间通量；它们改变局部积分格式与时间坐标，full_3d_pc 还锁定完整三维多物理栈和近似群延迟走离。": (
        "All four modes use time-by-wavelength transverse fluence. They change the local "
        "integration scheme and time coordinate; full_3d_pc also locks the complete 3D "
        "multiphysics stack and approximate group-delay walk-off."
    ),
    "所有 JOSA 模式都按有序时间×波长箱更新横向局域通量和真实布居；所选模式改变积分/时间坐标，full_3d_pc 另锁定 Pump/Seed FFT、三维热光与近似群延迟走离。": (
        "Every JOSA mode updates transverse local fluence and physical populations over "
        "ordered time-by-wavelength bins. The selected mode changes integration/time "
        "coordinates; full_3d_pc additionally locks Pump/Seed FFT, 3D thermo-optics, and "
        "approximate group-delay walk-off."
    ),
    "在有序时间×波长箱中联立光子通量输运与能级速率方程；front、peak、tail 依次看见已经被前一部分改变的布居，横向场仍可做采样衍射。源/展宽器相位只可先投影成输入时间强度；传播中不保留频率间相干相位，因而不预测 Kerr/SPM 或压缩器输出。": (
        "Couples photon-fluence transport and level-rate equations over ordered time-by-"
        "wavelength bins, so front, peak, and tail see successively depleted populations "
        "while the transverse field can still diffract on a sampled grid. Source/stretcher "
        "phase can only be projected into input temporal intensity; coherent phase is not "
        "retained during propagation, so Kerr/SPM and compressor output are not predicted."
    ),
    "JOSA 光子输运与能级布居在每个有序时间箱内共同更新，因此必须提交受激耗尽。": (
        "JOSA photon transport and level populations update together in each ordered "
        "time bin, so stimulated depletion must be committed."
    ),
    "JOSA 必须读取真实上下能级布居；外加小信号增益没有可守恒更新的布居储能库。": (
        "JOSA must read physical upper/lower populations; an imposed small-signal gain "
        "has no population energy reservoir that can be updated conservatively."
    ),
    "reduced_pulse_map 是 coherent_cpa 的低维周期代理；当前后端已是所选真实脉冲算符，因此改用完整 pulse_map。": (
        "reduced_pulse_map is a low-dimensional periodic proxy for coherent_cpa. The "
        "current backend is already the selected physical pulse operator, so the full "
        "pulse_map is used instead."
    ),
    "宽带 Seed 光谱通道构造": "Broadband Seed Spectral-Channel Setup",
    "Seed 晶体内传播": "In-Crystal Seed Propagation",
    "Seed 输出与脉冲诊断": "Seed Output and Pulse Diagnostics",
    "周期平均 Pump 探针": "Cycle-Average Pump Probe",
    "周期平均最终 Pump 闭合探针": "Final Cycle-Average Pump Closure Probe",
    "正在进入晶体单元": "Entering Crystal Cell",
    "光学子步已完成": "Optical Substep Complete",
    "晶体单元已完成": "Crystal Cell Complete",
    "正在构造的通道能量 (nJ)": "Channel Energy During Setup (nJ)",
    "正在构造的通道功率 (W)": "Channel Power During Setup (W)",
    "光谱通道 ": "Spectral Channel ",
    "输出诊断 ": "Output Diagnostics ",
    " · 子步 ": " · Substep ",
    " · 径向 ": " · Radial ",
    " · 软窗 ": " · Soft Windows ",
    "固定点 ": "Fixed Point ",
    "**L4.5：轴对称 Kerr + 瞬态增益**": (
        "**L4.5: Axisymmetric Kerr + Transient Gain**"
    ),
    "晶体内使用 E(f,r)，保留径向衍射/自聚焦、完整相干频率网格、色散和局域时间域 Kerr；增益仍按 front → peak → tail 自适应软窗逐次耗尽，周期工作点使用完整 pulse map。相对 L6 只删去非零方位角空间模式，不删时间顺序；适用于居中、圆对称且轴对称有效性门限通过的工况。": (
        "Inside the crystal this profile propagates E(f,r), retaining radial "
        "diffraction/self-focusing, the native coherent frequency grid, dispersion, "
        "and local time-domain Kerr. Gain depletion still advances through adaptive "
        "soft windows from front to peak to tail, and the periodic operating point "
        "uses the full pulse map. Relative to L6 it removes only nonzero azimuthal "
        "spatial modes, not temporal ordering; use it for centered circular systems "
        "only when the axisymmetric applicability gates pass."
    ),
    "应用 L4.5 轴对称瞬态 Kerr": "Apply L4.5 Axisymmetric Transient Kerr",
    "L4.5 轴对称瞬态 Kerr 已启用：晶体内传播完整 E(f,r)，保留径向衍射、色散、局域时间域 Kerr，以及 front → peak → tail 软窗增益；不保留非零方位角模式，因此必须检查结果中的轴对称投影、边界和热场/布居不对称门限。": (
        "L4.5 axisymmetric transient Kerr is active. The crystal propagates the full "
        "E(f,r) field with radial diffraction, dispersion, local time-domain Kerr, "
        "and front-to-peak-to-tail soft-window gain. Nonzero azimuthal modes are not "
        "retained, so inspect the projection, radial-boundary, and thermal/population "
        "asymmetry gates in the result."
    ),
    "不改材料、光束、采样数、网格、光路或热边界；会选择完整 pulse_map、开启软窗瞬态增益，并关闭增益多速率合并。": (
        "Preserves material, beams, sample count, grids, optical train, and thermal "
        "boundaries; selects the full pulse map, enables transient soft-window gain, "
        "and disables multirate gain grouping."
    ),
    "当前是 L4.5 轴对称瞬态 Kerr：求解 E(f,r) 与完整相干时间波形，front → peak → tail 软窗增益也被保留；省略的是非零方位角空间模式，不是时间分箱。下面的轴对称有效性门限必须通过。": (
        "This is L4.5 axisymmetric transient Kerr: E(f,r), the full coherent temporal "
        "waveform, and front-to-peak-to-tail soft-window gain are resolved. The omitted "
        "degrees of freedom are nonzero azimuthal spatial modes, not temporal bins. The "
        "axisymmetric applicability gates below must pass."
    ),
    "径向分箱": "Radial Bins",
    "增益频率点": "Gain Frequency Samples",
    "L4.5 软窗增益直接使用原生相干频率网格，不走 L4 的粗增益重构。": (
        "L4.5 soft-window gain uses the native coherent frequency grid directly and "
        "does not use L4 coarse-gain reconstruction."
    ),
    "任务状态": "Task Status",
    "暂停求解": "Pause Solve",
    "继续求解": "Resume Solve",
    "停止并取消本次求解": "Stop and Cancel This Solve",
    "已暂停": "Paused",
    "正在计算": "Computing",
    "已完成": "Completed",
    "已取消": "Cancelled",
    "已失败": "Failed",
    "已完成点": "Completed Points",
    "暂停遍历": "Pause Sweep",
    "继续遍历": "Resume Sweep",
    "停止并取消本次遍历": "Stop and Cancel This Sweep",
    "尚无完整遍历点可下载。": "No completed sweep point is available to download yet.",
    "后台进程正在建立第一个遍历点。": (
        "The background process is preparing the first sweep point."
    ),
    "冻结当前点及整个遍历工作进程；已完成点仍可立即下载。": (
        "Freeze the current point and the entire sweep worker; completed points remain "
        "available for immediate download."
    ),
    "从当前点原有的数值状态继续，不重复已经完成的点。": (
        "Continue from the current point's existing numerical state without repeating "
        "completed points."
    ),
    "终止当前未完成点；已经落盘的完成点会继续保留并可下载。": (
        "Terminate the unfinished point; completed points already stored on disk remain "
        "available for download."
    ),
    "参数遍历已暂停：当前点被冻结；下方下载只包含已经完整结束并落盘的点。": (
        "The sweep is paused and the current point is frozen. The download below includes "
        "only points that finished and were stored on disk."
    ),
    "参数遍历在独立后台进程中运行；关闭或刷新网页不会取消任务。": (
        "The sweep runs in an independent background process; closing or refreshing the "
        "page will not cancel it."
    ),
    "参数遍历已完成，全部点均已持久化。": (
        "The parameter sweep is complete and every point has been persisted."
    ),
    "参数遍历已取消；已完成点仍可下载。": (
        "The parameter sweep was cancelled; completed points remain downloadable."
    ),
    "参数遍历异常停止：": "The parameter sweep stopped unexpectedly: ",
    "后台遍历进程失败": "The background sweep process failed",
    "第一个点尚未完成，因此目前还没有可下载的部分。": (
        "The first point has not finished, so no partial download is available yet."
    ),
    "下载已完成的部分": "Download Completed Portion",
    "只打包已经完成的点；当前暂停或计算中的点不会写入。完整三维结果从后端逐点读取，不会长期占用网页内存。": (
        "Package only completed points; the paused or in-flight point is excluded. Full "
        "3D results are read one at a time from backend storage and are not retained in "
        "browser memory."
    ),
    "参数遍历：已完整持久化 ": "Parameter sweep: fully persisted ",
    "已持久化 ": "Persisted ",
    " 点 · 最近完成点：": " points · Latest completed point: ",
    " 点": " points",
    "已有后台单点求解正在运行": "A background single-point solve is already running",
    "已有后台参数遍历正在运行": "A background parameter sweep is already running",
    "已有后台参数遍历；请先继续完成或取消遍历。当前表单修改只用于下一次运行。": (
        "A background parameter sweep already exists. Resume it to completion or cancel "
        "it first. Current form edits apply only to the next run."
    ),
    "这是唯一会因页面按钮操作而终止当前单点求解的控制。未完成结果不会归档。": (
        "This is the only page control that terminates the active single-point solve. "
        "Incomplete results are not archived."
    ),
    "最近一次单点求解已按明确操作取消；未完成状态没有覆盖更早的成功结果。配置快照：`": (
        "The latest single-point solve was explicitly cancelled. Its incomplete state did "
        "not replace the earlier successful result. Configuration snapshot: `"
    ),
    "**当前阶段：后台进程启动**　·　阶段进度 `—`": (
        "**Current Stage: Starting Background Process** · Stage Progress `—`"
    ),
    "等待首个求解事件": "Waiting for the First Solver Event",
    "等待首个事件": "Waiting for the First Event",
    "**当前物理状态**　后台工作进程已创建，正在载入配置快照。": (
        "**Current Physical State** The background worker has been created and is loading "
        "the configuration snapshot."
    ),
    "冻结独立工作进程；配置快照、进度和内存状态都会保留。": (
        "Freeze the independent worker process while retaining its configuration snapshot, "
        "progress, and in-memory state."
    ),
    "从原来的数值状态继续，不会重新初始化或更换配置。": (
        "Continue from the existing numerical state without reinitializing or changing the "
        "configuration."
    ),
    "求解已明确暂停；数值进程仍保留，只有继续或取消按钮会改变它。": (
        "The solve is explicitly paused. Its numerical process is retained; only Resume or "
        "Cancel will change it."
    ),
    "求解运行在独立进程中；修改表单、切换页面或生成完整配置不会停止、重启或改写这次任务。": (
        "The solve runs in an independent process. Editing the form, switching pages, or "
        "generating the full configuration will not stop, restart, or alter this task."
    ),
    "运行配置快照 SHA-256：`": "Run Configuration Snapshot SHA-256: `",
    "` · 后台 PID：`": "` · Background PID: `",
    "`。当前表单修改只供下一次运行使用；结果包固定使用这个快照。": (
        "`. Current form edits apply only to the next run; the result archive always uses "
        "this snapshot."
    ),
    "已有后台单点求解；请在主页面明确暂停、继续或取消。当前表单仍可编辑，修改只用于下一次运行。": (
        "A background single-point solve is already active. Use the explicit Pause, Resume, "
        "or Cancel controls in the main pane. The form remains editable and changes apply "
        "only to the next run."
    ),
    "后台求解：": "Background Solve: ",
    "最近一次完整求解进度": "Most Recent Complete Solve Progress",
    "上方百分比按真实模块推进；下方直接读取求解器正在计算的程次、晶体网格、周期/脉冲、残差与物理输出。趋势图刷新会限频，不参与物理解算。": (
        "The overall percentage follows real solver modules. The live panel reads "
        "the active pass, crystal cell, period/pulse, residuals, and physical output "
        "directly from the solver. Plot refreshes are throttled and never enter the "
        "physical calculation."
    ),
    "当前热—光遍历的全部已完成脉冲都画在同一张图中；脉宽和光谱只保留最新快照，轴向单元与子步只保留当前状态。上一遍历的完整事件留在后端日志，不会重复发送到浏览器；这些显示数据不参与物理解算。": (
        "Every completed pulse in the current thermal-optical traversal is shown in one "
        "chart. Only the latest pulse-width and spectrum snapshots and the current axial "
        "or substep state stay live; complete prior-traversal events remain in the backend "
        "log instead of being resent to the browser. These display data never enter the "
        "physical calculation."
    ),
    "当前热—光遍历的全部已完成脉冲都画在同一张图中；脉宽和光谱只保留最新快照。轴向单元、光学子步、Pump 采样和热线性迭代不发送到网页；页面只接收完整脉冲和固定点阶段结果，这些显示数据不参与物理解算。": (
        "Every completed pulse in the current thermal-optical traversal is shown in one "
        "chart, while only the latest pulse-width and spectrum snapshots are retained. "
        "Axial cells, optical substeps, Pump samples, and thermal linear iterations are not "
        "sent to the browser; the page receives only completed pulses and fixed-point "
        "milestones. These display data never enter the physical calculation."
    ),
    "当前点只接收完整脉冲和固定点阶段事件；界面最多每 5 秒读取一次。这只降低网页观测频率，不会改变求解器内部的物理分辨率。": (
        "The current point sends only completed-pulse and fixed-point milestone events. "
        "The page polls at most once every 5 seconds. This lowers only the browser "
        "observation frequency and does not change the solver's physical resolution."
    ),
    "本次遍历已完成的全部点": "All Completed Points in This Sweep",
    "#### 本次遍历已完成的全部点": "#### All Completed Points in This Sweep",
    "后台工作进程已创建，正在载入不可变配置快照。": (
        "The background worker has been created and is loading the immutable "
        "configuration snapshot."
    ),
    "当前阶段正在建立下一份物理量快照。": (
        "The current stage is preparing the next physical snapshot."
    ),
    "面板保留停止前最后一份成功的物理量快照。": (
        "The panel retains the last successful physical snapshot before the solve stopped."
    ),
    "Seed 输出历史": "Seed Output History",
    "完成第一个 Seed 传播后显示输出能量；不会逐事件新增表格。": (
        "Output energy appears after the first Seed propagation; no per-event table is appended."
    ),
    "最近一次输出光谱预览": "Latest Output Spectrum Preview",
    "光谱只在首个、每 5 个及最终脉冲更新，避免刷新拖慢求解。": (
        "The spectrum updates on the first, every fifth, and final pulse to keep display "
        "refreshes from slowing the solve."
    ),
    "已记录输出快照": "Recorded Output Snapshots",
    "周期图评估 / 已报告脉冲": "Periodic-Map Evaluations / Reported Pulses",
    "能量 (nJ)": "Energy (nJ)",
    "等效平均功率": "Equivalent Average Power",
    "平均功率 (W)": "Average Power (W)",
    "波长 (nm)": "Wavelength (nm)",
    "当前阶段正在建立状态，尚无新的物理量快照。": (
        "The current stage is establishing its state; no new physical snapshot is available yet."
    ),
    "实时趋势：正在积累当前阶段的第二个数据点。": (
        "Live trend: waiting for the second data point in this stage."
    ),
    "已经过：": "Completed stages: ",
    "正在启动第一个求解阶段": "Starting the first solver stage",
    "求解完成，总用时 ": "Solve complete; total elapsed time ",
    "上方进度保留了停止前最后一个成功进入的模块。": (
        "The progress panel retains the last module entered before the solve stopped."
    ),
    "警告": "Warning",
    "错误": "Error",
    "说明": "Note",
    " 光轴": " Optical Axis",
    " 激光源": " Laser Source",
    " 激光源<extra></extra>": " Laser Source<extra></extra>",
    " D4σ 光束体": " D4σ Beam Envelope",
    " D4σ 描边": " D4σ Outline",
    " 运行后 D4σ 采样面": " Post-Run D4σ Sampling Plane",
    " 运行后采样束心": " Post-Run Sampled Beam Centroid",
    " 空位": " Empty Slot",
    "增益晶体": "Gain Crystal",
    "晶体预留位置": "Reserved Crystal Position",
    "晶体边界": "Crystal Boundary",
    "透镜": "Lens",
    "反射镜": "Mirror",
    "晶体体场": "Crystal Volume Field",
    "局部 x (mm)": "Local x (mm)",
    "局部 y (mm)": "Local y (mm)",
    "相对共享晶体入口位置 (mm)": "Position Relative to Shared Crystal Entrance (mm)",
    "晶体对齐显示（长自由空间已压缩）": (
        "Crystal-Aligned View (Long Free-Space Segments Compressed)"
    ),
    "三维光学平台 · Pump + Seed 冷态 D4σ 光束体 · 当前编辑 ": (
        "3D Optical Bench · Pump + Seed Cold-State D4σ Beams · Editing "
    ),
    "三维光学平台 · 冷态光束体 + 运行后 D4σ 采样环 · 当前强调 ": (
        "3D Optical Bench · Cold-State Beams + Post-Run D4σ Sampling Rings · Highlighting "
    ),
    "三维空气调光 · 晶体位置仅作占位 · 当前编辑 ": (
        "3D Free-Space Alignment · Crystal Position Is a Placeholder · Editing "
    ),
    "真实光程 %{customdata[0]:.5g} mm<br>x 束心 %{y:.5g} mm<br>y 束心 %{z:.5g} mm<br>D4σ rx %{customdata[1]:.5g} mm<br>D4σ ry %{customdata[2]:.5g} mm<extra></extra>": (
        "Physical Path Length %{customdata[0]:.5g} mm<br>x Centroid %{y:.5g} mm<br>"
        "y Centroid %{z:.5g} mm<br>D4σ rx %{customdata[1]:.5g} mm<br>"
        "D4σ ry %{customdata[2]:.5g} mm<extra></extra>"
    ),
    " D4σ 光束体<br>真实光程 %{customdata[0]:.5g} mm<br>D4σ rx %{customdata[1]:.5g} mm<br>D4σ ry %{customdata[2]:.5g} mm<extra></extra>": (
        " D4σ Beam Envelope<br>Physical Path Length %{customdata[0]:.5g} mm<br>"
        "D4σ rx %{customdata[1]:.5g} mm<br>D4σ ry %{customdata[2]:.5g} mm<extra></extra>"
    ),
    "%{customdata[0]}<br>运行后 D4σ rx %{customdata[1]:.5g} mm<br>运行后 D4σ ry %{customdata[2]:.5g} mm<extra></extra>": (
        "%{customdata[0]}<br>Post-Run D4σ rx %{customdata[1]:.5g} mm<br>"
        "Post-Run D4σ ry %{customdata[2]:.5g} mm<extra></extra>"
    ),
    "晶体内 z %{customdata:.5g} mm<br>局部 x %{y:.5g} mm<br>局部 y %{z:.5g} mm<br>%{value:.5g} ": (
        "In-Crystal z %{customdata:.5g} mm<br>Local x %{y:.5g} mm<br>"
        "Local y %{z:.5g} mm<br>%{value:.5g} "
    ),
    "<br>晶体已取下；该段按外部介质传播<extra></extra>": (
        "<br>Crystal Removed; This Segment Propagates in the External Medium<extra></extra>"
    ),
    "<br>晶体第": "<br>Crystal Pass ",
    "程<br>": "<br>",
    "有效": "Valid",
    "有效（有诊断警告）": "Valid (With Diagnostic Warnings)",
    "完成但无效": "Completed but Invalid",
    "执行失败": "Execution Failed",
    "处理中": "Processing",
    "未知": "Unknown",
    "是": "Yes",
    "否": "No",
    "开": "On",
    "关": "Off",
    "运行状态": "Run Status",
    "最终输出平均功率": "Final Average Output Power",
    "输出平均功率": "Average Output Power",
    "输出种子能量": "Output Seed Energy",
    "吸收泵浦功率": "Absorbed Pump Power",
    "总热负载": "Total Heat Load",
    "总能量增益": "Total Energy Gain",
    "晶体最高温度": "Maximum Crystal Temperature",
    "晶体最大温升": "Maximum Crystal Temperature Rise",
    "最大热功率密度": "Maximum Volumetric Heat Load",
    "最终最大局部增益": "Final Maximum Local Gain",
    "正提取能量": "Positive Extracted Energy",
    "种子净提取能量": "Net Seed Energy Extraction",
    "种子脉冲能量": "Seed Pulse Energy",
    "通过晶体次数": "Crystal Pass Count",
    "重复频率": "Repetition Rate",
    "上能级寿命": "Upper-Level Lifetime",
    "信号发射截面": "Signal Emission Cross Section",
    "掺杂离子密度": "Dopant Ion Density",
    "晶体折射率": "Crystal Refractive Index",
    "晶体长度": "Crystal Length",
    "泵浦功率": "Pump Power",
    "泵浦吸收截面": "Pump Absorption Cross Section",
    "泵浦吸收比例": "Pump Absorption Fraction",
    "泵浦 x 腰斑": "Pump x Waist",
    "泵浦 y 腰斑": "Pump y Waist",
    "种子 x 腰斑": "Seed x Waist",
    "种子 y 腰斑": "Seed y Waist",
    "Pump 中心波长": "Pump Center Wavelength",
    "Seed 中心波长": "Seed Center Wavelength",
    "Pump x/y 腰斑统一缩放": "Pump x/y Waist Scale",
    "Seed x/y 腰斑统一缩放": "Seed x/y Waist Scale",
    "非相干发光面：半径 + NA": "Incoherent Emitter: Radius + NA",
    "Gaussian 中心加密": "Gaussian Center Refinement",
    "共享晶体入口 = 0": "Shared Crystal Entrance = 0",
    "本束 LASER 源面 = 0": "This Beam's LASER Source Plane = 0",
    "当前查看": "Current View",
    "切换工况点": "Select Case",
    "工况": "Case",
    "工况点": "Cases",
    "单次运行": "Single Run",
    "含完整场": "With Full Fields",
    "有效点": "Valid Cases",
    "失败/无效": "Failed / Invalid",
    "完整场": "Full Fields",
    "求解器警告": "Solver Warnings",
    "模型/显示说明": "Model / Display Notes",
    "验证失败": "Validation Failures",
    "验证警告": "Validation Warnings",
    "执行错误": "Execution Error",
    "全部工况数据表": "All Cases",
    "问题与说明中心": "Issues and Notes",
    "下载当前单点完整结果包（可在结果分析页重新载入）": (
        "Download Current Single-Case Archive (Reloadable in Result Analysis)"
    ),
    "包含配置、完整三维场、光学面、摘要及全部 warning/validation 诊断。这里只包含当前单点；二维参数网格请使用遍历结果区的“下载全部遍历结果包”。严格验证未通过的结果也会保留，并在重新载入时明确标记为诊断结果。": (
        "Includes the configuration, complete 3D fields, optical planes, summary, "
        "and all warning/validation diagnostics. This button contains only the "
        "current case; use 'Download Complete Sweep Archive' in the sweep results "
        "for a 2D grid. Results that fail strict validation are retained and clearly "
        "marked as diagnostic when reloaded."
    ),
    "#### 二维遍历切片曲线": "#### 2D Sweep Slice Curves",
    "二维遍历切片曲线": "2D Sweep Slice Curves",
    "切片曲线指标": "Slice-Curve Metric",
    "在二维图中包含验证无效格点的原始指标": (
        "Include Raw Metrics from Invalid Grid Points in 2D Plots"
    ),
    "探索模式默认包含这些已完成工况；它们会进入热图和连线，但有效性警告与逐点诊断仍会保留。": (
        "Exploratory mode includes these completed cases in heatmaps and connected "
        "curves by default, while preserving validation warnings and per-case diagnostics."
    ),
    "左图固定第二参数并扫描第一参数；右图固定第一参数并扫描第二参数。每条细线代表一个固定值。": (
        "The left plot fixes the second parameter and scans the first; the right plot "
        "fixes the first parameter and scans the second. Each thin line represents "
        "one fixed value."
    ),
    "曲线包含已完成但未通过有效性门的原始指标；图上保持普通连线，其有效性状态请以顶部警告和逐点诊断表为准。": (
        "The curves include raw metrics from completed cases that failed validation. "
        "They remain connected for readability; use the warning banner and per-case "
        "diagnostics for validity status."
    ),
    " 个二维格点运行完成但未通过有效性门；探索模式下可将其原始指标纳入热图和切片曲线，但它们仍不是通过有效性合同的定量结果。": (
        " 2D grid points completed but failed validation. Exploratory mode may include "
        "their raw metrics in heatmaps and slice curves, but they are not validated "
        "quantitative results."
    ),
    "支持本网站下载的单次结果包、一维遍历结果包和二维遍历结果包；大型遍历包只先读取标量索引，切换工况时才加载该点完整场。": (
        "Supports single-case, 1D-sweep, and 2D-sweep archives downloaded from this "
        "application. Large sweep archives initially load only the scalar index; full "
        "fields are restored only when a case is selected."
    ),
    "载入不会重跑求解器，也不会覆盖当前表单。大型包不会一次解压全部三维场。": (
        "Loading does not rerun the solver or overwrite the current form. Large archives "
        "do not decompress all 3D fields at once."
    ),
    "正在校验结果包并载入遍历索引……": (
        "Validating the result archive and loading its sweep index…"
    ),
    "大型结果包的数据源已不可用；请重新选择该结果包。": (
        "The large result archive source is no longer available; select the archive again."
    ),
    "正在按需恢复工况点 ": "Restoring full fields for case ",
    " 的完整场……": " on demand…",
    "工况点 ": "Case ",
    " 无法载入：": " could not be loaded: ",
    "\n\n这个结果包包含执行失败或物理有效性失败。下方总览保留了其它成功点，遍历图会连线显示已完成点的原始指标，但未通过合同的点不能作为可信定量结果。": (
        "\n\nThis archive contains execution failures or physical-validation failures. "
        "The overview retains other successful cases, and sweep plots connect raw "
        "metrics from completed cases; cases that failed validation are not reliable "
        "quantitative results."
    ),
    "固定 ": "Fixed ",
    "：扫描 ": ": Scan ",
    "（无效诊断）": " (Invalid Diagnostic)",
    "完成但无效（不连线）": "Completed but Invalid (Not Connected)",
    "查看运行/遍历开始前的原始基准配置": (
        "View Original Baseline Configuration"
    ),
    "查看这个点的完整配置": "View Complete Case Configuration",
    "这个点通过结果有效性检查。": "This case passed result validation.",
    "这个点通过阻断性合同，但包含需要审阅的诊断质量警告。": (
        "This case passed all blocking contracts but has diagnostic-quality "
        "warnings that require review."
    ),
    "没有记录到执行错误、validation 问题或运行 warning。": (
        "No execution errors, validation issues, or runtime warnings were recorded."
    ),
    "运行方式与求解 / 高级求解器": "Run Mode and Solver / Advanced Solvers",
    "one_way — Pump→布居→热→Seed（快速单向耦合）": (
        "one_way — Pump → Population → Thermal → Seed (Fast One-Way Coupling)"
    ),
    "放大器设置": "Amplifier Settings",
    "光路预览": "Optical Path Preview",
    "SEED 元件与真实光程": "SEED Elements and Physical Path Length",
    "PUMP 元件与真实光程": "PUMP Elements and Physical Path Length",
    "预览定义、坐标与边界": "Preview Definition, Coordinates, and Boundaries",
    "文件与配置": "Files and Configuration",
    "材料库（XML / JSON）": "Material Library (XML / JSON)",
    "可选：Commercial App 快捷方案": "Optional: Commercial App Shortcut",
    "推荐：速度 / 物理折中方案": "Recommended: Speed / Physics Tradeoff",
    "低维离散脉冲周期代理图": "Reduced Discrete-Pulse Periodic Proxy Map",
    "低维离散脉冲初值加速": "Reduced Discrete-Pulse Initializer",
    "周期映射模型": "Periodic Map Model",
    "代理时间采样 / 通量包": "Proxy Time Samples / Fluence Packets",
    "代理频谱采样": "Proxy Spectral Samples",
    "低维离散脉冲代理图": "Reduced Discrete-Pulse Proxy Map",
    "低维离散脉冲代理图 — 推荐中间档，处理强单脉冲耗尽": (
        "Reduced Discrete-Pulse Proxy Map — Recommended Intermediate Mode "
        "for Strong Single-Pulse Depletion"
    ),
    "代理输出脉冲能量 (nJ)": "Proxy Output Pulse Energy (nJ)",
    "离散周期映射初值": "Discrete Periodic-Map Initial State",
    "先用低维脉冲图靠近固定点，再由完整主脉冲收敛": (
        "Use the Reduced Pulse Map as an Initializer, then Converge the Full Pulse Map"
    ),
    "最终页面报告收敛固定点上的 Seed 输出；开启加速后横轴是周期映射评估次数，不是系统真实经历的连续脉冲数。选择低维初值时，最终仍必须由当前完整主脉冲满足相同的周期收敛条件。": (
        "The final page reports the Seed output at the converged fixed point. "
        "With acceleration, the horizontal axis counts periodic-map evaluations, "
        "not consecutive physical pulses. When the reduced initializer is selected, "
        "the configured full pulse must still satisfy the same periodic convergence criteria."
    ),
    "平均稳态只用作数值初值。严格 pulse_map 用当前完整 Seed 求解器闭合；低维脉冲图也只用作初值，随后仍由完整主脉冲逐周期闭合。含上转换/交叉弛豫时会安全跳过平均初值。": (
        "The average steady state is only a numerical initial guess. The strict "
        "pulse_map closes with the complete configured Seed solver; the reduced "
        "pulse map is also only an initializer, after which the complete pulse "
        "closes every cycle. The average initializer is safely skipped for "
        "upconversion or cross-relaxation."
    ),
    "初始化顺序：周期平均 → 轴对称相干 E(f,r) Kerr 低维脉冲图 → 当前完整主脉冲 pulse map。reduced 只负责加速靠近固定点；最终布居和输出必须由完整主脉冲按上面的容差收敛。": (
        "Initialization order: cycle average → axisymmetric coherent E(f,r) Kerr "
        "reduced pulse map → configured full-pulse map. The reduced stage only "
        "accelerates approach to the fixed point; the full pulse must converge "
        "the final population and output to the tolerance above."
    ),
    "初始化顺序：周期平均 → 低维整脉冲 FN/布居图 → 当前完整主脉冲 pulse map。reduced 结果不会作为最终结果。": (
        "Initialization order: cycle average → reduced whole-pulse FN/population "
        "map → configured full-pulse map. The reduced result is never reported "
        "as the final solution."
    ),
    "最终页面报告收敛固定点上的 Seed 输出；开启加速后横轴是周期映射评估次数，不是系统真实经历的连续脉冲数。低维代理图若与最终全场脉冲不一致，验证会明确失败，不会伪装成定量结果。": (
        "The final page reports the Seed output at the converged fixed point. "
        "With acceleration, the horizontal axis counts periodic-map evaluations, "
        "not consecutive physical pulses. If the reduced proxy disagrees with "
        "the final full-field pulse, validation fails explicitly instead of "
        "presenting the result as quantitative."
    ),
    "离散脉冲周期图要求瞬态布居。": (
        "A discrete-pulse periodic map requires transient populations."
    ),
    "先用周期平均得到初值，再用 1 个整脉冲通量包 × 至少 33 个频谱积分通道收敛离散 FN/布居周期图；最后只传播一次完整 E(x,y,t) Kerr 场，并强制检查代理图与全场结果的一致性。": (
        "First obtain an initial state from the cycle average, then converge a "
        "discrete FN/population map with one whole-pulse fluence packet and at "
        "least 33 spectral quadrature channels. Finally propagate the complete "
        "E(x,y,t) Kerr field once and enforce proxy-to-full-field consistency."
    ),
    "先用周期平均得到初值，再用轴对称相干 E(f,r) Kerr 场收敛离散周期图；它保留径向衍射、自聚焦和频谱变化，但省略非零方位角模。最终传播与周期代理的差异仍会强制门控。": (
        "First obtain an initial state from the cycle average, then converge the "
        "discrete periodic map with an axisymmetric coherent E(f,r) Kerr field. "
        "It retains radial diffraction, self-focusing, and spectral evolution "
        "but omits nonzero azimuthal modes. The final-propagation-to-proxy "
        "difference remains a mandatory gate."
    ),
    "平均稳态只用作数值初值。严格 pulse_map 用当前完整 Seed 求解器闭合；低维代理图用离散整脉冲 FN 闭合，并在最终全场脉冲后做误差门控。含上转换/交叉弛豫时会安全跳过平均初值。": (
        "The average steady state is only a numerical initial guess. Strict "
        "pulse_map closes with the complete configured Seed solver; the reduced "
        "proxy closes with discrete whole-pulse FN and applies an error gate after "
        "the final full-field pulse. The average initializer is safely skipped "
        "for upconversion or cross-relaxation."
    ),
    "离散脉冲周期图必须把每次 Seed 受激提取反向提交到布居，因此 gain_saturation=true 由周期合同锁定。": (
        "A discrete-pulse periodic map must commit every stimulated Seed "
        "extraction back to the population, so the periodic contract locks "
        "gain_saturation=true."
    ),
    "代理周期图已收敛": "Proxy Periodic Map Converged",
    "代理周期图未收敛": "Proxy Periodic Map Not Converged",
    "周期固定点由所选的降阶脉冲代理图收敛；页面输出来自同一周期边界布居上的当前相干传播。上面的两项代理—最终传播差异是该加速闭合能否定量使用的强制门。": (
        "The periodic fixed point is converged with the selected reduced pulse "
        "proxy; the page output comes from the configured coherent propagation "
        "at the same periodic-boundary population. The two proxy-to-final-"
        "propagation differences above are mandatory quantitative-use gates."
    ),
    "查看低维代理周期稳态收敛": (
        "View Reduced-Proxy Periodic Steady-State Convergence"
    ),
    "低维 Pump + Seed 周期代理图评估次数": (
        "Reduced Pump + Seed Periodic-Proxy Evaluations"
    ),
    "周期代理模型": "Periodic Proxy Model",
    "周期代理时间采样 / 通量包": "Periodic Proxy Time Samples / Fluence Packets",
    "个": "items",
    "周期代理频谱采样": "Periodic Proxy Spectral Samples",
    "代理图与全场脉后布居差": (
        "Proxy-to-Full-Field Post-Pulse Population Difference"
    ),
    "代理图与全场输出能量差": (
        "Proxy-to-Full-Field Output-Energy Difference"
    ),
    " 次自洽迭代；": " self-consistency iterations; ",
    "它只提供初值，随后由低维离散脉冲代理图闭合并接受全场误差门控。": (
        "It only supplies the initial state; the reduced discrete-pulse proxy then "
        "closes the map and is subjected to a full-field error gate."
    ),
    "它只提供初值，最终解仍满足精确周期映射。": (
        "It only supplies the initial state; the final solution still satisfies "
        "the exact periodic map."
    ),
    "周期平均初值求解失败，已安全退回设定初始布居；": (
        "The cycle-average initializer failed and safely fell back to the "
        "configured initial population; "
    ),
    "仍继续运行。原因：": " continues running. Reason: ",
    "精确周期映射": "Exact Periodic Map",
    "先用周期平均作初值，再用所选的低维整脉冲 FN 或轴对称相干 Kerr 代理图闭合离散周期布居；最终运行一次当前配置的相干传播，并以代理—最终传播差异作为定量门。": (
        "Use the cycle average as an initializer, then close the discrete periodic "
        "population with the selected reduced whole-pulse FN or axisymmetric "
        "coherent-Kerr proxy. Run the configured coherent propagation once at the "
        "end and use the proxy-to-final-propagation difference as a quantitative "
        "gate."
    ),
    "低维脉冲代理图不传播时域自由载流子状态；已切换为完整逐脉冲周期固定点。": (
        "The reduced pulse proxy does not propagate a time-domain free-carrier "
        "state, so the solver switched to the complete pulse-by-pulse periodic "
        "fixed point."
    ),
    "**全场 Kerr 中间档**": "**Full-Field Kerr Intermediate Mode**",
    "应用全场 Kerr 中间档（推荐先跑）": (
        "Apply Full-Field Kerr Intermediate Mode (Recommended First Run)"
    ),
    "不改材料、光束、网格、光路、热边界或频谱数据。细微的前沿/尾沿增益耗尽需再用小网格的完整软窗模式做参考核对。": (
        "Does not change material, beam, grid, optical path, thermal boundary, "
        "or spectrum data. Check subtle front-to-tail gain depletion against the "
        "full soft-window mode on a small grid."
    ),
    "晶体外用逐波长 ABCD 到增益入口；晶体内保留完整复场 E(f,y,x)、空间衍射/干涉、色散和局域时间域 Kerr；周期布居先用低维离散脉冲图建立初值，再由完整主脉冲固定点收敛；脉内增益使用整脉冲 FN，并让 FN/布居按受增益指数约束的较粗步长更新，而 Kerr/色散仍走细步。它不会把光束反复投影回高斯；reduced 阶段只负责靠近解，最终布居和输出始终由完整主脉冲按配置容差闭合。": (
        "Uses per-wavelength ABCD transport outside the crystal up to the gain "
        "entrance. Inside the crystal it retains the complete complex field "
        "E(f,y,x), spatial diffraction and interference, dispersion, and local "
        "time-domain Kerr propagation. A reduced discrete-pulse map initializes "
        "the population before the configured full-pulse map closes the fixed "
        "point; intrapulse gain uses a whole-pulse FN closure. FN "
        "gain and population are updated on a coarser gain-exponent-bounded "
        "cadence while Kerr and dispersion retain their fine steps. It does not "
        "repeatedly project the beam back to a Gaussian. The reduced stage only "
        "accelerates approach to the solution; final population and output are "
        "always closed by the configured full pulse at the configured tolerance."
    ),
    "**Commercial App Gaussian-q 快捷方案**": "**Commercial App Gaussian-q Shortcut**",
    "当前为 Commercial App 式独立 HG Gaussian-q 通道：同一波长的各模态先汇总局部通量、同步竞争共享布居，再分别推进束宽和曲率；模态之间最终按非相干强度求和，不解析完整 x-y 复场衍射。": (
        "Commercial App-style independent HG Gaussian-q channels are active. Modes at "
        "one wavelength first combine local fluence and compete simultaneously "
        "for one shared population, then advance their beam sizes and curvatures "
        "independently. Modes are finally summed incoherently; complete x-y "
        "complex-field diffraction is not resolved."
    ),
    "Commercial App 最大总 HG 阶数 m+n": "Commercial App Maximum Total HG Order m+n",
    "只替换 Seed：HG 独立 q + 共享布居（推荐公平对比）": (
        "Replace Seed Only: Independent HG q + Shared Population "
        "(Recommended Fair Comparison)"
    ),
    "公平对比档：只把 Seed 传播降为独立 HG Gaussian-q 通道；": (
        "Fair-comparison mode: reduce only Seed propagation to independent "
        "HG Gaussian-q channels; "
    ),
    "公平对比档：只把 Seed 传播降为 Gaussian-q + 一维 effective-area SPM；": (
        "Fair-comparison mode: reduce only Seed propagation to Gaussian-q "
        "plus one-dimensional effective-area SPM; "
    ),
    "Pump、布居、周期工作点、热模型和热光耦合保持当前配置。适合先和 JOSA/相干 L 系列比较速度与输出差异。": (
        "the Pump, population, periodic operating point, thermal model, and "
        "thermo-optical coupling retain the current configuration. Use it "
        "first to compare speed and output against JOSA or the coherent L family."
    ),
    "Commercial App 的 transverse mode order 按总阶数解释；设为 2 时传播 HG00、HG10、HG01、HG20、HG11、HG02。": (
        "Commercial App transverse mode order is interpreted as total order. A value of "
        "2 propagates HG00, HG10, HG01, HG20, HG11, and HG02."
    ),
    "Commercial App Fast 为每个 HG 模态独立推进 Gaussian q 和能量；同一波长的所有模态先合计局部通量、只更新一次共享布居，再按局部输入比例分配输出，最后按非相干强度求和。当前共传播 ": (
        "Commercial App Fast independently advances Gaussian q and energy for every HG "
        "mode. All modes at one wavelength first combine local fluence, update "
        "the shared population once, distribute output in proportion to local "
        "input, and are finally summed incoherently. The current basis propagates "
    ),
    " 个模态，不使用 x-y FFT。": " modes without an x-y FFT.",
    "把整脉冲光谱能量与横向 Gaussian q 参数耦合；每层增益照常计算，热场、Kerr 和增益导引只保留等效二次矩。可选把互不相干 HG 模态作为独立 q 通道并共享局部布居，也可用一维 effective-area SPM 输出展宽。": (
        "Couples whole-pulse spectral energy to transverse Gaussian q parameters. "
        "Every layer still computes gain, while thermal, Kerr, and gain-guiding "
        "effects retain only equivalent quadratic moments. Optional incoherent "
        "HG modes use independent q channels with a shared local population; an "
        "alternative 1D effective-area SPM closure can report broadening."
    ),
    "把 HG 模态展开为独立 Gaussian-q/能量通道；同一波长的模态在每个时间与 z 步先汇总局部通量，只对共享布居提交一次受激跃迁，再按局部输入比例分配输出。热相位、Kerr 相位与增益导引仍投影回各模态自己的二阶矩。": (
        "Expands HG modes into independent Gaussian-q and energy channels. At "
        "each time and z step, modes of one wavelength first combine local "
        "fluence, submit one stimulated-transition update to the shared "
        "population, and distribute output by local input fraction. Thermal "
        "phase, Kerr phase, and gain guiding remain projected onto each mode's "
        "own second moments."
    ),
    "Commercial App Fast 为每个 HG 通道独立推进 Gaussian q；其他非相干 HG 后端使用 FFT。": (
        "Commercial App Fast advances an independent Gaussian q for every HG channel; "
        "other incoherent-HG backends use FFT propagation."
    ),
    "Commercial App Fast 的 HG 共享布居路径当前使用固定脉宽/有序增益时间箱。": (
        "The Commercial App Fast shared-population HG path currently uses a fixed pulse "
        "duration or ordered gain-time bins."
    ),
    "这里的 Commercial App Fast 是本程序实现的 Commercial App 风格降阶求解器，不是调用或复刻 Commercial App 可执行程序。它仍读取同一套材料截面、Pump、真实能级布居、光路和热场；主要把 Seed 横向复场降为共享的 Gaussian q 参数，从而用较低成本保留放大、热透镜、Kerr 透镜和增益导引的一阶影响。": (
        "Commercial App Fast is an Commercial App-style reduced-order solver implemented in this "
        "program; it neither invokes nor reproduces the Commercial App executable. It "
        "still reads the same material cross sections, Pump, physical level "
        "populations, optical path, and thermal field. Its principal reduction "
        "is to represent the transverse Seed field with shared Gaussian q "
        "parameters, retaining first-order amplification, thermal-lens, Kerr-"
        "lens, and gain-guiding effects at lower cost."
    ),
    "**它怎样整合现有多物理链路**": "**How It Integrates the Existing Multiphysics Chain**",
    "1. **Pump → 布居**：先由当前 Pump 和速率方程产生三维上下能级布居。  \n2. **逐层增益**：每个晶体层仍按材料截面和局部布居计算光谱增益与受激耗尽；开启脉内分箱后，front → peak → tail 共用并依次消耗同一份布居。  \n3. **横向降阶**：不保存完整 `E(x,y,t)`；把热相位、Kerr 和增益导引在光束附近拟合为局部二次项，再通过 ABCD/Gaussian-q 推进束宽和曲率。  \n4. **可选一维相干包络**：effective-area SPM 只沿时间/频率做一维 FFT，可把粗粒度光谱变化回灌到后续增益层，但不解析横向不同位置的 SPM。": (
        "1. **Pump to population**: The current Pump and rate equations first "
        "produce three-dimensional upper- and lower-level populations.  \n"
        "2. **Layerwise gain**: Every crystal layer still computes spectral "
        "gain and stimulated depletion from material cross sections and local "
        "populations. With intrapulse bins enabled, front, peak, and tail share "
        "and sequentially consume the same population.  \n"
        "3. **Transverse reduction**: The complete `E(x,y,t)` field is not "
        "stored. Thermal phase, Kerr, and gain guiding are fitted near the beam "
        "as local quadratic terms; ABCD/Gaussian-q transport then advances beam "
        "size and curvature.  \n"
        "4. **Optional 1D coherent envelope**: Effective-area SPM uses a one-"
        "dimensional time/frequency FFT and can feed coarse spectral changes "
        "back into later gain layers, but it does not resolve spatially varying "
        "SPM."
    ),
    "**加速来自哪里**": "**Where the Speedup Comes From**",
    "降阶项": "Reduced Component",
    "省掉的主要成本": "Main Avoided Cost",
    "横向 Seed 场": "Transverse Seed Field",
    "每个波长/层只推进 qx、qy 与能量": "Advance Only qx, qy, and Energy per Wavelength/Layer",
    "大量 x-y 复场 FFT、存储和逐时间片衍射": "Large x-y Complex-Field FFTs, Storage, and Per-Time-Slice Diffraction",
    "脉冲相干演化": "Coherent Pulse Evolution",
    "固定脉宽，或单独的一维 effective-area FFT": "Fixed Pulse Duration or a Separate 1D Effective-Area FFT",
    "完整 E(f,y,x) 的三维相干传播": "Three-Dimensional Coherent Propagation of the Complete E(f,y,x)",
    "热/Kerr/增益导引": "Thermal/Kerr/Gain Guiding",
    "局部抛物线/二阶矩等效透镜": "Local Parabolic/Second-Moment Equivalent Lens",
    "任意高阶波前和热点结构的全场更新": "Full-Field Updates of Arbitrary Higher-Order Wavefront and Hot-Spot Structure",
    "因此速度提升取决于 Nx×Ny、频率/时间采样数和晶体 Nz；它不是靠跳过材料增益计算，而是减少 Seed 场需要传播和保存的自由度。": (
        "The speedup therefore depends on Nx by Ny, the frequency/time sample "
        "count, and crystal Nz. It does not skip material-gain calculations; it "
        "reduces the Seed-field degrees of freedom that must be propagated and "
        "stored."
    ),
    "公平对比档：只把 Seed 传播降为 Gaussian-q + 一维 effective-area SPM；Pump、布居、周期工作点、热模型和热光耦合保持当前配置。适合先和 JOSA/相干 L 系列比较速度与输出差异。": (
        "Fair-comparison mode: only Seed propagation is reduced to Gaussian-q "
        "plus one-dimensional effective-area SPM. The Pump, population, periodic "
        "operating point, thermal model, and thermo-optical coupling retain the "
        "current configuration. Use it first to compare speed and output against "
        "JOSA or the coherent L family."
    ),
    "全链路极速档：除 Gaussian-q 外，还切换 cycle_average、one_way 热光耦合并关闭部分记录/焦散诊断。它适合参数扫描和趋势判断；因为周期工作点也改变，不能把绝对输出差异全部归因于 Seed 传播。": (
        "Full-chain maximum-speed mode: in addition to Gaussian-q, it selects "
        "cycle_average and one_way thermo-optical coupling and disables some "
        "recording and caustic diagnostics. It is intended for parameter scans "
        "and trend estimates. Because it also changes the periodic operating "
        "point, absolute output differences cannot be attributed solely to Seed "
        "propagation."
    ),
    "建议工作流：Commercial App Fast 用于快速扫描和对齐 Commercial App 风格结果；强脉内人口竞争用 JOSA 复核；需要频率间相干相位、色散、Kerr/SPM 或压缩质量时，再用相干 L / coherent_cpa 复核。不存在对所有问题都最高精度的单一后端。": (
        "Recommended workflow: use Commercial App Fast for rapid scans and Commercial App-style "
        "result alignment; verify strong intrapulse population competition with "
        "JOSA; and verify inter-frequency coherent phase, dispersion, Kerr/SPM, "
        "or compression quality with coherent L / coherent_cpa. No single "
        "backend has the highest fidelity for every problem."
    ),
    "全场 Kerr 中间档已启用：当前关闭逐软窗增益，整脉冲 FN 使用受控多速率更新，但仍传播完整 E(f,y,x)；这不是 Commercial App 高斯投影。": (
        "Full-field Kerr intermediate mode is active: soft-window gain is "
        "disabled and whole-pulse FN uses guarded multirate updates, but the "
        "complete E(f,y,x) field is still propagated. This is not an Commercial App "
        "Gaussian projection."
    ),
    "Commercial App SPM→增益回灌频率分箱": "Commercial App SPM-to-Gain Feedback Frequency Bins",
    "这些分箱覆盖完整相干 FFT 频率范围，而不是只覆盖输入非零频谱。这样 Kerr 新产生的边带不会被塞进最近的边缘增益通道；默认 65 仍远少于完整 x-y-t coherent CPA。": (
        "These bins cover the complete coherent FFT frequency span, not only "
        "the nonzero input spectrum. Kerr-generated sidebands therefore do not "
        "get forced into the nearest edge gain channel; the default 65 bins "
        "remain far fewer than a complete x-y-t coherent CPA field."
    ),
    "传播模型适用范围": "Propagation Model Applicability",
    "A. 脉冲、重复频率与 Fourier 输入": "A. Pulse, Repetition Rate, and Fourier Input",
    "相位、GDD/TOD与双强度恢复": "Phase, GDD/TOD, and Dual-Intensity Retrieval",
    "Fourier 采样与求解分箱": "Fourier Sampling and Solver Bins",
    "重建/输入光谱相位预览与下载": "Reconstructed/Input Spectral Phase Preview and Download",
    "B. Seed 增益输运（时间模型 + 空间闭合）": "B. Seed Gain Transport (Temporal Model + Spatial Closure)",
    "C. Jones 偏振与径向/切向光弹双折射": "C. Jones Polarization and Radial/Tangential Photoelastic Birefringence",
    "D. 空间光束、M² 与横模": "D. Spatial Beam, M², and Transverse Modes",
    "空间—光谱耦合（不同波长的腰斑/焦点/中心/角度）": "Spatio-Spectral Coupling (Wavelength-Dependent Waist/Focus/Center/Angle)",
    "E. Seed 独立光路与半径探针": "E. Independent Seed Optical Path and Radius Probe",
    "多程与光学平台": "Multipass Layout and Optical Bench",
    "启用多程放大": "Enable Multipass Amplification",
    (
        "关闭时固定为单程。开启后才显示多程模板与参数；推荐的物理模板会由"
        "真实传播时间和三维光斑位置自动产生重叠/竞争，不提供人为重叠开关。"
    ): (
        "When disabled, the model is fixed to one pass. Enabling it reveals "
        "multipass templates and parameters. The recommended physical templates "
        "derive overlap and competition from propagation time and 3D spot location; "
        "there is no manual overlap switch."
    ),
    (
        "当前为单程：只进行一次晶体命中，因此没有程间镜面、延时或重叠参数。"
        "如需多程，请在上方勾选“启用多程放大”。"
    ): (
        "The current model is single-pass: it has one crystal hit and therefore no "
        "inter-pass mirror, delay, or overlap parameters. Enable Multipass "
        "Amplification above to configure multiple passes."
    ),
    "三维掺杂浓度分布（可选）": "3D Dopant Concentration Profile (Optional)",
    "热、应力、色散与非线性": "Thermal, Stress, Dispersion, and Nonlinear Effects",
    "三维场抛物拟合诊断": "3D Field Parabolic-Fit Diagnostics",
    "参数遍历（可选工具）": "Parameter Sweep (Optional Tool)",
    "固定计算周期数": "Fixed Number of Calculation Cycles",
    "旁轴后端最大合成倾角（°）": "Maximum Combined Tilt for Paraxial Backends (°)",
    "FFT 倾斜载波最大 Nyquist 占比": "Maximum Nyquist Fraction for FFT Tilt Carrier",
    "FFT 线程数（0=全部可用核）": "FFT Threads (0 = All Available Cores)",
    "启用多线程 FFT 的最小数组元素数": "Minimum Array Elements for Multithreaded FFT",
    "泵浦功率 (W)": "Pump Power (W)",
    "中心波长 (m)": "Center Wavelength (m)",
    "1/e² 半径 wx (m)": "1/e² Radius wx (m)",
    "1/e² 半径 wy (m)": "1/e² Radius wy (m)",
    "内部超高斯 order x（高斯=1）": "Internal Super-Gaussian Order x (Gaussian = 1)",
    "内部超高斯 order y（高斯=1）": "Internal Super-Gaussian Order y (Gaussian = 1)",
    "几何泵浦效率": "Geometric Pump Efficiency",
    "束腰相对 PUMP LASER 源面 z (m)": "Waist z Relative to PUMP LASER Source Plane (m)",
    "腰斑相对 SEED LASER 源面 z (m)": "Waist z Relative to SEED LASER Source Plane (m)",
    "Pump 中心 offset x (m)": "Pump Center Offset x (m)",
    "Pump 中心 offset y (m)": "Pump Center Offset y (m)",
    "泵浦脉宽 FWHM (s)": "Pump Pulse Width FWHM (s)",
    "泵浦脉冲中心延迟 (s)": "Pump Pulse-Center Delay (s)",
    "从当前 Pump/布居时间窗口起点量到脉冲中心。重复运行时按周期回卷；重复频率为 0 的单次瞬态不会把窗口末端回卷到开头。": (
        "Measured from the start of the current Pump/population time window to "
        "the pulse center. Repetitive runs wrap it periodically; a single "
        "transient at zero repetition rate does not wrap the end of the window "
        "back to its start."
    ),
    "Pump 超高斯时间 order": "Pump Temporal Super-Gaussian Order",
    "文件周期覆盖合同": "File Period-Coverage Contract",
    "zero_outside 在文件时间范围外取 0；cover_period 要求文件显式覆盖完整 [0, 周期]。": (
        "zero_outside sets the waveform to zero beyond the file's time range; "
        "cover_period requires the file to explicitly cover the complete "
        "[0, period] interval."
    ),
    "载流子生成效率": "Free-Carrier Generation Efficiency",
    "强场损耗物理解读": "Strong-Field Loss Physical Interpretation",
    "β 证据状态": "Beta Evidence Status",
    "β 文献/数据引用": "Beta Literature/Data Reference",
    "β 测量样品编号": "Beta Measurement Sample ID",
    "宿主带隙 (eV)": "Host Bandgap (eV)",
    "阶数判据参考波长 (m)": "Order-Threshold Reference Wavelength (m)",
    "这里描述 β 的物理含义，不改变 dI/dz=-βI^K 的数值方程。phenomenological_sensitivity 只能用于敏感性扫描，不能作为 TPA/MPA 机制证据。": (
        "This describes the physical interpretation of beta without changing the "
        "numerical equation dI/dz=-beta*I^K. phenomenological_sensitivity is for "
        "sensitivity scans only and is not evidence for a TPA/MPA mechanism."
    ),
    "unverified/provisional_sensitivity 不支持定量机制结论；literature 必须填写可追溯原始文献，sample_measured 还必须填写样品编号。": (
        "unverified/provisional_sensitivity cannot support a quantitative mechanism "
        "claim; literature requires a traceable primary reference, and sample_measured "
        "also requires a sample identifier."
    ),
    "β、载流子寿命和 FCA 截面必须来自同一材料/样品、波长和脉宽条件的测量或可追溯文献。宿主带隙只能判断带间最低阶数，不能排除杂质、缺陷或顺序吸收；未验证的 phenomenological_sensitivity 只可用于敏感性扫描。": (
        "Beta, carrier lifetime, and FCA cross section must come from measurements or "
        "traceable literature for the same material/sample, wavelength, and pulse "
        "duration. The host bandgap only sets the minimum interband order and cannot "
        "exclude impurity, defect, or sequential absorption; an unverified "
        "phenomenological_sensitivity setting is for sensitivity scans only."
    ),
    "每份 K 光子吸收能量产生可存活电子-空穴对的量子效率；它直接缩放速率方程的生成项，不等同于不可逆损耗转热比例。": (
        "Quantum efficiency for producing surviving electron-hole pairs from "
        "each K-photon absorption energy packet. It directly scales the rate-"
        "equation source and is distinct from the irreversible-loss heat fraction."
    ),
    "Seed 横向 order x（1=Gaussian）": "Seed Transverse Order x (1 = Gaussian)",
    "Seed 横向 order y（1=Gaussian）": "Seed Transverse Order y (1 = Gaussian)",
    "Seed Top-hat 软边 order x（>1）": "Seed Top-Hat Soft-Edge Order x (> 1)",
    "Seed Top-hat 软边 order y（>1）": "Seed Top-Hat Soft-Edge Order y (> 1)",
    "Top-hat 在数值上使用高阶超高斯软边；order≤1 的旧配置会按 20 执行，表单在这里显式显示为 20。": (
        "The numerical top-hat uses a high-order super-Gaussian soft edge. "
        "Legacy order values <= 1 execute as 20 and are shown explicitly as 20 here."
    ),
    "直接指定环形幂指数": "Specify the Annular Power Exponent Directly",
    "强度为 (r²)^a exp(-2r²)。直接指定 a 时，峰值椭圆半径为 sqrt(a/2)；关闭时用峰值半径比例自动计算 a。": (
        "Intensity is (r²)^a exp(-2r²). When a is specified directly, the "
        "peak elliptical radius is sqrt(a/2); otherwise a is derived from the peak-radius ratio."
    ),
    "环形幂指数 a": "Annular Power Exponent a",
    "当前环峰位于归一化椭圆半径 r/w = {value}。": (
        "Current annular peak: normalized elliptical radius r/w = {value}."
    ),
    "环峰半径 / 1/e² 半径": "Annular Peak Radius / 1/e² Radius",
    "求解器使用 a=2·(环峰半径比例)²，0 会退化为中心峰。": (
        "The solver uses a = 2·(annular peak-radius ratio)²; zero reduces to a centered peak."
    ),
    "显式覆盖源面波前曲率": "Explicitly Override Source-Plane Wavefront Curvature",
    "开启后，这个曲率半径优先于上面的焦点位置来定义源面二次相位；正负号区分发散/会聚波前。": (
        "When enabled, this curvature radius takes precedence over the focus "
        "position above when defining source-plane quadratic phase; its sign distinguishes diverging and converging wavefronts."
    ),
    "Seed 源面波前曲率半径 (m)": "Seed Source-Plane Wavefront Curvature Radius (m)",
    "非零有限值；它会覆盖由 waist focus 推出的源面曲率。": (
        "A finite nonzero value; it overrides the source-plane curvature inferred from the waist focus."
    ),
    "当前焦点位置仍决定源面光斑尺度，但源面二次相位由显式曲率半径决定。": (
        "The focus position still determines the source-plane spot scale, but explicit curvature now determines its quadratic phase."
    ),
    "实际强度指数为 exp[-2|x/wx|^(2·order)]；profile=gaussian 是兼容别名，真实形状仍由这里的 order 决定。": (
        "The actual intensity exponent is exp[-2|x/wx|^(2·order)]; "
        "profile=gaussian is a compatibility alias, while this order still "
        "determines the physical shape."
    ),
    "Seed 腰斑中心 x (m)": "Seed Waist Center x (m)",
    "Seed 腰斑中心 y (m)": "Seed Waist Center y (m)",
    "Seed 横向椭圆旋转 (deg)": "Seed Transverse-Ellipse Rotation (deg)",
    "绕光轴旋转横向椭圆/超高斯主轴；不会改变光轴传播方向。": (
        "Rotate the transverse ellipse/super-Gaussian principal axes about the "
        "beam axis; this does not change the beam-axis direction."
    ),
    "Pump x / y 使用不同焦点位置（像散光束）": (
        "Use Different Pump x/y Focus Positions (Astigmatic Beam)"
    ),
    "允许 Pump 的 x、y 两轴拥有不同焦点位置；关闭时两轴共用同一焦面。": (
        "Allow the Pump x and y axes to have different focus positions; when "
        "disabled, both axes share one focal plane."
    ),
    "Pump 焦斑中心 x (m)": "Pump Focus Center x (m)",
    "Pump 焦斑中心 y (m)": "Pump Focus Center y (m)",
    "Pump 横向椭圆旋转 (deg)": "Pump Transverse-Ellipse Rotation (deg)",
    "Pump 光轴倾斜 x (deg)": "Pump Beam-Axis Tilt x (deg)",
    "Pump 光轴倾斜 y (deg)": "Pump Beam-Axis Tilt y (deg)",
    "这是源光轴方向，不是发散角；gain_medium 元件中的入射角另外决定晶体内斜程。": (
        "This is the source beam-axis direction, not its divergence; the "
        "incidence angle on each gain_medium element separately determines the "
        "oblique path inside the crystal."
    ),
    "每次晶体命中后翻转 x 场": "Flip the x Field after Each Crystal Hit",
    "在相邻晶体命中之间执行一次明确的 x 坐标成像反转；仅用于尚未由光路元件描述的真实翻转。": (
        "Apply one explicit x-coordinate image inversion between adjacent "
        "crystal hits; use it only for a real inversion not already represented "
        "by optical-path elements."
    ),
    "只在未被光路元件显式描述的成像翻转确实存在时开启；若 relay/坐标变换已建模，不要重复翻转。": (
        "Enable only for a real image inversion that is not explicitly described "
        "by optical elements; do not apply it again when a relay/coordinate "
        "transform is already modeled."
    ),
    "每个程间未建模附加延时 (s)": "Unmodeled Additional Delay per Interpass (s)",
    "只填写光路 JSON 未覆盖的额外飞行/电子延时；它会进入程间布居和自由载流子弛豫，也进入周期时间闭合。": (
        "Enter only extra flight/electronic delay not covered by the optical-path "
        "JSON. It contributes to interpass population and free-carrier relaxation "
        "and to cycle-time closure."
    ),
    "从设定的 t=0 初始布居开始": "Start from the Configured Population at t=0",
    "自动 — 长寿命/高重复频率时先求平均稳态": (
        "Auto — Solve the Average Steady State First for Long Lifetimes / High Repetition Rates"
    ),
    "长寿命/高重复频率时先求平均稳态": (
        "Solve the average steady state first for long lifetimes / high repetition rates"
    ),
    "严格使用配置的初始能级布居；可另外指定第一个 Seed 相对 t=0 的到达时间。": (
        "Use the configured initial level populations exactly; the arrival time "
        "of the first Seed relative to t=0 may be specified separately."
    ),
    "只在平均稳态适用且能安全收敛时用它初始化，否则保留配置初态；有限脉冲本身仍逐周期计算。": (
        "Use the average steady state only when it is applicable and converges "
        "safely; otherwise retain the configured initial state. The finite "
        "pulses are still calculated cycle by cycle."
    ),
    "强制先求 Pump+Seed 周期平均布居作为数值初值，再从该状态计算指定数量的离散脉冲。": (
        "First solve the Pump+Seed cycle-average population as a numerical "
        "initializer, then calculate the requested number of discrete pulses "
        "from that state."
    ),
    "先求周期平均工作点再计算有限脉冲串": (
        "Solve the Cycle-Average Operating Point before the Finite Pulse Train"
    ),
    "有限脉冲串初始布居": "Finite-Pulse-Train Initial Population",
    "平均稳态只决定第一个有限周期开始前的布居，不把有限脉冲串伪装成已收敛周期稳态。": (
        "The average steady state only initializes the population before the "
        "first finite cycle; it does not present the finite train as a converged "
        "periodic steady state."
    ),
    "选择平均稳态初值时，首脉冲延时没有独立 t=0 物理意义，已固定为 null。": (
        "With an average-steady-state initializer, first-pulse delay has no "
        "independent t=0 meaning and is fixed to null."
    ),
    "Pump/布居每周期时间采样（DMA）": "Pump/Population Time Samples per Cycle (DMA)",
    "泵浦光谱 FWHM (m)": "Pump Spectrum FWHM (m)",
    "这是所选光谱密度自变量上的强度谱 FWHM。super_gaussian 会按 order 重新定标，因此改变 order 不改变这里输入的半高全宽。": (
        "This is the intensity-spectrum FWHM in the selected spectral-density "
        "coordinate. A super-Gaussian is rescaled with its order, so changing "
        "the order does not change the entered half-maximum full width."
    ),
    "泵浦光谱采样": "Pump Spectral Samples",
    "PUMP LASER 到第一个元件的距离 (m)": "PUMP LASER to First Element Distance (m)",
    "SEED LASER 到第一个元件的距离 (m)": "SEED LASER to First Element Distance (m)",
    "与前一元件距离 (m)": "Distance from Previous Element (m)",
    "上一元件→GAIN 入射面距离 (m)": "Previous Element → GAIN Input Face Distance (m)",
    "元件附加群延迟 (s)": "Element Group-Delay Offset (s)",
    "晶体入射角 x (deg)": "Crystal Incidence Angle x (deg)",
    "晶体入射角 y (deg)": "Crystal Incidence Angle y (deg)",
    "距 PUMP LASER 源面的累计光程 (mm)": "Accumulated Path Length from PUMP LASER Source Plane (mm)",
    "距 SEED LASER 源面的累计光程 (mm)": "Accumulated Path Length from SEED LASER Source Plane (mm)",
    "Seed 重复频率 (Hz；0=单次)": "Seed Repetition Rate (Hz; 0 = Single Shot)",
    "时间强度 FWHM (s)": "Temporal Intensity FWHM (s)",
    "单脉冲总能量 (J)": "Total Pulse Energy (J)",
    "实时 FFT 采样": "Interactive FFT Samples",
    "时间求解分箱": "Temporal Solver Bins",
    "光谱求解分箱": "Spectral Solver Bins",
    "Seed 光轴倾斜 x (deg)": "Seed Optical-Axis Tilt x (deg)",
    "Seed 光轴倾斜 y (deg)": "Seed Optical-Axis Tilt y (deg)",
    "扫描起点 (m)": "Scan Start (m)",
    "扫描终点 (m)": "Scan End (m)",
    "扫描平面数": "Number of Scan Planes",
    "焦散传播介质折射率": "Caustic-Propagation Medium Refractive Index",
    "每程之间总透过率": "Total Inter-Pass Transmission",
    "晶体长度 (m)": "Crystal Length (m)",
    "晶体宽度 x (m)": "Crystal Width x (m)",
    "晶体高度 y (m)": "Crystal Height y (m)",
    "掺杂离子密度 (m⁻³)": "Dopant Ion Density (m⁻³)",
    "泵浦被动吸收 (m⁻¹)": "Passive Pump Absorption (m⁻¹)",
    "信号被动损耗 (m⁻¹)": "Passive Signal Loss (m⁻¹)",
    "泵浦吸收截面 σpa (m²)": "Pump Absorption Cross Section σpa (m²)",
    "泵浦发射截面 σpe (m²)": "Pump Emission Cross Section σpe (m²)",
    "泵浦激发量子效率": "Pump Excitation Quantum Efficiency",
    "信号发射截面 σe (m²)": "Signal Emission Cross Section σe (m²)",
    "信号吸收截面 σa (m²)": "Signal Absorption Cross Section σa (m²)",
    "上能级寿命 (s)": "Upper-Level Lifetime (s)",
    "泵浦能级寿命 (s)": "Pump-Level Lifetime (s)",
    "下能级寿命 (s)": "Lower-Level Lifetime (s)",
    "额外非辐射产热比例（百分点）": "Additional Nonradiative Heat Fraction (Percentage Points)",
    "沿 z 检查的拟合截面数": "Number of Fitted Cross-Sections Along z",
    "Zernike 最大径向阶数": "Maximum Zernike Radial Order",
    "OPD 径向平均分箱数": "OPD Radial-Averaging Bins",
    "导热率 kx (W/m/K)": "Thermal Conductivity kx (W/m/K)",
    "导热率 ky (W/m/K)": "Thermal Conductivity ky (W/m/K)",
    "导热率 kz (W/m/K)": "Thermal Conductivity kz (W/m/K)",
    "材料密度 (kg/m³)": "Material Density (kg/m³)",
    "比热 (J/kg/K)": "Specific Heat (J/kg/K)",
    "冷却水温度 (K)": "Coolant Temperature (K)",
    "水冷换热系数 h (W/m²/K)": "Water-Cooling Heat-Transfer Coefficient h (W/m²/K)",
    "材料数据参考温度 (K)": "Material-Data Reference Temperature (K)",
    "材料数据参考离子密度 (m⁻³)": "Material-Data Reference Ion Density (m⁻³)",
    "非线性折射率 n₂ (m²/W)": "Nonlinear Refractive Index n₂ (m²/W)",
    "x 窗口 (m)": "x Window (m)",
    "y 窗口 (m)": "y Window (m)",
    "起点 (W)": "Start (W)",
    "终点 (W)": "End (W)",
    "点数": "Number of Points",
    "精度预设": "Precision Preset",
    "种子求解模型": "Seed Solver Model",
    "热光耦合": "Thermo-Optic Coupling",
    "Seed 不可逆损耗产热耦合": "Seed Irreversible-Loss Heat Coupling",
    "泵浦耗尽": "Pump Depletion",
    "增益饱和": "Gain Saturation",
    "界面模型": "Interface Model",
    "布居动力学": "Population Dynamics",
    "种子空间传播": "Seed Spatial Propagation",
    "泵浦空间传播": "Pump Spatial Propagation",
    "晶体内 Seed 体数据": "In-Crystal Seed Volume Data",
    "FFT 计算后端": "FFT Compute Backend",
    "Pump 光源定义": "Pump Source Definition",
    "空间光斑": "Spatial Beam Profile",
    "泵浦方向": "Pump Direction",
    "泵浦波形": "Pump Waveform",
    "功率定义": "Power Definition",
    "泵浦光谱": "Pump Spectrum",
    "泵浦光谱密度自变量": "Pump Spectral-Density Coordinate",
    "宽带 Pump 空间输运": "Broadband Pump Spatial Transport",
    "插入位置": "Insertion Position",
    "插入元件": "Element to Insert",
    "选中并编辑元件": "Select Element to Edit",
    "通过方向": "Pass Direction",
    "时间强度来源": "Temporal Intensity Source",
    "Seed 空间增益闭合": "Seed Spatial Gain Closure",
    "时间–频率联合能量": "Joint Time–Frequency Energy",
    "Seed 偏振传播": "Seed Polarization Propagation",
    "横向模态模型": "Transverse Mode Model",
    "不同波长的横向源模型": "Wavelength-Dependent Transverse Source Model",
    "晶体形状": "Crystal Shape",
    "标定方式": "Calibration Method",
    "分布形式": "Profile Type",
    "作用方向": "Direction",
    "变化 (%)": "Relative Change (%)",
    "两侧/外缘 ↔ 中心（正值中心高）": (
        "Edges ↔ Center (Positive Means Higher at Center)"
    ),
    "单向：负坐标端 → 正坐标端": (
        "Monotonic: Negative-Coordinate Side → Positive-Coordinate Side"
    ),
    "x 方向": "x Direction",
    "y 方向": "y Direction",
    "z 方向": "z Direction",
    "径向（圆柱外圆周 ↔ 中心轴）": (
        "Radial (Cylinder Surface ↔ Central Axis)"
    ),
    "能级模型": "Energy-Level Model",
    "热场模型": "Thermal-Field Model",
    "热源账本": "Heat-Source Ledger",
    "主动 Pump 吸收产热模型": "Active Pump-Absorption Heat Model",
    "dn/dT 数据定义": "dn/dT Data Definition",
    "折射率模型": "Refractive-Index Model",
    "拟合半径来源": "Fit-Radius Source",
    "冷却边界": "Cooling Boundary",
    "应力模型": "Stress Model",
    "光谱参数反馈模式": "Spectroscopic-Parameter Feedback Mode",
    "材料表越界策略": "Material-Table Out-of-Range Policy",
    "上能级寿命随温度": "Upper-Level Lifetime vs. Temperature",
    "上能级寿命随掺杂浓度": "Upper-Level Lifetime vs. Dopant Concentration",
    "ASE 模型": "ASE Model",
    "非线性相位模型": "Nonlinear Phase Model",
    "不可逆强场损耗模型": "Irreversible Strong-Field Loss Model",
    "Pump 横向计算网格": "Pump Transverse Computational Grid",
    "x/y 网格分布": "x/y Grid Distribution",
    "结果有效性模式": "Result-Validity Mode",
    "遍历参数": "Sweep Parameter",
    "间隔": "Spacing",
    "配置入口": "Configuration Entry",
    "侧栏显示": "Sidebar View",
    "Commercial App 快捷方案": "Commercial App Shortcut",
    "频谱 FWHM 分箱": "Spectral FWHM Bins",
    "TL 脉冲 FWHM 分箱": "Transform-Limited Pulse FWHM Bins",
    "时间窗": "Time Window",
    "建议最少 FFT 点": "Suggested Minimum FFT Samples",
    "当前是全场 Kerr 中间档：x-y-t 复场、衍射、干涉、色散和 Kerr 仍被求解；省略的是脉内前沿→尾沿的逐软窗增益耗尽，周期布居若选择 cycle-average 也属于平均闭合。": (
        "This is the full-field Kerr intermediate mode: the x-y-t complex "
        "field, diffraction, interference, dispersion, and Kerr response are "
        "still solved. What is omitted is front-to-tail intrapulse gain "
        "depletion through sequential soft windows; cycle-average population, "
        "when selected, is also an averaged closure."
    ),
    "输入/输出归一化波长光谱密度（论文对比视图）": (
        "Normalized Input/Output Wavelength Spectral Density (Paper Comparison)"
    ),
    "输入/输出归一化波长光谱密度（对数尺度）": (
        "Normalized Input/Output Wavelength Spectral Density (Log Scale)"
    ),
    "线性尺度": "Linear Scale",
    "对数尺度": "Log Scale",
    "对数谱默认显示到相对峰值 10⁻⁶；更低的点只在图中隐藏，不会修改计算结果或归档数据。": (
        "The logarithmic spectrum is displayed down to 10⁻⁶ of the relative "
        "peak by default. Lower samples are hidden only in the plot; the "
        "computed and archived data remain unchanged."
    ),
    "归一化 dE/dλ": "Normalized dE/dλ",
    "这里画的是波长密度 dE/dλ，并分别按峰值归一化；不是每个数值分箱的能量。J/bin 在非均匀波长坐标上会产生纯绘图 Jacobian 偏差。": (
        "This plot shows the wavelength density dE/dλ, normalized to each "
        "curve's peak. It is not energy per numerical bin; plotting J/bin against a "
        "nonuniform wavelength coordinate introduces a purely graphical "
        "Jacobian distortion."
    ),
    "能量质量的原生分箱坐标：频率。": (
        "The native bin coordinate for the energy masses is frequency."
    ),
    "能量质量的原生分箱坐标：波长。": (
        "The native bin coordinate for the energy masses is wavelength."
    ),
    "绝对光谱密度与原生数值分箱": (
        "Absolute Spectral Density and Native Numerical Bins"
    ),
    "输入 dE/dλ": "Input dE/dλ",
    "输出 dE/dλ": "Output dE/dλ",
    "绝对波长光谱密度": "Absolute Wavelength Spectral Density",
    "原生离散光谱能量（数值诊断）": (
        "Native Discrete Spectral Energy (Numerical Diagnostic)"
    ),
    "每个数值分箱能量 (J)": "Energy per Numerical Bin (J)",
    "密度积分闭合误差：输入 λ域 ": (
        "Density-Integration Closure Error: Input λ-Domain "
    ),
    "输入 f域 ": "Input f-Domain ",
    "输出 λ域 ": "Output λ-Domain ",
    "输出 f域 ": "Output f-Domain ",
    "无法构造坐标守恒的光谱密度：": (
        "Could Not Construct a Coordinate-Conserving Spectral Density: "
    ),
    "输入 J/bin": "Input J/bin",
    "输出 J/bin": "Output J/bin",
    "，输入 f域 ": ", Input f-Domain ",
    "；输出 λ域 ": "; Output λ-Domain ",
    "，输出 f域 ": ", Output f-Domain ",
    "相干输入采样合同未通过：": "Coherent Input Sampling Contract Failed: ",
    "。细峰、旁瓣和 Kerr 展宽暂不应作定量结论。": (
        ". Fine peaks, side lobes, and Kerr broadening must not yet be "
        "interpreted quantitatively."
    ),
    "增益来源": "Gain Source",
    "光路 / 多程组织方式": "Optical Path / Multipass Layout",
    "输入坐标基准": "Input Coordinate Reference",
    "强度输入方式": "Intensity Input Mode",
    "保存各光学面": "Save Optical Planes",
    "使用独立光谱相位文件": "Use a Separate Spectral-Phase File",
    "x / y 使用不同腰斑位置（像散光束）": "Use Different x/y Waist Positions (Astigmatic Beam)",
    "扫描系统输出后的焦散并拟合 D4σ / M²": "Scan the Output Caustic and Fit D4σ / M²",
    "保存每个焦散平面的二维光斑": "Save the 2D Beam Profile at Each Caustic Plane",
    "相邻程自动反向": "Automatically Reverse Adjacent Passes",
    "晶体外区域作为硬孔径": "Use the Crystal Exterior as a Hard Aperture",
    "启用非均匀掺杂": "Enable Nonuniform Doping",
    "把 Seed 被动吸收作为热源": "Include Passive Seed Absorption as a Heat Source",
    "诊断热折射率与增益的抛物近似质量": "Diagnose Parabolic-Fit Quality for Thermal Index and Gain",
    "严格模式要求所有外部 Seed 损耗具名闭合": "Require Named Closure for All External Seed Losses in Strict Mode",
    "正式结果要求样品级材料实验标定证据": "Require Sample-Level Experimental Material Calibration for Formal Results",
    "二维参数网格（两个输入同时遍历）": "2D Parameter Grid (Sweep Two Inputs Together)",
    "插入增益晶体": "Insert Gain Crystal",
    "晶体聚焦视图": "Crystal-Focused View",
    "求 Pump + Seed 周期稳态": "Solve Pump + Seed Periodic Steady State",
    "独立均匀光学网格（双网格耦合）": "Independent Uniform Optical Grid (Dual-Grid Coupling)",
    "生成 / 刷新当前模型完整说明": "Generate / Refresh Complete Model Description",
    "生成完整物理说明": "Generate Complete Physics Description",
    "应用所选 Commercial App 方案": "Apply Selected Commercial App Profile",
    "清除逐项覆盖，恢复由精度预设控制": "Clear Per-Feature Overrides and Restore Preset Control",
    "插入薄透镜": "Insert Thin Lens",
    "应用下方所选参数": "Apply Selected Parameters Below",
    "删除所选": "Delete Selected",
    "右侧详细显示 PUMP": "Show PUMP Details on the Right",
    "右侧详细显示 SEED": "Show SEED Details on the Right",
    "运行参数遍历": "Run Parameter Sweep",
    "下载当前光谱相位 CSV": "Download Current Spectral Phase CSV",
    "二维 D4σ 包络": "2D D4σ Envelope",
    "三维光路": "3D Optical Path",
    "输入场附加相位": "Additional Input-Field Phase",
    "ZnSe / 展宽器相位": "ZnSe / Stretcher Phase",
    "D4σ x 半径": "D4σ x Radius",
    "D4σ y 半径": "D4σ y Radius",
    "x 束心 offset": "x Beam-Centroid Offset",
    "y 束心 offset": "y Beam-Centroid Offset",
    "Seed 单脉冲能量": "Seed Pulse Energy",
    "附加色散前 时间 FWHM": "Temporal FWHM Before Additional Dispersion",
    "GDD/TOD/FOD 后 时间 FWHM": "Temporal FWHM After GDD/TOD/FOD",
    "x 发散半角": "x Divergence Half-Angle",
    "y 发散半角": "y Divergence Half-Angle",
    "上传配置 JSON（上传后自动回填表单）": "Upload Configuration JSON (Automatically Populates the Form)",
    "外部材料列表": "External Material List",
    "泵浦截面响应 CSV：λ(nm), σabs, σem": "Pump Cross-Section Response CSV: λ(nm), σabs, σem",
    "信号截面响应 CSV：λ(nm), σabs, σem": "Signal Cross-Section Response CSV: λ(nm), σabs, σem",
    "commercial_app_fast — Commercial App 式热/Kerr 快速": "commercial_app_fast — Fast Commercial App-Style Thermal/Kerr Mode",
    "balanced — 平衡": "balanced — Balanced",
    "fast — 快速": "fast — Fast",
    "detailed — 详细": "detailed — Detailed",
    "auto — 自动": "auto — Automatic",
    "fluence — 快速单色通量（可选 FN）": "fluence — Fast Monochromatic Fluence (Optional FN)",
    "commercial_app_fast — Gaussian duct / 等效热-Kerr透镜": "commercial_app_fast — Gaussian Duct / Equivalent Thermal-Kerr Lens",
    "spectral_time — 多色×时间/FN": "spectral_time — Multicolor × Time / FN",
    "josa_transport — JOSA 时空输运": "josa_transport — JOSA Spatiotemporal Transport",
    "coherent_cpa — 相干 CPA": "coherent_cpa — Coherent CPA",
    "thermal_optical_fixed_point — Pump↔热场↔布居（自洽；Seed 读取收敛场）": "thermal_optical_fixed_point — Pump ↔ Thermal Field ↔ Population (Self-Consistent; Seed Reads the Converged Field)",
    "off — 不把 Seed 强场损耗送入热场": "off — Do Not Feed Seed Strong-Field Losses into the Thermal Field",
    "one_way_postprocess — Seed 损耗→热场→再传播一次 Seed": "one_way_postprocess — Seed Loss → Thermal Field → One Additional Seed Propagation",
    "auto — 由精度预设决定": "auto — Determined by Precision Preset",
    "auto — 按精度预设": "auto — Follow Precision Preset",
    "on — 开": "on — Enabled",
    "off — 关": "off — Disabled",
    "transient — 时序/周期布居": "transient — Time-Dependent / Periodic Population",
    "steady_state — 仅 Pump 代数稳态（不含 Seed）": "steady_state — Pump-Only Algebraic Steady State (No Seed)",
    "frozen — 固定布居": "frozen — Fixed Population",
    "hybrid_abcd_fft — 晶体外逐波长 ABCD，晶体内 FFT": "hybrid_abcd_fft — Per-Wavelength ABCD Outside the Crystal, FFT Inside",
    "fft — 从源面开始全程 FFT": "fft — Full FFT Propagation from the Source Plane",
    "abcd — 快速 Gaussian ABCD（晶体内采样传播）": "abcd — Fast Gaussian ABCD (Sampled In-Crystal Propagation)",
    "off — 规定光斑，不传播": "off — Prescribed Beam Profile, No Propagation",
    "prescribed_caustic — 直接规定晶体内焦散": "prescribed_caustic — Directly Prescribe the In-Crystal Caustic",
    "super_gaussian_abcd — 自相似 SG 强度包络 + ABCD": "super_gaussian_abcd — Self-Similar SG Intensity Envelope + ABCD",
    "stochastic_raytrace — 随机光线 + 三维路径沉积": "stochastic_raytrace — Stochastic Rays + 3D Path Deposition",
    "hybrid_abcd_fft — 晶体外 ABCD，晶体内 FFT": "hybrid_abcd_fft — ABCD Outside the Crystal, FFT Inside",
    "fft — 全程采样波动光学": "fft — Sampled Wave-Optics Propagation Throughout",
    "abcd — 快速二阶矩 Gaussian 传播": "abcd — Fast Second-Moment Gaussian Propagation",
    "last_and_cumulative — 末程 + 累计（推荐）": "last_and_cumulative — Last Pass + Cumulative (Recommended)",
    "last_only — 仅末程（最省内存）": "last_only — Last Pass Only (Lowest Memory)",
    "per_pass — 每一程 + 累计（内存较大）": "per_pass — Every Pass + Cumulative (Higher Memory)",
    "直接束腰 + M²（常用 Gaussian 输入）": "Direct Waist + M² (Common Gaussian Input)",
    "径向超高斯：束腰半径 + 发散角": "Radial Super-Gaussian: Waist Radius + Divergence Angle",
    "shared — 共享空间场（快速降阶）": "shared — Shared Spatial Field (Fast Reduced Order)",
    "per_wavelength — 各波长独立空间输运": "per_wavelength — Independent Spatial Transport per Wavelength",
    "最前面": "At the Beginning",
    "在 ": "After ",
    " 后": "",
    "薄透镜": "Thin Lens",
    "厚透镜": "Thick Lens",
    "自由空间": "Free Space",
    "光阑": "Aperture",
    "观察面": "Observation Plane",
    "双程双透镜回程成像": "Double-Pass, Dual-Lens Return Imaging",
    "local_field — 局域复场增益（默认，可产生增益导引）": "local_field — Local Complex-Field Gain (Default; Can Produce Gain Guiding)",
    "gaussian_mode_overlap — 高斯模态与 3D 反转重叠（无增益导引）": "gaussian_mode_overlap — Gaussian-Mode / 3D-Inversion Overlap (No Gain Guiding)",
    "M² 有效高斯（快速）": "Effective M² Gaussian (Fast)",
    "HG 强度集合（快速降阶）": "HG Intensity Ensemble (Fast Reduced Order)",
    "HG 模态逐个传播（共享布居）": "Propagate HG Modes Individually (Shared Population)",
    "共享同一个源面参数（兼容/默认）": "Share One Set of Source-Plane Parameters (Compatible / Default)",
    "参数化色差与空间啁啾": "Parameterized Chromatic Aberration and Spatial Chirp",
    "逐波长参数表（缺项自动沿用参数化值）": "Per-Wavelength Parameter Table (Missing Values Use the Parameterized Baseline)",
    "cuboid — 六面体/长方体": "cuboid — Rectangular Cuboid",
    "cylinder — 圆柱体": "cylinder — Cylinder",
    "体积平均值 — 上面的离子密度保持为平均浓度（推荐）": "Volume Average — Keep the Ion Density Above as the Mean Concentration (Recommended)",
    "峰值 — 上面的离子密度作为最高浓度": "Peak — Treat the Ion Density Above as the Maximum Concentration",
    "不归一化 — 上面的离子密度乘原始相对分布": "No Normalization — Multiply the Ion Density Above by the Raw Relative Profile",
    "four_level — 内置四能级级联（4 个布居）": "four_level — Built-In Four-Level Cascade (4 Populations)",
    "three_level — 内置三能级级联（3 个布居）": "three_level — Built-In Three-Level Cascade (3 Populations)",
    "two_level — 内置二能级（2 个布居）": "two_level — Built-In Two-Level System (2 Populations)",
    "custom — 自定义 N 能级/跃迁": "custom — Custom N-Level / Transition Network",
    "auto — 跟随精度预设/材料开关": "auto — Follow Precision Preset / Material Switches",
    "off — 关闭": "off — Disabled",
    "steady_3d — 三维稳态": "steady_3d — 3D Steady State",
    "transient_3d — 三维瞬态": "transient_3d — 3D Transient",
    "pump_component_ledger — 量子亏损快速账本": "pump_component_ledger — Fast Quantum-Defect Ledger",
    "event_energy_ledger — 逐跃迁/衰变/非线性事件账本": "event_energy_ledger — Per-Transition / Decay / Nonlinear Event Ledger",
    "lifetime_quantum_defect — 寿命自动分解：量子亏损 + 非辐射衰变（推荐）": "lifetime_quantum_defect — Lifetime Decomposition: Quantum Defect + Nonradiative Decay (Recommended)",
    "quantum_defect — 吸收能量守恒：1−ηq·λp/λref": "quantum_defect — Absorbed-Energy Conservation: 1−ηq·λp/λref",
    "scaled_quantum_defect — 可调倍率×(1−λp/λref)": "scaled_quantum_defect — Adjustable Scale × (1−λp/λref)",
    "spectral_quantum_defect — 按 Pump 光谱与局部截面计算量子亏损": "spectral_quantum_defect — Compute Quantum Defect from Pump Spectrum and Local Cross Sections",
    "spectral_scaled_quantum_defect — 按 Pump 光谱与局部截面计算可调量子亏损": "spectral_scaled_quantum_defect — Adjustable Quantum Defect from Pump Spectrum and Local Cross Sections",
    "explicit_fraction — 直接输入实测主动吸收产热比例": "explicit_fraction — Enter a Measured Active-Absorption Heat Fraction",
    "measured_free_expansion — 自由膨胀实测热光系数": "measured_free_expansion — Measured Free-Expansion Thermo-Optic Coefficient",
    "intrinsic_fixed_strain — 固定应变本征热光系数": "intrinsic_fixed_strain — Intrinsic Fixed-Strain Thermo-Optic Coefficient",
    "automatic — 使用有效晶体横截面": "automatic — Use the Effective Crystal Cross-Section",
    "manual — 手动指定椭圆拟合半径": "manual — Specify Elliptical Fit Radii Manually",
    "seed_entrance — 使用 Seed 晶体入口 D4σ 半径": "seed_entrance — Use the Seed Crystal-Entrance D4σ Radius",
    "water — 水冷/对流": "water — Water Cooling / Convection",
    "fixed_temperature — 边界固定温度": "fixed_temperature — Fixed Boundary Temperature",
    "adiabatic — 绝热": "adiabatic — Adiabatic",
    "none — 关闭": "none — Disabled",
    "isotropic_proxy — 快速局部风险代理": "isotropic_proxy — Fast Local Risk Proxy",
    "thermoelastic_fem — 三维热弹性力平衡": "thermoelastic_fem — 3D Thermoelastic Force Balance",
    "anisotropic_thermoelastic_fem — 各向异性张量 FEM": "anisotropic_thermoelastic_fem — Anisotropic Tensor FEM",
    "off — 使用材料导入/表单给定的固定参数": "off — Use Fixed Parameters from Material Import / Form",
    "operating_point — 在统一工作温度和参考掺杂下修正": "operating_point — Correct at a Uniform Operating Temperature and Reference Doping",
    "local_3d — 热光固定点内逐体素重新计算": "local_3d — Recompute Each Voxel Inside the Thermo-Optic Fixed Point",
    "record — 记录端点夹持并交给结果有效性门判断": "record — Record Endpoint Clamping and Defer to the Result-Validity Gate",
    "error — 一旦超出材料标定范围立即报错": "error — Fail Immediately Outside the Material Calibration Range",
    "six_ray — 六方向快速筛选": "six_ray — Fast Six-Direction Screening",
    "angular_discrete_ordinates — 任意角离散方向 + 寄生本征值": "angular_discrete_ordinates — Arbitrary-Angle Discrete Ordinates + Parasitic Eigenvalue",
    "与 Seed/BPM 共用": "Shared with Seed/BPM",
    "使用物理网格": "Use the Physical Grid",
    "独立均匀网格": "Independent Uniform Grid",
    "均匀网格": "Uniform Grid",
    "exploratory — 保留结果并醒目标出无效点与诊断警告": "exploratory — Keep Results and Clearly Mark Invalid Points and Diagnostic Warnings",
    "strict — 无效结果禁止正式导出，遍历仍保留诊断记录": "strict — Block Formal Export of Invalid Results; Keep Sweep Diagnostics",
    "表单": "Form",
    "完整 JSON": "Full JSON",
    "只替换 Seed：Effective-area SPM + M²（推荐公平对比）": "Seed Only: Effective-Area SPM + M² (Recommended for Fair Comparisons)",
    "全链路极速：同时简化周期与热光耦合（会改变工作点）": "Fastest Full Chain: Also Simplify Periodic and Thermo-Optic Coupling (Changes the Operating Point)",
    "L3.5 趋势扫描：Effective-area 7PA + 载流子/FCA": "L3.5 Trend Scan: Effective-Area 7PA + Carriers/FCA",
    "L3.5 强场趋势档：Gaussian-q + 一维相干包络，按 Gaussian 有效面积矩计算多光子损耗、载流子生成、FCA 与载流子折射；局部跨脉冲载流子定态直接求解，不进入慢速全局 pulse-map。它适合找拐点和筛选参数，最终数值仍应由 L4/L4.5 复核。": (
        "L3.5 strong-field trend profile: Gaussian-q plus a one-dimensional "
        "coherent envelope evaluates multiphoton loss, carrier generation, "
        "FCA, and carrier refraction with a Gaussian effective-area moment. "
        "The local inter-pulse carrier steady state is solved directly instead "
        "of entering the slow global pulse map. Use it to locate rollovers and "
        "screen parameters, then verify final values with L4/L4.5."
    ),
    "建议工作流：Commercial App Fast 用于快速扫描和对齐 Commercial App 风格结果；L3.5 用于 7PA/载流子/FCA 趋势筛选；强脉内人口竞争用 JOSA 复核；需要频率间相干相位、色散、Kerr/SPM 或压缩质量时，再用相干 L / coherent_cpa 复核。不存在对所有问题都最高精度的单一后端。": (
        "Suggested workflow: use Commercial App Fast for rapid scans and Commercial App-style "
        "alignment, L3.5 to screen 7PA/carrier/FCA trends, and JOSA to verify "
        "strong intrapulse population competition. Use coherent L / "
        "coherent_cpa when inter-frequency phase, dispersion, Kerr/SPM, or "
        "compression quality matters. No single backend has the highest "
        "fidelity for every question."
    ),
    "coherent_cpa 使用完整采样场；Commercial App 使用 L3.5 Gaussian effective-area 矩闭合。两者都与纯相位 Kerr 分账。": (
        "coherent_cpa uses the fully sampled field; Commercial App uses the L3.5 "
        "Gaussian effective-area moment closure. Both are accounted "
        "separately from phase-only Kerr."
    ),
    "由泵浦和速率方程求解": "Solve from Pump and Rate Equations",
    "直接指定小信号增益": "Specify Small-Signal Gain Directly",
    "一键设置程数": "Set the Pass Count",
    "自己搭建光学平台": "Build a Custom Optical Bench",
    "time — 只给时间强度，FFT生成光谱": "time — Temporal Intensity Only; Generate Spectrum with FFT",
    "spectrum — 只给光谱强度，IFFT生成时间": "spectrum — Spectral Intensity Only; Generate Time Waveform with IFFT",
    "dual — 同时给时间与光谱强度，自动恢复相位": "dual — Temporal + Spectral Intensity; Retrieve Phase Automatically",
    "Pump LASER→首元件距离": "Pump LASER → First Element Distance",
    "Seed LASER→首元件距离": "Seed LASER → First Element Distance",
    "Pump 前一元件→首个 GAIN 距离": "Pump Previous Element → First GAIN Distance",
    "Seed 前一元件→首个 GAIN 距离": "Seed Previous Element → First GAIN Distance",
    "Seed 损耗→热场→再传播一次 Seed": "Seed Loss → Thermal Field → One Additional Seed Propagation",
    "各波长独立空间输运": "Independent Spatial Transport per Wavelength",
    "自由膨胀实测热光系数": "Measured Free-Expansion Thermo-Optic Coefficient",
    "固定应变本征热光系数": "Intrinsic Fixed-Strain Thermo-Optic Coefficient",
    "水冷/对流": "Water Cooling / Convection",
    "边界固定温度": "Fixed Boundary Temperature",
    "绝热": "Adiabatic",
    "快速局部风险代理": "Fast Local Risk Proxy",
    "三维热弹性力平衡": "3D Thermoelastic Force Balance",
    "各向异性张量 FEM": "Anisotropic Tensor FEM",
    "记录端点夹持并交给结果有效性门判断": "Record Endpoint Clamping and Defer to the Result-Validity Gate",
    "只给时间强度，FFT生成光谱": "Temporal Intensity Only; Generate Spectrum with FFT",
    "只给光谱强度，IFFT生成时间": "Spectral Intensity Only; Generate Time Waveform with IFFT",
    "同时给时间与光谱强度，自动恢复相位": "Temporal + Spectral Intensity; Retrieve Phase Automatically",
    "切换整个界面的表单、说明、图表注释和结果显示；不会改变配置值或重新拼接旧结果。": "Switch forms, descriptions, chart annotations, and result displays across the entire interface; configuration values and saved results remain unchanged.",
    "横截面 (x-y)": "Cross-Section (x-y)",
    "纵截面 (x-z)": "Longitudinal Section (x-z)",
    "纵截面 (y-z)": "Longitudinal Section (y-z)",
    " D4σ 包络": " D4σ Envelope",
    " 运行后 D4σ 采样": " Post-Run D4σ Sample",
    "计算前冷态 q 包络 + 运行后实际 D4σ 采样面": "Pre-Run Cold-State q Envelope + Actual Post-Run D4σ Samples",
    "Pump / Seed 计算前冷态 q 包络": "Pump / Seed Pre-Run Cold-State q Envelope",
    "保守 D4σ 横向包络 (mm)": "Conservative Transverse D4σ Envelope (mm)",
    "二维冷态/运行后 D4σ": "2D Cold-State / Post-Run D4σ",
    "最终脉冲能量": "Final Pulse Energy",
    "瞬态结果": "Transient Result",
    "材料域最小值": "Material-Domain Minimum",
    "材料域平均值": "Material-Domain Mean",
    "材料域最大值": "Material-Domain Maximum",
    "四个工作区共享同一份完整配置；切换页面不会丢失未显示参数。": (
        "All four workspaces share one complete configuration; switching "
        "views preserves every setting."
    ),
    "当前生效模型": "Active Model",
    "**当前生效模型**": "**Active Model**",
    "布居": "Population",
    "热": "Thermal",
    "` · 布居 `": "` · Population `",
    "` · 热 `": "` · Thermal `",
    "选择求解模型、耦合方式和材料能级；高级覆盖项默认收起。": (
        "Select the solver, coupling strategy, and material energy-level "
        "model. Advanced overrides are collapsed by default."
    ),
    "算例名称": "Case Name",
    "信号模型": "Signal Model",
    "单色通量": "Monochromatic Fluence",
    "Commercial App 式 Gaussian 快速": "Fast Commercial App-Style Gaussian",
    "多色 × 时间": "Multiwavelength × Time",
    "JOSA 输运": "JOSA Transport",
    "相干 CPA": "Coherent CPA",
    "空间传播": "Spatial Propagation",
    "平台与网格": "Platform and Grid",
    "自建 · ": "Custom · ",
    "快速 · ": "Fast · ",
    " 程": " Pass",
    "精度预设：": "Precision Preset: ",
    "　·　布居：": " · Population: ",
    "　·　热场：": " · Thermal Field: ",
    "### 配置体检与模块联动": "### Configuration Health and Module Coordination",
    "配置体检与模块联动": "Configuration Health and Module Coordination",
    "模块链": "Module Chain",
    "用户选择": "Requested",
    "实际生效": "Effective",
    "状态": "Status",
    "编辑位置": "Edit Location",
    "自动托管": "Auto-Managed",
    "已就绪": "Ready",
    "关闭": "Off",
    "Seed 场与传播": "Seed Field and Propagation",
    "Pump 传播与主动吸收": "Pump Propagation and Active Absorption",
    "Seed 空间增益与横模": "Seed Spatial Gain and Transverse Modes",
    "Seed 偏振与双折射": "Seed Polarization and Birefringence",
    "多程、方向与程间光路": "Multipass Direction and Inter-Pass Path",
    "周期稳态与布居": "Periodic Steady State and Population",
    "热光闭环": "Thermo-Optic Loop",
    "应力、光弹与端面": "Stress, Photoelasticity, and End Faces",
    "物理网格与光学网格": "Physical and Optical Grids",
    "ASE 与强场": "ASE and Strong-Field Effects",
    "运行 / Pump / Seed": "Run / Pump / Seed",
    "运行 → Pump + Seed 周期稳态": "Run → Pump + Seed Periodic Steady State",
    "运行 → 多物理耦合 / 材料 / 热": "Run → Multiphysics Coupling / Materials / Thermal",
    "材料 / 热 → 应力与光弹": "Materials / Thermal → Stress and Photoelasticity",
    "材料 / 热 → ASE 与强场": "Materials / Thermal → ASE and Strong-Field Effects",
    "查看六条核心模块合同": "View Six Core Module Contracts",
    "查看 10 条核心模块合同": "View 10 Core Module Contracts",
    "① 模型 / ② 光束 → Pump": "① Model / ② Beams → Pump",
    "② 光束 → Seed 增益输运 / 空间光束": "② Beams → Seed Gain Transport / Spatial Beam",
    "② 光束 → Seed → Jones 偏振": "② Beams → Seed → Jones Polarization",
    "① 模型 → 多程方式 / ② 光束 → 多程与光学平台": "① Model → Multipass Method / ② Beams → Multipass and Optical Bench",
    "Commercial App 中每个 Gaussian-q/HG 通道按三维反转重叠取得标量增益，并可通过共享局部布居提交受激耗尽；增益不会把单个通道重塑成任意横向场。": "In Commercial App, each Gaussian-q/HG channel obtains scalar gain from its overlap with the 3D inversion and can submit stimulated depletion through the shared local population; gain does not reshape one channel into an arbitrary transverse field.",
    "该模式仍传播衍射、走离、孔径和完整三维热相位，但增益只改变总功率，不改变横向模态。fluence 路径仅支持单色且必须关闭空间 FN 饱和。": "This mode still propagates diffraction, walk-off, apertures, and the full 3D thermal phase, but gain changes only total power rather than the transverse mode. The fluence path is monochromatic and requires spatial FN saturation to be off.",
    "完整遍历结果与后台作业的关联已丢失；已禁止生成只有摘要的伪完整结果包。": "The association between the complete sweep results and the background job has been lost; generation of a summary-only package disguised as a complete package is blocked.",
    "。请保留当前页面和后台作业现场；恢复完整结果关联后才会重新提供下载。": ". Keep the current page and background-job state; downloading will be restored only after the complete-result association is recovered.",
    "下载摘要遍历结果包（不含完整场）": "Download Summary Sweep Package (No Full Fields)",
    "明确的摘要包：保留每个遍历点的页面指标、有效性状态、warning/validation 诊断和扫描配置，但不包含完整三维场、光学面或可重新绘制的逐点 SimulationResult。": "Explicit summary package: retains page metrics, validity state, warning/validation diagnostics, and sweep configuration for every point, but excludes full 3D fields, optical planes, and per-point SimulationResult data needed for replotting.",
    " 点摘要包，不是完整结果包；文件名带 summary_only，载入时不会被误认为包含完整场。": "-point summary package, not a complete result package; the filename contains summary_only so loading cannot mistake it for a package with full fields.",
    "这是 ": "This is a ",
    "② 光束 → Seed → Jones 偏振与双折射": "② Beams → Seed → Jones Polarization and Birefringence",
    "② 光束 → Seed 增益输运": "② Beams → Seed Gain Transport",
    "查看 ": "View ",
    " 条核心模块合同": " Core Module Contracts",
    "主动吸收输运": "Active Absorption Transport",
    "用户/导入配置": "User / Imported Configuration",
    "用配置的固定主动吸收系数推进 Pump 衰减与吸收账本；能级激发仍由 Pump 光子流、跃迁截面和量子效率求解。": "Advance Pump attenuation and the absorption ledger with the configured fixed active-absorption coefficient; level excitation is still solved from Pump photon flux, transition cross sections, and quantum efficiency.",
    "由局部能级布居和 Pump 吸收/发射截面计算净主动吸收，因此 Pump 耗尽可随饱和状态变化。": "Compute net active absorption from local level populations and Pump absorption/emission cross sections, so Pump depletion can vary with saturation state.",
    "相干局域算符的最多子步不能小于最少子步；已同步扩大上限。": "The maximum coherent local-operator substep count cannot be below the minimum; the upper limit was increased accordingly.",
    "瞬态增益的最大软窗数不能小于初始软窗数；已同步扩大自适应上限。": "The maximum transient-gain soft-window count cannot be below the initial count; the adaptive upper limit was increased accordingly.",
    "因果多程求解器传播有限 Seed 脉冲，不能把 CW 当作一个无限长脉冲；已退回兼容的顺序多程。": "The causal multipass solver propagates a finite Seed pulse and cannot treat CW as an infinitely long pulse; it was changed to compatible sequential multipass propagation.",
    "关闭不适用于 CW 的因果模板后，保留其晶体命中次数作为顺序多程次数。": "After disabling the causal template that is inapplicable to CW, its crystal-hit count was retained as the sequential pass count.",
    "物理多程模板从晶体正向入口注入，后续方向由模板/返回路径产生。": "The physical multipass template injects at the forward crystal entrance; subsequent directions are generated by the template or return path.",
    "物理多程必须解析 Seed 脉冲包络及程间飞行时间。": "Physical multipass must resolve the Seed pulse envelope and inter-pass flight time.",
    "物理多程的时间插值至少需要三个输入时间样本。": "Physical multipass time interpolation requires at least three input time samples.",
    "物理多程目前只接受可分离时间–频率输入；非可分离联合矩阵没有对应的绝对时间场。": "Physical multipass currently accepts only separable time-frequency input; a nonseparable joint matrix has no corresponding absolute-time field.",
    "物理多程当前不传播互不相干的 HG 模态组；保留一个等效 M² 场。": "Physical multipass does not currently propagate an incoherent HG mode ensemble; one effective-M2 field is retained.",
    "物理多程必须在每个命中位置提交局域增益与共享布居竞争。": "Physical multipass must submit local gain and shared-population competition at every hit position.",
    "当前 Jones 后端是单色 whole-fluence，而物理多程需要 spectral_time；保留多程求解器并退回标量偏振。": "The current Jones backend is monochromatic whole-fluence while physical multipass requires spectral_time; the multipass solver was retained and polarization was changed to scalar.",
    "标量偏振不能继续持有 Jones 双折射算符。": "Scalar polarization cannot retain a Jones birefringence operator.",
    "因果 multipass 模板拥有每一段程间光路，不能再叠加旧 between_pass 元件。": "The causal multipass template owns every inter-pass path and cannot be combined with legacy between_pass elements.",
    "因果 multipass 使用自己的 per-transition 透过率；旧全局程间透过率必须为 1，避免重复扣损耗。": "Causal multipass uses its own per-transition transmission; the legacy global inter-pass transmission must be one to avoid counting loss twice.",
    "因果 multipass 的成像/坐标变换由模板显式给出，不能再叠加旧全局翻转。": "Causal multipass specifies imaging and coordinate transforms explicitly in its template, so the legacy global flip cannot also be applied.",
    "多程模板的晶体命中数是唯一程次数来源。": "The multipass template crystal-hit count is the sole source of the pass count.",
    "simulation.passes 是因果模板编译后命中数的派生断言。": "simulation.passes is a derived assertion of the hit count compiled from the causal template.",
    "多程 pulse-map 必须把程间暗弛豫放在同一绝对时间账本中，才能闭合周期边界。": "A multipass pulse map must place inter-pass dark relaxation on the same absolute-time ledger to close the periodic boundary.",
    "双折射算符作用于 Jones 两分量；标量偏振下必须关闭。": "The birefringence operator acts on two Jones components and must be off for scalar polarization.",
    "当前 Jones 偏振传播使用单色 whole-fluence 两分量场。": "The current Jones polarization path uses a monochromatic two-component whole-fluence field.",
    "Jones Ex/Ey 必须共享每个横向单元的局域增益，不能使用标量模态重叠或抛物线拟合闭合。": "Jones Ex/Ey must share local gain in every transverse cell and cannot use scalar modal-overlap or parabolic-fit closures.",
    "当前 Jones 后端只支持单色一频点传播。": "The current Jones backend supports only monochromatic single-frequency propagation.",
    "当前 Jones 后端只支持一个单色频点。": "The current Jones backend supports only one monochromatic frequency bin.",
    "Jones 两分量与互不相干 HG 模态组尚无联合传播合同；保留等效 M² 场。": "No joint propagation contract yet exists for two Jones components and an incoherent HG ensemble; an effective-M2 field is retained.",
    "径向/切向热双折射必须读取三维温度场。": "Radial/tangential thermal birefringence must read a 3D temperature field.",
    "径向/切向热双折射从三维热折射率场恢复温度差。": "Radial/tangential thermal birefringence recovers temperature differences from the 3D thermo-optic index field.",
    "径向平面应力降阶式与独立 FEM 应力 Δn 不能同时叠加。": "The reduced radial plane-stress model cannot be combined with an independent FEM stress-induced delta-n.",
    "径向平面应力 Jones 模型自己生成双折射，关闭独立张量光弹耦合以避免重复计数。": "The radial plane-stress Jones model generates birefringence itself, so independent tensor photoelastic coupling is disabled to avoid double counting.",
    "distributed_opd 依赖 FEM 应力/位移场，而径向平面应力 Jones 降阶式要求 stress_model=none；已关闭冲突的端面形变。": "distributed_opd depends on FEM stress/displacement fields while the reduced radial plane-stress Jones model requires stress_model=none; the conflicting end-face deformation was disabled.",
    "张量光弹 Jones 传播必须先求三维温度场。": "Tensor-photoelastic Jones propagation must first solve a 3D temperature field.",
    "张量光弹 Jones 传播需要三维热/应变折射率场。": "Tensor-photoelastic Jones propagation requires a 3D thermal/strain refractive-index field.",
    "张量光弹 Jones 传播必须读取各向异性热弹性 FEM 应变张量。": "Tensor-photoelastic Jones propagation must read the anisotropic thermoelastic FEM strain tensor.",
    "张量 Jones 延迟器需要 FEM 的 tensor_photoelastic 耦合输出。": "The tensor Jones retarder requires the FEM tensor_photoelastic coupling output.",
    "Commercial App 脉内弛豫必须更新真实上下能级布居；外加小信号增益没有可推进的储能库。": "Commercial App intrapulse relaxation must update physical upper/lower level populations; an imposed small-signal gain has no stored-energy reservoir to advance.",
    "M²x/M²y 不是 1": "M2x/M2y is not one",
    "束腰": "waist",
    "横向阶数": "transverse order",
    "不相等": "are unequal",
    "x/y 焦点不相等": "x/y focal positions are unequal",
    "Seed 与光学网格中心不重合": "the Seed and optical-grid centers do not coincide",
    "Seed 存在入射倾角": "the Seed has a nonzero incidence angle",
    "存在波长相关空间源": "a wavelength-dependent spatial source is active",
    "轴对称降阶不适用于当前物理光束（": "The axisymmetric reduction is inapplicable to the current physical beam (",
    "）；不改写束腰、M²、中心或角度，改用对应的完整笛卡尔相干求解器。": "); the waist, M2, center, and angle were preserved and the corresponding full Cartesian coherent solver is used.",
    "所选相干精度档包含 front→peak→tail 软窗增益，必须启用瞬态增益算符。": "The selected coherent-fidelity profile includes front-to-peak-to-tail soft-window gain and therefore requires the transient-gain operator.",
    "所选相干精度档使用整脉冲增益闭合，不能同时叠加软窗瞬态增益。": "The selected coherent-fidelity profile uses a whole-pulse gain closure and cannot also apply soft-window transient gain.",
    "轴对称相干传播依赖对称自适应局域算符分裂。": "Axisymmetric coherent propagation requires symmetric adaptive local-operator splitting.",
    "L4.5 轴对称瞬态增益必须由完整逐脉冲映射闭合周期布居。": "L4.5 axisymmetric transient gain must close the periodic population with the complete pulse map.",
    "自由载流子由时间域强场电离/吸收算符产生；强场损耗关闭时该子状态必须停用，重新开启父模块后会恢复原设置。": "Free carriers are generated by the time-domain strong-field ionization/absorption operator; this child state must be inactive when strong-field loss is off and its prior setting is restored when the parent is re-enabled.",
    "轴对称 Kerr 初值代理只与 axisymmetric_kerr 主求解器共享同一降阶自由度；当前主求解器改用相位不敏感初值。": "The reduced axisymmetric Kerr initializer shares its reduced degrees of freedom only with the axisymmetric_kerr main solver; the current main solver uses the phase-insensitive initializer instead.",
    "局域抛物线增益拟合只属于 fluence/Commercial App 的降阶空间闭合；当前后端保留自己的局域增益算符。": "The local parabolic gain fit belongs only to the reduced spatial closure used by fluence/Commercial App; the current backend retains its own local gain operator.",
    "抛物线增益拟合与当前 stationary photon-flux 闭合只支持 CW；有限脉冲保留局域增益算符。": "The parabolic gain fit and its stationary photon-flux closure support CW only; finite pulses retain the local gain operator.",
    "Gaussian 模态重叠闭合只支持 fluence 或 Commercial App；多色、JOSA 和 coherent CPA 保留各自的局域增益算符。": "The Gaussian modal-overlap closure supports only fluence or Commercial App; multicolor, JOSA, and coherent CPA retain their respective local gain operators.",
    "Gaussian 模态重叠当前只定义单色标量增益；宽带 Seed 改回逐波长局域增益。": "Gaussian modal overlap currently defines only scalar monochromatic gain; a broadband Seed falls back to wavelength-resolved local gain.",
    "Gaussian 模态重叠只接受 effective_m2、纯 HG00，或 Commercial App 的逐模态 HG 共享布居合同。": "Gaussian modal overlap accepts effective_m2, pure HG00, or Commercial App's per-mode HG shared-population contract only.",
    "Gaussian 模态重叠不能代表超高斯、top-hat、环形或自定义横向形状；保留局域增益。": "Gaussian modal overlap cannot represent super-Gaussian, top-hat, annular, or custom transverse profiles; local gain is retained.",
    "当前模态重叠投影是轴对齐椭圆；旋转椭圆必须使用局域场增益。": "The current modal-overlap projection assumes an axis-aligned ellipse; a rotated ellipse must use local-field gain.",
    "寿命量子亏损需要周期平均/稳态上能级布居；当前瞬态或 pulse-map 只返回周期边界布居，已退回不重复计数的量子亏损账本。若要逐事件热量，请使用 event_energy_ledger。": "Lifetime quantum-defect heating requires a cycle-averaged or steady-state upper-level population; the current transient or pulse-map solver returns only the population at cycle boundaries, so the non-duplicating quantum-defect ledger is used instead. Use event_energy_ledger for event-resolved heat.",
    "当前 ASE 自发源只支持一个独立 Signal 上能级储库；所选能级/非均匀展宽网络包含多个储库，已关闭 ASE 而不合并真实布居。": "The current ASE spontaneous source supports only one independent Signal upper-level reservoir; the selected level or inhomogeneous-broadening network has multiple reservoirs, so ASE is disabled without merging physical populations.",
    (
        "full_3d_pc 的实际求解合同固定使用三维热场（auto 默认 steady_3d，"
        "也可预先指定 transient_3d）；这里仍保留退出该层级后的用户配置。"
    ): (
        "The full_3d_pc runtime contract always uses a 3D thermal field "
        "(auto resolves to steady_3d, while transient_3d may be requested "
        "explicitly). The user's post-full_3d_pc setting remains preserved."
    ),
    (
        "“用户选择”保留入口意图，“实际生效”来自预设解析和全部依赖闭包；"
        "自动托管表示该行至少有一个从属值会随上级条件释放。"
    ): (
        "“Requested” preserves the entry intent; “Effective” includes preset "
        "resolution and the complete dependency closure. Auto-managed means "
        "at least one child value in the row will be released with its parent."
    ),
    "查看自动托管明细（原值 → 生效值）": (
        "View Auto-Managed Details (Previous → Effective)"
    ),
    "保留当前联动值，转为手动设置": (
        "Keep Current Coordinated Values as Manual Settings"
    ),
    (
        "清除这些值的自动托管标记；以后关闭上级功能时，"
        "只要配置仍然相容，当前值就不会被恢复。"
    ): (
        "Remove automatic ownership from these values. If the configuration "
        "remains compatible, disabling the parent later will no longer restore them."
    ),
    (
        "上级条件关闭后，这些值会自动恢复联动前设置；材料常数、光束尺寸、"
        "功率和能量不会被相容性联动猜测。"
    ): (
        "When the parent condition is disabled, these values return to their "
        "pre-coordination settings. Compatibility coordination never guesses "
        "material constants, beam dimensions, power, or energy."
    ),
    "当前有 ": "There are currently ",
    " 项从属设置由依赖合同自动托管；配置已通过运行前校验。": (
        " child settings auto-managed by dependency contracts; the configuration passed pre-run validation."
    ),
    "最近已释放 ": "Recently released ",
    " 项不再需要的从属设置并恢复原值；当前配置可运行。": (
        " child settings that were no longer required and restored their previous values; the configuration is runnable."
    ),
    "运行前相容性检查通过：Seed `": "Pre-Run Compatibility Check Passed: Seed `",
    (
        "保持 auto 表示跟随精度预设；只有这里显式选择的值才覆盖预设。"
        "full_3d_pc 和周期稳态的硬约束会在下方锁定并注明来源。"
    ): (
        "Keep a setting on auto to follow the selected precision preset. "
        "Only explicit choices here override that preset. Hard constraints "
        "from full_3d_pc and periodic steady state are locked below with "
        "their dependency source shown."
    ),
    "**计算性能**": "**Compute Performance**",
    "运行前冷态 q 参数用于快速调光；运行后的虚线 D4σ 和热场结果会显示在结果区。": (
        "The pre-run cold-state q parameters support quick alignment. "
        "Post-run D4σ samples and thermal-field results appear in the results area."
    ),
    "当前 ": "Current ",
    " · 真实总光程 ": " · Total Physical Path Length ",
    "晶体内": "In Crystal",
    "晶体预留段（外部介质）": "Reserved Crystal Segment (External Medium)",
    "最小 D4σ 半径：x ": " · Minimum D4σ Radius: x ",
    " µm，y ": " µm, y ",
    "实线是运行前冷态 q 包络；每束只有上、下两条线，包络同时包含 x/y 两方向。": (
        "Solid lines show the pre-run cold-state q envelope. Each beam uses one "
        "upper and one lower boundary covering both x and y."
    ),
    "橙色矩形是共享的同一块 gain。": (
        " The orange rectangle marks the shared gain crystal."
    ),
    "生成从 Pump/Seed 发光面、实际光路、能级布居、局部增益、热与应力反馈直到出光面验证的物理说明。": (
        "Generate a physics description from the Pump/Seed source planes and "
        "physical optical path through energy-level populations, local gain, "
        "thermal and stress feedback, and output-plane validation."
    ),
    "下载完整结果包（可在结果分析页重新载入）": (
        "Download Full Result Archive (Reload in Result Analysis)"
    ),
    (
        "这是推荐归档格式；一个文件即可恢复本页的完整可视化和诊断。"
        "包内 SHA-256 只校验内容完整性，不是作者身份或来源签名。"
    ): (
        "This is the recommended archive format. One file restores every "
        "visualization and diagnostic on this page. The archive's SHA-256 "
        "checks content integrity only; it is not an author or provenance signature."
    ),
    "从左侧四个工作区完成设置后点击“运行当前配置”。建议先用 balanced 和较小网格确认工作流，再在“材料 / 热”和“网格 / 验证”页打开详细模型并做收敛检查。": (
        "Complete the four workspaces on the left, then click “Run Current "
        "Configuration.” Start with the balanced preset and a smaller grid to "
        "verify the workflow, then enable detailed models and convergence checks "
        "under Materials / Thermal and Grid / Validation."
    ),
    (
        "清爽设置收起逐项后端覆盖和长参数区；完整设置显示全部入口并展开高级求解器。"
        "Pump、Seed、材料和网格长区仍按需展开，避免滚动时丢失工作区上下文。"
    ): (
        "Essentials collapses per-feature backend overrides and long parameter "
        "sections. Full Settings exposes every entry and expands the advanced "
        "solver. Long Pump, Seed, material, and grid sections still open on demand "
        "so their workspace context remains visible while scrolling."
    ),
    (
        "完整设置已启用：全部参数入口可用；长表单保持折叠，当前模块摘要始终可见。"
    ): (
        "Full Settings is enabled: every parameter entry is available; long "
        "forms remain collapsed and the current module summary stays visible."
    ),
    "Pump 与 Seed 分开定义；": "Pump and Seed are defined separately; ",
    "Pump 与 Seed 分开定义；Pump ": "Pump and Seed are defined separately; Pump ",
    "材料链摘要：热 ": "Material-chain summary: Thermal ",
    " · 折射率 ": " · Refractive index ",
    " · 应力 ": " · Stress ",
    "网格摘要：": "Grid summary: ",
    " · 独立光学网格 ": " · Independent optical grid ",
    "依赖锁定来源：": "Dependency lock source: ",
    "热光固定点": "Thermo-Optic Fixed Point",
    (
        "。热场固定为 steady_3d，折射率固定为 thermal_3d；关闭最后一个上游条件后，"
        "自动托管值会恢复到联动前设置。"
    ): (
        ". The thermal field is fixed to steady_3d and the refractive index to "
        "thermal_3d. After the last upstream condition is disabled, auto-managed "
        "values return to their pre-coordination settings."
    ),
    "local_3d 光谱反馈": "local_3d spectroscopic feedback",
    "Seed 强场产热固定点": "Seed strong-field heat fixed point",
    (
        "full_3d_pc 的求解合同要求三维稳态热场；切换 JOSA 模式后才会释放该限制。"
    ): (
        "The full_3d_pc solver contract requires a 3D steady thermal field; "
        "switching JOSA mode releases this constraint."
    ),
    "实时预览控制台": "Live Preview Console",
    "**实时预览控制台**": "**Live Preview Console**",
    "预览中插入增益晶体": "Insert Gain Crystal in Preview",
    "聚焦到晶体区域": "Focus on Crystal Region",
    (
        "开启后只压缩预览中的长自由空间，让晶体附近的腰斑和元件更容易比较；"
        "真实光程、光束计算和正式求解配置都不会改变。"
    ): (
        "When enabled, only long free-space segments in the preview are "
        "compressed so waists and elements near the crystal are easier to "
        "compare. Physical path length, beam calculations, and the solver "
        "configuration remain unchanged."
    ),
    (
        "上面两个开关只改变冷态预览，不会开关正式求解中的晶体、热场、增益或传播模块。"
    ): (
        "The two switches above affect only the cold-state preview; they do not "
        "enable or disable the crystal, thermal field, gain, or propagation modules in the solver."
    ),
    "载入与复核流程": "Load and Review Workflow",
    "### 载入与复核流程": "### Load and Review Workflow",
    "1 · 载入": "1 · Load",
    "完整结果包与配置快照": "complete result archive and configuration snapshot",
    "2 · 体检": "2 · Validate",
    "整组有效性、告警与失败点": "set-wide validity, warnings, and failed cases",
    "3 · 深入": "3 · Explore",
    "逐点三维场、光路和光谱": "per-case 3D fields, optical paths, and spectra",
    "**1 · 载入**  完整结果包与配置快照": (
        "**1 · Load**  Complete result archive and configuration snapshot"
    ),
    "**2 · 体检**  整组有效性、告警与失败点": (
        "**2 · Validate**  Set-wide validity, warnings, and failed cases"
    ),
    "**3 · 深入**  逐点三维场、光路和光谱": (
        "**3 · Explore**  Per-case 3D fields, optical paths, and spectra"
    ),
    (
        "载入只解析结果快照，不会重跑求解器，也不会覆盖放大器仿真页的当前表单。"
    ): (
        "Loading parses the result snapshot only; it neither reruns the solver "
        "nor overwrites the current form on the Amplifier Simulation page."
    ),
    (
        "结果包会同时保留配置、三维场、记录光学面、validation 结论、"
        "运行 warning 和自动依赖托管记录。"
    ): (
        "The archive preserves the configuration, 3D fields, recorded optical "
        "planes, validation conclusions, runtime warnings, and automatic-dependency ownership records."
    ),
    "本次运行结果": "Current Run Results",
    (
        "以下内容来自一次完整、不可拼接的配置快照；表单后续修改只更新上方冷态预览，"
        "直到再次运行才会替换这里的布居、热场、增益和输出结果。"
    ): (
        "The content below comes from one complete, non-composable configuration "
        "snapshot. Later form edits update only the cold-state preview above; "
        "population, thermal, gain, and output results here change only after another run."
    ),
    "结果摘要": "Result Summary",
    "结果快照：Seed ": "Result snapshot: Seed ",
    " · 布居 ": " · Population ",
    " · 热 ": " · Thermal ",
    " · 网格 ": " · Grid ",
    (
        "最多耦合迭代不能小于最少迭代；已同步扩大上限，"
        "避免固定点在达到允许停止条件前被截断。"
    ): (
        "Maximum coupling iterations cannot be below the minimum. The upper "
        "limit was increased so the fixed point is not truncated before it may stop."
    ),
    (
        "所选 x/y HG 基底包含更多独立模态；已同步扩大传播容量，"
        "避免只显示高阶设置却在求解前被容量上限阻断。"
    ): (
        "The selected x/y HG basis contains more independent modes. Propagation "
        "capacity was increased so the displayed high-order basis is not blocked before the run."
    ),
    (
        "逐脉冲固定点必须在最少周期之后仍留出连续满足容差的周期；"
        "已扩大最多周期数，使所选停止条件在数学上可达。"
    ): (
        "The pulse-by-pulse fixed point needs enough cycles after the minimum "
        "to satisfy the tolerance consecutively. The maximum was increased so "
        "the selected stopping condition is mathematically reachable."
    ),
    (
        "周期平均的最多自洽迭代不能小于最少迭代；已同步扩大上限。"
    ): (
        "Maximum cycle-average iterations cannot be below the minimum; the upper limit was increased."
    ),
    (
        "自由载流子在相邻脉冲间不能充分衰减，cycle-average 会丢失真实状态；"
        "已切换为逐脉冲周期固定点。"
    ): (
        "Free carriers do not decay sufficiently between pulses, so cycle-average "
        "would discard a real state. The method was changed to a pulse-by-pulse periodic fixed point."
    ),
    (
        "上级依赖已不再要求该值，已释放自动托管并恢复联动前设置。"
    ): (
        "The upstream dependency no longer requires this value; automatic "
        "ownership was released and the pre-coordination setting restored."
    ),
    "百分比按真实模块和固定点迭代推进；自适应提前收敛时会跳到下一阶段，ETA 只是根据已完成比例给出的动态估计。": (
        "Progress follows the actual modules and fixed-point iterations. If an "
        "adaptive stage converges early, progress advances to the next stage; "
        "ETA is a dynamic estimate based on completed work."
    ),
    "默认先打开可拖动旋转的三维光路；“三维场与可拖动切片”可选择任意已保存体场、切换 x/y/z 方向并拖动切片位置。这里显示的是本次结果快照，不会重新求解。": (
        "The draggable 3D optical path opens first. Under “3D Fields and "
        "Draggable Slices,” select any saved volume field, choose the x, y, or z "
        "orientation, and drag the slice position. This is a saved result snapshot; "
        "no solver rerun occurs."
    ),
    "查看其他结果指标": "View Additional Result Metrics",
    "查看模型适用性与布居数值修正": (
        "View Model Applicability and Population Numerical Corrections"
    ),
    "查看 Seed 各波长在增益入口的空间状态": (
        "View Seed Spatial State by Wavelength at the Gain Entrance"
    ),
    "查看实际增益输运后端": "View the Active Gain-Transport Backend",
    "x 截面（y=": "x cross-section (y=",
    "y 截面（x=": "y cross-section (x=",
    " 峰值处 x / y 光斑截面": " x / y Beam Profiles at the Peak",
    "冷态实线/半透明光束体来自运行前 q 参数；运行后虚线和采样环来自求解器实际复场的二阶矩，因此包含本次计算采用的材料折射率、热场 Δn、增益、孔径裁剪和 FFT 传播。采样面之间仅连线辅助阅读，不宣称是连续热态 q 解。": (
        "Cold-state solid lines and translucent beam envelopes come from the "
        "pre-run q parameters. Post-run dashed outlines and sampling rings come "
        "from second moments of the solver’s actual complex field, so they include "
        "the material refractive index, thermal Δn, gain, aperture clipping, and "
        "FFT propagation used in this run. Lines between sampling planes are visual "
        "guides only and do not claim a continuous hot-state q solution."
    ),
    "当前详细标签（Pump 与 Seed 光束体始终同时显示）": (
        "Active Detail Beam (Pump and Seed Envelopes Remain Visible)"
    ),
    "晶体内叠加场（映射到第一程晶体）": (
        "In-Crystal Overlay Field (Mapped to the First Crystal Pass)"
    ),
    "泵后反转密度": "Post-Pump Inversion Density",
    "光束": "Beam",
    "外缘能量占比": "Boundary Energy Fraction",
    "周期 FFT 合同适用": "Periodic FFT Applies",
    "外缘占比超阈值": "Boundary Fraction Above Threshold",
    "程次": "Pass",
    "z 分箱": "z Bin",
    "采样语义": "Sampling Semantics",
    "外缘能量占比取晶体路径上最坏截面；超过 0.1% 时提示扩大光学窗口，只有所选后端在该路径上实际使用周期 FFT 时才会判为绕回风险；非周期后端保留占比供诊断，但不会误判为 FFT 失效。": (
        "The boundary-energy fraction reports the worst cross-section along the "
        "crystal path. Values above 0.1% prompt an optical-window expansion. A "
        "wraparound risk is reported only when the selected backend actually uses "
        "periodic FFT propagation on that path; nonperiodic backends retain the "
        "fraction for diagnostics without falsely flagging FFT failure."
    ),
    "半透明连续光束体是冷态 q 预览；虚线椭圆环是运行后实际 D4σ 截面，晶体内部按光学 z 分箱自动采样：Pump 为单元平均并放在中心，Seed 为每个 split-step 单元出口。当前选择的 Seed/Pump 最后绘制，不再被另一束遮住。": (
        "The translucent continuous beam envelope is the cold-state q preview; "
        "dashed elliptical rings are actual post-run D4σ cross-sections. Inside "
        "the crystal, samples follow the optical z bins: Pump values are cell "
        "averages placed at cell centers, while Seed values are taken at each "
        "split-step cell exit. The selected Seed or Pump beam is drawn last so the "
        "other beam does not obscure it."
    ),
    "逐程与记录光学面": "Pass-by-Pass Results and Recorded Optical Planes",
    "表中 M² 是各采样复场的原始二阶矩；顶部结果会明确标注其为采样二阶矩、配置模态合同或包含 effective-M² 传输因子的组合代理量。": (
        "Table M² values are raw second moments of the sampled complex fields. "
        "The headline result explicitly identifies whether it is a sampled second "
        "moment, a configured modal contract, or a combined proxy that includes the "
        "effective-M² transport factor."
    ),
    "执行当前配置，并在右侧显示真实模块、固定点迭代、残差、速度和粗略剩余时间。": (
        "Run the current configuration and show the active modules, fixed-point "
        "iterations, residuals, speed, and estimated time remaining."
    ),
    "载入一个完整结果包，先检查整组工况的有效性与告警，再逐点恢复三维场、光路、光谱和诊断。": (
        "Load a complete results package to review validity and warnings across "
        "all cases, then restore each case’s 3D fields, optical path, spectra, "
        "and diagnostics."
    ),
    "请先在“放大器仿真”页运行单个工况或参数遍历，并下载完整结果包；之后可在这里离线重新载入，不需要再次求解。": (
        "Run a single case or parameter sweep under Amplifier Simulation and "
        "download the complete results package. You can then reload it here "
        "offline without running the solver again."
    ),
    "未收敛": "Not Converged",
    "平均稳态未收敛": "Average Steady State Not Converged",
    "CW 泵未收敛": "CW Pump Not Converged",
    "尚不可判定": "Not Yet Determined",
    "未配置独立 τrad": "No Independent τrad Configured",
    "完全固定的晶体表面（留空=全部无牵引）": (
        "Fully Fixed Crystal Faces (Empty = All Traction-Free)"
    ),
    "超过预算会明确报错，不会静默粗化应力热点。可使用较粗物理网格和独立精细光学网格。": (
        "Exceeding the budget raises an explicit error; stress hot spots are never "
        "silently coarsened. A coarser physical grid may be paired with a separate "
        "fine optical grid."
    ),
    "不改变传播物理，只决定保存末程、累计或逐程三维诊断场。": (
        "Does not change propagation physics; it only selects last-pass, cumulative, "
        "or per-pass 3D diagnostic-field storage."
    ),
    "使用固定折射率，不把求得的温度或应力场反馈到光学传播。": (
        "Uses a fixed refractive index and does not feed the solved temperature or "
        "stress field back into optical propagation."
    ),
    "按椭圆归一化半径构造二维径向超高斯，而不是 x、y 两个一维函数的乘积。": (
        "Builds a 2D radial super-Gaussian from the normalized elliptical radius, "
        "not a product of independent 1D x and y functions."
    ),
    "传播复光谱场 E(x,y,ω)，经逆 FFT 得到瞬时时域场；增益、色散、Kerr 相位和不可逆强场损耗按分步算符组合，因此可以预测相位、展宽与压缩质量。": (
        "Propagates the complex spectral field E(x,y,ω) and obtains the instantaneous "
        "time-domain field by inverse FFT. Split operators combine gain, dispersion, "
        "Kerr phase, and irreversible strong-field loss, enabling phase, broadening, "
        "and compression-quality predictions."
    ),
    "固定随机种子会让热光/布居迭代重复使用同一束光线，避免 Monte Carlo 噪声被误判为不收敛。射线数控制统计误差，晶体网格控制沉积空间分辨率。": (
        "A fixed random seed reuses the same ray bundle across thermo-optic/population "
        "iterations so Monte Carlo noise is not mistaken for non-convergence. Ray count "
        "controls statistical error; the crystal grid controls deposition resolution."
    ),
    "开启 Anderson 加速并保留残差/守恒回退；关闭后使用原始松弛固定点迭代。": (
        "Enables Anderson acceleration with residual and conservation safeguards; "
        "when disabled, the original relaxed fixed-point iteration is used."
    ),
    "开启后 Seed 提取会降低局部反转并限制增益；关闭时使用小信号增益。": (
        "When enabled, Seed extraction depletes local inversion and limits gain; "
        "when disabled, small-signal gain is used."
    ),
    "开启时 Seed 的局部受激跃迁会反向扣减布居并限制输出；关闭时保持小信号增益。": (
        "When enabled, local Seed-stimulated transitions update the populations and "
        "limit output; when disabled, the small-signal gain is retained."
    ),
    "能量去向与热源：只把不可逆部分送入导热方程": (
        "Energy Destinations and Heat Sources: Send Only Irreversible Terms to the "
        "Heat-Conduction Equation"
    ),
    "要求界面、孔径和程间能量差都落入具名账本；任何未解释损耗都会使严格结果无效。": (
        "Requires interface, aperture, and between-pass energy differences to appear "
        "in named ledger terms; any unexplained loss invalidates a strict result."
    ),
    "逐事件账本会保留导入配置中的 decay_partitions 和 nonlinear_partitions；每条通道必须明确分配到热、荧光、ASE 或未解析项。高级分配请在 JSON 中编辑。": (
        "The event ledger preserves imported decay_partitions and nonlinear_partitions. "
        "Every channel must be assigned explicitly to heat, fluorescence, ASE, or an "
        "unresolved term; edit advanced allocations in JSON."
    ),
    " · 共 ": " · total ",
    " 轮": " iterations",
    "准备中": "Preparing",
    "（上限 ": " (limit ",
    "固定点": "Fixed point",
    "，共 ": ", total ",
    "固定点准备中": "Preparing fixed point",
    "固定点第 ": "Fixed-point iteration ",
    "解析正反向脉冲在晶体中的同时竞争": (
        "Resolve Simultaneous Forward/Backward Pulse Competition in the Crystal"
    ),
    "在绝对时间 CFL 网格上同时推进正向和反向时间片；镜距决定真实往返延迟，同一时刻的两个方向共享局部饱和增益。": (
        "Advance forward and backward time slices on one absolute-time CFL grid. "
        "Mirror distance sets the physical round-trip delay, and both directions "
        "share the local saturated gain when they coexist."
    ),
    "在由晶体群速度确定的绝对时间 CFL 网格上，同时推进正向和反向脉冲。镜面位置与实际回程光路决定反射延迟；同一时间、同一晶体单元中的两个方向共同消耗一次局部饱和增益事件，因此镜面贴面只是零外部延迟极限。": (
        "Advance forward and backward pulses together on an absolute-time CFL grid "
        "set by the crystal group velocity. Mirror position and the physical return "
        "path set the reflection delay; both directions consume one shared local "
        "saturated-gain event when they occupy the same cell and time, so a mirror "
        "on the crystal is only the zero-external-delay limit."
    ),
    "已锁定 spectral_time、两程和反向第二程。镜面贴合是 D=0 的极限；非零 D 会由传播时间自然产生部分重叠或无重叠。": (
        "Locked to spectral_time, two passes, and a reversed second pass. A mirror "
        "on the crystal is the D=0 limit; nonzero D naturally produces partial or "
        "zero overlap through flight time."
    ),
    "直接给镜距（自动生成去程→镜面→回程）": (
        "Set Mirror Distance (Build Outbound → Mirror → Return Automatically)"
    ),
    "使用上方程间光路 JSON": "Use the Interpass Optical-Path JSON Above",
    "回程路径来源": "Return-Path Source",
    "在同一绝对时间轴上推进能级暗弛豫": (
        "Advance Dark Level Relaxation on the Same Absolute-Time Clock"
    ),
    "开启后，能级寿命和已配置的暗跃迁与正反向光场共用同一个绝对时间账本；关闭仅适合已证明脉冲/往返时间远短于所有弛豫时间的近似。": (
        "When enabled, level lifetimes and configured dark transitions share one "
        "absolute-time ledger with the forward/backward fields. Disable only for a "
        "validated approximation in which pulse and round-trip times are much shorter "
        "than every relaxation time."
    ),
    "晶体后表面→镜面单程距离 D (m)": (
        "Crystal Back Surface → Mirror One-Way Distance D (m)"
    ),
    "回程镜功率反射率": "Return-Mirror Power Reflectivity",
    "双向绝对时间最大分箱数": "Maximum Bidirectional Absolute-Time Bins",
    "镜距介质相位折射率": "Mirror-Gap Phase Refractive Index",
    "镜距介质群折射率": "Mirror-Gap Group Refractive Index",
    "直接镜距模式已经自动生成完整往返光路；请清空上方程间光路 JSON，或改选“使用上方程间光路 JSON”，避免重复计算距离。": (
        "Direct mirror-distance mode already builds the complete round-trip path. "
        "Clear the interpass optical-path JSON above, or select the JSON return path, "
        "to avoid counting the distance twice."
    ),
    "能量加权边界占比": "Energy-Weighted Boundary Fraction",
    "最坏单频边界占比": "Worst Single-Frequency Boundary Fraction",
    "该频点总能量权重": "Total-Energy Weight of This Frequency Bin",
    "最坏频点波长": "Worst-Bin Wavelength",
    "轴对称径向边界阻断只使用总能量加权占比；最坏单频值仅用于定位。能量权重可忽略的频谱尾部即使局部比例很高，也不会单独判废整个结果。": (
        "Axisymmetric radial-boundary blocking uses only the total-energy-weighted "
        "fraction; the worst single-frequency value is diagnostic only. A negligible-"
        "energy spectral tail does not invalidate the whole result solely because its "
        "local fraction is high."
    ),
}


# Causal finite-multipass form and result diagnostics.  Keeping these exact
# source strings together makes it straightforward to audit the conditional
# form: none of these controls are rendered in the default single-pass mode.
_EN_OVERRIDES.update(
    {
        "先把双程、双侧镜、Herriott、4f 中继或论文逐程条件编译成晶体命中序列，再在同一个绝对时间 CFL 网格上推进每一程。重叠由脉宽、晶体群速度和程间光程自然产生；同一时间真正落在同一三维网格单元的各程通量共同更新局部布居。": (
            "Compile the double-pass, two-sided-mirror, Herriott, 4f-relay, or paper "
            "per-pass conditions into a crystal-hit sequence, then advance every pass "
            "on one absolute-time CFL grid. Pulse duration, crystal group velocity, and "
            "inter-pass optical path create overlap naturally; all passes that actually "
            "occupy the same 3D cell at the same time update its local population together."
        ),
        "重叠没有开关：程序由脉冲包络、晶体群速度、每段外部光程和每程二维命中位置自动判断。时间重叠但光斑不相交时，也不会产生假的局部增益竞争。": (
            "There is no overlap switch. The pulse envelope, crystal group velocity, "
            "external path length, and 2D hit location determine overlap automatically. "
            "Time-coincident but spatially separate spots do not create false local gain competition."
        ),
        "双程返回镜 — 晶体后单镜折返": "Double-Pass Return Mirror — One Mirror Behind the Crystal",
        "双侧镜有限多程 — 理想开关或几何出射": "Finite Two-Sided-Mirror Multipass — Ideal Switch or Geometric Extraction",
        "Herriott 两曲面镜 — 旁轴光斑轨道": "Two-Curved-Mirror Herriott Cell — Paraxial Spot Orbit",
        "4f / 中继成像 — 控制每程束腰演化": "4f / Relay Imaging — Control Per-Pass Waist Evolution",
        "论文逐程边界条件 — 直接给命中/延时": "Paper Per-Pass Boundary Conditions — Supply Hits and Delays",
        "多程模板": "Multipass Template",
        "x 曲率半径 Rx (m；0=平面)": "x Curvature Radius Rx (m; 0 = Plane)",
        "y 曲率半径 Ry (m；0=平面)": "y Curvature Radius Ry (m; 0 = Plane)",
        "镜面中心 x (m)": "Mirror Center x (m)",
        "镜面中心 y (m)": "Mirror Center y (m)",
        "有效口径半径 ax (m；0=不限)": "Clear-Aperture Radius ax (m; 0 = Unlimited)",
        "有效口径半径 ay (m；0=不限)": "Clear-Aperture Radius ay (m; 0 = Unlimited)",
        "模板只生成晶体命中序列和程间算子，不要求在三维画布中摆放镜架。论文模式用于文献没有给完整镜路、只给逐程条件的情况。": (
            "A template generates only the crystal-hit sequence and inter-pass operators; "
            "it does not require placing mirror mounts on a 3D canvas. Paper mode is for "
            "sources that report per-pass conditions but omit the complete mirror path."
        ),
        "双程模板固定为 2 次晶体命中；D=0 只是镜面贴端面的极限。": (
            "The double-pass template has exactly two crystal hits; D = 0 is only the "
            "limit in which the mirror touches the end face."
        ),
        "##### 两次晶体命中边界": "##### Two Crystal-Hit Boundaries",
        "所有物理多程模板共用的时间、介质与数值设置": "Shared Timing, Medium, and Numerical Settings for Physical Multipass Templates",
        "暗弛豫最大时间步 (s；0=自动)": "Maximum Dark-Relaxation Time Step (s; 0 = Auto)",
        "双程返回镜": "Double-Pass Return Mirror",
        "双侧镜": "Two-Sided Mirrors",
        "4f 中继": "4f Relay",
        "逐程边界": "Per-Pass Boundaries",
        "因果 ": "Causal ",
        "物理多程模板 — 自动时序、空间重叠与共享布居（推荐）": "Physical Multipass Template — Automatic Timing, Spatial Overlap, and Shared Population (Recommended)",
        "多程建模方式": "Multipass Modeling Method",
        "物理多程模板": "Physical Multipass Template",
        "晶体端面 → 镜面距离 (m)": "Crystal End Face → Mirror Distance (m)",
        "晶体后端面 → 返回镜 D (m)": "Crystal Back End Face → Return Mirror D (m)",
        "返回镜功率反射率": "Return-Mirror Power Reflectivity",
        "返回镜 Rx (m；0=平面)": "Return-Mirror Rx (m; 0 = Plane)",
        "返回镜 Ry (m；0=平面)": "Return-Mirror Ry (m; 0 = Plane)",
        "返回镜中心 x (m)": "Return-Mirror Center x (m)",
        "返回镜中心 y (m)": "Return-Mirror Center y (m)",
        "返回镜有效半口径 ax (m；0=不限)": "Return-Mirror Clear Semiaperture ax (m; 0 = Unlimited)",
        "返回镜有效半口径 ay (m；0=不限)": "Return-Mirror Clear Semiaperture ay (m; 0 = Unlimited)",
        "首程中心 x (m)": "First-Pass Center x (m)",
        "首程中心 y (m)": "First-Pass Center y (m)",
        "回程中心 x (m)": "Return-Pass Center x (m)",
        "回程中心 y (m)": "Return-Pass Center y (m)",
        "首程 θx (°)": "First-Pass θx (°)",
        "首程 θy (°)": "First-Pass θy (°)",
        "回程局部 θx (°)": "Return-Pass Local θx (°)",
        "回程局部 θy (°)": "Return-Pass Local θy (°)",
        "##### 注入主光线（模板据此生成每程光斑轨道）": "##### Injection Chief Ray (Generates the Per-Pass Spot Orbit)",
        "每段程间附加功率透过率": "Additional Power Transmission per Inter-Pass Segment",
        "外部路径相位折射率": "External-Path Phase Refractive Index",
        "外部路径群折射率": "External-Path Group Refractive Index",
        "每段程间未被模板覆盖的附加延时 (s)": "Additional Delay Not Covered by Each Template Segment (s)",
        "活动场内存上限 (MiB)": "Active-Field Memory Limit (MiB)",
        "允许相对群折射率跨度": "Allowed Relative Group-Index Span",
        "允许相对 cosθ 跨度": "Allowed Relative cosθ Span",
        "有限程数按几何注入/出射孔解释；程序不会把一个无开关的被动闭腔假装成恰好 N 程停止。": (
            "A finite pass count is interpreted through geometric injection and extraction; "
            "an unswitched passive closed cavity is never assumed to stop after exactly N passes."
        ),
        "理想开关（Pockels/偏振门等效）": "Ideal Switch (Equivalent Pockels/Polarization Gate)",
        "几何分离出射（最后一程进入取出孔）": "Geometric Extraction (Final Pass Reaches an Output Port)",
        "有限程数如何结束": "How the Finite Pass Sequence Ends",
        "两面被动镜本身不会在指定的第 N 程自动停止；这里必须声明实验使用主动开关，还是依靠光斑/角度分离进入出射孔。": (
            "Two passive mirrors do not stop automatically after pass N. State whether the "
            "experiment uses an active switch or spot/angle separation into an output port."
        ),
        "左侧镜": "Left Mirror",
        "右侧镜": "Right Mirror",
        "注入中心 x (m)": "Injection Center x (m)",
        "注入中心 y (m)": "Injection Center y (m)",
        "注入 θx (°)": "Injection θx (°)",
        "注入 θy (°)": "Injection θy (°)",
        "使用对象面→F1→F2→镜面→F2→F1→对象面的往返中继算子。当前要求往返 ABCD 满足单位放大且 B=0；不满足时配置会明确报出矩阵。": (
            "Uses the object plane → F1 → F2 → mirror → F2 → F1 → object plane "
            "round-trip relay. The current operator requires unit magnification and B = 0; "
            "otherwise validation reports the configured ABCD matrix."
        ),
        "##### 每程取样位置/角度程序": "##### Per-Pass Sampling Position/Angle Program",
        "这里不是反推论文没写出的镜面。hits 是每次晶体入口的方向/中心/角度；transitions 是上一程出晶体到下一程入晶体的光程、折返镜或额外延时。": (
            "This does not infer mirrors omitted by a paper. hits gives each crystal-entry "
            "direction, center, and angle; transitions gives the path length, return mirror, "
            "or additional delay from one crystal exit to the next entrance."
        ),
        "晶体命中 hits JSON": "Crystal-Hit hits JSON",
        "程间 transitions JSON": "Inter-Pass transitions JSON",
        "多程晶体命中": "Multipass Crystal Hits",
        "多程程间路由": "Multipass Inter-Pass Routes",
        "镜面/中继自身损耗之外的公共附加损耗。": "Common additional loss beyond the mirror or relay loss itself.",
        "只填模板几何光程之外的电子/光纤等已知延时。": "Enter only known electronic, fiber, or other delay outside the template geometry.",
        "绝对时间最大分箱数": "Maximum Absolute-Time Bins",
        "不同程共用一个 z-CFL 时钟的显式误差界；超过即拒绝计算，不会静默把大角度差当成相同速度。": (
            "Explicit error bound for sharing one z-CFL clock across passes. Exceeding it "
            "rejects the run instead of silently treating large angular differences as equal speed."
        ),
        "两镜顶点间距 L (m)": "Mirror-Vertex Separation L (m)",
        "晶体中心距左镜 (m)": "Crystal-Center Distance from Left Mirror (m)",
        "左侧镜参数": "Left-Mirror Parameters",
        "右侧镜参数": "Right-Mirror Parameters",
        "对象面→F1 (m)": "Object Plane → F1 (m)",
        "F1 焦距 (m)": "F1 Focal Length (m)",
        "F2 焦距 (m)": "F2 Focal Length (m)",
        "F2→镜面 (m)": "F2 → Mirror (m)",
        "每次完整往返中继功率透过率": "Power Transmission per Complete Relay Round Trip",
        "中继光轴中心 x (m)": "Relay-Axis Center x (m)",
        "中继光轴中心 y (m)": "Relay-Axis Center y (m)",
        "每程中心增量 Δx (m)": "Per-Pass Center Increment Δx (m)",
        "每程中心增量 Δy (m)": "Per-Pass Center Increment Δy (m)",
        "每程 Δθx (°)": "Per-Pass Δθx (°)",
        "每程 Δθy (°)": "Per-Pass Δθy (°)",
        "每项可含 name、direction、center_x_m、center_y_m、incidence_angle_x_deg、incidence_angle_y_deg 和界面参数。": (
            "Each item may contain name, direction, center_x_m, center_y_m, "
            "incidence_angle_x_deg, incidence_angle_y_deg, and interface parameters."
        ),
        "每项 route_model 可为 delay_only、free_space、round_trip_mirror 或 optical_train；再给 geometric_length_m、折射率、透过率、extra_delay_s，以及相应镜面参数或 route 元件数组。delay_only 会把几何长度计入因果时钟，但不会让采样场衍射或平移。": (
            "Each route_model may be delay_only, free_space, round_trip_mirror, or optical_train. "
            "Supply geometric_length_m, refractive indices, transmission, extra_delay_s, and "
            "the corresponding mirror parameters or route element array. delay_only advances "
            "the causal clock by its geometric length but does not diffract or translate the sampled field."
        ),
        "模板": "Template",
        "晶体命中": "Crystal Hits",
        "自动检测重叠": "Automatically Detected Overlap",
        "最多同时活动程": "Maximum Simultaneously Active Passes",
        "重叠来自模板光程、脉冲包络与逐程二维命中；表中的对齐位移是模板主光线/论文边界条件与传播后复包络之间的注册量，不改变能量。": (
            "Overlap follows from the template path, pulse envelope, and per-pass 2D hits. "
            "The reported alignment shift registers the propagated complex envelope to the "
            "template chief ray or paper boundary condition and does not change energy."
        ),
        "每段未建模附加延时 (s)": "Unmodeled Additional Delay per Segment (s)",
        "有": "Yes",
        "无": "No",
        "中心 x (mm)": "Center x (mm)",
        "中心 y (mm)": "Center y (mm)",
        "入口 wx (mm)": "Entrance wx (mm)",
        "入口 wy (mm)": "Entrance wy (mm)",
        "程间": "Transition",
        "几何路长 (m)": "Geometric Path Length (m)",
        "精确延时 (ns)": "Exact Delay (ns)",
        "量化误差 (ps)": "Quantization Error (ps)",
        "路由前透过率": "Pre-Route Transmission",
        "命中合同": "Hit Contract",
        "开启时，所有程的飞行时间与能级寿命/暗跃迁共用同一时间账本；只有已证明完整多程历时远短于全部动力学时间尺度时才应关闭。": (
            "When enabled, all pass flight times share one clock with level lifetimes and "
            "dark transitions. Disable it only after showing that the complete multipass "
            "duration is much shorter than every kinetic timescale."
        ),
    }
)


# Strong-field, Commercial App, and carrier-state copy added with the corresponding
# physical contracts. Exact entries prevent the fragment translator from
# mixing Chinese domain terms into otherwise English help text.
_EN_OVERRIDES.update(
    {
        "载流子场残差": "Carrier-Field Residual",
        "已有后台单点或参数遍历正在运行；该设置将在下一次运行时生效。": (
            "A background single-point solve or parameter sweep is already "
            "running; this setting will take effect on the next run."
        ),
        "固定等效脉宽只计算 Kerr 透镜；一维 effective-area 模型另沿时间轴计算相干 SPM/色散，可输出光谱展宽和压缩波形；Commercial App effective-area 多光子损耗、载流子、FCA 与载流子折射也使用这条时间包络。": (
            "Fixed effective duration computes only the Kerr lens. The 1-D "
            "effective-area model additionally propagates coherent SPM and "
            "dispersion along time and can report spectral broadening and a "
            "compressed waveform. Commercial App effective-area multiphoton loss, "
            "carriers, FCA, and carrier refraction use the same temporal envelope."
        ),
        "最多迭代只是上限，达到容差会提前停止。Cr:ZnS 热光算例通常需要约 15–20 次；最终以结果中的 converged、布居残差、输出残差和已启用的载流子影响残差为准。": (
            "The maximum iteration count is only a safety cap; the solver stops "
            "early after meeting tolerance. Cr:ZnS thermo-optical cases often "
            "need about 15–20 iterations. Use the reported converged flag, "
            "population residual, output residual, and any enabled carrier-effect "
            "residual as the final criteria."
        ),
        "、所选载流子影响残差": ", and the selected carrier-effect residual",
        "按载流子光学效应（FCA 光深、相位和吸收能量；推荐）": (
            "Use Carrier Optical Effects (FCA Optical Depth, Phase, and "
            "Absorbed Energy; Recommended)"
        ),
        "按完整载流子密度场最大相对变化（更严格）": (
            "Use Maximum Relative Change of the Full Carrier-Density Field "
            "(Stricter)"
        ),
        "跨脉冲载流子收敛判据": "Inter-Pulse Carrier Convergence Metric",
        "收敛判据：布居、输出能量": (
            "Convergence criteria: population and output energy"
        ),
        "的残差都 ≤ ": " residuals are all ≤ ",
        "这是周期固定点的第三个停止条件。光学效应判据检查载流子对当前 Seed 真正能造成的 FCA、相位和吸收能量变化，密度场判据则要求所有网格点的载流子密度本身都收敛。": (
            "This is the third stopping condition for the periodic fixed point. "
            "The optical-effect metric checks changes in the FCA, phase, and "
            "absorbed energy that carriers can actually impose on the current "
            "Seed. The field metric instead requires the carrier density itself "
            "to converge at every grid point."
        ),
        "每个 z 层仍计算三维局域增益/布居交换；横向传播保留 Gaussian q，时间上另做一条一维 FFT。多光子损耗总量使用二维 Gaussian 的 K 阶强度矩，载流子/FCA 是能量等效标量闭合；它不解析径向载流子输运或不同半径处的光谱展宽，定量结论需由 coherent_cpa 复核。": (
            "Each z layer still computes 3-D local gain and population exchange. "
            "Transverse propagation retains Gaussian q while a separate 1-D FFT "
            "is propagated in time. Total multiphoton loss uses the Kth intensity "
            "moment of the 2-D Gaussian; carriers and FCA use an energy-equivalent "
            "scalar closure. Radial carrier transport and radius-dependent spectral "
            "broadening are not resolved, so quantitative conclusions require a "
            "coherent_cpa cross-check."
        ),
        "这里描述 β 的物理含义，不改变沿真实光束路径的 dI/ds=-βI^K 数值方程。phenomenological_sensitivity 只能用于敏感性扫描，不能作为 TPA/MPA 机制证据。": (
            "This describes the physical interpretation of beta without changing "
            "the numerical equation dI/ds = -beta I^K along the true beam path. "
            "phenomenological_sensitivity is for sensitivity scans only and is not "
            "evidence for a TPA or MPA mechanism."
        ),
        "沿光束真实路径满足 dI/ds=-β I^K；I 是垂直于光束的强度。单位随 K 改变，请直接使用论文给出的 SI 值，程序不会猜 Cr:ZnS 系数。": (
            "Along the true beam path, dI/ds = -beta I^K, where I is the "
            "intensity normal to the beam. Units vary with K. Enter the SI value "
            "reported by the source; the program does not guess a Cr:ZnS coefficient."
        ),
        "这是 Commercial App L3.5 的一维 effective-area 强场结果：频谱/时间包络是相干的，多光子损耗总量使用 Gaussian K 阶矩，载流子/FCA 使用能量等效标量状态；横向场仍由 Gaussian q 表示。它不等价于解析径向载流子与时空耦合的完整 x-y-t coherent CPA。": (
            "This is a 1-D effective-area strong-field result at Commercial App L3.5. "
            "The spectral and temporal envelope is coherent, total multiphoton "
            "loss uses the Gaussian Kth moment, and carriers and FCA use an "
            "energy-equivalent scalar state; the transverse field remains Gaussian q. "
            "It is not a full x-y-t coherent CPA solution with radial carrier and "
            "spatiotemporal coupling."
        ),
        "自由载流子由强场电离/吸收算符产生；强场损耗关闭时该子状态必须停用，重新开启父模块后会恢复原设置。": (
            "Free carriers are generated by the strong-field ionization/absorption "
            "operator. This child state must be disabled when strong-field loss is "
            "off; its previous setting is restored when the parent module is re-enabled."
        ),
        "Commercial App 只接受专用的 effective-area 强场损耗闭合；完整 x-y-t 强场算符属于 coherent_cpa。": (
            "Commercial App accepts only its dedicated effective-area strong-field loss "
            "closure; the full x-y-t strong-field operator belongs to coherent_cpa."
        ),
        "Commercial App effective-area 强场损耗需要一维相干时间包络。": (
            "Commercial App effective-area strong-field loss requires a 1-D coherent temporal envelope."
        ),
        "已选择 Commercial App effective-area 强场损耗，因此启用材料多光子模型。": (
            "The material multiphoton model is enabled because Commercial App effective-area "
            "strong-field loss is selected."
        ),
        "强场损耗只在 coherent_cpa 完整时间域或 Commercial App effective-area 专用闭合中定义。": (
            "Strong-field loss is defined only for the full coherent_cpa time-domain "
            "field or the dedicated Commercial App effective-area closure."
        ),
        "Commercial App 非相干 HG 通道需要 fixed_duration 的逐模态共享布居合同；当前单 Gaussian effective-area 强场矩不能同时保持各 HG 模态的局部 K 阶强度矩，因此保留显式 HG 并关闭该强场损耗近似。": (
            "Commercial App incoherent HG channels require the fixed_duration per-mode, "
            "shared-population contract. The current single-Gaussian effective-area "
            "strong-field moment cannot also retain each HG mode's local Kth intensity "
            "moment, so explicit HG is preserved and this strong-field approximation is disabled."
        ),
        "Commercial App HG 关闭 effective-area 强场父算符后，自由载流子子状态也必须关闭。": (
            "The free-carrier child state must also be disabled when Commercial App HG disables "
            "the parent effective-area strong-field operator."
        ),
        "路径坐标 %{customdata[0]:.5g} mm<br>晶体轴向 z %{customdata[1]:.5g} mm<br>局部 x %{y:.5g} mm<br>局部 y %{z:.5g} mm<br>%{value:.5g} ": (
            "Path Coordinate %{customdata[0]:.5g} mm<br>Crystal-Axis z "
            "%{customdata[1]:.5g} mm<br>Local x %{y:.5g} mm<br>Local y "
            "%{z:.5g} mm<br>%{value:.5g} "
        ),
        "把整脉冲光谱能量与横向 Gaussian q 参数耦合；每层增益照常计算，热场、Kerr 和增益导引只保留等效二次矩。可选把互不相干 HG 模态作为独立 q 通道并共享局部布居，也可用一维 effective-area 包络计算 SPM、Gaussian K 阶矩强场损耗与载流子闭合。": (
            "Couple whole-pulse spectral energy to transverse Gaussian-q parameters. "
            "Each layer still computes gain, while thermal field, Kerr, and gain "
            "guiding retain only equivalent quadratic moments. Optional incoherent HG "
            "modes propagate as independent q channels with shared local population; "
            "a 1-D effective-area envelope can instead compute SPM, Gaussian Kth-moment "
            "strong-field loss, and the carrier closure."
        ),
        "把三维反转与 Gaussian 或 Commercial App 显式 HG 通道做重叠积分，每个 z 层只产生模态标量增益，不产生任意高阶增益导引。": (
            "Integrate the overlap between 3-D inversion and a Gaussian or explicit "
            "Commercial App HG channel. Each z layer produces only modal scalar gain and cannot "
            "generate arbitrary higher-order gain guiding."
        ),
        "在 Commercial App 一维相干包络上，用当前 Gaussian 有效面积及其精确 K 阶弱损耗矩闭合多光子吸收，并可推进标量载流子、FCA 与载流子折射率；不解析径向载流子梯度。": (
            "On the Commercial App 1-D coherent envelope, close multiphoton absorption with the "
            "current Gaussian effective area and its exact weak-loss Kth moment. Scalar "
            "carriers, FCA, and carrier refraction may also be propagated; radial carrier "
            "gradients are not resolved."
        ),
        "冲突：Gaussian 模态重叠闭合只定义于单色 fluence 或 Commercial App 模态通道。": (
            "Conflict: Gaussian modal-overlap closure is defined only for monochromatic "
            "fluence or Commercial App modal channels."
        ),
        "冲突：完整时间域强场损耗属于 coherent_cpa；effective-area 强场矩闭合属于 commercial_app_fast。": (
            "Conflict: full time-domain strong-field loss belongs to coherent_cpa; "
            "the effective-area strong-field moment closure belongs to commercial_app_fast."
        ),
        "增益与布居仍使用一个整脉冲 FN 包。": (
            "Gain and population still use one whole-pulse Frantz–Nodvik packet."
        ),
        "沿一维相干时间/频率包络用当前 Gaussian 模有效面积计算 SPM、材料色散": (
            "Use the current Gaussian-mode effective area along a 1-D coherent "
            "time/frequency envelope to compute SPM and material dispersion"
        ),
        "每个晶体层的局部光谱增益和布居交换仍由整脉冲 FN 包闭合，横向热相位、Kerr 透镜与增益导引投影回 Gaussian 二阶矩并解析推进 q。": (
            "Local spectral gain and population exchange in every crystal layer remain "
            "closed by a whole-pulse Frantz–Nodvik packet. Transverse thermal phase, "
            "Kerr lensing, and gain guiding are projected back to Gaussian second "
            "moments and q is advanced analytically."
        ),
        "、Gaussian K 阶矩强场损耗及标量载流子响应；": (
            ", Gaussian Kth-moment strong-field loss, and scalar carrier response;"
        ),
        "用一维相干时间/频率包络与当前 Gaussian 有效面积推进材料色散、SPM": (
            "Advance material dispersion and SPM with a 1-D coherent time/frequency "
            "envelope and the current Gaussian effective area"
        ),
    }
)


# High-visibility form/result strings whose legacy bulk catalog entries were
# linguistically plausible enough to evade token-level cleanup but changed the
# physical meaning.  Keep exact overrides here so an English UI never turns
# population into "residence", amplification into zoom, or Pump into "Pumpup".
_EN_OVERRIDES.update(
    {
        " 个。下方“交互式三维结果工作台”默认显示可拖动旋转的三维光路；第二个页签提供 x/y/z 可拖动切片。": (
            " entries. The Interactive 3D Results Workbench below opens with a "
            "rotatable 3D optical path; the second tab provides draggable x/y/z slices."
        ),
        "**。光学 Pump 的净吸收系数由下能级吸收减去上能级受激发射得到；开启耗尽时，这个净吸收直接扣减下游 Pump。激发效率 ηq 只乘入产生有效能级跃迁的吸收速率，而不伪造 Pump已经被吸收的光学能量；当前 ηq=": (
            "**. Optical Pump net absorption is lower-level absorption minus "
            "upper-level stimulated emission. With depletion enabled, that net "
            "absorption directly removes downstream Pump power. Excitation "
            "efficiency ηq scales only the rate of useful level transitions; it "
            "does not fabricate or remove already absorbed optical energy. Current ηq="
        ),
        "Pump / Seed 输入面　→　实际光学电路　→　三维 Pump 吸收与能级布居　→　Seed 局部增益和多程提取　→　能量去向与热源　→　三维热/应力/折射率反馈　→　网格守恒映射　→　出光面与定量有效性合同": (
            "Pump / Seed input planes → physical optical paths → 3D Pump absorption "
            "and level populations → local Seed gain and multipass extraction → "
            "energy destinations and heat sources → 3D thermal/stress/index feedback "
            "→ conservative grid coupling → output plane and quantitative validity contract"
        ),
        "Seed 放大": "Seed Amplification",
        "Seed-only 保留 Pump、三维布居、周期方法、热模型和热光固定点；全链路极速还会改变这些运行合同，因此不能直接比较绝对输出功率。": (
            "Seed-only retains the Pump, 3D population, periodic method, thermal "
            "model, and thermo-optic fixed point. Full-chain fast mode also changes "
            "those runtime contracts, so its absolute output power is not directly comparable."
        ),
        "full_3d_pc 已锁定：Seed FFT、Pump FFT、瞬态最终脉冲布居、泵浦耗尽、增益饱和和光学面记录。退出该 JOSA 层级后可重新编辑这些项。": (
            "full_3d_pc locks Seed FFT, Pump FFT, transient final-pulse populations, "
            "Pump depletion, gain saturation, and optical-plane recording. Leave "
            "this JOSA tier to edit those settings again."
        ),
        "两束光均以半透明、封口并带同色细描边的椭圆 D4σ 光束体显示；滚轮缩放，拖动旋转。": (
            "Both beams are shown as translucent, capped elliptical D4σ volumes "
            "with matching outlines; use the wheel to zoom and drag to rotate."
        ),
        "光谱超高斯阶数": "Spectral Super-Gaussian Order",
        "冲突：Seed 强场产热 fixed_point 需要 thermal_optical_fixed_point。": (
            "Conflict: Seed strong-field heat coupling in fixed_point mode requires "
            "thermal_optical_fixed_point."
        ),
        "周期固定点必须保存并更新真实能级布居，不能每个周期重新建立外加 g₀。": (
            "A periodic fixed point must retain and update the physical level "
            "populations; it cannot recreate an imposed g₀ at every cycle."
        ),
        "周期平均初值求解失败，已安全退回设定初始布居；精确周期映射仍继续运行。原因：": (
            "The cycle-average initializer failed, so the configured initial "
            "population was restored safely; the exact periodic map will still run. Reason:"
        ),
        "在时延坐标中联立光子通量输运与能级速率方程；front、peak、tail 依次看见已经被前一部分改变的布居。所选 JOSA 层级决定是否同时保留完整三维横向传播。": (
            "Couple photon-flux transport to the level-rate equations in a retarded-time "
            "coordinate. The front, peak, and tail sequentially see populations changed "
            "by earlier parts of the pulse. The selected JOSA tier determines whether "
            "full 3D transverse propagation is also retained."
        ),
        "当一个时间窗引起的局部布居跳变过大时自动拆分该窗，降低排序误差并保留能量守恒。": (
            "Split a time window automatically when it causes an excessive local "
            "population jump, reducing ordering error while preserving energy conservation."
        ),
        "把输入离子密度解释为全晶体体积平均值，并缩放有源区域使离散体积积分保持该平均值。": (
            "Interpret the input ion density as a whole-crystal volume average and "
            "scale the active region so the discrete volume integral preserves that average."
        ),
        "放大后": "After Amplification",
        "放大后 FWHM": "Amplified FWHM",
        "每一步都重新传播平均 Pump/Seed 光子流并求局部代数布居；停止容差为 ": (
            "At every iteration, repropagate the average Pump/Seed photon fluxes and "
            "solve the local algebraic populations; the stopping tolerance is "
        ),
        "求解局部离子布居。": "Solve the local ion populations.",
        "沿一维相干时间/频率包络用当前 Gaussian 模有效面积计算 SPM 和材料色散；增益与布居仍使用一个整脉冲 FN 包。": (
            "Compute SPM and material dispersion along a 1D coherent time/frequency "
            "envelope using the current Gaussian-mode effective area; gain and "
            "population still use one whole-pulse Frantz–Nodvik packet."
        ),
        "泵浦 ABCD/FFT 与界面": "Pump ABCD/FFT and Interfaces",
        "泵浦光谱 CSV：λ(nm), 相对功率谱": (
            "Pump Spectrum CSV: λ (nm), Relative Power Spectrum"
        ),
        "泵浦光谱 FWHM (Hz)": "Pump Spectrum FWHM (Hz)",
        "泵浦光谱点：": "Pump Spectral Points:",
        "泵浦激发量子效率始终只由能级窗口控制布居速率。lifetime_quantum_defect 用总寿命与辐射寿命之差，按实际上能级布居计算非辐射热，不把荧光效率重复乘到激发率；quantum_defect 还把未形成有效激发的吸收能量计入热；scaled_quantum_defect 不乘该量子效率，而使用独立倍率。spectral 两项会按每个 Pump 波长的功率权重和局部净吸收截面计算空间变化的热比例。": (
            "Pump excitation quantum efficiency only scales population-transfer rates "
            "in the level model. lifetime_quantum_defect derives nonradiative heat from "
            "the difference between total and radiative lifetimes using the actual "
            "upper-level population; it does not multiply fluorescence efficiency into "
            "the excitation rate again. quantum_defect also deposits absorbed energy "
            "that did not create a useful excitation. scaled_quantum_defect uses an "
            "independent multiplier instead of that quantum efficiency. The two spectral "
            "modes compute a spatially varying heat fraction from per-wavelength Pump "
            "power weights and local net-absorption cross sections."
        ),
        "泵浦频率密度 FWHM 输入": "Pump Frequency-Density FWHM Input",
        "由 Pump 吸收截面、受激跃迁、自发寿命和能级网络求局部布居，再由 σeNu−σaNl 得到增益。": (
            "Solve local populations from Pump absorption cross sections, stimulated "
            "transitions, spontaneous lifetimes, and the level network; then compute "
            "gain as σeNu − σaNl."
        ),
        "相干模块失败时": "On Coherent-Module Failure",
        "JOSA 子模块失败时": "On JOSA-Module Failure",
        "瞬态增益子模块失败时": "On Transient-Gain Failure",
        "当前 emitter_na + stochastic_raytrace 直接在 z=0 发光面采样位置和 NA；focus_z_m / focus_x_m / focus_y_m 不参与光线生成。导入值会保留，但不显示成一个实际生效的自由度。": (
            "With emitter_na + stochastic_raytrace, rays are sampled directly "
            "from positions and NA at the z=0 emitting surface. focus_z_m, "
            "focus_x_m, and focus_y_m do not generate the rays. Imported values "
            "are preserved but are not shown as an active degree of freedom."
        ),
        "prescribed_caustic 直接规定晶体内焦散，因此从晶体入口计；SG-ABCD、普通 ABCD 和 FFT 从 PUMP LASER 源面传播，因此从源面计。": (
            "prescribed_caustic specifies the in-crystal caustic relative to "
            "the crystal entrance. SG-ABCD, ordinary ABCD, and FFT propagate "
            "from the PUMP LASER source plane, so their coordinate starts there."
        ),
        "逐脉冲周期映射必须提交每次 Pump/Seed 事件后的瞬态布居。": (
            "A pulse-resolved periodic map must commit the transient population after "
            "every Pump/Seed event."
        ),
        "；在总能量固定时它决定时间包络、峰值功率和 Kerr/强场效应。": (
            "; at fixed total energy it determines the temporal envelope, peak power, "
            "and Kerr/strong-field effects."
        ),
        "Seed 双网格 · ": "Seed Dual Grid · ",
        "Pump 独立网格 · custom": "Pump Custom Grid",
        "Pump/布居每周期时间采样数": "Pump/Population Samples per Cycle",
        "瞬态模型使用 ρ·cp·∂T/∂t 和隐式 Euler；允许全绝热边界，此时吸收热会累积为温升。这里的 Δt_thermal=总热时间/热步数，与 Pump/布居时间步 Δt_population 独立；通常热时间步可以更粗。": (
            "The transient model solves ρ·cp·∂T/∂t with backward Euler and permits "
            "fully adiabatic boundaries, for which deposited heat accumulates as a "
            "temperature rise. Here Δt_thermal is the total thermal duration divided "
            "by the number of thermal steps and is independent of the Pump/population "
            "step Δt_population; thermal steps can usually be coarser."
        ),
        "快速平均稳态会把 Eseed × 重复频率转换成 Seed 平均光子流，并与 Pump、自发衰减、受激发射/再吸收一起求三维代数稳态；迭代次数不是经过的脉冲数。": (
            "The fast cycle-averaged solver converts Seed pulse energy times repetition "
            "rate into an average photon flux, then solves the 3D algebraic steady state "
            "with Pump excitation, spontaneous decay, stimulated emission, and "
            "reabsorption. Its iteration count is not a number of elapsed pulses."
        ),
        "适合 25 MHz 这类高重复频率且单脉冲只改变少量布居的放大器。最终仍会用当前 FN/JOSA/CPA 脉冲求解器打一发校验；若单脉冲布居跳变 >1%，结果会提示改用严格模式。": (
            "Use this for high-repetition-rate amplifiers such as 25 MHz when one pulse "
            "changes only a small fraction of the population. The selected FN/JOSA/CPA "
            "pulse solver still performs a one-pulse check on the converged state; a "
            "population jump above 1% triggers a recommendation to use the strict "
            "pulse-resolved mode."
        ),
        "快速模式直接自洽求平均光子流稳态；严格模式重复计算“一个完整 Pump 周期 → 一个 Seed 脉冲”直到固定点。": (
            "Fast mode solves the cycle-averaged photon-flux steady state self-consistently. "
            "Strict mode repeats one complete Pump cycle followed by one Seed pulse until "
            "the resulting state map reaches a fixed point."
        ),
        "steady_state — Pump + Seed 平均稳态（由运行页接管）": (
            "steady_state — Pump + Seed Cycle-Averaged Steady State "
            "(Controlled by Run Mode)"
        ),
        "周期平均稳态已在运行页接管布居。": (
            "Population is controlled by the cycle-averaged steady-state settings above."
        ),
    }
)


_EN_OVERRIDES.update(
    {
        "同时计算点数": "Concurrent Points",
        "1 个点保留当前完整实时阶段图；2–3 个点使用独立进程，实时区只显示每个点的阶段与百分比。": (
            "One point retains the full live stage plots. Two or three points "
            "use isolated processes, and the live area shows only each point's "
            "stage and percentage."
        ),
        "Pump 功率遍历需要把上一点的收敛布居、温度和 Δn 作为下一点 warm start，因此自动保持串行。": (
            "A Pump-power sweep uses the previous point's converged population, "
            "temperature, and Δn as the next warm start, so it remains serial "
            "automatically."
        ),
        "串行模式：显示当前点的完整实时诊断图和阶段明细。": (
            "Serial mode shows the current point's complete live diagnostics "
            "and stage details."
        ),
        "并行模式：最多同时运行 ": "Parallel mode runs up to ",
        " 个独立点；每点保持当前配置的内部并行，不额外限制线程。实时区只保留状态和百分比，完整三维结果仍逐点落盘。": (
            " independent points concurrently. Each point keeps the current "
            "internal parallelism settings with no additional thread cap. The "
            "live area retains only status and percentage, while complete 3D "
            "results are still persisted point by point."
        ),
        "二维网格包含 Pump 功率时没有唯一的“上一点”路径，因此各点独立求解，不会使用一维 Pump 扫描的连续 warm start。": (
            "A 2D grid containing Pump power has no unique previous-point path, "
            "so its points are solved independently without the continuous warm "
            "start used by a 1D Pump sweep."
        ),
        "预计执行 ": "Expected to run ",
        " 次完整模拟，分为约 ": " full simulations in approximately ",
        " 组；每个失败点会被单独记录，不会使整组遍历丢失。": (
            " batches; each failed point is recorded independently without "
            "discarding the rest of the sweep."
        ),
        "并行参数遍历进度": "Parallel Sweep Progress",
        "正在运行的点": "Active Points",
        "并行模式只传输每个点的阶段、状态和百分比，不生成实时场图；完整场与全部诊断仍在后台按点保存。": (
            "Parallel mode transfers only each point's stage, status, and "
            "percentage; it does not generate live field plots. Complete fields "
            "and diagnostics are still saved per point in the background."
        ),
        "并行活动点": "Active Parallel Points",
        "并行槽 ": "Parallel Slot ",
        "空闲": "Idle",
        "等待下一个点": "Waiting for the next point",
        "本次遍历已停止": "This sweep has stopped",
        "当前状态：": "Current status: ",
        "冻结所有正在运行的点及整个遍历工作进程；已完成点仍可立即下载。": (
            "Freeze every active point and the whole sweep worker; completed "
            "points remain immediately downloadable."
        ),
    }
)


_TARGET_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\blight(?: |-)?(?:roads?|ways?|paths?|wide)\b", re.IGNORECASE), "optical path"),
    (re.compile(r"\bhot fields?\b|\bheat fields?\b", re.IGNORECASE), "thermal field"),
    (re.compile(r"\brefraction rates?\b", re.IGNORECASE), "refractive index"),
    (re.compile(r"\blife expectancy\b", re.IGNORECASE), "lifetime"),
    (re.compile(r"\blean marks?\b", re.IGNORECASE), "waist"),
    (re.compile(r"\btowed rotation\b", re.IGNORECASE), "drag to rotate"),
    (re.compile(r"\bresult(?:s)? analysis desk\b", re.IGNORECASE), "Result Analysis Workbench"),
    (re.compile(r"\binteractive 3-?D result desk\b", re.IGNORECASE), "Interactive 3D Results Workbench"),
    (re.compile(r"\b(?:fTT|FTT)\b"), "FFT"),
    (re.compile(r"\b3-D\b"), "3D"),
)


_SOURCE_AWARE_FIXES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("布居", re.compile(r"\b(?:burrows?|bulges?|budock|burden)\b", re.IGNORECASE), "population"),
    ("相干脉冲", re.compile(r"\b(?:arterial|trunk) pulse\b", re.IGNORECASE), "coherent pulse"),
    ("残差", re.compile(r"\bdisabled?\b", re.IGNORECASE), "residual"),
    ("腰斑", re.compile(r"\btrails?\b", re.IGNORECASE), "waist"),
    ("能级", re.compile(r"\bcapable\b", re.IGNORECASE), "energy level"),
    ("截面", re.compile(r"\bcross-the-board\b", re.IGNORECASE), "cross section"),
    ("热光", re.compile(r"\b(?:hot light|mother optical|thermal optical)\b", re.IGNORECASE), "thermo-optic"),
    ("元件", re.compile(r"\bwidgets?\b", re.IGNORECASE), "element"),
    ("光程", re.compile(r"\b(?:species|radium)\b", re.IGNORECASE), "path length"),
    ("传播", re.compile(r"\b(?:dissemination|disseminates?|spreads?)\b", re.IGNORECASE), "propagation"),
    ("强度", re.compile(r"\bstrength\b", re.IGNORECASE), "intensity"),
    ("光谱", re.compile(r"\bspectroscopy\b", re.IGNORECASE), "spectrum"),
    ("稳态", re.compile(r"\bstability\b", re.IGNORECASE), "steady state"),
    ("收敛", re.compile(r"\b(?:condensate|condensation|contraction)\b", re.IGNORECASE), "convergence"),
    ("验证", re.compile(r"\b(?:authentication|certification)\b", re.IGNORECASE), "validation"),
    ("掺杂", re.compile(r"\b(?:mixed|adulterated)\b", re.IGNORECASE), "dopant"),
    ("高斯", re.compile(r"\b(?:goss|goz|gorges|gorge|gossian)\b", re.IGNORECASE), "Gaussian"),
    ("遍历", re.compile(r"\b(?:walking|walk through|history)\b", re.IGNORECASE), "sweep"),
    ("反转", re.compile(r"\breverse\b", re.IGNORECASE), "inversion"),
    ("Seed", re.compile(r"\bFeed\b"), "Seed"),
    (
        "二阶矩",
        re.compile(
            r"\b(?:second[- ](?:step|stage|order) rectangular|quadratic moment)\b",
            re.IGNORECASE,
        ),
        "second moment",
    ),
)


_EN_OVERRIDES.update(
    {
        "多程晶体命中次数": "Multipass Crystal Hits",
        "多程：模板、命中与程间光路": "Multipass: Template, Hits, and Inter-Pass Routes",
        "已迁移的快速顺序多程 — 非时域后端降阶执行": "Migrated Sequential Multipass — Reduced Execution for Non-Time-Domain Backends",
        "相邻程反向通过晶体": "Reverse Crystal Direction on Adjacent Passes",
        "程间光学元件 route JSON": "Inter-Pass Optical-Element Route JSON",
        "这个配置由旧快速逐程自动迁移而来。它已使用统一的多程命中/程间路由，但原始文件没有足够镜路信息，程序不会假造因果重叠；可继续用于 Commercial App、fluence 或 CPA 的完整逐程计算。新建物理多程请切换到上面的镜路模板。": (
            "This configuration was migrated from the retired quick-pass model. "
            "It now uses the unified multipass hit/transition plan, but the source "
            "file did not contain enough mirror geometry to infer causal overlap. "
            "It remains available for complete sequential Commercial App, fluence, or CPA "
            "runs; choose a physical route template for a new multipass case."
        ),
        "只影响 Seed 的入射与出射光路；多次晶体命中和程间元件统一在下方“多程”中配置。": (
            "This controls only the Seed input and output routes. Configure repeated "
            "crystal hits and inter-pass elements in Multipass below."
        ),
        "多程命中与程间光路": "Multipass Hits and Inter-Pass Routes",
        "单程": "Single Pass",
        "① 模型 → 启用多程 / ② 光束 → 多程模板": "① Model → Enable Multipass / ② Beams → Multipass Template",
        "因果多程需要在一条绝对时间轴上传播各个脉冲包，因此使用 spectral_time。": "Causal multipass propagation carries every pulse packet on one absolute time axis, so spectral_time is used.",
        "所有晶体命中与程间光路已经统一由 simulation.multipass 编译。": "All crystal hits and inter-pass routes are compiled by the unified simulation.multipass contract.",
        "迁移顺序计划": "Migrated Sequential Plan",
        "因果": "Causal",
        "迁移多程程间光路": "Migrated Multipass Inter-Pass Route",
        "当前是单程；请先启用多程，再遍历晶体命中次数。": "The current configuration is single-pass. Enable multipass before sweeping the crystal-hit count.",
        "只用于自动迁移的顺序多程计划：开启后相邻命中交替沿 ±z 访问晶体；物理模板的方向由模板本身决定。": "Sequential multipass plan used only for automatic migration: when enabled, adjacent hits alternate along ±z; physical templates define their own directions.",
        "单程默认": "Single-Pass Default",
        "统一多程模板 ": "Unified Multipass Template ",
        " 次晶体命中（": " Crystal Hits (",
        "）；模板转移段拥有程间光程、损耗、方向和边界条件": "); template transitions own inter-pass optical length, loss, direction, and boundary conditions",
        "CW 没有有限脉冲包络，统一多程计划保留命中与光路并改用完整逐程执行。": "CW has no finite pulse envelope, so the unified multipass plan retains its hits and routes but uses complete sequential-pass execution.",
        "统一多程模板的晶体命中数是唯一程次数来源。": "The unified multipass template crystal-hit count is the sole source of the pass count.",
        "内部逐程后端从统一多程计划派生命中次数。": "Complete-pass backends derive their hit count from the unified multipass plan.",
        "多程定义已统一到 simulation.multipass。": "Multipass configuration is unified under simulation.multipass.",
        "程间损耗由统一多程 transition 记录，关闭旧全局重复扣损耗。": "Inter-pass loss is recorded by unified multipass transitions; the retired global duplicate loss is cleared.",
        "成像翻转由统一多程 transition 的光学元件记录。": "Imaging flips are recorded by optical elements in unified multipass transitions.",
        "程间元件已归入统一多程 transition。": "Inter-pass elements have been moved into unified multipass transitions.",
    }
)


# Physical-grid position and independently placed local refinement focus.
_EN_OVERRIDES.update(
    {
        "物理网格中心 x (m)": "Physical-Grid Center x (m)",
        "物理网格中心 y (m)": "Physical-Grid Center y (m)",
        "定义固定实验室坐标中的物理计算域中心；不会自动移动晶体、Pump 或 Seed。": (
            "Set the physical computational-domain center in fixed laboratory "
            "coordinates. This does not automatically move the crystal, Pump, or Seed."
        ),
        "独立设置 Gaussian 加密中心": "Set an Independent Gaussian Refinement Focus",
        "独立设置 x 加密中心": "Set an Independent x Refinement Focus",
        "独立设置 y 加密中心": "Set an Independent y Refinement Focus",
        "开启后在固定物理 x 边界内把非均匀单元聚集到指定实验室坐标；关闭时跟随物理网格中心 x，不移动任何物体或光束。": (
            "When enabled, cluster nonuniform cells at the specified laboratory x "
            "coordinate inside the fixed physical boundary. When disabled, follow the "
            "physical-grid center x without moving any object or beam."
        ),
        "开启后在固定物理 y 边界内把非均匀单元聚集到指定实验室坐标；关闭时跟随物理网格中心 y，不移动任何物体或光束。": (
            "When enabled, cluster nonuniform cells at the specified laboratory y "
            "coordinate inside the fixed physical boundary. When disabled, follow the "
            "physical-grid center y without moving any object or beam."
        ),
        "加密中心逐轴设置；未独立指定的轴跟随对应物理网格中心。只改变单元分布，不会移动物理边界、晶体、Pump、Seed 或独立光学窗口。": (
            "Configure the refinement focus independently by axis. An unspecified axis "
            "follows the corresponding physical-grid center. This changes only the cell "
            "distribution and does not move the physical boundary, crystal, Pump, Seed, "
            "or independent optical windows."
        ),
        "关闭时，加密中心逐轴跟随物理网格中心。开启后只改变非均匀物理单元的分布，不会移动物理边界、晶体、Pump、Seed 或独立光学窗口。": (
            "When disabled, each refinement-focus coordinate follows the physical-grid "
            "center. Enabling this changes only the nonuniform physical-cell distribution; "
            "it does not move the physical boundary, crystal, Pump, Seed, or independent "
            "optical windows."
        ),
        "物理网格加密中心 x (m)": "Physical-Grid Refinement Focus x (m)",
        "物理网格加密中心 y (m)": "Physical-Grid Refinement Focus y (m)",
        "必须严格位于物理 x 窗口内部；最小单元会集中在该坐标附近。": (
            "Must lie strictly inside the physical x window; the smallest cells cluster "
            "near this coordinate."
        ),
        "必须严格位于物理 y 窗口内部；最小单元会集中在该坐标附近。": (
            "Must lie strictly inside the physical y window; the smallest cells cluster "
            "near this coordinate."
        ),
        "x sinh 映射边缘/加密点局部尺度比": (
            "x sinh-Mapping Edge-to-Focus Local Scale Ratio"
        ),
        "y sinh 映射边缘/加密点局部尺度比": (
            "y sinh-Mapping Edge-to-Focus Local Scale Ratio"
        ),
        "这是连续 sinh 映射的边缘/加密点局部导数比；有限 Nx 下实际单元宽度比会略小。": (
            "This is the edge-to-focus local derivative ratio of the continuous sinh "
            "mapping; at finite Nx, the actual cell-width ratio is slightly smaller."
        ),
        "这是连续 sinh 映射的边缘/加密点局部导数比；有限 Ny 下实际单元宽度比会略小。": (
            "This is the edge-to-focus local derivative ratio of the continuous sinh "
            "mapping; at finite Ny, the actual cell-width ratio is slightly smaller."
        ),
        "局部加密用于布居/热/应力物理网格；独立均匀光学网格用于相应的 Pump 或 Seed/BPM 传播，并通过有限体积守恒映射双向耦合。": (
            "Local refinement applies to the population/thermal/stress physical grid. "
            "Independent uniform optical grids carry the corresponding Pump or Seed/BPM "
            "propagation and couple bidirectionally through conservative finite-volume "
            "mapping."
        ),
        "局部加密会把更多 x/y 单元放到指定加密中心。未启用独立光学网格时，FFT/BPM 要求均匀网格；Seed 必须为 off，Pump 可用 prescribed_caustic、super_gaussian_abcd 或 stochastic_raytrace。": (
            "Local refinement places more x/y cells near the selected focus. Without an "
            "independent optical grid, FFT/BPM requires a uniform grid: Seed must be off, "
            "while Pump may use prescribed_caustic, super_gaussian_abcd, or "
            "stochastic_raytrace."
        ),
        "full_3d_pc 固定使用 FFT；局部加密物理网格必须配一个独立均匀光学网格。": (
            "full_3d_pc is locked to FFT; a locally refined physical grid requires an "
            "independent uniform optical grid."
        ),
        "切换到局部加密相容传播模式": (
            "Switch to a Propagation Mode Compatible with Local Refinement"
        ),
        "当前使用单一网格；打开“独立均匀光学网格”可让局部加密晶体网格与 ABCD/FFT 同时使用。": (
            "A single grid is currently in use. Enable the independent uniform optical "
            "grid to combine a locally refined crystal grid with ABCD/FFT."
        ),
        "局部加密物理网格不能直接做 FFT；Seed 自动启用独立均匀光学网格。": (
            "A locally refined physical grid cannot run FFT directly; an independent "
            "uniform Seed optical grid was enabled automatically."
        ),
        "采样 Pump 在局部加密物理网格上需要复用均匀光学网格。": (
            "A sampled Pump on a locally refined physical grid must reuse a uniform "
            "optical grid."
        ),
        "采样 Pump 已选择共用光学网格；局部加密物理网格不能直接执行 FFT，因此自动启用均匀光学网格。": (
            "The sampled Pump is configured to share the optical grid. Because a locally "
            "refined physical grid cannot execute FFT directly, the uniform optical grid "
            "was enabled automatically."
        ),
        "把布居、热和应力物理网格向指定加密点聚集；未独立指定时逐轴跟随物理网格中心，FFT 仍必须使用独立均匀光学网格。": (
            "Cluster the population, thermal, and stress physical grid around a selected "
            "focus. Unspecified coordinates follow the physical-grid center; FFT still "
            "requires an independent uniform optical grid."
        ),
        "仅把布居、热和应力物理单元向指定加密点聚集；默认跟随物理网格中心，光学 FFT 仍需独立均匀网格并守恒映射。": (
            "Cluster only the population, thermal, and stress physical cells around a "
            "selected focus, which follows the physical-grid center by default. Optical "
            "FFT still requires an independent uniform grid with conservative mapping."
        ),
        "跟随物理网格中心": "Follow the Physical-Grid Center",
        "独立指定（未填写轴仍跟随物理网格中心）": (
            "Specified Independently (Unspecified Axes Still Follow the Physical-Grid Center)"
        ),
        "物理网格加密中心": "Physical-Grid Refinement Focus",
        "有效加密中心为 `": "Effective Refinement Focus: `",
        "只重分配固定物理边界内的 x/y 单元；不会移动晶体、Pump、Seed 或独立光学窗口。": (
            "Redistribute only the x/y cells inside the fixed physical boundary; this does "
            "not move the crystal, Pump, Seed, or independent optical windows."
        ),
        "`。有效加密中心为 `": "`. Effective refinement focus: `",
        "`。FFT 只能在均匀光学网格上运行；若物理网格局部加密，则通过有限体积单元重叠把光学功率密度映射到晶体网格，再把布居增益和 Δn 映射回光学网格。映射以积分功率/能量守恒为合同，不会把局部加密误当成 FFT 的非均匀采样。": (
            "`. FFT runs only on a uniform optical grid. With local physical-grid "
            "refinement, finite-volume cell overlaps map optical power density to the "
            "crystal grid and map population gain and Δn back to the optical grid. The "
            "mapping contract conserves integrated power/energy and never treats local "
            "refinement as nonuniform FFT sampling."
        ),
        "时间分箱能量": "Temporal Bin Energy",
        "每个时间分箱能量 (J)": "Energy per Temporal Bin (J)",
        "仅有一个时间分箱，缺少时间边界；显示分箱能量，不能据此推断功率。": (
            "Only one temporal bin is available without time boundaries. "
            "Bin energy is shown; power cannot be inferred."
        ),
        "FFT 只能在均匀光学网格上运行；若物理网格局部加密，则通过有限体积单元重叠把光学功率密度映射到晶体网格，再把布居增益和 Δn 映射回光学网格。映射以积分功率/能量守恒为合同，不会把局部加密误当成 FFT 的非均匀采样。": (
            "FFT runs only on a uniform optical grid. With local physical-grid "
            "refinement, finite-volume cell overlaps map optical power density to the "
            "crystal grid and map population gain and Δn back to the optical grid. The "
            "mapping contract conserves integrated power/energy and never treats local "
            "refinement as nonuniform FFT sampling."
        ),
    }
)


def normalize_locale(locale: Any) -> str:
    """Return one supported locale token without leaking display labels."""

    value = str(locale or ZH_CN).strip().lower().replace("_", "-")
    return EN if value.startswith("en") else ZH_CN


def has_han(text: Any) -> bool:
    return isinstance(text, str) and bool(_HAN_RE.search(text))


@lru_cache(maxsize=1)
def english_catalog() -> dict[str, str]:
    with _CATALOG_PATH.open("r", encoding="utf-8") as stream:
        loaded = json.load(stream)
    if not isinstance(loaded, dict):
        raise TypeError("English UI catalog must be a JSON object")
    catalog = {str(source): str(target) for source, target in loaded.items()}
    catalog.update(_EN_OVERRIDES)
    return catalog


@lru_cache(maxsize=1)
def _fragments_by_first_character() -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for source in english_catalog():
        if not source or not has_han(source):
            continue
        grouped.setdefault(source[0], []).append(source)
    return {
        first: tuple(sorted(messages, key=len, reverse=True))
        for first, messages in grouped.items()
    }


def _translate_fragments(text: str) -> str:
    """Translate f-string fragments while leaving inserted values untouched."""

    catalog = english_catalog()
    grouped = _fragments_by_first_character()
    rendered: list[str] = []
    cursor = 0
    while cursor < len(text):
        match = next(
            (
                source
                for source in grouped.get(text[cursor], ())
                if text.startswith(source, cursor)
            ),
            None,
        )
        if match is None:
            rendered.append(text[cursor])
            cursor += 1
            continue
        rendered.append(catalog[match])
        cursor += len(match)
    return "".join(rendered)


def _polish_english(source: str, target: str) -> str:
    polished = target
    for pattern, replacement in _TARGET_FIXES:
        polished = pattern.sub(replacement, polished)
    for marker, pattern, replacement in _SOURCE_AWARE_FIXES:
        if marker in source:
            polished = pattern.sub(replacement, polished)
    for chinese, english in (
        ("，", ", "),
        ("。", ". "),
        ("；", "; "),
        ("：", ": "),
        ("（", " ("),
        ("）", ")"),
    ):
        polished = polished.replace(chinese, english)
    polished = re.sub(r"[ \t]+\n", "\n", polished)
    polished = re.sub(r" {2,}", " ", polished)
    return polished.rstrip()


@lru_cache(maxsize=16384)
def translate_text(text: str, locale: str = ZH_CN) -> str:
    """Translate one display string without mutating its underlying value."""

    source = str(text)
    if normalize_locale(locale) != EN or not has_han(source):
        return source
    target = english_catalog().get(source)
    if target is None:
        target = _translate_fragments(source)
    return _polish_english(source, target)


def localize_source_message(text: Any, locale: str = ZH_CN) -> str:
    """Localize solver-originated English messages without changing schemas."""

    source = str(text)
    if normalize_locale(locale) == ZH_CN:
        return _ZH_SOURCE_MESSAGES.get(source, source)
    return translate_text(source, EN)


def localize_structure(
    value: Any, locale: str, *, translate_keys: bool = True,
    translate_values: bool = True,
) -> Any:
    """Localize labels independently of free-text table values.

    Editors can translate their column labels while preserving user data with
    ``translate_values=False``. Controlled enum values opt into translation.
    """

    if normalize_locale(locale) != EN:
        return value
    if isinstance(value, str):
        return translate_text(value, locale) if translate_values else value
    if isinstance(value, dict):
        return {
            (translate_text(str(key), locale) if translate_keys and isinstance(key, str) else key):
            localize_structure(item, locale, translate_keys=translate_keys,
                               translate_values=translate_values)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [localize_structure(item, locale, translate_keys=translate_keys,
                                   translate_values=translate_values) for item in value]
    if isinstance(value, tuple):
        return tuple(localize_structure(item, locale, translate_keys=translate_keys,
                                       translate_values=translate_values) for item in value)
    return value


def reverse_catalog(locale: str) -> dict[str, str]:
    """Map translated display values back to their stable Chinese source IDs."""

    if normalize_locale(locale) != EN:
        return {}
    # Prefer longer source strings when an automatic translation collision
    # exists; form-specific exact values are usually covered by overrides.
    reverse: dict[str, str] = {}
    for source in sorted(english_catalog(), key=len, reverse=True):
        reverse.setdefault(translate_text(source, EN), source)
    return reverse


def restore_localized_structure(
    value: Any, locale: str, *, translate_keys: bool = True,
    translate_values: bool = True,
) -> Any:
    """Restore controlled labels without guessing the meaning of free text."""

    if normalize_locale(locale) != EN:
        return value
    if isinstance(value, str):
        return reverse_catalog(locale).get(value, value) if translate_values else value
    if isinstance(value, dict):
        reverse = reverse_catalog(locale) if translate_keys else {}
        return {
            reverse.get(key, key) if translate_keys and isinstance(key, str) else key:
            restore_localized_structure(item, locale, translate_keys=translate_keys,
                                        translate_values=translate_values)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [restore_localized_structure(item, locale, translate_keys=translate_keys,
                                            translate_values=translate_values) for item in value]
    if isinstance(value, tuple):
        return tuple(restore_localized_structure(item, locale, translate_keys=translate_keys,
                                                translate_values=translate_values) for item in value)
    return value
