"""Human-readable explanations of the configured amplifier solver chain.

The Streamlit UI and exported model description both use this module.  Keeping
the explanation outside the page prevents the prose from drifting away from
the same resolver that selects the runtime algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .config import resolve_physics, resolve_population_dynamics, resolve_signal_model, resolve_thermal_model
from .josa_modes import JOSA_FULL_3D_MODE
from .multipass import uses_causal_clock


@dataclass(frozen=True)
class ModelFeature:
    stage: str
    feature: str
    configured: str
    effective: str
    source: str
    explanation: str


@dataclass(frozen=True)
class ModelStage:
    title: str
    explanation: str
    key_relations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelDescription:
    stages: tuple[ModelStage, ...]
    features: tuple[ModelFeature, ...]
    notices: tuple[str, ...]

    @property
    def paragraphs(self) -> tuple[str, ...]:
        """Compatibility view used by older callers and lightweight tests."""

        return tuple(stage.explanation for stage in self.stages)

    def as_markdown(self) -> str:
        lines = ["# 当前放大器模型说明", "", "## 端到端计算链路", ""]
        for index, stage in enumerate(self.stages, start=1):
            lines.extend((f"### {index}. {stage.title}", "", stage.explanation, ""))
            if stage.key_relations:
                lines.append("关键物理关系：")
                lines.extend(f"- {relation}" for relation in stage.key_relations)
                lines.append("")
        lines.extend(("## 当前模型功能清单（含关闭与锁定项）", ""))
        for item in self.features:
            lines.append(
                f"- **{item.stage} · {item.feature}**：`{item.effective}`；"
                f"来源：{item.source}。{item.explanation}"
            )
        if self.notices:
            lines.extend(("", "## 锁定、冲突与适用性提示", ""))
            lines.extend(f"- {notice}" for notice in self.notices)
        return "\n".join(lines).strip() + "\n"


_OPTION_PRINCIPLES: dict[str, str] = {
    "auto": "由精度预设或输入数据解析；最终算法必须以“实际生效”而不是 auto 字样为准。",
    "fast": "使用低成本近似建立趋势和工作流，通常关闭三维热场与耗尽。",
    "balanced": "在三维空间传播、耗尽和运行时间之间折中，适合作为一般定量起点。",
    "detailed": "优先完整空间传播和三维热模型，计算量与采样要求最高。",
    "fluence": "把脉冲压缩为单色横向能量通量；可做 Frantz–Nodvik 饱和，但不保留光谱相位或脉内时间结构。",
    "spectral_time": "在波长和时间采样上输运能量，并按局部截面应用增益；保留光谱强度和脉冲包络，不保留频率间相干相位。",
    "josa_transport": "按时间×波长×横向空间输运光子通量并依次更新真实布居，适合强脉内耗尽；它不保留频率间相干相位，四个模式改变积分/时间坐标，full_3d_pc 还锁定多物理栈和近似群延迟走离。",
    "coherent_cpa": "传播复光谱场，保留振幅与相位，可计算色散、压缩、Kerr 相位和时间域强场损耗。",
    "commercial_app_fast": "把整脉冲光谱能量与横向 Gaussian q 参数耦合；每层增益照常计算，热场、Kerr 和增益导引只保留等效二次矩。可选把互不相干 HG 模态作为独立 q 通道并共享局部布居，也可用一维 effective-area 包络计算 SPM、Gaussian K 阶矩强场损耗与载流子闭合。",
    "one_way": "按 Pump→布居→热场→Seed 顺序只计算一次；Seed 和热透镜不反向修改 Pump 固定点。",
    "thermal_optical_fixed_point": "反复求 Pump、布居、热场与折射率，直至吸收、反转和热相位共同收敛。",
    "cycle_average": "用重复频率把单脉冲能量变成平均光子流，求代数周期平均稳态；适合单脉冲布居扰动较小的高重复频率系统。",
    "reduced_pulse_map": "先用周期平均作初值，再用所选的低维整脉冲 FN 或轴对称相干 Kerr 图靠近周期固定点；它只负责初始化，最终仍由当前完整主脉冲 pulse map 按配置容差收敛。",
    "pulse_map": "逐次执行完整 Pump 周期和 Seed 脉冲映射，直到周期边界布居闭合；成本高但能处理强单脉冲耗尽。",
    "anderson": "利用若干历史残差构造固定点候选步，并用守恒与残差门拒绝不可靠加速步。",
    "transient": "显式保留时间演化或脉冲前后布居变化。",
    "steady_state": "求解净跃迁速率为零的稳态布居，不代表系统真实经历了相同数量的脉冲。",
    "frozen": "跳过 Pump 预备和暗态布居推进，使用配置初态；Seed 传播是否耗尽布居仍由增益饱和开关决定。",
    "fft": "在均匀横向网格上传播复场，包含衍射、孔径和三维折射率相位；必须满足窗口与 Nyquist 边界。",
    "hybrid_abcd_fft": "晶体外用解析 ABCD 传递 q 参数，晶体内切换到均匀 FFT 网格响应三维增益和热折射率。",
    "abcd": "用一阶光学和二阶矩传播束宽/曲率，速度快，但不能生成任意横向衍射结构。",
    "prescribed_caustic": "直接按给定 waist、焦点和 M²重建晶体内光斑；不传播外部复场，明显耗尽时需检查坐标固定近似。",
    "super_gaussian_abcd": "保持超高斯强度包络并用 ABCD 更新尺度，适合非相干平顶束的快速传播。",
    "stochastic_raytrace": "采样非相干几何光线，经光学元件和晶体路径传播，并把吸收沉积到三维体素。",
    "local_field": "每个横向单元使用自己的局部增益作用于场，可产生增益导引和空间孔烧蚀。",
    "gaussian_mode_overlap": "把三维反转与 Gaussian 或 Commercial App 显式 HG 通道做重叠积分，每个 z 层只产生模态标量增益，不产生任意高阶增益导引。",
    "parabolic_gain_fit": "在指定横向椭圆内把真实三维增益逐层拟合为轴向二次分布，再作用于复场；它复现 Commercial App Gauss-BPM 的增益导管近似，不改动泵浦、布居或光谱物理参数。",
    "sampled_phase": "直接使用每个网格层的三维热折射率相位，保留采样到的非抛物像差结构。",
    "local_parabolic": "只在指定椭圆邻域内拟合局部二次热透镜，再用拟合相位传播；适合与 Gaussian/Commercial App 式局部透镜近似对齐。",
    "axis_aligned": "只保留 x² 与 y² 曲率，明确舍弃 xy 交叉项；适合主轴已与网格对齐的像散热透镜。",
    "full_xy": "保留 x²、xy 与 y² 完整二维二次型，可表示相对于网格旋转的局部像散主轴。",
    "full_coherent_span": "把 Commercial App 粗增益回灌箱铺满相干 FFT 频率范围，优先保留由 Kerr 新产生的边带。",
    "seed_energy_support": "只在输入 Seed 的有效能量支撑内布置回灌箱，分辨率较集中，但必须检查新生边带的支撑外泄漏。",
    "coating": "使用配置的前后表面功率透过率，适合已知镀膜数据。",
    "fresnel": "用法向 Fresnel 系数计算界面损耗。",
    "fresnel_angle": "按入射角和偏振计算 Fresnel 系数。",
    "ideal": "界面不引入反射损耗，仅用于理想化比较。",
    "pump_component_ledger": "把离子主动吸收、材料被动吸收、ASE 和 Seed 不可逆损耗分项记账；主动吸收再按所选产热比例转换。",
    "event_energy_ledger": "按每条吸收、受激发射、衰变和非线性事件追踪能量去向，可分配到热、荧光、ASE 或未解析项。",
    "lifetime_quantum_defect": "量子亏损与非辐射衰变分开计算；非辐射热由总寿命和辐射寿命差得到，避免重复使用经验荧光效率。",
    "quantum_defect": "按吸收光子能量与有效跃迁参考能量之差计算热，并把未形成有效激发的吸收能量纳入守恒。",
    "scaled_quantum_defect": "在基础量子亏损上乘独立经验倍率，用于已有热测量标定但缺少完整衰变分解时。",
    "spectral_quantum_defect": "对每个 Pump 波长和局部净吸收截面计算量子亏损，再按局部功率加权。",
    "spectral_scaled_quantum_defect": "光谱分辨量子亏损再乘独立倍率。",
    "explicit_fraction": "直接把实测比例作用于主动吸收功率；最少假设，但不外推未测工况。",
    "steady_3d": "求解三维稳态导热方程与边界换热，输出长期平均温度场。",
    "transient_3d": "按热容和时间步推进三维热扩散，保留升温与冷却历史。",
    "cg": "对守恒有限体积稀疏线性系统使用预条件共轭梯度，适合对称正定导热问题。",
    "sor": "用红黑逐点超松弛迭代温度场，内存低但通常比稀疏 CG 慢。",
    "constant": "使用固定折射率，不把求得的温度或应力场反馈到光学传播。",
    "thermal_3d": "由局部温度、dn/dT、应力光弹和端面形变构造三维光程差。",
    "operating_point": "在统一工作温度和参考掺杂浓度下修正寿命与截面。",
    "local_3d": "在热光固定点内逐体素按局部温度/掺杂重新计算寿命与截面。",
    "scale_table": "对用户或材料库给出的坐标—缩放表做受控插值，并按越界策略记录或报错。",
    "linear_scale": "用参考点附近的一阶相对斜率缩放材料参数，只适合标定范围内。",
    "stern_volmer": "用 Stern–Volmer 型浓度猝灭律缩短总寿命。",
    "six_ray": "沿 ±x、±y、±z 六个方向近似 ASE 输运，用于快速寄生风险筛选。",
    "angular_discrete_ordinates": "在离散立体角方向上输运 ASE，并用反射边界幂迭代估计主寄生本征值。",
    "kerr_time_domain": "在相干时间域按 n₂I 累积瞬时非线性相位，不产生热。",
    "kerr_parabolic_lens": "把横向 Kerr 相位投影成等效 Gaussian 二次透镜；强度由所选时间闭合决定，CW 使用平均强度，脉冲使用有效脉宽或一维包络。可选的一维 SPM 由独立时间模型计算。",
    "strong_field_time_domain": "按瞬时时间强度计算多光子吸收和可选自由载流子吸收，并把不可逆损耗单独记账。",
    "strong_field_effective_area": "在 Commercial App 一维相干包络上，用当前 Gaussian 有效面积及其精确 K 阶弱损耗矩闭合多光子吸收，并可推进标量载流子、FCA 与载流子折射率；不解析径向载流子梯度。",
    "rate_equation": "推进自由载流子产生与复合速率方程，再由载流子密度计算 FCA 和折射率变化。",
    "exploratory": "保留完成但无效的结果和全部诊断，适合调参；不会把失败点冒充定量结论。",
    "strict": "同样执行诊断，但阻止验证失败结果的正式导出。",
    "uniform": "使用均匀物理网格，便于 FFT 或规则有限差分，但细腰区可能需要更多全局点数。",
    "refined_region": "指定中心细网格的覆盖宽度和格子数；区域内均匀，区域外几何渐变，物理边界和总格子数保持不变。",
    "center_stretched": "把布居、热和应力物理网格向指定加密点聚集；未独立指定时逐轴跟随物理网格中心，FFT 仍必须使用独立均匀光学网格。",
    "shared": "Pump 复用 Seed 的独立均匀光学网格。",
    "independent": "为 Pump 单独配置均匀光学窗口和采样数。",
    "off": "关闭该模块；它不会参与求解或能量账本。",
    "on": "显式开启该模块，而不是交给精度预设决定。",
    "scipy": "使用 scipy.fft 的 pocketfft 后端；大数组可按配置使用多个 CPU worker，数值归一化与 NumPy 路径一致。",
    "numpy": "使用 NumPy FFT；依赖最少且结果确定，但没有本程序可控的多线程 worker。",
    "volume_average": "把输入离子密度解释为全晶体体积平均值，并缩放有源区域使离散体积积分保持该平均值。",
    "peak": "把输入离子密度解释为掺杂轮廓的峰值；其他体素按相对轮廓降低，不再守恒全体积平均值。",
    "none": "直接把输入密度乘到轮廓上，不做平均值或峰值再标定。",
    "waist_m2": "用腰斑半径、腰斑位置和 M²定义每个轴的等效束参量；适合已知聚焦光束。",
    "waist_divergence": "由腰斑与远场发散角反推等效 M²，再建立传播初值。",
    "emitter_na": "以发光面半径和数值孔径定义非相干相空间；随机光线后端直接采样位置与方向。",
    "gaussian": "使用高斯强度/能量密度并按给定总功率或总能量归一化。",
    "sech2": "使用 sech² 时间强度包络；给定宽度按强度 FWHM 解释。",
    "rectangular": "在指定宽度内使用常值强度、窗外为零；尖锐边缘需要足够时间采样。",
    "super_gaussian": "使用超高斯分布；阶数越高越接近平顶，同时边缘需要更细采样。",
    "radial_super_gaussian": "按椭圆归一化半径构造二维径向超高斯，而不是 x、y 两个一维函数的乘积。",
    "top_hat": "使用有限支撑的均匀平顶分布；硬边缘的 FFT 衍射和采样要求更高。",
    "annular": "使用环形横向分布，并将环宽和半径共同归一化到总功率。",
    "file": "从外部采样表读取分布，检查有限值与坐标单调性后插值并按声明的纵轴单位归一化。",
    "cw": "把 Pump 视为连续波；输入功率直接进入平均吸收与稳态速率方程。",
    "average": "输入值解释为周期平均功率；有脉冲波形时再由占空比恢复瞬时功率。",
    "monochromatic": "全部 Pump 功率位于中心波长，只查询该波长的吸收/发射截面。",
    "lorentzian": "使用洛伦兹光谱线型并在数值窗口内重新归一化；长尾需要足够宽的波长范围。",
    "relative": "文件纵轴只给相对权重，清洗和积分后归一化到表单中的单脉冲总能量。",
    "energy_per_bin_J": "每行纵轴是该波长分箱内的绝对能量；所有分箱直接求和得到脉冲能量。",
    "energy_density_J_per_nm": "纵轴是每纳米光谱能量密度；按非均匀波长坐标积分得到脉冲能量。",
    "separable": "把联合时间—波长权重写成两个边缘分布的外积，不表示频率随时间的相关性。",
    "matrix": "直接使用时间×波长非负权重矩阵，可表示空间无关的时间—频率强度相关，但不包含复相位。",
    "fallback": "子模块失败时回滚其局部状态并调用明确记录的低阶后端；结果会标记实际回退，不能冒充原模型。",
    "error": "子模块失败立即停止该点，保留错误诊断，不用低阶结果替代。",
    "scalar": "只传播一个标量复振幅，不解析偏振旋转或两个正交分量间能量交换。",
    "jones": "同时传播 Ex、Ey 两个复分量；增益饱和由总强度决定，双折射作为 Jones 算符作用。",
    "elliptical": "由振幅方位角和 Ey−Ex 相对相位共同建立椭圆偏振。",
    "right_circular": "建立等振幅、相差 +π/2 的右旋圆偏振 Jones 初态。",
    "left_circular": "建立等振幅、相差 −π/2 的左旋圆偏振 Jones 初态。",
    "radial_tangential_plane_stress": "用轴对称平面应力降阶式生成径向/切向相位延迟；不代表任意夹持与晶向的完整张量解。",
    "tensor_photoelastic": "把三维应力/应变张量通过旋转后的光弹张量映射为各向异性介电扰动。",
    "sequential": "按固定顺序逐个局部算符推进；成本低，但强非线性时算符排序误差较大。",
    "symmetric_adaptive": "用对称分步和局部增益、相位、损耗变化量估计所需子步；这些阈值不是完整方程的数值误差保证。",
    "amplitude_phase": "复光谱文件的两列数值解释为振幅与相位，构造 A·exp(iφ)。振幅还会按所选 per_frequency、per_wavelength 或 per_bin 域合同换算。",
    "real_imag": "复光谱文件的两列数值直接解释为复场实部和虚部。",
    "forward": "沿光路定义的 +z 方向穿过晶体，吸收、布居提交和记录面按此前后顺序执行。",
    "backward": "沿 −z 方向穿过同一晶体，z 单元访问顺序和对应界面方向反转。",
    "cuboid": "使用长方体有效域和六个平面边界，所有体积分与热/力学边界按盒体裁剪。",
    "cylinder": "使用圆柱有效域；圆外体素不参与材料积分，侧壁换热按曲面面积离散。",
    "two_level": "Pump 与 Signal 共享同一对能级，必须同时计入基态吸收与受激发射。",
    "three_level": "Signal 下能级为基态或显著占据能级，存在再吸收和温度敏感透明阈值。",
    "four_level": "Signal 下能级快速排空，通常再吸收较弱，但仍按配置跃迁和实际布居求解。",
    "custom": "使用配置中的任意能级、光致跃迁和非线性过程网络，并逐项检查粒子数守恒。",
    "fixed_rk4": "以固定内部步长做四阶 Runge–Kutta；可重复且开销可预测，但刚性过程需要更小步长。",
    "adaptive_solve_ivp": "由误差容限自适应选择步长；适合时间尺度差异大的刚性/近刚性局部速率方程。",
    "uniform_cone": "在 NA 圆锥立体角内均匀采样方向，表示均匀角密度而非 Lambert 余弦辐亮度。",
    "lambertian": "按相对表面法线的余弦律采样方向，并按 NA 截断。",
    "segment_length": "把每条光线在体素内的真实路径长度作为沉积权重，守恒分配吸收能量。",
    "nearest": "把射线段能量归入最近体素；速度快但会增加沉积噪声和网格依赖。",
    "edge": "材料表越界时固定使用最近端点值，同时记录越界诊断；只适合明确接受端点外推为常值的情况。",
    "direct": "直接分解稀疏线性系统；小/中型网格稳健，但内存增长通常高于迭代法。",
    "distributed_opd": "把 FEM 端面法向位移作为分布式光程差相位反馈到相应晶体界面。",
    "euler_xyz_deg": "用按约定顺序的 xyz 欧拉角构造晶体到实验坐标的正交旋转。",
    "rotation_matrix": "直接输入 3×3 旋转矩阵；运行前检查正交性与右手性。",
    "linear": "在线性坐标上等间距取遍历点，适合包含零或关注绝对增量的参数。",
    "log": "在正数范围按对数等距取点，适合跨多个数量级的参数。",
    "seed": "切换到 Seed 的传播/结果诊断；不会隐藏或关闭 Pump 的正式耦合计算。",
    "pump": "切换到 Pump 的传播/结果诊断；不会隐藏或关闭 Seed 的正式耦合计算。",
    "automatic": "由当前光束二阶矩和有效占用范围自动选择拟合半径。",
    "manual": "使用用户给定半径做抛物拟合，便于固定不同工况的比较口径。",
    "seed_entrance": "使用 Seed 在晶体入口的束斑尺度作为所有 z 层的统一拟合半径。",
}


_CONTROL_OPTION_PRINCIPLES: dict[str, dict[str, str]] = {
    "精度预设": {key: _OPTION_PRINCIPLES[key] for key in ("commercial_app_fast", "fast", "balanced", "detailed")},
    "记录层级": {
        "simple": "只保留原有最终结果、收敛摘要和完整末次脉冲，不增加逐轮或逐 Z 数据。",
        "medium": "在简单档基础上保留外层热光与内层周期/布居循环的标量历史，不保存逐层场。",
        "detailed": "在中等档基础上按物理 Z 单元聚合纵向标量；自适应子步不会各自保存完整场。",
    },
    "种子求解模型": {key: _OPTION_PRINCIPLES[key] for key in ("auto", "commercial_app_fast", "fluence", "spectral_time", "josa_transport", "coherent_cpa")},
    "热光耦合": {key: _OPTION_PRINCIPLES[key] for key in ("one_way", "thermal_optical_fixed_point")},
    "Pump + Seed 稳态方法": {key: _OPTION_PRINCIPLES[key] for key in ("cycle_average", "pulse_map")},
    "内层稳态求法": {key: _OPTION_PRINCIPLES[key] for key in ("cycle_average", "pulse_map")},
    "离散周期映射初值": {
        "initial_population": "直接从配置或上一个遍历点提供的周期边界布居开始完整主脉冲固定点。",
        "cycle_average": "先求周期平均工作点作为数值初值，随后仍由完整主脉冲固定点收敛。",
        "reduced_pulse_map": _OPTION_PRINCIPLES["reduced_pulse_map"],
        "auto": "根据寿命与重复周期判断是否使用周期平均初值；最终仍由完整主脉冲固定点收敛。",
    },
    "有限脉冲串初始布居": {
        "从设定的 t=0 初始布居开始": "严格使用配置的初始能级布居；可另外指定第一个 Seed 相对 t=0 的到达时间。",
        "自动 — 长寿命/高重复频率时先求平均稳态": "只在平均稳态适用且能安全收敛时用它初始化，否则保留配置初态；有限脉冲本身仍逐周期计算。",
        "先求周期平均工作点再计算有限脉冲串": "强制先求 Pump+Seed 周期平均布居作为数值初值，再从该状态计算指定数量的离散脉冲。",
    },
    "布居动力学": {key: _OPTION_PRINCIPLES[key] for key in ("auto", "transient", "steady_state", "frozen")},
    "种子空间传播": {key: _OPTION_PRINCIPLES[key] for key in ("auto", "hybrid_abcd_fft", "fft", "abcd", "off")},
    "泵浦空间传播": {key: _OPTION_PRINCIPLES[key] for key in ("auto", "prescribed_caustic", "super_gaussian_abcd", "stochastic_raytrace", "hybrid_abcd_fft", "fft", "abcd")},
    "Seed 空间增益闭合": {key: _OPTION_PRINCIPLES[key] for key in ("local_field", "gaussian_mode_overlap", "parabolic_gain_fit")},
    "界面模型": {key: _OPTION_PRINCIPLES[key] for key in ("auto", "coating", "fresnel", "fresnel_angle", "ideal")},
    "热源账本": {key: _OPTION_PRINCIPLES[key] for key in ("pump_component_ledger", "event_energy_ledger")},
    "主动 Pump 吸收产热模型": {key: _OPTION_PRINCIPLES[key] for key in ("lifetime_quantum_defect", "quantum_defect", "scaled_quantum_defect", "spectral_quantum_defect", "spectral_scaled_quantum_defect", "explicit_fraction")},
    "热场模型": {key: _OPTION_PRINCIPLES[key] for key in ("auto", "off", "steady_3d", "transient_3d")},
    "稳态热线性求解器": {key: _OPTION_PRINCIPLES[key] for key in ("auto", "cg", "sor")},
    "折射率模型": {key: _OPTION_PRINCIPLES[key] for key in ("auto", "constant", "thermal_3d")},
    "光谱参数反馈模式": {key: _OPTION_PRINCIPLES[key] for key in ("off", "operating_point", "local_3d")},
    "上能级寿命随温度": {key: _OPTION_PRINCIPLES[key] for key in ("off", "linear_scale", "scale_table")},
    "上能级寿命随掺杂浓度": {key: _OPTION_PRINCIPLES[key] for key in ("off", "stern_volmer", "scale_table")},
    "ASE 模型": {key: _OPTION_PRINCIPLES[key] for key in ("off", "six_ray", "angular_discrete_ordinates")},
    "非线性相位模型": {key: _OPTION_PRINCIPLES[key] for key in ("auto", "off", "kerr_parabolic_lens", "kerr_time_domain")},
    "不可逆强场损耗模型": {key: _OPTION_PRINCIPLES[key] for key in ("off", "strong_field_time_domain")},
    "自由载流子模型": {key: _OPTION_PRINCIPLES[key] for key in ("off", "rate_equation")},
    "物理网格横向间距": {key: _OPTION_PRINCIPLES[key] for key in ("uniform", "center_stretched", "refined_region")},
    "Pump 光学网格": {key: _OPTION_PRINCIPLES[key] for key in ("shared", "independent")},
    "强度输入方式": {
        "time": "由时间复场生成光谱；缺少时间相位时采用零相位。",
        "spectrum": "由复光谱生成时间场；缺少光谱相位时采用平坦相位。",
        "dual": "用时间与光谱强度共同约束相位恢复；可行解未必唯一。",
        "independent": "分别定义时间和光谱强度，不强制两者组成同一个傅里叶一致复场。",
    },
    "结果有效性模式": {key: _OPTION_PRINCIPLES[key] for key in ("exploratory", "strict")},
    "功率定义": {
        "average": "把填写值作为周期平均功率；脉冲 Pump 会结合波形积分和重复频率恢复峰值/单脉冲能量。",
        "peak": "把填写值作为波形峰值功率；程序积分归一化波形得到单脉冲能量和平均功率。",
    },
    "输入偏振态": {
        key: {
            "linear": "建立两个同相 Jones 分量，方位角决定 Ex 与 Ey 的振幅比例。",
            "elliptical": "由振幅方位角和 Ey−Ex 相对相位共同建立椭圆偏振。",
            "right_circular": "建立等振幅、相差 +π/2 的右旋圆偏振 Jones 初态。",
            "left_circular": "建立等振幅、相差 −π/2 的左旋圆偏振 Jones 初态。",
        }[key]
        for key in ("linear", "elliptical", "right_circular", "left_circular")
    },
    "增益来源": {
        "由泵浦和速率方程求解": "由局部 Pump 吸收和能级速率方程求布居，再计算波长相关净增益；可与耗尽、热和周期稳态闭环。",
        "直接指定小信号增益": "跳过 Pump→布居反演，直接使用给定 g0；适合后端验证，不可解释泵浦效率、反转或热账本。",
    },
    "光路 / 多程组织方式": {
        "一键设置程数": "由一次晶体路径加程间元件模板生成指定次数的 gain 命中，适合规则多程。",
        "自己搭建光学平台": "按表中 gain_medium、透镜、反射镜、孔径和自由空间的实际顺序传播，每次晶体命中可独立定向。",
    },
    "配置入口": {
        "表单": "由受约束控件编辑并即时显示相容性；未知插件字段仍从底层配置保留。",
        "完整 JSON": "直接编辑完整配置树并经过同一 schema/物理验证，适合批量粘贴或高级字段。",
    },
    "侧栏显示": {
        "清爽设置": "把逐项后端覆盖和长参数区合并折叠；高级字段仍可按需展开，不会关闭任何已选物理模块。",
        "完整设置": "显示全部后端覆盖和高级数值选项，便于逐项审查与调试；与清爽设置共享同一份配置。",
    },
    "输入坐标基准": {
        "本束 LASER 源面 = 0": "各束以自己的发光面为 z=0，元件距离按该束传播顺序累计。",
        "共享晶体入口 = 0": "Pump 与 Seed 都以同一晶体入口为 z=0，源和前级元件位于负坐标，便于比较两路焦点。",
    },
    "横向模态模型": {
        "M² 有效高斯（快速）": "把 M²压缩成单个等效 Gaussian 的束宽/发散，最快但不解析高阶模态的独立孔径与增益。",
        "HG 强度集合（快速降阶）": "用非相干 Hermite–Gaussian 强度权重重建横向分布，共享一个传播尺度。",
        "HG 模态逐个传播（共享布居）": "每个 HG 模态独立经过光路和局部增益，再以强度相加；模态共享同一布居耗尽账本。",
    },
    "不同波长的横向源模型": {
        "共享同一个源面参数（兼容/默认）": "所有波长使用相同腰斑、曲率和中心位置；只因 λ 不同而有传播差异。",
        "参数化色差与空间啁啾": "按波长连续改变腰斑位置、尺度或横向中心，用少量斜率表示色差/空间啁啾。",
        "逐波长参数表（缺项自动沿用参数化值）": "在指定波长节点读取独立源参数并插值；缺少的量继续使用参数化基线。",
    },
    "Pump 横向计算网格": {
        "与 Seed/BPM 共用": "Pump 与 Seed 共用独立均匀光学网格；内存较省，但两束尺寸差异大时难兼顾窗口和分辨率。",
        "使用物理网格": "Pump 直接在布居/热物理网格计算；仅适用于不要求均匀 FFT 采样的 Pump 后端。",
        "独立均匀网格": "Pump 使用自己的均匀窗口与 Nx/Ny，并通过有限体积重叠与物理网格守恒交换吸收和折射率。",
    },
    "x/y 网格分布": {
        "均匀网格": "物理网格横向单元等宽；简单稳健，FFT 复用时也是唯一允许的形式。",
        "Gaussian 中心加密": "仅把布居、热和应力物理单元向指定加密点聚集；默认跟随物理网格中心，光学 FFT 仍需独立均匀网格并守恒映射。",
    },
    "晶体内 Seed 体数据": {
        "last_and_cumulative": "保存最后一程和所有程累计三维场，兼顾常用诊断与内存。",
        "last_only": "只保存最后一次晶体命中的三维场，传播物理不变且内存最低。",
        "per_pass": "保存每一次晶体命中及累计场，便于逐程诊断但内存随程数增长。",
    },
    "计算层级 / JOSA 输运模式": {
        "retarded_pc": "在延迟时间 T=t−z/vg 中用 predictor–corrector 推进；时间分箱与 Nz 独立，适合一般 front→tail 耗尽。",
        "full_3d_pc": "在三维横向网格上联立 Pump/Seed 输运和瞬态布居，并锁定 FFT、耗尽、热光及诊断合同；群延迟混箱采用能量守恒相位闭合近似，不允许静默回退 FN。",
        "pc_cfl": "按 Δt=ngΔz/c 的论文 CFL 网格使用 predictor–corrector；改变 Nz 会同时改变时间采样。",
        "explicit_cfl": "按论文 CFL 网格做单次显式推进，最接近基础离散式，但时间/空间分辨率不足时误差最大。",
    },
    "局部算符模式": {
        "sequential": _OPTION_PRINCIPLES["sequential"],
        "symmetric_adaptive": _OPTION_PRINCIPLES["symmetric_adaptive"],
    },
    "相干模块失败时": {
        "fallback": _OPTION_PRINCIPLES["fallback"],
        "error": _OPTION_PRINCIPLES["error"],
    },
    "瞬态增益子模块失败时": {
        "fallback": _OPTION_PRINCIPLES["fallback"],
        "error": _OPTION_PRINCIPLES["error"],
    },
    "JOSA 子模块失败时": {
        "fallback": _OPTION_PRINCIPLES["fallback"],
        "error": _OPTION_PRINCIPLES["error"],
    },
    "插入元件": {
        "薄透镜": "插入薄透镜 ABCD/相位算符，焦距决定波前曲率变化而不增加传播长度。",
        "厚透镜": "插入含前后折射面和内部厚度的厚透镜传递，材料折射率与色散可随波长变化。",
        "自由空间": "插入只产生传播距离的节点，用于显式组织坐标和记录位置。",
        "反射镜": "插入反射/折返节点，并按设定曲率、透过/反射损耗和传播方向作用。",
        "光阑": "插入有限孔径，截去的功率进入具名外部光学损耗账本。",
        "观察面": "插入只记录场和二阶矩、不改变场的诊断平面。",
        "双程双透镜回程成像": "插入预定义的双程回程成像组合，并展开为可审计的独立元件顺序。",
    },
    "拟合半径来源": {
        key: _OPTION_PRINCIPLES[key]
        for key in ("automatic", "manual", "seed_entrance")
    },
}


_CONTROL_CONTEXTS: dict[str, str] = {
    "时间强度来源": "定义 Seed 在时间上的归一化强度包络；它影响峰值强度、瞬态耗尽和 Fourier 光谱。",
    "频率/光谱强度来源": "定义 Seed 能量在波长上的分配；每个分箱查询自己的截面并参与增益窄化。",
    "光谱文件纵轴": "声明导入光谱第二列的物理单位，决定总能量是归一化得到还是由文件积分得到。",
    "强度输入方式": "选择单域傅里叶构造、双域相位恢复，或明确的独立时间／光谱近似。",
    "时间–频率联合能量": "只用于非相干 spectral_time，决定时间箱与波长箱的能量是否可分离。",
    "计算层级 / JOSA 输运模式": "四种模式都使用时间×波长×横向空间通量；它们改变局部积分格式与时间坐标，full_3d_pc 还锁定完整三维多物理栈和近似群延迟走离。",
    "Seed 偏振传播": "决定是否传播 Ex/Ey 两个复分量及其双折射耦合。",
    "输入偏振态": "定义 Jones 模式的归一化输入向量，不改变总 Seed 能量。",
    "晶体内双折射": "把热应力或完整光弹张量转换为两个偏振分量间的局部相位延迟。",
    "空间光斑": "定义源面横向强度形状；所有形状都会归一化到同一输入功率/能量。",
    "横向模态模型": "决定 M²只是等效束质因子，还是展开为多个非相干 HG 模态。",
    "不同波长的横向源模型": "决定宽带 Seed 各波长是否共用源面几何，以及是否表示色差和空间啁啾。",
    "局部算符模式": "决定 coherent CPA 中增益、色散、Kerr 与损耗算符的排列和自适应子步。",
    "复光谱文件格式": "声明外部复场两列的复数编码；读取后仍按总脉冲能量规范化。",
    "标定方式": "决定非均匀掺杂轮廓如何与表单中的参考离子密度对应。",
    "掺杂分布中心切片": "只选择三维掺杂预览的中心切面，不改变求解网格或材料。",
    "插入位置": "决定新光学元件在有序路径中的索引，因此改变其前后自由传播和场状态。",
    "插入元件": "选择要加入真实光路的传递算符类型；元件参数随后独立编辑。",
    "选中并编辑元件": "选择现有路径节点进行编辑，不复制或重排其他节点。",
    "通过方向": "决定 gain_medium 命中时访问晶体 z 单元和前后界面的方向。",
    "当前查看": "只切换实时预览的主标签，Pump 与 Seed 的正式配置都保留。",
    "精确周期映射初值": "决定 pulse_map 固定点从冷态、Pump 预稳态或上次收敛态开始；只影响收敛路径。",
    "晶体内 Seed 体数据": "控制三维诊断场的保留范围，不改变传播和布居提交。",
    "FFT 计算后端": "选择均匀光学网格复场变换的实现；物理归一化和边界诊断一致。",
    "晶轴/偏振": "选择材料库各向异性截面/折射率曲线的晶轴分支。",
    "材料快照越界策略": "决定导入材料表超出测量坐标时停止还是使用端点，同时总会记录越界诊断。",
    "晶体形状": "决定有效材料体素、光学裁剪、换热面积和力学边界几何。",
    "能级模型": "决定能级拓扑、Pump/Signal 跃迁连接和再吸收结构。",
    "上转换/交叉弛豫数值推进": "决定自定义非线性能级项的时间积分器；不会改变配置的速率常数。",
    "Pump 光源定义": "只定义 Pump 源面相空间；传播后端另行选择。",
    "角分布": "定义 emitter+NA 随机光线的方向概率密度。",
    "射线到晶体网格的沉积": "决定每条 Pump 光线的吸收能量如何守恒分配到三维体素。",
    "泵浦方向": "决定 Pump 穿过晶体的 z 顺序和入口界面。",
    "泵浦波形": "定义 Pump 的周期内时间功率，从而影响瞬态布居和平均热负载。",
    "功率定义": "声明填写的 Pump 功率是平均值还是脉冲峰值，避免重复乘占空比。",
    "文件纵轴": "声明 Pump 波形文件第二列的功率含义与归一化方式。",
    "泵浦光谱": "定义 Pump 功率在波长上的分配，使吸收截面、量子亏损和耗尽可逐波长计算。",
    "宽带 Pump 空间输运": "决定宽带 Pump 各波长共享一条空间路径还是逐波长传播。",
    "圆柱热边界面积离散": "决定曲面体素对圆柱侧壁换热面积的几何估计方法。",
    "dn/dT 数据定义": "声明折射率温度系数是否包含自由热膨胀的光弹性贡献；这是 dn/dT，并非光程 d(nL)/dT。端面位移产生的几何光程另行计算。",
    "拟合半径来源": "决定热透镜/增益抛物拟合使用的横向半径，只影响诊断拟合。",
    "冷却边界": "选择每个外表面的 Dirichlet、对流或绝热热边界条件。",
    "应力模型": "决定是否求热弹性应力，以及使用轴对称降阶还是三维 FEM。",
    "力学线性求解器": "选择离散 FEM 位移系统的直接或迭代线性代数后端。",
    "FEM 应力/应变到折射率": "决定光弹反馈使用应力形式还是应变形式的材料张量。",
    "完全固定的晶体表面（留空=全部无牵引）": "选择位移为零的 FEM 边界；留空采用自由表面并用最小约束去除刚体模。",
    "晶体取向输入": "定义材料晶轴到实验 xyz 网格的旋转。",
    "光弹矩阵作用的应变": "选择总应变 εtotal 或弹性应变 εelastic=εtotal−αΔT；两者均使用工程剪切 Voigt 约定。固定应变 dn/dT 应配总应变，已含自由膨胀的实测 dn/dT 应配弹性应变。",
    "FEM 端面位移光学反馈": "决定是否把热致端面形变作为分布式相位加入光路。",
    "材料表越界策略": "控制温度/浓度依赖表越过测量范围时的处理，并保留相应验证记录。",
    "遍历参数": "选择每个独立仿真点唯一改写的主输入，其他配置保持不变。",
    "第二遍历参数": "选择二维笛卡尔网格的第二个独立输入。",
    "间隔": "决定一维遍历点在线性坐标还是对数坐标等距生成。",
    "第二参数间隔": "决定二维遍历第二轴在线性或对数坐标取点。",
    "输出量（可多选）": "选择遍历表和图中提取的结果/诊断列；不会改变每个点的求解。",
    "三维诊断场": "选择已经计算并保存的体数据进行显示，不触发新的物理求解。",
    "切片方向": "选择三维场的正交切面及相应坐标轴。",
    "查看光学面": "选择已保存的传播平面进行光斑/M²诊断；未保存平面不会事后重建。",
    "当前详细标签（Pump 与 Seed 光束体始终同时显示）": "只切换详细信息卡的主对象；三维图仍同时显示两束，正式结果不变。",
    "晶体内叠加场（映射到第一程晶体）": "选择一项已保存的三维物理场，守恒映射到第一程晶体位置叠加显示；不会把显示映射反馈到求解。",
}


_TOGGLE_PRINCIPLES: dict[str, str] = {
    "求 Pump + Seed 周期稳态": "开启后求重复运行的周期边界固定点；关闭时只计算指定周期数或单次预备 Pump。",
    "自动固定点加速": "开启 Anderson 加速并保留残差/守恒回退；关闭后使用原始松弛固定点迭代。",
    "自动加速热光固定点": "用带保护的 Anderson 历史加速热折射率固定点；候选步不可靠时自动退回普通迭代。",
    "泵浦耗尽": "开启后 Pump 吸收会沿传播路径降低剩余 Pump 功率并改变下游布居；关闭时使用未耗尽 Pump 分布。",
    "增益饱和": "开启后 Seed 提取会降低局部反转并限制增益；关闭时使用小信号增益。",
    "保存各光学面": "只控制诊断数据保存，不改变传播物理；关闭可节省内存，但不能定量解释未记录平面的光斑/M²。",
    "启用独立均匀 Seed 光学网格": "开启后 FFT/BPM 在独立均匀网格上运行，并与物理网格守恒映射；关闭时复用物理网格。",
    "把 Seed 被动吸收作为热源": "只把 Seed 的不可逆被动吸收记入热源；主动增益提取不是吸收热。",
    "使用导入的 Sellmeier 模型": "按波长计算材料折射率和色散相位；关闭后只使用固定折射率与手工高阶色散。",
    "保存每个 z 单元的模态重叠增益与功率": "只保存 Gaussian 模态重叠闭合的轴向标量增益和功率诊断，不改变该闭合本身。",
    "脉冲内部推进自发弛豫/自定义非线性过程": "在 JOSA 每个脉冲子步同时推进寿命衰变、上转换和交叉弛豫；脉宽远短于这些时间尺度时可关闭以提速。",
    "显示 JOSA 数值保护设置": "只展开失败策略、单步增益建议值和 CFL 分箱上限，不改变已保存配置。",
    "按局部物理变化自动增加子步": "根据单步增益、相位和不可逆损耗变化量估计所需子步；达到子步上限后仍超出目标的结果不能通过相应有效性门。",
    "逐时间窗瞬态增益（front → peak → tail）": "让相干脉冲各时间窗依次读取并提交布居，使脉冲前沿可以改变后沿所见增益。",
    "按单窗布居耗尽自动加密": "当一个时间窗引起的局部布居跳变过大时自动拆分该窗，降低排序误差并保留能量守恒。",
    "脉冲内部同时推进能级弛豫/上转换": "在相干瞬态增益子步中同步推进非受激能级过程；超短脉冲通常可关闭。",
    "高级：用复光谱文件覆盖上面的统一强度定义": "使用外部复光谱直接替换重建场的振幅和相位，但仍检查中心波长、采样和总脉冲能量。",
    "启用非均匀掺杂": "用三维相对掺杂轮廓替代常数离子密度，并按标定方式保持平均值、峰值或原始比例。",
    "启用未掺杂/低掺杂端帽": "在同一热传导基质内降低端部活性离子密度；端帽仍参与导热、应力和光学传播。",
    "保存 VTK": "把已经求得的三维体数据额外写成 VTK 供外部可视化；不改变求解。",
    "晶体外区域作为硬孔径": "光场映射到晶体截面外的部分被作为孔径损耗扣除并记账；关闭时不做几何硬裁剪。",
    "应用吸收/发射/再吸收光谱": "选择材料后载入其波长相关截面曲线；关闭则保留当前配置里的截面参数。",
    "应用 Sellmeier 折射率与材料色散": "选择材料后载入其 Sellmeier 系数；不会覆盖手工附加的 GDD/TOD/FOD。",
    "使用库中的默认产热比例": "选择材料时同时采用其默认主动吸收产热模型/参数；关闭则保留当前热源设置。",
    "扫描系统输出后的焦散并拟合 D4σ / M²": "从正式系统输出复场向外独立传播多个平面并拟合二阶矩焦散；只做诊断，不反馈晶体。",
    "保存每个焦散平面的二维光斑": "保留焦散扫描的二维数组供查看/下载；关闭仍计算拟合所需二阶矩。",
    "相邻程自动反向": "在一键多程模式中让相邻 gain 命中交替沿 ±z 访问晶体；关闭则全部同向。",
    "相邻程反向通过晶体": "只用于自动迁移的顺序多程计划：开启后相邻命中交替沿 ±z 访问晶体；物理模板的方向由模板本身决定。",
    "诊断热折射率与增益的抛物近似质量": "把实际二维场拟合为二次曲面并报告残差，判断薄透镜/高斯降阶是否可信；不替换实际场。",
    "指定初始晶体温度": "为瞬态热求解提供非均匀/非环境初态；稳态解若收敛则与初态无关。",
    "暂用上能级总寿命作为辐射寿命": "仅在缺少独立辐射寿命时用于 ASE 自发源近似；会明确标记，不能解释非辐射量子效率。",
    "严格模式要求所有外部 Seed 损耗具名闭合": "要求界面、孔径和程间能量差都落入具名账本；任何未解释损耗都会使严格结果无效。",
    "二维参数网格（两个输入同时遍历）": "生成两个参数的笛卡尔积，每个格点仍运行一套独立完整配置。",
    "在图中显示验证无效点（红色 ×，不连入有效趋势线）": "显示已完成但阻断性验证失败的遍历点供诊断，不把它们连接成定量趋势。",
    "在二维图中标出验证无效格点（红色 ×，不进入有效热图）": "用红色标记无效格点，但将其从定量热图数值矩阵排除。",
    "插入增益晶体": "只在实时光路预览中插入/移除晶体界面与长度，帮助比较几何；不修改正式运行配置。",
    "晶体聚焦视图": "压缩长自由空间、放大晶体附近的横向尺度，便于观察 waist；不改变坐标或传播。",
    "晶体聚焦视图（压缩长自由空间）": "仅改变三维结果图的纵向显示映射，不改变任何求解数据。",
    "独立均匀光学网格（双网格耦合）": "为 FFT/BPM 建立独立均匀横向采样，并与布居/热非均匀物理网格做有限体积守恒映射。",
    "独立设置 x 加密中心": "开启后在固定物理 x 边界内把非均匀单元聚集到指定实验室坐标；关闭时跟随物理网格中心 x，不移动任何物体或光束。",
    "独立设置 y 加密中心": "开启后在固定物理 y 边界内把非均匀单元聚集到指定实验室坐标；关闭时跟随物理网格中心 y，不移动任何物体或光束。",
    "显示三维体渲染": "切换已保存体数据的体渲染图；关闭后仍可查看二维切片。",
    "x / y 使用不同腰斑位置（像散光束）": "允许 x、y 两个正交轴拥有不同 waist z 位置和 q 参数，以表示像散；关闭时两轴共用同一焦面。",
    "Pump x / y 使用不同焦点位置（像散光束）": "允许 Pump 的 x、y 两轴拥有不同焦点位置；关闭时两轴共用同一焦面。",
    "每次晶体命中后翻转 x 场": "在相邻晶体命中之间执行一次明确的 x 坐标成像反转；仅用于尚未由光路元件描述的真实翻转。",
}


def _token(value: Any) -> str:
    text = str(value)
    return text.split(" — ", 1)[0].split(" ", 1)[0]


def build_choice_help(
    label: str,
    options: Iterable[Any],
    existing_help: str | None = None,
) -> str:
    """Build a question-mark tooltip that explains every available choice."""

    label_text = str(label)
    guide = _CONTROL_OPTION_PRINCIPLES.get(label_text, {})
    lines: list[str] = []
    if existing_help:
        lines.append(str(existing_help).strip())
    context = _CONTROL_CONTEXTS.get(label_text)
    if context and (not existing_help or context not in str(existing_help)):
        lines.append("这个控件的作用：" + context)
    lines.append("选项原理：")
    for option in options:
        option_text = str(option)
        token = _token(option)
        explanation = (
            guide.get(option_text)
            or guide.get(token)
            or _OPTION_PRINCIPLES.get(option_text)
            or _OPTION_PRINCIPLES.get(token)
        )
        if explanation is None and " — " in option_text:
            explanation = option_text.split(" — ", 1)[1].strip()
        if explanation is None:
            if label_text == "插入位置":
                explanation = f"把新元件插入“{option_text}”对应的有序路径位置；前后节点顺序和累计传播距离随之更新。"
            elif label_text == "选中并编辑元件":
                explanation = f"选择现有节点“{option_text}”作为编辑对象；其余路径节点及顺序保持不变。"
            elif label_text in {"遍历参数", "第二遍历参数"}:
                explanation = f"每个遍历点只改写“{option_text}”，随后重新运行完整耦合与验证，不能把无效点连入定量趋势。"
            elif label_text == "输出量（可多选）":
                explanation = f"从每个已完成且通过相应有效性门的结果中提取“{option_text}”，用于表格/曲线，不反馈求解。"
            else:
                scope = context or f"它决定“{label_text}”这一环节采用哪一个分支"
                explanation = (
                    f"{scope} 选择“{option_text}”时只改变该环节，并由实际生效栈显示它是否被上级模型锁定。"
                )
        lines.append(f"• {option_text}：{explanation}")
    lines.append("若控件被禁用，当前值由相容性合同锁定，不会静默采用另一个可编辑值。")
    return "\n".join(dict.fromkeys(line for line in lines if line))


def build_toggle_help(label: str, existing_help: str | None = None) -> str:
    lines = [str(existing_help).strip()] if existing_help else []
    principle = _TOGGLE_PRINCIPLES.get(str(label))
    if principle:
        lines.append(principle)
    elif not existing_help:
        lines.append(
            f"开启时执行“{label}”所述行为，关闭时完全跳过该行为；"
            "若控件被禁用，则状态由上级模型相容性合同锁定，并会在实际生效栈中说明来源。"
        )
    return "\n".join(dict.fromkeys(line for line in lines if line))


def build_field_help(
    label: str,
    existing_help: str | None = None,
    *,
    control_kind: str = "number_input",
) -> str:
    """Return a concrete tooltip for scalar, text, and file inputs.

    Choice controls already document every branch.  Scalar inputs used to be
    much less consistent, especially for recently added thermal, coherent,
    and grid settings.  These rules provide unit-aware physical context while
    preserving richer inline explanations whenever a control supplies one.
    """

    if existing_help and str(existing_help).strip():
        return str(existing_help).strip()
    text = str(label).strip()
    lower = text.lower()
    exact = {
        "算例名称": "给本次配置一个可追溯名称；它只用于结果包、图表和工况识别，不改变数值模型。",
        "材料名称": "记录当前材料快照的名称，便于结果追溯；真正参与求解的是同一区块中的截面、寿命、热学和光学参数。",
        "名称": "设置当前光路元件的显示名称；用于路径表、记录平面和损耗账本定位，不改变元件类型。",
        "上传配置 JSON（上传后自动回填表单）": "读取一份 JSON 配置并回填全部表单；缺省字段由默认值补齐，导入后仍会执行相容性联动和完整校验。",
        "泵浦截面响应 CSV：λ(nm), σabs, σem": "导入 Pump 波长、吸收截面和发射截面表；传播时每个 Pump 光谱分箱按自己的波长插值截面。",
        "信号截面响应 CSV：λ(nm), σabs, σem": "导入 Seed 波长、吸收截面和发射截面表；用于逐波长净增益、再吸收和增益窄化。",
        "复光谱 CSV：λ(nm), amplitude, phase(rad) 或 real, imag": "导入 Seed 的复光谱振幅与相位；程序会检查采样并按配置的单脉冲总能量重新归一化。",
        "fft 线程数（0=全部可用核）": "限制单次 FFT 可使用的工作线程数；0 交给后端使用可用 CPU 核，不改变传播物理。",
        "启用多线程 FFT 的最小数组元素数": "数组小于该规模时避免线程调度开销，大于该规模时允许多线程 FFT；只影响速度。",
    }
    if lower in exact:
        return exact[lower]
    if text in exact:
        return exact[text]
    if control_kind == "file_uploader":
        return f"导入“{text}”所述数据；文件会先做格式、有限值和坐标检查，再写入当前配置，不会静默覆盖无关模块。"
    if control_kind == "download_button":
        return f"把当前会话中的“{text}”导出到本机；下载只读取当前结果或预览，不会修改配置或重新求解。"
    if control_kind == "button":
        if "运行" in text or "求解" in text:
            return f"执行“{text}”；启动前先检查模块相容性，运行期间在主页面报告阶段、迭代、残差、速度和预计剩余时间。"
        if "插入" in text:
            return f"执行“{text}”；在当前选中位置加入对应光路元件，随后会重新编号并更新右侧光路预览。"
        if "删除" in text:
            return f"执行“{text}”；只移除当前选中的条目，其他光路元件和求解配置保持不变。"
        if "应用" in text:
            return f"执行“{text}”；把当前编辑区或所选方案写回表单，并立即重新做模块联动与配置校验。"
        if "详细显示" in text:
            return f"执行“{text}”；切换右侧主预览正在检查的光束，不会改变 Pump 或 Seed 的物理输入。"
        if "生成" in text or "刷新" in text:
            return f"执行“{text}”；根据当前配置更新说明或预览，本操作本身不会启动数值求解。"
        if "恢复" in text or "重置" in text:
            return f"执行“{text}”；恢复该区域的预设值并重新回填相关表单，其他模块不会被无条件覆盖。"
        return f"执行“{text}”所述操作；动作只作用于当前配置或结果区，完成后界面会显示状态与影响范围。"
    if "gdd" in lower or "tod" in lower or "fod" in lower or lower.startswith("gd "):
        return f"设置{text}；它作为光谱相位多项式参与脉冲展宽/压缩，影响时间波形和峰值功率，但不改变光谱能量。"
    if "n₂" in text or "n2" in lower or "非线性折射率" in text:
        return f"设置{text}；它决定 Kerr 非线性相位、B-integral、SPM 展宽和等效 Kerr 透镜强度。"
    if "dn/dt" in lower:
        return f"设置{text}；局部温升乘该系数形成热致折射率变化并反馈到允许的光学传播后端。"
    if "m²" in text or "m2" in lower:
        return f"设置{text}；它控制该轴的等效衍射/发散和焦散，M²=1 表示理想基模。"
    if "波长" in text:
        return f"设置{text}；该值决定光子能量、材料截面插值、折射率/色散以及衍射尺度。"
    if "截面" in text or "σ" in text:
        return f"设置{text}；局部光子通量乘该截面得到吸收或受激发射速率，并直接影响反转和增益。"
    if "寿命" in text:
        return f"设置{text}；它决定相应能级的自发弛豫时间尺度、重复运行记忆和可能的非辐射产热。"
    if "脉宽" in text or "时间强度 fwhm" in lower or "脉冲时长" in text:
        return f"设置{text}；在总能量固定时它决定时间包络、峰值功率和 Kerr/强场效应。"
    if "功率" in text:
        return f"设置{text}；它按当前平均/峰值定义进入光子通量、吸收、布居和热负载计算。"
    if "能量" in text:
        return f"设置{text}；它归一化单次 Seed/Pump 事件，并影响增益饱和、峰值功率和平均输出。"
    if "半径" in text or "腰斑" in text or "窗口" in text:
        return f"设置{text}；它决定横向场尺度、强度归一化和边界余量，窗口过小会触发截断/FFT 边界诊断。"
    if "焦" in text or "距离" in text or "光程" in text or "起点" in text or "终点" in text:
        return f"设置{text}；它改变光路中的纵向位置或扫描范围，从而影响束宽、曲率、焦点及记录平面。"
    if "角" in text or "倾斜" in text or "na" in lower:
        return f"设置{text}；它改变光轴方向或入射几何，并参与走离、界面折射和横向采样适用性检查。"
    if "透过" in text or "损耗" in text or "吸收" in text:
        return f"设置{text}；它从相应光功率/能量账本扣除能量，并按模块合同决定是否转化为热。"
    if "导热" in text or "比热" in text or "冷却" in text or "温度" in text:
        return f"设置{text}；它进入热方程或材料工作点，影响温升、热透镜、应力和温度相关光谱参数。"
    if "密度" in text:
        return f"设置{text}；它决定单位体积的活性离子、质量或材料响应尺度，并参与体积分守恒。"
    if "nx" in lower or "ny" in lower or "nz" in lower or "采样" in text or "分箱" in text or "时间步" in text:
        return f"设置{text}；更高采样通常提高空间/时间/光谱分辨率，同时近似按相应维度增加内存和运行时间。"
    if "迭代" in text or "周期数" in text or "连续满足" in text or "子步" in text:
        return f"设置{text}；它是收敛或自适应推进的上下限，达到容差时可提前结束。"
    if "容差" in text or "残差" in text:
        return f"设置{text}；残差低于该阈值才判定相应固定点/线性求解器收敛，设得更小通常更慢。"
    if "松弛" in text:
        return f"设置{text}；较小值抑制固定点振荡但增加迭代次数，较大值收敛更快但可能不稳定。"
    if "阶" in text or "order" in lower:
        return f"设置{text}；它控制分布边缘、展开阶数或拟合复杂度，阶数越高通常需要更细采样。"
    if "折射率" in text:
        return f"设置{text}；它决定界面折射、光程、群延迟和晶体内衍射尺度。"
    if "长度" in text or "宽度" in text or "高度" in text or "直径" in text:
        return f"设置{text}；它定义材料/脉冲/几何范围，并参与体积分、边界条件和传播步长。"
    if control_kind == "text_input":
        return f"设置“{text}”的可追溯文本标识；它用于界面和结果记录，不会替代同模块的物理参数。"
    return (
        f"设置“{text}”的数值；它会写入当前展开模块并参与下一次求解。"
        "问号说明保留单位和模块边界；实际是否被上级模式锁定请同时查看配置体检。"
    )


def _principle(token: Any, fallback: str) -> str:
    return _OPTION_PRINCIPLES.get(str(token), fallback)


_ELEMENT_NAMES = {
    "free_space": "自由空间段",
    "thin_lens": "薄透镜",
    "double_pass_lens_relay": "双程双透镜回程成像",
    "thick_lens": "厚透镜",
    "mirror": "反射镜",
    "aperture": "孔径",
    "gain_medium": "增益晶体命中",
    "interface": "折射率界面",
    "tilt": "光轴倾斜",
    "coordinate_break": "坐标系转折",
    "phase_screen": "相位屏",
    "dispersive": "色散元件",
    "loss": "损耗元件",
    "observation": "观察面",
    "plane": "记录平面",
}


def _describe_optical_path(config: dict[str, Any], beam: str) -> str:
    multipass = config.get("simulation", {}).get("multipass", {})
    multipass_active = beam == "seed" and bool(
        multipass.get("enabled", False)
    )
    elements = list(config.get("optical_train", {}).get(beam, []))
    from .optics import split_gain_path

    pre, base_marker, post = split_gain_path(elements)
    if multipass_active:
        from .dispersion import material_refractive_index
        from .multipass import compile_multipass_plan, sequential_plan_components

        plan = compile_multipass_plan(
            multipass,
            base_marker,
            crystal_length_m=float(config["grid"]["length_z_m"]),
            crystal_refractive_index=material_refractive_index(
                config["material"], float(config["seed"]["wavelength_m"]), role="signal",
            ),
            global_extra_delay_s=float(
                config["simulation"].get("unmodeled_between_pass_delay_s", 0.0)
            ),
        )
        markers, segments = sequential_plan_components(plan)
        elements = list(pre)
        for index, marker in enumerate(markers):
            elements.append(marker)
            if index < len(segments):
                elements.extend(segments[index])
        elements.extend(post)
    else:
        elements = [*pre, base_marker, *post]
    steps = [f"{beam.upper()} 光源面"]
    for element in elements:
        kind = str(element.get("type", "element"))
        name = str(element.get("name", kind))
        if kind == "free_space":
            steps.append(f"自由空间 {float(element.get('length_m', 0.0)) * 1e3:.6g} mm")
        else:
            steps.append(f"{_ELEMENT_NAMES.get(kind, kind)}“{name}”")
    return " → ".join(steps)


def describe_configuration(config: dict[str, Any]) -> ModelDescription:
    """Describe the configured and effective end-to-end solver without running it."""

    physics = config.get("physics", {})
    effective = resolve_physics(config)
    signal_requested = str(config.get("simulation", {}).get("signal_model", "auto"))
    signal_model = resolve_signal_model(config)
    josa_mode = str(config.get("seed", {}).get("josa", {}).get("mode", "retarded_pc"))
    full_josa = signal_model == "josa_transport" and josa_mode == JOSA_FULL_3D_MODE
    coupling = str(config.get("multiphysics", {}).get("coupling_model", "one_way"))
    periodic = config.get("simulation", {}).get("periodic_steady_state", {})
    periodic_enabled = bool(periodic.get("enabled", False))
    periodic_method = str(periodic.get("method", "pulse_map"))
    thermal_cfg = config.get("material", {}).get("thermal", {})
    thermal_effective = resolve_thermal_model(config)

    display_names = {
        "waist_m2": "腰斑 / M² 等效光束（waist_m2）",
        "emitter_na": "发光面半径 / NA 非相干源（emitter_na）",
        "gaussian": "Gaussian",
        "super_gaussian": "超高斯（super_gaussian）",
        "radial_super_gaussian": "径向超高斯（radial_super_gaussian）",
        "four_level": "四能级（four_level）",
        "three_level": "三能级（three_level）",
        "two_level": "二能级（two_level）",
        "custom": "自定义能级网络（custom）",
        "time": "时间域（time）",
        "spectrum": "光谱域（spectrum）",
        "dual": "时间—光谱双域（dual）",
        "independent": "独立时间/光谱输入（independent）",
        "True": "开启",
        "False": "关闭",
    }

    def display(value: Any) -> str:
        return display_names.get(str(value), str(value))

    def source_for(key: str) -> str:
        if full_josa and key in {
            "optical_propagation", "pump_propagation", "population_dynamics",
            "pump_depletion", "gain_saturation", "thermal_model",
            "refractive_index_model", "nonlinear_phase_model",
            "record_optical_planes",
        }:
            return "JOSA full_3d_pc 相容性合同锁定"
        if coupling == "thermal_optical_fixed_point" and key in {
            "thermal_model", "refractive_index_model"
        }:
            return "热光固定点相容性合同锁定"
        if (
            periodic_enabled
            and periodic_method in {"pulse_map", "reduced_pulse_map"}
            and key == "gain_saturation"
        ):
            return "逐脉冲周期稳态相容性合同要求"
        if periodic_enabled and key == "population_dynamics":
            return "周期稳态方法锁定"
        return "精度预设解析" if physics.get(key, "auto") == "auto" else "用户明确选择"

    features: list[ModelFeature] = []

    def add(
        stage: str,
        feature: str,
        configured: Any,
        actual: Any,
        source: str,
        explanation: str,
    ) -> None:
        features.append(
            ModelFeature(
                stage=stage,
                feature=feature,
                configured=str(configured),
                effective=str(actual),
                source=source,
                explanation=explanation,
            )
        )

    pump = config.get("pump", {})
    pump_source = str(pump.get("source_definition", {}).get("mode", "beam_waist"))
    add("Pump 发光面", "光源定义", pump_source, pump_source, "用户明确选择", (
        "由 waist/M²或 waist/发散角构造等效束参数。"
        if pump_source in {"beam_waist", "waist_m2", "waist_divergence"}
        else "由发光面半径和 NA 建立非相干辐亮度/光线包络，再转换为所选 Pump 传播后端的初始条件。"
    ))
    add("Pump 传播", "空间传播", physics.get("pump_propagation", "auto"), effective.get("pump_propagation"), source_for("pump_propagation"), _principle(effective.get("pump_propagation"), "计算 Pump 在晶体内的三维光强与吸收路径。"))
    absorption_transport = pump.get("absorption_transport", {})
    absorption_transport_model = str(
        absorption_transport.get("model", "population_dependent")
    )
    add(
        "Pump 传播",
        "主动吸收输运",
        absorption_transport_model,
        absorption_transport_model,
        "用户/导入配置",
        (
            "用配置的固定主动吸收系数推进 Pump 衰减与吸收账本；能级激发仍由 Pump 光子流、跃迁截面和量子效率求解。"
            if absorption_transport_model == "fixed_active_coefficient"
            else "由局部能级布居和 Pump 吸收/发射截面计算净主动吸收，因此 Pump 耗尽可随饱和状态变化。"
        ),
    )
    add("Pump 传播", "泵浦耗尽", physics.get("pump_depletion", "auto"), effective.get("pump_depletion"), source_for("pump_depletion"), "沿传播方向从 Pump 功率中扣除离子主动吸收和材料被动吸收，并把净吸收送入布居与能量账本。")

    levels = config.get("energy_levels", {})
    material = config.get("material", {})
    gain_override = levels.get("gain_override_m_inv")
    add("晶体内部", "增益来源", "gain_override" if gain_override is not None else "rate_equations", "gain_override" if gain_override is not None else "rate_equations", "用户明确选择", "直接使用给定小信号增益，不由 Pump 推导。" if gain_override is not None else "由 Pump 吸收截面、受激跃迁、自发寿命和能级网络求局部布居，再由 σeNu−σaNl 得到增益。")
    add("晶体内部", "能级网络", levels.get("model", "four_level"), levels.get("model", "four_level"), "材料/表单配置", "决定 Pump 与 Signal 分别连接哪些能级，以及衰变、上转换和交叉弛豫如何改变各能级粒子数。")
    add("晶体内部", "布居动力学", physics.get("population_dynamics", "auto"), resolve_population_dynamics(config), source_for("population_dynamics"), _principle(effective.get("population_dynamics"), "求解局部离子布居。"))
    doping = material.get("doping_profile", {})
    doping_enabled = bool(doping.get("enabled", False))
    doping_mode = str(doping.get("normalization", "volume_average"))
    add(
        "晶体内部",
        "三维掺杂分布",
        doping_mode if doping_enabled else "off",
        doping_mode if doping_enabled else "off",
        "材料/表单配置",
        (
            _principle(doping_mode, "按三维轮廓建立局部离子密度。").rstrip("。")
            + "；端帽和局部浓度会同步进入布居、吸收、热与材料响应。"
            if doping_enabled
            else "关闭时在有效晶体区域使用统一离子密度，不额外创建隐藏的端帽或浓度梯度。"
        ),
    )
    if periodic_enabled:
        add("晶体内部", "Pump + Seed 周期稳态", periodic_method, periodic_method, "用户开启", _principle(periodic_method, "求重复脉冲边界固定点。"))
        if periodic_method == "pulse_map":
            pulse_initializer = str(
                periodic.get("pulse_map_initialization", "auto")
            )
            add(
                "晶体内部",
                "离散周期映射初值",
                pulse_initializer,
                pulse_initializer,
                "用户选择的初始化层",
                _principle(
                    pulse_initializer,
                    "只建立完整主脉冲固定点的数值初值。",
                ),
            )
        if str(periodic.get("acceleration", {}).get("method", "off")) == "anderson":
            add("晶体内部", "周期固定点加速", "anderson", "anderson（带保护）", "用户开启", _OPTION_PRINCIPLES["anderson"])

    seed = config.get("seed", {})
    pulse_domain = str(seed.get("pulse_definition", {}).get("domain", "independent"))
    add("Seed 发光面", "脉冲定义", pulse_domain, pulse_domain, "用户明确选择", "由时间强度、光谱强度和可选光谱相位建立归一化输入脉冲；dual 模式会检查 Fourier 一致性。")
    add("Seed 放大", "增益输运", signal_requested, signal_model, "auto 输入解析" if signal_requested == "auto" else "用户明确选择", _principle(signal_model, "传播 Seed 并应用局部增益。"))
    if signal_model == "josa_transport":
        add("Seed 放大", "JOSA 层级", josa_mode, josa_mode, "用户明确选择", "所有 JOSA 模式都按有序时间×波长箱更新横向局域通量和真实布居；所选模式改变积分/时间坐标，full_3d_pc 另锁定 Pump/Seed FFT、三维热光与近似群延迟走离。")
    add("Seed 传播", "空间传播", physics.get("optical_propagation", "auto"), effective.get("optical_propagation"), source_for("optical_propagation"), _principle(effective.get("optical_propagation"), "传播 Seed 横向场或束参数。"))
    spatial_gain = str(seed.get("spatial_gain_transport", {}).get("model", "local_field"))
    add("Seed 放大", "空间增益闭合", spatial_gain, spatial_gain, "独立模块", _principle(spatial_gain, "把三维反转映射到 Seed。"))
    add("Seed 放大", "增益饱和", physics.get("gain_saturation", "auto"), effective.get("gain_saturation"), source_for("gain_saturation"), "开启时 Seed 的局部受激跃迁会反向扣减布居并限制输出；关闭时保持小信号增益。")
    polarization = seed.get("polarization", {})
    polarization_model = str(polarization.get("model", "scalar"))
    birefringence_model = str(
        polarization.get("birefringence_model", "off")
    )
    add(
        "Seed 传播",
        "偏振传播",
        polarization_model,
        polarization_model,
        "用户明确选择",
        (
            "传播 Ex/Ey Jones 复分量，并用两分量总强度共同驱动饱和。"
            if polarization_model == "jones"
            else "使用标量场；不创建未配置的经验性退偏或偏振损耗。"
        ),
    )
    add(
        "Seed 传播",
        "晶体双折射",
        birefringence_model,
        birefringence_model,
        "用户明确选择",
        _principle(
            birefringence_model,
            "决定是否把应力/光弹响应作为 Jones 延迟器作用于 Seed。",
        ),
    )
    multipass = config.get("simulation", {}).get("multipass", {})
    multipass_active = bool(multipass.get("enabled", False))
    if multipass_active:
        passes = int(multipass.get("pass_count", 2))
        template = str(multipass.get("template", "double_pass"))
        execution = (
            "causal" if uses_causal_clock(config["simulation"], seed, signal_model)
            else "sequential"
        )
        effective_pass_mode = f"multipass:{template}:{execution}"
        pass_source = "simulation.multipass 编译模板"
    else:
        passes = 1
        effective_pass_mode = "single_pass"
        pass_source = "单程默认"
    add(
        "光学平台",
        "多程组织",
        "multipass" if multipass_active else "single_pass",
        f"{effective_pass_mode} / {passes} 次晶体命中",
        pass_source,
        "按每次 gain_medium 命中、传播方向、界面和程间透过率依次传递场与能量账本。",
    )
    add("光学平台", "晶体界面", physics.get("interface_model", "auto"), effective.get("interface_model"), source_for("interface_model"), _principle(effective.get("interface_model"), "计算入射与出射界面透过。"))

    heat_ledger = str(thermal_cfg.get("heat_source_model", "pump_component_ledger"))
    heat_fraction = thermal_cfg.get("heat_fraction", "quantum_defect")
    seed_heat = str(
        config.get("multiphysics", {}).get("seed_heat_coupling", "off")
    )
    add("热源", "能量账本", heat_ledger, heat_ledger, "用户明确选择", _principle(heat_ledger, "把不可逆能量交换转换为热源。"))
    if heat_ledger == "event_energy_ledger":
        add(
            "热源",
            "主动 Pump 产热",
            heat_fraction,
            "event_resolved_partitions",
            "逐事件账本接管",
            "按每条吸收、受激发射、衰变和非线性能级事件的能量分配产热；"
            "保留的快速 heat_fraction 字段本次不参与计算。",
        )
    else:
        add("热源", "主动 Pump 产热", heat_fraction, heat_fraction, "用户/材料配置", _principle(heat_fraction, "把主动吸收功率转换为热。"))
    add(
        "热源",
        "Seed 不可逆损耗产热耦合",
        seed_heat,
        seed_heat,
        "用户明确选择",
        _principle(
            seed_heat,
            "决定 Seed 被动/强场不可逆损耗是否只后处理一次或进入热光固定点。",
        ),
    )
    add("热与光反馈", "热场", physics.get("thermal_model", "auto"), thermal_effective, source_for("thermal_model"), _principle(thermal_effective, "求解温度场。"))
    if thermal_effective == "steady_3d":
        thermal_solver = str(thermal_cfg.get("linear_solver", "auto"))
        add("热与光反馈", "稳态热线性求解", thermal_solver, thermal_solver, "用户选择/自动后端", _principle(thermal_solver, "求解离散导热线性系统。"))
    response_mode = str(material.get("spectroscopic_response", {}).get("mode", "off"))
    add("热与光反馈", "寿命/截面反馈", response_mode, response_mode, "用户明确选择", _principle(response_mode, "决定温度和掺杂是否修改材料光谱参数。"))
    add("热与光反馈", "多物理耦合", coupling, coupling, "用户明确选择", _principle(coupling, "组织 Pump、布居、热和 Seed 的计算顺序。"))
    refraction = effective.get("refractive_index_model", "constant")
    add("热与光反馈", "折射率", physics.get("refractive_index_model", "auto"), refraction, source_for("refractive_index_model"), _principle(refraction, "把材料状态转换为光学相位。"))

    ase_model = str(config.get("ase", {}).get("model", "off"))
    add("寄生过程", "ASE / TPL", ase_model, ase_model, "用户明确选择", _principle(ase_model, "计算自发辐射放大及寄生振荡风险。"))
    stress_model = str(
        material.get("mechanical", {}).get("stress_model", "none")
    )
    add("寄生过程", "热应力", stress_model, stress_model, "用户明确选择", "由温度场求热弹性应力、光弹双折射和可选端面位移；关闭时不产生应力光程差。")
    nonlinear_phase = effective.get("nonlinear_phase_model", "off")
    nonlinear_loss = effective.get("nonlinear_loss_model", "off")
    add("强场过程", "非线性相位", physics.get("nonlinear_phase_model", "auto"), nonlinear_phase, source_for("nonlinear_phase_model"), _principle(nonlinear_phase, "计算可逆强度相关相位。"))
    add("强场过程", "不可逆强场损耗", physics.get("nonlinear_loss_model", "auto"), nonlinear_loss, source_for("nonlinear_loss_model"), _principle(nonlinear_loss, "计算多光子或载流子损耗。"))
    strong_field = material.get("strong_field", {})
    if nonlinear_loss != "off":
        mechanism = str(strong_field.get("mechanism", "unspecified"))
        multiphoton_order = int(strong_field.get("multiphoton_order", 2))
        carrier_model = str(
            strong_field.get("free_carrier", {}).get("model", "off")
        )
        heat_deposit_fraction = float(
            strong_field.get("deposit_fraction_as_heat", 1.0)
        )
        add(
            "强场过程",
            "不可逆损耗机制",
            mechanism,
            mechanism,
            "材料强场配置",
            "机制标签决定参数的物理解读与严格验证要求；它不由 K 或 β 自动猜测。",
        )
        add(
            "强场过程",
            "多光子阶数",
            multiphoton_order,
            multiphoton_order,
            "材料强场配置",
            "沿光束真实路径按 dI/ds=-βK·I^K 推进；βK 的量纲随 K 改变，不能跨阶直接复用。",
        )
        add(
            "强场过程",
            "自由载流子状态",
            carrier_model,
            carrier_model,
            "材料强场配置",
            _principle(
                carrier_model,
                "决定是否推进跨脉冲载流子产生、复合、吸收与折射率响应。",
            ),
        )
        add(
            "热源",
            "强场损耗直接热化比例",
            heat_deposit_fraction,
            heat_deposit_fraction,
            "材料强场经验闭合",
            "该比例把记录到的多光子与自由载流子光学损耗直接送入热源；"
            "未送入热源的部分没有独立的载流子内部能量状态，因此这是降阶热闭合，"
            "不是完整的载流子能量输运模型。",
        )
    if signal_model == "commercial_app_fast":
        commercial_app_config = seed.get("commercial_app_fast", {})
        commercial_app_temporal = str(
            commercial_app_config.get("temporal_model", "fixed_duration")
        )
        commercial_app_intrapulse_relaxation = bool(
            commercial_app_config.get("include_relaxation_during_pulse", False)
        )
        commercial_app_is_cw = str(seed.get("temporal", {}).get("type", "gaussian")) == "cw"
        commercial_app_temporal_effective = (
            "cw_average_intensity" if commercial_app_is_cw
            else "time_binned_intrapulse_population_relaxation" if commercial_app_intrapulse_relaxation
            else "one_dimensional_effective_area_strong_field"
            if nonlinear_loss == "strong_field_effective_area" and commercial_app_temporal == "effective_area_spm"
            else "one_dimensional_effective_area_spm" if commercial_app_temporal == "effective_area_spm"
            else "fixed_input_effective_duration"
        )
        commercial_app_quality = str(
            commercial_app_config.get("beam_quality_model", "configured_m2")
        )
        add(
            "强场过程",
            "Commercial App 时间闭合",
            commercial_app_temporal,
            commercial_app_temporal_effective,
            "CW 合同" if commercial_app_is_cw else "Seed Commercial App 配置",
            (
                "CW 直接使用传播中的平均强度闭合 Kerr 透镜；不人为构造脉冲，"
                "因此不定义 SPM 展宽、脉宽或压缩器输出。"
                if commercial_app_is_cw
                else (
                    "把脉冲能量分成有序增益时间箱，在相邻箱之间推进完整能级暗弛豫；"
                    "所有时间箱和 Gaussian-q 横向通道共享局部布居，不解析延迟时间输运。"
                    if commercial_app_intrapulse_relaxation
                    else (
                        "沿一维相干时间/频率包络用当前 Gaussian 模有效面积计算 SPM、材料色散"
                        + (
                            "、Gaussian K 阶矩强场损耗及标量载流子响应；"
                            if nonlinear_loss == "strong_field_effective_area"
                            else "；"
                        )
                        + "增益与布居仍使用一个整脉冲 FN 包。"
                        if commercial_app_temporal == "effective_area_spm"
                        else "使用输入能量/峰值功率得到的固定有效脉宽，只闭合 Kerr 透镜。"
                    )
                )
            ),
        )
        add(
            "输出面",
            "Commercial App M² 合同",
            commercial_app_quality,
            commercial_app_quality,
            "Seed Commercial App 配置",
            (
                "累计每次 Gaussian 投影前的超额角向二阶矩，作为残余像差风险代理；"
                "不冒充完整复场 ISO 测量。"
                if commercial_app_quality == "projection_residual"
                else "报告设定的 Gaussian 运输 M²，不推断被投影掉的高阶像差。"
            ),
        )

    grid = config.get("grid", {})
    spacing = str(grid.get("spacing_xy", "uniform"))
    optical_grid = grid.get("optical", {})
    pump_grid = grid.get("pump_optical", {})
    add("数值离散", "物理网格", spacing, f"{grid.get('nx')}×{grid.get('ny')}×{grid.get('nz')} / {spacing}", "用户明确选择", _principle(spacing, "承载三维布居、热源、温度和应力。"))
    physical_center_x = float(grid.get("center_x_m", 0.0))
    physical_center_y = float(grid.get("center_y_m", 0.0))
    refinement_x_raw = grid.get("refinement_center_x_m")
    refinement_y_raw = grid.get("refinement_center_y_m")
    refinement_x = (
        physical_center_x
        if refinement_x_raw is None
        else float(refinement_x_raw)
    )
    refinement_y = (
        physical_center_y
        if refinement_y_raw is None
        else float(refinement_y_raw)
    )
    refinement_effective = f"x={refinement_x:.6g} m, y={refinement_y:.6g} m"
    refinement_stage_note = (
        f"有效加密中心为 `{refinement_effective}`。"
        if spacing in {"center_stretched", "refined_region"}
        else ""
    )
    if spacing in {"center_stretched", "refined_region"}:
        refinement_configured = (
            "跟随物理网格中心"
            if refinement_x_raw is None and refinement_y_raw is None
            else "独立指定（未填写轴仍跟随物理网格中心）"
        )
        add(
            "数值离散",
            "物理网格加密中心",
            refinement_configured,
            refinement_effective,
            "用户明确选择",
            "只重分配固定物理边界内的 x/y 单元；不会移动晶体、Pump、Seed 或独立光学窗口。",
        )
    if spacing == "refined_region":
        region = grid.get("refined_region", {})
        region_summary = (
            f"{float(region.get('size_x_m', 200e-6)) * 1e6:.6g} × "
            f"{float(region.get('size_y_m', 200e-6)) * 1e6:.6g} μm; "
            f"{region.get('nx', 24)} × {region.get('ny', 24)} cells"
        )
        add("数值离散", "物理细网格区域", region_summary, region_summary,
            "用户明确选择", _OPTION_PRINCIPLES["refined_region"])
    optical_effective = "independent uniform" if bool(optical_grid.get("enabled", False)) else "physical grid"
    add("数值离散", "Seed 光学网格", optical_effective, optical_effective, "用户明确选择", "FFT/BPM 要求均匀横向采样；独立网格与物理网格之间使用功率/能量守恒映射。")
    pump_grid_mode = str(pump_grid.get("mode", "shared"))
    pump_grid_explanation = {
        "shared": "Pump 复用 Seed 的独立均匀光学网格。",
        "physical": "Pump 直接使用物理网格；只适用于不要求均匀 FFT 采样的传播后端。",
        "custom": "Pump 使用自己独立的均匀横向窗口和采样数，并与物理网格守恒交换功率。",
    }.get(pump_grid_mode, "决定 Pump 是否复用 Seed 光学采样。")
    add(
        "数值离散",
        "Pump 光学网格",
        pump_grid_mode,
        pump_grid_mode,
        "用户明确选择",
        pump_grid_explanation,
    )
    add(
        "数值离散",
        "光学面记录",
        physics.get("record_optical_planes", True),
        effective.get("record_optical_planes", True),
        source_for("record_optical_planes"),
        "只控制诊断平面保存和可追溯性，不改变场传播；关闭后不能事后重建未保存平面的 M²/光斑。",
    )
    fft_backend = str(config.get("performance", {}).get("fft_backend", "auto"))
    add(
        "数值离散",
        "FFT 执行后端",
        fft_backend,
        fft_backend,
        "性能配置",
        _principle(fft_backend, "选择等价归一化的 FFT 实现，不改变物理方程。"),
    )
    validation_mode = str(config.get("simulation", {}).get("validation_mode", "exploratory"))
    add("输出面", "结果有效性", validation_mode, validation_mode, "用户明确选择", _principle(validation_mode, "检查守恒、收敛、边界占用和适用范围。"))

    notices: list[str] = []
    locked = [item for item in features if "锁定" in item.source]
    if locked:
        notices.append("当前被上级模型锁定的功能：" + "、".join(f"{item.feature}={item.effective}" for item in locked) + "。这些不是隐藏覆盖。")
    if seed_heat == "fixed_point" and coupling != "thermal_optical_fixed_point":
        notices.append("冲突：Seed 强场产热 fixed_point 需要 thermal_optical_fixed_point。")
    if response_mode == "local_3d" and coupling != "thermal_optical_fixed_point":
        notices.append("冲突：局部三维光谱反馈需要热光固定点，不会静默退回统一工作温度。")
    if spatial_gain == "gaussian_mode_overlap" and signal_model not in {
        "fluence",
        "commercial_app_fast",
    }:
        notices.append("冲突：Gaussian 模态重叠闭合只定义于单色 fluence 或 Commercial App 模态通道。")
    nonlinear_phase_compatible = (
        nonlinear_phase == "off"
        or (
            nonlinear_phase == "kerr_time_domain"
            and signal_model == "coherent_cpa"
        )
        or (
            nonlinear_phase == "kerr_parabolic_lens"
            and signal_model == "commercial_app_fast"
        )
    )
    if not nonlinear_phase_compatible:
        notices.append("冲突：Kerr 时间域相位只属于 coherent_cpa，等效二次 Kerr 透镜只属于 commercial_app_fast。")
    nonlinear_loss_compatible = bool(
        nonlinear_loss == "off"
        or (
            nonlinear_loss == "strong_field_time_domain"
            and signal_model == "coherent_cpa"
        )
        or (
            nonlinear_loss == "strong_field_effective_area"
            and signal_model == "commercial_app_fast"
        )
    )
    if not nonlinear_loss_compatible:
        notices.append("冲突：完整时间域强场损耗属于 coherent_cpa；effective-area 强场矩闭合属于 commercial_app_fast。")
    if (
        heat_ledger == "pump_component_ledger"
        and heat_fraction == "lifetime_quantum_defect"
        and levels.get("upper_state_radiative_lifetime_s") is None
    ):
        notices.append("冲突：寿命分解产热需要独立的上能级辐射寿命 τrad。")
    if thermal_effective == "off" and refraction == "thermal_3d":
        notices.append(
            "当前热场关闭，因此 thermal_3d 折射率端口没有新的温度场可反馈；"
            "它不会在后台偷偷求热场，实际传播使用冷态/零热扰动。"
        )

    pump_profile = str(pump.get("profile", "gaussian"))
    pump_path = _describe_optical_path(config, "pump")
    seed_path = _describe_optical_path(config, "seed")
    pump_wavelength_nm = float(pump.get("wavelength_m", 0.0)) * 1e9
    seed_wavelength_nm = float(seed.get("wavelength_m", 0.0)) * 1e9
    quantum_efficiency = float(levels.get("pump_quantum_efficiency", 1.0))
    lifetime_s = float(levels.get("upper_state_lifetime_s", 0.0))
    radiative_lifetime = levels.get("upper_state_radiative_lifetime_s")
    radiative_text = (
        "未配置独立 τrad"
        if radiative_lifetime is None
        else f"τrad={float(radiative_lifetime):.6g} s"
    )

    extra_heat = float(thermal_cfg.get("extra_heat_fraction", 0.0))
    stokes_multiplier = float(thermal_cfg.get("stokes_shift_multiplier", 1.0))
    if heat_ledger == "event_energy_ledger":
        heat_algorithm = (
            "逐事件账本直接使用各跃迁/衰变/非线性通道的热、荧光、ASE 与未解析分配；"
            "旁边保留的快速主动产热模型本次被明确旁路，不会重复乘 heat_fraction。"
        )
        heat_relations = (
            "逐事件能量差 ΔEevent 按配置分配到 heat / fluorescence / ASE / unresolved",
            "各事件分配和必须闭合为 1，材料被动吸收仍全部转热",
        )
    elif heat_fraction == "lifetime_quantum_defect":
        heat_algorithm = (
            "先用量子亏损处理吸收事件，再从实际上能级布居按 "
            "Nupper hν(1/τtotal−1/τrad) 添加非辐射衰变热；受激提取降低 Nupper 后，"
            "这部分热会自动下降，因此不会把经验荧光效率重复乘两次。"
        )
        heat_relations = (
            "主动吸收即时热 Qabs = Pabs,active · max[0, 1 − ηq λp/λref]",
            "寿命非辐射热 Qnr = Nupper hc/λref · max(1/τtotal − 1/τrad, 0)",
            "lifetime 模式要求 extra_heat_fraction=0，避免重复计热",
        )
    elif heat_fraction == "quantum_defect":
        heat_algorithm = "主动吸收按含 Pump 激发量子效率的能量守恒量子亏损转热。"
        heat_relations = (
            f"Qactive = Pabs,active · (max[0, 1 − ηq λp/λref] + {extra_heat:.6g})",
        )
    elif heat_fraction == "scaled_quantum_defect":
        heat_algorithm = "主动吸收按独立实测/标定倍率缩放 Stokes 量子亏损；不再乘 Pump 激发效率。"
        heat_relations = (
            f"Qactive = Pabs,active · ({stokes_multiplier:.6g}·max[0, 1 − λp/λref] + {extra_heat:.6g})",
        )
    elif heat_fraction in {"spectral_quantum_defect", "spectral_scaled_quantum_defect"}:
        heat_algorithm = (
            "每个 Pump 波长先按其局部正净吸收加权，再应用相应量子亏损；"
            "因此宽带 Pump 的空间谱重分布会改变局部热比例。"
        )
        spectral_factor = (
            "ηq"
            if heat_fraction == "spectral_quantum_defect"
            else f"独立倍率 {stokes_multiplier:.6g}"
        )
        heat_relations = (
            f"Qactive(r) = Σλ Pabs,active(λ,r) · fheat(λ)，fheat 使用 {spectral_factor}",
        )
    else:
        try:
            explicit_fraction = float(heat_fraction)
        except (TypeError, ValueError):
            heat_algorithm = (
                f"当前主动产热值 `{heat_fraction}` 不是已知模型或数值比例；"
                "说明保留该无效输入，配置验证会阻止运行。"
            )
            heat_relations = ("无有效产热关系：必须先修正主动 Pump 产热模型",)
        else:
            heat_algorithm = "主动吸收直接乘配置/实测产热比例，不从波长或寿命外推。"
            heat_relations = (
                f"Qactive = Pabs,active · ({explicit_fraction:.6g} + {extra_heat:.6g})",
            )

    if thermal_effective == "off":
        coupling_algorithm = (
            "本次热场关闭：热源仍可进入能量账本，但不求温度 PDE，ΔT 与热致 Δn 均取零；"
            "因此 one_way/固定点热反馈端口本次不参与传播。"
        )
        thermal_relations = (
            "thermal_model=off：不求导热 PDE，ΔT=0，Δnthermal=0",
        )
    elif thermal_effective == "transient_3d":
        coupling_algorithm = (
            "按时间步推进热扩散；one_way 只把一次热历史传给最终 Seed，"
            "thermal_optical_fixed_point 则在每轮更新 Pump→布居→热源→温度→Δn→Pump。"
        )
        thermal_relations = (
            "ρcp ∂T/∂t = ∇·(k∇T) + Q",
        )
    else:
        coupling_algorithm = (
            "求三维稳态导热；one_way 只把一次温度/Δn 传给最终 Seed，"
            "thermal_optical_fixed_point 则反复更新 Pump→布居→热源→温度→Δn→Pump。"
        )
        thermal_relations = (
            "稳态导热：∇·(k∇T) + Q = 0",
        )
    if thermal_effective != "off":
        thermal_relations += (
            "慢变量折射率反馈 Δnslow = (dn/dT)ΔT + Δnstress（只包含已开启且相容的项）",
            (
                "固定点检查 Pump 吸收、布居和热折射率残差"
                if coupling == "thermal_optical_fixed_point"
                else "one_way 不迭代回写 Pump；结果说明不会把它标成自洽固定点"
            ),
        )
        if nonlinear_phase != "off":
            thermal_relations += (
                "瞬时 Kerr 由独立传播算符沿真实光束路径累积 φK=k0∫n2 I ds；它不属于热场保存的 Δnslow",
            )

    causal_multipass_active = uses_causal_clock(
        config.get("simulation", {}), seed, signal_model
    )
    signal_algorithm_key = (
        "spectral_time_causal_multipass"
        if signal_model == "spectral_time" and causal_multipass_active
        else signal_model
    )
    signal_algorithm = {
        "commercial_app_fast": (
            (
                "把 HG 模态展开为独立 Gaussian-q/能量通道；同一波长的模态在每个时间与 z "
                "步先汇总局部通量，只对共享布居提交一次受激跃迁，再按局部输入比例分配输出。"
                "热相位、Kerr 相位与增益导引仍投影回各模态自己的二阶矩。"
                if str(seed.get("spatial_modes", {}).get("model", "effective_m2"))
                == "incoherent_hg"
                else
                "把脉冲分成有序时间×波长通量箱，前沿、峰值和尾沿依次提交布居交换，并在"
                "时间箱之间推进能级暗弛豫；所有时间箱共享 Gaussian 二阶矩并解析推进 q。"
                if bool(
                    seed.get("commercial_app_fast", {}).get(
                        "include_relaxation_during_pulse", False
                    )
                )
                else
                (
                    "用一维相干时间/频率包络与当前 Gaussian 有效面积推进材料色散、SPM"
                    + (
                        "、Gaussian K 阶矩强场损耗及标量载流子响应；"
                        if nonlinear_loss == "strong_field_effective_area"
                        else "；"
                    )
                    + "每个晶体层的局部光谱增益和布居交换仍由整脉冲 FN 包闭合，"
                    "横向热相位、Kerr 透镜与增益导引投影回 Gaussian 二阶矩并解析推进 q。"
                    if str(
                        seed.get("commercial_app_fast", {}).get(
                            "temporal_model", "fixed_duration"
                        )
                    )
                    == "effective_area_spm"
                    else
                    "把整脉冲折叠为一个通量包但保留波长分箱；每个晶体层仍计算局部光谱增益和"
                    "布居交换，随后把热相位、Kerr 相位与增益导引投影回 Gaussian 二阶矩并解析推进 q。"
                    "输入展宽后的有效脉宽固定，因此不预测输出 SPM 光谱、脉宽或压缩峰值。"
                )
            )
        ),
        "fluence": (
            "把整脉冲压缩为横向通量；饱和开启时在每个局部单元使用 Frantz–Nodvik "
            "输入—输出关系，因此能描述能量提取饱和，但不能产生脉内前沿/尾沿差异。"
        ),
        "spectral_time": (
            "把脉冲能量分配到时间×波长箱，每个箱使用相应波长截面和局部 FN 增益，"
            "按传播顺序提交布居耗尽；保留时域包络和光谱强度，不传播频率间复相位。"
        ),
        "spectral_time_causal_multipass": (
            "先把双程、双侧镜、Herriott、4f 中继或论文逐程条件编译成晶体命中序列，"
            "再在同一个绝对时间 CFL 网格上推进每一程。重叠由脉宽、晶体群速度和程间"
            "光程自然产生；同一时间真正落在同一三维网格单元的各程通量共同更新局部布居。"
        ),
        "josa_transport": (
            "在有序时间×波长箱中联立光子通量输运与能级速率方程；front、peak、tail 依次看见"
            "已经被前一部分改变的布居，横向场仍可做采样衍射。源/展宽器相位只可先投影成"
            "输入时间强度；传播中不保留频率间相干相位，因而不预测 Kerr/SPM 或压缩器输出。"
        ),
        "coherent_cpa": (
            "传播复光谱场 E(x,y,ω)，按 exp(−iωt) 约定经 FFT 得到瞬时时域场；增益、色散、Kerr 相位和"
            "不可逆强场损耗按分步算符组合。增益使用实振幅放大与独立材料色散，未计算随布居变化的共振增益相位；"
            "瞬态增益软窗的能量收敛也不等于波形收敛，压缩质量需独立加密检验。"
        ),
    }.get(signal_algorithm_key, "按当前求解后端传播 Seed 并应用局部受激跃迁。")

    if signal_model == "commercial_app_fast" and commercial_app_is_cw:
        signal_algorithm = (
            "使用连续光平均强度推进光谱增益和 Gaussian-q 二阶矩，Kerr 采用平均强度的等效透镜；"
            "不构造脉冲时间包络，也不报告 SPM 脉冲展宽或压缩器输出。"
        )

    coherent_cfg = seed.get("coherent", {})
    radial_wave = (
        signal_model == "coherent_cpa"
        and coherent_cfg.get("fidelity_mode") in {"axisymmetric_kerr", "axisymmetric_transient_gain"}
    ) or (
        signal_model == "spectral_time"
        and seed.get("spectral_time", {}).get("spatial_backend") == "axisymmetric"
    )
    polar_commercial_app = signal_model == "commercial_app_fast" and seed.get("commercial_app_fast", {}).get("spatial_backend") == "axisymmetric"
    radial_cfg = coherent_cfg.get("axisymmetric", {})
    spatial_algorithm_note = ""
    if radial_wave:
        spatial_algorithm_note = (
            f"晶体内使用径向 Crank–Nicolson 传播，径向 {radial_cfg.get('radial_bins', 24)} 格，"
            f"角向闭合为 `{radial_cfg.get('azimuthal_fidelity', 'm0')}`；晶体外光路仍使用 Cartesian 表示。"
            "未保留角向阶数的误差由投影、材料场残差和边界占用共同检查。"
        )
        if signal_model == "coherent_cpa":
            signal_algorithm = (
                "在径向 / 有限角向网格上传播复光谱场 E(r,θ,ω)，保留相干 FFT、材料色散和已开启的局域 Kerr / 强场损耗。"
                + (
                    "增益按 front→peak→tail 软窗依次耗尽布居。"
                    if coherent_cfg.get("transient_gain", {}).get("enabled", False)
                    else "增益使用整脉冲 FN 闭合，不解析前沿与尾沿之间的增益耗尽差异。"
                )
            )
    elif polar_commercial_app:
        spatial_algorithm_note = (
            f"晶体内在 {radial_cfg.get('radial_bins', 24)} 个径向格点和 {radial_cfg.get('angular_samples', 12)} 个角向配点上积分；"
            "保留全部角向配点，传播仍为 Gaussian-q 二阶矩闭合，不使用相干径向衍射或有限 m 阶截断。"
        )
    if signal_model == "coherent_cpa":
        local = coherent_cfg.get("local_operator", {})
        if local.get("spatial_population_update") == "independent_groups":
            signal_algorithm += (
                "空间增益小片从父片的脉冲前布居独立开始，按各自厚度提取后在物理片末按体积平均回写；"
                "每个小片内的频率通道仍竞争同一局部布居。"
            )
        cap = local.get("multirate_gain", {}).get("maximum_updates_per_cell")
        if cap is not None:
            signal_algorithm += (
                f"每物理片最多调用 {int(cap)} 次增益；此次数上限替代自动合并阈值，光学子步仍独立推进。"
            )
    if coupling == "thermal_optical_fixed_point":
        coupling_algorithm += (
            "本次 joint 每轮推进指定周期数并更新热场；最终固定热场完成内层周期收敛，再用一次新热解验证闭合。"
            if config.get("multiphysics", {}).get("iteration_scheme", "nested") == "joint"
            else "本次 joint_frozen 在内外残差同时连续达标后固定热场，要求指定次数的脉冲验证全部通过；最终不重算热闭合，失败则继续交替迭代。"
            if config.get("multiphysics", {}).get("iteration_scheme") == "joint_frozen"
            else "本次 nested 在每轮热场更新前完成内层布居 / 周期求解；最终还要验证一次热闭合。"
        )

    if gain_override is None:
        gain_origin_text = (
            "最终每个体素、每个波长的净信号增益由真实上下能级布居计算，"
            "不是用平均反转代替。"
        )
        gain_relation = (
            "信号净增益 g(λ,r) = Σ[σe(λ,r) Nupper(r) − "
            "σa(λ,r) Nlower(r)] − αpassive"
        )
    else:
        gain_origin_text = (
            f"本次明确使用 gain_override_m_inv={float(gain_override):.6g} m⁻¹；"
            "Pump/布居仍可用于吸收和诊断，但不会被伪装成该 Signal 增益的来源。"
        )
        gain_relation = (
            f"Signal 小信号增益直接取 g={float(gain_override):.6g} m⁻¹；"
            "不由 σeNupper−σaNlower 反推"
        )

    active_heat_model = (
        "event_resolved_partitions"
        if heat_ledger == "event_energy_ledger"
        else str(heat_fraction)
    )
    gain_saturation_enabled = bool(effective.get("gain_saturation", False))
    if gain_saturation_enabled:
        spatial_gain_effect = (
            "因此可以产生增益导引和由 Seed 耗尽形成的空间孔烧蚀"
        )
        gain_transport_relations = (
            "小信号局部传播：Iout = Iin exp(g Δs)",
            (
                "联立光子通量输运 (∂z + vg⁻¹∂t)Φ = g(N)Φ 与能级速率方程"
                if signal_model == "josa_transport"
                else "FN 饱和：Fout = Fsat ln{1 + exp(g0Δs)[exp(Fin/Fsat) − 1]}"
            ),
            (
                "受激提取的光子数反向提交到上下能级布居，避免下一程重复使用同一份反转"
                if gain_override is None
                else "手动增益使用局部 Fsat·g 能量密度储库；每次提取减少该储库，不冒充由物理能级产生的反转"
            ),
        )
    else:
        spatial_gain_effect = (
            "因此仍可产生增益导引；但本次关闭饱和，不产生 Seed 耗尽导致的空间孔烧蚀"
        )
        gain_transport_relations = (
            "小信号局部传播：Iout = Iin exp(g Δs)",
            "gain_saturation=false：不应用 FN 饱和，也不把 Seed 受激提取反向提交到布居",
        )

    stages = (
        ModelStage(
            "Pump 与 Seed 输入面：从功率/能量变成规范化光场",
            (
                f"Pump 使用 **{display(pump_source)}** 和 **{display(pump_profile)}**，中心波长为 "
                f"{pump_wavelength_nm:.6g} nm；程序把横向分布归一化到输入功率，并用光子能量 "
                f"hνp 把强度换成光子通量。Seed 中心波长为 {seed_wavelength_nm:.6g} nm，"
                f"由 **{display(pulse_domain)}** 输入建立单脉冲能量、时间强度、光谱强度和可选相位。"
                "Pump 与 Seed 有各自的源面和传播路径，只在同一块增益晶体坐标中耦合。"
            ),
            (
                "Pump 光子能量 Ep = hc/λp，局部光子通量 Φp = Ip/Ep",
                "Seed 的时间/光谱权重分别归一化，使积分严格等于配置的单脉冲能量",
            ),
        ),
        ModelStage(
            "光学电路：把两个源面传播到每一次晶体命中",
            (
                f"Pump 实际光路为：**{pump_path}**。其空间后端是 `{effective.get('pump_propagation')}`；"
                f"Seed 实际光路为：**{seed_path}**，空间后端是 `{effective.get('optical_propagation')}`。"
                f"晶体界面使用 `{effective.get('interface_model')}`，Seed 采用 **{display(effective_pass_mode)}**，"
                f"总计 {passes} 次晶体命中。ABCD 只传递 q 参数/二阶矩，FFT 传播复场并承担衍射和孔径，"
                + (
                    spatial_algorithm_note
                    if radial_wave or polar_commercial_app
                    else "hybrid 则在晶体外使用 ABCD、晶体内使用 FFT；"
                )
                + "电路中的元件、自由空间和增益命中按实际顺序作用，不把多程简单折算成一段等效增益。"
            ),
            (
                "ABCD：qout = (A qin + B)/(C qin + D)",
                "FFT/BPM：在均匀横向网格上交替施加衍射传播和局部相位/增益算符",
            ),
        ),
        ModelStage(
            "Pump 吸收、能级布居与局部增益：晶体中的核心耦合",
            (
                f"晶体使用 **{material.get('name', 'material')}** 和 **{display(levels.get('model', 'four_level'))}**。"
                + (
                    "光学 Pump 使用配置的固定主动吸收系数，与实时布居计算分开；"
                    if absorption_transport_model == "fixed_active_coefficient"
                    else "光学 Pump 的净吸收系数由下能级吸收减去上能级受激发射得到；"
                )
                + "开启耗尽时，该吸收直接扣减下游 Pump。激发效率 ηq 只乘入产生有效能级跃迁的吸收速率，而不伪造 Pump"
                f"已经被吸收的光学能量；当前 ηq={quantum_efficiency:.6g}。布居采用 "
                f"`{resolve_population_dynamics(config)}`，总上能级寿命 τtotal={lifetime_s:.6g} s，{radiative_text}。"
                + (f"重复运行由 `{periodic_method}` 求周期边界固定点。" if periodic_enabled else "当前只按设定周期数运行，不额外求 Pump+Seed 周期固定点。")
                + f" {gain_origin_text}"
            ),
            (
                "Pump 有效激发率 Rpa = ηq σpa Φp Nlower；Pump 光学耗尽仍按原始净吸收功率守恒",
                "能级方程 dNi/dt = 光致跃迁 + 自发衰变 + 上转换/交叉弛豫等自定义过程",
                gain_relation,
            ),
        ),
        ModelStage(
            "Seed 在增益介质中的提取、饱和和多程传播",
            (
                f"当前 Seed 增益输运为 `{signal_model}`：{signal_algorithm} 横向增益闭合为 `{spatial_gain}`；"
                f"增益饱和为 **{display(effective.get('gain_saturation'))}**。`local_field` 会让每个横向单元"
                f"使用自己的 g(x,y,z)，{spatial_gain_effect}；Gaussian 模态重叠则只把"
                "三维反转投影成每个 z 层的标量模态增益。每次晶体命中后的场和布居都会成为下一程输入，"
                "界面、孔径和程间透过率均进入独立能量账本。"
            ),
            gain_transport_relations,
        ),
        ModelStage(
            "能量去向与热源：只把不可逆部分送入导热方程",
            (
                f"不可逆能量采用 `{heat_ledger}`，本次实际主动 Pump 产热为 `{active_heat_model}`。材料被动 Pump"
                "吸收全部转热；Seed 主动受激提取属于有用输出而不是热，只有明确启用的 Seed 被动吸收、"
                "多光子吸收、自由载流子吸收和 ASE 被动损耗才进入相应热源。"
                + heat_algorithm
            ),
            heat_relations
            + ("所有热源分量相加后仍必须与吸收、荧光逃逸、ASE 和未解析去向闭合",),
        ),
        ModelStage(
            "三维导热、应力与热光反馈",
            (
                f"热场为 `{thermal_effective}`，稳态后端为 `{thermal_cfg.get('linear_solver', 'auto')}`，"
                f"光谱参数反馈为 `{response_mode}`，热应力为 `{stress_model}`，折射率模型为 `{refraction}`。"
                f"整体耦合使用 `{coupling}`。{coupling_algorithm}"
                "温度相关寿命和截面只有在 operating_point/local_3d 被明确启用时才修改速率方程；local_3d"
                "必须位于固定点内，不会静默用统一温度冒充局部反馈。"
            ),
            thermal_relations,
        ),
        ModelStage(
            "三套网格如何交换物理量",
            (
                f"布居、热源、温度和应力位于 {grid.get('nx')}×{grid.get('ny')}×{grid.get('nz')} 的 `{spacing}`"
                f"物理网格；Seed 光学采样为 `{optical_effective}`，Pump 光学网格为 `{pump_grid_mode}`。"
                f"{refinement_stage_note}"
                "FFT 只能在均匀光学网格上运行；若物理网格局部加密，则通过有限体积单元重叠把光学功率"
                "密度映射到晶体网格，再把布居增益和 Δn 映射回光学网格。映射以积分功率/能量守恒为合同，"
                "不会把局部加密误当成 FFT 的非均匀采样。"
            ),
            (
                "光学→物理：保持 ∫I dA 与每个 z 层吸收功率",
                "物理→光学：投影 g 与 Δn，并分别记录投影修正和边界占用诊断",
            ),
        ),
        ModelStage(
            "出光面结果与定量有效性合同",
            (
                f"最终输出包含平均功率、单脉冲能量、空间二阶矩、光谱/时间结果以及已保存的三维场。"
                f"当前验证模式为 `{validation_mode}`；它检查 Pump 与 Seed 能量账本、布居守恒和数值投影、"
                "周期/多物理收敛、FFT 时间与空间边界占用、材料表越界、ASE clamp 以及模型适用范围。"
                "exploratory 会保留完成但无效的诊断点，strict 还会禁止其正式导出；二者都不会把阻断性失败"
                "包装成可信定量结论。"
            ),
            (
                "输出平均功率 Pout = Eout × frep（有重复频率时）",
                "能量、光子数和外部具名损耗分别闭合；记录平面/M²质量问题与核心物理失效分级报告",
            ),
        ),
    )
    return ModelDescription(
        stages=stages,
        features=tuple(features),
        notices=tuple(notices),
    )
