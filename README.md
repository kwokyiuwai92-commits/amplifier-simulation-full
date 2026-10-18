# Amplifier Simulation

完整版源码包，整理日期：2026-10-18。包含当前计算核心、网页界面和一个实际运行过的配置。

## 包含内容

- `app.py`、`src/`、`pyproject.toml`、`.streamlit/`：完整计算程序、界面、依赖声明和界面设置。
- `configs/example.json`：2026-09-15 九点位置扫描中第 9 个点的原始运行配置，参数未经修改。
- `README.md`：本说明。

## 安装与启动

先解压 ZIP，进入含有 `app.py` 的 `AmplifierSimulation` 文件夹，再执行下面的命令。
需要 Python；本包已使用 macOS / Apple Silicon / Python 3.10 检查安装、界面和配置加载。
建议使用 Python 3.10 或 3.11。首次安装依赖需要联网。

### macOS / Linux

```sh
python3 -m venv .venv
.venv/bin/python -m pip install ".[ui,performance]"
.venv/bin/python -m streamlit run app.py
```

### Windows（PowerShell 或命令提示符）

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install ".[ui,performance]"
.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器通常会自动打开；也可访问 http://localhost:8501 。
在左侧“配置文件”中上传 `configs/example.json`，表单会自动回填，然后点击运行。
以后启动时只需再次执行相应的 `streamlit run app.py` 命令。

Windows / Linux 尚未实机验证；程序主要使用通用 Python 库。
如果需要导入 HDF5 格式的 FROG 文件，可额外安装 `h5py`；本例输入已内嵌，无需该项。

## 模型名称

快速模型统一称为 **Commercial App**；代码与配置中的标识符为 `commercial_app_fast`。

## 示例配置

- Cr:ZnS 增益晶体长度 15 mm；前置 ZnSe 厚度 15 mm。
- Pump 12.765 W；Seed 20.58 nJ × 25 MHz = 0.5145 W。
- 掺杂离子密度 6.66 × 10²⁴ m⁻³；寿命保留系数 0.50。
- Pump / Seed 透镜到晶体距离 150 / 200 mm。
- HG 多模式关闭，M² x/y = 1；采用径向 m=0、±1 相干传播。
- 保留原始联合热光迭代、Kerr 与采样设置；7PA / FCA 关闭。

光谱、时间和相位输入已嵌入 JSON。里面的历史绝对路径仅用于记录来源，运行不需要原电脑的文件。
这是完整研究计算，耗时取决于硬件；本包未附历史结果。原历史运行已完成并收敛，但仍有未通过的数值检查；解释结果时应同时查看收敛和有效性诊断。
当前源码包含历史计算之后的修复，重新计算的数值可能与旧结果不同。

## 命令行运行（可选）

macOS / Linux：

```sh
.venv/bin/python -m amplifier_sim run configs/example.json -o output/example
```

Windows：

```powershell
.venv\Scripts\python.exe -m amplifier_sim run configs/example.json -o output/example
```

命令行结果通常包括 `summary.json`、`fields.npz` 和按配置生成的附加文件。
网页单次计算结果会保存到程序提示的位置，默认结果目录为 `exports/`。
全部 JSON 数值默认采用 SI 单位；界面另有单位换算显示。
