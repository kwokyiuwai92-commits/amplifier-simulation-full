# Amplifier Simulation

Complete source release, prepared on 18 October 2026. Includes the calculation core, web interface, and a previously executed example configuration.

[Chinese README](https://github.com/kwokyiuwai92-commits/amplifier-simulation-full/blob/main/README_zh.md)

## Contents

- `app.py`, `src/`, `pyproject.toml`, and `.streamlit/`: the complete calculation code, web interface, dependency definitions, and interface settings.
- `configs/example.json`: operating point 9 from the nine-point position scan completed on 15 September 2026. Physical and numerical settings are preserved; identifiers use this release's naming scheme.
- `README.md`: these instructions. The ZIP contains the English README only; GitHub also provides the Chinese README.

## Installation and launch

Extract the ZIP and open a terminal in the `AmplifierSimulation` folder containing `app.py`.
Use Python 3.10, 3.11, or 3.12. An internet connection is required for the initial dependency installation.

### macOS / Linux

```sh
python3 -m venv .venv
.venv/bin/python -m pip install ".[ui,performance]"
.venv/bin/python -m streamlit run app.py
```

### Windows (PowerShell or Command Prompt)

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install ".[ui,performance]"
.venv\Scripts\python.exe -m streamlit run app.py
```

The browser should open automatically. Otherwise, visit http://localhost:8501 .
Choose **English** in the sidebar if needed, open **Configuration File**, and upload `configs/example.json`. The form will populate automatically. Click **Run Current Configuration** to start the calculation.
For later sessions, rerun the appropriate `streamlit run app.py` command above.

The example embeds its input data. Importing HDF5 FROG files separately requires the optional `h5py` package.

## Model naming

The fast model is named **Commercial App**. Its code and configuration identifier is `commercial_app_fast`.

## Example settings

- Cr:ZnS gain crystal length: 15 mm; preceding ZnSe thickness: 15 mm.
- Pump power: 12.765 W; Seed: 20.58 nJ at 25 MHz, equivalent to 0.5145 W.
- Dopant ion density: 6.66 x 10^24 m^-3; upper-state lifetime scale: 0.50.
- Pump / Seed lens-to-crystal distances: 150 / 200 mm.
- HG multimode propagation is off; M-squared is 1 on both axes. The coherent spatial model uses a radial grid with azimuthal orders m = 0, +/-1.
- Original joint thermal-optical iteration, Kerr, and sampling settings are retained; 7PA and FCA are off.

Spectrum, temporal, and phase inputs are embedded in the JSON. Historical absolute paths are provenance records; the original computer's files are not required.

This is a full research calculation and can take substantial time. Historical results are not included. The historical run completed and converged but did not pass every numerical validity check; inspect both convergence and validity diagnostics before interpreting results quantitatively.
The current source includes fixes made after the historical calculation, so rerun values may differ from the archived results.

## Command-line execution (optional)

macOS / Linux:

```sh
.venv/bin/python -m amplifier_sim run configs/example.json -o output/example
```

Windows:

```powershell
.venv\Scripts\python.exe -m amplifier_sim run configs/example.json -o output/example
```

Command-line output normally includes `summary.json`, `fields.npz`, and any additional files requested by the configuration.
Single-run results from the web interface are saved to the displayed location; the default archive directory is `exports/`.
JSON values use SI units by default. The interface provides unit conversions for display.

## Verification

Installation, configuration loading, and the interface were checked on macOS / Apple Silicon with Python 3.10.
A separate extraction of the ZIP was then installed in a fresh Python 3.12 environment. Both a small-grid thermal-optical fixed-point calculation and a Commercial App calculation with the same coupling converged and passed their configured checks.
This verification did not rerun the complete 15 mm example. Windows and Linux have not been tested on physical machines.
