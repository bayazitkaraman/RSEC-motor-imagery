# Riemann Stereographic Energy Coupling for Motor-Imagery EEG Analysis

This repository contains reproducible code and summary outputs for analyzing motor-imagery-related EEG network reconfiguration using **Riemann Stereographic Energy Coupling (RSEC)**.

RSEC is a nonlinear sensor-space feature-extraction method that transforms EEG epochs into analytic signals, maps them through a Riemann stereographic energy representation, and detects transient shared-energy events across EEG channel pairs.

## Project Overview

The analysis uses the public **PhysioNet EEG Motor Movement/Imagery dataset** and evaluates global RSEC density across rest, motor imagery, and motor execution conditions.

The main analysis includes:

- RSEC shared-energy feature extraction
- Subject-level rest, imagery, and execution comparisons
- Motor imagery versus motor execution contrast
- Phase-randomized surrogate control
- Node-participation summaries
- Reproducible figure generation from saved summary CSV files

Raw EEG recordings are **not included** in this repository. Full reruns load the PhysioNet EEGMMI dataset through MNE-Python.

## Repository Structure

```text
rsec_eeg/                         Core RSEC feature, statistic, and surrogate functions
scripts/                          Analysis, verification, and figure-generation scripts
results/summary/                  Small CSV files containing subject-level/statistical summaries
results/figures/                  Figures generated from the summary CSV files
docs/                             Reference manuscript PDF or related documentation
```

## Setup

### Python Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Verify Saved Results

This script checks that the included summary CSV files match the reported statistical values.

```powershell
python .\scripts\verify_reported_statistics.py
```

Expected final line:

```text
All reported p-values verified.
```

## Recreate Figures from Saved Summaries

```powershell
python .\scripts\make_figures_from_summaries.py
```

Generated figures are saved to:

```text
results/figures/
```

Main generated figures:

```text
fig2_task_minus_rest_deltas.png
fig3_real_minus_surrogate_delta.png
fig4_top_node_participation_imagery.png
```

## Reproduced Summary Values

The saved summaries and full rerun commands reproduce the following subject-level results:

```text
Imagery vs rest:
Δ = -0.002258, p = 0.00148, d_z = -0.354

Execution vs rest:
Δ = -0.000005, p = 0.373, d_z = -0.0007

Imagery Δ vs execution Δ:
Δ = -0.002254, p = 8.02e-10, d_z = -0.567

Real Δ vs phase-randomized surrogate Δ:
Δ = -0.002289, p = 0.00153
```

The main interpretation is that motor imagery significantly reduced global RSEC density relative to rest, whereas motor execution did not produce a comparable global-density change. The imagery-related effect also differed from the phase-randomized surrogate control.

## Full EEG Rerun

Full reruns recompute the analysis from the PhysioNet EEGMMI recordings. These commands may take time depending on hardware and whether the data are already downloaded.

```powershell
$subjects = 1..109 | Where-Object { $_ -notin 88,92,100 }
```

### Motor Imagery

```powershell
python .\scripts\run_physionet_eegmmi.py `
    --subjects $subjects `
    --mode imagery_lr `
    --rest-run 1 `
    --data-dir "C:\Users\Bayazit Karaman\mne_data" `
    --out-dir outputs_physionet_imagery_fresh `
    --duration-sec 4 `
    --band 1 31 `
    --decimals 3 `
    --max-epochs 50
```

### Motor Execution

```powershell
python .\scripts\run_physionet_eegmmi.py `
    --subjects $subjects `
    --mode execution_lr `
    --rest-run 1 `
    --data-dir "C:\Users\Bayazit Karaman\mne_data" `
    --out-dir outputs_physionet_execution_fresh `
    --duration-sec 4 `
    --band 1 31 `
    --decimals 3 `
    --max-epochs 50
```

### Compare Imagery and Execution

```powershell
python .\scripts\compare_physionet_modes.py `
    --imagery-dir outputs_physionet_imagery_fresh `
    --execution-dir outputs_physionet_execution_fresh `
    --out-dir outputs_spmb_compare_fresh
```

### Phase-Randomized Surrogate Control

```powershell
python .\scripts\run_physionet_surrogate.py `
    --subjects $subjects `
    --mode imagery_lr `
    --rest-run 1 `
    --data-dir "C:\Users\Bayazit Karaman\mne_data" `
    --out-dir outputs_surrogate_imagery_fresh `
    --duration-sec 4 `
    --band 1 31 `
    --decimals 3 `
    --max-epochs 50 `
    --seed 42
```

## Notes on Reproducibility

The repository includes small summary CSV files and generated figures so the reported results can be checked without rerunning the full EEG pipeline. Full EEG reruns should use the same subject list, preprocessing parameters, rounding precision, epoch duration, and surrogate seed to reproduce the saved statistics.

Excluded subjects:

```text
88, 92, 100
```

These subjects were excluded because baseline and task recordings had inconsistent sampling frequencies.

## Citation

A citation entry will be added after publication.