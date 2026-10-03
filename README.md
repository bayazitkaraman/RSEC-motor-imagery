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
- Native power, coherence, PLV, PLI and wPLI comparisons on matched epochs
- Precision, reference, cohort/sampling-rate and artifact sensitivity
- Participant-level global and regional consistency, with uncertainty intervals

Raw EEG recordings are **not included** in this repository. Full reruns load the PhysioNet EEGMMI dataset through MNE-Python.

## Repository Structure

```text
rsec_eeg/                         Core RSEC feature, statistic, and surrogate functions
scripts/                          Analysis, verification, and figure-generation scripts
results/summary/                  Subject-level and statistical summaries
results/figures/                  Figures and tables generated from saved summaries
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

This script checks data integrity, recomputes the primary paired tests and effect sizes from saved participant summaries, and checks correction families and supporting-result coverage. It does not rerun raw EEG processing or the paired bootstrap comparison analysis.

```powershell
python .\scripts\verify_reported_statistics.py
```

Expected final line:

```text
All result checks passed.
```

## Recreate Figures from Saved Summaries

```powershell
python .\scripts\make_figures_from_summaries.py
```

Generated figures are saved to:

```text
results/figures/
```

Figures and tables:

```text
pipeline.png
participant_spatial_results.pdf / .png
fig3_real_minus_surrogate_delta.png
comparison_uncertainty.pdf / .png
table_primary.tex, table_comparisons.tex, table_sensitivity.tex
```

## Reproduced Summary Values

The all-epoch primary cohort contains 106 participants. Primary results are:

```text
Imagery vs rest:
Delta = -0.002258, raw p = 0.001479, Holm p = 0.004438, d_z = -0.354

Execution vs rest:
Delta = -0.000001615, raw/Holm p = 0.366517, d_z = -0.000255

Imagery vs execution:
Delta = -0.002257, raw p = 8.02e-10, Holm p = 3.21e-9, d_z = -0.569

Real imagery-rest delta vs phase-randomized surrogate delta:
Delta = -0.002289, raw p = 0.001528, Holm p = 0.004438
```

The main interpretation is that motor imagery significantly reduced global RSEC density relative to rest, whereas motor execution did not produce a comparable global-density change. The imagery-related effect also differed from the phase-randomized surrogate control.

The matched conventional comparison is a separate analysis with balanced 15-epoch selections and a 54-test Holm family. All 30 imagery-related between-method effect-difference intervals include zero; superiority or equivalence is not established. Frontopolar participation decreased in 78/106 participants, but the amplitude-rejection analyses concern **global density**, not direct validation of regional neural sources. Strict frontopolar thresholds reversed the global imagery-rest mean in small retained cohorts. Both sides of those comparisons used the same 1-31 Hz filtering.

## Full EEG Rerun

The following commands recompute the primary RSEC condition summaries and surrogate control from PhysioNet EEGMMI. Conventional-comparison and sensitivity results are provided as summary CSV files, not as a complete raw-EEG rerun workflow. The command-line summaries use t intervals; `primary_statistics.csv` contains participant-bootstrap mean-difference intervals. Commands may take time depending on hardware and data availability. The preprocessing environment used MNE-Python 1.7.1.

```powershell
$subjects = 1..109 | Where-Object { $_ -notin 88,92,100 }
```

### Motor Imagery

```powershell
python .\scripts\run_physionet_eegmmi.py `
    --subjects $subjects `
    --mode imagery_lr `
    --rest-run 1 `
    --data-dir ".\mne_data" `
    --out-dir outputs_physionet_imagery `
    --duration-sec 4 `
    --band 1 31 `
    --decimals 3 `
    --max-epochs 0
```

### Motor Execution

```powershell
python .\scripts\run_physionet_eegmmi.py `
    --subjects $subjects `
    --mode execution_lr `
    --rest-run 1 `
    --data-dir ".\mne_data" `
    --out-dir outputs_physionet_execution `
    --duration-sec 4 `
    --band 1 31 `
    --decimals 3 `
    --max-epochs 0
```

### Compare Imagery and Execution

```powershell
python .\scripts\compare_physionet_modes.py `
    --imagery-dir outputs_physionet_imagery `
    --execution-dir outputs_physionet_execution `
    --out-dir outputs_spmb_compare
```

### Phase-Randomized Surrogate Control

```powershell
python .\scripts\run_physionet_surrogate.py `
    --subjects $subjects `
    --mode imagery_lr `
    --rest-run 1 `
    --data-dir ".\mne_data" `
    --out-dir outputs_surrogate_imagery `
    --duration-sec 4 `
    --band 1 31 `
    --decimals 3 `
    --max-epochs 0 `
    --seed 42
```

## Notes on Reproducibility

The repository includes small summary CSV files and generated figures so the reported results can be checked without rerunning the full EEG pipeline. Full EEG reruns should use the same subject list, preprocessing parameters, rounding precision, epoch duration, and surrogate seed to reproduce the saved statistics.

Excluded subjects:

```text
88, 92, 100
```

These subjects were excluded because baseline and task recordings had inconsistent sampling frequencies.

All-109-participant resampling checks are included in `cohort_sensitivity.csv`. The primary analysis uses all available task epochs (`--max-epochs 0`). Energy uses the natural logarithm, with `1e-12` stabilizers in normalization and the logarithm.

The summary CSV files include all 45 paired method comparisons, all 21 sensitivity tests, and all nine RSEC artifact/aggregation settings. Artifact files retain the full statistical correction families; RSEC rows are identified by `metric=rsec`.

The participant and comparison figures and three tables can be regenerated without raw EEG. The pipeline and surrogate images are supplied directly in `results/figures/`.

## Citation

A citation entry will be added after publication.
