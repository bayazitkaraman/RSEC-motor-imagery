# Riemann Stereographic Energy Coupling for Motor-Imagery EEG Analysis

This repository contains reproducible code and summary outputs for analyzing motor-imagery-related EEG network reconfiguration using **Riemann Stereographic Energy Coupling (RSEC)**.

**Current manuscript:** IEEE SPMB 2026, accepted with mandatory revision. The October 2, 2026 draft and its supporting results are available in [docs/spmb_2026](docs/spmb_2026). This is not a final published article or an accepted camera-ready version.

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
results/summary/spmb_2026/        Current manuscript's 12 supporting-result CSV files
results/figures/spmb_2026/        Current figures and generated LaTeX tables
docs/spmb_2026/                   Clean/red PDFs, portable LaTeX source ZIP and result index
results/summary/                  Original submitted-paper summaries (historical)
results/figures/                  Original submitted-paper figures (historical)
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

This script checks archive/data integrity, recomputes the primary paired tests and effect sizes from saved participant summaries, checks correction families and verifies supporting-result coverage. It does not rerun raw EEG processing or the paired bootstrap comparison analysis.

```powershell
python .\scripts\verify_spmb_2026_results.py
```

Expected final line:

```text
All current SPMB result checks passed.
```

## Recreate Figures from Saved Summaries

```powershell
python .\scripts\make_spmb_2026_assets.py
```

Generated figures are saved to:

```text
results/figures/spmb_2026/
```

Current manuscript assets:

```text
pipeline.png
participant_spatial_results.pdf / .png
fig3_real_minus_surrogate_delta.png
comparison_uncertainty.pdf / .png
table_primary.tex, table_comparisons.tex, table_sensitivity.tex
```

## Reproduced Summary Values

The current all-epoch primary cohort contains 106 participants. The source archive describes the complete analysis settings. Primary results are:

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

The following commands recompute the primary RSEC condition summaries and surrogate control from PhysioNet EEGMMI. They do not run every conventional-comparison or sensitivity analysis. Those analyses are distributed here as validated saved results, not as a complete raw-EEG rerun workflow. The legacy runner reports t intervals; the manuscript's mean-difference intervals are participant-bootstrap intervals in the current saved results. Commands may take time depending on hardware and data availability. For the audited preprocessing environment, MNE-Python 1.7.1 was used.

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
    --out-dir outputs_physionet_imagery_fresh `
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
    --out-dir outputs_physionet_execution_fresh `
    --duration-sec 4 `
    --band 1 31 `
    --decimals 3 `
    --max-epochs 0
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
    --data-dir ".\mne_data" `
    --out-dir outputs_surrogate_imagery_fresh `
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

All-109-participant resampling checks are included in `cohort_sensitivity.csv`. The current primary analysis uses all available task epochs; `--max-epochs 0` disables the earlier 50-epoch cap. This adds two execution epochs for subject 89 and explains the small change in execution-related summaries. The RSEC transformation and event rule are unchanged. Energy uses the natural logarithm, with `1e-12` stabilizers in normalization and the logarithm.

The current archive contains all 45 paired method comparisons (including 15 execution-rest comparisons not plotted), all 21 sensitivity tests (including seven execution-rest tests not tabulated), and all nine RSEC artifact/aggregation settings. See the [supporting-results index](docs/spmb_2026/SUPPLEMENT_INDEX.txt) for correction-family and provenance details. Historical linear-feature rows in the artifact files are preserved for provenance; they are not manuscript baselines.

The original `scripts/verify_reported_statistics.py` and `scripts/make_figures_from_summaries.py` still operate on the historical top-level summaries. Use the `spmb_2026` commands above for the current manuscript. The two new data-derived figures and three tables can be regenerated without raw EEG; the author's pipeline image and original surrogate image are preserved unchanged from the source archive.

## Manuscript and Submission Status

The clean and red-marked PDFs contain the same current text. Red identifies additions/replacements against the approved original-based draft; deletions are recorded in `CHANGES.diff` inside the source ZIP. The archive compiles in Overleaf with `SPMBRiemannSteBK.tex` or `SPMBRiemannSteBK_changes.tex` as the main file.

The nine-page manuscript has no imposed page-count target. The [official 2026 guidelines](https://isip.piconepress.com/conferences/ieee_spmb/2026/html/guidelines.shtml) prefer four to six pages; organizer approval for the final length and a fresh similarity report remain necessary. Repository publication is not conference submission or approval. No raw EEG or private future-paper material is included.

## Citation

A citation entry will be added after publication.
