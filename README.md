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

The recorded study runtime used **Python 3.9.19**, NumPy 1.24.3, SciPy 1.13.1, pandas 2.2.2, Matplotlib 3.8.4, MNE-Python 1.7.1 and MNE-Connectivity 0.7.0. For exact-version reproduction, use Python 3.9.19 and install `requirements-reproduction.txt` instead of the general requirements. This file records the scientific dependency closure actually present in the study runtime, including its MNE-Connectivity dependency overlay; a clean installation of the exported pins has not been tested. The machine-readable record is `results/summary/reproduction_environment.json`.

All 38 recorded scientific package versions and their active dependency requirements were checked successfully. A whole-environment `pip check` found five unrelated conflicts in the shared GPU development environment, listed in that record; this is not a claim that the entire shared environment is conflict-free.

## Verify Saved Results

This script checks data integrity, recomputes the primary paired tests and effect sizes from saved participant summaries, and checks correction families and supporting-result coverage. It does not rerun raw EEG processing or the paired bootstrap comparison analysis.

```powershell
python .\scripts\verify_reported_statistics.py
```

Expected final line:

```text
All result checks passed.
```

To reconstruct all 10 statistical tables, including the four-test primary family, 10,000-resample mean intervals, and 100,000-resample paired effect-size intervals:

```powershell
python .\scripts\test_pipeline.py
python .\scripts\verify_reproduction.py
```

`verify_reproduction.py` runs the statistical construction scripts from the supplied participant/epoch inputs, compares every reported field with the saved tables, and removes its temporary outputs. It does not process raw EEG. The expected final line is `All 10 reported statistical tables reproduced, including bootstrap interval endpoints.`

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

The following commands recompute the primary conditions, surrogate control, conventional comparisons and sensitivity analyses. Run commands from the repository root. The primary runners download missing PhysioNet EEGMMI recordings through MNE; the other raw-analysis scripts read those locally cached EDFs. Full-cohort processing can take substantial time.

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

This descriptive helper has a three-test Holm family and t intervals. The manuscript uses the separate four-test participant-bootstrap construction below, after the surrogate run.

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

### Primary Statistical Table

```powershell
python .\scripts\summarize_primary_statistics.py `
    --imagery-summary outputs_physionet_imagery\subject_summary.csv `
    --execution-summary outputs_physionet_execution\subject_summary.csv `
    --surrogate-summary outputs_surrogate_imagery\surrogate_subject_summary.csv
```

This validates participant alignment and shared rest values, computes all four contrasts, applies one four-test Holm correction, and constructs the 10,000-resample percentile mean intervals (seed 20261001, reset for each contrast). Without input arguments it uses the supplied participant summaries.

### Matched Conventional Comparisons

```powershell
python .\scripts\run_comparison_analysis.py --data-dir .\mne_data
python .\scripts\run_comparison_analysis.py --data-dir .\mne_data --low-frequency-check --out-dir outputs_low_frequency
python .\scripts\summarize_comparison_statistics.py `
    --input outputs_comparisons\comparison_subject_features.csv `
    --low-frequency-input outputs_low_frequency\low_frequency_subject_features.csv
```

Each condition uses 15 epochs. Task results average 20 selections without replacement within each selection; the same selections are used across methods and bands. Selection seed 20261001 is combined with participant and condition indices through NumPy `SeedSequence`. The saved selections are in `comparison_epoch_selections.json`. Native Fourier coherence, PLV, PLI and wPLI are computed with MNE-Connectivity, alongside untransformed EEG power and full-pipeline RSEC. The summarizer computes 54 signed tests, 45 paired effect-size comparisons (100,000 participant resamples; seed 20261002; Bonferroni-adjusted percentile intervals), and the 12-test low-frequency check.

### Precision, Reference, Band and Spatial Analyses

```powershell
python .\scripts\run_feature_analysis.py --data-dir .\mne_data
python .\scripts\summarize_feature_statistics.py --run-dir outputs_features
```

These commands construct the seven-setting, 21-test sensitivity family and the participant-level channel and regional results. The spatial summary reader preserves float64 rank ties using round-trip parsing of saved node differences.

### Amplitude and Surface-Laplacian Analyses

```powershell
python .\scripts\run_artifact_analysis.py --data-dir .\mne_data
python .\scripts\summarize_artifact_statistics.py --input outputs_artifacts\artifact_epoch_features.csv
```

Artifact inputs retain all diagnostic metrics used in the complete 72-test correction family. The paper's RSEC rows use `metric=rsec`; diagnostic linear-energy and Hilbert-based fields in the supporting inputs are not the paper's conventional baselines.

### All-Participant Sampling-Rate Checks

The all-109 checks also need runs 1, 3, 4, 7, 8, 11 and 12 for participants 88, 92 and 100. Download these into the same MNE cache before running:

```powershell
python -c "from mne.datasets import eegbci; [eegbci.load_data(s, [1,3,4,7,8,11,12], path='mne_data', update_path=False) for s in [88,92,100]]"
python .\scripts\run_cohort_analysis.py --data-dir .\mne_data
python .\scripts\summarize_cohort_statistics.py --input-dir outputs_cohort
python .\scripts\audit_annotation_containment.py --data-dir .\mne_data --out-dir outputs_annotation_audit
```

Common-rate analyses resample each continuous recording to 160 or 128 Hz, jointly remap task events, and then apply the 1-31 Hz filter. The containment audit reads task annotations and recording lengths, without filtering or feature extraction.

## Notes on Reproducibility

The repository includes the participant/epoch inputs, generating scripts, statistical summaries and figures. Full EEG reruns must use the recorded environment, participant order, preprocessing parameters, rounding precision, epoch duration, selection procedure and seeds. The surrogate RNG advances across participants; a subset run is a smoke test, not a reproduction of the full-cohort surrogate draws.

Excluded subjects:

```text
88, 92, 100
```

These subjects were excluded because baseline and task recordings had inconsistent sampling frequencies.

All-109-participant resampling checks are included in `cohort_sensitivity.csv`. The primary analysis uses all available task epochs (`--max-epochs 0`). Energy uses the natural logarithm, with `1e-12` stabilizers in normalization and the logarithm.

The summary CSV files include all 45 paired method comparisons, all 21 sensitivity tests, and all nine RSEC artifact/aggregation settings. Artifact files retain the full statistical correction families; RSEC rows are identified by `metric=rsec`.

The participant and comparison figures and three tables can be regenerated without raw EEG. The pipeline and surrogate images are supplied directly in `results/figures/`.

Validation reproduced all 10 statistical tables from their saved inputs, including confidence-interval endpoints. Bounded raw checks covered both surrogate cap modes, participant 1 across native comparators and reference/band/precision/artifact analyses, and participants 1 and 88 at both common sampling rates. This validation did not rerun every raw-data analysis for the full cohort.

The annotation audit covers all 654 task recordings: all 9,844 retained task epochs (9,544 in the primary cohort) fit their task labels. One incomplete terminal candidate in participant 104, run 8, is excluded by the recording boundary. Raw feature extraction rejects nonfinite input, and retained task windows are checked against annotation durations.

## Citation

A citation entry will be added after publication.
