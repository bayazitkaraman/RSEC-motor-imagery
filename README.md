# RSEC SPMB Conference Project

This is the clean, conference-only reproducibility project for:

**Riemann Stereographic Energy Coupling Suggests Motor-Imagery-Specific EEG Network Reconfiguration**

This project is intentionally separated from the journal project so conference and journal results do not get mixed.

## What is included

```text
rsec_eeg/                         Core RSEC feature, statistic, and surrogate functions
scripts/                          Reproducibility and figure scripts
results/summary/                  Small CSV summaries used for the SPMB paper
results/figures/                  Figures generated from the summary CSV files
docs/RSEC_SPMB_current_draft.pdf  Current reference PDF draft
```

Raw PhysioNet EEG files are **not** included. Full reruns download/load data through MNE-Python.

## Quick setup on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick check without rerunning EEG analysis

This verifies that the included CSV summaries match the conference result values.

```powershell
python .\scripts\verify_reported_statistics.py
```

Expected final line:

```text
All conference p-values verified.
```

## Recreate conference figures from included CSV summaries

```powershell
python .\scripts\make_figures_from_summaries.py
```

Generated figures:

```text
results/figures/fig2_task_minus_rest_deltas.png
results/figures/fig3_real_minus_surrogate_delta.png
results/figures/fig4_top_node_participation_imagery.png
```

## Official conference summary values in this project

Use these values consistently in the conference paper:

```text
Imagery vs rest:                   delta = -0.002123, p = 0.00415, d_z = -0.300
Execution vs rest:                 delta =  0.000247, p = 0.248,   d_z =  0.034
Imagery delta vs execution delta:  delta = -0.002370, p = 8.72e-7, d_z = -0.443
Real delta vs surrogate delta:     delta = -0.002184, p = 0.00333
```

## Full EEG rerun commands

Full reruns can take time. Use these only when you want to recompute outputs from PhysioNet EEGMMI.

```powershell
$subjects = 1..109 | Where-Object { $_ -notin 88,92,100 }

python .\scripts\run_physionet_eegmmi.py `
    --subjects $subjects `
    --mode imagery_lr `
    --data-dir mne_data `
    --out-dir outputs_physionet_imagery_50 `
    --max-epochs 50

python .\scripts\run_physionet_eegmmi.py `
    --subjects $subjects `
    --mode execution_lr `
    --data-dir mne_data `
    --out-dir outputs_physionet_execution_50 `
    --max-epochs 50

python .\scripts\compare_physionet_modes.py `
    --imagery-dir outputs_physionet_imagery_50 `
    --execution-dir outputs_physionet_execution_50 `
    --out-dir outputs_spmb_compare
```

Surrogate control:

```powershell
python .\scripts\run_physionet_surrogate.py `
    --subjects $subjects `
    --mode imagery_lr `
    --data-dir mne_data `
    --out-dir outputs_surrogate_imagery_50 `
    --max-epochs 50 `
    --seed 42
```

## Create a new GitHub repository from this project

```powershell
git init
git add .
git commit -m "Initial RSEC SPMB conference project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/RSEC-SPMB-Conference.git
git push -u origin main
```

Use a separate GitHub repository for the journal project.
