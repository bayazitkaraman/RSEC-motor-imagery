# run_precision_sensitivity.ps1
# Run from the repository root:
#   cd "C:\Users\Bayazit Karaman\Desktop\Projects\riemann-stereographic-energy-coupling"
#   .\scripts\run_precision_sensitivity.ps1

$bad = @(88, 92, 100)
$subjects = 1..109 | Where-Object { $_ -notin $bad }
$dataDir = "C:\Users\Bayazit Karaman\mne_data"
$decimalsList = @(2, 3, 4, 5)

foreach ($d in $decimalsList) {
    Write-Host "==============================="
    Write-Host "Running precision d=$d"
    Write-Host "==============================="

    python .\scripts\run_physionet_eegmmi.py `
        --subjects $subjects `
        --mode imagery_lr `
        --max-epochs 20 `
        --data-dir "$dataDir" `
        --decimals $d `
        --out-dir "outputs_sensitivity_d${d}_imagery"

    python .\scripts\run_physionet_eegmmi.py `
        --subjects $subjects `
        --mode execution_lr `
        --max-epochs 20 `
        --data-dir "$dataDir" `
        --decimals $d `
        --out-dir "outputs_sensitivity_d${d}_execution"

    python .\scripts\compare_physionet_modes.py `
        --imagery-dir "outputs_sensitivity_d${d}_imagery" `
        --execution-dir "outputs_sensitivity_d${d}_execution" `
        --out-dir "outputs_sensitivity_d${d}_compare"
}

python .\scripts\summarize_precision_sensitivity.py `
    --base-dir "." `
    --decimals 2 3 4 5 `
    --out-dir "outputs_precision_sensitivity_summary"

Write-Host "Done. Summary saved to outputs_precision_sensitivity_summary"
