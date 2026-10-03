"""Check current saved results, not a raw-EEG or bootstrap rerun."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsec_eeg.stats import holm_adjust

DATA = ROOT / "results/summary"


def main():
    manifest = json.loads((DATA / "data_manifest.json").read_text())
    assert len(manifest) == 13
    for name, expected_hash in manifest.items():
        content = (DATA / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected_hash, name

    primary = pd.read_csv(DATA / "primary_statistics.csv").set_index("contrast")
    subjects = pd.read_csv(DATA / "primary_subject_summary.csv").set_index("subject")
    assert set(subjects.index) == set(range(1, 110)) - {88, 92, 100}
    assert len(subjects) == 106 and not subjects.isna().any().any()
    contrasts = {
        "imagery-rest": subjects.imagery - subjects.rest,
        "execution-rest": subjects.execution - subjects.rest,
        "imagery-execution": subjects.imagery - subjects.execution,
    }
    surrogate = pd.read_csv(ROOT / "results/summary/surrogate_subject_summary.csv")
    surrogate = surrogate.pivot(index="subject", columns="data_type", values="delta_task_minus_rest")
    assert set(surrogate.index) == set(subjects.index)
    contrasts["real-surrogate"] = surrogate.real - surrogate.phase_randomized_surrogate
    for name, values in contrasts.items():
        record = primary.loc[name]
        np.testing.assert_allclose(values.mean(), record.mean_delta, atol=1e-14, rtol=0)
        np.testing.assert_allclose(values.mean() / values.std(ddof=1), record.dz, atol=1e-11, rtol=0)
        probability = wilcoxon(values, zero_method="wilcox", correction=False, method="approx").pvalue
        np.testing.assert_allclose(probability, record.p, atol=1e-12, rtol=0)
        assert int((values < 0).sum()) == int(record.n_decrease)
    np.testing.assert_allclose(holm_adjust(primary.p.to_numpy()), primary.p_holm, atol=1e-12, rtol=0)

    comparison = pd.read_csv(DATA / "statistics_54.csv")
    assert len(comparison) == 54 and comparison.n.eq(106).all()
    np.testing.assert_allclose(holm_adjust(comparison.p.to_numpy()), comparison.p_holm_54, atol=1e-12, rtol=0)
    paired = pd.read_csv(DATA / "paired_effect_comparisons_45.csv")
    assert len(paired) == 45
    assert paired.groupby("contrast").size().eq(15).all()
    imagery = paired[paired.contrast.ne("execution-rest")]
    assert len(imagery) == 30 and ((imagery.ci_low < 0) & (imagery.ci_high > 0)).all()
    sensitivity = pd.read_csv(DATA / "sensitivity_statistics.csv")
    assert len(sensitivity) == 21 and sensitivity.groupby("setting").size().eq(3).all()
    artifact = pd.read_csv(DATA / "amplitude_aggregation_statistics.csv")
    artifact = artifact[artifact.metric.eq("rsec")]
    assert len(artifact) == 27 and artifact.groupby("setting").size().eq(3).all()
    for setting, count in (("fp-ptp-150", 15), ("fp-ptp-250", 34)):
        record = artifact[artifact.setting.eq(setting) & artifact.contrast.eq("imagery-rest")].iloc[0]
        assert record.n == count and record.mean_delta > 0 > record.matched_original_mean_delta
    nodes = pd.read_csv(DATA / "imagery_node_deltas_by_subject.csv").set_index("subject")
    assert set(nodes.index) == set(subjects.index) and nodes.shape == (106, 64)
    assert int((nodes[["Fp1", "Fpz", "Fp2"]].mean(axis=1) < 0).sum()) == 78
    assert int((nodes[["C3", "Cz", "C4"]].mean(axis=1) < 0).sum()) == 54
    print(primary[["mean_delta", "dz", "p", "p_holm", "n_decrease"]].to_string())
    print("Checked 13 CSV hashes, primary paired statistics, 54 method/contrast tests,")
    print("45 paired comparisons, 21 sensitivity tests, and 9 RSEC artifact settings.")
    print("No raw EEG processing or paired bootstrap comparisons were rerun.")
    print("All result checks passed.")


if __name__ == "__main__":
    main()
