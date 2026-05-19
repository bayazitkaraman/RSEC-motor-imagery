"""Verify the reported SPMB statistics from included summary CSV files."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
stats = pd.read_csv(ROOT / "results" / "summary" / "spmb_statistical_summary.csv")
surr = pd.read_csv(ROOT / "results" / "summary" / "surrogate_statistical_summary.csv")

print("Main statistics")
print(stats.to_string(index=False))
print("\nSurrogate statistics")
print(surr.to_string(index=False))

expected = {
    "Imagery: task vs rest": 0.0014794698640171469,
    "Execution: task vs rest": 0.3732464119585539,
    "imagery delta vs execution delta": 8.016083848109271e-10,
}
for name, p in expected.items():
    got = float(stats.loc[stats["comparison"].eq(name), "p_value"].iloc[0])
    assert abs(got - p) < 1e-12, (name, got, p)

surrogate_p = float(surr.loc[surr["comparison"].eq("real delta vs phase-randomized surrogate delta"), "p_value"].iloc[0])
assert abs(surrogate_p - 0.0015284602405239852) < 1e-12
print("\nAll reported p-values verified.")
