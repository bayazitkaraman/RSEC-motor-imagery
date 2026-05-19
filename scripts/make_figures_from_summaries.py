"""Recreate paper figures from included subject-level summary CSV files."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
summary = ROOT / "results" / "summary"
fig_dir = ROOT / "results" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

combined = pd.read_csv(summary / "combined_subject_summary.csv")
vals = [combined.query("mode=='Imagery'")["delta_task_minus_rest"].values,
        combined.query("mode=='Execution'")["delta_task_minus_rest"].values]
fig, ax = plt.subplots(figsize=(4.8, 3.4))
ax.boxplot(vals, tick_labels=["Imagery", "Execution"], showmeans=True)
rng = np.random.default_rng(42)
for i, v in enumerate(vals, start=1):
    ax.scatter(i + rng.normal(0, 0.035, len(v)), v, s=10, alpha=0.45)
ax.axhline(0, linestyle="--", linewidth=1)
ax.set_ylabel("Task minus rest RSEC density")
ax.set_title("Subject-level task-minus-rest effect")
fig.subplots_adjust(left=0.20, right=0.96, top=0.88, bottom=0.16)
fig.savefig(fig_dir / "fig2_task_minus_rest_deltas.png", dpi=300)
plt.close(fig)

surr = pd.read_csv(summary / "surrogate_subject_summary.csv")
wide = surr.pivot(index="subject", columns="data_type", values="delta_task_minus_rest").reset_index()
wide["real_minus_surrogate_delta"] = wide["real"] - wide["phase_randomized_surrogate"]
values = wide["real_minus_surrogate_delta"].values
fig, ax = plt.subplots(figsize=(4.8, 3.35))
ax.boxplot([values], tick_labels=["Real − surrogate"], showmeans=True)
x = 1 + rng.normal(0, 0.035, size=len(values))
ax.scatter(x, values, s=14, alpha=0.65)
ax.axhline(0, linestyle="--", linewidth=1)
ax.set_ylabel("Real delta − surrogate delta")
ax.set_title("Phase-randomized surrogate control")
ax.text(1.0, values.max() - 0.08*(values.max()-values.min()),
        f"mean = {values.mean():.6f}\np = 0.00153", ha="center", va="top", fontsize=9)
fig.subplots_adjust(left=0.25, right=0.96, top=0.88, bottom=0.18)
fig.savefig(fig_dir / "fig3_real_minus_surrogate_delta.png", dpi=300)
plt.close(fig)

node = pd.read_csv(summary / "node_participation_change_imagery.csv")
top = node.sort_values("absolute_change_sum", ascending=False).head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(4.8, 3.4))
ax.barh(top["channel"], top["absolute_change_sum"])
ax.set_xlabel("Sum of absolute pairwise change")
ax.set_ylabel("Electrode")
ax.set_title("Top node participation changes: imagery")
fig.subplots_adjust(left=0.18, right=0.96, top=0.88, bottom=0.18)
fig.savefig(fig_dir / "fig4_top_node_participation_imagery.png", dpi=300)
plt.close(fig)

print(f"Figures saved to {fig_dir}")
