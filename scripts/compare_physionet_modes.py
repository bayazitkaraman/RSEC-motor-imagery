"""Compare RNT/AHN results between PhysioNet EEGMMI modes.

This script is intended for the SPMB conference paper analysis.
It compares outputs created by run_physionet_eegmmi.py for two modes,
usually motor imagery and motor execution.

Expected input folders:
    outputs_physionet_imagery_50/
        subject_summary.csv
        condition_summary.csv
        task_minus_rest_pair_matrix_mean.csv
    outputs_physionet_execution_50/
        subject_summary.csv
        condition_summary.csv
        task_minus_rest_pair_matrix_mean.csv

Example:
    python compare_physionet_modes.py \
        --imagery-dir outputs_physionet_imagery_50 \
        --execution-dir outputs_physionet_execution_50 \
        --out-dir outputs_spmb_compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def _read_subject_summary(path: Path, mode_label: str) -> pd.DataFrame:
    csv_path = path / "subject_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Please pass a completed output folder.")

    df = pd.read_csv(csv_path)
    required = {"subject", "rest", "task"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    out = df[["subject", "rest", "task"]].copy()
    out["mode"] = mode_label
    out["delta_task_minus_rest"] = out["task"] - out["rest"]
    out["percent_change"] = 100.0 * out["delta_task_minus_rest"] / out["rest"].replace(0, np.nan)
    return out


def _wilcoxon_task_vs_rest(df: pd.DataFrame) -> dict:
    paired = df[["rest", "task"]].dropna()
    stat, p = stats.wilcoxon(paired["task"], paired["rest"])
    delta = paired["task"] - paired["rest"]
    return {
        "comparison": f"{df['mode'].iloc[0]}: task vs rest",
        "test": "Wilcoxon signed-rank",
        "n_subjects": int(len(paired)),
        "statistic": float(stat),
        "p_value": float(p),
        "rest_mean": float(paired["rest"].mean()),
        "task_mean": float(paired["task"].mean()),
        "delta_mean": float(delta.mean()),
        "delta_median": float(delta.median()),
        "delta_std": float(delta.std(ddof=1)),
        "cohens_dz": float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) > 0 else np.nan,
    }


def _compare_deltas(imagery: pd.DataFrame, execution: pd.DataFrame) -> dict:
    merged = imagery[["subject", "delta_task_minus_rest"]].merge(
        execution[["subject", "delta_task_minus_rest"]],
        on="subject",
        suffixes=("_imagery", "_execution"),
    )
    if len(merged) < 2:
        raise ValueError("Need at least two matched subjects to compare deltas.")

    diff = merged["delta_task_minus_rest_imagery"] - merged["delta_task_minus_rest_execution"]
    stat, p = stats.wilcoxon(
        merged["delta_task_minus_rest_imagery"],
        merged["delta_task_minus_rest_execution"],
    )
    return {
        "comparison": "imagery delta vs execution delta",
        "test": "Wilcoxon signed-rank on task-minus-rest deltas",
        "n_subjects": int(len(merged)),
        "statistic": float(stat),
        "p_value": float(p),
        "imagery_delta_mean": float(merged["delta_task_minus_rest_imagery"].mean()),
        "execution_delta_mean": float(merged["delta_task_minus_rest_execution"].mean()),
        "difference_mean_imagery_minus_execution": float(diff.mean()),
        "difference_median_imagery_minus_execution": float(diff.median()),
        "difference_std": float(diff.std(ddof=1)),
        "cohens_dz": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else np.nan,
    }


def _save_density_plot(combined: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for _, row in combined.iterrows():
        rows.append({"subject": row["subject"], "mode": row["mode"], "condition": "Rest", "density": row["rest"]})
        rows.append({"subject": row["subject"], "mode": row["mode"], "condition": "Task", "density": row["task"]})
    long = pd.DataFrame(rows)

    labels = []
    data = []
    for mode in ["Imagery", "Execution"]:
        for condition in ["Rest", "Task"]:
            vals = long[(long["mode"] == mode) & (long["condition"] == condition)]["density"].to_numpy()
            labels.append(f"{mode}\n{condition}")
            data.append(vals)

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("Mean RNT/AHN density per subject")
    ax.set_title("Global RNT/AHN density by condition")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "fig_global_density_boxplot.png", dpi=300)
    plt.close(fig)


def _save_delta_plot(combined: pd.DataFrame, out_dir: Path) -> None:
    data = [
        combined[combined["mode"] == "Imagery"]["delta_task_minus_rest"].to_numpy(),
        combined[combined["mode"] == "Execution"]["delta_task_minus_rest"].to_numpy(),
    ]
    fig = plt.figure(figsize=(6.5, 4.2))
    ax = fig.add_subplot(111)
    ax.boxplot(data, labels=["Imagery", "Execution"], showmeans=True)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylabel("Task minus rest RSEC density")
    ax.set_title("Subject-level task-minus-rest effect")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "fig_delta_task_minus_rest_boxplot.png", dpi=300)
    plt.close(fig)


def _save_paired_line_plot(df: pd.DataFrame, mode: str, out_dir: Path) -> None:
    fig = plt.figure(figsize=(5.5, 4.2))
    ax = fig.add_subplot(111)
    for _, row in df.iterrows():
        ax.plot([0, 1], [row["rest"], row["task"]], marker="o", linewidth=0.8, alpha=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Rest", "Task"])
    ax.set_ylabel("Mean RNT/AHN density per subject")
    ax.set_title(f"Paired subject changes: {mode}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"fig_paired_lines_{mode.lower()}.png", dpi=300)
    plt.close(fig)


def _read_matrix(folder: Path, filename: str) -> pd.DataFrame | None:
    path = folder / filename
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def _save_matrix_heatmap(mat: pd.DataFrame, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    vmax = np.nanmax(np.abs(mat.to_numpy()))
    if vmax == 0 or not np.isfinite(vmax):
        vmax = 1.0
    im = ax.imshow(mat.to_numpy(), aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(mat.index, fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _save_node_participation(mat: pd.DataFrame, title: str, out_path: Path) -> pd.DataFrame:
    vals = mat.to_numpy(dtype=float)
    positive = np.maximum(vals, 0).sum(axis=1)
    negative = np.minimum(vals, 0).sum(axis=1)
    absolute = np.abs(vals).sum(axis=1)
    node_df = pd.DataFrame({
        "channel": mat.index,
        "positive_change_sum": positive,
        "negative_change_sum": negative,
        "absolute_change_sum": absolute,
    }).sort_values("absolute_change_sum", ascending=False)

    top = node_df.head(20).sort_values("absolute_change_sum", ascending=True)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.barh(top["channel"], top["absolute_change_sum"])
    ax.set_xlabel("Sum of absolute pairwise task-minus-rest change")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return node_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare imagery and execution RNT/AHN output folders.")
    parser.add_argument("--imagery-dir", type=Path, required=True)
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_spmb_compare"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    imagery = _read_subject_summary(args.imagery_dir, "Imagery")
    execution = _read_subject_summary(args.execution_dir, "Execution")
    combined = pd.concat([imagery, execution], ignore_index=True)
    combined.to_csv(args.out_dir / "combined_subject_summary.csv", index=False)

    stats_rows = [
        _wilcoxon_task_vs_rest(imagery),
        _wilcoxon_task_vs_rest(execution),
        _compare_deltas(imagery, execution),
    ]
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(args.out_dir / "spmb_statistical_summary.csv", index=False)

    _save_density_plot(combined, args.out_dir)
    _save_delta_plot(combined, args.out_dir)
    _save_paired_line_plot(imagery, "Imagery", args.out_dir)
    _save_paired_line_plot(execution, "Execution", args.out_dir)

    for label, folder in [("imagery", args.imagery_dir), ("execution", args.execution_dir)]:
        mat = _read_matrix(folder, "task_minus_rest_pair_matrix_mean.csv")
        if mat is not None:
            _save_matrix_heatmap(
                mat,
                f"Task-minus-rest pairwise RNT/AHN change: {label}",
                args.out_dir / f"fig_pairwise_delta_heatmap_{label}.png",
            )
            node_df = _save_node_participation(
                mat,
                f"Top electrode participation changes: {label}",
                args.out_dir / f"fig_top_node_changes_{label}.png",
            )
            node_df.to_csv(args.out_dir / f"node_participation_change_{label}.csv", index=False)

    print("\nSPMB comparison statistics:")
    print(stats_df.to_string(index=False))
    print(f"\nSaved SPMB comparison outputs to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
