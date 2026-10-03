"""Run a phase-randomized surrogate control for RSEC PhysioNet EEGMMI analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in [PROJECT_ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rsec_eeg.stats import mean_ci_95, wilcoxon_signed_rank
from rsec_eeg.surrogate import phase_randomize_epoch
from run_physionet_eegmmi import (
    RUN_SETS,
    _concat,
    _extract_baseline_windows,
    _extract_task_epochs,
    _read_raws,
    analyze_segments,
    summarize_and_compare,
)


def _value_col(df: pd.DataFrame) -> str:
    if "rsec_density" in df.columns:
        return "rsec_density"
    if "ahn_density" in df.columns:
        return "ahn_density"
    raise ValueError("Expected rsec_density or ahn_density column.")


def _cohens_dz(delta: np.ndarray) -> float:
    sd = np.std(delta, ddof=1)
    return float(np.mean(delta) / sd) if sd > 0 else np.nan


def _make_surrogate_segments(segments: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if segments.ndim != 3 or len(segments) == 0:
        raise ValueError("At least one EEG epoch is required for surrogate generation.")
    return np.stack([phase_randomize_epoch(seg, rng=rng) for seg in segments], axis=0)


def _analyze_condition_pair(
    subject: int,
    condition: str,
    real_segments: np.ndarray,
    surrogate_segments: np.ndarray,
    ch_names: list[str],
    decimals: int,
):
    real_df, _, _ = analyze_segments(
        real_segments,
        condition=condition,
        subject=subject,
        channel_names=ch_names,
        decimals=decimals,
    )
    real_df["data_type"] = "real"

    surrogate_df, _, _ = analyze_segments(
        surrogate_segments,
        condition=condition,
        subject=subject,
        channel_names=ch_names,
        decimals=decimals,
    )
    surrogate_df["data_type"] = "phase_randomized_surrogate"
    return pd.concat([real_df, surrogate_df], ignore_index=True)


def _paired_rest_task(df: pd.DataFrame, data_type: str) -> dict:
    sub = df[df["data_type"] == data_type]
    comparison = summarize_and_compare(sub, rest_label="rest", task_label="task")
    out = comparison["paired_test"] or {}
    out["data_type"] = data_type
    return out


def _compare_real_surrogate_deltas(df: pd.DataFrame) -> dict:
    col = _value_col(df)
    subject_mean = (
        df.groupby(["subject", "data_type", "condition"])[col]
        .mean()
        .reset_index()
        .pivot_table(index=["subject", "data_type"], columns="condition", values=col)
        .reset_index()
    )
    subject_mean["delta_task_minus_rest"] = subject_mean["task"] - subject_mean["rest"]
    real = subject_mean[subject_mean["data_type"] == "real"][["subject", "delta_task_minus_rest"]]
    sur = subject_mean[
        subject_mean["data_type"] == "phase_randomized_surrogate"
    ][["subject", "delta_task_minus_rest"]]
    merged = real.merge(sur, on="subject", suffixes=("_real", "_surrogate"))

    real_delta = merged["delta_task_minus_rest_real"].to_numpy()
    surrogate_delta = merged["delta_task_minus_rest_surrogate"].to_numpy()
    diff = real_delta - surrogate_delta
    stat, p = wilcoxon_signed_rank(real_delta, surrogate_delta)
    ci_low, ci_high = mean_ci_95(diff)

    return {
        "comparison": "real delta vs phase-randomized surrogate delta",
        "test": "Wilcoxon signed-rank on task-minus-rest deltas",
        "n_subjects": int(len(merged)),
        "statistic": float(stat),
        "p_value": float(p),
        "real_delta_mean": float(real_delta.mean()),
        "surrogate_delta_mean": float(surrogate_delta.mean()),
        "difference_mean_real_minus_surrogate": float(diff.mean()),
        "difference_median_real_minus_surrogate": float(np.median(diff)),
        "difference_std": float(np.std(diff, ddof=1)),
        "difference_ci95_low": ci_low,
        "difference_ci95_high": ci_high,
        "cohens_dz": _cohens_dz(diff),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-randomized surrogate control for RSEC.")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--mode", choices=sorted(RUN_SETS), default="imagery_lr")
    parser.add_argument("--rest-run", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("mne_data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_surrogate_imagery"))
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--band", type=float, nargs=2, default=[1.0, 31.0])
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=0, help="Maximum epochs per condition; 0 uses all available epochs.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_epochs <= 0:
        args.max_epochs = None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_df = []

    for subject in args.subjects:
        print(f"Analyzing surrogate subject {subject}...")

        rest_raw = _concat(_read_raws(subject, [args.rest_run], args.data_dir, args.band[0], args.band[1]))
        rest_segments, ch_names, sfreq = _extract_baseline_windows(rest_raw, args.duration_sec, args.max_epochs)

        task_raw = _concat(_read_raws(subject, RUN_SETS[args.mode], args.data_dir, args.band[0], args.band[1]))
        task_segments, task_ch_names, task_sfreq = _extract_task_epochs(task_raw, args.duration_sec, args.max_epochs)

        if ch_names != task_ch_names:
            raise RuntimeError("Rest and task channel names differ after preprocessing.")
        if abs(sfreq - task_sfreq) > 1e-6:
            raise RuntimeError("Rest and task sampling frequencies differ.")

        rest_surrogate = _make_surrogate_segments(rest_segments, rng)
        task_surrogate = _make_surrogate_segments(task_segments, rng)

        all_df.append(_analyze_condition_pair(
            subject, "rest", rest_segments, rest_surrogate, ch_names, args.decimals
        ))
        all_df.append(_analyze_condition_pair(
            subject, "task", task_segments, task_surrogate, ch_names, args.decimals
        ))

    all_results = pd.concat(all_df, ignore_index=True)
    col = _value_col(all_results)
    all_results.to_csv(args.out_dir / "surrogate_epoch_level_results.csv", index=False)

    subject_summary = (
        all_results.groupby(["subject", "data_type", "condition"])[col]
        .mean()
        .reset_index()
        .pivot_table(index=["subject", "data_type"], columns="condition", values=col)
        .reset_index()
    )
    subject_summary["delta_task_minus_rest"] = subject_summary["task"] - subject_summary["rest"]
    subject_summary.to_csv(args.out_dir / "surrogate_subject_summary.csv", index=False)

    stats_rows = [
        _paired_rest_task(all_results, "real"),
        _paired_rest_task(all_results, "phase_randomized_surrogate"),
        _compare_real_surrogate_deltas(all_results),
    ]
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(args.out_dir / "surrogate_statistical_summary.csv", index=False)

    wide = subject_summary.pivot(index="subject", columns="data_type", values="delta_task_minus_rest").reset_index()
    if {"real", "phase_randomized_surrogate"}.issubset(wide.columns):
        values = wide["real"] - wide["phase_randomized_surrogate"]
        p = float(
            stats_df.loc[
                stats_df["comparison"].eq("real delta vs phase-randomized surrogate delta"),
                "p_value",
            ].iloc[0]
        )
        fig, ax = plt.subplots(figsize=(4.8, 3.35))
        ax.boxplot([values], showmeans=True)
        ax.set_xticks([1], ["Real − surrogate"])
        x = 1 + rng.normal(0, 0.035, size=len(values))
        ax.scatter(x, values, s=14, alpha=0.65)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_ylabel("Real delta − surrogate delta", labelpad=10)
        ax.set_title("Phase-randomized surrogate control", pad=8)
        ax.text(
            1.0,
            values.max() - 0.08 * (values.max() - values.min()),
            f"mean = {values.mean():.6f}\np = {p:.5g}",
            ha="center",
            va="top",
            fontsize=9,
        )
        fig.subplots_adjust(left=0.25, right=0.96, top=0.88, bottom=0.18)
        fig.savefig(args.out_dir / "fig3_real_minus_surrogate_delta.png", dpi=300)
        plt.close(fig)

    print("\nSurrogate control statistics:")
    print(stats_df.to_string(index=False))
    print(f"\nSaved surrogate outputs to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
