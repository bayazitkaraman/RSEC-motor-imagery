"""
summarize_precision_sensitivity.py

Summarize RNT/AHN rounding-precision sensitivity results.

Expected folders:
  outputs_sensitivity_d2_compare/spmb_statistical_summary.csv
  outputs_sensitivity_d3_compare/spmb_statistical_summary.csv
  ...

Run:
  python summarize_precision_sensitivity.py --base-dir . --decimals 2 3 4 5 --out-dir outputs_precision_sensitivity_summary
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _find_row(df: pd.DataFrame, keyword: str) -> pd.Series:
    mask = df["comparison"].astype(str).str.contains(keyword, case=False, regex=False)
    if not mask.any():
        raise ValueError(f"Could not find comparison containing: {keyword}")
    return df.loc[mask].iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--decimals", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_precision_sensitivity_summary"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for d in args.decimals:
        stats_path = args.base_dir / f"outputs_sensitivity_d{d}_compare" / "spmb_statistical_summary.csv"
        if not stats_path.exists():
            print(f"WARNING: missing {stats_path}")
            continue

        df = pd.read_csv(stats_path)

        imagery = _find_row(df, "Imagery: task vs rest")
        execution = _find_row(df, "Execution: task vs rest")
        delta_diff = _find_row(df, "imagery delta vs execution delta")

        rows.append({
            "decimals": d,
            "n_subjects": int(imagery["n_subjects"]),
            "imagery_delta_mean": float(imagery["delta_mean"]),
            "imagery_p": float(imagery["p_value"]),
            "imagery_dz": float(imagery["cohens_dz"]),
            "execution_delta_mean": float(execution["delta_mean"]),
            "execution_p": float(execution["p_value"]),
            "execution_dz": float(execution["cohens_dz"]),
            "imagery_minus_execution_delta": float(delta_diff["difference_mean_imagery_minus_execution"]),
            "imagery_vs_execution_p": float(delta_diff["p_value"]),
            "imagery_vs_execution_dz": float(delta_diff["cohens_dz"]),
            "direction_stable": float(imagery["delta_mean"]) < 0 and float(delta_diff["difference_mean_imagery_minus_execution"]) < 0,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_dir / "precision_sensitivity_summary.csv", index=False)

    if out_df.empty:
        print("No results found.")
        return

    # Plot 1: deltas across precision values
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(out_df["decimals"], out_df["imagery_delta_mean"], marker="o", label="Imagery")
    ax.plot(out_df["decimals"], out_df["execution_delta_mean"], marker="o", label="Execution")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Rounding precision d")
    ax.set_ylabel("Mean task-minus-rest density")
    ax.set_title("RNT/AHN precision sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "precision_sensitivity_deltas.png", dpi=300)
    fig.savefig(args.out_dir / "precision_sensitivity_deltas.pdf")
    plt.close(fig)

    # Plot 2: p-values across precision values
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(out_df["decimals"], out_df["imagery_p"], marker="o", label="Imagery vs rest")
    ax.plot(out_df["decimals"], out_df["execution_p"], marker="o", label="Execution vs rest")
    ax.plot(out_df["decimals"], out_df["imagery_vs_execution_p"], marker="o", label="Imagery Δ vs execution Δ")
    ax.axhline(0.05, linestyle="--", linewidth=1)
    ax.set_xlabel("Rounding precision d")
    ax.set_ylabel("p-value")
    ax.set_yscale("log")
    ax.set_title("Statistical stability across precision")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "precision_sensitivity_pvalues.png", dpi=300)
    fig.savefig(args.out_dir / "precision_sensitivity_pvalues.pdf")
    plt.close(fig)

    print("\nPrecision sensitivity summary:")
    print(out_df.to_string(index=False))
    print(f"\nSaved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
