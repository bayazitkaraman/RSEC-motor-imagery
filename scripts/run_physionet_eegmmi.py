"""Run RNT/AHN analysis on the public PhysioNet EEG Motor Movement/Imagery dataset.

Run from the repository root, for example:
    python .\scripts\run_physionet_eegmmi.py --subjects 1 --mode imagery_lr --max-epochs 10

This version imports from the package folder:
    rnt_ahn_eeg/
        features.py
        stats.py
        surrogate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow the script to be run from either the repository root or the scripts folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci
from scipy import stats

from rsec_eeg.features import analyze_epochs


RUN_SETS = {
    "imagery_lr": [4, 8, 12],       # left vs right hand motor imagery
    "execution_lr": [3, 7, 11],     # left vs right hand motor execution
    "imagery_hf": [6, 10, 14],      # both hands vs both feet motor imagery
    "execution_hf": [5, 9, 13],     # both hands vs both feet motor execution
}


def fixed_length_windows(
    data: np.ndarray,
    sfreq: float,
    window_sec: float,
    step_sec: float | None = None,
) -> np.ndarray:
    """Split continuous EEG data into fixed-length windows.

    Parameters
    ----------
    data:
        Array shaped channels x times.
    sfreq:
        Sampling frequency.
    window_sec:
        Window length in seconds.
    step_sec:
        Step length in seconds. If None, non-overlapping windows are used.

    Returns
    -------
    windows:
        Array shaped windows x channels x times.
    """
    if step_sec is None:
        step_sec = window_sec

    n_window = int(round(window_sec * sfreq))
    n_step = int(round(step_sec * sfreq))

    if n_window <= 0 or n_step <= 0:
        raise ValueError("window_sec and step_sec must be positive.")
    if data.shape[-1] < n_window:
        return np.empty((0, data.shape[0], n_window), dtype=data.dtype)

    starts = range(0, data.shape[-1] - n_window + 1, n_step)
    return np.stack([data[:, start:start + n_window] for start in starts], axis=0)


def analyze_segments(
    segments: np.ndarray,
    *,
    condition: str,
    subject: int,
    channel_names: list[str],
    decimals: int = 3,
):
    """Analyze a stack of EEG segments using RNT shared-energy features."""
    if segments.size == 0:
        raise ValueError(f"No segments available for subject {subject}, condition {condition}.")

    result = analyze_epochs(segments, channel_names=channel_names, decimals=decimals)
    rows = []
    for epoch_index, density in enumerate(result["densities"]):
        rows.append({
            "subject": subject,
            "condition": condition,
            "epoch": epoch_index,
            "ahn_density": float(density),
        })

    return (
        pd.DataFrame(rows),
        result["pair_matrix_mean"],
        result["node_participation_mean"],
    )


def summarize_and_compare(results: pd.DataFrame, rest_label: str = "rest", task_label: str = "task") -> dict:
    """Create condition, subject, and paired-test summaries."""
    condition_summary = (
        results.groupby("condition")["ahn_density"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )

    subject_summary = (
        results.groupby(["subject", "condition"])["ahn_density"]
        .mean()
        .reset_index()
        .pivot(index="subject", columns="condition", values="ahn_density")
        .reset_index()
    )

    paired_test = None
    if rest_label in subject_summary.columns and task_label in subject_summary.columns:
        paired = subject_summary[["subject", rest_label, task_label]].dropna()
        delta = paired[task_label] - paired[rest_label]
        stat, p_value = stats.wilcoxon(paired[task_label], paired[rest_label])
        dz = float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) > 0 else np.nan
        paired_test = {
            "test": "Wilcoxon signed-rank",
            "n_subjects": int(len(paired)),
            "statistic": float(stat),
            "p_value": float(p_value),
            "task_minus_rest_mean": float(delta.mean()),
            "cohens_dz": dz,
        }

    return {
        "condition_summary": condition_summary,
        "subject_summary": subject_summary,
        "paired_test": paired_test,
    }


def _read_raws(subject: int, runs: list[int], data_dir: Path, l_freq: float, h_freq: float):
    files = eegbci.load_data(subject, runs, path=str(data_dir), verbose="ERROR")
    raws = []
    for f in files:
        raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
        eegbci.standardize(raw)
        raw.pick_types(eeg=True)
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, match_case=False, on_missing="ignore")
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose="ERROR")
        raws.append(raw)
    return raws


def _concat(raws):
    if len(raws) == 1:
        return raws[0]
    return mne.concatenate_raws(raws, verbose="ERROR")


def _extract_task_epochs(raw, duration_sec: float, max_epochs: int | None):
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    selected = {name: code for name, code in event_id.items() if name.upper() in {"T1", "T2"}}
    if not selected:
        raise RuntimeError(f"Could not find T1/T2 task annotations. Found: {event_id}")

    sfreq = float(raw.info["sfreq"])
    epochs = mne.Epochs(
        raw,
        events,
        event_id=selected,
        tmin=0.0,
        tmax=duration_sec - 1.0 / sfreq,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )
    if max_epochs is not None:
        epochs = epochs[:max_epochs]
    return epochs.get_data(copy=True), epochs.ch_names, sfreq


def _extract_baseline_windows(raw, duration_sec: float, max_epochs: int | None):
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    windows = fixed_length_windows(data, sfreq=sfreq, window_sec=duration_sec, step_sec=duration_sec)
    if max_epochs is not None:
        windows = windows[:max_epochs]
    return windows, raw.ch_names, sfreq


def run_subject(
    subject: int,
    data_dir: Path,
    mode: str,
    rest_run: int,
    duration_sec: float,
    band: tuple[float, float],
    decimals: int,
    max_epochs: int | None,
):
    l_freq, h_freq = band

    rest_raw = _concat(_read_raws(subject, [rest_run], data_dir, l_freq, h_freq))
    rest_segments, ch_names, sfreq = _extract_baseline_windows(rest_raw, duration_sec, max_epochs)

    task_runs = RUN_SETS[mode]
    task_raw = _concat(_read_raws(subject, task_runs, data_dir, l_freq, h_freq))
    task_segments, task_ch_names, task_sfreq = _extract_task_epochs(task_raw, duration_sec, max_epochs)

    if ch_names != task_ch_names:
        raise RuntimeError("Rest and task channel names differ after preprocessing.")
    if abs(sfreq - task_sfreq) > 1e-6:
        raise RuntimeError("Rest and task sampling frequencies differ.")

    rest_df, rest_pairs, rest_nodes = analyze_segments(
        rest_segments,
        condition="rest",
        subject=subject,
        channel_names=ch_names,
        decimals=decimals,
    )
    task_df, task_pairs, task_nodes = analyze_segments(
        task_segments,
        condition="task",
        subject=subject,
        channel_names=ch_names,
        decimals=decimals,
    )

    return pd.concat([rest_df, task_df], ignore_index=True), ch_names, rest_pairs, task_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="RNT/AHN analysis for PhysioNet EEGMMI.")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--mode", choices=sorted(RUN_SETS), default="imagery_lr")
    parser.add_argument("--rest-run", type=int, default=1, help="1=eyes open, 2=eyes closed")
    parser.add_argument("--data-dir", type=Path, default=Path("mne_data"))
    parser.add_argument("--out-dir", type=Path, default=None, help="Output folder. Defaults to outputs_physionet_<mode>.")
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--band", type=float, nargs=2, default=[1.0, 31.0])
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=20)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"outputs_physionet_{args.mode}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    all_df = []
    all_rest_pairs = []
    all_task_pairs = []
    ch_names = None

    for subject in args.subjects:
        print(f"Analyzing subject {subject}...")
        df, names, rest_pairs, task_pairs = run_subject(
            subject=subject,
            data_dir=args.data_dir,
            mode=args.mode,
            rest_run=args.rest_run,
            duration_sec=args.duration_sec,
            band=(float(args.band[0]), float(args.band[1])),
            decimals=args.decimals,
            max_epochs=args.max_epochs,
        )
        all_df.append(df)
        all_rest_pairs.append(rest_pairs)
        all_task_pairs.append(task_pairs)
        ch_names = names

    results = pd.concat(all_df, ignore_index=True)
    comparison = summarize_and_compare(results, rest_label="rest", task_label="task")

    results.to_csv(args.out_dir / "epoch_level_results.csv", index=False)
    comparison["condition_summary"].to_csv(args.out_dir / "condition_summary.csv", index=False)
    comparison["subject_summary"].to_csv(args.out_dir / "subject_summary.csv", index=False)

    if ch_names is not None:
        rest_pair_mean = np.mean(np.stack(all_rest_pairs, axis=0), axis=0)
        task_pair_mean = np.mean(np.stack(all_task_pairs, axis=0), axis=0)
        pd.DataFrame(rest_pair_mean, index=ch_names, columns=ch_names).to_csv(
            args.out_dir / "rest_pair_matrix_mean.csv"
        )
        pd.DataFrame(task_pair_mean, index=ch_names, columns=ch_names).to_csv(
            args.out_dir / "task_pair_matrix_mean.csv"
        )
        pd.DataFrame(task_pair_mean - rest_pair_mean, index=ch_names, columns=ch_names).to_csv(
            args.out_dir / "task_minus_rest_pair_matrix_mean.csv"
        )

    print("\nCondition summary:")
    print(comparison["condition_summary"].to_string(index=False))
    print("\nPaired test:")
    print(comparison["paired_test"])

    subject_summary = comparison["subject_summary"].set_index("subject")
    if {"rest", "task"}.issubset(subject_summary.columns):
        ax = subject_summary[["rest", "task"]].plot(kind="bar", figsize=(9, 4))
        ax.set_xlabel("Subject")
        ax.set_ylabel("Mean RNT/AHN density")
        ax.set_title(f"PhysioNet EEGMMI RNT/AHN density: {args.mode}")
        plt.tight_layout()
        plt.savefig(args.out_dir / "ahn_density_by_subject.png", dpi=200)
        plt.close()

    print(f"\nSaved outputs to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
