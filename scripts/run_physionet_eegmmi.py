"""Run RSEC analysis on the public PhysioNet EEG Motor Movement/Imagery dataset.

Example:
    python scripts/run_physionet_eegmmi.py --subjects 1 --mode imagery_lr --max-epochs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci

from rsec_eeg.features import analyze_epochs
from rsec_eeg.stats import paired_wilcoxon


RUN_SETS = {
    "imagery_lr": [4, 8, 12],
    "execution_lr": [3, 7, 11],
    "imagery_hf": [6, 10, 14],
    "execution_hf": [5, 9, 13],
}


def fixed_length_windows(
    data: np.ndarray,
    sfreq: float,
    window_sec: float,
    step_sec: float | None = None,
) -> np.ndarray:
    """Split continuous EEG data into fixed-length windows."""
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
    """Analyze a stack of EEG segments using RSEC shared-energy features."""
    if segments.size == 0:
        raise ValueError(f"No segments available for subject {subject}, condition {condition}.")

    result = analyze_epochs(segments, channel_names=channel_names, decimals=decimals)
    rows = []
    for epoch_index, density in enumerate(result["densities"]):
        rows.append({
            "subject": subject,
            "condition": condition,
            "epoch": epoch_index,
            "rsec_density": float(density),
        })

    return (
        pd.DataFrame(rows),
        result["pair_matrix_mean"],
        result["node_participation_mean"],
    )


def summarize_and_compare(
    results: pd.DataFrame,
    rest_label: str = "rest",
    task_label: str = "task",
) -> dict:
    """Create condition, subject, and paired-test summaries."""
    value_col = "rsec_density"
    if value_col not in results.columns and "ahn_density" in results.columns:
        value_col = "ahn_density"

    condition_summary = (
        results.groupby("condition")[value_col]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )

    subject_summary = (
        results.groupby(["subject", "condition"])[value_col]
        .mean()
        .reset_index()
        .pivot(index="subject", columns="condition", values=value_col)
        .reset_index()
    )

    paired_test = None
    if rest_label in subject_summary.columns and task_label in subject_summary.columns:
        paired = subject_summary[["subject", rest_label, task_label]].dropna()
        paired_test = paired_wilcoxon(
            task=paired[task_label].to_numpy(),
            rest=paired[rest_label].to_numpy(),
        )
        paired_test["task_minus_rest_mean"] = paired_test["delta_mean"]

    return {
        "condition_summary": condition_summary,
        "subject_summary": subject_summary,
        "paired_test": paired_test,
    }


def _read_raws(subject: int, runs: list[int], data_dir: Path, l_freq: float, h_freq: float):
    files = eegbci.load_data(subject, runs, path=str(data_dir), update_path=False, verbose="ERROR")
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
    if max_epochs is not None and max_epochs > 0:
        epochs = epochs[:max_epochs]
    # Match retained events to annotation intervals, including cropped/concatenated raws.
    annotations = raw.annotations
    starts = raw.time_as_index(annotations.onset, use_rounding=True, origin=annotations.orig_time)
    if annotations.orig_time is not None:
        starts += raw.first_samp
    bounds = {(int(start), selected[label]): (float(onset), float(length))
              for start, label, onset, length in zip(
                  starts, annotations.description, annotations.onset, annotations.duration)
              if label in selected}
    tolerance = 0.5 / sfreq + 1e-10
    for event in epochs.events:
        onset, length = bounds[(int(event[0]), int(event[2]))]
        offset = event[0] / sfreq - onset
        if offset < -tolerance or offset + duration_sec > length + tolerance:
            raise ValueError("Task epoch extends beyond its T1/T2 annotation interval.")
    return epochs.get_data(copy=True), epochs.ch_names, sfreq


def _extract_baseline_windows(raw, duration_sec: float, max_epochs: int | None):
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    windows = fixed_length_windows(data, sfreq=sfreq, window_sec=duration_sec, step_sec=duration_sec)
    if max_epochs is not None and max_epochs > 0:
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

    return (
        pd.concat([rest_df, task_df], ignore_index=True),
        ch_names,
        rest_pairs,
        task_pairs,
        rest_nodes,
        task_nodes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RSEC analysis for PhysioNet EEGMMI.")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--mode", choices=sorted(RUN_SETS), default="imagery_lr")
    parser.add_argument("--rest-run", type=int, default=1, help="1=eyes open, 2=eyes closed")
    parser.add_argument("--data-dir", type=Path, default=Path("mne_data"))
    parser.add_argument("--out-dir", type=Path, default=None, help="Output folder. Defaults to outputs_physionet_<mode>.")
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--band", type=float, nargs=2, default=[1.0, 31.0])
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=0,
        help="Maximum epochs per condition. Use 0 to analyze all available epochs."
    )
    args = parser.parse_args()

    if args.max_epochs <= 0:
        args.max_epochs = None

    if args.out_dir is None:
        args.out_dir = Path(f"outputs_physionet_{args.mode}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    all_df = []
    all_rest_pairs = []
    all_task_pairs = []
    all_rest_nodes = []
    all_task_nodes = []
    node_rows = []
    ch_names = None

    for subject in args.subjects:
        print(f"Analyzing subject {subject}...")
        df, names, rest_pairs, task_pairs, rest_nodes, task_nodes = run_subject(
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
        all_rest_nodes.append(rest_nodes)
        all_task_nodes.append(task_nodes)
        ch_names = names

        node_rows.append(pd.DataFrame({
            "subject": subject,
            "channel": names,
            "rest_node_participation_mean": rest_nodes,
            "task_node_participation_mean": task_nodes,
            "delta_node_participation_mean": task_nodes - rest_nodes,
            "abs_delta_node_participation_mean": np.abs(task_nodes - rest_nodes),
        }))

    results = pd.concat(all_df, ignore_index=True)
    comparison = summarize_and_compare(results, rest_label="rest", task_label="task")

    results.to_csv(args.out_dir / "epoch_level_results.csv", index=False)
    comparison["condition_summary"].to_csv(args.out_dir / "condition_summary.csv", index=False)
    comparison["subject_summary"].to_csv(args.out_dir / "subject_summary.csv", index=False)

    if comparison["paired_test"] is not None:
        pd.DataFrame([comparison["paired_test"]]).to_csv(args.out_dir / "paired_test.csv", index=False)

    if node_rows:
        pd.concat(node_rows, ignore_index=True).to_csv(
            args.out_dir / "node_participation_by_subject.csv",
            index=False,
        )

    if ch_names is not None:
        rest_pair_mean = np.mean(np.stack(all_rest_pairs, axis=0), axis=0)
        task_pair_mean = np.mean(np.stack(all_task_pairs, axis=0), axis=0)
        rest_node_mean = np.mean(np.stack(all_rest_nodes, axis=0), axis=0)
        task_node_mean = np.mean(np.stack(all_task_nodes, axis=0), axis=0)

        pd.DataFrame(rest_pair_mean, index=ch_names, columns=ch_names).to_csv(
            args.out_dir / "rest_pair_matrix_mean.csv"
        )
        pd.DataFrame(task_pair_mean, index=ch_names, columns=ch_names).to_csv(
            args.out_dir / "task_pair_matrix_mean.csv"
        )
        pd.DataFrame(task_pair_mean - rest_pair_mean, index=ch_names, columns=ch_names).to_csv(
            args.out_dir / "task_minus_rest_pair_matrix_mean.csv"
        )
        pd.DataFrame({
            "channel": ch_names,
            "rest_node_participation_mean": rest_node_mean,
            "task_node_participation_mean": task_node_mean,
            "delta_node_participation_mean": task_node_mean - rest_node_mean,
            "abs_delta_node_participation_mean": np.abs(task_node_mean - rest_node_mean),
        }).to_csv(args.out_dir / "node_participation_mean.csv", index=False)

    print("\nCondition summary:")
    print(comparison["condition_summary"].to_string(index=False))
    print("\nPaired test:")
    print(comparison["paired_test"])

    subject_summary = comparison["subject_summary"].set_index("subject")
    if {"rest", "task"}.issubset(subject_summary.columns):
        ax = subject_summary[["rest", "task"]].plot(kind="bar", figsize=(9, 4))
        ax.set_xlabel("Subject")
        ax.set_ylabel("Mean RSEC density")
        ax.set_title(f"PhysioNet EEGMMI RSEC density: {args.mode}")
        plt.tight_layout()
        plt.savefig(args.out_dir / "rsec_density_by_subject.png", dpi=200)
        plt.close()

    print(f"\nSaved outputs to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
