"""Statistical helper functions."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def cohens_dz(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    return float(np.mean(delta) / np.std(delta, ddof=1))


def paired_wilcoxon(task: np.ndarray, rest: np.ndarray) -> dict:
    delta = np.asarray(task, dtype=float) - np.asarray(rest, dtype=float)
    stat, p = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
    return {
        "n_subjects": int(len(delta)),
        "statistic": float(stat),
        "p_value": float(p),
        "rest_mean": float(np.mean(rest)),
        "task_mean": float(np.mean(task)),
        "delta_mean": float(np.mean(delta)),
        "delta_median": float(np.median(delta)),
        "delta_std": float(np.std(delta, ddof=1)),
        "cohens_dz": cohens_dz(delta),
    }
