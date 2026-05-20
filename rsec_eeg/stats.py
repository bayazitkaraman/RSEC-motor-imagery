"""Statistical helper functions for RSEC analyses."""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import wilcoxon


def cohens_dz(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    sd = np.std(delta, ddof=1)
    return float(np.mean(delta) / sd) if sd > 0 else np.nan


def mean_ci_95(delta: np.ndarray) -> tuple[float, float]:
    """Return the two-sided 95% t confidence interval for the mean delta."""
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) < 2:
        return np.nan, np.nan
    se = stats.sem(delta)
    half_width = stats.t.ppf(0.975, df=len(delta) - 1) * se
    mean = float(np.mean(delta))
    return float(mean - half_width), float(mean + half_width)


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray | None = None) -> tuple[float, float]:
    """Wilcoxon signed-rank test with explicit two-sided settings.

    Uses SciPy's asymptotic method when available and falls back to older
    SciPy-compatible method names when needed.
    """
    kwargs = {
        "zero_method": "wilcox",
        "alternative": "two-sided",
    }
    try:
        return wilcoxon(x, y, method="asymptotic", **kwargs)
    except ValueError:
        try:
            return wilcoxon(x, y, method="approx", **kwargs)
        except TypeError:
            return wilcoxon(x, y, **kwargs)
    except TypeError:
        try:
            return wilcoxon(x, y, method="approx", **kwargs)
        except TypeError:
            return wilcoxon(x, y, **kwargs)


def paired_wilcoxon(task: np.ndarray, rest: np.ndarray) -> dict:
    task = np.asarray(task, dtype=float)
    rest = np.asarray(rest, dtype=float)
    delta = task - rest
    stat, p = wilcoxon_signed_rank(task, rest)
    ci_low, ci_high = mean_ci_95(delta)

    return {
        "test": "Wilcoxon signed-rank",
        "n_subjects": int(len(delta)),
        "statistic": float(stat),
        "p_value": float(p),
        "rest_mean": float(np.mean(rest)),
        "task_mean": float(np.mean(task)),
        "delta_mean": float(np.mean(delta)),
        "delta_median": float(np.median(delta)),
        "delta_std": float(np.std(delta, ddof=1)),
        "delta_ci95_low": ci_low,
        "delta_ci95_high": ci_high,
        "cohens_dz": cohens_dz(delta),
    }


def holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Holm-adjust p-values while preserving the original order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0

    for rank, idx in enumerate(order):
        adjusted = min((m - rank) * p[idx], 1.0)
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = running_max

    out = np.empty(m, dtype=float)
    out[order] = adjusted_sorted
    return out
