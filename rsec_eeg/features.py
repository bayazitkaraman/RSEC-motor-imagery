"""Core RNT shared-energy feature extraction functions."""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt


def bandpass_filter(data: np.ndarray, sfreq: float, low: float = 1.0, high: float = 31.0) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter for data shaped channels x times."""
    nyq = sfreq / 2.0
    sos = butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def rnt_energy(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute Riemann stereographic chordal-distance energy.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Real-valued EEG epoch.
    eps : float
        Small regularizer for epoch-wise channel normalization.

    Returns
    -------
    energy : ndarray, shape (n_channels, n_times)
        Channel-wise RNT energy time series.
    """
    analytic = hilbert(data, axis=-1)
    max_amp = np.max(np.abs(analytic), axis=-1, keepdims=True)
    eta = analytic / (max_amp + eps)
    rho = np.sqrt(1.0 / (np.abs(eta) ** 2 + 1.0))
    return -np.log(rho + eps)


def rnt_shared_energy_events(data: np.ndarray, decimals: int = 3) -> tuple[float, np.ndarray, np.ndarray]:
    """Detect fixed-precision RNT shared-energy events.

    Returns global density, pairwise event-count matrix, and node participation.
    """
    energy = rnt_energy(data)
    rounded = np.round(energy, decimals=decimals)
    n_channels, n_times = rounded.shape
    pair_counts = np.zeros((n_channels, n_channels), dtype=np.float64)
    upper = np.triu_indices(n_channels, k=1)
    n_events = 0.0

    for t in range(n_times):
        eq = rounded[:, None, t] == rounded[None, :, t]
        vals = eq[upper].astype(float)
        n_events += vals.sum()
        pair_counts[upper] += vals

    pair_counts[(upper[1], upper[0])] = pair_counts[upper]
    n_pairs = n_channels * (n_channels - 1) / 2.0
    density = n_events / (n_times * n_pairs)
    node_participation = pair_counts.sum(axis=1)
    return float(density), pair_counts, node_participation


def analyze_epochs(epochs: np.ndarray, channel_names: list[str] | None = None, decimals: int = 3) -> dict:
    """Analyze a stack of epochs shaped epochs x channels x times."""
    densities = []
    pair_matrices = []
    node_scores = []
    for epoch in epochs:
        density, pair_counts, node_part = rnt_shared_energy_events(epoch, decimals=decimals)
        densities.append(density)
        pair_matrices.append(pair_counts)
        node_scores.append(node_part)
    return {
        "densities": np.asarray(densities),
        "pair_matrix_mean": np.mean(pair_matrices, axis=0),
        "node_participation_mean": np.mean(node_scores, axis=0),
        "channel_names": channel_names,
    }
