"""Core RSEC shared-energy feature extraction functions."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    low: float = 1.0,
    high: float = 31.0,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter for data shaped channels x times."""
    nyq = sfreq / 2.0
    sos = butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def rnt_energy(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute Riemann stereographic chordal-distance energy with natural log.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Real-valued EEG epoch.
    eps : float
        Numerical regularizer added to the channel normalization denominator
        and to chordal distance inside the natural logarithm. The manuscript
        uses 1e-12 for both, with the input EEG expressed in volts.

    Returns
    -------
    energy : ndarray, shape (n_channels, n_times)
        Channel-wise RSEC energy time series.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be shaped (n_channels, n_times).")

    analytic = hilbert(data, axis=-1)
    max_amp = np.max(np.abs(analytic), axis=-1, keepdims=True)
    eta = analytic / (max_amp + eps)
    rho = np.sqrt(1.0 / (np.abs(eta) ** 2 + 1.0))
    return -np.log(rho + eps)


def rnt_shared_energy_events(
    data: np.ndarray,
    decimals: int = 3,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Detect fixed-precision RSEC shared-energy events.

    Returns
    -------
    density : float
        Global RSEC density, normalized by n_times * n_channel_pairs.
    pair_counts : ndarray, shape (n_channels, n_channels)
        Symmetric pairwise event-count matrix.
    node_participation : ndarray, shape (n_channels,)
        Normalized node participation for each channel. Values are row-wise
        pair counts divided by n_times * (n_channels - 1), so values lie in
        [0, 1] when pair counts are valid event counts.
    """
    if decimals < 0:
        raise ValueError("decimals must be non-negative.")

    energy = rnt_energy(data)
    n_channels, n_times = energy.shape
    if n_channels < 2:
        raise ValueError("At least two channels are required.")
    if n_times < 1:
        raise ValueError("At least one time sample is required.")

    scale = 10 ** decimals
    quantized = np.rint(energy * scale).astype(np.int64)

    pair_counts = np.zeros((n_channels, n_channels), dtype=np.float64)
    upper = np.triu_indices(n_channels, k=1)
    n_events = 0.0

    for t in range(n_times):
        eq = quantized[:, None, t] == quantized[None, :, t]
        vals = eq[upper].astype(np.float64)
        n_events += vals.sum()
        pair_counts[upper] += vals

    pair_counts[(upper[1], upper[0])] = pair_counts[upper]

    n_pairs = n_channels * (n_channels - 1) / 2.0
    density = n_events / (n_times * n_pairs)
    node_participation = pair_counts.sum(axis=1) / (n_times * (n_channels - 1))

    return float(density), pair_counts, node_participation


def analyze_epochs(
    epochs: np.ndarray,
    channel_names: list[str] | None = None,
    decimals: int = 3,
) -> dict:
    """Analyze a stack of epochs shaped epochs x channels x times."""
    epochs = np.asarray(epochs, dtype=float)
    if epochs.ndim != 3:
        raise ValueError("epochs must be shaped (n_epochs, n_channels, n_times).")
    if epochs.shape[0] == 0:
        raise ValueError("At least one epoch is required.")

    densities = []
    pair_matrices = []
    node_scores = []

    for epoch in epochs:
        density, pair_counts, node_part = rnt_shared_energy_events(epoch, decimals=decimals)
        densities.append(density)
        pair_matrices.append(pair_counts)
        node_scores.append(node_part)

    return {
        "densities": np.asarray(densities, dtype=float),
        "pair_matrix_mean": np.mean(pair_matrices, axis=0),
        "node_participation_mean": np.mean(node_scores, axis=0),
        "channel_names": channel_names,
    }
