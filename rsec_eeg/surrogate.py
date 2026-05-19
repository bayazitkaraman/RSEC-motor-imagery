"""Surrogate EEG controls."""
from __future__ import annotations

import numpy as np


def phase_randomize_epoch(epoch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomize one epoch while preserving the amplitude spectrum.

    Parameters
    ----------
    epoch : ndarray, shape (n_channels, n_times)
        Real-valued EEG epoch.
    rng : numpy Generator
        Random number generator.
    """
    fft = np.fft.rfft(epoch, axis=-1)
    amp = np.abs(fft)
    phase = np.angle(fft)
    random_phase = rng.uniform(-np.pi, np.pi, size=phase.shape)
    random_phase[..., 0] = phase[..., 0]
    if epoch.shape[-1] % 2 == 0:
        random_phase[..., -1] = phase[..., -1]
    randomized = amp * np.exp(1j * random_phase)
    return np.fft.irfft(randomized, n=epoch.shape[-1], axis=-1)


def phase_randomize_epochs(epochs: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([phase_randomize_epoch(ep, rng) for ep in epochs], axis=0)
