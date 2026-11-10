"""Non-learned breathing traces from pixels alone -- the floor every method must clear.

``snr``: pixels weighted by in-band variance fraction, signed to agree, averaged.
``motion``: motion energy, the mean absolute frame difference over the crop, band-passed
(rectified: it follows how fast the face moves, not which way). None sees the thermistor; polarity is fixed afterwards by
:func:`.common.blind_polarity`.
"""

import numpy as np

from .common import FS, bandpass


def motion_energy(frames: np.ndarray, fs: float = FS) -> np.ndarray:
    """``(T, H, W)`` gray -> band-passed mean absolute frame difference over the crop."""
    x = frames.reshape(len(frames), -1).astype(np.float64)
    energy = np.abs(np.diff(x, axis=0, prepend=x[:1])).mean(1)
    return bandpass(energy - energy.mean(), fs)


def snr_weighted(
    frames: np.ndarray, fs: float = FS, top_fraction: float = 0.1
) -> np.ndarray:
    """``(T, H, W)`` gray -> band-power-weighted, sign-aligned pixel average."""
    raw = frames.reshape(len(frames), -1).astype(np.float64)
    raw = raw - raw.mean(0)
    band = bandpass(raw, fs)
    total = raw.var(0)
    snr = band.var(0) / np.where(total > 0, total, np.inf)
    k = max(1, round(top_fraction * len(snr)))
    chosen = np.argsort(snr)[::-1][:k]
    reference = band[:, chosen[0]]
    signs = np.sign(band[:, chosen].T @ reference)
    signs[signs == 0] = 1.0
    weights = snr[chosen] * signs
    return band[:, chosen] @ weights / np.abs(weights).sum()
