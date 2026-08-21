"""JONSWAP variance spectrum generator."""

import numpy as np

GRAVITY = 9.81


def jonswap_spectrum(
    frequencies_hz: np.ndarray,
    hs: float,
    tp: float,
    gamma: float = 3.3,
) -> np.ndarray:
    """Return a JONSWAP 1D variance density spectrum S(f) in m^2/Hz.

    Args:
        frequencies_hz: Frequency bins (Hz). Must all be positive.
        hs:             Significant wave height (m).
        tp:             Peak period (s).
        gamma:          Peak enhancement factor (default 3.3 for wind seas).
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    fp = 1.0 / tp

    # Normalisation constant alpha is derived from Hs after computing the
    # un-normalised spectrum shape, so we first build the shape then scale.
    sigma = np.where(frequencies_hz <= fp, 0.07, 0.09)
    r = np.exp(-((frequencies_hz - fp) ** 2) / (2.0 * sigma ** 2 * fp ** 2))

    # Phillips-JONSWAP shape (un-normalised)
    with np.errstate(divide="ignore", invalid="ignore"):
        shape = (
            frequencies_hz ** -5
            * np.exp(-1.25 * (frequencies_hz / fp) ** -4)
            * gamma ** r
        )
    shape = np.where(np.isfinite(shape), shape, 0.0)

    # Scale alpha so that 4*sqrt(m0) == hs
    m0_shape = float(np.trapezoid(shape, x=frequencies_hz))
    if m0_shape <= 0.0:
        return np.zeros_like(frequencies_hz)
    alpha = (hs / 4.0) ** 2 / m0_shape

    return alpha * shape
