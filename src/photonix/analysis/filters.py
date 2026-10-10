"""Pass-band metrics for filters and demultiplexers.

Inputs are wavelength (µm) and linear power transmission. Levels are in dB
relative to the pass-band peak unless stated otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["to_db", "bandwidth", "PassbandMetrics", "passband_metrics", "channel_crosstalk",
           "extinction_ratio_db"]


def to_db(power, floor: float = 1e-30):
    """``10 log10(power)`` with a floor so zeros map to a finite value."""
    return 10.0 * np.log10(np.maximum(np.asarray(power, dtype=float), floor))


def _edges(x, y_db, i0, level):
    def walk(step):
        i = i0
        while 0 <= i + step < len(y_db):
            if y_db[i + step] < level:
                a, b = y_db[i], y_db[i + step]
                return x[i] + (level - a) * (x[i + step] - x[i]) / (b - a)
            i += step
        return None

    return walk(-1), walk(+1)


def bandwidth(wavelength, transmission, level_db: float = -3.0, *, center=None) -> tuple[float, float, float]:
    """Edges and width of the pass band ``level_db`` below its peak.

    The band is grown from the peak (or the sample nearest ``center``) until
    the response first falls below the level on each side.

    Returns
    -------
    (left_edge, right_edge, width) in µm.

    Examples
    --------
    >>> import numpy as np
    >>> wl = np.linspace(1.54, 1.56, 20001)
    >>> T = np.exp(-4 * np.log(2) * ((wl - 1.55) / 0.002) ** 2)    # 2 nm FWHM Gaussian
    >>> round(bandwidth(wl, T, level_db=-10 * np.log10(2))[2] * 1e3, 3)   # half power
    2.0
    >>> round(bandwidth(wl, T)[2] * 1e3, 3)       # -3.000 dB is just inside half power
    1.997
    """
    if level_db >= 0:
        raise ValueError("level_db must be negative (dB below the peak)")
    x = np.asarray(wavelength, dtype=float)
    y = to_db(transmission)
    i0 = int(np.argmax(y)) if center is None else int(np.argmin(np.abs(x - center)))
    left, right = _edges(x, y, i0, y[i0] + level_db)
    if left is None or right is None:
        raise ValueError(f"the response does not fall {level_db} dB below its peak within the sweep")
    return float(left), float(right), float(right - left)


@dataclass(frozen=True)
class PassbandMetrics:
    """Summary of one pass band (wavelengths in µm, levels in dB)."""

    peak_wavelength: float
    insertion_loss_db: float
    bandwidth_1db: float
    bandwidth_3db: float
    bandwidth_20db: float | None
    ripple_db: float
    shape_factor: float | None


def passband_metrics(wavelength, transmission, *, center=None) -> PassbandMetrics:
    """Insertion loss, 1/3/20-dB bandwidths, in-band ripple and shape factor.

    ``ripple_db`` is the peak-to-valley variation inside the 1-dB band;
    ``shape_factor`` is ``BW_20dB / BW_3dB`` (1 is an ideal box). The 20-dB
    figures are ``None`` when the sweep does not reach 20 dB of rejection.

    Examples
    --------
    >>> import numpy as np
    >>> wl = np.linspace(1.54, 1.56, 20001)
    >>> T = 0.9 * np.exp(-4 * np.log(2) * ((wl - 1.55) / 0.002) ** 2)
    >>> m = passband_metrics(wl, T)
    >>> round(m.insertion_loss_db, 3), round(m.bandwidth_3db * 1e3, 3)
    (0.458, 1.997)
    """
    x = np.asarray(wavelength, dtype=float)
    t = np.asarray(transmission, dtype=float)
    y = to_db(t)
    i0 = int(np.argmax(y)) if center is None else int(np.argmin(np.abs(x - center)))
    l1, r1, bw1 = bandwidth(x, t, -1.0, center=x[i0])
    _l3, _r3, bw3 = bandwidth(x, t, -3.0, center=x[i0])
    try:
        bw20: float | None = bandwidth(x, t, -20.0, center=x[i0])[2]
    except ValueError:
        bw20 = None
    band = (x >= l1) & (x <= r1)
    ripple = float(y[band].max() - y[band].min()) if np.any(band) else 0.0
    return PassbandMetrics(
        peak_wavelength=float(x[i0]),
        insertion_loss_db=float(-y[i0]),
        bandwidth_1db=bw1,
        bandwidth_3db=bw3,
        bandwidth_20db=bw20,
        ripple_db=ripple,
        shape_factor=None if bw20 is None else bw20 / bw3,
    )


def channel_crosstalk(wavelength, spectra, centers=None) -> np.ndarray:
    """Crosstalk matrix of a multi-channel filter in dB.

    ``spectra`` is ``(n_channels, n_wavelengths)``. Entry ``[k, j]`` is the
    power of channel ``j`` relative to channel ``k`` at channel ``k``'s
    centre (the diagonal is 0 dB). ``centers`` defaults to each channel's peak.

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> d = px.components.awg_design(n_channels=4, channel_spacing=0.0016)
    >>> wl = np.linspace(1.545, 1.555, 8001)
    >>> s = px.components.awg(wl=wl, n_channels=4, channel_spacing=0.0016)
    >>> T = np.stack([px.to_numpy(px.power(s[("o1", f"o{k + 2}")])) for k in range(4)])
    >>> X = channel_crosstalk(wl, T, d["channel_wavelengths"])
    >>> bool(np.all(np.diag(X) == 0) and X[1, 0] < -20)
    True
    """
    x = np.asarray(wavelength, dtype=float)
    s = np.asarray(spectra, dtype=float)
    if s.ndim != 2 or s.shape[1] != x.size:
        raise ValueError("spectra must have shape (n_channels, n_wavelengths)")
    n = s.shape[0]
    if centers is None:
        idx = np.argmax(s, axis=1)
    else:
        idx = np.array([int(np.argmin(np.abs(x - c))) for c in centers])
    out = np.empty((n, n))
    for k in range(n):
        ref = to_db(s[k, idx[k]])
        out[k] = to_db(s[:, idx[k]]) - ref
    return out


def extinction_ratio_db(transmission) -> float:
    """``10 log10(max / min)`` of a power trace."""
    t = np.asarray(transmission, dtype=float)
    return float(to_db(t.max()) - to_db(t.min()))
