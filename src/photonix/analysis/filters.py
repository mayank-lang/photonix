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


def _sorted(wavelength, transmission):
    x = np.asarray(wavelength, dtype=float)
    t = np.asarray(transmission, dtype=float)
    if x.ndim != 1 or x.shape != t.shape:
        raise ValueError("wavelength and transmission must be equal-length 1-D arrays")
    order = np.argsort(x)
    return x[order], t[order]


def _peak_index(x, y_db, center) -> int:
    """Index of the pass-band peak: the global maximum, or the local maximum
    reached by hill-climbing from the sample nearest ``center``."""
    if center is None:
        return int(np.argmax(y_db))
    i = int(np.argmin(np.abs(x - center)))
    while True:
        if i > 0 and y_db[i - 1] > y_db[i]:
            i -= 1
        elif i < len(y_db) - 1 and y_db[i + 1] > y_db[i]:
            i += 1
        else:
            return i


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

    The band is grown from the peak until the response first falls below the
    level on each side. With ``center`` the peak is the local maximum reached
    from the sample nearest ``center`` (to pick one channel of a multi-channel
    spectrum). The sweep may be in either order.

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
    x, t = _sorted(wavelength, transmission)
    y = to_db(t)
    i0 = _peak_index(x, y, center)
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

    ``ripple_db`` is the peak-to-valley variation between the outermost local
    maxima inside the 1-dB band -- the in-band ripple with the roll-off
    excluded (0 for a single-peaked pass band). ``shape_factor`` is
    ``BW_20dB / BW_3dB`` (1 is an ideal box). The 20-dB figures are ``None``
    when the sweep does not reach 20 dB of rejection. ``center`` selects the
    pass band whose peak is nearest to it.

    Examples
    --------
    >>> import numpy as np
    >>> wl = np.linspace(1.54, 1.56, 20001)
    >>> T = 0.9 * np.exp(-4 * np.log(2) * ((wl - 1.55) / 0.002) ** 2)
    >>> m = passband_metrics(wl, T)
    >>> round(m.insertion_loss_db, 3), round(m.bandwidth_3db * 1e3, 3)
    (0.458, 1.997)
    """
    x, t = _sorted(wavelength, transmission)
    y = to_db(t)
    i0 = _peak_index(x, y, center)
    l1, r1, bw1 = bandwidth(x, t, -1.0, center=x[i0])
    _l3, _r3, bw3 = bandwidth(x, t, -3.0, center=x[i0])
    try:
        bw20: float | None = bandwidth(x, t, -20.0, center=x[i0])[2]
    except ValueError:
        bw20 = None
    yb = y[(x >= l1) & (x <= r1)]
    ripple = 0.0
    if yb.size >= 3:
        d = np.diff(yb)
        maxima = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0)) + 1
        if maxima.size >= 2:
            ripple = float(yb.max() - yb[maxima[0]:maxima[-1] + 1].min())
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
