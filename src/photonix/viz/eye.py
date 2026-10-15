"""Eye-diagram plotting (matplotlib; never switches the backend or calls ``plt.show``)."""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["plot_eye"]


def plot_eye(eye: Any, *, ax=None, max_traces: int = 400, color: str = "C0", alpha: float = 0.08,
             show_stats: bool = True, scale: float = 1.0, unit: str | None = None):
    """Plot a :class:`photonix.timedomain.EyeDiagram`.

    Draws up to ``max_traces`` folded traces against time in unit intervals,
    marks the sampling instant(s) and decision threshold(s) and, with
    ``show_stats``, annotates Q, eye height and extinction ratio. ``scale``
    multiplies the signal for display (e.g. ``1e3`` with ``unit="mA"``).

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> from photonix.timedomain import drive_waveform, eye_diagram, prbs
    >>> bits = prbs(7, 254)
    >>> eye = eye_diagram(drive_waveform(bits, baud=10e9, samples_per_symbol=16, rise_time_20_80_s=25e-12),
    ...                   baud=10e9, symbols=bits)
    >>> ax = plot_eye(eye)
    >>> ax.get_xlabel()
    'Time (UI)'
    """
    if ax is None:
        # pyplot only when a figure must be created; the user's backend is never switched (matplotlib picks
        # a non-interactive one by itself on a headless machine).
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(6, 4))
    traces = np.asarray(eye.traces) * scale
    if traces.shape[0] > max_traces:
        pick = np.linspace(0, traces.shape[0] - 1, max_traces).astype(int)
        traces = traces[pick]
    t = np.asarray(eye.t_ui)
    ax.plot(t, traces.T, color=color, alpha=alpha, lw=0.8)
    for k in range(int(np.floor(t.max())) + 1):
        if t.min() <= k + 0.5 <= t.max():
            ax.axvline(k + 0.5, color="0.4", lw=0.6, ls=":")
    for th in np.atleast_1d(eye.thresholds):
        ax.axhline(float(th) * scale, color="0.35", lw=0.7, ls="--")
    label = unit if unit is not None else eye.unit
    ax.set_xlabel("Time (UI)")
    ax.set_ylabel(f"Signal ({label})" if label else "Signal")
    if show_stats:
        txt = f"Q = {eye.q_factor:.2f}\nheight = {eye.height * scale:.3g} {label}".rstrip()
        er = eye.extinction_ratio_db
        if np.isfinite(er):
            txt += f"\nER = {er:.2f} dB"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=8,
                bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8})
    ax.grid(True, alpha=0.3)
    return ax
