"""Plot transmission/phase spectra from scattering dictionaries."""
from __future__ import annotations

from collections.abc import Iterable

from photonix.core.backend import to_numpy
from photonix.core.sparams import phase as _phase
from photonix.core.sparams import power
from photonix.core.types import PortPair, SDict

__all__ = ["plot_spectrum", "plot_phase"]


def _get_ax(ax):
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    return ax


def plot_spectrum(
    sdict: SDict,
    wl,
    ports: Iterable[PortPair] | None = None,
    *,
    ax=None,
    unit: str = "dB",
):
    """Plot the power transmission of one or more port pairs vs wavelength.

    Parameters
    ----------
    sdict
        The scattering dictionary to plot.
    wl
        Wavelength array (µm), matching the array length of the coefficients.
    ports
        Iterable of ``(in, out)`` pairs. Defaults to every key in ``sdict``.
    unit
        ``"dB"`` (10*log10) or ``"linear"``.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> import photonix as px
    >>> wl = px.linspace(1.5, 1.6, 201)
    >>> s = px.circuit.mzi(delta_length=40.0)(wl=wl)
    >>> ax = px.viz.plot_spectrum(s, wl, [("in0", "out0")])
    >>> ax.get_xlabel()
    'Wavelength (µm)'
    """
    ax = _get_ax(ax)
    wl_np = to_numpy(wl)
    if ports is None:
        ports = list(sdict)
    for p in ports:
        t = to_numpy(power(sdict[p]))
        y = 10.0 * __import__("numpy").log10(t + 1e-12) if unit == "dB" else t
        ax.plot(wl_np, y, label=f"{p[0]}→{p[1]}")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Transmission (dB)" if unit == "dB" else "Transmission")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_phase(sdict: SDict, wl, ports: Iterable[PortPair] | None = None, *, ax=None):
    """Plot the phase (radians) of one or more port pairs vs wavelength."""
    ax = _get_ax(ax)
    wl_np = to_numpy(wl)
    if ports is None:
        ports = list(sdict)
    for p in ports:
        ax.plot(wl_np, to_numpy(_phase(sdict[p])), label=f"{p[0]}→{p[1]}")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Phase (rad)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax
