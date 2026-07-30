"""Plot waveguide mode field profiles."""
from __future__ import annotations

from photonix.core.backend import to_numpy

__all__ = ["plot_mode"]


def plot_mode(field, x=None, y=None, *, ax=None, cmap: str = "RdBu_r"):
    """Heatmap of a 2-D mode field profile.

    Parameters
    ----------
    field
        2-D array of the field component to display.
    x, y
        Optional 1-D coordinate arrays (µm) for the axes extent.
    ax
        Optional matplotlib Axes.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> f = np.exp(-((np.linspace(-2,2,50)[:,None])**2 + np.linspace(-2,2,50)[None,:]**2))
    >>> ax = px.viz.plot_mode(f)
    >>> ax.get_xlabel()
    'x (µm)'
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    f = to_numpy(field)
    if f.ndim != 2 or f.size == 0:
        raise ValueError(f"field must be a non-empty 2-D array, got shape {f.shape}.")
    # Eigenmode fields are generally complex.  A signed field plot displays
    # the real component; callers wanting magnitude can pass abs(field).
    if __import__("numpy").iscomplexobj(f):
        f = f.real
    extent = None
    if x is not None and y is not None:
        x, y = to_numpy(x), to_numpy(y)
        if x.ndim != 1 or y.ndim != 1 or len(x) != f.shape[1] or len(y) != f.shape[0]:
            raise ValueError("x/y coordinate lengths must match the field columns/rows.")
        extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    vmax = float(abs(f).max()) or 1.0
    im = ax.imshow(f, origin="lower", extent=extent, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    return ax
