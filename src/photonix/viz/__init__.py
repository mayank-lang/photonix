"""photonix visualization subpackage (matplotlib, Agg-safe).

Every function returns a matplotlib ``Axes`` and never calls ``plt.show()`` so it
composes into larger figures and runs headless.
"""
from __future__ import annotations

from .circuit import plot_netlist
from .layout import plot_cell
from .modes import plot_mode
from .spectrum import plot_phase, plot_spectrum

__all__ = ["plot_spectrum", "plot_phase", "plot_mode", "plot_cell", "plot_netlist"]
