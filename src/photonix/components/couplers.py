"""Coupler / splitter component models.

Conventions
-----------
A 2x2 coupler exposes four ports: ``o1``/``o2`` on the input (left) side and
``o3``/``o4`` on the output (right) side, arranged::

    o1 ──┐        ┌── o4
         │ couple │
    o2 ──┘        └── o3

with the *through* (bar) paths ``o1->o4`` and ``o2->o3`` and the *cross* paths
``o1->o3`` and ``o2->o4``. The coupler is lossless, reciprocal and unitary; the
cross path carries the canonical 90-degree phase (factor ``-j``).
"""
from __future__ import annotations

from photonix.core.backend import xp
from photonix.core.types import SDict

__all__ = ["directional_coupler", "coupler", "mmi1x2", "mmi2x2"]


def directional_coupler(*, wl=1.55, coupling: float = 0.5) -> SDict:
    """Lossless reciprocal 2x2 directional coupler.

    Parameters
    ----------
    wl : float or array
        Wavelength(s) in µm. Used only for broadcasting; ``coupling`` may itself
        be an array over ``wl`` for a dispersive coupler.
    coupling : float or array
        Power cross-coupling coefficient in [0, 1]. ``coupling=0.5`` is a 50/50
        coupler.

    Returns
    -------
    SDict
        Through amplitude ``t = sqrt(1 - coupling)`` on ``o1<->o4``, ``o2<->o3``;
        cross amplitude ``k = -j*sqrt(coupling)`` on ``o1<->o3``, ``o2<->o4``.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.directional_coupler(coupling=0.5)
    >>> t = px.power(s[("o1","o4")]); k = px.power(s[("o1","o3")])
    >>> abs(float(t + k) - 1.0) < 1e-12
    True
    """
    wl = xp.asarray(wl)
    c = xp.asarray(coupling) * xp.ones_like(xp.real(wl)) if xp.ndim(wl) else xp.asarray(coupling)
    # NOTE: use xp.asarray(..., dtype=complex) rather than .astype: under the
    # NumPy backend a scalar expression like ``-1j * np.float64`` collapses to a
    # plain Python complex (np.float64 subclasses float), which has no .astype.
    t = xp.asarray(xp.sqrt(1.0 - c), dtype=complex)
    k = xp.asarray(-1j * xp.sqrt(c), dtype=complex)
    return {
        ("o1", "o4"): t, ("o4", "o1"): t,
        ("o2", "o3"): t, ("o3", "o2"): t,
        ("o1", "o3"): k, ("o3", "o1"): k,
        ("o2", "o4"): k, ("o4", "o2"): k,
    }


def coupler(*, wl=1.55, coupling: float = 0.5) -> SDict:
    """Alias of :func:`directional_coupler`."""
    return directional_coupler(wl=wl, coupling=coupling)


def mmi1x2(*, wl=1.55, loss_db: float = 0.0) -> SDict:
    """Ideal 1x2 multimode-interference splitter.

    Input ``o1`` splits equally to outputs ``o2`` and ``o3`` (reciprocal).

    Idealization note: a reciprocal, lossless 3-port cannot be matched at all
    ports, so this standard compact model is deliberately *sub-unitary*: the
    antisymmetric (odd) coherent combination of ``o2``/``o3`` inputs is fully
    dissipated even at ``loss_db=0`` (singular values 1, 1, 0). Physically that
    power leaves through the unguided/radiation modes of a real MMI, which a
    matched 3-port cannot represent. Use the EME-backed
    :func:`photonix.em.components.mmi1x2` when back-reflections and the odd-mode
    response matter.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.mmi1x2()
    >>> abs(float(px.power(s[("o1","o2")]) + px.power(s[("o1","o3")])) - 1.0) < 1e-12
    True
    """
    wl = xp.asarray(wl)
    amp = (10.0 ** (-loss_db / 20.0)) / xp.sqrt(2.0)
    a = (amp * xp.ones_like(xp.real(wl))).astype(complex)
    return {
        ("o1", "o2"): a, ("o2", "o1"): a,
        ("o1", "o3"): a, ("o3", "o1"): a,
    }


def mmi2x2(*, wl=1.55, loss_db: float = 0.0) -> SDict:
    """Ideal 2x2 50/50 MMI coupler (ports o1,o2 in; o3,o4 out).

    Implemented as a lossless 50/50 directional coupler with optional excess
    ``loss_db`` applied to every path.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.mmi2x2()
    >>> set(s) >= {("o1","o3"), ("o1","o4")}
    True
    """
    base = directional_coupler(wl=wl, coupling=0.5)
    amp = 10.0 ** (-loss_db / 20.0)
    return {k: amp * v for k, v in base.items()}
