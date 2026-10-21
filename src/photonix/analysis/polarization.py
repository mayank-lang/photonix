"""Polarization analysis: Jones/Mueller/Stokes, PDL, extinction, DGD and principal states.

Post-processing on NumPy arrays (``complex128``), applicable to simulated or
measured data. Jones matrices are ``J[..., out, in]`` in the ``(TE0, TM0)``
(or fibre ``(x, y)``) basis.

Conventions
-----------
Stokes parameters are ``S_i = E^H sigma_i E`` with ``sigma_0 = I``,
``sigma_1 = diag(1, -1)``, ``sigma_2 = [[0, 1], [1, 0]]`` and
``sigma_3 = [[0, -j], [j, 0]]``, i.e. ``(|Ex|^2 + |Ey|^2, |Ex|^2 - |Ey|^2,
2 Re(Ex* Ey), 2 Im(Ex* Ey))``; the Mueller matrix of ``J`` is
``M_ij = tr(J^H sigma_i J sigma_j) / 2``, consistent with that definition.

DGD follows Heffner's Jones-matrix eigenanalysis (IEEE PTL 4(9):1066, 1992):
with ``rho_1, rho_2`` the eigenvalues of ``J(omega_2) J(omega_1)^{-1}``,
``DGD = |arg(rho_1 / rho_2)| / |omega_2 - omega_1|``; the output principal
states are the eigenvectors of that matrix and the input ones those of
``J(omega_1)^{-1} J(omega_2)``. The frequency step must keep
``DGD * d_omega < pi``.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "jones_from_sdict",
    "pdl_db",
    "polarization_extinction_db",
    "stokes",
    "jones_to_mueller",
    "dgd",
    "principal_states",
]

_C_UM_PER_PS = 299.792458
_PAULI = np.array([
    [[1, 0], [0, 1]],
    [[1, 0], [0, -1]],
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
], dtype=complex)


def jones_from_sdict(s, in_port: str = "o1", out_port: str = "o2",
                     modes: Sequence[str] = ("TE0", "TM0"),
                     out_modes: Sequence[str] | None = None) -> np.ndarray:
    """Forward transfer matrix ``J[..., out, in]`` between two modal ports of an SDict.

    Missing entries are zero. ``out_modes`` defaults to ``modes``.

    Examples
    --------
    >>> from photonix.components.polarization import polarization_rotator
    >>> J = jones_from_sdict(polarization_rotator())
    >>> J.shape
    (2, 2)
    """
    out_modes = tuple(modes) if out_modes is None else tuple(out_modes)
    rows = []
    for mo in out_modes:
        rows.append([np.asarray(s.get((f"{in_port}:{mi}", f"{out_port}:{mo}"), 0.0), dtype=complex)
                     for mi in modes])
    shape = np.broadcast_shapes(*(v.shape for r in rows for v in r))
    return np.stack([np.stack([np.broadcast_to(v, shape) for v in r], axis=-1) for r in rows], axis=-2)


def pdl_db(jones) -> np.ndarray:
    """Polarization-dependent loss ``10 log10(sigma_max^2 / sigma_min^2)`` in dB.

    Examples
    --------
    >>> import numpy as np
    >>> bool(np.isclose(pdl_db(np.diag([1.0, 0.5])), 20 * np.log10(2.0), rtol=0, atol=1e-12))
    True
    """
    sv = np.linalg.svd(np.asarray(jones, dtype=complex), compute_uv=False)
    return 20.0 * np.log10(sv[..., 0] / sv[..., -1])


def polarization_extinction_db(s, in_port: str, out_port: str, *, mode: str = "TE0",
                               other: str = "TM0") -> np.ndarray:
    """Polarization extinction ratio at ``out_port`` in dB.

    ``10 log10(P(in:mode -> out) / P(in:other -> out))``, with the power at
    ``out_port`` summed over all its modes.

    Examples
    --------
    >>> s = {("o1:TE0", "o2:TE0"): 1.0, ("o1:TM0", "o2:TE0"): 0.01}
    >>> float(polarization_extinction_db(s, "o1", "o2"))
    40.0
    """
    def power(src: str):
        total = 0.0
        for (i, o), v in s.items():
            if i == f"{in_port}:{src}" and o.rpartition(":")[0] == out_port:
                total = total + np.abs(np.asarray(v)) ** 2
        return np.asarray(total, dtype=float)

    return 10.0 * np.log10(power(mode) / power(other))


def stokes(field) -> np.ndarray:
    """Stokes vector ``(S0, S1, S2, S3)`` of Jones vector(s) ``field[..., 2]``.

    Examples
    --------
    >>> import numpy as np
    >>> stokes(np.array([1.0, 1.0j]) / np.sqrt(2)).round(12).tolist()
    [1.0, 0.0, 0.0, 1.0]
    """
    e = np.asarray(field, dtype=complex)
    return np.real(np.einsum("...i,kij,...j->...k", e.conj(), _PAULI, e))


def jones_to_mueller(jones) -> np.ndarray:
    """Mueller matrix (``[..., 4, 4]``) of Jones matrix(es) ``J[..., 2, 2]``.

    Satisfies ``stokes(J @ E) == jones_to_mueller(J) @ stokes(E)``.

    Examples
    --------
    >>> import numpy as np
    >>> M = jones_to_mueller(np.array([[0, 1], [1, 0]]))       # swap x <-> y
    >>> M.round(12).tolist()[1]
    [0.0, -1.0, 0.0, 0.0]
    """
    J = np.asarray(jones, dtype=complex)
    Jh = np.conj(np.swapaxes(J, -1, -2))
    m = np.einsum("...ab,ibc,...cd,jda->...ij", Jh, _PAULI, J, _PAULI)
    return 0.5 * np.real(m)


def _omega(wavelengths) -> np.ndarray:
    wl = np.asarray(wavelengths, dtype=float)
    if wl.ndim != 1 or wl.size < 2:
        raise ValueError("wavelengths must be a 1-D array of at least two samples")
    return 2.0 * np.pi * _C_UM_PER_PS / wl


def dgd(wavelengths, jones) -> np.ndarray:
    """Differential group delay (ps) between consecutive wavelength samples.

    Parameters
    ----------
    wavelengths
        Wavelengths in µm, shape ``(n,)``.
    jones
        Jones matrices ``(n, 2, 2)`` (leading ensemble axes allowed:
        ``(..., n, 2, 2)``).

    Returns
    -------
    numpy.ndarray
        ``(..., n - 1)`` DGD values (ps), one per wavelength interval (Heffner
        1992; module docs).

    Examples
    --------
    >>> import numpy as np
    >>> wl = np.array([1.55, 1.5501])
    >>> w = 2 * np.pi * 299.792458 / wl
    >>> J = np.array([np.diag([np.exp(-0.5j * x * 2.0), np.exp(0.5j * x * 2.0)]) for x in w])   # 2 ps
    >>> round(float(dgd(wl, J)[0]), 9)
    2.0
    """
    w = _omega(wavelengths)
    J = np.asarray(jones, dtype=complex)
    if J.shape[-3] != w.size:
        raise ValueError(f"jones has {J.shape[-3]} samples, wavelengths {w.size}")
    T = J[..., 1:, :, :] @ np.linalg.inv(J[..., :-1, :, :])
    rho = np.linalg.eigvals(T)
    return np.abs(np.angle(rho[..., 0] / rho[..., 1])) / np.abs(np.diff(w))


def principal_states(wavelengths, jones, *, output: bool = False) -> np.ndarray:
    """Principal states of polarization per wavelength interval.

    Returns unit Jones vectors as columns, shape ``(..., n - 1, 2, 2)``: the
    input PSPs (eigenvectors of ``J(w1)^-1 J(w2)``) or, with
    ``output=True``, the output PSPs (eigenvectors of ``J(w2) J(w1)^-1``).

    Examples
    --------
    >>> import numpy as np
    >>> wl = np.array([1.55, 1.5501])
    >>> w = 2 * np.pi * 299.792458 / wl
    >>> J = np.array([np.diag([np.exp(-1j * x), np.exp(1j * x)]) for x in w])
    >>> psp = principal_states(wl, J)[0]
    >>> bool(np.allclose(np.abs(psp), np.eye(2)) or np.allclose(np.abs(psp), np.eye(2)[::-1]))
    True
    """
    _omega(wavelengths)
    J = np.asarray(jones, dtype=complex)
    a, b = J[..., :-1, :, :], J[..., 1:, :, :]
    T = b @ np.linalg.inv(a) if output else np.linalg.inv(a) @ b
    _vals, vecs = np.linalg.eig(T)
    return vecs / np.linalg.norm(vecs, axis=-2, keepdims=True)
