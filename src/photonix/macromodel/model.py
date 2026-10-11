"""Rational (pole-residue) macromodels of S-parameters and their realizations.

A :class:`RationalModel` represents every S-parameter entry as a delayed
rational function of the baseband angular frequency ``Omega = omega - omega_c``
about an optical carrier ``omega_c = 2 pi carrier_hz``::

    S[out, in](Omega) = exp(-1j Omega tau[out, in])
                        * (sum_k R_k[out, in] / (1j Omega - p_k) + D[out, in])

All entries share one set of poles ``p_k`` (``Re p_k < 0``). The sign
conventions are Photonix's: time dependence ``exp(+1j omega t)`` and propagation
``exp(-1j beta L)``, so a causal, stable impulse response has left-half-plane
poles and a pure delay is ``exp(-1j Omega tau)``. The constant carrier phase
``exp(-1j omega_c tau)`` of a delay is absorbed into ``R_k`` and ``D``.

An optical baseband impulse response is complex, so poles and residues need not
come in conjugate pairs. A *real* (RF) system is the special case
``carrier_hz == 0`` with conjugate-symmetric poles/residues and a real ``D``.

Exports
-------
* ``RationalModel.__call__`` -- an SDict model ``f(*, wl, **params)`` built from
  backend (``xp``) arithmetic, so it is ``jit``/``grad``-compatible and drops
  into :func:`photonix.circuit.circuit_from_netlist`.
* :meth:`RationalModel.to_statespace` -- Gilbert realization ``(A, B, C, D)``.
* :meth:`RationalModel.discretize` -- zero/first-order-hold discrete model.
* :meth:`RationalModel.simulate` -- exact time-domain response (including
  fractional delays) for piecewise-linear input envelopes.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from photonix.core.backend import xp
from photonix.core.constants import C0_UM_S
from photonix.core.types import SDict

from ._common import dense_data, wavelengths_to_hz

__all__ = ["RationalModel", "StateSpace", "DiscreteStateSpace"]

_FORMAT = "photonix.rational-model"
_FORMAT_VERSION = 1


# --------------------------------------------------------------------------- #
# Small numerical kernels
# --------------------------------------------------------------------------- #
def _phi1(x: np.ndarray) -> np.ndarray:
    """``(exp(x) - 1) / x`` without cancellation near ``x = 0``."""
    x = np.asarray(x, dtype=complex)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-2
    xs = x[small]
    out[small] = 1 + xs / 2 * (1 + xs / 3 * (1 + xs / 4 * (1 + xs / 5 * (1 + xs / 6))))
    xl = x[~small]
    out[~small] = np.expm1(xl) / xl
    return out


def _phi2(x: np.ndarray) -> np.ndarray:
    """``(exp(x) - 1 - x) / x**2`` without cancellation near ``x = 0``."""
    x = np.asarray(x, dtype=complex)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-2
    xs = x[small]
    out[small] = 0.5 * (1 + xs / 3 * (1 + xs / 4 * (1 + xs / 5 * (1 + xs / 6 * (1 + xs / 7)))))
    xl = x[~small]
    out[~small] = (np.expm1(xl) - xl) / (xl * xl)
    return out


def _check_ports(ports: Sequence[str], n: int) -> tuple[str, ...]:
    names = tuple(str(p) for p in ports)
    if len(names) != n or len(set(names)) != n or any(not p for p in names):
        raise ValueError(f"ports must contain {n} unique, non-empty names, got {names!r}")
    return names


def _delayed_core_response(omega, poles, residues, d, delays):
    """NumPy evaluation of the delayed pole-residue form at baseband ``omega``."""
    omega = np.asarray(omega, dtype=float).reshape(-1)
    core = np.broadcast_to(d, (omega.size, *d.shape)).astype(complex)
    if poles.size:
        basis = 1.0 / (1j * omega[:, None] - poles[None, :])
        core = core + np.einsum("nk,kij->nij", basis, residues)
    return core * np.exp(-1j * omega[:, None, None] * delays[None, :, :])


def _factor(r: np.ndarray, minimal: bool, rmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Factor a residue matrix ``r = C_k @ B_k`` (rank-revealing SVD if ``minimal``)."""
    n = r.shape[0]
    if not minimal:
        return np.eye(n, dtype=r.dtype), r.copy()
    u, sv, vh = np.linalg.svd(r)
    keep = sv > 1e-13 * max(rmax, 1e-300)
    return vh[keep], u[:, keep] * sv[keep]


# --------------------------------------------------------------------------- #
# State-space containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StateSpace:
    """Continuous-time realization of a :class:`RationalModel`'s rational core.

    ``x' = A x + B u``, ``y = C x + D u`` reproduces the *delay-free* core
    ``sum_k R_k / (s - p_k) + D`` with ``s = 1j Omega`` (rad/s). The full model
    applies ``delays[out, in]`` (seconds) to each input-output path. Input and
    output vectors follow ``ports``; arrays are NumPy (complex unless the
    realization was requested in real form for a real system).
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    delays: np.ndarray
    carrier_hz: float
    ports: tuple[str, ...]

    @property
    def n_states(self) -> int:
        """Number of states (order of the realization)."""
        return int(self.A.shape[0])

    def response(self, freq_hz) -> np.ndarray:
        """Evaluate ``S[:, out, in]`` (with delays) at absolute frequencies in Hz."""
        f = np.asarray(freq_hz, dtype=float).reshape(-1)
        omega = 2.0 * np.pi * (f - self.carrier_hz)
        n = self.D.shape[0]
        out = np.empty((omega.size, n, n), dtype=complex)
        eye = np.eye(self.n_states)
        for k, w in enumerate(omega):
            core = self.C @ np.linalg.solve(1j * w * eye - self.A, self.B) if self.n_states else 0.0
            out[k] = core + self.D
        return out * np.exp(-1j * omega[:, None, None] * self.delays[None])


@dataclass(frozen=True)
class DiscreteStateSpace:
    """Discrete-time equivalent of a rational core sampled every ``dt`` seconds.

    ``x[k+1] = Ad x[k] + Bd u[k]``, ``y[k] = Cd x[k] + Dd u[k]``. With
    ``method="zoh"`` the recursion is exact for inputs held constant between
    samples; with ``method="foh"`` it is exact for piecewise-linear inputs
    (the input ramps from zero at ``t = -dt``). Delays are *not* applied by the
    recursion; ``delays / dt`` is available as :attr:`delay_samples`. Use
    :meth:`RationalModel.simulate` for an exact delayed simulation.
    """

    Ad: np.ndarray
    Bd: np.ndarray
    Cd: np.ndarray
    Dd: np.ndarray
    dt: float
    method: str
    delays: np.ndarray
    carrier_hz: float
    ports: tuple[str, ...]

    @property
    def delay_samples(self) -> np.ndarray:
        """Path delays in (fractional) samples, ``delays / dt``."""
        return self.delays / self.dt

    def simulate(self, u) -> np.ndarray:
        """Run the delay-free recursion for input samples ``u`` of shape ``(K, n)``."""
        u = np.asarray(u, dtype=complex)
        if u.ndim == 1:
            u = u[:, None]
        if u.ndim != 2 or u.shape[1] != self.Bd.shape[1]:
            raise ValueError(f"u must have shape (K, {self.Bd.shape[1]})")
        n_states = self.Ad.shape[0]
        y = u @ self.Dd.T
        if n_states == 0:
            return y
        from scipy.signal import lfilter

        v = u @ self.Bd.T  # (K, n_states) state drive
        diagonal = np.allclose(self.Ad, np.diag(np.diag(self.Ad)), rtol=0.0, atol=0.0)
        if diagonal:
            lam = np.diag(self.Ad)
            x = np.empty_like(v)
            for s in range(n_states):
                x[:, s] = lfilter([0.0, 1.0], [1.0, -lam[s]], v[:, s])
        else:
            x = np.zeros_like(v)
            state = np.zeros(n_states, dtype=complex)
            for k in range(v.shape[0]):
                x[k] = state
                state = self.Ad @ state + v[k]
        return y + x @ self.Cd.T

    def response(self, freq_hz) -> np.ndarray:
        """Discrete-time frequency response with ``z = exp(1j Omega dt)`` (delays applied)."""
        f = np.asarray(freq_hz, dtype=float).reshape(-1)
        omega = 2.0 * np.pi * (f - self.carrier_hz)
        n = self.Dd.shape[0]
        out = np.empty((omega.size, n, n), dtype=complex)
        eye = np.eye(self.Ad.shape[0])
        for k, w in enumerate(omega):
            z = np.exp(1j * w * self.dt)
            core = self.Cd @ np.linalg.solve(z * eye - self.Ad, self.Bd) if self.Ad.size else 0.0
            out[k] = core + self.Dd
        return out * np.exp(-1j * omega[:, None, None] * self.delays[None])


# --------------------------------------------------------------------------- #
# Rational model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class RationalModel:
    """Delayed pole-residue macromodel of an ``n``-port S-matrix.

    Parameters
    ----------
    ports
        Port names; matrix index ``i`` is ``ports[i]``.
    carrier_hz
        Carrier frequency in Hz (``0`` for a real/RF model in absolute frequency).
    poles
        ``(P,)`` complex poles in rad/s of baseband angular frequency, all with
        ``Re p < 0``.
    residues
        ``(P, n, n)`` complex residue matrices in rad/s, ``residues[k, out, in]``.
    d
        ``(n, n)`` complex direct term ``D[out, in]`` (the response far from
        every pole).
    delays
        ``(n, n)`` non-negative path delays in seconds, ``delays[out, in]``
        (default zero).
    metadata
        JSON-serializable provenance (fit band, error, settings).

    Notes
    -----
    Evidence: the representation is exact by construction; the accuracy of a
    *fitted* model is whatever :func:`~photonix.macromodel.vector_fit` reports
    in ``metadata`` and is only established inside the fitted band.

    Instances are immutable and hash/compare by identity (array fields have no
    meaningful ``==``), so a model can serve as a dictionary key or a static
    argument of ``jax.jit``.

    Examples
    --------
    >>> import numpy as np
    >>> m = RationalModel(("o1", "o2"), 193.4e12, np.array([-1e11 + 0j]),
    ...                   np.array([[[0, 1e11], [1e11, 0]]], dtype=complex), np.zeros((2, 2)))
    >>> s = m(wl=1.55)
    >>> sorted(s)[:2]
    [('o1', 'o1'), ('o1', 'o2')]
    >>> round(abs(complex(m.response([193.4e12])[0, 1, 0])), 12)   # on resonance: R / |p| = 1
    1.0
    """

    ports: tuple[str, ...]
    carrier_hz: float
    poles: np.ndarray
    residues: np.ndarray
    d: np.ndarray
    delays: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        poles = np.asarray(self.poles, dtype=complex).reshape(-1)
        d = np.asarray(self.d, dtype=complex)
        if d.ndim != 2 or d.shape[0] != d.shape[1]:
            raise ValueError("d must be a square (n, n) matrix")
        n = d.shape[0]
        ports = _check_ports(self.ports, n)
        residues = np.asarray(self.residues, dtype=complex)
        if residues.shape != (poles.size, n, n):
            raise ValueError(f"residues must have shape {(poles.size, n, n)}, got {residues.shape}")
        given = self.delays
        delays = np.zeros((n, n)) if given is None or np.size(given) == 0 else np.asarray(given, dtype=float)
        if delays.shape != (n, n):
            raise ValueError(f"delays must have shape {(n, n)}, got {delays.shape}")
        carrier = float(self.carrier_hz)
        for name, arr in (("poles", poles), ("residues", residues), ("d", d)):
            if not (np.all(np.isfinite(arr.real)) and np.all(np.isfinite(arr.imag))):
                raise ValueError(f"{name} must be finite")
        if not np.all(np.isfinite(delays)) or np.any(delays < 0):
            raise ValueError("delays must be finite and non-negative (a negative delay is non-causal)")
        if not np.isfinite(carrier) or carrier < 0:
            raise ValueError("carrier_hz must be finite and non-negative")
        if np.any(poles.real >= 0):
            raise ValueError("all poles must be strictly stable (Re p < 0)")
        metadata = dict(self.metadata)
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-serializable") from exc
        object.__setattr__(self, "ports", ports)
        object.__setattr__(self, "carrier_hz", carrier)
        object.__setattr__(self, "poles", poles)
        object.__setattr__(self, "residues", residues)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "delays", delays)
        object.__setattr__(self, "metadata", metadata)

    # ----------------------------------------------------------------- basics
    @property
    def n_ports(self) -> int:
        """Number of ports ``n``."""
        return len(self.ports)

    @property
    def n_poles(self) -> int:
        """Number of (common) poles ``P``."""
        return int(self.poles.size)

    @property
    def carrier_wl(self) -> float:
        """Carrier wavelength in micrometres (``inf`` for a real/RF model)."""
        return C0_UM_S / self.carrier_hz if self.carrier_hz > 0 else float("inf")

    def replace(self, **changes: Any) -> RationalModel:
        """Return a copy with fields replaced (validated again)."""
        return replace(self, **changes)

    def conjugate_partners(self, rtol: float = 1e-9) -> np.ndarray | None:
        """Index of each pole's conjugate partner if the model is a real system.

        Returns ``None`` unless ``carrier_hz == 0``, ``D`` is real, and every
        pole/residue pair is matched by its complex conjugate (real poles must
        have real residues). Real poles are their own partner.
        """
        if self.carrier_hz != 0.0:
            return None
        scale = max(float(np.max(np.abs(self.d), initial=0.0)), 1e-300)
        if np.max(np.abs(self.d.imag), initial=0.0) > rtol * scale:
            return None
        p = self.poles
        if p.size == 0:
            return np.zeros(0, dtype=int)
        pscale = float(np.max(np.abs(p)))
        rscale = max(float(np.max(np.abs(self.residues))), 1e-300)
        partner = np.empty(p.size, dtype=int)
        for k in range(p.size):
            if abs(p[k].imag) <= rtol * pscale:  # a real pole is its own partner
                j = k
            else:
                dist = np.abs(p - np.conj(p[k]))
                j = int(np.argmin(dist))
                if dist[j] > rtol * pscale:
                    return None
            if np.max(np.abs(self.residues[j] - np.conj(self.residues[k]))) > rtol * rscale:
                return None
            partner[k] = j
        return partner

    @property
    def is_real_system(self) -> bool:
        """True for a real (RF) system: conjugate-symmetric poles/residues, real ``D``."""
        return self.conjugate_partners() is not None

    def port_delays(self, atol: float | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        """Decompose ``delays[out, in] = a[out] + b[in]`` if possible.

        A separable delay matrix is a set of per-port reference-plane delays.
        It leaves every singular value of ``S(Omega)`` unchanged, which makes
        the Hamiltonian passivity test of the delay-free core exact for the
        whole model. Only entries with a nonzero rational part are checked.
        Returns ``(a, b)`` (seconds) or ``None`` if no decomposition exists.
        """
        tau = self.delays
        n = self.n_ports
        active = (np.max(np.abs(self.residues), axis=0, initial=0.0) > 0) | (np.abs(self.d) > 0)
        tol = (1e-9 * max(float(np.max(tau, initial=0.0)), 1e-15)) if atol is None else float(atol)
        rows, cols = np.nonzero(active)
        if rows.size == 0:
            return np.zeros(n), np.zeros(n)
        # Least-squares a_i + b_j = tau_ij on the active entries, gauge b_0-ish fixed by lstsq.
        system = np.zeros((rows.size, 2 * n))
        system[np.arange(rows.size), rows] = 1.0
        system[np.arange(rows.size), n + cols] = 1.0
        sol, *_ = np.linalg.lstsq(system, tau[rows, cols], rcond=None)
        if np.max(np.abs(system @ sol - tau[rows, cols])) > tol:
            return None
        a, b = sol[:n], sol[n:]
        shift = min(float(np.min(a)), 0.0)  # make both non-negative where possible
        return a - shift, b + shift

    # ------------------------------------------------------------- evaluation
    def _omega(self, freq_hz) -> np.ndarray:
        return 2.0 * np.pi * (np.asarray(freq_hz, dtype=float) - self.carrier_hz)

    def response(self, freq_hz) -> np.ndarray:
        """Evaluate ``S[:, out, in]`` at absolute frequencies (Hz) with NumPy."""
        f = np.asarray(freq_hz, dtype=float)
        out = _delayed_core_response(self._omega(f), self.poles, self.residues, self.d, self.delays)
        return out.reshape(*f.shape, self.n_ports, self.n_ports)

    def __call__(
        self,
        *,
        wl=1.55,
        poles=None,
        residues=None,
        d=None,
        delays=None,
        **_settings: Any,
    ) -> SDict:
        """Photonix component model: ``S`` at wavelength(s) ``wl`` (um) as an SDict.

        ``poles``, ``residues``, ``d`` and ``delays`` override the stored arrays
        (same shapes), which makes them differentiable parameters under JAX.
        Other keyword settings are accepted and ignored, as for any fixed model.
        Every ``(in, out)`` pair is returned. The evaluation uses only backend
        arithmetic and is ``jit``/``grad``-compatible.
        """
        wl = xp.asarray(wl)
        omega = 2.0 * xp.pi * (C0_UM_S / wl - self.carrier_hz)
        p = xp.asarray(self.poles if poles is None else poles)
        r = xp.asarray(self.residues if residues is None else residues)
        dd = xp.asarray(self.d if d is None else d)
        tau = xp.asarray(self.delays if delays is None else delays)
        core = dd + 0.0j
        if self.n_poles:
            basis = 1.0 / (1j * omega[..., None] - p)
            core = core + xp.einsum("...k,kij->...ij", basis, r)
        h = core * xp.exp(-1j * omega[..., None, None] * tau)
        return {
            (p_in, p_out): h[..., i, j]
            for j, p_in in enumerate(self.ports)
            for i, p_out in enumerate(self.ports)
        }

    def rms_error(self, wavelengths, S, *, ports: Sequence[str] | None = None) -> float:
        """Root-mean-square ``|S_model - S|`` over all samples and all ``n**2`` entries.

        ``S`` is ``(N, n, n)`` in ``S[:, out, in]`` order (or an SDict, or
        ``(N,)`` for a one-port).
        """
        f = wavelengths_to_hz(wavelengths)
        data, _ = dense_data(S, f.size, self.ports if ports is None else ports)
        err = self.response(f) - data
        return float(np.sqrt(np.mean(np.abs(err) ** 2)))

    def to_dataset(self, wavelengths, *, metadata: dict[str, Any] | None = None):
        """Sample the model into a :class:`~photonix.core.SParameterDataset`."""
        from photonix.core.dataset import SParameterDataset

        wl = np.asarray(wavelengths, dtype=float)
        info = {"source": "photonix.macromodel.RationalModel", "n_poles": self.n_poles, **(metadata or {})}
        return SParameterDataset(wl, self.ports, self.response(wavelengths_to_hz(wl)), info)

    def impulse_response(self, t_s) -> np.ndarray:
        """Baseband impulse response ``h[:, out, in]`` at times ``t_s`` (seconds).

        ``h(t) = sum_k R_k exp(p_k (t - tau)) for t >= tau`` (units 1/s). The
        direct term contributes ``D delta(t - tau)``, which is not sampled.
        """
        t = np.asarray(t_s, dtype=float).reshape(-1)
        shifted = t[:, None, None] - self.delays[None]
        out = np.zeros((t.size, self.n_ports, self.n_ports), dtype=complex)
        causal = shifted >= 0
        for k, p in enumerate(self.poles):
            out += np.where(causal, self.residues[k][None] * np.exp(p * np.where(causal, shifted, 0.0)), 0.0)
        return out

    # ----------------------------------------------------------- realizations
    def _complex_blocks(self, minimal: bool):
        """Per-pole realization blocks ``(p, Bk, Ck)`` with ``Ck @ Bk = R_k``."""
        rmax = float(np.max(np.abs(self.residues), initial=0.0))
        blocks = []
        for k, p in enumerate(self.poles):
            bk, ck = _factor(self.residues[k], minimal, rmax)
            if bk.shape[0]:
                blocks.append((p, bk, ck))
        return blocks

    def to_statespace(self, *, minimal: bool = True, real: bool = False) -> StateSpace:
        """Gilbert realization of the delay-free rational core.

        ``A = blockdiag(p_k I_{r_k})``, ``B_k = V_k^H`` and ``C_k = U_k Sigma_k``
        from the SVD of each residue matrix (``r_k`` its numerical rank, or
        ``n`` with ``minimal=False``, giving ``B = [I; I; ...]``,
        ``C = [R_1, R_2, ...]``). With ``real=True`` (real systems only) each
        conjugate pair is converted to a real ``2 r_k`` block so that ``A, B,
        C, D`` are real.

        Examples
        --------
        >>> import numpy as np
        >>> m = RationalModel(("o1",), 0.0, np.array([-1.0 + 2j, -1.0 - 2j]),
        ...                   np.array([[[1 - 1j]], [[1 + 1j]]]), np.array([[0.1]]))
        >>> ss = m.to_statespace(real=True)
        >>> ss.A.dtype.kind, ss.n_states
        ('f', 2)
        >>> bool(np.allclose(ss.response([0.3]), m.response([0.3])))
        True
        """
        n = self.n_ports
        if real:
            partners = self.conjugate_partners()
            if partners is None:
                raise ValueError("a real realization requires a real system (carrier_hz == 0, conjugate symmetry)")
            rmax = float(np.max(np.abs(self.residues), initial=0.0))
            a_parts, b_parts, c_parts = [], [], []
            for k, p in enumerate(self.poles):
                j = int(partners[k])
                if j == k:  # real pole with a real residue matrix
                    bk, ck = _factor(self.residues[k].real, minimal, rmax)
                    if bk.shape[0]:
                        a_parts.append(np.eye(bk.shape[0]) * p.real)
                        b_parts.append(bk)
                        c_parts.append(ck)
                    continue
                if p.imag < 0:  # realized together with its partner (Im p > 0)
                    continue
                bk, ck = _factor(self.residues[k], minimal, rmax)
                r = bk.shape[0]
                if not r:
                    continue
                eye = np.eye(r)
                a_parts.append(np.block([[eye * p.real, -eye * p.imag], [eye * p.imag, eye * p.real]]))
                b_parts.append(np.vstack([bk.real, bk.imag]))
                c_parts.append(np.hstack([2 * ck.real, -2 * ck.imag]))
            size = sum(a.shape[0] for a in a_parts)
            A = np.zeros((size, size))
            offset = 0
            for a in a_parts:
                m = a.shape[0]
                A[offset:offset + m, offset:offset + m] = a
                offset += m
            B = np.vstack(b_parts) if b_parts else np.zeros((0, n))
            C = np.hstack(c_parts) if c_parts else np.zeros((n, 0))
            return StateSpace(A, B, C, self.d.real.copy(), self.delays.copy(), 0.0, self.ports)
        blocks = self._complex_blocks(minimal)
        diag = np.concatenate([np.full(bk.shape[0], p) for p, bk, _ in blocks]) if blocks else np.zeros(0, complex)
        B = np.vstack([bk for _, bk, _ in blocks]) if blocks else np.zeros((0, n), complex)
        C = np.hstack([ck for _, _, ck in blocks]) if blocks else np.zeros((n, 0), complex)
        return StateSpace(np.diag(diag), B, C, self.d.copy(), self.delays.copy(), self.carrier_hz, self.ports)

    def discretize(self, dt: float, method: str = "foh") -> DiscreteStateSpace:
        """Exact discrete-time equivalent of the rational core with sample time ``dt`` (s).

        ``method="zoh"`` is exact for piecewise-constant inputs and
        ``method="foh"`` (triangle hold) for piecewise-linear inputs. The
        realization is the minimal complex modal one, so ``Ad`` is diagonal.

        Examples
        --------
        >>> import numpy as np
        >>> m = RationalModel(("o1",), 0.0, np.array([-2.0 + 0j]), np.array([[[2.0 + 0j]]]), np.zeros((1, 1)))
        >>> dss = m.discretize(0.01, "zoh")
        >>> y = dss.simulate(np.ones(301))           # unit step
        >>> round(float(y[-1, 0].real), 6), round(float(1 - np.exp(-2 * 3.0)), 6)
        (0.997521, 0.997521)
        """
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if method not in ("zoh", "foh"):
            raise ValueError("method must be 'zoh' or 'foh'")
        ss = self.to_statespace(minimal=True)
        lam_c = np.diag(ss.A)
        x = lam_c * dt
        lam = np.exp(x)
        g0 = dt * _phi1(x)
        if method == "zoh":
            bd = g0[:, None] * ss.B
            dd = ss.D.copy()
        else:
            g1 = dt * _phi2(x)
            bd = (lam * g1 + g0 - g1)[:, None] * ss.B
            dd = ss.D + ss.C @ (g1[:, None] * ss.B)
        return DiscreteStateSpace(np.diag(lam), bd, ss.C.copy(), dd, dt, method, self.delays.copy(),
                                  self.carrier_hz, self.ports)

    def simulate(self, u, dt: float) -> np.ndarray:
        """Exact time-domain output for a piecewise-linear input envelope.

        Parameters
        ----------
        u
            Input samples ``(K, n)`` (or ``(K,)`` for a one-port) at
            ``t_k = k dt``, complex baseband envelopes about the carrier. The
            input is the piecewise-linear interpolant of the samples and is zero
            for ``t <= -dt``.
        dt
            Sample spacing in seconds.

        Returns
        -------
        numpy.ndarray
            Output envelopes ``y[k, out]`` at ``t_k``, including every
            (fractional) path delay. Each pole is propagated in closed form, so
            the only approximation is the piecewise-linear input itself.

        Examples
        --------
        >>> import numpy as np
        >>> m = RationalModel(("o1", "o2"), 0.0, np.zeros(0), np.zeros((0, 2, 2)),
        ...                   np.array([[0, 1], [1, 0]]), delays=np.full((2, 2), 2.5e-12))
        >>> t = np.arange(40) * 1e-12
        >>> u = np.stack([t * 1e12, np.zeros_like(t)], axis=1)       # ramp into port o1
        >>> y = m.simulate(u, 1e-12)
        >>> float(y[10, 1].real)                                    # ramp delayed by 2.5 ps
        7.5
        """
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        u = np.asarray(u, dtype=complex)
        if u.ndim == 1:
            u = u[:, None]
        n = self.n_ports
        if u.ndim != 2 or u.shape[1] != n:
            raise ValueError(f"u must have shape (K, {n})")
        K = u.shape[0]
        from scipy.signal import lfilter

        p = self.poles
        x = p * dt
        lam = np.exp(x)
        g0 = dt * _phi1(x)
        g1 = dt * _phi2(x)
        y = np.zeros((K, n), dtype=complex)
        for j in range(n):  # one input at a time keeps memory at O(K P)
            column_r = self.residues[:, :, j]
            if not np.any(column_r) and not np.any(self.d[:, j]):
                continue
            # z[k, pole]: modal state of each pole driven by input j alone, at t_k (exact for linear ramps).
            z = np.empty((K, p.size), dtype=complex)
            for k in range(p.size):
                z[:, k] = lfilter([g1[k], g0[k] - g1[k]], [1.0, -lam[k]], u[:, j])
            for i in range(n):
                r = self.residues[:, i, j]
                dij = self.d[i, j]
                if not np.any(r) and dij == 0:
                    continue
                q = self.delays[i, j] / dt
                m = int(np.floor(q + 1e-12))
                frac = max(q - m, 0.0)
                delta = (1.0 - frac) * dt  # t_k - tau = t_{k-m-1} + delta
                pad = m + 1
                u_pad = np.concatenate([np.zeros(pad, complex), u[:, j]])
                ua = u_pad[:K]
                ub = u_pad[1:K + 1]
                out = dij * (ua + (ub - ua) * (delta / dt))
                if p.size:
                    z_pad = np.concatenate([np.zeros((pad, p.size), complex), z])[:K]
                    xd = p * delta
                    alpha = r * np.exp(xd)
                    beta = np.sum(r * delta * _phi1(xd))
                    gamma = np.sum(r * (delta * delta / dt) * _phi2(xd))
                    out = out + z_pad @ alpha + beta * ua + gamma * (ub - ua)
                y[:, i] += out
        return y

    # ------------------------------------------------------------------- I/O
    def save_npz(self, path: str | Path) -> None:
        """Write a portable, pickle-free, versioned NPZ file."""
        np.savez_compressed(
            Path(path),
            format=np.asarray(_FORMAT),
            format_version=np.asarray(_FORMAT_VERSION, dtype=np.int64),
            ports=np.asarray(self.ports, dtype=str),
            carrier_hz=np.asarray(self.carrier_hz, dtype=float),
            poles=self.poles,
            residues=self.residues,
            d=self.d,
            delays_s=self.delays,
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> RationalModel:
        """Load :meth:`save_npz` output with pickle disabled and version checks.

        Examples
        --------
        >>> import numpy as np, tempfile, os
        >>> m = RationalModel(("o1",), 1.9e14, np.array([-1e11 + 0j]), np.ones((1, 1, 1)), np.zeros((1, 1)))
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     m.save_npz(os.path.join(tmp, "m.npz"))
        ...     back = RationalModel.load_npz(os.path.join(tmp, "m.npz"))
        >>> bool(np.array_equal(back.poles, m.poles)) and back.ports == ("o1",)
        True
        """
        with np.load(Path(path), allow_pickle=False) as data:
            if str(np.asarray(data["format"]).item()) != _FORMAT:
                raise ValueError("not a Photonix rational-model file")
            version = int(np.asarray(data["format_version"]).item())
            if version != _FORMAT_VERSION:
                raise ValueError(f"unsupported rational-model format version {version}")
            return cls(
                tuple(str(p) for p in np.asarray(data["ports"])),
                float(np.asarray(data["carrier_hz"]).item()),
                np.asarray(data["poles"], dtype=complex),
                np.asarray(data["residues"], dtype=complex),
                np.asarray(data["d"], dtype=complex),
                np.asarray(data["delays_s"], dtype=float),
                json.loads(str(np.asarray(data["metadata_json"]).item())),
            )
