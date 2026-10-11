"""Sample-mode (fixed-step, complex-envelope) time-domain circuit simulation.

A :class:`TimeDomainCircuit` connects time-domain elements
(:mod:`photonix.timedomain.elements`) exactly as the frequency-domain solver
connects scattering dictionaries, and steps the whole network forward one
sample at a time.

Algorithm
---------
Stack every element port into one wave vector. Each element splits into an
instantaneous part and strictly causal *channels* (``b = D a + y``, with
``y_k`` depending only on incident waves before step ``k``), and the
connections give ``a = Gamma b + P u`` with ``u`` the waves injected at the
external ports. Time-varying (quasi-static) elements contribute their
outgoing waves ``z_k = S_k a_T`` at their terminals ``T``. Eliminating the
instantaneous algebraic loop once,

.. code-block:: text

    b = (I - D Gamma)^-1 (D P u + Q w + E_T z),       a = Gamma b + P u,

so the incident waves feeding every channel, the time-varying blocks and the
recorded outputs are fixed linear maps of ``(u_k, w_k, z_k)``, where ``w_k``
are the channel outputs. Each step therefore costs one small gather from the
channels' circular history buffers, a few matrix-vector products and (only
when time-varying elements are present) a tiny per-step solve
``(I - S_k F_z) z = S_k (F_u u + F_w w)`` that is precomputed for the whole
record. Any element with at least one sample of delay breaks a feedback
loop, so rings, CROWs and loop mirrors are explicit and causal (the approach
of Photontorch, Laporte et al. 2019); a loop with no delay at all is an
algebraic loop that is solved exactly, and a *singular* one (a lossless loop
with zero delay) is rejected.

Under JAX the stepping runs in :func:`jax.lax.scan` inside :func:`jax.jit`
and is differentiable end to end (with respect to element coefficients, input
waveforms and control waveforms); under NumPy it is a plain loop.

Conventions are those of :mod:`photonix.timedomain.signal`: complex envelopes
in sqrt(W) about the carrier ``carrier_wl`` with ``exp(+j omega t)`` time
dependence, so the envelope response of the network at frequency offset
``nu`` approximates the frequency-domain ``S(f_c + nu)``.

References
----------
* F. Laporte, J. Dambre, P. Bienstman, "Highly parallel simulation and
  optimization of photonic circuits in time and frequency domain based on the
  deep-learning framework PyTorch," Sci. Rep. 9, 5918 (2019).
* Ansys Lumerical INTERCONNECT, "Sample mode" simulation documentation.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from photonix.core.backend import HAS_JAX, xp

from .elements import CompiledTD, CompositeTD, DelayLineTD, TDElement
from .signal import Waveform

__all__ = ["TimeDomainCircuit", "flatten_elements"]

_Terminal = tuple[str, str]


# --------------------------------------------------------------------------- #
# Flattening of composite elements
# --------------------------------------------------------------------------- #
def flatten_elements(
    elements: Mapping[str, TDElement],
    connections: Mapping[_Terminal, _Terminal],
    ports: Mapping[str, _Terminal],
) -> tuple[dict[str, TDElement], dict[_Terminal, _Terminal], dict[str, _Terminal], dict[_Terminal, _Terminal]]:
    """Expand every :class:`~photonix.timedomain.elements.CompositeTD` into leaf elements.

    Sub-elements are named ``"<instance>/<sub>"`` (recursively).

    Returns
    -------
    flat, connections, ports, terminal_map
        Leaf elements, connections and external ports in terms of leaf
        terminals, and the map from every original ``(instance, port)``
        terminal to its leaf terminal.

    Examples
    --------
    >>> from photonix.timedomain.elements import CompositeTD, DelayLineTD
    >>> comp = CompositeTD({"a": DelayLineTD(1e-12), "b": DelayLineTD(2e-12)},
    ...                    {("a", "o2"): ("b", "o1")}, {"o1": ("a", "o1"), "o2": ("b", "o2")})
    >>> flat, conns, ports, _ = flatten_elements({"x": comp}, {}, {"in": ("x", "o1"), "out": ("x", "o2")})
    >>> sorted(flat), ports["out"]
    (['x/a', 'x/b'], ('x/b', 'o2'))
    """
    flat: dict[str, TDElement] = {}
    conns: dict[_Terminal, _Terminal] = {}
    tmap: dict[_Terminal, _Terminal] = {}

    def expand(prefix: str, el: TDElement) -> dict[str, _Terminal]:
        if isinstance(el, CompositeTD):
            sub_maps = {sub: expand(f"{prefix}/{sub}", sub_el) for sub, sub_el in el.elements.items()}
            for (sa, pa), (sb, pb) in el.connections.items():
                conns[sub_maps[sa][pa]] = sub_maps[sb][pb]
            return {p: sub_maps[sub][sp] for p, (sub, sp) in el.port_map.items()}
        if prefix in flat:
            raise ValueError(f"duplicate element name {prefix!r} after flattening")
        flat[prefix] = el
        return {p: (prefix, p) for p in el.ports}

    for name, el in elements.items():
        if "/" in name or "," in name:
            raise ValueError(f"instance names may not contain '/' or ',' (got {name!r})")
        for p, term in expand(name, el).items():
            tmap[(name, p)] = term

    def resolve(t: _Terminal) -> _Terminal:
        if t not in tmap:
            raise KeyError(f"terminal {t!r} does not exist (unknown instance or port)")
        return tmap[t]

    for a, b in connections.items():
        conns[resolve(a)] = resolve(b)
    out_ports = {ext: resolve(t) for ext, t in ports.items()}
    return flat, conns, out_ports, tmap


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Group:
    """One circular history buffer shared by sources with similar maximum lag."""

    length: int                 # buffer length L (>= every lag read from it)
    src: tuple[int, ...]        # source indices stored in this buffer's columns
    lags: tuple[int, ...]       # per tap
    cols: tuple[int, ...]       # per tap: buffer column
    chans: tuple[int, ...]      # per tap: channel index


@dataclass(frozen=True)
class _Structure:
    """The static (shape) part of an assembled network; hashable."""

    n_ext: int
    n_rec: int
    n_chan: int
    n_src: int
    groups: tuple[_Group, ...]
    iir: tuple[int, ...]        # channel indices with feedback
    p_max: int
    n_tv: int


def _set_block(mat, idx: Sequence[int], block):
    idx_a = np.asarray(idx, dtype=int)
    if hasattr(mat, "at"):
        return mat.at[np.ix_(idx_a, idx_a)].set(block)
    mat = np.array(mat, copy=True)
    mat[np.ix_(idx_a, idx_a)] = np.asarray(block)
    return mat


def _is_concrete(x) -> bool:
    try:
        np.asarray(x)
        return True
    except Exception:  # noqa: BLE001 - jax tracers
        return False


def _assemble(compiled: Mapping[str, CompiledTD], conns: Mapping[_Terminal, _Terminal],
              ports: Mapping[str, _Terminal], record: Sequence[_Terminal]):
    """Reduce the network to the per-step linear maps (see the module docstring)."""
    term_index: dict[_Terminal, int] = {}
    for name, c in compiled.items():
        for p in c.ports:
            term_index[(name, p)] = len(term_index)
    n = len(term_index)

    # -- connections and external ports -------------------------------------- #
    gamma = np.zeros((n, n))
    used: set[_Terminal] = set()
    for a, b in conns.items():
        for t in (a, b):
            if t not in term_index:
                raise KeyError(f"connection {a} <-> {b} references unknown terminal {t!r}")
            if t in used:
                raise ValueError(f"terminal {t!r} appears in more than one connection")
            used.add(t)
        if a == b:
            raise ValueError(f"a terminal cannot be connected to itself: {a!r}")
        gamma[term_index[a], term_index[b]] = 1.0
        gamma[term_index[b], term_index[a]] = 1.0
    ext_names = list(ports)
    p_ext = np.zeros((n, len(ext_names)))
    for j, ext in enumerate(ext_names):
        t = ports[ext]
        if t not in term_index:
            raise KeyError(f"external port {ext!r} references unknown terminal {t!r}")
        if t in used:
            raise ValueError(f"external port {ext!r} references the internally connected terminal {t!r}")
        p_ext[term_index[t], j] = 1.0
    if np.any(p_ext.sum(axis=1) > 1):
        raise ValueError("a terminal is exposed as more than one external port")

    # -- instantaneous matrix and time-varying terminals --------------------- #
    d0 = xp.zeros((n, n), dtype=complex)
    tv_terms: list[int] = []
    tv_blocks: list[tuple[str, Any]] = []
    for name, c in compiled.items():
        idx = [term_index[(name, p)] for p in c.ports]
        if c.time_varying is not None:
            tv_terms.extend(idx)
            tv_blocks.append((name, c.time_varying))
        else:
            d0 = _set_block(d0, idx, xp.asarray(c.D, dtype=complex))

    # -- channels ------------------------------------------------------------ #
    chans = []
    for name, c in compiled.items():
        for ch in c.channels:
            chans.append((term_index[(name, ch.src)], term_index[(name, ch.dst)], ch))
    n_chan = len(chans)
    src_terms = list(dict.fromkeys(s for s, _, _ in chans))
    src_slot = {t: i for i, t in enumerate(src_terms)}
    q = np.zeros((n, n_chan))
    for ci, (_, dst, _) in enumerate(chans):
        q[dst, ci] = 1.0

    # Tap list per source, then group sources by power-of-two class of max lag.
    max_lag = [0] * len(src_terms)
    for s, _, ch in chans:
        max_lag[src_slot[s]] = max(max_lag[src_slot[s]], ch.n0 + ch.n_taps - 1)
    classes: dict[int, list[int]] = {}
    for si, ml in enumerate(max_lag):
        classes.setdefault(int(math.ceil(math.log2(max(ml, 1)))), []).append(si)
    groups: list[_Group] = []
    coefs: list[Any] = []
    for cls in sorted(classes):
        members = classes[cls]
        col_of = {si: i for i, si in enumerate(members)}
        lags: list[int] = []
        cols: list[int] = []
        cidx: list[int] = []
        taps: list[Any] = []
        for ci, (s, _, ch) in enumerate(chans):
            si = src_slot[s]
            if si not in col_of:
                continue
            m = ch.n_taps
            lags.extend(ch.n0 + k for k in range(m))
            cols.extend([col_of[si]] * m)
            cidx.extend([ci] * m)
            taps.append(xp.reshape(xp.asarray(ch.b, dtype=complex), (m,)))
        length = max(max_lag[si] for si in members)
        groups.append(_Group(int(length), tuple(members), tuple(lags), tuple(cols), tuple(cidx)))
        coefs.append(xp.concatenate(taps) if taps else xp.zeros(0, dtype=complex))

    iir = [ci for ci, (_, _, ch) in enumerate(chans) if ch.a is not None]
    p_max = max((chans[ci][2].n_feedback for ci in iir), default=0)
    a_iir = xp.zeros((p_max, len(iir)), dtype=complex)
    for j, ci in enumerate(iir):
        a = xp.reshape(xp.asarray(chans[ci][2].a, dtype=complex), (-1,))
        col = xp.concatenate([a, xp.zeros(p_max - int(a.shape[0]), dtype=complex)])
        a_iir = a_iir.at[:, j].set(col) if hasattr(a_iir, "at") else _np_setcol(a_iir, j, col)

    # -- eliminate the instantaneous loop ------------------------------------ #
    n_tv = len(tv_terms)
    e_t = np.zeros((n, n_tv))
    for j, t in enumerate(tv_terms):
        e_t[t, j] = 1.0
    gamma_x = xp.asarray(gamma, dtype=complex)
    lhs = xp.eye(n, dtype=complex) - d0 @ gamma_x
    if _is_concrete(lhs) and n:
        cond = np.linalg.cond(np.asarray(lhs))
        if not np.isfinite(cond) or cond > 1e12:
            raise ValueError(
                "the network has a singular instantaneous (zero-delay) loop: some feedback path contains no "
                "element with memory. Add the waveguide (delay) that closes the loop."
            )
    rhs = xp.concatenate([d0 @ xp.asarray(p_ext, dtype=complex), xp.asarray(q, dtype=complex),
                          xp.asarray(e_t, dtype=complex)], axis=1)
    b_lin = xp.linalg.solve(lhs, rhs) if n else rhs
    a_lin = gamma_x @ b_lin + xp.concatenate(
        [xp.asarray(p_ext, dtype=complex), xp.zeros((n, n_chan + n_tv), dtype=complex)], axis=1)
    ne = len(ext_names)

    def split(rows):
        return rows[:, :ne], rows[:, ne:ne + n_chan], rows[:, ne + n_chan:]

    src_idx = np.asarray(src_terms, dtype=int)
    rec_idx = np.asarray([term_index[t] for t in record], dtype=int)
    tv_idx = np.asarray(tv_terms, dtype=int)
    xu, xw, xz = split(a_lin[src_idx, :] if len(src_idx) else xp.zeros((0, ne + n_chan + n_tv), dtype=complex))
    ou, ow, oz = split(b_lin[rec_idx, :] if len(rec_idx) else xp.zeros((0, ne + n_chan + n_tv), dtype=complex))
    fu, fw, fz = split(a_lin[tv_idx, :] if n_tv else xp.zeros((0, ne + n_chan + n_tv), dtype=complex))

    structure = _Structure(ne, len(record), n_chan, len(src_terms), tuple(groups), tuple(iir), p_max, n_tv)
    arrays = {
        "xu": xu, "xw": xw, "xz": xz, "ou": ou, "ow": ow, "oz": oz, "fu": fu, "fw": fw, "fz": fz,
        "coefs": tuple(coefs), "a_iir": a_iir,
    }
    info = {"ext": ext_names, "tv_blocks": tv_blocks, "tv_terms": tv_terms, "term_index": term_index}
    return structure, arrays, info


def _np_setcol(mat, j, col):
    mat = np.array(mat, copy=True)
    mat[:, j] = np.asarray(col)
    return mat


# --------------------------------------------------------------------------- #
# Time stepping
# --------------------------------------------------------------------------- #
def _dense_tap_matrix(st: _Structure, g: _Group, coef):
    """``(n_chan, n_taps)`` matrix that sums tap products into channel outputs."""
    rows = np.asarray(g.chans, dtype=int)
    cols = np.arange(len(g.chans))
    mat = xp.zeros((st.n_chan, len(g.chans)), dtype=complex)
    if hasattr(mat, "at"):
        return mat.at[rows, cols].set(coef)
    mat = np.array(mat)
    mat[rows, cols] = np.asarray(coef)
    return mat


def _make_jax_runner(st: _Structure, unroll: int):
    import jax
    from jax import lax

    lag_idx = [np.asarray(g.lags, dtype=np.int64) for g in st.groups]
    col_idx = [np.asarray(g.cols, dtype=np.int64) for g in st.groups]
    src_idx = [np.asarray(g.src, dtype=np.int64) for g in st.groups]
    iir_idx = np.asarray(st.iir, dtype=np.int64)
    jnp = xp

    def run(arr, u_seq, w_seq):
        mats = [_dense_tap_matrix(st, g, c) for g, c in zip(st.groups, arr["coefs"], strict=True)]
        bufs0 = tuple(jnp.zeros((g.length, len(g.src)), dtype=complex) for g in st.groups)
        whist0 = jnp.zeros((st.p_max, len(st.iir)), dtype=complex)

        def body(carry, inp):
            bufs, whist, k = carry
            u = inp[0]
            w = jnp.zeros(st.n_chan, dtype=complex)
            for gi, g in enumerate(st.groups):
                vals = bufs[gi][(k - lag_idx[gi]) % g.length, col_idx[gi]]
                w = w + mats[gi] @ vals
            if st.iir:
                fb = jnp.sum(arr["a_iir"] * whist, axis=0)
                w = w.at[iir_idx].add(-fb)
            x = arr["xu"] @ u + arr["xw"] @ w
            out = arr["ou"] @ u + arr["ow"] @ w
            if st.n_tv:
                z = inp[1] @ (arr["fu"] @ u + arr["fw"] @ w)
                x = x + arr["xz"] @ z
                out = out + arr["oz"] @ z
            new_bufs = tuple(
                b.at[k % g.length].set(x[src_idx[gi]]) for gi, (b, g) in enumerate(zip(bufs, st.groups, strict=True))
            )
            if st.iir:
                whist = jnp.concatenate([w[iir_idx][None, :], whist[:-1]], axis=0)
            return (new_bufs, whist, k + 1), out

        xs = (u_seq, w_seq) if st.n_tv else (u_seq,)
        _, outs = lax.scan(body, (bufs0, whist0, jnp.asarray(0, dtype=jnp.int64)), xs, unroll=unroll)
        return outs

    return jax.jit(run)


def _run_numpy(st: _Structure, arr, u_seq, w_seq):
    mats = [np.asarray(_dense_tap_matrix(st, g, c)) for g, c in zip(st.groups, arr["coefs"], strict=True)]
    lag_idx = [np.asarray(g.lags, dtype=int) for g in st.groups]
    col_idx = [np.asarray(g.cols, dtype=int) for g in st.groups]
    src_idx = [np.asarray(g.src, dtype=int) for g in st.groups]
    iir_idx = np.asarray(st.iir, dtype=int)
    bufs = [np.zeros((g.length, len(g.src)), dtype=complex) for g in st.groups]
    whist = np.zeros((st.p_max, len(st.iir)), dtype=complex)
    a = {k: np.asarray(v) for k, v in arr.items() if k != "coefs"}
    u_seq = np.asarray(u_seq)
    n_steps = u_seq.shape[0]
    outs = np.zeros((n_steps, st.n_rec), dtype=complex)
    w = np.zeros(st.n_chan, dtype=complex)
    for k in range(n_steps):
        w[:] = 0.0
        for gi, g in enumerate(st.groups):
            w += mats[gi] @ bufs[gi][(k - lag_idx[gi]) % g.length, col_idx[gi]]
        if st.iir:
            w[iir_idx] -= np.sum(a["a_iir"] * whist, axis=0)
        u = u_seq[k]
        x = a["xu"] @ u + a["xw"] @ w
        out = a["ou"] @ u + a["ow"] @ w
        if st.n_tv:
            z = w_seq[k] @ (a["fu"] @ u + a["fw"] @ w)
            x = x + a["xz"] @ z
            out = out + a["oz"] @ z
        outs[k] = out
        for gi, g in enumerate(st.groups):
            bufs[gi][k % g.length] = x[src_idx[gi]]
        if st.iir:
            whist[1:] = whist[:-1]
            whist[0] = w[iir_idx]
    return outs


# --------------------------------------------------------------------------- #
# Public circuit object
# --------------------------------------------------------------------------- #
RecordSpec = Literal["external", "all"] | Sequence[str | tuple[str, str]]


class TimeDomainCircuit:
    """A photonic circuit simulated sample by sample in the complex-envelope picture.

    Parameters
    ----------
    elements
        ``instance -> TDElement``. :class:`~photonix.timedomain.elements.CompositeTD`
        sub-networks are flattened (sub-elements appear as ``"inst/sub"``).
    connections
        ``(inst_a, port_a) -> (inst_b, port_b)``, as in a
        :class:`~photonix.circuit.Netlist`.
    ports
        External port name ``-> (instance, port)``.
    dt
        Time step in seconds. The envelope bandwidth is ``+/- 1 / (2 dt)``
        about the carrier; elements were designed for this step.
    carrier_wl
        Carrier wavelength in µm.

    Notes
    -----
    Unconnected, unexposed element ports are matched terminations (no incident
    light), as in :func:`photonix.circuit.evaluate_circuit`.

    Examples
    --------
    An all-pass ring (coupler plus a ring waveguide) driven by a CW tone: the
    steady-state output equals the frequency-domain transmission.

    >>> import numpy as np, photonix as px
    >>> from photonix.timedomain import TimeDomainCircuit, cw
    >>> nl = px.circuit.Netlist(name="ring")
    >>> _ = nl.add("c", "directional_coupler", coupling=0.2).add("rw", "straight", length=2 * np.pi * 10.0)
    >>> _ = nl.connect(("c", "o3"), ("rw", "o1")).connect(("rw", "o2"), ("c", "o2"))
    >>> _ = nl.expose("o1", ("c", "o1")).expose("o2", ("c", "o4"))
    >>> tdc = TimeDomainCircuit.from_netlist(nl, carrier_wl=1.55, dt=0.05e-12)
    >>> out = tdc.simulate({"o1": cw(1e-3, 3000, 0.05e-12)})
    >>> t_td = complex(out["o2"].samples[-1]) / np.sqrt(1e-3)
    >>> t_fd = complex(px.circuit.circuit_from_netlist(nl)(wl=1.55)[("o1", "o2")])
    >>> bool(abs(t_td - t_fd) < 1e-9)
    True
    """

    def __init__(
        self,
        elements: Mapping[str, TDElement],
        connections: Mapping[_Terminal, _Terminal],
        ports: Mapping[str, _Terminal],
        *,
        dt: float,
        carrier_wl: float = 1.55,
        name: str | None = None,
    ) -> None:
        if not (isinstance(dt, (int, float)) and dt > 0 and math.isfinite(dt)):
            raise ValueError("dt must be a positive number of seconds")
        if not float(carrier_wl) > 0:
            raise ValueError("carrier_wl must be a positive wavelength in µm")
        self.dt = float(dt)
        self.carrier_wl = float(carrier_wl)
        self.name = name
        self.elements: dict[str, TDElement] = dict(elements)
        self.connections = dict(connections)
        self.ports = dict(ports)
        self._flat, self._conns, self._ports, self._tmap = flatten_elements(self.elements, self.connections, self.ports)
        self._flat = {k: v.frozen(self.dt) for k, v in self._flat.items()}
        self._rebuild: Callable[[Mapping[str, Any]], dict[str, TDElement]] | None = None
        self._cache: dict[Any, Any] = {}
        self._structure_ref: dict[Any, _Structure] = {}
        self._check_delays()

    # -- construction -------------------------------------------------------- #
    @classmethod
    def from_netlist(
        cls,
        netlist: Any,
        models: Mapping[str, Callable] | None = None,
        *,
        carrier_wl: float = 1.55,
        dt: float,
        overrides: Mapping[str, Mapping[str, Any]] | None = None,
        td_elements: Mapping[str, TDElement] | None = None,
        time_varying: Mapping[str, Mapping[str, str]] | None = None,
        interp: str = "lagrange3",
        fir_tol: float = 1e-4,
        max_taps: int = 4096,
    ) -> TimeDomainCircuit:
        """Build from a :class:`~photonix.circuit.Netlist` and a model registry.

        Every instance is converted with :func:`photonix.timedomain.to_td`
        (analytic delay lines for waveguides, memoryless couplers, structural
        rings, sampled FIR fits otherwise).

        Parameters
        ----------
        netlist
            The circuit topology (a ``Netlist``, or any object with a
            ``netlist`` attribute and optional ``models`` registry).
        models
            ``model_name -> model``; defaults to
            :data:`photonix.components.MODELS`.
        overrides
            ``{instance: {param: value}}`` on top of the netlist settings.
        td_elements
            ``{instance: TDElement}`` used verbatim instead of converting.
        time_varying
            ``{instance: {model_param: control_name}}``: the instance becomes a
            quasi-static modulated element driven by the named control
            waveforms passed to :meth:`simulate` (see
            :func:`photonix.timedomain.to_td_time_varying`).
        interp, fir_tol, max_taps
            Forwarded to :func:`~photonix.timedomain.to_td`.
        """
        from photonix.circuit.netlist import Netlist

        from .convert import to_td, to_td_time_varying

        if not isinstance(netlist, Netlist) and hasattr(netlist, "netlist"):
            models = models if models is not None else getattr(netlist, "models", None)
            netlist = netlist.netlist
        if not isinstance(netlist, Netlist):
            raise TypeError("from_netlist needs a photonix.circuit.Netlist")
        netlist.validate()
        if models is None:
            from photonix.components import MODELS

            registry: Mapping[str, Callable] = MODELS
        else:
            registry = models
        overrides = dict(overrides or {})
        td_elements = dict(td_elements or {})
        time_varying = dict(time_varying or {})
        for key in (*overrides, *td_elements, *time_varying):
            if key not in netlist.instances:
                raise KeyError(f"unknown instance {key!r}")
        opts = {"carrier_wl": float(carrier_wl), "dt": float(dt), "interp": interp,
                "fir_tol": fir_tol, "max_taps": max_taps}

        def build(inst: str, extra: Mapping[str, Any] | None, like: TDElement | None) -> TDElement:
            if inst in td_elements:
                return td_elements[inst]
            model_name = netlist.instances[inst]
            if model_name not in registry:
                raise KeyError(f"model {model_name!r} of instance {inst!r} is not in the registry")
            model = registry[model_name]
            settings = netlist.merged_settings(inst, {**overrides.get(inst, {}), **dict(extra or {})})
            if inst in time_varying:
                return to_td_time_varying(model, settings, control_params=time_varying[inst], like=like, **opts)
            return to_td(model, settings, like=like, **opts)

        elements = {inst: build(inst, None, None) for inst in netlist.instances}
        circ = cls(elements, netlist.connections, netlist.ports, dt=dt, carrier_wl=carrier_wl, name=netlist.name)
        frozen = {inst: el.frozen(float(dt)) for inst, el in elements.items()}

        def rebuild(extra: Mapping[str, Any]) -> dict[str, TDElement]:
            out = dict(frozen)
            for inst, params in extra.items():
                if inst not in netlist.instances:
                    raise KeyError(f"unknown instance {inst!r}")
                out[inst] = build(inst, params, frozen[inst])
            return out

        circ._rebuild = rebuild
        circ.elements = frozen
        return circ

    # -- introspection ------------------------------------------------------- #
    @property
    def external_ports(self) -> list[str]:
        """External port names (in the order given)."""
        return list(self._ports)

    @property
    def terminals(self) -> list[str]:
        """Every leaf terminal as ``"element,port"``."""
        return [f"{name},{p}" for name, el in self._flat.items() for p in el.ports]

    @property
    def controls(self) -> tuple[str, ...]:
        """Names of the control waveforms required by time-varying elements."""
        names: list[str] = []
        for el in self._flat.values():
            names.extend(getattr(el, "controls", ()))
        return tuple(dict.fromkeys(names))

    def describe(self) -> list[dict[str, Any]]:
        """One description per leaf element (type, ports, delays, fit errors)."""
        return [{"name": name, **el.describe(self.dt)} for name, el in self._flat.items()]

    def _check_delays(self) -> None:
        for name, el in self._flat.items():
            if isinstance(el, DelayLineTD):
                try:
                    d = el.delay_samples(self.dt)
                except ValueError:
                    continue
                if d < 1.0:
                    warnings.warn(
                        f"element {name!r}: delay {d * self.dt:.3g} s is shorter than one time step "
                        f"({self.dt:.3g} s); it acts (partly) instantaneously and its interpolation is "
                        "off-centre -- reduce dt if this delay matters",
                        stacklevel=3,
                    )

    # -- record resolution --------------------------------------------------- #
    def _record_terminals(self, record: RecordSpec) -> tuple[list[str], list[_Terminal]]:
        if isinstance(record, str):
            if record == "external":
                return list(self._ports), [self._ports[p] for p in self._ports]
            if record == "all":
                terms = [(name, p) for name, el in self._flat.items() for p in el.ports]
                return [f"{n},{p}" for n, p in terms], terms
            record = [record]
        names: list[str] = []
        terms: list[_Terminal] = []
        for item in record:
            if isinstance(item, tuple):
                key = item
            elif item in self._ports:
                names.append(item)
                terms.append(self._ports[item])
                continue
            elif "," in item:
                inst, port = item.rsplit(",", 1)
                key = (inst, port)
            else:
                raise KeyError(f"cannot record {item!r}: not an external port or an 'instance,port' terminal")
            leaf = self._tmap.get(key, key)
            if leaf[0] not in self._flat or leaf[1] not in self._flat[leaf[0]].ports:
                raise KeyError(f"cannot record {item!r}: unknown terminal")
            names.append(item if isinstance(item, str) else f"{item[0]},{item[1]}")
            terms.append(leaf)
        return names, terms

    # -- assembly with caching ------------------------------------------------ #
    def _prepare(self, overrides: Mapping[str, Any] | None, record_terms: Sequence[_Terminal]):
        if overrides:
            if self._rebuild is None:
                raise ValueError("overrides need a circuit built with TimeDomainCircuit.from_netlist")
            elements = self._rebuild(overrides)
            flat, conns, ports, _ = flatten_elements(elements, self.connections, self.ports)
        else:
            flat, conns, ports = self._flat, self._conns, self._ports
        compiled = {name: el.compile(self.dt) for name, el in flat.items()}
        st, arrays, info = _assemble(compiled, conns, ports, record_terms)
        return st, arrays, info

    def _runner(self, st: _Structure, unroll: int):
        key = ("runner", st, unroll)
        if key not in self._cache:
            self._cache[key] = _make_jax_runner(st, unroll)
        return self._cache[key]

    # -- simulation ------------------------------------------------------------ #
    def simulate(
        self,
        inputs: Mapping[str, Waveform | Any],
        n_steps: int | None = None,
        *,
        controls: Mapping[str, Waveform | Any] | None = None,
        record: RecordSpec = "external",
        overrides: Mapping[str, Mapping[str, Any]] | None = None,
        unroll: int = 8,
    ) -> dict[str, Waveform]:
        """Step the circuit through ``n_steps`` samples.

        Parameters
        ----------
        inputs
            External port ``->`` incident envelope (a :class:`Waveform` with
            this circuit's ``dt`` and carrier, or an array of samples in
            sqrt(W)). Ports not listed are dark. Shorter inputs are zero-padded.
        n_steps
            Number of steps; defaults to the longest input.
        controls
            Control waveforms (volts, mW, ...) for time-varying elements, by
            name (see :attr:`controls`); arrays or electrical
            :class:`Waveform`.
        record
            ``"external"`` (default) records the light leaving every external
            port; ``"all"`` records the outgoing wave at every terminal; or a
            list of external port names and ``"instance,port"`` terminals.
        overrides
            ``{instance: {param: value}}`` re-evaluated on top of the netlist
            settings (circuits from :meth:`from_netlist` only). The static
            layout (integer delays, tap counts) of every element is kept, so
            the values may be JAX tracers: this is how to differentiate a
            time-domain figure of merit with respect to circuit parameters.
        unroll
            Loop unrolling of :func:`jax.lax.scan` (speed only).

        Returns
        -------
        dict
            Recorded name ``->`` outgoing :class:`Waveform` (sqrt(W)).
        """
        names, rec_terms = self._record_terminals(record)
        st, arrays, info = self._prepare(overrides, rec_terms)
        key = ("structure", tuple(rec_terms))
        if overrides:
            ref = self._structure_ref.get(key)
            if ref is None:
                ref_st, _, _ = self._prepare(None, rec_terms)
                self._structure_ref[key] = ref = ref_st
            if ref != st:
                raise ValueError(
                    "the overrides changed the static structure of the network (delays or tap counts); "
                    "only coefficient-level changes can be traced -- rebuild the circuit instead"
                )
        ext = info["ext"]
        t0 = None
        lengths = []
        cols: dict[str, Any] = {}
        for port, sig in inputs.items():
            if port not in self._ports:
                raise KeyError(f"unknown external port {port!r}; ports are {list(self._ports)}")
            if isinstance(sig, Waveform):
                if not math.isclose(sig.dt, self.dt, rel_tol=1e-9):
                    raise ValueError(f"input {port!r} has dt = {sig.dt:g} s but the circuit uses {self.dt:g} s")
                if sig.carrier_wl is None or not math.isclose(sig.carrier_wl, self.carrier_wl, rel_tol=1e-12):
                    raise ValueError(
                        f"input {port!r} must be an optical envelope about the circuit carrier "
                        f"{self.carrier_wl} µm (got carrier_wl={sig.carrier_wl})"
                    )
                if t0 is not None and not math.isclose(sig.t0, t0, rel_tol=0, abs_tol=1e-3 * self.dt):
                    raise ValueError("all input waveforms must share the same start time t0")
                t0 = sig.t0
                data = sig.samples
            else:
                data = sig
            data = xp.asarray(data, dtype=complex)
            if data.ndim != 1:
                raise ValueError(f"input {port!r} must be one-dimensional (time)")
            cols[port] = data
            lengths.append(int(data.shape[0]))
        if n_steps is None:
            if not lengths:
                raise ValueError("give n_steps when no input is provided")
            n_steps = max(lengths)
        n_steps = int(n_steps)
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        u_cols = []
        for port in ext:
            if port in cols:
                d = cols[port]
                m = int(d.shape[0])
                d = d[:n_steps] if m >= n_steps else xp.concatenate([d, xp.zeros(n_steps - m, dtype=complex)])
            else:
                d = xp.zeros(n_steps, dtype=complex)
            u_cols.append(d)
        u_seq = xp.stack(u_cols, axis=1) if u_cols else xp.zeros((n_steps, 0), dtype=complex)
        t0 = 0.0 if t0 is None else float(t0)

        w_seq = None
        if st.n_tv:
            w_seq = self._tv_matrices(info, arrays, n_steps, t0, controls)
        if HAS_JAX:
            outs = self._runner(st, int(unroll))(arrays, u_seq, w_seq)
        else:
            outs = _run_numpy(st, arrays, u_seq, w_seq)
        return {nm: Waveform(outs[:, j], self.dt, t0, self.carrier_wl, "sqrt(W)") for j, nm in enumerate(names)}

    def _tv_matrices(self, info, arrays, n_steps: int, t0: float, controls):
        controls = dict(controls or {})
        t = t0 + self.dt * np.arange(n_steps)
        ctl: dict[str, Any] = {}
        for k, v in controls.items():
            data = v.samples if isinstance(v, Waveform) else v
            data = xp.asarray(data)
            m = int(data.shape[0])
            if m < n_steps:
                data = xp.concatenate([data, xp.broadcast_to(data[-1:], (n_steps - m,))])
            ctl[k] = data[:n_steps]
        blocks = [el.matrices(t, ctl) for _, el in info["tv_blocks"]]
        n_tv = sum(int(b.shape[-1]) for b in blocks)
        s_all = xp.zeros((n_steps, n_tv, n_tv), dtype=complex)
        off = 0
        for b in blocks:
            m = int(b.shape[-1])
            if hasattr(s_all, "at"):
                s_all = s_all.at[:, off:off + m, off:off + m].set(b)
            else:
                s_all[:, off:off + m, off:off + m] = np.asarray(b)
            off += m
        eye = xp.eye(n_tv, dtype=complex)
        lhs = eye[None, :, :] - s_all @ arrays["fz"][None, :, :]
        return xp.linalg.solve(lhs, s_all)

    # -- convenience analyses ------------------------------------------------- #
    def impulse_response(self, in_port: str, n_steps: int, *, record: RecordSpec = "external",
                         overrides: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
        """Unit-sample response: outputs for ``u_0 = 1`` at ``in_port`` (dimensionless).

        The continuous-time impulse response is approximately ``h_k / dt``;
        the DTFT of the unit-sample response is the network's discrete
        frequency response (see :meth:`discrete_response`).
        """
        imp = xp.zeros(int(n_steps), dtype=complex)
        imp = imp.at[0].set(1.0) if hasattr(imp, "at") else _np_first(imp)
        out = self.simulate({in_port: imp}, int(n_steps), record=record, overrides=overrides)
        return {k: v.samples for k, v in out.items()}

    def discrete_response(self, nu, *, control_values: Mapping[str, float] | None = None):
        """Frequency response actually implemented by the discretized network.

        Evaluates every element's ``z``-domain response
        ``D + sum z^-n0 B(z) / A(z)`` at ``z = exp(j 2 pi nu dt)`` and combines
        them with the frequency-domain circuit solver. A time-domain
        simulation reproduces this response exactly (up to round-off), so the
        difference between it and the model's ``S(f_c + nu)`` is the whole
        discretization (interpolation / fit) error.

        Parameters
        ----------
        nu
            Frequency offsets from the carrier in Hz (array).
        control_values
            Constant control values for time-varying elements.

        Returns
        -------
        SDict
            ``(in_port, out_port) -> array over nu`` for the external ports.
        """
        from photonix.circuit.solver import evaluate_circuit

        nu = np.atleast_1d(np.asarray(nu, dtype=float))
        z = np.exp(2j * np.pi * nu * self.dt)
        sd: dict[str, dict] = {}
        for name, el in self._flat.items():
            c = el.compile(self.dt)
            n = len(c.ports)
            if c.time_varying is not None:
                vals = dict(control_values or {})
                ctl = {k: np.asarray([vals.get(k, 0.0)]) for k in c.time_varying.controls}
                mat = np.broadcast_to(np.asarray(c.time_varying.matrices(np.zeros(1), ctl))[0], (nu.size, n, n))
            else:
                mat = np.broadcast_to(np.asarray(c.D, dtype=complex), (nu.size, n, n)).copy()
                idx = {p: i for i, p in enumerate(c.ports)}
                for ch in c.channels:
                    b = np.asarray(ch.b, dtype=complex)
                    num = sum(b[m] * z ** (-(ch.n0 + m)) for m in range(b.size))
                    den = 1.0 + 0j
                    if ch.a is not None:
                        a = np.asarray(ch.a, dtype=complex)
                        den = 1.0 + sum(a[p] * z ** (-(p + 1)) for p in range(a.size))
                    mat[:, idx[ch.dst], idx[ch.src]] += num / den
            sd[name] = {(pi, po): mat[:, i_o, i_i] for i_i, pi in enumerate(c.ports) for i_o, po in enumerate(c.ports)
                        if np.any(mat[:, i_o, i_i] != 0)}
            if not sd[name]:
                sd[name] = {(c.ports[0], c.ports[0]): np.zeros(nu.size, dtype=complex)}
        return evaluate_circuit(sd, self._conns, self._ports)


def _np_first(arr):
    arr = np.array(arr, copy=True)
    arr[0] = 1.0
    return arr
