"""Differentiable S-parameter circuit solver.

Given the scattering dictionary of every instance plus the internal connections
and exposed ports, the solver returns the composite :data:`SDict` of the whole
circuit. The method is a single linear solve and therefore:

* handles arbitrary topologies **including feedback loops** (ring resonators);
* is fully differentiable and ``jit``-able (it is just matrix algebra);
* vectorizes over a wavelength/parameter batch dimension.

Algorithm
---------
Stack every instance port into one wave vector. With ``b = S a`` (output waves
from input waves, block-diagonal ``S``) and the interconnection relation
``a = Gamma b + u`` (``Gamma`` routes each internal connection ``a_p = b_q``;
``u`` injects light at exposed ports), eliminating ``b`` gives::

    b = (I - S Gamma)^{-1} S u

so the composite transfer matrix is ``M = (I - S Gamma)^{-1} S`` and the
exposed-port block of ``M`` is the circuit's scattering matrix.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

from photonix.core.backend import xp
from photonix.core.types import Model, SDict, ports_of

from .netlist import Netlist

__all__ = ["evaluate_circuit", "circuit_from_netlist", "mzi", "ring"]


def _broadcast_batch(sdicts: Mapping[str, SDict]) -> tuple[int, ...]:
    batch: tuple[int, ...] = ()
    for sd in sdicts.values():
        for v in sd.values():
            batch = xp.broadcast_shapes(batch, xp.asarray(v).shape)
    return batch


def evaluate_circuit(
    instances: Mapping[str, SDict],
    connections: Mapping[tuple[str, str], tuple[str, str]],
    ports: Mapping[str, tuple[str, str]],
) -> SDict:
    """Combine instance ``SDict``s into the composite circuit ``SDict``.

    Parameters
    ----------
    instances
        ``instance_name -> SDict`` for every placed component.
    connections
        Internal links ``(inst_a, port_a) -> (inst_b, port_b)``.
    ports
        Exposed ports ``external_name -> (instance, port)``.

    Returns
    -------
    SDict
        Keyed by ``(external_in, external_out)`` over the exposed ports.

    Examples
    --------
    >>> import photonix as px
    >>> import photonix.components as c
    >>> A = c.straight(wl=1.55, length=10.0)
    >>> B = c.straight(wl=1.55, length=10.0)
    >>> s = px.circuit.evaluate_circuit(
    ...     {"a": A, "b": B},
    ...     {("a", "o2"): ("b", "o1")},
    ...     {"in0": ("a", "o1"), "out0": ("b", "o2")},
    ... )
    >>> abs(float(px.power(s[("in0", "out0")])) - 1.0) < 1e-9
    True
    """
    # 1. Namespaced terminal list and index map.
    terminals: list[tuple[str, str]] = []
    for inst, sd in instances.items():
        for p in ports_of(sd):
            terminals.append((inst, p))
    index = {t: i for i, t in enumerate(terminals)}
    n = len(terminals)
    if n == 0:
        return {}

    batch = _broadcast_batch(instances)

    # 2. Block-diagonal scattering matrix S with row=output, col=input.
    S = xp.zeros((*batch, n, n), dtype=complex)
    for inst, sd in instances.items():
        for (p_in, p_out), val in sd.items():
            i = index[(inst, p_out)]
            j = index[(inst, p_in)]
            v = xp.broadcast_to(xp.asarray(val, dtype=complex), batch) if batch else xp.asarray(val, dtype=complex)
            S = S.at[..., i, j].set(v) if hasattr(S, "at") else _np_set(S, i, j, v)

    # 3. Interconnection matrix Gamma: a_p = b_q for each connected (p, q).
    Gamma = xp.zeros((n, n), dtype=complex)
    for a, b in connections.items():
        if a not in index or b not in index:
            raise KeyError(f"Connection {a} <-> {b} references an unknown instance port.")
        ia, ib = index[a], index[b]
        if hasattr(Gamma, "at"):
            Gamma = Gamma.at[ia, ib].set(1.0).at[ib, ia].set(1.0)
        else:
            Gamma[ia, ib] = 1.0
            Gamma[ib, ia] = 1.0

    # 4. Solve (I - S Gamma) M = S  ->  M = (I - S Gamma)^{-1} S.
    eye = xp.eye(n, dtype=complex)
    A = eye - S @ Gamma
    M = xp.linalg.solve(A, S)

    # 5. Extract exposed-port block. key = (external_in, external_out).
    out: SDict = {}
    for e_in, t_in in ports.items():
        ji = index[t_in]
        for e_out, t_out in ports.items():
            io = index[t_out]
            out[(e_in, e_out)] = M[..., io, ji]
    return out


def _np_set(S, i, j, v):  # NumPy fallback path
    S = S.copy()
    S[..., i, j] = v
    return S


def circuit_from_netlist(netlist: Netlist, models: Mapping[str, Callable]) -> Model:
    """Compile a :class:`Netlist` + model registry into a differentiable model.

    Returns a callable ``f(*, wl=1.55, **overrides) -> SDict`` where ``overrides``
    is ``{instance_name: {param: value}}`` applied on top of the netlist settings.

    Examples
    --------
    >>> import photonix as px
    >>> f = px.circuit.mzi(delta_length=20.0)
    >>> s = f(wl=1.55)
    >>> ("in0", "out0") in s
    True
    """
    netlist.validate()

    def model(*, wl=1.55, **overrides) -> SDict:
        inst_sdicts: dict[str, SDict] = {}
        for inst, model_name in netlist.instances.items():
            if model_name not in models:
                raise KeyError(f"Model {model_name!r} for instance {inst!r} not in registry.")
            settings = netlist.merged_settings(inst, overrides.get(inst))
            inst_sdicts[inst] = models[model_name](wl=wl, **settings)
        return evaluate_circuit(inst_sdicts, netlist.connections, netlist.ports)

    model.__name__ = netlist.name or "circuit"
    return model


# --------------------------------------------------------------------------- #
# Convenience circuit builders (compose the component library)
# --------------------------------------------------------------------------- #
def mzi(
    *,
    delta_length: float = 20.0,
    length: float = 50.0,
    coupling: float = 0.5,
    neff: float = 2.4,
    ng: float = 4.2,
    loss_db_cm: float = 0.0,
) -> Model:
    """Build a Mach-Zehnder interferometer as a real two-coupler circuit.

    The returned model is assembled from two directional couplers and two
    waveguide arms via the circuit solver (not the analytic shortcut), so its
    gradients flow through the full interconnection. Exposed ports
    ``in0``/``in1`` (inputs) and ``out0``/``out1`` (outputs), labelled to match
    :func:`photonix.components.mzi`: ``out0`` is the bar output (the
    ``sin^2(dphi/2)`` fringe for 50/50 couplers, through-through + cross-cross
    paths) and ``out1`` the cross output.
    """
    from photonix import components as _c

    models = {"coupler": _c.directional_coupler, "straight": _c.straight}
    nl = Netlist(name="mzi")
    nl.add("c1", "coupler", coupling=coupling)
    nl.add("c2", "coupler", coupling=coupling)
    nl.add("top", "straight", length=length + delta_length, neff=neff, ng=ng, loss_db_cm=loss_db_cm)
    nl.add("bot", "straight", length=length, neff=neff, ng=ng, loss_db_cm=loss_db_cm)
    nl.connect(("c1", "o4"), ("top", "o1"))
    nl.connect(("top", "o2"), ("c2", "o1"))
    nl.connect(("c1", "o3"), ("bot", "o1"))
    nl.connect(("bot", "o2"), ("c2", "o2"))
    nl.expose("in0", ("c1", "o1"))
    nl.expose("in1", ("c1", "o2"))
    # Port labels match the analytic components.mzi: out0 = bar (in0 goes
    # through-through via the top arm into c2.o4, plus cross-cross via the
    # bottom arm), out1 = cross (c2.o3).
    nl.expose("out0", ("c2", "o4"))
    nl.expose("out1", ("c2", "o3"))
    return circuit_from_netlist(nl, models)


def ring(
    *,
    radius: float = 10.0,
    coupling: float = 0.2,
    neff: float = 2.4,
    ng: float = 4.2,
    loss_db_cm: float = 2.0,
) -> Model:
    """Build an all-pass (single-bus) ring resonator as a real feedback circuit.

    A directional coupler with its drop side closed into a ring waveguide. The
    solver resolves the feedback loop. Exposed ports ``in0`` -> ``out0``.
    """
    from photonix import components as _c

    models = {"coupler": _c.directional_coupler, "straight": _c.straight}
    circ = 2.0 * 3.141592653589793 * radius
    nl = Netlist(name="ring")
    nl.add("cpl", "coupler", coupling=coupling)
    nl.add("rw", "straight", length=circ, neff=neff, ng=ng, loss_db_cm=loss_db_cm)
    nl.connect(("cpl", "o3"), ("rw", "o1"))
    nl.connect(("rw", "o2"), ("cpl", "o2"))
    nl.expose("in0", ("cpl", "o1"))
    nl.expose("out0", ("cpl", "o4"))
    return circuit_from_netlist(nl, models)
