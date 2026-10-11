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

The numerical work is done by :mod:`photonix.circuit.engine`, which compiles
the topology once, gathers all entry values in one stack, solves only for the
exposed columns and picks a dense, sparse-LU or network-reduction backend.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import cast

from photonix.core.types import Model, SDict

from . import engine as _engine
from .engine import Backend
from .netlist import Netlist

__all__ = ["evaluate_circuit", "circuit_from_netlist", "mzi", "ring"]

#: Legacy semantic names accepted on lookup for the built-in circuit builders.
MZI_PORT_ALIASES = {"in0": "o1", "in1": "o2", "out0": "o4", "out1": "o3"}
RING_PORT_ALIASES = {"in0": "o1", "out0": "o2"}


def evaluate_circuit(
    instances: Mapping[str, SDict],
    connections: Mapping[tuple[str, str], tuple[str, str]],
    ports: Mapping[str, tuple[str, str]],
    *,
    backend: Backend | str = "auto",
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
    backend
        Linear-solver backend of :mod:`photonix.circuit.engine`: ``"auto"``
        (dense for small circuits, sparse LU otherwise), ``"dense"``,
        ``"sparse"`` or ``"reduce"``. All solve the same equations; see the
        engine docs for their transformation support and cost.

    Returns
    -------
    SDict
        Keyed by ``(external_in, external_out)`` over the exposed ports.

    Raises
    ------
    CircuitSingularError
        If ``I - S Gamma`` is singular (a lossless resonance no exposed port
        excites) for concrete inputs.

    Notes
    -----
    An instance terminal that is neither connected nor exposed is terminated
    in a matched load (its incident wave is zero). This is useful for unused
    coupler ports, but is not equivalent to an open or reflective termination;
    model those explicitly when required. Instances with no path to any
    exposed port cannot affect the result and are skipped with a
    :class:`~photonix.circuit.engine.FloatingSubcircuitWarning`.

    The value-free structure is compiled once per distinct topology and
    cached (:data:`photonix.circuit.engine.TOPOLOGY_CACHE`), so repeated calls
    on one circuit only gather values and solve.

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
    topology = _engine.TOPOLOGY_CACHE.get(instances, connections, ports)
    if topology.m == 0:
        return {}
    return _engine.solve(topology, _engine.gather_values(topology, instances), backend=backend)


def circuit_from_netlist(
    netlist: Netlist,
    models: Mapping[str, Callable] | None = None,
    *,
    port_aliases: Mapping[str, str] | None = None,
    backend: Backend | str = "auto",
) -> Model:
    """Compile a :class:`Netlist` + model registry into a differentiable model.

    Returns a callable ``f(*, wl=1.55, **overrides) -> SDict`` where ``overrides``
    is ``{instance_name: {param: value}}`` applied on top of the netlist settings.

    Parameters
    ----------
    netlist
        The circuit topology.
    models
        ``model_name -> callable`` registry. Defaults to
        :data:`photonix.components.MODELS`, so a netlist extracted from a layout
        (:func:`photonix.layout.extract_netlist`, whose model names are the
        layout cell names) simulates without any extra wiring.
    port_aliases
        Optional legacy port names resolved on lookup only; see
        :class:`~photonix.core.types.AliasedSDict`.
    backend
        Engine backend, as for :func:`evaluate_circuit`.

    Notes
    -----
    Every instance model is evaluated on every call (no deduplication), and
    the compiled topology is cached on the returned model. Use
    :func:`photonix.circuit.compile_circuit` to also evaluate identical
    instances once per call.

    Examples
    --------
    >>> import photonix as px
    >>> f = px.circuit.mzi(delta_length=20.0)
    >>> s = f(wl=1.55)
    >>> ("o1", "o4") in s and ("in0", "out0") in s     # canonical + legacy
    True

    A layout round-trips straight into a simulation, no registry required:

    >>> from photonix.layout import Cell, components, extract_netlist
    >>> top = Cell("top")
    >>> _ = top.add_ref(components.straight(10.0), origin=(0, 0), name="a")
    >>> _ = top.add_ref(components.straight(10.0), origin=(10, 0), name="b")
    >>> s = px.circuit.circuit_from_netlist(extract_netlist(top))(wl=1.55)
    >>> abs(float(px.power(s[("a_o1", "b_o2")])) - 1.0) < 1e-9
    True
    """
    return _engine.compile_circuit(
        netlist,
        cast("Mapping[str, Callable[..., SDict]] | None", models),
        backend=backend,
        dedupe=False,
        port_aliases=port_aliases,
    )


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
    gradients flow through the full interconnection.

    Ports follow the 2x2 coupler convention of
    :func:`photonix.components.directional_coupler`: ``o1``/``o2`` on the input
    side, ``o3``/``o4`` on the output side, with the bar path ``o1 -> o4`` (the
    ``sin^2(dphi/2)`` fringe for 50/50 couplers, through-through + cross-cross)
    and the cross path ``o1 -> o3``. The legacy names ``in0``/``in1``/``out0``/
    ``out1`` remain valid lookup aliases.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.circuit.mzi(delta_length=20.0)(wl=1.55)
    >>> abs(complex(s[("in0", "out0")]) - complex(s[("o1", "o4")])) < 1e-12
    True
    """
    from photonix import components as _c

    models: dict[str, Callable[..., SDict]] = {
        "coupler": _c.directional_coupler,
        "straight": _c.straight,
    }
    nl = Netlist(name="mzi")
    nl.add("c1", "coupler", coupling=coupling)
    nl.add("c2", "coupler", coupling=coupling)
    nl.add("top", "straight", length=length + delta_length, neff=neff, ng=ng, loss_db_cm=loss_db_cm)
    nl.add("bot", "straight", length=length, neff=neff, ng=ng, loss_db_cm=loss_db_cm)
    nl.connect(("c1", "o4"), ("top", "o1"))
    nl.connect(("top", "o2"), ("c2", "o1"))
    nl.connect(("c1", "o3"), ("bot", "o1"))
    nl.connect(("bot", "o2"), ("c2", "o2"))
    nl.expose("o1", ("c1", "o1"))
    nl.expose("o2", ("c1", "o2"))
    # Bar output is c2.o4 (through-through via the top arm plus cross-cross via
    # the bottom arm); cross output is c2.o3 -- same sense as the coupler.
    nl.expose("o4", ("c2", "o4"))
    nl.expose("o3", ("c2", "o3"))
    return circuit_from_netlist(nl, models, port_aliases=MZI_PORT_ALIASES)


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
    solver resolves the feedback loop. Exposed ports ``o1`` (input) -> ``o2``
    (through); ``in0``/``out0`` remain valid lookup aliases.
    """
    from photonix import components as _c

    models: dict[str, Callable[..., SDict]] = {
        "coupler": _c.directional_coupler,
        "straight": _c.straight,
    }
    circ = 2.0 * math.pi * radius
    nl = Netlist(name="ring")
    nl.add("cpl", "coupler", coupling=coupling)
    nl.add("rw", "straight", length=circ, neff=neff, ng=ng, loss_db_cm=loss_db_cm)
    nl.connect(("cpl", "o3"), ("rw", "o1"))
    nl.connect(("rw", "o2"), ("cpl", "o2"))
    nl.expose("o1", ("cpl", "o1"))
    nl.expose("o2", ("cpl", "o4"))
    return circuit_from_netlist(nl, models, port_aliases=RING_PORT_ALIASES)
