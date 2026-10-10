"""Circuit builders for common multi-element photonic architectures.

Every builder returns a differentiable :class:`~photonix.core.types.Model`
assembled by the S-parameter solver from library components, so its response
includes every interconnection and its gradients flow through every element.

* :func:`lattice_filter` -- cascaded-MZI (FIR) lattice filter.
* :func:`crow` -- coupled-resonator optical waveguide (series-coupled rings).
* :func:`sagnac_mirror` -- loop mirror from a 2x2 coupler.
* :func:`clements_decompose` / :func:`mzi_mesh` -- program an arbitrary
  NxN unitary onto a rectangular (Clements) mesh of MZIs and simulate it,
  optionally with imperfect splitters and lossy phase shifters.

Physical MZI convention
-----------------------
A mesh unit is ``M(theta, phi) = B P(theta) B P(phi)``: an external phase
shifter ``phi`` on the upper input, a 50:50 coupler ``B``, an internal phase
shifter ``theta`` on the upper arm, and a second coupler. With this library's
coupler (bar ``1/sqrt 2``, cross ``-j/sqrt 2``) and phase-shifter
(``exp(-j phase)``) conventions::

    M = -j e^{-j theta/2} [[e^{-j phi} sin(theta/2),  cos(theta/2)],
                            [e^{-j phi} cos(theta/2), -sin(theta/2)]]

:func:`clements_decompose` is derived for exactly this matrix (rather than the
abstract ``T`` of Clements et al., Optica 3, 1460, 2016), so the returned
``theta``/``phi`` are the heater phases to apply.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from photonix.core.types import Model

from .netlist import Netlist
from .solver import circuit_from_netlist

__all__ = [
    "lattice_filter", "crow", "sagnac_mirror",
    "MeshProgram", "clements_decompose", "mzi_unit_matrix", "mesh_matrix", "mzi_mesh",
]


def _models():
    from photonix import components as c

    return {
        "coupler": c.directional_coupler,
        "straight": c.straight,
        "phase_shifter": c.phase_shifter,
    }


# --------------------------------------------------------------------------- #
# Lattice filter
# --------------------------------------------------------------------------- #
def lattice_filter(
    *,
    couplings: Sequence[float] = (0.5, 0.5),
    delta_lengths: Sequence[float] = (20.0,),
    phases: Sequence[float] | None = None,
    length: float = 10.0,
    neff: float = 2.4,
    ng: float = 4.2,
    loss_db_cm: float = 0.0,
) -> Model:
    """Cascaded-MZI lattice filter with ``len(couplings)`` couplers.

    Stage ``i`` joins coupler ``i`` to coupler ``i + 1`` through an upper arm
    of ``length + delta_lengths[i]`` and a lower arm of ``length``, plus an
    optional extra upper-arm phase ``phases[i]`` (radians). Ports follow the
    2x2 coupler: ``o1``/``o2`` in (upper/lower), ``o4``/``o3`` out
    (upper/lower).

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> f = px.circuit.lattice_filter(couplings=(0.5, 0.5), delta_lengths=(20.0,))
    >>> s = f(wl=1.55)
    >>> m = px.circuit.mzi(delta_length=20.0, length=10.0)(wl=1.55)
    >>> bool(np.isclose(complex(s[("o1", "o4")]), complex(m[("o1", "o4")])))
    True
    """
    couplings = list(couplings)
    delta_lengths = list(delta_lengths)
    if len(couplings) != len(delta_lengths) + 1:
        raise ValueError("a lattice needs exactly one more coupler than delay stages")
    phases = [0.0] * len(delta_lengths) if phases is None else list(phases)
    if len(phases) != len(delta_lengths):
        raise ValueError("phases must have one entry per delay stage")
    nl = Netlist(name="lattice_filter")
    for i, k in enumerate(couplings):
        nl.add(f"c{i}", "coupler", coupling=k)
    wg = {"neff": neff, "ng": ng, "loss_db_cm": loss_db_cm}
    for i, (dl, ph) in enumerate(zip(delta_lengths, phases, strict=True)):
        nl.add(f"top{i}", "straight", length=length + dl, **wg)
        nl.add(f"bot{i}", "straight", length=length, **wg)
        nl.add(f"ps{i}", "phase_shifter", phase0=ph, length=0.0)
        nl.connect((f"c{i}", "o4"), (f"top{i}", "o1"))
        nl.connect((f"top{i}", "o2"), (f"ps{i}", "o1"))
        nl.connect((f"ps{i}", "o2"), (f"c{i + 1}", "o1"))
        nl.connect((f"c{i}", "o3"), (f"bot{i}", "o1"))
        nl.connect((f"bot{i}", "o2"), (f"c{i + 1}", "o2"))
    last = len(couplings) - 1
    nl.expose("o1", ("c0", "o1"))
    nl.expose("o2", ("c0", "o2"))
    nl.expose("o4", (f"c{last}", "o4"))
    nl.expose("o3", (f"c{last}", "o3"))
    return circuit_from_netlist(nl, _models())


# --------------------------------------------------------------------------- #
# Coupled-resonator optical waveguide
# --------------------------------------------------------------------------- #
def crow(
    *,
    couplings: Sequence[float] = (0.2, 0.02, 0.2),
    radius: float = 10.0,
    neff: float = 2.4,
    ng: float = 4.2,
    loss_db_cm: float = 2.0,
) -> Model:
    """Series-coupled ring filter between two buses (``len(couplings) - 1`` rings).

    ``couplings[0]`` is the input bus to ring 1, ``couplings[k]`` ring ``k`` to
    ring ``k+1``, and ``couplings[-1]`` the last ring to the drop bus. Ports:
    ``o1`` input and ``o2`` through on the input bus; ``o3`` drop and ``o4`` add
    on the drop bus. Adjacent rings counter-rotate, so the drop port is on the
    input side of the drop bus for an odd ring count and on the far side for an
    even count; the builder wires ``o3``/``o4`` accordingly. One ring reproduces
    :func:`photonix.components.add_drop_ring`.

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> wl = px.linspace(1.54, 1.56, 4001)
    >>> s = px.circuit.crow(couplings=(0.3, 0.05, 0.3), loss_db_cm=0.0)(wl=wl)
    >>> total = px.power(s[("o1", "o2")]) + px.power(s[("o1", "o3")])
    >>> bool(np.allclose(px.to_numpy(total), 1.0))      # lossless: energy conserved
    True
    """
    couplings = list(couplings)
    n_rings = len(couplings) - 1
    if n_rings < 1:
        raise ValueError("crow needs at least two couplings (one ring)")
    half = math.pi * radius
    wg = {"length": half, "neff": neff, "ng": ng, "loss_db_cm": loss_db_cm}
    nl = Netlist(name="crow")
    # Coupler k: o1/o4 is its upper guide (left/right), o2/o3 its lower guide.
    for k, kappa in enumerate(couplings):
        nl.add(f"c{k}", "coupler", coupling=kappa)
    for r in range(1, n_rings + 1):
        nl.add(f"r{r}_right", "straight", **wg)
        nl.add(f"r{r}_left", "straight", **wg)
        # Ring r: lower half inside coupler r-1 (its upper guide), upper half
        # inside coupler r (its lower guide).
        nl.connect((f"c{r - 1}", "o4"), (f"r{r}_right", "o1"))
        nl.connect((f"r{r}_right", "o2"), (f"c{r}", "o3"))
        nl.connect((f"c{r}", "o2"), (f"r{r}_left", "o1"))
        nl.connect((f"r{r}_left", "o2"), (f"c{r - 1}", "o1"))
    nl.expose("o1", ("c0", "o2"))
    nl.expose("o2", ("c0", "o3"))
    top = f"c{n_rings}"
    if n_rings % 2:
        nl.expose("o3", (top, "o1"))
        nl.expose("o4", (top, "o4"))
    else:
        nl.expose("o3", (top, "o4"))
        nl.expose("o4", (top, "o1"))
    return circuit_from_netlist(nl, _models())


# --------------------------------------------------------------------------- #
# Loop mirror
# --------------------------------------------------------------------------- #
def sagnac_mirror(
    *,
    coupling: float = 0.5,
    loop_length: float = 50.0,
    neff: float = 2.4,
    ng: float = 4.2,
    loss_db_cm: float = 0.0,
) -> Model:
    """Sagnac loop mirror: a 2x2 coupler whose outputs are joined by a loop.

    Ports ``o1``/``o2``. For a lossless loop the reflection is
    ``|S11|^2 = 4 k (1 - k)`` and the transmission ``|S21|^2 = (1 - 2k)^2``, so
    a 50:50 coupler reflects everything.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.circuit.sagnac_mirror(coupling=0.5)(wl=1.55)
    >>> round(float(px.power(s[("o1", "o1")])), 12)
    1.0
    """
    nl = Netlist(name="sagnac_mirror")
    nl.add("c", "coupler", coupling=coupling)
    nl.add("loop", "straight", length=loop_length, neff=neff, ng=ng, loss_db_cm=loss_db_cm)
    nl.connect(("c", "o4"), ("loop", "o1"))
    nl.connect(("loop", "o2"), ("c", "o3"))
    nl.expose("o1", ("c", "o1"))
    nl.expose("o2", ("c", "o2"))
    return circuit_from_netlist(nl, _models())


# --------------------------------------------------------------------------- #
# Programmable unitary meshes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MeshProgram:
    """Phase settings of a rectangular MZI mesh.

    Attributes
    ----------
    n
        Number of modes.
    mzis
        ``(mode, theta, phi)`` per MZI in the order light traverses them;
        ``mode`` is the upper of the two adjacent modes it mixes.
    output_phases
        Phase-shifter settings (radians) of the final output phase screen.
    """

    n: int
    mzis: tuple[tuple[int, float, float], ...]
    output_phases: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        phases = np.array(self.output_phases, dtype=float)
        phases.setflags(write=False)
        object.__setattr__(self, "output_phases", phases)
        object.__setattr__(self, "mzis", tuple((int(m), float(t), float(p)) for m, t, p in self.mzis))

    def __eq__(self, other) -> bool:
        if not isinstance(other, MeshProgram):
            return NotImplemented
        return (self.n == other.n and self.mzis == other.mzis
                and tuple(self.output_phases) == tuple(other.output_phases))

    def __hash__(self) -> int:
        return hash((self.n, self.mzis, tuple(self.output_phases)))

    @property
    def n_mzis(self) -> int:
        return len(self.mzis)


def mzi_unit_matrix(theta: float, phi: float) -> np.ndarray:
    """The 2x2 transfer matrix of one physical mesh MZI (see module docs)."""
    s, c = math.sin(theta / 2), math.cos(theta / 2)
    e = np.exp(-1j * phi)
    return -1j * np.exp(-1j * theta / 2) * np.array([[e * s, c], [e * c, -s]])


def _embed(n: int, m: int, block: np.ndarray) -> np.ndarray:
    out = np.eye(n, dtype=complex)
    out[m:m + 2, m:m + 2] = block
    return out


def mesh_matrix(program: MeshProgram) -> np.ndarray:
    """Reconstruct the unitary a :class:`MeshProgram` implements (NumPy)."""
    u = np.eye(program.n, dtype=complex)
    for mode, theta, phi in program.mzis:
        u = _embed(program.n, mode, mzi_unit_matrix(theta, phi)) @ u
    return np.diag(np.exp(-1j * np.asarray(program.output_phases))) @ u


def clements_decompose(unitary, *, atol: float = 1e-9) -> MeshProgram:
    """Program an NxN unitary onto a rectangular MZI mesh.

    Follows the nulling order of Clements et al. (2016): alternately null the
    lower-left elements with MZIs acting on columns (input side) and rows
    (output side), leaving a diagonal phase screen; the output-side MZIs are
    then moved through that screen. All ``N(N-1)/2`` MZIs fit in a depth-``N``
    rectangular mesh, so optical paths are balanced.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import unitary_group
    >>> U = unitary_group.rvs(4, random_state=1)
    >>> prog = clements_decompose(U)
    >>> prog.n_mzis, bool(np.allclose(mesh_matrix(prog), U))
    (6, True)
    """
    u = np.array(unitary, dtype=complex)
    n = u.shape[0]
    if u.ndim != 2 or u.shape != (n, n) or n < 2:
        raise ValueError("unitary must be a square matrix of size >= 2")
    if not np.allclose(u.conj().T @ u, np.eye(n), atol=max(atol, 1e-12) * 1e3):
        raise ValueError("matrix is not unitary")

    right: list[tuple[int, float, float]] = []   # applied as U <- U M^dagger (input side)
    left: list[tuple[int, float, float]] = []    # applied as U <- M U (output side)
    for i in range(n - 1):
        if i % 2 == 0:
            for j in range(i + 1):
                r, col = n - 1 - j, i - j          # null U[r, col] with columns (col, col+1)
                a, b = u[r, col], u[r, col + 1]
                theta = 2.0 * math.atan2(abs(b), abs(a))
                # angle(-b) - angle(a), not angle(-b/a): the quotient overflows
                # for subnormal |a| although the angle is well defined.
                phi = float(np.angle(-b) - np.angle(a)) if abs(a) > 0 else 0.0
                u = u @ _embed(n, col, mzi_unit_matrix(theta, phi)).conj().T
                right.append((col, theta, phi))
        else:
            for j in range(i + 1):
                q, col = n - 1 - i + j, j          # null U[q, col] with rows (q-1, q)
                a, b = u[q - 1, col], u[q, col]
                theta = 2.0 * math.atan2(abs(a), abs(b))
                phi = float(np.angle(a) - np.angle(b)) if abs(a) > 0 and abs(b) > 0 else 0.0
                u = _embed(n, q - 1, mzi_unit_matrix(theta, phi)) @ u
                left.append((q - 1, theta, phi))
    d = np.diag(u).copy()
    if not np.allclose(u, np.diag(d), atol=1e3 * max(atol, 1e-12)):
        raise RuntimeError("Clements nulling did not reach a diagonal matrix")  # pragma: no cover

    # U = L_1^+ ... L_K^+ D R_J ... R_1. Push each L^+ through D:
    # M(t, f)^+ diag(d1, d2) = diag(-e^{j(t+f)} d2, -e^{jt} d2) M(t, angle(d2) - angle(d1)).
    moved: list[tuple[int, float, float]] = []
    for mode, theta, phi in reversed(left):
        d1, d2 = d[mode], d[mode + 1]
        new_phi = float(np.angle(d2) - np.angle(d1))
        d[mode] = -np.exp(1j * (theta + phi)) * d2
        d[mode + 1] = -np.exp(1j * theta) * d2
        moved.append((mode, theta, new_phi))
    # U = D' M'_1 ... M'_K R_J ... R_1: light meets R_1 first and M'_1 last, and
    # ``moved`` was built from L_K down to L_1, i.e. already in traversal order.
    order = list(right) + moved
    return MeshProgram(n=n, mzis=tuple(order), output_phases=np.mod(-np.angle(d), 2 * np.pi))


def mzi_mesh(
    program: MeshProgram,
    *,
    splitter_coupling: float = 0.5,
    phase_loss_db: float = 0.0,
) -> Model:
    """Simulate a programmed MZI mesh as a full component-level circuit.

    Each MZI is built from two :func:`~photonix.components.directional_coupler`
    instances and two :func:`~photonix.components.phase_shifter` instances, so
    fabrication imperfections can be studied directly: ``splitter_coupling``
    sets every coupler's power splitting (0.5 is ideal) and ``phase_loss_db``
    adds insertion loss to every phase shifter. Ports: inputs ``o1..oN`` (mode
    0 at the top) and outputs ``o(2N)..o(N+1)`` (mode 0 at ``o(2N)``), the
    counter-clockwise numbering of :func:`~photonix.components.mmi_nxn`, so
    ``S[(f"o{i+1}", f"o{2N-j}")] = U[j, i]``.

    Phase settings can be overridden at call time per instance, e.g.
    ``model(wl=1.55, int3={"phase0": 0.4})``.

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> from scipy.stats import unitary_group
    >>> U = unitary_group.rvs(3, random_state=7)
    >>> s = px.circuit.mzi_mesh(px.circuit.clements_decompose(U))(wl=1.55)
    >>> bool(np.isclose(complex(s[("o1", "o5")]), U[1, 0]))
    True
    """
    n = program.n
    nl = Netlist(name=f"mzi_mesh_{n}")
    current: list[tuple[str, str] | None] = [None] * n   # open terminal per mode
    starts: list[tuple[str, str] | None] = [None] * n

    def attach(mode: int, terminal_in: tuple[str, str], terminal_out: tuple[str, str]):
        previous = current[mode]
        if previous is None:
            starts[mode] = terminal_in
        else:
            nl.connect(previous, terminal_in)
        current[mode] = terminal_out

    for k, (mode, theta, phi) in enumerate(program.mzis):
        nl.add(f"ext{k}", "attenuated_phase", phase0=phi, loss_db=phase_loss_db)
        nl.add(f"bs{k}a", "coupler", coupling=splitter_coupling)
        nl.add(f"int{k}", "attenuated_phase", phase0=theta, loss_db=phase_loss_db)
        nl.add(f"bs{k}b", "coupler", coupling=splitter_coupling)
        nl.connect((f"ext{k}", "o2"), (f"bs{k}a", "o1"))
        nl.connect((f"bs{k}a", "o4"), (f"int{k}", "o1"))
        nl.connect((f"int{k}", "o2"), (f"bs{k}b", "o1"))
        nl.connect((f"bs{k}a", "o3"), (f"bs{k}b", "o2"))
        attach(mode, (f"ext{k}", "o1"), (f"bs{k}b", "o4"))
        attach(mode + 1, (f"bs{k}a", "o2"), (f"bs{k}b", "o3"))
    for mode in range(n):
        nl.add(f"out{mode}", "attenuated_phase", phase0=float(program.output_phases[mode]), loss_db=phase_loss_db)
        attach(mode, (f"out{mode}", "o1"), (f"out{mode}", "o2"))
        first, last = starts[mode], current[mode]
        assert first is not None and last is not None
        nl.expose(f"o{mode + 1}", first)
        nl.expose(f"o{2 * n - mode}", last)

    def attenuated_phase(*, wl=1.55, phase0=0.0, loss_db=0.0):
        from photonix import components as c

        base = c.phase_shifter(wl=wl, length=0.0, phase0=phase0)
        amp = 10.0 ** (-loss_db / 20.0)
        return {k: amp * v for k, v in base.items()}

    models = {**_models(), "attenuated_phase": attenuated_phase}
    return circuit_from_netlist(nl, models)
