"""Multimode and polarization-aware circuits on top of the single-mode engine.

A port that carries several guided modes is represented by one *modal
terminal* per mode, named ``"<port>:<mode>"`` (``"o1:TE0"``, ``"o2:TM0"``).
Modal terminals are ordinary :data:`~photonix.core.types.SDict` port names, so
the circuit engine solves multimode circuits unchanged; this module only
rewrites topology:

* :func:`lift_model` turns a single-mode model into a block-diagonal multimode
  one, evaluating it once per mode with per-mode settings (``neff``/``ng`` of
  each polarization, say) and optionally applying a port-side inter-mode
  coupling matrix ``C(wl)`` as ``S' = C_out S C_in^T`` (reciprocal, and
  lossless when ``C`` is unitary).
* :class:`ModalNetlist` expands a physical netlist: every physical connection
  ``(A, p) <-> (B, q)`` becomes ``(A, "p:m") <-> (B, "q:m")`` for every mode
  ``m`` both sides carry. A mode present on one side only is terminated in a
  matched load with a warning that names the dropped power path (or raises).
  Single-mode library models are lifted automatically to the modes declared
  for their instance.
* :func:`eme_to_modal` and :func:`meep_dataset_to_modal` give EME and Meep
  multimode S-matrices modal terminal names, replacing their two incompatible
  native conventions (``o1..o{n_in+n_out}`` and ``"o1:m1"``).

Evidence
--------
Level 3 (independent computation): single-mode modal netlists reproduce the
circuit builders exactly; an EME taper cut at its midpoint and recombined
through a :class:`ModalNetlist` equals a single EME run; a synthetic Meep-style
modal dataset cascaded with a lifted multimode straight equals the analytic
matrix product. See ``tests/test_modal_circuits.py`` and
``docs/HIERARCHY_AND_INTEROP.md``.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from photonix.core.backend import xp
from photonix.core.types import Model, SDict, ports_of

from . import engine as _engine
from .netlist import InstancePort, Netlist

__all__ = [
    "MODE_SEP",
    "modal_port",
    "split_modal_port",
    "is_modal",
    "lift_model",
    "ModalNetlist",
    "ModeMismatchWarning",
    "eme_to_modal",
    "meep_dataset_to_modal",
]

#: Separator between a physical port name and a mode label.
MODE_SEP = ":"


class ModeMismatchWarning(UserWarning):
    """A connection joins ports whose mode sets differ; unmatched modes are terminated."""


def modal_port(port: str, mode: str) -> str:
    """The modal terminal name ``f"{port}:{mode}"``.

    Examples
    --------
    >>> modal_port("o1", "TE0")
    'o1:TE0'
    """
    if not port or MODE_SEP in port:
        raise ValueError(f"physical port names must be non-empty and must not contain {MODE_SEP!r}: {port!r}")
    if not mode or MODE_SEP in mode:
        raise ValueError(f"mode labels must be non-empty and must not contain {MODE_SEP!r}: {mode!r}")
    return f"{port}{MODE_SEP}{mode}"


def split_modal_port(name: str) -> tuple[str, str | None]:
    """Split ``"o1:TE0"`` into ``("o1", "TE0")``; a plain port gives ``(name, None)``.

    Accepts the Meep multiport convention (``"o1:m1"``).

    Examples
    --------
    >>> split_modal_port("o2:TM0"), split_modal_port("o3"), split_modal_port("o1:m1")
    (('o2', 'TM0'), ('o3', None), ('o1', 'm1'))
    """
    port, sep, mode = name.rpartition(MODE_SEP)
    if not sep:
        return name, None
    return port, mode


def is_modal(sd: Mapping[tuple[str, str], Any]) -> bool:
    """True if every port of ``sd`` is a modal terminal (``"port:mode"``).

    Raises ``ValueError`` for a mix of modal and plain port names.
    """
    names = ports_of(cast(SDict, dict(sd))) if sd else []
    flags = {split_modal_port(n)[1] is not None for n in names}
    if len(flags) > 1:
        raise ValueError(f"SDict mixes modal and plain port names: {names}")
    return flags == {True}


# --------------------------------------------------------------------------- #
# Lifting single-mode models
# --------------------------------------------------------------------------- #
def _dense(sd: SDict, ports: Sequence[str]) -> Any:
    """``M[..., out, in]`` from an SDict (absent entries are zero)."""
    vals = [xp.asarray(v) for v in sd.values()]
    shape = np.broadcast_shapes(*(tuple(v.shape) for v in vals)) if vals else ()
    zero = xp.zeros(shape, dtype=complex)
    rows = [xp.stack([xp.broadcast_to(xp.asarray(sd.get((pin, pout), zero), dtype=complex), shape)
                      for pin in ports], axis=-1) for pout in ports]
    return xp.stack(rows, axis=-2)


def lift_model(
    model: Callable[..., SDict],
    modes: Sequence[str],
    *,
    per_mode: Mapping[str, Mapping[str, Any]] | None = None,
    coupling: Callable[[Any], Any] | Any | None = None,
    coupling_port: str | Sequence[str] | None = None,
) -> Model:
    """Turn a single-mode model into a multimode one with modal terminals.

    Parameters
    ----------
    model
        Single-mode model with plain ports ``o1..oN``.
    modes
        Mode labels, e.g. ``("TE0", "TM0")``.
    per_mode
        ``mode -> settings`` applied on top of the call's settings for that
        mode only (``{"TM0": {"neff": 1.8, "ng": 3.9}}``). A ``per_mode``
        keyword at call time is merged over this one.
    coupling
        Optional inter-mode coupling: an ``(n_modes, n_modes)`` array, or a
        callable ``C(wl)`` returning one (with optional leading batch axes).
        It is applied at the ports in ``coupling_port`` as
        ``S' = C_out S C_in^T``, i.e. a reciprocal mode mixer sits in front of
        those ports; it is lossless when ``C`` is unitary.
    coupling_port
        Port name(s) at which ``coupling`` acts; ``None`` means every port.

    Returns
    -------
    Model
        ``f(*, wl=1.55, per_mode=None, **settings)`` with ports ``"p:m"``.
        Without coupling, cross-mode entries are absent (exactly zero).

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> f = lift_model(px.components.straight, ("TE0", "TM0"), per_mode={"TM0": {"neff": 1.7}})
    >>> s = f(wl=1.55, length=10.0)
    >>> sorted(s)[:2]
    [('o1:TE0', 'o2:TE0'), ('o1:TM0', 'o2:TM0')]
    >>> te = px.components.straight(wl=1.55, length=10.0)[("o1", "o2")]
    >>> bool(np.isclose(complex(s[("o1:TE0", "o2:TE0")]), complex(te)))
    True
    """
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError(f"modes must be a non-empty sequence of unique labels, got {modes!r}")
    for m in modes:
        modal_port("o", m)
    static = {k: dict(v) for k, v in (per_mode or {}).items()}
    unknown = sorted(set(static) - set(modes))
    if unknown:
        raise ValueError(f"per_mode names unknown modes {unknown}; modes are {list(modes)}")
    if isinstance(coupling_port, str):
        coupled_ports: tuple[str, ...] | None = (coupling_port,)
    else:
        coupled_ports = None if coupling_port is None else tuple(coupling_port)

    def lifted(*, wl=1.55, per_mode: Mapping[str, Mapping[str, Any]] | None = None, **settings) -> SDict:
        call = {k: dict(v) for k, v in (per_mode or {}).items()}
        if set(call) - set(modes):
            raise ValueError(f"per_mode names unknown modes {sorted(set(call) - set(modes))}")
        out: SDict = {}
        base_ports: list[str] | None = None
        for m in modes:
            sd = model(wl=wl, **{**settings, **static.get(m, {}), **call.get(m, {})})
            if base_ports is None:
                base_ports = ports_of(sd)
                if any(MODE_SEP in p for p in base_ports):
                    raise ValueError("lift_model expects a single-mode model with plain port names")
            for (p, q), v in sd.items():
                out[(f"{p}{MODE_SEP}{m}", f"{q}{MODE_SEP}{m}")] = v
        if coupling is None:
            return out
        ports = [f"{p}{MODE_SEP}{m}" for p in cast(list, base_ports) for m in modes]
        n = len(modes)
        c = xp.asarray(coupling(wl) if callable(coupling) else coupling, dtype=complex)
        if tuple(c.shape[-2:]) != (n, n):
            raise ValueError(f"coupling matrix must be ({n}, {n}), got shape {tuple(c.shape)}")
        targets = set(cast(list, base_ports) if coupled_ports is None else coupled_ports)
        missing = targets - set(cast(list, base_ports))
        if missing:
            raise ValueError(f"coupling_port {sorted(missing)} not among the model ports {base_ports}")
        eye = xp.broadcast_to(xp.eye(n, dtype=complex), c.shape)
        blocks = [c if p in targets else eye for p in cast(list, base_ports)]
        big = _block_diag(blocks)
        mat = _dense(out, ports)
        res = big @ mat @ xp.swapaxes(big, -1, -2)
        return {(pin, pout): res[..., i, j] for j, pin in enumerate(ports) for i, pout in enumerate(ports)}

    lifted.__name__ = f"{getattr(model, '__name__', 'model')}_multimode"
    lifted.__doc__ = f"Multimode lift of {getattr(model, '__name__', 'a model')} over modes {list(modes)}."
    return lifted


def _block_diag(blocks: Sequence[Any]) -> Any:
    n = int(blocks[0].shape[-1])
    k = len(blocks)
    batch = tuple(blocks[0].shape[:-2])
    rows = []
    for i, b in enumerate(blocks):
        row = [b if j == i else xp.zeros((*batch, n, n), dtype=complex) for j in range(k)]
        rows.append(xp.concatenate(row, axis=-1))
    return xp.concatenate(rows, axis=-2)


# --------------------------------------------------------------------------- #
# Modal netlists
# --------------------------------------------------------------------------- #
def _lift_sdict(sd: SDict, modes: Sequence[str]) -> SDict:
    return {(f"{p}{MODE_SEP}{m}", f"{q}{MODE_SEP}{m}"): v for (p, q), v in sd.items() for m in modes}


def _port_modes(sd: SDict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name in ports_of(sd):
        port, mode = split_modal_port(name)
        out.setdefault(port, [])
        if mode is not None and mode not in out[port]:
            out[port].append(mode)
    return out


@dataclass
class ModalNetlist:
    """A physical netlist whose connections carry several modes.

    Parameters
    ----------
    netlist
        Physical topology (ports without mode labels, e.g. ``("ring", "o2")``).
    modes
        ``instance -> mode labels`` used to lift single-mode instance models.
        Instances whose model already returns modal terminals ignore it.
    default_modes
        Modes of instances not listed in ``modes``.
    on_mismatch
        ``"terminate"`` (default): a mode carried by only one side of a
        connection is terminated in a matched load and a
        :class:`ModeMismatchWarning` names the dropped path. ``"raise"``:
        raise ``ValueError`` instead.

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> nl = Netlist()
    >>> _ = nl.add("a", "straight", length=5.0).add("b", "straight", length=5.0)
    >>> _ = nl.connect(("a", "o2"), ("b", "o1"))
    >>> _ = nl.expose("in", ("a", "o1")).expose("out", ("b", "o2"))
    >>> f = ModalNetlist(nl, default_modes=("TE0", "TM0")).compile()
    >>> s = f(wl=1.55)
    >>> sorted({k[0] for k in s})
    ['in:TE0', 'in:TM0', 'out:TE0', 'out:TM0']
    >>> ("in:TE0", "out:TM0") in s and abs(complex(s[("in:TE0", "out:TM0")])) < 1e-15
    True
    """

    netlist: Netlist
    modes: Mapping[str, Sequence[str]] = field(default_factory=dict)
    default_modes: Sequence[str] = ("TE0",)
    on_mismatch: Literal["terminate", "raise"] = "terminate"

    def __post_init__(self) -> None:
        if not isinstance(self.netlist, Netlist):
            raise TypeError("ModalNetlist.netlist must be a Netlist")
        self.netlist.validate()
        unknown = sorted(set(self.modes) - set(self.netlist.instances))
        if unknown:
            raise ValueError(f"modes reference unknown instances {unknown}")
        self.modes = {k: tuple(v) for k, v in self.modes.items()}
        self.default_modes = tuple(self.default_modes)
        for ms in [*self.modes.values(), self.default_modes]:
            if not ms or len(set(ms)) != len(ms):
                raise ValueError(f"mode lists must be non-empty and unique, got {ms!r}")
            for m in ms:
                modal_port("o", m)
        if self.on_mismatch not in ("terminate", "raise"):
            raise ValueError("on_mismatch must be 'terminate' or 'raise'")

    def instance_modes(self, instance: str) -> tuple[str, ...]:
        """Mode labels used to lift ``instance`` if its model is single-mode."""
        return tuple(self.modes.get(instance, self.default_modes))

    def _structure(self, inst_sdicts: Mapping[str, SDict], warned: set[Hashable] | None = None
                   ) -> tuple[dict[InstancePort, InstancePort], dict[str, InstancePort]]:
        pm = {inst: _port_modes(sd) for inst, sd in inst_sdicts.items()}

        def modes_at(terminal: InstancePort) -> list[str]:
            inst, port = terminal
            if port not in pm[inst]:
                raise KeyError(f"instance {inst!r} has no port {port!r}; ports: {sorted(pm[inst])}")
            return pm[inst][port]

        connections: dict[InstancePort, InstancePort] = {}
        for a, b in self.netlist.connections.items():
            ma, mb = modes_at(a), modes_at(b)
            for m in ma:
                if m in mb:
                    connections[(a[0], f"{a[1]}{MODE_SEP}{m}")] = (b[0], f"{b[1]}{MODE_SEP}{m}")
            only_a = [m for m in ma if m not in mb]
            only_b = [m for m in mb if m not in ma]
            if only_a or only_b:
                msg = (f"connection {a[0]}.{a[1]} <-> {b[0]}.{b[1]}: modes {only_a} exist only on "
                       f"{a[0]}.{a[1]} and {only_b} only on {b[0]}.{b[1]}")
                if self.on_mismatch == "raise":
                    raise ValueError(msg)
                key = (a, b, tuple(only_a), tuple(only_b))
                if warned is None or key not in warned:
                    warnings.warn(msg + "; they are terminated in matched loads (power in them is dropped).",
                                  ModeMismatchWarning, stacklevel=4)
                    if warned is not None:
                        warned.add(key)
        ports: dict[str, InstancePort] = {}
        for ext, (inst, port) in self.netlist.ports.items():
            for m in modes_at((inst, port)):
                ports[f"{ext}{MODE_SEP}{m}"] = (inst, f"{port}{MODE_SEP}{m}")
        return connections, ports

    def _evaluate(self, registry: Mapping[str, Callable[..., SDict]], wl: Any, overrides: Mapping[str, Any],
                  dedupe: bool) -> dict[str, SDict]:
        memo: dict[Hashable, SDict] = {}
        out: dict[str, SDict] = {}
        for inst, model_name in self.netlist.instances.items():
            if model_name not in registry:
                raise KeyError(f"Model {model_name!r} for instance {inst!r} not in registry.")
            fn = registry[model_name]
            settings = self.netlist.merged_settings(inst, overrides.get(inst))
            key = (id(fn), tuple(sorted((k, _engine._freeze(v)) for k, v in settings.items())))
            sd = memo.get(key) if dedupe else None
            if sd is None:
                sd = fn(wl=wl, **settings)
                if dedupe:
                    memo[key] = sd
            out[inst] = sd if is_modal(sd) else _lift_sdict(sd, self.instance_modes(inst))
        return out

    def expand(self, models: Mapping[str, Callable[..., SDict]] | None = None, *, wl: Any = 1.55,
               **overrides: Any) -> tuple[Netlist, dict[str, Model]]:
        """The equivalent single-mode-terminal :class:`~photonix.circuit.Netlist` and its models.

        The modal port structure is read from one probe evaluation at ``wl``.
        Single-mode instance models are replaced by their lifts, registered
        under ``"<model>@<mode>+<mode>"``.

        Examples
        --------
        >>> nl = Netlist()
        >>> _ = nl.add("a", "straight")
        >>> _ = nl.expose("o1", ("a", "o1")).expose("o2", ("a", "o2"))
        >>> flat, models = ModalNetlist(nl, default_modes=("TE0", "TM0")).expand()
        >>> sorted(flat.ports), flat.instances
        (['o1:TE0', 'o1:TM0', 'o2:TE0', 'o2:TM0'], {'a': 'straight@TE0+TM0'})
        """
        registry = _registry(models)
        probe = self._evaluate(registry, wl, overrides, dedupe=True)
        connections, ports = self._structure(probe)
        instances: dict[str, str] = {}
        out_models: dict[str, Model] = {}
        for inst, model_name in self.netlist.instances.items():
            fn = registry[model_name]
            raw = fn(wl=wl, **self.netlist.merged_settings(inst, overrides.get(inst)))
            if is_modal(raw):
                instances[inst] = model_name
                out_models[model_name] = cast(Model, fn)
            else:
                ms = self.instance_modes(inst)
                name = f"{model_name}@{'+'.join(ms)}"
                instances[inst] = name
                out_models[name] = lift_model(fn, ms)
        flat = Netlist(instances=instances, connections=connections, ports=ports,
                       settings={k: dict(v) for k, v in self.netlist.settings.items()}, name=self.netlist.name)
        return flat, out_models

    def compile(self, models: Mapping[str, Callable[..., SDict]] | None = None, *,
                backend: _engine.Backend | str = "auto", dedupe: bool = True) -> Model:
        """Compile into ``f(*, wl=1.55, **overrides) -> SDict`` over modal external ports.

        External ports are ``"<ext>:<mode>"`` for every mode the exposed
        instance port carries; overrides are ``{instance: {param: value}}`` as
        for :func:`~photonix.circuit.compile_circuit`.
        """
        registry = _registry(models)
        if backend not in _engine.BACKENDS:
            raise ValueError(f"unknown circuit backend {backend!r}")
        cache = _engine._TopologyCache(maxsize=4)
        warned: set[Hashable] = set()

        def model(*, wl=1.55, **overrides) -> SDict:
            inst_sdicts = self._evaluate(registry, wl, overrides, dedupe)
            connections, ports = self._structure(inst_sdicts, warned)
            topo = cache.get(inst_sdicts, connections, ports)
            if topo.m == 0:
                return {}
            return _engine.solve(topo, _engine.gather_values(topo, inst_sdicts), backend=backend)

        model.__name__ = self.netlist.name or "modal_circuit"
        return model


def _registry(models: Mapping[str, Callable[..., SDict]] | None) -> Mapping[str, Callable[..., SDict]]:
    if models is not None:
        return models
    from photonix.components import MODELS

    return cast(Mapping[str, Callable[..., SDict]], MODELS)


# --------------------------------------------------------------------------- #
# EM solver results with modal terminals
# --------------------------------------------------------------------------- #
def eme_to_modal(
    result: Any,
    *,
    in_port: str = "o1",
    out_port: str = "o2",
    modes_in: Sequence[str],
    modes_out: Sequence[str],
) -> SDict:
    """Name an :class:`~photonix.em.eme.EMEResult`'s modes as modal terminals.

    ``modes_in[i]`` labels the ``i``-th mode of the input section and
    ``modes_out[j]`` the ``j``-th of the output section; transmission,
    reflection and the output-side reflection blocks are all included
    (``Tf[j, i]`` becomes ``s[(f"{in_port}:{modes_in[i]}", f"{out_port}:{modes_out[j]}")]``).

    Examples
    --------
    >>> import numpy as np, photonix.em as em
    >>> x = np.linspace(-3, 3, 241); dx = x[1] - x[0]
    >>> eps = np.where(np.abs(x) < 0.5, 3.4757**2, 1.444**2)
    >>> r = em.eme.eme_smatrix([em.eme.Section(eps, 5.0)], dx, 1.55, num_modes=2)
    >>> s = eme_to_modal(r, modes_in=("TE0", "TE1"), modes_out=("TE0", "TE1"))
    >>> bool(abs(abs(s[("o1:TE0", "o2:TE0")]) - 1) < 1e-9 and abs(s[("o1:TE0", "o2:TE1")]) < 1e-9)
    True
    """
    modes_in, modes_out = tuple(modes_in), tuple(modes_out)
    n_in, n_out = len(modes_in), len(modes_out)
    if n_in > result.Tf.shape[1] or n_out > result.Tf.shape[0]:
        raise ValueError(f"result holds {result.Tf.shape[1]} input and {result.Tf.shape[0]} output modes; "
                         f"asked for {n_in} and {n_out}")
    if in_port == out_port:
        raise ValueError("in_port and out_port must differ")
    a = [modal_port(in_port, m) for m in modes_in]
    b = [modal_port(out_port, m) for m in modes_out]
    out: SDict = {}
    for i in range(n_in):
        for j in range(n_out):
            out[(a[i], b[j])] = complex(result.Tf[j, i])
            out[(b[j], a[i])] = complex(result.Tb[i, j])
        for k in range(n_in):
            out[(a[i], a[k])] = complex(result.Rf[k, i])
    for j in range(n_out):
        for k in range(n_out):
            out[(b[j], b[k])] = complex(result.Rb[k, j])
    return out


def meep_dataset_to_modal(dataset: Any, label_map: Mapping[str, str] | None = None) -> Any:
    """Relabel a Meep multiport dataset's modal terminals (``"o1:m1"`` -> ``"o1:TE0"``).

    Parameters
    ----------
    dataset
        A :class:`~photonix.core.dataset.SParameterDataset` whose ports are
        modal terminals, as returned by
        :func:`photonix.em.meep.multiport.simulate_multiport_sparameters`.
    label_map
        ``band label -> mode label`` (``{"m1": "TE0", "m2": "TM0"}``);
        unmapped labels are kept.

    Returns
    -------
    SParameterDataset
        The same data with renamed ports; it is callable as a circuit model.

    Examples
    --------
    >>> import numpy as np
    >>> from photonix.core.dataset import SParameterDataset
    >>> ds = SParameterDataset(np.array([1.55]), ("o1:m1", "o2:m1"), np.array([[[0, 1], [1, 0]]]))
    >>> meep_dataset_to_modal(ds, {"m1": "TE0"}).ports
    ('o1:TE0', 'o2:TE0')
    """
    from photonix.core.dataset import SParameterDataset

    mapping = dict(label_map or {})
    ports = []
    for name in dataset.ports:
        port, mode = split_modal_port(name)
        if mode is None:
            raise ValueError(f"dataset port {name!r} is not a modal terminal")
        ports.append(modal_port(port, mapping.get(mode, mode)))
    if len(set(ports)) != len(ports):
        raise ValueError(f"label_map merges distinct terminals: {ports}")
    meta = {**dataset.metadata, "modal_label_map": mapping}
    return SParameterDataset(dataset.wavelengths, tuple(ports), dataset.s, meta)
